"""Hopper (sm90) cooperative persistent GEMM for ``tlx.ops.mm``.

Promoted from ``tutorials/hopper-persistent-gemm-ws-cooperative.py``. The
producer loads two private A tiles and one shared B tile for two replicated
WGMMA consumer tasks.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

from ..._catalog import InvalidInput
from ._shapes import SM90_FOCUS
from ._sm90_core import _allocate_buffers, _consume_tile, _k_tile_range, _run_producer, _split_acc, _split_acc2, _store_subtile

PERF_SHAPES = SM90_FOCUS

_MAX_SMEM = 232448 - 1024  # leave room for mbarriers and alignment
_SPLIT_K_CHOICES = (2, 3, 4, 5, 6, 8, 11, 16, 22, 32)
_MIN_K_TILES_PER_SPLIT = 4
_TORCH_TO_TL = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


@functools.lru_cache(maxsize=1)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


@triton.constexpr_function
def _n_piece(n, i):
    """i-th power-of-two piece of a tile N, largest first (0 past the last).

    TLX buffers and WGMMAs need power-of-two N, so a tile of any N is split
    into its binary pieces, e.g. 192 -> 128, 64.
    """
    pieces = []
    while n:
        pieces.append(1 << (n.bit_length() - 1))
        n -= pieces[-1]
    return pieces[i] if i < len(pieces) else 0


def _is_pow2(n):
    return n & (n - 1) == 0


@triton.constexpr_function
def _npot_subtile(n):
    """Two-piece N tiles whose pieces split into equal subtiles (192 = 4x32 + 2x32,
    96 = 4x16 + 2x16) share the power-of-two path's double-buffered epilogue."""
    return _n_piece(n, 1) > 0 and _n_piece(n, 2) == 0 and _n_piece(n, 0) == 2 * _n_piece(n, 1)


def matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BM"]
    block_n = nargs["BN"]
    block_k = nargs["BK"]
    block_m_split = block_m // nargs["NUM_MMA_GROUPS"]
    swap_ab = nargs["SWAP_AB"]

    if nargs.get("A_ROW_MAJOR", True):
        nargs["a_desc"].block_shape = [block_m_split, block_k]
    else:
        nargs["a_desc"].block_shape = [block_k, block_m_split]
    n0, n1, n2 = (_n_piece(block_n, i) for i in range(3))
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [block_k, block_n]
    else:
        nargs["b_desc"].block_shape = [n0, block_k]
    # Tail-piece descriptors; an unused one is never read, so give it a valid box.
    for piece, name in ((n1, "b_tail_desc"), (n2, "b_tail2_desc")):
        nargs[name].block_shape = [piece or n0, block_k]
    if swap_ab:
        # The kernel computes C^T; each consumer stores its transposed tile.
        epilogue_block = [block_n, block_m_split]
    elif _npot_subtile(block_n):
        epilogue_block = [block_m_split, n1 // 2]
    elif n1:
        # A non-power-of-two tile stores each N piece through its own descriptor.
        epilogue_block = [block_m_split, n0]
    else:
        epilogue_block = [block_m_split, block_n // 4 if nargs["EPILOGUE_SUBTILE"] else block_n]
    c = nargs["c_desc"]
    c.block_shape = epilogue_block
    for piece, name in ((n1, "c_tail_desc"), (n2, "c_tail2_desc")):
        nargs[name].block_shape = [block_m_split, piece] if piece else epilogue_block

    split_k = nargs["SPLIT_K"]
    workspace_desc = nargs["workspace_desc"]
    if split_k > 1:
        # Each split owns a tile-padded slab so a partial tile's TMA store never
        # spills into the next split. Partials stay fp32 until the reduction,
        # which sums them in a fixed order: split-K stays bitwise deterministic,
        # unlike fp32 atomics or a TMA reduce-add into one buffer.
        rows = split_k * _workspace_rows_per_split(c.shape[0], block_n if swap_ab else block_m)
        workspace = torch.empty((rows, c.shape[1]), device=c.base.device, dtype=torch.float32)
        workspace_desc.base = workspace
        workspace_desc.shape = list(workspace.shape)
        workspace_desc.strides = list(workspace.stride())
    else:
        workspace_desc.base = c.base
        workspace_desc.shape = list(c.shape)
        workspace_desc.strides = list(c.strides)
    workspace_desc.block_shape = epilogue_block


def _workspace_rows_per_split(M, block_m):
    return triton.cdiv(M, block_m) * block_m


def _config(block_m, block_n, num_stages, group_size_m=8, split_k=1, block_k=64, swap_ab=False):
    return triton.Config(
        {
            "BM": block_m,
            "BN": block_n,
            "BK": block_k,
            "GROUP_SIZE_M": group_size_m,
            "NUM_STAGES": num_stages,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": not swap_ab and _is_pow2(block_n),
            "SPLIT_K": split_k,
            "SWAP_AB": swap_ab,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=matmul_tma_set_block_size_hook,
    )


def _smem_bytes(block_m, block_n, block_k, num_stages, split_k, elem_bytes=2, swap_ab=False):
    ring = (block_m + block_n) * block_k * elem_bytes * num_stages
    # Subtiled epilogue: four (BM/2, BN/4) slots. Swap-AB: one transposed
    # (BN, BM/2) slot per consumer. Non-power-of-two BN: one slot per N piece
    # per consumer.
    if _npot_subtile(block_n) and not swap_ab:
        # Four (BM/2, N_TAIL/2) slots.
        return ring + block_m * _n_piece(block_n, 1) * elem_bytes
    whole_tile = swap_ab or not _is_pow2(block_n)
    epilogue = block_m * block_n // (1 if whole_tile else 2) * (4 if split_k > 1 else elem_bytes)
    return ring + epilogue


def _fits_smem(block_m, block_n, num_stages, split_k, block_k=64, swap_ab=False):
    return _smem_bytes(block_m, block_n, block_k, num_stages, split_k, swap_ab=swap_ab) <= _MAX_SMEM


_TILES = (
    # (BM, BN, ring depths)
    (128, 256, (3, 4)),
    (256, 128, (3, 4)),
    (128, 128, (4, 6)),
    (128, 64, (6, 8)),
)


def _full_configs():
    # GROUP_SIZE_M=1 walks tiles row-major so all N-tiles of a narrow-N, tall-M
    # product share the A panel in L2; square shapes prefer the grouped order.
    # Split-K variants are generated per tile; the shape prune keeps the ones
    # that fill the GPU.
    configs = []
    for block_m, block_n, depths in _TILES:
        for split_k in (1, ) + _SPLIT_K_CHOICES:
            for num_stages in depths:
                # The fp32 split-K epilogue doubles its SMEM; drop to the
                # deepest ring that still fits rather than skipping the tile.
                while num_stages > 2 and not _fits_smem(block_m, block_n, num_stages, split_k):
                    num_stages -= 1
                for group_size_m in ((1, 8) if split_k == 1 else (1, )):
                    configs.append(_config(block_m, block_n, num_stages, group_size_m, split_k))
    # The SMEM clamp can collapse two depths into one; keep the first.
    unique = {}
    for c in configs + _npot_configs():
        unique.setdefault(tuple(sorted(c.kwargs.items())), c)
    return list(unique.values())


# Non-power-of-two N tiles, run as up to three power-of-two WGMMA pieces each
# at least 32 wide: every piece re-reads A from SMEM, so narrow pieces cost more
# than the padding they save. Whole-K only, K-major B only.
_NPOT_BLOCK_NS = (96, 160, 192, 224)


def _npot_configs():
    configs = []
    for block_n in _NPOT_BLOCK_NS:
        num_stages = 6
        while num_stages > 2 and not _fits_smem(128, block_n, num_stages, 1):
            num_stages -= 1
        for group_size_m in (1, 8):
            configs.append(_config(128, block_n, num_stages, group_size_m))
    return configs


CONFIGS = _full_configs


def _split_k_valid(K, split_k, block_k=64):
    k_tiles = triton.cdiv(K, block_k)
    if split_k == 1:
        return True
    if k_tiles // split_k < _MIN_K_TILES_PER_SPLIT:
        return False
    # cdiv over-allocation can leave the last split empty, which deadlocks.
    return triton.cdiv(k_tiles, split_k) * (split_k - 1) < k_tiles


def _mn_tiles(M, N, block_m, block_n):
    return triton.cdiv(M, block_m) * triton.cdiv(N, block_n)


def _keep_config(c, M, N, K, num_sms):
    kw = c.kwargs
    bm, bn, split_k = kw["BM"], kw["BN"], kw["SPLIT_K"]
    if not _fits_smem(bm, bn, kw["NUM_STAGES"], split_k):
        return False
    if bn > 128 and N <= 128:
        return False
    # Smaller tiles only pay off when bigger ones leave SMs idle.
    if bn == 64 and _mn_tiles(M, N, 128, 128) >= num_sms:
        return False
    if bm == 128 and bn == 128 and N > 128 and _mn_tiles(M, N, 128, 256) >= 2 * num_sms:
        return False
    if split_k > 1:
        mn_tiles = _mn_tiles(M, N, bm, bn)
        if mn_tiles * split_k > num_sms or not _split_k_valid(K, split_k):
            return False
    return True


def _per_sm_work(c, M, N, K, num_sms):
    # MMA area times K steps on the busiest SM: the single-wave-or-more cost model
    # that decides whether a non-power-of-two tile beats padding.
    kw = c.kwargs
    units = _mn_tiles(M, N, kw["BM"], kw["BN"]) * kw["SPLIT_K"]
    return triton.cdiv(units, num_sms) * kw["BM"] * kw["BN"] * triton.cdiv(triton.cdiv(K, 64), kw["SPLIT_K"])


def _keep_npot(kept, M, N, K, num_sms, b_row_major):
    """Keep the two non-power-of-two tiles that most cut per-SM work, and only
    when they beat every power-of-two config by at least 3%."""
    pow2 = [c for c in kept if _is_pow2(c.kwargs["BN"])]
    npot = [c for c in kept if not _is_pow2(c.kwargs["BN"])]
    if b_row_major or not npot:
        return pow2
    if not pow2:
        return npot
    best_pow2 = min(_per_sm_work(c, M, N, K, num_sms) for c in pow2)
    # A 128x128 cooperative tile is SMEM-bandwidth bound (each consumer re-reads
    # B), and a two-piece 192 tile does more math per byte, so it wins even at
    # equal per-SM work (706x8192x2560: 44.6 vs 46.3 us). Against wider tiles
    # an NPOT tile must cut per-SM work by 3%.
    wide = any(c.kwargs["BM"] * c.kwargs["BN"] > 128 * 128
               for c in pow2
               if _per_sm_work(c, M, N, K, num_sms) == best_pow2)
    margin = 0.97 if wide else 1.0
    winners = sorted({c.kwargs["BN"]
                      for c in npot
                      if _per_sm_work(c, M, N, K, num_sms) <= margin * best_pow2}, key=lambda bn:
                     (_per_sm_work(_config(128, bn, 2), M, N, K, num_sms), -bn))[:2]
    return pow2 + [c for c in npot if c.kwargs["BN"] in winners]


def _prune_full_configs(configs, nargs, **kwargs):
    M, N, K = nargs["M"], nargs["N"], nargs["K"]
    num_sms = _get_num_sms()
    kept = [c for c in configs if _keep_config(c, M, N, K, num_sms)]
    kept = _keep_npot(kept, M, N, K, num_sms, kwargs.get("B_ROW_MAJOR", True))

    # Per tile shape and ring depth, keep split-K=1 plus the two split counts
    # that occupy the most SMs; the rest only add autotuning time.
    best_splits = {}
    for c in kept:
        kw = c.kwargs
        if kw["SPLIT_K"] > 1:
            best_splits.setdefault((kw["BM"], kw["BN"], kw["NUM_STAGES"]), set()).add(kw["SPLIT_K"])
    best_splits = {key: set(sorted(splits)[-2:]) for key, splits in best_splits.items()}
    kept = [
        c for c in kept
        if c.kwargs["SPLIT_K"] == 1 or c.kwargs["SPLIT_K"] in best_splits[(c.kwargs["BM"], c.kwargs["BN"],
                                                                           c.kwargs["NUM_STAGES"])]
    ]

    # With N <= 512 only 2-4 tiles span N, and grouped order (G=8) is bimodal on
    # tall shapes (e.g. 2Mx512x512 runs at 420 or ~480 TFLOPS), so autotune can
    # lock in a slow config from one lucky sample. Row-major order is stable.
    if N <= 512:
        kept = [c for c in kept if c.kwargs["GROUP_SIZE_M"] == 1] or kept
    return kept or [
        c for c in configs if _split_k_valid(K, c.kwargs["SPLIT_K"]) and _is_pow2(c.kwargs["BN"])
        and _fits_smem(c.kwargs["BM"], c.kwargs["BN"], c.kwargs["NUM_STAGES"], c.kwargs["SPLIT_K"])
    ]


def _smoke_configs():
    return [_config(128, 256, 3)]


SMOKE_CONFIGS = _smoke_configs

# Swap-AB computes C^T = B^T A^T so a skinny M lands on the WGMMA N dimension
# (as small as 16) instead of padding two 64-row consumer tiles. The kernel's M
# and N are the problem's N and M below.
_SWAP_AB_MAX_M = 64
_SWAP_BLOCK_NS = (16, 32, 64)


def _use_swap_ab(M, space):
    return M <= _SWAP_AB_MAX_M and space != "smoke"


def _swap_block_n(m):
    return next(bn for bn in _SWAP_BLOCK_NS if bn >= m)


def _deepest_ring(block_m, block_n, block_k, split_k, max_stages=6):
    num_stages = max_stages
    while num_stages > 2 and not _fits_smem(block_m, block_n, num_stages, split_k, block_k, swap_ab=True):
        num_stages -= 1
    return num_stages


def _swap_configs():
    return [
        _config(block_m, block_n, _deepest_ring(block_m, block_n, block_k, split_k), 1, split_k, block_k, True)
        # BM=256 and BK=64 measured slower on every skinny shape; a longer K
        # extent reads more contiguous bytes per row of the streamed operand.
        for block_m in (128, )
        for block_n in _SWAP_BLOCK_NS
        for block_k in (128, 256)
        for split_k in (1, ) + _SPLIT_K_CHOICES
    ]


def _prune_swap_configs(configs, nargs, **kwargs):
    # Kernel M/N are the problem's N/M.
    M, N, K = nargs["M"], nargs["N"], nargs["K"]
    num_sms = _get_num_sms()
    kept = []
    for c in configs:
        kw = c.kwargs
        bm, bk, split_k = kw["BM"], kw["BK"], kw["SPLIT_K"]
        if kw["BN"] != _swap_block_n(N) or not _split_k_valid(K, split_k, bk):
            continue
        if split_k > 1 and _mn_tiles(M, N, bm, kw["BN"]) * split_k > num_sms:
            continue
        kept.append(c)
    best_splits = {}
    for c in kept:
        if c.kwargs["SPLIT_K"] > 1:
            best_splits.setdefault((c.kwargs["BM"], c.kwargs["BK"]), set()).add(c.kwargs["SPLIT_K"])
    best_splits = {key: set(sorted(splits)[-2:]) for key, splits in best_splits.items()}
    return [
        c for c in kept
        if c.kwargs["SPLIT_K"] == 1 or c.kwargs["SPLIT_K"] in best_splits[(c.kwargs["BM"], c.kwargs["BK"])]
    ]


def _swap_heuristic_config(M, N, K):
    # Arguments are the problem's (M, N, K).
    num_sms = _get_num_sms()
    block_m, block_n, block_k = 128, _swap_block_n(M), 128
    mn_tiles = _mn_tiles(N, M, block_m, block_n)
    split_k = max([s for s in _SPLIT_K_CHOICES if mn_tiles * s <= num_sms and _split_k_valid(K, s, block_k)], default=1)
    return [_config(block_m, block_n, _deepest_ring(block_m, block_n, block_k, split_k), 1, split_k, block_k, True)]


def heuristic_config(M, N, K):
    num_sms = _get_num_sms()
    block_m, block_n = (256, 128) if M > N else (128, 256)
    if _mn_tiles(M, N, block_m, block_n) >= num_sms:
        # Square shapes favor 128x256 tiles. The 4-stage ring (the SMEM max)
        # wins at large K but regresses some tall shapes with K <= 512.
        return [_config(block_m, block_n, 4 if K >= 1024 else 3)]
    # Too few output tiles to fill the GPU: shrink the tile, then split K.
    block_m, block_n = 128, 128 if N > 64 else 64
    mn_tiles = _mn_tiles(M, N, block_m, block_n)
    split_k = max([s for s in _SPLIT_K_CHOICES if mn_tiles * s <= num_sms and _split_k_valid(K, s)], default=1)
    num_stages = 6 if block_n == 128 else 8
    while num_stages > 2 and not _fits_smem(block_m, block_n, num_stages, split_k):
        num_stages -= 1
    assert _fits_smem(block_m, block_n, num_stages, split_k)
    return [_config(block_m, block_n, num_stages, group_size_m=1, split_k=split_k)]


@triton.jit
def _reduce_split_k_kernel(workspace_ptr, c_ptr, M, N, ROWS_PER_SPLIT, SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr, OUTPUT_DTYPE: tl.constexpr):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    offs = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(workspace_ptr + s * ROWS_PER_SPLIT * N + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, acc.to(OUTPUT_DTYPE), mask=mask)


def _reduce_split_k(workspace, c, M, N, split_k):
    block_m, block_n = 16, 128
    _reduce_split_k_kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n))](
        workspace,
        c,
        M,
        N,
        workspace.shape[0] // split_k,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        OUTPUT_DTYPE=_TORCH_TO_TL[c.dtype],
    )


def _reduce_post_hook(nargs, exception=None):
    # Autotuning times the split-K partial pass plus its reduction.
    if exception is None and nargs["SPLIT_K"] > 1:
        c = nargs["c_desc"].base
        _reduce_split_k(nargs["workspace_desc"].base, c, c.shape[0], c.shape[1], nargs["SPLIT_K"])


@triton.jit
def _issue_piece_dots(a, b, b_tail, b_tail2, acc0, acc1, acc2, bars_full_a, bars_full_b, buf, phase,
                      consumer_id: tl.constexpr, NUM_STAGES: tl.constexpr, A_ROW_MAJOR: tl.constexpr,
                      N_TAIL: tl.constexpr, N_TAIL2: tl.constexpr):
    a_index = buf + NUM_STAGES * consumer_id
    tlx.barrier_wait(tlx.local_view(bars_full_a, a_index), phase)
    # Triton TR051: bounded unrolling misses the producer's TMA
    # completion after the shared B ring wraps.
    tlx.barrier_wait(tlx.local_view(bars_full_b, buf), phase)  # noqa: TR051
    data_a = tlx.local_view(a, a_index)
    a_operand = data_a if A_ROW_MAJOR else tlx.local_trans(data_a)
    acc0 = tlx.async_dot(a_operand, tlx.local_trans(tlx.local_view(b, buf)), acc0)
    if N_TAIL > 0:
        acc1 = tlx.async_dot(a_operand, tlx.local_trans(tlx.local_view(b_tail, buf)), acc1)
    if N_TAIL2 > 0:
        acc2 = tlx.async_dot(a_operand, tlx.local_trans(tlx.local_view(b_tail2, buf)), acc2)
    return acc0, acc1, acc2


@triton.jit
def _consume_tile_pieces(K, a, b, b_tail, b_tail2, acc0, acc1, acc2, bars_empty_a, bars_full_a, bars_empty_b,
                         bars_full_b, buf, phase, consumer_id: tl.constexpr, BK: tl.constexpr, NUM_STAGES: tl.constexpr,
                         A_ROW_MAJOR: tl.constexpr, N_TAIL: tl.constexpr, N_TAIL2: tl.constexpr):
    """_consume_tile for an N tile split into power-of-two pieces, one WGMMA per
    piece per K step, all reading the same A stage. B is K-major.

    The first K step is peeled, as in _consume_tile, so last_buf enters the loop
    as a value distinct from buf and is carried by it."""
    NUM_DOTS: tl.constexpr = 1 + (N_TAIL > 0) + (N_TAIL2 > 0)
    acc0, acc1, acc2 = _issue_piece_dots(a, b, b_tail, b_tail2, acc0, acc1, acc2, bars_full_a, bars_full_b, buf, phase,
                                         consumer_id, NUM_STAGES, A_ROW_MAJOR, N_TAIL, N_TAIL2)
    last_buf = buf
    phase = phase ^ (buf == NUM_STAGES - 1)
    buf = (buf + 1) % NUM_STAGES

    for _ in range(1, tl.cdiv(K, BK)):
        acc0, acc1, acc2 = _issue_piece_dots(a, b, b_tail, b_tail2, acc0, acc1, acc2, bars_full_a, bars_full_b, buf,
                                             phase, consumer_id, NUM_STAGES, A_ROW_MAJOR, N_TAIL, N_TAIL2)
        # Leave this step's WGMMAs in flight; the previous step's are done.
        acc0 = tlx.async_dot_wait(NUM_DOTS, acc0)
        if N_TAIL > 0:
            acc1 = tlx.async_dot_wait(NUM_DOTS, acc1)
        if N_TAIL2 > 0:
            acc2 = tlx.async_dot_wait(NUM_DOTS, acc2)
        tlx.barrier_arrive(tlx.local_view(bars_empty_a, last_buf + NUM_STAGES * consumer_id))
        tlx.barrier_arrive(tlx.local_view(bars_empty_b, last_buf))
        last_buf = buf
        phase = phase ^ (buf == NUM_STAGES - 1)
        buf = (buf + 1) % NUM_STAGES

    acc0 = tlx.async_dot_wait(0, acc0)
    if N_TAIL > 0:
        acc1 = tlx.async_dot_wait(0, acc1)
    if N_TAIL2 > 0:
        acc2 = tlx.async_dot_wait(0, acc2)
    tlx.barrier_arrive(tlx.local_view(bars_empty_a, last_buf + NUM_STAGES * consumer_id))
    tlx.barrier_arrive(tlx.local_view(bars_empty_b, last_buf))
    return acc0, acc1, acc2, buf, phase


@triton.jit
# Triton TR001: `_tuned` applies caller-selected autotune spaces lazily.
# triton-lint: assume NUM_STAGES=4, NUM_MMA_GROUPS=2
def matmul_kernel_tma_ws_hopper_cooperative(  # noqa: TR001
    a_desc,
    b_desc,
    c_desc,
    workspace_desc,
    b_tail_desc,
    b_tail2_desc,
    c_tail_desc,
    c_tail2_desc,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
    A_EVICTION: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    block_m_split: tl.constexpr = BM // NUM_MMA_GROUPS
    # A non-power-of-two BN runs as up to three power-of-two pieces (e.g.
    # 192 = 128 + 64), each with its own B ring, WGMMA and output store.
    N0: tl.constexpr = _n_piece(BN, 0)
    N_TAIL: tl.constexpr = _n_piece(BN, 1)
    N_TAIL2: tl.constexpr = _n_piece(BN, 2)
    tl.static_assert(_n_piece(BN, 3) == 0)
    tl.static_assert(N_TAIL == 0 or (SPLIT_K == 1 and not B_ROW_MAJOR and not SWAP_AB))
    # workspace_desc is fp32 for split-K and aliases c_desc otherwise.
    out_desc = workspace_desc if SPLIT_K > 1 else c_desc
    a, b, bars_empty_a, bars_full_a, bars_empty_b, bars_full_b = _allocate_buffers(
        a_desc,
        b_desc,
        BM,
        N0,
        BK,
        NUM_STAGES,
        NUM_MMA_GROUPS,
        A_ROW_MAJOR,
        B_ROW_MAJOR,
    )
    NPOT_SUBTILE: tl.constexpr = _npot_subtile(BN)
    if NPOT_SUBTILE:
        b_tail = tlx.local_alloc((N_TAIL, BK), tlx.dtype_of(b_desc), NUM_STAGES)
        c_smem_sub = tlx.local_alloc((block_m_split, N_TAIL // 2), tlx.dtype_of(c_desc), 2 * NUM_MMA_GROUPS)
    elif N_TAIL > 0:
        b_tail = tlx.local_alloc((N_TAIL, BK), tlx.dtype_of(b_desc), NUM_STAGES)
        c_smem_p0 = tlx.local_alloc((block_m_split, N0), tlx.dtype_of(c_desc), NUM_MMA_GROUPS)
        c_smem_p1 = tlx.local_alloc((block_m_split, N_TAIL), tlx.dtype_of(c_desc), NUM_MMA_GROUPS)
    else:
        b_tail = None
    if N_TAIL2 > 0:
        b_tail2 = tlx.local_alloc((N_TAIL2, BK), tlx.dtype_of(b_desc), NUM_STAGES)
        c_smem_p2 = tlx.local_alloc((block_m_split, N_TAIL2), tlx.dtype_of(c_desc), NUM_MMA_GROUPS)
    else:
        b_tail2 = None
    if EPILOGUE_SUBTILE:
        c_smem = tlx.local_alloc((block_m_split, BN // 4), tlx.dtype_of(out_desc), 2 * NUM_MMA_GROUPS)
    else:
        c_smem = None
    if SWAP_AB:
        c_smem_t = tlx.local_alloc((BN, block_m_split), tlx.dtype_of(out_desc), NUM_MMA_GROUPS)
    else:
        c_smem_t = None

    with tlx.async_tasks():
        with tlx.async_task("default"):
            _run_producer(
                M,
                N,
                K,
                a_desc,
                b_desc,
                a,
                b,
                bars_empty_a,
                bars_full_a,
                bars_empty_b,
                bars_full_b,
                NUM_SMS,
                BM,
                N0,
                BK,
                GROUP_SIZE_M,
                NUM_STAGES,
                NUM_MMA_GROUPS,
                A_ROW_MAJOR,
                B_ROW_MAJOR,
                A_EVICTION,
                SPLIT_K,
                b_tail_desc,
                b_tail,
                N_TAIL,
                b_tail2_desc,
                b_tail2,
                N_TAIL2,
            )

        with tlx.async_task(num_warps=4, replicate=2, registers=232):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_mn_tiles = num_pid_m * num_pid_n
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            consumer_id: tl.constexpr = tlx.async_task_replica_id()
            phase = 0
            buf = 0

            for tile_id in range(start_pid, num_mn_tiles * SPLIT_K, NUM_SMS):
                mn_tile_id = tile_id % num_mn_tiles
                group_id = mn_tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((mn_tile_id % num_pid_in_group) % group_size_m)
                pid_n = (mn_tile_id % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN
                split_id = tile_id // num_mn_tiles
                k_lo, k_hi = _k_tile_range(K, split_id, BK, SPLIT_K)
                offset_cm = offset_am + block_m_split * consumer_id

                if N_TAIL > 0:
                    acc0 = tl.zeros((block_m_split, N0), dtype=tl.float32)
                    acc1 = tl.zeros((block_m_split, N_TAIL), dtype=tl.float32)
                    # An unused third piece still needs a placeholder to thread through.
                    acc2 = tl.zeros((block_m_split, N_TAIL2 if N_TAIL2 > 0 else 8), dtype=tl.float32)
                    acc0, acc1, acc2, buf, phase = _consume_tile_pieces(
                        (k_hi - k_lo) * BK, a, b, b_tail, b_tail2, acc0, acc1, acc2, bars_empty_a, bars_full_a,
                        bars_empty_b, bars_full_b, buf, phase, consumer_id, BK, NUM_STAGES, A_ROW_MAJOR, N_TAIL,
                        N_TAIL2)
                    if NPOT_SUBTILE:
                        sub: tl.constexpr = N_TAIL // 2
                        s_0 = tlx.local_view(c_smem_sub, 2 * consumer_id)
                        s_1 = tlx.local_view(c_smem_sub, 2 * consumer_id + 1)
                        p_0, p_1, p_2, p_3 = _split_acc(acc0, block_m_split, N0)
                        t_0, t_1 = _split_acc2(acc1, block_m_split, N_TAIL)
                        _store_subtile(c_desc, s_0, p_0, offset_cm, offset_bn)
                        _store_subtile(c_desc, s_1, p_1, offset_cm, offset_bn + sub)
                        _store_subtile(c_desc, s_0, p_2, offset_cm, offset_bn + 2 * sub)
                        _store_subtile(c_desc, s_1, p_3, offset_cm, offset_bn + 3 * sub)
                        _store_subtile(c_desc, s_0, t_0, offset_cm, offset_bn + 4 * sub)
                        _store_subtile(c_desc, s_1, t_1, offset_cm, offset_bn + 5 * sub)
                    else:
                        # The previous tile's stores read these slots; drain them first.
                        tlx.async_descriptor_store_wait(0)
                        slot0 = tlx.local_view(c_smem_p0, consumer_id)
                        slot1 = tlx.local_view(c_smem_p1, consumer_id)
                        tlx.local_store(slot0, acc0.to(tlx.dtype_of(c_desc)))
                        tlx.local_store(slot1, acc1.to(tlx.dtype_of(c_desc)))
                        if N_TAIL2 > 0:
                            slot2 = tlx.local_view(c_smem_p2, consumer_id)
                            tlx.local_store(slot2, acc2.to(tlx.dtype_of(c_desc)))
                        tlx.fence("async_shared")
                        tlx.async_descriptor_store(c_desc, slot0, [offset_cm, offset_bn])
                        tlx.async_descriptor_store(c_tail_desc, slot1, [offset_cm, offset_bn + N0])
                        if N_TAIL2 > 0:
                            tlx.async_descriptor_store(c_tail2_desc, slot2, [offset_cm, offset_bn + N0 + N_TAIL])
                else:
                    acc, buf, phase = _consume_tile(
                        (k_hi - k_lo) * BK,
                        a,
                        b,
                        bars_empty_a,
                        bars_full_a,
                        bars_empty_b,
                        bars_full_b,
                        buf,
                        phase,
                        consumer_id,
                        BK,
                        NUM_STAGES,
                        A_ROW_MAJOR,
                        B_ROW_MAJOR,
                    )
                    _store_tile(acc, out_desc, c_smem, c_smem_t, offset_cm, offset_bn, split_id, num_pid_m, num_pid_n,
                                M, consumer_id, block_m_split, BM, BN, SPLIT_K, SWAP_AB, EPILOGUE_SUBTILE)

            if EPILOGUE_SUBTILE or SWAP_AB or N_TAIL > 0:
                tlx.async_descriptor_store_wait(0)


# With only a few K steps the warp-specialized kernel's fixed cost (barrier
# setup, producer/consumer hand-off, persistent scheduling) dominates; a plain
# one-tile-per-CTA kernel, software-pipelined by Triton, is cheaper there. The
# full space also tries it, with split-K, on skinny shapes that leave SMs idle.
_SMALL_K_MAX = 256
# Heuristic-space N cap for M > 64. Wider N gives the persistent WS kernel many
# N tiles per A tile to reuse; on 706x20992x128 WS ran 17.4 us vs 21.1 us here.
_SMALL_K_MAX_N = 8192


def _use_small_k(M, N, K, space):
    # The smoke space pins one WS config (see SMOKE_CONFIGS) so tests that
    # select it cover the WS lowering path; it never takes the small-K kernel.
    if space == "smoke" or K > _SMALL_K_MAX:
        return False
    if M <= _SWAP_AB_MAX_M:
        # One or two K steps: a small-M tile beats swap-AB's warp-specialized
        # kernel at any N; from K = 256 swap-AB's pipelining wins.
        return K <= 128
    if N > _SMALL_K_MAX_N:
        return False
    # Past a wave of tiles at K = 256, and for very wide N (A reuse), the
    # persistent WS kernel's store/compute overlap wins back its fixed cost.
    return K <= 128 or _mn_tiles(M, N, 128, 128) <= _get_num_sms()


def _small_k_set_block_size_hook(nargs):
    bm, bn, bk = nargs["BM"], nargs["BN"], nargs["BK"]
    nargs["a_desc"].block_shape = [bm, bk] if nargs["A_ROW_MAJOR"] else [bk, bm]
    nargs["b_desc"].block_shape = [bk, bn] if nargs["B_ROW_MAJOR"] else [bn, bk]
    c = nargs["c_desc"]
    c.block_shape = [bm, bn]
    workspace_desc = nargs["workspace_desc"]
    if nargs["SPLIT_K"] > 1:
        # Same tile-padded fp32 slabs and fixed-order reduction as the WS path.
        rows = nargs["SPLIT_K"] * _workspace_rows_per_split(nargs["M"], bm)
        workspace = torch.empty((rows, nargs["N"]), device=c.base.device, dtype=torch.float32)
        workspace_desc.base = workspace
        workspace_desc.shape = list(workspace.shape)
        workspace_desc.strides = list(workspace.stride())
    else:
        workspace_desc.base = c.base
        workspace_desc.shape = list(c.shape)
        workspace_desc.strides = list(c.strides)
    workspace_desc.block_shape = [bm, bn]


def _small_k_config(block_m, block_n, block_k, num_warps, num_stages, group_size_m=8, split_k=1):
    return triton.Config(
        {"BM": block_m, "BN": block_n, "BK": block_k, "GROUP_SIZE_M": group_size_m, "SPLIT_K": split_k},
        num_warps=num_warps,
        num_stages=num_stages,
        pre_hook=_small_k_set_block_size_hook,
    )


_SMALL_K_TILES = (
    # (BM, BN, BK, warps, stages)
    (64, 128, 32, 4, 4),
    (64, 128, 64, 4, 3),
    (64, 128, 128, 4, 2),
    (64, 256, 64, 8, 3),
    (64, 64, 64, 4, 3),
    (128, 64, 64, 4, 3),
    (128, 128, 32, 4, 4),
    (128, 128, 64, 4, 3),
    (128, 128, 64, 8, 3),
    (128, 128, 128, 8, 2),
    (128, 256, 64, 8, 3),
)
# Skinny-M tiles for M <= 64, where the WGMMA M of 64 would be mostly padding.
_SKINNY_SMALL_K_TILES = (
    (16, 32, 64, 2, 2),
    (16, 64, 64, 2, 2),
    (16, 64, 64, 4, 2),
    (16, 128, 64, 4, 2),
    (16, 64, 32, 2, 3),
    (32, 64, 64, 4, 2),
    (16, 32, 128, 2, 1),
)


def _small_k_tile(c):
    kw = c.kwargs
    return kw["BM"], kw["BN"], kw["BK"], c.num_warps, c.num_stages


def _small_k_configs():
    return [
        _small_k_config(*tile, split_k=split_k)
        for tile in _SMALL_K_TILES + _SKINNY_SMALL_K_TILES
        for split_k in (1, ) + _SPLIT_K_CHOICES
    ]


def _prune_small_k_configs(configs, nargs, **kwargs):
    """Tiles by M; split-K only past the small-K range, keeping per tile the two
    split counts that best fill one wave."""
    M, N, K = nargs["M"], nargs["N"], nargs["K"]
    tiles = _SKINNY_SMALL_K_TILES if M <= _SWAP_AB_MAX_M else _SMALL_K_TILES
    num_sms = _get_num_sms()
    kept, splits = [], {}
    for c in configs:
        tile, split_k = _small_k_tile(c), c.kwargs["SPLIT_K"]
        if tile not in tiles:
            continue
        if split_k > 1:
            if K <= _SMALL_K_MAX or not _split_k_valid(K, split_k, tile[2]):
                continue
            if _mn_tiles(M, N, tile[0], tile[1]) * split_k > num_sms:
                continue
            splits.setdefault(tile, set()).add(split_k)
        kept.append(c)
    best = {tile: set(sorted(s)[-2:]) for tile, s in splits.items()}
    kept = [c for c in kept if c.kwargs["SPLIT_K"] == 1 or c.kwargs["SPLIT_K"] in best[_small_k_tile(c)]]
    return kept or [c for c in configs if c.kwargs["SPLIT_K"] == 1]


def _small_k_heuristic_config(M, N, K):
    if M <= _SWAP_AB_MAX_M:
        return [_small_k_config(16, 32, 64, 2, 2)]
    return [_small_k_config(128, 128, 64, 8, 3)]


@triton.jit
def _small_k_kernel(a_desc, b_desc, c_desc, workspace_desc, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr, GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr, A_ROW_MAJOR: tl.constexpr,
                    B_ROW_MAJOR: tl.constexpr):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BM
    offs_n = pid_n * BN
    k_lo, k_hi = _k_tile_range(K, split_id, BK, SPLIT_K)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(k_lo, k_hi):
        offs_k = k * BK
        if A_ROW_MAJOR:
            a = a_desc.load([offs_m, offs_k])
        else:
            a = a_desc.load([offs_k, offs_m]).T
        if B_ROW_MAJOR:
            b = b_desc.load([offs_k, offs_n])
        else:
            b = b_desc.load([offs_n, offs_k]).T
        acc = tl.dot(a, b, acc)
    if SPLIT_K > 1:
        # Each split owns a tile-padded slab of num_pid_m * BM rows.
        workspace_desc.store([split_id * num_pid_m * BM + offs_m, offs_n], acc)
    else:
        c_desc.store([offs_m, offs_n], acc.to(tlx.dtype_of(c_desc)))


@functools.lru_cache(maxsize=None)
def _tuned_small_k(space, shape=None):
    if space == "heuristic":
        return triton.autotune(configs=_small_k_heuristic_config(*shape), key=["M", "N", "K"])(_small_k_kernel)
    return triton.autotune(configs=_small_k_configs(), key=["M", "N", "K"],
                           prune_configs_by={"early_config_prune":
                                             _prune_small_k_configs}, post_hook=_reduce_post_hook)(_small_k_kernel)


@triton.jit
def _store_tile(acc, out_desc, c_smem, c_smem_t, offset_cm, offset_bn, split_id, num_pid_m, num_pid_n, M,
                consumer_id: tl.constexpr, block_m_split: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                SPLIT_K: tl.constexpr, SWAP_AB: tl.constexpr, EPILOGUE_SUBTILE: tl.constexpr):
    if SWAP_AB:
        # acc is C^T[offset_cm:+BM/2, offset_bn:+BN]; store it transposed.
        # C rows past the problem's M land in this split's padding.
        slot = tlx.local_view(c_smem_t, consumer_id)
        tlx.async_descriptor_store_wait(0)
        tlx.local_store(slot, tl.trans(acc).to(tlx.dtype_of(out_desc)))
        tlx.fence("async_shared")
        tlx.async_descriptor_store(out_desc, slot, [offset_bn + split_id * num_pid_n * BN, offset_cm])
    # Split slabs are tile-padded, so rows past M are real workspace
    # memory; skip a consumer half that lies entirely past M.
    elif SPLIT_K == 1 or offset_cm < M:
        offset_cm += split_id * num_pid_m * BM
        if EPILOGUE_SUBTILE:
            acc_0, acc_1, acc_2, acc_3 = _split_acc(acc, block_m_split, BN)
            slot_0 = tlx.local_view(c_smem, 2 * consumer_id)
            slot_1 = tlx.local_view(c_smem, 2 * consumer_id + 1)
            _store_subtile(out_desc, slot_0, acc_0, offset_cm, offset_bn)
            _store_subtile(out_desc, slot_1, acc_1, offset_cm, offset_bn + BN // 4)
            _store_subtile(out_desc, slot_0, acc_2, offset_cm, offset_bn + BN // 2)
            _store_subtile(out_desc, slot_1, acc_3, offset_cm, offset_bn + 3 * (BN // 4))
        else:
            out_desc.store(
                [offset_cm, offset_bn],
                acc.to(tlx.dtype_of(out_desc)),
            )


@functools.lru_cache(maxsize=None)
def _tuned(space, shape=None, swap_ab=False):
    prune_configs_by = None
    if space == "heuristic":
        configs = _swap_heuristic_config(*shape) if swap_ab else heuristic_config(*shape)
    elif space == "full":
        configs = _swap_configs() if swap_ab else CONFIGS()
        prune_configs_by = {"early_config_prune": _prune_swap_configs if swap_ab else _prune_full_configs}
    elif space == "smoke":
        configs = SMOKE_CONFIGS()
    else:
        raise InvalidInput(f"sm90 tlx.ops.mm does not provide search space {space!r}; "
                           "expected 'heuristic', 'full', or 'smoke'")
    return triton.autotune(
        configs=configs,
        key=["M", "N", "K"],
        prune_configs_by=prune_configs_by,
        post_hook=_reduce_post_hook,
    )(matmul_kernel_tma_ws_hopper_cooperative)


_SKINNY_MAX_MN = 256

# (M, N, K, A row-major, B row-major, dtype) -> path chosen by the full space.
_PATH_CACHE = {}
# Tests pin one path, e.g. to exercise a pinned config of that kernel.
_FORCE_PATH = None


def _small_k_candidate(M, N, K, space):
    if space != "full":
        return _use_small_k(M, N, K, space)
    # Past the small-K range, one-tile-per-CTA (with split-K) can beat the
    # persistent WS kernel only on skinny shapes that underfill the GPU.
    return K <= _SMALL_K_MAX or (min(M, N) <= _SKINNY_MAX_MN and _mn_tiles(M, N, 128, 128) < _get_num_sms())


def _select_path(space, key, launchers):
    """Pick the kernel for a call.

    The full space autotunes every candidate kernel, times each one's best
    config the same way the autotuner does, and caches the winner per shape and
    layout. The heuristic space takes the first candidate in preference order
    without timing anything.
    """
    if _FORCE_PATH is not None:
        return _FORCE_PATH
    if len(launchers) == 1 or space != "full":
        return "small_k" if "small_k" in launchers else "ws"
    path = _PATH_CACHE.get(key)
    if path is None:
        timings = {}
        for name, launch in launchers.items():
            launch()  # autotunes this kernel on first use
            timings[name] = _cold_gpu_time(launch)
        path = min(timings, key=timings.get)
        _PATH_CACHE[key] = path
    return path


def _cold_gpu_time(fn, calls=10, reps=5):
    """Median GPU time of ``fn`` with a cold L2.

    These kernels run for a few microseconds, where host launch cost would
    dominate event timing and favor the path with the least Python, so time a
    CUDA graph of (L2 flush, fn) calls minus a graph of the flushes alone.
    """
    flush = torch.empty(2 * torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size,
                        dtype=torch.int8, device="cuda")
    graphs = []
    for with_fn in (True, False):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                flush.zero_()
                if with_fn:
                    fn()
        graphs.append(graph)

    def replay_ms(graph):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for graph in graphs:
        graph.replay()
    diffs = sorted(replay_ms(graphs[0]) - replay_ms(graphs[1]) for _ in range(reps))
    return diffs[reps // 2] / calls


def mm(a, b, *, out=None, space="full"):
    """Compute ``a @ b`` with the Hopper cooperative persistent GEMM."""
    M, K = a.shape
    _, N = b.shape

    if min(M, N, K) <= 0:
        raise InvalidInput("sm90 tlx.ops.mm requires positive M, N, and K")
    if a.data_ptr() % 16 != 0 or b.data_ptr() % 16 != 0:
        raise InvalidInput("sm90 tlx.ops.mm requires 16-byte-aligned operands")
    if not (a.is_contiguous() or a.T.is_contiguous()):
        raise InvalidInput("sm90 tlx.ops.mm requires A to be row- or column-major")
    if not (b.is_contiguous() or b.T.is_contiguous()):
        raise InvalidInput("sm90 tlx.ops.mm requires B to be row- or column-major")

    if out is not None:
        if (out.shape != (M, N) or out.device != a.device or out.dtype != a.dtype or not out.is_contiguous()):
            raise InvalidInput(f"out must be a contiguous {a.dtype} tensor with shape "
                               f"({M}, {N}) on A's device")
        c = out
    else:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if c.data_ptr() % 16 != 0:
        raise InvalidInput("sm90 tlx.ops.mm requires a 16-byte-aligned output")

    if space not in ("heuristic", "full", "smoke"):
        raise InvalidInput(f"sm90 tlx.ops.mm does not provide search space {space!r}; "
                           "expected 'heuristic', 'full', or 'smoke'")
    # A contiguous view of each operand's storage; for these, is_contiguous()
    # is ambiguous only on extent-1 dims, so descriptors use canonical strides.
    a_rm = a.is_contiguous()
    b_rm = b.is_contiguous()
    a_store = a if a_rm else a.T
    b_store = b if b_rm else b.T
    args = (a_store, b_store, a_rm, b_rm, c, M, N, K, space)
    launchers = {"ws": functools.partial(_launch_ws, *args)}
    if _small_k_candidate(M, N, K, space):
        launchers["small_k"] = functools.partial(_launch_small_k, *args)
    launchers[_select_path(space, (M, N, K, a_rm, b_rm, a.dtype), launchers)]()
    return c


def _descriptor(t):
    return TensorDescriptor(t, t.shape, [t.shape[1], 1], [1, 1])


def _launch_small_k(a_store, b_store, a_rm, b_rm, c, M, N, K, space):
    kernel = _tuned_small_k(space, (M, N, K) if space == "heuristic" else None)
    # The pre_hook swaps in an fp32 workspace once SPLIT_K is known.
    workspace_desc = _descriptor(c)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]), meta["SPLIT_K"])  # noqa: E731
    kernel[grid](_descriptor(a_store), _descriptor(b_store), _descriptor(c), workspace_desc, M, N, K, A_ROW_MAJOR=a_rm,
                 B_ROW_MAJOR=b_rm)
    # post_hook only fires while benchmarking, so reduce here for the real launch.
    split_k = kernel.best_config.kwargs["SPLIT_K"]
    if split_k > 1:
        _reduce_split_k(workspace_desc.base, c, M, N, split_k)


def _launch_ws(a_store, b_store, a_rm, b_rm, c, M, N, K, space):
    swap_ab = _use_swap_ab(M, space)
    if swap_ab:
        # Run the same kernel on (B^T, A^T) and write C^T tiles back to C. B^T
        # is row-major exactly when B is column-major, and likewise for A^T.
        kernel_m, kernel_n = N, M
        a_row_major, b_row_major = not b_rm, not a_rm
        a_src, b_src = b_store, a_store
    else:
        kernel_m, kernel_n = M, N
        a_row_major, b_row_major = a_rm, b_rm
        a_src, b_src = a_store, b_store
    c_desc = _descriptor(c)
    # The pre_hook swaps in an fp32 workspace once SPLIT_K is known.
    workspace_desc = _descriptor(c)
    # Narrower boxes for the N tail pieces of a non-power-of-two tile.
    b_tail_desc = _descriptor(b_src)
    b_tail2_desc = _descriptor(b_src)
    c_tail_desc = _descriptor(c)
    c_tail2_desc = _descriptor(c)
    num_sms = _get_num_sms()

    def grid(meta):
        num_tiles = triton.cdiv(kernel_m, meta["BM"]) * triton.cdiv(kernel_n, meta["BN"]) * meta["SPLIT_K"]
        return (min(num_sms, num_tiles), )

    kernel = _tuned(space, (M, N, K) if space == "heuristic" else None, swap_ab)
    kernel[grid](
        _descriptor(a_src),
        _descriptor(b_src),
        c_desc,
        workspace_desc,
        b_tail_desc,
        b_tail2_desc,
        c_tail_desc,
        c_tail2_desc,
        kernel_m,
        kernel_n,
        K,
        NUM_SMS=num_sms,
        A_ROW_MAJOR=a_row_major,
        B_ROW_MAJOR=b_row_major,
        # With N <= 1024 only a few N-tiles read each A panel, back to back;
        # keeping it in L2 serves the repeat reads. Larger N regresses slightly.
        # Under swap-AB the kernel's A is B, streamed once.
        A_EVICTION="evict_last" if N <= 1024 and not swap_ab else "",
    )
    # post_hook only fires while benchmarking, so reduce here for the real launch.
    split_k = kernel.best_config.kwargs["SPLIT_K"]
    if split_k > 1:
        _reduce_split_k(workspace_desc.base, c, M, N, split_k)
