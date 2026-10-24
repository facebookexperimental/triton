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
from ..._invariance import BitwiseInvarianceSpec, gemm_fixed_split_k
from ._shapes import SM90_FOCUS
from ._sm90_core import _allocate_buffers, _consume_tile, _k_tile_range, _run_producer, _split_acc, _store_subtile

PERF_SHAPES = SM90_FOCUS

_MAX_SMEM = 232448 - 1024  # leave room for mbarriers and alignment
_SPLIT_K_CHOICES = (2, 3, 4, 5, 6, 8, 11, 16, 22, 32)
_MIN_K_TILES_PER_SPLIT = 4
_TORCH_TO_TL = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
_FIXED_SPEC = BitwiseInvarianceSpec.dims("K")


@functools.lru_cache(maxsize=1)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


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
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [block_k, block_n]
    else:
        nargs["b_desc"].block_shape = [block_n, block_k]
    if swap_ab:
        # The kernel computes C^T; each consumer stores its transposed tile.
        epilogue_block = [block_n, block_m_split]
    else:
        epilogue_block = [block_m_split, block_n // 4 if nargs["EPILOGUE_SUBTILE"] else block_n]
    c = nargs["c_desc"]
    c.block_shape = epilogue_block

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
            "EPILOGUE_SUBTILE": not swap_ab,
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
    # (BN, BM/2) slot per consumer.
    epilogue = block_m * block_n // (1 if swap_ab else 2) * (4 if split_k > 1 else elem_bytes)
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
    for c in configs:
        unique.setdefault(tuple(sorted(c.kwargs.items())), c)
    return list(unique.values())


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


def _prune_full_configs(configs, nargs, **kwargs):
    M, N, K = nargs["M"], nargs["N"], nargs["K"]
    num_sms = _get_num_sms()
    kept = [c for c in configs if _keep_config(c, M, N, K, num_sms)]

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
        c for c in configs if _split_k_valid(K, c.kwargs["SPLIT_K"])
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


def _swap_heuristic_config(M, N, K, *, invariance=None):
    # Arguments are the problem's (M, N, K).
    num_sms = _get_num_sms()
    block_m, block_n, block_k = 128, _swap_block_n(M), 128
    mn_tiles = _mn_tiles(N, M, block_m, block_n)
    fixed = gemm_fixed_split_k(invariance)
    if fixed is not None:
        split_k = fixed
    else:
        split_k = max([s for s in _SPLIT_K_CHOICES if mn_tiles * s <= num_sms and _split_k_valid(K, s, block_k)],
                      default=1)
    return [_config(block_m, block_n, _deepest_ring(block_m, block_n, block_k, split_k), 1, split_k, block_k, True)]


def heuristic_config(M, N, K, *, invariance=None):
    num_sms = _get_num_sms()
    block_m, block_n = (256, 128) if M > N else (128, 256)
    if _mn_tiles(M, N, block_m, block_n) >= num_sms:
        # Square shapes favor 128x256 tiles. The 4-stage ring (the SMEM max)
        # wins at large K but regresses some tall shapes with K <= 512.
        return [_config(block_m, block_n, 4 if K >= 1024 else 3)]
    # Too few output tiles to fill the GPU: shrink the tile, then split K.
    block_m, block_n = 128, 128 if N > 64 else 64
    mn_tiles = _mn_tiles(M, N, block_m, block_n)
    fixed = gemm_fixed_split_k(invariance)
    if fixed is not None:
        split_k = fixed
    else:
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
# Triton TR001: `_tuned` applies caller-selected autotune spaces lazily.
# triton-lint: assume NUM_STAGES=4, NUM_MMA_GROUPS=2
def matmul_kernel_tma_ws_hopper_cooperative(  # noqa: TR001
    a_desc,
    b_desc,
    c_desc,
    workspace_desc,
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
    # workspace_desc is fp32 for split-K and aliases c_desc otherwise.
    out_desc = workspace_desc if SPLIT_K > 1 else c_desc
    a, b, bars_empty_a, bars_full_a, bars_empty_b, bars_full_b = _allocate_buffers(
        a_desc,
        b_desc,
        BM,
        BN,
        BK,
        NUM_STAGES,
        NUM_MMA_GROUPS,
        A_ROW_MAJOR,
        B_ROW_MAJOR,
    )
    if EPILOGUE_SUBTILE:
        c_smem = tlx.local_alloc((block_m_split, BN // 4), tlx.dtype_of(out_desc), 2 * NUM_MMA_GROUPS)
    if SWAP_AB:
        c_smem_t = tlx.local_alloc((BN, block_m_split), tlx.dtype_of(out_desc), NUM_MMA_GROUPS)

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
                BN,
                BK,
                GROUP_SIZE_M,
                NUM_STAGES,
                NUM_MMA_GROUPS,
                A_ROW_MAJOR,
                B_ROW_MAJOR,
                A_EVICTION,
                SPLIT_K,
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

                offset_cm = offset_am + block_m_split * consumer_id
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

            if EPILOGUE_SUBTILE or SWAP_AB:
                tlx.async_descriptor_store_wait(0)


def _tuned(space, shape=None, swap_ab=False, *, invariance=None):
    """Autotuned kernel per search space, keyed on the effective fixed schedule.

    Every non-empty invariance spec currently maps to the same `SPLIT_K`, so
    `dims("M")`, `dims("K")` and `all()` share one autotuner instead of
    retuning identical schedules.
    """
    return _tuned_for_fixed(space, shape, swap_ab, gemm_fixed_split_k(invariance))


@functools.lru_cache(maxsize=None)
def _tuned_for_fixed(space, shape, swap_ab, fixed):
    prune_configs_by = None
    if space == "heuristic":
        fixed_spec = None if fixed is None else _FIXED_SPEC
        configs = _swap_heuristic_config(*shape, invariance=fixed_spec) if swap_ab else heuristic_config(
            *shape, invariance=fixed_spec)
    elif space == "full":
        configs = _swap_configs() if swap_ab else CONFIGS()
        prune_configs_by = {"early_config_prune": _prune_swap_configs if swap_ab else _prune_full_configs}
    elif space == "smoke":
        configs = SMOKE_CONFIGS()
    else:
        raise InvalidInput(f"sm90 tlx.ops.mm does not provide search space {space!r}; "
                           "expected 'heuristic', 'full', or 'smoke'")
    if fixed is not None:
        configs = [c for c in configs if c.kwargs.get("SPLIT_K", 1) == fixed]
    return triton.autotune(
        configs=configs,
        key=["M", "N", "K"],
        prune_configs_by=prune_configs_by,
        post_hook=_reduce_post_hook,
    )(matmul_kernel_tma_ws_hopper_cooperative)


_tuned.cache_clear = _tuned_for_fixed.cache_clear


def mm(a, b, *, out=None, space="full", invariance=None):
    """Compute ``a @ b`` with the Hopper cooperative persistent GEMM.

    `invariance` is the resolved spec; a fixed schedule prunes split-K
    candidates from every space so the K reduction stays uniform.
    """
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

    swap_ab = _use_swap_ab(M, space)
    # A contiguous view of each operand's storage; for these, is_contiguous()
    # is ambiguous only on extent-1 dims, so descriptors use canonical strides.
    a_rm = a.is_contiguous()
    b_rm = b.is_contiguous()
    a_store = a if a_rm else a.T
    b_store = b if b_rm else b.T
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

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a_src, a_src.shape, [a_src.shape[1], 1], dummy_block)
    b_desc = TensorDescriptor(b_src, b_src.shape, [b_src.shape[1], 1], dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
    # The pre_hook swaps in an fp32 workspace once SPLIT_K is known.
    workspace_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    num_sms = _get_num_sms()

    def grid(meta):
        num_tiles = triton.cdiv(kernel_m, meta["BM"]) * triton.cdiv(kernel_n, meta["BN"]) * meta["SPLIT_K"]
        return (min(num_sms, num_tiles), )

    kernel = _tuned(space, (M, N, K) if space == "heuristic" else None, swap_ab, invariance=invariance)
    kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        workspace_desc,
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
    return c
