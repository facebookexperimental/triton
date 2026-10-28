"""Blackwell (sm100) MXFP8 GEMM -- the ``tlx.ops.mm_mxfp8`` implementation.

Promoted from ``tutorials/blackwell_gemm_ws_mxfp8.py``, which is now frozen.

The kernel computes ``C[M, N] = A[M, K] @ B[N, K].T`` from E4M3 data with one
E8M0 scale per 32 K values, and stores BF16. A TMA producer, a scaled-MMA
consumer accumulating in TMEM, and a BF16 epilogue run as warp-specialized
tasks over a static persistent tile schedule. Each config selects among
optional paths: 2-CTA clusters (``COLLAB_2CTA`` issues the MMA from the leader
only), split-K with FP32 partials, overlapped BN=256 accumulators
(``OVERLAP_ACC``), B resident in shared memory with whole-row tiles
(``B_RESIDENT`` + ``FUSE_N_TILES``), and async TMA stores (``ASYNC_STORE``).

As in ``mm/sm100.py``, autotune is applied lazily (``_tuned``) so the search
space can vary per caller. ``space="full"`` (the default) autotunes over the
pruned space plus the overlapped-accumulator and B-resident families;
``space="heuristic"`` launches one shape-picked config.
"""

from __future__ import annotations

import functools
import math

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

VEC_SIZE = 32
OUTPUT_DTYPE = tl.bfloat16

# Scaled MMA supports the mature 1-CTA path and the BF16-style paired-CTA
# control flow. In 2-CTA mode each CTA owns a distinct M tile, each CTA loads
# half of B data, and each CTA loads the full logical B scale tile.
SUPPORTED_NUM_CTAS = (1, 2)
DEFAULT_NUM_CTAS = 1
SUPPORTED_NUM_MMA_GROUPS = 1

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 8,
    "NUM_SMEM_BUFFERS": 3,
    "NUM_TMEM_BUFFERS": 2,
    "NUM_MMA_GROUPS": SUPPORTED_NUM_MMA_GROUPS,
    "EPILOGUE_SUBTILE": 4,
    "NUM_CTAS": DEFAULT_NUM_CTAS,
    "SPLIT_K": 1,
    "PEELED_FIRST_K": False,
}


@functools.lru_cache(maxsize=None)
def _num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _select_group_size_m(M, N, block_m):
    num_m_tiles = triton.cdiv(M, block_m)
    ratio = M / max(N, 1)
    if ratio > 10:
        return 1
    if ratio < 0.1:
        return min(64, num_m_tiles)
    return min(8, num_m_tiles)


def _heuristic_split_k(mn_tiles, k_tiles, num_sms):
    if mn_tiles >= num_sms:
        return 1
    return next((s for s in (8, 4, 2) if k_tiles // s >= 4), 1)


def _wastes_256_columns(N):
    return triton.cdiv(N, 256) * 256 * 4 > N * 5


def heuristic_config(M, N, K, num_sms=148):
    """One shape-picked config, measured against the full space on GB300.

    Problems with enough tiles use overlapped accumulators when BN=256 fits N
    (collaborative 2-CTA when the M tiles pair up), else BN=128 with a second
    TMEM buffer. Tiny problems keep the tutorial's original config.
    """
    m_tiles = triton.cdiv(M, 128)
    k_tiles = triton.cdiv(K, 128)
    if not _wastes_256_columns(N):
        mn_tiles = m_tiles * triton.cdiv(N, 256)
        split_k = _heuristic_split_k(mn_tiles, k_tiles, num_sms)
        if mn_tiles * split_k >= num_sms // 4:
            config = {
                **DEFAULT_CONFIG, "BLOCK_SIZE_N": 256, "NUM_TMEM_BUFFERS": 1, "OVERLAP_ACC": True, "SPLIT_K": split_k
            }
            if m_tiles % 2 == 0:
                return {**config, "NUM_CTAS": 2, "COLLAB_2CTA": True, "NUM_SMEM_BUFFERS": 6 if split_k == 1 else 5}
            return {**config, "NUM_SMEM_BUFFERS": 4 if split_k == 1 else 3}
    mn_tiles = m_tiles * triton.cdiv(N, 128)
    split_k = _heuristic_split_k(mn_tiles, k_tiles, num_sms)
    config = {**DEFAULT_CONFIG, "GROUP_SIZE_M": _select_group_size_m(M, N, 128), "SPLIT_K": split_k}
    if mn_tiles * split_k >= num_sms // 4:
        return {**config, "GROUP_SIZE_M": 4, "NUM_SMEM_BUFFERS": 4, "EPILOGUE_SUBTILE": 2, "PEELED_FIRST_K": True}
    return {**config, "NUM_SMEM_BUFFERS": 3 if split_k == 1 else 4}


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": g,
                "NUM_SMEM_BUFFERS": s,
                "NUM_TMEM_BUFFERS": t,
                "NUM_MMA_GROUPS": mma_groups,
                "EPILOGUE_SUBTILE": subtile,
                "NUM_CTAS": num_ctas,
                "SPLIT_K": split_k,
                "INTERLEAVE_EPILOGUE": interleave,
                "USE_WARP_BARRIER": use_warp_barrier,
                "PEELED_FIRST_K": num_ctas == 1,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(2, 1, 1) if num_ctas == 2 else None,
        )
        # ADS supplies the scaled-MMA tile sizes. All remaining tuning axes
        # mirror the BF16 tutorial's autotune space.
        for BN in [128, 256]
        for BK in [128, 256]
        for s in [2, 3, 4, 5, 6, 7, 8]
        for t in [1, 2, 3]
        for mma_groups in [1, 2]
        for subtile in [1, 2, 4, 8]
        for num_ctas in SUPPORTED_NUM_CTAS
        for split_k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 19, 24]
        for interleave in [0, 1]
        for g in [1, 2, 4, 8, 64]
        for use_warp_barrier in [False, True]
    ]


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    EPILOGUE_SUBTILE = nargs["EPILOGUE_SUBTILE"]
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    BLOCK_N_PER_CTA = BLOCK_N // NUM_CTAS
    SPLIT_K = nargs.get("SPLIT_K", 1)

    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    # B stays in its public [N, K] layout. In 2-CTA mode each CTA loads its
    # owned [BLOCK_N // NUM_CTAS, BLOCK_K] data slice and locally transposes it
    # before async_dot_scaled.
    nargs["b_desc"].block_shape = [BLOCK_N_PER_CTA, BLOCK_K]
    nargs["a_scale_desc"].block_shape = [1, BLOCK_M // 128, BLOCK_K // 128, 2, 256]
    # B scale is deliberately not split: each CTA loads the full logical B scale
    # tile required by 2-CTA scaled MMA.
    nargs["b_scale_desc"].block_shape = [1, BLOCK_N // 128, BLOCK_K // 128, 2, 256]
    nargs["out_desc"].block_shape = [BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]

    if SPLIT_K > 1:
        M = nargs["M"]
        N = nargs["N"]
        out = nargs["out_desc"].base
        # Each split owns the padded M range: a 2-CTA virtual tile past M must
        # not land in the next split's rows.
        rows_per_split = triton.cdiv(triton.cdiv(M, BLOCK_M), NUM_CTAS) * NUM_CTAS * BLOCK_M
        workspace = torch.empty((SPLIT_K * rows_per_split, N), device=out.device, dtype=torch.float32)
        nargs["workspace_desc"].base = workspace
        nargs["workspace_desc"].shape = list(workspace.shape)
    else:
        nargs["workspace_desc"].base = nargs["out_desc"].base
        nargs["workspace_desc"].shape = list(nargs["out_desc"].base.shape)
    nargs["workspace_desc"].block_shape = [BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]


def _config_error(meta, M, N, K):
    """Why ``meta`` cannot run on ``(M, N, K)``, or None if it can."""
    block_m = int(meta["BLOCK_SIZE_M"])
    block_n = int(meta["BLOCK_SIZE_N"])
    block_k = int(meta["BLOCK_SIZE_K"])
    num_ctas = int(meta.get("NUM_CTAS", 1))
    split_k = int(meta.get("SPLIT_K", 1))

    if block_m != 128:
        return "scaled MMA requires BLOCK_SIZE_M=128"
    if num_ctas not in SUPPORTED_NUM_CTAS:
        return "NUM_CTAS must be 1 or 2"
    if block_n % num_ctas != 0 or block_n // num_ctas > 256:
        return "scaled MMA supports at most 256 B columns per CTA"
    if block_k % 128 != 0:
        return "BLOCK_SIZE_K must cover complete scale tiles"
    if meta["EPILOGUE_SUBTILE"] not in (1, 2, 4):
        return "unsupported epilogue subtile"
    if meta.get("NUM_MMA_GROUPS", 1) != SUPPORTED_NUM_MMA_GROUPS:
        return "scaled MMA requires one 128-row MMA group"
    if int(meta.get("GROUP_SIZE_M", 1)) % num_ctas != 0:
        return "GROUP_SIZE_M must be a multiple of NUM_CTAS"
    if num_ctas == 2 and int(meta.get("NUM_TMEM_BUFFERS", 1)) != 1 and not meta.get("COLLAB_2CTA", False):
        return "2-CTA MXFP8 requires one TMEM buffer"
    if split_k < 1 or split_k > triton.cdiv(K, block_k):
        return "SPLIT_K must be in [1, K tiles]"
    if meta.get("B_RESIDENT", False):
        if (meta.get("OVERLAP_ACC", False) or meta.get("PEELED_FIRST_K", False) or split_k != 1
                or (num_ctas == 2 and not meta.get("COLLAB_2CTA", False))):
            return "B_RESIDENT requires SPLIT_K=1, no OVERLAP_ACC/PEELED_FIRST_K, and 1-CTA or COLLAB_2CTA"
        if (meta.get("B_RES_N_TILES") != triton.cdiv(N, block_n)
                or meta.get("B_RES_K_TILES") != triton.cdiv(K, block_k)):
            return "B_RES_N_TILES/B_RES_K_TILES must cover N and K"
    if meta.get("FUSE_N_TILES", 1) > 1:
        fuse = meta["FUSE_N_TILES"]
        if not meta.get("B_RESIDENT", False) or meta.get("B_RES_N_TILES", 1) % fuse:
            return "FUSE_N_TILES requires B_RESIDENT with B_RES_N_TILES divisible by FUSE_N_TILES"
        if fuse * block_n * int(meta.get("NUM_TMEM_BUFFERS", 1)) > 512:
            return "FUSE_N_TILES accumulators exceed 512 TMEM columns"
    if meta.get("N_INNER", False):
        if num_ctas != 1 or split_k != 1 or int(meta.get("GROUP_SIZE_M", 1)) != 1:
            return "N_INNER requires NUM_CTAS=1, SPLIT_K=1, GROUP_SIZE_M=1"
    if meta.get("COLLAB_2CTA", False):
        if num_ctas != 2 or meta.get("PEELED_FIRST_K", False):
            return "COLLAB_2CTA requires NUM_CTAS=2 without PEELED_FIRST_K"
    if meta.get("OVERLAP_ACC", False):
        if (num_ctas != 1 and not meta.get("COLLAB_2CTA", False)
            ) or block_n != 256 or meta["EPILOGUE_SUBTILE"] != 4 or meta.get("NUM_TMEM_BUFFERS") != 1:
            return ("OVERLAP_ACC requires BLOCK_SIZE_N=256, EPILOGUE_SUBTILE=4, NUM_TMEM_BUFFERS=1, "
                    "and NUM_CTAS=1 or COLLAB_2CTA")
        if meta.get("PEELED_FIRST_K", False):
            return "OVERLAP_ACC does not peel the first K block"
    return None


def _make_config(meta):
    meta = {**DEFAULT_CONFIG, **meta}
    return triton.Config(
        meta,
        num_warps=4,
        num_stages=1,
        pre_hook=matmul_tma_set_block_size_hook,
        ctas_per_cga=(2, 1, 1) if meta["NUM_CTAS"] == 2 else None,
    )


def preprocess_configs(configs, named_args, **kwargs):
    """The tutorial's pruning plus the overlapped-accumulator and B-resident families.

    Falls back to the heuristic if nothing survives.
    """
    M, N, K = named_args["M"], named_args["N"], named_args["K"]
    pruned = (_preprocess_configs(configs, named_args, **kwargs) + _overlap_configs(M, N, K, kwargs["NUM_SMS"]) +
              _b_resident_configs(M, N, K))
    if pruned:
        return pruned
    return [_make_config(heuristic_config(M, N, K, kwargs["NUM_SMS"]))]


_SMEM_LIMIT = 232448


def _b_resident_configs(M, N, K):
    """Collaborative 2-CTA configs that keep all of B in shared memory.

    Each CTA holds its half of every B tile plus all B scales once, streams only
    A through the ring, and computes a whole 128 x N row block (FUSE_N_TILES
    accumulators) per tile, so A is read once and B never reloads. Only offered
    when that leaves room for a ring of at least four A stages.
    """
    block_n, block_k = 128, 128
    if N % block_n or K % block_k:
        return []
    n_tiles, k_tiles = N // block_n, K // block_k
    if n_tiles * block_n > 512:
        return []  # fused accumulators must fit in TMEM
    b_bytes = n_tiles * k_tiles * (block_n // 2 * block_k + 2 * 256)
    a_stage_bytes = 128 * block_k + 2 * 256
    epilogue_bytes = 2 * 128 * block_n * 2  # async-store staging ring
    configs = []
    for buffers in (4, 5):
        if b_bytes + buffers * a_stage_bytes + epilogue_bytes + 4096 > _SMEM_LIMIT:
            continue
        for group in (2, 64):
            configs.append(
                _make_config({
                    "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k, "GROUP_SIZE_M": group, "NUM_SMEM_BUFFERS":
                    buffers, "NUM_TMEM_BUFFERS": 1, "EPILOGUE_SUBTILE": 1, "NUM_CTAS": 2, "COLLAB_2CTA": True,
                    "B_RESIDENT": True, "B_RES_N_TILES": n_tiles, "B_RES_K_TILES": k_tiles, "FUSE_N_TILES": n_tiles,
                    "ASYNC_STORE": True
                }))
    return configs


def _overlap_configs(M, N, K, num_sms):
    """1-CTA BN=256 configs with two accumulators sharing one epilogue subtile.

    Pruned on their own: the tutorial's shape-specific filters predate this
    family and would discard it.
    """
    if _wastes_256_columns(N):
        return []
    mn_tiles = triton.cdiv(M, 128) * triton.cdiv(N, 256)
    split_k = _heuristic_split_k(mn_tiles, triton.cdiv(K, 128), num_sms)
    # FP32 split-K staging doubles the epilogue's shared memory.
    pipelines = [(128, 4), (128, 3), (256, 2)] if split_k == 1 else [(128, 3)]
    base = {"BLOCK_SIZE_N": 256, "NUM_TMEM_BUFFERS": 1, "EPILOGUE_SUBTILE": 4, "SPLIT_K": split_k, "OVERLAP_ACC": True}
    configs = [
        _make_config({**base, "BLOCK_SIZE_K": block_k, "GROUP_SIZE_M": group, "NUM_SMEM_BUFFERS": buffers})
        for block_k, buffers in pipelines if block_k <= K for group in (4, 8, 64)
    ]
    # Collaborative 2-CTA: each CTA holds half of B, so deeper pipelines fit.
    configs += [
        _make_config({
            **base, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": group, "NUM_SMEM_BUFFERS": 6 if split_k == 1 else 5,
            "NUM_CTAS": 2, "COLLAB_2CTA": True
        }) for group in (4, 8, 64)
    ]
    return configs


def _preprocess_configs(configs, named_args, **kwargs):
    NUM_SMS = kwargs["NUM_SMS"]
    MAX_SHARED_MEMORY = 232 * 1024
    MAX_TENSOR_MEMORY = 256 * 1024
    MBARRIER_SIZE = 8

    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]

    pruned_configs = []
    for conf in configs:
        BLOCK_M = conf.kwargs["BLOCK_SIZE_M"]
        BLOCK_N = conf.kwargs["BLOCK_SIZE_N"]
        BLOCK_K = conf.kwargs["BLOCK_SIZE_K"]
        NUM_SMEM_BUFFERS = conf.kwargs["NUM_SMEM_BUFFERS"]
        NUM_TMEM_BUFFERS = conf.kwargs["NUM_TMEM_BUFFERS"]
        NUM_MMA_GROUPS = conf.kwargs.get("NUM_MMA_GROUPS", 1)
        NUM_CTAS = conf.kwargs.get("NUM_CTAS", 1)
        SPLIT_K = conf.kwargs.get("SPLIT_K", 1)
        EPILOGUE_SUBTILE = conf.kwargs["EPILOGUE_SUBTILE"]
        INTERLEAVE_EPILOGUE = conf.kwargs.get("INTERLEAVE_EPILOGUE", 0)
        USE_WARP_BARRIER = conf.kwargs.get("USE_WARP_BARRIER", False)
        PEELED_FIRST_K = conf.kwargs.get("PEELED_FIRST_K", False)

        if BLOCK_M != 128 or BLOCK_K % 128 != 0:
            continue
        if NUM_CTAS not in SUPPORTED_NUM_CTAS:
            continue
        # The space mirrors BF16, while the current scaled-MMA execution path
        # supports one MMA group and the non-interleaved mbarrier protocol.
        if NUM_MMA_GROUPS != SUPPORTED_NUM_MMA_GROUPS:
            continue
        if INTERLEAVE_EPILOGUE or USE_WARP_BARRIER:
            continue
        if PEELED_FIRST_K != (NUM_CTAS == 1):
            continue
        if BLOCK_N % NUM_CTAS != 0 or BLOCK_N // NUM_CTAS > 256:
            continue
        if EPILOGUE_SUBTILE not in (1, 2, 4):
            continue
        if BLOCK_N % EPILOGUE_SUBTILE != 0:
            continue
        if conf.ctas_per_cga != ((2, 1, 1) if NUM_CTAS == 2 else None):
            continue
        if conf.kwargs["GROUP_SIZE_M"] % NUM_CTAS != 0:
            continue
        if NUM_CTAS == 2 and NUM_TMEM_BUFFERS != 1:
            continue

        num_pid_m = math.ceil(M / BLOCK_M)
        if NUM_CTAS == 2:
            num_pid_m = ((num_pid_m + NUM_CTAS - 1) // NUM_CTAS) * NUM_CTAS
        num_mn_tiles = num_pid_m * math.ceil(N / BLOCK_N)
        k_tiles = math.ceil(K / BLOCK_K)
        logical_mn_tiles = math.ceil(M / 128) * math.ceil(N / 128)
        if logical_mn_tiles <= 8 and K <= 512:
            if not (BLOCK_N == 128 and BLOCK_K == 128 and NUM_SMEM_BUFFERS in (3, 4) and NUM_TMEM_BUFFERS in (1, 2)
                    and EPILOGUE_SUBTILE in (1, 4) and NUM_CTAS == 1 and SPLIT_K == 1):
                continue
        elif logical_mn_tiles <= 64 and N <= 512 and K <= 512:
            if not (BLOCK_N == 128 and BLOCK_K == 128 and conf.kwargs["GROUP_SIZE_M"] == 4 and NUM_SMEM_BUFFERS == 4
                    and NUM_TMEM_BUFFERS == 1 and EPILOGUE_SUBTILE == 1 and NUM_CTAS == 1 and SPLIT_K == 1):
                continue
        elif logical_mn_tiles <= 16 and K <= 2048:
            if not (BLOCK_N == 128 and BLOCK_K == 128 and conf.kwargs["GROUP_SIZE_M"] == 4 and NUM_SMEM_BUFFERS == 4
                    and NUM_TMEM_BUFFERS == 1 and EPILOGUE_SUBTILE == 1 and NUM_CTAS == 1 and SPLIT_K == 1):
                continue
        if SPLIT_K > 1:
            if num_mn_tiles >= NUM_SMS:
                continue
            if k_tiles < SPLIT_K:
                continue
            k_tiles_per_split = math.ceil(k_tiles / SPLIT_K)
            if k_tiles_per_split * (SPLIT_K - 1) >= k_tiles:
                continue
            if k_tiles // SPLIT_K < 4:
                continue

        rep_m = BLOCK_M // 128
        rep_n = BLOCK_N // 128
        rep_k = BLOCK_K // 128
        smem_a = BLOCK_M * BLOCK_K * NUM_SMEM_BUFFERS
        smem_b = (BLOCK_N // NUM_CTAS) * BLOCK_K * NUM_SMEM_BUFFERS
        smem_a_scale = rep_m * rep_k * 2 * 256 * NUM_SMEM_BUFFERS
        smem_b_scale = rep_n * rep_k * 2 * 256 * NUM_SMEM_BUFFERS
        smem_barriers = (2 * NUM_SMEM_BUFFERS + 2 * NUM_TMEM_BUFFERS) * MBARRIER_SIZE
        if NUM_CTAS == 2:
            smem_barriers += NUM_SMEM_BUFFERS * MBARRIER_SIZE
        total_smem = smem_a + smem_b + smem_a_scale + smem_b_scale + smem_barriers
        if total_smem > MAX_SHARED_MEMORY:
            continue

        total_tmem = BLOCK_M * BLOCK_N * 4 * NUM_TMEM_BUFFERS
        if total_tmem > MAX_TENSOR_MEMORY:
            continue

        pruned_configs.append(conf)

    if not pruned_configs:
        return pruned_configs

    def _total_tiles(c):
        return (math.ceil(M / c.kwargs["BLOCK_SIZE_M"]) * math.ceil(N / c.kwargs["BLOCK_SIZE_N"]) *
                c.kwargs.get("SPLIT_K", 1))

    def _num_waves(c):
        return math.ceil(_total_tiles(c) / NUM_SMS)

    def _tile_key(c):
        return (
            c.kwargs["BLOCK_SIZE_M"],
            c.kwargs["BLOCK_SIZE_N"],
            c.kwargs["BLOCK_SIZE_K"],
        )

    tile_groups = {}
    for conf in pruned_configs:
        tile_groups.setdefault(_tile_key(conf), []).append(conf)

    result = []
    for group_configs in tile_groups.values():
        min_waves = min(_num_waves(conf) for conf in group_configs)
        best = [conf for conf in group_configs if _num_waves(conf) == min_waves]
        max_split_k = max(conf.kwargs.get("SPLIT_K", 1) for conf in best)
        result.extend(conf for conf in best if conf.kwargs.get("SPLIT_K", 1) == max_split_k)
    pruned_configs = result

    # Keep the traversal families that win in both the BF16 tutorial and ADS
    # MXFP8. Aspect ratio is useful for pruning, but it must not force a single
    # CTA topology or discard G64 reuse on otherwise balanced shapes.
    imbalance_threshold = 10
    if M > N * imbalance_threshold:
        if K <= 512:
            # On GB300 a second TMEM buffer with a 2-way epilogue split also wins.
            target_tmem_buffers = (1, 2)
            pruned_configs = [
                conf for conf in pruned_configs
                if conf.kwargs["BLOCK_SIZE_K"] == 128 and conf.kwargs["GROUP_SIZE_M"] == 4
                and conf.kwargs["NUM_SMEM_BUFFERS"] == 4 and conf.kwargs["NUM_TMEM_BUFFERS"] in (
                    target_tmem_buffers if conf.kwargs["NUM_CTAS"] == 1 else (1, ))
                and conf.kwargs["EPILOGUE_SUBTILE"] in ((1, 2) if conf.kwargs["NUM_CTAS"] == 1 else (1, )) and (
                    (N > 256 and conf.kwargs["BLOCK_SIZE_N"] == 128 and conf.kwargs["NUM_CTAS"] == 1) or
                    (N <= 256 and conf.kwargs["BLOCK_SIZE_N"] == 256 and conf.kwargs["NUM_CTAS"] == 2))
            ]
        pruned_configs = [conf for conf in pruned_configs if conf.kwargs["GROUP_SIZE_M"] in (4, 64)]
        if 512 < K <= 2048:
            pruned_configs = [
                conf for conf in pruned_configs
                if conf.kwargs["BLOCK_SIZE_N"] == 128 and conf.kwargs["BLOCK_SIZE_K"] == 128 and conf.
                kwargs["GROUP_SIZE_M"] == 4 and conf.kwargs["NUM_SMEM_BUFFERS"] == 5 and conf.kwargs["NUM_TMEM_BUFFERS"]
                == 3 and conf.kwargs["EPILOGUE_SUBTILE"] == 4 and conf.kwargs["NUM_CTAS"] == 1
            ]
    elif N > M * imbalance_threshold:
        pruned_configs = [conf for conf in pruned_configs if conf.kwargs["GROUP_SIZE_M"] >= 32]
        min_logical_tiles = math.ceil(M / 128) * math.ceil(N / 256)
        if N > M * 16 and 512 < K < 2048:
            pruned_configs = [
                conf for conf in pruned_configs if conf.kwargs["BLOCK_SIZE_N"] == 128
                and conf.kwargs["BLOCK_SIZE_K"] == 128 and conf.kwargs["GROUP_SIZE_M"] == 64
                and conf.kwargs["NUM_SMEM_BUFFERS"] == 6 and conf.kwargs["NUM_TMEM_BUFFERS"] == 3
                and conf.kwargs["EPILOGUE_SUBTILE"] == 4 and conf.kwargs["NUM_CTAS"] == 1
            ]
        elif 8192 < K < 16384:
            pruned_configs = [
                conf for conf in pruned_configs if conf.kwargs["BLOCK_SIZE_N"] == 256
                and conf.kwargs["BLOCK_SIZE_K"] == 128 and conf.kwargs["GROUP_SIZE_M"] == 64
                and conf.kwargs["NUM_SMEM_BUFFERS"] == 6 and conf.kwargs["NUM_TMEM_BUFFERS"] == 1
                and conf.kwargs["EPILOGUE_SUBTILE"] == 4 and conf.kwargs["NUM_CTAS"] == 2
            ]
        elif N > M * 32 and 2048 <= K <= 8192:
            pruned_configs = [
                conf for conf in pruned_configs if conf.kwargs["BLOCK_SIZE_N"] == 256
                and conf.kwargs["BLOCK_SIZE_K"] == 128 and conf.kwargs["GROUP_SIZE_M"] == 64
                and conf.kwargs["NUM_SMEM_BUFFERS"] == 6 and conf.kwargs["NUM_TMEM_BUFFERS"] == 1
                and conf.kwargs["EPILOGUE_SUBTILE"] == 4 and conf.kwargs["NUM_CTAS"] == 2
            ]
        elif K >= 16384 and min_logical_tiles <= 4 * NUM_SMS:
            pruned_configs = [
                conf for conf in pruned_configs
                if conf.kwargs["BLOCK_SIZE_K"] == 128 and conf.kwargs["GROUP_SIZE_M"] == 64
                and conf.kwargs["NUM_SMEM_BUFFERS"] == 6 and conf.kwargs["EPILOGUE_SUBTILE"] == 4 and (
                    (conf.kwargs["BLOCK_SIZE_N"] == 128 and conf.kwargs["NUM_TMEM_BUFFERS"] == 2 and conf.
                     kwargs["NUM_CTAS"] == 1) or (conf.kwargs["BLOCK_SIZE_N"] == 256 and conf.kwargs["NUM_TMEM_BUFFERS"]
                                                  == 1 and conf.kwargs["NUM_CTAS"] == 2))
            ]
    else:
        if M >= 2048 and N >= 2048 and 2048 <= K <= 8192:
            pruned_configs = [
                conf for conf in pruned_configs
                if conf.kwargs["BLOCK_SIZE_N"] == 128 and conf.kwargs["BLOCK_SIZE_K"] == 256 and conf.
                kwargs["GROUP_SIZE_M"] == 4 and conf.kwargs["NUM_SMEM_BUFFERS"] == 4 and conf.kwargs["NUM_TMEM_BUFFERS"]
                == 1 and conf.kwargs["EPILOGUE_SUBTILE"] == 2 and conf.kwargs["NUM_CTAS"] == 2
            ]
        else:
            pruned_configs = [conf for conf in pruned_configs if conf.kwargs["GROUP_SIZE_M"] in (4, 8, 64)]

    # Match BF16's Pareto filter across pipeline resource dimensions.
    def _pipeline_key(conf):
        return (
            conf.kwargs["BLOCK_SIZE_M"],
            conf.kwargs["BLOCK_SIZE_N"],
            conf.kwargs["BLOCK_SIZE_K"],
            conf.kwargs["EPILOGUE_SUBTILE"],
            conf.kwargs["NUM_CTAS"],
            conf.kwargs.get("SPLIT_K", 1),
            conf.kwargs.get("INTERLEAVE_EPILOGUE", 0),
        )

    def _pipeline_value(conf):
        return (
            conf.kwargs["NUM_SMEM_BUFFERS"],
            conf.kwargs["NUM_TMEM_BUFFERS"],
            conf.kwargs["NUM_MMA_GROUPS"],
        )

    def _dominates(lhs, rhs):
        lhs_value = _pipeline_value(lhs)
        rhs_value = _pipeline_value(rhs)
        return all(x >= y for x, y in zip(lhs_value, rhs_value)) and any(x > y for x, y in zip(lhs_value, rhs_value))

    pipeline_groups = {}
    for conf in pruned_configs:
        pipeline_groups.setdefault(_pipeline_key(conf), []).append(conf)

    # More buffering can improve overlap, but it also consumes SMEM/TMEM and
    # can reduce occupancy. Preserve the ADS-proven lean and medium pipeline
    # points alongside the BF16 Pareto frontier instead of treating more
    # buffers as universally better.
    pipeline_anchors = {(4, 1), (4, 2), (5, 3), (6, 1), (6, 3)}
    result = []
    for group_configs in pipeline_groups.values():
        result.extend(
            conf for conf in group_configs
            if (conf.kwargs["NUM_SMEM_BUFFERS"], conf.kwargs["NUM_TMEM_BUFFERS"]) in pipeline_anchors or not any(
                _dominates(other, conf) for other in group_configs if other is not conf))
    return result


@triton.jit
def _first_tile(start_tile_id, N, BLOCK_SIZE_N: tl.constexpr, N_INNER: tl.constexpr):
    if N_INNER:
        return start_tile_id * tl.cdiv(N, BLOCK_SIZE_N)
    return start_tile_id


@triton.jit
def _next_tile(tile_id, num_programs, N, BLOCK_SIZE_N: tl.constexpr, N_INNER: tl.constexpr):
    # N_INNER: a CTA finishes every N tile of its M row before striding to the
    # next row, so the A tile is re-read from the local L2 instead of DRAM.
    if N_INNER:
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        tile_id += 1
        if tile_id % num_pid_n == 0:
            tile_id += (num_programs - 1) * num_pid_n
        return tile_id
    return tile_id + num_programs


@triton.jit
def _b_res_base(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr, B_RES_K_TILES: tl.constexpr):
    _, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    return pid_n * B_RES_K_TILES


@triton.jit
def _load_resident_b(b_desc, b_scale_desc, b_tiles, b_scale_tiles, b_ready, cluster_cta_rank,
                     BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, NUM_CTAS: tl.constexpr,
                     B_RES_N_TILES: tl.constexpr, B_RES_K_TILES: tl.constexpr):
    """Load this CTA's share of every B tile and every B scale tile once."""
    REP_N: tl.constexpr = BLOCK_SIZE_N // 128
    REP_K: tl.constexpr = BLOCK_SIZE_K // 128
    BLOCK_N_PER_CTA: tl.constexpr = BLOCK_SIZE_N // NUM_CTAS
    B_BYTES: tl.constexpr = B_RES_N_TILES * B_RES_K_TILES * BLOCK_N_PER_CTA * BLOCK_SIZE_K
    B_SCALE_BYTES: tl.constexpr = B_RES_N_TILES * B_RES_K_TILES * REP_N * REP_K * 2 * 256
    if NUM_CTAS == 2:
        # cta_group::2 completions land on the leader's barrier: both CTAs' data
        # bytes plus one multicast scale copy per CTA.
        tlx.barrier_expect_bytes(b_ready[0], 2 * (B_BYTES + B_SCALE_BYTES), pred=cluster_cta_rank == 0)
    else:
        tlx.barrier_expect_bytes(b_ready[0], B_BYTES + B_SCALE_BYTES)
    for n in tl.static_range(B_RES_N_TILES):
        for k in tl.static_range(B_RES_K_TILES):
            idx = n * B_RES_K_TILES + k
            if NUM_CTAS == 2:
                tlx.async_descriptor_load(b_desc, b_tiles[idx],
                                          [n * BLOCK_SIZE_N + cluster_cta_rank * BLOCK_N_PER_CTA, k * BLOCK_SIZE_K],
                                          b_ready[0], two_ctas=True)
                tlx.async_descriptor_load(b_scale_desc, b_scale_tiles[idx], [0, n * REP_N, k * REP_K, 0, 0], b_ready[0],
                                          pred=cluster_cta_rank == 0, multicast_targets=[0, 1], two_ctas=True)
            else:
                tlx.async_descriptor_load(b_desc, b_tiles[idx], [n * BLOCK_SIZE_N, k * BLOCK_SIZE_K], b_ready[0])
                tlx.async_descriptor_load(b_scale_desc, b_scale_tiles[idx], [0, n * REP_N, k * REP_K, 0, 0], b_ready[0])


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _compute_grid_info(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    SPLIT_K,
    NUM_CTAS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_tile_id = tl.program_id(axis=0)
    num_programs = NUM_SMS
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # Pad M tiles so adjacent CTA ranks in a 2-CTA cluster always map to the
    # same N tile. TMA descriptor OOB semantics zero-fill the virtual input tile
    # and discard its output store when the real M tile count is odd.
    num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_mn_tiles = num_pid_m * num_pid_n
    num_tiles = num_mn_tiles * SPLIT_K
    k_tiles_total = tl.cdiv(K, BLOCK_SIZE_K)
    return (
        start_tile_id,
        num_programs,
        num_pid_m,
        num_pid_in_group,
        num_mn_tiles,
        num_tiles,
        k_tiles_total,
    )


@triton.jit
def _compute_k_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K: tl.constexpr):
    if SPLIT_K == 1:
        k_tile_start = 0
        k_tile_end = k_tiles_total
    else:
        split_id = tile_id // num_mn_tiles
        k_tile_start = split_id * k_tiles_total // SPLIT_K
        k_tile_end = (split_id + 1) * k_tiles_total // SPLIT_K
    return k_tile_start, k_tile_end


@triton.jit
def _process_tile_epilogue_inner(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    num_mn_tiles,
    GROUP_SIZE_M,
    M,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    EPILOGUE_SUBTILE,
    SPLIT_K,
    out_desc,
    workspace_desc,
    accumulators,
    tmem_full,
    tmem_empty,
    tmem_buf,
    tmem_phase,
    NUM_CTAS: tl.constexpr,
    OVERLAP_ACC: tl.constexpr = False,
    COLLAB_2CTA: tl.constexpr = False,
    store_bufs=None,
    store_count=0,
    ASYNC_STORE: tl.constexpr = False,
    FUSE_N_TILES: tl.constexpr = 1,
):
    mn_tile_id = tile_id if SPLIT_K == 1 else tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    SUB_N: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    ACC_SHIFT: tl.constexpr = BLOCK_SIZE_N - SUB_N

    if SPLIT_K > 1:
        split_id = tile_id // num_mn_tiles
        store_desc = workspace_desc
        row_base = split_id * num_pid_m * BLOCK_SIZE_M
    else:
        store_desc = out_desc
        row_base = 0

    tlx.barrier_wait(tmem_full[tmem_buf], tmem_phase)
    for fused_n in tl.static_range(FUSE_N_TILES):
        for i in tl.static_range(EPILOGUE_SUBTILE):
            if OVERLAP_ACC:
                # Even tiles drain slot 0 in reverse and odd tiles drain slot 1
                # forward, so the first subtile is always the shared columns.
                if tmem_phase == 0:
                    result = tlx.local_load(tlx.subslice(accumulators[0], (EPILOGUE_SUBTILE - 1 - i) * SUB_N, SUB_N))
                    out_n = pid_n * BLOCK_SIZE_N + (EPILOGUE_SUBTILE - 1 - i) * SUB_N
                else:
                    result = tlx.local_load(tlx.subslice(accumulators[0], ACC_SHIFT + i * SUB_N, SUB_N))
                    out_n = pid_n * BLOCK_SIZE_N + i * SUB_N
                if i == 0:
                    # The shared columns are in registers; the next tile may reuse them.
                    if COLLAB_2CTA:
                        tlx.barrier_arrive(tmem_empty[0], arrive_count=1, remote_cta_rank=0)
                    else:
                        tlx.barrier_arrive(tmem_empty[0], arrive_count=1)
            else:
                acc_slice = tlx.local_slice(
                    accumulators[tmem_buf * FUSE_N_TILES + fused_n],
                    [0, i * SUB_N],
                    [BLOCK_SIZE_M, SUB_N],
                )
                result = tlx.local_load(acc_slice)
                if COLLAB_2CTA:
                    # The leader's MMA reuses the accumulator once both CTAs drained it.
                    tlx.barrier_arrive(tmem_empty[tmem_buf], arrive_count=1, remote_cta_rank=0)
                else:
                    tlx.barrier_arrive(tmem_empty[tmem_buf], arrive_count=1)
                out_n = (pid_n * FUSE_N_TILES + fused_n) * BLOCK_SIZE_N + i * SUB_N

            if ASYNC_STORE:
                # Two-slot SMEM ring: only the store issued two subtiles ago must
                # have drained before its slot is overwritten.
                stage = tlx.local_view(store_bufs, store_count % 2)
                tlx.async_descriptor_store_wait(1)
                tlx.local_store(stage, result.to(tlx.dtype_of(store_desc)))
                tlx.fence_async_shared()
                tlx.async_descriptor_store(store_desc, stage, [row_base + pid_m * BLOCK_SIZE_M, out_n])
                store_count += 1
            else:
                store_desc.store(
                    [row_base + pid_m * BLOCK_SIZE_M, out_n],
                    result.to(tlx.dtype_of(store_desc)),
                )
    return store_count


@triton.jit
def _process_tile_mma_overlap(
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    smem_count,
    tmem_phase,
    a_tiles,
    b_tiles,
    a_scale_tiles,
    b_scale_tiles,
    accumulators,
    a_scale_tmem,
    b_scale_tmem,
    smem_full,
    smem_empty,
    BLOCK_SIZE_N: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    TWO_CTAS: tl.constexpr = False,
):
    ACC_SHIFT: tl.constexpr = BLOCK_SIZE_N - BLOCK_SIZE_N // EPILOGUE_SUBTILE
    smem_buf, smem_phase = get_bufidx_phase(smem_count, NUM_SMEM_BUFFERS)
    for k_idx in range(0, k_tile_end - k_tile_start):
        tlx.barrier_wait(smem_full[smem_buf], smem_phase)
        # tcgen05.cp and tcgen05.mma issue in order from this thread, so the
        # single TMEM scale slot is safe to overwrite every k-block.
        tlx.tmem_copy(a_scale_tiles[smem_buf], a_scale_tmem[0])
        tlx.tmem_copy(b_scale_tiles[smem_buf], b_scale_tmem[0])
        # subslice offsets must be compile-time constants.
        if tmem_phase == 0:
            acc = tlx.subslice(accumulators[0], 0, BLOCK_SIZE_N)
        else:
            acc = tlx.subslice(accumulators[0], ACC_SHIFT, BLOCK_SIZE_N)
        tlx.async_dot_scaled(
            a_tiles[smem_buf],
            tlx.local_trans(b_tiles[smem_buf]),
            acc,
            a_scale_tmem[0],
            "e4m3",
            b_scale_tmem[0],
            "e4m3",
            use_acc=k_idx > 0,
            out_dtype=tl.float32,
            mBarriers=[smem_empty[smem_buf]],
            two_ctas=TWO_CTAS,
        )
        smem_count += 1
        smem_buf += 1
        if smem_buf == NUM_SMEM_BUFFERS:
            smem_buf = 0
            smem_phase ^= 1
    return smem_count


@triton.jit
def _process_tile_mma_inner(
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    smem_count,
    tmem_buf,
    a_tiles,
    b_tiles,
    a_scale_tiles,
    b_scale_tiles,
    accumulators,
    smem_full,
    smem_empty,
    cta_bars,
    NUM_CTAS: tl.constexpr,
    cluster_cta_rank,
    DO_MMA: tl.constexpr,
    USE_ACC_INITIAL: tl.constexpr = False,
    COLLAB_2CTA: tl.constexpr = False,
    B_RESIDENT: tl.constexpr = False,
    b_res_base=0,
    FUSE_N_TILES: tl.constexpr = 1,
    B_RES_K_TILES: tl.constexpr = 1,
):
    local_k_tiles = k_tile_end - k_tile_start
    smem_buf, smem_phase = get_bufidx_phase(smem_count, NUM_SMEM_BUFFERS)

    pred_cta0 = cluster_cta_rank == 0

    for k_idx in range(0, local_k_tiles):
        tlx.barrier_wait(smem_full[smem_buf], smem_phase)
        if NUM_CTAS == 2 and not COLLAB_2CTA:
            tlx.barrier_arrive(cta_bars[smem_buf], arrive_count=1, remote_cta_rank=0)
            tlx.barrier_wait(cta_bars[smem_buf], phase=smem_phase, pred=pred_cta0)
        if DO_MMA:
            # FUSE_N_TILES > 1 reuses one A k-chunk for every N tile of the row
            # against resident B; tcgen05 MMAs complete in order, so only the
            # last one releases the A stage.
            for n in tl.static_range(FUSE_N_TILES):
                if B_RESIDENT:
                    b_idx = b_res_base + n * B_RES_K_TILES + k_tile_start + k_idx
                else:
                    b_idx = smem_buf
                if n == FUSE_N_TILES - 1:
                    tlx.async_dot_scaled(
                        a_tiles[smem_buf],
                        tlx.local_trans(b_tiles[b_idx]),
                        accumulators[tmem_buf * FUSE_N_TILES + n],
                        a_scale_tiles[smem_buf],
                        "e4m3",
                        b_scale_tiles[b_idx],
                        "e4m3",
                        use_acc=USE_ACC_INITIAL or k_idx != 0,
                        out_dtype=tl.float32,
                        mBarriers=[smem_empty[smem_buf]],
                        two_ctas=NUM_CTAS == 2,
                    )
                else:
                    tlx.async_dot_scaled(
                        a_tiles[smem_buf],
                        tlx.local_trans(b_tiles[b_idx]),
                        accumulators[tmem_buf * FUSE_N_TILES + n],
                        a_scale_tiles[smem_buf],
                        "e4m3",
                        b_scale_tiles[b_idx],
                        "e4m3",
                        use_acc=USE_ACC_INITIAL or k_idx != 0,
                        out_dtype=tl.float32,
                        two_ctas=NUM_CTAS == 2,
                    )
        smem_count += 1
        smem_buf += 1
        if smem_buf == NUM_SMEM_BUFFERS:
            smem_buf = 0
            smem_phase ^= 1

    return smem_count


@triton.jit
def _process_tile_producer_inner(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    num_mn_tiles,
    GROUP_SIZE_M,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    a_tiles,
    b_tiles,
    a_scale_tiles,
    b_scale_tiles,
    smem_full,
    smem_empty,
    smem_count,
    SPLIT_K: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    cluster_cta_rank,
    COLLAB_2CTA: tl.constexpr = False,
    B_RESIDENT: tl.constexpr = False,
):
    mn_tile_id = tile_id if SPLIT_K == 1 else tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)

    REP_M: tl.constexpr = BLOCK_SIZE_M // 128
    REP_N: tl.constexpr = BLOCK_SIZE_N // 128
    # Local shadow of the module-level VEC_SIZE: the frontend rejects reads of
    # non-constexpr globals, and making that global tl.constexpr would disable
    # this kernel's fast dispatch path.
    VEC_SIZE: tl.constexpr = 32
    REP_K: tl.constexpr = BLOCK_SIZE_K // VEC_SIZE // 4
    BLOCK_N_PER_CTA: tl.constexpr = BLOCK_SIZE_N // NUM_CTAS
    A_BYTES: tl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_K
    B_BYTES: tl.constexpr = BLOCK_N_PER_CTA * BLOCK_SIZE_K
    A_SCALE_BYTES: tl.constexpr = REP_M * REP_K * 2 * 256
    # Full logical B scale is loaded by each CTA in 2-CTA mode.
    B_SCALE_BYTES: tl.constexpr = REP_N * REP_K * 2 * 256
    if B_RESIDENT:
        TOTAL_BYTES_PER_CTA: tl.constexpr = A_BYTES + A_SCALE_BYTES
    else:
        TOTAL_BYTES_PER_CTA: tl.constexpr = A_BYTES + B_BYTES + A_SCALE_BYTES + B_SCALE_BYTES
    smem_buf, smem_phase = get_bufidx_phase(smem_count, NUM_SMEM_BUFFERS)
    for k in range(k_tile_start, k_tile_end):
        offs_k = k * BLOCK_SIZE_K
        offs_scale_k = k * REP_K

        tlx.barrier_wait(smem_empty[smem_buf], smem_phase ^ 1)
        if COLLAB_2CTA:
            # cta_group::2 loads complete on the leader's barrier, which
            # therefore expects both CTAs' bytes; B scales are multicast.
            tlx.barrier_expect_bytes(smem_full[smem_buf], 2 * TOTAL_BYTES_PER_CTA, pred=cluster_cta_rank == 0)
            tlx.async_descriptor_load(a_desc, a_tiles[smem_buf], [pid_m * BLOCK_SIZE_M, offs_k], smem_full[smem_buf],
                                      eviction_policy="evict_last", two_ctas=True)
            if not B_RESIDENT:
                tlx.async_descriptor_load(b_desc, b_tiles[smem_buf],
                                          [pid_n * BLOCK_SIZE_N + cluster_cta_rank * BLOCK_N_PER_CTA, offs_k],
                                          smem_full[smem_buf], eviction_policy="evict_last", two_ctas=True)
            tlx.async_descriptor_load(a_scale_desc, a_scale_tiles[smem_buf], [0, pid_m * REP_M, offs_scale_k, 0, 0],
                                      smem_full[smem_buf], eviction_policy="evict_last", two_ctas=True)
            if not B_RESIDENT:
                tlx.async_descriptor_load(b_scale_desc, b_scale_tiles[smem_buf], [0, pid_n * REP_N, offs_scale_k, 0, 0],
                                          smem_full[smem_buf], pred=cluster_cta_rank == 0, eviction_policy="evict_last",
                                          multicast_targets=[0, 1], two_ctas=True)
        else:
            tlx.barrier_expect_bytes(smem_full[smem_buf], TOTAL_BYTES_PER_CTA)
            tlx.async_descriptor_load(
                a_desc,
                a_tiles[smem_buf],
                [pid_m * BLOCK_SIZE_M, offs_k],
                smem_full[smem_buf],
            )
            if not B_RESIDENT:
                tlx.async_descriptor_load(
                    b_desc,
                    b_tiles[smem_buf],
                    [pid_n * BLOCK_SIZE_N + cluster_cta_rank * BLOCK_N_PER_CTA, offs_k],
                    smem_full[smem_buf],
                )
            tlx.async_descriptor_load(
                a_scale_desc,
                a_scale_tiles[smem_buf],
                [0, pid_m * REP_M, offs_scale_k, 0, 0],
                smem_full[smem_buf],
            )
            if not B_RESIDENT:
                tlx.async_descriptor_load(
                    b_scale_desc,
                    b_scale_tiles[smem_buf],
                    [0, pid_n * REP_N, offs_scale_k, 0, 0],
                    smem_full[smem_buf],
                )
        smem_count += 1
        smem_buf += 1
        if smem_buf == NUM_SMEM_BUFFERS:
            smem_buf = 0
            smem_phase ^= 1

    return smem_count


@triton.jit
# Triton TR001: this reduction deliberately uses fixed 32x32 tiles.
def _reduce_k_kernel(  # noqa: TR001
    workspace_ptr,
    out_ptr,
    M,
    N,
    ROWS_PER_SPLIT,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    REDUCE_OUTPUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for split_id in range(SPLIT_K):
        partial = tl.load(workspace_ptr + base_offs + split_id * ROWS_PER_SPLIT * N, mask=mask, other=0.0)
        acc += partial.to(tl.float32)

    tl.store(out_ptr + base_offs, acc.to(REDUCE_OUTPUT_DTYPE), mask=mask)


def reduce_post_hook(nargs, exception=None):
    if exception is not None:
        return
    split_k = nargs.get("SPLIT_K", 1)
    if split_k <= 1:
        return
    M = nargs["M"]
    N = nargs["N"]
    workspace = nargs["workspace_desc"].base
    out = nargs["out_desc"].base
    reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    _reduce_k_kernel[reduce_grid](
        workspace,
        out,
        M,
        N,
        workspace.shape[0] // split_k,
        SPLIT_K=split_k,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        REDUCE_OUTPUT_DTYPE=OUTPUT_DTYPE,
    )


@triton.jit
# Triton TR001: scaled MMA requires 128-row A tiles; config pruning keeps the tutorial safe.
def _gemm_mxfp8_ws_kernel(  # noqa: TR001
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    out_desc,
    workspace_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    INTERLEAVE_EPILOGUE: tl.constexpr = 0,
    USE_WARP_BARRIER: tl.constexpr = False,
    PEELED_FIRST_K: tl.constexpr = False,
    OVERLAP_ACC: tl.constexpr = False,
    COLLAB_2CTA: tl.constexpr = False,
    N_INNER: tl.constexpr = False,
    ASYNC_STORE: tl.constexpr = False,
    B_RESIDENT: tl.constexpr = False,
    B_RES_N_TILES: tl.constexpr = 1,
    B_RES_K_TILES: tl.constexpr = 1,
    FUSE_N_TILES: tl.constexpr = 1,
    NUM_SMS: tl.constexpr = 148,
):
    tl.static_assert(BLOCK_SIZE_M == 128, "scaled MMA requires BLOCK_SIZE_M=128")
    tl.static_assert(
        BLOCK_SIZE_N % NUM_CTAS == 0 and BLOCK_SIZE_N // NUM_CTAS <= 256,
        "scaled MMA supports at most 256 B columns per CTA",
    )
    tl.static_assert(BLOCK_SIZE_K % 128 == 0, "BLOCK_SIZE_K must cover complete scale tiles")
    tl.static_assert(NUM_CTAS == 1 or NUM_CTAS == 2, "NUM_CTAS must be 1 or 2")
    tl.static_assert(NUM_MMA_GROUPS == 1, "scaled MMA uses one 128-row MMA group")

    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
    else:
        cluster_cta_rank = 0

    REP_M: tl.constexpr = BLOCK_SIZE_M // 128
    REP_N: tl.constexpr = BLOCK_SIZE_N // 128
    # Local shadow of the module-level VEC_SIZE: the frontend rejects reads of
    # non-constexpr globals, and making that global tl.constexpr would disable
    # this kernel's fast dispatch path.
    VEC_SIZE: tl.constexpr = 32
    REP_K: tl.constexpr = BLOCK_SIZE_K // VEC_SIZE // 4
    BLOCK_N_PER_CTA: tl.constexpr = BLOCK_SIZE_N // NUM_CTAS
    # A scheduled tile spans FUSE_N_TILES MMA tiles along N.
    SCHED_BLOCK_N: tl.constexpr = BLOCK_SIZE_N * FUSE_N_TILES
    if FUSE_N_TILES > 1:
        tl.static_assert(B_RESIDENT and not OVERLAP_ACC and B_RES_N_TILES % FUSE_N_TILES == 0)

    a_tiles = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_K),
        tlx.dtype_of(a_desc),
        NUM_SMEM_BUFFERS,
    )
    if B_RESIDENT:
        tl.static_assert(not OVERLAP_ACC and not PEELED_FIRST_K and SPLIT_K == 1)
        tl.static_assert(NUM_CTAS == 1 or COLLAB_2CTA)
        NUM_B_BUFFERS: tl.constexpr = B_RES_N_TILES * B_RES_K_TILES
    else:
        NUM_B_BUFFERS: tl.constexpr = NUM_SMEM_BUFFERS
    b_tiles = tlx.local_alloc(
        (BLOCK_N_PER_CTA, BLOCK_SIZE_K),
        tlx.dtype_of(b_desc),
        NUM_B_BUFFERS,
    )
    a_scale_tiles = tlx.local_alloc(
        (1, REP_M, REP_K, 2, 256),
        tl.uint8,
        NUM_SMEM_BUFFERS,
    )
    b_scale_tiles = tlx.local_alloc(
        (1, REP_N, REP_K, 2, 256),
        tl.uint8,
        NUM_B_BUFFERS,
    )
    # One-shot barrier: every resident B tile and scale has landed.
    b_ready = tlx.alloc_barriers(1, arrive_count=1)
    if OVERLAP_ACC:
        # Two accumulator slots: slot 0 owns columns [0, N) and slot 1 owns
        # [N - N / SUBTILE, 2N - N / SUBTILE), so they share one epilogue
        # subtile, which the epilogue drains first. The block scales are staged
        # in TMEM past the slots so the layout fits in 512 columns.
        tl.static_assert((NUM_CTAS == 1 or COLLAB_2CTA) and NUM_TMEM_BUFFERS == 1)
        tl.static_assert(BLOCK_SIZE_N == 256 and EPILOGUE_SUBTILE == 4)
        # Scale TMEM rows are the per-CTA MMA M / N; columns hold the rest.
        A_SCALE_TMEM_COLS: tl.constexpr = REP_M * REP_K * 2 * 256 // BLOCK_SIZE_M
        B_SCALE_TMEM_COLS: tl.constexpr = REP_N * REP_K * 2 * 256 // BLOCK_N_PER_CTA
        acc_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
        accumulators = tlx.local_alloc(
            (BLOCK_SIZE_M, 2 * BLOCK_SIZE_N),
            tl.float32,
            1,
            tlx.storage_kind.tmem,
            reuse=acc_alias,
        )
        acc_pad_n = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, 1, tlx.storage_kind.tmem, reuse=acc_alias)
        acc_pad_half = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), tl.float32, 1, tlx.storage_kind.tmem,
                                       reuse=acc_alias)
        acc_pad_quarter = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N // 4), tl.float32, 1, tlx.storage_kind.tmem,
                                          reuse=acc_alias)
        a_scale_tmem = tlx.local_alloc((BLOCK_SIZE_M, A_SCALE_TMEM_COLS), tl.uint8, 1, tlx.storage_kind.tmem,
                                       reuse=acc_alias)
        b_scale_tmem = tlx.local_alloc((BLOCK_N_PER_CTA, B_SCALE_TMEM_COLS), tl.uint8, 1, tlx.storage_kind.tmem,
                                       reuse=acc_alias)
        acc_alias.set_buffer_overlap(
            tlx.reuse_group(
                accumulators,
                tlx.reuse_group(
                    acc_pad_n,
                    acc_pad_half,
                    acc_pad_quarter,
                    a_scale_tmem,
                    b_scale_tmem,
                    group_type=tlx.reuse_group_type.distinct,
                ),
                group_type=tlx.reuse_group_type.shared,
            ))
        TMEM_EMPTY_ARRIVES: tl.constexpr = 1
    else:
        accumulators = tlx.local_alloc(
            (BLOCK_SIZE_M, BLOCK_SIZE_N),
            tl.float32,
            NUM_TMEM_BUFFERS * FUSE_N_TILES,
            tlx.storage_kind.tmem,
        )
        a_scale_tmem = a_scale_tiles
        b_scale_tmem = b_scale_tiles
        TMEM_EMPTY_ARRIVES: tl.constexpr = EPILOGUE_SUBTILE * FUSE_N_TILES

    smem_full = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    smem_empty = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(
        NUM_TMEM_BUFFERS,
        arrive_count=TMEM_EMPTY_ARRIVES * (2 if COLLAB_2CTA else 1),
    )
    if COLLAB_2CTA:
        tl.static_assert(NUM_CTAS == 2 and not PEELED_FIRST_K)
        tlx.fence_mbarrier_init_cluster()
    if NUM_CTAS == 2:
        cta_bars = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=2)
    else:
        cta_bars = smem_full
    if ASYNC_STORE:
        store_bufs = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE),
                                     tlx.dtype_of(workspace_desc if SPLIT_K > 1 else out_desc), 2)
    else:
        store_bufs = None
    with tlx.async_tasks(
            exclusive=True,
            no_ending_cluster_sync=True,
            mbarrier_try_wait_suspend_ns=50000,
    ):
        with tlx.async_task("default"):
            (
                start_tile_id,
                num_programs,
                num_pid_m,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                SCHED_BLOCK_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
                NUM_SMS,
            )
            tmem_count = 0
            store_count = 0
            tile_id = _first_tile(start_tile_id, N, SCHED_BLOCK_N, N_INNER)
            while tile_id < num_tiles:
                k_tile_start, k_tile_end = _compute_k_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K)
                if SPLIT_K == 1 or k_tile_end > k_tile_start:
                    tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                    store_count = _process_tile_epilogue_inner(
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        num_mn_tiles=num_mn_tiles,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        M=M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                        SPLIT_K=SPLIT_K,
                        out_desc=out_desc,
                        workspace_desc=workspace_desc,
                        accumulators=accumulators,
                        tmem_full=tmem_full,
                        tmem_empty=tmem_empty,
                        tmem_buf=tmem_buf,
                        tmem_phase=tmem_phase,
                        NUM_CTAS=NUM_CTAS,
                        OVERLAP_ACC=OVERLAP_ACC,
                        COLLAB_2CTA=COLLAB_2CTA,
                        store_bufs=store_bufs,
                        store_count=store_count,
                        ASYNC_STORE=ASYNC_STORE,
                        FUSE_N_TILES=FUSE_N_TILES,
                    )
                    tmem_count += 1
                tile_id = _next_tile(tile_id, num_programs, N, SCHED_BLOCK_N, N_INNER)
            if ASYNC_STORE:
                tlx.async_descriptor_store_wait(0)

        with tlx.async_task(num_warps=1, num_regs=48):
            (
                start_tile_id,
                num_programs,
                num_pid_m,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                SCHED_BLOCK_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
                NUM_SMS,
            )
            smem_count = 0
            tmem_count = 0
            if B_RESIDENT:
                if NUM_CTAS == 1 or cluster_cta_rank == 0:
                    tlx.barrier_wait(b_ready[0], 0)
            tile_id = _first_tile(start_tile_id, N, SCHED_BLOCK_N, N_INNER)
            while tile_id < num_tiles:
                k_tile_start, k_tile_end = _compute_k_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K)
                if SPLIT_K == 1 or k_tile_end > k_tile_start:
                    tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                    if OVERLAP_ACC:
                        if NUM_CTAS == 1 or cluster_cta_rank == 0:
                            tlx.barrier_wait(tmem_empty[tmem_buf], tmem_phase ^ 1)
                            smem_count = _process_tile_mma_overlap(
                                k_tile_start=k_tile_start,
                                k_tile_end=k_tile_end,
                                NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                                smem_count=smem_count,
                                tmem_phase=tmem_phase,
                                a_tiles=a_tiles,
                                b_tiles=b_tiles,
                                a_scale_tiles=a_scale_tiles,
                                b_scale_tiles=b_scale_tiles,
                                accumulators=accumulators,
                                a_scale_tmem=a_scale_tmem,
                                b_scale_tmem=b_scale_tmem,
                                smem_full=smem_full,
                                smem_empty=smem_empty,
                                BLOCK_SIZE_N=BLOCK_SIZE_N,
                                EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                                TWO_CTAS=NUM_CTAS == 2,
                            )
                            tlx.tcgen05_commit(tmem_full[tmem_buf], two_ctas=NUM_CTAS == 2)
                        else:
                            smem_count += k_tile_end - k_tile_start
                    elif COLLAB_2CTA:
                        # Only the leader issues the collaborative MMA; the peer's
                        # TMA completions land on the leader's barriers.
                        if cluster_cta_rank == 0:
                            tlx.barrier_wait(tmem_empty[tmem_buf], tmem_phase ^ 1)
                            smem_count = _process_tile_mma_inner(
                                k_tile_start=k_tile_start,
                                k_tile_end=k_tile_end,
                                NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                                smem_count=smem_count,
                                tmem_buf=tmem_buf,
                                a_tiles=a_tiles,
                                b_tiles=b_tiles,
                                a_scale_tiles=a_scale_tiles,
                                b_scale_tiles=b_scale_tiles,
                                accumulators=accumulators,
                                smem_full=smem_full,
                                smem_empty=smem_empty,
                                cta_bars=cta_bars,
                                NUM_CTAS=NUM_CTAS,
                                cluster_cta_rank=cluster_cta_rank,
                                DO_MMA=True,
                                COLLAB_2CTA=True,
                                B_RESIDENT=B_RESIDENT,
                                b_res_base=_b_res_base(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M,
                                                       B_RES_K_TILES * FUSE_N_TILES),
                                FUSE_N_TILES=FUSE_N_TILES,
                                B_RES_K_TILES=B_RES_K_TILES,
                            )
                            tlx.tcgen05_commit(tmem_full[tmem_buf], two_ctas=True)
                        else:
                            smem_count += k_tile_end - k_tile_start
                    else:
                        if PEELED_FIRST_K:
                            smem_buf, smem_phase = get_bufidx_phase(smem_count, NUM_SMEM_BUFFERS)
                            tlx.barrier_wait(smem_full[smem_buf], smem_phase)
                            tlx.barrier_wait(tmem_empty[tmem_buf], tmem_phase ^ 1)
                            if NUM_CTAS == 2:
                                pred_cta0 = cluster_cta_rank == 0
                                tlx.barrier_arrive(cta_bars[smem_buf], arrive_count=1, remote_cta_rank=0)
                                tlx.barrier_wait(
                                    cta_bars[smem_buf],
                                    phase=smem_phase,
                                    pred=pred_cta0,
                                )
                            tlx.async_dot_scaled(
                                a_tiles[smem_buf],
                                tlx.local_trans(b_tiles[smem_buf]),
                                accumulators[tmem_buf],
                                a_scale_tiles[smem_buf],
                                "e4m3",
                                b_scale_tiles[smem_buf],
                                "e4m3",
                                use_acc=False,
                                out_dtype=tl.float32,
                                mBarriers=[smem_empty[smem_buf]],
                                two_ctas=NUM_CTAS == 2,
                            )
                            smem_count += 1
                            k_tile_start += 1
                        else:
                            tlx.barrier_wait(tmem_empty[tmem_buf], tmem_phase ^ 1)
                        smem_count = _process_tile_mma_inner(
                            k_tile_start=k_tile_start,
                            k_tile_end=k_tile_end,
                            NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                            smem_count=smem_count,
                            tmem_buf=tmem_buf,
                            a_tiles=a_tiles,
                            b_tiles=b_tiles,
                            a_scale_tiles=a_scale_tiles,
                            b_scale_tiles=b_scale_tiles,
                            accumulators=accumulators,
                            smem_full=smem_full,
                            smem_empty=smem_empty,
                            cta_bars=cta_bars,
                            NUM_CTAS=NUM_CTAS,
                            cluster_cta_rank=cluster_cta_rank,
                            DO_MMA=True,
                            USE_ACC_INITIAL=PEELED_FIRST_K,
                            B_RESIDENT=B_RESIDENT,
                            b_res_base=_b_res_base(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M,
                                                   B_RES_K_TILES * FUSE_N_TILES),
                            FUSE_N_TILES=FUSE_N_TILES,
                            B_RES_K_TILES=B_RES_K_TILES,
                        )
                        if NUM_CTAS == 1:
                            # Signal accumulator readiness asynchronously so the MMA
                            # task can start the next tile while this one drains.
                            tlx.tcgen05_commit(tmem_full[tmem_buf])
                        else:
                            last_smem_buf, last_smem_phase = get_bufidx_phase(smem_count - 1, NUM_SMEM_BUFFERS)
                            tlx.barrier_wait(smem_empty[last_smem_buf], last_smem_phase)
                            tlx.barrier_arrive(tmem_full[tmem_buf], arrive_count=1)
                    tmem_count += 1
                tile_id = _next_tile(tile_id, num_programs, N, SCHED_BLOCK_N, N_INNER)

        with tlx.async_task(num_warps=1, num_regs=48):
            (
                start_tile_id,
                num_programs,
                num_pid_m,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                SCHED_BLOCK_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
                NUM_SMS,
            )
            smem_count = 0
            if B_RESIDENT:
                _load_resident_b(b_desc, b_scale_desc, b_tiles, b_scale_tiles, b_ready, cluster_cta_rank, BLOCK_SIZE_N,
                                 BLOCK_SIZE_K, NUM_CTAS, B_RES_N_TILES, B_RES_K_TILES)
            tile_id = _first_tile(start_tile_id, N, SCHED_BLOCK_N, N_INNER)
            while tile_id < num_tiles:
                k_tile_start, k_tile_end = _compute_k_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K)
                if SPLIT_K == 1 or k_tile_end > k_tile_start:
                    smem_count = _process_tile_producer_inner(
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        num_mn_tiles=num_mn_tiles,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        k_tile_start=k_tile_start,
                        k_tile_end=k_tile_end,
                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                        a_desc=a_desc,
                        a_scale_desc=a_scale_desc,
                        b_desc=b_desc,
                        b_scale_desc=b_scale_desc,
                        a_tiles=a_tiles,
                        b_tiles=b_tiles,
                        a_scale_tiles=a_scale_tiles,
                        b_scale_tiles=b_scale_tiles,
                        smem_full=smem_full,
                        smem_empty=smem_empty,
                        smem_count=smem_count,
                        SPLIT_K=SPLIT_K,
                        NUM_CTAS=NUM_CTAS,
                        cluster_cta_rank=cluster_cta_rank,
                        COLLAB_2CTA=COLLAB_2CTA,
                        B_RESIDENT=B_RESIDENT,
                    )
                tile_id = _next_tile(tile_id, num_programs, N, SCHED_BLOCK_N, N_INNER)


def _scale_5d(scale, rows, K, sf_layout):
    """View or pack E8M0 scales into the cuBLAS-blocked 5D tile layout."""
    k_chunks = K // (4 * VEC_SIZE)
    if sf_layout == "natural":
        # row = row_group * 32 + row_lane; within a 512-byte atom,
        # dest = row_lane * 16 + row_group * 4 + col.
        packed = scale.view(torch.uint8).view(rows // 128, 4, 32, k_chunks, 4).permute(0, 3, 2, 1, 4).contiguous()
        scale = packed.view(torch.float8_e8m0fnu)
    elif sf_layout != "cublas_blocked":
        raise ValueError(f"unsupported sf_layout {sf_layout!r}; expected 'natural' or 'cublas_blocked'")
    return scale.reshape(1, rows // 128, k_chunks, 2, 256)


def _autotuned(configs, prune=None):
    return triton.autotune(
        configs=configs,
        key=["M", "N", "K"],
        prune_configs_by={"early_config_prune": prune} if prune is not None else None,
        post_hook=reduce_post_hook,
    )(_gemm_mxfp8_ws_kernel)


@functools.lru_cache(maxsize=None)
def _tuned(space, heuristic_key=None):
    """Autotuned kernel per search space; ``heuristic_key`` keys only the heuristic one."""
    if space == "heuristic":
        return _autotuned([_make_config(heuristic_config(*heuristic_key))])
    return _autotuned(get_cuda_autotune_config(), preprocess_configs)


def _launch(kernel, a, a_scale, b, b_scale, *, out, sf_layout, num_sms):
    M, K = a.shape
    N = b.shape[0]
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    a_scale = _scale_5d(a_scale, M, K, sf_layout)
    b_scale = _scale_5d(b_scale, N, K, sf_layout)

    block_2d = [1, 1]
    block_5d = [1, 1, 1, 1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), block_2d)
    a_scale_desc = TensorDescriptor(a_scale, a_scale.shape, a_scale.stride(), block_5d)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), block_2d)
    b_scale_desc = TensorDescriptor(b_scale, b_scale.shape, b_scale.stride(), block_5d)
    out_desc = TensorDescriptor(out, out.shape, out.stride(), block_2d)
    # Dummy workspace; the pre_hook sizes a real one once SPLIT_K is known.
    workspace_desc = TensorDescriptor(out, out.shape, out.stride(), block_2d)

    def grid(META):
        num_ctas = META["NUM_CTAS"]
        num_pid_m = triton.cdiv(triton.cdiv(M, META["BLOCK_SIZE_M"]), num_ctas) * num_ctas
        block_n = META["BLOCK_SIZE_N"] * (META.get("FUSE_N_TILES") or 1)
        total_tiles = num_pid_m * triton.cdiv(N, block_n) * META["SPLIT_K"]
        return (min(num_sms, total_tiles), )

    kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        out_desc,
        workspace_desc,
        M,
        N,
        K,
        NUM_SMS=num_sms,
    )

    # post_hook only fires while benchmarking, so reduce here for the real launch.
    split_k = kernel.best_config.kwargs.get("SPLIT_K", 1)
    if split_k > 1:
        _reduce_k_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
            workspace_desc.base,
            out,
            M,
            N,
            workspace_desc.base.shape[0] // split_k,
            SPLIT_K=split_k,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            REDUCE_OUTPUT_DTYPE=OUTPUT_DTYPE,
        )
    return out


def _run_config(a, a_scale, b, b_scale, meta, *, out=None, sf_layout="natural"):
    """Launch one explicit config, bypassing the heuristic. Not part of the op API."""
    M, K = a.shape
    N = b.shape[0]
    error = _config_error({**DEFAULT_CONFIG, **meta}, M, N, K)
    if error is not None:
        raise ValueError(error)
    with torch.cuda.device(a.device):
        return _launch(_autotuned([_make_config(meta)]), a, a_scale, b, b_scale, out=out, sf_layout=sf_layout,
                       num_sms=_num_sms(a.device.index))


def mm_mxfp8(a, a_scale, b, b_scale, *, out=None, sf_layout="natural", space="full"):
    """``a @ b.T`` for pre-quantized MXFP8 operands on Blackwell, as BF16.

    Inputs are validated by ``tlx.ops.mm_mxfp8``. ``space`` is "full"
    (autotune, the default) or "heuristic" (one shape-picked config).
    """
    if space not in ("heuristic", "full"):
        raise ValueError(f"unsupported space {space!r}; expected 'heuristic' or 'full'")
    M, K = a.shape
    N = b.shape[0]
    with torch.cuda.device(a.device):
        num_sms = _num_sms(a.device.index)
        kernel = _tuned(space, (M, N, K, num_sms) if space == "heuristic" else None)
        return _launch(kernel, a, a_scale, b, b_scale, out=out, sf_layout=sf_layout, num_sms=num_sms)


__all__ = ["mm_mxfp8"]
