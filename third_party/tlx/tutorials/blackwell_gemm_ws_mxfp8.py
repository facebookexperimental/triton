"""Warp-specialized MXFP8 GEMM for NVIDIA Blackwell GPUs.

The kernel computes ``C[M, N] = A[M, K] @ B[N, K].T``. A and B contain
E4M3 data, and each group of 32 values along K has an E8M0 scale. The scale
tensors use the cuBLAS block-scaled layout expected by ``tlx.async_dot_scaled``.

Three specialized tasks overlap the work:

* the producer loads A, B, and both scale tiles with TMA;
* the MMA consumer accumulates block-scaled products in TMEM;
* the epilogue consumer converts FP32 accumulators to BF16 and stores them.

Quantization is intentionally outside this tutorial. ``matmul`` accepts the
``qdata`` and swizzled ``scale`` tensors returned by ``torchao``'s
``MXTensor.to_mx(..., is_swizzled_scales=True)``.
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


@functools.lru_cache(maxsize=1)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def _select_group_size_m(M, N, block_m):
    num_m_tiles = triton.cdiv(M, block_m)
    ratio = M / max(N, 1)
    if ratio > 10:
        return 1
    if ratio < 0.1:
        return min(64, num_m_tiles)
    return min(8, num_m_tiles)


def get_heuristic_config(M, N, K, num_sms=148):
    """Select a safe 1-CTA scaled-MMA configuration for explicit use."""
    block_m = 128
    block_n = 128
    block_k = 128
    num_mn_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    k_tiles = triton.cdiv(K, block_k)

    split_k = 1
    if num_mn_tiles < num_sms:
        for candidate in [8, 4, 2]:
            if k_tiles >= candidate and k_tiles // candidate >= 4:
                split_k = candidate
                break

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": _select_group_size_m(M, N, block_m),
        "NUM_SMEM_BUFFERS": 3 if split_k == 1 else 4,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": SUPPORTED_NUM_MMA_GROUPS,
        "EPILOGUE_SUBTILE": 4,
        "NUM_CTAS": DEFAULT_NUM_CTAS,
        "SPLIT_K": split_k,
        "pre_hook": matmul_tma_set_block_size_hook,
        "ctas_per_cga": None,
    }


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
        workspace = torch.empty((SPLIT_K * M, N), device=out.device, dtype=out.dtype)
        nargs["workspace_desc"].base = workspace
        nargs["workspace_desc"].shape = list(workspace.shape)
    else:
        nargs["workspace_desc"].base = nargs["out_desc"].base
        nargs["workspace_desc"].shape = list(nargs["out_desc"].base.shape)
    nargs["workspace_desc"].block_shape = [BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]


def _normalize_config(config: dict[str, object] | None) -> dict[str, object]:
    meta = {**DEFAULT_CONFIG, **(config or {})}
    aliases = {
        "BLOCK_M": "BLOCK_SIZE_M",
        "BLOCK_N": "BLOCK_SIZE_N",
        "BLOCK_K": "BLOCK_SIZE_K",
    }
    for old_key, new_key in aliases.items():
        if old_key not in meta:
            continue
        if new_key not in meta or config is not None and new_key not in config:
            meta[new_key] = meta[old_key]
        del meta[old_key]
    return meta


def _validate_config(meta: dict[str, object], M: int, N: int, K: int) -> None:
    block_m = int(meta["BLOCK_SIZE_M"])
    block_n = int(meta["BLOCK_SIZE_N"])
    block_k = int(meta["BLOCK_SIZE_K"])
    num_ctas = int(meta.get("NUM_CTAS", 1))

    assert block_m == 128, "scaled MMA requires BLOCK_SIZE_M=128"
    assert num_ctas in SUPPORTED_NUM_CTAS, "NUM_CTAS must be 1 or 2"
    assert (block_n % num_ctas == 0 and block_n // num_ctas <= 256), "scaled MMA supports at most 256 B columns per CTA"
    assert block_k % 128 == 0, "BLOCK_SIZE_K must cover complete scale tiles"
    assert meta["EPILOGUE_SUBTILE"] in (1, 2, 4), "unsupported epilogue subtile"
    assert (meta.get("NUM_MMA_GROUPS", 1) == SUPPORTED_NUM_MMA_GROUPS), "scaled MMA requires one 128-row MMA group"
    assert (int(meta.get("GROUP_SIZE_M", 1)) % num_ctas == 0), "GROUP_SIZE_M must be a multiple of NUM_CTAS"
    if num_ctas == 2:
        assert (int(meta.get("NUM_TMEM_BUFFERS", 1)) == 1), "2-CTA MXFP8 requires one TMEM buffer"

    split_k = int(meta.get("SPLIT_K", 1))
    k_tiles = triton.cdiv(K, block_k)
    assert split_k >= 1, "SPLIT_K must be positive"
    assert split_k <= k_tiles, "SPLIT_K creates empty K ranges"
    if split_k > 1:
        assert triton.cdiv(M, block_m) * triton.cdiv(N, block_n) > 0


def preprocess_configs(configs, named_args, **kwargs):
    NUM_SMS = _get_num_sms()
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
            target_tmem_buffers = (1, )
            pruned_configs = [
                conf for conf in pruned_configs
                if conf.kwargs["BLOCK_SIZE_K"] == 128 and conf.kwargs["GROUP_SIZE_M"] == 4
                and conf.kwargs["NUM_SMEM_BUFFERS"] == 4 and conf.kwargs["NUM_TMEM_BUFFERS"] in (
                    target_tmem_buffers if conf.kwargs["NUM_CTAS"] == 1 else (
                        1, )) and conf.kwargs["EPILOGUE_SUBTILE"] == 1 and (
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
):
    mn_tile_id = tile_id if SPLIT_K == 1 else tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    SUB_N: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE

    if SPLIT_K > 1:
        split_id = tile_id // num_mn_tiles
        store_desc = workspace_desc
        row_base = split_id * M
    else:
        store_desc = out_desc
        row_base = 0

    tlx.barrier_wait(tmem_full[tmem_buf], tmem_phase)
    for slice_id in tl.static_range(EPILOGUE_SUBTILE):
        acc_slice = tlx.local_slice(
            accumulators[tmem_buf],
            [0, slice_id * SUB_N],
            [BLOCK_SIZE_M, SUB_N],
        )
        result = tlx.local_load(acc_slice)
        tlx.barrier_arrive(tmem_empty[tmem_buf], arrive_count=1)

        store_desc.store(
            [
                row_base + pid_m * BLOCK_SIZE_M,
                pid_n * BLOCK_SIZE_N + slice_id * SUB_N,
            ],
            result.to(tlx.dtype_of(store_desc)),
        )


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
):
    local_k_tiles = k_tile_end - k_tile_start
    smem_buf, smem_phase = get_bufidx_phase(smem_count, NUM_SMEM_BUFFERS)

    pred_cta0 = cluster_cta_rank == 0

    for k_idx in range(0, local_k_tiles):
        tlx.barrier_wait(smem_full[smem_buf], smem_phase)
        if NUM_CTAS == 2:
            tlx.barrier_arrive(cta_bars[smem_buf], arrive_count=1, remote_cta_rank=0)
            tlx.barrier_wait(cta_bars[smem_buf], phase=smem_phase, pred=pred_cta0)
        if DO_MMA:
            tlx.async_dot_scaled(
                a_tiles[smem_buf],
                tlx.local_trans(b_tiles[smem_buf]),
                accumulators[tmem_buf],
                a_scale_tiles[smem_buf],
                "e4m3",
                b_scale_tiles[smem_buf],
                "e4m3",
                use_acc=USE_ACC_INITIAL or k_idx != 0,
                out_dtype=tl.float32,
                mBarriers=[smem_empty[smem_buf]],
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
    TOTAL_BYTES_PER_CTA: tl.constexpr = A_BYTES + B_BYTES + A_SCALE_BYTES + B_SCALE_BYTES
    smem_buf, smem_phase = get_bufidx_phase(smem_count, NUM_SMEM_BUFFERS)
    for k in range(k_tile_start, k_tile_end):
        offs_k = k * BLOCK_SIZE_K
        offs_scale_k = k * REP_K

        tlx.barrier_wait(smem_empty[smem_buf], smem_phase ^ 1)
        tlx.barrier_expect_bytes(smem_full[smem_buf], TOTAL_BYTES_PER_CTA)
        tlx.async_descriptor_load(
            a_desc,
            a_tiles[smem_buf],
            [pid_m * BLOCK_SIZE_M, offs_k],
            smem_full[smem_buf],
        )
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
        partial = tl.load(workspace_ptr + base_offs + split_id * M * N, mask=mask, other=0.0)
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
        SPLIT_K=split_k,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        REDUCE_OUTPUT_DTYPE=OUTPUT_DTYPE,
    )


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": preprocess_configs},
    post_hook=reduce_post_hook,
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

    a_tiles = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_K),
        tlx.dtype_of(a_desc),
        NUM_SMEM_BUFFERS,
    )
    b_tiles = tlx.local_alloc(
        (BLOCK_N_PER_CTA, BLOCK_SIZE_K),
        tlx.dtype_of(b_desc),
        NUM_SMEM_BUFFERS,
    )
    a_scale_tiles = tlx.local_alloc(
        (1, REP_M, REP_K, 2, 256),
        tl.uint8,
        NUM_SMEM_BUFFERS,
    )
    b_scale_tiles = tlx.local_alloc(
        (1, REP_N, REP_K, 2, 256),
        tl.uint8,
        NUM_SMEM_BUFFERS,
    )
    accumulators = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS,
        tlx.storage_kind.tmem,
    )

    smem_full = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    smem_empty = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(
        NUM_TMEM_BUFFERS,
        arrive_count=EPILOGUE_SUBTILE,
    )
    if NUM_CTAS == 2:
        cta_bars = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=2)
    else:
        cta_bars = smem_full
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
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
                NUM_SMS,
            )
            tmem_count = 0
            tile_id = start_tile_id
            while tile_id < num_tiles:
                k_tile_start, k_tile_end = _compute_k_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K)
                if SPLIT_K == 1 or k_tile_end > k_tile_start:
                    tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                    _process_tile_epilogue_inner(
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
                    )
                    tmem_count += 1
                tile_id += num_programs

        with tlx.async_task(num_warps=1, num_regs=24):
            (
                start_tile_id,
                num_programs,
                _num_pid_m,
                _num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
                NUM_SMS,
            )
            smem_count = 0
            tmem_count = 0
            tile_id = start_tile_id
            while tile_id < num_tiles:
                k_tile_start, k_tile_end = _compute_k_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K)
                if SPLIT_K == 1 or k_tile_end > k_tile_start:
                    tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
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
                    )
                    last_smem_buf, last_smem_phase = get_bufidx_phase(smem_count - 1, NUM_SMEM_BUFFERS)
                    tlx.barrier_wait(smem_empty[last_smem_buf], last_smem_phase)
                    tlx.barrier_arrive(tmem_full[tmem_buf], arrive_count=1)
                    tmem_count += 1
                tile_id += num_programs

        with tlx.async_task(num_warps=1, num_regs=24):
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
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
                NUM_SMS,
            )
            smem_count = 0
            tile_id = start_tile_id
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
                    )
                tile_id += num_programs


def _reshape_scale(scale: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    expected_elements = rows * k // VEC_SIZE
    assert (scale.numel() == expected_elements), f"Expected {expected_elements} E8M0 scales, got {scale.numel()}"
    return scale.reshape(1, rows // 128, k // 128, 2, 256)


def _make_descriptors(a, b, a_scale, b_scale, out):
    dummy_block_2d = [1, 1]
    dummy_block_5d = [1, 1, 1, 1, 1]
    return (
        TensorDescriptor(a, a.shape, a.stride(), dummy_block_2d),
        TensorDescriptor(a_scale, a_scale.shape, a_scale.stride(), dummy_block_5d),
        TensorDescriptor(b, b.shape, b.stride(), dummy_block_2d),
        TensorDescriptor(b_scale, b_scale.shape, b_scale.stride(), dummy_block_5d),
        TensorDescriptor(out, out.shape, out.stride(), dummy_block_2d),
    )


_BEST_CONFIG_CACHE: dict[tuple[int, int, int, int], dict[str, object]] = {}


def _config_to_meta(config) -> dict[str, object]:
    meta = dict(config.kwargs)
    meta.update({
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "ctas_per_cga": config.ctas_per_cga,
        "pre_hook": config.pre_hook,
    })
    return meta


def _launch_with_config(a_desc, a_scale_desc, b_desc, b_scale_desc, out_desc, M, N, K, meta):
    meta = dict(meta)
    ctas_per_cga = meta.pop("ctas_per_cga", None)
    pre_hook = meta.pop("pre_hook", None)
    num_warps = meta.pop("num_warps", 4)
    num_stages = meta.pop("num_stages", 1)
    num_ctas = int(meta.get("NUM_CTAS", 1))
    expected_ctas_per_cga = (2, 1, 1) if num_ctas == 2 else None
    if ctas_per_cga is None and num_ctas == 2:
        ctas_per_cga = expected_ctas_per_cga
    assert ctas_per_cga == expected_ctas_per_cga, "ctas_per_cga must match NUM_CTAS"

    workspace_desc = TensorDescriptor(out_desc.base, out_desc.shape, out_desc.strides, [1, 1])
    hook_args = {
        "a_desc": a_desc,
        "a_scale_desc": a_scale_desc,
        "b_desc": b_desc,
        "b_scale_desc": b_scale_desc,
        "out_desc": out_desc,
        "workspace_desc": workspace_desc,
        "M": M,
        "N": N,
        "K": K,
        **meta,
    }
    if pre_hook is not None:
        pre_hook(hook_args)
    else:
        matmul_tma_set_block_size_hook(hook_args)

    split_k = meta.get("SPLIT_K", 1)
    num_pid_m = triton.cdiv(M, meta["BLOCK_SIZE_M"])
    num_pid_m = ((num_pid_m + num_ctas - 1) // num_ctas) * num_ctas
    total_tiles = num_pid_m * triton.cdiv(N, meta["BLOCK_SIZE_N"]) * split_k
    grid = (min(_get_num_sms(), total_tiles), )
    _gemm_mxfp8_ws_kernel.fn[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        out_desc,
        workspace_desc,
        M,
        N,
        K,
        **meta,
        NUM_SMS=_get_num_sms(),
        num_warps=num_warps,
        num_stages=num_stages,
        ctas_per_cga=ctas_per_cga,
    )
    if split_k > 1:
        reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
        _reduce_k_kernel[reduce_grid](
            workspace_desc.base,
            out_desc.base,
            M,
            N,
            SPLIT_K=split_k,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            REDUCE_OUTPUT_DTYPE=OUTPUT_DTYPE,
        )


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    config: dict[str, object] | None = None,
) -> torch.Tensor:
    """Compute ``A @ B.T`` from pre-quantized MXFP8 tensors.

    Args:
        a: Contiguous E4M3 tensor with shape ``[M, K]``.
        b: Contiguous E4M3 tensor with shape ``[N, K]``.
        a_scale: Swizzled E8M0 scales for A, one scale per 32 K values.
        b_scale: Swizzled E8M0 scales for B, one scale per 32 K values.
        config: Optional pipeline configuration. Safe FP16-style keys use
            ``BLOCK_SIZE_*`` names; legacy ``BLOCK_*`` aliases are accepted.

    Returns:
        BF16 tensor with shape ``[M, N]``.
    """
    assert a.ndim == 2 and b.ndim == 2, "A and B must be matrices"
    assert a.dtype == torch.float8_e4m3fn, "A must use E4M3 data"
    assert b.dtype == torch.float8_e4m3fn, "B must use E4M3 data"
    assert a.is_contiguous() and b.is_contiguous(), "A and B must be contiguous"
    assert a.device == b.device, "A and B must be on the same device"
    assert (a_scale.device == a.device and b_scale.device == a.device), "Data and scales must be on the same device"
    assert a_scale.dtype == torch.float8_e8m0fnu
    assert b_scale.dtype == torch.float8_e8m0fnu

    M, K = a.shape
    N, b_k = b.shape
    assert K == b_k, "A and B must have the same K dimension"
    assert M % 128 == 0 and N % 128 == 0, "M and N must be multiples of 128"
    assert K % 128 == 0, "K must be a multiple of 128"

    a_scale = _reshape_scale(a_scale, M, K)
    b_scale = _reshape_scale(b_scale, N, K)
    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    a_desc, a_scale_desc, b_desc, b_scale_desc, out_desc = _make_descriptors(a, b, a_scale, b_scale, out)

    def alloc_fn(size: int, _alignment: int, _stream: int | None):
        return torch.empty(size, dtype=torch.int8, device=a.device)

    triton.set_allocator(alloc_fn)

    if config is None:
        device_index = a.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        cache_key = (device_index, M, N, K)
        cached_meta = _BEST_CONFIG_CACHE.get(cache_key)
        if cached_meta is not None:
            _launch_with_config(
                a_desc,
                a_scale_desc,
                b_desc,
                b_scale_desc,
                out_desc,
                M,
                N,
                K,
                cached_meta,
            )
            return out

        def grid(META):
            split_k = META.get("SPLIT_K", 1)
            num_ctas = META.get("NUM_CTAS", 1)
            num_pid_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
            num_pid_m = ((num_pid_m + num_ctas - 1) // num_ctas) * num_ctas
            total_tiles = num_pid_m * triton.cdiv(N, META["BLOCK_SIZE_N"]) * split_k
            return (min(_get_num_sms(), total_tiles), )

        workspace_desc = TensorDescriptor(out, out.shape, out.stride(), [1, 1])
        _gemm_mxfp8_ws_kernel[grid](
            a_desc,
            a_scale_desc,
            b_desc,
            b_scale_desc,
            out_desc,
            workspace_desc,
            M,
            N,
            K,
            NUM_SMS=_get_num_sms(),
        )
        best = _gemm_mxfp8_ws_kernel.best_config
        _BEST_CONFIG_CACHE[cache_key] = _config_to_meta(best)
        split_k = best.kwargs.get("SPLIT_K", 1)
        if split_k > 1:
            reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
            _reduce_k_kernel[reduce_grid](
                workspace_desc.base,
                out,
                M,
                N,
                SPLIT_K=split_k,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                REDUCE_OUTPUT_DTYPE=OUTPUT_DTYPE,
            )
        return out

    meta = _normalize_config(config)
    _validate_config(meta, M, N, K)
    _launch_with_config(a_desc, a_scale_desc, b_desc, b_scale_desc, out_desc, M, N, K, meta)
    return out
