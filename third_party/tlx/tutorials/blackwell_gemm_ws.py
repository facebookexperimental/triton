# TLX GEMM kernel optimized for Blackwell Warp Specialization
import functools
import math
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

# Track which (M, N, K) shapes have already printed their heuristic config
_printed_heuristic_configs = set()


# Cached SM count — never changes during program lifetime.
# Calling torch.cuda.get_device_properties() on every matmul() call
# adds measurable overhead that degrades benchmark throughput on fast kernels.
@functools.lru_cache(maxsize=1)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def get_heuristic_config(M, N, K, num_sms=148):
    """
    Select optimal GEMM config based on problem shape characteristics.

    The selection uses shape-characteristic rules (not exact shape matching):
    1. M/N ratio determines tile shape preference
    2. MN tiles vs SM count determines parallelization strategy (Split-K vs data-parallel)
    3. Arithmetic intensity determines pipeline depth

    Args:
        M, N, K: GEMM dimensions (A is MxK, B is KxN, C is MxN)
        num_sms: Number of SMs on the GPU (default 148 for B200)

    Returns:
        dict: Configuration parameters for the TLX GEMM kernel
    """
    MAX_SMEM = 232 * 1024  # 232KB shared memory limit
    MAX_TMEM = 256 * 1024  # 256KB tensor memory limit per SM

    # ==========================================================================
    # Shape-characteristic analysis
    # ==========================================================================
    mn_ratio = M / max(N, 1)
    is_tall_m = mn_ratio > 4  # M much larger than N
    is_tall_n = mn_ratio < 0.25  # N much larger than M

    # Estimate MN tiles with representative tile sizes
    # Use 256x128 for tall-M, 128x256 for tall-N, 256x256 for balanced
    if is_tall_m:
        ref_bm, ref_bn = 256, 128
    elif is_tall_n:
        ref_bm, ref_bn = 128, 256
    else:
        ref_bm, ref_bn = 256, 256

    num_tiles_m = math.ceil(M / ref_bm)
    num_tiles_n = math.ceil(N / ref_bn)
    num_mn_tiles = num_tiles_m * num_tiles_n

    is_gpu_saturated = num_mn_tiles >= num_sms
    is_undersaturated = num_mn_tiles < num_sms

    # ==========================================================================
    # Shape-characteristic config selection
    # ==========================================================================

    # Characteristic 1: Tall-M shapes benefit from 2-CTA B-tile sharing
    # When M >> N, adjacent M-tiles can share B via 2-CTA clusters
    # Use arithmetic intensity to select tile shape, and K size to select BLOCK_K
    if is_tall_m and is_gpu_saturated:
        arithmetic_intensity = K / max(min(M, N), 1)
        # For low arithmetic intensity (memory-bound), use narrower tiles with larger BLOCK_K
        if arithmetic_intensity <= 1.5:
            return {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
                "NUM_SMEM_BUFFERS": 2,
                "NUM_TMEM_BUFFERS": 2,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": 1,
                "NUM_CTAS": 2,
                "SPLIT_K": 1,
                "INTERLEAVE_EPILOGUE": 1,
                "ctas_per_cga": (2, 1, 1),
                "pre_hook": matmul_tma_set_block_size_hook,
            }
        else:
            # High arithmetic intensity: use wider tiles
            # For large K, use BLOCK_K=128 to reduce K-iterations
            # For smaller K, use BLOCK_K=64 with more SMEM buffers
            if K > N * 2:
                return {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
                    "NUM_SMEM_BUFFERS": 2,
                    "NUM_TMEM_BUFFERS": 1,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 4,
                    "NUM_CTAS": 2,
                    "SPLIT_K": 1,
                    "INTERLEAVE_EPILOGUE": 0,
                    "ctas_per_cga": (2, 1, 1),
                    "pre_hook": matmul_tma_set_block_size_hook,
                }
            else:
                return {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
                    "NUM_SMEM_BUFFERS": 4,
                    "NUM_TMEM_BUFFERS": 1,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 4,
                    "NUM_CTAS": 2,
                    "SPLIT_K": 1,
                    "INTERLEAVE_EPILOGUE": 1,
                    "ctas_per_cga": (2, 1, 1),
                    "pre_hook": matmul_tma_set_block_size_hook,
                }

    # Characteristic 2: Undersaturated GPU needs Split-K for parallelism
    if is_undersaturated:
        # Use MN product to determine tile size - larger MN benefits from wider tiles
        mn_product = M * N
        is_large_output = mn_product >= 1_000_000  # ~1M elements in output

        if is_large_output:
            block_m, block_n, block_k = 256, 128, 64
            k_tiles = math.ceil(K / block_k)
        else:
            block_m, block_n, block_k = 128, 64, 128
            k_tiles = math.ceil(K / block_k)

        split_k = 1
        # Prefer lower Split-K values that still provide enough parallelism
        for sk in [4, 2, 8]:
            if k_tiles >= sk and k_tiles // sk >= 4:
                split_k = sk
                break
        if split_k > 1:
            if is_large_output:
                # Larger output: wider tiles, more epilogue subtiling, fewer TMEM buffers
                return {
                    "BLOCK_SIZE_M": block_m,
                    "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": block_k,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, block_m),
                    "NUM_SMEM_BUFFERS": 4,
                    "NUM_TMEM_BUFFERS": 2,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 8,
                    "NUM_CTAS": 1,
                    "SPLIT_K": split_k,
                    "INTERLEAVE_EPILOGUE": 1,
                    "ctas_per_cga": None,
                    "pre_hook": matmul_tma_set_block_size_hook,
                }
            else:
                # Smaller output: narrower tiles
                return {
                    "BLOCK_SIZE_M": block_m,
                    "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": block_k,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, block_m),
                    "NUM_SMEM_BUFFERS": 4,
                    "NUM_TMEM_BUFFERS": 3,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 1,
                    "NUM_CTAS": 1,
                    "SPLIT_K": split_k,
                    "INTERLEAVE_EPILOGUE": 1,
                    "ctas_per_cga": None,
                    "pre_hook": matmul_tma_set_block_size_hook,
                }

    # Characteristic 3: GPU-saturated shapes use wide tiles for data reuse
    if is_gpu_saturated:
        return {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
            "NUM_SMEM_BUFFERS": 3,
            "NUM_TMEM_BUFFERS": 1,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": 4,
            "NUM_CTAS": 1,
            "SPLIT_K": 1,
            "INTERLEAVE_EPILOGUE": 1,
            "ctas_per_cga": None,
            "pre_hook": matmul_tma_set_block_size_hook,
        }

    # ==========================================================================
    # Fallback: General wave efficiency heuristic for remaining shapes
    # ==========================================================================

    # Candidate configs: (BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS, NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS, NUM_MMA_GROUPS, EPILOGUE_SUBTILE)
    # Based on autotuning results - best configs use BLOCK_K=128, 2-CTA clusters, and balanced buffers
    candidates = [
        # Best config for tall-M shapes (3159809, 384, 384) - prioritize before square config
        (256, 128, 128, 2, 2, 2, 2, 1),  # Best for (3159809, 384, 384)
        # Best config for large square matrices (8192x8192x8192)
        (256, 256, 64, 1, 3, 1, 2, 4),  # Best for 8192x8192x8192
        # Best config for large-K shapes (1024, 256, 16384) - needs Split-K
        (128, 64, 128, 1, 4, 3, 2, 1),  # Best for (1024, 256, 16384) with Split-K
        # 2-CTA configs with BLOCK_K=128 (best performing from autotuning)
        (256, 128, 64, 2, 5, 2, 2, 4),  # Best for (1152, 1024, 213120)
        (128, 256, 64, 2, 4, 2, 1, 2),  # Good general config
        (256, 64, 128, 2, 5, 2, 2, 4),  # Best for skinny-N shapes
        (128, 64, 128, 2, 5, 2, 2, 1),  # Best for (1152, 1024, 12800)
        # 1-CTA configs
        (256, 64, 128, 1, 5, 2, 2, 8),  # Good for skinny-N
        (128, 256, 64, 1, 3, 2, 1, 2),  # Wide tiles
        (128, 128, 64, 1, 4, 2, 1, 2),  # Square tiles
        (256, 128, 64, 1, 3, 1, 2, 2),  # Tall tiles
        (128, 64, 64, 1, 5, 2, 1, 1),  # Small tiles for small problems
        (64, 128, 64, 1, 5, 2, 1, 1),  # Small tiles, wide
        (64, 64, 64, 1, 6, 2, 1, 1),  # Smallest tiles
    ]

    def estimate_smem(bm, bn, bk, num_ctas, num_smem_buffers, num_mma_groups, epilogue_subtile):
        """Estimate shared memory usage for a config."""
        smem_a = bm * bk * 2 * num_smem_buffers
        smem_b = bk * (bn // num_ctas) * 2 * num_smem_buffers
        smem_epilog = bm * (bn // epilogue_subtile) * 2
        smem_barriers = num_smem_buffers * num_mma_groups * 8 * (2 if num_ctas == 2 else 1)
        return smem_a + smem_b + smem_epilog + smem_barriers

    def estimate_tmem(bm, bn, num_tmem_buffers):
        """Estimate tensor memory usage for a config."""
        # TMEM stores accumulator: BLOCK_M * BLOCK_N * sizeof(float) * num_buffers
        return bm * bn * 4 * num_tmem_buffers

    def compute_wave_score(bm, bn, num_ctas, split_k=1):
        """
        Compute wave efficiency score (lower is better).
        Score = fraction of SMs idle in the last wave.
        """
        ctas_m = (M + bm - 1) // bm
        ctas_n = (N + bn - 1) // bn
        # Round up ctas_m to multiple of num_ctas for cluster alignment
        ctas_m = ((ctas_m + num_ctas - 1) // num_ctas) * num_ctas
        total_ctas = ctas_m * ctas_n * split_k

        if total_ctas == 0:
            return float("inf"), 0, 0

        waves = (total_ctas + num_sms - 1) // num_sms
        fractional_waves = total_ctas / num_sms
        score = waves - fractional_waves  # 0 = perfect, 1 = worst
        return score, total_ctas, waves

    best_config = None
    best_score = float("inf")
    best_waves = float("inf")

    for bm, bn, bk, num_ctas, num_smem_buffers, num_tmem_buffers, num_mma_groups, epilogue_subtile in candidates:
        # Skip if SMEM exceeds limit
        smem = estimate_smem(bm, bn, bk, num_ctas, num_smem_buffers, num_mma_groups, epilogue_subtile)
        if smem > MAX_SMEM:
            continue

        # Skip if TMEM exceeds limit
        tmem = estimate_tmem(bm, bn, num_tmem_buffers)
        if tmem > MAX_TMEM:
            continue

        # Skip if MMA group size is invalid (must be <= 128 for hardware)
        if bm // num_mma_groups > 128:
            continue

        # Skip if tiles are larger than the problem
        if bm > M * 2 or bn > N * 2:
            continue

        # Compute wave efficiency
        score, total_ctas, waves = compute_wave_score(bm, bn, num_ctas)

        # Consider split-K only when MN tiles don't saturate GPU
        # Logic adapted from preprocess_configs
        split_k = 1
        num_tiles_m = math.ceil(M / bm)
        num_tiles_n = math.ceil(N / bn)
        num_mn_tiles = num_tiles_m * num_tiles_n

        if num_mn_tiles < num_sms:
            k_tiles = math.ceil(K / bk)
            # Try split-K values (higher first), each split must have enough K tiles
            for sk in [8, 4, 2]:
                if k_tiles >= sk and k_tiles // sk >= 4:
                    sk_score, sk_ctas, sk_waves = compute_wave_score(bm, bn, num_ctas, sk)
                    if sk_score < score or (sk_score == score and sk_ctas > total_ctas):
                        score, total_ctas, waves, split_k = sk_score, sk_ctas, sk_waves, sk
                    break  # Use the first valid split-K

        # Selection criteria:
        # 1. Prefer lower wave inefficiency score
        # 2. With same score, prefer fewer waves (less overhead)
        # 3. With same waves, prefer larger tiles (less total overhead)
        # 4. Prefer multi-CTA configs for better B-tile sharing
        score_slack = 0.1
        adjusted_score = score

        if (adjusted_score < best_score - score_slack
                or (adjusted_score < best_score + score_slack and waves < best_waves)
                or (adjusted_score < best_score + score_slack and waves == best_waves and num_ctas > 1)):
            best_score = adjusted_score
            best_waves = waves
            best_config = {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": _select_group_size_m(M, N, bm),
                "NUM_SMEM_BUFFERS": num_smem_buffers,
                "NUM_TMEM_BUFFERS": num_tmem_buffers,
                "NUM_MMA_GROUPS": num_mma_groups,
                "EPILOGUE_SUBTILE": epilogue_subtile,
                "NUM_CTAS": num_ctas,
                "SPLIT_K": split_k,
                "INTERLEAVE_EPILOGUE": 0,
                "ctas_per_cga": (num_ctas, 1, 1) if num_ctas > 1 else None,
                "pre_hook": matmul_tma_set_block_size_hook,
            }

    return best_config


def _select_group_size_m(M, N, block_m):
    """
    Select GROUP_SIZE_M based on the golden rule for tile scheduling.

    GROUP_SIZE_M controls how tiles are traversed:
    - GROUP_SIZE_M = 1: Column-major (sweep M first), reuses B tiles
    - GROUP_SIZE_M = large: Row-major (sweep N first), reuses A tiles

    Golden rule:
    - When M >> N: Use small GROUP_SIZE_M to reuse B (smaller dimension)
    - When N >> M: Use large GROUP_SIZE_M to reuse A (smaller dimension)
    - When M ~ N: Use moderate GROUP_SIZE_M for L2 locality
    """
    num_m_tiles = (M + block_m - 1) // block_m
    ratio = M / max(N, 1)

    if ratio > 10:
        # M >> N: sweep M, reuse B
        return 1
    elif ratio < 0.1:
        # N >> M: sweep N, reuse A
        return min(64, num_m_tiles)
    else:
        # Balanced: moderate group size for L2 locality
        return min(8, num_m_tiles)


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": g,
                "NUM_SMEM_BUFFERS": s,
                "NUM_TMEM_BUFFERS": t,
                "NUM_MMA_GROUPS": m,
                "EPILOGUE_SUBTILE": subtile,
                "NUM_CTAS": num_ctas,
                "SPLIT_K": split_k,
                "INTERLEAVE_EPILOGUE": interleave,
                "USE_WARP_BARRIER": uwb,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(num_ctas, 1, 1) if num_ctas > 1 else None,
        )
        for BM in [64, 128, 256]
        for BN in [64, 128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4, 5, 6, 7, 8]
        for t in [1, 2, 3]
        for m in [1, 2]
        for subtile in [1, 2, 4, 8]
        for num_ctas in [1, 2]
        for split_k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 19, 24]  # pruning selects one optimal SPLIT_K per tile group
        for interleave in [0, 1]
        for g in [1, 2, 8, 64]
        for uwb in [False, True]
    ]


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    NUM_MMA_GROUPS = nargs.get("NUM_MMA_GROUPS", 1)
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    BLOCK_N_PER_CTA = BLOCK_N // NUM_CTAS
    # For column-major inputs, TMA descriptor block shape matches the transposed view
    if nargs.get("A_ROW_MAJOR", True):
        nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    else:
        nargs["a_desc"].block_shape = [BLOCK_K, BLOCK_M_SPLIT]
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N_PER_CTA]
    else:
        nargs["b_desc"].block_shape = [BLOCK_N_PER_CTA, BLOCK_K]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    nargs["c_desc"].block_shape = [
        BLOCK_M // NUM_MMA_GROUPS,
        BLOCK_N // EPILOGUE_SUBTILE,
    ]
    SPLIT_K = nargs.get("SPLIT_K", 1)
    if SPLIT_K > 1:
        M = nargs["M"]
        N = nargs["N"]
        workspace = torch.empty((SPLIT_K * M, N), device=nargs["c_desc"].base.device, dtype=nargs["c_desc"].base.dtype)
        nargs["workspace_desc"].base = workspace
        nargs["workspace_desc"].shape = list(workspace.shape)
    else:
        nargs["workspace_desc"].base = nargs["c_desc"].base
        nargs["workspace_desc"].shape = list(nargs["c_desc"].base.shape)
    nargs["workspace_desc"].block_shape = [
        BLOCK_M // NUM_MMA_GROUPS,
        BLOCK_N // EPILOGUE_SUBTILE,
    ]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def preprocess_configs(configs, named_args, **kwargs):
    # Blackwell B200A resource limits
    NUM_SMS = _get_num_sms()
    MAX_SHARED_MEMORY = 232 * 1024  # bytes (232KB)
    MAX_TENSOR_MEMORY = 256 * 1024  # bytes (256KB TMEM per SM)

    MBARRIER_SIZE = 8  # bytes

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
        NUM_CTAS = conf.kwargs["NUM_CTAS"]
        NUM_MMA_GROUPS = conf.kwargs["NUM_MMA_GROUPS"]
        SPLIT_K = conf.kwargs.get("SPLIT_K", 1)
        EPILOGUE_SUBTILE = conf.kwargs["EPILOGUE_SUBTILE"]
        INTERLEAVE_EPILOGUE = conf.kwargs.get("INTERLEAVE_EPILOGUE", 0)
        GROUP_SIZE_M = conf.kwargs["GROUP_SIZE_M"]

        # Filter out invalid config that causes wrong hardware MMA
        if BLOCK_M // NUM_MMA_GROUPS > 128:
            continue
        # Pair-CTA MMA doesn't work with M=64 per MMA group
        if NUM_CTAS == 2 and BLOCK_M // NUM_MMA_GROUPS == 64:
            continue
        # GROUP_SIZE_M must be a multiple of NUM_CTAS so that consecutive
        # tile_ids (assigned to paired CTAs in a cluster) always map to the
        # same pid_n. Otherwise, at group boundaries a CTA pair can straddle
        # two different pid_n values, breaking 2-CTA B-tile sharing.
        if GROUP_SIZE_M % NUM_CTAS != 0:
            continue

        # EPILOGUE_SUBTILE must evenly divide BLOCK_N
        if BLOCK_N % EPILOGUE_SUBTILE != 0:
            continue

        # Interleaved epilogue requires NUM_MMA_GROUPS == 2
        if INTERLEAVE_EPILOGUE and NUM_MMA_GROUPS != 2:
            continue

        # Blackwell MMA requires BLOCK_M_SPLIT >= 64
        if BLOCK_M // NUM_MMA_GROUPS < 64:
            continue

        num_tiles_m = math.ceil(M / BLOCK_M)
        num_tiles_n = math.ceil(N / BLOCK_N)
        num_mn_tiles = num_tiles_m * num_tiles_n

        # BM=64 tiles help unsaturated shapes by providing more spatial tiles.
        # Skip them when the shape is already GPU-saturated with 128-tiles.
        if BLOCK_M == 64 and math.ceil(M / 128) * math.ceil(N / 128) >= NUM_SMS:
            continue

        # --- Split-K gating: only allow SPLIT_K > 1 for small shapes ---
        # Split-K helps when MN tiles are too few to saturate the GPU.
        # For large shapes with plenty of MN tiles, SPLIT_K=1 is better
        # since it avoids the atomic reduction overhead.
        if SPLIT_K > 1:
            if num_mn_tiles >= NUM_SMS:
                continue
            k_tiles = math.ceil(K / BLOCK_K)
            if k_tiles < SPLIT_K:
                continue
            # Reject SK values where cdiv overallocation leaves the last split empty
            # (causes deadlock: producer loop is empty but MMA consumer waits on barrier)
            k_tiles_per_split = math.ceil(k_tiles / SPLIT_K)
            if k_tiles_per_split * (SPLIT_K - 1) >= k_tiles:
                continue
            # Each split must have enough K tiles to be worthwhile
            if k_tiles // SPLIT_K < 4:
                continue

        # --- Shared Memory estimation ---
        smem_a = BLOCK_M * BLOCK_K * 2 * NUM_SMEM_BUFFERS
        smem_b_size = BLOCK_N // NUM_CTAS
        smem_b = BLOCK_K * smem_b_size * 2 * NUM_SMEM_BUFFERS
        smem_epilog = BLOCK_M * (BLOCK_N // EPILOGUE_SUBTILE) * 2
        smem_barriers = NUM_SMEM_BUFFERS * NUM_MMA_GROUPS * MBARRIER_SIZE
        if NUM_CTAS == 2:
            smem_barriers += NUM_SMEM_BUFFERS * NUM_MMA_GROUPS * MBARRIER_SIZE
        smem_barriers += NUM_TMEM_BUFFERS

        total_smem = smem_a + smem_b + smem_epilog + smem_barriers
        if total_smem > MAX_SHARED_MEMORY:
            continue

        # --- Tensor Memory (TMEM) estimation ---
        total_tmem = BLOCK_M * BLOCK_N * 4 * NUM_TMEM_BUFFERS
        if total_tmem > MAX_TENSOR_MEMORY:
            continue

        pruned_configs.append(conf)

    # Two-level SPLIT_K filter (per tile-size group):
    #   1. Minimize wave count (fewer waves = less wall-clock time).
    #   2. Within the same wave count, maximize SPLIT_K (more K-parallelism
    #      across SMs). E.g. with 148 SMs and 40 base tiles: SPLIT_K=3
    #      gives 120 tiles (120 SMs active, each does K/3 work) vs SPLIT_K=1
    #      giving 40 tiles (40 SMs active, each does K/1 work) — both 1 wave,
    #      but SPLIT_K=3 is faster because work is spread across more SMs.
    # Applied per (BM, BN, BK) group because different tile sizes have
    # vastly different compute characteristics.
    # Note: for saturated shapes, SPLIT_K>1 configs are already pruned by
    # the base_tiles >= NUM_SMS gate above, so only SPLIT_K=1 survives.
    if pruned_configs:

        def _total_tiles(c):
            return (math.ceil(M / c.kwargs["BLOCK_SIZE_M"]) * math.ceil(N / c.kwargs["BLOCK_SIZE_N"]) *
                    c.kwargs.get("SPLIT_K", 1))

        def _num_waves(c):
            return math.ceil(_total_tiles(c) / NUM_SMS)

        def _tile_key(c):
            return (c.kwargs["BLOCK_SIZE_M"], c.kwargs["BLOCK_SIZE_N"], c.kwargs["BLOCK_SIZE_K"])

        # Group by tile size
        tile_groups = {}
        for c in pruned_configs:
            tile_groups.setdefault(_tile_key(c), []).append(c)

        result = []
        for group_configs in tile_groups.values():
            min_waves = min(_num_waves(c) for c in group_configs)
            best = [c for c in group_configs if _num_waves(c) == min_waves]
            max_sk = max(c.kwargs.get("SPLIT_K", 1) for c in best)
            best = [c for c in best if c.kwargs.get("SPLIT_K", 1) == max_sk]
            result.extend(best)

        pruned_configs = result

    # --- Golden Rule: sweep the large dimension, fix the small one ---
    # A[M,K] changes with M; B[K,N] changes with N.
    # GROUP_SIZE_M controls how many M-tiles are grouped before advancing N.
    #   GROUP_SIZE_M = 1  → sweep M first (column-major), B (small-N side) reused
    #   GROUP_SIZE_M = large → sweep N first (row-major), A (small-M side) reused
    # When M >> N: prefer small GROUP_SIZE_M (sweep M, fix B for reuse)
    # When N >> M: prefer large GROUP_SIZE_M (sweep N, fix A for reuse)
    if pruned_configs:
        IMBALANCE_THRESHOLD = 10  # ratio at which we enforce the rule
        if M > N * IMBALANCE_THRESHOLD:
            # M >> N: keep the smallest GROUP_SIZE_M to sweep M. 2-CTA requires
            # GROUP_SIZE_M to be a multiple of NUM_CTAS (see the pairing gate
            # above), so its smallest valid value is NUM_CTAS (2), not 1 —
            # forcing 1 here would silently prune every 2-CTA config.
            pruned_configs = [c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] == c.kwargs["NUM_CTAS"]]
        elif N > M * IMBALANCE_THRESHOLD:
            # N >> M: keep only large GROUP_SIZE_M to sweep N
            pruned_configs = [c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] >= 32]
        else:
            # Balanced M ≈ N: keep moderate GROUP_SIZE_M for L2 locality
            pruned_configs = [c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] == 8]

    # Pareto-optimal filtering on (NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS,
    # NUM_MMA_GROUPS): these are independent resource dimensions where more
    # buffers / groups generally means better pipelining, but no single
    # dimension dominates the others.  Keep a config unless another config
    # in the same (BM, BN, BK, SUBTILE, NUM_CTAS, SPLIT_K) group dominates
    # it (>= in all dimensions, > in at least one).
    if pruned_configs:

        def _group_key(c):
            return (
                c.kwargs["BLOCK_SIZE_M"],
                c.kwargs["BLOCK_SIZE_N"],
                c.kwargs["BLOCK_SIZE_K"],
                c.kwargs["EPILOGUE_SUBTILE"],
                c.kwargs["NUM_CTAS"],
                c.kwargs.get("SPLIT_K", 1),
                c.kwargs.get("INTERLEAVE_EPILOGUE", 0),
            )

        def _val(c):
            return (
                c.kwargs["NUM_SMEM_BUFFERS"],
                c.kwargs["NUM_TMEM_BUFFERS"],
                c.kwargs["NUM_MMA_GROUPS"],
            )

        def _dominates(a, b):
            """Return True if a dominates b (>= in all, > in at least one)."""
            va, vb = _val(a), _val(b)
            return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))

        groups = {}
        for c in pruned_configs:
            groups.setdefault(_group_key(c), []).append(c)

        pruned_configs = []
        for members in groups.values():
            for c in members:
                if not any(_dominates(other, c) for other in members if other is not c):
                    pruned_configs.append(c)

    return pruned_configs


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
):
    """Compute common grid information used across async tasks."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Pad num_pid_m to multiple of NUM_CTAS so CTA clusters tile evenly along M.
    num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_mn_tiles = num_pid_m * num_pid_n
    num_tiles = num_mn_tiles * SPLIT_K
    k_tiles_total = tl.cdiv(K, BLOCK_SIZE_K)
    return start_pid, num_pid_m, num_pid_n, num_pid_in_group, num_mn_tiles, num_tiles, k_tiles_total


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
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
    SPLIT_K,
    INTERLEAVE_EPILOGUE,
    c_desc,
    workspace_desc,
    c_smem_buffers,
    tmem_buffers,
    tmem_full_bars,
    tmem_empty_bars,
    cur_tmem_buf,
    tmem_read_phase,
    NUM_CTAS: tl.constexpr,
):
    """Process epilogue for a single tile."""
    mn_tile_id = tile_id if SPLIT_K == 1 else tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS

    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    if SPLIT_K > 1:
        split_id = tile_id // num_mn_tiles
        out_desc = workspace_desc
        row_base = split_id * M
    else:
        out_desc = c_desc
        row_base = 0

    if INTERLEAVE_EPILOGUE:
        # Interleaved TMA stores across two groups to improve memory throughput.
        # Pattern: wait g0, store g0s0, wait g1, store g1s0,
        #          then alternate g0/g1 for slices 1-3.
        buf_idx_0 = 0 * NUM_TMEM_BUFFERS + cur_tmem_buf
        buf_idx_1 = 1 * NUM_TMEM_BUFFERS + cur_tmem_buf
        acc_tmem_0 = tmem_buffers[buf_idx_0]
        acc_tmem_1 = tmem_buffers[buf_idx_1]
        offs_am_0 = pid_m * BLOCK_SIZE_M + 0 * BLOCK_M_SPLIT
        offs_am_1 = pid_m * BLOCK_SIZE_M + 1 * BLOCK_M_SPLIT

        # --- Wait for group 0, store group 0 slice 0 ---
        tlx.barrier_wait(tmem_full_bars[buf_idx_0], tmem_read_phase)
        acc_sub = tlx.local_slice(acc_tmem_0, [0, 0 * slice_size], [BLOCK_M_SPLIT, slice_size])
        result = tlx.local_load(acc_sub)
        if NUM_CTAS == 2:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1, remote_cta_rank=0)
        else:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1)
        c = result.to(tlx.dtype_of(out_desc))
        c_smem = c_smem_buffers[0]
        tlx.local_store(c_smem, c)
        tlx.async_descriptor_store(
            out_desc,
            c_smem,
            [row_base + offs_am_0, offs_bn + 0 * slice_size],
            eviction_policy="evict_first",
        )

        # --- Wait for group 1, store group 1 slice 0 ---
        tlx.barrier_wait(tmem_full_bars[buf_idx_1], tmem_read_phase)
        acc_sub = tlx.local_slice(acc_tmem_1, [0, 0 * slice_size], [BLOCK_M_SPLIT, slice_size])
        result = tlx.local_load(acc_sub)
        if NUM_CTAS == 2:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1, remote_cta_rank=0)
        else:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1)
        c = result.to(tlx.dtype_of(out_desc))
        c_smem = c_smem_buffers[1]
        tlx.local_store(c_smem, c)
        tlx.async_descriptor_store(
            out_desc,
            c_smem,
            [row_base + offs_am_1, offs_bn + 0 * slice_size],
            eviction_policy="evict_first",
        )

        # --- Slices 1-3: alternate group 0, group 1 ---
        for slice_id in tl.static_range(1, EPILOGUE_SUBTILE):
            # Group 0
            acc_sub = tlx.local_slice(acc_tmem_0, [0, slice_id * slice_size], [BLOCK_M_SPLIT, slice_size])
            result = tlx.local_load(acc_sub)
            if NUM_CTAS == 2:
                tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1)
            c = result.to(tlx.dtype_of(out_desc))
            c_smem = c_smem_buffers[0]
            tlx.async_descriptor_store_wait(1)
            tlx.local_store(c_smem, c)
            tlx.async_descriptor_store(
                out_desc,
                c_smem,
                [row_base + offs_am_0, offs_bn + slice_id * slice_size],
                eviction_policy="evict_first",
            )

            # Group 1
            acc_sub = tlx.local_slice(acc_tmem_1, [0, slice_id * slice_size], [BLOCK_M_SPLIT, slice_size])
            result = tlx.local_load(acc_sub)
            if NUM_CTAS == 2:
                tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1)
            c = result.to(tlx.dtype_of(out_desc))
            c_smem = c_smem_buffers[1]
            tlx.async_descriptor_store_wait(1)
            tlx.local_store(c_smem, c)
            tlx.async_descriptor_store(
                out_desc,
                c_smem,
                [row_base + offs_am_1, offs_bn + slice_id * slice_size],
                eviction_policy="evict_first",
            )
    else:
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            # Wait for TMEM to be filled
            buf_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

            tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_read_phase)

            # load the result from TMEM to registers
            acc_tmem = tmem_buffers[buf_idx]
            offs_am = pid_m * BLOCK_SIZE_M + group_id * BLOCK_M_SPLIT
            for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                acc_tmem_subslice = tlx.local_slice(
                    acc_tmem,
                    [0, slice_id * slice_size],
                    [BLOCK_M_SPLIT, slice_size],
                )
                result = tlx.local_load(acc_tmem_subslice)
                if NUM_CTAS == 2:
                    tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1, remote_cta_rank=0)
                else:
                    tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
                c = result.to(tlx.dtype_of(out_desc))
                c_smem = c_smem_buffers[(group_id * EPILOGUE_SUBTILE + slice_id) % 2]
                tlx.async_descriptor_store_wait(1)
                tlx.local_store(c_smem, c)
                tlx.async_descriptor_store(
                    out_desc,
                    c_smem,
                    [row_base + offs_am, offs_bn + slice_id * slice_size],
                    eviction_policy="evict_first",
                )

    # Wait for all TMA stores to complete
    tlx.async_descriptor_store_wait(0)


@triton.jit
def _process_tile_mma_inner(
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
    buffers_A,
    buffers_B,
    tmem_buffers,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    tmem_full_bars,
    cur_tmem_buf,
    tmem_empty_bars,
    tmem_write_phase,
    smem_accum_cnt,
    NUM_CTAS,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
):
    """Process MMA for a single tile over [k_tile_start, k_tile_end). Returns updated smem_accum_cnt."""
    local_k_tiles = k_tile_end - k_tile_start

    # Peeled first K-iteration
    buf, phase = get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

    if NUM_MMA_GROUPS == 1:
        tlx.barrier_wait(tmem_empty_bars[cur_tmem_buf], tmem_write_phase ^ 1)

    # In the single-group path B_smem_full_bars aliases A_smem_full_bars, so
    # this one wait covers both TMA loads. Multi-group waits on B here and each
    # A subtile below.
    tlx.barrier_wait(B_smem_full_bars[buf], phase)

    # Process first K iteration (peeled) with use_acc=False
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        # Calculate buffer indices
        a_buf = group_id * NUM_SMEM_BUFFERS + buf
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

        if NUM_MMA_GROUPS > 1:
            # Wait for this A subtile buffer to be loaded.
            tlx.barrier_wait(A_smem_full_bars[a_buf], phase)

            cur_barrier_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
            tlx.barrier_wait(tmem_empty_bars[cur_barrier_idx], tmem_write_phase ^ 1)

        # Transpose SMEM buffers if inputs were column-major
        a_operand = tlx.local_trans(buffers_A[a_buf]) if not A_ROW_MAJOR else buffers_A[a_buf]
        b_operand = tlx.local_trans(buffers_B[buf]) if not B_ROW_MAJOR else buffers_B[buf]

        # Perform MMA: use_acc=False for first K iteration (clears accumulator).
        # This helper only runs on CTA0 in 2-CTA mode.
        tlx.async_dot(
            a_operand,
            b_operand,
            tmem_buffers[acc_buf],
            use_acc=False,
            mBarriers=[A_smem_empty_bars[a_buf]],
            two_ctas=NUM_CTAS == 2,
            out_dtype=tl.float32,
        )

    smem_accum_cnt += 1

    # Remaining K iterations with use_acc=True
    for _ in range(1, local_k_tiles):
        # Advance the ring buffer incrementally (avoids the non-power-of-2
        # divide/modulo of get_bufidx_phase in the hot K-loop).
        buf += 1
        if buf == NUM_SMEM_BUFFERS:
            buf = 0
            phase ^= 1

        # In the single-group path this aliases the combined A/B full barrier.
        tlx.barrier_wait(B_smem_full_bars[buf], phase)

        # Process all subtiles for this K iteration
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            # Calculate buffer indices
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

            if NUM_MMA_GROUPS > 1:
                # Wait for this A subtile buffer to be loaded.
                tlx.barrier_wait(A_smem_full_bars[a_buf], phase)

            # Transpose SMEM buffers if inputs were column-major
            a_operand = tlx.local_trans(buffers_A[a_buf]) if not A_ROW_MAJOR else buffers_A[a_buf]
            b_operand = tlx.local_trans(buffers_B[buf]) if not B_ROW_MAJOR else buffers_B[buf]

            # Perform MMA: use_acc=True for remaining K iterations.
            tlx.async_dot(
                a_operand,
                b_operand,
                tmem_buffers[acc_buf],
                use_acc=True,
                mBarriers=[A_smem_empty_bars[a_buf]],
                two_ctas=NUM_CTAS == 2,
                out_dtype=tl.float32,
            )

        smem_accum_cnt += 1

    # Signal the epilogue when the MMAs complete via an async tcgen05 commit,
    # instead of blocking on the last MMA's A_smem_empty and then arriving on
    # tmem_full. The commit makes tmem_full track completion of the prior
    # tcgen05 MMAs asynchronously, so the MMA warpgroup can return and start
    # the next tile's MMAs while this tile's final MMAs still drain -- closing
    # the per-tile pipeline bubble between consecutive K-loops. In 2-CTA mode
    # the commit multicasts the mbarrier signal to both CTAs' tmem_full (one
    # arrive each), matching the previous local + remote_cta_rank arrives.
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        tlx.tcgen05_commit(tmem_full_bars[acc_buf], two_ctas=NUM_CTAS == 2)

    return smem_accum_cnt


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
    NUM_MMA_GROUPS,
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    a_desc,
    b_desc,
    buffers_A,
    buffers_B,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    smem_accum_cnt,
    NUM_CTAS,
    cluster_cta_rank,
    SPLIT_K: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
):
    """Process TMA loads for a single tile with all subtiles over [k_tile_start, k_tile_end)."""
    mn_tile_id = tile_id if SPLIT_K == 1 else tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    dsize: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // NUM_CTAS)
    expected_bytes: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K // NUM_CTAS
    leader_only_mma: tl.constexpr = NUM_CTAS == 2
    is_leader = cluster_cta_rank == 0

    local_k_tiles = k_tile_end - k_tile_start

    # Ring-buffer index tracked incrementally (avoids the non-power-of-2
    # divide/modulo of get_bufidx_phase in the hot K-loop).
    buf, phase = get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
    # Iterate along K dimension for this split's range
    for k_idx in range(0, local_k_tiles):
        k = k_tile_start + k_idx
        offs_k = k * BLOCK_SIZE_K

        offs_am = pid_m * BLOCK_SIZE_M
        if NUM_MMA_GROUPS == 1:
            tlx.barrier_wait(A_smem_empty_bars[buf], phase ^ 1)
            # Both TMA loads contribute bytes to the same full barrier.
            a_expected_bytes: tl.constexpr = dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
            combined_bytes: tl.constexpr = expected_bytes + a_expected_bytes
            if leader_only_mma:
                tlx.barrier_expect_bytes(A_smem_full_bars[buf], combined_bytes * NUM_CTAS, pred=is_leader)
            else:
                tlx.barrier_expect_bytes(A_smem_full_bars[buf], combined_bytes)
            if not B_ROW_MAJOR:
                tlx.async_descriptor_load(b_desc, buffers_B[buf], [offs_bn, offs_k], A_smem_full_bars[buf],
                                          eviction_policy="evict_last", two_ctas=leader_only_mma)
            else:
                tlx.async_descriptor_load(b_desc, buffers_B[buf], [offs_k, offs_bn], A_smem_full_bars[buf],
                                          eviction_policy="evict_last", two_ctas=leader_only_mma)

            if not A_ROW_MAJOR:
                tlx.async_descriptor_load(a_desc, buffers_A[buf], [offs_k, offs_am], A_smem_full_bars[buf],
                                          eviction_policy="evict_last", two_ctas=leader_only_mma)
            else:
                tlx.async_descriptor_load(a_desc, buffers_A[buf], [offs_am, offs_k], A_smem_full_bars[buf],
                                          eviction_policy="evict_last", two_ctas=leader_only_mma)
        else:
            a0_buf = buf
            tlx.barrier_wait(A_smem_empty_bars[a0_buf], phase ^ 1)
            a_expected_bytes: tl.constexpr = dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
            if leader_only_mma:
                tlx.barrier_expect_bytes(A_smem_full_bars[a0_buf], a_expected_bytes * NUM_CTAS, pred=is_leader)
            else:
                tlx.barrier_expect_bytes(A_smem_full_bars[a0_buf], a_expected_bytes)
            if not A_ROW_MAJOR:
                tlx.async_descriptor_load(a_desc, buffers_A[a0_buf], [offs_k, offs_am], A_smem_full_bars[a0_buf],
                                          eviction_policy="evict_last", two_ctas=leader_only_mma)
            else:
                tlx.async_descriptor_load(a_desc, buffers_A[a0_buf], [offs_am, offs_k], A_smem_full_bars[a0_buf],
                                          eviction_policy="evict_last", two_ctas=leader_only_mma)

            a1_buf = NUM_SMEM_BUFFERS + buf
            tlx.barrier_wait(A_smem_empty_bars[a1_buf], phase ^ 1)
            if leader_only_mma:
                tlx.barrier_expect_bytes(B_smem_full_bars[buf], expected_bytes * NUM_CTAS, pred=is_leader)
            else:
                tlx.barrier_expect_bytes(B_smem_full_bars[buf], expected_bytes)
            if not B_ROW_MAJOR:
                tlx.async_descriptor_load(b_desc, buffers_B[buf], [offs_bn, offs_k], B_smem_full_bars[buf],
                                          eviction_policy="evict_last", two_ctas=leader_only_mma)
            else:
                tlx.async_descriptor_load(b_desc, buffers_B[buf], [offs_k, offs_bn], B_smem_full_bars[buf],
                                          eviction_policy="evict_last", two_ctas=leader_only_mma)

            if leader_only_mma:
                tlx.barrier_expect_bytes(A_smem_full_bars[a1_buf], a_expected_bytes * NUM_CTAS, pred=is_leader)
            else:
                tlx.barrier_expect_bytes(A_smem_full_bars[a1_buf], a_expected_bytes)
            if not A_ROW_MAJOR:
                tlx.async_descriptor_load(
                    a_desc,
                    buffers_A[a1_buf],
                    [offs_k, offs_am + BLOCK_M_SPLIT],
                    A_smem_full_bars[a1_buf],
                    eviction_policy="evict_last",
                    two_ctas=leader_only_mma,
                )
            else:
                tlx.async_descriptor_load(
                    a_desc,
                    buffers_A[a1_buf],
                    [offs_am + BLOCK_M_SPLIT, offs_k],
                    A_smem_full_bars[a1_buf],
                    eviction_policy="evict_last",
                    two_ctas=leader_only_mma,
                )

        smem_accum_cnt += 1
        buf += 1
        if buf == NUM_SMEM_BUFFERS:
            buf = 0
            phase ^= 1

    return smem_accum_cnt


TORCH_DTYPE_TO_TRITON = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.jit
def _reduce_k_kernel(
    workspace_ptr,
    c_ptr,
    M,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        ws_offs = base_offs + s * M * N
        partial = tl.load(workspace_ptr + ws_offs, mask=mask, other=0.0)
        acc += partial.to(tl.float32)

    tl.store(c_ptr + base_offs, acc.to(OUTPUT_DTYPE), mask=mask)


_l2_flush_buf = {}


def _flush_l2_cache(device):
    """Evict L2 cache by writing a buffer larger than L2 (64 MB for B200)."""
    if device not in _l2_flush_buf:
        _l2_flush_buf[device] = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device=device)
    _l2_flush_buf[device].zero_()


def reduce_post_hook(nargs, exception=None):
    if exception is not None:
        return
    split_k = nargs.get("SPLIT_K", 1)
    if split_k > 1:
        M = nargs["M"]
        N = nargs["N"]
        workspace = nargs["workspace_desc"].base
        c = nargs["c_desc"].base
        reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
        # num_warps=4 is the default; passing it as a meta-kwarg forces the C
        # dispatch proxy to fall back to the slow launch path (specialize.cc
        # bails on non-param kwargs). Param kwargs below are merged in C and stay
        # fast, so keep them but drop the redundant num_warps.
        _reduce_k_kernel[reduce_grid](
            workspace,
            c,
            M,
            N,
            SPLIT_K=split_k,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[workspace.dtype],
        )


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": preprocess_configs},
    post_hook=reduce_post_hook,
)
@triton.jit
def matmul_kernel_tma_ws_blackwell(
    a_desc,
    b_desc,
    c_desc,
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
    INTERLEAVE_EPILOGUE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_CLC_STAGES: tl.constexpr = 1,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
    USE_WARP_BARRIER: tl.constexpr = False,
):
    # allocate NUM_SMEM_BUFFERS buffers
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    if not A_ROW_MAJOR:
        buffers_A = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_M_SPLIT),
            tlx.dtype_of(a_desc),
            NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        )
    else:
        buffers_A = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_SIZE_K),
            tlx.dtype_of(a_desc),
            NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        )
    # In 2-CTA mode, each CTA only needs to load BLOCK_N // NUM_CTAS of B.
    if not B_ROW_MAJOR:
        buffers_B = tlx.local_alloc((BLOCK_SIZE_N // NUM_CTAS, BLOCK_SIZE_K), tlx.dtype_of(b_desc), NUM_SMEM_BUFFERS)
    else:
        buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N // NUM_CTAS), tlx.dtype_of(b_desc), NUM_SMEM_BUFFERS)
    # NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    # Each buffer holds one subtile: BLOCK_M_SPLIT x BLOCK_SIZE_N
    # Total buffers: NUM_TMEM_BUFFERS * NUM_MMA_GROUPS
    tmem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )

    # Allocate SMEM buffers for epilogue TMA store (at least 2 for multi-buffering)
    NUM_EPILOGUE_SMEM_BUFFERS: tl.constexpr = NUM_MMA_GROUPS if NUM_MMA_GROUPS > 2 else 2
    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    c_smem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, slice_size),
        tlx.dtype_of(c_desc),
        NUM_EPILOGUE_SMEM_BUFFERS,
    )

    # CTA pairs are placed along M dim
    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
    else:
        cluster_cta_rank = 0

    # allocate barriers - each subtile needs its own barriers
    # NUM_SMEM_BUFFERS barriers per subtile for synchronization
    A_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    A_smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    if NUM_MMA_GROUPS == 1:
        # A and B TMA loads share the same transaction-byte-counted barrier.
        B_smem_full_bars = A_smem_full_bars
    else:
        B_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    # NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    tmem_empty_arrivals: tl.constexpr = EPILOGUE_SUBTILE * NUM_CTAS
    if USE_WARP_BARRIER:
        tmem_empty_bars = tlx.alloc_warp_barrier(num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, num_warps=4,
                                                 num_arrivals=tmem_empty_arrivals)
    else:
        tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
                                             arrive_count=tmem_empty_arrivals)

    # Each of the three async tasks consumes CLC responses on both cluster CTAs.
    clc_context = tlx.clc_create_context(num_consumers=3 * NUM_CTAS, num_stages=NUM_CLC_STAGES)

    with tlx.async_tasks(
            exclusive=True,
            no_ending_cluster_sync=True,
            mbarrier_try_wait_suspend_ns=50000,
    ):
        with tlx.async_task("default"):  # epilogue consumer
            (
                start_pid,
                num_pid_m,
                num_pid_n,
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
            )

            tmem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_producer = 1
            clc_phase_consumer = 0

            while tile_id != -1:
                # Prefetch the next dynamically claimed tile while processing
                # the current tile to cover the persistent-loop boundary.
                tlx.clc_producer(clc_context, clc_phase_producer, multi_ctas=NUM_CTAS == 2)
                clc_phase_producer ^= 1
                if SPLIT_K == 1:
                    # Fast path: one split covers all K-tiles; avoids the
                    # runtime-divisor `tile_id // num_mn_tiles` division.
                    k_tile_start = 0
                    k_tile_end = k_tiles_total
                else:
                    split_id = tile_id // num_mn_tiles
                    k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                    k_tile_start = split_id * k_tiles_per_split
                    k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)
                if SPLIT_K == 1 or k_tile_end > k_tile_start:
                    cur_tmem_buf, tmem_read_phase = get_bufidx_phase(tmem_accum_cnt, NUM_TMEM_BUFFERS)
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
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                        SPLIT_K=SPLIT_K,
                        INTERLEAVE_EPILOGUE=INTERLEAVE_EPILOGUE,
                        c_desc=c_desc,
                        workspace_desc=workspace_desc,
                        c_smem_buffers=c_smem_buffers,
                        tmem_buffers=tmem_buffers,
                        tmem_full_bars=tmem_full_bars,
                        tmem_empty_bars=tmem_empty_bars,
                        cur_tmem_buf=cur_tmem_buf,
                        tmem_read_phase=tmem_read_phase,
                        NUM_CTAS=NUM_CTAS,
                    )
                    tmem_accum_cnt += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=NUM_CTAS == 2)
                clc_phase_consumer ^= 1

        with tlx.async_task(num_warps=1, num_regs=24):  # MMA consumer
            (
                start_pid,
                num_pid_m,
                num_pid_n,
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
            )

            tmem_accum_cnt = 0
            smem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_consumer = 0

            while tile_id != -1:
                # Compute K range for this split
                if SPLIT_K == 1:
                    # Fast path: one split covers all K-tiles; avoids the
                    # runtime-divisor `tile_id // num_mn_tiles` division.
                    k_tile_start = 0
                    k_tile_end = k_tiles_total
                else:
                    split_id = tile_id // num_mn_tiles
                    k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                    k_tile_start = split_id * k_tiles_per_split
                    k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)

                # Skip tiles whose split has zero K-tiles
                if SPLIT_K == 1 or k_tile_end > k_tile_start:
                    # In 2-CTA mode, only CTA0 issues the collaborative MMA. CTA1's
                    # producer still loads its local operand slice, and cta_group::2
                    # TMA completion routes readiness to CTA0's barriers. Skipping
                    # the inner MMA loop on CTA1 avoids duplicating its address,
                    # predicate, and wait bookkeeping.
                    if NUM_CTAS == 1 or cluster_cta_rank == 0:
                        cur_tmem_buf, tmem_write_phase = get_bufidx_phase(tmem_accum_cnt, NUM_TMEM_BUFFERS)
                        smem_accum_cnt = _process_tile_mma_inner(
                            k_tile_start=k_tile_start,
                            k_tile_end=k_tile_end,
                            NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                            NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                            NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                            buffers_A=buffers_A,
                            buffers_B=buffers_B,
                            tmem_buffers=tmem_buffers,
                            A_smem_full_bars=A_smem_full_bars,
                            B_smem_full_bars=B_smem_full_bars,
                            A_smem_empty_bars=A_smem_empty_bars,
                            tmem_full_bars=tmem_full_bars,
                            cur_tmem_buf=cur_tmem_buf,
                            tmem_empty_bars=tmem_empty_bars,
                            tmem_write_phase=tmem_write_phase,
                            smem_accum_cnt=smem_accum_cnt,
                            NUM_CTAS=NUM_CTAS,
                            A_ROW_MAJOR=A_ROW_MAJOR,
                            B_ROW_MAJOR=B_ROW_MAJOR,
                        )
                    else:
                        # Keep CTA1's ring-buffer position aligned with CTA0 so
                        # subsequent producer/CLC iterations derive identical
                        # buffer indices and phases, without executing MMA work.
                        smem_accum_cnt += k_tile_end - k_tile_start
                    # Both CTAs advance the logical TMEM tile counter because
                    # their epilogue and CLC tasks consume the same tile stream.
                    tmem_accum_cnt += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=NUM_CTAS == 2)
                clc_phase_consumer ^= 1

        with tlx.async_task(num_warps=1, num_regs=24):  # producer, TMA load
            (
                start_pid,
                num_pid_m,
                num_pid_n,
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
            )

            smem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_consumer = 0

            while tile_id != -1:
                # Compute K range for this split
                if SPLIT_K == 1:
                    # Fast path: one split covers all K-tiles; avoids the
                    # runtime-divisor `tile_id // num_mn_tiles` division.
                    k_tile_start = 0
                    k_tile_end = k_tiles_total
                else:
                    split_id = tile_id // num_mn_tiles
                    k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                    k_tile_start = split_id * k_tiles_per_split
                    k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)

                # Skip tiles whose split has zero K-tiles
                if SPLIT_K == 1 or k_tile_end > k_tile_start:
                    smem_accum_cnt = _process_tile_producer_inner(
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        num_mn_tiles=num_mn_tiles,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        k_tile_start=k_tile_start,
                        k_tile_end=k_tile_end,
                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                        a_desc=a_desc,
                        b_desc=b_desc,
                        buffers_A=buffers_A,
                        buffers_B=buffers_B,
                        A_smem_full_bars=A_smem_full_bars,
                        B_smem_full_bars=B_smem_full_bars,
                        A_smem_empty_bars=A_smem_empty_bars,
                        smem_accum_cnt=smem_accum_cnt,
                        NUM_CTAS=NUM_CTAS,
                        cluster_cta_rank=cluster_cta_rank,
                        SPLIT_K=SPLIT_K,
                        A_ROW_MAJOR=A_ROW_MAJOR,
                        B_ROW_MAJOR=B_ROW_MAJOR,
                    )
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=NUM_CTAS == 2)
                clc_phase_consumer ^= 1


def matmul(a, b, config=None):
    """Matrix multiplication using TLX GEMM kernel.

    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
        config: Optional dict with kernel config. If None and
                TLX_GEMM_USE_HEURISTIC=1, uses shape-dependent heuristic
                selection. If heuristic fails, falls back to full autotuning.

    Returns:
        Output matrix C of shape (M, N)
    """
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Detect column-major inputs.
    # A column-major (M, K) tensor has strides (1, M); its .T is row-major (K, M).
    a_row_major = a.is_contiguous()
    b_row_major = b.is_contiguous()

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    if not a_row_major:
        a_t = a.T  # (K, M) with strides (M, 1) — row-major
        a_desc = TensorDescriptor(a_t, a_t.shape, a_t.stride(), dummy_block)
    else:
        a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    if not b_row_major:
        b_t = b.T  # (N, K) with strides (K, 1) — row-major
        b_desc = TensorDescriptor(b_t, b_t.shape, b_t.stride(), dummy_block)
    else:
        b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = _get_num_sms()

    # Use heuristic config if no config provided and env var is set
    use_heuristic = os.environ.get("TLX_GEMM_USE_HEURISTIC", "0") == "1"
    if config is None and use_heuristic:
        config = get_heuristic_config(M, N, K, NUM_SMS)
        if config is not None and os.environ.get("TRITON_PRINT_AUTOTUNING") == "1":
            shape_key = (M, N, K)
            if shape_key not in _printed_heuristic_configs:
                _printed_heuristic_configs.add(shape_key)
                config_str = ", ".join(f"{k}: {v}" for k, v in config.items() if k not in ("pre_hook", "ctas_per_cga"))
                print(f"heuristic config selected: {config_str};")

    if config is not None:
        # Extract ctas_per_cga before removing - we need it for cluster launch
        ctas_per_cga = config.pop("ctas_per_cga", None)
        # Extract and run pre_hook if present
        pre_hook = config.pop("pre_hook", None)
        split_k = config.get("SPLIT_K", 1)
        if split_k > 1:
            workspace = torch.empty((split_k * M, N), device=a.device, dtype=a.dtype)
            workspace_desc = TensorDescriptor(workspace, workspace.shape, workspace.stride(), dummy_block)
        else:
            workspace_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
        hook_args = {
            "a_desc": a_desc,
            "b_desc": b_desc,
            "c_desc": c_desc,
            "workspace_desc": workspace_desc,
            "M": M,
            "N": N,
            "K": K,
            "A_ROW_MAJOR": a_row_major,
            "B_ROW_MAJOR": b_row_major,
            **config,
        }
        if pre_hook:
            pre_hook(hook_args)
        else:
            matmul_tma_set_block_size_hook(hook_args)
        NUM_CTAS = config.get("NUM_CTAS", 1)
        num_pid_m = triton.cdiv(M, config["BLOCK_SIZE_M"])
        num_pid_n = triton.cdiv(N, config["BLOCK_SIZE_N"])
        num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
        total_tiles = num_pid_m * num_pid_n * split_k
        grid = (total_tiles, )
        matmul_kernel_tma_ws_blackwell.fn[grid](
            a_desc,
            b_desc,
            c_desc,
            workspace_desc,
            M,
            N,
            K,
            A_ROW_MAJOR=a_row_major,
            B_ROW_MAJOR=b_row_major,
            NUM_SMS=NUM_SMS,
            ctas_per_cga=ctas_per_cga,
            **config,
        )
        # Run separate reduction kernel for split-K
        if split_k > 1:
            reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
            # num_warps=4 is the default; dropping it keeps the C dispatch proxy
            # fast path (see reduce_post_hook).
            _reduce_k_kernel[reduce_grid](
                workspace_desc.base,
                c,
                M,
                N,
                SPLIT_K=split_k,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[a.dtype],
            )
    else:
        # Pass c as dummy workspace_desc. Pre_hook dynamically allocates
        # the right-sized workspace per config based on SPLIT_K.
        workspace_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

        def grid(META):
            NUM_CTAS = META["NUM_CTAS"]
            num_pid_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
            num_pid_n = triton.cdiv(N, META["BLOCK_SIZE_N"])
            # Pad num_pid_m to multiple of NUM_CTAS so CTA clusters tile evenly along M.
            num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
            mn_tiles = num_pid_m * num_pid_n
            total_tiles = mn_tiles * META["SPLIT_K"]
            return (total_tiles, )

        matmul_kernel_tma_ws_blackwell[grid](
            a_desc,
            b_desc,
            c_desc,
            workspace_desc,
            M,
            N,
            K,
            A_ROW_MAJOR=a_row_major,
            B_ROW_MAJOR=b_row_major,
            NUM_SMS=NUM_SMS,
        )
        # Run split-K reduction after the autotuner picks and launches the kernel.
        # The autotuner's post_hook only runs during benchmarking, not production calls.
        best = matmul_kernel_tma_ws_blackwell.best_config
        split_k = best.kwargs.get("SPLIT_K", 1)
        if split_k > 1:
            workspace = workspace_desc.base
            reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
            # num_warps=4 is the default; dropping it keeps the C dispatch proxy
            # fast path (see reduce_post_hook).
            _reduce_k_kernel[reduce_grid](
                workspace,
                c,
                M,
                N,
                SPLIT_K=split_k,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[a.dtype],
            )
    return c
