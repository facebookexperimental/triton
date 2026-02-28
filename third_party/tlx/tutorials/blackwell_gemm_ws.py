# TLX GEMM kernel optimized for Blackwell Warp Specialization
import math
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

# Track which (M, N, K) shapes have already printed their heuristic config
_printed_heuristic_configs = set()


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
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(num_ctas, 1, 1) if num_ctas > 1 else None,
        )
        for BM in [128, 256]
        for BN in [64, 128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4, 5, 6, 7]
        for t in [1, 2, 3]
        for m in [1, 2]
        for subtile in [1, 2, 4, 8]
        for num_ctas in [1, 2]
        for split_k in [1, 4]
        for g in [1, 8, 64]
    ]


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    NUM_MMA_GROUPS = nargs.get("NUM_MMA_GROUPS", 1)
    nargs["a_desc"].block_shape = [BLOCK_M // NUM_MMA_GROUPS, BLOCK_K]
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N // NUM_CTAS]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    nargs["c_desc"].block_shape = [
        BLOCK_M // NUM_MMA_GROUPS,
        BLOCK_N // EPILOGUE_SUBTILE,
    ]
    if nargs.get("SPLIT_K", 1) > 1:
        nargs["c_desc"].base.zero_()


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
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
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

        # Filter out invalid config that causes wrong hardware MMA
        if BLOCK_M // NUM_MMA_GROUPS > 128:
            continue

        # EPILOGUE_SUBTILE must evenly divide BLOCK_N
        if BLOCK_N % EPILOGUE_SUBTILE != 0:
            continue

        num_tiles_m = math.ceil(M / BLOCK_M)
        num_tiles_n = math.ceil(N / BLOCK_N)
        num_mn_tiles = num_tiles_m * num_tiles_n

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

    # Prefer configs that maximize SM utilization.
    # If any config produces enough tiles to fill every SM, discard those
    # that don't.  Otherwise, keep only configs whose tile count matches the
    # best available utilization so we don't waste SMs.
    if pruned_configs:

        def _total_tiles(c):
            return (math.ceil(M / c.kwargs["BLOCK_SIZE_M"]) * math.ceil(N / c.kwargs["BLOCK_SIZE_N"]) *
                    c.kwargs.get("SPLIT_K", 1))

        max_tiles = max(_total_tiles(c) for c in pruned_configs)
        if max_tiles >= NUM_SMS:
            pruned_configs = [c for c in pruned_configs if _total_tiles(c) >= NUM_SMS]
        else:
            pruned_configs = [c for c in pruned_configs if _total_tiles(c) == max_tiles]

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
            # M >> N: keep only small GROUP_SIZE_M to sweep M
            pruned_configs = [c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] == 1]
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
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


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
    c_desc,
    c_smem_buffers,
    tmem_buffers,
    tmem_full_bars,
    tmem_empty_bars,
    cur_tmem_buf,
    tmem_read_phase,
):
    """Process epilogue for a single tile."""
    mn_tile_id = tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS

    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE

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
            # Signal MMA consumer after each slice
            tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
            c = result.to(tlx.dtype_of(c_desc))
            if SPLIT_K == 1:
                # Store to SMEM then use async TMA store to global
                c_smem = c_smem_buffers[group_id]
                tlx.async_descriptor_store_wait(0)
                tlx.local_store(c_smem, c)
                tlx.fence_async_shared()
                tlx.async_descriptor_store(c_desc, c_smem, [offs_am, offs_bn + slice_id * slice_size])
            else:
                c_desc.store(
                    [offs_am, offs_bn + slice_id * slice_size],
                    c,
                    store_reduce="add",
                )
    # Wait for all TMA stores to complete
    tlx.async_descriptor_store_wait(0)


@triton.jit
def _process_tile_mma_inner(
    k_tiles,
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
    cta_bars,
    pred_cta0,
):
    """Process MMA for a single tile over [k_tile_start, k_tile_end). Returns updated smem_accum_cnt."""
    local_k_tiles = k_tile_end - k_tile_start

    # Peeled first K-iteration: wait for data before acquiring TMEM
    buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

    # wait for current phase(round) of load for this buf
    tlx.barrier_wait(B_smem_full_bars[buf], phase)

    # Process first K iteration (peeled) with use_acc=False
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        # Calculate buffer indices
        a_buf = group_id * NUM_SMEM_BUFFERS + buf
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

        # Wait for this A subtile buffer to be loaded
        tlx.barrier_wait(A_smem_full_bars[a_buf], phase)

        # Wait for epilogue to be done with all TMEM buffers (after data is ready)
        cur_barrier_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        tlx.barrier_wait(tmem_empty_bars[cur_barrier_idx], tmem_write_phase ^ 1)

        # CTA0 waits for CTA0 and CTA1 to finish loading A and B before issuing dot op
        if NUM_CTAS == 2:
            tlx.barrier_arrive(cta_bars[a_buf], arrive_count=1, remote_cta_rank=0)
            tlx.barrier_wait(cta_bars[a_buf], phase=phase, pred=pred_cta0)

        # Perform MMA: use_acc=False for first K iteration (clears accumulator)
        tlx.async_dot(
            buffers_A[a_buf],
            buffers_B[buf],
            tmem_buffers[acc_buf],
            use_acc=False,
            mBarriers=[A_smem_empty_bars[a_buf]],
            two_ctas=NUM_CTAS == 2,
            out_dtype=tl.float32,
        )

    smem_accum_cnt += 1

    # Remaining K iterations with use_acc=True
    for _ in range(1, local_k_tiles):
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

        # wait for current phase(round) of load for this buf
        tlx.barrier_wait(B_smem_full_bars[buf], phase)

        # Process all subtiles for this K iteration
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            # Calculate buffer indices
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

            # Wait for this A subtile buffer to be loaded
            tlx.barrier_wait(A_smem_full_bars[a_buf], phase)

            # CTA0 waits for CTA0 and CTA1 to finish loading A and B before issuing dot op
            if NUM_CTAS == 2:
                tlx.barrier_arrive(cta_bars[a_buf], arrive_count=1, remote_cta_rank=0)
                tlx.barrier_wait(cta_bars[a_buf], phase=phase, pred=pred_cta0)

            # Perform MMA: use_acc=True for remaining K iterations
            tlx.async_dot(
                buffers_A[a_buf],
                buffers_B[buf],
                tmem_buffers[acc_buf],
                use_acc=True,
                mBarriers=[A_smem_empty_bars[a_buf]],
                two_ctas=NUM_CTAS == 2,
                out_dtype=tl.float32,
            )

        smem_accum_cnt += 1

    # Wait for last MMA to complete and signal epilogue for all subtiles
    last_buf, last_phase = _get_bufidx_phase(smem_accum_cnt - 1, NUM_SMEM_BUFFERS)
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        a_buf = group_id * NUM_SMEM_BUFFERS + last_buf
        tlx.barrier_wait(A_smem_empty_bars[a_buf], last_phase)
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        # Done filling this buffer, signal epilogue consumer
        tlx.barrier_arrive(tmem_full_bars[acc_buf], 1)

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
):
    """Process TMA loads for a single tile with all subtiles over [k_tile_start, k_tile_end)."""
    mn_tile_id = tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    dsize: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // NUM_CTAS)
    expected_bytes: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K // NUM_CTAS

    local_k_tiles = k_tile_end - k_tile_start

    # Iterate along K dimension for this split's range
    for k_idx in range(0, local_k_tiles):
        k = k_tile_start + k_idx
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        offs_k = k * BLOCK_SIZE_K

        # Load A for the first group
        a_buf = buf
        tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)
        offs_am = pid_m * BLOCK_SIZE_M
        tlx.barrier_expect_bytes(A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K)
        tlx.async_descriptor_load(a_desc, buffers_A[a_buf], [offs_am, offs_k], A_smem_full_bars[a_buf])

        # Load B once per K iteration (shared across all subtiles)
        last_a_buf = (NUM_MMA_GROUPS - 1) * NUM_SMEM_BUFFERS + buf
        tlx.barrier_wait(A_smem_empty_bars[last_a_buf], phase ^ 1)
        tlx.barrier_expect_bytes(B_smem_full_bars[buf], expected_bytes)
        tlx.async_descriptor_load(b_desc, buffers_B[buf], [offs_k, offs_bn], B_smem_full_bars[buf])

        # Load all remaining A subtiles for this K iteration
        for group_id in tl.static_range(1, NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf

            tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)

            offs_am2 = offs_am + group_id * BLOCK_M_SPLIT

            tlx.barrier_expect_bytes(A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K)
            tlx.async_descriptor_load(a_desc, buffers_A[a_buf], [offs_am2, offs_k], A_smem_full_bars[a_buf])

        smem_accum_cnt += 1

    return smem_accum_cnt


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": preprocess_configs},
)
@triton.jit
def matmul_kernel_tma_ws_blackwell(
    a_desc,
    b_desc,
    c_desc,
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
    NUM_SMS: tl.constexpr,
):
    # allocate NUM_SMEM_BUFFERS buffers
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    buffers_A = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_K),
        tlx.dtype_of(a_desc),
        NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
    )
    # In 2-CTA mode, each CTA only needs to load BLOCK_N // NUM_CTAS of B.
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

    # Allocate SMEM buffer for epilogue TMA store (one per MMA group)
    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    c_smem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, slice_size),
        tlx.dtype_of(c_desc),
        NUM_MMA_GROUPS,
    )

    # CTA pairs are placed along M dim
    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        cta_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
                                      arrive_count=2)  # CTA0 waits for CTA1's data before mma
    else:
        cluster_cta_rank = 0
        pred_cta0 = False
        cta_bars = None

    # allocate barriers - each subtile needs its own barriers
    # NUM_SMEM_BUFFERS barriers per subtile for synchronization
    A_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    A_smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    B_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    # NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=EPILOGUE_SUBTILE)

    with tlx.async_tasks():
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

            while tile_id < num_tiles:
                cur_tmem_buf, tmem_read_phase = _get_bufidx_phase(tmem_accum_cnt, NUM_TMEM_BUFFERS)
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
                    c_desc=c_desc,
                    c_smem_buffers=c_smem_buffers,
                    tmem_buffers=tmem_buffers,
                    tmem_full_bars=tmem_full_bars,
                    tmem_empty_bars=tmem_empty_bars,
                    cur_tmem_buf=cur_tmem_buf,
                    tmem_read_phase=tmem_read_phase,
                )
                tmem_accum_cnt += 1
                tile_id += NUM_SMS

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

            while tile_id < num_tiles:
                # Compute K range for this split
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)

                cur_tmem_buf, tmem_write_phase = _get_bufidx_phase(tmem_accum_cnt, NUM_TMEM_BUFFERS)
                smem_accum_cnt = _process_tile_mma_inner(
                    k_tiles=k_tiles_total,
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
                    cta_bars=cta_bars,
                    pred_cta0=pred_cta0,
                )
                tmem_accum_cnt += 1
                tile_id += NUM_SMS

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

            while tile_id < num_tiles:
                # Compute K range for this split
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)

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
                )
                tile_id += NUM_SMS


def matmul(a, b, config=None, use_heuristic=True):
    """Matrix multiplication using TLX GEMM kernel.

    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
        config: Optional dict with kernel config. If None and use_heuristic=True,
                uses shape-dependent heuristic selection. If heuristic fails,
                falls back to full autotuning.
        use_heuristic: When config is None, try heuristic config selection first.
                      Default True for faster kernel launch.

    Returns:
        Output matrix C of shape (M, N)
    """
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # Use heuristic config if no config provided and heuristic is enabled
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
        if pre_hook:
            pre_hook({"a_desc": a_desc, "b_desc": b_desc, "c_desc": c_desc, **config})
        else:
            matmul_tma_set_block_size_hook({"a_desc": a_desc, "b_desc": b_desc, "c_desc": c_desc, **config})
        NUM_CTAS = config.get("NUM_CTAS", 1)
        num_pid_m = triton.cdiv(M, config["BLOCK_SIZE_M"])
        num_pid_n = triton.cdiv(N, config["BLOCK_SIZE_N"])
        num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
        total_tiles = num_pid_m * num_pid_n * config.get("SPLIT_K", 1)
        grid = (min(NUM_SMS, total_tiles), )
        matmul_kernel_tma_ws_blackwell.fn[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            NUM_SMS=NUM_SMS,
            ctas_per_cga=ctas_per_cga,
            **config,
        )
    else:

        def grid(META):
            NUM_CTAS = META["NUM_CTAS"]
            num_pid_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
            num_pid_n = triton.cdiv(N, META["BLOCK_SIZE_N"])
            # Pad num_pid_m to multiple of NUM_CTAS so CTA clusters tile evenly along M.
            num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
            mn_tiles = num_pid_m * num_pid_n
            total_tiles = mn_tiles * META["SPLIT_K"]
            return (min(NUM_SMS, total_tiles), )

        matmul_kernel_tma_ws_blackwell[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            NUM_SMS=NUM_SMS,
        )
    return c
