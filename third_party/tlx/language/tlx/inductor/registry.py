import dataclasses
import logging
from typing import Any, Generator

log = logging.getLogger(__name__)

import torch
from torch._inductor import config
from torch._inductor.kernel_inputs import KernelInputs, MMKernelInputs
from torch._inductor.template_heuristics.registry import register_template_heuristic
from torch._inductor.template_heuristics.triton import (
    CUDAConfigHeuristic,
    GemmConfig,
    ROCmMMTemplateConfigHeuristic,
    TMATemplateConfigMixin,
)
from torch._inductor.template_heuristics.triton_addmm import AddMMConfigMixin
from torch._inductor.utils import get_num_sms

# IS_ROCM was dropped from torch._inductor.template_heuristics.triton on newer
# nightlies (the module now branches on ``torch.version.hip`` inline). Derive it
# locally so the fork loads across torch versions; matches torch's own definition
# (CUDA vs ROCm keyed on torch.version.hip).
IS_ROCM = torch.version.hip is not None

try:
    from torch._inductor.utils import get_default_kpack
except ImportError:
    # get_default_kpack is absent from some torch nightlies (notably ROCm
    # wheels). Mirror torch's own definition so the fork loads across versions:
    # 0 on CUDA; on AMD, kpack keyed on arch/block_k.
    def get_default_kpack(block_k: int = 16) -> int:
        if not torch.version.hip:
            return 0
        try:
            arch = torch.cuda.get_device_properties(0).gcnArchName
        except Exception:
            arch = ""
        if "gfx942" in arch and block_k <= 16:
            return 1
        return 2


def _sizevar_hint(sizevars, expr, fallback):
    # ``optimization_hint`` is the newer name; older torch (e.g. the current ROCm
    # wheel) exposes ``size_hint`` with the same fallback semantics (example-input
    # hint, ``fallback`` when unbacked/symbolic). Use whichever exists.
    fn = getattr(sizevars, "optimization_hint", None) or sizevars.size_hint
    return fn(expr, fallback=fallback)


from . import tlx_config
from .mm_templates import (
    amd_addmm_warppipe_template,
    amd_bmm_warppipe_persistent_template,
    amd_bmm_warppipe_template,
    blackwell_gemm_ws_template,
)


@dataclasses.dataclass
class TlxGemmConfig(GemmConfig):
    """
    Gemm configuration for TLX templates with TLX-specific parameters.
    """

    group_size_m: int = dataclasses.field(kw_only=True, default=8)
    smem_num: int = dataclasses.field(kw_only=True, default=3)
    tmem_num: int = dataclasses.field(kw_only=True, default=2)
    epilogue_subtile: int = dataclasses.field(kw_only=True, default=1)
    num_mma_groups: int = dataclasses.field(kw_only=True, default=1)
    num_ctas: int = dataclasses.field(kw_only=True, default=1)
    split_k: int = dataclasses.field(kw_only=True, default=1)
    interleave_epilogue: int = dataclasses.field(kw_only=True, default=0)


# ---------------------------------------------------------------------
# Heuristic config selection (matches tlx_matmul_ws behavior)
# Implemented directly in Python to avoid dependency on YAML rule engine files
# ---------------------------------------------------------------------

import math as _math


def _amd_num_xcds() -> int:
    """Number of XCDs (chiplets) on the current ROCm GPU, for the L2 chiplet swizzle.

    gfx942 (MI300X) and gfx950 (MI350X) have 8. Returns 1 -- which makes the swizzle a
    no-op (identity) -- for non-HIP or arches with no known XCD count, since the template
    is registered for all ROCm, not just the 8-XCD parts.
    """
    if not torch.version.hip:
        return 1
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if "gfx942" in arch or "gfx950" in arch:
        return 8
    return 1


def _select_group_size_m(M: int, N: int, block_m: int) -> int:
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


def _is_config_valid(
    config: dict[str, Any], tma_epilogue_store: bool = False, smem_margin: int = 0
) -> bool:
    """Check if a config is valid based on hardware constraints."""
    # Upstream uses 232*1024 as a loose estimate, but the actual Blackwell
    # hardware limit is 232448 bytes.  We use the real limit because
    # epilogue fusion can add SMEM beyond what the formula captures.
    MAX_SHARED_MEMORY = 232448  # B200 SMEM per SM (actual hardware limit)
    MAX_TMEM_COLUMNS = 512  # TMEM columns per SM (Blackwell hardware limit)

    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    num_ctas = config["NUM_CTAS"]
    num_smem_buffers = config["NUM_SMEM_BUFFERS"]
    num_tmem_buffers = config["NUM_TMEM_BUFFERS"]
    num_mma_groups = config["NUM_MMA_GROUPS"]
    epilogue_subtile = config["EPILOGUE_SUBTILE"]

    # Check MMA groups constraint
    if block_m // num_mma_groups > 128:
        return False

    # Pair-CTA MMA requires M=128 per MMA group
    if num_ctas == 2 and block_m // num_mma_groups < 128:
        return False

    # Check epilogue subtile
    if block_n % epilogue_subtile != 0:
        return False

    # Shared memory estimation — matches upstream estimate_smem exactly.
    # Split-K fp32 workspace overhead is torchTLX-specific: upstream uses output
    # dtype (bf16) for the workspace, but torchTLX uses fp32 for accuracy.
    smem_a = block_m * block_k * 2 * num_smem_buffers
    smem_b_size = block_n // num_ctas
    smem_b = block_k * smem_b_size * 2 * num_smem_buffers
    if tma_epilogue_store:
        smem_epilog = block_m * (block_n // epilogue_subtile) * 2
    else:
        smem_epilog = 0
    smem_barriers = num_smem_buffers * num_mma_groups * 8
    if num_ctas == 2:
        smem_barriers += num_smem_buffers * num_mma_groups * 8
    total_smem = smem_a + smem_b + smem_epilog + smem_barriers

    split_k = config.get("SPLIT_K", 1)
    if split_k > 1:
        # torchTLX stores fp32 partials to workspace (upstream uses bf16).
        # Account for the fp32 ws_smem_buffers allocated in the template.
        block_m_split = block_m // num_mma_groups
        slice_size = block_n // epilogue_subtile
        num_epilogue_smem_buffers = max(num_mma_groups, 2)
        smem_ws = block_m_split * slice_size * 4 * num_epilogue_smem_buffers
        total_smem += smem_ws

    if total_smem + smem_margin > MAX_SHARED_MEMORY:
        return False

    # TMEM estimation (columns, not bytes)
    total_tmem_columns = block_n * num_tmem_buffers * num_mma_groups
    if total_tmem_columns > MAX_TMEM_COLUMNS:
        return False

    return True


def _fix_config_if_needed(
    config: dict[str, Any], tma_epilogue_store: bool = False
) -> dict[str, Any] | None:
    """
    Fix config to stay within shared memory limits.
    Returns None if config cannot be fixed.
    """
    if _is_config_valid(config, tma_epilogue_store=tma_epilogue_store):
        return config
    # Config overflows SMEM (e.g., split-K fp32 workspace overhead).
    # Return None so get_heuristic_config falls through to the candidate
    # scorer, and ultimately to the autotuning pool if no scorer config fits.
    return None


def _select_split_k(K: int, block_k: int) -> int:
    """Select split-K factor for undersaturated shapes.

    Tries split factors in order [4, 2, 8]; picks the first where each split
    has at least 4 K-tiles.  Returns 1 if no split factor is suitable.
    """
    k_tiles = _math.ceil(K / block_k)
    for sk in [4, 2, 8]:
        k_tiles_per_split = _math.ceil(k_tiles / sk)
        if k_tiles_per_split * (sk - 1) >= k_tiles:
            continue  # last split would be empty
        if k_tiles // sk >= 4:
            return sk
    return 1


def get_heuristic_config(
    M: int,
    N: int,
    K: int,
    num_sms: int = 148,
    tma_epilogue_store: bool = False,
) -> dict[str, Any] | None:
    """
    Select optimal GEMM config based on problem shape characteristics.

    This implements heuristic rules for TLX Blackwell GEMM kernel configuration.
    Rules are evaluated in order (first match wins).

    Heuristic Rules Index:
    ----------------------
    Rule 0: Fallback (Candidate Scoring)
        - Condition: No single-config rule matches, or block size validation fails
        - Config: Selected via _candidate_scorer_evaluate()
        - Use case: Edge cases and shapes not covered by explicit rules

    Rule 1a: Tall-M, alt-tiling, moderate M (torchTLX extension)
        - Condition: is_tall_m AND gpu_saturated AND use_alt_tiling AND m_tiles <= num_sms//2
        - Config: (128, 256, 64), 1-CTA, 3 SMEM buffers
        - Use case: Shapes where BN=256 wastes tiles (N<=256, N unaligned, low tile count)

    Rule 1b: Tall-M Low-AI (or alt-tiling fallback)
        - Condition: is_tall_m AND gpu_saturated AND (AI <= 1.5 OR use_alt_tiling)
        - Config: (256, 128, 128), 2-CTA, 2 SMEM buffers, INTERLEAVE=1
        - Use case: Memory-bound tall-M shapes

    Rule 3: Tall-M High-AI, K > N*2
        - Condition: is_tall_m AND gpu_saturated AND AI > 1.5 AND K > N*2
        - Config: (256, 256, 128), 2-CTA, 2 SMEM buffers, EPILOGUE_SUBTILE=4
        - Use case: Compute-bound tall-M with large K relative to N

    Rule 4: Tall-M High-AI, K <= N*2
        - Condition: is_tall_m AND gpu_saturated AND AI > 1.5 AND K <= N*2
        - Config: (256, 256, 64), 2-CTA, 4 SMEM buffers, EPILOGUE_SUBTILE=4, INTERLEAVE=1
        - Use case: Compute-bound tall-M with K similar to or smaller than N

    Rule 5: Undersaturated Large-Output
        - Condition: undersaturated AND MN >= 1M
        - Config: (256, 128, 64), 4 SMEM buffers, EPILOGUE_SUBTILE=8
        - Use case: Large output matrices that don't saturate GPU

    Rule 6: Undersaturated Small-Output
        - Condition: undersaturated AND MN < 1M
        - Config: (128, 64, 128), 4 SMEM buffers
        - Use case: Small output matrices that don't saturate GPU

    Rule 7: GPU-Saturated General (Wide-N)
        - Condition: gpu_saturated AND NOT is_tall_m
        - Config: (256, 256, 64), 1-CTA, 3 SMEM buffers, EPILOGUE_SUBTILE=4, INTERLEAVE=1
        - Use case: GPU-saturated shapes with balanced or wide-N dimensions

    Args:
        M, N, K: GEMM dimensions (A is MxK, B is KxN, C is MxN)
        num_sms: Number of SMs on the GPU (default 148 for B200)

    Returns:
        dict: Configuration parameters for the TLX GEMM kernel,
        or None if no valid config can be determined.
    """
    # --- Compute derived features ---
    mn_ratio = M / max(N, 1)
    is_tall_m = mn_ratio > 4
    is_tall_n = mn_ratio < 0.25

    # Reference block sizes for tile counting
    ref_bm = 256 if is_tall_m else (128 if is_tall_n else 256)
    ref_bn = 128 if is_tall_m else (256 if is_tall_n else 256)

    num_mn_tiles = _math.ceil(M / ref_bm) * _math.ceil(N / ref_bn)
    gpu_saturated = num_mn_tiles >= num_sms
    undersaturated = num_mn_tiles < num_sms
    mn_product = M * N
    is_large_output = mn_product >= 1000000

    config = None

    # --- Rule matching (first match wins) ---

    # Characteristic 1: Tall-M saturated shapes
    if is_tall_m and gpu_saturated:
        arithmetic_intensity = K / max(min(M, N), 1)

        # Check if the default high-AI path (Rule 3: BM=256 BN=256) would
        # be suboptimal.  Three triggers:
        #   1. N <= 256: BN=256 >= N, config would be rejected downstream
        #   2. Few total tiles at BN=256: poor wave efficiency
        #   3. N < 1024 and N not aligned to 256: significant tile waste
        use_alt_tiling = False
        if arithmetic_intensity > 1.5:
            tiles_bn256 = _math.ceil(M / 256) * _math.ceil(N / 256)
            if N <= 256:
                use_alt_tiling = True
            elif tiles_bn256 < 4 * num_sms:
                use_alt_tiling = True
            elif N < 1024 and N % 256 != 0:
                use_alt_tiling = True

        # Rule 1a/1b: Tall-M Low-AI, or high-AI with suboptimal BN=256
        if arithmetic_intensity <= 1.5 or use_alt_tiling:
            m_tiles_256 = _math.ceil(M / 256)
            # Rule 1a: Moderate-M — use BM=128 1-CTA to avoid 2-CTA
            # coordination overhead and improve tile granularity.
            if use_alt_tiling and m_tiles_256 <= num_sms // 2:
                config = {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 64,
                    "NUM_SMEM_BUFFERS": 3,
                    "NUM_TMEM_BUFFERS": 2,
                    "NUM_MMA_GROUPS": 1,
                    "EPILOGUE_SUBTILE": 2,
                    "NUM_CTAS": 1,
                    "SPLIT_K": 1,
                    "INTERLEAVE_EPILOGUE": 0,
                }
            # Rule 1b: Large-M — BM=256 2-CTA streams A efficiently.
            else:
                config = {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "NUM_SMEM_BUFFERS": 2,
                    "NUM_TMEM_BUFFERS": 2,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 1,
                    "NUM_CTAS": 2,
                    "SPLIT_K": 1,
                    "INTERLEAVE_EPILOGUE": 1,
                }
        # Rule 3: Tall-M High-AI, K > N*2 — large BLOCK_K for fewer K-iterations
        elif K > N * 2:
            config = {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "NUM_SMEM_BUFFERS": 2,
                "NUM_TMEM_BUFFERS": 1,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": 4,
                "NUM_CTAS": 2,
                "SPLIT_K": 1,
                "INTERLEAVE_EPILOGUE": 0,
            }
        # Rule 4: Tall-M High-AI, K <= N*2 — more SMEM buffers, interleaved epilogue
        else:
            config = {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "NUM_SMEM_BUFFERS": 4,
                "NUM_TMEM_BUFFERS": 1,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": 4,
                "NUM_CTAS": 2,
                "SPLIT_K": 1,
                "INTERLEAVE_EPILOGUE": 1,
            }

    # Characteristic 2: Undersaturated shapes — use split-K to improve parallelism.
    # The template writes fp32 partials to a workspace; a separate reduction
    # kernel sums the partials after the main GEMM.
    #
    # Note: upstream only returns Rule 5/6 when split_k > 1 and falls through
    # to the candidate scorer otherwise.  torchTLX intentionally returns the
    # config even with split_k=1 to provide a deterministic tile size for
    # undersaturated shapes (the candidate scorer may pick a suboptimal config
    # for these shapes since its wave-efficiency scoring doesn't account for
    # the tile shape preferences encoded in Rules 5/6).
    elif undersaturated and is_large_output:
        block_k = 64
        split_k = _select_split_k(K, block_k)
        # Rule 5: Undersaturated Large-Output
        config = {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": block_k,
            "NUM_SMEM_BUFFERS": 4,
            "NUM_TMEM_BUFFERS": 2,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": 8,
            "NUM_CTAS": 1,
            "SPLIT_K": split_k,
            "INTERLEAVE_EPILOGUE": 1,
        }
    elif undersaturated and not is_large_output:
        block_k = 128
        split_k = _select_split_k(K, block_k)
        # Rule 6: Undersaturated Small-Output
        config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": block_k,
            "NUM_SMEM_BUFFERS": 4,
            "NUM_TMEM_BUFFERS": 3,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": 1,
            "NUM_CTAS": 1,
            "SPLIT_K": split_k,
            "INTERLEAVE_EPILOGUE": 1,
        }

    # Rule 7: GPU-Saturated General (Wide-N)
    elif gpu_saturated:
        config = {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "NUM_SMEM_BUFFERS": 3,
            "NUM_TMEM_BUFFERS": 1,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": 4,
            "NUM_CTAS": 1,
            "SPLIT_K": 1,
            "INTERLEAVE_EPILOGUE": 1,
        }

    # Rule 0: Fallback (Candidate Scoring)
    if config is None:
        config = _candidate_scorer_evaluate(M, N, K, num_sms)

    if config is None:
        return None

    # Validate block sizes don't exceed problem dimensions
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]

    if block_n >= N or block_k >= K:
        config = _candidate_scorer_evaluate(M, N, K, num_sms)
        if config is None:
            return None
        if config["BLOCK_SIZE_N"] >= N or config["BLOCK_SIZE_K"] >= K:
            return None

    # Validate and fix config if needed
    config = _fix_config_if_needed(config, tma_epilogue_store=tma_epilogue_store)
    if config is None:
        config = _candidate_scorer_evaluate(M, N, K, num_sms)
        if config is not None:
            config = _fix_config_if_needed(
                config, tma_epilogue_store=tma_epilogue_store
            )
        if config is None:
            return None

    # Post-process: add GROUP_SIZE_M and ctas_per_cga
    block_m = config["BLOCK_SIZE_M"]
    num_ctas = config["NUM_CTAS"]
    config["GROUP_SIZE_M"] = _select_group_size_m(M, N, block_m)
    # GROUP_SIZE_M must be a multiple of NUM_CTAS so that consecutive
    # tile_ids (paired CTAs in a cluster) map to the same pid_n.
    if num_ctas > 1:
        gsm = config["GROUP_SIZE_M"]
        config["GROUP_SIZE_M"] = ((gsm + num_ctas - 1) // num_ctas) * num_ctas
    config["ctas_per_cga"] = (num_ctas, 1, 1) if num_ctas > 1 else None

    return config


# --- Candidate scoring (fallback when rules don't match) ---

_CANDIDATES = [
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "NUM_CTAS": 2,
        "NUM_SMEM_BUFFERS": 2,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 2,
        "EPILOGUE_SUBTILE": 1,
    },
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "NUM_CTAS": 1,
        "NUM_SMEM_BUFFERS": 3,
        "NUM_TMEM_BUFFERS": 1,
        "NUM_MMA_GROUPS": 2,
        "EPILOGUE_SUBTILE": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "NUM_CTAS": 1,
        "NUM_SMEM_BUFFERS": 4,
        "NUM_TMEM_BUFFERS": 3,
        "NUM_MMA_GROUPS": 2,
        "EPILOGUE_SUBTILE": 1,
    },
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "NUM_CTAS": 2,
        "NUM_SMEM_BUFFERS": 5,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 2,
        "EPILOGUE_SUBTILE": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "NUM_CTAS": 2,
        "NUM_SMEM_BUFFERS": 4,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 1,
        "EPILOGUE_SUBTILE": 2,
    },
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "NUM_CTAS": 2,
        "NUM_SMEM_BUFFERS": 5,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 2,
        "EPILOGUE_SUBTILE": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "NUM_CTAS": 2,
        "NUM_SMEM_BUFFERS": 5,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 2,
        "EPILOGUE_SUBTILE": 1,
    },
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "NUM_CTAS": 1,
        "NUM_SMEM_BUFFERS": 5,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 2,
        "EPILOGUE_SUBTILE": 8,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "NUM_CTAS": 1,
        "NUM_SMEM_BUFFERS": 3,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 1,
        "EPILOGUE_SUBTILE": 2,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "NUM_CTAS": 1,
        "NUM_SMEM_BUFFERS": 4,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 1,
        "EPILOGUE_SUBTILE": 2,
    },
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "NUM_CTAS": 1,
        "NUM_SMEM_BUFFERS": 3,
        "NUM_TMEM_BUFFERS": 1,
        "NUM_MMA_GROUPS": 2,
        "EPILOGUE_SUBTILE": 2,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "NUM_CTAS": 1,
        "NUM_SMEM_BUFFERS": 5,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 1,
        "EPILOGUE_SUBTILE": 1,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "NUM_CTAS": 1,
        "NUM_SMEM_BUFFERS": 5,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 1,
        "EPILOGUE_SUBTILE": 1,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "NUM_CTAS": 1,
        "NUM_SMEM_BUFFERS": 6,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 1,
        "EPILOGUE_SUBTILE": 1,
    },
]


def _candidate_scorer_evaluate(
    M: int, N: int, K: int, num_sms: int
) -> dict[str, Any] | None:
    """Score candidates by wave efficiency and return best."""
    MAX_SMEM = 232448  # B200 SMEM per SM (actual hardware limit)
    MAX_TMEM = 256 * 1024

    best_config = None
    best_score = float("inf")
    best_waves = float("inf")

    for cfg in _CANDIDATES:
        bm = cfg["BLOCK_SIZE_M"]
        bn = cfg["BLOCK_SIZE_N"]
        bk = cfg["BLOCK_SIZE_K"]
        num_ctas = cfg["NUM_CTAS"]
        num_smem_buffers = cfg["NUM_SMEM_BUFFERS"]
        num_tmem_buffers = cfg["NUM_TMEM_BUFFERS"]
        num_mma_groups = cfg["NUM_MMA_GROUPS"]
        epilogue_subtile = cfg["EPILOGUE_SUBTILE"]

        # Constraint checks
        smem_a = bm * bk * 2 * num_smem_buffers
        smem_b = bk * (bn // num_ctas) * 2 * num_smem_buffers
        smem_epilog = bm * (bn // epilogue_subtile) * 2
        smem_barriers = (
            num_smem_buffers * num_mma_groups * 8 * (2 if num_ctas == 2 else 1)
        )
        total_smem = smem_a + smem_b + smem_epilog + smem_barriers
        if total_smem > MAX_SMEM:
            continue

        total_tmem = bm * bn * 4 * num_tmem_buffers
        if total_tmem > MAX_TMEM:
            continue

        if bm // num_mma_groups > 128:
            continue

        # Block sizes must be strictly less than problem dimensions for correctness
        if bn >= N or bk >= K:
            continue

        if bm > M * 2:
            continue

        # Wave efficiency scoring
        ctas_m = (M + bm - 1) // bm
        ctas_n = (N + bn - 1) // bn
        ctas_m = ((ctas_m + num_ctas - 1) // num_ctas) * num_ctas
        total_ctas = ctas_m * ctas_n

        if total_ctas == 0:
            continue

        waves = (total_ctas + num_sms - 1) // num_sms
        fractional_waves = total_ctas / num_sms
        score = waves - fractional_waves

        # Consider split-K for undersaturated shapes
        split_k = 1
        num_tiles_m = _math.ceil(M / bm)
        num_tiles_n = _math.ceil(N / bn)
        num_mn_tiles = num_tiles_m * num_tiles_n

        if num_mn_tiles < num_sms:
            k_tiles = _math.ceil(K / bk)
            for sk in [8, 4, 2]:
                k_tiles_per_split = _math.ceil(k_tiles / sk)
                if k_tiles_per_split * (sk - 1) >= k_tiles:
                    continue  # last split would be empty
                if k_tiles >= sk and k_tiles // sk >= 4:
                    sk_ctas_m = ((num_tiles_m + num_ctas - 1) // num_ctas) * num_ctas
                    sk_total_ctas = sk_ctas_m * num_tiles_n * sk
                    sk_waves = (sk_total_ctas + num_sms - 1) // num_sms
                    sk_frac = sk_total_ctas / num_sms
                    sk_score = sk_waves - sk_frac
                    if sk_score < score or (
                        sk_score == score and sk_total_ctas > total_ctas
                    ):
                        score, total_ctas, waves, split_k = (
                            sk_score,
                            sk_total_ctas,
                            sk_waves,
                            sk,
                        )
                    break

        # Selection
        score_slack = 0.1
        if (
            score < best_score - score_slack
            or (score < best_score + score_slack and waves < best_waves)
            or (
                score < best_score + score_slack
                and waves == best_waves
                and num_ctas > 1
            )
        ):
            best_score = score
            best_waves = waves
            best_config = dict(cfg)
            best_config["SPLIT_K"] = split_k
            best_config["INTERLEAVE_EPILOGUE"] = 0

    return best_config


class TLXMatmulWSConfigMixin(TMATemplateConfigMixin):
    """Mixin for TLX Matmul WS template with TLX-specific parameters and config validation."""

    # Blackwell B200A resource limits
    MAX_SHARED_MEMORY = 232448  # B200 SMEM per SM (actual hardware limit)
    MAX_TMEM_COLUMNS = 512  # TMEM columns per SM (Blackwell hardware limit)
    MBARRIER_SIZE = 8  # bytes

    @staticmethod
    def _is_valid_config(
        block_m: int,
        block_n: int,
        block_k: int,
        num_smem_buffers: int,
        num_tmem_buffers: int,
        num_mma_groups: int,
        num_ctas: int,
        epilogue_subtile: int,
    ) -> bool:
        """
        Check if config is valid based on hardware constraints.
        Based on preprocess_configs from tritonbench/operators/gemm/tlx_matmul.py

        Returns:
            True if config is valid, False if should be pruned.
        """
        # Rule 1: Filter out invalid config that causes wrong hardware MMA
        if block_m // num_mma_groups > 128:
            return False

        # Rule 1b: Pair-CTA MMA requires M=128 per MMA group
        if num_ctas == 2 and block_m // num_mma_groups < 128:
            return False

        # Rule 2: EPILOGUE_SUBTILE must evenly divide BLOCK_N
        if block_n % epilogue_subtile != 0:
            return False

        # Rule 3: Estimate Shared Memory Usage
        # buffers_A: BLOCK_M x BLOCK_K x float16 x NUM_SMEM_BUFFERS
        smem_a = block_m * block_k * 2 * num_smem_buffers
        # buffers_B: BLOCK_K x BLOCK_N x float16 x NUM_SMEM_BUFFERS
        # In NUM_CTAS=2 mode, each CTA only loads half of B
        smem_b_size = block_n // num_ctas
        smem_b = block_k * smem_b_size * 2 * num_smem_buffers
        # Epilogue staging buffer
        smem_epilog = block_m * (block_n // epilogue_subtile) * 2

        smem_barriers = (
            num_smem_buffers * num_mma_groups * TLXMatmulWSConfigMixin.MBARRIER_SIZE
        )
        if num_ctas == 2:
            smem_barriers += (
                num_smem_buffers * num_mma_groups * TLXMatmulWSConfigMixin.MBARRIER_SIZE
            )

        total_smem = smem_a + smem_b + smem_epilog + smem_barriers
        if total_smem > TLXMatmulWSConfigMixin.MAX_SHARED_MEMORY:
            return False

        # Rule 4: Estimate Tensor Memory (TMEM) Usage
        total_tmem_columns = block_n * num_tmem_buffers * num_mma_groups
        if total_tmem_columns > TLXMatmulWSConfigMixin.MAX_TMEM_COLUMNS:
            return False

        return True

    # Safety margin (bytes) added to the static SMEM estimate to account for
    # epilogue fusion overhead (alignment padding, extra barriers, etc.) that
    # can push the actual kernel SMEM past the hardware limit at launch time.
    _SMEM_SAFETY_MARGIN = 8192

    @classmethod
    def _is_valid_config_with_margin(
        cls,
        block_m: int,
        block_n: int,
        block_k: int,
        num_smem_buffers: int,
        num_tmem_buffers: int,
        num_mma_groups: int,
        num_ctas: int,
        epilogue_subtile: int,
    ) -> bool:
        """Like _is_valid_config but with a safety margin on SMEM."""
        smem_a = block_m * block_k * 2 * num_smem_buffers
        smem_b_size = block_n // num_ctas
        smem_b = block_k * smem_b_size * 2 * num_smem_buffers
        smem_epilog = block_m * (block_n // epilogue_subtile) * 2

        smem_barriers = num_smem_buffers * num_mma_groups * cls.MBARRIER_SIZE
        if num_ctas == 2:
            smem_barriers += num_smem_buffers * num_mma_groups * cls.MBARRIER_SIZE

        total_smem = smem_a + smem_b + smem_epilog + smem_barriers
        return total_smem + cls._SMEM_SAFETY_MARGIN <= cls.MAX_SHARED_MEMORY

    def adjust_kernel_inputs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> KernelInputs:
        """
        Adjust kernel inputs for TLX template.

        TLX Blackwell GEMM template requires both A and B to be row-major
        (stride[-1] == 1). If either is column-major, make it contiguous to
        convert to row-major layout.

        Note: Unlike triton_blackwell_ws_persistent_device_tma_mm.py.jinja which uses
        A_ROW_MAJOR/B_ROW_MAJOR flags to handle column-major inputs via transposed TMA
        descriptor + .T after load, the TLX template cannot easily support this approach
        because:
        1. tlx.async_dot operates directly on SMEM buffers with fixed shapes
        2. Loading column-major data with transposed descriptor yields transposed layout
           in SMEM
        3. There's no simple transpose before async_dot (unlike tl.dot which accepts .T)

        Making A/B contiguous here matches the standalone TLX kernel behavior
        (a.contiguous(), b.contiguous() in tritonbench wrapper) and ensures correct
        results with minimal template changes.
        """
        from torch._inductor.ir import ExternKernel

        assert isinstance(kernel_inputs, MMKernelInputs), "Expect MMKernelInputs"

        strides = kernel_inputs.strides_hinted()
        nodes = list(kernel_inputs.nodes())
        mat1_idx = kernel_inputs._mat1_idx
        mat2_idx = kernel_inputs._mat2_idx
        changed = False

        # Check A's layout from strides
        # Column-major A has stride pattern [1, M] where inner dim stride is not 1
        a_strides = strides[mat1_idx]
        is_a_col_major = a_strides[-2] == 1 and a_strides[-1] != 1
        if is_a_col_major:
            mat_a_contiguous = ExternKernel.require_contiguous(nodes[mat1_idx])
            nodes[mat1_idx] = mat_a_contiguous
            changed = True

        # Check B's layout from strides
        # Column-major B has stride pattern [1, N] where first stride is 1
        b_strides = strides[mat2_idx]
        is_b_col_major = b_strides[-2] == 1 and b_strides[-1] != 1
        if is_b_col_major:
            mat_b_contiguous = ExternKernel.require_contiguous(nodes[mat2_idx])
            nodes[mat2_idx] = mat_b_contiguous
            changed = True

        if changed:
            return MMKernelInputs(
                nodes,
                kernel_inputs.output_layout(),
                mat1_idx=mat1_idx,
                mat2_idx=mat2_idx,
            )

        return kernel_inputs

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate TLX template configs with TLX-specific parameters,
        adjusting NUM_CTAS based on problem size.

        Behavior by mode:
        - "force": yields a single heuristic config (if available), no autotuning
        - "allow": yields heuristic config + additional autotuning configs from
          self.mm_configs, so TLX competes via autotuning with cublas/triton
        """
        import math

        # Get M, N, K from kernel inputs for compatibility check
        assert isinstance(kernel_inputs, MMKernelInputs), "Expect MMKernelInputs"
        m, n, k = kernel_inputs.mnk_hinted()
        num_sms = get_num_sms()
        is_allow_mode = config.triton.tlx_mode == "allow"

        # Try heuristic config selection first (matches tlx_matmul_ws behavior).
        # Always validate without TMA epilogue store, matching upstream TLX.
        # TMA store adds SMEM overhead that would reject SMEM-tight heuristic
        # configs (e.g. 256x256x64 3-buffer); _yield_tma_variants filters
        # TMA=1 variants that don't fit in SMEM.
        heuristic_config = (
            get_heuristic_config(m, n, k, num_sms, tma_epilogue_store=False)
            if tlx_config.use_heuristic_config
            else None
        )
        if heuristic_config is not None:
            # Convert config keys to template kwargs
            template_kwargs: dict[str, Any] = {
                "BLOCK_M": heuristic_config["BLOCK_SIZE_M"],
                "BLOCK_N": heuristic_config["BLOCK_SIZE_N"],
                "BLOCK_K": heuristic_config["BLOCK_SIZE_K"],
                "GROUP_SIZE_M": heuristic_config["GROUP_SIZE_M"],
                "NUM_SMEM_BUFFERS": heuristic_config["NUM_SMEM_BUFFERS"],
                "NUM_TMEM_BUFFERS": heuristic_config["NUM_TMEM_BUFFERS"],
                "NUM_MMA_GROUPS": heuristic_config["NUM_MMA_GROUPS"],
                "EPILOGUE_SUBTILE": heuristic_config["EPILOGUE_SUBTILE"],
                "BLOCK_M_SPLIT": heuristic_config["BLOCK_SIZE_M"]
                // heuristic_config["NUM_MMA_GROUPS"],
                "slice_size": heuristic_config["BLOCK_SIZE_N"]
                // heuristic_config["EPILOGUE_SUBTILE"],
                "NUM_CTAS": heuristic_config["NUM_CTAS"],
                "SPLIT_K": heuristic_config.get("SPLIT_K", 1),
                "INTERLEAVE_EPILOGUE": heuristic_config.get("INTERLEAVE_EPILOGUE", 0),
                "num_stages": 1,
                "num_warps": 4,
                "NUM_SMS": num_sms,
            }

            # Check NUM_CTAS=2 compatibility
            BLOCK_M = template_kwargs["BLOCK_M"]
            BLOCK_N = template_kwargs["BLOCK_N"]
            NUM_CTAS = template_kwargs["NUM_CTAS"]

            if NUM_CTAS == 2:
                num_tiles_m = math.ceil(m / BLOCK_M)
                num_tiles_n = math.ceil(n / BLOCK_N)
                ctas_compatible = (
                    num_tiles_m % 2 == 0 and (num_tiles_m * num_tiles_n) % 2 == 0
                )
                if not ctas_compatible:
                    template_kwargs["NUM_CTAS"] = 1
                    NUM_CTAS = 1

            # In allow mode, validate SMEM with margin before yielding heuristic
            # config — autotuning configs provide fallback if this is skipped.
            # In force mode, always yield since it's the only config.
            BLOCK_K = template_kwargs["BLOCK_K"]
            skip_heuristic = is_allow_mode and not self._is_valid_config_with_margin(
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                heuristic_config["NUM_SMEM_BUFFERS"],
                heuristic_config["NUM_TMEM_BUFFERS"],
                heuristic_config["NUM_MMA_GROUPS"],
                NUM_CTAS,
                heuristic_config["EPILOGUE_SUBTILE"],
            )
            heuristic_yielded = False
            if skip_heuristic:
                log.debug(
                    "Heuristic config (%d,%d,%d) exceeds SMEM with margin, skipping",
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                )
            else:
                # Add ctas_per_cga for NUM_CTAS=2 mode
                if NUM_CTAS == 2:
                    template_kwargs["ctas_per_cga"] = (2, 1, 1)

                SPLIT_K = template_kwargs.get("SPLIT_K", 1)
                if SPLIT_K > 1:
                    # Split-K writes fp32 partials to workspace via its own TMA path;
                    # TMA_EPILOGUE_STORE controls the non-split-K output store path.
                    template_kwargs["TMA_EPILOGUE_STORE"] = 0
                    yield template_kwargs
                    heuristic_yielded = True
                else:
                    for tma_kwargs in self._yield_tma_variants(template_kwargs):
                        yield tma_kwargs
                        heuristic_yielded = True

            # In force mode, return after yielding the heuristic config.
            # Fall through to autotuning if heuristic was skipped (e.g.
            # config exceeds SMEM with TMA epilogue buffers).
            if not is_allow_mode and heuristic_yielded:
                return

        # Yield autotuning configs from self.mm_configs via the base class pipeline
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs,
            op_name,
        ):
            # Add TLX-specific defaults, allowing override from config
            template_kwargs = {
                **template_kwargs,
                "GROUP_SIZE_M": template_kwargs.get("GROUP_SIZE_M", 8),
                "NUM_SMEM_BUFFERS": template_kwargs.get("NUM_SMEM_BUFFERS", 3),
                "NUM_TMEM_BUFFERS": template_kwargs.get("NUM_TMEM_BUFFERS", 2),
                "EPILOGUE_SUBTILE": template_kwargs.get("EPILOGUE_SUBTILE", 1),
                "NUM_MMA_GROUPS": template_kwargs.get("NUM_MMA_GROUPS", 1),
                "NUM_CTAS": template_kwargs.get("NUM_CTAS", 1),
                "SPLIT_K": template_kwargs.get("SPLIT_K", 1),
                "INTERLEAVE_EPILOGUE": template_kwargs.get("INTERLEAVE_EPILOGUE", 0),
            }
            template_kwargs["BLOCK_M_SPLIT"] = (
                template_kwargs["BLOCK_M"] // template_kwargs["NUM_MMA_GROUPS"]
            )
            template_kwargs["slice_size"] = (
                template_kwargs["BLOCK_N"] // template_kwargs["EPILOGUE_SUBTILE"]
            )

            BLOCK_M = template_kwargs["BLOCK_M"]
            BLOCK_N = template_kwargs["BLOCK_N"]
            BLOCK_K = template_kwargs["BLOCK_K"]
            NUM_CTAS = template_kwargs["NUM_CTAS"]

            # Skip configs where block sizes >= problem dimensions (causes accuracy issues)
            if BLOCK_N >= n or BLOCK_K >= k:
                continue

            # Interleaved epilogue hardcodes two MMA groups (buf_idx_0/1).
            if (
                template_kwargs["INTERLEAVE_EPILOGUE"]
                and template_kwargs["NUM_MMA_GROUPS"] != 2
            ):
                continue

            # 2-CTA cluster configs cause illegal memory access during
            # autotuning for tall-M shapes.  The heuristic path selects
            # 2-CTA with shape guards; the autotuning pool uses 1-CTA.
            if NUM_CTAS == 2:
                template_kwargs = {**template_kwargs, "NUM_CTAS": 1}
                NUM_CTAS = 1

            # Re-validate SMEM with actual NUM_CTAS (may have changed above).
            # Use a safety margin to account for epilogue fusion overhead that
            # the static estimate doesn't capture (alignment, extra barriers).
            NUM_SMEM_BUFFERS = template_kwargs["NUM_SMEM_BUFFERS"]
            NUM_TMEM_BUFFERS = template_kwargs["NUM_TMEM_BUFFERS"]
            NUM_MMA_GROUPS = template_kwargs["NUM_MMA_GROUPS"]
            EPILOGUE_SUBTILE = template_kwargs["EPILOGUE_SUBTILE"]
            if not self._is_valid_config_with_margin(
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                NUM_SMEM_BUFFERS,
                NUM_TMEM_BUFFERS,
                NUM_MMA_GROUPS,
                NUM_CTAS,
                EPILOGUE_SUBTILE,
            ):
                continue

            # Add ctas_per_cga for NUM_CTAS=2 mode
            if NUM_CTAS == 2:
                template_kwargs = {
                    **template_kwargs,
                    "ctas_per_cga": (2, 1, 1),
                }

            # Yield TMA epilogue store variant(s)
            yield from self._yield_tma_variants(template_kwargs)

    @staticmethod
    def _yield_tma_variants(
        template_kwargs: dict[str, Any],
    ) -> Generator[dict[str, Any], None, None]:
        """Yield TMA_EPILOGUE_STORE=1 (preferred) or TMA=0 fallback.

        Upstream TLX always uses TMA descriptor stores for the epilogue.
        tl.store is incompatible with NUM_CTAS=2 (MultiCTAReduction pass
        can't distribute direct stores across CTAs), so 2-CTA configs
        must use TMA and are skipped if they don't fit in SMEM.

        For 1-CTA configs, fall back to TMA=0 (tl.store) when TMA=1
        exceeds SMEM — this can happen when epilogue fusion adds SMEM
        for fused tensors (e.g. bias) beyond what the base estimate covers."""
        config_dict = {
            "BLOCK_SIZE_M": template_kwargs["BLOCK_M"],
            "BLOCK_SIZE_N": template_kwargs["BLOCK_N"],
            "BLOCK_SIZE_K": template_kwargs["BLOCK_K"],
            "NUM_SMEM_BUFFERS": template_kwargs["NUM_SMEM_BUFFERS"],
            "NUM_TMEM_BUFFERS": template_kwargs["NUM_TMEM_BUFFERS"],
            "NUM_MMA_GROUPS": template_kwargs["NUM_MMA_GROUPS"],
            "EPILOGUE_SUBTILE": template_kwargs["EPILOGUE_SUBTILE"],
            "NUM_CTAS": template_kwargs["NUM_CTAS"],
            "SPLIT_K": template_kwargs.get("SPLIT_K", 1),
        }
        num_ctas = template_kwargs.get("NUM_CTAS", 1)
        if num_ctas == 1:
            # For 1-CTA, use a margin to account for epilogue fusion SMEM
            # (e.g. bias buffer ~4-5KB) that the base formula doesn't capture.
            # Without this, TMA=1 compiles OK during benchmarking (no fusion)
            # but can fail after fusion adds SMEM in the final codegen.
            _TMA_SMEM_MARGIN = 8192
            if _is_config_valid(
                config_dict, tma_epilogue_store=True, smem_margin=_TMA_SMEM_MARGIN
            ):
                yield {**template_kwargs, "TMA_EPILOGUE_STORE": 1}
            else:
                yield {**template_kwargs, "TMA_EPILOGUE_STORE": 0}
        else:
            # 2-CTA requires TMA (tl.store incompatible with MultiCTAReduction)
            if _is_config_valid(config_dict, tma_epilogue_store=True):
                yield {**template_kwargs, "TMA_EPILOGUE_STORE": 1}


@register_template_heuristic(
    amd_addmm_warppipe_template.uid, "cuda", register=IS_ROCM, op_name="addmm"
)
class ROCmAddMMWarpPipeTemplateConfigHeuristic(
    AddMMConfigMixin, ROCmMMTemplateConfigHeuristic
):
    """TLX warp-pipelined addmm heuristic for ROCm (col-major B, MI350X/gfx950).

    Hand-pipelined (num_stages=1): async-prefetches NUM_BUFFERS K-tiles into multi-buffered
    LDS and overlaps the loads with the MFMA via tlx.warp_pipeline_stage. Configs are the
    standalone MI350X winners. Correctness requires K_ITERS > NUM_BUFFERS, so a config is
    emitted only when cdiv(K, BLOCK_K) > NUM_BUFFERS is statically known (also declines
    dynamic/unknown K). The kernel needs col-major B; adjust_kernel_inputs enforces it
    (the col-major prep that used to live in OSS tuned_addmm).
    """

    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, NUM_BUFFERS)
    # BLOCK_N=256 tiles mirror the amd_addmm_glu tutorial winners for the M=1024
    # regime; on gfx950 they beat the BLOCK_N<=128 tiles on large-N shapes (e.g.
    # 1024x22272x1024 reaches ~98% of rocBLAS, up from ~92%). LDS: (128x256x64,NB2)
    # = 96KB, (128x256x32,NB3) = 72KB -- both fit gfx950 (256x256x64 does not).
    # (128x256x64,NB3) = 144KB fits gfx950's 160KB (occupancy 1); it is the deeper-
    # prefetch tile that won the standalone split-K sweep on low-occupancy large-K
    # (e.g. 1024x6144x22272 at SK=4), which the NB=2 variant alone could not reach.
    WARPPIPE_CONFIGS = [
        (64, 64, 128, 8, 8, 3),
        (64, 64, 64, 8, 8, 3),
        (128, 128, 64, 8, 8, 2),
        (64, 128, 64, 8, 8, 2),
        (128, 256, 64, 8, 8, 2),
        (128, 256, 64, 8, 8, 3),
        (128, 256, 32, 8, 8, 3),
    ]

    def adjust_kernel_inputs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> KernelInputs:
        # The warp-pipe kernel async-loads B as (BLOCK_N, BLOCK_K) with K contiguous, so it
        # requires col-major B (stride_bk == 1). REQUIRE it here (free for an nn.Linear weight)
        # rather than gate on the current -- possibly unfrozen -- stride; this replaces the
        # require_stride_order that used to live in OSS tuned_addmm.
        from torch._inductor.ir import ExternKernel

        kernel_inputs = super().adjust_kernel_inputs(kernel_inputs, op_name)
        if not isinstance(kernel_inputs, MMKernelInputs):
            return kernel_inputs
        nodes = list(kernel_inputs.nodes())
        mat2_idx = kernel_inputs._mat2_idx
        nodes[mat2_idx] = ExternKernel.require_stride_order(nodes[mat2_idx], [0, 1])
        return MMKernelInputs(
            nodes,
            scalars=kernel_inputs._scalars,
            out_dtype=kernel_inputs.out_dtype(),
            mat1_idx=kernel_inputs._mat1_idx,
            mat2_idx=mat2_idx,
        )

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        import sympy
        from torch._inductor.virtualized import V

        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")
        # The warp-pipe win is on fp16/bf16 latency-bound addmm; skip other dtypes. dtype()
        # defaults to input 0 = the bias for addmm, so check mat1 (the matrix operand) instead
        # so an fp32 bias doesn't wrongly decline a bf16/fp16 addmm.
        if kernel_inputs.dtype(kernel_inputs._mat1_idx) not in (
            torch.float16,
            torch.bfloat16,
        ):
            return
        m, n, k = kernel_inputs.mnk_symbolic()
        out_dtype = kernel_inputs.out_dtype()
        sizevars = V.graph.sizevars
        # async_load lowers to a 32-bit block pointer, so this template is only valid when M*K
        # and N*K fit int32. Use optimization_hint (not statically_known) so dynamic/symbolic M
        # is allowed via the example-input hint -- the int32 buffer-index cast keeps it correct
        # under AOTI dynamic shapes.
        int32_max = 2**31 - 1
        if not (
            _sizevar_hint(sizevars, m * k, int32_max) < int32_max
            and _sizevar_hint(sizevars, n * k, int32_max) < int32_max
        ):
            return
        # 16-byte (128-bit transaction) alignment: the fp16/bf16 padded direct-to-LDS async_copy
        # lowers to 128-bit loads, so each row start must be 16-byte aligned -- i.e. the row stride
        # K*itemsize bytes must be a multiple of 16 (K % 8 == 0 for fp16/bf16). Without this guard
        # an unaligned K reaches the template and fails at async_copy legalization (T280910119);
        # decline up front so it falls back to aten cleanly. Also declines dynamic/unknown K.
        itemsize = torch.finfo(kernel_inputs.dtype(kernel_inputs._mat1_idx)).bits // 8
        if not sizevars.statically_known_true(sympy.Eq(sympy.Mod(k * itemsize, 16), 0)):
            return
        num_xcds = _amd_num_xcds()
        # split-K only helps grids that leave CUs idle. NUM_SMS is the device CU count
        # (get_num_sms() maps to multi_processor_count = CUs on ROCm; 256 on gfx950/MI350X);
        # a grid with fewer MN tiles than this is undersaturated and benefits from
        # partitioning K across extra programs (summed by _reduce_k_kernel).
        NUM_SMS = get_num_sms()
        # split-K bypasses store_output's bias epilogue; the reduction re-adds only a
        # plain bias (i.e. alpha*(A@B) + beta*bias with alpha=beta=1). Restrict split-K
        # to that case -- unit-scalar addmm and plain mm both qualify. sympy Symbol == 1
        # returns a plain False, so this stays safe for symbolic scalars.
        scalars = getattr(kernel_inputs, "_scalars", None) or {}
        allow_split_k = scalars.get("alpha", 1) == 1 and scalars.get("beta", 1) == 1
        m_hint = sizevars.optimization_hint(m, fallback=NUM_SMS)
        n_hint = sizevars.optimization_hint(n, fallback=NUM_SMS)
        for (
            block_m,
            block_n,
            block_k,
            group_m,
            num_warps,
            num_buffers,
        ) in self.WARPPIPE_CONFIGS:
            # MFMA requires block_m/block_n be multiples of matrix_instr_nonkdim (16).
            if block_m % 16 != 0 or block_n % 16 != 0:
                continue
            # warp-pipeline correctness guard: K_ITERS > NUM_BUFFERS.
            if not sizevars.statically_known_true(sympy.Gt(k, num_buffers * block_k)):
                continue
            # SPLIT_K=1 (plain data-parallel) is always offered. Add split candidates
            # only for undersaturated grids, and only when each split still runs
            # K_ITERS/SPLIT_K > NUM_BUFFERS iters (statically checked via k > NB*BK*SK)
            # so the warp-pipeline prologue/drain stays well-formed.
            tiles = ((m_hint + block_m - 1) // block_m) * (
                (n_hint + block_n - 1) // block_n
            )
            split_ks = [1]
            if allow_split_k and tiles < NUM_SMS:
                for sk in (2, 4, 8):
                    # Cap at ~4 waves. For memory-bound large-K the sweet spot is
                    # often >1 wave (more concurrent HBM requests -> higher effective
                    # bandwidth); the standalone sweep's winners filled to 300%
                    # (e.g. 1024x6144x22272 at SK=4 -> 768 wg / 3 waves). A 2-wave cap
                    # excluded exactly those, so autotune fell back to a slower SK=1.
                    if tiles * sk > 4 * NUM_SMS:
                        break
                    # correctness: the balanced K-partition gives each split
                    # base = K_ITERS // SK iters; require base > NUM_BUFFERS so every
                    # split's warp-pipeline prologue/drain is well-formed. Sufficient
                    # static condition on K: k > (NUM_BUFFERS + 1) * BLOCK_K * SK.
                    if sizevars.statically_known_true(
                        sympy.Gt(k, (num_buffers + 1) * block_k * sk)
                    ):
                        split_ks.append(sk)
            for split_k in split_ks:
                triton_config = self.triton_config(
                    1,  # num_stages=1: TLX is hand-pipelined; auto software-pipelining must be off
                    num_warps,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=group_m,
                    NUM_BUFFERS=num_buffers,
                    NUM_XCDS=num_xcds,
                    SPLIT_K=split_k,
                    matrix_instr_nonkdim=16,
                    waves_per_eu=0,
                    kpack=get_default_kpack(block_k),
                )
                yield self._convert_config_to_template_kwargs(
                    triton_config, m, n, k, out_dtype
                )


@register_template_heuristic(
    amd_bmm_warppipe_template.uid, "cuda", register=IS_ROCM, op_name="bmm"
)
class ROCmBMMWarpPipeTemplateConfigHeuristic(ROCmMMTemplateConfigHeuristic):
    """TLX warp-pipelined bmm heuristic for ROCm (MI350X/gfx950).

    Same warp-pipe core as the addmm (async_load prefetch into multi-buffered LDS + MFMA via
    tlx.warp_pipeline_stage), plus a batch axis and a per-batch int64 base advance. No bias, no
    col-major transpose (bmm B is [BATCH,K,N] row-major), no split-K -- a plain data-parallel
    baseline for Inductor autotune iteration.

    Dual path selected by K's 16-byte alignment (the template's USE_ASYNC constexpr):
      * (K*itemsize) % 16 == 0 (K % 8 for fp16/bf16): USE_ASYNC=1, the direct-to-LDS async_load
        warp-pipe (needs K_ITERS = K // BLOCK_K >= NUM_BUFFERS for a well-formed prologue; the
        K % BLOCK_K remainder is a sync-tail).
      * otherwise (unaligned K -- the direct-to-LDS async_copy cannot legalize on CDNA4 -- or
        dynamic K): USE_ASYNC=0, the register-path fallback (tl.load->tl.dot, auto-pipelined),
        correct for ANY K (T280910119). Common gate (fp16/bf16 only): per-batch M*K and N*K fit
        int32 (the within-batch offset is int32 on both paths).
    """

    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, NUM_BUFFERS)
    WARPPIPE_CONFIGS = [
        (256, 256, 64, 16, 8, 2),
        (256, 128, 64, 8, 8, 2),
        (128, 256, 64, 8, 8, 2),
        (128, 128, 64, 8, 8, 3),
        (128, 128, 128, 8, 8, 2),
        (128, 64, 128, 8, 8, 3),
        (64, 128, 64, 8, 8, 3),
        (64, 64, 128, 8, 4, 3),
        (64, 64, 64, 8, 4, 3),
        (32, 128, 128, 4, 4, 3),
        (32, 256, 128, 4, 4, 2),
        (16, 256, 128, 4, 4, 3),
    ]

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        import sympy
        from torch._inductor.virtualized import V

        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")
        if kernel_inputs.dtype(kernel_inputs._mat1_idx) not in (
            torch.float16,
            torch.bfloat16,
        ):
            return
        m, n, k = kernel_inputs.mnk_symbolic()
        out_dtype = kernel_inputs.out_dtype()
        sizevars = V.graph.sizevars
        int32_max = 2**31 - 1
        # per-batch offsets must fit int32 (the batch offset is applied separately in int64).
        if not (
            _sizevar_hint(sizevars, m * k, int32_max) < int32_max
            and _sizevar_hint(sizevars, n * k, int32_max) < int32_max
        ):
            return
        # DUAL PATH by K alignment (the USE_ASYNC constexpr picks the template branch):
        #  * (K*itemsize) % 16 == 0 (K % 8 for fp16/bf16, 16-byte-aligned rows) -> USE_ASYNC=1, the
        #    fast direct-to-LDS async_load warp-pipe.
        #  * otherwise (e.g. odd K -- which the direct-to-LDS async_copy cannot legalize on CDNA4 --
        #    or dynamic/unknown K, treated as unaligned) -> USE_ASYNC=0, the register-path fallback
        #    (tl.load->registers->tl.dot, auto-pipelined; ~0.76x rocBLAS, T280910119). This is
        #    correct for ANY K, so unaligned-K bmm still gets a Triton candidate rather than only aten.
        itemsize = torch.finfo(kernel_inputs.dtype(kernel_inputs._mat1_idx)).bits // 8
        use_async = sizevars.statically_known_true(
            sympy.Eq(sympy.Mod(k * itemsize, 16), 0)
        )
        num_xcds = _amd_num_xcds()
        for (
            block_m,
            block_n,
            block_k,
            group_m,
            num_warps,
            num_buffers,
        ) in self.WARPPIPE_CONFIGS:
            # MFMA requires block_m/block_n be multiples of matrix_instr_nonkdim (16).
            if block_m % 16 != 0 or block_n % 16 != 0:
                continue
            # async path only: its prologue prefetches NUM_BUFFERS full K-tiles, so require
            # K_ITERS = K // BLOCK_K >= NB. The register path has no prologue -> it takes any K.
            if use_async and not sizevars.statically_known_true(
                sympy.Ge(k, num_buffers * block_k)
            ):
                continue
            triton_config = self.triton_config(
                # async is hand-pipelined (num_stages=1, auto software-pipelining off); the register
                # path relies on the auto-pipeliner (num_stages=3) to overlap its tl.loads.
                1 if use_async else 3,
                num_warps,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_M=group_m,
                NUM_BUFFERS=num_buffers,
                NUM_XCDS=num_xcds,
                USE_ASYNC=use_async,
                matrix_instr_nonkdim=16,
                waves_per_eu=0,
                kpack=get_default_kpack(block_k),
            )
            yield self._convert_config_to_template_kwargs(
                triton_config, m, n, k, out_dtype
            )


@register_template_heuristic(
    amd_bmm_warppipe_persistent_template.uid, "cuda", register=IS_ROCM, op_name="bmm"
)
class ROCmBMMWarpPipePersistentTemplateConfigHeuristic(
    ROCmBMMWarpPipeTemplateConfigHeuristic
):
    """Persistent variant of the warp-pipe bmm heuristic. Injects NUM_SMS (the persistent grid
    launches NUM_SMS programs that stride over tiles) and, because the persistent template has only
    the async path, skips the non-persistent parent's USE_ASYNC=0 register configs (unaligned K).
    Both async variants compete in the aten.bmm autotune -- persistent wins on large-K (amortizes
    launch/setup), non-persistent on thin-M/small-tile shapes."""

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        num_sms = get_num_sms()
        for kwargs in super()._get_template_configs_impl(kernel_inputs, op_name):
            # The persistent template is async-only (no register-path branch), so skip the parent's
            # USE_ASYNC=0 register configs (emitted for unaligned K); the non-persistent template
            # covers those. Only aligned-K (USE_ASYNC=1) configs get a persistent variant.
            if not kwargs.get("USE_ASYNC", True):
                continue
            kwargs["NUM_SMS"] = num_sms
            yield kwargs


@register_template_heuristic(
    blackwell_gemm_ws_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
class TemplateGemmWSConfigHeuristic(TLXMatmulWSConfigMixin, CUDAConfigHeuristic):
    """
    Blackwell TLX Warp-Specialized GEMM template from tritonbench.

    Uses MMA groups and persistent kernel for optimized performance on Blackwell GPUs.
    """

    def should_run(self, inputs: KernelInputs) -> bool:
        """
        Override to allow TLX templates to run without max_autotune when tlx_mode is set.
        """
        if config.triton.tlx_mode in ("allow", "force"):
            return True
        return super().should_run(inputs)

    def _get_extra_config_key_and_kwargs(
        self, conf: GemmConfig
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return get_tlx_config_key_and_kwargs(conf)

    def __init__(self) -> None:
        super().__init__()
        # Autotuning configs for "allow" mode: these compete alongside the heuristic
        # config against cublas/triton. Selected via benchmarking on representative
        # workloads — only configs that beat cublas for at least one shape are included.
        # Tuple format: (BM, BN, BK, smem_num, tmem_num, num_mma_groups, epilogue_subtile, num_ctas)
        # Upstream TLX candidate table — used as the autotuning pool.
        # Format: (BM, BN, BK, smem_num, tmem_num, mma_groups, subtile, num_ctas)
        _AUTOTUNE_CONFIGS = [
            (256, 128, 128, 2, 2, 2, 1, 2),
            (256, 256, 64, 3, 1, 2, 4, 1),
            (128, 64, 128, 4, 3, 2, 1, 1),
            (256, 128, 64, 5, 2, 2, 4, 2),
            (128, 256, 64, 4, 2, 1, 2, 2),
            (256, 64, 128, 5, 2, 2, 4, 2),
            (128, 64, 128, 5, 2, 2, 1, 2),
            (256, 64, 128, 5, 2, 2, 8, 1),
            (128, 256, 64, 3, 2, 1, 2, 1),
            (128, 128, 64, 4, 2, 1, 2, 1),
            (256, 128, 64, 3, 1, 2, 2, 1),
            (128, 64, 64, 5, 2, 1, 1, 1),
            (64, 128, 64, 5, 2, 1, 1, 1),
            (64, 64, 64, 6, 2, 1, 1, 1),
        ]
        self.mm_configs = [
            TlxGemmConfig(
                BM,
                BN,
                BK,
                1,  # num_stages
                4,  # num_warps
                group_size_m=8,
                smem_num=s,
                tmem_num=t,
                num_mma_groups=m,
                epilogue_subtile=subtile,
                num_ctas=num_ctas,
                split_k=1,
            )
            for BM, BN, BK, s, t, m, subtile, num_ctas in _AUTOTUNE_CONFIGS
            # Prune invalid configs based on hardware constraints
            if TLXMatmulWSConfigMixin._is_valid_config(
                block_m=BM,
                block_n=BN,
                block_k=BK,
                num_smem_buffers=s,
                num_tmem_buffers=t,
                num_mma_groups=m,
                num_ctas=num_ctas,
                epilogue_subtile=subtile,
            )
        ]


# export tuple of CUDA options in TLX
tlx_only_cuda_options = ["ctas_per_cga"]


def get_tlx_config_key_and_kwargs(
    conf: GemmConfig,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Extract TLX-specific key fields and kwargs from a TlxGemmConfig.

    Returns:
        A tuple of (key_fields, kwargs) where:
        - key_fields: tuple of TLX-specific fields to add to deduplication key
        - kwargs: dict of TLX-specific kwargs to add to TritonConfig

    If the config is not a TlxGemmConfig, returns empty tuple and empty dict.
    """
    if not isinstance(conf, TlxGemmConfig):
        return (), {}

    key_fields = (
        conf.group_size_m,
        conf.smem_num,
        conf.tmem_num,
        conf.epilogue_subtile,
        conf.num_mma_groups,
        conf.num_ctas,
        conf.split_k,
        conf.interleave_epilogue,
    )
    kwargs = {
        "GROUP_SIZE_M": conf.group_size_m,
        "NUM_SMEM_BUFFERS": conf.smem_num,
        "NUM_TMEM_BUFFERS": conf.tmem_num,
        "EPILOGUE_SUBTILE": conf.epilogue_subtile,
        "NUM_MMA_GROUPS": conf.num_mma_groups,
        "NUM_CTAS": conf.num_ctas,
        "SPLIT_K": conf.split_k,
        "INTERLEAVE_EPILOGUE": conf.interleave_epilogue,
    }
    return key_fields, kwargs


# Use a factory to defer the import of TLXInductorChoices, avoiding
# circular import: template_heuristics/__init__ -> tlx -> choices ->
# InductorChoices -> template_heuristics/__init__
def _tlx_choices_factory():
    from .choices import TLXInductorChoices

    return TLXInductorChoices()


config.inductor_choices_class = _tlx_choices_factory

# ---------------------------------------------------------------------------
# Override TritonTemplateKernel to support async TMA store (TLX-specific).
#
# async_tma_store is not exposed in OSS.  The TMA_EPILOGUE_STORE template
# kwarg (set by _get_template_configs_impl above) flows through the meta
# dict.  The overrides below extract it in __init__, set the store mode in
# store_output, and dispatch to the TLX codegen in store.
# ---------------------------------------------------------------------------
from torch._inductor.codegen.triton import (
    BlockPtrOptions,
    DeferredLine,
    TensorDescriptorOptions,
)
from .codegen import codegen_async_tma_store
from torch._inductor.select_algorithm import TritonTemplate, TritonTemplateKernel
from torch._inductor.virtualized import V

# -- generate: inject split-K workspace_arg via the standard mechanism ------
# The workspace_arg must flow through generate() so the autotuning benchmark
# allocates the workspace tensor.  Creating it in __init__ is too late —
# generate() has already captured workspace_arg=None for the benchmark request.
_orig_tt_generate = TritonTemplate.generate


def _tlx_tt_generate(self, input_nodes, layout, *args, **kwargs):  # type: ignore[no-untyped-def]
    split_k = int(kwargs.get("SPLIT_K", 1))
    if split_k > 1:
        from torch._inductor.codegen.common import WorkspaceArg, WorkspaceZeroMode

        # SPLIT_K > 1 forces TMA_EPILOGUE_STORE=0, so the TMA descriptor
        # workspace (ws_ptr) from TMAWorkspaceMixin is unused.  Replace it
        # with the split-K fp32 partial-results workspace.
        # UNINITIALIZED (no zero-fill): every valid output element is written exactly
        # once -- one (tile, split) program per element, masked to [0,M)x[0,N) -- and
        # _reduce_k_kernel reads with the same mask, so a zeroed workspace is
        # unnecessary. ZERO_ON_CALL here would re-zero the full SPLIT_K*M*N fp32 buffer
        # every call (tens of MB), which dominates and erased the split-K win.
        kwargs["workspace_arg"] = WorkspaceArg(
            count=split_k * layout.size[0] * layout.size[1],
            zero_mode=WorkspaceZeroMode.UNINITIALIZED,
            device=layout.device,
            outer_name=WorkspaceArg.unique_name("split_k_ws_"),
            inner_name="split_k_ws",
            dtype=torch.float32,
        )
    return _orig_tt_generate(self, input_nodes, layout, *args, **kwargs)


TritonTemplate.generate = _tlx_tt_generate  # type: ignore[method-assign]

# -- __init__: extract TMA_EPILOGUE_STORE from meta, set tma_store=True -----
_orig_ttk_init = TritonTemplateKernel.__init__


def _tlx_ttk_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    meta = kwargs.get("meta", {})
    async_tma_store = bool(meta.get("TMA_EPILOGUE_STORE", 0))
    split_k = int(meta.get("SPLIT_K", 1))
    if async_tma_store:
        kwargs["tma_store"] = True

    _orig_ttk_init(self, *args, **kwargs)
    self.async_tma_store = async_tma_store
    self._tlx_split_k = split_k


TritonTemplateKernel.__init__ = _tlx_ttk_init  # type: ignore[method-assign]

# -- store_output: accept async_tma_store_buf_idx, set mode="async_tma" -----
_orig_store_output = TritonTemplateKernel.store_output


def _tlx_store_output(  # type: ignore[no-untyped-def]
    self, *args, async_tma_store_buf_idx=None, **kwargs
):
    if getattr(self, "async_tma_store", False):
        if async_tma_store_buf_idx is not None:
            V.kernel.async_tma_store_buf_idx = async_tma_store_buf_idx
        # Signal the store override to use async TMA mode instead of
        # the regular TMA mode that OSS store_output will select.
        self._tlx_async_tma_store_active = True
    try:
        return _orig_store_output(self, *args, **kwargs)
    finally:
        self._tlx_async_tma_store_active = False


# Jinja template_env uses fn.__name__ to build the dict key — preserve it.
_tlx_store_output.__name__ = "store_output"
TritonTemplateKernel.store_output = _tlx_store_output  # type: ignore[method-assign]

# -- store: intercept TMA mode when async TMA is active --------------------
_orig_tk_store = TritonTemplateKernel.store


def _tlx_store(self, name, index, value, mode=None):  # type: ignore[no-untyped-def]
    if mode == "tma" and getattr(self, "_tlx_async_tma_store_active", False):
        # Redirect from regular TMA store to async TMA store.
        var = self.args.output(name)
        original_index = index
        dtype = V.graph.get_dtype(name)

        tma_compatibility_checker = self.tma_compatibility_checker_cls(
            self,
            dtype,
            for_store=True,
            force=True,
        )
        indexing = self.indexing(
            index,
            dense_indexing=True,
            block_ptr=False,
            tma_compatibility_checker=tma_compatibility_checker,
        )

        if hasattr(indexing, "index") and self._has_stride1_on_rdim(indexing.index):
            self.stores_with_contiguous_rdim.append(name)

        is_inplace = name in self.args.inplace_buffers
        is_broadcasted = self.is_broadcasted(original_index)
        if is_inplace and is_broadcasted:
            self.stores.writeline(DeferredLine(name, "tl.debug_barrier()"))

        if not isinstance(indexing, (BlockPtrOptions, TensorDescriptorOptions)):
            # Output indexing isn't TMA-compatible — fall back to regular store.
            return _orig_tk_store(self, name, index, value, mode=mode)
        block_descriptor, _other = self.codegen_block_ptr(name, var, indexing)
        codegen_async_tma_store(self, name, indexing, block_descriptor, value)
        return
    return _orig_tk_store(self, name, index, value, mode=mode)


TritonTemplateKernel.store = _tlx_store  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# compute_epilogue: run fused epilogue ops without emitting a store.
#
# When TMA_EPILOGUE_STORE is active, the template needs to:
#   1. Run the fused epilogue (relu, bias add, etc.) to get the final value
#   2. Manually TMA-store that value via tlx.async_descriptor_store
#
# The standard store_output can't do this because V.ops.store targets the
# template's intermediate buffer (which is removed when fused), while the
# epilogue nodes emit tl.store to the final buffer — bypassing TMA entirely.
#
# compute_epilogue solves this by:
#   - Using the same index setup as store_output (tma_store=True path)
#   - Running epilogue node codegen but redirecting stores to variable
#     assignment ({result_name} = {value}) instead of tl.store
#   - Returning the hook placeholder so the template can emit TMA store code
#
# output_ptr: resolves the correct output buffer for TMA descriptor creation.
# When epilogue fusion is active, the TMA descriptor must point to the FINAL
# output buffer (e.g., relu output), not the template's intermediate output
# (e.g., raw mm result).
# ---------------------------------------------------------------------------

import itertools

from torch.utils._ordered_set import OrderedSet


def _tlx_output_ptr(self):  # type: ignore[no-untyped-def]
    """Get the output pointer for TMA descriptor creation.

    When epilogue fusion is active, returns the final output buffer pointer
    (set by codegen_template_body via _final_output_name) so TMA descriptors
    write directly to the fused output.
    """
    name = getattr(self, "_final_output_name", self.output_node.get_name())
    return self.args.output(name)


_tlx_output_ptr.__name__ = "output_ptr"
TritonTemplateKernel.output_ptr = _tlx_output_ptr  # type: ignore[method-assign]


def _tlx_get_compute_epilogue_subgraph_name(self, i):  # type: ignore[no-untyped-def]
    return f"<COMPUTE_EPILOGUE_{i}>"


TritonTemplateKernel._get_compute_epilogue_subgraph_name = (  # type: ignore[method-assign]
    _tlx_get_compute_epilogue_subgraph_name
)


def _tlx_get_compute_epilogue_count(self):  # type: ignore[no-untyped-def]
    total = next(self._compute_epilogue_ctr)
    self._compute_epilogue_ctr = itertools.count(start=total - 1, step=1)
    return total


TritonTemplateKernel._tlx_get_compute_epilogue_count = (  # type: ignore[method-assign]
    _tlx_get_compute_epilogue_count
)

# Patch __init__ to add compute_epilogue counter
_orig_ttk_init_for_ctr = TritonTemplateKernel.__init__


def _tlx_ttk_init_with_ctr(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    _orig_ttk_init_for_ctr(self, *args, **kwargs)
    self._compute_epilogue_ctr = itertools.count()


TritonTemplateKernel.__init__ = _tlx_ttk_init_with_ctr  # type: ignore[method-assign]


def _tlx_compute_epilogue(  # type: ignore[no-untyped-def]
    self,
    indices,
    val,
    result_name="fused_result",
    indent_width=4,
    val_shape=None,
):
    """Apply epilogue fusion ops and assign result to result_name, without emitting a store.

    Used by templates that handle the store themselves (e.g., TMA async store).
    Same index setup as store_output with block_indexing=True, tma_store=True.
    """
    import sympy
    from torch._inductor.codegen.common import OpOverrides
    from torch._inductor.utils import sympy_dot, triton_type_to_torch

    subgraph_name = self._get_compute_epilogue_subgraph_name(
        next(self._compute_epilogue_ctr)
    )
    with self.create_subgraph_body(subgraph_name, clear_cse=True):
        assert isinstance(indices, (list, tuple))
        assert isinstance(val, str)
        assert val_shape and len(val_shape) == 2, (
            "compute_epilogue requires a 2D val_shape"
        )
        assert self.template_mask is None

        indices = list(map(OpOverrides.paren, indices))
        index_symbols = [sympy.Symbol(x, integer=True) for x in indices]
        lengths = [V.graph.sizevars.simplify(s) for s in self.output_node.get_size()]
        assert len(indices) == len(lengths)

        output_layout = self.output_node.get_layout()
        self.template_out = val

        # Use the tma_store index setup path (same as store_output with
        # block_indexing=True and self.tma_store=True).
        intermediate_lines: list[str] = []
        epilogue_index_symbols: list[sympy.Symbol] = []
        val_shape_copy = list(val_shape)
        for i, range_tree in enumerate(self.range_trees[:-1]):
            name = range_tree.name
            symbol = range_tree.symbol()
            epilogue_index_symbols.append(symbol)
            lookup_output = range_tree.lookup(sympy.S.One, lengths[i])
            old_name = lookup_output.symbol()
            lookup_output.set_name(name)
            range_tree.var_list[range_tree.var_list.index(old_name)] = symbol
            range_val = range_tree.var_ranges[old_name]
            del range_tree.var_ranges[old_name]
            range_tree.var_ranges[symbol] = range_val
            intermediate_lines.extend(
                self._generate_index_from_tma_index(
                    name,
                    "xoffset" if name == "xindex" else "yoffset",
                    index_symbols[i],
                    val_shape[i],
                    i,
                    len(val_shape),
                    block_name=range_tree.symt.name,
                )
            )
            intermediate_lines.append(
                self._generated_mask_for_tma(
                    name,
                    self.size(None, i),
                    "xmask" if name == "xindex" else "ymask",
                )
            )
            val_shape_copy[i] = range_tree.symt.name
        val_shape = tuple(val_shape_copy)

        index_symbols = epilogue_index_symbols
        contiguous_index = sympy_dot(output_layout.stride, index_symbols)

        for line in intermediate_lines:
            self.body.writeline(line)

        self.template_out_shape = val_shape
        acc_dtype = (
            triton_type_to_torch(self.meta["ACC_TYPE"])
            if "ACC_TYPE" in self.meta
            else torch.float32
        )
        epilogue_args = [V.kernel.cse.namedvar(val, dtype=acc_dtype, shape=val_shape)]
        for input_node in itertools.chain(
            self.input_nodes[: self.prefix_args],
            self.input_nodes[len(self.input_nodes) - self.suffix_args :],
        ):
            input_node.freeze_layout()
            epilogue_arg = V.kernel.cse.generate(
                self.compute,
                input_node.make_loader()(index_symbols),
                dtype=acc_dtype,
                shape=input_node.get_size(),
            )
            epilogue_args.append(epilogue_arg)
            self.frozen_layouts_cnt += 1

        # Instead of V.ops.store, we store to the template output's CSE cache
        # (for store-to-load forwarding) and emit a variable assignment for
        # the final result.  Epilogue nodes codegen'd later into this subgraph
        # will pick up the accumulator from store_cache and apply their ops.
        # Their stores are redirected to assignments by _TLXComputeOnlyHandler.
        fused = self.epilogue_fn(*epilogue_args)
        V.kernel.cse.store_cache[self.output_node.get_name()] = fused
        self.body.writeline(f"{result_name} = {fused}")

        # Mark that the template output buffer was "stored" for CSE purposes
        self.store_buffer_names.add(self.output_node.get_name())

        # Save result_name so epilogue node stores can be redirected
        self._tlx_compute_epilogue_result_name = result_name
        self.codegen_body()

    return self._register_hook(
        subgraph_name, self._make_codegen_hook(subgraph_name, indent_width)
    )


_tlx_compute_epilogue.__name__ = "compute_epilogue"
TritonTemplateKernel.compute_epilogue = _tlx_compute_epilogue  # type: ignore[method-assign]

# -- codegen_template_body: wrap to set _final_output_name and handle
#    COMPUTE_EPILOGUE subgraphs --
# The epilogue-fusion codegen API below (codegen_template_body,
# _emit_post_kernel_code, _compute_fusion_metadata, get_unfused_epilogues) only
# exists on newer torch. On older wheels (e.g. the current ROCm nightly) these
# base methods are absent; grab them defensively so the module still imports and
# the core template path keeps working — the wrappers are only installed when the
# base method exists.
_orig_codegen_template_body = getattr(
    TritonTemplateKernel, "codegen_template_body", None
)


def _tlx_codegen_template_body(  # type: ignore[no-untyped-def]
    self,
    scheduling,
    template_node,
    epilogue_nodes,
    prologue_nodes,
    buf_name_to_prologue_group,
    prologue_preserves_zero_mask_fn,
    render,
):
    # Set _final_output_name so output_ptr() resolves to the fused output.
    if epilogue_nodes:
        last_names = epilogue_nodes[-1].get_buffer_names()
        if len(last_names) == 1:
            self._final_output_name = next(iter(last_names))

    # Wrap the original render to also handle COMPUTE_EPILOGUE subgraphs.
    orig_render = render

    def _render_with_compute_epilogue():
        result = orig_render()

        # After render, codegen epilogue nodes into COMPUTE_EPILOGUE subgraphs,
        # redirecting their stores to variable assignments.
        num_ce = self._tlx_get_compute_epilogue_count()
        for i in range(num_ce):
            subgraph_name = self._get_compute_epilogue_subgraph_name(i)
            result_name = getattr(
                self, "_tlx_compute_epilogue_result_name", "fused_result"
            )
            with self.set_subgraph_body(subgraph_name):
                # Redirect epilogue stores to variable assignments
                orig_store = self.store

                def _redirect_store(name, index, value, mode=None):  # type: ignore[no-untyped-def]
                    self.store_buffer_names.add(name)
                    self.cse.store_cache[name] = value
                    if name not in V.graph.removed_buffers:
                        self.compute.writeline(f"{result_name} = {value}")

                self.store = _redirect_store  # type: ignore[method-assign]
                try:
                    for node in epilogue_nodes:
                        node.codegen(self.split_and_set_ranges(node.get_ranges()))
                finally:
                    self.store = orig_store  # type: ignore[method-assign]
                self.cse.invalidate(OrderedSet())

        return result

    return _orig_codegen_template_body(
        self,
        scheduling,
        template_node,
        epilogue_nodes,
        prologue_nodes,
        buf_name_to_prologue_group,
        prologue_preserves_zero_mask_fn,
        _render_with_compute_epilogue,
    )


if _orig_codegen_template_body is not None:
    TritonTemplateKernel.codegen_template_body = _tlx_codegen_template_body  # type: ignore[method-assign]

# -- render: add compute_epilogue and output_ptr to template env --
_orig_render = TritonTemplateKernel.render


def _tlx_render(self, template, kwargs, record_input_dependent_tracked_event=False):  # type: ignore[no-untyped-def]
    # Register compute_epilogue and output_ptr as extra template env functions
    # so they're available in the jinja template.
    if getattr(self, "async_tma_store", False):
        self._register_extra_template_env_fns(self.compute_epilogue, self.output_ptr)
    elif getattr(self, "_tlx_split_k", 1) > 1:
        # split-K writes partials to split_k_ws and never store_output()s, so the
        # output arg would be pruned from the kernel signature -- but the autotuning
        # harness still passes `out` positionally (arg-count mismatch). Expose
        # output_ptr() so the template can reference it and keep it in the signature;
        # the real output is written later by _reduce_k_kernel.
        self._register_extra_template_env_fns(self.output_ptr)
    return _orig_render(self, template, kwargs, record_input_dependent_tracked_event)


TritonTemplateKernel.render = _tlx_render  # type: ignore[method-assign]

# -- _emit_post_kernel_code: emit split-K reduction kernel after main GEMM --
_orig_emit_post_kernel = getattr(
    TritonTemplateKernel, "_emit_post_kernel_code", None
)


def _tlx_emit_post_kernel_code(self, wrapper, kernel_name):  # type: ignore[no-untyped-def]
    split_k = getattr(self, "_tlx_split_k", 1)
    if split_k > 1 and self.workspace_arg is not None:
        from torch._inductor.codegen.triton import triton_type
        from .reduce_k import emit_reduce_k_call

        # Determine output buffer name and dtype
        out_name = self.output_node.get_name()
        out_dtype = self.output_node.get_layout().dtype
        output_triton_dtype = triton_type(out_dtype)

        # M and N from call_sizes
        from torch._inductor.codegen.wrapper import pexpr

        M_expr = pexpr(self.call_sizes[0])
        N_expr = pexpr(self.call_sizes[1])

        # addmm bias is applied by store_output's epilogue in the non-split path, which
        # split-K bypasses -> it must be re-added in the reduction. For addmm the bias is
        # the prefix input node (input_nodes[0], prefix_args=1); plain mm has none.
        bias_name = None
        stride_bias_m = 0
        stride_bias_n = 1
        if getattr(self, "prefix_args", 0) >= 1 and self.input_nodes:
            bias_node = self.input_nodes[0]
            bias_name = bias_node.get_name()
            bsize = bias_node.get_size()
            bstride = bias_node.get_layout().stride
            sizevars = V.graph.sizevars

            def _sh(expr):
                return int(sizevars.optimization_hint(expr, fallback=1))

            if len(bsize) == 1:
                # [N] bias broadcast over M
                stride_bias_m, stride_bias_n = 0, _sh(bstride[0])
            elif len(bsize) == 2:
                stride_bias_m = 0 if _sh(bsize[0]) == 1 else _sh(bstride[0])
                stride_bias_n = 0 if _sh(bsize[1]) == 1 else _sh(bstride[1])
            else:
                bias_name = None  # unexpected rank; skip (should not happen for addmm)

        emit_reduce_k_call(
            wrapper,
            ws_name=self.workspace_arg.outer_name,
            output_name=out_name,
            M_expr=M_expr,
            N_expr=N_expr,
            split_k=split_k,
            output_triton_dtype=output_triton_dtype,
            bias_name=bias_name,
            stride_bias_m=stride_bias_m,
            stride_bias_n=stride_bias_n,
        )
    _orig_emit_post_kernel(self, wrapper, kernel_name)


if _orig_emit_post_kernel is not None:
    TritonTemplateKernel._emit_post_kernel_code = _tlx_emit_post_kernel_code  # type: ignore[method-assign]


# -- _compute_fusion_metadata: disable epilogue fusion for SPLIT_K > 1 ------
#
# Split-K bypasses store_output (writes partials to workspace via tl.store),
# so scheduler-level epilogue fusion can't work — the fused ops would be
# silently dropped.  Instead, mark all epilogue nodes as "unfused" so the
# scheduler codegen's them as separate kernels after reduce_k.
_orig_compute_fusion_metadata = getattr(
    TritonTemplateKernel, "_compute_fusion_metadata", None
)


def _tlx_compute_fusion_metadata(  # type: ignore[no-untyped-def]
    self, scheduling, epilogue_nodes, prologue_nodes, buf_name_to_prologue_group
):
    split_k = getattr(self, "_tlx_split_k", 1)
    if split_k > 1 and epilogue_nodes:
        from collections import defaultdict

        self._epilogue_nodes_by_subgraph = defaultdict(list)
        self._unfused_epilogues = list(epilogue_nodes)
        self._prologue_sources = {}
        self._scheduling_ref = scheduling
    elif _orig_compute_fusion_metadata is not None:
        _orig_compute_fusion_metadata(
            self,
            scheduling,
            epilogue_nodes,
            prologue_nodes,
            buf_name_to_prologue_group,
        )


if _orig_compute_fusion_metadata is not None:
    TritonTemplateKernel._compute_fusion_metadata = _tlx_compute_fusion_metadata  # type: ignore[method-assign]

# -- get_unfused_epilogues: return split-K unfused epilogues -----------------
_orig_get_unfused_epilogues = getattr(
    TritonTemplateKernel, "get_unfused_epilogues", None
)


def _tlx_get_unfused_epilogues(self):  # type: ignore[no-untyped-def]
    unfused = getattr(self, "_unfused_epilogues", None)
    if unfused:
        return unfused
    return _orig_get_unfused_epilogues(self)


if _orig_get_unfused_epilogues is not None:
    TritonTemplateKernel.get_unfused_epilogues = _tlx_get_unfused_epilogues  # type: ignore[method-assign]

# -- call_kernel: codegen unfused epilogues after split-K reduce_k -----------
_orig_call_kernel = TritonTemplateKernel.call_kernel


def _tlx_call_kernel(self, name, node=None, deallocate_ws=True):  # type: ignore[no-untyped-def]
    _orig_call_kernel(self, name, node=node, deallocate_ws=deallocate_ws)
    # Codegen unfused epilogues after reduce_k (emitted in _emit_post_kernel_code)
    unfused = getattr(self, "_unfused_epilogues", [])
    scheduling = getattr(self, "_scheduling_ref", None)
    if unfused and scheduling is not None:
        for epi_node in unfused:
            scheduling.codegen_node(epi_node)


TritonTemplateKernel.call_kernel = _tlx_call_kernel  # type: ignore[method-assign]
