"""Test that the rule engine produces identical configs to the old imperative code.

Runs both implementations on a grid of (M, N, K) shapes and asserts every
config field matches (except Python-only fields: GROUP_SIZE_M, ctas_per_cga,
pre_hook — those are applied by the post-processor, so they're tested
end-to-end via the new get_heuristic_config as well).
"""

import math
import sys
from pathlib import Path

import pytest

# Make rule_engine importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from rule_engine import CandidateScorer, RuleEngine

# ---------------------------------------------------------------------------
# Original implementation (verbatim copy for comparison)
# ---------------------------------------------------------------------------

_SENTINEL_HOOK = object()  # stand-in for matmul_tma_set_block_size_hook


def _select_group_size_m(M, N, block_m):
    num_m_tiles = (M + block_m - 1) // block_m
    ratio = M / max(N, 1)
    if ratio > 10:
        return 1
    elif ratio < 0.1:
        return min(64, num_m_tiles)
    else:
        return min(8, num_m_tiles)


def original_get_heuristic_config(M, N, K, num_sms=148):
    MAX_SMEM = 232 * 1024
    MAX_TMEM = 256 * 1024

    mn_ratio = M / max(N, 1)
    is_tall_m = mn_ratio > 4
    is_tall_n = mn_ratio < 0.25

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

    if is_tall_m and is_gpu_saturated:
        arithmetic_intensity = K / max(min(M, N), 1)
        if arithmetic_intensity <= 1.5:
            return {
                "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
                "NUM_SMEM_BUFFERS": 2, "NUM_TMEM_BUFFERS": 2,
                "NUM_MMA_GROUPS": 2, "EPILOGUE_SUBTILE": 1,
                "NUM_CTAS": 2, "SPLIT_K": 1,
                "ctas_per_cga": (2, 1, 1), "pre_hook": _SENTINEL_HOOK,
            }
        else:
            if K > N * 2:
                return {
                    "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
                    "NUM_SMEM_BUFFERS": 2, "NUM_TMEM_BUFFERS": 1,
                    "NUM_MMA_GROUPS": 2, "EPILOGUE_SUBTILE": 4,
                    "NUM_CTAS": 2, "SPLIT_K": 1,
                    "ctas_per_cga": (2, 1, 1), "pre_hook": _SENTINEL_HOOK,
                }
            else:
                return {
                    "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
                    "NUM_SMEM_BUFFERS": 4, "NUM_TMEM_BUFFERS": 1,
                    "NUM_MMA_GROUPS": 2, "EPILOGUE_SUBTILE": 4,
                    "NUM_CTAS": 2, "SPLIT_K": 1,
                    "ctas_per_cga": (2, 1, 1), "pre_hook": _SENTINEL_HOOK,
                }

    if is_undersaturated:
        mn_product = M * N
        is_large_output = mn_product >= 1_000_000

        if is_large_output:
            block_m, block_n, block_k = 256, 128, 64
            k_tiles = math.ceil(K / block_k)
        else:
            block_m, block_n, block_k = 128, 64, 128
            k_tiles = math.ceil(K / block_k)

        split_k = 1
        for sk in [4, 2, 8]:
            if k_tiles >= sk and k_tiles // sk >= 4:
                split_k = sk
                break
        if split_k > 1:
            if is_large_output:
                return {
                    "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": block_k,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, block_m),
                    "NUM_SMEM_BUFFERS": 4, "NUM_TMEM_BUFFERS": 2,
                    "NUM_MMA_GROUPS": 2, "EPILOGUE_SUBTILE": 8,
                    "NUM_CTAS": 1, "SPLIT_K": split_k,
                    "ctas_per_cga": None, "pre_hook": _SENTINEL_HOOK,
                }
            else:
                return {
                    "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": block_k,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, block_m),
                    "NUM_SMEM_BUFFERS": 4, "NUM_TMEM_BUFFERS": 3,
                    "NUM_MMA_GROUPS": 2, "EPILOGUE_SUBTILE": 1,
                    "NUM_CTAS": 1, "SPLIT_K": split_k,
                    "ctas_per_cga": None, "pre_hook": _SENTINEL_HOOK,
                }

    if is_gpu_saturated:
        return {
            "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
            "NUM_SMEM_BUFFERS": 3, "NUM_TMEM_BUFFERS": 1,
            "NUM_MMA_GROUPS": 2, "EPILOGUE_SUBTILE": 4,
            "NUM_CTAS": 1, "SPLIT_K": 1,
            "ctas_per_cga": None, "pre_hook": _SENTINEL_HOOK,
        }

    # Fallback candidate scoring
    candidates = [
        (256, 128, 128, 2, 2, 2, 2, 1),
        (256, 256, 64, 1, 3, 1, 2, 4),
        (128, 64, 128, 1, 4, 3, 2, 1),
        (256, 128, 64, 2, 5, 2, 2, 4),
        (128, 256, 64, 2, 4, 2, 1, 2),
        (256, 64, 128, 2, 5, 2, 2, 4),
        (128, 64, 128, 2, 5, 2, 2, 1),
        (256, 64, 128, 1, 5, 2, 2, 8),
        (128, 256, 64, 1, 3, 2, 1, 2),
        (128, 128, 64, 1, 4, 2, 1, 2),
        (256, 128, 64, 1, 3, 1, 2, 2),
        (128, 64, 64, 1, 5, 2, 1, 1),
        (64, 128, 64, 1, 5, 2, 1, 1),
        (64, 64, 64, 1, 6, 2, 1, 1),
    ]

    def estimate_smem(bm, bn, bk, num_ctas, num_smem_buffers, num_mma_groups, epilogue_subtile):
        smem_a = bm * bk * 2 * num_smem_buffers
        smem_b = bk * (bn // num_ctas) * 2 * num_smem_buffers
        smem_epilog = bm * (bn // epilogue_subtile) * 2
        smem_barriers = num_smem_buffers * num_mma_groups * 8 * (2 if num_ctas == 2 else 1)
        return smem_a + smem_b + smem_epilog + smem_barriers

    def estimate_tmem(bm, bn, num_tmem_buffers):
        return bm * bn * 4 * num_tmem_buffers

    def compute_wave_score(bm, bn, num_ctas, split_k=1):
        ctas_m = (M + bm - 1) // bm
        ctas_n = (N + bn - 1) // bn
        ctas_m = ((ctas_m + num_ctas - 1) // num_ctas) * num_ctas
        total_ctas = ctas_m * ctas_n * split_k
        if total_ctas == 0:
            return float("inf"), 0, 0
        waves = (total_ctas + num_sms - 1) // num_sms
        fractional_waves = total_ctas / num_sms
        score = waves - fractional_waves
        return score, total_ctas, waves

    best_config = None
    best_score = float("inf")
    best_waves = float("inf")

    for bm, bn, bk, num_ctas, num_smem_buffers, num_tmem_buffers, num_mma_groups, epilogue_subtile in candidates:
        smem = estimate_smem(bm, bn, bk, num_ctas, num_smem_buffers, num_mma_groups, epilogue_subtile)
        if smem > MAX_SMEM:
            continue
        tmem = estimate_tmem(bm, bn, num_tmem_buffers)
        if tmem > MAX_TMEM:
            continue
        if bm // num_mma_groups > 128:
            continue
        if bm > M * 2 or bn > N * 2:
            continue

        score, total_ctas, waves = compute_wave_score(bm, bn, num_ctas)

        split_k = 1
        num_tiles_m_local = math.ceil(M / bm)
        num_tiles_n_local = math.ceil(N / bn)
        num_mn_tiles_local = num_tiles_m_local * num_tiles_n_local

        if num_mn_tiles_local < num_sms:
            k_tiles = math.ceil(K / bk)
            for sk in [8, 4, 2]:
                if k_tiles >= sk and k_tiles // sk >= 4:
                    sk_score, sk_ctas, sk_waves = compute_wave_score(bm, bn, num_ctas, sk)
                    if sk_score < score or (sk_score == score and sk_ctas > total_ctas):
                        score, total_ctas, waves, split_k = sk_score, sk_ctas, sk_waves, sk
                    break

        score_slack = 0.1
        adjusted_score = score

        if (adjusted_score < best_score - score_slack
                or (adjusted_score < best_score + score_slack and waves < best_waves)
                or (adjusted_score < best_score + score_slack and waves == best_waves and num_ctas > 1)):
            best_score = adjusted_score
            best_waves = waves
            best_config = {
                "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": _select_group_size_m(M, N, bm),
                "NUM_SMEM_BUFFERS": num_smem_buffers,
                "NUM_TMEM_BUFFERS": num_tmem_buffers,
                "NUM_MMA_GROUPS": num_mma_groups,
                "EPILOGUE_SUBTILE": epilogue_subtile,
                "NUM_CTAS": num_ctas, "SPLIT_K": split_k,
                "ctas_per_cga": (num_ctas, 1, 1) if num_ctas > 1 else None,
                "pre_hook": _SENTINEL_HOOK,
            }

    return best_config


# ---------------------------------------------------------------------------
# New implementation (rule engine)
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "tutorials" / "configs"
_rules = RuleEngine(_CONFIGS_DIR / "blackwell_gemm_ws_rules.yaml")
_candidates = CandidateScorer(_CONFIGS_DIR / "blackwell_gemm_ws_candidates.yaml")


def new_get_heuristic_config(M, N, K, num_sms=148):
    config = _rules.evaluate(M=M, N=N, K=K, num_sms=num_sms)
    if config is None:
        config = _candidates.evaluate(M=M, N=N, K=K, num_sms=num_sms)
    if config is None:
        return None
    # Apply post-processing (same as _gemm_post_process)
    block_m = config["BLOCK_SIZE_M"]
    num_ctas = config["NUM_CTAS"]
    config["GROUP_SIZE_M"] = _select_group_size_m(M, N, block_m)
    config["ctas_per_cga"] = (num_ctas, 1, 1) if num_ctas > 1 else None
    config["pre_hook"] = _SENTINEL_HOOK
    return config


# ---------------------------------------------------------------------------
# Comparison keys (all config fields except pre_hook, which is a function ref)
# ---------------------------------------------------------------------------

_CONFIG_KEYS = [
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
    "GROUP_SIZE_M", "NUM_SMEM_BUFFERS", "NUM_TMEM_BUFFERS",
    "NUM_MMA_GROUPS", "EPILOGUE_SUBTILE", "NUM_CTAS", "SPLIT_K",
    "ctas_per_cga",
]


# ---------------------------------------------------------------------------
# Test shapes: a representative grid
# ---------------------------------------------------------------------------

_M_VALUES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 3159809]
_N_VALUES = [32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
_K_VALUES = [64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 12800, 16384, 213120]
_NUM_SMS = 148


def _shape_grid():
    """Generate a representative grid of (M, N, K) shapes."""
    shapes = []
    for M in _M_VALUES:
        for N in _N_VALUES:
            for K in _K_VALUES:
                shapes.append((M, N, K))
    return shapes


@pytest.mark.parametrize("M,N,K", _shape_grid(), ids=lambda v: f"{v}" if not isinstance(v, tuple) else None)
def test_equivalence(M, N, K):
    old = original_get_heuristic_config(M, N, K, _NUM_SMS)
    new = new_get_heuristic_config(M, N, K, _NUM_SMS)

    if old is None:
        assert new is None, f"Shape ({M},{N},{K}): old returned None but new returned {new}"
        return

    assert new is not None, f"Shape ({M},{N},{K}): old returned config but new returned None"

    for key in _CONFIG_KEYS:
        old_val = old[key]
        new_val = new[key]
        assert old_val == new_val, (
            f"Shape ({M},{N},{K}): mismatch on {key}: "
            f"old={old_val}, new={new_val}\n"
            f"  old config: { {k: old[k] for k in _CONFIG_KEYS} }\n"
            f"  new config: { {k: new[k] for k in _CONFIG_KEYS} }"
        )
