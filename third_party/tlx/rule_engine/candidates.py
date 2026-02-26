"""CandidateScorer: load a candidates YAML file, filter, and score.

The candidates file lists concrete configs.  Each one is checked against
hardware constraints and then scored by wave efficiency.  The best-scoring
candidate wins.

Typical usage::

    scorer = CandidateScorer("blackwell_gemm_ws_candidates.yaml")
    config = scorer.evaluate(M=4096, N=4096, K=4096, num_sms=148)
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml


class CandidateScorer:
    """Scored candidate search driven by a YAML file."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        with open(path) as f:
            spec = yaml.safe_load(f)

        self.inputs: list[str] = spec["inputs"]
        self.hardware: dict[str, int] = spec["hardware"]
        self.configs: list[dict] = spec["configs"]

    # ------------------------------------------------------------------
    # Resource estimation (mirrors the Python originals exactly)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_smem(bm, bn, bk, num_ctas, num_smem_buffers, num_mma_groups, epilogue_subtile):
        smem_a = bm * bk * 2 * num_smem_buffers
        smem_b = bk * (bn // num_ctas) * 2 * num_smem_buffers
        smem_epilog = bm * (bn // epilogue_subtile) * 2
        smem_barriers = num_smem_buffers * num_mma_groups * 8 * (2 if num_ctas == 2 else 1)
        return smem_a + smem_b + smem_epilog + smem_barriers

    @staticmethod
    def _estimate_tmem(bm, bn, num_tmem_buffers):
        return bm * bn * 4 * num_tmem_buffers

    @staticmethod
    def _compute_wave_score(M, N, bm, bn, num_ctas, num_sms, split_k=1):
        """Compute wave efficiency score (lower is better).

        Returns (score, total_ctas, waves).
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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(self, **kwargs) -> dict | None:
        """Score all candidates and return the best one.

        Args:
            **kwargs: Input values (M, N, K, num_sms).

        Returns:
            A config dict for the best candidate, or ``None`` if every
            candidate is filtered out.
        """
        M = kwargs["M"]
        N = kwargs["N"]
        K = kwargs["K"]
        num_sms = kwargs["num_sms"]
        MAX_SMEM = self.hardware["MAX_SMEM"]
        MAX_TMEM = self.hardware["MAX_TMEM"]

        best_config = None
        best_score = float("inf")
        best_waves = float("inf")

        for cfg in self.configs:
            bm = cfg["BLOCK_SIZE_M"]
            bn = cfg["BLOCK_SIZE_N"]
            bk = cfg["BLOCK_SIZE_K"]
            num_ctas = cfg["NUM_CTAS"]
            num_smem_buffers = cfg["NUM_SMEM_BUFFERS"]
            num_tmem_buffers = cfg["NUM_TMEM_BUFFERS"]
            num_mma_groups = cfg["NUM_MMA_GROUPS"]
            epilogue_subtile = cfg["EPILOGUE_SUBTILE"]

            # --- constraint checks ---
            smem = self._estimate_smem(bm, bn, bk, num_ctas, num_smem_buffers, num_mma_groups, epilogue_subtile)
            if smem > MAX_SMEM:
                continue

            tmem = self._estimate_tmem(bm, bn, num_tmem_buffers)
            if tmem > MAX_TMEM:
                continue

            if bm // num_mma_groups > 128:
                continue

            if bm > M * 2 or bn > N * 2:
                continue

            # --- wave efficiency scoring ---
            score, total_ctas, waves = self._compute_wave_score(M, N, bm, bn, num_ctas, num_sms)

            # Consider split-K when MN tiles don't saturate GPU
            split_k = 1
            num_tiles_m = math.ceil(M / bm)
            num_tiles_n = math.ceil(N / bn)
            num_mn_tiles = num_tiles_m * num_tiles_n

            if num_mn_tiles < num_sms:
                k_tiles = math.ceil(K / bk)
                for sk in [8, 4, 2]:
                    if k_tiles >= sk and k_tiles // sk >= 4:
                        sk_score, sk_ctas, sk_waves = self._compute_wave_score(
                            M, N, bm, bn, num_ctas, num_sms, sk
                        )
                        if sk_score < score or (sk_score == score and sk_ctas > total_ctas):
                            score, total_ctas, waves, split_k = sk_score, sk_ctas, sk_waves, sk
                        break  # Use the first valid split-K

            # --- selection ---
            score_slack = 0.1
            adjusted_score = score

            if (
                adjusted_score < best_score - score_slack
                or (adjusted_score < best_score + score_slack and waves < best_waves)
                or (adjusted_score < best_score + score_slack and waves == best_waves and num_ctas > 1)
            ):
                best_score = adjusted_score
                best_waves = waves
                best_config = dict(cfg)
                best_config["SPLIT_K"] = split_k

        return best_config
