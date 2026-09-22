from __future__ import annotations

from .f01_double_layernorm_lds_retention import CASE as DOUBLE_LAYERNORM_LDS_RETENTION
# TODO: Re-enable these registrations after the required PyTorch custom-op
# autotuning support lands.
# from .f02_gemm_layernorm import CASE as GEMM_LAYERNORM
# from .f03_gemm_rmsnorm import CASE as GEMM_RMSNORM

CASES = (DOUBLE_LAYERNORM_LDS_RETENTION, )
