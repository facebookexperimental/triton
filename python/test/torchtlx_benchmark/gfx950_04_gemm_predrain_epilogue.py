# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx950_04_gemm_predrain_epilogue"
CANDIDATE_NAME = "predrain_prefetch"
CANDIDATE_CODE_MARKERS = ("TorchTLX pre-drain epilogue prefetch: gemm_drain", )
INTERLEAVE_VARIANTS = True
_COMMON_CONFIG: dict[str, object] = {
    "force_disable_caches": True,
    "max_autotune": True,
    "max_autotune_gemm_backends": "TRITON",
    "enable_caching_generated_triton_templates": False,
    "triton.tlx_mode": "force",
}
BASELINE_CONFIG: dict[str, object] = {
    **_COMMON_CONFIG,
}
CANDIDATE_CONFIG: dict[str, object] = {
    **_COMMON_CONFIG,
}
ATOL = 5.0e-1
RTOL = 3.0e-2

M = 1024
N = 21568
K = 256
DTYPE_NAME = "bf16"

_CONFIGS = {
    256: (64, 64, 64, 8, 8, 3),
    512: (64, 64, 64, 8, 8, 3),
    1024: (128, 128, 64, 8, 8, 2),
    1536: (128, 128, 64, 8, 8, 2),
    2048: (128, 128, 64, 8, 8, 2),
}
SHAPES = tuple({"m": M, "n": N, "k": k, "dtype": DTYPE_NAME} for k in _CONFIGS)


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int, choices=_CONFIGS)
    parser.add_argument("--dtype", choices=("fp16", "bf16"))


def configure(args) -> None:
    global M, N, K, DTYPE_NAME
    M = args.m if args.m is not None else M
    N = args.n if args.n is not None else N
    K = args.k if args.k is not None else K
    DTYPE_NAME = args.dtype if args.dtype is not None else DTYPE_NAME


def problem() -> str:
    return f"M={M} K={K} N={N} dtype={DTYPE_NAME}"


@contextlib.contextmanager
def compile_context(variant: str):
    """Pin one non-persistent template/config so the A/B changes only placement."""
    from triton.language.extra.tlx.inductor import mm_templates, registry

    heuristic = registry.Gfx950AddMMWarpPipeConfigHeuristic
    old_configs = heuristic.WARPPIPE_CONFIGS
    old_prefetch = heuristic._PREDRAIN_EPILOGUE_PREFETCH
    original_append_tlx = mm_templates.append_tlx

    def append_only_warppipe(templates, op_name, kernel_inputs):
        original_append_tlx(templates, op_name, kernel_inputs)
        if mm_templates.gfx950_addmm_warppipe_template in templates:
            templates[:] = [mm_templates.gfx950_addmm_warppipe_template]
        return templates

    try:
        heuristic.WARPPIPE_CONFIGS = [_CONFIGS[K]]
        heuristic._PREDRAIN_EPILOGUE_PREFETCH = variant == "candidate"
        mm_templates.append_tlx = append_only_warppipe
        yield
    finally:
        heuristic.WARPPIPE_CONFIGS = old_configs
        heuristic._PREDRAIN_EPILOGUE_PREFETCH = old_prefetch
        mm_templates.append_tlx = original_append_tlx


def model(
    bias: torch.Tensor,
    a: torch.Tensor,
    weight: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    value = torch.addmm(bias, a, weight.t())
    return value + value * y


def baseline_model(
    bias: torch.Tensor,
    a: torch.Tensor,
    weight: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return model(bias, a, weight, y)


def candidate_model(
    bias: torch.Tensor,
    a: torch.Tensor,
    weight: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return model(bias, a, weight, y)


def comparison_models():
    """Distinct code objects keep both Dynamo variants live for interleaving."""
    return baseline_model, candidate_model


def make_inputs() -> tuple[torch.Tensor, ...]:
    dtype = torch.float16 if DTYPE_NAME == "fp16" else torch.bfloat16
    torch.manual_seed(0)
    return (
        torch.randn((N, ), device="cuda", dtype=dtype),
        torch.randn((M, K), device="cuda", dtype=dtype),
        torch.randn((N, K), device="cuda", dtype=dtype),
        torch.randn((M, N), device="cuda", dtype=dtype),
    )
