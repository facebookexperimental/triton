from __future__ import annotations

from typing import NamedTuple

from ..._shape_suites import FocusRegistry, FocusSuite


class LinearRowscaleShape(NamedTuple):
    """``x`` of shape ``(m, k)``, ``weight`` of shape ``(n, k)``, a bias and an FP32 row scale."""

    m: int
    n: int
    k: int
    dtype: str


SYNTHETIC: tuple[LinearRowscaleShape, ...] = (
    LinearRowscaleShape(129, 1024, 368, "fp16"),
    LinearRowscaleShape(1000, 136, 64, "bf16"),
    LinearRowscaleShape(1, 8, 8, "fp16"),
)

# Ranking-model attention-MLP output projections, each a per-candidate generated matrix: 160 x 368 (N = 58880,
# K = 368), 96 x 368 (N = 35328, K = 368) and 240 x 128 (N = 30720, K = 128), over the merged batch's candidate counts
# (average 671, p99 767, at most 768).
PRODUCTS = ((58880, 368), (35328, 368), (30720, 128))
SM90_1 = FocusSuite(
    name="sm90_1",
    op="linear_rowscale",
    shapes=tuple(LinearRowscaleShape(m, n, k, "fp16") for n, k in PRODUCTS for m in (600, 671, 704, 768)),
)
FOCUS_SUITES = (SM90_1, )
DEFAULT_SUITES = {"sm90": ("sm90_1", )}
FOCUS = FocusRegistry("linear_rowscale", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def inputs(entry, device="cuda"):
    import torch

    m, n, k, dtype = entry
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    x = torch.randn(m, k, device=device, dtype=dtype)
    weight = (torch.rand(n, k, device=device, dtype=dtype) * 2 - 1) * k**-0.5
    bias = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * k**-0.5
    row_scale = torch.rand(m, device=device, dtype=torch.float32) + 0.5
    return x, weight, bias, row_scale


def flops(m, n, k):
    return 2 * m * n * k


def label(m, n, k, dtype) -> str:
    return f"((), {{'dtype': '{dtype}', 'M': '{m}', 'N': '{n}', 'K': '{k}'}})"
