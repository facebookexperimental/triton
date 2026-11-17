"""Shapes for ``tlx.ops.rmsnorm_cat``.

A shape names a part layout, a batch and an output count. ``inputs`` builds the parts as the strided views a caller
passes: the over-arch layout reproduces the views Inductor hands its fused kernel in production.
"""

from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class RmsNormCatShape(NamedTuple):
    batch: int
    layout: str
    outputs: int
    dtype: str


# The over-arch input of a production ranking model: per candidate, cat(float_proj[8192], fm_norm(fm)[32 x 128],
# dot_norm(dots)[4352 x 2]) -> RMSNorm(20992) for each of one or two over-arch towers. fm is a slice of a GEMM output
# ([96, B, 128], so its rows are B * 128 apart) and dots is a bmm output [B, 2, 4352] read transposed. Inductor fuses
# this into one reduction kernel per tower count. B is the candidates per batch: 692 is the most common batch in the
# model's trace, 768 the p99.
OVER_ARCH = "ranking_over_arch"
LAYOUTS = (OVER_ARCH, "plain_1", "mixed_small", "transposed_c4")

SYNTHETIC: tuple[RmsNormCatShape, ...] = (
    RmsNormCatShape(3, "plain_1", 1, "fp16"),
    RmsNormCatShape(5, "mixed_small", 2, "bf16"),
    RmsNormCatShape(7, "transposed_c4", 3, "fp16"),
    RmsNormCatShape(4, OVER_ARCH, 2, "fp16"),
)

SM90_1 = FocusSuite(
    name="sm90_1",
    op="rmsnorm_cat",
    shapes=tuple(RmsNormCatShape(batch, OVER_ARCH, outputs, "fp16") for batch in (692, 768, 256) for outputs in (1, 2)),
)
FOCUS_SUITES = (SM90_1, )
DEFAULT_SUITES = {"sm90": ("sm90_1", )}
FOCUS = FocusRegistry("rmsnorm_cat", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(batch, layout, outputs, dtype) -> str:
    return f"((), {{'dtype': '{dtype}', 'B': '{batch}', 'layout': '{layout}', 'outputs': '{outputs}'}})"


def _weight(n, dtype, device):
    import torch

    return torch.rand(n, device=device, dtype=dtype) + 0.5


def inputs(shape, device="cuda"):
    """``(parts, part_weights, weights)`` for ``shape``; parts are views into freshly allocated buffers."""
    import torch

    batch, layout, outputs, dtype_name = shape
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]

    def randn(*size):
        return torch.randn(*size, device=device, dtype=dtype)

    if layout == OVER_ARCH:
        float_proj = randn(batch, 20992)[:, :8192]
        fm = randn(96, batch, 128)[64:].permute(1, 0, 2)
        dots = randn(batch, 2, 4352).permute(0, 2, 1)
        parts = [float_proj, fm, dots]
        part_weights = [None, _weight(4096, dtype, device), _weight(8704, dtype, device)]
    elif layout == "plain_1":
        parts, part_weights = [randn(batch, 1024)], [None]
    elif layout == "mixed_small":
        parts = [randn(batch, 256), randn(batch, 12, 32), randn(batch, 2, 100).permute(0, 2, 1)]
        part_weights = [None, _weight(384, dtype, device), _weight(200, dtype, device)]
    elif layout == "transposed_c4":
        parts = [randn(batch, 64), randn(batch, 4, 48).permute(0, 2, 1)]
        part_weights = [None, _weight(192, dtype, device)]
    else:
        raise ValueError(f"unknown layout {layout!r}")
    n = sum(p[0].numel() for p in parts)
    return parts, part_weights, [_weight(n, dtype, device) for _ in range(outputs)]


def reference(parts, part_weights, weights, eps=1e-5):
    """The eager composition, in fp32: each normalized part and the concatenation are rounded to the I/O dtype."""
    import torch

    def rms(x, w):
        x = x.float()
        return x / torch.sqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()

    dtype = parts[0].dtype
    flat = [p.reshape(p.shape[0], -1) for p in parts]
    cat = torch.cat([f if w is None else rms(f, w).to(dtype) for f, w in zip(flat, part_weights)], dim=1)
    return [rms(cat, w).to(dtype) for w in weights]


def bytes_moved(parts, outputs) -> int:
    """Compulsory HBM traffic of one call: each part read once, each output written once."""
    n = sum(p[0].numel() for p in parts)
    return parts[0].shape[0] * n * parts[0].element_size() * (1 + outputs)
