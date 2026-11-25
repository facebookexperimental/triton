from __future__ import annotations

from typing import NamedTuple

from ..._shape_suites import FocusRegistry, FocusSuite


class RmsnormGateShape(NamedTuple):
    """``tokens`` rows of ``heads`` 64-element heads. The gate is the last ``heads * 64`` columns of a wider row
    (attention projection output: query, key, value and gate columns), so its row stride is not its width."""
    tokens: int
    heads: int


SYNTHETIC: tuple[RmsnormGateShape,
                 ...] = tuple(RmsnormGateShape(tokens, heads) for tokens in (1, 3, 300) for heads in (8, 16, 40))

# 16 heads of 64 (1024 columns, gate row stride 2304): decode and speculative-verify batches, then prefill.
GFX950_1 = FocusSuite(name="gfx950_1", op="rmsnorm_gate",
                      shapes=tuple(RmsnormGateShape(t, 16) for t in (1, 4, 8, 16, 18, 32, 64, 128)))
GFX950_2 = FocusSuite(name="gfx950_2", op="rmsnorm_gate",
                      shapes=tuple(RmsnormGateShape(t, 16) for t in (512, 2048, 8192, 16384)))
GFX950_ALL = FocusSuite(name="gfx950_all", op="rmsnorm_gate", includes=("gfx950_1", "gfx950_2"))

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("rmsnorm_gate", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(shape: RmsnormGateShape) -> str:
    return f"(({shape.tokens}, {shape.heads * 64}), {{'heads': {shape.heads}}})"


def bytes_moved(shape: RmsnormGateShape) -> int:
    """Compulsory traffic: BF16 input, gate and output."""
    return shape.tokens * shape.heads * 64 * 2 * 3


def inputs(shape: RmsnormGateShape, seed: int = 0, device="cuda") -> dict:
    """Keyword arguments of ``tlx.ops.rmsnorm_gate`` (and of ``reference``); the gate is a column slice."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    T, N = shape.tokens, shape.heads * 64
    x = (torch.randn(T, N, device=device, generator=gen) * torch.rand(T, N, device=device, generator=gen) * 4)
    proj = torch.randn(T, N + N // 4 + 256, device=device, generator=gen) * 3
    return dict(x=x.to(torch.bfloat16), gate=proj.to(torch.bfloat16)[:, -N:], eps=1e-5)


def reference(x, gate, eps=1e-6):
    """Per-head sum of squares in the op's order (pairs, then folds in halves), then the FP32 norm and gate."""
    import torch

    T, N = x.shape
    xf = x.float()
    s = (xf * xf).view(T, N // 64, 32, 2).sum(-1)
    while s.shape[-1] > 1:
        half = s.shape[-1] // 2
        s = s[..., :half] + s[..., half:]
    rstd = 1.0 / torch.sqrt(s / 64 + eps)
    y = xf.view(T, N // 64, 64) * rstd
    return (y * torch.sigmoid(gate.float().view(T, N // 64, 64))).view(T, N).to(torch.bfloat16)
