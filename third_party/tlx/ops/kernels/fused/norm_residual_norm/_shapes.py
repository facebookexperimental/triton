from __future__ import annotations

from typing import NamedTuple

from ..._shape_suites import FocusRegistry, FocusSuite


class NormResidualNormShape(NamedTuple):
    """``tokens`` rows of width ``H`` at one boundary kind: ``attn`` (unweighted post norm), ``ffn`` (input-scale
    post norm) or ``final`` (``ffn`` with a BF16 pre-norm input). All three use a per-channel gate."""
    tokens: int
    H: int
    boundary: str


BOUNDARIES = ("attn", "ffn", "final")
SYNTHETIC: tuple[NormResidualNormShape, ...] = tuple(
    NormResidualNormShape(tokens, 4096, boundary) for tokens in (1, 3, 300) for boundary in BOUNDARIES)

# Hidden 8192 residual stream; decode including 8-token speculative verify batches, then prefill extends.
GFX950_1 = FocusSuite(
    name="gfx950_1", op="norm_residual_norm",
    shapes=tuple(NormResidualNormShape(t, 8192, b) for t in (1, 4, 8, 9, 12, 16, 18) for b in BOUNDARIES))
GFX950_2 = FocusSuite(
    name="gfx950_2", op="norm_residual_norm", shapes=tuple(
        NormResidualNormShape(t, 8192, b)
        for t in (27, 64, 151, 512, 1024, 2048, 4096, 8192, 16384)
        for b in BOUNDARIES))
GFX950_ALL = FocusSuite(name="gfx950_all", op="norm_residual_norm", includes=("gfx950_1", "gfx950_2"))

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("norm_residual_norm", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(shape: NormResidualNormShape) -> str:
    return f"(({shape.tokens}, {shape.H}), {{'boundary': '{shape.boundary}'}})"


def bytes_moved(shape: NormResidualNormShape) -> int:
    """Compulsory traffic: BF16 branch and output, FP32 stream and new stream."""
    return shape.tokens * shape.H * (2 + 4 + 4 + 2)


def inputs(shape: NormResidualNormShape, seed: int = 0, device="cuda") -> dict:
    """Keyword arguments of ``tlx.ops.norm_residual_norm`` (and of ``reference``) for one boundary."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    T, H = shape.tokens, shape.H
    branch = (torch.randn(T, H, device=device, generator=gen) * 3).to(torch.bfloat16)
    stream = torch.randn(T, H, device=device, generator=gen) * 10
    gate = torch.randn(H, device=device, generator=gen) / 0.3
    beta = torch.sigmoid(gate)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-gate) * (1 + beta), min=1e-3))
    pre_weight = (1 + 0.1 * torch.randn(H, device=device, generator=gen)).to(torch.bfloat16)
    post_weight = None
    if shape.boundary != "attn":
        post_weight = (1 + 0.1 * torch.randn(H, device=device, generator=gen)).to(torch.bfloat16)
    return dict(stream=stream, branch=branch, post_weight=post_weight, post_norm_input_scale=shape.boundary != "attn",
                pre_weight=pre_weight, alpha=alpha, beta=beta, post_eps=1e-8, pre_eps=1e-5,
                pre_norm_bf16=shape.boundary == "final")


def reference(stream, branch, post_weight=None, post_norm_input_scale=False, pre_weight=None, alpha=None, beta=None,
              post_eps=1e-6, pre_eps=1e-6, pre_norm_bf16=False):
    """The unfused module sequence: BF16 post norm, FP32 gated residual add, pre norm with a BF16 output."""
    import torch
    import torch.nn.functional as F

    H = branch.shape[1]
    if post_norm_input_scale and post_weight is not None:
        y = F.rms_norm(post_weight * branch, (H, ), None, post_eps)
    else:
        y = F.rms_norm(branch, (H, ), post_weight, post_eps)
    y = y.to(torch.bfloat16).float()
    new_stream = stream + y if alpha is None else alpha * stream + beta * y
    q = new_stream.to(torch.bfloat16).float() if pre_norm_bf16 else new_stream
    out = F.rms_norm(q, (H, ), None, pre_eps)
    if pre_weight is not None:
        out = out * pre_weight.float()
    return out.to(torch.bfloat16), new_stream
