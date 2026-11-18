from __future__ import annotations

from typing import NamedTuple

from ..._shape_suites import FocusRegistry, FocusSuite


class MoeRouterTopkAlignShape(NamedTuple):
    """``tokens`` bf16 rows of width ``hidden`` routed to ``topk`` of ``experts``; this rank holds ``local_experts``
    experts from ``local_start``, sorted into ``block_m``-row blocks."""
    tokens: int
    experts: int
    hidden: int
    topk: int
    local_experts: int
    local_start: int
    block_m: int


SYNTHETIC: tuple[MoeRouterTopkAlignShape, ...] = (
    MoeRouterTopkAlignShape(3, 64, 1024, 4, 16, 16, 16),
    MoeRouterTopkAlignShape(20, 128, 2048, 8, 128, 0, 32),
    MoeRouterTopkAlignShape(100, 256, 4096, 2, 64, 192, 16),
)


def _ep8(tokens: int, block_m: int) -> MoeRouterTopkAlignShape:
    # 256 experts with an 8192 hidden size, top 8, on the third of eight expert-parallel ranks.
    return MoeRouterTopkAlignShape(tokens, 256, 8192, 8, 32, 64, block_m)


# Decode batches of 1 to 64 requests: the router GEMV and the split-K buckets (the grouped GEMM takes 16-row blocks).
GFX950_1 = FocusSuite(name="gfx950_1", op="moe_router_topk_align",
                      shapes=tuple(_ep8(t, 16) for t in (1, 4, 8, 9, 16, 32, 64)))
# Prefill chunks; the block grows with the batch as the grouped GEMM's does.
PREFILL = (_ep8(151, 16), _ep8(2048, 64), _ep8(8192, 64))
GFX950_2 = FocusSuite(name="gfx950_2", op="moe_router_topk_align", shapes=PREFILL)
GFX950_ALL = FocusSuite(name="gfx950_all", op="moe_router_topk_align", includes=("gfx950_1", "gfx950_2"))

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("moe_router_topk_align", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(shape: MoeRouterTopkAlignShape) -> str:
    return (f"(({shape.tokens}, {shape.hidden}), {{'experts': '{shape.experts}', 'topk': '{shape.topk}', "
            f"'local': '{shape.local_start}+{shape.local_experts}', 'block_m': '{shape.block_m}'}})")


def inputs(shape: MoeRouterTopkAlignShape, seed: int = 0, device="cuda") -> dict:
    """Normalized bf16 hidden states, an fp32 router weight at the scale of a trained router, a small correction bias,
    and the routing scalars of a sigmoid router."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape.tokens, shape.hidden, device=device, generator=gen).to(torch.bfloat16)
    weight = torch.randn(shape.experts, shape.hidden, device=device, generator=gen) * shape.hidden**-0.5
    bias = torch.randn(shape.experts, device=device, generator=gen) * 0.05
    return dict(x=x, weight=weight, correction_bias=bias, topk=shape.topk, num_local_experts=shape.local_experts,
                local_expert_start=shape.local_start, block_m=shape.block_m, router_multiplier=0.5, route_scale=2.5,
                eps=1e-20)


def reference(x, weight, correction_bias, topk, num_local_experts, local_expert_start, block_m, router_multiplier,
              route_scale, eps):
    """The chain of ops this op replaces: ``router_gemm``, ``moe_routing`` and ``tlx.ops.moe``'s slot sort."""
    from triton.tlx.ops import moe_routing, router_gemm
    from triton.tlx.ops.kernels.moe.gfx950 import _align

    logits = router_gemm(x, weight)
    weights, ids = moe_routing(logits, correction_bias, topk, router_multiplier=router_multiplier,
                               route_scale=route_scale, eps=eps)
    return (weights, ids, *_align(ids, num_local_experts, local_expert_start, block_m))
