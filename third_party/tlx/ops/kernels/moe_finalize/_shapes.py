from __future__ import annotations

from typing import NamedTuple

import triton
import triton.language as tl

from .._shape_suites import FocusRegistry, FocusSuite


class MoeFinalizeShape(NamedTuple):
    """``tokens`` routed to ``topk`` of ``experts`` experts, of which this rank holds ``local``; hidden size ``H``."""
    tokens: int
    H: int
    topk: int
    experts: int
    local: int


SYNTHETIC: tuple[MoeFinalizeShape, ...] = (
    MoeFinalizeShape(5, 1000, 3, 16, 16),
    MoeFinalizeShape(300, 2048, 4, 64, 8),
    MoeFinalizeShape(37, 8192, 8, 256, 256),
)

# MoE pretraining with expert parallelism over a combine buffer: every slot of a rank's 16384 tokens holds its expert
# row (local = experts). Routed hidden sizes of the MoE tiers: 1536, 2048, 3072, 4096.
SM100_1 = FocusSuite(name="sm100_1", op="moe_finalize",
                     shapes=tuple(MoeFinalizeShape(16384, h, 8, 256, 256) for h in (1536, 2048, 3072, 4096)))
# Expert-parallel inference: 64 and 32 of 256 experts per rank, most slots dead.
SM100_2 = FocusSuite(name="sm100_2", op="moe_finalize",
                     shapes=(MoeFinalizeShape(4096, 4096, 8, 256, 64), MoeFinalizeShape(16384, 4096, 8, 256, 32)))
SM100_ALL = FocusSuite(name="sm100_all", op="moe_finalize", includes=("sm100_1", "sm100_2"))

FOCUS_SUITES = (SM100_1, SM100_2, SM100_ALL)
DEFAULT_SUITES = {"sm100": ("sm100_all", )}
FOCUS = FocusRegistry("moe_finalize", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))
EPS = 1e-5


def label(shape: MoeFinalizeShape) -> str:
    return (f"(({shape.tokens}, {shape.topk}, {shape.H}), {{'dtype': 'bf16', 'experts': '{shape.experts}', "
            f"'local': '{shape.local}'}})")


def inputs(shape: MoeFinalizeShape, seed: int = 0, rank: int = 0, device="cuda"):
    """Routed expert outputs grouped by expert, as an expert GEMM writes them, with the slot-to-row map.

    Returns a dict: ``expert_out`` ``(R, H)`` bf16 (rows of different scales), ``row_index`` ``(T, K)`` int32 (-1 for
    slots routed to other ranks), ``topk_weights`` ``(T, K)`` fp32, ``topk_ids`` ``(T, K)`` int32, ``expert_mask``
    ``(E,)`` bool for the local experts and ``norm_weight`` ``(H,)`` bf16.
    """
    import torch

    T, H, K, E, L = shape
    gen = torch.Generator(device=device).manual_seed(seed)
    weights, ids = torch.rand(T, E, generator=gen, device=device).topk(K, dim=1)
    topk_weights = (weights / weights.sum(1, keepdim=True)).contiguous()
    topk_ids = ids.to(torch.int32).contiguous()
    first = rank * L
    expert_mask = torch.zeros(E, dtype=torch.bool, device=device)
    expert_mask[first:first + L] = True
    is_local = expert_mask[topk_ids.long()]
    order = torch.argsort(torch.where(is_local, topk_ids, E).reshape(-1), stable=True)
    rows = int(is_local.sum())
    row_index = torch.full((T * K, ), -1, dtype=torch.int32, device=device)
    row_index[order[:rows]] = torch.arange(rows, dtype=torch.int32, device=device)
    expert_out = (torch.randn(rows, H, generator=gen, device=device) *
                  torch.rand(rows, 1, generator=gen, device=device).mul(4)).to(torch.bfloat16)
    norm_weight = (1 + 0.1 * torch.randn(H, generator=gen, device=device)).to(torch.bfloat16)
    return dict(expert_out=expert_out, row_index=row_index.view(T, K), topk_weights=topk_weights, topk_ids=topk_ids,
                expert_mask=expert_mask, norm_weight=norm_weight)


def unpermute(expert_out, row_index):
    """The ``(T, K, H)`` layout: slot ``(t, k)`` holds its expert row, zeros where ``row_index`` is -1. Graph safe."""
    import torch

    T, K = row_index.shape
    if expert_out.shape[0] == 0:
        return expert_out.new_zeros(T, K, expert_out.shape[1])
    rows = expert_out[row_index.clamp(min=0).long()]
    return torch.where((row_index >= 0)[:, :, None], rows, 0.0).to(expert_out.dtype)


def unfused_finalize(dense, topk_weights, norm_weight=None, input_scale=True, eps=EPS):
    """The finalize of the ``(T, K, H)`` layout as one Triton kernel: a ``(K, H)`` tile per token, reduced along
    ``H``, then along ``K`` (padded to a power of two with zero terms, as the op does), with the op's warp count."""
    import torch

    from . import sm100

    T, K, H = dense.shape
    out = torch.empty(T, H, device=dense.device, dtype=torch.bfloat16)
    if T:
        _unfused_kernel[(T, )](dense, topk_weights, norm_weight if norm_weight is not None else dense, out, eps, H, K,
                               triton.next_power_of_2(K), triton.next_power_of_2(H), norm_weight is not None,
                               input_scale, num_warps=sm100.num_warps(triton.next_power_of_2(H)))
    return out


@triton.jit
def _unfused_kernel(Y, TOPK_WEIGHTS, WEIGHT, OUT, eps, H: tl.constexpr, K: tl.constexpr, K_POW2: tl.constexpr,
                    BLOCK_H: tl.constexpr, HAS_WEIGHT: tl.constexpr, INPUT_SCALE: tl.constexpr):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    slots = tl.arange(0, K_POW2)
    mask = cols < H
    smask = slots < K
    x = tl.load(Y + token.to(tl.int64) * K * H + slots[:, None] * H + cols[None, :],
                mask=smask[:, None] & mask[None, :], other=0.0)
    if INPUT_SCALE:
        if HAS_WEIGHT:
            w = tl.load(WEIGHT + cols, mask=mask, other=0.0)
            n = (x * w[None, :]).to(tl.float32)
        else:
            n = x.to(tl.float32)
        y = n * tl.rsqrt(tl.sum(n * n, axis=1) / H + eps)[:, None]
    else:
        x = x.to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, axis=1) / H + eps)[:, None]
        if HAS_WEIGHT:
            y = y * tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)[None, :]
    y = y.to(tl.bfloat16).to(tl.float32)
    s = tl.load(TOPK_WEIGHTS + token * K + slots, mask=smask, other=0.0)
    terms = tl.where(smask[:, None], (y * s[:, None]).to(tl.bfloat16).to(tl.float32), 0.0)
    tl.store(OUT + token.to(tl.int64) * H + cols, tl.sum(terms, axis=0).to(tl.bfloat16), mask=mask)


def reference(dense, topk_weights, norm_weight=None, input_scale=True, eps=EPS, live=None):
    """Eager torch finalize of the ``(T, K, H)`` layout with the op's rounding points; slots where ``live`` is False
    contribute nothing."""
    import torch

    x = dense.float()
    w = norm_weight.float() if norm_weight is not None else None
    if input_scale:
        n = (x * w).to(torch.bfloat16).float() if w is not None else x
        y = n * torch.rsqrt(n.pow(2).mean(-1, keepdim=True) + eps)
    else:
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if w is not None:
            y = y * w
    y = y.to(torch.bfloat16).float()
    terms = (y * topk_weights.float()[:, :, None]).to(torch.bfloat16).float()
    if live is not None:
        terms = torch.where(live[:, :, None] != 0, terms, 0.0)
    acc = terms[:, 0]
    for k in range(1, terms.shape[1]):
        acc = acc + terms[:, k]
    return acc.to(torch.bfloat16)
