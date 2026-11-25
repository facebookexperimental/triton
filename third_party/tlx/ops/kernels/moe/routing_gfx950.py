"""FP32 sigmoid top-k MoE routing with a selection-only correction bias, on gfx950.

Each row lives in one wave, so no reduction crosses waves. Short launches give a row the whole wave. Long launches
use ``tlx.require_layout`` to pack four rows into a wave, 16 lanes per row: each reduction then takes four
cross-lane steps instead of six and every step serves four rows. Every reduction is exact (max, min, and a sum of
one nonzero term), so both layouts give bitwise-identical rows.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

_WAVE = 64
# One row per wave is the lower-latency layout; its cost steps up with every extra wave per SIMD (4 per CU), and the
# four-rows-per-wave layout overtakes it after three such steps on MI355X.
ROW_WAVES_PER_CU = 12
# Element offsets at or past this need 64-bit arithmetic.
INT32_OFFSETS = 1 << 31


@functools.lru_cache(maxsize=None)
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def pick_config(tokens: int, experts_padded: int, num_cus: int) -> int:
    """Rows per wave (one wave per program). A row needs at least one expert per lane."""
    rows_per_wave = 1 if tokens <= ROW_WAVES_PER_CU * num_cus else 4
    return max(rows_per_wave, _WAVE // experts_padded)


@triton.jit
def route_rows(logits_ptr, rows64, logits, bias, mask, experts, stride_lm, stride_le, multiplier, route_scale, eps,
               ROWS: tl.constexpr, TOPK: tl.constexpr, TOPK_PAD: tl.constexpr, E_PAD: tl.constexpr, COL64: tl.constexpr,
               FROM_TILE: tl.constexpr):
    """Top-k ids and weights of a (ROWS, E_PAD) tile of logits, in whatever layout the tile has: every reduction is
    exact, so every layout gives the same bits. Rows and experts outside ``mask`` get id -1 and weight 0. The winners'
    logits are reloaded from ``logits_ptr``, or with ``FROM_TILE`` picked out of the tile (a sum of one nonzero term,
    so the same value)."""
    # Selection key of a padded or already-picked expert: below every real score's key. A local constant, not a
    # module global, keeps the launch on Triton's C fast path.
    TAKEN: tl.constexpr = -(1 << 31)
    scores = tl.sigmoid(logits * multiplier)
    # Order-preserving int32 keys: an int max is one fused DPP op per cross-lane step, where a float max also needs a
    # move and a NaN-canonicalizing max. A key is never -0.0 (a sigmoid is never -0.0), so int equality is float
    # equality.
    biased = scores + bias
    bits = biased.to(tl.int32, bitcast=True)
    # A NaN's key would rank above +inf; like a float max, never pick it.
    key = tl.where(mask & (biased == biased), bits ^ ((bits >> 31) & 0x7FFFFFFF), TAKEN)

    slots = tl.arange(0, TOPK_PAD)[None, :]
    picked_ids = tl.full((ROWS, TOPK_PAD), -1, tl.int32)
    picked_logits = tl.zeros((ROWS, TOPK_PAD), tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(key, axis=1)
        idx = tl.min(tl.where(key == best[:, None], experts, E_PAD), axis=1)
        # A row with fewer than TOPK non-NaN experts runs out: every key is TAKEN, so the slot gets no expert.
        idx = tl.where(best == TAKEN, -1, idx)
        picked_ids = tl.where(slots == k, idx[:, None], picked_ids)
        if FROM_TILE:
            pick = tl.sum(tl.where(experts == idx[:, None], logits, 0.0), axis=1)
            picked_logits = tl.where(slots == k, pick[:, None], picked_logits)
        key = tl.where(experts == idx[:, None], TAKEN, key)

    # -1 also marks padded slots and rows past num_tokens (whose keys are all TAKEN).
    found = picked_ids >= 0
    if not FROM_TILE:
        picked_logits = tl.load(
            logits_ptr + rows64[:, None] * stride_lm + (picked_ids.to(tl.int64) if COL64 else picked_ids) * stride_le,
            mask=found, other=0.0)
    # The winners' scores, recomputed from their logits with the same float ops as the selection used.
    picked = tl.sigmoid(picked_logits * multiplier)
    picked = tl.where(found, picked, 0.0)
    score_sum = tl.zeros((ROWS, ), tl.float32)
    for k in tl.static_range(TOPK):
        score_sum += tl.sum(tl.where(slots == k, picked, 0.0), axis=1)

    inv = route_scale / (score_sum + eps)
    return tl.where(found, picked * inv[:, None], 0.0), picked_ids


@triton.jit
def _routing_kernel(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    num_valid_ptr,
    num_tokens,
    num_experts,
    stride_lm,
    stride_le,
    stride_b,
    stride_wm,
    stride_im,
    multiplier,
    route_scale,
    eps,
    split_stride,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    E_PAD: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_NUM_VALID: tl.constexpr,
    COL64: tl.constexpr,
    SPLITS: tl.constexpr,
):
    LANES: tl.constexpr = 64 // ROWS
    PER_LANE: tl.constexpr = E_PAD // LANES
    layout: tl.constexpr = tlx.layout(
        shape=((LANES, ROWS), (PER_LANE, )),
        stride=((PER_LANE, E_PAD), (1, )),
    )
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_mask = rows < num_tokens
    rows64 = rows.to(tl.int64)
    offs = tl.arange(0, E_PAD)
    mask = tlx.require_layout(row_mask[:, None] & (offs < num_experts)[None, :], layout)
    experts = tlx.require_layout(tl.zeros((ROWS, E_PAD), tl.int32) + offs[None, :], layout)

    # An expert stride below 2**31 can still put the last column past 2**31 elements.
    cols = experts.to(tl.int64) if COL64 else experts
    tile = rows64[:, None] * stride_lm + cols * stride_le
    logits = tl.load(logits_ptr + tile, mask=mask, other=0.0)
    # SPLITS > 1: logits_ptr holds router_gemm's split-K partials; add them in split order, as its reduction does.
    for s in tl.static_range(1, SPLITS):
        logits += tl.load(logits_ptr + split_stride.to(tl.int64) * s + tile, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols * stride_b, mask=mask, other=0.0).to(tl.float32)
    weights, picked_ids = route_rows(logits_ptr, rows64, logits, bias, mask, experts, stride_lm, stride_le, multiplier,
                                     route_scale, eps, ROWS, TOPK, TOPK_PAD, E_PAD, COL64, SPLITS > 1)
    slots = tl.arange(0, TOPK_PAD)[None, :]
    out_mask = row_mask[:, None] & (slots < TOPK)
    if HAS_NUM_VALID:
        pad = (rows >= tl.load(num_valid_ptr))[:, None]
        picked_ids = tl.where(pad, -1, picked_ids)
        weights = tl.where(pad, 0.0, weights)
    tl.store(weights_ptr + rows64[:, None] * stride_wm + slots, weights, mask=out_mask)
    tl.store(ids_ptr + rows64[:, None] * stride_im + slots, picked_ids, mask=out_mask)


def moe_routing(router_logits, correction_bias, topk, router_multiplier, route_scale, eps, num_valid_tokens, out,
                splits=1):
    """With ``splits > 1``, ``router_logits`` is ``(splits, T, E)``: router_gemm's split-K partials, summed here."""
    T, E = router_logits.shape[-2:]
    if out is None:
        weights = torch.empty(T, topk, device=router_logits.device, dtype=torch.float32)
        ids = torch.empty(T, topk, device=router_logits.device, dtype=torch.int32)
    else:
        weights, ids = out
    if T == 0:
        return weights, ids
    e_pad = triton.next_power_of_2(E)
    rows = pick_config(T, e_pad, _num_cus(router_logits.device.index))
    _routing_kernel[(triton.cdiv(T, rows), )](
        router_logits,
        correction_bias,
        weights,
        ids,
        num_valid_tokens,
        T,
        E,
        router_logits.stride(-2),
        router_logits.stride(-1),
        correction_bias.stride(0),
        weights.stride(0),
        ids.stride(0),
        float(router_multiplier),
        float(route_scale),
        float(eps),
        T * E,
        TOPK=topk,
        TOPK_PAD=triton.next_power_of_2(topk),
        E_PAD=e_pad,
        ROWS=rows,
        HAS_NUM_VALID=num_valid_tokens is not None,
        COL64=(E - 1) * max(router_logits.stride(-1), correction_bias.stride(0)) >= INT32_OFFSETS,
        SPLITS=splits,
        num_warps=1,
    )
    return weights, ids
