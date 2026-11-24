"""MoE finalize on SM100: per-expert RMSNorm of the routed outputs, then the top-k weighted sum, forward and backward.

The forward gives a token's ``K`` slots to one program as a ``(K, H)`` tile, the tile of the unfused finalize, so the
row reductions and the sum over the slots follow the same tree and give the same bits; the sum over a tile's slot axis
is the left-to-right order. Both forward kernels take one warp per 256 elements of the padded row, at most eight, so
every warp lies along ``H`` and a row is reduced by the same threads in both. Dead slots are masked out of the tile's
loads. Programs are persistent.
The ``row_index`` layout and tiles larger than 32768 elements (``K * H``, which would spill) take a slot loop instead,
one token per program, which reduces each row across the program with the same tree: it skips dead slots with a
uniform branch, so it reads only the live rows, and with rows in expert order it was 10% faster at 16384 tokens with
every slot live and 4x faster with 32 of 256 experts per rank.

The backward takes blocks of slots, so the three layouts are one loop: a slot maps to its row (``row_index``, or the
slot itself) and to whether it contributes. It recomputes the norm from the expert row, then returns, per slot,
``d s = sum(g * y)`` and the gradient of the row, and per program a partial FP32 sum of the norm weight's gradient.
The BF16 roundings of the forward are kept where the eager chain stores BF16 tensors: ``g * s`` and the gradient of
the norm input. The weight gradient accumulates its products in FP32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_MAX_TILE = 32768
_BWD_TILE = 8192
# Four warps per slot block: 408 vs 655 us (H = 1536) and 918 vs 1330 us (H = 4096) at 16384 tokens, top-8.
_BWD_WARPS = 4
_BWD_PER_SM = 8


def num_warps(block_h):
    """Warps of the forward kernels: one per 256 elements of the padded row, at most eight. With more, a
    ``(K, H)`` tile of a short row puts warps on the slot axis and its reductions leave the slot kernel's tree."""
    return min(8, max(1, block_h // 256))


@triton.jit
def _live_slots(slot, valid, ROW_INDEX, TOPK_IDS, EXPERT_MASK, num_experts, num_rows, MODE: tl.constexpr):
    """Row and liveness of each slot: ``row_index`` within ``[0, num_rows)`` (MODE 2), the slot itself with expert
    validity (MODE 1), or every slot (MODE 0)."""
    if MODE == 2:
        row = tl.load(ROW_INDEX + slot, mask=valid, other=-1).to(tl.int64)
        live = valid & (row >= 0) & (row < num_rows)
    elif MODE == 1:
        ids = tl.load(TOPK_IDS + slot, mask=valid, other=-1)
        in_range = valid & (ids >= 0) & (ids < num_experts)
        live = in_range & (tl.load(EXPERT_MASK + tl.where(in_range, ids, 0), mask=in_range, other=0) != 0)
        row = slot
    else:
        live = valid
        row = slot
    return row, live


@triton.jit
def _fwd_kernel(
    Y,  # (T, K, H) expert outputs (the row_index layout takes the slot loop)
    ROW_INDEX,  # unused placeholder for _live_slots
    TOPK_IDS,  # (T, K) expert of each slot (MODE 1)
    EXPERT_MASK,  # (E,) nonzero for the experts whose rows Y holds (MODE 1)
    TOPK_WEIGHTS,  # (T, K) fp32
    WEIGHT,  # (H,) norm weight, bf16 or fp32 (HAS_WEIGHT); all kernels use bf16(WEIGHT + gain_center)
    OUT,  # (T, H)
    T,
    num_experts,
    eps,
    gain_center,
    H: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MODE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_GAIN: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_H)
    cmask = cols < H
    slots = tl.arange(0, K_POW2)
    if HAS_WEIGHT:
        w = tl.load(WEIGHT + cols, mask=cmask, other=0.0)
        if HAS_GAIN:
            w = w.to(tl.float32) + gain_center
        w = w.to(tl.bfloat16)
    for token in tl.range(tl.program_id(0), T, tl.num_programs(0)):
        slot = token.to(tl.int64) * K + slots
        row, live = _live_slots(slot, slots < K, ROW_INDEX, TOPK_IDS, EXPERT_MASK, num_experts, 0, MODE)
        x = tl.load(Y + row[:, None] * H + cols[None, :], mask=live[:, None] & cmask[None, :], other=0.0)
        # Dead and padding slots load as zero rows; with eps = 0 their rsqrt is inf, so their rstd is set to 0.
        if INPUT_SCALE:
            if HAS_WEIGHT:
                n = (x * w[None, :]).to(tl.float32)
            else:
                n = x.to(tl.float32)
            rstd = tl.where(live, tl.rsqrt(tl.sum(n * n, axis=1) / H + eps), 0.0)
            y = n * rstd[:, None]
        else:
            x = x.to(tl.float32)
            rstd = tl.where(live, tl.rsqrt(tl.sum(x * x, axis=1) / H + eps), 0.0)
            y = x * rstd[:, None]
            if HAS_WEIGHT:
                y = y * w.to(tl.float32)[None, :]
        y = y.to(tl.bfloat16).to(tl.float32)
        s = tl.load(TOPK_WEIGHTS + slot, mask=live, other=0.0)
        terms = (y * s[:, None]).to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + token.to(tl.int64) * H + cols, tl.sum(terms, axis=0).to(tl.bfloat16), mask=cmask)


@triton.jit
def _slot_kernel(
    Y,  # rows of expert outputs: (R, H) with ROW_INDEX, else (T, K, H)
    ROW_INDEX,  # (T, K) row of each slot in Y, a slot whose row is outside [0, R) has none (MODE 2)
    TOPK_IDS,  # (T, K) expert of each slot (MODE 1)
    EXPERT_MASK,  # (E,) nonzero for the experts whose rows Y holds (MODE 1)
    TOPK_WEIGHTS,  # (T, K) fp32
    WEIGHT,  # (H,) norm weight, bf16 or fp32 (HAS_WEIGHT)
    OUT,  # (T, H)
    num_experts,
    num_rows,  # R (MODE 2)
    eps,
    gain_center,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    K_POW2: tl.constexpr,
    MODE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_GAIN: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    if HAS_WEIGHT:
        w = tl.load(WEIGHT + cols, mask=mask, other=0.0)
        if HAS_GAIN:
            w = w.to(tl.float32) + gain_center
        w = w.to(tl.bfloat16)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    first = token.to(tl.int64) * K
    if MODE == 1:
        # Liveness from vector loads up front shortens the per-slot chain of dependent loads (id, then mask).
        slots = tl.arange(0, K_POW2)
        ids = tl.load(TOPK_IDS + first + slots, mask=slots < K, other=-1)
        in_range = (ids >= 0) & (ids < num_experts)
        live_slots = in_range & (tl.load(EXPERT_MASK + tl.where(in_range, ids, 0), mask=in_range, other=0) != 0)
        live_slots = live_slots.to(tl.int32)
    for k in tl.static_range(K):
        slot = first + k
        if MODE == 2:
            row = tl.load(ROW_INDEX + slot).to(tl.int64)
            live = (row >= 0) & (row < num_rows)
        elif MODE == 1:
            live = tl.max(tl.where(slots == k, live_slots, 0), axis=0) != 0
            row = slot
        else:
            live = True
            row = slot
        if live:
            x = tl.load(Y + row * H + cols, mask=mask, other=0.0)
            if INPUT_SCALE:
                if HAS_WEIGHT:
                    norm_input = (x * w).to(tl.float32)
                else:
                    norm_input = x.to(tl.float32)
                rstd = tl.rsqrt(tl.sum(norm_input * norm_input, axis=0) / H + eps)
                normed = norm_input * rstd
            else:
                x = x.to(tl.float32)
                rstd = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
                normed = x * rstd
                if HAS_WEIGHT:
                    normed = normed * w.to(tl.float32)
            normed = normed.to(tl.bfloat16).to(tl.float32)
            s = tl.load(TOPK_WEIGHTS + slot)
            acc += (normed * s).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + token.to(tl.int64) * H + cols, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd_kernel(
    GOUT,  # (T, H) bf16
    Y,
    ROW_INDEX,
    TOPK_IDS,
    EXPERT_MASK,
    TOPK_WEIGHTS,
    WEIGHT,
    DY,  # gradient of Y, Y's shape
    DS,  # (T, K) fp32
    DW_PART,  # (num_programs, H) fp32
    NAMED,  # (R,) int32 zeros (MODE 2): counts the slots naming each row
    num_slots,
    num_experts,
    num_rows,  # R (MODE 2)
    eps,
    gain_center,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MODE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_GAIN: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    cmask = cols < H
    offs = tl.arange(0, BLOCK_S)
    if HAS_WEIGHT:
        w = tl.load(WEIGHT + cols, mask=cmask, other=0.0)
        if HAS_GAIN:
            w = w.to(tl.float32) + gain_center
        w = w.to(tl.bfloat16)
        wf = w.to(tl.float32)
    dw = tl.zeros([BLOCK_H], dtype=tl.float32)
    for block in tl.range(pid, tl.cdiv(num_slots, BLOCK_S), tl.num_programs(0)):
        slot = block.to(tl.int64) * BLOCK_S + offs
        valid = slot < num_slots
        row, live = _live_slots(slot, valid, ROW_INDEX, TOPK_IDS, EXPERT_MASK, num_experts, num_rows, MODE)
        if MODE == 2:
            tl.atomic_add(NAMED + row, 1, mask=live, sem="relaxed")
        m = live[:, None] & cmask[None, :]
        x = tl.load(Y + row[:, None] * H + cols[None, :], mask=m, other=0.0)
        g = tl.load(GOUT + (slot // K)[:, None] * H + cols[None, :], mask=m, other=0.0).to(tl.float32)
        s = tl.load(TOPK_WEIGHTS + slot, mask=live, other=0.0)
        gy = (g * s[:, None]).to(tl.bfloat16).to(tl.float32)
        if INPUT_SCALE:
            if HAS_WEIGHT:
                n = (x * w[None, :]).to(tl.float32)
            else:
                n = x.to(tl.float32)
            rstd = tl.where(live, tl.rsqrt(tl.sum(n * n, axis=1) / H + eps), 0.0)
            xhat = n * rstd[:, None]
            y = xhat.to(tl.bfloat16).to(tl.float32)
            corr = tl.sum(gy * xhat, axis=1) / H
            gn = (rstd[:, None] * (gy - xhat * corr[:, None])).to(tl.bfloat16).to(tl.float32)
            if HAS_WEIGHT:
                dx = gn * wf[None, :]
                dw += tl.sum(gn * x.to(tl.float32), axis=0)
            else:
                dx = gn
        else:
            xf = x.to(tl.float32)
            rstd = tl.where(live, tl.rsqrt(tl.sum(xf * xf, axis=1) / H + eps), 0.0)
            xhat = xf * rstd[:, None]
            if HAS_WEIGHT:
                y = (xhat * wf[None, :]).to(tl.bfloat16).to(tl.float32)
                gxhat = gy * wf[None, :]
                dw += tl.sum(gy * xhat, axis=0)
            else:
                y = xhat.to(tl.bfloat16).to(tl.float32)
                gxhat = gy
            corr = tl.sum(gxhat * xhat, axis=1) / H
            dx = rstd[:, None] * (gxhat - xhat * corr[:, None])
        tl.store(DS + slot, tl.sum(g * y, axis=1), mask=valid)
        # A dead slot's gradient is 0 (x and g load as 0, and its rstd is 0). With row_index, rows that no slot names
        # are zero-filled by the launcher, so only live rows are written.
        store_mask = m if MODE == 2 else valid[:, None] & cmask[None, :]
        tl.store(DY + row[:, None] * H + cols[None, :], dx.to(tl.bfloat16), mask=store_mask)
    if HAS_WEIGHT:
        tl.store(DW_PART + pid * H + cols, dw, mask=cmask)


_CHECK_BLOCK = 4096


@triton.jit(debug=True)
def _rows_named_once_kernel(NAMED, num_rows, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    named = tl.load(NAMED + offs, mask=offs < num_rows, other=0)
    tl.device_assert(named <= 1, "tlx.ops.moe_finalize: a row of expert_out is named by more than one row_index slot")


def _check_rows_named_once(named):
    """Fail a device-side assertion when a row of the ``row_index`` layout was named by two slots, whose expert-row
    gradients would overwrite each other. ``debug=True`` keeps the assertion compiled in."""
    _rows_named_once_kernel[(triton.cdiv(named.numel(), _CHECK_BLOCK), )](named, named.numel(), BLOCK=_CHECK_BLOCK)


def _mode(row_index, topk_ids):
    return 2 if row_index is not None else (1 if topk_ids is not None else 0)


def _programs(device, per_sm, work):
    return max(1, min(work, torch.cuda.get_device_properties(device).multi_processor_count * per_sm))


def finalize_forward(expert_out, topk_weights, row_index, topk_ids, expert_mask, norm_weight, input_scale, eps,
                     gain_center=0.0, out=None):
    """The forward without autograd; fills ``out`` when given."""
    T, K = topk_weights.shape
    H = expert_out.shape[-1]
    if out is None:
        out = torch.empty(T, H, device=expert_out.device, dtype=torch.bfloat16)
    if T == 0:
        return out
    K_POW2, BLOCK_H = triton.next_power_of_2(K), triton.next_power_of_2(H)
    if row_index is not None or K_POW2 * BLOCK_H > _MAX_TILE:
        _slot_kernel[(T, )](
            expert_out,
            row_index if row_index is not None else topk_weights,
            topk_ids if topk_ids is not None else topk_weights,
            expert_mask if expert_mask is not None else topk_weights,
            topk_weights,
            norm_weight if norm_weight is not None else topk_weights,
            out,
            expert_mask.numel() if expert_mask is not None else 0,
            expert_out.shape[0],
            float(eps),
            float(gain_center),
            H=H,
            K=K,
            BLOCK_H=BLOCK_H,
            K_POW2=K_POW2,
            MODE=_mode(row_index, topk_ids),
            HAS_WEIGHT=norm_weight is not None,
            HAS_GAIN=gain_center != 0,
            INPUT_SCALE=bool(input_scale),
            num_warps=num_warps(BLOCK_H),
        )
        return out
    _fwd_kernel[(_programs(expert_out.device, 16, T), )](
        expert_out,
        topk_weights,
        topk_ids if topk_ids is not None else topk_weights,
        expert_mask if expert_mask is not None else topk_weights,
        topk_weights,
        norm_weight if norm_weight is not None else topk_weights,
        out,
        T,
        expert_mask.numel() if expert_mask is not None else 0,
        float(eps),
        float(gain_center),
        H=H,
        K=K,
        K_POW2=K_POW2,
        BLOCK_H=BLOCK_H,
        MODE=_mode(None, topk_ids),
        HAS_WEIGHT=norm_weight is not None,
        HAS_GAIN=gain_center != 0,
        INPUT_SCALE=bool(input_scale),
        num_warps=num_warps(BLOCK_H),
    )
    return out


def finalize_backward(grad_out, expert_out, topk_weights, row_index, topk_ids, expert_mask, norm_weight, input_scale,
                      eps, gain_center=0.0, grad_expert_out=None):
    """Gradients of ``expert_out`` (written into ``grad_expert_out`` when given), ``topk_weights`` and
    ``norm_weight`` (None without a weight), without autograd."""
    T, K = topk_weights.shape
    H = expert_out.shape[-1]
    grad_out = grad_out.contiguous()
    mode = _mode(row_index, topk_ids)
    if grad_expert_out is None:
        grad_expert_out = torch.empty_like(expert_out)
    named = topk_weights
    if mode == 2:
        grad_expert_out.zero_()
        named = torch.zeros(max(1, expert_out.shape[0]), device=expert_out.device, dtype=torch.int32)
    grad_topk_weights = torch.empty(T, K, device=expert_out.device, dtype=torch.float32)
    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_S = max(1, min(4, _BWD_TILE // BLOCK_H))
    nprog = _programs(expert_out.device, _BWD_PER_SM, triton.cdiv(T * K, BLOCK_S))
    has_weight = norm_weight is not None
    dw_part = torch.empty((nprog, H) if has_weight else (1, ), device=expert_out.device, dtype=torch.float32)
    if T:
        _bwd_kernel[(nprog, )](
            grad_out,
            expert_out,
            row_index if row_index is not None else topk_weights,
            topk_ids if topk_ids is not None else topk_weights,
            expert_mask if expert_mask is not None else topk_weights,
            topk_weights,
            norm_weight if has_weight else topk_weights,
            grad_expert_out,
            grad_topk_weights,
            dw_part,
            named,
            T * K,
            expert_mask.numel() if expert_mask is not None else 0,
            expert_out.shape[0],
            float(eps),
            float(gain_center),
            H=H,
            K=K,
            BLOCK_S=BLOCK_S,
            BLOCK_H=BLOCK_H,
            MODE=mode,
            HAS_WEIGHT=has_weight,
            HAS_GAIN=gain_center != 0,
            INPUT_SCALE=bool(input_scale),
            num_warps=_BWD_WARPS,
        )
    elif has_weight:
        dw_part.zero_()
    if mode == 2:
        _check_rows_named_once(named)
    grad_weight = dw_part.sum(0).to(norm_weight.dtype) if has_weight else None
    return grad_expert_out, grad_topk_weights, grad_weight


class _MoEFinalize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, expert_out, topk_weights, norm_weight, row_index, topk_ids, expert_mask, input_scale, eps,
                gain_center):
        ctx.save_for_backward(expert_out, topk_weights, norm_weight, row_index, topk_ids, expert_mask)
        ctx.input_scale, ctx.eps, ctx.gain_center = input_scale, eps, gain_center
        return finalize_forward(expert_out, topk_weights, row_index, topk_ids, expert_mask, norm_weight, input_scale,
                                eps, gain_center)

    @staticmethod
    def backward(ctx, grad_out):
        expert_out, topk_weights, norm_weight, row_index, topk_ids, expert_mask = ctx.saved_tensors
        gy, gs, gw = finalize_backward(grad_out, expert_out, topk_weights, row_index, topk_ids, expert_mask,
                                       norm_weight, ctx.input_scale, ctx.eps, ctx.gain_center)
        need = ctx.needs_input_grad
        return (gy if need[0] else None, gs if need[1] else None, gw if need[2] else None) + (None, ) * 6


def moe_finalize(expert_out, topk_weights, row_index, topk_ids, expert_mask, norm_weight, input_scale, eps, gain_center,
                 out):
    if out is None and torch.is_grad_enabled() and any(t is not None and t.requires_grad
                                                       for t in (expert_out, topk_weights, norm_weight)):
        return _MoEFinalize.apply(expert_out, topk_weights, norm_weight, row_index, topk_ids, expert_mask,
                                  bool(input_scale), float(eps), float(gain_center))
    return finalize_forward(expert_out, topk_weights, row_index, topk_ids, expert_mask, norm_weight, input_scale, eps,
                            gain_center, out)
