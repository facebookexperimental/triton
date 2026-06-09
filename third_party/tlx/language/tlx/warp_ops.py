"""
TLX Warp-Level Operations

This module provides warp-level synchronization and voting primitives
for NVIDIA GPUs.
"""

import triton
import triton.language as tl_module
import triton.language.core as tl


@triton.jit
def warp_redux(x, op: tl.constexpr):
    """Warp-level reduction with implicit broadcast.

    Each element is replaced with the reduction result across all 32 lanes in
    its warp. The result is automatically replicated to every lane — no
    explicit broadcast needed.

    Requires SM100 (Blackwell) for f32 ops.

    Args:
        x: f32 tensor
        op: "max", "min", "abs_max", or "abs_min"
    """
    if op == "abs_max":
        return _warp_redux_abs_max(x)
    elif op == "abs_min":
        return _warp_redux_abs_min(x)
    elif op == "max":
        return _warp_redux_max(x)
    elif op == "min":
        return _warp_redux_min(x)


@triton.jit
def _warp_redux_max(x):
    return tl_module.inline_asm_elementwise(
        "{ .reg .b32 t; mov.b32 t, $1; redux.sync.max.f32 t, t, 0xFFFFFFFF; mov.b32 $0, t; }",
        "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _warp_redux_min(x):
    return tl_module.inline_asm_elementwise(
        "{ .reg .b32 t; mov.b32 t, $1; redux.sync.min.f32 t, t, 0xFFFFFFFF; mov.b32 $0, t; }",
        "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _warp_redux_abs_max(x):
    return tl_module.inline_asm_elementwise(
        "{ .reg .b32 t; mov.b32 t, $1; and.b32 t, t, 0x7FFFFFFF; redux.sync.max.f32 t, t, 0xFFFFFFFF; mov.b32 $0, t; }",
        "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _warp_redux_abs_min(x):
    return tl_module.inline_asm_elementwise(
        "{ .reg .b32 t; mov.b32 t, $1; and.b32 t, t, 0x7FFFFFFF; redux.sync.min.f32 t, t, 0xFFFFFFFF; mov.b32 $0, t; }",
        "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1,
    )


@tl.builtin
def vote_ballot_sync(
    mask: tl.constexpr,
    pred: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """
    Perform a warp-level vote ballot operation.

    Collects a predicate from each thread in the warp and returns a 32-bit
    mask where each bit represents the predicate value from the corresponding
    lane. Only threads specified by `mask` participate in the vote.

    Args:
        mask: A 32-bit mask specifying which threads participate. Threads with
              their corresponding bit set in the mask must execute with the
              same mask value. Use 0xFFFFFFFF for all threads.
        pred: A boolean predicate. Can be either a scalar i1 or a tensor of i1

    Returns:
        If pred is scalar: A 32-bit integer where bit N is set if thread N's
                          predicate was true and thread N is in the mask.
        If pred is tensor: A tensor of i32 with the same shape, where each
                          element contains the warp's ballot result.

    Example:
        # Scalar predicate - check if any thread has a non-zero value
        ballot = tlx.vote_ballot_sync(0xFFFFFFFF, x != 0)

        # Tensor predicate - it will be distributed to warps/threads according to layout
        pred_tensor = values < threshold  # tensor<128x1xi1>
        ballot = tlx.vote_ballot_sync(0xFFFFFFFF, pred_tensor)  # tensor<128x1xi32>

    PTX instruction generated:
        vote.sync.ballot.b32 dest, predicate, membermask;

    Note:
        - All threads in mask must execute the instruction with identical mask
        - The sync variant ensures warp convergence before the vote
    """
    if pred.dtype != tl.int1:
        pred = pred != 0

    mask_val = mask.value if isinstance(mask, tl.constexpr) else mask
    mask_handle = _semantic.builder.get_int32(mask_val)
    result = _semantic.builder.vote_ballot_sync(mask_handle, pred.handle)

    if pred.type.is_block():
        shape = [s.value if hasattr(s, "value") else s for s in pred.shape]
        ret_ty = tl.block_type(tl.int32, shape)
        return _semantic.tensor(result, ret_ty)
    return _semantic.tensor(result, tl.int32)
