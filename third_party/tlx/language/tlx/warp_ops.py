"""
TLX Warp-Level Operations

This module provides warp-level synchronization and voting primitives
for NVIDIA GPUs.
"""

import triton.language.core as tl


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
    # Ensure pred is i1/bool type
    if pred.dtype != tl.int1:
        pred = pred != 0

    # Get mask as i32 value
    if isinstance(mask, tl.constexpr):
        mask_val = mask.value
    else:
        mask_val = mask

    mask_handle = _semantic.builder.get_int32(mask_val)
    result = _semantic.builder.vote_ballot_sync(mask_handle, pred.handle)

    # Determine result type based on predicate type
    # If pred is a tensor, result will be tensor of i32 with same shape
    if pred.type.is_block():
        # Tensor case - create block_type with same shape but i32 element type
        shape = [s.value if hasattr(s, "value") else s for s in pred.shape]
        ret_ty = tl.block_type(tl.int32, shape)
        return _semantic.tensor(result, ret_ty)
    else:
        # Scalar case
        return _semantic.tensor(result, tl.int32)
