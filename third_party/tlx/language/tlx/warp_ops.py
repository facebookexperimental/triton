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
        pred: A boolean predicate (i1) for each thread.

    Returns:
        A 32-bit integer where bit N is set if thread N's predicate was true
        and thread N is in the mask.

    Example:
        # Check if any thread in the warp has a non-zero value
        has_value = x != 0
        ballot = tlx.vote_ballot_sync(0xFFFFFFFF, has_value)
        # ballot will have bit N set if thread N has x != 0

    PTX instruction generated:
        vote.sync.ballot.b32 dest, predicate, membermask;

    Note:
        - Requires compute capability 3.0 or higher
        - All threads in mask must execute the instruction with identical mask
        - The sync variant ensures warp convergence before the vote
    """
    # Ensure pred is i1 type
    if pred.dtype != tl.int1:
        pred = pred != 0

    # Get mask as i32 value
    if isinstance(mask, tl.constexpr):
        mask_val = mask.value
    else:
        mask_val = mask

    mask_handle = _semantic.builder.get_int32(mask_val)
    result = _semantic.builder.vote_ballot_sync(mask_handle, pred.handle)
    return _semantic.tensor(result, tl.int32)
