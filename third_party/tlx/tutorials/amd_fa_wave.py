"""Eight-wave gfx950 FlashAttention shaped after the standalone Wave kernel.

This remains separate from ``amd_fa_cluster`` because its packetized score and
output state is arranged specifically for two four-wave cohorts.  Four physical
K and V LDS slots let those cohorts run as a ping-pong pipeline without reusing
a slot that the other cohort still consumes.

The normal path uses an adaptive softmax reference.  A reference is retained
while new row maxima remain numerically safe and the accumulators are rescaled
only when it must advance.  Callers with a tight input bound may still request
the fixed-reference specialization explicitly through ``qk_max_abs``.  Bounds
whose full score span can leave the normal exponent range are rejected; those
inputs must use the adaptive path.
"""

import math

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.core import _aggregate as aggregate

BLOCK_M = tl.constexpr(256)
BLOCK_N = tl.constexpr(64)
HEAD_DIM = tl.constexpr(128)
LDS_STAGES = tl.constexpr(4)
XCDS = tl.constexpr(8)

# Let ``x_i`` denote a base-2-scaled attention logit.  After processing some
# tiles, the online-softmax state represented with reference ``r`` is
#
#     denominator = sum_i 2 ** (x_i - r)
#     accumulator = sum_i 2 ** (x_i - r) * v_i.
#
# Their ratio is the softmax result, independently of ``r``.  Conventional
# online softmax keeps ``r`` equal to the largest logit seen so far.  For a new
# tile with maximum ``m``, it chooses ``c = max(r, m)`` and changes reference by
# rescaling the old state with
#
#                         alpha = 2 ** (r - c).
#
# This kernel deliberately allows ``r`` to lag behind the true maximum.  If it
# retains ``r``, every term in both state components is larger than the state
# represented at ``c`` by the same common factor ``2 ** (c - r)``.  The factor
# cancels in ``accumulator / denominator``.  Therefore delaying the reference
# change neither clips nor approximates the softmax; in exact arithmetic it
# only selects a different common scale for its numerator and denominator.
#
# The reason not to retain ``r`` indefinitely is numerical range.  The largest
# positive exponent argument introduced by the new tile is bounded by
# ``m - r <= c - r``.  We retain ``r`` only while ``c - r <= H``, which caps
# every positive weight at ``2 ** H``.  With H=8 that cap is 256.  Once the
# bound is exceeded, the state is rebased to ``c`` using ``alpha`` exactly as
# in conventional online softmax.  H controls when that exact rebase occurs;
# it is not a score bound or an approximation tolerance.
SOFTMAX_REFERENCE_HEADROOM_LOG2 = tl.constexpr(8.0)

# The bounded specialization fixes its reference at the positive score bound,
# so its smallest exponent argument can be twice that bound below zero.  The
# gfx950 exp2 path flushes values below the f32 normal range; keep every accepted
# envelope strictly above that boundary to prevent a zero softmax denominator.
FIXED_REFERENCE_MAX_LOG2_SPAN = 126.0

# A BF16 product is exact in FP32.  Bound the 128 subsequent accumulation
# roundings with gamma_n = n*u/(1-n*u), u=2^-24, then keep the absolute dot sum
# strictly below the largest finite FP32 value before applying the softmax scale.
_FP32_UNIT_ROUNDOFF = 2.0**-24
_QK_ACCUMULATION_ERROR = (128 * _FP32_UNIT_ROUNDOFF) / (1.0 - 128 * _FP32_UNIT_ROUNDOFF)
MAX_QK_ABS_FOR_FINITE_F32_DOT = math.sqrt(torch.finfo(torch.float32).max / (128 * (1.0 + _QK_ACCUMULATION_ERROR)))

# The current symbolic address expressions are signed i32 through the Wave
# bridge.  Keep both the materialized tensor stride and every flattened element
# offset representable until the kernel migrates those expressions to i64.
MAX_SIGNED_I32 = (1 << 31) - 1

RESERVED_LAUNCH_OPTIONS = frozenset({
    "Q",
    "K",
    "V",
    "Out",
    "grid",
    "N_CTX",
    "BATCH",
    "HEADS",
    "TOTAL_HEADS",
    "SM_SCALE",
    "LOG2_SCORE_BOUND",
    "ADAPTIVE_REFERENCE",
    "num_warps",
})


@triton.jit
def _sum_combine(lhs, rhs):
    return lhs + rhs


@triton.jit
def _max_combine(lhs, rhs):
    return tl.maximum(lhs, rhs, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _split_last_2(value):
    rows: tl.constexpr = value.shape[0]
    columns: tl.constexpr = value.shape[1]
    return value.reshape([rows, 2, columns // 2]).permute(0, 2, 1).split()


@triton.jit
def _split_last_4(value):
    lower, upper = _split_last_2(value)
    value0, value1 = _split_last_2(lower)
    value2, value3 = _split_last_2(upper)
    return value0, value1, value2, value3


@triton.jit
def _split_last_16(value):
    value0, value1, value2, value3 = _split_last_4(value)
    value00, value01, value02, value03 = _split_last_4(value0)
    value04, value05, value06, value07 = _split_last_4(value1)
    value08, value09, value10, value11 = _split_last_4(value2)
    value12, value13, value14, value15 = _split_last_4(value3)
    return (
        value00,
        value01,
        value02,
        value03,
        value04,
        value05,
        value06,
        value07,
        value08,
        value09,
        value10,
        value11,
        value12,
        value13,
        value14,
        value15,
    )


@triton.jit
def _join_last_2(lower, upper):
    rows: tl.constexpr = lower.shape[0]
    columns: tl.constexpr = lower.shape[1]
    return tl.join(lower, upper).permute(0, 2, 1).reshape([rows, columns * 2])


@triton.jit
def _mfma_packet_to_registers(packet):
    """Expose one 32-column MFMA packet's physical register axis.

    The binary axes of each transposed 32x32 MFMA accumulator packet are:

      row[7:0], column[4:0]
        = warp[2:0], lane[4:0], register[3:2],
          lane[5], register[1:0].

    Moving lane[5] next to the other workitem bits produces the standalone
    Wave kernel's ``[workitem, register]`` packet without exchanging values
    between workitems.
    """
    binary = packet.reshape([2] * 13)
    workitem_register = binary.permute(
        0,
        1,
        2,
        10,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        12,
    )
    return workitem_register.reshape([8 * 64, 16])


@triton.jit
def _registers_to_mfma_packet(registers):
    """Invert :func:`_mfma_packet_to_registers`."""
    binary = registers.reshape([2] * 13)
    logical = binary.permute(
        0,
        1,
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        3,
        11,
        12,
    )
    return logical.reshape([BLOCK_M, BLOCK_N // 2])


@triton.jit
def _pin_workitem_layout(value):
    workitem_layout: tl.constexpr = (tlx.distributed_linear_layout_encoding.make(
        reg_bases=[],
        lane_bases=[
            [1],
            [2],
            [4],
            [8],
            [16],
            [32],
        ],
        warp_bases=[
            [64],
            [128],
            [256],
        ],
        block_bases=[],
        shape=[8 * 64],
    ))
    return tlx.release_layout(tlx.require_layout(value, workitem_layout))


@triton.jit
def _duplicate_rows_to_workitems(rows):
    per_wave_rows = rows.reshape([8, 32])
    per_workitem = tl.broadcast_to(per_wave_rows[:, None, :], (8, 2, 32))
    return _pin_workitem_layout(per_workitem.reshape([8 * 64]))


@triton.jit
def _reduce_score_registers(registers0, registers1):
    scores0 = _registers_to_mfma_packet(registers0)
    scores1 = _registers_to_mfma_packet(registers1)
    scores = _join_last_2(scores0, scores1)
    return _duplicate_rows_to_workitems(tl.sum(scores, axis=1))


@triton.jit
def _reduce_max_score_registers(registers0, registers1):
    scores0 = _registers_to_mfma_packet(registers0)
    scores1 = _registers_to_mfma_packet(registers1)
    scores = _join_last_2(scores0, scores1)
    return _duplicate_rows_to_workitems(tl.max(scores, axis=1))


@triton.jit
def _prepare_adaptive_score_registers(scores, qk_scale: tl.constexpr):
    """Expose score registers and compute their scaled row maximum."""
    score0, score1 = _split_last_2(scores)
    registers0 = _mfma_packet_to_registers(score0)
    registers1 = _mfma_packet_to_registers(score1)
    if qk_scale < 0.0:
        registers0 = registers0 * qk_scale
        registers1 = registers1 * qk_scale
        tile_max = _reduce_max_score_registers(registers0, registers1)
    else:
        tile_max = _reduce_max_score_registers(registers0, registers1) * qk_scale
    return registers0, registers1, tile_max


@triton.jit
def _adaptive_rebase_decision(row_max, tile_max):
    """Return the next row maximum and one uniform rebase decision per wave.

    Each wave owns 32 rows, duplicated across its two 32-lane halves by the
    reduction.  ``warp_ballot == -1`` therefore means that all 32 rows remain
    within the logarithmic headroom.  If any row exceeds it, rebasing every row
    in the wave is algebraically safe: scaling its old denominator and
    accumulator by the same factor preserves their ratio.  Ordered comparison
    also makes a NaN take the rebase path and continue propagating.
    """
    candidate = tl.maximum(
        row_max,
        tile_max,
        propagate_nan=tl.PropagateNan.ALL,
    )
    advance = candidate - row_max
    row_is_within_headroom = advance <= SOFTMAX_REFERENCE_HEADROOM_LOG2
    needs_rebase = tlx.warp_ballot(row_is_within_headroom) != -1
    return candidate, needs_rebase


@triton.jit
def _scale_accumulator_rows(accumulator, scale):
    """Scale an MFMA accumulator through its physical register packet."""
    registers = _mfma_packet_to_registers(accumulator)
    registers = registers * scale[:, None]
    return _registers_to_mfma_packet(registers)


@triton.jit
def _pv_mfma(
    probabilities,
    values,
    accumulator,
    p_layout: tl.constexpr,
    v_layout: tl.constexpr,
    mma_layout: tl.constexpr,
):
    probabilities = tlx.require_layout(probabilities, p_layout)
    values = tlx.require_layout(values, v_layout)
    accumulator = tlx.require_layout(accumulator, mma_layout)
    accumulator = tl.dot(probabilities, values, accumulator)
    return tlx.release_layout(accumulator)


@triton.jit
def _load_value_fragment(
    value_view,
    wait_token,
    ROW: tl.constexpr,
    COLUMN: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    v_layout: tl.constexpr,
):
    fragment_view = tlx.local_slice(
        value_view,
        [ROW, COLUMN],
        [ROWS, COLUMNS],
    )
    fragment = tlx.local_load(
        fragment_view,
        token=wait_token,
        layout=v_layout,
        relaxed=True,
    )
    return tlx.release_layout(fragment)


@aggregate
class SoftmaxPending:
    registers0: tl.tensor
    registers1: tl.tensor


@aggregate
class ProbabilityFragments:
    p0: tl.tensor
    p1: tl.tensor
    p2: tl.tensor
    p3: tl.tensor


@aggregate
class SoftmaxState:
    acc0: tl.tensor
    acc1: tl.tensor
    acc2: tl.tensor
    acc3: tl.tensor
    row_max: tl.tensor
    row_sum: tl.tensor

    @triton.jit
    def create():
        return SoftmaxState(
            tl.zeros((BLOCK_M, HEAD_DIM // 4), tl.float32),
            tl.zeros((BLOCK_M, HEAD_DIM // 4), tl.float32),
            tl.zeros((BLOCK_M, HEAD_DIM // 4), tl.float32),
            tl.zeros((BLOCK_M, HEAD_DIM // 4), tl.float32),
            _pin_workitem_layout(tl.full((8 * 64, ), -1.0e30, tl.float32), ),
            _pin_workitem_layout(tl.zeros((8 * 64, ), tl.float32), ),
        )

    @triton.jit
    def rebase(self, reference):
        alpha_rows = tl.math.exp2(self.row_max - reference)
        return SoftmaxState(
            _scale_accumulator_rows(self.acc0, alpha_rows),
            _scale_accumulator_rows(self.acc1, alpha_rows),
            _scale_accumulator_rows(self.acc2, alpha_rows),
            _scale_accumulator_rows(self.acc3, alpha_rows),
            reference,
            self.row_sum * alpha_rows,
        )

    @triton.jit
    def prepare(
        self,
        scores,
        qk_scale: tl.constexpr,
        log2_score_bound: tl.constexpr,
        INITIAL: tl.constexpr,
        ADAPTIVE_REFERENCE: tl.constexpr,
    ):
        state = self
        if ADAPTIVE_REFERENCE:
            score_registers0, score_registers1, tile_max = _prepare_adaptive_score_registers(
                scores,
                qk_scale,
            )
            if INITIAL:
                reference = tile_max
            else:
                candidate, needs_rebase = _adaptive_rebase_decision(
                    self.row_max,
                    tile_max,
                )
                if needs_rebase:
                    reference = candidate
                    state = self.rebase(reference)
                else:
                    reference = self.row_max
            if INITIAL:
                state = SoftmaxState(
                    self.acc0,
                    self.acc1,
                    self.acc2,
                    self.acc3,
                    reference,
                    self.row_sum,
                )
            # Keep the scale and translation as a multiply-add expression.
            # The global fast-math policy permits contraction, so Wave can
            # form the same FMA as its native FA implementation without a
            # kernel-specific fused operation in the bridge.
            if qk_scale < 0.0:
                registers0 = score_registers0 - reference[:, None]
                registers1 = score_registers1 - reference[:, None]
            else:
                registers0 = score_registers0 * qk_scale - reference[:, None]
                registers1 = score_registers1 * qk_scale - reference[:, None]
        else:
            score0, score1 = _split_last_2(scores)
            score0 = score0 * qk_scale + (-log2_score_bound)
            score1 = score1 * qk_scale + (-log2_score_bound)
            registers0 = _mfma_packet_to_registers(score0)
            registers1 = _mfma_packet_to_registers(score1)

        registers0 = tl.math.exp2(registers0)
        registers1 = tl.math.exp2(registers1)
        return (
            state,
            SoftmaxPending(
                registers0,
                registers1,
            ),
        )

    @triton.jit
    def prepare_adaptive_pending(
        self,
        scores,
        qk_scale: tl.constexpr,
    ):
        """Prepare the next score tile without rebasing the current PV state.

        Keeping the rebase commit separate lets Wave overlap this independent
        score work with the remaining PV MFMAs.  The reference is still chosen
        by the same wave-uniform ballot as prepare; only the state scaling
        is deferred until the current PV tile is fully accumulated.
        """
        score_registers0, score_registers1, tile_max = _prepare_adaptive_score_registers(
            scores,
            qk_scale,
        )
        candidate, needs_rebase = _adaptive_rebase_decision(
            self.row_max,
            tile_max,
        )
        reference = tl.where(needs_rebase, candidate, self.row_max)
        if qk_scale < 0.0:
            registers0 = score_registers0 - reference[:, None]
            registers1 = score_registers1 - reference[:, None]
        else:
            registers0 = score_registers0 * qk_scale - reference[:, None]
            registers1 = score_registers1 * qk_scale - reference[:, None]

        registers0 = tl.math.exp2(registers0)
        registers1 = tl.math.exp2(registers1)
        return (
            SoftmaxPending(
                registers0,
                registers1,
            ),
            reference,
            needs_rebase,
        )

    @triton.jit
    def commit_adaptive_reference(
        self,
        reference,
        needs_rebase,
    ):
        state = self
        if needs_rebase:
            state = self.rebase(reference)
        return state

    @triton.jit
    def finish(
        self,
        pending,
        out_dtype: tl.constexpr,
    ):
        registers0 = pending.registers0
        registers1 = pending.registers1
        tile_sum = _reduce_score_registers(
            registers0,
            registers1,
        )
        row_sum = self.row_sum + tile_sum
        score_register_layout: tl.constexpr = (tlx.distributed_linear_layout_encoding.make(
            reg_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
            ],
            lane_bases=[
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
            ],
            warp_bases=[
                [64, 0],
                [128, 0],
                [256, 0],
            ],
            block_bases=[],
            shape=[8 * 64, 16],
        ))
        registers0 = tlx.require_layout(registers0, score_register_layout)
        registers1 = tlx.require_layout(registers1, score_register_layout)
        registers0 = tlx.release_layout(tlx.cast_preserve_layout(registers0, out_dtype))
        registers1 = tlx.release_layout(tlx.cast_preserve_layout(registers1, out_dtype))
        probabilities0 = _registers_to_mfma_packet(registers0)
        probabilities1 = _registers_to_mfma_packet(registers1)
        p0, p1 = _split_last_2(probabilities0)
        p2, p3 = _split_last_2(probabilities1)
        probabilities = ProbabilityFragments(
            p0,
            p1,
            p2,
            p3,
        )
        return (
            SoftmaxState(
                self.acc0,
                self.acc1,
                self.acc2,
                self.acc3,
                self.row_max,
                row_sum,
            ),
            probabilities,
        )


@triton.jit
def _load_value_fragments(
    value_view,
    wait_token,
    v_layout: tl.constexpr,
):
    return (
        _load_value_fragment(value_view, wait_token, 0, 0, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 0, 32, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 0, 64, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 0, 96, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 16, 0, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 16, 32, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 16, 64, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 16, 96, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 32, 0, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 32, 32, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 32, 64, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 32, 96, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 48, 0, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 48, 32, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 48, 64, 16, 32, v_layout),
        _load_value_fragment(value_view, wait_token, 48, 96, 16, 32, v_layout),
    )


@triton.jit
def _accumulate_prefix_body(
    state,
    probabilities,
    value_fragments,
    next_scores,
    p_layout: tl.constexpr,
    v_layout: tl.constexpr,
    mma_layout: tl.constexpr,
    qk_scale: tl.constexpr,
    log2_score_bound: tl.constexpr,
    ADAPTIVE_REFERENCE: tl.constexpr,
):
    """Accumulate one PV tile using the standalone kernel's 3+13 split.

    The split exposes individual MFMA candidates to Wave.  The barrier fixes
    only the three-instruction prefix; Wave remains free to interleave the
    remaining independent MFMAs with the next tile's bounded-softmax work.
    """
    p0 = probabilities.p0
    p1 = probabilities.p1
    p2 = probabilities.p2
    p3 = probabilities.p3
    (
        v00,
        v01,
        v02,
        v03,
        v10,
        v11,
        v12,
        v13,
        v20,
        v21,
        v22,
        v23,
        v30,
        v31,
        v32,
        v33,
    ) = value_fragments
    acc0 = state.acc0
    acc1 = state.acc1
    acc2 = state.acc2
    acc3 = state.acc3

    acc0 = _pv_mfma(p0, v00, acc0, p_layout, v_layout, mma_layout)
    acc1 = _pv_mfma(p0, v01, acc1, p_layout, v_layout, mma_layout)
    acc2 = _pv_mfma(p0, v02, acc2, p_layout, v_layout, mma_layout)
    tlx.sched_barrier()
    acc3 = _pv_mfma(p0, v03, acc3, p_layout, v_layout, mma_layout)
    acc0 = _pv_mfma(p1, v10, acc0, p_layout, v_layout, mma_layout)
    acc1 = _pv_mfma(p1, v11, acc1, p_layout, v_layout, mma_layout)
    acc2 = _pv_mfma(p1, v12, acc2, p_layout, v_layout, mma_layout)
    acc3 = _pv_mfma(p1, v13, acc3, p_layout, v_layout, mma_layout)
    if ADAPTIVE_REFERENCE:
        pending, reference, needs_rebase = state.prepare_adaptive_pending(
            next_scores,
            qk_scale,
        )
    acc0 = _pv_mfma(p2, v20, acc0, p_layout, v_layout, mma_layout)
    acc1 = _pv_mfma(p2, v21, acc1, p_layout, v_layout, mma_layout)
    acc2 = _pv_mfma(p2, v22, acc2, p_layout, v_layout, mma_layout)
    acc3 = _pv_mfma(p2, v23, acc3, p_layout, v_layout, mma_layout)
    acc0 = _pv_mfma(p3, v30, acc0, p_layout, v_layout, mma_layout)
    acc1 = _pv_mfma(p3, v31, acc1, p_layout, v_layout, mma_layout)
    acc2 = _pv_mfma(p3, v32, acc2, p_layout, v_layout, mma_layout)
    acc3 = _pv_mfma(p3, v33, acc3, p_layout, v_layout, mma_layout)
    state = SoftmaxState(
        acc0,
        acc1,
        acc2,
        acc3,
        state.row_max,
        state.row_sum,
    )
    if ADAPTIVE_REFERENCE:
        return (
            state.commit_adaptive_reference(reference, needs_rebase),
            pending,
        )
    return state.prepare(
        next_scores,
        qk_scale,
        log2_score_bound,
        False,
        ADAPTIVE_REFERENCE,
    )


@triton.jit
def _accumulate_value_fragments(
    state,
    probabilities,
    value_fragments,
    p_layout: tl.constexpr,
    v_layout: tl.constexpr,
    mma_layout: tl.constexpr,
):
    p0 = probabilities.p0
    p1 = probabilities.p1
    p2 = probabilities.p2
    p3 = probabilities.p3
    (
        v00,
        v01,
        v02,
        v03,
        v10,
        v11,
        v12,
        v13,
        v20,
        v21,
        v22,
        v23,
        v30,
        v31,
        v32,
        v33,
    ) = value_fragments
    acc0 = _pv_mfma(p0, v00, state.acc0, p_layout, v_layout, mma_layout)
    acc1 = _pv_mfma(p0, v01, state.acc1, p_layout, v_layout, mma_layout)
    acc2 = _pv_mfma(p0, v02, state.acc2, p_layout, v_layout, mma_layout)
    acc3 = _pv_mfma(p0, v03, state.acc3, p_layout, v_layout, mma_layout)
    acc0 = _pv_mfma(p1, v10, acc0, p_layout, v_layout, mma_layout)
    acc1 = _pv_mfma(p1, v11, acc1, p_layout, v_layout, mma_layout)
    acc2 = _pv_mfma(p1, v12, acc2, p_layout, v_layout, mma_layout)
    acc3 = _pv_mfma(p1, v13, acc3, p_layout, v_layout, mma_layout)
    acc0 = _pv_mfma(p2, v20, acc0, p_layout, v_layout, mma_layout)
    acc1 = _pv_mfma(p2, v21, acc1, p_layout, v_layout, mma_layout)
    acc2 = _pv_mfma(p2, v22, acc2, p_layout, v_layout, mma_layout)
    acc3 = _pv_mfma(p2, v23, acc3, p_layout, v_layout, mma_layout)
    acc0 = _pv_mfma(p3, v30, acc0, p_layout, v_layout, mma_layout)
    acc1 = _pv_mfma(p3, v31, acc1, p_layout, v_layout, mma_layout)
    acc2 = _pv_mfma(p3, v32, acc2, p_layout, v_layout, mma_layout)
    acc3 = _pv_mfma(p3, v33, acc3, p_layout, v_layout, mma_layout)
    return SoftmaxState(
        acc0,
        acc1,
        acc2,
        acc3,
        state.row_max,
        state.row_sum,
    )


@triton.jit
def _load_query(
    Q,
    q_base,
    pid_m,
    stride_qm,
    q_load_layout: tl.constexpr,
    q_layout: tl.constexpr,
):
    """Load Q directly in MFMA operand ownership.

    Pinning the offsets and result prevents a 64 KiB blocked-to-dot scratch
    redistribution, which would not fit beside the four K/V stage rings.
    """
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offsets = offs_m[:, None] * stride_qm + offs_d[None, :]
    offsets = tlx.require_layout(offsets, q_load_layout)
    q = tlx.buffer_load(Q + q_base, offsets)
    q = tlx.require_layout(q, q_layout)
    return tlx.release_layout(q)


@triton.jit
def _normalize_output_fragment(
    inverse_row_sum,
    acc,
    out_dtype: tl.constexpr,
    mma_layout: tl.constexpr,
):
    acc = tlx.require_layout(acc, mma_layout)
    register_layout: tl.constexpr = (tlx.distributed_linear_layout_encoding.make(
        reg_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
        ],
        lane_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
        ],
        warp_bases=[
            [64, 0],
            [128, 0],
            [256, 0],
        ],
        block_bases=[],
        shape=[8 * 64, 16],
    ))
    binary = acc.reshape([2] * 13)
    registers = binary.permute(
        0,
        1,
        2,
        10,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        12,
    ).reshape([8 * 64, 16])
    scale = tl.broadcast_to(inverse_row_sum[:, None], (8 * 64, 16))
    scale = tlx.require_layout(scale, register_layout)
    registers = registers * scale
    registers = tlx.cast_preserve_layout(registers, out_dtype)
    binary = registers.reshape([2] * 13)
    output = binary.permute(
        0,
        1,
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        3,
        11,
        12,
    ).reshape([BLOCK_M, BLOCK_N // 2])
    return tlx.release_layout(output)


@triton.jit
def _store_output(
    Out,
    o_base,
    pid_m,
    stride_om: tl.constexpr,
    row_sum,
    acc0,
    acc1,
    acc2,
    acc3,
    store_layout: tl.constexpr,
    mma_layout: tl.constexpr,
):
    # Each lane initially owns sixteen values from each 32-column MFMA packet.
    # Join the four packets and redistribute them to the coalesced IO layout:
    # lane bits 0..4 select the row, lane bit 5 selects columns 0/8, and the
    # first three register bits select eight consecutive columns.  Each lane
    # can then publish its output using eight 128-bit stores.
    inverse_row_sum = 1.0 / row_sum
    output0 = _normalize_output_fragment(
        inverse_row_sum,
        acc0,
        Out.dtype.element_ty,
        mma_layout,
    )
    output1 = _normalize_output_fragment(
        inverse_row_sum,
        acc1,
        Out.dtype.element_ty,
        mma_layout,
    )
    output2 = _normalize_output_fragment(
        inverse_row_sum,
        acc2,
        Out.dtype.element_ty,
        mma_layout,
    )
    output3 = _normalize_output_fragment(
        inverse_row_sum,
        acc3,
        Out.dtype.element_ty,
        mma_layout,
    )
    output01 = _join_last_2(output0, output1)
    output23 = _join_last_2(output2, output3)
    output = _join_last_2(output01, output23)
    output = tlx.require_layout(output, store_layout)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offsets = offs_m[:, None] * stride_om + offs_d[None, :]
    offsets = tlx.require_layout(offsets, store_layout)
    tlx.buffer_store(
        output,
        Out + o_base,
        offsets,
    )


@triton.jit
def _issue_tile(
    pointers,
    tile,
    stride_n,
    buffer,
    slot,
    LDS_PADDING: tl.constexpr,
):
    # The padding value is part of the JIT specialization key.  K and V use
    # distinct padded memdesc layouts even though issuing the DMA itself does
    # not inspect the padding value.
    _ = LDS_PADDING
    tile_offset = tile * BLOCK_N * stride_n
    token = tlx.async_load(
        pointers + tile_offset,
        tlx.local_view(buffer, slot),
    )
    return tlx.async_load_commit_group([token])


@triton.jit
def _stage_end():
    """Match standalone Wave's CTA-convergence boundary."""
    tl.debug_barrier()


@triton.jit
def _wait_stage_end(PENDING: tl.constexpr, tokens):
    """Publish an explicit async wait at a standalone stage boundary."""
    ready = tlx.async_load_wait_group(PENDING, tokens=tokens)
    return ready


@triton.jit
def _score_phase(
    state,
    probabilities,
    current_ready,
    q,
    k_buffer,
    current_k_slot: tl.constexpr,
    q_layout: tl.constexpr,
    k_layout: tl.constexpr,
    mma_layout: tl.constexpr,
):
    q = tlx.require_layout(q, q_layout)
    current_k = tlx.local_load(
        tlx.local_trans(tlx.local_view(k_buffer, current_k_slot)),
        token=current_ready,
        layout=k_layout,
        relaxed=True,
    )
    state, p_dot = state.finish(
        probabilities,
        q.dtype,
    )
    _stage_end()
    score_acc = tlx.zeros(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        layout=mma_layout,
    )
    next_scores = tl.dot(q, current_k, score_acc)
    next_scores = tlx.release_layout(next_scores)
    q = tlx.release_layout(q)
    return state, p_dot, next_scores


@triton.jit
def _warp_pipeline_phase(
    state,
    probabilities,
    q,
    k_ptrs,
    v_ptrs,
    k_buffer,
    v_buffer,
    stride_kn,
    stride_vn,
    tile,
    current_k_ready,
    previous_v_ready,
    prefetched_v_ready,
    PHASE: tl.constexpr,
    q_layout: tl.constexpr,
    k_layout: tl.constexpr,
    p_layout: tl.constexpr,
    v_layout: tl.constexpr,
    mma_layout: tl.constexpr,
    qk_scale: tl.constexpr,
    log2_score_bound: tl.constexpr,
    ADAPTIVE_REFERENCE: tl.constexpr,
):
    previous_v_slot: tl.constexpr = PHASE
    current_k_slot: tl.constexpr = (PHASE + 1) % LDS_STAGES
    k_prefetch_slot: tl.constexpr = (PHASE + 3) % LDS_STAGES
    v_prefetch_slot: tl.constexpr = (PHASE + 2) % LDS_STAGES

    current_ready = tlx.async_load_wait_group(
        2,
        tokens=[current_k_ready, previous_v_ready],
    )

    with tlx.warp_pipeline_stage("softmax"):
        q = tlx.require_layout(q, q_layout)
        current_k = tlx.local_load(
            tlx.local_trans(tlx.local_view(k_buffer, current_k_slot)),
            token=current_ready,
            layout=k_layout,
            relaxed=True,
        )
        state, p_dot = state.finish(
            probabilities,
            q.dtype,
        )

    with tlx.warp_pipeline_stage("qk"):
        next_k_ready = _issue_tile(
            k_ptrs,
            tile + 3,
            stride_kn,
            k_buffer,
            k_prefetch_slot,
            8,
        )
        score_acc = tlx.zeros(
            (BLOCK_M, BLOCK_N),
            tl.float32,
            layout=mma_layout,
        )
        next_scores = tl.dot(q, current_k, score_acc)
        next_scores = tlx.release_layout(next_scores)
        q = tlx.release_layout(q)

    with tlx.warp_pipeline_stage("value"):
        value_fragments = _load_value_fragments(
            tlx.local_view(v_buffer, previous_v_slot),
            current_ready,
            v_layout,
        )

    with tlx.warp_pipeline_stage("pv"):
        state, probabilities = _accumulate_prefix_body(
            state,
            p_dot,
            value_fragments,
            next_scores,
            p_layout,
            v_layout,
            mma_layout,
            qk_scale,
            log2_score_bound,
            ADAPTIVE_REFERENCE,
        )
        future_v_ready = _issue_tile(
            v_ptrs,
            tile + 2,
            stride_vn,
            v_buffer,
            v_prefetch_slot,
            32,
        )

    return (
        state,
        probabilities,
        next_k_ready,
        prefetched_v_ready,
        future_v_ready,
    )


@triton.jit
def _pipeline(
    state,
    q,
    k_ptrs,
    v_ptrs,
    k_buffer,
    v_buffer,
    stride_kn,
    stride_vn,
    tile_count,
    q_layout: tl.constexpr,
    k_layout: tl.constexpr,
    p_layout: tl.constexpr,
    v_layout: tl.constexpr,
    mma_layout: tl.constexpr,
    qk_scale: tl.constexpr,
    log2_score_bound: tl.constexpr,
    ADAPTIVE_REFERENCE: tl.constexpr,
):
    q = tlx.require_layout(q, q_layout)
    # Prime K0, V0, and K1.  Every completion used by a local load below comes
    # from an explicit wait_group; LDS aliasing is never used as a dependency.
    k_ready0 = _issue_tile(k_ptrs, 0, stride_kn, k_buffer, 0, 8)
    v_ready0 = _issue_tile(v_ptrs, 0, stride_vn, v_buffer, 0, 32)
    k_ready1 = _issue_tile(k_ptrs, 1, stride_kn, k_buffer, 1, 8)

    # Only K0 is required for the first QK.  A FIFO wait retains V0 and K1 as
    # issue-order dependencies without also demanding their completion.  The
    # K0 local load consumes this explicit wait result; later V0/K1 loads
    # consume the next explicit wait result.
    wait_k0 = tlx.async_load_wait_group(2)
    k0 = tlx.local_load(
        tlx.local_trans(tlx.local_view(k_buffer, 0)),
        token=wait_k0,
        layout=k_layout,
        relaxed=True,
    )
    _stage_end()
    score_acc = tlx.zeros(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        layout=mma_layout,
    )
    scores = tl.dot(q, k0, score_acc)
    scores = tlx.release_layout(scores)
    _stage_end()
    state, probabilities = state.prepare(
        scores,
        qk_scale,
        log2_score_bound,
        True,
        ADAPTIVE_REFERENCE,
    )

    # K2 has its own physical slot; unlike the legacy cluster kernel, no
    # barrier-and-overwrite of K0 is needed here.
    k_ready2 = _issue_tile(k_ptrs, 2, stride_kn, k_buffer, 2, 8)
    # Tile i consumes V[i] and K[i+1], reads K[i+2], and publishes K[i+3] and
    # V[i+2].  Spell out all four ring phases so each physical slot remains
    # static in the generated Wave program.
    # Peel phase zero so the loop enters with real PV accumulator values.  In
    # addition to matching the four-slot ring, this lets Wave keep the MFMA
    # accumulator register groups intact across the loop backedge.
    q = tlx.release_layout(q)
    v_ready1 = _issue_tile(
        v_ptrs,
        1,
        stride_vn,
        v_buffer,
        1,
        32,
    )
    (
        state,
        probabilities,
        k_ready3,
        previous_v_ready,
        prefetched_v_ready,
    ) = _warp_pipeline_phase(
        state,
        probabilities,
        q,
        k_ptrs,
        v_ptrs,
        k_buffer,
        v_buffer,
        stride_kn,
        stride_vn,
        0,
        k_ready1,
        v_ready0,
        v_ready1,
        0,
        q_layout,
        k_layout,
        p_layout,
        v_layout,
        mma_layout,
        qk_scale,
        log2_score_bound,
        ADAPTIVE_REFERENCE,
    )
    for first_tile in tl.range(1, tile_count - 3, 4, num_stages=0):
        (
            state,
            probabilities,
            k_ready0,
            previous_v_ready,
            prefetched_v_ready,
        ) = _warp_pipeline_phase(
            state,
            probabilities,
            q,
            k_ptrs,
            v_ptrs,
            k_buffer,
            v_buffer,
            stride_kn,
            stride_vn,
            first_tile,
            k_ready2,
            previous_v_ready,
            prefetched_v_ready,
            1,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            mma_layout,
            qk_scale,
            log2_score_bound,
            ADAPTIVE_REFERENCE,
        )
        (
            state,
            probabilities,
            k_ready1,
            previous_v_ready,
            prefetched_v_ready,
        ) = _warp_pipeline_phase(
            state,
            probabilities,
            q,
            k_ptrs,
            v_ptrs,
            k_buffer,
            v_buffer,
            stride_kn,
            stride_vn,
            first_tile + 1,
            k_ready3,
            previous_v_ready,
            prefetched_v_ready,
            2,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            mma_layout,
            qk_scale,
            log2_score_bound,
            ADAPTIVE_REFERENCE,
        )
        (
            state,
            probabilities,
            k_ready2,
            previous_v_ready,
            prefetched_v_ready,
        ) = _warp_pipeline_phase(
            state,
            probabilities,
            q,
            k_ptrs,
            v_ptrs,
            k_buffer,
            v_buffer,
            stride_kn,
            stride_vn,
            first_tile + 2,
            k_ready0,
            previous_v_ready,
            prefetched_v_ready,
            3,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            mma_layout,
            qk_scale,
            log2_score_bound,
            ADAPTIVE_REFERENCE,
        )
        (
            state,
            probabilities,
            k_ready3,
            previous_v_ready,
            prefetched_v_ready,
        ) = _warp_pipeline_phase(
            state,
            probabilities,
            q,
            k_ptrs,
            v_ptrs,
            k_buffer,
            v_buffer,
            stride_kn,
            stride_vn,
            first_tile + 3,
            k_ready1,
            previous_v_ready,
            prefetched_v_ready,
            0,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            mma_layout,
            qk_scale,
            log2_score_bound,
            ADAPTIVE_REFERENCE,
        )
    # Drain the final three output tiles without out-of-range DMA requests.
    tile_nm3 = tile_count - 3
    tile_nm2 = tile_count - 2
    tile_nm1 = tile_count - 1

    v_ready_nm2 = prefetched_v_ready
    ready_nm3 = _wait_stage_end(2, [k_ready2, previous_v_ready])
    state, p_dot, next_scores = _score_phase(
        state,
        probabilities,
        ready_nm3,
        q,
        k_buffer,
        tile_nm2 % LDS_STAGES,
        q_layout,
        k_layout,
        mma_layout,
    )
    _stage_end()
    value_fragments = _load_value_fragments(
        tlx.local_view(v_buffer, tile_nm3 % LDS_STAGES),
        ready_nm3,
        v_layout,
    )
    _stage_end()
    state, probabilities = _accumulate_prefix_body(
        state,
        p_dot,
        value_fragments,
        next_scores,
        p_layout,
        v_layout,
        mma_layout,
        qk_scale,
        log2_score_bound,
        ADAPTIVE_REFERENCE,
    )

    v_ready_nm1 = _issue_tile(
        v_ptrs,
        tile_nm1,
        stride_vn,
        v_buffer,
        tile_nm1 % LDS_STAGES,
        32,
    )
    ready_nm2 = _wait_stage_end(1, [k_ready3, v_ready_nm2])
    state, p_dot, next_scores = _score_phase(
        state,
        probabilities,
        ready_nm2,
        q,
        k_buffer,
        tile_nm1 % LDS_STAGES,
        q_layout,
        k_layout,
        mma_layout,
    )
    _stage_end()
    value_fragments = _load_value_fragments(
        tlx.local_view(v_buffer, tile_nm2 % LDS_STAGES),
        ready_nm2,
        v_layout,
    )
    _stage_end()
    state, probabilities = _accumulate_prefix_body(
        state,
        p_dot,
        value_fragments,
        next_scores,
        p_layout,
        v_layout,
        mma_layout,
        qk_scale,
        log2_score_bound,
        ADAPTIVE_REFERENCE,
    )

    state, p_dot = state.finish(
        probabilities,
        q.dtype,
    )
    ready_nm1 = _wait_stage_end(0, [v_ready_nm1])
    value_fragments = _load_value_fragments(
        tlx.local_view(v_buffer, tile_nm1 % LDS_STAGES),
        ready_nm1,
        v_layout,
    )
    _stage_end()
    state = _accumulate_value_fragments(
        state,
        p_dot,
        value_fragments,
        p_layout,
        v_layout,
        mma_layout,
    )
    return state


@triton.jit
def _attn_fwd_wave_pipeline(
    Q,
    K,
    V,
    Out,
    N_CTX: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    SM_SCALE: tl.constexpr,
    LOG2_SCORE_BOUND: tl.constexpr,
    ADAPTIVE_REFERENCE: tl.constexpr,
):
    stride_m: tl.constexpr = HEAD_DIM
    stride_h: tl.constexpr = N_CTX * HEAD_DIM
    stride_z: tl.constexpr = HEADS * stride_h

    raw_m = tl.program_id(0)
    raw_head = tl.program_id(1)
    m_blocks: tl.constexpr = N_CTX // BLOCK_M
    total_programs: tl.constexpr = m_blocks * TOTAL_HEADS
    tl.assume(raw_m < m_blocks)
    tl.assume(raw_head < TOTAL_HEADS)
    raw_pid = raw_head * m_blocks + raw_m
    if total_programs % XCDS == 0:
        programs_per_xcd: tl.constexpr = total_programs // XCDS
        pid = (raw_pid % XCDS) * programs_per_xcd + raw_pid // XCDS
    else:
        pid = raw_pid
    tl.assume(pid < total_programs)
    pid_m = pid % m_blocks
    flat_head = pid // m_blocks
    head = flat_head % HEADS
    batch = flat_head // HEADS
    tl.assume(pid_m < N_CTX // BLOCK_M)
    tl.assume(flat_head < TOTAL_HEADS)
    tl.assume(head < HEADS)
    tl.assume(batch < BATCH)

    q_base = batch * stride_z + head * stride_h
    k_base = batch * stride_z + head * stride_h
    v_base = batch * stride_z + head * stride_h
    o_base = batch * stride_z + head * stride_h

    mma_layout: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[8, 1],
    )
    q_layout: tl.constexpr = tlx.dot_operand_layout(
        0,
        mma_layout,
        k_width=8,
    )
    q_load_layout: tl.constexpr = (tlx.distributed_linear_layout_encoding.make(
        reg_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 16],
            [0, 32],
            [0, 64],
        ],
        lane_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [0, 8],
        ],
        warp_bases=[
            [32, 0],
            [64, 0],
            [128, 0],
        ],
        block_bases=[],
        shape=[256, 128],
    ))
    k_layout: tl.constexpr = tlx.dot_operand_layout(
        1,
        mma_layout,
        k_width=8,
    )
    p_layout: tl.constexpr = tlx.dot_operand_layout(
        0,
        mma_layout,
        k_width=4,
    )
    v_layout: tl.constexpr = tlx.dot_operand_layout(
        1,
        mma_layout,
        k_width=4,
    )
    q = _load_query(
        Q,
        q_base,
        pid_m,
        stride_m,
        q_load_layout,
        q_layout,
    )
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    k_ptrs = K + k_base + offs_n[:, None] * stride_m + offs_d[None, :]
    v_ptrs = V + v_base + offs_n[:, None] * stride_m + offs_d[None, :]
    k_shared_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
        [(512, 8)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [8, 0],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [0, 64],
        ],
        [BLOCK_N, HEAD_DIM],
    ))
    v_shared_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [8, 0],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [0, 64],
        ],
        [BLOCK_N, HEAD_DIM],
    ))
    k_buffer = tlx.local_alloc(
        (BLOCK_N, HEAD_DIM),
        K.dtype.element_ty,
        LDS_STAGES,
        layout=k_shared_layout,
    )
    v_buffer = tlx.local_alloc(
        (BLOCK_N, HEAD_DIM),
        V.dtype.element_ty,
        LDS_STAGES,
        layout=v_shared_layout,
    )
    state = SoftmaxState.create()
    state = _pipeline(
        state,
        q,
        k_ptrs,
        v_ptrs,
        k_buffer,
        v_buffer,
        stride_m,
        stride_m,
        N_CTX // BLOCK_N,
        q_layout,
        k_layout,
        p_layout,
        v_layout,
        mma_layout,
        SM_SCALE * 1.4426950408889634,
        LOG2_SCORE_BOUND,
        ADAPTIVE_REFERENCE,
    )

    _store_output(
        Out,
        o_base,
        pid_m,
        stride_m,
        state.row_sum,
        state.acc0,
        state.acc1,
        state.acc2,
        state.acc3,
        q_load_layout,
        mma_layout,
    )


def _storage_ranges_overlap(lhs, rhs):
    """Return whether two contiguous tensors address any common byte."""
    if lhs.device != rhs.device or lhs.numel() == 0 or rhs.numel() == 0:
        return False
    lhs_begin = lhs.data_ptr()
    rhs_begin = rhs.data_ptr()
    lhs_end = lhs_begin + lhs.numel() * lhs.element_size()
    rhs_end = rhs_begin + rhs.numel() * rhs.element_size()
    return lhs_begin < rhs_end and rhs_begin < lhs_end


def attention(
    q,
    k,
    v,
    sm_scale=None,
    causal=False,
    *,
    qk_max_abs=None,
    out=None,
    warmup=False,
    **compiler_options,
):
    """Run the separate eight-wave attention kernel.

    The default adaptive reference is numerically stable for unrestricted
    inputs.  ``qk_max_abs`` explicitly selects the bounded specialization.
    ``out`` may be exactly ``q``, but it must not partially overlap ``q`` or
    overlap any part of ``k`` or ``v``.
    """
    reserved_options = sorted(RESERVED_LAUNCH_OPTIONS.intersection(compiler_options))
    if reserved_options:
        names = ", ".join(reserved_options)
        raise TypeError(f"amd_fa_wave compiler options must not override reserved launch keys: {names}")
    if causal:
        raise ValueError("amd_fa_wave only implements non-causal attention")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.ndim != 4:
            raise ValueError(f"{name} must be rank 4 (B, H, N, D), got rank {tensor.ndim}")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must have dtype torch.bfloat16, got {tensor.dtype}")
    for name, tensor in (("k", k), ("v", v)):
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q (q={q.device}, {name}={tensor.device})")
        if tensor.shape != q.shape:
            raise ValueError(f"{name} must have shape {tuple(q.shape)}, got {tuple(tensor.shape)}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    batch, heads, sequence, head_dim = q.shape
    if batch <= 0 or heads <= 0:
        raise ValueError(f"q batch and head dimensions must be positive, got B={batch}, H={heads}")
    if sequence < 256 or sequence % 256 != 0:
        raise ValueError(f"q sequence length must be a multiple of 256 and at least 256, got N={sequence}")
    if head_dim != 128:
        raise ValueError(f"q head dimension must be 128, got D={head_dim}")
    batch_stride = heads * sequence * head_dim
    last_element_offset = batch * batch_stride - 1
    if batch_stride > MAX_SIGNED_I32 or last_element_offset > MAX_SIGNED_I32:
        raise ValueError(f"amd_fa_wave batch stride ({batch_stride}) and last element offset "
                         f"({last_element_offset}) must not exceed the signed-i32 address limit "
                         f"({MAX_SIGNED_I32})")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    sm_scale = float(sm_scale)
    if not math.isfinite(sm_scale):
        raise ValueError("sm_scale must be finite")
    adaptive_reference = qk_max_abs is None
    if adaptive_reference:
        log2_score_bound = 0.0
    else:
        if not math.isfinite(qk_max_abs) or qk_max_abs <= 0:
            raise ValueError("qk_max_abs must be finite and greater than zero")
        if qk_max_abs >= MAX_QK_ABS_FOR_FINITE_F32_DOT:
            raise ValueError(f"qk_max_abs ({qk_max_abs:g}) exceeds the conservative raw FP32 QK limit "
                             f"({MAX_QK_ABS_FOR_FINITE_F32_DOT:g}) for head dimension 128")
        log2_score_bound = (math.log2(math.e) * head_dim * abs(sm_scale) * qk_max_abs * qk_max_abs)
        log2_score_span = 2.0 * abs(log2_score_bound)
        if not math.isfinite(log2_score_span) or log2_score_span >= FIXED_REFERENCE_MAX_LOG2_SPAN:
            raise ValueError(f"fixed-reference log2 score span ({log2_score_span:g}) must be less than "
                             f"{FIXED_REFERENCE_MAX_LOG2_SPAN:g}; use qk_max_abs=None for adaptive softmax")
    if out is None:
        output = torch.empty_like(q)
    else:
        output = out
        if output.ndim != 4:
            raise ValueError(f"out must be rank 4 (B, H, N, D), got rank {output.ndim}")
        if output.dtype != torch.bfloat16:
            raise ValueError(f"out must have dtype torch.bfloat16, got {output.dtype}")
        if output.device != q.device:
            raise ValueError(f"out must be on the same device as q (q={q.device}, out={output.device})")
        if output.shape != q.shape:
            raise ValueError(f"out must have shape {tuple(q.shape)}, got {tuple(output.shape)}")
        if not output.is_contiguous():
            raise ValueError("out must be contiguous")
    if _storage_ranges_overlap(output, k):
        raise ValueError("amd_fa_wave out must not overlap k")
    if _storage_ranges_overlap(output, v):
        raise ValueError("amd_fa_wave out must not overlap v")
    if _storage_ranges_overlap(output, q) and not output.is_set_to(q):
        raise ValueError("amd_fa_wave out may overlap q only when it is exactly the same tensor view")
    grid = (sequence // 256, batch * heads, 1)
    launch_options = {
        **compiler_options,
        "N_CTX": sequence,
        "BATCH": batch,
        "HEADS": heads,
        "TOTAL_HEADS": batch * heads,
        "SM_SCALE": sm_scale,
        "LOG2_SCORE_BOUND": log2_score_bound,
        "ADAPTIVE_REFERENCE": adaptive_reference,
        "num_warps": 8,
    }
    if triton.runtime.driver.active.get_current_target().backend == "tlx_wave":
        launch_options.setdefault("tlx_wave_enable_multi_wave_specialize", False)

    args = (q, k, v, output)
    if warmup:
        return _attn_fwd_wave_pipeline.warmup(
            *args,
            grid=grid,
            **launch_options,
        )
    _attn_fwd_wave_pipeline[grid](*args, **launch_options)
    return output


__all__ = ["attention"]
