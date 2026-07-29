"""Eight-wave gfx950 FlashAttention shaped after the standalone Wave kernel.

This is a deliberately separate, bounded-input kernel rather than another mode
of ``amd_fa_cluster``.  It keeps four physical K and V LDS slots so Wave's two
four-wave cohorts can run as a ping-pong pipeline without reusing a slot that
the other cohort still consumes.

The fixed softmax reference is valid when every Q and K element satisfies
``abs(x) <= qk_max_abs``.  Subtracting that common reference from every score
does not change softmax, while removing the running-max rescale from the hot
loop.  The host wrapper makes the bound an explicit part of the API.
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


@triton.jit
def _sum_combine(lhs, rhs):
    return lhs + rhs


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
def _score_packet_to_registers(scores):
    """Expose one 32-column MFMA score packet's physical register axis.

    The binary axes of each transposed 32x32 MFMA accumulator packet are:

      row[7:0], column[4:0]
        = warp[2:0], lane[4:0], register[3:2],
          lane[5], register[1:0].

    Moving lane[5] next to the other workitem bits produces the standalone
    Wave kernel's ``[workitem, register]`` packet without exchanging values
    between workitems.
    """
    binary = scores.reshape([2] * 13)
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
def _registers_to_score_packet(registers):
    """Invert :func:`_score_packet_to_registers`."""
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
def _reduce_score_registers(registers0, registers1):
    """Match Wave's scalar component reduction before the lane-half exchange."""
    (
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
    ) = _split_last_16(registers0)
    (
        value16,
        value17,
        value18,
        value19,
        value20,
        value21,
        value22,
        value23,
        value24,
        value25,
        value26,
        value27,
        value28,
        value29,
        value30,
        value31,
    ) = _split_last_16(registers1)
    local_sum = value00 + value01
    local_sum += value02
    local_sum += value03
    local_sum += value04
    local_sum += value05
    local_sum += value06
    local_sum += value07
    local_sum += value08
    local_sum += value09
    local_sum += value10
    local_sum += value11
    local_sum += value12
    local_sum += value13
    local_sum += value14
    local_sum += value15
    local_sum += value16
    local_sum += value17
    local_sum += value18
    local_sum += value19
    local_sum += value20
    local_sum += value21
    local_sum += value22
    local_sum += value23
    local_sum += value24
    local_sum += value25
    local_sum += value26
    local_sum += value27
    local_sum += value28
    local_sum += value29
    local_sum += value30
    local_sum += value31

    # workitem = warp * 64 + lane.  Reshape it so lane bit 5 becomes
    # the two-element reduction axis, exactly matching Wave's xor-32 exchange.
    lane_halves = local_sum.reshape([8, 2, 32]).permute(0, 2, 1)
    lane_halves = lane_halves.reshape([BLOCK_M, 2])
    return tl.reduce(lane_halves, 1, _sum_combine)


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
class BoundedSoftmaxPending:
    score0: tl.tensor
    score1: tl.tensor


@aggregate
class ProbabilityFragments:
    p0: tl.tensor
    p1: tl.tensor
    p2: tl.tensor
    p3: tl.tensor


@aggregate
class BoundedSoftmaxState:
    acc0: tl.tensor
    acc1: tl.tensor
    acc2: tl.tensor
    acc3: tl.tensor
    row_sum: tl.tensor

    @triton.jit
    def create():
        return BoundedSoftmaxState(
            tl.zeros((BLOCK_M, HEAD_DIM // 4), tl.float32),
            tl.zeros((BLOCK_M, HEAD_DIM // 4), tl.float32),
            tl.zeros((BLOCK_M, HEAD_DIM // 4), tl.float32),
            tl.zeros((BLOCK_M, HEAD_DIM // 4), tl.float32),
            tl.zeros((BLOCK_M, ), tl.float32),
        )

    @triton.jit
    def prepare(
        self,
        scores,
        qk_scale: tl.constexpr,
        log2_score_bound: tl.constexpr,
        mma_layout: tl.constexpr,
    ):
        score0, score1 = _split_last_2(scores)
        score0 = tl.fma(score0, qk_scale, -log2_score_bound)
        registers0 = tl.math.exp2(_score_packet_to_registers(score0))
        score1 = tl.fma(score1, qk_scale, -log2_score_bound)
        registers1 = _score_packet_to_registers(score1)
        lower8, tail8 = _split_last_2(registers1)
        head4, upper4 = _split_last_2(lower8)
        lower2, tail2 = _split_last_2(upper4)
        head1, tail1 = _split_last_2(lower2)
        middle2 = _join_last_2(tl.math.exp2(head1), tail1)
        middle4 = _join_last_2(middle2, tail2)
        lower8 = _join_last_2(tl.math.exp2(head4), middle4)
        registers1 = _join_last_2(lower8, tail8)
        return BoundedSoftmaxPending(
            registers0,
            registers1,
        )

    @triton.jit
    def finish(
        self,
        pending,
        out_dtype: tl.constexpr,
    ):
        registers0 = pending.score0
        registers1 = pending.score1
        lower8, tail8 = _split_last_2(registers1)
        head4, upper4 = _split_last_2(lower8)
        lower2, tail2 = _split_last_2(upper4)
        head1, tail1 = _split_last_2(lower2)
        middle2 = _join_last_2(
            head1,
            tl.math.exp2(tail1),
        )
        middle4 = _join_last_2(middle2, tl.math.exp2(tail2))
        lower8 = _join_last_2(head4, middle4)
        registers1 = _join_last_2(lower8, tl.math.exp2(tail8))
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
        probabilities0 = _registers_to_score_packet(registers0)
        probabilities1 = _registers_to_score_packet(registers1)
        p0, p1 = _split_last_2(probabilities0)
        p2, p3 = _split_last_2(probabilities1)
        probabilities = ProbabilityFragments(
            p0,
            p1,
            p2,
            p3,
        )
        return (
            BoundedSoftmaxState(
                self.acc0,
                self.acc1,
                self.acc2,
                self.acc3,
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
    acc0 = _pv_mfma(p2, v20, acc0, p_layout, v_layout, mma_layout)
    acc1 = _pv_mfma(p2, v21, acc1, p_layout, v_layout, mma_layout)
    acc2 = _pv_mfma(p2, v22, acc2, p_layout, v_layout, mma_layout)
    acc3 = _pv_mfma(p2, v23, acc3, p_layout, v_layout, mma_layout)
    acc0 = _pv_mfma(p3, v30, acc0, p_layout, v_layout, mma_layout)
    acc1 = _pv_mfma(p3, v31, acc1, p_layout, v_layout, mma_layout)
    acc2 = _pv_mfma(p3, v32, acc2, p_layout, v_layout, mma_layout)
    acc3 = _pv_mfma(p3, v33, acc3, p_layout, v_layout, mma_layout)
    pending = state.prepare(
        next_scores,
        qk_scale,
        log2_score_bound,
        mma_layout,
    )

    return BoundedSoftmaxState(
        acc0,
        acc1,
        acc2,
        acc3,
        state.row_sum,
    ), pending


@triton.jit
def _accumulate_with_prefix_barrier(
    state,
    probabilities,
    value_fragments,
    next_scores,
    p_layout: tl.constexpr,
    v_layout: tl.constexpr,
    mma_layout: tl.constexpr,
    qk_scale: tl.constexpr,
    log2_score_bound: tl.constexpr,
):
    return _accumulate_prefix_body(
        state,
        probabilities,
        value_fragments,
        next_scores,
        p_layout,
        v_layout,
        mma_layout,
        qk_scale,
        log2_score_bound,
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
    return BoundedSoftmaxState(
        acc0,
        acc1,
        acc2,
        acc3,
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
def _store_output_fragment(
    Out,
    o_base,
    pid_m,
    stride_om: tl.constexpr,
    row_sum,
    acc,
    FRAGMENT: tl.constexpr,
    mma_layout: tl.constexpr,
):
    output = acc / row_sum[:, None]
    output = tlx.require_layout(output, mma_layout)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = FRAGMENT * (HEAD_DIM // 4) + tl.arange(0, HEAD_DIM // 4)
    offsets = offs_m[:, None] * stride_om + offs_d[None, :]
    offsets = tlx.require_layout(offsets, mma_layout)
    tlx.buffer_store(
        output.to(Out.dtype.element_ty),
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
    # Match the standalone kernel's common K/V copy ownership without
    # converting the pointer tensor into a long-lived register layout.
    copy_layout: tl.constexpr = (tlx.distributed_linear_layout_encoding.make(
        reg_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 64],
        ],
        lane_bases=[
            [0, 8],
            [0, 16],
            [0, 32],
            [8, 0],
            [16, 0],
            [32, 0],
        ],
        warp_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
        ],
        block_bases=[],
        shape=[BLOCK_N, HEAD_DIM],
    ))
    tile_offset = tile * BLOCK_N * stride_n
    token = tlx.async_load(
        pointers + tile_offset,
        tlx.local_view(buffer, slot),
        copy_layout=copy_layout,
    )
    return tlx.async_load_commit_group([token])


@triton.jit
def _stage_end():
    """Match standalone Wave's scheduling and CTA-convergence boundary."""
    tlx.sched_barrier()
    tl.debug_barrier()
    tlx.sched_barrier()


@triton.jit
def _wait_stage_end(PENDING: tl.constexpr, tokens):
    """Publish an explicit async wait at a standalone stage boundary."""
    tlx.sched_barrier()
    ready = tlx.async_load_wait_group(PENDING, tokens=tokens)
    tlx.sched_barrier()
    return ready


@triton.jit
def _score_phase(
    state,
    probabilities,
    current_ready,
    q,
    k_ptrs,
    k_buffer,
    stride_kn,
    tile,
    current_k_slot: tl.constexpr,
    k_prefetch_slot: tl.constexpr,
    q_layout: tl.constexpr,
    k_layout: tl.constexpr,
    mma_layout: tl.constexpr,
    PREFETCH: tl.constexpr,
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
    next_k_ready = current_ready
    if PREFETCH:
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
    tlx.sched_barrier()
    return state, p_dot, next_scores, next_k_ready


@triton.jit
def _pipeline_phase(
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
    publish_k_ready,
    previous_v_ready,
    PHASE: tl.constexpr,
    q_layout: tl.constexpr,
    k_layout: tl.constexpr,
    p_layout: tl.constexpr,
    v_layout: tl.constexpr,
    mma_layout: tl.constexpr,
    qk_scale: tl.constexpr,
    log2_score_bound: tl.constexpr,
):
    previous_v_slot: tl.constexpr = PHASE
    current_k_slot: tl.constexpr = (PHASE + 1) % LDS_STAGES
    k_prefetch_slot: tl.constexpr = (PHASE + 3) % LDS_STAGES
    current_v_ready = _issue_tile(
        v_ptrs,
        tile + 1,
        stride_vn,
        v_buffer,
        current_k_slot,
        32,
    )
    current_ready = _wait_stage_end(
        2,
        [current_k_ready, previous_v_ready],
    )
    state, p_dot, next_scores, next_k_ready = _score_phase(
        state,
        probabilities,
        current_ready,
        q,
        k_ptrs,
        k_buffer,
        stride_kn,
        tile,
        current_k_slot,
        k_prefetch_slot,
        q_layout,
        k_layout,
        mma_layout,
        True,
    )
    _stage_end()
    # The dominating wait above supplies V completion.  Carry the next-K issue
    # token as an additional publication dependency, matching the standalone
    # kernel's cohort hand-off without imposing a scheduler-side ordering rule.
    value_fragments = _load_value_fragments(
        tlx.local_view(v_buffer, previous_v_slot),
        publish_k_ready,
        v_layout,
    )
    _stage_end()

    state, probabilities = _accumulate_with_prefix_barrier(
        state,
        p_dot,
        value_fragments,
        next_scores,
        p_layout,
        v_layout,
        mma_layout,
        qk_scale,
        log2_score_bound,
    )

    return state, probabilities, next_k_ready, current_v_ready


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
):
    q = tlx.require_layout(q, q_layout)
    # Prime K0, V0, and K1.  Every completion used by a local load below comes
    # from an explicit wait_group; LDS aliasing is never used as a dependency.
    k_ready0 = _issue_tile(k_ptrs, 0, stride_kn, k_buffer, 0, 8)
    v_ready0 = _issue_tile(v_ptrs, 0, stride_vn, v_buffer, 0, 32)
    k_ready1 = _issue_tile(k_ptrs, 1, stride_kn, k_buffer, 1, 8)

    wait_k0 = _wait_stage_end(2, [k_ready0, k_ready1, v_ready0])
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
    probabilities = state.prepare(
        scores,
        qk_scale,
        log2_score_bound,
        mma_layout,
    )

    # K2 has its own physical slot; unlike the legacy cluster kernel, no
    # barrier-and-overwrite of K0 is needed here.
    k_ready2 = _issue_tile(k_ptrs, 2, stride_kn, k_buffer, 2, 8)
    # Match the standalone kernel's eight-wave cohort stagger.  The trailing
    # four waves cross one barrier event before the leading cohort.
    trailing_cohort = tlx.warp_id() >= 4
    tlx.sched_barrier()
    if trailing_cohort:
        _stage_end()
        tlx.set_priority(3)
    tlx.sched_barrier()

    # Tile i consumes V[i] and K[i+1], reads K[i+2], and publishes K[i+3] and
    # V[i+2].  Spell out all four ring phases so each physical slot remains
    # static in the generated Wave program.
    # Peel phase zero so the loop enters with real PV accumulator values.  In
    # addition to matching the four-slot ring, this lets Wave keep the MFMA
    # accumulator register groups intact across the loop backedge.
    q = tlx.release_layout(q)
    state, probabilities, k_ready3, previous_v_ready = _pipeline_phase(
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
        k_ready2,
        v_ready0,
        0,
        q_layout,
        k_layout,
        p_layout,
        v_layout,
        mma_layout,
        qk_scale,
        log2_score_bound,
    )
    for first_tile in tl.range(1, tile_count - 3, 4, num_stages=0):
        state, probabilities, k_ready0, previous_v_ready = _pipeline_phase(
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
            k_ready3,
            previous_v_ready,
            1,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            mma_layout,
            qk_scale,
            log2_score_bound,
        )
        state, probabilities, k_ready1, previous_v_ready = _pipeline_phase(
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
            k_ready0,
            previous_v_ready,
            2,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            mma_layout,
            qk_scale,
            log2_score_bound,
        )
        state, probabilities, k_ready2, previous_v_ready = _pipeline_phase(
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
            k_ready1,
            previous_v_ready,
            3,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            mma_layout,
            qk_scale,
            log2_score_bound,
        )
        state, probabilities, k_ready3, previous_v_ready = _pipeline_phase(
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
            k_ready2,
            previous_v_ready,
            0,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            mma_layout,
            qk_scale,
            log2_score_bound,
        )
    # Drain the final three output tiles without out-of-range DMA requests.
    tile_nm3 = tile_count - 3
    tile_nm2 = tile_count - 2
    tile_nm1 = tile_count - 1

    v_ready_nm2 = _issue_tile(
        v_ptrs,
        tile_nm2,
        stride_vn,
        v_buffer,
        tile_nm2 % LDS_STAGES,
        32,
    )
    ready_nm3 = _wait_stage_end(2, [k_ready2, previous_v_ready])
    state, p_dot, next_scores, _ = _score_phase(
        state,
        probabilities,
        ready_nm3,
        q,
        k_ptrs,
        k_buffer,
        stride_kn,
        tile_nm3,
        tile_nm2 % LDS_STAGES,
        0,
        q_layout,
        k_layout,
        mma_layout,
        False,
    )
    _stage_end()
    value_fragments = _load_value_fragments(
        tlx.local_view(v_buffer, tile_nm3 % LDS_STAGES),
        k_ready3,
        v_layout,
    )
    _stage_end()
    state, probabilities = _accumulate_with_prefix_barrier(
        state,
        p_dot,
        value_fragments,
        next_scores,
        p_layout,
        v_layout,
        mma_layout,
        qk_scale,
        log2_score_bound,
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
    state, p_dot, next_scores, _ = _score_phase(
        state,
        probabilities,
        ready_nm2,
        q,
        k_ptrs,
        k_buffer,
        stride_kn,
        tile_nm2,
        tile_nm1 % LDS_STAGES,
        0,
        q_layout,
        k_layout,
        mma_layout,
        False,
    )
    _stage_end()
    value_fragments = _load_value_fragments(
        tlx.local_view(v_buffer, tile_nm2 % LDS_STAGES),
        k_ready0,
        v_layout,
    )
    _stage_end()
    state, probabilities = _accumulate_with_prefix_barrier(
        state,
        p_dot,
        value_fragments,
        next_scores,
        p_layout,
        v_layout,
        mma_layout,
        qk_scale,
        log2_score_bound,
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
    leading_cohort = tlx.warp_id() < 4
    tlx.sched_barrier()
    if leading_cohort:
        _stage_end()
    tlx.set_priority(0)
    tlx.sched_barrier()
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
    state = BoundedSoftmaxState.create()
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
    )

    _store_output_fragment(
        Out,
        o_base,
        pid_m,
        stride_m,
        state.row_sum,
        state.acc0,
        0,
        mma_layout,
    )
    _store_output_fragment(
        Out,
        o_base,
        pid_m,
        stride_m,
        state.row_sum,
        state.acc1,
        1,
        mma_layout,
    )
    _store_output_fragment(
        Out,
        o_base,
        pid_m,
        stride_m,
        state.row_sum,
        state.acc2,
        2,
        mma_layout,
    )
    _store_output_fragment(
        Out,
        o_base,
        pid_m,
        stride_m,
        state.row_sum,
        state.acc3,
        3,
        mma_layout,
    )


def attention(
    q,
    k,
    v,
    sm_scale=None,
    causal=False,
    *,
    qk_max_abs=1.0,
    out=None,
    warmup=False,
    **kwargs,
):
    """Run the separate eight-wave bounded-input attention kernel."""
    assert not causal, "amd_fa_wave only implements non-causal attention"
    assert q.dtype == torch.bfloat16, "amd_fa_wave requires BF16 inputs"
    assert q.ndim == 4
    batch, heads, sequence, head_dim = q.shape
    assert k.shape == q.shape and v.shape == q.shape
    assert head_dim == 128
    assert sequence % 256 == 0
    assert sequence // 64 >= 4
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert math.isfinite(qk_max_abs) and qk_max_abs > 0

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    log2_score_bound = (math.log2(math.e) * head_dim * float(sm_scale) * qk_max_abs * qk_max_abs)
    output = torch.empty_like(q) if out is None else out
    assert output.shape == q.shape and output.dtype == q.dtype
    assert output.is_contiguous()
    grid = (sequence // 256, batch * heads, 1)
    launch_options = {
        "N_CTX": sequence,
        "BATCH": batch,
        "HEADS": heads,
        "TOTAL_HEADS": batch * heads,
        "SM_SCALE": float(sm_scale),
        "LOG2_SCORE_BOUND": log2_score_bound,
        "num_warps": 8,
        "waves_per_eu": 2,
        **kwargs,
    }
    if triton.runtime.driver.active.get_current_target().backend == "tlx_wave":
        launch_options.setdefault("tlx_wave_enable_multi_wave_specialize", True)

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
