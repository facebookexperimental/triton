"""Blackwell MXFP8 Flash Attention forward and backward implementation."""

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.cuda.inline_ptx_lib import _fma_f32x2, _mul_f32x2, _sub_f32x2
from triton.language.extra.subtile_ops import _split_n_2D
from triton.language.extra.tlx.mxfp8_utils import (
    _cvt_e4m3x4_f32,
    _fused_amax_to_e8m0,
    _to_mxfp8_32x32_block,
    _to_mxfp8_block_with_block_amax,
)
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

try:
    from torchao.prototype.mx_formats.mx_tensor import MXTensor, ScaleCalculationMode
except ImportError:
    MXTensor = None
    ScaleCalculationMode = None

_HAS_MXFP8_QUANTIZATION = MXTensor is not None and ScaleCalculationMode is not None


def _mxf8_host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    nargs["desc_m"].block_shape = [BLOCK_M_SPLIT]
    VEC_SIZE = 32
    REP_M = math.ceil(BLOCK_M_SPLIT / 128)
    REP_N = math.ceil(math.ceil(BLOCK_N / VEC_SIZE) / 4)
    REP_HEAD = math.ceil(HEAD_DIM / 128)
    nargs["desc_q_scale"].block_shape = [1, REP_M, REP_HEAD, 2, 256]
    nargs["desc_k_scale"].block_shape = [1, REP_N, REP_HEAD, 2, 256]
    # V_scale has scales along N dimension (for P @ V), so dimensions are swapped
    nargs["desc_v_scale"].block_shape = [1, REP_HEAD, REP_N, 2, 256]


# TODO: Tune. These are just copied
mxfp8_configs = [
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 4,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_Q_SCALE_TMEM_BUFFERS": 1,
            "NUM_KV_SCALE_TMEM_BUFFERS": 2,
            "GROUP_SIZE_N": 1,
            "RESCALE_OPT": True,
            "UNROLL_KV": False,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=_mxf8_host_descriptor_pre_hook,
    ),
]


def prune_configs_by_hdim_mxfp8(configs, _named_args, **_kwargs):
    return configs


@triton.jit
def _reduce_or(x, y):
    return x | y


@triton.jit
def _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE: tl.constexpr):
    if STAGE == 1:
        # First part of STAGE == 3 in _get_fused_loop_bounds
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # Second part of STAGE == 3 in _get_fused_loop_bounds
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        tl.static_assert(STAGE == 3)
        # Maps to STAGE=1 in _get_fused_loop_bounds
        lo, hi = 0, N_CTX
    return lo, hi


@triton.jit
def _get_fused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE: tl.constexpr):
    if STAGE == 1:
        return 0, N_CTX
    else:
        tl.static_assert(STAGE == 3)
        return 0, (start_m + 1) * BLOCK_M


@triton.jit
def _compute_offsets(
    tile_idx,
    H,
    num_pid_n,
    num_pid_in_group,
    N_CTX,
    BLOCK_M: tl.constexpr,
    STAGE: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
):
    group_id = tile_idx // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    start_m = (tile_idx % num_pid_in_group) // group_size_n
    off_hz = first_pid_n + (tile_idx % group_size_n)
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    lo, hi = _get_fused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)
    kv_offset_y = offset_y + lo
    return start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y


@triton.jit
def _mask_scalar(qk, col_limit_right, s, i):
    # Forward orientation: keep columns c < col_limit_right, mask c >= it to -inf.
    col_lim_right_s = col_limit_right - s
    col_lim_right_cur = max(col_lim_right_s, 0)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return tl.where(mask_i_bit, qk, -float("inf"))


@triton.jit
def _mask_scalar_transposed(qk, col_limit_left, s, i):
    # Transposed orientation (backward qkT[N, M]): keep columns c >= col_limit_left
    # (i.e. keep keys where m >= n), mask c < it to -inf. Mirror of _mask_scalar:
    # same per-16-block bitmask, but the kept side is flipped.
    col_lim_left_cur = max(col_limit_left - s, 0)
    mask = -1 << col_lim_left_cur
    keep_i_bit = (mask & (1 << i)) != 0
    return tl.where(keep_i_bit, qk, -float("inf"))


@triton.jit
def _apply_causal_mask(qk, col_limit, BLOCK_N: tl.constexpr, TRANSPOSED: tl.constexpr):
    # Apply causal mask via a bitmask calculated for each block of 16 elements.
    # This allows the efficient R2P (register to predicate) instruction to be used at the SASS level.
    # Credit to Tri Dao,
    # https://github.com/Dao-AILab/flash-attention/commit/bac1001e4f6caa09d70537495d6746a685a2fa78
    #
    # NOTE: We use map_elementwise here in order to generate an interleaved sequence of instructions
    # that processes one element of qk at a time. This improves ptxas's resulting SASS.
    #
    # Forward (TRANSPOSED=False): qk is [M, N], col_limit is the per-query right
    # bound on keys (offs_m - start_n + 1). Backward (TRANSPOSED=True): qk is the
    # transposed qkT [N, M], col_limit is the per-key left bound on queries
    # (offs_n - curr_m); masks columns m < n so only m >= n survive.
    offs = tl.arange(0, BLOCK_N)[None, :]
    s = offs & ~0xF
    i = offs & 0xF
    if TRANSPOSED:
        return tl.map_elementwise(_mask_scalar_transposed, qk, col_limit, s, i)
    return tl.map_elementwise(_mask_scalar, qk, col_limit, s, i)


@triton.jit
def _expcast_e4m3x4(x, addend, mult):
    """E4M3 code = RNE(x * mult + addend - 1024), clamped below at 0, packed 4 per register (no MUFU, no F2FP.E4M3)."""
    # code + 1024 lies in [1024, 2048) where f16 ulp is 1: the f16 cvt rounds to the integer code in the
    # low byte, and max with 1024 maps negative and -inf to code 0. max.f16x2 also maps NaN to code 0, so a
    # NaN in S yields a finite but wrong output row where the exp2 path yields a NaN row.
    # The clamp floor is code 0 (exact 0.0), not the min subnormal, so masked keys get exactly zero weight. It
    # flushes below ~2^-6.9 * 2^e (baseline: ~2^-10 * 2^e), >= 15.7 binades under the block max.
    # pack=4 returns 4 E4M3 bytes in one b32, as in tlx.mxfp8_utils._cvt_e4m3x4_f32.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 xc_a01, xc_a23, xc_c01, xc_c23, xc_m2, xc_z01, xc_z23;
            .reg .f32 xc_z0, xc_z1, xc_z2, xc_z3;
            .reg .b32 xc_h01, xc_h23, xc_lo;
            mov.b64 xc_a01, {$1, $2};
            mov.b64 xc_a23, {$3, $4};
            mov.b64 xc_c01, {$5, $6};
            mov.b64 xc_c23, {$7, $8};
            mov.b64 xc_m2, {$9, $9};
            fma.rn.f32x2 xc_z01, xc_a01, xc_m2, xc_c01;
            fma.rn.f32x2 xc_z23, xc_a23, xc_m2, xc_c23;
            mov.b64 {xc_z0, xc_z1}, xc_z01;
            mov.b64 {xc_z2, xc_z3}, xc_z23;
            cvt.rn.f16x2.f32 xc_h01, xc_z1, xc_z0;
            cvt.rn.f16x2.f32 xc_h23, xc_z3, xc_z2;
            mov.b32 xc_lo, 0x64006400;
            max.f16x2 xc_h01, xc_h01, xc_lo;
            max.f16x2 xc_h23, xc_h23, xc_lo;
            prmt.b32 $0, xc_h01, xc_h23, 0x6420;
        }
        """,
        "=r,f,f,f,f,f,f,f,f,f,f,f,f",
        [x, addend, mult],
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _softmax_inner_loop(
    qk_empties,
    qk_fulls,
    qk_tiles,
    p_empties,
    p_fulls,
    p_tiles,
    p_scale_tiles,
    alpha_empties,
    alpha_fulls,
    alpha_tiles,
    cid,
    accum_cnt_qk,
    qk_scale,
    offs_m,
    m_i,
    l_i,
    start_m,
    N_CTX,
    out_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    STAGE: tl.constexpr,
    SHARE_SCALE_BUFFERS: tl.constexpr = False,
    RESCALE_OPT: tl.constexpr = False,
    P_EXPCAST: tl.constexpr = False,
):
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2
    NUM_BLOCKS: tl.constexpr = BLOCK_N // VEC_SIZE
    if P_EXPCAST:
        tl.static_assert(out_dtype == tl.float8e4nv)

    lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)

    for start_n in tl.range(lo, hi, BLOCK_N):
        _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
        # Hoisted from its original position (between alpha compute and alpha
        # store) to here. SYNCS.PHASECHK.TRYWAIT acts as a ptxas scheduling
        # fence — in the original position it splits the softmax compute+store
        # into two separately-scheduled regions (~680 SASS instructions apart).
        # Moving both SYNCS waits adjacent here keeps the entire softmax body
        # as one scheduling region, letting ptxas pipeline more aggressively.
        # ncu confirms: -1.7% cycles (7.92M vs 8.06M), same instruction count.
        tlx.barrier_wait(tlx.local_view(alpha_empties, cid), qk_phase ^ 1)
        tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
        qk = tlx.local_load(tlx.local_view(qk_tiles, cid))
        if SHARE_SCALE_BUFFERS:
            NAMED_BAR_QK_EMPTY: tl.constexpr = 9
            NUM_THREADS_QK_EMPTY: tl.constexpr = 160
            tlx.named_barrier_arrive(NAMED_BAR_QK_EMPTY + cid, NUM_THREADS_QK_EMPTY)
        else:
            tlx.barrier_arrive(tlx.local_view(qk_empties, cid))

        if STAGE == 2:
            col_limit_right = (offs_m - start_n + 1)[:, None]
            qk = _apply_causal_mask(qk, col_limit_right, BLOCK_N, TRANSPOSED=False)

        qk_reshaped = tl.reshape(qk, [BLOCK_M_SPLIT, NUM_BLOCKS, VEC_SIZE])
        block_maxes = tl.max(qk_reshaped, 2)
        row_max = tl.max(block_maxes, 1)

        if RESCALE_OPT:
            m_ij = tl.maximum(m_i, row_max)
            alpha_ = (m_i - m_ij) * qk_scale
            alpha = tl.math.exp2(alpha_)
            rescale_mask = alpha_ >= -8.0
            alpha = tl.where(rescale_mask, 1.0, alpha)
            m_ij = tl.where(rescale_mask, m_i, m_ij)
        else:
            m_ij = tl.maximum(m_i, row_max * qk_scale)
            alpha = tl.math.exp2(m_i - m_ij)

        tlx.local_store(tlx.local_view(alpha_tiles, cid), alpha[:, None])
        tlx.barrier_arrive(tlx.local_view(alpha_fulls, cid))

        if RESCALE_OPT:
            m_scaled = m_ij * qk_scale
        else:
            m_scaled = m_ij
        if P_EXPCAST:
            # ExpCast-FP8 (arXiv 2609.15810) with the MX block scale kept: the E4M3 byte read as an integer is
            # ~8*log2(v) + 56, so code = RNE(8*(u - e) + 56 + beta) with u = log2(p) and e the block's E8M0
            # exponent. beta = -0.3443 centres the error of 1 + f vs 2^f; per-element ratio to exact p is in
            # [0.929, 1.075]. e = ceil(u_max - log2(448)) matches the baseline RCEIL scale and keeps code <= 126.
            # NaN in S is encoded as P = 0. NaN detection is left to model-level guards; in training the bwd
            # still yields NaN grads since it recomputes P with exp2. Debug builds (TRITON_DEBUG=1) assert here.
            tl.device_assert(qk == qk, "NaN in S: ExpCast encodes it as P = 0")
            block_u = block_maxes * qk_scale - m_scaled[:, None]
            e_blk = tl.minimum(tl.maximum(tl.math.ceil(block_u - 8.807354922057604), -127.0), 127.0)
            addend = (1079.6557 - 8.0 * m_scaled[:, None]) - 8.0 * e_blk
            p_fp8_3d = _expcast_e4m3x4(qk_reshaped, tl.reshape(addend, [BLOCK_M_SPLIT, NUM_BLOCKS, 1]), qk_scale * 8.0)
            e_int = e_blk.to(tl.int32)
            p_scale = (e_int + 127).to(tl.uint8)
            # l sums the decoded P, keeping numerator and denominator consistent. e_blk = -127 decodes to 0.0,
            # not 2^-127; such blocks are fully masked (all codes 0) or > ~118 binades under the row max.
            blk_sum = tl.sum(p_fp8_3d.to(tl.float32), 2)
            l_ij = tl.sum(blk_sum * ((e_int + 127) << 23).to(tl.float32, bitcast=True), 1)

            tlx.barrier_wait(tlx.local_view(p_empties, cid), qk_phase ^ 1)
            tlx.local_store(tlx.local_view(p_tiles, cid), tl.reshape(p_fp8_3d, [BLOCK_M_SPLIT, BLOCK_N]))
            tlx.local_store(tlx.local_view(p_scale_tiles, cid), p_scale)
            tlx.barrier_arrive(tlx.local_view(p_fulls, cid))
        else:
            qk = _fma_f32x2(qk, qk_scale, -m_scaled[:, None])
            p_i = tl.math.exp2(qk)

            # Derive block amax from pre-computed block maxes via monotonicity
            # of exp2: max(exp2(x)) == exp2(max(x)), avoiding 128 max(abs())
            # ops per row in the MXFP8 conversion.
            block_amax = tl.math.exp2(block_maxes * qk_scale - m_scaled[:, None])

            # Compute row sum before p_empties wait: if MMA is still doing the
            # previous PV GEMM, the wait stalls — fill that time with the sum.
            l_ij = tl.sum(p_i, 1)

            tlx.barrier_wait(tlx.local_view(p_empties, cid), qk_phase ^ 1)
            p_fp8, p_scale = _to_mxfp8_block_with_block_amax(
                p_i,
                block_amax,
                VEC_SIZE,
                out_dtype,
            )
            tlx.local_store(tlx.local_view(p_tiles, cid), p_fp8)
            tlx.local_store(tlx.local_view(p_scale_tiles, cid), p_scale)
            tlx.barrier_arrive(tlx.local_view(p_fulls, cid))

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        accum_cnt_qk += 1

    return m_i, l_i, accum_cnt_qk


@triton.jit
def _softmax_task_tile(
    qk_empties,
    qk_fulls,
    qk_tiles,
    p_empties,
    p_fulls,
    p_tiles,
    p_scale_tiles,
    alpha_empties,
    alpha_fulls,
    alpha_tiles,
    l_empties,
    l_fulls,
    l_tiles,
    m_tiles,
    tile_id,
    tile_count,
    accum_cnt_qk,
    H,
    num_pid_n,
    num_pid_in_group,
    N_CTX,
    sm_scale,
    p_dtype: tl.constexpr,
    cid,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    STAGE: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    SHARE_SCALE_BUFFERS: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
    P_EXPCAST: tl.constexpr = False,
):
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2
    start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
        tile_id,
        H,
        num_pid_n,
        num_pid_in_group,
        N_CTX,
        BLOCK_M,
        STAGE,
        GROUP_SIZE_N,
    )
    m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    offs_m = (start_m * BLOCK_M) + ((cid * BLOCK_M_SPLIT) + tl.arange(0, BLOCK_M_SPLIT))
    if STAGE & 1:
        m_i, l_i, accum_cnt_qk = _softmax_inner_loop(
            qk_empties,
            qk_fulls,
            qk_tiles,
            p_empties,
            p_fulls,
            p_tiles,
            p_scale_tiles,
            alpha_empties,
            alpha_fulls,
            alpha_tiles,
            cid,
            accum_cnt_qk,
            qk_scale,
            offs_m,
            m_i,
            l_i,
            start_m,
            N_CTX,
            p_dtype,
            BLOCK_M,
            BLOCK_N,
            VEC_SIZE,
            STAGE=4 - STAGE,
            SHARE_SCALE_BUFFERS=SHARE_SCALE_BUFFERS,
            RESCALE_OPT=RESCALE_OPT,
            P_EXPCAST=P_EXPCAST,
        )

    if STAGE & 2:
        m_i, l_i, accum_cnt_qk = _softmax_inner_loop(
            qk_empties,
            qk_fulls,
            qk_tiles,
            p_empties,
            p_fulls,
            p_tiles,
            p_scale_tiles,
            alpha_empties,
            alpha_fulls,
            alpha_tiles,
            cid,
            accum_cnt_qk,
            qk_scale,
            offs_m,
            m_i,
            l_i,
            start_m,
            N_CTX,
            p_dtype,
            BLOCK_M,
            BLOCK_N,
            VEC_SIZE,
            STAGE=2,
            SHARE_SCALE_BUFFERS=SHARE_SCALE_BUFFERS,
            RESCALE_OPT=RESCALE_OPT,
            P_EXPCAST=P_EXPCAST,
        )

    _, phase = get_bufidx_phase(tile_count, 1)
    if not SHARE_SCALE_BUFFERS:
        # Wait for L to be empty if it has its own buffer.
        tlx.barrier_wait(l_empties[cid], phase ^ 1)
    tlx.local_store(l_tiles[cid], l_i[:, None])
    tlx.local_store(m_tiles[cid], m_i[:, None])
    tlx.barrier_arrive(l_fulls[cid])
    return accum_cnt_qk


@triton.jit
def _correction_and_final_normalization_task_tile(
    alpha_fulls,
    alpha_empties,
    alpha_tiles,
    acc_fulls,
    acc_empties,
    acc_tiles,
    l_fulls,
    l_empties,
    l_tiles,
    m_tiles,
    o_empties,
    o_fulls,
    o_tiles,
    m_out_tiles,
    desc_m,
    desc_o,
    tile_id,
    tile_count,
    accum_cnt,
    H,
    num_pid_n,
    num_pid_in_group,
    N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_ACC_SLICES: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
):
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2
    start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
        tile_id,
        H,
        num_pid_n,
        num_pid_in_group,
        N_CTX,
        BLOCK_M,
        STAGE,
        GROUP_SIZE_N,
    )
    # The first alpha handshake has no accumulator multiply because the first
    # PV MMA overwrites the accumulator with use_acc=False.
    _, phase = get_bufidx_phase(accum_cnt, 1)
    for cid in tl.static_range(0, NUM_MMA_GROUPS):
        tlx.barrier_wait(alpha_fulls[cid], phase)
        tlx.barrier_arrive(alpha_empties[cid])
        tlx.barrier_arrive(acc_fulls[cid])
    accum_cnt += 1

    for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
        _, phase = get_bufidx_phase(accum_cnt, 1)
        for cid in tl.static_range(0, NUM_MMA_GROUPS):
            # -- update output accumulator --
            tlx.barrier_wait(alpha_fulls[cid], phase)
            alpha_1 = tlx.local_load(alpha_tiles[cid])
            tlx.barrier_arrive(alpha_empties[cid])
            if RESCALE_OPT:
                pred = alpha_1 < 1.0
                ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
                should_rescale = ballot_result != 0
                should_rescale_red = tl.reduce(should_rescale, axis=0, combine_fn=_reduce_or)
                should_rescale_scalar = tl.reshape(should_rescale_red, ())
                if should_rescale_scalar:
                    for slice_id in tl.static_range(0, NUM_ACC_SLICES):
                        subslice = tlx.subslice(
                            acc_tiles[cid],
                            HEAD_DIM * slice_id // NUM_ACC_SLICES,
                            HEAD_DIM // NUM_ACC_SLICES,
                        )
                        acc = tlx.local_load(subslice)
                        acc = _mul_f32x2(acc, alpha_1)
                        tlx.local_store(subslice, acc)
            else:
                for slice_id in tl.static_range(0, NUM_ACC_SLICES):
                    subslice = tlx.subslice(
                        acc_tiles[cid],
                        HEAD_DIM * slice_id // NUM_ACC_SLICES,
                        HEAD_DIM // NUM_ACC_SLICES,
                    )
                    acc = tlx.local_load(subslice)
                    acc = _mul_f32x2(acc, alpha_1)
                    tlx.local_store(subslice, acc)
            tlx.barrier_arrive(acc_fulls[cid])
        accum_cnt += 1

    _, phase = get_bufidx_phase(tile_count, 1)
    for cid in tl.static_range(0, NUM_MMA_GROUPS):
        # epilogue — critical path first (produce o_tiles),
        # then non-critical M (LSE) store to GMEM
        tlx.barrier_wait(l_fulls[cid], phase)
        l = tlx.local_load(l_tiles[cid])
        scale = 1 / l

        tlx.barrier_wait(acc_empties[cid], phase)
        tlx.barrier_wait(o_empties[cid], phase ^ 1)
        for slice_id in tl.static_range(0, NUM_ACC_SLICES):
            subslice = tlx.subslice(
                acc_tiles[cid],
                HEAD_DIM * slice_id // NUM_ACC_SLICES,
                HEAD_DIM // NUM_ACC_SLICES,
            )
            acc = tlx.local_load(subslice)
            acc = _mul_f32x2(acc, scale)
            acc = acc.to(tlx.dtype_of(desc_o))
            subslice_o = tlx.local_slice(
                o_tiles[cid],
                [0, HEAD_DIM * slice_id // NUM_ACC_SLICES],
                [BLOCK_M_SPLIT, HEAD_DIM // NUM_ACC_SLICES],
            )
            tlx.local_store(subslice_o, acc)
        tlx.barrier_arrive(o_fulls[cid])
        l = tlx.local_load(l_tiles[cid])
        m = tlx.local_load(m_tiles[cid])
        tlx.barrier_arrive(l_empties[cid])

        if RESCALE_OPT:
            m = m * sm_scale * 1.44269504
        m += tl.math.log2(l)
        m_offset = off_hz * N_CTX + start_m * BLOCK_M + cid * BLOCK_M_SPLIT
        tlx.async_descriptor_store_wait(0)
        tlx.local_store(m_out_tiles[cid], tl.reshape(m, [BLOCK_M_SPLIT]))
        tlx.async_descriptor_store(desc_m, m_out_tiles[cid], [m_offset.to(tl.int32)])

    return accum_cnt


@triton.autotune(
    configs=mxfp8_configs,
    key=["N_CTX", "HEAD_DIM", "STAGE"],
    prune_configs_by={"early_config_prune": prune_configs_by_hdim_mxfp8},
)
@triton.jit
def _attn_fwd_mxf8_ws(sm_scale, desc_m,  #
                      Z, H, desc_q, desc_k, desc_v, desc_o, desc_q_scale, desc_k_scale, desc_v_scale, N_CTX,  #
                      HEAD_DIM: tl.constexpr,  #
                      BLOCK_M: tl.constexpr,  #
                      BLOCK_N: tl.constexpr,  #
                      STAGE: tl.constexpr,  #
                      NUM_BUFFERS_Q: tl.constexpr,  #
                      NUM_BUFFERS_KV: tl.constexpr,  #
                      NUM_BUFFERS_QK: tl.constexpr,  #
                      NUM_MMA_GROUPS: tl.constexpr,  #
                      NUM_Q_SCALE_TMEM_BUFFERS: tl.constexpr,  #
                      NUM_KV_SCALE_TMEM_BUFFERS: tl.constexpr,  #
                      GROUP_SIZE_N: tl.constexpr,  #
                      RESCALE_OPT: tl.constexpr,  #
                      UNROLL_KV: tl.constexpr = False,  #
                      P_EXPCAST: tl.constexpr = False,  #
                      ):
    """
    This kernel is adapted from the Blackwell FA kernel for MXFP8.

    P is converted to FP8 online with per-block E8M0 scales and stored in
    TMEM alongside its scales, matching the BF16 kernel's pattern of keeping
    P in TMEM for the PV scaled dot.
    """
    tl.static_assert(NUM_MMA_GROUPS == 2)
    tl.static_assert(NUM_BUFFERS_QK == 1)
    tl.static_assert(NUM_BUFFERS_Q == 1)
    tl.static_assert(not UNROLL_KV)

    # Define if we need to do buffer sharing for the scales.
    SHARE_SCALE_BUFFERS: tl.constexpr = (HEAD_DIM == 128) and (BLOCK_N == 128)

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2
    NUM_ACC_SLICES: tl.constexpr = 4

    # Compute p_dtype from V descriptor
    p_dtype = tlx.dtype_of(desc_v)

    Q_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_q))
    K_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_k))
    V_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_v))
    P_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(p_dtype)

    # Scale tile dimensions for 5D TMA (only used when USE_SCALE_MMA is True)
    # Using ceiling division for block sizes that may not fully use the hardware
    REP_M: tl.constexpr = triton.cdiv(BLOCK_M_SPLIT, 128)
    REP_N: tl.constexpr = triton.cdiv(BLOCK_N, 128)
    VEC_SIZE: tl.constexpr = 32
    REP_HEAD: tl.constexpr = triton.cdiv(triton.cdiv(HEAD_DIM, VEC_SIZE), 4)

    # Compute bytes per element for each tensor type
    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))
    qk_dtype = tl.float32

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = Z * H
    num_pid_in_group = num_pid_m * GROUP_SIZE_N

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    o_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_o), NUM_MMA_GROUPS)
    m_out_tiles = tlx.local_alloc((BLOCK_M_SPLIT, ), tl.float32, NUM_MMA_GROUPS)

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    o_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    o_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    # 5D scale buffers: [1, REP_M/N, REP_HEAD, 2, 256]
    # For FP8, scales are stored in TMEM
    # Single allocation with NUM_MMA_GROUPS * NUM_BUFFERS_Q buffers for q_scale
    q_scale_tiles = tlx.local_alloc((1, REP_M, REP_HEAD, 2, 256), tl.uint8, NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_scale_tiles = tlx.local_alloc((1, REP_N, REP_HEAD, 2, 256), tl.uint8, NUM_BUFFERS_KV)

    # Calculate scale bytes for barrier expect
    Q_SCALE_BYTES: tl.constexpr = REP_M * REP_HEAD * 2 * 256
    K_SCALE_BYTES: tl.constexpr = REP_N * REP_HEAD * 2 * 256
    V_SCALE_BYTES: tl.constexpr = REP_N * REP_HEAD * 2 * 256

    # TMEM scale buffers for explicit SMEM->TMEM transfer (2D shape for tcgen05 scales layout)
    Q_SCALE_TMEM_COLS: tl.constexpr = Q_SCALE_BYTES // BLOCK_M_SPLIT
    K_SCALE_TMEM_COLS: tl.constexpr = K_SCALE_BYTES // BLOCK_N
    V_SCALE_TMEM_ROWS: tl.constexpr = REP_HEAD * 128
    V_SCALE_TMEM_COLS: tl.constexpr = V_SCALE_BYTES // V_SCALE_TMEM_ROWS
    if SHARE_SCALE_BUFFERS:
        # We don't have enough TMEM space to hold the scale transfer. We need to have a creative
        # reuse strategy that so QK[0] can share space with Q_SCALES
        tl.static_assert(NUM_Q_SCALE_TMEM_BUFFERS == 1)
        tl.static_assert(NUM_KV_SCALE_TMEM_BUFFERS == 2)
        # Define the shared buffer.
        qk_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
        qk_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_N),
            qk_dtype,
            NUM_MMA_GROUPS,
            tlx.storage_kind.tmem,
            reuse=qk_storage_alias,
        )
        l_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, 1),
            tl.float32,
            NUM_MMA_GROUPS * NUM_BUFFERS_QK,
            tlx.storage_kind.tmem,
            reuse=qk_storage_alias,
        )
        m_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, 1),
            tl.float32,
            NUM_MMA_GROUPS * NUM_BUFFERS_QK,
            tlx.storage_kind.tmem,
            reuse=qk_storage_alias,
        )
        q_scale_tmem = tlx.local_alloc(
            (BLOCK_M_SPLIT, Q_SCALE_TMEM_COLS),
            tl.uint8,
            2 * NUM_Q_SCALE_TMEM_BUFFERS,
            tlx.storage_kind.tmem,
            reuse=qk_storage_alias,
        )
        k_scale_tmem = tlx.local_alloc(
            (BLOCK_N, K_SCALE_TMEM_COLS),
            tl.uint8,
            NUM_KV_SCALE_TMEM_BUFFERS,
            tlx.storage_kind.tmem,
            reuse=qk_storage_alias,
        )
        v_scale_tmem = tlx.local_alloc(
            (V_SCALE_TMEM_ROWS, V_SCALE_TMEM_COLS),
            tl.uint8,
            NUM_KV_SCALE_TMEM_BUFFERS,
            tlx.storage_kind.tmem,
            reuse=qk_storage_alias,
        )
        p_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_N),
            tlx.dtype_of(desc_v),
            NUM_MMA_GROUPS,
            tlx.storage_kind.tmem,
            reuse=qk_storage_alias,
        )
        p_scale_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_N // VEC_SIZE),
            tl.uint8,
            NUM_MMA_GROUPS,
            tlx.storage_kind.tmem,
            reuse=qk_storage_alias,
        )
        # Define the reuse strategy.
        # QK and P have sequential lifetimes (QK consumed by softmax before P produced),
        # so they share the same TMEM region. P in FP8 (32 cols) fits within QK's FP32 space (128 cols).
        # QK[0] : |                              BLK_M/2 * BLOCK_N * fp32                                       |
        # L[0]: |BLK_M/2*1*fp32|
        # M[0]:                    |BLK_M/2*1*fp32|
        # Q_SCALES[1]:                                           |512*uint8|
        # K_SCALES[1]:                                                     |512*uint8|
        # V_SCALES[0]:                                                               |512*uint8|
        # P[0]:                                                                      |BLK_M/2*BLK_N*fp8|
        # P_SCALES[0]:                                                                         |BLK_M/2*4*uint8|
        qk_storage_alias.set_buffer_overlap(
            tlx.reuse_group(
                qk_tiles,
                tlx.reuse_group(
                    l_tiles,
                    m_tiles,
                    q_scale_tmem,
                    v_scale_tmem,
                    k_scale_tmem,
                    p_tiles,
                    p_scale_tiles,
                    group_type=tlx.reuse_group_type.distinct,
                ),
                group_type=tlx.reuse_group_type.shared,
            ))

    else:
        # We have enough TMEM space to isolate every buffer.
        qk_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_N), qk_dtype, NUM_MMA_GROUPS, tlx.storage_kind.tmem)
        l_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, 1),
            tl.float32,
            NUM_MMA_GROUPS * NUM_BUFFERS_QK,
            tlx.storage_kind.tmem,
        )
        m_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, 1),
            tl.float32,
            NUM_MMA_GROUPS * NUM_BUFFERS_QK,
            tlx.storage_kind.tmem,
        )
        q_scale_tmem = tlx.local_alloc(
            (BLOCK_M_SPLIT, Q_SCALE_TMEM_COLS),
            tl.uint8,
            2 * NUM_Q_SCALE_TMEM_BUFFERS,
            tlx.storage_kind.tmem,
        )
        k_scale_tmem = tlx.local_alloc(
            (BLOCK_N, K_SCALE_TMEM_COLS),
            tl.uint8,
            NUM_KV_SCALE_TMEM_BUFFERS,
            tlx.storage_kind.tmem,
        )
        v_scale_tmem = tlx.local_alloc(
            (V_SCALE_TMEM_ROWS, V_SCALE_TMEM_COLS),
            tl.uint8,
            NUM_KV_SCALE_TMEM_BUFFERS,
            tlx.storage_kind.tmem,
        )
        p_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_N),
            tlx.dtype_of(desc_v),
            NUM_MMA_GROUPS,
            tlx.storage_kind.tmem,
        )
        p_scale_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_N // VEC_SIZE),
            tl.uint8,
            NUM_MMA_GROUPS,
            tlx.storage_kind.tmem,
        )

    alpha_tiles = tlx.local_alloc((BLOCK_M_SPLIT, 1), tl.float32, NUM_MMA_GROUPS)
    acc_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem)

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    p_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    alpha_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4)
    alpha_empties = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4)
    l_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    l_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    clc_context = tlx.clc_create_context(num_consumers=6)

    with tlx.async_tasks():
        with tlx.async_task("default", registers=64):
            accum_cnt = 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_producer = 1
            clc_phase_consumer = 0
            while tile_id != -1:
                tlx.clc_producer(clc_context, clc_phase_producer)
                clc_phase_producer ^= 1
                accum_cnt = _correction_and_final_normalization_task_tile(
                    alpha_fulls,
                    alpha_empties,
                    alpha_tiles,
                    acc_fulls,
                    acc_empties,
                    acc_tiles,
                    l_fulls,
                    l_empties,
                    l_tiles,
                    m_tiles,
                    o_empties,
                    o_fulls,
                    o_tiles,
                    m_out_tiles,
                    desc_m,
                    desc_o,
                    tile_id,
                    tile_count,
                    accum_cnt,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    sm_scale,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    HEAD_DIM=HEAD_DIM,
                    STAGE=STAGE,
                    GROUP_SIZE_N=GROUP_SIZE_N,
                    NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                    NUM_ACC_SLICES=NUM_ACC_SLICES,
                    RESCALE_OPT=RESCALE_OPT,
                )
                tile_count += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1
            tlx.async_descriptor_store_wait(0)

        with tlx.async_task(num_warps=4, registers=176, replicate=NUM_MMA_GROUPS):
            accum_cnt_qk = 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            cid = tlx.async_task_replica_id()
            while tile_id != -1:
                accum_cnt_qk = _softmax_task_tile(
                    qk_empties,
                    qk_fulls,
                    qk_tiles,
                    p_empties,
                    p_fulls,
                    p_tiles,
                    p_scale_tiles,
                    alpha_empties,
                    alpha_fulls,
                    alpha_tiles,
                    l_empties,
                    l_fulls,
                    l_tiles,
                    m_tiles,
                    tile_id,
                    tile_count,
                    accum_cnt_qk,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    sm_scale,
                    p_dtype,
                    cid,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    HEAD_DIM=HEAD_DIM,
                    VEC_SIZE=VEC_SIZE,
                    STAGE=STAGE,
                    GROUP_SIZE_N=GROUP_SIZE_N,
                    SHARE_SCALE_BUFFERS=SHARE_SCALE_BUFFERS,
                    RESCALE_OPT=RESCALE_OPT,
                    P_EXPCAST=P_EXPCAST,
                )
                tile_count += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # mma group
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            accum_cnt_qk = 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0

            while tile_id != -1:
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )

                q_bufIdx, q_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_Q)
                _, l_phase = get_bufidx_phase(tile_count, 1)
                if SHARE_SCALE_BUFFERS:
                    # With 2 buffers we always swap index 1/0
                    q0_tmem = 1
                    q1_tmem = 0
                else:
                    q0_tmem = (tile_count % NUM_Q_SCALE_TMEM_BUFFERS) * 2
                    q1_tmem = q0_tmem + 1
                k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

                _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
                NAMED_BAR_QK_EMPTY: tl.constexpr = 9
                NUM_THREADS_QK_EMPTY: tl.constexpr = 160
                if SHARE_SCALE_BUFFERS:
                    k0_tmem = 1
                    k1_tmem = 0
                    v0_tmem = 0
                else:
                    kv_scale_tmem_idx = accum_cnt_qk % NUM_KV_SCALE_TMEM_BUFFERS
                    k0_tmem = kv_scale_tmem_idx
                    k1_tmem = kv_scale_tmem_idx
                    v0_tmem = kv_scale_tmem_idx

                # PROLOGUE: Q0*K, Q1*K
                # With UNROLL_KV (non-causal only), tiles after the first skip
                # this — the transition at the end of the previous tile already
                # computed Q0*K and Q1*K.
                if not (UNROLL_KV and STAGE == 1) or tile_count == 0:
                    tlx.barrier_wait(q_fulls[q_bufIdx], q_phase)
                    tlx.tmem_copy(q_scale_tiles[0], q_scale_tmem[q0_tmem])

                    tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                    k_tile = tlx.local_trans(kv_tiles[k_bufIdx])

                    # -- compute q0 @ k ----
                    tlx.tmem_copy(kv_scale_tiles[k_bufIdx], k_scale_tmem[k0_tmem])
                    if SHARE_SCALE_BUFFERS:
                        tlx.barrier_wait(p_empties[0], qk_phase ^ 1)
                        tlx.barrier_wait(l_empties[0], l_phase ^ 1)
                    else:
                        tlx.barrier_wait(qk_empties[0], qk_phase ^ 1)
                    tlx.async_dot_scaled(
                        q_tiles[0],
                        k_tile,
                        qk_tiles[0],
                        q_scale_tmem[q0_tmem],
                        Q_FP8_FORMAT,
                        k_scale_tmem[k0_tmem],
                        K_FP8_FORMAT,
                        use_acc=False,
                        mBarriers=[qk_fulls[0]],
                    )

                    # -- compute q1 @ k ----
                    tlx.barrier_wait(q_fulls[q_bufIdx + NUM_BUFFERS_Q], q_phase)
                    if SHARE_SCALE_BUFFERS:
                        tlx.named_barrier_wait(NAMED_BAR_QK_EMPTY, NUM_THREADS_QK_EMPTY)
                    tlx.tmem_copy(q_scale_tiles[1], q_scale_tmem[q1_tmem])
                    if SHARE_SCALE_BUFFERS:
                        tlx.tmem_copy(kv_scale_tiles[k_bufIdx], k_scale_tmem[k1_tmem])
                    if SHARE_SCALE_BUFFERS:
                        tlx.barrier_wait(p_empties[1], qk_phase ^ 1)
                        tlx.barrier_wait(l_empties[1], l_phase ^ 1)
                    else:
                        tlx.barrier_wait(qk_empties[1], qk_phase ^ 1)
                    tlx.async_dot_scaled(
                        q_tiles[1],
                        k_tile,
                        qk_tiles[1],
                        q_scale_tmem[q1_tmem],
                        Q_FP8_FORMAT,
                        k_scale_tmem[k1_tmem],
                        K_FP8_FORMAT,
                        use_acc=False,
                        mBarriers=[
                            qk_fulls[1],
                            kv_empties[k_bufIdx],
                        ],
                    )

                # -- compute p0 @ v ----
                # wait for the V buffer to be populated by the producer
                tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                tlx.barrier_wait(acc_fulls[0], qk_phase)
                # Explicit SMEM->TMEM scale transfer
                tlx.tmem_copy(kv_scale_tiles[v_bufIdx], v_scale_tmem[v0_tmem])
                tlx.barrier_wait(p_fulls[0], qk_phase)
                tlx.async_dot_scaled(
                    p_tiles[0],
                    kv_tiles[v_bufIdx],
                    acc_tiles[0],
                    p_scale_tiles[0],
                    P_FP8_FORMAT,
                    v_scale_tmem[v0_tmem],
                    V_FP8_FORMAT,
                    use_acc=False,
                    mBarriers=[p_empties[0]],
                )

                acc1_init = False

                for i in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                    v_bufIdx_prev = v_bufIdx
                    qk_phase_prev = qk_phase

                    accum_cnt_qk += 1
                    accum_cnt_kv += 2
                    k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                    if SHARE_SCALE_BUFFERS:
                        # Indices based on which value of QK must be live/dead.
                        k0_tmem = 1
                        v1_tmem = 1
                        k1_tmem = 0
                        v0_tmem = 0
                    else:
                        # All buffers are the same for the same iteration.
                        kv_scale_tmem_idx = accum_cnt_qk % NUM_KV_SCALE_TMEM_BUFFERS
                        k0_tmem = kv_scale_tmem_idx
                        # V1 uses the previous location.
                        v1_tmem = v0_tmem
                        k1_tmem = kv_scale_tmem_idx
                        v0_tmem = kv_scale_tmem_idx

                    # -- compute q0 @ k ----
                    # wait for the K buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                    k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                    _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
                    if SHARE_SCALE_BUFFERS:
                        tlx.named_barrier_wait(NAMED_BAR_QK_EMPTY + 1, NUM_THREADS_QK_EMPTY)
                        tlx.tmem_copy(q_scale_tiles[0], q_scale_tmem[q0_tmem])

                    # Explicit SMEM->TMEM scale transfer
                    tlx.tmem_copy(kv_scale_tiles[k_bufIdx], k_scale_tmem[k0_tmem])
                    # Wait for the QK output to be available.
                    if SHARE_SCALE_BUFFERS:
                        tlx.barrier_wait(p_empties[0], qk_phase ^ 1)
                    else:
                        tlx.barrier_wait(qk_empties[0], qk_phase ^ 1)
                    tlx.async_dot_scaled(
                        q_tiles[0],
                        k_tile,
                        qk_tiles[0],
                        q_scale_tmem[q0_tmem],
                        Q_FP8_FORMAT,
                        k_scale_tmem[k0_tmem],
                        K_FP8_FORMAT,
                        use_acc=False,
                        mBarriers=[qk_fulls[0]],
                    )

                    # -- compute p1 @ v from the previous iteration----
                    tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
                    tlx.barrier_wait(p_fulls[1], qk_phase_prev)
                    if SHARE_SCALE_BUFFERS:
                        # Need to copy V back into the new location.
                        tlx.tmem_copy(kv_scale_tiles[v_bufIdx_prev], v_scale_tmem[v1_tmem])
                    tlx.async_dot_scaled(
                        p_tiles[1],
                        kv_tiles[v_bufIdx_prev],
                        acc_tiles[1],
                        p_scale_tiles[1],
                        P_FP8_FORMAT,
                        v_scale_tmem[v1_tmem],
                        V_FP8_FORMAT,
                        use_acc=acc1_init,
                        mBarriers=[kv_empties[v_bufIdx_prev], p_empties[1]],
                    )

                    acc1_init = True

                    # -- compute q1 @ k ----
                    if SHARE_SCALE_BUFFERS:
                        tlx.named_barrier_wait(NAMED_BAR_QK_EMPTY, NUM_THREADS_QK_EMPTY)
                        tlx.tmem_copy(q_scale_tiles[1], q_scale_tmem[q1_tmem])
                        # Copy k into the new buffer space
                        tlx.tmem_copy(kv_scale_tiles[k_bufIdx], k_scale_tmem[k1_tmem])

                    # Wait for the QK output to be available.
                    if SHARE_SCALE_BUFFERS:
                        tlx.barrier_wait(p_empties[1], qk_phase ^ 1)
                    else:
                        tlx.barrier_wait(qk_empties[1], qk_phase ^ 1)

                    tlx.async_dot_scaled(
                        q_tiles[1],
                        k_tile,
                        qk_tiles[1],
                        q_scale_tmem[q1_tmem],
                        Q_FP8_FORMAT,
                        k_scale_tmem[k1_tmem],
                        K_FP8_FORMAT,
                        use_acc=False,
                        mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                    )

                    # -- compute p0 @ v ----
                    # wait for the V buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)

                    tlx.barrier_wait(acc_fulls[0], qk_phase)
                    # Explicit SMEM->TMEM scale transfer
                    tlx.tmem_copy(kv_scale_tiles[v_bufIdx], v_scale_tmem[v0_tmem])
                    tlx.barrier_wait(p_fulls[0], qk_phase)
                    tlx.async_dot_scaled(
                        p_tiles[0],
                        kv_tiles[v_bufIdx],
                        acc_tiles[0],
                        p_scale_tiles[0],
                        P_FP8_FORMAT,
                        v_scale_tmem[v0_tmem],
                        V_FP8_FORMAT,
                        use_acc=True,
                        mBarriers=[p_empties[0]],
                    )

                tlx.tcgen05_commit(q_empties[q_bufIdx])
                tlx.tcgen05_commit(q_empties[q_bufIdx + NUM_BUFFERS_Q])
                tlx.tcgen05_commit(acc_empties[0])

                if SHARE_SCALE_BUFFERS:
                    tlx.named_barrier_wait(NAMED_BAR_QK_EMPTY + 1, NUM_THREADS_QK_EMPTY)

                # -- compute p1 @ v ----
                tlx.barrier_wait(acc_fulls[1], qk_phase)
                tlx.barrier_wait(p_fulls[1], qk_phase)
                if SHARE_SCALE_BUFFERS:
                    v1_tmem = 1
                    tlx.tmem_copy(kv_scale_tiles[v_bufIdx], v_scale_tmem[v1_tmem])
                else:
                    v1_tmem = v0_tmem
                tlx.async_dot_scaled(
                    p_tiles[1],
                    kv_tiles[v_bufIdx],
                    acc_tiles[1],
                    p_scale_tiles[1],
                    P_FP8_FORMAT,
                    v_scale_tmem[v1_tmem],
                    V_FP8_FORMAT,
                    use_acc=acc1_init,
                    mBarriers=[acc_empties[1], kv_empties[v_bufIdx], p_empties[1]],
                )

                accum_cnt_qk += 1
                accum_cnt_kv += 2
                tile_count += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # load
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )

                # Compute scale offsets based on tile position
                # Scale tensor is 5D: [B*H, M//128, HEAD_DIM//128, 2, 256] for Q
                # Scale tensor is 5D: [B*H, N//128, HEAD_DIM//128, 2, 256] for K/V
                # TMA offset: [batch_head, row_block, head_block, 0, 0]
                # Q scale offset: start_m covers 256 rows (2 scale blocks of 128 each)
                # Q0 is first half, Q1 is second half
                q_scale_m_offset_q0 = start_m * 2 * REP_M
                q_scale_m_offset_q1 = (start_m * 2 * REP_M) + REP_M
                # K/V scale offset: compute which BLOCK_N-sized data block we're in,
                # then convert to scale chunk offset (REP_N chunks per data block)
                kv_scale_n_offset = (lo // BLOCK_N) * REP_N

                # load q0 + scale
                q_bufIdx, q_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(
                    q_fulls[q_bufIdx],
                    (Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM) + Q_SCALE_BYTES,
                )
                qo_offset_y_split = qo_offset_y
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx])
                # 5D TMA offset: [batch_head, m_offset, head_offset, 0, 0]
                # off_hz is the combined batch*H + head index
                tlx.async_descriptor_load(
                    desc_q_scale,
                    q_scale_tiles[q_bufIdx],
                    [off_hz, q_scale_m_offset_q0, 0, 0, 0],
                    q_fulls[q_bufIdx],
                )

                # loop over loading k, v
                k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                # wait for the K buffer to be released by the consumer
                k_empty = tlx.local_view(kv_empties, k_bufIdx)
                tlx.barrier_wait(k_empty, k_phase ^ 1)

                # load K + scale
                k_full = tlx.local_view(kv_fulls, k_bufIdx)
                k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                tlx.barrier_expect_bytes(k_full, (K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM) + K_SCALE_BYTES)
                tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)
                # 5D TMA offset: [batch_head, n_offset, head_offset, 0, 0]
                tlx.async_descriptor_load(
                    desc_k_scale,
                    kv_scale_tiles[k_bufIdx],
                    [off_hz, kv_scale_n_offset, 0, 0, 0],
                    k_full,
                )

                # load q1 + scale
                q_bufIdx += NUM_BUFFERS_Q
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(
                    q_fulls[q_bufIdx],
                    (Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM) + Q_SCALE_BYTES,
                )
                qo_offset_y_split = qo_offset_y + BLOCK_M_SPLIT
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx])

                tlx.async_descriptor_load(
                    desc_q_scale,
                    q_scale_tiles[q_bufIdx],
                    [off_hz, q_scale_m_offset_q1, 0, 0, 0],
                    q_fulls[q_bufIdx],
                )

                v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(kv_empties, v_bufIdx)
                tlx.barrier_wait(v_empty, v_phase ^ 1)
                # load V + scale
                v_full = tlx.local_view(kv_fulls, v_bufIdx)
                v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                tlx.barrier_expect_bytes(v_full, V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM + V_SCALE_BYTES)
                tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)
                # V_scale 5D TMA offset: [batch_head, head_offset, n_offset, 0, 0]
                # V_scale has shape [B*H, HEAD_DIM//128, N//128, 2, 256] (swapped vs K_scale)
                tlx.async_descriptor_load(
                    desc_v_scale,
                    kv_scale_tiles[v_bufIdx],
                    [off_hz, 0, kv_scale_n_offset, 0, 0],
                    v_full,
                )

                kv_offset_y += BLOCK_N
                kv_scale_n_offset += REP_N
                accum_cnt_kv += 2

                for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                    k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # wait for the K buffer to be released by the consumer
                    k_empty = tlx.local_view(kv_empties, k_bufIdx)
                    tlx.barrier_wait(k_empty, k_phase ^ 1)
                    # load K + scale
                    k_full = tlx.local_view(kv_fulls, k_bufIdx)
                    k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                    tlx.barrier_expect_bytes(k_full, (K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM) + K_SCALE_BYTES)
                    tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)
                    # 5D TMA offset: [batch_head, n_offset, head_offset, 0, 0]
                    tlx.async_descriptor_load(
                        desc_k_scale,
                        kv_scale_tiles[k_bufIdx],
                        [off_hz, kv_scale_n_offset, 0, 0, 0],
                        k_full,
                    )

                    v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                    # wait for the V buffer to be released by the consumer
                    v_empty = tlx.local_view(kv_empties, v_bufIdx)
                    tlx.barrier_wait(v_empty, v_phase ^ 1)
                    # load V
                    v_full = tlx.local_view(kv_fulls, v_bufIdx)
                    v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                    tlx.barrier_expect_bytes(v_full, (V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM) + V_SCALE_BYTES)
                    tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)
                    # V_scale 5D TMA offset: [batch_head, head_offset, n_offset, 0, 0]
                    # V_scale has shape [B*H, HEAD_DIM//128, N//128, 2, 256] (swapped vs K_scale)
                    tlx.async_descriptor_load(
                        desc_v_scale,
                        kv_scale_tiles[v_bufIdx],
                        [off_hz, 0, kv_scale_n_offset, 0, 0],
                        v_full,
                    )

                    kv_offset_y += BLOCK_N
                    kv_scale_n_offset += REP_N
                    accum_cnt_kv += 2

                tile_count += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # output store group
        with tlx.async_task(num_warps=1, registers=24):
            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                _, _, _, _, qo_offset_y, _ = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )
                _, phase = get_bufidx_phase(tile_count, 1)
                for cid in tl.static_range(0, NUM_MMA_GROUPS):
                    tlx.barrier_wait(o_fulls[cid], phase)
                    qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                    tlx.async_descriptor_store(desc_o, o_tiles[cid], [qo_offset_y_split, 0])
                    tlx.async_descriptor_store_wait(0)
                    tlx.barrier_arrive(o_empties[cid])

                tile_count += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1


# ===========================================================================
# Backward pass (MXFP8)
#
# Dense and causal. Assumes N_CTX is a multiple of BLOCK_M1 (= 128) and
# BLOCK_N1 (= 128). HEAD_DIM = 128 only.
# ===========================================================================


@triton.jit  # pragma: no cover
def _attn_bwd_mxf8_preprocess(
    O,
    DO,  #
    DQ,
    Delta,  #
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NEED_DQ: tl.constexpr = True,
):
    """Compute Delta = rowsum(O * dO) per query position.

    Dense layout: O / DO are [Z, H, N_CTX, HEAD_DIM] contiguous. Element
    (z, h, n, d) at offset ((z*H + h)*N_CTX + n)*HEAD_DIM + d. Delta is
    [Z, H, N_CTX] addressed as flat [Z*H*N_CTX].
    """
    pid0 = tl.program_id(0).to(tl.int64)
    start_m = pid0 * BLOCK_M
    off_m = start_m + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1).to(tl.int64)
    off_d = tl.arange(0, HEAD_DIM)
    off_h = off_hz % H
    off_z = off_hz // H
    base = (off_z * H + off_h) * N_CTX
    o_offsets = (base + off_m[:, None]) * HEAD_DIM + off_d[None, :]
    o_mask = off_m[:, None] < N_CTX
    o = tl.load(O + o_offsets, mask=o_mask, other=0.0)
    do = tl.load(DO + o_offsets, mask=o_mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    if NEED_DQ:
        tl.store(DQ + o_offsets, 0.0, mask=o_mask, eviction_policy="evict_first")
    tl.store(Delta + base + off_m, delta, mask=off_m < N_CTX)


def _mxf8_bwd_host_descriptor_pre_hook(nargs):
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    HEAD_DIM = nargs["HEAD_DIM"]
    EPILOGUE_SUBTILE = nargs["EPILOGUE_SUBTILE"]
    DQ_REDUCE_NCOL = nargs["DQ_REDUCE_NCOL"]
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    VEC_SIZE = 32
    REP_M = math.ceil(BLOCK_M1 / 128)
    REP_N = math.ceil(math.ceil(BLOCK_N1 / VEC_SIZE) / 4)
    REP_HEAD = math.ceil(math.ceil(HEAD_DIM / VEC_SIZE) / 4)

    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    if NUM_CTAS == 2:
        # 2-CTA pair descriptors: Q/dO row views are 64x128, Q/dO/K head views
        # are 128x64, and dQ uses the packed 4D reduce-add view whose block is
        # [1, 2, 64, 32] over [Z*H, 2, N_CTX, HEAD_DIM // 2].
        nargs["desc_q"].block_shape = [1, 1, BLOCK_M1 // 2, HEAD_DIM]
        nargs["desc_qh"].block_shape = [1, 1, BLOCK_M1, HEAD_DIM // 2]
        nargs["desc_kt"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM // 2]
        nargs["desc_k"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM]
        nargs["desc_v"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM]
        nargs["desc_do"].block_shape = [1, 1, BLOCK_M1 // 2, HEAD_DIM]
        nargs["desc_doh"].block_shape = [1, 1, BLOCK_M1, HEAD_DIM // 2]
        # dQ drain: two [2, 64, 32] TMA reduce-adds per 32-column half of the
        # head dim, over the [Z, H, N_CTX, 128] FP32 dQ. Matches the reference's
        # [2, 64, 32] desc_dq: offsets [z, h, m0, col].
        nargs["desc_dq"].block_shape = [1, 2, BLOCK_M1 // 2, HEAD_DIM // 4]
        # dK/dV epilogue: the reduction task stages [128, 64] SMEM halves
        # (dkv_store_buf), so the 4D TMA block must be [1, 1, 128, 64].
        if isinstance(nargs.get("desc_dk"), TensorDescriptor):
            nargs["desc_dk"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM // 2]
        if isinstance(nargs.get("desc_dv"), TensorDescriptor):
            nargs["desc_dv"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM // 2]
    else:
        nargs["desc_q"].block_shape = [1, 1, BLOCK_M1, HEAD_DIM]
        nargs["desc_k"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM]
        if isinstance(nargs.get("desc_v"), TensorDescriptor):
            nargs["desc_v"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM]
        nargs["desc_do"].block_shape = [1, 1, BLOCK_M1, HEAD_DIM]
        # dQ is reduced to GMEM in fixed-width column chunks. Keep this descriptor
        # independent of the dK/dV epilogue subtile factor.
        if isinstance(nargs.get("desc_dq"), TensorDescriptor):
            nargs["desc_dq"].block_shape = [1, 1, BLOCK_M1, DQ_REDUCE_NCOL]
        if isinstance(nargs.get("desc_dk"), TensorDescriptor):
            nargs["desc_dk"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE]
        if isinstance(nargs.get("desc_dv"), TensorDescriptor):
            nargs["desc_dv"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE]
    if isinstance(nargs.get("desc_m"), TensorDescriptor):
        nargs["desc_m"].block_shape = [BLOCK_M1]
    if isinstance(nargs.get("desc_delta"), TensorDescriptor):
        nargs["desc_delta"].block_shape = [BLOCK_M1]

    if isinstance(nargs.get("desc_q_scale"), TensorDescriptor):
        nargs["desc_q_scale"].block_shape = [1, REP_M, REP_HEAD, 2, 256]
        if isinstance(nargs.get("desc_q_dk_scale"), TensorDescriptor):
            # MMA 4 consumes Q with the sequence dimension as the reduction
            # axis, so its scale tensor follows the swapped convention.
            nargs["desc_q_dk_scale"].block_shape = [1, REP_HEAD, REP_M, 2, 256]
        nargs["desc_k_scale"].block_shape = [1, REP_N, REP_HEAD, 2, 256]
        if isinstance(nargs.get("desc_k_dq_scale"), TensorDescriptor):
            # MMA 5 consumes K with the sequence dimension as the reduction
            # axis, so its scale tensor also uses the swapped convention.
            nargs["desc_k_dq_scale"].block_shape = [1, REP_HEAD, REP_N, 2, 256]
        if isinstance(nargs.get("desc_v_scale"), TensorDescriptor):
            nargs["desc_v_scale"].block_shape = [1, REP_N, REP_HEAD, 2, 256]
        if isinstance(nargs.get("desc_do_scale"), TensorDescriptor):
            nargs["desc_do_scale"].block_shape = [1, REP_M, REP_HEAD, 2, 256]
        if isinstance(nargs.get("desc_do_dv_scale"), TensorDescriptor):
            # MMA 3 consumes dO with the query dimension as the reduction axis,
            # so its scale tensor follows the same swapped convention as V.
            nargs["desc_do_dv_scale"].block_shape = [1, REP_HEAD, REP_M, 2, 256]


# Single-config autotune (matches the JFA backward; the kernel structure
# is highly tuned for this exact shape - see D101699854 OPTIMIZATION_REPORT
# for the perf history that led to num_warps=16 + reg-trim on Reduction /
# Load tasks).
configs_bwd_1cta = [
    triton.Config(
        {
            "BLOCK_M1": 128,
            "BLOCK_N1": 128,
            "NUM_BUFFERS_KV": 1,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_DO": 2,
            "NUM_BUFFERS_DS": 1,
            "EPILOGUE_SUBTILE": 2,
            "DQ_REDUCE_NCOL": 32,
            "NUM_CTAS": 1,
        },
        num_warps=8,
        num_stages=1,
        pre_hook=_mxf8_bwd_host_descriptor_pre_hook,
    ),
]

configs_bwd_2cta = [
    triton.Config(
        {
            "BLOCK_M1": 128,
            "BLOCK_N1": 128,
            "NUM_BUFFERS_KV": 1,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_DO": 2,
            "NUM_BUFFERS_DS": 1,
            # Required by the common kernel signature even though the 2-CTA
            # path does not consume them; keep the 1-CTA values.
            "EPILOGUE_SUBTILE": 2,
            "DQ_REDUCE_NCOL": 32,
            "NUM_CTAS": 2,
        },
        num_warps=8,
        num_stages=1,
        pre_hook=_mxf8_bwd_host_descriptor_pre_hook,
        ctas_per_cga=(2, 1, 1),
        allowDependentTwoCTA=True,
    ),
]

configs_bwd_tlx = configs_bwd_1cta + configs_bwd_2cta


def prune_mxf8_bwd_configs_by_cta(configs, _named_args, **kwargs):
    requested = kwargs.get("REQUESTED_NUM_CTAS", 1)
    return [c for c in configs if c.kwargs.get("NUM_CTAS", 1) == requested]


@triton.jit
def _get_mxf8_start_m_bwd(pid, BLOCK_N1, STAGE: tl.constexpr):
    # Backward sweeps query (M) blocks for a fixed key/value (N) tile.
    # Causal (STAGE == 3): a key block at position start_n = pid * BLOCK_N1 only
    # receives gradient from query blocks at m >= start_n, so the sweep starts at
    # the diagonal (start_m == start_n). Non-causal (STAGE == 1): full sweep.
    if STAGE == 3:
        return pid * BLOCK_N1
    else:
        tl.static_assert(STAGE == 1)
        return 0


@triton.jit
def _get_mxf8_unfused_bwd_loop_bounds(start_n, N_CTX, BLOCK_N1, STAGE: tl.constexpr):
    if STAGE == 1:
        # Sometimes-true diagonal section for causal backward.
        lo, hi = start_n, start_n + BLOCK_N1
    elif STAGE == 2:
        # Automatically-true section after the diagonal.
        lo, hi = start_n + BLOCK_N1, N_CTX
    else:
        tl.static_assert(STAGE == 3)
        # Non-causal full sweep.
        lo, hi = 0, N_CTX
    return lo, hi


@triton.jit
def _get_mxf8_bwd_tile_info(
    tile_idx,
    n_tile_num,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    STAGE: tl.constexpr,
):
    off_seq_h = tile_idx // n_tile_num
    off_z = off_seq_h // H
    off_h = off_seq_h % H
    pid = tile_idx % n_tile_num
    start_n = pid * BLOCK_N1
    base_q = (off_z * H + off_h).to(tl.int64) * N_CTX
    start_m = _get_mxf8_start_m_bwd(pid, BLOCK_N1, STAGE)
    num_steps = (N_CTX - start_m) // BLOCK_M1
    return off_seq_h, off_z, off_h, pid, start_n, base_q, start_m, num_steps


@triton.jit
def _bwd_mxf8_compute_iter_1cta(
    blk_idx,
    qk_scale,
    qk_tiles,
    qk_fulls,
    qk_empties,
    p_tiles,
    p_scale_buf_smem,
    p_fulls,
    dp_tiles,
    dp_fulls,
    dp_empties,
    ds_tiles_smem,
    ds_scale_smem,
    ds_scale_dq_smem,
    sM_tiles,
    sD_tiles,
    ds_fulls,
    ds_empties,
    m_fulls,
    m_empties,
    d_fulls,
    d_empties,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    p_dtype: tl.constexpr,
    REP_N: tl.constexpr,
    REP_M: tl.constexpr,
    DS_NUM_SUBS: tl.constexpr,
    NEED_DQ: tl.constexpr,
    NEED_DK: tl.constexpr,
    NEED_DV: tl.constexpr,
    curr_m,
    start_n,
    MASK: tl.constexpr,
):
    DS_M_SUB: tl.constexpr = BLOCK_M1 // DS_NUM_SUBS
    _, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
    ds_buf_id, ds_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)
    m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
    d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
    # Read QK from TMEM, apply sm_scale -> P
    tlx.barrier_wait(qk_fulls[0], tmem_phase)
    # Drain QK before its scale and P aliases overwrite the same TMEM region.
    qkT = tlx.local_load(tlx.local_view(qk_tiles, 0), layout=_QK_SEPARABLE_LAYOUT)
    tlx.barrier_arrive(qk_empties[0])
    tlx.barrier_wait(m_fulls[m_buf_id], m_phase)
    m = tlx.local_load(sM_tiles[m_buf_id])

    qkT_scaled = _fma_f32x2(qkT, qk_scale, -m[None, :])
    # Clamp to prevent FP32 overflow downstream in P*dP
    qkT_scaled = tl.minimum(qkT_scaled, 20.0)
    if MASK:
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        col_limit_left = (offs_n - curr_m)[:, None]
        qkT_scaled = _apply_causal_mask(qkT_scaled, col_limit_left, BLOCK_M1, TRANSPOSED=True)
    pT = tl.math.exp2(qkT_scaled)

    if NEED_DV:
        # Quantize P^T -> TMEM with fixed pow2 scale.
        # P = exp2(QK·scale - m) ∈ [0, 1], so a fixed E8M0 scale works for all
        # blocks. E8M0=119 → scale=2^(-8), inv_scale=256. This eliminates the
        # per-block amax reshape, avoiding the convert_layout before tmem_store.
        P_FIXED_INV_SCALE: tl.constexpr = 256.0
        p_fp8 = _cvt_e4m3x4_f32(_mul_f32x2(pT, P_FIXED_INV_SCALE))
        tlx.local_store(tlx.local_view(p_tiles, 0), p_fp8)

        tlx.barrier_arrive(p_fulls[0])
    tlx.barrier_arrive(m_empties[m_buf_id])
    if NEED_DQ or NEED_DK:
        tlx.barrier_wait(d_fulls[d_buf_id], d_phase)
        pT_slices = _split_n_2D(pT, DS_NUM_SUBS)

        for subtile_id in tl.static_range(DS_NUM_SUBS):
            # Finish dS for the previous M-block.
            Di = tlx.local_load(tlx.local_slice(
                sD_tiles[d_buf_id],
                [subtile_id * DS_M_SUB],
                [DS_M_SUB],
            ))
            if subtile_id == 0:
                tlx.barrier_wait(dp_fulls[0], tmem_phase)
            dpT = tlx.local_load(
                tlx.subslice(tlx.local_view(dp_tiles, 0), DS_M_SUB * subtile_id, DS_M_SUB),
                layout=_DPT_SEPARABLE_LAYOUT,
            )
            if subtile_id == DS_NUM_SUBS - 1:
                tlx.barrier_arrive(dp_empties[0])

            dsT = _mul_f32x2(pT_slices[subtile_id], _sub_f32x2(dpT, Di[None, :]))
            # Masked causal tiles can manufacture NaNs at inactive positions. The
            # non-causal path is fully tiled and has no inactive lanes, so keep the
            # compare/select off its steady-state dS critical path.
            if MASK:
                dsT = tl.where(dsT == dsT, dsT, 0.0)
            # Share one dS payload and store only the requested scale layouts.
            if subtile_id == 0:
                tlx.barrier_wait(ds_empties[ds_buf_id], ds_phase ^ 1)
            ds_fp8, ds_scale = _to_mxfp8_32x32_block(
                dsT,
                VEC_SIZE,
                p_dtype,
            )
            tlx.local_store(
                tlx.local_slice(
                    tlx.local_view(ds_tiles_smem, ds_buf_id),
                    [0, subtile_id * DS_M_SUB],
                    [BLOCK_N1, DS_M_SUB],
                ),
                ds_fp8,
            )
            if NEED_DK:
                ds_scale_packed = ds_scale.reshape([REP_N, 4, 32, REP_M, 4 // DS_NUM_SUBS]).permute(0, 3, 2, 1, 4)
                tlx.local_store(
                    tlx.local_slice(
                        tlx.local_view(ds_scale_smem, 0),
                        [0, 0, 0, 0, subtile_id * (4 // DS_NUM_SUBS)],
                        [REP_N, REP_M, 32, 4, 4 // DS_NUM_SUBS],
                    ),
                    ds_scale_packed,
                )
            if NEED_DQ:
                ds_scale_dq_packed = ds_scale.reshape([REP_N, 4, 32, REP_M, 4 // DS_NUM_SUBS]).permute(3, 0, 2, 4, 1)
                tlx.local_store(
                    tlx.local_slice(
                        tlx.local_view(ds_scale_dq_smem, 0),
                        [0, 0, 0, subtile_id * (4 // DS_NUM_SUBS), 0],
                        [REP_M, REP_N, 32, 4 // DS_NUM_SUBS, 4],
                    ),
                    ds_scale_dq_packed,
                )
        tlx.barrier_arrive(ds_fulls[ds_buf_id])
        tlx.barrier_arrive(d_empties[d_buf_id])


@triton.jit
def _bwd_mxf8_compute_inner_loop(
    blk_idx,
    qk_scale,
    qk_tiles,
    qk_fulls,
    qk_empties,
    p_tiles,
    p_scale_smem,
    dp_tiles,
    dp_fulls,
    ds_scale_smem,
    ds_scale_dq_smem,
    sM_tiles,
    sD_tiles,
    ds_empties,
    m_fulls,
    m_empties,
    d_fulls,
    d_empties,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    p_dtype: tl.constexpr,
    start_m=0,
    end_m=0,
    num_steps=0,
    start_n=0,
    MASK: tl.constexpr = False,
    USE_2CTA: tl.constexpr = False,
    # --- 1CTA-only ---
    p_fulls=None,
    dp_empties=None,
    ds_tiles_smem=None,
    ds_fulls=None,
    NUM_BUFFERS_TMEM: tl.constexpr = 1,
    NUM_BUFFERS_DS: tl.constexpr = 1,
    REP_N: tl.constexpr = 1,
    REP_M: tl.constexpr = 1,
    DS_NUM_SUBS: tl.constexpr = 2,
    NEED_DQ: tl.constexpr = True,
    NEED_DK: tl.constexpr = True,
    NEED_DV: tl.constexpr = True,
    # --- 2CTA-only ---
    tc=0,
    rank=0,
    dpr_bars=None,
    dsT_tmem=None,
    dsT_fulls=None,
    ds_x=None,
    ds_xchg=None,
    ds_scale_xchg=None,
    xchg_ready=None,
    kt_scale_smem=None,
    b_scale_tmem0=None,
    b_scale_tmem1=None,
    kdq_ready=None,
    ktsc_fulls=None,
):
    """Shared per-KV/pair query-block loop (BF16 `_bwd_compute_inner_loop` role).

    Exactly one constexpr arm compiles: the 1CTA softmax-recompute/quantize
    sweep or the 2CTA pair sweep. Both narrow compute iters are preserved
    byte-for-byte; the wrapper only holds the short block loop.
    """
    if USE_2CTA:
        for j in range(num_steps):
            _bwd_mxf8_compute_iter_2cta(
                blk_idx,
                j,
                tc,
                rank,
                qk_scale,
                qk_tiles,
                qk_fulls,
                qk_empties,
                p_tiles,
                dp_tiles,
                dp_fulls,
                dpr_bars,
                dsT_tmem,
                dsT_fulls,
                ds_x,
                ds_xchg,
                ds_scale_smem,
                ds_scale_dq_smem,
                ds_scale_xchg,
                ds_empties,
                xchg_ready,
                kt_scale_smem,
                b_scale_tmem0,
                b_scale_tmem1,
                kdq_ready,
                ktsc_fulls,
                p_scale_smem,
                sM_tiles,
                sD_tiles,
                m_fulls,
                m_empties,
                d_fulls,
                d_empties,
                j * BLOCK_M1,
                M_STAGE,
                D_STAGE,
                BLOCK_M1,
                BLOCK_N1,
                VEC_SIZE,
                p_dtype,
            )
            blk_idx += 1
    else:
        curr_m = start_m
        for _ in range(0, (end_m - start_m) // BLOCK_M1):
            _bwd_mxf8_compute_iter_1cta(
                blk_idx,
                qk_scale,
                qk_tiles,
                qk_fulls,
                qk_empties,
                p_tiles,
                p_scale_smem,
                p_fulls,
                dp_tiles,
                dp_fulls,
                dp_empties,
                ds_tiles_smem,
                ds_scale_smem,
                ds_scale_dq_smem,
                sM_tiles,
                sD_tiles,
                ds_fulls,
                ds_empties,
                m_fulls,
                m_empties,
                d_fulls,
                d_empties,
                NUM_BUFFERS_TMEM,
                NUM_BUFFERS_DS,
                M_STAGE,
                D_STAGE,
                BLOCK_M1,
                BLOCK_N1,
                VEC_SIZE,
                p_dtype,
                REP_N,
                REP_M,
                DS_NUM_SUBS,
                NEED_DQ,
                NEED_DK,
                NEED_DV,
                curr_m,
                start_n,
                MASK,
            )
            blk_idx += 1
            curr_m += BLOCK_M1
    return blk_idx


# "Separable" thread-value layout for the 128x128 QK^T accumulator read out
# of TMEM, written purely as shape/stride (flat row-major offset = n*128 + m):
# value -> M, thread -> N. Pinning this on the qkT load makes P / dP / dS share
# one register layout, so the pT split is a free register relabel and the P f8
# store is convert-free (the separable unification, now expressed in source
# instead of hand-edited TTGIR). Specialized for the BLOCK_N1=BLOCK_M1=128,
# num_warps=8 bwd config.
_QK_SEPARABLE_LAYOUT = tlx.layout(
    shape=((32, 4, 2), (32, 2)),  # (thread, value)
    stride=((128, 4096, 32), (1, 64)),
)

# The matching separable layout for the 128x64 dP^T sub-tile loads (flat
# row-major offset = n * 64 + m). This is _QK_SEPARABLE_LAYOUT minus the stage
# register bit (the sub-tile is one of the two M halves), so dS = pT * (dP - Di)
# unifies on one layout and the dS path stays convert-free. Specialized for the
# DS_NUM_SUBS=2 (DS_M_SUB=64), num_warps=8 bwd config.
_DPT_SEPARABLE_LAYOUT = tlx.layout(
    shape=((32, 4, 2), (32, )),  # (thread, value)
    stride=((64, 2048, 32), (1, )),
)


class _TmemScalesLayout(tlx.tensor_memory_scales_layout_encoding):
    """TMEM scales layout with value equality, as a JIT global needs."""

    def __eq__(self, other):
        return isinstance(other, tlx.tensor_memory_scales_layout_encoding) and (
            self.CTASplitM,
            self.CTASplitN,
            self.ctaMode,
        ) == (other.CTASplitM, other.CTASplitN, other.ctaMode)

    def __hash__(self):
        return hash((self.CTASplitM, self.CTASplitN, self.ctaMode))


# TMEM scale layout of the full-N B scales of a two-CTA M64 scaled MMA.
_TWO_CTA_RHS_SCALES = _TmemScalesLayout(ctaMode=tlx.TMemCTAMode.TwoCTA_RHS)

_DEFAULT_SCALES = _TmemScalesLayout()


@triton.jit
def _bwd_mxf8_warp_sync_2cta(x):
    """bar.warp.sync over the full warp; returns x."""
    return tl.inline_asm_elementwise(
        "bar.warp.sync -1; mov.pred $0, $1;",
        "=b,b",
        [x],
        dtype=tl.int1,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _bwd_mxf8_pair_coords_2cta(
    pair,
    rank,
    n_tile_num,
    H,
    N_CTX,
    BLOCK_N1: tl.constexpr,
):
    """Dense (batch, head, key-block, first-key) of this CTA's tile of a pair.

    Pair ``pair`` owns key blocks ``2 * pair`` and ``2 * pair + 1`` of one
    (batch, head). Dense layout: pairs are ordered batch-major, then head,
    then key-pair.
    """
    # Dense layout: pairs are ordered batch-major, then head, then key-pair.
    pairs_per_seq_h = (n_tile_num + 1) // 2
    off_seq_h = pair // pairs_per_seq_h
    pair_in_seq = pair % pairs_per_seq_h
    off_z = off_seq_h // H
    off_h = off_seq_h % H
    blk = 2 * pair_in_seq + rank
    start_n = blk * BLOCK_N1
    return off_seq_h, off_z, off_h, blk, start_n


@triton.jit
def _bwd_mxf8_compute_iter_2cta(
    blk_idx,
    j,
    tc,
    rank,
    qk_scale,
    qk_tiles,
    qk_fulls,
    qk_empties,
    p_tiles,
    dp_tiles,
    dp_fulls,
    dpr_bars,
    dsT_tmem,
    dsT_fulls,
    ds_x,
    ds_xchg,
    ds_scale_smem,
    ds_scale_dq_smem,
    ds_scale_xchg,
    ds_empties,
    xchg_ready,
    kt_scale_smem,
    b_scale_tmem0,
    b_scale_tmem1,
    kdq_ready,
    ktsc_fulls,
    p_scale_smem,
    sM_tiles,
    sD_tiles,
    m_fulls,
    m_empties,
    d_fulls,
    d_empties,
    curr_m,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    p_dtype: tl.constexpr,
):
    """One query block of the Compute warps of a CTA pair.

    P and dS of this CTA's 128 keys against the block's 128 queries. dS^T goes
    to TMEM (A operand of the pair's dK product); its two 64-query halves feed
    the pair's dQ product: the half of this CTA's queries stays in local SMEM,
    the other half is copied to the peer over DSMEM.
    """
    DS_M_SUB: tl.constexpr = BLOCK_M1 // 2
    _, tmem_phase = get_bufidx_phase(blk_idx, 1)
    m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
    d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
    tlx.barrier_wait(qk_fulls[0], tmem_phase)
    # Same separable register layout as the 1-CTA path so the pT split is a
    # free relabel and the P f8 store is convert-free.
    qkT = tlx.local_load(tlx.local_view(qk_tiles, 0), layout=_QK_SEPARABLE_LAYOUT)
    # All warps have read qk: P and the dQ B scales may overwrite it. The
    # cross-CTA arrives below come from two lanes of every warp, without a
    # partition barrier of their own.
    tl.debug_barrier()
    two_lanes = (tlx.thread_id(0) % 32) < 2
    # Real peer publication: both CTAs arrive on the leader's qk_empties (an
    # 8-warp x 2-lane x 2-CTA warp barrier) once their compute warps drained
    # qk into registers, freeing RG1 for MMA 1's scales and the dQ products.
    tlx.barrier_arrive(qk_empties[0], 1, remote_cta_rank=0, pred=two_lanes)
    if blk_idx > 0:
        # B scales of the previous block's dQ products: K M-block exponents of
        # both key tiles, full head dim, in the two-CTA RHS TMEM layout.
        if j == 0:
            kslot = (tc - 1) % 2
        else:
            kslot = tc % 2
        if j == 1:
            tlx.barrier_wait(ktsc_fulls[0], tc & 1)
        tlx.tmem_copy(kt_scale_smem[2 * kslot], b_scale_tmem0[0])
        tlx.tmem_copy(kt_scale_smem[2 * kslot + 1], b_scale_tmem1[0])
        # Each warp arrives once its own part of the B scales is stored. The
        # leader's MMA warp waits kdq_ready before issuing the dQ products, so
        # these real arrives publish the B scales to the consumer.
        tlx.barrier_arrive(kdq_ready[0], 1, remote_cta_rank=0, pred=two_lanes)
    # The block's M/Delta landed before MMA 1 was issued.
    tlx.barrier_wait(m_fulls[m_buf_id], m_phase)
    tlx.barrier_wait(d_fulls[d_buf_id], d_phase)
    m = tlx.local_load(sM_tiles[m_buf_id])
    tlx.barrier_arrive(m_empties[m_buf_id])

    neg_qkT_scaled = _fma_f32x2(qkT, -qk_scale, m[None, :])
    qkT_scaled = tl.minimum(-neg_qkT_scaled, 20.0)
    pT = tl.math.exp2(qkT_scaled)
    # P <= 1: one fixed scale 2^-8 (stored once in p_scale_smem).
    P_FIXED_INV_SCALE: tl.constexpr = 256.0
    p_fp8 = _cvt_e4m3x4_f32(_mul_f32x2(pT, P_FIXED_INV_SCALE))
    tlx.local_store(tlx.local_view(p_tiles, 0), p_fp8)
    pT_slices = _split_n_2D(pT, 2)
    _, ds_phase = get_bufidx_phase(blk_idx, 1)

    # First query half. dS^T and the next block's qk scales live in the
    # columns of this half of dP (read by both warpgroups, different columns of
    # the same lanes): its read-arrive syncs the partition before any of them
    # is written; it also shows P stored.
    Di0 = tlx.local_load(tlx.local_slice(sD_tiles[d_buf_id], [0], [DS_M_SUB]))
    tlx.barrier_wait(dp_fulls[0], tmem_phase)
    dpT0 = tlx.local_load(
        tlx.subslice(tlx.local_view(dp_tiles, 0), 0, DS_M_SUB),
        layout=_DPT_SEPARABLE_LAYOUT,
    )
    tl.debug_barrier()
    # Real peer publication: both CTAs signal the leader's dpr_bars once their
    # compute warps drained dP half 0, freeing RG2 for MMA 4's dK scales.
    tlx.barrier_arrive(dpr_bars[0], 1, remote_cta_rank=0, pred=two_lanes)
    dsT0 = _mul_f32x2(pT_slices[0], _sub_f32x2(dpT0, Di0[None, :]))
    ds_fp8_0, ds_scale_0 = _to_mxfp8_32x32_block(dsT0, VEC_SIZE, p_dtype)
    tlx.local_store(tlx.subslice(tlx.local_view(dsT_tmem, 0), 0, DS_M_SUB), ds_fp8_0)
    # The previous block's dQ products have read ds_x of both CTAs and its
    # dK product the dS scales.
    tlx.barrier_wait(ds_empties[0], ds_phase ^ 1)
    _bwd_mxf8_publish_ds_subtile_2cta(
        0,
        rank,
        ds_fp8_0,
        ds_scale_0,
        ds_x,
        ds_xchg,
        ds_scale_smem,
        ds_scale_dq_smem,
        ds_scale_xchg,
    )
    # Second query half.
    Di1 = tlx.local_load(tlx.local_slice(sD_tiles[d_buf_id], [DS_M_SUB], [DS_M_SUB]))
    dpT1 = tlx.local_load(
        tlx.subslice(tlx.local_view(dp_tiles, 0), DS_M_SUB, DS_M_SUB),
        layout=_DPT_SEPARABLE_LAYOUT,
    )
    dsT1 = _mul_f32x2(pT_slices[1], _sub_f32x2(dpT1, Di1[None, :]))
    ds_fp8_1, ds_scale_1 = _to_mxfp8_32x32_block(dsT1, VEC_SIZE, p_dtype)
    tlx.local_store(tlx.subslice(tlx.local_view(dsT_tmem, 0), DS_M_SUB, DS_M_SUB), ds_fp8_1)
    _bwd_mxf8_publish_ds_subtile_2cta(
        1,
        rank,
        ds_fp8_1,
        ds_scale_1,
        ds_x,
        ds_xchg,
        ds_scale_smem,
        ds_scale_dq_smem,
        ds_scale_xchg,
    )
    tlx.barrier_arrive(d_empties[d_buf_id])
    tlx.fence("async_shared")
    # Every lane's SMEM stores precede its warp's arrive.
    two_lanes = _bwd_mxf8_warp_sync_2cta(two_lanes)
    # Real peer publication: both CTAs signal the leader's dsT_fulls once the
    # dS^T TMEM store is visible, gating the leader's MMA 4 (dK) issue.
    tlx.barrier_arrive(dsT_fulls[0], 1, remote_cta_rank=0, pred=two_lanes)
    # The relay warp copies the peer's half of dS over DSMEM.
    tlx.barrier_arrive(xchg_ready[0])


@triton.jit
def _bwd_mxf8_publish_ds_subtile_2cta(
    s: tl.constexpr,
    rank,
    ds_fp8,
    ds_scale,
    ds_x,
    ds_xchg,
    ds_scale_smem,
    ds_scale_dq_smem,
    ds_scale_xchg,
):
    """SMEM stores of one 64-query half of dS^T.

    The dK scales (this CTA's keys x all queries) stay local. The dQ operand
    and its scales go to the CTA owning these queries: kept in ds_x[rank] when
    that is this CTA, else staged for the DSMEM copy to the peer.
    """
    ds_scale_packed = ds_scale.reshape([1, 4, 32, 1, 2]).permute(0, 3, 2, 1, 4)
    tlx.local_store(
        tlx.local_slice(tlx.local_view(ds_scale_smem, 0), [0, 0, 0, 0, s * 2], [1, 1, 32, 4, 2]),
        ds_scale_packed,
    )
    # Rows 0-63 of a 128-row atom: the receiving CTA's M64 half.
    ds_scale_dq_packed = ds_scale.reshape([1, 4, 32, 1, 2]).permute(3, 0, 2, 4, 1)
    if s == rank:
        tlx.local_store(tlx.local_view(ds_x, rank), ds_fp8)
        tlx.local_store(
            tlx.local_slice(
                tlx.local_view(ds_scale_dq_smem, rank),
                [0, 0, 0, 0, 0],
                [1, 1, 32, 2, 4],
            ),
            ds_scale_dq_packed,
        )
    else:
        tlx.local_store(tlx.local_view(ds_xchg, 0), ds_fp8)
        tlx.local_store(
            tlx.local_slice(tlx.local_view(ds_scale_xchg, 0), [0, 0, 0, 0, 0], [1, 1, 32, 2, 4]),
            ds_scale_dq_packed,
        )


@triton.jit
def _bwd_mxf8_drain_dq_2cta(
    dq_phys,
    dq_fulls,
    dq_empties,
    dq_store_buf,
    desc_dq,
    blk_idx,
    m0,
    off_seq_h,
    sm_scale,
    M_SPLIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Drain this CTA's 64-query dQ partial of a block and reduce-add it with TMA.

    Same bytes as the reference drain: the two-CTA M64 accumulator is read
    through its physical view in two 32-column halves; each [128, 32] half
    holds two 64-row x 32-column groups that are stored as one [2, 64, 32]
    TMA reduce-add. FP32 partial x sm_scale into the zero-initialized FP32 dQ.
    """
    BOX_N: tl.constexpr = HEAD_DIM // 4
    _, tmem_phase = get_bufidx_phase(blk_idx, 1)
    tlx.barrier_wait(dq_fulls[0], tmem_phase)
    dq_a = tlx.local_load(tlx.subslice(tlx.local_view(dq_phys, 0), 0, BOX_N))
    dq_a = _mul_f32x2(dq_a, sm_scale)
    tlx.async_descriptor_store_wait(0)
    tlx.local_store(dq_store_buf[0], dq_a.to(tlx.dtype_of(desc_dq)))
    dq_b = tlx.local_load(tlx.subslice(tlx.local_view(dq_phys, 0), BOX_N, BOX_N))
    dq_b = _mul_f32x2(dq_b, sm_scale)
    tlx.barrier_arrive(dq_empties[0], 1, remote_cta_rank=0)
    tlx.fence("async_shared")
    # Same [2, 64, 32] view as the reference drain: rows 0-63 of the [128, 32]
    # half hold head-dim groups g and g + 1 of 64 columns each. The 4D TMA
    # descriptor views dQ as [Z, H, N_CTX, 128]: offsets [z, h, m0, col].
    tlx.async_descriptor_store_wait(0)
    tlx.async_descriptor_store(
        desc_dq,
        tlx.local_reshape(dq_store_buf[0], [1, 2, M_SPLIT, BOX_N]),
        [off_seq_h, 0, m0, 0],
        eviction_policy="evict_last",
        store_reduce="add",
    )
    tlx.async_descriptor_store_wait(0)
    tlx.local_store(dq_store_buf[1], dq_b.to(tlx.dtype_of(desc_dq)))
    tlx.async_descriptor_store(
        desc_dq,
        tlx.local_reshape(dq_store_buf[1], [1, 2, M_SPLIT, BOX_N]),
        [off_seq_h, 0, m0, BOX_N],
        eviction_policy="evict_last",
        store_reduce="add",
    )


@triton.jit
def _bwd_mxf8_publish_m64_2cta(pub_bars, phase):
    """Both MMA warps rendezvous after the leader has staged the scales."""
    tlx.barrier_arrive(pub_bars[0], 1, remote_cta_rank=0)
    tlx.barrier_arrive(pub_bars[0], 1, remote_cta_rank=1)
    tlx.barrier_wait(pub_bars[0], phase)


@triton.jit
def _bwd_mxf8_dq_products_2cta(
    ds_x,
    kt_smem,
    dq_tiles,
    ds_scale_dq_smem,
    a_scale_tmem0,
    a_scale_tmem1,
    b_scale_tmem0,
    b_scale_tmem1,
    pub_bars,
    pub_cnt,
    dq_fulls,
    ds_empties,
    DS_FP8_FORMAT: tl.constexpr,
    K_FP8_FORMAT: tl.constexpr,
):
    """Sum both KV tiles into each CTA's 64-query dQ partial.

    The caller waits for ds_fulls and kdq_ready before staging the A scales.
    The follower MMA warp participates in both publication phases, including
    the final drain, without depending on a subsequent Compute iteration.
    """
    tlx.tmem_copy(ds_scale_dq_smem[0], a_scale_tmem0[0])
    tlx.tmem_copy(ds_scale_dq_smem[1], a_scale_tmem1[0])
    _bwd_mxf8_publish_m64_2cta(pub_bars, pub_cnt & 1)
    tlx.async_dot_scaled(
        tlx.local_trans(ds_x[0]),
        kt_smem[0],
        dq_tiles[0],
        a_scale_tmem0[0],
        DS_FP8_FORMAT,
        b_scale_tmem0[0],
        K_FP8_FORMAT,
        use_acc=False,
        two_ctas=True,
        force_async=True,
    )
    _bwd_mxf8_publish_m64_2cta(pub_bars, (pub_cnt + 1) & 1)
    tlx.async_dot_scaled(
        tlx.local_trans(ds_x[1]),
        kt_smem[1],
        dq_tiles[0],
        a_scale_tmem1[0],
        DS_FP8_FORMAT,
        b_scale_tmem1[0],
        K_FP8_FORMAT,
        use_acc=True,
        two_ctas=True,
        mBarriers=[dq_fulls[0], ds_empties[0]],
    )
    return pub_cnt + 2


@triton.jit
def _bwd_mxf8_load_2cta(
    blk_idx,
    tc,
    pair,
    rank,
    n_tile_num,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    REP_M: tl.constexpr,
    REP_N: tl.constexpr,
    BLOCK_N1_: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    M_SPLIT: tl.constexpr,
    HD_SPLIT: tl.constexpr,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    kt_block=0,
    num_steps=0,
    is_leader=False,
    desc_q=None,
    desc_qh=None,
    desc_k=None,
    desc_kt=None,
    desc_v=None,
    desc_do=None,
    desc_doh=None,
    desc_m=None,
    desc_delta=None,
    desc_q_scale=None,
    desc_q_dk_scale=None,
    desc_k_scale=None,
    desc_k_dq_scale=None,
    desc_v_scale=None,
    desc_do_scale=None,
    desc_do_dv_scale=None,
    k_smem=None,
    v_smem=None,
    kt_smem=None,
    qt_smem=None,
    qh_smem=None,
    dot_smem=None,
    doh_smem=None,
    k_scale_smem=None,
    v_scale_smem=None,
    kt_scale_smem=None,
    q_scale_smem=None,
    q_dk_scale_smem=None,
    do_scale_smem=None,
    do_scale_dv_smem=None,
    sM_tiles=None,
    sD_tiles=None,
    kv_fulls=None,
    v_fulls=None,
    kt_fulls=None,
    q_fulls=None,
    do_fulls=None,
    ktsc_fulls=None,
    k_empties=None,
    kt_empties=None,
    v_empties=None,
    q_empties=None,
    do_empties=None,
    m_fulls=None,
    m_empties=None,
    d_fulls=None,
    d_empties=None,
):
    """2CTA per-pair TMA load sequence (BF16 `_bwd_load_2cta` role).

    Byte-for-byte the original 2CTA Load task pair body: K/V + scales, then
    the per-query-block Q/dO/M/Delta/KT loads. Returns (blk_idx, tc).
    """
    off_seq_h, off_z, off_h, blk, start_n = _bwd_mxf8_pair_coords_2cta(pair, rank, n_tile_num, H, N_CTX, BLOCK_N1)
    sf_off_seq_h = off_seq_h.to(tl.int32)
    kv_scale_n = (blk * REP_N).to(tl.int32)
    kv_buf, kv_phase = get_bufidx_phase(tc, 1)
    _, tile_phase = get_bufidx_phase(tc, 1)
    kslot = tc % 2
    # K (MMA 1 A, this CTA's key tile) with its K-block exponents.
    tlx.barrier_wait(k_empties[kv_buf], kv_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(kv_fulls[kv_buf], 2 * (BLOCK_N1 * HEAD_DIM + SCALE_BYTES))
    tlx.async_descriptor_load(
        desc_k,
        k_smem[kv_buf],
        [off_z, off_h, start_n, 0],
        kv_fulls[kv_buf],
        two_ctas=True,
    )
    tlx.async_descriptor_load(
        desc_k_scale,
        k_scale_smem[kv_buf],
        [sf_off_seq_h, kv_scale_n, 0, 0, 0],
        kv_fulls[kv_buf],
        two_ctas=True,
    )
    # V (MMA 2 A, this CTA's key tile) with its exponents.
    tlx.barrier_wait(v_empties[0], tile_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(v_fulls[0], 2 * (BLOCK_N1 * HEAD_DIM + SCALE_BYTES))
    tlx.async_descriptor_load(
        desc_v,
        v_smem[0],
        [off_z, off_h, start_n, 0],
        v_fulls[0],
        two_ctas=True,
    )
    tlx.async_descriptor_load(
        desc_v_scale,
        v_scale_smem[0],
        [sf_off_seq_h, kv_scale_n, 0, 0, 0],
        v_fulls[0],
        two_ctas=True,
    )
    for j in range(0, num_steps):
        curr_m = j * BLOCK_M1
        q_buf, q_phase = get_bufidx_phase(blk_idx, 2)
        do_buf, do_phase = get_bufidx_phase(blk_idx, 2)
        q_scale_m = (curr_m // 128 * REP_M).to(tl.int32)
        # Q: full 128-query block in both layouts (MMA 1 B reads
        # all 128 Q rows against this CTA's 128 K rows; MMA 4 B
        # reads all 128 Q rows x this CTA's 64 head-dim cols),
        # with both exponent sets: 2 data + 2 scale TMAs.
        tlx.barrier_wait(q_empties[q_buf], q_phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(
                q_fulls[q_buf],
                2 * (M_SPLIT * HEAD_DIM + BLOCK_M1 * HD_SPLIT + 2 * SCALE_BYTES),
            )
        # Only the leader issues TMA; the .cta_group::2 modifier
        # delivers both CTAs' halves to both SMEM copies.
        tlx.async_descriptor_load(
            desc_q,
            qt_smem[q_buf],
            [off_z, off_h, curr_m + rank * M_SPLIT, 0],
            q_fulls[q_buf],
            two_ctas=True,
        )
        tlx.async_descriptor_load(
            desc_qh,
            qh_smem[q_buf],
            [off_z, off_h, curr_m, rank * HD_SPLIT],
            q_fulls[q_buf],
            two_ctas=True,
        )
        tlx.async_descriptor_load(
            desc_q_scale,
            q_scale_smem[q_buf],
            [sf_off_seq_h, q_scale_m, 0, 0, 0],
            q_fulls[q_buf],
            two_ctas=True,
        )
        tlx.async_descriptor_load(
            desc_q_dk_scale,
            q_dk_scale_smem[q_buf],
            [sf_off_seq_h, 0, q_scale_m, 0, 0],
            q_fulls[q_buf],
            two_ctas=True,
        )
        # M / Delta for this query block.
        m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
        tlx.barrier_wait(m_empties[m_buf_id], m_phase ^ 1)
        tlx.barrier_expect_bytes(m_fulls[m_buf_id], 4 * BLOCK_M1)
        tlx.async_descriptor_load(
            desc_m,
            sM_tiles[m_buf_id],
            [(off_seq_h * N_CTX + curr_m).to(tl.int32)],
            m_fulls[m_buf_id],
        )
        d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
        tlx.barrier_wait(d_empties[d_buf_id], d_phase ^ 1)
        tlx.barrier_expect_bytes(d_fulls[d_buf_id], 4 * BLOCK_M1)
        tlx.async_descriptor_load(
            desc_delta,
            sD_tiles[d_buf_id],
            [(off_seq_h * N_CTX + curr_m).to(tl.int32)],
            d_fulls[d_buf_id],
        )
        # dO: full 128-query block in both layouts (MMA 2 B reads
        # all 128 dO rows against this CTA's 128 V rows; MMA 3 B
        # reads all 128 dO rows x this CTA's 64 head-dim cols),
        # with both exponent sets: 2 data + 2 scale TMAs.
        tlx.barrier_wait(do_empties[do_buf], do_phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(
                do_fulls[do_buf],
                2 * (M_SPLIT * HEAD_DIM + BLOCK_M1 * HD_SPLIT + 2 * SCALE_BYTES),
            )
        tlx.async_descriptor_load(
            desc_do,
            dot_smem[do_buf],
            [off_z, off_h, curr_m + rank * M_SPLIT, 0],
            do_fulls[do_buf],
            two_ctas=True,
        )
        tlx.async_descriptor_load(
            desc_doh,
            doh_smem[do_buf],
            [off_z, off_h, curr_m, rank * HD_SPLIT],
            do_fulls[do_buf],
            two_ctas=True,
        )
        tlx.async_descriptor_load(
            desc_do_scale,
            do_scale_smem[do_buf],
            [sf_off_seq_h, q_scale_m, 0, 0, 0],
            do_fulls[do_buf],
            two_ctas=True,
        )
        tlx.async_descriptor_load(
            desc_do_dv_scale,
            do_scale_dv_smem[do_buf],
            [sf_off_seq_h, 0, q_scale_m, 0, 0],
            do_fulls[do_buf],
            two_ctas=True,
        )
        if j == kt_block:
            # Both key tiles' full K rows (MMA 5 B reads 128 K
            # rows x full 128 head-dim cols per tile; each CTA's
            # half comes from the distributed SMEM view): 2 data
            # TMAs. Free once the previous pair's last dQ product,
            # issued after this pair's first qk, has read it; first
            # read by this pair's second block, so loaded after
            # that block's Q and dO. The tiles' M-block exponents
            # go to the pair-parity slot (local, non-collective).
            tlx.barrier_wait(kt_empties[0], tile_phase ^ 1)
            if is_leader:
                tlx.barrier_expect_bytes(kt_fulls[0], 2 * (2 * BLOCK_N1 * HD_SPLIT))
            blk_t0 = blk - rank
            blk_t1 = blk - rank + 1
            for t in tl.static_range(2):
                if t == 0:
                    blk_t = blk_t0
                else:
                    blk_t = blk_t1
                tlx.async_descriptor_load(
                    desc_kt,
                    kt_smem[t],
                    [off_z, off_h, blk_t * BLOCK_N1, rank * HD_SPLIT],
                    kt_fulls[0],
                    two_ctas=True,
                )
            tlx.barrier_expect_bytes(ktsc_fulls[0], 2 * SCALE_BYTES)
            for t in tl.static_range(2):
                if t == 0:
                    blk_t = blk_t0
                else:
                    blk_t = blk_t1
                tlx.async_descriptor_load(
                    desc_k_dq_scale,
                    kt_scale_smem[2 * kslot + t],
                    [
                        sf_off_seq_h,
                        0,
                        (blk_t * REP_N).to(tl.int32),
                        0,
                        0,
                    ],
                    ktsc_fulls[0],
                )
        blk_idx += 1
    return blk_idx, tc


@triton.jit
def _bwd_mxf8_mma_tail_2cta(
    blk_idx,
    kdq_ready,
    tc,
    num_steps,
    kdq_cnt,
    pub_cnt,
    qh_smem,
    kt_smem,
    qk_tiles,
    qk_fulls,
    qk_empties,
    p_tiles,
    dp_tiles,
    dp_fulls,
    dpr_bars,
    dv_tiles,
    dv_fulls,
    dv_empties,
    dk_tiles,
    dk_fulls,
    dk_empties,
    dq_tiles,
    dq_fulls,
    dq_empties,
    dsT_tmem,
    dsT_fulls,
    ds_x,
    ds_fulls,
    ds_empties,
    q_dk_scale_smem,
    ds_scale_smem,
    ds_scale_dq_smem,
    a_scale_tmem0,
    a_scale_tmem1,
    b_scale_tmem0,
    b_scale_tmem1,
    pub_bars,
    q_empties,
    ds_scale_dk_tmem,
    q_scale_dk_tmem,
    DS_FP8_FORMAT: tl.constexpr,
    Q_FP8_FORMAT: tl.constexpr,
    K_FP8_FORMAT: tl.constexpr,
):
    """2CTA global tail: delayed dK/dQ products of the last query block.

    Identical ops/order to the original in-task tail, including the follower
    publication participation (matched M64 arrives). Returns pub_cnt.
    """
    # MMA 4 and MMA 5 of the very last block.
    if blk_idx > 0:
        q_prev, _ = get_bufidx_phase(blk_idx - 1, 2)
        _, tph_prev = get_bufidx_phase(blk_idx - 1, 1)
        _, tph_pp = get_bufidx_phase(blk_idx - 2, 1)
        # Recompute the last loop iteration's indices: the loop above
        # ended with tc/pair already incremented, so tc - 1 / the
        # matching phase select the final pair's buffers (kv) and
        # the Q/dO slot of its last block, consistent with the
        # in-loop MMA 5 issue of the same block.
        kv_buf, kv_phase = get_bufidx_phase(tc - 1, 1)
        q_buf, q_phase = get_bufidx_phase(blk_idx - 1, 2)
        tlx.barrier_wait(dsT_fulls[0], tph_prev)
        tlx.tmem_copy(ds_scale_smem[0], ds_scale_dk_tmem[0])
        tlx.tmem_copy(q_dk_scale_smem[q_prev], q_scale_dk_tmem[0])
        tlx.async_dot_scaled(
            dsT_tmem[0],
            qh_smem[q_prev],
            dk_tiles[0],
            ds_scale_dk_tmem[0],
            DS_FP8_FORMAT,
            q_scale_dk_tmem[0],
            Q_FP8_FORMAT,
            use_acc=num_steps > 1,
            two_ctas=True,
            mBarriers=[q_empties[q_prev], dk_fulls[0]],
        )
        _, ds_phase = get_bufidx_phase(blk_idx - 1, 1)
        if blk_idx >= 2:
            tlx.barrier_wait(dq_empties[0], tph_pp)
        # K's scales are constant within a pair; reuse the last
        # publication rather than waiting for a nonexistent next block.
        _, kdq_phase = get_bufidx_phase(kdq_cnt - 1, 1)
        tlx.barrier_wait(ds_fulls[0], tph_prev)
        tlx.barrier_wait(kdq_ready[0], kdq_phase)
        _, kt_phase = get_bufidx_phase(tc - 1, 1)
        pub_cnt = _bwd_mxf8_dq_products_2cta(
            ds_x,
            kt_smem,
            dq_tiles,
            ds_scale_dq_smem,
            a_scale_tmem0,
            a_scale_tmem1,
            b_scale_tmem0,
            b_scale_tmem1,
            pub_bars,
            pub_cnt,
            dq_fulls,
            ds_empties,
            DS_FP8_FORMAT,
            K_FP8_FORMAT,
        )
    return pub_cnt


@triton.jit
def _bwd_mxf8_mma_dots_2cta(
    blk_idx,
    q_fulls,
    q_empties,
    do_fulls,
    do_empties,
    j,
    num_steps,
    kv_buf,
    kv_phase,
    tile_phase,
    tc,
    kdq_cnt,
    pub_cnt,
    k_smem,
    v_smem,
    k_scale_smem,
    v_scale_smem,
    q_scale_smem,
    q_dk_scale_smem,
    do_scale_smem,
    do_scale_dv_smem,
    p_scale_smem,
    ds_scale_smem,
    qt_smem,
    qh_smem,
    dot_smem,
    doh_smem,
    kt_smem,
    qk_tiles,
    qk_fulls,
    qk_empties,
    p_tiles,
    dp_tiles,
    dp_fulls,
    dpr_bars,
    dv_tiles,
    dv_fulls,
    dv_empties,
    dk_tiles,
    dk_fulls,
    dk_empties,
    dq_tiles,
    dq_fulls,
    dq_empties,
    dsT_tmem,
    dsT_fulls,
    ds_x,
    ds_fulls,
    ds_empties,
    ds_scale_dq_smem,
    a_scale_tmem0,
    a_scale_tmem1,
    b_scale_tmem0,
    b_scale_tmem1,
    pub_bars,
    k_scale_qk_tmem,
    q_scale_qk_tmem,
    v_scale_tmem,
    do_scale_dp_tmem,
    p_scale_tmem,
    do_scale_dv_tmem,
    ds_scale_dk_tmem,
    q_scale_dk_tmem,
    kv_fulls,
    v_fulls,
    kt_fulls,
    kt_empties,
    k_empties,
    v_empties,
    kdq_ready,
    K_FP8_FORMAT: tl.constexpr,
    Q_FP8_FORMAT: tl.constexpr,
    V_FP8_FORMAT: tl.constexpr,
    DO_FP8_FORMAT: tl.constexpr,
    P_FP8_FORMAT: tl.constexpr,
    DS_FP8_FORMAT: tl.constexpr,
):
    """2CTA per-query-block MMA dot sequence (BF16 `_bwd_mma_dots_2cta` role).

    Issues MMA 1/4/2/5/3 for one query block with the delayed cross-pair dK/dQ
    schedule. Returns the updated (blk_idx, kdq_cnt, pub_cnt) scalar tuple.
    """
    q_buf, q_phase = get_bufidx_phase(blk_idx, 2)
    do_buf, do_phase = get_bufidx_phase(blk_idx, 2)
    q_prev, _ = get_bufidx_phase(blk_idx - 1, 2)
    _, tph = get_bufidx_phase(blk_idx, 1)
    _, tph_prev = get_bufidx_phase(blk_idx - 1, 1)
    _, tph_pp = get_bufidx_phase(blk_idx - 2, 1)
    has_prev = blk_idx > 0
    # MMA 1: qkT = K @ Q^T into RG1 (dQ of block g - 2
    # drained); its scales in RG2, in the columns of the
    # first query half of dP of block g - 1 (read),
    # except the pair's first block: no previous dP.
    if j == 0:
        tlx.barrier_wait(kv_fulls[kv_buf], kv_phase)
    tlx.barrier_wait(q_fulls[q_buf], q_phase)
    if blk_idx >= 2:
        tlx.barrier_wait(dq_empties[0], tph_pp)
    tlx.barrier_wait(dpr_bars[0], tph_prev)
    tlx.tmem_copy(k_scale_smem[kv_buf], k_scale_qk_tmem[0])
    tlx.tmem_copy(q_scale_smem[q_buf], q_scale_qk_tmem[0])
    tlx.async_dot_scaled(
        k_smem[kv_buf],
        tlx.local_trans(qt_smem[q_buf]),
        qk_tiles[0],
        k_scale_qk_tmem[0],
        K_FP8_FORMAT,
        q_scale_qk_tmem[0],
        Q_FP8_FORMAT,
        use_acc=False,
        two_ctas=True,
        mBarriers=[qk_fulls[0]],
    )
    if j == num_steps - 1:
        tlx.tcgen05_commit(k_empties[kv_buf], two_ctas=True)
    # MMA 4: dK += dS^T @ Q of block g - 1, dS^T from TMEM.
    if has_prev:
        tlx.barrier_wait(dsT_fulls[0], tph_prev)
        if j == 1:
            # The pair's first dK product overwrites dK.
            tlx.barrier_wait(dk_empties[0], tile_phase ^ 1)
        tlx.tmem_copy(ds_scale_smem[0], ds_scale_dk_tmem[0])
        tlx.tmem_copy(q_dk_scale_smem[q_prev], q_scale_dk_tmem[0])
        if j == 0:
            # Last block of the previous pair: dK complete.
            tlx.async_dot_scaled(
                dsT_tmem[0],
                qh_smem[q_prev],
                dk_tiles[0],
                ds_scale_dk_tmem[0],
                DS_FP8_FORMAT,
                q_scale_dk_tmem[0],
                Q_FP8_FORMAT,
                use_acc=num_steps > 1,
                two_ctas=True,
                mBarriers=[q_empties[q_prev], dk_fulls[0]],
            )
        else:
            tlx.async_dot_scaled(
                dsT_tmem[0],
                qh_smem[q_prev],
                dk_tiles[0],
                ds_scale_dk_tmem[0],
                DS_FP8_FORMAT,
                q_scale_dk_tmem[0],
                Q_FP8_FORMAT,
                use_acc=j > 1,
                two_ctas=True,
                mBarriers=[q_empties[q_prev]],
            )
    # MMA 2: dpT = V @ dO^T into RG2; scales in RG1 once qk
    # was read.
    if j == 0:
        tlx.barrier_wait(v_fulls[0], tile_phase)
    tlx.barrier_wait(do_fulls[do_buf], do_phase)
    tlx.barrier_wait(qk_empties[0], tph)
    tlx.tmem_copy(v_scale_smem[0], v_scale_tmem[0])
    tlx.tmem_copy(do_scale_smem[do_buf], do_scale_dp_tmem[0])
    tlx.async_dot_scaled(
        v_smem[0],
        tlx.local_trans(dot_smem[do_buf]),
        dp_tiles[0],
        v_scale_tmem[0],
        V_FP8_FORMAT,
        do_scale_dp_tmem[0],
        DO_FP8_FORMAT,
        use_acc=False,
        two_ctas=True,
        mBarriers=[dp_fulls[0]],
    )
    if j == num_steps - 1:
        tlx.tcgen05_commit(v_empties[0], two_ctas=True)
    # MMA 5: dQ of block g - 1 into RG1 (qk_g was read).
    if has_prev:
        _, kdq_phase = get_bufidx_phase(kdq_cnt, 1)
        kdq_cnt += 1
        _, ds_phase = get_bufidx_phase(blk_idx - 1, 1)
        tlx.barrier_wait(ds_fulls[0], tph_prev)
        tlx.barrier_wait(kdq_ready[0], kdq_phase)
        _, kt_phase = get_bufidx_phase(tc, 1)
        if j == 1:
            tlx.barrier_wait(kt_fulls[0], tile_phase)
        pub_cnt = _bwd_mxf8_dq_products_2cta(
            ds_x,
            kt_smem,
            dq_tiles,
            ds_scale_dq_smem,
            a_scale_tmem0,
            a_scale_tmem1,
            b_scale_tmem0,
            b_scale_tmem1,
            pub_bars,
            pub_cnt,
            dq_fulls,
            ds_empties,
            DS_FP8_FORMAT,
            K_FP8_FORMAT,
        )
        if j == 0:
            # The previous pair's last read of K^T.
            tlx.tcgen05_commit(kt_empties[0], two_ctas=True)
    # MMA 3: dV (+)= P^T @ dO once P is stored (the dP-read
    # arrive follows it); the pair's first one overwrites dV.
    tlx.barrier_wait(dpr_bars[0], tph)
    if j == 0:
        tlx.barrier_wait(dv_empties[0], tile_phase ^ 1)
    tlx.tmem_copy(p_scale_smem[0], p_scale_tmem[0])
    tlx.tmem_copy(do_scale_dv_smem[do_buf], do_scale_dv_tmem[0])
    tlx.async_dot_scaled(
        p_tiles[0],
        doh_smem[do_buf],
        dv_tiles[0],
        p_scale_tmem[0],
        P_FP8_FORMAT,
        do_scale_dv_tmem[0],
        DO_FP8_FORMAT,
        use_acc=j > 0,
        two_ctas=True,
        mBarriers=[do_empties[do_buf]],
    )
    if j == num_steps - 1:
        tlx.tcgen05_commit(dv_fulls[0], two_ctas=True)
    blk_idx += 1
    return blk_idx, kdq_cnt, pub_cnt


@triton.jit
def _bwd_mxf8_mma_dots_1cta(
    blk_idx,
    _i,
    num_steps,
    kv_buf_id,
    kv_phase,
    k_smem,
    k_dq_smem,
    v_smem,
    q_smem,
    q_dk_smem,
    do_smem,
    qk_tiles,
    qk_fulls,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_tiles,
    dp_fulls,
    dp_empties,
    dv_tiles,
    dv_fulls,
    dv_empties,
    dk_tiles,
    dk_fulls,
    dk_empties,
    dq_tiles,
    dq_fulls,
    dq_empties,
    ds_tiles_smem,
    ds_fulls,
    ds_empties,
    q_fulls,
    q_empties,
    q_dk_fulls,
    q_dk_empties,
    do_fulls,
    do_empties,
    do_dv_fulls,
    k_dq_fulls,
    k_dq_empties,
    k_scale_smem,
    k_scale_dq_smem,
    v_scale_smem,
    q_scale_smem,
    q_dk_scale_smem,
    do_scale_smem,
    do_scale_dv_smem,
    p_scale_smem,
    ds_scale_smem,
    ds_scale_dq_smem,
    k_scale_tmem_prologue,
    q_scale_tmem_prologue,
    v_scale_tmem_prologue,
    do_scale_dp_tmem_prologue,
    do_scale_dv_tmem_prologue,
    p_scale_tmem_prologue,
    k_scale_qk_tmem,
    q_scale_qk_tmem,
    v_scale_tmem,
    do_scale_dp_tmem,
    p_scale_tmem,
    do_scale_dv_tmem,
    ds_scale_dk_tmem,
    q_scale_dk_tmem,
    ds_scale_dq_tmem,
    k_scale_dq_tmem,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    K_FP8_FORMAT: tl.constexpr,
    Q_FP8_FORMAT: tl.constexpr,
    V_FP8_FORMAT: tl.constexpr,
    DO_FP8_FORMAT: tl.constexpr,
    P_FP8_FORMAT: tl.constexpr,
    DS_FP8_FORMAT: tl.constexpr,
    NEED_DQ: tl.constexpr = True,
    NEED_DK: tl.constexpr = True,
    NEED_DV: tl.constexpr = True,
):
    """1CTA per-KV-tile MMA dot sequence: prolog + main loop + epilog.

    Byte-for-byte the original 1CTA MMA task body; the persistent tile loop
    stays at the task site and calls this once per K/V tile.
    """
    # --- Prolog: first M-block ---
    q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
    do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
    _, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
    _, tmem_phase_prev = get_bufidx_phase(blk_idx - 1, NUM_BUFFERS_TMEM)
    _, persistent_tmem_phase = get_bufidx_phase(_i, NUM_BUFFERS_TMEM)

    # MMA 1: qkT = K @ Q^T
    tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
    # REUSE_GROUP_1 SYNCHRONIZATION:
    # MMA 5: dq_empties[0].
    # with buffers: k_scale_dq_tmem, ds_scale_dq_tmem
    # MMA 2: Handled by ds_fulls before MMA 4.
    # with buffers: v_scale_tmem, do_scale_dp_tmem
    # REUSE_GROUP_4 SYNCHRONIZATION:
    # ALL PROLOGUES MUST WAIT FOR DK_TILES TO EMPTY
    if NEED_DK:
        tlx.barrier_wait(dk_empties[0], persistent_tmem_phase ^ 1)
    tlx.tmem_copy(q_scale_smem[q_buf_id], q_scale_tmem_prologue[0])
    tlx.tmem_copy(k_scale_smem[kv_buf_id], k_scale_tmem_prologue[0])
    qT = tlx.local_trans(q_smem[q_buf_id])

    tlx.async_dot_scaled(
        k_smem[kv_buf_id],
        qT,
        qk_tiles[0],
        k_scale_tmem_prologue[0],
        K_FP8_FORMAT,
        q_scale_tmem_prologue[0],
        Q_FP8_FORMAT,
        use_acc=False,
        mBarriers=[qk_fulls[0], q_empties[q_buf_id]],
    )

    if NEED_DQ or NEED_DK:
        # MMA 2: dpT = V @ dO^T  (dP shares TMEM with dQ via reuse)
        # REUSE_GROUP_3 SYNCHRONIZATION:
        # Linear order for all deps. For epilogue MMA 5 -> MMA 2.
        # MMA 5 handled by dq_empties[0] above.
        tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
        if NEED_DQ:
            tlx.barrier_wait(dq_empties[0], tmem_phase_prev)
        tlx.tmem_copy(v_scale_smem[kv_buf_id], v_scale_tmem_prologue[0])
        tlx.tmem_copy(do_scale_smem[do_buf_id], do_scale_dp_tmem_prologue[0])
        doT = tlx.local_trans(do_smem[do_buf_id])
        tlx.async_dot_scaled(
            v_smem[kv_buf_id],
            doT,
            dp_tiles[0],
            v_scale_tmem_prologue[0],
            V_FP8_FORMAT,
            do_scale_dp_tmem_prologue[0],
            DO_FP8_FORMAT,
            use_acc=False,
            mBarriers=[dp_fulls[0], do_empties[do_buf_id]] if not NEED_DV else [dp_fulls[0]],
        )

    if NEED_DV:
        # MMA 3: dV += P^T @ dO  (P_scale on-the-fly)
        # REUSE_GROUP_1 SYNCHRONIZATION:
        # p_fulls Waits for QK_EMPTIES implicitly
        tlx.barrier_wait(p_fulls[0], tmem_phase)
        tlx.barrier_wait(dv_empties[0], persistent_tmem_phase ^ 1)
        if not (NEED_DQ or NEED_DK):
            tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
        tlx.barrier_wait(do_dv_fulls[do_buf_id], do_phase)
        tlx.fence("async_shared")
        tlx.tmem_copy(p_scale_smem[0], p_scale_tmem_prologue[0])
        tlx.tmem_copy(do_scale_dv_smem[do_buf_id], do_scale_dv_tmem_prologue[0])
        # Fence for the p_scale
        tlx.async_dot_scaled(
            p_tiles[0],
            do_smem[do_buf_id],
            dv_tiles[0],
            p_scale_tmem_prologue[0],
            P_FP8_FORMAT,
            do_scale_dv_tmem_prologue[0],
            DO_FP8_FORMAT,
            use_acc=False,
            mBarriers=[
                do_empties[do_buf_id],
            ],
        )
    blk_idx += 1

    # Wait for MMA 4. This avoids needing to thread this into
    # the body/epilogue with a conditional.
    if NEED_DQ:
        tlx.barrier_wait(k_dq_fulls[kv_buf_id], kv_phase)

    # --- Main loop: iters 1 .. num_steps-1 ---
    for j in range(1, num_steps):
        q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
        _, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
        prev_blk_idx = blk_idx - 1
        q_buf_id_prev, q_phase_prev = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
        _, tmem_phase_prev = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
        ds_buf_id_prev, ds_phase_prev = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)

        # MMA 1: qkT = K @ Q^T (current)
        tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
        # REUSE_GROUP_1 SYNCHRONIZATION:
        # MMA 5: dq_empties[0].
        # with buffers: k_scale_dq_tmem, ds_scale_dq_tmem
        # MMA 2: Handled by ds_fulls before MMA 4.
        # Not needed for prologue.
        # with buffers: v_scale_tmem, do_scale_dp_tmem
        if NEED_DQ:
            tlx.barrier_wait(dq_empties[0], tmem_phase_prev ^ 1)
        # REUSE_GROUP_3: wait for Compute to finish reading dp_tiles
        # before overwriting with scale tmem_copies.
        if NEED_DQ or NEED_DK:
            tlx.barrier_wait(dp_empties[0], tmem_phase_prev)
        else:
            prev_do_buf_id, prev_do_phase = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DO)
            tlx.barrier_wait(do_empties[prev_do_buf_id], prev_do_phase)
        tlx.tmem_copy(k_scale_smem[kv_buf_id], k_scale_qk_tmem[0])
        tlx.tmem_copy(q_scale_smem[q_buf_id], q_scale_qk_tmem[0])
        qT = tlx.local_trans(q_smem[q_buf_id])
        tlx.async_dot_scaled(
            k_smem[kv_buf_id],
            qT,
            qk_tiles[0],
            k_scale_qk_tmem[0],
            K_FP8_FORMAT,
            q_scale_qk_tmem[0],
            Q_FP8_FORMAT,
            use_acc=False,
            mBarriers=[qk_fulls[0], q_empties[q_buf_id]],
        )

        if NEED_DQ or NEED_DK:
            tlx.barrier_wait(ds_fulls[ds_buf_id_prev], ds_phase_prev)
            tlx.barrier_wait(qk_empties[0], tmem_phase)
        if NEED_DQ:
            # MMA 5: dQ = dS^T_trans @ K (previous M-block)
            # REUSE_GROUP_3 SYNCHRONIZATION:
            # Linear order for all deps. For body MMA 2 -> MMA 5.
            # MMA 2 handled by ds_fulls[ds_buf_id_prev].
            # REUSE_GROUP_1: wait for Compute to finish reading qk_tiles
            # before overwriting with scale tmem_copies.
            # Copy the dQ-specific dS scales from SMEM to TMEM.
            # Fence for scale copies to be visible.
            tlx.fence("async_shared")
            tlx.tmem_copy(ds_scale_dq_smem[0], ds_scale_dq_tmem[0])
            tlx.tmem_copy(k_scale_dq_smem[kv_buf_id], k_scale_dq_tmem[0])
            tlx.async_dot_scaled(
                tlx.local_trans(ds_tiles_smem[ds_buf_id_prev]),
                k_dq_smem[kv_buf_id],
                dq_tiles[0],
                ds_scale_dq_tmem[0],
                DS_FP8_FORMAT,
                k_scale_dq_tmem[0],
                K_FP8_FORMAT,
                use_acc=False,
                mBarriers=[dq_fulls[0], ds_empties[ds_buf_id_prev]] if not NEED_DK else [dq_fulls[0]],
            )

        do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)

        if NEED_DQ or NEED_DK:
            # MMA 2: dpT = V @ dO^T (current). Issued before MMA 4 so
            # the dP that Compute is waiting on is not queued behind dK.
            # REUSE_GROUP_3 SYNCHRONIZATION:
            # dP overwrites dQ(prev): wait for Reduction to drain it.
            tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
            if NEED_DQ:
                tlx.barrier_wait(dq_empties[0], tmem_phase_prev)
            tlx.tmem_copy(v_scale_smem[kv_buf_id], v_scale_tmem[0])
            tlx.tmem_copy(do_scale_smem[do_buf_id], do_scale_dp_tmem[0])
            doT = tlx.local_trans(do_smem[do_buf_id])
            tlx.async_dot_scaled(
                v_smem[kv_buf_id],
                doT,
                dp_tiles[0],
                v_scale_tmem[0],
                V_FP8_FORMAT,
                do_scale_dp_tmem[0],
                DO_FP8_FORMAT,
                use_acc=False,
                mBarriers=[dp_fulls[0], do_empties[do_buf_id]] if not NEED_DV else [dp_fulls[0]],
            )

        if NEED_DK:
            # MMA 4: dK += dS^T @ Q (previous M-block, dS from SMEM)

            # REUSE_GROUP_1 SYNCHRONIZATION:
            # MMA 4 scales live in RG1; qk_empties above guarantees
            # Compute has drained qk_tiles, and the next MMA 1 is
            # ordered after this MMA by tcgen05 issue order. MMA 4 does
            # not touch RG3, so it no longer waits for the dQ drain.

            # REUSE_GROUP_4 SYNCHRONIZATION:
            # DK_EMPTIES must wait for the prologue
            # to finish. All of these are grouped into
            # another bucket.
            # MMA 1: Handled by QK iter 1 reusing QK
            # MMA 2: Handled by ds_fulls barrier in MMA 5
            # MMA 3: handled by same-warp-group tcgen05 issue order.
            tlx.barrier_wait(q_dk_fulls[q_buf_id_prev], q_phase_prev)
            # Fence for ds_scale_smem to be visible.
            tlx.fence("async_shared")
            tlx.tmem_copy(ds_scale_smem[0], ds_scale_dk_tmem[0])
            tlx.tmem_copy(q_dk_scale_smem[q_buf_id_prev], q_scale_dk_tmem[0])
            tlx.async_dot_scaled(
                ds_tiles_smem[ds_buf_id_prev],
                q_dk_smem[q_buf_id_prev],
                dk_tiles[0],
                ds_scale_dk_tmem[0],
                DS_FP8_FORMAT,
                q_scale_dk_tmem[0],
                Q_FP8_FORMAT,
                use_acc=(j - 1) > 0,
                mBarriers=[
                    ds_empties[ds_buf_id_prev],
                    q_dk_empties[q_buf_id_prev],
                ],
            )

        if NEED_DV:
            # MMA 3: dV += P^T @ dO (current)
            tlx.barrier_wait(p_fulls[0], tmem_phase)
            if not (NEED_DQ or NEED_DK):
                tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
            tlx.barrier_wait(do_dv_fulls[do_buf_id], do_phase)
            tlx.tmem_copy(do_scale_dv_smem[do_buf_id], do_scale_dv_tmem[0])
            tlx.tmem_copy(p_scale_smem[0], p_scale_tmem[0])
            # Fence for the p_scale
            tlx.fence("async_shared")
            tlx.async_dot_scaled(
                p_tiles[0],
                do_smem[do_buf_id],
                dv_tiles[0],
                p_scale_tmem[0],
                P_FP8_FORMAT,
                do_scale_dv_tmem[0],
                DO_FP8_FORMAT,
                use_acc=True,
                mBarriers=[
                    do_empties[do_buf_id],
                ],
            )
        blk_idx += 1

    if NEED_DV:
        tlx.tcgen05_commit(dv_fulls[0])

    # --- Epilog: last dK / dQ ---
    prev_blk_idx = blk_idx - 1
    q_buf_id, q_phase = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
    _, tmem_phase = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
    ds_buf_id, ds_phase = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)

    if NEED_DQ or NEED_DK:
        tlx.barrier_wait(ds_fulls[ds_buf_id], ds_phase)
    if NEED_DK:
        # MMA 4: dK += dS^T @ Q (last)
        # REUSE_GROUP_3 SYNCHRONIZATION:
        # Linear order for all deps. For epilogue MMA 2 -> MMA 4.
        # MMA 2 handled by ds_fulls[ds_buf_id].
        tlx.barrier_wait(q_dk_fulls[q_buf_id], q_phase)
        # Copy from SMEM to TMEM
        # Make the dS scales visible to the asynchronous copy.
        tlx.fence("async_shared")
        tlx.tmem_copy(q_dk_scale_smem[q_buf_id], q_scale_dk_tmem[0])
        tlx.tmem_copy(ds_scale_smem[0], ds_scale_dk_tmem[0])
        tlx.async_dot_scaled(
            ds_tiles_smem[ds_buf_id],
            q_dk_smem[q_buf_id],
            dk_tiles[0],
            ds_scale_dk_tmem[0],
            DS_FP8_FORMAT,
            q_scale_dk_tmem[0],
            Q_FP8_FORMAT,
            use_acc=num_steps > 1,
            mBarriers=[q_dk_empties[q_buf_id], dk_fulls[0]]
            if NEED_DQ else [q_dk_empties[q_buf_id], dk_fulls[0], ds_empties[ds_buf_id]],
        )
    if NEED_DQ:
        # MMA 5: dQ = dS^T_trans @ K (last)
        # REUSE_GROUP_3 SYNCHRONIZATION:
        # Linear order for all deps. For epilogue MMA 4 -> MMA 5.
        # MMA 4 is ordered before MMA 5 by same-warp-group tcgen05 issue order.
        tlx.barrier_wait(dq_empties[0], tmem_phase ^ 1)
        # Experiment: for the N_CTX=128 repro, MMA 5 epilogue can reuse
        # the dQ scales packed into SMEM during dS quantization and
        # copied into TMEM here.
        # Fence for ds_scale_dq_smem to be visible.
        tlx.fence("async_shared")
        tlx.tmem_copy(ds_scale_dq_smem[0], ds_scale_dq_tmem[0])
        tlx.tmem_copy(k_scale_dq_smem[kv_buf_id], k_scale_dq_tmem[0])
        tlx.async_dot_scaled(
            tlx.local_trans(ds_tiles_smem[ds_buf_id]),
            k_dq_smem[kv_buf_id],
            dq_tiles[0],
            ds_scale_dq_tmem[0],
            DS_FP8_FORMAT,
            k_scale_dq_tmem[0],
            K_FP8_FORMAT,
            use_acc=False,
            mBarriers=[
                dq_fulls[0],
                ds_empties[ds_buf_id],
                k_dq_empties[kv_buf_id],
            ],
        )
    return blk_idx


@triton.jit
def _bwd_mxf8_load_1cta(
    blk_idx,
    _i,
    tile_idx,
    tile_idx_start,
    tile_idx_step,
    tiles_per_sm,
    n_tile_num,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    STAGE: tl.constexpr,
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    desc_m,
    desc_delta,
    desc_q_scale,
    desc_q_dk_scale,
    desc_k_scale,
    desc_k_dq_scale,
    desc_v_scale,
    desc_do_scale,
    desc_do_dv_scale,
    k_smem,
    k_dq_smem,
    v_smem,
    q_smem,
    q_dk_smem,
    do_smem,
    q_scale_smem,
    q_dk_scale_smem,
    k_scale_smem,
    k_scale_dq_smem,
    v_scale_smem,
    do_scale_smem,
    do_scale_dv_smem,
    sM_tiles,
    sD_tiles,
    q_fulls,
    q_empties,
    q_dk_fulls,
    q_dk_empties,
    k_dq_fulls,
    k_dq_empties,
    do_fulls,
    do_empties,
    do_dv_fulls,
    m_fulls,
    m_empties,
    d_fulls,
    d_empties,
    Q_BYTES: tl.constexpr,
    K_BYTES: tl.constexpr,
    DO_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    REP_M: tl.constexpr,
    REP_N: tl.constexpr,
    NEED_DQ: tl.constexpr = True,
    NEED_DK: tl.constexpr = True,
    NEED_DV: tl.constexpr = True,
):
    """1CTA per-KV-tile TMA load sequence (BF16 `_bwd_load_1cta` role).

    Byte-for-byte the original 1CTA Load task body; the persistent tile loop
    stays at the task site and calls this once per K/V tile.
    """
    off_seq_h, off_z, off_h, pid, start_n, base_q, start_m, num_steps = (_get_mxf8_bwd_tile_info(
        tile_idx,
        n_tile_num,
        H,
        N_CTX,
        BLOCK_M1,
        BLOCK_N1,
        STAGE,
    ))
    # Scale TMA layout: [Z*H, REP_N (or REP_M), REP_HEAD, 2, 256].
    # Tile selects (z, h) via off_seq_h, and the M / N index via
    # the 2nd dim.
    sf_off_seq_h = off_seq_h.to(tl.int32)
    kv_scale_n = pid * REP_N

    # Load K data + scale and first Q + scale
    # Share 1 barrier.
    curr_m = start_m
    q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
    q_scale_m = (curr_m // 128) * REP_M
    # Share this barrier to signify K empty
    tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
    kv_buf_id, kv_phase = get_bufidx_phase(_i, NUM_BUFFERS_KV)
    tlx.barrier_expect_bytes(
        q_fulls[kv_buf_id],
        (K_BYTES * BLOCK_N1 * HEAD_DIM) + SCALE_BYTES + (Q_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES,
    )
    tlx.async_descriptor_load(
        desc_k,
        k_smem[kv_buf_id],
        [off_z, off_h, start_n, 0],
        q_fulls[kv_buf_id],
    )
    tlx.async_descriptor_load(
        desc_k_scale,
        k_scale_smem[kv_buf_id],
        [sf_off_seq_h, kv_scale_n.to(tl.int32), 0, 0, 0],
        q_fulls[kv_buf_id],
    )
    # Load first Q + scale
    tlx.async_descriptor_load(
        desc_q,
        q_smem[q_buf_id],
        [off_z, off_h, curr_m, 0],
        q_fulls[q_buf_id],
    )
    tlx.async_descriptor_load(
        desc_q_scale,
        q_scale_smem[q_buf_id],
        [sf_off_seq_h, q_scale_m, 0, 0, 0],
        q_fulls[q_buf_id],
    )
    m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
    tlx.barrier_wait(m_empties[m_buf_id], m_phase ^ 1)
    tlx.barrier_expect_bytes(m_fulls[m_buf_id], 4 * BLOCK_M1)
    tlx.async_descriptor_load(
        desc_m,
        sM_tiles[m_buf_id],
        [(base_q + curr_m).to(tl.int32)],
        m_fulls[m_buf_id],
    )

    # Load V data + scale and do data + scale.
    # Share 1 barrier. V / V_scale are per-tile KV buffers freed by the
    # same completion barrier as dO. MMA 3 arrives only after the
    # same-warp MMA 2 issue that reads V and dO, so this wait protects
    # both payloads before either single staging slot is refilled.
    do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
    do_scale_m = (curr_m // 128) * REP_M
    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
    # V is single-buffered per tile. With multiple dO slots, the
    # slot wait above only covers older steps, so also wait for
    # the previous step (the prior tile's last dP MMA, which reads
    # V) before overwriting V.
    prev_do_buf_id, prev_do_phase = get_bufidx_phase(blk_idx - 1, NUM_BUFFERS_DO)
    tlx.barrier_wait(do_empties[prev_do_buf_id], prev_do_phase, blk_idx > 0)
    tlx.barrier_expect_bytes(
        do_fulls[do_buf_id],
        DO_BYTES * BLOCK_M1 * HEAD_DIM +
        ((K_BYTES * BLOCK_N1 * HEAD_DIM + 2 * SCALE_BYTES) if NEED_DQ or NEED_DK else 0),
    )
    if NEED_DQ or NEED_DK:
        tlx.async_descriptor_load(
            desc_v,
            v_smem[kv_buf_id],
            [off_z, off_h, start_n, 0],
            do_fulls[do_buf_id],
        )
        tlx.async_descriptor_load(
            desc_v_scale,
            v_scale_smem[kv_buf_id],
            [sf_off_seq_h, kv_scale_n.to(tl.int32), 0, 0, 0],
            do_fulls[do_buf_id],
        )
    tlx.async_descriptor_load(
        desc_do,
        do_smem[do_buf_id],
        [off_z, off_h, curr_m, 0],
        do_fulls[do_buf_id],
    )
    if NEED_DQ or NEED_DK:
        tlx.async_descriptor_load(
            desc_do_scale,
            do_scale_smem[do_buf_id],
            [sf_off_seq_h, do_scale_m, 0, 0, 0],
            do_fulls[do_buf_id],
            eviction_policy="evict_last",
        )
    if NEED_DV:
        tlx.barrier_expect_bytes(do_dv_fulls[do_buf_id], SCALE_BYTES)
        tlx.async_descriptor_load(
            desc_do_dv_scale,
            do_scale_dv_smem[do_buf_id],
            [sf_off_seq_h, 0, do_scale_m, 0, 0],
            do_dv_fulls[do_buf_id],
            eviction_policy="evict_last",
        )
    if NEED_DQ or NEED_DK:
        d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
        tlx.barrier_wait(d_empties[d_buf_id], d_phase ^ 1)
        tlx.barrier_expect_bytes(d_fulls[d_buf_id], 4 * BLOCK_M1)
        tlx.async_descriptor_load(
            desc_delta,
            sD_tiles[d_buf_id],
            [(base_q + curr_m).to(tl.int32)],
            d_fulls[d_buf_id],
        )
    if NEED_DQ:
        tlx.barrier_wait(k_dq_empties[kv_buf_id], kv_phase ^ 1)
        tlx.barrier_expect_bytes(
            k_dq_fulls[kv_buf_id],
            (K_BYTES * BLOCK_N1 * HEAD_DIM) + SCALE_BYTES,
        )
        tlx.async_descriptor_load(
            desc_k,
            k_dq_smem[kv_buf_id],
            [off_z, off_h, start_n, 0],
            k_dq_fulls[kv_buf_id],
        )
        tlx.async_descriptor_load(
            desc_k_dq_scale,
            k_scale_dq_smem[kv_buf_id],
            [sf_off_seq_h, 0, kv_scale_n.to(tl.int32), 0, 0],
            k_dq_fulls[kv_buf_id],
        )
    curr_m += BLOCK_M1
    blk_idx += 1

    # Load subsequent Q / dO tiles.
    for _j in range(1, num_steps):
        prev_blk_idx = blk_idx - 1
        prev_m = curr_m - BLOCK_M1
        prev_q_buf_id, prev_q_phase = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
        q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
        do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
        q_scale_m = (curr_m // 128) * REP_M
        do_scale_m = (curr_m // 128) * REP_M

        tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
        tlx.barrier_expect_bytes(q_fulls[q_buf_id], (Q_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES)
        tlx.async_descriptor_load(
            desc_q,
            q_smem[q_buf_id],
            [off_z, off_h, curr_m, 0],
            q_fulls[q_buf_id],
        )
        tlx.async_descriptor_load(
            desc_q_scale,
            q_scale_smem[q_buf_id],
            [sf_off_seq_h, q_scale_m, 0, 0, 0],
            q_fulls[q_buf_id],
        )
        m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
        tlx.barrier_wait(m_empties[m_buf_id], m_phase ^ 1)
        tlx.barrier_expect_bytes(m_fulls[m_buf_id], 4 * BLOCK_M1)
        tlx.async_descriptor_load(
            desc_m,
            sM_tiles[m_buf_id],
            [(base_q + curr_m).to(tl.int32)],
            m_fulls[m_buf_id],
        )
        tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
        tlx.barrier_expect_bytes(
            do_fulls[do_buf_id],
            DO_BYTES * BLOCK_M1 * HEAD_DIM + (SCALE_BYTES if NEED_DQ or NEED_DK else 0),
        )
        tlx.async_descriptor_load(
            desc_do,
            do_smem[do_buf_id],
            [off_z, off_h, curr_m, 0],
            do_fulls[do_buf_id],
        )
        if NEED_DQ or NEED_DK:
            tlx.async_descriptor_load(
                desc_do_scale,
                do_scale_smem[do_buf_id],
                [sf_off_seq_h, do_scale_m, 0, 0, 0],
                do_fulls[do_buf_id],
                eviction_policy="evict_last",
            )
        if NEED_DV:
            tlx.barrier_expect_bytes(do_dv_fulls[do_buf_id], SCALE_BYTES)
            tlx.async_descriptor_load(
                desc_do_dv_scale,
                do_scale_dv_smem[do_buf_id],
                [sf_off_seq_h, 0, do_scale_m, 0, 0],
                do_dv_fulls[do_buf_id],
                eviction_policy="evict_last",
            )
        if NEED_DQ or NEED_DK:
            d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
            tlx.barrier_wait(d_empties[d_buf_id], d_phase ^ 1)
            tlx.barrier_expect_bytes(d_fulls[d_buf_id], 4 * BLOCK_M1)
            tlx.async_descriptor_load(
                desc_delta,
                sD_tiles[d_buf_id],
                [(base_q + curr_m).to(tl.int32)],
                d_fulls[d_buf_id],
            )
        if NEED_DK:
            # Q-for-dK is consumed after the current Q, dO, and Delta
            # paths. Defer its reuse wait so it cannot prevent those
            # critical-path TMA requests from being issued. The same
            # empty/full phase still protects the single staging slot.
            tlx.barrier_wait(q_dk_empties[prev_q_buf_id], prev_q_phase ^ 1)
            tlx.barrier_expect_bytes(
                q_dk_fulls[prev_q_buf_id],
                (Q_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES,
            )
            # prev_blk_idx is the global ring-buffer position; q_dk
            # addresses must stay local to the current (z, h, pid) tile.
            tlx.async_descriptor_load(
                desc_q,
                q_dk_smem[prev_q_buf_id],
                [off_z, off_h, prev_m, 0],
                q_dk_fulls[prev_q_buf_id],
            )
            tlx.async_descriptor_load(
                desc_q_dk_scale,
                q_dk_scale_smem[prev_q_buf_id],
                [sf_off_seq_h, 0, (prev_m // 128) * REP_M, 0, 0],
                q_dk_fulls[prev_q_buf_id],
            )
        curr_m += BLOCK_M1
        blk_idx += 1
    if NEED_DK:
        last_blk_idx = blk_idx - 1
        last_m = curr_m - BLOCK_M1
        last_q_buf_id, last_q_phase = get_bufidx_phase(last_blk_idx, NUM_BUFFERS_Q)
        tlx.barrier_wait(q_dk_empties[last_q_buf_id], last_q_phase ^ 1)
        tlx.barrier_expect_bytes(
            q_dk_fulls[last_q_buf_id],
            (Q_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES,
        )
        tlx.async_descriptor_load(
            desc_q,
            q_dk_smem[last_q_buf_id],
            [off_z, off_h, last_m, 0],
            q_dk_fulls[last_q_buf_id],
        )
        tlx.async_descriptor_load(
            desc_q_dk_scale,
            q_dk_scale_smem[last_q_buf_id],
            [sf_off_seq_h, 0, (last_m // 128) * REP_M, 0, 0],
            q_dk_fulls[last_q_buf_id],
        )
    return blk_idx, tile_idx


@triton.autotune(
    configs=configs_bwd_tlx,
    key=["N_CTX", "HEAD_DIM", "H", "STAGE", "REQUESTED_NUM_CTAS"],
    prune_configs_by={"early_config_prune": prune_mxf8_bwd_configs_by_cta},
)
@triton.jit  # pragma: no cover
def _attn_bwd_mxf8_ws(
    desc_q,
    desc_qh,
    desc_k,
    desc_kt,
    desc_v,
    desc_do,
    desc_doh,
    desc_dq,
    desc_dk,
    desc_dv,
    sm_scale,
    desc_m,
    desc_delta,
    Z,
    H,
    N_CTX,
    desc_q_scale,
    desc_q_dk_scale,
    desc_k_scale,
    desc_k_dq_scale,
    desc_v_scale,
    desc_do_scale,
    desc_do_dv_scale,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    DQ_REDUCE_NCOL: tl.constexpr,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    STAGE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    REQUESTED_NUM_CTAS: tl.constexpr,
    NEED_DQ: tl.constexpr = True,
    NEED_DK: tl.constexpr = True,
    NEED_DV: tl.constexpr = True,
) -> None:
    # One backward JIT for both CTA modes. NUM_CTAS is config-owned; the host
    # passes REQUESTED_NUM_CTAS separately so the autotune key keeps the two
    # modes' compilations apart. Exactly one static branch compiles, so the
    # peer/cluster ops of the 2-CTA pipeline are absent from 1-CTA codegen.
    USE_2CTA: tl.constexpr = NUM_CTAS == 2
    if USE_2CTA:
        tl.static_assert(NEED_DQ and NEED_DK and NEED_DV)
        tl.static_assert(HEAD_DIM == 128)
        tl.static_assert(BLOCK_N1 == 128)
        tl.static_assert(BLOCK_M1 == 128)
        tl.static_assert(STAGE == 1)
        tl.static_assert(NUM_BUFFERS_DS == 1)
        NUM_BUFFERS_TMEM: tl.constexpr = 1

        VEC_SIZE: tl.constexpr = 32
        M_SPLIT: tl.constexpr = BLOCK_M1 // 2
        HD_SPLIT: tl.constexpr = HEAD_DIM // 2
        REP_M: tl.constexpr = triton.cdiv(BLOCK_M1, 128)
        REP_N: tl.constexpr = triton.cdiv(triton.cdiv(BLOCK_N1, VEC_SIZE), 4)
        REP_HEAD: tl.constexpr = triton.cdiv(triton.cdiv(HEAD_DIM, VEC_SIZE), 4)

        Q_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
        K_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
        DO_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_do))
        SCALE_BYTES: tl.constexpr = REP_M * REP_HEAD * 2 * 256
        SCALE_TMEM_COLS: tl.constexpr = SCALE_BYTES // BLOCK_M1

        Q_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_q))
        K_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_k))
        V_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_v))
        DO_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_do))
        p_dtype = tlx.dtype_of(desc_q)
        P_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(p_dtype)
        DS_FP8_FORMAT: tl.constexpr = P_FP8_FORMAT

        qk_scale = sm_scale * 1.44269504  # sm_scale / ln(2) for exp2

        qk_scale = sm_scale * 1.44269504  # sm_scale / ln(2) for exp2

        rank = tlx.cluster_cta_rank()
        is_leader = rank == 0
        prog_id = tl.program_id(0)
        num_progs = tl.num_programs(0)
        cluster_id = prog_id // 2
        num_clusters = num_progs // 2
        num_steps: tl.constexpr = N_CTX // BLOCK_M1

        # Dense pair decomposition: each cluster owns a contiguous range of pairs.
        # Pair p owns key blocks 2p and 2p+1 of one (batch, head); a trailing odd
        # block pairs against a zeroed block. N_CTX % 256 == 0 keeps this exact.
        n_tile_num = N_CTX // BLOCK_N1
        pairs_per_seq_h = (n_tile_num + 1) // 2
        total_pairs = pairs_per_seq_h * Z * H

        # ===== TMEM =====
        # RG1 cols   0..127  qk (MMA 1); once Compute read it: P (MMA 3 A), dQ
        #                    (MMA 5, two-CTA M64) and the scales of MMA 2, 3, 5.
        # RG2 cols 128..255  dP (MMA 2); once Compute read it: dS^T (MMA 4 A) and
        #                    the scales of MMA 1 and MMA 4.
        # RG3 cols 256..383  dV          RG4 cols 384..511  dK
        tmem_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
        qk_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tl.float32, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        p_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), p_dtype, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        dq_tiles = tlx.local_alloc((M_SPLIT, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        # The two-CTA M64 accumulator occupies 128 lanes x 64 columns.
        dq_phys = tlx.local_alloc((BLOCK_M1, HD_SPLIT), tl.float32, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        v_scale_tmem = tlx.local_alloc((BLOCK_N1, 4), tl.uint8, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        do_scale_dp_tmem = tlx.local_alloc((BLOCK_M1, 4), tl.uint8, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        p_scale_tmem = tlx.local_alloc((BLOCK_N1, 4), tl.uint8, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        do_scale_dv_tmem = tlx.local_alloc((HEAD_DIM, 4), tl.uint8, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        a_scale_tmem0 = tlx.local_alloc((BLOCK_N1, 4), tl.uint8, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        a_scale_tmem1 = tlx.local_alloc((BLOCK_N1, 4), tl.uint8, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        # Full-N B scales of the M64 dQ products: staged by both CTAs' Compute
        # warps, so the two-CTA RHS layout is fixed here for every partition.
        b_scale_tmem0 = tlx.local_alloc(
            (HEAD_DIM, 4),
            tl.uint8,
            1,
            tlx.storage_kind.tmem,
            reuse=tmem_alias,
            layout=_TWO_CTA_RHS_SCALES,
        )
        b_scale_tmem1 = tlx.local_alloc(
            (HEAD_DIM, 4),
            tl.uint8,
            1,
            tlx.storage_kind.tmem,
            reuse=tmem_alias,
            layout=_TWO_CTA_RHS_SCALES,
        )
        # 4-column slots for the B scales (their two-CTA layout counts 2).
        b_pad0 = tlx.local_alloc(
            (HEAD_DIM, 4),
            tl.uint8,
            1,
            tlx.storage_kind.tmem,
            reuse=tmem_alias,
            layout=_DEFAULT_SCALES,
        )
        b_pad1 = tlx.local_alloc(
            (HEAD_DIM, 4),
            tl.uint8,
            1,
            tlx.storage_kind.tmem,
            reuse=tmem_alias,
            layout=_DEFAULT_SCALES,
        )
        dp_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tl.float32, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        dsT_tmem = tlx.local_alloc((BLOCK_N1, BLOCK_M1), p_dtype, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        k_scale_qk_tmem = tlx.local_alloc((BLOCK_N1, 4), tl.uint8, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        q_scale_qk_tmem = tlx.local_alloc((BLOCK_M1, 4), tl.uint8, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        ds_scale_dk_tmem = tlx.local_alloc((BLOCK_N1, 4), tl.uint8, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        q_scale_dk_tmem = tlx.local_alloc((HEAD_DIM, 4), tl.uint8, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        dv_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        dk_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem, reuse=tmem_alias)
        tmem_alias.set_buffer_overlap(
            tlx.reuse_group(
                tlx.reuse_group(
                    qk_tiles,
                    tlx.reuse_group(
                        tlx.reuse_group(dq_tiles, dq_phys, group_type=tlx.reuse_group_type.shared),
                        p_tiles,
                        v_scale_tmem,
                        do_scale_dp_tmem,
                        p_scale_tmem,
                        do_scale_dv_tmem,
                        a_scale_tmem0,
                        a_scale_tmem1,
                        tlx.reuse_group(b_scale_tmem0, b_pad0, group_type=tlx.reuse_group_type.shared),
                        tlx.reuse_group(b_scale_tmem1, b_pad1, group_type=tlx.reuse_group_type.shared),
                        group_type=tlx.reuse_group_type.distinct,
                    ),
                    group_type=tlx.reuse_group_type.shared,
                ),
                tlx.reuse_group(
                    dp_tiles,
                    tlx.reuse_group(
                        dsT_tmem,
                        k_scale_qk_tmem,
                        q_scale_qk_tmem,
                        ds_scale_dk_tmem,
                        q_scale_dk_tmem,
                        group_type=tlx.reuse_group_type.distinct,
                    ),
                    group_type=tlx.reuse_group_type.shared,
                ),
                dv_tiles,
                dk_tiles,
                group_type=tlx.reuse_group_type.distinct,
            ))

        # ===== Barriers =====
        # Cross-CTA publication barriers: arrived by both CTAs of the pair on the
        # leader's copy (remote_cta_rank=0), waited by the leader's MMA warp. The
        # compute arrives come from two lanes of every warp (2 lanes x 8 warps x
        # 2 CTAs = 32 arrivals), matching the reference's num_warps=1 warp
        # barriers.
        qk_empties = tlx.alloc_warp_barrier(num_barriers=1, num_warps=1)
        dpr_bars = tlx.alloc_warp_barrier(num_barriers=1, num_warps=1)
        dsT_fulls = tlx.alloc_warp_barrier(num_barriers=1, num_warps=1)
        kdq_ready = tlx.alloc_warp_barrier(num_barriers=1, num_warps=1)
        # Both MMA warps arrive on both copies, then wait on their local copy.
        pub_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=2)
        # The relay's ds_fulls arrival (dS data + A scales of both CTAs staged).
        ds_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=2)
        dq_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=2)
        dv_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=2)
        dk_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=2)
        # Signalled by the collaborative MMAs (both CTAs' copies).
        qk_fulls = tlx.alloc_barriers(num_barriers=1)
        dp_fulls = tlx.alloc_barriers(num_barriers=1)
        dq_fulls = tlx.alloc_barriers(num_barriers=1)
        dv_fulls = tlx.alloc_barriers(num_barriers=1)
        dk_fulls = tlx.alloc_barriers(num_barriers=1)
        ds_empties = tlx.alloc_barriers(num_barriers=1)
        q_empties = tlx.alloc_barriers(num_barriers=2)
        do_empties = tlx.alloc_barriers(num_barriers=2)
        k_empties = tlx.alloc_barriers(num_barriers=1)
        kt_empties = tlx.alloc_barriers(num_barriers=1)
        v_empties = tlx.alloc_barriers(num_barriers=1)
        # Two-CTA TMA loads (leader's copy counts both CTAs' bytes).
        kv_fulls = tlx.alloc_barriers(num_barriers=1)
        v_fulls = tlx.alloc_barriers(num_barriers=1)
        kt_fulls = tlx.alloc_barriers(num_barriers=1)
        q_fulls = tlx.alloc_barriers(num_barriers=2)
        do_fulls = tlx.alloc_barriers(num_barriers=2)
        # Local.
        ktsc_fulls = tlx.alloc_barriers(num_barriers=1)
        ds_peer_fulls = tlx.alloc_barriers(num_barriers=1)
        xchg_ready = tlx.alloc_warp_barrier(num_barriers=1, num_warps=8)
        m_fulls = tlx.alloc_barriers(num_barriers=M_STAGE)
        m_empties = tlx.alloc_warp_barrier(num_barriers=M_STAGE, num_warps=8)
        d_fulls = tlx.alloc_barriers(num_barriers=D_STAGE)
        d_empties = tlx.alloc_warp_barrier(num_barriers=D_STAGE, num_warps=8)

        # ===== SMEM =====
        k_smem = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k), 1)
        v_smem = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_v), 1)
        # Both key tiles' full K rows (MMA 5 B reads 128 rows x full head dim per
        # tile; the two-CTA MMA distributes columns across the pair).
        kt_smem = tlx.local_alloc((BLOCK_N1, HD_SPLIT), tlx.dtype_of(desc_kt), 2)
        # Full 128-query blocks in both layouts: Q for MMA 1 B (128 rows) and
        # MMA 4 B (128 rows x 64 head-dim cols); dO likewise for MMA 2/3 B.
        qt_smem = tlx.local_alloc((M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), 2)
        qh_smem = tlx.local_alloc((BLOCK_M1, HD_SPLIT), tlx.dtype_of(desc_q), 2)
        dot_smem = tlx.local_alloc((M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_do), 2)
        doh_smem = tlx.local_alloc((BLOCK_M1, HD_SPLIT), tlx.dtype_of(desc_do), 2)
        # dS^T halves owned by this CTA's queries (MMA 5 A, one per key tile).
        ds_x = tlx.local_alloc((BLOCK_N1, M_SPLIT), p_dtype, 2)
        ds_xchg = tlx.local_alloc((BLOCK_N1, M_SPLIT), p_dtype, 1)
        k_scale_smem = tlx.local_alloc((1, 1, 1, 2, 256), tl.uint8, 1)
        v_scale_smem = tlx.local_alloc((1, 1, 1, 2, 256), tl.uint8, 1)
        kt_scale_smem = tlx.local_alloc((1, 1, 1, 2, 256), tl.uint8, 4)
        q_scale_smem = tlx.local_alloc((1, 1, 1, 2, 256), tl.uint8, 2)
        q_dk_scale_smem = tlx.local_alloc((1, 1, 1, 2, 256), tl.uint8, 2)
        do_scale_smem = tlx.local_alloc((1, 1, 1, 2, 256), tl.uint8, 2)
        do_scale_dv_smem = tlx.local_alloc((1, 1, 1, 2, 256), tl.uint8, 2)
        sM_tiles = tlx.local_alloc((BLOCK_M1, ), tl.float32, M_STAGE)
        sD_tiles = tlx.local_alloc((BLOCK_M1, ), tl.float32, D_STAGE)
        p_scale_smem = tlx.local_alloc((1, 1, 32, 4, 4), tl.uint8, 1)
        ds_scale_smem = tlx.local_alloc((1, 1, 32, 4, 4), tl.uint8, 1)
        ds_scale_dq_smem = tlx.local_alloc((1, 1, 32, 4, 4), tl.uint8, 2)
        ds_scale_xchg = tlx.local_alloc((1, 1, 32, 4, 4), tl.uint8, 1)
        dkv_store_buf = tlx.local_alloc((BLOCK_N1, HEAD_DIM // 2), tl.bfloat16, 1)
        # One [128, 32] staging buffer per 32-column half of the dQ drain.
        dq_store_buf = tlx.local_alloc((2 * M_SPLIT, HEAD_DIM // 4), tlx.dtype_of(desc_dq), 2)

        tlx.fence_mbarrier_init_cluster()

        with tlx.async_tasks(exclusive=True, less_reg_mma=True):
            # ----- Compute warps: P and dS -----
            with tlx.async_task("default"):
                P_FIXED_E8M0: tl.constexpr = 119
                p_scale_const = tl.full([1, 1, 32, 4, 4], P_FIXED_E8M0, dtype=tl.uint8)
                tlx.local_store(tlx.local_view(p_scale_smem, 0), p_scale_const)
                E8M0_NEUTRAL_C: tl.constexpr = 127
                neutral = tl.full([1, 1, 32, 4, 4], E8M0_NEUTRAL_C, dtype=tl.uint8)
                tlx.local_store(tlx.local_view(ds_scale_dq_smem, 0), neutral)
                tlx.local_store(tlx.local_view(ds_scale_dq_smem, 1), neutral)
                tlx.local_store(tlx.local_view(ds_scale_xchg, 0), neutral)
                tlx.fence("async_shared")
                blk_idx = 0
                tc = 0
                num_steps = tl.cdiv(N_CTX, BLOCK_M1).to(tl.int32)
                # A contiguous range of pairs per cluster.
                pair = (cluster_id * total_pairs) // num_clusters
                pair_end = ((cluster_id + 1) * total_pairs) // num_clusters
                has_more = pair < pair_end
                while has_more:
                    blk_idx = _bwd_mxf8_compute_inner_loop(
                        blk_idx,
                        qk_scale,
                        qk_tiles,
                        qk_fulls,
                        qk_empties,
                        p_tiles,
                        p_scale_smem,
                        dp_tiles,
                        dp_fulls,
                        ds_scale_smem,
                        ds_scale_dq_smem,
                        sM_tiles,
                        sD_tiles,
                        ds_empties,
                        m_fulls,
                        m_empties,
                        d_fulls,
                        d_empties,
                        M_STAGE,
                        D_STAGE,
                        BLOCK_M1,
                        BLOCK_N1,
                        VEC_SIZE,
                        p_dtype,
                        num_steps=num_steps,
                        USE_2CTA=True,
                        tc=tc,
                        rank=rank,
                        dpr_bars=dpr_bars,
                        dsT_tmem=dsT_tmem,
                        dsT_fulls=dsT_fulls,
                        ds_x=ds_x,
                        ds_xchg=ds_xchg,
                        ds_scale_xchg=ds_scale_xchg,
                        xchg_ready=xchg_ready,
                        kt_scale_smem=kt_scale_smem,
                        b_scale_tmem0=b_scale_tmem0,
                        b_scale_tmem1=b_scale_tmem1,
                        kdq_ready=kdq_ready,
                        ktsc_fulls=ktsc_fulls,
                    )
                    tc += 1
                    pair += 1
                    has_more = pair < pair_end

            # ----- Reduction warps: dQ TMA reduce-add, dV / dK epilogue -----
            # Per pair: dQ of every query block, then the dV epilogue (its TMEM is
            # needed by the next pair's first dV product), then the dK epilogue.
            with tlx.async_task(num_warps=4, registers=176):
                blk_idx = 0
                tc = 0
                pair = (cluster_id * total_pairs) // num_clusters
                pair_end = ((cluster_id + 1) * total_pairs) // num_clusters
                has_more = pair < pair_end
                while has_more:
                    off_seq_h, off_z, off_h, blk, start_n = _bwd_mxf8_pair_coords_2cta(
                        pair, rank, n_tile_num, H, N_CTX, BLOCK_N1)
                    _, pers_phase = get_bufidx_phase(tc, 1)
                    for jb in range(0, num_steps):
                        _bwd_mxf8_drain_dq_2cta(
                            dq_phys,
                            dq_fulls,
                            dq_empties,
                            dq_store_buf,
                            desc_dq,
                            blk_idx,
                            (jb * BLOCK_M1 + rank * M_SPLIT).to(tl.int32),
                            off_seq_h.to(tl.int32),
                            sm_scale,
                            M_SPLIT,
                            HEAD_DIM,
                        )
                        blk_idx += 1
                    # dV epilogue: this CTA's key tile.
                    tlx.barrier_wait(dv_fulls[0], pers_phase)
                    for slice_id in tl.static_range(2):
                        dv_slice = tlx.local_slice(
                            dv_tiles[0],
                            [0, slice_id * HD_SPLIT],
                            [BLOCK_N1, HD_SPLIT],
                        )
                        dv = tlx.local_load(dv_slice)
                        if slice_id == 1:
                            tlx.barrier_arrive(dv_empties[0], 1, remote_cta_rank=0)
                        tlx.async_descriptor_store_wait(0)
                        tlx.local_store(dkv_store_buf[0], dv.to(tl.bfloat16))
                        tlx.async_descriptor_store(
                            desc_dv,
                            dkv_store_buf[0],
                            [
                                off_z,
                                off_h,
                                start_n,
                                slice_id * HD_SPLIT,
                            ],
                        )
                    # dK epilogue: this CTA's key tile (sm_scale applied).
                    tlx.barrier_wait(dk_fulls[0], pers_phase)
                    for slice_id in tl.static_range(2):
                        dk_slice = tlx.local_slice(
                            dk_tiles[0],
                            [0, slice_id * HD_SPLIT],
                            [BLOCK_N1, HD_SPLIT],
                        )
                        dk = tlx.local_load(dk_slice)
                        if slice_id == 1:
                            tlx.barrier_arrive(dk_empties[0], 1, remote_cta_rank=0)
                        dk = dk * sm_scale
                        tlx.async_descriptor_store_wait(0)
                        tlx.local_store(dkv_store_buf[0], dk.to(tl.bfloat16))
                        tlx.async_descriptor_store(
                            desc_dk,
                            dkv_store_buf[0],
                            [
                                off_z,
                                off_h,
                                start_n,
                                slice_id * HD_SPLIT,
                            ],
                        )
                    tc += 1
                    pair += 1
                    has_more = pair < pair_end
                tlx.async_descriptor_store_wait(0)

            # ----- MMA warp: the leader issues every collaborative product -----
            # One flat loop over query blocks of all pairs: block g issues MMA 1
            # (qk_g), MMA 4 (dK, block g - 1), MMA 2 (dP_g), MMA 5 (dQ, block
            # g - 1) and MMA 3 (dV += P_g dO_g). The products of a pair's last
            # block follow the next pair's first qk, so qk never waits for a dQ
            # drain.
            with tlx.async_task(num_warps=1, registers=48):
                if is_leader:
                    blk_idx = 0
                    tc = 0
                    kdq_cnt = 0
                    pub_cnt = 0
                    num_steps = tl.cdiv(N_CTX, BLOCK_M1).to(tl.int32)
                    pair = (cluster_id * total_pairs) // num_clusters
                    pair_end = ((cluster_id + 1) * total_pairs) // num_clusters
                    has_more = pair < pair_end
                    while has_more:
                        kv_buf, kv_phase = get_bufidx_phase(tc, 1)
                        _, tile_phase = get_bufidx_phase(tc, 1)
                        for j in range(0, num_steps):
                            blk_idx, kdq_cnt, pub_cnt = _bwd_mxf8_mma_dots_2cta(
                                blk_idx=blk_idx,
                                q_fulls=q_fulls,
                                q_empties=q_empties,
                                do_fulls=do_fulls,
                                do_empties=do_empties,
                                j=j,
                                num_steps=num_steps,
                                kv_buf=kv_buf,
                                kv_phase=kv_phase,
                                tile_phase=tile_phase,
                                tc=tc,
                                kdq_cnt=kdq_cnt,
                                pub_cnt=pub_cnt,
                                k_smem=k_smem,
                                v_smem=v_smem,
                                k_scale_smem=k_scale_smem,
                                v_scale_smem=v_scale_smem,
                                q_scale_smem=q_scale_smem,
                                q_dk_scale_smem=q_dk_scale_smem,
                                do_scale_smem=do_scale_smem,
                                do_scale_dv_smem=do_scale_dv_smem,
                                p_scale_smem=p_scale_smem,
                                ds_scale_smem=ds_scale_smem,
                                qt_smem=qt_smem,
                                qh_smem=qh_smem,
                                dot_smem=dot_smem,
                                doh_smem=doh_smem,
                                kt_smem=kt_smem,
                                qk_tiles=qk_tiles,
                                qk_fulls=qk_fulls,
                                qk_empties=qk_empties,
                                p_tiles=p_tiles,
                                dp_tiles=dp_tiles,
                                dp_fulls=dp_fulls,
                                dpr_bars=dpr_bars,
                                dv_tiles=dv_tiles,
                                dv_fulls=dv_fulls,
                                dv_empties=dv_empties,
                                dk_tiles=dk_tiles,
                                dk_fulls=dk_fulls,
                                dk_empties=dk_empties,
                                dq_tiles=dq_tiles,
                                dq_fulls=dq_fulls,
                                dq_empties=dq_empties,
                                dsT_tmem=dsT_tmem,
                                dsT_fulls=dsT_fulls,
                                ds_x=ds_x,
                                ds_fulls=ds_fulls,
                                ds_empties=ds_empties,
                                ds_scale_dq_smem=ds_scale_dq_smem,
                                a_scale_tmem0=a_scale_tmem0,
                                a_scale_tmem1=a_scale_tmem1,
                                b_scale_tmem0=b_scale_tmem0,
                                b_scale_tmem1=b_scale_tmem1,
                                pub_bars=pub_bars,
                                k_scale_qk_tmem=k_scale_qk_tmem,
                                q_scale_qk_tmem=q_scale_qk_tmem,
                                v_scale_tmem=v_scale_tmem,
                                do_scale_dp_tmem=do_scale_dp_tmem,
                                p_scale_tmem=p_scale_tmem,
                                do_scale_dv_tmem=do_scale_dv_tmem,
                                ds_scale_dk_tmem=ds_scale_dk_tmem,
                                q_scale_dk_tmem=q_scale_dk_tmem,
                                kv_fulls=kv_fulls,
                                v_fulls=v_fulls,
                                kt_fulls=kt_fulls,
                                kt_empties=kt_empties,
                                k_empties=k_empties,
                                v_empties=v_empties,
                                kdq_ready=kdq_ready,
                                K_FP8_FORMAT=K_FP8_FORMAT,
                                Q_FP8_FORMAT=Q_FP8_FORMAT,
                                V_FP8_FORMAT=V_FP8_FORMAT,
                                DO_FP8_FORMAT=DO_FP8_FORMAT,
                                P_FP8_FORMAT=P_FP8_FORMAT,
                                DS_FP8_FORMAT=DS_FP8_FORMAT,
                            )
                        tc += 1
                        pair += 1
                        has_more = pair < pair_end
                    pub_cnt = _bwd_mxf8_mma_tail_2cta(
                        blk_idx=blk_idx,
                        kdq_ready=kdq_ready,
                        tc=tc,
                        num_steps=num_steps,
                        kdq_cnt=kdq_cnt,
                        pub_cnt=pub_cnt,
                        qh_smem=qh_smem,
                        kt_smem=kt_smem,
                        qk_tiles=qk_tiles,
                        qk_fulls=qk_fulls,
                        qk_empties=qk_empties,
                        p_tiles=p_tiles,
                        dp_tiles=dp_tiles,
                        dp_fulls=dp_fulls,
                        dpr_bars=dpr_bars,
                        dv_tiles=dv_tiles,
                        dv_fulls=dv_fulls,
                        dv_empties=dv_empties,
                        dk_tiles=dk_tiles,
                        dk_fulls=dk_fulls,
                        dk_empties=dk_empties,
                        dq_tiles=dq_tiles,
                        dq_fulls=dq_fulls,
                        dq_empties=dq_empties,
                        dsT_tmem=dsT_tmem,
                        dsT_fulls=dsT_fulls,
                        ds_x=ds_x,
                        ds_fulls=ds_fulls,
                        ds_empties=ds_empties,
                        q_dk_scale_smem=q_dk_scale_smem,
                        ds_scale_smem=ds_scale_smem,
                        ds_scale_dq_smem=ds_scale_dq_smem,
                        a_scale_tmem0=a_scale_tmem0,
                        a_scale_tmem1=a_scale_tmem1,
                        b_scale_tmem0=b_scale_tmem0,
                        b_scale_tmem1=b_scale_tmem1,
                        pub_bars=pub_bars,
                        q_empties=q_empties,
                        ds_scale_dk_tmem=ds_scale_dk_tmem,
                        q_scale_dk_tmem=q_scale_dk_tmem,
                        DS_FP8_FORMAT=DS_FP8_FORMAT,
                        Q_FP8_FORMAT=Q_FP8_FORMAT,
                        K_FP8_FORMAT=K_FP8_FORMAT,
                    )
                else:
                    pair_start = (cluster_id * total_pairs) // num_clusters
                    pair_end = ((cluster_id + 1) * total_pairs) // num_clusters
                    for block in range((pair_end - pair_start) * num_steps):
                        _bwd_mxf8_publish_m64_2cta(pub_bars, 0)
                        _bwd_mxf8_publish_m64_2cta(pub_bars, 1)

            # ----- Load warp: TMA loads of FP8 data + scales -----
            with tlx.async_task(num_warps=1, registers=48):
                blk_idx = 0
                tc = 0
                num_steps = tl.cdiv(N_CTX, BLOCK_M1).to(tl.int32)
                kt_block = tl.minimum(num_steps - 1, 1)
                # A contiguous range of pairs per cluster.
                pair = (cluster_id * total_pairs) // num_clusters
                pair_end = ((cluster_id + 1) * total_pairs) // num_clusters
                has_more = pair < pair_end
                while has_more:
                    off_seq_h, off_z, off_h, blk, start_n = _bwd_mxf8_pair_coords_2cta(
                        pair, rank, n_tile_num, H, N_CTX, BLOCK_N1)
                    sf_off_seq_h = off_seq_h.to(tl.int32)
                    kv_scale_n = (blk * REP_N).to(tl.int32)
                    kv_buf, kv_phase = get_bufidx_phase(tc, 1)
                    _, tile_phase = get_bufidx_phase(tc, 1)
                    kslot = tc % 2
                    # K (MMA 1 A, this CTA's key tile) with its K-block exponents.
                    tlx.barrier_wait(k_empties[kv_buf], kv_phase ^ 1)
                    if is_leader:
                        tlx.barrier_expect_bytes(kv_fulls[kv_buf], 2 * (BLOCK_N1 * HEAD_DIM + SCALE_BYTES))
                    tlx.async_descriptor_load(
                        desc_k,
                        k_smem[kv_buf],
                        [off_z, off_h, start_n, 0],
                        kv_fulls[kv_buf],
                        two_ctas=True,
                    )
                    tlx.async_descriptor_load(
                        desc_k_scale,
                        k_scale_smem[kv_buf],
                        [sf_off_seq_h, kv_scale_n, 0, 0, 0],
                        kv_fulls[kv_buf],
                        two_ctas=True,
                    )
                    # V (MMA 2 A, this CTA's key tile) with its exponents.
                    tlx.barrier_wait(v_empties[0], tile_phase ^ 1)
                    if is_leader:
                        tlx.barrier_expect_bytes(v_fulls[0], 2 * (BLOCK_N1 * HEAD_DIM + SCALE_BYTES))
                    tlx.async_descriptor_load(
                        desc_v,
                        v_smem[0],
                        [off_z, off_h, start_n, 0],
                        v_fulls[0],
                        two_ctas=True,
                    )
                    tlx.async_descriptor_load(
                        desc_v_scale,
                        v_scale_smem[0],
                        [sf_off_seq_h, kv_scale_n, 0, 0, 0],
                        v_fulls[0],
                        two_ctas=True,
                    )
                    for j in range(0, num_steps):
                        curr_m = j * BLOCK_M1
                        q_buf, q_phase = get_bufidx_phase(blk_idx, 2)
                        do_buf, do_phase = get_bufidx_phase(blk_idx, 2)
                        q_scale_m = (curr_m // 128 * REP_M).to(tl.int32)
                        # Q: full 128-query block in both layouts (MMA 1 B reads
                        # all 128 Q rows against this CTA's 128 K rows; MMA 4 B
                        # reads all 128 Q rows x this CTA's 64 head-dim cols),
                        # with both exponent sets: 2 data + 2 scale TMAs.
                        tlx.barrier_wait(q_empties[q_buf], q_phase ^ 1)
                        if is_leader:
                            tlx.barrier_expect_bytes(
                                q_fulls[q_buf],
                                2 * (M_SPLIT * HEAD_DIM + BLOCK_M1 * HD_SPLIT + 2 * SCALE_BYTES),
                            )
                        # Only the leader issues TMA; the .cta_group::2 modifier
                        # delivers both CTAs' halves to both SMEM copies.
                        tlx.async_descriptor_load(
                            desc_q,
                            qt_smem[q_buf],
                            [off_z, off_h, curr_m + rank * M_SPLIT, 0],
                            q_fulls[q_buf],
                            two_ctas=True,
                        )
                        tlx.async_descriptor_load(
                            desc_qh,
                            qh_smem[q_buf],
                            [off_z, off_h, curr_m, rank * HD_SPLIT],
                            q_fulls[q_buf],
                            two_ctas=True,
                        )
                        tlx.async_descriptor_load(
                            desc_q_scale,
                            q_scale_smem[q_buf],
                            [sf_off_seq_h, q_scale_m, 0, 0, 0],
                            q_fulls[q_buf],
                            two_ctas=True,
                        )
                        tlx.async_descriptor_load(
                            desc_q_dk_scale,
                            q_dk_scale_smem[q_buf],
                            [sf_off_seq_h, 0, q_scale_m, 0, 0],
                            q_fulls[q_buf],
                            two_ctas=True,
                        )
                        # M / Delta for this query block.
                        m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
                        tlx.barrier_wait(m_empties[m_buf_id], m_phase ^ 1)
                        tlx.barrier_expect_bytes(m_fulls[m_buf_id], 4 * BLOCK_M1)
                        tlx.async_descriptor_load(
                            desc_m,
                            sM_tiles[m_buf_id],
                            [(off_seq_h * N_CTX + curr_m).to(tl.int32)],
                            m_fulls[m_buf_id],
                        )
                        d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
                        tlx.barrier_wait(d_empties[d_buf_id], d_phase ^ 1)
                        tlx.barrier_expect_bytes(d_fulls[d_buf_id], 4 * BLOCK_M1)
                        tlx.async_descriptor_load(
                            desc_delta,
                            sD_tiles[d_buf_id],
                            [(off_seq_h * N_CTX + curr_m).to(tl.int32)],
                            d_fulls[d_buf_id],
                        )
                        # dO: full 128-query block in both layouts (MMA 2 B reads
                        # all 128 dO rows against this CTA's 128 V rows; MMA 3 B
                        # reads all 128 dO rows x this CTA's 64 head-dim cols),
                        # with both exponent sets: 2 data + 2 scale TMAs.
                        tlx.barrier_wait(do_empties[do_buf], do_phase ^ 1)
                        if is_leader:
                            tlx.barrier_expect_bytes(
                                do_fulls[do_buf],
                                2 * (M_SPLIT * HEAD_DIM + BLOCK_M1 * HD_SPLIT + 2 * SCALE_BYTES),
                            )
                        tlx.async_descriptor_load(
                            desc_do,
                            dot_smem[do_buf],
                            [off_z, off_h, curr_m + rank * M_SPLIT, 0],
                            do_fulls[do_buf],
                            two_ctas=True,
                        )
                        tlx.async_descriptor_load(
                            desc_doh,
                            doh_smem[do_buf],
                            [off_z, off_h, curr_m, rank * HD_SPLIT],
                            do_fulls[do_buf],
                            two_ctas=True,
                        )
                        tlx.async_descriptor_load(
                            desc_do_scale,
                            do_scale_smem[do_buf],
                            [sf_off_seq_h, q_scale_m, 0, 0, 0],
                            do_fulls[do_buf],
                            two_ctas=True,
                        )
                        tlx.async_descriptor_load(
                            desc_do_dv_scale,
                            do_scale_dv_smem[do_buf],
                            [sf_off_seq_h, 0, q_scale_m, 0, 0],
                            do_fulls[do_buf],
                            two_ctas=True,
                        )
                        if j == kt_block:
                            # Both key tiles' full K rows (MMA 5 B reads 128 K
                            # rows x full 128 head-dim cols per tile; each CTA's
                            # half comes from the distributed SMEM view): 2 data
                            # TMAs. Free once the previous pair's last dQ product,
                            # issued after this pair's first qk, has read it; first
                            # read by this pair's second block, so loaded after
                            # that block's Q and dO. The tiles' M-block exponents
                            # go to the pair-parity slot (local, non-collective).
                            tlx.barrier_wait(kt_empties[0], tile_phase ^ 1)
                            if is_leader:
                                tlx.barrier_expect_bytes(kt_fulls[0], 2 * (2 * BLOCK_N1 * HD_SPLIT))
                            blk_t0 = blk - rank
                            blk_t1 = blk - rank + 1
                            for t in tl.static_range(2):
                                if t == 0:
                                    blk_t = blk_t0
                                else:
                                    blk_t = blk_t1
                                tlx.async_descriptor_load(
                                    desc_kt,
                                    kt_smem[t],
                                    [off_z, off_h, blk_t * BLOCK_N1, rank * HD_SPLIT],
                                    kt_fulls[0],
                                    two_ctas=True,
                                )
                            tlx.barrier_expect_bytes(ktsc_fulls[0], 2 * SCALE_BYTES)
                            for t in tl.static_range(2):
                                if t == 0:
                                    blk_t = blk_t0
                                else:
                                    blk_t = blk_t1
                                tlx.async_descriptor_load(
                                    desc_k_dq_scale,
                                    kt_scale_smem[2 * kslot + t],
                                    [
                                        sf_off_seq_h,
                                        0,
                                        (blk_t * REP_N).to(tl.int32),
                                        0,
                                        0,
                                    ],
                                    ktsc_fulls[0],
                                )
                        blk_idx += 1
                    tc += 1
                    pair += 1
                    has_more = pair < pair_end

            # ----- Relay: DSMEM exchange of dS halves, publish ds_x -----
            with tlx.async_task(num_warps=1, registers=24):
                blk_idx = 0
                num_steps = tl.cdiv(N_CTX, BLOCK_M1).to(tl.int32)
                # A contiguous range of pairs per cluster.
                pair = (cluster_id * total_pairs) // num_clusters
                pair_end = ((cluster_id + 1) * total_pairs) // num_clusters
                has_more = pair < pair_end
                while has_more:
                    for _j in range(0, num_steps):
                        _, ds_phase = get_bufidx_phase(blk_idx, 1)
                        # The peer's query half of this CTA's dS goes over DSMEM;
                        # the half arriving from the peer completes ds_x here.
                        tlx.barrier_wait(xchg_ready[0], ds_phase)
                        tlx.barrier_expect_bytes(ds_peer_fulls[0], BLOCK_N1 * M_SPLIT + SCALE_BYTES)
                        tlx.async_remote_shmem_copy(
                            dst=tlx.local_view(ds_x, rank),
                            src=tlx.local_view(ds_xchg, 0),
                            remote_cta_rank=1 - rank,
                            barrier=ds_peer_fulls[0],
                        )
                        tlx.async_remote_shmem_copy(
                            dst=tlx.local_view(ds_scale_dq_smem, rank),
                            src=tlx.local_view(ds_scale_xchg, 0),
                            remote_cta_rank=1 - rank,
                            barrier=ds_peer_fulls[0],
                        )
                        tlx.barrier_wait(ds_peer_fulls[0], ds_phase)
                        tlx.fence("async_shared")
                        tlx.barrier_arrive(ds_fulls[0], 1, remote_cta_rank=0)
                        blk_idx += 1
                    pair += 1
                    has_more = pair < pair_end
    else:
        tl.static_assert(HEAD_DIM == 128)
        tl.static_assert(BLOCK_N1 == 128)
        tl.static_assert(BLOCK_M1 == 128)
        tl.static_assert(HEAD_DIM % DQ_REDUCE_NCOL == 0)
        NUM_BUFFERS_TMEM: tl.constexpr = 1
        tl.static_assert(NUM_BUFFERS_TMEM == 1)

        VEC_SIZE: tl.constexpr = 32
        REP_M: tl.constexpr = triton.cdiv(BLOCK_M1, 128)
        REP_N: tl.constexpr = triton.cdiv(triton.cdiv(BLOCK_N1, VEC_SIZE), 4)
        REP_HEAD: tl.constexpr = triton.cdiv(triton.cdiv(HEAD_DIM, VEC_SIZE), 4)

        Q_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
        K_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
        DO_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_do))
        SCALE_BYTES: tl.constexpr = REP_M * REP_HEAD * 2 * 256
        SCALE_TMEM_COLS: tl.constexpr = SCALE_BYTES // BLOCK_M1

        Q_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_q))
        K_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_k))
        # dV-only launches omit desc_v; V shares Q's FP8 payload format.
        V_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_q))
        DO_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_do))
        p_dtype = tlx.dtype_of(desc_q)
        P_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(p_dtype)
        DS_FP8_FORMAT: tl.constexpr = P_FP8_FORMAT

        qk_scale = sm_scale * 1.44269504  # sm_scale / ln(2) for exp2

        # Tile decomposition:
        #   tile_idx -> (off_z, off_h, pid)
        #   pid = N-block index within (z, h)
        # Each N-tile sweeps query (M) blocks from start_m to N_CTX. For non-causal
        # (STAGE == 1) start_m is 0 (full sweep); for causal (STAGE == 3) start_m is
        # the diagonal (pid * BLOCK_N1). num_steps is therefore computed per-tile
        # inside each partition once pid is known.
        n_tile_num = N_CTX // BLOCK_N1
        prog_id = tl.program_id(0)
        num_progs = tl.num_programs(0)
        total_tiles = n_tile_num * Z * H

        tiles_per_sm = total_tiles // num_progs
        extra_tiles = total_tiles % num_progs
        if prog_id < extra_tiles:
            tiles_per_sm += 1
        if STAGE == 3:
            tile_idx_start = prog_id * (total_tiles // num_progs) + min(prog_id, extra_tiles)
            tile_idx_step = 1
        else:
            tl.static_assert(STAGE == 1)
            tile_idx_start = prog_id
            tile_idx_step = num_progs

        DS_NUM_SUBS: tl.constexpr = 2

        # ===== TMEM allocations =====
        # Single-region accumulator alias (qk/dp/p/dq overlap). Lifetime correctness
        # enforced by barriers - each user must finish before next writer enters.
        tmem_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
        qk_tiles = tlx.local_alloc(
            (BLOCK_N1, BLOCK_M1),
            tl.float32,
            NUM_BUFFERS_TMEM,
            tlx.storage_kind.tmem,
            reuse=tmem_storage_alias,
        )
        p_tiles = None
        if NEED_DV:
            p_tiles = tlx.local_alloc(
                (BLOCK_N1, BLOCK_M1),
                p_dtype,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        dv_tiles = None
        if NEED_DV:
            dv_tiles = tlx.local_alloc(
                (BLOCK_N1, HEAD_DIM),
                tl.float32,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        dp_tiles = None
        if NEED_DQ or NEED_DK:
            dp_tiles = tlx.local_alloc(
                (BLOCK_N1, BLOCK_M1),
                tl.float32,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        dq_tiles = None
        if NEED_DQ:
            dq_tiles = tlx.local_alloc(
                (BLOCK_M1, HEAD_DIM),
                tl.float32,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        dk_tiles = None
        if NEED_DK:
            dk_tiles = tlx.local_alloc(
                (BLOCK_N1, HEAD_DIM),
                tl.float32,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        ###### Scales #######

        # Prologue Scales:
        # Allocate separate prologue tiles because dq is unused at this stage.
        # This simplifies the scale check
        k_scale_tmem_prologue = tlx.local_alloc(
            (BLOCK_N1, SCALE_TMEM_COLS),
            tl.uint8,
            NUM_BUFFERS_TMEM,
            tlx.storage_kind.tmem,
            reuse=tmem_storage_alias,
        )
        q_scale_tmem_prologue = tlx.local_alloc(
            (BLOCK_M1, SCALE_TMEM_COLS),
            tl.uint8,
            NUM_BUFFERS_TMEM,
            tlx.storage_kind.tmem,
            reuse=tmem_storage_alias,
        )
        v_scale_tmem_prologue = None
        if NEED_DQ or NEED_DK:
            v_scale_tmem_prologue = tlx.local_alloc(
                (BLOCK_N1, SCALE_TMEM_COLS),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        do_scale_dp_tmem_prologue = None
        if NEED_DQ or NEED_DK:
            do_scale_dp_tmem_prologue = tlx.local_alloc(
                (BLOCK_N1, SCALE_TMEM_COLS),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        do_scale_dv_tmem_prologue = None
        if NEED_DV:
            do_scale_dv_tmem_prologue = tlx.local_alloc(
                (HEAD_DIM, SCALE_TMEM_COLS),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        p_scale_tmem_prologue = None
        if NEED_DV:
            p_scale_tmem_prologue = tlx.local_alloc(
                (BLOCK_N1, BLOCK_M1 // VEC_SIZE),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        # Body Scales
        # These are the scales used in the steady state.
        p_scale_tmem = None
        if NEED_DV:
            p_scale_tmem = tlx.local_alloc(
                (BLOCK_N1, BLOCK_M1 // VEC_SIZE),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        ds_scale_dk_tmem = None
        if NEED_DK:
            ds_scale_dk_tmem = tlx.local_alloc(
                (BLOCK_N1, BLOCK_M1 // VEC_SIZE),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        ds_scale_dq_tmem = None
        if NEED_DQ:
            ds_scale_dq_tmem = tlx.local_alloc(
                (BLOCK_N1, BLOCK_M1 // VEC_SIZE),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        k_scale_qk_tmem = tlx.local_alloc(
            (BLOCK_N1, SCALE_TMEM_COLS),
            tl.uint8,
            NUM_BUFFERS_TMEM,
            tlx.storage_kind.tmem,
            reuse=tmem_storage_alias,
        )
        k_scale_dq_tmem = None
        if NEED_DQ:
            k_scale_dq_tmem = tlx.local_alloc(
                (BLOCK_N1, SCALE_TMEM_COLS),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        v_scale_tmem = None
        if NEED_DQ or NEED_DK:
            v_scale_tmem = tlx.local_alloc(
                (BLOCK_N1, SCALE_TMEM_COLS),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        do_scale_dp_tmem = None
        if NEED_DQ or NEED_DK:
            do_scale_dp_tmem = tlx.local_alloc(
                (BLOCK_M1, SCALE_TMEM_COLS),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        do_scale_dv_tmem = None
        if NEED_DV:
            do_scale_dv_tmem = tlx.local_alloc(
                (BLOCK_M1, SCALE_TMEM_COLS),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        q_scale_qk_tmem = tlx.local_alloc(
            (BLOCK_M1, SCALE_TMEM_COLS),
            tl.uint8,
            NUM_BUFFERS_TMEM,
            tlx.storage_kind.tmem,
            reuse=tmem_storage_alias,
        )
        q_scale_dk_tmem = None
        if NEED_DK:
            q_scale_dk_tmem = tlx.local_alloc(
                (BLOCK_M1, SCALE_TMEM_COLS),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=tmem_storage_alias,
            )
        # Preserve the alias lifetimes when gradient MMAs are absent.
        rg1_items = ()
        if NEED_DV:
            rg1_items += (p_tiles, p_scale_tmem, do_scale_dv_tmem)
        if NEED_DQ or NEED_DK:
            rg1_items += (v_scale_tmem, do_scale_dp_tmem)
        if NEED_DQ:
            rg1_items += (ds_scale_dq_tmem, k_scale_dq_tmem)
        if NEED_DK:
            rg1_items += (ds_scale_dk_tmem, q_scale_dk_tmem)
        rg1: tl.constexpr = tlx.reuse_group(
            qk_tiles,
            tlx.reuse_group(*rg1_items, group_type=tlx.reuse_group_type.distinct),
            group_type=tlx.reuse_group_type.shared,
        )
        rg3: tl.constexpr = tlx.reuse_group(
            tlx.reuse_group(k_scale_qk_tmem, q_scale_qk_tmem, group_type=tlx.reuse_group_type.distinct),
            *((dp_tiles, ) if NEED_DQ or NEED_DK else ()),
            *((dq_tiles, ) if NEED_DQ else ()),
            group_type=tlx.reuse_group_type.shared,
        )
        prologue_items = (k_scale_tmem_prologue, q_scale_tmem_prologue)
        if NEED_DQ or NEED_DK:
            prologue_items += (v_scale_tmem_prologue, do_scale_dp_tmem_prologue)
        if NEED_DV:
            prologue_items += (do_scale_dv_tmem_prologue, p_scale_tmem_prologue)
        rg4_prologue: tl.constexpr = tlx.reuse_group(*prologue_items, group_type=tlx.reuse_group_type.distinct)
        rg4: tl.constexpr = (tlx.reuse_group(dk_tiles, rg4_prologue, group_type=tlx.reuse_group_type.shared)
                             if NEED_DK else rg4_prologue)
        groups: tl.constexpr = (rg1, dv_tiles, rg3, rg4) if NEED_DV else (rg1, rg3, rg4)
        tmem_storage_alias.set_buffer_overlap(tlx.reuse_group(*groups, group_type=tlx.reuse_group_type.distinct))

        # ===== TMEM barriers =====
        qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
        qk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
        dp_fulls = None
        if NEED_DQ or NEED_DK:
            dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
        dp_empties = None
        if NEED_DQ or NEED_DK:
            dp_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
        dq_fulls = None
        if NEED_DQ:
            dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
        dq_empties = None
        if NEED_DQ:
            dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
        dv_fulls = None
        if NEED_DV:
            dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
        dk_fulls = None
        if NEED_DK:
            dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
        dv_empties = None
        if NEED_DV:
            dv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
        dk_empties = None
        if NEED_DK:
            dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

        # ===== SMEM allocations =====
        k_smem = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
        k_dq_smem = None
        if NEED_DQ:
            k_dq_smem = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
        v_smem = None
        if NEED_DQ or NEED_DK:
            v_smem = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_KV)
        q_smem = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
        q_dk_smem = None
        if NEED_DK:
            q_dk_smem = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
        do_smem = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_do), NUM_BUFFERS_DO)
        # dK consumes dS^T while dQ consumes dS. MXFP8 quantization depends on the
        # reduction axis, so we keep separate internal encodings for the two GEMMs.
        ds_tiles_smem = None
        if NEED_DQ or NEED_DK:
            ds_tiles_smem = tlx.local_alloc((BLOCK_N1, BLOCK_M1), p_dtype, NUM_BUFFERS_DS)
        # SMEM storage spots for dS scales to enable
        # async transfers from SMEM to TMEM.
        # (1, REP_M, REP_HEAD, 2, 256)
        ds_scale_smem = None
        if NEED_DK:
            ds_scale_smem = tlx.local_alloc(
                (REP_N, REP_M, 32, 4, 4),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.smem,
            )
        p_scale_smem = None
        if NEED_DV:
            p_scale_smem = tlx.local_alloc(
                (REP_N, REP_M, 32, 4, 4),
                tl.uint8,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.smem,
            )
        ds_scale_dq_smem = None
        if NEED_DQ:
            ds_scale_dq_smem = tlx.local_alloc(
                (REP_M, REP_N, 32, 4, 4),
                tl.uint8,
                NUM_BUFFERS_TMEM * DS_NUM_SUBS,
                tlx.storage_kind.smem,
            )

        k_scale_smem = tlx.local_alloc((1, REP_N, REP_HEAD, 2, 256), tl.uint8, NUM_BUFFERS_KV)
        k_scale_dq_smem = None
        if NEED_DQ:
            k_scale_dq_smem = tlx.local_alloc((1, REP_HEAD, REP_N, 2, 256), tl.uint8, NUM_BUFFERS_KV)
        v_scale_smem = None
        if NEED_DQ or NEED_DK:
            v_scale_smem = tlx.local_alloc((1, REP_N, REP_HEAD, 2, 256), tl.uint8, NUM_BUFFERS_KV)
        q_scale_smem = tlx.local_alloc((1, REP_M, REP_HEAD, 2, 256), tl.uint8, NUM_BUFFERS_Q)
        q_dk_scale_smem = None
        if NEED_DK:
            q_dk_scale_smem = tlx.local_alloc((1, REP_HEAD, REP_M, 2, 256), tl.uint8, NUM_BUFFERS_Q)
        do_scale_smem = None
        if NEED_DQ or NEED_DK:
            do_scale_smem = tlx.local_alloc((1, REP_M, REP_HEAD, 2, 256), tl.uint8, NUM_BUFFERS_DO)
        do_scale_dv_smem = None
        if NEED_DV:
            do_scale_dv_smem = tlx.local_alloc((1, REP_HEAD, REP_M, 2, 256), tl.uint8, NUM_BUFFERS_DO)

        slice_size_alloc: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
        # TODO: Expose. This is set to 1 because its not on the critical path.
        NUM_DKV_STORE_BUFFERS: tl.constexpr = 1
        dkv_store_buf = None
        if NEED_DK or NEED_DV:
            dkv_store_buf = tlx.local_alloc((BLOCK_N1, slice_size_alloc), tl.bfloat16, NUM_DKV_STORE_BUFFERS)
        DQ_REDUCE_ITERS: tl.constexpr = HEAD_DIM // DQ_REDUCE_NCOL
        DQ_REDUCE_STAGES: tl.constexpr = DQ_REDUCE_ITERS
        dq_store_buf = None
        if NEED_DQ:
            dq_store_buf = tlx.local_alloc((BLOCK_M1, DQ_REDUCE_NCOL), tlx.dtype_of(desc_dq), DQ_REDUCE_STAGES)
        sM_tiles = tlx.local_alloc((BLOCK_M1, ), tl.float32, M_STAGE)
        sD_tiles = None
        if NEED_DQ or NEED_DK:
            sD_tiles = tlx.local_alloc((BLOCK_M1, ), tl.float32, D_STAGE)

        # ===== SMEM barriers =====
        k_dq_fulls = None
        if NEED_DQ:
            k_dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
        q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
        q_dk_fulls = None
        if NEED_DK:
            q_dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
        q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
        q_dk_empties = None
        if NEED_DK:
            q_dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
        do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
        do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
        do_dv_fulls = None
        if NEED_DV:
            do_dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
        ds_fulls = None
        if NEED_DQ or NEED_DK:
            ds_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)
        ds_empties = None
        if NEED_DQ or NEED_DK:
            ds_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)
        p_fulls = None
        if NEED_DV:
            p_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
        k_dq_empties = None
        if NEED_DQ:
            k_dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
        m_fulls = tlx.alloc_barriers(num_barriers=M_STAGE)
        m_empties = tlx.alloc_barriers(num_barriers=M_STAGE)
        d_fulls = None
        if NEED_DQ or NEED_DK:
            d_fulls = tlx.alloc_barriers(num_barriers=D_STAGE)
        d_empties = None
        if NEED_DQ or NEED_DK:
            d_empties = tlx.alloc_barriers(num_barriers=D_STAGE)

        # ===== Warp-specialized async tasks =====
        with tlx.async_tasks(exclusive=True, mbarrier_try_wait_suspend_ns=50000, less_reg_mma=True):
            # ----- Compute warp: softmax recompute + P/dS quantization -----
            # Default task -- its warp count comes from autotune num_warps (= 4).
            with tlx.async_task("default"):
                if NEED_DV:
                    # Pre-fill P scale SMEM once (constant E8M0=119 for all tiles).
                    P_FIXED_E8M0: tl.constexpr = 119
                    p_scale_const = tl.full([REP_N, REP_M, 32, 4, 4], P_FIXED_E8M0, dtype=tl.uint8)
                    tlx.local_store(p_scale_smem[0], p_scale_const)

                tile_idx = tile_idx_start
                blk_idx = 0
                for _i in range(tiles_per_sm):
                    _, persistent_tmem_phase = get_bufidx_phase(_i, NUM_BUFFERS_TMEM)
                    off_seq_h, off_z, off_h, pid, start_n, base_q, _start_m, _num_steps = (_get_mxf8_bwd_tile_info(
                        tile_idx,
                        n_tile_num,
                        H,
                        N_CTX,
                        BLOCK_M1,
                        BLOCK_N1,
                        STAGE,
                    ))

                    # Automatically false: causal blocks with curr_m < start_n are
                    # omitted by _get_start_m_bwd().
                    if STAGE & 1:
                        lo, hi = _get_mxf8_unfused_bwd_loop_bounds(start_n, N_CTX, BLOCK_N1, STAGE=4 - STAGE)
                        # Causal: sometimes-true diagonal section, kept in the
                        # prologue location. Non-causal: full automatically-true
                        # sweep.
                        blk_idx = _bwd_mxf8_compute_inner_loop(
                            blk_idx=blk_idx,
                            qk_scale=qk_scale,
                            qk_tiles=qk_tiles,
                            qk_fulls=qk_fulls,
                            qk_empties=qk_empties,
                            p_tiles=p_tiles,
                            p_scale_smem=p_scale_smem,
                            p_fulls=p_fulls,
                            dp_tiles=dp_tiles,
                            dp_fulls=dp_fulls,
                            dp_empties=dp_empties,
                            ds_tiles_smem=ds_tiles_smem,
                            ds_scale_smem=ds_scale_smem,
                            ds_scale_dq_smem=ds_scale_dq_smem,
                            sM_tiles=sM_tiles,
                            sD_tiles=sD_tiles,
                            ds_fulls=ds_fulls,
                            ds_empties=ds_empties,
                            m_fulls=m_fulls,
                            m_empties=m_empties,
                            d_fulls=d_fulls,
                            d_empties=d_empties,
                            NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                            NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                            M_STAGE=M_STAGE,
                            D_STAGE=D_STAGE,
                            BLOCK_M1=BLOCK_M1,
                            BLOCK_N1=BLOCK_N1,
                            VEC_SIZE=VEC_SIZE,
                            p_dtype=p_dtype,
                            REP_N=REP_N,
                            REP_M=REP_M,
                            DS_NUM_SUBS=DS_NUM_SUBS,
                            NEED_DQ=NEED_DQ,
                            NEED_DK=NEED_DK,
                            NEED_DV=NEED_DV,
                            start_m=lo,
                            end_m=hi,
                            start_n=start_n,
                            MASK=STAGE == 3,
                        )

                    if STAGE & 2:
                        lo, hi = _get_mxf8_unfused_bwd_loop_bounds(start_n, N_CTX, BLOCK_N1, STAGE=2)
                        # Automatically true: all remaining active M blocks are
                        # strictly below the diagonal and skip the causal mask.
                        blk_idx = _bwd_mxf8_compute_inner_loop(
                            blk_idx=blk_idx,
                            qk_scale=qk_scale,
                            qk_tiles=qk_tiles,
                            qk_fulls=qk_fulls,
                            qk_empties=qk_empties,
                            p_tiles=p_tiles,
                            p_scale_smem=p_scale_smem,
                            p_fulls=p_fulls,
                            dp_tiles=dp_tiles,
                            dp_fulls=dp_fulls,
                            dp_empties=dp_empties,
                            ds_tiles_smem=ds_tiles_smem,
                            ds_scale_smem=ds_scale_smem,
                            ds_scale_dq_smem=ds_scale_dq_smem,
                            sM_tiles=sM_tiles,
                            sD_tiles=sD_tiles,
                            ds_fulls=ds_fulls,
                            ds_empties=ds_empties,
                            m_fulls=m_fulls,
                            m_empties=m_empties,
                            d_fulls=d_fulls,
                            d_empties=d_empties,
                            NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                            NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                            M_STAGE=M_STAGE,
                            D_STAGE=D_STAGE,
                            BLOCK_M1=BLOCK_M1,
                            BLOCK_N1=BLOCK_N1,
                            VEC_SIZE=VEC_SIZE,
                            p_dtype=p_dtype,
                            REP_N=REP_N,
                            REP_M=REP_M,
                            DS_NUM_SUBS=DS_NUM_SUBS,
                            NEED_DQ=NEED_DQ,
                            NEED_DK=NEED_DK,
                            NEED_DV=NEED_DV,
                            start_m=lo,
                            end_m=hi,
                            start_n=start_n,
                            MASK=False,
                        )

                    # Epilogue: dK / dV TMA store
                    kv_buf_id, kv_phase = get_bufidx_phase(_i, NUM_BUFFERS_KV)

                    slice_size: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
                    if NEED_DV:
                        tlx.barrier_wait(dv_fulls[0], persistent_tmem_phase)
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            dv_slice = tlx.local_slice(
                                dv_tiles[0],
                                [0, slice_id * slice_size],
                                [BLOCK_N1, slice_size],
                            )
                            dv = tlx.local_load(dv_slice)
                            if slice_id == (EPILOGUE_SUBTILE - 1):
                                tlx.barrier_arrive(dv_empties[0])
                            tlx.async_descriptor_store_wait(0)
                            tlx.local_store(dkv_store_buf[0], dv.to(tl.bfloat16))
                            tlx.async_descriptor_store(
                                desc_dv,
                                dkv_store_buf[0],
                                [
                                    off_z,
                                    off_h,
                                    start_n,
                                    slice_id * slice_size,
                                ],
                            )

                    if NEED_DK:
                        tlx.barrier_wait(dk_fulls[0], persistent_tmem_phase)
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            dk_slice = tlx.local_slice(
                                dk_tiles[0],
                                [0, slice_id * slice_size],
                                [BLOCK_N1, slice_size],
                            )
                            dk = tlx.local_load(dk_slice)
                            if slice_id == (EPILOGUE_SUBTILE - 1):
                                tlx.barrier_arrive(dk_empties[0])
                            dk *= sm_scale
                            tlx.async_descriptor_store_wait(0)
                            tlx.local_store(dkv_store_buf[0], dk.to(tl.bfloat16))
                            tlx.async_descriptor_store(
                                desc_dk,
                                dkv_store_buf[0],
                                [
                                    off_z,
                                    off_h,
                                    start_n,
                                    slice_id * slice_size,
                                ],
                            )
                    tile_idx += tile_idx_step
                tlx.async_descriptor_store_wait(0)

            # ----- Reduction warp: TMA atomic-reduce-add of dQ to GMEM -----
            if NEED_DQ:
                with tlx.async_task(num_warps=4, registers=152):
                    tile_idx = tile_idx_start
                    blk_idx = 0
                    for _i in range(tiles_per_sm):
                        off_seq_h, off_z, off_h, pid, start_n, base_q, start_m, num_steps = (_get_mxf8_bwd_tile_info(
                            tile_idx,
                            n_tile_num,
                            H,
                            N_CTX,
                            BLOCK_M1,
                            BLOCK_N1,
                            STAGE,
                        ))
                        curr_m = start_m
                        for _ in range(num_steps):
                            _, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
                            tlx.barrier_wait(dq_fulls[0], tmem_phase)
                            # Drain the whole dQ accumulator before waiting on any prior
                            # reduce-add, so the dP MMA that reuses this TMEM is not held
                            # behind the TMA reduction queue.
                            dq_all = tlx.local_load(dq_tiles[0])
                            tlx.barrier_arrive(dq_empties[0])
                            dq_all = _mul_f32x2(dq_all, sm_scale)
                            dq_slices = _split_n_2D(dq_all, DQ_REDUCE_ITERS)
                            for slice_id in tl.static_range(DQ_REDUCE_ITERS):
                                dq_smem_idx = slice_id % DQ_REDUCE_STAGES
                                tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                                tlx.local_store(
                                    dq_store_buf[dq_smem_idx],
                                    dq_slices[slice_id].to(tlx.dtype_of(desc_dq)),
                                )
                                tlx.async_descriptor_store(
                                    desc_dq,
                                    dq_store_buf[dq_smem_idx],
                                    [
                                        off_z,
                                        off_h,
                                        curr_m,
                                        slice_id * DQ_REDUCE_NCOL,
                                    ],
                                    eviction_policy="evict_last",
                                    store_reduce="add",
                                )

                            curr_m += BLOCK_M1
                            blk_idx += 1
                        tile_idx += tile_idx_step
                    tlx.async_descriptor_store_wait(0)

            # ----- MMA warp: 5 blockscaled GEMMs per M-block -----
            with tlx.async_task(num_warps=1, registers=80):
                tile_idx = tile_idx_start
                blk_idx = 0
                for _i in range(tiles_per_sm):
                    kv_buf_id, kv_phase = get_bufidx_phase(_i, NUM_BUFFERS_KV)
                    off_seq_h, off_z, off_h, pid, start_n, base_q, start_m, num_steps = (_get_mxf8_bwd_tile_info(
                        tile_idx,
                        n_tile_num,
                        H,
                        N_CTX,
                        BLOCK_M1,
                        BLOCK_N1,
                        STAGE,
                    ))
                    blk_idx = _bwd_mxf8_mma_dots_1cta(
                        blk_idx=blk_idx,
                        _i=_i,
                        num_steps=num_steps,
                        kv_buf_id=kv_buf_id,
                        kv_phase=kv_phase,
                        k_smem=k_smem,
                        k_dq_smem=k_dq_smem,
                        v_smem=v_smem,
                        q_smem=q_smem,
                        q_dk_smem=q_dk_smem,
                        do_smem=do_smem,
                        qk_tiles=qk_tiles,
                        qk_fulls=qk_fulls,
                        qk_empties=qk_empties,
                        p_tiles=p_tiles,
                        p_fulls=p_fulls,
                        dp_tiles=dp_tiles,
                        dp_fulls=dp_fulls,
                        dp_empties=dp_empties,
                        dv_tiles=dv_tiles,
                        dv_fulls=dv_fulls,
                        dv_empties=dv_empties,
                        dk_tiles=dk_tiles,
                        dk_fulls=dk_fulls,
                        dk_empties=dk_empties,
                        dq_tiles=dq_tiles,
                        dq_fulls=dq_fulls,
                        dq_empties=dq_empties,
                        ds_tiles_smem=ds_tiles_smem,
                        ds_fulls=ds_fulls,
                        ds_empties=ds_empties,
                        q_fulls=q_fulls,
                        q_empties=q_empties,
                        q_dk_fulls=q_dk_fulls,
                        q_dk_empties=q_dk_empties,
                        do_fulls=do_fulls,
                        do_empties=do_empties,
                        do_dv_fulls=do_dv_fulls,
                        k_dq_fulls=k_dq_fulls,
                        k_dq_empties=k_dq_empties,
                        k_scale_smem=k_scale_smem,
                        k_scale_dq_smem=k_scale_dq_smem,
                        v_scale_smem=v_scale_smem,
                        q_scale_smem=q_scale_smem,
                        q_dk_scale_smem=q_dk_scale_smem,
                        do_scale_smem=do_scale_smem,
                        do_scale_dv_smem=do_scale_dv_smem,
                        p_scale_smem=p_scale_smem,
                        ds_scale_smem=ds_scale_smem,
                        ds_scale_dq_smem=ds_scale_dq_smem,
                        k_scale_tmem_prologue=k_scale_tmem_prologue,
                        q_scale_tmem_prologue=q_scale_tmem_prologue,
                        v_scale_tmem_prologue=v_scale_tmem_prologue,
                        do_scale_dp_tmem_prologue=do_scale_dp_tmem_prologue,
                        do_scale_dv_tmem_prologue=do_scale_dv_tmem_prologue,
                        p_scale_tmem_prologue=p_scale_tmem_prologue,
                        k_scale_qk_tmem=k_scale_qk_tmem,
                        q_scale_qk_tmem=q_scale_qk_tmem,
                        v_scale_tmem=v_scale_tmem,
                        do_scale_dp_tmem=do_scale_dp_tmem,
                        p_scale_tmem=p_scale_tmem,
                        do_scale_dv_tmem=do_scale_dv_tmem,
                        ds_scale_dk_tmem=ds_scale_dk_tmem,
                        q_scale_dk_tmem=q_scale_dk_tmem,
                        ds_scale_dq_tmem=ds_scale_dq_tmem,
                        k_scale_dq_tmem=k_scale_dq_tmem,
                        NUM_BUFFERS_Q=NUM_BUFFERS_Q,
                        NUM_BUFFERS_DO=NUM_BUFFERS_DO,
                        NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                        NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                        K_FP8_FORMAT=K_FP8_FORMAT,
                        Q_FP8_FORMAT=Q_FP8_FORMAT,
                        V_FP8_FORMAT=V_FP8_FORMAT,
                        DO_FP8_FORMAT=DO_FP8_FORMAT,
                        P_FP8_FORMAT=P_FP8_FORMAT,
                        DS_FP8_FORMAT=DS_FP8_FORMAT,
                        NEED_DQ=NEED_DQ,
                        NEED_DK=NEED_DK,
                        NEED_DV=NEED_DV,
                    )
                    tile_idx += tile_idx_step

            # ----- Load warp: TMA loads of FP8 data + scales -----
            with tlx.async_task(num_warps=1, registers=24):
                tile_idx = tile_idx_start
                blk_idx = 0
                for _i in range(tiles_per_sm):
                    blk_idx, tile_idx = _bwd_mxf8_load_1cta(
                        blk_idx=blk_idx,
                        _i=_i,
                        tile_idx=tile_idx,
                        tile_idx_start=tile_idx_start,
                        tile_idx_step=tile_idx_step,
                        tiles_per_sm=tiles_per_sm,
                        n_tile_num=n_tile_num,
                        H=H,
                        N_CTX=N_CTX,
                        BLOCK_M1=BLOCK_M1,
                        BLOCK_N1=BLOCK_N1,
                        STAGE=STAGE,
                        desc_q=desc_q,
                        desc_k=desc_k,
                        desc_v=desc_v,
                        desc_do=desc_do,
                        desc_m=desc_m,
                        desc_delta=desc_delta,
                        desc_q_scale=desc_q_scale,
                        desc_q_dk_scale=desc_q_dk_scale,
                        desc_k_scale=desc_k_scale,
                        desc_k_dq_scale=desc_k_dq_scale,
                        desc_v_scale=desc_v_scale,
                        desc_do_scale=desc_do_scale,
                        desc_do_dv_scale=desc_do_dv_scale,
                        k_smem=k_smem,
                        k_dq_smem=k_dq_smem,
                        v_smem=v_smem,
                        q_smem=q_smem,
                        q_dk_smem=q_dk_smem,
                        do_smem=do_smem,
                        q_scale_smem=q_scale_smem,
                        q_dk_scale_smem=q_dk_scale_smem,
                        k_scale_smem=k_scale_smem,
                        k_scale_dq_smem=k_scale_dq_smem,
                        v_scale_smem=v_scale_smem,
                        do_scale_smem=do_scale_smem,
                        do_scale_dv_smem=do_scale_dv_smem,
                        sM_tiles=sM_tiles,
                        sD_tiles=sD_tiles,
                        q_fulls=q_fulls,
                        q_empties=q_empties,
                        q_dk_fulls=q_dk_fulls,
                        q_dk_empties=q_dk_empties,
                        k_dq_fulls=k_dq_fulls,
                        k_dq_empties=k_dq_empties,
                        do_fulls=do_fulls,
                        do_empties=do_empties,
                        do_dv_fulls=do_dv_fulls,
                        m_fulls=m_fulls,
                        m_empties=m_empties,
                        d_fulls=d_fulls,
                        d_empties=d_empties,
                        Q_BYTES=Q_BYTES,
                        K_BYTES=K_BYTES,
                        DO_BYTES=DO_BYTES,
                        SCALE_BYTES=SCALE_BYTES,
                        NUM_BUFFERS_KV=NUM_BUFFERS_KV,
                        NUM_BUFFERS_Q=NUM_BUFFERS_Q,
                        NUM_BUFFERS_DO=NUM_BUFFERS_DO,
                        M_STAGE=M_STAGE,
                        D_STAGE=D_STAGE,
                        HEAD_DIM=HEAD_DIM,
                        REP_M=REP_M,
                        REP_N=REP_N,
                        NEED_DQ=NEED_DQ,
                        NEED_DK=NEED_DK,
                        NEED_DV=NEED_DV,
                    )
                    tile_idx += tile_idx_step


# ---------------------------------------------------------------------------
# Backward host wrapper
# ---------------------------------------------------------------------------


def attention_bwd(
        do,
        q,
        k,
        v,
        o,
        M,
        q_scale,
        q_dk_scale,
        k_scale,
        k_dq_scale,
        v_scale,
        do_scale,
        do_dv_scale,
        sm_scale,
        do_bf16=None,
        causal=False,
        use_2cta=False,
        *,
        needs_input_grad=(True, True, True),
):
    """MXFP8 attention backward.

    Operates on dense [Z, H, N_CTX, HEAD_DIM] tensors. Q, K, and dO are
    quantized with 32x32 blocks, so each tensor uses one FP8 E4M3 payload in
    both GEMM orientations. Their normal and reduction-axis-swapped E8M0
    scales remain distinct TMA-preshuffled tensors. V keeps its original
    orientation-specific MXFP8 encoding.

    The separate Q/Q_dK, K/K_dQ, and dO/dO_dV descriptors intentionally alias
    the same global payloads while preserving independent pipeline lifetimes:
      - MMA 1 uses Q/K with normal scales.
      - MMA 2 uses dO with normal scales.
      - MMA 3 reuses dO with its reduction-axis-swapped scales.
      - MMA 4 reuses Q with its reduction-axis-swapped scales.
      - MMA 5 reuses K with its reduction-axis-swapped scales.

    Returns requested gradients, with dQ in FP32 and dK/dV in BF16.
    Unrequested gradients are None. Direct callers request all gradients by default.

    Supports causal masking via `causal` (each key block only receives gradient
    from query blocks at or below the diagonal). Assumes N_CTX is a multiple of
    128.

    `use_2cta=True` is an internal launch toggle for A/B comparison only: it
    launches the same `_attn_bwd_ws` JIT with the 2-CTA config (non-causal,
    N_CTX a multiple of 256, HEAD_DIM 128). The default path is unchanged.
    """
    need_dq, need_dk, need_dv = needs_input_grad
    if not (need_dq or need_dk or need_dv):
        return None, None, None
    assert q.shape == k.shape == do.shape, "Q, K, dO must have the same shape"
    if need_dq or need_dk:
        assert v.shape == q.shape, "V must have the same shape as Q"
    Z, H, N_CTX, HEAD_DIM = q.shape
    assert HEAD_DIM == 128, "this kernel only supports HEAD_DIM = 128"
    assert N_CTX % 128 == 0, "N_CTX must be a multiple of 128 (BLOCK_M1)"
    if use_2cta:
        if not all(needs_input_grad):
            raise ValueError("2-CTA MXFP8 backward computes dQ, dK, and dV together")
        if causal:
            raise ValueError("2-CTA MXFP8 backward only supports non-causal attention")
        if N_CTX % 256 != 0:
            raise ValueError("2-CTA MXFP8 backward requires N_CTX a multiple of 256 (paired 128-key tiles)")

    y_dim = Z * H * N_CTX

    # Fuse dQ initialization into the Delta preprocess launch.
    dq = torch.empty(q.shape, device=q.device, dtype=torch.float32) if need_dq else None
    dk = torch.empty(k.shape, device=k.device, dtype=torch.bfloat16) if need_dk else None
    dv = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16) if need_dv else None
    delta = torch.empty_like(M) if need_dq or need_dk else None

    if need_dq or need_dk:
        PRE_BLOCK_M = 32
        preproc_grid = (triton.cdiv(N_CTX, PRE_BLOCK_M), Z * H)
        do_preproc = do_bf16 if do_bf16 is not None else do
        _attn_bwd_mxf8_preprocess[preproc_grid](
            o,
            do_preproc,
            dq,
            delta,
            H,
            N_CTX,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=PRE_BLOCK_M,
            NEED_DQ=need_dq,
        )

    dummy_block = [1, 1, 1, 1]
    dummy_5d = [1, 1, 1, 1, 1]
    desc_shape = [Z, H, N_CTX, HEAD_DIM]
    desc_strides = [H * N_CTX * HEAD_DIM, N_CTX * HEAD_DIM, HEAD_DIM, 1]

    desc_q = TensorDescriptor(q, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=desc_shape, strides=desc_strides,
                              block_shape=dummy_block) if need_dq or need_dk else None
    desc_do = TensorDescriptor(do, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
    if use_2cta:
        desc_qh = TensorDescriptor(q, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
        desc_kt = TensorDescriptor(k, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
        desc_doh = TensorDescriptor(do, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
        # Packed view for the two 64-row dQ reduce-adds.
        desc_dq = TensorDescriptor(
            dq,
            shape=[Z * H, 2, N_CTX, HEAD_DIM // 2],
            strides=[N_CTX * HEAD_DIM, HEAD_DIM // 2, HEAD_DIM, 1],
            block_shape=dummy_block,
        )
    else:
        # The launcher encodes even unused descriptors, so reuse valid ones.
        desc_qh, desc_kt, desc_doh = desc_q, desc_k, desc_do
        desc_dq = TensorDescriptor(dq, shape=desc_shape, strides=desc_strides,
                                   block_shape=dummy_block) if dq is not None else None
    desc_dk = TensorDescriptor(dk, shape=desc_shape, strides=desc_strides,
                               block_shape=dummy_block) if dk is not None else None
    desc_dv = TensorDescriptor(dv, shape=desc_shape, strides=desc_strides,
                               block_shape=dummy_block) if dv is not None else None
    desc_m = TensorDescriptor(M, shape=[y_dim], strides=[1], block_shape=[1])
    desc_delta = TensorDescriptor(delta, shape=[y_dim], strides=[1], block_shape=[1]) if delta is not None else None

    desc_q_scale = TensorDescriptor.from_tensor(q_scale, block_shape=dummy_5d)
    desc_q_dk_scale = TensorDescriptor.from_tensor(q_dk_scale, block_shape=dummy_5d) if need_dk else None
    desc_k_scale = TensorDescriptor.from_tensor(k_scale, block_shape=dummy_5d)
    desc_k_dq_scale = TensorDescriptor.from_tensor(k_dq_scale, block_shape=dummy_5d) if need_dq else None
    desc_v_scale = TensorDescriptor.from_tensor(v_scale, block_shape=dummy_5d) if need_dq or need_dk else None
    desc_do_scale = TensorDescriptor.from_tensor(do_scale, block_shape=dummy_5d) if need_dq or need_dk else None
    desc_do_dv_scale = TensorDescriptor.from_tensor(do_dv_scale, block_shape=dummy_5d) if need_dv else None

    device = q.device

    def alloc_fn(size: int, _align: int, _):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count

    def grid(meta):
        if use_2cta:
            # Grid counts CTAs (hardware divides by ctas_per_cga=2 into clusters).
            # Round the pair count up so every CTA has a cluster partner.
            total_pairs = triton.cdiv(triton.cdiv(N_CTX, meta["BLOCK_N1"]), 2) * Z * H
            n_progs = min(NUM_SMS, total_pairs * 2)
            n_progs = triton.cdiv(n_progs, 2) * 2
        else:
            total_tiles = triton.cdiv(N_CTX, meta["BLOCK_N1"]) * Z * H
            n_progs = min(NUM_SMS, total_tiles)
            if causal:
                n_progs = max(n_progs, triton.cdiv(total_tiles, 2))
        return (
            n_progs,
            1,
            1,
        )

    _attn_bwd_mxf8_ws[grid](
        desc_q,
        desc_qh,
        desc_k,
        desc_kt,
        desc_v,
        desc_do,
        desc_doh,
        desc_dq,
        desc_dk,
        desc_dv,
        sm_scale,
        desc_m,
        desc_delta,
        Z,
        H,
        N_CTX,
        desc_q_scale,
        desc_q_dk_scale,
        desc_k_scale,
        desc_k_dq_scale,
        desc_v_scale,
        desc_do_scale,
        desc_do_dv_scale,
        HEAD_DIM=HEAD_DIM,
        M_STAGE=2,
        D_STAGE=2,
        STAGE=3 if causal else 1,
        REQUESTED_NUM_CTAS=2 if use_2cta else 1,
        NEED_DQ=need_dq,
        NEED_DK=need_dk,
        NEED_DV=need_dv,
    )
    return dq, dk, dv


def swizzled_to_tma_preshuffled(swizzled_scales, M, K, block_size, batch):
    """
    Convert from to_blocked() swizzled format to TMA preshuffled format.

    Args:
        swizzled_scales: Swizzled scales, shape (A * B * C * 512,) or (A, B*C, 32, 16)
        M: Original row dimension of data tensor
        K: Original column dimension of data tensor
        block_size: Quantization block size (32 for MX, 16 for NVFP4)
        A: Batch dimension

    Returns:
        TMA preshuffled tensor of shape (A, B, C, 2, 256)
    """
    scale_rows = M
    scale_cols = K // block_size

    B = (scale_rows + 127) // 128  # ceil(M / 128)
    C = (scale_cols + 3) // 4  # ceil(scale_cols / 4)

    # Reshape: (A * B * C * 512,) -> (A, B, C, 512)
    sf_tiles = swizzled_scales.view(batch, B, C, 512)

    # Split each 512-byte SF tile into two 256-byte halves
    # (A, B, C, 512) -> (A, B, C, 2, 256)
    tma_format = sf_tiles.view(batch, B, C, 2, 256)

    return tma_format


@triton.jit
def _mxfp8_32x32_qdata_dual_scale_kernel(
    input_ptr,
    data_ptr,
    normal_scale_ptr,
    swapped_scale_ptr,
    NEED_NORMAL_SCALE: tl.constexpr = True,
    NEED_SWAPPED_SCALE: tl.constexpr = True,
):
    """Quantize one 32x128 tile and write both hardware scale layouts."""
    pid_m = tl.program_id(0)
    offs_m = pid_m * 32 + tl.arange(0, 32)
    offs_n = tl.arange(0, 128)
    input_block = tl.load(input_ptr + offs_m[:, None] * 128 + offs_n[None, :]).to(tl.float32)
    input_32x32 = tl.reshape(input_block, (1, 32, 4, 32))
    partial_amax = tl.max(tl.abs(input_32x32), axis=1)
    block_amax = tl.max(partial_amax, axis=2)
    scale_u32, quant_scale = _fused_amax_to_e8m0(block_amax, 1.0 / 448.0)
    block_scale = scale_u32.to(tl.uint8)
    data = tl.reshape(
        (input_32x32 * quant_scale[:, None, :, None]).to(tl.float8e4nv),
        (32, 128),
    )
    tl.store(data_ptr + offs_m[:, None] * 128 + offs_n[None, :], data)

    if NEED_NORMAL_SCALE:
        # Normal orientation: (M, 4) scales in blocked [M/128, 1, 32, 16].
        row_normal = offs_m[:, None]
        col_normal = tl.arange(0, 4)[None, :]
        row_in_128 = row_normal % 128
        normal_offset = ((row_normal // 128) * 32 + row_in_128 % 32) * 16 + (row_in_128 // 32 * 4 + col_normal)
        normal_scale = tl.reshape(tl.broadcast_to(block_scale[:, None, :], (1, 32, 4)), (32, 4))
        tl.store(normal_scale_ptr + normal_offset, normal_scale)

    if NEED_SWAPPED_SCALE:
        # Swapped orientation: expand each block scale over its 32 transposed rows.
        block_scale = tl.reshape(block_scale, (4, ))
        swapped_scale = tl.reshape(tl.broadcast_to(block_scale[:, None], (4, 32)), (128, ))
        row_swapped = tl.arange(0, 128)
        col_swapped = pid_m
        row_in_128 = row_swapped % 128
        swapped_offset = ((col_swapped // 4) * 32 + row_in_128 % 32) * 16 + (row_in_128 // 32 * 4 + col_swapped % 4)
        tl.store(swapped_scale_ptr + swapped_offset, swapped_scale)


def _quantize_mxfp8_32x32_operand(ref, *, need_normal_scale=True, need_swapped_scale=True):
    """Quantize once and emit the requested GEMM scale layouts."""
    Z, H, N_CTX, HEAD_DIM = ref.shape
    M = Z * H * N_CTX
    if HEAD_DIM != 128 or M % 128 != 0:
        raise RuntimeError("32x32 MXFP8 quantization requires HEAD_DIM=128 and rows divisible by 128")
    flat = ref.reshape(M, HEAD_DIM).contiguous()

    data = torch.empty_like(flat, dtype=torch.float8_e4m3fn)
    normal_scale = torch.empty(
        (M // 128, 1, 32, 16), dtype=torch.uint8, device=ref.device) if need_normal_scale else None
    swapped_scale = torch.empty(
        (1, M // 128, 32, 16), dtype=torch.uint8, device=ref.device) if need_swapped_scale else None
    _mxfp8_32x32_qdata_dual_scale_kernel[(M // 32, )](
        flat,
        data,
        normal_scale,
        swapped_scale,
        NEED_NORMAL_SCALE=need_normal_scale,
        NEED_SWAPPED_SCALE=need_swapped_scale,
        num_warps=1,
    )
    return (
        data.reshape_as(ref),
        swizzled_to_tma_preshuffled(normal_scale.view(torch.float8_e8m0fnu), N_CTX, HEAD_DIM, 32, Z *
                                    H) if need_normal_scale else None,
        swizzled_to_tma_preshuffled(swapped_scale.view(torch.float8_e8m0fnu), HEAD_DIM, N_CTX, 32, Z *
                                    H) if need_swapped_scale else None,
    )


def _quantize_mxfp8_operand(ref, transpose_for_reduction=False):
    """Quantize a BF16 operand to E4M3 data and TMA-preshuffled E8M0 scales."""
    if MXTensor is None or ScaleCalculationMode is None:
        raise RuntimeError("tlx.ops.flash_attn_mxfp8 requires torchao for MXFP8 quantization")
    Z, H, N_CTX, HEAD_DIM = ref.shape
    flat = ref.reshape(Z * H * N_CTX, HEAD_DIM).contiguous()
    quant_input = flat.t().contiguous() if transpose_for_reduction else flat
    mx = MXTensor.to_mx(
        quant_input,
        torch.float8_e4m3fn,
        scaling_mode=ScaleCalculationMode.RCEIL,
        is_swizzled_scales=True,
    )
    if transpose_for_reduction:
        data = mx.qdata.t().reshape_as(ref).contiguous()
        scale = swizzled_to_tma_preshuffled(mx.scale, HEAD_DIM, N_CTX, 32, Z * H)
    else:
        data = mx.qdata.reshape_as(ref).contiguous()
        scale = swizzled_to_tma_preshuffled(mx.scale, N_CTX, HEAD_DIM, 32, Z * H)
    return data, scale


def _forward_prequantized_with_lse(q, k, v, q_scale, k_scale, v_scale, sm_scale, causal, p_expcast=False):
    """Launch MXFP8 forward and return its BF16 output and FP32 logsumexp."""
    Z, H, N_CTX, HEAD_DIM = q.shape
    y_dim = Z * H * N_CTX
    o = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    m_tensor = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    dummy_block = [1, 1]
    dummy_5d = [1, 1, 1, 1, 1]

    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_m = TensorDescriptor(m_tensor, shape=[y_dim], strides=[1], block_shape=[1])
    desc_q_scale = TensorDescriptor.from_tensor(q_scale, block_shape=dummy_5d)
    desc_k_scale = TensorDescriptor.from_tensor(k_scale, block_shape=dummy_5d)
    desc_v_scale = TensorDescriptor.from_tensor(v_scale, block_shape=dummy_5d)

    device = q.device

    def alloc_fn(size: int, _align: int, _):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    def grid(meta):
        total_tiles = triton.cdiv(N_CTX, meta["BLOCK_M"]) * Z * H
        return (total_tiles, 1, 1)

    _attn_fwd_mxf8_ws[grid](
        sm_scale,
        desc_m,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        desc_q_scale,
        desc_k_scale,
        desc_v_scale,
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM,
        STAGE=3 if causal else 1,
        P_EXPCAST=p_expcast,
    )
    return o, m_tensor


class _MXFP8Attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal, p_expcast):
        need_dq, need_dk, _ = ctx.needs_input_grad[:3]
        q_fp8, q_scale, q_scale_dk = _quantize_mxfp8_32x32_operand(q, need_swapped_scale=need_dk)
        k_fp8, k_scale, k_scale_dq = _quantize_mxfp8_32x32_operand(k, need_swapped_scale=need_dq)
        v_fp8, v_scale = _quantize_mxfp8_operand(v, transpose_for_reduction=True)
        o, m_tensor = _forward_prequantized_with_lse(
            q_fp8,
            k_fp8,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            sm_scale,
            causal,
            p_expcast,
        )
        ctx.save_for_backward(
            v if need_dq or need_dk else None,
            q_fp8,
            k_fp8,
            o if need_dq or need_dk else None,
            m_tensor,
            q_scale,
            q_scale_dk,
            k_scale,
            k_scale_dq,
        )
        ctx.input_dtype = q.dtype
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        (
            v,
            q_fp8,
            k_fp8,
            o,
            m_tensor,
            q_scale,
            q_scale_dk,
            k_scale,
            k_scale_dq,
        ) = ctx.saved_tensors
        needs_input_grad = ctx.needs_input_grad[:3]
        need_dq, need_dk, need_dv = needs_input_grad
        if not any(needs_input_grad):
            return None, None, None, None, None, None
        v_bwd, v_scale_bwd = _quantize_mxfp8_operand(v) if need_dq or need_dk else (None, None)
        do_bf16 = do.to(torch.bfloat16).contiguous()
        do_fp8, do_scale, do_scale_dv = _quantize_mxfp8_32x32_operand(
            do_bf16,
            need_normal_scale=need_dq or need_dk,
            need_swapped_scale=need_dv,
        )

        dq, dk, dv = attention_bwd(
            do_fp8,
            q_fp8,
            k_fp8,
            v_bwd,
            o,
            m_tensor,
            q_scale,
            q_scale_dk,
            k_scale,
            k_scale_dq,
            v_scale_bwd,
            do_scale,
            do_scale_dv,
            ctx.sm_scale,
            do_bf16=do_bf16,
            causal=ctx.causal,
            needs_input_grad=needs_input_grad,
        )
        return (
            dq.to(ctx.input_dtype) if need_dq else None,
            dk.to(ctx.input_dtype) if need_dk else None,
            dv.to(ctx.input_dtype) if need_dv else None,
            None,
            None,
            None,
        )


def flash_attn_mxfp8(q, k, v, causal=False, sm_scale=None, *, space="full", p_expcast=False):
    """Differentiable Blackwell MXFP8 attention over BF16 master tensors.

    ``p_expcast`` encodes the forward P with ExpCast-FP8 instead of exp2; see _softmax_inner_loop.
    """
    if space not in ("full", "smoke"):
        raise ValueError(f"space must be 'full' or 'smoke', got {space!r}")
    if sm_scale is None:
        sm_scale = q.shape[-1]**-0.5
    return _MXFP8Attention.apply(q, k, v, sm_scale, causal, p_expcast)
