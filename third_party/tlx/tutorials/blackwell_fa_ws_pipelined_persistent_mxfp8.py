import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.language.extra.tlx.mxfp8_utils import _to_mxfp8_block, _to_mxfp8_block_with_block_amax
from torchao.prototype.mx_formats.mx_tensor import MXTensor, ScaleCalculationMode

DEVICE = triton.runtime.driver.active.get_active_torch_device()


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
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_Q_SCALE_TMEM_BUFFERS": 1,
            "NUM_KV_SCALE_TMEM_BUFFERS": 2,
            "GROUP_SIZE_N": 1,
            "RESCALE_OPT": True,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=_mxf8_host_descriptor_pre_hook,
    ),
]


def prune_configs_by_hdim_mxfp8(configs, named_args, **kwargs):
    return configs


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@triton.jit
def _mul_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _reduce_or(x, y):
    return x | y


@triton.jit
def _fma_f32x2(a, b, c):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc, rd;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mov.b64 rc, { $6, $7 };
            fma.rn.f32x2 rd, ra, rb, rc;
            mov.b64 { $0, $1 }, rd;
        }
        """,
        "=r,=r,r,r,r,r,r,r",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


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
    col_lim_right_s = col_limit_right - s
    col_lim_right_cur = max(col_lim_right_s, 0)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return tl.where(mask_i_bit, qk, -float("inf"))


@triton.jit
def _apply_causal_mask(qk, col_limit_right, BLOCK_N: tl.constexpr):
    # Apply causal mask via a bitmask calculated for each block of 16 elements.
    # This allows the efficient R2P (register to predicate) instruction to be used at the SASS level.
    # Credit to Tri Dao,
    # https://github.com/Dao-AILab/flash-attention/commit/bac1001e4f6caa09d70537495d6746a685a2fa78
    #
    # NOTE: We use map_elementiwse here in order to generate an interleaved sequence of instructions
    # that processes one element of qk at a time. This improves ptxas's resulting SASS.
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    s = offs_n & ~0xF
    i = offs_n & 0xF
    return tl.map_elementwise(_mask_scalar, qk, col_limit_right, s, i)


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
    HEAD_DIM: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    STAGE: tl.constexpr,
    SHARE_SCALE_BUFFERS: tl.constexpr = False,
    RESCALE_OPT: tl.constexpr = False,
):
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2
    NUM_BLOCKS: tl.constexpr = BLOCK_N // VEC_SIZE

    lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)

    for start_n in tl.range(lo, hi, BLOCK_N):
        _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
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
            qk = _apply_causal_mask(qk, col_limit_right, BLOCK_N)

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

        tlx.barrier_wait(tlx.local_view(alpha_empties, cid), qk_phase ^ 1)
        tlx.local_store(tlx.local_view(alpha_tiles, cid), alpha[:, None])
        tlx.barrier_arrive(tlx.local_view(alpha_fulls, cid))

        if RESCALE_OPT:
            m_scaled = m_ij * qk_scale
        else:
            m_scaled = m_ij
        qk = _fma_f32x2(qk, qk_scale, -m_scaled[:, None])
        p_i = tl.math.exp2(qk)

        # Derive block amax from pre-computed block maxes via monotonicity
        # of exp2: max(exp2(x)) == exp2(max(x)), avoiding 128 max(abs())
        # ops per row in the MXFP8 conversion.
        block_amax = tl.math.exp2(block_maxes * qk_scale - m_scaled[:, None])

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

        l_ij = tl.sum(p_i, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        accum_cnt_qk += 1

    return m_i, l_i, accum_cnt_qk


@triton.autotune(
    configs=mxfp8_configs,
    key=["N_CTX", "HEAD_DIM", "STAGE"],
    prune_configs_by={"early_config_prune": prune_configs_by_hdim_mxfp8},
)
@triton.jit
def _attn_fwd_mxf8_ws(sm_scale, M,  #
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

    # Define if we need to do buffer sharing for the scales.
    SHARE_SCALE_BUFFERS: tl.constexpr = (HEAD_DIM == 128) and (BLOCK_N == 128)

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2

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

    # original grid
    #   triton.cdiv(q.shape[2], META["BLOCK_M"]),
    #   q.shape[0] * q.shape[1],
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = Z * H
    num_pid_in_group = num_pid_m * GROUP_SIZE_N
    total_tiles = num_pid_m * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    o_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_o), NUM_MMA_GROUPS)

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
    V_SCALE_TMEM_COLS: tl.constexpr = V_SCALE_BYTES // HEAD_DIM
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
        alpha_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, 1),
            tl.float32,
            NUM_MMA_GROUPS * NUM_BUFFERS_QK,
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
            (HEAD_DIM, V_SCALE_TMEM_COLS),
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
        # Alpha[0]: |BLK_M/2*1*fp32|
        # L[0]:                    |BLK_M/2*1*fp32|
        # M[0]:                                   |BLK_M/2*1*fp32|
        # Q_SCALES[1]:                                           |512*uint8|
        # K_SCALES[1]:                                                     |512*uint8|
        # V_SCALES[0]:                                                               |512*uint8|
        # P[0]:                                                                      |BLK_M/2*BLK_N*fp8|
        # P_SCALES[0]:                                                                         |BLK_M/2*4*uint8|
        qk_storage_alias.set_buffer_overlap(
            tlx.reuse_group(
                qk_tiles,
                tlx.reuse_group(
                    alpha_tiles,
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
        alpha_tiles = tlx.local_alloc(
            (BLOCK_M_SPLIT, 1),
            tl.float32,
            NUM_MMA_GROUPS * NUM_BUFFERS_QK,
            tlx.storage_kind.tmem,
        )
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
        q_scale_tmem = tlx.local_alloc((BLOCK_M_SPLIT, Q_SCALE_TMEM_COLS), tl.uint8, 2 * NUM_Q_SCALE_TMEM_BUFFERS,
                                       tlx.storage_kind.tmem)
        k_scale_tmem = tlx.local_alloc((BLOCK_N, K_SCALE_TMEM_COLS), tl.uint8, NUM_KV_SCALE_TMEM_BUFFERS,
                                       tlx.storage_kind.tmem)
        v_scale_tmem = tlx.local_alloc((HEAD_DIM, V_SCALE_TMEM_COLS), tl.uint8, NUM_KV_SCALE_TMEM_BUFFERS,
                                       tlx.storage_kind.tmem)
        p_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_N), tlx.dtype_of(desc_v), NUM_MMA_GROUPS, tlx.storage_kind.tmem)
        p_scale_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_N // VEC_SIZE), tl.uint8, NUM_MMA_GROUPS,
                                        tlx.storage_kind.tmem)

    acc_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem)

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    p_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    alpha_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    alpha_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    l_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    l_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    with tlx.async_tasks():
        # correction group
        with tlx.async_task("default"):
            accum_cnt = 0
            phase = 0
            for i in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_idx,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )
                for _ in tl.range(lo, hi, BLOCK_N):
                    _, phase = _get_bufidx_phase(accum_cnt, 1)
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
                                acc = tlx.local_load(acc_tiles[cid])
                                acc = _mul_f32x2(acc, alpha_1)
                                tlx.local_store(acc_tiles[cid], acc)
                        else:
                            acc = tlx.local_load(acc_tiles[cid])
                            acc = _mul_f32x2(acc, alpha_1)
                            tlx.local_store(acc_tiles[cid], acc)
                        tlx.barrier_arrive(acc_fulls[cid])
                    accum_cnt += 1

                _, phase = _get_bufidx_phase(i, 1)
                for cid in tl.static_range(0, NUM_MMA_GROUPS):
                    # epilogue
                    tlx.barrier_wait(l_fulls[cid], phase)
                    l = tlx.local_load(l_tiles[cid])
                    m = tlx.local_load(m_tiles[cid])
                    tlx.barrier_arrive(l_empties[cid])
                    if RESCALE_OPT:
                        m = m * sm_scale * 1.44269504
                    m += tl.math.log2(l)
                    offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                    m_ptrs = M + off_hz * N_CTX + offs_m
                    tl.store(m_ptrs, tl.reshape(m, [BLOCK_M_SPLIT]))

                    tlx.barrier_wait(acc_empties[cid], phase)
                    tlx.barrier_wait(o_empties[cid], phase ^ 1)
                    scale = 1 / l
                    acc = tlx.local_load(acc_tiles[cid])
                    acc = _mul_f32x2(acc, scale)
                    acc = acc.to(tlx.dtype_of(desc_o))
                    tlx.local_store(o_tiles[cid], acc)
                    tlx.barrier_arrive(o_fulls[cid])

                tile_idx += num_progs

        # softmax groups
        with tlx.async_task(num_warps=4, registers=168, replicate=NUM_MMA_GROUPS):
            accum_cnt_qk = 0
            for i in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_idx,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )
                # initialize pointer to m and l
                m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
                acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)
                qk_scale = sm_scale
                qk_scale *= 1.44269504  # 1/log(2)

                cid = tlx.async_task_replica_id()
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
                        HEAD_DIM,
                        NUM_MMA_GROUPS,
                        VEC_SIZE,
                        STAGE=4 - STAGE,
                        SHARE_SCALE_BUFFERS=SHARE_SCALE_BUFFERS,
                        RESCALE_OPT=RESCALE_OPT,
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
                        HEAD_DIM,
                        NUM_MMA_GROUPS,
                        VEC_SIZE,
                        STAGE=2,
                        SHARE_SCALE_BUFFERS=SHARE_SCALE_BUFFERS,
                        RESCALE_OPT=RESCALE_OPT,
                    )

                # prepare l_i for the epilog
                _, phase = _get_bufidx_phase(i, 1)
                if not SHARE_SCALE_BUFFERS:
                    # Wait for L to be empty if it has its own buffer.
                    tlx.barrier_wait(l_empties[cid], phase ^ 1)
                tlx.local_store(l_tiles[cid], l_i[:, None])
                tlx.local_store(m_tiles[cid], m_i[:, None])
                tlx.barrier_arrive(l_fulls[cid])
                tile_idx += num_progs

            # mma group
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            accum_cnt_qk = 0

            for j in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_idx,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )

                q_bufIdx, q_phase = _get_bufidx_phase(j, NUM_BUFFERS_Q)
                _, l_phase = _get_bufidx_phase(j, 1)
                if SHARE_SCALE_BUFFERS:
                    # With 2 buffers we always swap index 1/0
                    q0_tmem = 1
                    q1_tmem = 0
                else:
                    q0_tmem = (j % NUM_Q_SCALE_TMEM_BUFFERS) * 2
                    q1_tmem = q0_tmem + 1
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

                # wait for the Q buffer to be populated by the producer
                tlx.barrier_wait(q_fulls[q_bufIdx], q_phase)
                # Explicit SMEM->TMEM scale transfer
                tlx.tmem_copy(q_scale_tiles[0], q_scale_tmem[q0_tmem])

                # wait for the K buffer to be populated by the producer
                tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                k_tile = tlx.local_trans(kv_tiles[k_bufIdx])

                # -- compute q0 @ k ----

                _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
                if SHARE_SCALE_BUFFERS:
                    # Indices based on which value of QK must be live/dead.
                    k0_tmem = 1
                    k1_tmem = 0
                    v0_tmem = 0
                else:
                    # All buffers are the same.
                    kv_scale_tmem_idx = accum_cnt_qk % NUM_KV_SCALE_TMEM_BUFFERS
                    k0_tmem = kv_scale_tmem_idx
                    k1_tmem = kv_scale_tmem_idx
                    v0_tmem = kv_scale_tmem_idx

                # Explicit SMEM->TMEM scale transfer
                tlx.tmem_copy(kv_scale_tiles[k_bufIdx], k_scale_tmem[k0_tmem])

                NAMED_BAR_QK_EMPTY: tl.constexpr = 9
                NUM_THREADS_QK_EMPTY: tl.constexpr = 160

                # Wait for the QK output to be available.
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

                # Explicit SMEM->TMEM scale transfer
                tlx.tmem_copy(q_scale_tiles[1], q_scale_tmem[q1_tmem])
                if SHARE_SCALE_BUFFERS:
                    # K_Scale must be copied to the new buffer
                    tlx.tmem_copy(kv_scale_tiles[k_bufIdx], k_scale_tmem[k1_tmem])

                # Wait for the QK output to be available.
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
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
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
                    _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
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
                    # Use the previous value of the buffer index
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
                tile_idx += num_progs

        # load
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            for i in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_idx,
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
                q_bufIdx, q_phase = _get_bufidx_phase(i, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_bufIdx],
                                         (Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM) + Q_SCALE_BYTES)
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
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
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
                tlx.barrier_expect_bytes(q_fulls[q_bufIdx],
                                         (Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM) + Q_SCALE_BYTES)
                qo_offset_y_split = qo_offset_y + BLOCK_M_SPLIT
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx])

                tlx.async_descriptor_load(
                    desc_q_scale,
                    q_scale_tiles[q_bufIdx],
                    [off_hz, q_scale_m_offset_q1, 0, 0, 0],
                    q_fulls[q_bufIdx],
                )

                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
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
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # wait for the K buffer to be released by the consumer
                    k_empty = tlx.local_view(kv_empties, k_bufIdx)
                    tlx.barrier_wait(k_empty, k_phase ^ 1)
                    # load K + scale
                    k_full = tlx.local_view(kv_fulls, k_bufIdx)
                    k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                    tlx.barrier_expect_bytes(k_full, (K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM) + K_SCALE_BYTES)
                    tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)
                    # 5D TMA offset: [batch_head, n_offset, head_offset, 0, 0]
                    # Compute offset based on relative position within this batch-head's N range
                    # kv_offset_y is absolute, base_offset_y is the start of this batch-head
                    tlx.async_descriptor_load(
                        desc_k_scale,
                        kv_scale_tiles[k_bufIdx],
                        [off_hz, kv_scale_n_offset, 0, 0, 0],
                        k_full,
                    )

                    v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
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

                tile_idx += num_progs

        # epilog group
        with tlx.async_task(num_warps=1, registers=24):
            # initialize offsets
            for i in range(0, tiles_per_sm):
                # initialize offsets
                _, _, _, _, qo_offset_y, _ = _compute_offsets(
                    tile_idx,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )
                _, phase = _get_bufidx_phase(i, 1)
                for cid in tl.static_range(0, NUM_MMA_GROUPS):
                    tlx.barrier_wait(o_fulls[cid], phase)
                    tlx.fence("async_shared")
                    qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                    tlx.async_descriptor_store(desc_o, o_tiles[cid], [qo_offset_y_split, 0])
                    tlx.async_descriptor_store_wait(0)
                    tlx.barrier_arrive(o_empties[cid])

                tile_idx += num_progs


# ===========================================================================
# Backward pass (MXFP8)
#
# Non-causal only. Assumes N_CTX is a multiple of BLOCK_M1 (= 128) and
# BLOCK_N1 (= 128). HEAD_DIM = 128 only. Matches the tile / dtype
# constraints of the forward kernel above.
# ===========================================================================


@triton.jit  # pragma: no cover
def _attn_bwd_preprocess(
    O,
    DO,  #
    Delta,  #
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
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
    tl.store(Delta + base + off_m, delta, mask=off_m < N_CTX)


def _mxf8_bwd_host_descriptor_pre_hook(nargs):
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    HEAD_DIM = nargs["HEAD_DIM"]
    EPILOGUE_SUBTILE = nargs["EPILOGUE_SUBTILE"]
    VEC_SIZE = 32
    REP_M = math.ceil(BLOCK_M1 / 128)
    REP_N = math.ceil(math.ceil(BLOCK_N1 / VEC_SIZE) / 4)
    REP_HEAD = math.ceil(math.ceil(HEAD_DIM / VEC_SIZE) / 4)

    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M1, HEAD_DIM]
    if isinstance(nargs.get("desc_q_dk"), TensorDescriptor):
        nargs["desc_q_dk"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N1, HEAD_DIM]
    if isinstance(nargs.get("desc_k_dq"), TensorDescriptor):
        nargs["desc_k_dq"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_do"].block_shape = [BLOCK_M1, HEAD_DIM]
    if isinstance(nargs.get("desc_do_dv"), TensorDescriptor):
        nargs["desc_do_dv"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_dq"].block_shape = [BLOCK_M1, HEAD_DIM // (EPILOGUE_SUBTILE * 2)]
    if isinstance(nargs.get("desc_dk"), TensorDescriptor):
        nargs["desc_dk"].block_shape = [BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE]
    if isinstance(nargs.get("desc_dv"), TensorDescriptor):
        nargs["desc_dv"].block_shape = [BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE]

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
        nargs["desc_v_scale"].block_shape = [1, REP_N, REP_HEAD, 2, 256]
        nargs["desc_do_scale"].block_shape = [1, REP_M, REP_HEAD, 2, 256]
        if isinstance(nargs.get("desc_do_dv_scale"), TensorDescriptor):
            # MMA 3 consumes dO with the query dimension as the reduction axis,
            # so its scale tensor follows the same swapped convention as V.
            nargs["desc_do_dv_scale"].block_shape = [1, REP_HEAD, REP_M, 2, 256]


# Single-config autotune (matches the JFA backward; the kernel structure
# is highly tuned for this exact shape - see D101699854 OPTIMIZATION_REPORT
# for the perf history that led to num_warps=16 + reg-trim on Reduction /
# Load tasks).
mxfp8_bwd_configs = [
    triton.Config(
        {
            "BLOCK_M1": 128,
            "BLOCK_N1": 128,
            "NUM_BUFFERS_KV": 1,
            "NUM_BUFFERS_Q": 2,
            "NUM_BUFFERS_DO": 1,
            "NUM_BUFFERS_DS": 1,
            "EPILOGUE_SUBTILE": 4,
        },
        num_warps=4,
        num_stages=1,
        pre_hook=_mxf8_bwd_host_descriptor_pre_hook,
    ),
]


@triton.jit
def _softmax_recompute_quantization_iter(
    M_off,
    D_off,
    curr_m,
    blk_idx,
    qk_scale,
    qk_tiles_subtile,
    qk_fulls,
    qk_empties,
    p_tiles,
    p_scale_buf,
    p_fulls,
    p_empties,
    dp_tiles_subtile,
    dp_fulls,
    dp_empties,
    ds_tiles_smem_subtile,
    ds_dq_tiles_smem_subtile,
    ds_scale_smem_subtile,
    ds_scale_dq_smem_subtile,
    ds_fulls,
    ds_empties,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    p_dtype: tl.constexpr,
    REP_N: tl.constexpr,
    REP_M: tl.constexpr,
    DS_NUM_SUBS: tl.constexpr,
):
    DS_M_SUB: tl.constexpr = BLOCK_M1 // DS_NUM_SUBS
    DS_SCALE_BLOCK_SIZE: tl.constexpr = (BLOCK_M1 // VEC_SIZE) // DS_NUM_SUBS
    BWD_NUM_BLOCKS: tl.constexpr = DS_M_SUB // VEC_SIZE
    _, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
    ds_buf_id, ds_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)
    for subtile_id in tl.static_range(DS_NUM_SUBS):
        offs_m = curr_m + tl.arange(subtile_id * DS_M_SUB, (subtile_id + 1) * DS_M_SUB)
        m = tl.load(M_off + offs_m)

        # Read QK from TMEM, apply sm_scale -> P
        if subtile_id == 0:
            tlx.barrier_wait(qk_fulls[0], tmem_phase)
        qkT = tlx.local_load(qk_tiles_subtile[subtile_id])
        # qk_tiles, dp_tiles, dq_tiles all share the same physical
        # TMEM cols 0-127 (qkdp_alias, single-buffered). Force the
        # TMEM read to drain into registers before signaling the
        # MMA partition that the region is empty - otherwise MMA 2
        # (which writes dp_tiles into the same cols) can overwrite
        # qk while this load is still in flight.
        if subtile_id == DS_NUM_SUBS - 1:
            tlx.barrier_arrive(qk_empties[0])

        qkT_scaled = qkT * qk_scale - m[None, :]
        # Clamp to prevent FP32 overflow downstream in P*dP
        qkT_scaled = tl.minimum(qkT_scaled, 20.0)
        pT = tl.math.exp2(qkT_scaled)

        # Block amax for P via monotonicity of exp2
        qkT_reshaped = tl.reshape(qkT_scaled, [BLOCK_N1, BWD_NUM_BLOCKS, VEC_SIZE])
        block_maxes_p = tl.max(qkT_reshaped, 2)
        block_amax_p = tl.math.exp2(block_maxes_p)

        # Quantize P^T -> TMEM (FP8 data + E8M0 scales)
        if subtile_id == 0:
            tlx.barrier_wait(p_empties[0], tmem_phase ^ 1)
        p_fp8, p_scale = _to_mxfp8_block_with_block_amax(
            pT,
            block_amax_p,
            VEC_SIZE,
            p_dtype,
        )
        tlx.local_store(tlx.subslice(p_tiles[0], DS_M_SUB * subtile_id, DS_M_SUB), p_fp8)
        tlx.local_store(tlx.subslice(p_scale_buf[0], DS_SCALE_BLOCK_SIZE * subtile_id, DS_SCALE_BLOCK_SIZE), p_scale)
        if subtile_id == DS_NUM_SUBS - 1:
            tlx.barrier_arrive(p_fulls[0])
        # Finish dS for the previous M-block.
        Di = tl.load(D_off + offs_m)
        if subtile_id == 0:
            tlx.barrier_wait(dp_fulls[0], tmem_phase)
        dpT = tlx.local_load(dp_tiles_subtile[subtile_id])
        if subtile_id == DS_NUM_SUBS - 1:
            tlx.barrier_arrive(dp_empties[0])

        dsT = _mul_f32x2(pT, (dpT - Di[None, :]))
        # NaN sanitization (boundary tiles)
        dsT = tl.where(dsT == dsT, dsT, 0.0)
        # Quantize dS twice: dK consumes dS^T, while dQ consumes dS
        # with the opposite reduction axis and therefore needs a
        # separate blockscaled encoding.
        if subtile_id == 0:
            tlx.barrier_wait(ds_empties[ds_buf_id], ds_phase ^ 1)
        ds_fp8, ds_scale = _to_mxfp8_block(
            dsT,
            VEC_SIZE,
            p_dtype,
        )
        tlx.local_store(tlx.local_view(ds_tiles_smem_subtile, ds_buf_id), ds_fp8)
        ds_scale_packed = ds_scale.reshape([REP_N, 4, 32, REP_M, 4 // DS_NUM_SUBS]).permute(0, 3, 2, 1, 4)
        tlx.local_store(tlx.local_view(ds_scale_smem_subtile, 0), ds_scale_packed)
        ds_dq_fp8, ds_scale_dq = _to_mxfp8_block(
            tl.trans(dsT),
            VEC_SIZE,
            p_dtype,
        )
        tlx.local_store(tlx.local_view(ds_dq_tiles_smem_subtile, ds_buf_id), ds_dq_fp8)
        ds_scale_dq_packed = ds_scale_dq.reshape([REP_M, 4 // DS_NUM_SUBS, 32, REP_N, 4]).permute(0, 3, 2, 1, 4)
        tlx.local_store(tlx.local_view(ds_scale_dq_smem_subtile, 0), ds_scale_dq_packed)
        if subtile_id == DS_NUM_SUBS - 1:
            tlx.barrier_arrive(ds_fulls[ds_buf_id])


@triton.autotune(
    configs=mxfp8_bwd_configs,
    key=["N_CTX", "HEAD_DIM", "H"],
    restore_value=["dQ"],
)
@triton.jit  # pragma: no cover
def _attn_bwd_mxf8_ws(
    desc_q,
    desc_q_dk,
    desc_k,
    desc_k_dq,
    desc_v,
    desc_do,
    desc_do_dv,
    desc_dq,
    desc_dk,
    desc_dv,
    dQ,
    sm_scale,
    M_ptr,
    D_ptr,
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
) -> None:
    tl.static_assert(HEAD_DIM == 128)
    tl.static_assert(BLOCK_N1 == 128)
    tl.static_assert(BLOCK_M1 == 128)
    NUM_BUFFERS_TMEM: tl.constexpr = 1
    tl.static_assert(NUM_BUFFERS_TMEM == 1)

    VEC_SIZE: tl.constexpr = 32
    LN2: tl.constexpr = 0.6931471824645996
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

    # Tile decomposition (dense, non-causal):
    #   tile_idx -> (off_z, off_h, pid)
    #   pid = N-block index within (z, h)
    n_tile_num = N_CTX // BLOCK_N1
    num_steps = N_CTX // BLOCK_M1  # full M sweep per N-tile
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1
    tile_idx = prog_id

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
    qk_tiles_subtile = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1 // DS_NUM_SUBS),
        tl.float32,
        NUM_BUFFERS_TMEM * DS_NUM_SUBS,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    p_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        p_dtype,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    dv_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    dp_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    dp_tiles_subtile = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1 // DS_NUM_SUBS),
        tl.float32,
        NUM_BUFFERS_TMEM * DS_NUM_SUBS,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    dq_tiles = tlx.local_alloc(
        (BLOCK_M1, HEAD_DIM),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    dk_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    ds_tiles_tmem = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM),
        p_dtype,
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
    v_scale_tmem_prologue = tlx.local_alloc(
        (BLOCK_N1, SCALE_TMEM_COLS),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    do_scale_dp_tmem_prologue = tlx.local_alloc(
        (BLOCK_N1, SCALE_TMEM_COLS),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    do_scale_dv_tmem_prologue = tlx.local_alloc(
        (HEAD_DIM, SCALE_TMEM_COLS),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    p_scale_tmem_prologue = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1 // VEC_SIZE),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    # Body Scales
    # These are the scales used in the steady state.
    p_scale_tmem = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1 // VEC_SIZE),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    ds_scale_dk_tmem = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1 // VEC_SIZE),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
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
    k_scale_dq_tmem = tlx.local_alloc(
        (BLOCK_N1, SCALE_TMEM_COLS),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    v_scale_tmem = tlx.local_alloc(
        (BLOCK_N1, SCALE_TMEM_COLS),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    do_scale_dp_tmem = tlx.local_alloc(
        (BLOCK_M1, SCALE_TMEM_COLS),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
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
    q_scale_dk_tmem = tlx.local_alloc(
        (BLOCK_M1, SCALE_TMEM_COLS),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=tmem_storage_alias,
    )
    # Define the reuse strategy.
    #
    # TMEM physical column map (128-col slots for f32 tiles, 4-col slots for
    # uint8 scale buffers). "shared" means items occupy the same physical
    # columns; writing one corrupts the other. Barrier synchronization between
    # the Compute, MMA, and Reduction tasks enforces non-overlapping lifetimes.
    #
    # RG1  cols   0..127  (shared: qk_tiles ↔ inner group)
    #   qk_tiles              cols   0..127   MMA 1 output, read by Compute
    #   p_tiles               cols   0..31    Compute → MMA 3
    #   v_scale_tmem          cols  32..35    MMA 2 scale
    #   do_scale_dp_tmem      cols  36..39    MMA 2 scale
    #   p_scale_tmem          cols  40..43    MMA 3 scale
    #   do_scale_dv_tmem      cols  44..47    MMA 3 scale
    #   ds_scale_dq_tmem      cols  48..51    MMA 5 scale
    #   k_scale_dq_tmem       cols  52..55    MMA 5 scale
    #
    # RG2  cols 128..255  (no overlap)
    #   dv_tiles              cols 128..255   MMA 3 accumulator
    #
    # RG3  cols 256..383  (shared: dp/dq_tiles ↔ inner group)
    #   dp_tiles              cols 256..383   MMA 2 output, read by Compute
    #   dq_tiles              cols 256..383   MMA 5 output, read by Reduction
    #   k_scale_qk_tmem       cols 288..291   MMA 1 scale (body only)
    #   q_scale_qk_tmem       cols 292..295   MMA 1 scale (body only)
    #   ds_scale_dk_tmem      cols 296..299   MMA 4 scale
    #   q_scale_dk_tmem       cols 300..303   MMA 4 scale
    #
    # RG4  cols 384..511  (shared: dk_tiles ↔ prologue scales)
    #   dk_tiles              cols 384..511   MMA 4 accumulator
    #   k_scale_tmem_prologue cols 384..387   MMA 1 scale (prologue only)
    #   q_scale_tmem_prologue cols 388..391   MMA 1 scale (prologue only)
    #   v_scale_tmem_prologue cols 392..395   MMA 2 scale (prologue only)
    #   do_scale_dp_tmem_prol cols 396..399   MMA 2 scale (prologue only)
    #   do_scale_dv_tmem_prol cols 400..403   MMA 3 scale (prologue only)
    #   p_scale_tmem_prologue cols 404..407   MMA 3 scale (prologue only)
    # TODO: Extend this API to move formally support "aliasing" a buffer
    # with a subtiled version.
    tmem_storage_alias.set_buffer_overlap(
        tlx.reuse_group(
            # RG1: qk_tiles shared with MMA 2/3/5 scale buffers.
            # qk_empties (Compute → MMA 4) and dp_empties (Compute → MMA 1)
            # enforce that Compute has drained qk/dp before scales overwrite.
            tlx.reuse_group(
                qk_tiles,
                tlx.reuse_group(
                    qk_tiles_subtile,
                    group_size=DS_NUM_SUBS,
                ),
                tlx.reuse_group(
                    p_tiles,
                    v_scale_tmem,
                    do_scale_dp_tmem,
                    p_scale_tmem,
                    do_scale_dv_tmem,
                    ds_scale_dq_tmem,
                    k_scale_dq_tmem,
                    group_type=tlx.reuse_group_type.distinct,
                ),
                group_type=tlx.reuse_group_type.shared,
            ),
            # RG2: dv_tiles — no overlap, persistent across the M-loop.
            dv_tiles,
            # RG3: dp/dq_tiles shared with MMA 1/4 scale buffers (body only).
            # dp_empties (Compute → MMA 1 body) prevents scale tmem_copies
            # from corrupting dp_tiles before the Compute task reads them.
            tlx.reuse_group(
                dp_tiles,
                tlx.reuse_group(
                    dp_tiles_subtile,
                    group_size=DS_NUM_SUBS,
                ),
                dq_tiles,
                tlx.reuse_group(
                    ds_tiles_tmem,
                    k_scale_qk_tmem,
                    q_scale_qk_tmem,
                    ds_scale_dk_tmem,
                    q_scale_dk_tmem,
                    group_type=tlx.reuse_group_type.distinct,
                ),
                group_type=tlx.reuse_group_type.shared,
            ),
            # RG4: dk_tiles shared with prologue-only scale buffers.
            # dk_empties prevents the next tile's prologue from overwriting
            # dk_tiles before the Compute task stores it to GMEM.
            tlx.reuse_group(
                dk_tiles,
                tlx.reuse_group(
                    k_scale_tmem_prologue,
                    q_scale_tmem_prologue,
                    v_scale_tmem_prologue,
                    do_scale_dp_tmem_prologue,
                    do_scale_dv_tmem_prologue,
                    tlx.reuse_group(
                        p_scale_tmem_prologue,
                        group_type=tlx.reuse_group_type.shared,
                    ),
                    group_type=tlx.reuse_group_type.distinct,
                ),
                group_type=tlx.reuse_group_type.shared,
            ),
            group_type=tlx.reuse_group_type.distinct,
        ))

    # ===== TMEM barriers =====
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dp_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    # ===== SMEM allocations =====
    k_smem = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    k_dq_smem = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k_dq), NUM_BUFFERS_KV)
    v_smem = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS_KV)
    q_smem = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    q_dk_smem = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    do_smem = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_do), NUM_BUFFERS_DO)
    do_dv_smem = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_do_dv), NUM_BUFFERS_DO)
    # dK consumes dS^T while dQ consumes dS. MXFP8 quantization depends on the
    # reduction axis, so we keep separate internal encodings for the two GEMMs.
    ds_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
    ds_tiles_smem = tlx.local_alloc((BLOCK_N1, BLOCK_M1), p_dtype, NUM_BUFFERS_DS, reuse=ds_storage_alias)
    ds_tiles_smem_subtile = tlx.local_alloc((BLOCK_N1, BLOCK_M1 // DS_NUM_SUBS), p_dtype, NUM_BUFFERS_DS * DS_NUM_SUBS,
                                            reuse=ds_storage_alias)
    ds_dq_tiles_smem = tlx.local_alloc((BLOCK_M1, BLOCK_N1), p_dtype, NUM_BUFFERS_DS, reuse=ds_storage_alias)
    ds_dq_tiles_smem_subtile = tlx.local_alloc((BLOCK_M1 // DS_NUM_SUBS, BLOCK_N1), p_dtype,
                                               NUM_BUFFERS_DS * DS_NUM_SUBS, reuse=ds_storage_alias)
    # SMEM storage spots for dS scales to enable
    # async transfers from SMEM to TMEM.
    # (1, REP_M, REP_HEAD, 2, 256)
    ds_scale_smem = tlx.local_alloc(
        (REP_N, REP_M, 32, 4, 4),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.smem,
        reuse=ds_storage_alias,
    )
    ds_scale_smem_subtile = tlx.local_alloc(
        (REP_N, REP_M, 32, 4, 4 // DS_NUM_SUBS),
        tl.uint8,
        NUM_BUFFERS_TMEM * DS_NUM_SUBS,
        tlx.storage_kind.smem,
        reuse=ds_storage_alias,
    )
    ds_scale_dq_smem = tlx.local_alloc(
        (REP_M, REP_N, 32, 4, 4),
        tl.uint8,
        NUM_BUFFERS_TMEM * DS_NUM_SUBS,
        tlx.storage_kind.smem,
        reuse=ds_storage_alias,
    )
    ds_scale_dq_smem_subtile = tlx.local_alloc(
        (REP_M, REP_N, 32, 4 // DS_NUM_SUBS, 4),
        tl.uint8,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.smem,
        reuse=ds_storage_alias,
    )
    ds_storage_alias.set_buffer_overlap(
        tlx.reuse_group(
            tlx.reuse_group(
                ds_tiles_smem,
                tlx.reuse_group(
                    ds_tiles_smem_subtile,
                    group_size=DS_NUM_SUBS,
                ),
                group_type=tlx.reuse_group_type.shared,
            ),
            tlx.reuse_group(
                ds_dq_tiles_smem,
                tlx.reuse_group(
                    ds_dq_tiles_smem_subtile,
                    group_size=DS_NUM_SUBS,
                ),
                group_type=tlx.reuse_group_type.shared,
            ),
            tlx.reuse_group(
                ds_scale_smem,
                tlx.reuse_group(
                    ds_scale_smem_subtile,
                    group_size=DS_NUM_SUBS,
                ),
                group_type=tlx.reuse_group_type.shared,
            ),
            tlx.reuse_group(
                ds_scale_dq_smem,
                tlx.reuse_group(
                    ds_scale_dq_smem_subtile,
                    group_size=DS_NUM_SUBS,
                ),
                group_type=tlx.reuse_group_type.shared,
            ),
            group_type=tlx.reuse_group_type.distinct,
        ))

    k_scale_smem = tlx.local_alloc((1, REP_N, REP_HEAD, 2, 256), tl.uint8, NUM_BUFFERS_KV)
    k_scale_dq_smem = tlx.local_alloc((1, REP_HEAD, REP_N, 2, 256), tl.uint8, NUM_BUFFERS_KV)
    v_scale_smem = tlx.local_alloc((1, REP_N, REP_HEAD, 2, 256), tl.uint8, NUM_BUFFERS_KV)
    q_scale_smem = tlx.local_alloc((1, REP_M, REP_HEAD, 2, 256), tl.uint8, NUM_BUFFERS_Q)
    q_dk_scale_smem = tlx.local_alloc((1, REP_M, REP_HEAD, 2, 256), tl.uint8, NUM_BUFFERS_Q)
    do_scale_smem = tlx.local_alloc((1, REP_M, REP_HEAD, 2, 256), tl.uint8, NUM_BUFFERS_DO)
    do_scale_dv_smem = tlx.local_alloc((1, REP_HEAD, REP_M, 2, 256), tl.uint8, NUM_BUFFERS_DO)

    slice_size_alloc: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
    # TODO: Actually expose.
    NUM_DKV_STORE_BUFFERS: tl.constexpr = 1
    dkv_store_buf = tlx.local_alloc((BLOCK_N1, slice_size_alloc), tl.bfloat16, NUM_DKV_STORE_BUFFERS)
    DQ_REDUCE_NCOL: tl.constexpr = HEAD_DIM // (EPILOGUE_SUBTILE * 2)
    DQ_REDUCE_ITERS: tl.constexpr = HEAD_DIM // DQ_REDUCE_NCOL
    DQ_REDUCE_STAGES: tl.constexpr = 2
    dq_store_buf = tlx.local_alloc((BLOCK_M1, DQ_REDUCE_NCOL), tlx.dtype_of(desc_dq), DQ_REDUCE_STAGES)

    # ===== SMEM barriers =====
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_dv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    ds_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)
    ds_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    p_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    k_dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    # ===== Warp-specialized async tasks =====
    with tlx.async_tasks():
        # ----- Compute warp: softmax recompute + P/dS quantization -----
        # Default task -- its warp count comes from autotune num_warps (= 4).
        with tlx.async_task("default"):
            blk_idx = 0
            kv_tile_idx = 0
            for _i in range(tiles_per_sm):
                _, persistent_tmem_phase = _get_bufidx_phase(_i, NUM_BUFFERS_TMEM)
                off_seq_h = tile_idx // n_tile_num
                off_z = off_seq_h // H
                off_h = off_seq_h % H
                pid = tile_idx % n_tile_num
                start_n = pid * BLOCK_N1
                base_q = (off_z * H + off_h).to(tl.int64) * N_CTX
                # M / D are [Z, H, N_CTX] flat with stride N_CTX between heads.
                M_off = M_ptr + base_q
                D_off = D_ptr + base_q

                curr_m = 0

                # Prologue: produce P for the first M-block.
                # Call _softmax_recompute_quantization_iter with
                # p_scale_tmem_prologue
                _softmax_recompute_quantization_iter(
                    M_off,
                    D_off,
                    curr_m,
                    blk_idx,
                    qk_scale,
                    qk_tiles_subtile,
                    qk_fulls,
                    qk_empties,
                    p_tiles,
                    p_scale_tmem_prologue,
                    p_fulls,
                    p_empties,
                    dp_tiles_subtile,
                    dp_fulls,
                    dp_empties,
                    ds_tiles_smem_subtile,
                    ds_dq_tiles_smem_subtile,
                    ds_scale_smem_subtile,
                    ds_scale_dq_smem_subtile,
                    ds_fulls,
                    ds_empties,
                    NUM_BUFFERS_TMEM,
                    NUM_BUFFERS_DS,
                    BLOCK_M1,
                    BLOCK_N1,
                    VEC_SIZE,
                    p_dtype,
                    REP_N,
                    REP_M,
                    DS_NUM_SUBS,
                )
                curr_m += BLOCK_M1
                blk_idx += 1

                for _ in range(1, num_steps):
                    # Call _softmax_recompute_quantization_iter with
                    # p_scale_tmem
                    _softmax_recompute_quantization_iter(
                        M_off,
                        D_off,
                        curr_m,
                        blk_idx,
                        qk_scale,
                        qk_tiles_subtile,
                        qk_fulls,
                        qk_empties,
                        p_tiles,
                        p_scale_tmem,
                        p_fulls,
                        p_empties,
                        dp_tiles_subtile,
                        dp_fulls,
                        dp_empties,
                        ds_tiles_smem_subtile,
                        ds_dq_tiles_smem_subtile,
                        ds_scale_smem_subtile,
                        ds_scale_dq_smem_subtile,
                        ds_fulls,
                        ds_empties,
                        NUM_BUFFERS_TMEM,
                        NUM_BUFFERS_DS,
                        BLOCK_M1,
                        BLOCK_N1,
                        VEC_SIZE,
                        p_dtype,
                        REP_N,
                        REP_M,
                        DS_NUM_SUBS,
                    )
                    curr_m += BLOCK_M1
                    blk_idx += 1

                # Epilogue: dK / dV TMA store
                kv_buf_id, kv_phase = _get_bufidx_phase(kv_tile_idx, NUM_BUFFERS_KV)

                tlx.barrier_wait(dv_fulls[0], persistent_tmem_phase)
                slice_size: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
                base_kv = base_q
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
                    tlx.fence("async_shared")
                    tlx.async_descriptor_store(
                        desc_dv,
                        dkv_store_buf[0],
                        [
                            (base_kv + start_n).to(tl.int32),
                            slice_id * slice_size,
                        ],
                    )

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
                    tlx.fence("async_shared")
                    tlx.async_descriptor_store(
                        desc_dk,
                        dkv_store_buf[0],
                        [
                            (base_kv + start_n).to(tl.int32),
                            slice_id * slice_size,
                        ],
                    )
                kv_tile_idx += 1
                tile_idx += num_progs
            tlx.async_descriptor_store_wait(0)

        # ----- Reduction warp: TMA atomic-reduce-add of dQ to GMEM -----
        with tlx.async_task(num_warps=4, registers=88):
            blk_idx = 0
            for _i in range(tiles_per_sm):
                off_seq_h = tile_idx // n_tile_num
                off_z = off_seq_h // H
                off_h = off_seq_h % H
                base_q = (off_z * H + off_h).to(tl.int64) * N_CTX

                curr_m = 0
                for _ in range(num_steps):
                    _, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
                    tlx.barrier_wait(dq_fulls[0], tmem_phase)
                    for slice_id in tl.static_range(DQ_REDUCE_ITERS):
                        dq_smem_idx = slice_id % DQ_REDUCE_STAGES
                        dq_slice = tlx.local_slice(
                            dq_tiles[0],
                            [0, slice_id * DQ_REDUCE_NCOL],
                            [BLOCK_M1, DQ_REDUCE_NCOL],
                        )
                        dq = tlx.local_load(dq_slice)
                        dq = dq * LN2
                        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                        tlx.local_store(
                            dq_store_buf[dq_smem_idx],
                            dq.to(tlx.dtype_of(desc_dq)),
                        )
                        tlx.fence("async_shared")
                        tlx.async_descriptor_store(
                            desc_dq,
                            dq_store_buf[dq_smem_idx],
                            [
                                (base_q + curr_m).to(tl.int32),
                                slice_id * DQ_REDUCE_NCOL,
                            ],
                            store_reduce="add",
                        )

                    tlx.barrier_arrive(dq_empties[0])
                    curr_m += BLOCK_M1
                    blk_idx += 1
                tile_idx += num_progs
            tlx.async_descriptor_store_wait(0)

        # ----- MMA warp: 5 blockscaled GEMMs per M-block -----
        with tlx.async_task(num_warps=1, registers=24):
            blk_idx = 0
            kv_tile_idx = 0
            for _i in range(tiles_per_sm):
                kv_buf_id, kv_phase = _get_bufidx_phase(kv_tile_idx, NUM_BUFFERS_KV)
                # --- Prolog: first M-block ---
                q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                _, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
                _, tmem_phase_prev = _get_bufidx_phase(blk_idx - 1, NUM_BUFFERS_TMEM)
                _, persistent_tmem_phase = _get_bufidx_phase(_i, NUM_BUFFERS_TMEM)

                # MMA 1: qkT = K @ Q^T
                tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)
                tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
                # REUSE_GROUP_1 SYNCHRONIZATION:
                # MMA 3: p_empties
                # with buffers: p_tiles, p_scale_tmem, do_scale_dv_tmem
                # MMA 5: dq_empties[0].
                # with buffers: k_scale_dq_tmem, ds_scale_dq_tmem
                # MMA 2: Handled by ds_fulls before MMA 4.
                # with buffers: v_scale_tmem, do_scale_dp_tmem
                tlx.barrier_wait(p_empties[0], tmem_phase ^ 1)
                tlx.barrier_wait(dq_empties[0], tmem_phase_prev)
                # REUSE_GROUP_4 SYNCHRONIZATION:
                # ALL PROLOGUES MUST WAIT FOR DK_TILES TO EMPTY
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

                # MMA 2: dpT = V @ dO^T  (dP shares TMEM with dQ via reuse)
                # REUSE_GROUP_3 SYNCHRONIZATION:
                # Linear order for all deps. For epilogue MMA 5 -> MMA 2.
                # MMA 5 handled by dq_empties[0] above.
                tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)
                tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
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
                    mBarriers=[dp_fulls[0], do_empties[do_buf_id]],
                )

                # MMA 3: dV += P^T @ dO  (P_scale on-the-fly)
                # REUSE_GROUP_1 SYNCHRONIZATION:
                # p_fulls Waits for QK_EMPTIES implicitly
                tlx.barrier_wait(p_fulls[0], tmem_phase)
                tlx.barrier_wait(dv_empties[0], persistent_tmem_phase ^ 1)
                tlx.barrier_wait(do_dv_fulls[do_buf_id], do_phase)
                tlx.tmem_copy(do_scale_dv_smem[do_buf_id], do_scale_dv_tmem_prologue[0])
                # Fence for the p_scale
                tlx.fence("async_shared")
                tlx.async_dot_scaled(
                    p_tiles[0],
                    do_dv_smem[do_buf_id],
                    dv_tiles[0],
                    p_scale_tmem_prologue[0],
                    P_FP8_FORMAT,
                    do_scale_dv_tmem_prologue[0],
                    DO_FP8_FORMAT,
                    use_acc=False,
                    mBarriers=[
                        do_dv_empties[do_buf_id],
                        p_empties[0],
                    ],
                )
                blk_idx += 1

                # Wait for MMA 4. This avoids needing to thread this into
                # the body/epilogue with a conditional.
                tlx.barrier_wait(k_dq_fulls[kv_buf_id], kv_phase)

                # --- Main loop: iters 1 .. num_steps-1 ---
                for j in range(1, num_steps):
                    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                    _, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
                    prev_blk_idx = blk_idx - 1
                    q_buf_id_prev, q_phase_prev = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                    _, tmem_phase_prev = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
                    ds_buf_id_prev, ds_phase_prev = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)

                    # MMA 1: qkT = K @ Q^T (current)
                    tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
                    # REUSE_GROUP_1 SYNCHRONIZATION:
                    # MMA 3: p_empties
                    # with buffers: p_tiles, p_scale_tmem, do_scale_dv_tmem
                    # MMA 5: dq_empties[0].
                    # with buffers: k_scale_dq_tmem, ds_scale_dq_tmem
                    # MMA 2: Handled by ds_fulls before MMA 4.
                    # Not needed for prologue.
                    # with buffers: v_scale_tmem, do_scale_dp_tmem
                    tlx.barrier_wait(p_empties[0], tmem_phase ^ 1)
                    tlx.barrier_wait(dq_empties[0], tmem_phase_prev ^ 1)
                    # REUSE_GROUP_3: wait for Compute to finish reading dp_tiles
                    # before overwriting with scale tmem_copies.
                    tlx.barrier_wait(dp_empties[0], tmem_phase_prev)
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

                    # MMA 5: dQ = dS^T_trans @ K (previous M-block)
                    # REUSE_GROUP_3 SYNCHRONIZATION:
                    # Linear order for all deps. For body MMA 2 -> MMA 5.
                    # MMA 2 handled by ds_fulls[ds_buf_id_prev].
                    tlx.barrier_wait(ds_fulls[ds_buf_id_prev], ds_phase_prev)
                    # REUSE_GROUP_1: wait for Compute to finish reading qk_tiles
                    # before overwriting with scale tmem_copies.
                    tlx.barrier_wait(qk_empties[0], tmem_phase)
                    # Copy the dQ-specific dS scales from SMEM to TMEM.
                    # Fence for scale copies to be visible.
                    tlx.fence("async_shared")
                    tlx.tmem_copy(ds_scale_dq_smem[0], ds_scale_dq_tmem[0])
                    tlx.tmem_copy(k_scale_dq_smem[kv_buf_id], k_scale_dq_tmem[0])
                    tlx.async_dot_scaled(
                        ds_dq_tiles_smem[ds_buf_id_prev],
                        k_dq_smem[kv_buf_id],
                        dq_tiles[0],
                        ds_scale_dq_tmem[0],
                        DS_FP8_FORMAT,
                        k_scale_dq_tmem[0],
                        K_FP8_FORMAT,
                        use_acc=False,
                        mBarriers=[dq_fulls[0]],
                    )

                    # MMA 4: dK += dS^T @ Q (previous M-block, dS from SMEM)

                    # REUSE_GROUP_1 SYNCHRONIZATION:
                    # qk_empties waits for MMA 1 to finish above.

                    # REUSE_GROUP_3 SYNCHRONIZATION:
                    # Linear order for all deps. For body MMA 5 -> MMA 4.
                    # MMA 5 handled by dq_empties[0].

                    # REUSE_GROUP_4 SYNCHRONIZATION:
                    # DK_EMPTIES must wait for the prologue
                    # to finish. All of these are grouped into
                    # another bucket.
                    # MMA 1: Handled by QK iter 1 reusing QK
                    # MMA 2: Handled by ds_fulls barrier in MMA 5
                    # MMA 3: Handled by p_empties in MMA 1
                    tlx.barrier_wait(q_dk_fulls[q_buf_id_prev], q_phase_prev)
                    # Copy from SMEM to TMEM
                    # TODO: Blocked on TLX feature
                    # tlx.tmem_copy(ds_tiles_smem[ds_buf_id_prev], ds_tiles_tmem[0])
                    tlx.barrier_wait(dq_empties[0], tmem_phase_prev)
                    # Fence for ds_scale_smem to be visible.
                    tlx.fence("async_shared")
                    tlx.tmem_copy(ds_scale_smem[0], ds_scale_dk_tmem[0])
                    tlx.tmem_copy(q_dk_scale_smem[q_buf_id_prev], q_scale_dk_tmem[0])
                    tlx.async_dot_scaled(
                        # TODO: ds_tiles_tmem[0],
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

                    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)

                    # MMA 2: dpT = V @ dO^T (current)
                    # REUSE_GROUP_3 SYNCHRONIZATION:
                    # Linear order for all deps. For body MMA 4 -> MMA 2.
                    # MMA 4 handled by q_dk_empties[q_buf_id_prev]. This is fine
                    # because we just need to reclaim the inputs.
                    tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                    tlx.barrier_wait(q_dk_empties[q_buf_id_prev], q_phase_prev)
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
                        mBarriers=[dp_fulls[0], do_empties[do_buf_id]],
                    )

                    # MMA 3: dV += P^T @ dO (current)
                    tlx.barrier_wait(p_fulls[0], tmem_phase)
                    tlx.barrier_wait(do_dv_fulls[do_buf_id], do_phase)
                    tlx.tmem_copy(do_scale_dv_smem[do_buf_id], do_scale_dv_tmem[0])
                    # Fence for the p_scale
                    tlx.fence("async_shared")
                    tlx.async_dot_scaled(
                        p_tiles[0],
                        do_dv_smem[do_buf_id],
                        dv_tiles[0],
                        p_scale_tmem[0],
                        P_FP8_FORMAT,
                        do_scale_dv_tmem[0],
                        DO_FP8_FORMAT,
                        use_acc=True,
                        mBarriers=[
                            do_dv_empties[do_buf_id],
                            p_empties[0],
                        ],
                    )
                    blk_idx += 1

                tlx.tcgen05_commit(dv_fulls[0])
                # Signal once MMA 1 is done.
                tlx.tcgen05_commit(k_empties[kv_buf_id])

                # --- Epilog: last dK / dQ ---
                prev_blk_idx = blk_idx - 1
                q_buf_id, q_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                _, tmem_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
                ds_buf_id, ds_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)

                # MMA 4: dK += dS^T @ Q (last)
                # REUSE_GROUP_3 SYNCHRONIZATION:
                # Linear order for all deps. For epilogue MMA 2 -> MMA 4.
                # MMA 2 handled by ds_fulls[ds_buf_id].
                tlx.barrier_wait(ds_fulls[ds_buf_id], ds_phase)
                tlx.barrier_wait(q_dk_fulls[q_buf_id], q_phase)
                # Copy from SMEM to TMEM
                # Fence for ds_scale_smem to be visiible.
                tlx.fence("async_shared")
                # TODO: Blocked on TLX feature
                # tlx.tmem_copy(ds_tiles_smem[ds_buf_id], ds_tiles_tmem[0])
                tlx.tmem_copy(q_dk_scale_smem[q_buf_id], q_scale_dk_tmem[0])
                tlx.tmem_copy(ds_scale_smem[0], ds_scale_dk_tmem[0])
                tlx.async_dot_scaled(
                    # TODO: ds_tiles_tmem[0],
                    ds_tiles_smem[ds_buf_id],
                    q_dk_smem[q_buf_id],
                    dk_tiles[0],
                    ds_scale_dk_tmem[0],
                    DS_FP8_FORMAT,
                    q_scale_dk_tmem[0],
                    Q_FP8_FORMAT,
                    use_acc=num_steps > 1,
                    mBarriers=[
                        q_dk_empties[q_buf_id],
                        dk_fulls[0],
                    ],
                )
                # MMA 5: dQ = dS^T_trans @ K (last)
                # REUSE_GROUP_3 SYNCHRONIZATION:
                # Linear order for all deps. For epilogue MMA 4 -> MMA 5.
                # MMA 4 handled by q_dk_empties[q_buf_id].
                tlx.barrier_wait(dq_empties[0], tmem_phase ^ 1)
                tlx.barrier_wait(q_dk_empties[q_buf_id], q_phase)
                # Experiment: for the N_CTX=128 repro, MMA 5 epilogue can reuse
                # the dQ scales packed into SMEM during dS quantization and
                # copied into TMEM here.
                # Fence for ds_scale_dq_smem to be visible.
                tlx.fence("async_shared")
                tlx.tmem_copy(ds_scale_dq_smem[0], ds_scale_dq_tmem[0])
                tlx.tmem_copy(k_scale_dq_smem[kv_buf_id], k_scale_dq_tmem[0])
                tlx.async_dot_scaled(
                    ds_dq_tiles_smem[ds_buf_id],
                    k_dq_smem[kv_buf_id],
                    dq_tiles[0],
                    ds_scale_dq_tmem[0],
                    DS_FP8_FORMAT,
                    k_scale_dq_tmem[0],
                    K_FP8_FORMAT,
                    use_acc=False,
                    mBarriers=[dq_fulls[0], ds_empties[ds_buf_id], k_dq_empties[kv_buf_id]],
                )
                kv_tile_idx += 1
                tile_idx += num_progs

        # ----- Load warp: TMA loads of FP8 data + scales -----
        with tlx.async_task(num_warps=1, registers=24):
            blk_idx = 0
            kv_tile_idx = 0
            for _i in range(tiles_per_sm):
                off_seq_h = tile_idx // n_tile_num
                off_z = off_seq_h // H
                off_h = off_seq_h % H
                pid = tile_idx % n_tile_num
                start_n = pid * BLOCK_N1
                base_q = (off_z * H + off_h).to(tl.int64) * N_CTX
                base_kv = base_q
                # Scale TMA layout: [Z*H, REP_N (or REP_M), REP_HEAD, 2, 256].
                # Tile selects (z, h) via off_seq_h, and the M / N index via
                # the 2nd dim.
                sf_off_seq_h = off_seq_h
                kv_scale_n = pid * REP_N

                # Load K data + scale
                kv_buf_id, kv_phase = _get_bufidx_phase(kv_tile_idx, NUM_BUFFERS_KV)
                tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
                tlx.barrier_expect_bytes(k_fulls[kv_buf_id], (K_BYTES * BLOCK_N1 * HEAD_DIM) + SCALE_BYTES)
                tlx.async_descriptor_load(
                    desc_k,
                    k_smem[kv_buf_id],
                    [(base_kv + start_n).to(tl.int32), 0],
                    k_fulls[kv_buf_id],
                )
                tlx.async_descriptor_load(
                    desc_k_scale,
                    k_scale_smem[kv_buf_id],
                    [sf_off_seq_h.to(tl.int32), kv_scale_n.to(tl.int32), 0, 0, 0],
                    k_fulls[kv_buf_id],
                )
                tlx.barrier_wait(k_dq_empties[kv_buf_id], kv_phase ^ 1)
                tlx.barrier_expect_bytes(
                    k_dq_fulls[kv_buf_id],
                    (K_BYTES * BLOCK_N1 * HEAD_DIM) + SCALE_BYTES,
                )
                tlx.async_descriptor_load(
                    desc_k_dq,
                    k_dq_smem[kv_buf_id],
                    [(base_kv + start_n).to(tl.int32), 0],
                    k_dq_fulls[kv_buf_id],
                )
                tlx.async_descriptor_load(
                    desc_k_dq_scale,
                    k_scale_dq_smem[kv_buf_id],
                    [sf_off_seq_h.to(tl.int32), 0, kv_scale_n.to(tl.int32), 0, 0],
                    k_dq_fulls[kv_buf_id],
                )

                # Load V data + scale
                # Share 1 barrier
                tlx.barrier_expect_bytes(v_fulls[kv_buf_id], K_BYTES * BLOCK_N1 * HEAD_DIM + SCALE_BYTES)
                tlx.async_descriptor_load(
                    desc_v,
                    v_smem[kv_buf_id],
                    [(base_kv + start_n).to(tl.int32), 0],
                    v_fulls[kv_buf_id],
                )
                tlx.async_descriptor_load(
                    desc_v_scale,
                    v_scale_smem[kv_buf_id],
                    [sf_off_seq_h.to(tl.int32), kv_scale_n.to(tl.int32), 0, 0, 0],
                    v_fulls[kv_buf_id],
                )

                # Load first Q + scale
                curr_m = 0
                q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                q_scale_m = (curr_m // 128) * REP_M
                tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_buf_id], (Q_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES)
                tlx.async_descriptor_load(
                    desc_q,
                    q_smem[q_buf_id],
                    [(base_q + curr_m).to(tl.int32), 0],
                    q_fulls[q_buf_id],
                )
                tlx.async_descriptor_load(
                    desc_q_scale,
                    q_scale_smem[q_buf_id],
                    [sf_off_seq_h.to(tl.int32), q_scale_m, 0, 0, 0],
                    q_fulls[q_buf_id],
                )

                # Load first dO + scale
                do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                do_scale_m = (curr_m // 128) * REP_M
                tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                tlx.barrier_expect_bytes(
                    do_fulls[do_buf_id],
                    (DO_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES,
                )
                tlx.async_descriptor_load(
                    desc_do,
                    do_smem[do_buf_id],
                    [(base_q + curr_m).to(tl.int32), 0],
                    do_fulls[do_buf_id],
                )
                tlx.async_descriptor_load(
                    desc_do_scale,
                    do_scale_smem[do_buf_id],
                    [sf_off_seq_h.to(tl.int32), do_scale_m, 0, 0, 0],
                    do_fulls[do_buf_id],
                )
                tlx.barrier_wait(do_dv_empties[do_buf_id], do_phase ^ 1)
                tlx.barrier_expect_bytes(
                    do_dv_fulls[do_buf_id],
                    (DO_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES,
                )
                tlx.async_descriptor_load(
                    desc_do_dv,
                    do_dv_smem[do_buf_id],
                    [(base_q + curr_m).to(tl.int32), 0],
                    do_dv_fulls[do_buf_id],
                )
                tlx.async_descriptor_load(
                    desc_do_dv_scale,
                    do_scale_dv_smem[do_buf_id],
                    [sf_off_seq_h.to(tl.int32), 0, do_scale_m, 0, 0],
                    do_dv_fulls[do_buf_id],
                )
                curr_m += BLOCK_M1
                blk_idx += 1

                # Load subsequent Q / dO tiles.
                for _j in range(1, num_steps):
                    prev_blk_idx = blk_idx - 1
                    prev_m = curr_m - BLOCK_M1
                    prev_q_buf_id, prev_q_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                    q_scale_m = (curr_m // 128) * REP_M
                    do_scale_m = (curr_m // 128) * REP_M

                    tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                    tlx.barrier_expect_bytes(q_fulls[q_buf_id], (Q_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES)
                    tlx.async_descriptor_load(
                        desc_q,
                        q_smem[q_buf_id],
                        [(base_q + curr_m).to(tl.int32), 0],
                        q_fulls[q_buf_id],
                    )
                    tlx.async_descriptor_load(
                        desc_q_scale,
                        q_scale_smem[q_buf_id],
                        [sf_off_seq_h.to(tl.int32), q_scale_m, 0, 0, 0],
                        q_fulls[q_buf_id],
                    )
                    tlx.barrier_wait(q_dk_empties[prev_q_buf_id], prev_q_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        q_dk_fulls[prev_q_buf_id],
                        (Q_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES,
                    )
                    # prev_blk_idx is the global ring-buffer position; q_dk
                    # addresses must stay local to the current (z, h, pid) tile.
                    tlx.async_descriptor_load(
                        desc_q_dk,
                        q_dk_smem[prev_q_buf_id],
                        [(base_q + prev_m).to(tl.int32), 0],
                        q_dk_fulls[prev_q_buf_id],
                    )
                    tlx.async_descriptor_load(
                        desc_q_dk_scale,
                        q_dk_scale_smem[prev_q_buf_id],
                        [sf_off_seq_h.to(tl.int32), 0, (prev_m // 128) * REP_M, 0, 0],
                        q_dk_fulls[prev_q_buf_id],
                    )

                    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        do_fulls[do_buf_id],
                        (DO_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES,
                    )
                    tlx.async_descriptor_load(
                        desc_do,
                        do_smem[do_buf_id],
                        [(base_q + curr_m).to(tl.int32), 0],
                        do_fulls[do_buf_id],
                    )
                    tlx.async_descriptor_load(
                        desc_do_scale,
                        do_scale_smem[do_buf_id],
                        [sf_off_seq_h.to(tl.int32), do_scale_m, 0, 0, 0],
                        do_fulls[do_buf_id],
                    )
                    tlx.barrier_wait(do_dv_empties[do_buf_id], do_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        do_dv_fulls[do_buf_id],
                        (DO_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES,
                    )
                    tlx.async_descriptor_load(
                        desc_do_dv,
                        do_dv_smem[do_buf_id],
                        [(base_q + curr_m).to(tl.int32), 0],
                        do_dv_fulls[do_buf_id],
                    )
                    tlx.async_descriptor_load(
                        desc_do_dv_scale,
                        do_scale_dv_smem[do_buf_id],
                        [sf_off_seq_h.to(tl.int32), 0, do_scale_m, 0, 0],
                        do_dv_fulls[do_buf_id],
                    )
                    curr_m += BLOCK_M1
                    blk_idx += 1
                last_blk_idx = blk_idx - 1
                last_m = curr_m - BLOCK_M1
                last_q_buf_id, last_q_phase = _get_bufidx_phase(last_blk_idx, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_dk_empties[last_q_buf_id], last_q_phase ^ 1)
                tlx.barrier_expect_bytes(
                    q_dk_fulls[last_q_buf_id],
                    (Q_BYTES * BLOCK_M1 * HEAD_DIM) + SCALE_BYTES,
                )
                tlx.async_descriptor_load(
                    desc_q_dk,
                    q_dk_smem[last_q_buf_id],
                    [(base_q + last_m).to(tl.int32), 0],
                    q_dk_fulls[last_q_buf_id],
                )
                tlx.async_descriptor_load(
                    desc_q_dk_scale,
                    q_dk_scale_smem[last_q_buf_id],
                    [sf_off_seq_h.to(tl.int32), 0, (last_m // 128) * REP_M, 0, 0],
                    q_dk_fulls[last_q_buf_id],
                )
                kv_tile_idx += 1
                tile_idx += num_progs


# ---------------------------------------------------------------------------
# Backward host wrapper
# ---------------------------------------------------------------------------


def attention_bwd(
    do,
    do_dv,
    q,
    q_dk,
    k,
    k_dq,
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
):
    """MXFP8 attention backward.

    Operates on dense [Z, H, N_CTX, HEAD_DIM] tensors. Q / K / V are FP8 E4M3
    with E8M0 block scales pre-quantized in TMA-preshuffled 5D layout
    (matches the forward kernel's scale convention).

    Backward uses dO in two incompatible GEMM orientations:
      - MMA 2 consumes dO^T, so `do` / `do_scale` must be quantized in the
        original [N_CTX, HEAD_DIM] layout.
      - MMA 3 consumes dO directly, so `do_dv` / `do_dv_scale` must be
        quantized with the reduction axis (N_CTX) as the blocked dimension.
      - MMA 4 consumes Q directly, so `q_dk` / `q_dk_scale` must use the same
        reduction-axis-swapped encoding.
      - MMA 5 consumes K directly, so `k_dq` / `k_dq_scale` must also use the
        reduction-axis-swapped encoding.

    Returns (dQ, dK, dV) with dQ in FP32, dK / dV in BF16.

    Non-causal only. Assumes N_CTX is a multiple of 128.
    """
    assert q.shape == q_dk.shape == k.shape == k_dq.shape == v.shape == do.shape, (
        "Q, Q_dK, K, K_dQ, V, dO must have the same shape")
    Z, H, N_CTX, HEAD_DIM = q.shape
    assert HEAD_DIM == 128, "this kernel only supports HEAD_DIM = 128"
    assert N_CTX % 128 == 0, "N_CTX must be a multiple of 128 (BLOCK_M1)"

    y_dim = Z * H * N_CTX

    # Compute Delta = rowsum(O * dO) into an [Z, H, N_CTX] FP32 buffer.
    delta = torch.empty_like(M)

    PRE_BLOCK_M = 32
    preproc_grid = (triton.cdiv(N_CTX, PRE_BLOCK_M), Z * H)
    do_preproc = do_bf16 if do_bf16 is not None else do
    _attn_bwd_preprocess[preproc_grid](
        o,
        do_preproc,
        delta,
        Z,
        H,
        N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=PRE_BLOCK_M,
    )

    # Allocate outputs. dQ is FP32 (TMA reduce-add accumulation target).
    dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    dk = torch.zeros(k.shape, device=k.device, dtype=torch.bfloat16)
    dv = torch.zeros(v.shape, device=v.device, dtype=torch.bfloat16)

    dummy_block = [1, 1]
    dummy_5d = [1, 1, 1, 1, 1]

    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_q_dk = TensorDescriptor(q_dk, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_k_dq = TensorDescriptor(k_dq, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_do = TensorDescriptor(do, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_do_dv = TensorDescriptor(do_dv, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_dq = TensorDescriptor(dq, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_dk = TensorDescriptor(dk, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_dv = TensorDescriptor(dv, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)

    desc_q_scale = TensorDescriptor.from_tensor(q_scale, block_shape=dummy_5d)
    desc_q_dk_scale = TensorDescriptor.from_tensor(q_dk_scale, block_shape=dummy_5d)
    desc_k_scale = TensorDescriptor.from_tensor(k_scale, block_shape=dummy_5d)
    desc_k_dq_scale = TensorDescriptor.from_tensor(k_dq_scale, block_shape=dummy_5d)
    desc_v_scale = TensorDescriptor.from_tensor(v_scale, block_shape=dummy_5d)
    desc_do_scale = TensorDescriptor.from_tensor(do_scale, block_shape=dummy_5d)
    desc_do_dv_scale = TensorDescriptor.from_tensor(do_dv_scale, block_shape=dummy_5d)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(meta):
        total_tiles = triton.cdiv(N_CTX, meta["BLOCK_N1"]) * Z * H
        return (
            min(NUM_SMS, total_tiles),
            1,
            1,
        )

    _attn_bwd_mxf8_ws[grid](
        desc_q,
        desc_q_dk,
        desc_k,
        desc_k_dq,
        desc_v,
        desc_do,
        desc_do_dv,
        desc_dq,
        desc_dk,
        desc_dv,
        dq,
        sm_scale,
        M,
        delta,
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
    )
    return dq, dk, dv


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, q_scale, k_scale, v_scale, sm_scale, causal):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        stage = 3 if causal else 1

        o = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        extra_kern_args = {}

        m_tensor = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            q,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        desc_o = TensorDescriptor(
            o,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        assert k_scale is not None and v_scale is not None and q_scale is not None, (
            "All scales must be provided for MXFP8")
        dummy_block_shape = [1, 1, 1, 1, 1]
        desc_q_scale = TensorDescriptor.from_tensor(q_scale, block_shape=dummy_block_shape)
        desc_k_scale = TensorDescriptor.from_tensor(k_scale, block_shape=dummy_block_shape)
        desc_v_scale = TensorDescriptor.from_tensor(v_scale, block_shape=dummy_block_shape)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        def grid(META):
            return (
                min(
                    NUM_SMS,
                    triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1],
                ),
                1,
                1,
            )

        ctx.grid = grid
        _attn_fwd_mxf8_ws[grid](
            sm_scale,
            m_tensor,  #
            q.shape[0],
            q.shape[1],  #
            desc_q,
            desc_k,
            desc_v,
            desc_o,  #
            desc_q_scale,
            desc_k_scale,
            desc_v_scale,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, m_tensor)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


def generate_tensor_with_block_distributions(
    reference_tensor: torch.Tensor,
    min_max_ranges: list[tuple[float, float]],
    block_size: int = 32,
    num_pregenerated_blocks: int = 100,
) -> torch.Tensor:
    """
    Generate a tensor with the same shape as reference_tensor but with different
    distributions for different blocks. Fully vectorized - no Python loops.

    Parameters:
    -----------
    reference_tensor : torch.Tensor
        The reference tensor whose shape, dtype, device, and properties to copy.
    min_max_ranges : list[tuple[float, float]]
        List of [min, max] value ranges. Each block will be assigned a range
        cyclically from this list.
    block_size : int
        The size of each block (default: 32 for MXFP8).
    num_pregenerated_blocks : int
        Number of random blocks to pre-generate for each range (default: 100).

    Returns:
    --------
    torch.Tensor
        A new tensor with the same shape as reference_tensor but with varying
        distributions across blocks.
    """
    device = reference_tensor.device
    dtype = reference_tensor.dtype
    requires_grad = reference_tensor.requires_grad
    shape = reference_tensor.shape

    total_elements = reference_tensor.numel()
    num_blocks = (total_elements + block_size - 1) // block_size
    num_ranges = len(min_max_ranges)

    # Pre-generate random blocks for all ranges at once
    # Shape: [num_ranges, num_pregenerated_blocks, block_size]
    all_blocks = []
    for min_val, max_val in min_max_ranges:
        blocks = (torch.rand(num_pregenerated_blocks, block_size, device=device, dtype=dtype) * (max_val - min_val) +
                  min_val)
        all_blocks.append(blocks)
    all_blocks = torch.stack(all_blocks)  # [num_ranges, num_pregenerated, block_size]

    # Generate random indices on GPU (not CPU!)
    range_indices = torch.randint(0, num_ranges, (num_blocks, ), device=device)
    block_indices = torch.randint(0, num_pregenerated_blocks, (num_blocks, ), device=device)

    # Use advanced indexing to select all blocks at once - NO PYTHON LOOP!
    selected_blocks = all_blocks[range_indices, block_indices]  # [num_blocks, block_size]

    # Flatten and take only the elements we need
    generated_tensor = selected_blocks.flatten()[:total_elements]

    # Reshape to original shape
    generated_tensor = generated_tensor.view(shape).contiguous()

    # Set requires_grad if needed
    if requires_grad:
        generated_tensor.requires_grad_(True)

    return generated_tensor


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


def generate_attention_inputs(shape, device, dtype):
    """Generate Q, K, V tensors for attention.

    For FP8 dtype, generates MXFP8 quantized tensors.
    For other dtypes, generates random tensors with the specified dtype.

    Args:
        shape: Tuple of (Z, H, N_CTX, HEAD_DIM)
        device: Device to create tensors on
        dtype: Data type for the tensors

    Returns:
        Tuple of ((q, q_scale, q_ref), (k, k_scale, k_ref), (v, v_scale, v_ref))
        where scales are None for non-FP8 dtypes and ref tensors are bf16 copies.
    """
    # Generate bf16 reference tensors first
    orig_dtype = torch.bfloat16
    q_ref = torch.empty(shape, device=device, dtype=orig_dtype).normal_(mean=0.0, std=0.5).contiguous()
    k_ref = torch.empty(shape, device=device, dtype=orig_dtype).normal_(mean=0.0, std=0.5).contiguous()
    v_ref = torch.empty(shape, device=device, dtype=orig_dtype).normal_(mean=0.0, std=0.5).contiguous()
    # Convert to 2D for MXFP8
    q_2d = q_ref.reshape(shape[0] * shape[1] * shape[2], shape[3]).contiguous()
    k_2d = k_ref.reshape(shape[0] * shape[1] * shape[2], shape[3]).contiguous()
    # Transpose V so we can quantize along the N dimension
    v_2d = v_ref.reshape(shape[0] * shape[1] * shape[2], shape[3]).contiguous()
    v_2d_t = v_2d.t().contiguous()

    q_mx = MXTensor.to_mx(
        q_2d,
        dtype,
        scaling_mode=ScaleCalculationMode.RCEIL,
        is_swizzled_scales=True,
    )
    k_mx = MXTensor.to_mx(
        k_2d,
        dtype,
        scaling_mode=ScaleCalculationMode.RCEIL,
        is_swizzled_scales=True,
    )
    v_mx = MXTensor.to_mx(
        v_2d_t,
        dtype,
        scaling_mode=ScaleCalculationMode.RCEIL,
        is_swizzled_scales=True,
    )
    q_data = q_mx.qdata.reshape(shape).contiguous()
    k_data = k_mx.qdata.reshape(shape).contiguous()
    v_data = v_mx.qdata.t().reshape(shape).contiguous()
    q_scale = swizzled_to_tma_preshuffled(q_mx.scale, shape[2], shape[3], 32, shape[0] * shape[1])
    k_scale = swizzled_to_tma_preshuffled(k_mx.scale, shape[2], shape[3], 32, shape[0] * shape[1])
    v_scale = swizzled_to_tma_preshuffled(v_mx.scale, shape[3], shape[2], 32, shape[0] * shape[1])
    return (q_data, q_scale, q_ref), (k_data, k_scale, k_ref), (v_data, v_scale, v_ref)


def attention(q, k, v, q_scale, k_scale, v_scale, sm_scale, causal, config=None):
    if config is None:
        return _attention.apply(q, k, v, q_scale, k_scale, v_scale, sm_scale, causal)

    # Non-autotuned path with explicit config
    HEAD_DIM_K = q.shape[-1]
    stage = 3 if causal else 1
    o = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    m_tensor = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    dummy_block = [1, 1]
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)

    dummy_block_shape = [1, 1, 1, 1, 1]
    desc_q_scale = TensorDescriptor.from_tensor(q_scale, block_shape=dummy_block_shape)
    desc_k_scale = TensorDescriptor.from_tensor(k_scale, block_shape=dummy_block_shape)
    desc_v_scale = TensorDescriptor.from_tensor(v_scale, block_shape=dummy_block_shape)

    # Apply pre_hook to set block shapes
    nargs = {
        **config,
        "HEAD_DIM": HEAD_DIM_K,
        "desc_q": desc_q,
        "desc_k": desc_k,
        "desc_v": desc_v,
        "desc_o": desc_o,
        "desc_q_scale": desc_q_scale,
        "desc_k_scale": desc_k_scale,
        "desc_v_scale": desc_v_scale,
    }
    _mxf8_host_descriptor_pre_hook(nargs)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (min(NUM_SMS, triton.cdiv(q.shape[2], config["BLOCK_M"]) * q.shape[0] * q.shape[1]), 1, 1)
    _attn_fwd_mxf8_ws.fn[grid](
        sm_scale,
        m_tensor,
        q.shape[0],
        q.shape[1],
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        desc_q_scale,
        desc_k_scale,
        desc_v_scale,
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        num_stages=1,
        **config,
    )
    return o
