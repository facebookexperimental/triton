import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.language.extra.tlx.mxfp8_utils import _to_mxfp8_block
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
            "GROUP_SIZE_N": 1,
        },
        num_stages=0,
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
):
    lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)

    for start_n in tl.range(lo, hi, BLOCK_N):
        _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
        tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
        qk = tlx.local_load(tlx.local_view(qk_tiles, cid))

        if STAGE == 2:
            col_limit_right = (offs_m - start_n + 1)[:, None]
            qk = _apply_causal_mask(qk, col_limit_right, BLOCK_N)

        # compute m_i, p in registers
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        tlx.barrier_wait(tlx.local_view(alpha_empties, cid), qk_phase ^ 1)
        # Use alpha[0] for cid=0, and alpha[BLOCK_N] for cid=1
        tlx.local_store(tlx.local_view(alpha_tiles, cid * BLOCK_N), alpha[:, None])
        tlx.barrier_arrive(tlx.local_view(alpha_fulls, cid))

        qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
        p_i = tl.math.exp2(qk)
        tlx.barrier_wait(tlx.local_view(p_empties, cid), qk_phase ^ 1)
        # Convert p_i to mxFP8 + write out the scales.
        _to_mxfp8_block(
            p_i,
            tlx.local_view(p_tiles, cid),
            tlx.local_view(p_scale_tiles, cid),
            VEC_SIZE,
            out_dtype,
        )
        tlx.fence_async_shared()
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
                      GROUP_SIZE_N: tl.constexpr,  #
                      ):
    tl.static_assert(NUM_MMA_GROUPS == 2)
    tl.static_assert(NUM_BUFFERS_QK == 1)
    tl.static_assert(NUM_BUFFERS_Q == 1)

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
    p_scale_tiles = tlx.local_alloc((1, REP_M, REP_N, 2, 256), tl.uint8, NUM_MMA_GROUPS)

    q_scale_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    q_scale_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_scale_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_scale_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    # Calculate scale bytes for barrier expect
    Q_SCALE_BYTES: tl.constexpr = REP_M * REP_HEAD * 2 * 256
    K_SCALE_BYTES: tl.constexpr = REP_N * REP_HEAD * 2 * 256
    V_SCALE_BYTES: tl.constexpr = REP_N * REP_HEAD * 2 * 256

    qk_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    # TODO: Update this diagram
    # Shared buffer for QK, P and Alpha, l, and m.
    # A single QK buffer is split evenly:
    #   - First half  : stores P
    #   - Second half  : stores Alpha, l, and m
    #     QK : |                              BLK_M/2 * BLOCK_N * fp32                  |
    #     P:                                                |  BLK_M/2 * BLOCK_N * fp16 |
    #  Alpha : |BLK_M/2*1*fp32|
    #     l :                 |BLK_M/2*1*fp32|
    #     m :                                |BLK_M/2*1*fp32|
    # When working with smaller dtypes (e.g. FP8), we pad the original data to match the original
    # boundaries and prevent overlap.
    qk_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_N), qk_dtype, NUM_MMA_GROUPS, tlx.storage_kind.tmem,
                               reuse=qk_storage_alias)
    # Note: P is not shared because of scaled_dot requirements.
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS,
    )
    alpha_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_storage_alias,
    )
    l_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_storage_alias,
    )
    m_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_storage_alias,
    )

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
                        # Use alpha[0] for cid=0, and alpha[BLOCK_N] for cid=1
                        alpha_1 = tlx.local_load(alpha_tiles[cid * BLOCK_N])
                        tlx.barrier_arrive(alpha_empties[cid])
                        acc = tlx.local_load(acc_tiles[cid])
                        acc = _mul_f32x2(acc, alpha_1)
                        tlx.local_store(acc_tiles[cid], acc)
                        tlx.barrier_arrive(acc_fulls[cid])
                    accum_cnt += 1

                _, phase = _get_bufidx_phase(i, 1)
                for cid in tl.static_range(0, NUM_MMA_GROUPS):
                    # epilogue
                    tlx.barrier_wait(l_fulls[cid], phase)
                    # Use l[1]/l[1+BLOCK_N] and m[2][2 + BLOCK_N]
                    # to disambigulate from alpha[0]/alpha[BLOCK_N]
                    l = tlx.local_load(l_tiles[cid * BLOCK_N + 1])
                    m = tlx.local_load(m_tiles[cid * BLOCK_N + 2])
                    tlx.barrier_arrive(qk_empties[cid])
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
                    )

                if STAGE & 2:
                    m_i, l_i, accum_cnt_qk = _softmax_inner_loop(
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
                    )

                # prepare l_i for the epilog
                # Use l[1]/l[1+BLOCK_N] and m[2][2 + BLOCK_N]
                # to disambigulate from alpha[0]/alpha[BLOCK_N]
                tlx.local_store(l_tiles[cid * BLOCK_N + 1], l_i[:, None])
                tlx.local_store(m_tiles[cid * BLOCK_N + 2], m_i[:, None])
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
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

                # wait for the K buffer to be populated by the producer
                tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)

                # wait for the Q buffer to be populated by the producer
                tlx.barrier_wait(q_fulls[q_bufIdx], q_phase)

                # -- compute q0 @ k ----
                k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                tlx.barrier_wait(qk_empties[0], q_phase ^ 1)
                # Wait for Q and K scales to be loaded by the load group
                # Use q_scale buffer index 0 for group 0 (q0)
                tlx.barrier_wait(q_scale_fulls[q_bufIdx], q_phase)
                # k_bufIdx is always 0 (accum_cnt_kv % 2 where accum_cnt_kv increments by 2)
                tlx.barrier_wait(kv_scale_fulls[k_bufIdx], k_phase)
                tlx.async_dot_scaled(
                    q_tiles[0],
                    k_tile,
                    qk_tiles[0],
                    q_scale_tiles[0],
                    Q_FP8_FORMAT,
                    kv_scale_tiles[k_bufIdx],
                    K_FP8_FORMAT,
                    use_acc=False,
                    mBarriers=[qk_fulls[0]],
                )

                # -- compute q1 @ k ----
                tlx.barrier_wait(q_fulls[q_bufIdx + NUM_BUFFERS_Q], q_phase)
                tlx.barrier_wait(qk_empties[1], q_phase ^ 1)
                # Wait for Q1 scale (K scale already waited above)
                # Use q_scale buffer index 1 for group 1 (q1)
                tlx.barrier_wait(q_scale_fulls[q_bufIdx + NUM_BUFFERS_Q], q_phase)
                tlx.async_dot_scaled(
                    q_tiles[1],
                    k_tile,
                    qk_tiles[1],
                    q_scale_tiles[1],
                    Q_FP8_FORMAT,
                    kv_scale_tiles[k_bufIdx],
                    K_FP8_FORMAT,
                    use_acc=False,
                    mBarriers=[
                        qk_fulls[1],
                        kv_empties[k_bufIdx],
                        kv_scale_empties[k_bufIdx],
                    ],
                )

                _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

                # -- compute p0 @ v ----
                # wait for the V buffer to be populated by the producer
                tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                tlx.barrier_wait(acc_fulls[0], qk_phase)
                # Wait for V scale before the loop
                # v_bufIdx is always 1 since (accum_cnt_kv + 1) % 2 = 1
                tlx.barrier_wait(kv_scale_fulls[v_bufIdx], v_phase)
                # Use p[0] for cid=0, and p[1] for cid=1
                tlx.barrier_wait(p_fulls[0], qk_phase)
                tlx.async_dot_scaled(
                    p_tiles[0],
                    kv_tiles[v_bufIdx],
                    acc_tiles[0],
                    p_scale_tiles[0],
                    P_FP8_FORMAT,
                    kv_scale_tiles[v_bufIdx],
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

                    # -- compute q0 @ k ----
                    # wait for the K buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                    k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                    _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

                    # Wait for K scale to be loaded by the load group
                    # k_bufIdx is always 0 (accum_cnt_kv % 2 where accum_cnt_kv increments by 2)
                    tlx.barrier_wait(kv_scale_fulls[k_bufIdx], k_phase)
                    tlx.async_dot_scaled(
                        q_tiles[0],
                        k_tile,
                        qk_tiles[0],
                        q_scale_tiles[0],
                        Q_FP8_FORMAT,
                        kv_scale_tiles[k_bufIdx],
                        K_FP8_FORMAT,
                        use_acc=False,
                        mBarriers=[qk_fulls[0]],
                    )

                    # -- compute p1 @ v from the previous iteration----
                    tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
                    tlx.barrier_wait(p_fulls[1], qk_phase_prev)
                    # V scale for v_bufIdx_prev was already loaded
                    # v_bufIdx_prev is always 1, use explicit buffer
                    tlx.async_dot_scaled(
                        p_tiles[1],
                        kv_tiles[v_bufIdx_prev],
                        acc_tiles[1],
                        p_scale_tiles[1],
                        P_FP8_FORMAT,
                        kv_scale_tiles[v_bufIdx_prev],
                        V_FP8_FORMAT,
                        use_acc=acc1_init,
                        mBarriers=[kv_empties[v_bufIdx_prev], kv_scale_empties[v_bufIdx_prev], p_empties[1]],
                    )

                    acc1_init = True

                    # -- compute q1 @ k ----
                    tlx.async_dot_scaled(
                        q_tiles[1],
                        k_tile,
                        qk_tiles[1],
                        q_scale_tiles[1],
                        Q_FP8_FORMAT,
                        kv_scale_tiles[k_bufIdx],
                        K_FP8_FORMAT,
                        use_acc=False,
                        mBarriers=[qk_fulls[1], kv_empties[k_bufIdx], kv_scale_empties[k_bufIdx]],
                    )

                    # -- compute p0 @ v ----
                    # wait for the V buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)

                    tlx.barrier_wait(acc_fulls[0], qk_phase)
                    # Wait for V scale before the loop
                    # v_bufIdx is always 1, use explicit buffer
                    tlx.barrier_wait(kv_scale_fulls[v_bufIdx], v_phase)
                    tlx.barrier_wait(p_fulls[0], qk_phase)
                    tlx.async_dot_scaled(
                        p_tiles[0],
                        kv_tiles[v_bufIdx],
                        acc_tiles[0],
                        p_scale_tiles[0],
                        P_FP8_FORMAT,
                        kv_scale_tiles[v_bufIdx],
                        V_FP8_FORMAT,
                        use_acc=True,
                        mBarriers=[p_empties[0]],
                    )

                tlx.tcgen05_commit(q_empties[q_bufIdx])
                tlx.tcgen05_commit(q_scale_empties[q_bufIdx])
                tlx.tcgen05_commit(q_empties[q_bufIdx + NUM_BUFFERS_Q])
                tlx.tcgen05_commit(q_scale_empties[q_bufIdx + NUM_BUFFERS_Q])
                tlx.tcgen05_commit(acc_empties[0])

                # -- compute p1 @ v ----
                tlx.barrier_wait(acc_fulls[1], qk_phase)
                tlx.barrier_wait(p_fulls[1], qk_phase)
                # Use p[1] for cid=1
                # V scale already waited above, add v_scale_empties to mBarriers
                # v_bufIdx is always 1, use explicit buffer
                tlx.async_dot_scaled(
                    p_tiles[1],
                    kv_tiles[v_bufIdx],
                    acc_tiles[1],
                    p_scale_tiles[1],
                    P_FP8_FORMAT,
                    kv_scale_tiles[v_bufIdx],
                    V_FP8_FORMAT,
                    use_acc=acc1_init,
                    mBarriers=[acc_empties[1], kv_empties[v_bufIdx], kv_scale_empties[v_bufIdx], p_empties[1]],
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

                # load q0
                q_bufIdx, q_phase = _get_bufidx_phase(i, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_bufIdx], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM)
                qo_offset_y_split = qo_offset_y
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx])

                # Use q_scale buffer index 0 for group 0 (q0)
                tlx.barrier_wait(q_scale_empties[0], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_scale_fulls[0], Q_SCALE_BYTES)
                # 5D TMA offset: [batch_head, m_offset, head_offset, 0, 0]
                # off_hz is the combined batch*H + head index
                tlx.async_descriptor_load(
                    desc_q_scale,
                    q_scale_tiles[0],
                    [off_hz, q_scale_m_offset_q0, 0, 0, 0],
                    q_scale_fulls[0],
                )

                # loop over loading k, v
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                # wait for the K buffer to be released by the consumer
                k_empty = tlx.local_view(kv_empties, k_bufIdx)
                tlx.barrier_wait(k_empty, k_phase ^ 1)

                # load K
                k_full = tlx.local_view(kv_fulls, k_bufIdx)
                k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                tlx.barrier_expect_bytes(k_full, K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                # Load K scale - k_bufIdx is always 0, use explicit buffer 0
                tlx.barrier_wait(kv_scale_empties[k_bufIdx], k_phase ^ 1)
                tlx.barrier_expect_bytes(kv_scale_fulls[k_bufIdx], K_SCALE_BYTES)
                # 5D TMA offset: [batch_head, n_offset, head_offset, 0, 0]
                tlx.async_descriptor_load(
                    desc_k_scale,
                    kv_scale_tiles[k_bufIdx],
                    [off_hz, kv_scale_n_offset, 0, 0, 0],
                    kv_scale_fulls[k_bufIdx],
                )

                # load q1
                q_bufIdx += NUM_BUFFERS_Q

                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_bufIdx], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM)
                qo_offset_y_split = qo_offset_y + BLOCK_M_SPLIT
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx])

                # Load Q scale for q1 - use q_scale buffer index 1 for group 1
                tlx.barrier_wait(q_scale_empties[1], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_scale_fulls[1], Q_SCALE_BYTES)
                tlx.async_descriptor_load(
                    desc_q_scale,
                    q_scale_tiles[1],
                    [off_hz, q_scale_m_offset_q1, 0, 0, 0],
                    q_scale_fulls[1],
                )

                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(kv_empties, v_bufIdx)
                tlx.barrier_wait(v_empty, v_phase ^ 1)
                # load V
                v_full = tlx.local_view(kv_fulls, v_bufIdx)
                v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                tlx.barrier_expect_bytes(v_full, V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                # Load V scale - v_bufIdx is always 1, use explicit buffer
                tlx.barrier_wait(kv_scale_empties[v_bufIdx], v_phase ^ 1)
                tlx.barrier_expect_bytes(kv_scale_fulls[v_bufIdx], V_SCALE_BYTES)
                # V_scale 5D TMA offset: [batch_head, head_offset, n_offset, 0, 0]
                # V_scale has shape [B*H, HEAD_DIM//128, N//128, 2, 256] (swapped vs K_scale)
                tlx.async_descriptor_load(
                    desc_v_scale,
                    kv_scale_tiles[v_bufIdx],
                    [off_hz, 0, kv_scale_n_offset, 0, 0],
                    kv_scale_fulls[v_bufIdx],
                )

                kv_offset_y += BLOCK_N
                kv_scale_n_offset += REP_N
                accum_cnt_kv += 2

                for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # wait for the K buffer to be released by the consumer
                    k_empty = tlx.local_view(kv_empties, k_bufIdx)
                    tlx.barrier_wait(k_empty, k_phase ^ 1)
                    # load K
                    k_full = tlx.local_view(kv_fulls, k_bufIdx)
                    k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                    tlx.barrier_expect_bytes(k_full, K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                    tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                    # Load K scale - k_bufIdx is always 0, use explicit buffer 0
                    tlx.barrier_wait(kv_scale_empties[k_bufIdx], k_phase ^ 1)
                    tlx.barrier_expect_bytes(kv_scale_fulls[k_bufIdx], K_SCALE_BYTES)
                    # 5D TMA offset: [batch_head, n_offset, head_offset, 0, 0]
                    # Compute offset based on relative position within this batch-head's N range
                    # kv_offset_y is absolute, base_offset_y is the start of this batch-head
                    tlx.async_descriptor_load(
                        desc_k_scale,
                        kv_scale_tiles[k_bufIdx],
                        [off_hz, kv_scale_n_offset, 0, 0, 0],
                        kv_scale_fulls[k_bufIdx],
                    )

                    v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                    # wait for the V buffer to be released by the consumer
                    v_empty = tlx.local_view(kv_empties, v_bufIdx)
                    tlx.barrier_wait(v_empty, v_phase ^ 1)
                    # load V
                    v_full = tlx.local_view(kv_fulls, v_bufIdx)
                    v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                    tlx.barrier_expect_bytes(v_full, V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                    tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                    # Load V scale - v_bufIdx is always 1, use explicit buffer
                    tlx.barrier_wait(kv_scale_empties[v_bufIdx], v_phase ^ 1)
                    tlx.barrier_expect_bytes(kv_scale_fulls[v_bufIdx], V_SCALE_BYTES)
                    # V_scale 5D TMA offset: [batch_head, head_offset, n_offset, 0, 0]
                    # V_scale has shape [B*H, HEAD_DIM//128, N//128, 2, 256] (swapped vs K_scale)
                    tlx.async_descriptor_load(
                        desc_v_scale,
                        kv_scale_tiles[v_bufIdx],
                        [off_hz, 0, kv_scale_n_offset, 0, 0],
                        kv_scale_fulls[v_bufIdx],
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
                    tlx.fence_async_shared()
                    qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                    tlx.async_descriptor_store(desc_o, o_tiles[cid], [qo_offset_y_split, 0])
                    tlx.async_descriptor_store_wait(0)
                    tlx.barrier_arrive(o_empties[cid])

                tile_idx += num_progs


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
        **config,
    )
    return o
