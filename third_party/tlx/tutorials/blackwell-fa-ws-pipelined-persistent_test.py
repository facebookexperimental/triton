import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    HEAD_DIM = nargs["HEAD_DIM"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]


configs = [
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_MMA_SLICES": 2,
            "GROUP_SIZE_N": 1,
        },
        num_stages=0,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    ),
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_MMA_SLICES": 2,
            "GROUP_SIZE_N": 4,
        },
        num_stages=0,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    ),
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 6,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_MMA_SLICES": 2,
            "GROUP_SIZE_N": 1,
        },
        num_stages=0,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    ),
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 6,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_MMA_SLICES": 2,
            "GROUP_SIZE_N": 4,
        },
        num_stages=0,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    ),
]


def prune_configs_by_hdim(configs, named_args, **kwargs):
    HEAD_DIM = kwargs["HEAD_DIM"]
    STAGE = kwargs["STAGE"]
    target_kv_buffers = 6 if HEAD_DIM == 64 else 3
    target_group_size_n = 4 if STAGE == 3 else 1
    return [
        conf for conf in configs if conf.kwargs.get("NUM_BUFFERS_KV", 0) == target_kv_buffers
        and conf.kwargs.get("GROUP_SIZE_N", 0) == target_group_size_n
    ]


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
def _get_unfused_bwd_loop_bounds(N_CTX, BLOCK_M, STAGE: tl.constexpr):
    if STAGE == 1:
        # First part of STAGE == 3
        lo, hi = 0, N_CTX
    elif STAGE == 2:
        # Second part of STAGE == 3 in this function
        lo, hi = N_CTX, N_CTX
    else:
        tl.static_assert(STAGE == 3)
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
def _split_n(x, SPLIT_FACTOR: tl.constexpr):
    if SPLIT_FACTOR == 1:
        return (x, )
    else:
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        return _split_n(x0, SPLIT_FACTOR // 2) + _split_n(x1, SPLIT_FACTOR // 2)


@triton.jit
def _join_n(xs):
    if len(xs) == 1:
        return xs[0]
    else:
        x0 = _join_n(xs[:len(xs) // 2])
        x1 = _join_n(xs[len(xs) // 2:])
        x = tl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] * 2])
        return x


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
    p_fulls,
    p_tiles,
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
    NUM_MMA_SLICES: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
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
        qks = _split_n(qk, NUM_MMA_SLICES)
        ps = ()
        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
            # prepare p for the v dot
            # Use p[NUM_MMA_SLICES + slice_id] for cid=0, and
            # p[NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id] for cid=1
            p_bufIdx = cid * NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id
            p_i = tl.math.exp2(qks[slice_id])
            tlx.local_store(tlx.local_view(p_tiles, p_bufIdx), p_i.to(out_dtype))
            tlx.barrier_arrive(tlx.local_view(p_fulls, slice_id + cid * NUM_MMA_SLICES))
            ps = ps + (p_i, )

        p = _join_n(ps)
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        accum_cnt_qk += 1

    return m_i, l_i, accum_cnt_qk


@triton.autotune(
    configs=configs,
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "STAGE"],
    prune_configs_by={"early_config_prune": prune_configs_by_hdim},
)
@triton.jit
def _attn_fwd_ws(sm_scale, M,  #
                 Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                 HEAD_DIM: tl.constexpr,  #
                 BLOCK_M: tl.constexpr,  #
                 BLOCK_N: tl.constexpr,  #
                 FP8_OUTPUT: tl.constexpr,  #
                 STAGE: tl.constexpr,  #
                 NUM_BUFFERS_Q: tl.constexpr,  #
                 NUM_BUFFERS_KV: tl.constexpr,  #
                 NUM_BUFFERS_QK: tl.constexpr,  #
                 NUM_MMA_GROUPS: tl.constexpr,  #
                 NUM_MMA_SLICES: tl.constexpr,  #
                 GROUP_SIZE_N: tl.constexpr,  #
                 ):
    tl.static_assert(NUM_MMA_GROUPS == 2)
    tl.static_assert(NUM_BUFFERS_QK == 1)
    tl.static_assert(NUM_BUFFERS_Q == 1)

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2

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

    # allocate TMEM buffers and barriers
    qk_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_N), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem)
    # Shared buffer for QK, P and Alpha, l, and m.
    # A single QK buffer is split evenly:
    #   - First half  : stores P
    #   - Second half  : stores Alpha, l, and m
    #     QK : |                              BLK_M/2 * BLOCK_N * fp32                  |
    #     P:                                                |  BLK_M/2 * BLOCK_N * fp16 |
    #  Alpha : |BLK_M/2*1*fp32|
    #     l :                 |BLK_M/2*1*fp32|
    #     m :                                |BLK_M/2*1*fp32|
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N // NUM_MMA_SLICES),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS * NUM_MMA_SLICES * 2,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    alpha_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    l_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    m_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )

    acc_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem)

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_MMA_SLICES)
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
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            subslice = tlx.subslice(
                                acc_tiles[cid],
                                HEAD_DIM * slice_id // NUM_MMA_SLICES,
                                HEAD_DIM // NUM_MMA_SLICES,
                            )
                            acc = tlx.local_load(subslice)
                            # acc = acc * alpha_1
                            acc = _mul_f32x2(acc, alpha_1)
                            tlx.local_store(subslice, acc)
                        tlx.barrier_arrive(acc_fulls[cid])
                    accum_cnt += 1

                _, phase = _get_bufidx_phase(i, 1)
                for cid in tl.static_range(0, NUM_MMA_GROUPS):
                    # epilogue
                    tlx.barrier_wait(l_fulls[cid], phase)
                    # Use l[1]/l[1+BLOCK_N] and m[2][2 + BLOCK_N]
                    # to disambigulate from alpha[0]/alpha[BLOCK_N]
                    l = tlx.local_load(l_tiles[cid * BLOCK_N + 1])
                    tlx.barrier_arrive(qk_empties[cid])
                    m = tlx.local_load(m_tiles[cid * BLOCK_N + 2])
                    m += tl.math.log2(l)
                    offs_m = (start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT))
                    m_ptrs = M + off_hz * N_CTX + offs_m
                    tl.store(m_ptrs, tl.reshape(m, [BLOCK_M_SPLIT]))

                    tlx.barrier_wait(acc_empties[cid], phase)
                    tlx.barrier_wait(o_empties[cid], phase ^ 1)
                    scale = 1 / l
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        subslice = tlx.subslice(
                            acc_tiles[cid],
                            HEAD_DIM * slice_id // NUM_MMA_SLICES,
                            HEAD_DIM // NUM_MMA_SLICES,
                        )
                        acc = tlx.local_load(subslice)
                        acc = _mul_f32x2(acc, scale)
                        acc = acc.to(tlx.dtype_of(desc_o))
                        subslice_o = tlx.local_slice(
                            o_tiles[cid],
                            [0, HEAD_DIM * slice_id // NUM_MMA_SLICES],
                            [BLOCK_M_SPLIT, HEAD_DIM // NUM_MMA_SLICES],
                        )
                        tlx.local_store(subslice_o, acc)
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
                out_dtype = tlx.dtype_of(desc_v)

                cid = tlx.async_task_replica_id()
                offs_m = (start_m * BLOCK_M) + ((cid * BLOCK_M_SPLIT) + tl.arange(0, BLOCK_M_SPLIT))
                if STAGE & 1:
                    m_i, l_i, accum_cnt_qk = _softmax_inner_loop(
                        qk_fulls,
                        qk_tiles,
                        p_fulls,
                        p_tiles,
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
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM,
                        NUM_MMA_SLICES,
                        NUM_MMA_GROUPS,
                        STAGE=4 - STAGE,
                    )

                if STAGE & 2:
                    m_i, l_i, accum_cnt_qk = _softmax_inner_loop(
                        qk_fulls,
                        qk_tiles,
                        p_fulls,
                        p_tiles,
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
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM,
                        NUM_MMA_SLICES,
                        NUM_MMA_GROUPS,
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
                _, _, lo, hi, _, _ = _compute_offsets(
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
                tlx.async_dot(
                    q_tiles[0],
                    k_tile,
                    qk_tiles[0],
                    use_acc=False,
                    mBarriers=[qk_fulls[0]],
                )

                # -- compute q1 @ k ----
                tlx.barrier_wait(q_fulls[q_bufIdx + NUM_BUFFERS_Q], q_phase)
                tlx.barrier_wait(qk_empties[1], q_phase ^ 1)
                tlx.async_dot(
                    q_tiles[1],
                    k_tile,
                    qk_tiles[1],
                    use_acc=False,
                    mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                )

                _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

                # -- compute p0 @ v ----
                # wait for the V buffer to be populated by the producer
                tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                tlx.barrier_wait(acc_fulls[0], qk_phase)
                # Use p[NUM_MMA_SLICES + slice_id] for cid=0, and
                # p[NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id] for cid=1
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    tlx.barrier_wait(p_fulls[slice_id + 0 * NUM_MMA_SLICES], qk_phase)
                    kv_slice = tlx.local_slice(
                        kv_tiles[v_bufIdx],
                        [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                        [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                    )
                    p_bufIdx = NUM_MMA_SLICES + slice_id
                    tlx.async_dot(
                        p_tiles[p_bufIdx],
                        kv_slice,
                        acc_tiles[0],
                        use_acc=slice_id > 0,
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

                    tlx.async_dot(
                        q_tiles[0],
                        k_tile,
                        qk_tiles[0],
                        use_acc=False,
                        mBarriers=[qk_fulls[0]],
                    )

                    # -- compute p1 @ v from the previous iteration----
                    tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        tlx.barrier_wait(p_fulls[slice_id + 1 * NUM_MMA_SLICES], qk_phase_prev)
                        kv_slice = tlx.local_slice(
                            kv_tiles[v_bufIdx_prev],
                            [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                            [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                        )
                        p_bufIdx = (1 * NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id)
                        use_acc = acc1_init if slice_id == 0 else True
                        mBarriers = ([kv_empties[v_bufIdx_prev]] if slice_id == NUM_MMA_SLICES - 1 else [])
                        tlx.async_dot(
                            p_tiles[p_bufIdx],
                            kv_slice,
                            acc_tiles[1],
                            use_acc=use_acc,
                            mBarriers=mBarriers,
                        )

                    acc1_init = True

                    # -- compute q1 @ k ----
                    tlx.async_dot(
                        q_tiles[1],
                        k_tile,
                        qk_tiles[1],
                        use_acc=False,
                        mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                    )

                    # -- compute p0 @ v ----
                    # wait for the V buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)

                    tlx.barrier_wait(acc_fulls[0], qk_phase)
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        tlx.barrier_wait(p_fulls[slice_id + 0 * NUM_MMA_SLICES], qk_phase)
                        # Use p[1] for cid=0, and p[3] for cid=1
                        kv_slice = tlx.local_slice(
                            kv_tiles[v_bufIdx],
                            [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                            [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                        )
                        p_bufIdx = NUM_MMA_SLICES + slice_id
                        tlx.async_dot(
                            p_tiles[p_bufIdx],
                            kv_slice,
                            acc_tiles[0],
                            use_acc=True,
                        )

                tlx.tcgen05_commit(q_empties[q_bufIdx])
                tlx.tcgen05_commit(q_empties[q_bufIdx + NUM_BUFFERS_Q])
                tlx.tcgen05_commit(acc_empties[0])

                # -- compute p1 @ v ----
                tlx.barrier_wait(acc_fulls[1], qk_phase)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    tlx.barrier_wait(p_fulls[slice_id + NUM_MMA_SLICES], qk_phase)
                    # Use p[1] for cid=0, and p[3] for cid=1
                    kv_slice = tlx.local_slice(
                        kv_tiles[v_bufIdx],
                        [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                        [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                    )
                    p_bufIdx = (1 * NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id)
                    use_acc = acc1_init if slice_id == 0 else True
                    mBarriers = ([acc_empties[1], kv_empties[v_bufIdx]] if slice_id == NUM_MMA_SLICES - 1 else [])
                    tlx.async_dot(
                        p_tiles[p_bufIdx],
                        kv_slice,
                        acc_tiles[1],
                        use_acc=use_acc,
                        mBarriers=mBarriers,
                    )

                accum_cnt_qk += 1
                accum_cnt_kv += 2
                tile_idx += num_progs

        # load
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            for i in range(0, tiles_per_sm):
                # initialize offsets
                _, _, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_idx,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )

                # load q0
                q_bufIdx, q_phase = _get_bufidx_phase(i, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_bufIdx], 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
                qo_offset_y_split = qo_offset_y
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx])

                # loop over loading k, v
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                # wait for the K buffer to be released by the consumer
                k_empty = tlx.local_view(kv_empties, k_bufIdx)
                tlx.barrier_wait(k_empty, k_phase ^ 1)

                # load K
                k_full = tlx.local_view(kv_fulls, k_bufIdx)
                k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                # load q1
                q_bufIdx += NUM_BUFFERS_Q
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_bufIdx], 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
                qo_offset_y_split = qo_offset_y + BLOCK_M_SPLIT
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx])

                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(kv_empties, v_bufIdx)
                tlx.barrier_wait(v_empty, v_phase ^ 1)
                # load V
                v_full = tlx.local_view(kv_fulls, v_bufIdx)
                v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                kv_offset_y += BLOCK_N
                accum_cnt_kv += 2

                for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # wait for the K buffer to be released by the consumer
                    k_empty = tlx.local_view(kv_empties, k_bufIdx)
                    tlx.barrier_wait(k_empty, k_phase ^ 1)
                    # load K
                    k_full = tlx.local_view(kv_fulls, k_bufIdx)
                    k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                    tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                    tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                    v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                    # wait for the V buffer to be released by the consumer
                    v_empty = tlx.local_view(kv_empties, v_bufIdx)
                    tlx.barrier_wait(v_empty, v_phase ^ 1)
                    # load V
                    v_full = tlx.local_view(kv_fulls, v_bufIdx)
                    v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                    tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                    tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                    kv_offset_y += BLOCK_N
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


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    desc_q,
    k,
    v,
    sm_scale,  #
    desc_do,  #
    desc_dq,
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    off_bh,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        q = desc_q.load([(off_bh + curr_m).to(tl.int32), 0])
        qT = tl.trans(q)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = desc_do.load([(off_bh + curr_m).to(tl.int32), 0])
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Compute dq = tl.dot(tl.trans(dsT), k)
        dq = tl.dot(tl.trans(dsT), k) * LN2
        desc_dq.atomic_add([(off_bh + curr_m).to(tl.int32), 0], dq)
        # Increment pointers.
        curr_m += step_m

    return dk, dv


def _bwd_host_descriptor_pre_hook(nargs):
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    HEAD_DIM = nargs["HEAD_DIM"]
    nargs["desc_q"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_do"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_dq"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_dv"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_dk"].block_shape = [BLOCK_N1, HEAD_DIM]


configs_bwd = [
    triton.Config(
        {
            "BLOCK_M1": 32,
            "BLOCK_N1": 128,
            "BLOCK_M2": 128,
            "BLOCK_N2": 32,
        },
        num_warps=4,
        num_stages=1,
        pre_hook=_bwd_host_descriptor_pre_hook,
    )
]


@triton.autotune(configs=configs_bwd, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd_triton(
    desc_q,
    desc_k,
    desc_v,
    sm_scale,  #
    desc_do,  #
    desc_dq,
    desc_dk,
    desc_dv,  #
    M,
    D,
    # shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
):
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    off_bh = ((stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)) // stride_tok
    pid = tl.program_id(0)

    # offset pointers for batch/head
    M += off_chz
    D += off_chz

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    start_n = pid * BLOCK_N1
    start_m = 0

    # load K and V: they stay in SRAM throughout the inner loop.
    k = desc_k.load([(off_bh + start_n).to(tl.int32), 0])
    v = desc_v.load([(off_bh + start_n).to(tl.int32), 0])
    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        desc_q, k, v, sm_scale,  #
        desc_do,  #
        desc_dq, M, D,  #
        stride_tok, stride_d,  #
        off_bh, H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False,  #
    )

    desc_dv.store(
        [(off_bh + start_n).to(tl.int32), 0],
        dv.to(tlx.dtype_of(desc_dv)),
    )

    # Write back dK.
    dk *= sm_scale
    desc_dk.store(
        [(off_bh + start_n).to(tl.int32), 0],
        dk.to(tlx.dtype_of(desc_dk)),
    )


@triton.jit
def bwd_calculate_offsets(
    tile_idx,
    n_tile_num,
    stride_z,
    stride_h,
    stride_tok,
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
):
    bhid = tile_idx // n_tile_num
    pid = tile_idx % n_tile_num
    off_chz = (bhid * N_CTX).to(tl.int64)
    off_bh = ((stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)) // stride_tok
    start_n = pid * BLOCK_N1
    start_m = 0
    num_steps = (N_CTX - start_m) // BLOCK_M1
    return off_chz, off_bh, start_m, start_n, num_steps


def _bwd_host_descriptor_pre_hook_tlx(nargs):
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    HEAD_DIM = nargs["HEAD_DIM"]
    EPILOGUE_SUBTILE = nargs["EPILOGUE_SUBTILE"]

    nargs["desc_q"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_do"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_dq"].block_shape = [BLOCK_M1, HEAD_DIM // EPILOGUE_SUBTILE]
    nargs["desc_dv"].block_shape = [BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE]
    nargs["desc_dk"].block_shape = [BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE]


configs_bwd_tlx = [
    triton.Config(
        {
            "BLOCK_M1": 128,
            "BLOCK_N1": 128,
            "BLOCK_M2": 128,
            "BLOCK_N2": 128,
            "NUM_BUFFERS_KV": 1,
            "NUM_BUFFERS_Q": 2,
            "NUM_BUFFERS_DO": 1,
            "NUM_BUFFERS_DS": 1,
            "NUM_BUFFERS_TMEM": 1,
            "EPILOGUE_SUBTILE": 4,
        },
        num_warps=4,
        num_stages=1,
        pre_hook=_bwd_host_descriptor_pre_hook_tlx,
    )
]


# TODO: Unused. Fix layout issue inside TLX.
@triton.jit
def _bwd_compute_inner_loop(
    start_n,
    qk_fulls,
    qk_tiles,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_fulls,
    dp_tiles,
    ds_tiles,
    ds_fulls,
    M,
    D,
    curr_m,
    blk_idx,
    step_m,
    do_out_dtype,
    q_out_dtype,
    N_CTX,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    STAGE: tl.constexpr,
):
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    lo, hi = _get_unfused_bwd_loop_bounds(N_CTX, BLOCK_M1, STAGE)
    num_steps = (hi - lo) // BLOCK_M1
    for _ in range(num_steps):
        tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
        ds_buf_id, _ = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)

        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)

        # wait for qkT = tl.dot(k, qT)
        tlx.barrier_wait(tlx.local_view(qk_fulls, tmem_buf_id), tmem_phase)
        qkT = tlx.local_load(tlx.local_view(qk_tiles, tmem_buf_id))
        tlx.barrier_arrive(tlx.local_view(qk_empties, tmem_buf_id))

        pT = tl.math.exp2(qkT - m[None, :])
        if STAGE == 1:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)

        # ppT *= qk_scale
        ppT = pT
        ppT = ppT.to(do_out_dtype)
        tlx.local_store(tlx.local_view(p_tiles, tmem_buf_id), ppT)
        tlx.barrier_arrive(tlx.local_view(p_fulls, tmem_buf_id))

        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)

        # Wait for dpT = tl.dot(v, tl.trans(do))
        tlx.barrier_wait(tlx.local_view(dp_fulls, tmem_buf_id), tmem_phase)
        dpT = tlx.local_load(tlx.local_view(dp_tiles, tmem_buf_id))
        # No need to release dP, as dP uses the same tmem as dQ
        # in the same iteration. Release dQ instead later.
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(q_out_dtype)
        tlx.local_store(tlx.local_view(ds_tiles, ds_buf_id), dsT)
        tlx.fence_async_shared()
        tlx.barrier_arrive(tlx.local_view(ds_fulls, ds_buf_id))
        curr_m += step_m
        blk_idx += 1
    return curr_m, blk_idx


@triton.autotune(configs=configs_bwd_tlx, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd_ws(
    desc_q,
    desc_k,
    desc_v,
    sm_scale,  #
    desc_do,  #
    desc_dq,
    desc_dk,
    desc_dv,  #
    M,
    D,
    # shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    H,
    Z,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    STAGE: tl.constexpr,
):
    # original grid
    #   triton.cdiv(q.shape[2], META["BLOCK_N1"]),
    #   1,
    #   q.shape[0] * q.shape[1],
    n_tile_num = tl.cdiv(N_CTX, BLOCK_N1)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    # allocate smem buffers
    k_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    v_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS_KV)
    q_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    do_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_do), NUM_BUFFERS_DO)

    # Use SMEM for dsT
    ds_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tlx.dtype_of(desc_q), NUM_BUFFERS_DS)

    # allocate barriers for smem buffers
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    ds_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    # allocate tmem buffers
    qk_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem)
    p_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_do),
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    dp_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
    )

    dq_tiles = tlx.local_alloc(
        (BLOCK_M1, HEAD_DIM),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=dp_tiles,
    )
    dv_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem)
    dk_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem)

    # allocate barriers for tmem buffers
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    with tlx.async_tasks():
        # reduction
        with tlx.async_task("default"):
            blk_idx = 0
            for i in range(0, tiles_per_sm):
                off_chz, off_bh, start_m, _, num_steps = bwd_calculate_offsets(
                    tile_idx,
                    n_tile_num,
                    stride_z,
                    stride_h,
                    stride_tok,
                    H,
                    N_CTX,
                    BLOCK_M1,
                    BLOCK_N1,
                )
                curr_m = start_m
                step_m = BLOCK_M1
                for _ in range(num_steps):
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

                    # wait for dq = tl.dot(tl.trans(dsT), k)
                    tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                    slice_size: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
                    for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                        dq_slice = tlx.local_slice(
                            dq_tiles[tmem_buf_id],
                            [0, slice_id * slice_size],
                            [BLOCK_M1, slice_size],
                        )
                        dq = tlx.local_load(dq_slice)
                        dq = dq * LN2
                        desc_dq.atomic_add([(off_bh + curr_m).to(tl.int32), slice_id * slice_size], dq)

                    # release dq
                    tlx.barrier_arrive(dq_empties[tmem_buf_id])
                    # Increment pointers.
                    curr_m += step_m
                    blk_idx += 1
                tile_idx += num_progs

        # compute
        with tlx.async_task(num_warps=8, registers=192, replicate=1):
            blk_idx = 0
            for i in range(0, tiles_per_sm):
                off_chz, off_bh, _, start_n, _ = bwd_calculate_offsets(
                    tile_idx,
                    n_tile_num,
                    stride_z,
                    stride_h,
                    stride_tok,
                    H,
                    N_CTX,
                    BLOCK_M1,
                    BLOCK_N1,
                )
                # offset pointers for batch/head
                M_updated = M + off_chz
                D_updated = D + off_chz
                curr_m = 0
                step_m = BLOCK_M1
                do_out_dtype = tlx.dtype_of(desc_do)
                q_out_dtype = tlx.dtype_of(desc_q)
                if STAGE & 1:
                    offs_n = start_n + tl.arange(0, BLOCK_N1)
                    lo, hi = _get_unfused_bwd_loop_bounds(N_CTX, BLOCK_M1, STAGE=4 - STAGE)
                    num_steps = (hi - lo) // BLOCK_M1
                    for _ in range(num_steps):
                        tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
                        ds_buf_id, ds_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)

                        offs_m = curr_m + tl.arange(0, BLOCK_M1)
                        m = tl.load(M_updated + offs_m)

                        # wait for qkT = tl.dot(k, qT)
                        tlx.barrier_wait(tlx.local_view(qk_fulls, tmem_buf_id), tmem_phase)
                        qkT = tlx.local_load(tlx.local_view(qk_tiles, tmem_buf_id))
                        tlx.barrier_arrive(tlx.local_view(qk_empties, tmem_buf_id))

                        pT = tl.math.exp2(qkT - m[None, :])
                        if STAGE == 3:
                            mask = offs_m[None, :] >= offs_n[:, None]
                            pT = tl.where(mask, pT, 0.0)

                        # ppT *= qk_scale
                        ppT = pT
                        ppT = ppT.to(do_out_dtype)
                        tlx.local_store(tlx.local_view(p_tiles, tmem_buf_id), ppT)
                        tlx.barrier_arrive(tlx.local_view(p_fulls, tmem_buf_id))

                        # D (= delta) is pre-divided by ds_scale.
                        Di = tl.load(D_updated + offs_m)

                        # Wait for dpT = tl.dot(v, tl.trans(do))
                        tlx.barrier_wait(tlx.local_view(dp_fulls, tmem_buf_id), tmem_phase)
                        dpT = tlx.local_load(tlx.local_view(dp_tiles, tmem_buf_id))
                        # No need to release dP, as dP uses the same tmem as dQ
                        # in the same iteration. Release dQ instead later.
                        dsT = pT * (dpT - Di[None, :])
                        dsT = dsT.to(q_out_dtype)
                        tlx.local_store(tlx.local_view(ds_tiles, ds_buf_id), dsT)
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(tlx.local_view(ds_fulls, ds_buf_id))
                        curr_m += step_m
                        blk_idx += 1
                # TODO: Add the STAGE & 2 handling when we can determine bounds to divide
                # the work across two loops, based on optimizing out the mask.
                # epilogue
                kv_buf_id, kv_phase = _get_bufidx_phase(i, NUM_BUFFERS_KV)

                tlx.barrier_wait(dv_fulls[kv_buf_id], kv_phase)
                slice_size: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
                for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                    dv_slice = tlx.local_slice(
                        dv_tiles[kv_buf_id],
                        [0, slice_id * slice_size],
                        [BLOCK_N1, slice_size],
                    )
                    dv = tlx.local_load(dv_slice)
                    desc_dv.store(
                        [(off_bh + start_n).to(tl.int32), slice_id * slice_size],
                        dv.to(tlx.dtype_of(desc_dv)),
                    )
                tlx.barrier_arrive(dv_empties[kv_buf_id])
                tlx.barrier_wait(dk_fulls[kv_buf_id], kv_phase)
                for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                    dk_slice = tlx.local_slice(
                        dk_tiles[kv_buf_id],
                        [0, slice_id * slice_size],
                        [BLOCK_N1, slice_size],
                    )
                    dk = tlx.local_load(dk_slice)
                    dk *= sm_scale
                    desc_dk.store(
                        [(off_bh + start_n).to(tl.int32), slice_id * slice_size],
                        dk.to(tlx.dtype_of(desc_dk)),
                    )
                tlx.barrier_arrive(dk_empties[kv_buf_id])
                tile_idx += num_progs

        # mma
        with tlx.async_task(num_warps=1, registers=48):
            blk_idx = 0
            for i in range(0, tiles_per_sm):
                _, _, start_m, _, num_steps = bwd_calculate_offsets(
                    tile_idx,
                    n_tile_num,
                    stride_z,
                    stride_h,
                    stride_tok,
                    H,
                    N_CTX,
                    BLOCK_M1,
                    BLOCK_N1,
                )

                kv_buf_id, kv_phase = _get_bufidx_phase(i, NUM_BUFFERS_KV)
                tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)
                tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)

                # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
                tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
                curr_m = start_m
                step_m = BLOCK_M1

                # -----------------------------------------------------------
                # Prolog
                #
                # 1. qkT = tl.dot(k, qT)
                # 2. dpT = tl.dot(v, tl.trans(do))
                # 3. dv += tl.dot(ppT, do)
                # -----------------------------------------------------------

                q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

                # Compute qkT = tl.dot(k, qT)
                tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
                tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
                qT = tlx.local_trans(q_tiles[q_buf_id])
                tlx.async_dot(
                    k_tiles[kv_buf_id],
                    qT,
                    qk_tiles[tmem_buf_id],
                    use_acc=False,
                    mBarriers=[qk_fulls[tmem_buf_id]],
                )

                # Compute dpT = tl.dot(v, tl.trans(do))
                tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                # As dP uses the same tmem as dQ, wait for dQ release.
                tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
                doT = tlx.local_trans(do_tiles[do_buf_id])
                tlx.async_dot(
                    v_tiles[kv_buf_id],
                    doT,
                    dp_tiles[tmem_buf_id],
                    use_acc=False,
                    mBarriers=[dp_fulls[tmem_buf_id]],
                )

                # Compute dv += tl.dot(ppT, do)
                tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
                tlx.barrier_wait(dv_empties[kv_buf_id], kv_phase ^ 1)
                tlx.async_dot(
                    p_tiles[tmem_buf_id],
                    do_tiles[do_buf_id],
                    dv_tiles[kv_buf_id],
                    use_acc=False,
                    mBarriers=[do_empties[do_buf_id]],
                )
                blk_idx += 1
                # -----------------------------------------------------------
                # Main loop
                # 1. qkT = tl.dot(k, qT)
                # 2. dq = tl.dot(tl.trans(dsT), k) from previous iteration
                # 3. dk += tl.dot(dsT, tl.trans(qT)) from previous iteration
                # 4. dpT = tl.dot(v, tl.trans(do))
                # 5. dv += tl.dot(ppT, do)
                # -----------------------------------------------------------
                tlx.barrier_wait(dk_empties[kv_buf_id], kv_phase ^ 1)
                for i in range(1, num_steps):
                    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
                    # Compute qkT = tl.dot(k, qT)
                    tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
                    tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
                    qT = tlx.local_trans(q_tiles[q_buf_id])
                    tlx.async_dot(
                        k_tiles[kv_buf_id],
                        qT,
                        qk_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[qk_fulls[tmem_buf_id]],
                    )

                    prev_blk_idx = blk_idx - 1
                    q_buf_id_prev, _ = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                    tmem_buf_id_prev, tmem_phase_prev = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
                    ds_buf_id_prev, ds_phase_prev = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)

                    # Compute dq = tl.dot(tl.trans(dsT), k) from previous iteration
                    tlx.barrier_wait(ds_fulls[tmem_buf_id_prev], ds_phase_prev)
                    tlx.barrier_wait(dq_empties[tmem_buf_id_prev], tmem_phase_prev ^ 1)
                    dsT_view = tlx.local_trans(ds_tiles[ds_buf_id_prev])
                    tlx.async_dot(
                        dsT_view,
                        k_tiles[kv_buf_id],
                        dq_tiles[tmem_buf_id_prev],
                        use_acc=False,
                        mBarriers=[
                            dq_fulls[tmem_buf_id_prev],
                        ],
                    )

                    # Compute dk += tl.dot(dsT, tl.trans(qT)) from previous iteration
                    tlx.async_dot(
                        ds_tiles[ds_buf_id_prev],
                        q_tiles[q_buf_id_prev],
                        dk_tiles[kv_buf_id],
                        use_acc=(i - 1) > 0,
                        mBarriers=[
                            q_empties[q_buf_id_prev],
                        ],
                    )

                    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                    # Compute dpT = tl.dot(v, tl.trans(do))
                    tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                    # As dP uses the same tmem as dQ, wait for dQ release.
                    tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
                    doT = tlx.local_trans(do_tiles[do_buf_id])
                    tlx.async_dot(
                        v_tiles[kv_buf_id],
                        doT,
                        dp_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[dp_fulls[tmem_buf_id]],
                    )

                    # Compute dv += tl.dot(ppT, do)
                    tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
                    tlx.async_dot(
                        p_tiles[tmem_buf_id],
                        do_tiles[do_buf_id],
                        dv_tiles[kv_buf_id],
                        use_acc=True,
                        mBarriers=[do_empties[do_buf_id]],
                    )
                    blk_idx += 1

                tlx.tcgen05_commit(dv_fulls[kv_buf_id])

                # -----------------------------------------------------------
                # Epilog
                # 4. dk += tl.dot(dsT, tl.trans(qT))
                # 5. dq = tl.dot(tl.trans(dsT), k)
                # -----------------------------------------------------------
                prev_blk_idx = blk_idx - 1
                q_buf_id, _ = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                tmem_buf_id, tmem_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
                ds_buf_id, ds_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)
                # Compute dk += tl.dot(dsT, tl.trans(qT))
                tlx.barrier_wait(ds_fulls[tmem_buf_id], ds_phase)
                tlx.async_dot(
                    ds_tiles[ds_buf_id],
                    q_tiles[q_buf_id],
                    dk_tiles[kv_buf_id],
                    use_acc=num_steps > 1,
                    mBarriers=[q_empties[q_buf_id], dk_fulls[tmem_buf_id]],
                )

                # Compute dq = tl.dot(tl.trans(dsT), k)
                tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
                dsT_view = tlx.local_trans(ds_tiles[ds_buf_id])
                tlx.async_dot(
                    dsT_view,
                    k_tiles[kv_buf_id],
                    dq_tiles[tmem_buf_id],
                    use_acc=False,
                    mBarriers=[
                        dq_fulls[tmem_buf_id],
                    ],
                )
                tlx.tcgen05_commit(k_empties[kv_buf_id])
                tile_idx += num_progs

        # load
        with tlx.async_task(num_warps=1, registers=88):
            blk_idx = 0
            for i in range(0, tiles_per_sm):
                _, off_bh, start_m, start_n, num_steps = bwd_calculate_offsets(
                    tile_idx,
                    n_tile_num,
                    stride_z,
                    stride_h,
                    stride_tok,
                    H,
                    N_CTX,
                    BLOCK_M1,
                    BLOCK_N1,
                )
                # Load K
                kv_buf_id, kv_phase = _get_bufidx_phase(i, NUM_BUFFERS_KV)
                tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
                tlx.barrier_expect_bytes(k_fulls[kv_buf_id], 2 * BLOCK_N1 * HEAD_DIM)  # float16
                tlx.async_descriptor_load(
                    desc_k,
                    k_tiles[kv_buf_id],
                    [(off_bh + start_n).to(tl.int32), 0],
                    k_fulls[kv_buf_id],
                )

                # Load Q
                curr_m = start_m
                step_m = BLOCK_M1
                q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_buf_id], 2 * BLOCK_M1 * HEAD_DIM)
                tlx.async_descriptor_load(
                    desc_q,
                    q_tiles[q_buf_id],
                    [(off_bh + curr_m).to(tl.int32), 0],
                    q_fulls[q_buf_id],
                )

                # Note: No need for v_empties because k is finished after v.
                # Load V
                tlx.barrier_expect_bytes(v_fulls[kv_buf_id], 2 * BLOCK_N1 * HEAD_DIM)  # float16
                tlx.async_descriptor_load(
                    desc_v,
                    v_tiles[kv_buf_id],
                    [(off_bh + start_n).to(tl.int32), 0],
                    v_fulls[kv_buf_id],
                )

                # Load dO
                do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                tlx.barrier_expect_bytes(do_fulls[do_buf_id], 2 * BLOCK_M1 * HEAD_DIM)
                tlx.async_descriptor_load(
                    desc_do,
                    do_tiles[do_buf_id],
                    [(off_bh + curr_m).to(tl.int32), 0],
                    do_fulls[do_buf_id],
                )
                curr_m += step_m
                blk_idx += 1

                for _ in range(1, num_steps):
                    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                    # Load Q
                    tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                    tlx.barrier_expect_bytes(q_fulls[q_buf_id], 2 * BLOCK_M1 * HEAD_DIM)
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_buf_id],
                        [(off_bh + curr_m).to(tl.int32), 0],
                        q_fulls[q_buf_id],
                    )

                    # Load dO
                    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                    tlx.barrier_expect_bytes(do_fulls[do_buf_id], 2 * BLOCK_M1 * HEAD_DIM)
                    tlx.async_descriptor_load(
                        desc_do,
                        do_tiles[do_buf_id],
                        [(off_bh + curr_m).to(tl.int32), 0],
                        do_fulls[do_buf_id],
                    )
                    curr_m += step_m
                    blk_idx += 1

                tile_idx += num_progs


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        stage = 3 if causal else 1

        o = torch.empty_like(q)
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            q,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        if q.dtype == torch.float8_e5m2:
            desc_v = TensorDescriptor(
                v,
                shape=[HEAD_DIM_K, y_dim],
                strides=[q.shape[2], 1],
                block_shape=dummy_block,
            )
        else:
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
        _attn_fwd_ws[grid](
            sm_scale,
            M,  #
            q.shape[0],
            q.shape[1],  #
            desc_q,
            desc_k,
            desc_v,
            desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM,  #
        )

        dummy_block = [1, 1]
        HEAD_DIM = ctx.HEAD_DIM
        desc_k = TensorDescriptor(
            arg_k,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_q = TensorDescriptor(
            q,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_do = TensorDescriptor(
            do,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_dq = TensorDescriptor(
            dq,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_dk = TensorDescriptor(
            dk,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_dv = TensorDescriptor(
            dv,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        def grid_persistent(meta):
            return (
                min(
                    NUM_SMS,
                    # tiles along N (K/V)
                    triton.cdiv(N_CTX, meta["BLOCK_N1"]) * BATCH * N_HEAD,
                ),
                1,
                1,
            )

        stage = 3 if ctx.causal else 1

        _attn_bwd_ws[grid_persistent](
            desc_q, desc_k, desc_v, ctx.sm_scale, desc_do, desc_dq, desc_dk, desc_dv,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, BATCH,  #
            N_CTX,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            STAGE=stage,  #
        )

        return dq, dk, dv, None, None, None, None


attention = _attention.apply


@pytest.mark.skipif(
    not is_blackwell(),
    reason="Requires Hopper GPU",
)
@pytest.mark.parametrize("Z", [8])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("N_CTX", [1024])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"])
@pytest.mark.parametrize("causal", [True, False])
def test_op(Z, H, N_CTX, HEAD_DIM, mode, provider, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)
    if mode == "bwd":
        dout = torch.randn_like(q)
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, sm_scale, causal).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return

    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if (torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a"):
        rtol = 1e-2
    torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func as flash_attn_func, )

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = False
BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
print(N_HEADS)
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=["triton-fp16"] + (["flash"] if HAS_FLASH else []),
            line_names=["Triton [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="TFLOPS",
            plot_name=f"fused-attention-ws-pipelined-persistent-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "mode": mode,
            },
        ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn(
            (BATCH, N_CTX, 3, H, HEAD_DIM),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        fn = lambda: flash_attn_func(qkv)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    if is_blackwell():
        print("Running benchmarks...")
        bench_flash_attention.run(print_data=True)
    else:
        print("Skipping benchmarks, no Blackwell GPU found.")
