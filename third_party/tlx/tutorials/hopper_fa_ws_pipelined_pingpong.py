import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
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
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_BUFFERS': 2, 'NUM_MMA_WARPS': 8, 'NUM_MMA_GROUPS': 2},
                  num_stages=1, num_warps=4, pre_hook=_host_descriptor_pre_hook),
]


@triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
@triton.jit
def _attn_fwd_ws_pipelined_pingpong(sm_scale, M,  #
                                    Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                                    HEAD_DIM: tl.constexpr,  #
                                    BLOCK_M: tl.constexpr,  #
                                    BLOCK_N: tl.constexpr,  #
                                    FP8_OUTPUT: tl.constexpr,  #
                                    NUM_BUFFERS: tl.constexpr,  #
                                    NUM_MMA_WARPS: tl.constexpr,  #
                                    NUM_MMA_GROUPS: tl.constexpr,  #
                                    ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    # allocate buffers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS)
    k_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS)
    v_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS)

    # allocate barriers
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS)
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)

    with tlx.async_tasks(exclusive=True):
        # producer group
        with tlx.async_task("default"):
            # initialize offsets
            start_m = tl.program_id(0)
            off_hz = tl.program_id(1)
            off_z = off_hz // H
            off_h = off_hz % H
            offset_y = off_z * (N_CTX * H) + off_h * N_CTX
            qo_offset_y = offset_y + start_m * BLOCK_M
            lo, hi = 0, N_CTX
            kv_offset_y = offset_y + lo

            # load q: it will stay in SRAM throughout
            for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                tlx.barrier_expect_bytes(q_fulls[cid], 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
                qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                tlx.async_descriptor_load(desc_q, q_tiles[cid], [qo_offset_y_split, 0], q_fulls[cid])

            # loop over loading k, v
            kv_phase = 0
            acc_cnt = 0
            for _ in tl.range(lo, hi, BLOCK_N):
                buf_id = acc_cnt % NUM_BUFFERS
                # buffers in a row share the same phase
                kv_phase = kv_phase ^ (buf_id == 0)

                # wait for the K buffer to be released by the consumer
                tlx.barrier_wait(k_empties[buf_id], kv_phase)
                # load K
                tlx.barrier_expect_bytes(k_fulls[buf_id], 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_k, k_tiles[buf_id], [kv_offset_y, 0], k_fulls[buf_id])

                # wait for the V buffer to be released by the consumer
                tlx.barrier_wait(v_empties[buf_id], kv_phase)
                # load V
                tlx.barrier_expect_bytes(v_fulls[buf_id], 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_v, v_tiles[buf_id], [kv_offset_y, 0], v_fulls[buf_id])

                kv_offset_y += BLOCK_N
                acc_cnt += 1

        # consumer group
        with tlx.async_task(num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS, registers=232, replicate=NUM_MMA_GROUPS):
            # initialize pointer to m and l
            m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)

            # load scales
            qk_scale = sm_scale
            qk_scale *= 1.44269504  # 1/log(2)

            # wait for the Q buffer to be populated by the producer
            cid: tl.constexpr = tlx.async_task_replica_id()
            tlx.barrier_wait(q_fulls[cid], 0)

            lo, hi = 0, N_CTX
            k_phase = 0
            v_phase = 1
            k_buf_id = 0
            v_buf_id = 0

            # wait for the K[0] buffer to be populated by the producer
            tlx.barrier_wait(k_fulls[k_buf_id], k_phase)

            # -- compute qk[0] ----
            k_tile = tlx.local_trans(k_tiles[k_buf_id])

            if cid == 0:
                # Consumer 0 waits for Consumer 1 to reach synchronization point at barrier 9.
                tlx.named_barrier_wait(9, 256)
            else:
                # Consumer 1 signals its arrival at barrier 9.
                tlx.named_barrier_arrive(9, 256)
                # Then waits at barrier 10 until Consumer 0 finishes issuing its async_dot.
                tlx.named_barrier_wait(10, 256)

            qk = tlx.async_dot(q_tiles[cid], k_tile)

            if cid == 0:
                # After issuing async_dot, Consumer 0 signals barrier 10 to unblock Consumer 1.
                tlx.named_barrier_arrive(10, 256)
            else:
                # Consumer 1 signals barrier 9 to unblock Consumer 0.
                tlx.named_barrier_arrive(9, 256)

            # wait for the MMA to complete
            qk = tlx.async_dot_wait(0, qk)
            # release the K buffer
            tlx.barrier_arrive(k_empties[k_buf_id], 1)

            # -- compute m_i and l_i ----
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            # -- compute correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            # -- update output accumulator[0] --
            acc = acc * alpha[:, None]
            l_ij = tl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            acc_cnt = 1

            # loop over k, v and update accumulator
            for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                k_buf_id = acc_cnt % NUM_BUFFERS
                # buffers in a row share the same phase
                k_phase = k_phase ^ (k_buf_id == 0)

                # wait for the K buffer to be populated by the producer
                tlx.barrier_wait(k_fulls[k_buf_id], k_phase)

                # compute qk for the current iteration
                k_tile = tlx.local_trans(k_tiles[k_buf_id])

                if cid == 0:
                    # Consumer 0 waits for Consumer 1 to reach synchronization point at barrier 9.
                    tlx.named_barrier_wait(9, 256)
                else:
                    # Then waits at barrier 10 until Consumer 0 finishes issuing its async_dot.
                    tlx.named_barrier_wait(10, 256)

                qk = tlx.async_dot(q_tiles[cid], k_tile)

                if cid == 0:
                    # After issuing async_dot, Consumer 0 signals barrier 10 to unblock Consumer 1.
                    tlx.named_barrier_arrive(10, 256)
                else:
                    # Consumer 1 signals barrier 9 to unblock Consumer 0.
                    tlx.named_barrier_arrive(9, 256)

                # compute pv from the previous iteration
                # wait for the previous V buffer to be populated by the producer
                v_buf_id = (acc_cnt - 1) % NUM_BUFFERS
                v_phase = v_phase ^ (v_buf_id == 0)
                tlx.barrier_wait(v_fulls[v_buf_id], v_phase)
                # prepare p and v for the dot
                p = p.to(tlx.dtype_of(desc_k))
                acc = tlx.async_dot(p, v_tiles[v_buf_id], acc)

                # wait for the current qk MMA to complete
                qk = tlx.async_dot_wait(1, qk)
                # release the K buffer
                tlx.barrier_arrive(k_empties[k_buf_id], 1)

                # -- compute m_i and l_i ----
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
                p = tl.math.exp2(qk)
                # -- compute correction factor
                alpha = tl.math.exp2(m_i - m_ij)
                l_ij = tl.sum(p, 1)
                # update m_i and l_i
                l_i = l_i * alpha + l_ij
                m_i = m_ij

                # -- update output accumulator --
                # wait for the previous pv MMA to complete
                acc = tlx.async_dot_wait(0, acc)
                # release the V buffer
                tlx.barrier_arrive(v_empties[v_buf_id], 1)
                acc = acc * alpha[:, None]
                acc_cnt += 1

            # compute pv from the last iteration
            # wait for the V buffer to be populated by the producer
            v_buf_id = (acc_cnt - 1) % NUM_BUFFERS
            v_phase = v_phase ^ (v_buf_id == 0)
            tlx.barrier_wait(v_fulls[v_buf_id], v_phase)
            # prepare p and v for the dot
            p = p.to(tlx.dtype_of(desc_k))
            acc = tlx.async_dot(p, v_tiles[v_buf_id], acc)
            # wait for the MMA to complete
            acc = tlx.async_dot_wait(0, acc)
            # release the V buffer
            tlx.barrier_arrive(v_empties[v_buf_id], 1)

            # epilogue
            start_m = tl.program_id(0)
            off_hz = tl.program_id(1)
            off_z = off_hz // H
            off_h = off_hz % H
            offset_y = off_z * (N_CTX * H) + off_h * N_CTX
            qo_offset_y = offset_y + start_m * BLOCK_M
            qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
            m_i += tl.math.log2(l_i)
            acc = acc / l_i[:, None]
            offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
            m_ptrs = M + off_hz * N_CTX + offs_m
            tl.store(m_ptrs, m_i)
            desc_o.store([qo_offset_y_split, 0], acc.to(tlx.dtype_of(desc_o)))


@triton.jit
# Triton TR001: backward preprocess uses the fixed 128-row FA tile from the wrapper.
def _attn_bwd_preprocess(  # noqa: TR001
    O,
    DO,
    Delta,
    DQ,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    mask = (off_m[:, None] < N_CTX) & (off_n[None, :] < HEAD_DIM)
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :],
        mask=mask,
        other=0.0,
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=off_m < N_CTX)
    tl.store(
        DQ + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :],
        tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32),
        mask=mask,
    )


def _host_descriptor_bwd_pre_hook(nargs):
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    NUM_MMA_GROUPS_BWD = nargs["NUM_MMA_GROUPS_BWD"]
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_do"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_dq"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_dk"].block_shape = [BLOCK_N // NUM_MMA_GROUPS_BWD, HEAD_DIM]
    nargs["desc_dv"].block_shape = [BLOCK_N // NUM_MMA_GROUPS_BWD, HEAD_DIM]


configs_bwd = [
    triton.Config(
        {
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 2,
            "NUM_BUFFERS_DQ_STORE": 2,
            "NUM_MMA_GROUPS_BWD": 2,
            "NUM_MMA_WARPS_BWD": 8,
            "BWD_REGISTERS": 240,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=_host_descriptor_bwd_pre_hook,
    ),
]


@triton.autotune(configs=configs_bwd, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd_tlx(
    desc_q,
    desc_k,
    desc_v,
    sm_scale,
    desc_do,
    desc_dq,
    desc_dk,
    desc_dv,
    M,
    D,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DQ_STORE: tl.constexpr,
    NUM_MMA_GROUPS_BWD: tl.constexpr,
    NUM_MMA_WARPS_BWD: tl.constexpr,
    BWD_REGISTERS: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int32)
    pid = tl.program_id(0)

    M += off_chz
    D += off_chz

    row_base = off_chz
    start_n = pid * BLOCK_N
    global_start_n = row_base + start_n
    num_steps = N_CTX // BLOCK_M
    CID_BLOCK_N: tl.constexpr = BLOCK_N // NUM_MMA_GROUPS_BWD

    kv_atom_layout: tl.constexpr = tlx.nv_mma_shared_layout_encoding(
        (CID_BLOCK_N, HEAD_DIM // NUM_MMA_GROUPS_BWD),
        [1, 0],
        tlx.dtype_of(desc_k),
        [1, 1],
        [1, 1],
        [1, 0],
        False,
        True,
    )
    kv_smem_layout: tl.constexpr = kv_atom_layout.tile_to_shape((BLOCK_N, HEAD_DIM))
    k_smem = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), 1, layout=kv_smem_layout)
    v_smem = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_v), 1, layout=kv_smem_layout)
    qdo_atom_layout: tl.constexpr = tlx.nv_mma_shared_layout_encoding(
        (BLOCK_M, HEAD_DIM // NUM_MMA_GROUPS_BWD),
        [1, 0],
        tlx.dtype_of(desc_q),
        [1, 1],
        [1, 1],
        [1, 0],
        False,
        True,
    )
    qdo_smem_layout: tl.constexpr = qdo_atom_layout.tile_to_shape((BLOCK_M, HEAD_DIM))
    q_smem = tlx.local_alloc((BLOCK_M, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q, layout=qdo_smem_layout)
    do_smem = tlx.local_alloc((BLOCK_M, HEAD_DIM), tlx.dtype_of(desc_do), NUM_BUFFERS_Q, layout=qdo_smem_layout)
    dq_store_smem = tlx.local_alloc(
        (BLOCK_M, HEAD_DIM),
        tlx.dtype_of(desc_dq),
        NUM_BUFFERS_DQ_STORE * NUM_MMA_GROUPS_BWD,
    )
    score_atom_layout: tl.constexpr = tlx.nv_mma_shared_layout_encoding(
        (CID_BLOCK_N, BLOCK_M),
        [1, 0],
        tlx.dtype_of(desc_q),
        [1, 1],
        [1, 1],
        [1, 0],
        False,
        True,
    )
    score_smem_layout: tl.constexpr = score_atom_layout.tile_to_shape((BLOCK_N, BLOCK_M))
    score_smem_full = tlx.local_alloc(
        (BLOCK_N, BLOCK_M),
        tlx.dtype_of(desc_q),
        1,
        layout=score_smem_layout,
    )

    kv_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=NUM_MMA_GROUPS_BWD)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=NUM_MMA_GROUPS_BWD)

    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))
    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    DO_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_do))

    with tlx.async_tasks():
        with tlx.async_task("default"):
            k_load_view = tlx.local_reinterpret(
                k_smem[0], tlx.dtype_of(desc_k), [BLOCK_N, HEAD_DIM], layout=kv_atom_layout, pin=False
            )
            v_load_view = tlx.local_reinterpret(
                v_smem[0], tlx.dtype_of(desc_v), [BLOCK_N, HEAD_DIM], layout=kv_atom_layout, pin=False
            )
            tlx.barrier_expect_bytes(
                kv_full[0],
                BLOCK_N * HEAD_DIM * (K_BYTES_PER_ELEM + V_BYTES_PER_ELEM),
            )
            tlx.async_descriptor_load(desc_k, k_load_view, [global_start_n, 0], kv_full[0])
            tlx.async_descriptor_load(desc_v, v_load_view, [global_start_n, 0], kv_full[0])

            for blk in range(num_steps):
                q_buf = blk % NUM_BUFFERS_Q
                q_phase = (blk // NUM_BUFFERS_Q) & 1
                start_m = blk * BLOCK_M
                global_start_m = row_base + start_m
                q_load_view = tlx.local_reinterpret(
                    q_smem[q_buf],
                    tlx.dtype_of(desc_q),
                    [BLOCK_M, HEAD_DIM],
                    layout=qdo_atom_layout,
                    pin=False,
                )
                do_load_view = tlx.local_reinterpret(
                    do_smem[q_buf],
                    tlx.dtype_of(desc_do),
                    [BLOCK_M, HEAD_DIM],
                    layout=qdo_atom_layout,
                    pin=False,
                )
                tlx.barrier_wait(q_empties[q_buf], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_buf], BLOCK_M * HEAD_DIM * Q_BYTES_PER_ELEM)
                tlx.async_descriptor_load(desc_q, q_load_view, [global_start_m, 0], q_fulls[q_buf])

                tlx.barrier_wait(do_empties[q_buf], q_phase ^ 1)
                tlx.barrier_expect_bytes(do_fulls[q_buf], BLOCK_M * HEAD_DIM * DO_BYTES_PER_ELEM)
                tlx.async_descriptor_load(desc_do, do_load_view, [global_start_m, 0], do_fulls[q_buf])

        with tlx.async_task(
            num_warps=NUM_MMA_WARPS_BWD // NUM_MMA_GROUPS_BWD,
            registers=BWD_REGISTERS,
            replicate=NUM_MMA_GROUPS_BWD,
        ):
            cid: tl.constexpr = tlx.async_task_replica_id()
            cid_start_n: tl.constexpr = cid * CID_BLOCK_N
            k_view = tlx.local_reinterpret(
                k_smem[0], tlx.dtype_of(desc_k), [BLOCK_N, HEAD_DIM], layout=kv_atom_layout, pin=False
            )
            v_view = tlx.local_reinterpret(
                v_smem[0], tlx.dtype_of(desc_v), [BLOCK_N, HEAD_DIM], layout=kv_atom_layout, pin=False
            )
            score_view = tlx.local_reinterpret(
                score_smem_full[0],
                tlx.dtype_of(desc_q),
                [BLOCK_N, BLOCK_M],
                layout=score_atom_layout,
                pin=False,
            )
            score_view_t = tlx.local_trans(score_view)
            k_slice = tlx.local_slice(k_view, [cid_start_n, 0], [CID_BLOCK_N, HEAD_DIM])
            v_slice = tlx.local_slice(v_view, [cid_start_n, 0], [CID_BLOCK_N, HEAD_DIM])
            score_smem = tlx.local_slice(score_view, [cid_start_n, 0], [CID_BLOCK_N, BLOCK_M])
            score_smem_t = tlx.local_slice(score_view_t, [0, cid_start_n], [BLOCK_M, CID_BLOCK_N])
            tlx.barrier_wait(kv_full[0], 0)
            dk = tl.zeros([CID_BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dv = tl.zeros([CID_BLOCK_N, HEAD_DIM], dtype=tl.float32)
            for blk in range(num_steps):
                q_buf = blk % NUM_BUFFERS_Q
                q_phase = (blk // NUM_BUFFERS_Q) & 1
                start_m = blk * BLOCK_M
                offs_m = start_m + tl.arange(0, BLOCK_M)

                tlx.barrier_wait(q_fulls[q_buf], q_phase)
                q = tlx.local_reinterpret(
                    q_smem[q_buf],
                    tlx.dtype_of(desc_q),
                    [BLOCK_M, HEAD_DIM],
                    layout=qdo_atom_layout,
                    pin=False,
                )
                qT = tlx.local_trans(q)

                qkT = tlx.async_dot(k_slice, qT)
                m = tl.load(M + offs_m)
                Di = tl.load(D + offs_m)
                tlx.barrier_wait(do_fulls[q_buf], q_phase)
                do = tlx.local_reinterpret(
                    do_smem[q_buf],
                    tlx.dtype_of(desc_do),
                    [BLOCK_M, HEAD_DIM],
                    layout=qdo_atom_layout,
                    pin=False,
                )
                doT = tlx.local_trans(do)
                qkT = tlx.async_dot_wait(0, qkT)
                qkT *= sm_scale * RCP_LN2
                pT = tl.math.exp2(qkT - m[None, :])
                dpT = tlx.async_dot(v_slice, doT)
                dv = tlx.async_dot(pT.to(tlx.dtype_of(desc_q)), do, dv)
                dpT = tlx.async_dot_wait(1, dpT).to(tl.float32)
                dsT = pT * (dpT - Di[None, :])
                dk = tlx.async_dot(dsT.to(tlx.dtype_of(desc_q)), q, dk)
                dv = tlx.async_dot_wait(1, dv)
                tlx.barrier_arrive(do_empties[q_buf], 1)
                tlx.local_store(score_smem, dsT.to(tlx.dtype_of(desc_q)))
                tlx.fence_async_shared()

                dq = tlx.async_dot(score_smem_t, k_slice)
                dk = tlx.async_dot_wait(1, dk)
                tlx.barrier_arrive(q_empties[q_buf], 1)
                dq = tlx.async_dot_wait(0, dq)
                dq *= sm_scale
                dq_store_buf = cid * NUM_BUFFERS_DQ_STORE + blk % NUM_BUFFERS_DQ_STORE
                tlx.async_descriptor_store_wait(NUM_BUFFERS_DQ_STORE - 1)
                tlx.local_store(dq_store_smem[dq_store_buf], dq.to(tlx.dtype_of(desc_dq)))
                tlx.fence_async_shared()
                tlx.async_descriptor_store(
                    desc_dq,
                    dq_store_smem[dq_store_buf],
                    [row_base + start_m, 0],
                    store_reduce="add",
                )

            dk *= sm_scale
            tlx.async_descriptor_store_wait(0)
            dkv_store_buf: tl.constexpr = cid * NUM_BUFFERS_DQ_STORE
            tlx.local_store(dq_store_smem[dkv_store_buf], dk.to(tlx.dtype_of(desc_dk)))
            tlx.fence_async_shared()
            tlx.async_descriptor_store(
                desc_dk,
                dq_store_smem[dkv_store_buf],
                [global_start_n + cid_start_n, 0],
            )
            tlx.async_descriptor_store_wait(0)
            tlx.local_store(dq_store_smem[dkv_store_buf], dv.to(tlx.dtype_of(desc_dv)))
            tlx.fence_async_shared()
            tlx.async_descriptor_store(
                desc_dv,
                dq_store_smem[dkv_store_buf],
                [global_start_n + cid_start_n, 0],
            )
            tlx.async_descriptor_store_wait(0)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        if q.dtype == torch.float8_e5m2:
            desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block)
        else:
            desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        _attn_fwd_ws_pipelined_pingpong[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        BLOCK_N = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o,
            do,
            delta,
            dq,
            N_CTX,
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,
        )
        grid = (N_CTX // BLOCK_N, 1, BATCH * N_HEAD)
        y_dim = BATCH * N_HEAD * N_CTX
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_v = TensorDescriptor(v, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_do = TensorDescriptor(do, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_dq = TensorDescriptor(dq, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_dk = TensorDescriptor(dk, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_dv = TensorDescriptor(dv, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        _attn_bwd_tlx[grid](
            desc_q,
            desc_k,
            desc_v,
            ctx.sm_scale,
            desc_do,
            desc_dq,
            desc_dk,
            desc_dv,
            M,
            delta,
            N_HEAD,
            N_CTX,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        return dq, dk, dv, None


def attention(q, k, v, sm_scale, config=None):
    if config is None:
        return _attention.apply(q, k, v, sm_scale)

    # Non-autotuned path with explicit config
    HEAD_DIM_K = q.shape[-1]
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    dummy_block = [1, 1]
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    if q.dtype == torch.float8_e5m2:
        desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block)
    else:
        desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)

    # Apply pre_hook to set block shapes
    nargs = {
        **config, "HEAD_DIM": HEAD_DIM_K, "desc_q": desc_q, "desc_k": desc_k, "desc_v": desc_v, "desc_o": desc_o,
        "FP8_OUTPUT": q.dtype == torch.float8_e5m2
    }
    _host_descriptor_pre_hook(nargs)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    grid = (triton.cdiv(q.shape[2], config["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
    _attn_fwd_ws_pipelined_pingpong.fn[grid](
        sm_scale,
        M,
        q.shape[0],
        q.shape[1],
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        FP8_OUTPUT=q.dtype == torch.float8_e5m2,
        **config,
    )
    return o
