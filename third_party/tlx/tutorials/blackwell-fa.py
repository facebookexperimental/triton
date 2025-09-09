import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    HEAD_DIM = nargs["HEAD_DIM"]
    BLOCK_M_SPLIT = BLOCK_M // 2
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'NUM_STAGES': 2, 'NUM_MI_BUFFER': 3}, num_stages=0, num_warps=4,
                      pre_hook=_host_descriptor_pre_hook),
    ]


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'Z', 'H', 'N_CTX', 'HEAD_DIM'],
)
@triton.jit
def tlx_attention_fwd(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
    NUM_MI_BUFFER: tl.constexpr,
):
    # dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    # initialize offsets
    #offs_n = tl.arange(0, BLOCK_N)

    # allocate buffers for q0, q1
    buffers_q0 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float16, 1)
    buffers_q1 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float16, 1)
    barrier_q0 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_q1 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    # allocate NUM_STAGES buffers for k, v
    buffer_k = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.float16, NUM_STAGES)  # k
    buffer_v = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.float16, NUM_STAGES)  # v
    # for each iteration, where we pass from softmax to correction
    buffer_m_i0 = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, NUM_MI_BUFFER)  # share m_i and m_ij slice 0
    buffer_m_i1 = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, NUM_MI_BUFFER)  # share m_i and m_ij slice 1
    # buffer_m_i, buffer_l_i, slice 0 and slice 1
    # for the final output, to pass from softmax to epilogue
    buffer_m_i0_final = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, 1)
    buffer_l_i0_final = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, 1)
    buffer_m_i1_final = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, 1)
    buffer_l_i1_final = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, 1)

    # allocate tmem
    result = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)
    result_2 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)
    result_3 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)
    result_4 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)

    # allocate barriers for channels
    consumer_release_k = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    consumer_k = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    consumer_release_v = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    consumer_v = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    producer_commit_qk0 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    producer_qk0 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    # used in correction for acc0
    producer_commit_acc0_correction = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    producer_commit_acc0 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    producer_p0 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    producer_commit_p0 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    producer_commit_qk1 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    producer_qk1 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    # used in correction for acc1
    producer_commit_acc1_correction = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    producer_commit_acc1 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    producer_p1 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    producer_commit_p1 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    # barrier for m_i, m_ij
    # m_i and m_ij share the same buffer
    consumer_m_i0 = tlx.alloc_barriers(num_barriers=NUM_MI_BUFFER, arrive_count=1)
    consumer_release_m_i0 = tlx.alloc_barriers(num_barriers=NUM_MI_BUFFER, arrive_count=1)
    consumer_m_i1 = tlx.alloc_barriers(num_barriers=NUM_MI_BUFFER, arrive_count=1)
    consumer_release_m_i1 = tlx.alloc_barriers(num_barriers=NUM_MI_BUFFER, arrive_count=1)

    # causal = False
    with tlx.async_tasks():
        with tlx.async_task("default"):  # correction
            accum_cnt = 0
            accum_cnt_m_i = 0
            lo, hi = 0, N_CTX
            for start_n in tl.range(lo, hi, BLOCK_N):
                # data slice 0
                val_121, val_120 = _get_bufidx_phase(accum_cnt_m_i, NUM_MI_BUFFER)
                # wait for m_i is done
                tlx.barrier_wait(consumer_m_i0[val_121], val_120)
                m_i = tlx.local_load(buffer_m_i0[val_121])  #, tlx.storage_kind.smem)  # m_i
                tlx.barrier_arrive(consumer_release_m_i0[val_121], 1)

                val_130, val_129 = _get_bufidx_phase(accum_cnt_m_i + 1, NUM_MI_BUFFER)
                # wait for m_ij is done
                tlx.barrier_wait(consumer_m_i0[val_130], val_129)
                m_ij = tlx.local_load(buffer_m_i0[val_130])  # m_ij
                tlx.barrier_arrive(consumer_release_m_i0[val_130], 1)

                alpha = tl.math.exp2(m_i - m_ij)
                buf, phase = _get_bufidx_phase(accum_cnt, 1)
                tlx.barrier_wait(producer_commit_acc0[0], phase)  # acc0
                # subtiling to reduce register pressure when hDim is 128
                view_8 = tlx.local_view(result_2, 0)
                view_9 = tlx.subslice(view_8, 0, HEAD_DIM // 2)  # N = 0
                view_10 = tlx.subslice(view_8, HEAD_DIM // 2, HEAD_DIM // 2)  # N = 64
                result_7 = tlx.local_load(view_9)  #, tlx.storage_kind.tmem)
                val_141 = result_7 * alpha[:, None]
                tlx.local_store(view_9, val_141)  #, tlx.storage_kind.tmem)
                result_8 = tlx.local_load(view_10)  #, tlx.storage_kind.tmem)
                val_142 = result_8 * alpha[:, None]
                tlx.local_store(view_10, val_142)  #, tlx.storage_kind.tmem)
                tlx.barrier_arrive(producer_commit_acc0_correction[0], 1)  # acc0

                # data slice 1
                tlx.barrier_wait(consumer_m_i1[val_121], val_120)
                m_i1 = tlx.local_load(buffer_m_i1[val_121])  #, tlx.storage_kind.smem)  # m_i
                tlx.barrier_arrive(consumer_release_m_i1[val_121], 1)

                tlx.barrier_wait(consumer_m_i1[val_130], val_129)
                m_ij1 = tlx.local_load(buffer_m_i1[val_130])  # m_ij
                tlx.barrier_arrive(consumer_release_m_i1[val_130], 1)

                alpha1 = tl.math.exp2(m_i1 - m_ij1)
                tlx.barrier_wait(producer_commit_acc1[0], phase)  # acc1
                # subtiling to reduce register pressure when hDim is 128
                view_19 = tlx.local_view(result, 0)
                view_20 = tlx.subslice(view_19, 0, HEAD_DIM // 2)  # N = 0
                view_21 = tlx.subslice(view_19, 64, HEAD_DIM // 2)  # N = 64
                result_9 = tlx.local_load(view_20)  #, tlx.storage_kind.tmem)
                val_157 = result_9 * alpha1[:, None]
                tlx.local_store(view_20, val_157)  #, tlx.storage_kind.tmem)
                result_10 = tlx.local_load(view_21)  #, tlx.storage_kind.tmem)
                val_158 = result_10 * alpha1[:, None]
                tlx.local_store(view_21, val_158)  #, tlx.storage_kind.tmem)
                tlx.barrier_arrive(producer_commit_acc1_correction[0], 1)  # acc1

                # update loop variables
                accum_cnt = accum_cnt + 1
                accum_cnt_m_i = accum_cnt_m_i + 2

        with tlx.async_task(num_warps=1):  # gemm
            lo, hi = 0, N_CTX
            # dot0_slice0
            view_3 = tlx.local_view(buffer_k, 0)  # k
            view_4 = tlx.local_trans(view_3)
            # wait for q0 and k, barrier will be consumer_k, consumer_k
            tlx.barrier_wait(consumer_k[0], 0)
            tlx.barrier_wait(producer_qk0[0], 0)
            # dot0_slice1: q0 . k, producer_commit_qk0
            view_7 = tlx.local_view(producer_commit_qk0, 0)
            tlx.async_dot(buffers_q0[0], view_4, result_3[0], mBarriers=[view_7])
            tlx.barrier_wait(producer_qk1[0], 0)  # has predicate?
            view_9 = tlx.local_view(result_4, 0)
            # producer_commit_qk1
            tlx.async_dot(buffers_q1[0], view_4, result_4[0], mBarriers=[consumer_release_k[0], producer_commit_qk1[0]])

            accum_cnt = 0
            for start_n in tl.range(lo, hi, BLOCK_N):
                val_121 = hi - 128
                # dot1_slice0_iter_i: p0 . v
                buf_v, phase_v = _get_bufidx_phase(accum_cnt, 2)
                view_13 = tlx.local_view(buffer_v, buf_v)
                tlx.barrier_wait(consumer_v[buf_v], phase_v)
                buf, phase = _get_bufidx_phase(accum_cnt, 1)
                tlx.barrier_wait(producer_commit_acc0_correction[0], phase ^ 1)
                tlx.barrier_wait(producer_commit_p0, phase)
                view_15 = tlx.local_reinterpret(result_3[0], tl.float16)
                # releave v, consumer_release for p0 via producer_p0, producer_commit_acc0
                tlx.async_dot(view_15, view_13, result_2[0], mBarriers=[producer_commit_acc0[0], producer_p0[0]])
                # dot0_slice0_iter_i+1: q0 . k
                val_131, val_132 = _get_bufidx_phase(accum_cnt + 1, 2)
                view_20 = tlx.local_view(buffer_k, val_131)
                view_21 = tlx.local_trans(view_20)
                tlx.barrier_wait(consumer_k[val_131], val_132)  #, pred=start_n < val_121)
                tlx.barrier_wait(producer_qk0[0], phase ^ 1)  #, pred=start_n < val_121)
                # commit qk0
                tlx.async_dot(buffers_q0[0], view_21, result_3[0], mBarriers=[view_7])  #pred=start_n < val_121
                view_22 = tlx.local_reinterpret(view_9, tl.float16)
                tlx.barrier_wait(producer_commit_acc1_correction[0], phase ^ 1)
                tlx.barrier_wait(producer_commit_p1, phase)
                # dot1_slice1_iter_i: p1 . v
                # release v, commit result
                tlx.async_dot(view_22, view_13, result[0],
                              mBarriers=[consumer_release_v[buf_v], producer_commit_acc1[0], producer_p1[0]])
                tlx.barrier_wait(producer_qk1[0], phase ^ 1)  #, pred=start_n < val_121)
                # dot0_slice1_iter_i+1: q1 . k
                # release k, commit qk1
                tlx.async_dot(buffers_q1[0], view_21, view_9,
                              mBarriers=[consumer_release_k[val_131], producer_commit_qk1[0]])  #pred=start_n < val_121

                accum_cnt = accum_cnt + 1

        with tlx.async_task(num_warps=1):  # load
            lo, hi = 0, N_CTX
            # producer_acquire for q0: consumer_release
            buffer_q0 = tlx.local_view(buffers_q0, 0)
            buffer_q1 = tlx.local_view(buffers_q1, 0)
            tlx.barrier_expect_bytes(barrier_q0[0], 32768)
            tlx.async_descriptor_load(desc_q, buffer_q0, [qo_offset_y, 0], barrier_q0[0])
            tlx.barrier_expect_bytes(barrier_q1[0], 32768)
            tlx.async_descriptor_load(desc_q, buffer_q1, [qo_offset_y + BLOCK_M // 2, 0], barrier_q1[0])

            arg70 = offset_y
            accum_cnt = 0
            for start_n in tl.range(lo, hi, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                # val_126 = hi - 256
                # load v_iter_i
                bufIdx, phase = _get_bufidx_phase(accum_cnt, 2)
                tlx.barrier_wait(consumer_release_v[bufIdx], phase ^ 1)  # producer acquire
                view_8 = tlx.local_view(consumer_v, bufIdx)
                tlx.barrier_expect_bytes(view_8, 32768)
                view_9 = tlx.local_view(buffer_v, bufIdx)
                tlx.async_descriptor_load(desc_v, view_9, [arg70, 0], view_8)

                # load k_iter_i with predicate
                tlx.barrier_wait(consumer_release_k[bufIdx], phase ^ 1)  # acquire
                view_11 = tlx.local_view(consumer_k, bufIdx)
                tlx.barrier_expect_bytes(view_11, 32768)
                view_12 = tlx.local_view(buffer_k, bufIdx)
                tlx.async_descriptor_load(desc_k, view_12, [arg70, 0], view_11)  #, pred=start_n < val_126)

                # update args
                arg70 = arg70 + 128
                accum_cnt = accum_cnt + 1

        with tlx.async_task(num_warps=4):  # softmax0
            lo, hi = 0, N_CTX
            # inputs; m_i, qk
            cst_9 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
            cst_10 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
            l_i = cst_10
            m_i = cst_9
            qk_scale = sm_scale
            qk_scale *= 1.44269504  # 1/log(2)

            accum_cnt_m_i = 0
            accum_cnt = 0
            for start_n in tl.range(lo, hi, BLOCK_N):
                bufIdx, phase = _get_bufidx_phase(accum_cnt, 1)
                view_1 = tlx.local_view(producer_commit_qk0, bufIdx)
                tlx.barrier_wait(view_1, phase)
                view_2 = tlx.local_view(result_3, bufIdx)
                result_14 = tlx.local_load(view_2)  #, tlx.storage_kind.tmem)
                view_3 = tlx.local_view(producer_qk0, bufIdx)
                tlx.barrier_arrive(view_3, 1)

                m_ij = tl.maximum(m_i, tl.max(result_14, 1) * qk_scale)

                # need to store m_i for correction to use
                bufIdx_m_i, phase_m_i = _get_bufidx_phase(accum_cnt_m_i, NUM_MI_BUFFER)
                tlx.barrier_wait(consumer_release_m_i0[bufIdx_m_i], phase_m_i)
                tlx.local_store(buffer_m_i0[bufIdx_m_i], m_i)
                tlx.barrier_arrive(consumer_m_i0[bufIdx_m_i], 1)
                # store m_ij
                bufIdx_m_ij, phase_m_ij = _get_bufidx_phase(accum_cnt_m_i + 1, NUM_MI_BUFFER)
                view_6 = tlx.local_view(consumer_release_m_i0, bufIdx_m_ij)
                tlx.barrier_wait(view_6, phase_m_ij)
                tlx.local_store(buffer_m_i0[bufIdx_m_ij], m_ij)
                tlx.barrier_arrive(consumer_m_i0[bufIdx_m_ij], 1)

                val_133 = result_14 * qk_scale - m_ij[:, None]
                val_134 = tl.math.exp2(val_133)  # p
                val_136 = tl.math.exp2(m_i - m_ij)  # alpha
                val_137 = tl.sum(val_134, 1)  # l_ij
                p = val_134.to(tl.float16)
                view_8 = tlx.local_reinterpret(result_3[0], tl.float16)
                # producer acquire for p0
                tlx.barrier_wait(producer_p0, phase)
                tlx.local_store(view_8, p)  #, tlx.storage_kind.tmem)
                # producer commit for p0
                tlx.barrier_arrive(producer_commit_p0[0], 1)
                l_i = l_i * val_136 + val_137  # new value for l_i

                m_i = m_ij
                accum_cnt_m_i = accum_cnt_m_i + 2
                accum_cnt = accum_cnt + 1
            tlx.local_store(buffer_l_i0_final, l_i)
            tlx.local_store(buffer_m_i0_final, m_i)

        with tlx.async_task(num_warps=4):  # softmax1
            lo, hi = 0, N_CTX
            cst_9 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
            cst_10 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
            l_i = cst_10
            m_i = cst_9
            qk_scale = sm_scale
            qk_scale *= 1.44269504  # 1/log(2)

            accum_cnt_m_i = 0
            accum_cnt = 0
            for start_n in tl.range(lo, hi, BLOCK_N):
                bufIdx, phase = _get_bufidx_phase(accum_cnt, 1)
                view_1 = tlx.local_view(producer_commit_qk1, bufIdx)
                tlx.barrier_wait(view_1, phase)
                # view_2 = tlx.local_view(result_4, bufIdx)
                result_14 = tlx.local_load(result_4[bufIdx])  #, tlx.storage_kind.tmem)
                view_3 = tlx.local_view(producer_qk1, bufIdx)
                tlx.barrier_arrive(view_3, 1)

                m_ij = tl.maximum(m_i, tl.max(result_14, 1) * qk_scale)

                # need to store m_i for correction to use
                bufIdx_m_i, phase_m_i = _get_bufidx_phase(accum_cnt_m_i, NUM_MI_BUFFER)
                view_15 = tlx.local_view(consumer_m_i1, bufIdx_m_i)
                tlx.barrier_wait(consumer_release_m_i1[bufIdx_m_i], phase_m_i)
                tlx.local_store(buffer_m_i1[bufIdx_m_i], m_i)
                tlx.barrier_arrive(view_15, 1)
                # store m_ij
                bufIdx_m_ij, phase_m_ij = _get_bufidx_phase(accum_cnt_m_i + 1, NUM_MI_BUFFER)
                # view_6 = tlx.local_view(consumer_release_m_i1, bufIdx_m_ij)
                tlx.barrier_wait(consumer_release_m_i1[bufIdx_m_ij], phase_m_ij)
                tlx.local_store(buffer_m_i1[bufIdx_m_ij], m_ij)
                tlx.barrier_arrive(consumer_m_i1[bufIdx_m_ij], 1)

                val_133 = result_14 * qk_scale - m_ij[:, None]
                val_134 = tl.math.exp2(val_133)  # p
                val_136 = tl.math.exp2(m_i - m_ij)  # alpha
                val_137 = tl.sum(val_134, 1)  # l_ij
                p = val_134.to(tl.float16)
                view_8 = tlx.local_reinterpret(result_4[0], tl.float16)
                # producer acquire for p1
                tlx.barrier_wait(producer_p1, phase)
                tlx.local_store(view_8, p)  #, tlx.storage_kind.tmem)
                # commit for p1
                tlx.barrier_arrive(producer_commit_p1, 1)
                l_i = l_i * val_136 + val_137  # new value for l_i

                m_i = m_ij
                accum_cnt_m_i = accum_cnt_m_i + 2
                accum_cnt = accum_cnt + 1
            tlx.local_store(buffer_l_i1_final, l_i)
            tlx.local_store(buffer_m_i1_final, m_i)

        # hitting assert _is_async_task(self, stmt) if not inside async_tasks
        # epilogue
        with tlx.async_task(num_warps=4):  # epilogue
            val_88 = tlx.local_load(buffer_l_i1_final[0])
            val_90 = tlx.local_load(buffer_m_i1_final[0])
            val_91 = tlx.local_load(buffer_l_i0_final[0])
            val_93 = tlx.local_load(buffer_m_i0_final[0])

            val_95 = val_93 + tl.math.log2(val_91)
            offs_m0 = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
            m_ptrs = M + off_hz * N_CTX + offs_m0
            tl.store(m_ptrs, val_95)
            result_5 = tlx.local_load(result_2[0])  #, tlx.storage_kind.tmem)
            val_102 = result_5 / val_91[:, None]  # check layout
            acc = val_102.to(tl.float16)
            #buffer_104 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float16, 1)
            # fence_async_shared
            c_buffers = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float16, tl.constexpr(1))
            tlx.local_store(c_buffers[0], acc)
            tlx.async_descriptor_store(desc_o, c_buffers[0], [qo_offset_y, 0])
            # tma_store_wait

            val_107 = val_90 + tl.math.log2(val_88)
            offs_m1 = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
            m_ptrs = M + off_hz * N_CTX + offs_m1
            tl.store(m_ptrs, val_107)
            result_6 = tlx.local_load(result[0])  #, tlx.storage_kind.tmem)
            val_111 = result_6 / val_88[:, None]  # check layout
            acc = val_111.to(tl.float16)
            #buffer_113 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float16, 1)
            # fence_async_shared
            c_buffers1 = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float16, tl.constexpr(1))
            tlx.local_store(c_buffers1[0], acc)
            tlx.async_descriptor_store(desc_o, c_buffers1[0], [qo_offset_y + BLOCK_M_SPLIT, 0])
            # tma_store_wait


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
        tlx_attention_fwd[grid](
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


attention = _attention.apply


@pytest.mark.parametrize("Z", [8])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("N_CTX", [1024])
@pytest.mark.parametrize("HEAD_DIM", [128])
@pytest.mark.parametrize("mode", ["fwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"])
def test_op(Z, H, N_CTX, HEAD_DIM, mode, provider, dtype=torch.float16):
    if mode == "bwd":
        pytest.skip("Backward pass not supported.")
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
    # M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, sm_scale).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = False
BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
# vary seq length for fixed head and batch=4
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="provider",
        line_vals=["triton-fp16"] + (["flash"] if HAS_FLASH else []),
        line_names=["Triton [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name=f"fused-attention-ws-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "HEAD_DIM": HEAD_DIM,
            "mode": "fwd",
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
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
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
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
