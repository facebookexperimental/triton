
import torch

import triton
import triton.language as tl
import triton.tlx.language as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=0,
                      num_warps=8),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def tlx_attention_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets 
    offs_m0 = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)
    offs_m1 = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i0 = tl.zeros([BLOCK_M//2], dtype=tl.float32) - float("inf")
    l_i0 = tl.zeros([BLOCK_M//2], dtype=tl.float32) + 1.0
    acc0 = tl.zeros([BLOCK_M//2, HEAD_DIM], dtype=tl.float32)

    m_i1 = tl.zeros([BLOCK_M//2], dtype=tl.float32) - float("inf")
    l_i1 = tl.zeros([BLOCK_M//2], dtype=tl.float32) + 1.0
    acc1 = tl.zeros([BLOCK_M//2, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # allocate buffers for q0, q1
    buffers_q0 = tlx.local_alloc((BLOCK_M//2, HEAD_DIM), tl.float16, 1)
    buffers_q1 = tlx.local_alloc((BLOCK_M//2, HEAD_DIM), tl.float16, 1)
    q0 = tlx.local_view(buffers_q0, 0)
    q1 = tlx.local_view(buffers_q1, 0)
    tlx.async_descriptor_load(desc_q, [], q0) # tma without mask
    tlx.async_descriptor_load(desc_q, [], q1)

    # load q0, q1 via tlx
    #q0 = desc_q.load([qo_offset_y, 0])
    #q1 = desc_q.load([qo_offset_y + BLOCK_M//2, 0])

    # allocate NUM_STAGES buffers for k, v
    buffers_k = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.float16, NUM_STAGES)
    buffers_v = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.float16, NUM_STAGES)

    # allocate barriers for channels

    # warpspec
    # causal = False
    lo, hi = 0, N_CTX
    with tlx.async_tasks():
        with tlx.async_task("default"): # correction
           for start_n in tl.range(lo, hi, BLOCK_N):
               # data slice 0
               tlx.barrier_wait(barrier[bufIdx], phase)
               tlx.local_load # m_i
               tlx.barrier_arrive(barrier[bufIdx])
               tlx.barrier_wait(
               tlx.local_load # m_ij
               tlx.barrier_arrive(barrier[bufIdx])
               alpha = tl.math.exp2(m_i - m_ij)
               barrier_wait # acc0
               # subtiling to reduce register pressure when hDim is 128
               tmem_load
               acc0 = acc0 * alpha[:, None]
               tmem_store
               tmem_load
               acc0 = acc0 * alpha[:, None]
               tmem_store
               tlx.barrier_arrive( # acc0

               # data slice 1
               tlx.barrier_wait(barrier[bufIdx], phase)
               tlx.local_load # m_i
               tlx.barrier_arrive(barrier[bufIdx])
               tlx.barrier_wait(
               tlx.local_load # m_ij
               tlx.barrier_arrive(barrier[bufIdx])
               alpha = tl.math.exp2(m_i - m_ij)
               barrier_wait
               tmem_load
               acc0 = acc0 * alpha[:, None]
               tmem_store
               tmem_load
               acc0 = acc0 * alpha[:, None]
               tmem_store
               tlx.barrier_arrive(
               
        with tlx.async_task(num_warps=1): # gemm
           # dot0_slice0
           tlx.async_dot(q0[bufIdx], k[bufIdx], acc0, False, True, barrier0, barrier1)
           # dot0_slice1
           for start_n in tl.range(lo, hi, BLOCK_N):
               # dot1_slice0_iter_i
               # dot0_slice0_iter_i+1
               # dot1_slice1_iter_i
               # dot0_slice1_iter_i+1

        with tlx.async_task(num_warps=1): # load
           # load k_iter_0, k_iter_1
           for start_n in tl.range(lo, hi, BLOCK_N):
               start_n = tl.multiple_of(start_n, BLOCK_N)
               # load v_iter_i
               # load k_iter_i+2 with predicate

        with tlx.async_task(num_warps=4): # softmax0
           for start_n in tl.range(lo, hi, BLOCK_N):
        with tlx.async_task(num_warps=4): # softmax1
           for start_n in tl.range(lo, hi, BLOCK_N):

    # epilogue


