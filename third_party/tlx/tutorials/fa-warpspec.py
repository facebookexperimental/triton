
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
           # set up arguments
           arg26 = arg27 = arg28 = arg30 = 0
           arg29 = -1
           for start_n in tl.range(lo, hi, BLOCK_N):
               # data slice 0
               # convert from ttng.wait_barrier to tlx.barrier_wait, trace the barrier and the phase
               val_116 = arg26 ^ 1
               val_117 = arg27 + 1
               val_121 = 1 if val_117 == 3 else val_117
               val_120 = arg28 ^ 1 if val_117 == 3 else arg28
               view_1 = tlx.local_view(buffer_63, val_121)
               view_2 = tlx.local_view(barrier_64, val_121)
               view_3 = tlx.local_view(barrier_65, val_121)
               tlx.barrier_wait(view_2, val_120)
               m_i = tlx.local_load(view_1, tlx.storage_kind.smem) # m_i
               tlx.barrier_arrive(view_3, 1)

               val_126 = arg29 + 1
               val_130 = 1 if val_126 == 3 else val_126
               val_129 = arg30 ^ 1 if val_126 == 3 else arg30
               view_4 = tlx.local_view(buffer_63, val_130)
               view_5 = tlx.local_view(barrier_64, val_130)
               view_6 = tlx.local_view(barrier_65, val_130)
               tlx.barrier_wait(view_5, val_129)
               m_ij = tlx.local_load(view_4) # m_ij
               tlx.barrier_arrive(view_6, 1)

               alpha = tl.math.exp2(m_i - m_ij)
               view_7 = tlx.local_view(barrier_45, 0)
               barrier_wait(view_7, arg26) # acc0
               # subtiling to reduce register pressure when hDim is 128
               view_8 = tlx.local_view(result_2, 0)
               view_9 = tlx.subslice(view_8, 0) # N = 0
               view_10 = tlx.subslice(view_8, 64) # N = 64
               result_7 = tlx.local_load(view_9, tlx.storage_kind.tmem)
               val_141 = result_7 * alpha[:, None]
               tlx.local_store(view_9, val_141, tlx.storage_kind.tmem)
               result_8 = tlx.local_load(view_10, tlx.storage_kind.tmem)
               val_142 = result_8 * alpha[:, None]
               tlx.local_store(view_10, val_142, tlx.storage_kind.tmem)
               view_11 = tlx.local_view(barrier_43, 0)
               tlx.barrier_arrive(view_11, 1) # acc0

               # data slice 1
               view_12 = tlx.local_view(buffer_73, val_121)
               view_13 = tlx.local_view(barrier_74, val_121)
               view_14 = tlx.local_view(barrier_75, val_121)
               tlx.barrier_wait(view_13, val_120)
               m_i1 = tlx.local_load(view_12, tlx.storage_kind.smem) # m_i
               tlx.barrier_arrive(view_14, 1)

               view_15 = tlx.local_view(buffer_73, val_130)
               view_16 = tlx.local_view(barrier_74, val_130)
               view_17 = tlx.local_view(barrier_75, val_130)
               tlx.barrier_wait(view_16, val_129)
               m_ij1 = tlx.local_load(view_15) # m_ij
               tlx.barrier_arrive(view_17, 1)

               alpha1 = tl.math.exp2(m_i1 - m_ij1)
               view_18 = tlx.local_view(barrier_57, 0)
               barrier_wait(view_18, arg26) # acc1
               # subtiling to reduce register pressure when hDim is 128
               view_19 = tlx.local_view(result, 0)
               view_20 = tlx.subslice(view_19, 0) # N = 0
               view_21 = tlx.subslice(view_19, 64) # N = 64
               result_9 = tlx.local_load(view_20, tlx.storage_kind.tmem)
               val_157 = result_9 * alpha1[:, None]
               tlx.local_store(view_20, val_157, tlx.storage_kind.tmem)
               result_10 = tlx.local_load(view_21, tlx.storage_kind.tmem)
               val_158 = result_10 * alpha1[:, None]
               tlx.local_store(view_21, val_158, tlx.storage_kind.tmem)
               view_22 = tlx.local_view(barrier_55, 0)
               tlx.barrier_arrive(view_22, 1) # acc1

               # update loop variables
               arg26 = val_116
               arg27 = val_121
               arg28 = val_120
               arg29 = val_130
               arg30 = val_129

        with tlx.async_task(num_warps=1): # gemm
           # dot0_slice0
           view_1 = tlx.local_view(barrier_24, 0)
           view_2 = tlx.local_view(barrier_27, 0)
           view_3 = tlx.local_view(buffer_23, 0)
           view_4 = tlx.trans(view_3)
           tlx.barrier_wait(view_2, 0)
           view_5 = tlx.local_view(barrier_41, 0)
           tlx.barrier_wait(view_5, 0)
           view_6 = tlx.local_view(result_3, 0)
           # TODO: support pred and multiple barriers
           view_7 = tlx.local_view(barrier_39, 0)
           tlx.async_dot(buffer_14, view_4, view_6, mmav5=True, useD=False, pred=True, view_1, view_7)
           view_8 = tlx.local_view(barrier_53, 0)
           tlx.barrier_wait(view_8, 0) # has predicate?
           # dot0_slice1
           view_9 = tlx.local_view(result_4, 0)
           view_10 = tlx.local_view(barrier_51, 0)
           tlx.async_dot(buffer_18, view_4, view_9, mmav5=True, useD=False, pred=True, view_1, view_10)

           arg65 = arg66 = arg67 = 0
           for start_n in tl.range(lo, hi, BLOCK_N):
               val_121 = hi - 128
               # dot1_slice0_iter_i
               view_11 = tlx.local_view(barrier_33, arg65)
               view_12 = tlx.local_view(barrier_36, arg65)
               view_13 = tlx.local_view(buffer_32, arg65)
               tlx.barrier_wait(view_12, arg66)
               view_14 = tlx.local_view(barrier_43, 0)
               val_126 = arg67 ^ 1
               tlx.barrier_wait(view_14, val_126)
               tlx.barrier_wait(barrier_49, arg67)
               # TODO: reinterpret
               view_15 = tlx.reinterpret(view_6)
               view_16 = tlx.local_view(result_2, 0)
               view_17 = tlx.local_view(barrier_45, 0)
               tlx.async_dot(view_15, view_13, view_16, mmav5=True, useD=True, pred=True, view_11, view_17, barrier_47)
               # dot0_slice0_iter_i+1
               val_131 = 0 if val_128 == 2 else val_128
               val_132 = arg66 ^ 1 if val_128 == 2 else arg66
               view_18 = tlx.local_view(barrier_24, val_131)
               view_19 = tlx.local_view(barrier_27, val_131)
               view_20 = tlx.local_view(buffer_23, val_131)
               view_21 = tlx.trans(view_20)
               tlx.barrier_wait(view_19, val_132, start_n < val_121)
               tlx.barrier_wait(view_5, arg67 ^ 1, start_n < val_121)
               tlx.async_dot(buffer_14, view_21, view_6, mmav5=True, useD=False, pred=start_n < val_121, view_18, view_7)
               view_22 = tlx.reinterpret(view_9)
               view_23 = tlx.local_view(barrier_55, 0)
               tlx.barrier_wait(view_23, arg67 ^ 1)
               tlx.barrier_wait(barrier_61, arg67)
               # dot1_slice1_iter_i
               view_24 = tlx.local_view(result, 0)
               view_25 = tlx.local_view(barrier_57, 0)
               tlx.async_dot(view_22, view_13, view_24, mmav5=True, useD=True, pred=True, view_11, view_25, barrier_59)
               tlx.barrier_wait(view_8, arg67 ^ 1, start_n < val_121)
               # dot0_slice1_iter_i+1
               tlx.async_dot(barrier_18, view_21, view_9, mmav5=True, useD=False, pred=start_n < val_121, view_18, view_10)

               # update arg65/arg66/arg67
               arg65 = val_131
               arg66 = val_132
               arg67 = val_126

        with tlx.async_task(num_warps=1): # load
           view_1 = tlx.local_view(barrier_24, 0)
           tlx.barrier_wait(view_1, 0)
           view_2 = tlx.local_view(barrier_27, 0)
           tlx.barrier_expect_bytes(view_2, 32768)
           view_3 = tlx.local_view(buffer_23, 0)
           tlx.async_descriptor_load(arg9, view_3, [val_5, 0], view_2)

           view_4 = tlx.local_view(barrier_24, 1)
           tlx.barrier_wait(view_4, 0, pred=hi > 128)
           view_5 = tlx.local_view(barrier_27, 1)
           tlx.barrier_expect_bytes(view_5, 32768)
           view_6 = tlx.local_view(buffer_23, 1)
           tlx.async_descriptor_load(arg9, view_6, [val_5 + 128, 0], view_5, pred=hi > 128)
           # load k_iter_0, k_iter_1
           arg65 = val_5 + 128
           arg66 = 1
           arg67 = arg68 = arg69 = 0
           arg70 = val_5
           for start_n in tl.range(lo, hi, BLOCK_N):
               start_n = tl.multiple_of(start_n, BLOCK_N)
               val_126 = hi - 256
               # load v_iter_i
               view_7 = tlx.local_view(barrier_33, arg68)
               tlx.barrier_wait(view_7, arg69)
               view_8 = tlx.local_view(barrier_36, arg68)
               tlx.barrier_expect_bytes(view_8, 32768)
               view_9 = tlx.local_view(buffer_32, arg68)
               tlx.async_descriptor_load(arg14, view_9, [arg70, 0], view_8)

               # load k_iter_i+2 with predicate
               val_132 = arg68 + 1
               val_135 = 0 if val_132 == 2 else val_132
               val_136 = arg69 ^ 1 if val_132 == 2 else arg69
               val_137 = arg65 + 128
               val_138 = arg66 + 1
               val_141 = 0 if val_138 == 2 else val_138
               val_142 = arg67 ^ 1 if val_138 == 2 else arg67
               view_10 = tlx.local_view(barrier_24, val_141)
               tlx.barrier_wait(view_10, val_142, pred=start_n < val_126)
               view_11 = tlx.local_view(barrier_27, val_141)
               tlx.barrier_expect_bytes(view_11, 32768)
               view_12 = tlx.local_view(buffer_23, val_141)
               tlx.async_descriptor_load(arg9, view_12, [val_137, 0], view_11, pred=start_n < val_126)

               # update args
               arg70 = arg65
               arg65 = val_137
               arg66 = val_141
               arg67 = val_142
               arg68 = val_135
               arg69 = val_136

        with tlx.async_task(num_warps=4): # softmax0
           arg65 = cst_10
           arg66 = cst_9
           arg67 = arg68 = arg69 = 0
           qk_scale = val_13
           for start_n in tl.range(lo, hi, BLOCK_N):
               view_1 = tlx.local_view(barrier_39, 0)
               tlx.barrier_wait(view_1, arg67)
               view_2 = tlx.local_view(result_3, 0)
               result_14 = tlx.local_load(view_2, tlx.storage_kind.tmem)
               view_3 = tlx.local_view(barrier_41, 0)
               tlx.barrier_arrive(view_3, 1)

               m_ij = tl.maximum(arg66, tl.max(result_14, 1) * qk_scale)

               val_122 = arg68 + 1
               val_125 = arg69 ^ 1 if val_122 == 3 else arg69
               val_126 = 1 if val_122 == 3 else val_122
               view_4 = tlx.local_view(buffer_63, val_126)
               view_5 = tlx.local_view(barrier_64, val_126)
               view_6 = tlx.local_view(barrier_65, val_126)
               tlx.barrier_wait(view_6, val_125)
               tlx.local_store(view_4, m_ij)
               tlx.barrier_arrive(view_5, 1)

               val_133 = result_14 * qk_scale - m_ij[:, None]
               val_134 = tl.math.exp2(val_133) # p
               val_136 = tl.math.exp2(arg66 - m_ij) # alpha
               val_137 = tl.sum(val_134, 1) # l_ij
               p = val_134.to(tl.float16)
               view_7 = tlx.local_view(result_3, 0)
               view_8 = tlx.reinterpret(view_7)
               tlx.barrier_wait(barrier_47, arg67)
               tlx.local_store(view_8, p, tlx.storage_kind.tmem)
               tlx.barrier_arrive(barrier_49, 1)
               val_141 = arg65 * val_136 + val_137 # new value for l_i

               arg65 = val_141
               arg66 = m_ij
               arg67 = arg67 ^ 1
               arg68 = val_126
               arg69 = val_125
        tlx.local_store(buffer_84, arg65)
        tlx.local_store(buffer_83, arg66)

        with tlx.async_task(num_warps=4): # softmax1
           arg65 = cst_10
           arg66 = cst_9
           arg67 = arg68 = arg69 = 0
           qk_scale = val_13
           for start_n in tl.range(lo, hi, BLOCK_N):
               view_1 = tlx.local_view(barrier_51, 0)
               tlx.barrier_wait(view_1, arg67)
               view_2 = tlx.local_view(result_4, 0)
               result_14 = tlx.local_load(view_2, tlx.storage_kind.tmem)
               view_3 = tlx.local_view(barrier_53, 0)
               tlx.barrier_arrive(view_3, 1)

               m_ij = tl.maximum(arg66, tl.max(result_14, 1) * qk_scale)

               val_122 = arg68 + 1
               val_125 = arg69 ^ 1 if val_122 == 3 else arg69
               val_126 = 1 if val_122 == 3 else val_122
               view_4 = tlx.local_view(buffer_73, val_126)
               view_5 = tlx.local_view(barrier_74, val_126)
               view_6 = tlx.local_view(barrier_75, val_126)
               tlx.barrier_wait(view_6, val_125)
               tlx.local_store(view_4, m_ij)
               tlx.barrier_arrive(view_5, 1)

               val_133 = result_14 * qk_scale - m_ij[:, None]
               val_134 = tl.math.exp2(val_133) # p
               val_136 = tl.math.exp2(arg66 - m_ij) # alpha
               val_137 = tl.sum(val_134, 1) # l_ij
               p = val_134.to(tl.float16)
               view_7 = tlx.local_view(result_4, 0)
               view_8 = tlx.reinterpret(view_7)
               tlx.barrier_wait(barrier_59, arg67)
               tlx.local_store(view_8, p, tlx.storage_kind.tmem)
               tlx.barrier_arrive(barrier_61, 1)
               val_141 = arg65 * val_136 + val_137 # new value for l_i

               arg65 = val_141
               arg66 = m_ij
               arg67 = arg67 ^ 1
               arg68 = val_126
               arg69 = val_125
        tlx.local_store(buffer_86, arg65)
        tlx.local_store(buffer_85, arg66)

    # epilogue
    val_88 = tlx.local_load(buffer_86)
    val_90 = tlx.local_load(buffer_85)
    val_91 = tlx.local_load(buffer_84)
    val_93 = tlx.local_load(buffer_83)

    val_95 = val_93 + tl.math.log2(val_91)
    m_ptrs = arg1 + val_1 * arg24 + val_10
    tl.store(m_ptrs, val_95)
    view_1 = tlx.local_view(result_2, 0)
    result_5 = tlx.local_load(view_1, tlx.storage_kind.tmem)
    val_102 = result_5 / val_91[:, None] # check layout
    acc = val_102.to(tl.float16)
    buffer_104 = tlx.local_alloc((BLOCK_M//2, HEAD_DIM), tl.float16, 1)
    # fence_async_shared
    async_descriptor_store(arg19, buffer_104, [val_7, 0])
    # tma_store_wait

    val_107 = val_90 + tl.math.log2(val_88)
    m_ptrs = arg1 + val_1 * arg24 + val_12
    tl.store(m_ptrs, val_107)
    view_2 = tlx.local_view(result, 0)
    result_6 = tlx.local_load(view_2, tlx.storage_kind.tmem)
    val_111 = result_6 / val_88[:, None] # check layout
    acc = val_111.to(tl.float16)
    buffer_113 = tlx.local_alloc((BLOCK_M//2, HEAD_DIM), tl.float16, 1)
    # fence_async_shared
    async_descriptor_store(arg19, buffer_113, [val_17, 0])
    # tma_store_wait

