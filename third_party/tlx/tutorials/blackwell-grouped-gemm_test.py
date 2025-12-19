"""
Group GEMM
============================
This group gemm kernel launches a fixed number of CTA to compute a group
of gemms. The scheduling is static and we do it on device.
"""

# Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Optional
import torch
import pytest

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_blackwell

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "NUM_SM": 84,
        }),
        triton.Config({
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "NUM_SM": 128,
        }),
        triton.Config({
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "NUM_SM": 84,
        }),
        triton.Config({
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "NUM_SM": 128,
        }),
        triton.Config({
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "NUM_SM": num_sms(),
        }),
        triton.Config({
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "NUM_SM": num_sms(),
        }),
    ],
    key=["group_size"],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def group_gemm_fn(group_A, group_B):
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META["NUM_SM"], )
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
    )

    return group_C


tma_configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK}, num_stages=s, num_warps=w)
    for BM in [128]
    for BN in [128, 256]
    for BK in [64, 128]
    for s in ([3, 4])
    for w in [4, 8]
]


@triton.autotune(
    tma_configs,
    key=["group_a_ptrs", "group_b_ptrs", "gropup_c_ptrs", "group_size"],
)
@triton.jit
def grouped_matmul_tma_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # is the output FP8 or FP16
    FP8: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8 else tl.float16
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )

            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            # iterate through the tiles in the current gemm problem
            while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                k = gk
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # do regular gemm here
                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    accumulator += tl.dot(a, b.T)

                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N

                c = accumulator.to(dtype)
                c_desc.store([offs_cm, offs_cn], c)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


tlx_configs = [
    triton.Config(
        {
            "BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK, "NUM_SMEM_BUFFERS": s, "NUM_TMEM_BUFFERS": t,
            "EPILOGUE_SUBTILE": subtile, "PAIR_CTA": pair_cta, "ctas_per_cga": (2, 1, 1) if pair_cta else None
        },
        num_warps=4,
        num_stages=1,
    )
    for BM in [128]
    for BN in [128, 256]
    for BK in [64, 128]
    for s in [2, 3, 4]
    for t in [2]
    for subtile in [1, 2, 4]
    for pair_cta in [True, False]
]


@triton.autotune(
    tlx_configs,
    key=["group_a_ptrs", "group_b_ptrs", "gropup_c_ptrs", "group_size"],
)
@triton.jit
def grouped_matmul_tlx_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,  #
    NUM_TMEM_BUFFERS: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    # is the output FP8 or FP16
    FP8: tl.constexpr,
    PAIR_CTA: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8 else tl.float16

    # CTA pairs along M dim
    if PAIR_CTA:
        cluster_cta_rank = tlx.cluster_cta_rank()  # 2cta specific
        pred_cta0 = cluster_cta_rank == 0

        # 2cta specific
        cta_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS,
                                      arrive_count=2)  # CTA0 waits for CTA1's data before mma

    # allocate NUM_SMEM_BUFFERS buffers
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype, NUM_SMEM_BUFFERS)
    if PAIR_CTA:
        buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N // 2), dtype, NUM_SMEM_BUFFERS)
    else:
        buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype, NUM_SMEM_BUFFERS)
    # use multiple TMEM buffers to overlap MMA and epilogue
    tmem_buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem)

    # allocate barriers
    smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):  # epilogue consumer
            tile_idx = tl.program_id(0)
            last_problem_end = 0
            accum_cnt_tmem = 0
            for g in range(group_size):
                # get the gemm size of the current problem
                gm = tl.load(group_gemm_sizes + g * 3)
                gn = tl.load(group_gemm_sizes + g * 3 + 1)
                gk = tl.load(group_gemm_sizes + g * 3 + 2)
                num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
                if PAIR_CTA:
                    num_m_tiles = ((num_m_tiles + 1) // 2) * 2  # round up to even number
                num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
                num_tiles = num_m_tiles * num_n_tiles
                if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                    ldc = tl.load(g_lds + g * 3 + 2)
                    c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))
                    c_desc = tl.make_tensor_descriptor(
                        c_ptr,
                        shape=[gm, gn],
                        strides=[ldc, 1],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE],
                    )

                    # iterate through the tiles in the current gemm problem
                    while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                        # figure out tile coordinates
                        tile_idx_in_gemm = tile_idx - last_problem_end
                        tile_m_idx = tile_idx_in_gemm % num_m_tiles
                        tile_n_idx = tile_idx_in_gemm // num_m_tiles

                        tmem_buf, tmem_phase = _get_bufidx_phase(accum_cnt_tmem, NUM_TMEM_BUFFERS)
                        tlx.barrier_wait(tmem_full_bars[tmem_buf], tmem_phase)

                        # load the result from TMEM to registers
                        acc_tmem = tmem_buffers[tmem_buf]

                        offs_cm = tile_m_idx * BLOCK_SIZE_M
                        offs_cn = tile_n_idx * BLOCK_SIZE_N

                        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
                        # In Pair CTA mode, if there're in fact odd number of tiles along M dim, we extend it by
                        # "one virtual tile" for the non-leader CTA of the last pair. This CTA loads operand B,
                        # do cross-CTA synchronization like usual, but just don't write to result back to SMEM
                        if PAIR_CTA and offs_cm < gm:
                            for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                                acc_slice = tlx.local_slice(
                                    acc_tmem,
                                    [0, slice_id * slice_size],
                                    [BLOCK_SIZE_M, slice_size],
                                )
                                result = tlx.local_load(acc_slice)
                                c = result.to(tl.float16)
                                c_desc.store([offs_cm, offs_cn + slice_id * slice_size], c)

                        # done storing this buffer, signal MMA consumer to resume writing to it
                        tlx.barrier_arrive(tmem_empty_bars[tmem_buf], 1)
                        accum_cnt_tmem += 1
                        # go to the next tile by advancing NUM_SM
                        tile_idx += NUM_SM

                # get ready to go to the next gemm problem
                last_problem_end = last_problem_end + num_tiles

        with tlx.async_task(num_warps=1, num_regs=48):  # MMA consumer
            tile_idx = tl.program_id(0)
            last_problem_end = 0
            accum_cnt_smem = 0
            accum_cnt_tmem = 0
            for g in range(group_size):
                # get the gemm size of the current problem
                gm = tl.load(group_gemm_sizes + g * 3)
                gn = tl.load(group_gemm_sizes + g * 3 + 1)
                gk = tl.load(group_gemm_sizes + g * 3 + 2)
                num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
                if PAIR_CTA:
                    num_m_tiles = ((num_m_tiles + 1) // 2) * 2  # round up to even number
                num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
                num_tiles = num_m_tiles * num_n_tiles
                if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                    # iterate through the tiles in the current gemm problem
                    while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                        k = gk

                        # do regular gemm here
                        tmem_buf, tmem_phase = _get_bufidx_phase(accum_cnt_tmem, NUM_TMEM_BUFFERS)

                        # wait epilogue consumer to be done with the buffer before reusing it
                        tlx.barrier_wait(tmem_empty_bars[tmem_buf], tmem_phase ^ 1)

                        for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                            smem_buf, smem_phase = _get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
                            # wait for current phase(round) of load for this buf
                            tlx.barrier_wait(smem_full_bars[smem_buf], smem_phase)
                            # buffer is now ready with loaded data, tlx.async_dot will signal `mBarrier` when done
                            if PAIR_CTA:
                                cta_bar = tlx.remote_view(cta_bars[smem_buf], 0)
                                tlx.barrier_arrive(cta_bar, 1)
                                tlx.barrier_wait(cta_bar, phase=smem_phase, pred=pred_cta0)

                            tlx.async_dot(
                                buffers_A[smem_buf],
                                buffers_B[smem_buf],
                                tmem_buffers[tmem_buf],
                                use_acc=kk > 0,
                                mBarriers=[smem_empty_bars[smem_buf]],
                                two_ctas=PAIR_CTA,
                                out_dtype=tl.float32,
                            )
                            accum_cnt_smem += 1

                        # done filling this buffer, signal epilogue consumer
                        if PAIR_CTA:
                            # in pair CTA mode, we need to explicitly wait for the final MMA because tcgen05_commit
                            # would not block non-leader CTA until MMA is done
                            last_smem_buf, last_smem_phase = _get_bufidx_phase(accum_cnt_smem - 1, NUM_SMEM_BUFFERS)
                            tlx.barrier_wait(smem_empty_bars[last_smem_buf], last_smem_phase)
                            tlx.barrier_arrive(tmem_full_bars[tmem_buf], 1)
                        else:
                            tlx.tcgen05_commit(tmem_full_bars[tmem_buf])
                        accum_cnt_tmem += 1
                        # go to the next tile by advancing NUM_SM
                        tile_idx += NUM_SM

                # get ready to go to the next gemm problem
                last_problem_end = last_problem_end + num_tiles

        with tlx.async_task(num_warps=1, num_regs=48):  # producer, TMA load
            tile_idx = tl.program_id(0)
            last_problem_end = 0
            accum_cnt = 0
            accum_cnt_outer = 0

            # Allocate global scratch for tensor descriptors (pipelining)
            # We need NUM_SMEM_BUFFERS + 1 descriptor buffers to avoid descriptor conflicts:
            # A load can only be issued after the previous load (NUM_SMEM_BUFFERS stages away) completes.
            # If that previous load used a different descriptor, we need an extra buffer to ensure
            # the next load doesn't overwrite a descriptor that's still in use.
            desc_a_ptrs = tlx.allocate_tensor_descriptor(num=NUM_SMEM_BUFFERS + 1)
            desc_b_ptrs = tlx.allocate_tensor_descriptor(num=NUM_SMEM_BUFFERS + 1)

            for g in range(group_size):
                # get the gemm size of the current problem
                gm = tl.load(group_gemm_sizes + g * 3)
                gn = tl.load(group_gemm_sizes + g * 3 + 1)
                gk = tl.load(group_gemm_sizes + g * 3 + 2)
                num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
                if PAIR_CTA:
                    num_m_tiles = ((num_m_tiles + 1) // 2) * 2  # round up to even number
                num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
                num_k_tiles = tl.cdiv(gk, BLOCK_SIZE_K)
                num_tiles = num_m_tiles * num_n_tiles
                if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                    # pick up a tile from the current gemm problem
                    lda = tl.load(g_lds + g * 3)
                    ldb = tl.load(g_lds + g * 3 + 1)

                    a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
                    b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))

                    desc_buf, _ = _get_bufidx_phase(accum_cnt_outer, NUM_SMEM_BUFFERS + 1)

                    # Create tensor descriptors in global scratch (for pipelining across problems)
                    tlx.make_tensor_descriptor(
                        desc_ptr=desc_a_ptrs[desc_buf],
                        base=a_ptr,
                        shape=[gm, gk],
                        strides=[lda, 1],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                    )

                    if PAIR_CTA:
                        tlx.make_tensor_descriptor(
                            desc_ptr=desc_b_ptrs[desc_buf],
                            base=b_ptr,
                            shape=[gk, gn],
                            strides=[ldb, 1],
                            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N // 2],
                        )
                    else:
                        tlx.make_tensor_descriptor(
                            desc_ptr=desc_b_ptrs[desc_buf],
                            base=b_ptr,
                            shape=[gk, gn],
                            strides=[ldb, 1],
                            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                        )

                    # iterate through the tiles in the current gemm problem
                    while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                        # figure out tile coordinates
                        tile_idx_in_gemm = tile_idx - last_problem_end
                        tile_m_idx = tile_idx_in_gemm % num_m_tiles
                        tile_n_idx = tile_idx_in_gemm // num_m_tiles

                        # Reinterpret descriptor pointers for TMA operations
                        a_desc = tlx.reinterpret_tensor_descriptor(
                            desc_ptr=desc_a_ptrs[desc_buf],
                            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                            dtype=dtype,
                        )
                        if PAIR_CTA:
                            b_desc = tlx.reinterpret_tensor_descriptor(
                                desc_ptr=desc_b_ptrs[desc_buf],
                                block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N // 2],
                                dtype=dtype,
                            )
                        else:
                            b_desc = tlx.reinterpret_tensor_descriptor(
                                desc_ptr=desc_b_ptrs[desc_buf],
                                block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                                dtype=dtype,
                            )

                        # do regular gemm here
                        offs_am = tile_m_idx * BLOCK_SIZE_M
                        if PAIR_CTA:
                            offs_bn = tile_n_idx * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // 2)
                        else:
                            offs_bn = tile_n_idx * BLOCK_SIZE_N

                        for kk in range(0, num_k_tiles):
                            buf, phase = _get_bufidx_phase(accum_cnt, NUM_SMEM_BUFFERS)
                            tlx.barrier_wait(smem_empty_bars[buf], phase ^ 1)
                            if PAIR_CTA:
                                # todo: we can alternatively check offs_am < gm and omit loading A for the virtual tile
                                tlx.barrier_expect_bytes(smem_full_bars[buf], 2 * (BLOCK_SIZE_M + BLOCK_SIZE_N // 2) *
                                                         BLOCK_SIZE_K)  # float16
                            else:
                                tlx.barrier_expect_bytes(
                                    smem_full_bars[buf],
                                    tlx.size_of(dtype) * (BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K)
                            # todo: we can alternatively check offs_am < gm and omit loading A for the virtual tile
                            tlx.async_descriptor_load(a_desc, buffers_A[buf], [offs_am, kk * BLOCK_SIZE_K],
                                                      smem_full_bars[buf])
                            tlx.async_descriptor_load(b_desc, buffers_B[buf], [kk * BLOCK_SIZE_K, offs_bn],
                                                      smem_full_bars[buf])
                            accum_cnt += 1

                        # go to the next tile by advancing NUM_SM
                        tile_idx += NUM_SM

                    accum_cnt_outer += 1
                # get ready to go to the next gemm problem
                last_problem_end = last_problem_end + num_tiles


def group_gemm_tma_fn(group_A, group_B):
    assert supports_tma()

    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[1]
        M, K = A.shape
        N, K = B.shape
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]
    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    # we use a fixed number of CTA, and it's auto-tunable

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (META["NUM_SM"], )
    grouped_matmul_tma_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        FP8=torch.float8_e4m3fn == group_A[0].dtype,
        NUM_SM=num_sms(),
    )
    return group_C


def group_gemm_tlx_fn(group_A, group_B):
    assert supports_tma()

    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[1]
        M, K = A.shape
        N, K = B.shape
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]
    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    # we use a fixed number of CTA, and it's auto-tunable

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (META["NUM_SM"], )
    grouped_matmul_tlx_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        FP8=torch.float8_e4m3fn == group_A[0].dtype,
        NUM_SM=num_sms(),
    )
    return group_C


@pytest.mark.skipif(
    not is_blackwell(),
    reason="Requires Blackwell GPU",
)
def test_op():
    group_m = [1024, 512, 256, 128]
    group_n = [1024, 512, 256, 128]
    group_k = [1024, 512, 256, 128]
    group_A = []
    group_B = []
    group_B_T = []
    assert len(group_m) == len(group_n)
    assert len(group_n) == len(group_k)
    group_size = len(group_m)
    for i in range(group_size):
        M = group_m[i]
        N = group_n[i]
        K = group_k[i]
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)

    tri_out = group_gemm_tlx_fn(group_A, group_B)
    ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
    for i in range(group_size):
        torch.testing.assert_close(ref_out[i], tri_out[i], atol=1e-2, rtol=1e-2)


# only launch the kernel, no tensor preparation here to remove all overhead
def triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size):
    grid = lambda META: (META["NUM_SM"], )
    grouped_matmul_kernel[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        sizes,
        lds,
        group_size,
    )


def triton_tma_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size, dtype):
    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    grid = lambda META: (META["NUM_SM"], )
    grouped_matmul_tma_kernel[grid](a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size, FP8=torch.float8_e4m3fn == dtype,
                                    NUM_SM=num_sms())


def triton_tlx_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size, dtype):
    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    grid = lambda META: (META["NUM_SM"], )
    grouped_matmul_tlx_kernel[grid](a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size, FP8=torch.float8_e4m3fn == dtype,
                                    NUM_SM=num_sms())


def torch_perf_fn(group_A, group_B):
    for a, b in zip(group_A, group_B):
        torch.matmul(a, b)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=["N"],
        x_vals=[2**i for i in range(7, 11)],  # different possible values for `x_name`
        line_arg="provider",
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=["cublas", "triton"] + (["triton-tma"] if supports_tma() else []) + ["tlx"],
        # label name for the lines
        line_names=["cuBLAS", "Triton"] + (["Triton + TMA"] if supports_tma() else []) + ["TLX"],
        # line styles
        styles=[("green", "-"), ("blue", "-")] + ([("red", "-")] if supports_tma() else []) + [("orange", "-")],
        ylabel="runtime(ms)",  # label name for the y-axis
        plot_name="group-gemm-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark_square_matrices(N, provider):
    group_size = 4
    group_A = []
    group_B = []
    group_B_T = []
    A_addrs = []
    B_addrs = []
    B_T_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = torch.rand((N, N), device=DEVICE, dtype=torch.float16)
        B = torch.rand((N, N), device=DEVICE, dtype=torch.float16)
        C = torch.empty((N, N), device=DEVICE, dtype=torch.float16)
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        B_T_addrs.append(B_T.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [N, N, N]
        g_lds += [N, N, N]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_b_t_ptrs = torch.tensor(B_T_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size), quantiles=quantiles)
    if provider == "triton-tma":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_tma_perf_fn(d_a_ptrs, d_b_t_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, dtype=torch.
                                       float16),
            quantiles=quantiles,
        )
    if provider == "tlx":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_tlx_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, dtype=torch.float16
                                       ),
            quantiles=quantiles,
        )
    # Calculate TFLOPS: group_size * (2 * M * N * K) / (time_in_seconds * 1e12)
    # For square matrices: M = N = K = N
    total_flops = group_size * (2 * N * N * N)
    tflops = total_flops / (ms * 1e-3) / 1e12
    return tflops


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=["M"],
        x_vals=[2**i for i in range(7, 11)],  # different possible values for `x_name`
        line_arg="provider",
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=["cublas", "triton"] + (["triton-tma"] if supports_tma() else []) + ["tlx"],
        # label name for the lines
        line_names=["cuBLAS", "Triton"] + (["Triton + TMA"] if supports_tma() else []) + ["TLX"],
        # line styles
        styles=[("green", "-"), ("blue", "-")] + ([("red", "-")] if supports_tma() else []) + [("orange", "-")],
        ylabel="runtime(ms)",  # label name for the y-axis
        plot_name="group-gemm-performance-m-8192-k-8192",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark_batches(M, provider):
    N = 8192
    K = 8192
    group_size = 4
    group_A = []
    group_B = []
    group_B_T = []
    A_addrs = []
    B_addrs = []
    B_T_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    g_T_lds = []
    group_C = []
    for i in range(group_size):
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
        C = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        B_T_addrs.append(B_T.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]
        g_T_lds += [A.stride(0), B_T.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_b_t_ptrs = torch.tensor(B_T_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
    d_g_t_lds = torch.tensor(g_T_lds, dtype=torch.int32, device=DEVICE)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size), quantiles=quantiles)
    if provider == "triton-tma":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_tma_perf_fn(d_a_ptrs, d_b_t_ptrs, d_c_ptrs, d_g_sizes, d_g_t_lds, group_size, dtype=torch.
                                       float16),
            quantiles=quantiles,
        )
    if provider == "tlx":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_tlx_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, dtype=torch.float16
                                       ),
            quantiles=quantiles,
        )
    # Calculate TFLOPS: group_size * (2 * M * N * K) / (time_in_seconds * 1e12)
    total_flops = group_size * (2 * M * N * K)
    tflops = total_flops / (ms * 1e-3) / 1e12
    return tflops


if __name__ == "__main__":
    benchmark_square_matrices.run(show_plots=True, print_data=True)
    benchmark_batches.run(show_plots=True, print_data=True)
