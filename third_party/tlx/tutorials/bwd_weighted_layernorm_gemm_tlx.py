#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tuned TLX implementation of GEMM -> affine LayerNorm backward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tlx.ops.kernels.mm import _sm100_core as _core
from triton.tlx.ops.kernels.mm import sm100
from triton.tools.tensor_descriptor import TensorDescriptor

import bwd_weighted_layernorm_gemm as baseline


@triton.jit
def _fused_epilogue_single_slice(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    tmem_buffers,
    tmem_full_bars,
    tmem_empty_bars,
    cur_tmem_buf,
    tmem_phase,
    x_ptr,
    gamma_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    final_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    """One-register-pass epilogue for tiles small enough to stay resident."""
    pid_m, _ = _core._compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    block_m_split: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    cols = tl.arange(0, BLOCK_SIZE_N)
    gamma = tl.load(gamma_ptr + cols).to(tl.float32)
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        buf_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_phase)
        dy = tlx.local_load(tmem_buffers[buf_idx]).to(tl.bfloat16).to(tl.float32)
        if NUM_CTAS == 2:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1, remote_cta_rank=0)
        else:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)

        row_base = pid_m * BLOCK_SIZE_M + group_id * block_m_split
        rows = row_base + tl.arange(0, block_m_split)
        offsets = rows[:, None] * BLOCK_SIZE_N + cols[None, :]
        x = tl.load(x_ptr + offsets).to(tl.float32)
        mean = tl.load(mean_ptr + rows).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows).to(tl.float32)
        xhat = (x - mean[:, None]) * rstd[:, None]
        wdy = dy * gamma[None, :]
        c1 = tl.sum(xhat * wdy, axis=1, keep_dims=True) / BLOCK_SIZE_N
        c2 = tl.sum(wdy, axis=1, keep_dims=True) / BLOCK_SIZE_N
        dx = (wdy - (xhat * c1 + c2)) * rstd[:, None]
        dx = dx.to(tl.bfloat16).to(tl.float32)
        residual = tl.load(residual_ptr + offsets).to(tl.float32)
        tl.store(final_ptr + offsets, (residual + dx).to(tl.bfloat16))

        partial_row = pid_m * NUM_MMA_GROUPS + group_id
        partial_offsets = partial_row * BLOCK_SIZE_N + cols
        tl.store(partial_dw_ptr + partial_offsets, tl.sum(dy * xhat, axis=0))
        tl.store(partial_db_ptr + partial_offsets, tl.sum(dy, axis=0))


@triton.jit
def _fused_epilogue(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    tmem_buffers,
    xhat_tmem,
    tmem_full_bars,
    tmem_empty_bars,
    cur_tmem_buf,
    tmem_phase,
    x_ptr,
    gamma_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    final_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    pid_m, _ = _core._compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    block_m_split: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    slice_n: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE

    for group_id in tl.static_range(NUM_MMA_GROUPS):
        buf_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_phase)
        tmem = tmem_buffers[buf_idx]
        row_base = pid_m * BLOCK_SIZE_M + group_id * block_m_split
        rows = row_base + tl.arange(0, block_m_split)
        mean = tl.load(mean_ptr + rows).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows).to(tl.float32)
        c1 = tl.zeros((block_m_split,), tl.float32)
        c2 = tl.zeros((block_m_split,), tl.float32)

        # First pass computes the two complete-row LayerNorm reductions.
        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
            cols = slice_id * slice_n + tl.arange(0, slice_n)
            tmem_slice = tlx.local_slice(
                tmem,
                [0, slice_id * slice_n],
                [block_m_split, slice_n],
            )
            dy = tlx.local_load(tmem_slice).to(tl.bfloat16).to(tl.float32)
            offsets = rows[:, None] * BLOCK_SIZE_N + cols[None, :]
            x = tl.load(x_ptr + offsets).to(tl.float32)
            xhat = (x - mean[:, None]) * rstd[:, None]
            xhat_slice = tlx.local_slice(
                xhat_tmem[0],
                [0, slice_id * slice_n],
                [block_m_split, slice_n],
            )
            tlx.local_store(xhat_slice, xhat)
            gamma = tl.load(gamma_ptr + cols).to(tl.float32)
            wdy = dy * gamma[None, :]
            c1 += tl.sum(xhat * wdy, axis=1)
            c2 += tl.sum(wdy, axis=1)
        c1 /= BLOCK_SIZE_N
        c2 /= BLOCK_SIZE_N

        # Second pass forms dx/final and emits compact FP32 column partials.
        partial_row = pid_m * NUM_MMA_GROUPS + group_id
        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
            cols = slice_id * slice_n + tl.arange(0, slice_n)
            tmem_slice = tlx.local_slice(
                tmem,
                [0, slice_id * slice_n],
                [block_m_split, slice_n],
            )
            dy = tlx.local_load(tmem_slice).to(tl.bfloat16).to(tl.float32)
            if NUM_CTAS == 2:
                tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)

            offsets = rows[:, None] * BLOCK_SIZE_N + cols[None, :]
            xhat_slice = tlx.local_slice(
                xhat_tmem[0],
                [0, slice_id * slice_n],
                [block_m_split, slice_n],
            )
            xhat = tlx.local_load(xhat_slice).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols).to(tl.float32)
            wdy = dy * gamma[None, :]
            dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
            dx = dx.to(tl.bfloat16).to(tl.float32)
            residual = tl.load(residual_ptr + offsets).to(tl.float32)
            tl.store(final_ptr + offsets, (residual + dx).to(tl.bfloat16))

            partial_offsets = partial_row * BLOCK_SIZE_N + cols
            tl.store(partial_dw_ptr + partial_offsets, tl.sum(dy * xhat, axis=0))
            tl.store(partial_db_ptr + partial_offsets, tl.sum(dy, axis=0))


@triton.jit
def _pair_parallel_epilogue_tile(
    tile_id,
    part,
    num_pid_in_group,
    num_pid_m,
    tmem_buffers,
    xhat_tmem,
    tmem_full_bars,
    tmem_empty_bars,
    row_sum,
    row_dot,
    row_full,
    row_empty,
    tmem_count,
    x_ptr,
    gamma_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    final_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    pid_m, _ = _core._compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    part_n: tl.constexpr = BLOCK_SIZE_N // 2
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = part * part_n + tl.arange(0, part_n)
    offsets = rows[:, None] * BLOCK_SIZE_N + cols[None, :]
    tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
    tlx.barrier_wait(tmem_full_bars[tmem_buf], tmem_phase)
    dy_slice = tlx.local_slice(
        tmem_buffers[tmem_buf], [0, part * part_n], [BLOCK_SIZE_M, part_n]
    )
    dy = tlx.local_load(dy_slice).to(tl.bfloat16).to(tl.float32)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    mean = tl.load(mean_ptr + rows).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows).to(tl.float32)
    xhat = (x - mean[:, None]) * rstd[:, None]
    xhat_slice = tlx.local_slice(
        xhat_tmem[0], [0, part * part_n], [BLOCK_SIZE_M, part_n]
    )
    tlx.local_store(xhat_slice, xhat)
    gamma = tl.load(gamma_ptr + cols).to(tl.float32)
    wdy = dy * gamma[None, :]
    local_sum = tl.sum(wdy, axis=1, keep_dims=True)
    local_dot = tl.sum(xhat * wdy, axis=1, keep_dims=True)

    reduce_buf, reduce_phase = get_bufidx_phase(tmem_count, 2)
    reduce_offset = reduce_buf * 2
    tlx.barrier_wait(row_empty[reduce_buf], reduce_phase ^ 1)
    tlx.local_store(row_sum[reduce_offset + part], local_sum)
    tlx.local_store(row_dot[reduce_offset + part], local_dot)
    tlx.barrier_arrive(row_full[reduce_buf], 1)
    tlx.barrier_wait(row_full[reduce_buf], reduce_phase)
    sum_all = tl.zeros((BLOCK_SIZE_M, 1), tl.float32)
    dot_all = tl.zeros((BLOCK_SIZE_M, 1), tl.float32)
    for p in tl.static_range(2):
        sum_all += tlx.local_load(tlx.local_view(row_sum, reduce_offset + p))
        dot_all += tlx.local_load(tlx.local_view(row_dot, reduce_offset + p))
    tlx.barrier_arrive(row_empty[reduce_buf], 1)

    dy = tlx.local_load(dy_slice).to(tl.bfloat16).to(tl.float32)
    if NUM_CTAS == 2:
        tlx.barrier_arrive(tmem_empty_bars[tmem_buf], 1, remote_cta_rank=0)
    else:
        tlx.barrier_arrive(tmem_empty_bars[tmem_buf], 1)
    xhat = tlx.local_load(xhat_slice).to(tl.float32)
    wdy = dy * gamma[None, :]
    dx = (wdy - (xhat * (dot_all / BLOCK_SIZE_N) + sum_all / BLOCK_SIZE_N)) * rstd[
        :, None
    ]
    dx = dx.to(tl.bfloat16).to(tl.float32)
    residual = tl.load(residual_ptr + offsets).to(tl.float32)
    tl.store(final_ptr + offsets, (residual + dx).to(tl.bfloat16))
    partial_offsets = pid_m * BLOCK_SIZE_N + cols
    tl.store(partial_dw_ptr + partial_offsets, tl.sum(dy * xhat, axis=0))
    tl.store(partial_db_ptr + partial_offsets, tl.sum(dy, axis=0))


@triton.jit
def fused_weighted_layernorm_bwd_tlx(
    gradient_desc,
    weight_desc,
    x_ptr,
    gamma_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    final_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    PARALLEL_EPILOGUE: tl.constexpr,
):
    block_m_split: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    buffers_a = tlx.local_alloc(
        (block_m_split, BLOCK_SIZE_K),
        tlx.dtype_of(gradient_desc),
        NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
    )
    buffers_b = tlx.local_alloc(
        (BLOCK_SIZE_N // NUM_CTAS, BLOCK_SIZE_K),
        tlx.dtype_of(weight_desc),
        NUM_SMEM_BUFFERS,
    )
    tmem_buffers = tlx.local_alloc(
        (block_m_split, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )
    # Preserve FP32 xhat across the two LayerNorm passes. This avoids a second
    # global X read without introducing a new numerical rounding boundary.
    xhat_tmem = tlx.local_alloc(
        (block_m_split, BLOCK_SIZE_N),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
    )
    row_sum = tlx.local_alloc((block_m_split, 1), tl.float32, 4)
    row_dot = tlx.local_alloc((block_m_split, 1), tl.float32, 4)
    row_full = tlx.alloc_barriers(num_barriers=2, arrive_count=2)
    row_empty = tlx.alloc_barriers(num_barriers=2, arrive_count=2)

    cluster_cta_rank = tlx.cluster_cta_rank() if NUM_CTAS == 2 else 0
    a_full = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    a_empty = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    if NUM_MMA_GROUPS == 1:
        # The core's single-group path combines the A and B TMA transactions
        # on one byte-counted barrier.
        b_full = a_full
    else:
        b_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    tmem_empty = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=(2 if PARALLEL_EPILOGUE else EPILOGUE_SUBTILE) * NUM_CTAS,
    )
    clc = tlx.clc_create_context(
        num_consumers=(4 if PARALLEL_EPILOGUE else 3) * NUM_CTAS,
        num_stages=1,
    )

    with tlx.async_tasks(
        exclusive=True,
        no_ending_cluster_sync=True,
        mbarrier_try_wait_suspend_ns=50000,
    ):
        with tlx.async_task("default"):
            (
                start_pid,
                num_pid_m,
                _,
                num_pid_in_group,
                _,
                _,
                _,
            ) = _core._compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                1,
                NUM_CTAS,
            )
            tile_id = start_pid
            tmem_count = 0
            producer_phase = 1
            consumer_phase = 0
            while tile_id != -1:
                tlx.clc_producer(clc, producer_phase, multi_ctas=NUM_CTAS == 2)
                producer_phase ^= 1
                tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                if PARALLEL_EPILOGUE:
                    _pair_parallel_epilogue_tile(
                        tile_id,
                        0,
                        num_pid_in_group,
                        num_pid_m,
                        tmem_buffers,
                        xhat_tmem,
                        tmem_full,
                        tmem_empty,
                        row_sum,
                        row_dot,
                        row_full,
                        row_empty,
                        tmem_count,
                        x_ptr,
                        gamma_ptr,
                        mean_ptr,
                        rstd_ptr,
                        residual_ptr,
                        final_ptr,
                        partial_dw_ptr,
                        partial_db_ptr,
                        GROUP_SIZE_M,
                        BLOCK_SIZE_M,
                        BLOCK_SIZE_N,
                        NUM_TMEM_BUFFERS,
                        NUM_CTAS,
                    )
                elif EPILOGUE_SUBTILE == 1:
                    _fused_epilogue_single_slice(
                        tile_id,
                        num_pid_in_group,
                        num_pid_m,
                        tmem_buffers,
                        tmem_full,
                        tmem_empty,
                        tmem_buf,
                        tmem_phase,
                        x_ptr,
                        gamma_ptr,
                        mean_ptr,
                        rstd_ptr,
                        residual_ptr,
                        final_ptr,
                        partial_dw_ptr,
                        partial_db_ptr,
                        GROUP_SIZE_M,
                        BLOCK_SIZE_M,
                        BLOCK_SIZE_N,
                        NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS,
                        NUM_CTAS,
                    )
                else:
                    _fused_epilogue(
                        tile_id,
                        num_pid_in_group,
                        num_pid_m,
                        tmem_buffers,
                        xhat_tmem,
                        tmem_full,
                        tmem_empty,
                        tmem_buf,
                        tmem_phase,
                        x_ptr,
                        gamma_ptr,
                        mean_ptr,
                        rstd_ptr,
                        residual_ptr,
                        final_ptr,
                        partial_dw_ptr,
                        partial_db_ptr,
                        GROUP_SIZE_M,
                        BLOCK_SIZE_M,
                        BLOCK_SIZE_N,
                        NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS,
                        EPILOGUE_SUBTILE,
                        NUM_CTAS,
                    )
                tmem_count += 1
                tile_id = tlx.clc_consumer(
                    clc, consumer_phase, multi_ctas=NUM_CTAS == 2
                )
                consumer_phase ^= 1

        if PARALLEL_EPILOGUE:
            with tlx.async_task(num_warps=8, num_regs=128):
                (
                    start_pid,
                    num_pid_m,
                    _,
                    num_pid_in_group,
                    _,
                    _,
                    _,
                ) = _core._compute_grid_info(
                    M,
                    N,
                    K,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    GROUP_SIZE_M,
                    1,
                    NUM_CTAS,
                )
                tile_id = start_pid
                tmem_count = 0
                consumer_phase = 0
                while tile_id != -1:
                    _pair_parallel_epilogue_tile(
                        tile_id,
                        1,
                        num_pid_in_group,
                        num_pid_m,
                        tmem_buffers,
                        xhat_tmem,
                        tmem_full,
                        tmem_empty,
                        row_sum,
                        row_dot,
                        row_full,
                        row_empty,
                        tmem_count,
                        x_ptr,
                        gamma_ptr,
                        mean_ptr,
                        rstd_ptr,
                        residual_ptr,
                        final_ptr,
                        partial_dw_ptr,
                        partial_db_ptr,
                        GROUP_SIZE_M,
                        BLOCK_SIZE_M,
                        BLOCK_SIZE_N,
                        NUM_TMEM_BUFFERS,
                        NUM_CTAS,
                    )
                    tmem_count += 1
                    tile_id = tlx.clc_consumer(
                        clc, consumer_phase, multi_ctas=NUM_CTAS == 2
                    )
                    consumer_phase ^= 1

        with tlx.async_task(num_warps=1, num_regs=24):
            (
                start_pid,
                _,
                _,
                _,
                num_mn_tiles,
                _,
                k_tiles,
            ) = _core._compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                1,
                NUM_CTAS,
            )
            tile_id = start_pid
            smem_count = 0
            tmem_count = 0
            consumer_phase = 0
            while tile_id != -1:
                if NUM_CTAS == 1 or cluster_cta_rank == 0:
                    tmem_buf, tmem_phase = get_bufidx_phase(
                        tmem_count, NUM_TMEM_BUFFERS
                    )
                    smem_count = _core._process_tile_mma_inner(
                        0,
                        k_tiles,
                        NUM_SMEM_BUFFERS,
                        NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS,
                        buffers_a,
                        buffers_b,
                        tmem_buffers,
                        a_full,
                        b_full,
                        a_empty,
                        tmem_full,
                        tmem_buf,
                        tmem_empty,
                        tmem_phase,
                        smem_count,
                        NUM_CTAS,
                        A_ROW_MAJOR=True,
                        B_ROW_MAJOR=False,
                    )
                else:
                    smem_count += k_tiles
                tmem_count += 1
                tile_id = tlx.clc_consumer(
                    clc, consumer_phase, multi_ctas=NUM_CTAS == 2
                )
                consumer_phase ^= 1

        with tlx.async_task(num_warps=1, num_regs=24):
            (
                start_pid,
                num_pid_m,
                _,
                num_pid_in_group,
                num_mn_tiles,
                _,
                k_tiles,
            ) = _core._compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                1,
                NUM_CTAS,
            )
            tile_id = start_pid
            smem_count = 0
            consumer_phase = 0
            while tile_id != -1:
                smem_count = _core._process_tile_producer_inner(
                    tile_id,
                    num_pid_in_group,
                    num_pid_m,
                    num_mn_tiles,
                    GROUP_SIZE_M,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    NUM_MMA_GROUPS,
                    0,
                    k_tiles,
                    NUM_SMEM_BUFFERS,
                    gradient_desc,
                    weight_desc,
                    buffers_a,
                    buffers_b,
                    a_full,
                    b_full,
                    a_empty,
                    smem_count,
                    NUM_CTAS,
                    cluster_cta_rank,
                    1,
                    A_ROW_MAJOR=True,
                    B_ROW_MAJOR=False,
                )
                tile_id = tlx.clc_consumer(
                    clc, consumer_phase, multi_ctas=NUM_CTAS == 2
                )
                consumer_phase ^= 1


@triton.jit
def _nsplit_mma_tile(
    k_tiles,
    buffers_a,
    buffers_b,
    tmem,
    a_full,
    b_full,
    a_empty,
    tmem_full,
    tmem_empty,
    smem_count,
    tmem_buf,
    tmem_phase,
    NUM_SMEM_BUFFERS: tl.constexpr,
):
    buf, phase = get_bufidx_phase(smem_count, NUM_SMEM_BUFFERS)
    tlx.barrier_wait(a_full[buf], phase)
    tlx.barrier_wait(b_full[buf], phase)
    tlx.barrier_wait(tmem_empty[tmem_buf], tmem_phase ^ 1)
    tlx.async_dot(
        buffers_a[buf],
        tlx.local_trans(buffers_b[buf]),
        tmem[tmem_buf],
        use_acc=False,
        mBarriers=[a_empty[buf]],
        out_dtype=tl.float32,
    )
    smem_count += 1
    for _ in range(1, k_tiles):
        buf, phase = get_bufidx_phase(smem_count, NUM_SMEM_BUFFERS)
        tlx.barrier_wait(a_full[buf], phase)
        tlx.barrier_wait(b_full[buf], phase)
        tlx.async_dot(
            buffers_a[buf],
            tlx.local_trans(buffers_b[buf]),
            tmem[tmem_buf],
            use_acc=True,
            mBarriers=[a_empty[buf]],
            out_dtype=tl.float32,
        )
        smem_count += 1
    last_buf, last_phase = get_bufidx_phase(smem_count - 1, NUM_SMEM_BUFFERS)
    tlx.barrier_wait(a_empty[last_buf], last_phase)
    tlx.barrier_arrive(tmem_full[tmem_buf], 1)
    return smem_count


@triton.jit
def _nsplit_producer_tile(
    tile_id,
    num_pid_n,
    k_tiles,
    gradient_desc,
    weight_desc,
    buffers_a,
    buffers_b,
    a_full,
    b_full,
    a_empty,
    smem_count,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
):
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    expected_a: tl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_K * 2
    expected_b: tl.constexpr = BLOCK_SIZE_N * BLOCK_SIZE_K * 2
    for k in range(0, k_tiles):
        buf, phase = get_bufidx_phase(smem_count, NUM_SMEM_BUFFERS)
        tlx.barrier_wait(a_empty[buf], phase ^ 1)
        tlx.barrier_expect_bytes(a_full[buf], expected_a)
        tlx.async_descriptor_load(
            gradient_desc,
            buffers_a[buf],
            [pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K],
            a_full[buf],
            eviction_policy="evict_first",
        )
        tlx.barrier_expect_bytes(b_full[buf], expected_b)
        tlx.async_descriptor_load(
            weight_desc,
            buffers_b[buf],
            [pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K],
            b_full[buf],
            eviction_policy="evict_last",
        )
        smem_count += 1
    return smem_count


@triton.jit
def fused_weighted_layernorm_bwd_nsplit_tlx(
    gradient_desc,
    weight_desc,
    x_ptr,
    gamma_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    final_ptr,
    dweight_acc_ptr,
    dbias_acc_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    CACHE_XHAT: tl.constexpr,
    CACHE_XHAT_SMEM: tl.constexpr,
    COL_STATS: tl.constexpr,
    STATS_IN_REGS: tl.constexpr,
    SPLIT_STATS: tl.constexpr,
):
    buffers_a = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_K),
        tlx.dtype_of(gradient_desc),
        NUM_SMEM_BUFFERS,
    )
    buffers_b = tlx.local_alloc(
        (BLOCK_SIZE_N, BLOCK_SIZE_K),
        tlx.dtype_of(weight_desc),
        NUM_SMEM_BUFFERS,
    )
    tmem = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS,
        tlx.storage_kind.tmem,
    )
    xhat_tmem = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
    )
    xhat_smem = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, 1)
    dw_accum = tlx.local_alloc((1, BLOCK_SIZE_N), tl.float32, 1)
    db_accum = tlx.local_alloc((1, BLOCK_SIZE_N), tl.float32, 1)

    a_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    b_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS,
        arrive_count=EPILOGUE_SUBTILE * (2 if SPLIT_STATS else 1),
    )

    num_dsmem_buffers: tl.constexpr = 2
    reduce_full = tlx.alloc_barriers(num_barriers=num_dsmem_buffers)
    reduce_empty = tlx.alloc_barriers(num_barriers=num_dsmem_buffers, arrive_count=1)
    reduce_sum = tlx.local_alloc((BLOCK_SIZE_M, 1), tl.float32, 2 * num_dsmem_buffers)
    reduce_dot = tlx.local_alloc((BLOCK_SIZE_M, 1), tl.float32, 2 * num_dsmem_buffers)
    expected_reduce_bytes: tl.constexpr = BLOCK_SIZE_M * 4 * 2
    cta_rank = tlx.cluster_cta_rank()
    clc = tlx.clc_create_context(
        num_consumers=(4 if SPLIT_STATS else 3) * 2, num_stages=1
    )

    with tlx.async_tasks():
        with tlx.async_task("default"):
            tlx.cluster_barrier()
            start_pid = tl.program_id(0)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            tile_id = start_pid
            tmem_count = 0
            producer_phase = 1
            consumer_phase = 0
            reduce_buf = 0
            reduce_full_phase = 0
            reduce_empty_phase = 1
            dw_regs = tl.zeros((1, BLOCK_SIZE_N), tl.float32)
            db_regs = tl.zeros((1, BLOCK_SIZE_N), tl.float32)
            if COL_STATS and not SPLIT_STATS and not STATS_IN_REGS:
                tlx.local_store(dw_accum[0], tl.zeros((1, BLOCK_SIZE_N), tl.float32))
                tlx.local_store(db_accum[0], tl.zeros((1, BLOCK_SIZE_N), tl.float32))

            while tile_id != -1:
                tlx.clc_producer(clc, producer_phase, multi_ctas=True)
                producer_phase ^= 1
                pid_m = tile_id // num_pid_n
                pid_n = tile_id % num_pid_n
                rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offsets = rows[:, None] * N + cols[None, :]
                tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(tmem_full[tmem_buf], tmem_phase)
                dy_tmem = tmem[tmem_buf]

                local_sum = tl.zeros((BLOCK_SIZE_M, 1), tl.float32)
                local_dot = tl.zeros((BLOCK_SIZE_M, 1), tl.float32)
                slice_n: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
                mean = tl.load(mean_ptr + rows).to(tl.float32)
                rstd = tl.load(rstd_ptr + rows).to(tl.float32)
                for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                    local_cols = slice_id * slice_n + tl.arange(0, slice_n)
                    dy_slice = tlx.local_slice(
                        dy_tmem,
                        [0, slice_id * slice_n],
                        [BLOCK_SIZE_M, slice_n],
                    )
                    dy = tlx.local_load(dy_slice).to(tl.bfloat16).to(tl.float32)
                    x = tl.load(
                        x_ptr
                        + rows[:, None] * N
                        + (pid_n * BLOCK_SIZE_N + local_cols)[None, :]
                    ).to(tl.float32)
                    xhat = (x - mean[:, None]) * rstd[:, None]
                    xhat_slice = tlx.local_slice(
                        xhat_tmem[0],
                        [0, slice_id * slice_n],
                        [BLOCK_SIZE_M, slice_n],
                    )
                    if CACHE_XHAT_SMEM:
                        tlx.local_store(xhat_smem[0], xhat)
                    elif CACHE_XHAT:
                        tlx.local_store(xhat_slice, xhat)
                    gamma = tl.load(gamma_ptr + pid_n * BLOCK_SIZE_N + local_cols).to(
                        tl.float32
                    )
                    wdy = dy * gamma[None, :]
                    local_sum += tl.sum(wdy, axis=1, keep_dims=True)
                    local_dot += tl.sum(xhat * wdy, axis=1, keep_dims=True)

                tlx.barrier_wait(reduce_empty[reduce_buf], reduce_empty_phase)
                tlx.barrier_expect_bytes(reduce_full[reduce_buf], expected_reduce_bytes)
                reduce_offset = reduce_buf * 2
                tlx.local_store(reduce_sum[reduce_offset + cta_rank], local_sum)
                tlx.local_store(reduce_dot[reduce_offset + cta_rank], local_dot)
                other_rank = 1 - cta_rank
                tlx.async_remote_shmem_store(
                    dst=reduce_sum[reduce_offset + cta_rank],
                    src=local_sum,
                    remote_cta_rank=other_rank,
                    barrier=reduce_full[reduce_buf],
                )
                tlx.async_remote_shmem_store(
                    dst=reduce_dot[reduce_offset + cta_rank],
                    src=local_dot,
                    remote_cta_rank=other_rank,
                    barrier=reduce_full[reduce_buf],
                )
                tlx.barrier_wait(reduce_full[reduce_buf], reduce_full_phase)
                sum_all = tl.zeros((BLOCK_SIZE_M, 1), tl.float32)
                dot_all = tl.zeros((BLOCK_SIZE_M, 1), tl.float32)
                for rank in tl.static_range(2):
                    sum_all += tlx.local_load(
                        tlx.local_view(reduce_sum, reduce_offset + rank)
                    )
                    dot_all += tlx.local_load(
                        tlx.local_view(reduce_dot, reduce_offset + rank)
                    )
                mean_wdy = sum_all / N
                mean_dot = dot_all / N

                for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                    local_cols = slice_id * slice_n + tl.arange(0, slice_n)
                    dy_slice = tlx.local_slice(
                        dy_tmem,
                        [0, slice_id * slice_n],
                        [BLOCK_SIZE_M, slice_n],
                    )
                    dy = tlx.local_load(dy_slice).to(tl.bfloat16).to(tl.float32)
                    tlx.barrier_arrive(tmem_empty[tmem_buf], 1)
                    xhat_slice = tlx.local_slice(
                        xhat_tmem[0],
                        [0, slice_id * slice_n],
                        [BLOCK_SIZE_M, slice_n],
                    )
                    if CACHE_XHAT_SMEM:
                        xhat = tlx.local_load(xhat_smem[0]).to(tl.float32)
                    elif CACHE_XHAT:
                        xhat = tlx.local_load(xhat_slice).to(tl.float32)
                    else:
                        x = tl.load(
                            x_ptr
                            + rows[:, None] * N
                            + (pid_n * BLOCK_SIZE_N + local_cols)[None, :]
                        ).to(tl.float32)
                        xhat = (x - mean[:, None]) * rstd[:, None]
                    gamma = tl.load(gamma_ptr + pid_n * BLOCK_SIZE_N + local_cols).to(
                        tl.float32
                    )
                    wdy = dy * gamma[None, :]
                    dx = (wdy - (xhat * mean_dot + mean_wdy)) * rstd[:, None]
                    dx = dx.to(tl.bfloat16).to(tl.float32)
                    output_offsets = (
                        rows[:, None] * N + (pid_n * BLOCK_SIZE_N + local_cols)[None, :]
                    )
                    residual = tl.load(residual_ptr + output_offsets).to(tl.float32)
                    tl.store(
                        final_ptr + output_offsets,
                        (residual + dx).to(tl.bfloat16),
                    )

                    if COL_STATS and not SPLIT_STATS:
                        partial_dw = tl.sum(dy * xhat, axis=0, keep_dims=True)
                        partial_db = tl.sum(dy, axis=0, keep_dims=True)
                        if STATS_IN_REGS:
                            dw_regs += partial_dw
                            db_regs += partial_db
                        else:
                            dw_slice = tlx.local_slice(
                                dw_accum[0],
                                [0, slice_id * slice_n],
                                [1, slice_n],
                            )
                            db_slice = tlx.local_slice(
                                db_accum[0],
                                [0, slice_id * slice_n],
                                [1, slice_n],
                            )
                            dw = tlx.local_load(dw_slice)
                            db = tlx.local_load(db_slice)
                            tlx.local_store(dw_slice, dw + partial_dw)
                            tlx.local_store(db_slice, db + partial_db)

                tlx.barrier_arrive(
                    reduce_empty[reduce_buf], 1, remote_cta_rank=other_rank
                )
                reduce_buf ^= 1
                if reduce_buf == 0:
                    reduce_full_phase ^= 1
                    reduce_empty_phase ^= 1
                tmem_count += 1
                tile_id = tlx.clc_consumer(clc, consumer_phase, multi_ctas=True)
                consumer_phase ^= 1

            if COL_STATS and not SPLIT_STATS:
                if STATS_IN_REGS:
                    final_dw = dw_regs
                    final_db = db_regs
                else:
                    final_dw = tlx.local_load(dw_accum[0])
                    final_db = tlx.local_load(db_accum[0])
                final_cols = cta_rank * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                tl.atomic_add(
                    dweight_acc_ptr + final_cols,
                    tl.reshape(final_dw, (BLOCK_SIZE_N,)),
                )
                tl.atomic_add(
                    dbias_acc_ptr + final_cols,
                    tl.reshape(final_db, (BLOCK_SIZE_N,)),
                )

        if SPLIT_STATS:
            with tlx.async_task(num_warps=4, num_regs=128):
                tlx.cluster_barrier()
                start_pid = tl.program_id(0)
                num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
                tile_id = start_pid
                tmem_count = 0
                consumer_phase = 0
                dw_regs = tl.zeros((1, BLOCK_SIZE_N), tl.float32)
                db_regs = tl.zeros((1, BLOCK_SIZE_N), tl.float32)
                while tile_id != -1:
                    pid_m = tile_id // num_pid_n
                    pid_n = tile_id % num_pid_n
                    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offsets = rows[:, None] * N + cols[None, :]
                    tmem_buf, tmem_phase = get_bufidx_phase(
                        tmem_count, NUM_TMEM_BUFFERS
                    )
                    tlx.barrier_wait(tmem_full[tmem_buf], tmem_phase)
                    dy = tlx.local_load(tmem[tmem_buf]).to(tl.bfloat16).to(tl.float32)
                    tlx.barrier_arrive(tmem_empty[tmem_buf], 1)
                    x = tl.load(x_ptr + offsets).to(tl.float32)
                    mean = tl.load(mean_ptr + rows).to(tl.float32)
                    rstd = tl.load(rstd_ptr + rows).to(tl.float32)
                    xhat = (x - mean[:, None]) * rstd[:, None]
                    dw_regs += tl.sum(dy * xhat, axis=0, keep_dims=True)
                    db_regs += tl.sum(dy, axis=0, keep_dims=True)
                    tmem_count += 1
                    tile_id = tlx.clc_consumer(clc, consumer_phase, multi_ctas=True)
                    consumer_phase ^= 1

                final_cols = cta_rank * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                tl.atomic_add(
                    dweight_acc_ptr + final_cols,
                    tl.reshape(dw_regs, (BLOCK_SIZE_N,)),
                )
                tl.atomic_add(
                    dbias_acc_ptr + final_cols,
                    tl.reshape(db_regs, (BLOCK_SIZE_N,)),
                )

        with tlx.async_task(num_warps=1, num_regs=24):
            tlx.cluster_barrier()
            start_pid = tl.program_id(0)
            tile_id = start_pid
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            smem_count = 0
            tmem_count = 0
            consumer_phase = 0
            while tile_id != -1:
                tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                smem_count = _nsplit_mma_tile(
                    k_tiles,
                    buffers_a,
                    buffers_b,
                    tmem,
                    a_full,
                    b_full,
                    a_empty,
                    tmem_full,
                    tmem_empty,
                    smem_count,
                    tmem_buf,
                    tmem_phase,
                    NUM_SMEM_BUFFERS,
                )
                tmem_count += 1
                tile_id = tlx.clc_consumer(clc, consumer_phase, multi_ctas=True)
                consumer_phase ^= 1

        with tlx.async_task(num_warps=1, num_regs=24):
            tlx.cluster_barrier()
            start_pid = tl.program_id(0)
            tile_id = start_pid
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            smem_count = 0
            consumer_phase = 0
            while tile_id != -1:
                smem_count = _nsplit_producer_tile(
                    tile_id,
                    num_pid_n,
                    k_tiles,
                    gradient_desc,
                    weight_desc,
                    buffers_a,
                    buffers_b,
                    a_full,
                    b_full,
                    a_empty,
                    smem_count,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    NUM_SMEM_BUFFERS,
                )
                tile_id = tlx.clc_consumer(clc, consumer_phase, multi_ctas=True)
                consumer_phase ^= 1

        with tlx.async_task(num_warps=2, num_regs=24):
            tlx.cluster_barrier()


CONFIG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 2,
    "NUM_SMEM_BUFFERS": 2,
    "NUM_TMEM_BUFFERS": 1,
    "NUM_MMA_GROUPS": 2,
    "EPILOGUE_SUBTILE": 16,
    "NUM_CTAS": 2,
    "PARALLEL_EPILOGUE": False,
}

NSPLIT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "NUM_SMEM_BUFFERS": 6,
    "NUM_TMEM_BUFFERS": 2,
    "EPILOGUE_SUBTILE": 1,
    "CACHE_XHAT": False,
    "CACHE_XHAT_SMEM": False,
    "COL_STATS": True,
    "STATS_IN_REGS": True,
    "SPLIT_STATS": False,
}


def make_outputs(rows: int) -> baseline.TensorMap:
    partial_rows = triton.cdiv(rows, CONFIG["BLOCK_SIZE_M"]) * CONFIG["NUM_MMA_GROUPS"]
    return baseline.make_outputs(rows, partial_rows)


def run_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    *,
    epilogue_warps: int = 4,
) -> None:
    rows = inputs["gradient"].shape[0]
    block_m_split = CONFIG["BLOCK_SIZE_M"] // CONFIG["NUM_MMA_GROUPS"]
    gradient_desc = TensorDescriptor(
        inputs["gradient"],
        inputs["gradient"].shape,
        inputs["gradient"].stride(),
        [block_m_split, CONFIG["BLOCK_SIZE_K"]],
    )
    weight_desc = TensorDescriptor(
        inputs["projection_weight"],
        inputs["projection_weight"].shape,
        inputs["projection_weight"].stride(),
        [CONFIG["BLOCK_SIZE_N"] // CONFIG["NUM_CTAS"], CONFIG["BLOCK_SIZE_K"]],
    )
    num_pid_m = sm100._padded_num_pid_m(
        rows, CONFIG["BLOCK_SIZE_M"], CONFIG["NUM_CTAS"]
    )
    grid = (num_pid_m,)
    launch_options = {"ctas_per_cga": (2, 1, 1)} if CONFIG["NUM_CTAS"] == 2 else {}
    cast(Any, fused_weighted_layernorm_bwd_tlx)[grid](
        gradient_desc,
        weight_desc,
        inputs["x"],
        inputs["gamma"],
        inputs["mean"],
        inputs["rstd"],
        inputs["residual"],
        outputs["final"],
        outputs["partial_dw"],
        outputs["partial_db"],
        rows,
        baseline.FEATURES,
        baseline.GEMM_K,
        **CONFIG,
        num_warps=epilogue_warps,
        num_stages=1,
        **launch_options,
    )
    partial_rows = num_pid_m * CONFIG["NUM_MMA_GROUPS"]
    cast(Any, baseline.finish_weighted_layernorm_dwdb)[
        (triton.cdiv(baseline.FEATURES, 32),)
    ](
        outputs["partial_dw"],
        outputs["partial_db"],
        outputs["dweight"],
        outputs["dbias"],
        partial_rows,
        N=baseline.FEATURES,
        BLOCK_ROWS=256,
        BLOCK_N=32,
        num_warps=8,
    )


def run_tlx_nsplit(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    accum_dw: torch.Tensor,
    accum_db: torch.Tensor,
    *,
    epilogue_warps: int = 8,
) -> None:
    rows = inputs["gradient"].shape[0]
    accum_dw.zero_()
    accum_db.zero_()
    gradient_desc = TensorDescriptor(
        inputs["gradient"],
        inputs["gradient"].shape,
        inputs["gradient"].stride(),
        [NSPLIT_CONFIG["BLOCK_SIZE_M"], NSPLIT_CONFIG["BLOCK_SIZE_K"]],
    )
    weight_desc = TensorDescriptor(
        inputs["projection_weight"],
        inputs["projection_weight"].shape,
        inputs["projection_weight"].stride(),
        [NSPLIT_CONFIG["BLOCK_SIZE_N"], NSPLIT_CONFIG["BLOCK_SIZE_K"]],
    )
    grid = (
        triton.cdiv(rows, NSPLIT_CONFIG["BLOCK_SIZE_M"])
        * triton.cdiv(baseline.FEATURES, NSPLIT_CONFIG["BLOCK_SIZE_N"]),
    )
    cast(Any, fused_weighted_layernorm_bwd_nsplit_tlx)[grid](
        gradient_desc,
        weight_desc,
        inputs["x"],
        inputs["gamma"],
        inputs["mean"],
        inputs["rstd"],
        inputs["residual"],
        outputs["final"],
        accum_dw,
        accum_db,
        rows,
        baseline.FEATURES,
        baseline.GEMM_K,
        **NSPLIT_CONFIG,
        num_warps=epilogue_warps,
        num_stages=1,
        ctas_per_cga=(2, 1, 1),
    )
    cast(Any, baseline.cast_dwdb)[(1,)](
        accum_dw,
        accum_db,
        outputs["dweight"],
        outputs["dbias"],
        N=baseline.FEATURES,
        num_warps=4,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=baseline.DEFAULT_ROWS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--verification-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.rows % NSPLIT_CONFIG["BLOCK_SIZE_M"]:
        parser.error(f"--rows must be divisible by {NSPLIT_CONFIG['BLOCK_SIZE_M']}")
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")

    inputs = baseline.make_inputs(args.rows, 0)
    reference = baseline.make_outputs(
        args.rows, triton.cdiv(args.rows, baseline.SEED_BLOCK_M)
    )
    candidate = baseline.make_outputs(args.rows, 1)
    accum_dw = torch.empty((baseline.FEATURES,), device="cuda", dtype=torch.float32)
    accum_db = torch.empty_like(accum_dw)
    baseline.run_unfused(inputs, reference)
    run_tlx_nsplit(inputs, candidate, accum_dw, accum_db)
    torch.cuda.synchronize()
    accuracy_by_seed = {"0": baseline.accuracy(reference, candidate)}
    for seed in dict.fromkeys(args.verification_seeds):
        if seed == 0:
            continue
        seed_inputs = baseline.make_inputs(args.rows, seed)
        seed_reference = baseline.make_outputs(
            args.rows, triton.cdiv(args.rows, baseline.SEED_BLOCK_M)
        )
        seed_candidate = baseline.make_outputs(args.rows, 1)
        seed_accum_dw = torch.empty_like(accum_dw)
        seed_accum_db = torch.empty_like(accum_db)
        baseline.run_unfused(seed_inputs, seed_reference)
        run_tlx_nsplit(
            seed_inputs,
            seed_candidate,
            seed_accum_dw,
            seed_accum_db,
        )
        torch.cuda.synchronize()
        accuracy_by_seed[str(seed)] = baseline.accuracy(seed_reference, seed_candidate)

    passed = all(
        metric["passed"]
        for seed_result in accuracy_by_seed.values()
        for metric in seed_result.values()
    )
    report: dict[str, Any] = {
        "shape_mnk": [args.rows, baseline.FEATURES, baseline.GEMM_K],
        "config": NSPLIT_CONFIG,
        "accuracy": accuracy_by_seed,
        "passed": passed,
    }
    if not args.check_only:
        seed_outputs = baseline.make_outputs(
            args.rows, triton.cdiv(args.rows, baseline.SEED_BLOCK_M)
        )
        timings = {
            "unfused_ms": baseline.benchmark(
                lambda: baseline.run_unfused(inputs, reference),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused_seed_ms": baseline.benchmark(
                lambda: baseline.run_seed(inputs, seed_outputs),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused_tlx_ms": baseline.benchmark(
                lambda: run_tlx_nsplit(inputs, candidate, accum_dw, accum_db),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
        }
        report["timings"] = timings
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
