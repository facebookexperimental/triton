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

"""TLX saved-LayerNorm GEMM prologue prototypes for T290048886."""

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
from triton.tlx.ops.kernels.mm import sm100
from triton.tools.tensor_descriptor import TensorDescriptor

import bwd_saved_layernorm_gemm as baseline
import bwd_layernorm_mul_dropout_buf157_tlx as splitk_base


@triton.jit
def saved_layernorm_projection_tlx(
    x_desc,
    weight_desc,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_SLOTS: tl.constexpr,
    NUM_B_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NORM_WARPS: tl.constexpr,
    NORM_REGS: tl.constexpr,
):
    k_tiles: tl.constexpr = K // BLOCK_K
    n_tiles: tl.constexpr = N // BLOCK_N
    a_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, A_SLOTS * k_tiles)
    b_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.bfloat16, NUM_B_BUFFERS)
    tmem = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        NUM_TMEM_BUFFERS,
        tlx.storage_kind.tmem,
    )
    a_full = tlx.alloc_barriers(A_SLOTS * k_tiles, arrive_count=1)
    a_normalized = tlx.alloc_barriers(A_SLOTS, arrive_count=1)
    a_empty = tlx.alloc_barriers(A_SLOTS, arrive_count=1)
    b_full = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    b_empty = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    tmem_full = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    num_m_tiles: tl.constexpr = M // BLOCK_M

    with tlx.async_tasks():
        with tlx.async_task("default"):
            pid_m = tl.program_id(0)
            tmem_count = 0
            while pid_m < num_m_tiles:
                for pid_n in tl.static_range(n_tiles):
                    tmem_buf, phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                    tlx.barrier_wait(tmem_full[tmem_buf], phase)
                    result = tlx.local_load(tmem[tmem_buf])
                    tlx.barrier_arrive(tmem_empty[tmem_buf], 1)
                    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                    bias = tl.load(bias_ptr + columns).to(tl.float32)
                    offsets = rows[:, None] * N + columns[None, :]
                    tl.store(
                        output_ptr + offsets,
                        (result + bias[None, :]).to(tl.bfloat16),
                    )
                    tmem_count += 1
                pid_m += NUM_PROGRAMS

        with tlx.async_task(num_warps=1, num_regs=24):
            pid_m = tl.program_id(0)
            tile_count = 0
            b_count = 0
            tmem_count = 0
            while pid_m < num_m_tiles:
                a_slot, a_phase = get_bufidx_phase(tile_count, A_SLOTS)
                tlx.barrier_wait(a_normalized[a_slot], a_phase)
                for pid_n in tl.static_range(n_tiles):
                    tmem_buf, tmem_phase = get_bufidx_phase(
                        tmem_count, NUM_TMEM_BUFFERS
                    )
                    tlx.barrier_wait(tmem_empty[tmem_buf], tmem_phase ^ 1)
                    for k in tl.static_range(k_tiles):
                        a_buf = a_slot * k_tiles + k
                        b_buf, b_phase = get_bufidx_phase(b_count, NUM_B_BUFFERS)
                        tlx.barrier_wait(b_full[b_buf], b_phase)
                        if pid_n == n_tiles - 1 and k == k_tiles - 1:
                            tlx.async_dot(
                                a_buffers[a_buf],
                                b_buffers[b_buf],
                                tmem[tmem_buf],
                                use_acc=k > 0,
                                mBarriers=[b_empty[b_buf], a_empty[a_slot]],
                                out_dtype=tl.float32,
                            )
                        else:
                            tlx.async_dot(
                                a_buffers[a_buf],
                                b_buffers[b_buf],
                                tmem[tmem_buf],
                                use_acc=k > 0,
                                mBarriers=[b_empty[b_buf]],
                                out_dtype=tl.float32,
                            )
                        b_count += 1
                    last_b, last_phase = get_bufidx_phase(b_count - 1, NUM_B_BUFFERS)
                    tlx.barrier_wait(b_empty[last_b], last_phase)
                    tlx.barrier_arrive(tmem_full[tmem_buf], 1)
                    tmem_count += 1
                tile_count += 1
                pid_m += NUM_PROGRAMS

        with tlx.async_task(num_warps=NORM_WARPS, num_regs=NORM_REGS):
            pid_m = tl.program_id(0)
            tile_count = 0
            while pid_m < num_m_tiles:
                a_slot, a_phase = get_bufidx_phase(tile_count, A_SLOTS)
                rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                mean = tl.load(mean_ptr + rows).to(tl.float32)
                rstd = tl.load(rstd_ptr + rows).to(tl.float32)
                columns = tl.arange(0, BLOCK_K)
                for k in range(k_tiles):
                    a_buf = a_slot * k_tiles + k
                    tlx.barrier_wait(a_full[a_buf], a_phase)
                    x = tlx.local_load(a_buffers[a_buf]).to(tl.float32)
                    gamma = tl.load(gamma_ptr + k * BLOCK_K + columns).to(tl.float32)
                    beta = tl.load(beta_ptr + k * BLOCK_K + columns).to(tl.float32)
                    hidden = (x - mean[:, None]) * rstd[:, None]
                    hidden = hidden * gamma[None, :] + beta[None, :]
                    tlx.local_store(a_buffers[a_buf], hidden.to(tl.bfloat16))
                tlx.fence("async_shared")
                tlx.barrier_arrive(a_normalized[a_slot], 1)
                tile_count += 1
                pid_m += NUM_PROGRAMS

        with tlx.async_task(num_warps=1, num_regs=40):
            pid_m = tl.program_id(0)
            tile_count = 0
            b_count = 0
            while pid_m < num_m_tiles:
                a_slot, a_phase = get_bufidx_phase(tile_count, A_SLOTS)
                tlx.barrier_wait(a_empty[a_slot], a_phase ^ 1)
                for k in tl.static_range(k_tiles):
                    a_buf = a_slot * k_tiles + k
                    tlx.barrier_expect_bytes(a_full[a_buf], 2 * BLOCK_M * BLOCK_K)
                    tlx.async_descriptor_load(
                        x_desc,
                        a_buffers[a_buf],
                        [pid_m * BLOCK_M, k * BLOCK_K],
                        a_full[a_buf],
                    )
                for pid_n in tl.static_range(n_tiles):
                    for k in tl.static_range(k_tiles):
                        b_buf, b_phase = get_bufidx_phase(b_count, NUM_B_BUFFERS)
                        tlx.barrier_wait(b_empty[b_buf], b_phase ^ 1)
                        tlx.barrier_expect_bytes(b_full[b_buf], 2 * BLOCK_K * BLOCK_N)
                        tlx.async_descriptor_load(
                            weight_desc,
                            b_buffers[b_buf],
                            [k * BLOCK_K, pid_n * BLOCK_N],
                            b_full[b_buf],
                            eviction_policy="evict_last",
                        )
                        b_count += 1
                tile_count += 1
                pid_m += NUM_PROGRAMS


@triton.jit
def saved_layernorm_projection_weight_stationary_tlx(
    x_desc,
    weight_desc,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PROGRAMS_PER_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_SLOTS: tl.constexpr,
    NUM_B_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NORM_WARPS: tl.constexpr,
    NORM_REGS: tl.constexpr,
):
    k_tiles: tl.constexpr = K // BLOCK_K
    n_tiles: tl.constexpr = N // BLOCK_N
    a_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, A_SLOTS * k_tiles)
    tl.static_assert(NUM_B_BUFFERS == k_tiles)
    b_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.bfloat16, NUM_B_BUFFERS)
    tmem = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        NUM_TMEM_BUFFERS,
        tlx.storage_kind.tmem,
    )
    a_full = tlx.alloc_barriers(A_SLOTS * k_tiles, arrive_count=1)
    a_normalized = tlx.alloc_barriers(A_SLOTS, arrive_count=1)
    a_empty = tlx.alloc_barriers(A_SLOTS, arrive_count=1)
    b_full = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    tmem_full = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    pid = tl.program_id(0)
    pid_n = pid % n_tiles
    first_pid_m = pid // n_tiles
    num_m_tiles: tl.constexpr = M // BLOCK_M

    with tlx.async_tasks():
        with tlx.async_task("default"):
            pid_m = first_pid_m
            tmem_count = 0
            while pid_m < num_m_tiles:
                tmem_buf, phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(tmem_full[tmem_buf], phase)
                result = tlx.local_load(tmem[tmem_buf])
                tlx.barrier_arrive(tmem_empty[tmem_buf], 1)
                rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                bias = tl.load(bias_ptr + columns).to(tl.float32)
                offsets = rows[:, None] * N + columns[None, :]
                tl.store(
                    output_ptr + offsets,
                    (result + bias[None, :]).to(tl.bfloat16),
                )
                tmem_count += 1
                pid_m += PROGRAMS_PER_N

        with tlx.async_task(num_warps=1, num_regs=24):
            pid_m = first_pid_m
            tile_count = 0
            tmem_count = 0
            while pid_m < num_m_tiles:
                a_slot, a_phase = get_bufidx_phase(tile_count, A_SLOTS)
                tlx.barrier_wait(a_normalized[a_slot], a_phase)
                tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(tmem_empty[tmem_buf], tmem_phase ^ 1)
                for k in tl.static_range(k_tiles):
                    a_buf = a_slot * k_tiles + k
                    tlx.barrier_wait(b_full[k], 0)
                    if k == k_tiles - 1:
                        tlx.async_dot(
                            a_buffers[a_buf],
                            b_buffers[k],
                            tmem[tmem_buf],
                            use_acc=k > 0,
                            mBarriers=[a_empty[a_slot]],
                            out_dtype=tl.float32,
                        )
                    else:
                        tlx.async_dot(
                            a_buffers[a_buf],
                            b_buffers[k],
                            tmem[tmem_buf],
                            use_acc=k > 0,
                            out_dtype=tl.float32,
                        )
                tlx.barrier_wait(a_empty[a_slot], a_phase)
                tlx.barrier_arrive(tmem_full[tmem_buf], 1)
                tile_count += 1
                tmem_count += 1
                pid_m += PROGRAMS_PER_N

        with tlx.async_task(num_warps=NORM_WARPS, num_regs=NORM_REGS):
            pid_m = first_pid_m
            tile_count = 0
            while pid_m < num_m_tiles:
                a_slot, a_phase = get_bufidx_phase(tile_count, A_SLOTS)
                rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                mean = tl.load(mean_ptr + rows).to(tl.float32)
                rstd = tl.load(rstd_ptr + rows).to(tl.float32)
                columns = tl.arange(0, BLOCK_K)
                for k in range(k_tiles):
                    a_buf = a_slot * k_tiles + k
                    tlx.barrier_wait(a_full[a_buf], a_phase)
                    x = tlx.local_load(a_buffers[a_buf]).to(tl.float32)
                    gamma = tl.load(gamma_ptr + k * BLOCK_K + columns).to(tl.float32)
                    beta = tl.load(beta_ptr + k * BLOCK_K + columns).to(tl.float32)
                    hidden = (x - mean[:, None]) * rstd[:, None]
                    hidden = hidden * gamma[None, :] + beta[None, :]
                    tlx.local_store(a_buffers[a_buf], hidden.to(tl.bfloat16))
                tlx.fence("async_shared")
                tlx.barrier_arrive(a_normalized[a_slot], 1)
                tile_count += 1
                pid_m += PROGRAMS_PER_N

        with tlx.async_task(num_warps=1, num_regs=40):
            for k in tl.static_range(k_tiles):
                tlx.barrier_expect_bytes(b_full[k], 2 * BLOCK_K * BLOCK_N)
                tlx.async_descriptor_load(
                    weight_desc,
                    b_buffers[k],
                    [k * BLOCK_K, pid_n * BLOCK_N],
                    b_full[k],
                    eviction_policy="evict_last",
                )
            pid_m = first_pid_m
            tile_count = 0
            while pid_m < num_m_tiles:
                a_slot, a_phase = get_bufidx_phase(tile_count, A_SLOTS)
                tlx.barrier_wait(a_empty[a_slot], a_phase ^ 1)
                for k in tl.static_range(k_tiles):
                    a_buf = a_slot * k_tiles + k
                    tlx.barrier_expect_bytes(a_full[a_buf], 2 * BLOCK_M * BLOCK_K)
                    tlx.async_descriptor_load(
                        x_desc,
                        a_buffers[a_buf],
                        [pid_m * BLOCK_M, k * BLOCK_K],
                        a_full[a_buf],
                    )
                tile_count += 1
                pid_m += PROGRAMS_PER_N


@triton.jit
def saved_layernorm_dual_persistent_tlx(
    x_ptr,
    projection_weight_desc,
    gradient_desc,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    bias_ptr,
    projection_ptr,
    workspace_ptr,
    M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    PROJECTION_N: tl.constexpr,
    DWEIGHT_N: tl.constexpr,
    FEATURES: tl.constexpr,
    H_SLOTS: tl.constexpr,
    B_SLOTS: tl.constexpr,
    PRODUCER_WARPS: tl.constexpr,
    PRODUCER_REGS: tl.constexpr,
    DO_PROJECTION: tl.constexpr,
):
    projection_block_n: tl.constexpr = PROJECTION_N // 6
    dweight_block_n: tl.constexpr = DWEIGHT_N // 4
    projection_k_tiles: tl.constexpr = FEATURES // BLOCK_ROWS
    h_buffers_0 = tlx.local_alloc((BLOCK_ROWS, FEATURES // 2), tl.bfloat16, H_SLOTS)
    h_buffers_1 = tlx.local_alloc((BLOCK_ROWS, FEATURES // 2), tl.bfloat16, H_SLOTS)
    projection_b = tlx.local_alloc(
        (BLOCK_ROWS, projection_block_n), tl.bfloat16, B_SLOTS
    )
    gradient_b = tlx.local_alloc((BLOCK_ROWS, dweight_block_n), tl.bfloat16, H_SLOTS)
    projection_tmem = tlx.local_alloc(
        (BLOCK_ROWS, projection_block_n),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
    )
    dweight_tmem = tlx.local_alloc(
        (FEATURES // 2, dweight_block_n),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
    )
    h_full = tlx.alloc_barriers(H_SLOTS, arrive_count=1)
    h_empty = tlx.alloc_barriers(H_SLOTS, arrive_count=1)
    projection_b_full = tlx.alloc_barriers(B_SLOTS, arrive_count=1)
    projection_b_empty = tlx.alloc_barriers(B_SLOTS, arrive_count=1)
    gradient_full = tlx.alloc_barriers(H_SLOTS, arrive_count=1)
    gradient_empty = tlx.alloc_barriers(H_SLOTS, arrive_count=1)
    projection_full = tlx.alloc_barriers(1, arrive_count=1)
    projection_empty = tlx.alloc_barriers(1, arrive_count=1)
    dweight_full = tlx.alloc_barriers(1, arrive_count=1)

    cta_rank = tl.program_id(0) % 8
    split_id = tl.program_id(0) // 8
    rows_per_split: tl.constexpr = tl.cdiv(M, SPLIT_K)
    rows_per_split = tl.cdiv(rows_per_split, BLOCK_ROWS) * BLOCK_ROWS
    row_start = split_id * rows_per_split
    row_end = tl.minimum(row_start + rows_per_split, M)

    with tlx.async_tasks(no_ending_cluster_sync=True):
        with tlx.async_task("default", num_regs=160):
            row = row_start
            tile_count = 0
            while row < row_end:
                if DO_PROJECTION and cta_rank < 6:
                    tlx.barrier_wait(projection_full[0], tile_count & 1)
                    projection = tlx.local_load(projection_tmem[0])
                    tlx.barrier_arrive(projection_empty[0], 1)
                    rows = row + tl.arange(0, BLOCK_ROWS)
                    columns = cta_rank * projection_block_n + tl.arange(
                        0, projection_block_n
                    )
                    bias = tl.load(bias_ptr + columns).to(tl.float32)
                    offsets = rows[:, None] * PROJECTION_N + columns[None, :]
                    tl.store(
                        projection_ptr + offsets,
                        (projection + bias[None, :]).to(tl.bfloat16),
                    )
                row += BLOCK_ROWS
                tile_count += 1

            tlx.barrier_wait(dweight_full[0], 0)
            for n_start in tl.static_range(0, dweight_block_n, 32):
                tile = tlx.local_load(
                    tlx.local_slice(
                        dweight_tmem[0],
                        [0, n_start],
                        [FEATURES // 2, 32],
                    )
                )
                rows = (cta_rank // 4) * (FEATURES // 2) + tl.arange(0, FEATURES // 2)
                columns = (cta_rank % 4) * dweight_block_n + n_start + tl.arange(0, 32)
                offsets = (split_id * FEATURES + rows[:, None]) * DWEIGHT_N + columns[
                    None, :
                ]
                tl.store(workspace_ptr + offsets, tile)

        with tlx.async_task(num_warps=1, num_regs=24):
            row = row_start
            tile_count = 0
            projection_b_count = 0
            while row < row_end:
                h_buf, h_phase = get_bufidx_phase(tile_count, H_SLOTS)
                tlx.barrier_wait(h_full[h_buf], h_phase)
                tlx.barrier_wait(gradient_full[h_buf], h_phase)
                if DO_PROJECTION and cta_rank < 6:
                    tlx.barrier_wait(projection_empty[0], (tile_count & 1) ^ 1)
                    for k in tl.static_range(projection_k_tiles):
                        b_buf, b_phase = get_bufidx_phase(projection_b_count, B_SLOTS)
                        tlx.barrier_wait(projection_b_full[b_buf], b_phase)
                        if k < 2:
                            h_slice = tlx.local_slice(
                                h_buffers_0[h_buf],
                                [0, k * BLOCK_ROWS],
                                [BLOCK_ROWS, BLOCK_ROWS],
                            )
                        else:
                            h_slice = tlx.local_slice(
                                h_buffers_1[h_buf],
                                [0, (k - 2) * BLOCK_ROWS],
                                [BLOCK_ROWS, BLOCK_ROWS],
                            )
                        tlx.async_dot(
                            h_slice,
                            projection_b[b_buf],
                            projection_tmem[0],
                            use_acc=k > 0,
                            mBarriers=[projection_b_empty[b_buf]],
                            out_dtype=tl.float32,
                        )
                        projection_b_count += 1
                    tlx.tcgen05_commit(projection_full[0])
                if cta_rank < 4:
                    h_group = h_buffers_0[h_buf]
                else:
                    h_group = h_buffers_1[h_buf]
                tlx.async_dot(
                    tlx.local_trans(h_group),
                    gradient_b[h_buf],
                    dweight_tmem[0],
                    use_acc=tile_count > 0,
                    mBarriers=[h_empty[h_buf], gradient_empty[h_buf]],
                    out_dtype=tl.float32,
                )
                row += BLOCK_ROWS
                tile_count += 1
            tlx.tcgen05_commit(dweight_full[0])

        with tlx.async_task(num_warps=PRODUCER_WARPS, num_regs=PRODUCER_REGS):
            row = row_start
            tile_count = 0
            projection_b_count = 0
            features_0 = tl.arange(0, FEATURES // 2)
            features_1 = FEATURES // 2 + tl.arange(0, FEATURES // 2)
            gamma_0 = tl.load(gamma_ptr + features_0).to(tl.float32)
            gamma_1 = tl.load(gamma_ptr + features_1).to(tl.float32)
            beta_0 = tl.load(beta_ptr + features_0).to(tl.float32)
            beta_1 = tl.load(beta_ptr + features_1).to(tl.float32)
            while row < row_end:
                h_buf, h_phase = get_bufidx_phase(tile_count, H_SLOTS)
                tlx.barrier_wait(h_empty[h_buf], h_phase ^ 1)
                tlx.barrier_wait(gradient_empty[h_buf], h_phase ^ 1)
                tlx.barrier_expect_bytes(
                    gradient_full[h_buf],
                    2 * BLOCK_ROWS * dweight_block_n,
                )
                tlx.async_descriptor_load(
                    gradient_desc,
                    gradient_b[h_buf],
                    [row, (cta_rank % 4) * dweight_block_n],
                    gradient_full[h_buf],
                )

                if DO_PROJECTION and cta_rank < 6:
                    for k in tl.static_range(2):
                        b_buf, b_phase = get_bufidx_phase(projection_b_count, B_SLOTS)
                        tlx.barrier_wait(projection_b_empty[b_buf], b_phase ^ 1)
                        tlx.barrier_expect_bytes(
                            projection_b_full[b_buf],
                            2 * BLOCK_ROWS * projection_block_n,
                        )
                        tlx.async_descriptor_load(
                            projection_weight_desc,
                            projection_b[b_buf],
                            [k * BLOCK_ROWS, cta_rank * projection_block_n],
                            projection_b_full[b_buf],
                            eviction_policy="evict_last",
                        )
                        projection_b_count += 1

                rows = row + tl.arange(0, BLOCK_ROWS)
                mean = tl.load(mean_ptr + rows).to(tl.float32)
                rstd = tl.load(rstd_ptr + rows).to(tl.float32)
                offsets_0 = rows[:, None] * FEATURES + features_0[None, :]
                offsets_1 = rows[:, None] * FEATURES + features_1[None, :]
                x_0 = tl.load(x_ptr + offsets_0).to(tl.float32)
                x_1 = tl.load(x_ptr + offsets_1).to(tl.float32)
                hidden_0 = (x_0 - mean[:, None]) * rstd[:, None]
                hidden_1 = (x_1 - mean[:, None]) * rstd[:, None]
                hidden_0 = hidden_0 * gamma_0[None, :] + beta_0[None, :]
                hidden_1 = hidden_1 * gamma_1[None, :] + beta_1[None, :]
                tlx.local_store(h_buffers_0[h_buf], hidden_0.to(tl.bfloat16))
                tlx.local_store(h_buffers_1[h_buf], hidden_1.to(tl.bfloat16))
                tlx.fence("async_shared")
                tlx.barrier_arrive(h_full[h_buf], 1)

                if DO_PROJECTION and cta_rank < 6:
                    for k in tl.static_range(2, projection_k_tiles):
                        b_buf, b_phase = get_bufidx_phase(projection_b_count, B_SLOTS)
                        tlx.barrier_wait(projection_b_empty[b_buf], b_phase ^ 1)
                        tlx.barrier_expect_bytes(
                            projection_b_full[b_buf],
                            2 * BLOCK_ROWS * projection_block_n,
                        )
                        tlx.async_descriptor_load(
                            projection_weight_desc,
                            projection_b[b_buf],
                            [k * BLOCK_ROWS, cta_rank * projection_block_n],
                            projection_b_full[b_buf],
                            eviction_policy="evict_last",
                        )
                        projection_b_count += 1
                row += BLOCK_ROWS
                tile_count += 1


PROJECTION_CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 256,
    "BLOCK_K": 128,
    "A_SLOTS": 2,
    "NUM_B_BUFFERS": 2,
    "NUM_TMEM_BUFFERS": 2,
    "NORM_WARPS": 8,
    "NORM_REGS": 96,
}

DWEIGHT_CONFIG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "NUM_SMEM_BUFFERS": 3,
    "NUM_TMEM_BUFFERS": 1,
    "NUM_MMA_GROUPS": 2,
    "EPILOGUE_SUBTILE": 32,
    "NUM_CTAS": 1,
    "SPLIT_K": 38,
    "INTERLEAVE_EPILOGUE": 1,
}

DUAL_PERSISTENT_CONFIG = {
    "SPLIT_K": 19,
    "BLOCK_ROWS": 64,
    "H_SLOTS": 2,
    "B_SLOTS": 2,
    "PRODUCER_WARPS": 8,
    "PRODUCER_REGS": 96,
    "DO_PROJECTION": True,
}


def run_projection_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    *,
    config: dict[str, int] | None = None,
    waves: int = 1,
    epilogue_warps: int = 4,
) -> None:
    config = PROJECTION_CONFIG if config is None else config
    rows = inputs["x"].shape[0]
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    block_k = config["BLOCK_K"]
    x_desc = TensorDescriptor(
        inputs["x"], inputs["x"].shape, inputs["x"].stride(), [block_m, block_k]
    )
    weight_desc = TensorDescriptor(
        inputs["projection_weight"],
        inputs["projection_weight"].shape,
        inputs["projection_weight"].stride(),
        [block_k, block_n],
    )
    num_sms = torch.cuda.get_device_properties(inputs["x"].device).multi_processor_count
    num_programs = min(triton.cdiv(rows, block_m), num_sms * waves)
    cast(Any, saved_layernorm_projection_tlx)[(num_programs,)](
        x_desc,
        weight_desc,
        inputs["gamma"],
        inputs["beta"],
        inputs["mean"],
        inputs["rstd"],
        inputs["projection_bias"],
        outputs["projection"],
        M=rows,
        N=baseline.PROJECTION,
        K=baseline.FEATURES,
        NUM_PROGRAMS=num_programs,
        **config,
        num_warps=epilogue_warps,
        num_stages=1,
    )


def run_projection_weight_stationary_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    *,
    config: dict[str, int] | None = None,
    epilogue_warps: int = 4,
) -> None:
    config = PROJECTION_CONFIG if config is None else config
    rows = inputs["x"].shape[0]
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    block_k = config["BLOCK_K"]
    x_desc = TensorDescriptor(
        inputs["x"], inputs["x"].shape, inputs["x"].stride(), [block_m, block_k]
    )
    weight_desc = TensorDescriptor(
        inputs["projection_weight"],
        inputs["projection_weight"].shape,
        inputs["projection_weight"].stride(),
        [block_k, block_n],
    )
    num_sms = torch.cuda.get_device_properties(inputs["x"].device).multi_processor_count
    n_tiles = baseline.PROJECTION // block_n
    programs_per_n = num_sms // n_tiles
    cast(Any, saved_layernorm_projection_weight_stationary_tlx)[
        (programs_per_n * n_tiles,)
    ](
        x_desc,
        weight_desc,
        inputs["gamma"],
        inputs["beta"],
        inputs["mean"],
        inputs["rstd"],
        inputs["projection_bias"],
        outputs["projection"],
        M=rows,
        N=baseline.PROJECTION,
        K=baseline.FEATURES,
        PROGRAMS_PER_N=programs_per_n,
        **config,
        num_warps=epilogue_warps,
        num_stages=1,
    )


def make_dweight_workspace(
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    config = DWEIGHT_CONFIG if config is None else config
    rows_per_split = sm100._workspace_rows_per_split(
        baseline.FEATURES, config["BLOCK_SIZE_M"], config["NUM_CTAS"]
    )
    return torch.empty(
        (config["SPLIT_K"] * rows_per_split, baseline.GRADIENT),
        device="cuda",
        dtype=torch.float32,
    )


def run_dweight_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    workspace: torch.Tensor,
    *,
    config: dict[str, int] | None = None,
    producer_warps: int = 16,
    producer_regs: int = 128,
    prologue_k: int = 64,
) -> None:
    config = dict(DWEIGHT_CONFIG if config is None else config)
    rows = inputs["x"].shape[0]
    block_m_split = config["BLOCK_SIZE_M"] // config["NUM_MMA_GROUPS"]
    gradient_desc = TensorDescriptor(
        inputs["gradient"],
        inputs["gradient"].shape,
        inputs["gradient"].stride(),
        [config["BLOCK_SIZE_K"], config["BLOCK_SIZE_N"] // config["NUM_CTAS"]],
    )
    output_block = [
        block_m_split,
        config["BLOCK_SIZE_N"] // config["EPILOGUE_SUBTILE"],
    ]
    output_desc = TensorDescriptor(
        outputs["dweight"],
        outputs["dweight"].shape,
        outputs["dweight"].stride(),
        output_block,
    )
    workspace_desc = TensorDescriptor(
        workspace, workspace.shape, workspace.stride(), output_block
    )
    num_pid_m = sm100._padded_num_pid_m(
        baseline.FEATURES, config["BLOCK_SIZE_M"], config["NUM_CTAS"]
    )
    num_pid_n = triton.cdiv(baseline.GRADIENT, config["BLOCK_SIZE_N"])
    grid = (num_pid_m * num_pid_n * config["SPLIT_K"],)
    launch_options = {"ctas_per_cga": (2, 1, 1)} if config["NUM_CTAS"] == 2 else {}
    cast(Any, splitk_base.layernorm_mul_dweight_tlx)[grid](
        inputs["x"],
        inputs["x"],
        inputs["gamma"],
        inputs["beta"],
        inputs["mean"],
        inputs["rstd"],
        gradient_desc,
        output_desc,
        workspace_desc,
        baseline.FEATURES,
        baseline.GRADIENT,
        rows,
        NUM_SMS=torch.cuda.get_device_properties(
            inputs["x"].device
        ).multi_processor_count,
        FP16_WORKSPACE=False,
        D=baseline.FEATURES,
        PROLOGUE_K=prologue_k,
        PRODUCER_WARPS=producer_warps,
        PRODUCER_REGS=producer_regs,
        SAVED_LAYERNORM_ONLY=True,
        **config,
        **launch_options,
    )
    sm100._reduce_k_kernel[
        (
            triton.cdiv(baseline.FEATURES, 32),
            triton.cdiv(baseline.GRADIENT, 32),
        )
    ](
        workspace,
        outputs["dweight"],
        baseline.FEATURES,
        baseline.GRADIENT,
        sm100._workspace_rows_per_split(
            baseline.FEATURES, config["BLOCK_SIZE_M"], config["NUM_CTAS"]
        ),
        SPLIT_K=config["SPLIT_K"],
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        OUTPUT_DTYPE=tl.bfloat16,
    )


def run_dual_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    workspace: torch.Tensor,
) -> None:
    run_projection_tlx(inputs, outputs)
    run_dweight_tlx(inputs, outputs, workspace)


def run_dual_persistent_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    workspace: torch.Tensor,
    *,
    config: dict[str, int] | None = None,
) -> None:
    config = DUAL_PERSISTENT_CONFIG if config is None else config
    block_rows = config["BLOCK_ROWS"]
    projection_block_n = baseline.PROJECTION // 6
    dweight_block_n = baseline.GRADIENT // 4
    projection_weight_desc = TensorDescriptor(
        inputs["projection_weight"],
        inputs["projection_weight"].shape,
        inputs["projection_weight"].stride(),
        [block_rows, projection_block_n],
    )
    gradient_desc = TensorDescriptor(
        inputs["gradient"],
        inputs["gradient"].shape,
        inputs["gradient"].stride(),
        [block_rows, dweight_block_n],
    )
    cast(Any, saved_layernorm_dual_persistent_tlx)[(config["SPLIT_K"] * 8,)](
        inputs["x"],
        projection_weight_desc,
        gradient_desc,
        inputs["gamma"],
        inputs["beta"],
        inputs["mean"],
        inputs["rstd"],
        inputs["projection_bias"],
        outputs["projection"],
        workspace,
        M=inputs["x"].shape[0],
        PROJECTION_N=baseline.PROJECTION,
        DWEIGHT_N=baseline.GRADIENT,
        FEATURES=baseline.FEATURES,
        **config,
        num_warps=4,
        num_stages=1,
    )
    cast(Any, baseline.finish_dweight)[
        (triton.cdiv(baseline.FEATURES * baseline.GRADIENT, 256),)
    ](
        workspace,
        outputs["dweight"],
        D=baseline.FEATURES,
        N=baseline.GRADIENT,
        SPLIT=config["SPLIT_K"],
        BLOCK=256,
        num_warps=8,
    )


def run_dual_seed(inputs: baseline.TensorMap, outputs: baseline.TensorMap) -> None:
    baseline.run_projection_seed(inputs, outputs)
    baseline.run_dweight_seed(inputs, outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=baseline.DEFAULT_ROWS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--verification-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.rows % PROJECTION_CONFIG["BLOCK_M"]:
        parser.error("--rows must be divisible by BLOCK_M")
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")

    accuracy_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
    inputs = baseline.make_inputs(args.rows, 0)
    reference = baseline.make_outputs(args.rows, with_hidden=True)
    seed_output = baseline.make_outputs(args.rows, with_hidden=False)
    candidate = baseline.make_outputs(args.rows, with_hidden=False)
    workspace = make_dweight_workspace()
    baseline.run_projection_seed(inputs, reference)
    baseline.run_dweight_unfused(inputs, reference)
    run_dual_seed(inputs, seed_output)
    run_dual_tlx(inputs, candidate, workspace)
    torch.cuda.synchronize()
    accuracy_by_seed["0"] = baseline.accuracy(
        reference, candidate, ("projection", "dweight")
    )
    for seed in dict.fromkeys(args.verification_seeds):
        if seed == 0:
            continue
        seed_inputs = baseline.make_inputs(args.rows, seed)
        seed_reference = baseline.make_outputs(args.rows, with_hidden=True)
        seed_candidate = baseline.make_outputs(args.rows, with_hidden=False)
        baseline.run_projection_seed(seed_inputs, seed_reference)
        baseline.run_dweight_unfused(seed_inputs, seed_reference)
        seed_workspace = make_dweight_workspace()
        run_dual_tlx(seed_inputs, seed_candidate, seed_workspace)
        torch.cuda.synchronize()
        accuracy_by_seed[str(seed)] = baseline.accuracy(
            seed_reference, seed_candidate, ("projection", "dweight")
        )
    passed = all(
        metric["passed"]
        for seed_result in accuracy_by_seed.values()
        for metric in seed_result.values()
    )
    report: dict[str, Any] = {
        "projection_shape_mnk": [
            args.rows,
            baseline.PROJECTION,
            baseline.FEATURES,
        ],
        "dweight_shape_mnk": [
            baseline.FEATURES,
            baseline.GRADIENT,
            args.rows,
        ],
        "projection_config": PROJECTION_CONFIG,
        "dweight_config": DWEIGHT_CONFIG,
        "accuracy": accuracy_by_seed,
        "passed": passed,
    }
    if not args.check_only:
        unfused = baseline.make_outputs(args.rows, with_hidden=True)
        report["timings"] = {
            "unfused_projection_ms": baseline.benchmark(
                lambda: baseline.run_projection_unfused(inputs, unfused),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused_seed_projection_ms": baseline.benchmark(
                lambda: baseline.run_projection_seed(inputs, seed_output),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused_tlx_projection_ms": baseline.benchmark(
                lambda: run_projection_tlx(inputs, candidate),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "unfused_dweight_ms": baseline.benchmark(
                lambda: baseline.run_dweight_unfused(inputs, unfused),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused_seed_dweight_ms": baseline.benchmark(
                lambda: baseline.run_dweight_seed(inputs, seed_output),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused_tlx_dweight_ms": baseline.benchmark(
                lambda: run_dweight_tlx(inputs, candidate, workspace),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "unfused_dual_ms": baseline.benchmark(
                lambda: baseline.run_dual_unfused(inputs, unfused),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused_seed_dual_ms": baseline.benchmark(
                lambda: run_dual_seed(inputs, seed_output),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused_tlx_dual_ms": baseline.benchmark(
                lambda: run_dual_tlx(inputs, candidate, workspace),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
        }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
