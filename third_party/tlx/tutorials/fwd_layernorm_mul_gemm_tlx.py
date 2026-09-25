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

"""Tuned TLX LayerNorm/mul GEMM-prologue prototype for T290058050."""

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
from triton.tools.tensor_descriptor import TensorDescriptor

import fwd_layernorm_mul_gemm as baseline


@triton.jit
def _producer_tile(
    tile_id,
    weight_desc,
    x_ptr,
    u_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    buffers_a,
    buffers_b,
    a_full,
    b_full,
    a_empty,
    b_empty,
    a_count,
    b_count,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_A_BUFFERS: tl.constexpr,
    NUM_B_BUFFERS: tl.constexpr,
):
    rows = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    mean = tl.load(mean_ptr + rows).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows).to(tl.float32)
    b_bytes: tl.constexpr = BLOCK_K * BLOCK_N * 2
    k_tiles: tl.constexpr = K // BLOCK_K
    for k in tl.static_range(k_tiles):
        a_buf, a_phase = get_bufidx_phase(a_count, NUM_A_BUFFERS)
        b_buf, b_phase = get_bufidx_phase(b_count, NUM_B_BUFFERS)
        tlx.barrier_wait(a_empty[a_buf], a_phase ^ 1)
        tlx.barrier_wait(b_empty[b_buf], b_phase ^ 1)

        tlx.barrier_expect_bytes(b_full[b_buf], b_bytes)
        tlx.async_descriptor_load(
            weight_desc,
            buffers_b[b_buf],
            [k * BLOCK_K, 0],
            b_full[b_buf],
            eviction_policy="evict_last",
        )

        feature_start = (k * BLOCK_K) % D
        features = feature_start + tl.arange(0, BLOCK_K)
        offsets = rows[:, None] * D + features[None, :]
        u = tl.load(u_ptr + offsets).to(tl.float32)
        if k * BLOCK_K < D:
            activation = u * tl.sigmoid(u)
        else:
            x = tl.load(x_ptr + offsets).to(tl.float32)
            gamma = tl.load(gamma_ptr + features).to(tl.float32)
            beta = tl.load(beta_ptr + features).to(tl.float32)
            normalized = (x - mean[:, None]) * rstd[:, None]
            activation = (normalized * gamma[None, :] + beta[None, :]) * u
        tlx.local_store(buffers_a[a_buf], activation.to(tl.bfloat16))
        tlx.barrier_arrive(a_full[a_buf], 1)
        a_count += 1
        b_count += 1
    return a_count, b_count


@triton.jit
def _mma_tile(
    buffers_a,
    buffers_b,
    tmem,
    a_full,
    b_full,
    a_empty,
    b_empty,
    tmem_full,
    tmem_empty,
    a_count,
    b_count,
    tmem_buf,
    tmem_phase,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_A_BUFFERS: tl.constexpr,
    NUM_B_BUFFERS: tl.constexpr,
):
    k_tiles: tl.constexpr = K // BLOCK_K
    a_buf, a_phase = get_bufidx_phase(a_count, NUM_A_BUFFERS)
    b_buf, b_phase = get_bufidx_phase(b_count, NUM_B_BUFFERS)
    tlx.barrier_wait(a_full[a_buf], a_phase)
    tlx.barrier_wait(b_full[b_buf], b_phase)
    tlx.barrier_wait(tmem_empty[tmem_buf], tmem_phase ^ 1)
    tlx.async_dot(
        buffers_a[a_buf],
        buffers_b[b_buf],
        tmem[tmem_buf],
        use_acc=False,
        mBarriers=[a_empty[a_buf], b_empty[b_buf]],
        out_dtype=tl.float32,
    )
    a_count += 1
    b_count += 1
    for _ in range(1, k_tiles):
        a_buf, a_phase = get_bufidx_phase(a_count, NUM_A_BUFFERS)
        b_buf, b_phase = get_bufidx_phase(b_count, NUM_B_BUFFERS)
        tlx.barrier_wait(a_full[a_buf], a_phase)
        tlx.barrier_wait(b_full[b_buf], b_phase)
        tlx.async_dot(
            buffers_a[a_buf],
            buffers_b[b_buf],
            tmem[tmem_buf],
            use_acc=True,
            mBarriers=[a_empty[a_buf], b_empty[b_buf]],
            out_dtype=tl.float32,
        )
        a_count += 1
        b_count += 1
    last_a_buf, last_a_phase = get_bufidx_phase(a_count - 1, NUM_A_BUFFERS)
    tlx.barrier_wait(a_empty[last_a_buf], last_a_phase)
    tlx.barrier_arrive(tmem_full[tmem_buf], 1)
    return a_count, b_count


@triton.jit
def layernorm_mul_gemm_tlx(
    weight_desc,
    x_ptr,
    u_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_A_BUFFERS: tl.constexpr,
    NUM_B_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    PRODUCER_WARPS: tl.constexpr,
    PRODUCER_REGS: tl.constexpr,
):
    buffers_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, NUM_A_BUFFERS)
    buffers_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.bfloat16, NUM_B_BUFFERS)
    tmem = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        NUM_TMEM_BUFFERS,
        tlx.storage_kind.tmem,
    )
    a_full = tlx.alloc_barriers(NUM_A_BUFFERS, arrive_count=1)
    b_full = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(NUM_A_BUFFERS, arrive_count=1)
    b_empty = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    tmem_full = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    num_tiles: tl.constexpr = M // BLOCK_M

    with tlx.async_tasks():
        with tlx.async_task("default"):
            tile_id = tl.program_id(0)
            tmem_count = 0
            while tile_id < num_tiles:
                tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(tmem_full[tmem_buf], tmem_phase)
                result = tlx.local_load(tmem[tmem_buf])
                tlx.barrier_arrive(tmem_empty[tmem_buf], 1)
                rows = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
                cols = tl.arange(0, BLOCK_N)
                offsets = rows[:, None] * N + cols[None, :]
                residual = tl.load(residual_ptr + offsets).to(tl.float32)
                tl.store(output_ptr + offsets, (result + residual).to(tl.bfloat16))
                tmem_count += 1
                tile_id += NUM_PROGRAMS

        with tlx.async_task(num_warps=1, num_regs=24):
            tile_id = tl.program_id(0)
            a_count = 0
            b_count = 0
            tmem_count = 0
            while tile_id < num_tiles:
                tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                a_count, b_count = _mma_tile(
                    buffers_a,
                    buffers_b,
                    tmem,
                    a_full,
                    b_full,
                    a_empty,
                    b_empty,
                    tmem_full,
                    tmem_empty,
                    a_count,
                    b_count,
                    tmem_buf,
                    tmem_phase,
                    K,
                    BLOCK_K,
                    NUM_A_BUFFERS,
                    NUM_B_BUFFERS,
                )
                tmem_count += 1
                tile_id += NUM_PROGRAMS

        with tlx.async_task(num_warps=PRODUCER_WARPS, num_regs=PRODUCER_REGS):
            tile_id = tl.program_id(0)
            a_count = 0
            b_count = 0
            while tile_id < num_tiles:
                a_count, b_count = _producer_tile(
                    tile_id,
                    weight_desc,
                    x_ptr,
                    u_ptr,
                    gamma_ptr,
                    beta_ptr,
                    mean_ptr,
                    rstd_ptr,
                    buffers_a,
                    buffers_b,
                    a_full,
                    b_full,
                    a_empty,
                    b_empty,
                    a_count,
                    b_count,
                    K,
                    D,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    NUM_A_BUFFERS,
                    NUM_B_BUFFERS,
                )
                tile_id += NUM_PROGRAMS


@triton.jit
def layernorm_mul_gemm_single_read_tlx(
    x_desc,
    u_desc,
    weight_desc,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    EPSILON_VALUE: tl.constexpr,
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
    feature_tiles: tl.constexpr = D // BLOCK_K
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
    num_tiles: tl.constexpr = M // BLOCK_M

    with tlx.async_tasks():
        with tlx.async_task("default"):
            tile_id = tl.program_id(0)
            tmem_count = 0
            while tile_id < num_tiles:
                tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(tmem_full[tmem_buf], tmem_phase)
                result = tlx.local_load(tmem[tmem_buf])
                tlx.barrier_arrive(tmem_empty[tmem_buf], 1)
                rows = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
                cols = tl.arange(0, BLOCK_N)
                offsets = rows[:, None] * N + cols[None, :]
                residual = tl.load(residual_ptr + offsets).to(tl.float32)
                tl.store(output_ptr + offsets, (result + residual).to(tl.bfloat16))
                tmem_count += 1
                tile_id += NUM_PROGRAMS

        with tlx.async_task(num_warps=1, num_regs=24):
            tile_id = tl.program_id(0)
            tile_count = 0
            b_count = 0
            tmem_count = 0
            while tile_id < num_tiles:
                a_slot, a_phase = get_bufidx_phase(tile_count, A_SLOTS)
                tlx.barrier_wait(a_normalized[a_slot], a_phase)
                tmem_buf, tmem_phase = get_bufidx_phase(tmem_count, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(tmem_empty[tmem_buf], tmem_phase ^ 1)
                for k in tl.static_range(k_tiles):
                    a_buf = a_slot * k_tiles + k
                    b_buf, b_phase = get_bufidx_phase(b_count, NUM_B_BUFFERS)
                    tlx.barrier_wait(b_full[b_buf], b_phase)
                    if k == k_tiles - 1:
                        tlx.async_dot(
                            a_buffers[a_buf],
                            b_buffers[b_buf],
                            tmem[tmem_buf],
                            use_acc=k > 0,
                            mBarriers=[
                                b_empty[b_buf],
                                a_empty[a_slot],
                                tmem_full[tmem_buf],
                            ],
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
                tile_count += 1
                tmem_count += 1
                tile_id += NUM_PROGRAMS

        with tlx.async_task(num_warps=NORM_WARPS, num_regs=NORM_REGS):
            tile_id = tl.program_id(0)
            tile_count = 0
            while tile_id < num_tiles:
                a_slot, a_phase = get_bufidx_phase(tile_count, A_SLOTS)
                rows = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
                sum_x = tl.zeros((BLOCK_M,), tl.float32)
                sum_x2 = tl.zeros((BLOCK_M,), tl.float32)
                for k in tl.static_range(k_tiles):
                    a_buf = a_slot * k_tiles + k
                    tlx.barrier_wait(a_full[a_buf], a_phase)
                    if k < feature_tiles:
                        x = tlx.local_load(a_buffers[a_buf]).to(tl.float32)
                        sum_x += tl.sum(x, axis=1)
                        sum_x2 += tl.sum(x * x, axis=1)
                mean = sum_x / D
                rstd = 1.0 / tl.sqrt(sum_x2 / D - mean * mean + EPSILON_VALUE)
                tl.store(mean_ptr + rows, mean)
                tl.store(rstd_ptr + rows, rstd)
                features = tl.arange(0, BLOCK_K)
                for k in tl.static_range(feature_tiles):
                    x_buf = a_slot * k_tiles + k
                    u_buf = a_slot * k_tiles + feature_tiles + k
                    x = tlx.local_load(a_buffers[x_buf]).to(tl.float32)
                    u = tlx.local_load(a_buffers[u_buf]).to(tl.float32)
                    gamma = tl.load(gamma_ptr + k * BLOCK_K + features).to(tl.float32)
                    beta = tl.load(beta_ptr + k * BLOCK_K + features).to(tl.float32)
                    normalized = (x - mean[:, None]) * rstd[:, None]
                    second = (normalized * gamma[None, :] + beta[None, :]) * u
                    tlx.local_store(
                        a_buffers[x_buf], (u * tl.sigmoid(u)).to(tl.bfloat16)
                    )
                    tlx.local_store(a_buffers[u_buf], second.to(tl.bfloat16))
                tlx.fence("async_shared")
                tlx.barrier_arrive(a_normalized[a_slot], 1)
                tile_count += 1
                tile_id += NUM_PROGRAMS

        with tlx.async_task(num_warps=1, num_regs=40):
            tile_id = tl.program_id(0)
            tile_count = 0
            b_count = 0
            while tile_id < num_tiles:
                a_slot, a_phase = get_bufidx_phase(tile_count, A_SLOTS)
                tlx.barrier_wait(a_empty[a_slot], a_phase ^ 1)
                for k in tl.static_range(feature_tiles):
                    x_buf = a_slot * k_tiles + k
                    u_buf = a_slot * k_tiles + feature_tiles + k
                    tlx.barrier_expect_bytes(a_full[x_buf], 2 * BLOCK_M * BLOCK_K)
                    tlx.async_descriptor_load(
                        x_desc,
                        a_buffers[x_buf],
                        [tile_id * BLOCK_M, k * BLOCK_K],
                        a_full[x_buf],
                    )
                    tlx.barrier_expect_bytes(a_full[u_buf], 2 * BLOCK_M * BLOCK_K)
                    tlx.async_descriptor_load(
                        u_desc,
                        a_buffers[u_buf],
                        [tile_id * BLOCK_M, k * BLOCK_K],
                        a_full[u_buf],
                    )
                for k in tl.static_range(k_tiles):
                    b_buf, b_phase = get_bufidx_phase(b_count, NUM_B_BUFFERS)
                    tlx.barrier_wait(b_empty[b_buf], b_phase ^ 1)
                    tlx.barrier_expect_bytes(b_full[b_buf], 2 * BLOCK_K * BLOCK_N)
                    tlx.async_descriptor_load(
                        weight_desc,
                        b_buffers[b_buf],
                        [k * BLOCK_K, 0],
                        b_full[b_buf],
                        eviction_policy="evict_last",
                    )
                    b_count += 1
                tile_count += 1
                tile_id += NUM_PROGRAMS


CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 256,
    "BLOCK_K": 128,
    "NUM_A_BUFFERS": 2,
    "NUM_B_BUFFERS": 2,
    "NUM_TMEM_BUFFERS": 1,
    "PRODUCER_WARPS": 8,
    "PRODUCER_REGS": 128,
}

SINGLE_READ_CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 256,
    "BLOCK_K": 128,
    "A_SLOTS": 2,
    "NUM_B_BUFFERS": 1,
    "NUM_TMEM_BUFFERS": 1,
    "NORM_WARPS": 8,
    "NORM_REGS": 96,
}


def run_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    *,
    config: dict[str, int] | None = None,
    stats_block_rows: int = 16,
    stats_warps: int = 4,
    waves: int = 1,
) -> None:
    config = CONFIG if config is None else config
    baseline.run_stats(
        inputs, outputs, block_rows=stats_block_rows, num_warps=stats_warps
    )
    run_projection_tlx(inputs, outputs, config=config, waves=waves)


def run_projection_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    *,
    config: dict[str, int] | None = None,
    waves: int = 1,
) -> None:
    config = CONFIG if config is None else config
    rows = inputs["x"].shape[0]
    weight_desc = TensorDescriptor(
        inputs["weight"],
        inputs["weight"].shape,
        inputs["weight"].stride(),
        [config["BLOCK_K"], config["BLOCK_N"]],
    )
    num_sms = torch.cuda.get_device_properties(inputs["x"].device).multi_processor_count
    num_tiles = triton.cdiv(rows, config["BLOCK_M"])
    num_programs = min(num_tiles, num_sms * waves)
    cast(Any, layernorm_mul_gemm_tlx)[(num_programs,)](
        weight_desc,
        inputs["x"],
        inputs["u"],
        inputs["gamma"],
        inputs["beta"],
        outputs["mean"],
        outputs["rstd"],
        inputs["residual"],
        outputs["projection"],
        M=rows,
        N=baseline.OUTPUT_FEATURES,
        K=baseline.GEMM_K,
        D=baseline.FEATURES,
        NUM_PROGRAMS=num_programs,
        **config,
        num_warps=4,
        num_stages=1,
    )


def run_single_read_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    *,
    config: dict[str, int] | None = None,
    waves: int = 1,
) -> None:
    config = SINGLE_READ_CONFIG if config is None else config
    rows = inputs["x"].shape[0]
    block_m = config["BLOCK_M"]
    block_k = config["BLOCK_K"]
    x_desc = TensorDescriptor(
        inputs["x"], inputs["x"].shape, inputs["x"].stride(), [block_m, block_k]
    )
    u_desc = TensorDescriptor(
        inputs["u"], inputs["u"].shape, inputs["u"].stride(), [block_m, block_k]
    )
    weight_desc = TensorDescriptor(
        inputs["weight"],
        inputs["weight"].shape,
        inputs["weight"].stride(),
        [block_k, config["BLOCK_N"]],
    )
    num_sms = torch.cuda.get_device_properties(inputs["x"].device).multi_processor_count
    num_tiles = triton.cdiv(rows, block_m)
    num_programs = min(num_tiles, num_sms * waves)
    cast(Any, layernorm_mul_gemm_single_read_tlx)[(num_programs,)](
        x_desc,
        u_desc,
        weight_desc,
        inputs["gamma"],
        inputs["beta"],
        outputs["mean"],
        outputs["rstd"],
        inputs["residual"],
        outputs["projection"],
        M=rows,
        N=baseline.OUTPUT_FEATURES,
        K=baseline.GEMM_K,
        D=baseline.FEATURES,
        EPSILON_VALUE=baseline.EPSILON,
        NUM_PROGRAMS=num_programs,
        **config,
        num_warps=4,
        num_stages=1,
    )


def run_seed(inputs: baseline.TensorMap, outputs: baseline.TensorMap) -> None:
    baseline.run_stats(inputs, outputs, block_rows=1, num_warps=8)
    baseline.run_seed(inputs, outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=baseline.DEFAULT_ROWS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--verification-seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--variant", choices=("streaming", "single-read"), default="streaming"
    )
    parser.add_argument("--block-m", type=int, default=CONFIG["BLOCK_M"])
    parser.add_argument("--block-k", type=int, default=CONFIG["BLOCK_K"])
    parser.add_argument("--a-buffers", type=int, default=CONFIG["NUM_A_BUFFERS"])
    parser.add_argument("--b-buffers", type=int, default=CONFIG["NUM_B_BUFFERS"])
    parser.add_argument("--tmem-buffers", type=int, default=CONFIG["NUM_TMEM_BUFFERS"])
    parser.add_argument("--producer-warps", type=int, default=CONFIG["PRODUCER_WARPS"])
    parser.add_argument("--producer-regs", type=int, default=CONFIG["PRODUCER_REGS"])
    parser.add_argument("--stats-block-rows", type=int, default=16)
    parser.add_argument("--stats-warps", type=int, default=4)
    parser.add_argument("--waves", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    config = {
        "BLOCK_M": args.block_m,
        "BLOCK_N": baseline.OUTPUT_FEATURES,
        "BLOCK_K": args.block_k,
        "NUM_A_BUFFERS": args.a_buffers,
        "NUM_B_BUFFERS": args.b_buffers,
        "NUM_TMEM_BUFFERS": args.tmem_buffers,
        "PRODUCER_WARPS": args.producer_warps,
        "PRODUCER_REGS": args.producer_regs,
    }
    if args.rows % config["BLOCK_M"]:
        parser.error(f"--rows must be divisible by {config['BLOCK_M']}")
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")

    def run_candidate(
        candidate_inputs: baseline.TensorMap, candidate_outputs: baseline.TensorMap
    ) -> None:
        if args.variant == "single-read":
            run_single_read_tlx(candidate_inputs, candidate_outputs, waves=args.waves)
        else:
            run_tlx(
                candidate_inputs,
                candidate_outputs,
                config=config,
                stats_block_rows=args.stats_block_rows,
                stats_warps=args.stats_warps,
                waves=args.waves,
            )

    inputs = baseline.make_inputs(args.rows, 0)
    reference = baseline.make_outputs(args.rows, with_intermediate=True)
    seed_output = baseline.make_outputs(args.rows, with_intermediate=False)
    candidate = baseline.make_outputs(args.rows, with_intermediate=False)
    baseline.run_reference(inputs, reference)
    run_seed(inputs, seed_output)
    run_candidate(inputs, candidate)
    torch.cuda.synchronize()
    reference_accuracy = {"0": baseline.accuracy(reference, candidate)}
    seed_accuracy = baseline.accuracy(reference, seed_output)
    tlx_vs_seed = {"0": baseline.accuracy(seed_output, candidate)}
    for seed in dict.fromkeys(args.verification_seeds):
        if seed == 0:
            continue
        seed_inputs = baseline.make_inputs(args.rows, seed)
        seed_reference = baseline.make_outputs(args.rows, with_intermediate=True)
        seed_fused = baseline.make_outputs(args.rows, with_intermediate=False)
        seed_candidate = baseline.make_outputs(args.rows, with_intermediate=False)
        baseline.run_reference(seed_inputs, seed_reference)
        run_seed(seed_inputs, seed_fused)
        run_candidate(seed_inputs, seed_candidate)
        torch.cuda.synchronize()
        reference_accuracy[str(seed)] = baseline.accuracy(
            seed_reference, seed_candidate
        )
        tlx_vs_seed[str(seed)] = baseline.accuracy(seed_fused, seed_candidate)

    passed = all(
        metric["passed"]
        for result_group in (reference_accuracy, tlx_vs_seed)
        for seed_result in result_group.values()
        for metric in seed_result.values()
    )
    report: dict[str, Any] = {
        "shape_mnk": [args.rows, baseline.OUTPUT_FEATURES, baseline.GEMM_K],
        "variant": args.variant,
        "config": SINGLE_READ_CONFIG if args.variant == "single-read" else config,
        "stats_block_rows": args.stats_block_rows,
        "stats_warps": args.stats_warps,
        "waves": args.waves,
        "reference_accuracy": reference_accuracy,
        "fused_seed_accuracy": seed_accuracy,
        "tlx_vs_seed": tlx_vs_seed,
        "passed": passed,
    }
    if not args.check_only:
        report["timings"] = {
            "unfused_ms": baseline.benchmark(
                lambda: baseline.run_unfused(inputs, reference),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused_seed_ms": baseline.benchmark(
                lambda: run_seed(inputs, seed_output),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused_tlx_ms": baseline.benchmark(
                lambda: run_candidate(inputs, candidate),
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
