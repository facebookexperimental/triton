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

"""OSS reproduction of the T289840347 GEMM-RMSNorm/SiLU backward region."""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, TypedDict, cast

import torch
import triton
import triton.language as tl


M = 5120
K = 2048
N = 512
ROWS_PER_PARTITION = 128
PARTITIONS = triton.cdiv(M, ROWS_PER_PARTITION)


class TensorMap(TypedDict):
    a: torch.Tensor
    b: torch.Tensor
    saved_input: torch.Tensor
    rstd: torch.Tensor
    weight: torch.Tensor
    projection: torch.Tensor
    dx: torch.Tensor
    dweight_partial: torch.Tensor
    dweight: torch.Tensor


@triton.jit
def rmsnorm_silu_backward_dx(
    saved_input,
    rstd_input,
    upstream,
    weight,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
):
    block_m: tl.constexpr = 16
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    columns = tl.arange(0, N)
    mask = rows[:, None] < M
    offsets = rows[:, None] * N + columns[None, :]
    saved = tl.load(saved_input + offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_input + rows, mask=rows < M, other=0.0)
    gradient = tl.load(upstream + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(weight + columns).to(tl.float32)
    normalized = saved * rstd[:, None]
    weighted = normalized * scale[None, :]
    sigmoid = tl.sigmoid(weighted)
    silu_gradient = gradient * sigmoid * (weighted * (1.0 - sigmoid) + 1.0)
    scaled_gradient = silu_gradient * scale[None, :]
    row_dot = tl.sum(normalized * scaled_gradient, axis=1)
    result = (scaled_gradient - normalized * (row_dot[:, None] / N)) * rstd[:, None]
    tl.store(output + offsets, result.to(tl.bfloat16), mask=mask)


@triton.jit
def rmsnorm_silu_backward_dweight_partial(
    saved_input,
    rstd_input,
    upstream,
    weight,
    partial_output,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS_PER_PARTITION: tl.constexpr,
):
    program = tl.program_id(0)
    column = program % N
    partition = program // N
    rows = partition * ROWS_PER_PARTITION + tl.arange(0, ROWS_PER_PARTITION)
    mask = rows < M
    offsets = rows * N + column
    saved = tl.load(saved_input + offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_input + rows, mask=mask, other=0.0)
    gradient = tl.load(upstream + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(weight + column).to(tl.float32)
    normalized = saved * rstd
    weighted = normalized * scale
    sigmoid = tl.sigmoid(weighted)
    contribution = gradient * sigmoid * (weighted * (1.0 - sigmoid) + 1.0)
    tl.store(partial_output + column + partition * N, tl.sum(contribution * normalized))


@triton.jit
def rmsnorm_silu_backward_dweight_finish(
    partial_input,
    output,
    N: tl.constexpr,
    PARTITIONS: tl.constexpr,
):
    block_n: tl.constexpr = 32
    columns = tl.program_id(0) * block_n + tl.arange(0, block_n)
    partitions = tl.arange(0, 128)
    mask = (columns[:, None] < N) & (partitions[None, :] < PARTITIONS)
    partial = tl.load(
        partial_input + columns[:, None] + partitions[None, :] * N,
        mask=mask,
        other=0.0,
    )
    tl.store(output + columns, tl.sum(partial, axis=1).to(tl.bfloat16))


@triton.jit
def fused_seed(
    a,
    b,
    saved_input,
    rstd_input,
    weight,
    gemm_output,
    dx_output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_tiles: tl.constexpr = triton.cdiv(M, BLOCK_M)
    for pid_m in range(pid, num_m_tiles, NUM_PROGRAMS):
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = tl.arange(0, N)
        accumulator = tl.zeros((BLOCK_M, N), tl.float32)
        for start_k in range(0, K, BLOCK_K):
            ks = start_k + tl.arange(0, BLOCK_K)
            a_tile = tl.load(a + rows[:, None] * K + ks[None, :])
            b_tile = tl.load(b + columns[None, :] * K + ks[:, None])
            accumulator = tl.dot(a_tile, b_tile, accumulator, allow_tf32=False)
        offsets = rows[:, None] * N + columns[None, :]
        gemm_value = accumulator.to(tl.bfloat16)
        tl.store(gemm_output + offsets, gemm_value)
        saved = tl.load(saved_input + offsets).to(tl.float32)
        rstd = tl.load(rstd_input + rows)
        scale = tl.load(weight + columns).to(tl.float32)
        normalized = saved * rstd[:, None]
        weighted = normalized * scale[None, :]
        sigmoid = tl.sigmoid(weighted)
        silu_gradient = (
            gemm_value.to(tl.float32) * sigmoid * (weighted * (1.0 - sigmoid) + 1.0)
        )
        scaled_gradient = silu_gradient * scale[None, :]
        row_dot = tl.sum(normalized * scaled_gradient, axis=1)
        dx = (scaled_gradient - normalized * (row_dot[:, None] / N)) * rstd[:, None]
        tl.store(dx_output + offsets, dx.to(tl.bfloat16))


def make_inputs(seed: int = 0) -> TensorMap:
    torch.manual_seed(seed)
    return cast(
        TensorMap,
        {
            "a": torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
            / math.sqrt(K),
            "b": torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
            / math.sqrt(K),
            "saved_input": torch.randn((M, N), device="cuda", dtype=torch.bfloat16)
            / math.sqrt(N),
            "rstd": torch.rand((M,), device="cuda", dtype=torch.float32) + 0.5,
            "weight": torch.randn((N,), device="cuda", dtype=torch.bfloat16),
        },
    )


def make_outputs(*, partial_rows: int = triton.cdiv(M, 64)) -> TensorMap:
    return cast(
        TensorMap,
        {
            "projection": torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
            "dx": torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
            "dweight_partial": torch.empty(
                (partial_rows, N), device="cuda", dtype=torch.float32
            ),
            "dweight": torch.empty((N,), device="cuda", dtype=torch.bfloat16),
        },
    )


def finish_dweight(inputs: TensorMap, outputs: TensorMap) -> None:
    cast(Any, rmsnorm_silu_backward_dweight_partial)[(PARTITIONS * N,)](
        inputs["saved_input"],
        inputs["rstd"],
        outputs["projection"],
        inputs["weight"],
        outputs["dweight_partial"],
        M=M,
        N=N,
        ROWS_PER_PARTITION=ROWS_PER_PARTITION,
        num_warps=4,
    )
    cast(Any, rmsnorm_silu_backward_dweight_finish)[(triton.cdiv(N, 32),)](
        outputs["dweight_partial"],
        outputs["dweight"],
        N=N,
        PARTITIONS=PARTITIONS,
        num_warps=4,
    )


def run_unfused(inputs: TensorMap, outputs: TensorMap) -> None:
    torch.mm(inputs["a"], inputs["b"].T, out=outputs["projection"])
    cast(Any, rmsnorm_silu_backward_dx)[(triton.cdiv(M, 16),)](
        inputs["saved_input"],
        inputs["rstd"],
        outputs["projection"],
        inputs["weight"],
        outputs["dx"],
        M=M,
        N=N,
        num_warps=8,
    )
    finish_dweight(inputs, outputs)


def run_seed(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_m: int = 64,
    block_k: int = 64,
    num_warps: int = 8,
    num_stages: int = 3,
) -> None:
    num_programs = min(
        torch.cuda.get_device_properties(inputs["a"].device).multi_processor_count,
        triton.cdiv(M, block_m),
    )
    cast(Any, fused_seed)[(num_programs,)](
        inputs["a"],
        inputs["b"],
        inputs["saved_input"],
        inputs["rstd"],
        inputs["weight"],
        outputs["projection"],
        outputs["dx"],
        M=M,
        N=N,
        K=K,
        NUM_PROGRAMS=num_programs,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    finish_dweight(inputs, outputs)


def accuracy(reference: TensorMap, candidate: TensorMap) -> dict[str, float]:
    return {
        name: float((reference[name].float() - candidate[name].float()).abs().max())
        for name in ("projection", "dx", "dweight")
    }


def benchmark(
    fn: Callable[[], None], *, warmup: int = 5, samples: int = 20, reps: int = 3
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    medians = []
    for _ in range(reps):
        values = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        medians.append(statistics.median(values))
    return medians
