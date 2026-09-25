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

"""OSS reproduction of the T289901422 SwiGLU-prologue dW GEMM."""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, TypedDict, cast

import torch
import triton
import triton.language as tl


M = 4096
N = 8192
K = 5120


class TensorMap(TypedDict):
    down_gradient: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    hidden: torch.Tensor
    dweight: torch.Tensor


@triton.jit
def reconstruct_hidden(
    gate,
    up,
    hidden,
    N_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N_ELEMENTS
    gate_value = tl.load(gate + offsets, mask=mask).to(tl.float32)
    up_value = tl.load(up + offsets, mask=mask).to(tl.float32)
    value = gate_value * tl.sigmoid(gate_value) * up_value
    tl.store(hidden + offsets, value.to(tl.bfloat16), mask=mask)


@triton.jit
def fused_seed(
    down_gradient,
    gate,
    up,
    dweight,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = N // BLOCK_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, K, BLOCK_K):
        ks = start_k + tl.arange(0, BLOCK_K)
        gradient = tl.load(down_gradient + ks[:, None] * M + rows[None, :]).to(
            tl.bfloat16
        )
        gate_value = tl.load(gate + ks[:, None] * N + columns[None, :]).to(tl.float32)
        up_value = tl.load(up + ks[:, None] * N + columns[None, :]).to(tl.float32)
        hidden = (gate_value * tl.sigmoid(gate_value) * up_value).to(tl.bfloat16)
        accumulator = tl.dot(
            gradient.T,
            hidden,
            accumulator,
            allow_tf32=False,
        )
    offsets = rows[:, None] * N + columns[None, :]
    tl.store(dweight + offsets, accumulator.to(tl.bfloat16))


def make_inputs(seed: int = 0) -> TensorMap:
    torch.manual_seed(seed)
    return cast(
        TensorMap,
        {
            "down_gradient": torch.randn((K, M), device="cuda", dtype=torch.bfloat16)
            / math.sqrt(K),
            "gate": torch.randn((K, N), device="cuda", dtype=torch.bfloat16),
            "up": torch.randn((K, N), device="cuda", dtype=torch.bfloat16),
        },
    )


def make_outputs() -> TensorMap:
    return cast(
        TensorMap,
        {
            "hidden": torch.empty((K, N), device="cuda", dtype=torch.bfloat16),
            "dweight": torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
        },
    )


def run_unfused(inputs: TensorMap, outputs: TensorMap) -> None:
    cast(Any, reconstruct_hidden)[(triton.cdiv(K * N, 1024),)](
        inputs["gate"],
        inputs["up"],
        outputs["hidden"],
        N_ELEMENTS=K * N,
        BLOCK=1024,
        num_warps=8,
    )
    torch.mm(inputs["down_gradient"].T, outputs["hidden"], out=outputs["dweight"])


def run_seed(inputs: TensorMap, outputs: TensorMap) -> None:
    block_m = 256
    block_n = 128
    cast(Any, fused_seed)[(triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)](
        inputs["down_gradient"],
        inputs["gate"],
        inputs["up"],
        outputs["dweight"],
        M=M,
        N=N,
        K=K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )


def accuracy(reference: TensorMap, candidate: TensorMap) -> dict[str, float | bool]:
    expected = reference["dweight"].float()
    difference = candidate["dweight"].float() - expected
    return {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected).clamp_min(1e-30)
        ),
        "exact": bool(torch.equal(reference["dweight"], candidate["dweight"])),
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
