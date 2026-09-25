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

"""OSS reproduction of the T289757910 GEMM/RMSNorm/SiLU region."""

from __future__ import annotations

import math
from typing import Any

import torch
import triton
import triton.language as tl


M = 5120
N = 512
K = 8192
EPS = 1e-5


@triton.jit
def rmsnorm_silu(
    source,
    weight,
    output,
    rstd_output,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
):
    block_m: tl.constexpr = 16
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    columns = tl.arange(0, N)
    mask = rows[:, None] < M
    offsets = rows[:, None] * N + columns[None, :]
    value = tl.load(source + offsets, mask=mask, other=0.0).to(tl.float32)
    square_sum = tl.sum(value * value, axis=1)
    rstd = tl.rsqrt(square_sum / N + EPS)
    scale = tl.load(weight + columns).to(tl.float32)
    normalized = value * rstd[:, None] * scale[None, :]
    activated = normalized * tl.sigmoid(normalized)
    tl.store(output + offsets, activated.to(tl.bfloat16), mask=mask)
    tl.store(rstd_output + rows, rstd, mask=rows < M)


@triton.jit
def fused_seed(
    a,
    b,
    weight,
    projection,
    output,
    rstd_output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program = tl.program_id(0)
    num_m_tiles: tl.constexpr = triton.cdiv(M, BLOCK_M)
    for pid_m in range(program, num_m_tiles, NUM_PROGRAMS):
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = tl.arange(0, N)
        accumulator = tl.zeros((BLOCK_M, N), tl.float32)
        for start_k in range(0, K, BLOCK_K):
            ks = start_k + tl.arange(0, BLOCK_K)
            a_tile = tl.load(a + rows[:, None] * K + ks[None, :])
            b_tile = tl.load(b + columns[None, :] * K + ks[:, None])
            accumulator = tl.dot(a_tile, b_tile, accumulator, allow_tf32=False)
        offsets = rows[:, None] * N + columns[None, :]
        value = accumulator.to(tl.bfloat16)
        tl.store(projection + offsets, value)
        value_fp32 = value.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(value_fp32 * value_fp32, axis=1) / N + EPS)
        scale = tl.load(weight + columns).to(tl.float32)
        normalized = value_fp32 * rstd[:, None] * scale[None, :]
        activated = normalized * tl.sigmoid(normalized)
        tl.store(output + offsets, activated.to(tl.bfloat16))
        tl.store(rstd_output + rows, rstd)


def make_inputs(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "a": torch.randn((M, K), device="cuda", dtype=torch.bfloat16) / math.sqrt(K),
        "b": torch.randn((N, K), device="cuda", dtype=torch.bfloat16) / math.sqrt(K),
        "weight": torch.randn((N,), device="cuda", dtype=torch.bfloat16) / math.sqrt(N),
    }


def make_outputs() -> dict[str, torch.Tensor]:
    return {
        "projection": torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
        "output": torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
        "rstd": torch.empty((M,), device="cuda", dtype=torch.float32),
    }


def run_unfused(
    inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
) -> None:
    torch.mm(inputs["a"], inputs["b"].T, out=outputs["projection"])
    rmsnorm_silu[(triton.cdiv(M, 16),)](
        outputs["projection"],
        inputs["weight"],
        outputs["output"],
        outputs["rstd"],
        M=M,
        N=N,
        EPS=EPS,
        num_warps=4,
        num_stages=3,
    )


def run_seed(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    block_m: int = 64,
    block_k: int = 64,
) -> None:
    num_programs = min(
        torch.cuda.get_device_properties(0).multi_processor_count,
        triton.cdiv(M, block_m),
    )
    fused_seed[(num_programs,)](
        inputs["a"],
        inputs["b"],
        inputs["weight"],
        outputs["projection"],
        outputs["output"],
        outputs["rstd"],
        M=M,
        N=N,
        K=K,
        EPS=EPS,
        NUM_PROGRAMS=num_programs,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=3,
    )


def accuracy(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> dict[str, dict[str, float | bool]]:
    tolerances = {
        "projection": (0.0, 0.0),
        "output": (5e-4, 1e-3),
        "rstd": (5e-5, 1e-5),
    }
    result = {}
    for name, (atol, rtol) in tolerances.items():
        reference = expected[name].float()
        candidate = actual[name].float()
        difference = candidate - reference
        result[name] = {
            "passed": bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol)),
            "relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference).clamp_min(1e-30)
            ),
            "max_abs": float(difference.abs().max()),
        }
    return result
