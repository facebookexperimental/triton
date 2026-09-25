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

"""OSS reproduction for the T290058050 LayerNorm/mul GEMM prologue."""

from __future__ import annotations

import statistics
from typing import Any, Callable, TypedDict, cast

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


DEFAULT_ROWS = 2_097_152
FEATURES = 256
GEMM_K = 512
OUTPUT_FEATURES = 256
EPSILON = 1e-6


class TensorMap(TypedDict, total=False):
    x: torch.Tensor
    u: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    weight: torch.Tensor
    residual: torch.Tensor
    intermediate: torch.Tensor
    mean: torch.Tensor
    rstd: torch.Tensor
    projection: torch.Tensor


@triton.jit
def layernorm_stats_or_materialize(
    x_ptr,
    u_ptr,
    gamma_ptr,
    beta_ptr,
    intermediate_ptr,
    mean_ptr,
    rstd_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    EPSILON_VALUE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    MATERIALIZE: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, D)
    offsets = rows[:, None] * D + cols[None, :]
    mask = rows[:, None] < M
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1, keep_dims=True) / D
    centered = x - mean
    rstd = tl.rsqrt(
        tl.sum(centered * centered, axis=1, keep_dims=True) / D + EPSILON_VALUE
    )
    tl.store(mean_ptr + rows, tl.reshape(mean, (BLOCK_ROWS,)), mask=rows < M)
    tl.store(rstd_ptr + rows, tl.reshape(rstd, (BLOCK_ROWS,)), mask=rows < M)

    if MATERIALIZE:
        u = tl.load(u_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols).to(tl.float32)
        beta = tl.load(beta_ptr + cols).to(tl.float32)
        normalized = centered * rstd
        affine = normalized * gamma[None, :] + beta[None, :]
        silu = u * tl.sigmoid(u)
        output_offsets = rows[:, None] * K + cols[None, :]
        tl.store(intermediate_ptr + output_offsets, silu.to(tl.bfloat16), mask=mask)
        tl.store(
            intermediate_ptr + output_offsets + D,
            (affine * u).to(tl.bfloat16),
            mask=mask,
        )


@triton.jit
def fused_seed(
    x_desc,
    u_desc,
    weight_desc,
    residual_desc,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    output_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_n: tl.constexpr = N // BLOCK_N
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    rows = start_m + tl.arange(0, BLOCK_M)
    mean = tl.load(mean_ptr + rows).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, K, BLOCK_K):
        source_start = start_k % D
        features = source_start + tl.arange(0, BLOCK_K)
        x = x_desc.load([start_m, source_start]).to(tl.float32)
        u = u_desc.load([start_m, source_start]).to(tl.float32)
        gamma = tl.load(gamma_ptr + features).to(tl.float32)
        beta = tl.load(beta_ptr + features).to(tl.float32)
        normalized = (x - mean[:, None]) * rstd[:, None]
        affine = normalized * gamma[None, :] + beta[None, :]
        a = tl.where(start_k < D, u * tl.sigmoid(u), affine * u).to(tl.bfloat16)
        b = weight_desc.load([start_k, start_n])
        accumulator = tl.dot(a, b, accumulator, allow_tf32=False)
    residual = residual_desc.load([start_m, start_n]).to(tl.float32)
    output_desc.store([start_m, start_n], (accumulator + residual).to(tl.bfloat16))


def make_inputs(rows: int, seed: int = 0) -> TensorMap:
    torch.manual_seed(seed)
    return {
        "x": torch.randn((rows, FEATURES), device="cuda", dtype=torch.bfloat16),
        "u": torch.randn((rows, FEATURES), device="cuda", dtype=torch.bfloat16),
        "gamma": torch.randn((FEATURES,), device="cuda", dtype=torch.bfloat16),
        "beta": torch.randn((FEATURES,), device="cuda", dtype=torch.bfloat16),
        "weight": torch.randn(
            (GEMM_K, OUTPUT_FEATURES), device="cuda", dtype=torch.bfloat16
        ),
        "residual": torch.randn(
            (rows, OUTPUT_FEATURES), device="cuda", dtype=torch.bfloat16
        ),
    }


def make_outputs(rows: int, *, with_intermediate: bool) -> TensorMap:
    outputs: TensorMap = {
        "mean": torch.empty((rows,), device="cuda", dtype=torch.float32),
        "rstd": torch.empty((rows,), device="cuda", dtype=torch.float32),
        "projection": torch.empty(
            (rows, OUTPUT_FEATURES), device="cuda", dtype=torch.bfloat16
        ),
    }
    if with_intermediate:
        outputs["intermediate"] = torch.empty(
            (rows, GEMM_K), device="cuda", dtype=torch.bfloat16
        )
    return outputs


def run_stats(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_rows: int = 8,
    num_warps: int = 4,
) -> None:
    rows = inputs["x"].shape[0]
    cast(Any, layernorm_stats_or_materialize)[(triton.cdiv(rows, block_rows),)](
        inputs["x"],
        inputs["u"],
        inputs["gamma"],
        inputs["beta"],
        outputs["projection"],
        outputs["mean"],
        outputs["rstd"],
        M=rows,
        D=FEATURES,
        K=GEMM_K,
        EPSILON_VALUE=EPSILON,
        BLOCK_ROWS=block_rows,
        MATERIALIZE=False,
        num_warps=num_warps,
    )


def run_materialize(
    inputs: TensorMap, outputs: TensorMap, *, block_rows: int = 8
) -> None:
    rows = inputs["x"].shape[0]
    cast(Any, layernorm_stats_or_materialize)[(triton.cdiv(rows, block_rows),)](
        inputs["x"],
        inputs["u"],
        inputs["gamma"],
        inputs["beta"],
        outputs["intermediate"],
        outputs["mean"],
        outputs["rstd"],
        M=rows,
        D=FEATURES,
        K=GEMM_K,
        EPSILON_VALUE=EPSILON,
        BLOCK_ROWS=block_rows,
        MATERIALIZE=True,
        num_warps=8,
    )


def run_unfused(inputs: TensorMap, outputs: TensorMap, *, block_rows: int = 8) -> None:
    run_materialize(inputs, outputs, block_rows=block_rows)
    torch.addmm(
        inputs["residual"],
        outputs["intermediate"],
        inputs["weight"],
        out=outputs["projection"],
    )


def run_reference(inputs: TensorMap, outputs: TensorMap) -> None:
    """Evaluate the BF16 boundary with an explicit FP32 GEMM accumulation."""
    run_materialize(inputs, outputs)
    projection = torch.mm(
        outputs["intermediate"], inputs["weight"], out_dtype=torch.float32
    )
    outputs["projection"].copy_(
        (projection + inputs["residual"].float()).to(torch.bfloat16)
    )


def run_seed(inputs: TensorMap, outputs: TensorMap) -> None:
    rows = inputs["x"].shape[0]
    block_m = 32
    block_n = OUTPUT_FEATURES
    block_k = 64
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
        [block_k, block_n],
    )
    residual_desc = TensorDescriptor(
        inputs["residual"],
        inputs["residual"].shape,
        inputs["residual"].stride(),
        [block_m, block_n],
    )
    output_desc = TensorDescriptor(
        outputs["projection"],
        outputs["projection"].shape,
        outputs["projection"].stride(),
        [block_m, block_n],
    )
    cast(Any, fused_seed)[(triton.cdiv(rows, block_m),)](
        x_desc,
        u_desc,
        weight_desc,
        residual_desc,
        inputs["gamma"],
        inputs["beta"],
        outputs["mean"],
        outputs["rstd"],
        output_desc,
        M=rows,
        N=OUTPUT_FEATURES,
        K=GEMM_K,
        D=FEATURES,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=3,
    )


def accuracy(reference: TensorMap, candidate: TensorMap) -> dict[str, dict[str, Any]]:
    limits = {"projection": 1e-4, "mean": 1e-6, "rstd": 1e-6}
    tolerances = {
        "projection": (2e-2, 2e-2),
        "mean": (1e-6, 1e-6),
        "rstd": (1e-6, 1e-6),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, limit in limits.items():
        actual = candidate[name].float()
        expected = reference[name].float()
        atol, rtol = tolerances[name]
        rel_l2 = float(
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected)
        )
        allclose = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
        result[name] = {
            "allclose": allclose,
            "atol": atol,
            "rtol": rtol,
            "relative_l2": rel_l2,
            "max_abs": float((actual - expected).abs().max()),
            "limit": limit,
            "passed": allclose or rel_l2 <= limit,
        }
    return result


def benchmark(
    fn: Callable[[], None], *, warmup: int, samples: int, reps: int
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
