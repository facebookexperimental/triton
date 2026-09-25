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

"""OSS reproduction for the T290048886 saved-LayerNorm GEMM fan-out."""

from __future__ import annotations

import statistics
from typing import Any, Callable, TypedDict, cast

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


DEFAULT_ROWS = 2_097_152
FEATURES = 256
PROJECTION = 768
GRADIENT = 1024
SPLIT_K = 128
EPSILON = 1e-6


class TensorMap(TypedDict, total=False):
    x: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    mean: torch.Tensor
    rstd: torch.Tensor
    projection_weight: torch.Tensor
    projection_bias: torch.Tensor
    gradient: torch.Tensor
    hidden: torch.Tensor
    projection: torch.Tensor
    dweight: torch.Tensor
    partial: torch.Tensor


@triton.jit
def materialize_saved_layernorm(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    hidden_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, D)
    mask = rows[:, None] < M
    offsets = rows[:, None] * D + columns[None, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=rows < M).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=rows < M).to(tl.float32)
    gamma = tl.load(gamma_ptr + columns).to(tl.float32)
    beta = tl.load(beta_ptr + columns).to(tl.float32)
    hidden = (x - mean[:, None]) * rstd[:, None]
    hidden = hidden * gamma[None, :] + beta[None, :]
    tl.store(hidden_ptr + offsets, hidden.to(tl.bfloat16), mask=mask)


@triton.jit
def projection_seed(
    x_desc,
    weight_desc,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    bias_ptr,
    output_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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
        columns = start_k + tl.arange(0, BLOCK_K)
        x = x_desc.load([start_m, start_k]).to(tl.float32)
        gamma = tl.load(gamma_ptr + columns).to(tl.float32)
        beta = tl.load(beta_ptr + columns).to(tl.float32)
        hidden = (x - mean[:, None]) * rstd[:, None]
        hidden = (hidden * gamma[None, :] + beta[None, :]).to(tl.bfloat16)
        weight = weight_desc.load([start_k, start_n])
        accumulator = tl.dot(hidden, weight, accumulator, allow_tf32=False)
    columns = start_n + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + columns).to(tl.float32)
    output_desc.store([start_m, start_n], (accumulator + bias[None, :]).to(tl.bfloat16))


@triton.jit
def dweight_seed(
    x_desc,
    gradient_desc,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    partial_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    split = tl.program_id(2)
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    chunk: tl.constexpr = tl.cdiv(K, SPLIT)
    split_start = split * chunk
    split_end = tl.minimum(split_start + chunk, K)
    features = start_m + tl.arange(0, BLOCK_M)
    gamma = tl.load(gamma_ptr + features).to(tl.float32)
    beta = tl.load(beta_ptr + features).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(split_start, split_end, BLOCK_K):
        rows = start_k + tl.arange(0, BLOCK_K)
        x = x_desc.load([start_k, start_m]).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=rows < K).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=rows < K).to(tl.float32)
        hidden = (x - mean[:, None]) * rstd[:, None]
        hidden = (hidden * gamma[None, :] + beta[None, :]).to(tl.bfloat16)
        gradient = gradient_desc.load([start_k, start_n])
        accumulator = tl.dot(hidden.T, gradient, accumulator, allow_tf32=False)
    partial_desc.store([split, start_m, start_n], tl.expand_dims(accumulator, axis=0))


@triton.jit
def finish_dweight(
    partial_ptr,
    output_ptr,
    D: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < D * N
    total = tl.zeros((BLOCK,), tl.float32)
    for split in range(SPLIT):
        total += tl.load(partial_ptr + split * D * N + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, total.to(tl.bfloat16), mask=mask)


def make_inputs(rows: int, seed: int = 0) -> TensorMap:
    torch.manual_seed(seed)
    x = torch.randn((rows, FEATURES), device="cuda", dtype=torch.bfloat16)
    mean = x.float().mean(dim=1)
    rstd = torch.rsqrt((x.float() - mean[:, None]).square().mean(dim=1) + EPSILON)
    projection_storage = torch.randn(
        (FEATURES, 1024), device="cuda", dtype=torch.bfloat16
    )
    return {
        "x": x,
        "gamma": torch.randn((FEATURES,), device="cuda", dtype=torch.bfloat16),
        "beta": torch.randn((FEATURES,), device="cuda", dtype=torch.bfloat16),
        "mean": mean,
        "rstd": rstd,
        "projection_weight": projection_storage[:, :PROJECTION],
        "projection_bias": torch.randn(
            (PROJECTION,), device="cuda", dtype=torch.bfloat16
        ),
        "gradient": torch.randn((rows, GRADIENT), device="cuda", dtype=torch.bfloat16),
    }


def make_outputs(rows: int, *, with_hidden: bool, split_k: int = SPLIT_K) -> TensorMap:
    outputs: TensorMap = {
        "projection": torch.empty(
            (rows, PROJECTION), device="cuda", dtype=torch.bfloat16
        ),
        "dweight": torch.empty(
            (FEATURES, GRADIENT), device="cuda", dtype=torch.bfloat16
        ),
        "partial": torch.empty(
            (split_k, FEATURES, GRADIENT), device="cuda", dtype=torch.float32
        ),
    }
    if with_hidden:
        outputs["hidden"] = torch.empty(
            (rows, FEATURES), device="cuda", dtype=torch.bfloat16
        )
    return outputs


def run_layernorm(
    inputs: TensorMap,
    hidden: torch.Tensor,
    *,
    block_rows: int = 8,
    num_warps: int = 4,
) -> None:
    rows = inputs["x"].shape[0]
    cast(Any, materialize_saved_layernorm)[(triton.cdiv(rows, block_rows),)](
        inputs["x"],
        inputs["gamma"],
        inputs["beta"],
        inputs["mean"],
        inputs["rstd"],
        hidden,
        M=rows,
        D=FEATURES,
        BLOCK_ROWS=block_rows,
        num_warps=num_warps,
    )


def run_projection_unfused(inputs: TensorMap, outputs: TensorMap) -> None:
    run_layernorm(inputs, outputs["hidden"])
    torch.addmm(
        inputs["projection_bias"],
        outputs["hidden"],
        inputs["projection_weight"],
        out=outputs["projection"],
    )


def run_dweight_unfused(inputs: TensorMap, outputs: TensorMap) -> None:
    run_layernorm(inputs, outputs["hidden"])
    torch.mm(outputs["hidden"].T, inputs["gradient"], out=outputs["dweight"])


def run_dual_unfused(inputs: TensorMap, outputs: TensorMap) -> None:
    run_layernorm(inputs, outputs["hidden"])
    torch.addmm(
        inputs["projection_bias"],
        outputs["hidden"],
        inputs["projection_weight"],
        out=outputs["projection"],
    )
    torch.mm(outputs["hidden"].T, inputs["gradient"], out=outputs["dweight"])


def run_projection_seed(inputs: TensorMap, outputs: TensorMap) -> None:
    rows = inputs["x"].shape[0]
    block_m, block_n, block_k = 128, 256, 64
    x_desc = TensorDescriptor(
        inputs["x"], inputs["x"].shape, inputs["x"].stride(), [block_m, block_k]
    )
    weight_desc = TensorDescriptor(
        inputs["projection_weight"],
        inputs["projection_weight"].shape,
        inputs["projection_weight"].stride(),
        [block_k, block_n],
    )
    output_desc = TensorDescriptor(
        outputs["projection"],
        outputs["projection"].shape,
        outputs["projection"].stride(),
        [block_m, block_n],
    )
    cast(Any, projection_seed)[
        (triton.cdiv(rows, block_m) * triton.cdiv(PROJECTION, block_n),)
    ](
        x_desc,
        weight_desc,
        inputs["gamma"],
        inputs["beta"],
        inputs["mean"],
        inputs["rstd"],
        inputs["projection_bias"],
        output_desc,
        M=rows,
        N=PROJECTION,
        K=FEATURES,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=3,
    )


def run_dweight_seed(
    inputs: TensorMap, outputs: TensorMap, *, split_k: int = SPLIT_K
) -> None:
    rows = inputs["x"].shape[0]
    block_m, block_n, block_k = 64, 128, 64
    x_desc = TensorDescriptor(
        inputs["x"], inputs["x"].shape, inputs["x"].stride(), [block_k, block_m]
    )
    gradient_desc = TensorDescriptor(
        inputs["gradient"],
        inputs["gradient"].shape,
        inputs["gradient"].stride(),
        [block_k, block_n],
    )
    partial_desc = TensorDescriptor(
        outputs["partial"],
        outputs["partial"].shape,
        outputs["partial"].stride(),
        [1, block_m, block_n],
    )
    cast(Any, dweight_seed)[
        (
            triton.cdiv(FEATURES, block_m),
            triton.cdiv(GRADIENT, block_n),
            split_k,
        )
    ](
        x_desc,
        gradient_desc,
        inputs["gamma"],
        inputs["beta"],
        inputs["mean"],
        inputs["rstd"],
        partial_desc,
        M=FEATURES,
        N=GRADIENT,
        K=rows,
        SPLIT=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=3,
    )
    cast(Any, finish_dweight)[(triton.cdiv(FEATURES * GRADIENT, 256),)](
        outputs["partial"],
        outputs["dweight"],
        D=FEATURES,
        N=GRADIENT,
        SPLIT=split_k,
        BLOCK=256,
        num_warps=8,
    )


def accuracy(
    reference: TensorMap, candidate: TensorMap, names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    limits = {"projection": 1e-4, "dweight": 1e-3}
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        actual = candidate[name].float()
        expected = reference[name].float()
        difference = actual - expected
        rel_l2 = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected).clamp_min(1e-30)
        )
        result[name] = {
            "relative_l2": rel_l2,
            "max_abs": float(difference.abs().max()),
            "limit": limits[name],
            "passed": rel_l2 <= limits[name],
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
