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

"""OSS benchmark for GEMM -> affine LayerNorm backward -> residual add.

This reproduces the two symmetric regions tracked by T290084482 without any
fbsource or Buck dependency.  The production GEMM shape is
``(M, N, K) = (2097152, 256, 1024)``.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable, TypedDict, cast

import torch
import triton
import triton.language as tl


DEFAULT_ROWS = 2_097_152
FEATURES = 256
GEMM_K = 1024
SEED_BLOCK_M = 32
SEED_BLOCK_K = 64


class TensorMap(TypedDict, total=False):
    gradient: torch.Tensor
    projection_weight: torch.Tensor
    x: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    mean: torch.Tensor
    rstd: torch.Tensor
    residual: torch.Tensor
    dy: torch.Tensor
    dx: torch.Tensor
    final: torch.Tensor
    dweight: torch.Tensor
    dbias: torch.Tensor
    partial_dw: torch.Tensor
    partial_db: torch.Tensor


@triton.jit
def weighted_layernorm_bwd(
    dy_ptr,
    x_ptr,
    gamma_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    final_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N)
    mask = rows[:, None] < M

    offsets = rows[:, None] * N + cols[None, :]
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=rows < M, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=rows < M, other=0.0).to(tl.float32)

    xhat = (x - mean[:, None]) * rstd[:, None]
    wdy = gamma[None, :] * dy
    c1 = tl.sum(xhat * wdy, axis=1) / N
    c2 = tl.sum(wdy, axis=1) / N
    dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]

    # Match the original materialized BF16 dx before its FP32 residual add.
    dx = dx.to(tl.bfloat16).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(final_ptr + offsets, (residual + dx).to(tl.bfloat16), mask=mask)

    partial_offsets = pid * N + cols
    tl.store(partial_dw_ptr + partial_offsets, tl.sum(dy * xhat, axis=0))
    tl.store(partial_db_ptr + partial_offsets, tl.sum(dy, axis=0))


@triton.jit
def weighted_layernorm_bwd_persistent(
    dx_ptr,
    dy_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    x_ptr,
    gamma_ptr,
    mean_ptr,
    rstd_ptr,
    M,
    N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_count = tl.num_programs(0)
    block_count = tl.cdiv(M, BLOCK_ROWS)
    blocks_per_tile = block_count // tile_count
    if pid < block_count % tile_count:
        blocks_per_tile += 1

    cols = tl.arange(0, N)
    gamma = tl.load(gamma_ptr + cols).to(tl.float32)
    acc_dw = tl.zeros((N,), tl.float32)
    acc_db = tl.zeros((N,), tl.float32)
    for index in range(0, blocks_per_tile):
        block = pid + index * tile_count
        rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        mask = rows[:, None] < M
        offsets = rows[:, None] * N + cols[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=rows < M, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=rows < M, other=0.0).to(tl.float32)
        xhat = (x - mean[:, None]) * rstd[:, None]
        wdy = gamma[None, :] * dy
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=1, keep_dims=True) / N
        c2 = tl.sum(wdy, axis=1, keep_dims=True) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd[:, None]
        tl.store(dx_ptr + offsets, dx.to(tl.bfloat16), mask=mask)
        acc_dw += tl.sum(dy * xhat, axis=0)
        acc_db += tl.sum(dy, axis=0)

    partial_offsets = pid * N + cols
    tl.store(partial_dw_ptr + partial_offsets, acc_dw)
    tl.store(partial_db_ptr + partial_offsets, acc_db)


@triton.jit
def finish_weighted_layernorm_dwdb(
    partial_dw_ptr,
    partial_db_ptr,
    dweight_ptr,
    dbias_ptr,
    PARTIAL_ROWS,
    N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    total_dw = tl.zeros((BLOCK_N,), tl.float32)
    total_db = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, PARTIAL_ROWS, BLOCK_ROWS):
        rows = start + tl.arange(0, BLOCK_ROWS)
        offsets = rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < PARTIAL_ROWS) & (cols[None, :] < N)
        total_dw += tl.sum(
            tl.load(partial_dw_ptr + offsets, mask=mask, other=0.0), axis=0
        )
        total_db += tl.sum(
            tl.load(partial_db_ptr + offsets, mask=mask, other=0.0), axis=0
        )
    mask = cols < N
    tl.store(dweight_ptr + cols, total_dw.to(tl.bfloat16), mask=mask)
    tl.store(dbias_ptr + cols, total_db.to(tl.bfloat16), mask=mask)


@triton.jit
def cast_dwdb(acc_dw_ptr, acc_db_ptr, dw_ptr, db_ptr, N: tl.constexpr):
    cols = tl.arange(0, N)
    tl.store(dw_ptr + cols, tl.load(acc_dw_ptr + cols).to(tl.bfloat16))
    tl.store(db_ptr + cols, tl.load(acc_db_ptr + cols).to(tl.bfloat16))


@triton.jit
def fused_seed(
    gradient_ptr,
    projection_weight_ptr,
    x_ptr,
    gamma_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    final_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N)
    accumulator = tl.zeros((BLOCK_M, N), tl.float32)
    for start_k in range(0, K, BLOCK_K):
        reduction = start_k + tl.arange(0, BLOCK_K)
        gradient = tl.load(gradient_ptr + rows[:, None] * K + reduction[None, :])
        weight = tl.load(projection_weight_ptr + cols[:, None] * K + reduction[None, :])
        accumulator = tl.dot(gradient, weight.T, accumulator, allow_tf32=False)

    dy = accumulator.to(tl.bfloat16).to(tl.float32)
    offsets = rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptr + offsets).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols).to(tl.float32)
    mean = tl.load(mean_ptr + rows).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows).to(tl.float32)
    xhat = (x - mean[:, None]) * rstd[:, None]
    wdy = gamma[None, :] * dy
    c1 = tl.sum(xhat * wdy, axis=1) / N
    c2 = tl.sum(wdy, axis=1) / N
    dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
    dx = dx.to(tl.bfloat16).to(tl.float32)
    residual = tl.load(residual_ptr + offsets).to(tl.float32)
    tl.store(final_ptr + offsets, (residual + dx).to(tl.bfloat16))
    partial_offsets = pid * N + cols
    tl.store(partial_dw_ptr + partial_offsets, tl.sum(dy * xhat, axis=0))
    tl.store(partial_db_ptr + partial_offsets, tl.sum(dy, axis=0))


def make_inputs(rows: int, seed: int = 0) -> TensorMap:
    torch.manual_seed(seed)
    x = torch.randn((rows, FEATURES), device="cuda", dtype=torch.bfloat16)
    x_f32 = x.float()
    mean = x_f32.mean(dim=1)
    rstd = torch.rsqrt((x_f32 - mean[:, None]).square().mean(dim=1) + 1e-6)
    return {
        "gradient": torch.randn((rows, GEMM_K), device="cuda", dtype=torch.bfloat16),
        "projection_weight": torch.randn(
            (FEATURES, GEMM_K), device="cuda", dtype=torch.bfloat16
        ),
        "x": x,
        "gamma": torch.randn((FEATURES,), device="cuda", dtype=torch.bfloat16),
        "beta": torch.randn((FEATURES,), device="cuda", dtype=torch.bfloat16),
        "mean": mean,
        "rstd": rstd,
        "residual": torch.randn((rows, FEATURES), device="cuda", dtype=torch.bfloat16),
    }


def make_outputs(rows: int, partial_rows: int) -> TensorMap:
    return {
        "dy": torch.empty((rows, FEATURES), device="cuda", dtype=torch.bfloat16),
        "dx": torch.empty((rows, FEATURES), device="cuda", dtype=torch.bfloat16),
        "final": torch.empty((rows, FEATURES), device="cuda", dtype=torch.bfloat16),
        "dweight": torch.empty((FEATURES,), device="cuda", dtype=torch.bfloat16),
        "dbias": torch.empty((FEATURES,), device="cuda", dtype=torch.bfloat16),
        "partial_dw": torch.empty(
            (partial_rows, FEATURES), device="cuda", dtype=torch.float32
        ),
        "partial_db": torch.empty(
            (partial_rows, FEATURES), device="cuda", dtype=torch.float32
        ),
    }


UNFUSED_CONFIG = {"BLOCK_ROWS": 16, "SHARDS_PER_SM": 8, "NUM_WARPS": 2}


def run_unfused(inputs: TensorMap, outputs: TensorMap) -> None:
    rows = inputs["gradient"].shape[0]
    torch.mm(
        inputs["gradient"],
        inputs["projection_weight"].T,
        out=outputs["dy"],
    )
    sms = torch.cuda.get_device_properties(
        inputs["gradient"].device
    ).multi_processor_count
    tile_count = max(1, min(sms * UNFUSED_CONFIG["SHARDS_PER_SM"], rows // 4))
    cast(Any, weighted_layernorm_bwd_persistent)[(tile_count,)](
        outputs["dx"],
        outputs["dy"],
        outputs["partial_dw"],
        outputs["partial_db"],
        inputs["x"],
        inputs["gamma"],
        inputs["mean"],
        inputs["rstd"],
        rows,
        N=FEATURES,
        BLOCK_ROWS=UNFUSED_CONFIG["BLOCK_ROWS"],
        num_warps=UNFUSED_CONFIG["NUM_WARPS"],
    )
    torch.add(inputs["residual"], outputs["dx"], out=outputs["final"])
    cast(Any, finish_weighted_layernorm_dwdb)[(triton.cdiv(FEATURES, 32),)](
        outputs["partial_dw"],
        outputs["partial_db"],
        outputs["dweight"],
        outputs["dbias"],
        tile_count,
        N=FEATURES,
        BLOCK_ROWS=256,
        BLOCK_N=32,
        num_warps=8,
    )


def run_seed(inputs: TensorMap, outputs: TensorMap) -> None:
    rows = inputs["gradient"].shape[0]
    partial_rows = triton.cdiv(rows, SEED_BLOCK_M)
    cast(Any, fused_seed)[(partial_rows,)](
        inputs["gradient"],
        inputs["projection_weight"],
        inputs["x"],
        inputs["gamma"],
        inputs["mean"],
        inputs["rstd"],
        inputs["residual"],
        outputs["final"],
        outputs["partial_dw"],
        outputs["partial_db"],
        M=rows,
        N=FEATURES,
        K=GEMM_K,
        BLOCK_M=SEED_BLOCK_M,
        BLOCK_K=SEED_BLOCK_K,
        num_stages=3,
        num_warps=8,
    )
    cast(Any, finish_weighted_layernorm_dwdb)[(triton.cdiv(FEATURES, 32),)](
        outputs["partial_dw"],
        outputs["partial_db"],
        outputs["dweight"],
        outputs["dbias"],
        partial_rows,
        N=FEATURES,
        BLOCK_ROWS=256,
        BLOCK_N=32,
        num_warps=8,
    )


def benchmark(
    fn: Callable[[], None], *, warmup: int, samples: int, reps: int
) -> list[float]:
    medians = []
    for _ in range(reps):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
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


def accuracy(
    reference: TensorMap, candidate: TensorMap
) -> dict[str, dict[str, float | bool]]:
    result: dict[str, dict[str, float | bool]] = {}
    limits = {"final": 1e-3, "dweight": 2e-3, "dbias": 2e-3}
    for name, limit in limits.items():
        ref = reference[name].float()
        got = candidate[name].float()
        diff = got - ref
        rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref))
        max_abs = float(diff.abs().max())
        result[name] = {
            "relative_l2": rel_l2,
            "max_abs": max_abs,
            "limit": limit,
            "passed": rel_l2 <= limit,
        }
    return result
