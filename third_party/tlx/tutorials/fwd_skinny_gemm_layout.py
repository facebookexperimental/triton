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

"""OSS reproduction of the T289757859 skinny GEMM-layout fusion."""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, TypedDict, cast

import torch
import triton
import triton.language as tl


M = 5120 * 256
N = 16
K = 64
GROUP_ROWS = 256


class TensorMap(TypedDict):
    a: torch.Tensor
    b: torch.Tensor
    projection: torch.Tensor
    layout: torch.Tensor


@triton.jit
def transpose_group_copy(
    source,
    destination,
    M: tl.constexpr,
    N: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M * N
    rows = offsets // N
    columns = offsets % N
    destination_offsets = (
        (rows // GROUP_ROWS) * (N * GROUP_ROWS)
        + columns * GROUP_ROWS
        + rows % GROUP_ROWS
    )
    value = tl.load(source + offsets, mask=mask)
    tl.store(destination + destination_offsets, value, mask=mask)


@triton.jit
def fused_seed(
    a,
    b,
    destination,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    num_pid_in_group: tl.constexpr = GROUP_M * num_pid_n
    for tile_id in range(start_pid, num_pid_m * num_pid_n, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + tile_id % group_size_m
        pid_n = tile_id % num_pid_in_group // group_size_m
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        safe_columns = tl.where(columns < N, columns, 0)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start_k in range(0, tl.cdiv(K, BLOCK_K)):
            ks = start_k * BLOCK_K + tl.arange(0, BLOCK_K)
            a_tile = tl.load(
                a + rows[:, None] * K + ks[None, :],
                mask=ks[None, :] < K,
                other=0.0,
            )
            b_tile = tl.load(
                b + safe_columns[None, :] * K + ks[:, None],
                mask=(ks[:, None] < K) & (columns[None, :] < N),
                other=0.0,
            )
            accumulator = tl.dot(a_tile, b_tile, accumulator, allow_tf32=False)
        mask = columns[None, :] < N
        destination_offsets = (
            (rows[:, None] // GROUP_ROWS) * (N * GROUP_ROWS)
            + columns[None, :] * GROUP_ROWS
            + rows[:, None] % GROUP_ROWS
        )
        tl.store(destination + destination_offsets, accumulator.to(tl.bfloat16), mask)


@triton.jit
def fused_narrow(
    a,
    b,
    destination,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, N)
    ks = tl.arange(0, K)
    a_tile = tl.load(a + rows[:, None] * K + ks[None, :])
    b_tile = tl.load(b + columns[:, None] * K + ks[None, :])
    accumulator = tl.dot(a_tile, b_tile.T, allow_tf32=False)
    destination_offsets = (
        (rows[:, None] // GROUP_ROWS) * (N * GROUP_ROWS)
        + columns[None, :] * GROUP_ROWS
        + rows[:, None] % GROUP_ROWS
    )
    tl.store(destination + destination_offsets, accumulator.to(tl.bfloat16))


@triton.jit
def fused_output_major(
    a,
    b,
    destination,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_ROWS
    rows = row_start + tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, N)
    ks = tl.arange(0, K)
    a_tile = tl.load(a + rows[:, None] * K + ks[None, :])
    b_tile = tl.load(b + columns[:, None] * K + ks[None, :])
    accumulator = tl.dot(b_tile, a_tile.T, allow_tf32=False)
    destination_offsets = (
        (row_start // GROUP_ROWS) * (N * GROUP_ROWS)
        + columns[:, None] * GROUP_ROWS
        + row_start % GROUP_ROWS
        + tl.arange(0, BLOCK_ROWS)[None, :]
    )
    tl.store(destination + destination_offsets, accumulator.to(tl.bfloat16))


@triton.jit
def fused_output_major_persistent(
    a,
    b,
    destination,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    columns = tl.arange(0, N)
    ks = tl.arange(0, K)
    b_tile = tl.load(b + columns[:, None] * K + ks[None, :])
    for tile in range(tl.program_id(0), M // BLOCK_ROWS, NUM_PROGRAMS):
        row_start = tile * BLOCK_ROWS
        rows = row_start + tl.arange(0, BLOCK_ROWS)
        a_tile = tl.load(a + rows[:, None] * K + ks[None, :])
        accumulator = tl.dot(b_tile, a_tile.T, allow_tf32=False)
        destination_offsets = (
            (row_start // GROUP_ROWS) * (N * GROUP_ROWS)
            + columns[:, None] * GROUP_ROWS
            + row_start % GROUP_ROWS
            + tl.arange(0, BLOCK_ROWS)[None, :]
        )
        tl.store(destination + destination_offsets, accumulator.to(tl.bfloat16))


def make_inputs(seed: int = 0) -> TensorMap:
    torch.manual_seed(seed)
    return cast(
        TensorMap,
        {
            "a": torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
            / math.sqrt(K),
            "b": torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
            / math.sqrt(K),
        },
    )


def make_outputs() -> TensorMap:
    return cast(
        TensorMap,
        {
            "projection": torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
            "layout": torch.empty(
                (M // GROUP_ROWS, N, GROUP_ROWS),
                device="cuda",
                dtype=torch.bfloat16,
            ),
        },
    )


def run_unfused(inputs: TensorMap, outputs: TensorMap) -> None:
    torch.mm(inputs["a"], inputs["b"].T, out=outputs["projection"])
    cast(Any, transpose_group_copy)[(triton.cdiv(M * N, 256),)](
        outputs["projection"],
        outputs["layout"],
        M=M,
        N=N,
        GROUP_ROWS=GROUP_ROWS,
        BLOCK=256,
        num_warps=8,
    )


def run_seed(inputs: TensorMap, outputs: TensorMap) -> None:
    cast(Any, fused_seed)[(_num_sms(),)](
        inputs["a"],
        inputs["b"],
        outputs["layout"],
        M=M,
        N=N,
        K=K,
        GROUP_ROWS=GROUP_ROWS,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        GROUP_M=8,
        NUM_SMS=_num_sms(),
        num_stages=3,
        num_warps=8,
    )


def run_narrow(
    inputs: TensorMap, outputs: TensorMap, *, block_m: int = 128, num_warps: int = 4
) -> None:
    cast(Any, fused_narrow)[(triton.cdiv(M, block_m),)](
        inputs["a"],
        inputs["b"],
        outputs["layout"],
        M=M,
        N=N,
        K=K,
        GROUP_ROWS=GROUP_ROWS,
        BLOCK_M=block_m,
        num_stages=1,
        num_warps=num_warps,
    )


def run_output_major(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_rows: int = 256,
    num_warps: int = 4,
    num_stages: int = 1,
) -> None:
    cast(Any, fused_output_major)[(triton.cdiv(M, block_rows),)](
        inputs["a"],
        inputs["b"],
        outputs["layout"],
        M=M,
        N=N,
        K=K,
        GROUP_ROWS=GROUP_ROWS,
        BLOCK_ROWS=block_rows,
        num_stages=num_stages,
        num_warps=num_warps,
    )


def run_output_major_persistent(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_rows: int = 256,
    num_programs: int | None = None,
    num_warps: int = 8,
    num_stages: int = 2,
    maxnreg: int | None = 64,
) -> None:
    if num_programs is None:
        num_programs = _num_sms() * 4
    launch_options: dict[str, int] = {
        "num_stages": num_stages,
        "num_warps": num_warps,
    }
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    cast(Any, fused_output_major_persistent)[(num_programs,)](
        inputs["a"],
        inputs["b"],
        outputs["layout"],
        M=M,
        N=N,
        K=K,
        GROUP_ROWS=GROUP_ROWS,
        BLOCK_ROWS=block_rows,
        NUM_PROGRAMS=num_programs,
        **launch_options,
    )


def _num_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def accuracy(reference: TensorMap, candidate: TensorMap) -> dict[str, float | bool]:
    expected = reference["layout"].float()
    difference = candidate["layout"].float() - expected
    return {
        "allclose": bool(
            torch.allclose(
                candidate["layout"], reference["layout"], atol=0.02, rtol=0.02
            )
        ),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected).clamp_min(1e-30)
        ),
        "exact": bool(torch.equal(reference["layout"], candidate["layout"])),
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
