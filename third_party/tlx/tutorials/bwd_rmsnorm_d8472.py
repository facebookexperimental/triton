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

"""OSS reproduction and tuning harness for T289881412."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable, TypedDict, cast

import torch
import triton
import triton.language as tl


N = 5120
D = 8472
X_STRIDE = 8512
SOURCE_PARTITIONS = 1024
MAX_PARTITIONS = 1280
MAX_COLUMN_TILES = 32


class TensorMap(TypedDict):
    x: torch.Tensor
    dy: torch.Tensor
    weight: torch.Tensor
    rstd: torch.Tensor
    dx: torch.Tensor
    dweight_partial: torch.Tensor
    dweight: torch.Tensor
    row_dot_partial: torch.Tensor


@triton.jit
def weighted_rmsnorm_backward_source(
    dx,
    dy,
    dweight_partial,
    x,
    weight,
    rstd,
    D,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PARTITIONS: tl.constexpr,
    X_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks: tl.constexpr = tl.cdiv(N, BLOCK_N)
    blocks_per_tile: tl.constexpr = num_blocks // PARTITIONS
    if pid < num_blocks % PARTITIONS:
        blocks_per_tile += 1

    columns = tl.arange(0, BLOCK_D)
    column_mask = columns < D
    scale = tl.load(weight + columns, mask=column_mask, other=0.0).to(tl.float32)
    accumulated_dweight = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for index in range(blocks_per_tile):
        row_block = pid + index * PARTITIONS
        rows = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < N) & column_mask[None, :]
        x_value = tl.load(
            x + rows[:, None] * X_STRIDE + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dy_value = tl.load(
            dy + rows[:, None] * D + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        inverse_rms = tl.load(rstd + rows, mask=rows < N, other=0.0)
        normalized = x_value * inverse_rms[:, None]
        scaled_gradient = scale[None, :] * dy_value
        row_dot = tl.sum(normalized * scaled_gradient, axis=1) / D
        result = (scaled_gradient - normalized * row_dot[:, None]) * inverse_rms[
            :, None
        ]
        tl.store(dx + rows[:, None] * D + columns[None, :], result, mask=mask)
        accumulated_dweight += tl.sum(dy_value * normalized, axis=0)

    tl.store(
        dweight_partial + pid * D + columns,
        accumulated_dweight,
        mask=column_mask,
    )


@triton.jit
def rmsnorm_backward_partial(
    x,
    dy,
    weight,
    rstd,
    row_dot_partial,
    dweight_partial,
    N: tl.constexpr,
    D: tl.constexpr,
    X_STRIDE: tl.constexpr,
    ROW_PARTIAL_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_block = tl.program_id(0)
    column_tile = tl.program_id(1)
    rows = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
    columns = column_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (rows[:, None] < N) & (columns[None, :] < D)
    x_value = tl.load(
        x + rows[:, None] * X_STRIDE + columns[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dy_value = tl.load(
        dy + rows[:, None] * D + columns[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(weight + columns, mask=columns < D, other=0.0).to(tl.float32)
    inverse_rms = tl.load(rstd + rows, mask=rows < N, other=0.0)

    normalized = x_value * inverse_rms[:, None]
    scaled_gradient = dy_value * scale[None, :]
    row_partial = tl.sum(normalized * scaled_gradient, axis=1)
    tl.store(
        row_dot_partial + rows * ROW_PARTIAL_STRIDE + column_tile,
        row_partial,
        mask=rows < N,
    )
    dweight = tl.sum(dy_value * normalized, axis=0)
    tl.store(
        dweight_partial + row_block * D + columns,
        dweight,
        mask=columns < D,
    )


@triton.jit
def rmsnorm_backward_finish(
    x,
    dy,
    weight,
    rstd,
    row_dot_partial,
    dx,
    N: tl.constexpr,
    D: tl.constexpr,
    X_STRIDE: tl.constexpr,
    ROW_PARTIAL_STRIDE: tl.constexpr,
    COLUMN_TILES: tl.constexpr,
    PARTIAL_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_block = tl.program_id(0)
    column_tile = tl.program_id(1)
    rows = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
    columns = column_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (rows[:, None] < N) & (columns[None, :] < D)

    partial_indices = tl.arange(0, PARTIAL_BLOCK)
    partials = tl.load(
        row_dot_partial + rows[:, None] * ROW_PARTIAL_STRIDE + partial_indices[None, :],
        mask=(rows[:, None] < N) & (partial_indices[None, :] < COLUMN_TILES),
        other=0.0,
    )
    row_dot = tl.sum(partials, axis=1) / D
    x_value = tl.load(
        x + rows[:, None] * X_STRIDE + columns[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dy_value = tl.load(
        dy + rows[:, None] * D + columns[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(weight + columns, mask=columns < D, other=0.0).to(tl.float32)
    inverse_rms = tl.load(rstd + rows, mask=rows < N, other=0.0)
    normalized = x_value * inverse_rms[:, None]
    scaled_gradient = dy_value * scale[None, :]
    result = (scaled_gradient - normalized * row_dot[:, None]) * inverse_rms[:, None]
    tl.store(dx + rows[:, None] * D + columns[None, :], result, mask=mask)


@triton.jit
def rmsnorm_backward_rowwise(
    x,
    dy,
    weight,
    rstd,
    dx,
    dweight_partial,
    N: tl.constexpr,
    D: tl.constexpr,
    X_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_block = tl.program_id(0)
    rows = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
    column_offsets = tl.arange(0, BLOCK_D)
    inverse_rms = tl.load(rstd + rows, mask=rows < N, other=0.0)
    weighted_dot = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for column_start in range(0, D, BLOCK_D):
        columns = column_start + column_offsets
        mask = (rows[:, None] < N) & (columns[None, :] < D)
        x_value = tl.load(
            x + rows[:, None] * X_STRIDE + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dy_value = tl.load(
            dy + rows[:, None] * D + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(weight + columns, mask=columns < D, other=0.0).to(tl.float32)
        weighted_dot += tl.sum(x_value * dy_value * scale[None, :], axis=1)
    row_dot = weighted_dot * inverse_rms / D

    for column_start in range(0, D, BLOCK_D):
        columns = column_start + column_offsets
        mask = (rows[:, None] < N) & (columns[None, :] < D)
        x_value = tl.load(
            x + rows[:, None] * X_STRIDE + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dy_value = tl.load(
            dy + rows[:, None] * D + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(weight + columns, mask=columns < D, other=0.0).to(tl.float32)
        normalized = x_value * inverse_rms[:, None]
        scaled_gradient = dy_value * scale[None, :]
        result = (scaled_gradient - normalized * row_dot[:, None]) * inverse_rms[
            :, None
        ]
        tl.store(
            dx + rows[:, None] * D + columns[None, :],
            result.to(tl.bfloat16),
            mask=mask,
        )
        dweight = tl.sum(dy_value * normalized, axis=0)
        tl.store(
            dweight_partial + row_block * D + columns,
            dweight,
            mask=columns < D,
        )


@triton.jit
def finish_dweight(
    dweight_partial,
    dweight,
    D: tl.constexpr,
    PARTITIONS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    columns = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    accumulated = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for offset in range(0, PARTITIONS, BLOCK_N):
        rows = offset + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < PARTITIONS) & (columns[None, :] < D)
        accumulated += tl.load(
            dweight_partial + rows[:, None] * D + columns[None, :],
            mask=mask,
            other=0.0,
        )
    result = tl.sum(accumulated, axis=0)
    tl.store(dweight + columns, result.to(tl.bfloat16), mask=columns < D)


def make_inputs(seed: int = 0) -> TensorMap:
    torch.manual_seed(seed)
    storage = torch.randn((N, X_STRIDE), device="cuda", dtype=torch.bfloat16)
    return cast(
        TensorMap,
        {
            "x": storage[:, :D],
            "dy": torch.randn((N, D), device="cuda", dtype=torch.bfloat16),
            "weight": torch.randn((D,), device="cuda", dtype=torch.bfloat16),
            "rstd": torch.rand((N,), device="cuda", dtype=torch.float32) + 0.5,
        },
    )


def make_outputs() -> TensorMap:
    return cast(
        TensorMap,
        {
            "dx": torch.empty((N, D), device="cuda", dtype=torch.bfloat16),
            "dweight_partial": torch.empty(
                (MAX_PARTITIONS, D), device="cuda", dtype=torch.float32
            ),
            "dweight": torch.empty((D,), device="cuda", dtype=torch.bfloat16),
            "row_dot_partial": torch.empty(
                (N, MAX_COLUMN_TILES), device="cuda", dtype=torch.float32
            ),
        },
    )


def run_finalizer(
    outputs: TensorMap,
    *,
    partitions: int,
    block_n: int = 256,
    block_d: int = 16,
    num_warps: int = 16,
    num_stages: int = 3,
    maxnreg: int | None = None,
) -> None:
    launch_options: dict[str, int] = {
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    cast(Any, finish_dweight)[(triton.cdiv(D, block_d),)](
        outputs["dweight_partial"],
        outputs["dweight"],
        D=D,
        PARTITIONS=partitions,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        **launch_options,
    )


def run_unfused(inputs: TensorMap, outputs: TensorMap) -> None:
    cast(Any, weighted_rmsnorm_backward_source)[(SOURCE_PARTITIONS,)](
        outputs["dx"],
        inputs["dy"],
        outputs["dweight_partial"],
        inputs["x"],
        inputs["weight"],
        inputs["rstd"],
        D=D,
        N=N,
        BLOCK_N=8,
        BLOCK_D=16384,
        PARTITIONS=SOURCE_PARTITIONS,
        X_STRIDE=X_STRIDE,
        num_warps=8,
        num_stages=3,
    )
    run_finalizer(outputs, partitions=SOURCE_PARTITIONS)


def run_partial(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_n: int = 8,
    block_d: int = 1024,
    partial_warps: int = 8,
    partial_stages: int = 3,
    partial_maxnreg: int | None = None,
) -> None:
    row_blocks = triton.cdiv(N, block_n)
    column_tiles = triton.cdiv(D, block_d)
    launch_options: dict[str, int] = {
        "num_warps": partial_warps,
        "num_stages": partial_stages,
    }
    if partial_maxnreg is not None:
        launch_options["maxnreg"] = partial_maxnreg
    cast(Any, rmsnorm_backward_partial)[(row_blocks, column_tiles)](
        inputs["x"],
        inputs["dy"],
        inputs["weight"],
        inputs["rstd"],
        outputs["row_dot_partial"],
        outputs["dweight_partial"],
        N=N,
        D=D,
        X_STRIDE=X_STRIDE,
        ROW_PARTIAL_STRIDE=MAX_COLUMN_TILES,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        **launch_options,
    )


def run_dx_finish(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_n: int = 8,
    block_d: int = 1024,
    finish_warps: int = 8,
    finish_stages: int = 3,
    finish_maxnreg: int | None = None,
) -> None:
    row_blocks = triton.cdiv(N, block_n)
    column_tiles = triton.cdiv(D, block_d)
    partial_block = triton.next_power_of_2(column_tiles)
    launch_options: dict[str, int] = {
        "num_warps": finish_warps,
        "num_stages": finish_stages,
    }
    if finish_maxnreg is not None:
        launch_options["maxnreg"] = finish_maxnreg
    cast(Any, rmsnorm_backward_finish)[(row_blocks, column_tiles)](
        inputs["x"],
        inputs["dy"],
        inputs["weight"],
        inputs["rstd"],
        outputs["row_dot_partial"],
        outputs["dx"],
        N=N,
        D=D,
        X_STRIDE=X_STRIDE,
        ROW_PARTIAL_STRIDE=MAX_COLUMN_TILES,
        COLUMN_TILES=column_tiles,
        PARTIAL_BLOCK=partial_block,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        **launch_options,
    )


def run_split(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_n: int = 8,
    block_d: int = 1024,
    partial_warps: int = 4,
    partial_stages: int = 4,
    partial_maxnreg: int | None = None,
    finish_warps: int = 4,
    finish_stages: int = 4,
    finish_maxnreg: int | None = None,
    dweight_block_n: int = 128,
    dweight_block_d: int = 64,
    dweight_warps: int = 4,
    dweight_stages: int = 3,
    dweight_maxnreg: int | None = None,
) -> None:
    row_blocks = triton.cdiv(N, block_n)
    run_partial(
        inputs,
        outputs,
        block_n=block_n,
        block_d=block_d,
        partial_warps=partial_warps,
        partial_stages=partial_stages,
        partial_maxnreg=partial_maxnreg,
    )
    run_dx_finish(
        inputs,
        outputs,
        block_n=block_n,
        block_d=block_d,
        finish_warps=finish_warps,
        finish_stages=finish_stages,
        finish_maxnreg=finish_maxnreg,
    )
    run_finalizer(
        outputs,
        partitions=row_blocks,
        block_n=dweight_block_n,
        block_d=dweight_block_d,
        num_warps=dweight_warps,
        num_stages=dweight_stages,
        maxnreg=dweight_maxnreg,
    )


def run_rowwise(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_n: int = 4,
    block_d: int = 2048,
    num_warps: int = 8,
    num_stages: int = 3,
    maxnreg: int | None = None,
    dweight_block_n: int = 256,
    dweight_block_d: int = 16,
    dweight_warps: int = 16,
    dweight_stages: int = 3,
) -> None:
    row_blocks = triton.cdiv(N, block_n)
    launch_options: dict[str, int] = {
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    cast(Any, rmsnorm_backward_rowwise)[(row_blocks,)](
        inputs["x"],
        inputs["dy"],
        inputs["weight"],
        inputs["rstd"],
        outputs["dx"],
        outputs["dweight_partial"],
        N=N,
        D=D,
        X_STRIDE=X_STRIDE,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        **launch_options,
    )
    run_finalizer(
        outputs,
        partitions=row_blocks,
        block_n=dweight_block_n,
        block_d=dweight_block_d,
        num_warps=dweight_warps,
        num_stages=dweight_stages,
    )


def run_fused(inputs: TensorMap, outputs: TensorMap) -> None:
    run_rowwise(inputs, outputs)


def accuracy(reference: TensorMap, candidate: TensorMap) -> dict[str, Any]:
    result = {}
    for name in ("dx", "dweight"):
        expected = reference[name].float()
        difference = candidate[name].float() - expected
        result[name] = {
            "allclose": bool(
                torch.allclose(candidate[name], reference[name], atol=0.02, rtol=0.02)
            ),
            "max_abs": float(difference.abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(expected).clamp_min(1e-30)
            ),
            "exact": bool(torch.equal(reference[name], candidate[name])),
        }
    return result


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--verification-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    accuracy_by_seed = {}
    for seed in dict.fromkeys(args.verification_seeds):
        inputs = make_inputs(seed)
        reference = make_outputs()
        candidate = make_outputs()
        run_unfused(inputs, reference)
        run_fused(inputs, candidate)
        torch.cuda.synchronize()
        accuracy_by_seed[str(seed)] = accuracy(reference, candidate)
    passed = all(
        bool(metric["allclose"])
        for seed_result in accuracy_by_seed.values()
        for metric in seed_result.values()
    )
    report: dict[str, Any] = {
        "task": "T289881412",
        "shape_nd": [N, D],
        "x_stride": X_STRIDE,
        "historical_fbsource_us": {
            "source": 2117.44,
            "previous_candidate": 195.50,
        },
        "winner_config": {
            "BLOCK_N": 4,
            "BLOCK_D": 2048,
            "num_warps": 8,
            "num_stages": 3,
            "maxnreg": None,
            "dweight_BLOCK_N": 256,
            "dweight_BLOCK_D": 16,
            "dweight_num_warps": 16,
            "scratch_partitions": 1280,
        },
        "numerical_contract": {
            "reduction_reassociation": "allowed",
            "input_dtype": "bfloat16",
            "accumulator_dtype": "float32",
            "output_dtype": "bfloat16",
            "atol": 0.02,
            "rtol": 0.02,
        },
        "accuracy": accuracy_by_seed,
        "passed": passed,
    }
    if passed and not args.check_only:
        inputs = make_inputs(0)
        reference = make_outputs()
        candidate = make_outputs()
        report["timings_ms"] = {
            "unfused": benchmark(
                lambda: run_unfused(inputs, reference),
                warmup=args.warmup,
                samples=args.samples,
                reps=args.reps,
            ),
            "fused": benchmark(
                lambda: run_fused(inputs, candidate),
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
