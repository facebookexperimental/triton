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

"""TLX down-GEMM with fused residual, weighted RMSNorm, and direct stores."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, cast

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

import fwd_gemm_rmsnorm_direct_store as baseline


@triton.jit
def _epilogue_part(
    part,
    accumulator_tmem,
    accumulator_full,
    accumulator_empty,
    pre_norm_smem,
    row_sum_smem,
    row_sum_full,
    residual_ptr,
    norm_weight_ptr,
    post_add_ptr,
    rstd_ptr,
    destination0_ptr,
    destination1_ptr,
    start_m,
    start_n,
    pid_n,
    BRANCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPILOGUE_PARTS: tl.constexpr,
    EPILOGUE_BLOCK_N: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    NORM_WIDTH: tl.constexpr,
    ROWS_PER_GEMM_ROW: tl.constexpr,
    EPS: tl.constexpr,
):
    part_n: tl.constexpr = BLOCK_N // EPILOGUE_PARTS
    rows = start_m + tl.arange(0, BLOCK_M)
    tlx.barrier_wait(accumulator_full[0], 0)
    local_row_sum = tl.zeros((BLOCK_M,), tl.float32)
    for n_offset in tl.static_range(0, part_n, EPILOGUE_BLOCK_N):
        local_columns = part * part_n + n_offset + tl.arange(0, EPILOGUE_BLOCK_N)
        columns = start_n + local_columns
        offsets = rows[:, None] * N + columns[None, :]
        accumulator_slice = tlx.local_slice(
            accumulator_tmem[0],
            [0, part * part_n + n_offset],
            [BLOCK_M, EPILOGUE_BLOCK_N],
        )
        pre_norm_slice = tlx.local_slice(
            pre_norm_smem[0],
            [0, part * part_n + n_offset],
            [BLOCK_M, EPILOGUE_BLOCK_N],
        )
        accumulator = tlx.local_load(accumulator_slice).to(tl.float32)
        residual = tl.load(residual_ptr + offsets).to(tl.float32)
        gemm_value = accumulator.to(tl.bfloat16).to(tl.float32)
        post_add = (residual + gemm_value).to(tl.bfloat16)
        tlx.local_store(pre_norm_slice, post_add)
        tl.store(post_add_ptr + offsets, post_add)
        post_add_fp32 = post_add.to(tl.float32)
        local_row_sum += tl.sum(post_add_fp32 * post_add_fp32, axis=1)
    if EPILOGUE_PARTS == 1:
        tlx.fence_async_shared()
        tlx.barrier_arrive(accumulator_empty[0], 1)
        rstd = 1.0 / tl.sqrt(local_row_sum / NORM_WIDTH + EPS)
    else:
        tlx.local_store(row_sum_smem[part], tl.expand_dims(local_row_sum, 1))
        tlx.fence_async_shared()
        tlx.barrier_arrive(accumulator_empty[0], 1)
        tlx.barrier_arrive(row_sum_full[0], 1)
        tlx.barrier_wait(row_sum_full[0], 0)
        row_sum = tl.zeros((BLOCK_M, 1), tl.float32)
        for index in tl.static_range(EPILOGUE_PARTS):
            row_sum += tlx.local_load(row_sum_smem[index])
        rstd_2d = 1.0 / tl.sqrt(row_sum / NORM_WIDTH + EPS)
        rstd = tl.reshape(rstd_2d, (BLOCK_M,))
    norm_rows = rows * ROWS_PER_GEMM_ROW + pid_n
    if part == 0:
        tl.store(rstd_ptr + norm_rows, rstd)
    for n_offset in tl.static_range(0, part_n, EPILOGUE_BLOCK_N):
        local_columns = part * part_n + n_offset + tl.arange(0, EPILOGUE_BLOCK_N)
        columns = start_n + local_columns
        pre_norm_slice = tlx.local_slice(
            pre_norm_smem[0],
            [0, part * part_n + n_offset],
            [BLOCK_M, EPILOGUE_BLOCK_N],
        )
        pre_norm = tlx.local_load(pre_norm_slice).to(tl.float32)
        norm_weight = tl.load(norm_weight_ptr + local_columns).to(tl.float32)
        normalized = (pre_norm * rstd[:, None] * norm_weight[None, :]).to(tl.bfloat16)
        destination0_offsets = rows[:, None] * (2 * N) + columns[None, :] + BRANCH * N
        destination1_offsets = (
            rows[:, None] * (112 * NORM_WIDTH)
            + columns[None, :]
            + 64 * NORM_WIDTH
            + BRANCH * N
        )
        tl.store(destination0_ptr + destination0_offsets, normalized)
        tl.store(destination1_ptr + destination1_offsets, normalized)


@triton.jit
def gemm_rmsnorm_direct_store_tlx(
    a_desc,
    b_desc,
    residual_ptr,
    norm_weight_ptr,
    post_add_ptr,
    rstd_ptr,
    destination0_ptr,
    destination1_ptr,
    BRANCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    EPILOGUE_PARTS: tl.constexpr,
    EPILOGUE_BLOCK_N: tl.constexpr,
    EPILOGUE_WARPS: tl.constexpr,
    EPILOGUE_REGS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NORM_WIDTH: tl.constexpr,
    ROWS_PER_GEMM_ROW: tl.constexpr,
    EPS: tl.constexpr,
):
    a_smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, NUM_SMEM_BUFFERS)
    b_smem = tlx.local_alloc((BLOCK_N, BLOCK_K), tl.bfloat16, NUM_SMEM_BUFFERS)
    accumulator_tmem = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, 1, tlx.storage_kind.tmem
    )
    pre_norm_smem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.bfloat16, 1, reuse=b_smem)
    row_sum_smem = tlx.local_alloc((BLOCK_M, 1), tl.float32, EPILOGUE_PARTS)
    operands_full = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    b_empty = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    accumulator_full = tlx.alloc_barriers(1, arrive_count=1)
    accumulator_empty = tlx.alloc_barriers(1, arrive_count=EPILOGUE_PARTS)
    row_sum_full = tlx.alloc_barriers(1, arrive_count=EPILOGUE_PARTS)

    tile_id = tl.program_id(0)
    num_pid_m: tl.constexpr = M // BLOCK_M
    pid_n = tile_id // num_pid_m
    pid_m = tile_id % num_pid_m
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    k_tiles: tl.constexpr = K // BLOCK_K

    with tlx.async_tasks():
        with tlx.async_task(
            "default",
            num_regs=EPILOGUE_REGS if EPILOGUE_REGS > 0 else None,
        ):
            _epilogue_part(
                0,
                accumulator_tmem,
                accumulator_full,
                accumulator_empty,
                pre_norm_smem,
                row_sum_smem,
                row_sum_full,
                residual_ptr,
                norm_weight_ptr,
                post_add_ptr,
                rstd_ptr,
                destination0_ptr,
                destination1_ptr,
                start_m,
                start_n,
                pid_n,
                BRANCH,
                BLOCK_M,
                BLOCK_N,
                EPILOGUE_PARTS,
                EPILOGUE_BLOCK_N,
                M,
                N,
                NORM_WIDTH,
                ROWS_PER_GEMM_ROW,
                EPS,
            )

        if EPILOGUE_PARTS == 2:
            with tlx.async_task(num_warps=EPILOGUE_WARPS, num_regs=EPILOGUE_REGS):
                _epilogue_part(
                    1,
                    accumulator_tmem,
                    accumulator_full,
                    accumulator_empty,
                    pre_norm_smem,
                    row_sum_smem,
                    row_sum_full,
                    residual_ptr,
                    norm_weight_ptr,
                    post_add_ptr,
                    rstd_ptr,
                    destination0_ptr,
                    destination1_ptr,
                    start_m,
                    start_n,
                    pid_n,
                    BRANCH,
                    BLOCK_M,
                    BLOCK_N,
                    EPILOGUE_PARTS,
                    EPILOGUE_BLOCK_N,
                    M,
                    N,
                    NORM_WIDTH,
                    ROWS_PER_GEMM_ROW,
                    EPS,
                )

        with tlx.async_task(num_warps=1, num_regs=24):
            tlx.barrier_wait(accumulator_empty[0], 1)
            for k_tile in tl.static_range(k_tiles):
                buffer_index, phase = get_bufidx_phase(k_tile, NUM_SMEM_BUFFERS)
                tlx.barrier_wait(operands_full[buffer_index], phase)
                tlx.async_dot(
                    a_smem[buffer_index],
                    tlx.local_trans(b_smem[buffer_index]),
                    accumulator_tmem[0],
                    use_acc=k_tile > 0,
                    mBarriers=[a_empty[buffer_index], b_empty[buffer_index]],
                    out_dtype=tl.float32,
                )
            tlx.tcgen05_commit(accumulator_full[0])

        with tlx.async_task(num_warps=1, num_regs=24):
            for k_tile in tl.static_range(k_tiles):
                buffer_index, phase = get_bufidx_phase(k_tile, NUM_SMEM_BUFFERS)
                tlx.barrier_wait(a_empty[buffer_index], phase ^ 1)
                tlx.barrier_wait(b_empty[buffer_index], phase ^ 1)
                tlx.barrier_expect_bytes(
                    operands_full[buffer_index],
                    2 * (BLOCK_M + BLOCK_N) * BLOCK_K,
                )
                tlx.async_descriptor_load(
                    a_desc,
                    a_smem[buffer_index],
                    [start_m, k_tile * BLOCK_K],
                    operands_full[buffer_index],
                )
                tlx.async_descriptor_load(
                    b_desc,
                    b_smem[buffer_index],
                    [start_n, k_tile * BLOCK_K],
                    operands_full[buffer_index],
                )


SEED_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": baseline.NORM_WIDTH,
    "BLOCK_K": 64,
    "NUM_SMEM_BUFFERS": 4,
    "EPILOGUE_PARTS": 1,
    "EPILOGUE_BLOCK_N": 256,
    "EPILOGUE_WARPS": 8,
    "EPILOGUE_REGS": 0,
}

CONFIG = dict(SEED_CONFIG)


_DESCRIPTORS: dict[tuple[str, int, int, int], tuple[torch.Tensor, TensorDescriptor]] = (
    {}
)


def _descriptor(
    name: str, branch: int, tensor: torch.Tensor, block: tuple[int, int]
) -> TensorDescriptor:
    key = (name, branch, block[0], block[1])
    cached = _DESCRIPTORS.get(key)
    if cached is None or cached[0] is not tensor:
        descriptor = TensorDescriptor(
            tensor, list(tensor.shape), list(tensor.stride()), list(block)
        )
        _DESCRIPTORS[key] = (tensor, descriptor)
        return descriptor
    return cached[1]


def run_tlx(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    *,
    config: dict[str, int] | None = None,
) -> None:
    config = CONFIG if config is None else config
    grid = (
        triton.cdiv(baseline.M, config["BLOCK_M"])
        * triton.cdiv(baseline.N, config["BLOCK_N"]),
    )
    kernel = cast(Any, gemm_rmsnorm_direct_store_tlx)
    for branch in range(2):
        a_desc = _descriptor(
            "a",
            branch,
            inputs["a"][branch],
            (config["BLOCK_M"], config["BLOCK_K"]),
        )
        b_desc = _descriptor(
            "b",
            branch,
            inputs["b"][branch],
            (config["BLOCK_N"], config["BLOCK_K"]),
        )
        launch_warps = 8 if config["EPILOGUE_PARTS"] == 1 else 4
        kernel[grid](
            a_desc,
            b_desc,
            inputs["residual"][branch],
            inputs["weight"][branch],
            outputs["post_add"][branch],
            outputs["rstd"][branch],
            outputs["destination0"],
            outputs["destination1"],
            BRANCH=branch,
            M=baseline.M,
            N=baseline.N,
            K=baseline.K,
            NORM_WIDTH=baseline.NORM_WIDTH,
            ROWS_PER_GEMM_ROW=baseline.ROWS_PER_GEMM_ROW,
            EPS=baseline.EPS,
            **config,
            num_warps=launch_warps,
            num_stages=1,
        )


def _time_once(call: Any) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def _measure(
    calls: dict[str, Any], warmup: int, samples: int, reps: int
) -> dict[str, list[float]]:
    for _ in range(warmup):
        for call in calls.values():
            call()
    torch.cuda.synchronize()
    result = {name: [] for name in calls}
    for rep in range(reps):
        samples_by_name = {name: [] for name in calls}
        names = list(calls)
        for sample in range(samples):
            if (rep + sample) % 2:
                names.reverse()
            for name in names:
                samples_by_name[name].append(_time_once(calls[name]))
        for name in calls:
            result[name].append(statistics.median(samples_by_name[name]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--verification-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--block-m", type=int, default=CONFIG["BLOCK_M"])
    parser.add_argument("--block-k", type=int, default=CONFIG["BLOCK_K"])
    parser.add_argument("--buffers", type=int, default=CONFIG["NUM_SMEM_BUFFERS"])
    parser.add_argument(
        "--epilogue-parts", type=int, choices=(1, 2), default=CONFIG["EPILOGUE_PARTS"]
    )
    parser.add_argument(
        "--epilogue-block-n", type=int, choices=(64, 128, 256), default=256
    )
    parser.add_argument("--epilogue-regs", type=int, default=0)
    args = parser.parse_args()

    config = {
        **CONFIG,
        "BLOCK_M": args.block_m,
        "BLOCK_K": args.block_k,
        "NUM_SMEM_BUFFERS": args.buffers,
        "EPILOGUE_PARTS": args.epilogue_parts,
        "EPILOGUE_BLOCK_N": args.epilogue_block_n,
        "EPILOGUE_WARPS": 8 if args.epilogue_parts == 1 else 4,
        "EPILOGUE_REGS": args.epilogue_regs or (128 if args.epilogue_parts == 2 else 0),
    }
    if baseline.M % config["BLOCK_M"] or baseline.K % config["BLOCK_K"]:
        raise ValueError("BLOCK_M and BLOCK_K must divide the fixed GEMM shape")
    part_n = config["BLOCK_N"] // config["EPILOGUE_PARTS"]
    if part_n % config["EPILOGUE_BLOCK_N"]:
        raise ValueError("EPILOGUE_BLOCK_N must divide each epilogue partition")

    accuracy_by_seed = {}
    for seed in dict.fromkeys(args.verification_seeds):
        inputs = baseline.make_inputs(seed)
        expected = baseline.make_outputs()
        actual = baseline.make_outputs()
        baseline.run_unfused(inputs, expected)
        run_tlx(inputs, actual, config=config)
        torch.cuda.synchronize()
        accuracy_by_seed[str(seed)] = baseline.accuracy(expected, actual)
    passed = all(
        metric["passed"]
        for seed_metrics in accuracy_by_seed.values()
        for metric in seed_metrics.values()
    )
    report: dict[str, Any] = {
        "shape_mnk": [baseline.M, baseline.N, baseline.K],
        "seed_config": SEED_CONFIG,
        "config": config,
        "accuracy": accuracy_by_seed,
        "passed": passed,
    }
    if not args.check_only:
        inputs = baseline.make_inputs(0)
        outputs = {name: baseline.make_outputs() for name in ("unfused", "seed", "tlx")}
        report["timings_us"] = _measure(
            {
                "unfused": lambda: baseline.run_unfused(inputs, outputs["unfused"]),
                "seed": lambda: run_tlx(inputs, outputs["seed"], config=SEED_CONFIG),
                "tlx": lambda: run_tlx(inputs, outputs["tlx"], config=config),
            },
            args.warmup,
            args.samples,
            args.reps,
        )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
