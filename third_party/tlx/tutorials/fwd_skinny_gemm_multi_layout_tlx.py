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

"""TLX implementation of the T289757878 GEMM multi-layout fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, cast

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

import fwd_skinny_gemm_multi_layout as baseline


@triton.jit
def skinny_gemm_multi_layout_tlx(
    a_desc,
    b,
    destination0,
    destination1,
    destination2,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_GROUPS: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    PRODUCER_WARPS: tl.constexpr,
    PRODUCER_REGS: tl.constexpr,
    EPILOGUE_WARPS: tl.constexpr,
    EPILOGUE_REGS: tl.constexpr,
):
    buffers_a = tlx.local_alloc(
        (BLOCK_M, BLOCK_K),
        tl.bfloat16,
        M_GROUPS * NUM_K_BLOCKS,
    )
    buffers_b = tlx.local_alloc(
        (BLOCK_N, BLOCK_K),
        tl.bfloat16,
        NUM_K_BLOCKS,
    )
    accumulators = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        M_GROUPS * NUM_TMEM_BUFFERS,
        tlx.storage_kind.tmem,
    )
    a_full = tlx.alloc_barriers(M_GROUPS * NUM_K_BLOCKS, arrive_count=1)
    a_empty = tlx.alloc_barriers(M_GROUPS * NUM_K_BLOCKS, arrive_count=1)
    b_full = tlx.alloc_barriers(NUM_K_BLOCKS, arrive_count=1)
    accumulator_full = tlx.alloc_barriers(M_GROUPS * NUM_TMEM_BUFFERS, arrive_count=1)
    accumulator_empty = tlx.alloc_barriers(M_GROUPS * NUM_TMEM_BUFFERS, arrive_count=1)

    tile_rows: tl.constexpr = BLOCK_M * M_GROUPS
    num_tiles: tl.constexpr = M // tile_rows
    start_tile = tl.program_id(0)

    with tlx.async_tasks():
        with tlx.async_task(
            "default", num_warps=EPILOGUE_WARPS, num_regs=EPILOGUE_REGS
        ):
            tile_count = 0
            for tile in range(start_tile, num_tiles, NUM_PROGRAMS):
                tmem_buf, tmem_phase = get_bufidx_phase(tile_count, NUM_TMEM_BUFFERS)
                for group in tl.static_range(M_GROUPS):
                    accumulator_index = group * NUM_TMEM_BUFFERS + tmem_buf
                    tlx.barrier_wait(accumulator_full[accumulator_index], tmem_phase)
                    result = tlx.local_load(accumulators[accumulator_index])
                    tlx.barrier_arrive(accumulator_empty[accumulator_index], 1)
                    rows = tile * tile_rows + group * BLOCK_M + tl.arange(0, BLOCK_M)
                    columns = tl.arange(0, BLOCK_N)
                    direct_offsets = rows[:, None] * N + columns[None, :]
                    transposed_offsets = (
                        (rows[:, None] // GROUP_ROWS) * (N * GROUP_ROWS)
                        + columns[None, :] * GROUP_ROWS
                        + rows[:, None] % GROUP_ROWS
                    )
                    value = result.to(tl.bfloat16)
                    tl.store(destination0 + direct_offsets, value)
                    tl.store(destination1 + direct_offsets, value)
                    tl.store(destination2 + transposed_offsets, value)
                tile_count += 1

        with tlx.async_task(num_warps=1, num_regs=24):
            for k_block in tl.static_range(NUM_K_BLOCKS):
                tlx.barrier_wait(b_full[k_block], 0)
            tile_count = 0
            for _tile in range(start_tile, num_tiles, NUM_PROGRAMS):
                tmem_buf, tmem_phase = get_bufidx_phase(tile_count, NUM_TMEM_BUFFERS)
                for group in tl.static_range(M_GROUPS):
                    accumulator_index = group * NUM_TMEM_BUFFERS + tmem_buf
                    tlx.barrier_wait(
                        accumulator_empty[accumulator_index], tmem_phase ^ 1
                    )
                    for k_block in tl.static_range(NUM_K_BLOCKS):
                        a_index = group * NUM_K_BLOCKS + k_block
                        tlx.barrier_wait(a_full[a_index], tile_count & 1)
                        tlx.async_dot(
                            buffers_a[a_index],
                            tlx.local_trans(buffers_b[k_block]),
                            accumulators[accumulator_index],
                            use_acc=k_block > 0,
                            mBarriers=[a_empty[a_index]],
                            out_dtype=tl.float32,
                        )
                    tlx.barrier_wait(a_empty[a_index], tile_count & 1)
                    tlx.barrier_arrive(accumulator_full[accumulator_index], 1)
                tile_count += 1

        with tlx.async_task(num_warps=PRODUCER_WARPS, num_regs=PRODUCER_REGS):
            columns = tl.arange(0, BLOCK_N)
            ks = tl.arange(0, BLOCK_K)
            for k_block in tl.static_range(NUM_K_BLOCKS):
                b_ks = k_block * BLOCK_K + ks
                b_value = tl.load(
                    b + columns[:, None] * K + b_ks[None, :],
                    mask=b_ks[None, :] < K,
                    other=0.0,
                )
                tlx.local_store(buffers_b[k_block], b_value)
                tlx.fence_async_shared()
                tlx.barrier_arrive(b_full[k_block], 1)

            tile_count = 0
            for tile in range(start_tile, num_tiles, NUM_PROGRAMS):
                for group in tl.static_range(M_GROUPS):
                    for k_block in tl.static_range(NUM_K_BLOCKS):
                        a_index = group * NUM_K_BLOCKS + k_block
                        tlx.barrier_wait(a_empty[a_index], (tile_count & 1) ^ 1)
                        tlx.barrier_expect_bytes(a_full[a_index], 2 * BLOCK_M * BLOCK_K)
                        tlx.async_descriptor_load(
                            a_desc,
                            buffers_a[a_index],
                            [
                                tile * tile_rows + group * BLOCK_M,
                                k_block * BLOCK_K,
                            ],
                            a_full[a_index],
                        )
                tile_count += 1


CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 64,
    "BLOCK_K": 64,
    "M_GROUPS": 2,
    "NUM_K_BLOCKS": 2,
    "NUM_TMEM_BUFFERS": 2,
    "CTAS_PER_SM": 1,
    "PRODUCER_WARPS": 8,
    "PRODUCER_REGS": 104,
    "EPILOGUE_WARPS": 4,
    "EPILOGUE_REGS": 96,
}


def run_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    *,
    config: dict[str, int] | None = None,
) -> None:
    config = CONFIG if config is None else config
    num_programs = min(
        baseline.M // (config["BLOCK_M"] * config["M_GROUPS"]),
        torch.cuda.get_device_properties(0).multi_processor_count
        * config["CTAS_PER_SM"],
    )
    a_desc = TensorDescriptor(
        inputs["a"],
        inputs["a"].shape,
        inputs["a"].stride(),
        [config["BLOCK_M"], config["BLOCK_K"]],
    )
    launch_config = dict(config)
    del launch_config["CTAS_PER_SM"]
    cast(Any, skinny_gemm_multi_layout_tlx)[(num_programs,)](
        a_desc,
        inputs["b"],
        outputs["layout0"],
        outputs["layout1"],
        outputs["layout2"],
        M=baseline.M,
        N=baseline.N,
        K=baseline.K,
        GROUP_ROWS=baseline.GROUP_ROWS,
        **launch_config,
        NUM_PROGRAMS=num_programs,
        num_warps=4,
        num_stages=1,
    )


def _measure(
    functions: dict[str, Callable[[], None]],
    *,
    warmup: int,
    samples: int,
    reps: int,
) -> dict[str, list[float]]:
    result = {name: [] for name in functions}
    for repetition in range(reps):
        for _ in range(warmup):
            for function in functions.values():
                function()
        torch.cuda.synchronize()
        values = {name: [] for name in functions}
        for sample in range(samples):
            ordered = list(functions.items())
            if (repetition + sample) % 2:
                ordered.reverse()
            for name, function in ordered:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function()
                end.record()
                end.synchronize()
                values[name].append(start.elapsed_time(end))
        for name in result:
            result[name].append(float(torch.tensor(values[name]).median()))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--verification-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")

    accuracy_by_seed: dict[str, dict[str, dict[str, dict[str, float | bool]]]] = {}
    for seed_value in dict.fromkeys(args.verification_seeds):
        inputs = baseline.make_inputs(seed_value)
        reference = baseline.make_outputs()
        seed = baseline.make_outputs()
        fused = baseline.make_outputs()
        tlx_output = baseline.make_outputs()
        baseline.run_unfused(inputs, reference)
        baseline.run_seed(inputs, seed)
        baseline.run_row_major_persistent(inputs, fused)
        run_tlx(inputs, tlx_output)
        torch.cuda.synchronize()
        accuracy_by_seed[str(seed_value)] = {
            "seed": baseline.accuracy(reference, seed),
            "fused": baseline.accuracy(reference, fused),
            "tlx": baseline.accuracy(reference, tlx_output),
        }
    passed = all(
        bool(metric["allclose"])
        for seed_metrics in accuracy_by_seed.values()
        for variant_metrics in seed_metrics.values()
        for metric in variant_metrics.values()
    )
    report: dict[str, Any] = {
        "task": "T289757878",
        "shape_mnk": [baseline.M, baseline.N, baseline.K],
        "outputs": [
            [baseline.M, baseline.N],
            [baseline.M, baseline.N],
            [5120, 64, 256],
        ],
        "historical_fbsource_us": {"unfused": 174.69, "fused_seed": 782.35},
        "winner_config": {
            "BLOCK_ROWS": 256,
            "NUM_PROGRAMS": "1 * SMs",
            "num_warps": 8,
            "num_stages": 3,
            "maxnreg": None,
        },
        "tlx_config": CONFIG,
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
    if not args.check_only and passed:
        inputs = baseline.make_inputs(0)
        unfused_output = baseline.make_outputs()
        fused_output = baseline.make_outputs()
        seed_output = baseline.make_outputs()
        tlx_output = baseline.make_outputs()
        report["primary_timings"] = _measure(
            {
                "unfused_ms": lambda: baseline.run_unfused(inputs, unfused_output),
                "fused_ms": lambda: baseline.run_row_major_persistent(
                    inputs, fused_output
                ),
            },
            warmup=args.warmup,
            samples=args.samples,
            reps=args.reps,
        )
        report["diagnostic_timings"] = _measure(
            {
                "seed_ms": lambda: baseline.run_seed(inputs, seed_output),
                "tlx_ms": lambda: run_tlx(inputs, tlx_output),
            },
            warmup=args.warmup,
            samples=args.samples,
            reps=args.reps,
        )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
