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

"""TLX implementation of the T289757919 batched GEMM-concat fusion."""

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

import fwd_batched_gemm_concat as baseline


@triton.jit
def batched_gemm_concat_tlx(
    a_desc,
    b_desc,
    features,
    extra,
    output,
    padded_output,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FEATURE_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    PRODUCER_WARPS: tl.constexpr,
    PRODUCER_REGS: tl.constexpr,
    EPILOGUE_WARPS: tl.constexpr,
    EPILOGUE_REGS: tl.constexpr,
):
    buffers_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, NUM_SMEM_BUFFERS)
    buffers_b = tlx.local_alloc((BLOCK_K, N), tl.bfloat16, NUM_SMEM_BUFFERS)
    accumulator = tlx.local_alloc(
        (BLOCK_M, N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem
    )
    operand_full = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    operand_empty = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    accumulator_full = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    accumulator_empty = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)

    start_batch = tl.program_id(0)

    with tlx.async_tasks():
        with tlx.async_task(
            "default", num_warps=EPILOGUE_WARPS, num_regs=EPILOGUE_REGS
        ):
            tile_count = 0
            for batch in range(start_batch, BATCH, NUM_PROGRAMS):
                rows = tl.arange(0, M)
                columns = tl.arange(0, N)
                channel = rows[:, None] * N + columns[None, :]
                base = batch * OUTPUT_WIDTH
                padded_base = batch * PADDED_WIDTH
                # This copy is independent of the contraction and executes while
                # the producer and tensor-core tasks build the projection.
                feature0 = tl.load(features + batch * 20480 + 16384 + channel)
                feature1 = tl.load(features + batch * 20480 + 18432 + channel)
                extra0 = tl.load(extra + batch * EXTRA_WIDTH + channel)
                extra1_offsets = tl.arange(0, 512)
                extra1_mask = extra1_offsets < EXTRA_WIDTH - M * N
                extra1 = tl.load(
                    extra + batch * EXTRA_WIDTH + M * N + extra1_offsets,
                    mask=extra1_mask,
                    other=0.0,
                )
                tl.store(output + base + M * N + channel, feature0)
                tl.store(output + base + 2 * M * N + channel, feature1)
                tl.store(output + base + M * N + FEATURE_WIDTH + channel, extra0)
                tl.store(
                    output + base + 2 * M * N + FEATURE_WIDTH + extra1_offsets,
                    extra1,
                    mask=extra1_mask,
                )
                tl.store(padded_output + padded_base + M * N + channel, feature0)
                tl.store(padded_output + padded_base + 2 * M * N + channel, feature1)
                tl.store(
                    padded_output + padded_base + M * N + FEATURE_WIDTH + channel,
                    extra0,
                )
                tl.store(
                    padded_output
                    + padded_base
                    + 2 * M * N
                    + FEATURE_WIDTH
                    + extra1_offsets,
                    extra1,
                    mask=extra1_mask,
                )

                tmem_buf, tmem_phase = get_bufidx_phase(tile_count, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(accumulator_full[tmem_buf], tmem_phase)
                projection = tlx.local_load(accumulator[tmem_buf]).to(tl.bfloat16)
                tlx.barrier_arrive(accumulator_empty[tmem_buf], 1)
                projection_rows = tl.arange(0, BLOCK_M)
                projection_channel = projection_rows[:, None] * N + columns[None, :]
                projection_mask = projection_rows[:, None] < M
                tl.store(
                    output + base + projection_channel,
                    projection,
                    mask=projection_mask,
                )
                tl.store(
                    padded_output + padded_base + projection_channel,
                    projection,
                    mask=projection_mask,
                )
                tile_count += 1

        with tlx.async_task(num_warps=1, num_regs=24):
            smem_count = 0
            tile_count = 0
            for _batch in range(start_batch, BATCH, NUM_PROGRAMS):
                tmem_buf, tmem_phase = get_bufidx_phase(tile_count, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(accumulator_empty[tmem_buf], tmem_phase ^ 1)
                for k_block in tl.static_range(NUM_K_BLOCKS):
                    smem_buf, smem_phase = get_bufidx_phase(
                        smem_count, NUM_SMEM_BUFFERS
                    )
                    tlx.barrier_wait(operand_full[smem_buf], smem_phase)
                    tlx.async_dot(
                        buffers_a[smem_buf],
                        buffers_b[smem_buf],
                        accumulator[tmem_buf],
                        use_acc=k_block > 0,
                        mBarriers=[operand_empty[smem_buf]],
                        out_dtype=tl.float32,
                    )
                    smem_count += 1
                last_buf, last_phase = get_bufidx_phase(
                    smem_count - 1, NUM_SMEM_BUFFERS
                )
                tlx.barrier_wait(operand_empty[last_buf], last_phase)
                tlx.barrier_arrive(accumulator_full[tmem_buf], 1)
                tile_count += 1

        with tlx.async_task(num_warps=PRODUCER_WARPS, num_regs=PRODUCER_REGS):
            smem_count = 0
            for batch in range(start_batch, BATCH, NUM_PROGRAMS):
                for k_block in tl.static_range(NUM_K_BLOCKS):
                    smem_buf, smem_phase = get_bufidx_phase(
                        smem_count, NUM_SMEM_BUFFERS
                    )
                    tlx.barrier_wait(operand_empty[smem_buf], smem_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        operand_full[smem_buf],
                        2 * (BLOCK_M + N) * BLOCK_K,
                    )
                    tlx.async_descriptor_load(
                        a_desc,
                        buffers_a[smem_buf],
                        [batch * M, k_block * BLOCK_K],
                        operand_full[smem_buf],
                    )
                    tlx.async_descriptor_load(
                        b_desc,
                        buffers_b[smem_buf],
                        [batch * K + k_block * BLOCK_K, 0],
                        operand_full[smem_buf],
                    )
                    smem_count += 1


CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_K": 128,
    "NUM_K_BLOCKS": 2,
    "NUM_SMEM_BUFFERS": 4,
    "NUM_TMEM_BUFFERS": 1,
    "CTAS_PER_SM": 1,
    "PRODUCER_WARPS": 1,
    "PRODUCER_REGS": 24,
    "EPILOGUE_WARPS": 8,
    "EPILOGUE_REGS": 64,
}


def run_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    *,
    config: dict[str, int] | None = None,
) -> None:
    config = CONFIG if config is None else config
    num_programs = min(
        baseline.BATCH,
        torch.cuda.get_device_properties(0).multi_processor_count
        * config["CTAS_PER_SM"],
    )
    launch_config = dict(config)
    del launch_config["CTAS_PER_SM"]
    block_k = config["BLOCK_K"]
    a_2d = inputs["a"].view(baseline.BATCH * baseline.M, baseline.K)
    b_2d = inputs["b"].view(baseline.BATCH * baseline.K, baseline.N)
    a_desc = TensorDescriptor(
        a_2d,
        a_2d.shape,
        a_2d.stride(),
        [config["BLOCK_M"], block_k],
    )
    b_desc = TensorDescriptor(
        b_2d,
        b_2d.shape,
        b_2d.stride(),
        [block_k, baseline.N],
    )
    cast(Any, batched_gemm_concat_tlx)[(num_programs,)](
        a_desc,
        b_desc,
        inputs["features"],
        inputs["extra"],
        outputs["output"],
        outputs["padded"],
        BATCH=baseline.BATCH,
        M=baseline.M,
        N=baseline.N,
        K=baseline.K,
        FEATURE_WIDTH=baseline.FEATURE_WIDTH,
        EXTRA_WIDTH=baseline.EXTRA_WIDTH,
        OUTPUT_WIDTH=baseline.OUTPUT_WIDTH,
        PADDED_WIDTH=baseline.PADDED_WIDTH,
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
        ordered = list(functions.items())
        if repetition % 2:
            ordered.reverse()
        for name, function in ordered:
            for _ in range(warmup):
                function()
            torch.cuda.synchronize()
            values = []
            for _ in range(samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
            result[name].append(float(torch.tensor(values).median()))
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
        baseline.run_direct_batch(inputs, fused)
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
        "task": "T289757919",
        "shape_bmnk": [baseline.BATCH, baseline.M, baseline.N, baseline.K],
        "historical_fbsource_us": {"unfused": 100.46, "fused_seed": 500.16},
        "winner_config": {
            "BLOCK_M": baseline.M,
            "BLOCK_N": baseline.N,
            "BLOCK_K": 128,
            "copy_order": "before_gemm",
            "num_warps": 8,
            "num_stages": 2,
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
                "fused_ms": lambda: baseline.run_direct_batch(inputs, fused_output),
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
