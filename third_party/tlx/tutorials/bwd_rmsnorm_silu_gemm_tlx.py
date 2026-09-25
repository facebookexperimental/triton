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

"""TLX GEMM with a fused D=512 RMSNorm/SiLU backward epilogue."""

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

import bwd_rmsnorm_silu_gemm as baseline


@triton.jit
def _rmsnorm_silu_epilogue_part(
    part,
    accumulator,
    accumulator_full,
    row_dot_smem,
    row_dot_full,
    saved_input,
    rstd_input,
    weight,
    gemm_output,
    dx_output,
    dweight_partial,
    row_start,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = row_start + tl.arange(0, BLOCK_M)
    columns = part * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + columns[None, :]
    rstd = tl.load(rstd_input + rows).to(tl.float32)
    scale = tl.load(weight + columns).to(tl.float32)
    tlx.barrier_wait(accumulator_full[part], 0)
    gradient_bf16 = tlx.local_load(accumulator[part]).to(tl.bfloat16)
    tl.store(gemm_output + offsets, gradient_bf16)
    gradient = gradient_bf16.to(tl.float32)
    saved = tl.load(saved_input + offsets).to(tl.float32)
    normalized = saved * rstd[:, None]
    weighted = normalized * scale[None, :]
    sigmoid = tl.sigmoid(weighted)
    silu_gradient = gradient * sigmoid * (weighted * (1.0 - sigmoid) + 1.0)
    scaled_gradient = silu_gradient * scale[None, :]
    local_dot = tl.sum(normalized * scaled_gradient, axis=1, keep_dims=True)
    tlx.local_store(row_dot_smem[part], local_dot)
    tlx.barrier_arrive(row_dot_full[0], 1)
    tlx.barrier_wait(row_dot_full[0], 0)
    row_dot = tl.zeros((BLOCK_M, 1), tl.float32)
    for group in tl.static_range(2):
        row_dot += tlx.local_load(row_dot_smem[group])
    row_dot /= N
    dx = (scaled_gradient - normalized * row_dot) * rstd[:, None]
    tl.store(dx_output + offsets, dx.to(tl.bfloat16))
    partial_offsets = (row_start // BLOCK_M) * N + columns
    tl.store(
        dweight_partial + partial_offsets,
        tl.sum(silu_gradient * normalized, axis=0),
    )


@triton.jit
def fused_rmsnorm_silu_backward_tlx(
    a_desc,
    b_desc,
    saved_input,
    rstd_input,
    weight,
    gemm_output,
    dx_output,
    dweight_partial,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_A_BUFFERS: tl.constexpr,
    NUM_B_BUFFERS: tl.constexpr,
    EPILOGUE_WARPS: tl.constexpr,
    EPILOGUE_REGS: tl.constexpr,
):
    n_groups: tl.constexpr = 2
    block_n: tl.constexpr = N // n_groups
    buffers_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, NUM_A_BUFFERS)
    buffers_b = tlx.local_alloc(
        (block_n, BLOCK_K), tl.bfloat16, NUM_B_BUFFERS * n_groups
    )
    accumulator = tlx.local_alloc(
        (BLOCK_M, block_n), tl.float32, n_groups, tlx.storage_kind.tmem
    )
    a_full = tlx.alloc_barriers(NUM_A_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(NUM_A_BUFFERS, arrive_count=n_groups)
    b_full = tlx.alloc_barriers(NUM_B_BUFFERS * n_groups, arrive_count=1)
    b_empty = tlx.alloc_barriers(NUM_B_BUFFERS * n_groups, arrive_count=1)
    accumulator_full = tlx.alloc_barriers(n_groups, arrive_count=1)
    row_dot_smem = tlx.local_alloc((BLOCK_M, 1), tl.float32, n_groups)
    row_dot_full = tlx.alloc_barriers(1, arrive_count=n_groups)
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    k_tiles: tl.constexpr = K // BLOCK_K

    with tlx.async_tasks():
        with tlx.async_task(
            "default", num_warps=EPILOGUE_WARPS, num_regs=EPILOGUE_REGS
        ):
            _rmsnorm_silu_epilogue_part(
                0,
                accumulator,
                accumulator_full,
                row_dot_smem,
                row_dot_full,
                saved_input,
                rstd_input,
                weight,
                gemm_output,
                dx_output,
                dweight_partial,
                row_start,
                N,
                BLOCK_M,
                block_n,
            )

        with tlx.async_task(num_warps=EPILOGUE_WARPS, num_regs=EPILOGUE_REGS):
            _rmsnorm_silu_epilogue_part(
                1,
                accumulator,
                accumulator_full,
                row_dot_smem,
                row_dot_full,
                saved_input,
                rstd_input,
                weight,
                gemm_output,
                dx_output,
                dweight_partial,
                row_start,
                N,
                BLOCK_M,
                block_n,
            )

        with tlx.async_task(num_warps=1, num_regs=24):
            for k in tl.static_range(k_tiles):
                a_buf, a_phase = get_bufidx_phase(k, NUM_A_BUFFERS)
                b_local, b_phase = get_bufidx_phase(k, NUM_B_BUFFERS)
                tlx.barrier_wait(a_full[a_buf], a_phase)
                for group in tl.static_range(n_groups):
                    b_buf = group * NUM_B_BUFFERS + b_local
                    tlx.barrier_wait(b_full[b_buf], b_phase)
                    tlx.async_dot(
                        buffers_a[a_buf],
                        tlx.local_trans(buffers_b[b_buf]),
                        accumulator[group],
                        use_acc=k > 0,
                        mBarriers=[a_empty[a_buf], b_empty[b_buf]],
                        out_dtype=tl.float32,
                    )
            for group in tl.static_range(n_groups):
                tlx.tcgen05_commit(accumulator_full[group])

        with tlx.async_task(num_warps=1, num_regs=40):
            for k in tl.static_range(k_tiles):
                a_buf, a_phase = get_bufidx_phase(k, NUM_A_BUFFERS)
                b_local, b_phase = get_bufidx_phase(k, NUM_B_BUFFERS)
                tlx.barrier_wait(a_empty[a_buf], a_phase ^ 1)
                tlx.barrier_expect_bytes(a_full[a_buf], 2 * BLOCK_M * BLOCK_K)
                tlx.async_descriptor_load(
                    a_desc,
                    buffers_a[a_buf],
                    [row_start, k * BLOCK_K],
                    a_full[a_buf],
                )
                for group in tl.static_range(n_groups):
                    b_buf = group * NUM_B_BUFFERS + b_local
                    tlx.barrier_wait(b_empty[b_buf], b_phase ^ 1)
                    tlx.barrier_expect_bytes(b_full[b_buf], 2 * block_n * BLOCK_K)
                    tlx.async_descriptor_load(
                        b_desc,
                        buffers_b[b_buf],
                        [group * block_n, k * BLOCK_K],
                        b_full[b_buf],
                        eviction_policy="evict_last",
                    )


CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_K": 64,
    "NUM_A_BUFFERS": 4,
    "NUM_B_BUFFERS": 2,
    "EPILOGUE_WARPS": 8,
    "EPILOGUE_REGS": 128,
}


def run_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    *,
    config: dict[str, int] | None = None,
) -> None:
    config = CONFIG if config is None else config
    a_desc = TensorDescriptor(
        inputs["a"],
        inputs["a"].shape,
        inputs["a"].stride(),
        [config["BLOCK_M"], config["BLOCK_K"]],
    )
    b_desc = TensorDescriptor(
        inputs["b"],
        inputs["b"].shape,
        inputs["b"].stride(),
        [baseline.N // 2, config["BLOCK_K"]],
    )
    cast(Any, fused_rmsnorm_silu_backward_tlx)[
        (triton.cdiv(baseline.M, config["BLOCK_M"]),)
    ](
        a_desc,
        b_desc,
        inputs["saved_input"],
        inputs["rstd"],
        inputs["weight"],
        outputs["projection"],
        outputs["dx"],
        outputs["dweight_partial"],
        M=baseline.M,
        N=baseline.N,
        K=baseline.K,
        **config,
        num_warps=4,
        num_stages=1,
    )
    cast(Any, baseline.rmsnorm_silu_backward_dweight_finish)[
        (triton.cdiv(baseline.N, 32),)
    ](
        outputs["dweight_partial"],
        outputs["dweight"],
        N=baseline.N,
        PARTITIONS=triton.cdiv(baseline.M, config["BLOCK_M"]),
        num_warps=4,
    )


def _accuracy(
    reference: baseline.TensorMap, candidate: baseline.TensorMap
) -> dict[str, dict[str, float | bool]]:
    result: dict[str, dict[str, float | bool]] = {}
    for name in ("projection", "dx", "dweight"):
        expected = reference[name].float()
        difference = candidate[name].float() - expected
        result[name] = {
            "relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(expected).clamp_min(1e-30)
            ),
            "max_abs": float(difference.abs().max()),
            "passed": bool(torch.equal(reference[name], candidate[name])),
        }
    return result


def _time_once(fn: Any) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _measure(
    unfused: Any,
    seed: Any,
    tlx_kernel: Any,
    *,
    warmup: int,
    samples: int,
    reps: int,
) -> dict[str, list[float]]:
    for _ in range(warmup):
        unfused()
        seed()
        tlx_kernel()
    torch.cuda.synchronize()
    result = {"unfused_ms": [], "fused_seed_ms": [], "fused_tlx_ms": []}
    for _ in range(reps):
        values = {name: [] for name in result}
        for _ in range(samples):
            values["unfused_ms"].append(_time_once(unfused))
            values["fused_seed_ms"].append(_time_once(seed))
            values["fused_tlx_ms"].append(_time_once(tlx_kernel))
        for name in result:
            result[name].append(statistics.median(values[name]))
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

    accuracy_by_seed = {}
    for seed_value in dict.fromkeys(args.verification_seeds):
        seed_inputs = baseline.make_inputs(seed_value)
        reference = baseline.make_outputs()
        candidate = baseline.make_outputs()
        baseline.run_unfused(seed_inputs, reference)
        run_tlx(seed_inputs, candidate)
        torch.cuda.synchronize()
        accuracy_by_seed[str(seed_value)] = _accuracy(reference, candidate)
    passed = all(
        metric["passed"]
        for seed_result in accuracy_by_seed.values()
        for metric in seed_result.values()
    )
    report: dict[str, Any] = {
        "shape_mnk": [baseline.M, baseline.N, baseline.K],
        "config": CONFIG,
        "accuracy": accuracy_by_seed,
        "passed": passed,
    }
    if not args.check_only:
        inputs = baseline.make_inputs(0)
        unfused_output = baseline.make_outputs()
        seed_output = baseline.make_outputs()
        tlx_output = baseline.make_outputs()
        report["timings"] = _measure(
            lambda: baseline.run_unfused(inputs, unfused_output),
            lambda: baseline.run_seed(inputs, seed_output),
            lambda: run_tlx(inputs, tlx_output),
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
