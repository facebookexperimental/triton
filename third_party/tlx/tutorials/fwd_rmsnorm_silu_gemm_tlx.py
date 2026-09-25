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

"""TLX GEMM with a fused weighted RMSNorm and SiLU epilogue."""

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
from triton.tlx.ops import mm as tlx_mm

import fwd_rmsnorm_silu_gemm as baseline


@triton.jit
def _epilogue(
    group,
    accumulator,
    accumulator_full,
    pre_norm,
    row_sum,
    row_sum_full,
    weight,
    projection,
    output,
    rstd_output,
    row_start,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
):
    rows = row_start + tl.arange(0, BLOCK_M)
    columns = group * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + columns[None, :]
    tlx.barrier_wait(accumulator_full[group], 0)
    value = tlx.local_load(accumulator[group]).to(tl.bfloat16)
    tl.store(projection + offsets, value)
    value_fp32 = value.to(tl.float32)
    partial = tl.sum(value_fp32 * value_fp32, axis=1, keep_dims=True)
    tlx.local_store(row_sum[group], partial)
    tlx.fence_async_shared()
    tlx.barrier_arrive(row_sum_full[0], 1)
    tlx.barrier_wait(row_sum_full[0], 0)

    square_sum = tl.zeros((BLOCK_M, 1), tl.float32)
    for index in tl.static_range(N_GROUPS):
        square_sum += tlx.local_load(row_sum[index])
    rstd_2d = tl.rsqrt(square_sum / N + EPS)
    rstd = tl.reshape(rstd_2d, (BLOCK_M,))
    if group == 0:
        tl.store(rstd_output + rows, rstd)

    scale = tl.load(weight + columns).to(tl.float32)
    normalized = value_fp32 * rstd[:, None] * scale[None, :]
    activated = normalized * tl.sigmoid(normalized)
    tl.store(output + offsets, activated.to(tl.bfloat16))


@triton.jit
def _epilogue_all(
    accumulator,
    accumulator_full,
    pre_norm,
    weight,
    projection,
    output,
    rstd_output,
    row_start,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
):
    rows = row_start + tl.arange(0, BLOCK_M)
    square_sum = tl.zeros((BLOCK_M,), tl.float32)
    for group in tl.static_range(N_GROUPS):
        columns = group * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets = rows[:, None] * N + columns[None, :]
        tlx.barrier_wait(accumulator_full[group], 0)
        value = tlx.local_load(accumulator[group]).to(tl.bfloat16)
        tl.store(projection + offsets, value)
        tlx.local_store(pre_norm[group], value)
        value_fp32 = value.to(tl.float32)
        square_sum += tl.sum(value_fp32 * value_fp32, axis=1)

    rstd = tl.rsqrt(square_sum / N + EPS)
    tl.store(rstd_output + rows, rstd)
    for group in tl.static_range(N_GROUPS):
        columns = group * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets = rows[:, None] * N + columns[None, :]
        value_fp32 = tlx.local_load(pre_norm[group]).to(tl.float32)
        scale = tl.load(weight + columns).to(tl.float32)
        normalized = value_fp32 * rstd[:, None] * scale[None, :]
        activated = normalized * tl.sigmoid(normalized)
        tl.store(output + offsets, activated.to(tl.bfloat16))


@triton.jit
def rmsnorm_silu_gemm_tlx(
    a_desc,
    b_desc,
    weight,
    projection,
    output,
    rstd_output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_A_BUFFERS: tl.constexpr,
    NUM_B_BUFFERS: tl.constexpr,
    EPILOGUE_WARPS: tl.constexpr,
    EPILOGUE_REGS: tl.constexpr,
    SPLIT_EPILOGUE: tl.constexpr,
):
    n_groups: tl.constexpr = N // BLOCK_N
    a_smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, NUM_A_BUFFERS)
    b_smem = tlx.local_alloc((BLOCK_N, BLOCK_K), tl.bfloat16, NUM_B_BUFFERS * n_groups)
    accumulator = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, n_groups, tlx.storage_kind.tmem
    )
    pre_norm = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.bfloat16, n_groups, reuse=b_smem)
    row_sum = tlx.local_alloc((BLOCK_M, 1), tl.float32, n_groups)
    a_full = tlx.alloc_barriers(NUM_A_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(NUM_A_BUFFERS, arrive_count=n_groups)
    b_full = tlx.alloc_barriers(NUM_B_BUFFERS * n_groups, arrive_count=1)
    b_empty = tlx.alloc_barriers(NUM_B_BUFFERS * n_groups, arrive_count=1)
    accumulator_full = tlx.alloc_barriers(n_groups, arrive_count=1)
    row_sum_full = tlx.alloc_barriers(1, arrive_count=n_groups)

    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    k_tiles: tl.constexpr = K // BLOCK_K

    with tlx.async_tasks():
        with tlx.async_task("default"):
            if SPLIT_EPILOGUE:
                _epilogue(
                    0,
                    accumulator,
                    accumulator_full,
                    pre_norm,
                    row_sum,
                    row_sum_full,
                    weight,
                    projection,
                    output,
                    rstd_output,
                    row_start,
                    M,
                    N,
                    BLOCK_M,
                    BLOCK_N,
                    n_groups,
                    EPS,
                )
            else:
                _epilogue_all(
                    accumulator,
                    accumulator_full,
                    pre_norm,
                    weight,
                    projection,
                    output,
                    rstd_output,
                    row_start,
                    M,
                    N,
                    BLOCK_M,
                    BLOCK_N,
                    n_groups,
                    EPS,
                )

        if SPLIT_EPILOGUE:
            with tlx.async_task(num_warps=EPILOGUE_WARPS, num_regs=EPILOGUE_REGS):
                _epilogue(
                    1,
                    accumulator,
                    accumulator_full,
                    pre_norm,
                    row_sum,
                    row_sum_full,
                    weight,
                    projection,
                    output,
                    rstd_output,
                    row_start,
                    M,
                    N,
                    BLOCK_M,
                    BLOCK_N,
                    n_groups,
                    EPS,
                )

        with tlx.async_task(num_warps=1, num_regs=24):
            for k_tile in tl.static_range(k_tiles):
                a_buffer, a_phase = get_bufidx_phase(k_tile, NUM_A_BUFFERS)
                b_local, b_phase = get_bufidx_phase(k_tile, NUM_B_BUFFERS)
                tlx.barrier_wait(a_full[a_buffer], a_phase)
                for group in tl.static_range(n_groups):
                    b_buffer = group * NUM_B_BUFFERS + b_local
                    tlx.barrier_wait(b_full[b_buffer], b_phase)
                    tlx.async_dot(
                        a_smem[a_buffer],
                        tlx.local_trans(b_smem[b_buffer]),
                        accumulator[group],
                        use_acc=k_tile > 0,
                        mBarriers=[a_empty[a_buffer], b_empty[b_buffer]],
                        out_dtype=tl.float32,
                    )
            for group in tl.static_range(n_groups):
                tlx.tcgen05_commit(accumulator_full[group])

        with tlx.async_task(num_warps=1, num_regs=32):
            for k_tile in tl.static_range(k_tiles):
                a_buffer, a_phase = get_bufidx_phase(k_tile, NUM_A_BUFFERS)
                b_local, b_phase = get_bufidx_phase(k_tile, NUM_B_BUFFERS)
                tlx.barrier_wait(a_empty[a_buffer], a_phase ^ 1)
                tlx.barrier_expect_bytes(a_full[a_buffer], 2 * BLOCK_M * BLOCK_K)
                tlx.async_descriptor_load(
                    a_desc,
                    a_smem[a_buffer],
                    [row_start, k_tile * BLOCK_K],
                    a_full[a_buffer],
                )
                for group in tl.static_range(n_groups):
                    b_buffer = group * NUM_B_BUFFERS + b_local
                    tlx.barrier_wait(b_empty[b_buffer], b_phase ^ 1)
                    tlx.barrier_expect_bytes(b_full[b_buffer], 2 * BLOCK_N * BLOCK_K)
                    tlx.async_descriptor_load(
                        b_desc,
                        b_smem[b_buffer],
                        [group * BLOCK_N, k_tile * BLOCK_K],
                        b_full[b_buffer],
                        eviction_policy="evict_last",
                    )


@triton.jit
def _cross_cta_sum(
    value,
    cta_rank,
    row_sum,
    row_sum_full,
    BLOCK_M: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    partial = tl.sum(value, axis=1, keep_dims=True)
    tlx.local_store(row_sum[cta_rank], partial)
    for peer in tl.static_range(NUM_CTAS):
        if cta_rank != peer:
            tlx.async_remote_shmem_store(
                dst=row_sum[cta_rank],
                src=partial,
                remote_cta_rank=peer,
                barrier=row_sum_full[0],
            )
    tlx.barrier_wait(row_sum_full[0], 0)
    total = tl.zeros((BLOCK_M, 1), tl.float32)
    for peer in tl.static_range(NUM_CTAS):
        total += tlx.local_load(row_sum[peer])
    return total


@triton.jit
def rmsnorm_silu_gemm_clustered_tlx(
    a_desc,
    b_desc,
    weight,
    projection,
    output,
    rstd_output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_A_BUFFERS: tl.constexpr,
    NUM_B_BUFFERS: tl.constexpr,
    EPILOGUE_WARPS: tl.constexpr,
    EPILOGUE_REGS: tl.constexpr,
    SPLIT_EPILOGUE: tl.constexpr,
):
    num_ctas: tl.constexpr = N // BLOCK_N
    cta_rank = tlx.cluster_cta_rank()
    start_m = tl.program_id(0) * BLOCK_M
    start_n = tl.program_id(1) * BLOCK_N
    k_tiles: tl.constexpr = K // BLOCK_K

    a_smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, NUM_A_BUFFERS)
    b_smem = tlx.local_alloc((BLOCK_N, BLOCK_K), tl.bfloat16, NUM_B_BUFFERS)
    accumulator = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, 1, tlx.storage_kind.tmem
    )
    pre_norm = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.bfloat16, 1, reuse=b_smem)
    row_sum = tlx.local_alloc((BLOCK_M, 1), tl.float32, num_ctas)
    a_full = tlx.alloc_barriers(NUM_A_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(NUM_A_BUFFERS, arrive_count=1)
    b_full = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    b_empty = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    accumulator_full = tlx.alloc_barriers(1, arrive_count=1)
    row_sum_full = tlx.alloc_barriers(1, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            rows = start_m + tl.arange(0, BLOCK_M)
            columns = start_n + tl.arange(0, BLOCK_N)
            offsets = rows[:, None] * N + columns[None, :]
            tlx.barrier_expect_bytes(
                row_sum_full[0],
                (num_ctas - 1) * BLOCK_M * tlx.size_of(tl.float32),
            )
            tlx.barrier_wait(accumulator_full[0], 0)
            value = tlx.local_load(accumulator[0]).to(tl.bfloat16)
            tl.store(projection + offsets, value)
            tlx.local_store(pre_norm[0], value)
            value_fp32 = value.to(tl.float32)
            square_sum = _cross_cta_sum(
                value_fp32 * value_fp32,
                cta_rank,
                row_sum,
                row_sum_full,
                BLOCK_M,
                num_ctas,
            )
            rstd_2d = tl.rsqrt(square_sum / N + EPS)
            rstd = tl.reshape(rstd_2d, (BLOCK_M,))
            if cta_rank == 0:
                tl.store(rstd_output + rows, rstd)
            value_fp32 = tlx.local_load(pre_norm[0]).to(tl.float32)
            scale = tl.load(weight + columns).to(tl.float32)
            normalized = value_fp32 * rstd[:, None] * scale[None, :]
            activated = normalized * tl.sigmoid(normalized)
            tl.store(output + offsets, activated.to(tl.bfloat16))

        with tlx.async_task(num_warps=1, num_regs=24):
            for k_tile in tl.static_range(k_tiles):
                a_buffer, a_phase = get_bufidx_phase(k_tile, NUM_A_BUFFERS)
                b_buffer, b_phase = get_bufidx_phase(k_tile, NUM_B_BUFFERS)
                tlx.barrier_wait(a_full[a_buffer], a_phase)
                tlx.barrier_wait(b_full[b_buffer], b_phase)
                tlx.async_dot(
                    a_smem[a_buffer],
                    tlx.local_trans(b_smem[b_buffer]),
                    accumulator[0],
                    use_acc=k_tile > 0,
                    mBarriers=[a_empty[a_buffer], b_empty[b_buffer]],
                    out_dtype=tl.float32,
                )
            tlx.tcgen05_commit(accumulator_full[0])

        with tlx.async_task(num_warps=1, num_regs=32):
            for k_tile in tl.static_range(k_tiles):
                a_buffer, a_phase = get_bufidx_phase(k_tile, NUM_A_BUFFERS)
                b_buffer, b_phase = get_bufidx_phase(k_tile, NUM_B_BUFFERS)
                tlx.barrier_wait(a_empty[a_buffer], a_phase ^ 1)
                tlx.barrier_wait(b_empty[b_buffer], b_phase ^ 1)
                tlx.barrier_expect_bytes(a_full[a_buffer], 2 * BLOCK_M * BLOCK_K)
                tlx.async_descriptor_load(
                    a_desc,
                    a_smem[a_buffer],
                    [start_m, k_tile * BLOCK_K],
                    a_full[a_buffer],
                )
                tlx.barrier_expect_bytes(b_full[b_buffer], 2 * BLOCK_N * BLOCK_K)
                tlx.async_descriptor_load(
                    b_desc,
                    b_smem[b_buffer],
                    [start_n, k_tile * BLOCK_K],
                    b_full[b_buffer],
                    eviction_policy="evict_last",
                )


CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "NUM_A_BUFFERS": 4,
    "NUM_B_BUFFERS": 2,
    "EPILOGUE_WARPS": 8,
    "EPILOGUE_REGS": 128,
    "LAUNCH_WARPS": 8,
    "CLUSTERED": 0,
    "SPLIT_EPILOGUE": 1,
}


_DESCRIPTORS: dict[tuple[str, int, int], tuple[torch.Tensor, TensorDescriptor]] = {}


def _descriptor(
    name: str, tensor: torch.Tensor, block: tuple[int, int]
) -> TensorDescriptor:
    key = (name, block[0], block[1])
    cached = _DESCRIPTORS.get(key)
    if cached is None or cached[0] is not tensor:
        descriptor = TensorDescriptor(
            tensor, list(tensor.shape), list(tensor.stride()), list(block)
        )
        _DESCRIPTORS[key] = (tensor, descriptor)
        return descriptor
    return cached[1]


def run_tlx(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    config: dict[str, int] | None = None,
) -> None:
    config = CONFIG if config is None else config
    launch_warps = config["LAUNCH_WARPS"]
    clustered = bool(config["CLUSTERED"])
    kernel_config = {
        key: value
        for key, value in config.items()
        if key not in ("LAUNCH_WARPS", "CLUSTERED")
    }
    a_desc = _descriptor("a", inputs["a"], (config["BLOCK_M"], config["BLOCK_K"]))
    b_desc = _descriptor("b", inputs["b"], (config["BLOCK_N"], config["BLOCK_K"]))
    kernel = rmsnorm_silu_gemm_clustered_tlx if clustered else rmsnorm_silu_gemm_tlx
    reduction_ctas = baseline.N // config["BLOCK_N"]
    grid = (
        (triton.cdiv(baseline.M, config["BLOCK_M"]), reduction_ctas)
        if clustered
        else (triton.cdiv(baseline.M, config["BLOCK_M"]),)
    )
    extra_launch = {"ctas_per_cga": (1, reduction_ctas, 1)} if clustered else {}
    cast(Any, kernel)[grid](
        a_desc,
        b_desc,
        inputs["weight"],
        outputs["projection"],
        outputs["output"],
        outputs["rstd"],
        M=baseline.M,
        N=baseline.N,
        K=baseline.K,
        EPS=baseline.EPS,
        **kernel_config,
        num_warps=launch_warps,
        num_stages=1,
        **extra_launch,
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
    names = list(calls)
    for rep in range(reps):
        samples_by_name = {name: [] for name in calls}
        for sample in range(samples):
            order = names if (rep + sample) % 2 == 0 else list(reversed(names))
            for name in order:
                samples_by_name[name].append(_time_once(calls[name]))
        for name in names:
            result[name].append(statistics.median(samples_by_name[name]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--verification-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--block-m", type=int, default=CONFIG["BLOCK_M"])
    parser.add_argument("--block-n", type=int, choices=(128, 256), default=256)
    parser.add_argument("--block-k", type=int, default=CONFIG["BLOCK_K"])
    parser.add_argument("--a-buffers", type=int, default=CONFIG["NUM_A_BUFFERS"])
    parser.add_argument("--b-buffers", type=int, default=CONFIG["NUM_B_BUFFERS"])
    parser.add_argument("--epilogue-warps", type=int, default=CONFIG["EPILOGUE_WARPS"])
    parser.add_argument("--epilogue-regs", type=int, default=CONFIG["EPILOGUE_REGS"])
    parser.add_argument("--launch-warps", type=int, default=CONFIG["LAUNCH_WARPS"])
    parser.add_argument(
        "--clustered",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["CLUSTERED"]),
    )
    parser.add_argument(
        "--split-epilogue",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["SPLIT_EPILOGUE"]),
    )
    args = parser.parse_args()
    config = {
        "BLOCK_M": args.block_m,
        "BLOCK_N": args.block_n,
        "BLOCK_K": args.block_k,
        "NUM_A_BUFFERS": args.a_buffers,
        "NUM_B_BUFFERS": args.b_buffers,
        "EPILOGUE_WARPS": args.epilogue_warps,
        "EPILOGUE_REGS": args.epilogue_regs,
        "LAUNCH_WARPS": args.launch_warps,
        "CLUSTERED": int(args.clustered),
        "SPLIT_EPILOGUE": int(args.split_epilogue),
    }
    if not args.clustered and args.block_n != 256:
        raise ValueError("the one-CTA kernel currently requires BLOCK_N=256")

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
        "config": config,
        "accuracy": accuracy_by_seed,
        "passed": passed,
    }
    if not args.check_only:
        inputs = baseline.make_inputs(0)
        outputs = {
            name: baseline.make_outputs()
            for name in ("unfused", "seed", "tlx_mm", "tlx")
        }
        report["timings_us"] = _measure(
            {
                "unfused": lambda: baseline.run_unfused(inputs, outputs["unfused"]),
                "triton_seed": lambda: baseline.run_seed(inputs, outputs["seed"]),
                "tlx_mm": lambda: tlx_mm(
                    inputs["a"],
                    inputs["b"].T,
                    out=outputs["tlx_mm"]["projection"],
                ),
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
