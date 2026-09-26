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

"""TLX GEMM with a fused weighted RMSNorm and SiLU epilogue.

Public API: ``fwd_rmsnorm_silu_gemm_tlx(a, b, weight) -> (projection, output, rstd)``
computing ``projection = a @ b.T`` in BF16, ``rstd = rsqrt(mean(projection^2) + eps)``
over the full ``N`` row, and ``output = silu(projection * rstd * weight)``.

Schedule (GB200 / sm_100a), column-split cluster:

* One CTA owns ``BLOCK_M x BLOCK_N`` of the output and walks the whole ``K``
  sequentially in one FP32 TMEM accumulator, so ``projection`` is bit-identical
  to cuBLAS. Splitting ``K`` across CTAs was measured to perturb ~1400 elements
  of ``projection`` by one BF16 ulp and to push ``output`` past the accuracy
  contract, so the K chain is deliberately never split.
* Parallelism instead comes from splitting ``N`` over ``NSPLIT`` CTAs of one
  cluster. Only the RMS row statistic crosses a CTA boundary, through
  distributed shared memory, which costs FP32 reassociation of a 2-term sum
  (measured `rstd` error 1.2e-7) rather than of the K chain.
* Four warp groups: an 8-warp epilogue, a 1-warp MMA issuer, and separate
  1-warp TMA producers for A and B so the two operand rings can carry
  different depths (A is the latency-critical one).
* The epilogue drains TMEM through a multi-buffered SMEM staging ring aliased
  onto the now-dead B ring and TMA-stores both results; storing the TMEM
  layout straight to global with `tl.store` was measured 10x slower.
"""

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


EPS = 1e-5


@triton.jit
def rmsnorm_silu_gemm_nsplit_tlx(
    a_desc,
    b_desc,
    p_desc,
    o_desc,
    weight,
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
    NUM_STAGE: tl.constexpr,
    NSPLIT: tl.constexpr,
    EPI_REGS: tl.constexpr,
    EPI_N: tl.constexpr,
):
    k_tiles: tl.constexpr = K // BLOCK_K
    n_chunks: tl.constexpr = BLOCK_N // EPI_N

    a_smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, NUM_A_BUFFERS)
    b_smem = tlx.local_alloc((BLOCK_N, BLOCK_K), tl.bfloat16, NUM_B_BUFFERS)
    accumulator = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, 1, tlx.storage_kind.tmem
    )
    stage = tlx.local_alloc((BLOCK_M, EPI_N), tl.bfloat16, NUM_STAGE, reuse=b_smem)
    row_sum = tlx.local_alloc((BLOCK_M, 1), tl.float32, NSPLIT)

    a_full = tlx.alloc_barriers(NUM_A_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(NUM_A_BUFFERS, arrive_count=1)
    b_full = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    b_empty = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    accumulator_full = tlx.alloc_barriers(1, arrive_count=1)
    row_sum_full = tlx.alloc_barriers(1, arrive_count=1)

    row_start = tl.program_id(0) * BLOCK_M
    rank = tl.program_id(1)
    col_start = rank * BLOCK_N

    with tlx.async_tasks():
        with tlx.async_task("default"):
            rows = row_start + tl.arange(0, BLOCK_M)
            if NSPLIT > 1:
                tlx.barrier_expect_bytes(
                    row_sum_full[0],
                    (NSPLIT - 1) * BLOCK_M * tlx.size_of(tl.float32),
                )
            tlx.barrier_wait(accumulator_full[0], 0)

            square_sum = tl.zeros((BLOCK_M,), tl.float32)
            for chunk in tl.static_range(n_chunks):
                value = tlx.local_load(
                    tlx.subslice(accumulator[0], chunk * EPI_N, EPI_N)
                ).to(tl.bfloat16)
                if chunk >= NUM_STAGE:
                    tlx.async_descriptor_store_wait(NUM_STAGE - 1)
                tlx.local_store(stage[chunk % NUM_STAGE], value)
                tlx.async_descriptor_store(
                    p_desc,
                    stage[chunk % NUM_STAGE],
                    [row_start, col_start + chunk * EPI_N],
                    eviction_policy="evict_first",
                )
                value_fp32 = value.to(tl.float32)
                square_sum += tl.sum(value_fp32 * value_fp32, axis=1)

            if NSPLIT > 1:
                partial = tl.reshape(square_sum, (BLOCK_M, 1))
                tlx.local_store(row_sum[rank], partial)
                for peer in tl.static_range(NSPLIT):
                    if rank != peer:
                        tlx.async_remote_shmem_store(
                            dst=row_sum[rank],
                            src=partial,
                            remote_cta_rank=peer,
                            barrier=row_sum_full[0],
                        )
                tlx.barrier_wait(row_sum_full[0], 0)
                total = tl.zeros((BLOCK_M, 1), tl.float32)
                for peer in tl.static_range(NSPLIT):
                    total += tlx.local_load(row_sum[peer])
                square_sum = tl.reshape(total, (BLOCK_M,))

            rstd = tl.rsqrt(square_sum / N + EPS)
            if rank == 0:
                tl.store(rstd_output + rows, rstd)

            for out_chunk in tl.static_range(n_chunks):
                value_fp32 = (
                    tlx.local_load(
                        tlx.subslice(accumulator[0], out_chunk * EPI_N, EPI_N)
                    )
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
                columns = col_start + out_chunk * EPI_N + tl.arange(0, EPI_N)
                if NSPLIT * BLOCK_N == N:
                    scale = tl.load(weight + columns).to(tl.float32)
                else:
                    scale = tl.load(weight + columns, mask=columns < N, other=0.0).to(
                        tl.float32
                    )
                normalized = value_fp32 * rstd[:, None] * scale[None, :]
                # silu, with the reciprocal and exponential taken at approximate
                # FP32 (~1e-7 relative) -- four orders below one BF16 ulp.
                activated = tl.fdiv(
                    normalized,
                    1.0 + tl.exp2(normalized * -1.4426950408889634),
                    ieee_rounding=False,
                )
                slot = (n_chunks + out_chunk) % NUM_STAGE
                tlx.async_descriptor_store_wait(NUM_STAGE - 1)
                tlx.local_store(stage[slot], activated.to(tl.bfloat16))
                tlx.async_descriptor_store(
                    o_desc,
                    stage[slot],
                    [row_start, col_start + out_chunk * EPI_N],
                    eviction_policy="evict_first",
                )
            tlx.async_descriptor_store_wait(0)

        with tlx.async_task(num_warps=1, num_regs=24):
            tlx.barrier_wait(a_full[0], 0)
            tlx.barrier_wait(b_full[0], 0)
            tlx.async_dot(
                a_smem[0],
                tlx.local_trans(b_smem[0]),
                accumulator[0],
                use_acc=False,
                mBarriers=[a_empty[0], b_empty[0]],
                out_dtype=tl.float32,
            )
            for index in range(1, k_tiles):
                a_buffer, a_phase = get_bufidx_phase(index, NUM_A_BUFFERS)
                b_buffer, b_phase = get_bufidx_phase(index, NUM_B_BUFFERS)
                tlx.barrier_wait(a_full[a_buffer], a_phase)
                tlx.barrier_wait(b_full[b_buffer], b_phase)
                tlx.async_dot(
                    a_smem[a_buffer],
                    tlx.local_trans(b_smem[b_buffer]),
                    accumulator[0],
                    use_acc=True,
                    mBarriers=[a_empty[a_buffer], b_empty[b_buffer]],
                    out_dtype=tl.float32,
                )
            tlx.tcgen05_commit(accumulator_full[0])

        with tlx.async_task(num_warps=1, num_regs=32):
            for bindex in range(k_tiles):
                b_buffer, b_phase = get_bufidx_phase(bindex, NUM_B_BUFFERS)
                tlx.barrier_wait(b_empty[b_buffer], b_phase ^ 1)
                tlx.barrier_expect_bytes(b_full[b_buffer], 2 * BLOCK_N * BLOCK_K)
                tlx.async_descriptor_load(
                    b_desc,
                    b_smem[b_buffer],
                    [col_start, bindex * BLOCK_K],
                    b_full[b_buffer],
                    eviction_policy="evict_last",
                )

        with tlx.async_task(num_warps=1, num_regs=32):
            for index in range(k_tiles):
                a_buffer, a_phase = get_bufidx_phase(index, NUM_A_BUFFERS)
                tlx.barrier_wait(a_empty[a_buffer], a_phase ^ 1)
                tlx.barrier_expect_bytes(a_full[a_buffer], 2 * BLOCK_M * BLOCK_K)
                tlx.async_descriptor_load(
                    a_desc,
                    a_smem[a_buffer],
                    [row_start, index * BLOCK_K],
                    a_full[a_buffer],
                )


CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "NUM_A_BUFFERS": 6,
    "NUM_B_BUFFERS": 4,
    "NUM_STAGE": 4,
    "NSPLIT": 2,
    "EPI_REGS": 232,
    "EPI_N": 64,
    "LAUNCH_WARPS": 8,
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


def make_outputs(m: int, n: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "projection": torch.empty((m, n), device=device, dtype=torch.bfloat16),
        "output": torch.empty((m, n), device=device, dtype=torch.bfloat16),
        "rstd": torch.empty((m,), device=device, dtype=torch.float32),
    }


def run_tlx(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    config: dict[str, int] | None = None,
) -> None:
    config = CONFIG if config is None else config
    launch_warps = config["LAUNCH_WARPS"]
    nsplit = config["NSPLIT"]
    kernel_config = {
        key: value for key, value in config.items() if key != "LAUNCH_WARPS"
    }
    m, k = inputs["a"].shape
    n = inputs["b"].shape[0]
    block_m = config["BLOCK_M"]
    epi_n = config["EPI_N"]
    a_desc = _descriptor("a", inputs["a"], (block_m, config["BLOCK_K"]))
    b_desc = _descriptor("b", inputs["b"], (config["BLOCK_N"], config["BLOCK_K"]))
    p_desc = _descriptor("p", outputs["projection"], (block_m, epi_n))
    o_desc = _descriptor("o", outputs["output"], (block_m, epi_n))
    grid = (triton.cdiv(m, block_m), nsplit)
    extra_launch = {"ctas_per_cga": (1, nsplit, 1)} if nsplit > 1 else {}
    cast(Any, rmsnorm_silu_gemm_nsplit_tlx)[grid](
        a_desc,
        b_desc,
        p_desc,
        o_desc,
        inputs["weight"],
        outputs["rstd"],
        M=m,
        N=n,
        K=k,
        EPS=EPS,
        **kernel_config,
        num_warps=launch_warps,
        num_stages=1,
        **extra_launch,
    )


def fwd_rmsnorm_silu_gemm_tlx(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: torch.Tensor,
    *,
    config: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused ``a @ b.T`` with a weighted RMSNorm and SiLU epilogue.

    Returns ``(projection, output, rstd)`` where ``projection`` is the bf16
    GEMM result the epilogue normalizes, ``output`` is the activated result and
    ``rstd`` is the per-row reciprocal RMS the normalization used.
    """
    outputs = make_outputs(a.shape[0], b.shape[0], a.device)
    run_tlx({"a": a, "b": b, "weight": weight}, outputs, config=config)
    return outputs["projection"], outputs["output"], outputs["rstd"]


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
    parser.add_argument("--block-n", type=int, default=CONFIG["BLOCK_N"])
    parser.add_argument("--block-k", type=int, default=CONFIG["BLOCK_K"])
    parser.add_argument("--a-buffers", type=int, default=CONFIG["NUM_A_BUFFERS"])
    parser.add_argument("--b-buffers", type=int, default=CONFIG["NUM_B_BUFFERS"])
    parser.add_argument("--stages", type=int, default=CONFIG["NUM_STAGE"])
    parser.add_argument("--n-split", type=int, default=CONFIG["NSPLIT"])
    parser.add_argument("--epilogue-regs", type=int, default=CONFIG["EPI_REGS"])
    parser.add_argument("--epilogue-n", type=int, default=CONFIG["EPI_N"])
    parser.add_argument("--launch-warps", type=int, default=CONFIG["LAUNCH_WARPS"])
    args = parser.parse_args()
    config = {
        "BLOCK_M": args.block_m,
        "BLOCK_N": args.block_n,
        "BLOCK_K": args.block_k,
        "NUM_A_BUFFERS": args.a_buffers,
        "NUM_B_BUFFERS": args.b_buffers,
        "NUM_STAGE": args.stages,
        "NSPLIT": args.n_split,
        "EPI_REGS": args.epilogue_regs,
        "EPI_N": args.epilogue_n,
        "LAUNCH_WARPS": args.launch_warps,
    }

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
