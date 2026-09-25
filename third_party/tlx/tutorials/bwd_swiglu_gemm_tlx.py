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

"""TLX dW GEMM with a fused BF16 SwiGLU reconstruction prologue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

import bwd_swiglu_gemm as baseline


@triton.jit
def _store_accumulator(
    group,
    accumulators,
    accumulator_full,
    dweight,
    start_m,
    start_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPILOGUE_SUBTILES: tl.constexpr,
):
    tlx.barrier_wait(accumulator_full[group], 0)
    for subtile in tl.static_range(EPILOGUE_SUBTILES):
        accumulator = tlx.subslice(
            accumulators[group],
            subtile * (BLOCK_N // EPILOGUE_SUBTILES),
            BLOCK_N // EPILOGUE_SUBTILES,
        )
        result = tlx.local_load(accumulator).to(tl.bfloat16)
        rows = start_m + group * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = (
            start_n
            + subtile * (BLOCK_N // EPILOGUE_SUBTILES)
            + tl.arange(0, BLOCK_N // EPILOGUE_SUBTILES)
        )
        tl.store(dweight + rows[:, None] * baseline.N + columns[None, :], result)


@triton.jit
def swiglu_gemm_tlx(
    down_gradient_desc,
    gate,
    up,
    dweight,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_GROUPS: tl.constexpr,
    NUM_A_BUFFERS: tl.constexpr,
    NUM_B_BUFFERS: tl.constexpr,
    B_RELEASE_GROUPS: tl.constexpr,
    PRODUCER_WARPS: tl.constexpr,
    PRODUCER_REGS: tl.constexpr,
    EPILOGUE_WARPS: tl.constexpr,
    EPILOGUE_REGS: tl.constexpr,
    EPILOGUE_SUBTILES: tl.constexpr,
):
    buffers_a = tlx.local_alloc(
        (BLOCK_K, BLOCK_M),
        tl.bfloat16,
        M_GROUPS * NUM_A_BUFFERS,
    )
    buffers_b = tlx.local_alloc(
        (BLOCK_K, BLOCK_N),
        tl.bfloat16,
        NUM_B_BUFFERS,
    )
    accumulators = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        M_GROUPS,
        tlx.storage_kind.tmem,
    )
    a_full = tlx.alloc_barriers(M_GROUPS * NUM_A_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(M_GROUPS * NUM_A_BUFFERS, arrive_count=1)
    b_full = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    b_empty = tlx.alloc_barriers(
        NUM_B_BUFFERS * B_RELEASE_GROUPS,
        arrive_count=M_GROUPS // B_RELEASE_GROUPS,
    )
    accumulator_full = tlx.alloc_barriers(M_GROUPS, arrive_count=1)

    num_pid_n: tl.constexpr = baseline.N // BLOCK_N
    pid = tl.program_id(0)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    start_m = pid_m * M_GROUPS * BLOCK_M
    start_n = pid_n * BLOCK_N
    k_tiles: tl.constexpr = baseline.K // BLOCK_K

    with tlx.async_tasks():
        with tlx.async_task(
            "default", num_warps=EPILOGUE_WARPS, num_regs=EPILOGUE_REGS
        ):
            for group in tl.static_range(M_GROUPS):
                _store_accumulator(
                    group,
                    accumulators,
                    accumulator_full,
                    dweight,
                    start_m,
                    start_n,
                    BLOCK_M,
                    BLOCK_N,
                    EPILOGUE_SUBTILES,
                )

        with tlx.async_task(num_warps=1, num_regs=24):
            for k in tl.static_range(k_tiles):
                buf, phase = get_bufidx_phase(k, NUM_B_BUFFERS)
                tlx.barrier_wait(b_full[buf], phase)
                a_local, a_phase = get_bufidx_phase(k, NUM_A_BUFFERS)
                for group in tl.static_range(M_GROUPS):
                    a_buf = group * NUM_A_BUFFERS + a_local
                    b_empty_buf = buf * B_RELEASE_GROUPS + group // (
                        M_GROUPS // B_RELEASE_GROUPS
                    )
                    tlx.barrier_wait(a_full[a_buf], a_phase)
                    tlx.async_dot(
                        tlx.local_trans(buffers_a[a_buf]),
                        buffers_b[buf],
                        accumulators[group],
                        use_acc=k > 0,
                        mBarriers=[a_empty[a_buf], b_empty[b_empty_buf]],
                        out_dtype=tl.float32,
                    )
            last_buf, last_phase = get_bufidx_phase(k_tiles - 1, NUM_B_BUFFERS)
            for release_group in tl.static_range(B_RELEASE_GROUPS):
                tlx.barrier_wait(
                    b_empty[last_buf * B_RELEASE_GROUPS + release_group],
                    last_phase,
                )
            for group in tl.static_range(M_GROUPS):
                tlx.tcgen05_commit(accumulator_full[group])

        with tlx.async_task(num_warps=1, num_regs=24):
            for k in tl.static_range(k_tiles):
                buf, phase = get_bufidx_phase(k, NUM_B_BUFFERS)
                a_local, a_phase = get_bufidx_phase(k, NUM_A_BUFFERS)
                for group in tl.static_range(M_GROUPS):
                    a_buf = group * NUM_A_BUFFERS + a_local
                    tlx.barrier_wait(a_empty[a_buf], a_phase ^ 1)
                    tlx.barrier_expect_bytes(a_full[a_buf], 2 * BLOCK_K * BLOCK_M)
                    tlx.async_descriptor_load(
                        down_gradient_desc,
                        buffers_a[a_buf],
                        [k * BLOCK_K, start_m + group * BLOCK_M],
                        a_full[a_buf],
                    )

        with tlx.async_task(num_warps=PRODUCER_WARPS, num_regs=PRODUCER_REGS):
            for k in tl.static_range(k_tiles):
                buf, phase = get_bufidx_phase(k, NUM_B_BUFFERS)
                ks = k * BLOCK_K + tl.arange(0, BLOCK_K)
                columns = start_n + tl.arange(0, BLOCK_N)
                offsets = ks[:, None] * baseline.N + columns[None, :]
                gate_value = tl.load(gate + offsets).to(tl.float32)
                up_value = tl.load(up + offsets).to(tl.float32)
                sigmoid = tl.sigmoid(gate_value)
                hidden = gate_value * sigmoid * up_value
                for release_group in tl.static_range(B_RELEASE_GROUPS):
                    tlx.barrier_wait(
                        b_empty[buf * B_RELEASE_GROUPS + release_group],
                        phase ^ 1,
                    )
                tlx.local_store(buffers_b[buf], hidden.to(tl.bfloat16))
                tlx.fence_async_shared()
                tlx.barrier_arrive(b_full[buf], 1)


# Rejected DSMEM broadcast experiment. It is intentionally not exposed by the
# driver because its all-to-all barrier protocol does not complete reliably.
@triton.jit
def _publish_hidden_slice(
    part: tl.constexpr,
    hidden,
    buffers_b,
    b_full,
    buf,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CLUSTER_CTAS: tl.constexpr,
):
    rows_per_cta: tl.constexpr = BLOCK_K // CLUSTER_CTAS
    destination = tlx.local_slice(
        buffers_b[buf],
        [part * rows_per_cta, 0],
        [rows_per_cta, BLOCK_N],
    )
    tlx.local_store(destination, hidden)
    for target in tl.static_range(CLUSTER_CTAS):
        if target != part:
            tlx.remote_shmem_store(
                dst=destination,
                src=hidden,
                remote_cta_rank=target,
            )


@triton.jit
def swiglu_gemm_cluster_tlx(
    down_gradient_desc,
    gate,
    up,
    dweight,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    CLUSTER_CTAS: tl.constexpr,
    PRODUCER_WARPS: tl.constexpr,
    PRODUCER_REGS: tl.constexpr,
    EPILOGUE_WARPS: tl.constexpr,
    EPILOGUE_REGS: tl.constexpr,
    EPILOGUE_SUBTILES: tl.constexpr,
):
    buffers_a = tlx.local_alloc((BLOCK_K, BLOCK_M), tl.bfloat16, NUM_BUFFERS)
    buffers_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.bfloat16, NUM_BUFFERS)
    accumulator = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, 1, tlx.storage_kind.tmem
    )
    a_full = tlx.alloc_barriers(NUM_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(NUM_BUFFERS, arrive_count=1)
    b_full = tlx.alloc_barriers(NUM_BUFFERS, arrive_count=CLUSTER_CTAS)
    b_empty = tlx.alloc_barriers(NUM_BUFFERS, arrive_count=1)
    reuse_ready = tlx.alloc_barriers(NUM_BUFFERS, arrive_count=CLUSTER_CTAS)
    accumulator_full = tlx.alloc_barriers(1, arrive_count=1)

    cta_rank = tlx.cluster_cta_rank()
    cluster_id = tl.program_id(0) // CLUSTER_CTAS
    num_pid_n: tl.constexpr = baseline.N // BLOCK_N
    pid_m_group = cluster_id // num_pid_n
    pid_n = cluster_id % num_pid_n
    start_m = (pid_m_group * CLUSTER_CTAS + cta_rank) * BLOCK_M
    start_n = pid_n * BLOCK_N
    k_tiles: tl.constexpr = baseline.K // BLOCK_K
    rows_per_cta: tl.constexpr = BLOCK_K // CLUSTER_CTAS
    with tlx.async_tasks():
        with tlx.async_task(
            "default", num_warps=EPILOGUE_WARPS, num_regs=EPILOGUE_REGS
        ):
            tlx.barrier_wait(accumulator_full[0], 0)
            for subtile in tl.static_range(EPILOGUE_SUBTILES):
                acc_slice = tlx.local_slice(
                    accumulator[0],
                    [0, subtile * (BLOCK_N // EPILOGUE_SUBTILES)],
                    [BLOCK_M, BLOCK_N // EPILOGUE_SUBTILES],
                )
                result = tlx.local_load(acc_slice).to(tl.bfloat16)
                rows = start_m + tl.arange(0, BLOCK_M)
                columns = (
                    start_n
                    + subtile * (BLOCK_N // EPILOGUE_SUBTILES)
                    + tl.arange(0, BLOCK_N // EPILOGUE_SUBTILES)
                )
                tl.store(
                    dweight + rows[:, None] * baseline.N + columns[None, :],
                    result,
                )

        with tlx.async_task(num_warps=1, num_regs=24):
            for k in tl.static_range(k_tiles):
                buf, phase = get_bufidx_phase(k, NUM_BUFFERS)
                tlx.barrier_wait(a_full[buf], phase)
                tlx.barrier_wait(b_full[buf], phase)
                tlx.async_dot(
                    tlx.local_trans(buffers_a[buf]),
                    buffers_b[buf],
                    accumulator[0],
                    use_acc=k > 0,
                    mBarriers=[a_empty[buf], b_empty[buf]],
                    out_dtype=tl.float32,
                )
            last_buf, last_phase = get_bufidx_phase(k_tiles - 1, NUM_BUFFERS)
            tlx.barrier_wait(a_empty[last_buf], last_phase)
            tlx.tcgen05_commit(accumulator_full[0])

        with tlx.async_task(num_warps=PRODUCER_WARPS, num_regs=PRODUCER_REGS):
            for k in tl.static_range(k_tiles):
                buf, phase = get_bufidx_phase(k, NUM_BUFFERS)
                tlx.barrier_wait(a_empty[buf], phase ^ 1)
                tlx.barrier_expect_bytes(a_full[buf], 2 * BLOCK_K * BLOCK_M)
                tlx.async_descriptor_load(
                    down_gradient_desc,
                    buffers_a[buf],
                    [k * BLOCK_K, start_m],
                    a_full[buf],
                )

                tlx.barrier_wait(b_empty[buf], phase ^ 1)
                for target in tl.static_range(CLUSTER_CTAS):
                    if cta_rank == target:
                        tlx.barrier_arrive(reuse_ready[buf], 1)
                    else:
                        tlx.barrier_arrive(reuse_ready[buf], 1, remote_cta_rank=target)
                tlx.barrier_wait(reuse_ready[buf], phase)
                part_start = k * BLOCK_K + cta_rank * rows_per_cta
                ks = part_start + tl.arange(0, rows_per_cta)
                columns = start_n + tl.arange(0, BLOCK_N)
                offsets = ks[:, None] * baseline.N + columns[None, :]
                gate_value = tl.load(gate + offsets).to(tl.float32)
                up_value = tl.load(up + offsets).to(tl.float32)
                hidden = gate_value * tl.sigmoid(gate_value) * up_value
                hidden = hidden.to(tl.bfloat16)
                if cta_rank == 0:
                    _publish_hidden_slice(
                        0,
                        hidden,
                        buffers_b,
                        b_full,
                        buf,
                        BLOCK_N,
                        BLOCK_K,
                        CLUSTER_CTAS,
                    )
                else:
                    _publish_hidden_slice(
                        1,
                        hidden,
                        buffers_b,
                        b_full,
                        buf,
                        BLOCK_N,
                        BLOCK_K,
                        CLUSTER_CTAS,
                    )
                for target in tl.static_range(CLUSTER_CTAS):
                    if cta_rank == target:
                        tlx.barrier_arrive(b_full[buf], 1)
                    else:
                        tlx.barrier_arrive(b_full[buf], 1, remote_cta_rank=target)


@triton.jit
def swiglu_gemm_2cta_tlx(
    down_gradient_desc,
    gate,
    up,
    dweight,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_GROUPS: tl.constexpr,
    NUM_A_BUFFERS: tl.constexpr,
    NUM_B_BUFFERS: tl.constexpr,
    PRODUCER_WARPS: tl.constexpr,
    PRODUCER_REGS: tl.constexpr,
    EPILOGUE_WARPS: tl.constexpr,
    EPILOGUE_REGS: tl.constexpr,
    EPILOGUE_SUBTILES: tl.constexpr,
):
    block_n_per_cta: tl.constexpr = BLOCK_N // 2
    buffers_a = tlx.local_alloc(
        (BLOCK_K, BLOCK_M),
        tl.bfloat16,
        M_GROUPS * NUM_A_BUFFERS,
    )
    buffers_b = tlx.local_alloc((BLOCK_K, block_n_per_cta), tl.bfloat16, NUM_B_BUFFERS)
    accumulators = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        M_GROUPS,
        tlx.storage_kind.tmem,
    )
    a_full = tlx.alloc_barriers(M_GROUPS * NUM_A_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(M_GROUPS * NUM_A_BUFFERS, arrive_count=1)
    b_full = tlx.alloc_barriers(NUM_B_BUFFERS, arrive_count=1)
    cta_ready = tlx.alloc_barriers(M_GROUPS * NUM_B_BUFFERS, arrive_count=2)
    accumulator_full = tlx.alloc_barriers(M_GROUPS, arrive_count=1)

    cta_rank = tlx.cluster_cta_rank()
    pred_leader = cta_rank == 0
    cluster_id = tl.program_id(0) // 2
    num_pid_n: tl.constexpr = baseline.N // BLOCK_N
    pid_m_group = cluster_id // num_pid_n
    pid_n = cluster_id % num_pid_n
    start_m = (pid_m_group * 2 + cta_rank) * M_GROUPS * BLOCK_M
    start_n = pid_n * BLOCK_N
    start_local_n = start_n + cta_rank * block_n_per_cta
    k_tiles: tl.constexpr = baseline.K // BLOCK_K

    with tlx.async_tasks():
        with tlx.async_task(
            "default", num_warps=EPILOGUE_WARPS, num_regs=EPILOGUE_REGS
        ):
            for group in tl.static_range(M_GROUPS):
                _store_accumulator(
                    group,
                    accumulators,
                    accumulator_full,
                    dweight,
                    start_m,
                    start_n,
                    BLOCK_M,
                    BLOCK_N,
                    EPILOGUE_SUBTILES,
                )

        with tlx.async_task(num_warps=1, num_regs=24):
            for k in tl.static_range(k_tiles):
                b_buf, b_phase = get_bufidx_phase(k, NUM_B_BUFFERS)
                a_local, a_phase = get_bufidx_phase(k, NUM_A_BUFFERS)
                tlx.barrier_wait(b_full[b_buf], b_phase)
                for group in tl.static_range(M_GROUPS):
                    a_buf = group * NUM_A_BUFFERS + a_local
                    sync_buf = group * NUM_B_BUFFERS + b_buf
                    tlx.barrier_wait(a_full[a_buf], a_phase)
                    tlx.barrier_arrive(cta_ready[sync_buf], 1, remote_cta_rank=0)
                    tlx.barrier_wait(cta_ready[sync_buf], b_phase, pred=pred_leader)
                    tlx.async_dot(
                        tlx.local_trans(buffers_a[a_buf]),
                        buffers_b[b_buf],
                        accumulators[group],
                        use_acc=k > 0,
                        mBarriers=[a_empty[a_buf]],
                        two_ctas=True,
                        out_dtype=tl.float32,
                    )
            for group in tl.static_range(M_GROUPS):
                tlx.tcgen05_commit(accumulator_full[group], two_ctas=True)

        with tlx.async_task(num_warps=1, num_regs=24):
            for k in tl.static_range(k_tiles):
                a_local, a_phase = get_bufidx_phase(k, NUM_A_BUFFERS)
                for group in tl.static_range(M_GROUPS):
                    a_buf = group * NUM_A_BUFFERS + a_local
                    tlx.barrier_wait(a_empty[a_buf], a_phase ^ 1)
                    tlx.barrier_expect_bytes(a_full[a_buf], 2 * BLOCK_K * BLOCK_M)
                    tlx.async_descriptor_load(
                        down_gradient_desc,
                        buffers_a[a_buf],
                        [k * BLOCK_K, start_m + group * BLOCK_M],
                        a_full[a_buf],
                    )

        with tlx.async_task(num_warps=PRODUCER_WARPS, num_regs=PRODUCER_REGS):
            for k in tl.static_range(k_tiles):
                b_buf, b_phase = get_bufidx_phase(k, NUM_B_BUFFERS)
                ks = k * BLOCK_K + tl.arange(0, BLOCK_K)
                columns = start_local_n + tl.arange(0, block_n_per_cta)
                offsets = ks[:, None] * baseline.N + columns[None, :]
                gate_value = tl.load(gate + offsets).to(tl.float32)
                up_value = tl.load(up + offsets).to(tl.float32)
                hidden = gate_value * tl.sigmoid(gate_value) * up_value
                last_a_local, last_a_phase = get_bufidx_phase(k, NUM_A_BUFFERS)
                last_a_buf = (M_GROUPS - 1) * NUM_A_BUFFERS + last_a_local
                tlx.barrier_wait(a_empty[last_a_buf], last_a_phase ^ 1)
                tlx.local_store(buffers_b[b_buf], hidden.to(tl.bfloat16))
                tlx.fence_async_shared()
                tlx.barrier_arrive(b_full[b_buf], 1)


CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "M_GROUPS": 4,
    "NUM_A_BUFFERS": 1,
    "NUM_B_BUFFERS": 3,
    "B_RELEASE_GROUPS": 1,
    "PRODUCER_WARPS": 16,
    "PRODUCER_REGS": 104,
    "EPILOGUE_WARPS": 4,
    "EPILOGUE_REGS": 64,
    "EPILOGUE_SUBTILES": 8,
}

CLUSTER_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "NUM_BUFFERS": 3,
    "CLUSTER_CTAS": 2,
    "PRODUCER_WARPS": 16,
    "PRODUCER_REGS": 104,
    "EPILOGUE_WARPS": 4,
    "EPILOGUE_REGS": 128,
    "EPILOGUE_SUBTILES": 4,
}

TWO_CTA_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "M_GROUPS": 2,
    "NUM_A_BUFFERS": 1,
    "NUM_B_BUFFERS": 3,
    "PRODUCER_WARPS": 16,
    "PRODUCER_REGS": 104,
    "EPILOGUE_WARPS": 4,
    "EPILOGUE_REGS": 64,
    "EPILOGUE_SUBTILES": 8,
}


def run_tlx(
    inputs: baseline.TensorMap,
    outputs: baseline.TensorMap,
    *,
    variant: str = "direct",
    config: dict[str, int] | None = None,
) -> None:
    defaults = {"direct": CONFIG, "2cta": TWO_CTA_CONFIG}
    config = defaults[variant] if config is None else config
    down_gradient_desc = TensorDescriptor(
        inputs["down_gradient"],
        inputs["down_gradient"].shape,
        inputs["down_gradient"].stride(),
        [config["BLOCK_K"], config["BLOCK_M"]],
    )
    if variant == "direct":
        grid = (
            triton.cdiv(baseline.M, config["M_GROUPS"] * config["BLOCK_M"])
            * triton.cdiv(baseline.N, config["BLOCK_N"]),
        )
        cast(Any, swiglu_gemm_tlx)[grid](
            down_gradient_desc,
            inputs["gate"],
            inputs["up"],
            outputs["dweight"],
            **config,
            num_warps=4,
            num_stages=1,
        )
    else:
        grid = (
            triton.cdiv(baseline.M, config["M_GROUPS"] * config["BLOCK_M"])
            * triton.cdiv(baseline.N, config["BLOCK_N"]),
        )
        cast(Any, swiglu_gemm_2cta_tlx)[grid](
            down_gradient_desc,
            inputs["gate"],
            inputs["up"],
            outputs["dweight"],
            **config,
            num_warps=4,
            num_stages=1,
            ctas_per_cga=(2, 1, 1),
        )


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
            for name, fn in (
                ("unfused_ms", unfused),
                ("fused_seed_ms", seed),
                ("fused_tlx_ms", tlx_kernel),
            ):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
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
    parser.add_argument("--variant", choices=("direct", "2cta"), default="direct")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")

    accuracy_by_seed = {}
    for seed_value in dict.fromkeys(args.verification_seeds):
        inputs = baseline.make_inputs(seed_value)
        reference = baseline.make_outputs()
        candidate = baseline.make_outputs()
        baseline.run_unfused(inputs, reference)
        run_tlx(inputs, candidate, variant=args.variant)
        torch.cuda.synchronize()
        accuracy_by_seed[str(seed_value)] = baseline.accuracy(reference, candidate)
    passed = all(metric["exact"] for metric in accuracy_by_seed.values())
    report: dict[str, Any] = {
        "shape_mnk": [baseline.M, baseline.N, baseline.K],
        "variant": args.variant,
        "config": {
            "direct": CONFIG,
            "2cta": TWO_CTA_CONFIG,
        }[args.variant],
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
            lambda: run_tlx(inputs, tlx_output, variant=args.variant),
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
