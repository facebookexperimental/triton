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

"""OSS reproduction of the T289757919 batched GEMM-concat fusion."""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, TypedDict, cast

import torch
import triton
import triton.language as tl


BATCH = 5120
M = 32
N = 64
K = 256
FEATURE_WIDTH = 4096
EXTRA_WIDTH = 2328
OUTPUT_WIDTH = M * N + FEATURE_WIDTH + EXTRA_WIDTH
PADDED_WIDTH = 8512


class TensorMap(TypedDict):
    a: torch.Tensor
    b: torch.Tensor
    features: torch.Tensor
    extra: torch.Tensor
    projection: torch.Tensor
    output: torch.Tensor
    padded: torch.Tensor


@triton.jit
def concat_copy(
    projection,
    features,
    extra,
    output,
    padded_output,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    FEATURE_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < BATCH * OUTPUT_WIDTH
    batch = offsets // OUTPUT_WIDTH
    column = offsets % OUTPUT_WIDTH
    channel = column
    value = tl.load(
        projection + batch * (M * N) + channel,
        mask=mask & (column < M * N),
        other=0.0,
    )
    value = tl.where(
        (column >= M * N) & (column < M * N + FEATURE_WIDTH),
        tl.load(
            features + batch * 20480 + 16384 + column - M * N,
            mask=mask & (column >= M * N) & (column < M * N + FEATURE_WIDTH),
            other=0.0,
        ),
        value,
    )
    value = tl.where(
        column >= M * N + FEATURE_WIDTH,
        tl.load(
            extra + batch * EXTRA_WIDTH + column - M * N - FEATURE_WIDTH,
            mask=mask & (column >= M * N + FEATURE_WIDTH),
            other=0.0,
        ),
        value,
    )
    tl.store(output + offsets, value, mask=mask)
    tl.store(padded_output + batch * PADDED_WIDTH + column, value, mask=mask)


@triton.jit
def fused_seed(
    a,
    b,
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    tiles_per_batch: tl.constexpr = num_pid_m * num_pid_n
    for tile_id in range(start_pid, BATCH * tiles_per_batch, NUM_SMS):
        batch = tile_id // tiles_per_batch
        tile_in_batch = tile_id % tiles_per_batch
        pid_m = tile_in_batch // num_pid_n
        pid_n = tile_in_batch % num_pid_n
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        safe_rows = tl.where(rows < M, rows, 0)
        safe_columns = tl.where(columns < N, columns, 0)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start_k in range(0, tl.cdiv(K, BLOCK_K)):
            ks = start_k * BLOCK_K + tl.arange(0, BLOCK_K)
            a_tile = tl.load(
                a + batch * M * K + safe_rows[:, None] * K + ks[None, :],
                mask=(rows[:, None] < M) & (ks[None, :] < K),
                other=0.0,
            )
            b_tile = tl.load(
                b + batch * K * N + ks[:, None] * N + safe_columns[None, :],
                mask=(ks[:, None] < K) & (columns[None, :] < N),
                other=0.0,
            )
            accumulator = tl.dot(a_tile, b_tile, accumulator, allow_tf32=False)
        mask = (rows[:, None] < M) & (columns[None, :] < N)
        channel = rows[:, None] * N + columns[None, :]
        value = accumulator.to(tl.bfloat16)
        base = batch * OUTPUT_WIDTH
        padded_base = batch * PADDED_WIDTH
        tl.store(output + base + channel, value, mask)
        tl.store(padded_output + padded_base + channel, value, mask)
        feature0 = tl.load(features + batch * 20480 + 16384 + channel, mask=mask)
        feature1 = tl.load(features + batch * 20480 + 18432 + channel, mask=mask)
        extra0 = tl.load(extra + batch * EXTRA_WIDTH + channel, mask=mask)
        extra1 = tl.load(
            extra + batch * EXTRA_WIDTH + 2048 + channel,
            mask=mask & (channel < 280),
            other=0.0,
        )
        tl.store(output + base + 2048 + channel, feature0, mask)
        tl.store(output + base + 4096 + channel, feature1, mask)
        tl.store(output + base + 6144 + channel, extra0, mask)
        tl.store(output + base + 8192 + channel, extra1, mask & (channel < 280))
        tl.store(padded_output + padded_base + 2048 + channel, feature0, mask)
        tl.store(padded_output + padded_base + 4096 + channel, feature1, mask)
        tl.store(padded_output + padded_base + 6144 + channel, extra0, mask)
        tl.store(
            padded_output + padded_base + 8192 + channel,
            extra1,
            mask & (channel < 280),
        )


@triton.jit
def fused_exact_batch(
    a,
    b,
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
    BLOCK_K: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    for batch in range(tl.program_id(0), BATCH, NUM_PROGRAMS):
        rows = tl.arange(0, M)
        columns = tl.arange(0, N)
        accumulator = tl.zeros((M, N), tl.float32)
        for start_k in range(0, tl.cdiv(K, BLOCK_K)):
            ks = start_k * BLOCK_K + tl.arange(0, BLOCK_K)
            a_tile = tl.load(
                a + batch * M * K + rows[:, None] * K + ks[None, :],
                mask=ks[None, :] < K,
                other=0.0,
            )
            b_tile = tl.load(
                b + batch * K * N + ks[:, None] * N + columns[None, :],
                mask=ks[:, None] < K,
                other=0.0,
            )
            accumulator = tl.dot(a_tile, b_tile, accumulator, allow_tf32=False)

        channel = rows[:, None] * N + columns[None, :]
        base = batch * OUTPUT_WIDTH
        padded_base = batch * PADDED_WIDTH
        gemm_value = accumulator.to(tl.bfloat16)
        tl.store(output + base + channel, gemm_value)
        tl.store(padded_output + padded_base + channel, gemm_value)

        copy_offsets = tl.arange(0, COPY_BLOCK)
        for start in tl.static_range(0, FEATURE_WIDTH, COPY_BLOCK):
            offsets = start + copy_offsets
            mask = offsets < FEATURE_WIDTH
            feature_value = tl.load(
                features + batch * 20480 + 16384 + offsets,
                mask=mask,
                other=0.0,
            )
            tl.store(output + base + M * N + offsets, feature_value, mask=mask)
            tl.store(
                padded_output + padded_base + M * N + offsets,
                feature_value,
                mask=mask,
            )
        for start in tl.static_range(0, 4096, COPY_BLOCK):
            offsets = start + copy_offsets
            mask = offsets < EXTRA_WIDTH
            extra_value = tl.load(
                extra + batch * EXTRA_WIDTH + offsets,
                mask=mask,
                other=0.0,
            )
            tl.store(
                output + base + M * N + FEATURE_WIDTH + offsets,
                extra_value,
                mask=mask,
            )
            tl.store(
                padded_output + padded_base + M * N + FEATURE_WIDTH + offsets,
                extra_value,
                mask=mask,
            )


@triton.jit
def fused_parallel_batch(
    a,
    b,
    features,
    extra,
    output,
    padded_output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FEATURE_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
):
    worker = tl.program_id(0)
    batch = tl.program_id(1)
    base = batch * OUTPUT_WIDTH
    padded_base = batch * PADDED_WIDTH

    # The GEMM and concat inputs are independent and populate disjoint output
    # ranges. Give them separate CTAs so the memory-only tail does not extend
    # the live range of the accumulator or serialize behind the contraction.
    if worker == 0:
        rows = tl.arange(0, M)
        columns = tl.arange(0, N)
        accumulator = tl.zeros((M, N), tl.float32)
        for start_k in range(0, tl.cdiv(K, BLOCK_K)):
            ks = start_k * BLOCK_K + tl.arange(0, BLOCK_K)
            a_tile = tl.load(
                a + batch * M * K + rows[:, None] * K + ks[None, :],
                mask=ks[None, :] < K,
                other=0.0,
            )
            b_tile = tl.load(
                b + batch * K * N + ks[:, None] * N + columns[None, :],
                mask=ks[:, None] < K,
                other=0.0,
            )
            accumulator = tl.dot(a_tile, b_tile, accumulator, allow_tf32=False)
        channel = rows[:, None] * N + columns[None, :]
        value = accumulator.to(tl.bfloat16)
        tl.store(output + base + channel, value)
        tl.store(padded_output + padded_base + channel, value)
    else:
        tail_offset = (worker - 1) * COPY_BLOCK + tl.arange(0, COPY_BLOCK)
        tail_width: tl.constexpr = FEATURE_WIDTH + EXTRA_WIDTH
        valid = tail_offset < tail_width
        feature_mask = tail_offset < FEATURE_WIDTH
        extra_offset = tail_offset - FEATURE_WIDTH
        extra_mask = valid & ~feature_mask
        copy_value = tl.load(
            features + batch * 20480 + 16384 + tail_offset,
            mask=feature_mask,
            other=0.0,
        )
        copy_value = tl.where(
            extra_mask,
            tl.load(
                extra + batch * EXTRA_WIDTH + extra_offset,
                mask=extra_mask,
                other=0.0,
            ),
            copy_value,
        )
        column = M * N + tail_offset
        tl.store(output + base + column, copy_value, mask=valid)
        tl.store(padded_output + padded_base + column, copy_value, mask=valid)


@triton.jit
def _store_direct_concat_tail(
    features,
    extra,
    output,
    padded_output,
    batch,
    M: tl.constexpr,
    N: tl.constexpr,
    FEATURE_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    COPY_LOAD_CACHE: tl.constexpr,
    OUTPUT_STORE_CACHE: tl.constexpr,
):
    rows = tl.arange(0, M)
    columns = tl.arange(0, N)
    channel = rows[:, None] * N + columns[None, :]
    base = batch * OUTPUT_WIDTH
    padded_base = batch * PADDED_WIDTH
    feature0 = tl.load(
        features + batch * 20480 + 16384 + channel,
        cache_modifier=COPY_LOAD_CACHE,
    )
    feature1 = tl.load(
        features + batch * 20480 + 18432 + channel,
        cache_modifier=COPY_LOAD_CACHE,
    )
    extra0 = tl.load(
        extra + batch * EXTRA_WIDTH + channel,
        cache_modifier=COPY_LOAD_CACHE,
    )
    extra1_offsets = tl.arange(0, 512)
    extra1_mask = extra1_offsets < EXTRA_WIDTH - M * N
    extra1 = tl.load(
        extra + batch * EXTRA_WIDTH + M * N + extra1_offsets,
        mask=extra1_mask,
        other=0.0,
        cache_modifier=COPY_LOAD_CACHE,
    )
    tl.store(
        output + base + M * N + channel,
        feature0,
        cache_modifier=OUTPUT_STORE_CACHE,
    )
    tl.store(
        output + base + 2 * M * N + channel,
        feature1,
        cache_modifier=OUTPUT_STORE_CACHE,
    )
    tl.store(
        output + base + M * N + FEATURE_WIDTH + channel,
        extra0,
        cache_modifier=OUTPUT_STORE_CACHE,
    )
    tl.store(
        output + base + 2 * M * N + FEATURE_WIDTH + extra1_offsets,
        extra1,
        mask=extra1_mask,
        cache_modifier=OUTPUT_STORE_CACHE,
    )
    tl.store(
        padded_output + padded_base + M * N + channel,
        feature0,
        cache_modifier=OUTPUT_STORE_CACHE,
    )
    tl.store(
        padded_output + padded_base + 2 * M * N + channel,
        feature1,
        cache_modifier=OUTPUT_STORE_CACHE,
    )
    tl.store(
        padded_output + padded_base + M * N + FEATURE_WIDTH + channel,
        extra0,
        cache_modifier=OUTPUT_STORE_CACHE,
    )
    tl.store(
        padded_output + padded_base + 2 * M * N + FEATURE_WIDTH + extra1_offsets,
        extra1,
        mask=extra1_mask,
        cache_modifier=OUTPUT_STORE_CACHE,
    )


@triton.jit
def fused_direct_batch(
    a,
    b,
    features,
    extra,
    output,
    padded_output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FEATURE_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COPY_FIRST: tl.constexpr,
    GEMM_LOAD_CACHE: tl.constexpr,
    COPY_LOAD_CACHE: tl.constexpr,
    OUTPUT_STORE_CACHE: tl.constexpr,
):
    batch = tl.program_id(0)
    if COPY_FIRST:
        _store_direct_concat_tail(
            features,
            extra,
            output,
            padded_output,
            batch,
            M,
            N,
            FEATURE_WIDTH,
            EXTRA_WIDTH,
            OUTPUT_WIDTH,
            PADDED_WIDTH,
            COPY_LOAD_CACHE,
            OUTPUT_STORE_CACHE,
        )

    rows = tl.arange(0, M)
    columns = tl.arange(0, N)
    accumulator = tl.zeros((M, N), tl.float32)
    for start_k in range(0, tl.cdiv(K, BLOCK_K)):
        ks = start_k * BLOCK_K + tl.arange(0, BLOCK_K)
        a_tile = tl.load(
            a + batch * M * K + rows[:, None] * K + ks[None, :],
            mask=ks[None, :] < K,
            other=0.0,
            cache_modifier=GEMM_LOAD_CACHE,
        )
        b_tile = tl.load(
            b + batch * K * N + ks[:, None] * N + columns[None, :],
            mask=ks[:, None] < K,
            other=0.0,
            cache_modifier=GEMM_LOAD_CACHE,
        )
        accumulator = tl.dot(a_tile, b_tile, accumulator, allow_tf32=False)

    channel = rows[:, None] * N + columns[None, :]
    base = batch * OUTPUT_WIDTH
    padded_base = batch * PADDED_WIDTH
    gemm_value = accumulator.to(tl.bfloat16)
    tl.store(
        output + base + channel,
        gemm_value,
        cache_modifier=OUTPUT_STORE_CACHE,
    )
    tl.store(
        padded_output + padded_base + channel,
        gemm_value,
        cache_modifier=OUTPUT_STORE_CACHE,
    )

    if not COPY_FIRST:
        _store_direct_concat_tail(
            features,
            extra,
            output,
            padded_output,
            batch,
            M,
            N,
            FEATURE_WIDTH,
            EXTRA_WIDTH,
            OUTPUT_WIDTH,
            PADDED_WIDTH,
            COPY_LOAD_CACHE,
            OUTPUT_STORE_CACHE,
        )


def make_inputs(seed: int = 0) -> TensorMap:
    torch.manual_seed(seed)
    return cast(
        TensorMap,
        {
            "a": torch.randn((BATCH, M, K), device="cuda", dtype=torch.bfloat16)
            / math.sqrt(K),
            "b": torch.randn((BATCH, K, N), device="cuda", dtype=torch.bfloat16)
            / math.sqrt(K),
            "features": torch.randn(
                (BATCH, 80, 256), device="cuda", dtype=torch.bfloat16
            )
            / math.sqrt(256),
            "extra": torch.randn(
                (BATCH, EXTRA_WIDTH), device="cuda", dtype=torch.bfloat16
            )
            / math.sqrt(EXTRA_WIDTH),
        },
    )


def make_outputs() -> TensorMap:
    return cast(
        TensorMap,
        {
            "projection": torch.empty(
                (BATCH, M, N), device="cuda", dtype=torch.bfloat16
            ),
            "output": torch.empty(
                (BATCH, OUTPUT_WIDTH), device="cuda", dtype=torch.bfloat16
            ),
            "padded": torch.empty(
                (BATCH, PADDED_WIDTH), device="cuda", dtype=torch.bfloat16
            )[:, :OUTPUT_WIDTH],
        },
    )


def run_unfused(inputs: TensorMap, outputs: TensorMap) -> None:
    torch.bmm(inputs["a"], inputs["b"], out=outputs["projection"])
    cast(Any, concat_copy)[(triton.cdiv(BATCH * OUTPUT_WIDTH, 256),)](
        outputs["projection"],
        inputs["features"],
        inputs["extra"],
        outputs["output"],
        outputs["padded"],
        BATCH=BATCH,
        M=M,
        N=N,
        FEATURE_WIDTH=FEATURE_WIDTH,
        EXTRA_WIDTH=EXTRA_WIDTH,
        OUTPUT_WIDTH=OUTPUT_WIDTH,
        PADDED_WIDTH=PADDED_WIDTH,
        BLOCK=256,
        num_warps=8,
    )


def run_seed(inputs: TensorMap, outputs: TensorMap) -> None:
    cast(Any, fused_seed)[(_num_sms(),)](
        inputs["a"],
        inputs["b"],
        inputs["features"],
        inputs["extra"],
        outputs["output"],
        outputs["padded"],
        BATCH=BATCH,
        M=M,
        N=N,
        K=K,
        FEATURE_WIDTH=FEATURE_WIDTH,
        EXTRA_WIDTH=EXTRA_WIDTH,
        OUTPUT_WIDTH=OUTPUT_WIDTH,
        PADDED_WIDTH=PADDED_WIDTH,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        NUM_SMS=_num_sms(),
        num_stages=3,
        num_warps=8,
    )


def run_exact_batch(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_k: int = 64,
    copy_block: int = 512,
    num_programs: int | None = None,
    num_warps: int = 8,
    num_stages: int = 3,
    maxnreg: int | None = None,
) -> None:
    if num_programs is None:
        num_programs = BATCH
    launch_options: dict[str, int] = {
        "num_stages": num_stages,
        "num_warps": num_warps,
    }
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    cast(Any, fused_exact_batch)[(num_programs,)](
        inputs["a"],
        inputs["b"],
        inputs["features"],
        inputs["extra"],
        outputs["output"],
        outputs["padded"],
        BATCH=BATCH,
        M=M,
        N=N,
        K=K,
        FEATURE_WIDTH=FEATURE_WIDTH,
        EXTRA_WIDTH=EXTRA_WIDTH,
        OUTPUT_WIDTH=OUTPUT_WIDTH,
        PADDED_WIDTH=PADDED_WIDTH,
        BLOCK_K=block_k,
        COPY_BLOCK=copy_block,
        NUM_PROGRAMS=num_programs,
        **launch_options,
    )


def run_parallel_batch(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_k: int = 64,
    copy_block: int = 2048,
    num_warps: int = 4,
    num_stages: int = 3,
    maxnreg: int | None = None,
) -> None:
    launch_options: dict[str, int] = {
        "num_stages": num_stages,
        "num_warps": num_warps,
    }
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    num_copy_workers = triton.cdiv(FEATURE_WIDTH + EXTRA_WIDTH, copy_block)
    cast(Any, fused_parallel_batch)[(1 + num_copy_workers, BATCH)](
        inputs["a"],
        inputs["b"],
        inputs["features"],
        inputs["extra"],
        outputs["output"],
        outputs["padded"],
        M=M,
        N=N,
        K=K,
        FEATURE_WIDTH=FEATURE_WIDTH,
        EXTRA_WIDTH=EXTRA_WIDTH,
        OUTPUT_WIDTH=OUTPUT_WIDTH,
        PADDED_WIDTH=PADDED_WIDTH,
        BLOCK_K=block_k,
        COPY_BLOCK=copy_block,
        **launch_options,
    )


def run_direct_batch(
    inputs: TensorMap,
    outputs: TensorMap,
    *,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 2,
    maxnreg: int | None = None,
    copy_first: bool = True,
    gemm_load_cache: str = "",
    copy_load_cache: str = "",
    output_store_cache: str = "",
) -> None:
    launch_options: dict[str, int] = {
        "num_stages": num_stages,
        "num_warps": num_warps,
    }
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    cast(Any, fused_direct_batch)[(BATCH,)](
        inputs["a"],
        inputs["b"],
        inputs["features"],
        inputs["extra"],
        outputs["output"],
        outputs["padded"],
        M=M,
        N=N,
        K=K,
        FEATURE_WIDTH=FEATURE_WIDTH,
        EXTRA_WIDTH=EXTRA_WIDTH,
        OUTPUT_WIDTH=OUTPUT_WIDTH,
        PADDED_WIDTH=PADDED_WIDTH,
        BLOCK_K=block_k,
        COPY_FIRST=copy_first,
        GEMM_LOAD_CACHE=gemm_load_cache,
        COPY_LOAD_CACHE=copy_load_cache,
        OUTPUT_STORE_CACHE=output_store_cache,
        **launch_options,
    )


def _num_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def accuracy(
    reference: TensorMap, candidate: TensorMap
) -> dict[str, dict[str, float | bool]]:
    result = {}
    for name in ("output", "padded"):
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
