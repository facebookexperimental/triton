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

"""Accuracy and profiler-time benchmark for the saved-projection SwiGLU GEMM."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from typing import Any

import torch

from fwd_swiglu_gemm_saved_tlx import fwd_swiglu_gemm_saved


M = 5120
TWO_N = 16384
K = 4096


def reference(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    projection = torch.matmul(a, b.t())
    gate, up = projection.chunk(2, dim=1)
    gate_fp32 = gate.float()
    output = (gate_fp32 * torch.sigmoid(gate_fp32) * up.float()).to(torch.bfloat16)
    return projection, output


def make_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((TWO_N, K), device="cuda", dtype=torch.bfloat16) / math.sqrt(K)
    return a, b


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = actual.float() - expected.float()
    expected_norm = torch.linalg.vector_norm(expected.float()).item()
    return {
        "max_abs": difference.abs().max().item(),
        "rel_l2": torch.linalg.vector_norm(difference).item()
        / max(expected_norm, 1e-12),
        "violations": int(
            (~torch.isclose(actual, expected, atol=1e-2, rtol=1e-2)).sum().item()
        ),
    }


def profiler_samples(
    function: Callable[[], Any], warmup: int, repetitions: int, blocks: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    quotient, remainder = divmod(repetitions, blocks)
    block_iterations = [quotient + (index < remainder) for index in range(blocks)]
    samples = []
    for inner in block_iterations:
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            for _ in range(inner):
                function()
            torch.cuda.synchronize()
        total_us = 0.0
        for event in profile.key_averages():
            value = getattr(event, "self_device_time_total", None)
            if value is None:
                value = getattr(event, "self_cuda_time_total", 0.0)
            total_us += float(value or 0.0)
        samples.append(total_us / 1000.0 / inner)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    args = parser.parse_args()

    accuracy: dict[str, Any] = {}
    for seed in args.seeds:
        a, b = make_inputs(seed)
        expected = reference(a, b)
        actual = fwd_swiglu_gemm_saved(a, b)
        torch.cuda.synchronize()
        accuracy[str(seed)] = {
            "projection": error_metrics(actual[0], expected[0]),
            "output": error_metrics(actual[1], expected[1]),
        }

    a, b = make_inputs(0)
    compiled_reference = torch.compile(reference, fullgraph=True)
    for _ in range(5):
        compiled_reference(a, b)
    torch.cuda.synchronize()

    fused_samples: list[float] = []
    unfused_samples: list[float] = []
    for _ in range(args.repeat):
        fused_samples.extend(
            profiler_samples(
                lambda: fwd_swiglu_gemm_saved(a, b),
                args.warmup,
                args.repetitions,
                args.blocks,
            )
        )
        unfused_samples.extend(
            profiler_samples(
                lambda: compiled_reference(a, b),
                args.warmup,
                args.repetitions,
                args.blocks,
            )
        )

    fused_ms = statistics.mean(fused_samples)
    unfused_ms = statistics.mean(unfused_samples)
    print(
        json.dumps(
            {
                "shape": [M, TWO_N, K],
                "accuracy": accuracy,
                "fused_ms": fused_ms,
                "unfused_compiled_ms": unfused_ms,
                "speedup": unfused_ms / fused_ms,
                "fused_samples_ms": fused_samples,
                "unfused_samples_ms": unfused_samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
