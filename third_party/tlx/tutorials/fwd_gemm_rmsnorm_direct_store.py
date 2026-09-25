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

"""OSS reference for the T289775012 GEMM/RMSNorm direct-store region."""

from __future__ import annotations

import math
from typing import Any

import torch
import triton
import triton.language as tl


M = 5120
N = 4096
K = 8192
NORM_WIDTH = 256
ROWS_PER_GEMM_ROW = N // NORM_WIDTH
EPS = 1e-5


@triton.jit
def add_in_place_kernel(value_ptr, residual_ptr, N_ELEMENTS: tl.constexpr):
    block: tl.constexpr = 256
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < N_ELEMENTS
    value = tl.load(value_ptr + offsets, mask=mask).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(value_ptr + offsets, (value + residual).to(tl.bfloat16), mask=mask)


@triton.jit
def rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    rstd_ptr,
    N_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    EPSILON: tl.constexpr,
):
    block_rows: tl.constexpr = 16
    row_indices = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    columns = tl.arange(0, WIDTH)
    mask = row_indices[:, None] < N_ROWS
    offsets = row_indices[:, None] * WIDTH + columns[None, :]
    value = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(value * value, axis=1) / WIDTH
    rstd = 1.0 / tl.sqrt(variance + EPSILON)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    normalized = value * rstd[:, None] * weight[None, :]
    tl.store(output_ptr + offsets, normalized.to(tl.bfloat16), mask=mask)
    tl.store(rstd_ptr + row_indices, rstd, mask=row_indices < N_ROWS)


@triton.jit
def copy_two_branches_kernel(
    branch0_ptr,
    branch1_ptr,
    destination0_ptr,
    destination1_ptr,
    N_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS_PER_GEMM_ROW: tl.constexpr,
):
    output_row = tl.program_id(0)
    columns = tl.arange(0, WIDTH)
    gemm_row = output_row // (2 * ROWS_PER_GEMM_ROW)
    branch_row = output_row % (2 * ROWS_PER_GEMM_ROW)
    from_branch1 = branch_row >= ROWS_PER_GEMM_ROW
    source_row = gemm_row * ROWS_PER_GEMM_ROW + branch_row % ROWS_PER_GEMM_ROW
    source_offset = source_row * WIDTH + columns
    branch0 = tl.load(branch0_ptr + source_offset)
    branch1 = tl.load(branch1_ptr + source_offset)
    value = tl.where(from_branch1, branch1, branch0)
    destination0_offset = output_row * WIDTH + columns
    destination1_row = gemm_row * 112 + 64 + branch_row
    destination1_offset = destination1_row * WIDTH + columns
    mask = output_row < N_ROWS
    tl.store(destination0_ptr + destination0_offset, value, mask=mask)
    tl.store(destination1_ptr + destination1_offset, value, mask=mask)


def make_inputs(seed: int = 0) -> dict[str, Any]:
    torch.manual_seed(seed)

    def scaled(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) / math.sqrt(K)

    return {
        "a": [scaled((M, K)), scaled((M, K))],
        "b": [scaled((N, K)), scaled((N, K))],
        "residual": [scaled((M, N)), scaled((M, N))],
        "weight": [scaled((NORM_WIDTH,)), scaled((NORM_WIDTH,))],
    }


def make_outputs() -> dict[str, Any]:
    return {
        "post_add": [
            torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
            torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
        ],
        "norm": [
            torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
            torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
        ],
        "rstd": [
            torch.empty(M * ROWS_PER_GEMM_ROW, device="cuda", dtype=torch.float32),
            torch.empty(M * ROWS_PER_GEMM_ROW, device="cuda", dtype=torch.float32),
        ],
        "destination0": torch.empty((M, 2 * N), device="cuda", dtype=torch.bfloat16),
        "destination1": torch.zeros(
            (M, 112 * NORM_WIDTH), device="cuda", dtype=torch.bfloat16
        ),
    }


def run_unfused(inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
    for branch in range(2):
        torch.mm(
            inputs["a"][branch], inputs["b"][branch].T, out=outputs["post_add"][branch]
        )
        add_in_place_kernel[(triton.cdiv(M * N, 256),)](
            outputs["post_add"][branch],
            inputs["residual"][branch],
            N_ELEMENTS=M * N,
            num_warps=4,
        )
        rmsnorm_kernel[(triton.cdiv(M * ROWS_PER_GEMM_ROW, 16),)](
            outputs["post_add"][branch],
            inputs["weight"][branch],
            outputs["norm"][branch],
            outputs["rstd"][branch],
            N_ROWS=M * ROWS_PER_GEMM_ROW,
            WIDTH=NORM_WIDTH,
            EPSILON=EPS,
            num_warps=4,
            num_stages=3,
        )
    copy_two_branches_kernel[(M * 2 * ROWS_PER_GEMM_ROW,)](
        outputs["norm"][0],
        outputs["norm"][1],
        outputs["destination0"],
        outputs["destination1"],
        N_ROWS=M * 2 * ROWS_PER_GEMM_ROW,
        WIDTH=NORM_WIDTH,
        ROWS_PER_GEMM_ROW=ROWS_PER_GEMM_ROW,
        num_warps=4,
    )


def accuracy(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, dict[str, float | bool]]:
    tolerances = {
        "post_add": (0.0, 0.0),
        "rstd": (5e-5, 1e-5),
        "destination0": (5e-4, 1e-3),
        "destination1": (5e-4, 1e-3),
    }
    pairs = []
    for name in ("post_add", "rstd"):
        pairs.extend(
            (f"{name}{branch}", expected[name][branch], actual[name][branch])
            for branch in range(2)
        )
    pairs.extend(
        (name, expected[name], actual[name])
        for name in ("destination0", "destination1")
    )
    result = {}
    for label, reference, candidate in pairs:
        tolerance_name = label[:-1] if label.startswith(("post_add", "rstd")) else label
        atol, rtol = tolerances[tolerance_name]
        difference = candidate.float() - reference.float()
        denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
        result[label] = {
            "passed": bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol)),
            "relative_l2": float(torch.linalg.vector_norm(difference) / denominator),
            "max_abs": float(difference.abs().max()),
        }
    return result
