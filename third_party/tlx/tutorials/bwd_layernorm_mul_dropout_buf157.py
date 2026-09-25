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

"""Standalone OSS benchmark for the RLLayer buf157 backward fusion.

This is the no-dropout, LayerNorm, concat-SiLU(U) specialization of the
original RLLayer benchmark.  It has no fbsource or Buck dependencies: its only
runtime dependencies are PyTorch and the Triton checkout being evaluated.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable, cast

import torch
import triton
import triton.language as tl


FEATURES = 256
CONCAT_FEATURES = 512
GRADIENT_FEATURES = 256
SPLIT_K = 128
BLOCK_K = 64
DEFAULT_ROWS = 2_097_152


@triton.jit
def _layernorm_mul_bwd_dx_du(
    dx_ptr,
    du_ptr,
    dy_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    x_ptr,
    u_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    D,
    N,
    BLOCK_D: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    """Specialized RLLayer backward used by the benchmark.

    The source configuration has training=False, dropout=0, concat_u=True,
    silu_u=True, concat_x=False, and mul_u_activation_type="none".
    """
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1
    if rows_per_tile == 0:
        return

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    row = pid
    row_i64 = row.to(tl.int64)
    tile_num_i64 = tile_num.to(tl.int64)
    x_ptr += row_i64 * stride_x
    u_ptr += row_i64 * stride_u
    dy_ptr += row_i64 * stride_dy
    dx_ptr += row_i64 * stride_dx
    du_ptr += row_i64 * stride_du
    if COMPUTE_Y:
        y_ptr += row_i64 * stride_y
    partial_dw_ptr += pid.to(tl.int64) * D + cols
    partial_db_ptr += pid.to(tl.int64) * D + cols

    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask).to(tl.float32)

    for _ in range(0, rows_per_tile):
        x = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)
        u = tl.load(u_ptr + cols, mask=mask, other=0).to(tl.float32)
        du_concat = tl.load(dy_ptr + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(dy_ptr + D + cols, mask=mask, other=0).to(tl.float32)
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        xhat = (x - mean) * rstd
        layernorm = xhat * weight + bias

        sigmoid_u = tl.sigmoid(u)
        silu_u = u * sigmoid_u
        dsilu_u = sigmoid_u + silu_u * (1.0 - sigmoid_u)
        du = dy * layernorm + du_concat * dsilu_u
        tl.store(du_ptr + cols, du.to(du_ptr.dtype.element_ty), mask=mask)

        layernorm_dy = dy * u
        weighted_dy = weight * layernorm_dy
        xhat = tl.where(mask, xhat, 0.0)
        weighted_dy = tl.where(mask, weighted_dy, 0.0)
        c1 = tl.sum(xhat * weighted_dy, axis=0) / D
        c2 = tl.sum(weighted_dy, axis=0) / D
        dx = (weighted_dy - (xhat * c1 + c2)) * rstd
        tl.store(dx_ptr + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)

        if COMPUTE_Y:
            tl.store(y_ptr + cols, silu_u.to(y_ptr.dtype.element_ty), mask=mask)
            tl.store(
                y_ptr + D + cols,
                (layernorm * u).to(y_ptr.dtype.element_ty),
                mask=mask,
            )
            y_ptr += tile_num_i64 * stride_y

        partial_dw += layernorm_dy * xhat
        partial_db += layernorm_dy
        x_ptr += tile_num_i64 * stride_x
        u_ptr += tile_num_i64 * stride_u
        dy_ptr += tile_num_i64 * stride_dy
        dx_ptr += tile_num_i64 * stride_dx
        du_ptr += tile_num_i64 * stride_du
        row += tile_num

    tl.store(partial_dw_ptr, partial_dw, mask=mask)
    tl.store(partial_db_ptr, partial_db, mask=mask)


def _bwd_dwdb_configs() -> list[triton.Config]:
    configs = []
    max_warps = (8, 16) if torch.version.hip is not None else (8, 16, 32)
    for block_n in (32, 64, 128, 256):
        for num_warps in max_warps:
            configs.append(triton.Config({"BLOCK_N": block_n}, num_warps=num_warps))
    return configs


@triton.autotune(configs=_bwd_dwdb_configs(), key=["D"])
@triton.jit
def _finish_layernorm_dwdb(
    partial_dw_ptr,
    partial_db_ptr,
    dw_ptr,
    db_ptr,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offsets = rows[:, None] * D + cols[None, :]
        dw += tl.load(partial_dw_ptr + offsets, mask=mask, other=0.0)
        db += tl.load(partial_db_ptr + offsets, mask=mask, other=0.0)
    tl.store(dw_ptr + cols, tl.sum(dw, axis=0).to(dw_ptr.dtype.element_ty), mask=cols < D)
    tl.store(db_ptr + cols, tl.sum(db, axis=0).to(db_ptr.dtype.element_ty), mask=cols < D)


@triton.jit
def _layernorm_mul_dweight_partial(
    x_ptr,
    u_ptr,
    gradient_ptr,
    partial_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    split = tl.program_id(2)
    features = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    base_features = features % D
    output_columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gamma = tl.load(gamma_ptr + base_features).to(tl.float32)
    beta = tl.load(beta_ptr + base_features).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    split_start = split * (K // SPLIT_K)
    for block in range(0, K // SPLIT_K // BLOCK_K):
        rows = split_start + block * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rows[:, None] * D + base_features[None, :]).to(tl.float32)
        u = tl.load(u_ptr + rows[:, None] * D + base_features[None, :]).to(tl.float32)
        mean = tl.load(mean_ptr + rows).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows).to(tl.float32)
        normalized = (x - mean[:, None]) * rstd[:, None]
        normalized = normalized * gamma[None, :] + beta[None, :]
        silu_u = u * tl.sigmoid(u)
        activation = tl.where(features[None, :] < D, silu_u, normalized * u)
        activation = activation.to(tl.bfloat16)
        gradient = tl.load(gradient_ptr + rows[:, None] * N + output_columns[None, :])
        accumulator = tl.dot(activation.T, gradient, accumulator, allow_tf32=False)
    offsets = split * M * N + features[:, None] * N + output_columns[None, :]
    tl.store(partial_ptr + offsets, accumulator)


@triton.jit
def _finish_dweight_splitk(
    partial_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M * N
    total = tl.zeros((BLOCK,), tl.float32)
    for split in range(0, SPLIT_K):
        total += tl.load(partial_ptr + split * M * N + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, total.to(tl.bfloat16), mask=mask)


def _run_backward(
    inputs: dict[str, torch.Tensor], compute_y: bool
) -> tuple[torch.Tensor, ...]:
    x = inputs["x"]
    rows = x.shape[0]
    dx = torch.empty_like(x)
    du = torch.empty_like(inputs["u"])
    y = (
        torch.empty((rows, CONCAT_FEATURES), dtype=x.dtype, device=x.device)
        if compute_y
        else torch.empty(0, dtype=x.dtype, device=x.device)
    )
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tile_num = max(1, min(sms * 64, rows // 4))
    partial_dw = torch.empty((tile_num, FEATURES), dtype=torch.float32, device=x.device)
    partial_db = torch.empty_like(partial_dw)
    dweight = torch.empty((FEATURES,), dtype=x.dtype, device=x.device)
    dbias = torch.empty_like(dweight)

    cast(Any, _layernorm_mul_bwd_dx_du)[(tile_num,)](
        dx,
        du,
        inputs["dy"],
        partial_dw,
        partial_db,
        x,
        inputs["u"],
        y if compute_y else None,
        inputs["gamma"],
        inputs["beta"],
        inputs["mean"],
        inputs["rstd"],
        dx.stride(0),
        du.stride(0),
        inputs["dy"].stride(0),
        x.stride(0),
        inputs["u"].stride(0),
        y.stride(0) if compute_y else 0,
        D=FEATURES,
        N=rows,
        BLOCK_D=FEATURES,
        COMPUTE_Y=compute_y,
        num_warps=1,
    )

    blocks = triton.next_power_of_2(sms * 4)
    block_d = triton.next_power_of_2(triton.cdiv(FEATURES, blocks))
    block_d = min(max(block_d, 4), 128)
    cast(Any, _finish_layernorm_dwdb)[(triton.cdiv(FEATURES, block_d),)](
        partial_dw,
        partial_db,
        dweight,
        dbias,
        tile_num,
        D=FEATURES,
        BLOCK_D=block_d,
    )
    return dx, du, dweight, dbias, y


def _make_inputs(rows: int) -> dict[str, torch.Tensor]:
    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    x = rand(rows, FEATURES)
    mean = x.float().mean(dim=1)
    rstd = torch.rsqrt((x.float() - mean[:, None]).square().mean(dim=1) + 1e-6)
    return {
        "dy": rand(rows, CONCAT_FEATURES),
        "x": x,
        "u": rand(rows, FEATURES),
        "gamma": rand(FEATURES),
        "beta": rand(FEATURES),
        "mean": mean,
        "rstd": rstd,
        "gradient": rand(rows, GRADIENT_FEATURES),
    }


def _make_outputs() -> dict[str, torch.Tensor]:
    return {
        "projection_dweight": torch.empty(
            (CONCAT_FEATURES, GRADIENT_FEATURES), device="cuda", dtype=torch.bfloat16
        ),
        "partial": torch.empty(
            (SPLIT_K, CONCAT_FEATURES, GRADIENT_FEATURES),
            device="cuda",
            dtype=torch.float32,
        ),
    }


def _record_backward_outputs(
    outputs: dict[str, torch.Tensor], result: tuple[torch.Tensor, ...]
) -> None:
    outputs.update(zip(("dx", "du", "dweight", "dbias"), result[:4]))


def _run_unfused(inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
    result = _run_backward(inputs, True)
    _record_backward_outputs(outputs, result)
    torch.mm(result[4].T, inputs["gradient"], out=outputs["projection_dweight"])


def _run_fused(inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
    rows = inputs["x"].shape[0]
    result = _run_backward(inputs, False)
    _record_backward_outputs(outputs, result)
    cast(Any, _layernorm_mul_dweight_partial)[(8, 2, SPLIT_K)](
        inputs["x"],
        inputs["u"],
        inputs["gradient"],
        outputs["partial"],
        inputs["gamma"],
        inputs["beta"],
        inputs["mean"],
        inputs["rstd"],
        CONCAT_FEATURES,
        GRADIENT_FEATURES,
        rows,
        FEATURES,
        SPLIT_K,
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=BLOCK_K,
        num_stages=3,
        num_warps=8,
    )
    cast(Any, _finish_dweight_splitk)[
        (triton.cdiv(CONCAT_FEATURES * GRADIENT_FEATURES, 256),)
    ](
        outputs["partial"],
        outputs["projection_dweight"],
        CONCAT_FEATURES,
        GRADIENT_FEATURES,
        SPLIT_K,
        BLOCK=256,
        num_warps=8,
    )


COMPARED_OUTPUTS = ("dx", "du", "dweight", "dbias", "projection_dweight")


def _accuracy(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> dict[str, dict[str, Any]]:
    result = {}
    for name in COMPARED_OUTPUTS:
        expected = reference[name]
        actual = candidate[name]
        difference = (actual.float() - expected.float()).abs()
        denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)
        relative_l2 = float(torch.linalg.vector_norm(difference) / denominator)
        relative_l2_limit = 1e-3 if name == "projection_dweight" else None
        allclose = bool(torch.allclose(actual, expected, atol=2e-2, rtol=2e-2))
        result[name] = {
            "allclose": allclose,
            "atol": 2e-2,
            "max_abs": float(difference.max()),
            "passed": allclose
            or (relative_l2_limit is not None and relative_l2 <= relative_l2_limit),
            "relative_l2": relative_l2,
            "relative_l2_limit": relative_l2_limit,
            "rtol": 2e-2,
        }
    return result


def _validate_seed(rows: int, seed: int) -> dict[str, dict[str, Any]]:
    torch.manual_seed(seed)
    inputs = _make_inputs(rows)
    reference = _make_outputs()
    candidate = _make_outputs()
    _run_unfused(inputs, reference)
    _run_fused(inputs, candidate)
    torch.cuda.synchronize()
    return _accuracy(reference, candidate)


def _time_us(call: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000.0


def _measure(
    unfused: Callable[[], None],
    fused: Callable[[], None],
    warmup: int,
    samples: int,
    reps: int,
) -> dict[str, Any]:
    unfused_medians = []
    fused_medians = []
    for repetition in range(reps):
        for _ in range(warmup):
            unfused()
            fused()
        torch.cuda.synchronize()
        unfused_samples = []
        fused_samples = []
        for sample in range(samples):
            calls = ((unfused, unfused_samples), (fused, fused_samples))
            if (repetition + sample) % 2:
                calls = tuple(reversed(calls))
            for call, destination in calls:
                destination.append(_time_us(call))
        unfused_medians.append(statistics.median(unfused_samples))
        fused_medians.append(statistics.median(fused_samples))
    unfused_us = min(unfused_medians)
    fused_us = min(fused_medians)
    return {
        "unfused_us": unfused_us,
        "fused_us": fused_us,
        "saved_us": unfused_us - fused_us,
        "speedup": unfused_us / fused_us,
        "unfused_rep_medians_us": unfused_medians,
        "fused_rep_medians_us": fused_medians,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--verification-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.reps) < 1:
        parser.error("--warmup, --samples, and --reps must be positive")
    if args.rows < SPLIT_K * BLOCK_K or args.rows % (SPLIT_K * BLOCK_K):
        parser.error(f"--rows must be a positive multiple of {SPLIT_K * BLOCK_K}")
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")

    torch.manual_seed(0)
    inputs = _make_inputs(args.rows)
    reference = _make_outputs()
    candidate = _make_outputs()

    def unfused() -> None:
        _run_unfused(inputs, reference)

    def fused() -> None:
        _run_fused(inputs, candidate)

    unfused()
    fused()
    torch.cuda.synchronize()
    accuracy = _accuracy(reference, candidate)
    accuracy_by_seed = {"0": accuracy}
    for seed in dict.fromkeys(args.verification_seeds):
        if seed != 0:
            accuracy_by_seed[str(seed)] = _validate_seed(args.rows, seed)
    passed = all(item["passed"] for values in accuracy_by_seed.values() for item in values.values())
    timing = None if args.check_only or not passed else _measure(
        unfused, fused, args.warmup, args.samples, args.reps
    )
    result = {
        "schema_version": 1,
        "case": "bwd_layernorm_mul_dropout_buf157",
        "metadata": {
            "shape_mnk": [CONCAT_FEATURES, GRADIENT_FEATURES, args.rows],
            "source_launch_indices": [96, 97],
            "eliminated_intermediate_bytes": args.rows * CONCAT_FEATURES * 2,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch_version": str(torch.__version__),
            "triton_version": str(triton.__version__),
        },
        "accuracy": accuracy,
        "accuracy_by_seed": accuracy_by_seed,
        "accuracy_passed": passed,
        "timing": timing,
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
