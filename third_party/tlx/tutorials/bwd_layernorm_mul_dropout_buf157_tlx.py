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

"""TLX fused reconstructed-Y dW GEMM for the RLLayer buf157 shape."""

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
from triton.tlx.ops.kernels.mm import _sm100_core as _core
from triton.tlx.ops.kernels.mm import sm100
from triton.tools.tensor_descriptor import TensorDescriptor

import bwd_layernorm_mul_dropout_buf157 as baseline


FEATURES = 256
GRADIENT_FEATURES = 256


@triton.jit
def _process_fused_producer_tile(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    num_mn_tiles,
    k_tiles_total,
    x_ptr,
    u_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    b_desc,
    buffers_a,
    buffers_b,
    a_full_bars,
    b_full_bars,
    a_empty_bars,
    smem_accum_cnt,
    cluster_cta_rank,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    D: tl.constexpr,
    PROLOGUE_K: tl.constexpr,
    SAVED_LAYERNORM_ONLY: tl.constexpr,
):
    mn_tile_id = tile_id % num_mn_tiles
    pid_m, pid_n = _core._compute_pid(
        mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M
    )
    k_tile_start, k_tile_end = _core._compute_k_tile_range(
        tile_id, num_mn_tiles, k_tiles_total, SPLIT_K
    )
    block_m_split: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // NUM_CTAS)
    b_bytes: tl.constexpr = (
        tlx.size_of(tlx.dtype_of(b_desc)) * BLOCK_SIZE_N * BLOCK_SIZE_K // NUM_CTAS
    )
    is_leader = cluster_cta_rank == 0

    local_k_tiles = k_tile_end - k_tile_start
    buf, phase = get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
    for k_idx in range(0, local_k_tiles):
        k_tile = k_tile_start + k_idx
        offs_k = k_tile * BLOCK_SIZE_K

        # The B TMA can overlap the register prologue. Its reuse is tied to the
        # final A group's empty barrier, matching the base GEMM protocol.
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            tlx.barrier_wait(a_empty_bars[a_buf], phase ^ 1)
        _core._expect_and_load(
            b_desc,
            buffers_b[buf],
            offs_k,
            offs_bn,
            b_full_bars[buf],
            b_bytes * NUM_CTAS,
            is_leader,
            NUM_CTAS == 2,
            False,
        )

        for group_id in tl.static_range(NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            feature_base = pid_m * BLOCK_SIZE_M + group_id * block_m_split
            features = feature_base + tl.arange(0, block_m_split)
            source_features = features if SAVED_LAYERNORM_ONLY else features % D
            gamma = tl.load(gamma_ptr + source_features).to(tl.float32)
            beta = tl.load(beta_ptr + source_features).to(tl.float32)

            for k_subtile in tl.static_range(0, BLOCK_SIZE_K, PROLOGUE_K):
                rows = offs_k + k_subtile + tl.arange(0, PROLOGUE_K)
                offsets = rows[:, None] * D + source_features[None, :]
                if SAVED_LAYERNORM_ONLY:
                    x = tl.load(x_ptr + offsets).to(tl.float32)
                    mean = tl.load(mean_ptr + rows).to(tl.float32)
                    rstd = tl.load(rstd_ptr + rows).to(tl.float32)
                    normalized = (x - mean[:, None]) * rstd[:, None]
                    activation = normalized * gamma[None, :] + beta[None, :]
                else:
                    u = tl.load(u_ptr + offsets).to(tl.float32)
                    if feature_base < D:
                        activation = u * tl.sigmoid(u)
                    else:
                        x = tl.load(x_ptr + offsets).to(tl.float32)
                        mean = tl.load(mean_ptr + rows).to(tl.float32)
                        rstd = tl.load(rstd_ptr + rows).to(tl.float32)
                        normalized = (x - mean[:, None]) * rstd[:, None]
                        activation = (normalized * gamma[None, :] + beta[None, :]) * u
                dst = tlx.local_slice(
                    buffers_a[a_buf],
                    [k_subtile, 0],
                    [PROLOGUE_K, block_m_split],
                )
                tlx.local_store(dst, activation.to(tl.bfloat16))

            if NUM_CTAS == 2:
                # Both CTAs publish their local A copy to CTA0; one 2-CTA MMA
                # consumes the pair.
                tlx.barrier_arrive(a_full_bars[a_buf], 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(a_full_bars[a_buf], 1)

        smem_accum_cnt += 1
        buf += 1
        if buf == NUM_SMEM_BUFFERS:
            buf = 0
            phase ^= 1
    return smem_accum_cnt


@triton.jit
def layernorm_mul_dweight_tlx(
    x_ptr,
    u_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    gradient_desc,
    output_desc,
    workspace_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    INTERLEAVE_EPILOGUE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FP16_WORKSPACE: tl.constexpr,
    D: tl.constexpr,
    PROLOGUE_K: tl.constexpr = 16,
    PRODUCER_WARPS: tl.constexpr = 4,
    PRODUCER_REGS: tl.constexpr = 128,
    SAVED_LAYERNORM_ONLY: tl.constexpr = False,
):
    block_m_split: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    buffers_a = tlx.local_alloc(
        (BLOCK_SIZE_K, block_m_split),
        tl.bfloat16,
        NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
    )
    buffers_b = tlx.local_alloc(
        (BLOCK_SIZE_K, BLOCK_SIZE_N // NUM_CTAS),
        tlx.dtype_of(gradient_desc),
        NUM_SMEM_BUFFERS,
    )
    tmem_buffers = tlx.local_alloc(
        (block_m_split, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )

    num_epilogue_buffers: tl.constexpr = NUM_MMA_GROUPS if NUM_MMA_GROUPS > 2 else 2
    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    workspace_dtype: tl.constexpr = tl.float16 if FP16_WORKSPACE else tl.float32
    output_smem = tlx.local_alloc(
        (block_m_split, slice_size),
        workspace_dtype,
        num_epilogue_buffers,
    )

    cluster_cta_rank = tlx.cluster_cta_rank() if NUM_CTAS == 2 else 0
    # A is produced by software in both CTAs and consumed by CTA0's 2-CTA MMA.
    a_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=NUM_CTAS,
    )
    a_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=1,
    )
    b_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=1,
    )
    tmem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=EPILOGUE_SUBTILE * NUM_CTAS,
    )
    clc_context = tlx.clc_create_context(num_consumers=3 * NUM_CTAS, num_stages=1)

    with tlx.async_tasks(
        exclusive=True,
        no_ending_cluster_sync=True,
        mbarrier_try_wait_suspend_ns=50000,
    ):
        with tlx.async_task("default"):
            (
                start_pid,
                num_pid_m,
                _,
                num_pid_in_group,
                num_mn_tiles,
                _,
                k_tiles_total,
            ) = _core._compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )
            tmem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_producer = 1
            clc_phase_consumer = 0
            while tile_id != -1:
                tlx.clc_producer(
                    clc_context, clc_phase_producer, multi_ctas=NUM_CTAS == 2
                )
                clc_phase_producer ^= 1
                k_tile_start, k_tile_end = _core._compute_k_tile_range(
                    tile_id, num_mn_tiles, k_tiles_total, SPLIT_K
                )
                if k_tile_end > k_tile_start:
                    tmem_buf, tmem_phase = get_bufidx_phase(
                        tmem_accum_cnt, NUM_TMEM_BUFFERS
                    )
                    sm100._process_tile_epilogue_inner(
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        num_mn_tiles=num_mn_tiles,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                        SPLIT_K=SPLIT_K,
                        INTERLEAVE_EPILOGUE=INTERLEAVE_EPILOGUE,
                        c_desc=output_desc,
                        workspace_desc=workspace_desc,
                        c_smem_buffers=output_smem,
                        tmem_buffers=tmem_buffers,
                        tmem_full_bars=tmem_full_bars,
                        tmem_empty_bars=tmem_empty_bars,
                        cur_tmem_buf=tmem_buf,
                        tmem_read_phase=tmem_phase,
                        NUM_CTAS=NUM_CTAS,
                    )
                    tmem_accum_cnt += 1
                tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=NUM_CTAS == 2
                )
                clc_phase_consumer ^= 1

        with tlx.async_task(num_warps=1, num_regs=24):
            (
                start_pid,
                _,
                _,
                _,
                num_mn_tiles,
                _,
                k_tiles_total,
            ) = _core._compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )
            tmem_accum_cnt = 0
            smem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                k_tile_start, k_tile_end = _core._compute_k_tile_range(
                    tile_id, num_mn_tiles, k_tiles_total, SPLIT_K
                )
                if k_tile_end > k_tile_start:
                    if cluster_cta_rank == 0:
                        tmem_buf, tmem_phase = get_bufidx_phase(
                            tmem_accum_cnt, NUM_TMEM_BUFFERS
                        )
                        smem_accum_cnt = _core._process_tile_mma_inner(
                            k_tile_start=k_tile_start,
                            k_tile_end=k_tile_end,
                            NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                            NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                            NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                            buffers_A=buffers_a,
                            buffers_B=buffers_b,
                            tmem_buffers=tmem_buffers,
                            A_smem_full_bars=a_full_bars,
                            B_smem_full_bars=b_full_bars,
                            A_smem_empty_bars=a_empty_bars,
                            tmem_full_bars=tmem_full_bars,
                            cur_tmem_buf=tmem_buf,
                            tmem_empty_bars=tmem_empty_bars,
                            tmem_write_phase=tmem_phase,
                            smem_accum_cnt=smem_accum_cnt,
                            NUM_CTAS=NUM_CTAS,
                            A_ROW_MAJOR=False,
                            B_ROW_MAJOR=True,
                        )
                    else:
                        smem_accum_cnt += k_tile_end - k_tile_start
                    tmem_accum_cnt += 1
                tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=NUM_CTAS == 2
                )
                clc_phase_consumer ^= 1

        with tlx.async_task(num_warps=PRODUCER_WARPS, num_regs=PRODUCER_REGS):
            (
                start_pid,
                num_pid_m,
                _,
                num_pid_in_group,
                num_mn_tiles,
                _,
                k_tiles_total,
            ) = _core._compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )
            smem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                k_tile_start, k_tile_end = _core._compute_k_tile_range(
                    tile_id, num_mn_tiles, k_tiles_total, SPLIT_K
                )
                if k_tile_end > k_tile_start:
                    smem_accum_cnt = _process_fused_producer_tile(
                        tile_id,
                        num_pid_in_group,
                        num_pid_m,
                        num_mn_tiles,
                        k_tiles_total,
                        x_ptr,
                        u_ptr,
                        gamma_ptr,
                        beta_ptr,
                        mean_ptr,
                        rstd_ptr,
                        gradient_desc,
                        buffers_a,
                        buffers_b,
                        a_full_bars,
                        b_full_bars,
                        a_empty_bars,
                        smem_accum_cnt,
                        cluster_cta_rank,
                        GROUP_SIZE_M,
                        BLOCK_SIZE_M,
                        BLOCK_SIZE_N,
                        BLOCK_SIZE_K,
                        NUM_MMA_GROUPS,
                        NUM_SMEM_BUFFERS,
                        NUM_CTAS,
                        SPLIT_K,
                        D,
                        PROLOGUE_K,
                        SAVED_LAYERNORM_ONLY,
                    )
                tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=NUM_CTAS == 2
                )
                clc_phase_consumer ^= 1


FP32_CONFIG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 2,
    "NUM_SMEM_BUFFERS": 4,
    "NUM_TMEM_BUFFERS": 1,
    "NUM_MMA_GROUPS": 2,
    "EPILOGUE_SUBTILE": 32,
    "NUM_CTAS": 2,
    "SPLIT_K": 76,
    "INTERLEAVE_EPILOGUE": 1,
}


def run_fused_tlx(
    inputs: dict[str, torch.Tensor],
    output: torch.Tensor,
    workspace: torch.Tensor,
    *,
    fp16_workspace: bool = False,
    config: dict[str, int] | None = None,
    prologue_k: int = 64,
    producer_warps: int = 16,
    producer_regs: int = 128,
) -> None:
    config = dict(FP32_CONFIG if config is None else config)
    rows = inputs["x"].shape[0]
    block_m_split = config["BLOCK_SIZE_M"] // config["NUM_MMA_GROUPS"]
    gradient = inputs["gradient"]
    gradient_desc = TensorDescriptor(
        gradient,
        gradient.shape,
        gradient.stride(),
        [config["BLOCK_SIZE_K"], config["BLOCK_SIZE_N"] // config["NUM_CTAS"]],
    )
    output_block = [
        block_m_split,
        config["BLOCK_SIZE_N"] // config["EPILOGUE_SUBTILE"],
    ]
    output_desc = TensorDescriptor(output, output.shape, output.stride(), output_block)
    workspace_desc = TensorDescriptor(
        workspace, workspace.shape, workspace.stride(), output_block
    )
    num_pid_m = sm100._padded_num_pid_m(
        2 * FEATURES, config["BLOCK_SIZE_M"], config["NUM_CTAS"]
    )
    num_pid_n = triton.cdiv(GRADIENT_FEATURES, config["BLOCK_SIZE_N"])
    grid = (num_pid_m * num_pid_n * config["SPLIT_K"],)
    launch_options = {"ctas_per_cga": (2, 1, 1)} if config["NUM_CTAS"] == 2 else {}
    cast(Any, layernorm_mul_dweight_tlx)[grid](
        inputs["x"],
        inputs["u"],
        inputs["gamma"],
        inputs["beta"],
        inputs["mean"],
        inputs["rstd"],
        gradient_desc,
        output_desc,
        workspace_desc,
        2 * FEATURES,
        GRADIENT_FEATURES,
        rows,
        NUM_SMS=torch.cuda.get_device_properties(
            inputs["x"].device
        ).multi_processor_count,
        FP16_WORKSPACE=fp16_workspace,
        D=FEATURES,
        PROLOGUE_K=prologue_k,
        PRODUCER_WARPS=producer_warps,
        PRODUCER_REGS=producer_regs,
        **config,
        **launch_options,
    )
    sm100._reduce_k_kernel[
        (triton.cdiv(2 * FEATURES, 32), triton.cdiv(GRADIENT_FEATURES, 32))
    ](
        workspace,
        output,
        2 * FEATURES,
        GRADIENT_FEATURES,
        sm100._workspace_rows_per_split(
            2 * FEATURES, config["BLOCK_SIZE_M"], config["NUM_CTAS"]
        ),
        SPLIT_K=config["SPLIT_K"],
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        OUTPUT_DTYPE=tl.bfloat16,
    )


def _make_workspace(config: dict[str, int], *, fp16_workspace: bool) -> torch.Tensor:
    rows_per_split = sm100._workspace_rows_per_split(
        2 * FEATURES, config["BLOCK_SIZE_M"], config["NUM_CTAS"]
    )
    return torch.empty(
        (config["SPLIT_K"] * rows_per_split, GRADIENT_FEATURES),
        device="cuda",
        dtype=torch.float16 if fp16_workspace else torch.float32,
    )


def _run_tlx_backward(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    workspace: torch.Tensor,
    *,
    fp16_workspace: bool,
) -> None:
    result = baseline._run_backward(inputs, False)
    baseline._record_backward_outputs(outputs, result)
    run_fused_tlx(
        inputs,
        outputs["projection_dweight"],
        workspace,
        fp16_workspace=fp16_workspace,
    )


def _validate_seed(
    rows: int, seed: int, *, fp16_workspace: bool
) -> dict[str, dict[str, Any]]:
    torch.manual_seed(seed)
    inputs = baseline._make_inputs(rows)
    reference = baseline._make_outputs()
    candidate = baseline._make_outputs()
    workspace = _make_workspace(FP32_CONFIG, fp16_workspace=fp16_workspace)
    baseline._run_unfused(inputs, reference)
    _run_tlx_backward(inputs, candidate, workspace, fp16_workspace=fp16_workspace)
    torch.cuda.synchronize()
    return baseline._accuracy(reference, candidate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=baseline.DEFAULT_ROWS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--verification-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--fp16-workspace", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.reps) < 1:
        parser.error("--warmup, --samples, and --reps must be positive")
    if args.rows < FP32_CONFIG["BLOCK_SIZE_K"] or (
        args.rows % FP32_CONFIG["BLOCK_SIZE_K"]
    ):
        parser.error(
            f"--rows must be a positive multiple of {FP32_CONFIG['BLOCK_SIZE_K']}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")

    torch.manual_seed(0)
    inputs = baseline._make_inputs(args.rows)
    reference = baseline._make_outputs()
    candidate = baseline._make_outputs()
    workspace = _make_workspace(FP32_CONFIG, fp16_workspace=args.fp16_workspace)

    def unfused() -> None:
        baseline._run_unfused(inputs, reference)

    def fused_tlx() -> None:
        _run_tlx_backward(
            inputs,
            candidate,
            workspace,
            fp16_workspace=args.fp16_workspace,
        )

    unfused()
    fused_tlx()
    torch.cuda.synchronize()
    accuracy_by_seed = {"0": baseline._accuracy(reference, candidate)}
    for seed in dict.fromkeys(args.verification_seeds):
        if seed != 0:
            accuracy_by_seed[str(seed)] = _validate_seed(
                args.rows, seed, fp16_workspace=args.fp16_workspace
            )
    passed = all(
        item["passed"]
        for values in accuracy_by_seed.values()
        for item in values.values()
    )
    timing = (
        None
        if args.check_only or not passed
        else baseline._measure(unfused, fused_tlx, args.warmup, args.samples, args.reps)
    )
    result = {
        "schema_version": 1,
        "case": "bwd_layernorm_mul_dropout_buf157_tlx",
        "metadata": {
            "shape_mnk": [2 * FEATURES, GRADIENT_FEATURES, args.rows],
            "workspace_dtype": "float16" if args.fp16_workspace else "float32",
            "tlx_config": FP32_CONFIG,
            "prologue_k": 64,
            "producer_warps": 16,
            "producer_regs": 128,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch_version": str(torch.__version__),
            "triton_version": str(triton.__version__),
        },
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
