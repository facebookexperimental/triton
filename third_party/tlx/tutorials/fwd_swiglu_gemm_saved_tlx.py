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

"""Saved-output RLLayer gate/up GEMM and SwiGLU epilogue, warp-specialized.

Public contract (unchanged from the seed)::

    projection = BF16(FP32_MMA(a, b.T))
    gate, up = projection.chunk(2, dim=1)
    output = BF16(FP32(gate) * sigmoid(FP32(gate)) * FP32(up))
    return projection, output

The target shape is ``a[5120,4096]`` and ``b[16384,4096]``.  GEMM accumulation
stays FP32 and the packed BF16 projection is a required output.

Schedule (GB200 / sm_100a).  The gate and up halves of a projection row block
are one ``BLOCK_M x 2*BLOCK_N`` slab of the packed output, so they are produced
by a *single* ``tcgen05`` MMA of width ``MMA_N = 2 * BLOCK_N`` into one FP32
TMEM accumulator rather than by two ``BLOCK_N``-wide MMAs into two
accumulators.  Under ``cta_group::2`` each CTA supplies half of the MMA's N,
which lands exactly on the gate/up split: CTA 0 fetches the gate weight tile,
CTA 1 the up weight tile, each a contiguous 2-D TMA box out of ``b[2N, K]``.
The A tile is then read from SMEM once per k-step instead of twice, which is
what the two-MMA form paid for the shared operand.

Four warp groups: a one-warp TMA producer, a one-warp MMA issuer, and two
epilogue groups.  ``EPILOGUE_PARTS`` fans the per-tile TMEM drain across those
epilogue groups so one tile's store chain is not a single serial instruction
stream, and ``NUM_TMEM_BUFFERS = 2`` lets tile ``i``'s drain overlap tile
``i+1``'s MMA; at ``MMA_N = 256`` the two buffers consume all 512 TMEM columns.

``TWO_CTA`` pairs adjacent ``M`` tiles into a ``cta_group::2`` cluster.  Shapes
that do not tile evenly fall back to the seed's masked pointer kernel.
"""

from __future__ import annotations

from typing import Any, cast

import ctypes

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor


# --- host-side L2 persisting carve-out -------------------------------------
# The per-instruction `evict_last` hint only biases replacement. The hardware
# additionally exposes a *reserved* L2 partition plus a per-stream address
# window whose hits are tagged persisting. `a` is 40 MiB against a 129 MiB L2
# and an 80 MiB maximum carve-out, so it fits, and it is the operand with reuse
# (read once per N tile) while `b` streams through exactly once.

_CUDA_LIMIT_PERSISTING_L2 = 0x06
_STREAM_ATTR_ACCESS_POLICY_WINDOW = 1
_ACCESS_PROPERTY_STREAMING = 1
_ACCESS_PROPERTY_PERSISTING = 2


class _AccessPolicyWindow(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_void_p),
        ("num_bytes", ctypes.c_size_t),
        ("hit_ratio", ctypes.c_float),
        ("hit_prop", ctypes.c_int),
        ("miss_prop", ctypes.c_int),
    ]


class _StreamAttrValue(ctypes.Union):
    _fields_ = [
        ("access_policy_window", _AccessPolicyWindow),
        ("pad", ctypes.c_byte * 64),
    ]


_cudart: Any = None
_carved_mib: dict[int, int] = {}


def _runtime() -> Any:
    global _cudart
    if _cudart is None:
        lib = ctypes.CDLL("libcudart.so")
        lib.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
        lib.cudaStreamSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _cudart = lib
    return _cudart


def _set_persist_window(
    tensor: torch.Tensor | None,
    carve_mib: int,
    hit_ratio: float,
    device: torch.device | int | None = None,
) -> bool:
    """Pin `tensor` in the reserved L2 partition, or clear the window if None."""
    selected_device = tensor.device if tensor is not None else device
    try:
        with torch.cuda.device(selected_device):
            device_index = torch.cuda.current_device()
            lib = _runtime()
            if carve_mib != _carved_mib.get(device_index):
                if lib.cudaDeviceSetLimit(
                    _CUDA_LIMIT_PERSISTING_L2, ctypes.c_size_t(carve_mib << 20)
                ):
                    return False
                _carved_mib[device_index] = carve_mib
            value = _StreamAttrValue()
            if tensor is None:
                value.access_policy_window = _AccessPolicyWindow(None, 0, 0.0, 0, 0)
            else:
                value.access_policy_window = _AccessPolicyWindow(
                    ctypes.c_void_p(tensor.data_ptr()),
                    tensor.numel() * tensor.element_size(),
                    hit_ratio,
                    _ACCESS_PROPERTY_PERSISTING,
                    _ACCESS_PROPERTY_STREAMING,
                )
            stream = torch.cuda.current_stream(selected_device)
            return not lib.cudaStreamSetAttribute(
                ctypes.c_void_p(stream.cuda_stream),
                _STREAM_ATTR_ACCESS_POLICY_WINDOW,
                ctypes.byref(value),
            )
    except (AttributeError, OSError):
        # Keep the TLX kernel usable on CUDA installations that do not expose
        # the optional runtime access-policy API.
        return False


@triton.jit
def _epi_store(desc, smem, row, col, EVICT: tl.constexpr):
    if EVICT == 0:
        tlx.async_descriptor_store(desc, smem, [row, col])
    elif EVICT == 1:
        tlx.async_descriptor_store(
            desc, smem, [row, col], eviction_policy="evict_first"
        )
    else:
        tlx.async_descriptor_store(desc, smem, [row, col], eviction_policy="evict_last")


@triton.jit
def _operand_load(desc, buf, row, col, bar, EVICT: tl.constexpr):
    """TMA operand load with a selectable L2 eviction hint.

    `A` is only 41.9 MB and is re-read once per N tile, so it is the operand
    worth keeping resident; `b` is 134.2 MB and streams through exactly once.
    Tagging them differently is the only lever over which of the two survives
    in L2.
    """
    if EVICT == 0:
        tlx.async_descriptor_load(desc, buf, [row, col], bar)
    elif EVICT == 1:
        tlx.async_descriptor_load(
            desc, buf, [row, col], bar, eviction_policy="evict_first"
        )
    else:
        tlx.async_descriptor_load(
            desc, buf, [row, col], bar, eviction_policy="evict_last"
        )


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _swiglu_gemm_ws(
    a_desc,
    b_desc,
    projection_desc,
    output_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    EPILOGUE_PARTS: tl.constexpr,
    SUBS_PER_PART: tl.constexpr,
    PART1_OFFSET: tl.constexpr,
    EPILOGUE_WARPS: tl.constexpr,
    NUM_EPI_BUFFERS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPI_REGS: tl.constexpr,
    TWO_CTA: tl.constexpr,
    EVICT_A: tl.constexpr,
    EVICT_B: tl.constexpr,
    EVICT_STORE: tl.constexpr,
):
    SUB_N: tl.constexpr = BLOCK_N // EPILOGUE_SUBTILE
    EPI_SLOTS: tl.constexpr = NUM_EPI_BUFFERS // EPILOGUE_PARTS
    MMA_N: tl.constexpr = 2 * BLOCK_N
    B_TILE_N: tl.constexpr = MMA_N // 2 if TWO_CTA else MMA_N

    buffers_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, NUM_SMEM_BUFFERS)
    buffers_b = tlx.local_alloc((B_TILE_N, BLOCK_K), tl.bfloat16, NUM_SMEM_BUFFERS)
    acc = tlx.local_alloc(
        (BLOCK_M, MMA_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem
    )
    epi_smem = tlx.local_alloc((BLOCK_M, SUB_N), tl.bfloat16, NUM_EPI_BUFFERS)

    operands_full = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    operands_empty = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    acc_full = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    acc_releases: tl.constexpr = (
        4 * EPILOGUE_SUBTILE if TWO_CTA else 2 * EPILOGUE_SUBTILE
    )
    acc_empty = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=acc_releases)
    cta_bars = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=2)

    if TWO_CTA:
        cta_rank = tlx.cluster_cta_rank()
    else:
        cta_rank = 0
    peer_rank = cta_rank ^ 1
    pred_leader = cta_rank % 2 == 0

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_K)

    CTAS: tl.constexpr = 2 if TWO_CTA else 1
    clc_ctx = tlx.clc_create_context(num_consumers=(2 + EPILOGUE_PARTS) * CTAS)

    with tlx.async_tasks():
        with tlx.async_task("default", num_regs=EPI_REGS):
            tile_counter = 0
            tile_id = start_pid
            clc_p_phase = 1
            clc_c_phase = 0
            while tile_id != -1:
                tlx.clc_producer(clc_ctx, clc_p_phase, multi_ctas=TWO_CTA)
                clc_p_phase ^= 1
                pid_m, pid_n = _compute_pid(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M
                )
                offs_m = pid_m * BLOCK_M
                offs_n = pid_n * BLOCK_N
                tbuf, tphase = get_bufidx_phase(tile_counter, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(acc_full[tbuf], tphase)
                for local_sub in tl.static_range(SUBS_PER_PART):
                    col = offs_n + local_sub * SUB_N
                    gate = tlx.local_load(
                        tlx.subslice(acc[tbuf], local_sub * SUB_N, SUB_N)
                    )
                    tlx.barrier_arrive(acc_empty[tbuf], 1)
                    if TWO_CTA:
                        tlx.barrier_arrive(
                            acc_empty[tbuf], 1, remote_cta_rank=peer_rank
                        )
                    gate_bf16 = gate.to(tl.bfloat16)
                    slot_g = (3 * local_sub) % EPI_SLOTS
                    tlx.async_descriptor_store_wait(EPI_SLOTS - 1)
                    tlx.local_store(epi_smem[slot_g], gate_bf16)
                    _epi_store(
                        projection_desc, epi_smem[slot_g], offs_m, col, EVICT_STORE
                    )
                    up = tlx.local_load(
                        tlx.subslice(acc[tbuf], BLOCK_N + local_sub * SUB_N, SUB_N)
                    )
                    tlx.barrier_arrive(acc_empty[tbuf], 1)
                    if TWO_CTA:
                        tlx.barrier_arrive(
                            acc_empty[tbuf], 1, remote_cta_rank=peer_rank
                        )
                    up_bf16 = up.to(tl.bfloat16)
                    slot_u = (3 * local_sub + 1) % EPI_SLOTS
                    tlx.async_descriptor_store_wait(EPI_SLOTS - 1)
                    tlx.local_store(epi_smem[slot_u], up_bf16)
                    _epi_store(
                        projection_desc, epi_smem[slot_u], offs_m, col + N, EVICT_STORE
                    )
                    gate_fp32 = gate_bf16.to(tl.float32)
                    up_fp32 = up_bf16.to(tl.float32)
                    activated = (gate_fp32 * tl.sigmoid(gate_fp32) * up_fp32).to(
                        tl.bfloat16
                    )
                    slot_o = (3 * local_sub + 2) % EPI_SLOTS
                    tlx.async_descriptor_store_wait(EPI_SLOTS - 1)
                    tlx.local_store(epi_smem[slot_o], activated)
                    _epi_store(output_desc, epi_smem[slot_o], offs_m, col, EVICT_STORE)
                tile_counter += 1
                raw_tile = tlx.clc_consumer(clc_ctx, clc_c_phase, multi_ctas=TWO_CTA)
                clc_c_phase ^= 1
                if raw_tile == -1:
                    tile_id = -1
                else:
                    tile_id = (raw_tile // CTAS) * CTAS + cta_rank
            tlx.async_descriptor_store_wait(0)

        if EPILOGUE_PARTS == 2:
            with tlx.async_task(num_warps=EPILOGUE_WARPS, num_regs=EPI_REGS):
                tile_counter = 0
                tile_id = start_pid
                clc_c_phase = 0
                while tile_id != -1:
                    pid_m, pid_n = _compute_pid(
                        tile_id, num_pid_in_group, num_pid_m, GROUP_M
                    )
                    offs_m = pid_m * BLOCK_M
                    offs_n = pid_n * BLOCK_N
                    tbuf, tphase = get_bufidx_phase(tile_counter, NUM_TMEM_BUFFERS)
                    tlx.barrier_wait(acc_full[tbuf], tphase)
                    for local_sub in tl.static_range(SUBS_PER_PART):
                        col = offs_n + PART1_OFFSET + local_sub * SUB_N
                        gate = tlx.local_load(
                            tlx.subslice(
                                acc[tbuf], PART1_OFFSET + local_sub * SUB_N, SUB_N
                            )
                        )
                        tlx.barrier_arrive(acc_empty[tbuf], 1)
                        if TWO_CTA:
                            tlx.barrier_arrive(
                                acc_empty[tbuf], 1, remote_cta_rank=peer_rank
                            )
                        gate_bf16 = gate.to(tl.bfloat16)
                        slot_g = EPI_SLOTS + (3 * local_sub) % EPI_SLOTS
                        tlx.async_descriptor_store_wait(EPI_SLOTS - 1)
                        tlx.local_store(epi_smem[slot_g], gate_bf16)
                        _epi_store(
                            projection_desc, epi_smem[slot_g], offs_m, col, EVICT_STORE
                        )
                        up = tlx.local_load(
                            tlx.subslice(
                                acc[tbuf],
                                BLOCK_N + PART1_OFFSET + local_sub * SUB_N,
                                SUB_N,
                            )
                        )
                        tlx.barrier_arrive(acc_empty[tbuf], 1)
                        if TWO_CTA:
                            tlx.barrier_arrive(
                                acc_empty[tbuf], 1, remote_cta_rank=peer_rank
                            )
                        up_bf16 = up.to(tl.bfloat16)
                        slot_u = EPI_SLOTS + (3 * local_sub + 1) % EPI_SLOTS
                        tlx.async_descriptor_store_wait(EPI_SLOTS - 1)
                        tlx.local_store(epi_smem[slot_u], up_bf16)
                        _epi_store(
                            projection_desc,
                            epi_smem[slot_u],
                            offs_m,
                            col + N,
                            EVICT_STORE,
                        )
                        gate_fp32 = gate_bf16.to(tl.float32)
                        up_fp32 = up_bf16.to(tl.float32)
                        activated = (gate_fp32 * tl.sigmoid(gate_fp32) * up_fp32).to(
                            tl.bfloat16
                        )
                        slot_o = EPI_SLOTS + (3 * local_sub + 2) % EPI_SLOTS
                        tlx.async_descriptor_store_wait(EPI_SLOTS - 1)
                        tlx.local_store(epi_smem[slot_o], activated)
                        _epi_store(
                            output_desc, epi_smem[slot_o], offs_m, col, EVICT_STORE
                        )
                    tile_counter += 1
                    raw_tile = tlx.clc_consumer(
                        clc_ctx, clc_c_phase, multi_ctas=TWO_CTA
                    )
                    clc_c_phase ^= 1
                    if raw_tile == -1:
                        tile_id = -1
                    else:
                        tile_id = (raw_tile // CTAS) * CTAS + cta_rank
                tlx.async_descriptor_store_wait(0)

        with tlx.async_task(num_warps=1, num_regs=24):
            tile_counter = 0
            smem_cnt = 0
            tile_id = start_pid
            clc_c_phase = 0
            while tile_id != -1:
                tbuf, twphase = get_bufidx_phase(tile_counter, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(acc_empty[tbuf], twphase ^ 1)
                for k in range(k_tiles):
                    buf, phase = get_bufidx_phase(smem_cnt, NUM_SMEM_BUFFERS)
                    tlx.barrier_wait(operands_full[buf], phase)
                    if TWO_CTA:
                        tlx.barrier_arrive(
                            cta_bars[buf], arrive_count=1, remote_cta_rank=0
                        )
                        tlx.barrier_wait(cta_bars[buf], phase=phase, pred=pred_leader)
                    tlx.async_dot(
                        buffers_a[buf],
                        tlx.local_trans(buffers_b[buf]),
                        acc[tbuf],
                        use_acc=k > 0,
                        mBarriers=[operands_empty[buf]],
                        two_ctas=TWO_CTA,
                        out_dtype=tl.float32,
                    )
                    smem_cnt += 1
                tlx.tcgen05_commit(acc_full[tbuf], two_ctas=TWO_CTA)
                tile_counter += 1
                raw_tile = tlx.clc_consumer(clc_ctx, clc_c_phase, multi_ctas=TWO_CTA)
                clc_c_phase ^= 1
                if raw_tile == -1:
                    tile_id = -1
                else:
                    tile_id = (raw_tile // CTAS) * CTAS + cta_rank

        with tlx.async_task(num_warps=1, num_regs=24):
            smem_cnt = 0
            tile_id = start_pid
            clc_c_phase = 0
            expected: tl.constexpr = 2 * (BLOCK_M + B_TILE_N) * BLOCK_K
            while tile_id != -1:
                pid_m, pid_n = _compute_pid(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M
                )
                offs_m = pid_m * BLOCK_M
                offs_b = cta_rank * N + pid_n * BLOCK_N
                for k in range(k_tiles):
                    buf, phase = get_bufidx_phase(smem_cnt, NUM_SMEM_BUFFERS)
                    tlx.barrier_wait(operands_empty[buf], phase ^ 1)
                    tlx.barrier_expect_bytes(operands_full[buf], expected)
                    _operand_load(
                        a_desc,
                        buffers_a[buf],
                        offs_m,
                        k * BLOCK_K,
                        operands_full[buf],
                        EVICT_A,
                    )
                    _operand_load(
                        b_desc,
                        buffers_b[buf],
                        offs_b,
                        k * BLOCK_K,
                        operands_full[buf],
                        EVICT_B,
                    )
                    smem_cnt += 1
                raw_tile = tlx.clc_consumer(clc_ctx, clc_c_phase, multi_ctas=TWO_CTA)
                clc_c_phase ^= 1
                if raw_tile == -1:
                    tile_id = -1
                else:
                    tile_id = (raw_tile // CTAS) * CTAS + cta_rank


@triton.jit
def _gemm_swiglu_saved_fallback(
    a,
    b,
    projection,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    for tile_id in range(start_pid, num_pid_m * num_pid_n, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + tile_id % group_size_m
        pid_n = tile_id % num_pid_in_group // group_size_m
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        safe_rows = tl.where(rows < M, rows, 0)
        safe_columns = tl.where(columns < N, columns, 0)
        gate = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        up = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
            ks = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
            a_tile = tl.load(
                a + safe_rows[:, None] * K + ks[None, :],
                mask=(rows[:, None] < M) & (ks[None, :] < K),
                other=0.0,
            )
            gate_weight = tl.load(
                b + safe_columns[None, :] * K + ks[:, None],
                mask=(columns[None, :] < N) & (ks[:, None] < K),
                other=0.0,
            )
            up_weight = tl.load(
                b + (safe_columns[None, :] + N) * K + ks[:, None],
                mask=(columns[None, :] < N) & (ks[:, None] < K),
                other=0.0,
            )
            gate = tl.dot(a_tile, gate_weight, gate, allow_tf32=False)
            up = tl.dot(a_tile, up_weight, up, allow_tf32=False)

        mask = (rows[:, None] < M) & (columns[None, :] < N)
        gate_bf16 = gate.to(tl.bfloat16)
        up_bf16 = up.to(tl.bfloat16)
        projection_offsets = rows[:, None] * (2 * N) + columns[None, :]
        tl.store(projection + projection_offsets, gate_bf16, mask=mask)
        tl.store(projection + projection_offsets + N, up_bf16, mask=mask)
        gate_fp32 = gate_bf16.to(tl.float32)
        up_fp32 = up_bf16.to(tl.float32)
        activated = gate_fp32 * tl.sigmoid(gate_fp32) * up_fp32
        tl.store(
            output + rows[:, None] * N + columns[None, :],
            activated.to(tl.bfloat16),
            mask=mask,
        )


CONFIG: dict[str, Any] = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 128,
    "GROUP_M": 128,
    "NUM_SMEM_BUFFERS": 3,
    "NUM_TMEM_BUFFERS": 2,
    "EPILOGUE_SUBTILE": 4,
    "EPILOGUE_PARTS": 2,
    "EPILOGUE_WARPS": 4,
    "NUM_EPI_BUFFERS": 4,
    "NUM_WARPS": 8,
    "EPI_REGS": 224,
    "TWO_CTA": True,
    "EVICT_A": 2,
    "EVICT_B": 1,
    "EVICT_STORE": 1,
    "L2_PERSIST_MIB": 48,
    "L2_HIT_RATIO": 1.0,
}


def _run_fallback(
    a: torch.Tensor, b: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    projection = torch.empty((m, 2 * n), device=a.device, dtype=torch.bfloat16)
    output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (min(num_sms, triton.cdiv(m, 128) * triton.cdiv(n, 256)),)
    cast(Any, _gemm_swiglu_saved_fallback)[grid](
        a,
        b,
        projection,
        output,
        M=m,
        N=n,
        K=k,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        GROUP_M=8,
        NUM_SMS=num_sms,
        num_warps=8,
        num_stages=2,
    )
    return projection, output


def fwd_swiglu_gemm_saved(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the saved packed projection and compact SwiGLU output."""
    config = CONFIG if config is None else {**CONFIG, **config}
    m, k = a.shape
    if b.ndim != 2 or b.shape[1] != k or b.shape[0] % 2:
        raise ValueError("b must have shape [2 * N, K]")
    n = b.shape[0] // 2
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    block_k = config["BLOCK_K"]
    two_cta = bool(config["TWO_CTA"])
    row_tiles = m // block_m
    if not two_cta or m % block_m or n % block_n or k % block_k or row_tiles % 2:
        return _run_fallback(a, b, n)

    projection = torch.empty((m, 2 * n), device=a.device, dtype=torch.bfloat16)
    output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_sms = num_sms // 2 * 2
    sub_n = block_n // config["EPILOGUE_SUBTILE"]

    a_desc = TensorDescriptor(a, [m, k], [k, 1], [block_m, block_k])
    b_desc = TensorDescriptor(b, [2 * n, k], [k, 1], [block_n, block_k])
    projection_desc = TensorDescriptor(
        projection, [m, 2 * n], [2 * n, 1], [block_m, sub_n]
    )
    output_desc = TensorDescriptor(output, [m, n], [n, 1], [block_m, sub_n])

    grid = (row_tiles * (n // block_n),)
    persist_mib = int(config["L2_PERSIST_MIB"])
    pinned = persist_mib > 0 and _set_persist_window(
        a, persist_mib, float(config["L2_HIT_RATIO"])
    )

    cast(Any, _swiglu_gemm_ws)[grid](
        a_desc,
        b_desc,
        projection_desc,
        output_desc,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=config["GROUP_M"],
        NUM_SMEM_BUFFERS=config["NUM_SMEM_BUFFERS"],
        NUM_TMEM_BUFFERS=config["NUM_TMEM_BUFFERS"],
        EPILOGUE_SUBTILE=config["EPILOGUE_SUBTILE"],
        EPILOGUE_PARTS=config["EPILOGUE_PARTS"],
        SUBS_PER_PART=config["EPILOGUE_SUBTILE"] // config["EPILOGUE_PARTS"],
        PART1_OFFSET=sub_n * (config["EPILOGUE_SUBTILE"] // config["EPILOGUE_PARTS"]),
        EPILOGUE_WARPS=config["EPILOGUE_WARPS"],
        NUM_EPI_BUFFERS=config["NUM_EPI_BUFFERS"],
        NUM_SMS=num_sms,
        EPI_REGS=config["EPI_REGS"],
        TWO_CTA=two_cta,
        EVICT_A=config["EVICT_A"],
        EVICT_B=config["EVICT_B"],
        EVICT_STORE=config["EVICT_STORE"],
        num_warps=config["NUM_WARPS"],
        num_stages=1,
        ctas_per_cga=(2, 1, 1),
    )
    if pinned:
        # Scope the window to this kernel so nothing else on the stream
        # inherits the policy.
        _set_persist_window(None, persist_mib, 0.0, a.device)
    return projection, output
