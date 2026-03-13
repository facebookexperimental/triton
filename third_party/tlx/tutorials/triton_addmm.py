# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

import math
from typing import List, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore

    HAS_TLX = True
except ImportError:
    tlx = None
    HAS_TLX = False

try:
    # @manual=//triton:triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    TMA_AVAILABLE = True
except ImportError:
    TMA_AVAILABLE = False
    pass

# ============================================================================
# Local implementations to replace generative_recommenders dependencies
# ============================================================================


def is_sm100_plus() -> bool:
    """Check if the current GPU is SM100 (Blackwell) or later."""
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 10  # SM100 = compute capability 10.0


def triton_autotune(configs, key, prune_configs_by=None, warmup=25, rep=100):
    """Local wrapper around triton.autotune."""

    def decorator(fn):
        return triton.autotune(
            configs=configs,
            key=key,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
        )(fn)

    return decorator


def triton_cc(annotations=None):
    """No-op decorator for compile-time constant annotations."""

    def decorator(fn):
        if annotations is not None:
            fn._triton_cc_annotations = annotations
        return fn

    return decorator


ENABLE_FULL_TURNING_SPACE = False


def _check_tma_alignment(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, min_alignment: int = 16) -> bool:
    """Check if tensors meet TMA alignment requirements.

    TMA (Tensor Memory Accelerator) on H100 requires:
    1. Base addresses to be 64-byte aligned
    2. Dimensions to be multiples of 64 for optimal performance
    3. Contiguous inner dimensions (stride=1)

    Args:
        x: Input tensor [M, K]
        w: Weight tensor [K, N]
        y: Bias tensor [N] or [M, N]
        min_alignment: Minimum alignment requirement (default: 64)

    Returns:
        True if all tensors meet TMA alignment requirements
    """
    _, K = x.shape
    KB, N = w.shape
    assert K == KB, f"incompatible dimensions {K}, {KB}"

    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f"incompatible dimensions {N}, {NY}"

    return (K % min_alignment == 0) and (N % min_alignment == 0)


def _prune_configs_for_pair_cta(configs, named_args, **kwargs):  # noqa
    M = named_args.get("M", 0)
    N = named_args.get("N", 0)

    pruned = []
    for c in configs:
        BLOCK_M = c.kwargs.get("BLOCK_M", 0)
        BLOCK_N = c.kwargs.get("BLOCK_N", 0)

        # BLOCK_N >= 64 required for PAIR_CTA
        if BLOCK_N < 64:
            continue

        # PAIR_CTA requires even number of M tiles and even total tiles
        num_tiles_m = math.ceil(M / BLOCK_M) if BLOCK_M > 0 else 0
        num_tiles_n = math.ceil(N / BLOCK_N) if BLOCK_N > 0 else 0
        total_tiles = num_tiles_m * num_tiles_n

        # PAIR_CTA (2-CTA) mode requires BLOCK_M >= 128 for tcgen05 MMA
        pair_cta_compatible = ((num_tiles_m % 2 == 0) and (total_tiles % 2 == 0)
                               and (BLOCK_M >= 128)  # Required for 2-CTA MMA
                               )

        # Disable PAIR_CTA for now - cluster configuration is not yet supported
        # in the autotuning path. When enabled, requires ctas_per_cga=(2, 1, 1)
        c.kwargs["PAIR_CTA"] = False  # pair_cta_compatible
        # Set ctas_per_cga for CUDA-native cluster launch semantics (TLX way)
        # c.ctas_per_cga = (2, 1, 1) if pair_cta_compatible else None

        pruned.append(c)
    return pruned


def get_mm_configs(pre_hook=None) -> List[triton.Config]:
    block_m_range = [32, 64, 128, 256]
    block_n_range = [32, 64, 128, 256]
    block_k_range = [32, 64]
    group_m_range = [4, 8]
    # WARP_SPECIALIZE only works with num_warps >=4
    num_warps_range = [4, 8] if is_sm100_plus() else [2, 4, 8]
    num_stage_range = [2, 3, 4, 5]
    if ENABLE_FULL_TURNING_SPACE:
        return [
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": block_k,
                    "GROUP_M": group_m,
                },
                num_stages=num_stages,
                num_warps=num_warps,
                pre_hook=pre_hook,
            )
            for block_m in block_m_range
            for block_n in block_n_range
            for block_k in block_k_range
            for group_m in group_m_range
            for num_stages in num_stage_range
            for num_warps in num_warps_range
        ]
    else:
        configs = [
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 64,
                    "GROUP_M": 8,
                },
                num_stages=3,
                num_warps=4,
                pre_hook=pre_hook,
            ),
        ]
        return [c for c in configs if c.num_warps >= 4]


@triton_cc(
    annotations={
        "M": "i32",
        "N": ("i32", 16),
        "K": ("i32", 16),
        "stride_xm": ("i32", 16),
        "stride_xk": ("i32", 1),
        "stride_wk": ("i32", 16),
        "stride_wn": ("i32", 1),
        "stride_ym": ("i32", 16),
        "stride_yn": ("i32", 1),
        "stride_zm": ("i32", 16),
        "stride_zn": ("i32", 1),
    }, )
@triton_autotune(
    configs=get_mm_configs(),
    key=["N", "K"],
)
@triton.jit
def _addmm_fwd(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
):
    pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
    pid = pid_0 * tl.num_programs(axis=1) + pid_1
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    z_mask = mask_m & mask_n
    if BROADCAST_Y:
        # y is a vector, broadcast to add to z
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_ym
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=z_mask)
    z = (accumulator + y.to(tl.float32)).to(z_ptr.dtype.element_ty)
    z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
    z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)


def _addmm_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]
    PAIR_CTA = nargs.get("PAIR_CTA", False)
    nargs["x_desc"].block_shape = [BLOCK_M, BLOCK_K]
    # In PAIR_CTA mode, each CTA loads BLOCK_N // 2 of W
    if PAIR_CTA:
        nargs["w_desc"].block_shape = [BLOCK_K, BLOCK_N // 2]
    else:
        nargs["w_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["z_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
        if nargs["BROADCAST_Y"]:
            nargs["y_desc"].block_shape = [1, BLOCK_N // 2]
        else:
            nargs["y_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["z_desc"].block_shape = [BLOCK_M, BLOCK_N]
        if nargs["BROADCAST_Y"]:
            nargs["y_desc"].block_shape = [1, BLOCK_N]
        else:
            nargs["y_desc"].block_shape = [BLOCK_M, BLOCK_N]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton_autotune(
    configs=get_mm_configs(pre_hook=_addmm_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
    prune_configs_by={"early_config_prune": _prune_configs_for_pair_cta},
)
@triton.jit
def _addmm_fwd_tma_persistent(
    x_desc,
    w_desc,
    y_desc,
    z_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    PAIR_CTA: tl.constexpr = False,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS)
        offs_xm = pid_m * BLOCK_M
        offs_wn = pid_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
            offs_k = k * BLOCK_K
            x = x_desc.load([offs_xm, offs_k])
            w = w_desc.load([offs_k, offs_wn])

            accumulator = tl.dot(x, w, accumulator, allow_tf32=ALLOW_TF32, two_ctas=PAIR_CTA)

        # Full tile offset for y and z (same for both CTAs)
        offs_wn_full = pid_n * BLOCK_N
        if BROADCAST_Y:
            y = y_desc.load([0, offs_wn_full])
        else:
            y = y_desc.load([offs_xm, offs_wn_full])
        z = (accumulator + y.to(tl.float32)).to(z_desc.dtype)
        z_desc.store([offs_xm, offs_wn_full], z)


@triton_autotune(
    configs=get_mm_configs(pre_hook=_addmm_tma_set_block_size_hook),
    key=["N", "K"],
)
@triton.jit
def _addmm_fwd_tma_ws(
    x_desc,
    w_desc,
    y_desc,
    z_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
):
    x_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), x_desc.dtype, NUM_SMEM_BUFFERS)
    w_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), w_desc.dtype, NUM_SMEM_BUFFERS)
    acc_tmem_buffer = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)

    if BROADCAST_Y:
        y_buffer = tlx.local_alloc((1, BLOCK_N), y_desc.dtype, tl.constexpr(1))
    else:
        y_buffer = tlx.local_alloc((BLOCK_M, BLOCK_N), y_desc.dtype, tl.constexpr(1))
    z_buffer = tlx.local_alloc((BLOCK_M, BLOCK_N), z_desc.dtype, tl.constexpr(1))

    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    y_load_barrier = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    with tlx.async_tasks():
        # Producer task: TMA loads
        with tlx.async_task("default"):
            pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
            pid = pid_0 * tl.num_programs(axis=1) + pid_1
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_xm = pid_m * BLOCK_M
            offs_wn = pid_n * BLOCK_N
            k_tiles = tl.cdiv(K, BLOCK_K)

            load_phase = 0
            for k in range(0, k_tiles):
                buf = k % int(NUM_SMEM_BUFFERS)

                # Wait for buffer to be free
                if k >= NUM_SMEM_BUFFERS:
                    tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)

                offs_k = k * BLOCK_K
                tlx.barrier_expect_bytes(
                    smem_full_bars[buf],
                    2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N),
                )
                tlx.async_descriptor_load(x_desc, x_buffers[buf], [offs_xm, offs_k], smem_full_bars[buf])
                tlx.async_descriptor_load(w_desc, w_buffers[buf], [offs_k, offs_wn], smem_full_bars[buf])

                load_phase = load_phase ^ (buf == NUM_SMEM_BUFFERS - 1)

        # Consumer task: async_dot MMA
        with tlx.async_task(num_warps=4, num_regs=232):
            pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
            pid = pid_0 * tl.num_programs(axis=1) + pid_1
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_xm = pid_m * BLOCK_M
            offs_wn = pid_n * BLOCK_N
            k_tiles = tl.cdiv(K, BLOCK_K)

            # Start async load of y early
            y_buf_view = tlx.local_view(y_buffer, 0)
            y_load_bar = tlx.local_view(y_load_barrier, 0)
            if BROADCAST_Y:
                tlx.barrier_expect_bytes(y_load_bar, 1 * BLOCK_N * 2)
                tlx.async_descriptor_load(y_desc, y_buf_view, [0, offs_wn], y_load_bar)
            else:
                tlx.barrier_expect_bytes(y_load_bar, BLOCK_M * BLOCK_N * 2)
                tlx.async_descriptor_load(y_desc, y_buf_view, [offs_xm, offs_wn], y_load_bar)

            dot_phase = 0
            for k in range(0, k_tiles):
                buf = k % int(NUM_SMEM_BUFFERS)
                tlx.barrier_wait(smem_full_bars[buf], dot_phase)

                tlx.async_dot(
                    x_buffers[buf],
                    w_buffers[buf],
                    acc_tmem_buffer[0],
                    use_acc=k > 0,
                    mBarriers=[smem_empty_bars[buf]],
                    out_dtype=tl.float32,
                )

                dot_phase = dot_phase ^ (buf == NUM_SMEM_BUFFERS - 1)

            last_buf = (k_tiles - 1) % NUM_SMEM_BUFFERS
            last_dot_phase = dot_phase ^ (last_buf == NUM_SMEM_BUFFERS - 1)
            tlx.barrier_wait(smem_empty_bars[last_buf], last_dot_phase)

            tmem_result = tlx.local_load(acc_tmem_buffer[0])

            tlx.barrier_wait(y_load_bar, 0)
            y = tlx.local_load(y_buf_view)

            z = (tmem_result + y.to(tl.float32)).to(z_desc.dtype)
            z_buf_view = tlx.local_view(z_buffer, 0)
            tlx.local_store(z_buf_view, z)
            tlx.async_descriptor_store(z_desc, z_buf_view, [offs_xm, offs_wn])
            tlx.async_descriptor_store_wait(0)


@triton_autotune(
    configs=get_mm_configs(pre_hook=_addmm_tma_set_block_size_hook),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs_for_pair_cta},
)
@triton.jit
def _addmm_fwd_tma_ws_persistent(
    x_desc,
    w_desc,
    y_desc,
    z_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    PAIR_CTA: tl.constexpr,
):
    # Allocate buffers once for all tiles
    x_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), x_desc.dtype, NUM_SMEM_BUFFERS)
    # In pair CTA mode, each CTA only needs to load half of W
    if PAIR_CTA:
        w_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N // 2), w_desc.dtype, NUM_SMEM_BUFFERS)
    else:
        w_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), w_desc.dtype, NUM_SMEM_BUFFERS)
    tmem_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem)
    if EPILOGUE_SUBTILE:
        if BROADCAST_Y:
            y_buffer_first = tlx.local_alloc((1, BLOCK_N // 2), y_desc.dtype, tl.constexpr(1))
            y_buffer_second = tlx.local_alloc((1, BLOCK_N // 2), y_desc.dtype, tl.constexpr(1))
        else:
            y_buffer_first = tlx.local_alloc((BLOCK_M, BLOCK_N // 2), y_desc.dtype, tl.constexpr(1))
            y_buffer_second = tlx.local_alloc((BLOCK_M, BLOCK_N // 2), y_desc.dtype, tl.constexpr(1))
    else:
        if BROADCAST_Y:
            y_buffer = tlx.local_alloc((1, BLOCK_N), y_desc.dtype, tl.constexpr(1))
        else:
            y_buffer = tlx.local_alloc((BLOCK_M, BLOCK_N), y_desc.dtype, tl.constexpr(1))

    if PAIR_CTA:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        cta_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=2)

    # Barriers for producer <-> MMA
    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    # Barriers for MMA <-> Epilogue
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    # Barriers for producer <-> Epilogue
    # y_load_bar: producer signals when y data is ready
    # y_empty_bar: epilogue signals when done using y buffer
    if EPILOGUE_SUBTILE:
        y_load_bar_first = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        y_load_bar_second = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        y_empty_bar_first = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        y_empty_bar_second = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    else:
        y_load_bar = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        y_empty_bar = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    with tlx.async_tasks():
        # Epilogue consumer: waits for Y from producer, adds bias, stores Z
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            tmem_read_phase = 0
            cur_tmem_buf = 0
            y_load_phase = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS)
                offs_xm = pid_m * BLOCK_M
                offs_wn = pid_n * BLOCK_N

                # Wait for MMA to finish computing this tile
                tlx.barrier_wait(tmem_full_bars[cur_tmem_buf], tmem_read_phase)
                tmem_read_phase = tmem_read_phase ^ (cur_tmem_buf == int(NUM_TMEM_BUFFERS) - 1)

                # Load result from TMEM and add bias
                acc_tmem = tmem_buffers[cur_tmem_buf]

                if EPILOGUE_SUBTILE:
                    # Wait for y loads from producer
                    # pyre-ignore[61]
                    y_bar_first = tlx.local_view(y_load_bar_first, 0)
                    # pyre-ignore[61]
                    y_bar_second = tlx.local_view(y_load_bar_second, 0)
                    # pyre-ignore[61]
                    y_empty_first = tlx.local_view(y_empty_bar_first, 0)
                    # pyre-ignore[61]
                    y_empty_second = tlx.local_view(y_empty_bar_second, 0)
                    tlx.barrier_wait(y_bar_first, y_load_phase)
                    tlx.barrier_wait(y_bar_second, y_load_phase)

                    # Process first half of the tile
                    acc_tmem_subslice1 = tlx.subslice(acc_tmem, 0, BLOCK_N // 2)
                    result = tlx.local_load(acc_tmem_subslice1)
                    # pyre-ignore[61]
                    y_buf_first_view = tlx.local_view(y_buffer_first, 0)
                    y = tlx.local_load(y_buf_first_view)
                    z = (result + y.to(tl.float32)).to(z_desc.dtype)
                    z_desc.store([offs_xm, offs_wn], z)

                    # Process second half of the tile
                    acc_tmem_subslice2 = tlx.subslice(acc_tmem, BLOCK_N // 2, BLOCK_N // 2)
                    result = tlx.local_load(acc_tmem_subslice2)
                    # pyre-ignore[61]
                    y_buf_second_view = tlx.local_view(y_buffer_second, 0)
                    y = tlx.local_load(y_buf_second_view)
                    z = (result + y.to(tl.float32)).to(z_desc.dtype)
                    z_desc.store([offs_xm, offs_wn + BLOCK_N // 2], z)

                    tlx.barrier_arrive(y_empty_first, 1)
                    tlx.barrier_arrive(y_empty_second, 1)
                    y_load_phase = y_load_phase ^ 1
                else:
                    # Wait for y load from producer
                    # pyre-ignore[61]
                    y_bar = tlx.local_view(y_load_bar, 0)
                    # pyre-ignore[61]
                    y_empty = tlx.local_view(y_empty_bar, 0)
                    tlx.barrier_wait(y_bar, y_load_phase)

                    # Load y from SMEM
                    # pyre-ignore[61]
                    y_buf_view = tlx.local_view(y_buffer, 0)
                    y = tlx.local_load(y_buf_view)
                    result = tlx.local_load(acc_tmem)
                    z = (result + y.to(tl.float32)).to(z_desc.dtype)
                    z_desc.store([offs_xm, offs_wn], z)

                    tlx.barrier_arrive(y_empty, 1)
                    y_load_phase = y_load_phase ^ 1

                # Signal MMA that this TMEM buffer is now free
                tlx.barrier_arrive(tmem_empty_bars[cur_tmem_buf], 1)

                cur_tmem_buf = (cur_tmem_buf + 1) % int(NUM_TMEM_BUFFERS)

        # MMA consumer: performs matrix multiplication
        with tlx.async_task(num_warps=4, num_regs=232):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_K)

            dot_phase = 0
            tmem_write_phase = 1
            cur_tmem_buf = 0
            processed_k_iters = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS)

                # Wait for epilogue to finish with this TMEM buffer
                tlx.barrier_wait(tmem_empty_bars[cur_tmem_buf], tmem_write_phase)
                tmem_write_phase = tmem_write_phase ^ (cur_tmem_buf == int(NUM_TMEM_BUFFERS) - 1)

                # Perform K-dimension reduction
                for k in range(0, k_tiles):
                    buf = (processed_k_iters + k) % int(NUM_SMEM_BUFFERS)
                    tlx.barrier_wait(smem_full_bars[buf], dot_phase)

                    # CTA0 waits for both CTA0 and CTA1 to finish loading before issuing dot op
                    # "Arrive Remote, Wait Local" pattern: all CTAs signal CTA 0's barrier, only CTA 0 waits
                    if PAIR_CTA:
                        # pyre-ignore[61]
                        tlx.barrier_arrive(cta_bars[buf], 1, remote_cta_rank=0)
                        # pyre-ignore[61]
                        tlx.barrier_wait(cta_bars[buf], phase=dot_phase, pred=pred_cta0)

                    tlx.async_dot(
                        x_buffers[buf],
                        w_buffers[buf],
                        tmem_buffers[cur_tmem_buf],
                        use_acc=(k > 0),
                        mBarriers=[smem_empty_bars[buf]],
                        two_ctas=PAIR_CTA,
                        out_dtype=tl.float32,
                    )

                    dot_phase = dot_phase ^ (buf == int(NUM_SMEM_BUFFERS) - 1)

                # Wait for last MMA to complete
                last_buf = (processed_k_iters + k_tiles - 1) % int(NUM_SMEM_BUFFERS)
                last_dot_phase = dot_phase ^ (last_buf == int(NUM_SMEM_BUFFERS) - 1)
                tlx.barrier_wait(smem_empty_bars[last_buf], last_dot_phase)

                # Signal epilogue that result is ready
                tlx.barrier_arrive(tmem_full_bars[cur_tmem_buf], 1)

                cur_tmem_buf = (cur_tmem_buf + 1) % int(NUM_TMEM_BUFFERS)
                processed_k_iters += k_tiles

        # Producer: TMA loads for X, W, and Y
        with tlx.async_task(num_warps=1, num_regs=24):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_K)

            load_phase = 0
            y_load_phase = 0
            processed_k_iters = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS)
                offs_xm = pid_m * BLOCK_M
                # Full tile offset for y loading (both CTAs use same y)
                offs_wn_full = pid_n * BLOCK_N
                # Split W into two parts so each CTA has different offset
                if PAIR_CTA:
                    # pyre-ignore[61]
                    offs_wn = pid_n * BLOCK_N + cluster_cta_rank * (BLOCK_N // 2)
                else:
                    offs_wn = pid_n * BLOCK_N

                for k in range(0, k_tiles):
                    buf = (processed_k_iters + k) % int(NUM_SMEM_BUFFERS)

                    # Wait for buffer to be free
                    tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)

                    offs_k = k * BLOCK_K
                    if PAIR_CTA:
                        tlx.barrier_expect_bytes(
                            smem_full_bars[buf],
                            2 * (BLOCK_M + BLOCK_N // 2) * BLOCK_K,
                        )
                    else:
                        tlx.barrier_expect_bytes(
                            smem_full_bars[buf],
                            2 * (BLOCK_M + BLOCK_N) * BLOCK_K,
                        )
                    tlx.async_descriptor_load(x_desc, x_buffers[buf], [offs_xm, offs_k], smem_full_bars[buf])
                    tlx.async_descriptor_load(w_desc, w_buffers[buf], [offs_k, offs_wn], smem_full_bars[buf])

                    load_phase = load_phase ^ (buf == int(NUM_SMEM_BUFFERS) - 1)

                # Loads y for the tile
                if EPILOGUE_SUBTILE:
                    # pyre-ignore[61]
                    y_buf_first_view = tlx.local_view(y_buffer_first, 0)
                    # pyre-ignore[61]
                    y_buf_second_view = tlx.local_view(y_buffer_second, 0)
                    # pyre-ignore[61]
                    y_bar_first = tlx.local_view(y_load_bar_first, 0)
                    # pyre-ignore[61]
                    y_bar_second = tlx.local_view(y_load_bar_second, 0)
                    # pyre-ignore[61]
                    y_empty_first = tlx.local_view(y_empty_bar_first, 0)
                    # pyre-ignore[61]
                    y_empty_second = tlx.local_view(y_empty_bar_second, 0)

                    # Wait for epilogue to finish using previous y data
                    if tile_id > start_pid:
                        tlx.barrier_wait(y_empty_first, y_load_phase ^ 1)
                        tlx.barrier_wait(y_empty_second, y_load_phase ^ 1)

                    if BROADCAST_Y:
                        tlx.barrier_expect_bytes(y_bar_first, 1 * (BLOCK_N // 2) * 2)
                        tlx.barrier_expect_bytes(y_bar_second, 1 * (BLOCK_N // 2) * 2)
                        tlx.async_descriptor_load(y_desc, y_buf_first_view, [0, offs_wn_full], y_bar_first)
                        tlx.async_descriptor_load(
                            y_desc,
                            y_buf_second_view,
                            [0, offs_wn_full + BLOCK_N // 2],
                            y_bar_second,
                        )
                    else:
                        tlx.barrier_expect_bytes(y_bar_first, BLOCK_M * (BLOCK_N // 2) * 2)
                        tlx.barrier_expect_bytes(y_bar_second, BLOCK_M * (BLOCK_N // 2) * 2)
                        tlx.async_descriptor_load(
                            y_desc,
                            y_buf_first_view,
                            [offs_xm, offs_wn_full],
                            y_bar_first,
                        )
                        tlx.async_descriptor_load(
                            y_desc,
                            y_buf_second_view,
                            [offs_xm, offs_wn_full + BLOCK_N // 2],
                            y_bar_second,
                        )

                    y_load_phase = y_load_phase ^ 1
                else:
                    # Load full y tile
                    # pyre-ignore[61]
                    y_buf_view = tlx.local_view(y_buffer, 0)
                    # pyre-ignore[61]
                    y_bar = tlx.local_view(y_load_bar, 0)
                    # pyre-ignore[61]
                    y_empty = tlx.local_view(y_empty_bar, 0)

                    # Wait for epilogue to finish using previous y data
                    if tile_id > start_pid:
                        tlx.barrier_wait(y_empty, y_load_phase ^ 1)

                    if BROADCAST_Y:
                        tlx.barrier_expect_bytes(y_bar, 1 * BLOCK_N * 2)
                        tlx.async_descriptor_load(y_desc, y_buf_view, [0, offs_wn_full], y_bar)
                    else:
                        tlx.barrier_expect_bytes(y_bar, BLOCK_M * BLOCK_N * 2)
                        tlx.async_descriptor_load(y_desc, y_buf_view, [offs_xm, offs_wn_full], y_bar)

                    y_load_phase = y_load_phase ^ 1

                processed_k_iters += k_tiles


@torch.fx.wrap
def triton_addmm_fwd_tma_persistent(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    warp_specialize: bool = False,
    pair_cta: bool = False,
) -> torch.Tensor:
    """Standard Triton 2-CTA implementation using tl.dot(..., two_ctas=True).
    
    Args:
        x: Input matrix (M, K)
        w: Weight matrix (K, N)
        y: Bias matrix (M, N) or (N,) for broadcast
        warp_specialize: Enable warp specialization
        pair_cta: Enable 2-CTA mode for cooperative MMA
    """
    M, K = x.shape
    _, N = w.shape

    is_y_1d = y.dim() == 1

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    # Configuration for PAIR_CTA mode (bypasses autotuner for proper cluster config)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    GROUP_M = 8

    # Check if PAIR_CTA is compatible with the problem size
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    # PAIR_CTA (2-CTA) mode requirements
    pair_cta_compatible = (
        (num_tiles_m % 2 == 0) and
        (total_tiles % 2 == 0) and
        (BLOCK_M >= 128) and
        (BLOCK_N >= 64)
    )
    PAIR_CTA = pair_cta and pair_cta_compatible

    # Set ctas_per_cga for 2-CTA cluster launch
    ctas_per_cga = (2, 1, 1) if PAIR_CTA else None

    # A dummy block value that will be overwritten by the hook
    dummy_block = [1, 1]
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    w_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    y = y.reshape(1, -1) if is_y_1d else y
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    y_desc = TensorDescriptor(y, y.shape, y.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    z_desc = TensorDescriptor(z, z.shape, z.stride(), dummy_block)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    if PAIR_CTA:
        # Use direct JIT call with ctas_per_cga for cluster launch
        # Note: For standard tl.dot with two_ctas=True, we set PAIR_CTA=False
        # in nargs because the kernel code loads full BLOCK_N and the backend
        # Transform2CTALoads pass handles adding offsets. However, the descriptor
        # shape needs to match what the kernel code expects.
        # TODO: The current approach doesn't properly split the work - both CTAs
        # load the same data. Need to investigate further.
        nargs = {
            "x_desc": x_desc,
            "w_desc": w_desc,
            "y_desc": y_desc,
            "z_desc": z_desc,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "PAIR_CTA": False,  # Don't split W for standard tl.dot
            "EPILOGUE_SUBTILE": False,
            "BROADCAST_Y": is_y_1d,
        }
        _addmm_tma_set_block_size_hook(nargs)

        grid = (min(NUM_SMS, total_tiles), )

        _addmm_fwd_tma_persistent.fn[grid](
            x_desc,
            w_desc,
            y_desc,
            z_desc,
            M,
            N,
            K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            BROADCAST_Y=is_y_1d,
            WARP_SPECIALIZE=warp_specialize,
            NUM_SMS=NUM_SMS,
            PAIR_CTA=PAIR_CTA,
            ctas_per_cga=ctas_per_cga,
        )
    else:
        # Use autotuner for non-PAIR_CTA mode
        def grid(meta):
            nonlocal x_desc, w_desc, z_desc
            BLOCK_M = meta["BLOCK_M"]
            BLOCK_N = meta["BLOCK_N"]
            return (min(
                NUM_SMS,
                triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            ), )

        _addmm_fwd_tma_persistent[grid](
            x_desc,
            w_desc,
            y_desc,
            z_desc,
            M,
            N,
            K,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            BROADCAST_Y=is_y_1d,
            WARP_SPECIALIZE=warp_specialize,
            NUM_SMS=NUM_SMS,
            PAIR_CTA=False,
        )
    return z


@torch.fx.wrap
def triton_addmm_fwd_tma_ws_tlx(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    _, N = w.shape

    is_y_1d = y.dim() == 1

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    w_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    y = y.reshape(1, -1) if is_y_1d else y
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    y_desc = TensorDescriptor(y, y.shape, y.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    z_desc = TensorDescriptor(z, z.shape, z.stride(), dummy_block)

    def grid(meta):
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        return (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

    _addmm_fwd_tma_ws[grid](
        x_desc, w_desc, y_desc, z_desc, M, N, K, ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32, BROADCAST_Y=is_y_1d,
        NUM_SMEM_BUFFERS=2,  # Double buffering
    )
    return z


@torch.fx.wrap
def triton_addmm_fwd_tma_ws_persistent_tlx(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    pair_cta: bool = True,
) -> torch.Tensor:
    M, K = x.shape
    _, N = w.shape

    is_y_1d = y.dim() == 1

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    if is_y_1d:
        NUM_SMEM_BUFFERS = 5
        NUM_TMEM_BUFFERS = 2
    else:
        NUM_SMEM_BUFFERS = 4
        NUM_TMEM_BUFFERS = 2
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten by the hook
    dummy_block = [1, 1]
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    w_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    y = y.reshape(1, -1) if is_y_1d else y
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    y_desc = TensorDescriptor(y, y.shape, y.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    z_desc = TensorDescriptor(z, z.shape, z.stride(), dummy_block)

    # Configuration - use fixed config for now since we're bypassing autotuner
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    GROUP_M = 8
    EPILOGUE_SUBTILE = True

    # Check if PAIR_CTA is compatible with the problem size
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    # PAIR_CTA (2-CTA) mode requirements:
    # - Even number of M tiles (so CTA pairs tile evenly)
    # - Even total tiles
    # - BLOCK_M >= 128 for tcgen05 MMA
    # - BLOCK_N >= 64 (each CTA loads BLOCK_N // 2)
    pair_cta_compatible = (
        (num_tiles_m % 2 == 0) and
        (total_tiles % 2 == 0) and
        (BLOCK_M >= 128) and
        (BLOCK_N >= 64)
    )
    PAIR_CTA = pair_cta and pair_cta_compatible

    # Set ctas_per_cga for 2-CTA cluster launch (2 CTAs in M dimension)
    ctas_per_cga = (2, 1, 1) if PAIR_CTA else None

    # Use the hook to properly set TMA descriptor block sizes
    nargs = {
        "x_desc": x_desc,
        "w_desc": w_desc,
        "y_desc": y_desc,
        "z_desc": z_desc,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "PAIR_CTA": PAIR_CTA,
        "EPILOGUE_SUBTILE": EPILOGUE_SUBTILE,
        "BROADCAST_Y": is_y_1d,
    }
    _addmm_tma_set_block_size_hook(nargs)

    grid = (min(NUM_SMS, total_tiles), )

    # Use direct JIT call (.fn[grid]) with ctas_per_cga for cluster launch
    # This bypasses autotuner but allows passing ctas_per_cga for 2-CTA mode
    _addmm_fwd_tma_ws_persistent.fn[grid](
        x_desc,
        w_desc,
        y_desc,
        z_desc,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=is_y_1d,
        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
        NUM_SMS=NUM_SMS,
        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
        PAIR_CTA=PAIR_CTA,
        ctas_per_cga=ctas_per_cga,
    )
    return z


@torch.fx.wrap
def triton_addmm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    KB, N = w.shape
    assert K == KB, f"incompatible dimensions {K}, {KB}"

    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f"incompatible dimensions {N}, {NY}"

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    grid = lambda meta: (  # noqa E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _addmm_fwd[grid](
        x,
        w,
        y,
        z,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0) if not is_y_1d else 0,
        y.stride(1) if not is_y_1d else y.stride(0),
        z.stride(0),
        z.stride(1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=is_y_1d,
    )
    return z


def triton_addmm_bwd(
    x: torch.Tensor,
    w: torch.Tensor,
    dz: torch.Tensor,
    is_y_1d: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_y_1d:
        dy = torch.sum(dz, dim=0)
    else:
        dy = dz
    dw = torch.mm(x.t(), dz)
    dx = torch.mm(dz, w.t())

    return dx, dw, dy


@torch.fx.wrap
def maybe_triton_addmm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # triton addmm is slower than torch (cublas) on AMD/Blackwell.
    # Default to pytorch addmm on AMD/Blackwell for now.
    if is_sm100_plus() or torch.version.hip is not None:
        return torch.addmm(y, x, w)
    else:
        return triton_addmm_fwd(x=x, w=w, y=y)


class _AddMmFunction(torch.autograd.Function):

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        ctx.is_y_1d = y.dim() == 1
        if is_sm100_plus() and TMA_AVAILABLE and _check_tma_alignment(x, w, y):
            if x.dtype == torch.float32 or HAS_TLX == False:
                # use TMA persistent kernel on sm100
                return triton_addmm_fwd_tma_persistent(x, w, y, warp_specialize=True)
            else:
                return triton_addmm_fwd_tma_ws_persistent_tlx(
                    x, w, y)  # tlx.async_dot doesn't support fp32 inputs because of WGMMA requirements
        else:
            return triton_addmm_fwd(x, w, y)

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (x, w) = ctx.saved_tensors
        return triton_addmm_bwd(x, w, dz, ctx.is_y_1d)


def triton_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    return _AddMmFunction.apply(mat1, mat2, input)


# ============================================================================
# Benchmark and Test Code
# ============================================================================


def benchmark_addmm(
    M: int,
    K: int,
    N: int,
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 10,
    num_iters: int = 100,
    device: str = "cuda",
) -> dict:
    """Benchmark triton_addmm against torch.addmm."""
    # Create input tensors
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(K, N, dtype=dtype, device=device)
    y = torch.randn(N, dtype=dtype, device=device)

    # Warmup
    for _ in range(num_warmup):
        _ = triton_addmm(y, x, w)
        _ = torch.addmm(y, x, w)
    torch.cuda.synchronize()

    # Benchmark triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _ = triton_addmm(y, x, w)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / num_iters

    # Benchmark torch
    start.record()
    for _ in range(num_iters):
        _ = torch.addmm(y, x, w)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / num_iters

    # Calculate TFLOPS
    flops = 2 * M * N * K + M * N  # matmul + add
    triton_tflops = flops / (triton_time * 1e-3) / 1e12
    torch_tflops = flops / (torch_time * 1e-3) / 1e12

    return {
        "M": M,
        "K": K,
        "N": N,
        "dtype": str(dtype),
        "triton_time_ms": triton_time,
        "torch_time_ms": torch_time,
        "triton_tflops": triton_tflops,
        "torch_tflops": torch_tflops,
        "speedup": torch_time / triton_time,
    }


def benchmark_tma_persistent(
    M: int,
    K: int,
    N: int,
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 10,
    num_iters: int = 100,
    device: str = "cuda",
    warp_specialize: bool = True,
) -> dict:
    """Benchmark triton_addmm_fwd_tma_persistent against torch.addmm."""
    if not TMA_AVAILABLE:
        return {"error": "TMA not available"}

    # Create input tensors
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(K, N, dtype=dtype, device=device)
    y = torch.randn(N, dtype=dtype, device=device)

    # Check alignment
    if not _check_tma_alignment(x, w, y):
        return {"error": "TMA alignment requirements not met"}

    # Warmup
    for _ in range(num_warmup):
        _ = torch.addmm(y, x, w)
    torch.cuda.synchronize()

    # Benchmark triton TMA persistent
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _ = triton_addmm_fwd_tma_persistent(x, w, y, warp_specialize=warp_specialize)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / num_iters

    # Benchmark torch
    start.record()
    for _ in range(num_iters):
        _ = torch.addmm(y, x, w)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / num_iters

    # Calculate TFLOPS
    flops = 2 * M * N * K + M * N  # matmul + add
    triton_tflops = flops / (triton_time * 1e-3) / 1e12
    torch_tflops = flops / (torch_time * 1e-3) / 1e12

    return {
        "M": M,
        "K": K,
        "N": N,
        "dtype": str(dtype),
        "warp_specialize": warp_specialize,
        "triton_time_ms": triton_time,
        "torch_time_ms": torch_time,
        "triton_tflops": triton_tflops,
        "torch_tflops": torch_tflops,
        "speedup": torch_time / triton_time,
    }


def test_correctness(
    M: int = 512,
    K: int = 512,
    N: int = 512,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> bool:
    """Test correctness of triton_addmm against torch.addmm."""
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(K, N, dtype=dtype, device=device)
    y = torch.randn(N, dtype=dtype, device=device)

    # Compute with triton
    triton_out = triton_addmm(y, x, w)

    # Compute with torch
    torch_out = torch.addmm(y, x, w)

    # Check correctness
    if dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2
    elif dtype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1
    else:
        rtol, atol = 1e-4, 1e-4

    is_close = torch.allclose(triton_out, torch_out, rtol=rtol, atol=atol)
    max_diff = (triton_out - torch_out).abs().max().item()

    return is_close, max_diff


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark triton_addmm")
    parser.add_argument("--M", type=int, default=4096, help="M dimension")
    parser.add_argument("--K", type=int, default=4096, help="K dimension")
    parser.add_argument("--N", type=int, default=4096, help="N dimension")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--test-only", action="store_true", help="Only run correctness test")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmark")
    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("Triton AddMM Benchmark")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"SM100+ (Blackwell): {is_sm100_plus()}")
    print(f"TMA Available: {TMA_AVAILABLE}")
    print(f"TLX Available: {HAS_TLX}")
    print("=" * 60)

    if not args.benchmark_only:
        # Test correctness
        print("\n[Correctness Test]")
        # Use sizes that are compatible with 2-CTA constraints
        # 2-CTA requires BLOCK_M >= 128, so we use larger matrices
        test_sizes = [
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (args.M, args.K, args.N),
        ]

        all_passed = True
        for M, K, N in test_sizes:
            try:
                is_correct, max_diff = test_correctness(M, K, N, dtype)
                status = "PASS" if is_correct else "FAIL"
                print(f"  [{status}] M={M}, K={K}, N={N}, max_diff={max_diff:.6f}")
                if not is_correct:
                    all_passed = False
            except Exception as e:
                print(f"  [ERROR] M={M}, K={K}, N={N}: {e}")
                all_passed = False

        if all_passed:
            print("\n✓ All correctness tests passed!")
        else:
            print("\n✗ Some correctness tests failed!")

    if not args.test_only:
        # Run benchmark
        print("\n[Benchmark]")
        print(f"  Shape: M={args.M}, K={args.K}, N={args.N}")
        print(f"  Dtype: {args.dtype}")
        print(f"  Warmup: {args.warmup}, Iters: {args.iters}")
        print()

        results = benchmark_addmm(
            M=args.M,
            K=args.K,
            N=args.N,
            dtype=dtype,
            num_warmup=args.warmup,
            num_iters=args.iters,
        )

        print(f"  Triton: {results['triton_time_ms']:.3f} ms "
              f"({results['triton_tflops']:.2f} TFLOPS)")
        print(f"  Torch:  {results['torch_time_ms']:.3f} ms "
              f"({results['torch_tflops']:.2f} TFLOPS)")
        print(f"  Speedup: {results['speedup']:.2f}x")

        # Benchmark multiple sizes
        print("\n[Multi-size Benchmark]")
        print("-" * 80)
        print(f"{'M':>6} {'K':>6} {'N':>6} {'Triton(ms)':>12} "
              f"{'Torch(ms)':>12} {'Triton TFLOPS':>14} {'Speedup':>8}")
        print("-" * 80)

        benchmark_sizes = [
            (4096, 4096, 4096),
        ]

        for M, K, N in benchmark_sizes:
            try:
                res = benchmark_addmm(M, K, N, dtype, args.warmup, args.iters)
                print(f"{M:>6} {K:>6} {N:>6} {res['triton_time_ms']:>12.3f} "
                      f"{res['torch_time_ms']:>12.3f} {res['triton_tflops']:>14.2f} "
                      f"{res['speedup']:>8.2f}x")
            except Exception as e:
                print(f"{M:>6} {K:>6} {N:>6} ERROR: {e}")

        print("-" * 80)

        # Benchmark TMA Persistent kernel (SM100+ only)
        if is_sm100_plus() and TMA_AVAILABLE:
            # Benchmark with warp specialization
            print("\n[TMA Persistent Benchmark (warp_specialize=True)]")
            print("-" * 80)
            print(f"{'M':>6} {'K':>6} {'N':>6} {'TMA(ms)':>12} "
                  f"{'Torch(ms)':>12} {'TMA TFLOPS':>14} {'Speedup':>8}")
            print("-" * 80)

            for M, K, N in benchmark_sizes:
                try:
                    res = benchmark_tma_persistent(M, K, N, dtype, args.warmup, args.iters, warp_specialize=True)
                    if "error" in res:
                        print(f"{M:>6} {K:>6} {N:>6} SKIP: {res['error']}")
                    else:
                        print(f"{M:>6} {K:>6} {N:>6} "
                              f"{res['triton_time_ms']:>12.3f} "
                              f"{res['torch_time_ms']:>12.3f} "
                              f"{res['triton_tflops']:>14.2f} "
                              f"{res['speedup']:>8.2f}x")
                except Exception as e:
                    err_str = str(e)
                    if "PassManager" in err_str:
                        print(f"{M:>6} {K:>6} {N:>6} COMPILER_BUG")
                    else:
                        print(f"{M:>6} {K:>6} {N:>6} ERROR: {e}")

            print("-" * 80)
        else:
            print("\n[TMA Persistent Benchmark] Skipped (requires SM100+ and TMA)")
