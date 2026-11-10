"""
MXFP TDM-pipelined GEMM for AMD gfx1250 using TLX.

``mxgemm_tdm_pipelined_kernel`` mirrors the structure of the Gluon MXFP GEMM
example: descriptor-backed loads for A/B and scales, optional A scales, L2
prefetch hints, and baseline or sliced K/N/MNK compute schedules.

``mxgemm_tdm_persistent_kernel`` is a persistent sliceMNK kernel for
256x256 output tiles with FP8 A and FP8 or FP4 B. It carries its TDM input
ring across output tiles, prefetches the next tile during the final K steps,
stages C through LDS for TDM stores, and can multicast inputs across
workgroup clusters. See README.md for the design. The kernel, its device
helpers, and ``pack_scale`` live in ``triton.tlx.ops.kernels.mm_mxfp8.gfx1250``,
which backs ``tlx.ops.mm_mxfp8`` on gfx1250; this module imports them and
adds the non-persistent kernel, the launcher with every schedule option,
the CLI, and benchmarking.

``bench.py`` runs the persistent kernel over a shape sweep in separate
processes.
"""
from collections.abc import Callable
import statistics
import time

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton.tlx.ops.kernels.mm_mxfp8.gfx1250 import (  # noqa: F401
    _XCD_REMAP_MODES, _mxgemm_c_slot, _mxgemm_c_slot_wait, _mxgemm_c_tdm_store, _mxgemm_can_be_cluster_step,
    _mxgemm_cluster_arrive, _mxgemm_cluster_program_id, _mxgemm_defer_bottom_k_step, _mxgemm_desc_group_rows,
    _mxgemm_flat_view, _mxgemm_grid_cols, _mxgemm_grid_masks, _mxgemm_grid_rows, _mxgemm_issue_load_a_data,
    _mxgemm_issue_load_a_scale, _mxgemm_issue_load_b_data, _mxgemm_issue_load_b_scale, _mxgemm_issue_loads,
    _mxgemm_load_a_operand, _mxgemm_load_b_operand, _mxgemm_native_dot, _mxgemm_native_operand, _mxgemm_overlap_store,
    _mxgemm_persistent_a, _mxgemm_persistent_b, _mxgemm_persistent_compute_256, _mxgemm_persistent_compute_top,
    _mxgemm_persistent_load, _mxgemm_persistent_store, _mxgemm_position_load_descriptor, _mxgemm_remap_program_id,
    _mxgemm_scale_cols, _mxgemm_scale_quads, _mxgemm_schedule_prefetch, _mxgemm_tile_offsets, _mxgemm_view_rows,
    _mxgemm_w256_scale_view, _operand_shared_layout, _partitioned_operand_shared_layout, _scale_shared_layout,
    mxgemm_tdm_persistent_kernel, pack_scale,
)


@triton.constexpr_function
def _b_shared_layout(block_shape, transpose_b):
    if transpose_b:
        return _operand_shared_layout(block_shape)
    return tlx.padded_shared_layout_encoding.with_identity_for([[block_shape[1], 16]], block_shape, [1, 0])


def is_gfx1250_available():
    try:
        target = triton.runtime.driver.active.get_current_target()
        return target.arch == "gfx1250"
    except Exception:
        return False


def fp8e8m0_to_float32(scale: torch.Tensor) -> torch.Tensor:
    scale = scale.view(torch.uint8).to(torch.int32)
    scale = scale << 23
    return scale.view(torch.float32)


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K):
    if a_scale is None:
        a_scale_f32 = torch.full((M, K), 1.0, dtype=torch.float32)
    else:
        a_scale_f32 = fp8e8m0_to_float32(a_scale).repeat_interleave(scale_block, dim=1)[:M, :K]
    b_scale_f32 = fp8e8m0_to_float32(b_scale).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]
    return torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32)


@triton.jit
def _mxgemm_issue_split_loads(
    a0_desc,
    a1_desc,
    b0_desc,
    b1_desc,
    a_scale_desc,
    b_scale_desc,
    a0_buf,
    a1_buf,
    b0_buf,
    b1_buf,
    a_scale_buf,
    b_scale_buf,
    load_idx,
    pred,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
    TDM_FUSION: tl.constexpr,
):
    slot = load_idx % NUM_BUFFERS
    a_offsets = [0, load_idx * BLOCK_K_PACKED_A]
    if TRANSPOSE_B:
        b_offsets = [0, load_idx * BLOCK_K_PACKED_B]
    else:
        b_offsets = [load_idx * BLOCK_K_PACKED_B, 0]
    if SCALE_PRESHUFFLE:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
    else:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]

    if TDM_FUSION == "partial":
        tl.static_assert(WITH_A_SCALE, "split partial TDM fusion requires WITH_A_SCALE")
        a0_load_desc = tlx.update_tensor_descriptor(a0_desc, add_offsets=a_offsets)
        a1_load_desc = tlx.update_tensor_descriptor(a1_desc, add_offsets=a_offsets)
        b0_load_desc = tlx.update_tensor_descriptor(b0_desc, add_offsets=b_offsets)
        b1_load_desc = tlx.update_tensor_descriptor(b1_desc, add_offsets=b_offsets)
        a_scale_load_desc = tlx.update_tensor_descriptor(a_scale_desc, add_offsets=scale_offsets)
        b_scale_load_desc = tlx.update_tensor_descriptor(b_scale_desc, add_offsets=scale_offsets)
        tlx.async_amd_descriptor_load_fused([
            (tlx.update_tensor_descriptor(a0_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(a0_buf, slot), 5),
            (tlx.update_tensor_descriptor(b0_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(b0_buf, slot), 10),
        ])
        tlx.async_amd_descriptor_load_fused([
            (tlx.update_tensor_descriptor(a1_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(a1_buf, slot), 5),
            (tlx.update_tensor_descriptor(b1_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(b1_buf, slot), 10),
        ])
        tlx.async_amd_descriptor_load_fused([
            (tlx.update_tensor_descriptor(a_scale_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(a_scale_buf, slot), 5),
            (tlx.update_tensor_descriptor(b_scale_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(b_scale_buf, slot), 10),
        ])
    else:
        tl.static_assert(TDM_FUSION == "none", "TDM_SPLIT supports TDM_FUSION values: none, partial")
        if WITH_A_SCALE:
            a_scale_load_desc = tlx.update_tensor_descriptor(a_scale_desc, add_offsets=scale_offsets)
            tlx.async_amd_descriptor_load(a_scale_load_desc, tlx.local_view(a_scale_buf, slot), [0, 0], pred=pred,
                                          clamp_bounds=False)
        b_scale_load_desc = tlx.update_tensor_descriptor(b_scale_desc, add_offsets=scale_offsets)
        a0_load_desc = tlx.update_tensor_descriptor(a0_desc, add_offsets=a_offsets)
        a1_load_desc = tlx.update_tensor_descriptor(a1_desc, add_offsets=a_offsets)
        b0_load_desc = tlx.update_tensor_descriptor(b0_desc, add_offsets=b_offsets)
        b1_load_desc = tlx.update_tensor_descriptor(b1_desc, add_offsets=b_offsets)
        tlx.async_amd_descriptor_load(b_scale_load_desc, tlx.local_view(b_scale_buf, slot), [0, 0], pred=pred,
                                      clamp_bounds=False)
        tlx.async_amd_descriptor_load(a0_load_desc, tlx.local_view(a0_buf, slot), [0, 0], pred=pred, clamp_bounds=False)
        tlx.async_amd_descriptor_load(b0_load_desc, tlx.local_view(b0_buf, slot), [0, 0], pred=pred, clamp_bounds=False)
        tlx.async_amd_descriptor_load(a1_load_desc, tlx.local_view(a1_buf, slot), [0, 0], pred=pred, clamp_bounds=False)
        tlx.async_amd_descriptor_load(b1_load_desc, tlx.local_view(b1_buf, slot), [0, 0], pred=pred, clamp_bounds=False)
    return load_idx + 1


@triton.jit
def _mxgemm_issue_l2_prefetches(
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    load_idx,
    pred,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
):
    if L2_PREFETCH_DISTANCE >= 0:
        prefetch_iteration = load_idx + L2_PREFETCH_DISTANCE
        if WITH_A_SCALE:
            if SCALE_PRESHUFFLE:
                a_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED]
            else:
                a_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED // 128]
            tlx.amd_descriptor_prefetch_tensor(a_scale_desc, a_scale_offsets, pred=pred)
        if SCALE_PRESHUFFLE:
            b_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED]
        else:
            b_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED // 128]
        tlx.amd_descriptor_prefetch_tensor(b_scale_desc, b_scale_offsets, pred=pred)
        tlx.amd_descriptor_prefetch_tensor(a_desc, [0, prefetch_iteration * BLOCK_K_PACKED_A], pred=pred)
        if TRANSPOSE_B:
            tlx.amd_descriptor_prefetch_tensor(b_desc, [0, prefetch_iteration * BLOCK_K_PACKED_B], pred=pred)
        else:
            tlx.amd_descriptor_prefetch_tensor(b_desc, [prefetch_iteration * BLOCK_K_PACKED_B, 0], pred=pred)


@triton.jit
def _mxgemm_issue_split_l2_prefetches(
    a0_desc,
    a1_desc,
    b0_desc,
    b1_desc,
    a_scale_desc,
    b_scale_desc,
    load_idx,
    pred,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
):
    if L2_PREFETCH_DISTANCE >= 0:
        prefetch_iteration = load_idx + L2_PREFETCH_DISTANCE
        if WITH_A_SCALE:
            if SCALE_PRESHUFFLE:
                a_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED]
            else:
                a_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED // 128]
            tlx.amd_descriptor_prefetch_tensor(a_scale_desc, a_scale_offsets, pred=pred)
        if SCALE_PRESHUFFLE:
            b_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED]
        else:
            b_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED // 128]
        tlx.amd_descriptor_prefetch_tensor(b_scale_desc, b_scale_offsets, pred=pred)
        tlx.amd_descriptor_prefetch_tensor(a0_desc, [0, prefetch_iteration * BLOCK_K_PACKED_A], pred=pred)
        tlx.amd_descriptor_prefetch_tensor(a1_desc, [0, prefetch_iteration * BLOCK_K_PACKED_A], pred=pred)
        if TRANSPOSE_B:
            b_offsets = [0, prefetch_iteration * BLOCK_K_PACKED_B]
        else:
            b_offsets = [prefetch_iteration * BLOCK_K_PACKED_B, 0]
        tlx.amd_descriptor_prefetch_tensor(b0_desc, b_offsets, pred=pred)
        tlx.amd_descriptor_prefetch_tensor(b1_desc, b_offsets, pred=pred)


@triton.jit
def _mxgemm_issue_l2_prefetches_prologue(
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    load_idx,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
):
    if L2_PREFETCH_DISTANCE >= 0:
        always = tl.full((), True, dtype=tl.int1)
        for i in tl.static_range(NUM_BUFFERS, NUM_BUFFERS + L2_PREFETCH_DISTANCE):
            _mxgemm_issue_l2_prefetches(a_desc, b_desc, a_scale_desc, b_scale_desc, load_idx, always, i,
                                        BLOCK_K_PACKED_A, BLOCK_K_PACKED_B, BLOCK_K_SCALE_PRESHUFFLED, TRANSPOSE_B,
                                        SCALE_PRESHUFFLE, WITH_A_SCALE)


@triton.jit
def _mxgemm_load_a_operand_split(
    a0_buf,
    a1_buf,
    a_scale_buf,
    wmma_idx,
    subtile_start_idx_m: tl.constexpr,
    subtile_start_idx_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIV_FACTOR_A: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_K_SCALE: tl.constexpr,
    BLOCK_M_PRESHUFFLED: tl.constexpr,
    SCALE_KWIDTH: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_SUBTILES_M: tl.constexpr,
    NUM_SUBTILES_K: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
):
    slot = wmma_idx % NUM_BUFFERS
    subtile_m: tl.constexpr = BLOCK_M // NUM_SUBTILES_M
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_m: tl.constexpr = subtile_start_idx_m * subtile_m
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    if subtile_start_idx_m == 0:
        a_view = tlx.local_view(a0_buf, slot)
    else:
        a_view = tlx.local_view(a1_buf, slot)
    if NUM_SUBTILES_K != 1:
        a_view = tlx.local_slice(a_view, [0, subtile_start_k // DIV_FACTOR_A], [subtile_m, subtile_k // DIV_FACTOR_A])
    a = tlx.local_load(a_view)

    if WITH_A_SCALE:
        if SCALE_PRESHUFFLE:
            scale_a_view = tlx.local_reshape(
                tlx.local_view(a_scale_buf, slot),
                [BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 32, 4, SCALE_KWIDTH],
            )
            scale_a_view = tlx.local_trans(scale_a_view, (0, 3, 2, 1, 4))
            scale_a_view = tlx.local_reshape(scale_a_view, [BLOCK_M, BLOCK_K_SCALE])
        else:
            scale_a_view = tlx.local_view(a_scale_buf, slot)
        scale_a_view = tlx.local_slice(scale_a_view, [subtile_start_m, subtile_start_k // SCALE_BLOCK],
                                       [subtile_m, subtile_k // SCALE_BLOCK])
        scale_a = tlx.local_load(scale_a_view)
    else:
        scale_a = tl.full((subtile_m, subtile_k // SCALE_BLOCK), 127, dtype=tl.uint8)

    return a, scale_a


@triton.jit
def _mxgemm_load_b_operand_split(
    b0_buf,
    b1_buf,
    b_scale_buf,
    wmma_idx,
    subtile_start_idx_k: tl.constexpr,
    subtile_start_idx_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIV_FACTOR_B: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_K_SCALE: tl.constexpr,
    BLOCK_N_PRESHUFFLED: tl.constexpr,
    SCALE_KWIDTH: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_SUBTILES_N: tl.constexpr,
    NUM_SUBTILES_K: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
):
    slot = wmma_idx % NUM_BUFFERS
    subtile_n: tl.constexpr = BLOCK_N // NUM_SUBTILES_N
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_n: tl.constexpr = subtile_start_idx_n * subtile_n
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    if subtile_start_idx_n == 0:
        b_view = tlx.local_view(b0_buf, slot)
    else:
        b_view = tlx.local_view(b1_buf, slot)
    if TRANSPOSE_B:
        if NUM_SUBTILES_K != 1:
            b_view = tlx.local_slice(b_view, [0, subtile_start_k // DIV_FACTOR_B],
                                     [subtile_n, subtile_k // DIV_FACTOR_B])
        b = tlx.local_load(tlx.local_trans(b_view))
    else:
        if NUM_SUBTILES_K != 1:
            b_view = tlx.local_slice(b_view, [subtile_start_k // DIV_FACTOR_B, 0],
                                     [subtile_k // DIV_FACTOR_B, subtile_n])
        b = tlx.local_load(b_view)

    if SCALE_PRESHUFFLE:
        scale_b_view = tlx.local_reshape(
            tlx.local_view(b_scale_buf, slot),
            [BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 32, 4, SCALE_KWIDTH],
        )
        scale_b_view = tlx.local_trans(scale_b_view, (0, 3, 2, 1, 4))
        scale_b_view = tlx.local_reshape(scale_b_view, [BLOCK_N, BLOCK_K_SCALE])
    else:
        scale_b_view = tlx.local_view(b_scale_buf, slot)
    scale_b_view = tlx.local_slice(scale_b_view, [subtile_start_n, subtile_start_k // SCALE_BLOCK],
                                   [subtile_n, subtile_k // SCALE_BLOCK])
    scale_b = tlx.local_load(scale_b_view)

    return b, scale_b


@triton.jit
def mxgemm_tdm_pipelined_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale,
        b_scale,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_scale,
        DTYPE_A: tl.constexpr,
        DTYPE_B: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        TRANSPOSE_B: tl.constexpr,
        NUM_BUFFERS: tl.constexpr,
        SCALE_PRESHUFFLE: tl.constexpr = True,
        WITH_A_SCALE: tl.constexpr = True,
        SCHEDULE: tl.constexpr = "baseline",
        TDM_FUSION: tl.constexpr = "none",
        L2_PREFETCH_DISTANCE: tl.constexpr = -1,
        TDM_SPLIT: tl.constexpr = False,
        TILES_PER_WARP: tl.constexpr = (2, 2),
):
    DIV_FACTOR_A: tl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    BLOCK_K_PACKED_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
    BLOCK_K_SCALE: tl.constexpr = BLOCK_K // SCALE_BLOCK
    SCALE_KWIDTH: tl.constexpr = 4 if BLOCK_K_SCALE >= 4 else BLOCK_K_SCALE
    BLOCK_M_PRESHUFFLED: tl.constexpr = BLOCK_M // 128
    BLOCK_N_PRESHUFFLED: tl.constexpr = BLOCK_N // 128
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr = BLOCK_K_SCALE * 128
    STORE_INSTR_M: tl.constexpr = 32 if (DTYPE_A == "e2m1" and DTYPE_B == "e2m1") else 16
    STORE_INSTR_SHAPE: tl.constexpr = (STORE_INSTR_M, 16, 128)
    if SCALE_PRESHUFFLE:
        STORE_WARP_BASES: tl.constexpr = ((0, 2), (2, 0))
        STORE_REG_BASES: tl.constexpr = ((0, 1), (1, 0))
    else:
        STORE_WARP_BASES: tl.constexpr = ((0, 1), (1, 0))
        STORE_REG_BASES: tl.constexpr = ()
    if TDM_FUSION == "4way":
        tl.static_assert(WITH_A_SCALE, "4-way TDM fusion requires WITH_A_SCALE")
        NUM_LOADS_IN_BATCH: tl.constexpr = 1
    elif TDM_FUSION == "2way":
        NUM_LOADS_IN_BATCH: tl.constexpr = 2
    elif TDM_FUSION == "partial":
        tl.static_assert(WITH_A_SCALE, "partial TDM fusion requires WITH_A_SCALE")
        NUM_LOADS_IN_BATCH: tl.constexpr = 2
    else:
        tl.static_assert(TDM_FUSION == "none", "TDM_FUSION must be one of: none, 2way, 4way, partial")
        NUM_LOADS_IN_BATCH: tl.constexpr = 4 if WITH_A_SCALE else 3
    if SCHEDULE == "sliceMNK":
        NUM_SUBTILES_M: tl.constexpr = 2
        NUM_SUBTILES_N: tl.constexpr = 2
        # Native MXFP WMMAs consume K128; two K64 halves would double the
        # matrix instructions for a K128 input stage.
        NUM_SUBTILES_K: tl.constexpr = 1 if BLOCK_K == 128 else 2
    elif SCHEDULE == "sliceNK":
        NUM_SUBTILES_M: tl.constexpr = 1
        NUM_SUBTILES_N: tl.constexpr = 2
        NUM_SUBTILES_K: tl.constexpr = 2
    elif SCHEDULE == "sliceK":
        NUM_SUBTILES_M: tl.constexpr = 1
        NUM_SUBTILES_N: tl.constexpr = 1
        NUM_SUBTILES_K: tl.constexpr = 2
    else:
        tl.static_assert(SCHEDULE == "baseline")
        NUM_SUBTILES_M: tl.constexpr = 1
        NUM_SUBTILES_N: tl.constexpr = 1
        NUM_SUBTILES_K: tl.constexpr = 1
    if TDM_SPLIT:
        tl.static_assert(SCHEDULE == "sliceMNK", "TDM_SPLIT is only supported for the sliceMNK schedule")
        tl.static_assert(BLOCK_M % 2 == 0 and BLOCK_N % 2 == 0, "TDM_SPLIT requires even BLOCK_M and BLOCK_N")
        if TDM_FUSION == "partial":
            tl.static_assert(WITH_A_SCALE, "split partial TDM fusion requires WITH_A_SCALE")
            SPLIT_LOADS_IN_BATCH: tl.constexpr = 3
        else:
            tl.static_assert(TDM_FUSION == "none", "TDM_SPLIT supports TDM_FUSION values: none, partial")
            SPLIT_LOADS_IN_BATCH: tl.constexpr = 6 if WITH_A_SCALE else 5

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if TDM_SPLIT:
        HALF_BLOCK_M: tl.constexpr = BLOCK_M // 2
        HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
        a0_desc = tl.make_tensor_descriptor(
            a_ptr + pid_m * BLOCK_M * stride_am,
            shape=[M, K // DIV_FACTOR_A],
            strides=[stride_am, tl.constexpr(1)],
            block_shape=[HALF_BLOCK_M, BLOCK_K_PACKED_A],
        )
        a1_desc = tl.make_tensor_descriptor(
            a_ptr + pid_m * BLOCK_M * stride_am + HALF_BLOCK_M * stride_am,
            shape=[M, K // DIV_FACTOR_A],
            strides=[stride_am, tl.constexpr(1)],
            block_shape=[HALF_BLOCK_M, BLOCK_K_PACKED_A],
        )
        if TRANSPOSE_B:
            b0_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn,
                shape=[N, K // DIV_FACTOR_B],
                strides=[stride_bn, tl.constexpr(1)],
                block_shape=[HALF_BLOCK_N, BLOCK_K_PACKED_B],
            )
            b1_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn + HALF_BLOCK_N * stride_bn,
                shape=[N, K // DIV_FACTOR_B],
                strides=[stride_bn, tl.constexpr(1)],
                block_shape=[HALF_BLOCK_N, BLOCK_K_PACKED_B],
            )
            b0_shape: tl.constexpr = (HALF_BLOCK_N, BLOCK_K_PACKED_B)
            b1_shape: tl.constexpr = (HALF_BLOCK_N, BLOCK_K_PACKED_B)
        else:
            b0_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn,
                shape=[K // DIV_FACTOR_B, N],
                strides=[stride_bk, tl.constexpr(1)],
                block_shape=[BLOCK_K_PACKED_B, HALF_BLOCK_N],
            )
            b1_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn + HALF_BLOCK_N * stride_bn,
                shape=[K // DIV_FACTOR_B, N],
                strides=[stride_bk, tl.constexpr(1)],
                block_shape=[BLOCK_K_PACKED_B, HALF_BLOCK_N],
            )
            b0_shape: tl.constexpr = (BLOCK_K_PACKED_B, HALF_BLOCK_N)
            b1_shape: tl.constexpr = (BLOCK_K_PACKED_B, HALF_BLOCK_N)
        a_split_layout: tl.constexpr = _operand_shared_layout([HALF_BLOCK_M, BLOCK_K_PACKED_A])
        b_split_layout: tl.constexpr = _b_shared_layout(b0_shape, TRANSPOSE_B)
        a0_buf = tlx.local_alloc((HALF_BLOCK_M, BLOCK_K_PACKED_A), tlx.dtype_of(a_ptr), NUM_BUFFERS,
                                 layout=a_split_layout)
        a1_buf = tlx.local_alloc((HALF_BLOCK_M, BLOCK_K_PACKED_A), tlx.dtype_of(a_ptr), NUM_BUFFERS,
                                 layout=a_split_layout)
        b0_buf = tlx.local_alloc(b0_shape, tlx.dtype_of(b_ptr), NUM_BUFFERS, layout=b_split_layout)
        b1_buf = tlx.local_alloc(b1_shape, tlx.dtype_of(b_ptr), NUM_BUFFERS, layout=b_split_layout)
    else:
        a_desc = tl.make_tensor_descriptor(
            a_ptr + pid_m * BLOCK_M * stride_am,
            shape=[M, K // DIV_FACTOR_A],
            strides=[stride_am, tl.constexpr(1)],
            block_shape=[BLOCK_M, BLOCK_K_PACKED_A],
        )
        if TRANSPOSE_B:
            b_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn,
                shape=[N, K // DIV_FACTOR_B],
                strides=[stride_bn, tl.constexpr(1)],
                block_shape=[BLOCK_N, BLOCK_K_PACKED_B],
            )
            b_layout: tl.constexpr = _operand_shared_layout([BLOCK_N, BLOCK_K_PACKED_B])
            b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K_PACKED_B), tlx.dtype_of(b_ptr), NUM_BUFFERS, layout=b_layout)
        else:
            b_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn,
                shape=[K // DIV_FACTOR_B, N],
                strides=[stride_bk, tl.constexpr(1)],
                block_shape=[BLOCK_K_PACKED_B, BLOCK_N],
            )
            b_layout: tl.constexpr = _b_shared_layout([BLOCK_K_PACKED_B, BLOCK_N], TRANSPOSE_B)
            b_buf = tlx.local_alloc((BLOCK_K_PACKED_B, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS, layout=b_layout)

    if SCALE_PRESHUFFLE:
        a_scale_desc = tl.make_tensor_descriptor(
            a_scale + pid_m * BLOCK_M_PRESHUFFLED * stride_scale,
            shape=[M // 128, K // SCALE_BLOCK * 128],
            strides=[stride_scale, tl.constexpr(1)],
            block_shape=[BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED],
        )
        b_scale_desc = tl.make_tensor_descriptor(
            b_scale + pid_n * BLOCK_N_PRESHUFFLED * stride_scale,
            shape=[N // 128, K // SCALE_BLOCK * 128],
            strides=[stride_scale, tl.constexpr(1)],
            block_shape=[BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED],
        )
        a_scale_shape: tl.constexpr = (BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED)
        b_scale_shape: tl.constexpr = (BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED)
    else:
        BLOCK_K_SCALE_LOAD: tl.constexpr = 16 if BLOCK_K_SCALE < 16 else BLOCK_K_SCALE
        a_scale_desc = tl.make_tensor_descriptor(
            a_scale + pid_m * BLOCK_M * stride_scale,
            shape=[M, K // SCALE_BLOCK],
            strides=[stride_scale, tl.constexpr(1)],
            block_shape=[BLOCK_M, BLOCK_K_SCALE_LOAD],
        )
        b_scale_desc = tl.make_tensor_descriptor(
            b_scale + pid_n * BLOCK_N * stride_scale,
            shape=[N, K // SCALE_BLOCK],
            strides=[stride_scale, tl.constexpr(1)],
            block_shape=[BLOCK_N, BLOCK_K_SCALE_LOAD],
        )
        a_scale_shape: tl.constexpr = (BLOCK_M, BLOCK_K_SCALE_LOAD)
        b_scale_shape: tl.constexpr = (BLOCK_N, BLOCK_K_SCALE_LOAD)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offsets = (stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]).to(tl.int32)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SCHEDULE == "sliceMNK":
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[M, N],
            strides=[stride_cm, tl.constexpr(1)],
            block_shape=[BLOCK_M, BLOCK_N],
        )
        c_off_m = pid_m * BLOCK_M
        c_off_n = pid_n * BLOCK_N

    a_scale_layout: tl.constexpr = _scale_shared_layout(a_scale_shape)
    b_scale_layout: tl.constexpr = _scale_shared_layout(b_scale_shape)
    if not TDM_SPLIT:
        a_layout: tl.constexpr = _operand_shared_layout([BLOCK_M, BLOCK_K_PACKED_A])
        a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K_PACKED_A), tlx.dtype_of(a_ptr), NUM_BUFFERS, layout=a_layout)
    a_scale_buf = tlx.local_alloc(a_scale_shape, tlx.dtype_of(a_scale), NUM_BUFFERS, layout=a_scale_layout)
    b_scale_buf = tlx.local_alloc(b_scale_shape, tlx.dtype_of(b_scale), NUM_BUFFERS, layout=b_scale_layout)

    K_ITERS = tl.cdiv(K, BLOCK_K)
    epilogue_lb = K_ITERS - (NUM_BUFFERS - 1)
    load_idx = 0
    wmma_idx = 0

    if SCHEDULE == "baseline" or SCHEDULE == "sliceK":
        _mxgemm_issue_l2_prefetches_prologue(a_desc, b_desc, a_scale_desc, b_scale_desc, load_idx, L2_PREFETCH_DISTANCE,
                                             NUM_BUFFERS, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B, BLOCK_K_SCALE_PRESHUFFLED,
                                             TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE)

    always = tl.full((), True, dtype=tl.int1)
    for _ in tl.static_range(NUM_BUFFERS - 1):
        if TDM_SPLIT:
            load_idx = _mxgemm_issue_split_loads(a0_desc, a1_desc, b0_desc, b1_desc, a_scale_desc, b_scale_desc, a0_buf,
                                                 a1_buf, b0_buf, b1_buf, a_scale_buf, b_scale_buf, load_idx, always,
                                                 BLOCK_K_PACKED_A, BLOCK_K_PACKED_B, BLOCK_K_SCALE_PRESHUFFLED,
                                                 NUM_BUFFERS, TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE, TDM_FUSION)
        else:
            load_idx = _mxgemm_issue_loads(
                a_desc,
                b_desc,
                a_scale_desc,
                b_scale_desc,
                a_buf,
                b_buf,
                a_scale_buf,
                b_scale_buf,
                load_idx,
                always,
                BLOCK_K_PACKED_A,
                BLOCK_K_PACKED_B,
                BLOCK_K_SCALE_PRESHUFFLED,
                NUM_BUFFERS,
                TRANSPOSE_B,
                SCALE_PRESHUFFLE,
                WITH_A_SCALE,
                TDM_FUSION,
            )

    tl.assume(K_ITERS > 0)
    if SCHEDULE == "sliceMNK":
        SUBTILE_M: tl.constexpr = BLOCK_M // 2
        SUBTILE_N: tl.constexpr = BLOCK_N // 2
        if TDM_SPLIT:
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 2) * SPLIT_LOADS_IN_BATCH)
            a00, scale_a00 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 0, 0, BLOCK_M, BLOCK_K,
                                                          DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED,
                                                          SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K,
                                                          SCALE_PRESHUFFLE, WITH_A_SCALE)
            b00, scale_b00 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 0, 0, BLOCK_N, BLOCK_K,
                                                          DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                          SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                          TRANSPOSE_B, SCALE_PRESHUFFLE)
            load_idx = _mxgemm_issue_split_loads(a0_desc, a1_desc, b0_desc, b1_desc, a_scale_desc, b_scale_desc, a0_buf,
                                                 a1_buf, b0_buf, b1_buf, a_scale_buf, b_scale_buf, load_idx, always,
                                                 BLOCK_K_PACKED_A, BLOCK_K_PACKED_B, BLOCK_K_SCALE_PRESHUFFLED,
                                                 NUM_BUFFERS, TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE, TDM_FUSION)
        else:
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 2) * NUM_LOADS_IN_BATCH)
            a00, scale_a00 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, 0, BLOCK_M, BLOCK_K, DIV_FACTOR_A,
                                                    SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED, SCALE_KWIDTH,
                                                    NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                    WITH_A_SCALE)
            b00, scale_b00 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 0, 0, BLOCK_N, BLOCK_K, DIV_FACTOR_B,
                                                    SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED, SCALE_KWIDTH,
                                                    NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                    SCALE_PRESHUFFLE)
            load_idx = _mxgemm_issue_loads(
                a_desc,
                b_desc,
                a_scale_desc,
                b_scale_desc,
                a_buf,
                b_buf,
                a_scale_buf,
                b_scale_buf,
                load_idx,
                always,
                BLOCK_K_PACKED_A,
                BLOCK_K_PACKED_B,
                BLOCK_K_SCALE_PRESHUFFLED,
                NUM_BUFFERS,
                TRANSPOSE_B,
                SCALE_PRESHUFFLE,
                WITH_A_SCALE,
                TDM_FUSION,
            )
        # Keep the first subtile's scales in the WMMA distribution across the
        # loop edge. Converting blocked loop-carried scales adds LDS exchanges
        # and barriers before the next iteration's first dot.
        PIN_LOOP_SCALES: tl.constexpr = (SCALE_PRESHUFFLE and SCALE_BLOCK == 32 and (BLOCK_K == 128 or BLOCK_K == 256)
                                         and STORE_INSTR_M == 16 and tlx.num_warps() == 4 and BLOCK_N == 256
                                         and (BLOCK_M == 128 or BLOCK_M == 256))
        if PIN_LOOP_SCALES:
            if BLOCK_M == 128:
                # A 64x128 subtile distributes all four waves along N.
                SCALE_A_LOOP_LAYOUT: tl.constexpr = tlx.layout(shape=((32, 4), (4, 2)), stride=((4, 0), (1, 128)))
                SCALE_B_LOOP_LAYOUT: tl.constexpr = tlx.layout(shape=((32, 4), (4, )), stride=((4, 128), (1, )))
            else:
                SCALE_A_LOOP_LAYOUT: tl.constexpr = tlx.layout(shape=((32, 2, 2), (4, 2)),
                                                               stride=((4, 0, 128), (1, 256)))
                SCALE_B_LOOP_LAYOUT: tl.constexpr = tlx.layout(shape=((32, 2, 2), (4, 2)),
                                                               stride=((4, 128, 0), (1, 256)))
            scale_a00 = tlx.require_layout(scale_a00, SCALE_A_LOOP_LAYOUT)
            scale_b00 = tlx.require_layout(scale_b00, SCALE_B_LOOP_LAYOUT)
        c00 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        c01 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        c10 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        c11 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        for i in tl.range(0, K_ITERS):
            if NUM_SUBTILES_K == 1:
                # Make the short stage's refill predicate available before its math.
                pred_load = i + 1 - epilogue_lb
                pred_load = (pred_load >> 31) & 1
            c00 = tlx.dot_scaled(a00, scale_a00, DTYPE_A, b00, scale_b00, DTYPE_B, c00, tiles_per_warp=TILES_PER_WARP)
            if TDM_SPLIT:
                b01, scale_b01 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 0, 1, BLOCK_N,
                                                              BLOCK_K, DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                              SCALE_PRESHUFFLE)
            else:
                b01, scale_b01 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 0, 1, BLOCK_N, BLOCK_K,
                                                        DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                        TRANSPOSE_B, SCALE_PRESHUFFLE)
            pred_prefetch = i - epilogue_lb
            pred_prefetch = (pred_prefetch >> 31) & 1
            if TDM_SPLIT:
                _mxgemm_issue_split_l2_prefetches(a0_desc, a1_desc, b0_desc, b1_desc, a_scale_desc, b_scale_desc,
                                                  load_idx, always, L2_PREFETCH_DISTANCE, BLOCK_K_PACKED_A,
                                                  BLOCK_K_PACKED_B, BLOCK_K_SCALE_PRESHUFFLED, TRANSPOSE_B,
                                                  SCALE_PRESHUFFLE, WITH_A_SCALE)
            else:
                _mxgemm_issue_l2_prefetches(a_desc, b_desc, a_scale_desc, b_scale_desc, load_idx, always,
                                            L2_PREFETCH_DISTANCE, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                            BLOCK_K_SCALE_PRESHUFFLED, TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE)
            c01 = tlx.dot_scaled(a00, scale_a00, DTYPE_A, b01, scale_b01, DTYPE_B, c01, tiles_per_warp=TILES_PER_WARP)
            if TDM_SPLIT:
                a10, scale_a10 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 1, 0, BLOCK_M,
                                                              BLOCK_K, DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                              WITH_A_SCALE)
            else:
                a10, scale_a10 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 1, 0, BLOCK_M, BLOCK_K,
                                                        DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K,
                                                        SCALE_PRESHUFFLE, WITH_A_SCALE)
            c10 = tlx.dot_scaled(a10, scale_a10, DTYPE_A, b00, scale_b00, DTYPE_B, c10, tiles_per_warp=TILES_PER_WARP)
            if NUM_SUBTILES_K == 2:
                if TDM_SPLIT:
                    b10, scale_b10 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 1, 0, BLOCK_N,
                                                                  BLOCK_K, DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                                  BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                                  NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                                  SCALE_PRESHUFFLE)
                else:
                    b10, scale_b10 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 1, 0, BLOCK_N, BLOCK_K,
                                                            DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                            BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                            NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                            SCALE_PRESHUFFLE)
            c11 = tlx.dot_scaled(a10, scale_a10, DTYPE_A, b01, scale_b01, DTYPE_B, c11, tiles_per_warp=TILES_PER_WARP)
            if NUM_SUBTILES_K == 2:
                if TDM_SPLIT:
                    a01, scale_a01 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 0, 1, BLOCK_M,
                                                                  BLOCK_K, DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                                  BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                                  NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                                  WITH_A_SCALE)
                else:
                    a01, scale_a01 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, 1, BLOCK_M, BLOCK_K,
                                                            DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                            BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                            NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                            WITH_A_SCALE)
                c00 = tlx.dot_scaled(a01, scale_a01, DTYPE_A, b10, scale_b10, DTYPE_B, c00,
                                     tiles_per_warp=TILES_PER_WARP)
                if TDM_SPLIT:
                    b11, scale_b11 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 1, 1, BLOCK_N,
                                                                  BLOCK_K, DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                                  BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                                  NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                                  SCALE_PRESHUFFLE)
                else:
                    b11, scale_b11 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 1, 1, BLOCK_N, BLOCK_K,
                                                            DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                            BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                            NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                            SCALE_PRESHUFFLE)
                c01 = tlx.dot_scaled(a01, scale_a01, DTYPE_A, b11, scale_b11, DTYPE_B, c01,
                                     tiles_per_warp=TILES_PER_WARP)
                if TDM_SPLIT:
                    a11, scale_a11 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 1, 1, BLOCK_M,
                                                                  BLOCK_K, DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                                  BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                                  NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                                  WITH_A_SCALE)
                else:
                    a11, scale_a11 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 1, 1, BLOCK_M, BLOCK_K,
                                                            DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                            BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                            NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                            WITH_A_SCALE)
            wmma_idx += 1
            if NUM_SUBTILES_K == 2:
                c10 = tlx.dot_scaled(a11, scale_a11, DTYPE_A, b10, scale_b10, DTYPE_B, c10,
                                     tiles_per_warp=TILES_PER_WARP)
                c11 = tlx.dot_scaled(a11, scale_a11, DTYPE_A, b11, scale_b11, DTYPE_B, c11,
                                     tiles_per_warp=TILES_PER_WARP)
                pred_load = i + 1 - epilogue_lb
                pred_load = (pred_load >> 31) & 1
            if TDM_SPLIT:
                load_idx = _mxgemm_issue_split_loads(a0_desc, a1_desc, b0_desc, b1_desc, a_scale_desc, b_scale_desc,
                                                     a0_buf, a1_buf, b0_buf, b1_buf, a_scale_buf, b_scale_buf, load_idx,
                                                     pred_load, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                                     BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS, TRANSPOSE_B,
                                                     SCALE_PRESHUFFLE, WITH_A_SCALE, TDM_FUSION)
                tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * SPLIT_LOADS_IN_BATCH)
                a00, scale_a00 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 0, 0, BLOCK_M,
                                                              BLOCK_K, DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                              WITH_A_SCALE)
                b00, scale_b00 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 0, 0, BLOCK_N,
                                                              BLOCK_K, DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                              SCALE_PRESHUFFLE)
            else:
                load_idx = _mxgemm_issue_loads(a_desc, b_desc, a_scale_desc, b_scale_desc, a_buf, b_buf, a_scale_buf,
                                               b_scale_buf, load_idx, pred_load, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                               BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS, TRANSPOSE_B, SCALE_PRESHUFFLE,
                                               WITH_A_SCALE, TDM_FUSION)
                tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * NUM_LOADS_IN_BATCH)
                a00, scale_a00 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, 0, BLOCK_M, BLOCK_K,
                                                        DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K,
                                                        SCALE_PRESHUFFLE, WITH_A_SCALE)
                b00, scale_b00 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 0, 0, BLOCK_N, BLOCK_K,
                                                        DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                        TRANSPOSE_B, SCALE_PRESHUFFLE)
            if PIN_LOOP_SCALES:
                scale_a00 = tlx.require_layout(scale_a00, SCALE_A_LOOP_LAYOUT)
                scale_b00 = tlx.require_layout(scale_b00, SCALE_B_LOOP_LAYOUT)
        # Stage each quadrant in its native WMMA register order. Joining them
        # puts the quadrant selectors first and defeats b128 LDS vectorization.
        c_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_ptr), 1)
        c_view = tlx.local_view(c_buf, 0)
        tlx.local_store(tlx.local_slice(c_view, [0, 0], [SUBTILE_M, SUBTILE_N]), c00.to(tlx.dtype_of(c_ptr)))
        tlx.local_store(tlx.local_slice(c_view, [0, SUBTILE_N], [SUBTILE_M, SUBTILE_N]), c01.to(tlx.dtype_of(c_ptr)))
        tlx.local_store(tlx.local_slice(c_view, [SUBTILE_M, 0], [SUBTILE_M, SUBTILE_N]), c10.to(tlx.dtype_of(c_ptr)))
        tlx.local_store(tlx.local_slice(c_view, [SUBTILE_M, SUBTILE_N], [SUBTILE_M, SUBTILE_N]),
                        c11.to(tlx.dtype_of(c_ptr)))
        tlx.async_amd_descriptor_store(c_desc, c_view, [c_off_m, c_off_n], clamp_bounds=True)
        tlx.async_amd_descriptor_wait(0)
    elif SCHEDULE == "sliceNK":
        SUBTILE_N: tl.constexpr = BLOCK_N // 2
        c0 = tl.zeros((BLOCK_M, SUBTILE_N), dtype=tl.float32)
        c1 = tl.zeros((BLOCK_M, SUBTILE_N), dtype=tl.float32)
        for i in tl.range(0, K_ITERS):
            pred = i - epilogue_lb
            pred = (pred >> 31) & 1
            load_idx = _mxgemm_issue_loads(a_desc, b_desc, a_scale_desc, b_scale_desc, a_buf, b_buf, a_scale_buf,
                                           b_scale_buf, load_idx, pred, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                           BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS, TRANSPOSE_B, SCALE_PRESHUFFLE,
                                           WITH_A_SCALE, TDM_FUSION)
            _mxgemm_issue_l2_prefetches(a_desc, b_desc, a_scale_desc, b_scale_desc, load_idx, always,
                                        L2_PREFETCH_DISTANCE, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                        BLOCK_K_SCALE_PRESHUFFLED, TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE)
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * NUM_LOADS_IN_BATCH)
            for kt in tl.static_range(2):
                a, scale_a = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, kt, BLOCK_M, BLOCK_K, DIV_FACTOR_A,
                                                    SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED, SCALE_KWIDTH,
                                                    NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                    WITH_A_SCALE)
                b0, scale_b0 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, kt, 0, BLOCK_N, BLOCK_K,
                                                      DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                      SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                      TRANSPOSE_B, SCALE_PRESHUFFLE)
                b1, scale_b1 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, kt, 1, BLOCK_N, BLOCK_K,
                                                      DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                      SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                      TRANSPOSE_B, SCALE_PRESHUFFLE)
                c0 = tlx.dot_scaled(a, scale_a, DTYPE_A, b0, scale_b0, DTYPE_B, c0, tiles_per_warp=TILES_PER_WARP)
                c1 = tlx.dot_scaled(a, scale_a, DTYPE_A, b1, scale_b1, DTYPE_B, c1, tiles_per_warp=TILES_PER_WARP)
            wmma_idx += 1
        acc = tl.join(c0, c1).permute(0, 2, 1).reshape((BLOCK_M, BLOCK_N))
        acc = tlx.require_amd_wmma_layout(acc, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                          instr_shape=STORE_INSTR_SHAPE)
        c_offsets_store = tlx.require_amd_wmma_layout(c_offsets, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                      instr_shape=STORE_INSTR_SHAPE)
        c_mask_store = tlx.require_amd_wmma_layout(c_mask, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                   instr_shape=STORE_INSTR_SHAPE)
        tlx.buffer_store(acc.to(tlx.dtype_of(c_ptr)), c_ptr, c_offsets_store, mask=c_mask_store)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in tl.range(0, K_ITERS):
            pred = i - epilogue_lb
            pred = (pred >> 31) & 1
            load_idx = _mxgemm_issue_loads(a_desc, b_desc, a_scale_desc, b_scale_desc, a_buf, b_buf, a_scale_buf,
                                           b_scale_buf, load_idx, pred, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                           BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS, TRANSPOSE_B, SCALE_PRESHUFFLE,
                                           WITH_A_SCALE, TDM_FUSION)
            _mxgemm_issue_l2_prefetches(a_desc, b_desc, a_scale_desc, b_scale_desc, load_idx, always,
                                        L2_PREFETCH_DISTANCE, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                        BLOCK_K_SCALE_PRESHUFFLED, TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE)
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * NUM_LOADS_IN_BATCH)
            for kt in tl.static_range(NUM_SUBTILES_K):
                a, scale_a = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, kt, BLOCK_M, BLOCK_K, DIV_FACTOR_A,
                                                    SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED, SCALE_KWIDTH,
                                                    NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                    WITH_A_SCALE)
                b, scale_b = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, kt, 0, BLOCK_N, BLOCK_K, DIV_FACTOR_B,
                                                    SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED, SCALE_KWIDTH,
                                                    NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                    SCALE_PRESHUFFLE)
                acc = tlx.dot_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, acc, tiles_per_warp=TILES_PER_WARP)
            wmma_idx += 1
        acc = tlx.require_amd_wmma_layout(acc, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                          instr_shape=STORE_INSTR_SHAPE)
        c_offsets_store = tlx.require_amd_wmma_layout(c_offsets, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                      instr_shape=STORE_INSTR_SHAPE)
        c_mask_store = tlx.require_amd_wmma_layout(c_mask, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                   instr_shape=STORE_INSTR_SHAPE)
        tlx.buffer_store(acc.to(tlx.dtype_of(c_ptr)), c_ptr, c_offsets_store, mask=c_mask_store)


DTYPE_TO_TRITON = {
    "float8_e5m2": "e5m2",
    "float8_e4m3": "e4m3",
    "float4": "e2m1",
}

# Grid cluster layouts "RxC" place R M tiles by C N tiles in one cluster. Rank r
# covers M tile r % R and N tile r // R; A is shared by the C ranks with equal
# r % R, B by the R consecutive ranks with equal r // R.
_MXGEMM_GRID_LAYOUTS = {"2x2": (2, 2), "4x2": (4, 2), "2x4": (2, 4), "4x4": (4, 4), "8x2": (8, 2), "2x8": (2, 8)}


def _mxgemm_persistent_programs(a, b, a_scale, b_scale, M, N, K, BM, BN, BK, num_buffers, group_m, transpose_b,
                                preshuffle, schedule, split, l2_prefetch, num_programs, scale_layout="block128"):
    if schedule != "sliceMNK" or not transpose_b or not preshuffle or split or l2_prefetch != -1:
        raise ValueError("persistent MXFP requires sliceMNK, transposed B, preshuffled scales, "
                         "unsplit descriptors, and L2 prefetch disabled")
    if group_m <= 0:
        raise ValueError("persistent MXFP requires a positive GROUP_SIZE_M")
    if min(M, N, K) <= 0 or M % BM or N % BN or K % BK or K // BK < num_buffers:
        raise ValueError("persistent MXFP requires nonempty full tiles and at least NUM_BUFFERS K tiles")
    sc_rows = 256 if scale_layout == "wave256" else 128
    if M != a.shape[0] or N != b.shape[0] or tuple(b_scale.shape) != (N // sc_rows, K // 32 * sc_rows):
        raise ValueError("persistent MXFP dimensions must match the data and preshuffled scales")
    if a_scale is None or tuple(a_scale.shape) != (M // sc_rows, K // 32 * sc_rows):
        raise ValueError("persistent MXFP dimensions must match the preshuffled A scales")
    if a.stride(1) != 1 or b.stride(1) != 1 or b_scale.stride(1) != 1:
        raise ValueError("persistent MXFP requires contiguous K dimensions")
    if a_scale.stride(1) != 1:
        raise ValueError("persistent MXFP requires preshuffled A scales with contiguous K")
    if num_programs is not None and (not isinstance(num_programs, int) or num_programs <= 0):
        raise ValueError("NUM_PROGRAMS must be a positive integer")
    if num_programs is None:
        num_programs = torch.cuda.get_device_properties(a.device).multi_processor_count
    return min(num_programs, (M // BM) * (N // BN))


def _mxfp_gemm_tflops(ms: float, M: int, N: int, K: int) -> float:
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def _validate_mxfp_scheduling(M, N, num_programs, *, persistent, block_m, block_n, group_m, xcd_remap, num_xcds,
                              xcd_chunk, cluster_size, cluster_barrier_interval=1, cluster_layout="linear",
                              cluster_multicast=True):
    if not isinstance(cluster_barrier_interval, int) or cluster_barrier_interval < 0:
        raise ValueError("CLUSTER_BARRIER_INTERVAL must be a nonnegative integer")
    if xcd_remap not in _XCD_REMAP_MODES:
        raise ValueError("XCD_REMAP must be none, balanced, or chunked")
    if num_xcds <= 0 or xcd_chunk <= 0:
        raise ValueError("NUM_XCDS and XCD_CHUNK must be positive")
    if cluster_size not in (1, 2, 4, 8, 16):
        raise ValueError("CLUSTER_SIZE must be 1, 2, 4, 8, or 16")
    if cluster_layout != "linear" and cluster_layout not in _MXGEMM_GRID_LAYOUTS:
        raise ValueError(f"CLUSTER_LAYOUT must be linear or one of {', '.join(_MXGEMM_GRID_LAYOUTS)}")
    grid_rows, grid_cols = _MXGEMM_GRID_LAYOUTS.get(cluster_layout, (1, 1))
    if cluster_layout != "linear" and grid_rows * grid_cols != cluster_size:
        raise ValueError(f"CLUSTER_LAYOUT={cluster_layout} requires CLUSTER_SIZE={grid_rows * grid_cols}")
    if not persistent and (xcd_remap != "none" or cluster_size > 1):
        raise ValueError("XCD remapping and clustering require persistence")
    if cluster_size == 1:
        return
    if (xcd_remap != "none" and cluster_layout in ("linear", "2x2")
            and (xcd_remap, num_xcds, xcd_chunk) != ("chunked", 8, 2)):
        raise ValueError("linear and 2x2 clusters require XCD_REMAP=none or chunked with NUM_XCDS=8 and XCD_CHUNK=2")
    if group_m not in (4, 8) or M % (group_m * block_m) or N % (max(2, grid_cols) * block_n):
        raise ValueError("clustering requires GROUP_SIZE_M=4 or 8, full M groups, and an even number of N tiles "
                         "(a multiple of the cluster columns for grid layouts)")
    if group_m % grid_rows:
        raise ValueError(f"CLUSTER_LAYOUT={cluster_layout} requires GROUP_SIZE_M divisible by {grid_rows}")
    # A linear cluster multicasts B to all its CTAs, so they must share one N
    # tile: the cluster may not span M groups. A TDM multicast also reaches at
    # most five CTAs.
    if cluster_layout == "linear" and cluster_multicast and cluster_size > min(group_m, 4):
        raise ValueError("CLUSTER_LAYOUT=linear with multicast requires CLUSTER_SIZE <= min(GROUP_SIZE_M, 4); "
                         "use a grid CLUSTER_LAYOUT for larger clusters")
    # Every cluster member must execute the same number of persistent tiles,
    # including taking the same has_next branches around cluster barriers.
    tiles = (M // block_m) * (N // block_n)
    if not isinstance(num_programs, int) or num_programs <= 0 or num_programs % 16 or tiles % num_programs:
        raise ValueError("clustering requires NUM_PROGRAMS divisible by 16 and dividing the output tile count; "
                         "set NUM_PROGRAMS explicitly if necessary")


def _init_data(dtype: str, rows: int, cols: int):
    if dtype == "float4":
        return MXFP4Tensor(size=(rows, cols)).random()
    if dtype == "float8_e5m2":
        return torch.randint(20, 40, (rows, cols), dtype=torch.uint8).view(torch.float8_e5m2)
    if dtype == "float8_e4m3":
        return torch.randint(20, 40, (rows, cols), dtype=torch.uint8).view(torch.float8_e4m3fn)
    raise ValueError(f"unsupported dtype: {dtype}")


def _mxfp_timed_batches(batch: Callable[[], float], warmup_seconds: float, duration_seconds: float):
    """Run timing batches for a warmup period, then return the median batch over the measurement period.

    Power-capped devices reach steady clocks only after seconds of load, so a
    single batch from a cold process overstates the kernel time.
    """
    deadline = time.monotonic() + warmup_seconds
    while time.monotonic() < deadline:
        batch()
    samples = []
    deadline = time.monotonic() + duration_seconds
    while not samples or time.monotonic() < deadline:
        samples.append(batch())
    return statistics.median(samples), len(samples)


def _validate_tiles_per_warp(tiles_per_warp):
    tiles_per_warp = tuple(tiles_per_warp)
    if len(tiles_per_warp) != 2:
        raise ValueError("TILES_PER_WARP requires two entries")
    if any(not isinstance(tile, int) or isinstance(tile, bool) or tile <= 0 for tile in tiles_per_warp):
        raise ValueError("TILES_PER_WARP entries must be positive integers")
    if any(tile & (tile - 1) for tile in tiles_per_warp):
        raise ValueError("TILES_PER_WARP entries must be powers of two")
    return tiles_per_warp


_OUT_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}

# Compile options for every mxgemm_tdm_pipelined_kernel launch. gfx1250's
# va_vdst anti-hints raise VGPR pressure; with BK256 FP8 B tiles they push the
# kernel past the register file and into scratch spills.
# WORKAROUND: LLVM's MFMA anti-hints (llvm/llvm-project#218075) put scratch accesses in the steady K loop.
PIPELINED_COMPILE_OPTIONS = {"va_vdst_anti_hints": False, "llvm_cl_opts": (("amdgpu-anti-hints", "false"), )}


def mxgemm_tdm_pipelined(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        BLOCK_M: int = 128,
        BLOCK_N: int = 128,
        BLOCK_K: int = 128,
        TRANSPOSE_B: bool = False,
        NUM_BUFFERS: int = 2,
        DTYPE_A: str = "e5m2",
        DTYPE_B: str = "e5m2",
        SCALE_PRESHUFFLE: bool = True,
        WITH_A_SCALE: bool = True,
        SCHEDULE: str = "baseline",
        L2_PREFETCH_DISTANCE: int = -1,
        M: int | None = None,
        N: int | None = None,
        K: int | None = None,
        TDM_FUSION: str = "none",
        TDM_SPLIT: bool = False,
        GROUP_SIZE_M: int = 8,
        BENCHMARK: str | Callable | None = None,
        BENCHMARK_NUM_ITERS: int = 32,
        BENCHMARK_WARMUP_SECONDS: float = 0.0,
        BENCHMARK_DURATION_SECONDS: float = 0.0,
        PERSISTENT: bool = False,
        NUM_PROGRAMS: int | None = None,
        CROSS_TILE_PREFETCH: bool = True,
        XCD_REMAP: str = "none",
        NUM_XCDS: int = 8,
        XCD_CHUNK: int = 2,
        CLUSTER_SIZE: int = 1,
        CLUSTER_MULTICAST: bool = True,
        CLUSTER_BARRIER_INTERVAL: int = 1,
        CLUSTER_LAYOUT: str = "linear",
        SPLIT_CLUSTER_BARRIER: bool = False,
        EARLY_C_STORES: bool = False,
        STATIC_SLOT_UNROLL: bool = False,
        OUTPUT_C_SLOTS: int | None = None,
        OVERLAP_PEEL: bool = False,
        MERGED_HANDOFF_BARRIER: bool = False,
        STATIC_CLUSTER_STEPS: bool = False,
        SCALE_LAYOUT: str = "block128",
        OUT_DTYPE: str = "fp32",
        TILES_PER_WARP: tuple[int, int] = (2, 2),
) -> torch.Tensor:
    """Run MXFP GEMM, optionally with the persistent sliceMNK kernel.

    ``OUT_DTYPE`` selects the C element type, ``"fp32"`` (default) or
    ``"fp16"``. Accumulation stays FP32; FP16 output converts in registers.

    ``BENCHMARK`` selects ``"eager"`` or ``"graph"`` timing, or accepts a
    callback receiving the allocation-free launch function.

    ``TILES_PER_WARP`` sets the non-persistent kernel's WMMA tiles per warp.

    ``PERSISTENT`` selects ``mxgemm_tdm_persistent_kernel``. It requires FP8
    E4M3 A with A scales, transposed B, preshuffled scales, sliceMNK, partial
    TDM fusion, unsplit descriptors, L2 prefetch disabled, 256x256 M/N
    tiles, and 2x2 WMMA tiles per warp. FP8 B uses BK128 with two to four input buffers; FP4 B uses BK256
    with three buffers (BK256 FP8 B uses two). M/N/K must contain full tiles,
    with at least ``NUM_BUFFERS`` K tiles. ``NUM_PROGRAMS`` defaults to the
    smaller of the CU count and the output tile count. C is staged through
    LDS and written with TDM stores: BK128 uses dedicated slots
    (``OUTPUT_C_SLOTS``, default two), and BK256 reuses retired A stages.

    ``CROSS_TILE_PREFETCH`` loads the next tile's inputs during the current
    tile's final K steps (BK128), or refills each A/B stage as its C slot
    retires (three-buffer FP4 BK256). With ``False``, each tile is primed
    after the previous tile's output stores.

    ``XCD_REMAP`` selects none, balanced, or chunked persistent program
    ordering. ``CLUSTER_SIZE`` groups workgroups whose matching TDM loads are
    multicast (``CLUSTER_MULTICAST``). ``CLUSTER_LAYOUT="linear"`` shares B
    within a cluster of consecutive M tiles; with multicast it allows at most
    ``min(GROUP_SIZE_M, 4)`` workgroups. ``"2x2"``, ``"4x2"``, ``"2x4"``, ``"4x4"``, ``"8x2"``,
    and ``"2x8"`` map an R x C cluster onto R M tiles by C N tiles, so each A
    tile is shared by C workgroups and each B tile by R. A cluster barrier
    every ``CLUSTER_BARRIER_INTERVAL`` K blocks (0: none) keeps the members'
    requests close enough to merge; each workgroup's own waits protect its LDS.
    Clustering requires GROUP_SIZE_M 4 or 8, full M groups, and a program
    count divisible by 16 that divides the output tile count.
    ``SPLIT_CLUSTER_BARRIER`` signals the periodic rendezvous before
    independent matrix work and waits just before the refill.
    ``EARLY_C_STORES`` interleaves the final quadrant dots with C stores.
    Both require three buffers with FP8 BK128 or FP4 BK256 tiles.

    ``STATIC_SLOT_UNROLL`` unrolls the three- or four-buffer FP8 BK128
    steady loop by its ring depth, so every step reads a loop-invariant LDS
    slot; A and B are then split over two LDS partitions. ``OVERLAP_PEEL``
    runs each next tile's first K step inside the current tile's C epilogue.
    It requires BK128, cross-tile prefetch, and more K tiles than buffers.

    ``MERGED_HANDOFF_BARRIER`` gives the four-buffer static loop one
    workgroup barrier per K step, split around the C01 WMMAs
    (``tlx.named_barrier_arrive`` / ``named_barrier_wait``): each step reads
    its slot's second halves during C00, signals, runs C01, waits, then
    refills the slot and reads the next one across C10 and C11.
    ``STATIC_CLUSTER_STEPS`` tests the periodic cluster rendezvous only in
    the unrolled steps that can reach it. Both require ``STATIC_SLOT_UNROLL``.

    ``SCALE_LAYOUT`` names the preshuffled scale format. ``"block128"``
    (default) stores one row per 128 M/N rows. ``"wave256"`` stores one row
    per 256 rows, ordered so each lane's 16 bytes hold the scales its wave
    needs for both 128-row quadrants: the merged step reads each operand's
    scales with one ``ds_load_b128``, and other FP8 BK128 schedules read one
    quadrant per ``ds_load_b64``. Pack scales with
    ``pack_scale(x, layout=...)`` using the same layout.
    """
    TILES_PER_WARP = _validate_tiles_per_warp(TILES_PER_WARP)
    if M is None:
        M = a.shape[0]
    if K is None:
        K = a.shape[1] * (2 if DTYPE_A == "e2m1" else 1)
    if N is None:
        if TRANSPOSE_B:
            N = b.shape[0]
        else:
            N = b.shape[1]
    if TRANSPOSE_B:
        Kb = b.shape[1] * (2 if DTYPE_B == "e2m1" else 1)
    else:
        Kb = b.shape[0] * (2 if DTYPE_B == "e2m1" else 1)
    assert K == Kb
    if OUT_DTYPE not in _OUT_DTYPES:
        raise ValueError(f"OUT_DTYPE must be one of {tuple(_OUT_DTYPES)}, got {OUT_DTYPE!r}")
    if SCALE_LAYOUT not in ("block128", "wave256"):
        raise ValueError("SCALE_LAYOUT must be 'block128' or 'wave256'")
    if not PERSISTENT and (CLUSTER_LAYOUT != "linear" or SPLIT_CLUSTER_BARRIER or EARLY_C_STORES or STATIC_SLOT_UNROLL
                           or OUTPUT_C_SLOTS is not None or OVERLAP_PEEL or MERGED_HANDOFF_BARRIER
                           or STATIC_CLUSTER_STEPS or SCALE_LAYOUT != "block128"):
        raise ValueError("CLUSTER_LAYOUT, SPLIT_CLUSTER_BARRIER, EARLY_C_STORES, STATIC_SLOT_UNROLL, "
                         "OUTPUT_C_SLOTS, OVERLAP_PEEL, MERGED_HANDOFF_BARRIER, STATIC_CLUSTER_STEPS, and "
                         "SCALE_LAYOUT require PERSISTENT")
    if PERSISTENT:
        if TILES_PER_WARP != (2, 2):
            raise ValueError("persistent MXFP requires TILES_PER_WARP=(2, 2)")
        if DTYPE_A != "e4m3" or not WITH_A_SCALE or TDM_FUSION != "partial" or BLOCK_M != 256 or BLOCK_N != 256:
            raise ValueError("persistent MXFP requires E4M3 A with A scales, partial TDM fusion, and 256x256 M/N tiles")
        if BLOCK_K == 128:
            if NUM_BUFFERS not in (2, 3, 4):
                raise ValueError("persistent BK128 MXFP requires 2/3/4 buffers")
        elif BLOCK_K == 256:
            if NUM_BUFFERS != 2 and not (NUM_BUFFERS == 3 and DTYPE_B == "e2m1"):
                raise ValueError("persistent BK256 MXFP requires 2 buffers, or 3 buffers with FP4 B")
            if CROSS_TILE_PREFETCH and not (NUM_BUFFERS == 3 and DTYPE_B == "e2m1"):
                raise ValueError("BK256 cross-tile prefetch requires three buffers and FP4 B; "
                                 "use CROSS_TILE_PREFETCH=False otherwise")
        else:
            raise ValueError("persistent MXFP requires BK128 or BK256")
        if (SPLIT_CLUSTER_BARRIER or EARLY_C_STORES) and not (NUM_BUFFERS == 3 and
                                                              ((BLOCK_K == 128 and DTYPE_B == "e4m3") or
                                                               (BLOCK_K == 256 and DTYPE_B == "e2m1"))):
            raise ValueError("SPLIT_CLUSTER_BARRIER and EARLY_C_STORES require three buffers with E4M3 BK128 or "
                             "FP4 BK256 B")
        if SPLIT_CLUSTER_BARRIER and (CLUSTER_SIZE == 1 or CLUSTER_BARRIER_INTERVAL <= 0):
            raise ValueError("SPLIT_CLUSTER_BARRIER requires clustering and a positive barrier interval")
        if STATIC_SLOT_UNROLL and not (NUM_BUFFERS in (3, 4) and BLOCK_K == 128 and DTYPE_B == "e4m3"):
            raise ValueError("STATIC_SLOT_UNROLL requires E4M3 B, BK128, and three or four buffers")
        if MERGED_HANDOFF_BARRIER and not (STATIC_SLOT_UNROLL and NUM_BUFFERS == 4):
            raise ValueError("MERGED_HANDOFF_BARRIER requires STATIC_SLOT_UNROLL with four buffers")
        if STATIC_CLUSTER_STEPS and not STATIC_SLOT_UNROLL:
            raise ValueError("STATIC_CLUSTER_STEPS requires STATIC_SLOT_UNROLL")
        if SCALE_LAYOUT == "wave256" and not (BLOCK_K == 128 and DTYPE_B == "e4m3"):
            raise ValueError("SCALE_LAYOUT='wave256' requires E4M3 B with BK128")
        if OVERLAP_PEEL and not (CROSS_TILE_PREFETCH and BLOCK_K == 128 and K // BLOCK_K > NUM_BUFFERS
                                 and not (SPLIT_CLUSTER_BARRIER or EARLY_C_STORES)):
            raise ValueError("OVERLAP_PEEL requires BK128, cross-tile prefetch, more K tiles than buffers, and no "
                             "SPLIT_CLUSTER_BARRIER or EARLY_C_STORES")
        if OUTPUT_C_SLOTS is not None and not (isinstance(OUTPUT_C_SLOTS, int) and OUTPUT_C_SLOTS >= 2
                                               and BLOCK_K == 128):
            raise ValueError("OUTPUT_C_SLOTS requires BK128 and at least two slots")
        NUM_PROGRAMS = _mxgemm_persistent_programs(a, b, a_scale, b_scale, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                                   NUM_BUFFERS, GROUP_SIZE_M, TRANSPOSE_B, SCALE_PRESHUFFLE, SCHEDULE,
                                                   TDM_SPLIT, L2_PREFETCH_DISTANCE, NUM_PROGRAMS, SCALE_LAYOUT)
    _validate_mxfp_scheduling(M, N, NUM_PROGRAMS, persistent=PERSISTENT, block_m=BLOCK_M, block_n=BLOCK_N,
                              group_m=GROUP_SIZE_M, xcd_remap=XCD_REMAP, num_xcds=NUM_XCDS, xcd_chunk=XCD_CHUNK,
                              cluster_size=CLUSTER_SIZE, cluster_barrier_interval=CLUSTER_BARRIER_INTERVAL,
                              cluster_layout=CLUSTER_LAYOUT, cluster_multicast=CLUSTER_MULTICAST)
    c = torch.empty((M, N), device=a.device, dtype=_OUT_DTYPES[OUT_DTYPE])
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    a_scale_arg = a_scale if WITH_A_SCALE else b_scale
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    def run_kernel():
        if PERSISTENT:
            return mxgemm_tdm_persistent_kernel[(NUM_PROGRAMS, )](
                a, b, c, a_scale_arg, b_scale, M, N, K, a.stride(0), b.stride(0), c.stride(0), a_scale_arg.stride(0),
                b_scale.stride(0), DTYPE_B=DTYPE_B, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                GROUP_SIZE_M=GROUP_SIZE_M, NUM_BUFFERS=NUM_BUFFERS, NUM_PROGRAMS=NUM_PROGRAMS,
                CROSS_TILE_PREFETCH=CROSS_TILE_PREFETCH, XCD_REMAP_MODE=_XCD_REMAP_MODES[XCD_REMAP], NUM_XCDS=NUM_XCDS,
                XCD_CHUNK=XCD_CHUNK, CLUSTER_SIZE=CLUSTER_SIZE, CLUSTER_MULTICAST=CLUSTER_MULTICAST,
                CLUSTER_BARRIER_INTERVAL=CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT=CLUSTER_LAYOUT,
                SPLIT_CLUSTER_BARRIER=SPLIT_CLUSTER_BARRIER, EARLY_C_STORES=EARLY_C_STORES,
                STATIC_SLOT_UNROLL=STATIC_SLOT_UNROLL, OUTPUT_C_SLOTS=OUTPUT_C_SLOTS, OVERLAP_PEEL=OVERLAP_PEEL,
                MERGED_HANDOFF_BARRIER=MERGED_HANDOFF_BARRIER, STATIC_CLUSTER_STEPS=STATIC_CLUSTER_STEPS,
                SCALE_WAVE256=SCALE_LAYOUT == "wave256", num_warps=4, waves_per_eu=1, ctas_per_cga=(CLUSTER_SIZE, 1, 1))
        return mxgemm_tdm_pipelined_kernel[grid](
            a,
            b,
            c,
            a_scale_arg,
            b_scale,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            stride_bk,
            stride_bn,
            c.stride(0),
            c.stride(1),
            b_scale.stride(0),
            DTYPE_A=DTYPE_A,
            DTYPE_B=DTYPE_B,
            SCALE_BLOCK=32,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            TRANSPOSE_B=TRANSPOSE_B,
            NUM_BUFFERS=NUM_BUFFERS,
            SCALE_PRESHUFFLE=SCALE_PRESHUFFLE,
            WITH_A_SCALE=WITH_A_SCALE,
            SCHEDULE=SCHEDULE,
            TDM_FUSION=TDM_FUSION,
            L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE,
            TDM_SPLIT=TDM_SPLIT,
            TILES_PER_WARP=TILES_PER_WARP,
            num_warps=4,
            waves_per_eu=1,
            **PIPELINED_COMPILE_OPTIONS,
        )

    if callable(BENCHMARK):
        BENCHMARK(run_kernel)
    elif BENCHMARK in ("graph", "eager"):
        if BENCHMARK == "graph":
            batch = lambda: triton.testing.do_bench_cudagraph(run_kernel, rep=BENCHMARK_NUM_ITERS)  # noqa: E731
        else:
            batch = lambda: triton.testing.do_bench(run_kernel, warmup=30, rep=BENCHMARK_NUM_ITERS)  # noqa: E731
        ms, batches = _mxfp_timed_batches(batch, BENCHMARK_WARMUP_SECONDS, BENCHMARK_DURATION_SECONDS)
        detail = f" (median of {batches} batches after {BENCHMARK_WARMUP_SECONDS:g} s warmup)" if batches > 1 else ""
        print(f"execution time: {ms} ms, {_mxfp_gemm_tflops(ms, M, N, K):.2f} TFLOPS{detail}")
    else:
        run_kernel()
    return c


_DEFAULT_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 128,
    "GROUP_SIZE_M": 8,
    "NUM_BUFFERS": 2,
    "DTYPE_A": "e5m2",
    "DTYPE_B": "e5m2",
    "SCALE_BLOCK": 32,
    "TRANSPOSE_B": False,
    "num_warps": 4,
    "waves_per_eu": 1,
    "TILES_PER_WARP": (2, 2),
}

_PERSISTENT_CONFIG_KEYS = ("PERSISTENT", "NUM_PROGRAMS", "CROSS_TILE_PREFETCH", "XCD_REMAP", "NUM_XCDS", "XCD_CHUNK",
                           "CLUSTER_SIZE", "CLUSTER_MULTICAST", "CLUSTER_BARRIER_INTERVAL", "CLUSTER_LAYOUT",
                           "SPLIT_CLUSTER_BARRIER", "EARLY_C_STORES", "STATIC_SLOT_UNROLL", "OUTPUT_C_SLOTS",
                           "OVERLAP_PEEL", "MERGED_HANDOFF_BARRIER", "STATIC_CLUSTER_STEPS", "SCALE_LAYOUT")


def matmul(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor, config=None) -> torch.Tensor:
    """C = (A * a_scale) @ (B * b_scale) using a TDM-pipelined MXFP kernel on AMD gfx1250.

    ``a_scale`` / ``b_scale`` must already be pre-shuffled with :func:`pack_scale`, using the
    ``layout`` that matches ``config["SCALE_LAYOUT"]``.
    When ``config["TRANSPOSE_B"]`` is set, ``b`` is the ``[N, K]`` transposed layout.
    ``PERSISTENT`` selects the persistent kernel; it and the other persistent
    options (``NUM_PROGRAMS``, ``CROSS_TILE_PREFETCH``, ``XCD_REMAP``, cluster
    options, ``STATIC_SLOT_UNROLL``, ``OUTPUT_C_SLOTS``, ``OVERLAP_PEEL``,
    ``MERGED_HANDOFF_BARRIER``, ``STATIC_CLUSTER_STEPS``, ``SCALE_LAYOUT``) are
    described in :func:`mxgemm_tdm_pipelined`. ``OUT_DTYPE`` selects FP32 or
    FP16 output.
    """
    cfg = dict(_DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)
    TRANSPOSE_B = cfg["TRANSPOSE_B"]

    M, packed_k = a.shape
    K = packed_k * (2 if cfg["DTYPE_A"] == "e2m1" else 1)
    if TRANSPOSE_B:
        N, Kb = b.shape
    else:
        Kb, N = b.shape
    Kb *= 2 if cfg["DTYPE_B"] == "e2m1" else 1
    assert K == Kb, f"K mismatch: A={a.shape}, B={b.shape}"

    if cfg.get("PERSISTENT", False):
        if cfg["num_warps"] != 4 or cfg["waves_per_eu"] != 1 or cfg["SCALE_BLOCK"] != 32:
            raise ValueError("persistent MXFP requires four warps, one wave per SIMD, and SCALE_BLOCK=32")
    persistent_only = ("CLUSTER_LAYOUT", "SPLIT_CLUSTER_BARRIER", "EARLY_C_STORES", "STATIC_SLOT_UNROLL",
                       "OUTPUT_C_SLOTS", "OVERLAP_PEEL", "MERGED_HANDOFF_BARRIER", "STATIC_CLUSTER_STEPS",
                       "SCALE_LAYOUT")
    if (cfg.get("PERSISTENT", False) or cfg.get("OUT_DTYPE", "fp32") != "fp32"
            or any(key in cfg for key in persistent_only)):
        return mxgemm_tdm_pipelined(a, b, a_scale, b_scale, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"],
                                    BLOCK_K=cfg["BLOCK_K"], TRANSPOSE_B=TRANSPOSE_B, NUM_BUFFERS=cfg["NUM_BUFFERS"],
                                    DTYPE_A=cfg["DTYPE_A"], DTYPE_B=cfg["DTYPE_B"], SCALE_PRESHUFFLE=cfg.get(
                                        "SCALE_PRESHUFFLE",
                                        True), WITH_A_SCALE=cfg.get("WITH_A_SCALE",
                                                                    True), SCHEDULE=cfg.get("SCHEDULE", "baseline"),
                                    L2_PREFETCH_DISTANCE=cfg.get("L2_PREFETCH_DISTANCE",
                                                                 -1), TDM_FUSION=cfg.get("TDM_FUSION", "none"),
                                    TDM_SPLIT=cfg.get("TDM_SPLIT", False), GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
                                    OUT_DTYPE=cfg.get("OUT_DTYPE",
                                                      "fp32"), TILES_PER_WARP=cfg.get("TILES_PER_WARP", (2, 2)),
                                    **{key: cfg[key]
                                       for key in _PERSISTENT_CONFIG_KEYS
                                       if key in cfg})

    _validate_mxfp_scheduling(M, N, cfg.get("NUM_PROGRAMS"), persistent=False, block_m=cfg["BLOCK_M"],
                              block_n=cfg["BLOCK_N"], group_m=cfg["GROUP_SIZE_M"],
                              xcd_remap=cfg.get("XCD_REMAP",
                                                "none"), num_xcds=cfg.get("NUM_XCDS",
                                                                          8), xcd_chunk=cfg.get("XCD_CHUNK", 2),
                              cluster_size=cfg.get("CLUSTER_SIZE",
                                                   1), cluster_barrier_interval=cfg.get("CLUSTER_BARRIER_INTERVAL", 1),
                              cluster_layout=cfg.get("CLUSTER_LAYOUT",
                                                     "linear"), cluster_multicast=cfg.get("CLUSTER_MULTICAST", True))
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]), )
    mxgemm_tdm_pipelined_kernel[grid](
        a,
        b,
        c,
        a_scale if cfg.get("WITH_A_SCALE", True) else b_scale,
        b_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        stride_bk,
        stride_bn,
        c.stride(0),
        c.stride(1),
        b_scale.stride(0),
        DTYPE_A=cfg["DTYPE_A"],
        DTYPE_B=cfg["DTYPE_B"],
        SCALE_BLOCK=cfg["SCALE_BLOCK"],
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        TRANSPOSE_B=TRANSPOSE_B,
        NUM_BUFFERS=cfg["NUM_BUFFERS"],
        SCALE_PRESHUFFLE=cfg.get("SCALE_PRESHUFFLE", True),
        WITH_A_SCALE=cfg.get("WITH_A_SCALE", True),
        SCHEDULE=cfg.get("SCHEDULE", "baseline"),
        TDM_FUSION=cfg.get("TDM_FUSION", "none"),
        TDM_SPLIT=cfg.get("TDM_SPLIT", False),
        L2_PREFETCH_DISTANCE=cfg.get("L2_PREFETCH_DISTANCE", -1),
        TILES_PER_WARP=_validate_tiles_per_warp(cfg.get("TILES_PER_WARP", (2, 2))),
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
        **PIPELINED_COMPILE_OPTIONS,
    )
    return c


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the TLX AMD MXFP TDM GEMM tutorial kernel")
    parser.add_argument("-M", type=int, default=2048, help="problem M size")
    parser.add_argument("-N", type=int, default=1024, help="problem N size")
    parser.add_argument("-K", type=int, default=8192, help="problem K size")
    parser.add_argument("-BM", type=int, default=256, help="BLOCK_M")
    parser.add_argument("-BN", type=int, default=256, help="BLOCK_N")
    parser.add_argument("-BK", type=int, default=256, help="BLOCK_K")
    parser.add_argument("--num_buffers", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--group_size_m", type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument("--dtype_a", type=str, default="float8_e4m3", choices=tuple(DTYPE_TO_TRITON))
    parser.add_argument("--dtype_b", type=str, default="float4", choices=tuple(DTYPE_TO_TRITON))
    parser.add_argument("--scale_preshuffled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with_a_scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transpose_b", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--schedule", type=str, default="sliceMNK",
                        choices=["baseline", "sliceK", "sliceNK", "sliceMNK"])
    parser.add_argument("--tdm_fusion", type=str, default="partial", choices=["none", "2way", "4way", "partial"])
    parser.add_argument("--partial_tdm", action="store_true",
                        help="Alias for --tdm_fusion partial, matching the Gluon CLI spelling")
    parser.add_argument("--tdm_split", action="store_true")
    parser.add_argument("--persistent", action="store_true", help="use the persistent sliceMNK kernel")
    parser.add_argument("--output_c_slots", type=int, default=None,
                        help="dedicated BK128 output staging slots (default: 2)")
    parser.add_argument(
        "--static_slot_unroll", action="store_true",
        help="unroll the three- or four-buffer BK128 steady loop by its ring depth with "
        "loop-invariant slots")
    parser.add_argument("--merged_handoff_barrier", action="store_true",
                        help="four-buffer static loop: one workgroup barrier per K step, split around the C01 WMMAs")
    parser.add_argument("--static_cluster_steps", action="store_true",
                        help="static loop: test the cluster rendezvous only in unrolled steps that can hit it")
    parser.add_argument(
        "--scale_layout", choices=("block128", "wave256"), default="block128",
        help="preshuffled scale format; wave256 reads both quadrants' scales with one ds_load_b128 per wave "
        "(persistent A8W8 BK128)")
    parser.add_argument("--overlap_peel", action="store_true",
                        help="run each next tile's first K step inside the BK128 output epilogue")
    parser.add_argument("--out_dtype", choices=tuple(_OUT_DTYPES), default="fp32",
                        help="C element type; FP16 converts the FP32 accumulators before storing")
    parser.add_argument("--num_programs", type=int, default=None, help="persistent workgroup count (default: CU count)")
    parser.add_argument("--xcd_remap", choices=tuple(_XCD_REMAP_MODES), default="none")
    parser.add_argument("--num_xcds", type=int, default=8)
    parser.add_argument("--xcd_chunk", type=int, default=2)
    parser.add_argument("--cluster_size", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--cluster_layout", choices=("linear", ) + tuple(_MXGEMM_GRID_LAYOUTS), default="linear",
                        help="map clusters to R M tiles by C N tiles with RxC (2x2 for four CTAs)")
    parser.add_argument("--split_cluster_barrier", action="store_true",
                        help="overlap periodic cluster rendezvous with independent steady-loop matrix work")
    parser.add_argument("--early_c_stores", action="store_true",
                        help="interleave final quadrant dots and output stores")
    parser.add_argument("--cluster_barrier_interval", type=int, default=1,
                        help="align cluster requests every N input K blocks; 0 disables all cluster barriers")
    parser.add_argument("--cluster_multicast", action=argparse.BooleanOptionalAction, default=True,
                        help="share data and scales within a cluster; disable to keep only cluster synchronization")
    parser.add_argument(
        "--cross_tile_prefetch", action=argparse.BooleanOptionalAction, default=True,
        help="prefetch next-tile inputs during the K tail, or overlap C retirement with BK256 FP4 refills")
    parser.add_argument("--l2_prefetch_distance", type=int, default=-1,
                        help="Prefetch distance in K iterations; -1 disables L2 prefetch")
    parser.add_argument("--benchmark_mode", choices=["eager", "graph", "none"], default="eager")
    parser.add_argument("--benchmark_warmup_seconds", type=float, default=0.0,
                        help="repeat timing batches for this long before measuring")
    parser.add_argument("--benchmark_duration_seconds", type=float, default=0.0,
                        help="report the median timing batch over this period (0: one batch)")
    parser.add_argument("--benchmark_num_iters", type=int, default=32,
                        help="timing repetition budget in milliseconds (not an iteration count)")
    parser.add_argument("--tiles_per_warp", nargs=2, type=int, choices=(1, 2), default=(2, 2), metavar=("M", "N"),
                        help="WMMA tiles per warp in M and N (default: 2 2)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    a = _init_data(args.dtype_a, args.M, args.K)
    b = _init_data(args.dtype_b, args.K, args.N)
    if args.with_a_scale:
        a_scale = MXScaleTensor(size=(args.M, triton.cdiv(args.K, 32))).random(high=32.0).data
    else:
        a_scale = None
    b_scale = MXScaleTensor(size=(args.N, triton.cdiv(args.K, 32))).random(high=32.0).data

    a_scale_input = (pack_scale(a_scale, layout=args.scale_layout)
                     if args.scale_preshuffled and a_scale is not None else a_scale)
    b_scale_input = pack_scale(b_scale, layout=args.scale_layout) if args.scale_preshuffled else b_scale
    if args.dtype_a == "float4":
        a = a.to_packed_tensor(dim=1)
    if args.dtype_b == "float4":
        b = b.to_packed_tensor(dim=0)

    a_d = a.data.contiguous().cuda() if args.dtype_a == "float4" else a.contiguous().cuda()
    if args.dtype_b == "float4":
        b_d = b.data.T.contiguous().cuda() if args.transpose_b else b.data.contiguous().cuda()
    else:
        b_d = b.T.contiguous().cuda() if args.transpose_b else b.contiguous().cuda()
    a_scale_d = a_scale_input.cuda() if a_scale_input is not None else None
    b_scale_d = b_scale_input.cuda()

    mxgemm_tdm_pipelined(
        a_d,
        b_d,
        a_scale_d,
        b_scale_d,
        args.BM,
        args.BN,
        args.BK,
        args.transpose_b,
        args.num_buffers,
        DTYPE_TO_TRITON[args.dtype_a],
        DTYPE_TO_TRITON[args.dtype_b],
        args.scale_preshuffled,
        args.with_a_scale,
        args.schedule,
        args.l2_prefetch_distance,
        args.M,
        args.N,
        args.K,
        "partial" if args.partial_tdm else args.tdm_fusion,
        args.tdm_split,
        args.group_size_m,
        None if args.benchmark_mode == "none" else args.benchmark_mode,
        args.benchmark_num_iters,
        BENCHMARK_WARMUP_SECONDS=args.benchmark_warmup_seconds,
        BENCHMARK_DURATION_SECONDS=args.benchmark_duration_seconds,
        PERSISTENT=args.persistent,
        NUM_PROGRAMS=args.num_programs,
        CROSS_TILE_PREFETCH=args.cross_tile_prefetch,
        XCD_REMAP=args.xcd_remap,
        NUM_XCDS=args.num_xcds,
        XCD_CHUNK=args.xcd_chunk,
        CLUSTER_SIZE=args.cluster_size,
        CLUSTER_MULTICAST=args.cluster_multicast,
        CLUSTER_BARRIER_INTERVAL=args.cluster_barrier_interval,
        CLUSTER_LAYOUT=args.cluster_layout,
        SPLIT_CLUSTER_BARRIER=args.split_cluster_barrier,
        EARLY_C_STORES=args.early_c_stores,
        STATIC_SLOT_UNROLL=args.static_slot_unroll,
        OUTPUT_C_SLOTS=args.output_c_slots,
        OVERLAP_PEEL=args.overlap_peel,
        MERGED_HANDOFF_BARRIER=args.merged_handoff_barrier,
        STATIC_CLUSTER_STEPS=args.static_cluster_steps,
        SCALE_LAYOUT=args.scale_layout,
        OUT_DTYPE=args.out_dtype,
        TILES_PER_WARP=tuple(args.tiles_per_warp),
    )
