"""
MXFP TDM-pipelined GEMM for AMD gfx1250 using TLX.

``mxgemm_tdm_pipelined_kernel`` mirrors the structure of the Gluon MXFP GEMM
example: descriptor-backed loads for A/B and scales, optional A scales, L2
prefetch hints, and baseline or sliced K/N/MNK compute schedules.

``mxgemm_tdm_persistent_kernel`` is a persistent sliceMNK kernel for
256x256 output tiles with FP8 A and FP8 or FP4 B. It carries its TDM input
ring across output tiles, prefetches the next tile during the final K steps,
stages C through LDS for TDM stores, and can multicast inputs across
workgroup clusters. See README.md for the design.

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


@triton.constexpr_function
def _operand_shared_layout(block_shape):
    block_k = block_shape[1]
    pad_interval = 256 if block_k <= 256 else block_k
    return tlx.padded_shared_layout_encoding.with_identity_for([[pad_interval, 16]], block_shape, [1, 0])


@triton.constexpr_function
def _scale_shared_layout(block_shape):
    return tlx.padded_shared_layout_encoding.with_identity_for([[256, 8]], block_shape, [1, 0])


@triton.constexpr_function
def _partitioned_operand_shared_layout(block_shape):
    # Group-major operand view [groups, 2 * rows, K]: alternate rows-row
    # pieces between two physical LDS partitions. Partition p holds pieces
    # p, p + 2, ... contiguously, so one TDM instruction per issuing wave fills
    # it, and _mxgemm_flat_view reads it as [groups * 2 * rows, K].
    groups, group_rows, cols = block_shape
    pad_interval = 256 if cols <= 256 else cols
    piece = tlx.padded_shared_layout_encoding.with_identity_for([[pad_interval, 16]], [groups, group_rows // 2, cols],
                                                                [2, 1, 0])
    return tlx.partitioned_shared_layout_encoding(2, 1, 1, piece)


@tl.core.builtin
def _mxgemm_view_rows(view, _semantic=None):
    return tl.constexpr(tl.core._unwrap_if_constexpr(view.type.shape[0]))


@tl.core.builtin
def _mxgemm_desc_group_rows(desc, _semantic=None):
    # Rows per group of a group-major [M // R, R, K] operand descriptor, or 0
    # for an ordinary [M, K] descriptor.
    shape = [tl.core._unwrap_if_constexpr(d) for d in desc.block_shape]
    return tl.constexpr(shape[1] if len(shape) == 3 else 0)


@tl.core.builtin
def _mxgemm_flat_view(view, _semantic=None):
    # Read a group-major [G, R, K] operand slot as [G * R, K].
    shape = [tl.core._unwrap_if_constexpr(d) for d in view.type.shape]
    if len(shape) != 3:
        return view
    return tlx.local_reshape(view, [shape[0] * shape[1], shape[2]], _semantic=_semantic)


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


def pack_scale(x: torch.Tensor, preshuffle_factor: int = 128) -> torch.Tensor:
    if x is None:
        return x
    non_k, k_scale = x.shape
    scale_kwidth = 4 if k_scale >= 4 else k_scale
    num_chunk_m = non_k // preshuffle_factor
    num_chunk_k = k_scale // scale_kwidth
    x = x.view(num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, scale_kwidth)
    x = x.permute(0, 3, 2, 1, 4).contiguous()
    return x.view(non_k // preshuffle_factor, k_scale * preshuffle_factor)


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
def _mxgemm_c_slot(buf, idx, ROWS: tl.constexpr, COLS: tl.constexpr):
    # C buffers that alias input ring slots keep the input slot pitch. A
    # narrower C type allocates proportionally more rows per slot with the
    # same element padding; use the leading ROWS rows of the indexed slot.
    view = buf[idx]
    if _mxgemm_view_rows(view) != ROWS:
        view = tlx.local_slice(view, [0, 0], [ROWS, COLS])
    return view


@triton.jit
def _mxgemm_position_load_descriptor(desc, offsets, pred):
    # Persistent loads inherit the enabled base descriptor's predicate.
    # Replacing it on every update can keep packed descriptor fields live
    # across the K loop and spill them into VGPR lanes.
    if _mxgemm_desc_group_rows(desc) != 0:
        offsets = [0, offsets[0], offsets[1]]
    if pred is None:
        return tlx.update_tensor_descriptor(desc, add_offsets=offsets, clamp_bounds=False)
    else:
        return tlx.update_tensor_descriptor(desc, add_offsets=offsets, clamp_bounds=False, pred=pred)


@triton.jit
def _mxgemm_issue_load_a_scale(
    a_scale_desc,
    a_scale_buf,
    load_idx,
    pred,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
    load_slot=None,
):
    if WITH_A_SCALE:
        if pred is None:
            pred = True
        slot = load_idx % NUM_BUFFERS if load_slot is None else load_slot
        if SCALE_PRESHUFFLE:
            scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
        else:
            scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]
        tlx.async_amd_descriptor_load(a_scale_desc, tlx.local_view(a_scale_buf, slot), scale_offsets, pred=pred,
                                      clamp_bounds=False)


@triton.jit
def _mxgemm_issue_load_b_scale(
    b_scale_desc,
    b_scale_buf,
    load_idx,
    pred,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    load_slot=None,
):
    if pred is None:
        pred = True
    slot = load_idx % NUM_BUFFERS if load_slot is None else load_slot
    if SCALE_PRESHUFFLE:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
    else:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]
    tlx.async_amd_descriptor_load(b_scale_desc, tlx.local_view(b_scale_buf, slot), scale_offsets, pred=pred,
                                  clamp_bounds=False)


@triton.jit
def _mxgemm_issue_load_a_data(
    a_desc,
    a_buf,
    load_idx,
    pred,
    BLOCK_K_PACKED_A: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    load_slot=None,
):
    if pred is None:
        pred = True
    slot = load_idx % NUM_BUFFERS if load_slot is None else load_slot
    tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, slot), [0, load_idx * BLOCK_K_PACKED_A], pred=pred,
                                  clamp_bounds=False)


@triton.jit
def _mxgemm_issue_load_b_data(
    b_desc,
    b_buf,
    load_idx,
    pred,
    BLOCK_K_PACKED_B: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    load_slot=None,
):
    if pred is None:
        pred = True
    slot = load_idx % NUM_BUFFERS if load_slot is None else load_slot
    if TRANSPOSE_B:
        b_offsets = [0, load_idx * BLOCK_K_PACKED_B]
    else:
        b_offsets = [load_idx * BLOCK_K_PACKED_B, 0]
    tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, slot), b_offsets, pred=pred, clamp_bounds=False)


@triton.jit
def _mxgemm_issue_loads(
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    a_buf,
    b_buf,
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
    CLUSTER_SIZE: tl.constexpr = 1,
    CLUSTER_MULTICAST: tl.constexpr = True,
    GROUP_SIZE_M: tl.constexpr = 8,
    XCD_REMAP_MODE: tl.constexpr = 0,
    load_slot=None,
    SCALES_FIRST: tl.constexpr = False,
    CLUSTER_LAYOUT: tl.constexpr = "linear",
):
    slot = load_idx % NUM_BUFFERS if load_slot is None else load_slot
    if CLUSTER_SIZE > 1 and CLUSTER_MULTICAST:
        tl.static_assert(WITH_A_SCALE and TDM_FUSION != "none")
        tl.static_assert(GROUP_SIZE_M == 4 or GROUP_SIZE_M == 8)
        rank = tlx.cluster_cta_rank()
        if CLUSTER_LAYOUT == "2x2" or (XCD_REMAP_MODE == 2 and CLUSTER_SIZE == 4 and GROUP_SIZE_M == 4):
            a_mask, b_mask = 5 << (rank & 1), 3 << (rank & 2)
        elif CLUSTER_LAYOUT != "linear":
            a_mask, b_mask = _mxgemm_grid_masks(rank, CLUSTER_LAYOUT)
        else:
            # Without remapping, consecutive cluster CTAs cover consecutive
            # M tiles at the same N, so only B and its scales are shared.
            a_mask, b_mask = 0, (1 << CLUSTER_SIZE) - 1
        data_masks = [a_mask, b_mask]
        all_masks = [a_mask, b_mask, a_mask, b_mask]
    else:
        data_masks = None
        all_masks = None
    if TRANSPOSE_B:
        b_offsets = [0, load_idx * BLOCK_K_PACKED_B]
    else:
        b_offsets = [load_idx * BLOCK_K_PACKED_B, 0]
    if SCALE_PRESHUFFLE:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
    else:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]

    # Spread each fused member over the same fraction of the workgroup when
    # two waves run on each SIMD. The additional waves load disjoint rows.
    WAVE_MASK_REPEAT: tl.constexpr = 0x11 if tlx.num_warps() == 8 else 1
    A_WAVES: tl.constexpr = 5 * WAVE_MASK_REPEAT
    B_WAVES: tl.constexpr = 10 * WAVE_MASK_REPEAT
    if TDM_FUSION == "4way":
        tl.static_assert(WITH_A_SCALE, "4-way TDM fusion requires WITH_A_SCALE")
        tlx.async_amd_descriptor_load_fused([
            (_mxgemm_position_load_descriptor(a_desc, [0, load_idx * BLOCK_K_PACKED_A],
                                              pred), tlx.local_view(a_buf, slot), 1 * WAVE_MASK_REPEAT),
            (_mxgemm_position_load_descriptor(b_desc, b_offsets, pred), tlx.local_view(b_buf,
                                                                                       slot), 2 * WAVE_MASK_REPEAT),
            (_mxgemm_position_load_descriptor(a_scale_desc, scale_offsets, pred), tlx.local_view(
                a_scale_buf, slot), 4 * WAVE_MASK_REPEAT),
            (_mxgemm_position_load_descriptor(b_scale_desc, scale_offsets, pred), tlx.local_view(
                b_scale_buf, slot), 8 * WAVE_MASK_REPEAT),
        ], multicast_masks=all_masks)
    elif TDM_FUSION == "2way":
        tlx.async_amd_descriptor_load_fused([
            (_mxgemm_position_load_descriptor(a_desc, [0, load_idx * BLOCK_K_PACKED_A],
                                              pred), tlx.local_view(a_buf, slot), 3 * WAVE_MASK_REPEAT),
            (_mxgemm_position_load_descriptor(b_desc, b_offsets, pred), tlx.local_view(b_buf,
                                                                                       slot), 12 * WAVE_MASK_REPEAT),
        ], multicast_masks=data_masks)
        if WITH_A_SCALE:
            tlx.async_amd_descriptor_load_fused([
                (_mxgemm_position_load_descriptor(a_scale_desc, scale_offsets, pred), tlx.local_view(
                    a_scale_buf, slot), 3 * WAVE_MASK_REPEAT),
                (_mxgemm_position_load_descriptor(b_scale_desc, scale_offsets, pred), tlx.local_view(
                    b_scale_buf, slot), 12 * WAVE_MASK_REPEAT),
            ], multicast_masks=data_masks)
        else:
            _mxgemm_issue_load_b_scale(b_scale_desc, b_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED,
                                       NUM_BUFFERS, SCALE_PRESHUFFLE, load_slot=slot)
    elif TDM_FUSION == "partial":
        tl.static_assert(WITH_A_SCALE, "partial TDM fusion requires WITH_A_SCALE")
        if SCALES_FIRST:
            # Scale-first schedules read scales early. Enqueue their small
            # transfers before the larger payload copies.
            tlx.async_amd_descriptor_load_fused([
                (_mxgemm_position_load_descriptor(a_scale_desc, scale_offsets, pred), tlx.local_view(a_scale_buf,
                                                                                                     slot), A_WAVES),
                (_mxgemm_position_load_descriptor(b_scale_desc, scale_offsets, pred), tlx.local_view(b_scale_buf,
                                                                                                     slot), B_WAVES),
            ], multicast_masks=data_masks)
        tlx.async_amd_descriptor_load_fused([
            (_mxgemm_position_load_descriptor(a_desc, [0, load_idx * BLOCK_K_PACKED_A],
                                              pred), tlx.local_view(a_buf, slot), A_WAVES),
            (_mxgemm_position_load_descriptor(b_desc, b_offsets, pred), tlx.local_view(b_buf, slot), B_WAVES),
        ], multicast_masks=data_masks)
        if not SCALES_FIRST:
            tlx.async_amd_descriptor_load_fused([
                (_mxgemm_position_load_descriptor(a_scale_desc, scale_offsets, pred), tlx.local_view(a_scale_buf,
                                                                                                     slot), A_WAVES),
                (_mxgemm_position_load_descriptor(b_scale_desc, scale_offsets, pred), tlx.local_view(b_scale_buf,
                                                                                                     slot), B_WAVES),
            ], multicast_masks=data_masks)
    else:
        tl.static_assert(TDM_FUSION == "none", "TDM_FUSION must be one of: none, 2way, 4way, partial")
        _mxgemm_issue_load_a_scale(a_scale_desc, a_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS,
                                   SCALE_PRESHUFFLE, WITH_A_SCALE, load_slot=slot)
        _mxgemm_issue_load_b_scale(b_scale_desc, b_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS,
                                   SCALE_PRESHUFFLE, load_slot=slot)
        _mxgemm_issue_load_a_data(a_desc, a_buf, load_idx, pred, BLOCK_K_PACKED_A, NUM_BUFFERS, load_slot=slot)
        _mxgemm_issue_load_b_data(b_desc, b_buf, load_idx, pred, BLOCK_K_PACKED_B, NUM_BUFFERS, TRANSPOSE_B,
                                  load_slot=slot)
    return load_idx + 1


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
def _mxgemm_load_a_operand(
    a_buf,
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
    SCALES_FIRST: tl.constexpr = False,
):
    slot = wmma_idx % NUM_BUFFERS
    subtile_m: tl.constexpr = BLOCK_M // NUM_SUBTILES_M
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_m: tl.constexpr = subtile_start_idx_m * subtile_m
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    a_view = _mxgemm_flat_view(tlx.local_view(a_buf, slot))
    if NUM_SUBTILES_M != 1 or NUM_SUBTILES_K != 1:
        a_view = tlx.local_slice(a_view, [subtile_start_m, subtile_start_k // DIV_FACTOR_A],
                                 [subtile_m, subtile_k // DIV_FACTOR_A])
    if not SCALES_FIRST:
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
        if NUM_SUBTILES_M != 1 or NUM_SUBTILES_K != 1:
            scale_a_view = tlx.local_slice(scale_a_view, [subtile_start_m, subtile_start_k // SCALE_BLOCK],
                                           [subtile_m, subtile_k // SCALE_BLOCK])
        elif not SCALE_PRESHUFFLE:
            scale_a_view = tlx.local_slice(scale_a_view, [0, 0], [BLOCK_M, BLOCK_K_SCALE])
        scale_a = tlx.local_load(scale_a_view)
    else:
        scale_a = tl.full((subtile_m, subtile_k // SCALE_BLOCK), 127, dtype=tl.uint8)

    if SCALES_FIRST:
        a = tlx.local_load(a_view)
    return a, scale_a


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
def _mxgemm_load_b_operand(
    b_buf,
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
    SCALES_FIRST: tl.constexpr = False,
):
    slot = wmma_idx % NUM_BUFFERS
    subtile_n: tl.constexpr = BLOCK_N // NUM_SUBTILES_N
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_n: tl.constexpr = subtile_start_idx_n * subtile_n
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    b_view = _mxgemm_flat_view(tlx.local_view(b_buf, slot))
    if TRANSPOSE_B:
        if NUM_SUBTILES_N != 1 or NUM_SUBTILES_K != 1:
            b_view = tlx.local_slice(b_view, [subtile_start_n, subtile_start_k // DIV_FACTOR_B],
                                     [subtile_n, subtile_k // DIV_FACTOR_B])
        b_view = tlx.local_trans(b_view)
    else:
        if NUM_SUBTILES_N != 1 or NUM_SUBTILES_K != 1:
            b_view = tlx.local_slice(b_view, [subtile_start_k // DIV_FACTOR_B, subtile_start_n],
                                     [subtile_k // DIV_FACTOR_B, subtile_n])
    if not SCALES_FIRST:
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
    if NUM_SUBTILES_N != 1 or NUM_SUBTILES_K != 1:
        scale_b_view = tlx.local_slice(scale_b_view, [subtile_start_n, subtile_start_k // SCALE_BLOCK],
                                       [subtile_n, subtile_k // SCALE_BLOCK])
    elif not SCALE_PRESHUFFLE:
        scale_b_view = tlx.local_slice(scale_b_view, [0, 0], [BLOCK_N, BLOCK_K_SCALE])
    scale_b = tlx.local_load(scale_b_view)

    if SCALES_FIRST:
        b = tlx.local_load(b_view)
    return b, scale_b


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

_XCD_REMAP_MODES = {"none": 0, "balanced": 1, "chunked": 2}


@triton.jit
def _mxgemm_remap_program_id(pid, NUM_PROGRAMS: tl.constexpr, XCD_REMAP_MODE: tl.constexpr, NUM_XCDS: tl.constexpr,
                             XCD_CHUNK: tl.constexpr):
    tl.static_assert(NUM_XCDS > 0 and XCD_CHUNK > 0)
    if XCD_REMAP_MODE == 0 or NUM_XCDS == 1:
        return pid
    elif XCD_REMAP_MODE == 1:
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        per_xcd = NUM_PROGRAMS // NUM_XCDS
        extra = NUM_PROGRAMS % NUM_XCDS
        return xcd * per_xcd + tl.minimum(xcd, extra) + local_pid
    else:
        tl.static_assert(XCD_REMAP_MODE == 2)
        aligned = NUM_PROGRAMS // (NUM_XCDS * XCD_CHUNK) * (NUM_XCDS * XCD_CHUNK)
        if pid >= aligned:
            return pid
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        return (local_pid // XCD_CHUNK) * NUM_XCDS * XCD_CHUNK + xcd * XCD_CHUNK + local_pid % XCD_CHUNK


# Grid cluster layouts "RxC" place R M tiles by C N tiles in one cluster. Rank r
# covers M tile r % R and N tile r // R; A is shared by the C ranks with equal
# r % R, B by the R consecutive ranks with equal r // R.
_MXGEMM_GRID_LAYOUTS = {"2x2": (2, 2), "4x2": (4, 2), "2x4": (2, 4), "4x4": (4, 4), "8x2": (8, 2), "2x8": (2, 8)}


@triton.jit
def _mxgemm_grid_rows(CLUSTER_LAYOUT: tl.constexpr):
    R: tl.constexpr = (2 if CLUSTER_LAYOUT == "2x2" or CLUSTER_LAYOUT == "2x4" or CLUSTER_LAYOUT == "2x8" else
                       (4 if CLUSTER_LAYOUT == "4x2" or CLUSTER_LAYOUT == "4x4" else
                        (8 if CLUSTER_LAYOUT == "8x2" else 1)))
    return R


@triton.jit
def _mxgemm_grid_cols(CLUSTER_LAYOUT: tl.constexpr):
    C: tl.constexpr = (2 if CLUSTER_LAYOUT == "2x2" or CLUSTER_LAYOUT == "4x2" or CLUSTER_LAYOUT == "8x2" else
                       (4 if CLUSTER_LAYOUT == "2x4" or CLUSTER_LAYOUT == "4x4" else
                        (8 if CLUSTER_LAYOUT == "2x8" else 1)))
    return C


@triton.jit
def _mxgemm_grid_masks(rank, CLUSTER_LAYOUT: tl.constexpr):
    R: tl.constexpr = _mxgemm_grid_rows(CLUSTER_LAYOUT)
    C: tl.constexpr = _mxgemm_grid_cols(CLUSTER_LAYOUT)
    # Bits R*c for every column c: a geometric series in 2^R.
    A_PATTERN: tl.constexpr = ((1 << (R * C)) - 1) // ((1 << R) - 1)
    B_PATTERN: tl.constexpr = (1 << R) - 1
    return A_PATTERN << (rank % R), B_PATTERN << (R * (rank // R))


@triton.jit
def _mxgemm_cluster_program_id(pid, NUM_PROGRAMS: tl.constexpr, XCD_REMAP_MODE: tl.constexpr, NUM_XCDS: tl.constexpr,
                               XCD_CHUNK: tl.constexpr, CLUSTER_LAYOUT: tl.constexpr = "linear"):
    if CLUSTER_LAYOUT == "2x2":
        # Remap whole clusters: physical rank bit 0 selects M, bit 1 selects N.
        # This keeps multicast partners together for every persistent stride.
        return 4 * _mxgemm_remap_program_id(pid // 4, NUM_PROGRAMS // 4, XCD_REMAP_MODE, NUM_XCDS, XCD_CHUNK) + pid % 4
    if CLUSTER_LAYOUT != "linear":
        S: tl.constexpr = _mxgemm_grid_rows(CLUSTER_LAYOUT) * _mxgemm_grid_cols(CLUSTER_LAYOUT)
        return S * _mxgemm_remap_program_id(pid // S, NUM_PROGRAMS // S, XCD_REMAP_MODE, NUM_XCDS, XCD_CHUNK) + pid % S
    return _mxgemm_remap_program_id(pid, NUM_PROGRAMS, XCD_REMAP_MODE, NUM_XCDS, XCD_CHUNK)


@triton.jit
def _mxgemm_tile_offsets(tile, num_m, num_n, GROUP_M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                         CLUSTER_LAYOUT: tl.constexpr = "linear"):
    group = tile // (GROUP_M * num_n)
    first_m = group * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    local = tile % (GROUP_M * num_n)
    if CLUSTER_LAYOUT == "2x2":
        # Clustered GEMM validates complete M groups. Keep the divisor constant
        # there; the general mapping still supports a shorter final group.
        if num_m % GROUP_M == 0:
            group_m = GROUP_M
        pair = local // 4
        return (first_m + 2 * (pair % (group_m // 2)) + local % 2) * BM, (2 * (pair //
                                                                               (group_m // 2)) + local // 2 % 2) * BN
    if CLUSTER_LAYOUT != "linear":
        # Clusters tile each complete M group as R x C blocks, M first.
        R: tl.constexpr = _mxgemm_grid_rows(CLUSTER_LAYOUT)
        C: tl.constexpr = _mxgemm_grid_cols(CLUSTER_LAYOUT)
        cluster = local // (R * C)
        rank = local % (R * C)
        return (first_m + R * (cluster % (GROUP_M // R)) + rank % R) * BM, (C * (cluster //
                                                                                 (GROUP_M // R)) + rank // R) * BN
    return (first_m + local % group_m) * BM, (local // group_m) * BN


@triton.jit
def _mxgemm_cluster_arrive(k, CLUSTER_SIZE: tl.constexpr, INTERVAL: tl.constexpr):
    if CLUSTER_SIZE > 1 and INTERVAL > 0:
        if k % INTERVAL == 0:
            # This rendezvous only aligns independent multicast requests.
            # Local LDS lifetime protection remains at the TDM refill.
            tlx.amd_cluster_arrive()
            tlx.amd_sched_barrier()


@triton.jit
def _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, off_m, off_n, k, slot,
                            BK: tl.constexpr, DA: tl.constexpr, DB: tl.constexpr, BUFFERS: tl.constexpr,
                            FUSION: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CLUSTER_MULTICAST: tl.constexpr,
                            GROUP_SIZE_M: tl.constexpr, XCD_REMAP_MODE: tl.constexpr,
                            CLUSTER_BARRIER_INTERVAL: tl.constexpr, CLUSTER_LAYOUT: tl.constexpr = "linear",
                            SPLIT_CLUSTER_BARRIER: tl.constexpr = False):
    # Keep the K coordinate independent of the LDS ring slot. Adding (k - slot)
    # here and slot in the loader hides the invariant per-wave global base
    # from LLVM and repeats descriptor address work inside the K loop.
    if _mxgemm_desc_group_rows(a_desc) != 0:
        ad = tlx.update_tensor_descriptor(a_desc, add_offsets=[off_m // _mxgemm_desc_group_rows(a_desc), 0, 0],
                                          clamp_bounds=False)
        bd = tlx.update_tensor_descriptor(b_desc, add_offsets=[off_n // _mxgemm_desc_group_rows(b_desc), 0, 0],
                                          clamp_bounds=False)
    else:
        ad = tlx.update_tensor_descriptor(a_desc, add_offsets=[off_m, 0], clamp_bounds=False)
        bd = tlx.update_tensor_descriptor(b_desc, add_offsets=[off_n, 0], clamp_bounds=False)
    asd = tlx.update_tensor_descriptor(as_desc, add_offsets=[off_m // 128, 0], clamp_bounds=False)
    bsd = tlx.update_tensor_descriptor(bs_desc, add_offsets=[off_n // 128, 0], clamp_bounds=False)
    if CLUSTER_SIZE > 1:
        if CLUSTER_BARRIER_INTERVAL == 1:
            if SPLIT_CLUSTER_BARRIER:
                tlx.amd_cluster_wait()
            else:
                tlx.cluster_barrier()
        elif CLUSTER_BARRIER_INTERVAL > 1:
            if k % CLUSTER_BARRIER_INTERVAL == 0:
                if SPLIT_CLUSTER_BARRIER:
                    tlx.amd_cluster_wait()
                else:
                    tlx.cluster_barrier()
    _mxgemm_issue_loads(ad, bd, asd, bsd, a_buf, b_buf, as_buf, bs_buf, k, None, BK // DA, BK // DB, BK // 32 * 128,
                        BUFFERS, True, True, True, FUSION, CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M,
                        XCD_REMAP_MODE, load_slot=slot, SCALES_FIRST=False, CLUSTER_LAYOUT=CLUSTER_LAYOUT)


@triton.jit
def _mxgemm_persistent_a(a_buf, as_buf, slot, m: tl.constexpr, k: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                         DA: tl.constexpr, BUFFERS: tl.constexpr, PREFETCH_SCALES: tl.constexpr = False):
    tl.assume(slot >= 0)
    tl.assume(slot < BUFFERS)
    a, scale = _mxgemm_load_a_operand(a_buf, as_buf, slot, m, k, BM, BK, DA, 32, BK // 32, BM // 128, 4, BUFFERS, 2,
                                      BK // 128, True, True,
                                      SCALES_FIRST=(PREFETCH_SCALES or ((BUFFERS == 4) and BK == 128)) and DA == 1)
    scale = tlx.require_layout(scale, tlx.layout(shape=((32, 2, 2), (4, BM // 128)), stride=((4, 0, 128), (1, 256))))
    return a, scale


@triton.jit
def _mxgemm_persistent_b(b_buf, bs_buf, slot, k: tl.constexpr, n: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         DB: tl.constexpr, BUFFERS: tl.constexpr, PREFETCH_SCALES: tl.constexpr = False):
    tl.assume(slot >= 0)
    tl.assume(slot < BUFFERS)
    b, scale = _mxgemm_load_b_operand(b_buf, bs_buf, slot, k, n, BN, BK, DB, 32, BK // 32, BN // 128, 4, BUFFERS, 2,
                                      BK // 128, True, True,
                                      SCALES_FIRST=(PREFETCH_SCALES or ((BUFFERS == 4) and BK == 128)) and DB == 1)
    scale = tlx.require_layout(scale, tlx.layout(shape=((32, 2, 2), (4, BN // 128)), stride=((4, 128, 0), (1, 256))))
    return b, scale


@triton.jit
def _mxgemm_schedule_prefetch(sync_id: tl.constexpr):
    for _ in tl.static_range(8):
        tlx.amd_sched_group_barrier(0x8, 1, sync_id)
        tlx.amd_sched_group_barrier(0x100, 2, sync_id)
    tlx.amd_sched_group_barrier(0x8, 1, sync_id)
    tlx.amd_sched_group_barrier(0x100, 1, sync_id)
    tlx.amd_sched_group_barrier(0x8, 7, sync_id)


@triton.jit
def _mxgemm_persistent_compute_top(c00, c01, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BM: tl.constexpr,
                                   BN: tl.constexpr, BK: tl.constexpr, DA: tl.constexpr, DB: tl.constexpr,
                                   DTYPE_B: tl.constexpr, BUFFERS: tl.constexpr, GROUPED_PREFETCH: tl.constexpr = False,
                                   ARRIVE_K=None, CLUSTER_SIZE: tl.constexpr = 1,
                                   CLUSTER_BARRIER_INTERVAL: tl.constexpr = 0, PIN_NATIVE: tl.constexpr = False,
                                   SCALES_FIRST: tl.constexpr = False):
    tl.static_assert(BK == 128)
    dot: tl.constexpr = _mxgemm_native_dot if PIN_NATIVE else tlx.dot_scaled
    b01, sb01 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 0, 1, BN, BK, DB, BUFFERS, PREFETCH_SCALES=SCALES_FIRST)
    c00 = dot(a00, sa00, "e4m3", b00, sb00, DTYPE_B, c00, tiles_per_warp=[2, 2])
    if GROUPED_PREFETCH:
        _mxgemm_schedule_prefetch(101)
    tlx.amd_sched_barrier()
    a10, sa10 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 0, BM, BK, DA, BUFFERS, PREFETCH_SCALES=SCALES_FIRST)
    if ARRIVE_K is not None:
        _mxgemm_cluster_arrive(ARRIVE_K, CLUSTER_SIZE, CLUSTER_BARRIER_INTERVAL)
    c01 = dot(a00, sa00, "e4m3", b01, sb01, DTYPE_B, c01, tiles_per_warp=[2, 2])
    if GROUPED_PREFETCH:
        _mxgemm_schedule_prefetch(102)
    tlx.amd_sched_barrier()
    # Retain the bottom-half operands in registers so the caller can refill
    # this LDS slot and read the next stage before finishing C10/C11.
    return c00, c01, a10, sa10, b01, sb01


@triton.jit
def _mxgemm_defer_bottom_k_step(c00, c01, c10, c11, a00, sa00, b00, sb00, a_desc, b_desc, as_desc, bs_desc, a_buf,
                                b_buf, as_buf, bs_buf, off_m, off_n, i, slot, next_slot, BM: tl.constexpr,
                                BN: tl.constexpr, BK: tl.constexpr, DA: tl.constexpr, DB: tl.constexpr,
                                DTYPE_B: tl.constexpr, BUFFERS: tl.constexpr, FUSION: tl.constexpr, LOADS: tl.constexpr,
                                GROUPED_PREFETCH: tl.constexpr, PREFETCH_B_AFTER_C10: tl.constexpr,
                                CLUSTER_SIZE: tl.constexpr, CLUSTER_MULTICAST: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                                XCD_REMAP_MODE: tl.constexpr, CLUSTER_BARRIER_INTERVAL: tl.constexpr,
                                CLUSTER_LAYOUT: tl.constexpr, SPLIT_CLUSTER_BARRIER: tl.constexpr):
    # One steady BK128 step of the deferred-bottom schedule. This matches the
    # DEFER_BOTTOM body of the persistent kernel's dynamic-slot K loop.
    # Pin every dot to the output store's native WMMA layout. A peeled step
    # starts from constant zeros and remainder steps follow the loop; without
    # pins, layout selection can pick different warp distributions for these
    # dots and insert accumulator conversions through LDS.
    c00, c01, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_top(
        c00, c01, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BM, BN, BK, DA, DB, DTYPE_B, BUFFERS,
        GROUPED_PREFETCH, ARRIVE_K=i + BUFFERS if SPLIT_CLUSTER_BARRIER else None, CLUSTER_SIZE=CLUSTER_SIZE,
        CLUSTER_BARRIER_INTERVAL=CLUSTER_BARRIER_INTERVAL, PIN_NATIVE=True)
    last_b00, last_sb00 = b00, sb00
    _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, off_m, off_n, i + BUFFERS,
                            slot, BK, DA, DB, BUFFERS, FUSION, CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M,
                            XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT=CLUSTER_LAYOUT,
                            SPLIT_CLUSTER_BARRIER=SPLIT_CLUSTER_BARRIER)
    tlx.async_amd_descriptor_wait((BUFFERS - 1) * LOADS)
    a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 0, 0, BM, BK, DA, BUFFERS)
    if not PREFETCH_B_AFTER_C10:
        b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, 0, 0, BN, BK, DB, BUFFERS)
    c10 = _mxgemm_native_dot(a10, sa10, "e4m3", last_b00, last_sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
    if GROUPED_PREFETCH:
        _mxgemm_schedule_prefetch(103)
    tlx.amd_sched_barrier()
    if PREFETCH_B_AFTER_C10:
        b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, 0, 0, BN, BK, DB, BUFFERS)
    c11 = _mxgemm_native_dot(a10, sa10, "e4m3", b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
    if GROUPED_PREFETCH:
        _mxgemm_schedule_prefetch(104)
    tlx.amd_sched_barrier()
    return c00, c01, c10, c11, a00, sa00, b00, sb00


@triton.jit
def _mxgemm_native_operand(value, index: tl.constexpr, dtype: tl.constexpr):
    parent: tl.constexpr = tlx.amd_wmma_layout(((0, 2), (2, 0)), ((0, 1), (1, 0)),
                                               instr_shape=(16, 16, 64 if dtype == "e2m1" else 128))
    return tlx.require_layout(value, tlx.dot_operand_layout(index, parent, k_width=16))


@triton.jit
def _mxgemm_native_dot(a, sa, dtype_a: tl.constexpr, b, sb, dtype_b: tl.constexpr, c,
                       tiles_per_warp: tl.constexpr = (2, 2)):
    a = _mxgemm_native_operand(a, 0, dtype_a)
    b = _mxgemm_native_operand(b, 1, dtype_b)
    c = tlx.require_amd_wmma_layout(c, warp_bases=((0, 2), (2, 0)), reg_bases=((0, 1), (1, 0)),
                                    instr_shape=(16, 16, 128))
    return tlx.dot_scaled(a, sa, dtype_a, b, sb, dtype_b, c, tiles_per_warp=tiles_per_warp)


@triton.jit
def _mxgemm_persistent_compute_256(c00, c01, c10, c11, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot,
                                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DA: tl.constexpr,
                                   DB: tl.constexpr, DTYPE_B: tl.constexpr, BUFFERS: tl.constexpr,
                                   PIN_NATIVE: tl.constexpr = False, ARRIVE_K=None, CLUSTER_SIZE: tl.constexpr = 1,
                                   CLUSTER_BARRIER_INTERVAL: tl.constexpr = 0):
    tl.static_assert(BK == 256)
    dot: tl.constexpr = _mxgemm_native_dot if PIN_NATIVE else tlx.dot_scaled
    c00 = dot(a00, sa00, "e4m3", b00, sb00, DTYPE_B, c00, tiles_per_warp=[2, 2])
    b01, sb01 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 0, 1, BN, BK, DB, BUFFERS)
    c01 = dot(a00, sa00, "e4m3", b01, sb01, DTYPE_B, c01, tiles_per_warp=[2, 2])
    a10, sa10 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 0, BM, BK, DA, BUFFERS)
    if ARRIVE_K is not None:
        _mxgemm_cluster_arrive(ARRIVE_K, CLUSTER_SIZE, CLUSTER_BARRIER_INTERVAL)
    c10 = dot(a10, sa10, "e4m3", b00, sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
    b10, sb10 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 1, 0, BN, BK, DB, BUFFERS)
    a01, sa01 = _mxgemm_persistent_a(a_buf, as_buf, slot, 0, 1, BM, BK, DA, BUFFERS)
    if DB != 2:
        c11 = dot(a10, sa10, "e4m3", b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
    # Bound operand lifetimes before gathering the remaining second half.
    # A8W4 keeps the first-half operands live across the refill.
    tlx.amd_sched_barrier()
    if DB != 2:
        c00 = dot(a01, sa01, "e4m3", b10, sb10, DTYPE_B, c00, tiles_per_warp=[2, 2])
    b11, sb11 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 1, 1, BN, BK, DB, BUFFERS)
    if DB != 2:
        c01 = dot(a01, sa01, "e4m3", b11, sb11, DTYPE_B, c01, tiles_per_warp=[2, 2])
    a11, sa11 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 1, BM, BK, DA, BUFFERS)
    if PIN_NATIVE:
        a01 = _mxgemm_native_operand(a01, 0, "e4m3")
        a11 = _mxgemm_native_operand(a11, 0, "e4m3")
        b10 = _mxgemm_native_operand(b10, 1, DTYPE_B)
        b11 = _mxgemm_native_operand(b11, 1, DTYPE_B)
        a10 = _mxgemm_native_operand(a10, 0, "e4m3")
        b01 = _mxgemm_native_operand(b01, 1, DTYPE_B)
    tlx.amd_sched_barrier()
    # All operands are in registers. Refill this slot and read the next stage
    # before finishing the deferred first- and second-half quadrants.
    return c00, c01, c10, c11, a01, sa01, a11, sa11, b10, sb10, b11, sb11, a10, sa10, b01, sb01


@triton.jit
def _mxgemm_c_slot_wait(PENDINGS: tl.constexpr):
    tlx.async_amd_descriptor_wait(PENDINGS)


@triton.jit
def _mxgemm_c_tdm_store(desc, view, offsets):
    tlx.async_amd_descriptor_store(desc, view, offsets, clamp_bounds=False)


@triton.jit
def _mxgemm_persistent_store(c_ptr, acc, off_m, off_n, stride_cm, BM: tl.constexpr, BN: tl.constexpr,
                             INSTR_M: tl.constexpr, c_buf=None, C_ROWS: tl.constexpr = 64, C_SLOTS: tl.constexpr = 2,
                             QUADRANT: tl.constexpr = 0, C_COLS: tl.constexpr = 128,
                             INITIAL_C_SLOTS_FREE: tl.constexpr = False, EXTRA_PENDING: tl.constexpr = 0,
                             PART_BEGIN: tl.constexpr = 0, PART_END: tl.constexpr = None):
    # EXTRA_PENDING: TDM operations issued after older chunks that each slot
    # reuse may leave pending. PART_BEGIN/PART_END store a subrange of the
    # 64-row chunks. Both apply to dedicated 64-row slots only.
    # Pin values so output extraction stays in registers. Direct stores also
    # pin their offsets below to avoid an implicit LDS transpose.
    WARP_BASES: tl.constexpr = ((0, 2), (2, 0))
    acc = tlx.require_amd_wmma_layout(acc, warp_bases=WARP_BASES, reg_bases=((0, 1), (1, 0)),
                                      instr_shape=(INSTR_M, 16, 128))
    # Narrow outputs convert in registers; staging and stores use C's type.
    acc = acc.to(tlx.dtype_of(c_ptr))
    base = c_ptr + off_m.to(tl.int64) * stride_cm + off_n
    if c_buf is not None:
        tl.static_assert(BM == 128 and BN == 128 and INSTR_M == 16)
        desc = tl.make_tensor_descriptor(base, [BM, BN], [stride_cm, tl.constexpr(1)], [C_ROWS, C_COLS])
        tl.static_assert(C_ROWS == 64 or (EXTRA_PENDING == 0 and PART_BEGIN == 0 and PART_END is None))
        if C_ROWS == 128 and C_COLS == 128:
            if not INITIAL_C_SLOTS_FREE or QUADRANT >= C_SLOTS:
                _mxgemm_c_slot_wait(C_SLOTS - 1)
            view = _mxgemm_c_slot(c_buf, QUADRANT % C_SLOTS, C_ROWS, C_COLS)
            tlx.local_store(view, acc)
            _mxgemm_c_tdm_store(desc, view, [0, 0])
        elif C_ROWS == 128:
            tl.static_assert(C_COLS == 64 and C_SLOTS == 2)
            # A packed-FP4 B stage holds half a quadrant. Split the
            # register-owned N bit and keep each TDM row's padding intact.
            left, right = tl.split(tl.reshape(acc, (BM, 2, C_COLS)).permute(0, 2, 1))
            for part in tl.static_range(2):
                if not INITIAL_C_SLOTS_FREE:
                    _mxgemm_c_slot_wait(1)
                view = _mxgemm_c_slot(c_buf, part, C_ROWS, C_COLS)
                tlx.local_store(view, left if part == 0 else right)
                _mxgemm_c_tdm_store(desc, view, [0, part * C_COLS])
        else:
            tl.static_assert(C_ROWS == 64 and C_SLOTS >= 2 and (C_COLS == 64 or C_COLS == 128))
            PARTS: tl.constexpr = 2 * (BN // C_COLS)
            for part in tl.static_range(PART_BEGIN, PARTS if PART_END is None else PART_END):
                # Dedicated slots are free on entry: the current tile's tail
                # waits retired any previous C stores. Start the first C_SLOTS
                # chunks while next-tile inputs remain in flight, then wait
                # only on reuse. Chunks rotate through slots across quadrants.
                # Chunk QUADRANT * PARTS + part, computed inline so tuple slot
                # views keep a constant index.
                if not INITIAL_C_SLOTS_FREE or QUADRANT * PARTS + part >= C_SLOTS:
                    _mxgemm_c_slot_wait(C_SLOTS - 1 + EXTRA_PENDING)
                lo, hi = tl.split(tl.reshape(acc, (2, 64, BN)).permute(1, 2, 0))
                if C_COLS == 128:
                    chunk = lo if part == 0 else hi
                    row, col = part * 64, 0
                else:
                    # N's 64-column split stays within each wave's registers.
                    # A 32-row split would require exchanging values across waves.
                    half = lo if part < 2 else hi
                    left, right = tl.split(tl.reshape(half, (64, 2, 64)).permute(0, 2, 1))
                    chunk = left if part % 2 == 0 else right
                    row, col = (part // 2) * 64, (part % 2) * 64
                view = c_buf[(QUADRANT * PARTS + part) % C_SLOTS]
                tlx.local_store(view, chunk)
                _mxgemm_c_tdm_store(desc, view, [row, col])
    else:
        offsets = tl.arange(0, BM)[:, None] * stride_cm + tl.arange(0, BN)[None, :]
        offsets = tlx.require_amd_wmma_layout(offsets, warp_bases=WARP_BASES, reg_bases=((0, 1), (1, 0)),
                                              instr_shape=(INSTR_M, 16, 128))
        tlx.buffer_store(acc, base, offsets)


@triton.jit
def _mxgemm_overlap_store(c_ptr, acc, off_m, off_n, stride_cm, c_buf, C_SLOTS: tl.constexpr, QUADRANT: tl.constexpr,
                          C_COLS: tl.constexpr, REFILL_CHUNK: tl.constexpr, LOADS: tl.constexpr, a_desc, b_desc,
                          as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, next_m, next_n, n_phase, BK: tl.constexpr,
                          DA: tl.constexpr, DB: tl.constexpr, BUFFERS: tl.constexpr, FUSION: tl.constexpr,
                          CLUSTER_SIZE: tl.constexpr, CLUSTER_MULTICAST: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                          XCD_REMAP_MODE: tl.constexpr, CLUSTER_BARRIER_INTERVAL: tl.constexpr,
                          CLUSTER_LAYOUT: tl.constexpr):
    # Store one 128x128 C quadrant through dedicated 64-row slots. Before
    # chunk REFILL_CHUNK, refill the next tile's stage-0 slot with stage
    # BUFFERS. TDM operations complete in order, so each later chunk's
    # slot-reuse wait also leaves the refill's LOADS operations pending.
    PARTS: tl.constexpr = 2 * (128 // C_COLS)
    SPLIT: tl.constexpr = REFILL_CHUNK - QUADRANT * PARTS
    if SPLIT > 0:
        _mxgemm_persistent_store(c_ptr, acc, off_m, off_n, stride_cm, 128, 128, 16, c_buf, 64, C_SLOTS, QUADRANT,
                                 C_COLS, True, PART_END=SPLIT if SPLIT < PARTS else PARTS)
    if SPLIT >= 0 and SPLIT < PARTS:
        _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, next_m, next_n, BUFFERS,
                                n_phase, BK, DA, DB, BUFFERS, FUSION, CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M,
                                XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT=CLUSTER_LAYOUT)
    if SPLIT < PARTS:
        _mxgemm_persistent_store(c_ptr, acc, off_m, off_n, stride_cm, 128, 128, 16, c_buf, 64, C_SLOTS, QUADRANT,
                                 C_COLS, True, EXTRA_PENDING=LOADS, PART_BEGIN=SPLIT if SPLIT > 0 else 0)


@triton.jit
def mxgemm_tdm_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale,
    b_scale,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_as: tl.constexpr,
    stride_bs: tl.constexpr,
    DTYPE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    CROSS_TILE_PREFETCH: tl.constexpr = True,
    XCD_REMAP_MODE: tl.constexpr = 0,
    NUM_XCDS: tl.constexpr = 8,
    XCD_CHUNK: tl.constexpr = 2,
    CLUSTER_SIZE: tl.constexpr = 1,
    CLUSTER_MULTICAST: tl.constexpr = True,
    CLUSTER_BARRIER_INTERVAL: tl.constexpr = 1,
    CLUSTER_LAYOUT: tl.constexpr = "linear",
    SPLIT_CLUSTER_BARRIER: tl.constexpr = False,
    EARLY_C_STORES: tl.constexpr = False,
    STATIC_SLOT_UNROLL: tl.constexpr = False,
    OUTPUT_C_SLOTS: tl.constexpr = None,
    OVERLAP_PEEL: tl.constexpr = False,
):
    """Full-tile, transposed-B sliceMNK GEMM with a persistent TDM ring."""
    tl.static_assert(BLOCK_M % 128 == 0 and BLOCK_N % 128 == 0 and (BLOCK_K == 128 or BLOCK_K == 256))
    tl.static_assert(NUM_BUFFERS >= 2 and NUM_PROGRAMS > 0 and GROUP_SIZE_M > 0)
    tl.static_assert(CLUSTER_SIZE == 1 or CLUSTER_SIZE == 2 or CLUSTER_SIZE == 4 or CLUSTER_SIZE == 8
                     or CLUSTER_SIZE == 16)
    tl.static_assert(CLUSTER_BARRIER_INTERVAL >= 0)
    tl.static_assert(CLUSTER_LAYOUT == "linear"
                     or (_mxgemm_grid_rows(CLUSTER_LAYOUT) > 1
                         and _mxgemm_grid_rows(CLUSTER_LAYOUT) * _mxgemm_grid_cols(CLUSTER_LAYOUT) == CLUSTER_SIZE))
    tl.static_assert(CLUSTER_LAYOUT == "linear" or GROUP_SIZE_M % _mxgemm_grid_rows(CLUSTER_LAYOUT) == 0)
    if SPLIT_CLUSTER_BARRIER or EARLY_C_STORES:
        tl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and NUM_BUFFERS == 3)
        tl.static_assert((BLOCK_K == 128 and DTYPE_B == "e4m3") or (BLOCK_K == 256 and DTYPE_B == "e2m1"))
    if SPLIT_CLUSTER_BARRIER:
        tl.static_assert(CLUSTER_SIZE > 1 and CLUSTER_BARRIER_INTERVAL > 0)
    if CLUSTER_SIZE > 1:
        # Grid layouts remap whole clusters (any bijection keeps multicast
        # partners together); the 2x2/linear masks assume the chunked 8x2 remap.
        tl.static_assert(XCD_REMAP_MODE == 0 or (XCD_REMAP_MODE == 2 and NUM_XCDS == 8 and XCD_CHUNK == 2)
                         or (CLUSTER_LAYOUT != "linear" and CLUSTER_LAYOUT != "2x2"))
        tl.static_assert(GROUP_SIZE_M == 4 or GROUP_SIZE_M == 8)
        tl.static_assert(NUM_PROGRAMS % 16 == 0)
    DB: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    OUTPUT_REUSE: tl.constexpr = BLOCK_K == 256
    # C buffers that alias input ring slots keep the input slot pitch: a
    # narrower C type allocates C_ROW_SCALE times the rows per slot.
    tl.static_assert(tlx.dtype_of(c_ptr) == tl.float32 or tlx.dtype_of(c_ptr) == tl.float16)
    C_ROW_SCALE: tl.constexpr = 32 // tlx.dtype_of(c_ptr).primitive_bitwidth
    # Static-slot steps pin their dots to the native output layout, so every
    # dot in that schedule must use the same pinned layout.
    dot: tl.constexpr = _mxgemm_native_dot if STATIC_SLOT_UNROLL or OVERLAP_PEEL else tlx.dot_scaled
    tl.static_assert(BLOCK_M == 256 and BLOCK_N == 256)
    if OUTPUT_REUSE:
        if CROSS_TILE_PREFETCH:
            tl.static_assert(NUM_BUFFERS == 3 and DB == 2,
                             "BK256 cross-tile output staging requires three buffers and FP4 B")
        tl.static_assert(NUM_BUFFERS == 2 or (NUM_BUFFERS == 3 and DB == 2))
    else:
        tl.static_assert(NUM_BUFFERS <= 4)
    # Keep independent bottom-half work after the refill and next operand loads.
    # In particular, periodic cluster barriers prevent LLVM from moving earlier
    # WMMAs across that boundary to cover the load latency.
    DEFER_BOTTOM: tl.constexpr = BLOCK_K == 128
    GROUPED_PREFETCH: tl.constexpr = DEFER_BOTTOM and (NUM_BUFFERS == 4) and DB == 1
    PREFETCH_B_AFTER_C10: tl.constexpr = GROUPED_PREFETCH
    # In the three-buffer dynamic-slot loop, read each operand's scale before
    # its payload. The next quadrant's first WMMA then waits only for its own
    # operands instead of draining every outstanding LDS read. The unrolled
    # static loop keeps the payload-first order.
    STEP_SCALES_FIRST: tl.constexpr = DEFER_BOTTOM and NUM_BUFFERS == 3 and not STATIC_SLOT_UNROLL
    LOADS: tl.constexpr = 2

    as_desc = tl.make_tensor_descriptor(a_scale, [M // 128, K // 32 * 128], [stride_as, tl.constexpr(1)],
                                        [BLOCK_M // 128, BLOCK_K // 32 * 128])
    bs_desc = tl.make_tensor_descriptor(b_scale, [N // 128, K // 32 * 128], [stride_bs, tl.constexpr(1)],
                                        [BLOCK_N // 128, BLOCK_K // 32 * 128])
    # With the static-slot loop, alternate 64-row pieces of A and B between
    # two 64 KiB LDS partitions so the waves of a quadrant, which read
    # alternating 32-row blocks of the same operand subtile, contend less for
    # one partition. Group-major descriptors [rows // R, R, K] with
    # R = 2 * PIECE_ROWS let each issuing wave fill one partition with one TDM
    # instruction. Smaller pieces give each wave pair only one partition half,
    # which unbalances the waves.
    PIECE_ROWS: tl.constexpr = 64 if STATIC_SLOT_UNROLL else 0
    if PIECE_ROWS:
        tl.static_assert(DB == 1 and BLOCK_K == 128, "partitioned A/B requires A8W8 BK128")
        R: tl.constexpr = 2 * PIECE_ROWS
        tl.static_assert(M % R == 0 and N % R == 0 and BLOCK_M % R == 0 and BLOCK_N % R == 0)
        a_desc = tl.make_tensor_descriptor(a_ptr, [M // R, R, K],
                                           [R * stride_am, stride_am, tl.constexpr(1)], [BLOCK_M // R, R, BLOCK_K])
        b_desc = tl.make_tensor_descriptor(b_ptr, [N // R, R, K // DB],
                                           [R * stride_bn, stride_bn, tl.constexpr(1)],
                                           [BLOCK_N // R, R, BLOCK_K // DB])
        a_buf = tlx.local_alloc((BLOCK_M // R, R, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS,
                                layout=_partitioned_operand_shared_layout([BLOCK_M // R, R, BLOCK_K]))
        b_buf = tlx.local_alloc((BLOCK_N // R, R, BLOCK_K // DB), tlx.dtype_of(b_ptr), NUM_BUFFERS,
                                layout=_partitioned_operand_shared_layout([BLOCK_N // R, R, BLOCK_K // DB]))
    else:
        a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, tl.constexpr(1)], [BLOCK_M, BLOCK_K])
        b_desc = tl.make_tensor_descriptor(b_ptr, [N, K // DB], [stride_bn, tl.constexpr(1)], [BLOCK_N, BLOCK_K // DB])
        a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS,
                                layout=_operand_shared_layout([BLOCK_M, BLOCK_K]))
        b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K // DB), tlx.dtype_of(b_ptr), NUM_BUFFERS,
                                layout=_operand_shared_layout([BLOCK_N, BLOCK_K // DB]))
    as_buf = tlx.local_alloc((BLOCK_M // 128, BLOCK_K // 32 * 128), tlx.dtype_of(a_scale), NUM_BUFFERS,
                             layout=_scale_shared_layout([BLOCK_M // 128, BLOCK_K // 32 * 128]))
    bs_buf = tlx.local_alloc((BLOCK_N // 128, BLOCK_K // 32 * 128), tlx.dtype_of(b_scale), NUM_BUFFERS,
                             layout=_scale_shared_layout([BLOCK_N // 128, BLOCK_K // 32 * 128]))
    # Two 16 KiB output slots (64x64 FP32 or 64x128 FP16) leave room for a
    # fourth BK128 input stage.
    C_COLS: tl.constexpr = (256 // tlx.dtype_of(c_ptr).primitive_bitwidth * 8) if NUM_BUFFERS == 4 else BLOCK_N // 2
    C_ROWS: tl.constexpr = 128 if OUTPUT_REUSE else 64
    # Dedicated staging defaults to two slots; OUTPUT_C_SLOTS (None selects
    # the default) lets output stores overlap more deeply.
    C_SLOTS: tl.constexpr = NUM_BUFFERS if OUTPUT_REUSE else (2 if OUTPUT_C_SLOTS is None else OUTPUT_C_SLOTS)
    if OUTPUT_C_SLOTS is not None:
        tl.static_assert(not OUTPUT_REUSE, "OUTPUT_C_SLOTS applies to dedicated output staging slots")
    if OUTPUT_REUSE:
        if CROSS_TILE_PREFETCH:
            # Match A's 69,632-byte physical slot stride. The ordinary C
            # padding has a 67,584-byte stride and crosses A-slot boundaries,
            # so retiring one such C slot would not free the matching A slot.
            c_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[128, 8]],
                                                                                         [128 * C_ROW_SCALE, 128],
                                                                                         [1, 0])
            c_buf = tlx.local_alloc((128 * C_ROW_SCALE, 128), tlx.dtype_of(c_ptr), NUM_BUFFERS, layout=c_layout,
                                    reuse=a_buf)
        else:
            # Without overlap, all C transfers drain before reloading A.
            c_buf = tlx.local_alloc((128, BLOCK_N // 2), tlx.dtype_of(c_ptr), NUM_BUFFERS, reuse=a_buf)
    else:
        c_buf = tlx.local_alloc((C_ROWS, C_COLS), tlx.dtype_of(c_ptr), C_SLOTS)
    num_m, num_n = M // BLOCK_M, N // BLOCK_N
    total_tiles = num_m * num_n
    k_iters = K // BLOCK_K
    tl.assume(k_iters >= NUM_BUFFERS)
    tile = _mxgemm_cluster_program_id(tl.program_id(0), NUM_PROGRAMS, XCD_REMAP_MODE, NUM_XCDS, XCD_CHUNK,
                                      CLUSTER_LAYOUT)
    phase = 0
    off_m, off_n = _mxgemm_tile_offsets(tile, num_m, num_n, GROUP_SIZE_M, BLOCK_M, BLOCK_N,
                                        CLUSTER_LAYOUT=CLUSTER_LAYOUT)
    for p in tl.static_range(NUM_BUFFERS):
        _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, off_m, off_n, p, p,
                                BLOCK_K, 1, DB, NUM_BUFFERS, "partial", CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M,
                                XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT=CLUSTER_LAYOUT)

    if OVERLAP_PEEL:
        # Every later tile's zero-accumulator first K step runs inside the
        # previous tile's C epilogue. Run the first tile's here, so each tile
        # enters its K loop at step 1 with the accumulators carried in.
        tl.static_assert(DEFER_BOTTOM and CROSS_TILE_PREFETCH and K // BLOCK_K > NUM_BUFFERS,
                         "OVERLAP_PEEL requires the BK128 staged-output schedule with cross-tile prefetch")
        tl.static_assert(not (EARLY_C_STORES or OUTPUT_REUSE), "OVERLAP_PEEL does not combine with EARLY_C_STORES")
        tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
        a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, 0, 0, 0, BLOCK_M, BLOCK_K, 1, NUM_BUFFERS)
        b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, 0, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS)
        c00, c01, c10, c11, a00, sa00, b00, sb00 = _mxgemm_defer_bottom_k_step(
            tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32), tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32),
            tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32), tl.zeros(
                (BLOCK_M // 2, BLOCK_N // 2), tl.float32), a00, sa00, b00, sb00, a_desc, b_desc, as_desc, bs_desc,
            a_buf, b_buf, as_buf, bs_buf, off_m, off_n, 0, 0, 1, BLOCK_M, BLOCK_N, BLOCK_K, 1, DB, DTYPE_B, NUM_BUFFERS,
            "partial", LOADS, GROUPED_PREFETCH, PREFETCH_B_AFTER_C10, CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M,
            XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT, SPLIT_CLUSTER_BARRIER)

    while tile < total_tiles:
        off_m, off_n = _mxgemm_tile_offsets(tile, num_m, num_n, GROUP_SIZE_M, BLOCK_M, BLOCK_N,
                                            CLUSTER_LAYOUT=CLUSTER_LAYOUT)
        if not OVERLAP_PEEL:
            c00 = tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32)
            c01 = tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32)
            c10 = tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32)
            c11 = tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32)
        if OVERLAP_PEEL:
            # The previous epilogue (or the prologue) acquired stage 1.
            step1_slot = tl.where(phase == NUM_BUFFERS - 1, 0, phase + 1)
        else:
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
        entry_slot = step1_slot if OVERLAP_PEEL else phase
        a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, entry_slot, 0, 0, BLOCK_M, BLOCK_K, 1, NUM_BUFFERS)
        b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, entry_slot, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS)
        steady_end = k_iters - NUM_BUFFERS
        slot = entry_slot if not STATIC_SLOT_UNROLL else phase
        if STATIC_SLOT_UNROLL:
            tl.static_assert(DEFER_BOTTOM and NUM_BUFFERS == 3,
                             "STATIC_SLOT_UNROLL requires the BK128 staged-output schedule with three buffers")
            # The ring slot repeats every NUM_BUFFERS steps. Unrolling by the
            # ring depth keeps each step's slot loop-invariant, so LDS operand
            # addresses leave the loop. A per-step VALU address add would
            # otherwise make each ds_load wait for in-flight WMMAs (va_vdst).
            slot1 = tl.where(slot == NUM_BUFFERS - 1, 0, slot + 1)
            slot2 = tl.where(slot1 == NUM_BUFFERS - 1, 0, slot1 + 1)
            slots = (slot, slot1, slot2)
            STEADY_STEPS: tl.constexpr = K // BLOCK_K - NUM_BUFFERS
            # Peel the first step: its dots see the tile's zero accumulators
            # as constants, which WMMA encodes as an inline C operand. This
            # removes the per-tile zeroing of 512 accumulator VGPRs. Step k
            # uses slots[k % NUM_BUFFERS] throughout.
            PEEL: tl.constexpr = 1 if STEADY_STEPS > 0 else 0
            # With OVERLAP_PEEL, step 0 already ran in the previous epilogue.
            for j in tl.static_range(0 if OVERLAP_PEEL else PEEL):
                c00, c01, c10, c11, a00, sa00, b00, sb00 = _mxgemm_defer_bottom_k_step(
                    c00, c01, c10, c11, a00, sa00, b00, sb00, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                    bs_buf, off_m, off_n, j, slots[j], slots[(j + 1) % NUM_BUFFERS], BLOCK_M, BLOCK_N, BLOCK_K, 1, DB,
                    DTYPE_B, NUM_BUFFERS, "partial", LOADS, GROUPED_PREFETCH, PREFETCH_B_AFTER_C10, CLUSTER_SIZE,
                    CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT,
                    SPLIT_CLUSTER_BARRIER)
            REST: tl.constexpr = STEADY_STEPS - PEEL
            UNROLLED_END: tl.constexpr = PEEL + REST - REST % NUM_BUFFERS
            for i in tl.range(PEEL, UNROLLED_END, NUM_BUFFERS):
                for j in tl.static_range(NUM_BUFFERS):
                    c00, c01, c10, c11, a00, sa00, b00, sb00 = _mxgemm_defer_bottom_k_step(
                        c00, c01, c10, c11, a00, sa00, b00, sb00, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                        as_buf, bs_buf, off_m, off_n, i + j, slots[(PEEL + j) % NUM_BUFFERS],
                        slots[(PEEL + j + 1) % NUM_BUFFERS], BLOCK_M, BLOCK_N, BLOCK_K, 1, DB, DTYPE_B, NUM_BUFFERS,
                        "partial", LOADS, GROUPED_PREFETCH, PREFETCH_B_AFTER_C10, CLUSTER_SIZE, CLUSTER_MULTICAST,
                        GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT, SPLIT_CLUSTER_BARRIER)
            for j in tl.static_range(REST % NUM_BUFFERS):
                c00, c01, c10, c11, a00, sa00, b00, sb00 = _mxgemm_defer_bottom_k_step(
                    c00, c01, c10, c11, a00, sa00, b00, sb00, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                    bs_buf, off_m, off_n, UNROLLED_END + j, slots[(UNROLLED_END + j) % NUM_BUFFERS],
                    slots[(UNROLLED_END + j + 1) % NUM_BUFFERS], BLOCK_M, BLOCK_N, BLOCK_K, 1, DB, DTYPE_B, NUM_BUFFERS,
                    "partial", LOADS, GROUPED_PREFETCH, PREFETCH_B_AFTER_C10, CLUSTER_SIZE, CLUSTER_MULTICAST,
                    GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT, SPLIT_CLUSTER_BARRIER)
            slot = slots[STEADY_STEPS % NUM_BUFFERS]
        # With STATIC_SLOT_UNROLL the steps above consume the whole range.
        for i in tl.range(steady_end if STATIC_SLOT_UNROLL else (1 if OVERLAP_PEEL else 0), steady_end):
            # Carry the physical slot so the hot loop needs only an increment
            # and wrap, including when K does not divide the ring depth.
            tl.assume(slot >= 0)
            tl.assume(slot < NUM_BUFFERS)
            next_slot = tl.where(slot == NUM_BUFFERS - 1, 0, slot + 1)
            if DEFER_BOTTOM:
                c00, c01, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_top(
                    c00, c01, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BLOCK_M, BLOCK_N, BLOCK_K, 1,
                    DB, DTYPE_B, NUM_BUFFERS, GROUPED_PREFETCH,
                    ARRIVE_K=i + NUM_BUFFERS if SPLIT_CLUSTER_BARRIER else None, CLUSTER_SIZE=CLUSTER_SIZE,
                    CLUSTER_BARRIER_INTERVAL=CLUSTER_BARRIER_INTERVAL, PIN_NATIVE=STATIC_SLOT_UNROLL or OVERLAP_PEEL,
                    SCALES_FIRST=STEP_SCALES_FIRST)
                last_b00, last_sb00 = b00, sb00
            else:
                c00, c01, c10, c11, a01, sa01, a11, sa11, b10, sb10, b11, sb11, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_256(
                    c00, c01, c10, c11, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BLOCK_M, BLOCK_N,
                    BLOCK_K, 1, DB, DTYPE_B, NUM_BUFFERS, PIN_NATIVE=False,
                    ARRIVE_K=i + NUM_BUFFERS if SPLIT_CLUSTER_BARRIER else None, CLUSTER_SIZE=CLUSTER_SIZE,
                    CLUSTER_BARRIER_INTERVAL=CLUSTER_BARRIER_INTERVAL)
            _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, off_m, off_n,
                                    i + NUM_BUFFERS, slot, BLOCK_K, 1, DB, NUM_BUFFERS, "partial", CLUSTER_SIZE,
                                    CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL,
                                    CLUSTER_LAYOUT=CLUSTER_LAYOUT, SPLIT_CLUSTER_BARRIER=SPLIT_CLUSTER_BARRIER)
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
            a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 0, 0, BLOCK_M, BLOCK_K, 1, NUM_BUFFERS,
                                             PREFETCH_SCALES=STEP_SCALES_FIRST)
            if not PREFETCH_B_AFTER_C10:
                b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS,
                                                 PREFETCH_SCALES=STEP_SCALES_FIRST)
            if OUTPUT_REUSE:
                if DB == 2:
                    c11 = dot(a10, sa10, "e4m3", b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                    c00 = dot(a01, sa01, "e4m3", b10, sb10, DTYPE_B, c00, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                    c01 = dot(a01, sa01, "e4m3", b11, sb11, DTYPE_B, c01, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                c10 = dot(a11, sa11, "e4m3", b10, sb10, DTYPE_B, c10, tiles_per_warp=[2, 2])
                tlx.amd_sched_barrier()
                c11 = dot(a11, sa11, "e4m3", b11, sb11, DTYPE_B, c11, tiles_per_warp=[2, 2])
                tlx.amd_sched_barrier()
            if DEFER_BOTTOM:
                c10 = dot(a10, sa10, "e4m3", last_b00, last_sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
                if GROUPED_PREFETCH:
                    _mxgemm_schedule_prefetch(103)
                tlx.amd_sched_barrier()
                if PREFETCH_B_AFTER_C10:
                    # C10 is the last user of old B0. Prefetching here permits
                    # register reuse and gives every quadrant one incoming operand.
                    b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS,
                                                     PREFETCH_SCALES=STEP_SCALES_FIRST)
                c11 = dot(a10, sa10, "e4m3", b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                if GROUPED_PREFETCH:
                    _mxgemm_schedule_prefetch(104)
                tlx.amd_sched_barrier()

            slot = next_slot

        # Only construct the next tile's offsets after the steady loop. The
        # final ring rotation can refill slots as soon as their operands are read.
        tlx.amd_sched_barrier()
        next_tile = tile + NUM_PROGRAMS
        has_next = next_tile < total_tiles
        next_m, next_n = _mxgemm_tile_offsets(tl.minimum(next_tile, total_tiles - 1), num_m, num_n, GROUP_SIZE_M,
                                              BLOCK_M, BLOCK_N, CLUSTER_LAYOUT=CLUSTER_LAYOUT)
        tlx.amd_sched_barrier()
        # Keep one tail body for three-stage A8W8. Repeating the entire
        # compute/refill body for each ring slot increases the code footprint
        # of the tile handoff.
        COMPACT_TAIL: tl.constexpr = DEFER_BOTTOM and NUM_BUFFERS == 3 and DTYPE_B == "e4m3"
        if COMPACT_TAIL:
            for j in tl.range(NUM_BUFFERS - 1 if EARLY_C_STORES else NUM_BUFFERS, loop_unroll_factor=1):
                tl.assume(slot >= 0)
                tl.assume(slot < NUM_BUFFERS)
                next_slot = tl.where(slot == NUM_BUFFERS - 1, 0, slot + 1)
                c00, c01, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_top(
                    c00, c01, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BLOCK_M, BLOCK_N, BLOCK_K, 1,
                    DB, DTYPE_B, NUM_BUFFERS, GROUPED_PREFETCH, PIN_NATIVE=STATIC_SLOT_UNROLL or OVERLAP_PEEL)
                last_b00, last_sb00 = b00, sb00
                if CROSS_TILE_PREFETCH and has_next:
                    _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, next_m,
                                            next_n, j, slot, BLOCK_K, 1, DB, NUM_BUFFERS, "partial", CLUSTER_SIZE,
                                            CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL,
                                            CLUSTER_LAYOUT=CLUSTER_LAYOUT)
                if j < NUM_BUFFERS - 1:
                    if CROSS_TILE_PREFETCH and has_next:
                        tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
                    else:
                        # Wait counts must be immediates. Select the count
                        # for the shrinking queue without duplicating math.
                        for pending_stage in tl.static_range(NUM_BUFFERS - 1):
                            if j == pending_stage:
                                tlx.async_amd_descriptor_wait((NUM_BUFFERS - pending_stage - 2) * LOADS)
                    a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 0, 0, BLOCK_M, BLOCK_K, 1, NUM_BUFFERS)
                    b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS)
                c10 = dot(a10, sa10, "e4m3", last_b00, last_sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
                tlx.amd_sched_barrier()
                c11 = dot(a10, sa10, "e4m3", b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                tlx.amd_sched_barrier()
                slot = next_slot
            if EARLY_C_STORES:
                # Peel only the last stage. Its four final quadrant dots
                # will alternate with C stores after all operands are read.
                tl.static_assert(BLOCK_K == 128)
                b01, sb01 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 0, 1, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS)
                a10, sa10 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 0, BLOCK_M, BLOCK_K, 1, NUM_BUFFERS)
                last_b00, last_sb00 = b00, sb00
                tlx.amd_sched_barrier()
                if CROSS_TILE_PREFETCH and has_next:
                    _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, next_m,
                                            next_n, NUM_BUFFERS - 1, slot, BLOCK_K, 1, DB, NUM_BUFFERS, "partial",
                                            CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                                            CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT=CLUSTER_LAYOUT)
        else:
            for j in tl.static_range(NUM_BUFFERS):
                slot = (phase + steady_end + j) % NUM_BUFFERS
                if DEFER_BOTTOM:
                    if EARLY_C_STORES and j == NUM_BUFFERS - 1:
                        b01, sb01 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 0, 1, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS)
                        a10, sa10 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 0, BLOCK_M, BLOCK_K, 1, NUM_BUFFERS)
                        tlx.amd_sched_barrier()
                    else:
                        c00, c01, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_top(
                            c00, c01, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BLOCK_M, BLOCK_N,
                            BLOCK_K, 1, DB, DTYPE_B, NUM_BUFFERS, GROUPED_PREFETCH, PIN_NATIVE=STATIC_SLOT_UNROLL
                            or OVERLAP_PEEL)
                    last_b00, last_sb00 = b00, sb00
                else:
                    c00, c01, c10, c11, a01, sa01, a11, sa11, b10, sb10, b11, sb11, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_256(
                        c00, c01, c10, c11, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BLOCK_M, BLOCK_N,
                        BLOCK_K, 1, DB, DTYPE_B, NUM_BUFFERS, PIN_NATIVE=False)
                if CROSS_TILE_PREFETCH and not OUTPUT_REUSE and has_next:
                    _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, next_m,
                                            next_n, j, slot, BLOCK_K, 1, DB, NUM_BUFFERS, "partial", CLUSTER_SIZE,
                                            CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL,
                                            CLUSTER_LAYOUT=CLUSTER_LAYOUT)
                if j < NUM_BUFFERS - 1:
                    if CROSS_TILE_PREFETCH and not OUTPUT_REUSE and has_next:
                        tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
                    else:
                        tlx.async_amd_descriptor_wait((NUM_BUFFERS - j - 2) * LOADS)
                    next_slot = (phase + steady_end + j + 1) % NUM_BUFFERS
                    a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 0, 0, BLOCK_M, BLOCK_K, 1, NUM_BUFFERS)
                    b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS)
                if OUTPUT_REUSE and not (EARLY_C_STORES and j == NUM_BUFFERS - 1):
                    if DB == 2:
                        c11 = dot(a10, sa10, "e4m3", b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                        tlx.amd_sched_barrier()
                        c00 = dot(a01, sa01, "e4m3", b10, sb10, DTYPE_B, c00, tiles_per_warp=[2, 2])
                        tlx.amd_sched_barrier()
                        c01 = dot(a01, sa01, "e4m3", b11, sb11, DTYPE_B, c01, tiles_per_warp=[2, 2])
                        tlx.amd_sched_barrier()
                    c10 = dot(a11, sa11, "e4m3", b10, sb10, DTYPE_B, c10, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                    c11 = dot(a11, sa11, "e4m3", b11, sb11, DTYPE_B, c11, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                if DEFER_BOTTOM and not (EARLY_C_STORES and j == NUM_BUFFERS - 1):
                    c10 = dot(a10, sa10, "e4m3", last_b00, last_sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                    c11 = dot(a10, sa10, "e4m3", b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
        # Dedicated staging preserves prefetched inputs. BK256 reuses A
        # after its final operand reads, and protects each slot before reload.
        tlx.amd_sched_barrier()
        output_buf = c_buf
        INITIAL_C_SLOTS_FREE: tl.constexpr = not OUTPUT_REUSE
        if EARLY_C_STORES:
            if BLOCK_K == 256:
                # Finish the low-K contribution before the last C11 dot.
                c11 = dot(a10, sa10, "e4m3", b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                tlx.amd_sched_barrier()
                c00 = dot(a01, sa01, "e4m3", b10, sb10, DTYPE_B, c00, tiles_per_warp=[2, 2])
            else:
                c00 = dot(a00, sa00, "e4m3", last_b00, last_sb00, DTYPE_B, c00, tiles_per_warp=[2, 2])
            tlx.amd_sched_barrier()
        if OVERLAP_PEEL:
            # Overlap the next tile's zero-accumulator first K step with
            # this tile's C stores. Once a quadrant's chunks are staged, its
            # registers take that quadrant's next-tile dot, whose WMMAs run
            # during the following slot-reuse waits. The tail prefetched all
            # next-tile stages; stage 0 is the oldest pending load.
            tl.static_assert(C_ROWS == 64 and INITIAL_C_SLOTS_FREE,
                             "OVERLAP_PEEL requires dedicated 64-row C staging slots")
            C_PARTS: tl.constexpr = 2 * ((BLOCK_N // 2) // C_COLS)
            C_CHUNKS: tl.constexpr = 4 * C_PARTS
            # Refill stage 0's slot with next-tile stage NUM_BUFFERS after
            # its last read (end of quadrant 1) and as early as possible.
            # TDM operations complete in order, so the refill may precede
            # only the last C_SLOTS chunks: each of those reuses a slot whose
            # previous chunk is older than the refill. Any earlier, a chunk
            # would wait for the refill's global loads.
            REFILL_CHUNK: tl.constexpr = (C_CHUNKS - C_SLOTS if C_CHUNKS - C_SLOTS > 2 * C_PARTS else 2 * C_PARTS)
            n_phase = (phase + k_iters) % NUM_BUFFERS
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
            if has_next:
                na00, nsa00 = _mxgemm_persistent_a(a_buf, as_buf, n_phase, 0, 0, BLOCK_M, BLOCK_K, 1, NUM_BUFFERS)
                nb00, nsb00 = _mxgemm_persistent_b(b_buf, bs_buf, n_phase, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS)
                for q in tl.static_range(4):
                    acc = c00 if q == 0 else (c01 if q == 1 else (c10 if q == 2 else c11))
                    _mxgemm_overlap_store(c_ptr, acc, off_m + (q // 2) * (BLOCK_M // 2),
                                          off_n + (q % 2) * (BLOCK_N // 2), stride_cm, output_buf, C_SLOTS, q, C_COLS,
                                          REFILL_CHUNK, LOADS, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                          bs_buf, next_m, next_n, n_phase, BLOCK_K, 1, DB, NUM_BUFFERS, "partial",
                                          CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                                          CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT)
                    tlx.amd_sched_barrier()
                    zero = tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32)
                    if q == 0:
                        c00 = _mxgemm_native_dot(na00, nsa00, "e4m3", nb00, nsb00, DTYPE_B, zero, tiles_per_warp=[2, 2])
                        nb01, nsb01 = _mxgemm_persistent_b(b_buf, bs_buf, n_phase, 0, 1, BLOCK_N, BLOCK_K, DB,
                                                           NUM_BUFFERS)
                    elif q == 1:
                        c01 = _mxgemm_native_dot(na00, nsa00, "e4m3", nb01, nsb01, DTYPE_B, zero, tiles_per_warp=[2, 2])
                        na10, nsa10 = _mxgemm_persistent_a(a_buf, as_buf, n_phase, 1, 0, BLOCK_M, BLOCK_K, 1,
                                                           NUM_BUFFERS)
                    elif q == 2:
                        c10 = _mxgemm_native_dot(na10, nsa10, "e4m3", nb00, nsb00, DTYPE_B, zero, tiles_per_warp=[2, 2])
                    else:
                        c11 = _mxgemm_native_dot(na10, nsa10, "e4m3", nb01, nsb01, DTYPE_B, zero, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                # Acquire stage 1: stages 2..NUM_BUFFERS-1, every C chunk,
                # and the refill are younger.
                tlx.async_amd_descriptor_wait((NUM_BUFFERS - 2) * LOADS + C_CHUNKS + LOADS)
            else:
                _mxgemm_persistent_store(c_ptr, c00, off_m, off_n, stride_cm, BLOCK_M // 2, BLOCK_N // 2, 16,
                                         output_buf, C_ROWS, C_SLOTS, 0, C_COLS, INITIAL_C_SLOTS_FREE)
                _mxgemm_persistent_store(c_ptr, c01, off_m, off_n + BLOCK_N // 2, stride_cm, BLOCK_M // 2, BLOCK_N // 2,
                                         16, output_buf, C_ROWS, C_SLOTS, 1, C_COLS, INITIAL_C_SLOTS_FREE)
                _mxgemm_persistent_store(c_ptr, c10, off_m + BLOCK_M // 2, off_n, stride_cm, BLOCK_M // 2, BLOCK_N // 2,
                                         16, output_buf, C_ROWS, C_SLOTS, 2, C_COLS, INITIAL_C_SLOTS_FREE)
                _mxgemm_persistent_store(c_ptr, c11, off_m + BLOCK_M // 2, off_n + BLOCK_N // 2, stride_cm,
                                         BLOCK_M // 2, BLOCK_N // 2, 16, output_buf, C_ROWS, C_SLOTS, 3, C_COLS,
                                         INITIAL_C_SLOTS_FREE)
        else:
            _mxgemm_persistent_store(c_ptr, c00, off_m, off_n, stride_cm, BLOCK_M // 2, BLOCK_N // 2, 16, output_buf,
                                     C_ROWS, C_SLOTS, 0, C_COLS, INITIAL_C_SLOTS_FREE)
        if EARLY_C_STORES:
            # Keep each completed quadrant's transfer ahead of the next
            # quadrant's independent matrix work. Do not drain A/B here.
            tlx.amd_sched_barrier()
            if BLOCK_K == 256:
                c01 = dot(a01, sa01, "e4m3", b11, sb11, DTYPE_B, c01, tiles_per_warp=[2, 2])
            else:
                c01 = dot(a00, sa00, "e4m3", b01, sb01, DTYPE_B, c01, tiles_per_warp=[2, 2])
            tlx.amd_sched_barrier()
        if not OVERLAP_PEEL:
            _mxgemm_persistent_store(c_ptr, c01, off_m, off_n + BLOCK_N // 2, stride_cm, BLOCK_M // 2, BLOCK_N // 2, 16,
                                     output_buf, C_ROWS, C_SLOTS, 1, C_COLS, INITIAL_C_SLOTS_FREE)
        if EARLY_C_STORES:
            tlx.amd_sched_barrier()
            if BLOCK_K == 256:
                c10 = dot(a11, sa11, "e4m3", b10, sb10, DTYPE_B, c10, tiles_per_warp=[2, 2])
            else:
                c10 = dot(a10, sa10, "e4m3", last_b00, last_sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
            tlx.amd_sched_barrier()
        if not OVERLAP_PEEL:
            _mxgemm_persistent_store(c_ptr, c10, off_m + BLOCK_M // 2, off_n, stride_cm, BLOCK_M // 2, BLOCK_N // 2, 16,
                                     output_buf, C_ROWS, C_SLOTS, 2, C_COLS, INITIAL_C_SLOTS_FREE)
            if EARLY_C_STORES:
                tlx.amd_sched_barrier()
                if BLOCK_K == 256:
                    c11 = dot(a11, sa11, "e4m3", b11, sb11, DTYPE_B, c11, tiles_per_warp=[2, 2])
                else:
                    c11 = dot(a10, sa10, "e4m3", b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                tlx.amd_sched_barrier()
            _mxgemm_persistent_store(c_ptr, c11, off_m + BLOCK_M // 2, off_n + BLOCK_N // 2, stride_cm, BLOCK_M // 2,
                                     BLOCK_N // 2, 16, output_buf, C_ROWS, C_SLOTS, 3, C_COLS, INITIAL_C_SLOTS_FREE)
        if OUTPUT_REUSE and not CROSS_TILE_PREFETCH:
            tlx.async_amd_descriptor_wait(0)
        tlx.amd_sched_barrier()
        if OUTPUT_REUSE and CROSS_TILE_PREFETCH:
            # Four quadrant stores use slots 0, 1, 2, 0. Their last uses
            # retire in order 1, 2, 0, which becomes the next input-ring phase.
            phase = (4 - NUM_BUFFERS) % NUM_BUFFERS
        else:
            # Keep the phase of slots recycled by the tail, including when
            # K's iteration count is not divisible by the ring depth.
            phase = (phase + k_iters) % NUM_BUFFERS
        tile = next_tile
        if (OUTPUT_REUSE or not CROSS_TILE_PREFETCH) and has_next:
            for p in tl.static_range(0, NUM_BUFFERS):
                if OUTPUT_REUSE and CROSS_TILE_PREFETCH:
                    # Retire only the C transfer using the slot being refilled.
                    # Younger refills increase the pending count by LOADS - 1;
                    # partial fusion therefore waits for counts 2, 3, then 4.
                    tlx.async_amd_descriptor_wait(NUM_BUFFERS - 1 + p * (LOADS - 1))
                _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, next_m, next_n,
                                        p, (phase + p) % NUM_BUFFERS, BLOCK_K, 1, DB, NUM_BUFFERS, "partial",
                                        CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                                        CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT=CLUSTER_LAYOUT)
    tlx.async_amd_descriptor_wait(0)
    if CLUSTER_SIZE > 1 and CLUSTER_BARRIER_INTERVAL > 0:
        tlx.async_amd_descriptor_wait(0)
        tlx.cluster_barrier()


def _mxgemm_persistent_programs(a, b, a_scale, b_scale, M, N, K, BM, BN, BK, num_buffers, group_m, transpose_b,
                                preshuffle, schedule, split, l2_prefetch, num_programs):
    if schedule != "sliceMNK" or not transpose_b or not preshuffle or split or l2_prefetch != -1:
        raise ValueError("persistent MXFP requires sliceMNK, transposed B, preshuffled scales, "
                         "unsplit descriptors, and L2 prefetch disabled")
    if group_m <= 0:
        raise ValueError("persistent MXFP requires a positive GROUP_SIZE_M")
    if min(M, N, K) <= 0 or M % BM or N % BN or K % BK or K // BK < num_buffers:
        raise ValueError("persistent MXFP requires nonempty full tiles and at least NUM_BUFFERS K tiles")
    if M != a.shape[0] or N != b.shape[0] or tuple(b_scale.shape) != (N // 128, K // 32 * 128):
        raise ValueError("persistent MXFP dimensions must match the data and preshuffled scales")
    if a_scale is None or tuple(a_scale.shape) != (M // 128, K // 32 * 128):
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
                              xcd_chunk, cluster_size, cluster_barrier_interval=1, cluster_layout="linear"):
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
    within a cluster; ``"2x2"``, ``"4x2"``, ``"2x4"``, ``"4x4"``, ``"8x2"``,
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

    ``STATIC_SLOT_UNROLL`` unrolls the three-buffer BK128 steady loop by its
    ring depth, so every step reads a loop-invariant LDS slot; A and B are
    then split over two LDS partitions. ``OVERLAP_PEEL`` runs each next
    tile's first K step inside the current tile's C epilogue. It requires
    BK128, cross-tile prefetch, and more K tiles than buffers.
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
    if not PERSISTENT and (CLUSTER_LAYOUT != "linear" or SPLIT_CLUSTER_BARRIER or EARLY_C_STORES or STATIC_SLOT_UNROLL
                           or OUTPUT_C_SLOTS is not None or OVERLAP_PEEL):
        raise ValueError("CLUSTER_LAYOUT, SPLIT_CLUSTER_BARRIER, EARLY_C_STORES, STATIC_SLOT_UNROLL, "
                         "OUTPUT_C_SLOTS, and OVERLAP_PEEL require PERSISTENT")
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
        if STATIC_SLOT_UNROLL and not (NUM_BUFFERS == 3 and BLOCK_K == 128 and DTYPE_B == "e4m3"):
            raise ValueError("STATIC_SLOT_UNROLL requires E4M3 B, BK128, and three buffers")
        if OVERLAP_PEEL and not (CROSS_TILE_PREFETCH and BLOCK_K == 128 and K // BLOCK_K > NUM_BUFFERS
                                 and not (SPLIT_CLUSTER_BARRIER or EARLY_C_STORES)):
            raise ValueError("OVERLAP_PEEL requires BK128, cross-tile prefetch, more K tiles than buffers, and no "
                             "SPLIT_CLUSTER_BARRIER or EARLY_C_STORES")
        if OUTPUT_C_SLOTS is not None and not (isinstance(OUTPUT_C_SLOTS, int) and OUTPUT_C_SLOTS >= 2
                                               and BLOCK_K == 128):
            raise ValueError("OUTPUT_C_SLOTS requires BK128 and at least two slots")
        NUM_PROGRAMS = _mxgemm_persistent_programs(a, b, a_scale, b_scale, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                                   NUM_BUFFERS, GROUP_SIZE_M, TRANSPOSE_B, SCALE_PRESHUFFLE, SCHEDULE,
                                                   TDM_SPLIT, L2_PREFETCH_DISTANCE, NUM_PROGRAMS)
    _validate_mxfp_scheduling(M, N, NUM_PROGRAMS, persistent=PERSISTENT, block_m=BLOCK_M, block_n=BLOCK_N,
                              group_m=GROUP_SIZE_M, xcd_remap=XCD_REMAP, num_xcds=NUM_XCDS, xcd_chunk=XCD_CHUNK,
                              cluster_size=CLUSTER_SIZE, cluster_barrier_interval=CLUSTER_BARRIER_INTERVAL,
                              cluster_layout=CLUSTER_LAYOUT)
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
                num_warps=4, waves_per_eu=1, ctas_per_cga=(CLUSTER_SIZE, 1, 1))
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
                           "OVERLAP_PEEL")


def matmul(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor, config=None) -> torch.Tensor:
    """C = (A * a_scale) @ (B * b_scale) using a TDM-pipelined MXFP kernel on AMD gfx1250.

    ``a_scale`` / ``b_scale`` must already be pre-shuffled with :func:`pack_scale`.
    When ``config["TRANSPOSE_B"]`` is set, ``b`` is the ``[N, K]`` transposed layout.
    ``PERSISTENT`` selects the persistent kernel; it and the other persistent
    options (``NUM_PROGRAMS``, ``CROSS_TILE_PREFETCH``, ``XCD_REMAP``, cluster
    options, ``STATIC_SLOT_UNROLL``, ``OUTPUT_C_SLOTS``, ``OVERLAP_PEEL``) are
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
                       "OUTPUT_C_SLOTS", "OVERLAP_PEEL")
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
                              xcd_remap=cfg.get("XCD_REMAP", "none"), num_xcds=cfg.get("NUM_XCDS", 8),
                              xcd_chunk=cfg.get("XCD_CHUNK", 2), cluster_size=cfg.get("CLUSTER_SIZE", 1),
                              cluster_barrier_interval=cfg.get("CLUSTER_BARRIER_INTERVAL",
                                                               1), cluster_layout=cfg.get("CLUSTER_LAYOUT", "linear"))
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
    parser.add_argument("--static_slot_unroll", action="store_true",
                        help="unroll the three-buffer BK128 steady loop by its ring depth with loop-invariant slots")
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

    a_scale_input = pack_scale(a_scale) if args.scale_preshuffled and a_scale is not None else a_scale
    b_scale_input = pack_scale(b_scale) if args.scale_preshuffled else b_scale
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
        OUT_DTYPE=args.out_dtype,
        TILES_PER_WARP=tuple(args.tiles_per_warp),
    )
