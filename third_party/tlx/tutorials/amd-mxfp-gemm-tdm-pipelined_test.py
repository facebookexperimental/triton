"""
MXFP TDM-pipelined GEMM for AMD gfx1250 using TLX.

This mirrors the structure of the Gluon MXFP GEMM example: descriptor-backed
loads for A/B and scales, optional A scales, L2 prefetch hints, and baseline or
sliced K/N/MNK compute schedules.
"""
import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.constexpr_function
def _operand_shared_layout(block_shape):
    block_k = block_shape[1]
    pad_interval = 256 if block_k <= 256 else block_k
    return tlx.padded_shared_layout_encoding.with_identity_for([[pad_interval, 16]], block_shape, [1, 0])


@triton.constexpr_function
def _scale_shared_layout(block_shape):
    return tlx.padded_shared_layout_encoding.with_identity_for([[256, 8]], block_shape, [1, 0])


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
def _mxgemm_issue_load_a_scale(
    a_scale_desc,
    a_scale_buf,
    load_idx,
    pred,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
):
    if WITH_A_SCALE:
        slot = load_idx % NUM_BUFFERS
        if SCALE_PRESHUFFLE:
            scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
        else:
            scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]
        tlx.async_amd_descriptor_load(a_scale_desc, tlx.local_view(a_scale_buf, slot), scale_offsets, pred=pred)


@triton.jit
def _mxgemm_issue_load_b_scale(
    b_scale_desc,
    b_scale_buf,
    load_idx,
    pred,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
):
    slot = load_idx % NUM_BUFFERS
    if SCALE_PRESHUFFLE:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
    else:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]
    tlx.async_amd_descriptor_load(b_scale_desc, tlx.local_view(b_scale_buf, slot), scale_offsets, pred=pred)


@triton.jit
def _mxgemm_issue_load_a_data(
    a_desc,
    a_buf,
    load_idx,
    pred,
    BLOCK_K_PACKED_A: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    slot = load_idx % NUM_BUFFERS
    tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, slot), [0, load_idx * BLOCK_K_PACKED_A], pred=pred)


@triton.jit
def _mxgemm_issue_load_b_data(
    b_desc,
    b_buf,
    load_idx,
    pred,
    BLOCK_K_PACKED_B: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    slot = load_idx % NUM_BUFFERS
    if TRANSPOSE_B:
        b_offsets = [0, load_idx * BLOCK_K_PACKED_B]
    else:
        b_offsets = [load_idx * BLOCK_K_PACKED_B, 0]
    tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, slot), b_offsets, pred=pred)


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
):
    slot = load_idx % NUM_BUFFERS
    if TRANSPOSE_B:
        b_offsets = [0, load_idx * BLOCK_K_PACKED_B]
    else:
        b_offsets = [load_idx * BLOCK_K_PACKED_B, 0]
    if SCALE_PRESHUFFLE:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
    else:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]

    if TDM_FUSION == "4way":
        tl.static_assert(WITH_A_SCALE, "4-way TDM fusion requires WITH_A_SCALE")
        tlx.async_amd_descriptor_load_group(
            [a_desc, b_desc, a_scale_desc, b_scale_desc],
            [
                tlx.local_view(a_buf, slot),
                tlx.local_view(b_buf, slot),
                tlx.local_view(a_scale_buf, slot),
                tlx.local_view(b_scale_buf, slot),
            ],
            [
                [0, load_idx * BLOCK_K_PACKED_A],
                b_offsets,
                scale_offsets,
                scale_offsets,
            ],
            [0b0001, 0b0010, 0b0100, 0b1000],
            preds=[pred, pred, pred, pred],
        )
    elif TDM_FUSION == "2way":
        tlx.async_amd_descriptor_load_group(
            [a_desc, b_desc],
            [tlx.local_view(a_buf, slot), tlx.local_view(b_buf, slot)],
            [[0, load_idx * BLOCK_K_PACKED_A], b_offsets],
            [0b0011, 0b1100],
            preds=[pred, pred],
        )
        if WITH_A_SCALE:
            tlx.async_amd_descriptor_load_group(
                [a_scale_desc, b_scale_desc],
                [tlx.local_view(a_scale_buf, slot),
                 tlx.local_view(b_scale_buf, slot)],
                [scale_offsets, scale_offsets],
                [0b0011, 0b1100],
                preds=[pred, pred],
            )
        else:
            _mxgemm_issue_load_b_scale(b_scale_desc, b_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED,
                                       NUM_BUFFERS, SCALE_PRESHUFFLE)
    elif TDM_FUSION == "partial":
        tl.static_assert(WITH_A_SCALE, "partial TDM fusion requires WITH_A_SCALE")
        tlx.async_amd_descriptor_load_group(
            [a_desc, b_desc],
            [tlx.local_view(a_buf, slot), tlx.local_view(b_buf, slot)],
            [[0, load_idx * BLOCK_K_PACKED_A], b_offsets],
            [0b0101, 0b1010],
            preds=[pred, pred],
        )
        tlx.async_amd_descriptor_load_group(
            [a_scale_desc, b_scale_desc],
            [tlx.local_view(a_scale_buf, slot), tlx.local_view(b_scale_buf, slot)],
            [scale_offsets, scale_offsets],
            [0b0101, 0b1010],
            preds=[pred, pred],
        )
    else:
        tl.static_assert(TDM_FUSION == "none", "TDM_FUSION must be one of: none, 2way, 4way, partial")
        _mxgemm_issue_load_a_scale(a_scale_desc, a_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS,
                                   SCALE_PRESHUFFLE, WITH_A_SCALE)
        _mxgemm_issue_load_b_scale(b_scale_desc, b_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS,
                                   SCALE_PRESHUFFLE)
        _mxgemm_issue_load_a_data(a_desc, a_buf, load_idx, pred, BLOCK_K_PACKED_A, NUM_BUFFERS)
        _mxgemm_issue_load_b_data(b_desc, b_buf, load_idx, pred, BLOCK_K_PACKED_B, NUM_BUFFERS, TRANSPOSE_B)
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
        tlx.async_amd_descriptor_load_group(
            [a0_desc, b0_desc],
            [tlx.local_view(a0_buf, slot), tlx.local_view(b0_buf, slot)],
            [a_offsets, b_offsets],
            [0b0101, 0b1010],
            preds=[pred, pred],
        )
        tlx.async_amd_descriptor_load_group(
            [a1_desc, b1_desc],
            [tlx.local_view(a1_buf, slot), tlx.local_view(b1_buf, slot)],
            [a_offsets, b_offsets],
            [0b0101, 0b1010],
            preds=[pred, pred],
        )
        tlx.async_amd_descriptor_load_group(
            [a_scale_desc, b_scale_desc],
            [tlx.local_view(a_scale_buf, slot), tlx.local_view(b_scale_buf, slot)],
            [scale_offsets, scale_offsets],
            [0b0101, 0b1010],
            preds=[pred, pred],
        )
    else:
        tl.static_assert(TDM_FUSION == "none", "TDM_SPLIT supports TDM_FUSION values: none, partial")
        _mxgemm_issue_load_a_scale(a_scale_desc, a_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS,
                                   SCALE_PRESHUFFLE, WITH_A_SCALE)
        _mxgemm_issue_load_b_scale(b_scale_desc, b_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS,
                                   SCALE_PRESHUFFLE)
        tlx.async_amd_descriptor_load(a0_desc, tlx.local_view(a0_buf, slot), a_offsets, pred=pred)
        tlx.async_amd_descriptor_load(b0_desc, tlx.local_view(b0_buf, slot), b_offsets, pred=pred)
        tlx.async_amd_descriptor_load(a1_desc, tlx.local_view(a1_buf, slot), a_offsets, pred=pred)
        tlx.async_amd_descriptor_load(b1_desc, tlx.local_view(b1_buf, slot), b_offsets, pred=pred)
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
):
    slot = wmma_idx % NUM_BUFFERS
    subtile_m: tl.constexpr = BLOCK_M // NUM_SUBTILES_M
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_m: tl.constexpr = subtile_start_idx_m * subtile_m
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    a_view = tlx.local_view(a_buf, slot)
    if NUM_SUBTILES_M != 1 or NUM_SUBTILES_K != 1:
        a_view = tlx.local_slice(a_view, [subtile_start_m, subtile_start_k // DIV_FACTOR_A],
                                 [subtile_m, subtile_k // DIV_FACTOR_A])
    a = tlx.local_load(a_view)

    if WITH_A_SCALE:
        if SCALE_PRESHUFFLE:
            scale_a_view = tlx.local_reshape(
                tlx.local_view(a_scale_buf, slot),
                [BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 128 // 4, 4, SCALE_KWIDTH],
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
                [BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 128 // 4, 4, SCALE_KWIDTH],
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
):
    slot = wmma_idx % NUM_BUFFERS
    subtile_n: tl.constexpr = BLOCK_N // NUM_SUBTILES_N
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_n: tl.constexpr = subtile_start_idx_n * subtile_n
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    b_view = tlx.local_view(b_buf, slot)
    if TRANSPOSE_B:
        if NUM_SUBTILES_N != 1 or NUM_SUBTILES_K != 1:
            b_view = tlx.local_slice(b_view, [subtile_start_n, subtile_start_k // DIV_FACTOR_B],
                                     [subtile_n, subtile_k // DIV_FACTOR_B])
        b = tlx.local_load(tlx.local_trans(b_view))
    else:
        if NUM_SUBTILES_N != 1 or NUM_SUBTILES_K != 1:
            b_view = tlx.local_slice(b_view, [subtile_start_k // DIV_FACTOR_B, subtile_start_n],
                                     [subtile_k // DIV_FACTOR_B, subtile_n])
        b = tlx.local_load(b_view)

    if SCALE_PRESHUFFLE:
        scale_b_view = tlx.local_reshape(
            tlx.local_view(b_scale_buf, slot),
            [BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 128 // 4, 4, SCALE_KWIDTH],
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
            [BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 128 // 4, 4, SCALE_KWIDTH],
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
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
    SCHEDULE: tl.constexpr,
    TDM_FUSION: tl.constexpr = "none",
    L2_PREFETCH_DISTANCE: tl.constexpr = -1,
    TDM_SPLIT: tl.constexpr = False,
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
        NUM_SUBTILES_K: tl.constexpr = 2
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
        c00 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        c01 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        c10 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        c11 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        for i in tl.range(0, K_ITERS):
            c00 = tlx.dot_scaled(a00, scale_a00, DTYPE_A, b00, scale_b00, DTYPE_B, c00, tiles_per_warp=[2, 2])
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
            c01 = tlx.dot_scaled(a00, scale_a00, DTYPE_A, b01, scale_b01, DTYPE_B, c01, tiles_per_warp=[2, 2])
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
            c10 = tlx.dot_scaled(a10, scale_a10, DTYPE_A, b00, scale_b00, DTYPE_B, c10, tiles_per_warp=[2, 2])
            if TDM_SPLIT:
                b10, scale_b10 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 1, 0, BLOCK_N,
                                                              BLOCK_K, DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                              SCALE_PRESHUFFLE)
            else:
                b10, scale_b10 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 1, 0, BLOCK_N, BLOCK_K,
                                                        DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                        TRANSPOSE_B, SCALE_PRESHUFFLE)
            c11 = tlx.dot_scaled(a10, scale_a10, DTYPE_A, b01, scale_b01, DTYPE_B, c11, tiles_per_warp=[2, 2])
            if TDM_SPLIT:
                a01, scale_a01 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 0, 1, BLOCK_M,
                                                              BLOCK_K, DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                              WITH_A_SCALE)
            else:
                a01, scale_a01 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, 1, BLOCK_M, BLOCK_K,
                                                        DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K,
                                                        SCALE_PRESHUFFLE, WITH_A_SCALE)
            c00 = tlx.dot_scaled(a01, scale_a01, DTYPE_A, b10, scale_b10, DTYPE_B, c00, tiles_per_warp=[2, 2])
            if TDM_SPLIT:
                b11, scale_b11 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 1, 1, BLOCK_N,
                                                              BLOCK_K, DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                              SCALE_PRESHUFFLE)
            else:
                b11, scale_b11 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 1, 1, BLOCK_N, BLOCK_K,
                                                        DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                        TRANSPOSE_B, SCALE_PRESHUFFLE)
            c01 = tlx.dot_scaled(a01, scale_a01, DTYPE_A, b11, scale_b11, DTYPE_B, c01, tiles_per_warp=[2, 2])
            if TDM_SPLIT:
                a11, scale_a11 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 1, 1, BLOCK_M,
                                                              BLOCK_K, DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                              WITH_A_SCALE)
            else:
                a11, scale_a11 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 1, 1, BLOCK_M, BLOCK_K,
                                                        DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K,
                                                        SCALE_PRESHUFFLE, WITH_A_SCALE)
            wmma_idx += 1
            c10 = tlx.dot_scaled(a11, scale_a11, DTYPE_A, b10, scale_b10, DTYPE_B, c10, tiles_per_warp=[2, 2])
            c11 = tlx.dot_scaled(a11, scale_a11, DTYPE_A, b11, scale_b11, DTYPE_B, c11, tiles_per_warp=[2, 2])
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
        acc_top = tl.join(c00, c01).permute(0, 2, 1).reshape((SUBTILE_M, BLOCK_N))
        acc_bot = tl.join(c10, c11).permute(0, 2, 1).reshape((SUBTILE_M, BLOCK_N))
        acc = tl.join(acc_top, acc_bot).permute(2, 0, 1).reshape((BLOCK_M, BLOCK_N))
        c_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_ptr), 1)
        c_view = tlx.local_view(c_buf, 0)
        tlx.local_store(c_view, acc.to(tlx.dtype_of(c_ptr)))
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
                c0 = tlx.dot_scaled(a, scale_a, DTYPE_A, b0, scale_b0, DTYPE_B, c0, tiles_per_warp=[2, 2])
                c1 = tlx.dot_scaled(a, scale_a, DTYPE_A, b1, scale_b1, DTYPE_B, c1, tiles_per_warp=[2, 2])
            wmma_idx += 1
        acc = tl.join(c0, c1).permute(0, 2, 1).reshape((BLOCK_M, BLOCK_N))
        acc = tlx.require_amd_wmma_layout(acc, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                          instr_shape=STORE_INSTR_SHAPE)
        c_offsets_store = tlx.require_amd_wmma_layout(c_offsets, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                      instr_shape=STORE_INSTR_SHAPE)
        c_mask_store = tlx.require_amd_wmma_layout(c_mask, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                   instr_shape=STORE_INSTR_SHAPE)
        tlx.buffer_store(acc, c_ptr, c_offsets_store, mask=c_mask_store)
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
                acc = tlx.dot_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, acc, tiles_per_warp=[2, 2])
            wmma_idx += 1
        acc = tlx.require_amd_wmma_layout(acc, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                          instr_shape=STORE_INSTR_SHAPE)
        c_offsets_store = tlx.require_amd_wmma_layout(c_offsets, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                      instr_shape=STORE_INSTR_SHAPE)
        c_mask_store = tlx.require_amd_wmma_layout(c_mask, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                   instr_shape=STORE_INSTR_SHAPE)
        tlx.buffer_store(acc, c_ptr, c_offsets_store, mask=c_mask_store)


DTYPE_TO_TRITON = {
    "float8_e5m2": "e5m2",
    "float8_e4m3": "e4m3",
    "float4": "e2m1",
}


def _mxfp_gemm_tflops(ms: float, M: int, N: int, K: int) -> float:
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def _init_data(dtype: str, rows: int, cols: int):
    if dtype == "float4":
        return MXFP4Tensor(size=(rows, cols)).random()
    if dtype == "float8_e5m2":
        return torch.randint(20, 40, (rows, cols), dtype=torch.uint8).view(torch.float8_e5m2)
    if dtype == "float8_e4m3":
        return torch.randint(20, 40, (rows, cols), dtype=torch.uint8).view(torch.float8_e4m3fn)
    raise ValueError(f"unsupported dtype: {dtype}")


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
    BENCHMARK: str | None = None,
    BENCHMARK_NUM_ITERS: int = 32,
) -> torch.Tensor:
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
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    a_scale_arg = a_scale if WITH_A_SCALE else b_scale
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    def run_kernel():
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
            num_warps=4,
            waves_per_eu=1,
        )

    if BENCHMARK == "graph":
        ms = triton.testing.do_bench_cudagraph(run_kernel, rep=BENCHMARK_NUM_ITERS)
        print(f"execution time: {ms} ms, {_mxfp_gemm_tflops(ms, M, N, K):.2f} TFLOPS")
    elif BENCHMARK == "eager":
        ms = triton.testing.do_bench(run_kernel, warmup=30, rep=BENCHMARK_NUM_ITERS)
        print(f"execution time: {ms} ms, {_mxfp_gemm_tflops(ms, M, N, K):.2f} TFLOPS")
    else:
        run_kernel()
    return c


@pytest.mark.parametrize("TDM_FUSION", ["none", "2way", "4way", "partial"])
def test_mxgemm_tdm_pipelined_compiles_gfx1250(TDM_FUSION):
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    src = ASTSource(
        fn=mxgemm_tdm_pipelined_kernel,
        signature={
            "a_ptr": "*fp8e5",
            "b_ptr": "*fp8e5",
            "c_ptr": "*fp32",
            "a_scale": "*i8",
            "b_scale": "*i8",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i64",
            "stride_ak": "i64",
            "stride_bk": "i64",
            "stride_bn": "i64",
            "stride_cm": "i64",
            "stride_cn": "i64",
            "stride_scale": "i64",
        },
        constexprs={
            "DTYPE_A": "e5m2",
            "DTYPE_B": "e5m2",
            "SCALE_BLOCK": 32,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 8,
            "TRANSPOSE_B": True,
            "NUM_BUFFERS": 2,
            "SCALE_PRESHUFFLE": True,
            "WITH_A_SCALE": True,
            "SCHEDULE": "baseline",
            "TDM_FUSION": TDM_FUSION,
            "L2_PREFETCH_DISTANCE": 2,
            "TDM_SPLIT": False,
        },
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    if TDM_FUSION == "none":
        assert "amdg.async_tdm_copy_global_to_local" in ttgir
        assert "amdg.async_tdm_group_copy_global_to_local" not in ttgir
    else:
        assert "amdg.async_tdm_group_copy_global_to_local" in ttgir
        assert "amdg.async_tdm_copy_global_to_local" not in ttgir
    if TDM_FUSION == "2way":
        assert "warp_masks = array<i32: 3, 12>" in ttgir
    elif TDM_FUSION == "4way":
        assert "warp_masks = array<i32: 1, 2, 4, 8>" in ttgir
    elif TDM_FUSION == "partial":
        assert "warp_masks = array<i32: 5, 10>" in ttgir
    assert "amdg.tdm_prefetch" in ttgir
    assert "tt.dot_scaled" in ttgir
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "wmma" in amdgcn


def test_mxgemm_tdm_split_compiles_gfx1250():
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    src = ASTSource(
        fn=mxgemm_tdm_pipelined_kernel,
        signature={
            "a_ptr": "*fp8e4nv",
            "b_ptr": "*u8",
            "c_ptr": "*fp32",
            "a_scale": "*i8",
            "b_scale": "*i8",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i64",
            "stride_ak": "i64",
            "stride_bk": "i64",
            "stride_bn": "i64",
            "stride_cm": "i64",
            "stride_cn": "i64",
            "stride_scale": "i64",
        },
        constexprs={
            "DTYPE_A": "e4m3",
            "DTYPE_B": "e2m1",
            "SCALE_BLOCK": 32,
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 256,
            "GROUP_SIZE_M": 8,
            "TRANSPOSE_B": True,
            "NUM_BUFFERS": 3,
            "SCALE_PRESHUFFLE": True,
            "WITH_A_SCALE": True,
            "SCHEDULE": "sliceMNK",
            "TDM_FUSION": "partial",
            "L2_PREFETCH_DISTANCE": -1,
            "TDM_SPLIT": True,
        },
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    assert "amdg.async_tdm_group_copy_global_to_local" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir
    assert "warp_masks = array<i32: 5, 10>" in ttgir
    assert "tt.dot_scaled" in ttgir
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "wmma" in amdgcn


@pytest.mark.skipif(not is_gfx1250_available(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("SEED", [0, 7])
@pytest.mark.parametrize(
    "M,N,K,SCHEDULE,DTYPE_A,DTYPE_B,BLOCK_M,BLOCK_N,BLOCK_K,NUM_BUFFERS,SCALE_PRESHUFFLE,WITH_A_SCALE,"
    "TDM_FUSION,L2_PREFETCH_DISTANCE,TDM_SPLIT",
    [
        (256, 256, 512, "baseline", "float8_e5m2", "float8_e5m2", 128, 128, 128, 2, True, True, "none", -1, False),
        (256, 256, 512, "baseline", "float8_e4m3", "float8_e5m2", 128, 128, 128, 2, False, False, "none", 2, False),
        (256, 256, 512, "sliceK", "float8_e4m3", "float8_e5m2", 128, 128, 256, 2, True, True, "2way", 2, False),
        (256, 512, 512, "sliceNK", "float8_e5m2", "float4", 256, 256, 256, 2, True, True, "2way", 2, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "none", 2, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "2way", 2, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "4way", 2, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "partial", 2, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float4", 256, 256, 256, 2, True, True, "partial", -1, True),
        (256, 512, 512, "sliceMNK", "float8_e4m3", "float8_e5m2", 128, 256, 256, 2, True, True, "4way", 2, False),
        (256, 256, 512, "baseline", "float4", "float4", 128, 128, 128, 2, True, True, "4way", 2, False),
        (384, 384, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "4way", 2, False),
        (384, 512, 768, "sliceMNK", "float8_e5m2", "float8_e4m3", 256, 256, 256, 2, True, True, "2way", 2, False),
    ],
)
def test_mxgemm_tdm_pipelined_gfx1250(TRANSPOSE_B, SEED, M, N, K, SCHEDULE, DTYPE_A, DTYPE_B, BLOCK_M, BLOCK_N, BLOCK_K,
                                      NUM_BUFFERS, SCALE_PRESHUFFLE, WITH_A_SCALE, TDM_FUSION, L2_PREFETCH_DISTANCE,
                                      TDM_SPLIT):
    torch.manual_seed(SEED)
    a = _init_data(DTYPE_A, M, K)
    b = _init_data(DTYPE_B, K, N)
    if WITH_A_SCALE:
        a_scale = MXScaleTensor(size=(M, triton.cdiv(K, 32))).random(high=32.0).data
    else:
        a_scale = None
    b_scale = MXScaleTensor(size=(N, triton.cdiv(K, 32))).random(high=32.0).data
    ref = torch_gemm_mxfp(a, b, a_scale, b_scale, 32, M, N, K)

    a_scale_input = pack_scale(a_scale) if SCALE_PRESHUFFLE else a_scale
    b_scale_input = pack_scale(b_scale) if SCALE_PRESHUFFLE else b_scale
    if DTYPE_A == "float4":
        a = a.to_packed_tensor(dim=1)
    if DTYPE_B == "float4":
        b = b.to_packed_tensor(dim=0)

    a_d = a.data.contiguous().cuda() if DTYPE_A == "float4" else a.contiguous().cuda()
    if DTYPE_B == "float4":
        b_d = b.data.T.contiguous().cuda() if TRANSPOSE_B else b.data.contiguous().cuda()
    else:
        b_d = b.T.contiguous().cuda() if TRANSPOSE_B else b.contiguous().cuda()
    if a_scale_input is not None:
        a_scale_d = a_scale_input.cuda()
    else:
        a_scale_d = None
    out = mxgemm_tdm_pipelined(a_d, b_d, a_scale_d, b_scale_input.cuda(), BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B,
                               NUM_BUFFERS, DTYPE_TO_TRITON[DTYPE_A], DTYPE_TO_TRITON[DTYPE_B], SCALE_PRESHUFFLE,
                               WITH_A_SCALE, SCHEDULE, L2_PREFETCH_DISTANCE, M, N, K, TDM_FUSION, TDM_SPLIT)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=2e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the TLX AMD MXFP TDM GEMM tutorial kernel")
    parser.add_argument("-M", type=int, default=2048, help="problem M size")
    parser.add_argument("-N", type=int, default=1024, help="problem N size")
    parser.add_argument("-K", type=int, default=8192, help="problem K size")
    parser.add_argument("-BM", type=int, default=256, help="BLOCK_M")
    parser.add_argument("-BN", type=int, default=256, help="BLOCK_N")
    parser.add_argument("-BK", type=int, default=256, help="BLOCK_K")
    parser.add_argument("--num_warps", type=int, default=4, choices=[4], help="kernel num_warps")
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
    parser.add_argument("--l2_prefetch_distance", type=int, default=-1,
                        help="Prefetch distance in K iterations; -1 disables L2 prefetch")
    parser.add_argument("--benchmark_mode", choices=["eager", "graph", "none"], default="eager")
    parser.add_argument("--benchmark_num_iters", type=int, default=32)
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

    benchmark = None if args.benchmark_mode == "none" else args.benchmark_mode
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
        benchmark,
        args.benchmark_num_iters,
    )
