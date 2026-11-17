"""gfx1250 MXFP8 GEMM backend: persistent TDM pipeline with multicast clusters.

The tutorial ``tutorials/amd_mxfp_gemm_gfx1250`` imports this kernel and its
helpers for its experiments, launcher options, and benchmarks.
Each persistent workgroup computes 256x256 output tiles with four waves. A
ring of LDS stages, filled by fused tensor data mover (TDM) loads, carries A,
B and their scales across K and into the next tile: the final K steps of one
tile prefetch the next tile's first stages. C is staged through dedicated LDS
slots and written with TDM stores. When the tile grid allows it, 4x4 clusters
of 16 workgroups share A and B through TDM multicast.

Up to K = 16384 the four-stage ring runs a statically unrolled loop with one
split workgroup barrier per K step, and each next tile's first K step
overlaps the current tile's C stores. Larger K uses the dynamic four-stage
loop. Two or three K tiles use a matching shorter ring.

Scales use the cuBLAS 128x4 blocked byte order (``block128``): blocked inputs
are consumed in place and natural ``[rows, K // 32]`` scales are packed on
every call.
"""

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..._catalog import InvalidInput


@triton.constexpr_function
def operand_shared_layout(block_shape):
    block_k = block_shape[1]
    pad_interval = 256 if block_k <= 256 else block_k
    return tlx.padded_shared_layout_encoding.with_identity_for([[pad_interval, 16]], block_shape, [1, 0])


@triton.constexpr_function
def scale_shared_layout(block_shape, wave256=False):
    if wave256:
        return tlx.padded_shared_layout_encoding.with_identity_for([[1024, 16]], block_shape, [1, 0])
    return tlx.padded_shared_layout_encoding.with_identity_for([[256, 8]], block_shape, [1, 0])


@tl.core.builtin
def _mxgemm_scale_cols(x, _semantic=None):
    # Columns of a preshuffled scale descriptor block or LDS buffer. With
    # BLOCK_K_SCALE scales per row and K step, "block128" rows hold
    # BLOCK_K_SCALE * 128 bytes and "wave256" rows BLOCK_K_SCALE * 256.
    shape = x.block_shape if hasattr(x, "block_shape") else x.type.shape
    return tl.constexpr(tl.core._unwrap_if_constexpr(shape[-1]))


@triton.jit
def _mxgemm_w256_scale_view(buf, slot):
    # Logical [256, 4] view of one wave256 scale slot: row = blk*128 + hi*64 + w*32 + l.
    view = tlx.local_reshape(tlx.local_view(buf, slot), [2, 32, 2, 2, 4])
    view = tlx.local_trans(view, (2, 3, 0, 1, 4))
    return tlx.local_reshape(view, [256, 4])


@triton.jit
def _mxgemm_scale_quads(s, IS_B: tl.constexpr):
    # Split a [256, 4] wave256 scale tensor into its two 128-row quadrants in
    # registers (the block bit is a per-lane register bit).
    t = tl.permute(tl.reshape(s, (2, 128, 4)), (1, 2, 0))
    q0, q1 = tl.split(t)
    if IS_B:
        lay: tl.constexpr = tlx.layout(shape=((32, 2, 2), (4, 2)), stride=((4, 128, 0), (1, 256)))
    else:
        lay: tl.constexpr = tlx.layout(shape=((32, 2, 2), (4, 2)), stride=((4, 0, 128), (1, 256)))
    return tlx.require_layout(q0, lay), tlx.require_layout(q1, lay)


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


def pack_scale(x: torch.Tensor, preshuffle_factor: int = 128, layout: str = "block128") -> torch.Tensor:
    if x is None:
        return x
    if layout not in ("block128", "wave256"):
        raise ValueError(f"unknown scale layout {layout!r}")
    if layout == "wave256":
        # Per 256 rows and 4 K scales, 1024 bytes ordered
        # [row bit 5][row % 32][row bit 7][row bit 6][4 K]: each lane's 16
        # bytes are the four scale dwords its wave needs for both 128-row
        # quadrants of a 256-row tile, one ds_load_b128.
        non_k, k_scale = x.shape
        if preshuffle_factor != 128 or non_k % 256 or k_scale % 4:
            raise ValueError("wave256 scales need 256-row and 4-scale multiples")
        # [tiles, blk, hi, w, l, kg, kw] -> [tiles, kg, w, l, blk, hi, kw]
        x = x.reshape(non_k // 256, 2, 2, 2, 32, k_scale // 4, 4).permute(0, 5, 3, 4, 1, 2, 6).contiguous()
        return x.view(non_k // 256, k_scale * 256)
    non_k, k_scale = x.shape
    scale_kwidth = 4 if k_scale >= 4 else k_scale
    num_chunk_m = non_k // preshuffle_factor
    num_chunk_k = k_scale // scale_kwidth
    x = x.view(num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, scale_kwidth)
    x = x.permute(0, 3, 2, 1, 4).contiguous()
    return x.view(non_k // preshuffle_factor, k_scale * preshuffle_factor)


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
def mxgemm_issue_loads(
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
            # That holds only while the cluster stays within one M group,
            # and a TDM multicast reaches at most five CTAs.
            tl.static_assert(CLUSTER_SIZE <= GROUP_SIZE_M and CLUSTER_SIZE <= 4,
                             "linear multicast clusters require CLUSTER_SIZE <= min(GROUP_SIZE_M, 4)")
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
def mxgemm_load_a_operand(
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
    RELAXED: tl.constexpr = False,
    LOAD: tl.constexpr = 3,
    BOTH: tl.constexpr = False,
):
    # LOAD bit 0 reads the payload and bit 1 the scale; a skipped part returns
    # the other part as a placeholder. BOTH reads a wave256 scale slot whole
    # ([256, 4], both 128-row quadrants) instead of this subtile's quadrant.
    slot = wmma_idx % NUM_BUFFERS
    subtile_m: tl.constexpr = BLOCK_M // NUM_SUBTILES_M
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_m: tl.constexpr = subtile_start_idx_m * subtile_m
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    a_view = _mxgemm_flat_view(tlx.local_view(a_buf, slot))
    if NUM_SUBTILES_M != 1 or NUM_SUBTILES_K != 1:
        a_view = tlx.local_slice(a_view, [subtile_start_m, subtile_start_k // DIV_FACTOR_A],
                                 [subtile_m, subtile_k // DIV_FACTOR_A])
    a = None
    scale_a = None
    if not SCALES_FIRST and LOAD & 1:
        a = tlx.local_load(a_view, relaxed=RELAXED)

    if (LOAD & 2) == 0:
        pass
    elif WITH_A_SCALE:
        if SCALE_PRESHUFFLE and _mxgemm_scale_cols(a_scale_buf) == BLOCK_K_SCALE * 256:
            tl.static_assert(BLOCK_M == 256 and BLOCK_K_SCALE == 4)
            scale_a_view = _mxgemm_w256_scale_view(a_scale_buf, slot)
        elif SCALE_PRESHUFFLE:
            scale_a_view = tlx.local_reshape(
                tlx.local_view(a_scale_buf, slot),
                [BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 32, 4, SCALE_KWIDTH],
            )
            scale_a_view = tlx.local_trans(scale_a_view, (0, 3, 2, 1, 4))
            scale_a_view = tlx.local_reshape(scale_a_view, [BLOCK_M, BLOCK_K_SCALE])
        else:
            scale_a_view = tlx.local_view(a_scale_buf, slot)
        if BOTH:
            pass
        elif NUM_SUBTILES_M != 1 or NUM_SUBTILES_K != 1:
            scale_a_view = tlx.local_slice(scale_a_view, [subtile_start_m, subtile_start_k // SCALE_BLOCK],
                                           [subtile_m, subtile_k // SCALE_BLOCK])
        elif not SCALE_PRESHUFFLE:
            scale_a_view = tlx.local_slice(scale_a_view, [0, 0], [BLOCK_M, BLOCK_K_SCALE])
        scale_a = tlx.local_load(scale_a_view, relaxed=RELAXED)
    else:
        scale_a = tl.full((subtile_m, subtile_k // SCALE_BLOCK), 127, dtype=tl.uint8)

    if SCALES_FIRST and LOAD & 1:
        a = tlx.local_load(a_view, relaxed=RELAXED)
    if (LOAD & 2) == 0:
        scale_a = a
    if (LOAD & 1) == 0:
        a = scale_a
    return a, scale_a


@triton.jit
def mxgemm_load_b_operand(
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
    RELAXED: tl.constexpr = False,
    LOAD: tl.constexpr = 3,
    BOTH: tl.constexpr = False,
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
    b = None
    scale_b = None
    if not SCALES_FIRST and LOAD & 1:
        b = tlx.local_load(b_view, relaxed=RELAXED)

    if SCALE_PRESHUFFLE and _mxgemm_scale_cols(b_scale_buf) == BLOCK_K_SCALE * 256:
        tl.static_assert(BLOCK_N == 256 and BLOCK_K_SCALE == 4)
        scale_b_view = _mxgemm_w256_scale_view(b_scale_buf, slot)
    elif SCALE_PRESHUFFLE:
        scale_b_view = tlx.local_reshape(
            tlx.local_view(b_scale_buf, slot),
            [BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 32, 4, SCALE_KWIDTH],
        )
        scale_b_view = tlx.local_trans(scale_b_view, (0, 3, 2, 1, 4))
        scale_b_view = tlx.local_reshape(scale_b_view, [BLOCK_N, BLOCK_K_SCALE])
    else:
        scale_b_view = tlx.local_view(b_scale_buf, slot)
    if BOTH:
        pass
    elif NUM_SUBTILES_N != 1 or NUM_SUBTILES_K != 1:
        scale_b_view = tlx.local_slice(scale_b_view, [subtile_start_n, subtile_start_k // SCALE_BLOCK],
                                       [subtile_n, subtile_k // SCALE_BLOCK])
    elif not SCALE_PRESHUFFLE:
        scale_b_view = tlx.local_slice(scale_b_view, [0, 0], [BLOCK_N, BLOCK_K_SCALE])
    if LOAD & 2:
        scale_b = tlx.local_load(scale_b_view, relaxed=RELAXED)

    if SCALES_FIRST and LOAD & 1:
        b = tlx.local_load(b_view, relaxed=RELAXED)
    if (LOAD & 2) == 0:
        scale_b = b
    if (LOAD & 1) == 0:
        b = scale_b
    return b, scale_b


XCD_REMAP_MODES = {"none": 0, "balanced": 1, "chunked": 2}


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
                            SPLIT_CLUSTER_BARRIER: tl.constexpr = False, CLUSTER_STEP: tl.constexpr = None):
    # CLUSTER_STEP: None tests k % CLUSTER_BARRIER_INTERVAL at run time;
    # False when the caller proves k can never be a rendezvous step.
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
    SROWS: tl.constexpr = 256 if _mxgemm_scale_cols(bs_desc) == BK // 32 * 256 else 128
    asd = tlx.update_tensor_descriptor(as_desc, add_offsets=[off_m // SROWS, 0], clamp_bounds=False)
    bsd = tlx.update_tensor_descriptor(bs_desc, add_offsets=[off_n // SROWS, 0], clamp_bounds=False)
    if CLUSTER_SIZE > 1:
        if CLUSTER_BARRIER_INTERVAL == 1:
            if SPLIT_CLUSTER_BARRIER:
                tlx.amd_cluster_wait()
            else:
                tlx.cluster_barrier()
        elif CLUSTER_BARRIER_INTERVAL > 1 and (CLUSTER_STEP is None or CLUSTER_STEP):
            if k % CLUSTER_BARRIER_INTERVAL == 0:
                if SPLIT_CLUSTER_BARRIER:
                    tlx.amd_cluster_wait()
                else:
                    tlx.cluster_barrier()
    mxgemm_issue_loads(ad, bd, asd, bsd, a_buf, b_buf, as_buf, bs_buf, k, None, BK // DA, BK // DB, BK // 32 * SROWS,
                       BUFFERS, True, True, True, FUSION, CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                       load_slot=slot, SCALES_FIRST=False, CLUSTER_LAYOUT=CLUSTER_LAYOUT)


@triton.constexpr_function
def _mxgemm_can_be_cluster_step(k0, step, interval):
    # Can k = k0 + step * n be a multiple of interval for some n >= 0?
    return (k0 % math.gcd(step, interval)) == 0


@triton.jit
def _mxgemm_persistent_a(a_buf, as_buf, slot, m: tl.constexpr, k: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                         DA: tl.constexpr, BUFFERS: tl.constexpr, PREFETCH_SCALES: tl.constexpr = False,
                         RELAXED: tl.constexpr = False, LOAD: tl.constexpr = 3, BOTH: tl.constexpr = False):
    # LOAD and BOTH: see mxgemm_load_a_operand. BOTH keeps a wave256 slot's
    # scales for both 128-row quadrants in one register tensor.
    tl.assume(slot >= 0)
    tl.assume(slot < BUFFERS)
    a, scale = mxgemm_load_a_operand(a_buf, as_buf, slot, m, k, BM, BK, DA, 32, BK // 32, BM // 128, 4, BUFFERS, 2,
                                     BK // 128, True, True,
                                     SCALES_FIRST=(PREFETCH_SCALES or ((BUFFERS == 4) and BK == 128)) and DA == 1,
                                     RELAXED=RELAXED, LOAD=LOAD, BOTH=BOTH)
    if (LOAD & 2) == 0:
        pass
    elif BOTH:
        tl.static_assert(BM == 256)
        scale = tlx.require_layout(scale, tlx.layout(shape=((32, 2, 2), (4, 2, 2)),
                                                     stride=((4, 0, 128), (1, 256, 512))))
    else:
        scale = tlx.require_layout(scale, tlx.layout(shape=((32, 2, 2), (4, BM // 128)),
                                                     stride=((4, 0, 128), (1, 256))))
    return a, scale


@triton.jit
def _mxgemm_persistent_b(b_buf, bs_buf, slot, k: tl.constexpr, n: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         DB: tl.constexpr, BUFFERS: tl.constexpr, PREFETCH_SCALES: tl.constexpr = False,
                         RELAXED: tl.constexpr = False, LOAD: tl.constexpr = 3, BOTH: tl.constexpr = False):
    tl.assume(slot >= 0)
    tl.assume(slot < BUFFERS)
    b, scale = mxgemm_load_b_operand(b_buf, bs_buf, slot, k, n, BN, BK, DB, 32, BK // 32, BN // 128, 4, BUFFERS, 2,
                                     BK // 128, True, True,
                                     SCALES_FIRST=(PREFETCH_SCALES or ((BUFFERS == 4) and BK == 128)) and DB == 1,
                                     RELAXED=RELAXED, LOAD=LOAD, BOTH=BOTH)
    if (LOAD & 2) == 0:
        pass
    elif BOTH:
        tl.static_assert(BN == 256)
        scale = tlx.require_layout(scale, tlx.layout(shape=((32, 2, 2), (4, 2, 2)),
                                                     stride=((4, 128, 0), (1, 256, 512))))
    else:
        scale = tlx.require_layout(scale, tlx.layout(shape=((32, 2, 2), (4, BN // 128)),
                                                     stride=((4, 128, 0), (1, 256))))
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
                                CLUSTER_LAYOUT: tl.constexpr, SPLIT_CLUSTER_BARRIER: tl.constexpr,
                                MERGED_BARRIER: tl.constexpr = False, CLUSTER_STEP: tl.constexpr = None):
    # One steady BK128 step of the deferred-bottom schedule. This matches the
    # DEFER_BOTTOM body of the persistent kernel's dynamic-slot K loop.
    # Pin every dot to the output store's native WMMA layout. A peeled step
    # starts from constant zeros and remainder steps follow the loop; without
    # pins, layout selection can pick different warp distributions for these
    # dots and insert accumulator conversions through LDS.
    if MERGED_BARRIER:
        # One workgroup rendezvous per step: read both second halves of this
        # slot during C00, split the rendezvous (s_barrier_signal /
        # s_barrier_wait) around C01's register-only WMMAs, then refill the
        # slot and read the next slot across C10 and C11 (32 WMMAs). The
        # barrier orders the slot's last reads before its refill and
        # publishes every wave's next-slot input; relaxed loads keep Membar
        # from adding barriers of its own around the refill.
        tl.static_assert(GROUPED_PREFETCH and PREFETCH_B_AFTER_C10)
        W256: tl.constexpr = _mxgemm_scale_cols(bs_buf) == BK // 32 * 256
        # Instruction counts behind the three scheduling regions below. Each of
        # the four waves computes a (BM/4)x(BN/4) quarter of every quadrant dot
        # (tiles_per_warp=[2, 2]) with 16x16 WMMAs, and reads its quarter of an
        # operand quadrant with ds_load_b128 (32 lanes x 16 bytes). An
        # operand's scales take one more read per quadrant; with wave256 scales
        # one read covers both quadrants, so the second halves read none.
        DOT_WMMAS: tl.constexpr = (BM // 4 // 16) * (BN // 4 // 16) * (BK // 128)
        A_READS: tl.constexpr = (BM // 4) * (BK // DA) // (32 * 16)
        B_READS: tl.constexpr = (BN // 4) * (BK // DB) // (32 * 16)
        C00_READS: tl.constexpr = B_READS + A_READS + (0 if W256 else 2)  # b01, a10 (+ sb01, sa10)
        C10_READS: tl.constexpr = A_READS + 1  # next slot's a00 and its scales
        C11_READS: tl.constexpr = B_READS + 1  # next slot's b00 and its scales
        # Hand-tuned density: DS reads issued after each WMMA, as early as
        # possible; the WMMAs left after the last read group cover the reads'
        # completion. C10 first spends one WMMA on each of the two TDM refill
        # loads (descriptor SALU, then the load).
        C00_READS_PER_WMMA: tl.constexpr = 4
        NEXT_READS_PER_WMMA: tl.constexpr = 3
        TDM_WMMAS: tl.constexpr = 2
        C00_GROUPS: tl.constexpr = (C00_READS + C00_READS_PER_WMMA - 1) // C00_READS_PER_WMMA
        C10_GROUPS: tl.constexpr = (C10_READS + NEXT_READS_PER_WMMA - 1) // NEXT_READS_PER_WMMA
        C11_GROUPS: tl.constexpr = (C11_READS + NEXT_READS_PER_WMMA - 1) // NEXT_READS_PER_WMMA
        tl.static_assert(LOADS == TDM_WMMAS)
        tl.static_assert(C00_GROUPS < DOT_WMMAS and TDM_WMMAS + C10_GROUPS < DOT_WMMAS and C11_GROUPS < DOT_WMMAS)
        if W256:
            # With wave256 scales, sa00/sb00 carry both quadrants' scales from
            # one ds_load_b128 per operand; read only the payloads here.
            sa00_all, sb00_all = sa00, sb00
            sa00, sa10 = _mxgemm_scale_quads(sa00_all, False)
            sb00, sb01 = _mxgemm_scale_quads(sb00_all, True)
            b01, _ = _mxgemm_persistent_b(b_buf, bs_buf, slot, 0, 1, BN, BK, DB, BUFFERS, RELAXED=True, LOAD=1)
            a10, _ = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 0, BM, BK, DA, BUFFERS, RELAXED=True, LOAD=1)
        else:
            b01, sb01 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 0, 1, BN, BK, DB, BUFFERS, RELAXED=True)
            a10, sa10 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 0, BM, BK, DA, BUFFERS, RELAXED=True)
        c00 = _mxgemm_native_dot(a00, sa00, "e4m3", b00, sb00, DTYPE_B, c00, tiles_per_warp=[2, 2])
        # Front-load the reads so the release drain before the signal has
        # several WMMAs of cover.
        for _ in tl.static_range(C00_GROUPS):
            tlx.amd_sched_group_barrier(0x8, 1, 107)
            tlx.amd_sched_group_barrier(0x100, C00_READS_PER_WMMA, 107)
        tlx.amd_sched_group_barrier(0x8, DOT_WMMAS - C00_GROUPS, 107)
        tlx.amd_sched_barrier()
        tlx.async_amd_descriptor_wait((BUFFERS - 2) * LOADS)
        tlx.named_barrier_arrive(0, tlx.num_warps() * 32)
        tlx.amd_sched_barrier()
        c01 = _mxgemm_native_dot(a00, sa00, "e4m3", b01, sb01, DTYPE_B, c01, tiles_per_warp=[2, 2])
        tlx.amd_sched_barrier()
        tlx.named_barrier_wait(0, tlx.num_warps() * 32)
        _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, off_m, off_n,
                                i + BUFFERS, slot, BK, DA, DB, BUFFERS, FUSION, CLUSTER_SIZE, CLUSTER_MULTICAST,
                                GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT=CLUSTER_LAYOUT,
                                SPLIT_CLUSTER_BARRIER=SPLIT_CLUSTER_BARRIER, CLUSTER_STEP=CLUSTER_STEP)
        last_b00, last_sb00 = b00, sb00
        a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 0, 0, BM, BK, DA, BUFFERS, RELAXED=True, BOTH=W256)
        c10 = _mxgemm_native_dot(a10, sa10, "e4m3", last_b00, last_sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
        tlx.amd_sched_group_barrier(0x4, 8, 108)
        tlx.amd_sched_group_barrier(0x8, 1, 108)
        tlx.amd_sched_group_barrier(0x10, 1, 108)
        tlx.amd_sched_group_barrier(0x4, 6, 108)
        tlx.amd_sched_group_barrier(0x8, 1, 108)
        tlx.amd_sched_group_barrier(0x10, 1, 108)
        for _ in tl.static_range(C10_GROUPS):
            tlx.amd_sched_group_barrier(0x8, 1, 108)
            tlx.amd_sched_group_barrier(0x100, NEXT_READS_PER_WMMA, 108)
        tlx.amd_sched_group_barrier(0x8, DOT_WMMAS - TDM_WMMAS - C10_GROUPS, 108)
        tlx.amd_sched_barrier()
        b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, 0, 0, BN, BK, DB, BUFFERS, RELAXED=True, BOTH=W256)
        c11 = _mxgemm_native_dot(a10, sa10, "e4m3", b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
        for _ in tl.static_range(C11_GROUPS):
            tlx.amd_sched_group_barrier(0x8, 1, 109)
            tlx.amd_sched_group_barrier(0x100, NEXT_READS_PER_WMMA, 109)
        tlx.amd_sched_group_barrier(0x8, DOT_WMMAS - C11_GROUPS, 109)
        tlx.amd_sched_barrier()
        return c00, c01, c10, c11, a00, sa00, b00, sb00
    else:
        c00, c01, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_top(
            c00, c01, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BM, BN, BK, DA, DB, DTYPE_B, BUFFERS,
            GROUPED_PREFETCH, ARRIVE_K=i + BUFFERS if SPLIT_CLUSTER_BARRIER else None, CLUSTER_SIZE=CLUSTER_SIZE,
            CLUSTER_BARRIER_INTERVAL=CLUSTER_BARRIER_INTERVAL, PIN_NATIVE=True)
        last_b00, last_sb00 = b00, sb00
        _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, off_m, off_n,
                                i + BUFFERS, slot, BK, DA, DB, BUFFERS, FUSION, CLUSTER_SIZE, CLUSTER_MULTICAST,
                                GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT=CLUSTER_LAYOUT,
                                SPLIT_CLUSTER_BARRIER=SPLIT_CLUSTER_BARRIER, CLUSTER_STEP=CLUSTER_STEP)
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
    MERGED_HANDOFF_BARRIER: tl.constexpr = False,
    STATIC_CLUSTER_STEPS: tl.constexpr = False,
    SCALE_WAVE256: tl.constexpr = False,
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
    tl.static_assert(tlx.dtype_of(c_ptr) == tl.float32 or tlx.dtype_of(c_ptr).primitive_bitwidth == 16)
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
    if MERGED_HANDOFF_BARRIER:
        tl.static_assert(STATIC_SLOT_UNROLL and NUM_BUFFERS == 4,
                         "MERGED_HANDOFF_BARRIER requires the four-buffer static-slot loop")
    if STATIC_CLUSTER_STEPS:
        tl.static_assert(STATIC_SLOT_UNROLL, "STATIC_CLUSTER_STEPS requires the static-slot loop")

    # "block128" scales: one row per 128 tile rows. "wave256" scales (see
    # pack_scale): one row per 256 tile rows, so each wave reads the scales of
    # both 128-row quadrants with a single ds_load_b128.
    if SCALE_WAVE256:
        tl.static_assert(BLOCK_K == 128 and DB == 1, "wave256 scales require A8W8 256x256x128 tiles")
    SC_ROWS: tl.constexpr = 256 if SCALE_WAVE256 else 128
    as_desc = tl.make_tensor_descriptor(a_scale, [M // SC_ROWS, K // 32 * SC_ROWS],
                                        [stride_as, tl.constexpr(1)], [BLOCK_M // SC_ROWS, BLOCK_K // 32 * SC_ROWS])
    bs_desc = tl.make_tensor_descriptor(b_scale, [N // SC_ROWS, K // 32 * SC_ROWS],
                                        [stride_bs, tl.constexpr(1)], [BLOCK_N // SC_ROWS, BLOCK_K // 32 * SC_ROWS])
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
                                layout=operand_shared_layout([BLOCK_M, BLOCK_K]))
        b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K // DB), tlx.dtype_of(b_ptr), NUM_BUFFERS,
                                layout=operand_shared_layout([BLOCK_N, BLOCK_K // DB]))
    as_buf = tlx.local_alloc((BLOCK_M // SC_ROWS, BLOCK_K // 32 * SC_ROWS), tlx.dtype_of(a_scale), NUM_BUFFERS,
                             layout=scale_shared_layout([BLOCK_M // SC_ROWS, BLOCK_K // 32 * SC_ROWS], SCALE_WAVE256))
    bs_buf = tlx.local_alloc((BLOCK_N // SC_ROWS, BLOCK_K // 32 * SC_ROWS), tlx.dtype_of(b_scale), NUM_BUFFERS,
                             layout=scale_shared_layout([BLOCK_N // SC_ROWS, BLOCK_K // 32 * SC_ROWS], SCALE_WAVE256))
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
        # The merged step reads both quadrants' wave256 scales at once.
        W256_PEEL_STEP: tl.constexpr = SCALE_WAVE256 and MERGED_HANDOFF_BARRIER
        a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, 0, 0, 0, BLOCK_M, BLOCK_K, 1, NUM_BUFFERS, BOTH=W256_PEEL_STEP)
        b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, 0, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS, BOTH=W256_PEEL_STEP)
        c00, c01, c10, c11, a00, sa00, b00, sb00 = _mxgemm_defer_bottom_k_step(
            tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32), tl.zeros(
                (BLOCK_M // 2, BLOCK_N // 2), tl.float32), tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32),
            tl.zeros((BLOCK_M // 2, BLOCK_N // 2),
                     tl.float32), a00, sa00, b00, sb00, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            off_m, off_n, 0, 0, 1, BLOCK_M, BLOCK_N, BLOCK_K, 1, DB, DTYPE_B, NUM_BUFFERS, "partial", LOADS,
            GROUPED_PREFETCH, PREFETCH_B_AFTER_C10, CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
            CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT, SPLIT_CLUSTER_BARRIER, MERGED_BARRIER=MERGED_HANDOFF_BARRIER)
        if W256_PEEL_STEP:
            sa00, _ = _mxgemm_scale_quads(sa00, False)
            sb00, _ = _mxgemm_scale_quads(sb00, True)

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
        # Merged static steps carry both quadrants' wave256 scales.
        W256_STEP: tl.constexpr = SCALE_WAVE256 and MERGED_HANDOFF_BARRIER and K // BLOCK_K - NUM_BUFFERS > 0
        a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, entry_slot, 0, 0, BLOCK_M, BLOCK_K, 1, NUM_BUFFERS,
                                         BOTH=W256_STEP)
        b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, entry_slot, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS,
                                         BOTH=W256_STEP)
        steady_end = k_iters - NUM_BUFFERS
        slot = entry_slot if not STATIC_SLOT_UNROLL else phase
        if STATIC_SLOT_UNROLL:
            tl.static_assert(DEFER_BOTTOM and (NUM_BUFFERS == 3 or NUM_BUFFERS == 4),
                             "STATIC_SLOT_UNROLL requires the BK128 staged-output schedule with three or four buffers")
            # The ring slot repeats every NUM_BUFFERS steps. Unrolling by the
            # ring depth keeps each step's slot loop-invariant, so LDS operand
            # addresses leave the loop. A per-step VALU address add would
            # otherwise make each ds_load wait for in-flight WMMAs (va_vdst).
            slot1 = tl.where(slot == NUM_BUFFERS - 1, 0, slot + 1)
            slot2 = tl.where(slot1 == NUM_BUFFERS - 1, 0, slot1 + 1)
            if NUM_BUFFERS == 4:
                slot3 = tl.where(slot2 == NUM_BUFFERS - 1, 0, slot2 + 1)
                slots = (slot, slot1, slot2, slot3)
            else:
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
                    SPLIT_CLUSTER_BARRIER, MERGED_BARRIER=MERGED_HANDOFF_BARRIER)
            REST: tl.constexpr = STEADY_STEPS - PEEL
            UNROLLED_END: tl.constexpr = PEEL + REST - REST % NUM_BUFFERS
            for i in tl.range(PEEL, UNROLLED_END, NUM_BUFFERS):
                for j in tl.static_range(NUM_BUFFERS):
                    # Step j refills stage i + j + NUM_BUFFERS with i = PEEL (mod NUM_BUFFERS):
                    # with STATIC_CLUSTER_STEPS, only steps that can reach a
                    # cluster rendezvous test for it.
                    c00, c01, c10, c11, a00, sa00, b00, sb00 = _mxgemm_defer_bottom_k_step(
                        c00, c01, c10, c11, a00, sa00, b00, sb00, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                        as_buf, bs_buf, off_m, off_n, i + j, slots[(PEEL + j) % NUM_BUFFERS],
                        slots[(PEEL + j + 1) % NUM_BUFFERS], BLOCK_M, BLOCK_N, BLOCK_K, 1, DB, DTYPE_B, NUM_BUFFERS,
                        "partial", LOADS, GROUPED_PREFETCH, PREFETCH_B_AFTER_C10, CLUSTER_SIZE, CLUSTER_MULTICAST,
                        GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT, SPLIT_CLUSTER_BARRIER,
                        MERGED_BARRIER=MERGED_HANDOFF_BARRIER, CLUSTER_STEP=_mxgemm_can_be_cluster_step(
                            PEEL + j +
                            NUM_BUFFERS, NUM_BUFFERS, CLUSTER_BARRIER_INTERVAL) if STATIC_CLUSTER_STEPS else None)
            for j in tl.static_range(REST % NUM_BUFFERS):
                c00, c01, c10, c11, a00, sa00, b00, sb00 = _mxgemm_defer_bottom_k_step(
                    c00, c01, c10, c11, a00, sa00, b00, sb00, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                    bs_buf, off_m, off_n, UNROLLED_END + j, slots[(UNROLLED_END + j) % NUM_BUFFERS],
                    slots[(UNROLLED_END + j + 1) % NUM_BUFFERS], BLOCK_M, BLOCK_N, BLOCK_K, 1, DB, DTYPE_B, NUM_BUFFERS,
                    "partial", LOADS, GROUPED_PREFETCH, PREFETCH_B_AFTER_C10, CLUSTER_SIZE, CLUSTER_MULTICAST,
                    GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT, SPLIT_CLUSTER_BARRIER,
                    MERGED_BARRIER=MERGED_HANDOFF_BARRIER)
            slot = slots[STEADY_STEPS % NUM_BUFFERS]
            if W256_STEP:
                sa00, _ = _mxgemm_scale_quads(sa00, False)
                sb00, _ = _mxgemm_scale_quads(sb00, True)
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


_BLOCK_M = 256
_BLOCK_N = 256
_BLOCK_K = 128
_GROUP_SIZE_M = 8
# Static unroll with the merged handoff barrier is measured up to K = 16384;
# beyond that the dynamic four-stage loop is faster.
_STATIC_MAX_K = 16384
_CLUSTER_LAYOUT = "4x4"
_CLUSTER_SIZE = 16
_CLUSTER_BARRIER_INTERVAL = 8


def _block128_scales(scale, rows, k, sf_layout):
    """Return ``[rows // 128, k // 32 * 128]`` uint8 scales in block128 order."""
    scale = scale.view(torch.uint8)
    if sf_layout == "natural":
        return pack_scale(scale)
    # The cuBLAS 128x4 atoms are already the block128 byte order.
    return scale.reshape(rows // 128, k // 32 * 128)


def _schedule(m, n, k, num_cus):
    """Pick the persistent schedule and launch shape for one problem."""
    k_tiles = k // _BLOCK_K
    tiles = (m // _BLOCK_M) * (n // _BLOCK_N)
    programs = min(num_cus, tiles)
    cfg = dict(NUM_BUFFERS=min(4, k_tiles), NUM_PROGRAMS=programs, CLUSTER_SIZE=1, CLUSTER_LAYOUT="linear",
               CLUSTER_BARRIER_INTERVAL=1, STATIC_SLOT_UNROLL=False, OVERLAP_PEEL=False, MERGED_HANDOFF_BARRIER=False,
               STATIC_CLUSTER_STEPS=False, OUTPUT_C_SLOTS=None)
    # Every cluster member must run the same number of tiles; 4x4 clusters
    # also need whole M groups and N tile columns.
    if (m % (_GROUP_SIZE_M * _BLOCK_M) == 0 and n % (4 * _BLOCK_N) == 0 and programs % _CLUSTER_SIZE == 0
            and tiles % programs == 0):
        cfg.update(CLUSTER_SIZE=_CLUSTER_SIZE, CLUSTER_LAYOUT=_CLUSTER_LAYOUT,
                   CLUSTER_BARRIER_INTERVAL=_CLUSTER_BARRIER_INTERVAL)
    if k_tiles > 4 and k <= _STATIC_MAX_K:
        cfg.update(STATIC_SLOT_UNROLL=True, OVERLAP_PEEL=True, MERGED_HANDOFF_BARRIER=True, OUTPUT_C_SLOTS=2,
                   STATIC_CLUSTER_STEPS=cfg["CLUSTER_SIZE"] > 1)
    return cfg


def mm_mxfp8(a, a_scale, b, b_scale, *, out=None, sf_layout="natural", space="heuristic"):
    """``tlx.ops.mm_mxfp8`` on gfx1250: BF16 ``a @ b.T`` from E4M3 data and E8M0 scales.

    The public wrapper has validated dtypes, layouts, scale sizes and ``out``.
    gfx1250 additionally needs M and N divisible by 256, at least two 128-wide
    K tiles, and natural or cuBLAS-blocked scales. There is one shape-picked
    schedule, so ``"full"`` and ``"heuristic"`` launch the same kernel.
    """
    if space not in ("heuristic", "full"):
        raise InvalidInput(f"tlx.ops.mm_mxfp8 on gfx1250 does not provide space={space!r}")
    if sf_layout not in ("natural", "cublas_blocked"):
        raise InvalidInput("tlx.ops.mm_mxfp8 on gfx1250 supports sf_layout 'natural' and 'cublas_blocked'; "
                           f"got {sf_layout!r}")
    m, k = a.shape
    n = b.shape[0]
    if m % _BLOCK_M or n % _BLOCK_N or k < 2 * _BLOCK_K:
        raise InvalidInput("gfx1250 tlx.ops.mm_mxfp8 requires M and N divisible by 256 and K >= 256; "
                           f"got M={m}, N={n}, K={k}")
    a_scale = _block128_scales(a_scale, m, k, sf_layout)
    b_scale = _block128_scales(b_scale, n, k, sf_layout)
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    cfg = _schedule(m, n, k, torch.cuda.get_device_properties(a.device).multi_processor_count)
    with torch.cuda.device(a.device):
        mxgemm_tdm_persistent_kernel[(cfg["NUM_PROGRAMS"], )](
            a, b, out, a_scale, b_scale, m, n, k, a.stride(0), b.stride(0), out.stride(0), a_scale.stride(0),
            b_scale.stride(0), DTYPE_B="e4m3", BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
            GROUP_SIZE_M=_GROUP_SIZE_M, NUM_BUFFERS=cfg["NUM_BUFFERS"], NUM_PROGRAMS=cfg["NUM_PROGRAMS"],
            CROSS_TILE_PREFETCH=True, XCD_REMAP_MODE=XCD_REMAP_MODES["chunked"], NUM_XCDS=8, XCD_CHUNK=2,
            CLUSTER_SIZE=cfg["CLUSTER_SIZE"], CLUSTER_MULTICAST=True,
            CLUSTER_BARRIER_INTERVAL=cfg["CLUSTER_BARRIER_INTERVAL"], CLUSTER_LAYOUT=cfg["CLUSTER_LAYOUT"],
            STATIC_SLOT_UNROLL=cfg["STATIC_SLOT_UNROLL"], OUTPUT_C_SLOTS=cfg["OUTPUT_C_SLOTS"],
            OVERLAP_PEEL=cfg["OVERLAP_PEEL"], MERGED_HANDOFF_BARRIER=cfg["MERGED_HANDOFF_BARRIER"],
            STATIC_CLUSTER_STEPS=cfg["STATIC_CLUSTER_STEPS"], num_warps=4, waves_per_eu=1,
            ctas_per_cga=(cfg["CLUSTER_SIZE"], 1, 1))
    return out
