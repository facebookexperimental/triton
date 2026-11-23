"""gfx1250 grouped MX GEMM (A8W4 and W4A4) backend for ``tlx.ops``."""

from __future__ import annotations

from typing import NamedTuple

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tlx.ops.kernels.mm_mxfp.gfx1250 import (
    PIPELINED_COMPILE_OPTIONS,
    mxgemm_load_a_operand,
    mxgemm_load_b_operand,
    mxgemm_position_load_descriptor,
    mxgemm_tdm_pipelined_kernel,
    operand_shared_layout,
    scale_shared_layout,
)

_BLOCK_N = 512
_BLOCK_K = 128


@triton.jit
def _build_grouped_tile_map(
    split_sizes,
    tile_meta0,
    tile_meta1,
    tile_count,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_TILES_PER_GROUP: tl.constexpr,
):
    group = tl.program_id(0)
    offsets = tl.arange(0, GROUP_BLOCK)
    sizes = tl.load(split_sizes + offsets, mask=offsets < NUM_GROUPS, other=0)
    tiles = tl.cdiv(sizes, BLOCK_M)
    before = offsets < group
    row_base = tl.sum(tl.where(before, sizes, 0), axis=0).to(tl.int64)
    tile_base = tl.sum(tl.where(before, tiles, 0), axis=0).to(tl.int64)
    group_rows = tl.load(split_sizes + group).to(tl.int64)
    group_tiles = tl.cdiv(group_rows, BLOCK_M)

    local_tile = tl.arange(0, MAX_TILES_PER_GROUP)
    mask = local_tile < group_tiles
    meta0 = row_base | (group_rows << 32)
    meta1 = group.to(tl.int64) | (local_tile.to(tl.int64) << 32)
    tl.store(tile_meta0 + tile_base + local_tile, meta0, mask=mask)
    tl.store(tile_meta1 + tile_base + local_tile, meta1, mask=mask)
    if group == NUM_GROUPS - 1:
        tl.store(tile_count, tile_base + group_tiles)
        tl.store(tile_count + 1, row_base + group_rows)


@triton.jit
def _w4p_issue(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, tile, k, slot: tl.constexpr, pred,
               TILES_N: tl.constexpr, K_STEPS: tl.constexpr):
    expert = tile // TILES_N
    row = (tile * K_STEPS + k) * 4
    tlx.async_amd_descriptor_load_fused([
        (mxgemm_position_load_descriptor(a_desc, [expert * 64, k * 64], pred, True), tlx.local_view(a_buf, slot), 3),
        (mxgemm_position_load_descriptor(b_desc, [row, 0], pred, True), tlx.local_view(b_buf, slot), 12),
    ])
    tlx.async_amd_descriptor_load_fused([
        (mxgemm_position_load_descriptor(as_desc, [expert * 64, k * 4], pred, True), tlx.local_view(as_buf, slot), 3),
        (mxgemm_position_load_descriptor(bs_desc, [row, 0], pred, True), tlx.local_view(bs_buf, slot), 12),
    ])


@triton.jit
def _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, slot: tl.constexpr,
              ld_tile, ld_k, ld_pred, TILES_N: tl.constexpr, K_STEPS: tl.constexpr, PENDING: tl.constexpr = 4):
    a00, scale_a00 = mxgemm_load_a_operand(a_buf, as_buf, slot, 0, 0, 64, 128, 2, 32, 4, 0, 4, 3, 2, 1, False, True,
                                           W4A4_NATIVE=True)
    b00, scale_b00 = mxgemm_load_b_operand(b_buf, bs_buf, slot, 0, 0, 512, 128, 2, 32, 4, 4, 4, 3, 2, 1, True, True,
                                           B_PRESHUFFLE=True, W4A4_NATIVE=True, B_WMMA_ORDER=True)
    a10, scale_a10 = mxgemm_load_a_operand(a_buf, as_buf, slot, 1, 0, 64, 128, 2, 32, 4, 0, 4, 3, 2, 1, False, True,
                                           W4A4_NATIVE=True)
    b01, scale_b01 = mxgemm_load_b_operand(b_buf, bs_buf, slot, 0, 1, 512, 128, 2, 32, 4, 4, 4, 3, 2, 1, True, True,
                                           B_PRESHUFFLE=True, W4A4_NATIVE=True, B_WMMA_ORDER=True)
    c00 = tlx.dot_scaled(a00, scale_a00, "e2m1", b00, scale_b00, "e2m1", c00, tiles_per_warp=(1, 2))
    c01 = tlx.dot_scaled(a00, scale_a00, "e2m1", b01, scale_b01, "e2m1", c01, tiles_per_warp=(1, 2))
    _w4p_issue(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, ld_tile, ld_k, slot, ld_pred, TILES_N,
               K_STEPS)
    c10 = tlx.dot_scaled(a10, scale_a10, "e2m1", b00, scale_b00, "e2m1", c10, tiles_per_warp=(1, 2))
    c11 = tlx.dot_scaled(a10, scale_a10, "e2m1", b01, scale_b01, "e2m1", c11, tiles_per_warp=(1, 2))
    tlx.async_amd_descriptor_wait(PENDING)
    return c00, c01, c10, c11


@triton.jit
def mxgemm_w4_persistent_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, stride_am, stride_cm, stride_scale,
                                NUM_PROGRAMS: tl.constexpr, NUM_TILES: tl.constexpr, TILES_N: tl.constexpr,
                                M_TOTAL: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    K_STEPS: tl.constexpr = K // 128
    MAX_TILES: tl.constexpr = 3
    tl.static_assert(K_STEPS == 32 and NUM_PROGRAMS <= NUM_TILES and NUM_TILES <= MAX_TILES * NUM_PROGRAMS)
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M_TOTAL, K // 2], strides=[stride_am, tl.constexpr(1)],
                                       block_shape=[64, 64])
    as_desc = tl.make_tensor_descriptor(a_scale, shape=[M_TOTAL, K // 32], strides=[stride_scale,
                                                                                    tl.constexpr(1)],
                                        block_shape=[64, 16])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[NUM_TILES * K_STEPS * 4, 8192], strides=[8192,
                                                                                              tl.constexpr(1)],
                                       block_shape=[4, 8192])
    bs_desc = tl.make_tensor_descriptor(b_scale, shape=[NUM_TILES * K_STEPS * 4, 512], strides=[512,
                                                                                                tl.constexpr(1)],
                                        block_shape=[4, 512])
    a_buf = tlx.local_alloc((64, 64), tlx.dtype_of(a_ptr), 3, layout=operand_shared_layout([64, 64]))
    b_buf = tlx.local_alloc((4, 8192), tlx.dtype_of(b_ptr), 3, layout=scale_shared_layout((4, 8192)))
    as_buf = tlx.local_alloc((64, 16), tlx.dtype_of(a_scale), 3, layout=scale_shared_layout((64, 16)))
    bs_buf = tlx.local_alloc((4, 512), tlx.dtype_of(b_scale), 3, layout=scale_shared_layout((4, 512)))
    c_buf = tlx.local_alloc((32, 512), tlx.dtype_of(c_ptr), 1)
    c_last = tlx.local_alloc((32, 512), tlx.dtype_of(c_ptr), 1, reuse=b_buf)
    pid = tl.program_id(axis=0)
    for p in tl.static_range(3):
        _w4p_issue(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, pid, p, p, None, TILES_N, K_STEPS)
    tlx.async_amd_descriptor_wait(4)
    c_dtype: tl.constexpr = tlx.dtype_of(c_ptr)
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M_TOTAL, N], strides=[stride_cm, tl.constexpr(1)],
                                       block_shape=[32, 512])
    d10 = tl.zeros((32, 256), dtype=c_dtype)
    d11 = tl.zeros((32, 256), dtype=c_dtype)
    d_row = pid * 0
    d_col = pid * 0
    for i in tl.static_range(MAX_TILES):
        tile = pid + i * NUM_PROGRAMS
        if tile < NUM_TILES:
            PHASE: tl.constexpr = (i * K_STEPS) % 3
            nxt = tile + NUM_PROGRAMS
            if i + 1 < MAX_TILES:
                has_next = nxt < NUM_TILES
            else:
                has_next = tl.full((), False, tl.int1)
            c00 = tl.zeros((32, 256), dtype=tl.float32)
            c01 = tl.zeros((32, 256), dtype=tl.float32)
            c10 = tl.zeros((32, 256), dtype=tl.float32)
            c11 = tl.zeros((32, 256), dtype=tl.float32)
            c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                           bs_buf, PHASE, tile, 3, None, TILES_N, K_STEPS, 4 if i == 0 else 5)
            c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                           bs_buf, (PHASE + 1) % 3, tile, 4, None, TILES_N, K_STEPS)
            if i > 0:
                # The previous tile's first-half store precedes this tile's
                # step-0 load in TDM order, so step 1's wait retired it.
                tl.debug_barrier()
                c_view = tlx.local_view(c_buf, 0)
                tlx.local_store(tlx.local_slice(c_view, [0, 0], [32, 256]), d10)
                tlx.local_store(tlx.local_slice(c_view, [0, 256], [32, 256]), d11)
                tlx.async_amd_descriptor_store(c_desc, c_view, [d_row, d_col])
            c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                           bs_buf, (PHASE + 2) % 3, tile, 5, None, TILES_N, K_STEPS, 4 if i == 0 else 5)
            for base in tl.range(3, 27, 3):
                for r in tl.static_range(3):
                    c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                                                   as_buf, bs_buf, (PHASE + r) % 3, tile, base + r + 3, None, TILES_N,
                                                   K_STEPS)
            c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                           bs_buf, PHASE, tile, 30, None, TILES_N, K_STEPS)
            c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                           bs_buf, (PHASE + 28) % 3, tile, 31, None, TILES_N, K_STEPS)
            for t in tl.static_range(3):
                c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                                               as_buf, bs_buf, (PHASE + 29 + t) % 3, nxt, t, has_next, TILES_N, K_STEPS)
            row0 = (tile // TILES_N) * 64
            col0 = (tile % TILES_N) * 512
            c_view = tlx.local_view(c_buf, 0)
            tlx.local_store(tlx.local_slice(c_view, [0, 0], [32, 256]), c00.to(c_dtype))
            tlx.local_store(tlx.local_slice(c_view, [0, 256], [32, 256]), c01.to(c_dtype))
            tlx.async_amd_descriptor_store(c_desc, c_view, [row0, col0])
            if has_next:
                d10 = c10.to(c_dtype)
                d11 = c11.to(c_dtype)
                d_row = row0 + 32
                d_col = col0
            else:
                c_view2 = tlx.local_view(c_last, 0)
                tlx.local_store(tlx.local_slice(c_view2, [0, 0], [32, 256]), c10.to(c_dtype))
                tlx.local_store(tlx.local_slice(c_view2, [0, 256], [32, 256]), c11.to(c_dtype))
                tlx.async_amd_descriptor_store(c_desc, c_view2, [row0 + 32, col0])
    tlx.async_amd_descriptor_wait(0)


def _pick_block_m(gm):
    return 16 if gm <= 4 else (32 if gm <= 256 else (64 if gm <= 4096 else 128))


class GroupedTileMap(NamedTuple):
    """Device tile map built by ``prepare_grouped_gemm_mxfp`` for one total row count."""

    tile_meta0: torch.Tensor
    tile_meta1: torch.Tensor
    tile_count: torch.Tensor
    gm: int
    block_m: int


def prepare_grouped_gemm_mxfp(split_sizes, gm):
    """Build a reusable tile map for ``gm`` total rows from device ``split_sizes``, with no host readback.

    The map uses the BLOCK_M that ``grouped_gemm_mxfp`` picks for ``gm`` and records both, so the op can reject a
    map built for a different row count.
    """
    groups = split_sizes.numel()
    block_m = _pick_block_m(gm)
    max_m_tiles = triton.cdiv(gm, block_m) + groups
    tile_meta0 = torch.empty(max_m_tiles, dtype=torch.int64, device=split_sizes.device)
    tile_meta1 = torch.empty_like(tile_meta0)
    tile_count = torch.zeros(2, dtype=torch.int64, device=split_sizes.device)
    if gm > 0:
        _build_grouped_tile_map[(groups, )](
            split_sizes,
            tile_meta0,
            tile_meta1,
            tile_count,
            NUM_GROUPS=groups,
            GROUP_BLOCK=triton.next_power_of_2(groups),
            BLOCK_M=block_m,
            MAX_TILES_PER_GROUP=triton.next_power_of_2(triton.cdiv(gm, block_m)),
        )
    return GroupedTileMap(tile_meta0, tile_meta1, tile_count, gm, block_m)


def preshuffle_grouped_weight_scale(w_scale, kstep_major=False):
    """Preshuffle E8M0 weight scales; kstep_major must match the payload's."""
    groups, n, k_scale = w_scale.shape
    if n % 128 or k_scale % 4:
        raise ValueError("Weight-scale N must be divisible by 128 and K/32 by 4")
    packed = w_scale.view(groups, n // 128, 4, 32, k_scale // 4, 4)
    packed = packed.permute(0, 1, 4, 3, 2, 5).contiguous()
    if kstep_major:
        if n % _BLOCK_N:
            raise ValueError(f"K-step-major weight scales need N divisible by {_BLOCK_N}")
        packed = packed.view(groups, n // _BLOCK_N, _BLOCK_N // 128, k_scale // 4, 512)
        packed = packed.permute(0, 1, 3, 2, 4).contiguous()
    return packed.view(groups, n // 128, k_scale * 128)


# (dk, dn) of each byte-offset bit of a [_BLOCK_N, _BLOCK_K // 2] B tile in W4A4
# WMMA register order; must match _b_wmma_order_layout() in the kernel.
_WMMA_ORDER_BASES = ((1, 0), (2, 0), (4, 0), (8, 0), (0, 1), (0, 2), (0, 4), (0, 8), (16, 0), (32, 0), (0, 16), (0, 32),
                     (0, 64), (0, 128), (0, 256))


def _wmma_order_index(device):
    offsets = torch.arange(_BLOCK_N * (_BLOCK_K // 2), device=device)
    k = torch.zeros_like(offsets)
    n = torch.zeros_like(offsets)
    for bit, (dk, dn) in enumerate(_WMMA_ORDER_BASES):
        set_bit = (offsets >> bit) & 1
        k |= set_bit * dk
        n |= set_bit * dn
    return n * (_BLOCK_K // 2) + k


def preshuffle_grouped_weight(w, kstep_major=False):
    """Preshuffle packed FP4 weights for B_PRESHUFFLE TDM loads.

    With kstep_major, each BLOCK_N-column tile stores its K steps back to back,
    each step in WMMA register order, so every K step's TDM transfer is one
    contiguous block.
    """
    groups, n, packed_k = w.shape
    if n % 128 or packed_k % 16:
        raise ValueError("Packed weight N must be divisible by 128 and K/2 by 16")
    if kstep_major:
        if n % _BLOCK_N or packed_k % (_BLOCK_K // 2):
            raise ValueError(f"K-step-major weights need N divisible by {_BLOCK_N} and K by {_BLOCK_K}")
        steps = packed_k // (_BLOCK_K // 2)
        tiles = w.view(groups, n // _BLOCK_N, _BLOCK_N, steps, _BLOCK_K // 2).permute(0, 1, 3, 2, 4)
        tiles = tiles.reshape(groups, n // _BLOCK_N, steps, _BLOCK_N * (_BLOCK_K // 2))
        return torch.gather(tiles, 3, _wmma_order_index(w.device).expand_as(tiles)).view_as(w)
    packed = w.view(groups, n // 128, 4, 32, packed_k // 16, 16)
    packed = packed.permute(0, 1, 4, 3, 2, 5).contiguous()
    return packed.view_as(w)


_SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)
_X_STORAGE_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}


def _check_split_sizes(split_sizes, gm, uniform_rows):
    if split_sizes.is_cuda and torch.cuda.is_current_stream_capturing():
        raise ValueError("validate=True reads split_sizes on the host and cannot run under graph capture")
    sizes = split_sizes.tolist()
    if any(rows < 0 for rows in sizes) or sum(sizes) != gm:
        raise ValueError(f"split_sizes must be nonnegative and sum to the {gm} rows of X; got sum {sum(sizes)}")
    if uniform_rows and any(rows != uniform_rows for rows in sizes):
        raise ValueError(f"uniform_grouped needs every group to have {uniform_rows} rows")


def _check_inputs(x, x_scale, w, w_scale, split_sizes, out, w_scale_preshuffled, x_dtype, w_dtype):
    tensors = (x, x_scale, w, w_scale, split_sizes)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("grouped_gemm_mxfp expects tensor inputs")
    if (x.ndim, x_scale.ndim, w.ndim, w_scale.ndim, split_sizes.ndim) != (2, 2, 3, 3, 1):
        raise ValueError("grouped_gemm_mxfp expects x/x_scale/w/w_scale/split_sizes ranks 2/2/3/3/1; got "
                         f"{x.ndim}/{x_scale.ndim}/{w.ndim}/{w_scale.ndim}/{split_sizes.ndim}")
    if x_dtype not in _X_STORAGE_DTYPES or w_dtype != "e2m1":
        raise ValueError("grouped_gemm_mxfp supports x_dtype 'e4m3' or 'e2m1' and w_dtype 'e2m1'; "
                         f"got x_dtype={x_dtype!r}, w_dtype={w_dtype!r}")
    if x.dtype != _X_STORAGE_DTYPES[x_dtype] or w.dtype != torch.uint8:
        raise ValueError(f"grouped_gemm_mxfp expects x_dtype={x_dtype!r} x stored as {_X_STORAGE_DTYPES[x_dtype]} "
                         f"and packed-E2M1 uint8 w; got x.dtype={x.dtype}, w.dtype={w.dtype}")
    if x_scale.dtype not in _SCALE_DTYPES or w_scale.dtype not in _SCALE_DTYPES:
        raise ValueError("grouped_gemm_mxfp expects E8M0 (or uint8) scales; "
                         f"got x_scale.dtype={x_scale.dtype}, w_scale.dtype={w_scale.dtype}")
    if split_sizes.dtype != torch.int32:
        raise ValueError(f"grouped_gemm_mxfp expects int32 split_sizes; got {split_sizes.dtype}")
    if not x.is_cuda or any(tensor.device != x.device for tensor in tensors[1:]):
        raise ValueError(f"grouped_gemm_mxfp expects all inputs on one GPU; got {[t.device for t in tensors]}")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("grouped_gemm_mxfp expects contiguous inputs")

    gm = x.shape[0]
    groups, n, packed_k = w.shape
    k = x.shape[1] if x_dtype == "e4m3" else 2 * x.shape[1]
    if 2 * packed_k != k or k % 32:
        raise ValueError(f"grouped_gemm_mxfp reduction dimensions must match and be divisible by 32; got "
                         f"x.shape={tuple(x.shape)} ({x.dtype}), w.shape={tuple(w.shape)}")
    if split_sizes.numel() != groups:
        raise ValueError(f"grouped_gemm_mxfp expects {groups} split_sizes; got {split_sizes.numel()}")
    if tuple(x_scale.shape) != (gm, k // 32):
        raise ValueError(f"grouped_gemm_mxfp expects x_scale shape {(gm, k // 32)}; got {tuple(x_scale.shape)}")
    w_scale_shape = (groups, n // 128, k // 32 * 128) if w_scale_preshuffled else (groups, n, k // 32)
    if (w_scale_preshuffled and n % 128) or tuple(w_scale.shape) != w_scale_shape:
        raise ValueError(f"grouped_gemm_mxfp expects {'preshuffled ' if w_scale_preshuffled else ''}w_scale shape "
                         f"{w_scale_shape}; got {tuple(w_scale.shape)}")
    if out is not None and (not isinstance(out, torch.Tensor) or out.dtype != torch.bfloat16
                            or tuple(out.shape) != (gm, n) or out.device != x.device or not out.is_contiguous()):
        raise ValueError(f"grouped_gemm_mxfp out must be a contiguous BF16 {(gm, n)} tensor on {x.device}")
    return gm, groups, n, k


def grouped_gemm_mxfp(
    x,
    x_scale,
    w,
    w_scale,
    split_sizes,
    *,
    x_dtype="e2m1",
    w_dtype="e2m1",
    out=None,
    tile_map=None,
    w_preshuffled=False,
    w_scale_preshuffled=False,
    uniform_grouped=False,
    w_kstep_major=False,
    validate=False,
):
    """Grouped MX GEMM on gfx1250: ``out[rows of group g] = x[rows of group g] @ w[g].T``.

    ``x_dtype`` and ``w_dtype`` name the MX formats: ``x`` is ``[GM, K]`` float8_e4m3fn for ``"e4m3"`` (A8W4) or
    ``[GM, K // 2]`` packed uint8 for ``"e2m1"`` (W4A4); ``w`` is ``[G, N, K // 2]`` packed E2M1 (``"e2m1"``, the
    only weight format). Scales are per-32 E8M0: ``x_scale [GM, K // 32]`` and ``w_scale [G, N, K // 32]``, or the
    ``preshuffle_grouped_weight_scale`` layout with ``w_scale_preshuffled=True``. Groups are contiguous along M;
    ``split_sizes`` is a device int32 ``[G]`` tensor and is not read on the host unless ``validate=True``, so the
    call is graph-capturable. Returns BF16 ``[GM, N]`` (``out`` if given).

    A ``tile_map`` from ``prepare_grouped_gemm_mxfp`` is used whenever given; otherwise one is built for
    256 < GM <= 4096. ``uniform_grouped`` and ``w_kstep_major`` are caller promises that every group has 64 rows;
    ``validate=True`` checks ``split_sizes`` on the host at the cost of a device sync.
    """
    gm, groups, n, k = _check_inputs(x, x_scale, w, w_scale, split_sizes, out, w_scale_preshuffled, x_dtype, w_dtype)
    if out is None:
        out = torch.empty((gm, n), dtype=torch.bfloat16, device=x.device)
    if gm == 0:
        return out
    x_scale_u8 = x_scale.view(torch.uint8)
    w_scale_u8 = w_scale.view(torch.uint8)
    block_m = _pick_block_m(gm)
    block_n = _BLOCK_N
    schedule = "baseline" if gm <= 4 else "sliceMNK"
    tiles_per_warp = (1, 2) if gm <= 256 or x_dtype == "e2m1" else (2, 2)
    uniform_grouped = (uniform_grouped and x_dtype == "e2m1" and block_m == 64 and gm == groups * block_m)
    if validate:
        _check_split_sizes(split_sizes, gm, block_m if uniform_grouped else 0)
    if tile_map is not None and (tile_map.gm != gm or tile_map.block_m != block_m):
        raise ValueError(f"tile_map was built for {tile_map.gm} rows (BLOCK_M={tile_map.block_m}); x has {gm} rows "
                         f"(BLOCK_M={block_m}). Rebuild it with prepare_grouped_gemm_mxfp(split_sizes, {gm})")
    max_m_tiles = groups if uniform_grouped else triton.cdiv(gm, block_m) + min(groups, gm) - 1
    use_tile_map = not uniform_grouped and (tile_map is not None or 256 < gm <= 4096)
    if use_tile_map:
        if tile_map is None:
            tile_map = prepare_grouped_gemm_mxfp(split_sizes, gm)
        tile_meta0, tile_meta1, tile_count = tile_map[:3]
    else:
        tile_meta0 = tile_meta1 = tile_count = split_sizes
    grid = (max_m_tiles * triton.cdiv(n, block_n), )
    static_ring = uniform_grouped and groups == 32 and k == 4096 and w_preshuffled and w_scale_preshuffled
    if w_kstep_major and (not static_ring or n % block_n):
        raise ValueError("w_kstep_major needs the uniform 32-group K=4096 W4A4 static ring, K-step-major "
                         f"preshuffled weights and weight scales, and N divisible by {block_n}")
    if w_kstep_major:
        programs = 2 * torch.cuda.get_device_properties(x.device).multi_processor_count
        tiles = groups * (n // block_n)
        if tiles > 3 * programs:
            raise ValueError(f"w_kstep_major runs at most 3 tiles per program; N={n} needs {tiles} tiles but "
                             f"{x.device} runs {programs} programs")
        mxgemm_w4_persistent_kernel[(min(programs,
                                         tiles), )](x, w, out, x_scale_u8, w_scale_u8, x.stride(0), out.stride(0),
                                                    x_scale_u8.stride(0), NUM_PROGRAMS=min(programs, tiles),
                                                    NUM_TILES=tiles, TILES_N=n // block_n, M_TOTAL=gm, N=n, K=k,
                                                    num_warps=4, waves_per_eu=2, **PIPELINED_COMPILE_OPTIONS)
        return out
    mxgemm_tdm_pipelined_kernel[grid](
        x,
        w,
        out,
        x_scale_u8,
        w_scale_u8,
        gm,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(2),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        x_scale_u8.stride(0),
        w_scale_u8.stride(1),
        split_sizes,
        groups,
        gm,
        w.stride(0),
        w_scale_u8.stride(0),
        tile_meta0,
        tile_meta1,
        tile_count,
        DTYPE_A=x_dtype,
        DTYPE_B=w_dtype,
        SCALE_BLOCK=32,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=_BLOCK_K,
        GROUP_SIZE_M=4,
        TRANSPOSE_B=True,
        NUM_BUFFERS=3,
        SCALE_PRESHUFFLE=False,
        B_SCALE_PRESHUFFLE=w_scale_preshuffled,
        B_PRESHUFFLE=w_preshuffled,
        WITH_A_SCALE=True,
        SCHEDULE=schedule,
        TDM_FUSION="2way",
        L2_PREFETCH_DISTANCE=-1,
        TDM_SPLIT=False,
        TILES_PER_WARP=tiles_per_warp,
        GROUPED=True,
        GROUPED_TILE_MAP=use_tile_map,
        UNIFORM_GROUPED=uniform_grouped,
        STATIC_N=n if uniform_grouped else 0,
        STATIC_K=k if block_m >= 64 else 0,
        FLY_QUADRANT_W4=uniform_grouped,
        W4A4_INTRAWAVE=uniform_grouped,
        W4A4_RELAXED_INTRAWAVE=uniform_grouped,
        W4A4_STATIC_RING=static_ring,
        num_warps=4,
        waves_per_eu=2 if block_m == 64 else 1,
        **PIPELINED_COMPILE_OPTIONS,
    )
    return out


__all__ = [
    "GroupedTileMap",
    "grouped_gemm_mxfp",
    "prepare_grouped_gemm_mxfp",
    "preshuffle_grouped_weight",
    "preshuffle_grouped_weight_scale",
]
