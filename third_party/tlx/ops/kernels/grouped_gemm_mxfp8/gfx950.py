"""gfx950 MXFP8 grouped GEMM for ``tlx.ops.grouped_gemm_mxfp8``.

Forward-only, persistent kernel for E4M3 operands, E8M0 per-32 microscales, and
BF16 output. Its tile bodies are shared with regular MXFP8 GEMM in
``tlx.ops.kernels.mxfp8_tiles_gfx950``. It reuses the FP16 gfx950 grouped
GEMM pipeline
(``grouped_gemm/gfx950.py``): a 256x256 output tile split into four 128x128
quadrants, one double-buffered LDS allocation per operand half-tile, and an
8-wave ``warp_pipeline_stage`` hot loop. ``tl.dot`` becomes ``tl.dot_scaled``
(CDNA4 16x16x128 scaled MFMA), and each half-tile's [128, 4] scale slice is
copied direct-to-LDS in the same commit group as its data.

Groups partition M. ``split_sizes`` stays on device; every program rescans it,
so the static persistent stride is graph-safe without a counter.

Unlike the FP16 kernel, this one keeps LLVM's default post-RA machine scheduler
(no ``TRITON_DISABLE_POST_MISCHED`` / nop strategy): with the scaled MFMAs and
scale reads it measured 1-8% faster across the TritonBench shapes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tlx.ops.kernels.grouped_gemm.gfx950 import (
    chiplet_transform_chunked,
    NUM_XCDS,
)
from triton.tlx.ops.kernels.grouped_gemm_mxfp8._scales import prepare_scales
from triton.tlx.ops.kernels.mxfp8_tiles_gfx950 import _generic_tile, _quad_tile

_BLOCK_K = 128
# AMD buffer descriptors expose at most this many bytes from each scalar base.
_MAX_BUFFER_BYTES = (1 << 31) - 2


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@triton.jit
def _device_trap_if(condition):
    """Abort the kernel (s_trap 2) when the uniform scalar condition holds."""
    if condition:
        tl.inline_asm_elementwise(
            "s_trap 2\nv_mov_b32 $0, 0",
            "=v",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _mxfp8_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    stride_b_scale_g,
    split_sizes_ptr,
    G,
    M,
    N,
    K,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    TILE_MODE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    K_MAJOR: tl.constexpr,
):
    """Persistent XCD-remapped scheduler over the concatenated per-group tile space."""
    pid = chiplet_transform_chunked(tl.program_id(0), NUM_SM, NUM_XCDS, XCD_CHUNK).to(tl.int64)
    k_chunks = K // BLOCK_SIZE_K

    if TILE_MODE == 0:
        HALF_M: tl.constexpr = BLOCK_SIZE_M // 2
        HALF_N: tl.constexpr = BLOCK_SIZE_N // 2
        tl.static_assert(HALF_M == 128 and HALF_N == 128 and BLOCK_SIZE_K == 128,
                         "padded LDS bases are pinned for [128, 128] byte half-tiles")
        # Same byte geometry as the gfx950 a4w4 tutorial's packed-FP4 half-tile:
        # pad 32 bytes every 1024 to break ds_read bank conflicts.
        smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [[1024, 32]],
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [1, 0],
                [32, 0],
                [64, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
            ],
            [HALF_M, BLOCK_SIZE_K],
        )
        dt_a = tlx.dtype_of(a_ptr)
        dt_s = tlx.dtype_of(a_scale_ptr)
        smem_a_top = tlx.local_alloc((HALF_M, BLOCK_SIZE_K), dt_a, 2, layout=smem_layout)
        smem_a_bot = tlx.local_alloc((HALF_M, BLOCK_SIZE_K), dt_a, 2, layout=smem_layout)
        smem_b_left = tlx.local_alloc((HALF_N, BLOCK_SIZE_K), dt_a, 2, layout=smem_layout)
        smem_b_right = tlx.local_alloc((HALF_N, BLOCK_SIZE_K), dt_a, 2, layout=smem_layout)
        if K_MAJOR:
            smem_sa_top = tlx.local_alloc((32, 16), dt_s, 2)
            smem_sa_bot = tlx.local_alloc((32, 16), dt_s, 2)
            smem_sb_left = tlx.local_alloc((32, 16), dt_s, 2)
            smem_sb_right = tlx.local_alloc((32, 16), dt_s, 2)
        else:
            smem_sa_top = tlx.local_alloc((HALF_M, BLOCK_SIZE_K // 32), dt_s, 2)
            smem_sa_bot = tlx.local_alloc((HALF_M, BLOCK_SIZE_K // 32), dt_s, 2)
            smem_sb_left = tlx.local_alloc((HALF_N, BLOCK_SIZE_K // 32), dt_s, 2)
            smem_sb_right = tlx.local_alloc((HALF_N, BLOCK_SIZE_K // 32), dt_s, 2)

    tile_idx = pid
    last_problem_end = tl.full((), 0, tl.int64)
    running_m = tl.full((), 0, tl.int64)
    num_n_tiles = tl.cdiv(N.to(tl.int64), BLOCK_SIZE_N)

    for g in range(G):
        m_size = tl.load(split_sizes_ptr + g).to(tl.int64)
        _device_trap_if((m_size < 0) | (running_m % 128 != 0) | (running_m + m_size > M))
        m_start = running_m
        num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
        num_tiles = num_m_tiles * num_n_tiles

        a_grp = a_ptr + m_start.to(tl.int64) * K
        b_grp = b_ptr + g.to(tl.int64) * N * K
        c_grp = c_ptr + m_start.to(tl.int64) * N
        bs_grp = b_scale_ptr + g.to(tl.int64) * stride_b_scale_g

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            local = tile_idx - last_problem_end
            num_pid_in_group = GROUP_SIZE_M * num_n_tiles
            group_id = local // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((local % num_pid_in_group) % group_size_m)
            pid_n = (local % num_pid_in_group) // group_size_m

            if TILE_MODE == 0:
                _quad_tile(
                    pid_m,
                    pid_n,
                    a_grp,
                    b_grp,
                    c_grp,
                    a_scale_ptr,
                    bs_grp,
                    m_start,
                    m_size,
                    N,
                    K,
                    k_chunks,
                    smem_a_top,
                    smem_a_bot,
                    smem_b_left,
                    smem_b_right,
                    smem_sa_top,
                    smem_sa_bot,
                    smem_sb_left,
                    smem_sb_right,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    N_ALIGNED=N_ALIGNED,
                    opaque_zero=(tile_idx // 0x7FFFFFFFFFFFFFFF).to(tl.int32),
                    K_MAJOR=K_MAJOR,
                )
            else:
                _generic_tile(
                    pid_m,
                    pid_n,
                    a_grp,
                    b_grp,
                    c_grp,
                    a_scale_ptr,
                    bs_grp,
                    m_start,
                    m_size,
                    N,
                    K,
                    k_chunks,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_STAGES=NUM_STAGES,
                    K_MAJOR=K_MAJOR,
                )
            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles
        running_m = running_m + m_size

    _device_trap_if(running_m != M)


def _pick_config(gm: int, n: int, nsm: int, *, k: int, allow_quad: bool = True) -> dict[str, int]:
    """Use quadrants when tiles fill the machine and each half fits its buffer descriptor."""
    quad_tiles = _cdiv(gm, 256) * _cdiv(n, 256)
    if allow_quad and quad_tiles >= nsm and 128 * k <= _MAX_BUFFER_BYTES:
        return {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "GROUP_SIZE_M": 4,
            "XCD_CHUNK": 32 if quad_tiles >= 2 * nsm else 8,
            "TILE_MODE": 0,
            "NUM_STAGES": 1,
        }
    bn = 256 if _cdiv(gm, 128) * _cdiv(n, 256) >= nsm else 128
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": bn,
        "GROUP_SIZE_M": 8,
        "XCD_CHUNK": 8,
        "TILE_MODE": 1,
        "NUM_STAGES": 2,
    }


def grouped_gemm_mxfp8(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    split_sizes: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    num_sms: int | None = None,
    sf_layout: str = "natural",
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    """Launch the forward-only gfx950 MXFP8 grouped GEMM backend."""
    if x.ndim != 2 or w.ndim not in (2, 3) or split_sizes.ndim != 1:
        raise ValueError("expected rank-2 x, rank-2/rank-3 w, and rank-1 split_sizes")
    if x.dtype != torch.float8_e4m3fn or w.dtype != torch.float8_e4m3fn:
        raise ValueError("gfx950 MXFP8 grouped GEMM supports E4M3 operands only")
    if x_scale.dtype != torch.float8_e8m0fnu or w_scale.dtype != torch.float8_e8m0fnu:
        raise ValueError("gfx950 MXFP8 grouped GEMM requires E8M0 scales")
    if split_sizes.dtype != torch.int32:
        raise ValueError("split_sizes must have dtype torch.int32")

    gm, k = x.shape
    g = split_sizes.shape[0]
    if w.ndim == 3:
        weight_groups, n, weight_k = w.shape
        if weight_groups != g:
            raise ValueError("w group count must match split_sizes")
    else:
        grouped_n, weight_k = w.shape
        if g == 0 or grouped_n % g != 0:
            raise ValueError("rank-2 w rows must be divisible by the group count")
        n = grouped_n // g
    if min(gm, g, n, k) <= 0 or weight_k != k:
        raise ValueError("grouped GEMM dimensions must be positive and K must match")
    if gm % 128 or k % _BLOCK_K:
        raise ValueError("GM and K must be divisible by 128")

    tensors = (x, x_scale, w, w_scale, split_sizes)
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("x, scales, w, and split_sizes must be contiguous")
    if x.device.type != "cuda" or any(tensor.device != x.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on x's device")

    device = x.device
    with torch.cuda.device(device):
        props = torch.cuda.get_device_properties(device)
        if not getattr(props, "gcnArchName", "").startswith("gfx950"):
            raise ValueError("gfx950 MXFP8 grouped GEMM requires a gfx950 device")
        if out is None:
            out = torch.empty((gm, n), device=device, dtype=torch.bfloat16)
        elif out.shape != (gm, n) or out.dtype != torch.bfloat16 or out.device != device or not out.is_contiguous():
            raise ValueError("out must be contiguous BF16 [GM, N] on x's device")

        # Natural scales are repacked anyway, so pack them in the K_MAJOR order the
        # gfx950 kernel reads with one ds_read_b32; blocked inputs stay zero-copy.
        k_major = sf_layout == "natural" and n % 128 == 0
        x_scale_5d, w_scale_5d = prepare_scales(x_scale, w_scale, gm=gm, g=g, n=n, k=k, sf_layout=sf_layout,
                                                k_major_chunks=k_major)
        launch_sms = props.multi_processor_count if num_sms is None else num_sms
        if launch_sms <= 0:
            raise ValueError("num_sms must be positive")
        cfg = _pick_config(gm, n, launch_sms, k=k)
        if config:
            cfg.update(config)
        # Direct-to-LDS needs a bounded byte span and provable pointer alignment.
        # Generic loads use full 64-bit addresses and support shifted storage.
        quad_safe = 128 * k <= _MAX_BUFFER_BYTES and all(tensor.data_ptr() % 16 == 0
                                                         for tensor in (x, w, x_scale_5d, w_scale_5d))
        if cfg["TILE_MODE"] == 0 and not quad_safe:
            cfg = _pick_config(gm, n, launch_sms, k=k, allow_quad=False)

        _mxfp8_grouped_gemm_kernel[(launch_sms, )](
            x,
            w,
            out,
            x_scale_5d,
            w_scale_5d,
            w_scale_5d.stride(0),
            split_sizes,
            g,
            gm,
            n,
            k,
            NUM_SM=launch_sms,
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=_BLOCK_K,
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            NUM_XCDS=NUM_XCDS,
            XCD_CHUNK=cfg["XCD_CHUNK"],
            TILE_MODE=cfg["TILE_MODE"],
            NUM_STAGES=cfg["NUM_STAGES"],
            N_ALIGNED=n % 128 == 0,
            K_MAJOR=k_major,
            num_warps=8,
            num_stages=1,
            matrix_instr_nonkdim=16,
            llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
        )
    return out


__all__ = ["grouped_gemm_mxfp8"]
