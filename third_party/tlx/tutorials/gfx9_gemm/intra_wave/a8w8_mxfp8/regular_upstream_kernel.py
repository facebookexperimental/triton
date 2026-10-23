"""Regular MXFP8 GEMM using current TLX's LDS-scale quadrant pipeline.

This has no group routing or persistent group scheduler. The hot tile helper is
shared with upstream tlx.ops.grouped_gemm_mxfp8 until the regular specialization
is independently tuned.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tlx.ops.kernels.grouped_gemm_mxfp8.gfx950 import _generic_tile, _quad_tile
from triton.tlx.ops.kernels.grouped_gemm.gfx950 import chiplet_transform_chunked
from triton.tlx.ops.kernels.grouped_gemm_mxfp8._scales import _swizzle_scale_to_5d


@triton.jit
def regular_mxfp8_generic_kernel(
    a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    _generic_tile(
        pid // (N // BN), pid % (N // BN),
        a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr,
        0, M, N, K, K // BK,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        NUM_STAGES=STAGES, K_MAJOR=True,
    )


def regular_mxfp8_generic(a, b, as_packed, bs_packed, *, out, return_kernel=False,
                          bm=128, bn=256, bk=128, stages=2, warps=8):
    m, k = a.shape
    n = b.shape[0]
    if m % bm or n % bn or k % bk:
        raise ValueError("matrix sizes must divide tile sizes")
    kernel = regular_mxfp8_generic_kernel[(m // bm * (n // bn),)](
        a, b, out, as_packed, bs_packed,
        M=m, N=n, K=k, BM=bm, BN=bn, BK=bk, STAGES=stages,
        num_warps=warps, num_stages=stages, matrix_instr_nonkdim=16,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    return (out, kernel) if return_kernel else out


@triton.jit
def regular_mxfp8_lds_scales_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    as_ptr,
    bs_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    HALF: tl.constexpr = 128
    BK: tl.constexpr = 128
    smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
            [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0], [16, 0],
        ],
        [HALF, BK],
    )
    data_dtype = tlx.dtype_of(a_ptr)
    scale_dtype = tlx.dtype_of(as_ptr)
    smem_a_top = tlx.local_alloc((HALF, BK), data_dtype, 2, layout=smem_layout)
    smem_a_bot = tlx.local_alloc((HALF, BK), data_dtype, 2, layout=smem_layout)
    smem_b_left = tlx.local_alloc((HALF, BK), data_dtype, 2, layout=smem_layout)
    smem_b_right = tlx.local_alloc((HALF, BK), data_dtype, 2, layout=smem_layout)
    smem_sa_top = tlx.local_alloc((32, 16), scale_dtype, 2)
    smem_sa_bot = tlx.local_alloc((32, 16), scale_dtype, 2)
    smem_sb_left = tlx.local_alloc((32, 16), scale_dtype, 2)
    smem_sb_right = tlx.local_alloc((32, 16), scale_dtype, 2)

    pid = tl.program_id(0)
    if XCD_CHUNK:
        pid = chiplet_transform_chunked(pid, (M // 256) * (N // 256), 8, XCD_CHUNK)
    pid_m = pid // (N // 256)
    pid_n = pid % (N // 256)
    _quad_tile(
        pid_m, pid_n, a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr,
        0, M, N, K, K // BK,
        smem_a_top, smem_a_bot, smem_b_left, smem_b_right,
        smem_sa_top, smem_sa_bot, smem_sb_left, smem_sb_right,
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=BK,
        N_ALIGNED=True,
        opaque_zero=pid // (1 << 30),
        K_MAJOR=True,
    )


def regular_mxfp8_lds_scales(a, b, as_packed, bs_packed, *, out,
                             return_kernel=False, agpr_alloc=False, xcd_chunk=0):
    m, k = a.shape
    n = b.shape[0]
    if m % 256 or n % 256 or k % 128:
        raise ValueError("M and N must be divisible by 256 and K by 128")
    kernel = regular_mxfp8_lds_scales_kernel[(m // 256 * (n // 256),)](
        a, b, out, as_packed, bs_packed,
        M=m, N=n, K=k, XCD_CHUNK=xcd_chunk,
        num_warps=8, num_stages=1, matrix_instr_nonkdim=16,
        llvm_fn_attrs=() if agpr_alloc else (("amdgpu-agpr-alloc", "0,0"),),
    )
    return (out, kernel) if return_kernel else out


def pack_regular_mxfp8_scale(scale):
    """Pack one natural [size, size/32] scale tensor for a square GEMM."""
    if scale.ndim != 2:
        raise ValueError("expected a rank-2 scale tensor")
    size, groups = scale.shape
    if size not in (1024, 2048, 4096, 8192, 16384) or groups != size // 32:
        raise ValueError("expected square 1K, 2K, 4K, 8K, or 16K scales")
    if scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError("scale must have E8M0 or uint8 dtype")
    if not scale.is_cuda:
        raise ValueError("scale must be on a GPU")
    return _swizzle_scale_to_5d(scale, batch=1, rows=size, outer_chunks=size // 128,
                                k_groups=size // 32, k_chunks=size // 128,
                                k_major_chunks=True)


def pack_regular_mxfp8_scales(a_scales, b_scales):
    """Pack natural A and B E8M0 scales; cache either result independently."""
    if a_scales.shape != b_scales.shape or a_scales.device != b_scales.device:
        raise ValueError("A and B scales must have the same shape and device")
    return pack_regular_mxfp8_scale(a_scales), pack_regular_mxfp8_scale(b_scales)


def regular_mxfp8_square(a, b, as_packed, bs_packed, *, out, return_kernel=False):
    """Run a square GEMM with prepacked K-major E8M0 scale tensors.

    Use :func:`pack_regular_mxfp8_scales` once before timing the GEMM. Weight
    scales can be cached if the weight matrix does not change.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("expected equal rank-2 A and B shapes")
    size = a.shape[0]
    if a.shape[1] != size or size not in (1024, 2048, 4096, 8192, 16384):
        raise ValueError("expected a square 1K, 2K, 4K, 8K, or 16K shape")
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise TypeError("A and B must use float8_e4m3fn")
    if out.shape != (size, size) or out.dtype != torch.bfloat16:
        raise ValueError("out must be a BF16 square matrix of the same size")
    if any(t.device != a.device or not t.is_contiguous() for t in (a, b, as_packed, bs_packed, out)):
        raise ValueError("A, B, packed scales, and out must be contiguous on one GPU")
    expected_scale_shape = (1, size // 128, size // 128, 2, 256)
    if (as_packed.shape != expected_scale_shape or bs_packed.shape != expected_scale_shape
            or as_packed.dtype != torch.uint8 or bs_packed.dtype != torch.uint8):
        raise ValueError("expected uint8 K-major blocked scale tensors")
    if size == 1024:
        return regular_mxfp8_generic(a, b, as_packed, bs_packed, out=out,
                                     return_kernel=return_kernel,
                                     bm=64, bn=64, stages=3, warps=4)
    if size == 2048:
        return regular_mxfp8_generic(a, b, as_packed, bs_packed, out=out,
                                     return_kernel=return_kernel,
                                     bm=64, bn=128, stages=2, warps=4)
    return regular_mxfp8_lds_scales(a, b, as_packed, bs_packed, out=out,
                                    return_kernel=return_kernel)
