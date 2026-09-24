"""gfx950 GEMM implementation for :func:`triton.tlx.ops.mm`.

The single public entry dispatches to reusable LocalSplitU, persistent,
register-resident, and direct-to-LDS execution paths in this module.
"""

import os
from functools import lru_cache
from types import MappingProxyType
from typing import NamedTuple

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..._catalog import InvalidInput

# Register-resident path.

_BLOCK_M = 256
_BLOCK_K = 64
_NUM_CU = 256
_MIN_KTILES_PER_SPLIT = 16

def _fixed_register_plan(block_m, block_n, block_k, group_m, num_xcds,
                         num_warps, num_stages):
    return MappingProxyType({
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "NUM_XCDS": num_xcds,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
        "kpack": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })


_SMALL_SQUARE_REGISTER_CONFIG = _fixed_register_plan(
    32, 16, 256, 4, 1, 2, 2
)
_MT64X64_BK256_REGISTER_CONFIG = _fixed_register_plan(
    64, 64, 256, 4, 8, 8, 2
)

_TUNED_SHAPE_CONFIGS = {
    (2048, 256, 1024): _MT64X64_BK256_REGISTER_CONFIG,
    (2041, 2041, 2048): _fixed_register_plan(
        128, 128, 128, 16, 8, 8, 2
    ),
}

_FP16_TUNED_SHAPE_CONFIGS = {
    (256, 257, 4096): _SMALL_SQUARE_REGISTER_CONFIG,
    (257, 257, 4096): _SMALL_SQUARE_REGISTER_CONFIG,
    (272, 3072, 4608): _MT64X64_BK256_REGISTER_CONFIG,
}


@triton.jit
# Triton TR001: callers select either a measured fixed plan or an autotuned one.
def _register_kernel_impl(  # noqa: TR001
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    row_sum_ptr,
    row_sum_sq_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    WRITE_STATS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_mn = grid_m * grid_n

    xcd_chunk: tl.constexpr = 4
    if NUM_XCDS != 1:
        aligned = (grid_mn // (NUM_XCDS * xcd_chunk)) * (NUM_XCDS * xcd_chunk)
        if pid < aligned:
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            pid = ((local_pid // xcd_chunk) * NUM_XCDS * xcd_chunk + xcd * xcd_chunk + local_pid % xcd_chunk)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    input_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    input_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    offs_m = (input_rows if M % BLOCK_M == 0 else tl.where(input_rows < M, input_rows, 0))
    offs_n = (input_cols if N % BLOCK_N == 0 else tl.where(input_cols < N, input_cols, 0))
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    reg_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    reg_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    full_k_tiles: tl.constexpr = K // BLOCK_K
    for k_idx in range(0, full_k_tiles):
        k = k_idx * BLOCK_K
        a_ptrs = (a_ptr + reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = (b_ptr + (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
    if K % BLOCK_K != 0:
        k = full_k_tiles * BLOCK_K
        k_mask = offs_k < K - k
        a_ptrs = (a_ptr + reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = (b_ptr + (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    idx_m = rows[:, None]
    idx_n = cols[None, :]
    mask = (idx_m < M) & (idx_n < N)
    if ADD_BIAS:
        bias_offsets = idx_m * stride_bias_m + idx_n * stride_bias_n
        bias = tl.load(
            bias_ptr + bias_offsets,
            mask=mask,
            eviction_policy="evict_last",
        )
        acc += bias.to(tl.float32)
    value = acc.to(c_ptr.dtype.element_ty)
    if WRITE_STATS:
        value_fp32 = tl.where(mask, value.to(tl.float32), 0.0)
        stats_offsets = rows * grid_n + pid_n
        if not IS_RMS_NORM:
            tl.store(
                row_sum_ptr + stats_offsets,
                tl.sum(value_fp32, axis=1),
                mask=rows < M,
            )
        tl.store(
            row_sum_sq_ptr + stats_offsets,
            tl.sum(value_fp32 * value_fp32, axis=1),
            mask=rows < M,
        )
    output_offsets = idx_m * stride_cm + idx_n * stride_cn
    tl.store(c_ptr + output_offsets, value, mask=mask)


def _launch_register_plan(a, b, *, config, bias=None, out=None, _validated=False):
    """Launch one validated register-resident plan."""
    m, k = a.shape
    b_k, n = b.shape
    if not _validated and k != b_k:
        raise ValueError(f"Incompatible matrix dimensions: {tuple(a.shape)} and "
                         f"{tuple(b.shape)}")
    if bias is not None:
        if bias.shape != (m, n):
            raise ValueError(f"Bias must expand to ({m}, {n}), got {tuple(bias.shape)}")
        if bias.device != a.device or bias.dtype != a.dtype:
            raise ValueError("Bias and matrix operands must have matching device and dtype")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    disable_agpr = (k == 256 and n > 256) or (k > 512 and (k % _BLOCK_K != 0 or m * n <= 2 * 1024 * 1024))
    launch_options = ({"llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"), )} if disable_agpr else {})
    if config["BLOCK_K"] == 128 and config["num_stages"] == 3:
        launch_options["reverse_local_assignment"] = True
    bias_ptr = bias if bias is not None else out
    grid = (triton.cdiv(m, config["BLOCK_M"]) * triton.cdiv(n, config["BLOCK_N"]), )
    _register_kernel_impl[grid](
        a,
        b,
        bias_ptr,
        out,
        out,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        out.stride(0),
        out.stride(1),
        ADD_BIAS=bias is not None,
        WRITE_STATS=False,
        IS_RMS_NORM=False,
        **config,
        **launch_options,
    )
    return out


def _register_split_k_for(grid_mn, k):
    min_ks = _MIN_KTILES_PER_SPLIT * _BLOCK_K
    best = 1
    for split_k in range(2, _NUM_CU // grid_mn + 1):
        split_size = k // split_k
        if (k % split_k == 0 and split_size >= min_ks and split_size % _BLOCK_K == 0):
            best = split_k
    return best


def _default_lds_block_m(m, n, k):
    large_grid = triton.cdiv(m, 256) * triton.cdiv(n, 256)
    large_fill = large_grid * _register_split_k_for(large_grid, k)
    if large_fill >= _NUM_CU // 2:
        return 256
    small_grid = triton.cdiv(m, 128) * triton.cdiv(n, 128)
    small_fill = small_grid * _register_split_k_for(small_grid, k)
    return 128 if small_fill > large_fill else 256


def _register_config_for(m, n, k):
    small_grid = triton.cdiv(m, 128) * triton.cdiv(n, 128)
    large_grid = triton.cdiv(m, 256) * triton.cdiv(n, 256)
    if not (k > 512 and k % _BLOCK_K == _BLOCK_K // 2 and large_grid < _NUM_CU <= small_grid):
        return None
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 8,
        "NUM_XCDS": 8,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
        "kpack": 1,
        "num_warps": 4,
        "num_stages": 4,
    }


def _intermediate_register_config(m, n, k):
    if n >= 2 * k:
        block_m, block_n, block_k = 128, 128, 128
        group_m, num_warps, num_stages = 16, 8, 2
    elif k >= 2 * n and 4 * m < 3 * triton.cdiv(m, 128) * 128:
        block_m, block_n, block_k = 64, 32, 128
        group_m, num_warps, num_stages = 8, 4, 2
    elif k >= 2 * n:
        block_m, block_n, block_k = 128, 64, 128
        group_m, num_warps, num_stages = 4, 8, 3
    else:
        block_m, block_n, block_k = 128, 64, 64
        group_m, num_warps, num_stages = 4, 4, 3
    grid_mn = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "NUM_XCDS": 8 if grid_mn >= _NUM_CU else 1,
        "matrix_instr_nonkdim": 16 if block_m == 64 else 32,
        "waves_per_eu": 0,
        "kpack": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _register_plan_for_shape(m, n, k, dtype=None):
    """Return the bounded register plan selected by the gfx950 geometry."""
    shape = (m, n, k)
    tuned = _TUNED_SHAPE_CONFIGS.get(shape)
    if dtype == torch.float16:
        tuned = _FP16_TUNED_SHAPE_CONFIGS.get(shape, tuned)
    if tuned is not None:
        return tuned

    config = _register_config_for(m, n, k)
    if config is not None:
        return MappingProxyType(config)

    block_m = _default_lds_block_m(m, n, k)
    padded_m = triton.cdiv(m, block_m) * block_m
    is_intermediate_m = _BLOCK_M // 4 < m < 4 * _BLOCK_M
    has_high_m_padding = 4 * m < 3 * padded_m
    if not is_intermediate_m or (block_m == _BLOCK_M and not has_high_m_padding):
        return None
    return MappingProxyType(_intermediate_register_config(m, n, k))


# Direct-to-LDS and Stream-K paths.

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
NUM_WARPS = 8
GROUP_SIZE_M = 4
NUM_XCDS = 8

MIN_K = 2 * BLOCK_K  # pipeline prefetches 2 whole K-tiles; the rest goes to the masked tail
KERNEL_NAME = "a16w16_8wave"
_LLVM_ATTRS = (("amdgpu-agpr-alloc", "0,0"), )
_READY_VALUE = 3


def _prune_register_configs(configs, named_args, **_):
    k = named_args["K"]
    if 128 <= k < 256 and named_args["M"] >= 16384 and named_args["N"] <= 128:
        preferred = [
            config for config in configs if config.kwargs["NUM_XCDS"] == 1 and config.kwargs["BLOCK_M"] == 128
            and config.kwargs["BLOCK_N"] == 64 and config.kwargs["BLOCK_K"] == 64 and config.kwargs["GROUP_M"] == 4
            and config.kwargs["waves_per_eu"] == 0 and config.num_warps == 4 and config.num_stages == 2
        ]
        if preferred:
            return preferred
    if k == 1536 and named_args["M"] == 3072 and named_args["N"] == 3072:
        preferred = [
            config for config in configs if config.kwargs["NUM_XCDS"] == 8 and config.kwargs["BLOCK_M"] == 128
            and config.kwargs["BLOCK_N"] == 128 and config.kwargs["BLOCK_K"] == 64 and config.kwargs["GROUP_M"] == 16
            and config.kwargs["waves_per_eu"] == 0 and config.num_warps == 4 and config.num_stages == 2
        ]
        if preferred:
            return preferred
    if k == 256 and named_args["M"] <= 1024 and named_args["N"] >= 16384:
        preferred = [
            config for config in configs
            if config.kwargs["BLOCK_M"] == 256 and config.kwargs["BLOCK_N"] == 128 and config.kwargs["BLOCK_K"] == 32
            and config.kwargs["GROUP_M"] == 4 and config.kwargs["waves_per_eu"] == 2 and config.num_warps == 4
        ]
        if preferred:
            return preferred
    return configs


# Triton TR001: autotune the register path across stock ROCm tiles and the
# deeper software-pipelined tiles used by the TorchTLX register path.
_REGISTER_CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": 1,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
        },
        num_warps=num_warps,
        num_stages=2,
    ) for block_m, block_n, block_k, group_m, num_warps, waves_per_eu in (
        (16, 16, 256, 4, 4, 2),
        (32, 16, 256, 4, 4, 0),
        (32, 32, 16, 8, 4, 2),
        (32, 32, 128, 8, 4, 0),
        (32, 64, 64, 8, 4, 0),
        (64, 16, 128, 8, 4, 2),
        (64, 32, 32, 8, 4, 0),
        (64, 32, 64, 8, 4, 0),
        (64, 32, 64, 8, 8, 0),
        (64, 32, 128, 8, 4, 0),
        (64, 64, 16, 8, 4, 0),
        (64, 64, 64, 4, 4, 0),
        (64, 64, 128, 16, 8, 0),
        (64, 64, 256, 4, 8, 0),
        (64, 128, 32, 4, 4, 2),
        (64, 128, 32, 8, 8, 0),
        (64, 128, 64, 4, 8, 0),
        (64, 128, 128, 4, 8, 0),
        (128, 32, 32, 8, 4, 0),
        (128, 32, 64, 8, 4, 0),
        (128, 64, 32, 8, 4, 2),
        (128, 64, 64, 16, 4, 0),
        (128, 64, 128, 4, 8, 0),
        (128, 128, 32, 16, 4, 2),
        (128, 128, 32, 16, 8, 0),
        (128, 128, 32, 16, 8, 2),
        (128, 128, 64, 16, 4, 0),
        (128, 128, 64, 8, 8, 0),
        (128, 128, 128, 16, 8, 0),
        (128, 256, 32, 16, 4, 2),
        (128, 256, 64, 4, 8, 0),
        (256, 64, 64, 4, 8, 0),
        (256, 128, 32, 4, 4, 2),
        (256, 128, 32, 16, 8, 0),
        (256, 128, 64, 4, 8, 0),
        (256, 256, 64, 4, 8, 0),
    )
]

_REGISTER_CONFIGS += [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": 1,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    ) for block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu in (
        (128, 64, 64, 4, 4, 2, 0),
        (128, 64, 64, 4, 4, 3, 0),
        (128, 64, 64, 16, 4, 3, 0),
        (128, 128, 64, 8, 4, 2, 0),
        (128, 128, 64, 8, 4, 3, 0),
        (128, 128, 64, 16, 4, 3, 0),
        (128, 256, 64, 8, 8, 3, 0),
        (128, 256, 64, 8, 8, 3, 1),
        (256, 128, 32, 4, 4, 3, 2),
        (256, 128, 32, 4, 4, 4, 2),
        (256, 256, 64, 4, 8, 3, 0),
        (256, 256, 64, 4, 8, 4, 0),
        (128, 256, 32, 8, 8, 3, 0),
    )
]

_REGISTER_CONFIGS += [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": 8,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    ) for block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu in (
        (128, 64, 64, 4, 4, 2, 0),
        (128, 64, 64, 4, 4, 3, 0),
        (128, 128, 64, 8, 4, 2, 0),
        (128, 128, 64, 8, 4, 3, 0),
        (128, 128, 64, 16, 4, 2, 0),
        (128, 128, 64, 16, 4, 3, 0),
        (256, 128, 32, 4, 4, 3, 2),
    )
]

# Coalesced SIMD register layout for the [HALF_M, HALF_N] = [128, 128] fp16 quadrant
# store (num_warps=8, warp_size=64): each thread holds 8 contiguous N elements ->
# 128-bit buffer_store_dwordx4. Applied to the epilogue store via tlx.require_layout
# so tritongpu-coalesce sets the store to this #linear layout and AMD
# OptimizeEpilogue leaves it alone (it only rewrites #blocked stores) -- keeping the
# wide coalesced store instead of the narrow MMA-accumulator (dwordx2) fallback.
_C_STORE_SIMD_LAYOUT = tlx.layout(shape=((16, 4, 8), (8, 4)), stride=((8, 128, 512), (1, 4096)))


def _swz_offset_bases(shape, contig_dim):
    """Padded-shared swizzle offset bases for a 2D fp16 half-tile, derived from the
    tile shape so both tile sizes share one path (no per-size branch).

    `contig_dim` is the K-contiguous axis (0 or 1); its bits come first (fastest),
    then the free axis contributes its high bits (>= bit 4) before its low bits --
    the row/col permutation that makes the direct-to-LDS ds_reads bank-conflict-free
    on the 128x64 / 64x128 halves. A 128-wide free axis simply carries the extra top
    bit ([64,0] resp. [0,64]) that a 64-wide one omits. Used for both operands: the
    a half-tile [HALF_M, BLOCK_K] has K on dim 1, the b half-tile [BLOCK_K, HALF_N]
    has K on dim 0."""

    def basis(dim, i):
        return [1 << i, 0] if dim == 0 else [0, 1 << i]

    free_dim = 1 - contig_dim
    # log2 of each extent: int(n).bit_length() - 1 == floor(log2(n)), exact for the
    # power-of-two tile extents here (integer math, no float log2).
    cb = int(shape[contig_dim]).bit_length() - 1
    fb = int(shape[free_dim]).bit_length() - 1
    contig = [basis(contig_dim, i) for i in range(cb)]
    free = ([basis(free_dim, i) for i in range(4, fb)] + [basis(free_dim, i) for i in range(min(4, fb))])
    return contig + free


# Swizzle offset bases per (square) tile size, computed once from the tile shape by
# _swz_offset_bases. The @jit body can't call the generator (only constexpr module
# values are referenceable inside @jit), so precompute the base lists here and build
# the layout in-body, selecting by the constexpr tile size.
# The bases are built for a half-tile (2x2 quadrant tiling): HALF = tile // 2.
_HALF_256 = 256 // 2  # half of the 256x256 tile
_HALF_128 = 128 // 2  # half of the 128x128 tile
_A_BASES_256 = tl.constexpr(_swz_offset_bases([_HALF_256, BLOCK_K], 1))
_A_BASES_128 = tl.constexpr(_swz_offset_bases([_HALF_128, BLOCK_K], 1))
_B_BASES_256 = tl.constexpr(_swz_offset_bases([BLOCK_K, _HALF_256], 0))
_B_BASES_128 = tl.constexpr(_swz_offset_bases([BLOCK_K, _HALF_128], 0))
# Direct-to-LDS offset layouts inferred by the aligned 256x256 path. Pinning
# these keeps a merely 16-byte-aligned leading stride from falling back to a
# blocked layout that the AMD buffer-load lowering cannot consume.
_A_OFFSET_LAYOUT_256 = tlx.layout(shape=((8, 8, 8), (8, 2)), stride=((8, 1024, 64), (1, 512)))
_B_OFFSET_LAYOUT_256 = tlx.layout(shape=((8, 8, 8), (8, 2)), stride=((1024, 16, 1), (128, 8)))

_register_kernel = triton.autotune(
    configs=_REGISTER_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_register_configs},
)(_register_kernel_impl)


def _launch_register(a, b, bias=None, config=None, out=None):
    """Launch the register-resident gfx950 GEMM path.

    ``config=None`` autotunes over `_REGISTER_CONFIGS`. Passing an explicit
    config dict bypasses the autotuner and launches that config directly --
    the same convention the Blackwell/Hopper tutorials use so correctness
    tests can pin a config instead of paying for a sweep.
    """
    if config is not None:
        return _launch_register_plan(
            a,
            b,
            bias=bias,
            config=config,
            out=out,
        )

    M, K = a.shape
    b_k, N = b.shape
    if K != b_k:
        raise ValueError(f"Incompatible matrix dimensions: {tuple(a.shape)} and {tuple(b.shape)}")
    if bias is not None:
        if bias.shape != (M, N):
            raise ValueError(f"Bias must expand to ({M}, {N}), got {tuple(bias.shape)}")
        if bias.device != a.device or bias.dtype != a.dtype:
            raise ValueError("Bias and matrix operands must have matching device and dtype")
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    disable_agpr = (K == 256 and N > 256) or (K > 512 and (K % BLOCK_K != 0 or M * N <= 2 * 1024 * 1024))
    launch_options = {"llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"), )} if disable_agpr else {}
    bias_ptr = bias if bias is not None else out
    args = (
        a,
        b,
        bias_ptr,
        out,
        out,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        out.stride(0),
        out.stride(1),
    )
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )
    _register_kernel[grid](
        *args,
        ADD_BIAS=bias is not None,
        WRITE_STATS=False,
        IS_RMS_NORM=False,
        **launch_options,
    )
    return out


@triton.jit
def matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, a_top_off, a_bot_off, b_left_off,
                b_right_off, ka, kb, n_steps, stride_ak, stride_bk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    """Compute one output tile over an even contiguous range of K64 steps.

    ``ka`` and ``kb`` are the initial element offsets along K. ``n_steps`` must
    be even and at least two. Both the original data-centric kernel and the new
    Stream-K kernel use this same LDS/MFMA pipeline.
    """
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2

    # Keep the direct-to-LDS producer contract local to this extracted helper.
    # K is contiguous in A's second tensor dimension and B's first tensor
    # dimension. The helper boundary otherwise hides those width/alignment
    # facts from AxisInfo and buffer-load lowering falls back to an illegal
    # scalar copy.
    a_top_off = tl.max_contiguous(tl.multiple_of(a_top_off, (1, 8)), (1, 8))
    a_bot_off = tl.max_contiguous(tl.multiple_of(a_bot_off, (1, 8)), (1, 8))
    b_left_off = tl.max_contiguous(tl.multiple_of(b_left_off, (8, 1)), (8, 1))
    b_right_off = tl.max_contiguous(tl.multiple_of(b_right_off, (8, 1)), (8, 1))

    k_step_a = BLOCK_K * stride_ak
    k_step_b = BLOCK_K * stride_bk
    a_top_off_n = a_top_off + k_step_a
    a_bot_off_n = a_bot_off + k_step_a
    b_left_off_n = b_left_off + k_step_b
    b_right_off_n = b_right_off + k_step_b
    acc_tl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)

    # ── Prologue: prefetch K-steps 0,1 into buffers 0,1 (8 commits) ──
    tlx.buffer_load_to_local(smem_b_left[0], b_ptr, b_left_off + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[0], b_ptr, b_right_off + kb)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_left[1], b_ptr, b_left_off_n + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off_n + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off_n + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[1], b_ptr, b_right_off_n + kb)
    tlx.async_load_commit_group()

    ka += BLOCK_K * stride_ak * 2
    kb += BLOCK_K * stride_bk * 2

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(smem_b_left[0], relaxed=True)
    a_top = tlx.local_load(smem_a_top[0], relaxed=True)

    # ── Main loop (2x unrolled): 8 (mfma + local_load + async refill) regions ──
    for k in tl.range(0, n_steps - 2, 2, num_stages=1):
        # --- sub-iter 0 (buffer 0) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_left[0], b_ptr, b_left_off + kb)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[0], relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[1], relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[0], b_ptr, b_right_off + kb)
            tlx.async_load_commit_group()

        # --- sub-iter 1 (buffer 1, _next offsets) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_left[1], b_ptr, b_left_off_n + kb)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[1], relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off_n + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[0], relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off_n + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[1], b_ptr, b_right_off_n + kb)
            tlx.async_load_commit_group()
            ka += BLOCK_K * stride_ak * 2
            kb += BLOCK_K * stride_bk * 2

    # ── Epilogue: last 2 pipelined K-steps, drain LDS loads ──
    # iter n_steps-2
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(5)
    l_idx: tl.constexpr = 0  # (n_steps - 2) % 2, always 0 since n_steps is even
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, l_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    tlx.async_load_wait_group(3)
    g_idx: tl.constexpr = 1  # 1 - l_idx
    b_left = tlx.local_load(tlx.local_view(smem_b_left, g_idx), relaxed=True)

    acc_br = tl.dot(a_bot, b_right, acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)

    # iter n_steps-1: finish ALL four mfmas before returning so the dot operands
    # die and the caller holds only the four f32 accumulators.
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, g_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    acc_br = tl.dot(a_bot, b_right, acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


def _streamk_schedule(M, N, K, block_m=BLOCK_M, block_n=BLOCK_N):
    """Build a persistent or variable-work Stream-K schedule."""
    num_pid_m = M // block_m
    num_pid_n = N // block_n
    total_tiles = num_pid_m * num_pid_n
    n_full = K // BLOCK_K
    k_pipe_steps = (n_full // 2) * 2
    k_pipe_pairs = k_pipe_steps // 2
    streamk_tiles = total_tiles % NUM_CU
    total_streamk_units = streamk_tiles * k_pipe_pairs
    # The optimized fixup assumes one resident full-tile wave followed by the
    # distributed tail. Multi-wave full-tile loops currently put the helper's
    # async waits inside an outer warp-pipeline region on the AMD pipeline pass.
    use_streamk = (K == k_pipe_steps * BLOCK_K and total_tiles - streamk_tiles == NUM_CU and streamk_tiles > 0
                   and total_streamk_units >= NUM_CU)
    units_per_program = total_streamk_units // NUM_CU if use_streamk else 0
    remainder_units = total_streamk_units % NUM_CU if use_streamk else 0

    return {
        "HAS_STREAMK": use_streamk,
        "HAS_K_TAIL": K != k_pipe_steps * BLOCK_K,
        "NUM_PROGRAMS": NUM_CU if use_streamk else min(NUM_CU, total_tiles),
        "NUM_FULL_TILES": total_tiles - streamk_tiles if use_streamk else total_tiles,
        "NUM_PID_M": num_pid_m,
        "NUM_PID_N": num_pid_n,
        "K_PIPE_STEPS": k_pipe_steps,
        "K_PIPE_PAIRS": k_pipe_pairs,
        "UNITS_PER_PROGRAM": units_per_program,
        "REMAINDER_UNITS": remainder_units,
    }


@triton.jit
def a16w16_8wave(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    workspace_ptr,
    row_sum_ptr,
    row_sum_sq_ptr,
    M,
    N,
    K,
    KS,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias_m,
    stride_bias_n,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    HAS_REGISTER_TAIL: tl.constexpr,
    USE_I64_A_OFFSETS: tl.constexpr,
    USE_I64_B_OFFSETS: tl.constexpr,
    USE_I64_C_OFFSETS: tl.constexpr,
    UNEVEN_SPLIT_K: tl.constexpr,
    HAS_M_TAIL: tl.constexpr,
    HAS_N_TAIL: tl.constexpr,
    PIN_OFFSET_LAYOUT: tl.constexpr,
    DEFER_EPILOGUE: tl.constexpr,
    WRITE_STATS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # ── Split-K: grid is GRID_MN*SPLIT_K. Peel off split_id, keep the MN pid for
    # the XCD/group remap below. Exact partitions use KS; uneven partitions
    # derive a contiguous whole-K64 range from split_id.
    # We do NOT shift a_ptr/b_ptr (AMD buffer_load builds its resource descriptor
    # from the raw kernel-arg pointer, so an arith'd base fails to lower); instead
    # the split's K byte-offset is folded into the running ka/kb offset (used by
    # every buffer_load) and into the masked-tail addresses. Partials go to a
    # (SPLIT_K*M, N) workspace (row_base=split_id*M); a reduce kernel sums (fp32).
    #
    # On the exact path, KS (per-split K length) is passed as a runtime ARG,
    # not computed as
    # K // SPLIT_K here: the in-kernel divide only proves divisibility 2 for large
    # SPLIT_K (K is known div-16, //8 -> div-2), which collapses the buffer_load
    # offset from the coalesced #linear layout to #blocked and fails to lower. As
    # an arg, KS gets Triton's div-by-16 specialization, so split_id*KS*stride
    # keeps enough divisibility for #linear.
    split_id = tl.program_id(0) // GRID_MN
    pid = tl.program_id(0) % GRID_MN
    if UNEVEN_SPLIT_K:
        full_k_tiles = K // BLOCK_K
        base_k_tiles = full_k_tiles // SPLIT_K
        extra_k_tiles = full_k_tiles % SPLIT_K
        split_start_tile = split_id * base_k_tiles + min(split_id, extra_k_tiles)
        split_k_tiles = base_k_tiles + (split_id < extra_k_tiles)
        split_ks = split_k_tiles * BLOCK_K
        split_start = split_start_tile * BLOCK_K
    else:
        split_ks = KS
        split_start = split_id * KS
    if USE_I64_A_OFFSETS:
        ak_split = split_start.to(tl.int64) * stride_ak
    else:
        ak_split = split_start * stride_ak
    if USE_I64_B_OFFSETS:
        bk_split = split_start.to(tl.int64) * stride_bk
    else:
        bk_split = split_start * stride_bk
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # ── Grid-level scheduling: XCD PID remap + GROUP_SIZE_M swizzle (v9-style) ──
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    if PIN_OFFSET_LAYOUT:
        stride_am = tl.multiple_of(stride_am, 8)
        stride_bn = tl.multiple_of(stride_bn, 8)

    TOP_M: tl.constexpr = 128 if BLOCK_M == 192 else BLOCK_M // 2
    BOTTOM_M: tl.constexpr = BLOCK_M - TOP_M
    HALF_N: tl.constexpr = BLOCK_N // 2

    # Four separate double-buffered LDS allocations — one per operand half-tile.
    # Pin the *swizzled* padded_shared layout (row/col-permuted offset bases) so
    # the ds_reads feeding the MFMAs are bank-conflict-free. The default inferred
    # padded layout ({order, shape})
    # conflicts on CDNA4 (measured 50M SQ_LDS_BANK_CONFLICT vs 0 for this one).
    # Swizzle bases are derived from the half-tile shape (_swz_offset_bases), so the
    # 256x256 (128x64 / 64x128 halves) and thin-N 128x128 (64x64 halves) tiles share
    # one path -- the 64-wide free axis just drops the top bit the 128-wide one adds.
    # TODO(perf): the 64x64 swizzle still shows ~1.5M SQ_LDS_BANK_CONFLICT (10%
    # LDS stall) vs 0 for 128x64. It can't be made conflict-free as a padded layout
    # (direct-to-LDS needs pad interval >=512, but 64x64 lacks a high offset bit for
    # the 4th MFMA row-bit); a swizzled_shared layout is conflict-free but slower
    # (gfx950 has no direct-to-LDS scattering -> extra write swizzle). Net: this
    # padded layout is the fastest option and still beats vendor -- the stall is the
    # price of the cheap direct-to-LDS write on a small square tile.
    a_top_bases: tl.constexpr = (_A_BASES_256 if TOP_M == 128 else _A_BASES_128)
    a_bot_bases: tl.constexpr = (_A_BASES_256 if BOTTOM_M == 128 else _A_BASES_128)
    b_bases: tl.constexpr = _B_BASES_256 if BLOCK_N == 256 else _B_BASES_128
    a_top_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_top_bases,
                                                                              [TOP_M, BLOCK_K])
    a_bot_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_bot_bases,
                                                                              [BOTTOM_M, BLOCK_K])
    b_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_bases, [BLOCK_K, HALF_N])
    smem_a_top = tlx.local_alloc((TOP_M, BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_top_shared)
    smem_a_bot = tlx.local_alloc((BOTTOM_M, BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_bot_shared)
    smem_b_left = tlx.local_alloc((BLOCK_K, HALF_N), tlx.dtype_of(b_ptr), 2, layout=b_shared)
    smem_b_right = tlx.local_alloc((BLOCK_K, HALF_N), tlx.dtype_of(b_ptr), 2, layout=b_shared)

    # The direct-to-LDS buffer_load write is coalesced only when each offset
    # tensor's #linear layout matches the swizzled LDS layout above. We pin only
    # the shared layouts; the matching offset layouts are inferred from them by
    # tlx-insert-require-layout (no explicit offset_layout= needed).
    offs_am = pid_m * BLOCK_M + tl.arange(0, TOP_M)
    if BLOCK_M == 192 or HAS_M_TAIL:
        offs_am_bot = pid_m * BLOCK_M + TOP_M + tl.arange(0, BOTTOM_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Direct-to-LDS vectorizes its address construction before masked zero-fill
    # lowering. Redirect padded edge rows/columns to valid elements so every
    # source address is legal; the corresponding accumulator lanes are later
    # discarded by the output masks. Keep this coordinate work entirely inside
    # constexpr tail branches so complete tiles retain the original address IR.
    if HAS_M_TAIL:
        global_am = tl.where(offs_am < M, offs_am, 0)
        global_am_bot = tl.where(offs_am_bot < M, offs_am_bot, 0)
    if HAS_N_TAIL:
        offs_bn_right = offs_bn + HALF_N
        global_bn = tl.where(offs_bn < N, offs_bn, 0)
        global_bn_right = tl.where(offs_bn_right < N, offs_bn_right, 0)

    # Widen coordinates before multiplying by strides so large tensors cannot
    # overflow while constructing the pointer offset.
    if USE_I64_A_OFFSETS:
        if HAS_M_TAIL:
            a_row_off = global_am.to(tl.int64)[:, None] * stride_am
            a_bot_row_off = global_am_bot.to(tl.int64)[:, None] * stride_am
        else:
            a_row_off = offs_am.to(tl.int64)[:, None] * stride_am
            if BLOCK_M == 192:
                a_bot_row_off = offs_am_bot.to(tl.int64)[:, None] * stride_am
        a_k_off = offs_k.to(tl.int64)[None, :] * stride_ak
    else:
        if HAS_M_TAIL:
            a_row_off = global_am[:, None] * stride_am
            a_bot_row_off = global_am_bot[:, None] * stride_am
        else:
            a_row_off = offs_am[:, None] * stride_am
            if BLOCK_M == 192:
                a_bot_row_off = offs_am_bot[:, None] * stride_am
        a_k_off = offs_k[None, :] * stride_ak
    if USE_I64_B_OFFSETS:
        if HAS_N_TAIL:
            b_col_off = global_bn.to(tl.int64)[None, :] * stride_bn
            b_right_col_off = global_bn_right.to(tl.int64)[None, :] * stride_bn
        else:
            b_col_off = offs_bn.to(tl.int64)[None, :] * stride_bn
        b_k_off = offs_k.to(tl.int64)[:, None] * stride_bk
    else:
        if HAS_N_TAIL:
            b_col_off = global_bn[None, :] * stride_bn
            b_right_col_off = global_bn_right[None, :] * stride_bn
        else:
            b_col_off = offs_bn[None, :] * stride_bn
        b_k_off = offs_k[:, None] * stride_bk
    if PIN_OFFSET_LAYOUT:
        a_row_off = tl.multiple_of(a_row_off, (8, 8))
        b_col_off = tl.multiple_of(b_col_off, (8, 8))
        if BLOCK_M == 192 or HAS_M_TAIL:
            a_bot_row_off = tl.multiple_of(a_bot_row_off, (8, 8))
        if HAS_N_TAIL:
            b_right_col_off = tl.multiple_of(b_right_col_off, (8, 8))
    a_top_off = a_row_off + a_k_off
    if BLOCK_M == 192 or HAS_M_TAIL:
        a_bot_off = a_bot_row_off + a_k_off
    else:
        a_bot_off = a_top_off + TOP_M * stride_am
    b_left_off = b_k_off + b_col_off
    if HAS_N_TAIL:
        b_right_off = b_k_off + b_right_col_off
    else:
        b_right_off = b_left_off + HALF_N * stride_bn
    if PIN_OFFSET_LAYOUT:
        a_top_off = tlx.require_layout(a_top_off, _A_OFFSET_LAYOUT_256)
        a_bot_off = tlx.require_layout(a_bot_off, _A_OFFSET_LAYOUT_256)
        b_left_off = tlx.require_layout(b_left_off, _B_OFFSET_LAYOUT_256)
        b_right_off = tlx.require_layout(b_right_off, _B_OFFSET_LAYOUT_256)
    a_k_mask = offs_k[None, :] < BLOCK_K
    a_top_mask = (offs_am[:, None] < M) & a_k_mask
    if BLOCK_M == 192 or HAS_M_TAIL:
        a_bot_mask = (offs_am_bot[:, None] < M) & a_k_mask
    else:
        a_bot_mask = ((offs_am[:, None] + TOP_M) < M) & a_k_mask
    b_left_mask = tl.broadcast_to(offs_bn[None, :] < N, b_left_off.shape)
    if HAS_N_TAIL:
        b_right_mask = tl.broadcast_to(offs_bn_right[None, :] < N, b_right_off.shape)
    else:
        b_right_mask = tl.broadcast_to((offs_bn[None, :] + HALF_N) < N, b_right_off.shape)

    # Keep this pipeline inline: its K-contiguous B producer layout is inferred
    # together with the bank-conflict-free LDS layout. Moving it through a JIT
    # helper boundary loses that relationship on current layout propagation.
    a_top_off_n = a_top_off + BLOCK_K * stride_ak
    a_bot_off_n = a_bot_off + BLOCK_K * stride_ak
    b_left_off_n = b_left_off + BLOCK_K * stride_bk
    b_right_off_n = b_right_off + BLOCK_K * stride_bk

    ka = ak_split
    kb = bk_split

    acc_tl = tl.zeros((TOP_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((BOTTOM_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((TOP_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((BOTTOM_M, HALF_N), dtype=tl.float32)

    # The pipeline consumes K in pairs of BLOCK_K tiles (prologue prefetches 2,
    # the loop 2/iter, the epilogue drains 2), so it covers only an EVEN number of
    # whole K-tiles: n_pipe. Any leftover -- an odd whole tile and/or a partial
    # final tile (K not a multiple of BLOCK_K) -- is handled by the masked scalar
    # tail after the epilogue.
    n_full = split_ks // BLOCK_K
    n_pipe = (n_full // 2) * 2

    tlx.async_load(b_ptr + b_left_off + kb, smem_b_left[0], mask=b_left_mask, other=0.0)
    tlx.async_load_commit_group()
    tlx.async_load(a_ptr + a_top_off + ka, smem_a_top[0], mask=a_top_mask, other=0.0)
    tlx.async_load_commit_group()
    tlx.async_load(a_ptr + a_bot_off + ka, smem_a_bot[0], mask=a_bot_mask, other=0.0)
    tlx.async_load_commit_group()
    tlx.async_load(b_ptr + b_right_off + kb, smem_b_right[0], mask=b_right_mask, other=0.0)
    tlx.async_load_commit_group()

    tlx.async_load(b_ptr + b_left_off_n + kb, smem_b_left[1], mask=b_left_mask, other=0.0)
    tlx.async_load_commit_group()
    tlx.async_load(a_ptr + a_top_off_n + ka, smem_a_top[1], mask=a_top_mask, other=0.0)
    tlx.async_load_commit_group()
    tlx.async_load(a_ptr + a_bot_off_n + ka, smem_a_bot[1], mask=a_bot_mask, other=0.0)
    tlx.async_load_commit_group()
    tlx.async_load(b_ptr + b_right_off_n + kb, smem_b_right[1], mask=b_right_mask, other=0.0)
    tlx.async_load_commit_group()

    ka += BLOCK_K * stride_ak * 2
    kb += BLOCK_K * stride_bk * 2

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(smem_b_left[0], relaxed=True)
    a_top = tlx.local_load(smem_a_top[0], relaxed=True)

    for k in tl.range(0, n_pipe - 2, 2, num_stages=1):
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
            tlx.async_load(b_ptr + b_left_off + kb, smem_b_left[0], mask=b_left_mask, other=0.0)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[0], relaxed=True)
            tlx.async_load(a_ptr + a_top_off + ka, smem_a_top[0], mask=a_top_mask, other=0.0)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[1], relaxed=True)
            tlx.async_load(a_ptr + a_bot_off + ka, smem_a_bot[0], mask=a_bot_mask, other=0.0)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[1], relaxed=True)
            tlx.async_load(b_ptr + b_right_off + kb, smem_b_right[0], mask=b_right_mask, other=0.0)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
            tlx.async_load(b_ptr + b_left_off_n + kb, smem_b_left[1], mask=b_left_mask, other=0.0)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[1], relaxed=True)
            tlx.async_load(a_ptr + a_top_off_n + ka, smem_a_top[1], mask=a_top_mask, other=0.0)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[0], relaxed=True)
            tlx.async_load(a_ptr + a_bot_off_n + ka, smem_a_bot[1], mask=a_bot_mask, other=0.0)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[0], relaxed=True)
            tlx.async_load(b_ptr + b_right_off_n + kb, smem_b_right[1], mask=b_right_mask, other=0.0)
            tlx.async_load_commit_group()
            ka += BLOCK_K * stride_ak * 2
            kb += BLOCK_K * stride_bk * 2

    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(5)
    l_idx: tl.constexpr = 0
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, l_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    tlx.async_load_wait_group(3)
    g_idx: tl.constexpr = 1
    b_left = tlx.local_load(tlx.local_view(smem_b_left, g_idx), relaxed=True)

    acc_br = tl.dot(a_bot, b_right, acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)

    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, g_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    acc_br = tl.dot(a_bot, b_right, acc_br)

    # ── Masked scalar tail: K columns past the pipelined region (an odd leftover
    # tile and/or a partial final tile). Plain masked tl.load + tl.dot -- no LDS,
    # no pipeline. The K-mask zeros the missing contraction elements (they add 0
    # to C = sum_k A*B), so this is correct for arbitrary K. Runs 0-2 iterations;
    # the whole-tile even hot path (n_pipe*BLOCK_K == K) skips it entirely.
    if HAS_REGISTER_TAIL:
        offs_am_bot = (pid_m * BLOCK_M + TOP_M + tl.arange(0, BOTTOM_M))
        offs_bn_right = offs_bn + HALF_N
        for kk in tl.range(
                n_pipe * BLOCK_K,
                split_ks,
                BLOCK_K,
                num_stages=1,
        ):
            offs_kt = kk + offs_k
            k_mask = offs_kt < split_ks
            a_top_t = tl.load(a_ptr + ak_split + offs_am[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                              mask=(offs_am[:, None] < M) & k_mask[None, :], other=0.0)
            a_bot_t = tl.load(a_ptr + ak_split + offs_am_bot[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                              mask=(offs_am_bot[:, None] < M) & k_mask[None, :], other=0.0)
            b_left_t = tl.load(b_ptr + bk_split + offs_kt[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                               mask=k_mask[:, None] & (offs_bn[None, :] < N), other=0.0)
            b_right_t = tl.load(b_ptr + bk_split + offs_kt[:, None] * stride_bk + offs_bn_right[None, :] * stride_bn,
                                mask=k_mask[:, None] & (offs_bn_right[None, :] < N), other=0.0)
            acc_tl = tl.dot(a_top_t, b_left_t, acc_tl)
            acc_bl = tl.dot(a_bot_t, b_left_t, acc_bl)
            acc_tr = tl.dot(a_top_t, b_right_t, acc_tr)
            acc_br = tl.dot(a_bot_t, b_right_t, acc_br)

    offs_cm_top = pid_m * BLOCK_M + tl.arange(0, TOP_M)
    offs_cm_bot = (pid_m * BLOCK_M + TOP_M + tl.arange(0, BOTTOM_M))
    offs_cn_left = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_cn_right = offs_cn_left + HALF_N
    m_top = offs_cm_top[:, None] < M
    m_bot = offs_cm_bot[:, None] < M
    n_left = offs_cn_left[None, :] < N
    n_right = offs_cn_right[None, :] < N
    if USE_I64_C_OFFSETS:
        c_row_top = offs_cm_top.to(tl.int64)[:, None] * stride_cm
        c_row_bot = offs_cm_bot.to(tl.int64)[:, None] * stride_cm
        c_col_left = offs_cn_left.to(tl.int64)[None, :] * stride_cn
        c_col_right = offs_cn_right.to(tl.int64)[None, :] * stride_cn
    else:
        c_row_top = offs_cm_top[:, None] * stride_cm
        c_row_bot = offs_cm_bot[:, None] * stride_cm
        c_col_left = offs_cn_left[None, :] * stride_cn
        c_col_right = offs_cn_right[None, :] * stride_cn
    c_top_left = c_row_top + c_col_left
    c_bot_left = c_row_bot + c_col_left
    c_top_right = c_row_top + c_col_right
    c_bot_right = c_row_bot + c_col_right

    if SPLIT_K == 1 and not DEFER_EPILOGUE:
        if ADD_BIAS:
            acc_tl += tl.load(
                bias_ptr + stride_bias_m * offs_cm_top[:, None] + stride_bias_n * offs_cn_left[None, :],
                mask=m_top & n_left,
                other=0.0,
            ).to(tl.float32)
            acc_bl += tl.load(
                bias_ptr + stride_bias_m * offs_cm_bot[:, None] + stride_bias_n * offs_cn_left[None, :],
                mask=m_bot & n_left,
                other=0.0,
            ).to(tl.float32)
            acc_tr += tl.load(
                bias_ptr + stride_bias_m * offs_cm_top[:, None] + stride_bias_n * offs_cn_right[None, :],
                mask=m_top & n_right,
                other=0.0,
            ).to(tl.float32)
            acc_br += tl.load(
                bias_ptr + stride_bias_m * offs_cm_bot[:, None] + stride_bias_n * offs_cn_right[None, :],
                mask=m_bot & n_right,
                other=0.0,
            ).to(tl.float32)

        et = c_ptr.dtype.element_ty
        c_tl = acc_tl.to(et)
        c_bl = acc_bl.to(et)
        c_tr = acc_tr.to(et)
        c_br = acc_br.to(et)
        if WRITE_STATS:
            top_left = tl.where(m_top & n_left, c_tl.to(tl.float32), 0.0)
            bot_left = tl.where(m_bot & n_left, c_bl.to(tl.float32), 0.0)
            top_right = tl.where(m_top & n_right, c_tr.to(tl.float32), 0.0)
            bot_right = tl.where(m_bot & n_right, c_br.to(tl.float32), 0.0)
            top_sum_sq = tl.sum(top_left * top_left, axis=1) + tl.sum(
                top_right * top_right,
                axis=1,
            )
            bot_sum_sq = tl.sum(bot_left * bot_left, axis=1) + tl.sum(
                bot_right * bot_right,
                axis=1,
            )
            top_stats_offsets = offs_cm_top * num_pid_n + pid_n
            bot_stats_offsets = offs_cm_bot * num_pid_n + pid_n
            if not IS_RMS_NORM:
                tl.store(
                    row_sum_ptr + top_stats_offsets,
                    tl.sum(top_left, axis=1) + tl.sum(top_right, axis=1),
                    mask=offs_cm_top < M,
                )
                tl.store(
                    row_sum_ptr + bot_stats_offsets,
                    tl.sum(bot_left, axis=1) + tl.sum(bot_right, axis=1),
                    mask=offs_cm_bot < M,
                )
            tl.store(
                row_sum_sq_ptr + top_stats_offsets,
                top_sum_sq,
                mask=offs_cm_top < M,
            )
            tl.store(
                row_sum_sq_ptr + bot_stats_offsets,
                bot_sum_sq,
                mask=offs_cm_bot < M,
            )

        # Direct store to C.
        if TOP_M == 128 and BOTTOM_M == 128 and HALF_N == 128:
            # Stop the wide epilogue-store layout from propagating backward
            # through the extracted tile function. Split-K and the 128 tile keep
            # the original inferred accumulator layout.
            acc_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                           warps_per_cta=[2, 4])
            acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
            acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
            acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
            acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)
            # Pin each 128x128 quadrant to the coalesced SIMD #linear layout (no LDS
            # staging) so OptimizeEpilogue keeps the wide dwordx4 store.
            # _C_STORE_SIMD_LAYOUT is derived for the 128x128 quadrant, so it only
            # applies to the 256x256 tile; a smaller tile (64x64 quadrant) uses a
            # plain store.
            L: tl.constexpr = tlx.layout(
                shape=((16, 4, 8), (8, 4)),
                stride=((8, 128, 512), (1, 4096)),
            )
            c_tl = tlx.require_layout(c_tl, L)
            # Static guard (no device code): the pin must survive coalesce /
            # remove-layout-conversions / AMD optimize-epilogue so the store stays
            # a wide dwordx4. Fails compilation if a future change drops the pin.
            tlx.assert_same_layout(c_tl, L)
            tl.store(c_ptr + c_top_left, c_tl, mask=m_top & n_left)
            tl.store(c_ptr + c_bot_left, tlx.require_layout(c_bl, L), mask=m_bot & n_left)
            tl.store(c_ptr + c_top_right, tlx.require_layout(c_tr, L), mask=m_top & n_right)
            tl.store(c_ptr + c_bot_right, tlx.require_layout(c_br, L), mask=m_bot & n_right)
        else:
            tl.store(c_ptr + c_top_left, c_tl, mask=m_top & n_left)
            tl.store(c_ptr + c_bot_left, c_bl, mask=m_bot & n_left)
            tl.store(c_ptr + c_top_right, c_tr, mask=m_top & n_right)
            tl.store(c_ptr + c_bot_right, c_br, mask=m_bot & n_right)
    else:
        # Split-K: every split writes its fp32 partial into its workspace slice
        # (rows [split_id*M, split_id*M+M)). Mask stays in relative-M coords; the
        # row offset is added only to the store index.
        rb = split_id * M
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_top)[:, None] + stride_cn * offs_cn_left[None, :], acc_tl,
                 mask=m_top & n_left)
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_bot)[:, None] + stride_cn * offs_cn_left[None, :], acc_bl,
                 mask=m_bot & n_left)
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_top)[:, None] + stride_cn * offs_cn_right[None, :], acc_tr,
                 mask=m_top & n_right)
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_bot)[:, None] + stride_cn * offs_cn_right[None, :], acc_br,
                 mask=m_bot & n_right)


@triton.jit
def _matmul_full_tile(a_ptr, b_ptr, c_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, pid_m, pid_n, K,
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K_PIPE_STEPS: tl.constexpr,
                      HAS_K_TAIL: tl.constexpr, c_layout: tl.constexpr):
    """Compute and store one complete output tile."""
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2
    offs_m = tl.arange(0, HALF_M)
    offs_n = tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m_top = pid_m * BLOCK_M + offs_m
    offs_m_bot = offs_m_top + HALF_M
    offs_n_left = pid_n * BLOCK_N + offs_n
    offs_n_right = offs_n_left + HALF_N
    a_top_off = offs_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_bot_off = offs_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_left_off = offs_k[:, None] * stride_bk + offs_n_left[None, :] * stride_bn
    b_right_off = offs_k[:, None] * stride_bk + offs_n_right[None, :] * stride_bn
    acc_tl, acc_bl, acc_tr, acc_br = matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right,
                                                 a_top_off, a_bot_off, b_left_off, b_right_off, 0, 0, K_PIPE_STEPS,
                                                 stride_ak, stride_bk, BLOCK_M, BLOCK_N, BLOCK_K)
    if HAS_K_TAIL:
        # Mask odd full and/or partial K64 steps left after the even pipelined prefix.
        for kk in tl.range(K_PIPE_STEPS * BLOCK_K, K, BLOCK_K, num_stages=1):
            offs_kt = kk + offs_k
            k_mask = offs_kt < K
            a_top = tl.load(a_ptr + offs_m_top[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                            mask=k_mask[None, :], other=0.0)
            a_bot = tl.load(a_ptr + offs_m_bot[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                            mask=k_mask[None, :], other=0.0)
            b_left = tl.load(b_ptr + offs_kt[:, None] * stride_bk + offs_n_left[None, :] * stride_bn,
                             mask=k_mask[:, None], other=0.0)
            b_right = tl.load(b_ptr + offs_kt[:, None] * stride_bk + offs_n_right[None, :] * stride_bn,
                              mask=k_mask[:, None], other=0.0)
            acc_tl = tl.dot(a_top, b_left, acc_tl)
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
            acc_tr = tl.dot(a_top, b_right, acc_tr)
            acc_br = tl.dot(a_bot, b_right, acc_br)
    et: tl.constexpr = c_ptr.dtype.element_ty
    tl.store(c_ptr + offs_m_top[:, None] * stride_cm + offs_n_left[None, :] * stride_cn,
             tlx.require_layout(acc_tl.to(et), c_layout))
    tl.store(c_ptr + offs_m_bot[:, None] * stride_cm + offs_n_left[None, :] * stride_cn,
             tlx.require_layout(acc_bl.to(et), c_layout))
    tl.store(c_ptr + offs_m_top[:, None] * stride_cm + offs_n_right[None, :] * stride_cn,
             tlx.require_layout(acc_tr.to(et), c_layout))
    tl.store(c_ptr + offs_m_bot[:, None] * stride_cm + offs_n_right[None, :] * stride_cn,
             tlx.require_layout(acc_br.to(et), c_layout))


@triton.jit
def _grouped_tile_coords(tile_id, NUM_PID_M: tl.constexpr, NUM_PID_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Map a linear tile ID to the grouped M-major output grid."""
    tiles_per_group: tl.constexpr = GROUP_SIZE_M * NUM_PID_N
    group_id = tile_id // tiles_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_PID_M - first_pid_m, GROUP_SIZE_M)
    tile_in_group = tile_id % tiles_per_group
    pid_m = first_pid_m + tile_in_group % group_size_m
    pid_n = tile_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _wait_for_streamk_partial(locks_ptr, slot, ready_value):
    """Wait until one producer has published its partial tile."""
    while tl.load(locks_ptr + slot, cache_modifier=".cv", volatile=True) != ready_value:
        pass


@triton.jit
def _reduce_and_store_streamk_quadrant(
    partials_ptr,
    c_ptrs,
    first_contributor,
    num_contributors: tl.constexpr,
    partial_off,
    tile_elems: tl.constexpr,
    acc_layout: tl.constexpr,
    c_layout: tl.constexpr,
):
    """Reduce one quadrant across a tile's contributors and store it."""
    acc = tlx.require_layout(tl.load(partials_ptr + first_contributor * tile_elems + partial_off, cache_modifier=".cv"),
                             acc_layout, pin=False)
    for peer in range(1, num_contributors):
        acc += tlx.require_layout(
            tl.load(partials_ptr + (first_contributor + peer) * tile_elems + partial_off, cache_modifier=".cv"),
            acc_layout, pin=False)
    tl.store(c_ptrs, tlx.require_layout(acc.to(c_ptrs.dtype.element_ty), c_layout))


@triton.jit
def streamk_kernel(a_ptr, b_ptr, c_ptr, partials_ptr, locks_ptr, ready_value, K, stride_am, stride_ak, stride_bk,
                   stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   NUM_XCDS: tl.constexpr, NUM_CU: tl.constexpr, NUM_PROGRAMS: tl.constexpr, HAS_STREAMK: tl.constexpr,
                   NUM_FULL_TILES: tl.constexpr, HAS_K_TAIL: tl.constexpr, NUM_PID_M: tl.constexpr,
                   NUM_PID_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, K_PIPE_STEPS: tl.constexpr,
                   K_PIPE_PAIRS: tl.constexpr, UNITS_PER_PROGRAM: tl.constexpr, REMAINDER_UNITS: tl.constexpr):
    """Persistent full tiles plus owner or distributed Stream-K fixup."""
    pid = tl.program_id(0)
    contributors_per_tile: tl.constexpr = K_PIPE_PAIRS // max(UNITS_PER_PROGRAM, 1)
    # When every flattened interval is one equal segment of one tile, all
    # contributors can participate in fixup instead of serializing it in the
    # tile owner. This is a reduction optimization, not a separate schedule.
    DISTRIBUTED_FIXUP: tl.constexpr = (HAS_STREAMK and NUM_FULL_TILES == NUM_PROGRAMS and REMAINDER_UNITS == 0
                                       and K_PIPE_PAIRS % max(UNITS_PER_PROGRAM, 1) == 0
                                       and (contributors_per_tile == 2 or contributors_per_tile == 4))
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                   warps_per_cta=[2, 4])
    a_bases: tl.constexpr = _A_BASES_256 if BLOCK_M == 256 else _A_BASES_128
    b_bases: tl.constexpr = _B_BASES_256 if BLOCK_N == 256 else _B_BASES_128
    a_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_bases, [HALF_M, BLOCK_K])
    b_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_bases, [BLOCK_K, HALF_N])
    et: tl.constexpr = a_ptr.dtype.element_ty
    smem_a_top = tlx.local_alloc((HALF_M, BLOCK_K), et, 2, layout=a_layout)
    smem_a_bot = tlx.local_alloc((HALF_M, BLOCK_K), et, 2, layout=a_layout)
    smem_b_left = tlx.local_alloc((BLOCK_K, HALF_N), et, 2, layout=b_layout)
    smem_b_right = tlx.local_alloc((BLOCK_K, HALF_N), et, 2, layout=b_layout)
    offs_m = tl.arange(0, HALF_M)
    offs_n = tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)
    C: tl.constexpr = _C_STORE_SIMD_LAYOUT if BLOCK_M == 256 and BLOCK_N == 256 else acc_layout
    tile_elems: tl.constexpr = BLOCK_M * BLOCK_N
    partial_tl_off = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    partial_bl_off = partial_tl_off + HALF_M * BLOCK_N
    partial_tr_off = partial_tl_off + HALF_N
    partial_br_off = partial_bl_off + HALF_N
    stream_pid = pid
    if HAS_STREAMK:
        stream_pid = (pid % NUM_XCDS) * (NUM_CU // NUM_XCDS) + pid // NUM_XCDS
        # Fuse TritonBLAS-style lock initialization into the resident kernel;
        # each program clears the slot it may later publish.
        tl.store(locks_ptr + stream_pid, 0, cache_modifier=".wt")
        tl.debug_barrier()

    if HAS_STREAMK and NUM_FULL_TILES == NUM_PROGRAMS:
        # Fast path: each Stream-K program owns one full tile, so no loop is needed.
        head_pid_m = stream_pid % NUM_PID_M
        head_pid_n = stream_pid // NUM_PID_M
        _matmul_full_tile(a_ptr, b_ptr, c_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, head_pid_m,
                          head_pid_n, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M,
                          BLOCK_N, BLOCK_K, K_PIPE_STEPS, HAS_K_TAIL, C)
    else:
        # General path for both persistent and generic Stream-K.
        pids_per_xcd: tl.constexpr = (NUM_FULL_TILES + NUM_XCDS - 1) // NUM_XCDS
        remainder_xcds: tl.constexpr = NUM_FULL_TILES % NUM_XCDS
        tall_xcds: tl.constexpr = NUM_XCDS if remainder_xcds == 0 else remainder_xcds
        for virtual_pid in range(pid, NUM_FULL_TILES, NUM_PROGRAMS):
            xcd = virtual_pid % NUM_XCDS
            local_pid = virtual_pid // NUM_XCDS
            if xcd < tall_xcds:
                tile_id = xcd * pids_per_xcd + local_pid
            else:
                tile_id = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
            pid_m, pid_n = _grouped_tile_coords(tile_id, NUM_PID_M, NUM_PID_N, GROUP_SIZE_M)
            _matmul_full_tile(a_ptr, b_ptr, c_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, pid_m, pid_n, K,
                              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N,
                              BLOCK_K, K_PIPE_STEPS, HAS_K_TAIL, C)
    if not HAS_STREAMK:
        return

    if DISTRIBUTED_FIXUP:
        # Unlike owner-based (standard Stream-K) fixup, every contributor publishes
        # its partial tile, synchronizes, and reduces a disjoint output region.
        # This lets all contributing CTAs actively reduce instead of relying on a
        # single owner CTA, as in standard Stream-K.
        contributor_id = stream_pid % contributors_per_tile
        tail_tile = NUM_FULL_TILES + stream_pid // contributors_per_tile

        tail_pid_m, tail_pid_n = _grouped_tile_coords(tail_tile, NUM_PID_M, NUM_PID_N, GROUP_SIZE_M)
        tail_offs_m_top = tail_pid_m * BLOCK_M + offs_m
        tail_offs_m_bot = tail_offs_m_top + HALF_M
        tail_offs_n_left = tail_pid_n * BLOCK_N + offs_n
        tail_offs_n_right = tail_offs_n_left + HALF_N
        tail_a_top_off = tail_offs_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
        tail_a_bot_off = tail_offs_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
        tail_b_left_off = offs_k[:, None] * stride_bk + tail_offs_n_left[None, :] * stride_bn
        tail_b_right_off = offs_k[:, None] * stride_bk + tail_offs_n_right[None, :] * stride_bn
        segment_k_steps: tl.constexpr = UNITS_PER_PROGRAM * 2
        segment_k_offset = contributor_id * segment_k_steps * BLOCK_K
        acc_tl, acc_bl, acc_tr, acc_br = matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right,
                                                     tail_a_top_off, tail_a_bot_off, tail_b_left_off, tail_b_right_off,
                                                     segment_k_offset * stride_ak, segment_k_offset * stride_bk,
                                                     segment_k_steps, stride_ak, stride_bk, BLOCK_M, BLOCK_N, BLOCK_K)

        # Pin and publish all four partial quadrants before any contributor waits.
        # This avoids cyclic dependencies and lets the MFMA accumulators die before
        # fixup. Publishing a runtime-selected quadrant instead costs more VGPR and
        # select work than it saves in workspace traffic on gfx950.
        acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
        acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
        acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
        acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)
        partial_base = partials_ptr + stream_pid * tile_elems
        partial_tl_ptr = tlx.require_layout(partial_base + partial_tl_off, acc_layout, pin=False)
        partial_bl_ptr = tlx.require_layout(partial_base + partial_bl_off, acc_layout, pin=False)
        partial_tr_ptr = tlx.require_layout(partial_base + partial_tr_off, acc_layout, pin=False)
        partial_br_ptr = tlx.require_layout(partial_base + partial_br_off, acc_layout, pin=False)
        tl.store(partial_tl_ptr, acc_tl, cache_modifier=".wt")
        tl.store(partial_bl_ptr, acc_bl, cache_modifier=".wt")
        tl.store(partial_tr_ptr, acc_tr, cache_modifier=".wt")
        tl.store(partial_br_ptr, acc_br, cache_modifier=".wt")
        tl.debug_barrier()
        tl.store(locks_ptr + stream_pid, ready_value, cache_modifier=".wt")

        # Cooperatively reduce the tile, assigning consecutive output quadrants to
        # each contributor so all programs remain useful during fixup.
        first_contributor = stream_pid - contributor_id
        for peer in range(contributors_per_tile):
            _wait_for_streamk_partial(locks_ptr, first_contributor + peer, ready_value)
        # Two contributors own the left and right halves; four contributors
        # own one quadrant each.
        bottom_left_owner: tl.constexpr = 0 if contributors_per_tile == 2 else 1
        top_right_owner: tl.constexpr = 1 if contributors_per_tile == 2 else 2
        bottom_right_owner: tl.constexpr = 1 if contributors_per_tile == 2 else 3
        if contributor_id == 0:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_top[:, None] * stride_cm + tail_offs_n_left[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_tl_off, tile_elems, acc_layout, C)
        if contributor_id == bottom_left_owner:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_bot[:, None] * stride_cm + tail_offs_n_left[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_bl_off, tile_elems, acc_layout, C)
        if contributor_id == top_right_owner:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_top[:, None] * stride_cm + tail_offs_n_right[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_tr_off, tile_elems, acc_layout, C)
        if contributor_id == bottom_right_owner:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_bot[:, None] * stride_cm + tail_offs_n_right[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_br_off, tile_elems, acc_layout, C)
    else:
        # Standard Stream-K fixup path: Divide remaining flattened pairs of K
        # pipeline steps almost evenly across CUs.
        logical_pid = stream_pid
        streamk_base = NUM_FULL_TILES * K_PIPE_PAIRS
        start_unit = (streamk_base + logical_pid * UNITS_PER_PROGRAM + min(logical_pid, REMAINDER_UNITS))
        last_unit = (streamk_base + (logical_pid + 1) * UNITS_PER_PROGRAM + min(logical_pid + 1, REMAINDER_UNITS))

        while start_unit < last_unit:
            tile_id = start_unit // K_PIPE_PAIRS
            tile_start = tile_id * K_PIPE_PAIRS
            tile_end = tile_start + K_PIPE_PAIRS
            segment_end = min(last_unit, tile_end)
            pid_m, pid_n = _grouped_tile_coords(tile_id, NUM_PID_M, NUM_PID_N, GROUP_SIZE_M)
            tile_offs_m_top = pid_m * BLOCK_M + offs_m
            tile_offs_m_bot = tile_offs_m_top + HALF_M
            tile_offs_n_left = pid_n * BLOCK_N + offs_n
            tile_offs_n_right = tile_offs_n_left + HALF_N
            tile_a_top_off = tile_offs_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
            tile_a_bot_off = tile_offs_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
            tile_b_left_off = offs_k[:, None] * stride_bk + tile_offs_n_left[None, :] * stride_bn
            tile_b_right_off = offs_k[:, None] * stride_bk + tile_offs_n_right[None, :] * stride_bn
            k_step = (start_unit - tile_start) * 2 * BLOCK_K
            acc_tl, acc_bl, acc_tr, acc_br = matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left,
                                                         smem_b_right, tile_a_top_off, tile_a_bot_off, tile_b_left_off,
                                                         tile_b_right_off, k_step * stride_ak, k_step * stride_bk,
                                                         (segment_end - start_unit) * 2, stride_ak, stride_bk, BLOCK_M,
                                                         BLOCK_N, BLOCK_K)
            acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
            acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
            acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
            acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)

            if start_unit != tile_start:
                # A contributor publishes one partial. Its range can then start
                # the next tile, so continue instead of returning immediately.
                base = logical_pid * tile_elems
                partial_tl_ptr = tlx.require_layout(partials_ptr + base + partial_tl_off, acc_layout, pin=False)
                partial_bl_ptr = tlx.require_layout(partials_ptr + base + partial_bl_off, acc_layout, pin=False)
                partial_tr_ptr = tlx.require_layout(partials_ptr + base + partial_tr_off, acc_layout, pin=False)
                partial_br_ptr = tlx.require_layout(partials_ptr + base + partial_br_off, acc_layout, pin=False)
                tl.store(partial_tl_ptr, acc_tl, cache_modifier=".wt")
                tl.store(partial_bl_ptr, acc_bl, cache_modifier=".wt")
                tl.store(partial_tr_ptr, acc_tr, cache_modifier=".wt")
                tl.store(partial_br_ptr, acc_br, cache_modifier=".wt")
                tl.debug_barrier()
                tl.store(locks_ptr + logical_pid, ready_value, cache_modifier=".wt")
            else:
                # The program owning the first work unit of a tile collects the
                # following contributors, exactly as the TritonBLAS fixup does.
                covered_end = segment_end
                next_pid = logical_pid + 1
                while covered_end < tile_end:
                    _wait_for_streamk_partial(locks_ptr, next_pid, ready_value)
                    peer_base = next_pid * tile_elems
                    acc_tl += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_tl_off, cache_modifier=".cv"), acc_layout, pin=False)
                    acc_bl += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_bl_off, cache_modifier=".cv"), acc_layout, pin=False)
                    acc_tr += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_tr_off, cache_modifier=".cv"), acc_layout, pin=False)
                    acc_br += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_br_off, cache_modifier=".cv"), acc_layout, pin=False)
                    covered_end += UNITS_PER_PROGRAM + (next_pid < REMAINDER_UNITS)
                    next_pid += 1
                tl.store(c_ptr + tile_offs_m_top[:, None] * stride_cm + tile_offs_n_left[None, :] * stride_cn,
                         tlx.require_layout(acc_tl.to(et), C))
                tl.store(c_ptr + tile_offs_m_bot[:, None] * stride_cm + tile_offs_n_left[None, :] * stride_cn,
                         tlx.require_layout(acc_bl.to(et), C))
                tl.store(c_ptr + tile_offs_m_top[:, None] * stride_cm + tile_offs_n_right[None, :] * stride_cn,
                         tlx.require_layout(acc_tr.to(et), C))
                tl.store(c_ptr + tile_offs_m_bot[:, None] * stride_cm + tile_offs_n_right[None, :] * stride_cn,
                         tlx.require_layout(acc_br.to(et), C))
            start_unit = segment_end


_TORCH_TO_TL = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


@triton.jit
def _reduce_k_kernel(workspace_ptr, bias_ptr, c_ptr, M, N, stride_bias_m, stride_bias_n, stride_cm, stride_cn,
                     SPLIT_K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                     OUTPUT_DTYPE: tl.constexpr, ADD_BIAS: tl.constexpr):
    # Sum the SPLIT_K partials (each a contiguous (M, N) slab in workspace) into
    # C with fp32 accumulation. Small tiles (32x32) so small outputs still spawn
    # many CTAs -- else the reduce is CTA-starved and dominates (D97513062).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        partial = tl.load(workspace_ptr + base_offs + s * M * N, mask=mask, other=0.0)
        acc += partial.to(tl.float32)
    if ADD_BIAS:
        bias = tl.load(
            bias_ptr + offs_m[:, None] * stride_bias_m + offs_n[None, :] * stride_bias_n,
            mask=mask,
            other=0.0,
        )
        acc += bias.to(tl.float32)
    output_offsets = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr + output_offsets, acc.to(OUTPUT_DTYPE), mask=mask)


NUM_CU = 256  # gfx950 (CDNA4) compute units
# Minimum K-tiles per split. Two forces set this floor: (1) below it the per-split
# prologue/epilogue overhead dominates the shrinking K work; (2) more splits means
# a proportionally larger fp32 workspace for the reduce to stream back (reduce cost
# ~ SPLIT_K*M*N), so an over-split that only marginally improves the GEMM loses the
# gain to the reduce. Every measured production optimum uses >= 16 tiles/split
# (e.g. K=12288 wants SPLIT_K=12 (16 tiles) not 16 (12 tiles); the latter fills the
# CUs but its extra reduce traffic makes it net slower).
MIN_KTILES_PER_SPLIT = 16

# Tile candidates, largest first. The big tile is the tuned default; the smaller
# one is used only when the big tile can't fill the CUs (see choose_tile).
# choose_tile scans the fallbacks generically, so adding another tile here (e.g.
# (64, 64)) needs no logic change; today only the 128x128 fallback is used.
TILE_CANDIDATES = ((256, 256), (128, 128))


def _lds_split_k_for(grid_mn, K):
    """Largest SPLIT_K keeping grid_mn*SK within one CU wave and each split a whole,
    BLOCK_K-aligned chunk of >= MIN_KTILES_PER_SPLIT tiles.

    All divisors of K are considered, not just powers of two: for K with odd factors
    (e.g. 22272 = 64*348) a non-pow2 SPLIT_K divides K and fills the CUs far more
    precisely than the nearest pow2 (SPLIT_K=12 -> 192 CUs vs SPLIT_K=4 -> 64 CUs).
    The scan is <= NUM_CU/grid_mn (~20) iterations, negligible at compile time."""
    min_ks = MIN_KTILES_PER_SPLIT * BLOCK_K
    best = 1
    for sk in range(2, NUM_CU // grid_mn + 1):  # grid_mn*sk <= NUM_CU
        ks = K // sk
        if K % sk == 0 and ks >= min_ks and ks % BLOCK_K == 0:
            best = sk  # fill = grid_mn*sk grows with sk, so the last valid sk wins
    return best


def _fill_with_uneven_split_k(grid_mn, K, split_k):
    """Increase split_k with whole-K64, unevenly sized partitions."""
    exact_fill = grid_mn * split_k
    if (K % BLOCK_K != 0 or exact_fill >= NUM_CU or exact_fill * 4 >= NUM_CU * 3):
        return split_k
    k_tiles = K // BLOCK_K
    max_split = min(
        NUM_CU // grid_mn,
        k_tiles // MIN_KTILES_PER_SPLIT,
    )
    return max(split_k, max_split)


@lru_cache(maxsize=None)
def _lds_plan_for_shape(M, N, K):
    """Pick (BLOCK_M, BLOCK_N, SPLIT_K) by CU fill -- no shape hardcoding.

    Prefer the tuned 256x256 tile; it is more MFMA-efficient per work-group than the
    128x128 tile. Fall back to the smaller tile only when the 256 grid leaves most of
    the machine idle even after split-K (fill < NUM_CU/2) -- the genuinely thin-N /
    small-tile-count shapes (e.g. N=256, gmn=8): the 4x-denser MN grid then reaches
    occupancy the big tile can't. When the big tile fills at least half the CUs, its
    efficiency beats a full grid of small tiles, so it is kept (comparing raw
    work-group counts across tile sizes is apples-to-oranges -- a 128 tile does 1/4
    the work -- so a bigger small-tile count does not mean it is faster)."""
    bm, bn = TILE_CANDIDATES[0]
    gmn = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    sk = _lds_split_k_for(gmn, K)
    best_fill = gmn * sk
    if best_fill < NUM_CU // 2:  # big tile leaves most CUs idle even with split-K
        for cbm, cbn in TILE_CANDIDATES[1:]:
            g = triton.cdiv(M, cbm) * triton.cdiv(N, cbn)
            s = _lds_split_k_for(g, K)
            if g * s > best_fill:  # smaller tile fills the machine better
                bm, bn, sk, best_fill = cbm, cbn, s, g * s
    grid_mn = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    sk = _fill_with_uneven_split_k(grid_mn, K, sk)
    return bm, bn, sk


def choose_split_k(M, N, K):
    """Back-compat: SPLIT_K for the auto-chosen tile."""
    return _lds_plan_for_shape(M, N, K)[2]


def _needs_i64_offsets(tensor):
    """Return whether this view can address beyond signed i32 byte offsets."""
    if any(stride < 0 for stride in tensor.stride()):
        return True
    max_element_offset = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    max_byte_offset = max_element_offset * tensor.element_size()
    return max_byte_offset > (1 << 31) - 1


_STRONG_LDS_PLANS = {
    (677, 4096, 8192): (192, 256, 4),
}


def _strong_lds_plan(M, N, K):
    return _STRONG_LDS_PLANS.get((M, N, K))


@lru_cache(maxsize=None)
def _matmul_plan(M, N, K, dtype):
    """Cache the pure shape-based dispatch decision used by ``matmul``."""
    strong_lds_plan = _strong_lds_plan(M, N, K)
    if strong_lds_plan is not None:
        return "lds", strong_lds_plan
    register_config = _register_plan_for_shape(M, N, K, dtype)
    if register_config is not None:
        return "register", register_config
    return "lds", _lds_plan_for_shape(M, N, K)


def _launch_lds(a, b, bias=None, SPLIT_K=None, TILE=None, K_LIMIT=None, DEFER_EPILOGUE=False, out=None):
    """Launch the shared gfx950 GEMM core, optionally with a fused bias."""
    M, input_k = a.shape
    b_k, N = b.shape
    assert input_k == b_k, "Incompatible dimensions"
    K = input_k if K_LIMIT is None else K_LIMIT
    assert 0 < K <= input_k, f"K_LIMIT={K} must be in (0, {input_k}]"
    if bias is not None:
        assert bias.shape == (M, N), f"Bias must expand to ({M}, {N}), got {tuple(bias.shape)}"
        assert bias.device == a.device, "Bias and matrix operands must be on the same device"
        assert bias.dtype == a.dtype, "Bias and matrix operands must have the same dtype"
        if _needs_i64_offsets(bias):
            raise ValueError("gfx950 inter-wave GEMM bias exceeds signed-i32 byte offsets; "
                             f"shape={tuple(bias.shape)}, strides={bias.stride()}")
    if TILE is not None:
        BM, BN = TILE
        grid_mn = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        SPLIT_K = _lds_split_k_for(grid_mn, K) if SPLIT_K is None else SPLIT_K
    elif SPLIT_K is None:
        BM, BN, SPLIT_K = _lds_plan_for_shape(M, N, K)
    else:
        BM, BN = BLOCK_M, BLOCK_N  # explicit SPLIT_K override keeps the default tile
    uneven_split_k = K % SPLIT_K != 0
    KS = K // SPLIT_K
    # Each split is big enough for the 2-tile prologue and starts on a 16-byte
    # boundary. Full BLOCK_K tiles use direct-to-LDS; the remainder is handled by
    # the masked register tail in the kernel.
    if uneven_split_k:
        assert K % BLOCK_K == 0, (f"Uneven Split-K requires K={K} to be divisible by BLOCK_K={BLOCK_K}")
    min_ks = ((K // BLOCK_K // SPLIT_K) * BLOCK_K if uneven_split_k else KS)
    assert min_ks >= 2 * BLOCK_K, (f"K/SPLIT_K={min_ks} must be at least {2 * BLOCK_K}")
    assert min_ks * a.element_size() % 16 == 0, (f"K/SPLIT_K={min_ks} must preserve 16-byte split alignment")
    c = torch.empty((M, N), device=a.device, dtype=a.dtype) if out is None else out
    GRID_MN = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    if SPLIT_K > 1 or DEFER_EPILOGUE:
        workspace_shape = (SPLIT_K * M, N)
        workspace_view = torch.empty(workspace_shape, device="meta", dtype=torch.float32)
        if _needs_i64_offsets(workspace_view):
            raise ValueError("gfx950 inter-wave GEMM FP32 workspace exceeds signed-i32 byte offsets; "
                             f"shape={workspace_shape}, SPLIT_K={SPLIT_K}, DEFER_EPILOGUE={DEFER_EPILOGUE}")
        # fp32 workspace: partials are stored without a rounding step, so the
        # split-K result matches a single fp32-accumulated GEMM (an fp16 workspace
        # would lose ~1e-1 near cancellation). The reduce sums in fp32 too.
        workspace = torch.empty(workspace_shape, device=a.device, dtype=torch.float32)
    else:
        workspace = c  # dummy; the kernel writes c_ptr directly when SPLIT_K==1
    bias_ptr = bias if bias is not None else c
    stride_bias_m = bias.stride(0) if bias is not None else 0
    stride_bias_n = bias.stride(1) if bias is not None else 0
    kernel_output = workspace if SPLIT_K > 1 or DEFER_EPILOGUE else c
    use_i64_c_offsets = _needs_i64_offsets(kernel_output)
    a16w16_8wave[(GRID_MN * SPLIT_K, )](
        a,
        b,
        bias_ptr,
        c,
        workspace,
        c,
        c,
        M,
        N,
        K,
        KS,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        stride_bias_m,
        stride_bias_n,
        kernel_output.stride(0),
        kernel_output.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=4 if M == N and K >= 8192 else (2 if M <= 1024 and N >= 16384 else GROUP_SIZE_M),
        NUM_XCDS=1 if M == N and K >= 8192 else NUM_XCDS,
        GRID_MN=GRID_MN,
        SPLIT_K=SPLIT_K,
        ADD_BIAS=bias is not None,
        HAS_REGISTER_TAIL=(uneven_split_k or KS % (2 * BLOCK_K) != 0),
        USE_I64_A_OFFSETS=_needs_i64_offsets(a),
        USE_I64_B_OFFSETS=_needs_i64_offsets(b),
        USE_I64_C_OFFSETS=use_i64_c_offsets,
        UNEVEN_SPLIT_K=uneven_split_k,
        HAS_M_TAIL=M % BM != 0,
        HAS_N_TAIL=N % BN != 0,
        PIN_OFFSET_LAYOUT=K_LIMIT is not None,
        DEFER_EPILOGUE=DEFER_EPILOGUE,
        WRITE_STATS=False,
        IS_RMS_NORM=False,
        num_warps=4 if BM == 128 else NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=16,
        # Forbid AGPRs: f32 accumulators write VGPRs directly (packs tighter, no
        # v_accvgpr moves around each mfma). Essential to match the reference perf.
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
        enable_sched_group_barrier_scheduler=True,
    )
    if SPLIT_K > 1:
        # Adaptive reduce tile: small outputs need many small CTAs to fill the CUs;
        # large outputs are BW-bound and prefer big tiles for burst efficiency
        # (measured: 32x32 -> 4.5 TB/s vs 128x128 -> 5.4 TB/s on Pooler).
        big = (M * N) >= (2048 * 2048)
        rbm, rbn, rw = (64, 256, 4) if big else (32, 32, 4)
        reduce_grid = (triton.cdiv(M, rbm), triton.cdiv(N, rbn))
        _reduce_k_kernel[reduce_grid](
            workspace,
            bias_ptr,
            c,
            M,
            N,
            stride_bias_m,
            stride_bias_n,
            c.stride(0),
            c.stride(1),
            SPLIT_K=SPLIT_K,
            BLOCK_SIZE_M=rbm,
            BLOCK_SIZE_N=rbn,
            OUTPUT_DTYPE=_TORCH_TO_TL[a.dtype],
            ADD_BIAS=bias is not None,
            num_warps=rw,
        )
    if DEFER_EPILOGUE:
        return workspace, c
    return c


def _lds_matmul(a, b, SPLIT_K=None):
    """C = A @ B. `a` is (M, K), `b` is (K, N).

    SPLIT_K partitions the K reduction across SPLIT_K programs per output tile
    (grid = GRID_MN*SPLIT_K), landing fp32 partials in a (SPLIT_K*M, N) workspace
    that a separate fp32 reduce kernel sums into C. This fills the CUs on small-N /
    small-tile-count shapes where the M/N tile grid alone can't. SPLIT_K is chosen
    automatically from the shape (pass an int to override); SPLIT_K=1 launches the
    plain kernel (no workspace, no reduce). The fp32 workspace keeps the result
    deterministic without atomics; as with any Split-K scheme, the changed
    reduction order may introduce normal fp32 rounding differences from the
    non-split kernel.
    """
    if SPLIT_K is None:
        M, K = a.shape
        N = b.shape[1]
        path, config = _matmul_plan(M, N, K, a.dtype)
        if path == "register":
            return _launch_register(a, b, config=config)
        block_m, block_n, split_k = config
        return _launch_lds(
            a,
            b,
            SPLIT_K=split_k,
            TILE=(block_m, block_n),
        )
    return _launch_lds(a, b, SPLIT_K=SPLIT_K)


def _validate_streamk(a, b):
    assert a.is_cuda and b.is_cuda
    assert a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16),\
        "streamk_matmul requires matching FP16 or BF16 operands"
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0,\
        f"M and N must be multiples of {BLOCK_M}"
    assert K >= MIN_K, f"K must be at least {MIN_K}"
    assert K % (2 * BLOCK_K) == 0, f"K must be a multiple of {2 * BLOCK_K}"
    return M, N, K


def _choose_streamk_tile(M, N):
    """Use a smaller persistent tile only when the default grid underfills the GPU."""
    BM, BN = TILE_CANDIDATES[0]
    grid_mn = (M // BM) * (N // BN)
    if grid_mn < NUM_CU // 2:
        for candidate_m, candidate_n in TILE_CANDIDATES[1:]:
            candidate_grid = (M // candidate_m) * (N // candidate_n)
            if grid_mn < candidate_grid <= NUM_CU:
                BM, BN, grid_mn = candidate_m, candidate_n, candidate_grid
    return BM, BN


def streamk_matmul(a, b):
    """Run one persistent kernel with a variable-work Stream-K tail when profitable."""
    M, N, K = _validate_streamk(a, b)
    BM, BN = _choose_streamk_tile(M, N)
    schedule = _streamk_schedule(M, N, K, block_m=BM, block_n=BN)
    # Keep the extracted async pipeline out of an outer multi-wave tile loop;
    # the AMD warp-pipeline pass cannot nest its waits in that region. The
    # data-centric kernel keeps the same pipeline inline for this case.
    if not schedule["HAS_STREAMK"] and schedule["NUM_FULL_TILES"] > schedule["NUM_PROGRAMS"]:
        return _lds_matmul(a, b)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if schedule["HAS_STREAMK"]:
        partials = torch.empty((NUM_CU, BM, BN), device=a.device, dtype=torch.float32)
        locks = torch.empty((NUM_CU, ), device=a.device, dtype=torch.int32)
    else:
        partials = locks = c
    streamk_kernel[(schedule["NUM_PROGRAMS"], )](a, b, c, partials, locks, _READY_VALUE, K, a.stride(0), a.stride(1),
                                                 b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=BM,
                                                 BLOCK_N=BN, BLOCK_K=BLOCK_K, NUM_XCDS=NUM_XCDS, NUM_CU=NUM_CU,
                                                 GROUP_SIZE_M=GROUP_SIZE_M, **schedule, num_warps=NUM_WARPS,
                                                 num_stages=1, matrix_instr_nonkdim=16, llvm_fn_attrs=_LLVM_ATTRS)
    return c


# Persistent N160/N192 paths.

TILE = tl.constexpr(32)
_PERSISTENT_BLOCK_K = tl.constexpr(64)
K_BLOCKS = tl.constexpr(96)
N_GROUP_FRAGMENTS = tl.constexpr(4)

# (BLOCK_M, BLOCK_N, NUM_PID_N, NUM_PROGRAMS, TILES_PER_PROGRAM)
_MT256X160_TILE_SPEC = (256, 160, 128, 256, 2)
_MT256X192_TILE_SPEC = (256, 192, 128, 256, 2)

# The pipeline schedule is compile-time data, separate from the reusable load,
# LDS, MFMA, persistent traversal, and epilogue machinery below.
_M8_A1_READ_PLAN = (
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 3),
    (3, 3),
    (6, 2),
    (0, 0),
)
_MT256X160_B_PUBLISH_PLAN = (
    ((2, 0, 3), (-1, 0, 0)),
    ((2, 3, 2), (3, 0, 1)),
    ((3, 1, 4), (-1, 0, 0)),
    ((4, 0, 3), (-1, 0, 0)),
    ((1, 0, 2), (4, 3, 2)),
)
_MT256X160_FINISH_MFMA_PLAN = (7, 8, 9, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34)
_MT256X160_PIPELINE_SPEC = (
    _M8_A1_READ_PLAN,
    4,
    _MT256X160_B_PUBLISH_PLAN,
    _MT256X160_FINISH_MFMA_PLAN,
)

# N192 reuses the same N128-plus-tail decomposition with two independent N32
# tail images.  B publishes cover rows 1-3 and the first four columns of row 4;
# the final LDS reads cover the rest of row 4 and rows 5-6.
_MT256X192_B_PUBLISH_PLAN = (
    ((1, 0, 4), (-1, 0, 0)),
    ((1, 4, 2), (2, 0, 2)),
    ((2, 2, 4), (-1, 0, 0)),
    ((3, 0, 4), (-1, 0, 0)),
    ((3, 4, 2), (4, 0, 2)),
    ((4, 2, 2), (-1, 0, 0)),
)
_MT256X192_FINISH_MFMA_PLAN = tuple(range(28, 42))
_MT256X192_PIPELINE_SPEC = (
    _M8_A1_READ_PLAN,
    5,
    _MT256X192_B_PUBLISH_PLAN,
    _MT256X192_FINISH_MFMA_PLAN,
)

_SPECIALIZATIONS = {
    "mt256x160": (
        (1024, 20480, 6144),
        _MT256X160_TILE_SPEC,
        _MT256X160_PIPELINE_SPEC,
    ),
    "mt256x192": (
        (1024, 24576, 6144),
        _MT256X192_TILE_SPEC,
        _MT256X192_PIPELINE_SPEC,
    ),
}
_SHAPE_DEFAULTS = {
    (1024, 20480, 6144): "mt256x160",
    (1024, 24576, 6144): "mt256x192",
}

# Every complete group of four N32 fragments shares one N128 LDS image.  Any
# remaining fragments use separate N32 images, avoiding power-of-two padding.
_B_BASES = tl.constexpr([[1 << bit, 0] for bit in range(6)] + [[0, 1 << bit] for bit in (4, 5, 6, 0, 1, 2, 3)])

# Four waves jointly own the first four N32 accumulator fragments as one 32x128
# region.  Each lane gets sixteen contiguous N values, allowing four narrow
# stores to become two 128-bit stores after one layout conversion.
_C_STORE_32X128_LAYOUT = tlx.layout(
    shape=((16, 4, 2, 2), (16, )),
    stride=((128, 16, 2048, 64), (1, )),
)


@triton.jit
def _global_loads(source, kb, tile_spec: tl.constexpr):
    a_ptr, b_ptr, stride_ak, stride_bk, a_offsets, b_offsets = source
    a_ptr += kb * _PERSISTENT_BLOCK_K * stride_ak
    b_ptr += kb * _PERSISTENT_BLOCK_K * stride_bk
    a = [tlx.buffer_load(a_ptr, a_offsets[mi], contiguity=8) for mi in range(tl.constexpr(tile_spec[0] // TILE))]
    b = [tlx.buffer_load(b_ptr, b_offsets[nj], contiguity=8) for nj in range(tl.constexpr(tile_spec[1] // TILE))]
    return tl.tuple(a + b)


@triton.jit
def _local_store_one(stage, value, index: tl.constexpr, tile_spec: tl.constexpr):
    a_buffers, b_buffers = stage
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_main_fragments: tl.constexpr = n_main_groups * N_GROUP_FRAGMENTS
    if index < m_fragments:
        tlx.local_store(tlx.local_view(a_buffers[index], 0), value)
    else:
        nj: tl.constexpr = index - m_fragments
        view = tlx.local_slice(
            tlx.local_view(
                b_buffers[nj // N_GROUP_FRAGMENTS if nj < n_main_fragments else n_main_groups + nj - n_main_fragments],
                0,
            ),
            [
                0,
                (nj % N_GROUP_FRAGMENTS) * TILE if nj < n_main_fragments else 0,
            ],
            [_PERSISTENT_BLOCK_K, TILE],
        )
        tlx.local_store(view, value)


@triton.jit
def _local_store_all(stage, values, tile_spec: tl.constexpr):
    load_groups: tl.constexpr = tl.constexpr(tile_spec[0] // TILE + tile_spec[1] // TILE)
    for index in tl.static_range(load_groups):
        _local_store_one(stage, values[index], index, tile_spec)


@triton.jit
def _local_load_a(stage, kh: tl.constexpr, mi: tl.constexpr, dot_a: tl.constexpr):
    a_buffers = stage[0]
    view = tlx.local_slice(
        tlx.local_view(a_buffers[mi], 0),
        [0, kh * TILE],
        [TILE, TILE],
    )
    return tlx.require_layout(tlx.local_load(view, relaxed=True), dot_a, pin=False)


@triton.jit
def _local_load_b(
    stage,
    kh: tl.constexpr,
    nj: tl.constexpr,
    dot_b: tl.constexpr,
    tile_spec: tl.constexpr,
):
    b_buffers = stage[1]
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_main_fragments: tl.constexpr = n_main_groups * N_GROUP_FRAGMENTS
    view = tlx.local_slice(
        tlx.local_view(
            b_buffers[nj // N_GROUP_FRAGMENTS if nj < n_main_fragments else n_main_groups + nj - n_main_fragments],
            0,
        ),
        [
            kh * TILE,
            (nj % N_GROUP_FRAGMENTS) * TILE if nj < n_main_fragments else 0,
        ],
        [TILE, TILE],
    )
    value = tlx.local_load(view, relaxed=True)
    return tlx.require_layout(value, dot_b, pin=False)


@triton.jit
def _local_load_b_row(
    stage,
    kh: tl.constexpr,
    dot_b: tl.constexpr,
    tile_spec: tl.constexpr,
):
    return tl.tuple(
        [_local_load_b(stage, kh, nj, dot_b, tile_spec) for nj in range(tl.constexpr(tile_spec[1] // TILE))])


@triton.jit
def _mfma_part(
    a_operand,
    b_operands,
    acc,
    mi: tl.constexpr,
    first_nj: tl.constexpr,
    count: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    tile_spec: tl.constexpr,
):
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    accumulators: tl.constexpr = tl.constexpr(tile_spec[0] // TILE * tile_spec[1] // TILE)
    return tl.tuple([
        tlx.amd_scheduled_mfma(
            tlx.require_layout(a_operand, dot_a, pin=False),
            tlx.require_layout(b_operands[index % n_fragments], dot_b, pin=False),
            tlx.require_layout(acc[index], mma, pin=False),
            accumulator_role="persistent",
            resident_operand=None,
            initialize=initialize,
        ) if (index // n_fragments == mi and index % n_fragments >= first_nj and index % n_fragments < first_nj + count)
        else acc[index] for index in range(accumulators)
    ])


@triton.jit
def _mfma_row(
    a_operand,
    b_operands,
    acc,
    mi: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    tile_spec: tl.constexpr,
):
    return _mfma_part(
        a_operand,
        b_operands,
        acc,
        mi,
        0,
        tl.constexpr(tile_spec[1] // TILE),
        mma,
        dot_a,
        dot_b,
        initialize,
        tile_spec,
    )


@triton.jit
def _global_prefetch_one(source, kb, index: tl.constexpr, tile_spec: tl.constexpr):
    a_ptr, b_ptr, stride_ak, stride_bk, a_offsets, b_offsets = source
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    if index < m_fragments:
        return tlx.buffer_load(
            a_ptr + kb * _PERSISTENT_BLOCK_K * stride_ak,
            a_offsets[index],
            contiguity=8,
        )
    else:
        return tlx.buffer_load(
            b_ptr + kb * _PERSISTENT_BLOCK_K * stride_bk,
            b_offsets[index - m_fragments],
            contiguity=8,
        )


@triton.jit
def _publish_a_row(
    a_operand,
    b_operands,
    acc,
    old_value,
    next_stage,
    source,
    future_kb,
    mi: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Advance one A fragment across three K64 pipeline generations.

    ``a_operand`` and ``b_operands`` compute one accumulator row for K(t),
    ``old_value`` is A(K(t+1)) already prefetched in VGPRs, and ``future``
    becomes A(K(t+2)) in VGPRs.  The K(t) MFMA row is split around the future
    global load to balance load-latency coverage against VGPR lifetime.
    """
    # One accumulator row contains BLOCK_N / 32 logical C[32, 32]
    # fragments.  This is 5 fragments for N160 and 6 for N192.
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)

    # K(t+1): retire the previously prefetched A[32, 64] value from VGPRs
    # into the next LDS stage.
    _local_store_one(next_stage, old_value, mi, tile_spec)

    # K(t): compute columns [0, mfmas_before_prefetch) of accumulator row mi.
    acc = _mfma_part(
        a_operand,
        b_operands,
        acc,
        mi,
        0,
        tl.constexpr(pipeline_spec[1]),
        mma,
        dot_a,
        dot_b,
        initialize,
        tile_spec,
    )

    # K(t+2): issue this row's next global A[32, 64] load into VGPRs.
    future = _global_prefetch_one(source, future_kb, mi, tile_spec)

    # K(t): compute the remaining columns
    # [mfmas_before_prefetch, n_fragments).  For N160 the split is 4 + 1;
    # for N192 it is 5 + 1.
    acc = _mfma_part(
        a_operand,
        b_operands,
        acc,
        mi,
        tl.constexpr(pipeline_spec[1]),
        n_fragments - tl.constexpr(pipeline_spec[1]),
        mma,
        dot_a,
        dot_b,
        initialize,
        tile_spec,
    )
    return acc, future


@triton.jit
def _publish_b_fragment(
    a_operands,
    b_operands,
    acc,
    prefetched,
    next_stage,
    source,
    future_kb,
    nj: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    index: tl.constexpr = m_fragments + nj
    _local_store_one(next_stage, prefetched[index], index, tile_spec)
    acc = _mfma_part(
        a_operands[0],
        b_operands,
        acc,
        0,
        nj,
        1,
        mma,
        dot_a,
        dot_b,
        False,
        tile_spec,
    )

    future = _global_prefetch_one(source, future_kb, index, tile_spec)

    # Spread the MFMA rows across publish groups according to a tile-specific
    # compile-time plan, so the pipeline core itself is independent of 8x5.
    for part in tl.static_range(tl.constexpr(len(pipeline_spec[2][nj]))):
        if tl.constexpr(pipeline_spec[2][nj][part][2]) > 0:
            acc = _mfma_part(
                a_operands[tl.constexpr(pipeline_spec[2][nj][part][0])],
                b_operands,
                acc,
                tl.constexpr(pipeline_spec[2][nj][part][0]),
                tl.constexpr(pipeline_spec[2][nj][part][1]),
                tl.constexpr(pipeline_spec[2][nj][part][2]),
                mma,
                dot_a,
                dot_b,
                False,
                tile_spec,
            )
    return acc, future


@triton.jit
def _publish_b_rows(
    a_operands,
    b_operands,
    acc,
    prefetched,
    next_stage,
    source,
    future_kb,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Publish B fragments and issue their tile-specific K1 MFMA spans."""
    future = tl.tuple([])
    for nj in tl.static_range(tl.constexpr(tile_spec[1] // TILE)):
        acc, value = _publish_b_fragment(
            a_operands,
            b_operands,
            acc,
            prefetched,
            next_stage,
            source,
            future_kb,
            nj,
            mma,
            dot_a,
            dot_b,
            pipeline_spec,
            tile_spec,
        )
        future += tl.tuple([value])
    return acc, future


@triton.jit
def _finish_read(
    next_stage,
    a_operands,
    b_operands,
    acc,
    read: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    if read < n_fragments:
        value = _local_load_b(next_stage, 0, read, dot_b, tile_spec)
    else:
        value = _local_load_a(next_stage, 0, read - n_fragments, dot_a)
    flat: tl.constexpr = tl.constexpr(pipeline_spec[3][read])
    acc = _mfma_part(
        a_operands[flat // n_fragments],
        b_operands,
        acc,
        flat // n_fragments,
        flat % n_fragments,
        1,
        mma,
        dot_a,
        dot_b,
        False,
        tile_spec,
    )
    return acc, value


@triton.jit
def _finish_iteration(
    next_stage,
    a_operands,
    b_operands,
    acc,
    early_prefetched,
    late_prefetched,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Pair next-stage LDS reads with the tile-specific final MFMA sequence."""
    next_a = tl.tuple([])
    next_b = tl.tuple([])
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    for read in tl.static_range(m_fragments + n_fragments):
        acc, value = _finish_read(
            next_stage,
            a_operands,
            b_operands,
            acc,
            read,
            mma,
            dot_a,
            dot_b,
            pipeline_spec,
            tile_spec,
        )
        if read < n_fragments:
            next_b += tl.tuple([value])
        else:
            next_a += tl.tuple([value])
    acc = _mfma_row(
        a_operands[m_fragments - 1],
        b_operands,
        acc,
        m_fragments - 1,
        mma,
        dot_a,
        dot_b,
        False,
        tile_spec,
    )
    return acc, early_prefetched + late_prefetched, next_a, next_b


@triton.jit
def _pipeline(
    source,
    future_kb,
    current_stage,
    next_stage,
    current_a,
    current_b,
    prefetched,
    acc,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Advance the manual pipeline by one K64 block.

    Entry state:
      * ``current_stage`` contains K(t), while ``current_a``/``current_b``
        already hold its first K32 half (kh=0) in operand VGPRs.
      * ``prefetched`` contains the complete K(t+1) A/B tile in VGPRs.
      * ``next_stage`` is available to receive K(t+1).

    Exit state:
      * K(t) has been fully accumulated.
      * ``next_stage`` contains K(t+1), whose kh=0 operands are returned.
      * the complete K(t+2) A/B tile is returned in prefetch VGPRs.
    """
    # Future global prefetches for K(t+2).  ``a1`` and ``b1`` are not K(t+1):
    # they are the second K32 half (kh=1) of the current K(t) LDS stage.
    future_a = tl.tuple([])
    a1 = tl.tuple([])
    b1 = tl.tuple([])
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    for mi in tl.static_range(m_fragments):
        # K(t).kh1: spread the smaller B operand window across the first
        # n_fragments loop iterations instead of issuing one late LDS burst.
        if mi < n_fragments:
            b1 += tl.tuple([_local_load_b(current_stage, 1, mi, dot_b, tile_spec)])

        # In one interleaved step: publish A(mi,K(t+1)) from VGPRs to the next
        # LDS stage, compute row mi of K(t).kh0, and prefetch A(mi,K(t+2)).
        acc, future = _publish_a_row(
            current_a[mi],
            current_b,
            acc,
            prefetched[mi],
            next_stage,
            source,
            future_kb,
            mi,
            mma,
            dot_a,
            dot_b,
            initialize,
            pipeline_spec,
            tile_spec,
        )
        future_a += tl.tuple([future])

        # K(t).kh1: read the configured A operand span from current LDS.  The
        # plan covers every A fragment exactly once while controlling lifetime.
        if tl.constexpr(pipeline_spec[0][mi][1]) > 0:
            a1 += tl.tuple([
                _local_load_a(current_stage, 1, index, dot_a) for index in range(
                    tl.constexpr(pipeline_spec[0][mi][0]),
                    tl.constexpr(pipeline_spec[0][mi][0]) + tl.constexpr(pipeline_spec[0][mi][1]),
                )
            ])

    # Publish B(K(t+1)) to next LDS and prefetch B(K(t+2)), while a1/b1
    # compute most of the current K(t).kh1 accumulator updates.
    acc, late_prefetched = _publish_b_rows(
        a1,
        b1,
        acc,
        prefetched,
        next_stage,
        source,
        future_kb,
        mma,
        dot_a,
        dot_b,
        pipeline_spec,
        tile_spec,
    )

    # All A/B fragments of K(t+1) are now in next_stage.  Make them visible
    # before reading its kh=0 operands; pair those reads with the remaining
    # K(t).kh1 MFMA updates in _finish_iteration.
    tl.debug_barrier()
    return _finish_iteration(
        next_stage,
        a1,
        b1,
        acc,
        future_a,
        late_prefetched,
        mma,
        dot_a,
        dot_b,
        pipeline_spec,
        tile_spec,
    )


@triton.jit
def _pipeline_pair(
    kb,
    source,
    stage0,
    stage1,
    current_a,
    current_b,
    prefetched,
    acc,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Consume K(kb) and K(kb+1), restoring the LDS stage orientation.

    On entry, stage0/current_a/current_b represent K(kb), and ``prefetched``
    is K(kb+1).  The first call advances to K(kb+1), swapping the current LDS
    stage from stage0 to stage1.  The second advances to K(kb+2), swapping it
    back to stage0.  The numeric argument passed to ``_pipeline`` is the
    future global-prefetch block, not the block currently being consumed.
    """
    # Consume K(kb), publish K(kb+1) into stage1, and prefetch K(kb+2).
    acc, prefetched, current_a, current_b = _pipeline(
        source,
        kb + 2,
        stage0,
        stage1,
        current_a,
        current_b,
        prefetched,
        acc,
        mma,
        dot_a,
        dot_b,
        initialize,
        pipeline_spec,
        tile_spec,
    )
    # Consume K(kb+1), publish K(kb+2) into stage0, and prefetch K(kb+3).
    # Accumulators were initialized by the first call, so initialize=False.
    return _pipeline(
        source,
        kb + 3,
        stage1,
        stage0,
        current_a,
        current_b,
        prefetched,
        acc,
        mma,
        dot_a,
        dot_b,
        False,
        pipeline_spec,
        tile_spec,
    )


@triton.jit
def _make_global_tile_load_addresses(
    a_ptr,
    b_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    tile_id,
    tile_spec: tl.constexpr,
):
    """Build the A/B base pointers and fragment offsets for one output tile."""
    block_m: tl.constexpr = tl.constexpr(tile_spec[0])
    block_n: tl.constexpr = tl.constexpr(tile_spec[1])
    num_pid_n: tl.constexpr = tl.constexpr(tile_spec[2])
    m_fragments: tl.constexpr = block_m // TILE
    n_fragments: tl.constexpr = block_n // TILE
    rk = tl.arange(0, _PERSISTENT_BLOCK_K)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    block_m_offset = pid_m * block_m
    block_n_offset = pid_n * block_n
    a_offsets = tl.tuple([
        (block_m_offset + mi * TILE + tl.arange(0, TILE))[:, None] * stride_am + rk[None, :] * stride_ak
        for mi in range(m_fragments)
    ])
    b_offsets = tl.tuple([
        rk[:, None] * stride_bk + (block_n_offset + nj * TILE + tl.arange(0, TILE))[None, :] * stride_bn
        for nj in range(n_fragments)
    ])
    return tl.tuple([a_ptr, b_ptr, stride_ak, stride_bk, a_offsets, b_offsets])


@triton.jit
def _consume_k32(
    stage,
    kh: tl.constexpr,
    acc,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    tile_spec: tl.constexpr,
):
    b_operands = _local_load_b_row(stage, kh, dot_b, tile_spec)
    for mi in tl.static_range(tl.constexpr(tile_spec[0] // TILE)):
        acc = _mfma_row(
            _local_load_a(stage, kh, mi, dot_a),
            b_operands,
            acc,
            mi,
            mma,
            dot_a,
            dot_b,
            False,
            tile_spec,
        )
    return acc


@triton.jit
def _compute_full_tile(
    source,
    preloaded_k0,
    stage0,
    stage1,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Compute one output tile across all 96 K64 blocks.

    ``preloaded_k0`` is the already-issued global load of K0 held in VGPRs.
    The prologue publishes K0 to stage0 and prefetches K1.  Pipeline pairs
    consume K0..K93 and leave K94 in stage0 plus K95 in prefetch VGPRs.  The
    explicit epilogue drains K94/K95 without issuing the out-of-range K96/K97
    prefetches a final pair would require.
    """
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)

    # Prologue: publish the caller's preloaded K0 from VGPRs to stage0, read
    # K0.kh0 into operand VGPRs, and prefetch the complete K1 into VGPRs.
    _local_store_all(stage0, preloaded_k0, tile_spec)
    tl.debug_barrier()
    current_b = _local_load_b_row(stage0, 0, dot_b, tile_spec)
    current_a = tl.tuple([_local_load_a(stage0, 0, mi, dot_a) for mi in range(m_fragments)])
    prefetched = _global_loads(source, 1, tile_spec)

    # Create one persistent FP32 accumulator for every logical C[32, 32]
    # fragment.  The first pair consumes K0/K1 and prepares K2/K3.
    zero = tlx.zeros((TILE, TILE), tl.float32, layout=mma)
    acc = tl.tuple([zero for _ in range(m_fragments * n_fragments)])
    acc, prefetched, current_a, current_b = _pipeline_pair(
        0,
        source,
        stage0,
        stage1,
        current_a,
        current_b,
        prefetched,
        acc,
        mma,
        dot_a,
        dot_b,
        True,
        pipeline_spec,
        tile_spec,
    )

    # Steady state: pair(kb) consumes K(kb)/K(kb+1) and prepares
    # K(kb+2)/K(kb+3).  kb takes 2,4,...,92, so this covers K2..K93 and
    # exits with K94 in stage0/current operands plus K95 in prefetch VGPRs.
    for kb in tl.range(2, K_BLOCKS - 2, 2, num_stages=1):
        acc, prefetched, current_a, current_b = _pipeline_pair(
            kb,
            source,
            stage0,
            stage1,
            current_a,
            current_b,
            prefetched,
            acc,
            mma,
            dot_a,
            dot_b,
            False,
            pipeline_spec,
            tile_spec,
        )

    # Epilogue: publish K95 into stage1 while consuming the already-resident
    # K94.kh0 operands, then consume K94.kh1 from stage0.  After the barrier,
    # both K32 halves of K95 are safe to read and accumulate from stage1.
    _local_store_all(stage1, prefetched, tile_spec)
    for mi in tl.static_range(m_fragments):
        acc = _mfma_row(
            current_a[mi],
            current_b,
            acc,
            mi,
            mma,
            dot_a,
            dot_b,
            False,
            tile_spec,
        )
    acc = _consume_k32(stage0, 1, acc, mma, dot_a, dot_b, tile_spec)
    tl.debug_barrier()
    for kh in tl.static_range(2):
        acc = _consume_k32(stage1, kh, acc, mma, dot_a, dot_b, tile_spec)
    return acc


@triton.jit
def _global_store_output(
    c_ptr,
    acc,
    stride_cm,
    stride_cn,
    tile_id,
    mma: tl.constexpr,
    tile_spec: tl.constexpr,
):
    block_m: tl.constexpr = tl.constexpr(tile_spec[0])
    block_n: tl.constexpr = tl.constexpr(tile_spec[1])
    num_pid_n: tl.constexpr = tl.constexpr(tile_spec[2])
    m_fragments: tl.constexpr = block_m // TILE
    n_fragments: tl.constexpr = block_n // TILE
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_main_fragments: tl.constexpr = n_main_groups * N_GROUP_FRAGMENTS
    n_tail_fragments: tl.constexpr = n_fragments - n_main_fragments
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    rm = pid_m * block_m + tl.arange(0, TILE)
    rn = pid_n * block_n + tl.arange(0, TILE)
    for mi in tl.static_range(m_fragments):
        for group in tl.static_range(n_main_groups):
            rn_wide = (pid_n * block_n + group * N_GROUP_FRAGMENTS * TILE + tl.arange(0, N_GROUP_FRAGMENTS * TILE))
            offsets = (c_ptr + (rm + mi * TILE)[:, None] * stride_cm + rn_wide[None, :] * stride_cn)
            lo = tl.cat(
                tlx.require_layout(
                    acc[mi * n_fragments + group * N_GROUP_FRAGMENTS],
                    mma,
                    pin=False,
                ),
                tlx.require_layout(
                    acc[mi * n_fragments + group * N_GROUP_FRAGMENTS + 1],
                    mma,
                    pin=False,
                ),
                dim=1,
            )
            hi = tl.cat(
                tlx.require_layout(
                    acc[mi * n_fragments + group * N_GROUP_FRAGMENTS + 2],
                    mma,
                    pin=False,
                ),
                tlx.require_layout(
                    acc[mi * n_fragments + group * N_GROUP_FRAGMENTS + 3],
                    mma,
                    pin=False,
                ),
                dim=1,
            )
            value = tl.cat(lo, hi, dim=1)
            value = tlx.require_layout(value.to(c_ptr.dtype.element_ty), _C_STORE_32X128_LAYOUT)
            tlx.assert_same_layout(value, _C_STORE_32X128_LAYOUT)
            tl.store(offsets, value)
        for tail in tl.static_range(n_tail_fragments):
            offsets = tlx.require_layout(
                c_ptr + (rm + mi * TILE)[:, None] * stride_cm + (rn +
                                                                 (n_main_fragments + tail) * TILE)[None, :] * stride_cn,
                mma,
                pin=False,
            )
            value = tlx.require_layout(
                acc[mi * n_fragments + n_main_fragments + tail],
                mma,
                pin=False,
            )
            tl.store(offsets, value)


@triton.jit
def _commit_accumulators(acc, mma: tl.constexpr, tile_spec: tl.constexpr):
    accumulators: tl.constexpr = tl.constexpr(tile_spec[0] // TILE * tile_spec[1] // TILE)
    values = [tlx.require_layout(acc[index], mma, pin=False) for index in range(accumulators)]
    return tlx.amd_mfma_commit(tl.tuple(values))


@triton.jit
def _local_alloc_stage(
    a_layout: tl.constexpr,
    b_main_layout: tl.constexpr,
    b_tail_layout: tl.constexpr,
    tile_spec: tl.constexpr,
):
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_tail_fragments: tl.constexpr = (n_fragments - n_main_groups * N_GROUP_FRAGMENTS)
    a_buffers = tl.tuple(
        [tlx.local_alloc((TILE, _PERSISTENT_BLOCK_K), tl.float16, 1, layout=a_layout) for _ in range(m_fragments)])
    b_buffers = tl.tuple([
        tlx.local_alloc(
            (_PERSISTENT_BLOCK_K, N_GROUP_FRAGMENTS * TILE),
            tl.float16,
            1,
            layout=b_main_layout,
        ) for _ in range(n_main_groups)
    ] + [
        tlx.local_alloc((_PERSISTENT_BLOCK_K, TILE), tl.float16, 1, layout=b_tail_layout)
        for _ in range(n_tail_fragments)
    ])
    return tl.tuple([a_buffers, b_buffers])


@triton.jit
def _persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    TILE_SPEC: tl.constexpr,
    PIPELINE_SPEC: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 2],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    a_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(64, 16)],
                                                                                  [TILE, _PERSISTENT_BLOCK_K],
                                                                                  order=[1, 0]))
    b_main_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases([(512, 16)], _B_BASES,
                                                                                [_PERSISTENT_BLOCK_K, 128]))
    b_tail_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(256, 16)],
                                                                                       [_PERSISTENT_BLOCK_K, TILE],
                                                                                       order=[0, 1]))
    stage0 = _local_alloc_stage(a_layout, b_main_layout, b_tail_layout, TILE_SPEC)
    stage1 = _local_alloc_stage(a_layout, b_main_layout, b_tail_layout, TILE_SPEC)

    # Persistently assign a compile-time group of adjacent N tiles to each
    # program.  For example, 128 N tiles with two tiles/program produce 64
    # programs per M row: program 0 -> tiles 0/1, ..., program 63 -> 126/127.
    program = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.constexpr(TILE_SPEC[2])
    tiles_per_program: tl.constexpr = tl.constexpr(TILE_SPEC[4])
    tl.static_assert(tiles_per_program > 0)
    tl.static_assert(num_pid_n % tiles_per_program == 0)
    programs_per_m = num_pid_n // tiles_per_program
    program_m = program // programs_per_m
    program_n = program % programs_per_m
    first_tile = (program_m * num_pid_n + program_n * tiles_per_program)

    # Prime the persistent traversal with the first tile's complete K0 slice.
    tile_load_addresses = _make_global_tile_load_addresses(
        a_ptr,
        b_ptr,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        first_tile,
        TILE_SPEC,
    )
    prefetched_k0 = _global_loads(tile_load_addresses, 0, TILE_SPEC)

    for tile_offset in tl.static_range(tiles_per_program):
        tile_id = first_tile + tile_offset
        acc_tile = _compute_full_tile(
            tile_load_addresses,
            prefetched_k0,
            stage0,
            stage1,
            mma,
            dot_a,
            dot_b,
            PIPELINE_SPEC,
            TILE_SPEC,
        )

        # Before the current tile's epilogue, issue the next tile's K0 loads.
        # At most one completed accumulator tile and one future K0 prefetch are
        # live together, independent of tiles_per_program.
        if tile_offset + 1 < tiles_per_program:
            next_tile_load_addresses = _make_global_tile_load_addresses(
                a_ptr,
                b_ptr,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                tile_id + 1,
                TILE_SPEC,
            )
            next_prefetched_k0 = _global_loads(next_tile_load_addresses, 0, TILE_SPEC)

        acc_tile = _commit_accumulators(acc_tile, mma, TILE_SPEC)
        _global_store_output(
            c_ptr,
            acc_tile,
            stride_cm,
            stride_cn,
            tile_id,
            mma,
            TILE_SPEC,
        )

        if tile_offset + 1 < tiles_per_program:
            tile_load_addresses = next_tile_load_addresses
            prefetched_k0 = next_prefetched_k0


def _validate_specialization(m, n, tile_spec, pipeline_spec):
    block_m, block_n, num_pid_n, num_programs, tiles_per_program = tile_spec
    m_fragments = block_m // int(TILE)
    n_fragments = block_n // int(TILE)
    a1_read_plan, mfmas_before_prefetch, b_publish_plan, finish_plan = (pipeline_spec)
    assert m % block_m == 0 and n % block_n == 0
    assert num_pid_n == n // block_n
    assert tiles_per_program > 0
    assert num_pid_n % tiles_per_program == 0
    assert num_programs * tiles_per_program == (m // block_m) * num_pid_n
    assert len(a1_read_plan) == m_fragments
    assert 0 <= mfmas_before_prefetch <= n_fragments
    assert len(b_publish_plan) == n_fragments
    assert len(finish_plan) == m_fragments + n_fragments

    a1_reads = []
    for first, count in a1_read_plan:
        assert 0 <= first <= m_fragments and 0 <= count <= m_fragments - first
        a1_reads.extend(range(first, first + count))
    assert sorted(a1_reads) == list(range(m_fragments))

    # K1 coverage consists of row0's mandatory B publishes, the extra spans
    # attached to each publish, the final-read plan, and the last MFMA row.
    mfma_coverage = list(range(n_fragments))
    for parts in b_publish_plan:
        for mi, first_nj, count in parts:
            if count == 0:
                continue
            assert 0 <= mi < m_fragments
            assert 0 <= first_nj < n_fragments
            assert first_nj + count <= n_fragments
            mfma_coverage.extend(mi * n_fragments + nj for nj in range(first_nj, first_nj + count))
    assert all(0 <= flat < m_fragments * n_fragments for flat in finish_plan)
    mfma_coverage.extend(finish_plan)
    mfma_coverage.extend(range((m_fragments - 1) * n_fragments, m_fragments * n_fragments))
    assert sorted(mfma_coverage) == list(range(m_fragments * n_fragments))


def _persistent_plan_for_shape(m, n, k, dtype):
    if dtype != torch.float16:
        return None
    return _SHAPE_DEFAULTS.get((m, n, k))


def _persistent_supports(a, b):
    """Return whether a and b select one of the persistent specializations."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return False
    m, k = a.shape
    _, n = b.shape
    return a.dtype == b.dtype and _persistent_plan_for_shape(m, n, k, a.dtype) is not None


def _launch_persistent(a, b, out=None, specialization=None):
    """Run a compile-time gfx950 persistent GEMM specialization."""
    assert a.ndim == 2 and b.ndim == 2
    m, k = a.shape
    kb, n = b.shape
    assert k == kb
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if specialization is None:
        specialization = os.environ.get("TLX_GFX950_TILE")
    if specialization is None:
        specialization = _persistent_plan_for_shape(m, n, k, a.dtype)
        assert specialization is not None
    assert specialization in _SPECIALIZATIONS
    expected_shape, tile_spec, pipeline_spec = _SPECIALIZATIONS[specialization]
    assert (m, n, k) == expected_shape
    _validate_specialization(m, n, tile_spec, pipeline_spec)
    _persistent_kernel[(tile_spec[3], )](
        a,
        b,
        out,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        TILE_SPEC=tile_spec,
        PIPELINE_SPEC=pipeline_spec,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
        enable_sched_group_barrier_scheduler=True,
        sched_group_barrier_mfma_per_dwordx4=1,
        regclass_priority_trumps_globalness=True,
        reverse_local_assignment=True,
    )
    return out


# LocalSplitU path and public dispatch.

__all__ = ["mm", "matmul", "supports"]


# The initial implementation deliberately exposes only measured plans. A
# follow-up change adds bounded plan generation for neighboring shapes without
# changing this register-staged execution mechanism.
class _Plan(NamedTuple):
    tile_m: int
    tile_n: int
    local_split_u: int
    wave_k: int
    k_width: int


_KNOWN_PLANS = {
    # Sixteen short K32 chains maximize wave-level latency hiding. N32 keeps
    # one workgroup per CU and gives every A load two independent MFMA users.
    (7, 8192, 2048):
    _Plan(tile_m=16, tile_n=32, local_split_u=16, wave_k=32, k_width=8),
    # N16 gives two output tiles per CU. Four K256 partitions provide enough
    # wave-level latency hiding without the long-lived K1024 operands.
    (7, 2048, 4096):
    _Plan(
        tile_m=16,
        tile_n=16,
        local_split_u=4,
        wave_k=256,
        k_width=8,
    ),
}


@lru_cache(maxsize=None)
def _device_arch(device):
    """Return the AMD architecture for a CUDA device, or an empty string."""
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "gcnArchName", "").split(":", 1)[0]


@triton.jit
def _load_dot_operands(
    a_ptr,
    b_ptr,
    global_rows,
    global_cols,
    split_ids,
    rk,
    macro_k_base,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    WAVE_K: tl.constexpr,
    K_WIDTH: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
):
    """Load one macro-K slice directly into the two MFMA operand layouts."""
    split_k = macro_k_base + split_ids[:, None, None] * WAVE_K
    a_offsets = (global_rows[None, :, None] * stride_am + (split_k + rk[None, None, :]) * stride_ak)
    b_offsets = ((split_k + rk[None, :, None]) * stride_bk + global_cols[None, None, :] * stride_bn)
    # The offsets already have their dot-operand layouts, so the loaded values
    # reach MFMA registers without an intervening conversion through LDS.
    # K_WIDTH is also the largest contiguous run owned by one lane; claiming a
    # wider buffer vector would cross the lane's two disjoint K runs.
    a_offsets = tlx.require_layout(a_offsets, dot_a)
    b_offsets = tlx.require_layout(b_offsets, dot_b)
    a = tlx.buffer_load(a_ptr, a_offsets, contiguity=K_WIDTH)
    b = tlx.buffer_load(b_ptr, b_offsets, cache=".cg", contiguity=K_WIDTH)
    return a, b


@triton.jit
def _local_split_u_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    K_WIDTH: tl.constexpr,
    WAVE_K: tl.constexpr,
    LOCAL_SPLIT_U: tl.constexpr,
):
    """Compute one MxN tile by partitioning K across the CTA's waves."""
    MACRO_K: tl.constexpr = WAVE_K * LOCAL_SPLIT_U
    tl.static_assert(M <= TILE_M)

    pid_n = tl.program_id(0).to(tl.int32)
    split_ids = tl.arange(0, LOCAL_SPLIT_U).to(tl.int32)
    rows = tl.arange(0, TILE_M).to(tl.int32)
    # Padded rows may read any valid A row because their results are discarded.
    global_rows = tl.where(rows < M, rows, 0)
    local_cols = tl.arange(0, TILE_N).to(tl.int32)
    output_cols = pid_n * TILE_N + local_cols
    global_cols = tl.where(output_cols < N, output_cols, 0)
    rk = tl.arange(0, WAVE_K).to(tl.int32)

    # The leading batch axis is the LocalSplitU partition. Mapping that axis
    # one-to-one onto waves keeps the wave-specific K coordinate explicit.
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[LOCAL_SPLIT_U, 1, 1],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=K_WIDTH)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=K_WIDTH)

    tl.static_assert(K % MACRO_K == 0)
    acc = tlx.zeros(
        (LOCAL_SPLIT_U, TILE_M, TILE_N),
        tl.float32,
        layout=mma,
    )

    current_a, current_b = _load_dot_operands(
        a_ptr,
        b_ptr,
        global_rows,
        global_cols,
        split_ids,
        rk,
        0,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        WAVE_K,
        K_WIDTH,
        dot_a,
        dot_b,
    )

    # One-stage register pipeline: issue K(t+1)'s global loads before K(t)'s
    # dot. The chosen WAVE_K controls the prefetch lifetime and register cost.
    for macro in tl.range(0, K // MACRO_K - 1, num_stages=1):
        next_k = (macro + 1) * MACRO_K
        next_a, next_b = _load_dot_operands(
            a_ptr,
            b_ptr,
            global_rows,
            global_cols,
            split_ids,
            rk,
            next_k,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            WAVE_K,
            K_WIDTH,
            dot_a,
            dot_b,
        )
        acc = tl.dot(
            current_a,
            current_b,
            acc,
            allow_tf32=False,
            out_dtype=tl.float32,
        )
        current_a = next_a
        current_b = next_b

    acc = tl.dot(
        current_a,
        current_b,
        acc,
        allow_tf32=False,
        out_dtype=tl.float32,
    )

    if LOCAL_SPLIT_U == 16 and TILE_N == 32:
        # This swizzle is tuned for the dense U16/N32 partial tile. Its encoding
        # depends on TILE_N, so other U16 widths deliberately use the default
        # layout instead of silently inheriting different swizzle parameters.
        partial_layout: tl.constexpr = tlx.swizzled_layout(2, 2, 3, order=[2, 1, 0])
        partial_buffer = tlx.local_alloc(
            (LOCAL_SPLIT_U, TILE_M, TILE_N),
            tl.float32,
            1,
            layout=partial_layout,
        )
    else:
        partial_buffer = tlx.local_alloc((LOCAL_SPLIT_U, TILE_M, TILE_N), tl.float32, 1)
    partial_view = tlx.local_view(partial_buffer, 0)
    tlx.local_store(partial_view, acc)
    tl.debug_barrier()

    tl.static_assert(LOCAL_SPLIT_U == 2 or LOCAL_SPLIT_U == 4 or LOCAL_SPLIT_U == 8 or LOCAL_SPLIT_U == 16)
    result = tl.reshape(
        tlx.local_load(tlx.local_slice(
            partial_view,
            [0, 0, 0],
            [1, TILE_M, TILE_N],
        )),
        (TILE_M, TILE_N),
    )
    # Load and immediately consume each subsequent partial. This preserves
    # increasing split-id association without keeping every partial live.
    for split in tl.static_range(1, LOCAL_SPLIT_U):
        partial = tlx.local_load(tlx.local_slice(
            partial_view,
            [split, 0, 0],
            [1, TILE_M, TILE_N],
        ))
        result += tl.reshape(partial, (TILE_M, TILE_N))

    output_rows = tl.arange(0, TILE_M).to(tl.int32)
    output_ptrs = (c_ptr + output_rows[:, None] * stride_cm + output_cols[None, :] * stride_cn)
    tl.store(
        output_ptrs,
        result.to(c_ptr.dtype.element_ty),
        mask=(output_rows[:, None] < M)
        & (output_cols[None, :] < N),
    )


def _problem_for(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return None
    m, k = a.shape
    _, n = b.shape
    if not (a.dtype in (torch.float16, torch.bfloat16) and b.dtype == a.dtype and a.is_cuda and a.device == b.device
            and _device_arch(a.device) == "gfx950" and a.stride(1) == 1 and b.stride(0) == 1):
        return None
    return m, n, k


def _plan_for(a, b):
    """Return the measured LocalSplitU plan, otherwise ``None``."""
    problem = _problem_for(a, b)
    if problem is None or a.dtype != torch.float16:
        return None
    return _KNOWN_PLANS.get(problem)


def _launch_validated(a, b, out, plan):
    """Launch a plan after operand and output validation has completed."""
    m, k = a.shape
    _, n = b.shape
    tile_m, tile_n, local_split_u, wave_k, k_width = plan
    if m > tile_m:
        raise InvalidInput(f"gfx950 LocalSplitU plan covers at most {tile_m} rows; got M={m}")
    launch_options = {"sink_insts_to_avoid_spills": True}
    _local_split_u_kernel[(triton.cdiv(n, tile_n), )](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        TILE_M=tile_m,
        TILE_N=tile_n,
        K_WIDTH=k_width,
        WAVE_K=wave_k,
        LOCAL_SPLIT_U=local_split_u,
        num_warps=local_split_u,
        num_stages=1,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
        **launch_options,
    )
    return out


def _register_plan_for(a, b):
    problem = _problem_for(a, b)
    if problem is None:
        return None
    m, n, k = problem
    if min(m, n, k) <= 0:
        return None
    return _register_plan_for_shape(m, n, k, a.dtype)


def _lds_plan_for(a, b):
    problem = _problem_for(a, b)
    if problem is None:
        return None
    m, n, k = problem
    if min(m, n) <= 0 or k < 128 or k * a.element_size() % 16 != 0:
        return None
    block_m, block_n, split_k = _lds_plan_for_shape(m, n, k)
    if not _valid_lds_split(k, split_k, a.element_size()):
        return None
    return block_m, block_n, split_k


def _valid_lds_split(k, split_k, element_size):
    if k % split_k == 0:
        split_size = k // split_k
    elif k % BLOCK_K == 0:
        split_size = (k // BLOCK_K // split_k) * BLOCK_K
    else:
        return False
    return (split_size >= 2 * BLOCK_K and split_size * element_size % 16 == 0)


@lru_cache(maxsize=None)
def _dispatch_plan(m, n, k, dtype, element_size):
    if dtype == torch.float16:
        local_split_u_plan = _KNOWN_PLANS.get((m, n, k))
        if local_split_u_plan is not None:
            return "local_split_u", local_split_u_plan
        persistent_plan = _persistent_plan_for_shape(m, n, k, dtype)
        if persistent_plan is not None:
            return "persistent", persistent_plan
    strong_lds_plan = _strong_lds_plan(m, n, k)
    if strong_lds_plan is not None:
        return "lds", strong_lds_plan
    register_plan = _register_plan_for_shape(m, n, k, dtype)
    if register_plan is not None:
        return "register", register_plan
    if min(m, n) <= 0 or k < 128 or k * element_size % 16 != 0:
        return None
    block_m, block_n, split_k = _lds_plan_for_shape(m, n, k)
    if not _valid_lds_split(k, split_k, element_size):
        return None
    return "lds", (block_m, block_n, split_k)


def _dispatch_for(a, b):
    problem = _problem_for(a, b)
    if problem is None:
        return None
    return _dispatch_plan(*problem, a.dtype, a.element_size())


def supports(a, b):
    """Return whether a and b select a validated gfx950 GEMM plan."""
    return _dispatch_for(a, b) is not None


def _launch_dispatch(a, b, out, dispatch):
    path, plan = dispatch
    if path == "persistent":
        return _launch_persistent(a, b, out=out)
    if path == "register":
        return _launch_register_plan(
            a,
            b,
            config=plan,
            out=out,
            _validated=True,
        )
    if path == "lds":
        block_m, block_n, split_k = plan
        return _launch_lds(
            a,
            b,
            SPLIT_K=split_k,
            TILE=(block_m, block_n),
            out=out,
        )
    return _launch_validated(a, b, out, plan)


def matmul(a, b, out=None):
    """Run the selected gfx950 GEMM specialization."""
    dispatch = _dispatch_for(a, b)
    if dispatch is None:
        raise InvalidInput("gfx950 mm does not support "
                           f"a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    m, _ = a.shape
    _, n = b.shape
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    elif not isinstance(out, torch.Tensor):
        raise InvalidInput("gfx950 mm output must be a torch.Tensor; "
                           f"got {type(out).__name__}")
    elif out.shape != (m, n):
        raise InvalidInput(f"gfx950 mm output shape must be {(m, n)}; "
                           f"got {tuple(out.shape)}")
    elif out.dtype != a.dtype:
        raise InvalidInput(f"gfx950 mm output dtype must be {a.dtype}; "
                           f"got {out.dtype}")
    elif out.device != a.device:
        raise InvalidInput(f"gfx950 mm output device must be {a.device}; "
                           f"got {out.device}")

    return _launch_dispatch(a, b, out, dispatch)


def mm(a, b, *, space="heuristic"):
    """Run the trusted gfx950 entry selected after ``tlx.ops.mm`` validation."""
    if space != "heuristic":
        raise InvalidInput("gfx950 mm currently supports space='heuristic' only")
    if not a.is_cuda or a.stride(1) != 1 or b.stride(0) != 1:
        raise InvalidInput("gfx950 mm does not support "
                           f"a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    m, k = a.shape
    _, n = b.shape
    dispatch = _dispatch_plan(
        m,
        n,
        k,
        a.dtype,
        a.element_size(),
    )
    if dispatch is None:
        raise InvalidInput("gfx950 mm does not support "
                           f"a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    # Keep this catalog hot path inline: ``tlx.ops.mm`` already validated the
    # inputs, and another Python call is material for the small-M kernels.
    path, plan = dispatch
    if path == "persistent":
        return _launch_persistent(a, b, out=out)
    if path == "register":
        return _launch_register_plan(
            a,
            b,
            config=plan,
            out=out,
            _validated=True,
        )
    if path == "lds":
        block_m, block_n, split_k = plan
        return _launch_lds(
            a,
            b,
            SPLIT_K=split_k,
            TILE=(block_m, block_n),
            out=out,
        )
    return _launch_validated(a, b, out, plan)
