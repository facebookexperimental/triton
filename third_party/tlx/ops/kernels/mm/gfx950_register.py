"""Geometry-selected register-resident GEMM plans for gfx950."""

import torch
import triton
import triton.language as tl


_BLOCK_M = 256
_BLOCK_K = 64
_NUM_CU = 256
_MIN_KTILES_PER_SPLIT = 16


@triton.jit
def _kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
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
):
    pid = tl.program_id(0).to(tl.int32)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_mn = grid_m * grid_n

    xcd_chunk: tl.constexpr = 4
    if NUM_XCDS != 1:
        aligned = (
            grid_mn // (NUM_XCDS * xcd_chunk)
        ) * (NUM_XCDS * xcd_chunk)
        if pid < aligned:
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            pid = (
                (local_pid // xcd_chunk) * NUM_XCDS * xcd_chunk
                + xcd * xcd_chunk
                + local_pid % xcd_chunk
            )

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    input_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    input_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    offs_m = (
        input_rows
        if M % BLOCK_M == 0
        else tl.where(input_rows < M, input_rows, 0)
    )
    offs_n = (
        input_cols
        if N % BLOCK_N == 0
        else tl.where(input_cols < N, input_cols, 0)
    )
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
        a_ptrs = (
            a_ptr
            + reg_m[:, None] * stride_am
            + (k + offs_k[None, :]) * stride_ak
        )
        b_ptrs = (
            b_ptr
            + (k + offs_k[:, None]) * stride_bk
            + reg_n[None, :] * stride_bn
        )
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
    if K % BLOCK_K != 0:
        k = full_k_tiles * BLOCK_K
        k_mask = offs_k < K - k
        a_ptrs = (
            a_ptr
            + reg_m[:, None] * stride_am
            + (k + offs_k[None, :]) * stride_ak
        )
        b_ptrs = (
            b_ptr
            + (k + offs_k[:, None]) * stride_bk
            + reg_n[None, :] * stride_bn
        )
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
    output_offsets = idx_m * stride_cm + idx_n * stride_cn
    tl.store(c_ptr + output_offsets, acc, mask=mask)


def launch(a, b, *, config, bias=None, out=None):
    """Launch one validated register-resident plan."""
    m, k = a.shape
    b_k, n = b.shape
    if k != b_k:
        raise ValueError(
            f"Incompatible matrix dimensions: {tuple(a.shape)} and "
            f"{tuple(b.shape)}"
        )
    if bias is not None:
        if bias.shape != (m, n):
            raise ValueError(
                f"Bias must expand to ({m}, {n}), got {tuple(bias.shape)}"
            )
        if bias.device != a.device or bias.dtype != a.dtype:
            raise ValueError(
                "Bias and matrix operands must have matching device and dtype"
            )
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    disable_agpr = (k == 256 and n > 256) or (
        k > 512
        and (k % _BLOCK_K != 0 or m * n <= 2 * 1024 * 1024)
    )
    launch_options = (
        {"llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"),)}
        if disable_agpr
        else {}
    )
    if config["BLOCK_K"] == 128 and config["num_stages"] == 3:
        launch_options["reverse_local_assignment"] = True
    bias_ptr = bias if bias is not None else out
    grid = (
        triton.cdiv(m, config["BLOCK_M"])
        * triton.cdiv(n, config["BLOCK_N"]),
    )
    _kernel[grid](
        a,
        b,
        bias_ptr,
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
        **config,
        **launch_options,
    )
    return out


def _split_k_for(grid_mn, k):
    min_ks = _MIN_KTILES_PER_SPLIT * _BLOCK_K
    best = 1
    for split_k in range(2, _NUM_CU // grid_mn + 1):
        split_size = k // split_k
        if (
            k % split_k == 0
            and split_size >= min_ks
            and split_size % _BLOCK_K == 0
        ):
            best = split_k
    return best


def _default_lds_block_m(m, n, k):
    large_grid = triton.cdiv(m, 256) * triton.cdiv(n, 256)
    large_fill = large_grid * _split_k_for(large_grid, k)
    if large_fill >= _NUM_CU // 2:
        return 256
    small_grid = triton.cdiv(m, 128) * triton.cdiv(n, 128)
    small_fill = small_grid * _split_k_for(small_grid, k)
    return 128 if small_fill > large_fill else 256


def _full_grid_config(m, n, k):
    small_grid = triton.cdiv(m, 128) * triton.cdiv(n, 128)
    large_grid = triton.cdiv(m, 256) * triton.cdiv(n, 256)
    if not (
        k > 512
        and k % _BLOCK_K == _BLOCK_K // 2
        and large_grid < _NUM_CU <= small_grid
    ):
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


def _intermediate_config(m, n, k):
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


def plan_for(m, n, k):
    """Return the bounded register plan selected by the gfx950 geometry."""
    config = _full_grid_config(m, n, k)
    if config is not None:
        return config

    block_m = _default_lds_block_m(m, n, k)
    padded_m = triton.cdiv(m, block_m) * block_m
    is_intermediate_m = _BLOCK_M // 4 < m < 4 * _BLOCK_M
    has_high_m_padding = 4 * m < 3 * padded_m
    if not is_intermediate_m or (
        block_m == _BLOCK_M and not has_high_m_padding
    ):
        return None
    return _intermediate_config(m, n, k)
