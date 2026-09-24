"""Shared MI300X (gfx942/CDNA3) GEMM implementation for ``mm`` and ``addmm``.

A direct-load kernel serves both operations through a compact heuristic or
full autotune space. Short-M ``mm`` shapes use an intra-CTA K split instead.
"""

import functools
from typing import NamedTuple

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


class _LocalSplitUPlan(NamedTuple):
    tile_m: int
    tile_n: int
    local_split_u: int
    wave_k: int
    k_width: int


# Debug/ablation toggle. Eligibility remains restricted by
# ``_precheck_local_split_u`` below.
ENABLE_LOCAL_SPLIT_U = True

_MEASURED_LOCAL_SPLIT_U_PLANS = {
    # These cover the current gfx942_2 focus shapes and can be generalized after broader measurements.
    (7, 8192, 2048):
    _LocalSplitUPlan(tile_m=16, tile_n=32, local_split_u=16, wave_k=64, k_width=8),
    (7, 2048, 4096):
    _LocalSplitUPlan(tile_m=16, tile_n=16, local_split_u=16, wave_k=64, k_width=8),
}


@triton.jit
def _load_local_split_u_operands_gfx942(
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
    split_k = macro_k_base + split_ids[:, None, None] * WAVE_K
    a_offsets = global_rows[None, :, None] * stride_am + (split_k + rk[None, None, :]) * stride_ak
    b_offsets = (split_k + rk[None, :, None]) * stride_bk + global_cols[None, None, :] * stride_bn
    a_offsets = tlx.require_layout(a_offsets, dot_a)
    b_offsets = tlx.require_layout(b_offsets, dot_b)
    a = tlx.buffer_load(a_ptr, a_offsets, contiguity=K_WIDTH)
    b = tlx.buffer_load(b_ptr, b_offsets, cache=".cg", contiguity=K_WIDTH)
    return a, b


@triton.jit
def _local_split_u_kernel_gfx942(
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
    """Compute a short-M tile with one K partition per wave."""
    MACRO_K: tl.constexpr = WAVE_K * LOCAL_SPLIT_U
    tl.static_assert(K % MACRO_K == 0)

    tl.static_assert(M <= TILE_M)
    pid_n = tl.program_id(0).to(tl.int32)
    split_ids = tl.arange(0, LOCAL_SPLIT_U).to(tl.int32)
    rows = tl.arange(0, TILE_M).to(tl.int32)
    global_rows = tl.where(rows < M, rows, 0)
    local_cols = tl.arange(0, TILE_N).to(tl.int32)
    output_cols = pid_n * TILE_N + local_cols
    global_cols = tl.where(output_cols < N, output_cols, 0)
    rk = tl.arange(0, WAVE_K).to(tl.int32)

    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[LOCAL_SPLIT_U, 1, 1],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=K_WIDTH)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=K_WIDTH)
    acc = tlx.zeros((LOCAL_SPLIT_U, TILE_M, TILE_N), tl.float32, layout=mma)

    current_a, current_b = _load_local_split_u_operands_gfx942(
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
    for macro in tl.range(0, K // MACRO_K - 1, num_stages=1):
        next_a, next_b = _load_local_split_u_operands_gfx942(
            a_ptr,
            b_ptr,
            global_rows,
            global_cols,
            split_ids,
            rk,
            (macro + 1) * MACRO_K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            WAVE_K,
            K_WIDTH,
            dot_a,
            dot_b,
        )
        acc = tl.dot(current_a, current_b, acc, allow_tf32=False, out_dtype=tl.float32)
        current_a = next_a
        current_b = next_b
    acc = tl.dot(current_a, current_b, acc, allow_tf32=False, out_dtype=tl.float32)

    if LOCAL_SPLIT_U == 16 and TILE_N == 32:
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

    result = tl.reshape(
        tlx.local_load(tlx.local_slice(partial_view, [0, 0, 0], [1, TILE_M, TILE_N])),
        (TILE_M, TILE_N),
    )
    for split in tl.static_range(1, LOCAL_SPLIT_U):
        partial = tlx.local_load(tlx.local_slice(partial_view, [split, 0, 0], [1, TILE_M, TILE_N]))
        result += tl.reshape(partial, (TILE_M, TILE_N))

    output_rows = tl.arange(0, TILE_M).to(tl.int32)
    output_ptrs = c_ptr + output_rows[:, None] * stride_cm + output_cols[None, :] * stride_cn
    tl.store(
        output_ptrs,
        result.to(c_ptr.dtype.element_ty),
        mask=(output_rows[:, None] < M) & (output_cols[None, :] < N),
    )


@triton.jit
def _direct_matmul_kernel_gfx942(
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
    XCD_CHUNK: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    SPLIT_M_128_32: tl.constexpr = False,
):
    """Register-staged GEMM with per-operand cache and XCD policy."""
    pid = tl.program_id(0).to(tl.int32)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    grid_mn = grid_m * grid_n

    # Stripe complete chunks over the eight XCDs.  Leave a short tail in its
    # original order so no remapped pid can escape the output-tile grid.
    if NUM_XCDS != 1:
        aligned = (grid_mn // (NUM_XCDS * XCD_CHUNK)) * (NUM_XCDS * XCD_CHUNK)
        if pid < aligned:
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            pid = ((local_pid // XCD_CHUNK) * NUM_XCDS * XCD_CHUNK + xcd * XCD_CHUNK + local_pid % XCD_CHUNK)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    if SPLIT_M_128_32:
        # Triton tensor dimensions must be powers of two. Represent BM=160 as
        # two panels while sharing the B tile and K loop.
        tl.static_assert(BLOCK_M == 160)
        tl.static_assert(K % BLOCK_K == 0)
        base_m = pid_m * BLOCK_M
        base_n = pid_n * BLOCK_N
        offs_m0 = (base_m + tl.arange(0, 128).to(tl.int32)) % M
        offs_m1 = (base_m + 128 + tl.arange(0, 32).to(tl.int32)) % M
        offs_n = (base_n + tl.arange(0, BLOCK_N).to(tl.int32)) % N
        offs_k = tl.arange(0, BLOCK_K).to(tl.int32)

        acc0 = tl.zeros((128, BLOCK_N), tl.float32)
        acc1 = tl.zeros((32, BLOCK_N), tl.float32)
        for k in range(0, K, BLOCK_K):
            b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
            a0_ptrs = a_ptr + offs_m0[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
            a1_ptrs = a_ptr + offs_m1[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
            b = tl.load(b_ptrs)
            a0 = tl.load(a0_ptrs)
            a1 = tl.load(a1_ptrs)
            acc0 = tl.dot(a0, b, acc0, allow_tf32=False, out_dtype=tl.float32)
            acc1 = tl.dot(a1, b, acc1, allow_tf32=False, out_dtype=tl.float32)

        rows0 = base_m + tl.arange(0, 128).to(tl.int32)
        rows1 = base_m + 128 + tl.arange(0, 32).to(tl.int32)
        cols = base_n + tl.arange(0, BLOCK_N).to(tl.int32)
        idx_n = cols[None, :]
        idx_m0 = rows0[:, None]
        idx_m1 = rows1[:, None]
        mask0 = (idx_m0 < M) & (idx_n < N)
        mask1 = (idx_m1 < M) & (idx_n < N)
        if ADD_BIAS:
            bias0 = tl.load(
                bias_ptr + idx_m0 * stride_bias_m + idx_n * stride_bias_n,
                mask=mask0,
                eviction_policy="evict_last",
            )
            bias1 = tl.load(
                bias_ptr + idx_m1 * stride_bias_m + idx_n * stride_bias_n,
                mask=mask1,
                eviction_policy="evict_last",
            )
            acc0 += bias0.to(tl.float32)
            acc1 += bias1.to(tl.float32)
        tl.store(c_ptr + idx_m0 * stride_cm + idx_n * stride_cn, acc0, mask=mask0)
        tl.store(c_ptr + idx_m1 * stride_cm + idx_n * stride_cn, acc1, mask=mask1)
    else:
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)) % N
        offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
        reg_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
        reg_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        even_k = K % BLOCK_K == 0
        k_main = K if even_k else (K // BLOCK_K) * BLOCK_K
        for k in range(0, k_main, BLOCK_K):
            a_ptrs = a_ptr + reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
            b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
        if not even_k:
            a_ptrs = a_ptr + reg_m[:, None] * stride_am + (k_main + offs_k[None, :]) * stride_ak
            b_ptrs = b_ptr + (k_main + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
            tail = offs_k < K - k_main
            a = tl.load(a_ptrs, mask=tail[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=tail[:, None], other=0.0)
            acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
        idx_m = rows[:, None]
        idx_n = cols[None, :]
        mask = (idx_m < M) & (idx_n < N)
        if ADD_BIAS:
            bias = tl.load(
                bias_ptr + idx_m * stride_bias_m + idx_n * stride_bias_n,
                mask=mask,
                eviction_policy="evict_last",
            )
            acc += bias.to(tl.float32)
        tl.store(c_ptr + idx_m * stride_cm + idx_n * stride_cn, acc, mask=mask)


@triton.jit
def matmul_kernel_gfx942(
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
    XCD_CHUNK: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    SPLIT_M_128_32: tl.constexpr = False,
    USE_LOCAL_SPLIT_U: tl.constexpr = False,
    LOCAL_SPLIT_U: tl.constexpr = 1,
    K_WIDTH: tl.constexpr = 1,
):
    """Dispatch one autotune candidate to its selected GEMM implementation."""
    if USE_LOCAL_SPLIT_U:
        _local_split_u_kernel_gfx942(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M,
            BLOCK_N,
            K_WIDTH,
            BLOCK_K,
            LOCAL_SPLIT_U,
        )
    else:
        _direct_matmul_kernel_gfx942(
            a_ptr,
            b_ptr,
            bias_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_bias_m,
            stride_bias_n,
            stride_cm,
            stride_cn,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_M,
            NUM_XCDS,
            XCD_CHUNK,
            ADD_BIAS,
            SPLIT_M_128_32,
        )


def _config(block_m, block_n, block_k, group_m, num_warps, *, waves_per_eu=0, kpack=1, split_m_128_32=False):
    # This overlaps register-staged global loads; it is not an explicit
    # two-buffer LDS allocation.
    meta = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "NUM_XCDS": 8,
        "XCD_CHUNK": 8,
        "waves_per_eu": waves_per_eu,
        "kpack": kpack,
    }
    if split_m_128_32:
        meta["SPLIT_M_128_32"] = True
    return triton.Config(meta, num_warps=num_warps, num_stages=2)


def _local_split_u_config(plan):
    return triton.Config(
        {
            "BLOCK_M": plan.tile_m,
            "BLOCK_N": plan.tile_n,
            "BLOCK_K": plan.wave_k,
            "GROUP_M": 1,
            "NUM_XCDS": 1,
            "XCD_CHUNK": 1,
            "USE_LOCAL_SPLIT_U": True,
            "LOCAL_SPLIT_U": plan.local_split_u,
            "K_WIDTH": plan.k_width,
            "waves_per_eu": 0,
            "kpack": 1,
        },
        num_warps=plan.local_split_u,
        num_stages=1,
    )


def _configs():
    """Compact generic search space for the direct-load kernel."""
    return [
        _config(64, 64, 64, 4, 4),
        _config(64, 64, 128, 8, 8),
        _config(128, 64, 64, 4, 8),
        _config(64, 128, 64, 8, 8),
        _config(128, 128, 32, 8, 4),
        _config(128, 128, 64, 8, 8),
        _config(256, 128, 32, 8, 8),
        _config(128, 256, 32, 8, 8),
        _config(256, 256, 64, 8, 8),
    ]


CONFIGS = _configs


def _smoke_configs():
    return [_config(64, 64, 64, 4, 4), _config(128, 128, 32, 8, 4)]


SMOKE_CONFIGS = _smoke_configs


def heuristic_config(M, N, K):
    """Choose one direct-load configuration without runtime autotuning."""
    if (M, N, K) == (2048, 10240, 25408):
        return [_config(160, 512, 32, 8, 8, split_m_128_32=True)]
    if min(M, N) <= 64:
        return [_config(64, 64, 64, 4, 4)]
    if K <= 256:
        return [_config(128, 128, 32, 8, 4)]
    wide_workgroups = triton.cdiv(M, 256) * triton.cdiv(N, 256)
    if M >= 2048 and N >= 2048 and wide_workgroups >= 256:
        return [_config(256, 256, 64, 8, 8)]
    if M < N:
        return [_config(128, 256, 32, 8, 8)]
    if N < M:
        return [_config(256, 128, 32, 8, 8)]
    return [_config(128, 128, 64, 8, 8)]


def _candidate_configs(shape, enable_local_split_u=False):
    """Return the candidate universe, including a shape-specific incumbent."""
    configs = CONFIGS()
    if enable_local_split_u:
        configs.append(_local_split_u_config(_MEASURED_LOCAL_SPLIT_U_PLANS[shape]))
    incumbent = heuristic_config(*shape)[0]
    incumbent_key = (incumbent.kwargs, incumbent.num_warps, incumbent.num_stages)
    if not any((config.kwargs, config.num_warps, config.num_stages) == incumbent_key for config in configs):
        configs.append(incumbent)
    return configs


@functools.lru_cache(maxsize=None)
def _tuned(space, shape=None, enable_local_split_u=False):
    """Autotuned GEMM kernel per search space."""
    if space == "heuristic":
        configs = heuristic_config(*shape)
    elif space == "full":
        configs = _candidate_configs(shape, enable_local_split_u)
    elif space == "smoke":
        configs = SMOKE_CONFIGS()
    else:
        raise ValueError(f"Unknown gfx942 MM search space: {space}")
    return triton.autotune(configs=configs, key=["M", "N", "K", "ADD_BIAS"])(matmul_kernel_gfx942)


def _validate_operands(a, b, out):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected A[M, K] and B[K, N], got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same GPU")
    if a.dtype != b.dtype:
        raise ValueError("A and B must have the same dtype")
    M, K = a.shape
    N = b.shape[1]
    if out is not None:
        if out.shape != (M, N) or out.device != a.device or out.dtype != a.dtype or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous {a.dtype} tensor with shape ({M}, {N}) on A's device")
    return M, N, K


def _bias_strides(bias, M, N, a):
    if bias.device != a.device or bias.dtype != a.dtype:
        raise ValueError("input must match A's device and dtype")
    if bias.ndim == 1:
        if bias.shape[0] != N:
            raise ValueError(f"1-D addmm input must have shape ({N},), got {tuple(bias.shape)}")
        return 0, bias.stride(0)
    if bias.ndim == 2 and bias.shape[0] in (1, M) and bias.shape[1] in (1, N):
        return (0 if bias.shape[0] == 1 else bias.stride(0), 0 if bias.shape[1] == 1 else bias.stride(1))
    raise ValueError(f"addmm input with shape {tuple(bias.shape)} is not broadcastable to ({M}, {N})")


def _precheck_local_split_u(a, b, bias):
    """Enable LocalSplitU only for measured MM shapes and layouts."""
    shape = (a.shape[0], b.shape[1], a.shape[1])
    return (ENABLE_LOCAL_SPLIT_U and bias is None and a.dtype == torch.float16 and a.stride(1) == 1 and b.stride(0) == 1
            and shape in _MEASURED_LOCAL_SPLIT_U_PLANS)


def _launch_local_split_u(a, b, out, plan):
    M, K = a.shape
    N = b.shape[1]
    _local_split_u_kernel_gfx942[(triton.cdiv(N, plan.tile_n), )](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        TILE_M=plan.tile_m,
        TILE_N=plan.tile_n,
        K_WIDTH=plan.k_width,
        WAVE_K=plan.wave_k,
        LOCAL_SPLIT_U=plan.local_split_u,
        num_warps=plan.local_split_u,
        num_stages=1,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
    )
    return out


def _gemm(a, b, bias=None, *, out=None, space="heuristic"):
    M, N, K = _validate_operands(a, b, out)
    bias_strides = _bias_strides(bias, M, N, a) if bias is not None else (0, 0)
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    enable_local_split_u = _precheck_local_split_u(a, b, bias)
    if space == "heuristic" and enable_local_split_u:
        plan = _MEASURED_LOCAL_SPLIT_U_PLANS[(M, N, K)]
        return _launch_local_split_u(a, b, out, plan)

    def grid(meta):
        if meta.get("USE_LOCAL_SPLIT_U", False):
            return (triton.cdiv(N, meta["BLOCK_N"]), )
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )

    shape = (M, N, K) if space in ("heuristic", "full") else None
    kernel = _tuned(space, shape, space == "full" and enable_local_split_u)
    bias_ptr = bias if bias is not None else out
    kernel[grid](
        a,
        b,
        bias_ptr,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias_strides[0],
        bias_strides[1],
        out.stride(0),
        out.stride(1),
        ADD_BIAS=bias is not None,
        matrix_instr_nonkdim=16,
    )
    return out


def mm(a, b, *, out=None, space="heuristic"):
    """Compute ``a @ b`` using the gfx942 direct-load GEMM kernel."""
    return _gemm(a, b, out=out, space=space)


# Compatibility entry point used by the kernel-optimization agent.
matmul = mm
