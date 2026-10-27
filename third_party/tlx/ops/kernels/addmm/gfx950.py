"""MI350X (gfx950/CDNA4) ``tlx.ops.addmm`` implementation."""

import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.inductor.gfx950_addmm_warppipe_core import (
    gfx950_addmm_compute_register,
    gfx950_addmm_warppipe_compute_async,
)

from ..mm import gfx950 as _mm


_WARPPIPE_CONFIGS = (
    (64, 64, 128, 8, 8, 3),
    (64, 64, 64, 8, 8, 3),
    (128, 128, 64, 8, 8, 2),
    (64, 128, 64, 8, 8, 2),
    (128, 256, 64, 8, 8, 2),
    (128, 256, 64, 8, 8, 3),
    (128, 256, 32, 8, 8, 3),
)

_ADDMM_TUNED_CONFIGS = {
    (4_096, 192, 2_048, torch.float16): (64, 64, 128, 8, 8, 3),
    (4_096, 192, 2_048, torch.bfloat16): (64, 64, 128, 8, 8, 3),
    (1_024, 1_024, 864, torch.float16): (64, 64, 128, 8, 8, 3),
    (64_000, 256, 256, torch.float16): (128, 128, 64, 8, 8, 2),
    (61_440, 5_120, 2_048, torch.float16): (128, 256, 64, 8, 8, 3),
    (2_252_800, 256, 256, torch.float16): (128, 256, 32, 8, 8, 3),
    (61_440, 3_840, 4_096, torch.float16): (128, 256, 64, 8, 8, 3),
    (61_440, 5_120, 7_744, torch.float16): (128, 256, 64, 8, 8, 3),
    (1_024, 6_144, 4_096, torch.float16): (128, 256, 64, 8, 8, 3),
    (4_096, 2_048, 6_144, torch.float16): (128, 256, 64, 8, 8, 3),
    (4_096, 2_048, 4_096, torch.float16): (128, 256, 64, 8, 8, 3),
    (1_024, 4_096, 6_144, torch.float16): (128, 256, 64, 8, 8, 3),
    (256, 2_048, 359, torch.float16): (64, 64, 128, 8, 8, 3),
    (256, 282, 282, torch.float16): (64, 64, 128, 8, 8, 3),
    (3_072, 2_048, 2_048, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 10_240, 128, torch.float16): (128, 256, 32, 8, 8, 3),
    (256, 1_024, 2_048, torch.float16): (64, 64, 128, 8, 8, 3),
    (3_072, 3, 1_024, torch.float16): (64, 64, 128, 8, 8, 3),
    (3_072, 2_048, 797, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 242_176, 1_892, torch.float16): (128, 256, 64, 8, 8, 2),
    (3_072, 5_888, 44_536, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 38_848, 607, torch.float16): (128, 256, 64, 8, 8, 2),
    (3_072, 2_048, 6_640, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 40_440, 256, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 22_240, 512, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 44_536, 128, torch.float16): (128, 256, 32, 8, 8, 3),
    (3_072, 44_032, 128, torch.float16): (128, 256, 32, 8, 8, 3),
    (3_072, 2_304, 5_888, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 128, 44_536, torch.float16): (64, 64, 128, 8, 8, 3),
    (3_072, 5_888, 2_304, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 2_048, 6_144, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 2_304, 5_120, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 1_536, 5_888, torch.float16): (128, 256, 64, 8, 8, 3),
    (3_072, 2_048, 3_965, torch.float16): (128, 256, 64, 8, 8, 3),
    (2_048, 9_216, 2_304, torch.bfloat16): (128, 256, 64, 8, 8, 3),
    (3_072, 2_304, 4_608, torch.float16): (128, 256, 64, 8, 8, 3),
    (1_024, 4_096, 6_145, torch.bfloat16): (128, 128, 64, 8, 8, 2),
    (3_072, 5_888, 1_536, torch.float16): (128, 256, 64, 8, 8, 3),
}

_ADDMM_SPLIT_K = {
    (1_024, 4_096, 6_144, torch.float16): 2,
    (256, 1_024, 2_048, torch.float16): 4,
    (3_072, 3, 1_024, torch.float16): 2,
    (3_072, 128, 44_536, torch.float16): 2,
}


@triton.jit
def _addmm_warppipe_kernel(
    bias_ptr,
    a_ptr,
    b_ptr,
    split_k_ws_ptr,
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
    NUM_BUFFERS: tl.constexpr,
    USE_ASYNC: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PREFETCH_BIAS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    split_id = pid % SPLIT_K
    pid = pid // SPLIT_K
    grid_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    grid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    grid_mn: tl.constexpr = grid_m * grid_n
    xcd_chunk: tl.constexpr = 4
    aligned: tl.constexpr = grid_mn // (NUM_XCDS * xcd_chunk) * (NUM_XCDS * xcd_chunk)
    if pid < aligned:
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        pid = local_pid // xcd_chunk * (NUM_XCDS * xcd_chunk) + xcd * xcd_chunk + local_pid % xcd_chunk

    width: tl.constexpr = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    safe_rows = rows % M
    safe_cols = cols % N
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
    a_base = safe_rows[:, None] * stride_am
    b_base = safe_cols * stride_bn
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    bias_ptrs = bias_ptr + rows[:, None] * stride_bias_m + cols[None, :] * stride_bias_n

    if USE_ASYNC:
        k_iters: tl.constexpr = K // BLOCK_K
        base_iters: tl.constexpr = k_iters // SPLIT_K
        remainder_iters: tl.constexpr = k_iters - base_iters * SPLIT_K
        k_lo = split_id * base_iters + tl.minimum(split_id, remainder_iters)
        n_iters = base_iters + (split_id < remainder_iters).to(tl.int32)
        tail_width = (K - k_iters * BLOCK_K) * (split_id == SPLIT_K - 1).to(tl.int32)
        smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
        smem_b = tlx.local_alloc((BLOCK_N, BLOCK_K), tlx.dtype_of(b_ptr), NUM_BUFFERS)
        acc, bias = gfx950_addmm_warppipe_compute_async(
            a_ptr,
            b_ptr,
            smem_a,
            smem_b,
            a_base,
            b_base,
            offs_k,
            k_lo,
            n_iters,
            k_iters * BLOCK_K,
            tail_width,
            stride_ak,
            stride_bk,
            bias_ptrs if SPLIT_K == 1 else a_ptr,
            mask if SPLIT_K == 1 else False,
            SPLIT_K == 1 and PREFETCH_BIAS,
            False,
            tl.float32,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            NUM_BUFFERS,
        )
        if SPLIT_K == 1 and not PREFETCH_BIAS:
            bias = tl.load(bias_ptrs, mask=mask, other=0.0).to(tl.float32)
    else:
        acc = gfx950_addmm_compute_register(
            a_ptr,
            b_ptr,
            a_base,
            b_base,
            offs_k,
            K,
            stride_ak,
            stride_bk,
            False,
            tl.float32,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )
        bias = tl.load(bias_ptrs, mask=mask, other=0.0).to(tl.float32)
    if SPLIT_K == 1:
        output_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
        tl.store(output_ptrs, (acc + bias).to(c_ptr.dtype.element_ty), mask=mask)
    else:
        workspace_offsets = split_id * M * N + rows[:, None] * N + cols[None, :]
        tl.store(split_k_ws_ptr + workspace_offsets, acc, mask=mask)


def _warppipe_triton_config(config, use_async, split_k=1, prefetch_bias=True):
    block_m, block_n, block_k, group_m, num_warps, num_buffers = config
    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": 8,
            "NUM_BUFFERS": num_buffers,
            "USE_ASYNC": use_async,
            "SPLIT_K": split_k,
            "PREFETCH_BIAS": prefetch_bias,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": 0,
            "kpack": 1,
        },
        num_warps=num_warps,
        num_stages=1 if use_async else 3,
    )


@functools.lru_cache(maxsize=None)
def _tuned_warppipe(m, n, k, dtype, use_async):
    del dtype
    configs = [
        _warppipe_triton_config(config, use_async, prefetch_bias=prefetch_bias)
        for config in _WARPPIPE_CONFIGS
        for prefetch_bias in ((True, False) if use_async else (False, ))
        if not use_async or k // config[2] > config[5]
        if not ((m, n, k) == (3_072, 1_024, 2_048) and config[:3] == (64, 64, 64))
    ]
    return triton.autotune(configs=configs, key=[])(_addmm_warppipe_kernel)


def _launch_warppipe(a, b, bias, out, *, tune, config=None, split_k=1, prefetch_bias=None):
    m, k = a.shape
    n = b.shape[1]
    if config is None:
        config = _WARPPIPE_CONFIGS[0]
    use_async = k * a.element_size() % 16 == 0 and k // config[2] > config[5]
    if prefetch_bias is None:
        prefetch_bias = use_async and m * n < 8 * 1024 * 1024
    kernel = _tuned_warppipe(m, n, k, a.dtype, use_async) if tune else _addmm_warppipe_kernel
    config = _warppipe_triton_config(config, use_async, split_k, prefetch_bias)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]) * meta["SPLIT_K"], )
    kwargs = {} if tune else dict(config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages)
    workspace = (torch.empty((split_k * m, n), device=a.device, dtype=torch.float32) if split_k > 1 else out)
    kernel[grid](
        bias,
        a,
        b,
        workspace,
        out,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        *bias.stride(),
        *out.stride(),
        **kwargs,
    )
    if split_k > 1:
        block_m, block_n = (128, 128) if m * n >= 2048 * 2048 else (32, 32)
        _mm._reduce_k_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
            workspace,
            bias,
            out,
            m,
            n,
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            SPLIT_K=split_k,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            OUTPUT_DTYPE=_mm._TORCH_TO_TL[a.dtype],
            ADD_BIAS=True,
            num_warps=8 if block_m == 128 else 4,
        )
    return out


def _validate_operands(a, b, out):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected A[M, K] and B[K, N], got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same GPU")
    if a.dtype != b.dtype:
        raise ValueError("A and B must have the same dtype")
    if any(stride <= 0 for stride in (*a.stride(), *b.stride())):
        raise ValueError("gfx950 addmm requires positive matrix strides")
    m, k = a.shape
    n = b.shape[1]
    if out is not None:
        if out.shape != (m, n) or out.device != a.device or out.dtype != a.dtype or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous {a.dtype} tensor with shape ({m}, {n}) on A's device")
    return m, n, k


def _expanded_input(input, m, n, a):
    if input.device != a.device or input.dtype != a.dtype:
        raise ValueError("input must match A's device and dtype")
    if input.ndim == 1:
        if input.shape[0] != n:
            raise ValueError(f"1-D addmm input must have shape ({n},), got {tuple(input.shape)}")
        return input.reshape(1, n).expand(m, n)
    if input.ndim == 2 and input.shape[0] in (1, m) and input.shape[1] in (1, n):
        return input.expand(m, n)
    raise ValueError(f"addmm input with shape {tuple(input.shape)} is not broadcastable to ({m}, {n})")


def _launch_uncovered_shape(a, b, bias, out):
    config = _mm._register_plan_for_shape(a.shape[0], b.shape[1], a.shape[1], a.dtype)
    if config is None:
        config = _mm._intermediate_register_config(a.shape[0], b.shape[1], a.shape[1])
    return _mm._launch_register_plan(a, b, bias=bias, config=config, out=out)


def _launch(a, b, bias, out, space):
    if space == "full":
        return _launch_warppipe(a, b, bias, out, tune=True)
    config = _ADDMM_TUNED_CONFIGS.get((a.shape[0], b.shape[1], a.shape[1], a.dtype))
    if config is not None:
        split_k = _ADDMM_SPLIT_K.get((a.shape[0], b.shape[1], a.shape[1], a.dtype), 1)
        return _launch_warppipe(a, b, bias, out, tune=False, config=config, split_k=split_k)
    dispatch = _mm._dispatch_for(a, b)
    if dispatch is None:
        return _launch_uncovered_shape(a, b, bias, out)
    return _mm._launch_dispatch(a, b, out, dispatch, bias=bias)


def addmm(input, a, b, *, out=None, space="heuristic"):
    """Compute ``input + a @ b`` with fused gfx950 GEMM epilogues.

    ``input`` may be ``(N,)`` or a two-dimensional tensor broadcastable to
    ``(M, N)``. Matrix and input scale factors are both one.
    """
    if space not in ("heuristic", "full", "smoke"):
        raise ValueError(f"Unknown gfx950 addmm search space: {space!r}")
    m, n, k = _validate_operands(a, b, out)
    bias = _expanded_input(input, m, n, a)
    result = out if out is not None else torch.empty((m, n), device=a.device, dtype=a.dtype)
    if m == 0 or n == 0:
        return result
    if k == 0:
        result.copy_(bias)
        return result
    return _launch(a, b, bias, result, space)
