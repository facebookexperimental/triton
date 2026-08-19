"""Standalone fused addmm for gfx950 using TLX GEMM kernels.

The operation is ``out = bias + a @ b``. ``a`` must be row-major and ``b``
must be column-major so the kernel can issue coalesced loads. The bias may be
one-dimensional ``(N,)`` or broadcastable to ``(M, N)``.
"""

import torch

import triton
import triton.language as tl

from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16.matmul_kernel import (
    BLOCK_K,
    _launch,
    _launch_register,
)


_PATH_AUTOTUNE_WARMUP = 25
_PATH_AUTOTUNE_REP = 100
_PATH_CACHE: dict[tuple[object, ...], str] = {}
_TAIL_BLOCK_K = 2 * BLOCK_K
_MAX_DEFERRED_EPILOGUE_ELEMENTS = 16 * 1024 * 1024
_TAIL_CONFIGS = [
    triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n}, num_warps=num_warps)
    for block_m, block_n, num_warps in (
        (64, 64, 4),
        (64, 128, 4),
        (128, 64, 4),
        (128, 128, 8),
    )
]


def _can_use_inter_wave(a: torch.Tensor) -> bool:
    return a.shape[1] >= 2 * BLOCK_K and a.shape[1] % BLOCK_K == 0


def _can_use_inter_wave_tail(a: torch.Tensor, b: torch.Tensor) -> bool:
    M, K = a.shape
    N = b.shape[1]
    output_elements = M * N
    return (
        K > 1536
        and K % BLOCK_K != 0
        and K * a.element_size() % 16 == 0
        and a.element_size() == 2
        and 2 * 1024 * 1024 < output_elements <= _MAX_DEFERRED_EPILOGUE_ELEMENTS
    )


def _path_key(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    split_k,
) -> tuple[object, ...]:
    return (
        a.device.type,
        a.device.index,
        a.dtype,
        a.shape,
        b.shape,
        bias.stride(),
        a.stride(),
        b.stride(),
        split_k,
    )


@triton.autotune(configs=_TAIL_CONFIGS, key=["M", "N", "K_TAIL"])
@triton.jit
def _addmm_tail_kernel(
    a_ptr,
    b_ptr,
    main_ptr,
    bias_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K_OFFSET: tl.constexpr,
    K_TAIL: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, _TAIL_BLOCK_K)
    a = tl.load(
        a_ptr + offs_m[:, None] * stride_am + (K_OFFSET + offs_k[None, :]) * stride_ak,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_TAIL),
        other=0.0,
    )
    b = tl.load(
        b_ptr + (K_OFFSET + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
        mask=(offs_k[:, None] < K_TAIL) & (offs_n[None, :] < N),
        other=0.0,
    )
    acc = tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
    c_offsets = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    main = tl.load(main_ptr + c_offsets, mask=mask, other=0.0)
    bias_offsets = offs_m[:, None] * stride_bias_m + offs_n[None, :] * stride_bias_n
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)
    tl.store(c_ptr + c_offsets, main + acc + bias.to(tl.float32), mask=mask)


def _launch_inter_wave_with_tail(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run aligned K through inter-wave and finish the tail from its fp32 partials."""
    M, K = a.shape
    _, N = b.shape
    k_main = K // _TAIL_BLOCK_K * _TAIL_BLOCK_K
    main, out = _launch(a, b, SPLIT_K=1, TILE=(256, 256), K_LIMIT=k_main, DEFER_EPILOGUE=True)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _addmm_tail_kernel[grid](
        a,
        b,
        main,
        bias,
        out,
        M,
        N,
        k_main,
        K - k_main,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias.stride(0),
        bias.stride(1),
    )
    return out


def _autotune_path(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    split_k,
) -> str:
    key = _path_key(bias, a, b, split_k)
    cached = _PATH_CACHE.get(key)
    if cached is not None:
        return cached

    register = lambda: _launch_register(a, b, bias=bias)
    register_output = register()
    candidates = {"register": register}
    if _can_use_inter_wave(a):
        inter_wave = lambda: _launch(a, b, bias=bias, SPLIT_K=split_k)
        inter_wave_output = inter_wave()
        if torch.allclose(register_output, inter_wave_output, rtol=1e-2, atol=1e-2):
            candidates["inter_wave"] = inter_wave
    elif _can_use_inter_wave_tail(a, b):
        inter_wave_tail = lambda: _launch_inter_wave_with_tail(bias, a, b)
        inter_wave_tail_output = inter_wave_tail()
        if torch.allclose(register_output, inter_wave_tail_output, rtol=1e-2, atol=1e-2):
            candidates["inter_wave_tail"] = inter_wave_tail
    timings = {
        name: triton.testing.do_bench(
            candidate,
            warmup=_PATH_AUTOTUNE_WARMUP,
            rep=_PATH_AUTOTUNE_REP,
            return_mode="median",
        )
        for name, candidate in candidates.items()
    }
    winner = min(timings, key=timings.__getitem__)
    _PATH_CACHE[key] = winner
    return winner


def addmm(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor, SPLIT_K=None):
    """Return ``bias + a @ b`` using the fastest valid gfx950 TLX path."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("addmm expects two-dimensional matrix operands")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible matrix dimensions: {tuple(a.shape)} and {tuple(b.shape)}")
    if not a.is_contiguous():
        raise ValueError("a must be row-major contiguous")
    if b.stride(0) != 1:
        raise ValueError("b must be column-major (stride(0) == 1)")

    M, _ = a.shape
    _, N = b.shape
    try:
        bias_2d = torch.broadcast_to(bias, (M, N))
    except RuntimeError as error:
        raise ValueError(f"Bias shape {tuple(bias.shape)} is not broadcastable to ({M}, {N})") from error
    if SPLIT_K not in (None, 1):
        if not _can_use_inter_wave(a):
            raise ValueError("SPLIT_K is only supported by the inter-wave kernel")
        return _launch(a, b, bias=bias_2d, SPLIT_K=SPLIT_K)
    winner = _autotune_path(bias_2d, a, b, SPLIT_K)
    if winner == "inter_wave":
        return _launch(a, b, bias=bias_2d, SPLIT_K=SPLIT_K)
    if winner == "inter_wave_tail":
        return _launch_inter_wave_with_tail(bias_2d, a, b)
    return _launch_register(a, b, bias=bias_2d)
