"""Standalone fused addmm for gfx950 using TLX GEMM kernels.

The operation is ``out = bias + a @ b``. ``a`` must be row-major and ``b``
must be column-major so the kernel can issue coalesced loads. The bias may be
one-dimensional ``(N,)`` or broadcastable to ``(M, N)``.
"""

import torch

import triton

from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16.matmul_kernel import (
    BLOCK_K,
    _launch,
    _launch_register,
)


_PATH_AUTOTUNE_WARMUP = 25
_PATH_AUTOTUNE_REP = 100
_PATH_CACHE: dict[tuple[object, ...], str] = {}


def _can_use_inter_wave(a: torch.Tensor) -> bool:
    return a.shape[1] >= 2 * BLOCK_K and a.shape[1] % BLOCK_K == 0


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
    return _launch_register(a, b, bias=bias_2d)
