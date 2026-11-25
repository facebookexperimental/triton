"""FP32 MoE router logits ``x.float() @ weight.T`` for a BF16 ``x`` and an FP32 ``weight``, on Blackwell (sm100).

The method is gfx950's (see that module): each FP32 weight element is the exact sum of three BF16 values, so every
product on the BF16 matrix units is exact and accumulates in FP32, and the weight stays FP32 in memory. Its kernels
are plain Triton and run unchanged on Blackwell; this module supplies the configs, tuned on GB300 at E = 256,
K = 8192, and the same two-kernel split:

- Up to ``GEMV_MAX_TOKENS`` tokens a GEMV reads whole FP32 weight rows per token, one program per expert block and
  token, with no cross-program reduction.
- Above that, BF16 matrix tiles split K over programs and a second launch adds the partials in split order, so the
  result is deterministic. Unlike gfx950's matrix cores, the tensor cores truncate as they accumulate, so the error
  grows with the K each program covers; every tile config splits K into runs of at most 2048.

As on gfx950, the accumulation order depends on the token count's bucket.
"""

from __future__ import annotations

import torch
import triton

from . import gfx950 as _kernels

GEMV_MAX_TOKENS = 12

# Up to this many tokens: (experts per program, K per load, num_warps).
_GEMV_CONFIGS = (
    (1, 1, 8192, 8),
    (2, 2, 8192, 4),
    (6, 4, 4096, 4),
    (9, 4, 2048, 4),
    (12, 4, 4096, 4),
)
# Up to this many tokens (None: any): (BLOCK_M, BLOCK_N, BLOCK_K, splits, num_warps, num_stages).
_MFMA_CONFIGS = (
    (32, 16, 64, 128, 16, 8, 3),
    (64, 32, 64, 64, 16, 8, 3),
    (128, 32, 128, 64, 16, 8, 3),
    (512, 128, 64, 64, 16, 8, 3),
    (1024, 128, 64, 64, 8, 8, 3),
    (None, 128, 64, 64, 4, 8, 3),
)


def gemv_config(T: int, E: int, K: int):
    for max_tokens, block_n, block_k, num_warps in _GEMV_CONFIGS:
        if T <= max_tokens:
            return _kernels._largest_divisor(E, block_n, 1), _kernels._largest_divisor(K, block_k, 1024), num_warps
    raise ValueError(f"{T} tokens take the tile kernel")


def mfma_config(T: int, E: int, K: int):
    for max_tokens, block_m, block_n, block_k, splits, num_warps, num_stages in _MFMA_CONFIGS:
        if max_tokens is None or T <= max_tokens:
            block_n = _kernels._largest_divisor(E, block_n, 16)
            while splits > 1 and K % (splits * block_k):
                splits //= 2
            return block_m, block_n, block_k, splits, num_warps, num_stages
    raise AssertionError("unreachable")


def router_gemm(x, weight, out):
    T, K = x.shape
    E = weight.shape[0]
    if out is None:
        out = torch.empty(T, E, device=x.device, dtype=torch.float32)
    if T == 0:
        return out
    row64 = (T - 1) * x.stride(0) + K >= _kernels.INT32_OFFSETS
    if T <= GEMV_MAX_TOKENS:
        block_n, block_k, num_warps = gemv_config(T, E, K)
        _kernels._gemv_kernel[(E // block_n, T)](x, weight, out, x.stride(0), E, K, block_n, block_k, row64,
                                                 num_warps=num_warps)
        return out
    block_m, block_n, block_k, splits, num_warps, num_stages = mfma_config(T, E, K)
    grid = (E // block_n, splits, triton.cdiv(T, block_m))
    if splits == 1:
        _kernels._mfma_kernel[grid](x, weight, out, T, x.stride(0), E, K, K, block_m, block_n, block_k, row64,
                                    num_warps=num_warps, num_stages=num_stages)
        return out
    partials = torch.empty(splits, T, E, device=x.device, dtype=torch.float32)
    row64 = row64 or splits * T * E >= _kernels.INT32_OFFSETS
    _kernels._mfma_kernel[grid](x, weight, partials, T, x.stride(0), E, K, K // splits, block_m, block_n, block_k,
                                row64, num_warps=num_warps, num_stages=num_stages)
    block = 1024
    _kernels._reduce_splits_kernel[(triton.cdiv(T * E, block), )](partials, out, T * E, splits, block, row64,
                                                                  num_warps=4)
    return out
