"""Unpacked E4M3/E8M0 quantization for gfx950 attention backward.

Reduction groups contain 32 values. Sequence-axis quantization is performed
on the original tensor, not by transposing an already quantized operand.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rounded_mul(x, y):
    # Split-statistics launches disable FP fusion to round this product separately.
    return x * y


@triton.jit
def _centered_log_probs(scores, row_max, log_norm, sm_scale):
    scaled = _rounded_mul(scores, sm_scale * 1.4426950408889634)
    return (scaled - row_max) - log_norm


@triton.jit
def _scale_exponent(amax):
    # RCEIL(float32(amax * (1 / 448))) as an E8M0 byte. Since
    # 448 = 1.75 * 2**8, the mantissa cutoff is 0x600000; adding
    # 0x1fffff rounds strictly larger mantissas up to the next exponent.
    # Work on amax itself so subnormal scales cannot flush to zero.
    bits = amax.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(((bits + 0x1fffff) >> 23).to(tl.int32) - 8, 0)
    # At the minimum E8M0 scale, FP32 multiplication rounds the first
    # float above 448 * 2**-127 back to 2**-127 (bits 0x04600001).
    exponent = tl.where(bits <= 0x04600001, 0, exponent)
    return tl.where(bits >= 0x7f800000, 255, exponent).to(tl.uint32)


@triton.jit
def _decode_scale(exponent, RECIPROCAL: tl.constexpr = False):
    exponent = exponent.to(tl.uint32)
    if RECIPROCAL:
        bits = (254 - exponent) << 23
    else:
        # E8M0 byte 0 means 2**-127, a subnormal FP32 value, not zero.
        bits = tl.where(exponent == 0, 0x00400000, exponent << 23)
    return tl.where(exponent == 255, 0x7fc00000, bits).to(tl.float32, bitcast=True)


@triton.jit
def _quantize_store(x, Y, S, head, n, N: tl.constexpr, D: tl.constexpr, SEQUENCE: tl.constexpr, BLOCK_N: tl.constexpr,
                    TRANSPOSE_STORAGE: tl.constexpr = False):
    d = tl.arange(0, D)
    if SEQUENCE:
        grouped = x.reshape(BLOCK_N // 32, 32, D)
        amax = tl.max(tl.abs(grouped), 1)
    else:
        grouped = x.reshape(BLOCK_N, D // 32, 32)
        amax = tl.max(tl.abs(grouped), 2)
    # Zero uses E8M0 byte 0; its finite reciprocal keeps the payload zero.
    exponent = _scale_exponent(amax)
    inv_scale = _decode_scale(exponent, RECIPROCAL=True)
    if SEQUENCE:
        scaled = grouped * inv_scale[:, None, :]
    else:
        scaled = grouped * inv_scale[:, :, None]
    y = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv).reshape(BLOCK_N, D)
    if TRANSPOSE_STORAGE:
        tl.store(Y + (head * D + d[None, :]) * N + n[:, None], y, n[:, None] < N)
    else:
        tl.store(Y + (head * N + n[:, None]) * D + d[None, :], y, n[:, None] < N)
    if SEQUENCE:
        block = tl.program_id(0) * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
        tl.store(S + (head * D + d[None, :]) * (N // 32) + block[:, None], exponent.to(tl.uint8), block[:, None]
                 < N // 32)
    else:
        sd = tl.arange(0, D // 32)
        tl.store(S + (head * N + n[:, None]) * (D // 32) + sd[None, :], exponent.to(tl.uint8), n[:, None] < N)


@triton.jit
def _quantize(X, Y, S, N: tl.constexpr, D: tl.constexpr, STRIDE_B: tl.constexpr, STRIDE_H: tl.constexpr,
              STRIDE_N: tl.constexpr, STRIDE_D: tl.constexpr, H: tl.constexpr, SEQUENCE: tl.constexpr,
              BLOCK_N: tl.constexpr, TRANSPOSE_STORAGE: tl.constexpr = False):
    head = tl.program_id(1).to(tl.int64)
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, D)
    x = tl.load(X + (head // H) * STRIDE_B + (head % H) * STRIDE_H + n[:, None] * STRIDE_N + d[None, :] * STRIDE_D,
                n[:, None] < N, 0).to(tl.float32)
    _quantize_store(x, Y, S, head, n, N, D, SEQUENCE, BLOCK_N, TRANSPOSE_STORAGE)


@triton.jit
def _quantize_backward(Q, K, V, DO, QDK, KDQ, VB, DO8, DODV, QDKS, KDQS, VS, DOS, DODVS, N: tl.constexpr,
                       D: tl.constexpr, BLOCK_N: tl.constexpr, TRANSPOSE_STORAGE: tl.constexpr = False):
    head = tl.program_id(1).to(tl.int64)
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, D)
    offset = (head * N + n[:, None]) * D + d[None, :]
    mask = n[:, None] < N
    q = tl.load(Q + offset, mask, 0).to(tl.float32)
    _quantize_store(q, QDK, QDKS, head, n, N, D, True, BLOCK_N, TRANSPOSE_STORAGE)
    k = tl.load(K + offset, mask, 0).to(tl.float32)
    _quantize_store(k, KDQ, KDQS, head, n, N, D, True, BLOCK_N, TRANSPOSE_STORAGE)
    v = tl.load(V + offset, mask, 0).to(tl.float32)
    _quantize_store(v, VB, VS, head, n, N, D, False, BLOCK_N)
    do = tl.load(DO + offset, mask, 0).to(tl.float32)
    _quantize_store(do, DO8, DOS, head, n, N, D, False, BLOCK_N)
    _quantize_store(do, DODV, DODVS, head, n, N, D, True, BLOCK_N, TRANSPOSE_STORAGE)


def quantize_mxfp8(x, transpose_for_reduction=False, *, transpose_storage=False):
    """Return logical payload [B,H,N,D] and canonical uint8 scales.

    Scales are [B,H,N,D/32], or [B,H,D,N/32] for sequence reduction.
    The latter is used for Q in dK, K in dQ, and dO in dV. With
    ``transpose_storage``, these payloads have strides [..., 1, N].
    """
    if (x.ndim != 4 or any(size == 0 for size in x.shape) or x.shape[-1] % 32 or x.shape[-1] & (x.shape[-1] - 1)
            or x.shape[-2] % 32):
        raise ValueError("MXFP8 operands require nonempty rank four, N divisible by 32, and power-of-two D >= 32")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("MXFP8 quantization requires a floating-point master tensor")
    if transpose_storage and not transpose_for_reduction:
        raise ValueError("transposed payload storage is only supported for sequence reduction")
    b, h, n, d = x.shape
    shape = (b, h, d, n // 32) if transpose_for_reduction else (b, h, n, d // 32)
    with torch.cuda.device(x.device):
        y = torch.empty((b, h, d, n) if transpose_storage else x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        if transpose_storage:
            y = y.transpose(-1, -2)
        s = torch.empty(shape, dtype=torch.uint8, device=x.device)
        _quantize[(triton.cdiv(n, 32), b * h)](x, y, s, n, d, *x.stride(), h, transpose_for_reduction, 32,
                                               transpose_storage, num_warps=4)
    return y, s


def quantize_backward_operands(q, k, v, do, *, transpose_storage=False):
    """Quantize all backward-only operands in one launch.

    Return (payload, scale) pairs for Q_dK, K_dQ, V, dO, dO_dV, respectively.
    Q/K/V and contiguous dO are the BF16 master tensors, not FP8 payloads.
    """
    tensors = (q, k, v, do)
    if (q.ndim != 4 or q.shape[-1] != 128 or q.shape[-2] % 32 or any(size == 0 for size in q.shape)
            or any(x.shape != q.shape or x.device != q.device or x.dtype != torch.bfloat16 or not x.is_contiguous()
                   for x in tensors)):
        raise ValueError("backward quantization requires contiguous, same-device BF16 [B,H,N,128], N % 32 == 0")
    b, h, n, d = q.shape
    with torch.cuda.device(q.device):
        payloads = []
        for sequence in (True, True, False, False, True):
            transposed = sequence and transpose_storage
            payload = torch.empty((b, h, d, n) if transposed else q.shape, device=q.device, dtype=torch.float8_e4m3fn)
            payloads.append(payload.transpose(-1, -2) if transposed else payload)
        scales = [
            torch.empty((b, h, d, n // 32) if sequence else (b, h, n, d // 32), device=q.device, dtype=torch.uint8)
            for sequence in (True, True, False, False, True)
        ]
        _quantize_backward[(triton.cdiv(n, 32), b * h)](q, k, v, do, *payloads, *scales, n, d, 32, transpose_storage,
                                                        num_warps=4)
    return tuple(zip(payloads, scales))
