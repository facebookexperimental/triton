# K=96 NVFP4 (mxf4nvf4 .block16) end-to-end correctness test.
#
# This is a SM103a-only feature (Blackwell B300). The kernel uses
# tl.dot_scaled with two NVFP4 (E2M1 + E4M3 scale) operands and BLOCK_K=96.
# Triton emits a single tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16
# per (M, N) tile that consumes K=96 in one shot.
#
# This test must be RUN on a B300 (sm_103a). It will skip on any other
# device. The compiler-side tests (lit + C++ unittests) cover the descriptor
# encoding and verifier behaviour without hardware.

import pytest
import torch
import triton
import triton.language as tl


def _is_sm103():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    # sm_103a presents as compute capability 10.3.
    return props.major == 10 and props.minor == 3


@triton.jit
def _nvfp4_k96_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # E2M1 packed: 2 fp4 per i8.
    PACKED_K: tl.constexpr = BLOCK_K // 2
    SCALE_K: tl.constexpr = BLOCK_K // 16  # = 6 for K=96

    a_offsets = (tl.arange(0, BLOCK_M)[:, None] * stride_am + tl.arange(0, PACKED_K)[None, :] * stride_ak)
    b_offsets = (tl.arange(0, PACKED_K)[:, None] * stride_bk + tl.arange(0, BLOCK_N)[None, :] * stride_bn)
    a = tl.load(a_ptr + a_offsets)
    b = tl.load(b_ptr + b_offsets)

    a_scale = tl.load(a_scale_ptr + tl.arange(0, BLOCK_M)[:, None] * SCALE_K + tl.arange(0, SCALE_K)[None, :])
    b_scale = tl.load(b_scale_ptr + tl.arange(0, BLOCK_N)[:, None] * SCALE_K + tl.arange(0, SCALE_K)[None, :])

    c = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1")

    out_offsets = (tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
    tl.store(out_ptr + out_offsets, c.to(tl.float32))


@pytest.mark.skipif(not _is_sm103(), reason="K=96 mxf4nvf4 (.block16) requires sm_103a (B300)")
def test_nvfp4_k96_dot_scaled_correctness():
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 96
    PACKED_K = BLOCK_K // 2
    SCALE_K = BLOCK_K // 16

    device = "cuda"
    torch.manual_seed(0)
    # E2M1 (fp4) operands must be packed as uint8 for tl.dot_scaled (NOT int8 —
    # the frontend rejects int8). The reference below views as uint8 anyway.
    a_packed = torch.randint(0, 256, (BLOCK_M, PACKED_K), dtype=torch.uint8, device=device)
    b_packed = torch.randint(0, 256, (PACKED_K, BLOCK_N), dtype=torch.uint8, device=device)
    # E4M3 scales (FP8). Constrain to small-magnitude representable values
    # so the reference dequant + matmul fits comfortably in fp32.
    a_scale = (torch.randn(BLOCK_M, SCALE_K, device=device).to(torch.float8_e4m3fn))
    b_scale = (torch.randn(BLOCK_N, SCALE_K, device=device).to(torch.float8_e4m3fn))

    out = torch.empty(BLOCK_M, BLOCK_N, device=device, dtype=torch.float32)

    _nvfp4_k96_kernel[(1, )](
        a_packed,
        a_scale,
        b_packed,
        b_scale,
        out,
        a_packed.stride(0),
        a_packed.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Reference: dequantize fp4 + dequantize scales + fp32 matmul.
    # Use the same E2M1 lookup-table semantics the hardware applies.
    def _dequant_fp4_e4m3(packed_i8: torch.Tensor, scale_e4m3: torch.Tensor, M: int, K: int) -> torch.Tensor:
        # Each i8 holds two E2M1 values; broadcast scales (one per 16 K
        # elements) across the 16-element block.
        # E2M1 lookup table (sign + 2-bit exponent + 1-bit mantissa).
        # values for unsigned 4-bit nibble 0..7: [0, 0.5, 1, 1.5, 2, 3, 4, 6];
        # nibble 8..15 are the negatives of 0..7.
        lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                           device=packed_i8.device, dtype=torch.float32)
        ui = packed_i8.view(torch.uint8).to(torch.long)
        lo = lut[ui & 0xF]
        hi = lut[(ui >> 4) & 0xF]
        # interleave [lo, hi, lo, hi, ...] along K
        unpacked = torch.stack((lo, hi), dim=-1).reshape(M, K)
        scale_f32 = scale_e4m3.to(torch.float32)
        # Broadcast scale across 16-element K blocks.
        scale_bcast = scale_f32.repeat_interleave(16, dim=-1)
        return unpacked * scale_bcast

    a_deq = _dequant_fp4_e4m3(a_packed, a_scale, BLOCK_M, BLOCK_K)
    b_deq = _dequant_fp4_e4m3(b_packed.t().contiguous(), b_scale, BLOCK_N, BLOCK_K).t()
    ref = a_deq @ b_deq

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
