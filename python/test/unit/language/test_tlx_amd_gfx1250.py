"""TLX AMD tests -- gfx1250."""
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hip_gfx1250
from triton.language.extra.tlx.tutorials.amd_mxfp_gemm_tdm_pipelined import (
    matmul as _amd_mxfp_matmul,
    pack_scale as _amd_mxfp_pack_scale,
)
from triton.tools.mxfp import MXScaleTensor
from triton.language.extra.tlx.tutorials.amd_fa_tdm_pipelined import attention as _amd_fa_tdm_attention
from triton.language.extra.tlx.tutorials.amd_tdm_gemm_pipelined import (
    matmul as _amd_tdm_matmul,
    matmul_tdm_pipelined_single_warp_per_simd_schedule as _amd_tdm_single_warp_matmul,
)


@triton.jit
def _async_amd_desc_load_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _async_amd_desc_load_fused_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, N], [N, 1], [M, N])
    b_desc = tl.make_tensor_descriptor(b_ptr, [M, N], [N, 1], [M, N])
    a_buf = tlx.local_alloc((M, N), tl.float16, 1)
    b_buf = tlx.local_alloc((M, N), tl.float16, 1)
    a_smem = tlx.local_view(a_buf, 0)
    b_smem = tlx.local_view(b_buf, 0)
    a_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    b_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    token = tlx.async_amd_descriptor_load_fused([
        (a_desc, a_smem, 0b0011),
        (b_desc, b_smem, 0b1100),
    ])
    tlx.async_amd_descriptor_wait(tokens=[token])
    result = tlx.local_load(a_smem) + tlx.local_load(b_smem)
    offsets = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(output_ptr + offsets, result)


@triton.jit
def _async_amd_desc_store_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc_in = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    desc_out = tl.make_tensor_descriptor(y_ptr, [M, N], [N, 1], [M, N])
    # Separate buffers for load vs store — they get different encodings
    # (padded for load, swizzled for store) and can't share a buffer
    # until alignTDMDescriptorEncodings is ported.
    load_buf = tlx.local_alloc((M, N), tl.float16, 1)
    store_buf = tlx.local_alloc((M, N), tl.float16, 1)
    load_view = tlx.local_view(load_buf, 0)
    store_view = tlx.local_view(store_buf, 0)
    tlx.async_amd_descriptor_load(desc_in, load_view, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    data = tlx.local_load(load_view)
    tlx.local_store(store_view, data)
    tlx.async_amd_descriptor_store(desc_out, store_view, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)


@triton.jit
def _update_tensor_descriptor_store_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc_in = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    desc_out = tl.make_tensor_descriptor(y_ptr, [M, N], [N, 1], [M, N])
    load_buf = tlx.local_alloc((M, N), tl.float16, 1)
    store_buf = tlx.local_alloc((M, N), tl.float16, 1)
    load_view = tlx.local_view(load_buf, 0)
    store_view = tlx.local_view(store_buf, 0)

    pred = tl.program_id(0) == 0
    desc_in = tlx.update_tensor_descriptor(desc_in, set_bounds=[M, N], pred=pred)
    offset_m = desc_in.shape[0] - M
    offset_n = (desc_in.strides[1] - 1).to(tl.int32)
    desc_in = tlx.update_tensor_descriptor(desc_in, add_offsets=[offset_m, offset_n])
    tlx.async_amd_descriptor_load(desc_in, load_view)
    tlx.async_amd_descriptor_wait(0)
    tlx.local_store(store_view, tlx.local_load(load_view))

    desc_out = tlx.update_tensor_descriptor(desc_out, add_offsets=[0, 0], clamp_bounds=True)
    tlx.async_amd_descriptor_store(desc_out, store_view)
    tlx.async_amd_descriptor_wait(0)


@triton.jit
def _local_reshape_kernel(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    offsets = tl.arange(0, ROWS * COLS)
    values = tl.load(input_ptr + offsets)

    flat_buffers = tlx.local_alloc((ROWS * COLS, ), tl.float32, 1)
    flat = tlx.local_view(flat_buffers, 0)
    tlx.local_store(flat, values)

    reshaped = tlx.local_reshape(flat, [ROWS, COLS])
    result = tlx.local_load(reshaped)

    offs_m = tl.arange(0, ROWS)
    offs_n = tl.arange(0, COLS)
    output_offsets = offs_m[:, None] * COLS + offs_n[None, :]
    tl.store(output_ptr + output_offsets, result)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("M, N", [(32, 32), (64, 128)])
def test_async_amd_desc_load_correctness_gfx1250(device, M, N):
    """async_amd_descriptor_load produces correct results on gfx1250."""
    x = torch.randn(M, N, dtype=torch.float16, device=device)
    output = torch.empty_like(x)
    _async_amd_desc_load_kernel[(1, )](x, output, M=M, N=N)
    torch.testing.assert_close(x, output)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_async_amd_desc_load_fused_correctness_gfx1250(device):
    rows, cols = 16, 32
    a = torch.randn((rows, cols), device=device, dtype=torch.float16)
    b = torch.randn((rows, cols), device=device, dtype=torch.float16)
    output = torch.empty_like(a)
    _async_amd_desc_load_fused_kernel[(1, )](a, b, output, M=rows, N=cols)
    torch.testing.assert_close(output, a + b)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_update_tensor_descriptor_roundtrip_gfx1250(device):
    x = torch.randn((32, 32), dtype=torch.float16, device=device)
    y = torch.empty_like(x)
    _update_tensor_descriptor_store_kernel[(1, )](x, y, M=32, N=32)
    torch.testing.assert_close(x, y)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("M, N", [(32, 32), (64, 128)])
def test_async_amd_desc_store_correctness_gfx1250(device, M, N):
    """TDM load → store round-trip produces correct results on gfx1250."""
    x = torch.randn(M, N, dtype=torch.float16, device=device)
    y = torch.zeros_like(x)
    _async_amd_desc_store_kernel[(1, )](x, y, M=M, N=N)
    torch.testing.assert_close(x, y)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_local_reshape_correctness_gfx1250(device):
    """End-to-end: local_reshape reinterprets a flat LDS buffer as a 2D tile."""
    rows, cols = 8, 8
    inp = torch.arange(rows * cols, dtype=torch.float32, device=device)
    out = torch.empty((rows, cols), dtype=torch.float32, device=device)
    _local_reshape_kernel[(1, )](inp, out, ROWS=rows, COLS=cols)
    torch.testing.assert_close(out, inp.reshape(rows, cols))


@triton.jit
def tlx_square_non_ws(
    x_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EXPECTED_ARRIVAL_COUNT: tl.constexpr,
):
    """Pairs of arrive/wait across alternating phases, with work interleaved."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    bars = tlx.alloc_barriers(num_barriers=1, arrive_count=EXPECTED_ARRIVAL_COUNT)
    bar = tlx.local_view(bars, 0)

    x = tl.load(x_ptr + offsets, mask=mask)

    p = 0
    tlx.barrier_arrive(bar=bar)
    tlx.barrier_wait(bar=bar, phase=p)

    z = x * x

    p = p ^ 1
    tlx.barrier_arrive(bar=bar)
    tlx.barrier_wait(bar=bar, phase=p)

    tl.store(z_ptr + offsets, z, mask=mask)

    p = p ^ 1
    tlx.barrier_arrive(bar=bar)
    tlx.barrier_wait(bar=bar, phase=0)


def run_tlx_square(func, BLOCK_SIZE, device, expected_arrival_count=1):
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    z = torch.empty_like(x)

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    kernel = func[grid](x, z, n_elements, BLOCK_SIZE, expected_arrival_count)

    torch.testing.assert_close(z, x * x, check_dtype=False)
    return kernel


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_wait_arrive_non_ws_gfx1250(BLOCK_SIZE, device):
    # Four waves per workgroup all arrive on the one barrier.
    kernel = run_tlx_square(tlx_square_non_ws, BLOCK_SIZE, device, expected_arrival_count=4)

    ttgir = kernel.asm["ttgir"]
    assert ((ttgir.count("amdgpu.init_barrier") == 1) and (ttgir.count("amdgpu.read_barrier_phase") == 3)
            and (ttgir.count("amdgpu.arrive_barrier") == 3)), f"TTGIR {ttgir}"


def _mxfp_e8m0_to_float32(scale):
    bits = scale.view(torch.uint8).to(torch.int32) << 23
    return bits.view(torch.float32)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("tdm_fusion", ["none", "partial"])
def test_mxgemm_tdm_split_correctness_gfx1250(device, tdm_fusion):
    torch.manual_seed(0)
    M = N = 128
    K = 1536
    scale_block = 32
    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).view(torch.float8_e4m3fn)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).view(torch.float8_e4m3fn)
    a_scale = MXScaleTensor(size=(M, triton.cdiv(K, scale_block))).random(high=32.0).data
    b_scale = MXScaleTensor(size=(N, triton.cdiv(K, scale_block))).random(high=32.0).data

    a_scale_f32 = _mxfp_e8m0_to_float32(a_scale).repeat_interleave(scale_block, dim=1)[:M, :K]
    b_scale_f32 = _mxfp_e8m0_to_float32(b_scale).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]
    expected = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32)

    actual = _amd_mxfp_matmul(
        a.contiguous().to(device),
        b.T.contiguous().to(device),
        _amd_mxfp_pack_scale(a_scale).to(device),
        _amd_mxfp_pack_scale(b_scale).to(device),
        config={
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 256,
            "SCALE_BLOCK": 32,
            "NUM_BUFFERS": 3,
            "DTYPE_A": "e4m3",
            "DTYPE_B": "e4m3",
            "SCHEDULE": "sliceMNK",
            "TDM_FUSION": tdm_fusion,
            "TDM_SPLIT": True,
            "TRANSPOSE_B": True,
            "num_warps": 4,
            "waves_per_eu": 1,
        },
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=2e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_amd_tdm_gemm_pipelined_correctness_gfx1250(device):
    torch.manual_seed(0)
    a = torch.randn((128, 64), device=device, dtype=torch.float16)
    b = torch.randn((64, 128), device=device, dtype=torch.float16)
    actual = _amd_tdm_matmul(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
def test_amd_tdm_gemm_single_warp_correctness_gfx1250(device, TRANSPOSE_B):
    torch.manual_seed(0)
    M = N = 256
    K = 512
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    b_input = b.T.contiguous() if TRANSPOSE_B else b
    actual = _amd_tdm_single_warp_matmul(a, b_input, TRANSPOSE_B=TRANSPOSE_B)
    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("SEQLEN", [640, 896])
def test_amd_fa_tdm_pipelined_correctness_gfx1250(device, SEQLEN):
    torch.manual_seed(0)
    q = torch.randn((1, 1, SEQLEN, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    actual = _amd_fa_tdm_attention(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v).to(torch.float32)
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
