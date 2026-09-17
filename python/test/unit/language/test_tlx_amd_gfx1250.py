"""TLX AMD tests -- gfx1250."""
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.tutorials import amd_tdm_gemm_pipelined as _gfx1250_gemm
from triton.language.extra.tlx.tutorials import amd_mxfp_gemm_tdm_pipelined as _gfx1250_mxfp
from triton.language.extra.tlx.tutorials import amd_fa_tdm_pipelined as _gfx1250_attention
from triton.language.extra.tlx.tutorials.amd_grouped_gemm_gfx1250 import (
    amd_grouped_gemm_gfx1250_test as _gfx1250_grouped, )
from triton.tools.mxfp import MXScaleTensor
from triton._internal_testing import is_hip_gfx1250


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
def _tdm_fused_positioned_kernel(a, b, output, row_offset, col_offset, pred, CLAMP: tl.constexpr,
                                 SET_BOUNDS: tl.constexpr):
    # Leave backing storage before and after the descriptor's logical extent
    # so unclamped backward/forward updates are valid memory accesses.
    a_desc = tl.make_tensor_descriptor(a + 32 * 256, [128, 128], [256, 1], [64, 64])
    b_desc = tl.make_tensor_descriptor(b + 32 * 256, [128, 128], [256, 1], [64, 64])
    a_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[row_offset, col_offset], pred=pred, clamp_bounds=CLAMP)
    b_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[row_offset, col_offset], clamp_bounds=CLAMP)
    if SET_BOUNDS:
        a_desc = tlx.update_tensor_descriptor(a_desc, set_bounds=[16, 32])
        b_desc = tlx.update_tensor_descriptor(b_desc, set_bounds=[16, 32])
    a_buf = tlx.local_alloc((64, 64), tl.float16, 1)
    b_buf = tlx.local_alloc((64, 64), tl.float16, 1)
    a_view = tlx.local_view(a_buf, 0)
    b_view = tlx.local_view(b_buf, 0)
    # A false predicate must preserve the old LDS contents.
    tlx.local_store(a_view, tl.full((64, 64), -3, tl.float16))
    token = tlx.async_amd_descriptor_load_fused([(a_desc, a_view, 3), (b_desc, b_view, 12)])
    tlx.async_amd_descriptor_wait(tokens=[token])
    offsets = tl.arange(0, 64)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(output + offsets, tlx.local_load(a_view))
    tl.store(output + 64 * 64 + offsets, tlx.local_load(b_view))


@triton.jit
def _async_amd_desc_store_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc_in = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    desc_out = tl.make_tensor_descriptor(y_ptr, [M, N], [N, 1], [M, N])
    # Exercise separate input and output staging allocations.
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


@triton.jit
def _async_amd_desc_warp_token_wait_kernel(src, dst, n):
    desc = tl.make_tensor_descriptor(src, [64, 32], [32, 1], [32, 32])
    buffers = tlx.local_alloc((32, 32), tl.float16, 2)
    for i in tl.range(0, n, num_stages=1):
        with tlx.warp_pipeline_stage("load", priority=1):
            token0 = tlx.async_amd_descriptor_load(desc, tlx.local_view(buffers, 0), [0, 0])
            token1 = tlx.async_amd_descriptor_load(desc, tlx.local_view(buffers, 1), [32, 0])
        tlx.async_amd_descriptor_wait(tokens=[token0])
        with tlx.warp_pipeline_stage("store", priority=0):
            value = tlx.local_load(tlx.local_view(buffers, 0))
            offsets = tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
            tl.store(dst + i * 1024 + offsets, value)
        tlx.async_amd_descriptor_wait(tokens=[token1])


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("iterations", [1, 3])
def test_async_amd_desc_token_wait_warp_pipeline_gfx1250(device, iterations):
    import re

    src = torch.randn((64, 32), dtype=torch.float16, device=device)
    dst = torch.empty((iterations, 32, 32), dtype=torch.float16, device=device)
    compiled = _async_amd_desc_warp_token_wait_kernel[(1, )](src, dst, iterations, num_stages=1)
    # Wait for the first load while allowing the second load to remain in flight.
    assert re.search(r"s_wait_tensorcnt\s+(?:0x)?1\b", compiled.asm["amdgcn"])
    assert re.search(r"s_wait_tensorcnt\s+(?:0x)?0\b", compiled.asm["amdgcn"])
    torch.testing.assert_close(dst, src[:32].expand(iterations, -1, -1))


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_async_amd_desc_load_fused_correctness_gfx1250(device):
    rows, cols = 16, 32
    a = torch.randn((rows, cols), device=device, dtype=torch.float16)
    b = torch.randn((rows, cols), device=device, dtype=torch.float16)
    output = torch.empty_like(a)
    _async_amd_desc_load_fused_kernel[(1, )](a, b, output, M=rows, N=cols)
    torch.testing.assert_close(output, a + b)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("row_offset,col_offset", [(-32, 0), (0, 64), (96, 96)])
@pytest.mark.parametrize("clamp,set_bounds,pred", [(False, False, True), (True, False, True), (False, True, True),
                                                   (False, False, False)])
def test_tdm_fused_positioned_offsets(device, row_offset, col_offset, clamp, set_bounds, pred):
    a = torch.randn((192, 256), device=device, dtype=torch.float16)
    b = torch.randn_like(a)
    output = torch.empty((2, 64, 64), device=device, dtype=torch.float16)
    _tdm_fused_positioned_kernel[(1, )](a, b, output, row_offset, col_offset, pred, clamp, set_bounds)
    for index, source in enumerate((a, b)):
        expected = source[32 + row_offset:96 + row_offset, col_offset:64 + col_offset].clone()
        if set_bounds:
            expected[16:, :] = 0
            expected[:, 32:] = 0
        elif clamp:
            rows = 0 if row_offset < 0 else min(64, max(0, 128 - row_offset))
            cols = min(64, max(0, 128 - col_offset))
            expected[rows:, :] = 0
            expected[:, cols:] = 0
        if index == 0 and not pred:
            expected.fill_(-3)
        torch.testing.assert_close(output[index], expected)


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
    assert ((ttgir.count("amdg.init_barrier") == 1) and (ttgir.count("amdg.read_barrier_phase") == 3)
            and (ttgir.count("amdg.arrive_barrier") == 3)), f"TTGIR {ttgir}"
    assert "s_wait_dscnt" in kernel.asm["amdgcn"]
    assert "s_waitcnt" not in kernel.asm["amdgcn"]


@triton.jit
def _tdm_copy_view_kernel(input_ptr, other_ptr, output_ptr, row, VIEW: tl.constexpr, MODE: tl.constexpr,
                          PADDED: tl.constexpr = True):
    narrow_store: tl.constexpr = MODE == "store" and (VIEW == "transpose" or VIEW == "compatible_transpose"
                                                      or VIEW == "reshape" or VIEW == "inner_slice")
    interval: tl.constexpr = 64 if narrow_store else 128
    order: tl.constexpr = [0, 1] if VIEW == "compatible_transpose" else [1, 0]
    if PADDED:
        layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(interval, 8)], [64, 128], order)
    else:
        layout: tl.constexpr = tlx.swizzled_layout(0, 0, 0, order=order)
    buffers = tlx.local_alloc((64, 128), tl.float16, 1, layout=layout)
    full = tlx.local_view(buffers, 0)
    if MODE == "store":
        offsets = tl.arange(0, 64)[:, None] * 128 + tl.arange(0, 128)[None, :]
        tlx.local_store(full, tl.load(input_ptr + offsets))
    if VIEW == "transpose" or VIEW == "compatible_transpose":
        view = tlx.local_trans(full)
        M: tl.constexpr = 128
        N: tl.constexpr = 64
    elif VIEW == "reshape":
        view = tlx.local_reshape(full, [128, 64])
        M: tl.constexpr = 128
        N: tl.constexpr = 64
    elif VIEW == "slice":
        view = tlx.local_slice(full, [32, 0], [32, 128])
        M: tl.constexpr = 32
        N: tl.constexpr = 128
    elif VIEW == "dynamic_slice":
        view = tlx.local_slice(full, [row, 0], [32, 128])
        M: tl.constexpr = 32
        N: tl.constexpr = 128
    elif VIEW == "inner_slice":
        view = tlx.local_slice(full, [0, 64], [64, 64])
        M: tl.constexpr = 64
        N: tl.constexpr = 64
    else:
        view = full
        M: tl.constexpr = 64
        N: tl.constexpr = 128
    if MODE == "store":
        desc = tl.make_tensor_descriptor(output_ptr, [M, N], [N, 1], [M, N])
        tlx.async_amd_descriptor_store(desc, view, [0, 0])
        tlx.async_amd_descriptor_wait(pendings=0)
    else:
        desc = tl.make_tensor_descriptor(input_ptr, [M, N], [N, 1], [M, N])
        if MODE == "fused":
            other_buffers = tlx.local_alloc((M, N), tl.float16, 1)
            other = tlx.local_view(other_buffers, 0)
            other_desc = tl.make_tensor_descriptor(other_ptr, [M, N], [N, 1], [M, N])
            other_desc = tlx.update_tensor_descriptor(other_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
            desc = tlx.update_tensor_descriptor(desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
            token = tlx.async_amd_descriptor_load_fused([(desc, view, 3), (other_desc, other, 12)])
        else:
            token = tlx.async_amd_descriptor_load(desc, view, [0, 0])
        tlx.async_amd_descriptor_wait(tokens=[token])
        values = tlx.local_load(view)
        tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], values)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize(
    "view, padded",
    [("full", True), ("compatible_transpose", True), ("reshape", True), ("slice", True), ("dynamic_slice", True),
     ("full", False), ("slice", False), ("dynamic_slice", False)],
)
@pytest.mark.parametrize("mode", ["load", "fused", "store"])
def test_tdm_copy_view_correctness_gfx1250(device, view, padded, mode):
    shape = (32, 128) if "slice" in view else ((64, 128) if view == "full" else (128, 64))
    x = torch.randn((64, 128) if mode == "store" else shape, device=device, dtype=torch.float16)
    expected = x
    if mode == "store":
        if "slice" in view:
            expected = x[32:, :]
        elif view == "compatible_transpose":
            expected = x.T
        elif view == "reshape":
            expected = x.reshape(shape)
    output = torch.full(shape, float("nan"), device=device, dtype=torch.float16)
    compiled = _tdm_copy_view_kernel[(1, )](x, x, output, 32, VIEW=view, MODE=mode, PADDED=padded)
    if view == "dynamic_slice":
        assert "ttg.memdesc_dynamic_subslice" in compiled.asm["ttgir"]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@triton.jit
def _tdm_reused_descriptor_kernel(input_ptr, output_ptr, FUSED: tl.constexpr):
    first_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(256, 8)], [64, 128])
    second_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(256, 8)], [32, 128])
    first_buffers = tlx.local_alloc((64, 128), tl.float16, 1, layout=first_layout)
    second_buffers = tlx.local_alloc((32, 128), tl.float16, 1, layout=second_layout)
    first = tlx.local_slice(tlx.local_view(first_buffers, 0), [0, 0], [32, 128])
    second = tlx.local_view(second_buffers, 0)
    desc = tl.make_tensor_descriptor(input_ptr, [32, 128], [128, 1], [32, 128])
    desc = tlx.update_tensor_descriptor(desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    if FUSED:
        tlx.async_amd_descriptor_load_fused([(desc, first, 3), (desc, second, 12)])
    else:
        tlx.async_amd_descriptor_load(desc, first)
        tlx.async_amd_descriptor_load(desc, second)
    tlx.async_amd_descriptor_wait(pendings=0)
    values = tlx.local_load(first) + tlx.local_load(second)
    tl.store(output_ptr + tl.arange(0, 32)[:, None] * 128 + tl.arange(0, 128)[None, :], values)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("fused", [False, True])
def test_tdm_descriptor_reuse_different_allocations_gfx1250(device, fused):
    x = torch.randn((32, 128), device=device, dtype=torch.float16)
    output = torch.empty_like(x)
    _tdm_reused_descriptor_kernel[(1, )](x, output, FUSED=fused)
    torch.testing.assert_close(output, x + x, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("M,N,K", [(128, 128, 64), (256, 256, 128), (512, 512, 256)])
def test_gfx1250_matmul_tdm_pipelined(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device=triton.runtime.driver.active.get_active_torch_device(), dtype=torch.float16)
    b = torch.randn((K, N), device=triton.runtime.driver.active.get_active_torch_device(), dtype=torch.float16)

    triton_out = _gfx1250_gemm.matmul(a, b, config={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32})
    torch_out = torch.matmul(a, b)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
def test_gfx1250_matmul_tdm_pipelined_single_warp_per_simd_schedule(TRANSPOSE_B):
    torch.manual_seed(0)
    M, N, K = 256, 256, 512
    a = torch.randn((M, K), device=triton.runtime.driver.active.get_active_torch_device(), dtype=torch.float16)
    b = torch.randn((K, N), device=triton.runtime.driver.active.get_active_torch_device(), dtype=torch.float16)
    if TRANSPOSE_B:
        b = b.T.contiguous()

    triton_out = _gfx1250_gemm.matmul_tdm_pipelined_single_warp_per_simd_schedule(
        a,
        b,
        TRANSPOSE_B=TRANSPOSE_B,
    )
    b_ref = b.T if TRANSPOSE_B else b
    torch_out = torch.matmul(a.to(torch.float32), b_ref.to(torch.float32)).to(torch.bfloat16)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("SEED", [0, 7])
@pytest.mark.parametrize(
    "M,N,K,SCHEDULE,DTYPE_A,DTYPE_B,BLOCK_M,BLOCK_N,BLOCK_K,NUM_BUFFERS,SCALE_PRESHUFFLE,WITH_A_SCALE,"
    "TDM_FUSION,L2_PREFETCH_DISTANCE,TDM_SPLIT",
    [
        (256, 256, 512, "baseline", "float8_e5m2", "float8_e5m2", 128, 128, 128, 2, True, True, "none", -1, False),
        (256, 256, 512, "baseline", "float8_e4m3", "float8_e5m2", 128, 128, 128, 2, False, False, "none", 2, False),
        (256, 256, 512, "sliceK", "float8_e4m3", "float8_e5m2", 128, 128, 256, 2, True, True, "2way", 2, False),
        (256, 512, 512, "sliceNK", "float8_e5m2", "float4", 256, 256, 256, 2, True, True, "2way", 2, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "none", 2, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "2way", 2, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "4way", 2, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "partial", 2, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float4", 256, 256, 256, 2, True, True, "partial", -1, True),
        (256, 512, 512, "sliceMNK", "float8_e4m3", "float8_e5m2", 128, 256, 256, 2, True, True, "4way", 2, False),
        (256, 256, 512, "baseline", "float4", "float4", 128, 128, 128, 2, True, True, "4way", 2, False),
        (384, 384, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "4way", 2, False),
        (384, 512, 768, "sliceMNK", "float8_e5m2", "float8_e4m3", 256, 256, 256, 2, True, True, "2way", 2, False),
    ],
)
def test_gfx1250_mxgemm_tdm_pipelined(TRANSPOSE_B, SEED, M, N, K, SCHEDULE, DTYPE_A, DTYPE_B, BLOCK_M, BLOCK_N, BLOCK_K,
                                      NUM_BUFFERS, SCALE_PRESHUFFLE, WITH_A_SCALE, TDM_FUSION, L2_PREFETCH_DISTANCE,
                                      TDM_SPLIT):
    torch.manual_seed(SEED)
    a = _gfx1250_mxfp._init_data(DTYPE_A, M, K)
    b = _gfx1250_mxfp._init_data(DTYPE_B, K, N)
    if WITH_A_SCALE:
        a_scale = MXScaleTensor(size=(M, triton.cdiv(K, 32))).random(high=32.0).data
    else:
        a_scale = None
    b_scale = MXScaleTensor(size=(N, triton.cdiv(K, 32))).random(high=32.0).data
    ref = _gfx1250_mxfp.torch_gemm_mxfp(a, b, a_scale, b_scale, 32, M, N, K)

    a_scale_input = _gfx1250_mxfp.pack_scale(a_scale) if SCALE_PRESHUFFLE else a_scale
    b_scale_input = _gfx1250_mxfp.pack_scale(b_scale) if SCALE_PRESHUFFLE else b_scale
    if DTYPE_A == "float4":
        a = a.to_packed_tensor(dim=1)
    if DTYPE_B == "float4":
        b = b.to_packed_tensor(dim=0)

    a_d = a.data.contiguous().cuda() if DTYPE_A == "float4" else a.contiguous().cuda()
    if DTYPE_B == "float4":
        b_d = b.data.T.contiguous().cuda() if TRANSPOSE_B else b.data.contiguous().cuda()
    else:
        b_d = b.T.contiguous().cuda() if TRANSPOSE_B else b.contiguous().cuda()
    if a_scale_input is not None:
        a_scale_d = a_scale_input.cuda()
    else:
        a_scale_d = None
    out = _gfx1250_mxfp.mxgemm_tdm_pipelined(a_d, b_d, a_scale_d, b_scale_input.cuda(), BLOCK_M, BLOCK_N, BLOCK_K,
                                             TRANSPOSE_B, NUM_BUFFERS, _gfx1250_mxfp.DTYPE_TO_TRITON[DTYPE_A],
                                             _gfx1250_mxfp.DTYPE_TO_TRITON[DTYPE_B], SCALE_PRESHUFFLE, WITH_A_SCALE,
                                             SCHEDULE, L2_PREFETCH_DISTANCE, M, N, K, TDM_FUSION, TDM_SPLIT)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=2e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("BATCH,H,SEQLEN", [(1, 8, 1024),  # multi-head
                                            (2, 4, 1024),  # multi-batch + multi-head
                                            (1, 16, 2048),  # many heads, longer seqlen
                                            (1, 2, 896),  # non-128-multiple -> masked remainder path
                                            (1, 1, 640),  # small -> remainder peel path
                                            ])
def test_gfx1250_attn_fwd_tdm_pipelined(BATCH, H, SEQLEN):
    torch.manual_seed(0)
    D = 128
    q = torch.randn((BATCH, H, SEQLEN, D), device=triton.runtime.driver.active.get_active_torch_device(),
                    dtype=torch.bfloat16)
    k = torch.randn((BATCH, H, SEQLEN, D), device=triton.runtime.driver.active.get_active_torch_device(),
                    dtype=torch.bfloat16)
    v = torch.randn((BATCH, H, SEQLEN, D), device=triton.runtime.driver.active.get_active_torch_device(),
                    dtype=torch.bfloat16)
    sm_scale = 1.0 / (D**0.5)
    out = _gfx1250_attention.attn_fwd_tdm_pipelined(q, k, v, sm_scale)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v).to(torch.float32)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-2, rtol=5e-2)


test_gfx1250_grouped_gemm_phase0_ragged = _gfx1250_grouped.test_grouped_gemm_phase0_ragged_gfx1250

test_gfx1250_grouped_gemm_tdm_packed_ragged_m = _gfx1250_grouped.test_grouped_gemm_tdm_packed_ragged_m_gfx1250

test_gfx1250_grouped_gemm_tdm_asymmetric_depth3 = _gfx1250_grouped.test_grouped_gemm_tdm_asymmetric_depth3_gfx1250

test_gfx1250_grouped_gemm_tdm_cross_tile_prefetch = _gfx1250_grouped.test_grouped_gemm_tdm_cross_tile_prefetch_gfx1250

test_gfx1250_grouped_gemm_tdm_xcd_remap = _gfx1250_grouped.test_grouped_gemm_tdm_xcd_remap_gfx1250
