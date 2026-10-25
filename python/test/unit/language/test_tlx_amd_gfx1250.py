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
    MULTICAST_MASKS: tl.constexpr = None,
    SHARED_LAYOUT: tl.constexpr = None,
    MASK_SHIFT_BITS: tl.constexpr = 0,
):
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, N], [N, 1], [M, N])
    b_desc = tl.make_tensor_descriptor(b_ptr, [M, N], [N, 1], [M, N])
    if SHARED_LAYOUT is None:
        a_buf = tlx.local_alloc((M, N), tl.float16, 1)
        b_buf = tlx.local_alloc((M, N), tl.float16, 1)
    else:
        a_buf = tlx.local_alloc((M, N), tl.float16, 1, layout=SHARED_LAYOUT)
        b_buf = tlx.local_alloc((M, N), tl.float16, 1, layout=SHARED_LAYOUT)
    a_smem = tlx.local_view(a_buf, 0)
    b_smem = tlx.local_view(b_buf, 0)
    # A missing member load must not pass by reading a previous launch's LDS.
    tlx.local_store(a_smem, tl.full((M, N), float("nan"), tl.float16))
    tlx.local_store(b_smem, tl.full((M, N), float("nan"), tl.float16))
    tlx.workgroup_barrier()
    a_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    b_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    if MASK_SHIFT_BITS and MULTICAST_MASKS is not None:
        shift = tlx.cluster_cta_rank() & MASK_SHIFT_BITS
        token = tlx.async_amd_descriptor_load_fused([(a_desc, a_smem, 0b0011), (b_desc, b_smem, 0b1100)],
                                                    multicast_masks=(MULTICAST_MASKS[0] << shift,
                                                                     MULTICAST_MASKS[1] << shift))
    else:
        token = tlx.async_amd_descriptor_load_fused([(a_desc, a_smem, 0b0011), (b_desc, b_smem, 0b1100)],
                                                    multicast_masks=MULTICAST_MASKS)
    tlx.async_amd_descriptor_wait(tokens=[token])
    # TDM completion is per wave. Consumers only read their own CTA's LDS,
    # so the handoff requires workgroup synchronization, not a cluster barrier.
    tlx.workgroup_barrier()
    a = tlx.local_load(a_smem)
    b = tlx.local_load(b_smem)
    offsets = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    # Check each member separately: A+B would also pass if A and B were swapped.
    tl.store(output_ptr + offsets, a)
    tl.store(output_ptr + M * N + offsets, b)


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
    output = torch.full((2, rows, cols), float("nan"), device=device, dtype=torch.float16)
    _async_amd_desc_load_fused_kernel[(1, )](a, b, output, M=rows, N=cols)
    torch.testing.assert_close(output, torch.stack((a, b)), rtol=0, atol=0)


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


@triton.jit
def _amd_wmma_queuing_kernel():
    pass


@pytest.mark.parametrize("enabled", [None, False, True])
def test_amd_wmma_queuing_compiles(enabled):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = ASTSource(_amd_wmma_queuing_kernel, signature={})
    options = {} if enabled is None else dict(enable_wmma_queuing=enabled)
    compiled = triton.compile(source, target=GPUTarget("hip", "gfx1250", 32), options=options)
    assert compiled.metadata.enable_wmma_queuing == bool(enabled)
    assert "llvm.amdgcn.s.setreg" not in compiled.asm["ttir"]
    # A result-free intrinsic must survive optimization even in an empty kernel.
    assert compiled.asm["llir"].count("call void @llvm.amdgcn.s.setreg(i32 154, i32 1)") == int(bool(enabled))
    assert "asm sideeffect" not in compiled.asm["llir"]
    instruction = "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1"
    assert compiled.asm["amdgcn"].count(instruction) == int(bool(enabled))
    assert "s_mov_b32" not in compiled.asm["amdgcn"]


def test_amd_wmma_queuing_cache_key():
    from triton.backends.amd.compiler import HIPBackend
    from triton.backends.compiler import GPUTarget

    backend = HIPBackend(GPUTarget("hip", "gfx1250", 32))
    default = backend.parse_options({})
    disabled = backend.parse_options(dict(enable_wmma_queuing=False))
    enabled = backend.parse_options(dict(enable_wmma_queuing=True))
    assert default.hash() == disabled.hash()
    assert enabled.hash() != disabled.hash()


@pytest.mark.parametrize("value", [0, 1, -1, 1.0, "true"])
def test_amd_wmma_queuing_invalid_option(value):
    from triton.backends.amd.compiler import HIPOptions

    with pytest.raises(ValueError, match="enable_wmma_queuing must be a bool"):
        HIPOptions(arch="gfx1250", enable_wmma_queuing=value)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1200", "gfx1201"])
def test_amd_wmma_queuing_unsupported_target(arch):
    from triton.backends.amd.compiler import HIPOptions

    assert not HIPOptions(arch=arch).enable_wmma_queuing
    with pytest.raises(ValueError, match="requires an AMD gfx1250 target"):
        HIPOptions(arch=arch, enable_wmma_queuing=True)


@pytest.mark.parametrize("independent_ctas", [False, True])
@pytest.mark.parametrize("cluster_size,masks,message", [
    (4, (-1, 3), "outside the cluster"),
    (4, (16, 3), "outside the cluster"),
    (4, (3, ), "one multicast mask per member"),
    (4, (1.5, 3), "scalar integers"),
    (8, (63, 3), "at most 5 recipients"),
])
def test_tdm_fused_multicast_invalid_masks(cluster_size, masks, message, independent_ctas):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.errors import CompilationError
    source = ASTSource(_async_amd_desc_load_fused_kernel,
                       signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "output_ptr":
                                  "*fp16"}, constexprs={"M": 64, "N": 64, "MULTICAST_MASKS": masks})
    options = dict(ctas_per_cga=(cluster_size, 1, 1)) if independent_ctas else dict(num_ctas=cluster_size)
    with pytest.raises(CompilationError, match=message):
        triton.compile(source, target=GPUTarget("hip", "gfx1250", 32), options=dict(num_warps=4, **options))


def _tdm_fused_distributed_constants(num_ctas, split_m, mask_mode):
    replicas = num_ctas // split_m
    shared = tlx.swizzled_shared_layout_encoding(
        vectorSize=1,
        perPhase=1,
        maxPhase=1,
        order=[1, 0],
        numCTAs=num_ctas,
        numCTAsPerCGA=[split_m, replicas],
        numCTASplit=[split_m, 1],
        numCTAOrder=[1, 0],
    )
    # Adjacent CTAs share a row partition, matching the default register layout
    # without requiring cross-CTA shared loads. Explicit masks disable one
    # member's multicast despite its replicated layout; test both member orders.
    mask = (1 << replicas) - 1
    masks = {"inferred": None, "explicit": (mask, 0), "explicit_b": (0, mask), "disabled": (0, 0)}[mask_mode]
    return dict(M=64, N=64, SHARED_LAYOUT=shared, MULTICAST_MASKS=masks, MASK_SHIFT_BITS=num_ctas - replicas)


def _check_tdm_fused_multicast_codegen(llvm_ir, num_ctas, split_m, mask_mode):
    from triton._filecheck import run_filecheck

    # Numerical correctness cannot distinguish multicast from separate loads.
    # Follow descriptor group 1, whose low 16 bits select recipients, all the
    # way to the TDM intrinsic. Also check which producer waves select each mask.
    enabled = (mask_mode in ("inferred", "explicit"), mask_mode in ("inferred", "explicit_b"))
    checks = ["CHECK: [[WAVE:%[^ ]+]] = {{.*}}call i32 @llvm.amdgcn.wave.id()"]
    dynamic_mask = split_m > 1 and any(enabled)
    if dynamic_mask:
        # With four CTAs and two row partitions, recipient groups are {0,1}
        # and {2,3}. Layout inference clears the replica bit; explicit masks
        # select the partition bit. Both must reach the descriptor's mask.
        partition_bits = -2 if mask_mode == "inferred" else 2
        checks += [
            "CHECK: [[CTA:%[^ ]+]] = {{.*}}call i32 @llvm.amdgcn.cluster.workgroup.id.x()",
            f"CHECK: [[SHIFT:%[^ ]+]] = and i32 [[CTA]], {partition_bits}",
            "CHECK: [[MASK:%[^ ]+]] = shl {{.*}}i32 3, [[SHIFT]]",
        ]
        mask_value = "[[MASK]]"
        if mask_mode == "inferred":
            checks += ["CHECK: [[CLEAN_MASK:%[^ ]+]] = and i32 [[MASK]], -327681"]
            mask_value = "[[CLEAN_MASK]]"
        checks += [f"CHECK-DAG: [[WORD:%[^ ]+]] = or {{{{.*}}}}i32 {mask_value}, 65536"]

    for multicast in dict.fromkeys(enabled):
        group = "MULTICAST" if multicast else "UNICAST"
        if multicast and dynamic_mask:
            checks += [
                "CHECK-DAG: [[PARTIAL:%[^ ]+]] = insertelement <8 x i32> {{.*}}, i32 [[WORD]], i64 0",
                f"CHECK-DAG: [[{group}:%[^ ]+]] = insertelement <8 x i32> [[PARTIAL]], i32 {{{{.*}}}}, i64 2",
            ]
        else:
            # data_size=1 (f16) occupies bit 16; all other control bits are zero.
            word = 65536 | ((1 << num_ctas) - 1 if multicast else 0)
            checks += [
                f"CHECK-DAG: [[{group}:%[^ ]+]] = insertelement <8 x i32> "
                f"<i32 {word}, {{{{.*}}}}>, i32 {{{{.*}}}}, i64 2"
            ]

    groups = ["[[MULTICAST]]" if multicast else "[[UNICAST]]" for multicast in enabled]
    if enabled[0] != enabled[1]:
        checks += [
            "CHECK: [[IS_B:%[^ ]+]] = icmp ugt i32 [[WAVE]], 1",
            f"CHECK: [[GROUP1:%[^ ]+]] = select i1 [[IS_B]], <8 x i32> {groups[1]}, <8 x i32> {groups[0]}",
        ]
        group1 = "[[GROUP1]]"
    else:
        group1 = groups[0]
    checks += [
        "CHECK: call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> {{[^,]+}}, <8 x i32> " + group1 + ",",
        "CHECK-NOT: = load {{.*}}ptr addrspace(3)",
        "CHECK: call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)",
        "CHECK-NOT: = load {{.*}}ptr addrspace(3)",
        "CHECK: call void @llvm.amdgcn.s.barrier()",
        "CHECK: = load {{.*}}ptr addrspace(3)",
    ]
    run_filecheck("tdm_fused_multicast", llvm_ir, "\n".join(checks))


@pytest.mark.parametrize("num_ctas,split_m", [(2, 1), (4, 1), (4, 2)])
@pytest.mark.parametrize("mask_mode", ["inferred", "explicit", "explicit_b", "disabled"])
def test_tdm_fused_multicast_distributed_compiles(num_ctas, split_m, mask_mode):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = ASTSource(_async_amd_desc_load_fused_kernel,
                       signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "output_ptr":
                                  "*fp16"}, constexprs=_tdm_fused_distributed_constants(num_ctas, split_m, mask_mode))
    compiled = triton.compile(source, target=GPUTarget("hip", "gfx1250", 32),
                              options=dict(num_warps=4, num_ctas=num_ctas))
    assert compiled.metadata.num_ctas == num_ctas
    assert (" multicast " in compiled.asm["ttgir"]) == (mask_mode != "inferred")
    assert compiled.asm["amdgcn"].count("tensor_load_to_lds") == 1
    if split_m > 1:
        assert "CGALayout = [[0, 0], [1, 0]]" in compiled.asm["ttgir"]
    assert "amdg.cluster_barrier" not in compiled.asm["ttgir"]
    assert "s_barrier_signal -3" not in compiled.asm["amdgcn"]
    assert "s_barrier_wait -3" not in compiled.asm["amdgcn"]
    _check_tdm_fused_multicast_codegen(compiled.asm["llir"], num_ctas, split_m, mask_mode)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("num_ctas,split_m", [(2, 1), (4, 1), (4, 2)])
@pytest.mark.parametrize("mask_mode", ["inferred", "explicit", "explicit_b", "disabled"])
def test_tdm_fused_multicast_distributed_correctness(device, num_ctas, split_m, mask_mode):
    # Distinct members and row partitions expose swapped descriptors, incorrect
    # CTA offsets, and missing stores from any CTA. Build the reference on CPU.
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randint(1, 1000, (2, 64, 64), generator=generator).to(torch.float16)
    inputs[1].neg_()
    a, b = inputs[0].to(device), inputs[1].to(device)
    output = torch.full((2, 64, 64), float("nan"), dtype=torch.float16, device=device)
    constants = _tdm_fused_distributed_constants(num_ctas, split_m, mask_mode)
    _async_amd_desc_load_fused_kernel[(1, )](a, b, output, **constants, num_ctas=num_ctas)
    torch.testing.assert_close(output.cpu(), inputs, rtol=0, atol=0)


@pytest.mark.parametrize("cluster_size,multicast", [(1, False), (2, True), (4, True), (4, False)])
@pytest.mark.parametrize("group_size", [1, 2])
@pytest.mark.parametrize("interface", ["packed", "list"])
@pytest.mark.parametrize("enable_wmma_queuing", [False, True])
def test_grouped_gemm_multicast_compiles(cluster_size, multicast, group_size, interface, enable_wmma_queuing):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton import knobs

    constants = dict(NUM_PROGRAMS=16, BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=4, NUM_BUFFERS=2,
                     L2_PREFETCH_DISTANCE=0, C_STAGING_MODE=0, CROSS_TILE_PREFETCH=True, XCD_REMAP_MODE=2, NUM_XCDS=8,
                     XCD_CHUNK=2, K=512, group_size=group_size, CLUSTER_SIZE=cluster_size, CLUSTER_MULTICAST=multicast)
    if interface == "packed":
        kernel = _gfx1250_grouped.grouped_gemm_tdm_kernel
        signature = _gfx1250_grouped._grouped_gemm_tdm_compile_signature()
        attrs = _gfx1250_grouped._grouped_gemm_tdm_compile_attrs()
    else:
        kernel = _gfx1250_grouped.grouped_gemm_tdm_list_kernel
        constants["ALLOW_MN_TAILS"] = False
        signature = _gfx1250_grouped._grouped_gemm_tdm_list_compile_signature()
        attrs = _gfx1250_grouped._grouped_gemm_tdm_list_compile_attrs()
    source = ASTSource(kernel, signature={k: v
                                          for k, v in signature.items()
                                          if k not in constants}, constexprs=constants, attrs=attrs)
    assert knobs.runtime.add_stages_inspection_hook is None
    compiled = triton.compile(
        source, target=GPUTarget("hip", "gfx1250", 32),
        options=dict(num_warps=4, waves_per_eu=1, ctas_per_cga=(cluster_size, 1, 1),
                     enable_wmma_queuing=enable_wmma_queuing))
    assert compiled.metadata.num_ctas == 1
    assert compiled.metadata.ctas_per_cga == (cluster_size, 1, 1)
    assert "llvm.amdgcn.s.setreg" not in compiled.asm["ttir"]
    assert compiled.asm["llir"].count("call void @llvm.amdgcn.s.setreg(i32 154, i32 1)") == int(enable_wmma_queuing)
    instruction = "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1"
    assert compiled.asm["amdgcn"].count(instruction) == int(enable_wmma_queuing)
    if enable_wmma_queuing:
        assert compiled.asm["amdgcn"].index(instruction) < compiled.asm["amdgcn"].index("v_wmma_")
    assert f'"amdgpu-cluster-dims"="{cluster_size},1,1"' in compiled.asm["llir"]
    assert compiled.metadata.global_scratch_size == 0
    assert compiled.metadata.shared <= 320 * 1024
    assert ("amdg.cluster_barrier_arrive" in compiled.asm["ttgir"]) == (cluster_size > 1)
    assert ("amdg.cluster_barrier_wait" in compiled.asm["ttgir"]) == (cluster_size > 1)
    assert (" multicast " in compiled.asm["ttgir"]) == (multicast and cluster_size > 1)


@pytest.mark.parametrize("dims,num_ctas,arch", [
    ((3, 1, 1), 1, "gfx1250"),
    ((4, 2, 1), 1, "gfx1250"),
    ((4, 1), 1, "gfx1250"),
    ((4, 1, 1), 2, "gfx1250"),
    ((4, 1, 1), 1, "gfx950"),
])
def test_amd_independent_cluster_invalid_options(dims, num_ctas, arch):
    from triton.backends.amd.compiler import HIPOptions
    with pytest.raises(ValueError, match="ctas_per_cga"):
        HIPOptions(arch=arch, num_ctas=num_ctas, ctas_per_cga=dims)


@pytest.mark.parametrize("m_list,n,num_programs,valid", [
    ([4096, 4096], 4096, 32, True),
    ([4096, 4096], 4096, 256, True),
    ([4096, 2048], 4096, 32, False),
    ([0, 4096], 4096, 32, False),
    ([4096, 4096], 768, 32, False),
    ([4096, 4096], 4096, 24, False),
    ([4096, 4096], 4096, 48, False),
])
def test_grouped_gemm_multicast_config_validation(m_list, n, num_programs, valid):

    def validate():
        _gfx1250_grouped._validate_grouped_gemm_cluster_config(m_list, n, num_programs, block_m=256, block_n=256,
                                                               block_k=128, group_m=4, tdm_pipeline_depth=2,
                                                               l2_prefetch_distance=0, c_staging_mode=0,
                                                               cross_tile_prefetch=True, auto_config=False,
                                                               xcd_remap_mode="chunked", num_xcds=8, xcd_chunk=2)

    if valid:
        validate()
    else:
        with pytest.raises(ValueError):
            validate()


@triton.jit
def _amd_independent_cluster_ids(output):
    pid = tl.program_id(0)
    rank = tlx.cluster_cta_rank()
    tlx.cluster_barrier()
    tl.store(output + pid, pid * 16 + rank)


@pytest.mark.parametrize("arch,cluster_size", [
    ("gfx1250", None),
    ("gfx1250", 1),
    ("gfx1250", 2),
    ("gfx1250", 4),
    ("gfx950", None),
    ("gfx950", 1),
])
def test_amd_cluster_barrier_compiles(arch, cluster_size):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    options = dict(num_warps=4)
    if cluster_size is not None:
        options["ctas_per_cga"] = (cluster_size, 1, 1)
    source = ASTSource(_amd_independent_cluster_ids, signature={"output": "*i32"})
    compiled = triton.compile(source, target=GPUTarget("hip", arch, 32 if arch == "gfx1250" else 64), options=options)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.barrier local" in ttgir
    assert "amdg.cluster_barrier_arrive" in ttgir
    assert "amdg.cluster_barrier_wait" in ttgir
    assert "s_barrier" in compiled.asm["amdgcn"]
    multi_cta = cluster_size is not None and cluster_size > 1
    assert ("s_barrier_signal -3" in compiled.asm["amdgcn"]) == multi_cta
    assert ("s_barrier_wait -3" in compiled.asm["amdgcn"]) == multi_cta


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
def test_amd_independent_cluster_launch_grid_and_cache():
    output = torch.empty((8, ), device="cuda", dtype=torch.int32)
    pid = torch.arange(8, dtype=torch.int32)
    # Switching cluster shapes with one JIT signature must retain physical
    # program IDs and must not reuse a cached launch with the wrong cluster.
    for size in (1, 4, 2, 1):
        _amd_independent_cluster_ids[(8, )](output, ctas_per_cga=(size, 1, 1))
        torch.testing.assert_close(output.cpu(), pid * 16 + pid % size)
    with pytest.raises(ValueError, match="grid X must be divisible"):
        _amd_independent_cluster_ids[(3, )](output, ctas_per_cga=(4, 1, 1))


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("cluster_size,multicast", [(1, False), (2, True), (4, True), (4, False)])
@pytest.mark.parametrize("group_size", [1, 2])
@pytest.mark.parametrize("interface", ["packed", "list"])
def test_grouped_gemm_multicast_correctness(cluster_size, multicast, group_size, interface):
    # Distinct CPU inputs detect sharing the wrong tile; two persistent tiles
    # per program exercise ring reuse, and G=2 also exercises a group boundary.
    torch.manual_seed(123)
    m, n, k = 2048, 1024, 512
    a = torch.randn((group_size * m, k), dtype=torch.float16)
    b = torch.randn((group_size, n, k), dtype=torch.float16)
    offsets = torch.arange(group_size + 1, dtype=torch.int32) * m
    config = dict(block_m=256, block_n=256, num_programs=16, cross_tile_prefetch=True, xcd_remap_mode="chunked",
                  cluster_size=cluster_size, cluster_multicast=multicast)
    if interface == "packed":
        out = _gfx1250_grouped.grouped_gemm_tdm(a.cuda(), b.cuda(), offsets.cuda(), **config).cpu()
        outputs = list(out.split(m))
    else:
        # Separate allocations prevent an accidental dependence on packed
        # group strides from passing this check.
        group_a = [value.cuda() for value in a.split(m)]
        group_b = [value.cuda().T for value in b.unbind()]
        outputs = _gfx1250_grouped.grouped_gemm_tdm_list(group_a, group_b, **config)
    for group in range(group_size):
        expected = (a[group * m:(group + 1) * m].float() @ b[group].float().T).half()
        torch.testing.assert_close(outputs[group].cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("block_m,block_n,depth,staging,cross,k", [
    (128, 128, 2, 0, False, 256),
    (128, 128, 3, 0, False, 384),
    (128, 128, 4, 0, False, 512),
    (128, 256, 2, 1, True, 512),
    (256, 128, 2, 1, True, 512),
    (256, 256, 2, 0, False, 384),
    (256, 256, 2, 0, True, 512),
    (256, 256, 2, 1, True, 512),
])
def test_grouped_gemm_list_ragged_m(block_m, block_n, depth, staging, cross, k):
    torch.manual_seed(321)
    # Alternating sizes exercise successive group transitions and within-group
    # persistent reuse. B is shared by two groups, which pointer tables allow.
    m_list = [block_m, 3 * block_m, 2 * block_m]
    n = 2 * block_n
    cpu_a = [torch.randn((m, k), dtype=torch.float16) for m in m_list]
    cpu_b = [torch.randn((n, k), dtype=torch.float16) for _ in m_list]
    cpu_b[2] = cpu_b[0]
    group_a = []
    group_b = []
    group_c = []
    for index, (a, b, m) in enumerate(zip(cpu_a, cpu_b, m_list)):
        # Aligned offset views expose assumptions about allocation bases.
        shift = 8 * (index + 1)
        a_storage = torch.empty(a.numel() + shift, device="cuda", dtype=torch.float16)
        a_view = a_storage[shift:].view_as(a)
        a_view.copy_(a)
        group_a.append(a_view)
        group_b.append(b.cuda().T)
        c_storage = torch.empty(m * n + shift, device="cuda", dtype=torch.float16)
        group_c.append(c_storage[shift:].view(m, n))
    group_b[2] = group_b[0]
    config = dict(block_m=block_m, block_n=block_n, tdm_pipeline_depth=depth, c_staging_mode=staging,
                  cross_tile_prefetch=cross, num_programs=2)
    actual = _gfx1250_grouped.grouped_gemm_tdm_list(tuple(group_a), tuple(group_b), group_c, **config)
    offsets = [0]
    for m in m_list:
        offsets.append(offsets[-1] + m)
    packed = _gfx1250_grouped.grouped_gemm_tdm(
        torch.cat(cpu_a).cuda(),
        torch.stack(cpu_b).cuda(), torch.tensor(offsets, dtype=torch.int32, device="cuda"), **config).cpu()
    for index, (out, a, b) in enumerate(zip(actual, cpu_a, cpu_b)):
        assert out is group_c[index]
        expected = (a.float() @ b.float().T).half()
        torch.testing.assert_close(out.cpu(), expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out.cpu(), packed[offsets[index]:offsets[index + 1]], atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("staging,cross", [(0, False), (0, True), (1, True)])
def test_grouped_gemm_list_gfx950_metadata(staging, cross):
    from triton.tlx.ops.kernels.grouped_gemm import gfx950

    torch.manual_seed(456)
    n, k = 256, 512
    cpu_a = [torch.randn((m, k), dtype=torch.float16) for m in (256, 512, 256)]
    cpu_b = [torch.randn((n, k), dtype=torch.float16) for _ in cpu_a]
    group_a = [a.cuda() for a in cpu_a]
    group_b = [b.cuda().T for b in cpu_b]
    # Reuse the actual gfx950 builder's five device tables and outputs, with
    # no gfx1250 metadata builder, prefix sum, repacking, or host N argument.
    a_ptrs, b_ptrs, c_ptrs, sizes, lds, groups, _, outputs = gfx950._make_grouped_gemm_args(group_a, group_b)
    _gfx1250_grouped.grouped_gemm_tdm_list_kernel[(2, )](a_ptrs, b_ptrs, c_ptrs, sizes, lds, groups, K=k,
                                                         NUM_PROGRAMS=2, BLOCK_M=256, BLOCK_N=256, BLOCK_K=128,
                                                         GROUP_M=4, NUM_BUFFERS=2, L2_PREFETCH_DISTANCE=0,
                                                         C_STAGING_MODE=staging, CROSS_TILE_PREFETCH=cross,
                                                         XCD_REMAP_MODE=0, NUM_XCDS=8, XCD_CHUNK=2, num_warps=4,
                                                         waves_per_eu=1, enable_wmma_queuing=True)
    for output, a, b in zip(outputs, cpu_a, cpu_b):
        torch.testing.assert_close(output.cpu(), (a.float() @ b.float().T).half(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("n", [1, 129, 256, 257])
@pytest.mark.parametrize("block_m,block_n,depth,staging,cross,k,prefetch", [
    (256, 256, 2, 0, False, 256, 0),
    (128, 128, 3, 0, False, 384, 0),
    (128, 128, 4, 0, False, 512, 0),
    (256, 256, 2, 0, True, 512, 0),
    (256, 256, 2, 1, True, 512, 0),
    (256, 256, 2, 1, True, 512, 2),
    (128, 128, 2, 0, False, 256, 0),
    (128, 256, 2, 1, True, 512, 0),
    (128, 256, 3, 0, False, 384, 0),
])
def test_grouped_gemm_list_unaligned_shapes(n, block_m, block_n, depth, staging, cross, k, prefetch):
    torch.manual_seed(654)
    # Include sub-tile groups, both sides of a tile boundary, persistent reuse,
    # and a gap longer than the cross-group lookahead window.
    ms = [0, 1, block_m - 1, 0, 0, 0, 0, block_m + 1, 2 * block_m + 1, 0]
    cpu_a = [torch.randn((m, k), dtype=torch.float16) for m in ms]
    cpu_b = [torch.randn((n, k), dtype=torch.float16) for _ in ms]
    group_a, group_b, outputs, storage = [], [], [], []
    for m, a, b in zip(ms, cpu_a, cpu_b):
        # Keep poison beyond each logical input extent while exercising
        # positioned descriptors at the final M/N tiles.
        a_backing = torch.full((m + block_m, k), float("nan"), dtype=torch.float16, device="cuda")
        b_backing = torch.full((n + block_n, k), float("nan"), dtype=torch.float16, device="cuda")
        a_backing[:m].copy_(a)
        b_backing[:n].copy_(b)
        group_a.append(a_backing[:m])
        group_b.append(b_backing[:n].T)
        c_backing = torch.full((8 + m * n + block_m * (n + block_n), ), -123, dtype=torch.float16, device="cuda")
        outputs.append(c_backing[8:8 + m * n].view(m, n))
        storage.append(c_backing)
    actual = _gfx1250_grouped.grouped_gemm_tdm_list(group_a, group_b, outputs, block_m=block_m, block_n=block_n,
                                                    tdm_pipeline_depth=depth, c_staging_mode=staging,
                                                    cross_tile_prefetch=cross, l2_prefetch_distance=prefetch,
                                                    num_programs=2)
    for m, a, b, out, backing in zip(ms, cpu_a, cpu_b, actual, storage):
        torch.testing.assert_close(out.cpu(), (a.float() @ b.float().T).half(), atol=1e-2, rtol=1e-2)
        guard = backing.cpu()
        assert (guard[:8] == -123).all()
        assert (guard[8 + m * n:] == -123).all()


@pytest.mark.parametrize("staging,cross,tdm_stores", [(0, False, 1), (0, True, 8), (1, True, 0)])
def test_grouped_gemm_list_tails_preserve_output_config(staging, cross, tdm_stores):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    constants = dict(K=512, group_size=3, NUM_PROGRAMS=2, BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=4,
                     NUM_BUFFERS=2, L2_PREFETCH_DISTANCE=0, C_STAGING_MODE=staging, CROSS_TILE_PREFETCH=cross,
                     XCD_REMAP_MODE=0, NUM_XCDS=8, XCD_CHUNK=2, CLUSTER_SIZE=1, CLUSTER_MULTICAST=True,
                     ALLOW_MN_TAILS=True)
    signature = {
        k: v
        for k, v in _gfx1250_grouped._grouped_gemm_tdm_list_compile_signature().items()
        if k not in constants
    }
    source = ASTSource(_gfx1250_grouped.grouped_gemm_tdm_list_kernel, signature=signature, constexprs=constants,
                       attrs=_gfx1250_grouped._grouped_gemm_tdm_list_compile_attrs())
    compiled = triton.compile(source, target=GPUTarget("hip", "gfx1250", 32),
                              options=dict(num_warps=4, waves_per_eu=1, enable_wmma_queuing=True))
    asm = compiled.asm["amdgcn"]
    assert asm.count("tensor_store_from_lds") == tdm_stores
    assert ("global_store" in asm) == (tdm_stores == 0)
    assert compiled.metadata.global_scratch_size == 0


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("staging,cross", [(0, False), (0, True), (1, True)])
@pytest.mark.parametrize("tails", [False, True])
def test_grouped_gemm_list_device_metadata_updates(staging, cross, tails):
    torch.manual_seed(789)
    n, k, capacity = 257 if tails else 256, 512, 512
    # Force lookahead beyond its four-group window, including leading,
    # interior and trailing empty groups. M changes without replacing any
    # device table. Different strides also catch using the current group's
    # strides for a prefetched group's descriptors.
    shapes = [(0, 256, 0, 0, 0, 0, 512, 256, 0), (256, 0, 0, 0, 0, 256, 0, 0, 512)]
    if tails:
        shapes = [(0, 1, 0, 0, 0, 0, 257, 255, 0), (255, 0, 0, 0, 0, 1, 0, 0, 511)]
    groups = len(shapes[0])
    cpu_a, cpu_b, group_a, group_b, outputs = [], [], [], [], []
    strides = []
    for g in range(groups):
        lda, ldb, ldc = k + 8 * g, k + 16 * g, n + 8 * g
        a = torch.randn((capacity, lda), dtype=torch.float16)
        b = torch.randn((n, ldb), dtype=torch.float16)
        cpu_a.append(a[:, :k])
        cpu_b.append(b[:, :k])
        group_a.append(a.cuda())
        group_b.append(b.cuda())
        outputs.append(torch.empty((capacity, ldc), dtype=torch.float16, device="cuda"))
        strides.append((lda, ldb, ldc))
    ptrs = [
        torch.tensor([x.data_ptr()
                      for x in tensors], dtype=torch.int64, device="cuda")
        for tensors in (group_a, group_b, outputs)
    ]
    sizes = torch.tensor([[m, n, k] for m in shapes[0]], dtype=torch.int32, device="cuda")
    lds = torch.tensor(strides, dtype=torch.int32, device="cuda")

    def launch():
        return _gfx1250_grouped.grouped_gemm_tdm_list_kernel[(2, )](
            *ptrs, sizes, lds, groups, K=k, NUM_PROGRAMS=2, BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=4,
            NUM_BUFFERS=2, L2_PREFETCH_DISTANCE=0, C_STAGING_MODE=staging, CROSS_TILE_PREFETCH=cross, XCD_REMAP_MODE=0,
            NUM_XCDS=8, XCD_CHUNK=2, num_warps=4, waves_per_eu=1, enable_wmma_queuing=True)

    launch()  # Compile and initialize the launcher before checking allocations.
    torch.cuda.synchronize()
    for ms in shapes:
        sizes.copy_(torch.tensor([[m, n, k] for m in ms], dtype=torch.int32))
        for output in outputs:
            output.fill_(float("nan"))
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        launch()
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() == allocated
        for m, output, a, b in zip(ms, outputs, cpu_a, cpu_b):
            actual = output.cpu()
            expected = (a[:m].float() @ b.float().T).half()
            torch.testing.assert_close(actual[:m, :n], expected, atol=1e-2, rtol=1e-2)
            assert torch.isnan(actual[m:]).all()
            assert torch.isnan(actual[:m, n:]).all()


@pytest.mark.parametrize("dimension", ["N", "K"])
def test_grouped_gemm_list_rejects_ragged_nk(dimension):
    shapes = [(128, 256, 256), (256, 384 if dimension == "N" else 256, 384 if dimension == "K" else 256)]
    group_a = [torch.empty((m, k), dtype=torch.float16) for m, n, k in shapes]
    group_b = [torch.empty((n, k), dtype=torch.float16).T for m, n, k in shapes]
    with pytest.raises(ValueError, match="same N and K"):
        _gfx1250_grouped.grouped_gemm_tdm_list(group_a, group_b)


@pytest.mark.parametrize("config,k,match", [
    ({"TILE_MODE": 0}, 512, "unsupported.*config keys"),
    ({"HAS_K_TAIL": True}, 512, "unsupported.*config keys"),
    ({"BLOCK_SIZE_M": 64}, 512, "BLOCK_SIZE_M"),
    ({"BLOCK_SIZE_N": 192}, 512, "BLOCK_SIZE_N"),
    ({"BLOCK_SIZE_K": 64}, 512, "BLOCK_SIZE_K"),
    ({"num_warps": 8}, 512, "num_warps"),
    ({"waves_per_eu": 2}, 512, "waves_per_eu"),
    ({"CLUSTER_SIZE": 3}, 512, "CLUSTER_SIZE"),
    ({"NUM_BUFFERS": 1}, 512, "NUM_BUFFERS"),
    ({"NUM_BUFFERS": 3}, 512, "cross-tile prefetch requires depth 2"),
    ({"NUM_BUFFERS": 3, "C_STAGING_MODE": 1, "CROSS_TILE_PREFETCH": False}, 512, "full C buffer"),
    ({"BLOCK_SIZE_M": 128}, 512, "alias-C cross-tile"),
    ({"C_STAGING_MODE": 2}, 512, "C_STAGING_MODE"),
    ({"XCD_REMAP_MODE": 3}, 512, "XCD_REMAP_MODE"),
    ({"GROUP_SIZE_M": 0}, 512, "GROUP_SIZE_M"),
    ({"NUM_XCDS": 0}, 512, "NUM_XCDS"),
    ({"XCD_CHUNK": 0}, 512, "XCD_CHUNK"),
    ({"NUM_SM": 0}, 512, "NUM_SM"),
    ({"NUM_SM": 1.5}, 512, "NUM_SM must be an int"),
    ({"NUM_SM": True}, 512, "NUM_SM must be an int"),
    ({"CROSS_TILE_PREFETCH": 1}, 512, "must be a bool"),
    ({"L2_PREFETCH_DISTANCE": -1}, 512, "nonnegative"),
    ({"L2_PREFETCH_DISTANCE": 5}, 512, "number of K blocks"),
    ({"XCD_CHUNK": 1 << 30}, 512, "NUM_XCDS \\* XCD_CHUNK"),
    ({"GROUP_SIZE_M": 1 << 30}, 512, "GROUP_SIZE_M \\* number of N tiles"),
    ({"NUM_BUFFERS": 4, "CROSS_TILE_PREFETCH": False}, 512, "unsupported gfx1250 TDM schedule"),
    ({}, 264, "full K tiles"),
    ({}, 128, "fill the TDM pipeline"),
    ({}, 384, "even number of K blocks"),
    ([], 512, "config must be a dict"),
])
def test_grouped_gemm_interface_rejects_config(config, k, match, monkeypatch):
    a = torch.empty((1, k), dtype=torch.float16)
    b = torch.empty((257, k), dtype=torch.float16).T

    def no_allocation(*args, **kwargs):
        pytest.fail("invalid configuration reached device allocation")

    monkeypatch.setattr(torch, "empty", no_allocation)
    monkeypatch.setattr(torch, "tensor", no_allocation)
    with pytest.raises(ValueError, match=match):
        _gfx1250_grouped._make_grouped_gemm_args([a], [b], config)


@pytest.mark.parametrize("condition,match", [
    ("ragged_n", "same N and K"),
    ("ragged_k", "same N and K"),
    ("dtype", "FP16"),
    ("row_major_b", "canonical strides"),
    ("padded_a", "canonical strides"),
    ("unaligned_a", "16-byte aligned"),
    ("grad", "forward-only"),
    ("empty_groups", "same nonzero length"),
    ("length", "same nonzero length"),
    ("rank", "rank-2"),
    ("non_tensor", "rank-2"),
    ("k_mismatch", "mismatched reduction"),
    ("empty_n", "positive"),
    ("cpu", "requires an AMD gfx1250"),
])
def test_grouped_gemm_interface_rejects_inputs(condition, match, monkeypatch):
    a = [torch.empty((1, 512), dtype=torch.float16) for _ in range(2)]
    b = [torch.empty((257, 512), dtype=torch.float16).T for _ in range(2)]
    if condition == "ragged_n":
        b[1] = torch.empty((256, 512), dtype=torch.float16).T
    elif condition == "ragged_k":
        a[1] = torch.empty((1, 256), dtype=torch.float16)
        b[1] = torch.empty((257, 256), dtype=torch.float16).T
    elif condition == "dtype":
        a[0] = a[0].float()
    elif condition == "row_major_b":
        b[0] = b[0].contiguous()
    elif condition == "padded_a":
        a[0] = torch.empty((1, 520), dtype=torch.float16)[:, :512]
    elif condition == "unaligned_a":
        a[0] = torch.empty(513, dtype=torch.float16)[1:].view(1, 512)
    elif condition == "grad":
        a[0].requires_grad_()
    elif condition == "empty_groups":
        a, b = [], []
    elif condition == "length":
        b.pop()
    elif condition == "rank":
        a[0] = a[0].view(-1)
    elif condition == "non_tensor":
        a[0] = None
    elif condition == "k_mismatch":
        b[0] = b[0][:-1]
    elif condition == "empty_n":
        b[0] = b[0][:, :0]

    def no_allocation(*args, **kwargs):
        pytest.fail("invalid input reached output/metadata allocation")

    monkeypatch.setattr(torch, "empty", no_allocation)
    monkeypatch.setattr(torch, "tensor", no_allocation)
    with pytest.raises(ValueError, match=match):
        _gfx1250_grouped.grouped_gemm(a, b)


def test_grouped_gemm_interface_rejects_other_arch(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(gcnArchName="gfx950:sramecc+"))
    with pytest.raises(ValueError, match="requires gfx1250; got gfx950"):
        _gfx1250_grouped._require_grouped_gemm_gfx1250(torch.device("cuda"))


def test_grouped_gemm_interface_defaults_match_bench(monkeypatch):
    import sys
    from triton.language.extra.tlx.tutorials.amd_grouped_gemm_gfx1250 import bench

    checked = []

    def check_case(args, case, kernel_variant, index, total):
        cfg = _gfx1250_grouped._grouped_gemm_interface_launch_config(
            _gfx1250_grouped._grouped_gemm_interface_config(None, case.k), [case.m] * case.groups, case.n, case.k, 256,
            auto_cluster=True)
        for key, value in dict(
                BLOCK_SIZE_M=args.block_m,
                BLOCK_SIZE_N=args.block_n,
                BLOCK_SIZE_K=args.block_k,
                GROUP_SIZE_M=args.group_m,
                NUM_BUFFERS=args.tdm_pipeline_depth,
                L2_PREFETCH_DISTANCE=args.l2_prefetch_distance,
                C_STAGING_MODE=int(args.dedicated_c_buffer),
                CROSS_TILE_PREFETCH=args.cross_tile_prefetch,
                XCD_REMAP_MODE=_gfx1250_grouped._XCD_REMAP_MODES[args.xcd_remap],
                NUM_XCDS=args.num_xcds,
                XCD_CHUNK=args.xcd_chunk,
                CLUSTER_SIZE=args.cluster_size,
                CLUSTER_MULTICAST=args.cluster_multicast,
        ).items():
            assert cfg[key] == value, (case, key)
        assert cfg["ALLOW_MN_TAILS"] is False
        assert cfg["NUM_SM"] == 256
        checked.append(case)
        return bench.BenchResult(case, kernel_variant, case.tensor_gib, 0, None, None)

    monkeypatch.setattr(sys, "argv", ["bench.py", "--dry-run"])
    monkeypatch.setattr(bench, "_run_case", check_case)
    monkeypatch.setattr(bench, "_print_summary", lambda results: None)
    assert bench.main() == 0
    assert len(checked) == 2 * len(bench.DEFAULT_CASES)


@pytest.mark.parametrize("ms,n,num_sm,config,expected", [
    ([2048, 2048], 1024, 16, {}, 4),
    ([2048, 2048], 1024, 16, {"CLUSTER_SIZE": 1}, 1),
    ([2048, 2048], 1024, 16, {"CLUSTER_SIZE": 2}, 2),
    ([2048, 2048], 1024, 16, {"CLUSTER_SIZE": 4}, 4),
    ([2048, 2048], 1024, 16, {"CROSS_TILE_PREFETCH": False}, 1),
    ([2048, 2048], 1024, 16, {"C_STAGING_MODE": 1}, 1),
    ([2048, 2048], 1024, 16, {"XCD_REMAP_MODE": 0}, 1),
    ([2048, 2048], 1024, 24, {}, 1),
    ([3072, 3072], 1024, 32, {}, 1),
    ([2048, 1024], 1024, 16, {}, 1),
    ([0, 2048], 1024, 16, {}, 1),
    ([0, 0], 1024, 16, {}, 1),
    ([257, 513], 257, 16, {}, 1),
    ([1024], 512, 32, {}, 1),
])
def test_grouped_gemm_interface_selects_cluster(ms, n, num_sm, config, expected):
    cfg = _gfx1250_grouped._grouped_gemm_interface_launch_config(
        _gfx1250_grouped._grouped_gemm_interface_config(config, 512), ms, n, 512, num_sm, auto_cluster="CLUSTER_SIZE"
        not in config)
    assert cfg["CLUSTER_SIZE"] == expected
    if expected == 1 and "CLUSTER_SIZE" not in config:
        explicit = dict(config, CLUSTER_SIZE=4)
        with pytest.raises(ValueError, match="clustering requires"):
            _gfx1250_grouped._grouped_gemm_interface_launch_config(
                _gfx1250_grouped._grouped_gemm_interface_config(explicit, 512), ms, n, 512, num_sm, auto_cluster=False)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("cluster_size", [None, 2, 1])
def test_grouped_gemm_interface_multicast_launch(cluster_size, monkeypatch):
    torch.manual_seed(1250)
    m, n, k = 2048, 1024, 256
    cpu_a = [torch.randn((m, k), dtype=torch.float16) for _ in range(2)]
    cpu_b = [torch.randn((n, k), dtype=torch.float16) for _ in range(2)]
    a, b = [x.cuda() for x in cpu_a], [x.cuda().T for x in cpu_b]
    config = {} if cluster_size is None else dict(CLUSTER_SIZE=cluster_size)
    monkeypatch.setattr(_gfx1250_grouped, "_active_num_cus", lambda device: 16)
    *launch_args, outputs = _gfx1250_grouped._make_grouped_gemm_args(a, b, config)
    assert launch_args[-1]["CLUSTER_SIZE"] == (4 if cluster_size is None else cluster_size)
    assert launch_args[-1]["NUM_SM"] == 16
    assert launch_args[-1]["ALLOW_MN_TAILS"] is False
    _gfx1250_grouped._perf_fn(*launch_args)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _gfx1250_grouped._perf_fn(*launch_args)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == allocated
    for out, x, w in zip(outputs, cpu_a, cpu_b):
        torch.testing.assert_close(out.cpu(), (x.float() @ w.float().T).half(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("ms,n", [([0, 1, 255, 257], 257), ([256, 512], 256), ([0, 0], 1)])
def test_grouped_gemm_interface_list_api(ms, n):
    torch.manual_seed(1250)
    a = [torch.randn((m, 256), dtype=torch.float16) for m in ms]
    b = [torch.randn((n, 256), dtype=torch.float16) for _ in ms]
    outputs = _gfx1250_grouped.grouped_gemm(tuple(x.cuda() for x in a), tuple(x.cuda().T for x in b))
    assert isinstance(outputs, list) and len(outputs) == len(ms)
    for out, x, w in zip(outputs, a, b):
        torch.testing.assert_close(out.cpu(), (x.float() @ w.float().T).half(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("config,k", [
    ({}, 512),
    ({"CROSS_TILE_PREFETCH": False}, 384),
    ({"C_STAGING_MODE": 1}, 512),
    ({"BLOCK_SIZE_N": 128, "C_STAGING_MODE": 1}, 512),
    ({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "NUM_BUFFERS": 3, "CROSS_TILE_PREFETCH": False}, 384),
])
def test_grouped_gemm_interface_prepared_launch(config, k, monkeypatch):
    torch.manual_seed(1250)
    ms, n = [1, 257], 257
    cpu_a = [torch.randn((m, k), dtype=torch.float16) for m in ms]
    cpu_b = [torch.randn((n, k), dtype=torch.float16) for _ in ms]
    a, b = [x.cuda() for x in cpu_a], [x.cuda().T for x in cpu_b]
    args = _gfx1250_grouped._make_grouped_gemm_args(a, b, dict(config, NUM_SM=2))
    *launch_args, outputs = args
    assert len(args) == 8
    assert [tuple(t.shape) for t in args[:5]] == [(2, ), (2, ), (2, ), (6, ), (6, )]
    assert args[6]["NUM_SM"] == 2
    assert args[6]["CLUSTER_SIZE"] == 1
    assert args[6]["ALLOW_MN_TAILS"] is True
    assert args[0].cpu().tolist() == [x.data_ptr() for x in a]
    assert args[1].cpu().tolist() == [x.data_ptr() for x in b]
    _gfx1250_grouped._perf_fn(*launch_args)
    torch.cuda.synchronize()

    def no_setup(*args, **kwargs):
        pytest.fail("prepared launch allocated tables/outputs or read device metadata")

    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "tensor"):
            patch.setattr(torch, name, no_setup)
        for name in ("cpu", "item", "tolist"):
            patch.setattr(torch.Tensor, name, no_setup)
        _gfx1250_grouped._perf_fn(*launch_args)
        torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == allocated
    for out, x, w in zip(outputs, cpu_a, cpu_b):
        torch.testing.assert_close(out.cpu(), (x.float() @ w.float().T).half(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("condition,match", [
    ("dtype", "metadata table 0"),
    ("shape", "metadata table 3"),
    ("strides", "metadata table 4"),
    ("count", "positive int32 group count"),
    ("config", "cfg must include"),
    ("tail_flag", "ALLOW_MN_TAILS must be a bool"),
    ("cluster_tails", "clustered launches require ALLOW_MN_TAILS=False"),
    ("cluster_grid", "positive and divisible by 16"),
    ("device", "requires an AMD gfx1250"),
])
def test_grouped_gemm_interface_rejects_metadata(condition, match):
    tables = [torch.empty(2, dtype=torch.int64) for _ in range(3)]
    tables += [torch.empty(6, dtype=torch.int32) for _ in range(2)]
    g, cfg = 2, dict(K=512, NUM_SM=2, ALLOW_MN_TAILS=True, CLUSTER_SIZE=1)
    if condition == "dtype":
        tables[0] = tables[0].int()
    elif condition == "shape":
        tables[3] = tables[3][:-1]
    elif condition == "strides":
        tables[4] = torch.empty(12, dtype=torch.int32)[::2]
    elif condition == "count":
        g = 0
    elif condition == "config":
        cfg = {}
    elif condition == "tail_flag":
        cfg["ALLOW_MN_TAILS"] = 1
    elif condition == "cluster_tails":
        cfg.update(CLUSTER_SIZE=4, NUM_SM=16)
    elif condition == "cluster_grid":
        cfg.update(CLUSTER_SIZE=4, ALLOW_MN_TAILS=False)
    with pytest.raises(ValueError, match=match):
        _gfx1250_grouped._perf_fn(*tables, g, cfg)


@pytest.mark.parametrize("m,n,k,match", [
    (128, 256, 264, "full K tiles"),
    (128, 256, 128, "fill the TDM pipeline"),
])
def test_grouped_gemm_list_rejects_unsupported_tiles(m, n, k, match):
    a = torch.empty((m, k), dtype=torch.float16)
    b = torch.empty((n, k), dtype=torch.float16).T
    with pytest.raises(ValueError, match=match):
        _gfx1250_grouped.grouped_gemm_tdm_list([a], [b], num_programs=2)


def test_grouped_gemm_list_rejects_clustered_tails():
    a = torch.empty((257, 256), dtype=torch.float16)
    b = torch.empty((257, 256), dtype=torch.float16).T
    with pytest.raises(ValueError, match="tails require cluster_size=1"):
        _gfx1250_grouped.grouped_gemm_tdm_list([a], [b], block_m=256, block_n=256, num_programs=2, c_staging_mode=0,
                                               cross_tile_prefetch=True, cluster_size=2)


def test_grouped_gemm_list_tail_config_selection():
    cfg = _gfx1250_grouped._pick_grouped_gemm_config([1, 257], 257, 512, 2, allow_mn_tails=True)
    assert cfg["block_k"] == 128
    with pytest.raises(ValueError, match="no validated"):
        _gfx1250_grouped._pick_grouped_gemm_config([1, 257], 257, 512, 2)


@pytest.mark.parametrize("condition", ["row_major_b", "padded_a", "unaligned_a", "empty_k"])
def test_grouped_gemm_list_rejects_layout(condition):
    a = torch.empty((128, 256), dtype=torch.float16)
    b = torch.empty((256, 256), dtype=torch.float16).T
    match = "canonical strides"
    if condition == "row_major_b":
        b = b.T
    elif condition == "padded_a":
        a = torch.empty((128, 264), dtype=torch.float16)[:, :256]
    elif condition == "unaligned_a":
        a = torch.empty(128 * 256 + 1, dtype=torch.float16)[1:].view(128, 256)
        match = "16-byte aligned"
    else:
        a = a[:, :0]
        b = b[:0]
        match = "positive"
    with pytest.raises(ValueError, match=match):
        _gfx1250_grouped.grouped_gemm_tdm_list([a], [b])


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("block_m,staging,cross", [(128, 0, False), (256, 0, True), (256, 1, True)])
@pytest.mark.parametrize("m_tiles", [[0, 1, 0, 0, 0, 0, 2, 0], [0, 0, 0]])
def test_grouped_gemm_list_empty_m(block_m, staging, cross, m_tiles):
    torch.manual_seed(123)
    n, k = 512, 512
    cpu_a = [torch.randn((tiles * block_m, k), dtype=torch.float16) for tiles in m_tiles]
    cpu_b = [torch.randn((n, k), dtype=torch.float16) for _ in m_tiles]
    outputs = _gfx1250_grouped.grouped_gemm_tdm_list(
        [a.cuda() for a in cpu_a],
        [b.cuda().T for b in cpu_b],
        block_m=block_m,
        block_n=256,
        num_programs=2,
        c_staging_mode=staging,
        cross_tile_prefetch=cross,
    )
    assert len(outputs) == len(m_tiles)
    for out, a, b in zip(outputs, cpu_a, cpu_b):
        expected = (a.float() @ b.float().T).half()
        torch.testing.assert_close(out.cpu(), expected, atol=1e-2, rtol=1e-2)


@triton.constexpr_function
def _tdm_partitioned_view_layout(partition_dim):
    shape = [64, 128] if partition_dim == 0 else [128, 64]
    inner = tlx.padded_shared_layout_encoding.with_identity_for([(128, 8)], shape)
    return tlx.partitioned_shared_layout_encoding(2, 1, partition_dim, inner)


@triton.jit
def _tdm_partitioned_views_kernel(input_ptr, output_ptr, PARTITION_DIM: tl.constexpr, VIEW: tl.constexpr,
                                  FUSED: tl.constexpr):
    encoding: tl.constexpr = _tdm_partitioned_view_layout(PARTITION_DIM)
    a_buffers = tlx.local_alloc((128, 128), tl.bfloat16, 2, layout=encoding)
    b_buffers = tlx.local_alloc((128, 128), tl.bfloat16, 2, layout=encoding)
    a_desc = tl.make_tensor_descriptor(input_ptr, [256, 128], [128, 1], [128, 128])
    b_desc = tl.make_tensor_descriptor(input_ptr + 256 * 128, [256, 128], [128, 1], [128, 128])
    SIZE: tl.constexpr = 64 if VIEW == 'slice' or VIEW == 'dynamic' else 128
    for stage in tl.static_range(2):
        a = tlx.local_view(a_buffers, stage)
        b = tlx.local_view(b_buffers, stage)
        if FUSED:
            ad = tlx.update_tensor_descriptor(a_desc, add_offsets=[stage * 128, 0])
            bd = tlx.update_tensor_descriptor(b_desc, add_offsets=[stage * 128, 0])
            token = tlx.async_amd_descriptor_load_fused([(ad, a, 3), (bd, b, 12)])
            tlx.async_amd_descriptor_wait(tokens=[token])
        else:
            tlx.async_amd_descriptor_load(a_desc, a, [stage * 128, 0])
            tlx.async_amd_descriptor_load(b_desc, b, [stage * 128, 0])
            tlx.async_amd_descriptor_wait(0)
        if VIEW == 'transpose':
            a = tlx.local_trans(a)
            b = tlx.local_trans(b)
        elif VIEW == 'slice':
            a = tlx.local_slice(a, [64, 64], [64, 64])
            b = tlx.local_slice(b, [64, 64], [64, 64])
        elif VIEW == 'dynamic':
            # The host initializes this element to 64, but the compiler must
            # select the physical partition using its runtime value.
            offset = tl.load(input_ptr).to(tl.int32) & 0x40
            a = tlx.local_slice(a, [offset, offset], [64, 64])
            b = tlx.local_slice(b, [offset, offset], [64, 64])
        offsets = tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        tl.store(output_ptr + (stage * 2) * SIZE * SIZE + offsets, tlx.local_load(a))
        tl.store(output_ptr + (stage * 2 + 1) * SIZE * SIZE + offsets, tlx.local_load(b))


@pytest.mark.skipif(not is_hip_gfx1250(), reason='Requires gfx1250')
@pytest.mark.parametrize('partition_dim', [0, 1])
@pytest.mark.parametrize('view', ['full', 'transpose', 'slice', 'dynamic'])
@pytest.mark.parametrize('fused', [False, True])
def test_tdm_partitioned_views_correctness(partition_dim, view, fused):
    torch.manual_seed(1250)
    source = torch.randn((512, 128), dtype=torch.bfloat16, device='cuda')
    if view == 'dynamic':
        source[0, 0] = 64
    size = 64 if view in ('slice', 'dynamic') else 128
    output = torch.empty((4, size, size), dtype=source.dtype, device='cuda')
    kernel = _tdm_partitioned_views_kernel.warmup(source, output, partition_dim, view, fused, grid=(1, ), num_warps=4)
    assert kernel.metadata.shared <= 327680
    import re
    assert int(re.search(r'\.private_segment_fixed_size:\s*(\d+)', kernel.asm['amdgcn'])[1]) == 0
    _tdm_partitioned_views_kernel[(1, )](source, output, partition_dim, view, fused, num_warps=4)
    expected = torch.stack((source[:128], source[256:384], source[128:256], source[384:512]))
    if view == 'transpose':
        expected = expected.transpose(-1, -2)
    elif view in ('slice', 'dynamic'):
        expected = expected[:, 64:, 64:]
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert '#ttg.partitioned_shared' in kernel.asm['ttgir']


@pytest.mark.parametrize('partitions,groups,dimension,message', [
    (1, 1, 0, 'num_partitions'),
    (3, 1, 0, 'num_partitions'),
    (2, 0, 0, 'num_groups'),
    (2, 3, 0, 'num_groups'),
    (2, 1, -1, 'partition_dim'),
])
def test_partitioned_shared_invalid_parameters(partitions, groups, dimension, message):
    inner = tlx.padded_shared_layout_encoding.with_identity_for([(128, 8)], [64, 128])
    with pytest.raises(ValueError, match=message):
        tlx.partitioned_shared_layout_encoding(partitions, groups, dimension, inner)


def test_partitioned_shared_permute_preserves_bases():
    inner = tlx.padded_shared_layout_encoding.with_bases([(8, 2)], [[1, 0], [0, 1], [2, 0], [0, 2]], [4, 4])
    original = tlx.partitioned_shared_layout_encoding(2, 1, 0, inner)
    transposed = original.make_permute([1, 0])
    assert transposed.partition_dim == 1
    assert transposed.partition_layout.offset_bases == [[0, 1], [1, 0], [0, 2], [2, 0]]
    assert transposed.make_permute([1, 0]).partition_layout.offset_bases == inner.offset_bases


# Keep the eight-wave policy tests independent of the measured OCC1 selector.
@pytest.mark.parametrize("batch,hq,hk,n,num_cus,packed_dq,splits,pingpong,spread_dq,block_m,cache_score_k,tile_major", [
    (8, 64, 8, 1024, 192, True, 3, True, False, 64, False, True),
    (8, 64, 8, 2048, 192, True, 3, True, False, 64, False, True),
    (15, 64, 8, 16384, 192, True, 3, True, False, 64, False, True),
    (8, 64, 8, 16384, 384, True, 3, True, False, 64, False, True),
    (8, 64, 8, 16384, 192, False, 3, True, True, 64, False, True),
    (8, 32, 8, 2048, 192, True, 3, True, False, 64, False, True),
    (8, 64, 8, 768, 192, True, 3, True, False, 64, False, False),
    (1, 1, 1, 256, 192, True, 2, True, False, 64, False, False),  # two steps per split
    (1, 1, 1, 256, 1, True, 2, True, False, 64, False, False),
    (8, 64, 8, 1152, 192, True, 1, True, False, 32, False, False),
    (2, 6, 2, 384, 192, True, 1, False, False, 32, False, False),
    (8, 64, 8, 1024, 192, False, 3, True, True, 64, False, True),
    (8, 64, 8, 2048, 192, False, 3, True, True, 64, False, True),
    (8, 32, 8, 2048, 192, False, 3, True, True, 64, False, True),
    (8, 64, 8, 1152, 192, False, 1, True, False, 32, False, False),
    # Literal MHA expectations from the matched 1K-16K cache/split sweep.
    (16, 16, 16, 1024, 192, True, 1, True, False, 32, True, True),
    (16, 16, 16, 2048, 192, True, 1, True, False, 32, True, True),
    (16, 16, 16, 4096, 192, True, 1, True, False, 32, True, True),
    (16, 16, 16, 8192, 192, True, 1, True, False, 32, True, True),
    (16, 16, 16, 16384, 192, True, 3, True, False, 64, False, True),
    (16, 16, 16, 1024, 192, False, 1, True, False, 32, True, True),
    (16, 16, 16, 2048, 192, False, 1, True, False, 32, True, True),
    (16, 16, 16, 4096, 192, False, 1, True, False, 32, True, True),
    (16, 16, 16, 8192, 192, False, 1, True, False, 32, True, True),
    (16, 16, 16, 16384, 192, False, 3, True, True, 64, False, True),
])
def test_attention_backward_base_split_selection(batch, hq, hk, n, num_cus, packed_dq, splits, pingpong, spread_dq,
                                                 block_m, cache_score_k, tile_major):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base as select_config

    original = select_config(n, packed_dq=packed_dq)
    assert original.q_splits == 1 and not original.pingpong and not original.spread_dq
    assert original.packed_dq == packed_dq and not original.cache_score_k and not original.tile_major
    config = select_config(n, batch, hq, hk, num_cus=num_cus, packed_dq=packed_dq)
    assert config.block_n == original.block_n
    assert config.q_splits == splits
    assert config.pingpong == pingpong
    assert config.spread_dq == spread_dq
    assert config.block_m == block_m
    assert config.cache_score_k == cache_score_k
    assert config.packed_dq == packed_dq
    assert config.tile_major == tile_major
    if pingpong:
        assert config.reuse_q and not config.regalloc
        assert config.preprocess_block == 64 and config.convert_block == 4096


@pytest.mark.parametrize("shape,packed_dq,partition,front,bridge", [
    ((8, 64, 8, 2048), True, True, 1, 0),
    ((8, 64, 8, 2048), False, True, 1, 0),
    ((8, 64, 8, 16384), True, True, 2, 2),
    ((8, 64, 8, 16384), False, True, 1, 0),
    ((3, 64, 8, 4096), True, True, 1, 0),
    ((4, 32, 4, 16384), True, True, 2, 2),
    ((16, 16, 16, 1024), True, True, 0, 0),  # BM32: cached score K replaces the front regions
    ((16, 16, 16, 1024), False, True, 0, 4),
    ((16, 16, 16, 2048), False, True, 0, 4),
    ((16, 16, 16, 4096), False, True, 0, 4),
    ((16, 16, 16, 2048), True, True, 0, 0),
    ((16, 16, 16, 4096), True, True, 0, 0),
    ((16, 16, 16, 8192), False, True, 0, 0),
    ((16, 16, 16, 16384), True, True, 1, 0),
    ((2, 16, 16, 4096), False, True, 0, 4),
    ((8, 16, 16, 2048), False, True, 0, 4),
    ((16, 32, 32, 1024), False, True, 0, 4),
    ((4, 16, 16, 4096), False, True, 0, 4),
    ((1, 8, 8, 4096), False, True, 1, 0),  # MHA grid below one wave: BM64 path
    ((8, 64, 8, 1152), True, False, 0, 0),  # BN128
    ((2, 6, 2, 384), True, False, 0, 0),
])
def test_attention_backward_base_lds_selection(shape, packed_dq, partition, front, bridge):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base as select_config

    b, hq, hk, n = shape
    config = select_config(n, b, hq, hk, num_cus=192, packed_dq=packed_dq)
    assert config.partition_lds == partition
    assert config.front_regions == front and config.bridge_regions == bridge
    legacy = select_config(n, packed_dq=packed_dq)
    assert not legacy.partition_lds and legacy.front_regions == 0 and legacy.bridge_regions == 0


@pytest.mark.parametrize("n", [1024, 2048, 4096, 8192, 16384])
@pytest.mark.parametrize("packed_dq", [False, True])
def test_attention_backward_base_cluster_selection(n, packed_dq):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base as select_config

    config = select_config(n, 16, 16, 16, num_cus=192, packed_dq=packed_dq)
    assert config.cluster_dq and config.tile_major
    assert config.q_splits == (3 if n == 16384 else 1) and config.block_n == 256
    assert select_config(n, 8, 16, 16, num_cus=192, packed_dq=packed_dq).cluster_dq
    gqa = select_config(n, 8, 64, 8, num_cus=192, packed_dq=packed_dq)
    assert gqa.cluster_dq and gqa.tile_major and gqa.q_splits == 3
    # Legacy n-only calls and other sequence lengths stay unchanged.
    assert not select_config(n, packed_dq=packed_dq).cluster_dq
    assert select_config(8192, 16, 16, 16, num_cus=192, packed_dq=packed_dq).cluster_dq
    assert select_config(8192, 8, 64, 8, num_cus=192, packed_dq=packed_dq).cluster_dq
    assert not select_config(n + 512, 16, 16, 16, num_cus=192, packed_dq=packed_dq).cluster_dq
    assert not select_config(n + 512, 8, 64, 8, num_cus=192, packed_dq=packed_dq).cluster_dq


def _compile_attention_backward_kernel(config, shape, *, compile_options=None, constants_overrides=None):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _compiler_options, _delayed_bwd_kernel

    b, hq, hk, n = shape
    constants = dict(
        HQ=hq, HK=hk, N=n, SM_SCALE=128**-0.5, BM=config.block_m, BN=config.block_n, NW=config.num_warps,
        TDM=config.use_tdm, STAGES=config.stages, DELAYED=config.delayed, UNROLL=config.unroll,
        FRAGMENT_FENCES=config.fragment_fences, FRAGMENT_FENCE_MASK=config.fragment_fence_mask,
        FRAGMENTED=config.fragmented, RESIDENT_KV=config.resident_kv, ORDERED=config.ordered,
        PACKED_DQ=config.packed_dq, REMATERIALIZE=config.rematerialize, OUTER_M=config.outer_m,
        ASYNC_STATS=config.async_stats, Q_SPLITS=config.q_splits, REUSE_Q=config.reuse_q, PINGPONG=config.pingpong,
        SPREAD_DQ=config.spread_dq, LOG2_LSE=config.log2_lse, CACHE_SCORE_K=config.cache_score_k,
        TILE_MAJOR=config.tile_major, PARTITION_LDS=config.partition_lds, FRONT_REGIONS=config.front_regions,
        BRIDGE_REGIONS=config.bridge_regions, CLUSTER_DQ=config.cluster_dq,
        DQ_SLICES=max(1, n // config.block_n // 16) if config.cluster_dq else 1, OCC1=config.occ1,
        OCC1_K_BANDS=config.occ1_k_bands, OCC1_PREFETCH=config.occ1_prefetch, OCC1_SCHED=config.occ1_sched,
        PHASE_IGLP=config.phase_iglp, OCC1_DQ_FIRST=config.occ1_dq_first, OCC1_LOAD_AFTER_DK=config.occ1_load_after_dk)
    constants["SPLIT_KV_STORES"] = config.occ1 and config.packed_dq and config.q_splits > 1
    tail_splits = getattr(config, "tail_splits", 1)
    tail_base = tail_owners = 0
    if tail_splits > 1:
        from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _tail_partition
        tail_base, tail_owners = _tail_partition(config, b, hk, n)
    constants.update(TAIL_SPLITS=tail_splits, TAIL_BASE=tail_base, TAIL_OWNERS=tail_owners)
    if tail_splits == 1:
        constants.update(TAIL_DK=None, TAIL_DV=None)
    constants.update(constants_overrides or {})
    signature = {name: "*bf16" for name in ("Q", "K", "V", "DO")}
    signature.update({name: "*fp32" for name in ("LSE", "Delta")})
    signature["DQA"] = "*bf16" if config.packed_dq else "*fp32"
    signature.update({name: "*fp32" if config.q_splits > 1 else "*bf16" for name in ("DK", "DV")})
    if tail_splits > 1:
        signature.update(DK="*fp32", DV="*fp32", TAIL_DK="*bf16", TAIL_DV="*bf16")
    attrs = {(_delayed_bwd_kernel.arg_names.index(name), ): [("tt.divisibility", 16)] for name in signature}
    if config.occ1:
        # Match runtime specialization: these buffers have a known byte range.
        qe, ke = b * hq * n * 128, b * hk * n * 128
        sizes = dict(Q=qe * 2, K=ke * 2, V=ke * 2, DO=qe * 2, LSE=b * hq * n * 4, Delta=b * hq * n * 4,
                     DQA=qe * (2 if config.packed_dq else 4) * constants["DQ_SLICES"],
                     DK=ke * (4 * config.q_splits if config.q_splits > 1 else 2),
                     DV=ke * (4 * config.q_splits if config.q_splits > 1 else 2))
        if tail_splits > 1:
            sizes.update(DK=tail_owners * config.block_n * 128 * 4 * tail_splits,
                         DV=tail_owners * config.block_n * 128 * 4 * tail_splits, TAIL_DK=ke * 2, TAIL_DV=ke * 2)
        for name, size in sizes.items():
            if size <= 2**31 - 1:
                attrs[(_delayed_bwd_kernel.arg_names.index(name), )].append(("tt.pointer_range", 32))
    return triton.compile(ASTSource(_delayed_bwd_kernel, signature, constants,
                                    attrs=attrs), target=GPUTarget("hip", "gfx1250", 32),
                          options={**_compiler_options(config, n), **(compile_options or {})})


def _attention_backward_isa_loop(assembly):
    import re

    lines = assembly.splitlines()
    labels = {}
    loops = []
    for index, line in enumerate(lines):
        label = re.match(r"^([.\w]+):", line)
        if label:
            labels[label[1]] = index
        branch = re.match(r"\s*s_(?:cbranch_\w+|branch)\s+([.\w]+)", line)
        if branch and labels.get(branch[1], index) < index:
            loops.append(lines[labels[branch[1]]:index + 1])
    return max(loops, key=lambda body: sum("v_wmma_" in line for line in body))


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("spread_dq,log2_lse,cache_score_k", [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (False, False, True),
    (False, True, True),
])
def test_attention_backward_fp32_dq_addressing(spread_dq, log2_lse, cache_score_k):
    import dataclasses
    import re
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base as select_config

    # FP32 dQ atomics must share one address VGPR with immediate offsets.
    # Otherwise each atomic rewrites the address and waits on s_wait_xcnt.
    hq, hk = (16, 16) if cache_score_k else (64, 8)
    config = dataclasses.replace(select_config(2048, 16 if cache_score_k else 8, hq, hk, num_cus=192,
                                               packed_dq=False), spread_dq=spread_dq, log2_lse=log2_lse,
                                 cache_score_k=cache_score_k, block_m=32 if cache_score_k else 64,
                                 q_splits=1 if cache_score_k else 3, front_regions=0 if cache_score_k else 1)
    kernel = _compile_attention_backward_kernel(config, (16 if cache_score_k else 8, hq, hk, 2048))
    memory_atomics = [
        line for line in kernel.asm["amdgcn"].splitlines() if re.match(r"\s*(?:buffer|global|flat)_atomic_", line)
    ]
    # Cluster dQ atomics use SE scope; split dK/dV partials keep device scope.
    assert memory_atomics and all("scope:SCOPE_SE" in line or "scope:SCOPE_DEV" in line for line in memory_atomics)
    loop = _attention_backward_isa_loop(kernel.asm["amdgcn"])
    atomics = [line for line in loop if "buffer_atomic_add_f32" in line]
    scope = "scope:SCOPE_SE" if config.cluster_dq else "scope:SCOPE_DEV"
    assert all(scope in line for line in atomics)
    assert (any("0x3fb8aa3b" in line for line in loop)) == (not log2_lse)
    assert len(atomics) == (16 if cache_score_k else 32)
    addresses = [re.search(r",\s*(v\d+)(?:\s*/\*(v\d+)\*/)?\s*,\s*s\[", line) for line in atomics]
    assert len({match[2] or match[1] for match in addresses}) == 1
    assert not any("s_wait_xcnt" in line for line in loop)
    assert sum("v_wmma_" in line for line in loop) == (80 if cache_score_k else 160)
    assert sum(bool(re.match(r"\s*ds_load", line)) for line in loop) == (136 if cache_score_k else 240)
    assert int(re.search(r"\.vgpr_count:\s*(\d+)", kernel.asm["amdgcn"])[1]) <= 512
    assert int(re.search(r"\.private_segment_fixed_size:\s*(\d+)", kernel.asm["amdgcn"])[1]) == 0


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("shape,packed_dq", [((8, 64, 8, 16384), True), ((8, 64, 8, 16384), False),
                                             ((16, 16, 16, 1024), True), ((16, 16, 16, 1024), False)])
def test_attention_backward_partitioned_lds_assembly(shape, packed_dq):
    import re
    from pathlib import Path
    from triton.tlx.ops.kernels.flash_attn import gfx1250_bwd

    config = gfx1250_bwd._select_config_base(shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq)
    assert config.partition_lds
    kernel = _compile_attention_backward_kernel(config, shape)
    assembly, llir = kernel.asm["amdgcn"], kernel.asm["llir"]
    assert kernel.metadata.shared <= 327680
    assert int(re.search(r"\.private_segment_fixed_size:\s*(\d+)", assembly)[1]) == 0
    assert int(re.search(r"\.vgpr_count:\s*(\d+)", assembly)[1]) <= 512
    loop = _attention_backward_isa_loop(assembly)
    assert sum("v_wmma_" in line for line in loop) == (160 if config.block_m == 64 else 80)
    assert sum(bool(re.match(r"\s*ds_load", line)) for line in loop) == (240 if config.block_m == 64 else 136)
    assert sum("s_barrier_wait -1" in line for line in loop) == 2
    assert "partitionDim = 1" in kernel.asm["ttgir"]
    # The dS seed constructs both physical piece pointers at its source
    # location. Every ring stage of each piece must stay in its own 64 KiB
    # LDS partition.
    lines = Path(gfx1250_bwd.__file__).read_text().splitlines()
    seed_line = next(i for i, line in enumerate(lines, 1) if "tlx.local_store(tlx.local_view(dsb, (step_start" in line)
    locations = re.findall(r"!(\d+) = !DILocation\(line: " + str(seed_line) + r",", llir)
    bases = set()
    for line in llir.splitlines():
        if any(re.search(r"!dbg !" + location + r"\b", line) for location in locations):
            bases.update(int(value) for value in re.findall(r"@global_smem, i32 (\d+)", line))
    assert len(bases) == 2
    piece_bytes = 3 * config.block_m * (config.block_n // 2) * 2 * 528 // 512
    partitions = [set(range(base // 65536, (base + piece_bytes - 1) // 65536 + 1)) for base in sorted(bases)]
    assert partitions[0].isdisjoint(partitions[1])
    assert all(base + piece_bytes <= kernel.metadata.shared for base in bases)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [True, False])
def test_attention_backward_bridge_regions_assembly(packed_dq):
    import dataclasses
    import re
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base as select_config

    shape = (8, 64, 8, 16384)
    config = dataclasses.replace(select_config(shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq),
                                 front_regions=2, bridge_regions=2)
    kernel = _compile_attention_backward_kernel(config, shape)
    assembly = kernel.asm["amdgcn"]
    assert kernel.metadata.shared <= 327680
    assert int(re.search(r"\.private_segment_fixed_size:\s*(\d+)", assembly)[1]) == 0
    assert int(re.search(r"\.vgpr_count:\s*(\d+)", assembly)[1]) <= 512
    loop = _attention_backward_isa_loop(assembly)
    assert sum("v_wmma_" in line for line in loop) == 160
    assert sum(bool(re.match(r"\s*ds_load", line)) for line in loop) == 240
    assert sum("s_barrier_wait -1" in line for line in loop) == 2
    regions = kernel.asm["llir"].split("call void @llvm.amdgcn.sched.barrier(i32 0)")
    assert len(regions) >= 24
    # Native WMMA blocks and their next-region LDS reads stay in separate
    # full-fence intervals. The final blocks need no next read.
    blocks = [region for region in regions if region.count("call <8 x float> @llvm.amdgcn.wmma") in (4, 8)]
    assert len(blocks) >= 16
    assert sum("ptr addrspace(3)" in block for block in blocks) >= 12


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("bridge_regions", [3, 4])
def test_attention_backward_early_q_assembly(packed_dq, bridge_regions):
    import dataclasses
    from pathlib import Path
    import re
    from triton.tlx.ops.kernels.flash_attn import gfx1250_bwd

    shape = (16, 16, 16, 1024)
    config = dataclasses.replace(
        gfx1250_bwd._select_config_base(shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq),
        bridge_regions=bridge_regions)
    kernel = _compile_attention_backward_kernel(config, shape)
    assembly = kernel.asm["amdgcn"]
    assert kernel.metadata.shared <= 327680
    assert int(re.search(r"\.vgpr_count:\s*(\d+)", assembly)[1]) <= 512
    assert int(re.search(r"\.private_segment_fixed_size:\s*(\d+)", assembly)[1]) == 0
    loop = _attention_backward_isa_loop(assembly)
    assert sum("v_wmma_" in line for line in loop) == 80
    assert sum(bool(re.match(r"\s*ds_load", line)) for line in loop) == 136
    assert sum("s_barrier_wait -1" in line for line in loop) == 2
    assert not any("s_wait_xcnt" in line for line in loop)
    if not packed_dq:
        atomics = [line for line in loop if "buffer_atomic_add_f32" in line]
        addresses = [re.search(r",\s*(v\d+)(?:\s*/\*(v\d+)\*/)?\s*,\s*s\[", line) for line in atomics]
        assert len(atomics) == 16
        assert len({match[2] or match[1] for match in addresses}) == 1

    source = Path(gfx1250_bwd.__file__).read_text().splitlines()

    def marker(name):
        matches = [i for i, line in enumerate(source, 1) if line.strip() == "# " + name]
        assert len(matches) == 1, (name, matches)
        return matches[0]

    begin = marker("early-q-seed-begin")
    end = marker("early-q-seed-end")
    join_line = marker("early-q-front-join") + 1
    assert source[join_line - 1].strip() == "tlx.workgroup_barrier()"
    location = None
    seed_reads = []
    front_joins = []
    lookahead_reads = []
    lookahead_begin = marker("early-q-lookahead-begin")
    lookahead_end = marker("early-q-lookahead-end")
    for index, line in enumerate(loop):
        if line.lstrip().startswith(".loc"):
            match = re.search(r"gfx1250_bwd\.py:(\d+):", line)
            location = int(match[1]) if match else None
        if location is not None and begin < location < end and re.match(r"\s*ds_load", line):
            seed_reads.append(index)
        if location is not None and lookahead_begin < location < lookahead_end and re.match(r"\s*ds_load", line):
            lookahead_reads.append(index)
        if location == join_line and "s_barrier_wait -1" in line:
            front_joins.append(index)
    # The first panel moves across the existing front rendezvous. It adds
    # neither an operand read nor a workgroup barrier to the steady loop.
    assert len(seed_reads) == 4
    # LLVM may rotate the loop so its other rendezvous appears first in
    # assembly. Identify the front rendezvous by its source location.
    assert len(front_joins) == 1
    assert max(seed_reads) < front_joins[0]

    # Mode 4 loads exactly three further Q panels through its lookahead
    # branch. This also detects an earlier branch shadowing the new path.
    assert len(lookahead_reads) == (12 if bridge_regions == 4 else 0)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize(
    "overrides",
    [dict(block_m=64),
     dict(cache_score_k=False),
     dict(q_splits=2),
     dict(reuse_q=False),
     dict(spread_dq=True)])
@pytest.mark.parametrize("bridge_regions", [3, 4])
def test_attention_backward_early_q_invalid_config(overrides, bridge_regions):
    import dataclasses
    from triton.tlx.ops.kernels.flash_attn import gfx1250_bwd

    q = torch.empty((1, 4, 256, 128), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((1, 4, 256), dtype=torch.float32, device="cuda")
    config = gfx1250_bwd.BackwardConfig(block_m=32, block_n=256, fragmented=True, ordered=True, async_stats=True,
                                        pingpong=True, reuse_q=True, cache_score_k=True, bridge_regions=bridge_regions)
    config = dataclasses.replace(config, **overrides)
    with pytest.raises(ValueError, match="early Q read requires"):
        gfx1250_bwd.fa_backward(q, q, q, q, q, lse, config=config)


def _attention_backward_policies():
    import itertools

    # The original transport, staging and scheduling controls.
    controls = [
        (2, False, True, 32, 128, 2, None),
        (4, False, True, 32, 128, 2, None),
        (4, True, True, 32, 128, 2, None),
        (8, True, True, 32, 128, 2, None),
        (4, True, False, 32, 128, 2, None),
        (8, True, True, 64, 256, 2, None),
        (8, True, True, 64, 256, 3, None),
        (8, True, True, 32, 128, 4, None),
        (8, True, True, 64, 256, 2, 0),
        (8, True, True, 64, 256, 2, 4086),
    ]
    rows = [row + (1, False, False, 0, False, False, False, False) for row in controls]
    for bn, splits, packed in itertools.product((128, 256), (1, 2, 3), (False, True)):
        rows.append((8, True, True, 64, bn, 2, None, splits, packed, True, 0, False, False, False, False))
    for splits in (1, 3):
        rows.append((4, True, False, 32, 128, 2, None, splits, True, True, 0, False, False, False, False))
    for splits in (2, 3):
        rows.append((8, True, True, 32, 256, 2, None, splits, True, True, 64, False, False, False, False))
    for bm, bn, splits, packed in itertools.product((32, 64), (128, 256), (1, 3), (False, True)):
        rows.append((8, True, True, bm, bn, 2, None, splits, packed, True, 0, True, False, False, False))
    for splits in (2, 4):
        rows.append((8, True, True, 64, 256, 2, None, splits, True, True, 0, True, False, False, False))
    for bn, splits, packed in itertools.product((128, 256), (1, 3), (False, True)):
        rows.append((4, True, True, 32, bn, 2, None, splits, packed, True, 0, False, False, False, False))
    for bm, splits in itertools.product((32, 64), (2, 3)):
        rows.append((8, True, True, bm, 256, 2, None, splits, False, True, 0, True, True, False, False))
    for packed in (True, False):
        rows.append((8, True, True, 64, 256, 2, None, 3, packed, True, 0, True, not packed, True, False))
    # Measured MHA schedule, with and without preprocessing the log2 LSE.
    for packed, log2_lse in itertools.product((True, False), (False, True)):
        rows.append((8, True, True, 32, 256, 2, None, 1, packed, True, 0, True, False, log2_lse, True))
    rows = [row + (False, False, 0, 0) for row in rows]
    # LDS partitions, front/bridge regions and grid order, alone and as selected.
    for packed in (True, False):
        gqa = (8, True, True, 64, 256, 2, None, 3, packed, True, 0, True, not packed, False, False)
        mha = (8, True, True, 32, 256, 2, None, 1, packed, True, 0, True, False, False, True)
        rows.append(gqa + (False, True, 0, 0))
        rows.append(gqa + (False, False, 1, 0))
        rows.append(gqa + (False, False, 0, 1))
        rows.append(gqa + (True, False, 0, 0))
        rows.append(gqa + (False, True, 1, 0))
        rows.append(gqa + (True, True, 2, 2))
        rows.append(mha + (False, True, 0, 0))
        rows.append(mha + (False, True, 0, 3))
        rows.append(mha + (False, True, 0, 4))
        rows.append(mha[:14] + (False, ) + (False, True, 1, 0))
    return rows


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize(
    "num_warps,use_tdm,delayed,bm,bn,stages,fence_mask,q_splits,packed_dq,full_policy,outer_m,pingpong,spread_dq,log2_lse,cache_score_k,tile_major,partition_lds,front_regions,bridge_regions",
    _attention_backward_policies())
@pytest.mark.parametrize("batch,hq,hk,n", [(1, 1, 1, 256), (2, 6, 2, 512), (1, 4, 4, 512)])
def test_attention_backward_delayed_gqa(num_warps, use_tdm, delayed, bm, bn, stages, fence_mask, batch, hq, hk, n,
                                        q_splits, packed_dq, full_policy, outer_m, pingpong, spread_dq, log2_lse,
                                        cache_score_k, tile_major, partition_lds, front_regions, bridge_regions):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import BackwardConfig, allocate_workspace, fa_backward

    torch.manual_seed(2709)
    q = torch.randn((batch, hq, n, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, hk, n, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    qr, kr, vr = [t.float().requires_grad_() for t in (q, k, v)]
    scores = qr @ kr.repeat_interleave(hq // hk, dim=1).transpose(-1, -2) * 128**-0.5
    out = scores.softmax(-1) @ vr.repeat_interleave(hq // hk, dim=1)
    expected = torch.autograd.grad(out, (qr, kr, vr), dout.float())
    lse = scores.detach().logsumexp(-1)
    # Immediate four-wave controls need the same spill-free allocation as delayed ones.
    wide_registers = full_policy and num_warps == 4
    config = BackwardConfig(block_m=bm, block_n=bn, num_warps=num_warps, use_tdm=use_tdm, delayed=delayed,
                            stages=stages, fragment_fences=fence_mask is not None, fragment_fence_mask=fence_mask or 0,
                            q_splits=q_splits, packed_dq=packed_dq, fragmented=full_policy, ordered=full_policy,
                            async_stats=full_policy and use_tdm, rematerialize=full_policy and not wide_registers,
                            regalloc=full_policy and not pingpong, waves_per_eu=1 if wide_registers else 0,
                            outer_m=outer_m, reuse_q=full_policy, pingpong=pingpong, spread_dq=spread_dq,
                            log2_lse=log2_lse, cache_score_k=cache_score_k, tile_major=tile_major,
                            partition_lds=partition_lds, front_regions=front_regions, bridge_regions=bridge_regions)
    if pingpong and bm == 64 and bn == 256 and q_splits == 1:
        with pytest.raises(ValueError, match="requires at least two Q splits"):
            fa_backward(q, k, v, out.detach().bfloat16(), dout, lse, config=config)
        return
    if pingpong and (hq // hk) * n // bm // q_splits < 2:
        with pytest.raises(ValueError, match="at least two steps per Q split"):
            fa_backward(q, k, v, out.detach().bfloat16(), dout, lse, config=config)
        return
    workspace = allocate_workspace(q, k, config=config)
    assert len(workspace) == (7 if q_splits > 1 else 5) + int(log2_lse)
    if q_splits > 1:
        assert all(value.dtype == torch.float32 and value.shape == k.shape for value in workspace[5:7])
    for _ in range(2):
        actual = fa_backward(q, k, v, out.detach().bfloat16(), dout, lse, config=config, workspace=workspace)
        if log2_lse:
            torch.testing.assert_close(workspace[-1], lse * 1.4426950408889634, rtol=0, atol=0)
        for name, value, ref in zip(("dq", "dk", "dv"), actual, expected):
            assert torch.isfinite(value).all()
            nrmse = ((value.float() - ref).square().mean() / ref.square().mean()).sqrt()
            assert nrmse < 0.008, (name, nrmse.item())


@pytest.mark.parametrize("changes,n", [
    ({}, 1536),
    ({}, 6144),
    ({"tile_major": False}, 1024),
    ({"block_n": 128}, 1024),
])
def test_attention_backward_cluster_validation(changes, n):
    import dataclasses
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import BackwardConfig, _compiler_options

    config = dataclasses.replace(BackwardConfig(tile_major=True, cluster_dq=True), **changes)
    with pytest.raises(ValueError, match="cluster_dq requires"):
        _compiler_options(config, n)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("n", [512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("packed_dq", [False, True])
def test_attention_backward_cluster_assembly(n, packed_dq):
    import dataclasses
    import re
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base as select_config

    config = dataclasses.replace(select_config(1024, 16, 16, 16, num_cus=192, packed_dq=packed_dq), tile_major=True,
                                 cluster_dq=True)
    kernel = _compile_attention_backward_kernel(config, (1, 16, 16, n))
    # Keep the cluster launch fixed when isolating the atomic scope change.
    baseline = _compile_attention_backward_kernel(dataclasses.replace(config, cluster_dq=False), (1, 16, 16, n),
                                                  compile_options={"ctas_per_cga": (min(n // 256, 16), 1, 1)},
                                                  constants_overrides={"DQ_SLICES": max(1, n // 4096)})
    assembly = kernel.asm["amdgcn"]
    atomics = [line for line in assembly.splitlines() if re.match(r"\s*buffer_atomic_", line)]
    # With one Q split, the main kernel has only dQ atomics; dK/dV are stores.
    assert len(atomics) == (16 if packed_dq else 32)
    assert all("scope:SCOPE_SE" in line for line in atomics)
    # LLVM's existing instruction prefetch also uses SE scope. It is not an
    # atomic and must remain identical in the complete instruction comparison.
    memory_atomics = [line for line in assembly.splitlines() if re.match(r"\s*(?:buffer|global|flat)_atomic_", line)]
    assert sum("scope:SCOPE_SE" in line for line in memory_atomics) == len(atomics)
    assert kernel.metadata.shared == baseline.metadata.shared
    assert int(re.search(r"\.vgpr_count:\s*(\d+)", assembly)[1]) <= 512
    assert int(re.search(r"\.private_segment_fixed_size:\s*(\d+)", assembly)[1]) == 0
    loop = _attention_backward_isa_loop(assembly)
    assert sum("v_wmma_" in line for line in loop) == 80
    assert sum(bool(re.match(r"\s*ds_load", line)) for line in loop) == 136
    assert sum("s_barrier_wait" in line for line in loop) == 2
    assert not any("s_wait_xcnt" in line for line in loop)

    def instructions(text):
        return [
            " ".join(re.sub(r"/\*.*?\*/", "", line).split("//")[0].split())
            for line in text.splitlines()
            if re.match(r"\s*(?:s_|v_|ds_|buffer_|flat_|global_|scratch_|tensor_|image_)\w", line)
        ]

    assert instructions(assembly) == [
        line.replace("scope:SCOPE_DEV", "scope:SCOPE_SE") if "buffer_atomic_" in line else line
        for line in instructions(baseline.asm["amdgcn"])
    ]


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("shape", [(8, 64, 8, 2048), (8, 64, 8, 8192)])
def test_attention_backward_cluster_split_assembly(packed_dq, shape):
    import dataclasses
    import re
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base as select_config

    config = dataclasses.replace(select_config(shape[3], *shape[:3], num_cus=192, packed_dq=packed_dq), tile_major=True,
                                 cluster_dq=True)
    assert config.q_splits > 1
    kernel = _compile_attention_backward_kernel(config, shape)
    baseline = _compile_attention_backward_kernel(dataclasses.replace(config, cluster_dq=False), shape,
                                                  compile_options={"ctas_per_cga": (min(shape[3] // 256, 16), 1, 1)},
                                                  constants_overrides={"DQ_SLICES": max(1, shape[3] // 4096)})

    def instructions(text):
        return [
            " ".join(re.sub(r"/\*.*?\*/", "", line).split("//")[0].split())
            for line in text.splitlines()
            if re.match(r"\s*(?:s_|v_|ds_|buffer_|flat_|global_|scratch_|tensor_|image_)\w", line)
        ]

    assert kernel.metadata.shared == baseline.metadata.shared
    assert int(re.search(r"\.vgpr_count:\s*(\d+)", kernel.asm["amdgcn"])[1]) <= 512
    assert int(re.search(r"\.private_segment_fixed_size:\s*(\d+)", kernel.asm["amdgcn"])[1]) == 0
    new, old = instructions(kernel.asm["amdgcn"]), instructions(baseline.asm["amdgcn"])
    assert len(new) == len(old)
    changed = [(a, b) for a, b in zip(new, old) if a != b]
    # Only dQ atomics move to SE scope; the dK/dV split partials stay device scoped.
    assert changed and all(
        a.startswith("buffer_atomic_") and a == b.replace("scope:SCOPE_DEV", "scope:SCOPE_SE") for a, b in changed)
    atomics = [line for line in new if line.startswith("buffer_atomic_")]
    assert sum("scope:SCOPE_SE" in line for line in atomics) == len(changed)
    assert sum("scope:SCOPE_DEV" in line for line in atomics) > 0


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("shape", [(1, 4, 4, 512), (4, 16, 16, 1024), (2, 16, 16, 2048), (1, 16, 16, 4096),
                                   (1, 8, 2, 1024), (2, 6, 2, 512), (1, 64, 8, 2048), (1, 4, 4, 8192), (1, 8, 2, 8192),
                                   (1, 4, 4, 16384)])
def test_attention_backward_cluster_correctness(shape, packed_dq, monkeypatch):
    import dataclasses
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base as select_config, allocate_workspace, fa_backward

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(2709)
    b, hq, hk, n = shape
    q, dout = [torch.randn((b, hq, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    k, v = [torch.randn((b, hk, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    qr, kr, vr = [t.float().requires_grad_() for t in (q, k, v)]
    scores = qr @ kr.repeat_interleave(hq // hk, 1).transpose(-1, -2) * 128**-0.5
    out = scores.softmax(-1) @ vr.repeat_interleave(hq // hk, 1)
    expected = torch.autograd.grad(out, (qr, kr, vr), dout.float())
    case = (q, k, v, out.detach().bfloat16(), dout, scores.detach().logsumexp(-1))
    # GQA keeps its Q-split policy; uneven splits share one cluster per split.
    policy = select_config(n, b, hq, hk, num_cus=192, packed_dq=packed_dq) if hq > hk else select_config(
        1024, 16, 16, 16, num_cus=192, packed_dq=packed_dq)
    config = dataclasses.replace(policy, tile_major=True, cluster_dq=True)
    base_config = dataclasses.replace(config, cluster_dq=False)
    old_workspace = allocate_workspace(q, k, config=base_config)
    workspace = allocate_workspace(q, k, config=config)

    def nrmse(value, ref):
        return ((value.float() - ref.float()).square().mean() / ref.float().square().mean()).sqrt().item()

    production = []
    for call in range(5):
        value = fa_backward(*case, config=base_config, workspace=old_workspace)
        if call == 0:
            production_kv = (value[1].clone(), value[2].clone())
        production.append(value[0].clone())
    production_noise = max(nrmse(value, production[0]) for value in production[1:])
    bound = 2 * production_noise + 1e-3 if packed_dq else 1e-4
    for call in range(5):
        workspace[1].fill_(float("nan"))
        actual = fa_backward(*case, config=config, workspace=workspace)
        if call < 3:
            for value, ref in zip(actual, expected):
                assert torch.isfinite(value).all()
                assert nrmse(value, ref) < 0.012
        assert nrmse(actual[0], production[0]) <= bound
        assert all(nrmse(value, ref) <= 1e-4 for value, ref in zip(actual[1:], production_kv))


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
def test_attention_backward_cluster_workspace_validation():
    import dataclasses
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base as select_config, allocate_workspace, fa_backward

    q = torch.empty((1, 4, 8192, 128), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device="cuda", dtype=torch.float32)
    config = dataclasses.replace(select_config(1024, 16, 16, 16, num_cus=192), tile_major=True, cluster_dq=True)
    workspace = allocate_workspace(q, q, config=config)
    assert workspace[1].shape == (2, ) + q.shape
    wrong = (workspace[0], torch.empty_like(q), *workspace[2:])
    with pytest.raises(ValueError, match="invalid backward workspace"):
        fa_backward(q, q, q, q, q, lse, config=config, workspace=wrong)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("slices", [2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_attention_backward_convert_slices(slices, packed):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _convert_outputs

    size = 2 * 64 * 128
    dtype = torch.bfloat16 if packed else torch.float32
    source = torch.randint(-128, 128, (slices, size), device="cuda").to(dtype) / 16
    offsets = torch.arange(size, device="cuda", dtype=torch.int64)
    shuffled = (offsets & ~(1024 | 16)) | ((offsets & 1024) >> 6) | ((offsets & 16) << 6)
    if packed:
        shuffled = ((offsets & ~(15 | 16 | 32 | 128 | 1024)) | ((offsets & 15) << 1)
                    | ((offsets & 16) << 6) | ((offsets & 32) << 2) | ((offsets & 128) >> 7)
                    | ((offsets & 1024) >> 5))
    expected = source[:, shuffled].sum(0, dtype=torch.float32).bfloat16()
    result = torch.empty((size, ), device="cuda", dtype=torch.bfloat16)
    _convert_outputs[(triton.cdiv(size, 4096), )](source, result, result, result, result, result, size, 0, 1.0, 4096,
                                                  PACKED=packed, DQ_SLICES=slices, num_warps=4)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("slices", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block_m", [64, 128, 256])
def test_attention_backward_zero_dq_slices_tail(slices, dtype, block_m):
    from triton.tlx.ops.kernels.flash_attn.gfx950_bwd import _run_bwd_zero_dq

    batch_heads, n_ctx, head_dim = 6, 70, 128
    elements = slices * batch_heads * n_ctx * head_dim
    storage = torch.full((elements + 128, ), -123.0, device="cuda", dtype=dtype)
    dq_acc = storage[:elements].view(slices, batch_heads, n_ctx, head_dim)
    dq_acc.fill_(float("nan"))
    _run_bwd_zero_dq(dq_acc, batch_heads, n_ctx, head_dim, block_m=block_m, dq_slices=slices)
    torch.testing.assert_close(dq_acc, torch.zeros_like(dq_acc), atol=0, rtol=0)
    torch.testing.assert_close(storage[elements:], torch.full_like(storage[elements:], -123.0), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
def test_attention_backward_split_preprocess_order(monkeypatch):
    from triton.tlx.ops.kernels.flash_attn import gfx1250_bwd as backward

    q = torch.empty((1, 4, 512, 128), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device="cuda", dtype=torch.float32)
    config = backward.select_config(512, 1, 4, 4, num_cus=192, packed_dq=False)
    assert config.q_splits > 1  # covers the dK/dV partial zeroing
    workspace = backward.allocate_workspace(q, q, config=config)
    calls = []

    def stats(*args, **kwargs):
        assert len(args) == 3 and "dq_acc" not in kwargs
        assert args[2] is workspace[0]
        calls.append("stats")

    def zero(dq_acc, batch_heads, n_ctx, head_dim, **kwargs):
        assert dq_acc is workspace[1]
        assert (batch_heads, n_ctx, head_dim) == (4, 512, 128)
        partials = workspace[5:7] if config.q_splits > 1 else (None, None)
        assert kwargs.pop("dk_acc") is partials[0] and kwargs.pop("dv_acc") is partials[1]
        assert kwargs == {"block_m": config.preprocess_block, "dq_slices": 1, "heads": 4, "kv_heads": 4}
        calls.append("zero")

    class RecordKernel:

        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                calls.append(self.name)

            return record

    monkeypatch.setattr(backward, "_run_bwd_preprocess", stats)
    monkeypatch.setattr(backward, "_run_bwd_zero_dq", zero)
    monkeypatch.setattr(backward, "_delayed_bwd_kernel", RecordKernel("main"))
    monkeypatch.setattr(backward, "_convert_outputs", RecordKernel("convert"))
    backward.fa_backward(q, q, q, q, q, lse, config=config, workspace=workspace)
    assert calls == ["stats", "zero", "main", "convert"]


def _attention_backward_occ1_config(packed_dq, *, q_splits=1):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import BackwardConfig

    return BackwardConfig(block_m=32, num_warps=4, stages=3, ordered=True, packed_dq=packed_dq, outer_m=64,
                          async_stats=True, queue_wmma=True, regalloc=True, q_splits=q_splits, waves_per_eu=1,
                          convert_block=8192 if q_splits > 1 and packed_dq else 4096, tile_major=True, cluster_dq=True,
                          occ1=True, occ1_k_bands=2, occ1_load_after_dk=True)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("dq_slices", [1, 2, 4])
@pytest.mark.parametrize("partials", [False, True])
@pytest.mark.parametrize("block", [1024, 2048, 4096, 8192])
def test_attention_backward_convert_vectorized(packed_dq, dq_slices, partials, block):
    import re
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _convert_outputs

    torch.manual_seed(2709)
    size, kv_size = 65536, 8192
    dtype = torch.bfloat16 if packed_dq else torch.float32
    dqa = torch.randn((dq_slices, size), device="cuda", dtype=dtype)
    dka, dva = [torch.randn((kv_size, ), device="cuda", dtype=torch.float32) for _ in range(2)]
    dq = torch.full((size, ), float("nan"), device="cuda", dtype=torch.bfloat16)
    dk, dv = [torch.full((kv_size, ), float("nan"), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    indices = torch.arange(size, device="cuda", dtype=torch.int64)
    shuffled = ((indices & ~(1024 | 16)) | ((indices & 1024) >> 6) | ((indices & 16) << 6))
    if packed_dq:
        shuffled = ((indices & ~(15 | 16 | 32 | 128 | 1024)) | ((indices & 15) << 1) | ((indices & 16) << 6) |
                    ((indices & 32) << 2) | ((indices & 128) >> 7) | ((indices & 1024) >> 5))
    expected = dqa[0, shuffled].float()
    for i in range(1, dq_slices):
        expected += dqa[i, shuffled].float()
    if not packed_dq:
        expected *= 128**-0.5
    args = (dqa, dq, dka, dva, dk, dv, size, kv_size, 128**-0.5, block)
    kwargs = dict(PACKED=packed_dq, PARTIALS=partials, DQ_SLICES=dq_slices, SCALE_DQ=not packed_dq, num_warps=4)
    kernel = _convert_outputs.warmup(*args, grid=(triton.cdiv(size, block), ), **kwargs)
    asm = kernel.asm["amdgcn"]
    assert int(re.search(r"\.private_segment_fixed_size:\s*(\d+)", asm)[1]) == 0
    if block >= 2048:
        loads = re.findall(r"(?:buffer|global)_load_\w+", asm)
        assert loads and all(op.endswith("b128") for op in loads)
    _convert_outputs[(triton.cdiv(size, block), )](*args, **kwargs)
    assert torch.equal(dq, expected.bfloat16())
    if partials:
        assert torch.equal(dk, (dka * 128**-0.5).bfloat16())
        assert torch.equal(dv, dva.bfloat16())
    else:
        assert torch.isnan(dk).all() and torch.isnan(dv).all()


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("shape,q_splits", [
    ((1, 4, 4, 512), 1),
    ((1, 8, 2, 512), 3),
    ((1, 4, 4, 8192), 1),
    # With three stages, the 3-way split has exactly 2/3/3 outers; the
    # 4-way split has exactly two. Exercise the guaranteed prologue loads.
    ((1, 4, 4, 512), 2),
    ((1, 4, 4, 512), 3),
    ((1, 4, 4, 512), 4),
    ((2, 4, 4, 512), 2),
    ((2, 4, 4, 512), 3),
    ((2, 4, 4, 512), 4),
    # GROUP=8 exercises the split seed across query heads and uneven 3-way tails.
    ((1, 64, 8, 512), 2),
    ((1, 64, 8, 512), 3),
    ((1, 64, 8, 512), 4),
    ((2, 64, 8, 1024), 2),
    ((2, 64, 8, 1024), 3),
    ((2, 64, 8, 1024), 4),
    ((8, 64, 8, 512), 2),
    ((8, 64, 8, 512), 3),
    ((8, 64, 8, 512), 4),
])
def test_attention_backward_occ1_correctness(shape, q_splits, packed_dq, monkeypatch):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base, allocate_workspace, fa_backward

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(2709)
    b, hq, hk, n = shape
    q, dout = [torch.randn((b, hq, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    k, v = [torch.randn((b, hk, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    qr, kr, vr = [t.float().requires_grad_() for t in (q, k, v)]
    scores = qr @ kr.repeat_interleave(hq // hk, 1).transpose(-1, -2) * 128**-0.5
    out = scores.softmax(-1) @ vr.repeat_interleave(hq // hk, 1)
    expected = torch.autograd.grad(out, (qr, kr, vr), dout.float())
    case = (q, k, v, out.detach().bfloat16(), dout, scores.detach().logsumexp(-1))
    config = _attention_backward_occ1_config(packed_dq, q_splits=q_splits)
    baseline = _select_config_base(n, b, hq, hk, num_cus=192, packed_dq=packed_dq)
    workspace = allocate_workspace(q, k, config=config)
    base_workspace = allocate_workspace(q, k, config=baseline)
    assert workspace[1].shape == ((2, ) + q.shape if n == 8192 else q.shape)
    if q_splits > 1:
        assert workspace[5].shape == ((q_splits, ) + k.shape if packed_dq else k.shape)
        assert workspace[6].shape == workspace[5].shape

    def nrmse(value, ref):
        return ((value.float() - ref.float()).square().mean() / ref.float().square().mean()).sqrt().item()

    production = []
    for call in range(5):
        actual = fa_backward(*case, config=baseline, workspace=base_workspace)
        production.append(tuple(value.clone() for value in actual))
    noise = max(nrmse(value[0], production[0][0]) for value in production[1:])
    bound = 2 * noise + 1e-3 if packed_dq else 1e-4
    for call in range(5):
        workspace[1].fill_(float("nan"))
        if q_splits > 1:
            for partial in workspace[5:7]:
                partial.fill_(float("nan"))
        actual = fa_backward(*case, config=config, workspace=workspace)
        assert all(torch.isfinite(value).all() for value in actual)
        if call < 3:
            assert all(nrmse(value, ref) < 0.012 for value, ref in zip(actual, expected))
        assert nrmse(actual[0], production[0][0]) <= bound
        if not packed_dq:
            assert all(nrmse(value, ref) <= 1e-4 for value, ref in zip(actual[1:], production[0][1:]))


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("shape,q_splits", [
    ((1, 8, 2, 512), 3),
    ((1, 8, 2, 4096), 1),
    ((1, 8, 2, 8192), 1),
    ((1, 4, 4, 512), 2),
    ((1, 4, 4, 512), 3),
    ((1, 4, 4, 512), 4),
    ((1, 64, 8, 512), 2),
    ((1, 64, 8, 512), 3),
    ((1, 64, 8, 512), 4),
])
def test_attention_backward_occ1_assembly(shape, q_splits, packed_dq):
    import re

    config = _attention_backward_occ1_config(packed_dq, q_splits=q_splits)
    kernel = _compile_attention_backward_kernel(config, shape)
    assembly = kernel.asm["amdgcn"]
    assert int(re.search(r"\.vgpr_count:\s*(\d+)", assembly)[1]) <= 1024
    assert int(re.search(r"\.private_segment_fixed_size:\s*(\d+)", assembly)[1]) == 0
    assert kernel.metadata.shared == 209392
    loop = _attention_backward_isa_loop(assembly)
    if q_splits > 1:
        # Guaranteed prologue TDM issues must keep the two-stage lookahead;
        # a skippable prologue path previously clamped this wait to 0x1.
        tensor_waits = [
            int(value, 0) for line in loop for value in re.findall(r"s_wait_tensorcnt\s+(0x[0-9a-f]+|\d+)", line)
        ]
        assert 8 in tensor_waits
        assert 1 not in tensor_waits
    wmma = sum("v_wmma_" in line for line in loop)
    lds = sum(bool(re.match(r"\s*ds_load", line)) for line in loop)
    assert wmma == 320
    assert lds / wmma <= 0.80
    # The phase barrier must wait only for LDS, never for outstanding dQ stores.
    assert not any("s_wait_storecnt" in line for line in loop)
    assert not any("s_wait_xcnt" in line for line in loop)
    atomics = [line for line in loop if "buffer_atomic_" in line]
    assert len(atomics) == (32 if packed_dq else 64)
    assert all("scope:SCOPE_SE" in line for line in atomics)
    # Both phases reuse one address VGPR each; register-owned deltas are immediates.
    addresses = [re.search(r",\s*(v\d+)(?:\s*/\*(v\d+)\*/)?\s*,\s*s\[", line) for line in atomics]
    assert all(addresses)
    assert len({match[2] or match[1] for match in addresses}) <= 2
    if q_splits > 1:
        epilogue = [
            line for line in assembly.splitlines() if "buffer_atomic_add_f32" in line and "scope:SCOPE_DEV" in line
        ]
        if packed_dq:
            assert not epilogue
            assert "buffer_store_b" in assembly or "global_store_b" in assembly
        else:
            # FP32 retains the committed shared-partial atomic epilogue.
            assert len(epilogue) == 512


def test_attention_backward_occ1_war_flags_default_off():
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import BackwardConfig

    config = BackwardConfig()
    assert not config.occ1_dq_first
    assert not config.occ1_load_after_dk


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("shape,expected_drains", [((16, 16, 16, 16384), 32), ((8, 64, 8, 8192), 31)])
@pytest.mark.parametrize("packed_dq", [False, True])
def test_attention_backward_occ1_war_drain_assembly(shape, expected_drains, packed_dq):
    import dataclasses
    import re

    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import select_config

    config = select_config(shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq)
    builds = [
        _compile_attention_backward_kernel(dataclasses.replace(config, occ1_load_after_dk=enabled), shape)
        for enabled in (False, True)
    ]
    drains = []
    for kernel in builds:
        assembly = kernel.asm["amdgcn"]
        assert int(re.search(r"\.vgpr_count:\s*(\d+)", assembly)[1]) <= 1024
        assert int(re.search(r"\.private_segment_fixed_size:\s*(\d+)", assembly)[1]) == 0
        assert kernel.metadata.shared == 209392
        loop = _attention_backward_isa_loop(assembly)
        drains.append(sum("s_wait_alu depctr_va_vdst(0)" in line for line in loop))
        assert not any("s_wait_xcnt" in line for line in loop)
        waits = [int(value, 0) for line in loop for value in re.findall(r"s_wait_tensorcnt\s+(0x[0-9a-f]+|\d+)", line)]
        assert 8 in waits
        assert 1 not in waits
    # Count instructions rather than matching source text or fixed line numbers.
    assert drains[0] == expected_drains
    assert drains[1] < drains[0]


@pytest.mark.parametrize("shape,packed_dq,occ1", [
    ((16, 16, 16, 1024), False, True),
    ((16, 16, 16, 1024), True, True),
    ((8, 64, 8, 1024), False, True),
    ((8, 64, 8, 1024), True, True),
    ((16, 64, 8, 1024), False, True),
    ((16, 64, 8, 1024), True, True),
    ((16, 16, 16, 2048), False, True),
    ((16, 16, 16, 2048), True, True),
    ((8, 64, 8, 2048), False, True),
    ((8, 64, 8, 2048), True, True),
    ((16, 64, 8, 2048), False, True),
    ((16, 64, 8, 2048), True, True),
    ((16, 16, 16, 4096), False, True),
    ((16, 16, 16, 4096), True, True),
    ((8, 64, 8, 4096), False, True),
    ((8, 64, 8, 4096), True, True),
    ((16, 64, 8, 4096), False, True),
    ((16, 64, 8, 4096), True, True),
    ((16, 16, 16, 8192), False, True),
    ((16, 16, 16, 8192), True, True),
    ((8, 64, 8, 8192), False, True),
    ((8, 64, 8, 8192), True, True),
    ((16, 64, 8, 8192), False, True),
    ((16, 64, 8, 8192), True, True),
    ((16, 16, 16, 16384), False, True),
    ((16, 16, 16, 16384), True, True),
    ((8, 64, 8, 16384), False, True),
    ((8, 64, 8, 16384), True, True),
    ((15, 64, 8, 16384), False, True),
    ((15, 64, 8, 16384), True, True),
])
def test_attention_backward_occ1_selection(shape, packed_dq, occ1):
    import dataclasses

    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base, select_config

    config = select_config(shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq)
    assert config.occ1 == occ1
    splits = 3 if shape[-1] == 1024 and shape[1] > shape[2] and tuple(shape) not in ((8, 64, 8, 1024),
                                                                                     (16, 64, 8, 1024)) else 1
    expected = _attention_backward_occ1_config(packed_dq, q_splits=splits) if occ1 else _select_config_base(
        shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq)
    if tuple(shape) + (packed_dq, ) in ((8, 64, 8, 2048, False), (8, 64, 8, 2048, True), (16, 64, 8, 2048, False),
                                        (16, 64, 8, 2048, True), (8, 64, 8, 4096, False), (8, 64, 8, 4096, True)):
        expected = dataclasses.replace(expected, tail_splits=3)
    measured = {(8, 64, 8, 1024, False): (1, 3, 4096), (8, 64, 8, 1024, True):
                (1, 3, 4096), (16, 64, 8, 1024, False): (1, 3, 4096), (16, 64, 8, 1024, True): (1, 3, 4096),
                (16, 16, 16, 1024, False): (1, 2, 4096), (16, 16, 16, 4096, False): (1, 3, 4096),
                (16, 16, 16, 4096, True): (1, 2, 4096), (16, 64, 8, 4096, False): (1, 3, 4096), (16, 64, 8, 4096, True):
                (1, 3, 4096), (8, 64, 8, 8192, False): (1, 3, 4096), (8, 64, 8, 8192, True): (1, 3, 4096),
                (16, 64, 8, 8192, False): (1, 3, 4096), (16, 64, 8, 8192, True): (1, 3, 4096), (8, 64, 8, 16384, False):
                (1, 3, 4096), (8, 64, 8, 16384, True): (1, 3, 4096)}
    policy = measured.get(tuple(shape) + (packed_dq, ))
    if policy is not None:
        expected = dataclasses.replace(expected, q_splits=policy[0], tail_splits=policy[1], convert_block=policy[2])
    assert config == expected


@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("shape,enabled", [
    ((8, 64, 8, 1024), True),
    ((16, 64, 8, 1024), True),
    ((8, 128, 16, 1024), True),
    ((16, 32, 4, 1024), True),
    ((1, 64, 8, 1024), False),
    ((2, 64, 8, 1024), False),
    ((4, 64, 8, 1024), False),
    ((9, 64, 8, 1024), False),
    ((8, 32, 8, 1024), False),
    ((8, 128, 8, 1024), False),
    ((8, 64, 8, 512), False),
])
def test_attention_backward_occ1_split_selection(shape, enabled, packed_dq):
    import dataclasses

    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base, select_config

    config = select_config(shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq)
    if enabled:
        if tuple(shape) in ((8, 64, 8, 1024), (16, 64, 8, 1024)):
            assert config == dataclasses.replace(_attention_backward_occ1_config(packed_dq), tail_splits=3)
            assert config.convert_block == 4096
        else:
            assert config == _attention_backward_occ1_config(packed_dq, q_splits=3)
            assert config.convert_block == (8192 if packed_dq else 4096)
    else:
        assert config == _select_config_base(shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq)
    # The same shape falls below four split rounds on the larger offline device.
    assert select_config(1024, 8, 64, 8, num_cus=384,
                         packed_dq=packed_dq) == _select_config_base(1024, 8, 64, 8, num_cus=384, packed_dq=packed_dq)


@pytest.mark.parametrize("packed_dq", [False, True])
def test_attention_backward_occ1_selection_fallback(packed_dq):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _select_config_base, select_config

    for n in (512, 1024, 2048, 4096, 8192, 16384):
        assert select_config(n, packed_dq=packed_dq) == _select_config_base(n, packed_dq=packed_dq)
    for shape in ((1, 8, 8, 4096), (8, 64, 8, 1152), (16, 16, 16, 1536), (1, 64, 8, 2048), (2, 64, 8, 4096),
                  (1, 8, 1, 8192), (1, 2, 2, 16384)):
        args = (shape[-1], *shape[:3])
        assert select_config(*args, num_cus=192,
                             packed_dq=packed_dq) == _select_config_base(*args, num_cus=192, packed_dq=packed_dq)
    # One-split short MHA matches the base grid; 512 owners clear the split-fill guard.
    for shape in ((2, 16, 16, 2048), (1, 64, 8, 16384)):
        assert select_config(shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq).occ1


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("kv_slices", [2, 3, 4])
def test_attention_backward_convert_split_stores(packed_dq, kv_slices):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _convert_outputs

    torch.manual_seed(2709)
    dqa = torch.zeros(2048, device="cuda", dtype=torch.bfloat16 if packed_dq else torch.float32)
    dq = torch.full((2048, ), float("nan"), device="cuda", dtype=torch.bfloat16)
    dka, dva = [torch.randn((kv_slices, 2048), device="cuda", dtype=torch.float32) for _ in range(2)]
    dk, dv = [torch.full((2048, ), float("nan"), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    _convert_outputs[(1, )](dqa, dq, dka, dva, dk, dv, 2048, 2048, 128**-0.5, 2048, PACKED=packed_dq, PARTIALS=True,
                            KV_SLICES=kv_slices, num_warps=4)
    expected = []
    for partial in (dka, dva):
        value = partial[0].clone()
        for i in range(1, kv_slices):
            value = value + partial[i]
        expected.append(value)
    assert torch.equal(dk, (expected[0] * 128**-0.5).bfloat16())
    assert torch.equal(dv, expected[1].bfloat16())
    assert torch.count_nonzero(dq).item() == 0


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("block,size,kv_size,slices,packed_dq", [
    pytest.param(block, group * 8192, 8192, slices, packed,
                 id=f"b{block}-g{group}-s{slices}-{'packed' if packed else 'fp32'}")
    for block in (1024, 2048, 4096, 8192)
    for group in (1, 3, 8)
    for slices in (1, 2, 3, 4)
    for packed in (False, True)
] + [
    pytest.param(8192, size, kv_size, slices, packed, id=f"tail{size}-s{slices}-{'packed' if packed else 'fp32'}")
    for size, kv_size in ((24960, 8320), (65792, 8224))
    for slices in (1, 2, 3, 4)
    for packed in (False, True)
])
def test_attention_backward_convert_balanced_partials(block, size, kv_size, slices, packed_dq):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _convert_outputs

    torch.manual_seed(2709)
    # Padding belongs only to the unchanged dQ permutation, not to KV slices.
    padded = triton.cdiv(size, 2048) * 2048
    dtype = torch.bfloat16 if packed_dq else torch.float32
    dqa = torch.randn((padded, ), device="cuda", dtype=dtype)
    dka, dva = [torch.randn((slices, kv_size), device="cuda", dtype=torch.float32) for _ in range(2)]
    dq = torch.full((size, ), float("nan"), device="cuda", dtype=torch.bfloat16)
    dk, dv = [torch.full((kv_size, ), float("nan"), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    indices = torch.arange(size, device="cuda", dtype=torch.int64)
    shuffled = ((indices & ~(1024 | 16)) | ((indices & 1024) >> 6) | ((indices & 16) << 6))
    if packed_dq:
        shuffled = ((indices & ~(15 | 16 | 32 | 128 | 1024)) | ((indices & 15) << 1) | ((indices & 16) << 6) |
                    ((indices & 32) << 2) | ((indices & 128) >> 7) | ((indices & 1024) >> 5))
    expected_dq = dqa[shuffled].float()
    if not packed_dq:
        expected_dq *= 128**-0.5
    expected_dk, expected_dv = dka[0].clone(), dva[0].clone()
    for i in range(1, slices):
        expected_dk += dka[i]
        expected_dv += dva[i]
    _convert_outputs[(triton.cdiv(size, block), )](dqa, dq, dka, dva, dk, dv, size, kv_size, 128**-0.5, block,
                                                   PACKED=packed_dq, PARTIALS=True, SCALE_DQ=not packed_dq,
                                                   KV_SLICES=slices, num_warps=4)
    assert torch.isfinite(dq).all() and torch.isfinite(dk).all() and torch.isfinite(dv).all()
    assert torch.equal(dq, expected_dq.bfloat16())
    assert torch.equal(dk, (expected_dk * 128**-0.5).bfloat16())
    assert torch.equal(dv, expected_dv.bfloat16())


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("shape", [(16, 16, 16, 1024), (8, 64, 8, 1024), (16, 64, 8, 1024), (8, 64, 8, 2048),
                                   (16, 64, 8, 2048), (8, 64, 8, 4096), (1, 4, 4, 512), (2, 4, 4, 512), (1, 8, 1, 512),
                                   (2, 8, 1, 512)])
def test_attention_backward_occ1_tail_assembly(shape, packed_dq):
    import dataclasses
    import re

    config = dataclasses.replace(_attention_backward_occ1_config(packed_dq), tail_splits=3)
    kernel = _compile_attention_backward_kernel(config, shape)
    assembly = kernel.asm["amdgcn"]
    assert int(re.search(r"\.vgpr_count:\s*(\d+)", assembly)[1]) <= 1024
    assert int(re.search(r"\.private_segment_fixed_size:\s*(\d+)", assembly)[1]) == 0
    assert kernel.metadata.shared == 209392
    loop = _attention_backward_isa_loop(assembly)
    waits = [int(x, 0) for line in loop for x in re.findall(r"s_wait_tensorcnt\s+(0x[0-9a-f]+|\d+)", line)]
    assert 8 in waits and 1 not in waits
    assert sum("v_wmma_" in line for line in loop) == 320
    assert all("scope:SCOPE_SE" in line for line in loop if "buffer_atomic_" in line)
    assert not any("SCOPE_DEV" in line and "atomic_" in line for line in assembly.splitlines())


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("kv_slices", [2, 3, 4])
def test_attention_backward_convert_compact_tail(packed_dq, kv_slices):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _convert_outputs

    torch.manual_seed(2709)
    size, tail_size, prefix, block = 16384, 1024, 3072, 4096
    dtype = torch.bfloat16 if packed_dq else torch.float32
    dqa = torch.zeros(size, device="cuda", dtype=dtype)
    dq = torch.full((size, ), float("nan"), device="cuda", dtype=torch.bfloat16)
    dka, dva = [torch.randn((kv_slices, tail_size), device="cuda", dtype=torch.float32) for _ in range(2)]
    dk, dv = [torch.full((prefix + tail_size, ), 17, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    _convert_outputs[(triton.cdiv(size, block), )](dqa, dq, dka, dva, dk[prefix:], dv[prefix:], size, tail_size,
                                                   128**-0.5, block, PACKED=packed_dq, PARTIALS=True,
                                                   SCALE_DQ=not packed_dq, KV_SLICES=kv_slices, num_warps=4)
    values = []
    for partial in (dka, dva):
        value = partial[0].clone()
        for i in range(1, kv_slices):
            value += partial[i]
        values.append(value)
    assert torch.count_nonzero(dq).item() == 0
    assert torch.all(dk[:prefix] == 17) and torch.all(dv[:prefix] == 17)
    assert torch.equal(dk[prefix:], (values[0] * 128**-0.5).bfloat16())
    assert torch.equal(dv[prefix:], values[1].bfloat16())


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("shape", [(1, 4, 4, 512), (2, 4, 4, 512), (1, 8, 1, 512), (2, 8, 1, 512), (13, 16, 16, 512),
                                   (13, 64, 8, 512), (16, 64, 8, 512)])
def test_attention_backward_occ1_tail_correctness(shape, packed_dq, monkeypatch):
    import dataclasses

    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _tail_partition, allocate_workspace, fa_backward

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(2709)
    b, hq, hk, n = shape
    q, dout = [torch.randn((b, hq, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    k, v = [torch.randn((b, hk, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    qr, kr, vr = [x.float().requires_grad_() for x in (q, k, v)]
    scores = qr @ kr.repeat_interleave(hq // hk, 1).transpose(-1, -2) * 128**-0.5
    out = scores.softmax(-1) @ vr.repeat_interleave(hq // hk, 1)
    expected = torch.autograd.grad(out, (qr, kr, vr), dout.float())
    case = (q, k, v, out.detach().bfloat16(), dout, scores.detach().logsumexp(-1))
    base = _attention_backward_occ1_config(packed_dq)
    config = dataclasses.replace(base, tail_splits=3)
    prefix, tail = _tail_partition(config, b, hk, n)
    hw, cw = [allocate_workspace(q, k, config=c) for c in (base, config)]
    assert cw[5].shape == (3, tail, 256, 128)

    def error(a, e):
        return ((a.float() - e.float()).square().sum() / e.float().square().sum()).sqrt().item()

    first, noise = None, 0
    for i in range(5):
        actual = fa_backward(*case, config=base, workspace=hw)
        if first is None:
            first = tuple(x.clone() for x in actual)
        noise = max(noise, error(actual[0], first[0]))
    base_errors = [error(x, e) for x, e in zip(first, expected)]
    repeat = None
    for i in range(5):
        for x in (cw[1], *cw[2:7]):
            x.fill_(float("nan"))
        actual = fa_backward(*case, config=config, workspace=cw)
        assert all(torch.isfinite(x).all() for x in actual)
        assert error(actual[0], first[0]) <= (2 * noise + 1e-3 if packed_dq else 1e-4)
        for j in (1, 2):
            assert error(actual[j], expected[j]) < 0.012
            assert error(actual[j], expected[j]) <= 1.1 * base_errors[j]
            assert error(actual[j], first[j]) <= 5e-3
            if prefix:
                assert torch.equal(actual[j].reshape(-1)[:prefix * 256 * 128],
                                   first[j].reshape(-1)[:prefix * 256 * 128])
        if repeat is None:
            repeat = tuple(x.clone() for x in actual[1:])
        if packed_dq:
            assert all(torch.equal(x, e) for x, e in zip(actual[1:], repeat))


@pytest.mark.parametrize("shape,packed_dq", [((8, 64, 8, 2048), False), ((8, 64, 8, 2048), True),
                                             ((16, 64, 8, 2048), False), ((16, 64, 8, 2048), True),
                                             ((8, 64, 8, 4096), False), ((8, 64, 8, 4096), True)])
def test_attention_backward_occ1_tail_selection(shape, packed_dq):
    import dataclasses

    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import select_config

    config = select_config(shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq)
    assert config == dataclasses.replace(_attention_backward_occ1_config(packed_dq), tail_splits=3)
    assert config.q_splits == 1 and config.convert_block == 4096
    assert select_config(shape[-1], *shape[:3], num_cus=384, packed_dq=packed_dq).tail_splits == 1


@pytest.mark.parametrize("b,hk,n", [(3, 16, 1024), (6, 8, 2048), (3, 8, 4096)])
def test_attention_backward_occ1_tail_exact_round_rejected(b, hk, n):
    import dataclasses

    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import _tail_partition

    config = dataclasses.replace(_attention_backward_occ1_config(False), tail_splits=3)
    with pytest.raises(ValueError, match="nonempty dispatch tail"):
        _tail_partition(config, b, hk, n)


@pytest.mark.parametrize("shape,packed_dq,q_splits,tail_splits,convert_block", [((16, 16, 16, 2048), False, 1, 1, 4096),
                                                                                ((16, 16, 16, 2048), True, 1, 1, 4096),
                                                                                ((16, 16, 16, 4096), False, 1, 3, 4096),
                                                                                ((16, 16, 16, 4096), True, 1, 2, 4096),
                                                                                ((16, 64, 8, 4096), False, 1, 3, 4096),
                                                                                ((16, 64, 8, 4096), True, 1, 3, 4096),
                                                                                ((8, 64, 8, 8192), False, 1, 3, 4096),
                                                                                ((8, 64, 8, 8192), True, 1, 3, 4096),
                                                                                ((16, 64, 8, 8192), False, 1, 3, 4096),
                                                                                ((16, 64, 8, 8192), True, 1, 3, 4096),
                                                                                ((8, 64, 8, 16384), False, 1, 3, 4096),
                                                                                ((8, 64, 8, 16384), True, 1, 3, 4096)])
def test_attention_backward_occ1_dispatch_tail_selection(shape, packed_dq, q_splits, tail_splits, convert_block):
    import dataclasses

    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import select_config

    config = select_config(shape[-1], *shape[:3], num_cus=192, packed_dq=packed_dq)
    expected = dataclasses.replace(_attention_backward_occ1_config(packed_dq, q_splits=q_splits),
                                   tail_splits=tail_splits, convert_block=convert_block)
    assert config == expected
    # Each additional policy is measured on exactly the explicit 192-CU tuple.
    other = select_config(shape[-1], *shape[:3], num_cus=384, packed_dq=packed_dq)
    assert other.q_splits == 1 and other.tail_splits == 1
    for batch in (shape[0] - 1, shape[0] + 1):
        other = select_config(shape[-1], batch, shape[1], shape[2], num_cus=192, packed_dq=packed_dq)
        assert other.q_splits == 1 and other.tail_splits == 1
