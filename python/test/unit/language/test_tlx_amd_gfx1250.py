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


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("DTYPE_B", ["float8_e4m3", "float4"])
@pytest.mark.parametrize("TILES_PER_WARP", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_gfx1250_mxgemm_tdm_tiles_per_warp(DTYPE_B, TILES_PER_WARP):
    M = N = 256
    K = 512
    torch.manual_seed(7)
    a = _gfx1250_mxfp._init_data("float8_e4m3", M, K)
    b = _gfx1250_mxfp._init_data(DTYPE_B, K, N)
    a_scale = MXScaleTensor(size=(M, K // 32)).random(high=32.0).data
    b_scale = MXScaleTensor(size=(N, K // 32)).random(high=32.0).data
    ref = _gfx1250_mxfp.torch_gemm_mxfp(a, b, a_scale, b_scale, 32, M, N, K)

    b_data = b.to_packed_tensor(dim=0).data if DTYPE_B == "float4" else b
    a_d = a.contiguous().cuda()
    b_d = b_data.T.contiguous().cuda()
    a_scale_d = _gfx1250_mxfp.pack_scale(a_scale).cuda()
    b_scale_d = _gfx1250_mxfp.pack_scale(b_scale).cuda()
    out = _gfx1250_mxfp.mxgemm_tdm_pipelined(a_d, b_d, a_scale_d, b_scale_d, BLOCK_M=256, BLOCK_N=256, BLOCK_K=256,
                                             TRANSPOSE_B=True, NUM_BUFFERS=2, DTYPE_A="e4m3",
                                             DTYPE_B=_gfx1250_mxfp.DTYPE_TO_TRITON[DTYPE_B], SCHEDULE="sliceMNK",
                                             TDM_FUSION="partial", TILES_PER_WARP=TILES_PER_WARP)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=2e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("BATCH,H,SEQLEN_Q,SEQLEN_K", [(1, 8, 1024, 1024), (2, 4, 1024, 1024), (1, 16, 2048, 2048),
                                                       (1, 2, 896, 896), (1, 1, 640, 640),
                                                       (1, 2, 63, 65),  # one partial key tile
                                                       (1, 2, 128, 128),  # one full key tile
                                                       (1, 2, 129, 129),  # two key tiles with partial Q/K boundaries
                                                       (2, 2, 129, 257),  # partial Q and K tiles, independent lengths
                                                       (1, 2, 897, 897),  # final K tile contains a single valid row
                                                       (1, 2, 90000, 90000),  # long sequence with a partial final tile
                                                       ])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gfx1250_attn_fwd_tdm_pipelined(BATCH, H, SEQLEN_Q, SEQLEN_K, dtype):
    torch.manual_seed(0)
    D = 128
    q = torch.randn((BATCH, H, SEQLEN_Q, D), device=triton.runtime.driver.active.get_active_torch_device(), dtype=dtype)
    k = torch.randn((BATCH, H, SEQLEN_K, D), device=triton.runtime.driver.active.get_active_torch_device(), dtype=dtype)
    v = torch.randn_like(k)
    sm_scale = 1.0 / (D**0.5)
    out = _gfx1250_attention.attn_fwd_tdm_pipelined(q, k, v, sm_scale)
    assert torch.isfinite(out).all()
    # Avoid a quadratic reference allocation for long attention, but check every
    # output element for finiteness and sample both sides of tile boundaries.
    if SEQLEN_Q > 2048:
        boundaries = [0, 127, 128, 255, 256, 65535, 65536, SEQLEN_Q - 2, SEQLEN_Q - 1]
        rows = torch.tensor(sorted({row for row in boundaries if 0 <= row < SEQLEN_Q}), device=q.device)
        q_ref = q.index_select(2, rows)
        actual = out.index_select(2, rows)
    else:
        q_ref, actual = q, out
    scores = (q_ref.double() @ k.double().transpose(-1, -2)) * sm_scale
    ref = scores.softmax(dim=-1) @ v.double()
    atol, rtol = (2e-2, 1e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)
    torch.testing.assert_close(actual.double(), ref, atol=atol, rtol=rtol)
    # A loose absolute bound alone misses large relative errors at long lengths.
    tolerance = 0.005 if dtype == torch.bfloat16 else 0.0008
    assert (actual.double() - ref).square().mean().sqrt() < tolerance * ref.square().mean().sqrt()


# Automatic narrow/medium routes plus a forced wide tile exercise every replay
# geometry without allocating an unusually large adversarial attention input.
@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seqlen_k,block_m,block_n", [(512, None, None), (1025, None, None), (1025, 256, 64)])
@pytest.mark.parametrize("recovery_case", ["exp_overflow", "probability_overflow", "common_offset"])
def test_gfx1250_attn_fwd_tdm_pipelined_recovery(dtype, seqlen_k, block_m, block_n, recovery_case):
    torch.manual_seed(1)
    device = triton.runtime.driver.active.get_active_torch_device()
    q = torch.randn((1, 2, 257, 128), device=device, dtype=dtype)
    k = torch.randn((1, 2, seqlen_k, 128), device=device, dtype=dtype)
    v = torch.randn_like(k).abs().add_(0.25)
    scale = 128**-0.5
    _gfx1250_attention.attn_fwd_tdm_pipelined(q, k, v, scale, BLOCK_M=block_m, BLOCK_N=block_n)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _gfx1250_attention.attn_fwd_tdm_pipelined(q, k, v, scale, BLOCK_M=block_m, BLOCK_N=block_n)

    # Reuse one graph across ordinary -> unsafe -> ordinary inputs. The
    # numerical decision must be recomputed on the device for every replay.
    for unsafe in (False, True, False):
        if not unsafe:
            q.normal_()
            k.normal_()
        elif recovery_case == "exp_overflow":
            q.fill_(1)
            k.zero_()
            k[:, :, 128:].fill_(16)
        elif recovery_case == "probability_overflow":
            q.zero_()
            q[..., 0] = 1
            k.zero_()
            k[:, :, 128:, 0] = 200
        else:
            q.zero_()
            q[..., 0] = 256
            q[..., 1] = 1
            k.zero_()
            k[..., 0] = 256
            k[..., 1] = torch.arange(k.shape[2], device=device) % 9 - 4
        ref = ((q.double() @ k.double().transpose(-1, -2)) * scale).softmax(-1) @ v.double()
        for _ in range(3):
            graph.replay()
            assert torch.isfinite(out).all()
            tolerance = 0.005 if dtype == torch.bfloat16 else 0.0008
            torch.testing.assert_close(out.double(), ref, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["offset", "padded", "odd_stride"])
@pytest.mark.parametrize("block_m,block_n", [(64, 128), (128, 64), (256, 64)])
def test_gfx1250_attn_fwd_tdm_pipelined_strides(dtype, layout, block_m, block_n):
    torch.manual_seed(2)
    device = triton.runtime.driver.active.get_active_torch_device()
    batch, heads, sq, sk, dim = 2, 2, 257, 1025, 128

    def inputs(length):
        shape = (batch, heads, length, dim)
        if layout == "offset":
            tensor = torch.empty(batch * heads * length * dim + 1, dtype=dtype, device=device)[1:].view(shape)
        else:
            padding = 1 if layout == "odd_stride" else 16
            tensor = torch.empty((*shape[:-1], dim + padding), dtype=dtype, device=device)[..., :dim]
        return tensor.normal_()

    q, k, v = inputs(sq), inputs(sk), inputs(sk)
    scale = dim**-0.5
    ref = ((q.double() @ k.double().transpose(-1, -2)) * scale).softmax(-1) @ v.double()
    # A partial query tile must preserve storage on both sides of the output.
    storage = torch.full((q.numel() + 512, ), -12345.0, device=device)
    out = storage[256:-256].view(q.shape)
    grid = (batch, heads, triton.cdiv(sq, block_m))

    def launch():
        _gfx1250_attention.attn_fwd_tdm_pipelined_kernel[grid](q, k, v, out, *q.stride(), *k.stride(), *v.stride(),
                                                               *out.stride(), scale, sq, sk, block_m, block_n, dim,
                                                               USE_FIXED_REFERENCE=True, num_warps=4, waves_per_eu=1)

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(5):
        out.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(out).all()
        tolerance = 0.005 if dtype == torch.bfloat16 else 0.0008
        torch.testing.assert_close(out.double(), ref, atol=tolerance, rtol=tolerance)
        assert (out.double() - ref).square().mean().sqrt() < tolerance * ref.square().mean().sqrt()
        assert (storage[:256] == -12345.0).all()
        assert (storage[-256:] == -12345.0).all()


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
