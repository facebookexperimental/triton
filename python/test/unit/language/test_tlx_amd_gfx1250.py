"""TLX AMD tests -- gfx1250."""
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.tutorials import amd_tdm_gemm_pipelined as _gfx1250_gemm
from triton.language.extra.tlx.tutorials.amd_mxfp_gemm_gfx1250 import amd_mxfp_gemm_tdm_pipelined as _gfx1250_mxfp
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
@pytest.mark.parametrize("row_offset,col_offset", [(-32, 0), (0, -32), (0, 64), (96, 96)])
@pytest.mark.parametrize("clamp,set_bounds,pred", [(False, False, True), (True, False, True), (False, True, True),
                                                   (False, False, False)])
def test_tdm_fused_positioned_offsets(device, row_offset, col_offset, clamp, set_bounds, pred):
    a = torch.randn((192, 256), device=device, dtype=torch.float16)
    b = torch.randn_like(a)
    output = torch.empty((2, 64, 64), device=device, dtype=torch.float16)
    _tdm_fused_positioned_kernel[(1, )](a, b, output, row_offset, col_offset, pred, clamp, set_bounds)
    for index, source in enumerate((a, b)):
        # Address the backing allocation directly: an unclamped negative
        # column offset reads the preceding row's padding before column zero.
        rows = 32 + row_offset + torch.arange(64, device=device)
        cols = col_offset + torch.arange(64, device=device)
        expected = source.flatten()[rows[:, None] * 256 + cols[None, :]]
        if set_bounds:
            expected[16:, :] = 0
            expected[:, 32:] = 0
        elif clamp:
            rows = 0 if row_offset < 0 else min(64, max(0, 128 - row_offset))
            cols = 0 if col_offset < 0 else min(64, max(0, 128 - col_offset))
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
        (384, 384, 1280, "sliceMNK", "float8_e4m3", "float4", 256, 256, 256, 2, True, True, "partial", -1, True),
        (384, 384, 768, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 256, 2, True, True, "partial", -1, True),
        (256, 512, 1024, "sliceMNK", "float8_e4m3", "float4", 128, 256, 256, 3, True, True, "partial", -1, True),
        (384, 384, 640, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 128, 3, True, True, "partial", -1, True),
        (384, 384, 640, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 128, 4, True, True, "partial", -1, True),
        (384, 384, 640, "sliceMNK", "float8_e4m3", "float4", 256, 256, 128, 4, True, True, "partial", -1, True),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float8_e4m3", 256, 256, 128, 4, True, True, "partial", -1, False),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float4", 256, 256, 128, 3, False, False, "none", -1, True),
        (256, 256, 512, "sliceMNK", "float4", "float4", 256, 256, 128, 2, True, True, "partial", -1, True),
        (384, 384, 640, "sliceMNK", "float8_e4m3", "float8_e4m3", 128, 256, 128, 4, True, True, "partial", -1, True),
        (256, 256, 512, "sliceMNK", "float8_e4m3", "float4", 256, 256, 256, 2, False, False, "none", -1, True),
        (256, 256, 512, "sliceMNK", "float4", "float4", 256, 256, 256, 2, True, True, "partial", -1, True),
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


def _mxgemm_ring_phase_params():
    params = []
    for k, buffers in [(2, 2), (3, 2), (3, 3), (4, 3), (5, 3), (4, 4), (5, 4), (6, 4), (7, 4)]:
        for prefetch in (False, True):
            params.append(
                pytest.param(k, buffers, prefetch, "float8_e4m3", 128, 1, id=f"a8w8-k{k}-b{buffers}-p{prefetch}"))
    for k in (3, 4, 5, 6):
        for prefetch in (False, True):
            params.append(pytest.param(k, 3, prefetch, "float4", 256, 1, id=f"a8w4-k{k}-p{prefetch}"))
    for k in (2, 3):
        params.append(pytest.param(k, 2, False, "float4", 256, 1, id=f"a8w4-k{k}-b2"))
        params.append(pytest.param(k, 2, False, "float8_e4m3", 256, 1, id=f"a8w8-bk256-k{k}-b2"))
    for k in (3, 4, 5, 6, 9):
        params.append(pytest.param(k, 3, True, "float8_e4m3", 128, 4, id=f"a8w8-cluster-k{k}"))
    for k in (3, 6, 9):
        for prefetch in (False, True):
            params.append(pytest.param(k, 3, prefetch, "float4", 256, 4, id=f"a8w4-cluster-k{k}-p{prefetch}"))
    return params


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("K_ITERS,NUM_BUFFERS,CROSS_TILE_PREFETCH,DTYPE_B,BLOCK_K,CLUSTER_SIZE",
                         _mxgemm_ring_phase_params())
def test_mxgemm_persistent_ring_phase(K_ITERS, NUM_BUFFERS, CROSS_TILE_PREFETCH, DTYPE_B, BLOCK_K, CLUSTER_SIZE,
                                      CLUSTER_LAYOUT="linear", SPLIT_CLUSTER_BARRIER=False, EARLY_C_STORES=False,
                                      CLUSTER_BARRIER_INTERVAL=4, STATIC_SLOT_UNROLL=False, OUTPUT_C_SLOTS=None,
                                      OVERLAP_PEEL=False, OUT_DTYPE="fp32"):
    # Ten tiles over three programs exercise uneven tile counts, changes in
    # both M and N, the zero-length steady loop, and input/output slot reuse
    # across ring wrap. Cluster cases give each program two tiles; K5/K9 also
    # exercise periodic cluster synchronization in the steady loop and across
    # the tile transition.
    M, N = (2048, 1024) if CLUSTER_SIZE > 1 else (1280, 512)
    programs = 16 if CLUSTER_SIZE > 1 else 3
    K = K_ITERS * BLOCK_K
    torch.manual_seed(123)
    a = (torch.randn((M, K)) * 0.5).to(torch.float8_e4m3fn)
    b = _gfx1250_mxfp._init_data(DTYPE_B, K, N)
    a_scale = torch.randint(125, 130, (M, K // 32), dtype=torch.uint8)
    b_scale = torch.randint(125, 130, (N, K // 32), dtype=torch.uint8)
    ref = _gfx1250_mxfp.torch_gemm_mxfp(a, b, a_scale, b_scale, 32, M, N, K)
    b_data = b.to_packed_tensor(dim=0) if DTYPE_B == "float4" else b
    out = _gfx1250_mxfp.matmul(
        a.cuda(),
        b_data.T.contiguous().cuda(),
        _gfx1250_mxfp.pack_scale(a_scale).cuda(),
        _gfx1250_mxfp.pack_scale(b_scale).cuda(),
        config=dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=BLOCK_K, NUM_BUFFERS=NUM_BUFFERS, DTYPE_A="e4m3",
                    DTYPE_B=_gfx1250_mxfp.DTYPE_TO_TRITON[DTYPE_B], TRANSPOSE_B=True, SCHEDULE="sliceMNK",
                    TDM_FUSION="partial", PERSISTENT=True, NUM_PROGRAMS=programs,
                    CROSS_TILE_PREFETCH=CROSS_TILE_PREFETCH, CLUSTER_SIZE=CLUSTER_SIZE,
                    CLUSTER_BARRIER_INTERVAL=CLUSTER_BARRIER_INTERVAL, CLUSTER_LAYOUT=CLUSTER_LAYOUT,
                    SPLIT_CLUSTER_BARRIER=SPLIT_CLUSTER_BARRIER, EARLY_C_STORES=EARLY_C_STORES,
                    STATIC_SLOT_UNROLL=STATIC_SLOT_UNROLL, OUTPUT_C_SLOTS=OUTPUT_C_SLOTS, OVERLAP_PEEL=OVERLAP_PEEL,
                    OUT_DTYPE=OUT_DTYPE))
    if OUT_DTYPE == "fp32":
        torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=1e-4)
    else:
        # FP32 accumulation rounded once to FP16 (relative step 2**-11).
        assert out.dtype == torch.float16
        torch.testing.assert_close(out.cpu().float(), ref, atol=2e-3, rtol=1e-3)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("k_iters", [3, 4, 5, 6, 9, 10])
@pytest.mark.parametrize("cluster_size", [1, 4])
def test_mxgemm_static_slot_unroll_ring_phase(k_iters, cluster_size):
    # Cover an empty steady loop, zero/one/two steps after the ring-depth
    # unroll, and every starting slot across uneven tile assignments.
    test_mxgemm_persistent_ring_phase(k_iters, 3, True, "float8_e4m3", 128, cluster_size,
                                      CLUSTER_LAYOUT="2x2" if cluster_size == 4 else "linear", STATIC_SLOT_UNROLL=True)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("k_iters", [3, 4, 5, 9])
@pytest.mark.parametrize("cluster_size", [1, 4])
@pytest.mark.parametrize("static_slot_unroll", [False, True])
def test_mxgemm_output_c_slots_ring_phase(k_iters, cluster_size, static_slot_unroll):
    # Three dedicated C slots rotate across quadrant boundaries and leave three
    # stores pending into the next tile, at every ring phase and both schedules.
    test_mxgemm_persistent_ring_phase(k_iters, 3, True, "float8_e4m3", 128, cluster_size,
                                      CLUSTER_LAYOUT="2x2" if cluster_size == 4 else "linear",
                                      STATIC_SLOT_UNROLL=static_slot_unroll, OUTPUT_C_SLOTS=3)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("schedule,k_iters", [(schedule, k)
                                              for schedule in ("static", "dynamic")
                                              for k in (4, 5, 6, 9)] + [("4buf", k) for k in (5, 6, 7, 10)])
@pytest.mark.parametrize("cluster_size", [1, 4])
def test_mxgemm_overlap_peel_ring_phase(schedule, k_iters, cluster_size):
    # Each next tile's first K step runs in the previous C epilogue, and its
    # first ring refill precedes the last C chunks. Cover every starting slot,
    # uneven tile counts, the last tile's plain stores, short steady loops, and
    # both C staging shapes: three 64x128 slots, or two 64x64 slots with four
    # buffers (where the refill splits a quadrant's stores).
    four = schedule == "4buf"
    test_mxgemm_persistent_ring_phase(k_iters, 4 if four else 3, True, "float8_e4m3", 128, cluster_size,
                                      CLUSTER_LAYOUT="2x2" if cluster_size == 4 else "linear",
                                      STATIC_SLOT_UNROLL=schedule == "static", OUTPUT_C_SLOTS=None if four else 3,
                                      OVERLAP_PEEL=True)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("schedule,slots,k_iters",
                         [("static", 6, k)
                          for k in (4, 5, 9)] + [("static", 3, 5)] + [("dynamic", 6, k)
                                                                      for k in (4, 5, 6)] + [("4buf", None, k)
                                                                                             for k in (5, 7)])
@pytest.mark.parametrize("cluster_size", [1, 4])
def test_mxgemm_fp16_output_ring_phase(schedule, slots, k_iters, cluster_size):
    # FP16 C halves the dedicated staging slots. Cover the overlapped epilogue
    # with six or three 64x128 slots on static and dynamic K loops, and two
    # 64x128 slots with four buffers, across ring phases, uneven tile counts,
    # and multicast clusters.
    four = schedule == "4buf"
    test_mxgemm_persistent_ring_phase(k_iters, 4 if four else 3, True, "float8_e4m3", 128, cluster_size,
                                      CLUSTER_LAYOUT="2x2" if cluster_size == 4 else "linear",
                                      STATIC_SLOT_UNROLL=schedule == "static", OUTPUT_C_SLOTS=slots, OVERLAP_PEEL=True,
                                      OUT_DTYPE="fp16")


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("schedule", ["a8w4-bench", "a8w4-prefetch", "a8w4-prefetch-cluster", "a8w4-two-buffers"])
def test_mxgemm_fp16_output_a8w4_ring_phase(schedule):
    # BK256 C aliases A input slots; FP16 slots hold twice the rows at the
    # same physical pitch.
    k_iters, buffers, prefetch, cluster, extra = {
        "a8w4-bench": (5, 3, False, 4, dict(CLUSTER_LAYOUT="2x2", SPLIT_CLUSTER_BARRIER=True, EARLY_C_STORES=True)),
        "a8w4-prefetch": (4, 3, True, 1, {}),
        "a8w4-prefetch-cluster": (9, 3, True, 4, dict(CLUSTER_LAYOUT="2x2")),
        "a8w4-two-buffers": (3, 2, False, 1, {}),
    }[schedule]
    test_mxgemm_persistent_ring_phase(k_iters, buffers, prefetch, "float4", 256, cluster, OUT_DTYPE="fp16", **extra)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
def test_mxgemm_fp16_output_nonpersistent():
    # The nonpersistent kernel converts before its direct store.
    M, N, K = (512, 512, 1024)
    torch.manual_seed(123)
    a = (torch.randn((M, K)) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn((K, N)) * 0.5).to(torch.float8_e4m3fn)
    sa = torch.randint(125, 130, (M, K // 32), dtype=torch.uint8)
    sb = torch.randint(125, 130, (N, K // 32), dtype=torch.uint8)
    expected = _gfx1250_mxfp.torch_gemm_mxfp(a, b, sa, sb, 32, M, N, K)
    config = dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, NUM_BUFFERS=2, num_warps=4, DTYPE_A="e4m3", DTYPE_B="e4m3",
                  TRANSPOSE_B=True, SCALE_PRESHUFFLE=True, WITH_A_SCALE=True, SCHEDULE="sliceMNK", TDM_FUSION="partial",
                  OUT_DTYPE="fp16")
    result = _gfx1250_mxfp.matmul(a.cuda(),
                                  b.T.contiguous().cuda(),
                                  _gfx1250_mxfp.pack_scale(sa).cuda(),
                                  _gfx1250_mxfp.pack_scale(sb).cuda(), config=config)
    assert result.dtype == torch.float16
    torch.testing.assert_close(result.cpu().float(), expected, atol=2e-3, rtol=1e-3)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("k_iters", [3, 4, 5, 9])
@pytest.mark.parametrize("dtype_b,block_k", [("float8_e4m3", 128), ("float4", 256)])
@pytest.mark.parametrize("schedule", ["2x2", "split", "early", "combined"])
def test_mxgemm_cluster_and_output_schedule_ring_phase(k_iters, dtype_b, block_k, schedule):
    # Signed inputs, varying scales, both short/steady K, and every three-slot
    # ring phase. The unclustered case also has uneven output tile assignments.
    cluster = 1 if schedule == "early" else 4
    test_mxgemm_persistent_ring_phase(k_iters, 3, block_k == 128, dtype_b, block_k, cluster,
                                      CLUSTER_LAYOUT="2x2" if schedule in ("2x2", "combined") else "linear",
                                      SPLIT_CLUSTER_BARRIER=schedule in ("split", "combined"), EARLY_C_STORES=schedule
                                      in ("early", "combined"), CLUSTER_BARRIER_INTERVAL=1 if k_iters == 5 else 4)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("k_iters", [4, 5, 9])
@pytest.mark.parametrize("layout", ["4x2", "2x4", "4x4"])
@pytest.mark.parametrize("schedule", ["a8w8", "a8w8-4buf", "a8w4"])
def test_mxgemm_grid_cluster_ring_phase(k_iters, layout, schedule):
    # Grid clusters multicast A to the C CTAs of a row and B to the R CTAs of
    # a column. Use each operand's bench schedules: A8W8 with the static slot
    # unroll and three C slots or with a four-stage ring, and A8W4 with split
    # barriers and early C stores.
    rows, cols = map(int, layout.split("x"))
    a8w8 = schedule != "a8w4"
    static = schedule == "a8w8"
    test_mxgemm_persistent_ring_phase(k_iters, 4 if schedule == "a8w8-4buf" else 3, a8w8,
                                      "float8_e4m3" if a8w8 else "float4", 128 if a8w8 else 256, rows * cols,
                                      CLUSTER_LAYOUT=layout, STATIC_SLOT_UNROLL=static,
                                      OUTPUT_C_SLOTS=3 if static else None, SPLIT_CLUSTER_BARRIER=not a8w8,
                                      EARLY_C_STORES=not a8w8)


@pytest.mark.parametrize("config,match", [
    ({"TRANSPOSE_B": False}, "transposed B"),
    ({"NUM_PROGRAMS": 0}, "positive integer"),
    ({"NUM_BUFFERS": 4}, "at least NUM_BUFFERS"),
    ({"M": 384}, "full tiles"),
    ({"SCALE_PRESHUFFLE": False}, "preshuffled scales"),
    ({"TDM_SPLIT": True}, "unsplit descriptors"),
    ({"BLOCK_M": 128}, "256x256 M/N tiles"),
    ({"TDM_FUSION": "4way"}, "partial TDM fusion"),
    ({"BLOCK_K": 256, "NUM_BUFFERS": 2}, "BK256 cross-tile prefetch"),
    ({"CLUSTER_LAYOUT": "invalid"}, "CLUSTER_LAYOUT must be"),
    ({"CLUSTER_LAYOUT": "2x2"}, "CLUSTER_LAYOUT=2x2 requires CLUSTER_SIZE=4"),
    ({"CLUSTER_LAYOUT": "4x4"}, "CLUSTER_LAYOUT=4x4 requires CLUSTER_SIZE=16"),
    ({"SPLIT_CLUSTER_BARRIER": True}, "SPLIT_CLUSTER_BARRIER requires clustering"),
    ({"NUM_BUFFERS": 2, "EARLY_C_STORES": True}, "require three buffers"),
    ({"NUM_BUFFERS": 2, "STATIC_SLOT_UNROLL": True}, "STATIC_SLOT_UNROLL requires"),
    ({"OVERLAP_PEEL": True}, "OVERLAP_PEEL requires"),
    ({"OUTPUT_C_SLOTS": 1}, "OUTPUT_C_SLOTS requires"),
    ({"PERSISTENT": False, "OVERLAP_PEEL": True}, "require PERSISTENT"),
])
def test_mxgemm_persistent_invalid_config(config, match):
    # Validation happens before any GPU allocation or launch.
    M = config.pop("M", 512)
    a = torch.empty((M, 384), dtype=torch.float8_e4m3fn)
    b = torch.empty((512, 384), dtype=torch.float8_e4m3fn)
    a_scales = torch.empty((M // 128, 1536), dtype=torch.uint8)
    b_scales = torch.empty((4, 1536), dtype=torch.uint8)
    cfg = dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, NUM_BUFFERS=3, DTYPE_A="e4m3", DTYPE_B="e4m3", TRANSPOSE_B=True,
               SCHEDULE="sliceMNK", TDM_FUSION="partial", PERSISTENT=True, NUM_PROGRAMS=2)
    cfg.update(config)
    if not cfg["TRANSPOSE_B"]:
        b = b.T.contiguous()
    with pytest.raises(ValueError, match=match):
        _gfx1250_mxfp.matmul(a, b, a_scales, b_scales, config=cfg)


@triton.jit
def _mxfp_cluster_tile_ids(output, NUM_M: tl.constexpr, NUM_N: tl.constexpr, PROGRAMS: tl.constexpr,
                           GROUP_M: tl.constexpr, REMAP: tl.constexpr, LAYOUT: tl.constexpr = "2x2"):
    pid = tl.program_id(0)
    tile = _gfx1250_mxfp._mxgemm_cluster_program_id(pid, PROGRAMS, REMAP, 8, 2, LAYOUT)
    for step in range(NUM_M * NUM_N // PROGRAMS):
        off_m, off_n = _gfx1250_mxfp._mxgemm_tile_offsets(tile, NUM_M, NUM_N, GROUP_M, 256, 256, LAYOUT)
        tl.store(output + step * PROGRAMS + pid, off_m // 256 * NUM_N + off_n // 256)
        tile += PROGRAMS


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("num_m,num_n,programs", [(16, 8, 32), (12, 8, 32), (16, 8, 64), (12, 16, 64)])
@pytest.mark.parametrize("group_m", [4, 8])
@pytest.mark.parametrize("remap", [0, 2])
def test_mxgemm_cluster_2x2_tile_coverage(num_m, num_n, programs, group_m, remap):
    # Check the mapping separately from GEMM, including a short final M group.
    # Whole-cluster remapping must preserve the physical multicast rank bits.
    # Sixty-four programs exercise a full remap chunk; 32 cover its remainder.
    output = torch.empty(num_m * num_n, dtype=torch.int32, device="cuda")
    _mxfp_cluster_tile_ids[(programs, )](output, num_m, num_n, programs, group_m, remap, num_warps=4,
                                         ctas_per_cga=(4, 1, 1))
    ids = output.cpu().reshape(-1, 4)
    torch.testing.assert_close(ids.flatten().sort().values, torch.arange(num_m * num_n, dtype=torch.int32))
    rows, cols = ids // num_n, ids % num_n
    assert torch.equal(rows[:, 0], rows[:, 2]) and torch.equal(rows[:, 1], rows[:, 3])
    assert torch.equal(cols[:, 0], cols[:, 1]) and torch.equal(cols[:, 2], cols[:, 3])
    assert torch.equal(rows[:, 1], rows[:, 0] + 1)
    assert torch.equal(cols[:, 2], cols[:, 0] + 1)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("layout", ["4x2", "2x4", "4x4", "8x2", "2x8"])
@pytest.mark.parametrize("num_m,num_n,programs", [(16, 16, 32), (8, 16, 64), (16, 8, 128), (16, 16, 256)])
@pytest.mark.parametrize("remap", [0, 2])
def test_mxgemm_cluster_grid_tile_coverage(layout, num_m, num_n, programs, remap):
    # Every tile is covered once, and rank r of each R x C cluster covers M
    # tile r % R and N tile r // R of one aligned block, matching the A
    # (equal r % R) and B (equal r // R) multicast masks. Chunked XCD remapping
    # permutes whole clusters (eight XCDs, chunk two).
    rows_per, cols_per = map(int, layout.split("x"))
    size = rows_per * cols_per
    output = torch.empty(num_m * num_n, dtype=torch.int32, device="cuda")
    _mxfp_cluster_tile_ids[(programs, )](output, num_m, num_n, programs, 8, remap, layout, num_warps=4,
                                         ctas_per_cga=(size, 1, 1))
    ids = output.cpu().reshape(-1, size)
    torch.testing.assert_close(ids.flatten().sort().values, torch.arange(num_m * num_n, dtype=torch.int32))
    rows, cols = ids // num_n, ids % num_n
    rank = torch.arange(size)
    assert torch.equal(rows - rows[:, :1], (rank % rows_per).expand_as(rows))
    assert torch.equal(cols - cols[:, :1], (rank // rows_per).expand_as(cols))
    assert torch.equal(rows[:, 0] % rows_per, torch.zeros_like(rows[:, 0]))
    assert torch.equal(cols[:, 0] % cols_per, torch.zeros_like(cols[:, 0]))


def test_mxgemm_bench_defaults_and_overrides():
    from triton.language.extra.tlx.tutorials.amd_mxfp_gemm_gfx1250 import bench

    # Defaults: 4x4 clusters with an eight-block barrier interval and chunked
    # XCD remapping of whole clusters for both variants, each with FP32 and
    # then FP16 output. MX8xMX8 uses a four-stage ring with two C slots from
    # K=16384, static slot unroll from K=8192, and the overlap peel with three
    # stages. MX8xMX4 splits cluster barriers and starts C stores early.
    _, shapes, variants = bench.parse_benchmark_args(["--dry-run"])
    assert [(variant, args.out_dtype) for variant, _, args in variants] == [("mx8xmx8", "fp32"), ("mx8xmx8", "fp16"),
                                                                            ("mx8xmx4", "fp32"), ("mx8xmx4", "fp16")]
    for variant, dtype_b, args in variants:
        for shape in shapes:
            cmd = bench._command(bench._case_args(args, shape), shape, dtype_b)
            assert cmd[cmd.index("--cluster_layout") + 1] == "4x4"
            assert cmd[cmd.index("--cluster_size") + 1] == "16"
            assert cmd[cmd.index("--cluster_barrier_interval") + 1] == "8"
            assert cmd[cmd.index("--xcd_remap") + 1] == "chunked"
            a8w8 = variant == "mx8xmx8"
            four_stage = a8w8 and shape[2] >= 16384
            assert cmd[cmd.index("-BK") + 1] == ("128" if a8w8 else "256")
            assert cmd[cmd.index("--num_buffers") + 1] == ("4" if four_stage else "3")
            slots = None if not a8w8 or four_stage else ("3" if args.out_dtype == "fp32" else "6")
            assert (cmd[cmd.index("--output_c_slots") + 1] if "--output_c_slots" in cmd else None) == slots
            assert ("--static_slot_unroll" in cmd) == (a8w8 and shape[2] == 8192)
            assert ("--overlap_peel" in cmd) == (a8w8 and not four_stage)
            assert ("--no-cross_tile_prefetch" in cmd) == (not a8w8)
            assert ("--split_cluster_barrier" in cmd) == ("--early_c_stores" in cmd) == (not a8w8)
            assert (cmd[cmd.index("--out_dtype") + 1] if "--out_dtype" in cmd else "fp32") == args.out_dtype

    _, shapes, variants = bench.parse_benchmark_args(
        ["--cluster-size", "4", "--cluster-layout", "2x2", "--xcd-remap", "none", "--out-dtype", "fp16", "--dry-run"])
    assert [args.out_dtype for _, _, args in variants] == ["fp16", "fp16"]
    for _, dtype_b, args in variants:
        cmd = bench._command(bench._case_args(args, shapes[0]), shapes[0], dtype_b)
        assert cmd[cmd.index("--cluster_layout") + 1] == "2x2"
        assert cmd[cmd.index("--xcd_remap") + 1] == "none"

    # Opting out of the overlap peel, and forcing it on for the four-stage ring.
    for flag, expected in (("--no-overlap-peel", (False, False)), ("--overlap-peel", (True, True))):
        _, shapes, variants = bench.parse_benchmark_args(
            ["--variant", "mx8xmx8", "--out-dtype", "fp32", flag, "--dry-run"])
        (_, dtype_b, args), = variants
        for shape, want in zip((shapes[0], shapes[2]), expected):
            cmd = bench._command(bench._case_args(args, shape), shape, dtype_b)
            assert ("--overlap_peel" in cmd) == want


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("BATCH,H,SEQLEN_Q,SEQLEN_K", [(1, 8, 1024, 1024), (2, 4, 1024, 1024), (1, 16, 2048, 2048),
                                                       (1, 2, 896, 896), (1, 1, 640, 640), (1, 8, 2048, 2048),
                                                       (1, 2, 4096, 4096),  # underfilled long-K grids
                                                       (1, 2, 63, 65),  # one partial key tile
                                                       (1, 2, 128, 128),  # one full key tile
                                                       (1, 2, 129, 129),  # two key tiles with partial Q/K boundaries
                                                       (2, 2, 129, 257),  # partial Q and K tiles, independent lengths
                                                       (1, 2, 897, 897),  # final K tile contains a single valid row
                                                       (2, 3, 129, 65537),  # long-K fallback across batches/heads
                                                       (1, 40, 90000, 90000),  # long heads, partial tiles
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
        boundaries = [0, 31, 32, 63, 64, 127, 128, 255, 256, 65535, 65536, SEQLEN_Q - 2, SEQLEN_Q - 1]
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


# Automatic narrow/long-K routes plus forced medium/wide tiles exercise replay
# without allocating an unusually large adversarial query sequence.
@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seqlen_k,block_m,block_n", [(512, None, None), (1025, None, None), (1025, 128, 64),
                                                      (1025, 256, 64), (65537, None, None)])
@pytest.mark.parametrize("recovery_case", ["exp_overflow", "probability_overflow", "common_offset"])
def test_gfx1250_attn_fwd_tdm_pipelined_recovery(dtype, seqlen_k, block_m, block_n, recovery_case):
    torch.manual_seed(1)
    device = triton.runtime.driver.active.get_active_torch_device()
    heads = 32 if seqlen_k >= 65536 else 2
    q = torch.randn((1, heads, 257, 128), device=device, dtype=dtype)
    k = torch.randn((1, heads, seqlen_k, 128), device=device, dtype=dtype)
    v = torch.randn_like(k).abs().add_(0.25)
    scale = 128**-0.5
    # Mix safe and unsafe heads so replaying the wrong tile cannot repair all
    # outputs accidentally through a different permutation of workgroups.
    q_bad = q[:, -1:] if heads >= 32 else q
    k_bad = k[:, -1:] if heads >= 32 else k
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
            q_bad.fill_(1)
            k_bad.zero_()
            k_bad[:, :, 128:].fill_(16)
        elif recovery_case == "probability_overflow":
            q_bad.zero_()
            q_bad[..., 0] = 1
            k_bad.zero_()
            k_bad[:, :, 128:, 0] = 200
        else:
            q_bad.zero_()
            q_bad[..., 0] = 256
            q_bad[..., 1] = 1
            k_bad.zero_()
            k_bad[..., 0] = 256
            k_bad[..., 1] = torch.arange(k_bad.shape[2], device=device) % 9 - 4
        ref = ((q.double() @ k.double().transpose(-1, -2)) * scale).softmax(-1) @ v.double()
        for _ in range(3):
            graph.replay()
            assert torch.isfinite(out).all()
            tolerance = 0.005 if dtype == torch.bfloat16 else 0.0008
            torch.testing.assert_close(out.double(), ref, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["offset", "padded", "odd_stride"])
@pytest.mark.parametrize("seqlen_k,block_m,block_n,query_tiles_first", [
    (1025, 32, 128, False),
    (1025, 64, 128, False),
    (1025, 128, 64, False),
    (1025, 256, 64, False),
    pytest.param(1025, 128, 64, True, id="query_first"),
    # Q prefetch overlaps K/V in both the two-tile and longer startup paths.
    (129, 64, 128, False),
    (257, 64, 128, False),
    (257, 32, 128, False),
])
def test_gfx1250_attn_fwd_tdm_pipelined_strides(dtype, layout, seqlen_k, block_m, block_n, query_tiles_first):
    torch.manual_seed(2)
    device = triton.runtime.driver.active.get_active_torch_device()
    batch, heads, sq, sk, dim = 2, 2, 257, seqlen_k, 128

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
                                                               USE_FIXED_REFERENCE=True,
                                                               QUERY_TILES_FIRST=query_tiles_first,
                                                               num_warps=2 if block_m == 32 else 4, waves_per_eu=1)

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


@triton.jit
def _amd_independent_cluster_ids(output, SPLIT: tl.constexpr = False):
    pid = tl.program_id(0)
    rank = tlx.cluster_cta_rank()
    if SPLIT:
        tlx.amd_cluster_arrive()
        tlx.amd_cluster_wait()
    else:
        tlx.cluster_barrier()
    tl.store(output + pid, pid * 16 + rank)


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


@triton.jit
def _tdm_multicast_skew_kernel(a_ptr, b_ptr, out_ptr, CLUSTER_SIZE: tl.constexpr, BEFORE_LOAD: tl.constexpr,
                               ITERS: tl.constexpr):
    pid = tl.program_id(0)
    rank = tlx.cluster_cta_rank()
    a_desc = tl.make_tensor_descriptor(a_ptr, [ITERS * 32, 32], [32, 1], [32, 32])
    b_desc = tl.make_tensor_descriptor(b_ptr, [ITERS * 32, 32], [32, 1], [32, 32])
    a_buf = tlx.local_alloc((32, 32), tl.int32, 1)
    b_buf = tlx.local_alloc((32, 32), tl.int32, 1)
    a_view, b_view = tlx.local_view(a_buf, 0), tlx.local_view(b_buf, 0)
    mask: tl.constexpr = (1 << CLUSTER_SIZE) - 1
    offsets = tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
    for i in range(ITERS):
        # Delay one CTA either before requesting, or while its LDS slot is live.
        # Peers can advance several iterations, including reusing the same slot.
        if BEFORE_LOAD and rank == 1:
            tl.inline_asm_elementwise(".rept 4\ns_sleep 15\n.endr\ns_mov_b32 $0, 0", constraints="=s", args=[],
                                      dtype=tl.int32, is_pure=False, pack=1)
        ad = tlx.update_tensor_descriptor(a_desc, add_offsets=[i * 32, 0])
        bd = tlx.update_tensor_descriptor(b_desc, add_offsets=[i * 32, 0])
        tlx.async_amd_descriptor_load_fused([(ad, a_view, 5), (bd, b_view, 10)], multicast_masks=[mask, mask])
        tlx.async_amd_descriptor_wait(0)
        if not BEFORE_LOAD and rank == 1:
            tl.inline_asm_elementwise(".rept 4\ns_sleep 15\n.endr\ns_mov_b32 $0, 0", constraints="=s", args=[],
                                      dtype=tl.int32, is_pure=False, pack=1)
        tlx.workgroup_barrier()
        value = tlx.local_load(a_view) + tlx.local_load(b_view)
        tl.store(out_ptr + (pid * ITERS + i) * 1024 + offsets, value)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("cluster_size", [2, 4])
@pytest.mark.parametrize("before_load", [False, True])
def test_tdm_multicast_skew_without_cluster_barriers(device, cluster_size, before_load):
    iters = 8
    a_cpu = torch.arange(iters * 1024, dtype=torch.int32).reshape(iters * 32, 32)
    b_cpu = a_cpu * 7 + 11
    a, b = a_cpu.to(device), b_cpu.to(device)
    out = torch.empty((cluster_size, iters * 32, 32), device=device, dtype=torch.int32)
    compiled = _tdm_multicast_skew_kernel[(cluster_size, )](a, b, out, cluster_size, before_load, iters, num_warps=4,
                                                            ctas_per_cga=(cluster_size, 1, 1))
    assert "amdg.cluster_barrier" not in compiled.asm["ttgir"]
    torch.testing.assert_close(out.cpu(), (a_cpu + b_cpu).expand(cluster_size, -1, -1), atol=0, rtol=0)


@triton.jit
def _partitioned_tdm_reshape_kernel(x_ptr, output_ptr, ROWS: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
                                    GROUP_MAJOR: tl.constexpr):
    G: tl.constexpr = ROWS // R
    if GROUP_MAJOR:
        # [G, R, K] with R rows per group split into two pieces along dim 1.
        # One TDM load fills each partition; read it back as a flat view.
        piece: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[K, 8]], [G, R // 2, K], [2, 1, 0])
        layout: tl.constexpr = tlx.partitioned_shared_layout_encoding(2, 1, 1, piece)
        desc = tl.make_tensor_descriptor(x_ptr, [G, R, K], [R * K, K, 1], [G, R, K])
        buf = tlx.local_alloc((G, R, K), tl.float16, 1, layout=layout)
        view = tlx.local_view(buf, 0)
        tlx.async_amd_descriptor_load(desc, view, [0, 0, 0])
        tlx.async_amd_descriptor_wait(pendings=0)
        flat = tlx.local_reshape(view, [ROWS, K])
        x = tlx.local_load(flat)
        offsets = tl.arange(0, ROWS)[:, None] * K + tl.arange(0, K)[None, :]
        tl.store(output_ptr + offsets, x)
    else:
        # [ROWS, K] with G groups of two R / 2-row pieces along dim 0; read it
        # back through the unfolded [G, R, K] view.
        piece: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[K, 8]], [R // 2, K], [1, 0])
        layout: tl.constexpr = tlx.partitioned_shared_layout_encoding(2, G, 0, piece)
        desc = tl.make_tensor_descriptor(x_ptr, [ROWS, K], [K, 1], [ROWS, K])
        buf = tlx.local_alloc((ROWS, K), tl.float16, 1, layout=layout)
        view = tlx.local_view(buf, 0)
        tlx.async_amd_descriptor_load(desc, view, [0, 0])
        tlx.async_amd_descriptor_wait(pendings=0)
        grouped = tlx.local_reshape(view, [G, R, K])
        x = tlx.local_load(grouped)
        offsets = (tl.arange(0, G)[:, None, None] * (R * K) + tl.arange(0, R)[None, :, None] * K +
                   tl.arange(0, K)[None, None, :])
        tl.store(output_ptr + offsets, x)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("group_major", [True, False])
@pytest.mark.parametrize("ROWS, R", [(256, 128), (256, 64)])
@pytest.mark.parametrize("num_warps", [2, 4, 8])
def test_partitioned_tdm_load_reshape_gfx1250(device, group_major, ROWS, R, num_warps):
    """TDM fills a partitioned buffer that is read back through a reshaped view.

    With more warps than pieces, the extra warps of a group-major load must
    split the groups rather than the pieces, so each warp's tile stays
    contiguous in its partition.
    """
    K = 64
    x = torch.randn(ROWS, K, dtype=torch.float16, device=device)
    output = torch.empty_like(x)
    _partitioned_tdm_reshape_kernel[(1, )](x, output, ROWS=ROWS, K=K, R=R, GROUP_MAJOR=group_major, num_warps=num_warps)
    torch.testing.assert_close(x, output, atol=0, rtol=0)


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
                          occ1=True, occ1_k_bands=2, occ1_load_after_dk=True, occ1_pad_k=not packed_dq or q_splits == 1,
                          log2_lse=True, merge_stats=True)


def _attention_backward_occ1_pad_k(config):
    import dataclasses

    # Split packed kernels keep the unpadded K image; padding spills them.
    pad_k = not config.packed_dq or config.q_splits == 1 and config.tail_splits == 1
    return dataclasses.replace(config, occ1_pad_k=pad_k) if config.occ1 else config


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


def _attention_backward_causal_config(packed_dq, q_splits=1):
    import dataclasses

    return dataclasses.replace(_attention_backward_occ1_config(packed_dq, q_splits=q_splits),
                               occ1_k_bands=0 if q_splits == 4 else 1)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("packed_dq", [False, True])
@pytest.mark.parametrize("shape,q_splits", [
    ((1, 1, 1, 512), 1),
    ((2, 4, 4, 1024), 1),
    ((1, 4, 4, 4096), 1),
    ((1, 2, 2, 8192), 1),
    # GROUP > 1 restarts each query head at the KV tile's first unmasked row.
    ((1, 8, 2, 512), 1),
    ((1, 8, 2, 2048), 1),
    ((2, 64, 8, 512), 1),
    # Splits interleave each tile's outers; two dQ slices at 16K.
    ((1, 4, 4, 1024), 2),
    ((2, 8, 2, 1024), 2),
    ((2, 8, 2, 1024), 4),
    ((1, 64, 8, 512), 4),
    ((1, 2, 2, 16384), 2),
])
def test_attention_backward_causal_correctness(shape, q_splits, packed_dq, monkeypatch):
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import allocate_workspace, fa_backward, select_config

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(2709)
    b, hq, hk, n = shape
    q, dout = [torch.randn((b, hq, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    k, v = [torch.randn((b, hk, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    qr, kr, vr = [t.float().requires_grad_() for t in (q, k, v)]
    scores = qr @ kr.repeat_interleave(hq // hk, 1).transpose(-1, -2) * 128**-0.5
    causal = torch.ones((n, n), device="cuda", dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    out = scores.softmax(-1) @ vr.repeat_interleave(hq // hk, 1)
    expected = torch.autograd.grad(out, (qr, kr, vr), dout.float())
    case = (q, k, v, out.detach().bfloat16(), dout, scores.detach().logsumexp(-1))
    config = _attention_backward_causal_config(packed_dq, q_splits)
    if q_splits == 1:
        assert select_config(n, b, hq, hk, num_cus=192, packed_dq=packed_dq, causal=True) == config
    workspace = allocate_workspace(q, k, config=config, causal=True)

    def nrmse(value, ref):
        return ((value.float() - ref.float()).square().mean() / ref.float().square().mean()).sqrt().item()

    for call in range(3):
        for buffer in workspace:
            buffer.fill_(float("nan"))
        actual = fa_backward(*case, causal=True, config=config, workspace=workspace)
        assert all(torch.isfinite(value).all() for value in actual)
        assert all(nrmse(value, ref) < 0.012 for value, ref in zip(actual, expected))


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("changes", [dict(q_splits=3), dict(q_splits=4), dict(tail_splits=3), dict(occ1=False)])
def test_attention_backward_causal_validation(changes):
    import dataclasses
    from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import fa_backward

    b, hq, hk, n = 2, 4, 4, 1024
    q, o, do = [torch.randn((b, hq, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    k, v = [torch.randn((b, hk, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    lse = torch.zeros((b, hq, n), device="cuda", dtype=torch.float32)
    config = dataclasses.replace(_attention_backward_occ1_config(True), **changes)
    with pytest.raises(ValueError, match="causal gfx1250 backward requires OCC1 without tail splits"):
        fa_backward(q, k, v, o, do, lse, causal=True, config=config)


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
