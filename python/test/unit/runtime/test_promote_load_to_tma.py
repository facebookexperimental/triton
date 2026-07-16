"""End-to-end tests for auto-TMA promotion (TRITON_AUTO_TMA=1).

The PromoteLoadToTMA pass rewrites eligible masked block loads into
tt.descriptor_load, which the sm_90+ pipeline lowers to real TMA copies. These
tests verify the rewrite produces correct results on Hopper+ (H100/H200).
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_cuda


def _has_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


@pytest.fixture(autouse=True)
def _auto_tma_env():
    # TMA descriptors are built into global scratch, which needs an allocator.
    if is_cuda():
        triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.auto_tma = True
        yield


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@pytest.mark.skipif(not _has_hopper(), reason="auto-TMA requires Hopper+ (sm_90)")
def test_auto_tma_vector_add():
    N = 8192
    x = torch.randn(N, device="cuda", dtype=torch.float16)
    y = torch.randn(N, device="cuda", dtype=torch.float16)
    out = torch.empty(N, device="cuda", dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    _add_kernel[grid](x, y, out, N, BLOCK=256)
    torch.testing.assert_close(out, x + y, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _has_hopper(), reason="auto-TMA requires Hopper+ (sm_90)")
def test_auto_tma_vector_add_non_multiple():
    # N not a multiple of BLOCK exercises the masked / OOB-zero-fill tail.
    N = 8000
    x = torch.randn(N, device="cuda", dtype=torch.float16)
    y = torch.randn(N, device="cuda", dtype=torch.float16)
    out = torch.empty(N, device="cuda", dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    _add_kernel[grid](x, y, out, N, BLOCK=256)
    torch.testing.assert_close(out, x + y, atol=1e-2, rtol=1e-2)


@triton.jit
def _scale_2d_kernel(x_ptr, out_ptr, M, N, stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(out_ptr + offs_m[:, None] * stride_m + offs_n[None, :], x * 2.0, mask=mask)


@pytest.mark.skipif(not _has_hopper(), reason="auto-TMA requires Hopper+ (sm_90)")
def test_auto_tma_2d_scale():
    M, N = 200, 300  # non-multiples of BLOCK to exercise the mask
    BLOCK_M, BLOCK_N = 64, 64
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _scale_2d_kernel[grid](x, out, M, N, x.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.testing.assert_close(out, x * 2.0, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _has_hopper(), reason="auto-TMA requires Hopper+ (sm_90)")
def test_auto_tma_no_allocator_needed():
    # Host-built TMA (launch.h recipe core) must NOT require a runtime allocator:
    # the CUtensorMap is built on the host, not in device global scratch. With
    # the null allocator active, a device-built descriptor would raise "Kernel
    # requires a runtime memory allocation"; the host-built recipe path runs
    # cleanly. This is the decisive proof that auto-TMA no longer depends on a
    # global-scratch allocator.
    from triton.runtime._allocation import NullAllocator

    triton.set_allocator(NullAllocator())
    N = 8192
    x = torch.randn(N, device="cuda", dtype=torch.float16)
    y = torch.randn(N, device="cuda", dtype=torch.float16)
    out = torch.empty(N, device="cuda", dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    _add_kernel[grid](x, y, out, N, BLOCK=256)
    torch.testing.assert_close(out, x + y, atol=1e-2, rtol=1e-2)


@triton.jit
def _knob_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@pytest.mark.skipif(not _has_hopper(), reason="auto-TMA requires Hopper+ (sm_90)")
def test_auto_tma_per_call_option():
    # With the global TRITON_AUTO_TMA knob OFF, the per-call `auto_tma` compile
    # option must independently control promotion: auto_tma=True promotes the
    # load to a descriptor_load (host-built TMA, no allocator), auto_tma=False
    # leaves it an ordinary load.
    triton.knobs.nvidia.auto_tma = False  # override the autouse fixture's scope
    from triton.runtime._allocation import NullAllocator

    triton.set_allocator(NullAllocator())
    N, BLOCK = 8192, 256
    x = torch.randn(N, device="cuda", dtype=torch.float16)
    y = torch.randn(N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(N, BLOCK), )

    out_on = torch.empty(N, device="cuda", dtype=torch.float16)
    k_on = _knob_add_kernel[grid](x, y, out_on, N, BLOCK=BLOCK, auto_tma=True)
    torch.testing.assert_close(out_on, x + y, atol=1e-2, rtol=1e-2)
    assert "tt.descriptor_load" in k_on.asm["ttir"], "auto_tma=True should promote to TMA"

    out_off = torch.empty(N, device="cuda", dtype=torch.float16)
    k_off = _knob_add_kernel[grid](x, y, out_off, N, BLOCK=BLOCK, auto_tma=False)
    torch.testing.assert_close(out_off, x + y, atol=1e-2, rtol=1e-2)
    assert "tt.descriptor_load" not in k_off.asm["ttir"], "auto_tma=False should stay a plain load"


@pytest.mark.skipif(not _has_hopper(), reason="auto-TMA requires Hopper+ (sm_90)")
def test_auto_tma_autotune_config():
    # auto_tma is an autotunable per-Config option (like num_warps): the
    # autotuner compiles both the non-TMA and TMA variants and picks one; both
    # must be correct. Global knob OFF so only the Config drives promotion.
    triton.knobs.nvidia.auto_tma = False  # override the autouse fixture's scope
    from triton.runtime._allocation import NullAllocator

    triton.set_allocator(NullAllocator())

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 256}, num_warps=4, auto_tma=False),
            triton.Config({"BLOCK": 256}, num_warps=4, auto_tma=True),
        ],
        key=["N"],
    )
    @triton.jit
    def _autotuned_add(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    N = 8192
    x = torch.randn(N, device="cuda", dtype=torch.float16)
    y = torch.randn(N, device="cuda", dtype=torch.float16)
    out = torch.empty(N, device="cuda", dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    _autotuned_add[grid](x, y, out, N)
    torch.testing.assert_close(out, x + y, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _has_hopper(), reason="auto-TMA requires Hopper+ (sm_90)")
def test_auto_tma_block_over_256_not_promoted():
    # cuTensorMapEncodeTiled caps every box (block) dim at 256 elements, and the
    # host recipe path builds the CUtensorMap with box_dim == block_shape (no
    # clamp / decompose in launch.h). So a BLOCK > 256 load must NOT be auto-TMA
    # promoted -- it would fail encode at launch. It stays an ordinary load and
    # still computes the right result.
    N, BLOCK = 8192, 512  # 512 > 256 element box cap
    x = torch.randn(N, device="cuda", dtype=torch.float16)
    y = torch.randn(N, device="cuda", dtype=torch.float16)
    out = torch.empty(N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(N, BLOCK), )
    k = _add_kernel[grid](x, y, out, N, BLOCK=BLOCK)
    torch.testing.assert_close(out, x + y, atol=1e-2, rtol=1e-2)
    assert "tt.descriptor_load" not in k.asm["ttir"], (
        "BLOCK>256 exceeds the TMA box limit and must not be auto-TMA promoted")


@triton.jit
def _mixed_block_kernel(a_ptr, b_ptr, oa_ptr, ob_ptr, Na, Nb, BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr):
    pid = tl.program_id(0)
    offs_a = pid * BLOCK_A + tl.arange(0, BLOCK_A)
    mask_a = offs_a < Na
    a = tl.load(a_ptr + offs_a, mask=mask_a)
    tl.store(oa_ptr + offs_a, a + 1.0, mask=mask_a)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < Nb
    b = tl.load(b_ptr + offs_b, mask=mask_b)
    tl.store(ob_ptr + offs_b, b + 2.0, mask=mask_b)


@pytest.mark.skipif(not _has_hopper(), reason="auto-TMA requires Hopper+ (sm_90)")
def test_auto_tma_mixed_promoted_and_oversize_load():
    # A single kernel with one TMA-eligible load (BLOCK_A=256, promoted to
    # tt.descriptor_load) and one oversize load (BLOCK_B=512 > the 256 element box
    # cap, left as an ordinary load). Promotion is per-load, so the eligible load
    # must promote while the oversize one stays behind. The host then encodes only
    # the legal 256-box descriptor, so the launch must not fail (an unclamped 512
    # box would make cuTensorMapEncodeTiled reject the whole launch). Verifies
    # exactly one descriptor_load in the IR and correct results for both loads.
    BLOCK_A, BLOCK_B = 256, 512  # 256 <= box cap (promoted); 512 > cap (not promoted)
    ntiles = 8
    Na, Nb = ntiles * BLOCK_A, ntiles * BLOCK_B
    a = torch.randn(Na, device="cuda", dtype=torch.float16)
    b = torch.randn(Nb, device="cuda", dtype=torch.float16)
    oa = torch.empty(Na, device="cuda", dtype=torch.float16)
    ob = torch.empty(Nb, device="cuda", dtype=torch.float16)
    k = _mixed_block_kernel[(ntiles, )](a, b, oa, ob, Na, Nb, BLOCK_A=BLOCK_A, BLOCK_B=BLOCK_B)
    torch.testing.assert_close(oa, a + 1.0, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ob, b + 2.0, atol=1e-2, rtol=1e-2)
    assert k.asm["ttir"].count("tt.descriptor_load") == 1, (
        "exactly the BLOCK_A=256 load must be auto-TMA promoted; the BLOCK_B=512 "
        "load exceeds the box cap and must stay an ordinary load")
