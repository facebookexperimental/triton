"""End-to-end tests for auto-TMA promotion (TRITON_AUTO_TMA=1).

The PromoteLoadToTMA pass rewrites eligible masked block loads into
tt.descriptor_load, which the sm_90+ pipeline lowers to real TMA copies. These
tests verify the rewrite produces correct results on Hopper+ (H100/H200).
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_cuda, torch_dtype_name


def _has_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


pytestmark = pytest.mark.skipif(not _has_hopper(), reason="Requires Hopper or newer (sm90+)")


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


def test_auto_tma_vector_add():
    N = 8192
    x = torch.randn(N, device="cuda", dtype=torch.float16)
    y = torch.randn(N, device="cuda", dtype=torch.float16)
    out = torch.empty(N, device="cuda", dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    _add_kernel[grid](x, y, out, N, BLOCK=256)
    torch.testing.assert_close(out, x + y, atol=1e-2, rtol=1e-2)


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


def test_auto_tma_2d_scale():
    M, N = 200, 300  # non-multiples of BLOCK to exercise the mask
    BLOCK_M, BLOCK_N = 64, 64
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _scale_2d_kernel[grid](x, out, M, N, x.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.testing.assert_close(out, x * 2.0, atol=1e-2, rtol=1e-2)


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


def test_auto_tma_block_over_256_promoted():
    # A BLOCK > 256 load IS auto-TMA promoted: the host recipe encodes a
    # getTMABlockShape-clamped box (<=256) while the descriptor_load keeps the
    # full-block SMEM result, and the device TMA lowering multi-copies the box to
    # fill the tile. Both x and y loads (BLOCK=512) promote; result is correct.
    N, BLOCK = 8192, 512  # 512 > 256 element box cap -> clamped box + multi-copy
    x = torch.randn(N, device="cuda", dtype=torch.float16)
    y = torch.randn(N, device="cuda", dtype=torch.float16)
    out = torch.empty(N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(N, BLOCK), )
    k = _add_kernel[grid](x, y, out, N, BLOCK=BLOCK)
    torch.testing.assert_close(out, x + y, atol=1e-2, rtol=1e-2)
    assert k.asm["ttir"].count("tt.descriptor_load") == 2, (
        "both BLOCK=512 loads must be auto-TMA promoted (clamped box + multi-copy)")


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


def test_auto_tma_mixed_promoted_loads():
    # A single kernel with a BLOCK_A=256 load and a BLOCK_B=512 load. With box>256
    # support BOTH promote: the 512 load encodes a getTMABlockShape-clamped (<=256)
    # box and the device lowering multi-copies to fill its full-block SMEM.
    # Verifies two descriptor_loads and correct results for both.
    BLOCK_A, BLOCK_B = 256, 512  # both promoted; 512 uses clamped box + multi-copy
    ntiles = 8
    Na, Nb = ntiles * BLOCK_A, ntiles * BLOCK_B
    a = torch.randn(Na, device="cuda", dtype=torch.float16)
    b = torch.randn(Nb, device="cuda", dtype=torch.float16)
    oa = torch.empty(Na, device="cuda", dtype=torch.float16)
    ob = torch.empty(Nb, device="cuda", dtype=torch.float16)
    k = _mixed_block_kernel[(ntiles, )](a, b, oa, ob, Na, Nb, BLOCK_A=BLOCK_A, BLOCK_B=BLOCK_B)
    torch.testing.assert_close(oa, a + 1.0, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ob, b + 2.0, atol=1e-2, rtol=1e-2)
    assert k.asm["ttir"].count("tt.descriptor_load") == 2, (
        "both the BLOCK_A=256 and BLOCK_B=512 loads must be auto-TMA promoted")


# Store promotion is WS-gated (see PromoteLoadToTMA.cpp): a tl.range with
# warp_specialize=True carries the tt.warp_specialize attr, which is what
# isTMAEligibleStore keys off. This kernel makes the store's contiguous box dim
# (BLOCK_N) the thing under test.
@triton.jit
def _ws_scale_store(x_ptr, o_ptr, M, N, stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_m[:, None] * stride_m + offs_n[None, :], mask=mask, other=0.0)
        tl.store(o_ptr + offs_m[:, None] * stride_m + offs_n[None, :], x * 2.0, mask=mask)


def test_auto_tma_store_block_over_256_promoted():
    # STORE path with contiguous box dim BLOCK_N=512 > 256 IS promoted: the store
    # recipe encodes a getTMABlockShape-clamped box and the device
    # AsyncTMACopyLocalToGlobal lowering multi-copies from the full-block SMEM to
    # fill the tile. Result is correct.
    M, N = 128, 512
    BLOCK_M, BLOCK_N = 32, 512  # BLOCK_N=512 > 256 box cap; BLOCK_M small so the
    # 3-stage load pipeline + store staging of the 512-wide tile fits SMEM.
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    o = torch.empty((M, N), device="cuda", dtype=torch.float16)
    grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )
    k = _ws_scale_store[grid](x, o, M, N, x.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_SMS=NUM_SMS, num_warps=4)
    torch.testing.assert_close(o, x * 2.0, atol=1e-2, rtol=1e-2)
    assert "tt.descriptor_store" in k.asm["ttir"], (
        "BLOCK_N=512 store must be auto-TMA store-promoted (clamped box + multi-copy)")


# A store whose mask carries an extra (non-rectangular) predicate on top of the
# per-dim boundary. Store promotion may only reduce a mask to the descriptor's
# globalDim bounds when the mask is exactly AND_d(offs_d < E_d); an extra term
# would make the TMA store write elements the original masked off.
@triton.jit
def _ws_extra_pred_store(x_ptr, o_ptr, M, N, stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rect = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        # other=1.0 (non-zero) keeps this load ineligible for auto-TMA, so the
        # kernel has no promoted TMA load -- isolates the store path and avoids an
        # unrelated WS partition-scheduler issue with a TMA load in a trivial
        # elementwise WS loop. rect is fully in-bounds here, so the fill is unused.
        x = tl.load(x_ptr + offs_m[:, None] * stride_m + offs_n[None, :], mask=rect, other=1.0)
        # extra predicate beyond the rectangular boundary: skip column 0.
        mask = rect & (offs_n[None, :] > 0)
        tl.store(o_ptr + offs_m[:, None] * stride_m + offs_n[None, :], x * 2.0, mask=mask)


def test_auto_tma_store_nonrectangular_mask_not_promoted():
    # A store whose mask is not a pure rectangular boundary (here: an extra
    # skip-column-0 predicate) must NOT be promoted to a descriptor_store: the TMA
    # store bounds only via globalDim, so it would write the whole in-bounds box
    # and clobber the elements the extra predicate masked off (a miscompile). It
    # must stay an ordinary masked store and leave the masked-off elements
    # untouched.
    M, N = 128, 128
    BLOCK_M, BLOCK_N = 64, 128  # <=256 so the box guard is not what rejects it; single n-tile
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    o = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )
    k = _ws_extra_pred_store[grid](x, o, M, N, x.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_SMS=NUM_SMS,
                                   num_warps=4)
    assert "tt.descriptor_store" not in k.asm["ttir"], (
        "a non-rectangular store mask must not be auto-TMA store-promoted")
    cols = torch.arange(N, device="cuda")[None, :]
    ref = torch.where(cols > 0, x * 2.0, torch.zeros_like(x))
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=1e-2)


def test_auto_tma_outer_dim_over_256():
    # OUTER (strided) box dim BLOCK_M=512 > 256, inner BLOCK_N=64. getTMABlockShape
    # caps the outer box at 256 and the device lowering multi-copies 512/256=2
    # along it; the inner dim is within the swizzle width. Promoted + correct.
    # (A 2D tile with BOTH dims > 256 is omitted: 512x512 fp16 SMEM exceeds the
    # hardware limit -- unrelated to the TMA box logic.)
    M, N = 1024, 64
    BLOCK_M, BLOCK_N = 512, 64  # outer 512 > 256 -> clamped box + multi-copy
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    k = _scale_2d_kernel[grid](x, out, M, N, x.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.testing.assert_close(out, x * 2.0, atol=1e-2, rtol=1e-2)
    assert "tt.descriptor_load" in k.asm["ttir"], (
        "BLOCK_M=512 outer-dim load must be auto-TMA promoted (clamped box + multi-copy)")


@triton.jit
def _ws_load_only_scale(x_ptr, o_ptr, M, N, stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rect = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        # rectangular mask + zero fill -> LOAD promotes to auto-TMA (BLOCK_N=512)
        x = tl.load(x_ptr + offs_m[:, None] * stride_m + offs_n[None, :], mask=rect, other=0.0)
        # extra predicate -> STORE is NOT promoted, isolating the load path
        smask = rect & (offs_n[None, :] > 0)
        tl.store(o_ptr + offs_m[:, None] * stride_m + offs_n[None, :], x * 2.0, mask=smask)


def test_auto_tma_autows_load_over_256():
    # Exercise a host-recipe TMA load inside AutoWS with BLOCK_N > 256.
    M, N = 128, 512
    BLOCK_M, BLOCK_N = 32, 512
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    o = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )
    k = _ws_load_only_scale[grid](x, o, M, N, x.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_SMS=NUM_SMS,
                                  num_warps=4)
    ref = x * 2.0
    ref[:, 0] = 0.0
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=1e-2)
    assert "tt.descriptor_load" in k.asm["ttir"], "expected the BLOCK_N=512 load to promote"


def test_auto_tma_metaws_store_over_256():
    # Exercise a promoted store through meta-WS with BLOCK_N > 256.
    M, N = 128, 512
    BLOCK_M, BLOCK_N = 32, 512
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    o = torch.empty((M, N), device="cuda", dtype=torch.float16)
    grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        k = _ws_scale_store[grid](x, o, M, N, x.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_SMS=NUM_SMS,
                                  num_warps=4)
    torch.testing.assert_close(o, x * 2.0, atol=1e-2, rtol=1e-2)
    assert "tt.descriptor_load" in k.asm["ttir"], (
        "meta-WS BLOCK_N=512 load must be auto-TMA promoted (clamped box + multi-copy)")
    assert "tt.descriptor_store" in k.asm["ttir"], (
        "meta-WS BLOCK_N=512 store must be auto-TMA store-promoted (clamped box + multi-copy)")


def test_auto_tma_device_block_over_256(monkeypatch):
    # DEVICE-descriptor mode (TRITON_AUTO_TMA_DEVICE=1, added in this diff): the load
    # is promoted to an in-kernel tt.make_tensor_descriptor. Contiguous BLOCK_N=512 >
    # 256. Unlike the host recipe, the device path clamps the box via getTMABlockShape
    # at descriptor-build time (TMAUtilities::createTMADesc) and the device lowering
    # multi-copies to fill the tile, so box>256 promotes and computes correctly. Non-WS,
    # so it does not touch the AutoWS partition bug that blocks the host store path.
    monkeypatch.setenv("TRITON_AUTO_TMA_DEVICE", "1")
    M, N = 128, 512
    BLOCK_M, BLOCK_N = 64, 512  # contiguous 512 > 256 -> device box clamp + multi-copy
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    k = _scale_2d_kernel[grid](x, out, M, N, x.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.testing.assert_close(out, x * 2.0, atol=1e-2, rtol=1e-2)
    assert "tt.make_tensor_descriptor" in k.asm["ttir"], "expected a device descriptor"
    assert "tt.descriptor_load" in k.asm["ttir"], ("device-descriptor BLOCK_N=512 load must be auto-TMA promoted")


@triton.jit
def _desc_copy_kernel(desc):
    desc.store([0, 0], desc.load([0, 0]))


def _assert_recipes_match_explicit(k, x):
    # Compare with a host TensorDescriptor; the pass sees signless ints, which map to unsigned.
    from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST
    from triton.tools.tensor_descriptor import TensorDescriptor
    ref = x.view({torch.int32: torch.uint32, torch.int64: torch.uint64}.get(x.dtype, x.dtype))
    ref_k = _desc_copy_kernel.warmup(TensorDescriptor.from_tensor(ref, [64, 64]), grid=(1, ))
    expected = TMA_DTYPE_DEVICE_TO_HOST[ref_k.metadata.tensordesc_meta[0]["elem_type"]]
    recipes = k.metadata.auto_tma_recipes
    assert recipes and all(r["elem_type"] == expected for r in recipes)


@triton.jit
def _copy_2d_kernel(x_ptr, out_ptr, M, N, stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * stride_m + offs_n[None, :], mask=mask)
    tl.store(out_ptr + offs_m[:, None] * stride_m + offs_n[None, :], x, mask=mask)


@pytest.mark.parametrize("dtype", [
    torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64, torch.bfloat16,
    torch.float8_e4m3fn, torch.float8_e5m2
], ids=torch_dtype_name)
def test_auto_tma_load_dtype_encoding(dtype):
    # Masked copy of random bytes. Check the recipe before launch: a bad elem_type fails the launch.
    M, N = 200, 304
    x = torch.randint(0, 256, (M, N * dtype.itemsize), device="cuda", dtype=torch.uint8).view(dtype)
    out = torch.zeros_like(x)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    k = _copy_2d_kernel.warmup(x, out, M, N, x.stride(0), BLOCK_M=64, BLOCK_N=64, grid=grid)
    assert "tt.descriptor_load" in k.asm["ttir"]
    _assert_recipes_match_explicit(k, x)
    _copy_2d_kernel[grid](x, out, M, N, x.stride(0), BLOCK_M=64, BLOCK_N=64)
    assert torch.equal(out.view(torch.uint8), x.view(torch.uint8))


@triton.jit
def _ws_atomic_add_2d_kernel(x_ptr, out_ptr, cnt_ptr, M, N, stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                             NUM_SMS: tl.constexpr):
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = tl.cdiv(M, BLOCK_M) * num_pid_n
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, warp_specialize=True):
        offs_m = (tile_id // num_pid_n) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = (tile_id % num_pid_n) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_m[:, None] * stride_m + offs_n[None, :], mask=mask, other=0)
        tl.atomic_add(out_ptr + offs_m[:, None] * stride_m + offs_n[None, :], x, mask=mask, sem="relaxed")
        tl.atomic_add(cnt_ptr, 1, sem="relaxed")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32, torch.float64], ids=torch_dtype_name)
def test_auto_tma_atomic_add(dtype):
    # The atomic consumes a promoted load. f64 (no TMA reduce-add) and the scalar counter stay plain atomics.
    M, N = 512, 384
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    x, out = (torch.randint(-1000, 1000, (M, N), device="cuda").to(dtype) for _ in range(2))
    ref, cnt = out + x, torch.zeros(1, device="cuda", dtype=torch.int32)
    num_tiles = triton.cdiv(M, 64) * triton.cdiv(N, 64)
    grid = (min(NUM_SMS, num_tiles), )
    args = (x, out, cnt, M, N, x.stride(0))
    kwargs = dict(BLOCK_M=64, BLOCK_N=64, NUM_SMS=NUM_SMS, num_warps=4)
    k = _ws_atomic_add_2d_kernel.warmup(*args, **kwargs, grid=grid)
    ttir = k.asm["ttir"]
    assert "tt.descriptor_load" in ttir and ("tt.descriptor_reduce" in ttir) == (dtype != torch.float64)
    _assert_recipes_match_explicit(k, x)
    _ws_atomic_add_2d_kernel[grid](*args, **kwargs)
    assert torch.equal(out, ref) and cnt.item() == num_tiles
