import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell, is_cuda, is_hip_cdna4
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# The FA4 "separable" layout for a 128x128 TMEM tile, written purely as
# shape/stride (a CuTe thread-value layout). The two top-level modes are
# (thread, value); strides are flat row-major offsets into the tile
# (offset = n * 128 + m). The compiler splits the thread bits into lane/warp and
# the value bits into registers, producing the #linear encoding below (the
# "value -> M, thread -> N" separable layout used by the MXFP8 attention
# backward).
# (thread, value) shape/stride for the FA4 "separable" 128x128 TMEM tile.
_SEPARABLE_QK_SHAPE = ((32, 4, 2), (32, 2))
_SEPARABLE_QK_STRIDE = ((128, 4096, 32), (1, 64))


def _separable_qk_layout():
    return tlx.layout(
        shape=_SEPARABLE_QK_SHAPE,  # (thread, value)
        stride=_SEPARABLE_QK_STRIDE,
    )


def _cute_shape_stride(shape, stride):
    """Render a (thread, value) shape/stride pair in CuTe Shape:Stride form,
    matching the DumpLayout emitter (`_N` for a single mode, `(_a,_b,...)`
    otherwise)."""

    def group(modes):
        if len(modes) == 1:
            return f"_{modes[0]}"
        return "(" + ",".join(f"_{m}" for m in modes) + ")"

    def side(groups):
        return "(" + ",".join(group(g) for g in groups) + ")"

    return f"{side(shape)}:{side(stride)}"


_SEPARABLE_QK_LINEAR = ("#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 64]], "
                        "lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], "
                        "warp = [[32, 0], [64, 0], [0, 32]], block = []}>")


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_layout_shape_stride_maps_to_linear():
    """A shape/stride `tlx.layout` passed to `local_load(layout=...)` lowers to
    the expected #linear encoding (no register/lane/warp on the user surface)."""

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        qk = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(qk, 0)
        x = tlx.local_load(v, layout=LAYOUT)
        tlx.local_store(v, x)

    # 3 warp bases -> 2**3 = 8 warps; the layout requires num_warps == 8.
    compiled = kernel.warmup(_separable_qk_layout(), grid=(1, ), num_warps=8)
    ttgir = compiled.asm["ttgir"]
    assert "no_verify_layout" not in ttgir
    assert _SEPARABLE_QK_LINEAR in ttgir


@pytest.mark.skipif(not is_cuda(), reason="Need CUDA")
def test_dump_layout_cute(capfd, monkeypatch):
    """`tlx.dump_layout` prints the resolved layout in CuTe Shape:Stride form to
    the compiler log and is erased from the IR."""
    # The diagnostic prints during compilation, so force a (re)compile instead
    # of hitting the on-disk kernel cache.
    monkeypatch.setattr(triton.knobs.compilation, "always_compile", True)

    @triton.jit
    def kernel(BLOCK: tl.constexpr):
        x = tl.arange(0, BLOCK)  # register tensor
        tlx.dump_layout(x)
        buf = tlx.local_alloc((BLOCK, ), tl.int32, tl.constexpr(1))  # SMEM buffer
        v = tlx.local_view(buf, 0)
        tlx.local_store(v, x)
        tlx.dump_layout(v)

    compiled = kernel.warmup(64, grid=(1, ), num_warps=4)
    err = capfd.readouterr().err

    # Register tensor -> CuTe thread-value layout.
    assert "tlx.dump_layout" in err
    assert "cute: ((_32,_2,_2),_1):((_1,_32,_0),_0)" in err
    # SMEM buffer -> single strided CuTe layout.
    assert "cute: _64:_1" in err
    # The diagnostic ops are consumed (erased) and never reach the final IR.
    assert "tlx.dump_layout" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_dump_layout_round_trips_shape_stride(capfd, monkeypatch):
    """Dumping a tensor that carries the separable `tlx.layout` reproduces the
    same CuTe (thread, value) shape/stride it was built from."""
    monkeypatch.setattr(triton.knobs.compilation, "always_compile", True)

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        qk = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(qk, 0)
        x = tlx.local_load(v, layout=LAYOUT)
        tlx.dump_layout(x)
        tlx.local_store(v, x)

    kernel.warmup(_separable_qk_layout(), grid=(1, ), num_warps=8)
    err = capfd.readouterr().err
    # The dumped layout is exactly the (thread, value) shape/stride the tensor
    # was built from -> it round-trips through the compiler.
    expected = _cute_shape_stride(_SEPARABLE_QK_SHAPE, _SEPARABLE_QK_STRIDE)
    assert f"cute: {expected}" in err


def test_swizzled_layout_cute_mapping():
    """`tlx.swizzled_layout(B, M, S)` is the CuTe Swizzle<B,M,S> (positional args).
    It resolves to Triton's (vec, perPhase, maxPhase) for a given contiguous extent,
    per the inverse of DumpLayout's emitCuteSwizzle. Pure-Python, no GPU."""

    # vec = 2**M, maxPhase = 2**B, perPhase = 2**(S+M) // numContig.
    # Mirror the SwizzledSharedEncoding doc examples (order=[1,0], numContig = shape[1]):
    #   vec=1, perPhase=1, maxPhase=4 over a width-4 tile -> Swizzle<2,0,2>
    enc = tlx.swizzled_layout(2, 0, 2, order=[1, 0])._to_encoding(shape=[4, 4])
    assert (enc.vectorSize, enc.perPhase, enc.maxPhase) == (1, 1, 4)
    #   vec=1, perPhase=2, maxPhase=4 over a width-4 tile -> Swizzle<2,0,3>
    enc = tlx.swizzled_layout(2, 0, 3, order=[1, 0])._to_encoding(shape=[4, 4])
    assert (enc.vectorSize, enc.perPhase, enc.maxPhase) == (1, 2, 4)
    #   vec=2, perPhase=1, maxPhase=4 over a width-8 tile -> Swizzle<2,1,2>
    enc = tlx.swizzled_layout(2, 1, 2, order=[1, 0])._to_encoding(shape=[4, 8])
    assert (enc.vectorSize, enc.perPhase, enc.maxPhase) == (2, 1, 4)

    # A Swizzle that would give perPhase < 1 for the extent is rejected.
    with pytest.raises(AssertionError):
        tlx.swizzled_layout(2, 0, 0, order=[1, 0])._to_encoding(shape=[4, 8])


def test_swizzled_layout_vs_register_layout():
    """`swizzled_layout` is a shared-memory layout; the shape/stride `tlx.layout`
    stays a register layout. `tlx.layout(swizzled_layout(...))` also accepts one
    (eagerly resolving the trivial default). Pure-Python, no GPU."""

    # The no-swizzle default is shape-independent -> tlx.layout resolves it eagerly.
    a = tlx.layout(tlx.swizzled_layout.make_default(rank=2))
    assert type(a) is tlx.swizzled_shared_layout_encoding  # exact type -> `type() is` checks hold
    assert (a.vectorSize, a.perPhase, a.maxPhase, a.order) == (1, 1, 1, [1, 0])

    # A real swizzle is deferred (needs the buffer shape): tlx.layout returns it as-is.
    atom = tlx.layout(tlx.swizzled_layout(2, 0, 2, order=[1, 0]))
    assert isinstance(atom, tlx.swizzled_layout)

    # the shape/stride form is unchanged: a register layout
    r = _separable_qk_layout()
    assert isinstance(r, tlx.layout) and not isinstance(r, tlx.shared_layout_encoding)

    # tlx.layout() with neither a swizzled_layout nor shape/stride is rejected
    with pytest.raises(AssertionError):
        tlx.layout()


@pytest.mark.skipif(not is_cuda(), reason="Need CUDA")
def test_swizzled_layout_lowers_to_swizzled_shared():
    """`tlx.swizzled_layout(...)` used directly as a `local_alloc` layout lowers to
    the `#ttg.swizzled_shared` encoding. The trivial default matches the legacy
    `swizzled_shared_layout_encoding` byte-for-byte; a real Swizzle<B,M,S> resolves
    its perPhase from the buffer shape."""

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        x = tl.zeros((128, 64), tl.float16)
        buf = tlx.local_alloc((128, 64), tl.float16, tl.constexpr(1), layout=LAYOUT)
        v = tlx.local_view(buf, 0)
        tlx.local_store(v, x)

    # Trivial swizzled_layout == constructing the legacy encoding directly.
    cute = tlx.swizzled_layout.make_default(rank=2)
    direct = tlx.swizzled_shared_layout_encoding.make_default(rank=2)
    ttgir_cute = kernel.warmup(cute, grid=(1, ), num_warps=4).asm["ttgir"]
    ttgir_direct = kernel.warmup(direct, grid=(1, ), num_warps=4).asm["ttgir"]
    assert "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>" in ttgir_cute
    assert ttgir_cute == ttgir_direct

    # A real Swizzle<3,0,6> over a width-64 tile -> vec=1, maxPhase=8,
    # perPhase = 2**(6+0)//64 = 1.
    swz = kernel.warmup(tlx.swizzled_layout(3, 0, 6, order=[1, 0]), grid=(1, ), num_warps=4).asm["ttgir"]
    assert "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>" in swz


# ---------------------------------------------------------------------------
# "Does the compiler understand user layouts?" -- adversarial end-to-end cases.
#
# Every user-pinned layout rides through the pipeline as #tlx.user_layout<...>
# and must (a) survive to the final TTGIR as the exact concrete layout the user
# asked for, and (b) leave no #tlx.user_layout / no_verify_layout residue.
# ---------------------------------------------------------------------------


def _assert_no_layout_residue(ttgir):
    # Match the *encoding* form (#tlx.user_layout<...>) specifically: the TMEM
    # register-layout path sets an unrelated op attribute literally named
    # `tlx.user_layout` (see triton_tlx.cc), which must not trip this check.
    assert "#tlx.user_layout" not in ttgir, "user-layout wrapper encoding leaked into final IR"
    assert "#tlx.no_verify_layout" not in ttgir, "no-verify wrapper encoding leaked into final IR"


@pytest.mark.skipif(not is_cuda(), reason="Need CUDA")
def test_user_shared_layout_survives_readback():
    """Start from the read side: alloc with a user swizzle, write, then read it
    back. The buffer's exact user swizzle must survive to final TTGIR."""

    @triton.jit
    def kernel(SW: tl.constexpr):
        x = tl.zeros((128, 64), tl.float16)
        buf = tlx.local_alloc((128, 64), tl.float16, tl.constexpr(1), layout=SW)
        v = tlx.local_view(buf, 0)
        tlx.local_store(v, x)
        y = tlx.local_load(v)  # read back
        tlx.local_store(v, y)

    # Swizzle<3,0,6> over width-64 -> vec=1, perPhase=1, maxPhase=8.
    ttgir = kernel.warmup(tlx.swizzled_layout(3, 0, 6, order=[1, 0]), grid=(1, ), num_warps=4).asm["ttgir"]
    assert "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>" in ttgir
    _assert_no_layout_residue(ttgir)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell (num_warps=8 register layout + TMEM)")
def test_user_shared_and_register_layouts_coexist():
    """The compiler must honor BOTH a user shared layout (swizzle, on an SMEM
    alloc) and a user register layout (#linear, on a TMEM read) in one kernel:
    both the swizzled_shared and the #linear appear in final TTGIR.

    (The register layout is pinned via a TMEM read -- the SMEM read path lets
    RemoveLayoutConversions relax the register layout when the only consumer is a
    layout-flexible store, so it wouldn't be a reliable probe on its own.)"""

    @triton.jit
    def kernel(SW: tl.constexpr, REG: tl.constexpr):
        # user shared layout on an SMEM buffer, read back
        x = tl.zeros((128, 64), tl.float16)
        sbuf = tlx.local_alloc((128, 64), tl.float16, tl.constexpr(1), layout=SW)
        sv = tlx.local_view(sbuf, 0)
        tlx.local_store(sv, x)
        s = tlx.local_load(sv)
        # user register layout on a TMEM read
        qk = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        qv = tlx.local_view(qk, 0)
        r = tlx.local_load(qv, layout=REG)
        tlx.local_store(qv, r)
        tlx.local_store(sv, s)

    # Swizzle<3,0,6> over width-64 -> vec=1, perPhase=1, maxPhase=8.
    sw = tlx.swizzled_layout(3, 0, 6, order=[1, 0])
    ttgir = kernel.warmup(sw, _separable_qk_layout(), grid=(1, ), num_warps=8).asm["ttgir"]
    assert "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>" in ttgir
    assert _SEPARABLE_QK_LINEAR in ttgir
    _assert_no_layout_residue(ttgir)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell (num_warps=8 register layout)")
def test_user_register_layout_anchored_on_smem():
    """A user register layout on an SMEM read is anchored end-to-end even when the
    only consumer is a layout-flexible store: the #linear survives all the layout
    passes (coalesce, remove-layout-conversions, ...). Regression for the case
    where the load was previously relaxed to #blocked."""

    @triton.jit
    def kernel(REG: tl.constexpr):
        x = tl.zeros((128, 128), tl.float16)
        buf = tlx.local_alloc((128, 128), tl.float16, tl.constexpr(1))
        v = tlx.local_view(buf, 0)
        tlx.local_store(v, x)
        y = tlx.local_load(v, layout=REG)  # only consumer is a flexible store
        tlx.local_store(v, y)

    ttgir = kernel.warmup(_separable_qk_layout(), grid=(1, ), num_warps=8).asm["ttgir"]
    assert _SEPARABLE_QK_LINEAR in ttgir
    _assert_no_layout_residue(ttgir)


@pytest.mark.skipif(not is_cuda(), reason="Need CUDA")
def test_user_padded_shared_layout_survives():
    """The wrapper is general across the shared family, not just swizzled: a
    user-pinned padded_shared layout must survive as #ttg.padded_shared."""

    @triton.jit
    def kernel(PAD: tl.constexpr):
        x = tl.zeros((128, 64), tl.float16)
        buf = tlx.local_alloc((128, 64), tl.float16, tl.constexpr(1), layout=PAD)
        v = tlx.local_view(buf, 0)
        tlx.local_store(v, x)
        y = tlx.local_load(v)
        tlx.local_store(v, y)

    pad = tlx.padded_shared_layout_encoding.with_identity_for([(64, 8)], [128, 64], [1, 0])
    ttgir = kernel.warmup(pad, grid=(1, ), num_warps=4).asm["ttgir"]
    assert "#ttg.padded_shared" in ttgir
    _assert_no_layout_residue(ttgir)


@pytest.mark.skipif(not is_cuda(), reason="Need CUDA")
def test_user_shared_layout_multibuffer_views():
    """A user swizzle on a multi-buffered alloc survives across several local_view
    subviews that are each read back."""

    @triton.jit
    def kernel(SW: tl.constexpr):
        x = tl.zeros((128, 64), tl.float16)
        buf = tlx.local_alloc((128, 64), tl.float16, tl.constexpr(2), layout=SW)
        v0 = tlx.local_view(buf, 0)
        v1 = tlx.local_view(buf, 1)
        tlx.local_store(v0, x)
        tlx.local_store(v1, x)
        a = tlx.local_load(v0)
        b = tlx.local_load(v1)
        tlx.local_store(v0, a + b)

    ttgir = kernel.warmup(tlx.swizzled_layout(3, 0, 6, order=[1, 0]), grid=(1, ), num_warps=4).asm["ttgir"]
    assert "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>" in ttgir
    _assert_no_layout_residue(ttgir)


@pytest.mark.skipif(not is_cuda(), reason="Need CUDA")
def test_user_shared_layout_in_loop():
    """A user-pinned buffer read inside a loop keeps its swizzle (adversarial:
    loop-carried IV / region-carried values must not drop the wrapper)."""

    @triton.jit
    def kernel(SW: tl.constexpr, N: tl.constexpr):
        x = tl.zeros((128, 64), tl.float16)
        buf = tlx.local_alloc((128, 64), tl.float16, tl.constexpr(1), layout=SW)
        v = tlx.local_view(buf, 0)
        tlx.local_store(v, x)
        acc = tl.zeros((128, 64), tl.float16)
        for _ in range(N):
            acc += tlx.local_load(v)
        tlx.local_store(v, acc)

    ttgir = kernel.warmup(tlx.swizzled_layout(3, 0, 6, order=[1, 0]), 4, grid=(1, ), num_warps=4).asm["ttgir"]
    assert "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>" in ttgir
    _assert_no_layout_residue(ttgir)


# ---------------------------------------------------------------------------
# User-pinned swizzled shared + linear (offset) layouts:
# padded_shared_layout_encoding.with_bases, distributed_linear_layout, and
# buffer_load_to_local(offset_layout=...). The constants below are the exact
# Gluon a16w16 swizzles the intra_wave/a16w16 kernel pins (tile [HALF_M=128,
# BLOCK_K=64]) to clear CDNA4 LDS bank conflicts, so these tests double as
# documentation of that known-good layout.
# ---------------------------------------------------------------------------

_A16W16_SHARED_INTERVALS = [(512, 16)]
_A16W16_SHARED_OFFSET_BASES = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0],
                               [2, 0], [4, 0], [8, 0]]
_A16W16_TILE = [128, 64]
_A16W16_LOAD_REG = [[0, 1], [0, 2], [0, 4], [8, 0]]
_A16W16_LOAD_LANE = [[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]]
_A16W16_LOAD_WARP = [[1, 0], [2, 0], [4, 0]]


def test_with_bases_builds_swizzled_padded_encoding():
    """`padded_shared_layout_encoding.with_bases` records the explicit linear
    (offset) component instead of the identity {order, shape}. Pure-Python."""
    enc = tlx.padded_shared_layout_encoding.with_bases(_A16W16_SHARED_INTERVALS, _A16W16_SHARED_OFFSET_BASES,
                                                       _A16W16_TILE)
    assert enc.intervals == [512]
    assert enc.paddings == [16]
    assert enc.order == [1, 0]  # reversed(range(rank))
    assert enc.offset_bases == _A16W16_SHARED_OFFSET_BASES
    assert enc.block_bases == []
    assert enc.shape == _A16W16_TILE


def test_distributed_linear_layout_construction():
    """`distributed_linear_layout` keeps the explicit reg/lane/warp bases (the
    Gluon DistributedLinearLayout analogue). Pure-Python."""
    layout = tlx.distributed_linear_layout(reg_bases=_A16W16_LOAD_REG, lane_bases=_A16W16_LOAD_LANE,
                                           warp_bases=_A16W16_LOAD_WARP, shape=_A16W16_TILE)
    assert layout.reg_bases == _A16W16_LOAD_REG
    assert layout.lane_bases == _A16W16_LOAD_LANE
    assert layout.warp_bases == _A16W16_LOAD_WARP
    assert layout.block_bases == []
    assert layout.shape == _A16W16_TILE
    assert "distributed_linear_layout" in repr(layout)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_user_pinned_swizzled_padded_survives_amd():
    """A user-pinned *swizzled* padded_shared (built with `with_bases`) survives
    to final TTGIR as #ttg.padded_shared with the explicit {offset = ...} form,
    not the identity {order, shape}, and leaves no #tlx.user_layout residue."""

    @triton.jit
    def kernel(PAD: tl.constexpr, M: tl.constexpr, K: tl.constexpr):
        x = tl.zeros((M, K), tl.float16)
        buf = tlx.local_alloc((M, K), tl.float16, tl.constexpr(1), layout=PAD)
        v = tlx.local_view(buf, 0)
        tlx.local_store(v, x)
        y = tlx.local_load(v)
        tlx.local_store(v, y)

    pad = tlx.padded_shared_layout_encoding.with_bases(_A16W16_SHARED_INTERVALS, _A16W16_SHARED_OFFSET_BASES,
                                                       _A16W16_TILE)
    ttgir = kernel.warmup(pad, _A16W16_TILE[0], _A16W16_TILE[1], grid=(1, ), num_warps=8).asm["ttgir"]
    assert "#ttg.padded_shared" in ttgir
    assert "offset = [" in ttgir  # the explicit bases form, not {order, shape}
    _assert_no_layout_residue(ttgir)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_buffer_load_to_local_offset_layout_amd():
    """buffer_load_to_local(offset_layout=distributed_linear_layout(...)) pins
    the offset tensor's #linear so the direct-to-LDS load matches the pinned
    swizzled shared layout and lowers all the way to amdgcn."""

    @triton.jit
    def kernel(a_ptr, SHARED: tl.constexpr, LOAD: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
               STRIDE_M: tl.constexpr):
        offs_m = tl.arange(0, M)
        offs_k = tl.arange(0, K)
        off = offs_m[:, None] * STRIDE_M + offs_k[None, :]
        smem = tlx.local_alloc((M, K), tl.float16, tl.constexpr(1), layout=SHARED)
        tlx.buffer_load_to_local(smem[0], a_ptr, off, offset_layout=LOAD)

    pad = tlx.padded_shared_layout_encoding.with_bases(_A16W16_SHARED_INTERVALS, _A16W16_SHARED_OFFSET_BASES,
                                                       _A16W16_TILE)
    load = tlx.distributed_linear_layout(reg_bases=_A16W16_LOAD_REG, lane_bases=_A16W16_LOAD_LANE,
                                         warp_bases=_A16W16_LOAD_WARP, shape=_A16W16_TILE)
    M, K = _A16W16_TILE
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    compiled = kernel.warmup(a, pad, load, M, K, K, grid=(1, ), num_warps=8)
    ttgir = compiled.asm["ttgir"]
    # The offset tensor is pinned to the given #linear (so the direct-to-LDS
    # write can match the swizzled shared layout), and the load stays a single
    # direct-to-LDS `amdg.buffer_load_to_local`.
    assert "#ttg.linear" in ttgir
    assert "amdg.buffer_load_to_local" in ttgir
    # It lowers all the way to amdgcn (the direct-to-LDS width/alignment
    # requirements are met by the pinned offset layout).
    assert compiled.asm.get("amdgcn")
