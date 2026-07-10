import pytest
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell, is_cuda
import triton.language.extra.tlx as tlx

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
