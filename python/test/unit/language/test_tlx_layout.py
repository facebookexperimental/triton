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


@triton.jit
def _pinned_row_max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def _pinned_add_combine(a, b):
    return a + b


@triton.jit
def _pinned_fma_helper(a, b, c):
    # A @triton.jit helper -> tt.call. When called with pinned (placeholder)
    # args whose result is consumed downstream, TritonTLXFixup specializes the
    # monomorphized callee (params + return + FunctionType) to the placeholder.
    return a * b + c


# Row-per-thread layout for a [128,128] tile with 4 warps: each thread (lane l,
# warp w) owns row w*32+l with all 128 columns in registers (matches the HSTU
# forward softmax kernel's pinned QK layout).
def _row_per_thread_layout():
    return tlx.layout(shape=((32, 4), (128, )), stride=((128, 4096), (1, )))


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_pinned_layout_propagates_through_elementwise():
    """A pinned (no_verify) register layout that feeds arith/math elementwise
    ops (where / add / sub / exp2) and a tl.reduce alongside default-layout
    siblings must still compile.

    arith/math verifiers use MLIR's generic SameOperandsAndResultType check,
    which compares tensor encodings literally and ignores #tlx.no_verify_layout
    (unlike triton ops, whose DialectInferLayoutInterface honors it). The
    make_ttir TritonTLXFixup pass propagates the placeholder across these ops
    (elementwise, select condition via require_layout, and scf region-carried
    values) so the module verifies; the concrete layout is resolved later.
    Before that fixup this raised: 'arith.addf' op requires the same encoding
    for all operands and results.

    Note: reductions must use tl.reduce (a direct tt.reduce, which honors
    no_verify), not tl.max/tl.sum (which lower to a tt.call whose param is
    null-encoded and would reject the pinned operand).
    """

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(buf, 0)
        x = tlx.local_load(v, layout=LAYOUT)  # pinned no_verify<#linear>
        # Position mask built in the default layout, mixed into the pinned
        # tensor via arith.select (tl.where) + arith.addf. The select condition
        # (default-layout i1) is converted with require_layout; true/false/result
        # take the placeholder.
        offs = tl.arange(0, 128)
        mask = offs[:, None] >= offs[None, :]
        x = x + tl.where(mask, 0.0, -float("inf"))
        # Thread-local reduce (direct tt.reduce), broadcast back (arith.subf),
        # then math.exp2.
        m = tl.reduce(x, 1, _pinned_row_max_combine)
        p = tl.math.exp2(x - m[:, None])
        tlx.local_store(v, p)

    compiled = kernel.warmup(_row_per_thread_layout(), grid=(1, ), num_warps=4)
    # Compiled successfully and the placeholder was fully resolved downstream.
    assert "no_verify_layout" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_pinned_reduce_is_thread_local():
    """tl.reduce over axis=1 of a pinned row-per-thread layout keeps the pinned
    #linear as the slice parent (each thread owns a full row -> the reduce is
    thread-local, no cross-lane shuffle). Both a max and a sum reduce compile."""

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(buf, 0)
        x = tlx.local_load(v, layout=LAYOUT)
        m = tl.reduce(x, 1, _pinned_row_max_combine)
        p = tl.math.exp2(x - m[:, None])
        s = tl.reduce(p, 1, _pinned_add_combine)
        p = p * s[:, None]
        tlx.local_store(v, p)

    compiled = kernel.warmup(_row_per_thread_layout(), grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    assert "no_verify_layout" not in ttgir
    # Reduces are over axis 1 (N), i.e. thread-local for the row-per-thread pin.
    assert "axis = 1" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_pinned_propagates_through_jit_call():
    """A pinned tensor fed to a @triton.jit helper (which lowers to a tt.call)
    whose result is consumed by arith compiles. Triton monomorphizes the callee
    with an encoding-stripped signature, so TritonTLXFixup specializes the
    callee's params, return operand and FunctionType (and nested calls) to the
    placeholder to keep the CallOpInterface contract."""

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(buf, 0)
        x = tlx.local_load(v, layout=LAYOUT)
        y = _pinned_fma_helper(x, x, x)  # tt.call with pinned args
        z = tl.math.exp2(y)  # arith consumes the (pinned) call result
        tlx.local_store(v, z)

    compiled = kernel.warmup(_row_per_thread_layout(), grid=(1, ), num_warps=4)
    assert "no_verify_layout" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_pinned_scf_loop_carried_reduce():
    """The online-softmax running-max pattern: a loop-carried accumulator updated
    from a reduce over a pinned tensor. TritonTLXFixup propagates the placeholder
    through scf.for init / region-iter-arg / yield / result."""

    @triton.jit
    def kernel(LAYOUT: tl.constexpr, N: tl.constexpr):
        buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(buf, 0)
        x = tlx.local_load(v, layout=LAYOUT)
        m = tl.zeros([128], dtype=tl.float32) - float("inf")
        for _ in tl.range(0, N):
            m = tl.maximum(m, tl.reduce(x, 1, _pinned_row_max_combine))
        p = tl.math.exp2(x - m[:, None])
        tlx.local_store(v, p)

    compiled = kernel.warmup(_row_per_thread_layout(), 4, grid=(1, ), num_warps=4)
    assert "no_verify_layout" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_pinned_scf_loop_carried_const_init():
    """A loop-carried accumulator initialized directly from a bare constant
    (tl.zeros) and updated from a pinned tensor. The fixup must bridge the
    constant init with require_layout (retyping it in place would corrupt the
    constant's value attr once resolve strips the wrapper) while retyping the
    loop's own iter-arg / result."""

    @triton.jit
    def kernel(LAYOUT: tl.constexpr, N: tl.constexpr):
        buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(buf, 0)
        x = tlx.local_load(v, layout=LAYOUT)
        acc = tl.zeros([128, 128], dtype=tl.float32)  # bare constant loop init
        for _ in tl.range(0, N):
            acc = acc + x  # pinned tensor combined in-loop
        tlx.local_store(v, acc)

    compiled = kernel.warmup(_row_per_thread_layout(), 4, grid=(1, ), num_warps=4)
    assert "no_verify_layout" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_pinned_softmax_end_to_end():
    """End-to-end mirror of the HSTU forward softmax island: pinned load -> mask
    (select+add) -> thread-local reduce -> fma helper (tt.call) -> exp2 -> row sum
    -> restructuring helper (reshape/split, auto-released) -> store. Exercises
    every propagation/relaxation path together, with no explicit release."""

    @triton.jit
    def _restructure_tail(p):
        a, b = p.reshape([128, 2, 64]).permute(0, 2, 1).split()
        return tl.join(a, b).reshape([128, 128])

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(buf, 0)
        x = tlx.local_load(v, layout=LAYOUT)
        offs = tl.arange(0, 128)
        mask = offs[:, None] >= offs[None, :]
        x = x + tl.where(mask, 0.0, -float("inf"))
        m = tl.reduce(x, 1, _pinned_row_max_combine)
        x = _pinned_fma_helper(x, 1.4426950408, -m[:, None])  # qk*scale - m
        p = tl.math.exp2(x)
        l = tl.reduce(p, 1, _pinned_add_combine)
        p = p * l[:, None]
        # Restructuring helper: the fixup auto-releases the pin on the call arg.
        y = _restructure_tail(p)
        tlx.local_store(v, y)

    compiled = kernel.warmup(_row_per_thread_layout(), grid=(1, ), num_warps=4)
    assert "no_verify_layout" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_pinned_reduction_through_tl_max_sum_call():
    """tl.max / tl.sum lower to a tt.call (standard.max / standard.sum) whose
    monomorphized signature is encoding-stripped. With a pinned operand,
    TritonTLXFixup specializes that reduction callee (params + the
    fixpoint-inferred slice-of-pin return), so the natural tl.max / tl.sum
    compile on a pinned tensor -- no tl.reduce / explicit combine needed.

    This is the compiler-side alternative to a lit test: the pre-specialization
    IR (a tt.call whose pinned operand does not match the null-encoded callee
    param) cannot be parsed by triton-opt (CallOp verifies operand==param at
    parse), so the reduction-callee specialization is exercised through warmup.
    """

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(buf, 0)
        x = tlx.local_load(v, layout=LAYOUT)
        m = tl.max(x, 1)  # -> tt.call standard.max, pinned operand
        p = tl.math.exp2(x - m[:, None])
        s = tl.sum(p, 1)  # -> tt.call standard.sum, pinned operand
        p = p * s[:, None]
        tlx.local_store(v, p)

    compiled = kernel.warmup(_row_per_thread_layout(), grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    assert "no_verify_layout" not in ttgir
    # The reductions stay thread-local (over axis 1) after specialization.
    assert "axis = 1" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_pinned_auto_release_through_restructuring_call():
    """A pinned tensor fed to a @triton.jit helper that restructures it
    (reshape/permute/split, like subtile_ops._split_n_2D) compiles with no
    explicit release: TritonTLXFixup auto-inserts tlx.release_layout on the
    placeholder call args (the pin has no meaning across the restructure) and
    the tail runs in a compiler-chosen layout."""

    @triton.jit
    def _restructure_helper(x):
        a, b = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        return tl.join(a, b).reshape([x.shape[0], x.shape[1]])

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(buf, 0)
        x = tlx.local_load(v, layout=LAYOUT)
        x = tl.math.exp2(x)  # pinned arith
        y = _restructure_helper(x)  # tt.call, restructuring -> auto-release the arg
        tlx.local_store(v, y)

    compiled = kernel.warmup(_row_per_thread_layout(), grid=(1, ), num_warps=4)
    assert "no_verify_layout" not in compiled.asm["ttgir"]


@triton.jit
def _pinned_smem_store_helper(buf):
    x = tl.zeros((64, 128), tl.float32)
    tlx.local_store(buf[0], x)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_pinned_smem_layout_through_jit_call():
    """A shared-memory buffer allocated with an explicit layout
    (tlx.local_alloc(layout=...)) is a memdesc whose encoding is wrapped as
    #tlx.user_layout<#ttg.swizzled_shared<...>>. Passing it to a @triton.jit
    helper (tt.call) whose monomorphized param dropped the wrapper must still
    compile: TritonTLXFixup specializes the callee's memdesc param to the pinned
    layout so the call operand/param types match."""

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        buf = tlx.local_alloc((64, 128), tl.float32, tl.constexpr(2), layout=LAYOUT)
        _pinned_smem_store_helper(buf)

    # Compiles without a tt.call operand/param type mismatch.
    kernel.warmup(tlx.swizzled_layout(3, 0, 7, order=[1, 0]), grid=(1, ), num_warps=4)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_pinned_propagates_through_cast():
    """A cast (.to(dtype) -> arith.truncf / arith.extf) changes the element type
    but preserves shape and layout. TritonTLXFixup propagates the pinned encoding
    to the cast result (keeping the result's element type), so the arith cast
    verifier accepts operand and result as cast-compatible instead of rejecting
    the encoded-operand / unencoded-result mismatch."""

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(buf, 0)
        x = tlx.local_load(v, layout=LAYOUT)  # pinned f32
        y = (x * 2.0).to(tl.float16)  # arith.mulf (pinned) -> arith.truncf
        z = y.to(tl.float32)  # arith.extf back to f32, still pinned
        tlx.local_store(v, z)

    compiled = kernel.warmup(_row_per_thread_layout(), grid=(1, ), num_warps=4)
    assert "no_verify_layout" not in compiled.asm["ttgir"]


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
def test_assert_same_layout(monkeypatch):
    """`tlx.assert_same_layout` compares final layouts for both value/layout
    and value/value forms, then is erased after successful assertions."""
    monkeypatch.setattr(triton.knobs.compilation, "always_compile", True)

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        value = tlx.local_load(tlx.local_view(buf, 0), layout=LAYOUT)
        other = tlx.local_load(tlx.local_view(buf, 0), layout=LAYOUT)
        tlx.assert_same_layout(value, LAYOUT)
        tlx.assert_same_layout(value, other)
        tlx.local_store(tlx.local_view(buf, 0), value)

    compiled = kernel.warmup(_separable_qk_layout(), grid=(1, ), num_warps=8)
    assert "tlx.assert_same_layout" not in compiled.asm["ttgir"]


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


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_user_register_layout_smem_loop_carried_amd():
    """Alternating user/no-verify wrapper order must fully disappear on both
    sides of an AMD scf.for edge."""

    @triton.jit
    def kernel(REG: tl.constexpr, n):
        initial = tl.zeros((256, 8), tl.int8)
        buf = tlx.local_alloc((256, 8), tl.int8, tl.constexpr(1))
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, initial)
        value = tlx.local_load(view, layout=REG)
        for _ in tl.range(0, n, num_stages=1):
            value += 1
        tlx.local_store(view, value)

    reg = tlx.layout(
        shape=((32, 2, 4), (8, )),
        stride=((64, 1, 2), (8, )),
    )
    compiled = kernel.warmup(reg, 4, grid=(1, ), num_warps=4)
    _assert_no_layout_residue(compiled.asm["ttgir"])


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
# User-pinned swizzled shared layout (padded_shared_layout_encoding.with_bases)
# and the offset (register) layout inferred from it for buffer_load_to_local.
# The constants below are the exact Gluon a16w16 swizzles the inter_wave/a16w16
# kernel pins (tile [HALF_M=128, BLOCK_K=64]) to clear CDNA4 LDS bank conflicts,
# so these tests double as documentation of that known-good layout.
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
def test_buffer_load_to_local_infers_offset_layout_amd():
    """With only the swizzled shared layout pinned on the alloc (no explicit
    offset layout), tlx-insert-require-layout infers the matching offset
    tensor's #linear so the direct-to-LDS load coalesces and lowers to amdgcn.
    The inferred #linear must equal the hand-derived a16w16 load layout."""

    @triton.jit
    def kernel(a_ptr, SHARED: tl.constexpr, M: tl.constexpr, K: tl.constexpr, STRIDE_M: tl.constexpr):
        offs_m = tl.arange(0, M)
        offs_k = tl.arange(0, K)
        off = offs_m[:, None] * STRIDE_M + offs_k[None, :]
        smem = tlx.local_alloc((M, K), tl.float16, tl.constexpr(1), layout=SHARED)
        tlx.buffer_load_to_local(smem[0], a_ptr, off)

    pad = tlx.padded_shared_layout_encoding.with_bases(_A16W16_SHARED_INTERVALS, _A16W16_SHARED_OFFSET_BASES,
                                                       _A16W16_TILE)
    M, K = _A16W16_TILE
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    compiled = kernel.warmup(a, pad, M, K, K, grid=(1, ), num_warps=8)
    ttgir = compiled.asm["ttgir"]
    # The offset layout is inferred (not authored) and matches the hand-derived
    # a16w16 load layout, and the load stays a single direct-to-LDS op.
    expected = f"register = {_A16W16_LOAD_REG}, lane = {_A16W16_LOAD_LANE}, warp = {_A16W16_LOAD_WARP}"
    assert "#ttg.linear" in ttgir
    assert expected in ttgir, f"inferred offset layout mismatch; expected substring:\n{expected}\n\nttgir:\n{ttgir}"
    assert "amdg.buffer_load_to_local" in ttgir
    # It lowers all the way to amdgcn (the direct-to-LDS width/alignment
    # requirements are met by the inferred offset layout).
    assert compiled.asm.get("amdgcn")
