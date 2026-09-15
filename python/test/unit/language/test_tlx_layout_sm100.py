"""TLX layout tests -- Blackwell-only."""
import pytest
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
import triton.language.extra.tlx as tlx

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


def _row_per_thread_layout():
    return tlx.layout(shape=((32, 4), (128, )), stride=((128, 4096), (1, )))


def _column_per_thread_layout():
    return tlx.layout(shape=((32, 4), (128, )), stride=((1, 32), (128, )))


_COLUMN_PER_THREAD_LINEAR = ("#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], "
                             "lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], "
                             "warp = [[0, 32], [0, 64]], block = []}>")


@triton.jit
def _pinned_smem_store_helper(buf):
    x = tl.zeros((64, 128), tl.float32)
    tlx.local_store(buf[0], x)


def _assert_no_layout_residue(ttgir):
    # Match the *encoding* form (#tlx.user_layout<...>) specifically: the TMEM
    # register-layout path sets an unrelated op attribute literally named
    # `tlx.user_layout` (see triton_tlx.cc), which must not trip this check.
    assert "#tlx.user_layout" not in ttgir, "user-layout wrapper encoding leaked into final IR"
    assert "#tlx.no_verify_layout" not in ttgir, "no-verify wrapper encoding leaked into final IR"
    assert "ttg.require_layout" not in ttgir, "require_layout boundary leaked into final IR"


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
    -> restructuring helper (reshape/split, pin preserved) -> store. Exercises
    every propagation/inference path together, with no explicit release."""

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
        # The fixup specializes the helper and preserves the pin through it.
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
def test_pinned_preserved_through_restructuring_call():
    """A pinned tensor fed to a @triton.jit helper that restructures it
    (reshape/permute/split, like subtile_ops._split_n_2D) keeps its layout
    constraint. TritonTLXFixup specializes the helper signature and re-infers
    each result layout without inserting an implicit release."""

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
        y = _restructure_helper(x)
        tlx.local_store(v, y)

    compiled = kernel.warmup(_row_per_thread_layout(), grid=(1, ), num_warps=4)
    assert "tlx.release_layout" not in compiled.asm["ttir"]
    assert "no_verify_layout" not in compiled.asm["ttgir"]


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


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_require_layout_after_release_reanchors_layout():
    """A release boundary should permit a later explicit pin to establish a new
    hard layout anchor. The current layout-removal path drops that second pin
    when its only consumer is a layout-flexible local_store."""

    @triton.jit
    def kernel(ROW: tl.constexpr, COL: tl.constexpr):
        offs = tl.arange(0, 128)
        x = offs[:, None].to(tl.float32) + offs[None, :].to(tl.float32)
        row = tlx.require_layout(x, ROW)
        col = tlx.require_layout(tlx.release_layout(row), COL)
        out_buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1))
        tlx.local_store(tlx.local_view(out_buf, 0), col)

    compiled = kernel.warmup(_row_per_thread_layout(), _column_per_thread_layout(), grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    assert _COLUMN_PER_THREAD_LINEAR in ttgir
    _assert_no_layout_residue(ttgir)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_require_layout_after_release_reanchors_tmem_store():
    """A user pin that feeds a TMEM local_store must not be erased by the
    convert-layout cleanup before the required TMEM-compatible store layout."""

    @triton.jit
    def kernel(ROW: tl.constexpr, COL: tl.constexpr):
        offs = tl.arange(0, 128)
        x = offs[:, None].to(tl.float32) + offs[None, :].to(tl.float32)
        row = tlx.require_layout(x, ROW)
        store_value = tlx.require_layout(tlx.release_layout(row), COL)
        out_buf = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.local_store(tlx.local_view(out_buf, 0), store_value)

    compiled = kernel.warmup(_row_per_thread_layout(), _column_per_thread_layout(), grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    assert _COLUMN_PER_THREAD_LINEAR in ttgir
    _assert_no_layout_residue(ttgir)


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
