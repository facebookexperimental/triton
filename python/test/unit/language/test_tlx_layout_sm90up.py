"""TLX layout tests -- Hopper and newer."""
import pytest
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer
import triton.language.extra.tlx as tlx


def _assert_no_layout_residue(ttgir):
    # Match the *encoding* form (#tlx.user_layout<...>) specifically: the TMEM
    # register-layout path sets an unrelated op attribute literally named
    # `tlx.user_layout` (see triton_tlx.cc), which must not trip this check.
    assert "#tlx.user_layout" not in ttgir, "user-layout wrapper encoding leaked into final IR"
    assert "#tlx.no_verify_layout" not in ttgir, "no-verify wrapper encoding leaked into final IR"
    assert "ttg.require_layout" not in ttgir, "require_layout boundary leaked into final IR"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
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
