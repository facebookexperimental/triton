"""Compile-time check that an explicit non-padded layout on a TDM buffer is rejected.

TDM hardware verifiers and AMD descriptor layout assignment both require a
padded shared encoding on the alloc. The Python `tlx.local_alloc(layout=...)`
builder tags the alloc with `tlx.layout_is_explicit`, and
`TLXInsertRequireLayout` then emits a hard error so a user-supplied
incompatible layout fails loud instead of being silently overridden.
"""
import pytest

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.constexpr_function
def _swizzled_layout():
    return tlx.swizzled_shared_layout_encoding(vectorSize=1, perPhase=1, maxPhase=1, order=[1, 0], numCTAs=[1, 1],
                                               numCTAsPerCGA=[1, 1], numCTASplit=[1, 1], numCTAOrder=[1, 0])


@triton.constexpr_function
def _padded_layout(block_shape):
    return tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], block_shape, [1, 0])


@triton.jit
def _explicit_swizzled_kernel(a_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, tl.constexpr(1)],
                                       block_shape=[BLOCK_M, BLOCK_K])
    layout: tl.constexpr = _swizzled_layout()
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1, layout=layout)
    tlx.async_amd_descriptor_load(a_desc, a_buf, [0, 0])


@triton.jit
def _explicit_padded_kernel(a_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, tl.constexpr(1)],
                                       block_shape=[BLOCK_M, BLOCK_K])
    layout: tl.constexpr = _padded_layout([BLOCK_M, BLOCK_K])
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1, layout=layout)
    tlx.async_amd_descriptor_load(a_desc, a_buf, [0, 0])


@triton.jit
def _default_layout_kernel(a_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, tl.constexpr(1)],
                                       block_shape=[BLOCK_M, BLOCK_K])
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1)
    tlx.async_amd_descriptor_load(a_desc, a_buf, [0, 0])


def _compile(fn):
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile
    src = ASTSource(
        fn=fn,
        signature={"a_ptr": "*fp16"},
        constexprs={"M": 128, "K": 128, "BLOCK_M": 32, "BLOCK_K": 128},
    )
    return triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))


def test_explicit_swizzled_layout_on_tdm_buffer_errors(capfd):
    with pytest.raises(Exception):
        _compile(_explicit_swizzled_kernel)
    err = capfd.readouterr().err
    assert "TDM operand requires a padded shared encoding" in err, err


def test_explicit_padded_layout_on_tdm_buffer_compiles():
    _compile(_explicit_padded_kernel)


def test_default_layout_on_tdm_buffer_compiles():
    _compile(_default_layout_kernel)
