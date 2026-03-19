"""Minimal TLX utility functions for the plugin."""

import triton.language.core as tl


@tl.builtin
def dtype_of(ptr, _semantic=None):
    """Returns the element dtype of a pointer."""
    src_ty = ptr.type
    assert isinstance(src_ty, tl.pointer_type), "Expected pointer type"
    return tl.constexpr(src_ty.element_ty)
