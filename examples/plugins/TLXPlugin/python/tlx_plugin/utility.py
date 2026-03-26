"""Minimal TLX utility functions for the plugin."""

import os
import sys

import triton.language.core as tl


def ensure_plugin_on_path():
    """Add the TLXPlugin python directory to sys.path so `import tlx_plugin` works."""
    plugin_python_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    if plugin_python_dir not in sys.path:
        sys.path.insert(0, plugin_python_dir)


@tl.builtin
def dtype_of(ptr, _semantic=None):
    """Returns the element dtype of a pointer."""
    src_ty = ptr.type
    assert isinstance(src_ty, tl.pointer_type), "Expected pointer type"
    return tl.constexpr(src_ty.element_ty)
