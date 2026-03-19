"""TLX Plugin — out-of-tree Python DSL for TLX memory ops.

Provides local_alloc, local_view, local_store, local_load as plugin-based
replacements for triton.language.extra.tlx memory operations.

Usage:
    import tlx_plugin as tlx
    buf = tlx.local_alloc((M, K), tl.float16, 2)
    view = tlx.local_view(buf, 0)
    tlx.local_store(view, tensor)
    result = tlx.local_load(view)
"""

from .types import (
    buffered_tensor,
    buffered_tensor_type,
    layout_encoding,
    nv_mma_shared_layout_encoding,
    shared_layout_encoding,
    storage_kind,
    swizzled_shared_layout_encoding,
)
from .mem_ops import (
    local_alloc,
    local_load,
    local_store,
    local_view,
)
from .utility import dtype_of
# from . import custom_stages

# from triton import knobs
# knobs.runtime.add_stages_inspection_hook = custom_stages.inspect_stages_hook

__all__ = [
    "buffered_tensor",
    "buffered_tensor_type",
    "layout_encoding",
    "nv_mma_shared_layout_encoding",
    "shared_layout_encoding",
    "storage_kind",
    "swizzled_shared_layout_encoding",
    "local_alloc",
    "local_load",
    "local_store",
    "local_view",
    "dtype_of",
]
