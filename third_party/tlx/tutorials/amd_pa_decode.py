"""Compatibility imports for the production AMD paged-decode operator.

New integrations should import from ``triton.language.extra.tlx.ops``.
"""

from triton.language.extra.tlx.ops.amd_pa_decode import (
    allocate_5d_kv_cache,
    allocate_pa_decode_workspace,
    build_inputs,
    can_use_pa_decode_tlx,
    get_num_splits,
    get_pa_decode_config,
    pack_5d_kv_cache,
    pa_decode_tlx,
    ref_decode,
    reshape_and_cache_5d,
    unpack_5d_kv_cache,
)

__all__ = [
    "allocate_5d_kv_cache",
    "allocate_pa_decode_workspace",
    "build_inputs",
    "can_use_pa_decode_tlx",
    "get_num_splits",
    "get_pa_decode_config",
    "pack_5d_kv_cache",
    "pa_decode_tlx",
    "ref_decode",
    "reshape_and_cache_5d",
    "unpack_5d_kv_cache",
]
