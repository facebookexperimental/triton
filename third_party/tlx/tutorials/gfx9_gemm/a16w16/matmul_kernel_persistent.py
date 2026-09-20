"""Compatibility entry point for the gfx950 persistent GEMM tutorial."""

from triton.tlx.ops.kernels.mm.gfx950 import (
    _C_STORE_32X128_LAYOUT,
    _commit_accumulators,
    _compute_full_tile,
    _global_loads,
    _launch_persistent as matmul,
    _local_alloc_stage,
    _make_global_tile_load_addresses,
    _persistent_supports as supports,
)

__all__ = ["matmul", "supports"]
