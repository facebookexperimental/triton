# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Factory for _TritonDispatcher — full C dispatcher for Triton JIT.

Creates a _TritonDispatcher instance from a Level 0 metadata schema +
CUfunction handle.  The dispatcher pre-binds everything and uses vectorcall
for the entire dispatch: (grid_x, grid_y, grid_z, stream, *kernel_args).

The _TritonDispatcher type lives in driver.c (cuda_utils module) alongside
upstream's launchKernel — sharing extractors, type codes, and the
cuLaunchKernelEx handle.
"""

from __future__ import annotations

import re

from triton.runtime import _allocation
from triton.runtime.driver import driver


def _load_module():
    """Return the cuda_utils module (contains _TritonDispatcher type)."""
    return driver.active.utils


def _expand_schema_arg_types(schema):
    """Expand tensordesc types in the schema into flat component types.

    Returns (flat_types, tensordesc_info) where tensordesc_info is a list of
    (position_in_schema_args, ndim, meta) for each tensordesc arg, or None
    if there are no tensordesc args.
    """
    flat_types = []
    tensordesc_info = []
    tensordesc_meta = schema.get("tensordesc_meta") or []
    tensordesc_idx = 0

    for arg_pos, arg in enumerate(schema["args"]):
        ty = arg["type"]
        if ty.startswith("tensordesc"):
            meta = tensordesc_meta[tensordesc_idx] if tensordesc_idx < len(tensordesc_meta) else None
            match = re.match(r"tensordesc<([^[>]*)\[([^]]*)\]", ty)
            if match is None:
                raise ValueError(f"Cannot parse tensordesc type: {ty}")
            dtype = match.group(1)
            shape = match.group(2)
            ndim = shape.count(",") + 1
            tensordesc_info.append((arg_pos, ndim, meta))
            tensordesc_idx += 1

            if meta is None:
                # Host-side path: base pointer + shape + strides + padding flag
                flat_types.append("*" + dtype)
                for _ in range(2 * ndim):
                    flat_types.append("i64")
                flat_types.append("i1")
                # Host-side also appends shapes (i32) + strides (i64) as explicit kernel args
                for _ in range(ndim):
                    flat_types.append("i32")
                for _ in range(ndim):
                    flat_types.append("i64")
            else:
                # TMA path uses nvTmaDesc which the C dispatcher doesn't handle.
                return None, None
        else:
            flat_types.append(ty)

    return flat_types, tensordesc_info if tensordesc_info else None


class _TensorDescDispatcherWrapper:
    """Wrapper that expands tensordesc args before calling the C dispatcher."""

    __slots__ = ("_c_dispatcher", "_tensordesc_positions")

    def __init__(self, c_dispatcher, tensordesc_info):
        self._c_dispatcher = c_dispatcher
        # Only non-TMA tensordesc (meta=None) reaches here.
        self._tensordesc_positions = set(pos for pos, _, _ in tensordesc_info)

    def __call__(self, grid_x, grid_y, grid_z, stream, *kernel_args):
        expanded = []
        for i, arg in enumerate(kernel_args):
            if i in self._tensordesc_positions:
                # Host-side tensordesc: base ptr, shape, strides, padding, shape, strides
                expanded.append(arg.base)
                expanded.extend(arg.shape)
                expanded.extend(arg.strides)
                expanded.append(arg.padding == "nan")
                expanded.extend(arg.shape)
                expanded.extend(arg.strides)
            else:
                expanded.append(arg)
        return self._c_dispatcher(grid_x, grid_y, grid_z, stream, *expanded)


def make_triton_dispatcher(schema, cu_function: int):
    """Create a _TritonDispatcher from Level 0 schema + CUfunction handle.

    Args:
        schema: Level 0 launch metadata dict (from CompiledKernel.launch_metadata_schema)
        cu_function: CUfunction handle as uint64

    Returns:
        A callable _TritonDispatcher instance (or wrapper for tensordesc kernels).
        Call as: dispatcher(grid_x, grid_y, grid_z, stream, *kernel_args)
    """
    mod = _load_module()

    # Expand tensordesc types into flat component types for build_signature_metadata.
    # Returns None for tensordesc_info if TMA path is detected (unsupported by C dispatcher).
    flat_types, tensordesc_info = _expand_schema_arg_types(schema)
    if flat_types is None:
        return None
    sig_metadata = mod.build_signature_metadata(flat_types)
    arg_type_codes = tuple(sig_metadata)

    c_dispatcher = mod._TritonDispatcher(
        function=cu_function,
        num_warps=schema["num_warps"],
        num_ctas=schema["num_ctas"],
        shared_mem=schema["shared_mem"],
        launch_pdl=1 if schema.get("launch_pdl", False) else 0,
        launch_cooperative_grid=1 if schema.get("launch_cooperative_grid", False) else 0,
        launch_cluster=1 if schema.get("launch_cluster", False) else 0,
        arg_type_codes=arg_type_codes,
        has_global_scratch=schema.get("global_scratch_size", 0) > 0,
        has_profile_scratch=schema.get("profile_scratch_size", 0) > 0,
        global_scratch_size=schema.get("global_scratch_size", 0),
        global_scratch_align=schema.get("global_scratch_align", 1),
        profile_scratch_size=schema.get("profile_scratch_size", 0),
        profile_scratch_align=schema.get("profile_scratch_align", 1),
        allocator=_allocation._allocator,
        profile_allocator=_allocation._profile_allocator,
    )

    if tensordesc_info is not None:
        return _TensorDescDispatcherWrapper(c_dispatcher, tensordesc_info)
    return c_dispatcher


_fast_jit_type_cache = {}


def activate_fast_dispatch(jit_function, kernel):
    """Swap jit_function.__class__ to a C heap type with mp_subscript."""

    disp = getattr(kernel, '_dispatcher', None)
    if disp is None:
        return

    base_type = type(jit_function)
    if base_type not in _fast_jit_type_cache:
        mod = _load_module()
        fast_type = mod.create_fast_jit_type(base_type)
        _fast_jit_type_cache[base_type] = fast_type

    schema = getattr(kernel, 'launch_metadata_schema', None)
    num_args = len(schema["args"]) if schema else 0

    jit_function._runner_cache = {}
    jit_function._fast_dispatcher = disp
    jit_function._fast_num_args = num_args
    jit_function._fast_kernel = kernel
    jit_function._fast_get_stream = driver.active.get_current_stream
    jit_function._fast_get_device = driver.active.get_current_device

    jit_function.__class__ = _fast_jit_type_cache[base_type]
