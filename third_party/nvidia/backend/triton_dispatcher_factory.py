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

_bridge_registered = False


def _load_module():
    """Return the cuda_utils module (contains _TritonDispatcher type)."""
    global _bridge_registered
    mod = driver.active.utils
    if not _bridge_registered:
        _bridge_registered = True
        try:
            from triton._C._torch_bridge import get_tensor_access_capsule
            mod.register_tensor_bridge(get_tensor_access_capsule())
        except (ImportError, AttributeError):
            pass  # Bridge not available, fallback to slow path
    return mod


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
                # TMA path: nvTmaDesc + shapes (i32) + strides (i64)
                flat_types.append("nvTmaDesc")
                for _ in range(ndim):
                    flat_types.append("i32")
                for _ in range(ndim):
                    flat_types.append("i64")
        else:
            flat_types.append(ty)

    return flat_types, tensordesc_info if tensordesc_info else None


class _TensorDescDispatcherWrapper:
    """Wrapper that expands host-side tensordesc args before calling the C dispatcher.

    Only used for host-side tensordescs (meta=None). TMA tensordescs are
    expanded directly in C via EXTRACTOR_TENSORDESC_INDEX (type code 15).
    """

    __slots__ = ("_c_dispatcher", "_tensordesc_info_map")

    def __init__(self, c_dispatcher, tensordesc_info):
        self._c_dispatcher = c_dispatcher
        # Map from position in schema args → (ndim, meta)
        self._tensordesc_info_map = {pos: (ndim, meta) for pos, ndim, meta in tensordesc_info}

    def __call__(self, grid_x, grid_y, grid_z, stream, *kernel_args):
        expanded = []
        for i, arg in enumerate(kernel_args):
            info = self._tensordesc_info_map.get(i)
            if info is not None:
                _, meta = info
                assert meta is None, "TMA tensordesc should be handled in C, not Python wrapper"
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
        A callable _TritonDispatcher instance.
        Call as: dispatcher(grid_x, grid_y, grid_z, stream, *kernel_args)
    """
    mod = _load_module()

    # Expand tensordesc types into flat component types for build_signature_metadata.
    flat_types, tensordesc_info = _expand_schema_arg_types(schema)
    if flat_types is None:
        return None
    sig_metadata = mod.build_signature_metadata(flat_types)
    arg_type_codes = list(sig_metadata)

    # Build tma_meta for C-level TMA expansion
    tma_meta_list = None
    if tensordesc_info is not None:
        tma_meta_list = []

        # Compute flat_pos for each tensordesc entry
        flat_pos_for_td = []
        current_flat_pos = 0
        td_iter = 0
        for arg_pos, arg in enumerate(schema["args"]):
            ty = arg["type"]
            if ty.startswith("tensordesc"):
                _, ndim_td, meta_td = tensordesc_info[td_iter]
                flat_pos_for_td.append(current_flat_pos)
                if meta_td is None:
                    current_flat_pos += 1 + 2 * ndim_td + 1 + ndim_td + ndim_td
                else:
                    current_flat_pos += 1 + ndim_td + ndim_td
                td_iter += 1
            else:
                current_flat_pos += 1

        from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST
        for td_idx, (arg_pos, ndim, meta) in enumerate(tensordesc_info):
            if meta is None:
                # Host-side tensordesc: keep Python wrapper path
                continue

            flat_pos = flat_pos_for_td[td_idx]
            # Replace type codes: 15 for desc, 16 for shadow shape/stride slots
            arg_type_codes[flat_pos] = 15  # EXTRACTOR_TENSORDESC_INDEX

            shape_indices = []
            stride_indices = []
            for j in range(ndim):
                shape_idx = flat_pos + 1 + j
                arg_type_codes[shape_idx] = 16  # EXTRACTOR_SKIP_INDEX
                shape_indices.append(shape_idx)
            for j in range(ndim):
                stride_idx = flat_pos + 1 + ndim + j
                arg_type_codes[stride_idx] = 16  # EXTRACTOR_SKIP_INDEX
                stride_indices.append(stride_idx)

            tma_meta_list.append({
                "swizzle": meta["swizzle"],
                "elem_size": meta["elem_size"],
                "elem_type": TMA_DTYPE_DEVICE_TO_HOST[meta["elem_type"]],
                "ndim": ndim,
                "fp4_padded": meta["fp4_padded"],
                "block_size": meta["block_size"],
                "shape_param_indices": shape_indices,
                "stride_param_indices": stride_indices,
            })

        # If all tensordescs are host-side (no TMA), fall back to Python wrapper
        if not tma_meta_list:
            tma_meta_list = None

    c_dispatcher = mod._TritonDispatcher(
        function=cu_function,
        num_warps=schema["num_warps"],
        num_ctas=schema["num_ctas"],
        shared_mem=schema["shared_mem"],
        launch_pdl=1 if schema.get("launch_pdl", False) else 0,
        launch_cooperative_grid=1 if schema.get("launch_cooperative_grid", False) else 0,
        launch_cluster=1 if schema.get("launch_cluster", False) else 0,
        arg_type_codes=tuple(arg_type_codes),
        has_global_scratch=schema.get("global_scratch_size", 0) > 0,
        has_profile_scratch=schema.get("profile_scratch_size", 0) > 0,
        global_scratch_size=schema.get("global_scratch_size", 0),
        global_scratch_align=schema.get("global_scratch_align", 1),
        profile_scratch_size=schema.get("profile_scratch_size", 0),
        profile_scratch_align=schema.get("profile_scratch_align", 1),
        allocator=_allocation._allocator,
        profile_allocator=_allocation._profile_allocator,
        tma_meta=tma_meta_list,
    )

    # If there are host-side tensordescs (no TMA meta) that still need Python expansion,
    # use the wrapper. If all tensordescs are TMA (handled in C), return raw dispatcher.
    has_host_side_td = tensordesc_info and any(meta is None for _, _, meta in tensordesc_info)
    if has_host_side_td:
        # Filter to only host-side tensordesc entries for the wrapper
        host_td_info = [(pos, ndim, meta) for pos, ndim, meta in tensordesc_info if meta is None]
        return _TensorDescDispatcherWrapper(c_dispatcher, host_td_info)
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
