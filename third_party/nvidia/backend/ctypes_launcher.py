# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Pure-Python ctypes-based launcher for Triton CUDA kernels.

Replaces the C-compiled launcher with a Python implementation that uses ctypes
to call cuLaunchKernelEx directly. This eliminates the ~50s gcc compilation
step observed on CPU-constrained cluster environments.
"""

from __future__ import annotations

import ctypes
import re
import struct
from ctypes import c_int, c_uint, c_uint64, c_void_p

# ---------------------------------------------------------------------------
# CUDA driver types (mirrors cuda.h via ctypes)
# ---------------------------------------------------------------------------

CUresult = c_int
CUfunction = c_void_p
CUstream = c_void_p
CUdeviceptr = c_uint64

# CUlaunchAttribute and CUlaunchConfig structs
# See CUDA driver API docs for layout.

CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6
CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4
CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5
CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1


class CUlaunchAttributeValue_clusterDim(ctypes.Structure):
    _fields_ = [("x", c_uint), ("y", c_uint), ("z", c_uint)]


class CUlaunchAttributeValue(ctypes.Union):
    _fields_ = [
        ("value", c_int),
        ("clusterDim", CUlaunchAttributeValue_clusterDim),
        ("clusterSchedulingPolicyPreference", c_int),
        # pad to cover the full union size (64 bytes in CUDA headers)
        ("_pad", ctypes.c_char * 64),
    ]


class CUlaunchAttribute(ctypes.Structure):
    _fields_ = [
        ("id", c_int),
        ("_pad0", c_int),
        ("value", CUlaunchAttributeValue),
    ]


class CUlaunchConfig(ctypes.Structure):
    _fields_ = [
        ("gridDimX", c_uint),
        ("gridDimY", c_uint),
        ("gridDimZ", c_uint),
        ("blockDimX", c_uint),
        ("blockDimY", c_uint),
        ("blockDimZ", c_uint),
        ("sharedMemBytes", c_uint),
        ("hStream", CUstream),
        ("attrs", ctypes.POINTER(CUlaunchAttribute)),
        ("numAttrs", c_uint),
    ]


# ---------------------------------------------------------------------------
# Lazy-loaded CUDA driver handle
# ---------------------------------------------------------------------------

_libcuda = None
_cuLaunchKernelEx = None


def _get_cuLaunchKernelEx():
    global _libcuda, _cuLaunchKernelEx
    if _cuLaunchKernelEx is not None:
        return _cuLaunchKernelEx
    _libcuda = ctypes.CDLL("libcuda.so.1")
    _cuLaunchKernelEx = _libcuda.cuLaunchKernelEx
    _cuLaunchKernelEx.restype = CUresult
    _cuLaunchKernelEx.argtypes = [
        ctypes.POINTER(CUlaunchConfig),  # config
        CUfunction,  # f
        ctypes.POINTER(c_void_p),  # kernelParams
        ctypes.POINTER(c_void_p),  # extra
    ]
    return _cuLaunchKernelEx


_cuCtxGetCurrent = None
_cuDeviceGet = None
_cuDevicePrimaryCtxRetain = None
_cuCtxSetCurrent = None
_cuPointerGetAttribute = None


def _ensure_cuda_context():
    global _libcuda, _cuCtxGetCurrent, _cuDeviceGet, _cuDevicePrimaryCtxRetain, _cuCtxSetCurrent
    if _libcuda is None:
        _get_cuLaunchKernelEx()

    if _cuCtxGetCurrent is None:
        _cuCtxGetCurrent = _libcuda.cuCtxGetCurrent
        _cuCtxGetCurrent.restype = CUresult
        _cuCtxGetCurrent.argtypes = [ctypes.POINTER(c_void_p)]

        _cuDeviceGet = _libcuda.cuDeviceGet
        _cuDeviceGet.restype = CUresult
        _cuDeviceGet.argtypes = [ctypes.POINTER(c_int), c_int]

        _cuDevicePrimaryCtxRetain = _libcuda.cuDevicePrimaryCtxRetain
        _cuDevicePrimaryCtxRetain.restype = CUresult
        _cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(c_void_p), c_int]

        _cuCtxSetCurrent = _libcuda.cuCtxSetCurrent
        _cuCtxSetCurrent.restype = CUresult
        _cuCtxSetCurrent.argtypes = [c_void_p]

    pctx = c_void_p()
    _cuCtxGetCurrent(ctypes.byref(pctx))
    if not pctx.value:
        device = c_int()
        _cuDeviceGet(ctypes.byref(device), 0)
        _cuDevicePrimaryCtxRetain(ctypes.byref(pctx), device)
        _cuCtxSetCurrent(pctx)


def _init_pointer_validation():
    global _cuPointerGetAttribute, _libcuda
    if _cuPointerGetAttribute is not None:
        return
    if _libcuda is None:
        _get_cuLaunchKernelEx()
    _cuPointerGetAttribute = _libcuda.cuPointerGetAttribute
    _cuPointerGetAttribute.restype = CUresult
    _cuPointerGetAttribute.argtypes = [c_void_p, c_uint, CUdeviceptr]


# CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 2
_CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 2


def _get_device_pointer(obj, idx):
    """Extract a CUdeviceptr from a Python object (tensor, int, or None)."""
    if isinstance(obj, int):
        return obj
    if obj is None:
        return 0
    ptr = obj.data_ptr()
    # Validate pointer is accessible from device
    _init_pointer_validation()
    dev_ptr = c_uint64()
    status = _cuPointerGetAttribute(ctypes.byref(dev_ptr), _CU_POINTER_ATTRIBUTE_DEVICE_POINTER, c_uint64(ptr))
    if status == 1:  # CUDA_ERROR_INVALID_VALUE
        raise ValueError(f"Pointer argument (at {idx}) cannot be accessed from Triton (cpu tensor?)")
    elif status != 0:
        raise RuntimeError(f"cuPointerGetAttribute failed with error {status}")
    # Use the original data_ptr() value directly. The cuPointerGetAttribute call
    # above validates the pointer is device-accessible, but the returned dev_ptr
    # can be unreliable through ctypes on some platforms.
    return ptr


# ---------------------------------------------------------------------------
# TMA descriptor (CUtensorMap) support
# ---------------------------------------------------------------------------

# CUtensorMap is a 128-byte opaque struct passed by value to kernels
CUtensorMap = ctypes.c_byte * 128


def _get_tma_desc_ptr(obj):
    """Extract a CUtensorMap host pointer from a Python TMA descriptor object.

    Mirrors the C launcher's getTmaDesc(): tries tma_desc_cpu_ptr() first,
    then falls back to reading the tensorMap field from PyCUtensorMapObject
    at its known struct offset.
    """
    if hasattr(obj, "tma_desc_cpu_ptr"):
        ptr = obj.tma_desc_cpu_ptr()
        if not ptr:
            raise ValueError("tma_desc_cpu_ptr() returned NULL")
        if ptr % 64 != 0:
            raise ValueError("tma_desc_cpu_ptr() must be 64-byte aligned")
        return ptr
    # Fallback for PyCUtensorMapObject from the C extension (driver.c).
    # The struct layout is: PyObject_HEAD (16 bytes) + padding to 128-byte
    # alignment + CUtensorMap (128 bytes). Since the object itself is
    # allocated with 128-byte alignment (posix_memalign), the tensorMap
    # field is at offset 128.
    if type(obj).__name__ == "PyCUtensorMap":
        obj_addr = id(obj)
        map_ptr = obj_addr + 128
        if map_ptr % 128 != 0:
            raise ValueError(f"CUtensorMap must be aligned to 128B, but got address % 128 = {map_ptr % 128}")
        return map_ptr
    raise TypeError(f"Expected TMA descriptor object with tma_desc_cpu_ptr() method, "
                    f"got {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Float packing helpers (equivalent to pack_fp16/bf16/fp32/fp64 in C)
# ---------------------------------------------------------------------------


def _pack_fp16(f):
    """Pack a Python float to fp16 as uint16."""
    return struct.unpack("H", struct.pack("e", f))[0]


def _pack_bf16(f):
    """Pack a Python float to bf16 as uint16."""
    f32_bytes = struct.pack("f", f)
    u32 = struct.unpack("I", f32_bytes)[0]
    return u32 >> 16


def _pack_fp32(f):
    """Pack a Python float to fp32 as uint32."""
    return struct.unpack("I", struct.pack("f", f))[0]


def _pack_fp64(f):
    """Pack a Python float to fp64 as uint64."""
    return struct.unpack("Q", struct.pack("d", f))[0]


PACK_FUNCTIONS = {
    "fp16": _pack_fp16,
    "bf16": _pack_bf16,
    "fp32": _pack_fp32,
    "f32": _pack_fp32,
    "fp64": _pack_fp64,
}

# Maps Triton type strings to (ctypes_type, is_pointer, is_float)
TYPE_MAP = {
    # Pointer types
    "*": "pointer",
    # Integer types
    "i1": (ctypes.c_int8, False, False),
    "i8": (ctypes.c_int8, False, False),
    "i16": (ctypes.c_int16, False, False),
    "i32": (ctypes.c_int32, False, False),
    "i64": (ctypes.c_int64, False, False),
    "u1": (ctypes.c_uint8, False, False),
    "u8": (ctypes.c_uint8, False, False),
    "u16": (ctypes.c_uint16, False, False),
    "u32": (ctypes.c_uint32, False, False),
    "u64": (ctypes.c_uint64, False, False),
    # Float types
    "fp16": (ctypes.c_uint16, False, True),
    "bf16": (ctypes.c_uint16, False, True),
    "fp32": (ctypes.c_uint32, False, True),
    "f32": (ctypes.c_uint32, False, True),
    "fp64": (ctypes.c_uint64, False, True),
}

# ---------------------------------------------------------------------------
# Python launcher factory
# ---------------------------------------------------------------------------


def make_ctypes_launcher(constants, signature, tensordesc_meta):
    """Build a pure-Python launch function equivalent to the C-compiled launcher.

    Returns a callable with the same interface as the C module's ``launch``
    function, but without any C compilation step.

    Parameters match the existing ``make_launcher`` / ``CudaLauncher`` contract:
      launch(gridX, gridY, gridZ, stream, function,
             launch_cooperative_grid, launch_cluster, launch_pdl,
             global_scratch_obj, profile_scratch_obj,
             kernel_metadata, launch_metadata,
             launch_enter_hook, launch_exit_hook,
             *kernel_args)
    """
    # Build the arg processing pipeline for kernel-specific args.
    # Each entry is either None (constexpr, skip) or a handler function that
    # converts a Python value into a ctypes value for the kernel params array.
    #
    # wrap_handle_tensordesc expands each tensordesc arg into multiple flat
    # values before calling launch(), so arg_handlers must match the expanded
    # layout. This replicates _expand_signature from make_launcher.
    arg_handlers = []
    tensordesc_idx = 0
    for idx, ty in signature.items():
        if isinstance(ty, str) and ty.startswith("tensordesc"):
            meta = tensordesc_meta[tensordesc_idx] if tensordesc_meta else None
            tensordesc_idx += 1

            match = re.match(r"tensordesc<[^[>]*\[([^\]]*)\]", ty)
            if match is None:
                raise ValueError(f"Cannot parse tensordesc type: {ty}")
            ndim = match.group(1).count(",") + 1

            if meta is None:
                # Host-side decomposition: *dtype, i64*2n, i1, i32*n, i64*n
                def _handle_td_ptr(val, _idx=idx):
                    ptr = _get_device_pointer(val, _idx)
                    return CUdeviceptr(ptr)

                arg_handlers.append(_handle_td_ptr)
                for _ in range(2 * ndim):
                    arg_handlers.append(lambda val: ctypes.c_int64(val))
                arg_handlers.append(lambda val: ctypes.c_int8(val))
            else:
                # TMA path: nvTmaDesc, i32*n, i64*n
                def _handle_tma(val):
                    ptr = _get_tma_desc_ptr(val)
                    buf = CUtensorMap()
                    ctypes.memmove(buf, ptr, 128)
                    return buf

                arg_handlers.append(_handle_tma)

            # Both paths end with: i32*n, i64*n
            for _ in range(ndim):
                arg_handlers.append(lambda val: ctypes.c_int32(val))
            for _ in range(ndim):
                arg_handlers.append(lambda val: ctypes.c_int64(val))
            continue

        if isinstance(ty, tuple):
            raise NotImplementedError("tuple signature arguments are not yet supported in ctypes launcher")

        if idx in constants or ty == "constexpr":
            arg_handlers.append(None)
            continue

        if isinstance(ty, str) and ty[0] == "*":
            # Pointer argument
            def _handle_ptr(val, _idx=idx):
                ptr = _get_device_pointer(val, _idx)
                return CUdeviceptr(ptr)

            arg_handlers.append(_handle_ptr)
        elif ty in PACK_FUNCTIONS:
            # Float argument: passed as double from Python, packed to storage type
            pack_fn = PACK_FUNCTIONS[ty]
            ctype = TYPE_MAP[ty][0]

            def _handle_float(val, _pack=pack_fn, _ct=ctype):
                return _ct(_pack(val))

            arg_handlers.append(_handle_float)
        else:
            # Integer argument
            info = TYPE_MAP.get(ty)
            if info is None:
                raise ValueError(f"Unsupported type: {ty}")
            ctype = info[0]

            def _handle_int(val, _ct=ctype):
                if hasattr(val, "item"):
                    val = val.item()
                return _ct(val)

            arg_handlers.append(_handle_int)

    def launch(
        gridX,
        gridY,
        gridZ,
        stream,
        function,
        launch_cooperative_grid,
        launch_cluster,
        launch_pdl,
        global_scratch_obj,
        profile_scratch_obj,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *kernel_args,
    ):
        _ensure_cuda_context()

        (
            num_warps,
            num_ctas,
            shared_memory,
            clusterDimX,
            clusterDimY,
            clusterDimZ,
            _preferredClusterDimX,
            _preferredClusterDimY,
            _preferredClusterDimZ,
        ) = kernel_metadata

        # Call enter hook
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        if gridX * gridY * gridZ <= 0:
            if launch_exit_hook is not None:
                launch_exit_hook(launch_metadata)
            return

        # Process global_scratch
        global_scratch = CUdeviceptr(0)
        if global_scratch_obj is not None:
            global_scratch = CUdeviceptr(_get_device_pointer(global_scratch_obj, -1))

        # Process profile_scratch
        profile_scratch = CUdeviceptr(0)
        if profile_scratch_obj is not None:
            profile_scratch = CUdeviceptr(_get_device_pointer(profile_scratch_obj, -1))

        # Build kernel params array
        # Order: kernel_args..., global_scratch, profile_scratch
        param_values = []
        for i, handler in enumerate(arg_handlers):
            if handler is None:
                continue
            param_values.append(handler(kernel_args[i]))
        param_values.append(global_scratch)
        param_values.append(profile_scratch)

        n_params = len(param_values)
        param_ptrs = (c_void_p * n_params)()
        for i, val in enumerate(param_values):
            param_ptrs[i] = ctypes.addressof(val)

        # Build launch config
        launch_attrs = (CUlaunchAttribute * 4)()
        num_attrs = 0

        actual_gridX = gridX
        actual_gridY = gridY
        actual_gridZ = gridZ

        if num_ctas != 1:
            actual_gridX *= clusterDimX
            actual_gridY *= clusterDimY
            actual_gridZ *= clusterDimZ

        if launch_pdl:
            launch_attrs[num_attrs].id = (CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION)
            launch_attrs[num_attrs].value.value = 1
            num_attrs += 1

        if launch_cooperative_grid:
            launch_attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE
            launch_attrs[num_attrs].value.value = 1
            num_attrs += 1

        if launch_cluster or num_ctas != 1:
            launch_attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            launch_attrs[num_attrs].value.clusterDim.x = clusterDimX
            launch_attrs[num_attrs].value.clusterDim.y = clusterDimY
            launch_attrs[num_attrs].value.clusterDim.z = clusterDimZ
            num_attrs += 1

            launch_attrs[num_attrs].id = (CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE)
            launch_attrs[num_attrs].value.clusterSchedulingPolicyPreference = (CU_CLUSTER_SCHEDULING_POLICY_SPREAD)
            num_attrs += 1

        config = CUlaunchConfig()
        config.gridDimX = actual_gridX
        config.gridDimY = actual_gridY
        config.gridDimZ = actual_gridZ
        config.blockDimX = 32 * num_warps
        config.blockDimY = 1
        config.blockDimZ = 1
        config.sharedMemBytes = shared_memory
        config.hStream = c_void_p(stream)
        config.attrs = launch_attrs
        config.numAttrs = num_attrs

        cu_func = c_void_p(function)
        cuLaunchKernelEx = _get_cuLaunchKernelEx()
        err = cuLaunchKernelEx(
            ctypes.byref(config),
            cu_func,
            param_ptrs,
            None,
        )
        if err != 0:
            raise RuntimeError(f"Triton Error [CUDA]: cuLaunchKernelEx failed with error code {err}")

        # Call exit hook
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)

    return launch
