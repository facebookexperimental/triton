import ast
import functools
import os
import re
import subprocess
import triton
import ctypes
import sys
from pathlib import Path
from triton import knobs
from triton.runtime.build import compile_module_from_src
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import (
    GPUDriver,
    decompose_descriptor,
    wrap_handle_tensordesc_impl,
)

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
# Path to the shared data-driven launcher core. It lives in the nvidia backend
# dir (next to driver.c / driver.py) and is the canonical source for C/C++
# consumers (cc_library triton_launch_h). driver.c is compiled via a fixed
# Remote-Execution command that ignores include_dirs, so we inline this header
# into the driver.c source at build time (see CudaUtils.__init__) rather than
# relying on an -I path. In both source and packaged/buck builds the header
# sits next to this file.
_launch_h_candidates = [
    os.path.join(dirname, "launch.h"),
]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ["libcuda.so.1"]
PyCUtensorMap = None
PyKernelArg = None
ARG_CONSTEXPR = None
ARG_KERNEL = None
ARG_TUPLE = None
GSAN_PER_DEVICE_STATE_STRIDE = 1 << 30


@functools.lru_cache()
def libcuda_dirs():
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = "libcuda.so cannot found!\n"
    if locs:
        msg += "Possible files are located at %s." % str(locs)
        msg += "Please create a symlink of libcuda.so to any of the files."
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += " (requires sudo) to refresh the linker cache."
    assert any(os.path.exists(os.path.join(path, "libcuda.so.1")) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]


def _cuda_driver_is_active():
    candidates = ["libcuda.so.1"]
    try:
        candidates.extend([os.path.join(path, "libcuda.so.1") for path in libcuda_dirs()])
    except Exception:
        pass

    libcuda = None
    for candidate in candidates:
        try:
            libcuda = ctypes.CDLL(candidate)
            break
        except OSError:
            continue

    if libcuda is None:
        return False

    cu_init = libcuda.cuInit
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    if cu_init(0) != 0:
        return False

    cu_device_get_count = libcuda.cuDeviceGetCount
    cu_device_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_device_get_count.restype = ctypes.c_int
    count = ctypes.c_int()
    if cu_device_get_count(ctypes.byref(count)) != 0:
        return False

    return count.value > 0


# ------------------------
# Utils
# ------------------------


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # Inline the shared launcher core (launch.h) directly into the driver
        # source: the fbcode Remote-Execution compile uses a fixed clang command
        # that does not honor include_dirs, so an `#include "triton/runtime/
        # launch.h"` would not resolve. Substituting the header text keeps a
        # single source of truth (launch.h) while making the compile
        # self-contained.
        driver_src = Path(os.path.join(dirname, "driver.c")).read_text()
        launch_h_path = next((p for p in _launch_h_candidates if os.path.exists(p)), None)
        if launch_h_path is None:
            raise FileNotFoundError(f"launch.h not found in any of: {_launch_h_candidates}")
        launch_h_src = Path(launch_h_path).read_text()
        include_marker = '#include "nvidia/backend/launch.h"'
        if include_marker not in driver_src:
            raise RuntimeError(f"driver.c must contain the marker {include_marker!r} for "
                               "launch.h inlining, but it was not found")
        driver_src = driver_src.replace(include_marker, launch_h_src)
        mod = compile_module_from_src(
            src=driver_src,
            name="cuda_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
        global PyCUtensorMap
        global PyKernelArg
        global ARG_CONSTEXPR
        global ARG_KERNEL
        global ARG_TUPLE
        PyCUtensorMap = mod.PyCUtensorMap
        PyKernelArg = mod.PyKernelArg
        ARG_CONSTEXPR = mod.ARG_CONSTEXPR
        ARG_KERNEL = mod.ARG_KERNEL
        ARG_TUPLE = mod.ARG_TUPLE
        self.load_binary = mod.load_binary
        self.unload_module = mod.unload_module
        self.get_current_device = mod.get_current_device
        self.set_current_device = mod.set_current_device
        self.get_default_stream = mod.get_default_stream
        self.get_device_capability = mod.get_device_capability
        self.get_device_properties = mod.get_device_properties
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        self.fill_tma_descriptor_tiled = mod.fill_tma_descriptor_tiled
        self.fill_tma_descriptor_im2col = mod.fill_tma_descriptor_im2col
        # Test-only hook to exercise the shared-core recipe TMA encoder.
        self._test_construct_tma_desc = getattr(mod, "_test_construct_tma_desc", None)
        self._tma_desc_bytes = getattr(mod, "_tma_desc_bytes", None)
        self.launch = mod.launch
        self.build_signature_metadata = mod.build_signature_metadata
        self.fill_1d_tma_descriptor = mod.fill_1d_tma_descriptor
        self.fill_2d_tma_descriptor = mod.fill_2d_tma_descriptor
        self.fill_1d_tma_descriptor_type = mod.fill_1d_tma_descriptor_type
        self.fill_2d_tma_descriptor_type = mod.fill_2d_tma_descriptor_type
        self.TmaDescKernelParam = TmaDescKernelParam
        self._TritonDispatcher = mod._TritonDispatcher
        self._TritonJITRunner = mod._TritonJITRunner
        self._ProxyRunner = mod._ProxyRunner
        self.create_fast_jit_type = mod.create_fast_jit_type


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == "*":
        return "CUdeviceptr"
    if ty.startswith("tensordesc"):
        return "CUtensorMap"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
        "nvTmaDesc": "CUtensorMap",
    }[ty]


def _flatten_schema_arg_type(ty, out):
    """Flatten a schema arg ``type`` into the launcher-ABI scalar leaves.

    A tuple-typed kernel argument is serialized in the Level 0 schema as a
    stringified Python tuple, e.g. ``"('i32', 'constexpr')"``, ``"()"`` or
    ``"(('constexpr',), ('i32',))"``. The variadic launcher
    (``build_signature_metadata``) consumes only the flattened, non-constexpr
    scalar leaves, mirroring ``make_kernel_signature``'s ``_flatten_signature``
    plus the ``constexpr`` drop. Non-tuple types pass through unchanged; a type
    that looks like a tuple but doesn't parse is left as-is so the launcher's
    existing "Unknown data type" error still surfaces it.
    """
    if not ty.startswith("("):
        out.append(ty)
        return
    try:
        parsed = ast.literal_eval(ty)
    except (ValueError, SyntaxError):
        out.append(ty)
        return

    def _walk(node):
        if isinstance(node, tuple):
            for x in node:
                _walk(x)
        elif node != "constexpr":
            out.append(node)

    _walk(parsed)


def build_kernel_signature_from_schema(schema):
    """Derive kernel_signature bytes from Level 0 schema args array.

    This makes the Level 0 schema the source of truth for type dispatch in the
    shared variadic launcher (driver.c).  The schema's ``args`` list contains
    only non-constant kernel parameters with their types already resolved.
    """
    flat_types = []
    tensordesc_meta = schema.get("tensordesc_meta") or []
    tensordesc_idx = 0

    for arg in schema["args"]:
        ty = arg["type"]
        if ty.startswith("tensordesc"):
            meta = (tensordesc_meta[tensordesc_idx] if tensordesc_idx < len(tensordesc_meta) else None)
            tensordesc_idx += 1

            match = re.match(r"tensordesc<([^[>]*)\[([^]]*)\]", ty)
            dtype = match.group(1)
            shape = match.group(2)
            ndim = shape.count(",") + 1

            if meta is None:
                # Host TMA path: base pointer + shape + strides + padding flag
                # + round_f32_to_tf32 flag. Must mirror expand_signature()'s host
                # TMA path (two i1 flags) and make_tensordesc_arg()'s arg layout
                # (arg.padding == "nan", arg.round_f32_to_tf32); see PR #9295.
                flat_types.append("*" + dtype)
                for _ in range(2 * ndim):
                    flat_types.append("i64")
                flat_types.append("i1")
                flat_types.append("i1")
            else:
                # Device TMA path: nvTmaDesc
                flat_types.append("nvTmaDesc")

            for _ in range(ndim):
                flat_types.append("i32")
            for _ in range(ndim):
                flat_types.append("i64")
        else:
            _flatten_schema_arg_type(ty, flat_types)

    return triton.runtime.driver.active.utils.build_signature_metadata(flat_types)


def expand_signature(signature, tensordesc_meta):
    output = []
    tensordesc_idx = 0
    # Expand tensor descriptor arguments into either nvTmaDesc, shape and
    # strides, or base pointer, shape and strides depending on whether the
    # kernel was lowered to use the nvTmaDesc or not.
    for sig in signature:
        if isinstance(sig, str) and sig.startswith("tensordesc"):
            meta = tensordesc_meta[tensordesc_idx] if tensordesc_meta else None
            tensordesc_idx += 1

            match = re.match("tensordesc<([^[>]*)\\[([^]]*)\\]", sig)
            dtype = match.group(1)
            shape = match.group(2)
            ndim = shape.count(",") + 1

            if meta is None:
                output.append("*" + dtype)
                # Currently the host side tensor descriptors get passed in as a
                # tensor desc, shape, and strides. We have no way to use these
                # shape and strides when processing tensor descriptors which is
                # why we provide our own decomposition above. Sadly this means
                # we have to pass the shape and strides twice.
                for _ in range(2 * ndim):
                    output.append("i64")
                output.append("i1")
                output.append("i1")
            else:
                output.append("nvTmaDesc")

            for _ in range(ndim):
                output.append("i32")
            for _ in range(ndim):
                output.append("i64")
        else:
            output.append(sig)

    assert not tensordesc_meta or tensordesc_idx == len(tensordesc_meta)
    return output


def make_kernel_signature(signature):
    """
    Creates a kernel signature in C to be able to efficiently extract
    arguments in the launcher.
    """

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    flat_signature = []
    for sig in signature:
        _flatten_signature(sig, flat_signature)
    kernel_signature = [x for x in flat_signature if x != "constexpr"]

    return triton.runtime.driver.active.utils.build_signature_metadata(kernel_signature)


def annotate_arguments(signature):
    """
    This recreates the signature with annotations as C objects which can then
    be used to efficiently flatten tuples, and remove constexpr in the launcher.
    """
    annotated_arguments = []
    for sig in signature:
        if isinstance(sig, tuple):
            annotated_arguments.append((PyKernelArg(nested_tuple=annotate_arguments(sig), type=ARG_TUPLE)))
        elif sig != "constexpr":
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_KERNEL))
        else:
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_CONSTEXPR))
    return annotated_arguments


# The TMA dtype enum values are slightly different on host vs device...
TMA_DTYPE_DEVICE_TO_HOST = dict((i, i) for i in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9
TMA_TF32 = 11


class TmaDescKernelParam:
    TMA_DESC_SIZE = 128

    def __init__(self):
        import torch

        self.desc = torch.empty(self.TMA_DESC_SIZE, dtype=torch.uint8, device="cpu")

    # Return a CUtensorMap* pointer in host memory
    def tma_desc_cpu_ptr(self):
        return self.desc.data_ptr()


def make_tensordesc_arg(arg, metadata, _):
    if metadata is None:
        return decompose_descriptor(arg)

    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]
    is_im2col = metadata.get("is_im2col", False)

    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1
    padding = 1 if arg.padding == "nan" else 0

    if fp4_padded:
        expanded_shape = list(shape)
        expanded_shape[-1] *= 2
    else:
        expanded_shape = shape

    if arg.round_f32_to_tf32:
        elem_type = TMA_TF32

    if is_im2col:
        # Im2col mode - use im2col descriptor fill function
        # block_size from metadata is [pixelsPerColumn, channelsPerPixel] (possibly clamped)
        element_strides = (arg.element_strides if arg.element_strides is not None else [1] * len(shape))
        cu_tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor_im2col(
            arg.base.data_ptr(),
            swizzle,
            elem_size,
            TMA_DTYPE_DEVICE_TO_HOST[elem_type],
            block_size,
            expanded_shape,
            strides,
            padding,
            arg.pixel_box_lower_corner,
            arg.pixel_box_upper_corner,
            element_strides,
        )
    else:
        # Tiled mode - use existing tiled descriptor fill function
        cu_tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor_tiled(
            arg.base.data_ptr(),
            swizzle,
            elem_size,
            TMA_DTYPE_DEVICE_TO_HOST[elem_type],
            block_size,
            expanded_shape,
            strides,
            padding,
        )

    return [cu_tensor_map, *shape, *strides]


def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    return wrap_handle_tensordesc_impl(launcher, signature, tensordesc_meta, make_tensordesc_arg)


class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)

        launcher = triton.runtime.driver.active.utils.launch

        # kernel_signature + arg_annotations are both derived from the object
        # signature (src.signature): expand_signature flattens tensordescs, then
        # make_kernel_signature / annotate_arguments walk the actual argument-type
        # OBJECTS via isinstance(...). A NamedTuple *is* a tuple, so it flattens
        # structurally like any other tuple -- no reliance on str(ty). Deriving the
        # signature from the Level 0 schema's stringified types is a repr blind
        # spot: a NamedTuple stringifies to a class repr ("Function(fn=...,
        # captured=...)") rather than a tuple literal, so a string parser silently
        # mis-flattens it. For every non-NamedTuple case this is byte-identical to
        # the old schema path (asserted by
        # test_launch_metadata.py::test_schema_derived_signature_matches_legacy).
        expanded_signature = expand_signature(signature.values(), tensordesc_meta)
        self.kernel_signature = make_kernel_signature(expanded_signature)
        self.arg_annotations = annotate_arguments(expanded_signature)

        self.launch = wrap_handle_tensordesc(launcher, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl

        # Distinguish between Triton's way and TLX's way by checking if ctas_per_cga
        # was explicitly set:
        # - Triton's way: Uses num_ctas > 1. Grid is multiplied by num_ctas to get total CTAs.
        # - TLX's way (CUDA native): Uses ctas_per_cga to set cluster shape.
        #   Grid equals total CTAs, and ctas_per_cga regroups them into clusters.
        # When ctas_per_cga is set, num_ctas must be 1 to prevent multiplicative behavior.
        if getattr(metadata, "ctas_per_cga", None) is not None:
            self.num_ctas = 1
        else:
            self.num_ctas = metadata.num_ctas

    def __call__(
        self,
        gridX,
        gridY,
        gridZ,
        stream,
        function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):

        active_driver = triton.runtime.driver.active

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        def allocate_default_profile_scratch(size, align):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                return active_driver.allocate_default_profile_scratch(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        if _allocation.has_profile_allocator():
            profile_scratch = allocate_scratch(
                self.profile_scratch_size,
                self.profile_scratch_align,
                _allocation._profile_allocator,
            )
        else:
            profile_scratch = allocate_default_profile_scratch(self.profile_scratch_size, self.profile_scratch_align)

        self.launch(
            gridX,
            gridY,
            gridZ,
            stream,
            function,
            self.launch_cooperative_grid,
            self.launch_pdl,
            kernel_metadata,
            launch_metadata,
            launch_enter_hook,
            launch_exit_hook,
            global_scratch,
            profile_scratch,
            self.arg_annotations,
            self.kernel_signature,
            args,
        )


class CudaDriver(GPUDriver):

    def __init__(self):
        self.utils = CudaUtils()  # TODO: make static
        self.launcher_cls = CudaLauncher
        if sys.modules.get("torch") is not None:
            super().__init__()
        else:
            self.get_device_capability = self._get_device_capability
            self.get_current_stream = self._get_current_stream
            self.get_current_device = self._get_current_device
            self.set_current_device = self._set_current_device

    def _get_device_capability(self, device):
        return self.utils.get_device_capability(device)

    def _get_current_stream(self, device):
        # The CUDA driver API does not expose PyTorch's notion of the current
        # stream. In torch-free launches we fall back to the device's default
        # stream after making that device's primary context current.
        return self.utils.get_default_stream(device)

    def _get_current_device(self):
        return self.utils.get_current_device()

    def _set_current_device(self, device):
        self.utils.set_current_device(device)

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("cuda", capability, warp_size)

    def get_active_torch_device(self):
        import torch

        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch

        return torch.cuda

    @staticmethod
    def is_active():
        return _cuda_driver_is_active()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="cuda")

    def clear_cache(self, cache):
        cache.zero_()
