"""
LLVM-based kernel launcher for Triton JIT.

Generates LLVM IR for a per-kernel launcher function, compiles it to a native
host .o using Triton's built-in LLVM (with X86 backend), and wraps it with
ctypes for calling from Python.

No gcc, no clang — the same LLVM instance that produces the cubin compiles the
launcher to a host .o.
"""

import ctypes
import hashlib
import os
import struct
import subprocess
import tempfile

from triton._C.libtriton import llvm
from triton.backends.nvidia.driver import libcuda_dirs
from triton.runtime.cache import get_cache_manager
# ---------------------------------------------------------------------------
# Type mapping: Triton type string → (ctypes type, LLVM IR type, byte size)
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    "i1": (ctypes.c_int8, "i8", 1),
    "i8": (ctypes.c_int8, "i8", 1),
    "i16": (ctypes.c_int16, "i16", 2),
    "i32": (ctypes.c_int32, "i32", 4),
    "i64": (ctypes.c_int64, "i64", 8),
    "u1": (ctypes.c_uint8, "i8", 1),
    "u8": (ctypes.c_uint8, "i8", 1),
    "u16": (ctypes.c_uint16, "i16", 2),
    "u32": (ctypes.c_uint32, "i32", 4),
    "u64": (ctypes.c_uint64, "i64", 8),
    "fp16": (ctypes.c_uint16, "i16", 2),
    "bf16": (ctypes.c_uint16, "i16", 2),
    "fp32": (ctypes.c_float, "float", 4),
    "f32": (ctypes.c_float, "float", 4),
    "fp64": (ctypes.c_double, "double", 8),
}

# Float packing: convert Python float to the bit pattern expected by the kernel.
_FLOAT_PACK = {
    "fp16": lambda f: struct.unpack("H", struct.pack("e", f))[0],
    "bf16": lambda f: struct.unpack("H", struct.pack("f", f))[0] >> 16,
}


def _resolve_type(triton_ty):
    """Return (ctypes_type, llvm_ir_type, byte_size) for a Triton type string."""
    if triton_ty.startswith("*") or triton_ty.startswith("tensordesc"):
        return (ctypes.c_uint64, "i64", 8)  # CUdeviceptr
    entry = _TYPE_MAP.get(triton_ty)
    if entry is None:
        return (ctypes.c_uint64, "i64", 8)  # fallback: treat as pointer
    return entry


# ---------------------------------------------------------------------------
# LLVM IR generation
# ---------------------------------------------------------------------------


def make_launcher_ir(schema):
    """Generate LLVM IR text (.ll) for a kernel launcher from Level 0 schema.

    The generated function signature is:
        i32 @triton_launch_<name>(
            ptr %grid,      ; pointer to [3 x i32] grid dims
            i64 %stream,    ; CUstream (opaque handle)
            i64 %function,  ; CUfunction (opaque handle)
            ptr %params,    ; pointer to void*[] kernel params array
            i32 %num_params ; number of kernel params
        )

    The function builds CUlaunchConfig on the stack, then calls
    cuLaunchKernelEx (passed via a global that must be resolved before use).

    We use an even simpler approach: the Python wrapper builds the void* params
    array via ctypes. The LLVM function just handles CUlaunchConfig setup and
    the cuLaunchKernelEx call with baked-in constants (num_warps, shared_mem, etc).
    """
    kernel_name = schema["entry_name"]
    safe_name = kernel_name.replace(".", "_")
    num_warps = schema["num_warps"]
    num_ctas = schema["num_ctas"]
    shared_mem = schema["shared_mem"]
    launch_coop = 1 if schema["launch_cooperative_grid"] else 0
    launch_cluster = 1 if schema.get("launch_cluster", False) else 0
    launch_pdl = 1 if schema["launch_pdl"] else 0

    # CUlaunchConfig layout (x86_64, from CUDA driver API):
    #   unsigned int gridDimX, gridDimY, gridDimZ;     // 3 x i32, offset 0
    #   unsigned int blockDimX, blockDimY, blockDimZ;   // 3 x i32, offset 12
    #   unsigned int sharedMemBytes;                    // i32, offset 24
    #   <4 bytes padding>                               // padding to align hStream
    #   CUstream hStream;                               // ptr (i64), offset 32
    #   CUlaunchAttribute *attrs;                       // ptr, offset 40
    #   unsigned int numAttrs;                          // i32, offset 48
    # Total: 52 bytes + padding = 56 bytes (aligned to 8)

    # CUlaunchAttribute layout:
    #   CUlaunchAttributeID id;    // i32 (enum), offset 0
    #   <4 bytes padding>
    #   CUlaunchAttributeValue val; // 64-byte union, offset 8
    # Total: 72 bytes per attribute

    # Build launch attributes at IR generation time
    attrs = []
    if launch_pdl:
        # CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6
        attrs.append((6, 1))
    if launch_coop:
        # CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2
        attrs.append((2, 1))
    if launch_cluster or num_ctas > 1:
        if num_ctas > 1:
            # CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4
            # val.clusterDim.x = num_ctas, y = 1, z = 1 (first 3 x i32 of val)
            attrs.append((4, num_ctas, 1, 1))
        # CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5
        # val = CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1
        attrs.append((5, 1))
    num_attrs = len(attrs)

    block_dim_x = 32 * num_warps
    grid_dim_x_mul = num_ctas  # grid[0] * num_ctas

    ir_lines = []
    ir_lines.append(f'; Triton launcher for kernel: {kernel_name}')
    ir_lines.append(f'; num_warps={num_warps}, num_ctas={num_ctas}, shared_mem={shared_mem}')
    ir_lines.append('target triple = "x86_64-unknown-linux-gnu"')
    ir_lines.append(
        'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"')
    ir_lines.append('')

    # Declare cuLaunchKernelEx as an external function.
    # Signature: CUresult cuLaunchKernelEx(CUlaunchConfig*, CUfunction, void**, void**)
    # We model CUlaunchConfig* and CUfunction as opaque pointers (ptr).
    ir_lines.append('declare i32 @cuLaunchKernelEx(ptr, ptr, ptr, ptr)')
    ir_lines.append('')

    # The launcher function: takes grid ptr, stream, function, params array ptr.
    # Returns CUresult (i32).
    ir_lines.append(f'define i32 @triton_launch_{safe_name}(ptr %grid, i64 %stream, i64 %function, ptr %params) {{')
    ir_lines.append('entry:')

    # Alloca CUlaunchConfig (56 bytes, aligned 8)
    ir_lines.append('  %config = alloca [56 x i8], align 8')

    # gridDimX = grid[0] * num_ctas
    ir_lines.append('  %grid0_ptr = getelementptr i32, ptr %grid, i32 0')
    ir_lines.append('  %grid0 = load i32, ptr %grid0_ptr')
    if grid_dim_x_mul != 1:
        ir_lines.append(f'  %gridDimX = mul i32 %grid0, {grid_dim_x_mul}')
    else:
        ir_lines.append('  %gridDimX = add i32 %grid0, 0')  # identity

    # gridDimY = grid[1]
    ir_lines.append('  %grid1_ptr = getelementptr i32, ptr %grid, i32 1')
    ir_lines.append('  %gridDimY = load i32, ptr %grid1_ptr')

    # gridDimZ = grid[2]
    ir_lines.append('  %grid2_ptr = getelementptr i32, ptr %grid, i32 2')
    ir_lines.append('  %gridDimZ = load i32, ptr %grid2_ptr')

    # Store gridDim fields (offset 0, 4, 8)
    ir_lines.append('  %cfg_gridX = getelementptr i8, ptr %config, i32 0')
    ir_lines.append('  %cfg_gridX_i32 = bitcast ptr %cfg_gridX to ptr')
    ir_lines.append('  store i32 %gridDimX, ptr %cfg_gridX_i32')

    ir_lines.append('  %cfg_gridY = getelementptr i8, ptr %config, i32 4')
    ir_lines.append('  store i32 %gridDimY, ptr %cfg_gridY')

    ir_lines.append('  %cfg_gridZ = getelementptr i8, ptr %config, i32 8')
    ir_lines.append('  store i32 %gridDimZ, ptr %cfg_gridZ')

    # blockDimX, Y, Z (offset 12, 16, 20)
    ir_lines.append('  %cfg_blockX = getelementptr i8, ptr %config, i32 12')
    ir_lines.append(f'  store i32 {block_dim_x}, ptr %cfg_blockX')
    ir_lines.append('  %cfg_blockY = getelementptr i8, ptr %config, i32 16')
    ir_lines.append('  store i32 1, ptr %cfg_blockY')
    ir_lines.append('  %cfg_blockZ = getelementptr i8, ptr %config, i32 20')
    ir_lines.append('  store i32 1, ptr %cfg_blockZ')

    # sharedMemBytes (offset 24)
    ir_lines.append('  %cfg_smem = getelementptr i8, ptr %config, i32 24')
    ir_lines.append(f'  store i32 {shared_mem}, ptr %cfg_smem')

    # hStream (offset 32, ptr/i64)
    ir_lines.append('  %cfg_stream = getelementptr i8, ptr %config, i32 32')
    ir_lines.append('  %stream_ptr = inttoptr i64 %stream to ptr')
    ir_lines.append('  store ptr %stream_ptr, ptr %cfg_stream')

    # attrs pointer (offset 40)
    if num_attrs > 0:
        # Alloca attribute array: each CUlaunchAttribute is 72 bytes
        ir_lines.append(f'  %attrs = alloca [{num_attrs * 72} x i8], align 8')

        for i, attr in enumerate(attrs):
            attr_base = f'  %attr{i}_base = getelementptr i8, ptr %attrs, i32 {i * 72}'
            ir_lines.append(attr_base)

            # Store id (i32 at offset 0)
            ir_lines.append(f'  store i32 {attr[0]}, ptr %attr{i}_base')

            # Store val (at offset 8)
            val_ptr = f'  %attr{i}_val = getelementptr i8, ptr %attr{i}_base, i32 8'
            ir_lines.append(val_ptr)

            if attr[0] == 4:  # CLUSTER_DIMENSION: val.clusterDim.x/y/z
                ir_lines.append(f'  store i32 {attr[1]}, ptr %attr{i}_val')
                ir_lines.append(f'  %attr{i}_val_y = getelementptr i8, ptr %attr{i}_val, i32 4')
                ir_lines.append(f'  store i32 {attr[2]}, ptr %attr{i}_val_y')
                ir_lines.append(f'  %attr{i}_val_z = getelementptr i8, ptr %attr{i}_val, i32 8')
                ir_lines.append(f'  store i32 {attr[3]}, ptr %attr{i}_val_z')
            else:
                # Generic: store first i32 of val union
                ir_lines.append(f'  store i32 {attr[1]}, ptr %attr{i}_val')

        ir_lines.append('  %cfg_attrs = getelementptr i8, ptr %config, i32 40')
        ir_lines.append('  store ptr %attrs, ptr %cfg_attrs')
    else:
        ir_lines.append('  %cfg_attrs = getelementptr i8, ptr %config, i32 40')
        ir_lines.append('  store ptr null, ptr %cfg_attrs')

    # numAttrs (offset 48)
    ir_lines.append('  %cfg_nattrs = getelementptr i8, ptr %config, i32 48')
    ir_lines.append(f'  store i32 {num_attrs}, ptr %cfg_nattrs')

    # Cast CUfunction from i64 to ptr
    ir_lines.append('  %func_ptr = inttoptr i64 %function to ptr')

    # Call cuLaunchKernelEx(&config, function, params, null)
    ir_lines.append('  %ret = call i32 @cuLaunchKernelEx(ptr %config, ptr %func_ptr, ptr %params, ptr null)')
    ir_lines.append('  ret i32 %ret')
    ir_lines.append('}')

    return '\n'.join(ir_lines) + '\n'


# ---------------------------------------------------------------------------
# Compile LLVM IR → host .so
# ---------------------------------------------------------------------------


def _compile_ir_to_so(ir_text, cache_key):
    """Compile LLVM IR to a shared object, caching the result.

    Uses Triton's LLVM (which includes X86 backend) via translate_to_asm
    to produce a host .o, then links it into a .so with the system linker.
    """
    cache = get_cache_manager(cache_key)
    so_name = "launcher.so"
    cached_path = cache.get_file(so_name)
    if cached_path is not None:
        return cached_path

    # Compile LLVM IR → x86_64 object file
    obj_bytes = llvm.translate_to_asm(ir_text, "x86_64-unknown-linux-gnu",  # triple
                                      "generic",  # proc
                                      "",  # features
                                      [],  # flags
                                      False,  # enable_fp_fusion
                                      True,  # isObject → returns bytes
                                      )

    # Link .o → .so using ld
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "launcher.o")
        so_path = os.path.join(tmpdir, so_name)
        with open(obj_path, "wb") as f:
            f.write(obj_bytes)

        # Link: need -shared -lcuda (for cuLaunchKernelEx symbol)
        # Use cc as linker to handle platform details
        ld = os.environ.get("CC", "cc")
        # Find libcuda search paths (handles RE environments)
        link_cmd = [
            ld,
            "-shared",
            "-o",
            so_path,
            obj_path,
        ]
        for d in libcuda_dirs():
            # libcuda_dirs may return file paths; use dirname
            p = d if os.path.isdir(d) else os.path.dirname(d)
            if p:
                link_cmd.append(f"-L{p}")
        link_cmd += ["-lcuda", "-Wl,--no-as-needed"]
        subprocess.check_call(link_cmd)

        with open(so_path, "rb") as f:
            so_bytes = f.read()

    cached_path = cache.put(so_bytes, so_name, binary=True)
    return cached_path


# ---------------------------------------------------------------------------
# ctypes wrapper
# ---------------------------------------------------------------------------


def _build_args_struct(schema):
    """Build a ctypes.Structure subclass for the kernel args."""
    fields = []
    for arg in schema["args"]:
        ctype, _, _ = _resolve_type(arg["type"])
        fields.append((arg["name"], ctype))
    # Scratch pointers (appended if needed)
    if schema["global_scratch_size"] > 0:
        fields.append(("global_scratch", ctypes.c_uint64))
    if schema["profile_scratch_size"] > 0:
        fields.append(("profile_scratch", ctypes.c_uint64))
    ns = {"_fields_": fields}
    return type("args_t", (ctypes.Structure, ), ns)


def _get_cu_launch_kernel_ex():
    """Lazily resolve cuLaunchKernelEx from libcuda."""
    if not hasattr(_get_cu_launch_kernel_ex, "_handle"):
        libcuda = ctypes.CDLL("libcuda.so.1")
        _get_cu_launch_kernel_ex._handle = libcuda
    return _get_cu_launch_kernel_ex._handle


def make_launcher(schema):
    """Build a Python-callable launcher from Level 0 schema.

    Returns a function with signature:
        launch(gridX, gridY, gridZ, stream, function,
               launch_cooperative_grid, launch_cluster, launch_pdl,
               global_scratch, profile_scratch,
               kernel_metadata, launch_metadata,
               launch_enter_hook, launch_exit_hook,
               *kernel_args)

    This matches the calling convention of the existing make_launcher() C
    extension, so CudaLauncher.__call__ does not need to change.
    """
    kernel_name = schema["entry_name"]
    safe_name = kernel_name.replace(".", "_")
    func_name = f"triton_launch_{safe_name}"

    # Generate and compile LLVM IR
    ir_text = make_launcher_ir(schema)
    cache_key = hashlib.sha256(ir_text.encode()).hexdigest()
    so_path = _compile_ir_to_so(ir_text, cache_key)

    # Load the .so and get the launcher function
    lib = ctypes.CDLL(so_path)
    native_launch = getattr(lib, func_name)
    native_launch.restype = ctypes.c_int32
    native_launch.argtypes = [
        ctypes.POINTER(ctypes.c_uint32 * 3),  # grid[3]
        ctypes.c_uint64,  # stream
        ctypes.c_uint64,  # function
        ctypes.c_void_p,  # params (void**)
    ]

    # Build per-kernel args info
    args_info = schema["args"]
    has_global_scratch = schema["global_scratch_size"] > 0
    has_profile_scratch = schema["profile_scratch_size"] > 0
    n_kernel_args = len(args_info)

    # Pre-build the param pointer array type.
    # The CUDA kernel ALWAYS has global_scratch and profile_scratch as the
    # last two parameters (even when scratch_size is 0), so we must always
    # include them in the params array.
    n_total_params = n_kernel_args + 2  # +2 for global_scratch, profile_scratch
    ParamArrayType = ctypes.c_void_p * n_total_params

    # Pre-resolve types for each kernel arg
    arg_ctypes = []
    for arg in args_info:
        ctype, _, size = _resolve_type(arg["type"])
        is_ptr = arg["type"].startswith("*") or arg["type"].startswith("tensordesc")
        pack_fn = _FLOAT_PACK.get(arg["type"])
        arg_ctypes.append((ctype, size, is_ptr, pack_fn))

    # The wrapper closure
    def launch(gridX, gridY, gridZ, stream, function, launch_cooperative_grid, launch_cluster, launch_pdl,
               global_scratch, profile_scratch, kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook,
               *kernel_args):
        # Enter hook
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        # Empty grid is a no-op (preserve existing CudaLauncher behavior)
        if gridX * gridY * gridZ <= 0:
            if launch_exit_hook is not None:
                launch_exit_hook(launch_metadata)
            return

        # Build grid
        grid = (ctypes.c_uint32 * 3)(gridX, gridY, gridZ)

        # Build void* params[] — each element points to the arg value
        # We allocate ctypes objects for each arg to keep them alive
        param_holders = []
        params = ParamArrayType()
        for i, (karg, (ctype, size, is_ptr, pack_fn)) in enumerate(zip(kernel_args, arg_ctypes)):
            if is_ptr:
                # karg is a Python int (device pointer) or object with data_ptr()
                if hasattr(karg, 'data_ptr'):
                    val = ctypes.c_uint64(karg.data_ptr())
                elif isinstance(karg, int):
                    val = ctypes.c_uint64(karg)
                else:
                    val = ctypes.c_uint64(int(karg))
            elif pack_fn is not None:
                # fp16/bf16: pack Python float into the correct bit pattern
                val = ctype(pack_fn(karg))
            else:
                val = ctype(karg)
            param_holders.append(val)
            params[i] = ctypes.cast(ctypes.pointer(val), ctypes.c_void_p)

        idx = n_kernel_args
        # Global scratch — always present in kernel params (value 0 when unused)
        if global_scratch is not None and has_global_scratch:
            if hasattr(global_scratch, 'data_ptr'):
                gs_val = ctypes.c_uint64(global_scratch.data_ptr())
            else:
                gs_val = ctypes.c_uint64(int(global_scratch))
        else:
            gs_val = ctypes.c_uint64(0)
        param_holders.append(gs_val)
        params[idx] = ctypes.cast(ctypes.pointer(gs_val), ctypes.c_void_p)
        idx += 1

        # Profile scratch — always present in kernel params (value 0 when unused)
        if profile_scratch is not None and has_profile_scratch:
            if hasattr(profile_scratch, 'data_ptr'):
                ps_val = ctypes.c_uint64(profile_scratch.data_ptr())
            else:
                ps_val = ctypes.c_uint64(int(profile_scratch))
        else:
            ps_val = ctypes.c_uint64(0)
        param_holders.append(ps_val)
        params[idx] = ctypes.cast(ctypes.pointer(ps_val), ctypes.c_void_p)

        # Call the LLVM-compiled launcher
        try:
            err = native_launch(
                ctypes.pointer(grid),
                ctypes.c_uint64(stream),
                ctypes.c_uint64(function),
                ctypes.cast(params, ctypes.c_void_p),
            )
            if err != 0:
                raise RuntimeError(f"Triton kernel launch failed with CUDA error {err}")
        finally:
            # Exit hook
            if launch_exit_hook is not None:
                launch_exit_hook(launch_metadata)

    return launch
