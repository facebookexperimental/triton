import functools
import hashlib
import os
import re
import signal
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from triton import knobs
from triton._C.libtriton import ir, llvm, nvidia, passes, tlx
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton.runtime.errors import PTXASError


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:  # [m, n, k]
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        # For small M/N the input we can still use tensorcores with padding.
        if lhs_bitwidth == 8:
            return (1, 1, 32)
        else:
            return (1, 1, 16)

    return check_dot_compatibility


def get_ptxas(arch: int) -> knobs.NvidiaTool:
    return knobs.nvidia.ptxas_blackwell if arch >= 100 else knobs.nvidia.ptxas


@functools.lru_cache()
def get_ptxas_version(arch: int = 80):
    mock_ver = knobs.nvidia.mock_ptx_version
    if mock_ver is not None:
        return mock_ver  # This is not really a version of ptxas, but it is good enough for testing
    version = subprocess.check_output([get_ptxas(arch).path, "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor

    if major >= 13:
        base_ptx = 90
        return base_ptx + (major - 13) * 10 + minor

    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)


def get_ptx_version_from_options(options, arch: int):
    ptx_version = options.ptx_version
    if ptx_version is None:
        cuda_version = get_ptxas(arch).version
        ptx_version = ptx_get_version(cuda_version)
    return ptx_version


@functools.lru_cache()
def get_features(options, arch: int):
    ptx_version = get_ptx_version_from_options(options, arch)

    # PTX 8.6 is the max version supported by llvm c1188642.
    #
    # To check if a newer PTX version is supported, increase this value
    # and run a test.  If it's not supported, LLVM will print a warning
    # like "+ptx8.4 is not a recognized feature for this target".
    llvm_ptx_version = min(86, ptx_version)
    features = f'+ptx{llvm_ptx_version}'
    return features


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sm_arch_from_capability(capability: int):
    # TODO: Handle non-"a" sms
    suffix = "a" if capability >= 90 else ""
    return f"sm_{capability}{suffix}"


def _max_shared_mem_for_capability(capability: int) -> int:
    """Return CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN for a given SM capability.

    Tries querying the GPU driver first. Falls back to a static table for
    offline compilation environments (e.g. Triton CC on RE) where no GPU is present.
    """
    try:
        from triton.runtime.driver import driver as rt_driver
        return rt_driver.active.utils.get_device_properties(rt_driver.active.get_current_device())["max_shared_mem"]
    except (RuntimeError, Exception):
        pass
    # Fallback for offline compilation (no GPU present).
    # Values are CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN per
    # the CUDA Programming Guide "Technical Specifications per Compute Capability".
    _SMEM_SIZES = {
        70: 98304,  # V100:    96 KB per SM, optin = 96 KB
        75: 65536,  # Turing:  64 KB per SM, optin = 64 KB
        80: 166912,  # A100:   164 KB per SM, optin = 163 KB
        86: 101376,  # GA10x:  100 KB per SM, optin = 99 KB
        87: 166912,  # Orin:   164 KB per SM, optin = 163 KB
        89: 101376,  # AD10x:  100 KB per SM, optin = 99 KB
        90: 232448,  # H100:   228 KB per SM, optin = 227 KB
        100: 232448,  # B200:   228 KB per SM, optin = 227 KB
        103: 232448,  # GB300:  228 KB per SM, optin = 227 KB
        110: 232448,  # SM110: 228 KB per SM, optin = 227 KB
        120: 101376,  # SM120: 100 KB per SM, optin = 99 KB
    }
    # Try exact capability first (e.g. 86), then round to family base
    # (e.g. 86 -> 80) for unknown sub-variants, then fall back to 48 KB
    # (the default max shared mem per block without optin).
    return _SMEM_SIZES.get(capability, _SMEM_SIZES.get(capability // 10 * 10, 49152))


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 32
    minRegAutoWS: int = 24
    maxRegAutoWS: int = 152
    pingpongAutoWS: bool = False
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    ctas_per_cga: Optional[tuple] = None  # Alias for cluster_dims with CUDA semantics
    preferred_ctas_per_cga: Optional[tuple] = None  # Hint for preferred cluster size (CUDA 12.8+)
    ptx_version: int = None
    ptx_options: Optional[str] = knobs.nvidia.ptxas_options
    ir_override: Optional[str] = None  # filename of a user-defined IR (*.{ttir|ttgir|llir|ptx})
    enable_fp_fusion: bool = True
    enable_reflect_ftz: bool = True  # ftz in libdevice
    launch_cooperative_grid: bool = False
    launch_cluster: bool = False  # Blackwell cluster launcher
    launch_pdl: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee", 'bf16x3', 'bf16x6')
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'cuda'
    sanitize_overflow: bool = False
    arch: str = None
    instrumentation_mode: str = ""
    early_tma_store_lowering: bool = False
    generate_subtiled_region: bool = False

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = knobs.nvidia.libdevice_path or str(default_libdir / 'libdevice.10.bc')

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"

        # If ctas_per_cga is set, it overrides cluster_dims with CUDA semantics:
        # ctas_per_cga defines the cluster shape for regrouping grid CTAs.
        # num_ctas must be 1 when using ctas_per_cga since it's incompatible with
        # the multiplicative semantics of num_ctas.
        if self.ctas_per_cga is not None:
            # Ensure cluster_dims is all 1s to prevent conflicting cluster specifications.
            assert self.cluster_dims == (1, 1, 1) or self.cluster_dims == self.ctas_per_cga, (
                f"When using ctas_per_cga, cluster_dims must be default (1,1,1) or match ctas_per_cga to avoid conflicting "
                f"cluster specifications. Got cluster_dims={self.cluster_dims}")

            object.__setattr__(self, "cluster_dims", self.ctas_per_cga)
            object.__setattr__(self, "num_ctas", 1)

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @property
    def enable_iisan(self):
        return "iisan" in self.instrumentation_mode


class CUDABackend(BaseBackend):
    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cuda'

    def _parse_arch(self, arch):
        pattern = r"^sm(\d+)$"
        match = re.fullmatch(pattern, arch)
        if not match:
            raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
        return int(match.group(1))

    def get_target_name(self, options) -> str:
        capability = self._parse_arch(options.arch)
        return f"cuda:{capability}"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        # Enable debug mode for ConSan, so device-side assertions are not optimized out
        if any(mode in opts.get("instrumentation_mode", "") for mode in ["consan", "iisan"]):
            opts["debug"] = True
            opts["sanitize_overflow"] = False

        args = {'arch': knobs.runtime.override_arch or f"sm{self.target.arch}"}
        args.update({k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        capability = int(self._parse_arch(args["arch"]))

        if args.get("num_ctas", 1) > 1 and capability < 90:
            raise ValueError((f"num_ctas > 1 requires NVIDIA SM90+ (Hopper). "
                              f"Current target is sm_{capability}. This configuration will fail. "
                              f"Please set num_ctas=1 or target an SM90+ GPU."))

        if args.get("preferred_ctas_per_cga") is not None and capability < 100:
            raise ValueError((f"preferred_ctas_per_cga requires NVIDIA SM100+ (Blackwell). "
                              f"Current target is sm_{capability}."))

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CUDAOptions.supported_fp8_dtypes)
            if capability >= 89:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dot_operand_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dot_operand_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0

        return CUDAOptions(**args)

    def pack_metadata(self, metadata):
        preferred = getattr(metadata, "preferred_ctas_per_cga", None) or (0, 0, 0)
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            preferred[0],
            preferred[1],
            preferred[2],
        )

    def make_launch_metadata(self, metadata, src):
        """Produce a versioned, machine-readable JSON dict describing the kernel launch contract.

        This is the Level 0 metadata schema: a self-contained description of everything
        a launcher needs to know to call cuLaunchKernelEx for this kernel.  It is stored
        alongside the cubin as ``asm["launch_metadata"]`` and is intended to replace the
        implicit metadata bag that downstream consumers currently probe with hasattr guards.

        The schema is purely additive — existing ``pack_metadata()`` / ``make_launcher()``
        paths are not affected.
        """

        def _get(key, default=None):
            """Retrieve a field from metadata, which may be a dict or a namedtuple."""
            if isinstance(metadata, dict):
                return metadata.get(key, default)
            return getattr(metadata, key, default)

        cluster_dims = _get("cluster_dims") or (1, 1, 1)
        preferred = _get("preferred_ctas_per_cga") or (0, 0, 0)

        # Build the args array from src.signature, excluding compile-time constants.
        constants = getattr(src, "constants", {})
        # Normalize constant keys to tuple form for lookup.
        constant_keys = set()
        for k in constants:
            if isinstance(k, str):
                if hasattr(src, "fn"):
                    constant_keys.add((src.fn.arg_names.index(k), ))
                else:
                    constant_keys.add((k, ))
            elif isinstance(k, tuple):
                constant_keys.add(k)
            else:
                constant_keys.add((k, ))

        attrs = getattr(src, "attrs", {})
        arg_names = src.fn.arg_names if hasattr(src, "fn") else None

        args = []
        for idx, (key, ty) in enumerate(src.signature.items()):
            # Skip compile-time constants — they go in the "constants" dict.
            if (idx, ) in constant_keys:
                continue

            name = key if isinstance(key, str) else (arg_names[idx] if arg_names and idx < len(arg_names) else str(idx))
            arg_entry = {"name": name, "type": str(ty), "index": idx}

            # Check for tt.divisibility attribute.
            attr_specs = attrs.get((idx, ), [])
            for attr_name, attr_val in attr_specs:
                if attr_name == "tt.divisibility":
                    arg_entry["divisible_by"] = attr_val
            args.append(arg_entry)

        # Serialize constants: keys are stringified indices, values are the constant values.
        constants_dict = {}
        for k, v in constants.items():
            if isinstance(k, tuple):
                str_key = str(k[0]) if len(k) == 1 else str(k)
            elif isinstance(k, str):
                if arg_names:
                    str_key = str(arg_names.index(k))
                else:
                    str_key = k
            else:
                str_key = str(k)
            # Convert to JSON-serializable value
            if isinstance(v, (int, float, bool, str)) or v is None:
                constants_dict[str_key] = v
            else:
                constants_dict[str_key] = str(v)

        tensordesc_meta = _get("tensordesc_meta")

        schema = {
            "abi_version": 1,
            "entry_name": _get("name", ""),
            "num_warps": _get("num_warps"),
            "num_ctas": _get("num_ctas"),
            "shared_mem": _get("shared", 0),
            "cluster_dims": list(cluster_dims),
            "preferred_cluster_dims": list(preferred),
            "launch_cooperative_grid": _get("launch_cooperative_grid", False),
            "launch_cluster": _get("launch_cluster", False),
            "launch_pdl": _get("launch_pdl", False),
            "global_scratch_size": _get("global_scratch_size", 0),
            "global_scratch_align": _get("global_scratch_align", 128),
            "profile_scratch_size": _get("profile_scratch_size", 0),
            "profile_scratch_align": _get("profile_scratch_align", 1),
            "tmem_size": _get("tmem_size", 0),
            "args": args,
            "constants": constants_dict,
            "tensordesc_meta": tensordesc_meta or [],
        }
        return schema

    def make_launcher_src(self, metadata, src):
        """Generate a standalone C launcher source from Level 0 metadata.

        The generated C file includes ``triton/runtime/launch.h`` and implements
        a single entry point ``triton_launch_<kernel>()`` that sets up
        CUlaunchConfig with compile-time-known parameters baked in as constants,
        builds the kernel parameter array, and calls ``cuLaunchKernelEx``.

        The C source has NO dependency on Python.h — it is callable from C, C++,
        or via ctypes/cffi.  It is stored as ``asm["launcher_src"]`` for
        inspection and can be compiled by gcc/clang for use in TritonCC, AOT-T,
        or other C/C++ consumers.
        """
        launch_meta = self.make_launch_metadata(metadata, src)
        kernel_name = launch_meta["entry_name"]
        safe_name = kernel_name.replace(".", "_")

        # Type mapping: Triton type → C type for the args struct.
        # WARNING: This map must be kept in sync with Triton's type system.
        # If a new Triton type is added (e.g., fp8e4m3) and not present here,
        # we raise an error rather than silently generating incorrect code.
        _TYPE_TO_C = {
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
            "fp16": "uint16_t",
            "bf16": "uint16_t",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }

        def _c_type(triton_ty):
            if triton_ty.startswith("*"):
                return "CUdeviceptr"
            if triton_ty.startswith("tensordesc"):
                return "CUdeviceptr"  # host-side: passed as base pointer
            if triton_ty == "nvTmaDesc":
                return "CUtensorMap"
            c_ty = _TYPE_TO_C.get(triton_ty)
            if c_ty is None:
                # Unknown type — skip launcher generation so compilation
                # isn't blocked by types we haven't mapped yet.
                warnings.warn(f"Unknown Triton type '{triton_ty}' in launcher codegen, "
                              f"skipping launcher generation. Add it to _TYPE_TO_C in make_launcher_src().")
                return None
            return c_ty

        args = launch_meta["args"]
        num_warps = launch_meta["num_warps"]
        num_ctas = launch_meta["num_ctas"]
        shared_mem = launch_meta["shared_mem"]
        cluster_dims = launch_meta["cluster_dims"]
        preferred = launch_meta["preferred_cluster_dims"]
        launch_coop = 1 if launch_meta["launch_cooperative_grid"] else 0
        launch_cluster_flag = 1 if launch_meta.get("launch_cluster", False) else 0
        launch_pdl = 1 if launch_meta["launch_pdl"] else 0
        global_scratch_size = launch_meta["global_scratch_size"]
        profile_scratch_size = launch_meta["profile_scratch_size"]

        lines = []
        lines.append("/* Generated by Triton compiler — do not edit. */")
        lines.append(f"/* Kernel: {kernel_name} */")
        lines.append(f"/* ABI version: {launch_meta['abi_version']} */")
        lines.append("")
        lines.append('#include "triton/runtime/launch.h"')
        lines.append("")

        # ---- Args struct ----
        lines.append("typedef struct {")
        for arg in args:
            c_ty = _c_type(arg["type"])
            if c_ty is None:
                # Unsupported type — cannot generate a correct launcher.
                return f"/* Launcher not generated: unsupported arg type '{arg['type']}' for '{arg['name']}' */\n"
            lines.append(f"    {c_ty} {arg['name']};")
        lines.append(f"}} {safe_name}_args_t;")
        lines.append("")

        # ---- Launch function ----
        lines.append("/**")
        lines.append(f" * Launch {kernel_name}.")
        lines.append(" *")
        lines.append(" * Compile-time constants baked in:")
        lines.append(f" *   num_warps={num_warps}, num_ctas={num_ctas}, "
                     f"shared_mem={shared_mem}")
        lines.append(f" *   cluster_dims=[{cluster_dims[0]},{cluster_dims[1]},{cluster_dims[2]}]")
        lines.append(f" *   launch_pdl={launch_pdl}, cooperative={launch_coop}")
        if global_scratch_size > 0:
            lines.append(f" *   global_scratch_size={global_scratch_size}")
        if profile_scratch_size > 0:
            lines.append(f" *   profile_scratch_size={profile_scratch_size}")
        lines.append(" */")

        lines.append(f"CUresult triton_launch_{safe_name}(")
        lines.append("    const uint32_t grid[3],")
        lines.append("    CUstream stream,")
        lines.append("    CUfunction function,")
        lines.append(f"    {safe_name}_args_t *args,")
        # Always include scratch params for stable ABI across all kernels.
        # Callers pass 0/NULL when the kernel doesn't use scratch buffers.
        lines.append("    CUdeviceptr global_scratch,")
        lines.append("    CUdeviceptr profile_scratch")
        lines.append(") {")

        # Null checks
        lines.append("    if (!args) return CUDA_ERROR_INVALID_VALUE;")
        lines.append("    if (!function) return CUDA_ERROR_INVALID_HANDLE;")
        lines.append("")

        # Build params array
        param_names = [f"args->{arg['name']}" for arg in args]
        param_names.append("global_scratch")
        param_names.append("profile_scratch")

        lines.append("    /* Kernel parameter pointers */")
        for i, pname in enumerate(param_names):
            lines.append(f"    void *_param{i} = (void *)&{pname};")
        lines.append("    void *params[] = {")
        for i in range(len(param_names)):
            comma = "," if i < len(param_names) - 1 else ""
            lines.append(f"        _param{i}{comma}")
        lines.append("    };")
        lines.append("")

        # Build launch attributes (compile-time constants)
        lines.append("    /* Launch attributes (compile-time constants) */")
        lines.append("    CUlaunchAttribute attrs[TRITON_MAX_LAUNCH_ATTRS];")
        lines.append("    unsigned num_attrs = triton_build_launch_attrs(")
        lines.append("        attrs,")
        lines.append(f"        /*launch_pdl=*/{launch_pdl},")
        lines.append(f"        /*launch_cooperative_grid=*/{launch_coop},")
        lines.append(f"        /*num_ctas=*/{num_ctas},")
        lines.append(f"        /*launch_cluster=*/{launch_cluster_flag},")
        lines.append(f"        /*preferred_cluster_dim_x=*/{preferred[0]},")
        lines.append(f"        /*preferred_cluster_dim_y=*/{preferred[1]},")
        lines.append(f"        /*preferred_cluster_dim_z=*/{preferred[2]}")
        lines.append("    );")
        lines.append("")

        # Call triton_launch_kernel
        lines.append("    return triton_launch_kernel(")
        lines.append(f"        grid, /*num_warps=*/{num_warps}, /*num_ctas=*/{num_ctas},")
        lines.append(f"        /*shared_mem=*/{shared_mem}u, stream, function,")
        lines.append("        params, attrs, num_attrs")
        lines.append("    );")
        lines.append("}")

        return "\n".join(lines) + "\n"

    def get_codegen_implementation(self, options):
        import triton.language.extra.cuda as cuda
        capability = int(self._parse_arch(options.arch))
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if capability >= 80 else cuda.convert_custom_float8_sm70, "min_dot_size":
            min_dot_size(self.target)
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        nvidia.load_dialects(ctx)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        # Collect CUDA-specific warnings for Python emission
        cuda_warnings = mod.get_cuda_warnings(capability)
        for warning_msg in cuda_warnings:
            import warnings
            warnings.warn(warning_msg, stacklevel=2)

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # Pass cluster_dims as a list
        tlx.tlx_passes.add_triton_tlx_fixup(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas,
                                            list(opt.cluster_dims))
        passes.common.add_inliner(pm)
        # Handle storage lowering. In the future this may need
        # dummy layouts
        tlx.tlx_passes.add_tlx_storage_alias_lowering(pm)

        passes.ttir.add_rewrite_tensor_pointer(pm)
        if capability // 10 < 9:
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        # Set maxnreg on all kernels, if it was provided.
        if opt.maxnreg is not None:
            mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

        # Add minRegAutoWS attribute
        if opt.minRegAutoWS is not None:
            mod.set_attr("ttg.min_reg_auto_ws", ir.builder(mod.context).get_int32_attr(opt.minRegAutoWS))

        # Add maxRegAutoWS attribute
        if opt.maxRegAutoWS is not None:
            mod.set_attr("ttg.max_reg_auto_ws", ir.builder(mod.context).get_int32_attr(opt.maxRegAutoWS))

        # Add early TMA store lowering attribute
        if opt.early_tma_store_lowering:
            mod.set_attr("ttg.early_tma_store_lowering", ir.builder(mod.context).get_bool_attr(True))

        if opt.cluster_dims is not None:
            # Set cluster_info attributes on the module
            mod.set_attr(
                "ttg.cluster-dim-x",
                ir.builder(mod.context).get_int32_attr(opt.cluster_dims[0]),
            )
            mod.set_attr(
                "ttg.cluster-dim-y",
                ir.builder(mod.context).get_int32_attr(opt.cluster_dims[1]),
            )
            mod.set_attr(
                "ttg.cluster-dim-z",
                ir.builder(mod.context).get_int32_attr(opt.cluster_dims[2]),
            )
        pm = ir.pass_manager(mod.context)
        dump_enabled = pm.enable_debug()
        emuTF32 = (capability // 10 >= 8)
        passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        tlx.tlx_passes.add_tlx_propagate_layout(pm)
        # Only determine reg layouts after TMEM layout is finalized
        tlx.tlx_passes.add_tlx_resolve_placeholder_layouts(pm)
        tlx.tlx_passes.add_tlx_rewrite_local_alias(pm)
        passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        nvidia.passes.ttnvgpuir.add_plan_cta(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm, 0)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm, 0)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
        passes.ttir.add_loop_aware_cse(pm)
        use_meta_swp_schedule = knobs.nvidia.use_meta_ws and not knobs.nvidia.force_trunk_swp_schedule
        if capability // 10 in [8, 9]:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            if knobs.nvidia.use_meta_ws:
                nvidia.passes.hopper.add_data_partitioning(pm, 1)
                passes.ttgpuir.add_assign_latencies(pm, opt.num_stages, use_meta_swp_schedule)
                passes.ttgpuir.add_schedule_loops(pm, opt.num_stages, use_meta_swp_schedule)
            nvidia.passes.hopper.add_tma_store_lowering(pm)
            if knobs.nvidia.use_meta_ws:
                nvidia.passes.hopper.add_partition_scheduling_meta(pm)
            smem_budget = _max_shared_mem_for_capability(capability)
            generate_subtiled = opt.generate_subtiled_region or knobs.nvidia.generate_subtiled_region
            nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, capability, opt.pingpongAutoWS, dump_enabled,
                                                     smem_budget, generate_subtiled)
            if not knobs.nvidia.use_meta_ws:
                passes.ttgpuir.add_assign_latencies(pm, opt.num_stages, use_meta_swp_schedule)
                passes.ttgpuir.add_schedule_loops(pm, opt.num_stages, use_meta_swp_schedule)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        elif capability // 10 >= 10:
            if not knobs.nvidia.use_modulo_schedule:
                passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.ttgpuir.add_optimize_accumulator_init(pm)
            passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
            nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
            if knobs.nvidia.use_modulo_schedule is not None:
                # Modulo schedule runs BEFORE data partitioning so it can
                # see MMA ops before they're moved into WS regions. It
                # sets tt.autows annotations (stage/order) on MMA ops.
                # TRITON_USE_MODULO_SCHEDULE=1 (default algo: rau)
                # TRITON_USE_MODULO_SCHEDULE=sms|exhaustive|random
                nvidia.passes.hopper.add_modulo_schedule(pm)
            nvidia.passes.hopper.add_data_partitioning(pm, 1)
            # assign_latencies sets tt.latency on loads/MMAs (stage-distance
            # latencies). schedule_loops reads tt.latency AND tt.autows:
            # when MMA ops have tt.autows, scheduleKeyOpsAnnotation places
            # them at the annotated stages/clusters while scheduling all
            # other ops (loads, softmax, barriers) via the standard
            # latency-based heuristic. Without assign_latencies, the WS
            # pass's internal scheduleLoops has no latencies and can't
            # enter the code path that reads tt.autows annotations.
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages, use_meta_swp_schedule)
            passes.ttgpuir.add_schedule_loops(pm, opt.num_stages, use_meta_swp_schedule)
            if not knobs.nvidia.use_meta_ws:
                passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
            else:
                # use Meta's WS internally which supports both hopper and blackwell
                nvidia.passes.hopper.add_tma_store_lowering(pm)
                nvidia.passes.hopper.add_partition_scheduling_meta(pm)
                smem_budget = _max_shared_mem_for_capability(capability)
                generate_subtiled = opt.generate_subtiled_region or knobs.nvidia.generate_subtiled_region
                nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, capability, opt.pingpongAutoWS,
                                                         dump_enabled, smem_budget, generate_subtiled)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_optimize_partition_warps(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            # hoist again and allow hoisting out of if statements
            passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
            nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
        else:
            passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_coalesce_async_copy(pm)
        nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
        nvidia.passes.ttnvgpuir.add_lower_subtiled_region(pm)
        if capability // 10 >= 9:
            nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
            nvidia.passes.ttnvgpuir.add_tma_store_buffer_reuse(pm)
        smem_budget = _max_shared_mem_for_capability(capability)
        passes.ttgpuir.add_remove_layout_conversions(pm, 0)
        nvidia.passes.hopper.add_multi_cta_reduction(pm)
        # TODO: Find the optimal place in the pipeline for this pass.
        nvidia.passes.ttnvgpuir.add_prune_unused_barriers(pm)
        nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        # Optimize the number of warps and registers after TMA lowering, so
        # that any local loads eliminated by TMA lowering do not inflate them.
        if capability // 10 >= 9 and knobs.nvidia.use_meta_ws:
            passes.ttgpuir.add_optimize_partition_warps(pm)
        nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
        nvidia.passes.ttnvgpuir.add_lower_mma(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        # Budget-aware layout conversion elimination — runs last to ensure
        # converts whose scratch would exceed SMEM budget are eliminated
        # after all other passes that may introduce layout conversions.
        passes.ttgpuir.add_remove_layout_conversions(pm, smem_budget)

        pm.run(mod, 'make_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        # Track whether ctas_per_cga was explicitly set to distinguish between
        # Triton's way (num_ctas > 1) and TLX/CUDA way (ctas_per_cga set).
        metadata["ctas_per_cga"] = opt.ctas_per_cga
        metadata["preferred_ctas_per_cga"] = tuple(
            opt.preferred_ctas_per_cga) if opt.preferred_ctas_per_cga is not None else None
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def gluon_to_ttgir(self, src, metadata, options, capability):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_infer_coalesced_encodings(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod, 'gluon_to_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def make_llir(self, src, metadata, options, capability):
        ptx_version = get_ptx_version_from_options(options, self.target.arch)

        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.gluon.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
        nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
        nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm)
        if "consan" in options.instrumentation_mode:
            # Call ConcurrencySanitizerPass here, before allocating global scratch memory but after allocating tensor and shared
            passes.ttgpuir.add_concurrency_sanitizer(pm)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)
        # Print TTGIR to TLX mapping before final emission (for debugging/analysis)
        tlx_dump_dir = None
        tlx_saved_fd = None
        tlx_capture_file = None
        if knobs.nvidia.dump_tlx_benchmark:
            from triton.tools.tlx_benchmark_gen import setup_tlx_dump
            tlx_dump_dir, tlx_saved_fd, tlx_capture_file = setup_tlx_dump(pm, tlx.tlx_passes)
        elif knobs.nvidia.dump_ttgir_to_tlx:
            tlx.tlx_passes.add_tlx_print_ttgir_to_tlx(pm)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        nvidia.passes.hopper.add_tma_store_token_wait_lowering(pm)
        nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
        nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        passes.convert.add_nvvm_to_llvm(pm)
        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

        pm.run(mod, 'make_llir')

        # After pm.run(), restore stdout and generate TLX benchmark artifacts
        if tlx_dump_dir is not None:
            from triton.tools.tlx_benchmark_gen import finalize_tlx_dump
            finalize_tlx_dump(tlx_dump_dir, tlx_saved_fd, tlx_capture_file, metadata)

        if knobs.compilation.dump_ir_extract_di_local_variables:
            # comments below on why separate it
            if not knobs.compilation.disable_line_info:
                pm = ir.pass_manager(mod.context)
                pm.enable_debug()
                passes.llvmir.add_di_scope(pm)
                pm.run(mod, 'make_llir.disable_line_info')

            # insert dbg intrinsic with several DI Attribute including source
            # var name and type info note: unknown reason for now, but this
            # pass and add_di_scope has to be run separately, otherwise if we
            # put them into previous pipline, it trigger a segmentfault without
            # any error message; could be due to a bug in mlir or pybind11
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_local_variable(pm)
            pm.run(mod, 'make_llir.dump_ir_extract_di_local_variables')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        if knobs.compilation.enable_asan:
            raise RuntimeError(
                "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
        llvm_mod = llvm.to_module(mod, context)
        proc = sm_arch_from_capability(capability)
        features = get_features(options, self.target.arch)
        triple = 'nvptx64-nvidia-cuda'
        nvidia.set_short_ptr()
        llvm.attach_datalayout(llvm_mod, triple, proc, features)
        if options.enable_reflect_ftz:
            nvidia.set_nvvm_reflect_ftz(llvm_mod)

        if options.extern_libs and nvidia.has_extern_deps(llvm_mod):
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Get some metadata
        # warp-specialization mutates num_warps
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    def make_ptx(self, src, metadata, opt, capability):
        ptx_version = get_ptx_version_from_options(opt, self.target.arch)

        triple = 'nvptx64-nvidia-cuda'
        proc = sm_arch_from_capability(capability)
        features = get_features(opt, self.target.arch)
        flags = ["nvptx-mad-wide-opt"]
        ret = llvm.translate_to_asm(src, triple, proc, features, flags, opt.enable_fp_fusion, False)
        # Find kernel names (there should only be one)
        names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]
        # post-process
        ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret, flags=re.MULTILINE)
        if not knobs.compilation.dump_ir_extract_di_local_variables:
            # Remove the debug flag that prevents ptxas from optimizing the code
            # Note: if this flag is removed, the source var name and type info will be lost when ptx was compiled into cubin
            #           and we may not be able to see them in cuda-gdb
            ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        if knobs.nvidia.dump_nvptx:
            print("// -----// NVPTX Dump //----- //")
            print(ret)
        return ret

    def make_cubin(self, src, metadata, opt, capability):
        ptxas = get_ptxas(self.target.arch).path
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
                tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            debug_info = []
            if knobs.compilation.disable_line_info:
                # This option is ignored if used without -lineinfo
                debug_info += ["-lineinfo", "-suppress-debug-info"]
            elif knobs.nvidia.disable_ptxas_opt:
                # Synthesize complete debug info
                debug_info += ["-g"]
            else:
                # Only emit line info
                debug_info += ["-lineinfo"]

            fmad = [] if opt.enable_fp_fusion else ["--fmad=false"]
            arch = sm_arch_from_capability(capability)

            # Disable ptxas optimizations if requested
            disable_opt = ['--opt-level', '0'] if knobs.nvidia.disable_ptxas_opt else []

            # Accept more ptxas options if provided
            ptx_extra_options = opt.ptx_options.split(" ") if opt.ptx_options else []

            # Add --regAllocOptLevel=2 to work around ptxas 13.x bug
            reg_alloc = ['--regAllocOptLevel=2']

            ptxas_cmd = [
                ptxas, *debug_info, *fmad, '-v', *disable_opt, *reg_alloc, *ptx_extra_options, f'--gpu-name={arch}',
                fsrc.name, '-o', fbin
            ]
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if knobs.nvidia.dump_ptxas_log:
                    with open(flog.name) as log_file:
                        print(log_file.read())

                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                if e.returncode == 255:
                    error = 'Internal Triton PTX codegen error'
                elif e.returncode == 128 + signal.SIGSEGV:
                    error = '`ptxas` raised SIGSEGV'
                else:
                    error = f'`ptxas` failed with error code {e.returncode}'

                error = (f"{error}\n"
                         f"`ptxas` stderr:\n{log}\n"
                         f'Repro command: {" ".join(ptxas_cmd)}\n')

                print(f"""

================================================================
{error}

{src}
================================================================
please share the reproducer above with Triton project.
""")
                raise PTXASError(error)

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

    def add_stages(self, stages, options, language):
        capability = self._parse_arch(options.arch)
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options, capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.target.arch)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, capability)

    @functools.lru_cache()
    def hash(self):
        version = get_ptxas_version(self.target.arch)
        return f'{version}-{self.target.arch}'
