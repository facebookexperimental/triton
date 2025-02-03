from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import cpu, ir, llvm, nvidia, passes

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import os
import sys
import tempfile
from pathlib import Path


# from triton.backends.compiler import BaseBackend, GPUTarget
def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (2, 2, 2)


@dataclass(frozen=True)
class CPUOptions:
    # To emulate CPU-like behavior, we set warp_size, num_warps, and num_ctas to 1.
    # Note that we will explore a way of multiple threads/warps in the future.
    warp_size: int = 1
    num_warps: int = 1
    num_ctas: int = 1
    cluster_dims: tuple = (1, 1, 1)
    enable_fp_fusion: bool = True
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e5b16", "fp8e4nv")
    deprecated_fp8_dtypes: Tuple[str] = ()
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "tf32", "tf32x3")
    max_num_imprecise_acc_default: int = 0
    enable_fast_math: bool = True
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = "cpu"
    # TODO: Try to enable it.
    sanitize_overflow: bool = False
    arch: str = None
    # Max number of threads to be used for a kernel call.
    # Zero value is used to utilize all available CPU cores.
    num_threads: int = 0

    def __post_init__(self):
        assert self.num_warps == 1, f"{self.num_warps}: for now, num_warps must be 1 to emulate CPU-like behavior"

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "so"
        self.cpu_arch = target.arch
        self.cpu_name = llvm.get_cpu_name()
        self.cpu_features = llvm.get_cpu_features()

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        if "enable_fast_math" not in args:
            args["enable_fast_math"] = os.getenv("TRITON_CPU_FAST_MATH", "1") == "1"
        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CPUOptions.supported_fp8_dtypes)
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        # num_warps and num_ctas are only 1 for non-GPU.
        if "num_warps" in args and args["num_warps"] != 1:
            args["num_warps"] = 1
        if "num_ctas" in args and args["num_ctas"] != 1:
            args["num_ctas"] = 1
        return CPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": min_dot_size(self.target), "support_atomic_fmin_fmax": lambda: True}
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cpu import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        cpu.load_dialects(ctx)

    @staticmethod
    def parse_attr(desc):
        assert isinstance(desc, str)
        ret = []
        if "D" in desc:
            # This field is used to limit the maximum vectorizable bytes.
            # For Nvidia/PTX, it's 16 bytes. For CPU, we give a larger value.
            ret += [["tt.divisibility", 8192]]
        return ret

    @staticmethod
    def make_ttir(mod, metadata, opt):
        # This is the same as the Nvidia backend.
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttmir(mod, metadata, opt):
        # TODO: As a proof of concept, TTMIR is currently a subset of TTGIR,
        # which does not include GPU-specific stuffs. Ideally, it would be
        # TTIR --> TTMIR --> TTCIR or TTGIR.

        # Set up Diagnostic
        if os.environ.get("MLIR_ENABLE_REMARK", "0") == "1":
            srcMgr = llvm.source_mgr()
            diag = ir.source_mgr_diag(srcMgr, mod.context)
            mod.context.printOpOnDiagnostic(True)

        # TTIR -> TTMIR, which is simply TTGIR for now.
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        assert opt.num_warps == 1 and opt.warp_size == 1 and opt.num_ctas == 1
        # TODO: Eventually this should be add_convert_to_ttmir.
        # But for now, createConvertTritonToTritonGPUPass is still used.
        passes.ttir.add_convert_to_ttgpuir(pm, "cpu", opt.num_warps, opt.warp_size, opt.num_ctas)

        # Optimize TTMIR. Do not call any GPU-specific passes like using shared memory.
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("ttg.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups

        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # Set up Diagnostic
        if os.environ.get("MLIR_ENABLE_REMARK", "0") == "1":
            srcMgr = llvm.source_mgr()
            diag = ir.source_mgr_diag(srcMgr, mod.context)
            mod.context.printOpOnDiagnostic(True)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        cpu.passes.ttmir.add_to_llvmir(pm, llvm.get_cpu_features())
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)

        # Find kernel fn
        kernel_names = cpu.find_kernel_names(mod)
        assert len(kernel_names) == 1, f"expected exactly 1 kernel in a module, got {kernel_names}"

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()

        llvm_mod = llvm.to_module(mod, context)
        if llvm_mod is None:
            raise RuntimeError("Failed to convert LLVM IR")

        llvm.set_host_target(llvm_mod)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Get some metadata
        metadata["shared"] = 0
        metadata["name"] = kernel_names[0]
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_asm(src, metadata, options):
        return llvm.translate_to_host_asm(src, options.enable_fp_fusion, options.enable_fast_math)

    @staticmethod
    def make_so(src, metadata, options):
        from triton.runtime.build import _build
        import triton.backends.cpu.driver as cpu_driver

        with tempfile.TemporaryDirectory() as tmpdir:
            asm_path = os.path.join(tmpdir, "kernel.s")
            Path(asm_path).write_text(src)
            lib_dirs = cpu_driver.library_dirs
            libs = ["m", "TritonCPURuntime", "sleef"]
            so = _build("kernel", asm_path, tmpdir, lib_dirs, cpu_driver.include_dirs, libs)
            with open(so, "rb") as f:
                return f.read()

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttmir"] = lambda src, metadata: self.make_ttmir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
        stages["so"] = lambda src, metadata: self.make_so(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        # TODO: Get more detailed CPU info like raw brand name with supported ISAs.
        # Right now it would only return a simple string like "x86_64" or "aarch64".
        import platform

        return f"{platform.machine()}"
