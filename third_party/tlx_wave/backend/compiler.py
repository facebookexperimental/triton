from typing import Any

from dataclasses import dataclass
import os

from triton import knobs
from triton._C.libtriton import amd, ir, passes, tlx
import triton.backends.amd.compiler as amd_compiler
from triton.backends.compiler import GPUTarget, Language

from .converter import pipeline as converter_pipeline
from .wave_bridge_tools import (
    _compile_wave_module_to_hsaco,
    _verify_wave_module,
    _wave_opt,
    _wave_opt_sha256,
    _wave_pipelines_sha256,
)


_ENABLE_SPLIT_BARRIERS_ENV = "TRITON_TLX_WAVE_ENABLE_SPLIT_BARRIERS"


def _parse_bool_option(value, *, source):
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{source} must be a boolean")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"{source} must be a boolean")


@dataclass(frozen=True)
class TLXWaveOptions(amd_compiler.HIPOptions):
    # Enables WaveAMDMachine split-barrier lowering for eligible barriers. The
    # Wave pass is present in the backend pipeline but gated on this function
    # attribute so existing kernels keep the previous full-barrier behavior.
    tlx_wave_enable_split_barriers: bool = False

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(
            self,
            "tlx_wave_enable_split_barriers",
            _parse_bool_option(
                self.tlx_wave_enable_split_barriers,
                source="tlx_wave_enable_split_barriers",
            ),
        )


class TLXWaveBackend(amd_compiler.HIPBackend):
    """TLX-first AMD Wave backend scaffold.

    This backend deliberately reuses the HIP/TLX frontend and TTIR/TTGIR
    conversion path, then emits the current Wave IR handoff artifact.  The
    TTGIR-to-Wave bridge lives in Python so it can use Wave's own Python package
    and LLVM build without linking the two MLIR worlds in-process.
    """

    supports_native_tensor_specialization = False

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "tlx_wave"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "hsaco"

    def get_target_name(self, options) -> str:
        return f"tlx_wave:{options.arch}"

    def parse_options(self, opts) -> Any:
        opts = dict(opts)
        enable_split_barriers = opts.pop("tlx_wave_enable_split_barriers", None)
        if enable_split_barriers is None:
            enable_split_barriers = os.environ.get(_ENABLE_SPLIT_BARRIERS_ENV)
        enable_split_barriers = _parse_bool_option(
            enable_split_barriers,
            source="tlx_wave_enable_split_barriers",
        )
        opts["backend_name"] = "tlx_wave"
        # Match the AMD/HIP backend contract: matrix_instr_nonkdim=0 leaves
        # Triton's AMD matmul pass free to derive the MFMA shape from the tile.
        # The bridge validates the concrete TTGIR layout before emitting Wave IR.
        hip_options = super().parse_options(opts)
        options = TLXWaveOptions(
            **{
                name: getattr(hip_options, name)
                for name in amd_compiler.HIPOptions.__dataclass_fields__
            },
            tlx_wave_enable_split_barriers=enable_split_barriers,
        )
        if options.arch not in {"gfx942", "gfx950"}:
            raise ValueError(f"tlx_wave stage-1 scaffold only supports gfx942/gfx950, got {options.arch}")
        if options.warp_size != 64:
            raise ValueError(f"tlx_wave {options.arch} expects wave64, got warp_size={options.warp_size}")
        return options

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        tlx.tlx_passes.add_triton_tlx_fixup(pm, f"hip:{options.arch}", options.num_warps, options.warp_size,
                                            options.num_ctas, list((1, 1, 1)))
        passes.common.add_inliner(pm)
        # gfx942/gfx950 have no TDM path in this design; keep tensor descriptors on the
        # ordinary pointer path for the future Wave bridge.
        if not amd.supports_tdm(options.arch):
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        # Preserve TLX aliasing intent in concrete local_alias/memdesc form.
        tlx.tlx_passes.add_tlx_storage_alias_lowering(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "tlx_wave.make_ttir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        # The Wave bridge consumes the same TTGIR contract that AMD/LLVM
        # lowers. HIPBackend.make_ttgir already runs AMD layout cleanup; running
        # remove-layout-conversions again after CF lifting can reshape the
        # epilogue into a Wave-only form and drift from LLVM's memory/barrier
        # lowering.
        mod = amd_compiler.HIPBackend.make_ttgir(mod, metadata, options)
        passes.convert.triton_lift_cf_to_scf(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # Helpers with early returns can only become structurally inlineable
        # after CF lifting rewrites their bodies to single-block SCF regions.
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "tlx_wave.make_ttgir_post_cf_lift")
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_wave(src, metadata, options):
        # Preserve the same high-level warp-pipeline protocol consumed by the
        # LLVM backend. The conversion pass owns cluster formation, reuse
        # barriers, conditional phase barriers, and priority directives; the
        # Python bridge below only translates those explicit operations.
        pm = ir.pass_manager(src.context)
        pm.enable_debug()
        amd.passes.ttgpuir.add_update_async_wait_count(pm, options.arch)
        amd.passes.ttgpuir.add_warp_pipeline(pm)
        amd.passes.ttgpuir.add_warp_pipeline_conversion(pm, options.arch)
        pm.run(src, "tlx_wave.prepare_warp_pipeline")

        # Materialize compiler-owned local-memory barriers before importing
        # TTGIR.  These barriers include issue rendezvous between cross-wave
        # direct-to-LDS epochs; they do not complete an async DMA group, which
        # remains the sole responsibility of ttg.async_wait.
        allocation = passes.analysis.allocation(src)
        passes.analysis.membar(allocation).run()

        output = converter_pipeline.convert_ttgir_to_wave(
            src,
            enable_split_barriers=getattr(options, "tlx_wave_enable_split_barriers", False),
            waves_per_eu=getattr(options, "waves_per_eu", 0),
        )
        _validate_staged_converter_output(output, options)

        wave_text = output.emitted_module.text
        wave_opt = _wave_opt()
        _verify_wave_module(wave_text, wave_opt)
        _populate_staged_converter_metadata(metadata, output, options, wave_opt)
        return wave_text

    @staticmethod
    def make_hsaco(src, metadata, options):
        wave_opt = _wave_opt()
        hsaco = _compile_wave_module_to_hsaco(src, wave_opt, options.arch)
        metadata["tlx_wave_binary_stage"] = "wave-compile-kernels"
        metadata["tlx_wave_hsaco_size_bytes"] = len(hsaco)
        return hsaco

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise NotImplementedError("tlx_wave scaffold currently supports only Triton/TLX language input")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["wave"] = lambda src, metadata: self.make_wave(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    def hash(self):
        return (f"{self.target}:stage13-amd-membar-wave-scheduler-options:"
                f"wave-opt-sha256={_wave_opt_sha256()}:"
                f"wave-pipelines-sha256={_wave_pipelines_sha256()}")


def _validate_staged_converter_output(output, options):
    kernel = output.source_program.kernel
    expected_target = f"hip:{options.arch}"
    if kernel.target != expected_target:
        raise ValueError(f"tlx_wave staged converter expected TTGIR target {expected_target}, "
                         f"got {kernel.target}")
    if int(kernel.threads_per_warp or options.warp_size) != int(options.warp_size):
        raise ValueError("tlx_wave staged converter saw inconsistent wave size: "
                         f"TTGIR={kernel.threads_per_warp}, options={options.warp_size}")


def _populate_staged_converter_metadata(metadata, output, options, wave_opt):
    source_kernel = output.source_program.kernel
    target_program = output.target_program
    emitted = output.emitted_module
    wave_text = emitted.text
    target_values = {value.target_value_id: value for value in target_program.values}
    arg_types = [target_values[target_id].type for target_id in target_program.kernel.arg_target_ids]

    metadata["name"] = target_program.kernel.name
    # Match Triton's launch contract: metadata.shared is the dynamic LDS byte
    # count passed to hipModuleLaunchKernel. Bridge-managed shared allocations
    # are emitted as wave.alloc and resolved by the Wave pipeline into the
    # HSACO's group_segment_fixed_size, so passing them here would reserve the
    # same LDS twice.
    metadata["shared"] = 0
    metadata["tlx_wave_launch_shared_bytes"] = 0
    metadata["global_scratch_size"] = 0
    metadata["global_scratch_align"] = 1
    metadata["profile_scratch_size"] = 0
    metadata["profile_scratch_align"] = 1
    metadata["tlx_wave_status"] = "emitted_wave_staged_converter"
    metadata["tlx_wave_bridge_stage"] = "staged-converter"
    metadata["tlx_wave_wave_builder"] = "staged-converter"
    metadata["tlx_wave_emit_api"] = "structural-python"
    metadata["tlx_wave_plan_kind"] = "staged-converter"
    metadata["tlx_wave_arch"] = options.arch
    metadata["tlx_wave_enable_split_barriers"] = bool(getattr(options, "tlx_wave_enable_split_barriers", False))
    metadata["tlx_wave_ttgir_target"] = source_kernel.target
    metadata["tlx_wave_num_ctas"] = int(source_kernel.num_ctas or 1)
    metadata["tlx_wave_num_warps"] = int(source_kernel.num_warps or 1)
    metadata["tlx_wave_threads_per_warp"] = int(source_kernel.threads_per_warp or options.warp_size)
    metadata["tlx_wave_workgroup_size"] = (metadata["tlx_wave_num_warps"] * metadata["tlx_wave_threads_per_warp"])
    metadata["tlx_wave_num_kernel_args"] = len(target_program.kernel.arg_target_ids)
    metadata["tlx_wave_num_pointer_args"] = sum(1 for target_type in arg_types if target_type.kind == "pointer")
    metadata["tlx_wave_num_scalar_args"] = sum(1 for target_type in arg_types if target_type.kind == "scalar")
    metadata["tlx_wave_plan_num_values"] = len(target_program.values)
    metadata["tlx_wave_plan_num_ops"] = len(target_program.ops)
    metadata["tlx_wave_lds_size_bytes"] = emitted.lds_size
    metadata["tlx_wave_num_mmas"] = wave_text.count("waveamd.mma")
    metadata["tlx_wave_num_wave_joins"] = wave_text.count("wave.join")
    metadata["tlx_wave_num_async_waits"] = sum(1 for op in target_program.ops if op.kind == "async_wait")
    metadata["tlx_wave_num_dma_load_lds"] = wave_text.count("waveamd.dma_load_lds")
    metadata["tlx_wave_wave_opt"] = wave_opt
