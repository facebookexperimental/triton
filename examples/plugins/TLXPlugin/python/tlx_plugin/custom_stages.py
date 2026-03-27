from triton._C.libtriton import ir, passes, amd
from triton import knobs
import hashlib
import pathlib


def is_async_copy_enabled(arch):
    return (arch in ["gfx950", "gfx1250"]) if knobs.amd.use_async_copy is None else knobs.amd.use_async_copy


def is_pingpong_schedule_enabled(arch, use_async_copy):
    return (arch == "gfx942" or (arch == "gfx950" and use_async_copy is True)) \
        if knobs.amd.use_block_pingpong is None else knobs.amd.use_block_pingpong


def is_in_thread_transpose_enabled(arch):
    return (arch == "gfx942") if knobs.amd.use_in_thread_transpose is None else knobs.amd.use_in_thread_transpose


_cached_key = None
_cached_hash = None


def get_key():
    global _cached_key
    if _cached_key is None:
        _cached_key = pathlib.Path(__file__).read_text()
    return _cached_key


def get_hash():
    global _cached_hash
    if _cached_hash is None:
        _cached_hash = hashlib.sha256(get_key().encode('utf-8')).hexdigest()
    return _cached_hash


def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()

    backend_name = getattr(self, 'name', '') if self else ''
    target = getattr(options, 'arch', '') if options else ''

    if backend_name == 'amd' or (hasattr(options, 'arch') and str(target).startswith('gfx')):
        warp_size = getattr(options, 'warp_size', 64)

        def make_ttgir_wrapper(mod, metadata):
            # Phase 1: Plugin ConvertTritonToTritonGPU (make_ttgir_early)
            # This replaces the upstream ConvertTritonToTritonGPU and also
            # handles LocalStore/LocalLoad/AsyncCopy ops that upstream doesn't.
            # We do NOT run upstream add_convert_to_ttgpuir again — the plugin
            # version sets the same module attributes and converts the same ops.
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.plugin.tlx_convert_triton_to_tritongpu(
                pm, [f"hip:{target}", str(options.num_warps),
                     str(warp_size), str(options.num_ctas)])
            pm.run(mod, 'make_ttgir_early')

            # Phase 2: AMD TTGIR pipeline with TLX layout pass injected
            # This replicates the upstream AMD make_ttgir pipeline exactly,
            # with tlx_insert_and_propagate_layout added right after
            # accelerate_matmul and before remove_layout_conversions —
            # matching the in-tree TLX ordering.
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            emuTF32 = False
            passes.ttgpuir.add_coalesce(pm)
            passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.ttgpuir.add_optimize_thread_locality(pm)

            amd.passes.ttgpuir.add_accelerate_matmul(
                pm, options.arch, options.matrix_instr_nonkdim, options.kpack)
            # TLX: fix MemDesc encodings to match DotOp operand requirements
            passes.plugin.tlx_insert_and_propagate_layout(pm, [])

            passes.ttgpuir.add_remove_layout_conversions(pm)
            amd.passes.ttgpuir.add_optimize_epilogue(pm)
            amd.passes.ttgpuir.add_optimize_dot_operands(pm, options.arch)
            amd.passes.ttgpuir.add_hoist_layout_conversions(pm)
            amd.passes.ttgpuir.add_sink_layout_conversions(pm)

            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)

            use_async_copy = is_async_copy_enabled(options.arch)
            use_block_pingpong = is_pingpong_schedule_enabled(
                options.arch, use_async_copy)

            amd.passes.ttgpuir.add_schedule_loops(pm, options.num_stages)
            amd.passes.ttgpuir.add_pipeline(pm, use_async_copy,
                                            use_block_pingpong)
            if use_async_copy:
                amd.passes.ttgpuir.add_coalesce_async_copy(pm, options.arch)
            amd.passes.ttgpuir.add_convert_to_tensor_ops(pm)
            passes.common.add_canonicalizer(pm)
            if options.schedule_hint.lower() != "none":
                for hint in options.schedule_hint.split(","):
                    amd.passes.ttgpuir.insert_instruction_sched_hints(
                        pm, hint)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.ttgpuir.add_reduce_data_duplication(pm)
            if is_in_thread_transpose_enabled(options.arch):
                amd.passes.ttgpuir.add_in_thread_transpose(pm)
                passes.ttgpuir.add_remove_layout_conversions(pm)
            amd.passes.ttgpuir.add_move_up_prologue_loads(pm)
            if use_block_pingpong and options.num_stages > 1:
                amd.passes.ttgpuir.add_block_pingpong(pm, options.num_stages)

            if knobs.amd.use_buffer_ops:
                amd.passes.ttgpuir.add_canonicalize_pointers(pm)
                passes.common.add_canonicalizer(pm)
                amd.passes.ttgpuir.add_convert_to_buffer_ops(
                    pm,
                    options.arch,
                    knobs.amd.use_buffer_atomics,
                    knobs.amd.buffer_ops_analyze_small_tensor_range,
                )
                amd.passes.ttgpuir.add_optimize_buffer_op_ptr(pm)

            amd.passes.ttgpuir.add_fold_true_cmpi(pm)
            amd.passes.ttgpuir.add_prepare_if_combining(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.common.add_symbol_dce(pm)
            if getattr(options, 'instrumentation_mode', 'none') == "fpsan":
                amd.passes.ttgpuir.add_fp_sanitizer(pm)
                passes.ttgpuir.add_fp_sanitizer(pm)
            pm.run(mod, 'make_ttgir')
            metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
            return mod

        stages["ttgir"] = lambda src, metadata: make_ttgir_wrapper(src, metadata)
    else:
        # NVIDIA/CUDA: replace make_ttir to inject plugin pass after TTIR
        def make_ttir_wrapper(mod, metadata, opt, cap):
            mod = self.make_ttir(mod, metadata, opt, cap)
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.plugin.tlx_convert_triton_to_tritongpu(
                pm, [f"cuda:{cap}", str(opt.num_warps), '32',
                     str(opt.num_ctas)])
            pm.run(mod, 'tlx_conversion')
            return mod

        stages["ttir"] = lambda src, metadata: make_ttir_wrapper(
            src, metadata, options, capability)

    return get_key(), get_hash()
