from triton._C.libtriton import ir, passes, amd as _amd_c
import hashlib
import pathlib


def get_key():
    return pathlib.Path(__file__).read_text()


def get_hash():
    return hashlib.sha256(get_key().encode('utf-8')).hexdigest()


def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    # If the hook is called with no arguments we assume we're just after the
    # key and hash and don't want to actually execute the pipeline yet
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()

    backend_name = getattr(self, 'name', '') if self else ''
    target = getattr(options, 'arch', '') if options else ''

    if backend_name == 'amd' or (hasattr(options, 'arch') and str(target).startswith('gfx')):
        # AMD/HIP: replace make_ttgir to use the plugin's ConvertTritonToTritonGPU
        # which adds legalization for ttg.local_store/local_load/async_copy ops
        warp_size = getattr(options, 'warp_size', 64)

        def make_ttgir_wrapper(mod, metadata):
            # Step 1: Run the plugin's ConvertTritonToTritonGPU pass
            # (replaces passes.ttir.add_convert_to_ttgpuir)
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.plugin.tlx_convert_triton_to_tritongpu(
                pm, [f"hip:{target}", str(options.num_warps),
                     str(warp_size), str(options.num_ctas)])
            pm.run(mod, 'tlx_convert_triton_to_tritongpu')

            # Step 2: Run the remaining ttgir passes (matching AMD compiler.py make_ttgir)
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.ttgpuir.add_coalesce(pm)
            passes.ttgpuir.add_f32_dot_tc(pm, False)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.ttgpuir.add_optimize_thread_locality(pm)
            _amd_c.passes.ttgpuir.add_accelerate_matmul(
                pm, options.arch,
                getattr(options, 'matrix_instr_nonkdim', 0),
                getattr(options, 'kpack', 1))
            passes.ttgpuir.add_remove_layout_conversions(pm)
            _amd_c.passes.ttgpuir.add_optimize_epilogue(pm)
            _amd_c.passes.ttgpuir.add_optimize_dot_operands(pm, options.arch)
            _amd_c.passes.ttgpuir.add_hoist_layout_conversions(pm)
            _amd_c.passes.ttgpuir.add_sink_layout_conversions(pm)

            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)

            num_stages = getattr(options, 'num_stages', 0)
            _amd_c.passes.ttgpuir.add_schedule_loops(pm, num_stages)
            _amd_c.passes.ttgpuir.add_pipeline(pm, False, False)
            _amd_c.passes.ttgpuir.add_convert_to_tensor_ops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.ttgpuir.add_reduce_data_duplication(pm)
            _amd_c.passes.ttgpuir.add_move_up_prologue_loads(pm)

            _amd_c.passes.ttgpuir.add_fold_true_cmpi(pm)
            _amd_c.passes.ttgpuir.add_prepare_if_combining(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.common.add_symbol_dce(pm)
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
