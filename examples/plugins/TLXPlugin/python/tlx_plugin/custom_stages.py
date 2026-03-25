from triton._C.libtriton import ir, passes
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
        warp_size = getattr(options, 'warp_size', 64)

        def make_ttgir_wrapper(mod, metadata):
            # Run the plugin's ConvertTritonToTritonGPU first (handles TLX ops),
            # then delegate to the backend's make_ttgir for all remaining passes.
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.plugin.tlx_convert_triton_to_tritongpu(
                pm, [f"hip:{target}", str(options.num_warps),
                     str(warp_size), str(options.num_ctas)])
            pm.run(mod, 'tlx_convert_triton_to_tritongpu')
            return self.make_ttgir(mod, metadata, options)

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
