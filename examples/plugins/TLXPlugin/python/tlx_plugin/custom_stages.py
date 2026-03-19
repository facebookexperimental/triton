from triton._C.libtriton import ir, passes
import hashlib
import pathlib


def get_key():
    return pathlib.Path(__file__).read_text()


def get_hash():
    return hashlib.sha256(get_key().encode('utf-8')).hexdigest()


def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    # If the hook is called with no arguments we assume were just after the key and hash and don't want to
    # actually execute the pipeline yet
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()

    def make_ttir_wrapper(mod, metadata, opt, capability):
        mod = self.make_ttir(mod, metadata, opt, capability)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.plugin.tlx_convert_triton_to_tritongpu(pm, [f"cuda:{capability}", str(opt.num_warps), '32', str(opt.num_ctas)])
        pm.run(mod, 'tlx_conversion')
        return mod

    stages["ttir"] = lambda src, metadata: make_ttir_wrapper(src, metadata, options, capability)

    return get_key(), get_hash()
