import importlib
import sys
import os
import pytest
import torch

import triton
import triton.language as tl

# Add the plugin Python package to the path.
# Resolve to absolute path so it works regardless of cwd.
_plugin_python_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "examples", "plugins", "TLXPlugin", "python")
)
if _plugin_python_dir not in sys.path:
    sys.path.insert(0, _plugin_python_dir)
import tlx_plugin as tlx  # type: ignore[import-not-found]


def is_hopper_or_newer():
    try:
        return torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


# from triton._C import ir, passes
from triton._C.libtriton import ir, passes
import hashlib
import pathlib
from triton import knobs


# These two methods must be implemented and returned by the plugin hook.
# any changes in this entire file and the the plugin pipeline
# will trigger a recompile since the hash will change. To be
# less conservative, we could use a hash of the inspect_stages_hook
# function but then changes outside of the function won't be considered
# potentially causing a stale kernel hash
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

@triton.jit
def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr, DTYPE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    buf = tlx.local_alloc((BLOCK,), DTYPE, 1)
    view = tlx.local_view(buf, 0)
    tlx.local_store(view, x)
    y = tlx.local_load(view)
    tl.store(out_ptr + offs, y)

knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
BLOCK = 128
torch_dtype = tl.bfloat16
# {tl.float16: torch.float16, tl.float32: torch.float32,
#                tl.bfloat16: torch.bfloat16}[dtype]

x = torch.randn(BLOCK, device='cuda', dtype=torch.bfloat16)
out = torch.empty_like(x)

kernel[(1,)](x, out, BLOCK=BLOCK, DTYPE=tl.bfloat16)
torch.testing.assert_close(out, x)


