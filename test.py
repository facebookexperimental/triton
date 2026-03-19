import sys
import os
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
from tlx_plugin import custom_stages

from triton import knobs
knobs.runtime.add_stages_inspection_hook = custom_stages.inspect_stages_hook

@triton.jit
def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr, DTYPE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    buf = tlx.local_alloc((BLOCK,), DTYPE, 1)
    view = tlx.local_view(buf, 0)
    tlx.local_store(view, x)
    y = tlx.local_load(view)
    tl.store(out_ptr + offs, y)

BLOCK = 128
torch_dtype = tl.bfloat16
# {tl.float16: torch.float16, tl.float32: torch.float32,
#                tl.bfloat16: torch.bfloat16}[dtype]

x = torch.randn(BLOCK, device='cuda', dtype=torch.bfloat16)
out = torch.empty_like(x)

kernel[(1,)](x, out, BLOCK=BLOCK, DTYPE=tl.bfloat16)
torch.testing.assert_close(out, x)


