"""L1 correctness for the gfx950 TorchTLX ``addmm`` provider."""

import pytest
import torch
import torch._inductor.kernel.mm as inductor_mm
from torch._inductor.utils import fresh_cache
from triton._internal_testing import is_hip_cdna4

try:
    from triton.tlx.ops.kernels.addmm import gfx950_torch
    from triton.tlx.ops.kernels.addmm._shapes import GFX950_FOCUS, SYNTHETIC, inputs
except ImportError:  # not the fbtriton fork
    gfx950_torch = None

pytestmark = pytest.mark.skipif(gfx950_torch is None or not is_hip_cdna4(), reason="needs MI350X, FBTriton")

torch.manual_seed(0)

REL_PRECISION = {torch.float16: 2e-2, torch.bfloat16: 2e-2}


def test_forced_mode_assesses_only_tlx_candidates(monkeypatch):
    seen = []
    select = inductor_mm.autotune_select_algorithm

    def spy(name, choices, *args, **kwargs):
        seen.extend(choice.name for choice in choices)
        return select(name, choices, *args, **kwargs)

    monkeypatch.setattr(inductor_mm, "autotune_select_algorithm", spy)
    entry = [512, 192, 512, (512, 1), (1, 512), "fp16"]
    bias, a, b = inputs(entry, torch.float16)
    torch._dynamo.reset()
    with fresh_cache():
        gfx950_torch.addmm(bias, a, b, mode="force")

    assert seen, "addmm did not reach the algorithm selector"
    others = [name for name in seen if "tlx_" not in name]
    assert not others, (f"tlx_mode=force must leave only TLX candidates, but the selector was also "
                        f"offered {others}; full choice list {seen}")


#: (M, N, K) shapes the TorchTLX path cannot run yet.
FAILED_SHAPES = []


def _cases():
    entries = [] if gfx950_torch is None else list(SYNTHETIC) + list(GFX950_FOCUS)
    return [entry for entry in entries if tuple(entry[:3]) not in FAILED_SHAPES]


@pytest.mark.parametrize("m,n,k,a_strides,b_strides,dtype_name", _cases())
def test_forced_mode_matches_eager(m, n, k, a_strides, b_strides, dtype_name):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    entry = [m, n, k, a_strides, b_strides, dtype_name]
    bias, a, b = inputs(entry, dtype)

    torch._dynamo.reset()
    out = gfx950_torch.addmm(bias, a, b, mode="force")

    ref = torch.addmm(bias, a, b)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)

    del bias, a, b, out, ref
    torch.cuda.empty_cache()
