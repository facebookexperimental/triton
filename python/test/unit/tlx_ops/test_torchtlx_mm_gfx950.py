"""L1 correctness for the gfx950 TorchTLX ``mm`` provider.

Runs the hardware-agnostic synthetic cases plus the gfx950 focus suites. Other
architectures' focus suites may contain layouts for which gfx950 cannot emit a
TLX candidate, so they belong to their corresponding provider tests.
"""

import pytest
import torch
import torch._inductor.kernel.mm as inductor_mm
from torch._inductor.utils import fresh_cache
from triton._internal_testing import is_hip_cdna4

try:
    from triton.language.extra.tlx.inductor import gfx950_torch, tlx_config
    from triton.tlx.ops.kernels.mm._shapes import FOCUS, SYNTHETIC, operand
except ImportError:  # not the fbtriton fork
    gfx950_torch = None

pytestmark = pytest.mark.skipif(gfx950_torch is None or not is_hip_cdna4(), reason="needs MI350X, FBTriton")

torch.manual_seed(0)

REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}


def test_catalog_resolves_inductor_provider():
    from triton.tlx.ops._catalog import impl_for

    implementation, _ = impl_for("mm_torchtlx", arch="gfx950")
    assert implementation is gfx950_torch.mm


def test_forced_mode_assesses_only_tlx_candidates(monkeypatch):
    seen = []
    select = inductor_mm.autotune_select_algorithm

    def spy(name, choices, *args, **kwargs):
        seen.extend(choice.name for choice in choices)
        return select(name, choices, *args, **kwargs)

    monkeypatch.setattr(inductor_mm, "autotune_select_algorithm", spy)
    x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    torch._dynamo.reset()
    with fresh_cache(), tlx_config.patch(use_heuristic_config=True):
        gfx950_torch.mm(x, x, mode="force")

    assert seen, "mm did not reach the algorithm selector"
    others = [name for name in seen if "tlx_" not in name]
    assert not others, (f"tlx_mode=force must leave only TLX candidates, but the selector was also "
                        f"offered {others}; full choice list {seen}")


# TODO: Re-enable shapes here when their TorchTLX correctness failures are fixed.
FAILED_SHAPES = {
    (1024, 2048, 512, (1, 1024), (2048, 1), "fp16"),
    (2048, 2048, 2048, (1, 2048), (1, 2048), "fp16"),
    (1000, 1000, 1024, (1024, 1), (1000, 1), "fp16"),
    (1000, 1000, 200, (200, 1), (1000, 1), "fp16"),
    (1024, 2048, 512, (1, 1024), (2048, 1), "bf16"),
    (2048, 2048, 2048, (1, 2048), (1, 2048), "bf16"),
    (1000, 1000, 1024, (1024, 1), (1000, 1), "bf16"),
    (1000, 1000, 200, (200, 1), (1000, 1), "bf16"),
    (4096, 1894, 242432, (242432, 1), (1, 242432), "bf16"),
    (2048, 25408, 10240, (10240, 1), (1, 10240), "bf16"),
    (819200, 192, 1024, (1024, 1), (192, 1), "fp16"),
    (4096, 242432, 1894, (1894, 1), (242432, 1), "fp16"),
}


def _cases():
    entries = [] if gfx950_torch is None else (*SYNTHETIC, *FOCUS.shapes("gfx950"))
    return [entry for entry in dict.fromkeys(entries) if tuple(entry) not in FAILED_SHAPES]


@pytest.mark.parametrize("M,N,K,a_strides,b_strides,dtype_name", _cases())
def test_forced_mode_matches_eager(M, N, K, a_strides, b_strides, dtype_name):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    a, b = operand(M, K, a_strides, dtype), operand(K, N, b_strides, dtype)

    torch._dynamo.reset()
    with tlx_config.patch(use_heuristic_config=True):
        out = gfx950_torch.mm(a, b, mode="force")

    ref = torch.matmul(a, b)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)

    del a, b, out, ref
    torch.cuda.empty_cache()
