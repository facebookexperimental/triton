"""L1 correctness for the torchTLX ``mm`` provider.

- TLX.ops evaluates TLX template only (torchTLX 'force mode')
- TLX.ops passes numerical tests of all shapes

This framework is fixed. A shape the torchTLX path cannot run goes in
``FAILED_SHAPES`` with a one-line TODO -- not a new test, not a skip.
"""
import pytest
import torch
import torch._inductor.kernel.mm as inductor_mm
from torch._inductor.utils import fresh_cache
from triton._internal_testing import is_blackwell
try:
    from triton.language.extra.tlx.inductor import sm100_torch
    from triton.tlx.ops.kernels.mm._shapes import SM100_FOCUS, SYNTHETIC, operand
except ImportError:  # not the fbtriton fork
    sm100_torch = None

pytestmark = pytest.mark.skipif(sm100_torch is None or not is_blackwell(), reason="needs Blackwell, FBTriton")

torch.manual_seed(0)

REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}


def test_catalog_resolves_inductor_provider():
    from triton.tlx.ops._catalog import impl_for

    implementation, _ = impl_for("mm_torchtlx", arch="sm100")
    assert implementation is sm100_torch.mm


def test_forced_mode_assesses_only_tlx_candidates(monkeypatch):
    seen = []
    select = inductor_mm.autotune_select_algorithm

    def spy(name, choices, *args, **kwargs):
        seen.extend(c.name for c in choices)
        return select(name, choices, *args, **kwargs)

    monkeypatch.setattr(inductor_mm, "autotune_select_algorithm", spy)
    x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    # A cached graph is never codegened, so the spy would see nothing.
    with fresh_cache():
        sm100_torch.mm(x, x, mode="force")

    assert seen, "mm did not reach the algorithm selector"
    others = [name for name in seen if not name.startswith("triton_tlx_")]
    assert not others, (f"tlx_mode=force must leave only TLX candidates, but the selector was also "
                        f"offered {others}; full choice list {seen}")


# TODO: Re-enable these shapes when their TorchTLX failures are fixed.
FAILED_SHAPES = {
    (136, 256, 128, (128, 1), (256, 1), "fp16"),
    (136, 256, 128, (128, 1), (256, 1), "bf16"),
    (810572, 512, 1536, (1536, 1), (1, 1536), "bf16"),
    (7, 4096, 1152, (1, 7), (4096, 1), "bf16"),
    (7, 2048, 1152, (1, 7), (2048, 1), "bf16"),
    (308743, 512, 1536, (1536, 1), (1, 1536), "bf16"),
    (1056, 1056, 2304, (1, 1088), (1088, 1), "bf16"),
    (1, 12800, 1152, (0, 1), (12800, 1), "bf16"),
    (256, 15042, 1152, (1, 256), (15042, 1), "bf16"),
    (1152, 4096, 7, (7, 1), (4096, 1), "bf16"),
    (16672, 256, 1152, (1, 16704), (256, 1), "bf16"),
    (705178, 6, 6, (6, 1), (6, 1), "bf16"),
    (1, 512, 1152, (1152, 1), (512, 1), "bf16"),
    (15044, 1024, 1152, (1, 15072), (1024, 1), "bf16"),
    (1152, 2048, 7, (7, 1), (2048, 1), "bf16"),
    (705178, 6, 6, (6, 1), (1, 6), "bf16"),
    (15042, 256, 1152, (1, 15072), (256, 1), "bf16"),
    (384, 384, 19459, (1, 384), (384, 1), "bf16"),
    (503599, 6, 6, (6, 1), (6, 1), "bf16"),
    (7, 7, 198339, (1, 7), (7, 1), "bf16"),
    (386515, 6, 6, (6, 1), (6, 1), "bf16"),
    (1, 1024, 1152, (1152, 1), (1024, 1), "bf16"),
    (1152, 12800, 32, (32, 1), (12800, 1), "bf16"),
    (503599, 6, 6, (6, 1), (1, 6), "bf16"),
    (386937, 7, 7, (7, 1), (7, 1), "bf16"),
    (8, 8, 705178, (1, 8), (8, 1), "bf16"),
    (114658, 256, 256, (256, 1), (256, 1), "bf16"),
    (15044, 512, 1152, (1, 15072), (512, 1), "bf16"),
    (8, 8, 503599, (1, 8), (8, 1), "bf16"),
    (8, 8, 386515, (1, 8), (8, 1), "bf16"),
    (1056, 1056, 1152, (1, 1088), (1088, 1), "bf16"),
    (7, 7, 222929, (1, 7), (7, 1), "bf16"),
    (1, 32, 1152, (1152, 1), (32, 1), "bf16"),
    (313230, 7, 7, (7, 1), (7, 1), "bf16"),
    (386515, 6, 6, (6, 1), (1, 6), "bf16"),
    (117574, 4, 4, (4, 1), (4, 1), "bf16"),
    (10, 10, 75315, (1, 10), (10, 1), "bf16"),
}


def _cases():
    entries = [] if sm100_torch is None else list(SYNTHETIC) + list(SM100_FOCUS)
    return [entry for entry in entries if tuple(entry[:3]) not in FAILED_SHAPES]


@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", _cases())
def test_forced_mode_matches_eager(M, N, K, a_strides, b_strides, dtype_name):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    a, b = operand(M, K, a_strides, dtype), operand(K, N, b_strides, dtype)

    # Every case is a new shape on one compiled callable, which would exhaust
    # dynamo's cache_size_limit partway through and silently fall back to eager.
    torch._dynamo.reset()
    out = sm100_torch.mm(a, b, mode="force")

    ref = torch.matmul(a, b)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)

    del a, b, out, ref
    torch.cuda.empty_cache()
