"""L1 correctness for ``tlx.ops.mm``.

Runs the synthetic list plus this arch's focus list -- the same set the perf
suite measures. Shapes live in ``triton.tlx.ops.kernels.mm._shapes``, so one
disabled for a correctness bug is automatically not benchmarked either.

A shape the op declines (sm100 requires 16-byte-aligned TMA strides; gfx942 has
no such constraint) is reported as a skip with the reason, never as a pass.
"""

import time

import pytest
import torch

from triton._internal_testing import is_blackwell, is_hip_cdna3
from triton.tlx.ops import InvalidInput, UnsupportedOp
from triton.tlx.ops.kernels.mm._shapes import SYNTHETIC, operand


def _arch():
    """The catalog key for the GPU under test, or None if mm has no entry."""
    if is_blackwell():
        return "sm100"
    if is_hip_cdna3():
        return "gfx942"
    return None


ARCH = _arch()


def _assert_strides(t, wanted):
    """The operand has the recorded layout.

    A dim of extent 1 is exempt: its stride addresses nothing, so a captured 0
    and torch's own value describe the same bytes.
    """
    for dim, (got, want) in enumerate(zip(t.stride(), wanted)):
        if t.shape[dim] != 1:
            assert got == want, f"dim {dim}: stride {got}, recorded {want}"


def _shapes():
    """Synthetic plus this arch's focus list -- not every arch's.

    An earlier version ran the union, on the argument that another arch's real
    geometries are free coverage. That stopped being free once the sm100 list
    grew past a handful: correctness would inherit the whole perf sweep.
    """
    import importlib

    if ARCH is None:
        return list(SYNTHETIC)
    focus = importlib.import_module(f"triton.tlx.ops.kernels.mm.{ARCH}").PERF_SHAPES
    return list(SYNTHETIC) + list(focus)


pytestmark = pytest.mark.skipif(ARCH is None, reason="tlx.ops.mm has no implementation for this GPU")

torch.manual_seed(0)

# The heuristic space builds exactly one config per shape, so exceeding this is
# a compile-time regression rather than a slow test.
MAX_SECONDS_PER_CASE = 60

REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}

# The reference below is only ground truth in true fp32.
assert torch.get_float32_matmul_precision() == "highest"


@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", _shapes())
def test_mm(M, N, K, a_strides, b_strides, dtype_name):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    from triton.tlx.ops import mm as tlx_mm

    a = operand(M, K, a_strides, dtype)
    b = operand(K, N, b_strides, dtype)
    _assert_strides(a, a_strides)
    _assert_strides(b, b_strides)

    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        out = tlx_mm(a, b, arch=ARCH, space="heuristic")
    except (InvalidInput, UnsupportedOp) as declined:
        # An arch refusing a shape from another arch's list is expected and is
        # not a failure. Skipping rather than passing keeps it visible: a shape
        # silently not covered here is a shape nobody is checking.
        pytest.skip(f"{ARCH} declines this shape: {declined}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert elapsed < MAX_SECONDS_PER_CASE, (f"mm({M}x{N}x{K}, {dtype}) took {elapsed:.1f}s, "
                                            f"over the {MAX_SECONDS_PER_CASE}s budget")

    ref = a.float() @ b.float()
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out.float(), ref, atol=precision * ref.abs().max().item(), rtol=precision)

    # These are multi-gigabyte operands at the top of the focus lists; without
    # this the union OOMs partway through rather than at a diagnosable point.
    del a, b, out, ref
    torch.cuda.empty_cache()
