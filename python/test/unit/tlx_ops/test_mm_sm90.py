"""sm90 L1 correctness for ``tlx.ops.mm``."""

import contextlib

import pytest
import torch
from triton._internal_testing import is_hopper
from triton.tlx.ops import InvalidInput
from triton.tlx.ops.kernels.mm import sm90
from triton.tlx.ops.kernels.mm._shapes import SM90_FOCUS, SYNTHETIC

from mm_test_utils import REL_PRECISION, run_mm_case

pytestmark = pytest.mark.skipif(not is_hopper(), reason="Requires sm90")

ARCH = "sm90"
COOPERATIVE_CONFIGS = (
    (128, 256, 3),
    (128, 256, 4),
    (256, 128, 3),
    (256, 128, 4),
)


@pytest.mark.parametrize(
    "M, N, K, a_strides, b_strides, dtype_name",
    list(SYNTHETIC) + list(SM90_FOCUS),
)
def test_mm(M, N, K, a_strides, b_strides, dtype_name):
    run_mm_case(
        ARCH,
        M,
        N,
        K,
        a_strides,
        b_strides,
        dtype_name,
        allow_decline=False,
    )


def _inputs(M=256, N=256, K=128, dtype=torch.float16):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    return a, b


def _assert_close(actual, expected):
    precision = REL_PRECISION[actual.dtype]
    torch.testing.assert_close(
        actual,
        expected,
        atol=precision * expected.abs().max().item(),
        rtol=precision,
    )


def test_smoke_space_is_registered_and_runs():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs()
    actual = tlx_mm(a, b, space="smoke")
    _assert_close(actual, torch.matmul(a, b))


def test_out():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs(dtype=torch.bfloat16)
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    actual = tlx_mm(a, b, out=out, space="heuristic")
    assert actual is out
    _assert_close(actual, torch.matmul(a, b))


def test_rejects_invalid_out():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs()
    out = torch.empty((a.shape[0] - 1, b.shape[1]), device=a.device, dtype=a.dtype)
    with pytest.raises(InvalidInput, match="out must be"):
        tlx_mm(a, b, out=out)


def test_rejects_unknown_space():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs()
    with pytest.raises(InvalidInput, match="does not provide search space"):
        tlx_mm(a, b, space="unknown")


def test_rejects_unaligned_operand():
    from triton.tlx.ops import mm as tlx_mm

    storage = torch.randn((256 * 128 + 1, ), device="cuda", dtype=torch.float16)
    a = storage[1:].view(256, 128)
    _, b = _inputs()
    assert a.is_contiguous() and a.data_ptr() % 16 != 0
    with pytest.raises(InvalidInput, match="16-byte-aligned operands"):
        tlx_mm(a, b)


def test_rejects_unaligned_out():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs()
    storage = torch.empty((256 * 256 + 1, ), device="cuda", dtype=torch.float16)
    out = storage[1:].view(256, 256)
    assert out.is_contiguous() and out.data_ptr() % 16 != 0
    with pytest.raises(InvalidInput, match="16-byte-aligned output"):
        tlx_mm(a, b, out=out)


@contextlib.contextmanager
def _pinned_full_config(block_m, block_n, num_stages):
    original = sm90.CONFIGS
    sm90.CONFIGS = lambda: [sm90._config(block_m, block_n, num_stages)]
    sm90._tuned.cache_clear()
    try:
        yield
    finally:
        sm90.CONFIGS = original
        sm90._tuned.cache_clear()


@pytest.mark.parametrize("block_m, block_n, num_stages", COOPERATIVE_CONFIGS)
def test_cooperative_config(block_m, block_n, num_stages):
    from triton.tlx.ops import mm as tlx_mm

    # Five K tiles wrap both stage rings; 153 output tiles also make some SMs
    # revisit the persistent loop on an H100.
    a, b = _inputs(M=2176, N=2176, K=320)
    with _pinned_full_config(block_m, block_n, num_stages):
        actual = tlx_mm(a, b, space="full")

    _assert_close(actual, torch.matmul(a, b))
