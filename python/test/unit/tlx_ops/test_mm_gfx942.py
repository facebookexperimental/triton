"""gfx942 L1 correctness for ``tlx.ops.mm``.

Runs the hardware-agnostic synthetic list plus every MM focus suite. L2 perf
selects only the running host's configured default.

A shape the op declines is reported as a skip with the reason, never as a
pass.
"""

import pytest
from triton._internal_testing import is_hip_cdna3
from mm_test_utils import run_mm_case, shapes

pytestmark = pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942")

ARCH = "gfx942"


def test_gfx942_wide_heuristic_config():
    from triton.tlx.ops.kernels.mm import gfx942

    (config, ) = gfx942.heuristic_config(2048, 10240, 25408)
    assert config.kwargs["BLOCK_M"] == 160
    assert config.kwargs["BLOCK_N"] == 512
    assert config.kwargs["BLOCK_K"] == 32
    assert config.kwargs["GROUP_M"] == 8
    assert config.kwargs["XCD_CHUNK"] == 8
    assert config.kwargs["SPLIT_M_128_32"]
    assert config.num_warps == 8
    assert config.num_stages == 2


def test_full_space_does_not_use_tuned_config(monkeypatch):
    from triton.tlx.ops.kernels.mm import gfx942

    monkeypatch.setattr(gfx942, "_validate_operands", lambda *args: (819200, 1024, 192))
    monkeypatch.setattr(gfx942, "_align_rows", lambda tensor: tensor)

    def fail_tuned_config(*args):
        pytest.fail("space='full' must not use the frozen heuristic config")

    monkeypatch.setattr(gfx942, "tuned_config", fail_tuned_config)

    class FullSpaceReached(Exception):
        pass

    def full_space(space, shape):
        assert space == "full"
        assert shape is None
        raise FullSpaceReached

    monkeypatch.setattr(gfx942, "_tuned", full_space)
    with pytest.raises(FullSpaceReached):
        gfx942._gemm(object(), object(), out=object(), space="full")


@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", shapes())
def test_mm(M, N, K, a_strides, b_strides, dtype_name):
    run_mm_case(ARCH, M, N, K, a_strides, b_strides, dtype_name)
