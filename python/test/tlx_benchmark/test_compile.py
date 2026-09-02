"""Unit tests for the cold-compile guard. No GPU required."""

import os

import triton

from _harness import COLD_COMPILE_CAP_S, CompileStat, fresh_triton_cache


def test_cap_is_the_absolute_two_minutes():
    """The compile guard is an absolute ceiling, not a ratio against a
    baseline: cuBLAS has no compile step to be relatively worse than, and a
    relative gate ratchets -- three 15% regressions pass individually and
    double the wait."""
    assert COLD_COMPILE_CAP_S == 120.0


def test_over_cap():
    assert CompileStat(t_cold_s=119.0).over_cap is False
    assert CompileStat(t_cold_s=121.0).over_cap is True
    # Measured: tlx.ops.mm at space="full", 1024^3, on B200.
    assert CompileStat(t_cold_s=284.6, n_compiles=350, n_configs=348).over_cap is True


def test_to_dict_carries_the_diagnostic_counts():
    d = CompileStat(t_cold_s=284.6, n_compiles=350, n_configs=348).to_dict()
    assert d == {"t_cold_s": 284.6, "n_compiles": 350, "n_configs": 348, "cap_s": 120.0, "over_cap": True}


def test_fresh_triton_cache_restores_both_knob_and_env():
    before_knob = triton.knobs.cache.dir
    before_env = os.environ.get("TRITON_CACHE_DIR")
    with fresh_triton_cache() as tmp:
        assert triton.knobs.cache.dir == tmp
        assert os.environ["TRITON_CACHE_DIR"] == tmp
        assert os.path.isdir(tmp)
    assert triton.knobs.cache.dir == before_knob
    assert os.environ.get("TRITON_CACHE_DIR") == before_env
    assert not os.path.exists(tmp)


def test_fresh_triton_cache_restores_after_an_exception():
    before = triton.knobs.cache.dir
    try:
        with fresh_triton_cache():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert triton.knobs.cache.dir == before
