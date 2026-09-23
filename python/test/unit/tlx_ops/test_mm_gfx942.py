"""gfx942 L1 correctness for ``tlx.ops.mm``.

Runs the hardware-agnostic synthetic list plus every MM focus suite. L2 perf
selects only the running host's configured default.

A shape the op declines is reported as a skip with the reason, never as a
pass.
"""

import types

import pytest
import torch
import triton
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


@pytest.mark.parametrize("space", ["heuristic", "origami", "full"])
def test_space_reaches_requested_autotuner(monkeypatch, space):
    from triton.tlx.ops.kernels.mm import gfx942

    monkeypatch.setattr(gfx942, "_validate_operands", lambda *args: (819200, 1024, 192))

    class AutotunerReached(Exception):
        pass

    def tuned(requested_space, shape):
        assert requested_space == space
        assert shape == (819200, 1024, 192)
        raise AutotunerReached

    monkeypatch.setattr(gfx942, "_tuned", tuned)
    with pytest.raises(AutotunerReached):
        gfx942._gemm(object(), object(), out=object(), space=space)


def test_full_space_includes_shape_specific_incumbent():
    from triton.tlx.ops.kernels.mm import gfx942

    configs = gfx942._candidate_configs((2048, 10240, 25408))
    assert len(configs) == len(gfx942.CONFIGS()) + 1
    assert any(config.kwargs.get("SPLIT_M_128_32") for config in configs)


def test_unaligned_row_base_vectorizes():
    from triton.tlx.ops.kernels.mm import gfx942

    M, N, K = 128, 128, 70
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16).T
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    meta = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 64,
        "GROUP_M": 4,
        "NUM_XCDS": 8,
        "XCD_CHUNK": 8,
    }
    grid = (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )
    compiled = gfx942.matmul_kernel_gfx942[grid](
        a,
        b,
        out,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        0,
        0,
        out.stride(0),
        out.stride(1),
        ADD_BIAS=False,
        matrix_instr_nonkdim=16,
        num_warps=4,
        num_stages=2,
        **meta,
    )

    torch.testing.assert_close(out, a @ b, atol=1e-2, rtol=1e-2)
    a_loads = [line for line in compiled.asm["ttgir"].splitlines() if "amdg.buffer_load %a_ptr[" in line]
    assert any("contiguity = 8" in line for line in a_loads), "expected wide unaligned A load"
    assert "buffer_load_dwordx4" in compiled.asm["amdgcn"], "expected 16-byte VMEM load"


def test_gfx942_origami_ranking_preserves_tlx_configs():
    from triton.tlx.ops.kernels.mm import gfx942

    class Struct:
        pass

    class Dim3:

        def __init__(self, m, n, k):
            self.m, self.n, self.k = m, n, k

    seen = {}

    def rank_configs(problem, hardware, configs):
        seen["problem"] = problem
        seen["hardware"] = hardware
        seen["configs"] = configs
        return [types.SimpleNamespace(config=config) for config in reversed(configs)]

    fake_origami = types.SimpleNamespace(
        config_t=Struct,
        dim3_t=Dim3,
        get_hardware_for_device=lambda index: ("gfx942", index),
        problem_t=Struct,
        rank_configs=rank_configs,
        string_to_datatype=lambda name: name,
        transpose_t=types.SimpleNamespace(N="N", T="T"),
    )
    a = types.SimpleNamespace(
        device=types.SimpleNamespace(index=3),
        dtype=torch.float16,
        shape=(32, 64),
        stride=lambda: (64, 1),
    )
    b = types.SimpleNamespace(
        dtype=torch.float16,
        shape=(64, 48),
        stride=lambda: (1, 64),
    )
    configs = [
        gfx942._config(64, 64, 64, 4, 4),
        gfx942._config(64, 64, 64, 8, 4),
        gfx942._config(128, 128, 32, 8, 4, waves_per_eu=2),
    ]
    selected = gfx942._rank_configs_with_origami(
        fake_origami,
        configs,
        {"a_ptr": a, "b_ptr": b, "M": 32, "N": 48, "K": 64},
        top_k=2,
    )

    assert selected == [configs[2], configs[0], configs[1]]
    assert [config.occupancy for config in seen["configs"]] == [1, 2]
    assert all((config.mi.m, config.mi.n, config.mi.k) == (16, 16, 16) for config in seen["configs"])
    assert seen["problem"].a_transpose == "N"
    assert seen["problem"].b_transpose == "T"
    assert seen["hardware"] == ("gfx942", 3)


def test_gfx942_origami_missing_package_fails(monkeypatch):
    from triton.tlx.ops.kernels.mm import gfx942

    def missing_origami(name):
        raise ImportError("missing dependency")

    monkeypatch.setattr(gfx942.importlib, "import_module", missing_origami)
    with pytest.raises(RuntimeError, match="Could not import rocm-origami") as exc:
        gfx942._load_origami()
    assert isinstance(exc.value.__cause__, ImportError)


def test_gfx942_origami_keeps_heuristic_incumbent(monkeypatch):
    from triton.tlx.ops.kernels.mm import gfx942

    shape = (2048, 10240, 25408)
    configs = gfx942._candidate_configs(shape)
    monkeypatch.setattr(gfx942, "_load_origami", lambda: object())
    monkeypatch.setattr(gfx942, "_rank_configs_with_origami", lambda *args: [configs[0]])

    selected = gfx942._origami_prune_configs(
        configs,
        {"M": shape[0], "N": shape[1], "K": shape[2]},
    )

    assert selected[0] is configs[0]
    assert selected[-1].kwargs["SPLIT_M_128_32"]


def test_gfx942_origami_space_wiring():
    from triton.tlx.ops.kernels.mm import gfx942

    tuner = gfx942._tuned("origami", (128, 128, 128))
    assert len(tuner.configs) == len(gfx942.CONFIGS())
    assert tuner.early_config_prune is gfx942._origami_prune_configs
    assert {"stride_am", "stride_ak", "stride_bk", "stride_bn"} <= set(tuner.keys)


def test_gfx942_full_and_origami_share_candidate_universe():
    from triton.tlx.ops.kernels.mm import gfx942

    shape = (2048, 10240, 25408)
    full = gfx942._tuned("full", shape)
    origami = gfx942._tuned("origami", shape)
    assert full.configs == origami.configs
    assert any(config.kwargs.get("SPLIT_M_128_32") for config in full.configs)


@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", shapes())
def test_mm(M, N, K, a_strides, b_strides, dtype_name):
    run_mm_case(ARCH, M, N, K, a_strides, b_strides, dtype_name)
