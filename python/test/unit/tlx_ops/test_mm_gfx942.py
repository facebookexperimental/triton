"""gfx942 L1 correctness for ``tlx.ops.mm``.

Runs the hardware-agnostic synthetic list plus every MM focus suite. L2 perf
selects only the running host's configured default.

A shape the op declines is reported as a skip with the reason, never as a
pass.
"""

import pytest
import torch
import triton
from triton._internal_testing import is_hip_cdna3
from triton.tlx.ops import BitwiseInvarianceSpec
from mm_test_utils import run_mm_case, run_mm_invariance_case, shapes

pytestmark = pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942")

ARCH = "gfx942"


@pytest.mark.parametrize("m", [4178, 4942])
@pytest.mark.parametrize("n", [256, 512, 1280, 2560])
def test_deep_k_batch_boundaries(m, n):
    import triton.tlx.ops as ops

    k = 18688
    torch.manual_seed(m + n + k)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).t()
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    expected = torch.mm(a, b)
    for _ in range(3):
        assert ops.mm(a, b, out=out, space="heuristic") is out
        torch.testing.assert_close(out, expected, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("space", ["heuristic", "full"])
def test_global_split_k_disabled_uses_direct_path(monkeypatch, space):
    from triton.tlx.ops.kernels.mm import gfx942

    m, n, k = 4178, 256, 18688
    a = torch.empty((m, k), device="cuda", dtype=torch.float16)
    b = torch.empty((n, k), device="cuda", dtype=torch.float16).t()
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    assert gfx942.heuristic_config(m, n, k)[0].kwargs["SPLIT_K"] > 1
    monkeypatch.setattr(gfx942, "ENABLE_GLOBAL_SPLIT_K", False)
    assert "SPLIT_K" not in gfx942.heuristic_config(m, n, k)[0].kwargs
    assert all(
        config.kwargs.get("SPLIT_K", 1) == 1
        for config in gfx942._candidate_configs((m, n, k), enable_global_split_k=True))
    assert not gfx942._precheck_global_split_k(a, b, None, out)

    class DirectPathReached(Exception):
        pass

    def tuned(requested_space, shape, enable_local_split_u):
        assert requested_space == space
        assert shape == (m, n, k)
        raise DirectPathReached

    monkeypatch.setattr(gfx942, "_tuned", tuned)
    with pytest.raises(DirectPathReached):
        gfx942._gemm(a, b, out=out, space=space)


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


@pytest.mark.parametrize("space", ["heuristic", "full"])
def test_space_reaches_requested_autotuner(monkeypatch, space):
    from triton.tlx.ops.kernels.mm import gfx942

    monkeypatch.setattr(gfx942, "_validate_operands", lambda *args: (819200, 1024, 192))
    monkeypatch.setattr(gfx942, "_precheck_local_split_u", lambda *args: False)

    class AutotunerReached(Exception):
        pass

    def tuned(requested_space, shape, enable_local_split_u):
        assert requested_space == space
        assert shape == (819200, 1024, 192)
        assert not enable_local_split_u
        raise AutotunerReached

    monkeypatch.setattr(gfx942, "_tuned", tuned)
    with pytest.raises(AutotunerReached):
        gfx942._gemm(object(), object(), out=object(), space=space)


def test_full_space_includes_shape_specific_incumbent():
    from triton.tlx.ops.kernels.mm import gfx942

    configs = gfx942._candidate_configs((2048, 10240, 25408))
    assert len(configs) == len(gfx942.CONFIGS()) + 1
    assert any(config.kwargs.get("SPLIT_M_128_32") for config in configs)


@pytest.mark.parametrize("shape", [(7, 8192, 2048), (7, 2048, 4096)])
def test_full_space_adds_local_split_u_candidate(shape):
    from triton.tlx.ops.kernels.mm import gfx942

    direct_configs = gfx942._candidate_configs(shape)
    configs = gfx942._candidate_configs(shape, enable_local_split_u=True)
    local_configs = [config for config in configs if config.kwargs.get("USE_LOCAL_SPLIT_U")]

    assert len(configs) == len(direct_configs) + 1
    assert len(local_configs) == 1
    plan = gfx942._MEASURED_LOCAL_SPLIT_U_PLANS[shape]
    assert local_configs[0].kwargs["BLOCK_M"] == plan.tile_m
    assert local_configs[0].kwargs["BLOCK_N"] == plan.tile_n
    assert local_configs[0].kwargs["BLOCK_K"] == plan.wave_k
    assert local_configs[0].num_warps == plan.local_split_u


def test_heuristic_space_prunes_measured_local_split_u(monkeypatch):
    from triton.tlx.ops.kernels.mm import gfx942

    shape = (7, 8192, 2048)
    out = object()
    monkeypatch.setattr(gfx942, "_validate_operands", lambda *args: shape)
    monkeypatch.setattr(gfx942, "_precheck_local_split_u", lambda *args: True)

    def launch(*args):
        raise AssertionError("measured LocalSplitU plan must be pruned under a fixed schedule")

    monkeypatch.setattr(gfx942, "_launch_local_split_u", launch)

    class AutotunerReached(Exception):
        pass

    def tuned(requested_space, shape_arg, enable_local_split_u):
        assert requested_space == "heuristic"
        assert shape_arg == shape
        assert not enable_local_split_u
        raise AutotunerReached

    monkeypatch.setattr(gfx942, "_tuned", tuned)
    with pytest.raises(AutotunerReached):
        gfx942._gemm(object(), object(), out=out, space="heuristic", invariance=BitwiseInvarianceSpec.dims("M"))


def test_heuristic_space_launches_local_split_u_directly(monkeypatch):
    from triton.tlx.ops.kernels.mm import gfx942

    shape = (7, 8192, 2048)
    out = object()
    monkeypatch.setattr(gfx942, "_validate_operands", lambda *args: shape)
    monkeypatch.setattr(gfx942, "_precheck_local_split_u", lambda *args: True)

    def launch(a, b, out, plan):
        assert plan == gfx942._MEASURED_LOCAL_SPLIT_U_PLANS[shape]
        return out

    monkeypatch.setattr(gfx942, "_launch_local_split_u", launch)
    assert gfx942._gemm(object(), object(), out=out, space="heuristic") is out


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


@pytest.mark.parametrize("K", [64, 65, 70, 127, 129, 255, 513, 527, 528, 529])
@pytest.mark.parametrize("block_k", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("add_bias", [False, True])
def test_mm_small_k_tail(K, block_k, dtype, add_bias):
    from triton.tlx.ops.kernels.mm import gfx942

    torch.manual_seed(0)
    M, N = 67, 73
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((N, K), device="cuda", dtype=dtype).T
    bias = torch.randn((M, N), device="cuda", dtype=dtype)
    out = torch.empty((M, N), device="cuda", dtype=dtype)
    gfx942.matmul_kernel_gfx942[(triton.cdiv(M, 64) * triton.cdiv(N, 64), )](
        a,
        b,
        bias,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias.stride(0),
        bias.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=block_k,
        GROUP_M=4,
        NUM_XCDS=8,
        XCD_CHUNK=8,
        ADD_BIAS=add_bias,
        matrix_instr_nonkdim=16,
        num_warps=4,
        num_stages=2,
    )
    expected = a.float() @ b.float()
    if add_bias:
        expected += bias.float()
    expected = expected.to(dtype)
    precision = 1e-3 if dtype == torch.float16 else 8e-3
    torch.testing.assert_close(out, expected, atol=precision * expected.abs().max().item(), rtol=precision)


@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", shapes())
def test_mm(M, N, K, a_strides, b_strides, dtype_name):
    run_mm_case(ARCH, M, N, K, a_strides, b_strides, dtype_name)


_INVARIANCE_SPECS = [
    BitwiseInvarianceSpec.dims("M"),
    BitwiseInvarianceSpec.dims("K"),
    BitwiseInvarianceSpec.all(),
]

_INVARIANCE_SHAPES = [
    # LocalSplitU shape: the spec prunes the measured LSU plan.
    (7, 8192, 2048, (2048, 1), (8192, 1), "fp16"),
    (1024, 1024, 1024, (1024, 1), (1024, 1), "bf16"),
]


@pytest.mark.parametrize("spec", _INVARIANCE_SPECS)
@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", _INVARIANCE_SHAPES)
def test_mm_invariance(spec, M, N, K, a_strides, b_strides, dtype_name):
    """A non-empty spec gives exact per-dimension equality."""
    run_mm_invariance_case(ARCH, M, N, K, a_strides, b_strides, dtype_name, spec)


def test_full_space_fixed_prunes_lsu_and_separates_cache(monkeypatch):
    """Full space under a fixed schedule excludes LSU and keys the cache on it."""
    from types import SimpleNamespace

    from triton.tlx.ops.kernels.mm import gfx942

    shape = (7, 8192, 2048)
    out = object()
    monkeypatch.setattr(gfx942, "_validate_operands", lambda *args: shape)
    monkeypatch.setattr(gfx942, "_precheck_local_split_u", lambda *args: True)
    monkeypatch.setattr(gfx942, "_full_config_key", lambda a, b: "key")
    monkeypatch.setattr(
        gfx942, "_launch_local_split_u", lambda *args:
        (_ for _ in ()).throw(AssertionError("LSU winner must not be reused under a fixed schedule")))

    lsu_config = SimpleNamespace(kwargs={"USE_LOCAL_SPLIT_U": True}, num_warps=4, num_stages=2)

    class FakeKernel:

        def __init__(self):
            self._tlx_fast_configs = {("key", False, (0, 0), None): (lsu_config, True)}
            self.best_config = lsu_config

        def __getitem__(self, grid):
            raise AssertionError("fixed full-space call must not reuse the unconstrained LSU winner")

    seen = {}

    def tuned(requested_space, shape_arg, enable_local_split_u):
        seen["enable"] = enable_local_split_u
        return FakeKernel()

    monkeypatch.setattr(gfx942, "_tuned", tuned)
    with pytest.raises(AssertionError, match="must not reuse"):
        gfx942._gemm(object(), object(), out=out, space="full", invariance=BitwiseInvarianceSpec.dims("M"))
    assert seen["enable"] is False
