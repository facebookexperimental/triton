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


@pytest.mark.parametrize("space, expected_shape", [("heuristic", (819200, 1024, 192)), ("full", None)])
def test_space_reaches_requested_autotuner(monkeypatch, space, expected_shape):
    from triton.tlx.ops.kernels.mm import gfx942

    monkeypatch.setattr(gfx942, "_validate_operands", lambda *args: (819200, 1024, 192))

    class AutotunerReached(Exception):
        pass

    def tuned(requested_space, shape):
        assert requested_space == space
        assert shape == expected_shape
        raise AutotunerReached

    monkeypatch.setattr(gfx942, "_tuned", tuned)
    with pytest.raises(AutotunerReached):
        gfx942._gemm(object(), object(), out=object(), space=space)


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


@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", shapes())
def test_mm(M, N, K, a_strides, b_strides, dtype_name):
    run_mm_case(ARCH, M, N, K, a_strides, b_strides, dtype_name)
