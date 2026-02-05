import os

import pytest

import torch

import triton

from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import (
    matmul as _tlx_matmul_ws, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_clc import (
    matmul as _tlx_matmul_clc, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_pipelined import (
    matmul as _tlx_matmul_pipelined, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_2cta import (
    matmul as _tlx_matmul_2cta, )

from triton._internal_testing import is_blackwell

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Default configs for correctness testing (single config per kernel)
DEFAULT_CONFIGS = {
    "ws": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "NUM_SMEM_BUFFERS": 2,
        "NUM_TMEM_BUFFERS": 2,
        "EPILOGUE_SUBTILE": 1,
        "PAIR_CTA": False,
    }, "clc": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "NUM_SMEM_BUFFERS": 2,
        "NUM_TMEM_BUFFERS": 2,
        "EPILOGUE_SUBTILE": True,
    }, "pipelined": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "NUM_STAGES": 4,
    }, "2cta": None,  # Uses fixed config internally
}

# Registry of available matmul implementations
MATMUL_METHODS = {
    "ws": _tlx_matmul_ws,
    "clc": _tlx_matmul_clc,
    "pipelined": _tlx_matmul_pipelined,
    "2cta": _tlx_matmul_2cta,
}


def get_gemm_types():
    version = os.environ.get("TLX_GEMM_VERSION")
    if version:
        if version not in MATMUL_METHODS:
            raise ValueError(f"Invalid TLX_GEMM_VERSION: {version}. Valid options: {list(MATMUL_METHODS.keys())}")
        return [version]
    return list(MATMUL_METHODS.keys())


@pytest.mark.skipif(
    not is_blackwell(),
    reason="Requires Blackwell GPU",
)
@pytest.mark.parametrize("gemm_type", get_gemm_types())
@pytest.mark.parametrize("shape", ((4096, 4096, 4096), (8192, 8192, 8192)))
def test_blackwell_gemm(gemm_type, shape):
    matmul = MATMUL_METHODS[gemm_type]
    config = DEFAULT_CONFIGS[gemm_type]
    M, N, K = shape

    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    torch_output = torch.matmul(a, b)
    triton_output = matmul(a, b, config=config)

    torch.testing.assert_close(triton_output, torch_output)


# TODO. test_blackwell_attention, test_hopper_gemm, test_amd_gemm, test_hopper_attention, etc.
