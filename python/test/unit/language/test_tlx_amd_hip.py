"""TLX gfx950 compiler option tests requiring the HIP runtime."""

import re

import pytest
import torch

from triton import knobs
from triton._internal_testing import is_hip
from triton.language.extra.tlx.tutorials import amd_fa_cluster as _amd_fa_cluster_module
from triton.runtime.jit import MockTensor

pytestmark = pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")


@pytest.fixture
def amd_fa_cluster_vector_combine_codegen_gfx950(fresh_triton_cache, gfx950_beta_slp_copying):
    batch, heads, n_ctx, head_dim = 1, 64, 4096, 128
    tensor = MockTensor(torch.bfloat16, (batch, heads, n_ctx, head_dim))
    strides = (heads * n_ctx * head_dim, n_ctx * head_dim, head_dim, 1)
    args = (tensor, tensor, tensor, tensor, *strides, *strides, *strides, *strides, batch)
    common = {
        "H": heads,
        "N_CTX": n_ctx,
        "sm_scale": 1.0 / head_dim**0.5,
        "BLOCK_M": 256,
        "BLOCK_N": 32,
        "BUF_DEPTH": 2,
        "HEAD_DIM": head_dim,
        "USE_DIRECT_LOAD": False,
        "IS_CAUSAL": False,
        "STATIC_STRIDE_KN": head_dim,
        "grid": (n_ctx // 256, heads, batch),
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "enable_sched_group_barrier_scheduler": False,
        "llvm_fn_attrs": "",
    }

    with knobs.runtime.scope():
        knobs.runtime.override_arch = "gfx950"
        baseline = _amd_fa_cluster_module._attn_fwd_cluster_pipeline.warmup(*args, disable_vector_combine=False,
                                                                            **common)
        disabled = _amd_fa_cluster_module._attn_fwd_cluster_pipeline.warmup(*args, disable_vector_combine=True,
                                                                            **common)
    return baseline, disabled



def test_amd_fa_cluster_vector_combine_opt_out_reduces_transfers_gfx950(amd_fa_cluster_vector_combine_codegen_gfx950):
    baseline, disabled = amd_fa_cluster_vector_combine_codegen_gfx950
    baseline_asm = baseline.asm["amdgcn"]
    disabled_asm = disabled.asm["amdgcn"]

    def transfer_count(amdgcn):
        return amdgcn.count("v_accvgpr_read_b32") + amdgcn.count("v_accvgpr_write_b32")

    def instruction_count(amdgcn):
        return len(re.findall(r"^\s+[a-z].*", amdgcn, flags=re.MULTILINE))

    assert transfer_count(baseline_asm) - transfer_count(disabled_asm) >= 32
    assert instruction_count(disabled_asm) < instruction_count(baseline_asm)
    assert baseline.metadata.disable_vector_combine is False
    assert disabled.metadata.disable_vector_combine is True

