from __future__ import annotations

from ..common import CodeContract, FusionCase

M = 1024
N = 6144


def double_layernorm_silu(x, residual, weight1, bias1, weight2, bias2):
    import torch
    import torch.nn.functional as F

    value = residual + F.layer_norm(x, (N, ), weight1, bias1, 1.0e-5)
    return value * torch.sigmoid(F.layer_norm(value, (N, ), weight2, bias2, 1.0e-5))


def make_inputs():
    import torch

    torch.manual_seed(0)
    return tuple(
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for shape in ((M, N), (M, N), (N, ), (N, ), (N, ), (N, )))


CASE = FusionCase(
    name="gfx950_01_double_layernorm",
    problem="M=1024 N=6144 dtype=fp16",
    model=double_layernorm_silu,
    make_inputs=make_inputs,
    after_mode="allow",
    # Forced compares sibling subkernels directly. Autotuned mirrors the
    # production MultiKernel policy used by the FBSource reference.
    forced_before_config_overrides={"triton.multi_kernel": 2},
    forced_after_config_overrides={"triton.multi_kernel": 3},
    autotuned_before_config_overrides={"triton.multi_kernel": 1},
    autotuned_after_config_overrides={"triton.multi_kernel": 1},
    code=CodeContract(
        required_after=("tlx.local_alloc", "tlx.local_store", "tlx.local_load", "tl.debug_barrier()"),
        forbidden_before=("tlx.local_alloc", "tlx.local_store", "tlx.local_load"),
        forbidden_after=("tl.store(out_ptr1", ),
    ),
    atol=2.0e-2,
    rtol=2.0e-2,
)
