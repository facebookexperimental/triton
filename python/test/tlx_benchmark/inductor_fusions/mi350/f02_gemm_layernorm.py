from __future__ import annotations

from ..common import CodeContract, FusionCase, make_gemm_norm_inputs

SHAPE = (2032, 2560, 2560)


def gemm_layernorm(x, weight, gemm_bias, scale, norm_bias):
    import torch
    import torch.nn.functional as F

    value = torch.addmm(gemm_bias, x, weight.t())
    return F.layer_norm(value, (value.shape[-1], ), scale, norm_bias, 1.0e-5)


CASE = FusionCase(
    name="gfx950_02_gemm_layernorm",
    problem="M=2032 K=2560 N=2560 dtype=bf16",
    model=gemm_layernorm,
    make_inputs=lambda: make_gemm_norm_inputs(SHAPE, with_norm_bias=True),
    code=CodeContract(
        required_after=("a16w16_8wave", "tlx_gfx950_apply_norm"),
        expected_after_launches=2,
    ),
    requires_custom_op_autotuning=True,
)
