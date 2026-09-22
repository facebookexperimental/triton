from __future__ import annotations

from ..common import CodeContract, FusionCase, make_gemm_norm_inputs

SHAPE = (677, 8192, 4096)


def gemm_rmsnorm(x, weight, gemm_bias, scale):
    import torch
    import torch.nn.functional as F

    value = torch.addmm(gemm_bias, x, weight.t())
    return F.rms_norm(value, (value.shape[-1], ), scale, 1.0e-5)


CASE = FusionCase(
    name="gfx950_03_gemm_rmsnorm",
    problem="M=677 K=8192 N=4096 dtype=bf16",
    model=gemm_rmsnorm,
    make_inputs=lambda: make_gemm_norm_inputs(SHAPE, with_norm_bias=False),
    code=CodeContract(
        required_after=(
            "a16w16_8wave",
            "tlx_gfx950_addmm_rmsnorm_stats",
            "tlx_gfx950_apply_norm",
        ),
        expected_after_launches=3,
    ),
    requires_custom_op_autotuning=True,
)
