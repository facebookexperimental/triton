"""Compatibility entry points for the promoted gfx950 backward kernels.

The implementation now lives in
``triton.tlx.ops.kernels.flash_attn.gfx950_bwd``.  New callers should import
from that module; these names remain for existing tutorial users.
"""

import dataclasses

import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.flash_attn.gfx950_bwd import (
    GQA_BENCHMARK_SHAPES,
    SUPPORTED_SHAPES,
    fa_backward,
    fa_backward_input_support_error,
    fa_backward_support_error,
)


@dataclasses.dataclass(frozen=True)
class ReferenceCase:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    do: torch.Tensor
    lse: torch.Tensor
    sm_scale: float
    causal: bool
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def kernel_args(self):
        return (self.q, self.k, self.v, self.o, self.do, self.lse, self.sm_scale, self.causal)


def make_reference_case(shape, causal, seed=0):
    """Build forward state and FP32 autograd gradients one head at a time."""
    batch, heads, n_ctx, head_dim = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    dq = torch.empty(shape, device="cuda", dtype=torch.float32)
    dk = torch.empty_like(dq)
    dv = torch.empty_like(dq)
    sm_scale = head_dim**-0.5
    causal_mask = None
    if causal:
        causal_mask = torch.ones((n_ctx, n_ctx), device="cuda", dtype=torch.bool).triu(1)

    for batch_idx in range(batch):
        for head_idx in range(heads):
            q_ref = q[batch_idx, head_idx].float().requires_grad_(True)
            k_ref = k[batch_idx, head_idx].float().requires_grad_(True)
            v_ref = v[batch_idx, head_idx].float().requires_grad_(True)
            scores = torch.matmul(q_ref, k_ref.transpose(0, 1)) * sm_scale
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            lse_ref = torch.logsumexp(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)
            o_ref = torch.matmul(probs, v_ref)
            grads = torch.autograd.grad(
                o_ref,
                (q_ref, k_ref, v_ref),
                do[batch_idx, head_idx].float(),
            )
            with torch.no_grad():
                o[batch_idx, head_idx].copy_(o_ref)
                lse[batch_idx, head_idx].copy_(lse_ref)
                dq[batch_idx, head_idx].copy_(grads[0])
                dk[batch_idx, head_idx].copy_(grads[1])
                dv[batch_idx, head_idx].copy_(grads[2])

    return ReferenceCase(q, k, v, o, do, lse, sm_scale, causal, (dq, dk, dv))


__all__ = [
    "GQA_BENCHMARK_SHAPES",
    "ReferenceCase",
    "SUPPORTED_SHAPES",
    "fa_backward",
    "fa_backward_input_support_error",
    "fa_backward_support_error",
    "is_hip_cdna4",
    "make_reference_case",
]
