"""L1 correctness for `tlx.ops.flash_attn`.

No compile-time cap as in test_mm.py: this kernel has no shape heuristic, so the
correctness caller autotunes a small space and wall-clock says nothing about
compile time.
"""

import pytest
import torch

from triton._internal_testing import is_blackwell

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.flash_attn is sm100-only today")

torch.manual_seed(0)

ARCH = "sm100"

#  Hardware agnostic testsuite ``[Z, H, N_CTX, HEAD_DIM, causal]``
SHAPES = [
    [1, 1, 256, 64, False],
    [1, 2, 512, 64, True],
    # Both head dims: head dim selects the pipeline.
    [2, 4, 1024, 64, False],
    [2, 4, 1024, 128, False],
    [2, 4, 1024, 128, True],
    [4, 8, 2048, 128, True],
    [1, 16, 4096, 128, False],
    # Head-dominated grid.
    [2, 32, 2048, 64, False],
    [4, 8, 512, 64, True],
    # Long context.
    [1, 1, 8192, 128, True],
]

# atol tracks the magnitude of the result, not of the element compared; see
# test_mm.py.
REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}


def _qkv(Z, H, N_CTX, HEAD_DIM, dtype, requires_grad=False):
    return [
        torch.randn((Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=dtype).requires_grad_(requires_grad) for _ in range(3)
    ]


def _sdpa(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=q.shape[-1]**-0.5)


def _assert_close(out, ref, dtype):
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, causal", SHAPES)
def test_flash_attn_fwd(Z, H, N_CTX, HEAD_DIM, causal, dtype):
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    q, k, v = _qkv(Z, H, N_CTX, HEAD_DIM, dtype)
    out = tlx_flash_attn(q, k, v, causal=causal, arch=ARCH, space="smoke")
    _assert_close(out, _sdpa(q, k, v, causal), dtype)


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, causal", SHAPES)
def test_flash_attn_bwd(Z, H, N_CTX, HEAD_DIM, causal):
    """The kernel carries the full backward path even though the op defers it."""
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    q, k, v = _qkv(Z, H, N_CTX, HEAD_DIM, torch.float16, requires_grad=True)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    do = torch.randn_like(q)

    tlx_flash_attn(q, k, v, causal=causal, arch=ARCH, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal).backward(do)

    for got, want in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        _assert_close(got, want, torch.float16)


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [[2, 4, 1024, 128], [1, 16, 4096, 128], [2, 32, 2048, 64]])
def test_flash_attn_bwd_bf16_accuracy(Z, H, N_CTX, HEAD_DIM):
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    q, k, v = _qkv(Z, H, N_CTX, HEAD_DIM, torch.bfloat16, requires_grad=True)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    do = torch.randn_like(q)

    tlx_flash_attn(q, k, v, causal=False, arch=ARCH, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal=False).backward(do)

    for got, want in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        rel_l2 = torch.linalg.vector_norm(got.float() - want.float()) / torch.linalg.vector_norm(want.float())
        assert rel_l2 < REL_PRECISION[torch.bfloat16]
