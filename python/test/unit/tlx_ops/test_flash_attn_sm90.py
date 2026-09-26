"""L1 correctness for the Hopper ``tlx.ops.flash_attn`` backend."""

import pytest
import torch
from triton._internal_testing import is_hopper
from triton.tlx.ops.kernels.flash_attn._shapes import CORRECTNESS_SHAPES, SYNTHETIC

pytestmark = pytest.mark.skipif(not is_hopper(), reason="requires an sm90 GPU")

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
FWD_SHAPES = tuple(dict.fromkeys((*CORRECTNESS_SHAPES, *(shape._replace(dtype="bf16") for shape in SYNTHETIC))))
BWD_SHAPES = tuple(
    dict.fromkeys((
        *CORRECTNESS_SHAPES,
        *(shape._replace(dtype="bf16") for shape in SYNTHETIC if shape.head_dim == 64 and shape.batch == 1),
    )))


def _qkv(Z, H, N_CTX, HEAD_DIM, dtype, requires_grad=False):
    return [
        torch.randn(
            (Z, H, N_CTX, HEAD_DIM),
            device="cuda",
            dtype=dtype,
        ).requires_grad_(requires_grad) for _ in range(3)
    ]


def _sdpa(q, k, v, causal, scale=None):
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
        scale=scale,
    )


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", FWD_SHAPES)
def test_flash_attn_fwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name):
    from triton.tlx.ops import flash_attn

    dtype = DTYPES[dtype_name]
    torch.manual_seed(0)
    q, k, v = _qkv(Z, H, N_CTX, HEAD_DIM, dtype)
    scale = None if causal else 0.7
    out = flash_attn(
        q,
        k,
        v,
        causal=causal,
        sm_scale=scale,
        space="smoke",
    )
    ref = _sdpa(q, k, v, causal, scale=scale)
    atol = 1e-2 if dtype == torch.float16 else 4e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=0)


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", BWD_SHAPES)
def test_flash_attn_bwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name):
    from triton.tlx.ops import flash_attn

    dtype = DTYPES[dtype_name]
    torch.manual_seed(0)
    q, k, v = _qkv(Z, H, N_CTX, HEAD_DIM, dtype, requires_grad=True)
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    do = torch.randn_like(q)

    flash_attn(q, k, v, causal=causal, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal).backward(do)

    for got, expected in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, expected, atol=0.2, rtol=0.1)


@pytest.mark.parametrize(
    "head_dim,causal,expected_block_n,expected_num_buffers",
    (
        (64, False, 128, 3),
        (64, True, 128, 2),
        (128, False, 128, 2),
        (128, True, 128, 2),
    ),
)
def test_flash_attn_fwd_config(head_dim, causal, expected_block_n, expected_num_buffers):
    from triton.tlx.ops.kernels.flash_attn.sm90 import _prune_configs_by_head_dim, configs

    selected = _prune_configs_by_head_dim(
        configs,
        {},
        HEAD_DIM=head_dim,
        CAUSAL=causal,
        USE_BM192=False,
    )
    assert len(selected) == 1
    assert selected[0].kwargs["BLOCK_N"] == expected_block_n
    assert selected[0].kwargs["NUM_BUFFERS"] == expected_num_buffers


def test_flash_attn_fwd_bm192_config():
    from triton.tlx.ops.kernels.flash_attn.sm90 import _prune_configs_by_head_dim, configs

    selected = _prune_configs_by_head_dim(
        configs,
        {},
        HEAD_DIM=64,
        CAUSAL=False,
        USE_BM192=True,
    )
    assert len(selected) == 1
    assert selected[0].kwargs["BLOCK_M"] == 192
    assert selected[0].kwargs["BLOCK_N"] == 128
    assert selected[0].kwargs["NUM_BUFFERS"] == 3
    assert selected[0].kwargs["NUM_MMA_GROUPS"] == 3


@pytest.mark.parametrize("N_CTX", (1024, 2048, 4096, 8192))
def test_flash_attn_fwd_bm192(N_CTX):
    from triton.tlx.ops import flash_attn

    torch.manual_seed(0)
    q, k, v = _qkv(4, 48, N_CTX, 64, torch.bfloat16)
    out = flash_attn(q, k, v, causal=False, sm_scale=0.7, space="smoke")
    ref = _sdpa(q, k, v, causal=False, scale=0.7)
    torch.testing.assert_close(out, ref, atol=4e-2, rtol=0)


def test_flash_attn_fwd_launch_policy():
    from triton.tlx.ops.kernels.flash_attn.sm90 import _select_forward_policy

    _, steady_unroll, target_workers = _select_forward_policy(True, (4, 48, 1024, 128), torch.bfloat16, 128, 132)
    assert (steady_unroll, target_workers) == (1, 132)

    shape = (4, 48, 2048, 64)
    _, steady_unroll, target_workers = _select_forward_policy(True, shape, torch.bfloat16, 128, 132)
    assert (steady_unroll, target_workers) == (1, 132)

    for n_ctx in (1024, 2048, 4096, 8192):
        _, steady_unroll, target_workers = _select_forward_policy(False, (4, 48, n_ctx, 64), torch.bfloat16, 192, 132)
        assert (steady_unroll, target_workers) == (1, 132)

    _, steady_unroll, target_workers = _select_forward_policy(True, (4, 48, 4096, 64), torch.bfloat16, 128, 132)
    assert (steady_unroll, target_workers) == (2, 129)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_flash_attn_d64_causal_diagonal(dtype):
    from triton.tlx.ops import flash_attn

    torch.manual_seed(0)
    q, k, v = _qkv(1, 1, 128, 64, dtype)
    scale = 0.7
    out = flash_attn(q, k, v, causal=True, sm_scale=scale, space="smoke")
    ref = _sdpa(q, k, v, causal=True, scale=scale)
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=0)


@pytest.mark.parametrize("N_CTX", (64, 192, 320))
def test_flash_attn_rejects_partial_block_m(N_CTX):
    from triton.tlx.ops import flash_attn

    q, k, v = _qkv(1, 1, N_CTX, 64, torch.bfloat16)
    with pytest.raises(AssertionError):
        flash_attn(q, k, v, causal=False, space="smoke")


def test_flash_attn_rejects_d32():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = _qkv(1, 1, 128, 32, torch.float16)
    with pytest.raises(InvalidInput, match="does not support"):
        flash_attn(q, k, v, space="smoke")
