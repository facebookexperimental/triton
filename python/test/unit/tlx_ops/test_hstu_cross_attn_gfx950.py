"""gfx950 ``tlx.ops.hstu_cross_attn`` forward and backward correctness.

KV is the history sequence and Q holds only the targets, with independent
jagged offsets, checked against a dense PyTorch reference.
"""
import pytest
import torch
from triton._internal_testing import is_hip_cdna4

# The backward reduces dK/dV over q tiles and dQ through memory across KV
# blocks, in an order different from the reference's autograd graph, so the
# error grows with the block count rather than one rounding.
BWD_ATOL = 2e-3
BWD_RTOL = 2e-2
FWD_RTOL = 1.6e-2

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")


def _fwd_atol(num_softmax_heads):
    # Softmax rows land about one bf16 ULP from the reference on near-zero
    # outputs; silu matches tighter.
    return 8e-4 if num_softmax_heads > 0 else 2e-5


def _offsets(lengths):
    offsets = torch.zeros((lengths.numel() + 1, ), dtype=torch.int64, device=lengths.device)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    return offsets


def _run(lengths_kv, lengths_q, heads, num_softmax_heads, dtype, *, heads_kv=None, dim=128, causal=False,
         has_num_targets=True, shared_kv=False, check_backward=False, bwd_variant=None):
    from triton.tlx.ops import hstu_cross_attn
    from triton.tlx.ops.kernels.hstu_cross_attn import gfx950
    from triton.tlx.ops.kernels.hstu_cross_attn._reference import pytorch_hstu_cross_mha

    device = torch.device("cuda")
    lengths_kv = torch.tensor(lengths_kv, dtype=torch.int64, device=device)
    num_targets = torch.tensor(lengths_q, dtype=torch.int64, device=device)
    max_kv_len = int(lengths_kv.max().item())
    max_q_len = int(num_targets.max().item())
    seq_offsets = _offsets(lengths_kv)
    seq_offsets_q = _offsets(num_targets)
    heads_kv = heads if heads_kv is None else heads_kv
    alpha = dim**-0.5
    attn_scale = torch.tensor(1.0 / (max_kv_len + max_q_len), device=device)
    kernel_num_targets = num_targets if has_num_targets else None

    def rand(total, h):
        return torch.empty((total, h, dim), dtype=dtype, device=device).uniform_(-0.1, 0.1)

    q = rand(int(seq_offsets_q[-1]), heads)
    k = rand(int(seq_offsets[-1]), heads_kv)
    v = k if shared_kv else rand(int(seq_offsets[-1]), heads_kv)
    if check_backward:
        ref_q, ref_k = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
        ref_v = (k if shared_kv else v).detach().clone().requires_grad_()
        q, k = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
        v = k if shared_kv else v.detach().clone().requires_grad_()
    else:
        ref_q, ref_k, ref_v = q, k, v

    ref = pytorch_hstu_cross_mha(max_q_len, max_kv_len, alpha, ref_q, ref_k, ref_v, attn_scale, seq_offsets_q,
                                 seq_offsets, num_softmax_heads=num_softmax_heads, num_targets=kernel_num_targets,
                                 causal=causal)
    if bwd_variant is None:
        out = hstu_cross_attn(q, k, v, seq_offsets, max_kv_len, attn_scale, alpha=alpha, seq_offsets_q=seq_offsets_q,
                              max_q_len=max_q_len, num_softmax_heads=num_softmax_heads, num_targets=kernel_num_targets,
                              causal=causal, shared_kv=shared_kv, space="smoke")
    else:
        # The backward variant is not part of the public signature.
        gfx950._SPACE = "smoke"
        out = gfx950.tlx_gfx950_cross_attn_mha_wrapper(max_seq_len=max_kv_len, alpha=alpha, q=q, k=k, v=v,
                                                       seq_offsets=seq_offsets, attn_scale=attn_scale,
                                                       max_q_len=max_q_len, seq_offsets_q=seq_offsets_q,
                                                       num_softmax_heads=num_softmax_heads,
                                                       num_targets=kernel_num_targets, causal=causal,
                                                       shared_kv=shared_kv, bwd_variant=bwd_variant)
    torch.testing.assert_close(out, ref, atol=_fwd_atol(num_softmax_heads), rtol=FWD_RTOL)
    if not check_backward:
        return

    dout = torch.randn_like(out)
    if shared_kv:
        ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref, (ref_q, ref_k, ref_v), dout)
        expected = {"dq": ref_dq, "dk_shared": ref_dk + ref_dv}
        actual = dict(zip(("dq", "dk_shared"), torch.autograd.grad(out, (q, k), dout)))
    else:
        expected = dict(zip(("dq", "dk", "dv"), torch.autograd.grad(ref, (ref_q, ref_k, ref_v), dout)))
        actual = dict(zip(("dq", "dk", "dv"), torch.autograd.grad(out, (q, k, v), dout)))
    for name, grad in actual.items():
        torch.testing.assert_close(grad, expected[name], atol=BWD_ATOL, rtol=BWD_RTOL,
                                   msg=lambda formatted, name=name: f"{name}: {formatted}")


@requires_gfx950
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_hstu_cross_attn_pipeline_boundaries(dtype, causal):
    # KV lengths straddle every 32/64-row block boundary, including empty KV.
    torch.manual_seed(1007)
    _run([0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193],
         [7, 1, 31, 32, 33, 63, 64, 65, 7, 31, 32, 33, 64, 65], heads=1, num_softmax_heads=0, dtype=dtype,
         causal=causal, has_num_targets=not causal, check_backward=not causal)


@requires_gfx950
@pytest.mark.parametrize("num_softmax_heads", [0, 1, 2])
def test_hstu_cross_attn_heads(num_softmax_heads):
    torch.manual_seed(1001)
    _run([256, 100, 64, 0], [128, 31, 64, 5], heads=2, num_softmax_heads=num_softmax_heads, dtype=torch.bfloat16,
         check_backward=True)


@requires_gfx950
@pytest.mark.parametrize("num_softmax_heads", [0, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_hstu_cross_attn_shared_kv(num_softmax_heads, causal):
    # Shared K/V reads one staged LDS tile through two layouts (QK^T and PV).
    torch.manual_seed(1001)
    _run([256, 100, 64, 0], [128, 31, 64, 5], heads=2, num_softmax_heads=num_softmax_heads, dtype=torch.bfloat16,
         causal=causal, shared_kv=True, check_backward=not causal)


@requires_gfx950
@pytest.mark.parametrize("heads_kv", [1, 2])
def test_hstu_cross_attn_gqa(heads_kv):
    torch.manual_seed(1001)
    _run([256, 200, 64, 17], [128, 64, 31, 5], heads=4, heads_kv=heads_kv, num_softmax_heads=2, dtype=torch.bfloat16)


@requires_gfx950
@pytest.mark.parametrize("splits", [2, 3, 8])
@pytest.mark.parametrize("num_softmax_heads", [0, 1, 2])
def test_hstu_cross_attn_split_kv(splits, num_softmax_heads, monkeypatch):
    # The split heuristic only fires on large shapes; force the count. The KV
    # lengths give empty chunks, partial last blocks and several full blocks
    # per chunk, and the backward reads the merged M.
    from triton.tlx.ops.kernels.hstu_cross_attn import gfx950

    monkeypatch.setenv(gfx950.KV_SPLITS_ENV, str(splits))
    torch.manual_seed(1009)
    _run([0, 5, 64, 100, 257, 640, 1000], [3, 17, 64, 65, 128, 129, 200], heads=2, num_softmax_heads=num_softmax_heads,
         dtype=torch.bfloat16, check_backward=True)


@requires_gfx950
def test_hstu_cross_attn_bwd_split_softmax():
    torch.manual_seed(1001)
    _run([2048, 1792, 64, 0], [256, 250, 17, 33], heads=1, num_softmax_heads=1, dtype=torch.bfloat16, shared_kv=True,
         check_backward=True, bwd_variant="split_softmax")


@requires_gfx950
def test_hstu_cross_attn_rejects_invalid_inputs():
    from triton.tlx.ops import InvalidInput, hstu_cross_attn

    device = torch.device("cuda")
    offsets = torch.tensor([0, 8], dtype=torch.int64, device=device)
    scale = torch.tensor(0.1, device=device)
    q = torch.zeros((8, 1, 48), dtype=torch.float16, device=device)
    with pytest.raises(InvalidInput):
        hstu_cross_attn(q, q, q, offsets, 8, scale)
    q = torch.zeros((8, 1, 64), dtype=torch.float32, device=device)
    with pytest.raises(InvalidInput):
        hstu_cross_attn(q, q, q, offsets, 8, scale)
    q = torch.zeros((8, 1, 64), dtype=torch.float16, device=device)
    with pytest.raises(InvalidInput):
        hstu_cross_attn(q, q, q, offsets, 8, scale, seq_offsets_q=offsets)
    with pytest.raises(ValueError):
        hstu_cross_attn(q, q, q, offsets, 8, scale, space="heuristic")
