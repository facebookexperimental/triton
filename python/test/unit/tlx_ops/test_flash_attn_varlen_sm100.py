"""L1 correctness for ``tlx.ops.flash_attn_varlen`` on Blackwell: forward and backward, causal and windowed."""

import itertools

import pytest
import torch
from triton._internal_testing import is_blackwell

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.flash_attn_varlen on sm100 requires Blackwell")

FWD_TOL = dict(atol=2e-2, rtol=2e-2)
BWD_TOL = dict(atol=5e-2, rtol=5e-2)


def _cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)


def _reference(q, k, v, cu_q, cu_k, causal, window, scale):
    """FP32 attention per sequence with flash-attn's bottom-right band; a row with no key is 0."""
    left, right = window
    if causal:
        right = 0
    group = q.shape[1] // k.shape[1]
    outs = []
    for b in range(cu_q.numel() - 1):
        q0, q1, k0, k1 = cu_q[b].item(), cu_q[b + 1].item(), cu_k[b].item(), cu_k[b + 1].item()
        qs = q[q0:q1].float()
        ks = k[k0:k1].float().repeat_interleave(group, 1)
        vs = v[k0:k1].float().repeat_interleave(group, 1)
        s = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        rel = (torch.arange(k1 - k0, device=q.device)[None, :] - torch.arange(q1 - q0, device=q.device)[:, None] -
               ((k1 - k0) - (q1 - q0)))
        keep = torch.ones_like(rel, dtype=torch.bool)
        if left >= 0:
            keep &= rel >= -left
        if right >= 0:
            keep &= rel <= right
        s = s.masked_fill(~keep, float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)
        outs.append(torch.einsum("hqk,khd->qhd", p, vs))
    return torch.cat(outs) if outs else q.float() * 0


def _check(lens_q, lens_k, causal=False, window=(-1, -1), heads=2, kv_heads=None, head_dim=128, dtype=torch.bfloat16,
           backward=True):
    from triton.tlx.ops import flash_attn_varlen

    torch.manual_seed(20)
    kv_heads = kv_heads or heads
    q = torch.randn(sum(lens_q), heads, head_dim, device="cuda", dtype=dtype, requires_grad=backward)
    k = torch.randn(sum(lens_k), kv_heads, head_dim, device="cuda", dtype=dtype, requires_grad=backward)
    v = torch.randn(sum(lens_k), kv_heads, head_dim, device="cuda", dtype=dtype, requires_grad=backward)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    scale = head_dim**-0.5
    out = flash_attn_varlen(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k), causal=causal, window_size=window)
    ref = _reference(q, k, v, cu_q, cu_k, causal, window, scale)
    assert out.dtype == dtype and out.shape == q.shape
    torch.testing.assert_close(out.float(), ref, **FWD_TOL)
    if not backward:
        return
    do = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), do)
    ref_grads = torch.autograd.grad(ref, (q, k, v), do.float())
    for name, got, want in zip("qkv", grads, ref_grads):
        torch.testing.assert_close(got.float(), want.float(), **BWD_TOL, msg=lambda m, name=name: f"d{name}: {m}")


LENGTHS = [
    [1],
    # Partial, exact and one-past tiles of the 256-row forward / 128-row backward blocks.
    [5, 64, 65, 127, 128, 129, 255, 256, 257, 300],
    [700, 129, 1, 1000],
]
WINDOWS = [(-1, -1), (256, 0), (100, 100), (0, 0), (300, -1)]


@pytest.mark.parametrize("lens", LENGTHS)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", WINDOWS)
def test_self_attention(lens, causal, window):
    _check(lens, lens, causal=causal, window=window)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [(-1, -1), (64, 0)])
@pytest.mark.parametrize(
    "lens_q,lens_k",
    [
        # More keys than queries: causal rows see the whole key prefix.
        ([100, 37], [250, 600]),
        # Fewer keys than queries: causal rows before the offset see no key and return 0.
        ([300, 600], [120, 50]),
        # Empty sequences are skipped without disturbing their neighbours.
        ([0, 100, 0, 257], [0, 100, 0, 257]),
    ],
)
def test_ragged(lens_q, lens_k, causal, window):
    _check(lens_q, lens_k, causal=causal, window=window)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [(-1, -1), (64, 0)])
def test_short_queries(causal, window):
    """A few dozen queries per sequence over a long key sequence: candidate tokens over a user history."""
    lens_q = torch.randint(1, 129, (24, ), generator=torch.Generator().manual_seed(1)).tolist()
    lens_k = torch.randint(1, 2000, (24, ), generator=torch.Generator().manual_seed(2)).tolist()
    _check(lens_q, lens_k, causal=causal, window=window)


@pytest.mark.parametrize("causal, window", [(False, (-1, -1)), (False, (100, 100)), (True, (256, 0))])
def test_gqa(causal, window):
    _check([300, 129, 513], [300, 129, 513], causal=causal, window=window, heads=8, kv_heads=2)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtypes_and_head_dims(head_dim, dtype):
    _check([200, 700], [200, 700], causal=True, window=(256, 0), head_dim=head_dim, dtype=dtype)


def test_large_batch_keeps_index_order():
    """Beyond 2,048 sequences the forward skips longest-first ordering."""
    lens = torch.randint(1, 65, (2100, ), generator=torch.Generator().manual_seed(3)).tolist()
    _check(lens, lens, causal=True, heads=1, backward=False)


def test_encoder_shape():
    """The OneFlow ranker's encoder self-attention: causal with a 256-key window over many short sequences."""
    lens = torch.randint(1, 1500, (64, ), generator=torch.Generator().manual_seed(0)).tolist()
    _check(lens, lens, causal=True, window=(256, 0), heads=4)


def test_rejects_invalid_window():
    from triton.tlx.ops import InvalidInput, flash_attn_varlen

    q = torch.randn(10, 2, 128, device="cuda", dtype=torch.bfloat16)
    cu = _cu([10])
    with pytest.raises(InvalidInput):
        flash_attn_varlen(q, q, q, cu, cu, 10, 10, window_size=(-2, 0))
