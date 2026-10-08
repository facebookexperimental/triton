"""L1 correctness for ``tlx.ops.flash_attn_varlen`` on gfx950. Forward only."""

import itertools

import pytest
import torch
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.flash_attn_varlen._shapes import CORRECTNESS_SHAPES, seqlens

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="tlx.ops.flash_attn_varlen on gfx950 requires CDNA4")

TOLERANCE = {torch.float16: dict(atol=3e-3, rtol=1e-2), torch.bfloat16: dict(atol=2e-2, rtol=2e-2)}


def _cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)


def _reference(q, k, v, cu_q, cu_k, causal, scale):
    """FP32 attention per sequence, bottom-right causal; a row with no key is 0."""
    out = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    group = q.shape[1] // k.shape[1]
    for b in range(cu_q.numel() - 1):
        q0, q1, k0, k1 = cu_q[b].item(), cu_q[b + 1].item(), cu_k[b].item(), cu_k[b + 1].item()
        if q1 == q0:
            continue
        qs = q[q0:q1].float()
        ks = k[k0:k1].float().repeat_interleave(group, 1)
        vs = v[k0:k1].float().repeat_interleave(group, 1)
        s = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        if causal:
            rows = torch.arange(q1 - q0, device=q.device)[:, None]
            cols = torch.arange(k1 - k0, device=q.device)[None, :]
            s = s.masked_fill(cols > rows + (k1 - k0) - (q1 - q0), float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)
        out[q0:q1] = torch.einsum("hqk,khd->qhd", p, vs)
    return out


def _check(lens_q, lens_k, causal, heads=1, kv_heads=None, dtype=torch.float16, sm_scale=None):
    from triton.tlx.ops import flash_attn_varlen

    kv_heads = kv_heads or heads
    q = torch.randn(sum(lens_q), heads, 128, device="cuda", dtype=dtype)
    k = torch.randn(sum(lens_k), kv_heads, 128, device="cuda", dtype=dtype)
    v = torch.randn(sum(lens_k), kv_heads, 128, device="cuda", dtype=dtype)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    out = flash_attn_varlen(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k), sm_scale=sm_scale, causal=causal)
    ref = _reference(q, k, v, cu_q, cu_k, causal, 128**-0.5 if sm_scale is None else sm_scale)
    assert out.dtype == dtype and out.shape == q.shape
    torch.testing.assert_close(out.float(), ref, **TOLERANCE[dtype])


@pytest.mark.parametrize("batch,min_seqlen,max_seqlen,heads,head_dim,causal,dtype_name", CORRECTNESS_SHAPES)
def test_flash_attn_varlen_shapes(batch, min_seqlen, max_seqlen, heads, head_dim, causal, dtype_name):
    torch.manual_seed(20)
    lengths = seqlens(batch, min_seqlen, max_seqlen)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    _check(lengths, lengths, causal, heads=heads, dtype=dtype)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "lens_q,lens_k",
    [
        ([1], [1]),
        # Partial, exact and one-past tiles of the 64-row / 64-key blocks.
        ([5, 64, 65, 127, 129, 300], [5, 64, 65, 127, 129, 300]),
        ([0, 100, 0, 257], [0, 100, 0, 257]),
        # More keys than queries: causal rows see the whole key prefix.
        ([100, 37], [250, 600]),
        # Fewer keys than queries: causal rows before the offset see no key.
        ([300, 600], [120, 50]),
        ([2000, 1500], [2000, 1500]),
    ],
)
def test_flash_attn_varlen_lengths(lens_q, lens_k, causal):
    torch.manual_seed(20)
    _check(lens_q, lens_k, causal)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (4, 2), (8, 1)])
def test_flash_attn_varlen_heads(heads, kv_heads, dtype):
    torch.manual_seed(20)
    _check([300, 700, 64], [300, 700, 64], True, heads=heads, kv_heads=kv_heads, dtype=dtype, sm_scale=0.05)


def test_flash_attn_varlen_split_plan():
    from triton.tlx.ops.kernels.flash_attn_varlen.gfx950 import MAX_PARTIALS, _run_length, _split_plan

    assert _split_plan(1, 16, 256)[0] > 0 and _split_plan(10, 16, 256)[0] > 0
    assert _split_plan(13, 19, 256) == (0, 19, 1)
    for bh in range(1, 40):
        for num_m_blocks in range(1, 40):
            split, items_per_bh, max_splits = _split_plan(bh, num_m_blocks, 256)
            if split:
                splits = [m // split + 1 for m in range(num_m_blocks)]
                assert items_per_bh == sum(splits) and max_splits == max(splits)
                assert _run_length(bh * items_per_bh, max_splits) <= 32
                assert bh * (items_per_bh - num_m_blocks) <= MAX_PARTIALS


@pytest.mark.parametrize(
    "lengths,heads,kv_heads",
    [([1000], 1, 1), ([1000] * 10, 1, 1), ([1200] * 10, 1, 1), ([700, 1200, 950, 801, 1100, 760, 1000], 1, 1),
     ([1000, 640], 4, 2)],
)
def test_flash_attn_varlen_causal_split(lengths, heads, kv_heads):
    """Causal shapes whose query tiles are split by key range and merged."""
    torch.manual_seed(20)
    _check(lengths, lengths, True, heads=heads, kv_heads=kv_heads)


def _split_inputs(lengths, n=1):
    q, k, v = (torch.randn(n, sum(lengths), 1, 128, device="cuda", dtype=torch.float16) for _ in range(3))
    return q, k, v, _cu(lengths)


def test_flash_attn_varlen_split_repeat_and_graph():
    """The per-tile counters return to zero after each launch, eager or replayed."""
    from triton.tlx.ops import flash_attn_varlen

    torch.manual_seed(20)
    lengths = [1000] * 10
    q, k, v, cu = _split_inputs(lengths, 3)
    eager = [flash_attn_varlen(q[i], k[i], v[i], cu, cu, 1000, 1000, causal=True) for i in range(3)]
    for _ in range(3):
        torch.testing.assert_close(flash_attn_varlen(q[0], k[0], v[0], cu, cu, 1000, 1000, causal=True), eager[0],
                                   atol=0, rtol=0)
    sq, sk, sv = q[0].clone(), k[0].clone(), v[0].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flash_attn_varlen(sq, sk, sv, cu, cu, 1000, 1000, causal=True)
    for i in (1, 2, 0, 1):
        sq.copy_(q[i])
        sk.copy_(k[i])
        sv.copy_(v[i])
        graph.replay()
        torch.testing.assert_close(out, eager[i], atol=0, rtol=0)


def test_flash_attn_varlen_split_streams():
    """Launches on two streams at once use separate counters."""
    from triton.tlx.ops import flash_attn_varlen

    torch.manual_seed(20)
    lengths = [1000] * 4
    q, k, v, cu = _split_inputs(lengths, 2)
    expected = [flash_attn_varlen(q[i], k[i], v[i], cu, cu, 1000, 1000, causal=True) for i in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    outs = [[], []]
    for _ in range(20):
        for i, s in enumerate(streams):
            with torch.cuda.stream(s):
                outs[i].append(flash_attn_varlen(q[i], k[i], v[i], cu, cu, 1000, 1000, causal=True))
    torch.cuda.synchronize()
    for i in range(2):
        for out in outs[i]:
            torch.testing.assert_close(out, expected[i], atol=0, rtol=0)


def test_flash_attn_varlen_strided_qkv():
    """Q/K/V as head slices of one packed `(tokens, 3 * heads, 128)` projection."""
    from triton.tlx.ops import flash_attn_varlen

    torch.manual_seed(20)
    lengths = [700, 1000, 1200]
    cu = _cu(lengths)
    qkv = torch.randn(sum(lengths), 6, 128, device="cuda", dtype=torch.float16)
    q, k, v = qkv[:, 0:2], qkv[:, 2:4], qkv[:, 4:6]
    out = flash_attn_varlen(q, k, v, cu, cu, max(lengths), max(lengths), causal=True)
    ref = _reference(q, k, v, cu, cu, True, 128**-0.5)
    torch.testing.assert_close(out.float(), ref, **TOLERANCE[torch.float16])


def test_flash_attn_varlen_rejects_invalid_inputs():
    from triton.tlx.ops import InvalidInput, UnsupportedBackward, flash_attn_varlen

    q = torch.randn(100, 1, 128, device="cuda", dtype=torch.float16)
    cu = _cu([100])
    with pytest.raises(InvalidInput, match="int32"):
        flash_attn_varlen(q, q, q, cu.long(), cu.long(), 100, 100)
    with pytest.raises(InvalidInput):
        flash_attn_varlen(q[..., :64], q[..., :64], q[..., :64], cu, cu, 100, 100)
    with pytest.raises(InvalidInput, match="dtype"):
        flash_attn_varlen(q, q.bfloat16(), q, cu, cu, 100, 100)
    with pytest.raises(InvalidInput, match="rank-3"):
        flash_attn_varlen(q[None], q[None], q[None], cu, cu, 100, 100)
    with pytest.raises(InvalidInput, match="head count"):
        flash_attn_varlen(q.expand(100, 3, 128), q.expand(100, 2, 128), q.expand(100, 2, 128), cu, cu, 100, 100)
    with pytest.raises(UnsupportedBackward):
        flash_attn_varlen(q.requires_grad_(), q, q, cu, cu, 100, 100)
