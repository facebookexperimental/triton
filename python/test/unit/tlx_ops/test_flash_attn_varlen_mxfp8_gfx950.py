"""L1 correctness for ``tlx.ops.flash_attn_varlen_mxfp8`` on gfx950."""

import itertools

import pytest
import torch
from triton._internal_testing import is_hip_cdna4

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="tlx.ops.flash_attn_varlen_mxfp8 requires CDNA4")


def _cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)


def _reference(q, k, v, cu_q, cu_k, scale):
    """FP32 attention per sequence: (out, natural-log LSE [heads, tokens]).
    Query rows of a sequence with no keys stay zero with -inf LSE."""
    out = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    lse = torch.full((q.shape[1], q.shape[0]), float("-inf"), device=q.device)
    for b in range(cu_q.numel() - 1):
        q0, q1 = cu_q[b].item(), cu_q[b + 1].item()
        k0, k1 = cu_k[b].item(), cu_k[b + 1].item()
        if q1 == q0 or k1 == k0:
            continue
        qs = q[q0:q1].float()
        ks = k[k0:k1].float()
        vs = v[k0:k1].float()
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        lse[:, q0:q1] = torch.logsumexp(scores, dim=-1)
        out[q0:q1] = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vs)
    return out, lse


@pytest.mark.parametrize(
    "lens_q,lens_k,fast_exp",
    [
        # Two 128-key tiles per sequence, padded at the end only.
        ([96, 128], [160, 192], False),
        # Several query tiles; the last key tile is all padding.
        ([320, 512], [384, 640], False),
        # Sequences averaging at least 4096 keys, with the exact exponential.
        ([320, 512], [4096, 5000], False),
        # Empty sequences, and queries with no keys.
        ([0, 320, 0, 128], [0, 384, 0, 0], False),
        # The approximate exponential on short sequences.
        ([320, 512], [384, 640], True),
        # Padding ahead of each sequence as well as behind it.
        ([320, 512, 320], [100, 300, 4100], True),
        # Queries with no keys, and keys with no queries.
        ([128, 0, 320], [0, 300, 100], False),
        # Maximal padding: the padded sequences fill the K/V buffers exactly.
        ([64, 200], [257, 1], False),
    ],
)
def test_flash_attn_varlen_mxfp8_four_wave(lens_q, lens_k, fast_exp):
    """The public op against FP32 attention over ragged and empty sequences."""
    from triton.tlx.ops import flash_attn_varlen_mxfp8

    torch.manual_seed(20)
    q = torch.randn(sum(lens_q), 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lens_k), 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    out = flash_attn_varlen_mxfp8(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(lens_q),
        max(lens_k),
        causal=False,
        fast_exp=fast_exp,
    )
    ref, _ = _reference(q, k, v, cu_q, cu_k, 128**-0.5)
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.08)


def test_flash_attn_varlen_mxfp8_prequantized():
    """Q quantized ahead of time gives the same bits as Q quantized in the kernel."""
    from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8 import gfx950

    torch.manual_seed(20)
    lens_q, lens_k = [320, 512], [100, 300]
    q = torch.randn(sum(lens_q), 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lens_k), 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    scale = 128**-0.5

    q_state = gfx950.prepare_q(q, cu_q)
    kv_state = gfx950.prepare_kv(k, v, cu_k)
    out_pre, lse_pre = gfx950.forward_prequantized(q_state, kv_state, cu_q, cu_k, scale)
    out, lse, _ = gfx950.forward(q, k, v, cu_q, cu_k, scale)
    ref, ref_lse = _reference(q, k, v, cu_q, cu_k, scale)

    torch.testing.assert_close(out_pre, out, atol=0, rtol=0)
    torch.testing.assert_close(lse_pre, lse, atol=0, rtol=0)
    torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.08)
    torch.testing.assert_close(lse, ref_lse, atol=0.25, rtol=0.05)


@pytest.mark.parametrize("lens_k,fast", [([384, 640], False), ([4096, 5000], True)])
def test_flash_attn_varlen_mxfp8_default_fast_exp(lens_k, fast):
    """Without fast_exp, sequences averaging 4096 keys or more get the approximate exponential."""
    from triton.tlx.ops import flash_attn_varlen_mxfp8
    from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8 import gfx950

    torch.manual_seed(20)
    lens_q = [320, 512]
    q = torch.randn(sum(lens_q), 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lens_k), 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    scale = 128**-0.5

    kv_state = gfx950.prepare_kv(k, v, cu_k)
    out, lse, _ = gfx950.forward(q, k, v, cu_q, cu_k, scale, kv_state=kv_state)
    for explicit in (False, True):
        out_e, lse_e, _ = gfx950.forward(q, k, v, cu_q, cu_k, scale, fast_exp=explicit, kv_state=kv_state)
        assert torch.equal(lse, lse_e) == (explicit == fast)
        assert torch.equal(out, out_e) == (explicit == fast)
    torch.testing.assert_close(flash_attn_varlen_mxfp8(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k)), out, atol=0,
                               rtol=0)
    ref, ref_lse = _reference(q, k, v, cu_q, cu_k, scale)
    torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.08)
    torch.testing.assert_close(lse, ref_lse, atol=0.25, rtol=0.05)


def test_flash_attn_varlen_mxfp8_prepared_state_follows_offsets():
    """Prepared Q and K/V states are reused with the offset tensors they were
    made from, and rejected with any other one, even one holding the same
    values, after an in-place change to theirs, or as plain tuples."""
    from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8 import gfx950

    torch.manual_seed(20)
    scale = 128**-0.5
    lens_q, lens_k = [200, 64, 300], [300, 700, 90]
    q = torch.randn(sum(lens_q), 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lens_k), 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    q_state = gfx950.prepare_q(q, cu_q)
    kv_state = gfx950.prepare_kv(k, v, cu_k)

    out, lse, _ = gfx950.forward(q, k, v, cu_q, cu_k, scale, kv_state=kv_state)
    ref, ref_lse = _reference(q, k, v, cu_q, cu_k, scale)
    torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.08)
    torch.testing.assert_close(lse, ref_lse, atol=0.25, rtol=0.05)
    gfx950.forward_prequantized(q_state, kv_state, cu_q, cu_k, scale)

    with pytest.raises(ValueError, match="other sequence offsets"):
        gfx950.forward(q, k, v, cu_q, cu_k.clone(), scale, kv_state=kv_state)
    with pytest.raises(ValueError, match="other sequence offsets"):
        gfx950.forward_prequantized(q_state, kv_state, cu_q.clone(), cu_k, scale)
    with pytest.raises(ValueError, match="rows"):
        gfx950.forward(q, k[:-1], v[:-1], cu_q, cu_k, scale, kv_state=kv_state)
    with pytest.raises(ValueError, match="returned by prepare_kv"):
        gfx950.forward(q, k, v, cu_q, cu_k, scale, kv_state=tuple(kv_state))

    cu_k.copy_(_cu([700, 90, 300]))
    with pytest.raises(ValueError, match="other sequence offsets"):
        gfx950.forward(q, k, v, cu_q, cu_k, scale, kv_state=kv_state)
    out, lse, _ = gfx950.forward(q, k, v, cu_q, cu_k, scale, kv_state=gfx950.prepare_kv(k, v, cu_k))
    ref, ref_lse = _reference(q, k, v, cu_q, cu_k, scale)
    torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.08)
    torch.testing.assert_close(lse, ref_lse, atol=0.25, rtol=0.05)


@pytest.mark.parametrize("seen_before", [False, True])
def test_flash_attn_varlen_mxfp8_graph_replay_follows_offsets(seen_before):
    """A CUDA graph is captured once, then replayed after the offset tensors
    are overwritten with another batch's offsets: the replay must give that
    batch's result. Run with and without an earlier call on the same offset
    tensors. A call made between the capture and the first replay must be
    correct as well."""
    from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8 import gfx950

    torch.manual_seed(20)
    scale = 128**-0.5
    first = ([64, 300, 200], [700, 90, 300])
    second = ([200, 64, 300], [300, 700, 90])
    q = torch.randn(sum(first[0]), 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(first[1]), 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    def check(out, lse):
        ref, ref_lse = _reference(q, k, v, cu_q, cu_k, scale)
        torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.08)
        torch.testing.assert_close(lse, ref_lse, atol=0.25, rtol=0.05)

    # Compile every kernel on other offset tensors.
    gfx950.forward(q, k, v, _cu(second[0]), _cu(second[1]), scale)
    cu_q, cu_k = _cu(first[0]), _cu(first[1])
    if seen_before:
        gfx950.forward(q, k, v, cu_q, cu_k, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse, _ = gfx950.forward(q, k, v, cu_q, cu_k, scale)
    check(*gfx950.forward(q, k, v, cu_q, cu_k, scale)[:2])
    graph.replay()
    check(out, lse)
    cu_q.copy_(_cu(second[0]))
    cu_k.copy_(_cu(second[1]))
    graph.replay()
    check(out, lse)


@pytest.mark.parametrize("batch", [37, 300])
def test_flash_attn_varlen_mxfp8_dispatch_order(batch):
    """The dispatch tables built on the device. Every query tile of every
    sequence appears exactly once in the queue. The queue's two halves, the
    classes that the XCDs share out, carry about the same work. Each half
    lists its sequences by decreasing number of keys, so the longest work
    items start first."""
    from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8 import gfx950

    generator = torch.Generator().manual_seed(batch)
    lens_q = torch.randint(0, 400, (batch, ), generator=generator).tolist()
    lens_k = torch.randint(0, 3000, (batch, ), generator=generator).tolist()
    lens_q[3], lens_k[5], lens_k[6] = 0, 0, lens_k[7]
    meta = gfx950._build_metadata(_cu(lens_q), _cu(lens_k), sum(lens_q), sum(lens_k), 128, 256)
    start = meta["q_tile_start"].tolist()
    sizes = start[batch:]
    tiles = [-(-q // 128) for q in lens_q]
    assert sum(sizes) == sum(tiles) and max(sizes) <= meta["class_capacity"]

    owner = meta["item_tile"][:sum(sizes)].tolist()
    expected = [None] * sum(sizes)
    for sequence, count in enumerate(tiles):
        for tile in range(count):
            assert expected[start[sequence] + tile] is None
            expected[start[sequence] + tile] = sequence
    assert owner == expected

    cost = [count * k for count, k in zip(tiles, lens_k)]
    loads = []
    for lo, hi in ((0, sizes[0]), (sizes[0], sum(sizes))):
        sequences = list(dict.fromkeys(owner[lo:hi]))
        keys = [lens_k[sequence] for sequence in sequences]
        assert keys == sorted(keys, reverse=True)
        loads.append(sum(cost[sequence] for sequence in sequences))
    assert abs(loads[0] - loads[1]) <= max(cost)


def test_flash_attn_varlen_mxfp8_rejects_invalid_inputs():
    """Mismatched K/V shapes and empty offset tensors raise InvalidInput, and
    the kernel entry point rejects offsets of another dtype or shape."""
    from triton.tlx.ops import InvalidInput, flash_attn_varlen_mxfp8

    q = torch.randn(8, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(16, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(15, 4, 128, device="cuda", dtype=torch.bfloat16)
    cu_q = _cu([8])
    cu_k = _cu([16])
    with pytest.raises(InvalidInput, match="shape"):
        flash_attn_varlen_mxfp8(q, k, v, cu_q, cu_k, 8, 16)
    empty_offsets = torch.empty(0, device="cuda", dtype=torch.int32)
    with pytest.raises(InvalidInput, match="sequence offsets"):
        flash_attn_varlen_mxfp8(q, k, k, empty_offsets, empty_offsets, 8, 16)

    from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8 import gfx950

    with pytest.raises(ValueError, match="sequence offsets"):
        gfx950.forward(q, k, k, cu_q.long(), cu_k, 128**-0.5)
    with pytest.raises(ValueError, match="same shape"):
        gfx950.forward(q, k, k, cu_q, _cu([8, 8]), 128**-0.5)
