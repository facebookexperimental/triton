"""Correctness for the gfx950 ``tlx.ops`` paged-attention prefill operator."""

import inspect

import pytest
import torch
import triton
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import InvalidInput, UnsupportedBackward, paged_attention_prefill
from triton.tlx.ops.kernels.paged_attn import gfx950_prefill as pf

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="gfx950 paged prefill requires CDNA4")

DEVICE = triton.runtime.driver.active.get_active_torch_device()

SHAPES = {
    "one_row": ([1], [1]),
    "fresh": ([77], [77]),
    "prefix": ([200], [1000]),
    "ragged": ([5, 130, 0, 64], [5, 900, 17, 64]),
    "long_prefix": ([300, 1], [333, 4000]),
    "multi_tile": ([1000], [5000]),
}


def _inputs(q_lens, ctx_lens, q_heads, kv_heads, page_size, dtype=torch.bfloat16, head_dim=64, kv_dtype=None,
            k_scale=1.0, v_scale=1.0):
    """Inputs whose unreferenced pages and page tails past each context hold NaN, so any stray read shows. With
    ``kv_dtype=torch.float8_e4m3fn`` K and V are stored as value / ``k_scale`` and value / ``v_scale``."""
    inputs = pf.build_inputs(q_lens, ctx_lens, q_heads, kv_heads, head_dim, page_size, dtype=dtype, device=DEVICE,
                             extra_pages=3)
    query, key_cache, value_cache, cu, ctx, block_tables = inputs
    live = torch.zeros(key_cache.shape[0], key_cache.shape[1], dtype=torch.bool, device=DEVICE)
    for b, c in enumerate(ctx_lens):
        pages = block_tables[b, :triton.cdiv(c, page_size)].long()
        slots = torch.arange(pages.numel() * page_size, device=DEVICE) < c
        live[pages] = slots.view(-1, page_size)
    stale = ~live[:, :, None, None].expand_as(key_cache)
    key_cache.masked_fill_(stale, float("nan"))
    value_cache.masked_fill_(stale, float("nan"))
    if kv_dtype is not None:
        key_cache = (key_cache.float() / k_scale).to(kv_dtype)
        value_cache = (value_cache.float() / v_scale).to(kv_dtype)
        return query, key_cache, value_cache, cu, ctx, block_tables
    return inputs


def _descales(k_scale, v_scale):
    return tuple(torch.tensor([s], dtype=torch.float32, device=DEVICE) for s in (k_scale, v_scale))


def _prefill(inputs, max_seqlen_q, window=0, softmax_scale=None, out=None, max_seqlen_k=None, **overrides):
    """Call the op exactly as SGLang's aiter backend calls ``unified_attention``."""
    query, key_cache, value_cache, cu, ctx, block_tables = inputs
    kwargs = dict(q=query, k=key_cache, v=value_cache, out=torch.empty_like(query) if out is None else out,
                  cu_seqlens_q=cu, seqused_k=ctx, max_seqlen_q=max_seqlen_q,
                  max_seqlen_k=block_tables.shape[1] * key_cache.shape[1] if max_seqlen_k is None else max_seqlen_k,
                  softmax_scale=query.shape[2]**-0.5 if softmax_scale is None else softmax_scale, causal=True,
                  window_size=(window - 1, 0) if window > 0 else (-1, -1), block_table=block_tables, softcap=0.0,
                  q_descale=None, k_descale=None, v_descale=None, sinks=None)
    kwargs.update(overrides)
    return paged_attention_prefill(**kwargs)


def _assert_matches(out, expected):
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=2e-2)
    rel = (out.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 1e-2, f"relative L2 error {rel.item():.3e}"


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("window", [0, 1, 37, 2048], ids=lambda w: f"w{w}")
@pytest.mark.parametrize("page_size", [16, 64], ids=lambda p: f"p{p}")
@pytest.mark.parametrize("heads", [(64, 8), (32, 4), (16, 16)], ids=lambda h: f"q{h[0]}kv{h[1]}")
def test_paged_attention_prefill(heads, page_size, window, shape):
    q_lens, ctx_lens = SHAPES[shape]
    inputs = _inputs(q_lens, ctx_lens, *heads, page_size)
    out = _prefill(inputs, max(q_lens), window)
    _assert_matches(out, pf.ref_prefill(*inputs, sliding_window=window))


@pytest.mark.parametrize("window", [1, 37, 2048], ids=lambda w: f"w{w}")
@pytest.mark.parametrize("page_size", [16, 64], ids=lambda p: f"p{p}")
def test_keys_before_window_are_never_read(page_size, window):
    """An SWA pool recycles pages once every query has slid past them: their block-table entries point at
    unrelated (here NaN) pages, and slots before the window in the boundary page are stale too."""
    q_lens, ctx_lens = [1000, 300, 5, 64], [5000, 2400, 3000, 64]
    inputs = _inputs(q_lens, ctx_lens, 64, 8, page_size)
    expected = pf.ref_prefill(*inputs, sliding_window=window)
    query, key_cache, value_cache, cu, ctx, block_tables = inputs
    live = torch.cat([block_tables[b, :triton.cdiv(c, page_size)] for b, c in enumerate(ctx_lens)])
    poison = next(p for p in range(key_cache.shape[0]) if not (live == p).any())
    key_cache, value_cache, block_tables = key_cache.clone(), value_cache.clone(), block_tables.clone()
    key_cache[poison] = float("nan")
    value_cache[poison] = float("nan")
    for b, (q_len, c) in enumerate(zip(q_lens, ctx_lens)):
        first_key = max(c - q_len - window + 1, 0)
        block_tables[b, :first_key // page_size] = poison
        if first_key % page_size:
            page = block_tables[b, first_key // page_size].long()
            key_cache[page, :first_key % page_size] = float("nan")
            value_cache[page, :first_key % page_size] = float("nan")
    out = _prefill((query, key_cache, value_cache, cu, ctx, block_tables), max(q_lens), window)
    _assert_matches(out, expected)


@pytest.mark.parametrize("max_seqlen_k", [300, 4000], ids=["seq_interleaved", "seq_major"])
def test_tile_orders_match(max_seqlen_k):
    """Both tile orders compute the same rows: max_seqlen_k only changes the dispatch order."""
    q_lens, ctx_lens = [300, 1, 64], [333, 4000, 64]
    inputs = _inputs(q_lens, ctx_lens, 64, 8, 16)
    out = _prefill(inputs, max(q_lens), max_seqlen_k=max_seqlen_k)
    _assert_matches(out, pf.ref_prefill(*inputs))


def test_float16_int64_seqused_k_and_scale():
    q_lens, ctx_lens = [130, 7], [900, 7]
    inputs = _inputs(q_lens, ctx_lens, 64, 8, 16, dtype=torch.float16)
    out = torch.full_like(inputs[0], float("nan"))
    result = _prefill(inputs, max(q_lens), 256, softmax_scale=0.05, out=out, seqused_k=inputs[4].long())
    assert result is out
    _assert_matches(out, pf.ref_prefill(*inputs, sm_scale=0.05, sliding_window=256))


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("window", [0, 37, 2048], ids=lambda w: f"w{w}")
@pytest.mark.parametrize("heads", [(64, 8), (16, 16)], ids=lambda h: f"q{h[0]}kv{h[1]}")
def test_fp8_kv_cache(heads, window, shape):
    """FP8 e4m3fn K/V with per-tensor descales, as SGLang's FP8 KV pool passes them; K and V get different scales
    so a scale applied to the wrong operand shows."""
    q_lens, ctx_lens = SHAPES[shape]
    inputs = _inputs(q_lens, ctx_lens, *heads, 16, kv_dtype=torch.float8_e4m3fn, k_scale=0.05, v_scale=0.2)
    k_descale, v_descale = _descales(0.05, 0.2)
    out = _prefill(inputs, max(q_lens), window, k_descale=k_descale, v_descale=v_descale)
    _assert_matches(out, pf.ref_prefill(*inputs, sliding_window=window, k_descale=k_descale, v_descale=v_descale))


def test_fp8_kv_cache_float16_query_page64():
    q_lens, ctx_lens = [130, 7], [900, 7]
    inputs = _inputs(q_lens, ctx_lens, 64, 8, 64, dtype=torch.float16, kv_dtype=torch.float8_e4m3fn, k_scale=1.0,
                     v_scale=1.0)
    k_descale, v_descale = _descales(1.0, 1.0)
    out = _prefill(inputs, max(q_lens), 256, softmax_scale=0.05, k_descale=k_descale, v_descale=v_descale)
    _assert_matches(
        out, pf.ref_prefill(*inputs, sm_scale=0.05, sliding_window=256, k_descale=k_descale, v_descale=v_descale))


@pytest.mark.parametrize("head_dim", [64, 128])
def test_triton_reference_kernel(head_dim):
    q_lens, ctx_lens = [5, 130, 64], [5, 900, 64]
    inputs = _inputs(q_lens, ctx_lens, 64, 8, 16, head_dim=head_dim)
    out = pf.paged_attention_prefill_triton(*inputs, max(q_lens), sliding_window=37)
    _assert_matches(out, pf.ref_prefill(*inputs, sliding_window=37))


def test_signature_matches_unified_attention():
    aiter_attention = pytest.importorskip("aiter.ops.triton.attention.unified_attention")
    ours = inspect.signature(paged_attention_prefill).parameters
    theirs = inspect.signature(aiter_attention.unified_attention).parameters
    assert [(p.name, p.default) for p in theirs.values()] == [(p.name, p.default) for p in ours.values()][:-1]
    assert list(ours)[-1] == "space"


def test_rejects_unsupported_shapes():
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        _prefill(_inputs([8], [8], 64, 8, 16, head_dim=128), 8)
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        _prefill(_inputs([8], [8], 48, 8, 16), 8)


@pytest.mark.parametrize("override", [
    dict(causal=False),
    dict(softcap=30.0),
    dict(sinks=torch.zeros(64)),
    dict(q_descale=torch.ones(1)),
    dict(alibi_slopes=torch.zeros(64)),
    dict(shuffled_kv_cache=True),
    dict(skip_reduce=True),
], ids=lambda o: next(iter(o)))
def test_rejects_unsupported_features(override):
    with pytest.raises(InvalidInput, match="does not support"):
        _prefill(_inputs([8], [8], 64, 8, 16), 8, **override)


def test_rejects_invalid_inputs():
    inputs = _inputs([8], [8], 64, 8, 16)
    query, key_cache, value_cache, cu, ctx, block_tables = inputs
    bad = [
        dict(k=key_cache[..., None], v=value_cache[..., None]),
        dict(cu_seqlens_q=cu.long()),
        dict(cu_seqlens_q=cu[:1]),
        dict(q=query[:, :60], out=torch.empty_like(query[:, :60])),
        dict(out=torch.empty_like(query[:4])),
        dict(window_size=(4, 4)),
        dict(window_size=(-2, 0)),
        dict(max_seqlen_k=-1),
        dict(max_seqlen_k=2**31),
    ]
    for override in bad:
        with pytest.raises(InvalidInput):
            _prefill(inputs, 8, **override)
    with pytest.raises(InvalidInput):
        _prefill(inputs, 2**31)


def test_rejects_invalid_kv_descales():
    inputs = _inputs([8], [8], 64, 8, 16)
    fp8 = _inputs([8], [8], 64, 8, 16, kv_dtype=torch.float8_e4m3fn)
    k_descale, v_descale = _descales(1.0, 1.0)
    bad = [
        (inputs, dict(k_descale=k_descale, v_descale=v_descale)),
        (fp8, dict()),
        (fp8, dict(k_descale=k_descale)),
        (fp8, dict(k_descale=k_descale.double(), v_descale=v_descale)),
        (fp8, dict(k_descale=torch.ones(2, device=DEVICE), v_descale=v_descale)),
        (fp8, dict(k_descale=k_descale, v_descale=v_descale, v=inputs[2])),
    ]
    for case_inputs, override in bad:
        with pytest.raises(InvalidInput):
            _prefill(case_inputs, 8, **override)


def test_rejects_fp8_cache_spanning_int32():
    """FP8 tiles use int32 offsets, so a strided cache view is bounded by its last element, not its element count
    (meta tensors, so nothing is allocated)."""
    fp8 = _inputs([8], [8], 64, 4, 16, kv_dtype=torch.float8_e4m3fn)
    k_descale, v_descale = _descales(1.0, 1.0)

    def cache(blocks, block_stride):
        return torch.empty_strided((blocks, 16, 4, 64), (block_stride, 256, 64, 1), dtype=torch.float8_e4m3fn,
                                   device="meta")

    # Under 2**31 elements, but the last page starts at 2**32 (or exactly 2**31): rejected.
    for wide in (cache(262145, 16384), cache(65537, 32768)):
        assert wide.numel() < 2**31
        with pytest.raises(InvalidInput, match="spanning under 2\\*\\*31"):
            _prefill(fp8, 8, k=wide, v=wide, k_descale=k_descale, v_descale=v_descale)
    # Padded pages whose last element is below 2**31 pass this check and fail later on the meta device.
    fits = cache(65536, 32768)
    with pytest.raises(InvalidInput, match="all tensors on one device"):
        _prefill(fp8, 8, k=fits, v=fits, k_descale=k_descale, v_descale=v_descale)


def test_rejects_grad_tracked_out():
    inputs = _inputs([130], [900], 64, 8, 16)
    with pytest.raises(UnsupportedBackward):
        _prefill(inputs, 130, out=torch.empty_like(inputs[0]).requires_grad_())


@pytest.mark.parametrize("name", ["k_descale", "v_descale"])
def test_rejects_grad_tracked_descales(name):
    inputs = _inputs([130], [900], 64, 8, 16, kv_dtype=torch.float8_e4m3fn, k_scale=0.05, v_scale=0.2)
    descales = dict(zip(("k_descale", "v_descale"), _descales(0.05, 0.2)))
    descales[name].requires_grad_()
    with pytest.raises(UnsupportedBackward):
        _prefill(inputs, 130, **descales)
