"""Correctness for the gfx950 ``tlx.ops`` paged-attention decode operators."""

import math

import pytest
import torch
import triton
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import (
    InvalidInput,
    allocate_paged_kv_cache_5d,
    paged_attention_decode,
    paged_attention_decode_workspace,
    reshape_and_cache_5d,
)
from triton.tlx.ops.kernels.paged_attn import gfx950_decode as pa

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="gfx950 paged decode requires CDNA4")

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _kv_layout(cache_layout):
    return "NHD" if cache_layout == "nhd" else "HND"


def _inputs(ctx_lens, num_kv_heads, group, head_dim, page_size, query_length=1, cache_layout="5d"):
    return pa.build_inputs(len(ctx_lens), ctx_lens, num_kv_heads * group, num_kv_heads, head_dim, page_size,
                           query_length=query_length, device=DEVICE, cache_layout=cache_layout)


def _reference(inputs, num_kv_heads, group, query_length=1, sliding_window=0, kv_layout="HND"):
    query, key_cache, value_cache, context_lens, block_tables = inputs
    sm_scale = query.shape[2]**-0.5
    return pa.ref_decode(query, key_cache, value_cache, context_lens, block_tables, sm_scale, num_kv_heads * group,
                         num_kv_heads, query_length, sliding_window, kv_layout)


def _assert_matches(out, expected):
    """Long contexts average V to ~1e-3 magnitudes, below atol; relative L2 catches wrong keys."""
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
    rel = (out.float() - expected).norm() / expected.norm()
    assert rel < 1e-2, f"relative L2 error {rel.item():.3e}"


@pytest.mark.parametrize("query_length", [1, 2, 4, 8], ids=lambda q: f"qlen{q}")
@pytest.mark.parametrize("window", [1, 37, 256, 2048], ids=lambda w: f"w{w}")
@pytest.mark.parametrize("cache_layout", ["4d", "nhd", "5d"])
def test_paged_attention_decode_sliding_window(window, query_length, cache_layout):
    """Windows shorter and longer than the context, starting mid-page and mid-tile."""
    ctx_lens = [max(3, query_length), 40, 300, 2100, 4133]
    kv_layout = _kv_layout(cache_layout)
    inputs = _inputs(ctx_lens, 4, 8, 64, 16, query_length, cache_layout)
    query, key_cache, value_cache, _, block_tables = inputs
    workspace = paged_attention_decode_workspace(query, key_cache, value_cache, block_tables, query_length=query_length,
                                                 sliding_window=window, max_context_len=max(ctx_lens),
                                                 kv_layout=kv_layout)
    out = paged_attention_decode(*inputs, query_length=query_length, sliding_window=window,
                                 max_context_len=max(ctx_lens), workspace=workspace, kv_layout=kv_layout)
    expected = _reference(inputs, 4, 8, query_length, window, kv_layout)
    _assert_matches(out, expected)


@pytest.mark.parametrize("num_splits", [1, 3, 16], ids=lambda s: f"split{s}")
@pytest.mark.parametrize("plan", ["lds", "streaming", "gluon_compat"])
def test_sliding_window_plans(plan, num_splits):
    """Every kernel path skips out-of-window pages and masks the window edge."""
    query_length = 1 if plan == "gluon_compat" else 4
    inputs = _inputs([2049, 5000], 2, 8, 64, 16, query_length)
    kwargs = dict(streaming_kv=True) if plan == "streaming" else dict(
        gluon_compat=True) if plan == "gluon_compat" else {}
    out = torch.empty_like(inputs[0])
    pa.pa_decode_tlx(out, *inputs, 1.0 / math.sqrt(64), query_length=query_length, num_splits=num_splits,
                     sliding_window=1000, **kwargs)
    _assert_matches(out, _reference(inputs, 2, 8, query_length, 1000))


def test_sliding_window_never_reads_out_of_window_pages():
    """Out-of-window pages may already be recycled; fill them with NaN."""
    ctx_lens = [4133, 777]
    inputs = _inputs(ctx_lens, 4, 8, 64, 16)
    query, key_cache, value_cache, context_lens, block_tables = inputs
    expected = _reference(inputs, 4, 8, 1, 512)
    key_cache, value_cache = key_cache.clone(), value_cache.clone()
    for s, ctx in enumerate(ctx_lens):
        stale = block_tables[s, :max(0, ctx - 512) // 16].long()
        key_cache[stale] = float("nan")
        value_cache[stale] = float("nan")
    out = paged_attention_decode(query, key_cache, value_cache, context_lens, block_tables, sliding_window=512)
    _assert_matches(out, expected)


def test_paged_attention_decode_rejects_bad_window():
    inputs = _inputs([40], 2, 4, 64, 16)
    with pytest.raises(InvalidInput):
        paged_attention_decode(*inputs, sliding_window=-1)


@pytest.mark.parametrize("num_kv_heads", [2, 8, 16], ids=lambda h: f"kvh{h}")
@pytest.mark.parametrize("query_length", [1, 8], ids=lambda q: f"qlen{q}")
@pytest.mark.parametrize("window", [0, 2048], ids=lambda w: f"w{w}")
def test_page_major_gqa8_decode(window, query_length, num_kv_heads):
    """GQA 8, D64, page 16 on SGLang's page-major pool: decode and 8-row
    speculative verify on full and 2048-window layers. At 16 KV heads the NHD
    and HND shapes coincide, so only kv_layout tells them apart."""
    ctx_lens = [9, 2055, 3000, 9000]
    inputs = _inputs(ctx_lens, num_kv_heads, 8, 64, 16, query_length, "nhd")
    query, key_cache, value_cache, _, block_tables = inputs
    kwargs = dict(query_length=query_length, sliding_window=window, max_context_len=max(ctx_lens), kv_layout="NHD")
    workspace = paged_attention_decode_workspace(query, key_cache, value_cache, block_tables, **kwargs)
    out = paged_attention_decode(*inputs, workspace=workspace, **kwargs)
    _assert_matches(out, _reference(inputs, num_kv_heads, 8, query_length, window, "NHD"))


def test_verify_rows_attend_causally():
    """Each of the 8 verify rows sees exactly the keys up to its own position."""
    query_length = 8
    inputs = _inputs([100], 2, 8, 64, 16, query_length, "nhd")
    out = paged_attention_decode(*inputs, query_length=query_length, kv_layout="NHD")
    query, key_cache, value_cache, context_lens, block_tables = inputs
    for qpos in range(query_length):
        prefix = (query[qpos:qpos + 1], key_cache, value_cache, context_lens - (query_length - 1 - qpos), block_tables)
        _assert_matches(out[qpos:qpos + 1], _reference(prefix, 2, 8, 1, kv_layout="NHD"))


def test_paged_attention_decode_rejects_bad_kv_layout():
    inputs = _inputs([40], 2, 8, 64, 16)
    with pytest.raises(InvalidInput):
        paged_attention_decode(*inputs, kv_layout="NHD")
    with pytest.raises(InvalidInput):
        paged_attention_decode(*_inputs([40], 2, 8, 64, 16, cache_layout="nhd"), kv_layout="nhd")


def test_paged_attention_decode_rejects_too_many_query_rows():
    with pytest.raises(InvalidInput):
        paged_attention_decode(*_inputs([40], 2, 8, 64, 16, 9, "nhd"), query_length=9, kv_layout="NHD")


@pytest.mark.parametrize("query_length", [1, 2, 3, 4, 5, 8], ids=lambda q: f"qlen{q}")
@pytest.mark.parametrize("cache_layout", ["4d", "nhd", "5d"])
def test_paged_attention_decode(query_length, cache_layout):
    inputs = _inputs([40, 71], 2, 4, 128, 16, query_length, cache_layout)
    kv_layout = _kv_layout(cache_layout)
    out = paged_attention_decode(*inputs, query_length=query_length, kv_layout=kv_layout)
    _assert_matches(out, _reference(inputs, 2, 4, query_length, kv_layout=kv_layout))


@pytest.mark.parametrize("query_length", [1, 4], ids=lambda q: f"qlen{q}")
@pytest.mark.parametrize("page_size", [16, 64], ids=lambda p: f"p{p}")
def test_paged_attention_decode_workspace_and_out(page_size, query_length):
    ctx_lens = [513, 2049]
    inputs = _inputs(ctx_lens, 2, 8, 64, page_size, query_length)
    query, key_cache, value_cache, _, block_tables = inputs
    workspace = paged_attention_decode_workspace(query, key_cache, value_cache, block_tables, query_length=query_length,
                                                 max_context_len=max(ctx_lens))
    out = torch.empty_like(query)
    result = paged_attention_decode(*inputs, query_length=query_length, max_context_len=max(ctx_lens), out=out,
                                    workspace=workspace)
    assert result is out
    _assert_matches(out, _reference(inputs, 2, 8, query_length))


def test_paged_attention_decode_rejects_bad_inputs():
    query, key_cache, value_cache, context_lens, block_tables = _inputs([40], 2, 4, 64, 16)
    with pytest.raises(InvalidInput):
        paged_attention_decode(query, key_cache, value_cache, context_lens, block_tables.long())
    with pytest.raises(InvalidInput):
        paged_attention_decode(query[:, :7], key_cache, value_cache, context_lens, block_tables)
    with pytest.raises(InvalidInput):
        paged_attention_decode(query, key_cache, value_cache, context_lens, block_tables, query_length=2)


@pytest.mark.parametrize("query_length", [1, 2, 3, 4, 6, 8], ids=lambda q: f"qlen{q}")
@pytest.mark.parametrize("num_splits", [1, 4], ids=["split1", "split4"])
@pytest.mark.parametrize("cache_layout", ["4d", "nhd", "5d"])
def test_split_plans(num_splits, query_length, cache_layout):
    """Split-K paged decode with GQA, incl. speculative-decoding verification (query_length 1-8)."""
    inputs = _inputs([40, 71], 2, 4, 128, 16, query_length, cache_layout)
    kv_layout = _kv_layout(cache_layout)
    out = torch.empty_like(inputs[0])
    pa.pa_decode_tlx(out, *inputs, 1.0 / math.sqrt(128), query_length=query_length, num_splits=num_splits,
                     kv_layout=kv_layout)
    _assert_matches(out, _reference(inputs, 2, 4, query_length, kv_layout=kv_layout))


@pytest.mark.parametrize("cache_layout", ["4d", "nhd", "5d"])
def test_page64_high_splits(cache_layout):
    inputs = _inputs([65, 257, 513], 2, 4, 64, 64, cache_layout=cache_layout)
    kv_layout = _kv_layout(cache_layout)
    out = torch.empty_like(inputs[0])
    pa.pa_decode_tlx(out, *inputs, 1.0 / math.sqrt(64), num_splits=128, kv_layout=kv_layout)
    _assert_matches(out, _reference(inputs, 2, 4, kv_layout=kv_layout))


@pytest.mark.parametrize("cache_layout", ["4d", "nhd", "5d"])
def test_page16_tile_boundaries(cache_layout):
    inputs = _inputs([15, 16, 17, 63, 64, 65], 2, 4, 128, 16, cache_layout=cache_layout)
    kv_layout = _kv_layout(cache_layout)
    out = torch.empty_like(inputs[0])
    pa.pa_decode_tlx(out, *inputs, 1.0 / math.sqrt(128), num_splits=4, kv_layout=kv_layout)
    _assert_matches(out, _reference(inputs, 2, 4, kv_layout=kv_layout))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_reshape_and_cache_5d(dtype, strided):
    """Slot-mapped writes populate packed storage without a 4-D staging cache."""
    torch.manual_seed(42)
    num_blocks, num_kv_heads, page_size, head_dim = 5, 2, 16, 64
    key_cache, value_cache = allocate_paged_kv_cache_5d(num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype,
                                                        device=DEVICE)
    if strided:
        x = key_cache.shape[-1]
        key_cache = torch.empty(num_blocks, num_kv_heads, head_dim // x, page_size, x * 2, dtype=dtype,
                                device=DEVICE)[..., ::2]
        value_cache = torch.empty(num_blocks, num_kv_heads, page_size // x, head_dim, x * 2, dtype=dtype,
                                  device=DEVICE)[..., ::2]
    key_cache.zero_()
    value_cache.zero_()

    if strided:
        key = torch.randn(7, num_kv_heads, head_dim * 2, dtype=dtype, device=DEVICE)[..., ::2]
        value = torch.randn(7, num_kv_heads, head_dim * 2, dtype=dtype, device=DEVICE)[..., ::2]
    else:
        key = torch.randn(7, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
        value = torch.randn_like(key)
    slots = torch.tensor([31, 0, 64, 18, -1, 47, 5], dtype=torch.int64, device=DEVICE)
    reshape_and_cache_5d(key[:3], value[:3], key_cache, value_cache, slots[:3])
    reshape_and_cache_5d(key[3:], value[3:], key_cache, value_cache, slots[3:])

    actual_key, actual_value = pa.unpack_5d_kv_cache(key_cache, value_cache)
    expected_key = torch.zeros_like(actual_key)
    expected_value = torch.zeros_like(actual_value)
    for token, slot in enumerate(slots.tolist()):
        if slot >= 0:
            block, offset = divmod(slot, page_size)
            expected_key[block, :, offset, :] = key[token]
            expected_value[block, :, offset, :] = value[token]

    torch.testing.assert_close(actual_key, expected_key, atol=0, rtol=0)
    torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)


@pytest.mark.parametrize("page_size", [16, 64], ids=lambda p: f"p{p}")
@pytest.mark.parametrize("num_splits", [1, 4], ids=["fused", "split4"])
@pytest.mark.parametrize("query_length", [1, 4], ids=lambda q: f"qlen{q}")
def test_streaming_register(page_size, num_splits, query_length):
    """Packed K/V register pipeline matches dense attention across KV tiles."""
    inputs = _inputs([513, 2049], 2, 8, 64, page_size, query_length)
    query, key_cache, value_cache, _, block_tables = inputs
    out = torch.empty_like(query)
    workspace = None
    config = None
    if page_size == 64 and num_splits == 4 and query_length == 1:
        config = pa.get_pa_decode_config(query, key_cache, value_cache, block_tables, query_length=query_length,
                                         num_splits=num_splits, streaming_kv=True)
        workspace = pa.allocate_pa_decode_workspace(query, key_cache, config)
    pa.pa_decode_tlx(out, *inputs, 1.0 / math.sqrt(64), query_length=query_length, num_splits=num_splits,
                     streaming_kv=True, workspace=workspace, config=config)
    _assert_matches(out, _reference(inputs, 2, 8, query_length))


@pytest.mark.parametrize("num_splits", [1, 2, 32], ids=["fused", "split2", "split32"])
@pytest.mark.parametrize("page_size", [16, 64], ids=["p16", "p64"])
def test_gluon_compat(page_size, num_splits):
    """The qlen1/group8 Gluon-compatible ownership and K prefetch path is correct."""
    inputs = _inputs([257, 513], 2, 8, 64, page_size)
    out = torch.empty_like(inputs[0])
    pa.pa_decode_tlx(out, *inputs, 1.0 / math.sqrt(64), query_length=1, num_splits=num_splits, gluon_compat=True)
    _assert_matches(out, _reference(inputs, 2, 8))


@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("context", [8192, 32768], ids=["8k", "32k"])
def test_b1_auto_packed_config(num_kv_heads, context):
    """Packed qlen1 page-16 decode streams 256-key tiles with ~384 programs."""
    group, head_dim, page_size, x = 8, 64, 16, 8
    query = torch.empty((1, num_kv_heads * group, head_dim), dtype=torch.bfloat16, device=DEVICE)
    key_cache = torch.empty((1, num_kv_heads, head_dim // x, page_size, x), dtype=torch.bfloat16, device=DEVICE)
    value_cache = torch.empty((1, num_kv_heads, page_size // x, head_dim, x), dtype=torch.bfloat16, device=DEVICE)
    block_tables = torch.empty((1, context // page_size), dtype=torch.int32, device=DEVICE)

    config = pa.get_pa_decode_config(query, key_cache, value_cache, block_tables, max_context_len=context)
    assert not config.gluon_compat
    assert config.streaming_kv
    assert config.num_splits == min({4: 128, 8: 64}[num_kv_heads], context // 256)
    assert config.block_n == 256
    assert config.num_warps == 4
    assert config.waves_per_eu == 2

    control = pa.get_pa_decode_config(query, key_cache, value_cache, block_tables, max_context_len=context,
                                      gluon_compat=True)
    assert control.gluon_compat


_PLAN_KWARGS = {
    "lds": dict(streaming_kv=False, gluon_compat=False),
    "streaming": dict(streaming_kv=True),
    "gluon_compat": dict(gluon_compat=True),
}


def _huge_cache_like(key_cache, value_cache):
    """Copy a packed cache into the top blocks of one whose K/V exceed 2**31 elements."""
    per_block = key_cache[0].numel()
    num_blocks = (3 << 31) // per_block + key_cache.shape[0]
    huge_k, huge_v = allocate_paged_kv_cache_5d(num_blocks, *key_cache.shape[1:2], key_cache.shape[3],
                                                key_cache.shape[2] * key_cache.shape[4], dtype=key_cache.dtype,
                                                device=DEVICE)
    base = num_blocks - key_cache.shape[0]
    huge_k[base:] = key_cache
    huge_v[base:] = value_cache
    return huge_k, huge_v, base


@pytest.mark.parametrize("plan", ["lds", "streaming", "gluon_compat"])
def test_cache_over_2gi_elements(plan):
    """Serving KV pools exceed 2**31 elements per layer; pages near the top must still be addressed."""
    query_length = 1 if plan == "gluon_compat" else 4
    inputs = _inputs([1000, 3000], 4, 8, 64, 16, query_length)
    query, key_cache, value_cache, context_lens, block_tables = inputs
    expected = _reference(inputs, 4, 8, query_length)
    huge_k, huge_v, base = _huge_cache_like(key_cache, value_cache)
    out = torch.empty_like(query)
    pa.pa_decode_tlx(out, query, huge_k, huge_v, context_lens, block_tables + base, 1.0 / math.sqrt(64),
                     query_length=query_length, num_splits=4, **_PLAN_KWARGS[plan])
    _assert_matches(out, expected)


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_reshape_and_cache_5d_over_2gi_elements(slot_dtype):
    key_cache, value_cache = allocate_paged_kv_cache_5d(2, 4, 16, 64, dtype=torch.bfloat16, device=DEVICE)
    key_cache.zero_()
    value_cache.zero_()
    huge_k, huge_v, base = _huge_cache_like(key_cache, value_cache)
    key = torch.randn(3, 4, 64, dtype=torch.bfloat16, device=DEVICE)
    value = torch.randn_like(key)
    slots = torch.tensor([0, 17, 31], dtype=slot_dtype, device=DEVICE)
    reshape_and_cache_5d(key, value, key_cache, value_cache, slots)
    reshape_and_cache_5d(key, value, huge_k, huge_v, slots + base * 16)
    torch.testing.assert_close(huge_k[base:], key_cache, atol=0, rtol=0)
    torch.testing.assert_close(huge_v[base:], value_cache, atol=0, rtol=0)
