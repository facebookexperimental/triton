"""Correctness for the gfx950 ``tlx.ops`` paged-attention decode operators."""

import math

import pytest
import torch
import triton
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import (
    InvalidInput,
    UnsupportedBackward,
    allocate_paged_kv_cache_5d,
    paged_attention_decode,
    paged_attention_decode_workspace,
    reshape_and_cache_5d,
    rmsnorm_gate,
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


def test_paged_attention_decode_rejects_too_many_padded_rows():
    """5 x 12 = 60 real rows, but QLEN and GROUP pad to 8 x 16 = 128 MFMA rows."""
    with pytest.raises(InvalidInput):
        paged_attention_decode(*_inputs([40], 2, 12, 64, 16, 5, "nhd"), query_length=5, kv_layout="NHD")


@pytest.mark.parametrize("group,query_length", [(12, 1), (12, 4), (6, 5), (6, 8), (3, 8)])
@pytest.mark.parametrize("cache_layout", ["4d", "nhd", "5d"])
def test_non_power_of_two_group(group, query_length, cache_layout):
    inputs = _inputs([40, 71], 2, group, 64, 16, query_length, cache_layout)
    kv_layout = _kv_layout(cache_layout)
    out = paged_attention_decode(*inputs, query_length=query_length, kv_layout=kv_layout)
    _assert_matches(out, _reference(inputs, 2, group, query_length, kv_layout=kv_layout))


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


def _fp8_inputs(ctx_lens, num_kv_heads, group, query_length, cache_layout, q_fp8, k_scale=0.05, v_scale=0.2,
                q_scale=0.01):
    """FP8 e4m3fn K/V (and optionally Q) stored as value / scale, as SGLang's FP8 KV pool holds them; K, V and Q get
    different scales so a scale applied to the wrong operand shows."""
    query, key_cache, value_cache, context_lens, block_tables = _inputs(ctx_lens, num_kv_heads, group, 64, 16,
                                                                        query_length, cache_layout)
    fp8 = torch.float8_e4m3fn
    key_cache = (key_cache.float() / k_scale).to(fp8)
    value_cache = (value_cache.float() / v_scale).to(fp8)
    descales = dict(k_descale=torch.tensor([k_scale], device=DEVICE), v_descale=torch.tensor([v_scale], device=DEVICE))
    if q_fp8:
        query = (query.float() / q_scale).to(fp8)
        descales["q_descale"] = torch.tensor([q_scale], device=DEVICE)
    return (query, key_cache, value_cache, context_lens, block_tables), descales


@pytest.mark.parametrize("q_fp8", [False, True], ids=["bf16_q", "fp8_q"])
@pytest.mark.parametrize("window", [0, 37, 2048], ids=lambda w: f"w{w}")
@pytest.mark.parametrize("query_length", [1, 4, 8], ids=lambda q: f"qlen{q}")
@pytest.mark.parametrize("cache_layout", ["4d", "nhd"])
def test_fp8_kv_cache(cache_layout, query_length, window, q_fp8):
    """Full-attention decode (direct-load path), window layers and verify rows (LDS path) on FP8 caches."""
    ctx_lens = [max(3, query_length), 40, 2055, 9000]
    kv_layout = _kv_layout(cache_layout)
    inputs, descales = _fp8_inputs(ctx_lens, 4, 8, query_length, cache_layout, q_fp8)
    query, key_cache, value_cache, _, block_tables = inputs
    kwargs = dict(query_length=query_length, sliding_window=window, max_context_len=max(ctx_lens), kv_layout=kv_layout)
    workspace = paged_attention_decode_workspace(query, key_cache, value_cache, block_tables, **kwargs)
    out = paged_attention_decode(*inputs, workspace=workspace, **kwargs, **descales)
    assert out.dtype == torch.bfloat16
    sm_scale = query.shape[2]**-0.5
    expected = pa.ref_decode(query, key_cache, value_cache, inputs[3], block_tables, sm_scale, 32, 4, query_length,
                             window, kv_layout, **descales)
    _assert_matches(out, expected)


@pytest.mark.parametrize("num_kv_heads", [2, 16], ids=lambda h: f"kvh{h}")
def test_fp8_kv_cache_sglang_shape(num_kv_heads):
    """GQA 8, D64, page 16 page-major pool with FP8 Q and an FP16 output buffer."""
    inputs, descales = _fp8_inputs([9, 3000], num_kv_heads, 8, 1, "nhd", True)
    out = torch.empty(inputs[0].shape, dtype=torch.float16, device=DEVICE)
    result = paged_attention_decode(*inputs, kv_layout="NHD", out=out, **descales)
    assert result is out
    query, key_cache, value_cache, context_lens, block_tables = inputs
    expected = pa.ref_decode(query, key_cache, value_cache, context_lens, block_tables, 64**-0.5, num_kv_heads * 8,
                             num_kv_heads, 1, 0, "NHD", **descales)
    _assert_matches(out, expected)


def test_fp8_kv_cache_rejects_invalid_inputs():
    inputs, descales = _fp8_inputs([40], 2, 8, 1, "nhd", False)
    query, key_cache, value_cache, context_lens, block_tables = inputs
    bf16 = _inputs([40], 2, 8, 64, 16, cache_layout="nhd")
    fp8_q = (query.float() / 0.01).to(torch.float8_e4m3fn)
    q_descale = torch.tensor([0.01], device=DEVICE)
    bad = [
        ((query, key_cache, value_cache, context_lens, block_tables), {}),
        ((query, key_cache, value_cache, context_lens, block_tables), dict(k_descale=descales["k_descale"])),
        ((query, key_cache, value_cache, context_lens, block_tables), dict(descales, q_descale=q_descale)),
        ((fp8_q, key_cache, value_cache, context_lens, block_tables), descales),
        ((fp8_q, *bf16[1:]), dict(q_descale=q_descale)),
        (bf16, descales),
        ((query, key_cache, value_cache, context_lens, block_tables),
         dict(descales, out=torch.empty(query.shape, dtype=torch.float32, device=DEVICE))),
    ]
    for case_inputs, kwargs in bad:
        with pytest.raises(InvalidInput):
            paged_attention_decode(*case_inputs, kv_layout="NHD", **kwargs)
    k5, v5 = allocate_paged_kv_cache_5d(4, 2, 16, 64, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(InvalidInput):
        paged_attention_decode(query, k5.to(torch.float8_e4m3fn), v5.to(torch.float8_e4m3fn), context_lens,
                               block_tables[:, :3], **descales)


@pytest.mark.parametrize("window", [0, 37, 2047], ids=lambda w: f"w{w}")
@pytest.mark.parametrize("num_seqs", [1, 8, 16, 32, 64], ids=lambda n: f"bs{n}")
def test_fp8_two_kv_head_decode(num_seqs, window):
    """16 Q / 2 KV heads (TP8) FP8 decode as SGLang's CUDA graph calls it: page-major pool, a 32K plan bound, mixed
    context lengths. Window layers and full layers from 16 programs take the direct-load plans."""
    ctx_lens = [[3, 40, 2055, 9000, 700, 4097, 31000, 16][i % 8] for i in range(num_seqs)]
    inputs, descales = _fp8_inputs(ctx_lens, 2, 8, 1, "nhd", True)
    query, key_cache, value_cache, context_lens, block_tables = inputs
    kwargs = dict(sliding_window=window, max_context_len=32768, kv_layout="NHD")
    config = pa.get_pa_decode_config(query, key_cache, value_cache, block_tables, **kwargs)
    assert config.direct_kv and config.block_n == (256 if num_seqs >= (8 if window == 0 else 16) else 128)
    workspace = paged_attention_decode_workspace(query, key_cache, value_cache, block_tables, **kwargs)
    out = paged_attention_decode(*inputs, workspace=workspace, **kwargs, **descales)
    expected = pa.ref_decode(query, key_cache, value_cache, context_lens, block_tables, 64**-0.5, 16, 2, 1, window,
                             "NHD", **descales)
    _assert_matches(out, expected)


def test_fp8_two_kv_head_plans():
    """Plans for FP8 single-token page-major decode with 2 KV heads under a 32K bound (tuned on MI350X): direct loads
    on window layers, and a 256-key tile at ~512 programs on full layers from 16 programs; BF16 keeps its plans."""
    fp8 = torch.float8_e4m3fn
    expected = {  # (dtype, window, num_seqs) -> (direct_kv, block_n, kv_stages, num_splits)
        (fp8, 0, 4): (True, 64, 3, 64),
        (fp8, 0, 8): (True, 256, 1, 32),
        (fp8, 0, 16): (True, 256, 1, 16),
        (fp8, 0, 32): (True, 256, 1, 8),
        (fp8, 0, 64): (True, 256, 1, 4),
        (fp8, 2047, 1): (True, 128, 2, 17),
        (fp8, 2047, 8): (True, 128, 2, 17),
        (fp8, 2047, 16): (True, 256, 1, 9),
        (fp8, 2047, 32): (True, 256, 1, 8),
        (fp8, 2047, 64): (True, 256, 1, 4),
        (torch.bfloat16, 0, 16): (True, 64, 3, 16),
        (torch.bfloat16, 2047, 16): (False, 128, 1, 8),
    }
    cache = torch.empty(64, 16, 2, 64, dtype=fp8, device=DEVICE)
    for (dtype, window, num_seqs), plan in expected.items():
        query = torch.empty(num_seqs, 16, 64, dtype=dtype, device=DEVICE)
        block_tables = torch.empty(num_seqs, 2048, dtype=torch.int32, device=DEVICE)
        c = pa.get_pa_decode_config(query, cache.to(dtype), cache.to(dtype), block_tables, max_context_len=32768,
                                    sliding_window=window, kv_layout="NHD")
        assert (c.direct_kv, c.block_n, c.kv_stages, c.num_splits) == plan, (dtype, window, num_seqs)


@pytest.mark.parametrize("window", [0, 2047], ids=lambda w: f"w{w}")
def test_int64_context_lens(window):
    """SGLang passes int64 seq_lens; they give the same output as int32, with no cast kernel."""
    inputs, descales = _fp8_inputs([9, 3000, 9000] * 6, 2, 8, 1, "nhd", True)
    query, key_cache, value_cache, context_lens, block_tables = inputs
    kwargs = dict(sliding_window=window, max_context_len=9000, kv_layout="NHD")
    workspace = paged_attention_decode_workspace(query, key_cache, value_cache, block_tables, **kwargs)
    out32 = paged_attention_decode(*inputs, workspace=workspace, **kwargs, **descales)
    out64 = paged_attention_decode(query, key_cache, value_cache, context_lens.long(), block_tables,
                                   workspace=workspace, **kwargs, **descales)
    torch.testing.assert_close(out64, out32, atol=0, rtol=0)


def test_paged_attention_decode_rejects_grad_tracked_out():
    inputs = _inputs([40, 300], 2, 8, 64, 16, cache_layout="nhd")
    with pytest.raises(UnsupportedBackward):
        paged_attention_decode(*inputs, kv_layout="NHD", out=torch.empty_like(inputs[0]).requires_grad_())


def _projection_gate(num_tokens, num_q_heads, seed=0):
    """The output gate as a column slice of a wider projection output (SGLang's fused QKV + gate at TP8)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    qkv = torch.randn(num_tokens, 2304, generator=g, device=DEVICE).mul_(2).to(torch.bfloat16)
    return qkv[:, 2304 - num_q_heads * 64:]


@pytest.mark.parametrize("window", [0, 2047], ids=lambda w: f"w{w}")
@pytest.mark.parametrize("num_seqs", [1, 4, 16, 32], ids=lambda n: f"bs{n}")
def test_out_gate_is_rmsnorm_gate_of_the_output(num_seqs, window):
    """With out_gate, the reduce writes tlx.ops.rmsnorm_gate(output, out_gate) bit for bit (FP8 TP8 serving plan)."""
    ctx_lens = [[3, 40, 2055, 9000, 700, 4097, 31000, 16][i % 8] for i in range(num_seqs)]
    inputs, descales = _fp8_inputs(ctx_lens, 2, 8, 1, "nhd", True)
    query, key_cache, value_cache, context_lens, block_tables = inputs
    kwargs = dict(sliding_window=window, max_context_len=32768, kv_layout="NHD")
    gate = _projection_gate(num_seqs, 16)
    workspace = paged_attention_decode_workspace(query, key_cache, value_cache, block_tables, norm_gate=True, **kwargs)
    plain = paged_attention_decode(*inputs, workspace=workspace, **kwargs, **descales)
    expected = rmsnorm_gate(plain.reshape(num_seqs, -1), gate, eps=1e-5)
    out = paged_attention_decode(*inputs, workspace=workspace, out_gate=gate, out_norm_eps=1e-5, **kwargs, **descales)
    assert out.shape == plain.shape
    torch.testing.assert_close(out.reshape(num_seqs, -1), expected, atol=0, rtol=0)


@pytest.mark.parametrize("query_length", [1, 4], ids=lambda q: f"qlen{q}")
@pytest.mark.parametrize("cache_layout", ["4d", "nhd", "5d"])
def test_out_gate_bf16_plans(cache_layout, query_length):
    """BF16 caches, verify rows and every partition plan keep the bitwise rmsnorm_gate contract."""
    inputs = _inputs([query_length + 2, 300, 2055, 9000], 2, 8, 64, 16, query_length, cache_layout)
    num_tokens = inputs[0].shape[0]
    kv_layout = _kv_layout(cache_layout)
    gate = _projection_gate(num_tokens, 16, seed=1)
    plain = paged_attention_decode(*inputs, query_length=query_length, kv_layout=kv_layout)
    out = paged_attention_decode(*inputs, query_length=query_length, kv_layout=kv_layout, out_gate=gate)
    expected = rmsnorm_gate(plain.reshape(num_tokens, -1), gate)
    torch.testing.assert_close(out.reshape(num_tokens, -1), expected, atol=0, rtol=0)


def test_out_gate_needs_two_splits():
    """A one-split (fused) plan becomes two splits with out_gate, whose reduce applies the norm and gate; an empty
    split adds nothing."""
    inputs, descales = _fp8_inputs([40] * 256, 2, 8, 1, "nhd", True)
    query, key_cache, value_cache, context_lens, block_tables = inputs
    kwargs = dict(sliding_window=2047, max_context_len=32768, kv_layout="NHD")
    assert pa.get_pa_decode_config(query, key_cache, value_cache, block_tables, **kwargs).num_splits == 1
    assert pa.get_pa_decode_config(query, key_cache, value_cache, block_tables, norm_gate=True,
                                   **kwargs).num_splits == 2
    gate = _projection_gate(256, 16, seed=2)
    out = paged_attention_decode(*inputs, out_gate=gate, **kwargs, **descales)
    two = torch.empty_like(out)
    pa.pa_decode_tlx(two, query, key_cache, value_cache, context_lens, block_tables, 64**-0.5, num_splits=2,
                     max_context_len=32768, sliding_window=2047, kv_layout="NHD", **descales)
    torch.testing.assert_close(out.reshape(256, -1), rmsnorm_gate(two.reshape(256, -1), gate), atol=0, rtol=0)


@pytest.mark.parametrize("ctx_lens", [[40], [9], [40, 300, 2055] * 64], ids=["ctx40", "one_page", "bs192"])
def test_out_gate_packed_5d_one_split_plans(ctx_lens):
    """The packed 5-D plan picks one split for a block table within one 256-key tile and from 384 programs on; with
    out_gate it takes two (one page leaves the second split empty) and writes rmsnorm_gate of the two-split output."""
    inputs = _inputs(ctx_lens, 2, 8, 64, 16)
    query, key_cache, value_cache, context_lens, block_tables = inputs
    assert pa.get_pa_decode_config(query, key_cache, value_cache, block_tables).num_splits == 1
    assert pa.get_pa_decode_config(query, key_cache, value_cache, block_tables, norm_gate=True).num_splits == 2
    num_tokens = query.shape[0]
    gate = _projection_gate(num_tokens, 16, seed=3)
    out = paged_attention_decode(*inputs, out_gate=gate)
    two = torch.empty_like(query)
    pa.pa_decode_tlx(two, query, key_cache, value_cache, context_lens, block_tables, 64**-0.5, num_splits=2)
    expected = rmsnorm_gate(two.reshape(num_tokens, -1), gate)
    torch.testing.assert_close(out.reshape(num_tokens, -1), expected, atol=0, rtol=0)


@pytest.mark.parametrize("name", ["out_gate", "k_descale", "v_descale", "q_descale"])
def test_paged_attention_decode_rejects_grad_tracked_gate_and_descales(name):
    """Every float tensor argument is checked for autograd, not only the query, the caches and out."""
    inputs, descales = _fp8_inputs([40, 300], 2, 8, 1, "nhd", True)
    kwargs = dict(kv_layout="NHD", out_gate=_projection_gate(2, 16), **descales)
    kwargs[name] = kwargs[name].detach().requires_grad_()
    with pytest.raises(UnsupportedBackward):
        paged_attention_decode(*inputs, **kwargs)


def test_out_gate_rejects_bad_inputs():
    inputs, descales = _fp8_inputs([40, 300], 2, 8, 1, "nhd", True)
    gate = _projection_gate(2, 16)
    bad = [
        dict(out_gate=gate.float()),
        dict(out_gate=gate[:, :512]),
        dict(out_gate=gate[:1]),
        dict(out_gate=gate.t().contiguous().t()),
        dict(out_gate=gate, out_norm_eps=0.0),
        dict(out_gate=gate, out=torch.empty(2, 16, 64, dtype=torch.float16, device=DEVICE)),
    ]
    for kwargs in bad:
        with pytest.raises(InvalidInput):
            paged_attention_decode(*inputs, kv_layout="NHD", **descales, **kwargs)
    bf16 = _inputs([40, 300], 2, 8, 128, 16, cache_layout="nhd")
    with pytest.raises(InvalidInput):
        paged_attention_decode(*bf16, kv_layout="NHD", out_gate=torch.zeros(2, 2048, dtype=torch.bfloat16,
                                                                            device=DEVICE))
