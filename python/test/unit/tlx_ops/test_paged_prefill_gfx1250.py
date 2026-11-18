"""Paged BF16 GQA: independent FP64 reference, boundaries and recovery."""

import pytest
import torch
import triton
from triton._internal_testing import is_hip_gfx1250
from triton.tlx.ops import InvalidInput, Space, BitwiseInvarianceSpec, UnsupportedBackward, UnsupportedOp, paged_prefill, paged_attention_prefill

pytestmark = pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")


def _pack_cache(k, v):
    pages, _, heads, _ = k.shape
    kk = k.reshape(pages, 64, heads, 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    vv = v.reshape(pages, 8, 8, heads, 64).permute(0, 3, 1, 4, 2).contiguous()
    return kk, vv


def _packed_call(q, k, v, out, cu_q, used_k, table, max_q, max_k, window, shuffled):
    return paged_attention_prefill(q, k, v, out, cu_q, max_q, used_k, max_k, 0.125, True, (window - 1, 0) if window else
                                   (-1, -1), table, 0, None, None, None, shuffled_kv_cache=shuffled)


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("window", [0, 2048])
def test_paged_prefill_packed_runtime_graph(shuffled, window):
    from triton.tlx.ops.kernels.paged_prefill.gfx1250_wmma import packed_prefill_kernel

    torch.manual_seed(41)
    q = torch.randn((2048, 16, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((5 * 65 + 3, 64, 2, 64), device=q.device, dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.randperm(k.shape[0], device=q.device, dtype=torch.int32)[:5 * 65].reshape(5, 65)
    cu_q = torch.tensor([0, 0, 8, 73, 586, 1355], device=q.device, dtype=torch.int32)
    used_k = torch.tensor([0, 4097, 129, 1025, 2817], device=q.device, dtype=torch.int32)
    kp, vp = _pack_cache(k, v) if shuffled else (k, v)
    out = torch.empty_like(q)
    args = (q, kp, vp, out, cu_q, used_k, table, 1024, 4097, window, shuffled)
    _packed_call(*args)
    cache = packed_prefill_kernel.device_caches[torch.cuda.current_device()][0]
    compiled = set(cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _packed_call(*args)
    for mode, qlens, klens in (
        ("random", [0, 8, 65, 513, 769], [0, 4097, 129, 1025, 2817]),
        ("cap44", [1, 63, 17, 257, 769], [1, 128, 2049, 4097, 1025]),
        ("late_spike", [0, 8, 65, 513, 769], [0, 4097, 129, 1025, 2817]),
        ("negative", [1, 63, 17, 257, 769], [1, 128, 2049, 4097, 1025]),
        ("random", [0, 8, 65, 513, 769], [0, 4097, 129, 1025, 2817]),
    ):
        offsets = [0]
        for n in qlens:
            offsets.append(offsets[-1] + n)
        cu_q.copy_(torch.tensor(offsets, device=q.device, dtype=cu_q.dtype))
        used_k.copy_(torch.tensor(klens, device=q.device, dtype=used_k.dtype))
        q.normal_()
        k.normal_()
        v.normal_()
        if mode == "cap44":
            q.fill_(1)
            k.fill_(5.5)
        elif mode == "late_spike":
            q.fill_(1)
            k.zero_()
            for b, sk in enumerate(klens):
                if sk:
                    position = max(0, sk - 1024)
                    k[table[b, position // 64].long(), position % 64] = 32
        elif mode == "negative":
            q.fill_(2.0**50)
            k.fill_(-(2.0**50))
            v.fill_(1)
        for b, sk in enumerate(klens):
            if sk % 64:
                k[table[b, sk // 64].long(), sk % 64:] = float("nan")
                v[table[b, sk // 64].long(), sk % 64:] = float("nan")
        if shuffled:
            kk, vv = _pack_cache(k, v)
            kp.copy_(kk)
            vp.copy_(vv)
        out.fill_(float("nan"))
        graph.replay()
        for b, (sq, sk) in enumerate(zip(qlens, klens)):
            if sq:
                qq = q[offsets[b]:offsets[b + 1]][None]
                oo = out[offsets[b]:offsets[b + 1]][None]
                _check(qq, k, v, table[b:b + 1, :triton.cdiv(sk, 64)], sk, window, oo)
        assert torch.isnan(out[offsets[-1]:]).all()
        expected = out[:offsets[-1]].clone()
        out.fill_(float("nan"))
        graph.replay()
        assert torch.equal(out[:offsets[-1]], expected)
        _packed_call(*args)
        assert set(cache) == compiled


@pytest.mark.parametrize("shuffled", [False, True])
def test_paged_prefill_runtime_capacity_reuses_binary(shuffled):
    from triton.tlx.ops.kernels.paged_prefill.gfx1250_wmma import launch_packed

    binaries = []
    for sq, sk, capacity in ((769, 4097, 81), (1031, 8192, 145), (513, 2048, 49)):
        q, k, v, table, _ = _inputs(1, sq, sk)
        # Change both physical pool capacity and page-table row stride.
        larger = torch.randn((capacity, 64, 2, 64), device=q.device, dtype=q.dtype)
        larger[:k.shape[0]].copy_(k)
        values = torch.randn_like(larger)
        values[:v.shape[0]].copy_(v)
        bt = torch.full((1, capacity), -1, device=q.device, dtype=torch.int32)
        bt[:, :table.shape[1]].copy_(table)
        kk, vv = _pack_cache(larger, values) if shuffled else (larger, values)
        cu = torch.tensor([0, sq], device=q.device, dtype=torch.int32)
        used = torch.tensor([sk], device=q.device, dtype=torch.int64)
        out = torch.empty_like(q)
        compiled = launch_packed(q[0], kk, vv, bt, out[0], cu, used, sq, bm=128, shuffled=shuffled)
        _check(q, k, v, table, sk, 0, out, full=False)
        binaries.append(compiled)
    assert all(binary is binaries[0] for binary in binaries)


@pytest.mark.parametrize("kv_heads", [4, 16])
@pytest.mark.parametrize("window", [0, 2048])
def test_paged_prefill_shuffled_heads_and_query_offset(kv_heads, window):
    torch.manual_seed(43)
    offsets, lengths = [5, 5, 70, 839], [0, 129, 2817]
    q = torch.randn((851, kv_heads * 8, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((3 * 45 + 3, 64, kv_heads, 64), device=q.device, dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.randperm(k.shape[0], device=q.device, dtype=torch.int32)[:3 * 45].reshape(3, 45)
    # An empty sequence must never dereference its page table or write output.
    table[0].fill_(-1)
    for b in (1, 2):
        k[table[b, lengths[b] // 64].long(), lengths[b] % 64:] = float("nan")
        v[table[b, lengths[b] // 64].long(), lengths[b] % 64:] = float("nan")
    kp, vp = _pack_cache(k, v)
    cu = torch.tensor(offsets, device=q.device, dtype=torch.int32)
    used = torch.tensor(lengths, device=q.device, dtype=torch.int64)
    out = torch.full_like(q, float("nan"))
    _packed_call(q, kp, vp, out, cu, used, table, 769, 2817, window, True)
    for b in (1, 2):
        _check(q[offsets[b]:offsets[b + 1]][None], k, v, table[b:b + 1, :triton.cdiv(lengths[b], 64)], lengths[b],
               window, out[offsets[b]:offsets[b + 1]][None], full=False)
    assert torch.isnan(out[:offsets[0]]).all()
    assert torch.isnan(out[offsets[-1]:]).all()


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("batch", [0, 2])
def test_paged_prefill_empty_packed_cache(shuffled, batch):
    q = torch.empty((0, 16, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((0, 64, 2, 64), device=q.device, dtype=q.dtype)
    v = torch.empty_like(k)
    if shuffled:
        k, v = _pack_cache(k, v)
    cu = torch.zeros(batch + 1, device=q.device, dtype=torch.int32)
    used = torch.zeros(batch, device=q.device, dtype=torch.int32)
    table = torch.empty((batch, 0), device=q.device, dtype=torch.int32)
    out = torch.empty_like(q)
    assert _packed_call(q, k, v, out, cu, used, table, 0, 0, 0, shuffled) is out


@pytest.mark.parametrize("window", [0, 2048])
@pytest.mark.parametrize("sq", [256, 769])
@pytest.mark.parametrize("q_value,k_value", [(4096, 4224), (65536, 66560), (4128, 6368), (66048, 101888),
                                             (2.0**50, 2.0**50), (4096, -4224), (2.0**50, -(2.0**50))])
def test_paged_prefill_large_equal_logits_keep_page_weights(window, sq, q_value, k_value):
    q, k, v, table, _ = _inputs(1, sq, 4096)
    q.fill_(q_value)
    k.fill_(k_value)
    v.zero_()
    for page in range(table.shape[1]):
        v[table[0, page].long()] = page % 2
    out = paged_prefill(q, k, v, table, kv_length=4096, window=window)
    # Every key has exactly the same score: use an analytic uniform mean,
    # with alternating page values so first-page-only normalization fails.
    values = (torch.arange(4096, device=q.device) // 64 % 2).double()
    prefix_sum = torch.cat((torch.zeros(1, device=q.device, dtype=torch.float64), values.cumsum(0)))
    end = torch.arange(sq, device=q.device) + 4096 - sq + 1
    start = (end - window).clamp_min(0) if window else torch.zeros_like(end)
    expected = ((prefix_sum[end] - prefix_sum[start]) / (end - start))[None, :, None, None].expand_as(out)
    torch.testing.assert_close(out.double(), expected, atol=0.003, rtol=0.005)


@pytest.mark.parametrize("shuffled", [False, True])
def test_paged_prefill_packed_qk_cancellation_keeps_page_weights(shuffled):
    sq, sk, window = 4097, 8192, 2048
    q, k, v, table, _ = _inputs(1, sq, sk)
    q.fill_(2.0**15)
    k.fill_(2.0**15)
    k[..., 32:] = -(2.0**15)
    v.zero_()
    # Every dot product is exactly zero, but its first K=32 partial sum
    # would erase a small nonzero accumulator seed in FP32. Distinct values
    # at the window edges expose unequal weighting of boundary/interior pages.
    for page in (96, 97, 126, 127):
        v[table[0, page].long()] = 1
    kp, vp = _pack_cache(k, v) if shuffled else (k, v)
    cu = torch.tensor([0, sq], device=q.device, dtype=torch.int32)
    used = torch.tensor([sk], device=q.device, dtype=torch.int32)
    out = torch.full_like(q, float("nan"))
    _packed_call(q[0], kp, vp, out[0], cu, used, table, sq, sk, window, shuffled)
    pages = torch.arange(sk, device=q.device) // 64
    values = ((pages == 96) | (pages == 97) | (pages == 126) | (pages == 127)).double()
    prefix_sum = torch.cat((torch.zeros(1, device=q.device, dtype=torch.float64), values.cumsum(0)))
    end = torch.arange(sq, device=q.device) + sk - sq + 1
    start = (end - window).clamp_min(0)
    expected = ((prefix_sum[end] - prefix_sum[start]) / (end - start))[None, :, None, None].expand_as(out)
    torch.testing.assert_close(out.double(), expected, atol=0.003, rtol=0.005)
    torch.testing.assert_close(out[0, -1], torch.full_like(out[0, -1], 0.125), atol=0, rtol=0)


def test_paged_prefill_packed_rejects_metadata_rank_and_output_alias():
    q, k, v, table, _ = _inputs(1, 65, 129)
    cu = torch.tensor([0, 65], device=q.device, dtype=torch.int32)
    used = torch.tensor([129], device=q.device, dtype=torch.int32)
    with pytest.raises(InvalidInput, match="overlap"):
        _packed_call(q[0], k, v, q[0], cu, used, table, 65, 129, 0, False)
    bad_used = torch.tensor([[129, 129]], device=q.device, dtype=torch.int32)
    with pytest.raises(InvalidInput):
        _packed_call(q[0], k, v, torch.empty_like(q[0]), cu, bad_used, table, 65, 129, 0, False)


def test_paged_prefill_full_large_values_keep_recovery_guard():
    q, k, v, table, _ = _inputs(1, 2048, 2048)
    q.fill_(1)
    k.fill_(5.5)
    magnitude = (201 / 128) * 2.0**30
    v[table[0, :16].long()] = magnitude
    v[table[0, 16:].long()] = -magnitude
    v[table[0].long(), 63] = 0
    out = paged_prefill(q, k, v, table)
    # The last row has equal probabilities and exact cancellation. Widening
    # the uncentered denominator guard admits a finite but inaccurate PV sum.
    assert torch.count_nonzero(out[0, -1]) == 0


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("sk", [2048, 4096])
def test_paged_prefill_packed_large_values_cancellation(shuffled, sk):
    q, k, v, table, _ = _inputs(1, 2048, sk)
    q.fill_(1)
    k.fill_(5.5)
    magnitude = (201 / 128) * 2.0**30
    midpoint = sk // 128
    v[table[0, :midpoint].long()] = magnitude
    v[table[0, midpoint:].long()] = -magnitude
    v[table[0].long(), 63] = 0
    kp, vp = _pack_cache(k, v) if shuffled else (k, v)
    cu = torch.tensor([0, 2048], device=q.device, dtype=torch.int32)
    used = torch.tensor([sk], device=q.device, dtype=torch.int32)
    out = torch.empty_like(q)
    _packed_call(q[0], kp, vp, out[0], cu, used, table, 2048, sk, 0, shuffled)
    assert torch.isfinite(out).all()
    # The sampled center permits the fast path. FP32 PV accumulation can
    # leave roundoff at exact cancellation; bound it relative to the input.
    assert out[0, -1].float().abs().max() < magnitude * 1.0e-6


@pytest.mark.parametrize("shuffled", [False, True])
def test_paged_prefill_wide_runtime_tail_and_recovery(shuffled):
    max_q, sk = 16385, 65537
    q, k, v, table, _ = _inputs(1, max_q, sk)
    logical_values = (torch.arange(sk, device=q.device) // 64 % 2).double()
    page_values = (torch.arange(table.shape[1], device=q.device) % 2).to(v.dtype)
    v[table[0].long()] = page_values[:, None, None, None]
    v[table[0, -1].long(), 1:] = float("nan")
    kp, vp = _pack_cache(k, v) if shuffled else (k, v)
    cu = torch.tensor([0, max_q], device=q.device, dtype=torch.int32)
    used = torch.tensor([sk], device=q.device, dtype=torch.int32)
    out = torch.empty_like(q)
    args = (q[0], kp, vp, out[0], cu, used, table, max_q, sk, 0, shuffled)
    _packed_call(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _packed_call(*args)
    prefix_sum = torch.cat((torch.zeros(1, device=q.device, dtype=torch.float64), logical_values.cumsum(0)))
    for sq, mode in ((max_q, "cap44"), (65, "negative"), (63, "positive"), (64, "spike"), (max_q, "uniform")):
        cu[1] = sq
        q.fill_(1)
        k.zero_()
        if mode == "cap44":
            k.fill_(5.5)
        elif mode in ("negative", "positive"):
            q.fill_(2.0**50)
            k.fill_(-(2.0**50) if mode == "negative" else 2.0**50)
        elif mode == "spike":
            # An unsampled, fully visible key forces online recovery.
            k[table[0, 511].long(), 7] = 32
        k[table[0, -1].long(), 1:] = float("nan")
        if shuffled:
            kk, _ = _pack_cache(k, v)
            kp.copy_(kk)
        out.fill_(float("nan"))
        graph.replay()
        end = torch.arange(sq, device=q.device) + sk - sq + 1
        expected = torch.ones(sq, device=q.device, dtype=torch.float64) if mode == "spike" else prefix_sum[end] / end
        expected = expected[None, :, None, None].expand_as(out[:, :sq])
        torch.testing.assert_close(out[:, :sq].double(), expected, atol=0.003, rtol=0.005)
        assert torch.isnan(out[:, sq:]).all()


def _inputs(batch, sq, sk, q_offset=0, kv_heads=2):
    torch.manual_seed(41)
    q_storage = torch.full((batch * sq * kv_heads * 8 * 64 + q_offset + 1, ), 13, dtype=torch.bfloat16, device="cuda")
    q = q_storage[q_offset:-1].view(batch, sq, kv_heads * 8, 64).normal_()
    pages = triton.cdiv(sk, 64)
    k = torch.randn((batch * pages + 3, 64, kv_heads, 64), dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    table = torch.randperm(k.shape[0], device=q.device, dtype=torch.int32)[:batch * pages].reshape(batch, pages)
    if sk % 64:
        # Padding represents uninitialized cache slots and must not enter PV.
        k[table[:, -1].long(), sk % 64:] = float("nan")
        v[table[:, -1].long(), sk % 64:] = float("nan")
    return q, k, v, table, q_storage


def _check(q, k, v, table, sk, window, out, *, full=True):
    assert out.shape == q.shape and out.dtype == q.dtype
    assert torch.isfinite(out).all()
    sq = q.shape[1]
    if full:
        rows = torch.arange(sq, device=q.device)
    else:
        rows = torch.tensor(
            sorted(
                {x
                 for x in (0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 2047, 2048, sq - 2, sq - 1)
                 if 0 <= x < sq}), device=q.device)
    keys = torch.arange(sk, device=q.device)
    error_sq = torch.zeros((), device=q.device, dtype=torch.float64)
    reference_sq = torch.zeros_like(error_sq)
    for batch in range(q.shape[0]):
        kk = k[table[batch].long()].reshape(-1, k.shape[2], 64)[:sk].double()
        vv = v[table[batch].long()].reshape(-1, k.shape[2], 64)[:sk].double()
        for h in range(q.shape[2]):
            # Bound reference memory while checking every requested query row.
            for positions in rows.split(512):
                scores = q[batch, positions, h].double() @ kk[:, h // 8].T * 0.125
                end = positions[:, None] + sk - sq
                mask = keys[None, :] <= end
                if window:
                    mask &= keys[None, :] > end - window
                ref = scores.masked_fill(~mask, -float("inf")).softmax(-1) @ vv[:, h // 8]
                actual = out[batch, positions, h].double()
                torch.testing.assert_close(actual, ref, atol=0.025, rtol=0.025)
                error_sq += (actual - ref).square().sum()
                reference_sq += ref.square().sum()
    assert error_sq < 0.005**2 * reference_sq


@pytest.mark.parametrize("batch,sq,prefix,offset", [
    (1, 8, 4097, 0),
    (2, 63, 0, 1),
    (3, 65, 17, 0),
    (1, 255, 0, 0),
    (1, 256, 0, 0),
    (1, 512, 0, 0),
    (1, 768, 0, 0),
    (1, 769, 0, 0),
    (1, 2049, 63, 1),
    (1, 4096, 17, 0),
    (1, 4097, 96, 0),
    (2, 4097, 1, 0),
    (1, 8192, 32, 0),
    (1, 8193, 17, 0),
    (1, 8193, 32, 0),
    (1, 8193, 17, 1),
])
@pytest.mark.parametrize("window", [0, 2048])
def test_paged_prefill_boundaries(batch, sq, prefix, offset, window):
    sk = sq + prefix
    q, k, v, table, storage = _inputs(batch, sq, sk, offset)
    out = paged_prefill(q, k, v, table, kv_length=sk, window=window)
    _check(q, k, v, table, sk, window, out, full=sq <= 769)
    assert storage[-1] == 13
    if offset:
        assert storage[0] == 13


@pytest.mark.parametrize("kv_heads", [4, 16])
@pytest.mark.parametrize("window", [0, 2048])
@pytest.mark.parametrize("sq,sk,offset", [(65, 129, 1), (512, 512, 0), (769, 2817, 0), (2048, 8192, 0)])
def test_paged_prefill_head_counts(kv_heads, window, sq, sk, offset):
    q, k, v, table, storage = _inputs(2, sq, sk, offset, kv_heads)
    out = paged_prefill(q, k, v, table, kv_length=sk, window=window)
    _check(q, k, v, table, sk, window, out, full=sq <= 769)
    assert storage[-1] == 13
    if offset:
        assert storage[0] == 13
    # Replay the stable recovery path with the same head strides.
    q.fill_(1)
    k.fill_(16)
    out = paged_prefill(q, k, v, table, kv_length=sk, window=window)
    _check(q, k, v, table, sk, window, out, full=False)


@pytest.mark.parametrize("sq", [2048, 8192, 16384])
@pytest.mark.parametrize("window", [0, 2048])
def test_paged_prefill_full_rows_and_graph_replay(sq, window):
    q, k, v, table, _ = _inputs(1, sq, sq)
    expected = paged_prefill(q, k, v, table, window=window)
    _check(q, k, v, table, sq, window, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_prefill(q, k, v, table, window=window)
    for _ in range(20):
        out.fill_(float("nan"))
        graph.replay()
        assert torch.equal(expected, out)


@pytest.mark.parametrize("sq,prefix,window,offset", [
    (65, 0, 0, 0),
    (512, 0, 0, 0),
    (769, 0, 0, 0),
    (4097, 4097, 2048, 0),
    (8193, 17, 0, 0),
    (8193, 17, 2048, 1),
])
@pytest.mark.parametrize("mode", ["negative_extreme", "positive", "late_spike"])
def test_paged_prefill_recovery_recomputes_each_replay(sq, prefix, window, offset, mode):
    sk = sq + prefix
    q, k, v, table, _ = _inputs(1, sq, sk, offset)
    paged_prefill(q, k, v, table, kv_length=sk, window=window)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_prefill(q, k, v, table, kv_length=sk, window=window)
    for unsafe in (False, True, False):
        q.normal_()
        k.normal_()
        v.normal_()
        if unsafe and mode == "negative_extreme":
            q.fill_(2.0**50)
            k.fill_(-(2.0**50))
            v.fill_(1)
        elif unsafe and mode == "positive":
            q.fill_(1)
            k.fill_(16)
        elif unsafe:
            q.fill_(1)
            k.zero_()
            k[table[0, -1].long(), 0] = 32
        if sk % 64:
            k[table[:, -1].long(), sk % 64:] = float("nan")
            v[table[:, -1].long(), sk % 64:] = float("nan")
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
            if unsafe and mode == "negative_extreme":
                torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)
            else:
                _check(q, k, v, table, sk, window, out, full=False)


@pytest.mark.parametrize("bad", [
    "dtype", "shape", "strides", "table_dtype", "table_rank", "table_short", "kv_short", "kv_bool", "window",
    "window_float", "k_alignment", "cpu"
])
def test_paged_prefill_rejects_invalid_inputs(bad):
    q, k, v, table, _ = _inputs(1, 65, 65)
    kwargs = {}
    if bad == "dtype":
        q = q.float()
    elif bad == "shape":
        q = q[:, :, :8].contiguous()
    elif bad == "strides":
        q = torch.empty((*q.shape[:-1], 128), dtype=q.dtype, device=q.device)[..., ::2]
    elif bad == "table_dtype":
        table = table.long()
    elif bad == "table_rank":
        table = table.flatten()
    elif bad == "table_short":
        table = table[:, :1]
    elif bad == "kv_short":
        kwargs["kv_length"] = 64
    elif bad == "kv_bool":
        kwargs["kv_length"] = True
    elif bad == "window":
        kwargs["window"] = 1024
    elif bad == "window_float":
        kwargs["window"] = 0.0
    elif bad == "k_alignment":
        k = torch.empty(k.numel() + 1, dtype=k.dtype, device=k.device)[1:].view_as(k)
    else:
        q, k, v, table = (t.cpu() for t in (q, k, v, table))
    with pytest.raises(InvalidInput):
        paged_prefill(q, k, v, table, **kwargs)


def test_paged_prefill_dispatch_contract():
    q, k, v, table, _ = _inputs(1, 65, 65)
    q.requires_grad_(True)
    with pytest.raises(UnsupportedBackward):
        paged_prefill(q, k, v, table)
    with torch.no_grad():
        _check(q, k, v, table, 65, 0, paged_prefill(q, k, v, table, space="heuristic"))
        with pytest.raises(InvalidInput):
            paged_prefill(q, k, v, table, space="full")
        with pytest.raises(UnsupportedOp):
            paged_prefill(q, k, v, table, space=Space("heuristic", BitwiseInvarianceSpec({"SQ"})))


def test_paged_prefill_uses_tensor_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two gfx1250 GPUs")
    with torch.cuda.device(1):
        if not is_hip_gfx1250():
            pytest.skip("Requires gfx1250 on device 1")
        q, k, v, table, _ = _inputs(1, 65, 65)
    with torch.cuda.device(0):
        out = paged_prefill(q, k, v, table)
        assert torch.cuda.current_device() == 0
    with torch.cuda.device(1):
        _check(q, k, v, table, 65, 0, out)
