"""Public gfx1250 BF16 decode: causal paging, stable reduction and reusable graphs."""

import pytest
import torch
import triton
from triton._internal_testing import is_hip_gfx1250
from triton.tlx.ops import InvalidInput, UnsupportedOp, paged_attention_decode, paged_attention_decode_workspace
from triton.tlx.ops.kernels.paged_attn._testing import assert_close, make_inputs, pack_cache, reference

pytestmark = pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")


def _cache(inputs, packed):
    q, k, v, lengths, table = inputs
    if packed:
        k, v = pack_cache(k, v)
    return q, k, v, lengths, table


def _prepare(inputs, query_length=1, window=0, max_context_len=None, out=None):
    q, k, v, _, table = inputs
    workspace = paged_attention_decode_workspace(q, k, v, table, query_length=query_length, sliding_window=window,
                                                 max_context_len=max_context_len)
    out = torch.empty_like(q) if out is None else out
    kwargs = dict(query_length=query_length, sliding_window=window, max_context_len=max_context_len, out=out,
                  workspace=workspace)
    return out, workspace, lambda: paged_attention_decode(*inputs, **kwargs)


@pytest.mark.parametrize("query_length", [1, 7, 8])
@pytest.mark.parametrize("packed", [False, True], ids=["4d", "5d"])
@pytest.mark.parametrize("window", [0, 1, 63, 2048])
@pytest.mark.parametrize("kv_heads", [2, 4, 16])
def test_boundaries_and_poison(query_length, packed, window, kv_heads):
    lengths = [0, 1, 7, 63, 64, 65, 129, 2047, 2048, 2049, 8193]
    inputs = make_inputs(lengths, query_length, kv_heads=kv_heads)
    q, k, v, lens, table = inputs
    expected = reference(*inputs, query_length, window)
    # Poison both edges of a partially visible page and every unused table entry.
    for b, length in enumerate(lengths):
        first = max(0, length - query_length - window + 1) if window else 0
        if first % 64:
            k[table[b, first // 64], :, :first % 64] = float("nan")
            v[table[b, first // 64], :, :first % 64] = float("nan")
        if length % 64:
            k[table[b, length // 64], :, length % 64:] = float("nan")
            v[table[b, length // 64], :, length % 64:] = float("nan")
        table[b, :first // 64] = -1
        table[b, triton.cdiv(length, 64):] = -1
    storage = torch.full((q.numel() + 32, ), 37, dtype=q.dtype, device=q.device)
    out = storage[16:-16].view_as(q)
    out, workspace, run = _prepare(_cache(inputs, packed), query_length, window, out=out)
    out.fill_(float("nan"))
    for tensor in workspace or ():
        tensor.fill_(float("nan"))
    assert run() is out
    assert_close(out, expected)
    assert torch.all(storage[:16] == 37) and torch.all(storage[-16:] == 37)


@pytest.mark.parametrize("query_length", [1, 7, 8])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("case", ["zero", "peaky", "negative", "offset", "cancellation"])
def test_split_numerics(query_length, packed, case):
    inputs = make_inputs([65 if case == "negative" else 128], query_length)
    q, k, v, lens, table = inputs
    if case == "zero":
        q.zero_()
    elif case == "peaky":
        q.mul_(16)
        k.mul_(16)
    elif case == "negative":
        q.fill_(1)
        k.fill_(-1e30)
    elif case == "offset":
        q.fill_(16)
        k.add_(32)
    elif case == "cancellation":
        q.zero_()
        v[table[0, 0]].fill_(1)
        v[table[0, 0], :, 0].fill_(1.0078125)
        v[table[0, 1]].fill_(-1)
    expected = reference(*inputs, query_length)
    actual = paged_attention_decode(*_cache(inputs, packed), query_length=query_length)
    assert_close(actual, expected)
    if case == "cancellation":
        # BF16 split numerators would round this residual to zero.
        torch.testing.assert_close(actual[-1].double(), expected[-1], atol=0, rtol=0)


@pytest.mark.parametrize("query_length", [1, 7, 8])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("window", [0, 63, 2048, 4099])
@pytest.mark.parametrize("batch,kv_heads", [(12, 2), (16, 2), (65, 2), (32, 16), (33, 16)])
def test_graph_updates_and_independent_streams(query_length, packed, window, batch, kv_heads):
    # Exercise medium split counts, sequence rotation, and adjacent-head dispatch.
    # A full-context window must use the same plan as global attention.
    capacity = 4099
    inputs = make_inputs([capacity] * batch, query_length, kv_heads=kv_heads)
    q, k, v, lengths, table = inputs
    out, workspace, run = _prepare(_cache(inputs, packed), query_length, window, capacity)
    run()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    original_table = table.clone()
    generator = torch.Generator(device=q.device).manual_seed(81)
    for iteration in range(6):
        q.normal_(generator=generator)
        lengths.copy_(
            torch.tensor([0 if b == iteration else max(0, capacity - b * 113 - iteration * 67) for b in range(batch)],
                         dtype=torch.int32, device=q.device))
        table.copy_(original_table.roll(iteration + 1, 0))
        expected = reference(q, k, v, lengths, table, query_length, window)
        unused = torch.arange(table.shape[1], device=q.device)[None, :] >= (lengths[:, None] + 63) // 64
        table.masked_fill_(unused, -1)
        out.fill_(float("nan"))
        for tensor in workspace or ():
            tensor.fill_(float("nan"))
        graph.replay()
        assert_close(out, expected)
    # Independent workspace/output storage permits concurrent invocations.
    out2, _, run2 = _prepare(_cache(inputs, packed), query_length, window, capacity)
    run2()
    torch.cuda.synchronize()
    for _ in range(10):
        graph.replay()
        with torch.cuda.stream(stream):
            run2()
    torch.cuda.synchronize()
    assert_close(out, expected)
    assert_close(out2, expected)


@pytest.mark.parametrize("query_length", [1, 7, 8])
@pytest.mark.parametrize("scale", [0.0625, 0.5])
def test_scale_and_no_grad(query_length, scale):
    inputs = make_inputs([129, 4099], query_length)
    inputs[0].requires_grad_()
    with pytest.raises(UnsupportedOp):
        paged_attention_decode(*inputs, query_length=query_length)
    with torch.no_grad():
        actual = paged_attention_decode(*inputs, query_length=query_length, sm_scale=scale)
        assert_close(actual, reference(*inputs, query_length, scale=scale))


def test_invalid_inputs_and_workspace():
    inputs = make_inputs([129, 129], 1)
    q, k, v, lengths, table = inputs
    other_heads = make_inputs([129, 129], 1, kv_heads=4)
    other_workspace = paged_attention_decode_workspace(*other_heads[:3], other_heads[4])
    with pytest.raises(InvalidInput, match="workspace"):
        paged_attention_decode(*inputs, workspace=other_workspace)
    with pytest.raises(InvalidInput):
        paged_attention_decode(other_heads[0], k, v, lengths, table)
    for qlen in (0, True, 1.5):
        with pytest.raises(InvalidInput):
            paged_attention_decode(*inputs, query_length=qlen)
        with pytest.raises(InvalidInput):
            paged_attention_decode_workspace(q, k, v, table, query_length=qlen)
    for kw in (dict(sliding_window=-1), dict(sliding_window=True), dict(max_context_len=193), dict(max_context_len=-1),
               dict(max_context_len=True), dict(sm_scale=float("nan")), dict(out=q),
               dict(out=q.transpose(1, 2).contiguous().transpose(1, 2))):
        with pytest.raises(InvalidInput):
            paged_attention_decode(*inputs, **kw)
    with pytest.raises(InvalidInput):
        paged_attention_decode(q, k, v, lengths[:, None], table)
    with pytest.raises(InvalidInput):
        paged_attention_decode(q, k, v, lengths, table.long())
    with pytest.raises(InvalidInput):
        paged_attention_decode(q, k.transpose(2, 3), v, lengths, table)
    unaligned = torch.empty(k.numel() + 1, device=k.device, dtype=k.dtype)[1:].view_as(k)
    with pytest.raises(InvalidInput, match="aligned"):
        paged_attention_decode(q, unaligned, v, lengths, table)
    with pytest.raises(InvalidInput):
        paged_attention_decode(q.half(), k.half(), v.half(), lengths, table)
    workspace = paged_attention_decode_workspace(q, k, v, table)
    for bad in ((workspace[0], ), (workspace[0].half(), workspace[1]), (workspace[0], workspace[1][..., :1])):
        with pytest.raises(InvalidInput, match="workspace"):
            paged_attention_decode(*inputs, workspace=bad)
    # Correct shapes sharing backing storage must fail before a GPU write.
    shared = torch.empty(workspace[0].numel(), dtype=torch.float32, device=q.device)
    bad = (shared.view_as(workspace[0]), shared[:workspace[1].numel()].view_as(workspace[1]))
    with pytest.raises(InvalidInput, match="storage"):
        paged_attention_decode(*inputs, workspace=bad)
    alias_out = shared.view(torch.bfloat16)[:q.numel()].view_as(q)
    with pytest.raises(InvalidInput, match="storage"):
        paged_attention_decode(*inputs, out=alias_out, workspace=(bad[0], workspace[1]))


@pytest.mark.parametrize("query_length", [1, 7, 8])
@pytest.mark.parametrize("packed", [False, True])
def test_single_partition_and_empty_context(query_length, packed):
    inputs = _cache(make_inputs([0, 1, 63], query_length), packed)
    out, workspace, run = _prepare(inputs, query_length)
    assert workspace is None
    run()
    assert torch.isfinite(out).all() and torch.count_nonzero(out[:query_length]) == 0
    inputs[3].zero_()
    out, workspace, run = _prepare(inputs, query_length, max_context_len=0)
    assert workspace is None
    run()
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("query_length", [1, 7, 8])
@pytest.mark.parametrize("packed", [False, True])
def test_int32_context_boundary(query_length, packed):
    limit = 2**31 - 32768
    inputs = make_inputs([4096], query_length)
    expected = reference(*inputs, query_length, 2048)
    q, k, v, _, table = _cache(inputs, packed)
    large_table = torch.full((1, limit // 64 + 1), -1, dtype=torch.int32, device=q.device)
    large_table[:, limit // 64 - 33:limit // 64] = table[:, -33:]
    lengths = torch.tensor([limit], device=q.device, dtype=torch.int32)
    actual = paged_attention_decode(q, k, v, lengths, large_table, query_length=query_length, sliding_window=2048,
                                    max_context_len=limit)
    assert_close(actual, expected)
    with pytest.raises(InvalidInput, match="max_context_len"):
        paged_attention_decode(q, k, v, lengths, large_table, query_length=query_length, sliding_window=2048,
                               max_context_len=limit + 1)


@pytest.mark.parametrize("packed", [False, True])
def test_cache_address_beyond_4gib(packed):
    # Physical addressing must not wrap at 32-bit byte offsets.
    free, _ = torch.cuda.mem_get_info()
    if free < 11 * 1024**3:
        pytest.skip("requires 11 GiB free GPU memory")
    inputs = make_inputs([64], 8)
    expected = reference(*inputs, 8)
    q, k, v, lengths, table = _cache(inputs, packed)
    physical = 262144
    big_k = torch.empty((physical + 1, *k.shape[1:]), device=k.device, dtype=k.dtype)
    big_v = torch.empty_like(big_k)
    big_k[physical] = k[table[0, 0]]
    big_v[physical] = v[table[0, 0]]
    table.fill_(physical)
    actual = paged_attention_decode(q, big_k, big_v, lengths, table, query_length=8)
    assert_close(actual, expected)
