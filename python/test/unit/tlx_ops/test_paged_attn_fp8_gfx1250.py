"""Public-operator correctness for gfx1250 per-head E4M3FN paged decode."""
import pytest
import torch
import triton
from triton.tlx.ops import (paged_attention_decode_fp8, paged_attention_decode_fp8_workspace, InvalidInput,
                            UnsupportedOp, UnsupportedBackward, Space, BitwiseInvarianceSpec)


def _gfx1250_available():
    return bool(torch.cuda.is_available() and torch.version.hip
                and (torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx1250'))


def _make_test_inputs(lengths, hq=16, hkv=2, dim=64, page=64, query_length=1, fp8_q=False, seed=182,
                      max_context_len=None):
    """Quantize with distinct per-head scales, shuffle pages, poison unused slots."""
    torch.manual_seed(seed)
    batch = len(lengths)
    max_len = max(1, max(lengths))
    pages = triton.cdiv(max_len, page)
    blocks = batch * pages + 3
    ks = torch.linspace(0.025, 0.08, hkv, device='cuda')
    vs = torch.linspace(0.04, 0.12, hkv, device='cuda').flip(0).contiguous()
    qscale = torch.linspace(0.02, 0.06, hq, device='cuda') if fp8_q else None
    q = torch.randn(batch * query_length, hq, dim, device='cuda') * 0.4
    q = (q / qscale[None, :, None]).to(torch.float8_e4m3fn) if fp8_q else q.to(torch.bfloat16)
    k = torch.randn(blocks, hkv, page, dim, device='cuda') * 0.5
    v = torch.randn_like(k) * 0.8
    table = torch.randperm(blocks, device='cuda')[:batch * pages].reshape(batch, pages).to(torch.int32)
    offsets = torch.arange(pages * page, device='cuda').reshape(1, pages, page)
    invalid = offsets >= torch.tensor(lengths, device='cuda')[:, None, None]
    invalid = invalid.reshape(batch * pages, 1, page, 1)
    physical_pages = table.long().flatten()
    k[physical_pages] = k[physical_pages].masked_fill(invalid, float('nan'))
    v[physical_pages] = v[physical_pages].masked_fill(invalid, float('nan'))
    k = (k / ks[None, :, None, None]).to(torch.float8_e4m3fn)
    v = (v / vs[None, :, None, None]).to(torch.float8_e4m3fn)
    kp = k.reshape(blocks, hkv, page, dim // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    vp = v.reshape(blocks, hkv, page // 16, 16, dim).permute(0, 1, 2, 4, 3).contiguous()
    contexts = torch.tensor(lengths, device='cuda', dtype=torch.int32)
    if max_context_len is not None:
        # Capacity is metadata, not allocated/active KV. Unreferenced entries
        # must never be dereferenced, even with many empty split partitions.
        capacity = torch.full((batch, triton.cdiv(max_context_len, page)), -1, device='cuda', dtype=table.dtype)
        for seq, length in enumerate(lengths):
            active_pages = triton.cdiv(length, page)
            capacity[seq, :active_pages] = table[seq, :active_pages]
        table = capacity
    return (q, kp, vp, contexts, table, ks, vs, qscale, max_len, k, v)


def _reference(q, k, v, table, lengths, ks, vs, qs, query_length, causal, window_size):
    """CPU float64 attention over quantized inputs, returned as device FP32.

    Keep the oracle independent of GPU GEMM/softmax implementations. Captured
    failures of the former GPU FP32 reference disagreed with CPU float64 even
    when the attention kernel satisfied the unchanged numerical tolerance.
    """
    output_device = q.device
    q = q.detach().cpu().double()
    k, v = k.detach().cpu().double(), v.detach().cpu().double()
    table = table.detach().cpu()
    lengths = lengths.detach().cpu().tolist() if isinstance(lengths, torch.Tensor) else lengths
    ks, vs = ks.detach().cpu().double(), vs.detach().cpu().double()
    if qs is not None:
        q *= qs.detach().cpu().double()[None, :, None]
    hq, dim = q.shape[1:]
    hkv, page = k.shape[1:3]
    out = torch.zeros_like(q)
    for seq, length in enumerate(lengths):
        if not length:
            continue
        blocks = table[seq, :triton.cdiv(length, page)].long()
        kd = k[blocks].permute(1, 0, 2, 3).reshape(hkv, -1, dim)[:, :length]
        vd = v[blocks].permute(1, 0, 2, 3).reshape(hkv, -1, dim)[:, :length]
        kd *= ks[:, None, None]
        vd *= vs[:, None, None]
        mapping = torch.arange(hq, device=q.device) // (hq // hkv)
        kd, vd = (kd[mapping], vd[mapping])
        qr = q[seq * query_length:(seq + 1) * query_length].transpose(0, 1)
        scores = torch.matmul(qr, kd.transpose(1, 2)) * dim**(-0.5)
        positions = torch.arange(length, device=q.device)
        qpos = length - query_length + torch.arange(query_length, device=q.device)
        mask = torch.ones(query_length, length, device=q.device, dtype=torch.bool)
        if causal:
            mask &= positions[None, :] <= qpos[:, None]
        if window_size:
            mask &= positions[None, :] > qpos[:, None] - window_size
        probabilities = torch.softmax(scores.masked_fill(~mask[None], -float('inf')), -1).nan_to_num()
        out[seq * query_length:(seq + 1) * query_length] = torch.matmul(probabilities, vd).transpose(0, 1)
    return out.to(device=output_device, dtype=torch.float32)


@pytest.mark.parametrize('fp8_q', [False, True])
@pytest.mark.parametrize('causal,window,values', [(True, 0, [[2., 0.], [3., 8.]]), (True, 2, [[2., 0.], [4., 16.]]),
                                                  (False, 0, [[3., 8.], [3., 8.]])])
def test_fp8_reference_uniform_attention(fp8_q, causal, window, values):
    # Zero logits give exact uniform means; this also checks page mapping,
    # per-head descales, GQA sharing, causal/window masks and poisoned tails.
    q = torch.zeros((2, 4, 64), dtype=torch.float8_e4m3fn if fp8_q else torch.bfloat16)
    k = torch.full((2, 2, 16, 64), float('nan')).to(torch.float8_e4m3fn)
    v = k.clone()
    k[1, :, :3] = 0.
    v[1, 0, :3] = torch.tensor([2., 6., 10.])[:, None].expand(3, 64).to(v.dtype)
    v[1, 1, :3] = torch.tensor([-4., 4., 12.])[:, None].expand(3, 64).to(v.dtype)
    ks = torch.tensor([2., 4.])
    vs = torch.tensor([.5, 2.])
    qs = torch.tensor([1., 2., 4., 8.]) if fp8_q else None
    actual = _reference(q, k, v, torch.tensor([[1]]), [3], ks, vs, qs, 2, causal, window)
    expected = torch.tensor(values).repeat_interleave(2, dim=1)[:, :, None].expand(2, 4, 64)
    torch.testing.assert_close(actual, expected, atol=0., rtol=0.)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('fp8_q', [False, True], ids=['bf16-q', 'fp8-q'])
@pytest.mark.parametrize('dim,page,hq,hkv,qlen,lengths,causal,window,splits',
                         [(64, 64, 16, 2, 1, [2304], True, 0, 1), (64, 64, 16, 2, 1, [2304], True, 0, 4),
                          (64, 64, 16, 2, 1, [67, 8193], True, 0, 384), (64, 64, 128, 128, 1, [2304], True, 0, 4),
                          (128, 32, 4, 4, 1, [0, 1, 31, 97], True, 0, 8),
                          (128, 128, 8, 2, 1, [0, 67, 8193], True, 0, 384),
                          (128, 64, 64, 64, 1, [0, 67, 2304], True, 0, 12),
                          (128, 128, 64, 64, 1, [0, 67, 2304], True, 0, 6),
                          (128, 64, 64, 64, 1, [0, 67, 257], True, 128, 1),
                          (64, 128, 16, 1, 1, [257, 2303], True, 127, 4), (64, 64, 6, 2, 8, [1, 113], True, 37, 4),
                          (64, 16, 4, 2, 32, [17], False, 0, 3), (64, 64, 4, 2, 1, [8192, 32768], True, 0, 8),
                          (64, 64, 2, 1, 1, [131072, 262144], True, 0, 32)])
def test_fp8_kv_decode(fp8_q, dim, page, hq, hkv, qlen, lengths, causal, window, splits):
    q, kp, vp, contexts, table, ks, vs, qs, max_len, k, v = _make_test_inputs(lengths, hq, hkv, dim, page, qlen, fp8_q)
    actual = paged_attention_decode_fp8(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len, q_scale=qs,
                                        query_length=qlen, causal=causal, sliding_window=window, num_splits=splits)
    expected = _reference(q, k, v, table, lengths, ks, vs, qs, qlen, causal, window)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=0.0005, rtol=0.01)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('fp8_q', [False, True], ids=['bf16-q', 'fp8-q'])
@pytest.mark.parametrize('hq,hkv,dim,page,lengths', [(16, 1, 64, 64, [1024]), (32, 8, 64, 64, [2304]),
                                                     (64, 8, 128, 64, [8192]), (64, 64, 64, 64, [4096]),
                                                     (32, 32, 64, 128, [8192]), (64, 64, 64, 128, [8192]),
                                                     (128, 32, 128, 64, [8192]), (128, 128, 128, 128, [16384]),
                                                     (128, 128, 64, 64, [0, 67, 513, 1024]),
                                                     (16, 2, 128, 64, [1, 128, 511, 1024]), (16, 2, 64, 64, [2304] * 8),
                                                     (64, 64, 128, 64, [8192] * 8), (16, 2, 128, 128, [2304]),
                                                     (16, 2, 128, 128, [8192]), (16, 2, 128, 64, [4096]),
                                                     (16, 2, 128, 64, [8192]), (32, 4, 128, 64, [8192]),
                                                     (16, 2, 128, 64, [8193]), (16, 2, 64, 64, [8192]),
                                                     (16, 2, 64, 64, [32768]), (32, 4, 64, 128, [32768]),
                                                     (16, 2, 64, 64, [8192, 32768])])
def test_fp8_kv_decode_auto_geometry(fp8_q, hq, hkv, dim, page, lengths):
    q, kp, vp, contexts, table, ks, vs, qs, max_len, k, v = _make_test_inputs(lengths, hq=hq, hkv=hkv, dim=dim,
                                                                              page=page, fp8_q=fp8_q)
    actual = paged_attention_decode_fp8(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len, q_scale=qs)
    expected = _reference(q, k, v, table, contexts, ks, vs, qs, 1, True, 0)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=0.0005, rtol=0.01)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
def test_fp8_kv_decode_validation():
    q, kp, vp, contexts, table, ks, vs, qs, max_len, _, _ = _make_test_inputs([33], fp8_q=True)
    with pytest.raises(ValueError, match='q_scale'):
        paged_attention_decode_fp8(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len)
    with pytest.raises(ValueError, match='k_scale'):
        paged_attention_decode_fp8(q, kp, vp, contexts, table, ks[:1], vs, max_context_len=max_len, q_scale=qs)
    with pytest.raises(ValueError, match='sliding windows'):
        paged_attention_decode_fp8(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len, q_scale=qs,
                                   causal=False, sliding_window=16)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('tokens', [8, 32, 128, 512, 2048, 8192])
def test_fp8_kv_decode_query_sweep(tokens):
    lengths = [2304]
    q, kp, vp, contexts, table, ks, vs, _, max_len, k, v = _make_test_inputs(lengths, hq=16, hkv=2, query_length=tokens)
    actual = paged_attention_decode_fp8(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len,
                                        query_length=tokens, causal=False, num_splits=1)
    for first in range(0, tokens, 128):
        qr = q[first:first + 128]
        expected = _reference(qr, k, v, table, lengths, ks, vs, None, qr.shape[0], False, 0)
        torch.testing.assert_close(actual[first:first + 128].float(), expected, atol=0.0005, rtol=0.01)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('fp8_q', [False, True])
@pytest.mark.parametrize('splits', [4, 8, 12, 384])
@pytest.mark.parametrize('seed', [0, 19, 182])
@pytest.mark.parametrize('dim', [64, 128])
def test_fp8_kv_decode_peaked_and_workspace(fp8_q, splits, seed, dim):
    q, kp, vp, contexts, table, ks, vs, qs, max_len, k, v = _make_test_inputs([0, 67, 257], hq=16, hkv=2, dim=dim,
                                                                              query_length=8, fp8_q=fp8_q, seed=seed)
    ks = ks * torch.tensor([1.0, 64.0], device=q.device)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    workspace = (torch.empty((*q.shape[:2], splits, q.shape[2]),
                             device=q.device), torch.empty((*q.shape[:2], splits), device=q.device))
    actual = paged_attention_decode_fp8(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len, q_scale=qs,
                                        query_length=8, sliding_window=37, num_splits=splits, out=out,
                                        workspace=workspace)
    expected = _reference(q, k, v, table, [0, 67, 257], ks, vs, qs, 8, True, 37)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=0.0005, rtol=0.01)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('fp8_q', [False, True], ids=['bf16-q', 'fp8-q'])
@pytest.mark.parametrize('dim', [64, 128], ids=['D64', 'D128'])
@pytest.mark.parametrize('page', [64, 128], ids=['P64', 'P128'])
@pytest.mark.parametrize('splits', [None, 4, 8, 12], ids=['auto', 'split4', 'split8', 'split12'])
@pytest.mark.parametrize('workspace_mode', ['allocated', 'provided', 'graph'])
def test_fp8_kv_decode_partial_cancellation(fp8_q, dim, page, splits, workspace_mode):
    # Q=K=0 gives uniform attention. The first two pages sum to one;
    # FP16 partition means round away that residual before the FP32 merge.
    length, hq, hkv = 8 * page, 16, 2
    x = _make_test_inputs([length], hq=hq, hkv=hkv, dim=dim, page=page, fp8_q=fp8_q)
    q, _, _, contexts, table, _, _, qs, _, k, v = x
    q = torch.zeros(q.shape, device=q.device, dtype=q.dtype)
    k = torch.zeros(k.shape, device=q.device).to(torch.float8_e4m3fn)
    vf = torch.zeros(v.shape, device=q.device)
    first, second = table[0, :2].tolist()
    vf[first, :, :page - 1, 0] = 256.
    vf[first, :, page - 1, 0] = 1.
    vf[second, :, :page - 1, 0] = -256.
    v = vf.to(torch.float8_e4m3fn)
    blocks = k.shape[0]
    kp = k.reshape(blocks, hkv, page, dim // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    vp = v.reshape(blocks, hkv, page // 16, 16, dim).permute(0, 1, 2, 4, 3).contiguous()
    ks = torch.tensor([1., 2.], device=q.device)
    vs = torch.tensor([1., 8.], device=q.device)
    plan = dict(max_context_len=length, num_splits=splits)
    workspace = None if workspace_mode == 'allocated' else paged_attention_decode_fp8_workspace(
        q, kp, vp, table, **plan)
    out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    expected = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    expected[0, :, 0] = vs.repeat_interleave(hq // hkv) / length

    def call():
        return paged_attention_decode_fp8(q, kp, vp, contexts, table, ks, vs, q_scale=qs, out=out, workspace=workspace,
                                          **plan)

    call()
    if workspace_mode == 'graph':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        # Reused scratch must preserve the residual and reset empty rows.
        for active in [length, 0, length]:
            contexts.fill_(active)
            graph.replay()
            reference = expected if active else torch.zeros_like(expected)
            torch.testing.assert_close(out.float(), reference, atol=5e-4, rtol=1e-2)
    else:
        torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-2)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('dim', [64, 128])
@pytest.mark.parametrize('page', [64, 128])
@pytest.mark.parametrize('fp8_q', [False, True])
def test_fp8_kv_decode_tiny_probability_mass(dim, page, fp8_q):
    q, _, _, lengths, table, _, _, _, max_len, k, _ = _make_test_inputs([128], hq=16, hkv=2, dim=dim, page=page,
                                                                        fp8_q=fp8_q)
    qf = torch.zeros(q.shape, device=q.device)
    qf[..., 0] = 1.0
    q = qf.to(q.dtype)
    kf = torch.zeros(k.shape, device=q.device)
    kf[..., 0] = -144.0 if dim == 64 else -192.0
    vf = torch.full(k.shape, 448.0, device=q.device)
    first_page = table[0, 0].long()
    kf[first_page, :, 0, 0] = 0.0
    vf[first_page, :, 0, :] = 0.0
    k = kf.to(torch.float8_e4m3fn)
    v = vf.to(torch.float8_e4m3fn)
    blocks = k.shape[0]
    kp = k.reshape(blocks, 2, page, dim // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    vp = v.reshape(blocks, 2, page // 16, 16, dim).permute(0, 1, 2, 4, 3).contiguous()
    ks = torch.tensor([1.0, 1.125], device=q.device)
    vs = torch.tensor([1.0, 8.0], device=q.device)
    qs = torch.ones(16, device=q.device) if fp8_q else None
    actual = paged_attention_decode_fp8(q, kp, vp, lengths, table, ks, vs, max_context_len=max_len, q_scale=qs,
                                        num_splits=1)
    expected = _reference(q, k, v, table, [128], ks, vs, qs, 1, True, 0)
    assert float(expected.abs().max()) > 0.0007
    torch.testing.assert_close(actual.float(), expected, atol=0.0005, rtol=0.01)


def _cpu_inputs(fp8_q=False):
    query = torch.empty((1, 16, 64), dtype=torch.float8_e4m3fn if fp8_q else torch.bfloat16)
    key = torch.empty((2, 2, 4, 64, 16), dtype=torch.float8_e4m3fn)
    value = torch.empty((2, 2, 4, 64, 16), dtype=torch.float8_e4m3fn)
    lengths = torch.tensor([64], dtype=torch.int32)
    table = torch.tensor([[0]], dtype=torch.int32)
    return query, key, value, lengths, table, torch.ones(2), torch.ones(2)


@pytest.mark.parametrize('change,match', [
    (lambda x: {'query_length': 0}, 'query_length'),
    (lambda x: {'query_length': True}, 'query_length'),
    (lambda x: {'max_context_len': 0}, 'max_context_len'),
    (lambda x: {'max_context_len': 65}, 'block table'),
    (lambda x: {'sliding_window': -1}, 'sliding_window'),
    (lambda x: {'sliding_window': True}, 'sliding_window'),
    (lambda x: {'sliding_window': 16, 'causal': False}, 'causal'),
    (lambda x: {'num_splits': 0}, 'num_splits'),
    (lambda x: {'num_splits': 1025}, 'num_splits'),
    (lambda x: {'num_splits': True}, 'num_splits'),
    (lambda x: {'k_scale': torch.ones(1)}, 'k_scale'),
    (lambda x: {'v_scale': torch.ones(2, dtype=torch.float64)}, 'v_scale'),
    (lambda x: {'q_scale': torch.ones(16)}, 'q_scale'),
    (lambda x: {'key_cache': x[1].to(torch.bfloat16)}, 'E4M3FN'),
    (lambda x: {'context_lens': torch.tensor([1.0])}, 'context_lens'),
    (lambda x: {
        'query': x[0].repeat(2, 1, 1), 'block_tables': x[4].repeat(2, 1), 'context_lens': torch.ones(
            4, dtype=torch.int32)[::2]
    }, 'context_lens'),
    (lambda x: {'sm_scale': float('inf')}, 'sm_scale'),
    (lambda x: {'sm_scale': -1.0}, 'sm_scale'),
    (lambda x: {'out': torch.empty_like(x[0], dtype=torch.float32)}, 'out'),
    (lambda x: {'workspace': (torch.empty(1), )}, 'workspace'),
    (lambda x: {'workspace': (torch.empty(1), torch.empty(1))}, 'workspace'),
    (lambda x: {'space': 'full'}, 'space'),
    (lambda x: {'space': Space('heuristic', BitwiseInvarianceSpec.dims('D'))}, 'bitwise'),
])
def test_fp8_api_rejects_invalid_metadata_before_dispatch(change, match, monkeypatch):
    import triton.tlx.ops as ops

    def unexpected(*args, **kwargs):
        raise AssertionError('invalid metadata reached device dispatch')

    monkeypatch.setattr(ops, 'impl_for', unexpected)
    x = _cpu_inputs()
    kwargs = dict(zip(('query', 'key_cache', 'value_cache', 'context_lens', 'block_tables', 'k_scale', 'v_scale'), x))
    kwargs['max_context_len'] = 64
    kwargs.update(change(x))
    with pytest.raises((InvalidInput, UnsupportedOp), match=match):
        paged_attention_decode_fp8(**kwargs)


def test_fp8_api_requires_fp8_query_scale():
    with pytest.raises(InvalidInput, match='q_scale'):
        paged_attention_decode_fp8(*_cpu_inputs(True), max_context_len=64)


def test_fp8_api_dispatch_and_forward_only(monkeypatch):
    from contextlib import nullcontext
    import triton.tlx.ops as ops
    from triton.tlx.ops._catalog import OpSpec
    calls = []
    sentinel = object()

    def implementation(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    def dispatch(op, *, device):
        assert device == torch.device('cpu')
        assert op == 'paged_attention_decode_fp8'
        return implementation, OpSpec(op, 'gfx1250', 'per_head_wmma_tdm', 'unused')

    monkeypatch.setattr(ops, 'impl_for', dispatch)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    x = _cpu_inputs()
    x[0].requires_grad_(True)
    with pytest.raises(UnsupportedBackward):
        paged_attention_decode_fp8(*x, max_context_len=64)
    assert calls == []
    with torch.no_grad():
        assert paged_attention_decode_fp8(*x, max_context_len=64, space='heuristic') is sentinel
    assert calls[0][1]['sm_scale'] == 64**-.5
    assert calls[0][1]['num_splits'] is None


@pytest.mark.parametrize('splits', [1, 4, 8])
def test_fp8_workspace_public_planning_without_device_launch(splits, monkeypatch):
    from contextlib import nullcontext
    import triton.tlx.ops as ops
    from triton.tlx.ops._catalog import impl_for
    implementation, spec = impl_for('paged_attention_decode_fp8_workspace', arch='gfx1250')
    monkeypatch.setattr(ops, 'impl_for', lambda op, *, device: (implementation, spec))
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    q, k, v, _, table, _, _ = _cpu_inputs()
    buffers = paged_attention_decode_fp8_workspace(q, k, v, table, max_context_len=64, num_splits=splits)
    if splits == 1:
        assert buffers is None
    else:
        assert buffers[0].shape == (1, 16, splits, 64)
        assert buffers[1].shape == (1, 16, splits)
        assert all(t.dtype == torch.float32 and t.device == q.device for t in buffers)


def test_fp8_catalog_is_gfx1250_only():
    from triton.tlx.ops._catalog import CATALOG, impl_for
    for op in ['paged_attention_decode_fp8', 'paged_attention_decode_fp8_workspace']:
        assert [s.arch for s in CATALOG if s.op == op] == ['gfx1250']
        with pytest.raises(UnsupportedOp):
            impl_for(op, arch='gfx950')


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('splits', [1, 8])
@pytest.mark.parametrize('dim', [64, 128])
@pytest.mark.parametrize('window', [0, 37, 128])
@pytest.mark.parametrize('page', [64, 128])
def test_fp8_public_workspace_graph_replay(splits, dim, window, page):
    x = _make_test_inputs([67, 2304], dim=dim, page=page)
    q, k, v, lengths, table, ks, vs, qs, bound, kd, vd = x
    work = paged_attention_decode_fp8_workspace(q, k, v, table, max_context_len=bound, num_splits=splits,
                                                sliding_window=window)
    out = torch.empty_like(q, dtype=torch.bfloat16)

    def call():
        return paged_attention_decode_fp8(q, k, v, lengths, table, ks, vs, max_context_len=bound, num_splits=splits,
                                          sliding_window=window, out=out, workspace=work)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        graph.replay()
    expected = _reference(q, kd, vd, table, [67, 2304], ks, vs, qs, 1, True, window)
    torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-2)
    for replay_lengths in ([64, 2303], [0, 128]):
        lengths.copy_(torch.tensor(replay_lengths, device=lengths.device, dtype=lengths.dtype))
        graph.replay()
        expected = _reference(q, kd, vd, table, replay_lengths, ks, vs, qs, 1, True, window)
        torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-2)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('fp8_q', [False, True], ids=['bf16-q', 'fp8-q'])
@pytest.mark.parametrize('bound', [8192, 262144])
@pytest.mark.parametrize('hq,hkv,dim,page', [(16, 2, 64, 64), (16, 2, 128, 128), (32, 32, 64, 128), (32, 32, 128, 64),
                                             (6, 2, 64, 64)])
def test_fp8_kv_decode_ragged_large_bound_graph(fp8_q, bound, hq, hkv, dim, page):
    # A fixed graph bound can greatly exceed all actual lengths. Most requests
    # are short, one crosses several pages, and some become empty on replay.
    original = [0, 1, 63, 64, 65, 127, 128, 129] + [67] * 23 + [513]
    x = _make_test_inputs(original, hq=hq, hkv=hkv, dim=dim, page=page, fp8_q=fp8_q, max_context_len=bound)
    q, k, v, lengths, table, ks, vs, qs, _, kd, vd = x
    work = paged_attention_decode_fp8_workspace(q, k, v, table, max_context_len=bound)
    out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)

    def call():
        return paged_attention_decode_fp8(q, k, v, lengths, table, ks, vs, q_scale=qs, max_context_len=bound, out=out,
                                          workspace=work)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    # Keep graph, block table, workspace and context addresses fixed. Poisoned
    # scratch and NaN cache tails expose stale merges and reads past active KV.
    for active in [original, [min(n, 17) for n in original], [0] * len(original), original]:
        lengths.copy_(torch.tensor(active, device=lengths.device, dtype=lengths.dtype))
        out.fill_(float('nan'))
        if work is not None:
            for buffer in work:
                buffer.fill_(float('nan'))
        graph.replay()
        expected = _reference(q, kd, vd, table, active, ks, vs, qs, 1, True, 0)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-2)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('fp8_q', [False, True])
@pytest.mark.parametrize('dim', [64, 128])
@pytest.mark.parametrize('page', [64, 128])
@pytest.mark.parametrize('metadata_dtype', [torch.int32, torch.int64])
def test_fp8_kv_decode_batched_partial_cancellation(fp8_q, dim, page, metadata_dtype):
    # Positive and negative active partitions sum to one. Rounded FP16 partition
    # means would lose that residual. Other requests remain empty.
    length, batch, hq, hkv = 1024, 32, 16, 2
    x = _make_test_inputs([length] + [0] * (batch - 1), dim=dim, page=page, fp8_q=fp8_q, max_context_len=262144)
    q, _, _, contexts, table, _, _, qs, _, k, v = x
    q = torch.zeros_like(q)
    k = torch.zeros(k.shape, device=q.device).to(torch.float8_e4m3fn)
    vf = torch.zeros(v.shape, device=q.device)
    logical = torch.zeros((length, hkv, dim), device=q.device)
    logical[:511, :, 0] = 256.
    logical[511, :, 0] = 1.
    logical[512:1023, :, 0] = -256.
    ids = table[0, :length // page].long()
    vf[ids] = logical.reshape(-1, page, hkv, dim).permute(0, 2, 1, 3)
    v = vf.to(torch.float8_e4m3fn)
    blocks = k.shape[0]
    kp = k.reshape(blocks, hkv, page, dim // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    vp = v.reshape(blocks, hkv, page // 16, 16, dim).permute(0, 1, 2, 4, 3).contiguous()
    contexts, table = contexts.to(metadata_dtype), table.to(metadata_dtype)
    ks = torch.tensor([1., 2.], device=q.device)
    vs = torch.tensor([1., 8.], device=q.device)
    plan = dict(max_context_len=262144)
    work = paged_attention_decode_fp8_workspace(q, kp, vp, table, **plan)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)

    def call():
        paged_attention_decode_fp8(q, kp, vp, contexts, table, ks, vs, q_scale=qs, out=out, workspace=work, **plan)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for active in [length, 128, 0, length]:
        contexts[0] = active
        out.fill_(float('nan'))
        if work is not None:
            for buffer in work:
                buffer.fill_(float('nan'))
        graph.replay()
        expected = torch.zeros(out.shape, device=q.device, dtype=torch.float32)
        if active:
            expected[0, :, 0] = vs.repeat_interleave(hq // hkv) * (1. / length if active == length else 256.)
        torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-2)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('fp8_q', [False, True])
@pytest.mark.parametrize('dim', [64, 128])
@pytest.mark.parametrize('page', [64, 128])
def test_fp8_kv_decode_batched_work_replay(fp8_q, dim, page):
    # Exercise work reuse across requests and pages. Zero logits give exact
    # uniform-attention means over distinct quantized per-sequence/page labels.
    batch, hq, hkv, length, bound = 65, 16, 2, 32768, 262144
    pages = length // page
    blocks = batch * pages
    q = torch.zeros((batch, hq, dim * (2 if dim == 128 else 1)), device='cuda',
                    dtype=torch.float8_e4m3fn if fp8_q else torch.bfloat16)
    if dim == 128:
        q = q[..., ::2]
    k = torch.zeros((blocks, hkv, dim // 16, page, 16), device='cuda').to(torch.float8_e4m3fn)
    dtype = torch.int64 if fp8_q else torch.int32
    table = torch.full((batch, bound // page), -1, device='cuda', dtype=dtype)
    table[:, :pages] = torch.randperm(blocks, device='cuda').reshape(batch, pages).to(dtype)
    labels = (1 + torch.arange(batch, device='cuda')[:, None] % 17 +
              torch.arange(pages, device='cuda')[None, :] % 7).to(torch.float8_e4m3fn).float()
    ids = table[:, :pages].long().flatten()
    vf = torch.empty((blocks, hkv, page // 16, dim, 16), device='cuda')
    vf[ids, 0] = 2. * labels.flatten()[:, None, None, None]
    vf[ids, 1] = 4. * labels.flatten()[:, None, None, None]
    v = vf.to(torch.float8_e4m3fn)
    label_reference = labels.cpu().double()
    del vf
    lengths = torch.full((batch, ), length, device='cuda', dtype=dtype)
    ks = torch.tensor([1., 2.], device='cuda')
    vs = torch.tensor([.5, 2.], device='cuda')
    qs = torch.linspace(.02, .06, hq, device='cuda') if fp8_q else None
    work = paged_attention_decode_fp8_workspace(q, k, v, table, max_context_len=bound)
    out = torch.empty(q.shape, device='cuda', dtype=torch.bfloat16)

    def call():
        paged_attention_decode_fp8(q, k, v, lengths, table, ks, vs, q_scale=qs, max_context_len=bound, out=out,
                                   workspace=work)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for active in [[length] * batch, [[length, 1, 67, 128][i % 4] for i in range(batch)], [length] * batch]:
        lengths.copy_(torch.tensor(active, device='cuda', dtype=dtype))
        out.fill_(float('nan'))
        if work is not None:
            for buffer in work:
                buffer.fill_(float('nan'))
        graph.replay()
        active_cpu = torch.tensor(active)
        visible = (active_cpu[:, None] - torch.arange(pages)[None, :] * page).clamp(0, page)
        mean = (label_reference * visible).sum(1) / active_cpu.clamp_min(1)
        factors = torch.tensor([1., 8.], device='cuda').repeat_interleave(hq // hkv)
        expected = (mean.to(device='cuda', dtype=torch.float32)[:, None, None] * factors[None, :, None]).expand_as(out)
        torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-2)


@pytest.mark.skipif(not _gfx1250_available(), reason='Requires gfx1250')
@pytest.mark.parametrize('fp8_q', [False, True], ids=['bf16-q', 'fp8-q'])
@pytest.mark.parametrize('dim,page,hq,hkv,window', [(128, 64, 16, 2, 128), (128, 64, 64, 64, 128),
                                                    (64, 64, 128, 128, 37), (64, 64, 32, 32, 64), (64, 128, 16, 2, 37),
                                                    (64, 128, 128, 128, 128), (128, 128, 16, 1, 64),
                                                    (128, 128, 64, 64, 128), (128, 128, 16, 2, 128),
                                                    (64, 128, 16, 2, 128)])
@pytest.mark.parametrize('seed', [0, 19, 182])
def test_fp8_kv_decode_window_boundaries(fp8_q, dim, page, hq, hkv, window, seed):
    lengths = [0, 1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 2303, 2304, 2305]
    q, kp, vp, contexts, table, ks, vs, qs, max_len, k, v = _make_test_inputs(lengths, hq=hq, hkv=hkv, dim=dim,
                                                                              page=page, fp8_q=fp8_q, seed=seed)
    if seed == 19:
        q = (q.float() * 4).to(q.dtype)
    if seed == 182:
        storage = torch.empty((*q.shape[:-1], q.shape[-1] * 2), device=q.device, dtype=q.dtype)
        storage[..., ::2] = q
        q = storage[..., ::2]
        contexts = contexts.to(torch.int64)
        table = table.to(torch.int64)
    out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    actual = paged_attention_decode_fp8(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len, q_scale=qs,
                                        sliding_window=window, out=out)
    expected = _reference(q, k, v, table, contexts, ks, vs, qs, 1, True, window)
    assert actual.dtype == torch.bfloat16
    assert actual.data_ptr() == out.data_ptr()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=0.0005, rtol=0.01)
