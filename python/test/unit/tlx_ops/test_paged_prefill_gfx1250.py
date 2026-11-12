"""Paged BF16 GQA: independent FP64 reference, boundaries and recovery."""

import pytest
import torch
import triton
from triton._internal_testing import is_hip_gfx1250
from triton.tlx.ops import InvalidInput, Space, BitwiseInvarianceSpec, UnsupportedBackward, UnsupportedOp, paged_prefill

pytestmark = pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")


def _inputs(batch, sq, sk, q_offset=0):
    torch.manual_seed(41)
    q_storage = torch.full((batch * sq * 16 * 64 + q_offset + 1, ), 13, dtype=torch.bfloat16, device="cuda")
    q = q_storage[q_offset:-1].view(batch, sq, 16, 64).normal_()
    pages = triton.cdiv(sk, 64)
    k = torch.randn((batch * pages + 3, 64, 2, 64), dtype=q.dtype, device=q.device)
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
        kk = k[table[batch].long()].reshape(-1, 2, 64)[:sk].double()
        vv = v[table[batch].long()].reshape(-1, 2, 64)[:sk].double()
        for h in range(16):
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
