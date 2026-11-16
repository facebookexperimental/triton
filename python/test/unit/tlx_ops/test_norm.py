"""Analytical correctness and API contracts for GEO normalization ports."""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest
import torch
import triton
from triton._internal_testing import is_hopper_or_newer
from triton.tlx import ops
from triton.tlx.ops.kernels.norm._shapes import GEO

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'tlx_benchmark'))
from bench_norm import _prepare_call

pytestmark = pytest.mark.skipif(not is_hopper_or_newer(), reason='Requires Hopper or Blackwell')


def metric(a, b, minimum=35):
    if a is None or b is None:
        return {'pass_': a is b}
    if a.shape != b.shape or a.dtype != b.dtype:
        return {'pass_': False, 'shape': [list(a.shape), list(b.shape)], 'dtype': [str(a.dtype), str(b.dtype)]}
    raw = int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())
    if not a.is_floating_point() or a.dtype == torch.float8_e8m0fnu:
        return {'pass_': raw == 0, 'differing_bytes': raw}
    aa, bb = (a.float(), b.float())
    finite = bool(torch.isfinite(aa).all() & torch.isfinite(bb).all())
    error = (aa - bb).square().sum().item()
    signal = bb.square().sum().item()
    sqnr = 999.0 if error == 0 else 10 * math.log10(max(signal, 1e-30) / max(error, 1e-30))
    return {'pass_': finite and sqnr >= minimum, 'finite': finite, 'sqnr_db': sqnr, 'differing_bytes': raw}


def decode(payload, scales, block=32, fp4=False, global_scale=1, rows=None, groups=None, scale_columns=None):
    payload = payload.reshape(payload.shape[0], -1)
    m = payload.shape[0]
    n = payload.shape[1] * (2 if fp4 else 1)
    rows = torch.arange(m, device=payload.device)[:, None] if rows is None else rows
    groups = (torch.arange(n, device=payload.device) // block)[None, :] if groups is None else groups
    address = (rows // 128 * triton.cdiv(scale_columns or n // block, 4) +
               groups // 4) * 512 + rows % 32 * 16 + rows // 32 % 4 * 4 + groups % 4
    if fp4:
        nibble = torch.stack((payload & 15, payload >> 4), -1).reshape(m, n).long()
        values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=payload.device)[nibble & 7]
        values *= torch.where(nibble & 8 != 0, -1.0, 1.0)
    else:
        values = payload.float()
    sf = scales.flatten()[address].contiguous().view(torch.float8_e8m0fnu if block ==
                                                     32 else torch.float8_e4m3fn).float()
    return values * sf / global_scale


def norm_reference(c, x, w, b):
    family = c['family']
    xf = x.double() if family == 'plain' else x.float()
    mean = xf.mean(1, keepdim=True) if c['norm'] == 'layer' else torch.zeros_like(xf[:, :1])
    centered = xf - mean
    variance = centered.square().mean(
        1, keepdim=True) if family in ('plain', 'mxfp4',
                                       'preserving') else xf.square().mean(1, keepdim=True) - mean.square()
    rstd = torch.rsqrt(variance + (1e-06 if family == 'mxfp4' else 1e-05))
    if family in ('mxfp8', 'autograd', 'gdpa', 'gdpa_precomputed') and c['norm'] == 'layer':
        offset = -(mean * rstd)
        normalized = (xf.double() * rstd.double() + offset.double()).float().bfloat16()
        y = (normalized.double() * w.double() + b.double()).bfloat16()
    else:
        y = centered * rstd * w.to(xf.dtype)
        if b is not None:
            y = y + b.to(xf.dtype)
        y = y.to(x.dtype)
    if family == 'mxfp4':
        y = y.float().reshape(x.shape[0], x.shape[1] // 32, 32)
        for stage in range(5):
            stride = 1 << stage
            tile = y.reshape(*y.shape[:2], 32 // (2 * stride), 2, stride)
            a, b = (tile[..., 0, :], tile[..., 1, :])
            y = torch.stack((a + b, a - b), -2).bfloat16().float().reshape_as(y)
        y = (y * 0.1767766953125).bfloat16().reshape_as(x)
    return (y, mean.flatten().float(), rstd.flatten().float())


def grad_reference(xhat, dy, w, rstd, norm):
    xhat, dy = (xhat.double(), dy.double())
    wd = dy if w is None else dy * w.double()
    dx = wd - xhat * (wd * xhat).mean(1, keepdim=True)
    if norm == 'layer':
        dx = dx - wd.mean(1, keepdim=True)
    return (dx * rstd.double()[:, None], (dy * xhat).sum(0), dy.sum(0))


def _validate_geo(c, q, inputs):
    q = (q, ) if isinstance(q, torch.Tensor) else q
    t = q
    family = c['family']
    expected, representations = {
        'two_pass': (4, (0, 1, 2)), 'mxfp8': (5 if c['norm'] == 'layer' else 4, (0, 1, 2)), 'mxfp4':
        (4 if c['norm'] == 'layer' else 3, (0, 1)), 'nvfp4': (4 if c['norm'] == 'layer' else 3, (0, 1)), 'stacked':
        (6, (0, 1, 3, 4)), 'preserving': (10, (0, 2, 3, 8)), 'autograd': (4, (1, 2, 3)), 'gdpa': (4, (1, 2, 3)),
        'gdpa_precomputed': (5, (0, 1, 2)), 'saved': (3 if c['norm'] == 'layer' else 2, ()), 'dx_quant': (5, (2, 3, 4)),
        'nvfp4_backward': (3, ())
    }.get(family, (None, ()))
    if expected is not None and len(q) != expected:
        raise ValueError(f'{family} requires {expected} outputs, received {len(q)}')
    x, w, b = inputs[:3]
    dtypes = {
        'nvfp4': (torch.uint8, torch.uint8) + (torch.float32, ) * (len(q) - 2), 'stacked':
        (torch.uint8, torch.uint8, torch.float32, torch.uint8, torch.uint8, torch.float32), 'nvfp4_backward':
        (x.dtype, torch.float32, torch.float32), 'dx_quant':
        (x.dtype, torch.float8_e4m3fn if c.get('variant') == 'from_output' else torch.float32, torch.float8_e4m3fn,
         torch.uint8, torch.uint8)
    }.get(family)
    checks = []

    def check(a, b, minimum=35):
        checks.append(metric(a, b, minimum))

    for i, value in enumerate(q):
        if i in representations:
            dtype = torch.float8_e4m3fn if family not in (
                'mxfp4', 'nvfp4', 'stacked') and i == representations[0] else torch.float8_e8m0fnu if family in (
                    'mxfp8', 'two_pass') and i == 2 else torch.uint8
            assert value.dtype == dtype
        if dtypes is not None:
            assert value is not None and value.dtype == dtypes[i]
    if representations:
        off, pad, total = inputs[8:]
        padded_rows = torch.arange(x.shape[0], device=x.device)
        if family in ('mxfp8', 'preserving', 'gdpa', 'gdpa_precomputed', 'dx_quant'):
            for begin, end, pbegin in zip(off[:-1].tolist(), off[1:].tolist(), pad[:-1].tolist()):
                padded_rows[begin:end] = torch.arange(pbegin, pbegin + end - begin, device=x.device)
        head_major = family in ('gdpa', 'gdpa_precomputed', 'dx_quant')
        groups = torch.arange(x.shape[1], device=x.device)[None, :] // 32
        krows = padded_rows[:,
                            None] + groups // 4 * total if head_major else padded_rows[:,
                                                                                       None] if family == 'preserving' else None
        qindex, kindex = {'autograd': (1, 2), 'gdpa': (1, 2), 'dx_quant': (2, 3), 'preserving':
                          (0, 3)}.get(family, (0, 1))
        fp4 = family in ('mxfp4', 'nvfp4', 'stacked')
        block, scale = (16, 32) if family == 'nvfp4' else (16, 1) if family == 'stacked' else (32, 1)
        if family == 'dx_quant':
            ref = q[0].float()
        else:
            ref, expected_mean, expected_rstd = norm_reference(c, x, w,
                                                               None if c['norm'] == 'rms' or family == 'mxfp4' else b)
        for out in (t, ):
            payload = out[qindex].reshape(x.shape[0], -1)
            value = decode(payload, out[kindex], block, fp4, scale, rows=krows,
                           groups=groups % 4 if head_major else None, scale_columns=4 if head_major else None)
            check(value, ref.float(), 14 if fp4 else 28)
            stats_index = {
                'mxfp4': 2, 'nvfp4': 2, 'stacked': 2, 'preserving': 4, 'mxfp8': 3, 'gdpa_precomputed': 3, 'two_pass': 3
            }.get(family)
            if stats_index is not None and out[stats_index] is not None:
                check(out[stats_index], expected_rstd, 35)
                if family != 'stacked' and c['norm'] == 'layer':
                    check(out[stats_index + 1], expected_mean, 35)
            if family not in ('mxfp4', 'nvfp4', 'stacked'):
                trans = out[{'autograd': 3, 'gdpa': 3, 'dx_quant': 4}.get(family, 2)]
                if family == 'preserving':
                    trans = trans[:-x.shape[0] * (x.shape[1] // 32)]
                trans_value = decode(payload.T.contiguous(), trans, groups=padded_rows[None, :] // 32,
                                     scale_columns=total // 32)
                check(trans_value, value.T, 60)
            if family == 'preserving':
                row_sf = out[2][-x.shape[0] * (x.shape[1] // 32):].reshape(x.shape[0], -1)
                check(
                    payload.float() * row_sf[:, torch.arange(x.shape[1], device=x.device) // 32].contiguous().view(
                        torch.float8_e8m0fnu).float(), value, 60)
                check(
                    decode(payload, out[8], rows=padded_rows[:, None] + groups // 4 * total, groups=groups % 4,
                           scale_columns=4), value, 60)
                check(out[7].reshape_as(value).float(), value, 60)
                check(out[9], ref.float().sum(0), 35)
                check(out[6], ref.float().abs().amax().reshape_as(out[6]), 35)
            elif family in ('autograd', 'gdpa'):
                check(out[0], x if family == 'autograd' else ref, 35)
            elif family == 'stacked':
                check(out[5],
                      torch.full((1, ), 1 / inputs[7]['transpose_global_scale'], device=x.device, dtype=torch.float32))
                trans_ref = torch.nn.functional.pad(ref.T.float(), (0, total - x.shape[0]))
                trans_value = decode(out[3], out[4], 16, True, 64)
                check(trans_value, trans_ref, 14)
    if family in ('saved', 'dx_quant', 'nvfp4_backward'):
        dy, mean, rstd = inputs[3:6]
        xhat = x.float() if c['norm'] == 'rms' else x.float() - mean[:, None]
        xhat = xhat * rstd[:, None]
        if family == 'saved' and c['variant'] != 'raw':
            source = inputs[6][1].float()
            xhat = (source - (b.float() if c['norm'] == 'layer' else 0)) / w.float()
        elif family == 'nvfp4_backward':
            args = inputs[6]
            represented = decode(args[3], args[4], 16, True, args[5])
            xhat = represented.double() / w.double() - b.double() / w.double()
            rstd = args[7]
        elif family == 'dx_quant' and c['variant'] == 'from_output':
            xhat = (inputs[6][1].float() - x.float()).bfloat16().float()
            w = None
        reference = grad_reference(xhat, dy, w, rstd, c['norm'])
        for out in (t, ):
            count = 1 if family == 'dx_quant' and c['variant'] == 'from_output' else 3 if c['norm'] == 'layer' else 2
            for i in range(count):
                if out[i] is not None:
                    check(out[i], reference[i].to(out[i].dtype),
                          30 if family == 'dx_quant' and c['variant'] == 'bf16' else 35)
            if family == 'dx_quant' and c['variant'] == 'from_output':
                check(out[1].reshape_as(x).float() / 16, (x.float() - mean[:, None]) * inputs[6][-1][:, None], 28)
    if not all(v['pass_'] for v in checks):
        raise ValueError(json.dumps(checks))
    return checks


@pytest.mark.parametrize("norm", ["layer", "rms"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m,n", [(0, 384), (1, 1), (33, 127), (128, 384), (4096, 384), (3, 32768), (2, 65536)])
def test_plain(norm, dtype, m, n):
    torch.manual_seed(7)
    x = torch.randn((m, n + 8), device="cuda", dtype=dtype)[:, :n]
    w = torch.rand(n, device=x.device, dtype=dtype) + 0.5
    b = torch.randn_like(w)
    out = torch.empty_like(x)
    mu = torch.empty(m, device=x.device, dtype=torch.float32)
    rr = torch.empty_like(mu)
    y, _, mean, rstd = ops.norm_fwd(x, w, b, norm=norm, eps=1e-5, out=out, mean_out=mu, rstd_out=rr)
    assert y is out and rstd is rr and (norm == "rms" or mean is mu)
    xf = x.double()
    ref_mu = xf.mean(1, keepdim=True) if norm == "layer" else torch.zeros_like(xf[:, :1])
    hat = xf - ref_mu
    ref_rr = torch.rsqrt(hat.square().mean(1, keepdim=True) + 1e-5)
    hat *= ref_rr
    precision = 0.02 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(y.float(), (hat * w.double() + b.double()).float(), atol=precision, rtol=precision)
    dy = torch.randn_like(x)
    dx, dw, db = ops.norm_bwd(x, w, dy, rr, mean=mean, norm=norm, has_bias=True, dw_dtype=torch.float32,
                              db_dtype=torch.float32)
    ref = grad_reference(hat, dy, w, ref_rr.flatten(), norm)
    for actual, expected in zip((dx, dw, db), ref):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=precision,
            atol=precision * max(1, float(expected.abs().max())) if expected.numel() else precision)


@pytest.mark.parametrize("norm", ["layer", "rms"])
def test_autograd_residuals(norm):
    x, residual, post = [
        torch.randn((33, 384), device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    ]
    w = torch.randn(384, device="cuda", dtype=x.dtype, requires_grad=True)
    b = torch.randn_like(w, requires_grad=True)
    ref_x, ref_r, ref_post, ref_w, ref_b = [v.detach().float().requires_grad_() for v in (x, residual, post, w, b)]
    if norm == "layer":
        y, _, _ = ops.layernorm_add(x, residual, w, bias=b, eps=1e-5)
        ref = torch.nn.functional.layer_norm(ref_x + ref_r, (384, ), ref_w, ref_b, 1e-5)
        compared = ((x, ref_x), (residual, ref_r), (w, ref_w), (b, ref_b))
    else:
        y, saved, _ = ops.rmsnorm(x, w, b, residual, post, eps=1e-5)
        ref = torch.nn.functional.rms_norm(ref_x + ref_r, (384, ), ref_w, 1e-5) + ref_b + ref_post
        compared = ((x, ref_x), (residual, ref_r), (post, ref_post), (w, ref_w), (b, ref_b))
        torch.testing.assert_close(saved.float(), (x.float() + residual.float()).bfloat16().float())
    y.sum().backward()
    ref.sum().backward()
    torch.testing.assert_close(y.float(), ref, atol=0.04, rtol=0.02)
    for actual, expected in compared:
        torch.testing.assert_close(actual.grad.float(), expected.grad, atol=0.05, rtol=0.03)


@pytest.mark.parametrize("n", [0, 1, 8, 31, 128, 129, 384, 513, 2048, 16384, 32769, 65536])
@pytest.mark.parametrize("transpose", [False, True])
def test_softmax(n, transpose):
    torch.manual_seed(8)
    x = torch.randn((n, 3) if transpose else (3, n), device="cuda", dtype=torch.bfloat16)
    if transpose:
        x = x.T
    out = torch.empty_like(x)
    y = ops.softmax_fwd(x, out=out)
    assert y is out
    torch.testing.assert_close(y.float(), torch.softmax(x.float(), -1), atol=0.002, rtol=0.02)
    dy = torch.ones((1, n), device=x.device, dtype=x.dtype).expand_as(x)
    dx = ops.softmax_bwd(dy, y)
    expected = y.float() * (dy.float() - (dy.float() * y.float()).sum(1, keepdim=True))
    torch.testing.assert_close(dx.float(), expected, atol=1e-6, rtol=0.03)


GEO_CASES = tuple(dict.fromkeys(s._replace(m=min(s.m, 128)) for s in GEO if s.family not in ("plain", "softmax")))


@pytest.mark.parametrize("shape", GEO_CASES,
                         ids=lambda s: f"{s.family}-{s.norm}-{s.n}-{s.variant}-q{s.qrows}-j{s.jagged}")
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 10, reason="Requires Blackwell")
def test_geo_quantized_variants(shape):
    fn, values = _prepare_call(shape._asdict())
    out = fn()
    _validate_geo(shape._asdict(), out, values)


@pytest.mark.parametrize("norm", ["layer", "rms"])
@pytest.mark.parametrize("amax,layout", [("1x32", "rowmajor"), ("1x32", "mma"), ("1x32", "cublas"),
                                         ("32x1", "rowmajor"), ("32x32", "dual")])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 10, reason="Requires Blackwell")
def test_mxfp8_layouts(norm, amax, layout):
    x = torch.randn((33, 384), device="cuda", dtype=torch.bfloat16)
    w = torch.rand(384, device=x.device, dtype=x.dtype) + 0.5
    y, _mean, rr, q, k, t, colsum = ops.norm_quant_mxfp8_fwd(x, w, norm=norm, eps=1e-5, amax_block=amax,
                                                             scale_layout=layout, return_colsum=True)
    ref, _mu, rstd = norm_reference({"family": "plain", "norm": norm}, x, w, None)
    torch.testing.assert_close(y.float(), ref.float(), rtol=0.02, atol=0.02)
    torch.testing.assert_close(rr.flatten(), rstd, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(colsum.float(), y.float().sum(0), rtol=0.02, atol=0.1)
    if amax == "32x1":
        decoded = q.float() * k.T[torch.arange(33, device=x.device) // 32].contiguous().view(
            torch.float8_e8m0fnu).float()
    elif layout == "rowmajor":
        decoded = q.float() * k[:, torch.arange(384, device=x.device) // 32].contiguous().view(
            torch.float8_e8m0fnu).float()
    else:
        decoded = decode(q, k)
    assert metric(decoded, y.float(), 28)["pass_"]
    if layout == "dual":
        assert metric(decode(q.T.contiguous(), t), decoded.T, 60)["pass_"]


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 10, reason="Requires Blackwell")
def test_invalid_inputs():
    x = torch.randn((4, 384), device="cuda", dtype=torch.bfloat16)
    w = torch.ones(384, device=x.device, dtype=x.dtype)
    for fn in (lambda: ops.layernorm(x, w[:3]), lambda: ops.norm_fwd(x, norm="other"),
               lambda: ops.layernorm(x, w, eps=-1),
               lambda: ops.softmax_fwd(x, out=torch.empty(
                   (1, 384), device=x.device).expand_as(x)), lambda: ops.layernorm_mxfp4(x[:, :33], w[:33]),
               lambda: ops.layernorm_nvfp4_quantize(x, w, quant_block_shape=(2, 16)),
               lambda: ops.rmsnorm_mxfp8_quantize(x, w, 3, 64), lambda: ops.softmax_fwd(x, space="full")):
        with pytest.raises(ops.InvalidInput):
            fn()
    with pytest.raises(ops.UnsupportedBackward):
        ops.layernorm_mxfp4(x.requires_grad_(), w)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 10, reason="Requires Blackwell")
@pytest.mark.parametrize("gdpa", [False, True])
def test_mxfp8_routed_autograd(gdpa):
    shape = next(s for s in GEO_CASES if s.family == ('gdpa' if gdpa else 'autograd'))
    fn, values = _prepare_call(shape._asdict())
    x, w, b, dy = values[:4]
    for v in (x, w, b):
        v.requires_grad_()
    routed, *_ = fn()
    routed.backward(dy)
    _, mu, rr = norm_reference({"family": "mxfp8", "norm": "layer"}, x.detach(), w.detach(), b.detach())
    hat = (x.detach().float() - mu[:, None]) * rr[:, None]
    for actual, expected in zip((x.grad, w.grad, b.grad), grad_reference(hat, dy, w.detach(), rr, 'layer')):
        assert metric(actual, expected.to(actual.dtype), 35)['pass_']


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 10, reason="Requires Blackwell")
def test_quantized_output_buffers_and_saved_addends():
    c = next(s for s in GEO_CASES if s.family == 'stacked')._asdict()
    fn, values = _prepare_call(c)
    x, w, b, dy, mean, rr, args, kw = values[:8]
    out = fn()
    reused = tuple(torch.empty_like(v) for v in out)
    ops.layernorm_nvfp4_quantize_q0_jagged_transpose_inplace(*args, reused, **kw)
    for actual, expected in zip(reused, out):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    q, sf, rstd, _mu = ops.layernorm_nvfp4_quantize(x, w, bias=b, global_scale=32, save_rstd=True)
    ops.layernorm_nvfp4_quantize_inplace(x, w, reused[0], reused[1], 32, bias=b, rstd_out=reused[2])
    for actual, expected in zip(reused[:3], (q, sf, rstd)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    y, *_ = ops.norm_fwd(x, w, b, eps=1e-5)
    q2, sf2 = ops.nvfp4_quantize_normalized(y, 32)
    assert metric(decode(q2, sf2, 16, True, 32), y.float(), 14)['pass_']
    dv = torch.randn_like(dy).to(torch.float8_e4m3fn)
    add = torch.randn_like(dy)
    scales = torch.full((x.shape[0], x.shape[1] // 32), 127, device=x.device, dtype=torch.uint8)
    dx, dw, db, saved = ops.layernorm_backward(dy, x, w, rr, mean, save_x_norm=True, dv_fp8=dv, sf_dv=scales,
                                               dy_bf16_addend=add, norm_bias=b)
    hat = (x.float() - mean[:, None]) * rr[:, None]
    grad = (dy.float() + add.float()).bfloat16().float() + dv.float()
    for actual, expected in zip((dx, dw, db), grad_reference(hat, grad, w, rr, 'layer')):
        assert metric(actual, expected.to(actual.dtype), 35)['pass_']
    torch.testing.assert_close(saved, (hat * w.float() + b.float()).bfloat16(), rtol=0.02, atol=0.02)
