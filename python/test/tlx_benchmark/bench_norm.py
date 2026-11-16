"""GEO normalization kernels through the shared TLX perf runner.

Both providers replay CUDA graphs; allocations happen during preparation.
Quantized cases report TLX latency without a torch quantization speedup.
"""
from __future__ import annotations

import functools
import pathlib
import sys

import torch
from triton.tlx import ops
from triton.tlx.ops.kernels.norm._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.norm._shapes import SYNTHETIC, NormShape, flops, label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import Case, Prepared, close_enough, driver

OP = "norm_fwd"
REF_NAME = "torch (plain/softmax); CUDA graphs"
EXTRA_COLUMNS = ()
COLD_COMPILE = "none"


def cases(synthetic=False, suites=None):
    shapes = SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites)
    return [
        Case(op=OP, arch=driver.arch(), dtype="bfloat16", shape=tuple(s), label=label(s), direction=s.direction)
        for s in shapes
    ]


def _graph(fn, stream=None):
    stream = stream or torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fn()

    def replay(_keepalive=fn):
        graph.replay()
        return out

    return replay


def _prepare_call(c):
    m, n, family, norm = (c['m'], c['n'], c['family'], c['norm'])
    torch.manual_seed(314159 + m + n)
    x = torch.randn(m, n, device='cuda', dtype=torch.bfloat16)
    w = (torch.rand(n, device=x.device) + 0.5).bfloat16()
    b = torch.randn_like(w) * 0.1
    dy = torch.randn_like(x)
    mean = x.float().mean(1)
    hat = x.float() if norm == 'rms' else x.float() - mean[:, None]
    rstd = torch.rsqrt(hat.square().mean(1) + 1e-05)
    length = 100 if c.get('jagged') else 128
    off = torch.arange((m + length - 1) // length + 1, device=x.device, dtype=torch.int32) * length
    off[-1] = m
    pad = torch.arange(len(off), device=x.device, dtype=torch.int32) * 128
    total_padded = (len(off) - 1) * 128
    starts = off[:-1, None] + torch.arange(4, device=x.device, dtype=torch.int32)[None, :] * 32
    counts = (off[1:, None] - starts).clamp(0, 32).contiguous().flatten()
    starts = starts.contiguous().flatten()
    kw = {}
    if family == 'plain':
        fn = (lambda: ops.norm_fwd(x, w, norm=norm, eps=1e-05)) if c['direction'] == 'fwd' else lambda: ops.norm_bwd(
            x, w, dy, rstd, norm=norm, mean=mean, has_bias=norm == 'layer')
        return (fn, (x, w, None, dy, mean, rstd))
    if family == 'softmax':
        name = 'softmax_' + c['direction']
        args = (x, ) if c['direction'] == 'fwd' else (dy, torch.softmax(x.float(), -1).to(x.dtype))
        return (functools.partial(getattr(ops, name), *args), (x, w, None, dy, mean, rstd, args))
    if family in ('mxfp8', 'preserving', 'two_pass'):
        if family == 'mxfp8':
            name = norm + 'norm_mxfp8_quantize' if norm == 'layer' else 'rmsnorm_mxfp8_quantize'
            args = (x, w, n // 128, 128)
            kw = {'save_rstd': True, **({'bias': b} if norm == 'layer' else {})}
            if c.get('jagged'):
                kw.update(offsets=off, padded_offsets=pad, total_padded=total_padded)
        elif family == 'preserving':
            name = 'layernorm_mxfp8_quantize_numerics_preserving'
            args = (x, w, n // 128, 128, off, pad, total_padded)
            kw = {'bias': b, 'write_represented_v': True, 'return_colsum': True}
        else:
            name = 'rmsnorm_mxfp8_quantize_3232_2pass'
            args = (x, w, off, pad, n // 128, 128)
            kw = {'save_rstd': True, 'total_padded': total_padded}
    elif family in ('autograd', 'gdpa', 'gdpa_precomputed'):
        name = {
            'autograd': 'layernorm_mxfp8_with_backward', 'gdpa': 'layernorm_mxfp8_gdpa_with_backward',
            'gdpa_precomputed': 'layernorm_mxfp8_gdpa_precomputed_no_normalized'
        }[family]
        args = (x, w, b, n // 128, 128)
        if family != 'autograd':
            args += (off, pad, off.cpu(), total_padded, starts, counts)
    elif family in ('saved', 'dx_quant'):
        if family == 'saved':
            name = 'layernorm_backward' if norm == 'layer' else 'rmsnorm_backward'
            source = x
            if norm == 'layer':
                kw['norm_bias'] = b
            if c['variant'] != 'raw':
                name += '_from_x_norm'
                source = (hat * rstd[:, None] * w.float() + (b.float() if norm == 'layer' else 0)).bfloat16()
                if c['variant'] == 'fp8':
                    source = source.to(torch.float8_e4m3fn)
                    kw['sf_x_norm'] = torch.full((m, n // 32), 127, device=x.device, dtype=torch.uint8)
            args = (dy, source, w, rstd) + ((mean, ) if norm == 'layer' else ())
        else:
            name = 'rmsnorm_backward_quantize_dx_mxfp8'
            args = (dy, x, w, rstd, n // 128, 128, off, pad, total_padded, starts, counts)
            if c['variant'] == 'from_output':
                name += '_from_output_recompute_q_precomputed'
                final = x + (x.float() * rstd[:, None]).bfloat16()
                ln_rstd = torch.rsqrt((x.float() - mean[:, None]).square().mean(1) + 1e-05)
                args = (dy, final, x, rstd, n // 128, 128, off, pad, total_padded, starts, counts, mean, ln_rstd)
            else:
                name += '_bf16_precomputed' if c['variant'] == 'bf16' else '_precomputed'
    elif family == 'mxfp4':
        name = 'layernorm_mxfp4' if norm == 'layer' else 'rmsnorm_mxfp4'
        w = w.float()
        args = (x, w)
        kw = {'return_rstd': True, **({'return_mean': True} if norm == 'layer' else {})}
    elif family == 'nvfp4':
        name = 'layernorm_nvfp4_quantize'
        args = (x, w)
        kw = {
            'bias': b if norm == 'layer' else None, 'global_scale': 32, 'save_rstd': True, 'use_layernorm':
            norm == 'layer', 'quant_block_shape': (c['qrows'], 16)
        }
    elif family == 'stacked':
        name = 'layernorm_nvfp4_quantize_q0_jagged_transpose'
        args = (x, w, off, pad)
        kw = {
            'padded_tokens': total_padded, 'max_seq_len': length, 'bias': b, 'global_scale': 1,
            'transpose_global_scale': 64
        }
    else:
        payload, scales, saved_rstd, _ = ops.layernorm_nvfp4_quantize(x, w, bias=b, global_scale=32, save_rstd=True)
        name = 'layernorm_nvfp4_backward_from_x_norm_padded_dy'
        args = (dy, off, pad, payload, scales, 32, w, saved_rstd)
        kw = {'norm_bias': b, 'min_abs_weight': None}
    return (functools.partial(getattr(ops, name), *args,
                              **kw), (x, w, b, dy, mean, rstd, args, kw, off, pad, total_padded))


def prepare(case, space):
    if space != 'heuristic':
        raise ops.InvalidInput("normalization benchmarks support only space='heuristic'")
    shape = NormShape(*case.shape)
    c = shape._asdict()
    fn, values = _prepare_call(c)
    x, w, _, dy = values[:4]
    ref = None
    ref_stream = None
    if shape.family == "plain":
        if shape.direction == "fwd":
            ref = lambda: torch.nn.functional.layer_norm(x, (shape.n, ), w, eps=1e-5
                                                         ) if shape.norm == "layer" else torch.nn.functional.rms_norm(
                                                             x, (shape.n, ), w, eps=1e-5)
        else:
            rx, rw = x.detach().requires_grad_(), w.detach().requires_grad_()
            ref_stream = torch.cuda.Stream()
            ref_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ref_stream):
                if shape.norm == "layer":
                    rb = torch.zeros_like(w, requires_grad=True)
                    ref_y = torch.nn.functional.layer_norm(rx, (shape.n, ), rw, rb, 1e-5)
                    inputs = (rx, rw, rb)
                else:
                    ref_y = torch.nn.functional.rms_norm(rx, (shape.n, ), rw, 1e-5)
                    inputs = (rx, rw)
            ref = lambda: torch.autograd.grad(ref_y, inputs, dy, retain_graph=True)
    elif shape.family == "softmax":
        ref = (lambda: torch.softmax(x, -1)) if shape.direction == "fwd" else (
            lambda: values[6][1] * (dy - (dy.float() * values[6][1].float()).sum(-1, keepdim=True)))
    tlx_fn = _graph(fn)
    ref_fn = _graph(ref, ref_stream) if ref is not None else None

    def check():
        if ref_fn is None:
            return True, "quantized accuracy covered by L1"
        actual, expected = tlx_fn(), ref_fn()
        if shape.family == "plain":
            if shape.direction == "fwd":
                actual = actual[0]
            else:
                return close_enough(actual[0].float(), expected[0].float(), 0.02)
        return close_enough(actual.float(), expected.float(), 0.02)

    return Prepared(tlx_fn=tlx_fn, ref_fn=ref_fn, flop_count=flops(shape), check=check)


supported, default_json, run, main = driver.bind(sys.modules[__name__])
if __name__ == "__main__":
    raise SystemExit(main())
