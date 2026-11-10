"""Tune exact split-merge ownership using real attention partials."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from bench_decode import inputs, reference, tlx_source, timing, make_backend


@triton.jit
def merge_owned(PART, LSE, OUT, VS, D: tl.constexpr, S: tl.constexpr, BS: tl.constexpr, HQ: tl.constexpr,
                G: tl.constexpr, VL: tl.constexpr, PL: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    lse = tl.load(LSE + row * S + s, s < S, other=-float('inf'))
    if PL is not None: lse = tlx.require_layout(lse, PL)
    m = tl.max(lse, 0)
    m = tl.where(m == -float('inf'), 0.0, m)
    p = tl.exp2(lse - m)
    vals = tl.load(PART + (row * S + s[:, None]) * D + d[None, :], s[:, None] < S, other=0.0)
    vals = tlx.require_layout(vals, VL).to(tl.float32)
    weights = tlx.release_layout(p)
    value = tlx.release_layout(tl.sum(vals * weights[:, None], 0)) / tl.maximum(tl.sum(p, 0), 1e-20)
    if PART.dtype.element_ty == tl.float16: value *= tl.load(VS + (row % HQ) // G)
    tl.store(OUT + row * D + d, value)


def layouts(d, bs, dt, pin):
    st = 32 // dt
    vl = tlx.layout(shape=((dt, st), (d // dt, bs // st)), stride=((1, d), (dt, st * d)))
    pl = tlx.layout(shape=((dt, st), (bs // st, )), stride=((0, 1), (st, ))) if pin else None
    return vl, pl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    for name, default in [('heads', 16), ('kv-heads', 2), ('dim', 128), ('length', 2304), ('page', 128),
                          ('splits', 18)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--scales', default='shared')
    args = p.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    a = SimpleNamespace(**vars(args), tokens=1, window=0, fp8_q=False, causal=True, block_n=args.page, block_m=16,
                        warps=1)
    x = inputs(a)
    expected = reference(x, a)
    module = tlx_source()
    opt = module._optimized_module()
    original = opt._merge
    out = torch.empty_like(x.q, dtype=torch.bfloat16)
    work = (torch.empty((1, a.heads, a.splits, a.dim),
                        device=x.q.device), torch.empty((1, a.heads, a.splits), device=x.q.device))

    def attention():
        return module.fp8_kv_decode(x.q, x.kp, x.vp, x.contexts, x.table, x.ks, x.vs, max_context_len=a.length,
                                    num_splits=a.splits, variant='matrix_tdm_flat', block_n=a.page, num_warps=1,
                                    out=out, workspace=work)

    attention()
    torch.cuda.synchronize()
    threshold = 8  # This experiment uses the recorded EXP062 candidate convention.
    part = work[0].view(torch.float16) if a.splits >= threshold else work[0]
    bs = triton.next_power_of_2(a.splits)
    records = []
    payload = dict(workload=vars(args) | {'output': str(args.output)}, rows=records)

    def check(fn):
        actual = fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.float(), expected, atol=5e-4, rtol=1e-2)
        return float((actual.float() - expected).abs().max())

    def baseline():
        original[(a.heads, )](part, work[1], out, a.dim, a.splits, bs, x.vs, a.heads, a.heads // a.kv_heads,
                              num_warps=1)
        return out

    records.append(dict(name='original', max_abs_error=check(baseline), status='pass', **timing(baseline, 9, 30)))
    for dt in [1, 2, 4, 8, 16, 32]:
        if 32 // dt > bs: continue
        for pin in [False, True]:
            row = dict(name='owned', dimension_threads=dt, pin_lse=pin)
            try:
                vl, pl = layouts(a.dim, bs, dt, pin)

                def invoke():
                    merge_owned[(a.heads, )](part, work[1], out, x.vs, a.dim, a.splits, bs, a.heads,
                                             a.heads // a.kv_heads, vl, pl, num_warps=1)
                    return out

                row.update(max_abs_error=check(invoke), status='pass', **timing(invoke, 9, 30))
            except Exception as exc:
                row.update(status='failed', reason=str(exc))
            records.append(row)
            args.output.write_text(json.dumps(payload, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    passing = [r for r in records if r['status'] == 'pass' and r['name'] == 'owned']
    best = min(passing, key=lambda r: r['median_us'])
    vl, pl = layouts(a.dim, bs, best['dimension_threads'], best['pin_lse'])

    class Proxy:

        def __getitem__(self, grid):

            def launch(part, lse, out, d, s, bs, vs, hq, g, **kwargs):
                return merge_owned[grid](part, lse, out, vs, d, s, bs, hq, g, vl, pl, num_warps=1)

            return launch

    def old_attention():
        opt._merge = original
        return attention()

    def new_attention():
        opt._merge = Proxy()
        return attention()

    names = [('old_merge', old_attention), ('new_merge', new_attention)]
    if a.scales == 'shared': names.append(('aiter', make_backend('aiter_unified', x, a)))
    full = []
    try:
        for rnd in range(3):
            order = names[rnd % len(names):] + names[:rnd % len(names)]
            if rnd == 1: order = list(reversed(order))
            for name, fn in order:
                full.append(dict(backend=name, round=rnd, max_abs_error=check(fn), status='pass', **timing(fn, 21, 30)))
    finally:
        opt._merge = original
    payload.update(best=best, full_attention=full)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print('best', json.dumps(best), flush=True)


if __name__ == '__main__': main()
