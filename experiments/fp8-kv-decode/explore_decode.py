"""Gated TLX launch sweeps and order-rotated frozen/AITER comparisons."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

import torch
from bench_decode import inputs, reference, make_backend, timing, load_source, tlx_resources, tlx_source

ROOT = Path(__file__).resolve().parent


def frozen_call(x, a, explicit=False):
    frozen_dir = getattr(a, 'frozen_dir', None) or ROOT / 'results/2026-11-09_optimization/frozen'
    module = load_source('tlx_nov09_frozen', Path(frozen_dir) / 'amd_pa_decode_fp8.py')
    cfg = module.select_decode_config(x.q, x.kp, max_context_len=a.length, query_length=a.tokens, window_size=a.window)
    if explicit:
        cfg = dict(variant='matrix_tdm_flat', block_n=a.block_n, block_m=a.block_m, num_splits=a.splits,
                   num_warps=a.warps)
    ns = cfg['num_splits']
    out = torch.empty_like(x.q, dtype=torch.bfloat16)
    workspace = (torch.empty((a.tokens, a.heads, ns, a.dim),
                             device=x.q.device), torch.empty((a.tokens, a.heads, ns), device=x.q.device))
    fn = lambda: module.fp8_kv_decode(
        x.q, x.kp, x.vp, x.contexts, x.table, x.ks, x.vs, max_context_len=a.length, q_scale=x.qs
        if a.fp8_q else None, query_length=a.tokens, causal=a.causal, window_size=a.window, variant=cfg[
            'variant'], num_splits=ns, block_n=cfg['block_n'], block_m=cfg['block_m'], num_warps=cfg[
                'num_warps'], out=out, workspace=workspace)
    return fn


def evaluate(name, x, a, expected, samples, repeats):
    row = dict(backend=name, block_n=a.block_n, block_m=a.block_m, warps=a.warps, splits=a.splits)
    try:
        if name == 'tlx_archived_multipage':
            m = load_source('tlx_archived_multipage', ROOT / 'snapshots/2026-11-05_exp030_multipage.py')
            out = torch.empty_like(x.q, dtype=torch.bfloat16)
            fn = lambda: m.decode(x.q, x.kp, x.vp, x.contexts, x.table, x.ks, x.vs, q_scale=x.qs
                                  if a.fp8_q else None, query_length=a.tokens, causal=a.causal, window_size=a.window,
                                  sm_scale=a.dim**-.5, num_splits=a.splits, variant='matrix_tdm_flat', out=out, block_n=
                                  a.block_n, block_m=a.block_m, num_warps=a.warps)
        elif name == 'tlx_short_candidate':
            from short_decode_candidate import make_call
            fn = make_call(x, a, tlx_source()._optimized_module()._vector_layouts)
        else:
            fn = frozen_call(x, a, explicit=name
                             == 'tlx_frozen_launch') if name.startswith('tlx_frozen') else make_backend(name, x, a)
        actual = fn()
        torch.cuda.synchronize()
        error = actual.float() - expected
        row.update(max_abs_error=float(error.abs().max()), rms_error=float(error.square().mean().sqrt()))
        torch.testing.assert_close(actual.float(), expected, atol=5e-4, rtol=1e-2)
        row.update(status='pass', **timing(fn, samples, repeats))
    except Exception as exc:
        row.update(status='failed', reason=str(exc), traceback=traceback.format_exc())
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    for key, default in [('length', 2304), ('heads', 64), ('kv-heads', 64), ('dim', 128), ('tokens', 1), ('page', 64),
                         ('window', 0)]:
        p.add_argument('--' + key, type=int, default=default)
    p.add_argument('--scales', default='per-head', choices=['per-head', 'shared', 'unit'])
    p.add_argument('--fp8-q', action='store_true')
    p.add_argument('--variants', nargs='+', default=['tlx_matrix_tdm_flat'])
    p.add_argument('--splits', nargs='+', type=int, default=[1, 4, 8, 9, 12, 16, 24, 36, 64, 128])
    p.add_argument('--warps', nargs='+', type=int, default=[1, 2, 4])
    p.add_argument('--block-m', nargs='+', type=int, default=[16])
    p.add_argument('--block-n', nargs='+', type=int, default=[64])
    p.add_argument('--paired', type=Path, help='JSON row defining the candidate; benchmark versus frozen/AITER')
    p.add_argument('--frozen-launch', action='store_true', help='Also repeat frozen GPU code at the candidate launch')
    p.add_argument('--frozen-dir', type=Path,
                   help='Alternative frozen kernel directory for matched GPU-code comparisons')
    p.add_argument('--dump-ir', action='store_true')
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    a = SimpleNamespace(**{k: v for k, v in vars(args).items() if k not in ['output', 'variants', 'paired']})
    a.causal = a.tokens == 1
    a.splits = 16
    a.warps = 4
    a.block_m = 16
    a.block_n = 64
    x = inputs(a)
    expected = reference(x, a)
    records = []
    if args.paired:
        cfg = json.loads(args.paired.read_text())
        settings = []
        names = ['tlx_frozen'] + (['tlx_frozen_launch'] if args.frozen_launch else
                                  []) + [cfg['backend']] + (['aiter_unified'] if a.scales != 'per-head' else [])
        for round_index in range(3):
            order = names[round_index % len(names):] + names[:round_index % len(names)]
            if round_index == 1: order = list(reversed(order))
            settings.extend(
                (name, cfg['block_n'], cfg['block_m'], cfg['splits'], cfg['warps'], round_index) for name in order)
    else:
        settings = [(name, bn, bm, s, w, None)
                    for name in args.variants
                    for bn in args.block_n
                    for bm in args.block_m
                    for s in args.splits
                    for w in args.warps]
    for name, bn, bm, s, w, round_index in settings:
        a.block_n, a.block_m, a.splits, a.warps = bn, bm, s, w
        row = evaluate(name, x, a, expected, 21 if args.paired else 5, 30 if args.paired else 10)
        if round_index is not None: row['round'] = round_index
        records.append(row)
        payload = dict(workload={k: v for k, v in vars(args).items() if k not in ['output', 'paired']}, rows=records)
        args.output.write_text(json.dumps(payload, indent=2, default=str) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k not in ['graph_batch_mean_us', 'traceback']}), flush=True)
    if not args.paired:
        passing = sorted((r for r in records if r['status'] == 'pass'), key=lambda r: r['median_us'])
        if passing:
            args.output.with_name(args.output.stem + '_best.json').write_text(json.dumps(passing[0], indent=2) + '\n')
    try:
        args.output.with_name(args.output.stem +
                              '_resources.json').write_text(json.dumps(tlx_resources(), indent=2) + '\n')
    except Exception as exc:
        print('resource metadata:', str(exc), flush=True)
    if args.dump_ir:
        module = tlx_source()._optimized_module()
        for name in ('_matrix_decode', '_merge'):
            index = 0
            for cache in getattr(module, name).device_caches.values():
                for kernel in cache[0].values():
                    for kind in ('ttgir', 'amdgcn'):
                        args.output.with_name(args.output.stem + f'_{name}_{index}.{kind}').write_text(kernel.asm[kind])
                    index += 1


if __name__ == '__main__': main()
