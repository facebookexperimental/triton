"""Same-agent frozen/current/AITER matrix with explicit synthetic assumptions."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from bench_decode import inputs, reference, tlx_source
from explore_decode import evaluate


def cases(stage):
    result = []
    families = [('gqa64', 16, 2, 64), ('mha64', 128, 128, 64), ('gqa128', 16, 2, 128), ('mha128', 64, 64, 128)]

    def add(label, hq, hkv, dim, length=2304, page=64, tokens=1, window=0, fp8_q=False, scales='shared'):
        result.append(
            dict(case=label, heads=hq, kv_heads=hkv, dim=dim, length=length, page=page, tokens=tokens, window=window,
                 fp8_q=fp8_q, scales=scales, causal=tokens == 1))

    for label, hq, hkv, dim in families:
        if stage == 'd128' and dim != 128: continue
        for length in [1024, 2304, 8192, 32768, 131072, 262144]:
            add(f'{label}_l{length}', hq, hkv, dim, length)
        for length in [2304, 8192, 32768]:
            add(f'{label}_p128_l{length}', hq, hkv, dim, length, page=128)
        for window in [37, 128]:
            add(f'{label}_w{window}', hq, hkv, dim, window=window)
        add(f'{label}_unequal', hq, hkv, dim, scales='per-head')
        add(f'{label}_fp8_unit', hq, hkv, dim, fp8_q=True, scales='unit')
        add(f'{label}_fp8_unequal', hq, hkv, dim, fp8_q=True, scales='per-head')
        if stage == 'full':
            for tokens in [8, 32, 128, 512, 2048, 8192]:
                add(f'{label}_t{tokens}', hq, hkv, dim, tokens=tokens, scales='per-head')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['d128', 'full'], default='d128')
    p.add_argument('--case', nargs='+')
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--repeats', type=int, default=10)
    args = p.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    specs = cases(args.stage)
    if args.case: specs = [s for s in specs if s['case'] in args.case]
    payload = dict(specs=specs, rounds=args.rounds, samples=args.samples, repeats=args.repeats,
                   timing='GPU graph event batch means, setup excluded, clocks unlocked', cases=[])
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    for spec in specs:
        a = SimpleNamespace(**spec, splits=16, warps=4, block_n=64, block_m=16)
        x = inputs(a)
        expected = reference(x, a)
        cfg = tlx_source().select_decode_config(x.q, x.kp, max_context_len=a.length, query_length=a.tokens,
                                                window_size=a.window)
        record = dict(workload=spec, selected_config=cfg, rows=[])
        names = ['tlx_frozen', 'tlx_auto'] + (['aiter_unified'] if a.scales != 'per-head' and a.tokens == 1 else [])
        for round_index in range(args.rounds):
            order = names[round_index % len(names):] + names[:round_index % len(names)]
            if round_index == 1: order = list(reversed(order))
            for name in order:
                row = evaluate(name, x, a, expected, args.samples, args.repeats)
                row['round'] = round_index
                record['rows'].append(row)
        payload['cases'].append(record)
        args.output.write_text(json.dumps(payload, indent=2) + '\n')
        print(
            json.dumps(
                dict(
                    case=spec['case'], selected_config=cfg,
                    measurements=[{k: r[k]
                                   for k in ['backend', 'status', 'median_us', 'round']
                                   if k in r}
                                  for r in record['rows']])), flush=True)
        del x, expected


if __name__ == '__main__': main()
