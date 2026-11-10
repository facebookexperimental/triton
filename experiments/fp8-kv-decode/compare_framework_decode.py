"""Tune and independently repeat applicable SGLang/vLLM FP8 decode paths.

This compares unchanged pinned GPU kernels. Scalar-only paths remain restricted
controls. All candidates face the same numerical gate, including optional paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import traceback
from types import SimpleNamespace

import torch
import triton
from bench_decode import inputs, make_backend, reference, timing, REFERENCES

CASES = [
    ('gqa2304', 16, 2, 64, 2304, 0),
    ('mha2304', 128, 128, 64, 2304, 0),
    ('gqa8192', 16, 2, 64, 8192, 0),
    ('gqa32768', 16, 2, 64, 32768, 0),
    ('gqa262144', 16, 2, 64, 262144, 0),
    ('mha262144', 128, 128, 64, 262144, 0),
    ('swa128', 16, 2, 64, 2304, 128),
    ('d128', 64, 64, 128, 2304, 0),
]


def evaluate(name, splits, x, a, expected, samples, repeats, block_n=64):
    a.splits = splits
    a.block_n = block_n
    row = dict(backend=name, splits=splits, block_n=block_n)
    try:
        fn = make_backend(name, x, a)
        if hasattr(fn, "reference_config"): row["reference_config"] = fn.reference_config
        actual = fn()
        torch.cuda.synchronize()
        error = actual.float() - expected
        row.update(max_abs_error=float(error.abs().max()), rms_error=float(error.square().mean().sqrt()))
        try:
            torch.testing.assert_close(actual.float(), expected, atol=5e-4, rtol=1e-2)
            row.update(status='pass', performance_eligible=True)
        except AssertionError as exc:
            row.update(status='failed_accuracy', performance_eligible=False, reason=str(exc))
        row.update(timing(fn, samples, repeats))
    except NotImplementedError as exc:
        row.update(status='unsupported', reason=str(exc))
    except Exception as exc:
        row.update(status='failed', reason=str(exc), traceback=traceback.format_exc())
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', nargs='+')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--extend-from', type=Path)
    p.add_argument('--legacy-fp8-only', action='store_true',
                   help='Extend prior results with scalar-Q-scale legacy FP8-Q controls')
    p.add_argument('--aiter-companion', action='store_true',
                   help='Add the shared AITER operator used by eligible framework routes')
    p.add_argument('--repeat-only', action='store_true',
                   help='Repeat prior passing selections without running a new tuning sweep')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.repeat_only and not a.extend_from: p.error('--repeat-only requires --extend-from')
    specs = []
    for label, hq, hkv, dim, length, window in CASES:
        for scales in ('shared', 'per-head'):
            specs.append(
                dict(case=f'{label}_{scales}', heads=hq, kv_heads=hkv, dim=dim, length=length, window=window,
                     scales=scales, fp8_q=False))
    for label, hq, hkv in [('gqa_fp8_unit', 16, 2), ('mha_fp8_unit', 128, 128)]:
        specs.append(dict(case=label, heads=hq, kv_heads=hkv, dim=64, length=2304, window=0, scales='unit', fp8_q=True))
    if a.case: specs = [s for s in specs if s['case'] in a.case]
    if a.legacy_fp8_only:
        if not a.extend_from: p.error('--legacy-fp8-only requires --extend-from')
        specs = [s for s in specs if s['fp8_q'] and s['scales'] == 'unit']
    if a.aiter_companion:
        if not a.extend_from: p.error('--aiter-companion requires --extend-from')
        specs = [s for s in specs if s['scales'] == 'shared' and not s['fp8_q']]
    (a.output / 'spec.json').write_text(
        json.dumps(
            dict(
                cases=specs, reference_root=str(REFERENCES), torch=torch.__version__, triton=triton.__version__,
                seed=182, timing='GPU graph batch means; packing/compilation excluded; numerical gate unchanged',
                selection=('Prior passing tuning selections; three fresh order-rotated rounds; no new tuning' if a.
                           repeat_only else 'Median tuning samples then three fresh order-rotated verification rounds'
                           ), extend_from=str(a.extend_from) if a.extend_from else None, repeat_only=a.repeat_only,
                production_vllm='16 segments; threshold128/KV-heads; FP32 scratch; TD disabled by default on ROCm'),
            indent=2) + '\n')
    summary = []
    for spec in specs:
        params = SimpleNamespace(**spec, tokens=1, page=64, causal=True, splits=16, block_n=64, block_m=16, warps=4)
        x = inputs(params)
        expected = reference(x, params)
        records = []
        if a.extend_from:
            records = json.loads((a.extend_from / (spec['case'] + '_tuning.json')).read_text())['rows']
        settings = [('tlx_auto', 16, 64), ('vllm_unified', 16, 64), ('vllm_unified_td2d', 16, 64)]
        split_values = (4, 16, 64, 128) if a.quick else (4, 8, 16, 32, 64, 128, 256)
        for sp in split_values:
            settings.extend([('vllm_unified_3d', sp, 64), ('vllm_unified_td3d', sp, 64)])
        if not a.quick:
            for tile in (32, 64, 128):
                cap = max(16, triton.next_power_of_2(triton.cdiv(spec['length'], tile)))
                settings.extend(('vllm_unified_3d_tuned', sp, tile) for sp in split_values if sp <= cap)
        if spec['scales'] != 'per-head':
            for sp in split_values:
                settings.extend([('sglang', sp, 64), ('vllm_decode', sp, 64)])
            settings.append(('sglang_lean', 16, 64))
        if a.extend_from:
            settings = []
            for tile in (32, 64):
                settings.append(('vllm_unified_kvtd2d', 16, tile))
                cap = max(16, triton.next_power_of_2(triton.cdiv(spec['length'], tile)))
                settings.extend(('vllm_unified_kvtd3d', sp, tile) for sp in (4, 16, 64, 128, 256) if sp <= cap)
        if a.legacy_fp8_only:
            settings = [(name, sp, 64) for name in ('sglang', 'vllm_decode') for sp in split_values]
            settings.append(('sglang_lean', 16, 64))
        if a.aiter_companion: settings = [('aiter_unified', 16, 64)]
        if a.repeat_only: settings = []
        if a.repeat_only:
            (a.output / (spec['case'] + '_tuning.json')).write_text(
                json.dumps(dict(workload=spec, rows=records, inherited_from=str(a.extend_from), new_tuning=False),
                           indent=2) + '\n')
        # Deterministic configuration order. Final verification rotates families.
        broken_td = {}
        for name, sp, tile in settings:
            if name in broken_td:
                row = dict(backend=name, splits=sp, block_n=tile, status='not_retried',
                           reason='Same descriptor geometry already failed compilation; see first failure',
                           first_failure=broken_td[name])
            else:
                row = evaluate(name, sp, x, params, expected, 5, 10, tile)
                if row['status'] == 'failed' and name in ('vllm_unified_td2d', 'vllm_unified_td3d'):
                    broken_td[name] = dict(splits=sp, reason=row['reason'])
            records.append(row)
            (a.output / (spec['case'] + '_tuning.json')
             ).write_text(json.dumps(dict(workload=spec, rows=records), indent=2) + '\n')
        eligible = [r for r in records if r['status'] == 'pass']
        selected = [dict(backend='tlx_auto', splits=16, block_n=64)]
        families = {
            'vllm_native': lambda n: n.startswith('vllm'),
            'sglang_native': lambda n: n.startswith('sglang'),
        }
        if a.aiter_companion: families['aiter_shared_operator'] = lambda n: n == 'aiter_unified'
        choices = {}
        for family, predicate in families.items():
            candidates = [r for r in eligible if predicate(r['backend'])]
            if candidates:
                best = min(candidates, key=lambda r: r['median_us'])
                config = {k: best[k] for k in ('backend', 'splits', 'block_n')}
                choices[family] = config
                selected.append(config)
        # Keep the actual production-style vLLM16-split path visible even if a
        # different split count or TD path won tuning.
        prod = dict(backend='vllm_unified_3d', splits=16, block_n=64)
        if prod not in selected and any(r['backend'] == prod['backend'] and r['splits'] == 16 and r['status'] == 'pass'
                                        for r in records):
            selected.append(prod)
        verified = []
        for round_index in range(3):
            order = selected[round_index % len(selected):] + selected[:round_index % len(selected)]
            if round_index == 1: order = list(reversed(order))
            for cfg in order:
                row = evaluate(cfg['backend'], cfg['splits'], x, params, expected, 15, 20, cfg['block_n'])
                row['round'] = round_index
                verified.append(row)
        payload = dict(workload=spec, selected=choices, verification=verified)
        (a.output / (spec['case'] + '_verification.json')).write_text(json.dumps(payload, indent=2) + '\n')
        summary.append(payload)
        (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(
            json.dumps(
                dict(
                    case=spec['case'], selected=choices, measurements=[{
                        k: r[k]
                        for k in ('backend', 'splits', 'block_n', 'round', 'status', 'median_us')
                        if k in r
                    }
                                                                       for r in verified])), flush=True)
        del x, expected
    print('completed', len(summary), 'cases', flush=True)


if __name__ == '__main__': main()
