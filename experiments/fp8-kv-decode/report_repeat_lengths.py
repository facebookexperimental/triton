"""CPU-only rendering of the November9 three-length repeat and its provenance."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/2026-11-09_repeat_lengths'
STEM = '2026-11-09_step-10_repeat_lengths'


def main():
    rows = []
    for phase in ('shared_verified', 'native'):
        for case in json.loads((DATA / phase / 'summary.json').read_text()):
            spec = case['workload']
            choices = dict(tlx=dict(backend='tlx_auto', splits=16, block_n=64), **case['selected'])
            choices['vllm_stock3d'] = dict(backend='vllm_unified_3d', splits=16, block_n=64)
            for family, cfg in choices.items():
                runs = [r for r in case['verification'] if all(r[k] == v for k, v in cfg.items())]
                assert len(runs) == 3 and all(r['status'] == 'pass' for r in runs), (spec['case'], family)
                ts = [r['median_us'] for r in runs]
                row = dict(**spec, family=family, backend=cfg['backend'],
                           splits='auto' if family == 'tlx' else cfg['splits'],
                           block_n='auto' if family == 'tlx' else cfg['block_n'], median_us=statistics.median(ts),
                           min_round_us=min(ts), max_round_us=max(ts), rounds=3, max_abs_error=max(r['max_abs_error']
                                                                                                   for r in runs),
                           reference_config=runs[0].get('reference_config'),
                           raw_file=f'results/2026-11-09_repeat_lengths/{phase}/{spec["case"]}_verification.json')
                rows.append(row)
    assert len(rows) == 24 and len({r['case'] for r in rows}) == 6
    (ROOT / 'results' / f'{STEM}.json').write_text(json.dumps(rows, indent=2) + '\n')
    fields = [k for k in rows[0] if k != 'reference_config']
    with (ROOT / 'results' / f'{STEM}.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    lookup = {(r['length'], r['scales'], r['family']): r for r in rows}

    def value(length, scales, family):
        return lookup[(length, scales, family)]['median_us']

    lines = [
        '# Three-length FP8 KV decode repeat — 2026-11-09', '',
        '**Fresh measurements of the documented folded-GQA workload. The exact metadata behind the user-pasted table could not be recovered, '
        'so these are not claimed as an exact reproduction or a measured speedup over that table.**', '',
        f'[Chart](charts/{STEM}.svg) · [PNG](charts/{STEM}.png) · [CSV](results/{STEM}.csv) · '
        '[Manifest](results/2026-11-09_repeat_lengths/manifest.json) · [Experiment log](FP8_KV_DECODE_ATTN.md)', '',
        'Q=[1,16,64],2 KV heads,D64,page64,one causal decode token,BF16 Q,E4M3FN KV,BF16 output before projection. '
        'K/V use separate non-unit dequant scales. Shared-scale controls repeat each scalar across KV heads; '
        'the native companion uses unequal per-head K/V scales.', '',
        'Shared-scale restricted controls, µs. vLLM best uses the strongest passing configuration selected in the November5 sweep, '
        'including experimental host launch settings; no new tuning occurred. Stock vLLM3D is shown separately.', '',
        '| KV length | TLX auto | AITER unified | vLLM best previously tested | vLLM stock3D |',
        '| --- | ---: | ---: | ---: | ---: |'
    ]
    for length in (2304, 8192, 32768):
        lines.append('| ' + str(length) + ' | ' +
                     ' | '.join(f'{value(length,"shared",f):.2f}'
                                for f in ('tlx', 'aiter_shared_operator', 'vllm_native', 'vllm_stock3d')) + ' |')
    lines += [
        '', 'AITER remains scalar-only for this operator interface; these numbers are restricted controls. '
        'At32K, the launch-tuned vLLM control is slightly faster than AITER. AITER is one underlying kernel operator, '
        'not independently measured SGLang and vLLM servers.', '',
        '| KV length | TLX unequal-head scales µs | vLLM unequal-head scales µs | vLLM/TLX |',
        '| --- | ---: | ---: | ---: |'
    ]
    for length in (2304, 8192, 32768):
        t = value(length, 'per-head', 'tlx')
        v = value(length, 'per-head', 'vllm_native')
        lines.append(f'| {length} | {t:.2f} | {v:.2f} | {v/t:.2f}x |')
    lines += [
        '', 'Each reported value is the median of three fresh order-rotated round medians. '
        'Each round uses fifteen GPU graph-batch measurements of twenty invocations. All72 selected verification '
        'runs pass the unchanged FP32 oracle gate (`atol=5e-4,rtol=1e-2`). Packing, compilation, '
        'quantization and oracle setup are excluded. Repeated inputs follow the same warm-cache protocol; '
        'clocks are unlocked and no hardware utilization claim is made. The CSV and table below retain round ranges.',
        '',
        'The first attempted shared run is excluded: its immediately preceding snapshot showed19% GPU activity and16046MiB '
        'already in use. It is preserved under `shared/`; the reported run is `shared_verified/`, with a fresh idle '
        'snapshot (0% activity,161MiB). The native companion also began idle. This prevents those first samples from '
        'entering the medians; shared host activity and unlocked clocks still limit comparisons across dates.', '',
        'Runtime: same selected gfx1250,96-CU/144-GiB agent,CUDA_VISIBLE_DEVICES=0 in '
        '`adkankar-tlx-opt-20261105`; source-built TLX3.9.0+fb,Torch2.12/ROCm10.1. '
        'TLX and pinned GPU reference source hashes match the November5 manifest. '
        '[Pinned source paths and contract limits](REFERENCES.md). No GPU kernel changes.', '',
        '## Configurations and repeat ranges', '',
        '| KV length | Scales | Path | Splits | Actual KV tile | Median µs | Round range µs |',
        '| --- | --- | --- | --- | --- | ---: | --- |'
    ]
    for r in rows:
        tile = (r['reference_config'] or {}).get('kv_tile', 'auto' if r['family'] == 'tlx' else 'source default')
        lines.append(f'| {r["length"]} | {r["scales"]} | `{r["backend"]}` | {r["splits"]} | {tile} | '
                     f'{r["median_us"]:.2f} | {r["min_round_us"]:.2f}–{r["max_round_us"]:.2f} |')
    lines += [
        '',
        '`_kvtd3d` keeps the upstream GPU bodies but uses KV tensor descriptors without Q/output descriptors and tile64. '
        '`_3d_tuned` changes the Python decode tile helper. These are launch-tuned controls, not stock ROCm defaults. '
        'Stock3D supplies FP32 split scratch,16 segments,threshold128/KV-heads and descriptors disabled. '
        'The nominal split parameter in raw TLX call parameters does not select its actual launch: TLX uses automatic dispatch.',
        '',
        'To repeat, use `compare_framework_decode.py --repeat-only --aiter-companion` with the three shared case names '
        'and `--extend-from results/2026-11-05_framework_decode/framework_routes` (full paths relative to the experiment directory). '
        'Use the three per-head case names with `--repeat-only --extend-from .../final` for the native companion. '
        '[Environment and timing reproduction](REPRODUCE.md).', ''
    ]
    (ROOT / 'REPEAT_2026-11-09.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = {'tlx': 'TLX auto', 'aiter_shared_operator': 'AITER unified', 'vllm_native': 'vLLM previously selected'}
    colors = {'tlx': '#384eb7', 'aiter_shared_operator': '#27805a', 'vllm_native': '#d48028'}
    for ax, scale in zip(axes, ('shared', 'per-head')):
        families = ('tlx', 'aiter_shared_operator', 'vllm_native') if scale == 'shared' else ('tlx', 'vllm_native')
        for j, family in enumerate(families):
            selected = [lookup[(length, scale, family)] for length in (2304, 8192, 32768)]
            vals = [r['median_us'] for r in selected]
            x = [i + (j - (len(families) - 1) / 2) * .23 for i in range(3)]
            errors = [[r['median_us'] - r['min_round_us'] for r in selected],
                      [r['max_round_us'] - r['median_us'] for r in selected]]
            bars = ax.bar(x, vals, .21, label=labels[family], color=colors[family], yerr=errors, capsize=3)
            ax.bar_label(bars, labels=[f'{v:.2f}' for v in vals], padding=4, fontsize=9)
        ax.set_xticks(range(3), ['2304', '8192', '32768'])
        ax.set_xlabel('KV length')
        ax.set_ylabel('GPU invocation µs')
        ax.set_ylim(0, 15)
        ax.set_title('Shared-scale restricted controls' if scale == 'shared' else 'Unequal per-head KV scales')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', fontsize=9)
    fig.suptitle('Fresh FP8 KV decode repeats • 2026-11-09\nGQA16/2,D64,page64,BF16 Q • lower is faster', fontsize=14)
    fig.text(
        .03, .018, 'Three fresh rounds; error bars span round medians. Packing/compilation excluded; clocks unlocked.\n'
        'Sources/configurations fixed to November5. The original pasted table has unidentified metadata; no exact before/after claim.',
        fontsize=9)
    fig.tight_layout(rect=[0, .1, 1, .9])
    for suffix in ('svg', 'png'):
        fig.savefig(ROOT / 'charts' / f'{STEM}.{suffix}', dpi=150)
    print(
        json.dumps(
            {
                str(length): {
                    f: round(value(length, 'shared', f), 3)
                    for f in ('tlx', 'aiter_shared_operator', 'vllm_native', 'vllm_stock3d')
                }
                for length in (2304, 8192, 32768)
            }, indent=2))


if __name__ == '__main__': main()
