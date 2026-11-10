"""Render saved frozen/current/AITER audit data; no GPU execution."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--stem', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--title', required=True)
    a = p.parse_args()
    data = json.loads(a.input.read_text())
    rows = []
    for case in data['cases']:
        groups = collections.defaultdict(list)
        for r in case['rows']:
            groups[r['backend']].append(r)
        row = dict(**case['workload'], config=case['selected_config'])
        for family, name in [('frozen', 'tlx_frozen'), ('tlx', 'tlx_auto'), ('aiter', 'aiter_unified')]:
            runs = groups.get(name, [])
            eligible = bool(runs) and all(r['status'] == 'pass' for r in runs)
            row[family + '_status'] = 'pass' if eligible else 'failed' if runs else 'not_applicable'
            row[family + '_rounds'] = len(runs)
            if eligible:
                times = [r['median_us'] for r in runs]
                row.update({
                    family + '_us': statistics.median(times), family + '_min_us': min(times), family + '_max_us':
                    max(times)
                })
            elif runs:
                row[family + '_reason'] = next(r.get('reason', '') for r in runs if r['status'] != 'pass')
        if row['tlx_status'] == 'pass':
            for family in ('frozen', 'aiter'):
                if row[family + '_status'] == 'pass': row[family + '_over_tlx'] = row[family + '_us'] / row['tlx_us']
        ratio = row.get('aiter_over_tlx')
        row['comparison'] = 'no_eligible_aiter' if ratio is None else 'tlx_faster' if ratio > 1.03 else 'aiter_faster' if ratio < .97 else 'within_3_percent'
        rows.append(row)
    output = ROOT / 'results' / a.stem.name
    output.with_suffix('.json').write_text(json.dumps(rows, indent=2) + '\n')
    fields = [
        'case', 'heads', 'kv_heads', 'dim', 'length', 'page', 'tokens', 'window', 'fp8_q', 'scales', 'tlx_status',
        'tlx_us', 'tlx_min_us', 'tlx_max_us', 'frozen_status', 'frozen_us', 'aiter_status', 'aiter_us', 'aiter_min_us',
        'aiter_max_us', 'frozen_over_tlx', 'aiter_over_tlx', 'comparison'
    ]
    with output.with_suffix('.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    core = [r for r in rows if r['case'] in ('gqa64_l2304', 'mha64_l2304', 'mha128_l2304')]
    colors = {'frozen': '#888888', 'tlx': '#3155b8', 'aiter': '#2c815b'}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, family in enumerate(('frozen', 'tlx', 'aiter')):
        vals = [r.get(family + '_us', 0) for r in core]
        positions = [i + (j - 1) * .24 for i in range(len(core))]
        lo = [v - r.get(family + '_min_us', v) for v, r in zip(vals, core)]
        hi = [r.get(family + '_max_us', v) - v for v, r in zip(vals, core)]
        bars = ax.bar(positions, vals, .22,
                      label={'frozen': 'Frozen TLX', 'tlx': 'Current TLX', 'aiter':
                             'AITER unified'}[family], color=colors[family], yerr=[lo, hi], capsize=3)
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in vals], padding=4, fontsize=10)
    ax.set_xticks(range(len(core)), ['GQA16/2 D64', 'MHA128/128 D64', 'MHA64/64 D128'][:len(core)])
    ax.set_ylabel('GPU invocation µs • lower is faster')
    ax.set_ylim(0, max(r.get(f + '_us', 0) for r in core for f in colors) * 1.2)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title(a.title + '\nCore shared-scale controls, L2304/page64')
    fig.text(
        .03, .025,
        f'{data["rounds"]} round(s); medians of graph batch means; packing/compilation excluded; clocks unlocked.\nSynthetic shape mappings; unchanged FP32 numerical gate. AITER is a scalar-scale restricted operator control.',
        fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, 1])
    for suffix in ('svg', 'png'):
        fig.savefig(a.stem.with_suffix('.' + suffix), dpi=150)
    plt.close(fig)
    charts = []
    for family in ('gqa64', 'mha64', 'gqa128', 'mha128'):
        selected = [r for r in rows if r['case'].startswith(family + '_')]
        if not selected: continue
        fig, ax = plt.subplots(figsize=(10, max(6, len(selected) * .4)))
        for j, ref in enumerate(('frozen', 'aiter')):
            for i, r in enumerate(selected):
                ratio = r.get(ref + '_over_tlx')
                if ratio is None: continue
                pos = i + (j - .5) * .3
                ax.barh(pos, ratio, height=.27, color=colors[ref], label=ref if i == 0 else None)
                ax.text(ratio * 1.02, pos, f'{ratio:.2f}×', va='center', fontsize=8)
        ax.set_yticks(range(len(selected)), [r['case'].removeprefix(family + '_') for r in selected], fontsize=9)
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_xlim(.45, max(3, max(r.get(f + '_over_tlx', 0) for r in selected for f in ('frozen', 'aiter'))) * 1.15)
        ax.axvline(1, color='#333', linestyle='--')
        ax.set_xticks([.5, 1, 2, 3], ['0.5×', '1×', '2×', '3×'])
        ax.set_xlabel('Reference / current TLX latency • above1× favors TLX')
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_title(a.title + ' • ' + family + f'\n{data["rounds"]} round(s), only passing timings shown')
        ax.legend(loc='lower right')
        fig.tight_layout()
        stem = a.stem.with_name(a.stem.name + '_' + family)
        for suffix in ('svg', 'png'):
            fig.savefig(stem.with_suffix('.' + suffix), dpi=150)
        plt.close(fig)
        charts.append(stem.with_suffix('.svg'))
    counts = collections.Counter(r['comparison'] for r in rows)

    def relative(path):
        return str(path.resolve().relative_to(ROOT))

    def number(row, key):
        return f'{row[key]:.2f}' if key in row else '—'

    lines = [
        '# ' + a.title, '', f'{len(rows)} synthetic cases; {data["rounds"]} timing round(s) per implementation. '
        f'TLX correctness: {sum(r["tlx_status"]=="pass"for r in rows)}/{len(rows)} passing cases. '
        f'Eligible AITER comparisons: {sum(r["aiter_status"]=="pass"for r in rows)}. '
        f'At a 3% reporting band: {counts["tlx_faster"]} favor TLX, {counts["within_3_percent"]} near parity, '
        f'{counts["aiter_faster"]} favor AITER. This is the tested matrix, not a universal performance claim.', '',
        f'[Core chart]({relative(a.stem.with_suffix(".svg"))}) · [CSV]({relative(output.with_suffix(".csv"))}) · '
        f'[Data]({relative(output.with_suffix(".json"))}) · [Raw run]({relative(a.input)}) · [Experiment log](FP8_KV_DECODE_ATTN.md)',
        '', '| Case | Frozen TLX µs | Current TLX µs | AITER µs | AITER/current |',
        '| --- | ---: | ---: | ---: | ---: |'
    ]
    for r in core:
        lines.append('| ' + r['case'] + ' | ' + ' | '.join(number(r, k) for k in ('frozen_us', 'tlx_us', 'aiter_us')) +
                     f' | {r.get("aiter_over_tlx",0):.2f}x |')
    lines += [
        '', 'Family charts: ' + ' · '.join(f'[{p.stem.split("_")[-1]}]({relative(p)})' for p in charts), '',
        'The frozen code is the November 9 starting implementation; all comparisons run sequentially on CUDA_VISIBLE_DEVICES=2, '
        'the same 96-CU gfx1250 agent/compiler. Q/O shapes and cache values/page mappings are identical within each case. '
        'AITER uses the pinned unified operator with scalar KV scales. Shared/unit controls restrict scales so it can participate. '
        'Unequal-head scale and noncausal multi-query cases retain native TLX/frozen checks; AITER has no matching adapter for them. '
        'Failed AITER accuracy gates are excluded from performance rankings and retained in raw rows.', '',
        'Working mappings: GQA16/2,D64 orD128; MHA128/128,D64; MHA64/64,D128. Cache lengths 1024..256K, '
        'page64/128, assumed windows37/128. T>1 means noncausal queries over one shared cache, with T=8..8192. '
        'Exact model mappings/mask/window/projection remain undisclosed. BF16 output is before projection.', '',
        'The unchanged oracle uses FP32 attention on actual quantized inputs with atol=5e-4/rtol=1e-2. '
        f'Timing uses {data["samples"]} graph-batch samples of {data["repeats"]} invocations per round. '
        'Compilation, input packing, quantization and oracle setup are excluded. Reported values are medians of round medians; '
        'raw CSV includes round ranges. Repeated inputs are a warm-cache protocol; clocks unlocked. '
        'A one-round scout is provisional and does not establish repeatability.', '',
        '| Case | Current µs | Frozen µs | AITER µs | AITER/current | Result | Current launch |',
        '| --- | ---: | ---: | ---: | ---: | --- | --- |'
    ]
    for r in rows:
        cfg = r['config']
        launch = f'{cfg["variant"]}, S{cfg["num_splits"]}, W{cfg["num_warps"]}, BN{cfg["block_n"]}'
        ratio = f'{r["aiter_over_tlx"]:.2f}x' if 'aiter_over_tlx' in r else r['aiter_status']
        lines.append('| ' + r['case'] + ' | ' + ' | '.join(number(r, k) for k in ('tlx_us', 'frozen_us', 'aiter_us')) +
                     f' | {ratio} | {r["comparison"]} | {launch} |')
    a.report.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(cases=len(rows), classifications=dict(counts)), indent=2))


if __name__ == '__main__': main()
