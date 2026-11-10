"""Render the focused framework comparison from saved, independently repeated data.

CPU-only reporting. Run in the experiment container for matplotlib.
"""
import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/2026-11-05_framework_decode'
STEM = '2026-11-05_step-09_framework_decode'
FAMILIES = ('tlx', 'vllm_native', 'sglang_native', 'aiter_shared_operator', 'vllm_production')
LABELS = ('TLX auto', 'vLLM best tested', 'SGLang best tested', 'AITER shared operator', 'vLLM stock 16 segments')


def main():
    combined = {}
    for phase in ('final', 'fp8_validated', 'framework_routes'):
        for case in json.loads((DATA / phase / 'summary.json').read_text()):
            case[
                'raw_file'] = f'results/2026-11-05_framework_decode/{phase}/{case["workload"]["case"]}_verification.json'
            combined[case['workload']['case']] = case
    assert len(combined) == 18
    rows = []
    for case in combined.values():
        spec = case['workload']
        configs = dict(tlx=dict(backend='tlx_auto', splits=16, block_n=64), **case['selected'])
        prod = dict(backend='vllm_unified_3d', splits=16, block_n=64)
        if any(all(r[k] == v for k, v in prod.items()) for r in case['verification']): configs['vllm_production'] = prod
        for family, cfg in configs.items():
            runs = [r for r in case['verification'] if all(r[k] == v for k, v in cfg.items())]
            assert len(runs) == 3 and all(r['status'] == 'pass' for r in runs), (spec['case'], family)
            times = [r['median_us'] for r in runs]
            row = dict(**spec, family=family, **cfg, median_us=statistics.median(times), min_round_us=min(times),
                       max_round_us=max(times), rounds=3, max_abs_error=max(r['max_abs_error'] for r in runs),
                       raw_file=case['raw_file'])
            if 'reference_config' in runs[0]: row['reference_config'] = runs[0]['reference_config']
            rows.append(row)
    for row in rows:
        tlx = next(r['median_us'] for r in rows if r['case'] == row['case'] and r['family'] == 'tlx')
        row['reference_over_tlx'] = row['median_us'] / tlx
    (ROOT / 'results' / f'{STEM}.json').write_text(json.dumps(rows, indent=2) + '\n')
    fields = [
        'case', 'heads', 'kv_heads', 'dim', 'length', 'window', 'scales', 'fp8_q', 'family', 'backend', 'splits',
        'block_n', 'median_us', 'min_round_us', 'max_round_us', 'reference_over_tlx', 'rounds', 'max_abs_error',
        'raw_file'
    ]
    with (ROOT / 'results' / f'{STEM}.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(r['case'], r['family']): r for r in rows}

    def cell(case, family):
        row = lookup.get((case, family))
        return f'{row["median_us"]:.2f}' if row else '—'

    report = [
        '# SGLang/vLLM FP8 KV decode comparison — 2026-11-05', '',
        'TLX is faster than the eligible native SGLang/vLLM paths in these tested synthetic cases. '
        'AITER-backed shared-scale controls are much closer: long GQA and SWA are near parity, '
        'and **AITER wins the D128 control (12.23 µs versus TLX15.48 µs; TLX takes1.27x as long)**. '
        '**The earlier vLLM 2D-only numbers do not establish an advantage over its best decode path; this report supersedes those claims.**',
        '', f'[Chart](charts/{STEM}.svg) · [PNG](charts/{STEM}.png) · [CSV](results/{STEM}.csv) · '
        f'[Data](results/{STEM}.json) · [Manifest](results/2026-11-05_framework_decode/manifest.json) · [Experiment log](FP8_KV_DECODE_ATTN.md)',
        '', '## Matched BF16-Q shared-scale controls', '',
        'Same scalar K/V scale is repeated across heads so scalar-only sources can participate. '
        'This is a restricted control. Latencies are µs; each value is the median of three fresh, order-rotated round medians.',
        '',
        '| Case (Hq/Hkv, D, L) | TLX | vLLM best tested | SGLang best tested | AITER shared operator | vLLM stock 3D |',
        '| --- | ---: | ---: | ---: | ---: | ---: |'
    ]
    for case in combined:
        s = combined[case]['workload']
        if s['scales'] != 'shared': continue
        label = f'{s["heads"]}/{s["kv_heads"]}, D{s["dim"]}, L{s["length"]}' + (f', W{s["window"]}'
                                                                                if s['window'] else '')
        report.append('| ' + label + ' | ' + ' | '.join(cell(case, f) for f in FAMILIES) + ' |')
    report += [
        '', 'AITER is one deduplicated underlying operator, not an independent SGLang run and vLLM run. '
        'The vLLM ROCm AITER unified backend can route causal gfx1250 shuffled-cache decode to this operator; '
        'the tested call uses packed 5D cache and BF16 Q. SGLang also uses AITER, but its unified FP8-KV path may quantize Q '
        'and its vectorized 5D path calls a CDNA-only decoder at this revision. Those routes are not automatically equivalent to this BF16-Q control.',
        '', '## Native unequal per-KV-head scales', '',
        'Separate, unequal K and V scale vectors are used. SGLang standard/Lean and legacy vLLM accept scalar scales, '
        'so no native per-head timing is attributed to them. vLLM unified receives native per-token/head scale-cache views '
        'with zero strides across token/page axes, preserving the constant-per-head contract.', '',
        '| Case (Hq/Hkv, D, L) | TLX µs | vLLM best µs | vLLM/TLX | vLLM configuration |',
        '| --- | ---: | ---: | ---: | --- |'
    ]
    for case in combined:
        s = combined[case]['workload']
        if s['scales'] != 'per-head': continue
        r = lookup[(case, 'vllm_native')]
        report.append(
            f'| {s["heads"]}/{s["kv_heads"]}, D{s["dim"]}, L{s["length"]}' +
            (f', W{s["window"]}' if s['window'] else '') +
            f' | {cell(case,"tlx")} | {cell(case,"vllm_native")} | {r["reference_over_tlx"]:.2f}x | '
            f'`{r["backend"]}`, S{r["splits"]}, tile {r.get("reference_config",{}).get("kv_tile",r["block_n"])} |')
    report += [
        '', '## Full FP8 unit-scale controls', '', '| Case | TLX µs | SGLang eligible µs | vLLM eligible µs |',
        '| --- | ---: | ---: | ---: |'
    ]
    for case in ('gqa_fp8_unit', 'mha_fp8_unit'):
        report.append(f'| {case} | {cell(case,"tlx")} | {cell(case,"sglang_native")} | {cell(case,"vllm_native")} |')
    report += [
        '', 'SGLang GQA passes with S32 and unchanged FP8 Q payload; a common Q scale can be folded into the '
        'softmax scale (unit scale here). This is operator evidence, not a claim that a production backend advertises this Q dtype. '
        'SGLang MHA and legacy vLLM MHA fail AMD lowering with FP8 Q on this build. vLLM GQA/unified and Lean FP8-Q controls '
        'fail the unchanged numerical gate. Failures and diagnostic timings are preserved, excluded from the eligible ranking. '
        'These unit-scale controls do not prove arbitrary per-head full-FP8 support.', '',
        '## Selection and reproducibility', '', '- Source pins: vLLM `86fdabec0c82d0470c160aa4df7602ac22c207e6`, '
        'SGLang `9b6c796151f836907fbf93265984006ecfd6a790`, AITER `796ff6a35588058516bcaf13129bd0f0b22fb9ae`. '
        '[Pinned paths and framework routing audit](REFERENCES.md).',
        '- Native candidates: ordinary SGLang split decode, optional persistent Lean; legacy vLLM decode; unified 2D and 3D; '
        'optional tensor descriptors. Split sweep 4/8/16/32/64/128/256; tuned unified decode tiles32/64/128. '
        'KV-only TD tiles32/64 and splits4/16/64/128/256, bounded by context/tile policy. SGLang scalar split controls use source defaults besides splits.',
        '- Stock vLLM production-style 3D: sixteen segments, FP32 output/max/sum scratch, threshold128/KV-heads, '
        'default tile selection (BF16 decode16, FP8 decode32), descriptors disabled by default on ROCm. '
        'The 2D path is also retained because it wins SWA128 with unequal scales.',
        '- `_3d_tuned` changes only a copied Python launcher tile helper. `_kvtd3d`/`_kvtd2d` additionally disable '
        'Q/output descriptors while keeping KV descriptors. These are experimental launch configurations; all upstream GPU bodies are unchanged. '
        'Unmodified opt-in full-TD MHA fails AMD shared-layout lowering; the KV-only control compiles and can win shared-scale MHA.',
        '- Every selected candidate passes the same independent FP32 oracle on identical quantized payloads/page mappings: '
        '`atol=5e-4, rtol=1e-2`. Selection uses five graph-batch samples × ten replays; final verification uses '
        'three fresh rounds of fifteen samples × twenty replays. CSV includes round range and max absolute error. '
        'Selection minima are not final reported timings.',
        '- gfx1250, one96-CU/144-GiB DPX agent, Torch2.12/ROCm10.1, source-built TLX3.9 compiler shared across sources. '
        'GPU0 was checked idle; all GPU jobs ran sequentially; clocks unlocked. Repeated inputs are a warm-cache protocol. '
        'Compilation, quantization, input packing and oracle setup excluded; SGLang Lean metadata/lock-reset GPU work included. '
        'No full-framework serving or end-to-end inference claim.',
        '- Hq/Hkv mappings, T/mask semantics and actual SWA window are undisclosed. These explicit assumptions cover '
        'one-query decode, page64, D64/D128, L2304..256K and window128. They are not exact model acceptance. '
        'Attention output precedes the undefined projection.',
        '- [Reproduction commands](REPRODUCE.md). Raw selection/repeat JSON, compiler logs and source snapshots are under '
        '`results/2026-11-05_framework_decode/`. TLX kernel files are hash-identical to the preceding optimization report.',
        '', '## Selected configurations', '',
        '| Case | Family | Backend | Splits | Actual KV tile | µs | Round range µs |',
        '| --- | --- | --- | ---: | ---: | ---: | --- |'
    ]
    for r in rows:
        if r['family'] == 'tlx': continue
        tile = r.get('reference_config', {}).get('kv_tile', 'source default')
        report.append(f'| {r["case"]} | {r["family"]} | `{r["backend"]}` | {r["splits"]} | {tile} | '
                      f'{r["median_us"]:.2f} | {r["min_round_us"]:.2f}–{r["max_round_us"]:.2f} |')
    (ROOT / 'FRAMEWORK_COMPARISON.md').write_text('\n'.join(report) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(18, 9), gridspec_kw={'width_ratios': [1.4, 1.3, 0.9]})
    colors = {'vllm_native': '#2779bd', 'sglang_native': '#b95b25', 'aiter_shared_operator': '#33805b'}
    for ax, scale, title in zip(
            axes, ('shared', 'per-head', 'unit'),
        ('BF16 Q • shared KV scales', 'BF16 Q • unequal per-head KV scales', 'FP8 QKV • unit scales')):
        cases = [c for c in combined if combined[c]['workload']['scales'] == scale]
        families = [f for f in colors if any((c, f) in lookup for c in cases)]
        for j, family in enumerate(families):
            for i, case in enumerate(cases):
                r = lookup.get((case, family))
                if not r: continue
                pos = i + (j - (len(families) - 1) / 2) * .22
                ax.barh(pos, r['reference_over_tlx'], height=.20, color=colors[family],
                        label=LABELS[FAMILIES.index(family)] if i == 0 else None)
                ax.text(r['reference_over_tlx'] * 1.04, pos, f'{r["reference_over_tlx"]:.2f}×', va='center', fontsize=9)
        labels = []
        for case in cases:
            s = combined[case]['workload']
            r = lookup[(case, 'tlx')]
            labels.append(f'{s["heads"]}/{s["kv_heads"]} D{s["dim"]} L{s["length"]}' +
                          (f' W{s["window"]}' if s['window'] else '') + f'\nTLX {r["median_us"]:.2f} µs')
        ax.set_yticks(range(len(cases)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, pad=15)
        ax.axvline(1, color='#333', linestyle='--', linewidth=1)
        ax.set_xscale('log')
        ax.set_xlim(.55, 30)
        ax.set_xticks([1, 2, 5, 10, 20], ['1×', '2×', '5×', '10×', '20×'])
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
        ax.set_xlabel('Reference latency / TLX latency')
        if families: ax.legend(loc='lower right', fontsize=8)
        if scale == 'unit':
            ax.text(.1, .4, 'MHA: no eligible external path\non this compiler/gate.\nFailed attempts preserved.',
                    transform=ax.transAxes, fontsize=10)
    fig.suptitle(
        'FP8 KV decode: strongest passing tested source paths\nGFX1250 • three fresh rounds • below1×: reference faster; above1×: TLX faster',
        fontsize=16)
    fig.text(
        .02, .018,
        'Synthetic one-query/page64 cases; shared/unit scales are restricted controls. AITER is a deduplicated framework-backed operator.\n'
        'vLLM includes launch-tuned 3D and KV-only descriptors; production16-segment timings and exact configs are in the report. GPU event timing; packing excluded.',
        fontsize=10)
    fig.tight_layout(rect=[0, .075, 1, .91])
    for suffix in ('svg', 'png'):
        fig.savefig(ROOT / 'charts' / f'{STEM}.{suffix}', dpi=150)
    print(
        f'Wrote {len(combined)} cases, {len(rows)} verified configuration rows; {len(rows)*3} passing verification invocations'
    )


if __name__ == '__main__': main()
