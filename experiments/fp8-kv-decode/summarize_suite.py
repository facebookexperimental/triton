"""Save the measured case table, a comparison SVG, and a linked Markdown report."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from render_charts import text, write_svg

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directories', nargs='+', type=Path, required=True)
    p.add_argument('--results-root', type=Path, required=True)
    a = p.parse_args()
    rows = []
    prefix = "latest" if any(d.name == "latest" for d in a.directories) else "final"
    for directory in a.directories:
        specs = json.loads((directory / 'suite_spec.json').read_text())
        for spec in specs:
            path = directory / (spec['name'] + '.json')
            if not path.exists(): continue
            measured = json.loads(path.read_text())
            tlx = next((r for r in measured if r['backend'] == 'tlx_auto'), None)
            if tlx is None: continue
            candidates = [r for r in measured if not r['backend'].startswith('tlx') and r['status'] == 'pass']
            if spec['name'] in ('gqa_shared', 'mha_shared'):
                family = spec['name'].removesuffix('_shared')
                for tuned in a.results_root.glob(f'{prefix}_splits_{family}_*.json'):
                    candidates.extend(r for r in json.loads(tuned.read_text()) if r['status'] == 'pass')
            ref = min(candidates, key=lambda r: r['median_us']) if candidates else None
            baseline = next((r for r in measured if r['backend'] == 'tlx_baseline' and r['status'] == 'pass'), None)
            work = tlx['workload']
            rows.append(
                dict(
                    case=spec['name'], stage=spec['stage'], scales=work['scales'], tokens=work['tokens'],
                    length=work['length'], heads=work['heads'], kv_heads=work['kv_heads'], dim=work['dim'],
                    window=work['window'], fp8_q=work['fp8_q'], tlx_status=tlx['status'], tlx_us=tlx.get('median_us'),
                    max_abs_error=tlx.get('max_abs_error'), ref=ref['backend'] if ref else None,
                    ref_us=ref['median_us'] if ref else None,
                    speedup=ref['median_us'] / tlx['median_us'] if ref else None,
                    baseline_us=baseline['median_us'] if baseline else None, config=tlx.get('selected_config'),
                    source=str(path.resolve().relative_to(ROOT)), external_statuses={
                        r['backend']: r['status']
                        for r in measured
                        if not r['backend'].startswith('tlx')
                    }))
    artifact = ROOT / 'results/2026-11-05_step-06_final_comparison'
    artifact.with_suffix('.json').write_text(json.dumps(rows, indent=2) + '\n')
    fields = [
        'case', 'scales', 'tokens', 'length', 'heads', 'kv_heads', 'dim', 'window', 'fp8_q', 'tlx_status', 'tlx_us',
        'ref', 'ref_us', 'speedup', 'baseline_us', 'max_abs_error', 'source'
    ]
    with artifact.with_suffix('.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    e = [
        text(24, 35, 'TLX FP8 KV decode: final measured comparisons', 26, weight='bold'),
        text(24, 64, 'Synthetic mappings | E4M3 per-head scales | same compiler | lower latency is better', 15),
        text(
            24, 88,
            'Shared/unit scale cases are restricted controls. Failed numerical gates are excluded from best-valid comparisons.',
            13),
        text(
            24, 111,
            'Bars are normalized within each row. Ratios below 1.05 are small margins, not a broad performance guarantee.',
            13),
        text(24, 148, 'Case / scales', 15, weight='bold'),
        text(330, 148, 'TLX / best valid external path', 15, weight='bold'),
        text(860, 148, 'TLX us', 15, weight='bold'),
        text(960, 148, 'Reference us', 15, weight='bold'),
        text(1095, 148, 'Reference / TLX', 15, weight='bold')
    ]
    y = 170
    for row in rows:
        t = row['tlx_us']
        r = row['ref_us']
        ratio = row['speedup']
        e.extend([text(24, y + 14, row['case'], 13, weight='bold'), text(24, y + 32, row['scales'], 11, '#475569')])
        if r is not None:
            span = max(t, r)
            e.extend([
                f'<rect x="330" y="{y}" width="{440*t/span:.2f}" height="13" fill="#15803d"/>',
                f'<rect x="330" y="{y+17}" width="{440*r/span:.2f}" height="13" fill="#2563eb"/>',
                text(338, y + 28, row['ref'], 10, '#ffffff'),
                text(960, y + 20, f'{r:.2f}', 14),
                text(1095, y + 20, f'{ratio:.2f}x', 15, '#166534' if ratio >= 1.05 else '#92400e')
            ])
        else:
            e.append(text(330, y + 22, 'No eligible external comparison on this contract/gate', 12, '#92400e'))
        e.append(text(860, y + 20, f'{t:.2f}' if t is not None else row['tlx_status'], 14))
        y += 44
    write_svg(ROOT / 'charts/2026-11-05_step-06_final_comparison.svg', 1280, y + 20, 'Final FP8 KV decode comparison',
              'Each row uses its own bar scale; raw numerical and timing data are linked in Markdown.', e)
    report = [
        '# TLX FP8 KV decode results', '',
        f'{sum(r["tlx_status"]=="pass" for r in rows)}/{len(rows)} synthetic benchmark cases passed the unchanged FP32-oracle gate (`atol=5e-4, rtol=1e-2`).',
        '',
        'The format is plain E4M3FN with per-head scales. Exact model head mapping, T/mask semantics, SWA window and output projection remain undisclosed.',
        '',
        '[Comparison chart](charts/2026-11-05_step-06_final_comparison.svg) · [CSV](results/2026-11-05_step-06_final_comparison.csv) · [Raw rows](results/2026-11-05_step-06_final_comparison.json) · [Reproduction](REPRODUCE.md) · [Experiment log](FP8_KV_DECODE_ATTN.md)',
        '',
        'Base scalar controls use shared head scales so all BF16-Q reference paths can participate. Native unequal-per-head comparisons use vLLM unified. Full-FP8 unit-scale paths from AITER/TokenSpeed/vLLM fail this project\'s tighter numerical gate; their diagnostic latency and error remain in raw JSON, not in the eligible-reference ranking. SGLang/vLLM decode base controls include the best measured split count among4/16/64.',
        '', 'Paired order reversals (three runs per implementation, medians of graph batch means):', '',
        '| Case | TLX µs | AITER µs | Ratio |', '| --- | ---: | ---: | ---: |'
    ]
    for case in ('gqa', 'mha', 'long', 'window'):
        path = a.results_root / f'{prefix}_repeat_{case}.json'
        if not path.exists(): continue
        data = json.loads(path.read_text())
        tv = statistics.median(r['median_us'] for r in data if r['backend'] == 'tlx_auto' and r['status'] == 'pass')
        rv = statistics.median(r['median_us']
                               for r in data
                               if r['backend'] == 'aiter_unified' and r['status'] == 'pass')
        report.append(f'| {case} | {tv:.2f} | {rv:.2f} | {rv/tv:.3f}x |')
    report += [
        '',
        'The long-context and window ratios are small margins. Page128 and other near-equal results should be treated as competitive rather than universal wins. Clocks are unlocked; these are standalone kernel paths on one96-CU DPX agent, with repeated same-input graph timing. Reported p95 is across graph batch means. Packing and compilation are excluded. Caller timings are separate.',
        '', '| Case | Scale control | TLX µs | Best eligible external path | External µs | Ratio |',
        '| --- | --- | ---: | --- | ---: | ---: |'
    ]
    for r in rows:
        label = f'[{r["case"]}]({r["source"]})'
        ref = r['ref'] or 'No eligible path'
        rt = f'{r["ref_us"]:.2f}' if r['ref_us'] else '—'
        ratio = f'{r["speedup"]:.2f}x' if r['speedup'] else '—'
        report.append(f'| {label} | {r["scales"]} | {r["tlx_us"]:.2f} | {ref} | {rt} | {ratio} |')
    report += [
        '',
        'Retained changes: grouped WMMA, packed-page TDM double buffering, full-page mask peeling, FP16-scaled probabilities with FP32 softmax, tuned split/warp policy, direct single-split output, reusable workspace and bounded FP16 raw-V partials at high split counts.',
        '',
        'Final ATT captures and analyzer reports are under `results/2026-11-05_optimization/att_final_{gqa,mha,long}*`. Baseline waits motivated the work; final long-context trace still shows layout-permutation and explicit-wait costs. ATT fractions describe selected instruction samples, not whole-kernel utilization. Requested PMC counters returned zero and cannot confirm utilization or bank conflicts.',
        '',
        'Rejected attempts, compiler failures, earlier correctness failures, exact revisions and resource counts remain in the experiment ledger and snapshots. The production SGLang/vLLM dispatch has not been changed.'
    ]
    totals = dict(tests=0, failures=0, errors=0, skipped=0)
    for name in ('correctness_final_multiseed.xml', 'correctness_d128_highsplits.xml'):
        xml = a.results_root / name
        if xml.exists():
            suite = next(ET.parse(xml).getroot().iter('testsuite'))
            for key in totals:
                totals[key] += int(suite.attrib.get(key, 0))
    report.insert(
        4,
        f'Correctness: {totals["tests"]-totals["skipped"]-totals["failures"]-totals["errors"]} passed, {totals["skipped"]} skipped, {totals["failures"]} failures, {totals["errors"]} errors across the multiseed suite and additional D128 high-split cases. Includes three seeds for poisoned-tail/peaked/workspace cases.'
    )
    audit = a.results_root / 'remote_head_audit.json'
    if audit.exists():
        heads = json.loads(audit.read_text())
        report += [
            '',
            'Upstream main revisions were audited and refreshed before this report. Original pinned measurements remain preserved. Current comparison pins:',
            ''
        ]
        for repo, data in heads.items():
            sha = data.get('head')
            if sha: report.append(f'- {repo}: `{sha}`')
        report += [
            '',
            'TokenSpeed MHA decode is unchanged in the audit. SGLang changes host core lookup; vLLM adds verify paths and tuning for other RDNA architectures. New AITER block-scaled MXFP8 paths use a different scale contract; refreshed unified/dense E4M3 paths were rerun. See upstream_diff_audit.json and the saved diffs in the result directory.'
        ]
    lean_files = list(a.results_root.glob(f'{prefix}_lean_*.json'))
    if lean_files:
        report += [
            '', 'Optional SGLang Lean Attention controls (wrapper metadata/lock-reset GPU work included):', '',
            '| Case | Status under project gate | GPU invocation µs |', '| --- | --- | ---: |'
        ]
        for path in sorted(lean_files):
            for row in json.loads(path.read_text()):
                report.append(
                    f'| [{path.stem}](results/2026-11-05_optimization/{path.name}) | {row["status"]} | {row.get("median_us",float("nan")):.2f} |'
                )
    (ROOT / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    print(f'saved {len(rows)} cases')


if __name__ == '__main__': main()
