"""Render a frozen/candidate/AITER paired experiment, without executing GPUs."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--stem', type=Path, required=True)
    p.add_argument('--title', required=True)
    a = p.parse_args()
    data = json.loads(a.input.read_text())
    groups = collections.defaultdict(list)
    for row in data['rows']:
        groups[row['backend']].append(row)
    records = []
    for name, rows in groups.items():
        assert len(rows) == 3 and all(r['status'] == 'pass' for r in rows)
        times = [r['median_us'] for r in rows]
        records.append(
            dict(backend=name, median_us=statistics.median(times), min_round_us=min(times), max_round_us=max(times),
                 max_abs_error=max(r['max_abs_error'] for r in rows), source_file=str(a.input)))
    a.stem.parent.mkdir(exist_ok=True, parents=True)
    a.stem.with_suffix('.json').write_text(json.dumps(dict(workload=data['workload'], rows=records), indent=2) + '\n')
    with a.stem.with_suffix('.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [r['median_us'] for r in records]
    errors = [[r['median_us'] - r['min_round_us'] for r in records],
              [r['max_round_us'] - r['median_us'] for r in records]]
    colors = {'tlx_frozen': '#777777', 'tlx_frozen_launch': '#b8b8b8', 'aiter_unified': '#2c815b'}
    bars = ax.bar(range(len(vals)), vals, color=[colors.get(r['backend'], '#3155b8') for r in records], yerr=errors,
                  capsize=5)
    ax.bar_label(bars, labels=[f'{v:.2f} µs' for v in vals], padding=7)
    labels = [
        'Frozen TLX auto' if r['backend'] == 'tlx_frozen' else 'Frozen, matched launch' if r['backend']
        == 'tlx_frozen_launch' else 'AITER unified' if r['backend'] == 'aiter_unified' else 'TLX candidate'
        for r in records
    ]
    ax.set_xticks(range(len(vals)), labels)
    ax.set_ylabel('GPU invocation µs • lower is faster')
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.set_title(a.title)
    fig.text(
        .04, .02,
        'Three fresh order-rotated rounds; error bars span round medians. Same gfx1250 agent/compiler/input.\nShared-scale restricted control; packing/compilation excluded; clocks unlocked. Numerical gate unchanged.',
        fontsize=9)
    fig.tight_layout(rect=[0, .10, 1, 1])
    for suffix in ['svg', 'png']:
        fig.savefig(a.stem.with_suffix('.' + suffix), dpi=150)
    print(json.dumps(records, indent=2))


if __name__ == '__main__': main()
