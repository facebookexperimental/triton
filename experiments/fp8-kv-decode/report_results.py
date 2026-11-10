"""Render measured comparisons and their exact source rows without GPU dependencies."""
import argparse
import csv
from html import escape
import json
from pathlib import Path

from render_charts import text, write_svg

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--files', nargs='+', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--note', default='GPU graph batch means; lower is faster. Packing and compilation excluded.')
    a = p.parse_args()
    rows = []
    for path in a.files:
        data = json.loads(path.read_text())
        for row in data if isinstance(data, list) else data['results']:
            if 'median_us' in row:
                rows.append(dict(source=str(path.resolve().relative_to(ROOT)), **row))
    if not rows:
        raise ValueError('No measured rows')
    (ROOT / 'results' /
     (a.name + '.json')).write_text(json.dumps({'title': a.title, 'note': a.note, 'rows': rows}, indent=2) + '\n')
    with (ROOT / 'results' / (a.name + '.csv')).open('w') as f:
        fields = ['source', 'backend', 'status', 'median_us', 'p95_batch_mean_us', 'caller_median_us', 'max_abs_error']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    max_us = max(r['median_us'] for r in rows)
    e = [
        text(24, 36, a.title, 25, weight='bold'),
        text(24, 66, a.note, 14),
        text(24, 92, 'gfx1250 DPX / 96 CUs | E4M3 | numerical gate: atol=5e-4, rtol=1e-2', 14),
        text(24, 119, 'Orange: failed numerical gate; latency shown for diagnosis and ineligible for performance wins.',
             13, '#92400e')
    ]
    previous = None
    y = 155
    for row in rows:
        source = Path(row['source']).stem
        if source != previous:
            e.append(text(24, y, source, 15, weight='bold'))
            y += 26
            previous = source
        eligible = row['status'] == 'pass'
        color = ('#15803d' if row['backend'].startswith('tlx') else '#2563eb') if eligible else '#d97706'
        width = 700 * row['median_us'] / max_us
        e.extend([
            text(36, y + 18, row['backend'], 15),
            f'<rect x="260" y="{y}" width="{width:.2f}" height="25" fill="{color}" rx="3"/>',
            text(970, y + 18, f"{row['median_us']:.2f} us", 15, weight='bold'),
            text(1080, y + 18, f"max error {row.get('max_abs_error',float('nan')):.2g}", 13)
        ])
        y += 37
    write_svg(ROOT / 'charts' / (a.name + '.svg'), 1280, y + 24, a.title, a.note, e)
    print(ROOT / 'charts' / (a.name + '.svg'))


if __name__ == '__main__':
    main()
