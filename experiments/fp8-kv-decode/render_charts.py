"""Regenerate the initial source-capability and correctness SVG snapshots.

Uses Python's standard library; no GPU or plotting dependencies required.
Future measurement steps should save new dated data/charts, preserving these.
"""

import argparse
import csv
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLORS = {
    "yes": ("#dcfce7", "#166534"),
    "limited": ("#fef3c7", "#92400e"),
    "no": ("#fee2e2", "#991b1b"),
    "unknown": ("#e2e8f0", "#475569"),
}


def text(x, y, value, size=15, color="#0f172a", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}">{escape(str(value))}</text>')


def write_svg(path, width, height, title, description, elements):
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
        *elements,
        "</g></svg>\n",
    ]
    path.write_text("\n".join(svg))


def capabilities():
    data = json.loads((ROOT / "results/2026-10-27_step-00_capabilities.json").read_text())
    elements = [
        text(28, 38, "FP8 KV decode: reference comparison", 26, weight="bold"),
        text(28, 65, f"{data['date']} | gfx1250 target | source capabilities, not a performance ranking", 15),
        text(28, 91, "Kernel latency / speedup: UNMEASURED for every candidate", 16, "#92400e", "bold"),
    ]
    left, cell_width, top, row_height = 245, 155, 140, 64
    for col, label in enumerate(data["columns"]):
        elements.append(text(left + col * cell_width + 9, top, label, 14, weight="bold"))
    for row, entry in enumerate(data["rows"]):
        y = top + 18 + row * row_height
        elements += [
            text(28, y + 23, entry["name"], 17, weight="bold"),
            text(28, y + 44, entry["revision"], 13, "#475569")
        ]
        if len(entry["cells"]) != len(data["columns"]):
            raise ValueError(f"Wrong column count for {entry['name']}")
        for col, (label, status) in enumerate(entry["cells"]):
            fill, foreground = COLORS[status]
            x = left + col * cell_width
            elements += [
                f'<rect x="{x}" y="{y}" width="149" height="52" rx="5" fill="{fill}"/>',
                text(x + 9, y + 31, label, 14, foreground)
            ]
    legend_y = top + 18 + len(data["rows"]) * row_height + 20
    for index, (status, label) in enumerate([
        ("yes", "Supported / validated as labeled"),
        ("limited", "Restricted or adapter required"),
        ("no", "Unsupported at this interface"),
        ("unknown", "Unverified / not run"),
    ]):
        x = 28 + index * 295
        fill, foreground = COLORS[status]
        elements += [
            f'<rect x="{x}" y="{legend_y - 12}" width="13" height="13" fill="{fill}"/>',
            text(x + 19, legend_y, label, 12, foreground)
        ]
    for index, note in enumerate(data["notes"]):
        elements.append(text(28, legend_y + 33 + index * 23, note, 14, "#475569"))
    write_svg(ROOT / "charts/2026-10-27_step-00_capabilities.svg", 1205, legend_y + 125,
              "FP8 KV decode reference capabilities", data["evidence"], elements)


def correctness():
    with (ROOT / "results/2026-10-27_step-01_correctness.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    elements = [
        text(28, 38, "TLX gfx1250 correctness milestones", 26, weight="bold"),
        text(28, 68, "2026-10-27 | Independent FP32 attention over quantized inputs", 16),
        text(28, 96, "Different suite sizes. These bars show test coverage, not kernel speed.", 16, "#92400e"),
    ]
    max_total = max(int(row["total"]) for row in rows)
    scale = 600 / max_total
    for index, row in enumerate(rows):
        y = 145 + index * 105
        passed, total = int(row["passed"]), int(row["total"])
        elements += [
            text(28, y + 23, row["stage"], 17, weight="bold"),
            text(28, y + 47, row["source_status"], 13,
                 "#475569"), f'<rect x="285" y="{y}" width="{total * scale}" height="45" rx="5" fill="#e2e8f0"/>',
            f'<rect x="285" y="{y}" width="{passed * scale}" height="45" rx="5" fill="#15803d"/>',
            text(300 + total * scale, y + 29, f"{passed}/{total} passed", 17, "#166534", "bold")
        ]
    elements += [
        text(28, 385, "Expanded suite: BF16/FP8 Q, distinct head scales, shuffled pages, GQA, SWA, empty/tail cases,",
             15),
        text(28, 410, "split merge, T=8..8192, 8192-wide dense view and 128K/256K contexts (selected head counts).",
             15),
        text(28, 450, "All external references: source reviewed only. Kernel latency and speedup remain unmeasured.",
             15, "#475569")
    ]
    write_svg(ROOT / "charts/2026-10-27_step-01_correctness.svg", 1080, 480, "TLX FP8 decode correctness milestones",
              "Initial suite: 17/17. Expanded suite: 27/27. No performance measurements.", elements)


def bottleneck_forecast(data_path):
    data = json.loads(Path(data_path).read_text())
    elements = [
        text(28, 38, "FP8 KV decode: expected bottlenecks and optimization directions", 25, weight="bold"),
        text(28, 67, f"{data['date']} | Forecast from source inspection; not a measured ranking", 16, "#92400e"),
        text(28, 94, "Per-head K/V scales are required. No native FP8 WMMA or GQA load sharing in the prototype yet.",
             15),
    ]
    for x, label in [(28, "Potential limitation"), (330, "Most relevant workloads"), (700, "Optimization to test")]:
        elements.append(text(x, 140, label, 16, weight="bold"))
    for index, item in enumerate(data["hypotheses"]):
        y = 160 + index * 70
        elements.append(f'<rect x="20" y="{y}" width="1260" height="62" rx="5" fill="#e2e8f0"/>')
        elements += [text(28, y + 26, item["label"], 16, weight="bold"), text(330, y + 26, item["workload"], 15)]
        for line, label in enumerate(item["optimization_lines"]):
            elements.append(text(700, y + 24 + line * 22, label, 15))
    footer = 160 + len(data["hypotheses"]) * 70 + 28
    for index, label in enumerate([
            "Calculated K+V payload at width 8192: 36 MiB (L=2304), 2 GiB (128K), 4 GiB (256K).",
            "These are payload calculations, not measured HBM traffic or latency. Cache reuse and SWA affect reads.",
            "Scale placement is already hoisted; FP8 matrix arithmetic additionally needs a probability-range accuracy check.",
            "Validate mechanisms with ATT/PMC; establish wins with matched unprofiled timing and numerical checks.",
    ]):
        elements.append(text(28, footer + index * 25, label, 15, "#475569"))
    write_svg(ROOT / "charts" / f"{data['date']}_step-03_bottleneck_forecast.svg", 1305, footer + 105,
              "FP8 KV decode bottleneck forecast",
              "Qualitative predictions only; no measured speedups or kernel latency.", elements)


def workload_coverage(data_path):
    data = json.loads(Path(data_path).read_text())
    elements = [
        text(28, 38, "FP8 decode: target workloads versus existing evidence", 26, weight="bold"),
        text(28, 67, f"{data['date']} | Seven cases, 15 T/L combinations | Exact target acceptance remains pending",
             16),
        text(28, 95, "Evidence refers to the 2026-10-27 component suite. No new GPU run or performance measurement.",
             15, "#92400e"),
    ]
    columns = [(28, "Case"), (205, "T"), (385, "L"), (540, "Prior component evidence"), (1020, "Unresolved / gap")]
    for x, label in columns:
        elements.append(text(x, 140, label, 15, weight="bold"))
    for index, case in enumerate(data["cases"]):
        y = 160 + index * 70
        elements.append(f'<rect x="20" y="{y}" width="1325" height="62" rx="5" fill="#fef3c7"/>')
        elements += [
            text(28, y + 26, f"{case['id']}. {case['name']}", 16, weight="bold"),
            text(205, y + 26, case["chart_t"], 14),
            text(385, y + 26, case["chart_l"], 14)
        ]
        for line, label in enumerate(case["coverage_lines"]):
            elements.append(text(540, y + 24 + line * 22, label, 14, "#475569"))
        for line, label in enumerate(case["gap_lines"]):
            elements.append(text(1020, y + 24 + line * 22, label, 14, "#92400e"))
    footer = 160 + len(data["cases"]) * 70 + 26
    elements += [
        text(28, footer, "Dense Q/KV widths: 8192. Folded Q: [1,16,64]. Reported O: [T,1024] x [1024,8192].", 15),
        text(28, footer + 26, "Packed E4M3 cache: K [block,head,D/16,page,16]; V [block,head,page/16,D,16].", 15),
        text(
            28, footer + 52,
            "Amber rows require contract resolution; passing component tests does not establish exact workload acceptance.",
            15, "#92400e")
    ]
    destination = ROOT / "charts" / f"{data['date']}_step-02_workload_coverage.svg"
    write_svg(destination, 1370, footer + 80, "FP8 decode target workload coverage",
              "Seven reported workload cases compared to the existing component tests; exact acceptance pending.",
              elements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--workloads", type=Path, help="Render a workload coverage JSON snapshot")
    selection.add_argument("--forecast", type=Path, help="Render a qualitative bottleneck forecast JSON snapshot")
    args = parser.parse_args()
    if args.workloads:
        workload_coverage(args.workloads)
    elif args.forecast:
        bottleneck_forecast(args.forecast)
    else:
        capabilities()
        correctness()
