"""Render the November 9 retained optimization evidence; no GPU execution."""
import collections
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results/2026-11-09_optimization"
MATRIX = "2026-11-09_step-15_final_matrix"
FRAMEWORK = "2026-11-09_step-16_framework_repeat"


def main():
    subprocess.run([
        sys.executable,
        str(ROOT / "summarize_audit.py"),
        "--input",
        str(DATA / "repeated_final_v2.json"),
        "--stem",
        str(ROOT / "charts" / MATRIX),
        "--report",
        str(ROOT / "OPTIMIZATION_2026-11-09.md"),
        "--title",
        "TLX FP8 KV decode: final retained kernel",
    ], check=True)
    matrix = json.loads((ROOT / "results" / (MATRIX + ".json")).read_text())
    verification = json.loads((DATA / "framework_final_v2/summary.json").read_text())
    records = []
    family_labels = dict(tlx="TLX", sglang_native="SGLang best tested", vllm_native="vLLM best tested",
                         aiter_shared_operator="AITER")
    for case in verification:
        configurations = dict(tlx=dict(backend="tlx_auto", splits=16, block_n=64), **case["selected"])
        for family, cfg in configurations.items():
            runs = [r for r in case["verification"] if all(r[k] == v for k, v in cfg.items())]
            assert len(runs) == 3 and all(r["status"] == "pass" for r in runs)
            values = [r["median_us"] for r in runs]
            records.append(
                dict(
                    **case["workload"], family=family, **cfg, median_us=statistics.median(values), min_us=min(values),
                    max_us=max(values), raw_file=
                    f"results/2026-11-09_optimization/framework_final_v2/{case['workload']['case']}_verification.json"))
    (ROOT / "results" / (FRAMEWORK + ".json")).write_text(json.dumps(records, indent=2) + "\n")
    with (ROOT / "results" / (FRAMEWORK + ".csv")).open("w") as f:
        fields = [
            "case", "heads", "kv_heads", "dim", "length", "window", "scales", "fp8_q", "family", "backend", "splits",
            "block_n", "median_us", "min_us", "max_us", "raw_file"
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2))
    for ax, case in zip(axes, verification):
        selected = [r for r in records if r["case"] == case["workload"]["case"]]
        selected.sort(key=lambda r: list(family_labels).index(r["family"]))
        values = [r["median_us"] for r in selected]
        errors = [[r["median_us"] - r["min_us"] for r in selected], [r["max_us"] - r["median_us"] for r in selected]]
        bars = ax.bar(range(len(selected)), values, yerr=errors, capsize=3,
                      color=["#3155b8", "#ad7153", "#8d71a8", "#2c815b"])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, fontsize=9)
        ax.set_xticks(range(len(selected)), [family_labels[r["family"]].replace(" best tested", "") for r in selected],
                      rotation=25)
        spec = case["workload"]
        ax.set_title(f"Hq/Hkv {spec['heads']}/{spec['kv_heads']}, D{spec['dim']}")
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("GPU invocation µs • lower is faster")
    fig.suptitle("FP8 KV decode • final core framework repeats, L2304/page64")
    fig.text(
        .025, .025, "Three fresh order-rotated rounds; ranges span round medians. Shared-scale restricted controls.\n"
        "Prior passing reference configurations; unmodified GPU bodies. Packing/compilation excluded; clocks unlocked.",
        fontsize=9)
    fig.tight_layout(rect=[0, .10, 1, .94])
    for suffix in ("svg", "png"):
        fig.savefig(ROOT / "charts" / (FRAMEWORK + "." + suffix), dpi=150)
    plt.close(fig)

    xml = ET.parse(DATA / "correctness_final_v2.xml").getroot()
    suites = list(xml.iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    assert sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites) == 0
    counts = collections.Counter(r["comparison"] for r in matrix)
    regressions = [r for r in matrix if r.get("tlx_us", 0) > r.get("frozen_us", float("inf")) * 1.03]
    detail = [
        "## Retained changes and validation", "",
        "One-warp D128 decode with page/length-based splits closes the original dense D128 gap. "
        "Packed K/V LDS loads use matrix-operand register ownership, and one-warp K/V waits are staggered around QK/PV. "
        "Proven single-tile partitions remove dynamic online state. FP16 normalized raw-V partials start at 8 splits; "
        "LSE/merge remain FP32 and apply the separate V head scale. The original multi-warp loop/fused schedule "
        "is retained for larger query sets, and tiny windows/FP8-Q grouped D128 keep their better four-warp choices.",
        "", f"Final strict suite: **{tests-skipped} passed, {skipped} skipped**, unchanged atol=5e-4/rtol=1e-2. "
        "It covers unequal head scales, BF16/FP8 Q, poisoned tails, empty contexts, page16/32/64/128, "
        "peaked multiseed inputs, workspace reuse, S8/12 boundaries and aggregate tiny probability mass. "
        "All 80 benchmark cases also pass in each of three rounds. "
        "[JUnit](results/2026-11-09_optimization/correctness_final_v2.xml) · "
        "[Manifest](results/2026-11-09_optimization/manifest.json)", "",
        "The per-head E4M3FN contract is unchanged; this is not E8M0 block-scaled MXFP8. "
        "The original prototype partition/merge function ASTs are preserved. No compiler/native changes, "
        "reference GPU edits or relaxed accuracy tolerances were used.", "", "## Remaining gaps", "",
        f"Of 40 passing AITER controls, {counts['tlx_faster']} favor TLX beyond 3%, "
        f"{counts['within_3_percent']} are within 3%, and {counts['aiter_faster']} favor AITER. "
        "Eight other AITER cases fail this project's numerical gate and are excluded; "
        "32 cases have no matching AITER adapter for arbitrary per-head scales or the chosen multi-query semantics. "
        "This does not establish an upstream correctness bug under a different accuracy target.", "",
        "| AITER-favored case | TLX µs | AITER µs |", "| --- | ---: | ---: |"
    ]
    detail += [
        f"| {r['case']} | {r['tlx_us']:.2f} | {r['aiter_us']:.2f} |" for r in matrix
        if r["comparison"] == "aiter_faster"
    ]
    detail += ["", f"Current versus starting-TLX regressions beyond 3%: **{len(regressions)} cases**."]
    detail += [f"- {r['case']}: {r['frozen_us']:.2f} → {r['tlx_us']:.2f}µs." for r in regressions]
    detail += [
        "", "## Fresh core framework comparison", "",
        "Prior strongest passing SGLang/vLLM configurations are repeated without a new selection sweep. "
        "These are standalone source-kernel measurements with the shared compiler, not framework-serving throughput. "
        "AITER remains one deduplicated underlying operator; scalar shared-scale controls do not establish "
        "native arbitrary-head scaling for every reference.", "",
        f"[Framework chart](charts/{FRAMEWORK}.svg) · [CSV](results/{FRAMEWORK}.csv) · "
        f"[Data](results/{FRAMEWORK}.json)", "", "| Case | TLX | SGLang best tested | vLLM best tested | AITER |",
        "| --- | ---: | ---: | ---: | ---: |"
    ]
    for case in verification:
        spec = case["workload"]
        by_family = {r["family"]: r for r in records if r["case"] == spec["case"]}
        label = f"{'GQA' if spec['kv_heads'] < spec['heads'] else 'MHA'}, D{spec['dim']}"
        detail.append("| " + label + " | " + " | ".join(f"{by_family[f]['median_us']:.2f}" for f in family_labels) +
                      " |")
    detail += [
        "", "## Profiling and rejected experiments", "",
        "rocprofv3 ATT/analyzer captured valid old/new dense D128 samples. The old sample concentrated 57.1% "
        "of stall cycles at LDS waits; the newer sample concentrated 33.1% at KMCNT and 27.5% at tensor waits. "
        "These are selected-CU sampled instruction fractions, not GPU utilization or normalized kernel latency. "
        "Short-window captures missed instruction coverage, and the last AITER decoder retry was interrupted "
        "by the host reboot. No bottleneck claim is based on empty traces or previously all-zero PMC counters. "
        "[Old ATT](results/2026-11-09_optimization/att_d128_before_analysis.txt) · "
        "[Current dense ATT](results/2026-11-09_optimization/att_tlx_mha128_analysis.txt)", "",
        "Rejected attempts include vector replacements, multipage tiles, host descriptors, forced scheduling, "
        "VMEM scale loads, general single buffering, direct K/V loads, custom merge ownership and loop unrolling. "
        "Two-component FP8 probabilities were slower and failed tiny-mass accuracy tests; FP16 PV was restored "
        "and that regression is retained. Forced mixed-dtype AITER Gluon2D also failed and was excluded. "
        "The actual stock BF16-Q/FP8-KV SWA route is AITER Triton2D. "
        "[Dated experiment ledger041–077](FP8_KV_DECODE_ATTN.md) · [Pinned sources/routing](REFERENCES.md)", "",
        "Final comparisons followed a host reboot/container restart and reran frozen/current/reference together "
        "on the verified 96-CU/144-GiB runtime agent, CUDA_VISIBLE_DEVICES=2, PCI2:4:0. Earlier paired/scout "
        "charts remain separate. Clocks are unlocked; repeated same-input graph timings favor warm caches. "
        "The exact model head mapping, T/mask semantics, SWA window and output projection remain undisclosed. "
        "These finite synthetic tests cannot establish a win across every shape or end-to-end serving.", ""
    ]
    path = ROOT / "OPTIMIZATION_2026-11-09.md"
    base = path.read_text()
    marker = "| Case | Current µs | Frozen µs | AITER µs | AITER/current | Result | Current launch |"
    prefix, full_matrix = base.split(marker, 1)
    path.write_text(prefix + "\n".join(detail) + "\n## Complete matrix\n\n" + marker + full_matrix)
    print(
        json.dumps(
            dict(strict_passed=tests - skipped, skipped=skipped, comparisons=dict(counts),
                 original_tlx_regressions=len(regressions)), indent=2))


if __name__ == "__main__":
    main()
