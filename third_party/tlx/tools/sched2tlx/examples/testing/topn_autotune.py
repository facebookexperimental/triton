#!/usr/bin/env python3
"""Top-N schedule autotuning harness (perf-dir-4, 4a).

Closes the loop the TOPN dump infrastructure left open: the modulo pass's cost
model ranks candidate warp-group partitions, but the ranking is imperfect — the
true-best schedule may be rank 2 or 3. This driver, per case:

  1. runs triton-opt with TRITON_MODULO_DUMP_TOPN=N on the case's
     pre_modulo.ttgir, producing the pluralized ``schedule_graphs.json``
     ({"variants": [doc0, doc1, ...]}, best-predicted first);
  2. splits it into per-variant schedule_graph.json files and emits each with
     ``python -m sched2tlx`` (an emit failure marks that variant, not the run);
  3. benchmarks every emitted variant through ``perf_engine.py worker`` — one
     subprocess per variant so a faulting kernel cannot poison the next run's
     CUDA context; correctness (vs torch reference and, for variant 0, the
     handwritten kernel) gates a variant from winning;
  4. writes an aggregate JSON + prints a predicted-vs-measured table with the
     rank correlation and the measured winner.

Bench methodology is perf_engine's uniformly: triton.testing.do_bench (cold-L2
flush, median). Never mix these numbers with hand-rolled hot-L2 loops.

Usage (venv python with torch + triton and a Blackwell GPU):
    python topn_autotune.py --triton-opt /path/to/triton-opt \
        [--cases case1_simple_gemm,...] [--topn 3] [--out results_dir]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = TESTING_DIR.parent
TOOL_DIR = EXAMPLES_DIR.parent  # holds the sched2tlx/ package

WORKER_TIMEOUT_S = 900


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)  # lmod's gcc-9 libstdc++ breaks libtriton
    return env


def dump_variants(case_dir: Path, triton_opt: Path, topn: int, work: Path) -> list[dict]:
    """Run the modulo pass with the TOPN dump and return the variant docs."""
    ttgirs = sorted(case_dir.glob("*pre_modulo.ttgir"))
    if not ttgirs:
        raise FileNotFoundError(f"no *pre_modulo.ttgir in {case_dir}")
    base = work / "schedule_graph.json"
    env = _clean_env()
    env["TRITON_MODULO_DUMP_SCHEDULE"] = str(base)
    env["TRITON_MODULO_DUMP_TOPN"] = str(topn)
    proc = subprocess.run(
        [str(triton_opt), str(ttgirs[0]), "--nvgpu-modulo-schedule", "-o", os.devnull],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"triton-opt failed on {ttgirs[0].name}:\n{proc.stderr[-2000:]}")
    plural = work / "schedule_graphs.json"
    if not plural.exists():
        # TOPN may degrade to a single-variant dump (e.g. only one candidate).
        return [json.loads(base.read_text())]
    return json.loads(plural.read_text())["variants"]


def emit_variant(doc: dict, work: Path, k: int) -> tuple[Path | None, str]:
    """Write variant doc k and run sched2tlx on it. Returns (kernel.py, error)."""
    graph_json = work / f"variant{k}.schedule_graph.json"
    graph_json.write_text(json.dumps(doc))
    kernel_py = work / f"variant{k}.py"
    proc = subprocess.run(
        [sys.executable, "-m", "sched2tlx", str(graph_json), "-o", str(kernel_py)],
        cwd=str(TOOL_DIR),
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        return None, proc.stderr.strip().splitlines()[-1] if proc.stderr else "emit failed"
    return kernel_py, ""


def bench_variant(case_dir: Path, kernel_py: Path, out_json: Path, want_hw: bool) -> tuple[dict | None, str]:
    """Benchmark one variant in an isolated worker process."""
    cmd = [
        sys.executable,
        str(TESTING_DIR / "perf_engine.py"),
        "worker",
        "--case-dir", str(case_dir),
        "--generated", str(kernel_py),
        "--out", str(out_json),
    ]
    if not want_hw:
        cmd.append("--no-hw")
    try:
        proc = subprocess.run(
            cmd, env=_clean_env(), capture_output=True, text=True, timeout=WORKER_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT ({WORKER_TIMEOUT_S}s) — run third_party/tlx/killgpu.sh if the GPU is wedged"
    if proc.returncode != 0 or not out_json.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, tail[-1][:300] if tail else "worker failed"
    return json.loads(out_json.read_text()), ""


def predicted_cost(doc: dict) -> float:
    return sum(
        loop["schedule_loop"].get("partition_cost", 0.0) for loop in doc.get("loops", [])
    )


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def spearman(pred_ranks: list[int], meas_ranks: list[int]) -> float | None:
    n = len(pred_ranks)
    if n < 2:
        return None
    d2 = sum((p - m) ** 2 for p, m in zip(pred_ranks, meas_ranks))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def run_case(case_dir: Path, triton_opt: Path, topn: int, out_dir: Path) -> dict:
    work = out_dir / case_dir.name
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {"case": case_dir.name, "variants": []}

    try:
        docs = dump_variants(case_dir, triton_opt, topn, work)
    except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        result["error"] = str(e)
        return result

    for k, doc in enumerate(docs):
        entry: dict = {"variant_id": k, "predicted_cost": predicted_cost(doc)}
        kernel_py, err = emit_variant(doc, work, k)
        if kernel_py is None:
            entry.update(status="EMIT_FAIL", error=err)
            result["variants"].append(entry)
            continue
        bench, err = bench_variant(
            case_dir, kernel_py, work / f"variant{k}.result.json", want_hw=(k == 0)
        )
        if bench is None:
            entry.update(status="RUN_FAIL", error=err)
            result["variants"].append(entry)
            continue
        rows = bench["shapes"]
        entry.update(
            status="OK" if all(r["ok"] for r in rows) else "FAIL_CORRECTNESS",
            shapes=[r["shape"] for r in rows],
            gen_ms=[r["gen_ms"] for r in rows],
            throughput=[r["throughput"] for r in rows],
            unit=rows[0]["unit"] if rows else "",
            hw_ms=[r["hw_ms"] for r in rows] if k == 0 else None,
        )
        result["variants"].append(entry)

    # Ranking quality + winner over correctness-passing variants.
    ok = [v for v in result["variants"] if v.get("status") == "OK"]
    for v in ok:
        v["geomean_ms"] = geomean(v["gen_ms"])
    baseline = next((v for v in ok if v["variant_id"] == 0), None)
    if baseline:
        for v in ok:
            v["speedup_vs_rank0"] = baseline["geomean_ms"] / v["geomean_ms"]
    if ok:
        by_ms = sorted(ok, key=lambda v: v["geomean_ms"])
        result["winner"] = by_ms[0]["variant_id"]
        meas_rank = {v["variant_id"]: i for i, v in enumerate(by_ms)}
        pred_order = sorted(v["variant_id"] for v in ok)
        result["spearman"] = spearman(
            list(range(len(pred_order))), [meas_rank[vid] for vid in pred_order]
        )
        result["rank0_is_best"] = result["winner"] == (baseline["variant_id"] if baseline else None)
    return result


def print_case(result: dict) -> None:
    print(f"\n=== {result['case']} ===")
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    for v in result["variants"]:
        vid, cost = v["variant_id"], v["predicted_cost"]
        line = f"  v{vid} pred_cost={cost:.1f} {v['status']}"
        if v.get("status") == "OK":
            tps = "/".join(f"{t:.0f}" for t in v["throughput"])
            line += f" geomean={v['geomean_ms']:.3f}ms [{tps} {v['unit']}]"
            if "speedup_vs_rank0" in v:
                line += f" vs-rank0={v['speedup_vs_rank0']:.3f}x"
        elif v.get("error"):
            line += f" ({v['error'][:150]})"
        print(line)
    if "winner" in result:
        tag = "rank-0 confirmed" if result.get("rank0_is_best") else "COST MODEL MISRANKED"
        rho = result.get("spearman")
        rho_s = f" spearman={rho:.2f}" if rho is not None else ""
        print(f"  -> measured winner: v{result['winner']} ({tag}){rho_s}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triton-opt", required=True, help="path to a triton-opt with the modulo pass")
    ap.add_argument("--cases", help="comma-separated case dir names (default: all with bench_spec.py)")
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--out", default=str(TESTING_DIR / "topn_results"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cases:
        names = args.cases.split(",")
    else:
        names = [
            p.name
            for p in sorted(EXAMPLES_DIR.iterdir())
            if p.is_dir() and p.name.startswith("case") and (p / "bench_spec.py").exists()
        ]

    summary = []
    for name in names:
        result = run_case(EXAMPLES_DIR / name, Path(args.triton_opt).resolve(), args.topn, out_dir)
        print_case(result)
        summary.append(result)
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1))

    print("\n=== summary ===")
    for r in summary:
        if "winner" not in r:
            print(f"  {r['case']}: no correct variant ({r.get('error', 'see per-variant status')})")
            continue
        ok = [v for v in r["variants"] if v.get("status") == "OK"]
        best = next(v for v in ok if v["variant_id"] == r["winner"])
        gain = best.get("speedup_vs_rank0")
        gain_s = f" gain={gain:.3f}x" if gain else ""
        print(f"  {r['case']}: winner v{r['winner']}{gain_s} "
              f"({'ok' if r.get('rank0_is_best') else 'MISRANKED'})")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nresults in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
