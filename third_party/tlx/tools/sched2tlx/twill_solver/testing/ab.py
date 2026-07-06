"""GPU A/B gate for twill candidates.

Emits a baseline schedule-graph and a candidate schedule-graph to temp
``generated.py`` files, benchmarks each with ``perf_engine.py worker`` in an
isolated subprocess (so a faulting/hanging candidate can't poison the other's
CUDA context), and decides ACCEPT/REJECT:

  ACCEPT  iff the candidate is correct on every shape (``ok`` vs torch ref) AND
          not slower than the baseline beyond ``--eps`` on every shape.
  REJECT  otherwise -> the baseline is what should ship (no-regression floor).

This driver, not the solver's cost model, is the authoritative no-regression
guarantee: the solver proposes, the GPU disposes.

Run under the clean env (OHPC gcc9 stripped); this process's env is inherited by
the worker subprocesses. Usage:

  python -m twill_solver.testing.ab --case case3_FA \
      --baseline examples/case3_FA/schedule_graph.json \
      --candidate /tmp/twill/case3_FA_m1.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_EXAMPLES = _HERE.parents[2] / "examples"
_PERF_ENGINE = _EXAMPLES / "testing" / "perf_engine.py"
_KILLGPU = _HERE.parents[4] / "killgpu.sh"  # third_party/tlx/killgpu.sh


def _emit_str(graph_path: str) -> str:
    from twill_solver import emit_check
    return emit_check.emit_source(graph_path)


def _emit(graph_path: str, out_py: Path) -> None:
    out_py.write_text(_emit_str(graph_path))


def _bench(case_dir: Path, generated_py: Path, out_json: Path,
           timeout_s: float) -> dict | None:
    """Run one worker subprocess. Returns parsed JSON, or None on hang/crash."""
    # --no-hw: the A/B compares candidate vs baseline (both generated); the
    # handwritten reference is irrelevant here and avoids per-case hw_call setup.
    cmd = [sys.executable, str(_PERF_ENGINE), "worker",
           "--case-dir", str(case_dir), "--generated", str(generated_py),
           "--out", str(out_json), "--no-hw"]
    try:
        p = subprocess.run(cmd, timeout=timeout_s, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print(f"  [ab] HANG (>{timeout_s}s) benchmarking {generated_py.name}; "
              f"killing GPU", file=sys.stderr)
        _killgpu()
        return None
    if p.returncode != 0:
        print(f"  [ab] worker crashed rc={p.returncode}:\n{p.stderr[-2000:]}",
              file=sys.stderr)
        return None
    return json.loads(out_json.read_text())


def _killgpu() -> None:
    if _KILLGPU.exists():
        subprocess.run(["bash", str(_KILLGPU)], capture_output=True)


def _cmp(base: dict, cand: dict, eps: float) -> tuple[bool, list[str]]:
    """(accept, per-shape report lines)."""
    accept = True
    lines = []
    brows = {tuple(r["shape"]): r for r in base["shapes"]}
    for cr in cand["shapes"]:
        shape = tuple(cr["shape"])
        br = brows.get(shape)
        unit = cr["unit"]
        cok = cr["ok"]
        cms = cr["gen_ms"]
        bms = br["gen_ms"] if br else None
        speedup = (bms / cms) if (bms and cms) else float("nan")
        regress = (bms is not None) and (cms > bms * (1 + eps))
        if not cok or regress:
            accept = False
        flag = "OK" if cok else "INCORRECT"
        verdict = "REGRESS" if regress else ("+FASTER" if speedup > 1 + eps else "=same")
        if bms:
            lines.append(
                f"  {str(shape):<22} {flag:<10} "
                f"cand={cr['throughput']:.1f} base={br['throughput']:.1f} {unit}  "
                f"speedup={speedup:.3f}x  rel={cr['rel']:.1e}  [{verdict}]")
        else:
            lines.append(
                f"  {str(shape):<22} {flag:<10} "
                f"cand={cr['throughput']:.1f}{unit}  rel={cr['rel']:.1e}  (no baseline)")
    return accept, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, help="case dir name")
    ap.add_argument("--baseline", required=True, help="baseline schedule_graph.json")
    ap.add_argument("--candidate", required=True, help="candidate schedule_graph.json")
    ap.add_argument("--eps", type=float, default=0.03, help="perf regression tolerance")
    ap.add_argument("--timeout", type=float, default=240.0, help="per-bench timeout s")
    ap.add_argument("--ship", help="emit the WINNING generated.py here (candidate "
                    "if accepted, else baseline) — the authoritative no-regression gate")
    args = ap.parse_args(argv)

    case_dir = _EXAMPLES / args.case
    if not (case_dir / "bench_spec.py").exists():
        print(f"[ab] {args.case}: no bench_spec.py — cannot A/B (correctness only)",
              file=sys.stderr)
        return 3

    tmp = Path(tempfile.mkdtemp(prefix=f"twill_ab_{args.case}_"))
    base_py, cand_py = tmp / "baseline.py", tmp / "candidate.py"
    _emit(args.baseline, base_py)
    _emit(args.candidate, cand_py)

    print(f"[ab] {args.case}: benchmarking baseline ...", file=sys.stderr)
    base = _bench(case_dir, base_py, tmp / "base.json", args.timeout)
    if base is None:
        print(f"[ab] {args.case}: BASELINE failed to bench — aborting", file=sys.stderr)
        return 2
    print(f"[ab] {args.case}: benchmarking candidate ...", file=sys.stderr)
    cand = _bench(case_dir, cand_py, tmp / "cand.json", args.timeout)
    if cand is None:
        print(f"[ab] {args.case}: CANDIDATE hung/crashed -> REJECT (keep baseline)",
              file=sys.stderr)
        return 1

    accept, lines = _cmp(base, cand, args.eps)
    print(f"\n== A/B {args.case} (eps={args.eps}) ==")
    for ln in lines:
        print(ln)
    verdict = "ACCEPT (candidate correct + no regression)" if accept \
        else "REJECT (incorrect or regression -> keep baseline)"
    print(f"[ab] {args.case}: {verdict}")

    # Authoritative no-regression gate: ship the winner. If the candidate was
    # not strictly better-and-correct on every shape, we ship the baseline —
    # so the emitted kernel is never worse than what modulo produced.
    if args.ship:
        winner = args.candidate if accept else args.baseline
        Path(args.ship).write_text(_emit_str(winner))
        print(f"[ab] {args.case}: shipped {'candidate' if accept else 'baseline'} "
              f"-> {args.ship}")
    return 0 if accept else 1


if __name__ == "__main__":
    sys.exit(main())
