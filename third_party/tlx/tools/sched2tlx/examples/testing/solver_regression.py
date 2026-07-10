"""Solver-configuration regression gate for the modulo scheduler.

SolverMigrationNotes step-4 item 5 (the default-flip gate): for each CP-SAT
solver configuration and each regenerable case,

    pre_modulo TTGIR --(triton-opt -nvgpu-modulo-schedule, config env)-->
    schedule_graph.json --(sched2tlx)--> generated.py --> correctness
    (+ perf where a bench_spec exists)

compared against the COMMITTED kernel, plus the case3 FA absolute canary
(>= --canary-tflops at the largest shape) as the hard gate. Never touches
committed artifacts: candidates live in a temp dir; cases without a
bench_spec get their directory copied so run_generated.py imports the
candidate kernel.

A candidate whose non-comment codegen is byte-identical to the committed
kernel is reported as "parity" and skipped on the GPU — correctness and
perf are inherited by construction.

case2 is excluded: its pre_modulo.ttgir is 256-blockM, which the emitter
cannot lower yet (TMEM blockM <= 128 — see EmitterCaps in
ModuloSchedulePass.cpp and the guard-3 section of SolverMigrationNotes).

Usage (B200 node):
  python solver_regression.py                        # all configs, all cases
  python solver_regression.py --configs full,full-noguard --cases case3_FA
  python solver_regression.py --skip-perf            # correctness only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = TESTING_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parents[4]  # examples -> sched2tlx -> tools -> tlx -> third_party -> repo

# Each config names the scheduling pass triton-opt runs plus env tuning it.
# The joint solver is its own pass (nvgpu-joint-solver-schedule) since the
# modulo/joint split; it always uses the cpsat schedule backend, so the old
# rau-schedule + joint-partition combinations (TRITON_MODULO_CPSAT_JOINT on
# the modulo pass) no longer exist. Guard 1 is deleted outright, so the old
# "full-noguard" config is simply "full" now. joint-mode mapping: 0 = v2
# then v1 fallback (old CPSAT_JOINT=2), 1 = v1 only (old CPSAT_JOINT=1),
# 2 = v2 strict (new).
CONFIGS = {
    # cpsat schedule via the modulo pass; the driver pairs it with the
    # CP-SAT v1 partition (2026-07-10 promotion — heuristic partitions
    # mis-handle MinII-aggressive schedules), so this is "joint minus v2"
    "cpsat": {"pass": "-nvgpu-modulo-schedule",
              "env": {"TRITON_USE_MODULO_SCHEDULE": "cpsat"}},
    # partition-only joint solve (v1: cycles fixed)
    "joint1": {"pass": "-nvgpu-joint-solver-schedule=joint-mode=1", "env": {}},
    # cycles + wg in one solve, strict (no v1 fallback)
    "joint2": {"pass": "-nvgpu-joint-solver-schedule=joint-mode=2", "env": {}},
    # full solver stack: the joint pass default (v2 then v1 fallback)
    "full": {"pass": "-nvgpu-joint-solver-schedule", "env": {}},
    # full stack + streaming variable-latency classification (Twill §5.3:
    # no-incoming-dep TMA loads solve at latency 0 behind their ring)
    "full-stream": {"pass": "-nvgpu-joint-solver-schedule",
                    "env": {"TRITON_MODULO_STREAMING_VL": "1"}},
    # full stack + hard register budget at the SM cap (Twill REGISTERLIMIT;
    # forbids the oversubscribe-then-downscale partitions case4 v2 picked)
    "full-regcap": {"pass": "-nvgpu-joint-solver-schedule",
                    "env": {"TRITON_MODULO_REG_BUDGET": "65536"}},
}

CASES = {
    "case1_simple_gemm": "pre_modulo.ttgir",
    "case3_FA": "fa_fwd_nows_pre_modulo.ttgir",
    # FA-bwd: the joint solver's hardest partition canary (2026-07-09: its v2
    # partition exposed the WarpSpecializeOp tensor-capture emitter gap that
    # the whole run had been blind to because this case wasn't wired in).
    # No bench_spec — correctness-only via run_generated.py.
    "case4_FA_bwd": "fa_bwd_nows_pre_modulo.ttgir",
    "case5_addmm_bias": "addmm_bias_pre_modulo.ttgir",
    "case6_layernorm": "layernorm_fwd_pre_modulo.ttgir",
    "case7_wgrad_bias": "wgrad_bias_pre_modulo.ttgir",
}

CANARY_CASE = "case3_FA"


def find_triton_opt(cli_arg: str | None) -> Path:
    for cand in filter(None, [cli_arg, os.environ.get("TRITON_OPT")]):
        p = Path(cand)
        if p.exists():
            return p
    hits = glob.glob(str(REPO_ROOT / "build" / "cmake.*" / "bin" / "triton-opt"))
    if hits:
        return Path(hits[0])
    sys.exit("triton-opt not found: pass --triton-opt or set TRITON_OPT")


def noncomment(path: Path) -> str:
    return "\n".join(l for l in path.read_text().splitlines() if not l.lstrip().startswith("#"))


def dump_and_emit(opt: Path, case: str, cfg: dict, tmp: Path) -> Path | None:
    """Run the config's scheduling pass and emit the candidate kernel.
    Returns the generated.py path, or None on failure."""
    tmp.mkdir(parents=True, exist_ok=True)
    graph = tmp / f"{case}.schedule_graph.json"
    gen = tmp / f"{case}.generated.py"
    env = dict(os.environ)
    env.update(cfg["env"])
    env.setdefault("TRITON_MODULO_CPSAT_CMD", f"{sys.executable} -m triton.tools.modulo_cpsat")
    env["TRITON_MODULO_DUMP_SCHEDULE"] = str(graph)
    proc = subprocess.run(
        [str(opt), str(EXAMPLES_DIR / case / CASES[case]), "-allow-unregistered-dialect",
         cfg["pass"]],
        env=env, capture_output=True, text=True, timeout=1200)
    # Some cases exit 1 with a post-dump verifier error; judge by the marker.
    if "Dumped schedule graph" not in proc.stderr or not graph.exists():
        print(f"    dump FAILED: {proc.stderr.strip().splitlines()[-1:] or 'no output'}")
        return None
    emit = subprocess.run(
        [sys.executable, "-m", "sched2tlx", str(graph), "-o", str(gen)],
        env={**os.environ, "PYTHONPATH": str(EXAMPLES_DIR.parent)},
        capture_output=True, text=True, timeout=600)
    if emit.returncode != 0 or not gen.exists():
        print(f"    emit FAILED: {emit.stderr.strip().splitlines()[-1:] or 'no output'}")
        return None
    return gen


def run_correctness_only(case: str, gen: Path, tmp: Path) -> bool:
    """For cases without a bench_spec: run run_generated.py against the
    candidate kernel in a throwaway copy of the case dir.

    On failure, freeze the repro (schedule graph + generated.py + output)
    in a persistent directory: the CP-SAT solves are nondeterministic
    (wall-clock limits), so a failing draw may not recur for dozens of
    runs — the frozen graph replays deterministically through sched2tlx
    (case4 flake, 2026-07-10)."""
    work = tmp / f"{case}.work"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(EXAMPLES_DIR / case, work)
    shutil.copy(gen, work / "generated.py")
    proc = subprocess.run([sys.executable, "run_generated.py"], cwd=work,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        keep = Path(tempfile.mkdtemp(prefix=f"solver-regression-fail-{case}-"))
        for artifact in (tmp / f"{case}.schedule_graph.json", gen):
            if artifact.exists():
                shutil.copy(artifact, keep / artifact.name)
        (keep / "correctness.out").write_text(proc.stdout + proc.stderr)
        print(f"    correctness FAIL — repro frozen in {keep}")
    return proc.returncode == 0


def run_worker(case: str, gen: Path, tmp: Path) -> dict | None:
    out = tmp / f"{case}.{gen.stem}.perf.json"
    proc = subprocess.run(
        [sys.executable, str(TESTING_DIR / "perf_engine.py"), "worker",
         "--case-dir", str(EXAMPLES_DIR / case), "--generated", str(gen),
         "--out", str(out)],
        capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not out.exists():
        print(f"    perf worker FAILED: {proc.stderr.strip().splitlines()[-1:] or '?'}")
        return None
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--triton-opt")
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--perf-tol", type=float, default=0.05,
                    help="allowed per-shape slowdown vs the committed kernel")
    ap.add_argument("--canary-tflops", type=float, default=651.0,
                    help="case3 hard gate at the largest shape")
    ap.add_argument("--keep", action="store_true", help="keep the temp dir")
    args = ap.parse_args()

    opt = find_triton_opt(args.triton_opt)
    tmpdir = Path(tempfile.mkdtemp(prefix="solver-regression-"))
    print(f"triton-opt: {opt}\nworkdir: {tmpdir}\n")

    cases = [c for c in args.cases.split(",") if c in CASES]
    baselines: dict[str, dict] = {}
    failures = 0

    for cfg_name in args.configs.split(","):
        cfg = CONFIGS.get(cfg_name)
        if cfg is None:
            print(f"unknown config {cfg_name!r}, skipping")
            continue
        print(f"== config {cfg_name}: {cfg}")
        for case in cases:
            gen = dump_and_emit(opt, case, cfg, tmpdir / cfg_name)
            if gen is None:
                print(f"  {case}: GENERATION FAIL")
                failures += 1
                continue
            committed = EXAMPLES_DIR / case / "generated.py"
            if noncomment(gen) == noncomment(committed):
                print(f"  {case}: parity (codegen identical — correctness/perf inherited)")
                continue
            has_bench = (EXAMPLES_DIR / case / "bench_spec.py").exists()
            if args.skip_perf or not has_bench:
                ok = run_correctness_only(case, gen, tmpdir / cfg_name)
                print(f"  {case}: correctness {'PASS' if ok else 'FAIL'}"
                      f"{'' if has_bench else ' (no bench_spec: correctness only)'}")
                failures += 0 if ok else 1
                continue
            if case not in baselines:
                base = run_worker(case, committed, tmpdir)
                if base is None:
                    failures += 1
                    continue
                baselines[case] = {tuple(r["shape"]): r for r in base["shapes"]}
            res = run_worker(case, gen, tmpdir / cfg_name)
            if res is None:
                failures += 1
                continue
            worst = 1.0
            ok = True
            canary_val = None
            for r in res["shapes"]:
                ok &= bool(r["ok"])
                b = baselines[case].get(tuple(r["shape"]))
                if b:
                    worst = max(worst, r["gen_ms"] / b["gen_ms"])
                canary_val = r["throughput"]  # last row = largest shape
            verdicts = []
            if not ok:
                verdicts.append("CORRECTNESS FAIL")
            if worst > 1 + args.perf_tol:
                verdicts.append(f"PERF REGRESSION {worst:.2f}x vs committed")
            if case == CANARY_CASE and canary_val is not None:
                gate = canary_val >= args.canary_tflops
                verdicts.append(f"canary {canary_val:.1f}/{args.canary_tflops:.0f} "
                                f"{'OK' if gate else 'MISS'}")
                if not gate:
                    failures += 1
            print(f"  {case}: {'PASS' if ok and worst <= 1 + args.perf_tol else 'FAIL'} "
                  f"(worst {worst:.2f}x; {'; '.join(verdicts) or 'clean'})")
            failures += 0 if ok and worst <= 1 + args.perf_tol else 1

    if not args.keep:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\n{'ALL GREEN' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
