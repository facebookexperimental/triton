#!/usr/bin/env python
"""Phase 4 coverage matrix.

Runs each candidate workload under three TTGIR-SCHED configurations:
  * baseline (no env vars)
  * APPLY=1 default      (Phase 3 — per-M-row sched_barriers)
  * APPLY=1 STRIDE=0     (Phase 2 — no sched_barriers, IR-equivalent to
                          baseline modulo extract_slice + concat)

For each workload + config we record: did it compile? did it run? does the
numerical output match a reference? what TFLOPS (if applicable)?

The matrix is printed at the end as a markdown table.

Usage:
    HIP_VISIBLE_DEVICES=0 python phase4_coverage.py
"""
import os
import subprocess
import shlex
import sys
import time

# (workload-label, cwd, command, parse-tflops-from-output-fn)
def parse_one_run(stdout: str) -> str:
    # "_one_run_envcompare.py" → "OK 858.79 BLOCK_M: ..."
    for line in stdout.splitlines():
        if line.startswith("OK "):
            return line.split()[1]
        if line.startswith("FAIL"):
            return "FAIL"
    return "?"


def parse_tutorial(stdout: str) -> str:
    # 03-matrix-multiplication.py prints a table; grab the last numeric col
    last = ""
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            float(parts[-1])
            last = parts[-1]
        except ValueError:
            pass
    return last or "?"


WORKLOADS = [
    # (label, cwd, cmd-as-list, parser-fn, timeout-s)
    ("stand-alone matmul K=1024", "/home/mren/AMD/triton/claude/triton_kernels_baseline",
     ["python", "_one_run_envcompare.py", "1024"], parse_one_run, 120),
    ("stand-alone matmul K=2048", "/home/mren/AMD/triton/claude/triton_kernels_baseline",
     ["python", "_one_run_envcompare.py", "2048"], parse_one_run, 120),
    ("stand-alone matmul K=4096", "/home/mren/AMD/triton/claude/triton_kernels_baseline",
     ["python", "_one_run_envcompare.py", "4096"], parse_one_run, 180),
    ("stand-alone matmul K=8192", "/home/mren/AMD/triton/claude/triton_kernels_baseline",
     ["python", "_one_run_envcompare.py", "8192"], parse_one_run, 240),
]

# Per-workload env-var matrix.
MODES = [
    ("baseline",          {}),
    ("Phase 3 default",   {"TRITON_ENABLE_TTGIR_SCHED": "1", "TRITON_TTGIR_SCHED_APPLY": "1"}),
    ("Phase 2 (no bars)", {"TRITON_ENABLE_TTGIR_SCHED": "1", "TRITON_TTGIR_SCHED_APPLY": "1",
                           "TRITON_TTGIR_SCHED_BARRIER_STRIDE": "0"}),
]


def run(label, cwd, cmd, parser, env_extra, timeout):
    env = os.environ.copy()
    env.update(env_extra)
    env.setdefault("HIP_VISIBLE_DEVICES", "0")
    t0 = time.time()
    try:
        out = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                             text=True, timeout=timeout)
        dt = time.time() - t0
        rc = out.returncode
        merged = (out.stdout or "") + (out.stderr or "")
        tflops = parser(merged) if rc == 0 else "—"
        crash = ("Traceback" in merged or "Error" in merged or
                 "core dumped" in merged or rc != 0)
        return {"label": label, "tflops": tflops, "rc": rc,
                "crash": crash, "elapsed_s": round(dt, 1),
                "stderr_tail": (out.stderr or "").splitlines()[-1] if rc else ""}
    except subprocess.TimeoutExpired:
        return {"label": label, "tflops": "TIMEOUT", "rc": -1, "crash": True,
                "elapsed_s": timeout, "stderr_tail": ""}


def main():
    rows = []
    for label, cwd, cmd, parser, timeout in WORKLOADS:
        for mode, env in MODES:
            r = run(label, cwd, cmd, parser, env, timeout)
            print(f"[{label}] [{mode}] tf={r['tflops']} rc={r['rc']} "
                  f"crash={r['crash']} dt={r['elapsed_s']}s",
                  flush=True)
            rows.append((label, mode, r))

    # Markdown table
    print("\n## Phase 4 coverage matrix\n")
    print("| Workload | Baseline TF | Phase 3 default TF | Phase 2 (no bars) TF | Notes |")
    print("|---|---:|---:|---:|---|")
    # Group by workload
    by_wl = {}
    for label, mode, r in rows:
        by_wl.setdefault(label, {})[mode] = r
    for label, mr in by_wl.items():
        b = mr.get("baseline", {})
        p3 = mr.get("Phase 3 default", {})
        p2 = mr.get("Phase 2 (no bars)", {})
        notes = []
        if any(d.get("crash") for d in [b, p3, p2]):
            notes.append("crash in some mode")
        try:
            b_f = float(b.get("tflops", "nan"))
            p3_f = float(p3.get("tflops", "nan"))
            delta = (p3_f - b_f) / b_f * 100.0
            notes.append(f"Phase 3 Δ {delta:+.1f}%")
        except (ValueError, TypeError):
            pass
        print(f"| {label} | {b.get('tflops','?')} | {p3.get('tflops','?')} "
              f"| {p2.get('tflops','?')} | {'; '.join(notes) or '—'} |")


if __name__ == "__main__":
    main()
