#!/usr/bin/env python3
"""Before-vs-after regression driver for the sched2tlx emitter.

Given any diff that touches the sched2tlx tool (default: the current commit),
this re-emits every example case with the emitter BEFORE and AFTER the diff and
checks, for each auto-discovered case under ``examples/``:

  * correctness — the after-diff generated kernel matches the case's hand-written
    reference (the generated output is compared directly against the handwritten
    output, and both against the torch reference);
  * performance — the after-diff kernel is no slower than the before-diff kernel,
    per shape, within ``--perf-tol``.

Both checks are before-vs-after aware: a case that is already broken/slow *before*
the diff is not counted against it (only a true regression — ok/fast before,
broken/slow after — fails). Everything runs in-process so it works both under a
plain triton-enabled interpreter and inside a buck ``python_binary`` (par).

Run it from anywhere inside your fbsource checkout — the repo root is
auto-detected from the working directory (pass ``--repo-root`` only if you run
from outside the checkout).

Usage:
    # @fbcode//mode/dev-nosan is required: the default dev mode links ASAN,
    # which disables CUDA in torch.
    buck2 run @fbcode//mode/dev-nosan \
        //third-party/triton/beta/triton:sched2tlx_regression -- --diff D108804400
    # or, with a plain triton-enabled python:
    python3 run_regression.py --diff D108804400 [--cases case1_simple_gemm,...]

Needs a Blackwell GPU for the checks (gated; reported as SKIP otherwise). Run
third_party/tlx/killgpu.sh if a launch hangs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Make sibling modules importable whether run as a script, a module, or inside a
# par (where __file__ lives in the extracted tree).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import emit_helpers as eh  # noqa: E402
import perf_engine as pe  # noqa: E402

_UNSUPPORTED_MARKER = "_unsupported"


def _gpu_ok() -> bool:
    # Direct torch check (Blackwell == compute capability major 10/11) avoids
    # pulling triton._internal_testing, which imports pytest.
    try:
        import torch

        if not torch.cuda.is_available():
            print("GPU check: torch.cuda.is_available() is False")
            return False
        major, minor = torch.cuda.get_device_capability()
        bw = major in (10, 11)
        print(f"GPU check: device_count={torch.cuda.device_count()} capability={major}.{minor} blackwell={bw}")
        return bw
    except Exception as e:  # noqa: BLE001 - any import/runtime failure means "no usable GPU"
        print(f"GPU check raised: {type(e).__name__}: {str(e)[:160]}")
        return False


def _all_ok(res: dict | None) -> bool:
    return bool(res) and all(r["ok"] for r in res["shapes"])


def _max_gh(res: dict) -> float | None:
    vals = [r["rel_gen_vs_hw"] for r in res["shapes"] if r.get("rel_gen_vs_hw") is not None]
    return max(vals) if vals else None


def _perf_compare(before: dict | None, after: dict, tol: float) -> tuple[str, str]:
    if not _all_ok(before):
        return "OK", "BEFORE-BROKEN → after validated"
    before_ms = {tuple(r["shape"]): r["gen_ms"] for r in before["shapes"]}
    worst, worst_shape = 0.0, None
    for r in after["shapes"]:
        b = before_ms.get(tuple(r["shape"]))
        if not b or b <= 0:
            continue
        ratio = r["gen_ms"] / b
        if ratio > worst:
            worst, worst_shape = ratio, tuple(r["shape"])
    if worst > 1 + tol:
        return "FAIL", f"REGRESSION {worst:.2f}x slower at {worst_shape}"
    return "OK", f"{worst:.2f}x slowest-shape (no regression)"


def _run_bench_safe(side, case: str, gen_path: Path, want_hw: bool) -> dict | None:
    try:
        return pe.run_bench(side.examples / case, gen_path, want_hw=want_hw)
    except Exception as e:  # noqa: BLE001 - a kernel that faults is a data point, not a crash
        print(f"    ({'after' if want_hw else 'before'} bench raised {type(e).__name__}: {str(e)[:120]})", flush=True)
        return None


def _evaluate(before_tree, after_tree, case: str, before_gen: Path, after_gen: Path,
              tol: float) -> tuple[str, str, str, str]:
    """Return (correctness, corr_detail, perf, perf_detail) for one case."""
    after = _run_bench_safe(after_tree, case, after_gen, want_hw=True)
    if after is None:
        # after couldn't even run; only a regression if before could.
        before = _run_bench_safe(before_tree, case, before_gen, want_hw=True)
        if _all_ok(before):
            return "FAIL", "REGRESSION: after kernel errors, before worked", "SKIP", ""
        return "PREEXIST", "broken before+after (after errors)", "SKIP", ""

    if _all_ok(after):
        gh = _max_gh(after)
        corr_detail = "gen==hw" + (f" (max rel {gh:.1e})" if gh is not None else " (no hw)")
        before = _run_bench_safe(before_tree, case, before_gen, want_hw=False)
        perf, perf_detail = _perf_compare(before, after, tol)
        return "PASS", corr_detail, perf, perf_detail

    # after incorrect — classify vs before.
    bad = [r["shape"] for r in after["shapes"] if not r["ok"]]
    before = _run_bench_safe(before_tree, case, before_gen, want_hw=True)
    if _all_ok(before):
        return "FAIL", f"REGRESSION: ok before, wrong after at {bad}", "SKIP", "skipped (incorrect)"
    return "PREEXIST", f"broken before+after at {bad}", "SKIP", "skipped (incorrect)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", default=".", help="revision to test (before=<diff>^, after=<diff>); default current commit")
    ap.add_argument("--before", help="explicit before revision (overrides --diff)")
    ap.add_argument("--after", help="explicit after revision (overrides --diff)")
    ap.add_argument("--cases", help="comma-separated case dir names (default: all discovered)")
    ap.add_argument("--perf-tol", type=float, default=0.05, help="allowed per-shape slowdown fraction (default 0.05)")
    ap.add_argument("--repo-root", help="fbsource checkout root (default: auto-detect from cwd)")
    ap.add_argument("--keep", action="store_true", help="keep temp before/after trees for debugging")
    args = ap.parse_args(argv)

    before_rev = args.before or f"{args.diff}^"
    after_rev = args.after or args.diff

    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        repo_root = eh.find_repo_root(Path.cwd())

    gpu = _gpu_ok()
    if not gpu:
        print("NOTE: no Blackwell GPU detected — correctness/perf layers will be SKIPPED.\n")

    workdir = Path(tempfile.mkdtemp(prefix="s2t_regression_"))
    print(f"repo: {repo_root}")
    print(f"diff: before={before_rev}  after={after_rev}")
    print(f"workdir: {workdir}\n")

    before_tree, after_tree = eh.materialize(repo_root, workdir, before_rev, after_rev)

    discovered = sorted(
        p.name for p in after_tree.examples.iterdir()
        if p.is_dir() and p.name.startswith("case")
    )
    selected = args.cases.split(",") if args.cases else discovered

    print(f"{'case':<24}{'correctness':<12}{'perf':<8} detail", flush=True)
    print("-" * 92, flush=True)

    results = []
    failures = 0
    for case in selected:
        if not (after_tree.examples / case).is_dir():
            results.append((case, "SKIP", "SKIP", "no such case dir"))
            print(f"{case:<24}{'SKIP':<12}{'SKIP':<8} no such case dir", flush=True)
            continue

        try:
            before_gen = eh.emit_case_inproc(before_tree, case)
            after_gen = eh.emit_case_inproc(after_tree, case)
        except Exception as e:  # noqa: BLE001
            results.append((case, "FAIL", "SKIP", f"emit failed: {str(e)[:60]}"))
            failures += 1
            print(f"{case:<24}{'FAIL':<12}{'SKIP':<8} emit failed: {str(e)[:60]}", flush=True)
            continue

        if _UNSUPPORTED_MARKER in after_gen.read_text():
            results.append((case, "SKIP", "SKIP", "emitter placeholder (<..._unsupported>)"))
            print(f"{case:<24}{'SKIP':<12}{'SKIP':<8} emitter placeholder (<..._unsupported>)", flush=True)
            continue

        if not gpu:
            results.append((case, "SKIP", "SKIP", "no GPU"))
            print(f"{case:<24}{'SKIP':<12}{'SKIP':<8} no GPU", flush=True)
            continue

        if not (after_tree.examples / case / "bench_spec.py").exists():
            results.append((case, "SKIP", "SKIP", "no bench_spec.py (not configured for this suite)"))
            print(f"{case:<24}{'SKIP':<12}{'SKIP':<8} no bench_spec.py (not configured for this suite)", flush=True)
            continue

        corr, corr_detail, perf, perf_detail = _evaluate(
            before_tree, after_tree, case, before_gen, after_gen, args.perf_tol
        )
        if corr == "FAIL" or perf == "FAIL":
            failures += 1
        detail = corr_detail if corr != "PASS" else f"{perf_detail}; {corr_detail}"
        results.append((case, corr, perf, detail))
        print(f"{case:<24}{corr:<12}{perf:<8} {detail}", flush=True)

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"\n(kept temp trees at {workdir})")

    print(f"\n{'FAILED' if failures else 'OK'}: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
