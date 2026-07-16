#!/usr/bin/env python3
"""Unified performance/correctness harness for the sched2tlx emitter.

This single module consolidates what used to be five files (``emit_helpers``,
``perf_engine``, ``run_regression``, ``e2e_worker`` and ``run_e2e``). It exposes
four subcommands, dispatched on the first CLI argument:

* ``regression`` — before-vs-after driver. Given any diff that touches the tool
  (default: the current commit) it re-emits every example case with the emitter
  BEFORE and AFTER the change and checks correctness (generated == handwritten ==
  torch reference) and performance (after is no slower than before, per shape).
  Runs the Python emitter only; ``schedule_graph.json`` is treated as a fixture.

* ``e2e`` — end-to-end orchestrator (pure-stdlib host script). Answers "does my
  change help or hurt vs master, end to end?" by running the WHOLE pipeline
  (``pre_modulo.ttgir`` → C++ modulo schedule → TLX emit → GPU run) on BOTH the
  current checkout and a master-latest checkout, then printing a comparison
  table. Not part of any buck target — run it with plain ``python3``.

* ``e2e-worker`` — per-branch worker invoked (once per branch) by ``e2e`` via
  ``buck2 run``. Runs the full pipeline for one built toolchain and writes one
  branch JSON. Runs inside the ``sched2tlx_e2e`` buck ``python_binary``.

* ``bench`` — benchmark a case's committed ``generated.py`` against its
  handwritten reference and print a human table (the classic
  ``perf_generated_vs_handwritten.py`` behavior), generically.

The module is import-safe under a plain interpreter: torch/triton are imported
lazily (only inside the benchmarking functions), and the tree-materialization and
host-orchestration halves are pure stdlib. See ``README.md`` for full usage.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo-root-relative path of the sched2tlx tool (the dir holding ``sched2tlx/``
# and ``examples/``). Kept relative so ``sl`` paths and tree copies line up.
TOOL_RELPATH = "third-party/triton/beta/triton/third_party/tlx/tools/sched2tlx"

# Only files under these tool subdirectories affect emission / running.
_RELEVANT_SUBDIRS = ("sched2tlx/", "examples/")

# A case whose emitted source still contains this marker has an unclosed emitter
# gap; such cases are auto-skipped (no perf/correctness run).
_UNSUPPORTED_MARKER = "_unsupported"


def _sl(repo_root: Path, args: list[str], reason: str) -> str:
    """Run a read-only ``sl`` command and return stdout (text)."""
    return subprocess.run(
        ["sl", *args, "--reason", reason],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


# ===========================================================================
# Tree materialization — build coherent before/after copies of the tool for any
# diff (Sapling ``sl status --change`` + ``sl cat``) and re-emit each case's
# ``generated.py`` on each side. Pure stdlib — no torch/triton/GPU needed.
#
# The regression driver compares the kernel produced by the emitter *before* a
# change against the one produced *after* it. To stay robust for an arbitrary
# diff (one that may touch more than ``emitter.py`` — including a case's
# ``schedule_graph.json`` or ``handwritten.py``) we materialize two
# self-contained copies of the tool tree, then overwrite only the diff-touched
# files with their pre-/post-diff content. Untouched files are identical across
# revisions, so the working-copy version already present is correct.
# ===========================================================================


def find_repo_root(start: Path) -> Path:
    """Walk upward until we find the fbsource checkout root."""
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / TOOL_RELPATH).is_dir():
            return parent
    raise RuntimeError(f"could not locate repo root containing {TOOL_RELPATH} from {start}")


def changed_tool_files(repo_root: Path, diff_rev: str) -> list[tuple[str, str]]:
    """Return ``(status_code, relpath)`` for diff-touched files under the tool.

    ``status_code`` is ``M``/``A``/``R`` from ``sl status --change``.
    """
    out = _sl(
        repo_root,
        ["status", "--change", diff_rev],
        "list files changed by diff for sched2tlx before/after regression - sl help status",
    )
    changed: list[tuple[str, str]] = []
    for line in out.splitlines():
        if len(line) < 3 or line[1] != " ":
            continue
        code, relpath = line[0], line[2:].strip()
        if not relpath.startswith(TOOL_RELPATH + "/"):
            continue
        suffix = relpath[len(TOOL_RELPATH) + 1:]
        if suffix.startswith(_RELEVANT_SUBDIRS):
            changed.append((code, relpath))
    return changed


def _ignore_pycache(_dir: str, names: list[str]) -> list[str]:
    return [n for n in names if n == "__pycache__"]


def _place_file(side_dir: Path, suffix: str, repo_root: Path, relpath: str, rev: str, present: bool) -> None:
    """Overwrite ``side_dir/suffix`` with file content at ``rev`` (or remove it)."""
    dst = side_dir / suffix
    if not present:
        if dst.exists():
            dst.unlink()
        return
    content = _sl(
        repo_root,
        ["cat", "-r", rev, relpath],
        "fetch file at revision for sched2tlx before/after regression - sl help cat",
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content)


@dataclass
class SideTree:
    """A materialized snapshot of the tool at one revision."""

    root: Path  # contains ``sched2tlx/`` (package) and ``examples/``

    @property
    def examples(self) -> Path:
        return self.root / "examples"


def materialize(repo_root: Path, workdir: Path, before_rev: str, after_rev: str) -> tuple[SideTree, SideTree]:
    """Build coherent before/after copies of the tool tree under ``workdir``."""
    tool = repo_root / TOOL_RELPATH
    before = SideTree(workdir / "before")
    after = SideTree(workdir / "after")
    for side in (before, after):
        if side.root.exists():
            shutil.rmtree(side.root)
        side.root.mkdir(parents=True)
        shutil.copytree(tool / "sched2tlx", side.root / "sched2tlx", ignore=_ignore_pycache)
        shutil.copytree(tool / "examples", side.root / "examples", ignore=_ignore_pycache)

    for code, relpath in changed_tool_files(repo_root, after_rev):
        suffix = relpath[len(TOOL_RELPATH) + 1:]
        _place_file(before.root, suffix, repo_root, relpath, before_rev, present=(code != "A"))
        _place_file(after.root, suffix, repo_root, relpath, after_rev, present=(code != "R"))

    return before, after


def _purge_sched2tlx_modules() -> None:
    for name in [m for m in sys.modules if m == "sched2tlx" or m.startswith("sched2tlx.")]:
        del sys.modules[name]


def emit_case(side: SideTree, case_name: str) -> Path:
    """Run ``python -m sched2tlx`` for one case on this side; return the .py path.

    Writes ``generated.py`` into the side's case dir, overwriting the copied
    placeholder. Raises CalledProcessError if emission fails. Subprocess form —
    needs a plain interpreter with the ``sched2tlx`` package importable. For
    buck/par execution use :func:`emit_case_inproc`.
    """
    case_dir = side.examples / case_name
    graph = case_dir / "schedule_graph.json"
    out = case_dir / "generated.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(side.root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-m", "sched2tlx", str(graph), "-o", str(out)],
        cwd=str(side.root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def emit_case_inproc(side: SideTree, case_name: str) -> Path:
    """Emit one case's ``generated.py`` in-process (works inside a buck par).

    Imports this side's ``sched2tlx`` package fresh (purging any other side's
    copy first so before/after emitters never collide) and calls ``emit``.
    """
    case_dir = side.examples / case_name
    graph = case_dir / "schedule_graph.json"
    out = case_dir / "generated.py"

    _purge_sched2tlx_modules()
    sys.path.insert(0, str(side.root))
    try:
        schedule_graph = importlib.import_module("sched2tlx.schedule_graph")
        emitter = importlib.import_module("sched2tlx.emitter")
        src = emitter.emit(schedule_graph.load_graph(str(graph)))
    finally:
        try:
            sys.path.remove(str(side.root))
        except ValueError:
            pass
        _purge_sched2tlx_modules()
    out.write_text(src)
    return out


# ===========================================================================
# Perf engine — CUDA-event timing, shape sweep, correctness guard, and
# throughput/ratio reporting. The per-case specifics live in a tiny
# ``bench_spec.py`` next to each example (see ``README.md``), so a new case is
# picked up automatically once it ships one. Requires torch + triton + a GPU
# (imported lazily, only when actually benchmarking).
# ===========================================================================


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _alloc_fn(size: int, alignment: int, stream: Any):  # noqa: ANN401
    import torch

    return torch.empty(size, device="cuda", dtype=torch.int8)


def _bench(call) -> float:
    """Median kernel latency in ms.

    Uses ``triton.testing.do_bench`` (warmup, many reps, L2-cache flush, robust
    median) — far more stable than a hand-rolled CUDA-event loop, which matters
    for the before/after comparison on small/fast kernels where jitter would
    otherwise masquerade as a regression.
    """
    import triton

    ms = triton.testing.do_bench(call, warmup=50, rep=200, quantiles=[0.5])
    return float(ms[0] if isinstance(ms, (list, tuple)) else ms)


def run_bench(case_dir: Path, generated_path: Path, want_hw: bool = True) -> dict[str, Any]:
    """Benchmark the generated kernel (and optionally handwritten) for one case.

    Returns a JSON-serializable dict with a per-shape row list. Each row carries
    ``gen_ms``, ``hw_ms`` (or None), ``throughput``/``unit``, the relative error
    of the generated output vs the torch reference, and an ``ok`` flag.
    """
    import torch
    import triton

    triton.set_allocator(_alloc_fn)
    torch.manual_seed(0)

    spec = _load_module(case_dir / "bench_spec.py", f"bench_spec_{case_dir.name}")
    gen = _load_module(generated_path, f"gen_{case_dir.name}")
    hw = None
    hw_path = case_dir / "handwritten.py"
    if want_hw and hw_path.exists():
        try:
            hw = _load_module(hw_path, f"hw_{case_dir.name}")
        except Exception:  # noqa: BLE001 - handwritten import is best-effort for the table
            hw = None

    def _rel(a, b) -> float:
        denom = max(b.float().abs().max().item(), 1e-9)
        return (a.float() - b.float()).abs().max().item() / denom

    rows: list[dict[str, Any]] = []
    for shape in spec.SHAPES:
        inputs = spec.make_inputs(shape)

        # Correctness: generated vs torch reference, and (the user's ask) the
        # generated output directly vs the hand-written reference output.
        out = spec.gen_call(gen, inputs)
        torch.cuda.synchronize()
        ref = spec.reference(inputs)
        rel_ref = _rel(out, ref)
        nan = int(torch.isnan(out).sum().item())

        rel_gh = None
        rel_hw_ref = None
        hw_ms = None
        if hw is not None:
            try:
                hout = spec.hw_call(hw, inputs)
                torch.cuda.synchronize()
                rel_gh = _rel(out, hout)
                rel_hw_ref = _rel(hout, ref)
            except Exception:  # noqa: BLE001 - hw is only a reference baseline
                hout = None
            if hout is not None:
                hw_ms = _bench(lambda inp=inputs: spec.hw_call(hw, inp))

        # Re-run the generated kernel for timing after the hw call (which shares
        # output buffers in some specs) so gen output isn't left clobbered.
        gen_ms = _bench(lambda inp=inputs: spec.gen_call(gen, inp))

        ok = bool(rel_ref <= spec.TOL and nan == 0 and (rel_gh is None or rel_gh <= spec.TOL))

        work, scale, unit = spec.metric(shape)
        rows.append(
            {
                "shape": list(shape),
                "gen_ms": gen_ms,
                "hw_ms": hw_ms,
                "throughput": work / (gen_ms * 1e-3) / scale,
                "hw_throughput": (work / (hw_ms * 1e-3) / scale) if hw_ms else None,
                "unit": unit,
                "rel": rel_ref,
                "rel_gen_vs_hw": rel_gh,
                "rel_hw_vs_ref": rel_hw_ref,
                "ok": ok,
            }
        )
    return {"case": case_dir.name, "shapes": rows}


def _print_table(result: dict[str, Any]) -> None:
    print(f"\n== {result['case']} ==")
    print(f"{'shape':<22}{'gen':<12}{'hw':<12}{'gen/hw':<9}{'rel':<10}{'ok'}")
    print("-" * 75)
    for r in result["shapes"]:
        unit = r["unit"]
        gen = f"{r['throughput']:.1f} {unit}"
        if r["hw_throughput"]:
            hw = f"{r['hw_throughput']:.1f} {unit}"
            ratio = f"{r['throughput'] / r['hw_throughput']:.2f}x"
        else:
            hw, ratio = "-", "-"
        shape = "(" + ",".join(str(x) for x in r["shape"]) + ")"
        print(f"{shape:<22}{gen:<12}{hw:<12}{ratio:<9}{r['rel']:<10.2e}{'PASS' if r['ok'] else 'FAIL'}")


def bench_main(argv: list[str] | None = None) -> int:
    """``bench`` subcommand: human table over each case's committed generated.py."""
    ap = argparse.ArgumentParser(
        prog="perf_harness.py bench",
        description="benchmark cases' committed generated.py vs handwritten",
    )
    ap.add_argument("--cases", help="comma-separated case dir names (default: all with bench_spec.py)")
    args = ap.parse_args(argv)

    examples_dir = Path(__file__).resolve().parents[2]  # .../examples (perf_regression/ → testing/ → examples/)
    if args.cases:
        names = args.cases.split(",")
    else:
        names = [
            p.name
            for p in sorted(examples_dir.iterdir())
            if p.is_dir() and p.name.startswith("case") and (p / "bench_spec.py").exists()
        ]
    for name in names:
        case_dir = examples_dir / name
        gen = case_dir / "generated.py"
        if not (case_dir / "bench_spec.py").exists() or not gen.exists():
            print(f"== {name} == SKIP (missing bench_spec.py or generated.py)")
            continue
        _print_table(run_bench(case_dir, gen))
    return 0


def compare_main(argv: list[str] | None = None) -> int:
    """``compare`` subcommand: one row per case, ``--rev`` vs working tree.

    Compares committed KERNELS (each side's generated.py fixture), both run
    under the CURRENT build with the CURRENT bench_spec — cheap and exact when
    a diff only changes what the emitter produces. It does NOT rebuild the
    other revision's toolchain; for a full before/after of the emitter itself
    use ``regression``, and for a separately built master use ``e2e``.

    Scans every ``case*/`` dir in examples/ so new cases appear automatically;
    cases without a bench_spec.py are listed (not silently dropped). A case
    whose generated.py is byte-identical on both sides is measured once and
    reported "unchanged" — anything else would just be re-measuring noise.
    """
    ap = argparse.ArgumentParser(
        prog="perf_harness.py compare",
        description="per-case gen/hw summary table: a git revision's generated.py vs the working tree's",
    )
    ap.add_argument("--rev", default="origin/main", help="git revision for the left column (default: origin/main)")
    ap.add_argument("--cases", help="comma-separated case dir names (default: every case*/ in examples/)")
    args = ap.parse_args(argv)

    examples_dir = Path(__file__).resolve().parents[2]
    repo_root = Path(
        subprocess.check_output(
            ["git", "-C", str(examples_dir), "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    names = args.cases.split(",") if args.cases else sorted(
        p.name for p in examples_dir.iterdir() if p.is_dir() and p.name.startswith("case")
    )

    def _cell(result: dict[str, Any]) -> str:
        # Per-shape gen/hw ratios in SHAPES order; raw throughput when the
        # case has no handwritten baseline. Any correctness failure taints
        # the whole cell — a fast wrong kernel must not read as a win.
        parts = []
        for r in result["shapes"]:
            if r["hw_throughput"]:
                parts.append(f"{r['throughput'] / r['hw_throughput']:.2f}x")
            else:
                parts.append(f"{r['throughput']:.0f} {r['unit']}")
        if not all(r["ok"] for r in result["shapes"]):
            parts.append("FAIL")
        return ", ".join(parts) if parts else "-"

    rows: list[tuple[str, str, str]] = []
    for name in names:
        case_dir = examples_dir / name
        gen = case_dir / "generated.py"
        if not (case_dir / "bench_spec.py").exists() or not gen.exists():
            rows.append((name, "(no bench_spec)", "(no bench_spec)"))
            continue
        relpath = gen.resolve().relative_to(repo_root).as_posix()
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{args.rev}:{relpath}"],
            capture_output=True,
        )
        if proc.returncode != 0:
            rows.append((name, f"(not on {args.rev})", _cell(run_bench(case_dir, gen))))
        elif proc.stdout == gen.read_bytes():
            rows.append((name, _cell(run_bench(case_dir, gen)), "unchanged"))
        else:
            with tempfile.TemporaryDirectory() as td:
                rev_gen = Path(td) / "generated.py"
                rev_gen.write_bytes(proc.stdout)
                left = _cell(run_bench(case_dir, rev_gen))
            rows.append((name, left, _cell(run_bench(case_dir, gen))))

    left_hdr = (
        "master"
        if args.rev in ("origin/main", "main", "origin/master", "master")
        else args.rev
    )
    right_hdr = "branch"
    w0 = max(len("case"), *(len(r[0]) for r in rows)) + 2
    w1 = max(len(left_hdr), *(len(r[1]) for r in rows)) + 2
    print(f"\n{'case':<{w0}}{left_hdr:<{w1}}{right_hdr}")
    print("-" * (w0 + w1 + len(right_hdr)))
    for name, left, right in rows:
        print(f"{name:<{w0}}{left:<{w1}}{right}")
    print(
        "\ncells: per-shape gen/handwritten ratios (SHAPES order); both columns run"
        "\nunder the current build; 'unchanged' = byte-identical generated.py."
    )
    return 0


# ===========================================================================
# Regression driver (``regression`` subcommand) — before-vs-after emitter check.
#
# Given any diff that touches the sched2tlx tool (default: the current commit),
# re-emit every example case with the emitter BEFORE and AFTER the diff and, for
# each auto-discovered case, check correctness and performance. Both checks are
# before-vs-after aware: a case already broken/slow *before* the diff is not
# counted against it (only a true regression fails). Everything runs in-process
# so it works both under a plain triton-enabled interpreter and inside a buck
# ``python_binary`` (par). Needs a Blackwell GPU (gated; SKIP otherwise).
# ===========================================================================


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
        return run_bench(side.examples / case, gen_path, want_hw=want_hw)
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


def regression_main(argv: list[str] | None = None) -> int:
    """Before-vs-after emitter check for a diff: re-emit every case before and
    after the change and check correctness (gen==hw==torch) and per-shape perf
    (after no slower than before). Needs a Blackwell GPU (SKIP otherwise)."""
    ap = argparse.ArgumentParser(
        prog="perf_harness.py regression",
        description=regression_main.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        repo_root = find_repo_root(Path.cwd())

    gpu = _gpu_ok()
    if not gpu:
        print("NOTE: no Blackwell GPU detected — correctness/perf layers will be SKIPPED.\n")

    workdir = Path(tempfile.mkdtemp(prefix="s2t_regression_"))
    print(f"repo: {repo_root}")
    print(f"diff: before={before_rev}  after={after_rev}")
    print(f"workdir: {workdir}\n")

    before_tree, after_tree = materialize(repo_root, workdir, before_rev, after_rev)

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
            before_gen = emit_case_inproc(before_tree, case)
            after_gen = emit_case_inproc(after_tree, case)
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


# ===========================================================================
# End-to-end per-branch worker (``e2e-worker`` subcommand).
#
# Runs the FULL pipeline for ONE branch (one built toolchain) and writes one
# branch JSON. For each selected case, starting from the case's
# ``*pre_modulo.ttgir`` (the fixed source of truth):
#   1. dump the modulo schedule via this branch's ``triton-opt``;
#   2. emit TLX via this branch's ``sched2tlx`` package (--emitter-root);
#   3. run + bench in a fork-isolated child so a kernel that faults at scale
#      cannot poison the parent's / other cases' CUDA context.
# Only the toolchain under test varies between branches; the test definition
# (TTGIR input, ``bench_spec.py``, ``handwritten.py``) is always read from
# --case-root (the current checkout). Runs inside the ``sched2tlx_e2e`` buck
# ``python_binary``; the ``e2e`` orchestrator invokes it once per branch.
# ===========================================================================


def _resolve_triton_opt(override: str | None) -> str:
    """Locate this branch's compiled ``triton-opt`` binary.

    Inside the par it is shipped as a resource (``get_file_path``). Outside buck
    (manual runs), fall back to ``--triton-opt`` or ``$PATH``.
    """
    if override:
        return override
    try:
        from libfb.py import parutil  # type: ignore

        return parutil.get_file_path("triton-opt")
    except Exception:  # noqa: BLE001 - not in a par; fall back to PATH
        found = shutil.which("triton-opt")
        if found:
            return found
    raise RuntimeError("could not resolve triton-opt; pass --triton-opt <path>")


def _find_ttgir(case_dir: Path) -> Path | None:
    hits = sorted(glob.glob(str(case_dir / "*pre_modulo.ttgir")))
    return Path(hits[0]) if hits else None


def _dump_schedule(triton_opt: str, ttgir: Path, out_json: Path) -> None:
    """pre_modulo.ttgir -> schedule_graph.json via the C++ modulo pass.

    The JSON is a side effect gated by ``TRITON_MODULO_DUMP_SCHEDULE`` (the real
    env var — ``design.md`` documents wrong names). The transformed IR is
    discarded (``-o /dev/null``); we only want the dumped graph.
    """
    env = dict(os.environ)
    env["TRITON_MODULO_DUMP_SCHEDULE"] = str(out_json)
    subprocess.run(
        [triton_opt, "-allow-unregistered-dialect", "--nvgpu-modulo-schedule", str(ttgir), "-o", os.devnull],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    if not out_json.exists():
        raise RuntimeError(f"triton-opt did not dump a schedule graph to {out_json}")


def _emit(emitter_root: Path, graph_json: Path, out_py: Path) -> str:
    """schedule_graph.json -> generated.py via THIS branch's emitter.

    Imports the ``sched2tlx`` package under ``emitter_root`` fresh (purging any
    stale copy first) so the branch under test's emitter is the one exercised.
    Returns the emitted source (so the caller can check for placeholders).
    """
    _purge_sched2tlx_modules()
    sys.path.insert(0, str(emitter_root))
    try:
        schedule_graph = importlib.import_module("sched2tlx.schedule_graph")
        emitter = importlib.import_module("sched2tlx.emitter")
        src = emitter.emit(schedule_graph.load_graph(str(graph_json)))
    finally:
        try:
            sys.path.remove(str(emitter_root))
        except ValueError:
            pass
        _purge_sched2tlx_modules()
    out_py.write_text(src)
    return src


def _bench_isolated(case_dir: Path, generated: Path, out_json: Path) -> dict[str, Any] | None:
    """Run ``run_bench`` in a forked child, returning its result.

    The parent stays CUDA-free (it never imports torch/triton), so forking is
    safe; the child owns a fresh CUDA context. If the child dies (e.g. an
    illegal memory access at scale) the parent survives and we report the case
    as errored rather than crashing the whole branch run.
    """
    if out_json.exists():
        out_json.unlink()
    pid = os.fork()
    if pid == 0:  # child
        try:
            res = run_bench(case_dir, generated, want_hw=True)
            out_json.write_text(json.dumps(res))
            os._exit(0)
        except BaseException as e:  # noqa: BLE001 - any failure is a data point, not a crash
            try:
                out_json.write_text(json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"}))
            except Exception:  # noqa: BLE001
                pass
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    if not out_json.exists():
        return {"error": f"bench child died (status {status})"}
    data = json.loads(out_json.read_text())
    return data


def run_case(
    case: str,
    case_root: Path,
    emitter_root: Path,
    triton_opt: str,
    tmp: Path,
) -> dict[str, Any]:
    """Run the full pipeline for one case; return a status dict."""
    case_dir = case_root / case
    if not case_dir.is_dir():
        return {"status": "skipped", "reason": "no such case dir"}

    ttgir = _find_ttgir(case_dir)
    if ttgir is None:
        return {"status": "skipped", "reason": "no *pre_modulo.ttgir"}
    if not (case_dir / "bench_spec.py").exists():
        return {"status": "skipped", "reason": "no bench_spec.py"}

    graph_json = tmp / f"{case}.schedule_graph.json"
    generated = tmp / f"{case}.generated.py"
    try:
        _dump_schedule(triton_opt, ttgir, graph_json)
    except subprocess.CalledProcessError as e:
        return {"status": "errored", "reason": f"modulo dump failed: {(e.stderr or '')[:200]}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "errored", "reason": f"modulo dump failed: {str(e)[:200]}"}

    try:
        src = _emit(emitter_root, graph_json, generated)
    except Exception as e:  # noqa: BLE001
        return {"status": "errored", "reason": f"emit failed: {str(e)[:200]}"}

    if _UNSUPPORTED_MARKER in src:
        return {"status": "skipped", "reason": "emitter placeholder (<..._unsupported>)"}

    result = _bench_isolated(case_dir, generated, tmp / f"{case}.bench.json")
    if result is None or "error" in result:
        reason = (result or {}).get("error", "bench produced no result")
        return {"status": "errored", "reason": reason}
    return {"status": "ran", "shapes": result["shapes"]}


def e2e_worker_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="perf_harness.py e2e-worker",
        description="per-branch end-to-end worker (invoked by the e2e orchestrator via buck2 run)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--case-root", required=True, help="examples/ dir of the CURRENT checkout (fixed test definition)")
    ap.add_argument("--emitter-root", required=True, help="this branch's sched2tlx tool dir (holds the sched2tlx/ package)")
    ap.add_argument("--cases", required=True, help="comma-separated case dir names")
    ap.add_argument("--branch", required=True, help="label for this run (current|master)")
    ap.add_argument("--node", default="", help="resolved commit hash of this branch")
    ap.add_argument("--arch", default="", help="nvcc_arch this build targets (for the header)")
    ap.add_argument("--cuda", default="", help="cuda version this build targets (for the header)")
    ap.add_argument("--gpu", default="", help="physical GPU index in use (for the header)")
    ap.add_argument("--triton-opt", help="override path to triton-opt (default: par resource / $PATH)")
    ap.add_argument("--out", required=True, help="path to write this branch's JSON result")
    args = ap.parse_args(argv)

    case_root = Path(args.case_root).resolve()
    emitter_root = Path(args.emitter_root).resolve()
    triton_opt = _resolve_triton_opt(args.triton_opt)
    cases = [c for c in args.cases.split(",") if c]

    print(f"[{args.branch}] node={args.node or '?'} arch={args.arch or '?'} cuda={args.cuda or '?'} gpu={args.gpu or '?'}", flush=True)
    print(f"[{args.branch}] triton-opt: {triton_opt}", flush=True)
    print(f"[{args.branch}] emitter-root: {emitter_root}", flush=True)

    out: dict[str, Any] = {
        "branch": args.branch,
        "node": args.node,
        "arch": args.arch,
        "cuda": args.cuda,
        "gpu": args.gpu,
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix=f"s2t_e2e_{args.branch}_") as td:
        tmp = Path(td)
        for case in cases:
            print(f"[{args.branch}] {case} ...", flush=True)
            res = run_case(case, case_root, emitter_root, triton_opt, tmp)
            out["cases"][case] = res
            print(f"[{args.branch}] {case}: {res['status']}" + (f" ({res.get('reason')})" if res.get("reason") else ""), flush=True)

    Path(args.out).write_text(json.dumps(out))
    print(f"[{args.branch}] wrote {args.out}", flush=True)
    return 0


# ===========================================================================
# End-to-end orchestrator (``e2e`` subcommand) — pure-stdlib host script.
#
# For every in-scope case, runs the full pipeline (pre_modulo.ttgir -> modulo
# schedule -> TLX emit -> GPU run) on BOTH the current branch and master-latest,
# then prints a table comparing each branch's throughput as a fraction of the
# hand-written reference plus the current/master speedup. Correctness is a hard
# gate; performance is report-only. master-latest is built/run from a SEPARATE
# checkout (--master-repo), never by mutating the working checkout. It shells out
# to ``buck2 run`` per branch — run it with a plain ``python3``, NOT via buck.
# ===========================================================================

_EXAMPLES_REL = TOOL_RELPATH + "/examples"
_BUCK_TARGET = "fbsource//third-party/triton/beta/triton:sched2tlx_e2e"

_DEFAULT_CASES = ["case1_simple_gemm", "case2_persistent_gemm", "case3_FA", "case6_layernorm"]


def _nvidia_smi(query: str) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def detect_arch(arch_override: str | None, cuda_override: str | None) -> tuple[str, str]:
    """Return (nvcc_arch, cuda_version) for the local GPU.

    cap 9   -> h100a / 12.8 (Hopper)
    cap 10.0 -> b200a / 12.8 (B200 / GB200 / B100)
    cap 10.3 / 11 / GB300 by name -> b300a / 13.0 (B300 / GB300)
    """
    if arch_override and cuda_override:
        return arch_override, cuda_override
    caps = _nvidia_smi("compute_cap")
    names = _nvidia_smi("name")
    cap = caps[0] if caps else ""
    name = (names[0] if names else "").upper()
    major, minor = 0, 0
    m = re.match(r"(\d+)\.(\d+)", cap)
    if m:
        major, minor = int(m.group(1)), int(m.group(2))

    if "B300" in name or "GB300" in name or (major == 11) or (major == 10 and minor >= 3):
        arch, cuda = "b300a", "13.0"
    elif major == 10:
        arch, cuda = "b200a", "12.8"
    elif major == 9:
        arch, cuda = "h100a", "12.8"
    else:
        raise RuntimeError(f"unsupported GPU (cap={cap!r} name={name!r}); pass --arch/--cuda explicitly")
    return arch_override or arch, cuda_override or cuda


def pick_gpu(gpu_override: str | None) -> str:
    """Pick a free GPU using only nvidia-smi (no torch — the host is stdlib-only).

    Excludes GPUs with a running compute process, then takes the one with the
    least memory used. The worker (which has torch via buck) does the real
    busy-check when it launches the kernel; this only chooses a sane candidate.
    """
    if gpu_override is not None:
        return gpu_override
    rows = _nvidia_smi("index,uuid,memory.used")  # "0, GPU-xxx, 4"
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    busy_uuids = {ln.strip() for ln in apps.splitlines() if ln.strip()}

    candidates: list[tuple[int, str]] = []  # (memory_used, index)
    fallback: list[tuple[int, str]] = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 3:
            continue
        idx, uuid, mem = parts[0], parts[1], parts[2]
        try:
            mem_i = int(float(mem))
        except ValueError:
            mem_i = 0
        fallback.append((mem_i, idx))
        if uuid not in busy_uuids:
            candidates.append((mem_i, idx))
    pool = candidates or fallback
    if not pool:
        raise RuntimeError("no GPUs reported by nvidia-smi")
    return min(pool)[1]


def resolve_node(repo: Path, rev: str) -> str:
    return _sl(repo, ["log", "-r", rev, "-T", "{node}"], f"resolve {rev} for sched2tlx e2e harness - sl help log").strip()


def prepare_master(master_repo: Path, master_rev: str | None) -> str:
    """Pull + goto master-latest in the separate master checkout; return the node.

    Guarded by a lock so concurrent runs don't race the pull/goto. Never touches
    the current working checkout.
    """
    import fcntl

    lock_path = master_repo / ".s2t_e2e.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _sl(master_repo, ["pull"], "pull master for sched2tlx e2e harness - sl help pull")
        node = resolve_node(master_repo, master_rev or "remote/master")
        _sl(master_repo, ["goto", node], "goto master tip for sched2tlx e2e harness - sl help goto")
    return node


def run_branch(
    branch: str,
    repo: Path,
    case_root: Path,
    cases: list[str],
    node: str,
    arch: str,
    cuda: str,
    gpu: str,
    out_json: Path,
) -> None:
    emitter_root = repo / TOOL_RELPATH
    cmd = [
        "buck2", "run", "@mode/opt", "-m", "ovr_config//triton:beta",
        "-c", f"fbcode.nvcc_arch={arch}",
    ]
    if cuda:
        cmd += ["-c", f"fbcode.platform010_cuda_version={cuda}"]
    cmd += [
        _BUCK_TARGET, "--", "e2e-worker",
        "--case-root", str(case_root),
        "--emitter-root", str(emitter_root),
        "--cases", ",".join(cases),
        "--branch", branch,
        "--node", node,
        "--arch", arch,
        "--cuda", cuda,
        "--gpu", gpu,
        "--out", str(out_json),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"\n=== building + running [{branch}] from {repo}/fbcode (gpu {gpu}) ===", flush=True)
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo / "fbcode"), env=env, check=True)


def _rows_by_shape(branch_json: dict[str, Any], case: str) -> dict[tuple, dict[str, Any]]:
    entry = branch_json.get("cases", {}).get(case, {})
    if entry.get("status") != "ran":
        return {}
    return {tuple(r["shape"]): r for r in entry["shapes"]}


def _pct_hw(row: dict[str, Any]) -> str:
    hw = row.get("hw_throughput")
    if not hw:
        return "-"
    return f"{row['throughput'] / hw * 100:.0f}%"


def print_table_and_gate(cur: dict[str, Any], mas: dict[str, Any], cases: list[str]) -> int:
    print(
        f"\n=== sched2tlx e2e: master({mas.get('node','?')[:12]}) vs current({cur.get('node','?')[:12]})  "
        f"arch={cur.get('arch')} cuda={cur.get('cuda')} gpu={cur.get('gpu')} ===\n"
    )
    hdr = f"{'case':<22}{'shape':<20}{'master%hw':<11}{'current%hw':<12}{'speedup':<9}{'corr'}"
    print(hdr)
    print("-" * len(hdr))

    failures = 0
    for case in cases:
        cur_entry = cur.get("cases", {}).get(case, {})
        mas_entry = mas.get("cases", {}).get(case, {})
        cur_rows = _rows_by_shape(cur, case)
        mas_rows = _rows_by_shape(mas, case)

        # A case that was expected to run (has a bench def) but errored on a
        # branch is a hard correctness failure.
        for entry, label in ((cur_entry, "current"), (mas_entry, "master")):
            if entry.get("status") == "errored":
                failures += 1
                print(f"{case:<22}{'(errored)':<20}{'-':<11}{'-':<12}{'-':<9}FAIL [{label}: {entry.get('reason','')[:40]}]")

        if not cur_rows and not mas_rows:
            if cur_entry.get("status") == "skipped" or mas_entry.get("status") == "skipped":
                reason = cur_entry.get("reason") or mas_entry.get("reason") or ""
                print(f"{case:<22}{'(skipped)':<20}{'-':<11}{'-':<12}{'-':<9}SKIP  {reason[:40]}")
            continue

        shapes = sorted(set(cur_rows) | set(mas_rows))
        for shape in shapes:
            c = cur_rows.get(shape)
            m = mas_rows.get(shape)
            shape_s = "(" + ",".join(str(x) for x in shape) + ")"
            mas_pct = _pct_hw(m) if m else "-"
            cur_pct = _pct_hw(c) if c else "-"
            if c and m and m["throughput"] > 0:
                speedup = f"{c['throughput'] / m['throughput']:.2f}x"
            else:
                speedup = "-"
            ok = True
            if c is not None and not c["ok"]:
                ok = False
            if m is not None and not m["ok"]:
                ok = False
            corr = "PASS" if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"{case:<22}{shape_s:<20}{mas_pct:<11}{cur_pct:<12}{speedup:<9}{corr}")

    print()
    if failures:
        print(f"FAILED: {failures} correctness failure(s)")
    else:
        print("OK: 0 correctness failure(s)")
    return failures


def e2e_main(argv: list[str] | None = None) -> int:
    """Current-branch-vs-master end-to-end comparison: run the full pipeline
    (pre_modulo.ttgir -> modulo schedule -> TLX emit -> GPU run) on both the
    current checkout and a separate master checkout, then print a throughput
    table. Correctness is a hard gate; performance is report-only. Pure-stdlib
    host orchestrator — run with plain python3, not via buck."""
    ap = argparse.ArgumentParser(
        prog="perf_harness.py e2e",
        description=e2e_main.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--current-repo", help="current fbsource checkout (default: auto-detect from cwd)")
    ap.add_argument("--master-repo", help="separate fbsource checkout used to build master-latest")
    ap.add_argument("--master-rev", help="explicit master revision to pin (default: remote/master tip)")
    ap.add_argument("--cases", help="comma-separated case dir names (default: the 4 in-scope cases)")
    ap.add_argument("--arch", help="override nvcc_arch (default: auto-detect)")
    ap.add_argument("--cuda", help="override cuda version (default: auto-detect)")
    ap.add_argument("--gpu", help="override physical GPU index (default: find_working_gpu.sh)")
    ap.add_argument("--skip-master", action="store_true", help="run only the current branch (quick sanity)")
    ap.add_argument("--keep", action="store_true", help="keep the temp JSON dir")
    args = ap.parse_args(argv)

    current_repo = Path(args.current_repo).resolve() if args.current_repo else find_repo_root(Path.cwd())
    cases = [c for c in (args.cases.split(",") if args.cases else _DEFAULT_CASES) if c]
    case_root = current_repo / _EXAMPLES_REL

    arch, cuda = detect_arch(args.arch, args.cuda)
    gpu = pick_gpu(args.gpu)
    print(f"arch={arch} cuda={cuda} gpu={gpu}")

    cur_node = resolve_node(current_repo, ".")

    if not args.skip_master:
        if not args.master_repo:
            raise SystemExit("--master-repo is required (a second checkout to build master); or pass --skip-master")
        master_repo = Path(args.master_repo).resolve()
        if master_repo == current_repo:
            raise SystemExit("--master-repo must be a DIFFERENT checkout from --current-repo")
        print(f"preparing master in {master_repo} ...", flush=True)
        mas_node = prepare_master(master_repo, args.master_rev)
        print(f"master node: {mas_node}", flush=True)

    workdir = Path(tempfile.mkdtemp(prefix="s2t_e2e_out_"))
    cur_out = workdir / "current.json"
    mas_out = workdir / "master.json"
    try:
        run_branch("current", current_repo, case_root, cases, cur_node, arch, cuda, gpu, cur_out)
        cur = json.loads(cur_out.read_text())
        if args.skip_master:
            mas = {"cases": {}, "node": "(skipped)", "arch": arch, "cuda": cuda, "gpu": gpu}
        else:
            run_branch("master", master_repo, case_root, cases, mas_node, arch, cuda, gpu, mas_out)
            mas = json.loads(mas_out.read_text())
        failures = print_table_and_gate(cur, mas, cases)
    finally:
        if args.keep:
            print(f"(kept temp JSON at {workdir})")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    return 1 if failures else 0


# ===========================================================================
# Subcommand dispatch. The two buck targets (``sched2tlx_regression`` and
# ``sched2tlx_e2e``) share this module as their ``main_module``; they pass the
# ``regression`` / ``e2e-worker`` subcommand as the first argument after ``--``.
# ===========================================================================

_SUBCOMMANDS = {
    "regression": regression_main,
    "e2e": e2e_main,
    "e2e-worker": e2e_worker_main,
    "bench": bench_main,
    "compare": compare_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _SUBCOMMANDS:
        print(f"usage: perf_harness.py {{{'|'.join(_SUBCOMMANDS)}}} [args...]", file=sys.stderr)
        return 2
    return _SUBCOMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
