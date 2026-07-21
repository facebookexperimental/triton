#!/usr/bin/env python3
"""Performance/correctness harness for the sched2tlx emitter.

One subcommand, ``compare``: one row per case, four columns —

* ``case`` — the case dir (relative to ``examples/``);
* ``<rev> (gen/hw)`` — per-shape gen/handwritten throughput ratios for the
  ``--rev`` revision's committed ``generated.py`` (default ``origin/main``);
* ``branch (gen/hw)`` — the same for the working tree's ``generated.py``;
* ``improvement`` — per-shape % change of the branch's generated-kernel
  throughput over ``<rev>``'s (positive = the branch is faster).

Both columns run under the CURRENT build with the CURRENT ``bench_spec.py`` —
the harness compares committed KERNEL fixtures, not toolchains. Cases whose
``generated.py`` is byte-identical on both sides are measured once and shown
as "unchanged". Correctness is checked before timing; any failing shape taints
the cell with a FAIL marker — a fast wrong kernel must not read as a win.

Per-case specifics live in a tiny ``bench_spec.py`` next to each example (see
``README.md`` for the contract). Specs are discovered RECURSIVELY under
``examples/``, so nested variant dirs (``case9_scaled_mm/blockwise``) are
covered; top-level ``case*/`` dirs without any spec are still listed (never
silently dropped). Requires torch + triton + a GPU (imported lazily).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


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
    for the two-sided comparison on small/fast kernels where jitter would
    otherwise masquerade as a difference.
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

        # Correctness: generated vs torch reference, and the generated output
        # directly vs the hand-written reference output.
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


def _measure(case_dir: Path, gen_path: Path):
    """``run_bench`` in a fork-isolated child: a kernel that faults (or leaves a
    sticky CUDA error behind) is a table cell, not poison for every subsequent
    cell. The parent never touches CUDA — torch/triton are imported only in the
    child — so forking is safe and each measurement gets a fresh context."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_json = Path(tf.name)
    try:
        pid = os.fork()
        if pid == 0:  # child
            code = 0
            try:
                out_json.write_text(json.dumps(run_bench(case_dir, gen_path)))
            except BaseException as e:  # noqa: BLE001 - any failure is a data point
                code = 1
                try:
                    out_json.write_text(json.dumps({"error": f"{type(e).__name__}: {str(e)[:120]}"}))
                except Exception:  # noqa: BLE001
                    pass
            os._exit(code)
        _, status = os.waitpid(pid, 0)
        text = out_json.read_text() if out_json.exists() else ""
        if not text.strip():
            return f"(error: bench child died, status {status})"
        data = json.loads(text)
        if "error" in data:
            return f"(error: {data['error'][:70]})"
        return data
    finally:
        out_json.unlink(missing_ok=True)


def _cell(result) -> str:
    """Per-shape gen/hw ratios in SHAPES order; raw generated throughput when
    the case has no handwritten baseline. Any correctness failure taints the
    whole cell."""
    if isinstance(result, str):
        return result
    parts = []
    for r in result["shapes"]:
        if r["hw_throughput"]:
            parts.append(f"{r['throughput'] / r['hw_throughput']:.2f}x")
        else:
            parts.append(f"{r['throughput']:.0f} {r['unit']}")
    if not all(r["ok"] for r in result["shapes"]):
        parts.append("FAIL")
    return ", ".join(parts) if parts else "-"


def _improvement(left, right) -> str:
    """Per-shape % gain of the right side's GENERATED-kernel throughput over the
    left's. Computed from gen throughput directly (not the gen/hw ratios) so
    handwritten-side noise never leaks into the number. A shape that fails
    correctness on either side renders FAIL instead of a percentage — a fast
    wrong kernel must not read as a win."""
    if isinstance(left, str) or isinstance(right, str):
        return "-"
    base = {tuple(r["shape"]): r for r in left["shapes"]}
    parts = []
    for r in right["shapes"]:
        b = base.get(tuple(r["shape"]))
        if b is None:
            parts.append("-")
        elif not (r["ok"] and b["ok"]):
            parts.append("FAIL")
        else:
            parts.append(f"{(r['throughput'] / b['throughput'] - 1) * 100:+.1f}%")
    return ", ".join(parts) if parts else "-"


def compare_main(argv: list[str] | None = None) -> int:
    """``compare`` subcommand: one row per case, ``--rev`` vs working tree."""
    ap = argparse.ArgumentParser(
        prog="perf_harness.py compare",
        description="per-case gen/hw summary table: a git revision's generated.py vs the working tree's",
    )
    ap.add_argument("--rev", default="origin/main", help="git revision for the left column (default: origin/main)")
    ap.add_argument("--cases", help="comma-separated case dir names relative to examples/ (default: all discovered)")
    args = ap.parse_args(argv)

    examples_dir = Path(__file__).resolve().parents[2]  # .../examples (perf_regression/ → testing/ → examples/)
    repo_root = Path(
        subprocess.check_output(
            ["git", "-C", str(examples_dir), "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    if subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{args.rev}^{{commit}}"],
        capture_output=True,
    ).returncode != 0:
        print(f"error: --rev '{args.rev}' is not a commit in this repo", file=sys.stderr)
        return 2

    # Recursive spec discovery: any dir under examples/ holding a bench_spec.py
    # is a case (nested variants included). Top-level case*/ dirs with no spec
    # anywhere beneath them are still listed, never silently dropped.
    spec_names = sorted(
        p.parent.relative_to(examples_dir).as_posix() for p in examples_dir.rglob("bench_spec.py")
    )
    no_spec = [
        p.name
        for p in sorted(examples_dir.iterdir())
        if p.is_dir()
        and p.name.startswith("case")
        and not any(n == p.name or n.startswith(p.name + "/") for n in spec_names)
    ]
    if args.cases:
        names = []
        for n in (s.strip() for s in args.cases.split(",")):
            if not n:
                continue
            d = examples_dir / n
            if not (d / "bench_spec.py").exists() and d.is_dir():
                nested = sorted(
                    p.parent.relative_to(examples_dir).as_posix() for p in d.rglob("bench_spec.py")
                )
                if nested:  # e.g. --cases case9_scaled_mm → its variant subdirs
                    names.extend(nested)
                    continue
            names.append(n)
    else:
        names = sorted(spec_names + no_spec)

    # Each row is (case, left cell, right cell, improvement, bold_branch) —
    # bold_branch marks rows whose branch kernel was actually re-measured (the
    # fixture differs from --rev), so a scanning eye lands on what changed.
    rows: list[tuple[str, str, str, str, bool]] = []
    for name in names:
        case_dir = examples_dir / name
        gen = case_dir / "generated.py"
        if not case_dir.is_dir():
            rows.append((name, "(no such case dir)", "(no such case dir)", "-", False))
            continue
        if not (case_dir / "bench_spec.py").exists():
            rows.append((name, "(no bench_spec)", "(no bench_spec)", "-", False))
            continue
        if not gen.exists():
            rows.append((name, "(no generated.py)", "(no generated.py)", "-", False))
            continue
        relpath = gen.resolve().relative_to(repo_root).as_posix()
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{args.rev}:{relpath}"],
            capture_output=True,
        )
        if proc.returncode != 0:
            rows.append((name, f"(not on {args.rev})", _cell(_measure(case_dir, gen)), "-", True))
        elif proc.stdout == gen.read_bytes():
            rows.append((name, _cell(_measure(case_dir, gen)), "unchanged", "-", False))
        else:
            with tempfile.TemporaryDirectory() as td:
                rev_gen = Path(td) / "generated.py"
                rev_gen.write_bytes(proc.stdout)
                left = _measure(case_dir, rev_gen)
            right = _measure(case_dir, gen)
            rows.append((name, _cell(left), _cell(right), _improvement(left, right), True))

    rev_label = (
        "main"
        if args.rev in ("origin/main", "main")
        else ("master" if args.rev in ("origin/master", "master") else args.rev)
    )
    headers = ("case", f"{rev_label} (gen/hw)", "branch (gen/hw)", "improvement")
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    # ANSI bold only when a human is looking (or forced) — logs stay clean.
    use_bold = sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR"))

    def _rule(left: str, mid: str, right: str) -> None:
        print(left + mid.join("─" * (w + 2) for w in widths) + right)

    def _row(cells, bold_cols: frozenset[int] = frozenset()) -> None:
        parts = []
        for i, (c, w) in enumerate(zip(cells, widths)):
            pad = c.ljust(w)  # pad BEFORE wrapping: escapes must not skew width
            parts.append(f"\033[1m{pad}\033[0m" if use_bold and i in bold_cols else pad)
        print("│ " + " │ ".join(parts) + " │")

    print()
    _rule("┌", "┬", "┐")
    _row(headers)
    _rule("├", "┼", "┤")
    for name, left, right, imp, bold_branch in rows:
        _row((name, left, right, imp), frozenset({2}) if bold_branch else frozenset())
    _rule("└", "┴", "┘")
    print(
        "\ncells: per-shape gen/handwritten ratios (raw gen throughput when the case"
        "\nhas no handwritten baseline), SHAPES order; both columns run under the"
        f"\ncurrent build. improvement = branch gen throughput vs {rev_label}, per shape;"
        "\n'unchanged' = byte-identical generated.py (measured once)."
    )
    return 0


_SUBCOMMANDS = {
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
