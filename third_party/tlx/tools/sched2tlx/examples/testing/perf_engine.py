"""Generic performance engine for sched2tlx example kernels.

This factors out the machinery duplicated across each case's
``perf_generated_vs_handwritten.py`` (CUDA-event timing, shape sweep, a
correctness guard, and throughput/ratio reporting). The per-case specifics live
in a tiny ``bench_spec.py`` next to each example (see ``README.md``), so a new
case is picked up automatically once it ships one.

Two entry points:

* ``worker`` subcommand — benchmark the generated kernel for ONE case using a
  caller-supplied ``generated.py`` path, and write the measurements as JSON.
  The regression driver invokes this twice per case (before/after emitter) in
  separate processes so a kernel that faults at scale (the exact failure the
  before-emitter can produce) cannot poison the CUDA context of the other run.
* ``bench`` subcommand (default) — benchmark a case's committed ``generated.py``
  against its handwritten reference and print a human-readable table, i.e. the
  classic ``perf_generated_vs_handwritten.py`` behavior, but generic.

Requires torch + triton + a GPU (only when actually benchmarking).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    w = sub.add_parser("worker", help="benchmark one case's generated.py, write JSON")
    w.add_argument("--case-dir", required=True)
    w.add_argument("--generated", required=True, help="path to the generated.py to benchmark")
    w.add_argument("--out", required=True, help="path to write the JSON result")
    w.add_argument("--no-hw", action="store_true", help="skip the handwritten baseline")

    b = sub.add_parser("bench", help="benchmark cases' committed generated.py vs handwritten")
    b.add_argument("--cases", help="comma-separated case dir names (default: all with bench_spec.py)")

    args = ap.parse_args(argv)

    if args.cmd == "worker":
        case_dir = Path(args.case_dir).resolve()
        result = run_bench(case_dir, Path(args.generated).resolve(), want_hw=not args.no_hw)
        Path(args.out).write_text(json.dumps(result))
        return 0

    # default / "bench": human table over committed generated.py
    examples_dir = Path(__file__).resolve().parent.parent
    if args.cmd == "bench" and args.cases:
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


if __name__ == "__main__":
    sys.exit(main())
