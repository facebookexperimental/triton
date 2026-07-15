#!/usr/bin/env python3
"""Unified perf harness for torchTLX Inductor templates.

This is the git-repo / OSS short-term perf tool: it needs no tritonbench and no
fbcode, so every external contributor measures torchTLX the same way. It mirrors
the interface of the fbsource tritonbench one-shot command (``--op``, ``--only``,
``--precision``, ``--m/--n/--k``, ``--metrics``, ``--baseline``) so numbers are
comparable across the two paths. The long-term plan is native tritonbench
integration; until then, use this.

Recommended entry point (sets up the env for you)::

    scripts/run_torchtlx_perf.sh --op mm --only aten,torch_tlx_mm --precision bf16 \
        --m 4096 --n 4096 --k 4096 --metrics accuracy,tflops

Providers for ``--op mm``:
  * ``aten``          -- torch.matmul / cuBLAS (default baseline)
  * ``torch_tlx_mm``  -- torch.compile(mm) through the torchTLX Inductor template
                         (``config.triton.tlx_mode``); this is what we ship
  * ``tlx_ws``        -- the standalone TLX warp-specialized kernel (kernel ceiling)

Structured for future ops (e.g. flex attention) and providers: add an entry to
``OPS`` with its input builder, flop counter and provider table.

Facebook: if you are in fbsource, use tritonbench instead to collect perf.
"""

import argparse
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Callable

import torch

import triton

try:
    # Normal case: run as a module (python -m ...benchmarks.torchtlx_perf).
    from . import shapes as shape_catalog
except ImportError:
    # Fallback: run as a plain script (python path/to/torchtlx_perf.py); Python
    # puts the script's dir on sys.path, so a bare import resolves.
    import shapes as shape_catalog  # type: ignore

DEVICE = triton.runtime.driver.active.get_active_torch_device()

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
# Tolerances match the torchTLX unit tests (test_torchtlx_templates.py).
ATOL = {torch.float16: 0.01, torch.bfloat16: 2e-2}
RTOL = {torch.float16: 0.01, torch.bfloat16: 2e-2}


# ---------------------------------------------------------------------------
# Catalog of torchTLX templates (for --list / discoverability).
# ---------------------------------------------------------------------------
TEMPLATE_CATALOG = [
    # (inductor template name, op, arch, status)
    ("tlx_blackwell_gemm_ws", "mm", "Blackwell (B200/GB200)", "shipped"),
    ("tlx_amd_addmm_warppipe", "addmm", "AMD MI350X (gfx950)", "shipped"),
    ("tlx_flex_attention", "flex_attention", "Blackwell", "in development"),
]


@dataclass
class Provider:
    """A benchmarkable implementation of an op.

    ``build`` returns ``(fn, ctxs)`` where ``fn()`` runs the op once and ``ctxs``
    is a list of context managers that must be active while ``fn`` is called
    (compile + run happen under them). Returns ``None`` if unavailable here.
    """

    name: str
    build: Callable
    requires_blackwell: bool = False


# ---------------------------------------------------------------------------
# op = mm
# ---------------------------------------------------------------------------
def _mm_inputs(M, N, K, dtype):
    a = torch.randn((M, K), device=DEVICE, dtype=dtype)
    b = torch.randn((K, N), device=DEVICE, dtype=dtype)
    return a, b


def _mm_flops(M, N, K):
    return 2.0 * M * N * K


def _build_aten(inputs, args):
    a, b = inputs
    return (lambda: torch.matmul(a, b)), []


def _build_tlx_ws(inputs, args):
    a, b = inputs
    try:
        from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import (
            matmul as tlx_matmul,
        )
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError(f"tlx_ws unavailable: {e}") from e
    return (lambda: tlx_matmul(a, b)), []


def _build_torch_tlx_mm(inputs, args):
    a, b = inputs
    from torch._inductor import config as inductor_config
    from triton.language.extra.tlx.inductor import tlx_config

    ctxs = [
        inductor_config.patch(
            {
                "triton.tlx_mode": args.tlx_mode,
                "force_disable_caches": True,
                "enable_caching_generated_triton_templates": False,
            }
        ),
        tlx_config.patch(use_heuristic_config=args.use_heuristic_config),
    ]

    def mm(x, y):
        return torch.mm(x, y)

    # Compiled lazily; the first fn() call (warmup) triggers compilation, which
    # must run while ``ctxs`` are active -- the benchmark loop guarantees that.
    compiled = torch.compile(mm, dynamic=args.dynamic)
    return (lambda: compiled(a, b)), ctxs


MM_PROVIDERS = {
    "aten": Provider("aten", _build_aten),
    "torch_tlx_mm": Provider("torch_tlx_mm", _build_torch_tlx_mm, requires_blackwell=True),
    "tlx_ws": Provider("tlx_ws", _build_tlx_ws, requires_blackwell=True),
}


@dataclass
class Op:
    name: str
    make_inputs: Callable
    flops: Callable
    providers: dict = field(default_factory=dict)
    default_only: list = field(default_factory=list)


OPS = {
    "mm": Op(
        name="mm",
        make_inputs=_mm_inputs,
        flops=_mm_flops,
        providers=MM_PROVIDERS,
        default_only=["aten", "torch_tlx_mm", "tlx_ws"],
    ),
}


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------
def _is_blackwell() -> bool:
    try:
        from triton._internal_testing import is_blackwell

        return bool(is_blackwell())
    except Exception:
        return False


def bench_shape(op, shape, dtype, providers, baseline, metrics, args):
    """Return {provider_name: {tflops, ms, accuracy, speedup}} for one shape."""
    M, N, K = shape
    total_flops = op.flops(M, N, K)
    results = {}
    ref = None  # baseline output for accuracy

    quantiles = None

    # Build inputs ONCE and share across providers so the accuracy comparison is
    # apples-to-apples (all providers see the same data). Providers must not
    # mutate the inputs (matmul/compiled-mm are read-only).
    inputs = op.make_inputs(M, N, K, dtype)

    for name in providers:
        prov = op.providers[name]
        rec = {"tflops": None, "ms": None, "accuracy": None, "speedup": None}
        try:
            fn, ctxs = prov.build(inputs, args)
            with ExitStack() as es:
                for cm in ctxs:
                    es.enter_context(cm)
                out = fn()  # warmup / compile under the provider's contexts
                torch.cuda.synchronize()
                if "accuracy" in metrics:
                    if name == baseline:
                        ref = out.detach().float()
                        rec["accuracy"] = "ref"
                    elif ref is not None:
                        max_err = (out.detach().float() - ref).abs().max().item()
                        try:
                            torch.testing.assert_close(
                                out.float(), ref, atol=ATOL[dtype], rtol=RTOL[dtype]
                            )
                            rec["accuracy"] = f"PASS(max_err={max_err:.2e})"
                        except AssertionError:
                            rec["accuracy"] = f"FAIL(max_err={max_err:.2e})"
                ms = triton.testing.do_bench(
                    fn, warmup=args.warmup, rep=args.rep, quantiles=quantiles
                )
                rec["ms"] = ms
                rec["tflops"] = total_flops * 1e-12 / (ms * 1e-3)
        except Exception as e:
            rec["accuracy"] = f"ERROR({type(e).__name__})"
            if args.verbose:
                import traceback

                traceback.print_exc()
        results[name] = rec

    # speedup vs baseline
    base = results.get(baseline, {})
    base_ms = base.get("ms")
    for name, rec in results.items():
        if base_ms and rec.get("ms"):
            rec["speedup"] = base_ms / rec["ms"]

    return results


def _fmt_cell(rec, key):
    if key == "tflops":
        return f"{rec['tflops']:.1f}" if rec["tflops"] is not None else "n/a"
    if key == "speedup":
        return f"{rec['speedup']:.3f}x" if rec["speedup"] is not None else "n/a"
    if key == "accuracy":
        return rec["accuracy"] if rec["accuracy"] is not None else "-"
    return ""


def print_csv(rows, providers, dtype_name):
    """Machine-readable CSV (one line per (shape, provider))."""
    print("M,N,K,dtype,provider,tflops,speedup,accuracy")
    for shape, results in rows:
        M, N, K = shape
        for name in providers:
            rec = results[name]
            tflops = f"{rec['tflops']:.1f}" if rec["tflops"] is not None else "n/a"
            sp = f"{rec['speedup']:.3f}" if rec["speedup"] is not None else "n/a"
            acc = rec["accuracy"] if rec["accuracy"] is not None else ""
            print(f"{M},{N},{K},{dtype_name},{name},{tflops},{sp},{acc}")


def print_table(rows, providers, metrics, dtype_name, baseline):
    """One row per shape; providers become columns (dynamic widths).

    Each provider contributes a column per requested metric. speedup/accuracy are
    omitted for the baseline provider (trivially 1.000x / ref).
    """
    metric_cols = [m for m in ("tflops", "speedup", "accuracy") if m in metrics]
    short = {"tflops": "TFLOPS", "speedup": "spdup", "accuracy": "acc"}

    # Column spec: (provider, metric) pairs, grouped by provider.
    col_specs = []
    for p in providers:
        for m in metric_cols:
            if m in ("speedup", "accuracy") and p == baseline:
                continue
            col_specs.append((p, m))

    headers = ["M", "N", "K"] + [f"{p}:{short[m]}" for (p, m) in col_specs]
    # M/N/K and numeric metrics right-aligned; accuracy string left-aligned.
    aligns = ["r", "r", "r"] + ["l" if m == "accuracy" else "r" for (_, m) in col_specs]

    table = []
    for shape, results in rows:
        M, N, K = shape
        cells = [str(M), str(N), str(K)]
        cells += [_fmt_cell(results[p], m) for (p, m) in col_specs]
        table.append(cells)

    ncol = len(headers)
    widths = [len(headers[i]) for i in range(ncol)]
    for cells in table:
        for i in range(ncol):
            widths[i] = max(widths[i], len(cells[i]))

    def render(cells):
        out = [
            c.rjust(widths[i]) if aligns[i] == "r" else c.ljust(widths[i])
            for i, c in enumerate(cells)
        ]
        return "  ".join(out).rstrip()

    print(f"\n# dtype={dtype_name} baseline={baseline}")
    print(render(headers))
    print("-" * (sum(widths) + 2 * (ncol - 1)))
    for cells in table:
        print(render(cells))


def resolve_shapes(args):
    if args.m is not None or args.n is not None or args.k is not None:
        if None in (args.m, args.n, args.k):
            sys.exit("single-shape mode needs all of --m --n --k")
        return [(args.m, args.n, args.k)]
    if args.shapes:
        return shape_catalog.parse_shapes(args.shapes)
    sys.exit("provide a shape: --m/--n/--k (single) or --shapes MxNxK,... (batch)")


def main(argv=None):
    p = argparse.ArgumentParser(description="torchTLX perf harness (git-repo)")
    p.add_argument("--op", default="mm", choices=list(OPS.keys()))
    p.add_argument(
        "--only",
        default=None,
        help="comma-separated provider names (default: all for the op)",
    )
    p.add_argument("--baseline", default="aten", help="provider used for speedup + accuracy ref")
    p.add_argument("--precision", "--dtype", dest="precision", default="bf16", choices=list(DTYPES.keys()))
    p.add_argument("--metrics", default="tflops,speedup,accuracy")
    # single shape
    p.add_argument("--m", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    # batch shape
    p.add_argument("--shapes", default=None, help="ad-hoc shapes: MxNxK,MxNxK,...")
    # torch_tlx knobs
    p.add_argument("--tlx-mode", dest="tlx_mode", default="force", choices=["allow", "force"])
    p.add_argument(
        "--autotune",
        dest="use_heuristic_config",
        action="store_false",
        help="autotune TLX configs instead of the single heuristic config (slower)",
    )
    p.set_defaults(use_heuristic_config=True)
    p.add_argument("--dynamic", action="store_true", default=True, help="torch.compile dynamic shapes (default)")
    p.add_argument("--static", dest="dynamic", action="store_false", help="torch.compile static shapes")
    # do_bench
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    # output
    p.add_argument("--simple-output", action="store_true", help="one CSV-ish line per (shape,provider)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--list", action="store_true", help="list torchTLX templates and exit")
    args = p.parse_args(argv)

    if args.list:
        print("torchTLX templates:")
        print(f"  {'name':24} {'op':16} {'arch':26} status")
        for name, op, arch, status in TEMPLATE_CATALOG:
            print(f"  {name:24} {op:16} {arch:26} {status}")
        return 0

    op = OPS[args.op]
    only = args.only.split(",") if args.only else op.default_only
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    dtype = DTYPES[args.precision]

    blackwell = _is_blackwell()
    providers = []
    for name in only:
        if name not in op.providers:
            sys.exit(f"unknown provider {name!r} for op {args.op}; have {list(op.providers)}")
        if op.providers[name].requires_blackwell and not blackwell:
            print(f"[skip] {name}: requires Blackwell GPU", file=sys.stderr)
            continue
        providers.append(name)
    if not providers:
        sys.exit("no runnable providers")
    # accuracy/speedup need the baseline benchmarked first
    if args.baseline in providers:
        providers = [args.baseline] + [x for x in providers if x != args.baseline]

    shapes = resolve_shapes(args)
    print(
        f"# op={args.op} providers={providers} baseline={args.baseline} "
        f"dtype={args.precision} tlx_mode={args.tlx_mode} "
        f"heuristic_config={args.use_heuristic_config} dynamic={args.dynamic} "
        f"warmup={args.warmup} rep={args.rep}",
        file=sys.stderr,
    )
    rows = []
    for i, shape in enumerate(shapes):
        print(f"# [{i + 1}/{len(shapes)}] benchmarking M={shape[0]} N={shape[1]} K={shape[2]}", file=sys.stderr)
        results = bench_shape(op, shape, dtype, providers, args.baseline, metrics, args)
        rows.append((shape, results))

    if args.simple_output:
        print_csv(rows, providers, args.precision)
    else:
        print_table(rows, providers, metrics, args.precision, args.baseline)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
