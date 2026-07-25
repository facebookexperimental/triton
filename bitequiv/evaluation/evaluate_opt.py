"""evaluate_opt — M2 optimization-pass evaluation framework (support / correctness / performance).

WHAT THIS IS
------------
A **generic, pass-agnostic** ruler for a Triton *compiler optimization pass*
(M2's target is a reduction-layout pass, but this script knows nothing about it).
You name the pass(es) under test in ``triton-opt`` / ``mlir-opt`` vocabulary via
``--opt-passes``; the framework compiles each in-scope reduction config **with**
vs **without** those passes and answers three independent questions:

  Stage 1 — SUPPORT.      Does the kernel still COMPILE when the pass(es) are
      applied? (Is it accepted by the compilation passes?) Compile-only, no
      kernel launches — this is the DEFAULT stage.
  Stage 2 — CORRECTNESS.  Does the pass CHANGE the result bits vs the baseline
      compile, empirically (run both on GPU on the same seeds, compare bytes)?
      An optimization like M2 MUST be bit-identical, so the gate is 0 changes.
      OPT-IN (executes kernels on GPU).
  Stage 3 — PERFORMANCE.  Does the pass make the kernel FASTER? Benchmarks
      baseline vs optimized and reports the speedup. OPT-IN (executes kernels
      on GPU).

  Only Stage 1 runs by default; Stages 2 and 3 each run kernels on the GPU and
  are opt-in via ``--stages`` (e.g. ``--stages 1,2`` or ``--stages 1,2,3``).

DECOUPLED FROM ANY PASS
-----------------------
This script imports no pass by name. A pass named on the CLI that has no binding
in the current build is reported as *unavailable* and treated as baseline, so the
report shape is identical whether or not the pass exists ("return the same info
if some passes do not exist"). M2 and future milestones simply CALL this script
with their pass name; the framework does not depend on them.

MECHANISM
---------
In-process pipeline injection: a context manager patches the NVIDIA backend's
``CUDABackend.make_ttgir`` to append the requested passes at end-of-TTGIR, so the
kernel file (``eval_kernels.py``) and its compile/run/benchmark hooks are reused
UNCHANGED and the A/B launch path is identical — only the compile pipeline
differs. ``TRITON_ALWAYS_COMPILE`` + JIT-cache clearing force a fresh compile for
each variant so the patch actually takes effect.

SCOPE
-----
By default only ``reduction_ordering == "inner_tree"`` configs are evaluated (the
reductions M2 transforms). Kernels with no such config (e.g. welford) report
"out of M2 scope". Widen with ``--ordering``.

EXAMPLES
--------
  # harness self-check (no pass): opt == baseline, must report 0 bit-changes
  python -m bitequiv.evaluation.evaluate_opt --stages 1,2 --config-effort light
  # evaluate a real pass once it exists (M2 will do exactly this):
  python -m bitequiv.evaluation.evaluate_opt --opt-passes tritongpu-optimize-reduction-layout --stages 1,2,3
  # exercise the injection with a benign existing pass:
  python -m bitequiv.evaluation.evaluate_opt --opt-passes cse --stages 1,2
  # a pass whose binding needs an option (e.g. remove-layout-conversions' SMEM budget)
  # takes it in braces (mlir-opt-ish); without it the pass is reported needs-args, not FAIL:
  python -m bitequiv.evaluation.evaluate_opt --opt-passes "tritongpu-remove-layout-conversions{0}" --stages 1,2
  # measure an EXISTING pipeline pass by REMOVING it from the baseline (baseline = pipeline
  # WITHOUT it, optimized = pipeline WITH it, so speedup = how much the pass helps):
  python -m bitequiv.evaluation.evaluate_opt --disable-passes tritongpu-coalesce --stages 2,3

INJECT vs DISABLE
-----------------
Two ways to name the pass under test, both A/B "pipeline without P" vs "pipeline with P":
  * ``--opt-passes P``     — for a pass NOT in the standard pipeline (e.g. the future M2
    pass): baseline = pipeline, optimized = pipeline + P (appended at end-of-TTGIR).
  * ``--disable-passes P`` — for a pass ALREADY in the standard pipeline (e.g. coalesce):
    baseline = pipeline with P removed, optimized = the normal pipeline.
"""

from __future__ import annotations

import os

# Force a fresh compile for every warmup so the make_ttgir patch is not masked by
# Triton's in-memory / on-disk compile cache (the patch is not part of the cache
# key). Set before triton is imported so an import-time env read still sees it.
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import datetime  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

from bitequiv.evaluation import eval_kernels  # noqa: E402
from bitequiv.evaluation.eval_kernels import config_label, resolve_kernels  # noqa: E402

_DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "result_opt.txt")


# --------------------------------------------------------------------------- #
# Pass injection (the A/B toggle) — decoupled, name-driven, graceful if absent.
# --------------------------------------------------------------------------- #
def _pass_submodules():
    """The libtriton pass namespaces we resolve ``add_*`` bindings from."""
    from triton._C.libtriton import nvidia, passes

    subs = []
    for name in ("ttgpuir", "ttir", "common", "convert"):
        sub = getattr(passes, name, None)
        if sub is not None:
            subs.append(sub)
    npasses = getattr(nvidia, "passes", None)
    if npasses is not None:
        for name in ("ttgpuir", "ttnvgpuir", "hopper"):
            sub = getattr(npasses, name, None)
            if sub is not None:
                subs.append(sub)
    return subs


def _resolve_pass_site(name):
    """``(submodule, attr)`` of the ``add_*`` binding for a triton-opt/mlir-opt name, or None.

    Names are normalized (``-`` -> ``_``) and tried as ``add_<name>`` across the pass
    namespaces; a leading dialect token (e.g. ``tritongpu-``) is also tried stripped.
    Returning the *site* (not just the callable) lets us both call it (inject) and
    no-op it (disable). None if no binding exists in this build.
    """
    norm = name.strip().replace("-", "_")
    candidates = []
    if norm.startswith("add_"):
        candidates.append(norm)
    candidates.append(f"add_{norm}")
    if "_" in norm:  # drop a leading dialect token, e.g. tritongpu_foo -> foo
        candidates.append(f"add_{norm.split('_', 1)[1]}")
    for sub in _pass_submodules():
        for attr in candidates:
            if getattr(sub, attr, None) is not None:
                return sub, attr
    return None


def _resolve_pass(name):
    """The ``add_*`` callable for a pass name, or None if no binding exists."""
    site = _resolve_pass_site(name)
    return getattr(*site) if site is not None else None


_PROBE_CTX = None


def _probe_context():
    """A cached MLIRContext used only to probe pass-binding signatures."""
    global _PROBE_CTX
    if _PROBE_CTX is None:
        from triton._C.libtriton import ir
        _PROBE_CTX = ir.context()
    return _PROBE_CTX


def _classify(pass_specs):
    """Resolve + probe each ``(name, args)``. Returns ``(resolved, needs_args, unavailable)``:

      resolved    -- [(name, add_fn, args)]  binding exists and accepts these args (WILL run)
      needs_args  -- [(name, reason)]        binding exists but rejects these args (e.g. a
                                             required option not supplied) -> treated as baseline,
                                             NOT a compile regression
      unavailable -- [name]                  no binding in this build -> treated as baseline

    The probe adds the pass to a throwaway pass manager (no compile, no run); pybind
    raises ``TypeError`` if the args don't match the binding's signature.
    """
    from triton._C.libtriton import ir

    resolved, needs_args, unavailable = [], [], []
    for name, args in pass_specs:
        fn = _resolve_pass(name)
        if fn is None:
            unavailable.append(name)
            continue
        try:
            fn(ir.pass_manager(_probe_context()), *args)
        except TypeError as exc:  # signature mismatch (e.g. binding needs an option we didn't pass)
            needs_args.append((name, str(exc).splitlines()[0]))
            continue
        resolved.append((name, fn, args))
    return resolved, needs_args, unavailable


def _apply_passes(mod, resolved):
    """Run the resolved passes on a module via a fresh pass manager (mirrors make_ttgir).

    ``resolved`` is a list of ``(name, add_fn, args)``; each is applied as
    ``add_fn(pm, *args)`` — ``args`` come from ``name{...}`` on the CLI.
    """
    from triton._C.libtriton import ir

    pm = ir.pass_manager(mod.context)
    for _name, fn, args in resolved:
        fn(pm, *args)
    try:
        pm.run(mod, "evaluate_opt")
    except TypeError:
        pm.run(mod)
    return mod


@contextlib.contextmanager
def optimization(resolved):
    """Apply the ``resolved`` passes (``[(name, add_fn, args)]``) at end-of-TTGIR for
    the context's lifetime by patching ``CUDABackend.make_ttgir``. An empty list is a
    no-op (baseline). Restores ``make_ttgir`` on exit."""
    if not resolved:
        yield
        return

    from triton.backends.nvidia.compiler import CUDABackend

    original = CUDABackend.make_ttgir  # staticmethod access -> underlying function

    def patched(mod, metadata, opt, capability):
        mod = original(mod, metadata, opt, capability)
        return _apply_passes(mod, resolved)

    CUDABackend.make_ttgir = staticmethod(patched)
    try:
        yield
    finally:
        CUDABackend.make_ttgir = staticmethod(original)


def _clear_jit_caches():
    """Drop the eval kernels' in-memory compile caches so the next warmup recompiles."""
    from triton.runtime.jit import JITFunction

    for obj in vars(eval_kernels).values():
        if isinstance(obj, JITFunction):
            for attr in ("device_caches", "cache"):
                cache = getattr(obj, attr, None)
                if hasattr(cache, "clear"):
                    try:
                        cache.clear()
                    except Exception:  # noqa: BLE001
                        pass


@contextlib.contextmanager
def disabled(names):
    """Skip the named passes from ``make_ttgir``'s pipeline for the context's lifetime by
    no-op'ing their ``add_*`` bindings (so ``make_ttgir``'s call adds nothing). Empty =>
    no-op. Restores the bindings on exit. Names with no binding are ignored."""
    if not names:
        yield
        return
    saved = []
    for name in names:
        site = _resolve_pass_site(name)
        if site is None:
            continue
        sub, attr = site
        saved.append((sub, attr, getattr(sub, attr)))
        setattr(sub, attr, lambda *a, **k: None)
    try:
        yield
    finally:
        for sub, attr, orig in saved:
            setattr(sub, attr, orig)


@contextlib.contextmanager
def _variant(resolved, disable_names):
    """One compile variant of the pipeline: skip ``disable_names`` from the standard
    pipeline AND append ``resolved`` passes at end-of-TTGIR."""
    with disabled(disable_names):
        with optimization(resolved):
            yield


def _compile_variant(spec, config, size, variant):
    """Compile one config under a variant — a 0-arg factory returning the pipeline-patch
    context manager (so each compile gets a fresh one). Fresh + cache-busted."""
    _clear_jit_caches()
    with variant():
        return spec.compile(config, size)


# --------------------------------------------------------------------------- #
# Config scope
# --------------------------------------------------------------------------- #
def _in_scope(configs, ordering):
    """Keep only configs matching the ordering scope ('inner_tree' by default)."""
    if ordering == "all":
        return list(configs)
    return [c for c in configs if c.reduction_ordering == ordering]


# --------------------------------------------------------------------------- #
# Stage 1 — support
# --------------------------------------------------------------------------- #
def support(spec, opt_variant, ordering):
    """(verdict, detail): do ALL in-scope (light) configs compile under the optimized variant?"""
    configs = _in_scope(spec.config_space("light"), ordering)
    if not configs:
        return "SKIPPED", f"out of M2 scope (no {ordering} configs)"
    ok, first_error = 0, None
    for config in configs:
        try:
            _compile_variant(spec, config, spec.precision_size, opt_variant)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
    detail = f"{ok}/{len(configs)} compiled (optimized variant)"
    if ok == len(configs):
        return "SUPPORTED", detail
    verdict = "UNSUPPORTED" if ok == 0 else "PARTIAL"
    return verdict, f"{detail}; first error: {first_error}"


# --------------------------------------------------------------------------- #
# Stage 2 — correctness (empirical A/B bit compare)
# --------------------------------------------------------------------------- #
def correctness(spec, base_variant, opt_variant, config_effort, fuzzer_effort, ordering):
    """Compile baseline vs optimized per config, run on N seeds, flag any bit change."""
    from bitequiv.evaluation.equivalence_fuzzer import effort_repeats

    size = spec.precision_size
    configs = _in_scope(spec.config_space(config_effort), ordering)
    if not configs:
        return dict(name=spec.name, scope_empty=True, ordering=ordering)
    repeats = effort_repeats(fuzzer_effort)
    checked, base_fail, opt_fail, changed = 0, 0, 0, []
    for config in configs:
        try:
            ck_base = _compile_variant(spec, config, size, base_variant)
        except Exception:  # noqa: BLE001
            base_fail += 1
            continue
        try:
            ck_opt = _compile_variant(spec, config, size, opt_variant)
        except Exception:  # noqa: BLE001
            opt_fail += 1
            continue
        checked += 1
        for seed in range(repeats):
            if spec.run(config, ck_base, seed, size) != spec.run(config, ck_opt, seed, size):
                changed.append((config, seed))
                break
    return dict(name=spec.name, n_configs=len(configs), checked=checked, base_fail=base_fail, opt_fail=opt_fail,
                changed=changed, repeats=repeats)


# --------------------------------------------------------------------------- #
# Stage 3 — performance (opt-in)
# --------------------------------------------------------------------------- #
def performance(spec, base_variant, opt_variant, config_effort, ordering):
    """Benchmark baseline vs optimized per in-scope config; report speedups."""
    if not spec.supports_perf:
        return dict(name=spec.name, no_perf=True)
    size = spec.perf_size
    configs = _in_scope(spec.config_space(config_effort), ordering)
    if not configs:
        return dict(name=spec.name, scope_empty=True, ordering=ordering)
    rows, fails = [], 0
    for config in configs:
        try:
            _clear_jit_caches()
            with base_variant():
                base_ms, base_bytes, _ = spec.benchmark(config, size)
            _clear_jit_caches()
            with opt_variant():
                opt_ms, opt_bytes, _ = spec.benchmark(config, size)
        except Exception:  # noqa: BLE001
            fails += 1
            continue
        rows.append(dict(config=config, base_ms=base_ms, opt_ms=opt_ms, bit_identical=base_bytes == opt_bytes))
    return dict(name=spec.name, size=size, rows=rows, fails=fails)


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def _table(headers, rows):
    grid = [headers] + [[str(x) for x in r] for r in rows]
    widths = [max(len(row[i]) for row in grid) for i in range(len(headers))]
    fmt = lambda vals: "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    return [fmt(headers), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows]


def render_support(results):
    lines = ["STAGE 1 — support (compiles with the pass applied?)", "-" * 50, ""]
    lines += _table(["kernel", "verdict", "detail"], [[r["name"], r["verdict"], r["detail"]] for r in results])
    return lines + [""]


def render_correctness(results):
    lines = ["STAGE 2 — correctness (optimized bits == baseline bits?)", "-" * 55, ""]
    rows = []
    for r in results:
        if r.get("scope_empty"):
            rows.append([r["name"], f"(no {r['ordering']} configs)", "-", "-", "-"])
            continue
        regressions = r["opt_fail"]
        if r["changed"]:
            verdict = "BITS CHANGED"
        elif regressions:
            verdict = "COMPILE REGRESSION"
        else:
            verdict = "OK"
        rows.append([
            r["name"],
            f"{r['checked']}/{r['n_configs']}",
            str(len(r["changed"])),
            str(regressions),
            verdict,
        ])
    lines += _table(["kernel", "configs checked", "bit-changed", "compile-regressions", "verdict"], rows)
    lines.append("")
    lines.append("bit-changed = optimized output differed from baseline (MUST be 0).")
    lines.append("compile-regressions = baseline compiled but optimized failed to compile (MUST be 0).")
    for r in results:
        for config, seed in r.get("changed", []):
            lines.append(f"  CHANGED {r['name']}: seed {seed}  {config_label(config)}")
    return lines + [""]


def render_performance(results):
    lines = ["STAGE 3 — performance (baseline vs optimized)", "-" * 45, ""]
    for r in results:
        if r.get("no_perf"):
            lines.append(f"{r['name']}: no perf hook; skipped.")
            lines.append("")
            continue
        if r.get("scope_empty") or not r.get("rows"):
            lines.append(f"{r['name']}: no in-scope perf configs benchmarked.")
            lines.append("")
            continue
        rows, cols = r["size"]
        lines.append(f"{r['name']}  (size {rows}x{cols}, do_bench min-of-medians)")
        speedups = []
        for row in r["rows"]:
            valid = math.isfinite(row["opt_ms"]) and row["opt_ms"] > 0
            if valid:
                speedup = row["base_ms"] / row["opt_ms"]
                speedups.append(speedup)
                tail = f"= {speedup:.2f}x  bit_identical={row['bit_identical']}"
            else:
                tail = "opt timing invalid (excluded from summary)"
            lines.append(f"  {config_label(row['config'])}: base {row['base_ms']:.3f} ms -> opt "
                         f"{row['opt_ms']:.3f} ms  {tail}")
        if speedups:
            lines.append(f"  -> speedup min {min(speedups):.2f}x | median "
                         f"{statistics.median(speedups):.2f}x | max {max(speedups):.2f}x")
        else:
            lines.append("  -> no valid timings")
        lines.append("")
    return lines


def write_report(path, header, sections):
    lines = list(header)
    for section in sections:
        lines += section
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv):
    p = argparse.ArgumentParser(description="evaluate_opt — M2 optimization-pass evaluator (see module docstring).")
    p.add_argument(
        "--opt-passes", default="",
        help="triton-opt/mlir-opt pass name(s), space/comma separated; the optimization under test. "
        "A pass whose binding needs args takes them in braces, e.g. 'remove-layout-conversions{0}' or "
        "'pipeline{2,true}'. Empty => opt == baseline (harness self-check).")
    p.add_argument(
        "--disable-passes", default="",
        help="pass name(s) to REMOVE from the standard pipeline (space/comma separated) — the way to "
        "measure an EXISTING pipeline pass: baseline = pipeline WITHOUT it, optimized = pipeline WITH it "
        "(so speedup = how much that pass helps). e.g. --disable-passes tritongpu-coalesce")
    p.add_argument("--kernels", default="all", help="'all' or comma list: " + ",".join(eval_kernels.REGISTRY))
    p.add_argument(
        "--stages", default="1", help="comma list of stages to run. Default '1' (SUPPORT, compile-only). "
        "Stage 2 (CORRECTNESS) and Stage 3 (PERFORMANCE) both execute kernels on GPU and are opt-in, "
        "e.g. --stages 1,2 or --stages 1,2,3")
    p.add_argument("--config-effort", default="light", choices=("light", "heavy"))
    p.add_argument("--fuzzer-effort", default="fast", choices=("fast", "convincing"))
    p.add_argument("--ordering", default="inner_tree", choices=("inner_tree", "unordered", "all"),
                   help="which reduction ordering to evaluate (default inner_tree = M2's target)")
    p.add_argument("--out", default=_DEFAULT_OUT, help="result table path")
    return p.parse_args(argv)


def _coerce(val):
    """Coerce a CLI arg token to bool/int/float/str. Any ``key=`` prefix is dropped and
    the value used positionally (the libtriton pass bindings take positional args)."""
    val = val.strip()
    if "=" in val:
        val = val.split("=", 1)[1].strip()
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _pass_label(name, args):
    return name + ("{" + ",".join(map(str, args)) + "}" if args else "")


def _parse_passes(spec):
    """Parse ``--opt-passes`` into ``[(name, args)]``.

    Passes are separated by whitespace or top-level commas. A pass may carry the
    binding's positional args in ``mlir-opt``-ish braces, e.g.
    ``remove-layout-conversions{0}`` or ``remove-layout-conversions{budget=0}`` ->
    ``(name, [0])``; ``pipeline{2,true}`` -> ``(name, [2, True])``.
    """
    toks, buf, depth = [], [], 0
    for ch in spec:
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif depth == 0 and (ch.isspace() or ch == ","):
            if buf:
                toks.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        toks.append("".join(buf))

    out = []
    for tok in toks:
        if tok.endswith("}") and "{" in tok:
            name = tok[:tok.index("{")]
            args = [_coerce(a) for a in tok[tok.index("{") + 1:-1].split(",") if a.strip()]
        else:
            name, args = tok, []
        name = name.strip()
        if name:
            out.append((name, args))
    return out


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("no CUDA GPU available; evaluate_opt needs one to compile and run kernels. Skipping.")
        return 0

    stages = {int(s) for s in args.stages.split(",") if s.strip()}
    specs = resolve_kernels(args.kernels)
    resolved, needs_args, unavailable = _classify(_parse_passes(args.opt_passes))
    disable_names, disable_missing = [], []
    for _name, _a in _parse_passes(args.disable_passes):
        (disable_names if _resolve_pass_site(_name) is not None else disable_missing).append(_name)

    def base_variant():
        return _variant([], disable_names)

    def opt_variant():
        return _variant(resolved, [])

    device = torch.cuda.get_device_name()

    header = [
        "evaluate_opt result (M2 optimization-pass evaluation)",
        "=====================================================",
        f"generated:      {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"opt-passes:     {[_pass_label(n, a) for n, _, a in resolved] or '(none)'}",
        f"disable-passes: {disable_names or '(none)'}",
        "A/B:            baseline = pipeline minus disable-passes; optimized = pipeline plus opt-passes",
    ]
    if needs_args:
        header.append(f"needs-args:     {[n for n, _ in needs_args]}  "
                      "(binding needs args -> NOT applied, treated as baseline; supply e.g. name{val})")
    if unavailable:
        header.append(f"unavailable:    {unavailable}  (no binding in this build -> treated as baseline)")
    if disable_missing:
        header.append(f"disable-missing:{disable_missing}  (no binding in this build -> cannot disable)")
    header += [
        f"device:         {device}",
        f"ordering scope: {args.ordering}",
        f"config effort:  {args.config_effort}",
        f"kernels:        {', '.join(s.name for s in specs)}",
        f"stages:         {','.join(str(s) for s in sorted(stages))}",
        "",
    ]
    print("\n".join(header))

    sections = []
    total_changed = 0
    total_checked = 0
    total_regressions = 0

    if 1 in stages:
        print("== Stage 1: support ==", flush=True)
        results = []
        for spec in specs:
            verdict, detail = support(spec, opt_variant, args.ordering)
            results.append(dict(name=spec.name, verdict=verdict, detail=detail))
            print(f"  {spec.name}: {verdict} — {detail}", flush=True)
        sections.append(render_support(results))

    if 2 in stages:
        print("== Stage 2: correctness ==", flush=True)
        results = []
        for spec in specs:
            r = correctness(spec, base_variant, opt_variant, args.config_effort, args.fuzzer_effort, args.ordering)
            results.append(r)
            if r.get("scope_empty"):
                print(f"  {spec.name}: out of {args.ordering} scope", flush=True)
            else:
                total_changed += len(r["changed"])
                total_checked += r["checked"]
                total_regressions += r["opt_fail"]
                print(
                    f"    -> {r['checked']}/{r['n_configs']} checked | bit-changed {len(r['changed'])} "
                    f"| compile-regressions {r['opt_fail']} | base_fail {r['base_fail']}", flush=True)
        sections.append(render_correctness(results))

    if 3 in stages:
        print("== Stage 3: performance ==", flush=True)
        results = []
        for spec in specs:
            results.append(performance(spec, base_variant, opt_variant, args.config_effort, args.ordering))
        sections.append(render_performance(results))

    write_report(args.out, header, sections)
    print(f"\nWROTE {args.out}")
    if 2 in stages:
        print(f"CHECKED: total configs checked = {total_checked}")
        print(f"CORRECTNESS: total bit-changed configs (must be 0) = {total_changed}")
        print(f"REGRESSIONS: total compile regressions (must be 0) = {total_regressions}")
        if total_changed or total_regressions:
            reasons = []
            if total_changed:
                reasons.append(f"{total_changed} config(s) changed bits")
            if total_regressions:
                reasons.append(f"{total_regressions} config(s) failed to compile with the pass")
            print(f"RESULT: FAIL — {'; '.join(reasons)}.")
            return 1
        print("RESULT: PASS — the optimization preserved result bits and compiled everywhere (or no pass applied).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
