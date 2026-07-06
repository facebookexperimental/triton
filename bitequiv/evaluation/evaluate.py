"""bitequiv evaluation framework — the team's standard ruler for a PTX equivalence checker.

WHAT THIS IS
------------
One driver, four stages, run from the command line, that measure how good a
static bitwise-equivalence checker is at deciding which autotuner configs of a
reduction kernel produce identical output bits. The checker under test is
pluggable (``--checker``); by default it is the repo checker
``bitequiv.ptx_reduction.ptx_reduction_descriptor``. This script changes no
checker code — it only evaluates one.

WHY SEPARATE STAGES
-------------------
The question "is this checker good?" splits into independent questions, so
the framework answers them separately and you can run only the ones you need:

  Stage 1 — SUPPORT.   Does the checker even understand this kind of kernel?
      For each kernel it probes the checker on a reference config and reports
      SUPPORTED / LIMITED / UNSUPPORTED. Today this is a thin heuristic on the
      descriptor (empty / unparseable => UNSUPPORTED, a tensor-core ``mma`` guard
      => LIMITED, otherwise SUPPORTED) plus any limitation the kernel author
      declared (the ``cond_reduce`` control-flow kernel is the example). The repo
      checker accepts essentially every reduction, so this stage is a placeholder;
      ``kernel_support`` is the documented extension point for when a checker can
      declare real support itself.

  Stage 2 — PRECISION.   Of the configs it supports, how well does the checker
      partition them, and is it SOUND? It builds the config space, compiles each
      config to PTX, asks the checker to group them, and independently fuzzes
      every config (the empirical ground truth). It then reports, per kernel:
        * how many equivalence classes the checker found and the largest one
          (the bit-equivalent search space the checker recovers);
        * how many classes the fuzzer found and the largest one (the recovery
          CEILING — the most the checker could ever recover);
        * OVER-MERGES — pairs the checker called equal but the fuzzer separated.
          This is the soundness violation count and MUST be 0;
        * REFINES — whether the checker partition refines the empirical one (the
          formal statement of soundness).

  Stage 3 — PERFORMANCE (opt-in).   Given a bit-exactness constraint, how much
      speed is on the table? It benchmarks every kernel across its config space,
      finds the global-fastest CEILING (a normal autotuner, no equivalence
      constraint), then looks inside one checker-certified equivalence set:
      after verifying every member is byte-identical, it reports the fastest vs
      slowest member (the tuning freedom the checker hands the autotuner) and the
      best member vs the ceiling (the cost of demanding identical bits).

  Stage 4 — EQUIVALENCE UNDER REGISTER PRESSURE (opt-in).   Does the checker's
      verdict survive ptxas (PTX -> SASS) across a WHOLE diverse equivalence set? It
      fuzzes each checker-equivalence set (diverse num_warps / num_stages /
      enable_fp_fusion / block_n) while ALSO capping ``maxnreg`` low enough to make
      ptxas spill. ``maxnreg`` only adds a ``.maxnreg`` directive (same PTX body =>
      same checker classes) but changes ptxas allocation, so the set spans diverse
      configs AND register regimes. One checker class == one bit-class (over-merges 0)
      means equivalence holds across diverse configs even under spilling; a bit-split
      would flag any ptxas-induced break. Run at heavy config effort + the large tensor.

EFFORT KNOBS
------------
  --config-effort  light (~10 configs, quick gate)  |  heavy (full sweep, ~1000-3000)
  --fuzzer-effort  fast (10 random seeds)            |  convincing (1000 seeds)
A fuzzer can only refute equivalence, never prove it, so more seeds = stronger
evidence of soundness, never certainty.

OUTPUT
------
A plain, human-readable table file (``--out``, default
``bitequiv/evaluation/result.txt``; gitignored — regenerate on demand) plus a
short summary on stdout whose soundness line the pytest gate parses.

LAYOUT
------
  eval_kernels.py       — every test kernel + its KernelSpec (the registry).
  equivalence_fuzzer.py — standalone empirical oracle + partition/soundness math.
  evaluate.py (here)    — the staged CLI driver; orchestration only.

EXAMPLES
--------
  # quick smoke / what the pytest gate runs:
  python -m bitequiv.evaluation.evaluate --stages 1,2 --config-effort light --fuzzer-effort fast
  # full standard run with perf:
  python -m bitequiv.evaluation.evaluate --stages 1,2,3 --config-effort heavy --fuzzer-effort convincing
  # evaluate a different / experimental checker:
  python -m bitequiv.evaluation.evaluate --checker my.module:my_descriptor
"""

import argparse
import datetime
import importlib
import os
import sys

# Make ``bitequiv`` importable whether launched as ``-m bitequiv.evaluation.evaluate``
# or as a plain script path.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

from bitequiv.evaluation import eval_kernels, equivalence_fuzzer  # noqa: E402
from bitequiv.evaluation.eval_kernels import _AXIS_ORDER, config_label, resolve_kernels  # noqa: E402

_DEFAULT_CHECKER = "bitequiv.ptx_reduction:ptx_reduction_descriptor"
_DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "result.txt")


# --------------------------------------------------------------------------- #
# Checker plumbing
# --------------------------------------------------------------------------- #
def load_checker(spec):
    """Load a ``module:function`` checker into ``descriptor(asm_text) -> hashable``.

    The driver feeds the descriptor the IR text of the artifact selected by
    ``--artifact`` (``ptx`` for the PTX checker, ``ttgir`` for the TTGIR checker)."""
    if ":" not in spec:
        raise ValueError(f"--checker must be 'module:function', got {spec!r}")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _descriptor_verdict(desc):
    """Heuristic SUPPORTED/LIMITED/UNSUPPORTED from a checker descriptor alone.

    The repo descriptor is a compact *hash* per entry (e.g. ``ntid=x32|<hash>``),
    so we cannot read the tree out of it — the only checker-agnostic signals are:
    a non-empty parseable descriptor (the checker accepted the kernel), the
    plaintext ``unanalyzed-mma`` guard (a tensor-core op it does not model), or an
    empty / unparseable result. This is why Stage 1 is a placeholder: today's
    checker accepts essentially every reduction, so a real LIMITED/UNSUPPORTED
    signal must come from a declared limitation or a checker that self-declares
    support (the ``kernel_support`` extension point).
    """
    if not desc:
        return "UNSUPPORTED", "empty descriptor (no entry / no reduction reconstructed)"
    if not isinstance(desc, (tuple, list)):
        return "UNSUPPORTED", "checker could not parse the PTX"
    if any("unanalyzed-mma" in str(s) for s in desc):
        return "LIMITED", "contains a tensor-core (mma) op the checker does not model"
    return "SUPPORTED", "checker reconstructed a reduction descriptor"


def kernel_support(spec, checker, artifact="ptx"):
    """Stage 1 verdict for one kernel: (verdict, detail).

    A declared ``known_limitation`` (the author knows the checker can't model this
    kernel soundly, e.g. control flow) downgrades to LIMITED regardless of the
    descriptor heuristic — Stage 1 is a placeholder until the checker can declare
    real support itself.
    """
    ref = spec.config_space("light")[0]
    try:
        ck = spec.compile(ref, spec.precision_size)
        desc = checker(ck.asm[artifact])
    except Exception as exc:  # noqa: BLE001
        return "UNSUPPORTED", f"reference config failed to compile or check: {type(exc).__name__}"
    verdict, detail = _descriptor_verdict(desc)
    if spec.known_limitation:
        return "LIMITED", spec.known_limitation
    return verdict, detail


# --------------------------------------------------------------------------- #
# Stage 2 — precision
# --------------------------------------------------------------------------- #
def _spanned_axes(configs):
    """Which knobs vary within a set (= the tuning freedom that set recovers)."""
    spans = {}
    for axis in _AXIS_ORDER:
        vals = sorted({getattr(c, axis) for c in configs}, key=lambda v: (v is None, v))
        if len(vals) > 1:
            spans[axis] = vals
    return spans


def evaluate_precision(spec, checker, config_effort, fuzzer_effort, artifact="ptx"):
    """Compile + checker-partition + fuzz one kernel; return a result dict."""
    size = spec.precision_size
    configs = spec.config_space(config_effort)
    compiled, checker_key, fails = {}, {}, 0
    for config in configs:
        try:
            ck = spec.compile(config, size)
        except Exception:  # noqa: BLE001
            fails += 1
            continue
        compiled[config] = ck
        checker_key[config] = checker(ck.asm[artifact])
    ok = list(compiled)
    if not ok:
        return dict(name=spec.name, attempted=len(configs), ok=0, fails=fails)

    repeats = equivalence_fuzzer.effort_repeats(fuzzer_effort)
    empirical_key = equivalence_fuzzer.empirical_keys(
        lambda config, seed: spec.run(config, compiled[config], seed, size), ok, repeats)

    checker_sets = equivalence_fuzzer.partition(ok, checker_key)
    empirical_sets = equivalence_fuzzer.partition(ok, empirical_key)
    holds, straddle = equivalence_fuzzer.refines(ok, checker_key, empirical_key)
    largest = max(checker_sets, key=len)
    return dict(name=spec.name, attempted=len(configs), ok=len(ok), fails=fails, checker_n=len(checker_sets),
                checker_max=equivalence_fuzzer.max_class_size(checker_sets), empirical_n=len(empirical_sets),
                empirical_max=equivalence_fuzzer.max_class_size(empirical_sets),
                over_merges=equivalence_fuzzer.over_merges(ok, checker_key, empirical_key), refines=holds,
                straddle=straddle, largest_set=sorted(largest, key=config_label), largest_spans=_spanned_axes(largest))


# --------------------------------------------------------------------------- #
# Stage 3 — performance
# --------------------------------------------------------------------------- #
def evaluate_performance(spec, checker, artifact="ptx", config_effort="light"):
    """Benchmark a kernel across its config space; compare within-set spread to the global ceiling."""
    size = spec.perf_size
    configs = spec.config_space(config_effort)
    ms, bits, checker_key, fails = {}, {}, {}, 0
    for config in configs:
        try:
            milliseconds, output_bytes, asm = spec.benchmark(config, size)
        except Exception:  # noqa: BLE001
            fails += 1
            continue
        ms[config], bits[config], checker_key[config] = milliseconds, output_bytes, checker(asm[artifact])
    ok = list(ms)
    if not ok:
        return dict(name=spec.name, ok=0, fails=fails)

    ceiling = min(ms[c] for c in ok)
    ceiling_config = min(ok, key=lambda c: ms[c])
    slowest = max(ms[c] for c in ok)

    def best_slow(group):
        return min(ms[c] for c in group), max(ms[c] for c in group)

    checker_big = max(equivalence_fuzzer.partition(ok, checker_key), key=len)
    empirical_big = max(equivalence_fuzzer.partition(ok, bits), key=len)
    chk_fast, chk_slow = best_slow(checker_big)
    emp_fast, emp_slow = best_slow(empirical_big)
    return dict(name=spec.name, ok=len(ok), fails=fails, size=size, ceiling=ceiling, ceiling_config=ceiling_config,
                slowest=slowest, checker_set_size=len(checker_big),
                checker_byte_identical=len({bits[c]
                                            for c in checker_big}) == 1, checker_fast=chk_fast, checker_slow=chk_slow,
                empirical_set_size=len(empirical_big), empirical_fast=emp_fast, empirical_slow=emp_slow)


# --------------------------------------------------------------------------- #
# Stage 4 — equivalence under register pressure (post-ptxas / the PTX->SASS gap)
# --------------------------------------------------------------------------- #
def evaluate_ptxas_robustness(spec, checker, config_effort, fuzzer_effort, artifact, maxnreg_sweep):
    """Does the checker's equivalence verdict survive ptxas across a WHOLE diverse set?

    Take the full ``config_effort`` config space (diverse num_warps / num_stages /
    enable_fp_fusion / block_n) and ALSO cap ``maxnreg`` low enough to make ptxas spill. A
    ``.maxnreg`` directive leaves the PTX reduction body identical, so it does not change the
    checker descriptor -- the diverse configs and the register regimes fall into the SAME
    checker-equivalence classes. We then fuzz every ``(config, maxnreg)`` member and check each
    checker class is still a single bit-class. So this stresses the cross-config equivalence
    claim AND the PTX->SASS gap (register spilling) at once: one checker class == one bit-class
    (over-merges 0) means equivalence holds across diverse configs even under spilling.

    Uses ``precision_size`` (a sizable 8K-16K-column tensor): the larger ``perf_size`` overflows
    shared memory / trips ptxas at heavy config scale, and register pressure here is driven by the
    config diversity (num_stages / block_n) plus the ``maxnreg`` cap, not by tensor size.
    """
    size = spec.precision_size
    caps = [None] + list(maxnreg_sweep)
    base = spec.config_space(config_effort)
    compiled, checker_key, nregs, nspills, items, fails = {}, {}, {}, {}, [], 0
    for cfg in base:
        for cap in caps:
            key = (cfg, cap)
            try:
                ck = spec.compile(cfg, size, maxnreg=cap)
                spec.run(cfg, ck, 0, size)  # one launch populates ck.n_regs / ck.n_spills (lazy)
            except Exception:  # noqa: BLE001
                fails += 1
                continue
            compiled[key] = ck
            checker_key[key] = checker(ck.asm[artifact])
            nregs[key], nspills[key] = getattr(ck, "n_regs", None), getattr(ck, "n_spills", None)
            items.append(key)
    attempted = len(base) * len(caps)
    if not items:
        return dict(name=spec.name, ok=0, attempted=attempted, fails=fails)

    repeats = equivalence_fuzzer.effort_repeats(fuzzer_effort)
    empirical_key = equivalence_fuzzer.empirical_keys(lambda key, seed: spec.run(key[0], compiled[key], seed, size),
                                                      items, repeats)
    checker_sets = equivalence_fuzzer.partition(items, checker_key)
    empirical_sets = equivalence_fuzzer.partition(items, empirical_key)
    holds, _ = equivalence_fuzzer.refines(items, checker_key, empirical_key)
    largest = max(checker_sets, key=len)

    def _reg_range(keys):
        vals = [nregs[k] for k in keys if nregs[k] is not None]
        return (min(vals), max(vals)) if vals else None

    def _spill_range(keys):
        vals = [nspills[k] for k in keys if nspills[k] is not None]
        return (min(vals), max(vals)) if vals else None

    spans = {}
    for axis in _AXIS_ORDER:
        vals = sorted({getattr(k[0], axis) for k in largest}, key=lambda v: (v is None, v))
        if len(vals) > 1:
            spans[axis] = vals
    return dict(name=spec.name, size=size, ok=len(items), attempted=attempted, fails=fails, n_configs=len(base),
                caps=caps, checker_n=len(checker_sets), checker_max=len(largest), empirical_n=len(empirical_sets),
                empirical_max=equivalence_fuzzer.max_class_size(empirical_sets),
                over_merges=equivalence_fuzzer.over_merges(items, checker_key, empirical_key), refines=holds,
                n_spilled=sum(1 for k in items
                              if (nspills[k] or 0) > 0), reg_range=_reg_range(items), spill_range=_spill_range(items),
                largest_spans=spans, largest_maxnreg=sorted({k[1]
                                                             for k in largest}, key=lambda v: (v is None, v)),
                largest_reg_range=_reg_range(largest), largest_spilled=sum(1 for k in largest if (nspills[k] or 0) > 0))


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def _table(headers, rows):
    grid = [headers] + [[str(x) for x in r] for r in rows]
    widths = [max(len(row[i]) for row in grid) for i in range(len(headers))]
    fmt = lambda vals: "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    return [fmt(headers), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows[:]]


def render_stage1(results):
    lines = ["STAGE 1 — kernel support", "-" * 24, ""]
    rows = [[r["name"], r["verdict"], r["detail"]] for r in results]
    lines += _table(["kernel", "verdict", "detail"], rows)
    lines.append("")
    return lines


def render_stage2(results):
    lines = ["STAGE 2 — checker precision (soundness + partition)", "-" * 51, ""]
    rows = []
    for r in results:
        if not r.get("ok"):
            rows.append([r["name"], f"0/{r['attempted']}", "-", "-", "-", "-", "-"])
            continue
        rows.append([
            r["name"],
            f"{r['ok']}/{r['attempted']}",
            f"{r['checker_n']} (max {r['checker_max']})",
            f"{r['empirical_n']} (max {r['empirical_max']})",
            str(r["over_merges"]),
            "yes" if r["refines"] else "NO",
        ])
    lines += _table(["kernel", "configs", "checker sets", "empirical sets (ceiling)", "over-merges", "refines"],
                    [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in rows])
    lines.append("")
    lines.append("over-merges = configs the checker merged but the fuzzer separated (MUST be 0 = sound).")
    lines.append("empirical-sets max = the recovery ceiling (the largest bit-equivalent set that exists).")
    lines.append("")
    lines.append("Largest checker-equivalent set per kernel (the recovered tuning freedom):")
    for r in results:
        if not r.get("ok"):
            continue
        spans = r["largest_spans"]
        span_text = "; ".join(f"{a}∈{[v for v in vals]}" for a, vals in spans.items()) or "(singleton)"
        lines.append(f"  {r['name']}: {len(r['largest_set'])} configs spanning {span_text}")
        for config in r["largest_set"][:5]:
            lines.append(f"      - {config_label(config)}")
        if len(r["largest_set"]) > 5:
            lines.append(f"      - ... (+{len(r['largest_set']) - 5} more)")
    lines.append("")
    return lines


def render_stage3(results):
    lines = ["STAGE 3 — performance vs ceiling (optional)", "-" * 43, ""]
    for r in results:
        if not r.get("ok"):
            lines.append(f"{r['name']}: no configs benchmarked (failed {r.get('fails', 0)}).")
            lines.append("")
            continue
        rows, cols = r["size"]
        lines.append(f"{r['name']}  (size {rows}x{cols}, do_bench x3 min-of-medians)")
        lines.append(f"  unconstrained CEILING (fastest of all, bits NOT reproducible): "
                     f"{r['ceiling']:.3f} ms @ {config_label(r['ceiling_config'])}")
        lines.append(f"  full-space spread: {r['slowest']:.3f} ms slowest -> {r['slowest'] / r['ceiling']:.2f}x")
        lines.append(f"  largest CHECKER-certified set: {r['checker_set_size']} configs, "
                     f"byte-identical={r['checker_byte_identical']}; "
                     f"{r['checker_fast']:.3f}..{r['checker_slow']:.3f} ms "
                     f"({r['checker_slow'] / r['checker_fast']:.2f}x within-set freedom); "
                     f"best is {r['checker_fast'] / r['ceiling']:.2f}x off ceiling")
        lines.append(f"  largest EMPIRICAL (true) set: {r['empirical_set_size']} configs; "
                     f"{r['empirical_fast']:.3f}..{r['empirical_slow']:.3f} ms "
                     f"({r['empirical_slow'] / r['empirical_fast']:.2f}x) -> the freedom a perfect checker recovers")
        lines.append("")
    return lines


def render_stage4(results):
    lines = ["STAGE 4 — equivalence under register pressure (post-ptxas / the PTX->SASS gap)", "-" * 78, ""]
    lines.append("Fuzz each WHOLE checker-equivalence set (DIVERSE configs) while also capping maxnreg to make")
    lines.append("ptxas spill. maxnreg only adds a .maxnreg directive (same PTX body => same checker classes) but")
    lines.append("changes ptxas allocation. One checker class == one bit-class (over-merges 0) => equivalence holds")
    lines.append("across diverse configs AND register spilling.")
    lines.append("")
    total_om = 0
    for r in results:
        if not r.get("ok"):
            lines.append(f"{r['name']}: no members compiled ({r.get('attempted')} attempted, {r.get('fails')} failed).")
            lines.append("")
            continue
        total_om += r["over_merges"]
        nr, nc = r["size"]
        caps = ["none" if c is None else str(c) for c in r["caps"]]
        lines.append(f"{r['name']}  (size {nr}x{nc}; {r['n_configs']} configs x maxnreg{caps} = "
                     f"{r['ok']}/{r['attempted']} members)")
        rr, sr = r["reg_range"], r["spill_range"]
        lines.append(f"  ptxas variation: n_regs {rr[0]}..{rr[1]}" +
                     (f" | spill 0..{sr[1]} B" if sr and sr[1] else " | spill 0 B") +
                     f" | spilled members: {r['n_spilled']}/{r['ok']}")
        lines.append(f"  checker classes: {r['checker_n']} (max {r['checker_max']}) | empirical classes: "
                     f"{r['empirical_n']} (max {r['empirical_max']}) | over-merges: {r['over_merges']} "
                     f"| refines: {'yes' if r['refines'] else 'NO'}")
        span_txt = "; ".join(f"{a}∈{v}" for a, v in r["largest_spans"].items()) or "(singleton)"
        mnr = ["none" if c is None else c for c in r["largest_maxnreg"]]
        lgr = r["largest_reg_range"]
        lines.append(f"  largest class: {r['checker_max']} members spanning {span_txt}; maxnreg∈{mnr}" +
                     (f"; n_regs {lgr[0]}..{lgr[1]}" if lgr else "") +
                     f"; spilled {r['largest_spilled']}/{r['checker_max']}")
        lines.append(f"  => equivalence under register pressure: {'PRESERVED' if r['over_merges'] == 0 else 'BROKEN'}")
        lines.append("")
    lines.append(f"OVERALL over-merges across kernels (under forced spilling): {total_om} "
                 f"({'PTX equivalence survives ptxas across the whole set' if total_om == 0 else 'BREAK detected'})")
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
    p = argparse.ArgumentParser(description="bitequiv evaluation framework (see module docstring).")
    p.add_argument("--kernels", default="all", help="'all' or comma list: " + ",".join(eval_kernels.REGISTRY))
    p.add_argument("--stages", default="1,2", help="comma list of stages to run (Stage 3 perf is opt-in)")
    p.add_argument("--config-effort", default="light", choices=("light", "heavy"))
    p.add_argument("--fuzzer-effort", default="fast", choices=("fast", "convincing"))
    p.add_argument("--checker", default=_DEFAULT_CHECKER, help="module:function descriptor (default: repo checker)")
    p.add_argument("--artifact", default="ptx", choices=("ptx", "ttgir"),
                   help="which compiled IR artifact to feed the checker (default: ptx)")
    p.add_argument(
        "--allow-unsound", action="store_true",
        help="report over-merges but exit 0 instead of 1 — for checkers that are "
        "expectedly not sound (e.g. the TTGIR checker, blind to FMA contraction)")
    p.add_argument(
        "--maxnreg-sweep", default="16",
        help="Stage 4: comma list of maxnreg caps (added on top of the diverse config space to "
        "force ptxas spilling; a None/uncapped baseline is always included). Each cap multiplies "
        "the member count, so keep it small at heavy config effort")
    p.add_argument("--out", default=_DEFAULT_OUT, help="result table path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("no CUDA GPU available; the framework needs one to compile and run kernels. Skipping.")
        return 0

    stages = {int(s) for s in args.stages.split(",") if s.strip()}
    specs = resolve_kernels(args.kernels)
    checker = load_checker(args.checker)
    repeats = equivalence_fuzzer.effort_repeats(args.fuzzer_effort)
    device = torch.cuda.get_device_name()

    header = [
        "bitequiv evaluation result",
        "==========================",
        f"generated:      {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"checker:        {args.checker}",
        f"artifact:       {args.artifact}",
        f"device:         {device}",
        f"config effort:  {args.config_effort}",
        f"fuzzer effort:  {args.fuzzer_effort} ({repeats} seeds/config)",
        f"kernels:        {', '.join(s.name for s in specs)}",
        f"stages:         {','.join(str(s) for s in sorted(stages))}",
        "",
    ]
    print("\n".join(header))

    sections = []
    total_over_merges = 0
    any_unsound = False

    if 1 in stages:
        print("== Stage 1: support ==", flush=True)
        results = []
        for spec in specs:
            verdict, detail = kernel_support(spec, checker, args.artifact)
            results.append(dict(name=spec.name, verdict=verdict, detail=detail))
            print(f"  {spec.name}: {verdict} — {detail}", flush=True)
        sections.append(render_stage1(results))

    if 2 in stages:
        print("== Stage 2: precision ==", flush=True)
        results = []
        for spec in specs:
            n_configs = len(spec.config_space(args.config_effort))
            print(f"  {spec.name}: {n_configs} configs x {repeats} fuzz seeds ...", flush=True)
            r = evaluate_precision(spec, checker, args.config_effort, args.fuzzer_effort, args.artifact)
            results.append(r)
            if r.get("ok"):
                total_over_merges += r["over_merges"]
                any_unsound = any_unsound or not r["refines"]
                print(
                    f"    -> {r['ok']}/{r['attempted']} ok | checker max set {r['checker_max']} "
                    f"| empirical ceiling {r['empirical_max']} | over-merges {r['over_merges']} "
                    f"| refines {r['refines']}", flush=True)
            else:
                print(f"    -> 0/{r['attempted']} compiled (build drift?)", flush=True)
        sections.append(render_stage2(results))

    if 3 in stages:
        print("== Stage 3: performance ==", flush=True)
        results = []
        for spec in specs:
            if not spec.supports_perf:
                print(f"  {spec.name}: no perf hook; skipping.", flush=True)
                continue
            print(f"  {spec.name}: benchmarking {len(spec.config_space(args.config_effort))} configs ...", flush=True)
            results.append(evaluate_performance(spec, checker, args.artifact, args.config_effort))
        if results:
            sections.append(render_stage3(results))

    if 4 in stages:
        print("== Stage 4: equivalence under register pressure (post-ptxas) ==", flush=True)
        maxnreg_sweep = [int(x) for x in args.maxnreg_sweep.split(",") if x.strip()]
        results = []
        for spec in specs:
            members = len(spec.config_space(args.config_effort)) * (1 + len(maxnreg_sweep))
            print(
                f"  {spec.name}: ~{members} (config x maxnreg{['none'] + maxnreg_sweep}) members "
                f"x {repeats} fuzz seeds ...", flush=True)
            r = evaluate_ptxas_robustness(spec, checker, args.config_effort, args.fuzzer_effort, args.artifact,
                                          maxnreg_sweep)
            results.append(r)
            if r.get("ok"):
                print(
                    f"    -> {r['ok']}/{r['attempted']} members | spilled {r['n_spilled']} | checker classes "
                    f"{r['checker_n']} (max {r['checker_max']}) | over-merges {r['over_merges']}", flush=True)
            else:
                print(f"    -> 0/{r.get('attempted')} compiled (fails: {r.get('fails')})", flush=True)
        sections.append(render_stage4(results))

    write_report(args.out, header, sections)
    print(f"\nWROTE {args.out}")
    if 2 in stages:
        gate = "0; --allow-unsound" if args.allow_unsound else "must be 0"
        print(f"SOUNDNESS: total over-merges ({gate}) = {total_over_merges}")
        if any_unsound or total_over_merges:
            print("RESULT: UNSOUND — the checker merged configs the fuzzer proved different.")
            if not args.allow_unsound:
                return 1
            print("NOTE: --allow-unsound set — reporting the over-merges instead of failing "
                  "(expected for the TTGIR checker, which is blind to FMA contraction).")
            return 0
        print("RESULT: SOUND — checker partition refines the empirical partition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
