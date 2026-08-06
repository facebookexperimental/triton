"""bitequiv evaluation framework — the team's standard ruler for a PTX equivalence checker.

WHAT THIS IS
------------
One driver, a few opt-in stages, run from the command line, that measure how good
a static bitwise-equivalence checker is at deciding which autotuner configs of a
kernel produce identical output bits. The checker under test is pluggable
(``--checker``); by default it is the repo checker
``bitequiv.ptx_reduction:ptx_reduction_descriptor``. This script changes no
checker code — it only evaluates one.

THE STAGES (pick with ``--stages``, comma list of names)
-------------------------------------------------------
  precision   — Of the configs it supports, how well does the checker partition
      them, and is it SOUND? It builds the config space, compiles each config to
      PTX, asks the checker to group them, and independently fuzzes every config
      (the empirical ground truth). It reports, per kernel:
        * checker sets and the largest one (the tuning freedom the checker recovers);
        * empirical sets and the largest one (the recovery CEILING);
        * OVER-MERGES — pairs the checker called equal but the fuzzer separated.
          The soundness violation count; MUST be 0;
        * OVER-SPLITS — pairs the fuzzer merged but the checker separated. Recovery
          left on the table (safe, not a soundness bug — just conservative);
        * REFINES — whether the checker partition refines the empirical one.

  performance — Given a bit-exactness constraint, how much speed is on the table?
      Benchmarks every config, finds the global-fastest CEILING (a normal
      autotuner, no equivalence constraint), then looks inside one
      checker-certified set: after verifying every member is byte-identical it
      reports fastest vs slowest member (the freedom the checker hands the
      autotuner) and best member vs the ceiling (the cost of identical bits).

  regpressure — Does the checker's verdict survive ptxas (PTX -> SASS) across a
      WHOLE diverse set? Fuzzes each checker set while ALSO capping ``maxnreg`` low
      enough to make ptxas spill. ``.maxnreg`` leaves the PTX body identical (same
      checker classes) but changes ptxas allocation, so the set spans diverse
      configs AND register regimes. One checker class == one bit-class => equivalence
      holds even under spilling.

  korder      — K-reduction-order experiment (GEMM v5/v2 bit-invariance, split-K
      order sensitivity, 2-CTA). Placeholder here; filled by the korder stage.

  cublas      — bit-for-bit match against a cuBLAS reference. Placeholder.

EFFORT (``--effort``, applies to ``precision`` + ``performance`` only)
--------------------------------------------------------------------
  light  — a few priority kernels, one representative per bit-relevant bucket,
           small input, 10 fuzz seeds. A ~15-min smoke / demo run.
  mid    — the full max config space, larger input, 20 seeds. The per-diff table.
  heavy  — the full max config space, ML-scale input, 20 seeds. The headline run.
Input sizes scale with effort: GEMM 256^3 / 2048^3 / 8192^3; reduction 32x2048 /
512x8192 / 2048x16384. ``regpressure`` / ``korder`` / ``cublas`` always run at mid.
A fuzzer can only refute equivalence, never prove it, so more seeds = stronger
evidence of soundness, never certainty.

DATATYPE (``--dtypes``, comma list or ``all``)
----------------------------------------------
Each base kernel is materialized into one named spec per selected dtype, named
``<kernel>_<dtype>`` (e.g. ``gemm_f16``, ``gemm_bf16``, ``sum_f32``). dtype is a
first-class config axis for GEMMs, never collapsed — every dtype gets its own row.

OUTPUT
------
A plain, human-readable table file (``--out``, default
``bitequiv/evaluation/result.txt``; gitignored — regenerate on demand) plus a
short summary on stdout whose soundness line the pytest gate parses.

LAYOUT
------
  eval_kernels.py       — every test kernel + its KernelSpec (the registry / the
                          MAX config space each kernel declares).
  equivalence_fuzzer.py — standalone empirical oracle + partition/soundness math.
  evaluate.py (here)    — the staged CLI driver; orchestration + effort sampling.

EXAMPLES
--------
  # quick smoke / what the pytest gate runs:
  python -m bitequiv.evaluation.evaluate --stages precision --effort light
  # the per-diff table (precision + performance) over every kernel and dtype:
  python -m bitequiv.evaluation.evaluate --stages precision,performance --effort mid
  # just the GEMMs, bf16 only, headline soundness run:
  python -m bitequiv.evaluation.evaluate --kernels gemm,gemm_kgroup --dtypes bf16 --effort heavy
  # evaluate a different / experimental checker:
  python -m bitequiv.evaluation.evaluate --checker my.module:my_descriptor
"""

import argparse
import dataclasses
import datetime
import importlib
import inspect
import os
import sys

# Make ``bitequiv`` importable whether launched as ``-m bitequiv.evaluation.evaluate``
# or as a plain script path.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

from bitequiv.evaluation import eval_kernels, equivalence_fuzzer  # noqa: E402
from bitequiv.evaluation.eval_kernels import _AXIS_ORDER, build_configs, config_label, resolve_kernels  # noqa: E402

_DEFAULT_CHECKER = "bitequiv.ptx_reduction:ptx_reduction_descriptor"
_DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "result.txt")

# The effort knob. Only ``precision`` and ``performance`` read it; the other stages
# always run at ``mid``. Each level fixes three things at once: how many fuzz seeds
# per config, whether to use the full max config space or a small subsample, and the
# input tensor size (per kernel family, resolved in ``_effort_size``).
_EFFORT = {
    "light": dict(repeats=10, gemm_size=(256, 256, 256), reduction_size=(32, 2048),
                  default_kernels=("sum", "dot", "softmax", "gemm", "gemm_kgroup", "gemm_splitk")),
    "mid": dict(repeats=20, gemm_size=(2048, 2048, 2048), reduction_size=(512, 8192), default_kernels=None),
    "heavy": dict(repeats=20, gemm_size=(8192, 8192, 8192), reduction_size=(2048, 16384), default_kernels=None),
}
_LIGHT_PER_BUCKET = 4  # light keeps this many configs per bit-relevant bucket (some free-axis spread to merge)
_MID = "mid"  # stages other than precision/performance always run here
_STAGES = ("precision", "performance", "regpressure", "korder", "cublas")

# How much of the max config space each effort actually RUNS (input size scales separately in _EFFORT):
#   light — a few configs per bit-relevant bucket (smoke / demo);
#   mid   — the PROVEN old-experiment "heavy" curated axis values (the scope prior per-diff tables were
#           built on): a tractable subset of the full grid. Axes added later (gemm_num_ctas, gemm_combine,
#           gemm_block_m=32 / MMAv2) are pinned to their old single value so mid stays known-good; the
#           v5/v2 and 2-CTA stories are covered by the korder stage instead;
#   heavy — the full max grid (spec.max_config_space()).
_MID_AXIS_VALUES = {
    "reduction_ordering": ("unordered", "inner_tree"),
    "num_warps": (1, 2, 4, 8, 16, 32),
    "num_stages": (1, 2, 3, 4, 5, 6),
    "enable_fp_fusion": (True, False),
    "block_n": (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
    "gemm_block_m": (64, 128),
    "gemm_block_n": (64, 128, 256),
    "gemm_block_k": (16, 32, 64),
    "input_precision": ("tf32", "ieee", "tf32x3"),
    "max_num_imprecise_acc": (32, 64, 128),
    "gemm_num_splits": (1, 2, 4, 8),
    "gemm_num_ctas": (1, ),
    "gemm_combine": ("seq", ),
}


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


def make_run_checker(checker):
    """Adapt a checker to a uniform ``run(asm_text, config) -> hashable`` call.

    A checker that takes a second parameter (the TTGIR checker, which folds
    ``enable_fp_fusion`` — invisible in the IR — into its descriptor) is handed the config;
    a one-argument checker (the PTX checker, which reads fusion straight from the PTX) is
    called with the IR text alone, so it is completely unaffected by this plumbing."""
    try:
        params = [
            p for p in inspect.signature(checker).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        ]
        wants_config = len(params) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in params)
    except (TypeError, ValueError):
        wants_config = False
    if wants_config:
        return lambda asm_text, config: checker(asm_text, config)
    return lambda asm_text, config: checker(asm_text)
# --------------------------------------------------------------------------- #
# Effort sampling — the suite declares the MAX config space + a base size; the
# script picks how much of it to run, and at what input size, per effort.
# --------------------------------------------------------------------------- #
def _size_family(spec):
    """gemm (3-D M/N/K) | reduction (2-D rows/cols) | other (keep the spec's baked size)."""
    if spec.name.startswith("gemm") and len(spec.precision_size) == 3:
        return "gemm"
    if len(spec.precision_size) == 2:
        return "reduction"
    return "other"


def _effort_size(spec, effort, perf=False):
    """Input size for this kernel at this effort. GEMM/reduction families scale with effort;
    every other shape (3-D/4-D reductions, batched GEMM, ...) keeps its baked spec size."""
    fam = _size_family(spec)
    if fam == "gemm":
        return _EFFORT[effort]["gemm_size"]
    if fam == "reduction":
        return _EFFORT[effort]["reduction_size"]
    return spec.perf_size if perf else spec.precision_size


def _resolve_configs(spec, effort):
    """The config set to actually run at this effort (see _MID_AXIS_VALUES for the levels).

    heavy = the full max grid; mid = the proven old-experiment curated axis values (dtype pinned to the
    spec's, num_ctas/combine to their old single value); light = a few configs per *bit-relevant bucket*
    (every bit distinction stays present so the over-merge picture is honest, free-axis variation trimmed)."""
    if effort == "heavy":
        return spec.max_config_space()
    if effort == "mid":
        overrides = dict(_MID_AXIS_VALUES)
        overrides.update(spec.axis_values or {})  # a spec's own splitting-knob override wins
        if "dtype" in spec.axes and spec.valid_dtypes:
            overrides["dtype"] = spec.valid_dtypes
        configs = build_configs(spec.axes, overrides)
        if spec.config_filter is not None:
            configs = [c for c in configs if spec.config_filter(c)]
        return configs
    full = spec.max_config_space()  # light
    buckets = {}
    for config in full:
        key = tuple(getattr(config, axis) for axis in spec.bit_relevant_axes)
        buckets.setdefault(key, []).append(config)
    out = []
    for group in buckets.values():
        out.extend(group[:_LIGHT_PER_BUCKET])
    return out


def _materialize(specs, dtypes_selector):
    """Expand each base spec into one named spec per selected dtype (``<name>_<dtype>``).

    A GEMM's ``dtype`` config axis is pinned to the one dtype (so ``gemm_f16`` runs only
    f16 configs); a single-dtype kernel just gets the suffix on its name. ``dtypes_selector``
    is ``all`` or a comma list; a spec with no selected dtype among its ``valid_dtypes`` is
    dropped (e.g. ``--dtypes bf16`` drops the f32-only reductions)."""
    selected = None if dtypes_selector == "all" else {d.strip() for d in dtypes_selector.split(",") if d.strip()}
    out = []
    for spec in specs:
        for dt in (spec.valid_dtypes or ("f32", )):
            if selected is not None and dt not in selected:
                continue
            out.append(dataclasses.replace(spec, name=f"{spec.name}_{dt}", valid_dtypes=(dt, )))
    return out


def _spanned_axes(configs):
    """Which knobs vary within a set (= the tuning freedom that set recovers)."""
    spans = {}
    for axis in _AXIS_ORDER:
        vals = sorted({getattr(c, axis) for c in configs}, key=lambda v: (v is None, v))
        if len(vals) > 1:
            spans[axis] = vals
    return spans


# --------------------------------------------------------------------------- #
# Stage: precision (soundness + partition quality)
# --------------------------------------------------------------------------- #
def evaluate_precision(spec, run_checker, effort, artifact="ptx"):
    """Compile + checker-partition + fuzz one kernel; return a result dict."""
    size = _effort_size(spec, effort)
    configs = _resolve_configs(spec, effort)
    repeats = _EFFORT[effort]["repeats"]
    compiled, checker_key, fails = {}, {}, 0
    for config in configs:
        try:
            ck = spec.compile(config, size)
            spec.run(config, ck, 0, size)  # trial launch: a config can COMPILE yet OOM at LAUNCH
        except Exception:  # noqa: BLE001    (tcgen05 TMEM is rejected by the driver at launch, not
            fails += 1  # at compile), so skip compile OR launch failures instead of letting one bad
            continue  # config abort the whole kernel's eval (the regpressure loop already does this)
        compiled[config] = ck
        checker_key[config] = run_checker(ck.asm[artifact], config)
    ok = list(compiled)
    if not ok:
        return dict(name=spec.name, attempted=len(configs), ok=0, fails=fails)

    empirical_key = equivalence_fuzzer.empirical_keys(
        lambda config, seed: spec.run(config, compiled[config], seed, size), ok, repeats)

    checker_sets = equivalence_fuzzer.partition(ok, checker_key)
    empirical_sets = equivalence_fuzzer.partition(ok, empirical_key)
    holds, straddle = equivalence_fuzzer.refines(ok, checker_key, empirical_key)
    largest = max(checker_sets, key=len)
    return dict(
        name=spec.name, attempted=len(configs), ok=len(ok), fails=fails, checker_n=len(checker_sets),
        checker_max=equivalence_fuzzer.max_class_size(checker_sets), empirical_n=len(empirical_sets),
        empirical_max=equivalence_fuzzer.max_class_size(empirical_sets),
        over_merges=equivalence_fuzzer.over_merges(ok, checker_key, empirical_key),
        # over-splits = the mirror of over-merges: pairs the fuzzer MERGED but the checker split
        # (recovery the checker left on the table). Safe direction, so it is a quality number, not a gate.
        over_splits=equivalence_fuzzer.over_merges(ok, empirical_key, checker_key), refines=holds, straddle=straddle,
        largest_set=sorted(largest, key=config_label), largest_spans=_spanned_axes(largest))


# --------------------------------------------------------------------------- #
# Stage: performance
# --------------------------------------------------------------------------- #
def evaluate_performance(spec, run_checker, effort, artifact="ptx"):
    """Benchmark a kernel across its config space; compare within-set spread to the global ceiling."""
    size = _effort_size(spec, effort, perf=True)
    configs = _resolve_configs(spec, effort)
    ms, bits, checker_key, fails = {}, {}, {}, 0
    for config in configs:
        try:
            milliseconds, output_bytes, asm = spec.benchmark(config, size)
        except Exception:  # noqa: BLE001
            fails += 1
            continue
        ms[config], bits[config], checker_key[config] = milliseconds, output_bytes, run_checker(asm[artifact], config)
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
# Stage: regpressure (post-ptxas / the PTX->SASS gap) — always runs at mid
# --------------------------------------------------------------------------- #
def evaluate_regpressure(spec, run_checker, artifact, maxnreg_sweep):
    """Does the checker's equivalence verdict survive ptxas across a WHOLE diverse set?

    Take the full mid config space (diverse num_warps / num_stages / enable_fp_fusion /
    block_n) and ALSO cap ``maxnreg`` low enough to make ptxas spill. A ``.maxnreg``
    directive leaves the PTX reduction body identical, so it does not change the checker
    descriptor -- the diverse configs and the register regimes fall into the SAME checker
    classes. We then fuzz every ``(config, maxnreg)`` member and check each checker class is
    still a single bit-class. So this stresses the cross-config equivalence claim AND the
    PTX->SASS gap (register spilling) at once: one checker class == one bit-class (over-merges
    0) means equivalence holds across diverse configs even under spilling.

    Uses the mid precision-style size: the larger perf size overflows shared memory / trips
    ptxas at full config scale, and register pressure here is driven by config diversity
    (num_stages / block_n) plus the ``maxnreg`` cap, not by tensor size.
    """
    size = _effort_size(spec, _MID)
    repeats = _EFFORT[_MID]["repeats"]
    caps = [None] + list(maxnreg_sweep)
    base = _resolve_configs(spec, _MID)
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
            checker_key[key] = run_checker(ck.asm[artifact], cfg)
            nregs[key], nspills[key] = getattr(ck, "n_regs", None), getattr(ck, "n_spills", None)
            items.append(key)
    attempted = len(base) * len(caps)
    if not items:
        return dict(name=spec.name, ok=0, attempted=attempted, fails=fails)

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
# Stage: korder — K-reduction-order experiment (placeholder; filled by the korder step)
# --------------------------------------------------------------------------- #
def _mma_version(ck):
    """Which tensor-core path the compiled GEMM took, read from its asm."""
    ptx = ck.asm.get("ptx", "") or ""
    ttgir = ck.asm.get("ttgir", "") or ""
    if "tcgen05.mma" in ptx:
        return "v5(tcgen05.mma)"
    if "mma.sync" in ptx:
        return "v2(mma.sync)"
    if "wgmma" in ptx:
        return "wgmma(Hopper)"
    if "fma.rn.f32" in ptx:
        return "FMA(no tensor core)"
    if "tcgen05" in ttgir or "tmem" in ttgir:
        return "v5(ttgir)"
    return "?"


def _pair_on(spec, axis, va, vb):
    """Two configs from the spec's space that differ ONLY in ``axis`` (values ``va`` vs ``vb``),
    matched on every other axis. Returns (None, None) if no matched pair exists."""
    other = [a for a in spec.axes if a != axis]
    groups = {}
    for c in spec.max_config_space():
        key = tuple(getattr(c, a) for a in other)
        slot = groups.setdefault(key, {})
        val = getattr(c, axis)
        if val == va:
            slot.setdefault("a", c)
        elif val == vb:
            slot.setdefault("b", c)
    for slot in groups.values():
        if "a" in slot and "b" in slot:
            return slot["a"], slot["b"]
    return None, None


def _bit_verdict(bytes_a, bytes_b):
    return "BIT-IDENTICAL" if bytes_a == bytes_b else "DIFFER"


def evaluate_korder(specs, checker, artifact):
    """GEMM K-reduction-order experiment (always mid size). Two questions, decided by BITS
    (empirical, not the checker):

      * v5 vs v2 tensor core — for each ``gemm_<dtype>``, compile the same config at
        ``gemm_block_m=32`` (lowers to MMAv2 ``mma.sync``) and at ``64`` (MMAv5 ``tcgen05``)
        and compare output bits. Expectation from the GB300 study: f16/bf16/f32 are
        BIT-IDENTICAL across v5/v2, fp8 DIFFERs (so a checker may merge v5/v2 for the
        former, must not for fp8).
      * K-split order — for ``gemm_kgroup`` / ``gemm_splitk``, compile the same config at
        ``gemm_num_splits`` 1 vs 2 and compare bits. Splitting the K reduction changes the
        summation order, so this is EXPECTED to DIFFER (num_splits is bit-relevant)."""
    size = _EFFORT[_MID]["gemm_size"]
    v5v2, ksplit, notes = [], [], []
    for spec in specs:
        if _size_family(spec) != "gemm":
            continue
        base = spec.name.rsplit("_", 1)[0]  # strip the _<dtype> suffix
        if base == "gemm":
            c2, c5 = _pair_on(spec, "gemm_block_m", 32, 64)
            if not (c2 and c5):
                v5v2.append(dict(kernel=spec.name, note="no matched block_m 32-vs-64 pair"))
                continue
            try:
                ck2 = spec.compile(c2, size)
                b2 = spec.run(c2, ck2, 0, size)
                ck5 = spec.compile(c5, size)
                b5 = spec.run(c5, ck5, 0, size)
            except Exception as exc:  # noqa: BLE001
                v5v2.append(dict(kernel=spec.name, note=f"compile/run failed: {type(exc).__name__}"))
                continue
            v5v2.append(dict(kernel=spec.name, v2=_mma_version(ck2), v5=_mma_version(ck5),
                             verdict=_bit_verdict(b2, b5)))
        elif base in ("gemm_kgroup", "gemm_splitk"):
            c1, cN = _pair_on(spec, "gemm_num_splits", 1, 2)
            if not (c1 and cN):
                ksplit.append(dict(kernel=spec.name, note="no matched num_splits 1-vs-2 pair"))
                continue
            try:
                ck1 = spec.compile(c1, size)
                b1 = spec.run(c1, ck1, 0, size)
                ckN = spec.compile(cN, size)
                bN = spec.run(cN, ckN, 0, size)
            except Exception as exc:  # noqa: BLE001
                ksplit.append(dict(kernel=spec.name, note=f"compile/run failed: {type(exc).__name__}"))
                continue
            ksplit.append(dict(kernel=spec.name, splits="1 vs 2", verdict=_bit_verdict(b1, bN)))
    notes.append("2-CTA: num_ctas is bit-free on a bare matmul (a real cta_group::2 needs the TLX "
                 "cluster path, not modeled here) — placeholder.")
    return dict(v5v2=v5v2, ksplit=ksplit, notes=notes)


# --------------------------------------------------------------------------- #
# Stage: cublas — bit-for-bit match against a cuBLAS reference (placeholder)
# --------------------------------------------------------------------------- #
def evaluate_cublas(specs, checker, artifact):
    """Compare Triton GEMM output bits against a cuBLAS reference. Not yet wired up."""
    return dict(note="cublas stage not yet implemented (non-split-K GEMM ≡ cuBLAS is the free-tiling class)")


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def _table(headers, rows):
    grid = [headers] + [[str(x) for x in r] for r in rows]
    widths = [max(len(row[i]) for row in grid) for i in range(len(headers))]
    fmt = lambda vals: "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    return [fmt(headers), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows[:]]


def render_precision(results):
    lines = ["STAGE precision — checker soundness + partition", "-" * 47, ""]
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
            str(r["over_splits"]),
            "yes" if r["refines"] else "NO",
        ])
    lines += _table(
        ["kernel", "configs", "checker sets", "empirical sets (ceiling)", "over-merges", "over-splits", "refines"],
        rows)
    lines.append("")
    lines.append("over-merges = configs the checker merged but the fuzzer separated (MUST be 0 = sound).")
    lines.append("over-splits = bit-equal configs the checker separated (recovery left on the table; safe, not a bug).")
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


def render_performance(results):
    lines = ["STAGE performance — speed vs the bit-exact ceiling", "-" * 50, ""]
    for r in results:
        if not r.get("ok"):
            lines.append(f"{r['name']}: no configs benchmarked (failed {r.get('fails', 0)}).")
            lines.append("")
            continue
        size_txt = "x".join(str(d) for d in r["size"])  # 2-D (rows, cols) reductions or 3-D (M, N, K) GEMMs
        lines.append(f"{r['name']}  (size {size_txt}, do_bench x3 min-of-medians)")
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


def render_regpressure(results):
    lines = ["STAGE regpressure — equivalence under register pressure (post-ptxas / the PTX->SASS gap)", "-" * 87, ""]
    lines.append("Fuzz each WHOLE checker-equivalence set (DIVERSE configs) while also capping maxnreg to make")
    lines.append("ptxas spill. maxnreg only adds a .maxnreg directive (same PTX body => same checker classes) but")
    lines.append("changes ptxas allocation. One checker class == one bit-class (over-merges 0) => equivalence holds")
    lines.append("across diverse configs AND register spilling. Always run at mid effort.")
    lines.append("")
    total_om = 0
    for r in results:
        if not r.get("ok"):
            lines.append(f"{r['name']}: no members compiled ({r.get('attempted')} attempted, {r.get('fails')} failed).")
            lines.append("")
            continue
        total_om += r["over_merges"]
        size_txt = "x".join(str(d) for d in r["size"])  # 2-D (rows, cols) reductions or 3-D (M, N, K) GEMMs
        caps = ["none" if c is None else str(c) for c in r["caps"]]
        lines.append(f"{r['name']}  (size {size_txt}; {r['n_configs']} configs x maxnreg{caps} = "
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


def render_korder(result):
    lines = ["STAGE korder — GEMM K-reduction-order (v5/v2 tensor core, K-split order)", "-" * 70, ""]
    lines.append("Decided by output BITS (empirical), not the checker. mid size. v5/v2: same config at")
    lines.append("gemm_block_m=32 (MMAv2 mma.sync) vs 64 (MMAv5 tcgen05). K-split: same config, num_splits 1 vs 2.")
    lines.append("")
    lines.append("v5 vs v2 tensor core (expect f16/bf16/f32 BIT-IDENTICAL, fp8 DIFFER):")
    rows = [[r["kernel"], r.get("v5", "-"), r.get("v2", "-"), r.get("verdict", r.get("note", "-"))]
            for r in result["v5v2"]] or [["(no GEMM specs selected)", "-", "-", "-"]]
    lines += _table(["kernel", "v5 path", "v2 path", "bit-compare"], rows)
    lines.append("")
    lines.append("K-split order (expect DIFFER — splitting K changes the summation order):")
    rows = [[r["kernel"], r.get("splits", "-"), r.get("verdict", r.get("note", "-"))]
            for r in result["ksplit"]] or [["(no split-K specs selected)", "-", "-"]]
    lines += _table(["kernel", "num_splits", "bit-compare"], rows)
    lines.append("")
    for note in result["notes"]:
        lines.append(f"  {note}")
    lines.append("")
    return lines


def render_note(title, result):
    return [title, "-" * len(title), "", f"  {result['note']}", ""]


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
    p.add_argument("--kernels", default="",
                   help="'all', a comma list of base kernel names, or empty to use the effort default "
                   "(light = a few priority kernels; mid/heavy = all). Base names: " + ",".join(eval_kernels.REGISTRY))
    p.add_argument("--dtypes", default="all",
                   help="'all' or comma list (f16,bf16,f32,fp8); each base kernel is materialized into one "
                   "named spec per selected dtype (<kernel>_<dtype>)")
    p.add_argument("--stages", default="precision", help="comma list of stages to run: " + ",".join(_STAGES))
    p.add_argument("--effort", default="mid", choices=("light", "mid", "heavy"),
                   help="config-subset x fuzz-seeds x input-size for precision/performance "
                   "(other stages always run at mid)")
    p.add_argument("--checker", default=_DEFAULT_CHECKER, help="module:function descriptor (default: repo checker)")
    p.add_argument("--artifact", default="ptx", choices=("ptx", "ttgir"),
                   help="which compiled IR artifact to feed the checker (default: ptx)")
    p.add_argument(
        "--allow-unsound", action="store_true",
        help="report over-merges but exit 0 instead of 1 — for checkers that are "
        "expectedly not sound (e.g. the TTGIR checker, blind to FMA contraction)")
    p.add_argument(
        "--maxnreg-sweep", default="16",
        help="regpressure: comma list of maxnreg caps (added on top of the diverse config space to "
        "force ptxas spilling; a None/uncapped baseline is always included). Each cap multiplies "
        "the member count, so keep it small")
    p.add_argument("--out", default=_DEFAULT_OUT, help="result table path")
    return p.parse_args(argv)


def _select_specs(kernels_arg, effort, dtypes_arg):
    """Resolve --kernels (effort-default when empty) into base specs, then materialize per dtype."""
    if kernels_arg:
        selector = kernels_arg
    elif _EFFORT[effort]["default_kernels"]:
        selector = ",".join(_EFFORT[effort]["default_kernels"])
    else:
        selector = "all"
    return _materialize(resolve_kernels(selector), dtypes_arg)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("no CUDA GPU available; the framework needs one to compile and run kernels. Skipping.")
        return 0

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in _STAGES]
    if bad:
        print(f"unknown stage(s) {bad}; valid: {', '.join(_STAGES)}")
        return 2
    specs = _select_specs(args.kernels, args.effort, args.dtypes)
    checker = load_checker(args.checker)
    run_checker = make_run_checker(checker)
    device = torch.cuda.get_device_name()

    header = [
        "bitequiv evaluation result",
        "==========================",
        f"generated:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"checker:     {args.checker}",
        f"artifact:    {args.artifact}",
        f"device:      {device}",
        f"effort:      {args.effort}  (precision/performance; other stages run at mid; "
        f"{_EFFORT[args.effort]['repeats']} fuzz seeds; config = "
        f"{ {'light': 'per-bucket subsample', 'mid': 'old-heavy curated', 'heavy': 'full max grid'}[args.effort] })",
        f"dtypes:      {args.dtypes}",
        f"kernels:     {', '.join(s.name for s in specs)}",
        f"stages:      {', '.join(stages)}",
        "",
    ]
    print("\n".join(header))

    sections = []
    total_over_merges = 0
    any_unsound = False

    if "precision" in stages:
        print("== precision ==", flush=True)
        results = []
        for spec in specs:
            n_configs = len(_resolve_configs(spec, args.effort))
            print(f"  {spec.name}: {n_configs} configs x {_EFFORT[args.effort]['repeats']} fuzz seeds ...", flush=True)
            r = evaluate_precision(spec, run_checker, args.effort, args.artifact)
            results.append(r)
            if r.get("ok"):
                total_over_merges += r["over_merges"]
                any_unsound = any_unsound or not r["refines"]
                print(
                    f"    -> {r['ok']}/{r['attempted']} ok | checker max set {r['checker_max']} "
                    f"| empirical ceiling {r['empirical_max']} | over-merges {r['over_merges']} "
                    f"| over-splits {r['over_splits']} | refines {r['refines']}", flush=True)
            else:
                print(f"    -> 0/{r['attempted']} compiled (build drift?)", flush=True)
            write_report(args.out, header, [render_precision(results)])  # checkpoint per kernel (survive a reap)
        sections.append(render_precision(results))

    if "performance" in stages:
        print("== performance ==", flush=True)
        results = []
        for spec in specs:
            if not spec.supports_perf:
                print(f"  {spec.name}: no perf hook; skipping.", flush=True)
                continue
            print(f"  {spec.name}: benchmarking {len(_resolve_configs(spec, args.effort))} configs ...", flush=True)
            results.append(evaluate_performance(spec, run_checker, args.effort, args.artifact))
        if results:
            sections.append(render_performance(results))

    if "regpressure" in stages:
        print("== regpressure (post-ptxas; mid effort) ==", flush=True)
        maxnreg_sweep = [int(x) for x in args.maxnreg_sweep.split(",") if x.strip()]
        results = []
        for spec in specs:
            members = len(_resolve_configs(spec, _MID)) * (1 + len(maxnreg_sweep))
            print(
                f"  {spec.name}: ~{members} (config x maxnreg{['none'] + maxnreg_sweep}) members "
                f"x {_EFFORT[_MID]['repeats']} fuzz seeds ...", flush=True)
            r = evaluate_regpressure(spec, run_checker, args.artifact, maxnreg_sweep)
            results.append(r)
            if r.get("ok"):
                print(
                    f"    -> {r['ok']}/{r['attempted']} members | spilled {r['n_spilled']} | checker classes "
                    f"{r['checker_n']} (max {r['checker_max']}) | over-merges {r['over_merges']}", flush=True)
            else:
                print(f"    -> 0/{r.get('attempted')} compiled (fails: {r.get('fails')})", flush=True)
        sections.append(render_regpressure(results))

    if "korder" in stages:
        print("== korder (mid effort) ==", flush=True)
        r = evaluate_korder(specs, checker, args.artifact)
        for row in r["v5v2"]:
            print(f"  v5/v2 {row['kernel']}: {row.get('verdict', row.get('note'))}", flush=True)
        for row in r["ksplit"]:
            print(f"  K-split {row['kernel']}: {row.get('verdict', row.get('note'))}", flush=True)
        sections.append(render_korder(r))

    if "cublas" in stages:
        print("== cublas (mid effort) ==", flush=True)
        r = evaluate_cublas(specs, checker, args.artifact)
        print(f"  {r['note']}", flush=True)
        sections.append(render_note("STAGE cublas — bit-for-bit vs a cuBLAS reference", r))

    write_report(args.out, header, sections)
    print(f"\nWROTE {args.out}")
    if "precision" in stages:
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
