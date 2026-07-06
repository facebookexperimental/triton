"""Pass A.5 data-partition enumeration driver (step 1 of solver-side DP).

The modulo pass dumps every legal M-split of each MMA accumulator as
``data_partition_candidates`` in ``schedule_graph.json`` (single source of
legality: ``enumerateDataPartitionCandidates`` in ModuloSchedulePass.cpp).
This driver enumerates that surface: it re-runs the modulo pass once per
candidate factor via ``triton-opt`` on the case's ``pre_modulo.ttgir``
(``TRITON_DATA_PARTITION_N=<n>``), runs the twill buffering solve on every
resulting dump, ranks all variants by model cost, and (optionally) drives the
GPU A/B gate for the best non-baseline variant.

Two decision modes:

* default (step 1): rank the partitioned variants by model cost, propose the
  best for the GPU A/B gate — the model NEVER rules between baseline and
  partitioned.
* ``--joint`` (step 2): the CP-SAT selection model
  (``cpsat_model.select_dp_factor``) picks the factor, baseline included.
  Only meaningful because the A.5 occupancy de-bias (applyDataPartition
  charges the bundle the conserved MAC area + N× issue floor, not N× the
  full-tile occupancy) makes baseline-vs-partitioned model costs comparable,
  and ``ddg_model``'s A.5 c_smem shrink lets the solver see the SMEM a
  partitioned epilogue frees for deeper operand rings.

Either way the model is only the proposer. The authoritative no-regression
guarantee is ``twill_solver.testing.ab``: the solver proposes, the GPU
disposes.

Known limit (v1): partitioned epilogues store per-group (m_size, BN) tiles,
so the candidate kernel needs descriptor blocks that differ from the case's
static ``bench_spec.py``. Until bench_specs derive descriptor blocks from the
generated module, ``--ab`` rejects partitioned candidates for launcher
reasons; validate them manually with matching descriptors (a_desc/b_desc
block (BM, BK)/(BN, BK), c_desc block (m_size, BN)).

Usage (from third_party/tlx/tools/sched2tlx, in the clean env):

  python -m twill_solver.dp_enum examples/case2_persistent_gemm
  python -m twill_solver.dp_enum examples/case2_persistent_gemm \
      --ab --ship /tmp/case2_shipped.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import cpsat_model, ddg_model, emit_check, graph_writer

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[5]  # third_party/tlx/tools/sched2tlx/twill_solver


# ---------------------------------------------------------------------------
# triton-opt discovery + per-factor re-dump
# ---------------------------------------------------------------------------


def find_triton_opt(explicit: str | None) -> Path:
    """--triton-opt flag > $TRITON_OPT > this repo's cmake build tree."""
    for cand in (explicit, os.environ.get("TRITON_OPT")):
        if cand:
            p = Path(cand)
            if p.is_file():
                return p
            raise FileNotFoundError(f"triton-opt not found at {p}")
    hits = sorted(_REPO_ROOT.glob("build/cmake.*/bin/triton-opt"))
    if hits:
        return hits[-1]
    raise FileNotFoundError(
        "triton-opt not found; build the repo or pass --triton-opt/$TRITON_OPT")


def _clean_env() -> dict[str, str]:
    """Subprocess env for triton-opt: drop the OHPC LD_LIBRARY_PATH (its old
    libstdc++ breaks binaries built with the system gcc) and any ambient
    factor override so each re-dump is exactly the factor we set."""
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("TRITON_DATA_PARTITION_N", None)
    return env


def redump(triton_opt: Path, pre_modulo: Path, factor: int, out_sg: Path,
           out_ddg: Path, timeout_s: float) -> None:
    """Re-run the modulo pass on `pre_modulo` with data-partition factor
    `factor` (1 = off), dumping schedule graph + DDG."""
    env = _clean_env()
    env["TRITON_MODULO_DUMP_SCHEDULE"] = str(out_sg)
    env["TRITON_MODULO_DUMP_DDG"] = str(out_ddg)
    if factor > 1:
        env["TRITON_DATA_PARTITION_N"] = str(factor)
    cmd = [str(triton_opt), "--nvgpu-modulo-schedule", str(pre_modulo),
           "-o", os.devnull]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       timeout=timeout_s)
    if p.returncode != 0:
        raise RuntimeError(
            f"triton-opt failed for factor {factor} (rc={p.returncode}):\n"
            f"{p.stderr[-2000:]}")
    if not out_sg.exists():
        raise RuntimeError(
            f"triton-opt succeeded but no dump at {out_sg} (factor {factor})")


# ---------------------------------------------------------------------------
# per-variant solve + gate
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    factor: int  # 1 = baseline (unpartitioned)
    sg_path: Path | None = None       # raw re-dump
    opt_path: Path | None = None      # after buffering solve
    cost: tuple | None = None         # Solution.cost() — lower is better
    score: int | None = None          # cpsat_model.dp_score (cross-variant)
    IIs: list[tuple[int, int]] = field(default_factory=list)
    applied: bool = False             # dump really carries partition tags
    emit_ok: bool = False
    markers: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def viable(self) -> bool:
        return self.error is None and self.emit_ok


def _applied_factor(sg: dict[str, Any]) -> int:
    """Largest applied_n over the dump's candidate entries (1 = none)."""
    return max((c.get("applied_n", 1)
                for c in sg.get("data_partition_candidates", [])), default=1)


def solve_variant(v: Variant, time_limit_s: float) -> None:
    """Buffering-solve one re-dump (fallback: reproduce when ortools is
    missing), stamp the solution, run the emitter dry-run gate."""
    model = ddg_model.load(v.sg_path)
    mode = "buffering" if cpsat_model._HAVE_ORTOOLS else "reproduce"
    sol = cpsat_model.solve(
        model, cpsat_model.SolveOptions(mode=mode, time_limit_s=time_limit_s))
    v.opt_path = v.sg_path.with_suffix(".optimized.json")
    graph_writer.write(model, sol, v.opt_path)
    v.cost = sol.cost()
    v.score = cpsat_model.dp_score(sol)
    v.IIs = [(lid, sol.loops[lid].II) for lid in sorted(sol.loops)]
    v.emit_ok, v.markers = emit_check.dry_run(v.opt_path)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case_dir", help="example case dir (needs pre_modulo.ttgir"
                    " + schedule_graph.json)")
    ap.add_argument("--triton-opt", help="triton-opt binary (default: repo "
                    "build tree, or $TRITON_OPT)")
    ap.add_argument("--out-dir", help="working dir for dumps (default: temp)")
    ap.add_argument("--factors", help="comma list overriding the enumerated "
                    "factors (e.g. 2,4)")
    ap.add_argument("--time-limit", type=float, default=10.0,
                    help="CP-SAT budget per buffering solve (s)")
    ap.add_argument("--redump-timeout", type=float, default=600.0,
                    help="per-factor triton-opt timeout (s)")
    ap.add_argument("--joint", action="store_true",
                    help="step 2: let the CP-SAT selection model pick the "
                    "factor (baseline included) by de-biased model cost, "
                    "instead of proposing the best partitioned variant")
    ap.add_argument("--ab", action="store_true",
                    help="GPU A/B the best viable candidate vs the case's "
                    "committed schedule_graph.json (authoritative gate)")
    ap.add_argument("--eps", type=float, default=0.03,
                    help="A/B perf regression tolerance")
    ap.add_argument("--ship", help="with --ab: write the winning generated.py "
                    "here (candidate if accepted, else baseline)")
    args = ap.parse_args(argv)

    case_dir = Path(args.case_dir).resolve()
    pre_modulo = case_dir / "pre_modulo.ttgir"
    committed_sg = case_dir / "schedule_graph.json"
    for p in (pre_modulo, committed_sg):
        if not p.exists():
            print(f"[dp_enum] missing {p}", file=sys.stderr)
            return 2

    triton_opt = find_triton_opt(args.triton_opt)
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        tempfile.mkdtemp(prefix=f"dp_enum_{case_dir.name}_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dp_enum] {case_dir.name}: triton-opt={triton_opt}\n"
          f"[dp_enum] work dir: {out_dir}", file=sys.stderr)

    # Baseline re-dump (factor 1) doubles as the candidate-surface read: the
    # committed schedule_graph.json may predate data_partition_candidates.
    base = Variant(factor=1, sg_path=out_dir / "sg_dp1.json")
    redump(triton_opt, pre_modulo, 1, base.sg_path, out_dir / "ddg_dp1.json",
           args.redump_timeout)
    base_raw = json.loads(base.sg_path.read_text())
    surface = base_raw.get("data_partition_candidates", [])

    if args.factors:
        factors = sorted({int(t) for t in args.factors.split(",") if t.strip()})
    else:
        factors = sorted({f["n"] for c in surface for f in c["factors"]})
    if not factors:
        print(f"[dp_enum] {case_dir.name}: no legal data-partition candidates "
              f"(surface empty) — nothing to enumerate")
        return 0
    print(f"[dp_enum] surface: {json.dumps(surface)}\n"
          f"[dp_enum] enumerating factors: {factors}", file=sys.stderr)

    variants = [base]
    for n in factors:
        v = Variant(factor=n, sg_path=out_dir / f"sg_dp{n}.json")
        try:
            redump(triton_opt, pre_modulo, n, v.sg_path,
                   out_dir / f"ddg_dp{n}.json", args.redump_timeout)
            v.applied = _applied_factor(
                json.loads(v.sg_path.read_text())) == n
            if not v.applied:
                # Legality said yes but the planner didn't tag: surface it
                # rather than silently benchmarking a baseline clone.
                v.error = "factor not applied by the pass (applied_n != n)"
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            v.error = str(e)
        variants.append(v)

    if not cpsat_model._HAVE_ORTOOLS:
        print("[dp_enum] ortools missing — costs come from the committed "
              "schedules (no buffering solve)", file=sys.stderr)
    for v in variants:
        if v.error is None:
            try:
                solve_variant(v, args.time_limit)
            except Exception as e:  # noqa: BLE001 — one bad variant must not kill the sweep
                v.error = f"solve/emit failed: {type(e).__name__}: {e}"

    # Report. Model cost is lexicographic (II, prologue depth, -buffering);
    # a pre-filter in enumerate mode, the deciding objective in --joint.
    print(f"\n== dp_enum {case_dir.name} ==")
    print(f"{'factor':<8}{'II per loop':<24}{'score':<16}{'emit':<7}{'note'}")
    for v in variants:
        ii = str(v.IIs) if v.IIs else "-"
        emit = "OK" if v.emit_ok else ("-" if v.error else
                                       f"FAIL {v.markers[:1]}")
        note = v.error or (v.opt_path.name if v.opt_path else "")
        print(f"{v.factor:<8}{ii:<24}{str(v.score):<16}{emit:<7}{note}")

    viable = [v for v in variants if v.viable]
    chosen_factor = None
    if args.joint and viable:
        chosen_factor = cpsat_model.select_dp_factor(
            [(v.factor, v.score) for v in viable])
    summary = {
        "case": case_dir.name,
        "surface": surface,
        "joint_pick": chosen_factor,
        "variants": [{
            "factor": v.factor, "cost": v.cost, "score": v.score,
            "IIs": v.IIs, "emit_ok": v.emit_ok, "markers": v.markers,
            "error": v.error,
            "schedule_graph": str(v.opt_path or v.sg_path or ""),
        } for v in variants],
    }
    (out_dir / "dp_enum_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"[dp_enum] summary -> {out_dir / 'dp_enum_summary.json'}",
          file=sys.stderr)

    if args.joint:
        if chosen_factor is None:
            print(f"[dp_enum] {case_dir.name}: joint — no viable variant")
            return 0
        best = next(v for v in viable if v.factor == chosen_factor)
        if chosen_factor == 1:
            print(f"[dp_enum] joint pick: n=1 (baseline) — the de-biased "
                  f"model finds no partitioned variant worth shipping; "
                  f"nothing to A/B")
            return 0
        print(f"[dp_enum] joint pick: n={chosen_factor} ({best.opt_path}) — "
              f"selected over baseline on de-biased model cost; the GPU A/B "
              f"gate remains the no-regression authority")
    else:
        cands = sorted((v for v in viable if v.factor > 1),
                       key=lambda v: v.cost)
        if not cands:
            print(f"[dp_enum] {case_dir.name}: no viable partitioned variant "
                  f"(all failed re-dump, solve, or emit-check)")
            return 0
        best = cands[0]
        print(f"[dp_enum] best partitioned variant by model cost: "
              f"n={best.factor} ({best.opt_path})")
        if (base.cost is not None and best.cost is not None
                and best.cost > base.cost):
            print(f"[dp_enum] note: n={best.factor} model cost > baseline — "
                  f"GPU A/B decides")

    if not args.ab:
        print(f"[dp_enum] model cost is a pre-filter — run the GPU gate:\n"
              f"  python -m twill_solver.testing.ab --case {case_dir.name} "
              f"--baseline {committed_sg} --candidate {best.opt_path}")
        return 0

    # KNOWN LIMIT (v1): a partitioned epilogue stores per-group (m_size, BN)
    # tiles, so the kernel needs a c descriptor whose block is (m_size, BN) —
    # but bench_spec.py builds descriptors statically for the case's baseline
    # tile config. Until bench_specs derive descriptor blocks from the
    # generated module, the A/B worker will launch the candidate with
    # baseline-shaped descriptors and (correctly) REJECT it at compile or
    # correctness. Warn instead of silently burning GPU time.
    print(f"[dp_enum] WARNING: candidate n={best.factor} changes the epilogue "
          f"store block to (m_size, BN); {case_dir.name}/bench_spec.py builds "
          f"static baseline-shaped descriptors, so this A/B will reject the "
          f"candidate for launcher reasons, not kernel quality. Parameterize "
          f"bench_spec descriptor blocks before trusting a REJECT here.")

    from .testing import ab
    ab_args = ["--case", case_dir.name, "--baseline", str(committed_sg),
               "--candidate", str(best.opt_path), "--eps", str(args.eps)]
    if args.ship:
        ab_args += ["--ship", args.ship]
    return ab.main(ab_args)


if __name__ == "__main__":
    sys.exit(main())
