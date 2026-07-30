# W1 Implementation Design: Measured Coefficients + Interval Calibration + Robust Solving

> Status: **v1 design draft (2026-07-30, design only, not yet implemented)**
> Landing point: branch `hwu27/paper-joint-solver` (worktree `/projects/kzhou6/hwu27/triton-beta-3-paper-solver-wt`), package `third_party/tlx/tools/paper_joint_solver/`
> Related: `jos-upgrade-plan.md` §W1 (scheme-level design); main-repo ModuloScheduling comparison conclusions (half of the W1 material already exists: NVLatencyModel point values are already microbenchmark-calibrated; the three missing pieces are **intervals / automated fitting / sensitivity**); overview in `00-overview.md`
> Terminology: the paper's system is referred to exclusively as **JOS**.

## 1. Goals and Fidelity Discipline

Upgrade the cost coefficients in machine.py from "documentation / single-point microbenchmark estimates" to "three-state interval measurement + end-to-end fitting + robust solving + per-parameter stability certificates". **Not a single line of the paper-faithful 1:1 path changes**: all new functionality goes into the new subpackage `paper_joint_solver/calib/`, enabled by CLI switches that default to off; with all switches off, the solve results for the three fixtures must be byte-for-byte identical to the v6 final version (regression gate, goes into CI).

## 2. Module Layout and CLI

```
paper_joint_solver/calib/
  schema.py        # machine_calib.json dataclass + validation (load_calib(path) -> Calib)
  intervals.py     # interval algebra; normalize propagation (propagate(calib, norm_result) -> NormIntervals)
  fit.py           # DiffTune-style inverse fitting (fit(dataset, priors, gates) -> FittedTheta)
  sensitivity.py   # Hall-Posner health report (stability_report(solution, smt_ctx, calib) -> Report)
  report.py        # health-report table / calibration report rendering (markdown + json dual format)
  bench/
    common.py      # tlx.clock64 timing primitives, three-state companion streams (idle / tc_loaded / smem_loaded), sample statistics
    bench_mma.py   # tcgen05 shape family: M128N128K128/K64 re-measurement + new BN=64 sub-block (M64 tile is limited
                   #   by this build's TMEM, "Only supported for scales" — work around via half-column-read semantics on 128-row tiles, recorded in the audit)
    bench_tma.py   # TMA load/store ± multicast (1-CTA vs 2-CTA cluster paired measurement)
    bench_tmem.py  # TMEM load/store: 128x128 full-column + 128x64 half-column read
    bench_channel.py # complete cross-warp-group SMEM round-trip chain store+arrive+wait+load (replaces the hand-set spill_cost value)
    bench_barrier.py # mbarrier/named individual items + actual per-barrier SMEM padding accounting (eliminates the ~17KB blind spot at Q=3)
    run_all.py     # orchestrates all benches → machine_calib.json (with source/date/clock companion metadata)
```

`__main__.py` gains new switches (all default off):
`--calib PATH` (load a calibration file to override machine.py built-in values); `--robust {off,dual,gamma}` (default off = the paper's original semantics); `--gamma N` (budget for gamma mode, defaults to automatic per §6.3); `--sensitivity` (run the health report after a successful solve). Subcommand `python -m paper_joint_solver.calib fit --bench-jsonl ...`.

## 3. machine_calib.json schema (calib-0.1)

```json
{ "schema": "calib-0.1", "gpu": "B200", "clock_mhz": 1965, "date": "...", 
  "measured": {
    "mma.tcgen05.m128n128k64.fp16": {
      "kind": "latency", "unit": "cycles", "point": 559,
      "states": {"idle": [551,566], "tc_loaded": [559,601], "smem_loaded": [552,580]},
      "interval": [551,601],        // = [min(p20 across states), max(p80 across states)]
      "prior": 559,                 // current NVLatencyModel value (traceable)
      "source": "calib/bench/bench_mma.py@v1", "n": 200 },
    "tma.load.multicast_delta.kb32": { "...": "2-CTA paired-measurement delta; the interval is allowed to straddle 0" }
  },
  "fitted": { "spill_roundtrip_base": {"value": 150, "interval": [138,171], "fit_run": "..."} } }
```

Key name = `op-class.shape-key.dtype`; `interval` semantics: **idle-state p20 as the lower bound, worst-interference-state p80 as the upper bound** (W4's c_lb is taken directly from each key's `states.idle[0]`). machine.py consumption rule: on a key hit the built-in value is overridden; on a miss the built-in value is kept and the solve log lists the uncalibrated keys.

## 4. Microbenchmark Inventory (prior reuse vs. new measurement)

| Item | Prior (current NVLatencyModel value) | Disposition |
|---|---|---|
| TMA load | 460+6·KB+240·min(insts−1,2) | Re-measure into three-state intervals; keep the law's functional form |
| TMA store | 130+52·KB | Same as above |
| MMA lat/occ | 900/559; MACs/4096 | Re-measure intervals + **add BN=64 sub-block shape** |
| TMEM load | 532@16K nonlinear (179@8K) | Re-measure + **add 128×64 half-column read** |
| TMA multicast | none | **New measurement** (2-CTA cluster paired measurement, root cause of the B2 residual) |
| Cross-warp-group round trip | 150 + 16 cyc/KB | Re-measure three states (current value was measured by cross_wg_handoff.py, the predecessor of bench_channel) |
| mbarrier/named | 45 / 30 | Re-measure + **add actual per-barrier SMEM padding accounting** |
| reduce/SFU/elementwise | priced by input-element + axis-direction pricing table | Copy directly as prior point values; approximate intervals with ± the fitting residual (not re-measured item by item) |

Three-state protocol (a PipeThreader lesson): the companion streams are generated by background warp groups inside the same kernel — tc_loaded = a back-to-back tcgen05 chain; smem_loaded = ld/st shared bandwidth sweep. Run discipline: exclusive use of the whole dgx003 machine (`8b200.sh`), warm up to steady state, ≥200 samples per item recording p20/50/80, `nvidia-smi` clock traces saved alongside.

## 5. DiffTune Fitting Loop (fit.py)

- **θ (~15 scalars)**: TMA base/perKB, MMA occupancy scaling, TMEM load scaling, spill base/slope, mbarrier, plus one global scaling factor each for SFU/CUDA. **Fitting is constrained within the measured intervals** (the intervals double as box bounds — preventing compensatory degenerate solutions).
- **Dataset**: replicate the existing bench JSONL — skc_default/jos/var_bn64/var_qk2 ×4 seqlen, E3 kv 3/2/1 and the split_P 96/64/32/0 gradient, the three Phase B quota candidates.
- **Forward model**: `T_pred(θ;cfg) = II_model(θ,cfg)·n_iter + prologue(θ,cfg)`, where II_model is recomputed purely analytically with machine.py+DDG (does not go through SMT; the proxy semantics are recorded in the audit).
- **Optimization**: `scipy.optimize.least_squares` on relative error, with integerization as the final step.
- **Gates** (θ is rejected if any fails): (a) replication of the known E3 ordering — kv 3→2→1 monotonic degradation, split_P monotonic; (b) leave-one-fixture-out cross-validation — fitting on the fwd family predicts the bwd family's ordering, Kendall τ threshold 0.7.

## 6. Interval Propagation and Robust Encoding

**6.1 normalize.py propagation (intervals.py)**: before clustering, switch to an **interval-union test** (ops may be clustered only if their intervals overlap, and the representative interval is the union; take the max of the 10% tolerance and the largest half-width — otherwise clustering swallows genuine uncertainty). The medians go through the U=300 ILP to obtain non-uniform scalings s_i=C′[i]/C[i]; intervals map as `[floor(lo·s_i), ceil(hi·s_i)]`, with the lower bound clamped ≥1; operators zeroed out by the 1/32 rule have their intervals pinned to [0,0] and included in the report.

**6.2 CBC side (modulo_ilp.py)**: directional endpoints — DEPENDENCE/spill latencies take d⁺ (underestimation → stalls cost more), CAPACITY occupancy takes the nominal median (overestimation → throughput thrown away for nothing).

**6.3 SMT side (joint_smt.py)**:
- `--robust dual` (main path): dual-query at the same (II,L). `SAT(d⁺)` ⇒ robust certificate (feasible on every machine within the intervals); `SAT(d⁻)∧UNSAT(d⁺)` ⇒ flagged "interval-sensitive", handed to the §7 health report + W2 measured adjudication. The skeleton formula is shared; only the latency constants are swapped.
- `--robust gamma` (advanced): **note — the plan draft's direct δ_e encoding is unsound** (under an existential encoding the solver will pick δ=0 to escape; the direction is inverted). Replaced with **lazy adversarial cuts**: solve(nominal) → the adversary linearly scans the fixed schedule and picks the Γ edges with the largest violations to pin to d⁺ → add the cuts and re-solve → converge (monotone, ≤|E| rounds, reusing the serial + 150G gate discipline). Γ default = the 90th percentile of the count of coefficients whose fitting residual exceeds the interval half-width (empirically expected 2–3).

## 7. Hall-Posner Sensitivity Health Report (sensitivity.py)

For the certified solution, per parameter c:
- **Feasibility stability interval**: fix (M*,A*) as the assignment; each constraint containing c has a slack that is a linear function of c, intersected in closed form — pure Python, zero solver calls.
- **Optimality stability interval**: re-issue the UNSAT query for II*−1 (or, at the same II, minimal L−1) with named assumptions; yices extracts the unsat core, and for the core constraints containing c, bisection probes how far c can be relaxed before the core breaks (one SMT call per probe, ~log(range)·|core| calls, reusing the incremental context).
- Output (report.py, markdown+json): `parameter | calibrated value | measured interval | feasibility stability interval | optimality stability interval | verdict (SAFE / measured interval out of bounds / certificate invalidated) | remediation (raise II / reduce depth / change partitioning)`.

## 8. Validation Experiments (revisiting the three incidents) and Overall Acceptance

- **V1 BN=64**: after adding the shape, the model's predicted difference for the BN64/BN128 binding must have the **same sign and same order of magnitude** as the measured −91 TFLOPS (currently it is 0).
- **V2 2-CTA inversion**: the multicast paired measurement must reproduce the fwd direction (fa4_1cta 1275 > stock 1047); if the coefficient interval straddles 0, the report rules "undecidable, handed off to W2 measurement" — which also counts as passing.
- **V3 spill pricing**: bench_channel measurements, after normalization, land within [1,4] units (currently hand-tuned to 2); the subtiled (66,150) SAT point does not regress.
- **Overall acceptance**: with `--calib`, the solver's predicted ordering over the skc variant family reaches Kendall τ ≥ 0.8; the default-off regression gate stays byte-for-byte identical.

## 9. Test Plan / Effort Estimate / Risks

**Tests**: schema load validation, interval algebra, cluster-union rule, synthetic θ recovery (fitting correctness), closed-form vs. brute-force sensitivity cross-check on a toy DDG, `--robust dual` integration on the three fixtures, default-off byte-level regression gate.
**Effort estimate**: new bench code ~3d, intervals/propagation ~1.5d, fitting ~2d, dual ~1d, gamma cuts ~2d, sensitivity ~2d, tests and reporting ~1.5d ≈ **2.5 weeks** (including 2 exclusive measurement windows).
**Risks**: measurement windows are scarce (run the idle state needed by P0 first; interference states can be backfilled later); the BN=64 micro-kernel is limited by this build's TMEM tile restriction (workaround in the bench_mma note); gamma cuts increase the number of solve rounds (reuse the operations discipline); the calibration set is entirely FA-family — the extrapolation domain is honestly flagged in the report.

## 10. Cross-Workstream Interfaces and Follow-up Pointers

- **→W4**: the `states.idle` lower bound is directly the source of the audit ceiling's c_lb (same file, no second conversion).
- **←W3**: cross-warp-group channel costs measured by ncu flow back into `fitted.spill_roundtrip_*`.
- **←W2**: the "systematically slow structure predicates" distilled from ranking.jsonl enter the fitting candidate-correction table.
- **C++ write-back (outside the scope of this document)**: after fitting validation, write back to the main-repo NVLatencyModel via `TRITON_MODULO_LATENCY_CALIB` (the JSON override entry point), so ddg.json benefits at the source — see `50-cpp-surgical-patches.md` P1.
