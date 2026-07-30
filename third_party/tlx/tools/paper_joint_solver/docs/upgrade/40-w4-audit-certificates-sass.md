# W4 Implementation Design: Audit Certificates (Lower Bounds → TFLOPS Ceiling Table) + SASS Preservation Audit

> Status: **v1 design draft (2026-07-30, design only, not yet implemented)**.
> Landing point: branch `hwu27/paper-joint-solver` (worktree `triton-beta-3-paper-solver-wt`),
> new subpackage `third_party/tlx/tools/paper_joint_solver/paper_joint_solver/audit/`.
> Discipline: **purely offline consumer tool** — reads only ddg.json / bench JSONL / cubin / ncu reports,
> modifies no solver module and no C++; zero contact with the paper-faithful 1:1 path.
> Related: master plan `/projects/kzhou6/hwu27/jos-upgrade-plan.md` §W4; coefficient sources in `10-w1-*.md`
> (machine_calib.json); ddg-0.2 field additions in `50-cpp-surgical-patches.md` P3.

## 1. Goals and Positioning

Upgrade the relative rankings of replication report §20 ("JOS 215 vs FA4 1047") to an absolute
scale: how much headroom each bar has left relative to a **certified performance ceiling** (the
audit-oracle usage from the PLDI 1996 Showdown). At the same time, produce a SASS preservation
report that quantitatively answers "how much of the solver's schedule order did ptxas preserve" —
directly completing the ncu/ptxas attribution to-do left over from §20. Explicitly out of scope:
full Telamon B&B (its benefit is already covered by this table plus W2's measured ordering, and it
would require a major OMT/enumerator architecture rework); SLOTHY-style full SASS solving
(Blackwell has no public ISA semantics/scheduling tables); OMT engine replacement (same reason).

## 2. Two-Tier Certificate Design (Key Decision)

A single DDG lower bound cannot cover external implementations such as cuDNN/FA4 (their
tiles/dataflows differ, so the DDG bound is not sound for them). The certificate therefore has
two tiers:

- **Tier-1 machine roofline**: uses only implementation-independent quantities — TC peak
  (MACs/cycle/SM, per dtype) and measured achievable DRAM bandwidth plus the algorithm's minimum
  byte traffic. Sound for **any implementation of the same mathematical task**.
  External bars such as cuDNN/FA4/Triton are compared against Tier-1 only.
- **Tier-2 DDG ceiling**: layers the ResMII/RecMII/Issue/TMEM terms derived from this
  project's fixture DDGs on top of Tier-1. Sound only for implementations with the **same
  dataflow, same tile geometry, and same CTA mode** (the TLX-*/JOS/SKC family of bars).

Each certificate carries `assumptions[]` (algorithm, tile, dtype, CTA mode, clock) and a `tier` field;
bar-to-certificate applicability is decided by `domain` matching — on a mismatch, that cell is
marked "certificate not applicable". **2-CTA gets its own certificate**:
recompute BW_lb and the TMA-side ResMII_lb with B-operand traffic halved (§24 has empirically shown
that 2-CTA is a net loss on fwd and +7% on bwd, so both modes must have a table to consult).

## 3. E-A: Audit Ceiling Table (`audit/ceiling.py` + `audit/bounds.py`)

### 3.1 Inputs

1. `ddg.json` (ddg-0.1, three fixtures already in place): node pipeline/latency/self_latency/
   occupancy/min_warps, edge latency/distance, exported min_ii/res_mii/rec_mii. **Always
   compute in the raw cycle domain, bypassing the U=300 normalization distortion**.
2. The **lower-bound column** of `machine_calib.json` (a W1 deliverable) — the best values from
   the idle state of the three-state measurement (idle / TC-loaded / SMEM-loaded). Until W1
   completes, fall back to the current point values as a stand-in; tables and figure captions
   must be labeled "**provisional lower bound (uncalibrated)**", with field
   `calib_provenance: "provisional-point"`.
3. bench JSONL (already available from §20/§24): median TFLOPS + accompanying clock
   (measured records of 1950–1965 MHz; use the accompanying median clock at the time that bar
   was measured, not the nominal value).

### 3.2 Lower-Bound Terms (`bounds.py`; all return `fractions.Fraction`, never rounded up)

`II_lb = max(ResMII_lb, RecMII_lb, BW_lb, Issue_lb, TMEM_lb)`:

- **ResMII_lb**: per pipeline (TMA/TC/CUDA/SFU/TMEM — following machine.py's convention of
  TMEM as an independent pipeline), sum the occupancy lower bounds and take the maximum.
  **Must be recomputed independently; do not trust the res_mii field in ddg.json**: in the main
  repo, modulo reservation table placement reserves only `max(selfLatency,1)` slots while ResMII
  is computed from occupancy
  (`ModuloReservationTable.cpp:60-64` vs `DataDependenceGraph.cpp:331-347`) — the two domains are
  inconsistent; this tool uniformly recomputes in the occupancy domain and cross-checks against
  the exported value, emitting
  `WARN res_mii mismatch (recomputed X vs exported Y)` and adopting the recomputed value on disagreement.
- **RecMII_lb**: the rational maximum cycle ratio max(Σlat/Σδ). The implementation follows the
  main repo's "forward longest path + single back edge" enumeration (Floyd-Warshall + each
  distance>0 edge), but **without the ceil**. Cycles with multiple back edges may be missed by
  the enumeration → the lower bound can only get looser, so soundness is unharmed (this
  relaxation is noted in the docs and the report). Edge weights use c_lb.
- **BW_lb** = DRAM bytes per iteration / achievable bytes per cycle. Achievable bandwidth uses
  measured values (the W1 microbenchmark or an independent STREAM-style measurement), converted
  via `bytes/cycle = BW_GB/s ÷ f_clk_GHz`. Byte-count source:
  ddg-0.1 has no bytes field, so the **interim scheme** back-solves `KB = occupancy/6` from TMA
  load nodes (only when occupancy>30; for small tiles (≤5KB) with occupancy==30 the back-solve is
  not unique — count them at the 5KB upper bound and record this in assumptions), and back-solves
  the store side at 52 cyc/KB; bwd's dQ atomic-reduction traffic is added in per the shape
  convention of descriptor_reduce nodes. The **formal scheme** points to `50-cpp-surgical-patches.md`
  P3 (ddg-0.2 adds per-node bytes/MACs fields); once that lands, this term switches over automatically.
- **Issue_lb** = estimated warp instructions per iteration / 4 (at most 4 warp issues per SM per
  cycle). Instruction counts are estimated under **the most optimistic vectorization**
  (tile elements ÷ (32 lanes × widest packing); an async op counts as 1 issue
  plus its accompanying barrier instructions); the optimistic estimate keeps the lower bound sound.
- **TMEM_lb** = Σ_i (cols_i × minimum residency cycles_i) / 512 columns. Minimum residency = that
  accumulator's MMA occupancy + the duration of the corresponding tmem_load (an accumulator must
  live at least until it is read out).

The binding term (the argmax) is written into the certificate's `binding_term` for attribution
("this shape is limited by BW rather than TC").

### 3.3 TFLOPS Conversion and Presentation

`UB_TFLOPS = FLOPs_per_iter × f_clk / II_lb × N_SM`. FLOPs_per_iter follows the harness convention
(fwd 4·B·H·S²·D, bwd 10·B·H·S²·D, prorated to a single iteration on a single CTA); N_SM=148,
assuming full-wave steady state
(holds since the B=4/H=32 grid far exceeds 148; wave quantization and prologue/epilogue
amortization are recorded as assumptions — the steady-state approximation can only raise the UB
and does not break soundness). Outputs: (a) a RESULTS.md table
`bar | tier | measured | UB | %UB | binding_term`; (b) a plotting hook
`ceiling.plot_hook(ax, shape, certs)` for the Figure 8/11 replication scripts to overlay ceiling
lines (Tier-1 solid, Tier-2 dashed; out-of-domain bars are not drawn).

### 3.4 Soundness Argument (a section the doc must contain)

A lower bound is sound iff "any in-domain implementation must pay that cost". The only things
that can breach it are **work-reducing** speedups outside the model: 2-CTA operand sharing
(→ build a separate 2-CTA certificate), algorithmic restructuring (deferred rescaling
and the like, which change FLOPs/byte counts → excluded via the domain declaration), and tile
changes (the 80×128 kind → excluded via the domain declaration). Over-loose lower bounds
(RecMII missing cycles, optimistic Issue estimates) only weaken the conclusion (headroom appears
larger) and never produce a false certificate. Reverse acceptance check:
**any bar measuring above the UB of its applicable domain = a bug in the certificate or the
domain declaration, and must be fixed** — this is itself an audit of the cost model.

## 4. E-B: SASS Preservation Audit (`audit/sass_audit.py` + `audit/anchors.py`)

Subjects: one each of `skc_jos` (TLX backend) and the Phase B solver binding `fa4_1cta` (CuTe backend).

Pipeline: disassemble the cubin with `cuobjdump -sass / nvdisasm` → extract anchor sequences per
a **data-driven anchor table**
(`anchors.toml`: SASS mnemonic regexes for tcgen05-MMA, TMA bulk copy, `MUFU.EX2`, and mbarrier
arrive/wait; the concrete Blackwell mnemonics are confirmed at first disassembly and then entered
into the table — nothing is hardcoded in code) → map back to DDG nodes via the SKC skeleton's
structural correspondence (each warp-group partition's operator issue order is statically known
in the skeleton: the k-th EX2 tile ↔ the k-th cluster of MUFU.EX2). Four metrics:

1. **Order-preservation Kendall τ**: solver schedule order vs SASS static order, **computed
   separately within each warp-group partition**
   (each partition is an independent instruction stream; cross-partition static order is meaningless);
2. **Spill evidence**: `ptxas -v` regs/spill bytes + LDL/STL counts inside the SASS loop body
   (distinguished from the prologue);
3. **Skew survival check**: whether the QK(i) anchor still precedes PV(i−1) in the MMA
   partition's instruction stream (i.e., whether §21's +70 TFLOPS gain from skew=1
   was eaten by ptxas reordering);
4. **ncu stall attribution**: `long_scoreboard`/`barrier` stall shares mapped onto anchor
   neighborhoods via source-correlation.

Acceptance: anchor mapping rate ≥90%; on breakage (inlining/unrolling), fall back to loop-body
aggregate statistics and disclose this in the report.

## 5. E-C: Round-Trip Gate (`audit/roundtrip_gate.py`, optional, 0.5 day)

`skc_default` cubin → CuAssembler disassemble → reassemble → on-device correctness. Expected to
**fail** on sm_100a (official CuAssembler support stops at Ampere); record the full signature
either way (tool versions, error messages, diff). A failure formally seals off the CuAsmRL-style
SASS optimization route and writes it into the not-doing list.

## 6. Module Layout / API / CLI

```
paper_joint_solver/audit/
  __init__.py  __main__.py          # subcommands: ceiling | sass | roundtrip
  bounds.py    # res_mii_lb/rec_mii_lb/bw_lb/issue_lb/tmem_lb(ddg, calib, domain) -> Fraction
  ceiling.py   # compute(ddg, calib, domain) -> CeilingCert; render_table(); plot_hook()
  sass_audit.py# extract_anchors(); map_to_ddg(); tau_per_partition(); skew_survival(); report()
  anchors.toml # SASS anchor regex table (data-driven)
  roundtrip_gate.py
tests/test_audit_ceiling.py  tests/test_audit_sass.py
```

CLI examples:
`python -m paper_joint_solver.audit ceiling --ddg fixtures/subtiled/ddg.json --calib machine_calib.json --bench bench/results_fwd.json --domain 1cta --out ceiling_fwd.md`
`python -m paper_joint_solver.audit sass --cubin skc_jos.cubin --solution solution.json --binding skc_binding.json --ncu-rep skc_jos.ncu-rep --out sass_skc_jos.md`

## 7. Test Plan and Acceptance

- Unit tests: **lower-bound monotonicity** (lowering any c_lb → II_lb does not rise; tested
  per-term and at the max layer); a synthetic 5-node small graph checked term-by-term against
  hand calculations (including one distance=2 back edge to validate the rational RecMII); BW
  back-solving including the occupancy==30 ambiguity branch; τ and skew checks tested with
  synthetic anchor sequences.
- E-A acceptance: full coverage of the nine Figure 8 bars + the five Figure 11 bars × 4 shapes;
  no bar exceeds the UB of its applicable domain;
  every certificate contains binding_term and assumptions.
- E-B acceptance: two reports (skc_jos / Phase B fa4_1cta), with the mapping rate on target or
  the fallback disclosed.

## 8. Effort Estimate and Risks

E-A 2–3 days (bounds + table + plotting hook); E-B 2–3 days (anchor table confirmation + mapping
script); E-C 0.5 day. Risks: (1) Blackwell SASS mnemonics unknown → data-driven anchor table +
manual confirmation on first use; (2) BW byte back-solve
ambiguity → recorded explicitly in assumptions, eliminated once P3 lands; (3) ncu
source-correlation may lack line numbers on TLX artifacts → fall back to range-replay
partition-level aggregation; (4) over-loose lower bounds inflate apparent headroom →
report as-is; soundness unaffected.
