# JOS Upgrade Implementation Design Document Set — Overview and Index

> Status: v1 design draft (2026-07-30). Design only, not yet implemented.
> Location: this document set is versioned alongside the code on the `export-D114262310`
> branch (D114262310 "joint solver upgrade", based on main + the #2328 solver/codegen
> boundary-alignment commit; the paper_joint_solver package is already in-tree);
> worktree `/projects/kzhou6/hwu27/triton-beta-4-upgrade-wt`.
> Decision record: **Option 2 finalized** — extend with the paper_joint_solver package
> as the home base (upgrade work proceeds on this branch lineage; the reference branch
> at design time was `hwu27/paper-joint-solver`);
> the main-repo ModuloScheduling (C++) receives only three surgical patches, each as its
> own independent PR;
> we do **not** create the third copy of "open a separate branch and manually extract
> code from both sides" (the two systems already interoperate via the JSON protocol,
> fusion is unnecessary, and a copy would inevitably diverge).
> Upstream documents (outside the repo, cluster paths): `/projects/kzhou6/hwu27/jos-upgrade-plan.md` (workflow-level plan),
> `/projects/kzhou6/hwu27/repro-plan-b200-tlx.md` (replication status §1–24).
> Convention: the paper's system is referred to exclusively as JOS (arXiv:2512.18134); its original codename is not used.

## 1. Architecture and Seams

Existing pipeline (kept unchanged):

```
ModuloSchedulePass (C++, main repo)
  └─ TRITON_MODULO_DUMP_DDG → ddg.json ───────────────┐
paper_joint_solver (Python, solver branch)             │ ←—— seam = JSON protocol
  normalize(SCIP U=300) → modulo ILP(CBC) →           │
  joint SMT(Yices2) → Algorithm 1 + W minimization     │
  └─ solution.json → graph_writer → schedule_graph.json┘
Realization: emitter / SKC binding / Phase B CuTe binding → B200
```

All four workflows are **offline tools built around the seam**, going into four new
subpackages of `paper_joint_solver` (`calib/`, `enumerate/`, `oracle/`, `audit/`),
isolated behind CLI switches that default to off; the paper-faithful 1:1 path stays
byte-for-byte identical (pinned by a CI regression gate). C++ receives only three
pinpoint write-backs, and only after their value has been proven.

The post-upgrade optimality claim (replacing the paper's unconditional form):
**"Within the (I,L,W) space, relative to the verified fact set Φ and the calibrated
intervals [c⁻,c⁺], s\* is robustly feasible and measured-best within the near-optimal
frontier; it is within x% of the certified upper bound inside the applicable domain;
per-parameter stability intervals are given in the sensitivity health report."**

## 2. Document Index

| Document | Contents | New subpackage | CLI switches | C++ |
|---|---|---|---|---|
| [10-w1-calibration-robust.md](10-w1-calibration-robust.md) | Microbenchmark three-state intervals, DiffTune fitting, robust solving, Hall-Posner health report | `calib/` | `--calib` `--robust` `--sensitivity` | P1, deferred |
| [20-w2-topk-measured-ranking.md](20-w2-topk-measured-ranking.md) | samewarp signature enumeration, dual-candidate-source measured ranking, Kendall τ, distillation registry | `enumerate/` | `--enumerate-topk` | none |
| [30-w3-ptxas-oracle-loop.md](30-w3-ptxas-oracle-loop.md) | O_ptxas/O_ncu, L0 per-WG budget bisection, closed-loop protocol, bwd golden case | `oracle/` | `--oracle-loop` `--reg-budgets` | P2, deferred |
| [40-w4-audit-certificates-sass.md](40-w4-audit-certificates-sass.md) | Two-tier audit certificates (roofline + DDG upper bound), SASS preservation audit, E-C gate | `audit/` | standalone CLI | depends on P3 (temporary fallback available) |
| [50-cpp-surgical-patches.md](50-cpp-surgical-patches.md) | P1 calibration-override entry point / P2 per-WG budget knob / P3 ddg-0.2 added fields | — | env switches; not enabled = bit-identical | yes (~280/70/95 lines) |

## 3. Key Corrections Produced During Design (relative to jos-upgrade-plan.md)

Design refinement uncovered and corrected four problems in the plan draft, now baked
into the individual documents:

1. **W1's Γ robust encoding had the wrong quantifier direction**: the plan draft's
   direct δ_e boolean encoding lets the solver choose δ=0 itself as an escape hatch
   (an existential quantifier, where a universal is needed). Fix: lazy adversarial cuts
   (solve → the adversary pins the Γ worst-case edges → add cuts and re-solve,
   monotonically convergent); the d⁺/d⁻ dual-query is the primary path.
2. **The W3 golden case needs starting-point calibration**: in the replication, under
   the v6 default budgets the free optimum is already 4 groups (not the paper's 2).
   A discriminating probe must first find a relaxed budget B0 under which the 2-group
   analogue is SAT, so the paper's incident can be re-enacted and the closed loop shown
   to repair it automatically. The bwd realization goes through the SKC binding to
   bypass the jos_bwd emitter gap (L0_smem_3).
3. **The W4 certificate splits into two tiers**: the DDG upper bound is sound only for
   TLX/JOS-family bars with the same dataflow / same tile / same CTA mode; foreign bars
   such as cuDNN/FA4 can only use the Tier-1 machine roofline (TC peak + measured
   bandwidth). This patches the applicability hole in the plan draft's "one ceiling
   line drawn across the whole chart." ResMII does not trust the value exported in
   ddg.json; it recomputes in the occupancy domain itself (sidestepping the
   selfLatency/occupancy two-domain inconsistency in the main repo's modulo
   reservation table).
4. **P1's coefficient-consistency ruling**: TMA per_kb is shared as a single copy
   between the latency law and the occupancy formula; overriding them separately is
   forbidden (otherwise the model contradicts itself internally).

## 4. Milestone Mapping (continuing jos-upgrade-plan.md §5)

- **P0 (~1 week)**: W4 E-A audit ceiling table (Tier-1 all bars + Tier-2 family bars,
  with byte volumes temporarily back-solved) + the W3-L0 closed-loop golden case
  (including starting-point calibration). Zero C++.
- **P1 (~2–3 weeks)**: W1 in full (microbenchmarks → calib-0.1 → fitting gate →
  dual robust → health report) + the W2 enumerator plus the two measured campaigns
  (fwd/bwd). Zero C++.
- **P2 (as needed)**: the three C++ patches in order P3 (zero dependencies, can start
  immediately) → P1 (waits for the W1 calib to be finalized) → P2 (waits for the W3
  golden case to pass); W3-L1 window distillation (only if L0 is too tight); W4 E-B
  SASS audit + E-C round-trip gate.

## 5. Branches and Engineering Discipline

- All Python lands on `hwu27/paper-joint-solver`; each C++ patch gets its own small
  branch off main as an independent PR, inherited by the two worktrees via rebase.
- C++ changes must be recompiled via `pip install -e . --no-build-isolation` +
  `pre-commit run --all`; after a rebase, reinstall editable triton (beware the
  `_C/.tmp*` empty-shell hijack pitfall, see the closing section of document 50).
- New env vars are registered in the `include/triton/Tools/Sys/GetEnv.hpp` allowlist;
  P1/P2 participate in the compilation cache key, P3 does not.
- Fidelity regression gate: with all switches off, the solver results for the three
  fixtures (subtiled fwd / fwd / bwd) are byte-for-byte identical to the v6 final
  version.
- Measurement window: exclusive dgx003 + steady-state warmup + accompanying clock
  logging + killgpu.sh discipline; all solving runs serially + in separate processes +
  `ulimit -v 150G`.

## 6. Open Items (to be decided before implementation)

1. The scope of the W2 SKC protocol_overrides extension (the most elastic item in the
   out-of-family candidate coverage vs. effort estimate trade-off).
2. If W3 starting-point calibration cannot find a reasonable B0 that makes 2 groups
   SAT (the replication graph differing too much structurally from the paper's graph),
   the golden case is downgraded to the weaker statement "the closed loop converges
   from an arbitrary spill starting point to a spill-free structure."
3. Whether the W4 Tier-2 certificate automatically degrades to an interval certificate
   ([UB⁻, UB⁺] twin lines) along with the W1 intervals — the design documents leave a
   hook for this; the default is to ship the point-value version first.
4. ~~Whether this document set should move into the repo to be versioned with the
   code~~ — done (2026-07-30): moved into this branch at
   `third_party/tlx/tools/paper_joint_solver/docs/upgrade/`; the original cluster
   location `/projects/kzhou6/hwu27/jos-upgrade-docs/` retains a forwarding pointer.
