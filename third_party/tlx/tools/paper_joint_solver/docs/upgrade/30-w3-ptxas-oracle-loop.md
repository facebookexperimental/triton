# W3 Implementation Design: ptxas as an Oracle in the Loop (SMTO Semantics + ISDC Distillation Attachment Point)

> Status: v1 design draft (2026-07-30). **Design only, not implemented.**
> Landing spot: branch `hwu27/paper-joint-solver` (worktree `triton-beta-3-paper-solver-wt`);
> all new code goes into the `third_party/tlx/tools/paper_joint_solver/paper_joint_solver/oracle/` subpackage.
> Fidelity discipline: the paper-faithful 1:1 solving path (Figure 4/5/6 constraints, Algorithm 1, default CLI behavior) does not change by a single line;
> every feature of this workflow hangs behind the default-off `--oracle-loop` / `--reg-budgets` switches.
> When `--reg-budgets` is passed a single value uniform across all groups, the solution must be bit-for-bit identical to the existing `--reg-budget` (pinned by a regression test).
> **Zero C++ changes throughout.** Porting the main repo's ModuloScheduling budget-shrinking loop to C++ is a later optional item; see
> `50-cpp-surgical-patches.md` P2; the closed loop in this document is implemented entirely as a Python driver script.
> Related: overall plan `/projects/kzhou6/hwu27/jos-upgrade-plan.md` §W3; W1 doc (spillcost feedback),
> W2 doc (failure-signature criterion).

## 1. Goals and Semantics

Mechanize the manual intervention in the paper's (JOS) bwd narrative — "default budget → 2-warp-group
solution → massive ptxas spills → manually lower the register budget and re-run → 3-group solution
succeeds" — into an automated closed loop, and recast the optimality claim in SMTO's
consistent-with-oracles semantics:

> M\* is optimal at (I, L, W) relative to the **verified fact set Φ**. Φ is listed item by item (e.g., "ptxas 13.2 reports
> spill_store=B_k bytes for solution s_k"); the certificate is conditioned on Φ and the listed toolchain versions.

Loop shape = ISDC attachment point: run the real downstream tool (ptxas) once, **distill** the result into a small
number of constraints fed back for re-solving, iterate ≤3 rounds; no per-candidate full-compilation search, and no
a-priori conservative margins.

## 2. Pipeline and Data Flow

```
ddg.json ──► python -m paper_joint_solver (joint SMT, per-WG budgets) ──► solution.json
                     ▲                                                    │
                     │ distilled constraints (L0 budget shrink / L1 window constraints)  │ bind
                     │                                                    ▼
              oracle/distill.py ◄── Φ (facts.jsonl) ◄── oracle/ptxas.py ◄── SKC-bound kernel
                                                        oracle/ncu.py       (bwd main path)
                                                                            or emitter TLX
```

Binding-path choice: **the bwd golden case goes through SKC binding** (`skc/skeleton_bwd.py` + `binder.py`), bypassing
the known jos_bwd emitter gap (the `L0_smem_3` barrier is referenced but never allocated; replication plan §20). Decision
point: if oracle data via the emitter path is needed later (e.g., arbitrary partition topologies), fix that gap first;
this is an independent work item and does not block this workflow.

## 3. Oracle Interfaces (`oracle/ptxas.py`, `oracle/ncu.py`, `oracle/cache.py`)

### 3.1 O_ptxas (compile-time, primary oracle)

```python
@dataclass(frozen=True)
class PtxasFacts:
    reg_per_thread: dict[str, int]        # SASS entry/section name → register count
    spill_store_bytes: int                # ptxas -v "bytes spill stores" total
    spill_load_bytes: int
    stack_frame_bytes: int
    per_section_ldl_stl: dict[str, tuple[int, int]] | None  # per-section LDL/STL counts (§5.1, may be absent)
    ptxas_version: str                    # full `ptxas --version` string
    raw_log_path: str                     # archived raw log

def query_ptxas(solution_json: Path, binding: BindingSpec,
                cache: FactCache) -> PtxasFacts
```

Implementation: binding produces the kernel source → set `TRITON_DUMP_PTXAS_LOG=1` (or directly capture `ptxas -v` stderr)
to trigger one TLX→Triton→PTX→cubin compilation → regex parsing. Query cost 30–120 s each (ptxas itself takes only seconds;
the bulk is the Triton frontend). JOS is positioned as an offline tool (solving already takes tens of seconds to minutes),
so one query per round is affordable. ptxas is **deterministic** for identical input → cache everything.

`per_section_ldl_stl` is obtained by splitting `cuobjdump -sass` / `nvdisasm` output into sections by branch label, then
grepping the `LDL`/`STL` mnemonics and counting — this relies only on mnemonic text, not on Blackwell SASS encoding
semantics (§9 risk).

### 3.2 O_ncu (runtime, corroborating oracle)

```python
@dataclass(frozen=True)
class NcuFacts:
    achieved_occupancy: float
    local_mem_sectors: int          # l1tex local load+store sectors
    long_scoreboard_pct: float      # corroboration for stall attribution

def query_ncu(runner_cmd: list[str], kernel_regex: str,
              cache: FactCache) -> NcuFacts
```

Single purpose: **screening out false positives** — the case where ptxas reports spills but the local-memory traffic is
absorbed by L1/L2 and measurements show no harm (the L2 tier, §5.3). O_ncu requires a GPU and runs only once, on the
final solution after L0 converges.

### 3.3 Fact Cache Φ (`oracle/cache.py`)

- key = `sha256(canonical(schedule_graph.json or solution.json)) + ptxas_version`.
- Storage: `oracle_runs/<case>/facts.jsonl`, **append-only, grow-only**; when the ptxas version changes, old entries
  are not deleted — they are marked `stale=true` and a re-query is forced.
- API: `FactCache.get(key) -> Facts | None`, `FactCache.put(key, facts)`.

## 4. Per-WG Register Budgets (the solver-side prerequisite for L0)

Status quo: the REGISTERLIMIT constraint in `joint_smt.py` is already emitted one per `∀t,w`
(`Σ_v,i live[v,i,t]·opw[v,w]·regs(v) ≤ reg_limit()`); only the right-hand side is a single global scalar
(CLI `--reg-budget`). The change (inside the upgrade submodule, not the paper path):

1. `machine.py`: `reg_limit()` → `reg_limit(w: int) -> int`, backed internally by a
   `dict[int, int]`; any w not specified falls back to the global default (the paper's nominal value, converted per the
   num_warps convention of replication §19: TMA/TC-only group 1 warp → the 24-regs tier, groups containing CUDA/SFU 4 warps → the 152 tier).
2. CLI: `--reg-budgets "0:152,1:24,2:152,3:152"` (group id : per-thread budget). **Without the flag, behavior is fully
   identical to today**; passing a single `default:R` is equivalent to the old `--reg-budget R`.
3. `search.py` unchanged: after tightening, UNSAT at the same (I,L) naturally follows Algorithm 1's original semantics (L increment → I increment),
   together with the existing W minimization — this is precisely the mechanism of structural migration (2 groups → 3 groups); no new search primitives.

## 5. Three Distillation Tiers (`oracle/distill.py`, ordered by ROI)

### 5.1 L0: Per-Group Budget Bisection (core deliverable, ~3 days)

Signal: `spill_store_bytes + spill_load_bytes > ε` (ε=0, or relaxed after L2 corroboration).
Attribution: prefer locating the offending warp group via the per-section SASS LDL/STL counts of §3.1 (the
warp-specialize branch structure of SKC/emitter-generated code naturally segments each group's code in the SASS;
the group-name↔branch mapping is registered by the binder);
**if attribution is impossible (counts missing or scattered) → fall back to shrinking all compute groups uniformly**.

Shrink strategy (estimate first, then bisect):

```
First shrink: R_w' = min(R_w, reg_per_thread_w) − max(8, ceil(spill_bytes / threads_w / 4))
Afterwards: spill>0 → upper bound=R_w', keep lowering; spill=0 and first round → optional upward probe (one step up to verify no regression)
Step-size floor 8 regs; lower-bound guard: R_w' ≥ 24 (conventional minimum tier); if breached, abandon the shrink and mark the group "cannot tighten further"
```

This mechanizes the paper's manual `TRITON_MODULO_REG_BUDGET` budget lowering, and does it **per group independently**
(less collateral damage than a global lowering). The Phase B quota-derivation history (regs full-sum 1060 mis-accounting
→ residency filtering + per-cycle peak → 198/thread landing in the neighborhood of the expert value) proves that the
divergence between "model bookkeeping vs ptxas reality" is systematic and directionally predictable — L0 does not try to
fix the bookkeeping (that is W1's job); it only brackets with the black-box signal.

### 5.2 L1: Windowed Localization (enabled only when L0 over-tightens, ~2 weeks)

Trigger condition: L0's whole-group shrink declares UNSAT even a solution that W2 measurements have proven good (over-tightening).
Method: `nvdisasm --print-life-ranges` extracts live registers per instruction → use tcgen05/mbarrier/TMA
instructions as **anchors** to map the SASS order back onto the schedule timeline (the operator↔protocol-slot
correspondence registered by the binder) → locate the over-limit window [t1,t2] → distill into a windowed constraint:
`∀t∈[t1,t2]: Σ live·opw[·,w]·regs ≤ R_w − Δ`, or forbid co-location of specific operator pairs
(`¬samewarp(u,v)`). Phase B's mapping-chain experience: feasible, but the bookkeeping details are highly error-prone —
hence L1 is an optional fallback tier, and L0 does not depend on it.

### 5.3 L2: ncu Corroboration (~1 day)

Run O_ncu once on the L0-converged final solution: `local_mem_sectors ≈ 0` confirms the spills are truly eliminated; if
ptxas reports spills but sectors are low and TFLOPS does not drop, mark it a "false positive", relax ε to that spill
level, and record it in Φ.

## 6. Closed-Loop Protocol (`oracle/loop.py`)

```
budgets = starting budgets B0 (§7 starting-point calibration)
Φ = load FactCache
for round in 1..3:                              # ISDC experience: converges in 2–3 rounds
    sol = solve(ddg, reg_budgets=budgets, extra_constraints=distilled(Φ))
    #   UNSAT at the same (I,L) → search.py auto L/I increment + W minimization (where structural migration happens)
    kern = bind_skc(sol)                        # bwd main path; emitter as fallback
    facts = query_ptxas(sol, kern, Φ)
    evidence.append({round, sol_hash, facts summary, new_constraints})   # JSONL evidence chain
    if spill ≤ ε: break
    budgets = distill_l0(facts, budgets)        # only tighten, never relax
else: hand off to a human (with the full evidence chain attached)
Run O_ncu corroboration + measured performance on the final solution
```

Discipline: **constraints only tighten, never relax** (Φ monotone → no oscillation); each round's evidence chain is persisted to
`oracle_runs/<case>/evidence.jsonl`, schema:
`{round, solution_hash, ptxas_version, facts: {...}, action: {type: "shrink"|"window"|"none", detail}, solver_outcome: {I, L, W, sat}}`.
The report template emits the "optimal relative to the verified fact set Φ" phrasing together with the full text of Φ.

## 7. Golden Case: bwd Zero-Manual Structural Migration

**Starting-point calibration** (the key experimental-design detail): under the current replication v6, the free optimum
for bwd at default budgets is already W_min=4
(VL + 3 compute groups), unlike the paper's "default budget → 2 groups" (our regs(v) bookkeeping is tighter). To
re-enact the paper's incident, first use probing to find a **loose nominal budget B0** such that a 2-group analogue
(the narrow partition with the softmax+dS chain co-located) is SAT at
(95, 273) — the probes reuse the discriminating-probe sequence infrastructure of §22. B0 and the process by which it was
obtained are written into the experiment archive.

**Procedure**: start from B0 with zero manual environment variables and run the §6 loop. Expected evolution: round 1 solves the narrow partition →
ptxas spills → L0 tightens the offending group → round 2 REGISTERLIMIT UNSAT at the same (I,L) → Algorithm 1
L/W increment → 3-compute-group solution → round 3 spill≈0, converged.

**Acceptance**: within ≤3 rounds, produce spill≈ε with `strategy_report.classify_bwd` judging the structure to be three
compute groups; final-solution
TFLOPS ≥ the manually-lowered-budget version (skc_bwd_jos measurement convention); the evidence chain is complete and
third-party re-playable (same ddg + same B0 +
same ptxas version → same trajectory).

## 8. Secondary Case and Division-of-Labor Criterion

- **fwd channel-cost feedback**: for the fwd joint solution's cross-warp-group register channel (the SMEM
  round trip of alpha softmax→rescale, the 5 buffers/34KB channel synthesized by graph_writer), run O_ncu and feed the
  measured sectors/stall numbers
  back into W1's spillcost calibration (here the oracle **supplies data, not constraints**).
- **Division of labor with W1/W2**: W1 fixes coefficients up front; W2 discards candidates after the fact; **W3 re-solves
  only when failure is systematic** — the criterion
  = ≥ half of W2's top-k candidates share the same oracle failure signature (e.g., all spill). At that point discarding
  candidates is useless
  (the entire (I,L,W) region is infeasible in reality); only distilled constraints can push the search to a different structure.

## 9. Module Layout, Tests, Effort Estimate, Risks

```
paper_joint_solver/oracle/{__init__,ptxas,ncu,cache,distill,loop}.py
tests/test_oracle_{ptxas_parse,cache,distill_l0,budget_regression}.py
tests/fixtures/ptxas_v/*.txt        # real ptxas -v samples: with spill / no spill / multi-entry
oracle_runs/<case>/{facts,evidence}.jsonl
```

Test plan: ① parser unit tests pinned by fixtures (including actual Blackwell 13.x output format); ② cache hit /
version invalidation; ③ **budget regression**: `--reg-budgets` uniform across all groups vs the old `--reg-budget` —
solutions bit-for-bit identical;
④ distill_l0 pure-function determinism; ⑤ end-to-end dry-run: one bwd loop round (convergence not required) as a smoke test.

Effort estimate: L0 loop (incl. per-WG budgets + parser + driver) ~3 days; L2 ~1 day; L1 if triggered ~2 weeks.

Risks and mitigations: ① Blackwell SASS is undocumented — L0 only greps LDL/STL mnemonics and branch labels, no decoding;
if segmentation fails, fall back to uniform whole-group shrinking; ② ptxas version drift — Φ invalidates entries by
version tag and re-queries, and the certificate states the version; ③ L0
over-tightening excludes the true optimum — record the margin bound in at each shrink; if a W2-measured good solution
gets excluded, escalate to L1; ④ minutes-level compile
latency per round — the cache absorbs repeated queries; total loop cost ≤ 3×2 min, acceptable.
