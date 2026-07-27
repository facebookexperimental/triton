# D113349666 Paper and Code Review

Review date: 2026-07-23

Reviewed material:

- Paper: *Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs* (Twill, arXiv:2512.18134)
- Differential: D113349666
- Comparison base: `remote/master`
- Scope: joint solver, schedule-graph write-back, sched2tlx emitter changes, SKC/CuTe bindings, committed solver/performance artifacts, and tests

## Overall Assessment

The implementation follows several important elements of the paper: SCIP cost normalization, CBC modulo scheduling, a Yices2 QF_LIA joint formulation, monotone `(I, L)` search, time-resolved liveness, warp assignment, and zero latency for streaming operations.

However, the current code and committed evidence do not support the differential's claims of a "1:1 replication", automatic exact rediscovery of the FA4 strategy, or correctness of all generated B200 kernels. Several P1 issues can change solver feasibility and optimality, and the Phase B performance path does not implement the complete solved schedule.

**Recommendation: keep the differential in `Changes Planned` until the P1 issues are fixed, solver artifacts are regenerated, and the relevant correctness/performance measurements are rerun.**

## Findings

### P1: Lower-II timeouts can be promoted into an "optimal" result

[`joint_smt.py`](paper_joint_solver/joint_smt.py#L303) distinguishes `unknown` from `unsat`, but [`search.py`](paper_joint_solver/search.py#L59) advances to the next `L` or `I` whenever no solution object is returned. [`modulo_ilp.py`](paper_joint_solver/modulo_ilp.py#L100) similarly returns `None` for a timeout without an incumbent, while a timeout incumbent is returned with `optimal=False` and that flag is ignored by the search.

Consequently, a lower-II query that merely times out can be skipped, after which a higher-II solution is reported as optimal. This violates Algorithm 1, whose optimality argument requires lower values of `I` to be proven infeasible and the initial modulo schedule at a fixed `I` to be optimal.

There are two additional completeness problems:

- [`ddg.py`](paper_joint_solver/ddg.py#L109) bounds RecMII using the sum of node latencies even though recurrence weights come from edge latencies.
- [`modulo_ilp.py`](paper_joint_solver/modulo_ilp.py#L42) uses `critical_path + 2 * II` and the greedy result as hard ILP bounds, making the incomplete greedy scheduler semantically load-bearing rather than only a warm start.

### P1: Production cost normalization discards instruction multiplicity

[`normalize.py`](paper_joint_solver/normalize.py#L58) implements a multiplicity-weighted `1 <= sum(C') <= U` bound, consistent with treating `C` as the paper's per-instruction cost list. The production caller in [`ddg.py`](paper_joint_solver/ddg.py#L264) constructs `pool` with `sorted(set(...))`, erasing all multiplicities before calling the normalizer.

The implementation also performs a 10% pre-clustering pass and zeros costs at or below `max(cost) / 32` before the paper's normalization ILP. These are additional model transformations, not the normalization procedure described in section 5.2.

This changes the normalized costs consumed by every modulo and joint solve. All committed `(II, L, W)` artifacts must be regenerated after fixing or explicitly redefining this behavior. Edge-only cost clusters can also be absent from `pool`, causing a `KeyError` at [`ddg.py`](paper_joint_solver/ddg.py#L278) for otherwise valid graphs.

### P1: Register and memory feasibility use incorrect footprints

[`ddg.py`](paper_joint_solver/ddg.py#L125) parses both tensor results and `!ttg.memdesc` results as full value storage, then records `bytes / 4` as registers. As a result, a `!ttg.memdesc<64x128xf16>` handle is charged as 4096 registers. [`joint_smt.py`](paper_joint_solver/joint_smt.py#L52) can therefore classify shared-memory and Tensor Memory handles as register-resident tiles.

At the same time, [`machine.py`](paper_joint_solver/machine.py#L36) documents fixed SMEM/TMEM allocations performed outside the schedule graph, but both overheads default to zero and the normal CLI does not set them. Cross-WG channels are synthesized after solving, and `rewrite_schedule_graph()`/`validate()` do not recheck total emitted SMEM or TMEM capacity.

These issues invalidate REGISTERLIMIT, MEMORYCAPACITY, minimum-warp, SAT, and UNSAT conclusions until resource accounting is corrected.

### P1: CAPACITY and cross-warp spill constraints are not the paper's formulation

The paper uses a two-dimensional integer `RRT[v][f,c]`, allowing an operation to consume multiple functional units, non-contiguous rows, and arbitrary usage amounts. [`ddg.Node`](paper_joint_solver/ddg.py#L39) stores only one `pipeline` and one scalar `occupancy`; [`modulo_ilp.py`](paper_joint_solver/modulo_ilp.py#L84) and [`joint_smt.py`](paper_joint_solver/joint_smt.py#L121) model consecutive unit-demand slots on that single resource.

The paper also specifies `spillcost(u)`. The implementation stores one global spill cost in [`ddg.py`](paper_joint_solver/ddg.py#L75) and applies it to every cross-warp edge in [`joint_smt.py`](paper_joint_solver/joint_smt.py#L245), including edges carrying shared-memory or TMEM values.

The code may represent a useful restricted machine model, but the claims that Fig. 4/5/6 are implemented one-to-one and that the remaining machine-model choices are merely underspecified details are too strong.

### P1: Schedule-graph write-back silently underallocates live ring buffers

[`graph_writer.py`](paper_joint_solver/graph_writer.py#L165) computes the required ring depth from the rewritten live span, then silently caps it at the baseline depth plus two. Synthesized channels are independently reduced until their footprint is at most 16 KiB at [`graph_writer.py`](paper_joint_solver/graph_writer.py#L571), regardless of the computed live span.

The committed forward graph already violates the required invariant: [`schedule_graph_jos.json`](../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph_jos.json#L2945) has `II=66`, `live_start=100`, `live_end=672`, and `count=5`. The required depth is `ceil((672 - 100) / 66) = 9`.

This can overwrite data that is still live or deadlock the producer/consumer protocol. `validate()` currently checks only basic structure and paired counts, so it accepts this graph.

### P1: The generated backward JOS kernel does not compile

After repartitioning, a `local_alloc(value)` producer is in one warp group and its two MMA consumers are in different groups. No paired cross-WG data channel is synthesized. [`emitter.py`](../sched2tlx/sched2tlx/emitter.py#L2370) skips the bridge and assumes `cross_wg_barriers` already covers this case.

The resulting [`generated_hd128_jos.py`](../sched2tlx/examples/case4_FA_bwd/generated_hd128_jos.py#L177) references `L0_smem_3_full` and `L0_smem_3_empty`, which were never allocated, and it never stores the producer value into `L0_smem_3`. All four committed benchmark shapes fail compilation with this `NameError`, as recorded in [`results_bwd.json`](bench/results_bwd.json#L48).

This directly contradicts the Test Plan statement that all generated/skeleton kernels pass B200 correctness. The writer should synthesize the multi-consumer channel, or the solver/writer must reject this partition. Validation should also catch undefined emitted symbols or compile the generated fixture.

### P1: Phase B measures FA4 parameter binding, not the complete solved schedule

[`shim_fwd.py`](skc_cute/shim_fwd.py#L21) directly subclasses FA4's `FlashAttentionForwardSm100`. The forward binding in [`fwd_1cta.json`](skc_cute/bindings/fwd_1cta.json#L57):

- drops the exact normalized cycle schedule;
- drops the solver's `BN=64` geometry;
- freezes FA4's warp-role tuples and MMA issue order;
- describes `split_P_arrive` as a semantic approximation of the solver schedule.

The 1280 TFLOPS result is useful evidence that solver-derived register quotas or stage parameters can improve an FA4 implementation. It is not evidence that the complete `(M*, A*)` emitted by this solver attains the paper's performance or reproduces the paper's manual translation process.

The Phase B `m4` driver also does not run paired stock/1-CTA FA4 baselines in the same command, so the 1047-vs-1275-vs-1280 headline combines results from different artifact sets without recording a complete software/hardware environment.

### P1: The exact-FA4 and UNSAT-ablation claims overstate the evidence

The paper states that the joint optimizer's discovered Blackwell-forward strategy is exactly the FA4 strategy. The current unconstrained optimum explicitly lacks a dedicated MMA group. [`REPORT.md`](REPORT.md#L39) states that the complete FA4 structure is feasible only at `L+2..4` after applying manual co-location and separation probes. [`refit_check.py`](refit_check.py#L51) pins the already recorded FA4 partition and proves it SAT; it does not show that Algorithm 1 independently returns that partition.

The ablation driver in [`run_ablations.sh`](run_ablations.sh#L24) does not fix the paper's comparison point `(I, L)`, despite its comment saying it uses the same budget. Its no-subtiling command performs a full search and eventually finds a SAT solution at a larger II. [`__main__.py`](paper_joint_solver/__main__.py#L46) also collapses timeout, unknown, wall-clock exhaustion, and genuine exhaustive UNSAT into the same top-level `satisfiable: false` field.

The no-cross-warp experiment is structurally UNSAT under the combination of VARIABLELATENCY iff isolation and forcing every edge to remain same-warp. It is therefore not evidence that a nontrivial search discovered an UNSAT configuration.

### P2: Experimental claims and artifacts are incomplete or internally inconsistent

- The paper reports a 19-second Blackwell-forward joint solve. The committed end-to-end search takes 617.94 seconds in [`subtiled_joint_solution_v6.json`](subtiled_joint_solution_v6.json#L212). Comparing the paper's complete solve to individual 3-15 second SMT decisions is not the same scope.
- The paper's backward result changes from two groups to three groups after reducing the register budget and reports a small measured speedup. The current report records `W_min` changing from 4 to 8 and does not benchmark the reduced-register kernel.
- The paper's Blackwell-forward SWP-only and heuristic-WS bars are not implemented or measured, so the claim that joint optimization is necessary is not reproduced.
- [`results_e3.jsonl`](skc_cute/results_e3.jsonl#L2) is invalid JSON. [`run_phase_b.sh`](skc_cute/run_phase_b.sh#L12) discards stderr and exit status, so an empty result cannot distinguish a deadlock from a crash or other timeout.
- The backward 1042 TFLOPS headline is the identity-control value. The actual bound result at 16K is approximately 1036 TFLOPS in [`results_bwd.jsonl`](skc_cute/results_bwd.jsonl#L14). It remains within 2%, but the value is attributed to the wrong experiment.
- The claim that the paper likely measured against a slower 2-CTA FA4 baseline is speculative. The paper does not disclose its FA4 configuration, and the committed forward baseline/result rows are not paired under one recorded environment.

## Comparison With Master

Relative to `remote/master`, the differential changes 110 files with approximately 73,332 insertions and 66 deletions:

- 78 new files under `paper_joint_solver/`, approximately 12,306 lines;
- new DDG, TTGIR/MLIR, schedule-graph, generated-kernel, and benchmark fixtures under `sched2tlx/examples/`;
- 206 insertions and 66 deletions in `sched2tlx/sched2tlx/emitter.py`;
- a new TMEM alias test and additional paper-joint-solver tests.

The relevant Triton paths have the same delta when compared directly with `remote/master`; the current local parent contains an unrelated monorepo commit outside this scope.

## Areas That Match the Paper

The following implementation choices are substantially aligned with the paper:

- SCIP is used for the cost-normalization ILP and `U=300` is the default.
- CBC is used for the initial modulo schedule.
- Yices2 QF_LIA is used for the joint system.
- Integer start-time variables can represent the paper's one-hot `op[v,i,t]` grid when uniqueness and bounds are enforced.
- The implementation includes dependence, completion, liveness, memory, register, variable-latency, blocking, and warp-assignment constraints with the same broad structure as Figures 4-6.
- The search has the paper's outer `I` and inner `L` loop shape, and now exhausts the full constant-`ceil(L/I)` window by default.
- Streaming TMA loads are modeled with zero outgoing latency.
- The main B200 shapes, FP16 non-causal attention configuration, sequence lengths, and stated CUDA 13.0 environment match the paper's reported setup.
- The report correctly discloses that the paper's high-performance kernels were manually translated to CUDA rather than produced by a fully automatic Triton lowering path.

These alignments support describing the change as an experimental implementation inspired by, and partially replicating, Twill. They do not override the correctness and evidence gaps above.

## Test Verification

Commands run during this review:

```text
.venv/bin/python -m pytest -s --tb=short \
  third_party/tlx/tools/paper_joint_solver/tests
```

Result: collection failed with four `ModuleNotFoundError: pyscipopt` errors. The repository does not provide a working dependency setup for the advertised CPU-only test command.

```text
.venv/bin/python -m pytest -s --tb=short \
  third_party/tlx/tools/sched2tlx/test_tmem_alias.py \
  third_party/tlx/tools/paper_joint_solver/tests/test_graph_writer.py
```

Result: 7 passed.

The available tests do not directly exercise `solve_joint()` or `run_search()`, and they do not cover timeout/unknown handling, memory-descriptor footprints, per-producer spill cost, multi-resource RRTs, recurrence bounds, or rewritten ring-depth capacity.

No GPU performance run was performed as part of this review; performance conclusions above are based on the committed raw artifacts.

## Required Before Review

1. Preserve and propagate `unknown`/timeout separately from proven infeasibility; require optimal modulo results before claiming global optimality.
2. Fix normalization multiplicity and edge-cost handling, then regenerate every solver artifact.
3. Correct register/SMEM/TMEM footprints and include all fixed and synthesized allocations in the solved or post-write capacity checks.
4. Implement the paper's RRT/spill semantics or explicitly narrow and rename the claimed model.
5. Remove ring-depth caps that violate liveness, or reject schedules that cannot fit the required depth.
6. Fix the backward cross-WG `local_alloc(value)` channel and add compile/correctness coverage for the generated kernel.
7. Reframe Phase B as parameter-binding evidence, or implement the actual solved cycles, geometry, and assignment on the target backend.
8. Reproduce the exact-strategy and three ablation claims with committed, verdict-preserving artifacts at fixed `(I, L)`.
9. Provide a reproducible test/dependency target and rerun correctness, lint, and affected CI tests.
10. Rerun paired performance baselines and variants with complete environment metadata after the solver/model fixes.
