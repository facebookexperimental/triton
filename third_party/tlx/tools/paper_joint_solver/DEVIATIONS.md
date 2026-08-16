# DEVIATIONS

Every place this reproduction had to decide something the paper does not
decide, plus every mechanism that exists here and not in the paper.

The reproduction target is **literal fidelity**: where the paper is explicit,
the code follows it even when the result is worse. The degenerate solutions
the literal pipeline produces on compiler-extracted inputs are the
reproduction's finding, not a defect to be repaired. Nothing in this file is
a guard, a threshold, a retry or a fallback; the entries below are disclosures
of interpretation, not corrections.

Sections are grouped by the layer that owns the choice. Entries marked
**protocol** are execution-harness choices with no model content.

---

## 1. Implementation path: the automatic emitter arm is retired

The paper's own implementation story is manual: it emits software-pipelined IR
"or used by an expert as reference for a manual implementation". This
reproduction originally carried a second, automatic path that lowered a
solution to TLX through `graph_writer` and `emit_bench_kernels`. That arm is
**retired from the literal pipeline**:

- it was a disclosed deviation from day one (the paper describes no automatic
  code generator);
- under the literal single-shot normalization the solutions it consumes are
  degenerate, so the performance numbers it produced are not comparable to the
  paper's and its remaining value was a functional smoke test;
- keeping it alive would have required restoring a pre-literalization raw-DDG
  adapter as private legacy code, which costs more than the residual value.

Consequences:

- `run_emitter_cases.sh` is a stub that prints a retirement notice and exits
  non-zero. There is no emitter batch.
- The emitter solve entry and its extended constraint system (group/lane
  encoding on quantized widths, prefix and full-group lane masks, spill
  staging footprint, register-carried same-lane placement, fixed resource
  overheads, the SM-wide register bound, and the label-symmetry pruning that
  served them) are deleted. The joint constraint module is now exactly the
  paper's Figures 4/5/6 — strictly more literal than before.
- `graph_writer.py`, `emit_bench_kernels.py` and `bench/` are kept in the tree
  for reference with a RETIRED note in their module docstrings. They are not
  driven by any literal run. Their v6-era artifacts stand as the final
  performance record of the guarded regime.
- **One paper claim went down with that arm: the exposed depth parameter.**
  Section 5.3 says the pipeline depth of a streaming operation is exposed "as
  a parameter to an external auto-tuning system". Nothing on the literal path
  carries it. Neither `paper-joint-pipelined-ir-v3` nor
  `paper-joint-handoff-v2` has a depth, stage or ring field, and the handoff's
  lowering contract names only four orthogonal manual steps — memory
  allocation, data-layout conversion, synchronization placement, instruction
  selection. The only depth logic in the tree is `graph_writer._ring_depth`
  and the buffer counts it feeds, which belong to the retired arm. On the
  literal path the parameter is carried by the human: ring depth is chosen
  during manual CUDA authoring, inside the lowering decisions the handoff
  deliberately leaves open. So the claim is neither implemented nor simulated
  here; it is delegated to the authoring step, and the artifacts do not
  pretend otherwise.
- The literal implementation path is: `run_main_cases.sh` → paper artifacts
  (`paper-joint-pipelined-ir-v3` + `paper-joint-handoff-v2`) → manual CUDA
  authoring by a human, which is the paper's own workflow.
- **That authoring step is agent-assisted, and every artifact says so.** The
  paper's endpoint is an expert writing CUDA; here the expert works with an
  agent. The protocol is named in the artifacts rather than left implicit:
  each authoring package stamps `authoring_mode: "agent-assisted-manual-cuda"`
  and its schema string is `paper-joint-agent-assisted-manual-cuda-v2`
  (`skc/_schema.py:24-28`). What preserves the paper's *expert decision*
  semantics is that the audit will not take the agent's word for anything:
  `audit_authoring` demands four named human reviews — mapping, memory,
  synchronization and clean room — each with a named reviewer, an `approved`
  status, and a SHA-256 that must equal the digest of the manifest it approves,
  the clean-room digest being recomputed from the block itself
  (`skc/audit.py:1516-1551`). A missing, duplicated, unapproved or
  hash-mismatched review fails the audit closed, so a named human approval is
  load-bearing at exactly the point the paper puts the expert. These are
  hash-*bound* records, not signatures: a review is a reviewer name beside a
  digest, with nothing tying the name to the digest cryptographically. No
  artifact in the tree has passed them — every scaffold ships at
  `manual_completion_required` with `human_reviews` empty (section 7).
- `solve_joint_audit()` and the paper CLI's `--baseline-graph` / `--ir-out` /
  `--handoff-manifest-out` triple are **retained**: they are paper-path assets
  (the FA4 refit instrument and the manual-implementation inputs).
- `provenance.model` survives as a key; the only value now produced is
  `"paper"`. The emitter profile branch of `load_schedule_plan` remains,
  because `emit_bench_kernels.py` still calls it.

## 2. Normalization and curation (spec-1, spec-2)

- **Single-shot U=300.** The paper states one normalization budget
  (section 5.2). The retry ladder, the two-stage escalation and the
  resolution-error class are removed; `U` is passed once and whatever
  collapse it causes is the result. Explicitly passing
  `--normalization-u 300` on the command line keeps the input auditable.
- **What the cost pool C contains is a choice.** Section 5.2 says only "let
  the list of integers C be the original cycle counts" and never enumerates
  its members. Here C is the concatenation of three classes, in this order:
  every RRT maximal-segment duration, every edge latency — already zeroed
  where the section 5.3 streaming rule below applies — and every node's spill
  cost (`ddg.py:339-373`). It is a *positional* multiset — duplicates are kept
  and zeros join it — so the budget `1 <= sum(C') <= U` counts each occurrence
  rather than each distinct value, and the normalized list is read back entry
  by entry in the same order. Two consequences worth stating: a
  cost that appears many times consumes the budget many times, and the
  collapse fractions below are counts of *entries*, not of distinct costs.
  The segmentation is itself an extension. The paper defines the final
  reservation table but no projection from compiler cycle measurements onto
  it, so splitting each RRT into maximal constant-demand runs and normalizing
  the run durations independently is this reproduction's rule
  (`ddg.py:191-214`, which says so in its own docstring). It was chosen
  because it keeps the integer demand vectors exact and leaves exactly one
  final cycles(v) shared by COMPLETION and CAPACITY.
- **Degenerate baseline is the finding.** Under a single U=300 pass, large
  fractions of the positive cost pool collapse to zero: 21/81 for `fwd`,
  104/128 for `fwd_subtiled`, and 115/177 for the sub-tiled backward graph,
  which `bwd` and `bwd_lr4096` both consume. Counted positionally over the
  entries of C that start positive, measured on the committed golden curated
  fixtures and, for the backward graph, on the curation of
  `case4_FA_bwd_subtiled` that `tests/test_normalize.py` performs — the three
  graphs above are the whole input side of the canonical batch, and the
  backward pair differs only in its register budget. The published II=66/95
  are unreachable from these inputs. No recovery logic exists, and the
  expected column of every observation was rebaselined once to the degenerate
  numbers. The `lit1` artifact stem marks this semantic break. Of the four
  canonical cases only three left a solution artifact: `bwd_lr4096` exceeded
  the watchdog and is recorded as a did-not-terminate observation, which the
  rc contract treats as a normal terminal state (section 6, I-5).
- **Streaming "zero latency" is read as the outgoing edges only.** Section 5.3
  assigns streaming operations "zero latency in our cost models". The reading
  implemented here zeroes the latency of every edge leaving a streaming
  producer and applies no reduction at all to the producer's own cycles(v) or
  reservation table (`ddg.py:328-332`); curation only decides which nodes are
  streaming (`curate_ddg.py:537`). The alternative reading — cycles(v) = 0 and
  an empty RRT — fits the same sentence, and the paper's own worked example
  takes an edge's d from the corresponding reservation table, i.e. reads an
  execution span as a latency, so the sentence does not settle which span it
  means.

  The choice is not cosmetic, in two ways. It reshapes C, because the zeroing
  happens before the cost pool is assembled: every canonical graph has exactly
  two streaming TMA loads with one outgoing edge each, both of 556 cycles, so
  two large entries enter normalization as zeros. The collapse denominators
  above are 81, 128 and 177 under this rule and would be 83, 130 and 179
  without it, and since the ZLP minimizes F over all pairs, dropping the two
  largest edge costs moves the scaling of every other cost as well. It also
  leaves the producer inside the length and resource constraints: its
  normalized span still bounds the schedule length through `LEN(M)`
  (`modulo_ilp.py:62,100`) and COMPLETION (`joint_smt.py:184-190`), and its
  normalized reservation table still occupies the TMA unit in CAPACITY
  (`modulo_ilp.py:68-83`, `joint_smt.py:201-226`), where the other reading
  would remove it from both. So the reading moves the reachable (II, L) set
  and not only the bookkeeping. (556 is the curated input span; what reaches
  the constraints is the normalized one, bounded with everything else by
  `sum(C') <= U`.) The zeroed delay is not confined to DEPENDENCE either —
  `edge_latency` is also read by the cross-warp spill disjunction
  (`joint_smt.py:347`) and written into the emitted schedule and pipelined-IR
  artifacts (`schedule_plan.py:1078,1162`) — so the reading is visible in
  published output.

  The edge reading is kept because the purpose the paper gives the rule,
  letting a consumer schedule precisely against a producer that runs ahead, is
  a statement about the dependence rather than about the producer's own
  occupancy, and because a functional unit that is busy is busy whatever the
  cost model calls it. `tests/test_ddg_and_modulo.py:382` pins it. The other
  reading is equally undetermined by the paper and adopting it would
  invalidate every recorded case, so the choice is disclosed here rather than
  changed.
- **Curated G is the solver's only graph input.** The paper's system consumes
  a dependence graph it does not specify how to obtain; curation is the
  disclosed, hash-chained boundary between the compiler dump and the paper's
  own logic. The curated loader is fail-closed on unknown fields.
- **TMEM footprints are counted in columns**, converted from bytes at
  512 B/column in the curation layer; the manifest records the raw bytes.
- **Zero-cost pools need an anchor.** When a fixture's entire cost pool is
  zero, the normalization program's `sum(C') >= 1` forces one entry positive.
  This is literal behaviour, disclosed here because it surprises fixture
  authors, not worked around.

### Curation interpretations

The paper takes its dependence graph as given. Every rule that turns a compiler
dump into that graph is therefore a choice this reproduction had to make, and
each one is recorded in the curation manifest as well as here.

- **Graph membership is decided by a scheduling decision, not by the paper.**
  This is the one place where a decision from the guard-era emitter leaks into
  what the paper treats as pure input, and it deserves to be read as such. The
  paper's graph is a tile-level graph its authors curated by hand, and it never
  states the criterion. The proxy used here is the baseline emitter's own infra
  marking — a node is dropped when that emitter assigned it `warp_group < 0`.
  So the boundary of the "input" is drawn by a scheduler's prior output. In the
  backward cases this removes two `tt.descriptor_reduce` nodes (the epilogue
  reduction stores); whether the paper's authors would have removed the same
  nodes is unknowable, so the rule is applied as stated and every removal is
  listed in the manifest.
- **Completion tokens become ordinary dependence edges.** The paper's edge set
  is homogeneous and has no token concept at all. A completion signal is kept
  as the timing dependence it represents, and its effects on liveness, spill
  and streaming then fall out of the paper's own rules on the curated graph.
- **Blocking assignment rule.** The paper says only that an edge "is assigned"
  as blocking and treats the assignment as architectural modelling. The rule
  here: the source unit is asynchronous (TC or TMA), or the edge is a
  completion edge, since waiting on completion needs a barrier.
- **Variable latency means TMA loads.** The paper's examples are transfers and
  loads only, so stores are excluded — the most literal defensible reading.
- **Spill costs are synthesized when absent.** The paper assumes the annotation
  is already provided; the dumps carry none. Explicit values win; a register
  producer is priced at 30 (a machine fact for the round trip through shared
  memory); a memory-backed producer is priced at 0.
- **The tensor-memory port is treated as a fifth functional unit.** The paper
  delegates the unit set to the machine model and never lists that port as a
  functional unit, calling it a synchronization operation instead. The port
  being independent of the ALU is a hardware fact, so it is modelled and the
  divergence recorded.
- **Reservation tables are synthesized occupancy-front-loaded.** The shape is
  an assumption about missing dump input; the paper takes the table as given
  and states no synthesis rule.
- **Footprint ownership is narrowed to one node.** Tensor-memory footprints for
  accumulator and store operands are deduplicated per shared object and their
  ownership collapsed onto a single node, because the curated file carries only
  a per-node footprint. The multi-member truth stays in the manifest, so the
  solver's literal per-value sum remains physically correct.
- **Deleted chains are reconnected by summation.** A kept → dropped* → kept
  chain becomes one synthesized edge whose distance and delay are the sums
  along the chain. No canonical fixture contains such a chain; the rule exists
  for completeness and records itself in the manifest when it fires.

## 3. Search (spec-3)

- **I-1..I-4 solver substitution.** The paper's modulo-scheduling ILP is
  discharged to SCIP 10.0 (via PySCIPOpt) rather than the solver the paper
  names. The formulation is unchanged; only the back end differs. The version
  is pinned in `requirements.txt` so the arithmetic is reproducible.
- **Algorithm 1 is literal.** I ascends from 1, there is no lower bound, no
  warm start, no structural pre-check, no time limit and no span cap. The
  algorithm's only terminal state is sat, so the CLI's only normal exit code
  is 0.
- **At a fixed I the modulo ILP minimizes L.** `OPTIMAL-MODULO-SCHEDULE(G, I)`
  is asked for the feasible schedule with the smallest `LEN(M)`
  (`modulo_ilp.py:85`), so Algorithm 1's inner L search starts at the shortest
  length that I admits. The paper names the sub-problem but says nothing about
  choosing among the schedules feasible at the same I, and any such schedule
  would satisfy the text. The choice was documented only inside
  `modulo_ilp.py` itself (lines 9-11, 31, 37) and never in this ledger, which
  matters because the objective is part of the formulation the solver
  substitution bullet above calls unchanged. It is recorded here because it
  fixes where the L window begins, and so which point of the optimal set the
  search reports first.
- **The first-window probe is a published execution protocol, not solver
  logic** (**protocol**). Algorithm 1 does not terminate on a structurally
  unsatisfiable configuration — it raises I forever. Ablations that need
  bounded evidence use `python -m paper_joint_solver.probe`, which locates the
  smallest I with a modulo schedule, enumerates that I's whole L window, and
  records the verdict at every point (`execution_mode="first-window-probe"`,
  exit code always 0). This realizes the paper's section 6.2.2 claim
  ("unsatisfiable for the smallest I which makes the ZLP succeed, forcing a
  search over larger values of I and L") without asserting a terminal outcome
  for a search that never finishes. The canonical path never calls it.

## 4. Constraint system (spec-4, spec-6)

- **Warp sets, not groups.** `opw[v,w]` ranges over physical warps, so an
  operation's assignment is a *set* of warps and sets may overlap partially.
  Two operations are "on the same warp" only when their sets are **exactly
  equal**; a partial overlap is not co-location. This is the reading that lets
  the encoding express what the paper's text describes.
- **WARPUNIQUENESS for multi-warp operations** is read as *exactly*
  `min_warps` warps, not "at least". The paper states the constraint for the
  single-warp case only.
- **REGISTERLIMIT amortizes.** A value produced by a multi-warp operation is
  charged `ceil(regs / min_warps)` words to each member warp, not the full
  amount to each. The paper does not say how a distributed value is charged.
- **MEMORYCAPACITY sums per value**, each curated value contributing its own
  footprint.
- **VARIABLELATENCY occupies physical warp 0 by an iff.** `opw[v,0]` holds
  exactly when `v` has variable latency. Because it is an iff and not an
  implication, warp 0 is *unavailable to ordinary operations*: under a budget
  of B warps a regular operation has only B−1 choices. Two consequences worth
  stating: `num_warps_override=1` is structurally unsatisfiable for any model
  containing a regular operation, and "everything on one warp" must be
  written as B=2.
- **`--no-cross-warp` scopes to register-data edges.** The paper's Figure 6
  text ("data in the registers of warp w cannot be accessed on warp w′")
  supports the register-data reading, and that reading is now hardcoded as the
  only semantics; the alternative all-edges reading and its flag are gone.
  `--no-cross-warp` itself stays on the canonical surface because it is the
  paper's own ablation input. Under the ablation the spill-latency disjunction
  is skipped for *all* edges, not only register-data ones; the difference is
  observable only for a zero-register node carrying a positive spill cost,
  which no curated graph contains.
- **spillcost covers cross-warp register traffic only**, priced as added
  latency on the dependence. A cross-warp spill never charges modelled SMEM
  bytes or a barrier pair.
- **CONCURRENCY keeps the spill gating** that the paper mentions only in prose
  as an implementation aside ("while elided for brevity, the implementation
  includes a similar constraint for cross-warp spills", sec 4.3). A consumer of
  a cross-warp spill therefore stalls like a consumer of an asynchronous
  result, and because the spill depends on the warp assignment the constraint
  is gated on some in-edge actually being cross-warp.
- **A zero-cost operation still occupies one issue cycle for length purposes.**
  Where an operation's latency enters the schedule length it is read as
  `max(1, lat)` (`modulo_ilp.py:62,100`, `joint_smt.py:168,188`,
  `schedule_plan.py:853`), so a node whose latency normalizes to zero still
  consumes a cycle. This is pre-existing behaviour that predates the
  literalization and the paper does not say what a zero-latency operation
  costs in length. It is called out because the single-shot normalization
  collapses much of the cost pool to zero, which makes the clamp routine
  rather than exceptional.
- **L < I produces degenerate pipeline regions**: an empty prologue and an
  inverted epilogue, rather than a rejection. The paper's region split assumes
  L ≥ I and says nothing about the shorter case.
- **Per-warp issue order is serialized per warp.** A node belonging to several
  warps appears in each member warp's issue trace, so a trace is no longer a
  partition of the instruction list. The sort key `(stage, offset, node_id)`
  is warp-independent, so co-resident nodes keep a consistent relative order
  in every warp that shares them.

## 5. Reporting and audit (spec-5)

- **Every strategy verdict is an observation.** The classifier states its own
  criteria in a `criteria` block and no field of it is a gate. A `False`
  verdict is a recorded result. Under the degenerate baseline the forward case
  does not classify as the paper's reported shape; that is the finding.
- **Operationalized criteria.** The paper describes its strategies in prose.
  This reproduction fixes one operationalization each for the forward shape,
  the staging criterion, the exclusivity of the three-group backward strategy,
  the group count, and the "exactly two exponential groups" condition. Other
  operationalizations are defensible; the report prints the one used.
- **"Group" means "warp set".** The classifier needs a node→group map and the
  paper's solution has no groups, so identical warp sets define a group and
  overlapping sets do not. This matches the co-location reading in section 4.
- **The reduced register budget is 4096 words/warp.** The paper (page 12) says
  only "reduced register-per-thread budget" with no number; 4096 =
  128 regs/thread × 32 lanes, against a full budget of 255 × 32 = 8160. It is
  recorded as an experiment input in provenance and does not block emission.
- **The FA4 refit is an audit instrument absent from the paper.** It goes
  through `solve_joint_audit()`, pins the recorded partition at the free
  optimum's `(II, L)`, inherits the audited solution's own experiment inputs,
  and is uncapped. The free solution's own shape is recorded as an observation
  because the paper defines no tie-break inside the optimal set.
- **The FA4 template is projected onto the curated graph, not intersected
  with it.** The frozen template is keyed on the *raw* node ids, so it still
  carries the emitter-infra nodes (0 and 1 in the sub-tiled forward case) that
  curation deletes; the template file itself is never modified. At refit load
  time the partition is restricted to the curated node set using the curation
  manifest's own `action: "dropped"` records — the only surviving account of
  what curation removed. The exact-coverage check is kept and simply runs after
  the projection, so a template node that curation did *not* record as dropped,
  or a curated node the template omits, still fails closed. Dropping those two
  `arith` nodes leaves the template's group count at five, so the "exactly the
  same as proposed by FA4" shape under test is unchanged. The projected ids are
  disclosed in the refit metadata as `template_projected_nodes`.
- **The renderer refuses rather than reusing colors.** The degenerate bwd
  solution has 27 distinct warp sets against an 8-entry palette, and the
  renderer's pre-existing documented policy is to reject. That policy is
  unchanged — adding a fallback would be inventing a guard — so
  `run_main_cases.sh` no longer requests a figure and the literal batch emits
  no dot file.

## 6. Run scripts and harness (spec-7)

- **I-1 Warp cuts are literal constants 31 and 16** (**protocol**). The paper
  says only "Reducing the number of warps". The cuts are taken from the full
  machine budget W=32 (`machine.MAX_PHYSICAL_WARPS`): W−1 and W/2, as fixed
  constants with no conditional and no guard. The previous derivation from the
  baseline solution's own warp usage is removed.
- **I-2** The reduced register budget of 4096 — see section 5.
- **I-3** `--normalization-u 300` is passed explicitly; the value is the
  paper's stated choice and the pass is single-shot.
- **I-4 Ablation execution protocol** (**protocol**) — the first-window probe,
  see section 3. The baseline still runs the full literal search; the four
  modified configurations are probed. The protocol is disclosed in the script
  header, in each probe artifact, and in `ablation_report.md`.
- **I-5 Watchdog, default 86400 s** (**protocol**). Time bounds moved out of
  the solver and into the harness (`timeout(1)`, overridable via
  `WATCHDOG_S`). **No exit code is ever mapped to a verdict**: rc 0 is the only
  normal terminal, rc 124/137 is the observation "did not terminate", and any
  other rc — including argparse's rc 2 — is an infrastructure failure that
  aborts the batch. The previous `--ilp-seconds` / `--smt-seconds` /
  `--max-wall-s` budgets are gone; the halted v8 batch, whose first case
  recorded `unknown` at a SCIP time limit, is exactly the artifact that
  motivated this.
- **I-6 Single-core pinning** (**protocol**). The paper records its timings
  "on a single core of a Intel Xeon Platinum 8570", which licenses the
  `taskset` single-core binding and the CPU-model check. Host pinning and
  output-directory isolation are local policy.
- **I-7** The register-data scope of `--no-cross-warp` — see section 4.
- **I-8 Degenerate baseline** — see section 2. Scripts and validation contain
  no recovery logic, and every expectation is expressed as an
  expected-vs-observed line rather than a gate.
- **Validation gates only integrity.** Hash chains, schema identifiers,
  `provenance.model`, the host stamp, VCS provenance, overwrite refusal,
  scaffold fail-closure and the sat/rc mirror still exit non-zero. Solution
  shape, strategy class, refit outcome and TMEM accounting are printed as
  observations and never fail the batch.
- **The pinned host and CPU model are parameters.** `PAPER_HOST` and
  `PAPER_CPU_MODEL` default to `dgx003` and `8570` and are overridden
  explicitly for a run on other hardware. Only the second half of that default
  comes from the paper, which names a CPU ("a single core of a Intel Xeon
  Platinum 8570") and no machine; `dgx003` is a locally chosen host that
  carries that CPU. The pair is therefore one paper fact and one local choice,
  not the paper's own machine. The mechanism, and the `paper_comparable=yes`
  contract that canonical validation requires, are
  unchanged; the environment log always records the true hostname and the true
  `lscpu` model beside the pinned values, so an artifact set states plainly
  what it ran on. Reproductions on hardware other than the paper's are not
  wall-clock comparable to it.
- **`SOLVER_LIB_PATH` stays a colon-separated list.** No source in the solver,
  the audit packages or the tests references the BDD library by name, but on
  the paper host the `libyices.so` build carries it as a `DT_NEEDED` dynamic
  dependency, so a path holding only the yices directory fails in the loader at
  the first `import yices`. Source-level absence is therefore not the same as
  no dependency, and the environment contract keeps the multi-directory form:
  the variable names every directory needed to load `libyices`, not just the
  one it lives in.
- **Flag spelling.** The specs write `--num-warps`; the implemented CLI flag
  is `--num-warps-override`, matching the `num_warps_override` parameter name
  fixed in the interface table. The scripts call the implemented spelling.

- **A producer's provenance covers exactly what can change its artifact.**
  Every stamping producer records three things: its input hashes, the digest of
  the shared core it called, and the digest of itself. `probe.py` chooses which
  `(I, L)` points are probed and how verdicts are summarized, yet was covered by
  neither — editing it left existing probe artifacts valid. It now stamps its own
  `probe_sources_sha256` beside the core's `solver_sources_sha256`, matching the
  curation stage's `curator_sources_sha256`. It was deliberately *not* added to
  the core source tuple: that digest is stamped on every artifact, so a
  probe-only edit would invalidate solve artifacts that cannot depend on it.
  Over-invalidation is worse than the gap it closes — it teaches readers to
  override a hash mismatch, which corrodes the whole fail-closed discipline.
  The append is made in `probe.py` rather than in the shared
  `solution_provenance()` helper for the same reason: that helper lives in the
  core, so widening it would re-stamp everything.
  An audit of the other stamping producers found no second instance — the
  solve entry and the audit package are themselves inside the core tuple, the
  curation stage already carries its own digest, and the strategy and refit
  reports stamp no source digest at all (they reference their inputs by hash
  and are observations, not gates).
- **Naming discipline follows canonical output, not module ownership.** The
  audit package's four scaffold schemas were left renamed-pending on the
  grounds that the package sits outside the literalization's core, but its
  artifacts are produced by the canonical batch, which puts them inside it.
  The operative criterion is that whatever enters canonical output is bound by
  the naming discipline. All four now use the `paper-joint-` family; the
  emitter timing model uses the `emitter-` family, matching the pinned
  `emitter-joint-solution-v1`. The four audit schemas also carry a field
  reshape (`physical_groups` → `physical_warps`), so their rename and reshape
  land as one identity switch and the version moves to **v2**; splitting them
  would leave a window in which one string denoted two shapes. The emitter
  timing model is a pure rename with no shape change and stays at v1. Every
  consumer reads these through the shared constant, so each switch is atomic.
  `tests/test_naming_discipline.py` enforces
  this going forward: it is legal hygiene of the same class as the provenance
  checks, constraining what the tree may contain and never what the solver may
  decide.

## 7. Carried-over items still open for review

- **The manual-CUDA authoring artifact was migrated: `physical_groups` →
  `physical_warps`, schemas bumped to v2.** *(Ratified; the approval this class
  of artifact requires is the review of this entry.)* Removing the group
  encoding removed the only input to the group+lane indirection, so the scaffold
  carries one entry per physical warp and the field that used to bind a logical
  group to physical warps is gone — warp numbers come from the solution itself.
  Keeping the old name would have been the real invention: with the group
  concept abolished, `physical_groups` would have been a field name lying about
  its contents, which is worse than a schema change.

  The name stays `physical_warps` rather than becoming `warp_sets`, because the
  two are not the same object: the mapping manifest's field is a list of
  per-physical-warp *authoring records* (27 entries keyed by warp id, each
  carrying `thread_mapping`, `dispatch_anchor`, `entry_predicate`, `regions`,
  `reviewer`), while a solution's `warp_sets` is a node-keyed map to warp sets
  (29 entries, e.g. node 10 → warps 2, 4, 19, 28). Different cardinality, key
  space and direction; sharing a name would assert an identity that does not
  hold.

  All four audit schemas move to `v2` in the migration commit, so the rename off
  the old codename and the reshape are a single identity switch. Doing them
  separately would leave a window in which one string denoted two shapes. No
  hand-authored file in the tree carries the old field name — every authoring
  artifact present is a generated draft at `manual_completion_required`, and the
  superseded outputs that still contain `physical_groups` are append-never — so
  nothing human-written was migrated automatically.
- **Frozen pre-rename artifacts keep the old codename in their schema strings**
  (57 occurrences across the superseded v6/v7/v8 outputs under `solutions/` and
  `ablations_v7/`). They are append-never and are read only by the code that
  wrote them; no current reader accepts those strings. Every schema string in
  the source tree has been renamed, so these artifacts are the only place the
  codename survives, and the naming check enumerates them explicitly rather
  than tracking a count.
- **The archived pre-literalization batch is cited by its diff number**
  (D113349666), not by the git tag the retirement plan mentions: this tree is
  a Sapling checkout inside the monorepo and the git commit the original
  package referenced does not exist here.
- **The four v6-era probe scripts** are archived under `archive/` with a
  SUPERSEDED header. They target the pre-literalization solver surface, have
  no test coverage, and must not be ported. `bwd_skc_solve_final.py` was
  excluded from that archival list and stays at the package root; like the
  archived four it reads a solution field (`warp`) that the paper entry has
  never populated and that no longer exists on the solution object, so it has
  been non-functional since the search layer was made literal. It carries no
  test coverage either.

## 8. Emitter-chain notes kept for the record

These describe the retired arm and are retained only so the audit trail is
complete:

- the retired lowering re-materialized a cross-group operand chain at zero
  modelled cost rather than spilling it;
- it fell back to a default latency when rebuilding an edge whose timing was
  not recoverable from the baseline graph;
- its artifacts were self-declared non-paper via a `classification` stamp and
  were always kept in a separate namespace from the paper solutions.

## 9. The prepared regime

The literal path reproduces the paper's mechanism and, on compiler-extracted
inputs, produces a degenerate schedule (section 2, "Degenerate baseline is the
finding"). Four controlled experiments localized that gap in the *inputs*
rather than the solver. This section is the other half of that finding: a
parallel regime, `prepared`, that carries the input-preparation assumptions the
published results appear to need. It exists to make those assumptions
countable and checkable, not to claim they are the paper's.

Everything below is **outside the paper**. The paper's section 5.2 says only
"let the list of integers C be the original cycle counts" and never enumerates
C's members, gives no projection from compiler cycle measurements onto its
reservation table, and states one normalization budget and one objective. Each
item names where it does come from.

The regime is reached only by `python -m paper_joint_solver.prepared`; the
literal CLI gains no flag, no file in `schedule_plan._SOLVER_SOURCES` is
modified, and prepared artifacts carry their own schema
(`prepared-joint-solution-v1`, `prepared-joint-probe-v1`) so they can never be
read as paper artifacts. `provenance.model` stays `"paper"` — the solve really
is the unmodified Fig 4/5/6 system — and everything this regime changed is
recorded under `provenance.preparation`, including `prepared_sources_sha256`.

### A: cost-pool assembly

- **A1 — one pool entry per node, and only non-zero edges and spills.**
  Source: this reproduction. The literal pool enumerates every RRT maximal
  segment, every edge latency and every node's spill including the zeros
  (176 entries on `fwd_subtiled`); the prepared pool has 129. Zero entries
  consume a ZLP variable and 2(n-1) pairwise constraints while carrying no
  ratio, and each node's trailing idle segment duplicates information already
  held by its out-edges — curation writes every non-streaming edge's latency
  equal to its producer's latency, so a node's cost here is its RRT occupancy
  and its result delay reaches the solver through its edges.
- **A2 — CUDA-pipeline elementwise operators re-priced at
  `ceil(tile elements / 128)`; scalar and address arithmetic charged zero.**
  Source: the CUDA issue model (4 warps x 32 lanes), not the paper. The dump
  records latency 1 — often 0 — for these operators, which is the extractor's
  warp-parallel hiding convention rather than an issue cost, and is unusable
  from the RRT's point of view: it prices a 64-wide multiply and a
  16384-wide multiply identically. Tile extents are read from the raw dump's
  `result_types`; the curated schema does not carry them, so the raw dump is a
  second input, bound to the curated file by `curation_source.ddg_sha256`.
- **A3 — asynchronous MMA results are tokens, not scalars.** Source: the
  Blackwell programming model. `ttng.tc_gen5_mma` returns
  `!ttg.async.token`, which a naive "not a tensor, therefore scalar" reading
  of A2 would price at zero and erase a 128-cycle tensor-core reservation.
- **A4 — TMEM ports are their own functional unit.** Source: the Blackwell
  hardware description. Already true of the stock `MachineModel` and of
  curation's pipeline assignment, so the regime adds a fail-closed check
  rather than a machine of its own: a curated input that folds TMEM back onto
  CUDA is rejected instead of silently saturating the ALU column.

### B: normalization stack

- **B1 — 10% relative-tolerance clustering.** Source: this reproduction.
  Ascending scan over the positive values, the first value of a cluster is its
  anchor, later values within `anchor * 11/10` join, the representative is the
  members' integer mean rounded half up. Near-but-coprime measured costs
  otherwise pin F: the ZLP must reproduce an accidental ratio such as 133:128
  exactly and pays for it across the whole pool.
- **B2 — a 1/32 floor.** Source: this reproduction. After clustering, entries
  below `max/32` are zeroed. A pool spanning three orders of magnitude cannot
  be represented inside `sum(C') <= 300` at any useful resolution; the floor
  makes that compression an explicit, disclosed decision instead of one the
  ZLP takes implicitly by collapsing whatever it likes.
- **B3 — the paper's own ZLP, unmodified.** `normalize.normalize_costs` is
  imported, not reimplemented. This is the one step in the stack that *is*
  paper content.
- **B4 — a lexicographic second stage maximizing `sum(C')` at `F = F*`.**
  Source: this reproduction, and the load-bearing assumption of the whole
  regime. The paper minimizes F and says nothing about which point of the
  F-optimal face is returned; that face is not a singleton. On the prepared
  `fwd_subtiled` pool it holds both a degenerate point (`sum(C')=16`, 87.5% of
  positive entries at zero) and a rich one (`sum(C')=272`, 16.1%), at the same
  optimal F=348, and SCIP reports the degenerate end. A fresh model is built
  rather than the solved one re-optimized: changing the objective sense on a
  solved PySCIPOpt model does not re-solve the problem being asked.
- **B5 — a per-case normalization budget.** Source: this reproduction. 300 for
  `fwd_subtiled`, `bwd` and `bwd_lr4096`; 150 for `fwd`, whose formula size
  explodes at 300. The paper states one budget for all cases.

### What the regime does and does not recover

Measured on the committed golden curated fixtures, artifacts under
`prepared/`, reproduced by `run_prepared_cases.sh`.

- **Normalization health is recovered.** `fwd_subtiled` collapses 18 of 112
  positive entries (16.1%) with a surviving-value spread of 22.0, against
  104/128 and a single-valued spectrum on the literal path. `bwd` collapses
  1/66. B4 is what does it; B1-B3 alone leave `fwd_subtiled` at `sum(C')=16`
  and 87.5% collapse, indistinguishable from the literal path.
- **The II degeneracy is recovered.** `fwd_subtiled` solves at II=26, L=52
  against the literal II=2, L=5; `bwd` at II=30, L=85. The v6-era anchors
  record II=66 and II=95, so these are the same order of magnitude but not the
  same number — the anchors' II unit depends on their own normalization scale
  and is not expected to reproduce digit for digit.
- **`fwd` is not recovered and was not tuned around.** It stays degenerate at
  both U=150 and U=300 (`sum(C')=7`, 89.4% collapse, spread 1.0, II=3), so B5
  is not the cause: its own F-optimal face is poor at F*=320. Relaxing to
  `F <= 2F*` would give 24.2% collapse at spread 31 and `F <= 4F*` would give
  0%, but an F relaxation is a ninth assumption and is left for the plan to
  decide rather than adopted here.
- **The published *shapes* are not recovered, for one shared reason.**
  `fa4_like` is False on `fwd_subtiled` and `bwd_2wg_pingpong` is False on
  `bwd`; the exact-partition refit is UNSAT across the whole probed window
  `L in [52, 58]` at II=26; and `bwd_lr4096` uses *fewer* active warps than
  `bwd` (24 against 25) where the reported direction is more. All four follow
  from the same fact: the free solutions scatter operations across 18 and 17
  distinct warp sets respectively, where the reported shapes have five and
  two. Figure 6 quantifies `opw` over the machine's 32 physical warps and
  contains nothing that rewards using fewer distinct sets, so the solver has
  no reason to consolidate and does not. The v6-era anchors carry a `warp`
  group-*label* map with 4, 5 and 8 groups rather than a physical warp set,
  which is consistent with a bounded group pool that the literalization
  removed as un-paper-like. Restoring one would be a ninth assumption about
  the *model* rather than the inputs, and is out of this regime's scope as
  specified.
- **The UNSAT ablations hold, with one of the four carrying no weight.**
  Probed with the published first-window protocol, since Algorithm 1 does not
  terminate on a structurally unsatisfiable model. `warps_minus_one` (31
  warps), `warps_half` (16) and `no_cross_warp` are each UNSAT at every one of
  the 21 points in the first window of I=25. `no_subtiling` is UNSAT across
  its whole window too, but that window is a single point at I=2 because the
  `fwd` pool is the degenerate one above: the verdict is real and the evidence
  is empty, and it should not be read as support for the sub-tiling claim
  until `fwd` normalizes.
