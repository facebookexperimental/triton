---
description: An agent-driven runbook for validating each example case's ddg.json against its pre_modulo.ttgir. Instead of a brittle text parser, an LLM agent reads both files and compares them semantically, then emits a per-case PASS/FAIL table and the exact failing check for any case that fails.
---

# DDG Validation Harness (agent-driven)

Confirms that the `ddg.json` dumped for each example case (see
`generating_ddg_json.md`) is a faithful encoding of that case's
`pre_modulo.ttgir`. The harness answers one question per case: *was this
`ddg.json` generated correctly from its TTGIR?*

**This is an agent task, not a script.** A regex/text parser is too brittle for
real TTGIR — it trips over generic-form ops (`"tt.reduce"(...) ({ ... })`),
`^bb0` block arguments, type aliases (`#shared` → `#ttg.nvmma_shared<…>`),
TensorDescriptor argument flattening, multi-line op/signature formatting, and
computed (non-constant) loop bounds. An LLM reads both files and compares them
*semantically*, handling all of the above naturally. The trade-off: it is
non-deterministic, slower, and token-costly, so use it as an on-demand /
pre-commit check, **not** a cheap CI gate.

## Scope & inputs

Process every `caseN_*` directory under `examples/`. A case participates iff it
contains both:

- `*pre_modulo.ttgir` — the pre-modulo TTGIR (filename may be prefixed, e.g.
  `pre_modulo.ttgir`, `fa_fwd_nows_pre_modulo.ttgir`,
  `addmm_bias_pre_modulo.ttgir`, `layernorm_fwd_pre_modulo.ttgir`).
- `ddg.json` — the dump under test.

A case missing either file is reported as `SKIP` (not `FAIL`). If a case has a
`*pre_modulo.ttgir` but **no `ddg.json`**, generate it first by following
`generating_ddg_json.md` (build `triton-opt`, then run the
`TRITON_MODULO_DUMP_DDG` dump on that case's `pre_modulo.ttgir`), then re-run
this harness.

## How to run it

Read both files for a case (`Read` tool), work through the check catalog below,
and record PASS/FAIL + any concrete discrepancies. Cases are independent, so for
speed **spawn one subagent per case** (e.g. via the Agent tool / a workflow),
each given: the two file paths, this check catalog, and the output contract.
Then aggregate the rows into one table and concatenate the per-case error
blocks. Running inline (one case at a time in the main thread) is also fine for a
single case.

Per-case agent prompt, in essence:
> Read `<case>/<pre_modulo>.ttgir` and `<case>/ddg.json`. Decide whether the
> ddg.json faithfully encodes the TTGIR using checks A–F from
> `ddg_validation_harness.md`. Return a verdict (`PASS`/`FAIL`), the table-row
> fields, and for each failed check a line `[<id> <group>] expected vs actual`.

## How the agent should read the two files

- **Opaque op ids.** `ddg.json` op ids (`op_88510686050096`, …) are
  MLIR-pointer-derived and do **not** appear in the TTGIR. Do **not** try to map
  them by id. Map by *role*: kernel signature, op kinds within a scope, loop
  structure, and def-use relations.
- **Scopes.** Each `ops` entry has `scope` = `"function"` or `"loop:<id>"`.
  Group ddg ops by scope and compare each group against the corresponding region
  of the TTGIR (function body vs each `scf.for` body).
- **Loop identity.** ddg `loop_id`s follow the scheduler's order (often
  `[inner, outer]`), which is **not** TTGIR source order. Match a ddg loop to a
  TTGIR `scf.for` by structure — induction-var type, whether it nests another
  loop (`is_outer`), and its body's op-kind multiset — not by position.
- **Normalize types.** TTGIR arg/result types carry attribute dicts
  (`i32 {tt.divisibility = 16 : i32}`) and alias names (`#shared`). The dump
  strips the attrs and expands aliases (`#shared` →
  `#ttg.nvmma_shared<…>`). Compare modulo these: drop trailing `{…}` attr
  dicts, resolve `#alias` via the definitions at the top of the `.ttgir`, and
  map pointers `!tt.ptr<f16>` → `*f16`.
- **Descriptor flattening / name disambiguation.** TensorDescriptor params can
  appear as several params sharing one `loc("name")`. The dump keeps the first
  occurrence's bare name and suffixes the rest `_0, _1, …`. Expect that pattern;
  don't flag it as a mismatch.
- **Generic-form & regions.** Ops may be printed generically
  (`"tt.reduce"(%x) <{…}> ({ ^bb0(…): … })`). Count the ops inside such regions
  too; they appear in the `ops` table at `"function"` scope (they are not in a
  scheduled loop). Not every in-loop op becomes a DDG *node* (see E3).

## What "correct" means — the check catalog

Examples are from `case1_simple_gemm`. Each check has a stable id used in the
error report.

### A. Document self-consistency (ddg.json only)
- **A1 schema** — `schema_version == "ddg-0.1"`.
- **A2 config** — `schedule_algo` non-empty; `smem_budget_bytes` /
  `tmem_budget_bytes` positive.
- **A3 min-ii** — each loop's `min_ii == max(res_mii, rec_mii, max latency over
  super-nodes)`. The super-node term is the super-node node's **`latency`** field
  (its full pipelined cost, e.g. inner_ii × inner trip count), **not** its
  `inner_ii` — see `computeMinII` in `DataDependenceGraph.cpp`. For a loop with
  no super-node it reduces to `max(res_mii, rec_mii)`.
- **A4 ref integrity** — every operand `{"op": …}`, node `op_ref`, and edge
  `src`/`dst` resolves.

### B. Kernel identity & signature (vs TTGIR)
- **B1 name** — `kernel.name` == the `tt.func @<name>` (`gemm_kernel`).
- **B2 arg count/order** — after accounting for descriptor flattening.
- **B3 arg names** — match the params' `loc("name")`, with the `_0/_1` dedup
  rule applied.
- **B4 arg types** — match under the type normalization above
  (`!tt.ptr<f16>`→`*f16`, alias-expanded, attr-stripped).

### C. Op-table fidelity (vs TTGIR)
- **C1 count** — `len(ops)` equals the total op count of the `tt.func` body
  (count generic-form and in-region ops; case1: 33).
- **C2 per-scope kinds** — for each scope (function, and each matched loop), the
  multiset of op `kind`s matches the corresponding TTGIR region. (Catches a
  missing/extra/wrong-kind op without depending on exact emission order.)
- **C3 scope assignment** — ops inside an `scf.for` body get `"loop:<id>"`;
  everything else (including the `scf.for` op itself and ops in `tt.reduce`/
  `scf.if` regions) gets `"function"`.

### D. Loop structure (vs TTGIR)
- **D1 loop count** — `len(loops)` == number of `scf.for` ops (case1: 1;
  persistent cases: 2).
- **D2 bounds** — each loop's `lower_bound`/`upper_bound`/`step` match the
  `scf.for` operands. When a bound is a constant or a kernel arg, check the
  value/name (case1: `0` / `%K` / `64`). When it is a *computed* SSA value
  (e.g. persistent `num_tiles`), the dump encodes it as an `{"op": …}` ref —
  verify it points at the right defining op rather than a literal value.
- **D3 induction var** — `induction_var.type` matches the IV type (`i32`).
- **D4 trip count** — constant bounds ⇒ `trip_count == ceildiv(ub-lb, step)` and
  `trip_count_estimated == false`; any dynamic bound (case1 `ub == %K`) ⇒
  `trip_count_estimated == true`.
- **D5 is_outer** — `true` iff that `scf.for` body contains a nested `scf.for`.

### E. DDG nodes (vs op table & TTGIR)
- **E1 node→op kind** — every node's `op_kind == ops[op_ref].kind`.
- **E2 node scope** — `ops[op_ref].scope == "loop:<this loop_id>"` (a super-node
  references the inner `scf.for`).
- **E3 node coverage** — node `op_ref`s are exactly the loop-scope *schedulable*
  ops: the loop body minus the terminator (`scf.yield`) and minus pure
  scalar/index bookkeeping the scheduler doesn't model. Case1: 6 loop ops → 5
  nodes (yield excluded). Treat extra uncovered ops as a note unless a clearly
  schedulable op (load/alloc/mma/dot) is missing a node — that is a fail.
- **E4 super-node fields** — `is_super_node` ⇒ `inner_ii`/`prologue_latency`
  present; otherwise absent.

### F. DDG edges (vs TTGIR def-use)
- **F1 kind/distance** — `kind == "loop_carried"` iff `distance > 0`, else
  `"data"`.
- **F2 data-edge def-use** — for a `data` edge `src→dst`, the src node's op
  result is an operand of the dst node's op (check via the dst op's `operands`).
  Case1: `0→1` (load→local_alloc), `1→4`/`3→4` (local_alloc→mma).
- **F3 loop-carried def-use** — for a `loop_carried` edge with `distance d`, the
  src op result reaches the dst op across `d` iterations via the `scf.for`
  iter-args / `scf.yield`. Case1: `4→4` (mma accumulator, distance 1).

### Cost-model values are authoritative
Do **not** try to re-derive `latency` / `self_latency` / `occupancy` /
`min_warps` or the MinII *values* from the IR — those come from the scheduler's
cost model and are taken as given (only their internal consistency, A3, is
checked). Validating those numbers belongs to the modulo scheduler's own tests.

## Output contract

### Results table (one auto-discovered row per case)

Emit **one row per `caseN_*` directory found under `examples/`** — do not
hardcode a case list; whatever set is discovered at run time is what gets
reported, so new cases appear automatically with no edit to this doc. The rows
below are illustrative format only (columns, alignment, summary line), not a
fixed roster:

```
DDG Validation Harness
======================================================================================
case                     kernel                               loops  ops  nodes/edges  result
--------------------------------------------------------------------------------------
case1_simple_gemm        gemm_kernel                          1      33   5/5          PASS
case2_persistent_gemm    matmul_kernel_tma_persistent_simple  2      30   30/29        PASS
…                        (one row per discovered caseN_* directory)            …
--------------------------------------------------------------------------------------
<N> cases: <P> PASS, <F> FAIL, <S> SKIP
```

### Error details (after the table, only for FAIL/SKIP)

One line per failed check: `[<check_id> <group>] <expected> vs <ddg.json had>`,
with enough locator context (scope, op kind/position, loc name, node/edge) to
fix it. Illustrative shapes (not real failures of the current files):

```
FAIL  case6_layernorm  (2 checks failed)
  [C2 op-table] function scope: ttgir has 1 arith.mulf not present in ddg.json (loc layernorm_fwd_nows.py:50:20)
  [F1 edges] loop 0 edge {src:7,dst:9}: kind="data" but distance=1 (distance>0 must be "loop_carried")

SKIP  case4_foo  (no ddg.json — generate it first per generating_ddg_json.md)
```

Other representative messages:
- **B3** `arg #6 name: ttgir loc "stride_am" (deduped) vs ddg.json "stride_xx"`
- **C1** `op count: ttgir 33 vs ddg.json 32 (loop:0 missing a ttg.local_alloc)`
- **D2** `loop 0 upper_bound: ttgir %num_tiles (arith.muli) vs ddg.json {const:1024}`
- **E3** `loop 0: tt.descriptor_load (loc …:100) has no DDG node`
- **F2** `loop 0 edge {src:1,dst:4} (data): local_alloc result is not an operand of tc_gen5_mma`
