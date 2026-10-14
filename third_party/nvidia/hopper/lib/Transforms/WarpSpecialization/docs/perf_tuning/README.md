# AutoWS Perf Tuning Context (GEMM family, B200)

## Purpose

This folder makes the AutoWS GEMM config space precise. For each knob it says
exactly what the knob is, where it applies and what it needs, and it gives the
definitive pruning ruleset: configs that are illegal or impossible (a compile
error, silently ignored, a duplicate config, a hang, a wrong answer, or a
missing required knob value). It also gives the resource math and, per
feature, the mechanism: what the feature changes, relieves and costs. It does
not say which values to use for a problem.

These docs do not explain what a knob or pass is: that lives in the sibling
docs in `../`, and these docs link to them by file and section heading instead
of repeating them.

Scope: the GEMM family (`tl.dot` based mm, addmm, bmm, grouped mm and their
fused epilogues/prologues) on B200, compiled with the Meta WS backend
(`TRITON_USE_META_WS=1`). Persistent kernels only, one CTA per SM, so occupancy
is not modeled. Unflattened loops only (`flatten=False`). For now the leaves of
the tree are AutoWS configs only.

## Points of reference

A candidate is always compared against one of four mutually exclusive
references. Name the reference with every comparison.

| Reference | Definition |
|-----------|------------|
| cuBLAS | The cuBLAS baseline for the same op, shapes, dtypes and layouts. |
| Inductor | Inductor with its Triton AutoWS template forced. TBD: definition pending. |
| AutoWS | The kernel, including any fusion, written purely in Triton and compiled with AutoWS. |
| TLX | The kernel written purely in TLX, possibly translated Triton -> TTGIR -> TLX. |

## Decision tree (skeleton)

1. **Problem description.** Op; dtypes (inputs, accumulator, output);
   M, N, K and batch; operand layouts (row/column major, strides); epilogue
   and fusion (what is applied to the accumulator before the store, and any
   prologue on the operands).
2. **Derived quantities** (per candidate tile), rather than "small" or
   "large". Parameters: `P` = 148 SMs on B200; `ridge` = the arithmetic
   intensity below which a GEMM is memory bound.
   - `T` = output tiles = `ceil(M/BM) * ceil(N/BN) * batch * SPLIT_K`; 2-CTA
     does not change `T` (each CTA still owns one `BM x BN` tile). `W = T/P`
     waves.
   - `E_tail = T / (P * ceil(T/P))`: tail efficiency under static persistence
     (1.0 = no idle SMs in the last wave). `S_prog = ceil(T/P) * BM * BN`: the
     busiest program's work per K step.
   - `k_iters = ceil(K / (BK * SPLIT_K))`, against `num_stages`.
   - `w = 1 - M*N / (ceil(M/BM)*BM * ceil(N/BN)*BN)`: M/N padding waste.
   - `AI_min = 2*M*N*K / bytes(A, B, C)`, each operand read once and the output
     written once; compare with `ridge`.
   - `I_t = BM*BN / (BM + BN')`, with `BN' = BN/2` under 2-CTA and `BN`
     otherwise: FLOP per byte of operand SMEM fill.
   - `d_fit`: the planner's operand depth, closed form in
     [resources.md](resources.md#smem).
3. **Pruning ruleset.** Drop configs that give a compile error, are silently
   ignored, are a duplicate config, hang, give a wrong answer, or miss a
   required knob value.
4. **Resource limits.** Drop configs whose SMEM, TMEM or register demand does
   not fit.
5. **Feature branches and leaves.** The configs that survive, with each
   feature's mechanism in its feature file. Knob priorities (P0, then P1, then
   P2) are in [knobs.md](knobs.md).

## Rule format

Every rule is a row of a **Pruning ruleset** table: a condition, its status,
and a one-line explanation of what happens. Status is one of "compile error",
"silently ignored" (the knob is dropped), "duplicate config" (the config
compiles to the same kernel as another one), "hangs", "wrong answer" or
"requires" (needs another knob value). A cell lists two statuses when runs
differ. The rules are verifiable by running the condition; they carry no
citations. Mechanism sections explain; they do not prune.

## Feature file layout

Every feature file has the same sections, in this order.

1. **Title and scope.** One paragraph: what the feature is, which knobs, and a
   link to the existing doc that explains it.
2. **Knobs.** Table: Knob | Set via | Legal values | Default | Priority |
   Applies when | Prereqs. Priority is P0 (highest), P1 or P2 (lowest).
   "Applies when" is the scope where the knob has an effect, with what happens
   outside it. "Prereqs" lists conditions, or compiler changes and their state,
   that must hold before the knob is usable. Rows go P0, then P1, then P2.
3. **Pruning ruleset.** Table: Condition | Status | Why (one line).
4. **Mechanism.** At most about six bullets: changes -> relieves -> costs,
   with code references (file + symbol) inline.
5. **Interactions.** Links to rows in `interactions.md`.
6. **Hazards.** Links to entries in `hazards.md`.
7. **Not yet established.** Links to rows in [TRACKER.md](TRACKER.md).

## Citation rules

- Mechanism and resource facts cite code by repo-relative file plus symbol
  (function, pass, option, env var or diagnostic string), and docs by file plus
  section heading. Facts that cannot be checked in this tree become TRACKER
  todos or are dropped.
- Pruning-ruleset rows are verifiable by running them and carry no citation.
- No line numbers and no performance numbers. Compiler changes are named only
  in "Prereqs" cells.
- A rule that depends on a bug (hang, wrong answer) is deleted when the bug is
  fixed.

## Out of scope

- Explaining what knobs and passes do (see `../`, starting at
  [Overview.md](../Overview.md)).
- Which values to use for a given problem.
- Ops outside the GEMM family, GPUs other than B200, occupancy effects, and
  flattened loops.
- Leaves outside AutoWS (cuBLAS or non-WS configs), for now; tracked in
  [TRACKER.md](TRACKER.md).

## Files

The planned files and their status are listed in [TRACKER.md](TRACKER.md).
