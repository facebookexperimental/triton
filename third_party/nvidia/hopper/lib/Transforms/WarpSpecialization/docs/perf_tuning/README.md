# AutoWS Perf Tuning Context (GEMM family, B200)

## Purpose

This folder is context for pruning the AutoWS config space. Given one GEMM
problem on B200, it gives a structure that prunes every infeasible config
combination ("Never try (any problem)") and every inadvisable one for that
problem ("When to try it (depends on the problem)"). The candidate set is
whatever survives, of any size.

These docs say **when** to use which knob values. They do not explain what a
knob or pass is: that lives in the sibling docs in `../`, and these docs link
to them by file and section heading instead of repeating them.

Scope: the GEMM family (`tl.dot` based mm, addmm, bmm, grouped mm and their
fused epilogues/prologues) on B200, compiled with the Meta WS backend
(`TRITON_USE_META_WS=1`). For now the leaves of the tree are AutoWS configs
only.

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

Each step narrows the candidate set. Later PRs fill in one branch each; see
[TRACKER.md](TRACKER.md).

1. **Problem description.** Op; dtypes (inputs, accumulator, output);
   M, N, K and batch; operand layouts (row/column major, strides); epilogue
   and fusion (what is applied to the accumulator before the store, and any
   prologue on the operands).
2. **Regime, from derived quantities** rather than "small" or "large":
   - output tile count `ceil(M/BLOCK_M) * ceil(N/BLOCK_N) * batch` against the
     SM count (148 on B200): fewer tiles than SMs, about one wave, or many
     waves;
   - K iterations `ceil(K/BLOCK_K)` against `num_stages`: whether the
     pipeline ever reaches steady state;
   - arithmetic intensity `2*M*N*K / bytes(A, B, C)`: memory bound or compute
     bound.
3. **Never try (any problem).** Drop configs that give a compile error, are
   silently ignored, are a duplicate config, hang, or give a wrong answer.
4. **Resource limits.** Drop configs whose SMEM, TMEM or register demand does
   not fit.
5. **Feature branches.** One branch per feature (tiles and stages, 2-CTA, data
   partitioning, epilogue subtiling, TMA loads/stores, persistence, ...).
6. **Fusion branches.** What the epilogue or prologue fusion adds or rules out.
7. **Leaves.** The AutoWS configs that survive every step above, however
   many there are.

## Rule format

Every rule is one of two kinds.

- **Never try (any problem).** The config gives a compile error, is silently
  ignored, is a duplicate config, hangs, gives a wrong answer, or requires
  another knob value, whatever the problem. State the condition and the code
  that enforces or exhibits it.
- **When to try it (depends on the problem).** Derived from the mechanism:
  state what the knob changes, what that relieves and what it costs, and the
  problems for which that makes it worth trying or not.

```
### <short rule name>
- Kind: never try | when to try it
- When: <condition on the problem description or derived quantities>
- Rule: <keep / drop / prefer which knob values>
- Why: <never try: what goes wrong. When to try it: changes -> relieves -> costs>
- Code: <file + symbol>; Doc: <file + section heading>
```

## Feature file layout

Every feature file has the same sections, in this order.

1. **Title and scope.** One paragraph.
2. **Knobs.** Table: knob | how it is set | one line | link to the existing
   doc section.
3. **Never try (any problem).** Table: condition | status | why (one line) |
   code ref (file + symbol). Status is one of "compile error", "silently
   ignored", "duplicate config", "hangs", "wrong answer" or "requires" (needs
   another knob value). "(reproduced)" after a status means it was also
   confirmed by running the kernel. A cell lists two statuses when runs differ
   (e.g. "wrong answer (reproduced); hangs (reproduced)"). "silently ignored"
   means the knob is dropped; "duplicate config" means the config compiles to
   the same kernel as another one.
4. **Mechanism.** What the feature changes -> what it relieves -> what it
   costs, with code refs.
5. **When to try it (depends on the problem).** Derived from the mechanism, written as
   `IF <regime / other knobs> THEN skip | only try | prefer <values> BECAUSE
   <mechanism>`. "skip" and "only try" remove configs; "prefer" only orders
   them and removes nothing. A rule that does something else says so.
6. **Interactions.** Links to rows in `interactions.md`.
7. **Hazards.** Links to entries in `hazards.md`.
8. **Not yet established.** Links to rows in [TRACKER.md](TRACKER.md).

## Citation rules

- Every rule rests on a fact that can be checked in this tree. Facts that
  cannot be checked in code become TRACKER todos or are dropped.
- Cite code by repo-relative file plus symbol (function, pass, option, env
  var or diagnostic string). Cite docs by file plus section heading.
- No line numbers, no PR or issue numbers, and no dated performance numbers.
  State mechanisms ("X relieves Y at the cost of Z"), not results ("X wins").
- A rule that depends on a bug (hang, wrong answer) is deleted when the bug is
  fixed.

## Out of scope

- Explaining what knobs and passes do (see `../`, starting at
  [Overview.md](../Overview.md)).
- Measurement methodology and search or hill-climbing procedures.
- Ops outside the GEMM family, and GPUs other than B200.
- Leaves outside AutoWS (cuBLAS or non-WS configs), for now; tracked in
  [TRACKER.md](TRACKER.md).

## Files

The planned files and their status are listed in [TRACKER.md](TRACKER.md).
