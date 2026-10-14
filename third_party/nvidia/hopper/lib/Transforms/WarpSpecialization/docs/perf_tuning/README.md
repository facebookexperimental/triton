# AutoWS Perf Tuning Context (GEMM family, B200)

## Purpose

This folder is context an agent (or a human) uses to explore the AutoWS config
space for one GEMM problem on B200. It gives three things:

- the **math** that relates a problem and a config to resources and work
  (derived quantities below, resource arithmetic in [resources.md](resources.md));
- the **Never try (any problem)** constraints: configs that give a compile
  error, are silently ignored, are duplicate configs, hang, give a wrong
  answer, or require another knob value;
- the **mechanism** of each feature: what it changes, relieves and costs.

The docs do not prescribe which values to use for a given problem. The agent
explores what survives the constraints, including knob uses the docs do not
anticipate; findings from runs are folded back in later.

These docs do not explain what a knob or pass is: that lives in the sibling
docs in `../`, and these docs link to them by file and section heading instead
of repeating them.

Scope: the GEMM family (`tl.dot` based mm, addmm, bmm, grouped mm and their
fused epilogues/prologues) on B200, compiled with the Meta WS backend
(`TRITON_USE_META_WS=1`). Persistent kernels only, one CTA per SM, so occupancy
is not modeled. For now the leaves of the tree are AutoWS configs only.

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
   intensity below which a GEMM is memory bound (a parameter the agent sets).
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
3. **Never try (any problem).** Drop configs that give a compile error, are
   silently ignored, are a duplicate config, hang, or give a wrong answer.
4. **Resource limits.** Drop configs whose SMEM, TMEM or register demand does
   not fit.
5. **Explore** the surviving configs, using the mechanisms in the feature
   files (tiles and stages, 2-CTA, data partitioning, epilogue subtiling, TMA
   loads/stores, persistence, ...) and the fusion's needs. Knob priorities
   (P1 first, then P1 + P2) are in the "Knob coverage" table of
   [TRACKER.md](TRACKER.md).

## Rule format

Every rule is a **Never try (any problem)** row: the config gives a compile
error, is silently ignored, is a duplicate config, hangs, gives a wrong answer,
or requires another knob value, whatever the problem. It states the condition
and the code that enforces or exhibits it. Mechanism sections explain; they do
not prune.

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
   costs, with code refs and the arithmetic. Concise.
5. **Interactions.** Links to rows in `interactions.md`.
6. **Hazards.** Links to entries in `hazards.md`.
7. **Not yet established.** Links to rows in [TRACKER.md](TRACKER.md).

## Citation rules

- Every fact can be checked in this tree at a file + symbol (a diagnostic, a
  planner formula, a pass gate). "(reproduced)" may be added when a run also
  showed it. Anything else becomes a TRACKER todo or is dropped.
- Cite code by repo-relative file plus symbol (function, pass, option, env
  var or diagnostic string). Cite docs by file plus section heading.
- No line numbers, no PR or issue numbers, and no performance numbers.
- A rule that depends on a bug (hang, wrong answer) is deleted when the bug is
  fixed.

## Out of scope

- Explaining what knobs and passes do (see `../`, starting at
  [Overview.md](../Overview.md)).
- Prescribing which values to try for a problem; measurement methodology and
  search procedures.
- Ops outside the GEMM family, GPUs other than B200, and occupancy
  (non-persistent multi-CTA-per-SM) effects.
- Leaves outside AutoWS (cuBLAS or non-WS configs), for now; tracked in
  [TRACKER.md](TRACKER.md).

## Files

The planned files and their status are listed in [TRACKER.md](TRACKER.md).
