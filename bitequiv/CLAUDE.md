# Bitwise Equivalence & Triton Autotuning — Project Guide

> Project-scoped instructions for the `bitequiv/` intern project. This file is
> loaded **in addition to** the repo-root `CLAUDE.md` (Triton codebase
> architecture, build/test/format commands, path-scoped rules). Don't duplicate
> that here — this file is about *this project's goals, conventions, and state*.

> **SESSION BOOTSTRAP:** at the start of any session working on this project,
> read `bitequiv/PROGRESS.md` first — it holds current state (phase, milestone
> status, recent activity, key facts, open questions) and lists the docs to load.
> Update it at session end per its UPDATE PROTOCOL.

## 1. What this project is

Teach the Triton/TLX **autotuner to become constraint-aware** so it can both
*enforce bitwise equivalence* and *pick higher-performance configs given that
constraint*. Two intertwined threads:

1. **Bitwise equivalence** — two kernels are bitwise equivalent if, for the same
   input, they produce the same output down to every bit. Hard because FP math
   is non-associative, so we must enforce "FP ops happen in the same order."
2. **Constraint-aware autotuning** — prune "bad configs" (correctness bugs,
   non-equivalent reductions) and let the compiler optimize codegen *given* a
   declared constraint (e.g. ordered reduction → choose the best layout for it).

**Why it matters:** bitwise-exact numerics let customers skip 1000s-of-GPU,
week-long ladder tests; also needed for RL, batch-invariant inference, LLM
training. Today this is achieved by freezing layouts and disabling tuning (perf
penalty). We want to *prove/enforce* equivalence and recover the tuning freedom.

## 2. Roles & repos

- **Intern:** Ziteng Yang (IC4 PhD SWE).  **Intern manager:** Nick Riasanovsky.
  **Team manager:** Alexey Loginov.  **Peers:** Paul Zhang, Warren Deng.
  **XFN:** Elias Ellison, Jason Ansel (PyTorch compile).
- **Repo:** `facebookexperimental/triton` (this fork). Mirrored into
  `fbsource/third-party/triton/beta` ("inside fbcode", built with BUCK).
- Hardware: H100 and B200 (multiple Nvidia GPUs provisioned).

## 3. Milestones & timeline (12 weeks)

| Wk | Milestone | Focus |
|----|-----------|-------|
| 1–2 | **Starter** | Ramp on Triton compiler pipeline + autotuner. Autotune **pruning** (by pattern / IR / PTX artifact). Correctness-check hook into the autotuner. Targets: TMEM_LOAD filter bug, ttgir-based AutoWS example, PTX-based vectorization example. Land 10–15 example kernels as onboarding tutorials + doc gaps for numerics-modifying passes. |
| 3–5 | **M1: Reduction equivalence detect/enforce** | Standalone PTX analysis tooling to determine reduction ordering; experiment framework; autotuner integration (only keep bitwise-equiv configs). Cover multi-tensor / multi-dim reductions, FMA/sum/prod, varying block sizes. Example suite + design doc + progress post. **Must finish before midpoint (wk 5).** |
| 6–8 | **M2: Reduction layout optimization** | Compiler pass to pick the *best layout* given the ordered-reduction constraint (vs today's "same result regardless of layout"). Experiment framework across *all* configs (not just winner). NCU profiling. Target: close ordered-vs-unordered gap by up to 50% on worst configs. |
| 8–10 | **M3: GEMM equivalence detect/enforce** | Extend analysis to MMA/wgmma/tcgen05. Lower more configs to bitwise-equiv MMA (e.g. BLOCK_N=128 → two N=64 instrs). MMA constraint representation persisting across passes. Goal: bitwise-equiv to **cuBLAS** on ~5 shapes (H100 + B200). |
| 10+ | **Stretch** | M4 TC instruction optimization (data partitioning, num-stages, async ops) · M5 PyTorch/torch.compile fusion analysis · M6 AMD (AMDGCN/MFMA) · M7 GEMM+LayerNorm fusion. Order is independent. |
| 11–12 | **Wrap-up** | Polish, measurements, docs/runbooks/handoff, final presentation. |

## 4. Mental model (the core technical truth)

The reduction **tree shape** determines the FP result. Tree shape is an emergent
property of (a) which thread/lane holds which element (the **layout**) and (b) the
shuffle offset sequence in PTX. Same PTX + different layout → different pairing →
different rounding → not bitwise equivalent. See
`knowledge-base/tree-reduction-in-ptx-and-triton.md` for the full walkthrough.

The verification primitive is: *parse PTX → trace which registers hold which
input elements → reconstruct the add/fma tree → compare trees structurally.*

## 5. What already exists vs. what we build (don't reinvent)

**Already exists (search/reuse before building):**
- `inner_tree` / `reduction_ordering` — full compiler path Python→MLIR→PTX
  (count-up warp shuffles + balanced within-thread tree). Diff D100027220.
- `TRITON_STRICT_REDUCTION_ORDERING` env var (D101872700).
- `STABLE_REDUCTION` in layer_norm — production manual-sequential workaround
  (D104785121), motivated by a real 5.24% NE gap.
- TritonParse — IR/PTX parsing + multi-level diff, **text-level only** (no
  semantic tree analysis).
- Autotuner hooks: `early_config_prune`, `restore_value`, per-config IR/PTX dump,
  `CompiledKernel.metadata`.
- `triton_repro_bitwise.py` (Paul Zhang, D100024902); FBGEMM correctness-vs-perf
  pruning split (D75976113).

**Built so far (M1, in-tree — reuse, don't reinvent):**
- **TTGIR checker** — MLIR-native data-layout reduction-order checker
  (`bitequiv/ttgir_reduction.py` → C++ `lib/Analysis/ReductionOrder.cpp` via
  `toLinearLayout`; autotuner API in `bitequiv/equivalence_ttgir.py`). Fixes the
  reduction *association order*; **blind to FMA contraction** (layout-only).
- **PTX checker** — its sibling (`bitequiv/ptx_reduction.py`), reconstructs the
  reduction tree from PTX, so it *also* catches FMA contraction (below TTGIR).
- **Evaluation framework** — `bitequiv/evaluation/` measures either checker against an
  empirical fuzzer; pick the checker with `--checker module:function` and the IR it
  reads with `--artifact ptx|ttgir`.

**Still to build (first-ever):** reduction layout-optimization pass (M2); MMA
constraint representation + lowering (M3).

## 6. ⚠️ Critical guardrail for AI-assisted work

**Bitwise equivalence is foreign to the AI's optimization instincts.** When asked
to improve performance, an agent (including me) may silently undo an ordering
constraint and "succeed" on speed while breaking correctness. Therefore:

- **Correctness gates performance — always.** Never accept a perf change without
  re-running the bitwise-equivalence check.
- Build/extend **testing frameworks** that fail loudly when equivalence breaks,
  and treat them as the source of truth, not the diff's plausibility.
- When a constraint (ordered reduction, MMA N-restriction) is in play, state it
  explicitly in the prompt and in code comments so it survives refactors.
- Prefer **offline LLM analysis** or an explicit online "developer mode" for
  ambiguous equivalence questions; online-by-default is too slow.

## 7. Build / test / run (project-specific)

Repo-root `CLAUDE.md` + `.claude/rules/*` are authoritative. Quick reminders:
- C++/MLIR changes need rebuild: `pip install -e . --no-build-isolation`.
- Python-only changes: no rebuild.
- Always `pre-commit run --all` before considering work done.
- IR/PTX inspection: load the `ir-debugging` skill (`TRITON_KERNEL_DUMP`,
  `MLIR_ENABLE_DUMP`, `LLVM_IR_ENABLE_DUMP`, `TRITON_DUMP_PTXAS_LOG`).
- **Never run performance/benchmark tests unless explicitly asked.** Use the
  `kernel-perf-testing` skill when you are.
- If any test hangs for minutes, run `third_party/tlx/killgpu.sh`.

## 8. Knowledge base & session-summary conventions

All durable project knowledge lives under `bitequiv/knowledge-base/`:
- `knowledge-base/*.md` — teaching/reference docs (one concept per file, e.g.
  `tree-reduction-in-ptx-and-triton.md`).
- `knowledge-base/claude-chatting/*.md` — dated chat/work summaries.

**When the user says "summary today's chat" / "summary today's work":** write
`knowledge-base/claude-chatting/YYYY-MM-DD-<short-topic>.md` following the
existing template (see `2026-06-02-project-overview-and-ptx-reduction.md`):

```
# YYYY-MM-DD — <Title>
## What we covered          (numbered subsections, one per theme)
## Key diffs and file paths referenced   (D-numbers, absolute paths)
## Documents created
## Open questions for follow-up
```

Keep it factual and skimmable. Record D-numbers, file paths, decisions, and
open questions — these are the things future sessions can't re-derive. See
`knowledge-base/ai-workflow.md` for the full workflow + git practice.

## 9. Open questions (carry forward)

- Exact shape of the TMEM_LOAD accuracy bug?
- What was decided at the Nvidia collaboration day?
- PTX analysis: purely static, LLM-assisted, or hybrid?
- How does TLX's IR surface differ from standard Triton for this analysis?
- cuBLAS reduction/MMA ordering — needs experimental determination (M3).
