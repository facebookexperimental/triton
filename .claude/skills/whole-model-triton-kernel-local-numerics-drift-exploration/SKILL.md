---
name: whole-model-triton-kernel-local-numerics-drift-exploration
description: Compare Triton, TLX, and inductor kernel numerics across A/B compiler builds with per-config isolation tests and grouped impact reporting.
---

# Whole Model Triton Kernel Local Numerics Drift Exploration

Use this skill when asked to evaluate numerics risk of a Triton, TLX, or inductor change for a specific model.

## 1. Request model context

Ask for a doc or pointer that explains how to find information about the model, then collect:

- Model name, entry point, and repro command.
- Precision, dtypes, and representative shapes.
- How the model is compiled (inductor settings, Triton/TLX path, custom kernels).
- Where generated code or compile traces live.

Do not proceed on guesses. If the doc is missing fields above, ask follow-up questions.

## 2. Determine the kernel suite

From the model context, enumerate every Triton, TLX, or inductor-generated kernel the model produces:

- Inspect inductor output code, compile logs, and autotune artifacts.
- Record kernel name, source (Triton/TLX/inductor), signature, shapes, dtypes, and launch parameters.
- This list is the test suite. Keep it explicit and reviewable.

## 3. Specify the A/B

Define baseline (A) and candidate (B) as Triton version plus commit hash:

- Record `version` and `commit hash` for each side.
- When swapping Triton versions, identify the commit hash from the version's build metadata, pinned dependency, or source checkout and record how it was determined.
- State build or environment differences beyond the compiler change, if any.

## 4. Run each kernel in isolation

For each kernel in the suite:

- Build an isolation harness using model-realistic shapes, dtypes, strides, and value ranges from step 1.
- Reuse existing kernel-level accuracy tests where they exist.
- Run A and B on identical inputs and compare outputs.
- If a kernel has no accuracy test or the existing tests miss shapes, dtypes, or boundary conditions the model uses, add the missing test coverage before concluding.

## 5. Test every autotuning config

Do not test only the autotuner winner:

- Enumerate all autotuning configs for each kernel under both A and B.
- Run each config individually in isolation.
- Record per-config outputs and diffs so a config-specific regression is not hidden by winner selection.

## 6. Group results

Group every kernel into exactly one bucket:

1. Bitwise equivalent: A and B outputs match exactly across all tested configs.
2. Small numerics change, low probability of impacting model training: differences are within floating-point reassociation or rounding noise, small in magnitude and frequency, with no NaN/Inf divergence or distribution shift.
3. Large numerics change, possible compiler bug or training risk: large max/mean error, systematic bias, NaN/Inf mismatch, shape- or config-dependent blowups, or any pattern inconsistent with benign reassociation.

Report per-kernel evidence: comparison metric, worst config, failing inputs, and bucket rationale. Flag bucket 3 kernels with repro commands and suspected scope (single kernel, autotune config, dtype, or shape class).
