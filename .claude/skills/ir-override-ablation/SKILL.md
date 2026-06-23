---
name: ir-override-ablation
description: >
  Design and run Triton TTGIR debugging ablations using ir_override. Use when
  reducing a provided or dumped TTGIR, trying user-provided or agent-generated
  ablation/oblation ideas, updating a test harness around ir_override, or
  preserving a compile/runtime failure while simplifying IR to expose a
  fundamental compiler or lowering gap.
---

# IR Override Ablation

Use this skill to turn a noisy TTGIR reproducer into a smaller, evidence-backed
set of IR variants. The goal is not only to simplify the IR; it is to preserve
the debugging signal and keep the test oracle honest when edits intentionally
change kernel semantics.

## Setup

1. Identify the target signal before editing:
   - compile failure, pass assertion, crash, hang, wrong result, or diagnostic
   - the exact command and environment that reproduces it
   - the minimum architecture requirement, for example `sm90` or `sm100`
2. Get a baseline TTGIR:
   - Use the provided `.ttgir` directly when available.
   - Otherwise dump it with:
     ```bash
     TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=<dump_dir> <run_command>
     ```
3. Decide the override mechanism:
   - Prefer `triton.Config(..., ir_override="<variant.ttgir>")` when the
     harness already uses configs/autotune or can be cheaply adapted.
   - Use `TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=<dump_dir>` when testing
     against the dumped override directory layout is simpler.

## Ablation Loop

1. Build or accept a list of ideas. Each idea should state the hypothesis and
   the expected effect on the target signal.
2. Copy the baseline TTGIR to a named variant file before editing.
3. Classify the test oracle before running the variant:
   - `preserve`: the original numeric/reference output must still match.
   - `replace`: the edit intentionally changes output values; update the
     expected output to match the new semantics.
   - `relax`: success is reproducing a compile failure, crash, hang, or
     diagnostic; numeric equality is not the primary oracle.
   - `observe`: temporary exploration only; do not treat this as final evidence.
4. Update the harness when the oracle is not `preserve`.
   - Example: if a store operand is replaced by a constant tensor, expected
     output must become that constant over the same mask/shape, not the original
     reference computation.
   - If an edit removes a store, expected output may need to assert unchanged
     sentinel values instead of comparing to the old result.
5. Run exactly the command recorded for the variant with
   `TRITON_ALWAYS_COMPILE=1` to avoid cache reuse.
6. Record whether the target signal survives, weakens, disappears, or changes
   into a different failure.
7. Prefer single-concept edits. When multiple edits are needed, stage them as
   separate variants so the final reproducer has a defensible chain of evidence.

## Harness Rules

- Keep baseline and variant runs side by side until the final reproducer is
  chosen.
- Make oracle-changing behavior explicit in code or comments near the assertion.
- For wrong-result bugs, keep at least one shape that fails and one nearby shape
  that passes when practical.
- For hangs, set a short external timeout and run `third_party/tlx/killgpu.sh`
  if the GPU process survives longer than expected.
- Do not run performance benchmarks unless the user explicitly asks.

## Reporting

Return a compact result matrix:

```text
variant | hypothesis | oracle | result | target signal | kept?
```

Then state the smallest surviving reproducer, the concepts still present in the
IR, and the concepts successfully removed.

## References

- Use `references/variant-manifest-template.md` when tracking more than a few
  ideas or when handing work to another agent.
- Use `references/harness-template.md` when creating or adapting the Python
  harness around `ir_override`.
