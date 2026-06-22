# Variant Manifest Template

Use this template for TTGIR ablation runs that need more than ad hoc notes.
Keep entries short and update `result` immediately after each run.

```yaml
baseline:
  ttgir: path/to/baseline.ttgir
  command: "TRITON_ALWAYS_COMPILE=1 ..."
  target_signal: "compile assertion / crash / hang / wrong result / diagnostic"
  arch: "sm90"

variants:
  - id: v001
    ttgir: path/to/v001.ttgir
    hypothesis: "Remove unrelated epilogue math; partition failure should remain."
    edit_summary: "Replaced final add/mul chain with existing accumulator value."
    oracle_mode: preserve
    expected_behavior: "Same output as baseline reference and same failure signal."
    run_command: "TRITON_ALWAYS_COMPILE=1 ..."
    result: "pending"
    notes: ""
```

## Oracle Modes

- `preserve`: original expected output still applies.
- `replace`: expected output must be updated to the edited IR semantics.
- `relax`: assertion/crash/hang/diagnostic is the oracle.
- `observe`: exploratory only; convert to another mode before using as evidence.

## Result Vocabulary

- `survives`: target signal remains.
- `gone`: target signal disappears and the run otherwise succeeds.
- `changed`: a different failure appears.
- `invalid-ir`: parser/verifier rejects the edited TTGIR before the target path.
- `inconclusive`: run was interrupted, GPU unavailable, or timeout handling failed.
