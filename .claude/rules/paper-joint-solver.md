---
globs:
  - "third_party/tlx/tools/paper_joint_solver/**"
---

# Paper Joint Solver

Verbatim reproduction of the paper's joint SWP+WS solver. The degenerate result
is the finding, not a regression — read `DEVIATIONS.md` before "fixing"
anything the solver reports.

Python-only: no rebuild needed.

## Testing

```bash
buck2 test fbsource//third-party/triton/beta/triton:py_paper_joint_solver_test
```

The targets (`paper-joint-solver-py`, `py_paper_joint_solver_test`) are defined
in `third-party/triton/beta/BUCK.template`. `BUCK` is generated — after editing
the template, regenerate with:

```bash
buck run fbcode//triton/tools/reactor:reactor -- buckify triton beta
```

### Buck diverges from the pinned host (measured 2026-08-11)

- Buck's `pyscipopt` ships **SCIP 9.02**; `test_scip_suite_version_pin` pins
  10.0 and fails under buck. Solver output under buck is therefore not
  comparable to the canonical batch numbers, which came from SCIP 10. Do not
  relax the pin to make the test pass — it is what surfaces the mismatch.
- `test_run_scripts.py::test_ablation_validation_tolerates_a_did_not_terminate_case`
  and `test_viz.py::test_cli_round_trip` spawn subprocesses with a bare
  `sys.executable` plus `PYTHONPATH=PKG_ROOT`, which escapes buck's link tree,
  so `pyscipopt` is not importable in the child. Both fail under buck and pass
  where pyscipopt is installed ambiently.
- Buck collects 339 cases; the pinned host collects 425. The gap is not
  understood yet.

## Canonical batch

`run_main_cases.sh` and `run_ablations.sh` require `SOLVER_LIB_PATH` — colon
separated directories holding libyices and everything libyices links against.
Neither script runs in CI; both long-tail cases (`bwd_lr4096` and the 16-warp
ablation) exceed the 24h watchdog and are recorded as did-not-terminate
observations rather than failures.

## Frozen artifacts

`ablations_v7/` and the v6/v7/v8 stems under `solutions/` are superseded
evidence and are append-never; `test_naming_discipline.py` exempts them from
the codename scan. Provenance recorded in these artifacts is scrubbed of
buck-out paths and toolchain source URLs, which the open-source export gate
rejects. The scrubbing lives in the two run scripts, so re-running a batch does
not reintroduce them — keep it there rather than patching artifacts by hand.
