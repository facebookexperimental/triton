# R&D panel

This panel inherits [the general panel](../general_panel/README.md). It turns profiler
evidence and tickets into testable findings, then turns confirmed and reviewed
findings into durable knowledge.

## 1. Roles

| Role | Input | Output | May run commands | May conclude |
|---|---|---|---|---|
| profiler | profiling request | raw profiler output | yes | no |
| TL | profiles and tickets | findings and causal report | no | no |
| worker | one finding | artifacts and CLI verdict | yes | no |
| knowledge | confirmed findings | reviewed knowledge updates | no | no |
| manager | reports and artifacts | decision | no | yes |

Nothing that reasons may execute; nothing that executes may conclude.

## 2. Agent placement

- Architecture-specific profiler and knowledge-agent definitions live under
  `kernel_optimization/harnesses/<arch>/`.
- Kernel-specific TL-agent strategy lives under
  `kernel_optimization/harnesses/<arch>/targets/<kernel>/`.
- Architecture facts live in `<arch>/knowledge.md`; validated kernel facts live
  in `<kernel>/optimization_guidance.md`.

## 3. Handoff order

1. The profiler agent produces a valid baseline profile and raw artifacts.
2. The TL agent reads that evidence and proposes one causal finding with a
   predicted profile signal.
3. A worker implements the finding and invokes the build and perf-bench panels.
4. The manager accepts the CLI verdict and checks whether the predicted signal
   moved.
5. The knowledge agent records only generalized, human-approved insights.

If a stage cannot satisfy its contract, it reports a blocker instead of asking
the next agent to guess.

## 4. Findings and insights are different

- A **finding** is a proposed change that might improve performance and can be
  verified quickly. Findings, including failures, stay in run artifacts.
- An **insight** is a general causal conclusion that holds beyond one finding.
  Only the knowledge agent records insights, and every update requires human
  expert review.

The manager compares the finding's predicted profile signal with the measured
result:

| Outcome | CLI verdict | Predicted signal | Knowledge action |
|---|---|---|---|
| confirmed | promoted | moved | eligible after human review |
| unexplained win | promoted | did not move | do not claim the mechanism |
| rejected | not promoted | either | retain as a dead end |

An unexplained win may ship because the harness decides promotion, but it is not
evidence for the proposed mechanism.
