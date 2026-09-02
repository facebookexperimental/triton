# General panel — policies every panel inherits

These are cross-cutting rules shared by the R&D, build, and perf-bench panels.
Role-specific strategy belongs in the owning panel, not here.

## 1. The harness decides

The `kernel_optimization` CLI owns build, verification, benchmarking, and the
promotion decision. That loop is deterministic on purpose: a candidate
generator may propose source, but it cannot declare a candidate correct or
faster. No panel re-measures or overrides a verdict issued by the harness.

The separation is strict:

- Reasoning agents propose or assess work; they do not execute it.
- Workers execute assigned work and return artifacts; they do not decide.
- Managers make decisions from reports and artifacts.

## 2. Evidence is addressable

Every claim must resolve to a real artifact, trace, source location, test, or
ticket. Paths cited by a panel must exist. This is a mechanical gate, not a
request for an agent to be careful.

## 3. Agent definitions are human-owned

Agent and panel definitions are checked in and reviewed as code. No agent edits
an agent definition, including its own, without an explicit human instruction.

## 4. Specialized panels

- [R&D panel](../rd_panel/README.md): profiling, causal analysis, findings, and
  knowledge curation.
- [Build panel](../build_panel/README.md): rebuild and correctness-validation policy.
- [Perf-bench panel](../perf_bench_panel/README.md): controlled performance measurement
  and full-shape regression gates.
