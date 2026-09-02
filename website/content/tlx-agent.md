TLX Agent is a harness-driven optimization loop for complete Triton and TLX
kernel source files. It turns a fixed workload and a trusted measurement
contract into isolated candidate experiments, then promotes only a correct,
measured, and revalidated winner.

> The harness owns truth. TLX Agent does not invent correctness criteria or
> change the benchmark while it searches.

## When to use it

Use TLX Agent when you have an existing kernel and a concrete workload to
optimize. A request can start through a higher-level coding agent or by invoking
the public CLI directly.

You provide the kernel path, target workload, hardware, and optional search
budget. If an exact target bundle does not already exist, the higher-level agent
prepares and validates one before the optimization loop starts.

## Recommended: run through a coding agent

The recommended entry point is a coding agent that can inspect the repository,
prepare the measurement contract, and supervise the run. Ask it to use the TLX
kernel optimization agent rather than asking it to optimize the kernel manually.

For example:

```text
Use the TLX kernel optimization agent to optimize
<absolute-kernel-path> for BATCH=4, H=32, N_CTX=8192,
HEAD_DIM=128, causal=false, mode=bwd on B200.
Use the repository's authoritative correctness and performance tests,
return the live-log path when the run starts, and report the validated winner.
```

The coding agent should then:

- Read the bundled TLX Agent Skill and translate the request into the public CLI contract.
- Find or prepare the exact target bundle, validate and negative-test it, then freeze it for the run.
- Launch the public CLI instead of manually reproducing the optimization loop.
- Return the absolute live-log path as soon as the run starts so progress is observable.
- Monitor failures and requirements outside the frozen contract while TLX Agent owns candidate generation, profiling, correctness gates, promotion, and the winner commit.
- Report the final workload, baseline and winner latency, speedup, variance, correctness, artifacts, and commit result.

This route is preferred because the coding agent handles repository-specific setup
and recovery while the TLX Agent retains one consistent, measurable optimization
policy. Direct CLI use remains available when you already have a validated frozen
bundle.

## Input contract

| Input | Purpose |
|---|---|
| `kernel.py` | Complete source file under optimization |
| `harness.py` | Builds, verifies, benchmarks, and optionally profiles a candidate |
| `cases.json` | Workloads, weights, and protected correctness cases |
| `target.json` | Backend, architecture, device, environment, and target-specific guidance |
| `reference_kernel.py` | Optional trusted correctness oracle |
| `budget.json` | Optional round, timeout, speedup, variance, and repetition limits |
| `output/` | Fresh directory for logs, candidates, profiles, and final results |

The harness exposes four operations:

```python
def build(kernel_source: str, target: dict): ...
def verify(build_artifact, case: dict): ...
def benchmark(build_artifact, case: dict, repetitions: int): ...
def profile(build_artifact, case: dict, request: dict | None = None): ...
```

`profile` is optional. The other operations form the correctness and performance
boundary for every experiment.

## Prepare and freeze the target bundle

The higher-level coding agent owns target-bundle preparation. It should:

- Find an existing bundle that exactly matches the kernel entry point, workload, mode, backend, and architecture, or derive one from the authoritative correctness test and production benchmark.
- Run the untouched kernel through build, verification, repeated benchmarking, and summary profiling.
- Confirm that an invalid candidate fails to build and an incorrect but compilable candidate fails verification.
- Put kernel-specific invariants, known failed experiments, and profiler-to-knob guidance in `target.json` as `optimization_guidance`.
- Record absolute paths and content hashes, then freeze `harness.py`, `cases.json`, and `target.json` for the duration of the run.

Candidate generation must not edit the frozen bundle. This prevents the search
from improving its score by weakening correctness or changing the workload.

## Run the CLI

From the repository root:

```bash
PYTHONPATH=<repo-root> python -m third_party.tlx.tools.agents.rd_panel.kernel_optimization.cli \
  --kernel <absolute-kernel.py> \
  --harness <absolute-bundle/harness.py> \
  --cases <absolute-bundle/cases.json> \
  --target <absolute-bundle/target.json> \
  --output-dir <absolute-output-dir> \
  --provider codex \
  --max-rounds 5 \
  --candidates-per-round 2 \
  --max-candidate-seconds 600 \
  --max-total-seconds 3600 \
  --min-speedup 1.01 \
  --max-cv 0.10 \
  --benchmark-repetitions 10
```

Add `--reference-kernel`, `--budget`, `--arch`, or `--model` only when the run
requires them. Winner commits are enabled by default; use
`--no-commit-winner` for an artifact-only run.

## Optimization loop

Every candidate follows the same closed loop:

```text
build -> verify -> benchmark -> profile -> decide -> repeat
```

TLX Agent keeps candidate generation outside the live checkout. Each proposal
changes one measured subsystem and records its hypothesis, evidence, expected
effect, and risk. Incorrect, duplicate, unstable, and materially slower
candidates are rejected. Failed hypotheses and hardware evidence feed the next
round instead of being silently retried.

The current best candidate receives stronger profiling and final revalidation
before it can replace the original source.

## Live log and artifacts

Progress is written to stderr so it can be tee'd to a stable live-log path. The
final structured result is written to stdout and the output directory.

The live log reports:

- Baseline, candidate, promotion, rejection, and final-validation status.
- The candidate hypothesis, evidence, change, expected effect, and risk.
- Per-case correctness, median, p95, coefficient of variation, and speedup.
- Compact Proton and target-profiler evidence when available.
- Winner-commit status and diagnostics.

Full kernel source is not printed in the live log. Candidate source, raw
profiler reports, commands, normalized summaries, and the final winner remain in
the artifact directory for inspection.

## Profiling

Profiling is staged so expensive evidence is collected where it can change a
decision:

- Proton attributes wrapper, launch, main-kernel, and non-main-kernel time.
- The portable `native_profiler` request selects the target platform profiler. NVIDIA harnesses map it to NCU for duration, throughput, occupancy, register, memory-traffic, tensor-activity, and stall metrics.
- Optional per-warp Proton instrumentation can answer intra-kernel attribution questions for warp-specialized pipelines.

An ordinary Proton launch timeline does not prove that `tlx.async_task` regions
overlap. Per-warp instrumentation must select aligned logical work across task
warps, and the instrumented source is diagnostic-only: it is never benchmarked,
promoted, or committed.

## Promotion and final validation

A candidate is promotable only when every protected case passes, weighted
speedup reaches the configured threshold, variance stays within budget, and
hardware profiling shows no material main-kernel regression. It must also beat
the current best candidate.

The finalist is rebuilt and revalidated with the frozen bundle. A generated
`best_kernel.py` without final revalidation is not a completed run.

## Version-control behavior

After successful final validation, TLX Agent creates a local winner commit by
default. It detects Git or Mercurial from the kernel path, commits only the
baseline-to-winner delta, preserves unrelated staged and dirty work, and adds
`TLX agent authored` to the commit body.

If the live kernel changed during the run or edits overlap unsafely, the Agent
keeps the winner artifact and reports a commit failure instead of overwriting
user work. Submitting, landing, or pushing the commit is always a separate
explicit action.
