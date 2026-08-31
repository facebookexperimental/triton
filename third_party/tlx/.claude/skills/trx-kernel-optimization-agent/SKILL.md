---
name: trx-kernel-optimization-agent
description: >
  Execute the TLX Kernel Optimization Agent CLI on a Triton or TLX kernel.
  Use this skill whenever the user says "use the TLX agent", "use the kernel
  optimization agent", "用 TLX agent 优化", or asks Carl/Claude to optimize a
  kernel with the repository agent. The required outcome is an actual agent CLI
  invocation and its measured result, not a walkthrough or reimplementation.
---

# Run The TLX Kernel Optimization Agent

The executable is `third_party/tlx/tools/agents/kernel_optimization/cli.py`.
Use the user-facing name "TLX kernel optimization agent"; the on-disk skill
name is compatibility-only. Follow the layers below in order. Target-specific
profiling rules supplement, but never replace, the generic workflow.

## Layer 0: Invocation And Safety

1. Invoke the public CLI. Do not manually optimize the kernel or substitute a
   generic subagent before the first CLI attempt.
2. Resolve the repository root, absolute kernel path, and target bundle. Record
   initial source-control status and treat existing bytes as the user baseline.
   Never clean or revert a dirty worktree.
3. Keep candidate generation isolated from the live checkout. Candidate source
   may exist only in the provider's temporary workspace and output artifacts
   until final promotion.
4. Run stdout and stderr separately. Tee stderr to a stable absolute live log;
   write stdout JSON to a separate artifact. When starting a run, respond only
   with the absolute log path unless the user asks for more.
5. The CLI commits a revalidated winner by default. Use
   `--no-commit-winner` only for an explicitly requested artifact-only run.
   Submission remains a separate explicit action.
6. If the live kernel changes concurrently, do not overwrite it. Preserve the
   winner artifact and report the conflict.

Reading agent implementation is allowed only after the CLI reports an internal
failure that requires diagnosis. The first attempt must use the public contract.

## Layer 1: Inputs And Standard Loop

The CLI needs:

```text
kernel.py                  complete source file
bundle/harness.py          build, verify, benchmark, optional profile
bundle/cases.json          workloads, weights, protected cases
bundle/target.json         backend, architecture, device, environment
output/                    fresh artifact directory
```

Optional inputs are `reference_kernel.py` and `budget.json`. The higher-level
coding agent owns target-bundle preparation; the TLX Agent consumes the bundle
as a frozen trust boundary and must never generate or modify it during the
optimization loop.

Before invoking the CLI, the higher-level agent must:

1. Search for an existing bundle that exactly matches the kernel entry point,
   workload, mode, backend, and architecture. Do not silently reuse a nearby
   shape or provider.
2. If no exact bundle exists, create a run-specific bundle from the nearest
   authoritative correctness test and production benchmark. Keep it outside
   the candidate workspace and do not modify the user's kernel permanently.
3. Encode all requested workloads in `cases.json`, hardware and environment in
   `target.json`, and kernel-specific invariants, known failed experiments, and
   evidence-to-knob guidance in `target.json` as `optimization_guidance`.
4. Run the bundle against the untouched kernel before launching the Agent.
   Confirm build succeeds, every protected case passes, repeated benchmark
   samples are stable, and summary profiling selects the intended kernel.
5. Negative-test the trust boundary with an intentionally invalid or incorrect
   temporary candidate and confirm build or verification rejects it.
6. Freeze the validated bundle for the duration of the run. Record its absolute
   paths and content hashes in the run log, and pass all paths explicitly to the
   public CLI.

Read `references/input-contract.md` for the complete construction and
validation checklist. Do not start the optimization loop until the bundle is
validated.

For every candidate, the standard loop is:

```text
build -> verify -> benchmark -> profile -> decide -> repeat
```

The live log must include:

- hypothesis and evidence;
- one coherent source/configuration change;
- expected effect and risk;
- correctness, median, p95, CV, and speedup;
- Proton attribution plus target summary profile deltas and a concise decision.

Do not print kernel source or private chain-of-thought. Continue through the
configured round budget after an unpromoted round. Reject incorrect, duplicate,
unstable, and materially regressing candidates.

## Layer 2: Profiling Request And Artifacts

Harnesses that implement `profile` should accept a structured request with:

```json
{
  "level": "summary",
  "tools": ["proton", "ncu"],
  "experiment_id": "stable candidate or baseline id",
  "artifacts_dir": "/absolute/path/to/profile-artifacts",
  "reason": "why this profile was requested",
  "diagnostic_only": false
}
```

The legacy two-argument `profile(build_artifact, case)` contract remains
supported and means summary profiling for the default tools. Return compact,
normalized JSON inline. Store raw `.hatchet`, `.chrome_trace`, `.ncu-rep`, CSV,
mapping, and command files as artifacts, and reference them with absolute paths.

Profiling has three distinct layers:

- Proton wrapper/launch attribution: read `references/proton-profiling.md` and
  collect for every correctness-passing candidate.
- Target summary/deep profiling: for CUDA/NVIDIA, read
  `targets/nvidia/ncu-profiling.md`; other backends use their own target guide
  or the generic profile contract.
- Diagnostic-only Proton intra-kernel instrumentation: use only for attribution
  questions that cannot be answered from wrapper timelines or target counters.

## Layer 3: Proton Attribution

Use Proton to attribute wrapper, benchmark phase, launch, and profiler overhead.
An ordinary Proton timeline with `hook='triton'` that shows one kernel launch
does not prove `tlx.async_task` overlap; it only shows launch/wrapper
attribution around the compiled kernel. Do not infer per-task overlap from that
timeline.

Diagnostic intra-kernel attribution must use Proton instrumentation mode with
`backend='instrumentation'`, `data='trace'`, `granularity='warp'`, and explicit
Triton semantic enabled. Group warp lanes by known async-task warp ranges from
the source, generated metadata, or a saved mapping artifact. Do not request
`warp_group` granularity because the runtime rejects it today.

Instrumentation changes are diagnostic-only. Compiler transforms may move or
merge scopes, so instrumented source and timing must never be benchmarked,
promoted, committed, or used as speedup evidence.

## Layer 4: Target Profiling

Select target profiling guidance from `target.json`:

- CUDA/NVIDIA: read `targets/nvidia/ncu-profiling.md`.
- Other backends: use a sibling target guide when present; otherwise use the
  vendor-neutral `profile()` contract without inventing NVIDIA requirements.

Every correctness-passing candidate receives Proton attribution and the target
guide's summary profile. Escalate to the target guide's deep profile for:

- the baseline before candidate generation;
- a candidate within one percentage point of the promotion threshold;
- disagreement between endpoint benchmark, Proton attribution, and target
  summary profile;
- repeated rounds with no promoted candidate;
- schedule or WS hypotheses that need stall, occupancy, spill, or
  memory-hierarchy evidence;
- the final promoted candidate before commit.

Do not spend deep-profile time on incorrect or clearly slow candidates.
Unsupported counters must be reported as unavailable with JSON `null` and a
diagnostic, never as zero.

## Layer 5: Evidence-To-Action Policy

Each candidate hypothesis must cite measured evidence and change one subsystem
or tightly coupled invariant-preserving pair. Feed failed hypotheses and their
metric regressions into subsequent prompts as exclusions.

Generic interpretation rules:

- Higher kernel duration with lower compute and memory utilization indicates
  scheduling, serialization, synchronization, or insufficient parallelism.
- A benchmark-only change inside the noise floor with unchanged kernel metrics
  is not optimization evidence.
- Lower utilization without lower work, traffic, and duration is not a win.
- Register-budget or buffering changes require spill/occupancy evidence and a
  producer-consumer/barrier proof.
- Main-kernel profile time and public-wrapper benchmark time cover different
  scopes; do not subtract them to manufacture a bottleneck.

Use the target guide for vendor-specific counter interpretation.

## Layer 6: Promotion, Revalidation, And Commit

Promote only when:

- every protected case passes;
- weighted speedup meets the configured threshold;
- measurement variance is within budget;
- target profiling shows no material main-kernel regression.

Revalidate the finalist with benchmark, Proton attribution, summary target
profile, and required deep target profile. Only then may the default commit
occur. Auto-commit must detect Git or Mercurial from the kernel path, preserve
unrelated dirty/staged work, include `TLX agent authored` in the commit body,
and log VCS, revision, repository, target, subject, and failure diagnostics. If
commit fails, keep all artifacts and return the distinct commit-failure status.

## Layer 7: Command And Completion

From the repository root:

```bash
PYTHONPATH=<repo-root> python -m third_party.tlx.tools.agents.kernel_optimization.cli \
  --kernel <absolute-kernel.py> \
  --harness <absolute-harness.py> \
  --cases <absolute-cases.json> \
  --target <absolute-target.json> \
  --output-dir <absolute-output-dir> \
  --provider codex \
  --max-rounds 5 \
  --candidates-per-round 2 \
  --max-candidate-seconds 600 \
  --max-total-seconds 3600 \
  --min-speedup 1.01 \
  --max-cv 0.10 \
  --benchmark-repetitions 10 \
  --profile
```

Add `--reference-kernel`, `--budget`, `--arch`, `--model`, `--vcs`, or
`--commit-message` only when required. Do not invent a model name. Never run
`optimizer.py` directly.

The task is complete only after the actual CLI run reaches final revalidation
or a diagnosed blocking failure. Report the exact workload, GPU, baseline and
final latency, speedup, CV, correctness, stopping reason, commit result, and
absolute output directory. `best_kernel.py` by itself is not completion.
