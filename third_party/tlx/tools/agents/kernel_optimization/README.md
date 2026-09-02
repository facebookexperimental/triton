# TLX Kernel Optimization Agent

This directory contains the TLX-local optimization loop for standalone Triton and TLX
kernels. It lives inside the TLX codebase (`third_party/tlx/tools/agents/kernel_optimization/`) and
is the canonical location for the loop in this checkout; a future sync to
`third_party/tlx/` will copy from here.

The language used by the kernel is not part of the control-plane contract. A
user-supplied harness owns compilation, correctness, timing, and profiling.

The loop is:

```text
build -> verify -> benchmark -> profile -> propose source mutation -> repeat
```

The candidate generator can propose source, but it cannot declare a candidate correct or
faster. A candidate is promoted only when every protected case passes and the weighted
geometric-mean speedup and measurement-variance thresholds are met. Failed unprotected
cases are retained as diagnostics and excluded from the aggregate speedup.

## Harness contract

A harness is a Python file with these functions:

```python
def build(kernel_source: str, target: dict): ...
def verify(build_artifact, case: dict) -> bool | dict: ...
def benchmark(build_artifact, case: dict, repetitions: int) -> list[float] | dict: ...
def profile(build_artifact, case: dict) -> dict: ...  # optional
```

`build` may return an arbitrary in-process artifact. Returning a mapping with `success`,
`artifact`, and `diagnostics` fields makes build failure explicit. `verify` returns either
a bool or `{passed, diagnostics, metrics}`. `benchmark` returns microsecond samples or
`{samples_us, warmup_count, cache_policy}`. `profile` is optional; when present it is
called after a successful `verify` + `benchmark` pair and its return value (a JSON object)
is persisted per case.

The default harness mode is **subprocess isolation** (`worker.py` subprocess per candidate):
candidate state and imported kernel modules never leak across evaluations. An in-process
`StandaloneHarness` is available programmatically (via `from third_party.tlx.tools.agents.kernel_optimization.harness import StandaloneHarness`)
for debugging and unit tests.

Build/verify/benchmark/profile run in a new subprocess for every source candidate via
`worker.py`. On timeout the agent sends `SIGTERM` then `SIGKILL` to the whole process
group. Large profile payloads (>1MB inline JSON) are spilled to
`artifacts/profile_traces/` with a pointer left in `experiments/<id>/profile.json`.

## CLI

Run the module directly from the Triton source repository root:

```bash
python -m third_party.tlx.tools.agents.kernel_optimization.cli \
  --kernel my_kernel.py --reference-kernel reference_kernel.py \
  --output-dir /tmp/tlx-kernel-agent-run \
  --max-rounds 5 \
  --provider codex --arch blackwell

# Continue from a completed run without adopting its winner:
python -m third_party.tlx.tools.agents.kernel_optimization.cli \
  --kernel my_kernel.py --output-dir /tmp/tlx-kernel-agent-next \
  --prior-run /tmp/tlx-kernel-agent-run \
  --provider codex --arch blackwell

# A revalidated winner is committed by default:
python -m third_party.tlx.tools.agents.kernel_optimization.cli \
  --kernel my_kernel.py --output-dir /tmp/tlx-kernel-agent-run \
  --vcs auto \
  --commit-message "Optimize my kernel with TLX agent"
```

`--arch` selects `harnesses/<arch>/targets/<kernel>`; `harness`/`cases`/`target` can also be passed explicitly.

`--prior-run` accepts a completed output directory or its `experiments.json`. It
imports recomputed source hashes for exact cross-run deduplication and bounded,
sanitized experiment evidence for candidate prompts. It never mutates prior
artifacts or adopts the prior winner; the current kernel is always rebuilt and
validated as the new baseline.

`--reference-kernel` is optional: a trusted oracle kernel. When provided it is persisted to `reference_kernel.py` in the output dir, exposed to harness workers via `TLX_REFERENCE_KERNEL_PATH`, and shown (truncated) to Codex in the prompt as comparison context. `verify` may load it to compare candidate vs reference.

`--provider` is `codex` (default, shells `codex exec`) or `mock` (deterministic stub for
CI that replays canned candidates or echoes the current source). When `codex` is not
installed the provider fails fast with a clear error suggesting `--provider mock`.

Promotion checkpoint commits are enabled by default. Every candidate that passes the
promotion gates is committed immediately before the next candidate is generated. Use
`--no-commit-winner` for artifact-only runs. The CLI
finds the repository from the absolute kernel path and supports `--vcs auto|git|hg` without
using `sl`. Every promoted-candidate commit includes the body line `TLX agent authored`.
Existing unrelated staged and dirty work is preserved. If the target was already dirty, only
the Agent delta is committed and the original target edits remain unstaged/dirty; overlapping
edits fail safely. If final revalidation fails after promotions, the Agent creates a forward
rollback commit without the winner attribution and keeps the checkpoint commits in history.
Exit code `3` means a promotion or rollback commit failed. Ordered commit metadata is written
to `promotion_commits.json`; the compatibility summary remains in `auto_commit.json`.

The optimizer reports baseline, every candidate, and final revalidation performance
to stderr as soon as each evaluation completes. Each line includes status, aggregate
speedup, and per-case correctness, median, p95, CV, and speedup. Each try also logs a
bounded hypothesis/change/expected-effect/risk summary before evaluation and a concise
decision afterward. A requested commit emits one `commit status=committed|failed` event
with VCS, revision, repository, target file, subject, and attribution. Kernel source is
never printed to the live log. The final JSON remains on stdout so callers can parse it
independently of live progress.

`--budget` accepts an optional JSON file that overrides the `--max-*` / `--min-speedup` /
`--max-cv` flags (`{max_rounds, candidates_per_round, max_candidate_seconds,
max_total_seconds, min_speedup, max_cv, benchmark_repetitions}`).

`cases.json` is a list of `{case_id, parameters, weight, protected}` objects. `target.json`
contains `{backend, architecture, device, environment}`. The harness receives the full
`target` dict (including `environment` merged into `os.environ` for the worker) and each
`case` dict verbatim.

The output directory contains:

```text
best_kernel.py
result.json                 # KernelOptimizationResult (success, baseline, final, experiments, stopping_reason)
experiments.json            # alias of result.experiments (Google Doc compatibility)
baseline_profile.json       # aggregated per-case profile for the baseline
best_profile.json           # aggregated per-case profile for the promoted winner
auto_commit.json             # present when --commit-winner reaches finalization
artifacts/profile_traces/   # spilled large profile payloads
experiments/
  baseline/{kernel.py, result.json, profile.json}
  r001-c000/{kernel.py, incremental.patch, cumulative.patch, result.json, profile.json}
  r001-c001/...
```

Every returned candidate source is cached before deduplication, compilation, correctness,
or performance evaluation. `incremental.patch` compares against the exact current-best
parent used to generate that action; `cumulative.patch` compares against the original run
baseline. The live log records the absolute artifact paths and prints the complete incremental
patch between `incremental-diff-begin` and `incremental-diff-end` markers before evaluation.
Failed, duplicate, and rejected candidates keep these artifacts and log entries. If the
provider fails before returning source, the experiment has no patch paths because no candidate
exists to diff.

`--harness-mode` is retained for compatibility
isolation in the CLI path; `StandaloneHarness` is available via the Python API.

## TLX GEMM example

`harnesses/blackwell/targets/gemm/harness.py` runs any complete candidate source that exports
`matmul(a, b)`. It compares against `torch.matmul`, benchmarks with
`triton.testing.do_bench`, and reports latency and TFLOP/s. Its legacy two-argument
`profile(build_artifact, case)` returns latency and throughput, and can optionally collect a
basic Proton trace when `TRITON_PROTON` is set. It does not implement structured profile
requests or NCU collection.

### Target-supplied profiling

Canonical workflow guidance lives in `docs/profiling/proton.md` for Proton and
`docs/profiling/nvidia-ncu.md` for NVIDIA NCU. These documents guide harness and
run orchestration; they are not injected into candidate source prompts.

A target harness may implement `profile(build_artifact, case, request)` to honor structured
profile requests. Missing tools, unsupported metrics, and profiler failures should be returned
as diagnostics so correctness and benchmark results remain usable. Before freezing a CUDA
bundle, smoke-test that an expected Proton main launch has nonzero time and NCU duration is
non-null when those tools are available.

- **Triton Proton launch attribution:** handle `tools=["proton_launch"]` with an absolute
  `artifacts_dir`. A supporting harness should warm up, synchronize, collect one
  launch-attribution-only Proton tree, save raw artifacts, and call
  `parse_proton_launch_attribution()` with the target's exact `main_scope`.
- **Native profiler:** handle `tools=["native_profiler"]` by mapping this portable name to the
  target platform profiler. NVIDIA requests are resolved to NCU. Collect into an `.ncu-rep`,
  then call `export_ncu_report_details()` to persist and parse the details CSV; collection
  stdout contains status messages rather than metric rows when `--export` is used. Explicit
  `ncu` remains a compatible NVIDIA-only request.
- **Diagnostic instrumentation:** `proton_intra_kernel` requires a target-supplied instrumented
  replay. Instrumented source and timing must never be benchmarked, promoted, or committed.

Target-specific `harness.py`/`cases.json`/`target.json` live colocated under `harnesses/<arch>/targets/<kernel>/` (B200,
`sm_100` for blackwell and H100, `sm_90` for hopper); pick `--arch` to match the
device you are tuning for. Pass an existing TLX tutorial such as
`third_party/tlx/tutorials/blackwell_gemm_ws.py` as `--kernel`.

### Curated optimization knowledge

Two optional markdown files in the bundle are concatenated ahead of
`target.json`'s inline `optimization_guidance` string, widest scope first, and
delivered as that one field:

```text
harnesses/<arch>/knowledge.md                              # architecture-wide
harnesses/<arch>/targets/<kernel>/optimization_guidance.md # this target
```

Keeping them in the bundle is deliberate: the bundle is frozen and content-hashed
for the run, so the prompt a run saw is reproducible from its recorded hashes. A
global doc tree read at prompt-build time is not. Adding an architecture means
adding a directory; the candidate provider stays kernel- and arch-neutral and
never reads these paths itself. The resolved block is capped at 8 KB, with the
truncation announced in the text.

The contract for the content:

- **Human-write-only.** Agents read these files. A finding is promoted into one
  by a human who has read the evidence; agent output lands in `experiments/`
  under the run's output dir.
- **Mechanism, method and structure — not measurements.** Prefer how the hardware
  behaves, how to measure it correctly, and how the search space is shaped. A
  detailed figure copied in here goes stale with nothing to notice, and invites a
  candidate to pattern-match on the number instead of the mechanism that produced
  it. Cite where the evidence lives; do not reproduce it.
- **Do not restate a hardware quantity.** `hw/resources.py` is ground truth for
  CU/XCD counts, LDS and SMEM budgets, and executable heuristics tune against it.
  Cite the attribute and record the consequence, so a correction propagates
  instead of forking. Same reasoning, applied to numbers the codebase owns.
- **Every claim is still evidence-backed** — the evidence is cited by location
  (a run's artifacts, a benchmark suite, a docstring), not pasted in.
- **Measured-on-this-arch and ported-from-another-arch are separate sections.**
  A CDNA4 result is a hypothesis for CDNA3, not a fact about it; the prompt tells
  the model to weigh them differently, so mixing them removes that signal.
- **Concise beats complete.** This text is injected into every candidate prompt.
  An entry that does not change what a candidate would do costs tokens for
  nothing.

`harnesses/host/targets/vector_add/harness.py` is a minimal CPU-friendly harness for smoke tests
without a real GPU. Candidate must export `vector_add(a, b)`; on CPU the benchmark uses
synthetic `LATENCY_US` timing so unit tests pass on any host.

## H100 pilot

```bash
# Kernel-only: arch auto-resolved, or pass --arch hopper for H100
python -m third_party.tlx.tools.agents.kernel_optimization.cli \
  --kernel my_gemm_kernel.py --reference-kernel baseline_gemm.py \
  --arch hopper \
  --output-dir /tmp/tlx-agent-h100 \
  --max-rounds 3 --candidates-per-round 4 \
  --max-candidate-seconds 600 --max-total-seconds 3600 \
  --provider codex
```
