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

# A revalidated winner is committed by default:
python -m third_party.tlx.tools.agents.kernel_optimization.cli \
  --kernel my_kernel.py --output-dir /tmp/tlx-kernel-agent-run \
  --vcs auto \
  --commit-message "Optimize my kernel with TLX agent"
```

`--arch` selects `harnesses/<arch>/targets/<kernel>`; `harness`/`cases`/`target` can also be passed explicitly.

`--reference-kernel` is optional: a trusted oracle kernel. When provided it is persisted to `reference_kernel.py` in the output dir, exposed to harness workers via `TLX_REFERENCE_KERNEL_PATH`, and shown (truncated) to Codex in the prompt as comparison context. `verify` may load it to compare candidate vs reference.

`--provider` is `codex` (default, shells `codex exec`) or `mock` (deterministic stub for
CI that replays canned candidates or echoes the current source). When `codex` is not
installed the provider fails fast with a clear error suggesting `--provider mock`.

Winner commits are enabled by default and run only after successful final revalidation.
Use `--no-commit-winner` for artifact-only runs. The CLI
finds the repository from the absolute kernel path and supports `--vcs auto|git|hg` without
using `sl`. Every generated commit includes the body line `TLX agent authored`. Existing
unrelated staged and dirty work is preserved. If the target was already dirty, only the
baseline-to-winner delta is committed and the original target edits remain unstaged/dirty;
overlapping edits fail safely. Exit code `3` means optimization succeeded but commit failed,
and `best_kernel.py` remains available. Commit metadata is written to `auto_commit.json`.

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
  r001-c000/{kernel.py, result.json, profile.json}
  r001-c001/...
```

`--harness-mode` is retained for compatibility and currently always uses subprocess
isolation in the CLI path; `StandaloneHarness` is available via the Python API.

## TLX GEMM example

`harnesses/blackwell/targets/gemm/harness.py` runs any complete candidate source that exports
`matmul(a, b)`. It compares against `torch.matmul`, benchmarks with
`triton.testing.do_bench`, and reports latency and TFLOP/s. Its legacy two-argument
`profile(build_artifact, case)` returns latency and throughput, and can optionally collect a
basic Proton trace when `TRITON_PROTON` is set. It does not implement structured profile
requests or NCU collection.

### Target-supplied profiling

A target harness may implement `profile(build_artifact, case, request)` to honor structured
profile requests. Missing tools, unsupported metrics, and profiler failures should be returned
as diagnostics so correctness and benchmark results remain usable.

- **Triton Proton launch attribution:** handle `tools=["proton_launch"]` with an absolute
  `artifacts_dir`. A supporting harness should warm up, synchronize, collect one
  launch-attribution-only Proton tree, save raw artifacts, and return normalized totals.
- **Native profiler:** handle `tools=["native_profiler"]` by mapping this portable name to the
  target platform profiler. NVIDIA requests are resolved to NCU, and a supporting harness may
  collect summary or deep metrics and save command/query/CSV/stderr artifacts. Explicit `ncu`
  remains a compatible NVIDIA-only request.
- **Diagnostic instrumentation:** `proton_intra_kernel` requires a target-supplied instrumented
  replay. Instrumented source and timing must never be benchmarked, promoted, or committed.

Target-specific `harness.py`/`cases.json`/`target.json` live colocated under `harnesses/<arch>/targets/<kernel>/` (B200,
`sm_100` for blackwell and H100, `sm_90` for hopper); pick `--arch` to match the
device you are tuning for. Architecture-wide notes, known optimization tricks, and shared
target metadata can live directly under `harnesses/<arch>/`. Pass an existing TLX tutorial such as
`third_party/tlx/tutorials/blackwell_gemm_ws.py` as `--kernel`.

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
