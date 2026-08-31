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
```

`--arch` selects `harnesses/<arch>/targets/<kernel>`; `harness`/`cases`/`target` can also be passed explicitly.

`--reference-kernel` is optional: a trusted oracle kernel. When provided it is persisted to `reference_kernel.py` in the output dir, exposed to harness workers via `TLX_REFERENCE_KERNEL_PATH`, and shown (truncated) to Codex in the prompt as comparison context. `verify` may load it to compare candidate vs reference.

`--provider` is `codex` (default, shells `codex exec`) or `mock` (deterministic stub for
CI that replays canned candidates or echoes the current source). When `codex` is not
installed the provider fails fast with a clear error suggesting `--provider mock`.

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
`triton.testing.do_bench`, and reports latency and TFLOP/s. Set `TRITON_PROTON=1`
to also collect a Triton Proton trace (returned inside `profile()`).

### Profiling recipes

- **Triton Proton / Triton-MPP:** inside `profile()` wrap the kernel with
  `import triton.profiler as proton; proton.start("matmul"); module.matmul(a,b); proton.deactivate()`
  and return `{"proton_trace": ...}`. The agent persists it as-is.
- **NCU:** shell `ncu --csv --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ...`
  on a single rep, parse stdout, and return a small dict like
  `{"ncu": {"sm_throughput": ...}}`. Large CSV traces can be written to a file and
  returned as `{"artifact": "path/to/trace.csv"}` — the agent will spill >1MB payloads.

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
