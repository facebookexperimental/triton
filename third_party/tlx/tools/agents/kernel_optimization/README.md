# Triton/TLX Kernel Optimization Agent

This directory contains the TLX-local optimization loop for standalone Triton and TLX
kernels. It lives inside the TLX codebase (`third_party/tlx/tools/agents/kernel_optimization/`) and
is the canonical location for the loop in this checkout; a future sync to
`third_party/tlx/` will copy from here.

The language used by the kernel is not part of the control-plane contract. A
user-supplied harness owns compilation, correctness, timing, and profiling.
The built-in optimization policy classifies
launch, data-movement, compute, occupancy, synchronization, layout, and spill
bottlenecks, then asks for one measurable source or compiler hypothesis at a
time. A TLX or other reference kernel is optional design evidence. Without a
reference, the agent derives hypotheses from correctness-gated benchmark and
profiler results supplied by the harness.

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

`--reference-kernel` is optional: a trusted oracle kernel. When provided it is
persisted to `reference_kernel.py` in the output dir, exposed to harness
workers via `TLX_REFERENCE_KERNEL_PATH`, and copied in full into the candidate
agent's workspace. `verify` may load it to compare candidate vs reference.

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

## Generated fused Triton case

`fused_triton` adapts one generated PostFuser case to this same optimization
loop. The candidate is the case's complete `output_code_fused.py`; correctness
and CUDA-event timing run directly with a locally built Triton checkout; Buck
is not used. The shape-matched TLX implementation is resolved automatically
and made available to the candidate agent as `reference_kernel.py`. Pass
`--no-registry-reference` to exercise the latency/source-evidence path without it,
or `--reference-kernel PATH` to use another implementation.

Run the launcher from the Triton source repository root, wrapped in
`denoise.sh` on NVIDIA. The generated case directory must already contain
`output_code.py`, `output_code_fused.py`, `manifest.json`, and
`bench_vs_geo.py`. At session start the launcher asks which Triton checkout to
use, validates its `python/triton/_C/libtriton.so`, and verifies that the chosen
Python runtime imports that exact checkout. The Python runtime must provide
PyTorch; pass `--python-executable` when it differs from the launcher:

```bash
CASE=matmul_add
FBS=/path/to/fbsource
CUDA_VISIBLE_DEVICES=0 third_party/tlx/denoise.sh \
  python -m third_party.tlx.tools.agents.kernel_optimization.fused_triton \
  --case-dir "$FBS/fbcode/triton/tools/post-fuser/geo_unfused/$CASE" \
  --fbsource-root "$FBS" \
  --output-dir "/tmp/fused-agent-$CASE" \
  --python-executable /path/to/python-with-torch \
  --gpu-id 0 \
  --max-rounds 3 --candidates-per-round 2 \
  --benchmark-warmup-ms 500 \
  --benchmark-duration-ms 2000 \
  --reference-tuning-configs-json /path/to/tlx_candidates.json
```

If the reference needs process-local compatibility settings, pass them without
contaminating the candidate process, for example
`--reference-env TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`. Supplying any
`--reference-env` also isolates the two benchmark legs when AutoWS is disabled.

Each fused and TLX leg uses `triton.testing.do_bench(return_mode="all")`.
Warmup and measurement are duration-based, every timed execution starts after
an L2 cache flush, and the agent computes median, p95, and CV from the returned
raw device-time samples. The production flow requires TLX tuning before Triton
hill climbing. Provide a JSON list of explicit TLX candidates with
`--reference-tuning-configs-json`; the launcher runs every candidate in a fresh
process under the same L2-cold timing policy, rejects incorrect or noisy
candidates, writes `OUTPUT_DIR/config/tuned_tlx.json`, and pins that winner for
every Triton iteration. A previously generated winner can instead be supplied
with `--reference-config-json`. Use `--allow-untuned-reference` only for
diagnostic runs; those results are labelled as a TLX reference rather than
tuned TLX.

For a best-TLX comparison, candidate entries may contain `kernel_config` for a
benchmark's explicit-config path and/or `module_overrides` for a specialized TLX
module whose launch constants are fixed in Python. A module override names the
kernel-relative source path and every global constant to change. An
`environment` mapping can select a registry kernel's named single-config mode.
Include all kernel and launch fields (`num_warps`, `num_stages`, and cluster
shape where configurable), not just tile sizes. For example:

```json
{
  "configs": [
    {
      "name": "bm128_bn256_bk64_split6",
      "module_overrides": [
        {
          "relative_path": "compute/bf16/fused/fwd/example/shape/kernel.py",
          "values": {
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 64,
            "NUM_SMEM_BUFFERS": 2,
            "NUM_TMEM_BUFFERS": 1,
            "SPLIT_K": 6
          }
        }
      ]
    }
  ]
}
```

Keep direct kernel latency distinct from complete Python-wrapper latency: CUDA
events around a wrapper can include idle-stream time between the start event and
kernel submission.

For isolated TLX and AutoWS legs, final validation additionally uses identical
seeded inputs for one untimed launch, temporarily serializes the TLX outputs,
and records `tlx_precision_comparison` for every fused output. It reports shape,
dtype, exact-match fraction, maximum and mean absolute error, and maximum
relative error. The temporary reference tensors are removed after comparison
and their transfer cost is never included in performance timing.

Pass `--autows` when the candidate source annotates its recurring load/MMA loop
with `tl.range(..., warp_specialize=True)`. This enables the selected AutoWS
implementation only for the fused leg and benchmarks the hand-written TLX leg
in a separate process. An AutoWS result is not established until the exact final
TTGIR contains both `ttg.warp_specialize` and materialized partition regions.
When a matching TLX kernel uses TMA and persistent scheduling, first establish
a correct plain-Triton TMA/persistent candidate, then add AutoWS as a separate
experiment so the performance effects remain attributable.

`--autows` defaults to `--autows-implementation meta`. Also run a separate
session with `--autows-implementation upstream` when the standard Triton WS
lowering is deployable. The launcher removes both `TRITON_USE_META_WS` and
`TRITON_DISABLE_WSBARRIER_REORDER` from the upstream candidate and from the
isolated TLX reference. Treat the two AutoWS implementations as independent
variants: use fresh output/cache directories, prove physical partitions in each
final TTGIR, and retain whichever correctness-valid mode is faster.

Before compiler-level experiments, calculate the persistent tile count and its
number of waves over the target SM count. Sweep configurations that avoid an
underfilled first wave. Treat TMA operand loading and TMA epilogue publication
as separate choices: if final TTGIR shows that descriptor stores add staging or
wait overhead, benchmark a TMA-operand/direct-pointer-epilogue candidate rather
than discarding TMA wholesale. Keep the full-TMA form in that A/B test for large
residuals or outputs, where descriptor epilogue I/O can win. When folding casts
or padding, preserve graph-visible intermediate rounding and use descriptor
out-of-bounds zero fill only when it exactly matches the padding semantics.

For an unattended run, replace the prompt with
`--triton-dir /path/to/triton`. If native compiler code changed, run `make` in
that checkout before starting or continuing the agent. Python-only kernel or
agent changes do not require a rebuild.

The launcher deliberately disables automatic winner commits because generated
suite files are not checked in. At normal session completion it preserves the
generated `output_code_fused.py` baseline and atomically writes the final
correctness-validated winner to `output_code_fused_opt.py`. The optimized file
starts with a generated comment summary of the applied change, baseline and
winner wrapper latency, and the TLX comparison. A supplied
`--reference-config-json` marks that comparison as tuned TLX; otherwise it is
labelled only as the TLX reference. The launcher refuses to write the optimized
file if final verification did not pass. The run directory retains
`best_kernel.py`, `result.json`, and the per-experiment patches as an audit
trail. To continue learning without repeating rejected candidates, start from
the prior winner in a fresh output directory and pass the old run as evidence:

```bash
python -m third_party.tlx.tools.agents.kernel_optimization.fused_triton \
  --case-dir "$FBS/fbcode/triton/tools/post-fuser/geo_unfused/$CASE" \
  --kernel "/tmp/fused-agent-$CASE/best_kernel.py" \
  --prior-run "/tmp/fused-agent-$CASE" \
  --triton-dir /path/to/triton \
  --python-executable /path/to/python-with-torch \
  --fbsource-root "$FBS" \
  --output-dir "/tmp/fused-agent-$CASE-next"
```
