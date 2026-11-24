# TLX Kernel Optimization Input Contract

## Directory layout

Keep one target bundle together so another agent can reproduce the run:

```text
<target-bundle>/
  harness.py
  cases.json
  target.json
  budget.json          # optional
  reference_kernel.py  # optional
```

The kernel under optimization may live elsewhere in the repository. Pass every
path to the CLI as an absolute path.

## Kernel source

`--kernel` points to one complete Python source file. The optimizer sends its
entire contents to the candidate provider and expects each proposal to be a
complete replacement file, not a patch.

The file must preserve every public entry point used by `harness.py`. Keep host
wrappers and imports needed to compile the kernel in the same candidate source,
or make the harness inject the candidate into a stable wrapper without changing
the candidate contract.

Use `--reference-kernel` only for a trusted correctness oracle. The reference is
persisted in the output directory and exposed to harness workers as
`TLX_REFERENCE_KERNEL_PATH`.

## Harness ownership and preparation

The higher-level coding agent prepares and validates the harness before it
starts the TLX Agent CLI. The optimization loop treats the harness, cases, and
target metadata as immutable inputs. Candidate generation must never edit them.

Build a new bundle only after searching for an exact existing match. Derive it
from the repository's authoritative correctness test and production benchmark,
not from an ad hoc timing loop. The harness must exercise the same public entry
point, input preparation, precision, mode, and cache behavior that the user
asked to optimize.

Before handing the bundle to the TLX Agent:

1. Run `build`, `verify`, and `benchmark` on the untouched kernel.
2. Confirm every protected case passes and each benchmark returns positive
   microsecond samples with acceptable repeatability.
3. When profiling is enabled, confirm the profile selects the intended kernel
   and labels wrapper, launch, and kernel scopes accurately.
4. Run an intentionally invalid source and confirm `build` rejects it.
5. Run an intentionally incorrect but compilable source and confirm `verify`
   rejects it.
6. Place the validated files in a run-specific directory, compute content
   hashes, and do not modify them until the optimization run finishes.

Kernel-specific optimization knowledge belongs in the bundle's
`target.json` `optimization_guidance` string. Include algorithm and layout
invariants, synchronization and aliasing contracts, exact measurement scopes,
known failed configurations, and evidence-to-action rules. Do not put such
knowledge into the generic candidate provider.

The harness is a Python module with this API:

```python
def build(kernel_source: str, target: dict):
    """Compile or materialize the candidate and return its build artifact."""


def verify(build_artifact, case: dict):
    """Return bool or {passed, diagnostics, metrics}."""


def benchmark(build_artifact, case: dict, repetitions: int):
    """Return positive microsecond samples or timing metadata."""


def profile(build_artifact, case: dict, request: dict | None = None):
    """Optional. Return a compact JSON-serializable profile mapping."""
```

The legacy `profile(build_artifact, case)` form remains supported. It is treated
as a summary profile request for the default tools selected by `target.json`.
New harnesses should accept the structured request form so the agent can request
summary, deep, or diagnostic-only profiling without changing the harness API.

A successful `build` may return the artifact directly, or:

```python
{
    "success": True,
    "artifact": artifact,
    "diagnostics": "",
}
```

A failed build returns:

```python
{
    "success": False,
    "artifact": None,
    "diagnostics": "compiler output",
}
```

A verification result should include useful numerical metrics:

```python
{
    "passed": True,
    "diagnostics": "",
    "metrics": {
        "max_abs_error": 0.003,
        "max_rel_error": 0.021,
    },
}
```

A benchmark result uses microseconds, not milliseconds:

```python
{
    "samples_us": [81.2, 80.9, 81.0, 81.1, 80.8],
    "warmup_count": 25,
    "cache_policy": "warm",
}
```

## Profile requests

A structured profile request contains these fields:

```json
{
  "level": "summary",
  "tools": ["proton"],
  "experiment_id": "baseline-or-candidate-id",
  "artifacts_dir": "/absolute/path/to/output/profiles/baseline",
  "reason": "baseline before candidate generation",
  "diagnostic_only": false
}
```

Field meanings:

- `level`: `summary` or `deep`; use `diagnostic_only: true` for
  instrumentation-only requests.
- `tools`: profiler intents requested by the agent. `proton_launch` collects
  launch attribution, while `native_profiler` is mapped by the target harness
  to its platform profiler, such as NCU on NVIDIA.
- `experiment_id`: stable baseline, round, or candidate identifier used in
  artifact names and profile metadata.
- `artifacts_dir`: absolute directory where raw profiler outputs and commands
  must be written.
- `reason`: concise explanation for why profiling was requested.
- `diagnostic_only`: `true` only for instrumentation that perturbs source,
  compilation, or timing and must never be benchmarked, promoted, or committed.

`profile` returns a vendor-neutral, JSON-serializable mapping with a declared
profiling level, exact wrapper/launch/kernel scope, normalized semantic groups,
optional raw metrics, and diagnostics for unavailable data. Summary profiling
should be inexpensive enough for every correct candidate; deep profiling may use
slower target-specific tools. Diagnostic profiling is for attribution questions
that cannot be answered by normal benchmark, wrapper, launch, and target summary
data.

Return compact normalized JSON inline. Store raw profiler output under
`artifacts_dir`, and reference those files by absolute path. Raw artifacts may
include `.hatchet`, `.chrome_trace`, target profiler reports, CSV exports,
async-task or warp mapping files, and the exact commands that generated them.
Large payloads are spilled into the agent artifact directory automatically.

Generic Proton guidance lives in
`third_party/tlx/tools/agents/kernel_optimization/docs/profiling/proton.md`.
CUDA/NVIDIA bundles must follow
`third_party/tlx/tools/agents/kernel_optimization/docs/profiling/nvidia-ncu.md`.
Other targets must not fabricate
NVIDIA fields. Unsupported counters are omitted or represented as `null` with
diagnostics, never silently converted to zero.

Before freezing a CUDA bundle, its profile implementation must satisfy these
integration checks:

- pass the target's exact main-kernel scope to
  `parse_proton_launch_attribution(..., main_scope=...)` and verify the expected
  main launch has nonzero attributed time;
- collect NCU with an exported `.ncu-rep`, then call
  `export_ncu_report_details()` to run `ncu --import ... --page details --csv`;
- never parse NCU collection stdout as metric CSV when `--export` is enabled,
  because that stream contains profiler status lines rather than metric rows;
- retain the report, import command, details CSV, and stderr as absolute artifact
  paths, and verify at least normalized NCU duration is non-null when NCU is
  available.

A compact profile response should look like:

```json
{
  "level": "summary",
  "tools": ["proton"],
  "scope": {
    "case_id": "production-shape",
    "experiment_id": "baseline",
    "kernel": "exact_kernel_name_or_null"
  },
  "summary": {
    "median_us": 81.0,
    "p95_us": 81.2
  },
  "artifacts": {
    "trace": "/absolute/path/to/profile.chrome_trace",
    "commands": "/absolute/path/to/profile.commands.txt"
  },
  "diagnostics": []
}
```

### Buck and TritonBench harnesses

For a Buck-managed kernel, `build` must make the candidate visible to the exact
Buck target without permanently modifying the user's source. Prefer an isolated
worktree or a target-specific candidate import hook. If temporary replacement is
unavoidable:

- Verify the destination path exactly.
- Save the original bytes and mode.
- Replace only the designated kernel file.
- Restore it in `finally`, including timeout and exception paths.
- Serialize evaluations so candidates cannot race on the same source file.

`verify` must run the target's real correctness check. `benchmark` must run the
existing TritonBench+Buck command with the exact `--op`, `--only`, `--mode`, and
shape/config selectors, then parse only the requested row. Do not substitute a
unit-test duration or a different provider.

## Cases metadata

`cases.json` is a list. Each case has a stable ID, arbitrary harness parameters,
a positive weight, and a correctness protection flag:

```json
[
  {
    "case_id": "production-shape",
    "parameters": {
      "m": 4096,
      "n": 4096,
      "k": 4096,
      "dtype": "float16"
    },
    "weight": 10.0,
    "protected": true
  },
  {
    "case_id": "tail-shape",
    "parameters": {
      "m": 257,
      "n": 509,
      "k": 129,
      "dtype": "float16"
    },
    "weight": 1.0,
    "protected": true
  }
]
```

Every protected case must pass. Unprotected failures remain diagnostic and do
not contribute to aggregate speedup. Weight the production workload more
heavily, but keep representative correctness/tail cases protected.

## Target metadata

`target.json` describes execution, not the workload shape:

```json
{
  "backend": "cuda",
  "architecture": "B200",
  "device": "cuda:0",
  "environment": {
    "CUDA_VISIBLE_DEVICES": "0"
  },
  "optimization_guidance": "Preserve the documented layout and synchronization contracts. Change one measured bottleneck per candidate."
}
```

Valid architecture aliases include `blackwell`, `B200`, `sm_100`, `hopper`,
`H100`, and `sm_90`. The CLI validates the visible CUDA device against the
selected architecture.

Put required runtime knobs in `environment`, for example compiler feature flags,
cache directories, or Triton dump controls. Do not put shape parameters here;
they belong in `cases.json`.

## Budget metadata

An optional `budget.json` overrides CLI budget flags:

```json
{
  "max_rounds": 5,
  "candidates_per_round": 2,
  "max_candidate_seconds": 600,
  "max_total_seconds": 3600,
  "min_speedup": 1.01,
  "max_cv": 0.10,
  "benchmark_repetitions": 10
}
```

`min_speedup` must be at least `1.0`. Lower `max_cv` for stable microbenchmarks;
raise it only when the harness documents unavoidable variance.

## Full invocation example

```bash
cd /home/hoy/triton-fb
PYTHONPATH=/home/hoy/triton-fb \
python -m third_party.tlx.tools.agents.kernel_optimization.cli \
  --kernel /absolute/path/to/kernel.py \
  --reference-kernel /absolute/path/to/reference_kernel.py \
  --harness /absolute/path/to/target-bundle/harness.py \
  --cases /absolute/path/to/target-bundle/cases.json \
  --target /absolute/path/to/target-bundle/target.json \
  --budget /absolute/path/to/target-bundle/budget.json \
  --output-dir /absolute/path/to/output \
  --arch blackwell \
  --provider codex \
  --model sonnet \
  --profile
```

Before launching, run the same harness once against the original source and
confirm that correctness, benchmark parsing, and profile collection all work.
