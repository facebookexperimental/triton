# Proton Profiling Guidance

Use Proton for vendor-neutral attribution around the benchmark wrapper, launch
path, and optional diagnostic source instrumentation. Keep target-specific
counter requirements in sibling vendor guides such as `nvidia-ncu.md`; this file
only describes Proton usage and artifact expectations.

## Wrapper And Launch Attribution

Collect Proton attribution for every correctness-passing candidate and for the
baseline. This layer answers whether endpoint time is spent in the benchmark
wrapper, setup/teardown, launch path, synchronization, or the profiled kernel
window.

An ordinary Proton timeline using `hook='triton'` can show the wrapper and one
compiled Triton kernel launch. The target harness must identify that launch with
its exact scope rather than a name heuristic:

```python
attribution = parse_proton_launch_attribution(
    tree,
    main_scope=main_kernel_scope,
)
```

The scope is target knowledge and should not be added to the generic profile
request. Before freezing the harness, verify that the expected main launch has
nonzero `main_kernel_us` and helper CUDA launches are attributed as non-main.

This timeline does not show TLX async-task overlap inside the kernel. Do not
treat a single kernel region as evidence that producer, consumer, TMA, MMA, or
store tasks overlapped.

Return compact normalized JSON inline and keep raw files in the requested
`artifacts_dir`:

```json
{
  "proton": {
    "level": "summary",
    "scope": {
      "experiment_id": "candidate-r2-c1",
      "case_id": "case-id",
      "hook": "triton"
    },
    "wrapper": {
      "median_us": 81.0,
      "kernel_launches": 1,
      "kernel_window_us": 78.4
    },
    "artifacts": {
      "hatchet": "/absolute/path/to/proton.hatchet",
      "chrome_trace": "/absolute/path/to/proton.chrome_trace",
      "commands": "/absolute/path/to/proton.commands.txt"
    },
    "diagnostics": []
  }
}
```

Use absolute artifact paths for `.hatchet`, `.chrome_trace`, CSV exports, and
commands. Keep inline JSON compact; do not paste raw traces into the live log.

## Diagnostic Intra-Kernel Instrumentation

Use this layer only when wrapper attribution and target counters disagree or
when a specific async-task overlap hypothesis needs lane-level evidence.
Instrumentation perturbs source, compiler decisions, and timing, so it is
strictly diagnostic-only.

When requested, use Proton instrumentation mode with:

```python
backend = "instrumentation"
data = "trace"
granularity = "warp"
```

Enable Triton semantic explicitly for source-level scopes. Group warp lanes by
known async-task warp ranges from the kernel source, generated metadata, or a
saved mapping artifact. Persist the mapping and the instrumentation command as
absolute artifact paths so another agent can reproduce the grouping.

Do not request `warp_group` granularity because the runtime rejects it today.
Do not infer async-task overlap from CTA-level or kernel-level scopes.

## Diagnostic Safety

Compiler transforms may move, clone, merge, or delete semantic scopes. Treat
instrumented source locations as best-effort labels, not an ownership proof.
Always cross-check surprising ranges against the generated IR or a mapping file
before changing scheduling, barrier, or buffering code.

Instrumented source and timing must never be:

- used as benchmark or speedup evidence;
- promoted as a candidate;
- committed to the user's source tree;
- compared numerically against non-instrumented benchmark samples.

A diagnostic response should set `diagnostic_only` to `true` and include that in
the returned JSON:

```json
{
  "proton": {
    "level": "diagnostic",
    "diagnostic_only": true,
    "instrumentation": {
      "backend": "instrumentation",
      "data": "trace",
      "granularity": "warp",
      "triton_semantic": true,
      "warp_mapping": "/absolute/path/to/async_task_warp_mapping.json"
    },
    "artifacts": {
      "chrome_trace": "/absolute/path/to/proton.instrumented.chrome_trace",
      "commands": "/absolute/path/to/proton.instrumented.commands.txt"
    },
    "diagnostics": [
      "instrumented timing is diagnostic-only and must not be benchmarked"
    ]
  }
}
```
