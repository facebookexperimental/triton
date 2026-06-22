# sched2tlx example regression suite

A generic **before-vs-after** harness for the sched2tlx emitter. Given any diff
that touches the tool (default: the current commit) it re-emits every example
case with the emitter *before* and *after* the change and checks:

- **correctness** — the after-diff generated kernel agrees with the case's
  hand-written reference (both pass the case's own `run_*.py` against the torch
  reference, so they agree transitively);
- **performance** — the after-diff kernel is no slower than the before-diff
  kernel, per shape, within a tolerance.

It is generic in two ways: it works for **any diff** (revisions are extracted
via Sapling), and it **auto-discovers all cases** under `examples/` — a new
`caseN_*` directory is included with no change to the harness.

## Running

The recommended way is the buck target (run from anywhere inside your fbsource
checkout — the repo root is auto-detected from the working directory):

```bash
# default: test the current commit (before = .^, after = .)
buck2 run @fbcode//mode/dev-nosan //third-party/triton/beta/triton:sched2tlx_regression

# a specific diff / subset of cases / looser perf tolerance
buck2 run @fbcode//mode/dev-nosan //third-party/triton/beta/triton:sched2tlx_regression -- --diff D108804400
buck2 run @fbcode//mode/dev-nosan //third-party/triton/beta/triton:sched2tlx_regression -- --cases case1_simple_gemm,case3_FA --perf-tol 0.10
```

`@fbcode//mode/dev-nosan` is required: the default dev mode links ASAN, which
disables CUDA in torch. Pass `--repo-root <path>` only if running from outside a
checkout. With a plain triton-enabled interpreter you can also run
`python3 run_regression.py --diff D108804400` directly.

Correctness and perf need a Blackwell GPU (gated; reported as SKIP otherwise).
If a launch hangs, run `third_party/tlx/killgpu.sh`.

The generic perf engine can also be run on its own to compare a case's committed
`generated.py` against its handwritten reference (the classic
`perf_generated_vs_handwritten.py` behavior):

```bash
python3 perf_engine.py bench --cases case3_FA
```

## Files

- `emit_helpers.py` — materializes coherent before/after copies of the tool tree
  for any diff (Sapling `sl status --change` + `sl cat`) and re-emits each case's
  `generated.py` on each side. Pure stdlib.
- `perf_engine.py` — generic perf engine: CUDA-event timing, shape sweep,
  correctness guard, throughput/ratio reporting. `worker` mode benchmarks one
  `generated.py` and writes JSON (the driver runs it in a separate process per
  side so a kernel that faults at scale can't poison the other run).
- `run_regression.py` — the driver: discover cases → correctness → perf → table.

## Adding a new case (so it's picked up automatically)

A `caseN_<desc>/` directory under `examples/` is auto-discovered. To participate:

1. Ship `schedule_graph.json` (emitter input) and `run_generated.py` that exits 0
   iff the generated kernel is correct vs a torch reference (optionally
   `run_handwritten.py` likewise for the reference).
2. Ship a small `bench_spec.py` for the perf check, exposing:

   ```python
   SHAPES = [...]                       # shapes to sweep
   TOL = 1e-2                           # rel-error tolerance for the correctness guard
   def make_inputs(shape): ...          # -> dict of cuda tensors / scalars
   def gen_call(generated, inputs): ... # launch generated kernel; return output tensor
   def hw_call(handwritten, inputs): ...# launch handwritten kernel; return output tensor
   def metric(shape): return (work, scale, unit)  # e.g. (2*M*N*K, 1e12, "TFLOPS")
   def reference(inputs): ...           # torch reference; return output tensor
   ```

   See `examples/case1_simple_gemm/bench_spec.py` etc. for templates.

A case whose emitter output still contains an `<..._unsupported>` placeholder is
auto-skipped (no perf/correctness run) until the emitter gap is closed.
```
