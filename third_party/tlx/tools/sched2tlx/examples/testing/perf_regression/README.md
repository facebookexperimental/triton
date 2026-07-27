# sched2tlx example perf/correctness suite

**`perf_harness.py`** exposes one subcommand, **`compare`**: a per-case
summary table of a git revision's committed `generated.py` against the
working tree's, both measured under the **current build** with the current
`bench_spec.py`. It answers "what did my diff do to each case's kernel?"
cheaply and exactly whenever the diff's effect is captured by the committed
fixtures.

```bash
python3 perf_harness.py compare                     # working tree vs origin/main
python3 perf_harness.py compare --rev <rev>         # vs any revision
python3 perf_harness.py compare --cases case7_wgrad_bias,case9_scaled_mm/blockwise
```

Output — one row per case, four columns:

```
┌──────────────────┬────────────────────────────┬────────────────────────────┬────────────────────────────────┐
│ case             │ main (gen/hw)              │ branch (gen/hw)            │ improvement                    │
├──────────────────┼────────────────────────────┼────────────────────────────┼────────────────────────────────┤
│ case7_wgrad_bias │ 0.83x, 0.71x, 0.70x, 0.67x │ 1.00x, 0.94x, 0.91x, 0.89x │ +20.5%, +32.4%, +30.0%, +32.8% │
└──────────────────┴────────────────────────────┴────────────────────────────┴────────────────────────────────┘
```

- Cells are per-shape **gen/handwritten throughput ratios** (`SHAPES` order);
  a case with no `handwritten.py` shows raw generated throughput instead.
- **improvement** is the per-shape % change of the branch's generated-kernel
  throughput over the revision's — computed from gen throughput directly, so
  handwritten-side noise never leaks into it.
- Byte-identical `generated.py` on both sides is measured **once** and shown
  "unchanged" — anything else would just re-measure noise. Branch cells that
  were actually re-measured (fixture differs from `--rev`) print **bold** when
  stdout is a terminal (force in pipes with `FORCE_COLOR=1`).
- **Correctness is checked before timing** (generated vs torch reference, and
  vs the handwritten output where present, within the spec's `TOL`); any
  failing shape taints the cell with FAIL. A kernel that raises renders as an
  `(error: ...)` cell rather than crashing the table.

Needs torch + triton + a Blackwell GPU (imported lazily). If a launch hangs,
run `third_party/tlx/killgpu.sh`.

## Adding a new case (so it's picked up automatically)

`bench_spec.py` files are discovered **recursively** under `examples/` — a
flat `caseN_<desc>/` dir and a nested variant dir
(`case9_scaled_mm/blockwise/`) are equally fine. Top-level `case*/` dirs
without any spec are listed as `(no bench_spec)`, never silently dropped.
To participate, a case dir ships its committed `generated.py` plus:

```python
SHAPES = [...]                       # shapes to sweep
TOL = 1e-2                           # rel-error tolerance for the correctness guard
def make_inputs(shape): ...          # -> dict of cuda tensors / scalars
def gen_call(generated, inputs): ... # launch generated kernel; return output tensor
def hw_call(handwritten, inputs): ...# launch handwritten kernel; return output tensor
def metric(shape): return (work, scale, unit)  # e.g. (2*M*N*K, 1e12, "TFLOPS")
def reference(inputs): ...           # torch reference; return output tensor
```

`hw_call` is only invoked when the case dir has a `handwritten.py`; a case
without one (case8) reports raw throughput. See
`examples/case1_simple_gemm/bench_spec.py` etc. for templates.

## History

The harness once carried four more subcommands (`bench`, `regression`, `e2e`,
`e2e-worker` — before/after emitter regression via Sapling tree
materialization, and a buck-orchestrated two-toolchain end-to-end pipeline).
They were retired in favor of `compare`; recover them from git history if the
two-toolchain path is ever needed again.
