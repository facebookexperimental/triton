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

## One file, four subcommands

Everything lives in **`perf_harness.py`**. The first CLI argument selects a
subcommand:

| subcommand    | what it does                                                    | how to run                       |
|---------------|-----------------------------------------------------------------|----------------------------------|
| `regression`  | before-vs-after emitter check for a diff (Python emitter only)  | buck (`sched2tlx_regression`) or `python3` |
| `e2e`         | current-branch-vs-master end-to-end comparison (full pipeline)  | plain `python3` (host orchestrator) |
| `e2e-worker`  | per-branch worker invoked by `e2e` via `buck2 run`              | buck (`sched2tlx_e2e`), not run by hand |
| `bench`       | benchmark a case's committed `generated.py` vs its handwritten  | plain `python3`                  |

The module is import-safe under a plain interpreter: torch/triton are imported
lazily (only inside the benchmarking functions), and the tree-materialization
and host-orchestration halves are pure stdlib.

## End-to-end: current branch vs master (`e2e`)

The `e2e` subcommand is a separate, heavier path for the question "does my change
help or hurt end-to-end performance vs master?". Unlike `regression` (which only
re-runs the Python emitter and treats `schedule_graph.json` as a fixture), `e2e`
starts from each case's **`*pre_modulo.ttgir`** and runs the *whole* pipeline on
**two built toolchains**:

```
pre_modulo.ttgir
  → triton-opt --nvgpu-modulo-schedule   (C++ modulo pass → schedule_graph.json)
  → python -m sched2tlx                   (emitter → generated.py TLX)
  → triton compile + GPU run              (measure throughput; check correctness)
```

It builds and runs this on the **current** checkout and on **master-latest**
(built from a *separate* checkout, never by mutating your working copy), so the
comparison captures changes to both the C++ modulo scheduler and the Python
emitter. Both branches are measured against the *same* test definition — the
TTGIR input, `bench_spec.py` and `handwritten.py` are always read from the
current checkout — so only the toolchain under test differs.

It is **machine-agnostic**: the GPU arch (and required CUDA version) are
auto-detected and passed to buck dynamically, so the same command runs on
B200/GB200, B300/GB300 and B100. A free GPU is auto-picked.

```bash
# from anywhere inside your current checkout; --master-repo is a SECOND checkout
python3 perf_harness.py e2e \
    --current-repo /data/users/$USER/fbsource \
    --master-repo  /home/$USER/fbsource

# a subset of cases, or pin an explicit master revision
python3 perf_harness.py e2e --master-repo /home/$USER/fbsource --cases case1_simple_gemm,case3_FA
python3 perf_harness.py e2e --master-repo /home/$USER/fbsource --master-rev <node>

# quick single-branch sanity (no master build)
python3 perf_harness.py e2e --skip-master --cases case1_simple_gemm
```

Output is a table of each branch's throughput as a percentage of the
hand-written reference plus the current/master speedup:

```
case                  shape               master%hw  current%hw  speedup  corr
case6_layernorm       (65536,512)             85%        110%      1.29x   PASS
```

**Correctness is a hard gate** — a generated kernel that disagrees with the
torch reference *or* the hand-written reference beyond the case's `TOL` (on
either branch) makes the run exit non-zero. **Performance is report-only**
(never gates). Arch → (`nvcc_arch`, cuda) mapping: Hopper→`h100a`/12.8;
B200/GB200/B100→`b200a`/12.8; B300/GB300→`b300a`/13.0 (override with
`--arch`/`--cuda`). If a launch hangs, run `third_party/tlx/killgpu.sh`.

The `e2e` orchestrator builds and `buck2 run`s the `sched2tlx_e2e` binary once
per branch, passing the `e2e-worker` subcommand — you never invoke `e2e-worker`
by hand. Manual per-step debugging (single case) mirrors the pipeline: run
`triton-opt --nvgpu-modulo-schedule` with `TRITON_MODULO_DUMP_SCHEDULE=<path>`
on the `.ttgir`, then `python -m sched2tlx <path> -o generated.py`, then
`python3 perf_harness.py bench --cases <case>`.

## Running (before-vs-after regression)

The recommended way is the buck target (run from anywhere inside your fbsource
checkout — the repo root is auto-detected from the working directory). Pass the
`regression` subcommand after `--`:

```bash
# default: test the current commit (before = .^, after = .)
buck2 run @fbcode//mode/dev-nosan //third-party/triton/beta/triton:sched2tlx_regression -- regression

# a specific diff / subset of cases / looser perf tolerance
buck2 run @fbcode//mode/dev-nosan //third-party/triton/beta/triton:sched2tlx_regression -- regression --diff D108804400
buck2 run @fbcode//mode/dev-nosan //third-party/triton/beta/triton:sched2tlx_regression -- regression --cases case1_simple_gemm,case3_FA --perf-tol 0.10
```

`@fbcode//mode/dev-nosan` is required: the default dev mode links ASAN, which
disables CUDA in torch. Pass `--repo-root <path>` only if running from outside a
checkout. With a plain triton-enabled interpreter you can also run
`python3 perf_harness.py regression --diff D108804400` directly.

Correctness and perf need a Blackwell GPU (gated; reported as SKIP otherwise).
If a launch hangs, run `third_party/tlx/killgpu.sh`.

The generic perf engine can also be run on its own to compare a case's committed
`generated.py` against its handwritten reference (the classic
`perf_generated_vs_handwritten.py` behavior):

```bash
python3 perf_harness.py bench --cases case3_FA
```

## `perf_harness.py` structure

The single module is organized into clearly-banner-commented sections:

- **Tree materialization** — materializes coherent before/after copies of the
  tool tree for any diff (Sapling `sl status --change` + `sl cat`) and re-emits
  each case's `generated.py` on each side. Pure stdlib.
- **Perf engine** — CUDA-event timing, shape sweep, correctness guard,
  throughput/ratio reporting (`run_bench`); backs both `regression` and
  `e2e-worker`, and the standalone `bench` subcommand.
- **Regression driver** (`regression`) — discover cases → correctness → perf →
  table. Runs everything in-process (works under a plain interpreter or a par).
- **End-to-end worker** (`e2e-worker`) — per-branch worker: dumps the schedule
  from a case's `*pre_modulo.ttgir` via this branch's `triton-opt`, emits
  `generated.py` via this branch's emitter, then runs a fork-isolated bench.
  Runs inside the `sched2tlx_e2e` buck `python_binary`.
- **End-to-end orchestrator** (`e2e`) — pure-stdlib host script: detects GPU
  arch, picks a free GPU, prepares a master-latest checkout, `buck2 run`s the
  worker per branch with dynamic arch/cuda, and prints the master-vs-current
  table. Not part of the buck target — run it with plain `python3`.

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
