---
name: sched2tlx-perf-testing
description: >
  Run the sched2tlx perf/correctness harness over the modulo-scheduling
  example corpus (case1-9: GEMM, persistent GEMM, FA fwd/bwd, addmm+bias,
  LayerNorm, wgrad+bias, multiphase GEMM, scaled_mm). Use when the user asks
  to benchmark generated-vs-handwritten kernels, check corpus correctness,
  compare emitter revisions, or regenerate schedule_graph.json fixtures.
  Never run perf unless explicitly asked.
disable-model-invocation: true
---

# sched2tlx Perf & Correctness Harness

**Never run performance tests unless the user explicitly asks.**

Harness: `third_party/tlx/tools/sched2tlx/examples/testing/perf_regression/perf_harness.py`
Corpus: `third_party/tlx/tools/sched2tlx/examples/case*/`

## Performance testing prerequisites

Before running any performance test for a C++ scheduler change, rebuild
Triton, then regenerate both `schedule_graph.json` and `generated.py` for
every case being tested. Do not benchmark stale fixtures.

## Build and execution priority (critical)

Use one build path consistently for compilation, fixture regeneration,
correctness, and timing:

1. **Buck first.** If `buck2` is available, use `buck2 run` to compile and
   run every performance test, including `compare` and per-case benchmarks.
   Load the `running-with-buck` skill for the required working directory,
   `@mode/opt`, beta-Triton modifier, GPU architecture, and CUDA flags. Do not
   silently fall back to the repo venv when Buck is available. If the requested
   benchmark has no runnable Buck target, report the missing target instead.
2. **Repo venv fallback.** Only when `buck2` is unavailable, build and run all
   performance tests with `$REPO/.venv`. The login shell's lmod modules break
   both build and runtime, so prefix every Python and `triton-opt` invocation
   with `env -u LD_LIBRARY_PATH`. Ignore the lua/posix noise every command
   prints. Use one of the repo-venv rebuild methods below.

### Rebuilding Triton

- **Buck build.** On the Buck path, `buck2 build` rebuilds the selected target
  and its changed C++ dependencies incrementally; `buck2 run` does the same
  before running it. Follow the `running-with-buck` skill and never guess a
  target name. A Buck rebuild does not update source-tree `schedule_graph.json`
  or `generated.py` unless the selected target explicitly regenerates them.
- **Incremental repo-venv build.** After the development build has already been
  initialized, rebuild C++ changes from the Triton root with:
  ```bash
  env -u LD_LIBRARY_PATH \
    PATH="$REPO/.venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
    VIRTUAL_ENV="$REPO/.venv" PYTHON="$REPO/.venv/bin/python" make
  ```
- **Repo-venv editable rebuild.** Use this to initialize or refresh the
  editable Triton installation used by the fallback workflow:
  ```bash
  env -u LD_LIBRARY_PATH \
    PATH="$REPO/.venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
    VIRTUAL_ENV="$REPO/.venv" CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS=14 \
    uv pip install -e . --no-build-isolation
  ```
  `make dev-install-triton` is the Makefile wrapper for the same editable
  installation flow when `PYTHON` points to `$REPO/.venv/bin/python`.

Regardless of the rebuild path, regenerate `schedule_graph.json` and
`generated.py` for every selected case before performance testing a scheduler
change.

## The one command: compare

When Buck is available, run the applicable performance-runner target from
`fbsource/fbcode`, following the `running-with-buck` skill, and pass
`compare`, `--rev`, and `--cases` as program arguments after `--`.

Only when Buck is unavailable, run from
`examples/testing/perf_regression/`:

```
env -u LD_LIBRARY_PATH $REPO/.venv/bin/python perf_harness.py compare \
    [--rev origin/main] [--cases case7_wgrad_bias,case9_scaled_mm/blockwise]
```

One row per case, four columns:

| column | meaning |
|---|---|
| `case` | case dir relative to examples/ (nested variants like `case9_scaled_mm/blockwise` included) |
| `main (gen/hw)` | per-shape gen/handwritten throughput ratios for `--rev`'s committed generated.py (default origin/main) |
| `branch (gen/hw)` | the same for the working tree's generated.py |
| `improvement` | per-shape % change of the branch's GENERATED-kernel throughput vs `--rev`'s (positive = branch faster) |

Semantics:

- `bench_spec.py` files are discovered RECURSIVELY under examples/; top-level
  `case*/` dirs without any spec are listed as `(no bench_spec)`, never
  silently dropped. All of case1–case9 currently have specs.
- Compare fixture identity using `generated.py`, not `schedule_graph.json`:
  JSON op ids are pointer-derived and unstable across regenerations, while
  byte-identical generated source means the kernels are identical. When the
  revision's and working tree's `generated.py` are byte-identical, benchmark
  the revision's gen/hw result once for the left column, skip a duplicate
  working-tree benchmark, show `unchanged` in the branch column, and show `-`
  for improvement.
- Cases without a wired handwritten baseline (currently case8, which has no
  `handwritten.py`) show raw generated TFLOPS instead of a gen/hw ratio; the
  improvement column still works. case9_scaled_mm/blockwise wires `hw_call`
  to `handwritten.blackwell_scaled_mm_ws`, so it reports gen/hw ratios.
- Correctness (vs torch reference, and vs handwritten output where present)
  is checked before timing; any failing shape appends FAIL to the cell.
  A kernel that raises shows an `(error: ...)` cell instead of crashing the
  table.
- Both columns run under the CURRENT build — compare tests committed KERNEL
  fixtures, not toolchains. To evaluate a C++ scheduler change you must
  rebuild first and regenerate fixtures (below).

Deep-dive per-case scripts (outside the harness): case4
`perf_generated.py` (gen vs no-WS vs handwritten WS) and `run_generated.py`
(all three gradients); case8 `bench_general.py` (all three outputs +
pool-vs-sum A/B); any case's `run_*.py` runner for correctness-only
(case8's is `run_triple_gemm_nows.py`).

## Scheduler provenance

The corpus fixtures (`schedule_graph.json` and the committed `generated.py`)
are produced by the **Modulo Scheduling** pass.

## Regenerating fixtures

When Buck is available, regenerate fixtures with the Buck-built beta
`triton-opt` and the applicable sched2tlx Buck runner, following the same
build-path rule above. The commands below are only for the no-Buck venv
fallback:

```
TRITON_MODULO_DUMP_SCHEDULE=<case>/schedule_graph.json \
  build/cmake.*/bin/triton-opt -allow-unregistered-dialect \
  --nvgpu-modulo-schedule <case>/<kernel>_pre_modulo.ttgir -o /dev/null
env -u LD_LIBRARY_PATH PYTHONPATH=third_party/tlx/tools/sched2tlx \
  $REPO/.venv/bin/python -m sched2tlx <case>/schedule_graph.json -o <case>/generated.py
```
JSON op ids are pointer-derived and never byte-stable — regen always churns
`schedule_graph.json`; the meaningful diff and benchmark-identity signal is
`generated.py`.
Known: case3 may need `TRITON_MODULO_SELECT_VARIANT=2`; case2 fixtures are
ancient (fresh dumps differ, pre-existing); case8's committed generated.py
predates the emitter's multiphase support landing (regen produces a
single-phase kernel — don't "refresh" it casually).

## Benchmark methodology

- Use `triton.testing.do_bench` for every timing run. Configure a nonzero
  `warmup` before measurement; measured iterations must clear L2 before each
  invocation. Do not add ad-hoc CUDA-event timing loops.
- Check `nvidia-smi` first; if a run hangs for minutes, run
  `third_party/tlx/killgpu.sh`.
- One bench at a time — timing runs must not share the GPU.
