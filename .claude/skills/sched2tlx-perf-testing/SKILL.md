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

## Environment (this cluster — critical)

The login shell's lmod modules break both build and runtime. Prefix EVERY
python/triton-opt invocation with `env -u LD_LIBRARY_PATH` and use the repo
venv python (`$REPO/.venv/bin/python`). Ignore the lua/posix noise every
command prints. C++ changes require rebuild first:
`env -u LD_LIBRARY_PATH PATH="$REPO/.venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" VIRTUAL_ENV=$REPO/.venv CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS=14 uv pip install -e . --no-build-isolation`

## The one command: compare

Run from `examples/testing/perf_regression/`:

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
- Byte-identical generated.py on both sides → measured once, shown
  "unchanged" (improvement `-`).
- Cases without a wired handwritten baseline (case8 has no `handwritten.py`;
  case9_scaled_mm/blockwise has the file but its spec's `hw_call` is not
  wired yet) show raw generated TFLOPS instead of a gen/hw ratio; the
  improvement column still works.
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

## Scheduler provenance: Modulo Scheduling vs Joint Solver

Today the corpus fixtures (`schedule_graph.json` and the committed
`generated.py`) are produced by the **Modulo Scheduling** pass — it is the
only scheduler in the codebase, so `compare` numbers are unambiguous. A
**Joint Solver** may be introduced later as an alternative scheduler
producing the same artifacts. If you find that the codebase contains BOTH
the Modulo Scheduling code and Joint Solver code (e.g. a joint-solver
module or pass exists in the source tree alongside the modulo scheduler),
do not guess: **ask the user whether the comparison is about Modulo
Scheduling kernels or Joint Solver kernels** (i.e. which scheduler
produced — or should regenerate — the fixtures being measured) before
running `compare`.

## Regenerating fixtures

```
TRITON_MODULO_DUMP_SCHEDULE=<case>/schedule_graph.json \
  build/cmake.*/bin/triton-opt -allow-unregistered-dialect \
  --nvgpu-modulo-schedule <case>/<kernel>_pre_modulo.ttgir -o /dev/null
env -u LD_LIBRARY_PATH PYTHONPATH=third_party/tlx/tools/sched2tlx \
  $REPO/.venv/bin/python -m sched2tlx <case>/schedule_graph.json -o <case>/generated.py
```
JSON op ids are pointer-derived and never byte-stable — regen always churns
schedule_graph.json; the meaningful diff signal is `generated.py`.
Known: case3 may need `TRITON_MODULO_SELECT_VARIANT=2`; case2 fixtures are
ancient (fresh dumps differ, pre-existing); case8's committed generated.py
predates the emitter's multiphase support landing (regen produces a
single-phase kernel — don't "refresh" it casually).

## Methodology warnings

- `compare`/`run_bench` use `triton.testing.do_bench` (cold-L2). Ad-hoc
  CUDA-event loops (e.g. `perf_generated_vs_handwritten.py`) are hot-L2 and
  read higher. NEVER cross-compare numbers from the two.
- Check `nvidia-smi` first; if a run hangs for minutes, run
  `third_party/tlx/killgpu.sh`.
- One bench at a time — timing runs must not share the GPU.
