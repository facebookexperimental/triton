---
name: sched2tlx-perf-testing
description: >
  Run the sched2tlx perf/correctness harness over the modulo-scheduling
  example corpus (case1-8: GEMM, persistent GEMM, FA fwd/bwd, addmm+bias,
  LayerNorm, wgrad+bias, multiphase GEMM). Use when the user asks to
  benchmark generated-vs-handwritten kernels, check corpus correctness,
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

## Subcommands

Run from `examples/testing/perf_regression/`:

- **bench** — committed `generated.py` vs `handwritten.py`, per-case table
  (`shape | gen | hw | gen/hw | rel | ok`; correctness is checked before
  timing). Auto-discovers cases with a `bench_spec.py`
  (currently case1,2,3,5,6,7):
  `env -u LD_LIBRARY_PATH $REPO/.venv/bin/python perf_harness.py bench [--cases case6_layernorm,...]`
- **compare** — one row per case, two columns: `--rev`'s committed
  `generated.py` (default `origin/main`) vs the working tree's, cells =
  per-shape gen/handwritten ratios. Scans EVERY `case*/` in examples/
  (cases without bench_spec are listed as such); byte-identical kernels are
  measured once and shown "unchanged". Both columns run under the CURRENT
  build — it compares fixtures, not toolchains (use `regression`/`e2e` for
  that): `... perf_harness.py compare [--rev origin/main] [--cases ...]`
- **regression** — before/after emitter check for a diff that touches
  sched2tlx (re-emits every case both sides; only true regressions fail):
  `... perf_harness.py regression [--diff REV | --before X --after Y] [--perf-tol 0.05]`
- **e2e** — full pipeline from TTGIR (dump → emit → bench) with the current
  build: `... perf_harness.py e2e --skip-master --cases case6_layernorm`.
  The master-comparison mode (`--master-repo`) needs a separately built
  master checkout and is buck/fbsource-oriented.

## Cases the harness does NOT cover

- **case4_FA_bwd**: `cd examples/case4_FA_bwd && ... python perf_generated.py`
  (gen vs no-WS baseline vs handwritten WS); correctness via `run_generated.py`.
- **case8_multiphase_gemm**: `cd examples/case8_multiphase_gemm && ... python
  bench_general.py` (correctness + pool-vs-sum A/B; no handwritten reference).
- Correctness-only for any case: `cd examples/<case> && ... python run_generated.py`.

## Comparing against another revision's kernel

`run_bench(case_dir, generated_path)` takes the generated.py path separately,
so to bench another revision's kernel under identical methodology:
`git show <rev>:<path>/generated.py > /tmp/gen.py`, then in python:
`perf_harness._print_table(perf_harness.run_bench(CASE_DIR, Path("/tmp/gen.py")))`.
Cheap shortcut when only fixtures differ: kernels whose generated.py is
byte-identical across revisions need no re-measurement.

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
ancient (fresh dumps differ, pre-existing).

## Methodology warnings

- `bench`/`run_bench` use `triton.testing.do_bench` (cold-L2). Ad-hoc
  CUDA-event loops (e.g. `perf_generated_vs_handwritten.py`) are hot-L2 and
  read higher. NEVER cross-compare numbers from the two.
- Check `nvidia-smi` first; if a run hangs for minutes, run
  `third_party/tlx/killgpu.sh`.
- One bench at a time — timing runs must not share the GPU.
