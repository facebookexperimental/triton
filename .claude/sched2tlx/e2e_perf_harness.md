---
description: How to use the end-to-end performance + correctness harness for sched2tlx example kernels. It runs the full pipeline (pre_modulo.ttgir → modulo schedule → TLX emit → GPU run) on BOTH the current branch and master-latest, and prints a table of each branch's throughput as a percentage of the hand-written reference plus the current/master speedup. Correctness is a hard gate; performance is report-only. Machine-agnostic across B200/B300/B100.
---

# sched2tlx End-to-End Perf Harness

A one-click, machine-agnostic, CI-bot-style harness that answers: *does my
change help or hurt end-to-end performance vs master, and is it still correct?*
For every in-scope example case it runs the **whole** pipeline starting from the
case's `*pre_modulo.ttgir`, on **two built toolchains** (current branch and
master-latest), and prints a comparison table.

This is distinct from the before-vs-after **regression** harness (the
`regression` subcommand of `examples/testing/perf_regression/perf_harness.py`),
which only re-runs the Python emitter and treats `schedule_graph.json` as a
checked-in fixture. This harness starts from the TTGIR and rebuilds the compiler
on each branch, so it captures changes to **both** the C++ modulo scheduler and
the Python emitter.

## What it runs (per case, per branch)

```
pre_modulo.ttgir                       (the fixed source of truth)
  → triton-opt --nvgpu-modulo-schedule (C++ modulo pass → schedule_graph.json)
  → python -m sched2tlx                 (emitter → generated.py TLX)
  → triton compile + GPU run            (triton.testing.do_bench; correctness check)
```

Only the **toolchain under test** varies between branches — the branch's compiled
`triton-opt`, its `sched2tlx` emitter, and its `triton-py`. The **test
definition** is held fixed: the TTGIR input, `bench_spec.py`, and
`handwritten.py` are always read from the *current* checkout, so master and
current are measured against an identical benchmark and reference even if master
predates those files.

## Files

Everything lives in one file,
`third_party/tlx/tools/sched2tlx/examples/testing/perf_regression/perf_harness.py`,
organized into banner-commented sections and dispatched by subcommand:

- `e2e` subcommand — the host orchestrator (pure stdlib). Detects GPU arch, picks
  a free GPU, prepares master-latest in a separate checkout, `buck2 run`s the
  per-branch worker with dynamic arch/CUDA, merges the JSON results, prints the
  table, and applies the correctness gate. **Run it with plain `python3`, not via
  buck** — it coordinates two checkouts, each with its own buck build.
- `e2e-worker` subcommand — the per-branch worker, run inside the `sched2tlx_e2e`
  buck `python_binary` (invoked by the `e2e` orchestrator, never by hand). For
  each case it dumps the schedule via this branch's `triton-opt`, emits
  `generated.py` via this branch's emitter, then runs a fork-isolated `run_bench`,
  and writes one branch JSON.
- Perf-engine section — reused: per-shape `do_bench`, throughput, rel-error vs
  torch and vs handwritten (`run_bench`), plus the standalone `bench` subcommand.
- Each case's `bench_spec.py` — reused: `SHAPES`, `TOL`, `make_inputs`,
  `gen_call`, `hw_call`, `metric`, `reference`.

Buck target: `fbsource//third-party/triton/beta/triton:sched2tlx_e2e` (defined in
`third-party/triton/beta/BUCK.template`; ships `triton-opt` as a resource; no
baked-in CUDA modifier — arch/CUDA are passed at run time so it works on any
Blackwell part).

## How to run

From anywhere inside your **current** checkout. `--master-repo` must be a
**separate** fbsource checkout (the harness `sl pull`s + `sl goto`s master there;
it never mutates your working checkout).

```bash
cd third_party/tlx/tools/sched2tlx/examples/testing/perf_regression

# full run: all in-scope cases, current vs master-latest
python3 perf_harness.py e2e \
    --current-repo /data/users/$USER/fbsource \
    --master-repo  /home/$USER/fbsource

# a subset of cases
python3 perf_harness.py e2e --master-repo /home/$USER/fbsource --cases case1_simple_gemm,case3_FA

# pin an explicit master revision (reproducible CI)
python3 perf_harness.py e2e --master-repo /home/$USER/fbsource --master-rev <node>

# quick single-branch smoke (no master build) — do this first
python3 perf_harness.py e2e --skip-master --cases case1_simple_gemm
```

`--current-repo` defaults to auto-detection from the working directory. Other
flags: `--arch`/`--cuda` (override the auto-detected buck arch/CUDA),
`--gpu <idx>` (override GPU selection), `--keep` (keep the temp JSON dir for
debugging).

### Machine-agnostic arch selection

The GPU arch (and required CUDA version) are auto-detected from `nvidia-smi` and
passed to buck dynamically:

| GPU (compute capability)         | `nvcc_arch` | CUDA version |
|----------------------------------|-------------|--------------|
| Hopper H100 (9.0)                | `h100a`     | 12.8         |
| B200 / GB200 / B100 (10.0)       | `b200a`     | 12.8         |
| B300 / GB300 (10.3 / 11.x)       | `b300a`     | 13.0         |

Override with `--arch`/`--cuda` if detection is wrong for your host.

### GPU selection

The `e2e` orchestrator picks a free GPU using **only `nvidia-smi`** (no torch): it excludes
GPUs with a running compute process, then takes the one with the least memory
used, and pins `CUDA_VISIBLE_DEVICES` for both buck runs. (It deliberately does
**not** use `find_working_gpu.sh`, whose probe imports torch under a plain
`python`, which is unavailable on stdlib-only devservers.) The real busy-check
happens when the worker launches the kernel. If a launch hangs, run
`third_party/tlx/killgpu.sh`.

## Output

```
=== sched2tlx e2e: master(<node>) vs current(<node>)  arch=b200a cuda=12.8 gpu=0 ===

case                  shape               master%hw  current%hw  speedup  corr
------------------------------------------------------------------------------
case6_layernorm       (65536,512)         85%        110%        1.29x    PASS
case3_FA              (1,32,8192)          92%        115%        1.25x    PASS
...

OK: 0 correctness failure(s)
```

- **master%hw** / **current%hw** — that branch's generated-kernel throughput as a
  percentage of the case's hand-written reference on the same shape.
- **speedup** — current throughput ÷ master throughput (`>1.00x` = current is
  faster than master).
- **corr** — `PASS`/`FAIL` for that shape (see gating below); `SKIP` for a case
  that could not run (see skip reasons); a branch-specific error is annotated,
  e.g. `FAIL [master: modulo dump failed: ...]`.

### Correctness gate (hard) vs performance (report-only)

Correctness is checked on **both** branches, per case/shape: the generated
output must match the torch reference **and** the hand-written reference within
the case's `TOL` (and be NaN-free). Any correctness failure — or a case that was
expected to run but errored — makes the whole run **exit non-zero**. Performance
is never a gate: regressions and improvements are just printed. This matches the
CI intent "if correctness breaks, error out; otherwise just report the perf
table."

A case is **skipped** (not failed) when it has no `*pre_modulo.ttgir`, no
`bench_spec.py`, or the emitter still emits an `<..._unsupported>` placeholder.

## In-scope cases

Cases that ship both a `*pre_modulo.ttgir` and a `bench_spec.py`:
`case1_simple_gemm`, `case2_persistent_gemm`, `case3_FA`, `case6_layernorm`.
`case5_addmm_bias` and `case7_wgrad_bias` are not yet configured (no
`bench_spec.py`) and are skipped. To add a case, see "Adding a new case" in
`examples/testing/perf_regression/README.md` — it is picked up automatically once
it ships those two files.

## Manual per-step debugging (single case)

If the harness fails on a case, reproduce each stage by hand (from a built
`triton-opt` / triton-py env):

```bash
# 1. schedule dump (the real env var — design.md documents wrong names)
TRITON_MODULO_DUMP_SCHEDULE=/tmp/g.json \
  triton-opt -allow-unregistered-dialect --nvgpu-modulo-schedule \
    examples/case1_simple_gemm/*pre_modulo.ttgir -o /dev/null

# 2. emit TLX
python -m sched2tlx /tmp/g.json -o /tmp/generated.py

# 3. bench the committed generated.py vs handwritten (human table)
python3 perf_harness.py bench --cases case1_simple_gemm
```

See `generating_ddg_json.md` for how `triton-opt` is built and how the
`TRITON_MODULO_DUMP_*` dumps are driven, and `ddg_validation_harness.md` for the
sibling DDG-fidelity check.

## Prerequisites & known blockers

- **Blackwell GPU** required for the actual perf/correctness run (arch detection
  gates this).
- **Two checkouts.** `--master-repo` needs a second fbsource checkout to build
  master-latest without disturbing your working copy. Building master cold is the
  long pole (~20–40 min); buck caches artifacts when master is close to current.
  Pin `--master-rev` for reproducibility.
- **`buck2` local execution must work.** Building `triton-py` runs a
  `src_hash.txt` genrule that passes every triton source filename on one
  `execve` command line. On a checkout where that command line exceeds the kernel
  limit (`131072` chars) the build fails — locally as
  `Spawning executable /usr/bin/env failed: Failed to spawn a process`, on RE as
  `Command line argument or env too long`. This is a pre-existing triton build
  infra issue (independent of this harness — it also blocks
  `sched2tlx_regression`); if you hit it, report it to the triton/buck oncall
  (the fix is to pass `SRCS` via an argfile/manifest rather than argv). Diagnose
  with `buck2 build --remote-only fbsource//third-party/triton/beta/triton:src_hash.txt`,
  which surfaces the explicit length error.
