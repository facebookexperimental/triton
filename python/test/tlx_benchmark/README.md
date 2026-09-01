# TLX op perf suite

Performance and compile-time guardrails for the blessed `tlx.ops` catalog.
Correctness lives in `python/test/unit/tlx_ops/`; this suite assumes it passes.

Depends on `torch` and `triton` only. Where behaviour is taken from tritonbench
it is **ported, not imported** — each port names its source in a comment so the
drift is visible.

## Running

Always under `denoise.sh`, which fixes the power cap and binds to the GPU-local
NUMA node. Numbers taken without it are not comparable to anything.

The deterministic command — what CI and any review agent should use:

```bash
CUDA_VISIBLE_DEVICES=0 third_party/tlx/denoise.sh \
    python python/test/tlx_benchmark/bench_mm.py \
        --measure latency --space full --guard enforce --json /tmp/mm.json
```

| flag | choices | meaning |
|---|---|---|
| `--measure` | `all` `latency` `compile` | defaults to `all`; cheap at `--space heuristic` (~0.7 s cold compile per case), ~4 min **per case** at `--space full` |
| `--space` | `full` `heuristic` `smoke` | defaults to `heuristic`, matching `tlx.ops.mm` |
| `--dtype` | `fp16` `bf16` `both` | |
| `--guard` | `off` `report` `enforce` | `enforce` exits non-zero on a regression or compile-cap breach |
| `--replicates` | int | independent measurements per case; what the gate reads (default 10, each >=1000 timed iterations) |
| `--json` | path | machine-readable artifact (defaults to `/tmp/tlx_benchmark/mm.sm100.json`) |
| `--update-baseline` | | record this run; refuses noisy and host-bound cases |
| `--strict-env` | | fail rather than warn when the environment is not denoised |

`pytest` is the secondary front end, for the junitxml the b200 reporting
pipeline already consumes. It takes exactly the same options, and
`test_ops_perf.py` discovers every `bench_*.py`, so there is still one benchmark
file per op:

```bash
CUDA_VISIBLE_DEVICES=0 third_party/tlx/denoise.sh \
    python -m pytest python/test/tlx_benchmark/test_ops_perf.py --guard enforce
```

The harness's own unit tests need no GPU:

```bash
python -m pytest python/test/tlx_benchmark/test_harness.py \
    python/test/tlx_benchmark/test_denoise.py \
    python/test/tlx_benchmark/test_compile.py \
    python/test/tlx_benchmark/test_baseline.py
```

## Status

| Phase | Content | State |
|---|---|---|
| 0 | contract + shared shape list | done |
| 1 | `measure.py` — latency, replicates, outlier rejection, host-overhead | done |
| 2 | `denoise.py` — environment verification, operating-point trace | done |
| 3 | `compile.py` — `t_cold`, absolute 120 s cap | done |
| 4 | `bench_mm.py` — flagship op file and the CLI above | done |
| 5 | `baseline.py` / `report.py` — the guard | done |
| 6 | agent-facing entry point over the deterministic command | todo |

## Design decisions and the measurements behind them

All figures: B200, `CUDA_VISIBLE_DEVICES=0`, fp16, `space="heuristic"` (one
config, so this measures the kernel rather than the autotuner), five repeats of
the full measurement, "across-run spread" = largest one-sided deviation of the
per-run p50 from their median.

### Blocked sampling, not interleaved

Each provider's whole warmup+measure window runs before the next provider
starts, matching tritonbench (`utils/triton_op.py` reduces over providers per
input). Interleaving would cancel slow drift better for a ratio metric, but
matching tritonbench keeps our absolute milliseconds comparable to theirs.

### The measurement window is 3 s / 3 s, not estimate-derived

tritonbench's default sizes the window from a runtime estimate, which hands a
~1 ms kernel a 25 ms warmup — about 24 iterations, nowhere near thermal steady
state. Fixing the window at 3000/3000 (which is what the team already passes to
tritonbench for TLX perf) is the single largest denoise lever measured:

| mm shape (fp16) | unlocked GPU | + `denoise.sh` | + 3 s/3 s window |
|---|---|---|---|
| 8192×8192×8192 tlx | 13.94% | 5.47% | **1.67%** |
| 8192×8192×8192 torch | 16.64% | 7.34% | **0.44%** |
| 2048×2048×2048 tlx | 1.73% | 24.88% | 3.89% |
| 2048×2048×2048 torch | 4.77% | 0.16% | **0.15%** |
| 256×256×16384 tlx | 15.89% | 0.88% | 3.85% |
| 256×256×16384 torch | 0.21% | 0.00% | **0.00%** |

A 5% regression gate on the unlocked, short-window numbers would have been
pure noise. The noise floor is set at 2% on the strength of the right-hand
column.

### Small shapes are host-bound and cannot be gated

The two shapes still above 2% are not noisy machines — they are TLX's per-call
Python cost showing through. Median host-side time to *issue* one call, with no
synchronization:

| mm shape | `tlx.ops.mm` host | `torch.matmul` host | measured latency |
|---|---|---|---|
| 8192×8192×8192 | 43.1 µs (p95 60.7) | 9.0 µs | 1105 µs |
| 2048×2048×2048 | 62.5 µs (p95 76.0) | 8.9 µs | 54 µs |
| 256×256×16384 | 57.1 µs (p95 69.1) | 11.4 µs | 42 µs |

At 2048³ the whole measured latency is *less* than the host cost of issuing the
call: the GPU idles waiting for the launch, and what we would be gating is
`TensorDescriptor` construction and caching-allocator behaviour. Such cases get
`Status.HOST_BOUND` — reported with their numbers, never gated.

`HOST_BOUND_RATIO` is 1.5, not something larger, and the reasoning matters: the
host issues iteration N+1 while the GPU runs iteration N, so host cost does not
*add* to latency — it only starves the GPU once it exceeds kernel time. A first
attempt at 5.0 flagged `8192×8192×1024` (138 µs measured, 42 µs host, and torch
needs 128 µs for the same work) as unmeasurable, which is plainly wrong.

Two consequences worth carrying forward:

1. **The shared shape list will produce host-bound perf cases.** Correctness
   needs small shapes; perf cannot gate them. That is a reported status rather
   than a reason to split the list.
2. **Even at 8192³, TLX pays a ~4% host tax that torch does not** (43 µs on
   1105 µs vs 9 µs). The speedup ratio is therefore slightly biased against
   TLX. Fixing it means hoisting the per-call work out of `mm()`.

### `-lgc` does not lock the clock for compute-bound work on B200

`denoise.py` verifies the environment; it does not re-implement `denoise.sh`,
because duplicating privileged shell logic would give two definitions of
"denoised" and the copy would drift.

The first version of that verification checked "is the SM clock near maximum?"
and was wrong. Sampling the clock during a sustained 8192³ fp16 GEMM, with and
without `nvidia-smi -lgc 1965`:

| | median SM clock | event reasons |
|---|---|---|
| unmanaged | 832 MHz | `sw_power_cap` |
| `-lgc 1965` applied | 840 MHz | `sw_power_cap` |

`-lgc` reports success and changes nothing: at 750–850 W the card is
power-governed at ~840 MHz long before a 1965 MHz clock cap binds. A
clock-lock check would therefore have failed on every *correct* run. There is
a unit test asserting no such check exists.

What `denoise.sh` actually contributes for this workload is a fixed power cap,
persistence mode, and NUMA binding. NUMA binding is also the most reliable
signal that it wrapped the run at all, since it is unambiguous and
machine-independent. The rest of the stability comes from warming to steady
state.

So `check()` reports only unambiguous problems — a co-tenant process, missing
NUMA binding, persistence mode off, degrading throttle reasons, NVML missing —
and the operating point is *watched* rather than asserted.

### Watching the operating point

`stable()` samples SM clock and throttle reasons at 20 Hz through the whole
case via NVML through `ctypes` (~16 µs per sample; `nvidia-smi` forks a process
per reading and is far too expensive, and `pynvml` is not a dependency worth
taking). The trace goes into the artifact.

Its spread is `(p90 - p10) / median`, not `(max - min)`: the window necessarily
contains the ramp from the idle clock, so min/max reported ~25% on every run,
healthy or not. Deciles ignore the handful of ramp samples for the same reason
the latency path rejects outliers. A verified-clean run on an idle, denoised
B200 measures 6.5%.

`sw_power_cap` is deliberately excluded from the degrading set — a
compute-bound B200 reports it continuously — while `hw_slowdown`,
`hw_thermal_slowdown`, `sw_thermal_slowdown` and `hw_power_brake_slowdown` all
invalidate a window.

Two bugs this caught in its own right: the co-tenant check fired for real when
another user's 14 GB process appeared on GPU 1 mid-run (that trace's clock
dropped to 742 MHz), and device identity has to be resolved by UUID, since
torch indices are remapped by `CUDA_VISIBLE_DEVICES` while NVML and
`nvidia-smi` index physical devices.

### Headline is mean + CV; the gate is between-run reproducibility of the mean

Every metric column is TFLOP/s, higher better: `ref`/`tlx` at the mean latency,
and p50/p90/p99 at those latency percentiles. Because the percentiles are taken
on latency and then converted, the three descend — p99 is the *worst-case*
throughput. Percentiles are nearest-rank, so each is derived from a latency the
kernel actually produced rather than an interpolation between two it did not.

`CV%` is `sd/mean` over the within-run latency samples after IQR rejection.
Normalizing against the mean is what lets a 52 µs kernel and a 1095 µs one be
compared on stability at all; a bare standard deviation cannot.

Mean and CV together still cannot show a tail — two kernels can match on both
and diverge entirely at p99 — which is why the percentiles share the row.

The **gate** reads neither. It reads `rel_max_deviation` -- the between-run
deviation of the replicate means -- because that is the uncertainty on the
headline number, whereas CV is the width of one run.

### Why the gate is between-run, not within-run

These are very different quantities, and an earlier version conflated them.
Measured on a denoised B200 at the default window, `--space heuristic`, three
replicates:

| mm shape (fp16) | between-run reproducibility | within-run dispersion |
|---|---|---|
| 8192×8192×8192 | 0.2% | 5.6% |
| 8192×8192×1024 | 0.0% | 1.7% |
| 8192×8192×16384 | 0.1% | 2.4% |
| 8192×8192×8192, B col-major | 0.0% | 4.7% |

With a few thousand samples the median is far more stable than the distribution
is wide — the 5.6% width is the power-governed clock wandering, and it
correlates with the sampled clock trace. Gating on width rejected cases whose
reported number was solid. `Stat.spread` is therefore the replicate-to-replicate
figure and drives `NOISE_FLOOR = 2%`; `Stat.within_spread` keeps the width as a
diagnostic for whether the *machine* was steady.

The cost is three replicates per provider per case, which is the only way to
observe reproducibility at all.

### Compile time: `space="full"` was over the cap by 2×, so the default changed

`tlx.ops.mm(a, b)` used to default to `space="full"`. On a cold Triton cache:

| shape | `t_cold` | compilations | configs benchmarked |
|---|---|---|---|
| 1024³ | 284.6 s | 350 | 348 |
| 8192³ | 221.3 s | 252 | 300 |

Against a 120 s cap, so a first call was 2× over. **`tlx.ops.mm` now defaults
to `space="heuristic"`** — one analytically chosen config. Measured at 1024³ on
the same box:

| default | `t_cold` | compilations | configs |
|---|---|---|---|
| `space="heuristic"` (new default) | **0.69 s** | 2 | — |
| `space="full"` (explicit opt-in) | 283.26 s | 349 | 348 |

410× faster and comfortably inside the cap. The trade is real and is the
caller's to make: tuned configs are worth up to ~4× on small shapes
(1024³: 40.6 µs heuristic vs 10.4 µs full), so `space="full"` remains available.

`flash_attn`, `hstu_attn` and `kimi_delta_attention` still default to `"full"`.
Their only other space is `"smoke"`, which selects for lowering-path coverage
rather than speed, so defaulting to it would quietly ship a bad config. Each
needs its own `heuristic_config` before it can follow `mm`.

The cap is absolute rather than relative to a baseline because cuBLAS has no
compile step to be worse than, and because a relative gate ratchets — three 15%
regressions pass individually and double the wait.

`n_compiles` and `n_configs` come from `triton.knobs.compilation.listener` and
`triton.knobs.autotuning.listener`, so they are op-agnostic and require no
access to a kernel module's autotuner. They are what make a breach actionable:
a slow first call is almost always "pruning left too many configs", not "the
compiler got slower".

This is also why `--measure` splits `latency` from `compile`. At ~4 minutes per
case the two together would be a two-hour run.

### TODO — `tlx.ops.mm` allocates a workspace on every launch

`sm100.py::matmul_tma_set_block_size_hook` is the autotuner config `pre_hook`,
so it runs on *every* launch, and when `SPLIT_K > 1` it does a fresh
`torch.empty((SPLIT_K * M, N))`. At 2048³ (heuristic picks `SPLIT_K=4`) that is
a 32 MB allocation per call; at 256×256×16384, also `SPLIT_K=4`, it is 512 KB
and the shape is correspondingly stable. Allocation size tracks the instability,
which is what identifies the mechanism.

This is an op inefficiency, not a harness problem — the workspace could be
cached per `(shape, config)` like the L2 flush buffer already is. Not filed;
out of scope for this suite, recorded here because the harness is what found it.

### FIXED — split-K remainder tile

`[1000, 1000, 1024]` and `[64, 4096, 4096]` were disabled for a split-K bug and
are now **re-enabled**: fixed upstream by #3401 and verified on the rebase, both
0.0% wrong, as is the smaller repro `[4160, 512, 512]` which measured 4.0% wrong
before. They stay in the list as the regression test for that fix. L1 is 28
tests in 19.6 s.

### TODO — `NUM_CTAS=2` is wrong whenever the grid has more than one N-tile

**Still reproduces after #3401** — a different bug, with `SPLIT_K=1`. Found by
merging the perf shapes into the shared list: `1000000×512×512` failed L1
immediately. 49.9% of output elements are wrong, almost exactly one CTA of each
pair. **Pre-existing, not a promotion artifact**: the identical 49.9%
reproduces through the original tutorial entry point
`blackwell_gemm_ws.matmul(a, b, config=get_heuristic_config(...))`. Status is
**bypassed, not fixed** — the shape is commented out so a wrong-answer path
cannot be benchmarked. Full measurement table and the control row are in
`ops/kernels/mm/_shapes.py`; the shape is commented out there with a TODO.

Two things make this worse than an ordinary correctness bug. The heuristic
picks `NUM_CTAS=2` for ordinary large-M shapes, so it is reachable in
production. And autotuning cannot rescue it, because the autotuner ranks
configs by speed without ever checking their results — a wrong-answer config
can win.

It is also the argument for the shared shape list. This shape existed only in
tritonbench's perf set; putting perf and correctness on one list is what ran it
through `assert_close`.

### TODO — `space="full"` leaks GPU memory during autotuning

The first attempt at a full-space baseline OOMed after 7 shapes with **178 GiB
allocated** (not merely cached — `empty_cache()` between cases does not help,
the memory is live). Same `pre_hook` as above: it allocates
`torch.empty((SPLIT_K * M, N))` per config, and at `SPLIT_K=24, 8192×8192` that
is 3.2 GB for a *single* config out of ~348.

Consequence beyond this suite: `tlx.ops.mm` cannot be called with its default
`space="full"` across several large shapes in one process.

The same run confirmed the host-overhead diagnosis outright. At full space the
tuned configs choose `SPLIT_K=1`, no workspace is allocated per launch, and
host cost per call falls from ~56 µs to ~28 µs — exactly the predicted effect.
Tuned small shapes are also ~4× faster than the heuristic config picks
(1024³: 40.6 µs → 10.4 µs), so the heuristic is leaving a lot on the table.

### The committed baseline is `space="heuristic"`

`baselines/mm.sm100.json` records 8 cases — the four compute-bound shapes in
fp16 and bf16 — at 0.92–1.06× cuBLAS and ~1000–1073 TFLOP/s. The 16 host-bound
cases are refused rather than recorded.

This now matches `tlx.ops.mm`'s own default, so the baseline describes the path
users actually take. `load()` still refuses to compare a baseline recorded at
one space against a run at another: mm 1024³ is 40.6 µs at heuristic and 10.4 µs
at full, so a cross-space comparison would report a 4× "regression" that is only
a different search space.

### Known limitation — the clock trace spans the whole run

`stable()` wraps the entire run rather than each case, so its trace includes
the idle gaps between cases (allocation, compilation, host-overhead sampling).
Over an 11-minute baseline run that reads as `spread 0.54, stable: false` even
on a clean machine, which is not wrong but is not useful either. The per-case
signal that does gate is the latency reproducibility. Making the trace per-case
is straightforward and not yet done.
