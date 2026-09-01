# TLX op perf suite

Performance and compile-time guardrails for the blessed `tlx.ops` catalog.
Correctness lives in `python/test/unit/tlx_ops/`; this suite assumes it passes.

Depends on `torch` and `triton` only. Where behaviour is taken from tritonbench
it is **ported, not imported** — each port names its source in a comment so the
drift is visible.

## Running

Always under `denoise.sh`, which locks clocks and power and binds to the
GPU-local NUMA node. Numbers taken without it are not comparable to anything.

```bash
CUDA_VISIBLE_DEVICES=0 third_party/tlx/denoise.sh \
    python -m pytest python/test/tlx_benchmark/bench_mm.py
```

The harness's own unit tests need no GPU:

```bash
python -m pytest python/test/tlx_benchmark/test_harness.py
```

## Status

| Phase | Content | State |
|---|---|---|
| 0 | contract + shared shape list | done |
| 1 | `measure.py` — latency, outlier rejection, host-overhead detection | done |
| 2 | `denoise.py` — environment verification, operating-point trace, env capture | done |
| 3 | `compile.py` — `t_cold`, absolute 120 s cap | todo |
| 4 | `bench_mm.py` | todo |
| 5 | `baseline.py` / `report.py` — the guard | todo |
| 6 | CI wiring, opt-in per PR | todo |

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
`Status.HOST_BOUND` — reported with their numbers, never gated. See
`HOST_BOUND_RATIO` in `_harness/measure.py`.

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
