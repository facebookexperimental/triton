# TLX op perf suite

Performance and compile-time guardrails for the blessed `tlx.ops` catalog.
Correctness lives in `python/test/unit/tlx_ops/`; this suite assumes it passes.

Depends on `torch` and `triton` only. No dependency on `tritonbench`

## Running

One shot command with extreme simplicity

python python/test/tlx_benchmark/bench_{op}.py

```
options:
  -h, --help            show this help message and exit
  --device DEVICE       GPU index, or 'auto' (default) for the least-used one
  --space {heuristic,full,smoke}
                        autotune search space; 'heuristic' is what tlx.ops.mm uses by default, and measuring anything
                        else measures a path users do not take
  --head N              only the first N cases, for a quick look; never writes the baseline
  --json JSON           machine-readable artifact (default /tmp/tlx_benchmark/mm.sm100.json)
```

- built in (default on) denoise (freq-lock)
- built in (default on) GPU selection (least used)
- built in (default on) kernel selection by gpu arch

## Metric report

1. Input info (varying between ops): e.g. "3000000x256x1024 A:row B:col bfloat16" for mm
2. Core metrics: ref (TFLOP/s), TLX (TFLOP/s), speedup, compile time
3. Additional stats: samples, CV%, p50, p95, p99 (all based on TFLOP/s)
4. status: ok/pip/noisy/error/...

error: break, accuracy error
pip: speedup < 0.9 or compile time > 2 min
noisy: CV% > 3%
ok: everything else

TFLOP/s is the unit the harness **measures in**, not a conversion applied to the
report: each timed iteration is turned into TFLOP/s before outlier rejection, so
CV% and the percentiles describe the throughput distribution. Two consequences:

- `speedup` = TLX TFLOP/s / ref TFLOP/s. Above 1 still means TLX is faster.
- The percentiles are **literal**, so they ascend: `p99` is the *best* case,
  beaten by 1% of iterations. The worst case is `min`, in the JSON artifact.
  This is the opposite of the latency reading.

Latencies are not reported. Recover one as `flop_count / TFLOP/s`; both are in
the artifact. `tlx_host_us` stays in microseconds — host-side launch cost is not
device work, and expressing it as throughput would claim the GPU did those FLOPs
during it.

## Tests

Two files, and the split is unit-vs-GPU rather than one-per-module:

- `test_harness.py` — every `_harness` unit test. No GPU, runs in ~3s.
- `test_ops_perf.py` — the pytest front end over each `bench_<op>.py`. Launches
  real kernels and takes minutes; this is the one CI's junitxml comes from.

So `pytest test_harness.py` while iterating on the harness, and don't run
`pytest .` unless you meant to start a benchmark.

## shapes

1. Synthetic (general): used by L1 (correctness) and L2 (performance)
2. Focus shapes (vary between arch): used by L2 only. Need to match the GPU arch. e.g. mm shapes sm100 and mm shapes gfx942
