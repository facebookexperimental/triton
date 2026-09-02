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
  --head N              only the first N cases, for a quick look
  --synthetic           run the correctness shapes instead of this arch's focus list; they are
                        mostly too small to time, so this is for looking, not for gating
  --json JSON           machine-readable artifact (default /tmp/tlx_benchmark/mm.<arch>.json)
```

- built in (default on) denoise (freq-lock)
- built in (default on) GPU selection (least used)
- built in (default on) kernel selection by gpu arch

## Metric report

1. Input info (varying between ops), e.g. for mm:
   `((), {'dtype': 'bf16', 'strides': '[[32768, 1], [12800, 1]]', 'M': '2304', 'N': '12800', 'K': '32768'})`
2. Core metrics: ref (TFLOP/s), TLX (TFLOP/s), speedup, compile time
3. Additional stats: samples, CV%, p50, p95, p99 (all based on TFLOP/s)
4. status: ok/pip/noisy/error/...

error: break, accuracy error
pip: speedup < 0.9 or compile time > 2 min
noisy: CV% > 3%
ok: everything else

Stats are computed on per-iteration TFLOP/s, not converted from latency. So the
percentiles ascend: p99 is the best case, `min` (artifact only) the worst.

speedup = TLX / ref. >1 means TLX is faster.

Latency is not reported: it is `flop_count / TFLOP/s`, both in the artifact.
`tlx_host_us` stays in microseconds — host launch cost is not device work.

## Tests

- `test_harness.py` — `_harness` unit tests. No GPU, ~3s.
- `test_ops_perf.py` — pytest front end over each `bench_<op>.py`. Real kernels,
  minutes. CI's junitxml comes from here.

`pytest .` runs both, so it starts a benchmark.

## shapes

1. Synthetic (general): L1 only.
2. Focus shapes (arch specific): L2. Need to match the GPU arch. e.g. mm shapes sm100 and mm shapes gfx942

`--synthetic` runs list 1 under L2 instead. The focus list may be empty.

Each entry carries its own strides and dtype, so there is no dtype
cross-product. Strides, not a row/col flag: a leading stride wider than the row
is a padded slice, and 0 is a broadcast.
