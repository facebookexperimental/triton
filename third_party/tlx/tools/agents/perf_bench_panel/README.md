# Perf-bench panel

This panel inherits [the general panel](../general_panel/README.md). It owns reproducible
performance measurement and regression gating.

## 1. Measurement discipline

- Run performance tests only when explicitly requested.
- Validate correctness before measuring performance.
- Serialize GPU measurement: one benchmark at a time, on a pinned idle GPU,
  under `third_party/tlx/denoise.sh`. Concurrent load perturbs clocks, power,
  and thermals even when it uses another GPU on the same node.
- Record the device, environment, workload, measurement method, dispersion, and
  artifact paths. A number without this context is not comparable evidence.
- Preserve the benchmark process's exit status through every wrapper.

## 2. Promotion scope

A ticket may name one shape, but `tlx.ops` promotion closes on the full relevant
shape set in `_shapes.py`. A worker first measures the focused finding, then
re-runs the full shape set before publication. The manager rejects a local win
that is a global regression.

The profiler supplies raw evidence to the R&D panel. The perf-bench panel does
not invent mechanisms from counters or traces and does not override the CLI's
promotion verdict.
