# bitequiv evaluation framework

The team's standard ruler for a **PTX bitwise-equivalence checker**: how well does
it decide which autotuner configs of a reduction kernel produce *identical output
bits*? One driver, three stages, a pluggable checker, a plain-text result table.

The checker under test is the static PTX descriptor
(`bitequiv.ptx_reduction.ptx_reduction_descriptor`, no GPU launch) by default. The
ground truth it is measured against is an **empirical fuzzer**: launch a config on
many random inputs and compare output bits. The framework runs on a CUDA GPU (it
compiles and launches kernels); the checker itself is pure Python.

> This framework **evaluates** a checker; it does not change one. Run against
> today's repo checker it shows the baseline's heavy over-splitting — that is the
> honest baseline a future checker improvement is measured against.

## Files

| file | what it is |
|------|------------|
| `evaluate.py` | the 3-stage CLI driver (orchestration only); its docstring is the design summary |
| `eval_kernels.py` | every test kernel + its `KernelSpec` (the registry) |
| `equivalence_fuzzer.py` | standalone empirical oracle + partition / soundness math (no triton / bitequiv imports) |
| `result.txt` | generated result table (gitignored — regenerate on demand) |

## The three stages

- **Stage 1 — support.** For each kernel, probe the checker on a reference config:
  `SUPPORTED` / `LIMITED` / `UNSUPPORTED`. Today a thin heuristic on the descriptor
  plus any limitation the kernel author declared. `kernel_support()` is the
  documented extension point for when a checker can declare real support itself.
- **Stage 2 — precision (soundness + partition).** Build the config space, compile
  each config, let the checker group them, and independently fuzz every config.
  Report per kernel: checker classes (count + largest = recovered search space),
  empirical classes (count + largest = the recovery **ceiling**), **over-merges**
  (configs the checker merged but the fuzzer separated — the soundness violation,
  **must be 0**), and whether the checker partition **refines** the empirical one.
- **Stage 3 — performance (opt-in).** Benchmark every kernel across its config
  space (same coverage as Stage 2, per `--config-effort`), find the global-fastest
  **ceiling** (a normal, equivalence-blind autotuner), then inside one
  checker-certified set (after verifying every member is byte-identical) report
  fastest-vs-slowest (tuning freedom) and best-vs-ceiling (the cost of demanding
  identical bits).

## Effort knobs

| flag | values |
|------|--------|
| `--config-effort` | `light` (~10 configs, quick gate) · `heavy` (full sweep, ~1000–3000) |
| `--fuzzer-effort` | `fast` (10 seeds) · `convincing` (1000 seeds) |

A fuzzer can only ever *refute* equivalence, never prove it — more seeds means
stronger evidence of soundness, never certainty.

## Kernels

`sum_dim1_simple` (single-tile column sum), `sum_dim1_persistent` (looped column
sum), `welford` (mean/variance, 2 outputs), `sum` (row sum), `dot`
(row dot product), `cond_reduce` (column sum behind a data-dependent branch — the
control-flow example for Stage 1). All six are benchmarked in Stage 3.

## Run

```bash
# from the triton repo root, with triton importable (PYTHONPATH=$PWD/python or `pip install -e .`)
python -m bitequiv.evaluation.evaluate --stages 1,2 --config-effort light --fuzzer-effort fast   # quick smoke
python -m bitequiv.evaluation.evaluate --stages 1,2,3 --config-effort heavy --fuzzer-effort convincing  # full run
python -m bitequiv.evaluation.evaluate --kernels sum_dim1_persistent,welford --stages 2          # pick kernels
python -m bitequiv.evaluation.evaluate --checker my.module:my_descriptor                          # another checker
```

Flags: `--kernels all|<comma list>`, `--stages 1,2[,3]`, `--config-effort`,
`--fuzzer-effort`, `--checker module:function`, `--artifact ptx|ttgir`,
`--allow-unsound`, `--out <path>`.

The GPU-gated pytest `bitequiv/tests/test_ptx_kernel_suite.py` runs Stage 2
(light / fast) as a subprocess and asserts **over-merges == 0**. If a kernel
launch ever hangs, kill it with `third_party/tlx/killgpu.sh`.

## Running against the TTGIR checker

The framework is checker- and artifact-pluggable, so it also measures the **TTGIR**
reduction-order checker (`bitequiv.ttgir_reduction.ttgir_reduction_descriptor`, the
MLIR-native sibling of the PTX checker). Point `--checker` at it and feed it TTGIR
instead of PTX with `--artifact ttgir`:

```bash
python -m bitequiv.evaluation.evaluate \
  --checker bitequiv.ttgir_reduction:ttgir_reduction_descriptor \
  --artifact ttgir --allow-unsound --stages 1,2
```

The TTGIR checker is **expectedly not sound**: TTGIR fixes the reduction
*association order* but is provably blind to **FMA contraction** (whether a `mul`
feeding a reduction fuses into a single rounded `fma`), which is decided *below*
TTGIR and gated by `enable_fp_fusion`. So on the mul-fed kernels (`dot`, `welford`)
it over-merges configs whose bits differ — which is precisely the gap the PTX
checker closes. That is why we pass `--allow-unsound`: the framework still emits the
full report (with the honest over-merge count) and exits 0 instead of failing the
soundness gate. The over-merges it reports are the documented finding — compare the
TTGIR partition (this run) against the PTX one (the default run) to see exactly the
FMA-fusion pairs PTX separates and TTGIR cannot.

The GPU-gated pytest `bitequiv/tests/test_ttgir_kernel_suite.py` drives this and
asserts the report is produced (it does **not** assert over-merges == 0).

## What the baseline run shows

Against today's repo checker the soundness gate holds (over-merges == 0) but the
checker over-splits: the largest checker set is roughly `num_stages`-only, well
below the empirical ceiling. That gap is the recovery opportunity for a future
checker-improvement diff — which this same framework will measure.
