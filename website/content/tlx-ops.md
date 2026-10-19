Production-ready kernels promoted from the TLX tutorials into the FBTriton op library.

## What tlx.ops is

- Promote performant and prod-impactful ops from TLX tutorials into the FBTriton op library.
- Catalog existing OSS TLX kernels and pick the best kernel per (arch, op).
- Enforce the highest quality bar for both code change and CI.
- Demote `tutorials/` to a playground scratch with no CI coverage.

## Interface

```python
from triton.tlx.ops import mm as tlx_mm

out = tlx_mm(a, b)
```

`tlx.ops` infers the architecture from the input tensors and dispatches to the
matching registered implementation.

## Structure

```
third_party/tlx/
    ops/                    -> (symlink) python/triton/tlx/ops
        kernels/
            mm/          sm90.py  sm100.py  gfx942.py  gfx950.py
            addmm/       gfx942.py
            bmm/         gfx942.py
            ...
```

## Current availability

Implemented today:

```
mm/sm100.py
mm_bias_grad/sm100.py
mm_mxfp8/sm100.py
mm_mxfp6/sm100.py
flash_attn/sm100.py
flash_attn/gfx950.py
flash_attn_mxfp8/sm100.py
flash_attn_mxfp8/gfx950.py
flash_attn_varlen/gfx950.py
hstu_attn/sm100.py
hstu_cross_attn/gfx950.py
kda/sm100.py
```

Current kernel selections for the rest of the catalog:

```
kernels/
    mm/                   sm90.py  sm100.py  gfx942.py  gfx950.py
    mm_bias_grad/         sm100.py
    mm_mxfp8/             sm100.py
    mm_mxfp6/             sm100.py
    addmm/                gfx942.py
    bmm/                  gfx942.py
    flash_attn/           sm90.py  sm100.py  gfx950.py  (fwd + bwd)
    flash_attn_mxfp8/     sm100.py  gfx950.py           (gfx950 forward only)
    flash_attn_varlen/    gfx950.py                     (forward only)
    hstu_attn/            sm100.py  gfx942.py
                          _util.py  _stubs.py  _reference.py
    hstu_cross_attn/      gfx950.py                     (fwd + bwd, split-KV)
    kda/                  sm100.py
    gdpa/                 sm100.py  gfx950.py
    grouped_gemm/         sm100.py  gfx950.py
    grouped_gemm_mxfp8/   sm100.py  gfx950.py
    addmm_glu/            gfx950.py
    paged_decode/         gfx950.py
    ikbo_fa/              gfx950.py
    ikbo_lce/             gfx950.py
    multi_cta_layernorm/  sm100.py
    cross_attention/      sm100.py
    bmm_shared_a/         gfx950.py
```

## Linear weight and bias gradients

```python
from triton.tlx.ops import mm_bias_grad

d_weight, d_bias = mm_bias_grad(dout, input_act, space="full")
```

`dout` is contiguous `[batch, out_features]` and `input_act` is contiguous
`[batch, in_features]`, with matching BF16 dtype on the same Blackwell
GPU. Feature dimensions are positive multiples of eight; batch is positive
and may have a partial K tile. The result is `dout.T @ input_act` and
`dout.sum(0)`, in the input dtype. `out=(d_weight, d_bias)` supplies contiguous,
non-overlapping outputs. These explicit first-order gradients accept saved
activations requiring gradients and do not construct a higher-order graph.

The operation follows the same architecture layout as `mm`:

```
mm_bias_grad/
    __init__.py
    _shapes.py
    sm100.py
```

`sm100.py` contains the private JIT paths, configuration generation and pruning,
lazy autotuning and the launch entry point. The catalog registers one public
implementation for `(mm_bias_grad, sm100)`. `_shapes.py` uses the shared
`FocusRegistry` for `sm100_1` and defines synthetic correctness cases.

The kernel horizontally fuses GEMM and bias reduction through shared `dout`
tiles. Four warp tasks run the epilogue, MMA, TMA and bias reduction. The bias
task is replicated per MMA group; bias and MMA release each shared output-gradient
tile through a two-arrival completion barrier. The large-K heuristic uses two
CTAs, two MMA groups and eight default warps. BF16 split partials for weight and
bias are finalized together. The default
`space="heuristic"` selects one configuration; `space="full"` measures the
complete operation over tile sizes, buffer depths and split-K choices
spanning one and two occupancy waves. Below batch 65536, autotuning uses
CUDA graph timing, including the merged finalization. Larger batches use
ordinary-launch autotuning with cache flushing.

Correctness tests live in `python/test/unit/tlx_ops/test_mm_bias_grad_sm100.py`.
They cover BF16 production shapes, partial K and feature tiles,
split-K, non-power-of-two buffer depths, bias ownership and CUDA graph replay.
Run them from `fbsource/fbcode`:

```bash
CUDA_VISIBLE_DEVICES=1 buck2 run @mode/opt -m ovr_config//triton:beta \
  -c fbcode.nvcc_arch=b200a -m ovr_config//third-party/cuda/constraints:12.8 \
  --agent-context direct_call=true --agent-context reason="Check fused weight and bias gradients" \
  fbsource//third-party/triton/beta/triton:py_tlx_mm_bias_grad_test -- -s --tb=short -v
```

### Benchmark against unfused PT2

The TritonBench operator is `mm_bias_grad`, with `pt2_unfused_best` and `tlx`
providers. Both return the complete weight and bias pair. PT2 selects among
28 separately compiled serial and concurrent candidates before TritonBench
times the providers with CUDA graphs. Inputs are seeded random BF16 tensors.

From `fbsource/fbcode`:

```bash
CUDA_VISIBLE_DEVICES=1 ../third-party/triton/beta/triton/third_party/tlx/denoise.sh \
  buck2 run @mode/opt -m ovr_config//triton:beta \
  -c fbcode.nvcc_arch=b200a -m ovr_config//third-party/cuda/constraints:12.8 \
  -c fbcode.enable_gpu_sections=true \
  --agent-context direct_call=true --agent-context reason="Benchmark tlx.ops linear gradients" \
  fbcode//pytorch/tritonbench:run_tlx -- \
  --op mm_bias_grad --only pt2_unfused_best,tlx --metrics latency,speedup
```

For all candidate timings and a complete JSON report, the standalone entry
uses `tritonbench/operators/fb/mm_bias_grad/benchmark.py`.

From `fbsource/fbcode`, on an idle B200:

```bash
CUDA_VISIBLE_DEVICES=1 ../third-party/triton/beta/triton/third_party/tlx/denoise.sh \
  buck2 run @mode/opt -m ovr_config//triton:beta \
  -c fbcode.nvcc_arch=b200a -m ovr_config//third-party/cuda/constraints:12.8 \
  --agent-context direct_call=true --agent-context reason="Benchmark fused weight and bias gradients" \
  fbsource//third-party/triton/beta/triton:py_tlx_mm_bias_grad_perf -- \
  --space full --json /tmp/mm_bias_grad_benchmark.json
```

The default focus suite is `(1024, 384, 384)`, `(4096, 512, 384)`,
`(8192, 256, 256)` and `(294912, 512, 384)`, in `(batch, out, in)` order.
Use `--shape BATCH OUT IN` to select cases.

The benchmark separately compiles the PT2 GEMM and sum, making horizontal
fusion across them impossible. It uses exhaustive max-autotune GEMM searches
over ATEN, Triton and the combined ATEN/Triton/CUTLASS set, and compares
cuBLAS/cuBLASLt preferences and reduction coordinate descent enabled and
disabled. Each pair is measured both serially and with GEMM and reduction on
separate CUDA streams joined before completion. Every candidate is checked
against an FP32 reference. The fastest median end-to-end PT2 pair is selected
per shape, with all provider times, compile failures and autotune warnings
recorded in JSON.
The selected GEMM backend is recorded separately from the requested search
set, including any TorchTLX templates enabled in the current PT2 build.
The default search measures 28 unfused PT2 providers per shape. To remeasure
the winning serial and concurrent settings from a completed search, add
`--pt2-report /tmp/mm_bias_grad_benchmark.json` and write the new measurements
to a different `--json` path. The saved report must contain the complete
search and match the GPU model, SM count, software versions, dtype and fill.
The winners are compiled, checked and timed again alongside TLX.

Both implementations capture 64 complete output pairs per CUDA graph and use
CUDA events to time repeated graph launches, with the same inputs and output
dtype. Timing includes both outputs and TLX split-K finalization. Compilation and
autotuning precede timing. Three interleaved timing rounds follow a one-second
warmup. Timing inputs default to zeros for fixed-clock measurements; use
`--fill uniform` for populated inputs. Correctness always uses seeded random
inputs. Replaying the same input tensors is a warm-cache measurement; the
large production case's input footprint exceeds B200 L2.

### Configurations and split-K layout

`CONFIGS`, `SMOKE_CONFIGS` and `heuristic_config` define the full, smoke and
heuristic spaces in `sm100.py`. Autotune state initializes lazily per shape
and device, and the SM count follows the input device. Configuration pruning
accounts for SMEM, TMEM, paired-CTA tile ownership and empty K splits.
Split-K configurations require whole M tiles because workspace rows use M
as their split stride. Feature tails use a single split. Worker tasks use a
168-register budget, and autotuning includes the merged weight/bias finalization.

### B200 measurements

BF16 results on B200 with 148 SMs, Torch `2.16.0a0+fb`, Triton `3.9.0+fb`
and CUDA `12.8.0`, using the graph timing protocol above:

| Shape (batch, out, in) | TLX fused (us) | Best unfused PT2 (us) | Speedup | Selected PT2 GEMM / execution |
|---|---:|---:|---:|---|
| (1024, 384, 384) | 6.69 | 4.48 | 0.67x | ATEN / concurrent |
| (4096, 512, 384) | 8.00 | 8.36 | 1.04x | ATEN / concurrent |
| (8192, 256, 256) | 8.89 | 8.62 | 0.97x | TRITON / concurrent |
| (294912, 512, 384) | 122.58 | 138.45 | 1.13x | TRITON / serial |

The small cases remeasure the serial and concurrent winners selected by the
complete 28-provider search. The large case uses that complete-search round,
which produced the fastest observed PT2 baseline. Every row compares the
complete TLX and PT2 output pair within the same timing round. The 1024 and
8192 cases remain slower than the best current PT2 pair.

A subsequent large-case matched run measured TLX at 123.92 us and PT2 at
145.76 us. A graph-autotuning trial selected a slower 143.02 us TLX
configuration for this shape; the large-K policy uses
ordinary-launch autotuning. All reports retain the selected configurations,
individual timing samples and compiler diagnostics.

## MXFP8 GEMM

```python
out = triton.tlx.ops.mm_mxfp8(a, a_scale, b, b_scale, out=None, sf_layout="natural", space="full")
```

Computes `a @ b.T` from contiguous E4M3 `a` `[M, K]` and `b` `[N, K]` with one
E8M0 scale per 32 K values, returning contiguous BF16 `[M, N]`. `M`, `N` and
`K` must be multiples of 128. This is a forward-only SM100 operation promoted
from `tutorials/blackwell_gemm_ws_mxfp8.py`.

- `sf_layout="natural"`: `a_scale` is `[M, K // 32]` and `b_scale` is
  `[N, K // 32]`; they are packed into the blocked layout on every call.
- `sf_layout="cublas_blocked"`: scales are already in the cuBLAS 128x4 atom
  layout, e.g. torchao's `MXTensor.to_mx(..., is_swizzled_scales=True).scale`,
  and must contain exactly `M * K // 32` and `N * K // 32` bytes.
- `space="full"` (default) autotunes over 1-/2-CTA, split-K, overlapped-
  accumulator and pipeline-depth configs; the first call per shape compiles and
  benchmarks the pruned space. When all of B fits in shared memory next to at
  least four A stages (small N and K), it also tries a B-resident family: B is
  loaded once per CTA, each tile computes a whole 128 x N row so A is read from
  DRAM once, and the epilogue uses async TMA stores. `space="heuristic"`
  launches one shape-picked config without autotuning.
- A supplied `out` must be contiguous, non-overlapping BF16 `[M, N]` on
  `a.device` and is returned by identity.

## MXFP6 GEMM

```python
out = triton.tlx.ops.mm_mxfp6(a, a_scale, b, b_scale, out=None, sf_layout="natural", space="full")
```

Computes `a @ b.T` from contiguous dense-packed E2M3 `a` `[M, K]` and E3M2 `b`
`[N, K]` (uint8, 4 codes per 3 bytes LSB-first, K padded to the unpacked
width) with one E8M0 scale per 32 K values, returning contiguous BF16 `[M, N]`.
`M`, `N` and `K` must be multiples of 128. This is a forward-only SM100
operation mirrored from `kernels/mm_mxfp8/sm100.py` with the `mxf8f6f4` MMA kind.

- `sf_layout="natural"`: `a_scale` is `[M, K // 32]` and `b_scale` is
  `[N, K // 32]`; they are packed into the blocked layout on every call.
- `sf_layout="cublas_blocked"`: scales are already in the cuBLAS 128x4 atom
  layout, e.g. torchao's `MXTensor.to_mx(..., is_swizzled_scales=True).scale`,
  and must contain exactly `M * K // 32` and `N * K // 32` bytes.
- `space="full"` (default) autotunes over 1-/2-CTA, split-K, overlapped-
  accumulator and pipeline-depth configs; the first call per shape compiles and
  benchmarks the pruned space. When all of B fits in shared memory next to at
  least four A stages (small N and K), it also tries a B-resident family: B is
  loaded once per CTA, each tile computes a whole 128 x N row so A is read from
  DRAM once, and the epilogue uses async TMA stores. `space="heuristic"`
  launches one shape-picked config without autotuning.
- A supplied `out` must be contiguous, non-overlapping BF16 `[M, N]` on
  `a.device` and is returned by identity.
- Quantize with torchao (`elem_dtype="fp6_e2m3"` / `"fp6_e3m2"`), shift its
  unpacked codes down (`qdata >> 2`), and dense-pack; torchao emits unpacked
  codes, not the dense triples the kernel consumes.

## MXFP8 grouped GEMM

```python
out = triton.tlx.ops.grouped_gemm_mxfp8(
    x,
    x_scale,
    w,
    w_scale,
    split_sizes,
    out=None,
    num_sms=None,
    sf_layout="natural",
)
```

The exact public signature is
`(x, x_scale, w, w_scale, split_sizes, *, out=None, num_sms=None,
sf_layout='natural')`. This is a forward-only operation for SM100 (compute
capability 10.0) and gfx950 (MI350/MI355X): E4M3 inputs and E8M0 scale factors
produce a contiguous BF16 result. It has no backward implementation or fallback
on other architectures.

### Data and group layout

- `x` is contiguous E4M3 `[GM, K]` data, packed in expert order.
- `w` is contiguous E4M3 `[G, N, K]` or packed `[G * N, K]` data. For rank 2,
  `G` comes from `split_sizes.shape[0]`, and `w.shape[0]` must be divisible by
  `G`.
- `split_sizes` is contiguous int32 `[G]` on the same CUDA device. Values may be
  zero, must be nonnegative, every exclusive prefix must be divisible by 128,
  and the final sum must equal `GM`. Thus zero-sized experts are supported in
  any position, but a nonzero expert may start only at a 128-row boundary.
- `GM` and `K` must be positive and divisible by 128; `G` and `N` must be
  positive. TMA base pointers and row byte strides must be 16-byte aligned,
  which additionally requires `N` to be divisible by 8 for the BF16 output.
- Every input must be contiguous and on `x.device`.

The split values remain on device so dispatch does not introduce a host read or
synchronization. The kernel checks nonnegativity, aligned prefixes, and the final
sum. Violating one of those requirements triggers a device trap; callers that
need to observe the error immediately must synchronize the CUDA stream.

### Scale-factor layouts

Both scale tensors are contiguous E8M0. `sf_layout` accepts exactly these values:

- `"natural"` (default): `x_scale` has exact shape `[GM, K // 32]`.
  With rank-3 `w`, `w_scale` is `[G, N, K // 32]`; with rank-2 `w`, it is
  `[G * N, K // 32]`.
- `"cublas_blocked"`: both scales are rank-2 opaque byte tensors in the cuBLAS
  128-row by 4-scale-column atom layout. Let
  `SK = 4 * ceil((K // 32) / 4)`. `x_scale` must contain exactly
  `ceil(GM / 128) * 128 * SK` bytes, and `w_scale` exactly
  `G * ceil(N / 128) * 128 * SK` bytes. Tensor shape is otherwise opaque; byte
  order, not a semantic row/column interpretation, defines this layout.

### Output and scheduling

The result shape is `[GM, N]`. If `out` is supplied, it must be a contiguous,
non-overlapping BF16 tensor of that exact shape on `x.device`, and the operation
returns the same tensor object. `num_sms=None` launches one persistent CTA per SM
on the input device; an explicit `num_sms` must be an integer from 1 through the
device SM count.

The SM100 implementation uses fixed, shape-selected 1CTA and 2CTA configurations
with a dynamic atomic tile scheduler; it does not autotune. Large, sufficiently
saturated N/K shapes use a cooperative 2CTA cluster, while other shapes retain
the 1CTA path. Each launch owns a counter initialized to zero; that reset is part
of CUDA graph capture and is replayed before the kernel, so repeated graph
replays traverse all tiles.

The gfx950 implementation launches one persistent workgroup per CU (`num_sms`
counts CUs) with an XCD-aware static tile stride, so it needs no counter and is
graph-safe. When 256x256 tiles fill the device it uses the 2x2-quadrant
`warp_pipeline_stage` engine of the FP16 gfx950 grouped GEMM, with each
half-tile's scales copied direct-to-LDS alongside its data. Smaller problems
use a compiler-pipelined 128-row tile. It does not autotune.
