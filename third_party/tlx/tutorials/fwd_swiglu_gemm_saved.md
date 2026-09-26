# Saved-projection GEMM with SwiGLU epilogue

This OSS TLX kernel reproduces the forward contract from `T289757870`:

```text
projection = BF16(FP32_MMA(a[5120,4096], b[16384,4096].T))
gate, up = projection.chunk(2, dim=1)
output = BF16(FP32(gate) * sigmoid(FP32(gate)) * FP32(up))
return projection, output
```

This is an epilogue fusion. The packed BF16 projection remains materialized
because backward consumes it; only the split, sigmoid, multiplies, and compact
output store are fused into the GEMM launch. GEMM accumulation remains FP32,
and the kernel uses exact `tl.sigmoid` rather than an approximate variant.

## Winning GB200 schedule

The target route uses:

- a `BM128/BN128/BK128` two-CTA `tcgen05` MMA;
- one 256-wide accumulator covering the gate and up halves;
- three TMA operand stages and two FP32 TMEM buffers;
- two epilogue tasks with four 32-column subtiles;
- CLC dynamic tile scheduling;
- `evict_last` for reused A, `evict_first` for streaming B and stores; and
- a scoped 48 MiB persisting-L2 window for the 40 MiB A tensor.

The host wrapper resets the stream access-policy window after launch so later
work does not inherit it. Unsupported shapes use a masked FP32-accumulating
fallback.

## Results

On GPU 0 (GB200), with the GEO profiler-time harness locked to 1200 W and a
2062 MHz application clock, four official runs measured:

| Result | Latency |
|---|---:|
| Frozen fused seed | `0.854433 ms` |
| Tuned fused TLX | `0.392221 / 0.394003 / 0.396722 / 0.402738 ms` |
| Tuned median | `0.395362 ms` (`2.1604x`) |
| Independent GEO Supervisor | `0.399283 ms` (`2.1399x`) |
| Live compiled unfused | approximately `0.430-0.450 ms` |

After promotion, the standalone OSS harness on otherwise-idle physical GPU 1
measured `0.431894 ms` fused versus `0.461567 ms` compiled unfused (`1.0687x`).
The same promoted file measured `0.396338 ms` in the GEO-compatible fused-only
harness. The spread reflects the documented GB200 power/clock transient; the
paired result still shows a fused win.

Accuracy passes seeds 42 and 123. The saved projection is bit exact; compact
output max-absolute error is `0.0078125` and `0.00390625`, respectively, with
relative-L2 near `1e-5` and no tolerance violations.

The GEO ceiling loop later exhausted its model-provider quota after the
Supervisor had promoted this candidate. The promotion measurements and
sha-linked accuracy artifacts are complete; only the final repeated ceiling
bookkeeping was interrupted.

Run the standalone OSS check on an idle Blackwell GPU with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:third_party/tlx/tutorials \
  third_party/tlx/denoise.sh \
  /path/to/python \
  third_party/tlx/tutorials/fwd_swiglu_gemm_saved_bench.py
```
