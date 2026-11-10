# SGLang/vLLM FP8 KV decode comparison — 2026-11-05

TLX is faster than the eligible native SGLang/vLLM paths in these tested synthetic cases. AITER-backed shared-scale controls are much closer: long GQA and SWA are near parity, and **AITER wins the D128 control (12.23 µs versus TLX15.48 µs; TLX takes1.27x as long)**. **The earlier vLLM 2D-only numbers do not establish an advantage over its best decode path; this report supersedes those claims.**

[Chart](charts/2026-11-05_step-09_framework_decode.svg) · [PNG](charts/2026-11-05_step-09_framework_decode.png) · [CSV](results/2026-11-05_step-09_framework_decode.csv) · [Data](results/2026-11-05_step-09_framework_decode.json) · [Manifest](results/2026-11-05_framework_decode/manifest.json) · [Experiment log](FP8_KV_DECODE_ATTN.md)

## Matched BF16-Q shared-scale controls

Same scalar K/V scale is repeated across heads so scalar-only sources can participate. This is a restricted control. Latencies are µs; each value is the median of three fresh, order-rotated round medians.

| Case (Hq/Hkv, D, L) | TLX | vLLM best tested | SGLang best tested | AITER shared operator | vLLM stock 3D |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16/2, D64, L2304 | 3.88 | 5.62 | 9.33 | 4.55 | 7.59 |
| 128/128, D64, L2304 | 10.19 | 26.88 | 44.81 | 13.82 | 30.80 |
| 16/2, D64, L8192 | 4.43 | 6.59 | 14.92 | 4.92 | 16.14 |
| 16/2, D64, L32768 | 7.12 | 10.98 | 35.70 | 11.53 | 51.82 |
| 16/2, D64, L262144 | 17.64 | 48.88 | 108.02 | 18.06 | 388.88 |
| 128/128, D64, L262144 | 574.34 | 2208.05 | 3866.87 | 747.42 | 3810.88 |
| 16/2, D64, L2304, W128 | 2.91 | 3.89 | — | 2.94 | 7.35 |
| 64/64, D128, L2304 | 15.48 | 23.09 | 57.52 | 12.23 | 26.15 |

AITER is one deduplicated underlying operator, not an independent SGLang run and vLLM run. The vLLM ROCm AITER unified backend can route causal gfx1250 shuffled-cache decode to this operator; the tested call uses packed 5D cache and BF16 Q. SGLang also uses AITER, but its unified FP8-KV path may quantize Q and its vectorized 5D path calls a CDNA-only decoder at this revision. Those routes are not automatically equivalent to this BF16-Q control.

## Native unequal per-KV-head scales

Separate, unequal K and V scale vectors are used. SGLang standard/Lean and legacy vLLM accept scalar scales, so no native per-head timing is attributed to them. vLLM unified receives native per-token/head scale-cache views with zero strides across token/page axes, preserving the constant-per-head contract.

| Case (Hq/Hkv, D, L) | TLX µs | vLLM best µs | vLLM/TLX | vLLM configuration |
| --- | ---: | ---: | ---: | --- |
| 16/2, D64, L2304 | 3.91 | 5.48 | 1.40x | `vllm_unified_3d_tuned`, S128, tile 32 |
| 128/128, D64, L2304 | 10.21 | 31.46 | 3.08x | `vllm_unified_3d_tuned`, S8, tile 128 |
| 16/2, D64, L8192 | 4.44 | 6.70 | 1.51x | `vllm_unified_3d_tuned`, S128, tile 64 |
| 16/2, D64, L32768 | 7.15 | 12.34 | 1.73x | `vllm_unified_3d_tuned`, S128, tile 128 |
| 16/2, D64, L262144 | 17.66 | 55.58 | 3.15x | `vllm_unified_3d_tuned`, S256, tile 128 |
| 128/128, D64, L262144 | 574.37 | 3121.10 | 5.43x | `vllm_unified_3d_tuned`, S64, tile 128 |
| 16/2, D64, L2304, W128 | 2.88 | 4.72 | 1.64x | `vllm_unified`, S16, tile 32 |
| 64/64, D128, L2304 | 15.54 | 20.62 | 1.33x | `vllm_unified_3d_tuned`, S8, tile 128 |

## Full FP8 unit-scale controls

| Case | TLX µs | SGLang eligible µs | vLLM eligible µs |
| --- | ---: | ---: | ---: |
| gqa_fp8_unit | 3.73 | 9.34 | — |
| mha_fp8_unit | 9.62 | — | — |

SGLang GQA passes with S32 and unchanged FP8 Q payload; a common Q scale can be folded into the softmax scale (unit scale here). This is operator evidence, not a claim that a production backend advertises this Q dtype. SGLang MHA and legacy vLLM MHA fail AMD lowering with FP8 Q on this build. vLLM GQA/unified and Lean FP8-Q controls fail the unchanged numerical gate. Failures and diagnostic timings are preserved, excluded from the eligible ranking. These unit-scale controls do not prove arbitrary per-head full-FP8 support.

## Selection and reproducibility

- Source pins: vLLM `86fdabec0c82d0470c160aa4df7602ac22c207e6`, SGLang `9b6c796151f836907fbf93265984006ecfd6a790`, AITER `796ff6a35588058516bcaf13129bd0f0b22fb9ae`. [Pinned paths and framework routing audit](REFERENCES.md).
- Native candidates: ordinary SGLang split decode, optional persistent Lean; legacy vLLM decode; unified 2D and 3D; optional tensor descriptors. Split sweep 4/8/16/32/64/128/256; tuned unified decode tiles32/64/128. KV-only TD tiles32/64 and splits4/16/64/128/256, bounded by context/tile policy. SGLang scalar split controls use source defaults besides splits.
- Stock vLLM production-style 3D: sixteen segments, FP32 output/max/sum scratch, threshold128/KV-heads, default tile selection (BF16 decode16, FP8 decode32), descriptors disabled by default on ROCm. The 2D path is also retained because it wins SWA128 with unequal scales.
- `_3d_tuned` changes only a copied Python launcher tile helper. `_kvtd3d`/`_kvtd2d` additionally disable Q/output descriptors while keeping KV descriptors. These are experimental launch configurations; all upstream GPU bodies are unchanged. Unmodified opt-in full-TD MHA fails AMD shared-layout lowering; the KV-only control compiles and can win shared-scale MHA.
- Every selected candidate passes the same independent FP32 oracle on identical quantized payloads/page mappings: `atol=5e-4, rtol=1e-2`. Selection uses five graph-batch samples × ten replays; final verification uses three fresh rounds of fifteen samples × twenty replays. CSV includes round range and max absolute error. Selection minima are not final reported timings.
- gfx1250, one96-CU/144-GiB DPX agent, Torch2.12/ROCm10.1, source-built TLX3.9 compiler shared across sources. GPU0 was checked idle; all GPU jobs ran sequentially; clocks unlocked. Repeated inputs are a warm-cache protocol. Compilation, quantization, input packing and oracle setup excluded; SGLang Lean metadata/lock-reset GPU work included. No full-framework serving or end-to-end inference claim.
- Hq/Hkv mappings, T/mask semantics and actual SWA window are undisclosed. These explicit assumptions cover one-query decode, page64, D64/D128, L2304..256K and window128. They are not exact model acceptance. Attention output precedes the undefined projection.
- [Reproduction commands](REPRODUCE.md). Raw selection/repeat JSON, compiler logs and source snapshots are under `results/2026-11-05_framework_decode/`. TLX kernel files are hash-identical to the preceding optimization report.

## Selected configurations

| Case | Family | Backend | Splits | Actual KV tile | µs | Round range µs |
| --- | --- | --- | ---: | ---: | ---: | --- |
| gqa2304_shared | vllm_native | `vllm_unified_3d` | 256 | 16 | 5.62 | 5.61–5.64 |
| gqa2304_shared | sglang_native | `sglang` | 32 | source default | 9.33 | 9.32–9.34 |
| gqa2304_shared | aiter_shared_operator | `aiter_unified` | 16 | source default | 4.55 | 4.55–4.56 |
| gqa2304_shared | vllm_production | `vllm_unified_3d` | 16 | 16 | 7.59 | 7.59–7.60 |
| gqa2304_per-head | vllm_native | `vllm_unified_3d_tuned` | 128 | 32 | 5.48 | 5.48–5.49 |
| gqa2304_per-head | vllm_production | `vllm_unified_3d` | 16 | 16 | 7.94 | 7.94–7.94 |
| mha2304_shared | vllm_native | `vllm_unified_kvtd3d` | 16 | 64 | 26.88 | 26.84–26.88 |
| mha2304_shared | sglang_native | `sglang` | 32 | source default | 44.81 | 44.77–44.84 |
| mha2304_shared | aiter_shared_operator | `aiter_unified` | 16 | source default | 13.82 | 13.82–13.86 |
| mha2304_shared | vllm_production | `vllm_unified_3d` | 16 | 16 | 30.80 | 30.79–30.82 |
| mha2304_per-head | vllm_native | `vllm_unified_3d_tuned` | 8 | 128 | 31.46 | 31.43–31.47 |
| mha2304_per-head | vllm_production | `vllm_unified_3d` | 16 | 16 | 44.22 | 44.19–44.24 |
| gqa8192_shared | vllm_native | `vllm_unified_kvtd3d` | 128 | 64 | 6.59 | 6.59–6.59 |
| gqa8192_shared | sglang_native | `sglang` | 32 | source default | 14.92 | 14.92–14.93 |
| gqa8192_shared | aiter_shared_operator | `aiter_unified` | 16 | source default | 4.92 | 4.92–4.94 |
| gqa8192_shared | vllm_production | `vllm_unified_3d` | 16 | 16 | 16.14 | 16.13–16.16 |
| gqa8192_per-head | vllm_native | `vllm_unified_3d_tuned` | 128 | 64 | 6.70 | 6.69–6.70 |
| gqa8192_per-head | vllm_production | `vllm_unified_3d` | 16 | 16 | 17.94 | 17.92–17.96 |
| gqa32768_shared | vllm_native | `vllm_unified_kvtd3d` | 128 | 64 | 10.98 | 10.98–11.01 |
| gqa32768_shared | sglang_native | `sglang` | 32 | source default | 35.70 | 35.69–35.71 |
| gqa32768_shared | aiter_shared_operator | `aiter_unified` | 16 | source default | 11.53 | 11.51–11.56 |
| gqa32768_shared | vllm_production | `vllm_unified_3d` | 16 | 16 | 51.82 | 51.80–51.84 |
| gqa32768_per-head | vllm_native | `vllm_unified_3d_tuned` | 128 | 128 | 12.34 | 12.33–12.34 |
| gqa32768_per-head | vllm_production | `vllm_unified_3d` | 16 | 16 | 59.82 | 59.78–59.84 |
| gqa262144_shared | vllm_native | `vllm_unified_kvtd3d` | 256 | 64 | 48.88 | 48.87–48.92 |
| gqa262144_shared | sglang_native | `sglang_lean` | 16 | source default | 108.02 | 107.75–108.39 |
| gqa262144_shared | aiter_shared_operator | `aiter_unified` | 16 | source default | 18.06 | 18.06–18.08 |
| gqa262144_shared | vllm_production | `vllm_unified_3d` | 16 | 16 | 388.88 | 388.67–389.38 |
| gqa262144_per-head | vllm_native | `vllm_unified_3d_tuned` | 256 | 128 | 55.58 | 55.55–55.66 |
| gqa262144_per-head | vllm_production | `vllm_unified_3d` | 16 | 16 | 459.41 | 459.41–459.90 |
| mha262144_shared | vllm_native | `vllm_unified_kvtd3d` | 64 | 64 | 2208.05 | 2204.16–2209.01 |
| mha262144_shared | sglang_native | `sglang_lean` | 16 | source default | 3866.87 | 3864.73–3869.69 |
| mha262144_shared | aiter_shared_operator | `aiter_unified` | 16 | source default | 747.42 | 746.51–747.78 |
| mha262144_shared | vllm_production | `vllm_unified_3d` | 16 | 16 | 3810.88 | 3810.55–3811.44 |
| mha262144_per-head | vllm_native | `vllm_unified_3d_tuned` | 64 | 128 | 3121.10 | 3119.90–3121.99 |
| mha262144_per-head | vllm_production | `vllm_unified_3d` | 16 | 16 | 5102.73 | 5100.17–5103.92 |
| swa128_shared | vllm_native | `vllm_unified_kvtd2d` | 16 | 64 | 3.89 | 3.89–3.90 |
| swa128_shared | aiter_shared_operator | `aiter_unified` | 16 | source default | 2.94 | 2.93–2.95 |
| swa128_shared | vllm_production | `vllm_unified_3d` | 16 | 16 | 7.35 | 7.34–7.35 |
| swa128_per-head | vllm_native | `vllm_unified` | 16 | 32 | 4.72 | 4.72–4.72 |
| swa128_per-head | vllm_production | `vllm_unified_3d` | 16 | 16 | 7.68 | 7.67–7.69 |
| d128_shared | vllm_native | `vllm_unified_kvtd3d` | 16 | 64 | 23.09 | 23.07–23.12 |
| d128_shared | sglang_native | `sglang` | 32 | source default | 57.52 | 57.40–57.52 |
| d128_shared | aiter_shared_operator | `aiter_unified` | 16 | source default | 12.23 | 12.21–12.24 |
| d128_shared | vllm_production | `vllm_unified_3d` | 16 | 16 | 26.15 | 26.14–26.15 |
| d128_per-head | vllm_native | `vllm_unified_3d_tuned` | 8 | 128 | 20.62 | 20.61–20.66 |
| d128_per-head | vllm_production | `vllm_unified_3d` | 16 | 16 | 29.81 | 29.80–29.84 |
| gqa_fp8_unit | sglang_native | `sglang` | 32 | source default | 9.34 | 9.31–9.37 |
