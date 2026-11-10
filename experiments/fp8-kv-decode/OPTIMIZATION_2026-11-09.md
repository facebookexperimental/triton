# TLX FP8 KV decode: final retained kernel

80 synthetic cases; 3 timing round(s) per implementation. TLX correctness: 80/80 passing cases. Eligible AITER comparisons: 40. At a 3% reporting band: 31 favor TLX, 4 near parity, 5 favor AITER. This is the tested matrix, not a universal performance claim.

[Core chart](charts/2026-11-09_step-15_final_matrix.svg) · [CSV](results/2026-11-09_step-15_final_matrix.csv) · [Data](results/2026-11-09_step-15_final_matrix.json) · [Raw run](results/2026-11-09_optimization/repeated_final_v2.json) · [Experiment log](FP8_KV_DECODE_ATTN.md)

| Case | Frozen TLX µs | Current TLX µs | AITER µs | AITER/current |
| --- | ---: | ---: | ---: | ---: |
| gqa64_l2304 | 3.87 | 3.86 | 4.54 | 1.18x |
| mha64_l2304 | 10.24 | 8.50 | 13.90 | 1.64x |
| mha128_l2304 | 15.42 | 8.45 | 12.20 | 1.44x |

Family charts: [gqa64](charts/2026-11-09_step-15_final_matrix_gqa64.svg) · [mha64](charts/2026-11-09_step-15_final_matrix_mha64.svg) · [gqa128](charts/2026-11-09_step-15_final_matrix_gqa128.svg) · [mha128](charts/2026-11-09_step-15_final_matrix_mha128.svg)

The frozen code is the November 9 starting implementation; all comparisons run sequentially on CUDA_VISIBLE_DEVICES=2, the same 96-CU gfx1250 agent/compiler. Q/O shapes and cache values/page mappings are identical within each case. AITER uses the pinned unified operator with scalar KV scales. Shared/unit controls restrict scales so it can participate. Unequal-head scale and noncausal multi-query cases retain native TLX/frozen checks; AITER has no matching adapter for them. Failed AITER accuracy gates are excluded from performance rankings and retained in raw rows.

Working mappings: GQA16/2,D64 orD128; MHA128/128,D64; MHA64/64,D128. Cache lengths 1024..256K, page64/128, assumed windows37/128. T>1 means noncausal queries over one shared cache, with T=8..8192. Exact model mappings/mask/window/projection remain undisclosed. BF16 output is before projection.

The unchanged oracle uses FP32 attention on actual quantized inputs with atol=5e-4/rtol=1e-2. Timing uses 15 graph-batch samples of 20 invocations per round. Compilation, input packing, quantization and oracle setup are excluded. Reported values are medians of round medians; raw CSV includes round ranges. Repeated inputs are a warm-cache protocol; clocks unlocked. A one-round scout is provisional and does not establish repeatability.

## Retained changes and validation

One-warp D128 decode with page/length-based splits closes the original dense D128 gap. Packed K/V LDS loads use matrix-operand register ownership, and one-warp K/V waits are staggered around QK/PV. Proven single-tile partitions remove dynamic online state. FP16 normalized raw-V partials start at 8 splits; LSE/merge remain FP32 and apply the separate V head scale. The original multi-warp loop/fused schedule is retained for larger query sets, and tiny windows/FP8-Q grouped D128 keep their better four-warp choices.

Final strict suite: **739 passed, 8 skipped**, unchanged atol=5e-4/rtol=1e-2. It covers unequal head scales, BF16/FP8 Q, poisoned tails, empty contexts, page16/32/64/128, peaked multiseed inputs, workspace reuse, S8/12 boundaries and aggregate tiny probability mass. All 80 benchmark cases also pass in each of three rounds. [JUnit](results/2026-11-09_optimization/correctness_final_v2.xml) · [Manifest](results/2026-11-09_optimization/manifest.json)

The per-head E4M3FN contract is unchanged; this is not E8M0 block-scaled MXFP8. The original prototype partition/merge function ASTs are preserved. No compiler/native changes, reference GPU edits or relaxed accuracy tolerances were used.

## Remaining gaps

Of 40 passing AITER controls, 31 favor TLX beyond 3%, 4 are within 3%, and 5 favor AITER. Eight other AITER cases fail this project's numerical gate and are excluded; 32 cases have no matching AITER adapter for arbitrary per-head scales or the chosen multi-query semantics. This does not establish an upstream correctness bug under a different accuracy target.

| AITER-favored case | TLX µs | AITER µs |
| --- | ---: | ---: |
| gqa128_l8192 | 5.53 | 5.10 |
| gqa128_p128_l2304 | 4.70 | 4.50 |
| gqa128_p128_l8192 | 5.15 | 4.84 |
| gqa128_w128 | 3.54 | 3.08 |
| mha128_w128 | 3.74 | 3.10 |

Current versus starting-TLX regressions beyond 3%: **0 cases**.

## Fresh core framework comparison

Prior strongest passing SGLang/vLLM configurations are repeated without a new selection sweep. These are standalone source-kernel measurements with the shared compiler, not framework-serving throughput. AITER remains one deduplicated underlying operator; scalar shared-scale controls do not establish native arbitrary-head scaling for every reference.

[Framework chart](charts/2026-11-09_step-16_framework_repeat.svg) · [CSV](results/2026-11-09_step-16_framework_repeat.csv) · [Data](results/2026-11-09_step-16_framework_repeat.json)

| Case | TLX | SGLang best tested | vLLM best tested | AITER |
| --- | ---: | ---: | ---: | ---: |
| GQA, D64 | 3.83 | 9.37 | 5.61 | 4.55 |
| MHA, D64 | 8.65 | 44.75 | 26.84 | 14.07 |
| MHA, D128 | 8.26 | 57.50 | 23.08 | 12.23 |

## Profiling and rejected experiments

rocprofv3 ATT/analyzer captured valid old/new dense D128 samples. The old sample concentrated 57.1% of stall cycles at LDS waits; the newer sample concentrated 33.1% at KMCNT and 27.5% at tensor waits. These are selected-CU sampled instruction fractions, not GPU utilization or normalized kernel latency. Short-window captures missed instruction coverage, and the last AITER decoder retry was interrupted by the host reboot. No bottleneck claim is based on empty traces or previously all-zero PMC counters. [Old ATT](results/2026-11-09_optimization/att_d128_before_analysis.txt) · [Current dense ATT](results/2026-11-09_optimization/att_tlx_mha128_analysis.txt)

Rejected attempts include vector replacements, multipage tiles, host descriptors, forced scheduling, VMEM scale loads, general single buffering, direct K/V loads, custom merge ownership and loop unrolling. Two-component FP8 probabilities were slower and failed tiny-mass accuracy tests; FP16 PV was restored and that regression is retained. Forced mixed-dtype AITER Gluon2D also failed and was excluded. The actual stock BF16-Q/FP8-KV SWA route is AITER Triton2D. [Dated experiment ledger041–077](FP8_KV_DECODE_ATTN.md) · [Pinned sources/routing](REFERENCES.md)

Final comparisons followed a host reboot/container restart and reran frozen/current/reference together on the verified 96-CU/144-GiB runtime agent, CUDA_VISIBLE_DEVICES=2, PCI2:4:0. Earlier paired/scout charts remain separate. Clocks are unlocked; repeated same-input graph timings favor warm caches. The exact model head mapping, T/mask semantics, SWA window and output projection remain undisclosed. These finite synthetic tests cannot establish a win across every shape or end-to-end serving.

## Complete matrix

| Case | Current µs | Frozen µs | AITER µs | AITER/current | Result | Current launch |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| gqa64_l1024 | 3.65 | 3.69 | 3.94 | 1.08x | tlx_faster | matrix_tdm_flat, S16, W4, BN64 |
| gqa64_l2304 | 3.86 | 3.87 | 4.54 | 1.18x | tlx_faster | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_l8192 | 4.43 | 4.46 | 4.93 | 1.11x | tlx_faster | matrix_tdm_flat, S128, W4, BN64 |
| gqa64_l32768 | 7.17 | 7.11 | 11.64 | 1.62x | tlx_faster | matrix_tdm_flat, S128, W4, BN64 |
| gqa64_l131072 | 10.77 | 11.21 | 14.74 | 1.37x | tlx_faster | matrix_tdm_flat, S256, W1, BN64 |
| gqa64_l262144 | 16.80 | 17.67 | 18.03 | 1.07x | tlx_faster | matrix_tdm_flat, S256, W1, BN64 |
| gqa64_p128_l2304 | 3.90 | 4.25 | 4.28 | 1.10x | tlx_faster | matrix_tdm_flat, S18, W1, BN128 |
| gqa64_p128_l8192 | 4.03 | 4.73 | 4.94 | 1.23x | tlx_faster | matrix_tdm_flat, S64, W1, BN128 |
| gqa64_p128_l32768 | 5.79 | 6.96 | 6.33 | 1.09x | tlx_faster | matrix_tdm_flat, S256, W1, BN128 |
| gqa64_w37 | 2.32 | 2.31 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa64_w128 | 2.91 | 2.91 | 2.94 | 1.01x | within_3_percent | matrix_tdm_flat, S1, W4, BN64 |
| gqa64_unequal | 3.78 | 3.92 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_fp8_unit | 3.71 | 3.76 | — | failed | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_fp8_unequal | 3.67 | 3.75 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_t8 | 4.64 | 5.10 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_t32 | 6.45 | 7.80 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_t128 | 9.40 | 11.06 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W4, BN64 |
| gqa64_t512 | 24.75 | 24.76 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| gqa64_t2048 | 69.22 | 69.17 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa64_t8192 | 223.84 | 223.48 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha64_l1024 | 6.54 | 6.79 | 11.58 | 1.77x | tlx_faster | matrix_tdm_flat, S4, W1, BN64 |
| mha64_l2304 | 8.50 | 10.24 | 13.90 | 1.64x | tlx_faster | matrix_tdm_flat, S9, W1, BN64 |
| mha64_l8192 | 27.55 | 27.63 | 36.08 | 1.31x | tlx_faster | matrix_tdm_flat, S32, W1, BN64 |
| mha64_l32768 | 92.52 | 90.32 | 105.74 | 1.14x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha64_l131072 | 309.76 | 302.70 | 380.93 | 1.23x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha64_l262144 | 597.39 | 592.76 | 738.39 | 1.24x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha64_p128_l2304 | 8.67 | 10.24 | 12.96 | 1.49x | tlx_faster | matrix_tdm_flat, S9, W1, BN128 |
| mha64_p128_l8192 | 25.61 | 30.19 | 31.09 | 1.21x | tlx_faster | matrix_tdm_flat, S8, W1, BN128 |
| mha64_p128_l32768 | 80.25 | 103.09 | 94.21 | 1.17x | tlx_faster | matrix_tdm_flat, S32, W1, BN128 |
| mha64_w37 | 2.59 | 2.65 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W1, BN64 |
| mha64_w128 | 3.07 | 3.35 | 3.03 | 0.99x | within_3_percent | matrix_tdm_flat, S1, W1, BN64 |
| mha64_unequal | 8.34 | 10.21 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S9, W1, BN64 |
| mha64_fp8_unit | 7.86 | 9.62 | — | failed | no_eligible_aiter | matrix_tdm_flat, S9, W1, BN64 |
| mha64_fp8_unequal | 7.83 | 9.65 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S9, W1, BN64 |
| mha64_t8 | 19.02 | 18.98 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| mha64_t32 | 22.07 | 22.06 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| mha64_t128 | 49.83 | 49.84 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S2, W4, BN64 |
| mha64_t512 | 118.01 | 117.74 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha64_t2048 | 431.90 | 431.45 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha64_t8192 | 1726.29 | 1720.50 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa128_l1024 | 4.09 | 4.31 | 4.17 | 1.02x | within_3_percent | matrix_tdm_flat, S16, W1, BN64 |
| gqa128_l2304 | 4.48 | 4.59 | 5.28 | 1.18x | tlx_faster | matrix_tdm_flat, S36, W1, BN64 |
| gqa128_l8192 | 5.53 | 6.02 | 5.10 | 0.92x | aiter_faster | matrix_tdm_flat, S64, W1, BN64 |
| gqa128_l32768 | 9.14 | 10.08 | 11.55 | 1.26x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| gqa128_l131072 | 14.76 | 26.95 | 17.26 | 1.17x | tlx_faster | matrix_tdm_flat, S384, W1, BN64 |
| gqa128_l262144 | 27.00 | 51.39 | 35.79 | 1.33x | tlx_faster | matrix_tdm_flat, S384, W1, BN64 |
| gqa128_p128_l2304 | 4.70 | 5.38 | 4.50 | 0.96x | aiter_faster | matrix_tdm_flat, S18, W1, BN128 |
| gqa128_p128_l8192 | 5.15 | 6.14 | 4.84 | 0.94x | aiter_faster | matrix_tdm_flat, S64, W1, BN128 |
| gqa128_p128_l32768 | 7.58 | 14.19 | 8.40 | 1.11x | tlx_faster | matrix_tdm_flat, S128, W1, BN128 |
| gqa128_w37 | 2.81 | 2.82 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa128_w128 | 3.54 | 3.76 | 3.08 | 0.87x | aiter_faster | matrix_tdm_flat, S1, W1, BN64 |
| gqa128_unequal | 4.48 | 4.63 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W1, BN64 |
| gqa128_fp8_unit | 4.35 | 4.38 | — | failed | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_fp8_unequal | 4.35 | 4.38 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_t8 | 6.36 | 7.01 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_t32 | 9.50 | 12.54 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_t128 | 20.40 | 24.69 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W4, BN64 |
| gqa128_t512 | 49.29 | 49.25 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| gqa128_t2048 | 126.92 | 126.72 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa128_t8192 | 462.79 | 462.36 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_l1024 | 7.21 | 8.98 | 9.28 | 1.29x | tlx_faster | matrix_tdm_flat, S6, W1, BN64 |
| mha128_l2304 | 8.45 | 15.42 | 12.20 | 1.44x | tlx_faster | matrix_tdm_flat, S12, W1, BN64 |
| mha128_l8192 | 28.40 | 46.39 | 33.82 | 1.19x | tlx_faster | matrix_tdm_flat, S43, W1, BN64 |
| mha128_l32768 | 88.79 | 168.48 | 107.20 | 1.21x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha128_l131072 | 286.26 | 556.03 | 398.32 | 1.39x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha128_l262144 | 552.18 | 1080.59 | 772.45 | 1.40x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha128_p128_l2304 | 8.17 | 20.38 | 9.33 | 1.14x | tlx_faster | matrix_tdm_flat, S6, W1, BN128 |
| mha128_p128_l8192 | 27.86 | 73.29 | 27.86 | 1.00x | within_3_percent | matrix_tdm_flat, S8, W1, BN128 |
| mha128_p128_l32768 | 84.61 | 274.41 | 90.35 | 1.07x | tlx_faster | matrix_tdm_flat, S32, W1, BN128 |
| mha128_w37 | 2.94 | 2.94 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_w128 | 3.74 | 3.93 | 3.10 | 0.83x | aiter_faster | matrix_tdm_flat, S1, W1, BN64 |
| mha128_unequal | 8.44 | 15.59 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W1, BN64 |
| mha128_fp8_unit | 8.20 | 15.04 | — | failed | no_eligible_aiter | matrix_tdm_flat, S12, W1, BN64 |
| mha128_fp8_unequal | 8.21 | 15.06 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W1, BN64 |
| mha128_t8 | 22.96 | 22.92 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S6, W4, BN64 |
| mha128_t32 | 28.00 | 27.99 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S6, W4, BN64 |
| mha128_t128 | 52.35 | 52.34 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| mha128_t512 | 128.28 | 128.03 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_t2048 | 467.95 | 466.61 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_t8192 | 1799.30 | 1796.36 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
