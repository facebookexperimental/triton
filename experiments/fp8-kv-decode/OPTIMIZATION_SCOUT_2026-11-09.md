# TLX optimization: first80-case scout

80 synthetic cases; 1 timing round(s) per implementation. TLX correctness: 80/80 passing cases. Eligible AITER comparisons: 40. At a3% reporting band: 26 favor TLX, 3 near parity, 11 favor AITER. This is the tested matrix, not a universal performance claim.

[Core chart](charts/2026-11-09_step-13_full_scout.svg) · [CSV](results/2026-11-09_step-13_full_scout.csv) · [Data](results/2026-11-09_step-13_full_scout.json) · [Raw run](results/2026-11-09_optimization/scout_full_v1.json) · [Experiment log](FP8_KV_DECODE_ATTN.md)

| Case | Frozen TLX µs | Current TLX µs | AITER µs | AITER/current |
| --- | ---: | ---: | ---: | ---: |
| gqa64_l2304 | 4.12 | 4.16 | 4.84 | 1.16x |
| mha64_l2304 | 10.44 | 8.71 | 13.97 | 1.60x |
| mha128_l2304 | 15.53 | 9.00 | 12.89 | 1.43x |

Family charts: [gqa64](charts/2026-11-09_step-13_full_scout_gqa64.svg) · [mha64](charts/2026-11-09_step-13_full_scout_mha64.svg) · [gqa128](charts/2026-11-09_step-13_full_scout_gqa128.svg) · [mha128](charts/2026-11-09_step-13_full_scout_mha128.svg)

The frozen code is the November9 starting implementation; all comparisons run sequentially on CUDA_VISIBLE_DEVICES=2, the same96-CU gfx1250 agent/compiler. Q/O shapes and cache values/page mappings are identical within each case. AITER uses the pinned unified operator with scalar KV scales. Shared/unit controls restrict scales so it can participate. Unequal-head scale and noncausal multi-query cases retain native TLX/frozen checks; AITER has no matching adapter for them. Failed AITER accuracy gates are excluded from performance rankings and retained in raw rows.

Working mappings: GQA16/2,D64 orD128; MHA128/128,D64; MHA64/64,D128. Cache lengths1024..256K, page64/128, assumed windows37/128. T>1 means noncausal queries over one shared cache, with T8..8192. Exact model mappings/mask/window/projection remain undisclosed. BF16 output is before projection.

The unchanged oracle uses FP32 attention on actual quantized inputs with atol5e-4/rtol1e-2. Timing uses 5 graph-batch samples of 10 invocations per round. Compilation, input packing, quantization and oracle setup are excluded. Reported values are medians of round medians; raw CSV includes round ranges. Repeated inputs are a warm-cache protocol; clocks unlocked. A one-round scout is provisional and does not establish repeatability.

| Case | Current µs | Frozen µs | AITER µs | AITER/current | Result | Current launch |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| gqa64_l1024 | 3.90 | 4.02 | 4.24 | 1.09x | tlx_faster | matrix_tdm_flat, S16, W4, BN64 |
| gqa64_l2304 | 4.16 | 4.12 | 4.84 | 1.16x | tlx_faster | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_l8192 | 4.69 | 4.77 | 5.13 | 1.09x | tlx_faster | matrix_tdm_flat, S128, W4, BN64 |
| gqa64_l32768 | 7.46 | 7.36 | 11.82 | 1.59x | tlx_faster | matrix_tdm_flat, S128, W4, BN64 |
| gqa64_l131072 | 10.63 | 11.14 | 14.53 | 1.37x | tlx_faster | matrix_tdm_flat, S256, W1, BN64 |
| gqa64_l262144 | 17.06 | 17.89 | 18.14 | 1.06x | tlx_faster | matrix_tdm_flat, S256, W1, BN64 |
| gqa64_p128_l2304 | 4.52 | 4.61 | 4.53 | 1.00x | within_3_percent | matrix_tdm_flat, S18, W4, BN128 |
| gqa64_p128_l8192 | 4.95 | 4.96 | 5.18 | 1.05x | tlx_faster | matrix_tdm_flat, S64, W4, BN128 |
| gqa64_p128_l32768 | 7.23 | 7.21 | 6.60 | 0.91x | aiter_faster | matrix_tdm_flat, S128, W4, BN128 |
| gqa64_w37 | 2.61 | 2.62 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa64_w128 | 3.16 | 3.17 | 3.20 | 1.01x | within_3_percent | matrix_tdm_flat, S1, W4, BN64 |
| gqa64_unequal | 4.09 | 4.15 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_fp8_unit | 4.03 | 4.01 | — | failed | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_fp8_unequal | 3.99 | 4.01 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_t8 | 5.33 | 5.35 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_t32 | 7.91 | 8.04 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_t128 | 11.43 | 11.34 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W4, BN64 |
| gqa64_t512 | 25.49 | 24.91 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| gqa64_t2048 | 71.95 | 70.54 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa64_t8192 | 237.58 | 222.51 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha64_l1024 | 6.78 | 7.07 | 11.83 | 1.74x | tlx_faster | matrix_tdm_flat, S4, W1, BN64 |
| mha64_l2304 | 8.71 | 10.44 | 13.97 | 1.60x | tlx_faster | matrix_tdm_flat, S9, W1, BN64 |
| mha64_l8192 | 28.43 | 27.32 | 36.51 | 1.28x | tlx_faster | matrix_tdm_flat, S32, W1, BN64 |
| mha64_l32768 | 102.47 | 93.47 | 106.14 | 1.04x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha64_l131072 | 314.03 | 304.71 | 379.27 | 1.21x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha64_l262144 | 592.77 | 585.78 | 732.45 | 1.24x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha64_p128_l2304 | 9.02 | 10.69 | 13.27 | 1.47x | tlx_faster | matrix_tdm_flat, S9, W1, BN128 |
| mha64_p128_l8192 | 28.88 | 30.40 | 31.88 | 1.10x | tlx_faster | matrix_tdm_flat, S32, W1, BN128 |
| mha64_p128_l32768 | 101.83 | 107.45 | 95.18 | 0.93x | aiter_faster | matrix_tdm_flat, S128, W1, BN128 |
| mha64_w37 | 2.89 | 2.89 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha64_w128 | 3.62 | 3.61 | 3.29 | 0.91x | aiter_faster | matrix_tdm_flat, S1, W4, BN64 |
| mha64_unequal | 8.69 | 10.44 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S9, W1, BN64 |
| mha64_fp8_unit | 8.52 | 9.92 | — | failed | no_eligible_aiter | matrix_tdm_flat, S9, W1, BN64 |
| mha64_fp8_unequal | 8.57 | 9.92 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S9, W1, BN64 |
| mha64_t8 | 19.65 | 19.29 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| mha64_t32 | 22.74 | 22.03 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| mha64_t128 | 51.38 | 50.08 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S2, W4, BN64 |
| mha64_t512 | 124.21 | 117.90 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha64_t2048 | 462.04 | 429.08 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha64_t8192 | 1825.18 | 1679.80 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa128_l1024 | 4.47 | 4.60 | 4.41 | 0.99x | within_3_percent | matrix_tdm_flat, S16, W4, BN64 |
| gqa128_l2304 | 4.86 | 4.88 | 5.58 | 1.15x | tlx_faster | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_l8192 | 6.27 | 6.28 | 5.40 | 0.86x | aiter_faster | matrix_tdm_flat, S128, W4, BN64 |
| gqa128_l32768 | 10.17 | 10.31 | 11.72 | 1.15x | tlx_faster | matrix_tdm_flat, S128, W4, BN64 |
| gqa128_l131072 | 15.65 | 27.67 | 17.74 | 1.13x | tlx_faster | matrix_tdm_flat, S384, W1, BN64 |
| gqa128_l262144 | 27.59 | 51.48 | 36.20 | 1.31x | tlx_faster | matrix_tdm_flat, S384, W1, BN64 |
| gqa128_p128_l2304 | 5.71 | 5.65 | 4.78 | 0.84x | aiter_faster | matrix_tdm_flat, S18, W4, BN128 |
| gqa128_p128_l8192 | 6.47 | 6.42 | 5.11 | 0.79x | aiter_faster | matrix_tdm_flat, S64, W4, BN128 |
| gqa128_p128_l32768 | 14.39 | 14.36 | 8.66 | 0.60x | aiter_faster | matrix_tdm_flat, S128, W4, BN128 |
| gqa128_w37 | 3.05 | 3.07 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa128_w128 | 4.00 | 4.04 | 3.40 | 0.85x | aiter_faster | matrix_tdm_flat, S1, W4, BN64 |
| gqa128_unequal | 4.80 | 4.90 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_fp8_unit | 4.64 | 4.64 | — | failed | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_fp8_unequal | 4.66 | 4.63 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_t8 | 7.30 | 7.26 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_t32 | 12.73 | 12.85 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_t128 | 24.85 | 24.82 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W4, BN64 |
| gqa128_t512 | 49.06 | 49.34 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| gqa128_t2048 | 124.68 | 126.57 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa128_t8192 | 451.46 | 459.14 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_l1024 | 7.48 | 9.16 | 9.61 | 1.29x | tlx_faster | matrix_tdm_flat, S6, W1, BN64 |
| mha128_l2304 | 9.00 | 15.53 | 12.89 | 1.43x | tlx_faster | matrix_tdm_flat, S12, W1, BN64 |
| mha128_l8192 | 29.53 | 48.53 | 34.24 | 1.16x | tlx_faster | matrix_tdm_flat, S43, W1, BN64 |
| mha128_l32768 | 96.34 | 167.92 | 107.43 | 1.12x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha128_l131072 | 289.85 | 551.91 | 397.64 | 1.37x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha128_l262144 | 549.16 | 1063.12 | 770.06 | 1.40x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha128_p128_l2304 | 8.26 | 20.47 | 9.52 | 1.15x | tlx_faster | matrix_tdm_flat, S6, W1, BN128 |
| mha128_p128_l8192 | 30.01 | 73.15 | 28.43 | 0.95x | aiter_faster | matrix_tdm_flat, S22, W1, BN128 |
| mha128_p128_l32768 | 102.55 | 273.94 | 91.07 | 0.89x | aiter_faster | matrix_tdm_flat, S86, W1, BN128 |
| mha128_w37 | 3.23 | 3.19 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_w128 | 4.23 | 4.20 | 3.37 | 0.80x | aiter_faster | matrix_tdm_flat, S1, W4, BN64 |
| mha128_unequal | 8.25 | 15.52 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W1, BN64 |
| mha128_fp8_unit | 8.21 | 15.08 | — | failed | no_eligible_aiter | matrix_tdm_flat, S12, W1, BN64 |
| mha128_fp8_unequal | 8.24 | 15.10 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W1, BN64 |
| mha128_t8 | 23.05 | 23.03 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S6, W4, BN64 |
| mha128_t32 | 28.07 | 28.09 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S6, W4, BN64 |
| mha128_t128 | 52.06 | 52.13 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| mha128_t512 | 126.19 | 127.88 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_t2048 | 455.73 | 462.72 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_t8192 | 1747.72 | 1772.93 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
