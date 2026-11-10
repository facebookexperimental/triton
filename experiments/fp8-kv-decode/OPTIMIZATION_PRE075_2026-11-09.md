# TLX FP8 KV decode: retained optimizations

80 synthetic cases; 3 timing round(s) per implementation. TLX correctness: 80/80 passing cases. Eligible AITER comparisons: 40. At a3% reporting band: 31 favor TLX, 4 near parity, 5 favor AITER. This is the tested matrix, not a universal performance claim.

[Core chart](charts/2026-11-09_step-14_optimized_matrix.svg) · [CSV](results/2026-11-09_step-14_optimized_matrix.csv) · [Data](results/2026-11-09_step-14_optimized_matrix.json) · [Raw run](results/2026-11-09_optimization/repeated_full_retained.json) · [Experiment log](FP8_KV_DECODE_ATTN.md)

| Case | Frozen TLX µs | Current TLX µs | AITER µs | AITER/current |
| --- | ---: | ---: | ---: | ---: |
| gqa64_l2304 | 3.86 | 3.80 | 4.54 | 1.19x |
| mha64_l2304 | 10.19 | 8.43 | 13.93 | 1.65x |
| mha128_l2304 | 15.22 | 8.26 | 12.24 | 1.48x |

Family charts: [gqa64](charts/2026-11-09_step-14_optimized_matrix_gqa64.svg) · [mha64](charts/2026-11-09_step-14_optimized_matrix_mha64.svg) · [gqa128](charts/2026-11-09_step-14_optimized_matrix_gqa128.svg) · [mha128](charts/2026-11-09_step-14_optimized_matrix_mha128.svg)

The frozen code is the November9 starting implementation; all comparisons run sequentially on CUDA_VISIBLE_DEVICES=2, the same96-CU gfx1250 agent/compiler. Q/O shapes and cache values/page mappings are identical within each case. AITER uses the pinned unified operator with scalar KV scales. Shared/unit controls restrict scales so it can participate. Unequal-head scale and noncausal multi-query cases retain native TLX/frozen checks; AITER has no matching adapter for them. Failed AITER accuracy gates are excluded from performance rankings and retained in raw rows.

Working mappings: GQA16/2,D64 orD128; MHA128/128,D64; MHA64/64,D128. Cache lengths1024..256K, page64/128, assumed windows37/128. T>1 means noncausal queries over one shared cache, with T8..8192. Exact model mappings/mask/window/projection remain undisclosed. BF16 output is before projection.

The unchanged oracle uses FP32 attention on actual quantized inputs with atol5e-4/rtol1e-2. Timing uses 15 graph-batch samples of 20 invocations per round. Compilation, input packing, quantization and oracle setup are excluded. Reported values are medians of round medians; raw CSV includes round ranges. Repeated inputs are a warm-cache protocol; clocks unlocked. A one-round scout is provisional and does not establish repeatability.

| Case | Current µs | Frozen µs | AITER µs | AITER/current | Result | Current launch |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| gqa64_l1024 | 3.60 | 3.69 | 3.91 | 1.09x | tlx_faster | matrix_tdm_flat, S16, W4, BN64 |
| gqa64_l2304 | 3.80 | 3.86 | 4.54 | 1.19x | tlx_faster | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_l8192 | 4.41 | 4.44 | 4.89 | 1.11x | tlx_faster | matrix_tdm_flat, S128, W4, BN64 |
| gqa64_l32768 | 7.15 | 7.11 | 11.55 | 1.62x | tlx_faster | matrix_tdm_flat, S128, W4, BN64 |
| gqa64_l131072 | 10.40 | 10.86 | 14.30 | 1.37x | tlx_faster | matrix_tdm_flat, S256, W1, BN64 |
| gqa64_l262144 | 16.78 | 17.65 | 17.93 | 1.07x | tlx_faster | matrix_tdm_flat, S256, W1, BN64 |
| gqa64_p128_l2304 | 3.89 | 4.25 | 4.26 | 1.10x | tlx_faster | matrix_tdm_flat, S18, W1, BN128 |
| gqa64_p128_l8192 | 4.01 | 4.69 | 4.91 | 1.23x | tlx_faster | matrix_tdm_flat, S64, W1, BN128 |
| gqa64_p128_l32768 | 5.80 | 6.95 | 6.37 | 1.10x | tlx_faster | matrix_tdm_flat, S256, W1, BN128 |
| gqa64_w37 | 2.30 | 2.31 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa64_w128 | 2.88 | 2.90 | 2.91 | 1.01x | within_3_percent | matrix_tdm_flat, S1, W4, BN64 |
| gqa64_unequal | 3.75 | 3.89 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_fp8_unit | 3.64 | 3.69 | — | failed | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_fp8_unequal | 3.64 | 3.71 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_t8 | 4.56 | 5.10 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_t32 | 6.41 | 7.81 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa64_t128 | 9.54 | 11.06 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W4, BN64 |
| gqa64_t512 | 25.32 | 24.81 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| gqa64_t2048 | 71.92 | 69.18 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa64_t8192 | 243.40 | 224.07 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha64_l1024 | 6.63 | 6.87 | 11.61 | 1.75x | tlx_faster | matrix_tdm_flat, S4, W1, BN64 |
| mha64_l2304 | 8.43 | 10.19 | 13.93 | 1.65x | tlx_faster | matrix_tdm_flat, S9, W1, BN64 |
| mha64_l8192 | 27.50 | 28.03 | 36.41 | 1.32x | tlx_faster | matrix_tdm_flat, S32, W1, BN64 |
| mha64_l32768 | 92.67 | 90.07 | 106.34 | 1.15x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha64_l131072 | 310.94 | 304.28 | 381.77 | 1.23x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha64_l262144 | 598.01 | 593.17 | 739.30 | 1.24x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha64_p128_l2304 | 8.77 | 10.28 | 13.10 | 1.49x | tlx_faster | matrix_tdm_flat, S9, W1, BN128 |
| mha64_p128_l8192 | 25.81 | 30.50 | 31.63 | 1.23x | tlx_faster | matrix_tdm_flat, S8, W1, BN128 |
| mha64_p128_l32768 | 80.70 | 102.40 | 94.91 | 1.18x | tlx_faster | matrix_tdm_flat, S32, W1, BN128 |
| mha64_w37 | 2.61 | 2.65 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W1, BN64 |
| mha64_w128 | 3.10 | 3.37 | 3.07 | 0.99x | within_3_percent | matrix_tdm_flat, S1, W1, BN64 |
| mha64_unequal | 8.44 | 10.25 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S9, W1, BN64 |
| mha64_fp8_unit | 7.95 | 9.66 | — | failed | no_eligible_aiter | matrix_tdm_flat, S9, W1, BN64 |
| mha64_fp8_unequal | 7.92 | 9.67 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S9, W1, BN64 |
| mha64_t8 | 19.51 | 19.07 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| mha64_t32 | 22.58 | 21.87 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| mha64_t128 | 51.29 | 49.94 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S2, W4, BN64 |
| mha64_t512 | 126.16 | 118.21 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha64_t2048 | 470.72 | 432.29 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha64_t8192 | 1853.83 | 1724.08 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa128_l1024 | 4.11 | 4.40 | 4.18 | 1.02x | within_3_percent | matrix_tdm_flat, S16, W1, BN64 |
| gqa128_l2304 | 4.48 | 4.59 | 5.28 | 1.18x | tlx_faster | matrix_tdm_flat, S36, W1, BN64 |
| gqa128_l8192 | 5.57 | 6.07 | 5.16 | 0.93x | aiter_faster | matrix_tdm_flat, S64, W1, BN64 |
| gqa128_l32768 | 9.17 | 10.12 | 11.47 | 1.25x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| gqa128_l131072 | 14.81 | 27.00 | 17.32 | 1.17x | tlx_faster | matrix_tdm_flat, S384, W1, BN64 |
| gqa128_l262144 | 27.25 | 51.83 | 35.92 | 1.32x | tlx_faster | matrix_tdm_flat, S384, W1, BN64 |
| gqa128_p128_l2304 | 4.70 | 5.40 | 4.49 | 0.96x | aiter_faster | matrix_tdm_flat, S18, W1, BN128 |
| gqa128_p128_l8192 | 5.15 | 6.15 | 4.88 | 0.95x | aiter_faster | matrix_tdm_flat, S64, W1, BN128 |
| gqa128_p128_l32768 | 7.58 | 14.16 | 8.43 | 1.11x | tlx_faster | matrix_tdm_flat, S128, W1, BN128 |
| gqa128_w37 | 2.95 | 2.81 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W1, BN64 |
| gqa128_w128 | 3.54 | 3.77 | 3.07 | 0.87x | aiter_faster | matrix_tdm_flat, S1, W1, BN64 |
| gqa128_unequal | 4.48 | 4.59 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W1, BN64 |
| gqa128_fp8_unit | 4.62 | 4.36 | — | failed | no_eligible_aiter | matrix_tdm_flat, S36, W1, BN64 |
| gqa128_fp8_unequal | 4.65 | 4.36 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W1, BN64 |
| gqa128_t8 | 6.27 | 6.99 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_t32 | 9.50 | 12.58 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S36, W4, BN64 |
| gqa128_t128 | 20.49 | 24.69 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W4, BN64 |
| gqa128_t512 | 49.17 | 49.30 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| gqa128_t2048 | 126.12 | 127.65 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| gqa128_t8192 | 458.37 | 464.21 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_l1024 | 7.21 | 8.99 | 9.33 | 1.29x | tlx_faster | matrix_tdm_flat, S6, W1, BN64 |
| mha128_l2304 | 8.26 | 15.22 | 12.24 | 1.48x | tlx_faster | matrix_tdm_flat, S12, W1, BN64 |
| mha128_l8192 | 28.37 | 46.68 | 34.01 | 1.20x | tlx_faster | matrix_tdm_flat, S43, W1, BN64 |
| mha128_l32768 | 89.28 | 169.78 | 107.46 | 1.20x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha128_l131072 | 286.53 | 557.76 | 399.03 | 1.39x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha128_l262144 | 553.45 | 1086.12 | 773.59 | 1.40x | tlx_faster | matrix_tdm_flat, S128, W1, BN64 |
| mha128_p128_l2304 | 8.11 | 20.32 | 9.36 | 1.15x | tlx_faster | matrix_tdm_flat, S6, W1, BN128 |
| mha128_p128_l8192 | 27.95 | 73.72 | 28.03 | 1.00x | within_3_percent | matrix_tdm_flat, S8, W1, BN128 |
| mha128_p128_l32768 | 85.39 | 275.73 | 90.79 | 1.06x | tlx_faster | matrix_tdm_flat, S32, W1, BN128 |
| mha128_w37 | 3.13 | 2.94 | — | failed | no_eligible_aiter | matrix_tdm_flat, S1, W1, BN64 |
| mha128_w128 | 3.77 | 3.93 | 3.11 | 0.83x | aiter_faster | matrix_tdm_flat, S1, W1, BN64 |
| mha128_unequal | 8.31 | 15.33 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W1, BN64 |
| mha128_fp8_unit | 8.12 | 14.83 | — | failed | no_eligible_aiter | matrix_tdm_flat, S12, W1, BN64 |
| mha128_fp8_unequal | 8.12 | 14.87 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S12, W1, BN64 |
| mha128_t8 | 23.02 | 23.04 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S6, W4, BN64 |
| mha128_t32 | 28.01 | 28.00 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S6, W4, BN64 |
| mha128_t128 | 52.43 | 52.53 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S3, W4, BN64 |
| mha128_t512 | 127.62 | 128.95 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_t2048 | 463.57 | 469.21 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
| mha128_t8192 | 1777.23 | 1799.46 | — | not_applicable | no_eligible_aiter | matrix_tdm_flat, S1, W4, BN64 |
