# TLX FP8 KV decode results

**November9 optimization:** [final report](OPTIMIZATION_2026-11-09.md) · [matrix chart](charts/2026-11-09_step-15_final_matrix.svg) · [fresh reference chart](charts/2026-11-09_step-16_framework_repeat.svg). Same-run frozen→current TLX: GQA D64 3.87→3.86µs; MHA D64 10.24→8.50µs; MHA D128 15.42→8.45µs versus AITER12.20µs. Strict739 pass/8 skip;80/80 benchmark cases pass. Among40 eligible AITER controls,31 favor TLX beyond3%,4 near parity,5 favor AITER. No original-TLX regressions beyond3%. Earlier records below remain historical.

**Fresh November9 three-length repeat:** [report](REPEAT_2026-11-09.md) · [chart](charts/2026-11-09_step-10_repeat_lengths.svg). Fixed November5 sources/configurations; shared and unequal-head scale controls. Earlier dated results below remain preserved.

**Updated framework comparison:** [SGLang/vLLM report](FRAMEWORK_COMPARISON.md) · [step09 chart](charts/2026-11-05_step-09_framework_decode.svg). It adds stock/tuned vLLM3D, descriptors, SGLang/Lean and deduplicated AITER-backed controls:18 cases,198 passing selected verification invocations. The original `vllm_unified` numbers below used2D without split scratch and do not establish an advantage over the best decode path. Native per-head base GQA is now TLX3.91µs versus tuned vLLM5.48µs (~1.40x), rather than the earlier2D10.36x. AITER wins the D128 shared-scale control12.23µs versus TLX15.48µs. Earlier raw results are preserved as history.

40/40 synthetic benchmark cases passed the unchanged FP32-oracle gate (`atol=5e-4, rtol=1e-2`).

Correctness: 255 passed, 8 skipped, 0 failures, 0 errors across the multiseed suite and additional D128 high-split cases. Includes three seeds for poisoned-tail/peaked/workspace cases.
The format is plain E4M3FN with per-head scales. Exact model head mapping, T/mask semantics, SWA window and output projection remain undisclosed.

[Comparison chart](charts/2026-11-05_step-06_final_comparison.svg) · [CSV](results/2026-11-05_step-06_final_comparison.csv) · [Raw rows](results/2026-11-05_step-06_final_comparison.json) · [Reproduction](REPRODUCE.md) · [Experiment log](FP8_KV_DECODE_ATTN.md)

Historical step06 snapshot (superseded for strongest SGLang/vLLM claims by step09): base scalar controls use shared head scales so all BF16-Q reference paths can participate. Native unequal-per-head comparisons use vLLM unified. Full-FP8 unit-scale paths from AITER/TokenSpeed/vLLM fail this project's tighter numerical gate; their diagnostic latency and error remain in raw JSON, not in the eligible-reference ranking. The old SGLang/vLLM decode controls swept4/16/64. Step09 expands this to4..256 and finds an eligible SGLang full-FP8 GQA control9.34µs versus TLX3.73µs; the old “No eligible path” GQA unit row is superseded.

Paired order reversals (three runs per implementation, medians of graph batch means):

| Case | TLX µs | AITER µs | Ratio |
| --- | ---: | ---: | ---: |
| gqa | 3.93 | 4.56 | 1.161x |
| mha | 10.29 | 13.81 | 1.342x |
| long | 17.48 | 17.82 | 1.019x |
| window | 2.91 | 2.92 | 1.003x |

The long-context and window ratios are small margins. Page128 and other near-equal results should be treated as competitive rather than universal wins. Clocks are unlocked; these are standalone kernel paths on one96-CU DPX agent, with repeated same-input graph timing. Reported p95 is across graph batch means. Packing and compilation are excluded. Caller timings are separate.

| Case | Scale control | TLX µs | Best eligible external path | External µs | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| [gqa_shared](results/2026-11-05_optimization/latest/gqa_shared.json) | shared | 4.19 | aiter_unified | 4.90 | 1.17x |
| [gqa_perhead](results/2026-11-05_optimization/latest/gqa_perhead.json) | per-head | 4.21 | vllm_unified | 43.62 | 10.36x |
| [gqa_fp8_unit](results/2026-11-05_optimization/latest/gqa_fp8_unit.json) | unit | 4.05 | No eligible path | — | — |
| [gqa_fp8_perhead](results/2026-11-05_optimization/latest/gqa_fp8_perhead.json) | per-head | 4.02 | No eligible path | — | — |
| [mha_shared](results/2026-11-05_optimization/latest/mha_shared.json) | shared | 10.51 | aiter_unified | 14.30 | 1.36x |
| [mha_perhead](results/2026-11-05_optimization/latest/mha_perhead.json) | per-head | 10.52 | vllm_unified | 62.55 | 5.95x |
| [mha_fp8_unit](results/2026-11-05_optimization/latest/mha_fp8_unit.json) | unit | 10.00 | No eligible path | — | — |
| [mha_fp8_perhead](results/2026-11-05_optimization/latest/mha_fp8_perhead.json) | per-head | 10.02 | No eligible path | — | — |
| [folded_shared_t8](results/2026-11-05_optimization/latest/folded_shared_t8.json) | per-head | 5.36 | vllm_unified | 43.83 | 8.18x |
| [folded_shared_t32](results/2026-11-05_optimization/latest/folded_shared_t32.json) | per-head | 8.06 | vllm_unified | 45.31 | 5.62x |
| [folded_shared_t128](results/2026-11-05_optimization/latest/folded_shared_t128.json) | per-head | 11.36 | vllm_unified | 63.09 | 5.55x |
| [folded_shared_t512](results/2026-11-05_optimization/latest/folded_shared_t512.json) | per-head | 24.96 | vllm_unified | 157.66 | 6.32x |
| [folded_shared_t2048](results/2026-11-05_optimization/latest/folded_shared_t2048.json) | per-head | 69.32 | vllm_unified | 543.46 | 7.84x |
| [folded_shared_t8192](results/2026-11-05_optimization/latest/folded_shared_t8192.json) | per-head | 222.38 | vllm_unified | 2087.01 | 9.38x |
| [dense_shared_t8](results/2026-11-05_optimization/latest/dense_shared_t8.json) | per-head | 19.25 | vllm_unified | 63.87 | 3.32x |
| [dense_shared_t32](results/2026-11-05_optimization/latest/dense_shared_t32.json) | per-head | 21.41 | vllm_unified | 110.55 | 5.16x |
| [dense_shared_t128](results/2026-11-05_optimization/latest/dense_shared_t128.json) | per-head | 50.17 | vllm_unified | 305.04 | 6.08x |
| [dense_shared_t512](results/2026-11-05_optimization/latest/dense_shared_t512.json) | per-head | 117.58 | vllm_unified | 1082.44 | 9.21x |
| [dense_shared_t2048](results/2026-11-05_optimization/latest/dense_shared_t2048.json) | per-head | 429.00 | vllm_unified | 4200.74 | 9.79x |
| [dense_shared_t8192](results/2026-11-05_optimization/latest/dense_shared_t8192.json) | per-head | 1676.67 | vllm_unified | 16672.27 | 9.94x |
| [dense_shared_l1024](results/2026-11-05_optimization/latest/dense_shared_l1024.json) | shared | 7.11 | aiter_unified | 11.93 | 1.68x |
| [dense_shared_l8192](results/2026-11-05_optimization/latest/dense_shared_l8192.json) | shared | 27.32 | aiter_unified | 36.23 | 1.33x |
| [dense_shared_l32768](results/2026-11-05_optimization/latest/dense_shared_l32768.json) | shared | 90.68 | aiter_unified | 106.74 | 1.18x |
| [dense_shared_l131072](results/2026-11-05_optimization/latest/dense_shared_l131072.json) | shared | 294.00 | aiter_unified | 378.50 | 1.29x |
| [dense_shared_l262144](results/2026-11-05_optimization/latest/dense_shared_l262144.json) | shared | 568.68 | aiter_unified | 745.83 | 1.31x |
| [dense_l1024](results/2026-11-05_optimization/latest/dense_l1024.json) | per-head | 7.10 | vllm_unified | 28.95 | 4.07x |
| [dense_l8192](results/2026-11-05_optimization/latest/dense_l8192.json) | per-head | 28.74 | vllm_unified | 257.99 | 8.98x |
| [dense_l32768](results/2026-11-05_optimization/latest/dense_l32768.json) | per-head | 89.43 | vllm_unified | 1019.65 | 11.40x |
| [dense_l131072](results/2026-11-05_optimization/latest/dense_l131072.json) | per-head | 293.40 | vllm_unified | 4097.47 | 13.97x |
| [dense_l262144](results/2026-11-05_optimization/latest/dense_l262144.json) | per-head | 569.73 | vllm_unified | 8399.24 | 14.74x |
| [gqa_shared_l8192](results/2026-11-05_optimization/latest/gqa_shared_l8192.json) | shared | 4.71 | aiter_unified | 5.24 | 1.11x |
| [gqa_shared_l32768](results/2026-11-05_optimization/latest/gqa_shared_l32768.json) | shared | 7.41 | aiter_unified | 11.93 | 1.61x |
| [gqa_shared_l131072](results/2026-11-05_optimization/latest/gqa_shared_l131072.json) | shared | 11.20 | aiter_unified | 14.63 | 1.31x |
| [gqa_shared_l262144](results/2026-11-05_optimization/latest/gqa_shared_l262144.json) | shared | 17.75 | aiter_unified | 18.18 | 1.02x |
| [gqa_swa37_perhead](results/2026-11-05_optimization/latest/gqa_swa37_perhead.json) | per-head | 2.58 | vllm_unified | 3.60 | 1.40x |
| [gqa_swa37_shared](results/2026-11-05_optimization/latest/gqa_swa37_shared.json) | shared | 2.59 | No eligible path | — | — |
| [gqa_swa128_perhead](results/2026-11-05_optimization/latest/gqa_swa128_perhead.json) | per-head | 3.18 | vllm_unified | 4.99 | 1.57x |
| [gqa_swa128_shared](results/2026-11-05_optimization/latest/gqa_swa128_shared.json) | shared | 3.21 | aiter_unified | 3.23 | 1.01x |
| [d128_dense_perhead](results/2026-11-05_optimization/latest/d128_dense_perhead.json) | per-head | 15.56 | vllm_unified | 53.73 | 3.45x |
| [page128_gqa_shared](results/2026-11-05_optimization/latest/page128_gqa_shared.json) | shared | 4.55 | aiter_unified | 4.54 | 1.00x |

Retained changes: grouped WMMA, packed-page TDM double buffering, full-page mask peeling, FP16-scaled probabilities with FP32 softmax, tuned split/warp policy, direct single-split output, reusable workspace and bounded FP16 raw-V partials at high split counts.

Final ATT captures and analyzer reports are under `results/2026-11-05_optimization/att_final_{gqa,mha,long}*`. Baseline waits motivated the work; final long-context trace still shows layout-permutation and explicit-wait costs. ATT fractions describe selected instruction samples, not whole-kernel utilization. Requested PMC counters returned zero and cannot confirm utilization or bank conflicts.

Rejected attempts, compiler failures, earlier correctness failures, exact revisions and resource counts remain in the experiment ledger and snapshots. The production SGLang/vLLM dispatch has not been changed.

Upstream main revisions were audited and refreshed before this report. Original pinned measurements remain preserved. Current comparison pins:

- aiter: `796ff6a35588058516bcaf13129bd0f0b22fb9ae`
- vllm: `86fdabec0c82d0470c160aa4df7602ac22c207e6`
- sglang: `9b6c796151f836907fbf93265984006ecfd6a790`
- tokenspeed: `23791cea04c42d5b6c50980914ccca5a5c4eeb9e`

TokenSpeed MHA decode is unchanged in the audit. SGLang changes host core lookup; vLLM adds verify paths and tuning for other RDNA architectures. New AITER block-scaled MXFP8 paths use a different scale contract; refreshed unified/dense E4M3 paths were rerun. See upstream_diff_audit.json and the saved diffs in the result directory.

Optional SGLang Lean Attention controls (wrapper metadata/lock-reset GPU work included):

| Case | Status under project gate | GPU invocation µs |
| --- | --- | ---: |
| [latest_lean_gqa](results/2026-11-05_optimization/latest_lean_gqa.json) | failed_accuracy | 27.82 |
| [latest_lean_gqa_long](results/2026-11-05_optimization/latest_lean_gqa_long.json) | pass | 109.84 |
| [latest_lean_mha](results/2026-11-05_optimization/latest_lean_mha.json) | failed_accuracy | 45.87 |
| [latest_lean_mha_long](results/2026-11-05_optimization/latest_lean_mha_long.json) | pass | 3849.83 |
