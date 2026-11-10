# FP8 KV decode: references, comparisons, and experiments

Dates use **America/Los_Angeles**. Started 2026-10-27. Maintain this file as work proceeds, including unsuccessful attempts.

## Target workload matrix — supplied again 2026-11-05

The screenshot supplied on 2026-11-05 is the current target specification. The
[machine-readable matrix](results/2026-11-05_step-02_workloads.json) preserves the
reported shapes and unresolved fields. It expands to 15 T/L combinations across
seven cases, before any additional dtype/head/page/window variants.

| # | Case | T | KV length L | Q | K/V | Reported O | Precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Base parity | 1 | 2304 | [1,8192] | [2304,8192] | [1,1024] x [1024,8192] | BF16 Q + FP8 KV |
| 2 | Full FP8 | 1 | 2304 | [1,8192] | [2304,8192] | Same O | FP8 QKV |
| 3 | T-sweep | 8 / 32 / 128 / 512 / 2048 / 8192 | 2304 | [T,8192] | [2304,8192] | [T,1024] x [1024,8192] | FP8 KV |
| 4 | KV-sweep | 1 | 1K / 8K / 32K | [1,8192] | [L,8192] | Same O | FP8 KV |
| 5 | Long context | 1 | 128K / 256K | [1,8192] | [L,8192] | Same O | FP8 KV |
| 6 | GQA folded | 1 | 2304 | [1,16,64] equivalent | Paged K/V | Same O | FP8 KV |
| 7 | SWA | 1 | 2304 | Same as #1 | Same as #1 | Same O | FP8 KV |

The prior physical-cache agreement remains K `[block,head,dim/x,page,x]` and
V `[block,head,page/x,dim,x]`, with x=16 for plain E4M3FN. That defines packing,
but does not determine head counts, page size, masking, or model projection.

The user confirmed on 2026-11-05 that dense/folded head mappings,
T/batch/query/mask semantics, and SWA window size are unknown. These are not
prerequisites for exploratory kernel work: use explicit configurable synthetic
cases and retain their assumptions in results. The O factorization needs a contract and
weights before an output projection can be included. Rows specifying only FP8
KV do not explicitly fix Q's dtype; BF16 Q is a provisional inherited default.
K-labelled lengths are preserved as supplied; the existing tests use binary K
(1024 tokens) for numeric length comparisons. These are recorded assumptions,
not additional facts supplied by the screenshot.

The earlier 27 passing checks remain **component correctness evidence**, not
acceptance of all target rows. In particular, the earlier T-sweep used
Hq=16/Hkv=2/D=64 (Q width 1024); long-context tests used Hq=2/Hkv=1/D=64.
At step02 the 1K point had not been tested explicitly; the later40-case matrix now covers it and the full-width T-sweep. See the historical
[target-versus-coverage chart](charts/2026-11-05_step-02_workload_coverage.svg).

### Working decision following Stan/Hongtao's guidance

Hongtao described the supplied table as a dense view and recommended using the
physical cache layout from the 5D tutorial because detailed layouts are not
disclosed. Adopt that tutorial as the current implementation contract:

```text
K: [blocks, kv_heads, D/16, page_size, 16]
V: [blocks, kv_heads, page_size/16, D, 16]
Q: [batch * query_length, query_heads, D]
```

The V ordering is taken from the actual tutorial, rather than assuming K and V
have identical axes. The current FP8 prototype already implements these layouts
and head-indexed K/V scales. No kernel rewrite is needed for this guidance.

Use the current D64/Hq=Hkv=128 configuration as one synthetic interpretation of
dense width 8192; D128/Hq=Hkv=64 is another possible interpretation for a later
experiment. Keep the folded Q=[1,16,64] tests separate, with KV-head count an
explicit experiment parameter. For T, label batched one-query decode separately
from multiple queries over one shared cache; record the actual mask. SWA window
size is also an experiment parameter, not an inferred target value.

These are exploration choices, not a claim about the undisclosed model. The
[working-assumption record](results/2026-11-05_working_assumptions.json) preserves
the decision. Keep the earlier coverage chart as the historical comparison;
save new measured charts when experiments are actually run.

### Confirmed per-head scaling requirement — 2026-11-05

The user explicitly reconfirmed **per-head scales**. This is fixed while head
counts, query interpretation and window size remain configurable:

```text
K_real[token, kv_head, d] = K_fp8[token, kv_head, d] * k_scale[kv_head]
V_real[token, kv_head, d] = V_fp8[token, kv_head, d] * v_scale[kv_head]
```

K and V have separate scale vectors, constant over tokens/pages within each KV
head. GQA query heads sharing a KV head use that head's K/V scales. The current
prototype uses FP32 scale vectors and already loads `KScale + kv_head` and
`VScale + kv_head`; its FP8-Q path additionally has one Q scale per query head.
The earlier tests use distinct non-unit head scales.

Primary target comparisons must preserve arbitrary per-head scaling. A
scalar-only kernel can be used as a restricted shared-scale control, or through
a separately validated adapter whose costs are reported. It does not establish
support for the full target contract. In particular, do not replace unequal
head scales with a single scalar merely to make a baseline runnable.

Recorded effect: clarified acceptance and reference-comparison criteria; verified
the existing source matches. No kernel change or new timing/correctness run.

## Current result — 2026-11-09

- **Fresh three-length repeat:** [November9 report](REPEAT_2026-11-09.md), [step10 chart](charts/2026-11-09_step-10_repeat_lengths.svg), [CSV](results/2026-11-09_step-10_repeat_lengths.csv). Fixed November5 sources/configurations, folded-GQA16/2,D64,page64,BF16 Q. Shared controls L2304/8192/32768: TLX3.92/4.47/7.15µs,AITER4.55/4.90/11.71µs,previously selected vLLM5.62/6.59/10.98µs. Unequal-head TLX3.92/4.43/7.16µs versus vLLM5.42/6.70/12.34µs.72 verification runs pass. Original pasted-table metadata was not recovered; no exact historical reproduction/speedup claim.

- **Latest comparison:** [SGLang/vLLM report](FRAMEWORK_COMPARISON.md), [step09 chart](charts/2026-11-05_step-09_framework_decode.svg), [CSV](results/2026-11-05_step-09_framework_decode.csv).18 cases,66 selected configurations,198 passing verification invocations. Corrects weak vLLM2D-only claims: native per-head base GQA3.91µs TLX versus5.48µs tuned vLLM (~1.40x). AITER-backed D128 shared control wins12.23µs versus TLX15.48µs; long GQA and SWA remain near parity. Full-FP8 unit GQA now has an eligible SGLang control9.34µs versus TLX3.73µs.

- [Kernel/policy](../../third_party/tlx/tutorials/amd_pa_decode_fp8.py), [retained candidates](../../third_party/tlx/tutorials/amd_pa_decode_fp8_optimized.py), [contract/usage](../../third_party/tlx/tutorials/amd_pa_decode_fp8.rst).
- [Measured report](RESULTS.md), [current reference paths](REFERENCES.md), [reproduction](REPRODUCE.md), [source/runtime manifest](results/2026-11-05_optimization/manifest.json).
- **255 correctness checks passed;8 layout skips;40/40 synthetic benchmark cases pass the unchanged gate.** Covers full-width T=8..8192, dense L=1K..256K, folded GQA, both Q dtypes and assumed SWA37/128.
- Fresh-reference order reversals: base GQA TLX3.93us versus AITER4.56us (~1.16x); base MHA10.29 versus13.81us (~1.34x), on shared-scale restricted controls. Native unequal-per-head comparisons use vLLM unified. Long grouped256K, SWA128 and page128 are near parity; no universal win is claimed.
- Retained changes: grouped WMMA, TLX packed-page TDM double buffering, mask peeling, scaled FP16 probabilities, FP32 softmax/accumulation, tuned splits/warps and direct one-split output. At S>=256 bounded raw-V partial means use FP16; V head scaling moves to the final FP32 merge.
- Scope stays plain E4M3FN/per-head scales, with BF16 output before projection. Frozen baseline GPU bodies are AST-identical to the saved prototype. rocprof/ATT evidence is saved; all-zero PMC counters are unusable for utilization claims.

## Saved comparisons and evidence

| Major step | Comparison | Data / evidence | Interpretation |
| --- | --- | --- | --- |
| 00: reference selection | [Capability chart](charts/2026-10-27_step-00_capabilities.svg) | [Pinned capability data](results/2026-10-27_step-00_capabilities.json) | Source inspection; external kernels have not been executed here. |
| 01: working prototype | [Correctness chart](charts/2026-10-27_step-01_correctness.svg) | [Result data](results/2026-10-27_step-01_correctness.csv), [17-case log](results/2026-10-27_initial17_pytest.txt), [27-case log](results/2026-10-27_expanded27_pytest.txt), [source/environment manifest](results/2026-10-27_step-01_manifest.json) | Different-sized suites; counts show coverage growth, not a speed comparison. |
| 02: target workload reconciliation | [Target versus existing coverage](charts/2026-11-05_step-02_workload_coverage.svg) | [Target matrix and coverage data](results/2026-11-05_step-02_workloads.json) | Static mapping to the earlier suite; exact target acceptance is pending clarification. No new GPU run. |
| 03: bottleneck forecast | [Hypotheses and optimization directions](charts/2026-11-05_step-03_bottleneck_forecast.svg) | [Forecast data](results/2026-11-05_step-03_bottleneck_forecast.json) | Code-based observations and predictions only; no profiling or speedup measurement. |

| 04: aligned controls | [Measured baselines](charts/2026-11-05_step-04_aligned_baselines.svg) | [Data](results/2026-11-05_step-04_aligned_baselines.json) | Same compiler; early controls. |
| 05: packed movement | [Experiment comparison](charts/2026-11-05_step-05_movement_experiments.svg) | [Data](results/2026-11-05_step-05_movement_experiments.json) | Slower/failed candidates retained. |
| 06: historical40 cases | [Final comparison](charts/2026-11-05_step-06_final_comparison.svg) | [CSV](results/2026-11-05_step-06_final_comparison.csv), [raw rows](results/2026-11-05_step-06_final_comparison.json), [report](RESULTS.md) | Historical2D vLLM snapshot; strongest decode claims superseded by step09. |
| 07: ATT evidence | [Sampled stall mix](charts/2026-11-05_step-07_att_mix.svg) | [Data](results/2026-11-05_step-07_att_mix.json), analyzer reports under results/2026-11-05_optimization/ | Selected instruction samples, not utilization/speedup. |
| 08: full-FP8 gate | [Latency/error eligibility](charts/2026-11-05_step-08_fp8_accuracy_latency.svg) | [Data](results/2026-11-05_step-08_fp8_accuracy_latency.json) | Historical snapshot; step09 adds a passing SGLang GQA control. |
| 09: strongest tested framework paths | [Comparison chart](charts/2026-11-05_step-09_framework_decode.svg) | [CSV](results/2026-11-05_step-09_framework_decode.csv), [raw rows](results/2026-11-05_step-09_framework_decode.json), [report](FRAMEWORK_COMPARISON.md) | Stock/tuned3D, TD, Lean and AITER routing; selected paths repeated independently. |

| 10: three-length rerun | [Fresh repeat chart](charts/2026-11-09_step-10_repeat_lengths.svg) | [CSV](results/2026-11-09_step-10_repeat_lengths.csv), [raw data](results/2026-11-09_step-10_repeat_lengths.json), [report](REPEAT_2026-11-09.md) | Fixed pins/configs; shared and unequal-head controls; first busy-agent attempt excluded. |

Charts and their data are immutable dated snapshots. Create a new step/version when updating a comparison. The plotting script can regenerate these two snapshots with `python experiments/fp8-kv-decode/render_charts.py`.

The [legacy benchmark schema](results/benchmark_results.csv) remains a header-only historical template. Actual dated measurements, units, numerical gates, dispatch/configuration and source identifiers are in the result directories and the step06 CSV; estimated values are not substituted for measurements.

## Original reference-selection snapshot — 2026-10-27

These source-inspection notes predate the measurements and remain as history. [REFERENCES.md](REFERENCES.md) and [RESULTS.md](RESULTS.md) contain the refreshed, executed comparisons.

### Primary: AITER unified attention on gfx1250

Repository revision: `1dd826c42b38aa71adf058053b9c512a9b47d20f`.

- [Launcher: `unified_attention`](https://github.com/ROCm/aiter/blob/1dd826c42b38aa71adf058053b9c512a9b47d20f/aiter/ops/triton/attention/unified_attention.py#L228).
- [gfx1250 Gluon decode implementation](https://github.com/ROCm/aiter/blob/1dd826c42b38aa71adf058053b9c512a9b47d20f/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py#L1929).
- Why: closest architecture and paged-cache path; supports mixed BF16-Q/FP8-KV and full FP8, GQA, sliding windows, and split decode.
- Limits: descales are scalars. The gfx1250 Gluon 3D dispatch requires shuffled KV; the 2D fast path has different gates, including matching Q/KV dtypes. Record the actual selected kernel and any fallback, not just `backend="gluon"`.
- Comparison: shared non-unit per-tensor scales broadcast into our per-head vectors, plus unit scales. Repack outside timed kernel regions if layouts differ. These tests establish a shared workload, not arbitrary per-head support in AITER.

### Decode baseline: TokenSpeed gfx1250 paged MHA/GQA

Repository revision: `6fb2852d08ce20939a6a2351ff58e3e43ae25e8f`.

- [Launcher and restrictions](https://github.com/lightseekorg/tokenspeed/blob/6fb2852d08ce20939a6a2351ff58e3e43ae25e8f/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/mha/decode.py#L1159).
- Why: native gfx1250 WMMA decode, GQA, page sizes 64/128, head dimensions 64/128.
- Limits: Q/K/V must share a dtype; all explicit FP8 scales are rejected; q_len must be 1; sliding windows are rejected; cache interface is 4D.
- Comparison: full-FP8 QKV with unit scales, q_len=1, no SWA, matching dimensions/pages. Use identical FP8 values and equivalent page mappings. Do not describe this as a BF16-Q or per-head-scale comparison.

### Dense full-FP8 baseline: AITER MHA FP8

- [Launcher `mha_fwd`](https://github.com/ROCm/aiter/blob/1dd826c42b38aa71adf058053b9c512a9b47d20f/aiter/ops/triton/attention/mha_fp8.py#L56).
- [gfx1250 subtiled kernel](https://github.com/ROCm/aiter/blob/1dd826c42b38aa71adf058053b9c512a9b47d20f/aiter/ops/triton/_gluon_kernels/gfx1250/attention/mha_fp8.py#L138).
- Why: same E4M3 input representation, gfx1250 matrix instructions, GQA, unequal Q/K lengths, and causal/noncausal dense attention. Useful for the full-FP8 case and T-sweep.
- Limits: all Q/K/V must be E4M3FN; scales are scalar; dense layout is `[B,H,S,D]`. This is a dense MHA kernel, not a specialized paged decode path.
- Comparison: full-FP8 and shared scales; prepack dense and paged representations from the same quantized values. Report dense-vs-paged access differences explicitly.

### Scaling reference: vLLM Triton unified attention

Repository revision: `0855defde8ea382781ab4211c4f67d17f6b96f50`.

- [Unified attention](https://github.com/vllm-project/vllm/blob/0855defde8ea382781ab4211c4f67d17f6b96f50/vllm/v1/attention/ops/triton_unified_attention.py#L828).
- Why: FP8 per-token-per-KV-head scale caches, GQA, paged decode and SWA. Scale indexing is useful for checking our per-head contract.
- Comparison: for BF16 Q, broadcast each per-head K/V scale across cache slots in its scale-cache layout, preserving the same quantized payload. Record the extra scale-cache traffic and any materialization. This adapter is planned, not validated. Full-FP8 Q has a separate scale path; do not assume identical support there.

### Secondary split-KV references: SGLang / vLLM decode

- SGLang revision `010eb729bea2b7ef2bcd572a72c171f75ad900fb`: [`decode_attention.py`](https://github.com/sgl-project/sglang/blob/010eb729bea2b7ef2bcd572a72c171f75ad900fb/python/sglang/kernels/ops/attention/decode_attention.py#L1191). Useful production split/GQA baseline with scalar descales. Its grouped path contains gfx1250-specific arithmetic workarounds; preserve and report the selected path. FP8-Q query-scale parity has not been established.
- vLLM revision `0855defde8ea382781ab4211c4f67d17f6b96f50`: [`triton_decode_attention.py`](https://github.com/vllm-project/vllm/blob/0855defde8ea382781ab4211c4f67d17f6b96f50/vllm/v1/attention/ops/triton_decode_attention.py#L755). Simple scalar-scaled FP8-KV split-decode reference. It is related to SGLang's implementation, not an independent algorithm. BF16 Q is the initial comparison target.

### Other useful, constrained references

- AITER [`pa_decode.py`](https://github.com/ROCm/aiter/blob/1dd826c42b38aa71adf058053b9c512a9b47d20f/aiter/ops/triton/attention/pa_decode.py#L36): scalar and per-token-head scale modes. Its [per-token quantization test section](https://github.com/ROCm/aiter/blob/1dd826c42b38aa71adf058053b9c512a9b47d20f/op_tests/triton_tests/attention/test_pa_decode.py#L368) has a GQA-failing TODO. Treat as an implementation reference until validated.
- AITER [`pa_decode_bf16_asm`](https://github.com/ROCm/aiter/blob/1dd826c42b38aa71adf058053b9c512a9b47d20f/aiter/ops/attention.py#L413): despite its name, documented FP8 QKV and BF16 output. Documented specialization D64/page256/GQA8 with scalar descales; compare only on that supported configuration after verifying the dispatched assembly.
- TokenSpeed MLA/DeepSeek-specific kernels and SGLang's NVIDIA SM120 attention are useful design references but do not match the generic gfx1250 MHA/GQA contract.
- [User-supplied MI450X-MC overview](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1693667195): read version 14, last updated 2026-09-28. Local runtime reported 96 CUs and 144 GiB per DPX agent; the document's 168-active-CUs/package figure remains an unresolved discrepancy. Size launch experiments from observed device properties.

## Comparison rules

1. Freeze source revision/hash, compiler/runtime, GPU UUID, partition, clocks/power conditions, and kernel dispatch. External references are currently source-reviewed only.
2. Use identical quantized Q/K/V payloads, scale semantics, head mapping, lengths, page mappings, causal alignment, SWA boundaries, output dtype, and output-projection boundary.
3. Arbitrary per-head K/V scales are required for target comparisons. Label shared/unit-scale comparisons to scalar-only references as restricted controls; they do not establish full-contract support. Use validated adapters when applicable and report their costs. Mark unsupported cases N/A with a reason rather than inventing times.
4. Resolve T as batch versus query length explicitly. Current T-sweep checks use noncausal query rows over one shared cache. The 8192-wide test uses Hq=Hkv=128, D=64; this is an example interpretation, not a confirmed model layout. Long-context checks currently use fewer heads, not the full 8192-wide workload.
5. Establish numerical correctness against FP32 attention over the actual quantized inputs first. Record max absolute error, RMS error and acceptance threshold in future runs, in addition to pass/fail. Also distinguish error against quantized-input reference from error against original BF16 values.
6. Time kernel-only and caller-visible execution separately. Kernel-only excludes data generation, quantization, repacking, allocations and compilation. Report preprocessing and allocation costs separately; the current launcher allocates its split workspace on every call.
7. Use warmup, repeated GPU-event measurements and interleaved baseline/candidate trials on the same agent. Record median/p95 latency in microseconds, sample counts, workspace bytes, and noise; speedup = baseline median / candidate median. Avoid deriving a win from one sample.
8. Keep read-only benchmark inputs fixed. Capture raw results and commands. Save a per-workload latency/speedup chart for each major measured step. Use N/A for incompatible semantics and leave unmeasured cells empty.

## Experiment ledger

Entries EXP-001 through EXP-004 were reconstructed from the session on 2026-10-27. Exact start times and failed-source snapshots were not retained. Failed-attempt details below come from observed tool output; no missing performance data is inferred.

### EXP-001 — 2026-10-27 — Async staging for both packed K and V

- **Status:** compilation failed; superseded.
- **Trying:** one CTA per query head/split; TLX async global-to-LDS copies for both 5D cache tiles; FP32 QK/PV reductions.
- **Why / reasoning:** establish the physical layout and per-head scale contract while using TLX's explicit data movement. Avoid relying on native FP8 dot numerics in the first correctness baseline.
- **Expected benefit:** asynchronous staging could provide a foundation for load/compute overlap. No speedup was predicted numerically.
- **Validation:** compile and run the first BF16-Q, FP8-KV L=2304 case on gfx1250.
- **Actual result:** compiler could not legalize V's `ttg.async_copy_global_to_local`; the packed-V load layout had contiguity 1 for byte elements. Arbitrary token tails prevent assuming a wider copy. No kernel result or timing was obtained.
- **Decision:** retain async K, use masked synchronous V load plus TLX local store/load. This is a workaround for the observed lowering, not proof that all possible async V implementations are unsupported.
- **Evidence:** compiler diagnostic recorded in session; initial failed source/log not retained independently.

### EXP-002 — 2026-10-27 — Synchronous V staging

- **Status:** compilation failed; corrected in EXP-003.
- **Trying:** replace V async copies with masked `tl.load`, then TLX LDS staging.
- **Why / reasoning:** preserve arbitrary-tail masking and the physical 5D cache contract without requiring sub-vector async copies.
- **Expected benefit:** make the preliminary kernel compile; no performance improvement claim.
- **Validation:** same initial correctness case as EXP-001.
- **Actual result:** `tl.load(..., other=0)` attempted an unsupported int32-to-FP8 cast in the installed runtime. Compilation stopped; no output or timing was obtained.
- **Decision:** use floating zero (`other=0.0`) for FP8 Q/V masked loads.
- **Evidence:** compilation diagnostic recorded in session; failed source/log not retained independently.

### EXP-003 — 2026-10-27 — Working mixed-precision / per-head baseline

- **Status:** retained; correctness passed.
- **Trying:** async K + synchronous V staging, floating masked-load defaults, per-head descales, FP32 reductions, split merge and window masks.
- **Why / reasoning:** produce an executable reference with clear scale placement and high-precision accumulation before optimizing matrix instructions or data reuse.
- **Expected benefit:** correct BF16-Q/FP8-KV and full-FP8 outputs, including GQA and cache tails. This is a correctness milestone, not a measured optimization.
- **Validation:** 17-case embedded suite; compare against independently gathered dense FP32 attention using the actual quantized payload. Unequal non-unit head scales, shuffled pages and NaN-filled unused tails exercise indexing/masking.
- **Actual result:** **17 passed in 4.83 s** of total pytest wall time. Tested empty/partial contexts, SWA, GQA, multi-query masking, split reduction and 128K/256K contexts. The 4.83 s includes host/test/compilation work and is not kernel latency.
- **Decision:** retain as initial working design; expand shape coverage next.
- **Evidence:** [test log](results/2026-10-27_initial17_pytest.txt). Its earlier source version was not saved; do not attribute this log to the later 27-case source hash.

### EXP-004 — 2026-10-27 — Expand coverage and freeze the prototype

- **Status:** retained; correctness passed.
- **Trying:** add an 8192-wide dense-view case, all supplied T-sweep values on the folded configuration, block sizes 16/64, and reject noncontiguous physical caches.
- **Why / reasoning:** catch shape, page, and compiler-specialization errors before comparing or optimizing. Keep the unresolved workload interpretations explicit.
- **Expected benefit:** stronger correctness evidence; no intended kernel speedup.
- **Validation command:** `python -m pytest -s --tb=short third_party/tlx/tutorials/amd_pa_decode_fp8.py` with `/tmp/tlx-runtime` providing the isolated TLX runtime inside the test container. Tolerance `atol=5e-4, rtol=1e-2` against FP32 attention on quantized inputs; BF16 output required.
- **Actual result:** **27 passed in 6.33 s** total pytest wall time; Ruff, Python byte compilation and whitespace checks passed. No numerical error maxima were saved by this suite. No performance measurements were taken. The longer total duration is not evidence of a kernel regression because the suite changed.
- **Decision:** freeze as the initial comparison source; leave production dispatch unchanged.
- **Evidence:** [27-case test log](results/2026-10-27_expanded27_pytest.txt), [manifest](results/2026-10-27_step-01_manifest.json), [recoverable source patch](snapshots/2026-10-27_step-01_kernel.patch), [correctness chart](charts/2026-10-27_step-01_correctness.svg).

### EXP-005 — 2026-10-27 — Establish durable comparison records

- **Status:** completed; documentation/artifacts only.
- **Trying:** pin reference revisions, rank candidates, preserve baseline evidence and source, add reusable comparison-chart generation and a future measurement schema.
- **Why / reasoning:** prevent lost negative results, mismatched comparisons and unsupported speed claims as implementations evolve.
- **Expected benefit:** traceable experiments and reproducible comparison decisions; no kernel performance effect.
- **Actual result:** work log, capability/correctness data and charts, source/environment manifest, historical test logs, baseline patch, and benchmark CSV schema saved. Chart generation, snapshot verification and documentation links checked. No benchmark run.
- **Decision:** use this ledger and dated artifacts for subsequent work. Standing instructions are recorded in the repository's AGENTS.md.

### EXP-006 — 2026-11-05 — Reconcile the target shapes with existing evidence

- **Status:** completed; workload/documentation change only.
- **Trying:** preserve the supplied screenshot as the canonical seven-case matrix and compare its 15 T/L combinations to the current test parameters.
- **Why / reasoning:** component tests at smaller head counts or provisional masking must not be reported as full validation of the requested workload. The screenshot clarifies the table formatting but does not resolve the underlying head or mask contracts.
- **Expected benefit:** accurate acceptance criteria and explicit coverage gaps before timing or optimization work.
- **Validation/comparison plan:** inspect the actual parameter decorators without importing GPU modules, check the kernel against its frozen source hash, validate all seven cases and 15 expanded combinations, render a dated coverage chart, and check local documentation links.
- **Actual result/effect:** recorded the full matrix and saved the coverage chart/data. The 1K point is absent from prior tests; the T-sweep and long contexts used reduced widths; head/mask/window details remain pending. No kernel change, GPU execution, or new performance result. Updated AGENTS.md to follow the existing renamed log, FP8_KV_DECODE_ATTN.md.
- **Decision:** keep the established tests as component evidence; use explicit resolved head, query, mask and projection metadata before declaring any target row accepted.
- **Artifacts:** [matrix/data](results/2026-11-05_step-02_workloads.json), [coverage chart](charts/2026-11-05_step-02_workload_coverage.svg).

### EXP-007 — 2026-11-05 — Adopt the tutorial layout despite undisclosed metadata

- **Status:** completed; implementation-contract decision only.
- **Trying:** use the 5D tutorial as Hongtao advised while retaining the unknown model details as explicit parameters.
- **Why / reasoning:** the dense table does not describe physical storage, and the user does not have the hidden head/mask/window details. They should not block synthetic correctness or performance investigations.
- **Expected benefit:** continue useful kernel work with reproducible assumptions and avoid repeated requests for unavailable information.
- **Validation/comparison:** inspect the tutorial's actual K/V allocator and the FP8 launcher's shape checks; verify the existing kernel matches the chosen layouts; save the user clarification separately from inferred settings.
- **Actual result/effect:** confirmed K and V use the documented, distinct 5D orderings; the prototype already matches them. Saved the working-assumption record and persistent repository guidance. No kernel modification, new GPU run, or performance measurement; the existing charts and historical workload snapshot remain intact.
- **Decision:** proceed with declared synthetic configurations and compare references only on matching semantics. Exact model acceptance remains unknown rather than a prerequisite for exploration.
- **Artifacts:** [working assumptions](results/2026-11-05_working_assumptions.json), [historical coverage chart](charts/2026-11-05_step-02_workload_coverage.svg).

## Bottleneck forecast — 2026-11-05

These are hypotheses, not a measured bottleneck ranking. No ATT trace, PMC pass,
kernel latency or speedup result has been collected. Hardware limits must be
distinguished from the correctness prototype's current overheads.

1. **KV traffic at T=1, especially long contexts.** Under the reported FP8 width
   8192 for each of K and V, the payload for reading both caches once is
   `2 * L * 8192` bytes per request: 36 MiB at L=2304, 2 GiB at 128K and 4 GiB
   at 256K (binary K). These are calculated payload sizes, not measured HBM
   traffic; cache residency, GQA mapping and SWA change physical reads. Optimize
   coalescing/vector width and avoid extra copies; use split-KV to expose enough
   independent work to keep memory accesses in flight. A memory roofline needs
   measured effective bandwidth on the selected partition, not a package-wide
   specification.
2. **Repeated loads across query heads and query tokens.** The current grid is
   `(token, query_head, split)`, so GQA heads reload the same logical KV tiles.
   For grouping factor g=Hq/Hkv, grouping queries can eliminate up to g-fold
   redundant logical loads, but L2 reuse means an equivalent HBM or speedup
   factor is not guaranteed. Multiple query tokens over one shared cache should
   also be tiled together. Independent requests with different caches do not
   have this reuse. Tile groups must not cross KV-head boundaries because their
   scale pairs differ.
3. **FP8 storage with FP32 vector QK/PV.** The prototype converts cache tiles to
   FP32 and uses vector multiplies and reductions. For useful row counts (GQA
   groups or shared-cache query tiles), WMMA should improve arithmetic
   throughput and reduce reduction/communication overhead. A single MHA query
   may underfill matrix tiles; retain a vector candidate rather than assuming
   WMMA wins every shape. BF16 Q should have its own path; silently requantizing
   it to FP8 changes the parity contract.
4. **Packed V, LDS movement and insufficient overlap.** The loop has one K and
   one V buffer, synchronously stages V, then waits immediately for K before
   doing QK/PV. It does not prefetch the next tile across compute. Try legal
   vectorized packed loads or full-page staging, alternate register/shared
   layouts and then double buffering or TDM where supported. For the existing
   vector path, also test eliminating unnecessary V LDS store/load traffic.
   Wider tiles/prefetch can increase VGPR/LDS use and reduce occupancy, so
   measure spills, bank conflicts, wait states and active work together.
5. **Splits, merge and allocations.** Default splits depend only on total
   context length, not batch/head parallelism or the active window. The
   launcher always allocates partials and invokes a second merge kernel even
   when num_splits=1. Direct output for one split and reusable workspace are
   plausible low-complexity wins for short cases. Tune split count and tile
   sizes separately for sparse/saturated grids; more splits incur extra
   launches/workspace/reduction traffic and can lose on an already full GPU.
6. **SWA can launch work for masked-away splits.** The code partitions the full
   context before intersecting each split with the window. Correct masking
   avoids reading excluded KV, but empty splits still launch and write
   partials. Partition the active window, and size the grid from effective
   context, so reduced attention work also reduces scheduling/merge cost.

**Per-head scaling is not expected to be a leading bandwidth cost.** Each KV
head has two FP32 scalars. The current code already loads them outside the KV
loop, folds K's scale into QK logits and applies V's scale after normalization.
For FP8 Q, its query-head scale can likewise be folded into score scaling in a
matrix path. Preserve per-head indexing and do not materialize a dequantized
global KV cache. The main complication when adding FP8 WMMA is probability
representation/range in PV: FP32 softmax/accumulation alone does not remove
error from casting probabilities to BF16/FP8. Check output error and range
across uneven head scales, masked tails, peaked distributions, long contexts
and different split counts before accepting a speed improvement.

**Suggested order:** first measure matched per-head baselines; then test
single-split direct output/reusable workspace, grouped-query reuse, matrix
arithmetic with an accuracy gate, pipelining/layout changes, and shape-specific
split/SWA dispatch. The order after initial measurement should follow the
observed stalls and latency. Keep caller-visible and kernel-only timings
separate; use ATT/PMC to support a mechanism and unprofiled timings to establish
a performance effect.

References for the general mechanisms: [Flash-Decoding](https://princeton-nlp.github.io/flash-decoding/)
explains parallel KV chunks and their final reduction;
[FlashAttention-2](https://arxiv.org/abs/2307.08691) motivates reducing non-matmul
work and improving warp/block partitioning. Neither provides measured results
for this prototype or this host. Source locations and comparison signals are
saved in the [forecast data](results/2026-11-05_step-03_bottleneck_forecast.json).

### EXP-008 — 2026-11-05 — Start measured reference competition and ATT profiling

- **Status:** retained.
- **Trying:** establish reproducible source-pinned comparisons against AITER, vLLM, TokenSpeed and SGLang; profile the baseline with rocprofv3 and the installed ATT analyzer; optimize only with correctness and timing evidence.
- **Why / reasoning:** the existing prototype is a correctness baseline; the user now explicitly requests performance optimization and a win over the reference implementations.
- **Expected benefit:** identify actual limiting instructions and establish fair kernel-level and caller-level latency targets; no speedup assumed.
- **Contract:** retain the confirmed plain E4M3FN + per-head K/V scale contract despite the shorthand “mxfp8”; no silent conversion to block-scaled MXFP8. Scalar-scale references are restricted controls or clearly identified adapted variants.
- **Environment:** isolated container adkankar-tlx-opt-20261105; image rocm/sgl-dev:v0.5.23-rocm101-mi45x-20261101. CUDA_VISIBLE_DEVICES=0 after checking current occupancy: idle, no resident model; runtime identifies gfx1250, 96 CUs, 144 GiB, DPX/NPS2. Exact runtime/environment details will be saved with results. Existing source pins are retained initially.
- **Plan:** unprofiled timing and independent numerical checks first; ATT/PMC on one selected dispatch; staged optimization, reference comparisons and saved charts. A newer source-built TLX runtime is being prepared because fbtriton 3.7.4 lacks the newer gfx1250 TDM/WMMA and Gluon APIs used by the pinned fast references.
- **Actual result/effect:** Source-aligned benchmarks now execute all four source families. The final baseline/reference scope is pinned standalone GPU kernels with import adapters, not end-to-end framework serving. See holdout/suite_status.json and manifest.json. Initial mixed-runtime calibration is excluded from speedup claims. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-009 — 2026-11-05 — Vector layout, direct output and active-window partitioning

- **Status:** retained.
- **Trying:** register-resident vector candidate with TLX-pinned K/V load layouts, no explicit LDS staging, direct output for one split, and splits over the active causal/window interval. Keep the original kernel as variant=baseline.
- **Why / reasoning:** source inspection shows explicit K/V LDS round trips without data sharing, unconditional merge for a single split, and empty split scheduling under SWA.
- **Expected benefit:** reduce memory instructions, synchronization, and short-case launch overhead while retaining FP32 reductions and identical per-head scale semantics. No speedup assumed.
- **Validation:** original quantized-input FP32 oracle and adversarial tail/window cases; graph GPU timing and caller timing. Compare matching source controls and preserve failed candidates.
- **Actual result/effect:** Vector layouts required release_layout at the score-to-PV boundary to compile. Unequal-head-scale FP32 gates pass. At the initial matched BN32/S4 dense control vector26.07us versus baseline43.94us; later flat matrix is faster. Register-only vector/page variants remain explicit controls. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-010 — 2026-11-05 — Grouped-query WMMA candidate

- **Status:** retained.
- **Trying:** tile query rows within each KV-head group, reuse packed cache tiles, perform QK/PV with WMMA, and retain FP32 softmax and accumulators. Separate BF16-WMMA, FP8-QK-at-D64, and compensated-BF16-PV candidates; preserve BF16 Q inputs.
- **Why / reasoning:** original CTA-per-query-head path repeats KV loads and uses FP32 vector arithmetic. The AITER/TokenSpeed references show useful grouped matrix tiling; SGLang supplies a gfx1250 FP8-dot numerical caution.
- **Expected benefit:** higher reuse/arithmetic throughput for GQA and shared-cache multi-query cases. Native FP8 QK is initially limited to D64; compensated PV provides an accuracy fallback. No performance or accuracy result assumed.
- **Validation:** unequal per-head scales, tail/window/empty cases and the same FP32 oracle; sweep split count, tile size and warp count only after numerical gates pass.
- **Actual result/effect:** Grouped WMMA is retained with scaled-FP16 PV after EXP-017 corrected short/peaked failures. First held-out base GQA4.18us and MHA10.43us versus frozen baseline16.15/28.06us, with each policy's declared split count. See holdout/gqa_shared.json and holdout/mha_shared.json. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

## Follow-up candidates (not benchmarked)

| Work | Evidence / expected benefit | Required comparison |
| --- | --- | --- |
| Preserve WMMA operand layouts through packed-cache movement | Final long-context ATT still attributes16.6% of recorded stall cycles to lane permutations. A descriptor-view attempt hit a compiler assertion. | Validate a compiler/layout fix before testing performance; rebuild native changes and compare the full numerical suite. |
| Rotating-cache and real-request replay | Current data uses repeated same-input graphs. Cache residency and variable context lengths can change scheduling optima. | Save distinct warm/rotating-cache and caller-visible results; avoid inferring HBM utilization from current timing. |
| Exact model acceptance | Head mapping, T/mask semantics, production SWA window and projection remain undisclosed. | Replace explicit synthetic assumptions when real metadata is available; current kernels remain configurable. |
| Independent PMC confirmation | This profiler/build returns all-zero accepted counters despite nonzero decoded instructions. | Recheck on a compatible counter-enabled build; do not treat current zeros as hardware inactivity. |

## Entry template

```text
### EXP-NNN — YYYY-MM-DD — Title
- Status: planned / running / failed / retained / reverted / inconclusive
- Source before/after: revision or recoverable snapshot plus SHA256
- Trying: exact change and control
- Why / reasoning: observed bottleneck or correctness gap; supporting evidence
- Expected benefit: mechanism, affected cases, measurable prediction
- Environment and workload: GPU/partition, compiler/runtime, shapes, scales, masks, layout
- Validation/comparison: oracle, thresholds, baselines, commands, measurement protocol
- Actual result/effect: correctness/error, median/p95 us, noise, workspace, unsupported cases
- Decision: retain/revert/next action and reason
- Artifacts: raw results, logs, source/config, dated comparison chart and chart data
```

### EXP-011 — 2026-11-05 — Toolchain alignment and profiler repair

- **Status:** retained.
- **Trying / why:** build the repository's TLX 3.9 runtime so all four newer reference implementations and TLX use one compiler. Repair duplicate pip ROCm SDK core/devel libraries inside our isolated container so ATT can run.
- **Expected benefit:** valid same-compiler comparisons and instruction-level evidence; compiler changes alone must not be credited to kernel optimization.
- **Actual:** source-built wheel triton-3.9.0+git1b3fde8f, SHA256 87a8ab8d78fc6b6069b1d30b0526260b1ce0f9958f4f8e064bb30f574533e755, installed at /tmp/tlx-current in the experiment container. Core SDK matching library files now symlink to devel copies; originals retained as .att-original. Initial duplicate LLVM-option and HIP-registration failures are preserved in capture logs. No host runtime or other container changed.
- **ATT:** baseline GQA16/2 D64 L2304 S4, selected second dispatch, CU0/SIMD0/SE mask1. Analyzer reports sampled stalls dominated by s_wait_dscnt (60.7% of stall cycles) and s_wait_loadcnt (25.6%). The aggregate stall/latency ratio is 76.6%; this is not whole-kernel utilization or elapsed time. PMC confirmation remains pending.
- **Calibration:** early baseline/reference runs used different compiler versions and may overlap; exclude those from final speedup claims. On the aligned 3.9 runtime, baseline S4/BN64 was 16.10 us and matrix S4/BN64 14.82 us in initial calibration.
- **Artifacts:** results/2026-11-05_optimization/att_baseline_unified_sdk/, att_baseline_analysis.txt, candidates_gqa_first.json.

### EXP-012 — 2026-11-05 — Occupancy/split sweep and matched reference controls

- **Status:** retained.
- **Trying:** sweep BN32/64/128, 2/4 warps and split counts, then compare all source families sequentially on the 3.9 runtime. Separate unequal-per-head native comparisons from scalar/unit restricted controls.
- **Why / reasoning:** grouped GQA16/2 with S4 launches only eight partition CTAs for 96 CUs. Increased parallelism can outweigh partial/merge costs.
- **Expected benefit:** reduce underfill; determine where grouped versus vector arithmetic wins.
- **Actual so far:** vector layout constraints initially conflicted at the score-to-PV boundary; releasing the score layout fixed compilation. Vector candidates pass FP32 oracle, but matrix BN32 S72 ~5.35 us is faster in this initial sweep. This is tuning evidence, not a matched reference victory. All 62 attempted initial BF16 configs and errors are retained in tuning_gqa_bf16.json.
- **Next validation:** replay graph warmups before samples; rerun source controls on identical inputs/compiler; tune dense-width and full-FP8 cases and check masks/tails with the original thresholds.
- **Actual result/effect:** The initial tuning file preserves every tested configuration and failed compilation. Split/warp optima differ by shape; after later improvements the policy uses S36/W4 for base GQA and S9/W1 for base MHA. See tuning_gqa_bf16.json, warps_*.json and flat_mha_w1_s*.json. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-013 — 2026-11-05 — TDM packed-page movement

- **Status:** reverted.
- **Trying:** a separate matrix_tdm candidate that loads complete physical packed K/V pages with TLX TDM descriptors, reshapes them in registers, and masks all invalid V elements before PV. No global dequant/repack is timed or required.
- **Why / reasoning:** ATT points to wait-heavy staging, and aligned dense-width controls still favor AITER (14.25 us versus vector 26.07 us). Hardware descriptor movement may remove per-element address work and improve physical-cache bandwidth.
- **Expected benefit:** improved packed page movement and fewer global-load instructions; this first experiment tests the movement mechanism before adding pipeline depth.
- **Validation:** same BF16/FP8 Q numerical gate, poisoned tails, causal/SWA, one-split and split paths; descriptor variant restricted to page=BN=64/128 and aligned physical-page reads. Final output semantics unchanged.
- **Actual result/effect:** Correct but not selected: issue-and-wait packed descriptors regress dense S4 to33.10us versus vector24.46us; grouped one-page5.68us versus matrix5.37us. Raw tdm_* data and snapshots preserve this attempt. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-014 — 2026-11-05 — TDM double buffering

- **Status:** retained.
- **Trying:** stage the next complete packed page into a second LDS buffer before computing on the current page. Preserve matrix_tdm as the nonpipelined control.
- **Why / reasoning:** EXP-013 passed correctness but regressed dense-width S4 to 33.10 us (vector 24.46 us); issue-and-immediately-wait does not overlap descriptor loads with matrix/softmax work. GQA one-page splits likewise did not improve (5.68 versus matrix 5.37 us).
- **Expected benefit:** hide physical-page fetch latency for multi-page splits, with no expected benefit for one-page splits.
- **Validation:** identical error gate and matched split sweeps; ATT/PMC if retained.
- **Actual result/effect:** Double buffering improves the initial packed-descriptor dense S4 path25.59us versus33.07us. Flat descriptor movement supersedes this physical-group implementation; the two-buffer mechanism remains in the retained kernel. See tdm_mha_s4.json. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-015 — 2026-11-05 — Flat packed-page descriptors and fused TDM issue

- **Status:** retained.
- **Trying:** describe each packed KV head/page as a contiguous byte row, construct the base descriptors once, advance by physical block/head, and fuse K/V copies with disjoint warp hints.
- **Why / reasoning:** EXP-014 double buffering helps multi-page dense splits (S4: 25.59 versus 33.07 us) but still trails vector. AITER's source uses a flat packed-page descriptor, avoiding repeated descriptor construction and 2D physical grouping overhead.
- **Expected benefit:** reduce descriptor/address overhead while preserving the packed bytes and per-head scaling.
- **Validation:** same quantized-input oracle, then split/config sweep.
- **Actual result/effect:** Flat descriptors pass the oracle and reduce initial grouped latency to5.06us. Subsequent precision/warp/mask changes improve further. Raw flat_gqa.json and flat_mha_s*.json retain the initial result. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-016 — 2026-11-05 — Coalesced physical-page vector loads

- **Status:** retained.
- **Trying:** register-only loads of contiguous packed-page bytes, followed by logical reshape and FP32 vector attention. Keep the existing vector implementation as control.
- **Why / reasoning:** the dense TDM ATT sample includes substantial reshape/conversion/masking overhead around BF16 QK/PV, in addition to LDS waits. A scalar-address contiguous load may remove per-element paged address calculation without paying LDS or WMMA layout conversion.
- **Expected benefit:** faster dense MHA where grouping does not reuse KV and WMMA rows are mostly masked.
- **Validation:** identical inputs/threshold and poisoned physical-page tails; no synthetic layout repack added.
- **PMC limitation:** selected SQ LDS/wait/WMMA counters were accepted, but all 108 instances returned zero for both baseline and matrix, despite decoded LDS/WMMA instructions. These captures are unusable to confirm utilization or bank-conflict causality on this build. Preserve raw DBs and keep ATT causal interpretations provisional.
- **Actual result/effect:** Contiguous register page loads pass the strict gate and improve initial dense S12 to18.75us versus vector21.13us. Available as vector_page; final automatic policy selects faster matrix paths. See vpage_*.json. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-017 — 2026-11-05 — Scaled FP16 probabilities for accurate PV

- **Status:** retained.
- **Trying:** replace single BF16-PV multiplication with FP16 probabilities scaled by 32768, FP16 raw V and FP32 accumulation; divide the numerator by 32768 in the output epilogue. Preserve compensated BF16 PV as matrix_precise.
- **Why / reasoning:** expanded correctness had 119 passes / 10 failures / 8 unsupported skips. Uncompensated BF16 probabilities fail short-window and peaked cases (absolute errors up to ~0.00156). Compensated BF16 and vector passed. FP16 has three more significand bits; scaling keeps small probabilities away from FP16 subnormal range while p<=1 prevents overflow. E4M3 values are exactly representable in FP16.
- **Expected benefit:** meet the unchanged atol5e-4/rtol1e-2 gate with one PV WMMA rather than compensated two-dot PV. FP32 denominator and softmax remain unchanged.
- **Validation:** rerun the exact failing cases first, then the full 137-case suite. No tolerance relaxation. Previous candidate source and failed-case summary saved under snapshots/2026-11-05_exp016_* and results/2026-11-05_optimization/exp016_correctness_failure.json.
- **Actual result/effect:** All10 formerly failing cases pass after scaled FP16 PV. Expanded retained suites pass without changing atol5e-4/rtol1e-2. FP32 denominator and accumulation remain; BF16 compensated control remains explicit. See correctness_retained.xml. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-018 — 2026-11-05 — Preserve packed layout through LDS views

- **Status:** failed.
- **Trying:** express packed-page reshape/transpose on TLX shared-memory descriptors before loading registers, rather than reshuffling the loaded registers. Restrict native FP8 QK to FP8 Q, D64; use BF16 QK otherwise.
- **Why / reasoning:** ATT locates substantial layout/conversion work around QK/PV; TLX local_reshape/local_trans may let the compiler select a better LDS read directly into matrix operands. This follows AITER/TokenSpeed's explicit shared-memory views.
- **Expected benefit:** reduce register/LDS layout repair without changing cache bytes or input precision.
- **Validation:** compare matrix_tdm_view versus flat control, then expanded suite including FP8-QK and peaked cases.
- **EXP-017 outcome:** all ten previously failing cases now pass unchanged tolerances with scaled FP16 PV. Full-suite rerun pending.
- **Actual result/effect:** Shared-descriptor reshape/transposition hits a TlxPropagateLayout compiler assertion. It is excluded from the retained implementation; source snapshot2026-11-05_exp018_view.py and view_mha_first.json preserve the failure. No compiler patch attempted. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-019 — 2026-11-05 — One-warp tiles and smaller split merge

- **Status:** retained.
- **Trying:** allow one-warp vector/matrix CTAs, use separate descriptor copies for a one-warp TDM tile, and reduce merge warps for small split/D products.
- **Why / reasoning:** AITER's tuned gfx1250 configurations include one-warp CTAs. Our fixed four-warp tiles/merges can spend extra LDS/barrier work for tiny decode outputs.
- **Expected benefit:** lower synchronization/launch work on small cases; improve dense-page throughput by testing occupancy/register-pressure balance.
- **Validation:** gated matched split/warp sweeps followed by the full suite.
- **EXP-018 outcome:** descriptor-view candidate hit a compiler assertion in TlxPropagateLayout and is excluded from dispatch; this is a compiler limitation, not a numerical or measured performance result. Preserve source snapshot and compile diagnostics. No native compiler changes in this optimization task.
- **Actual result/effect:** One-warp dense partitions and smaller merge CTAs help, but one-warp grouped short partitions lose. Initial flat MHA S9/W1 reaches14.80us before mask peeling; grouped S36/W4 reaches4.57us. See warps_*.json and flat_mha_w1_s9.json. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-020 — 2026-11-05 — Shape-specific dispatch, reusable output/workspace

- **Status:** retained.
- **Trying:** select retained vector/matrix/flat-TDM variants and split counts using disclosed physical shapes plus explicitly synthetic head/query choices; provide a reproducible config query and reusable buffers.
- **Why / reasoning:** one configuration does not transfer. GQA16/2 prefers four-warp flat TDM with one page/split (~4.57 us initial), whereas MHA128/128 prefers one-warp flat TDM and larger chunks (~16.50 us at S12). Large query sets need fewer splits to avoid unnecessary merge traffic.
- **Expected benefit:** retain measured gains across short, long, GQA, dense-width and query-sweep families while reducing caller allocation cost. No claim of exact undisclosed model mapping.
- **Validation:** independent FP32 gate, all original semantic cases plus poisoned tails/peaked distributions, repeated matched-source comparisons; automatic policies remain conservative when tuning evidence is missing.
- **Actual result/effect:** Automatic selection, direct one-split output and reusable buffers are implemented. First29-case held-out matrix passes all TLX numerical gates, but revealed long-context/window gaps. Policy refinements follow in EXP-025..031; hidden model dimensions remain assumptions. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-021 — 2026-11-05 — BF16 Q as two FP8 components for D64 QK

- **Status:** reverted.
- **Trying:** for D64 BF16 Q, scale each row by a power of two, form E4M3 high and residual-times-16 components, and add two native FP8 QK products in FP32. Keep BF16-QK flat TDM as control and D128 fallback.
- **Why / reasoning:** remaining dense performance trails AITER. Native D64 FP8 QK avoids repeatedly converting packed FP8 K to BF16. Two FP8 K=64 products use the same instruction count as two BF16 K=32 products for D64, while removing the K conversion. Per-row Q preparation is amortized over KV tiles.
- **Expected benefit:** preserve BF16 query accuracy and reduce cache conversion cost; native FP8 contraction remains limited to 64 because of the referenced gfx1250 numerical issue at larger contraction dimensions.
- **Validation:** original BF16-Q oracle with unequal KV scales, short/window/peaked cases and matched dense controls. Reject if the unchanged numerical gate fails.
- **Actual result/effect:** Two-component FP8 QK passes the initial oracle but regresses dense S18 to26.52us versus16.25us BF16-QK flat control. Not selected. See dualqk_mha_first.json and the EXP-022 source snapshot. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-022 — 2026-11-05 — Four-buffer TDM lookahead

- **Status:** reverted.
- **Trying:** prime three physical pages, consume one while up to two later pages remain in flight, and reuse a four-buffer LDS ring. Drain all pending copies near the tail. One-warp separate copies and multi-warp fused copies have different per-wave wait counts.
- **Why / reasoning:** double buffering still waits for the next page at every iteration. TokenSpeed's peeled multi-buffer schedule and AITER's deeper cache pipeline suggest more overlap. The D64 two-component Q experiment passed the initial oracle but regressed MHA S18 (26.52 vs 16.25 us), so it is rejected from dispatch.
- **Expected benefit:** hide global latency in multi-page partitions; trade extra LDS for more outstanding loads. Single-page splits should keep the simpler flat path.
- **Validation:** same numerical oracle, short/empty/poisoned tail suite and repeated outputs to expose ring-buffer races; unprofiled timing versus the two-buffer control.
- **Actual result/effect:** Four-buffer ring passes the initial numerical gate but regresses dense S9 to20.19us versus14.84us two-buffer. Rejected from dispatch; deep_mha_s*.json and EXP-023 snapshot preserve it. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-023 — 2026-11-05 — Peel full-page masks and avoid unused-row softmax masks

- **Status:** retained.
- **Trying:** skip V tail sanitation on provably full physical pages using uniform control flow; mask invalid KV columns, causal positions and SWA as required, but do not mask padded query rows until stores. Specialize q_len=1 masking to scalar query position.
- **Why / reasoning:** all disclosed base contexts are page-aligned, but the generic path masks every V element and query row in every tile. Padded Q rows are zero and never stored, so computing their harmless softmax is sufficient. Invalid V tails still must be sanitized to prevent 0*NaN.
- **Expected benefit:** remove vector compare/select work from the common full-page decode path while preserving ragged/empty/SWA semantics.
- **Validation:** same poisoned-tail and short/peaked suite; immediate same-input AITER comparison. EXP-022 four-buffer pipeline regressed (MHA S9 20.19 vs two-buffer14.84 us), so dispatch retains two buffers.
- **Actual result/effect:** Mask peeling improves shared-scale MHA to10.46us versus AITER14.34us, and GQA4.19us versus4.85us. Independent first holdout reproduces10.43/14.09us and4.18/4.89us. See peel_*.json and holdout/*_shared.json. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-024 — 2026-11-05 — Retained implementation and held-out workload matrix

- **Status:** retained.
- **Trying:** remove rejected/dead candidates from the retained implementation (preserve snapshots), promote shape selection to default variant=auto, and run all four source families across explicit synthetic workload families.
- **Why / reasoning:** EXP-023 improved shared-scale MHA128/128 to 10.46 us versus AITER14.34 us, and GQA16/2 to4.19 versus4.85 us. Additional holdout is necessary before claiming transfer or repeatability. Full-FP8 unit-scale controls also measured TLX3.76us versus AITER4.56us / TokenSpeed6.55us for GQA, and TLX9.68us versus AITER13.50us / TokenSpeed28.92us for MHA. Those external full-FP8 paths fail our stricter numerical gate, and stay ineligible despite their measured latency.
- **Expected benefit:** a reviewable default kernel and defensible coverage/performance report rather than a single tuned shape. No exact undisclosed model acceptance or universal win assumed.
- **Validation:** retained suite173 passed /8 layout skips (181 total); previous expanded suite167 passed /8 skips. New synthetic matrix includes dense width8192, both Q dtypes, T=8..8192 over a shared noncausal folded cache, L=1K..256K, SWA37/128 assumptions, D128 and page128. run_suite.py saves case specs before execution and incremental status/raw logs.
- **Actual result/effect:** 29/29 synthetic TLX cases pass strict numerical gates. Base/T-sweep/dense-width cases beat their compared paths; first holdout loses to AITER at grouped128K/256K and SWA128 and is effectively tied at page128. Full-FP8 external paths fail this project's numerical gate and remain explicitly ineligible. Original holdout is preserved. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-025 — 2026-11-05 — One vector tile for short active windows

- **Status:** reverted.
- **Trying:** use BN64/128 instead of fixed BN32 for a <=128-token active context, retaining direct one-split output.
- **Why / reasoning:** held-out SWA128 is slower than AITER (4.81 vs3.21 us); the current tiny-window policy loops four times. A single vector tile may remove repeated softmax/reduction bookkeeping.
- **Expected benefit:** reduce launch body latency without adding a merge.
- **Validation:** SWA37/128 plus empty/tail/peaked cases, unchanged numerical gate.
- **Actual result/effect:** Larger vector tiles improve SWA128 from4.81 to3.69us but still trail AITER3.22us; warp1/2 do not improve it. Final tiny-window policy instead uses the tested one-split flat matrix path. See window*_bn*.json. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-026 — 2026-11-05 — Multiple physical pages per softmax tile

- **Status:** reverted.
- **Trying:** stage 2/4 independently paged physical KV blocks into one contiguous logical BN128/256 tile using TLX local_slice destinations; retain double buffering and FP16 scaled PV.
- **Why / reasoning:** the 29-case holdout passes TLX numerical gates, but long grouped contexts lose to AITER: 128K18.53 vs14.63 us, 256K33.48 vs18.11 us. BN64 performs repeated max/denominator/rescale work for every physical page. Larger logical tiles may amortize it without changing the page table/layout.
- **Expected benefit:** fewer online-softmax steps and larger matrix tiles, balanced against extra LDS/register pressure.
- **Validation:** direct frozen flat-path control; unequal per-head scales, randomized physical pages, multi-page poisoned tails, SWA and long-context reference comparisons. Preserve first holdout snapshot.
- **Actual result/effect:** BN128/256 logical tiles compile and pass the long-context oracle, but best256K result32.55us barely improves original33.48us and BN256 regresses57.8–63.0us. Multipage implementation is removed from retained code and preserved as2026-11-05_exp030_multipage.py. Single-page BN64 is retained. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-027 — 2026-11-05 — Lift the split cap for long grouped context

- **Status:** retained.
- **Trying:** test S256/384/512 and scale merge warp count with workspace reduction size.
- **Why / reasoning:** AITER's pinned kv_split config requests eight segments per CU for packed D64 caches. With only two KV heads on a 96-CU agent, its launch can use ~384 partitions; our prototype's inherited maximum128 artificially limits parallelism. EXP-026 BN128/256 did not fix the gap (256K best32.55 us versus original33.48; BN256 regressed to58–63 us).
- **Expected benefit:** shorter partition loops and more latency hiding; account for larger partial workspace and merge work. Keep long-context policy only after measured benefit.
- **Validation:** long-context numerical gate and non-power-of-two split merge; S384 must mask padded merge rows. Compare to the preserved first holdout and rerun AITER sequentially.
- **Actual result/effect:** Lifting the cap alone with four warps gives256K~27.1us (S384), still behind AITER. It enables the combined one-warp policy and non-power-of-two merge coverage. Supported explicit split range is now1..1024; automatic policies use<=512. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-028 — 2026-11-05 — Long-context warp occupancy and short-window matrix path

- **Status:** retained.
- **Trying:** combine larger split counts with one-warp partitions (matching AITER's pinned D64 3D config), and compare BN64 one-split matrix staging for SWA128.
- **Why / reasoning:** increasing splits alone with four warps improved 256K to~27us but still trails AITER18.11us. The source config also uses one warp and waves_per_eu=2; a split cap comparison alone did not reproduce that occupancy balance. SWA37 flat matrix measured2.60us versus vector3.24us, while BN128 multipage SWA128 still costs3.50us.
- **Expected benefit:** lower partition resource use on long context and less unnecessary per-head vector work for tiny windows.
- **Validation:** scalar-source controls plus unequal-scale oracle and preserved holdout baseline.
- **Actual result/effect:** One-warp256K S256 closes to17.94us, versus first AITER18.11us; small margin, requiring repeat validation. SWA128 BN64/W4/S1 reaches3.16us versus BN128~3.50us; SWA37 BN64/W4/S1 is2.60us. These configurations replace the earlier generic policies. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-029 — 2026-11-05 — Compiler occupancy hint for long one-warp tiles

- **Status:** inconclusive.
- **Trying:** match AITER's waves_per_eu=2 hint on the one-warp packed-page path; retain the default compiler occupancy setting as control.
- **Why / reasoning:** one-warp S256–512 closes the 256K gap to~17.94–18.14us (AITER18.11us first holdout), but that margin is too small to call a robust win. The source uses both one warp and an explicit occupancy hint.
- **Expected benefit:** influence register allocation/scheduling so memory/WMMA latency overlaps across waves.
- **Validation:** same numerical gate and matched controls; inspect compiled resource counts and repeat timings.
- **Actual result/effect:** waves_per_eu=2 changes partition register count241->237 but timing18.01->17.94us is a small/noisy effect. The retained implementation uses the default occupancy setting; the experimental hint alias is removed. See occupancy_l256k.json. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-030 — 2026-11-05 — Size the split merge from measured register cost

- **Status:** retained.
- **Trying:** numerically gated merge-only warp sweep on real attention partials for S36/128/256/384/512/1024, then validate the selected launch in complete attention timing.
- **Why / reasoning:** compiled metadata reports396 registers for the two-warp S256 D64 merge (zero reported spills). Only16 merge CTAs run in the folded case. More threads may reduce per-thread work and improve memory concurrency. Counter zeros cannot support a hardware-utilization claim; these are compiler resource counts.
- **Expected benefit:** lower final-reduction latency and improve the long-context total where partition time now approaches AITER.
- **Validation:** unchanged oracle, actual FP32 partial/LSE values, separate merge-only versus complete-kernel timings.
- **Actual result/effect:** All22 merge configurations pass their real-partial oracle. S36/128 favor one warp (~1.64/1.69us); S256 favors four (~2.21us); S384 favors eight (~2.64us); S512 favors four (~2.58us). Retained dispatch uses a simple size policy rather than treating a noisy single minimum as universal. Merge-only times are not complete attention latency. See merge_warp_sweep.json. Artifacts are under results/2026-11-05_optimization/ unless stated otherwise.

### EXP-031 — 2026-11-05 — FP16 raw-value partials at high split counts

- **Status:** retained.
- **Trying:** at S>=256 store normalized raw-V partial means in FP16 and apply the per-head V scale once in the final merge. Keep FP32 LSE, merge arithmetic and output BF16. Preserve the FP32 workspace-capacity API by viewing its opaque storage as FP16; internally allocated partial storage can use half the bytes.
- **Why / reasoning:** merge-only tuning gives~2.3us at S256, with396 compiled registers at two warps. E4M3 raw V values are bounded by448; their normalized nonnegative-probability weighted mean has the same bound, so FP16 partials cannot overflow. Applying V scale after merge preserves arbitrary head dequant multipliers. FP16 adds rounding, so the original gate remains mandatory.
- **Expected benefit:** reduce high-split partial write/read traffic and merge register cost.
- **Validation:** same FP32 oracle including unequal scales, non-power-of-two/empty splits and peaked distributions; reject on unchanged tolerance failure.
- **Actual result/effect:** expanded suite199 passed /8 layout skips (207 total), including S384 empty/non-power-of-two splits and peaked distributions. Matched256K control17.45us versus AITER17.81us; retain with small-margin caveat until repeated runs. Compiled merge register count396->168 (warp count also changed2->4). See halfpartials_l256k.json and correctness_optimized.xml.

### EXP-032 — 2026-11-05 — Final repeatability, reference split checks and ATT

- **Status:** retained.
- **Trying:** repeat the29-case synthetic matrix on retained code; sweep S4/16/64 for SGLang/vLLM decode controls; alternate AITER/TLX order on representative cases; capture final GQA/MHA/long kernels with the installed ATT skill.
- **Why / reasoning:** the final policy and FP16 partial encoding differ from the first holdout. Small long-context/window margins require repeated measurements, and caller-selected reference splits should not be artificially disadvantaged.
- **Expected benefit:** a reproducible final report with accuracy, timing variability, source limits, workspace behavior and profiler evidence. No universal win inferred from single noisy samples.
- **Validation:** same source pins, compiler and numerical gate; graph sample p95 is across batch means, not individual request latency. Raw failed gates remain visible; no PMC utilization claim from all-zero counters.
- **Actual result/effect:** 40 synthetic cases pass; correctness255 passed /8 skipped across the multiseed suite and additional D128 high-split checks. Source split sweeps, order reversals and three final ATT captures are saved. See RESULTS.md and the step06/07/08 charts.

### EXP-033 — 2026-11-05 — Refresh and validate moving upstream references

- **Status:** retained.
- **Trying:** audit current main revisions without mutating original checkouts; extract a second set of pinned sources under /references/current, then rerun relevant comparisons.
- **Why / reasoning:** remote HEADs moved. TokenSpeed's MHA decode file is unchanged; SGLang changes host core-count lookup; vLLM adds verification paths and gfx1100/gfx1201 launch tuning; AITER changes2D metadata specialization and adds block-scaled MXFP8 attention. Existing GFX1250 decode kernel bodies mostly remain, but fresh wrappers/window behavior need validation.
- **Expected benefit:** prevent stale-source claims while retaining the original baseline snapshots. New block-scaled MXFP8 operators have a different scale contract and are source references rather than per-head E4M3 timing baselines.
- **Validation:** same compiler, input seed and numerical gate; host import/platform adapters only; save updated exact pins and new raw results separately.
- **Actual result/effect:** Refreshed all four current main revisions in separate archives and reran40 cases plus source split/repeat controls. Base GQA/MHA paired ratios remain~1.16x/~1.34x versus AITER. Long/window/page128 margins are explicitly small. See latest/, manifest.json, upstream_diff_audit.json and REFERENCES.md.

### EXP-034 — 2026-11-05 — Optional SGLang Lean Attention control

- **Status:** retained.
- **Trying:** explicitly enable the source's optional ROCm Lean Attention decode path with its persistent-grid workspace, in addition to the ordinary SGLang split path.
- **Why / reasoning:** the ordinary direct-call default leaves Lean disabled. Its source accepts FP8 KV with scalar scales and may help long context, so comparison should include it rather than assuming the default is fastest. Source comments validate MI300X/MI355X; gfx1250 behavior and the project's numerical gate must be checked.
- **Expected benefit:** a stronger source baseline or a documented unsupported/numerically ineligible control. Wrapper metadata and lock-reset work are included in GPU invocation time.
- **Validation:** same quantized BF16-Q scalar-control oracle, base GQA/MHA and long GQA/dense contexts. No upstream GPU function changes.
- **Actual result/effect:** Lean executes: base GQA/MHA fails the project gate (~27.82/45.87us diagnostic latency). Long256K GQA/MHA passes but takes~109.84/3849.83us versus TLX~17.75/568.68us. Retain as measured optional control; not selected best. See latest_lean_*.json.

### EXP-035 — 2026-11-05 — Dedicated strongest SGLang/vLLM decode comparisons

- **Status:** retained.
- **Trying:** compare SGLang split decode, optional Lean, vLLM legacy decode, unified2D, production-style unified3D split-KV and opt-in tensor-descriptor variants. Tune reference splits, then repeat selected configurations against TLX. Audit AITER-backed framework routes.
- **Why / reasoning:** follow-up user request asks specifically for the strongest SGLang/vLLM baselines. Inspection found that prior vllm_unified calls omitted scratch/threshold arguments, forcing2D. Earlier numbers remain valid2D measurements but are not the best decode baseline and must not support broad vLLM claims. Production Triton backend supplies FP32 split output/max/sum buffers and16 segments.
- **Expected benefit:** expose a fair faster reference, even if this reduces or reverses the claimed TLX advantage. Per-head scale support and the unchanged numerical gate remain explicit.
- **Validation:** same gfx1250/96-CU agent and TLX3.9 compiler, same quantized payloads and page mappings; GPU0 checked idle. Scalar-scale references stay restricted controls; native per-head vLLM unified uses broadcast scale-cache views. Save selection and independent repeat data/charts. GPU reference bodies are unchanged.
- **Actual result/effect:** 18 focused cases;66 selected configurations all pass in three fresh order-rotated rounds (198 verification invocations). Native per-head base GQA TLX3.91µs versus best tested vLLM5.48µs (~1.40x), replacing the misleading broad interpretation of earlier2D10.36x. SGLang scalar base controls9.33/44.81µs (GQA/MHA) versus TLX3.88/10.19µs. Earlier data preserved; see FRAMEWORK_COMPARISON.md and step09 chart/CSV.

### EXP-036 — 2026-11-05 — Stronger launch-tuned vLLM3D controls

- **Status:** retained.
- **Trying:** sweep split counts and decode tile sizes32/64/128, alongside stock tile selection and the production16-segment configuration. Clone only the Python launcher globals to override _get_tile_size; upstream GPU bodies remain unchanged. Explicitly label these launch-tuned references separately from stock defaults.
- **Why / reasoning:** enabling3D reduces the native per-head grouped baseline from~44us to~8us at16 splits, and~5.7us after the pilot split sweep. BF16-Q stock decode tiles are16; this need not be optimal on gfx1250. User asks for best-performing relevant references, so a launch-parameter search is appropriate rather than retaining weak defaults.
- **Expected benefit:** a stronger competing baseline; no TLX speedup is assumed. The optional TD MHA path currently hits an AMD descriptor-layout compiler error and is ineligible on this build.
- **Validation:** same FP32 numerical gate, original kernel files, native unequal-head scales and shared scalar controls; independent order-rotated timing after selecting best passing configurations. Save complete logs and report production versus tuned settings.
- **Actual result/effect:** Production-style3D fixes the weak2D baseline. Stock16-segment GQA shared7.59µs improves to5.62µs atS256; native per-head tuned tile32/S128 gives5.48µs. Native per-head MHA tile128/S8 gives31.46µs versus stock44.22µs and TLX10.21µs. Full split/tile data and compiler failures saved under results/2026-11-05_framework_decode/. These are host launch controls, not production dispatch changes.

### EXP-037 — 2026-11-05 — vLLM KV-only descriptor launch control

- **Status:** retained as a reference control.
- **Trying:** retain the upstream GPU kernel but disable its optional Q/output tensor descriptors in the Python launcher while keeping KV descriptors enabled, using decode tiles32/64.
- **Why / reasoning:** the stock opt-in TD MHA configuration fails AMD lowering because the Q/output3D shared layout has order[2,0,1]. The kernel already exposes independent USE_TD and USE_TD_QO constexprs for non-power-of-two query grouping. A host-only flag choice can test the supported KV descriptor path without editing GPU code or the compiler.
- **Expected benefit:** a potentially faster MHA baseline; explicitly experimental launch configuration, not a stock ROCm default.
- **Validation:** unchanged numerical gate and input payload; compare against the best stock/tuned3D reference and TLX, preserve compilation failures.
- **Actual result/effect:** KV-only TD compiles where opt-in full Q/O TD MHA fails AMD layout lowering. Shared MHA best26.88µs and long dense2208.05µs beat the earlier tuned controls; native unequal-head best remains non-TD. Shared SWA128 KV-only2D3.89µs; native SWA best ordinary2D4.72µs. The experimental host flag is explicitly labeled; no GPU kernel or compiler edits.

### EXP-038 — 2026-11-05 — Validate legacy full-FP8 scalar-query controls

- **Status:** retained with restrictions.
- **Trying:** permit FP8 Q in SGLang/vLLM legacy controls when Q has the known common scalar scale from the synthetic input generator; fold that scalar into sm_scale. Unit-Q-scale controls require no change in the logits scale.
- **Why / reasoning:** prior adapters conservatively rejected all FP8-Q legacy calls because no q_scale argument exists. A common scalar Q scale can instead use the existing softmax scale argument; unequal per-query-head scales are still not supported. Source GPU kernels and payloads remain unchanged, and execution/accuracy must establish eligibility rather than an untested guard.
- **Expected benefit:** potentially stronger full-FP8 reference controls, correcting an incomplete earlier capability audit. This does not add per-head KV support to scalar-only kernels or change the BF16-Q comparisons.
- **Validation:** full-FP8 unit-scale GQA/MHA, original FP32 oracle and unchanged tolerance. Label scalar-Q-scale folding and distinguish operator eligibility from a production backend's advertised Q dtype.
- **Actual result/effect:** SGLang full-FP8 GQA passes atS32:9.34µs versus TLX3.73µs. Legacy vLLM GQA and optional Lean fail the gate; SGLang/vLLM MHA FP8-Q variants fail AMD lowering. Unequal Q-scale vectors remain rejected. Unit-scale operator evidence does not prove production advertised Q dtype or arbitrary-head full-FP8 support. See fp8_validated/ and fp8_console.log.

### EXP-039 — 2026-11-05 — Deduplicated AITER-backed framework control

- **Status:** retained as a deduplicated control.
- **Trying:** repeat the packed AITER unified operator beside the selected SGLang/vLLM native controls on all eight BF16-Q shared-scale cases.
- **Why / reasoning:** vLLM's ROCm AITER unified backend can dispatch this operator with shuffled KV on gfx1250. SGLang also routes through AITER, but its FP8 path may quantize Q or use a CDNA-only 5D decoder, so a generic framework label would hide precision/architecture differences. The prior AITER results are useful but a fresh companion run gives matched order-rotated timing.
- **Expected benefit:** prevent interpreting the native SGLang/vLLM Triton numbers as the fastest possible framework backend. Report one deduplicated underlying operator, with explicit routing/layout limits.
- **Validation:** identical shared-scale quantized payloads; same oracle/tolerance; three fresh rounds with selected native controls and TLX. No GPU source changes. Preserve raw source/configuration identifiers and link the prior AITER evidence.
- **Actual result/effect:** Fresh AITER shared controls: base GQA4.55µs versus TLX3.88µs (~1.17x), MHA13.82 versus10.19µs (~1.36x); GQA256K18.06 versus17.64µs and SWA2.94 versus2.91µs are near parity. **D128 AITER12.23µs beats TLX15.48µs (~1.27x)**, an unresolved TLX optimization opportunity. Source routing/precision/architecture limits are explicit; no independent framework labels are assigned to the same operator.

### EXP-040 — 2026-11-09 — Repeat the three-length comparison

- **Status:** completed; no kernel changes.
- **Trying:** rerun L2304/8192/32768 with retained TLX auto, pinned AITER unified, and previously selected passing vLLM3D configurations; retain stock16-segment vLLM and SGLang controls. Separately repeat TLX/vLLM with unequal per-head KV scales.
- **Why / reasoning:** user requests fresh numbers for a pasted three-row table. Its exact values/head mapping/asterisk definition cannot be recovered from saved records, so use the documented folded-GQA16/2,D64,page64,one-query BF16-Q/E4M3FN-KV setup and explicitly avoid claiming exact reproduction of the unidentified table. Shared-scale controls are necessary for AITER. Keep the November5 pins and configurations fixed to isolate a repeat from source/configuration changes.
- **Expected benefit:** fresh, accuracy-gated latency estimates and repeat ranges at all three lengths, with strong vLLM3D rather than an unidentified legacy/2D baseline. No speedup is assumed from the pasted numbers.
- **Validation/comparison plan:** GPU0 idle check; original source-built TLX3.9/Torch2.12/ROCm10.1 container, runtime96-CU gfx1250 agent. Three fresh order-rotated rounds, fifteen graph-batch samples of twenty invocations each; unchanged FP32 oracle atol5e-4/rtol1e-2. Preserve source/runtime hashes, raw repeat JSON, original pasted numbers, chart and CSV. No kernel changes or new tuning. Shared AITER timings remain one deduplicated underlying operator.
- **Actual result/effect:**72/72 selected verification runs pass. Shared L2304/8192/32768 medians: TLX3.92/4.47/7.15µs,AITER4.55/4.90/11.71µs,previously selected vLLM5.62/6.59/10.98µs; stock16-segment vLLM7.58/16.10/51.79µs. Native unequal-head TLX3.92/4.43/7.16µs versus vLLM5.42/6.70/12.34µs. First attempted shared run excluded because its immediately preceding snapshot showed19% activity/16046MiB used; reported shared_verified and native runs began at0%/161MiB. TLX and pinned GPU reference hashes unchanged. Three fresh rounds and ranges are saved in REPEAT_2026-11-09.md and step10 chart/CSV; raw JSON/manifest/source snapshots under results/2026-11-09_repeat_lengths/. No claim of exact reproduction of the unidentified pasted table.

### EXP-041 — 2026-11-09 — Freeze initial implementation and D128 baseline

- **Status:** completed.
- **Trying:** preserve both current kernel files, measure frozen/current TLX and AITER on D64 GQA/MHA and D128 MHA, capture D128 ATT and compiled resources before modifying GPU code.
- **Why / reasoning:** user authorizes roughly two hours of optimization across supported shapes/dimensions. The known failure is D128 MHA15.48µs versus AITER12.23µs. GPU0 is active with other work; select idle GPU2 (same96-CU gfx1250 agent class), and remeasure all before/after controls there. Do not treat cross-agent historical differences as improvements.
- **Expected benefit:** reproducible same-agent baseline and instruction-level evidence for the D128 gap. Preserve all prior snapshots and fixed AITER pins.
- **Validation:** unchanged per-head contract and FP32 gate, shared-scale AITER controls plus unequal-head correctness. ATT sampled-instruction fractions are not utilization. Prior PMC counters returned zero, so any causal utilization claim remains provisional.
- **Actual result/effect:** Frozen initial source SHA256s and clean GPU2 snapshot saved in [start.json](results/2026-11-09_optimization/start.json). Initial D128 repeat~15.08µs versus AITER~12.13µs. Captured [old D128 ATT](results/2026-11-09_optimization/att_d128_before_analysis.txt); sampled LDS waits justify testing movement/layout changes, not utilization conclusions.

### EXP-042 — 2026-11-09 — D128 partition warp and split balance

- **Status:** retained; final broad gate tracked in EXP070.
- **Trying:** numerically gated D128 MHA split/warp sweep with unchanged packed-page GPU algorithm; include warp1/2/4 and partition counts from1 through128.
- **Why / reasoning:** existing D128 dispatch retains four warps, while D64 MHA already benefits from one. D128 doubles packed K/V staging and operand/accumulator footprint, so warp count, residency and shorter partition loops may change the optimum. Test this first rather than assume a more elaborate rewrite is required.
- **Expected benefit:** close the D128 latency gap by balancing parallel partitions against merge overhead; preserve multiseed poisoned-tail/empty/SWA behavior.
- **Validation:** same payloads/scales/oracle, record failed configurations/resources, then repeat winners against frozen TLX/AITER. No policy retention from a single tuning minimum.
- **Actual result/effect:** All30 launch configurations pass. Three fresh rounds: frozen15.07µs, W1/S12 candidate8.85µs, AITER12.11µs. Retained one-warp D128 dense policy with page/length-based splits. [Raw paired data](results/2026-11-09_optimization/d128_policy_paired.json), [step11 chart](charts/2026-11-09_step-11_d128_launch.svg). Registers/thread increase with fewer warps; do not attribute gain to reduced per-thread registers.

### EXP-043 — 2026-11-09 — MHA vector versus WMMA at D128

- **Status:** rejected as faster replacements.
- **Trying:** compare vector/vector_page and ordinary matrix paths with the one-warp packed-page WMMA winner, sweeping short partitions.
- **Why / reasoning:** D128 MHA has one useful query row per KV head but pads to a16-row WMMA tile. The initial one-warp/S12 candidate reaches~9.23µs versus frozen~15.08 and AITER~12.13µs in first samples. A vector reduction may avoid padded arithmetic, although it may increase VALU work. Baseline ATT shows57.1% of sampled stall cycles at LDS waits, not proof of whole-GPU LDS utilization or bank conflict.
- **Expected benefit:** test whether eliminating padded WMMA work beats launch tuning; reject if extra vector reductions cost more.
- **Validation:** unchanged gate and unequal-head scales; all failed compiler/layout cases retained. Compare complete attention including merge, not isolated dot throughput.
- **Actual result/effect:** all45 vector/vector_page/ordinary matrix configurations pass, but best vector_page19.22µs is much slower than packed WMMA8.85µs. Preserve results and reject these replacements; padding alone does not establish a vector advantage.

### EXP-044 — 2026-11-09 — Broaden D128 tuning before dispatch retention

- **Status:** completed; dispatch retained subject to final repeat.
- **Trying:** extend one-warp partition checks to GQA,D128 page128 and short/long cache lengths, followed by a fixed same-agent before/after/AITER matrix. Keep the candidate separate until broad correctness and repeatability evidence supports dispatch.
- **Why / reasoning:** the D128 MHA/S12,W1 three-round result is8.85µs versus frozen15.07µs and AITER12.11µs, but one workload cannot establish a useful generic shape policy. Vector/vector_page/ordinary matrix alternatives all pass yet are slower (best~19.22µs), so retain packed-page WMMA as the starting algorithm.
- **Expected benefit:** a policy that transfers beyond the initial dense D128 case, with explicit exceptions for low-KV-head GQA and long context.
- **Validation:** vary length/head grouping/page sizes and both Q dtypes; same per-head-scale oracle; test selected launch policies against frozen kernels/AITER with packing excluded and no fabricated universal win.
- **Actual result/effect:** Broadened to80 cases across four families, page64/128, lengths1K..256K, both Q types, unequal scales, SWA and T8..8192. First scout passes80/80 but11 eligible controls favor AITER beyond3%; policy v2 reduces this to5. [Scout report](OPTIMIZATION_SCOUT_2026-11-09.md), [v2 raw](results/2026-11-09_optimization/scout_full_v2.json). Finite synthetic scope, no universal-win claim.

### EXP-045 — 2026-11-09 — Suppress the final unused TDM lookahead

- **Status:** retained.
- **Trying:** place next-page descriptor setup/load behind the block-uniform condition start+BN<hi, preserving the current consumed-buffer wait and local-load ordering.
- **Why / reasoning:** the existing loop always constructs an additional masked K/V descriptor pair, including its last iteration. Short split partitions commonly contain one to four pages, so redundant final descriptor setup can be a significant per-partition cost. The guard is uniform for the complete logical block and preserves the supported programming model.
- **Expected benefit:** reduce short-partition control/descriptor overhead without changing the payload, layout, FP32 softmax or scale handling; measure old/new GPU code at identical launches.
- **Validation:** matched frozen-launch control plus original-auto and AITER; empty/partial/poisoned pages and multiseed tests; reject on slowdown, correctness or compiler dependency failure.
- **Actual result/effect:** Matched frozen launch: long D12818.1129->17.8488µs; short D64 window2.7800->2.6987µs; dense D1288.8897->8.8364µs. Retained final-prefetch guard. [Long](results/2026-11-09_optimization/guard_long_paired.json), [window](results/2026-11-09_optimization/guard_swa_paired.json), [dense](results/2026-11-09_optimization/d128_guard_paired.json).

### EXP-046 — 2026-11-09 — Interleave one-warp K/V readiness and late V reads

- **Status:** retained.
- **Trying:** for the single-warp TDM path, wait for K while V may remain outstanding, compute QK/softmax, wait for V just before PV, and stagger next-K/next-V prefetches around those uses. Keep the multi-warp fused path unchanged because its transfers are issued by distinct warp masks.
- **Why / reasoning:** source inspection of pinned AITER shows K-ready and V-ready waits separated around QK. Our baseline waits for both before K, then loads both register tiles early. The D128 audit exposes remaining GQA/page128/SWA gaps despite the dense MHA win. Sequential one-warp TDM issue order makes wait1 meaningful: K(old),V(old),K(next),V(next). The last tile uses wait0 before V because no next K is issued.
- **Expected benefit:** overlap V transfer with QK and reduce V register lifetime; reduce page128 pressure without changing layout/precision. This is a scheduling hypothesis, not confirmed utilization evidence.
- **Validation:** preserve pre046 snapshot, compare identical launches, run poisoned-tail/empty/peaked multiseed tests and ATT. Do not apply the wait-count assumption to multi-warp fused transfers.
- **Actual result/effect:** Same-launch D128 dense8.9297->8.8043µs and page12810.2757->9.9609µs; small improvements, with different absolute levels by workload. Retained one-warp staggered waits/late V loads. Generated IR can sink guarded next-K setup; EXP056/069 attempts to force earlier issue lose. [Dense](results/2026-11-09_optimization/d128_interleave_paired.json), [page128](results/2026-11-09_optimization/d128_page128_interleave_paired.json).

### EXP-047 — 2026-11-09 — Revisit multipage tiles for short D128 windows

- **Status:** rejected for performance.
- **Trying:** test the preserved multipage TLX prototype at BN128,S1 on D128 SWA128, with warp1/2/4. This is an archived algorithm control, not the retained merge/policy.
- **Why / reasoning:** the earlier multipage attempt lost on long D64 contexts, but the current D128 SWA128 loop does two separate page softmax updates and still trails AITER (~4.1 versus3.4µs). One logical128-token tile may amortize that cost. Keep the original archived file immutable.
- **Expected benefit:** fewer online-softmax updates for a small bounded window; reject if the larger tile/register footprint dominates again.
- **Validation:** compare full BF16-Q per-head scaled attention against the unchanged oracle; only S1 avoids unrelated differences in the archived merge implementation.
- **Actual result/effect:** All three archived multipage S1 D128/SWA128 candidates pass but cost4.37–5.55µs, slower than the retained page loop. Archive remains immutable; no multipage implementation retained.

### EXP-048 — 2026-11-09 — One-tile vector decode for short windows

- **Status:** rejected for performance.
- **Trying:** prototype a direct per-query-head TLX vector kernel for at most128 visible tokens, with one softmax reduction and one output store; preserve paged physical K/V addressing, separate arbitrary head scales and BF16/FP8 Q handling.
- **Why / reasoning:** the existing vector path loops pages with online softmax and the matrix path pads scarce query rows. At tiny windows, duplicated GQA cache reads may cost less than two WMMA/online-softmax iterations. A dedicated bounded path can eliminate the loop and all split/merge workspace without approximating softmax.
- **Expected benefit:** improve D128 SWA and tiny cache lengths; retain only if complete attention latency and strict multiseed correctness support it. This does not assume all vector decode is faster (EXP043 rejected long-window vector controls).
- **Validation:** tail/empty/multi-page/window offsets, unequal scales, non-power-of-two GQA, both Q dtypes and peaked distributions. Dispatch must prove that all visible tokens fit the tile.
- **Actual result/effect:** Initial masked FP8 loads failed compilation because integer other=0 was used; corrected to0.0 and preserved initial logs. All three corrected candidates pass, but best4.57µs is slower than the matrix page path. Prototype retained only as experiment source.

### EXP-049 — 2026-11-09 — Unpack cache layout through shared-memory views

- **Status:** reverted after compiler failure.
- **Trying:** reinterpret the flat physical K/V LDS tiles as logical matrices with TLX local_reshape/local_trans before local_load, so the compiler can select matrix-operand register ownership directly. Preserve the TDM physical copy descriptors and memory allocation.
- **Why / reasoning:** current code loads a flat LDS tile into registers, then reshapes/transposes packed K/V and repairs layout for WMMA. D128/page128 uses~781 registers/thread at one warp and still loses to AITER. Pinned AITER performs shared descriptor view transforms before its operand load. Moving the same logical permutation into zero-copy shared views may eliminate register-level repair and shorten live ranges.
- **Expected benefit:** fewer register permutations/conversions and smaller live register footprint; no cache format, arithmetic, masking or precision changes.
- **Validation:** preserve pre049 snapshot, compare identical launches and inspect TTGIR/assembly/resources. If the compiler rejects a noncontiguous view reshape, retain the failure rather than changing compiler semantics. Revalidate async aliases/poisoned tails and all dimensions before retention.
- **Actual result/effect:** The noncontiguous memdesc reshape path fails TlxPropagateLayout with an LLVM dyn_cast assertion for NVMMASharedEncodingAttr. Saved full compiler reproducer/log and rejected_shared_views.py; restored pre049 GPU source without compiler edits.

### EXP-050 — 2026-11-09 — Load flat LDS in WMMA operand register ownership

- **Status:** retained.
- **Trying:** pin one-warp D128 K/V local-load register layouts to the inverse physical packing of the BF16/FP16 WMMA RHS fragment, while retaining the flat TDM LDS descriptors and register reshape/transposes.
- **Why / reasoning:** EXP049's noncontiguous shared-view route hits a TlxPropagateLayout assertion and is reverted with reproducers saved. The supported local_load(layout=...) API can express the same logical ownership without shared reshape operations. Derived the expected dot operand's linear bases using the installed Gluon IR layout builder (CPU-only metadata); the actual kernel remains TLX. For a32-lane warp, lanes map to16 column positions and the K-dimension bit8; registers hold K bits0..2, higher K groups, and higher columns. Inverse packing gives thread strides16,8 and register strides1,extent*16,256.
- **Expected benefit:** remove LDS-backed register redistribution before QK/PV, especially on page128 where the current one-warp specialization reports~781 registers/thread. Restrict the first trial to D128/one warp; do not assume FP8-Q D64 has the same fragment layout.
- **Validation:** same launches and pre050 snapshot, independent oracle, IR/assembly/resources and multi-seed stress tests. No compiler changes and no thread-varying scalar shortcuts.
- **Actual result/effect:** Same-launch D128/page1289.98->9.01µs versus AITER9.19µs; registers~781->747, shared64KiB unchanged, no reported spills. Small AITER margin requires broad repeat. [Raw](results/2026-11-09_optimization/d128_page128_layout_paired.json), [step12](charts/2026-11-09_step-12_operand_layout.svg).

### EXP-051 — 2026-11-09 — Extend operand-owned loads to D64 and FP8 Q

- **Status:** retained.
- **Trying:** retain the proven one-warp D128 register ownership and derive equivalent D64 BF16-Q and FP8-Q K-load layouts; PV continues using FP16 WMMA ownership. Sweep D128 short/page128 cases with the new loads.
- **Why / reasoning:** EXP050's identical-launch page128 D128 control improves~9.98->9.01µs and passes all three rounds, edging AITER~9.19µs. The direct packed-layout pin removes the register-layout repair without the compiler failure of shared descriptor reshapes. D64 BF16 Q uses the same8-element operand packing; FP8 QK uses16-element packing and a different lane K bit, so derive it explicitly rather than reusing the BF16 pin.
- **Expected benefit:** reduce LDS/permute overhead across both supported dimensions and query dtypes, potentially closing short-window and page128 margins.
- **Validation:** compare frozen/current/AITER across D64/D128,BF16/FP8 Q, both page sizes and multiple lengths; multiseed poisoned-tail and peaked/workspace tests. Keep broadening only after correctness and full attention latency validation.
- **Actual result/effect:** Derived separate FP8-Q D64 inverse K packing. Both Q types and both dimensions pass80/80 scout cases and the483-pass intermediate strict suite. [v2 scout](results/2026-11-09_optimization/scout_full_v2.json), [intermediate JUnit](results/2026-11-09_optimization/correctness_mid.xml). Final expanded suite/repeat is EXP070/072.

### EXP-052 — 2026-11-09 — Mark explicit-wait LDS loads as ready

- **Status:** retained as explicit readiness annotation; no standalone speed claim.
- **Trying:** use TLX local_load(relaxed=True) only after the corresponding explicit TDM wait, leaving refill barriers to the documented membar analysis.
- **Why / reasoning:** the API documents conservative producer-to-consumer dependency/wait tracking for default local_load and an explicit-ready marker when a preceding wait already orders the load. The one-warp interleaving has deliberate wait1/wait0 placement; redundant dependency tracking may constrain scheduling. This is not permission to omit waits or refill barriers.
- **Expected benefit:** avoid redundant producer wait dependencies/address live ranges while preserving RAW and consumer-to-refill ordering; reject if no measurable benefit or any correctness issue.
- **Validation:** preserve pre052 snapshot, identical-launch page128 and base D128 comparisons, multiseed empty/poisoned-tail tests; inspect generated waits/barriers and ATT before retention.
- **Actual result/effect:** Explicit-ready LDS loads compile and pass483 intermediate checks; paired page128 timing is near parity. This is not a measured optimization by itself; explicit TDM waits and compiler-inferred refill barriers remain. [Raw](results/2026-11-09_optimization/d128_page128_relaxed_paired.json), [JUnit](results/2026-11-09_optimization/correctness_mid.xml).

### EXP-053 — 2026-11-09 — Multipage WMMA with direct operand loads

- **Status:** reverted; no targeted win.
- **Trying:** generalize the retained packed TDM path to logical BN128 over two64-token pages, retaining direct operand-owned LDS reads and the staggered single-warp K/V pipeline. Keep page128 on one physical page per tile. This differs from EXP047's older unpinned archived implementation.
- **Why / reasoning:** D128 short windows and medium GQA remain behind AITER; their64-token page loop repeats softmax work. EXP047's archived multipage version lost, but EXP050/051 removed its LDS-to-dot redistribution cost. The inverse packed K layout needs a separate page-index register basis; V packing is contiguous across physical-page groups. Empty/partial tiles must preserve masked V sanitization and correct pending counts.
- **Expected benefit:** amortize softmax/control over128 logical tokens while preserving packed physical cache and exact head scales. Larger tiles may still regress long contexts; retain only targeted measured dispatch.
- **Validation:** new BN128/page64 explicit calls plus automatic D128 short-window path, independent poisoned-tail/empty tests and frozen/AITER controls. Partial waits allow NPAGES pending operations after K or V completion; multi-warp fused transfers continue using wait0. Prior full suite passed483/8 before this change.
- **Actual result/effect:** All multipage candidates pass. Direct-owned multipage SWA128 costs4.20µs and8K best~5.8µs, offering no clear benefit over the one-page retained code. Saved rejected_owned_multipage.py and restored the narrower public page=BN contract.

### EXP-054 — 2026-11-09 — Prove single-tile partitions from cache capacity

- **Status:** retained only for one-warp TDM.
- **Trying:** when query_length=1 and block-table capacity (or its conservative window/page-alignment bound) is at most SPLITS*BN, specialize the matrix kernel to one iteration per partition. Execute masked zero work for empty partitions; remove next-tile issue and loop-carried online state through compile-time folding.
- **Why / reasoning:** many GQA decode configurations partition to one page but still carry dynamic loops, max/denominator/accumulator state and descriptor lookahead control. A capacity bound is conservative even for shorter actual contexts and does not require host readback or a hidden alignment assumption. EXP053's larger tiles remain slower (~4.20µs SWA,~5.8µs at8K) and are reverted; this attacks control/state overhead while keeping the retained one-page data movement.
- **Expected benefit:** less loop/online-softmax state for one-page partitions, especially D128 GQA/page128; preserve multi-tile kernels for long partitions and windows that can span extra pages.
- **Validation:** empty and poisoned partitions, batch-variable lengths, both Q dtypes/per-head scales, same-launch before/after controls and IR showing constant one iteration. No reliance on a falsely small max_context_len hint.
- **Actual result/effect:** Capacity-based single-tile proof passes; improves several one-warp grouped/page128 choices. Four-warp specialization regressed and was excluded. Dynamic paths remain for all other configurations. [Sweep](results/2026-11-09_optimization/d128_single_tile_page128.json), [v2 scout](results/2026-11-09_optimization/scout_full_v2.json).

### EXP-055 — 2026-11-09 — Build TDM cache descriptors on the host

- **Status:** reverted for performance.
- **Trying:** pass native TensorDescriptor arguments for the existing flat K/V cache views rather than constructing their base descriptors in every GPU partition; keep device page-offset updates, copies, scales and arithmetic unchanged.
- **Why / reasoning:** baseline ATT shows descriptor/KMCNT setup waits among the largest stalls, and descriptor creation is repeated across many short partitions. The installed AMD backend supports native host TDM descriptors through make_tensordesc_arg/create_tdm_descriptor. Cache shape and pointers are already known on the host. This may reduce GPU setup, while Python/descriptor preparation can add caller overhead, so report both scopes and make no end-to-end claim.
- **Expected benefit:** lower partition setup latency on one-page GQA/SWA and page128; preserve identical physical cache contents. Captured-graph replay reuses the recorded launch.
- **Validation:** frozen identical-launch controls and full oracle, inspect kernel arguments/resources/waits; record caller timing separately. No repacking or quantization is added to hide GPU work.
- **Actual result/effect:** First host-descriptor trial had a host stride-variable naming bug; preserved failure rows and corrected it. Corrected identical-launch GQA D128/page128 costs~5.01µs versus frozen~4.66µs and AITER~4.39µs; caller setup also increases. Native host descriptor support works but is slower here; reverted.

### EXP-056 — 2026-11-09 — Preserve early next-K issue in the generated schedule

- **Status:** reverted for performance.
- **Trying:** add a TLX AMD compiler scheduling boundary after the guarded next-K issue and before QK, restricted to the one-warp multi-tile path. It is not a memory/workgroup barrier; all real TDM waits remain.
- **Why / reasoning:** dumped TTGIR merges identical guarded branches and moves next-K issue down beside the later V wait, reducing intended QK/prefetch overlap. The source order alone did not establish the machine schedule. An explicit supported scheduling marker may preserve the early boundary without issuing an extra masked final transfer.
- **Expected benefit:** better next-K overlap in two-page windows and multi-page partitions; inspect generated order rather than infer from source.
- **Validation:** same-launch frozen/pre056 comparisons, D128 SWA128 and longer decode, instruction/ATT inspection and correctness stress. No use of hardware thread IDs or hidden lane-dependent scalars.
- **Actual result/effect:** Scheduling boundary compiles and passes, but D128/SWA128 rises~3.45->3.66µs versus matched frozen launch. Explicit scheduling prevention is not automatically beneficial; full raw IR/assembly retained.

### EXP-057 — 2026-11-09 — Apply V scales once in every split merge

- **Status:** reverted; no measured benefit.
- **Trying:** extend the existing high-split FP16 raw-V partial convention to FP32 partials: partitions store unscaled normalized raw-V means and merge applies the KV-head V scale once. Single-split output keeps direct V scaling.
- **Why / reasoning:** EXP031 already uses this algebra for FP16 partials. FP32 partials do not need a different scale location. Moving VS removes repeated scale loads/multiplies from every partition while preserving arbitrary separate per-head scales and the opaque workspace API. It may also remove scalar memory readiness from short partitions.
- **Expected benefit:** less per-partition setup and output scaling; no added buffer or precision loss. The merged result remains mathematically sum_i(weight_i*rawVmean_i)*VS/sum_i(weight_i).
- **Validation:** independent FP32 oracle, both partial dtypes, all seeds/head groups/empty splits and caller workspace; same-launch frozen controls. Scheduling-marker trial EXP056 regressed and is reverted before this test.
- **Actual result/effect:** Extending raw-V means to FP32 partials passes but GQA/page128 repeat~4.66µs versus~4.62µs frozen is not an improvement. Restored the proven previous convention (raw FP16 high-split partials; scaled FP32 partials).

### EXP-058 — 2026-11-09 — Use explicit VMEM buffer loads for head scales

- **Status:** reverted for performance.
- **Trying:** load a one-element tensor of each KV-head scale through TLX buffer_load, then reduce its single element to the normal block-uniform scalar. Apply to matrix partitions and merge; retain the scale vectors and indexing unchanged.
- **Why / reasoning:** scalar-memory waits dominate several sampled setup regions. Explicit buffer operations can select the VMEM path instead of SMEM for tiny scale loads, potentially overlapping with Q/TDM traffic differently. This is a memory-path experiment, not a claim that VMEM is universally faster. The reduction has one logical value; it does not introduce lane-varying scalar state.
- **Expected benefit:** shorten short-window/short-partition scale readiness; reject on extra broadcast cost or compiler failure. No CPU scale extraction, scalar-only substitution or hidden repacking.
- **Validation:** generated memory instructions and same-launch short-window/page128 comparisons, then unchanged multiseed/per-head gates if promising.
- **Actual result/effect:** All scale-load trials compile and pass; the one-element VMEM route raises D128/SWA128~3.43->3.54µs. Reverted explicit buffer loads and restored ordinary scale indexing. No CPU scale extraction or contract change.

### EXP-059 — 2026-11-09 — One K/V buffer with register-consumption ordering

- **Status:** reverted for performance.
- **Trying:** use one LDS buffer per K and V on the single-warp path, retaining explicit readiness waits and relying on documented TLX membar analysis to order local reads before refills. Multi-warp fused loads retain two buffers.
- **Why / reasoning:** page128/D128 double buffering uses64KiB shared per partition. K is consumed into registers before next-K issue; V is consumed into registers before next-V issue. A single buffer can reduce shared capacity and address state, but may add necessary LDS-read-to-refill waits and hurt overlap. The full80-case scout passes every TLX numerical gate and exposes11 AITER-relative gaps, concentrated in page128, medium D128 GQA and short windows.
- **Expected benefit:** improve residency/throughput for large page128 caches, potentially reduce tiny-kernel setup; reject if synchronization costs dominate. No barrier is removed manually.
- **Validation:** same-launch frozen controls with compiler-inferred WAR ordering, variable/poisoned tails and long-context stress; inspect resources and ATT. No assumption that source sequencing alone orders async accesses.
- **Actual result/effect:** Single-buffer generalized path passes but matched long page128 D128 rises92.76->94.23µs despite halving shared capacity. Restored two slots. [Raw](results/2026-11-09_optimization/d128_singlebuffer_paired.json), [rejected source](results/2026-11-09_optimization/rejected_singlebuffer.py).

### EXP-060 — 2026-11-09 — Retune page128 and grouped one-page launches

- **Status:** retained; finite matrix only.
- **Trying:** sweep one-warp page128 MHA splits4..128 and grouped D128/page128 splits64/128/256/384, plus D64/D128 GQA one-page configurations after the new operand layouts/single-tile specialization.
- **Why / reasoning:** the80-case scout shows old launch heuristics still selecting four warps for grouped workloads and unnecessary/insufficient partitioning for page128. Prior launch minima predate the retained GPU layout changes. The one-buffer trial EXP059 passes but regresses the matched32K D128/page128 control (~94.23 versus92.76µs), so two buffers are restored.
- **Expected benefit:** choose launches suited to current code; account for smaller FP16 workspace at high split counts and merge overhead. No value-dependent dispatch or per-case hardcoded latency table.
- **Validation:** tune on unchanged unequal-head oracle, then repeat selected shape policies versus frozen/AITER; avoid interpreting single tuning minima or cross-date clocks as robust gains.
- **Actual result/effect:** Updated grouped/page128 warp/split policy. All80 v2 scout TLX gates pass; AITER-favored controls beyond3% decrease11->5, concentrated in D128 grouped medium lengths and128-token windows. [v2](results/2026-11-09_optimization/scout_full_v2.json), [first80 chart](charts/2026-11-09_step-13_full_scout.svg).

### EXP-061 — 2026-11-09 — Compensated FP8 probability/value products

- **Status:** reverted for accuracy and performance.
- **Trying:** on one-warp D128 packed decode, represent each probability as two E4M3 components, compute two native FP8 PV dots with raw FP8 V, and combine them with exact power-of-two scale factors in FP32. Keep FP32 softmax/denominator/accumulation and separate head scales.
- **Why / reasoning:** one FP8 WMMA consumes64 reduction elements versus32 for FP16, so two FP8 PV products can match the original FP16 dot instruction count while avoiding V upcast and reducing operand register storage. A residual component is needed because a single FP8 probability path fails this project's numerical gate. Two-component precision and tiny-probability range may still be insufficient; this is explicitly an accuracy/performance experiment, not an assumed retained change.
- **Expected benefit:** lower V conversion/register cost in page128 and short windows without changing stored cache format. Reject if any unchanged accuracy gate fails; do not relax tolerances or silently label inaccurate timings as wins.
- **Validation:** same-launch frozen controls, random and peaked/per-head cases, all supported dimensions/Q dtypes if the first gate passes. Save failed numerical results and source snapshot.
- **Actual result/effect:** Two-component FP8-PV~4.09µs versus retained~3.42µs, and4/16 tiny-probability-mass tests fail unchanged tolerance. Restored scaled-FP16 PV; retained regression. [Failed JUnit](results/2026-11-09_optimization/accuracy_fp8pv_tiny.xml), [paired](results/2026-11-09_optimization/d128_fp8pv_paired.json), [source](results/2026-11-09_optimization/rejected_compensated_fp8pv.py).

### EXP-062 — 2026-11-09 — FP16 partial workspace at moderate split counts

- **Status:** retained;739-pass strict gate and full matrix completed.
- **Trying:** extend the existing FP16 raw-V partial workspace from S>=256 to S>=8, retaining FP32 LSE, merge arithmetic and per-head VS application in merge. The preallocated FP32 storage remains opaque capacity, viewed as FP16 as before.
- **Why / reasoning:** moderate GQA/MHA partitions write/read hundreds of KiB of partial means. E4M3 raw values remain bounded, so partial means have ample FP16 range; only rounding can invalidate the accuracy target. This changes no Q/K/V payloads and can reduce merge/partial traffic without moving all FP32-partial scales (EXP057 did not help).
- **Expected benefit:** lower partial-store/merge cost in base D128 and page128 GQA; retain only if strict correctness and repeated complete attention timing improve.
- **Validation:** expanded multiseed peaked/workspace tests and new tiny-probability-mass cases. EXP061's two-FP8 probability prototype was slower (~4.09 versus3.42µs) and loses small probability mass:4/16 targeted tests fail. It is reverted and the regression test is retained with unchanged tolerance.
- **Actual result/effect:** Same-launch GQA/page1284.6930->4.6396µs (~1.1%), AITER4.4168µs. Small benefit, not enough to win this case. Extended partial precision threshold to8, adding explicit S8/12 strict stress tests in EXP070. [Paired data](results/2026-11-09_optimization/d128_halfpartials_paired.json).

### EXP-063 — 2026-11-09 — BF16 probability/value WMMA control

- **Status:** reverted for performance.
- **Trying:** replace the one-warp D128 FP16 scaled-probability PV dot with one BF16 PV dot (RNE probabilities), retaining BF16's exponent range, FP32 softmax/denominator/accumulation and the same raw FP8 cache/scales.
- **Why / reasoning:** the pinned AITER BF16-Q path uses BF16 PV probabilities; this avoids the scaled-FP16 probability multiply. Unlike EXP061's FP8 probability components, BF16 can represent the tiny probabilities in the new aggregate-tail regression. Its lower mantissa precision may still fail peaked/cancellation gates and must not be assumed acceptable.
- **Expected benefit:** test an AITER-relevant arithmetic tradeoff at the unchanged numerical target. Reject if any strict gate fails or complete attention is slower.
- **Validation:** same-launch repeats, full multiseed suite plus tiny-mass regression before retaining; no tolerance change or scalar/per-head substitution.
- **Actual result/effect:** First random gate passes, but BF16 PV~3.52µs versus scaled-FP16~3.44µs. Restored scaled-FP16 probability arithmetic; no lowered-precision win retained. [Raw](results/2026-11-09_optimization/d128_bf16pv_paired.json), [source](results/2026-11-09_optimization/rejected_bf16_pv.py).

### EXP-064 — 2026-11-09 — Explicit wide QK operand packing

- **Status:** reverted after compiler failures.
- **Trying:** pin D128 one-warp BF16 Q/K dot operands to the16-element WMMA operand packing used by the pinned AITER source, with the inverse packed K-load layout adjusted accordingly. PV remains the retained scaled-FP16 path.
- **Why / reasoning:** AITER's Q/K dot layout uses k_width16 even for BF16 Q with FP8 KV, while ordinary Triton dot lowering chose k_width8 in our dumps. Both express the same16x16x32 BF16 matrix instruction with different lane/register ownership. The existing TLX builder exposes the required WMMA/dot layout attributes, so a small private builtin can test this without compiler changes or arithmetic approximation. EXP063 BF16 PV passed its first random gate but was slower (~3.52 versus3.44µs) and is reverted.
- **Expected benefit:** reduce operand repair/load overhead by matching the native producer packing; reject if forced layouts add conversions or fail compiler verification.
- **Validation:** identical launch/frozen control, unchanged FP32 oracle and probability precision; inspect generated dot layout/assembly and run stress tests if promising.
- **Actual result/effect:** Initial private builtin import required core decorator; corrected. Forced kw16 trials then fail source materialization/layout reconciliation, including pinned score/accumulator retry. No native/compiler changes. [First](results/2026-11-09_optimization/d128_wideqk_paired.log), [pinned retry](results/2026-11-09_optimization/d128_wideqk_pinned.log), [source](results/2026-11-09_optimization/rejected_wide_qk.py).

### EXP-065 — 2026-11-09 — Matched ATT of the remaining short-window gap

- **Status:** partial capture; short-window coverage retries tracked separately.
- **Trying:** capture pinned AITER and current TLX D128/SWA128 with identical shared-scale benchmark inputs, plus current D128 MHA baseline shape, using the installed rocprof/ATT analyzer skill and one selected dispatch each.
- **Why / reasoning:** policy v2 passes all80 scout cases and reduces AITER-favored cases beyond3% from11 to5. Remaining gaps concentrate in D128 medium GQA and SWA. Earlier source-level scheduling hypotheses did not all translate to gains, so compare generated instruction/source behavior before retaining further pipeline changes.
- **Expected benefit:** distinguish repeated-softmax/layout/setup costs from actual sampled waits; no hardware-utilization conclusion from unavailable/all-zero PMC counters.
- **Validation:** same GPU2,CU0/SIMD0/SE-mask1 and iteration2; use unprofiled benchmarks for latency. Save raw traces, decoder output, source paths and analyzer reports. Different grids/instruction counts are not normalized utilization measurements.
- **Actual result/effect:** Current MHA D128 trace has valid sampled instructions:11329 stall/20110 latency,56.3% aggregate sampled fraction;~33.1% KMCNT,27.5% tensor-wait,3.9% DSCNT among stalls. Different grids forbid treating old/new sampled totals as kernel latency. Initial SWA AITER regex mismatched actual Triton2D route; corrected regex and widened-SE retries still have0 instruction samples. Occupancy shows waves on SIMD2/3 rather than selected0. These empty captures are not evidence of no stalls. [MHA analysis](results/2026-11-09_optimization/att_tlx_mha128_analysis.txt), [AITER retry](results/2026-11-09_optimization/att_aiter_swa_retry_analysis.txt), [TLX retry](results/2026-11-09_optimization/att_tlx_swa_retry_analysis.txt).

### EXP-066 — 2026-11-09 — Tune merge lane ownership on real partials

- **Status:** rejected for performance.
- **Trying:** compare one-warp split-merge register layouts with lane groups divided between split and dimension axes; optionally pin LSE weights to the matching slice layout. Use actual current attention partials and then benchmark the complete partition+merge path with the selected candidate.
- **Why / reasoning:** short grouped decode includes a separate merge launch whose global-load/reduction/store layout can be a meaningful fraction of total latency. A layout that reduces duplicate weight work may worsen coalescing or shuffles, so test the tradeoff rather than infer it from shape. This experiment does not alter softmax or partial precision.
- **Expected benefit:** lower merge time enough to close near-parity page128 grouped controls; reject kernel-only minima that do not improve total attention.
- **Validation:** unchanged independent oracle on real partials, three fresh full-attention order rotations, original merge as same-input control and AITER shared control; no new production mutation until validated.
- **Actual result/effect:** Initial custom weight ownership fails layout propagation; release-layout fix compiles all12 candidates. Best isolated merge1.842µs versus original1.621µs; no full-attention gain. Production merge unchanged. [Retry](results/2026-11-09_optimization/merge_layout_retry.json), [original failure](results/2026-11-09_optimization/merge_layout_gqa128.log).

### EXP-067 — 2026-11-09 — Audit the strongest applicable AITER short-window route

- **Status:** completed; forced route excluded for accuracy.
- **Trying:** record actual AITER SWA dispatch and test its unmodified Gluon2D kernel with a host-only eligibility override for BF16-Q/FP8-KV, retaining stock unified as the reference default.
- **Why / reasoning:** the initial ATT filter matched no AITER kernels. ELF symbol inspection proves stock mixed-dtype SWA calls kernel_unified_attention_2d (Triton), not the Gluon2D kernel: the Gluon2D gate requires equal Q/KV dtypes. Gluon imports are healthy and the source kernel contains an A16W8 path, so validate it as an explicitly experimental reference rather than assuming it is eligible/faster. No upstream GPU body is edited.
- **Expected benefit:** a complete reference audit and potentially stronger control; failed/slow forced routes remain diagnostics. Correct ATT filtering to the actual stock kernel.
- **Validation:** unchanged mixed-Q/KV oracle and scalar-control scope; label eligibility override separately from production dispatch. Source pins unchanged.
- **Actual result/effect:** Stock AITER Triton2D passes~3.07µs. Forcing mixed-dtype Gluon2D~2.95µs fails~99% output elements, maxabs~0.282. Its equality dtype gate is material and remains intact in default references. [Raw](results/2026-11-09_optimization/aiter_forced_swa.json). No inaccurate timing is eligible.

### EXP-068 — 2026-11-09 — Direct register loads for bounded short attention

- **Status:** reverted for performance.
- **Trying:** add a matrix_direct path that loads packed K/V through ordinary register loads instead of routing K through explicit async LDS staging. Use a128-token logical tile and a conservative no-page-alignment single-tile bound for short windows.
- **Why / reasoning:** the actual fastest stock AITER SWA route is its Triton2D kernel. The old TLX matrix fallback always staged K through LDS; previous direct-vector trials did not test direct WMMA register loads. Avoiding unnecessary staging may matter more than pipeline changes for at most128 keys. Keep scaled-FP16 probability precision and every head scale.
- **Expected benefit:** close short-window D128 losses with a simpler load path; use TLX TDM for longer contexts where it wins. No cache repacking or altered accuracy target.
- **Validation:** GQA/MHA,D64/D128,page64/128, aligned and unaligned/tail/empty windows, both Q dtypes and tiny probability mass. Retain only after repeat timings and the full strict suite.
- **Actual result/effect:** Ordinary/i32/buffer direct WMMA loads pass first numerical checks, but best i32~5.25µs and explicit-buffer~5.09µs versus retained TDM~3.4µs SWA. Larger one-warp tiles are especially slow. Removed all experimental direct variants/API flags, saved [source](results/2026-11-09_optimization/rejected_direct_vmem.py) and [12-candidate results](results/2026-11-09_optimization/d128_direct_buffer_swa.json).

### EXP-069 — 2026-11-09 — Predicate final lookahead without a runtime branch

- **Status:** reverted for performance.
- **Trying:** issue next-K and next-V descriptor operations unconditionally on the one-warp multi-tile path, using the descriptor's existing uniform validity predicate to suppress out-of-range memory accesses. Single-tile and multi-warp paths keep their existing behavior.
- **Why / reasoning:** the compiler merges identical next-tile guards and can sink the K lookahead past QK. EXP056's scheduling marker was slower. Removing that repeated control-flow guard may preserve overlap, at the cost of extra final masked descriptor operations. Pending counts must include those issued predicated operations; keep documented waits.
- **Expected benefit:** overlap next K with current softmax/PV on medium grouped contexts and two-page windows; reject if final descriptor setup costs dominate.
- **Validation:** compare matched pre069 source, D128 GQA/SWA and long MHA; unchanged numerical tests and explicit empty/poisoned tails. EXP068 ordinary/i32/buffer direct loads all passed their first gates but best~5.09µs versus retained~3.4µs SWA; rejected all direct-load API/code changes and saved source.
- **Actual result/effect:** Branch-free predicated lookahead passes but rises~3.43->3.58µs at identical W1/S1 D128 SWA launch. First driver call referenced a missing config file and did no kernel benchmark; corrected config retry saved. [Retry](results/2026-11-09_optimization/d128_predicated_lookahead_retry.json), [source](results/2026-11-09_optimization/rejected_predicated_lookahead.py).

### EXP-070 — 2026-11-09 — Final strict validation of moderate-split partials

- **Status:** completed;739 passed,8 skipped.
- **Trying:** expand peaked/multiseed/caller-workspace regression coverage to split counts8 and12 at the new FP16 boundary, alongside counts4/384 and all original tests. Run the tiny-probability-mass regression with restored FP16 probability arithmetic.
- **Why / reasoning:** moderate partial precision can change cancellation behavior even when random scouts pass. Existing stress cases bracketed the old threshold; explicit new boundary coverage verifies the retained precision choice at unchanged tolerances. EXP069 predicated lookahead passes but regresses~3.43->3.58µs; reverted and source saved.
- **Expected benefit:** validate all retained GPU/dispatch changes before reporting repeated gains. No speed claim from pytest elapsed time.
- **Validation:** full tutorial pytest -s --tb=short, independent FP32 oracle, unequal head scales/both Q dtypes, seeds0/19/182, poisoned tails/empty contexts/SWA/caller workspace. Preserve failures if any; revert a numerical optimization rather than relax the gate.
- **Actual result/effect:** Full strict suite739 passed,8 skipped in130.89s. Both new moderate-split boundaries and all16 tiny-mass regressions pass. FP16 probability arithmetic and moderate FP16 partials retained at unchanged tolerance. [JUnit](results/2026-11-09_optimization/correctness_final.xml), [log](results/2026-11-09_optimization/correctness_final.log). The subsequent EXP071 allocation experiment needs its own post-change gate.

### EXP-071 — 2026-11-09 — Single buffer only for proven single-tile partitions

- **Status:** allocation change reverted; policy follow-up retained.
- **Trying:** allocate one K/V LDS slot only when SINGLE_TILE is proven; retain two slots in every multi-tile path. Sweep grouped D128 moderate lengths with split counts that yield one tile, versus the pre071 source and AITER.
- **Why / reasoning:** EXP059's general one-buffer trial hurt long-context overlap and was reverted. A single-tile partition never issues another transfer or indexes a second buffer, so its extra capacity cannot enable overlap. This restricted allocation halves reserved LDS without changing read/refill ordering or extra WAR waits.
- **Expected benefit:** higher partition residency on page128 and medium grouped shapes; no assumption that smaller allocation alone improves latency.
- **Validation:** explicit same-launch frozen controls, compiler resource evidence, boundary/tail/empty tests and full repeated matrix if a candidate survives. Run after existing full suite to avoid source mutation during pytest.
- **Actual result/effect:** Allocation change is neutral: matched original5.0078µs versus candidate5.0172µs; original auto5.8042µs, AITER4.7103µs. The compiler already eliminates unused single-tile storage. Reverted the redundant allocation edit. The lower FP16 partial threshold changes the split tradeoff: page128/D128 grouped S64 is now faster than auto S32 at8K, while S128 remains best at32K. Retain a one-page target capped at128 splits for grouped D128/page128, preserving32K policy. [Paired](results/2026-11-09_optimization/singletile_buffer_paired.json), [32K sweep](results/2026-11-09_optimization/singletile_buffer_gqa128_p128_32k.json). No independent speed claim for smaller allocation.

### EXP-072 — 2026-11-09 — Final repeated matrix and framework controls

- **Status:** completed; all final comparisons saved.
- **Trying:** freeze final source, repeat all80 synthetic cases against initial frozen TLX and every eligible pinned AITER control, then repeat the three core cases against prior strongest passing SGLang/vLLM selections.
- **Why / reasoning:** the previous all80 runs were scouts; final timings need fresh rounds after all retained changes. EXP071 shows no allocation win but supports retuning page128 D128 grouped splits after the FP16 partial threshold change. Avoid cross-date/agent comparisons, tuning minima and inaccurate reference rankings.
- **Expected benefit:** a complete auditable final result, including remaining AITER wins and any starting-TLX regressions. The user asks to beat all shapes; the result must state where that target is not achieved.
- **Validation:** GPU2 clean-state snapshot, unchanged source pins/compiler, sequential jobs; three order-rotated rounds,15 graph samples of20 invocations; strict unchanged FP32 gate. Save core and four family charts with CSV/JSON. Repeat strict suite after final dispatch change.
- **Actual result/effect:** Final retained kernel passes739 strict tests/8 skips and80/80 synthetic benchmark cases in three rounds each. Of40 passing AITER controls:31 TLX wins beyond3%,4 within3%,5 AITER wins. No starting-TLX regressions beyond3%. Eight AITER cases fail unchanged accuracy and remain excluded. Fresh strongest-tested SGLang/vLLM/AITER core controls pass45/45 selected verification runs. [Final report](OPTIMIZATION_2026-11-09.md), [step15 matrix chart](charts/2026-11-09_step-15_final_matrix.svg), [step16 frameworks](charts/2026-11-09_step-16_framework_repeat.svg), [summary](results/2026-11-09_optimization/final_summary.json).

### EXP-073 — 2026-11-09 — Unroll the short-window page loop

- **Status:** reverted for performance.
- **Trying:** use supported Triton loop-unroll factor2 for one-warp TDM single-query/single-split windows, leaving loop bounds/masks, TDM ordering and all other paths unchanged.
- **Why / reasoning:** SWA128/page64 has two page updates in aligned tests; the dynamic loop carries online-softmax state and descriptor branches. Compiler unrolling may remove loop/control overhead or improve overlap without a new arithmetic path; it may also raise register pressure. Earlier forced schedule/predicate trials lost, so this is a measured compiler-option control.
- **Expected benefit:** close some remaining short-window D128 gap while preserving empty/tail behavior and complete attention scope.
- **Validation:** pre073 matched-launch snapshot, three fresh order-rotated rounds plus independent unchanged oracle; reject on compiler failure/latency regression and preserve source/IR. Only run final matrix after the decision.
- **Actual result/effect:** Unroll2 compiles and passes but costs~3.67µs versus matched original~3.44µs. Reverted. [Raw/IR](results/2026-11-09_optimization/d128_unroll2_paired.json), [source](results/2026-11-09_optimization/rejected_unroll2.py). Final retained source is pre073 plus the documented page128 grouped policy; no unrolling override remains.

### EXP-074 — 2026-11-09 — Recover after host/container interruption

- **Status:** completed; environment restored and revalidated.
- **Trying:** restore the dedicated experiment container after host reboot, verify runtime/compiler/GPU identity, repeat strict tests and frozen/current/reference comparisons sequentially.
- **Why / reasoning:** host boot time is Nov9 19:32:58 EST (00:32:58 UTC), container exited255 at00:33:33 UTC, no OOM flag. On the next user status turn at01:51 UTC all previous exec sessions were gone. AMD-SMI now reports294912MB physical memory versus prior smaller report; Torch still reports the identical96-CU/144-GiB gfx1250 runtime agent, PCI domain2/bus4/device0. Check before interpreting timings. Saved kernel/data/tests survived.
- **Expected benefit:** complete the final repeat without mixing cross-reboot absolute timings. Every new comparison re-runs both the frozen initial TLX and retained TLX on the same current agent; preserve earlier scout/paired evidence.
- **Validation:** dedicated container restart, clean GPU2 snapshot, Torch2.12/ROCm10.1/TLX3.9 unchanged, same source pins. Record post-interruption phase separately. Existing retained-source suite739 passed/8 skipped before interruption. ATT consecutive/SIMD retry was in post-workload decoder finalization when the host rebooted; raw files preserved, no completed AITER instruction statistics or performance conclusions from that retry.
- **Actual result/effect:** Dedicated container restarted; runtime confirmed unchanged96-CU/144-GiB gfx1250/Torch2.12/TLX3.9, PCI2:4:0. Before/after rerun together after host reboot. Retained kernel739 passes/8 skips; final80-case three-round comparison and45 framework repeats complete. [Manifest](results/2026-11-09_optimization/manifest.json), [final raw matrix](results/2026-11-09_optimization/repeated_final_v2.json). Initial consecutive/SIMD short-window retries provide no completed nonzero instruction statistics; do not infer utilization or stalls from them.

### EXP-075 — 2026-11-09 — Preserve the original multi-warp loop schedule

- **Status:** retained for multi-warp; one-warp induction refined in EXP077.
- **Trying:** restore original start-based loop induction for nonsingle-tile partitions and original fused prefetch/default LDS load semantics on multi-warp TDM. Keep all one-warp operand layouts and staggered waits, and constant0..1 loop for proven single-tile partitions.
- **Why / reasoning:** final repeated80-case matrix passes all TLX gates and shows core15.22->8.26µs D128, but exposes ten original-TLX regressions beyond3%, particularly4–9% on large-T multi-warp paths. The new page-index induction and one-warp scheduling changes should not impose a slower schedule on the original multi-warp path. This is a generic control-flow restoration, not dispatch by measured latency.
- **Expected benefit:** recover original multi-query throughput while retaining one-warp decode gains; possibly reduce scalar address arithmetic in medium GQA loops.
- **Validation:** compare pre075 snapshot against new identical launches on large-T and short D128 windows; then full strict suite and fresh80-case repeated matrix. Preserve the pre075 step14 chart/data/report separately.
- **Actual result/effect:** Three fresh matched rounds recover GQA D64/T8192 from pre075243.15µs to223.91µs (~7.9% lower latency); unchanged accuracy. This restores original-level throughput and retains new decode scheduling only where appropriate. [Paired raw](results/2026-11-09_optimization/restore_multiwarp_paired.json).

### EXP-076 — 2026-11-09 — Restore short-window and FP8-Q D128 grouped warp choices

- **Status:** retained; final strict and80-case repeat complete.
- **Trying:** select four warps for D128 windows<=64 and for FP8-Q D128 grouped page64 ordinary contexts; retain one-warp D128 BF16-Q, larger windows and long contexts.
- **Why / reasoning:** repeated matrix identifies~5–6% regressions versus initial TLX for37-token windows and FP8-Q D128 grouped base case, despite overall D128 gains. Old four-warp choices remain valid; dtype/window properties justify selective policy rather than forcing one warp across different work. EXP075 restores the original multi-warp schedule first.
- **Expected benefit:** recover earlier precision/window regimes while retaining D128 BF16 decode and long-context gains.
- **Validation:** repeat selected affected cases against original frozen TLX and eligible AITER, then all739 strict cases and80-case repeated matrix; retain only if current source improves measured coverage. No change to payload/scales/oracle.
- **Actual result/effect:** Three-round recovery controls pass: GQA D128 W37 current2.819µs versus original2.821; MHA W37 current2.927 versus2.917; FP8-Q D128 unit4.324 versus4.344 and unequal4.354 versus4.348. Recovered prior~5–6% losses to near parity without changing arithmetic. Medium GQA and W128 still favor AITER. [Raw](results/2026-11-09_optimization/recovered_policies.json).

### EXP-077 — 2026-11-09 — Keep induction specialization local to each warp regime

- **Status:** retained; final strict and repeated gates complete.
- **Trying:** retain the recovered original start-based induction only for multi-warp partitions; preserve the earlier tile-index induction for one-warp decode, with the same constant0..1 single-tile proof.
- **Why / reasoning:** EXP075 recovers multi-query throughput, but the new full matrix shows D64 MHA rising~8.43->9.31µs when the original induction is also applied to one warp. Preserve the proven schedules independently using compile-time warp-count selection; no duplicate kernel or value-dependent dispatch. D128 remains~8.3µs.
- **Expected benefit:** retain the dense D64 gain alongside large-T recovery rather than trading one for the other.
- **Validation:** final prior job completes first; compare compile-time specialization on the three core controls plus large-T, then all strict correctness/80-case repeats. Preserve preceding source/data separately.
- **Actual result/effect:** Compile-time induction choice retains one-warp decode gains and original multi-warp throughput. Final fresh core medians: GQA D64 original3.8699/current3.8559/AITER4.5442µs; MHA D64 original10.2351/current8.5022/AITER13.9029µs; MHA D128 original15.4236/current8.4482/AITER12.2020µs. All80 cases pass, no original-TLX regression beyond3%,739 strict tests pass/8 skip. We have not beaten AITER everywhere:five D128 controls still favor it. [Final chart](charts/2026-11-09_step-15_final_matrix.svg), [raw](results/2026-11-09_optimization/repeated_final_v2.json), [JUnit](results/2026-11-09_optimization/correctness_final_v2.xml).

### Final retained-pass artifacts — 2026-11-09

The final source snapshot and SHA256s are in [manifest](results/2026-11-09_optimization/manifest.json) / `final_source/`. Source archives/pins match the previous framework manifest. Original prototype partition/merge ASTs are unchanged. All new final GPU jobs run sequentially on the verified GPU2 runtime partition; CPU chart rendering is separate. This pass includes the host reboot interruption and recovery described in EXP074.

| Major step | Saved comparison | Source data |
| --- | --- | --- |
| D128 launch balance | [step11](charts/2026-11-09_step-11_d128_launch.svg) | [paired](results/2026-11-09_optimization/d128_policy_paired.json) |
| Packed operand loads | [step12](charts/2026-11-09_step-12_operand_layout.svg) | [paired](results/2026-11-09_optimization/d128_page128_layout_paired.json) |
| First80-case scout | [step13](charts/2026-11-09_step-13_full_scout.svg) | [CSV](results/2026-11-09_step-13_full_scout.csv) |
| Initial full repeated candidate | [step14](charts/2026-11-09_step-14_optimized_matrix.svg) | [CSV](results/2026-11-09_step-14_optimized_matrix.csv) |
| Final retained80-case matrix | [step15](charts/2026-11-09_step-15_final_matrix.svg) | [CSV](results/2026-11-09_step-15_final_matrix.csv) |
| Fresh best-tested reference repeats | [step16](charts/2026-11-09_step-16_framework_repeat.svg) | [CSV](results/2026-11-09_step-16_framework_repeat.csv) |

Four family charts accompany steps13/14/15. Earlier dated files and unsuccessful experiment rows/sources are preserved. The final [report](OPTIMIZATION_2026-11-09.md) gives the five remaining AITER-favored controls and excludes numerical failures. The user objective of winning across every shape is **not achieved** by this pass; all three original core controls now beat AITER, including the former D128 loss, and none of the80 tested cases regresses beyond3% against starting TLX.

### EXP-078 — 2026-11-10 — Prepare the tutorial and evidence for a PR

- **Status:** locally prepared; commit/push/PR publication awaits target and draft review.
- **Trying:** stage the new tutorial, optimized sibling, contract documentation, tutorial index, standard correctness-suite imports, reference harnesses and portable evidence on a feature branch. Preserve full history in a hash-verified archive while excluding raw profiler binaries and decoder output.
- **Why / reasoning:** the complete portable history would add roughly500K lines to the diff. The curated tracked evidence plus a compressed archive keeps current code reviewable and retains every prior portable record. Historical snapshots remain unformatted; live formatting must preserve all kernel/launcher/test function ASTs and benchmark values. Standard test imports must be visible to the existing kernel-only CI selector for both implementation files.
- **Expected benefit:** a reviewable, reproducible new-kernel PR with explicit per-head scope and honest remaining AITER gaps, without publishing unrelated local captures or account credentials.
- **Validation/comparison plan:** full repository pre-commit; function AST comparisons against measured snapshots; archive SHA256/path verification; semantic JSON and whitespace-only SVG checks; clean patch-application preflight against current upstream main. A fresh GPU rerun is attempted through test_correctness.py -k fp8_kv_decode rather than only the standalone file.
- **Actual result/effect:** all applicable pre-commit hooks pass; workflow YAML expansion is inapplicable. All kernel, launcher and embedded-test function ASTs match the739-pass snapshot, and all18 experiment-script ASTs match their archived originals. Both kernel file changes select the standard correctness suite. All1072 archive files pass SHA256 and safe-path checks. Staged patch applies to current upstream main. The host GPU driver/device was unavailable and the container stopped before the fresh GPU rerun produced output; historical739/8 and80-case results are retained, not represented as newly rerun. GitHub identity adityakankariya and ADMIN access to the configured fork are verified using a separate CLI configuration outside Git. [Preparation manifest](results/pr_preparation/manifest.json), [pre-commit output](results/pr_preparation/pre_commit.txt), [archive manifest](history/manifest.json).
