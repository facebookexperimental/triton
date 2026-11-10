# Reproducing the FP8 KV decode work

The contract is **plain E4M3FN KV with per-KV-head FP32 dequantization multipliers**. This is not E8M0/block-scaled MXFP8. BF16 or E4M3FN Q, per-query-head Q scales when needed, FP32 softmax/accumulation, BF16 output. See the tutorial RST for shapes, masking and workspace ownership.

The exact source revisions and hashes are in [manifest.json](results/2026-11-05_optimization/manifest.json). Reference checkouts are under `/tmp/fp8-kv-review/{aiter,vllm,sglang,tokenspeed}`, mounted at `/references` in the experiment container. Refreshed sources are archived under `/references/current/{aiter,vllm,sglang,tokenspeed}`; set `FP8_REFERENCE_ROOT=/references/current` for the final comparison pins. The original checkouts remain unchanged. The container mounts this repository at `/workspace`.

## Environment

- Container: `adkankar-tlx-opt-20261105`.
- Image: `rocm/sgl-dev:v0.5.23-rocm101-mi45x-20261101`.
- Torch: `2.12.0+rocm10.1.0`; source-built TLX runtime: `3.9.0+fb` at `/tmp/tlx-current`.
- Selected agent: `CUDA_VISIBLE_DEVICES=0`, gfx1250, 96 CUs, 144 GiB, DPX/NPS2. This is the runtime-visible partition, not a full-device specification.
- Clocks were not locked. Every GPU test/benchmark/profile was run sequentially on this selected agent. Other containers were not changed.

The runtime was built from this checkout with:

```bash
docker exec -e MAX_JOBS=24 -e TRITON_BUILD_PROTON=OFF adkankar-tlx-opt-20261105 \
  python -m pip wheel --no-build-isolation --no-deps /workspace -w /tmp/tlx-wheels
```

The wheel is `triton-3.9.0+git1b3fde8f-cp312-cp312-linux_x86_64.whl`, SHA256 `87a8ab8d78fc6b6069b1d30b0526260b1ce0f9958f4f8e064bb30f574533e755`, installed with `pip --target /tmp/tlx-current`. The build's generated codegen library is available in the installed runtime; it is not a kernel source change.

## Correctness and comparisons

From the repository root:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/tmp/tlx-current \
  adkankar-tlx-opt-20261105 python -m pytest -s --tb=short \
  third_party/tlx/tutorials/amd_pa_decode_fp8.py

docker exec -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/tmp/tlx-current \
  -e FP8_REFERENCE_ROOT=/references/current \
  adkankar-tlx-opt-20261105 python experiments/fp8-kv-decode/run_suite.py \
  --output experiments/fp8-kv-decode/results/new_run
```

Use a new output directory to preserve prior results. `run_suite.py --stages controls native fp8 queries dense_queries lengths long_controls dense_controls windows layout` selects families. The case specification is written before execution. Every backend is checked against independent FP32 attention on the same quantized inputs; `atol=5e-4, rtol=1e-2` is unchanged. Failed numerical gates retain their error and diagnostic latency, with `performance_eligible=false`.

The adapters in `bench_decode.py` remove framework imports and supply platform helpers, without changing the pinned GPU function bodies. Focused launch-tuned vLLM controls additionally change Python tile/descriptor flags, explicitly identified below. TokenSpeed's provider imports bind to this shared compiler. AITER's pinned Python attention code uses the image's compiled core only to support package import. vLLM's per-token-head scale interface receives zero-stride broadcast views of the per-head scalars. Packing/native-layout preparation is outside all timed regions for every implementation.

Scalar-only AITER/SGLang/vLLM-decode comparisons use explicitly labeled shared-scale controls. TokenSpeed uses explicitly labeled full-FP8 unit-scale controls. They are not native unequal-per-head baselines. Full-FP8 references have their own intended accuracy tradeoffs; failing this project's stricter gate does not establish an upstream correctness bug.

Timing uses three function warmups, graph capture, three graph-replay warmups, then GPU events around repeated invocations. Reported p95 is across **graph batch means**, not individual requests. Caller latency includes Python and synchronization and is reported separately. Repeated inputs are a warm-cache protocol; this is not an HBM roofline measurement. Compilation, quantization, input packing and the oracle are excluded.

The dense width8192 cases assume Hq=Hkv=128,D64; the alternate D128 case uses64/64. Folded/query cases assume16/2,D64. T>1 is a noncausal shared-cache query set; both folded16/2 and full-width128/128 T-sweeps are explicit cases. SWA37/128 are working assumptions. Exact model head mapping, T semantics, window and output projection remain undisclosed.

## Focused SGLang/vLLM comparison (step09)

The authoritative [focused report](FRAMEWORK_COMPARISON.md) corrects the earlier2D-only vLLM comparison. [Its manifest](results/2026-11-05_framework_decode/manifest.json) records unchanged TLX hashes and the earlier harness snapshot, plus source/audit identifiers. Original step06 data and its manifest are preserved. The commands below use new output directories and run sequentially on the selected idle agent:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/tmp/tlx-current \
  -e FP8_REFERENCE_ROOT=/references/current adkankar-tlx-opt-20261105 \
  python experiments/fp8-kv-decode/compare_framework_decode.py \
  --output experiments/fp8-kv-decode/results/new_framework/sweep

docker exec -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/tmp/tlx-current \
  -e FP8_REFERENCE_ROOT=/references/current adkankar-tlx-opt-20261105 \
  python experiments/fp8-kv-decode/compare_framework_decode.py \
  --extend-from experiments/fp8-kv-decode/results/new_framework/sweep \
  --output experiments/fp8-kv-decode/results/new_framework/final

docker exec -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/tmp/tlx-current \
  -e FP8_REFERENCE_ROOT=/references/current adkankar-tlx-opt-20261105 \
  python experiments/fp8-kv-decode/compare_framework_decode.py --legacy-fp8-only \
  --extend-from experiments/fp8-kv-decode/results/new_framework/final \
  --output experiments/fp8-kv-decode/results/new_framework/fp8_validated

docker exec -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/tmp/tlx-current \
  -e FP8_REFERENCE_ROOT=/references/current adkankar-tlx-opt-20261105 \
  python experiments/fp8-kv-decode/compare_framework_decode.py --aiter-companion \
  --extend-from experiments/fp8-kv-decode/results/new_framework/final \
  --output experiments/fp8-kv-decode/results/new_framework/framework_routes
```

The script writes tuning rows incrementally, then selects each native source family's fastest passing configuration for three fresh order-rotated rounds. The stock16-segment vLLM path remains visible. `--case gqa2304_per-head` can restrict a run. `--quick` is a pilot and does not reproduce the full sweep. The first historical sweep predates the legacy FP8-Q adapter relaxation; a fresh sweep with the current script tests those controls immediately.

`_3d_tuned` overrides the Python launcher's tile helper; `_kvtd3d`/`_kvtd2d` also disable Q/output descriptors while leaving KV descriptors enabled. These launch controls leave upstream GPU function bodies unchanged and are not production ROCm defaults. AITER companion timings represent one underlying operator, with framework precision/layout gates described in the report.

Regenerate the saved step09 report/CSV/chart without GPU execution (requires matplotlib):

```bash
docker exec adkankar-tlx-opt-20261105 \
  python experiments/fp8-kv-decode/report_framework_decode.py
```

The renderer reads the saved dated directory and preserves provenance links to each selected repeat JSON. To render a new run, change `DATA` and `STEM` in a copied renderer so dated snapshots remain preserved.

## ATT and PMC

The installed skill is `/root/.codex/skills/rocprofv3-att-trace-and-analyze/SKILL.md`; its analyzer is also mounted in the container at `/att-skill`. ATT used container rocprofv3 1.4.1 and its compatible decoder. Example:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/tmp/tlx-current \
  adkankar-tlx-opt-20261105 rocprofv3 --att --att-target-cu 0 \
  --att-simd-select 0 --att-shader-engine-mask 0x1 --att-buffer-size 0x4000000 \
  --kernel-include-regex _matrix_decode --kernel-iteration-range '[2]' \
  -d /workspace/experiments/fp8-kv-decode/results/new_att \
  -- python experiments/fp8-kv-decode/profile_decode.py \
  --variant matrix_tdm_flat --block-n 64 --splits 36 --warps 4

python /root/.codex/skills/rocprofv3-att-trace-and-analyze/scripts/analyze_att.py \
  experiments/fp8-kv-decode/results/new_att --top 15
```

The container initially had duplicate pip ROCm SDK core/devel shared libraries, causing duplicate LLVM option registration and HIP initialization failures under rocprof. Inside **this isolated container**, matching top-level core library filenames were symlinked to the devel copies, with originals preserved as `.att-original`. This repaired ATT. The host SDK and other containers were not changed. Initial failure logs remain in the results directory.

`SQ_INSTS_LDS`, `SQ_WAIT_ANY`, `SQ_INST_CYCLES_VALU_WMMA` were accepted by the counter collector but returned zero in all108 instances for both traced kernels, despite decoded LDS/WMMA instructions. [pmc_summary.json](results/2026-11-05_optimization/pmc_summary.json) and raw rocpd databases retain this limitation. Do not interpret zero as zero hardware activity. Compiled register/shared-memory counts are separate metadata, not utilization counters.

ATT percentages describe sampled instruction cycles for a selected CU/SIMD/SE and dispatch. Different grids execute different amounts of sampled work. Use unprofiled timing for speedups; never treat aggregate stall/latency or total trace cycles as whole-kernel utilization or latency.

## Artifacts

`FP8_KV_DECODE_ATTN.md` is the experiment ledger, including failed attempts and decisions. `snapshots/` preserves rejected implementations. `report_results.py` and `summarize_suite.py` regenerate charts/CSV/Markdown from saved measurements; they do not execute GPUs. Earlier snapshots and the first29-case holdout remain unchanged.

## November9 fixed-configuration three-length repeats

The [repeat report](REPEAT_2026-11-09.md) uses the same environment above, with no new tuning or kernel changes. The benchmark now supports `--repeat-only --extend-from ...` to select prior passing configurations and perform three fresh timing rounds. [Exact commands](results/2026-11-09_repeat_lengths/reproduce.sh) and [manifest](results/2026-11-09_repeat_lengths/manifest.json) are saved beside raw results. Check the selected GPU is idle before running; use fresh output directories to preserve snapshots. The script's copied tuning JSON is explicitly marked inherited and is not a new tuning measurement.

The original user-pasted table's exact metadata could not be recovered, so this rerun explicitly uses GQA16/2,D64,page64,one BF16-Q token with E4M3FN KV. Shared-scale AITER controls and native unequal-head TLX/vLLM cases are separate. The initial run on an already busy agent is preserved and excluded.

Regenerate this saved report/chart without GPU execution:

```bash
docker exec adkankar-tlx-opt-20261105 \
  python experiments/fp8-kv-decode/report_repeat_lengths.py
```

## November9 optimization pass

[Final optimization report](OPTIMIZATION_2026-11-09.md), [ledger](FP8_KV_DECODE_ATTN.md), and [manifest](results/2026-11-09_optimization/manifest.json) describe the retained source and intermediate failures. This pass uses **CUDA_VISIBLE_DEVICES=2**, the same96-CU/144-GiB gfx1250 runtime partition, because GPU0 was busy. A host reboot interrupted the session; final comparisons repeat both frozen and current code together after recovery. Never compare absolute values from different phases as a kernel speedup.

The starting GPU code is preserved in `results/2026-11-09_optimization/frozen`; the final snapshot is `final_source`. `step14_source` and `pre075` preserve the intermediate version before multi-query regression recovery. All reference source hashes remain at the November5 audit pins. No compiler/native changes were required.

Use a fresh output path (the following `new_run` is a placeholder) after checking GPU2 is idle. Run GPU commands sequentially:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=2 -e PYTHONPATH=/tmp/tlx-current \
  -e FP8_REFERENCE_ROOT=/references/current \
  adkankar-tlx-opt-20261105 pytest -s --tb=short \
  third_party/tlx/tutorials/amd_pa_decode_fp8.py

docker exec -e CUDA_VISIBLE_DEVICES=2 -e PYTHONPATH=/tmp/tlx-current \
  -e FP8_REFERENCE_ROOT=/references/current \
  adkankar-tlx-opt-20261105 python experiments/fp8-kv-decode/audit_decode.py \
  --stage full --rounds 3 --samples 15 --repeats 20 \
  --output experiments/fp8-kv-decode/results/new_run/matrix.json

docker exec -e CUDA_VISIBLE_DEVICES=2 -e PYTHONPATH=/tmp/tlx-current \
  -e FP8_REFERENCE_ROOT=/references/current \
  adkankar-tlx-opt-20261105 python experiments/fp8-kv-decode/compare_framework_decode.py \
  --repeat-only --aiter-companion --case gqa2304_shared mha2304_shared d128_shared \
  --extend-from experiments/fp8-kv-decode/results/2026-11-05_framework_decode/framework_routes \
  --output experiments/fp8-kv-decode/results/new_run/framework
```

Regenerate the saved November9 final report, charts and CSV/JSON using CPU only:

```bash
docker exec adkankar-tlx-opt-20261105 \
  python experiments/fp8-kv-decode/report_optimization.py
```

This renderer reads the saved `repeated_final_v2.json`, `framework_final_v2/summary.json` and `correctness_final_v2.xml`. Steps11–14 remain preserved as intermediate comparisons; step15 is the final80-case matrix, step16 the fresh strongest-tested framework controls.
