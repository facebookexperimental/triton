# Rule: measuring TLX GEMM perf — use the benchmark-harness skill

When benchmarking or profiling the TLX addmm/bmm templates here (`third_party/tlx/language/tlx/inductor/*.jinja`, `registry.py`) vs rocBLAS / split-K / Stream-K on AMD gfx950, load the **`gemm-perf-benchmark-harness`** skill FIRST (full text: `fbcode/inference_acceleration/ops/.claude/skills/gemm-perf-benchmark-harness/SKILL.md`).

It exists so you don't reproduce past perf results from scratch. The one rule you must not forget:

**RULE #0 — do NOT judge TLX-vs-rocBLAS from an isolated eager/JIT `torch.profiler` or `do_bench` wall-clock.** They under-measure production hipBLASLt by ~15-40% (measured: rocBLAS `addmm 1024x1024x12288` = 40us isolated vs 47us in production AOTI kineto) and give the WRONG winner. Use the **production AOTI kineto** (`mts_gpu_benchmark --gpu-trace --op-level-profiling`, local `/tmp/libkineto_activities_<pid>.json`) or the **autotune `do_bench`** from the AUTOTUNE logs. Note rocBLAS/hipBLASLt is itself a split-K (`Cijk_..._PostGSU8_...` reducer) on undersaturated high-K shapes.

The skill has the exact mts + JIT commands, the kineto parse snippet, the `sl cat -r <rev>` template-swap trick, and ground-truth numbers.
