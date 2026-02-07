# Key Concepts
Compilation Flow: Python DSL → TTIR (Triton IR) → TTGIR (Triton GPU IR) → LLVM IR → PTX/AMDGPU

# CRITICAL: Always rebuild after modifying C++ code:
```bash
pip install -e . --no-build-isolation
```
C++ changes require recompilation to take effect. Python-only changes do not.

# CRITICAL: Always run formatter after modifying code:
```bash
pre-commit run --all
```
# Testing Workflow

## Correctness First

Always validate correctness before anything else.

- Run all tests: `pytest third_party/tlx/tutorials/testing/test_correctness.py`
- Run a single kernel: `pytest third_party/tlx/tutorials/testing/test_correctness.py::test_<kernel_name>`

Available kernels: `blackwell_gemm_ws`, `blackwell_gemm_clc`, `blackwell_gemm_pipelined`, `blackwell_gemm_2cta`, `blackwell_fa_ws`, `blackwell_fa_ws_persistent`, `blackwell_fa_ws_pipelined`, `blackwell_fa_ws_pipelined_persistent`, `hopper_gemm_pipelined`, `hopper_gemm_ws`, `hopper_fa_ws`, `hopper_fa_ws_pipelined`, `hopper_fa_ws_pipelined_pingpong`, `hopper_fa_ws_pipelined_pingpong_persistent`

- For other kernels: `pytest third_party/tlx/tutorials/<KERNEL.py>`

## Performance Testing

**Never run performance tests unless explicitly asked.**

When performance testing is requested:

1. Run `nvidia-smi` to check GPU occupancy.
2. Pick the GPU with the lowest memory usage.
3. Set `CUDA_VISIBLE_DEVICES` to that GPU.
4. Run the appropriate denoise-wrapped benchmark:

**Hopper GPU:**
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_hopper_gemm_perf.py [--version {ws|pipelined}]
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_hopper_fa_perf.py [--version {ws|ws_pipelined|ws_pipelined_pingpong|ws_pipelined_pingpong_persistent}]
```

**Blackwell GPU:**
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_blackwell_gemm_perf.py [--version {ws|pipelined|clc|2cta}]
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_blackwell_fa_perf.py [--version {ws|ws_pipelined|ws_pipelined_pingpong|ws_pipelined_pingpong_persistent}]
```

**Other kernels:**
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/<KERNEL.py>
```

# CRITICAL: Run killgpu.sh
Run `third_party/tlx/killgpu.sh` to kill if any test runs a few minutes

# Commit messages
Don't commit unless the user explicitly asks you to.
When writing a commit message, don't make a bullet list of the individual
changes. Instead, if the PR is large, explain the order to review changes
(e.g., the logical progression), or if it's short just omit the bullet list
entirely.
Disclose that the PR was authored with Claude.
