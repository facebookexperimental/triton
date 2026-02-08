# Codebase Architecture

## Compilation Pipeline
Python DSL → TTIR (Triton IR) → TTGIR (Triton GPU IR) → LLVM IR → PTX/AMDGPU

## Subsystems
- **TLX DSL** (`third_party/tlx/language/tlx/`): Python frontend for low-level GPU primitives
- **TLX Dialect** (`third_party/tlx/dialect/`): MLIR dialect (C++/TableGen) for TLX ops
- **TLX Tutorials/Kernels** (`third_party/tlx/tutorials/`): Reference kernel implementations (Hopper/Blackwell GEMM and Flash Attention variants)
- **Core Triton compiler** (`python/triton/compiler/`, `lib/`, `include/`): TTIR and TTGIR lowering
- **NVIDIA backend** (`third_party/nvidia/`): PTX codegen, CUDA-specific passes
- **AMD backend** (`third_party/amd/`): AMDGPU codegen
- **Gluon** (`python/triton/experimental/gluon/`): Experimental high-level abstraction layer (upstream-synced, do not modify)

## Glossary
- **CTA**: Cooperative Thread Array (= thread block). A cluster groups multiple CTAs.
- **SMEM**: Shared memory — fast on-chip memory shared within a CTA
- **TMEM**: Tensor memory — Blackwell-only memory for MMA accumulators and scales
- **TMA**: Tensor Memory Accelerator — hardware unit for async bulk copies between global and shared memory
- **wgmma**: Warp Group Matrix Multiply-Accumulate — Hopper+ tensor core instruction
- **mbarrier**: Memory barrier — SMEM-allocated async barrier for producer-consumer sync
- **Named barrier**: Hardware-allocated barrier (indices 0-15), no SMEM needed
- **CLC**: Cluster Launch Control — Blackwell hardware for dynamic persistent kernels with work stealing
- **WS**: Warp Specialization — partitioning warps into producer/consumer roles via `tlx.async_tasks`
- **FA**: Flash Attention
- **GEMM**: General Matrix Multiply

## Debugging & IR Inspection

For IR debugging env vars (`TRITON_KERNEL_DUMP`, `MLIR_ENABLE_DUMP`, etc.),
Claude will load the `ir-debugging` skill when needed.

# Path-Scoped Rules

Subsystem-specific rules (rebuild requirements, test commands, reference docs)
live in `.claude/rules/` and load automatically based on which files are being
edited. See those files for context relevant to each subsystem.

# Development Workflow

## CRITICAL: Always rebuild after modifying C++ code:
- `pip install -e . --no-build-isolation` or `make dev-install-llvm`

C++ changes require recompilation to take effect. Python-only changes do not.

## CRITICAL: Always run formatter after modifying code:
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

Use the `kernel-perf-testing` skill for benchmark commands.

# CRITICAL: Run killgpu.sh
Run `third_party/tlx/killgpu.sh` to kill if any test runs a few minutes

# Commit messages
Don't commit unless the user explicitly asks you to.
When writing a commit message, don't make a bullet list of the individual
changes. Instead, if the PR is large, explain the order to review changes
(e.g., the logical progression), or if it's short just omit the bullet list
entirely.

Don't overwrite existing commits.

Disclose that the PR was authored with Claude.
