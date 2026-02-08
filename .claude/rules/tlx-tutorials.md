---
globs:
  - "third_party/tlx/tutorials/**"
---

# TLX Tutorial Kernels

Python-only: no rebuild needed. Each kernel file is self-contained with its own test harness.

## Correctness testing
- All kernels: `pytest third_party/tlx/tutorials/testing/test_correctness.py`
- Single kernel: `pytest third_party/tlx/tutorials/testing/test_correctness.py::test_<kernel_name>`

Available kernels: `blackwell_gemm_ws`, `blackwell_gemm_clc`, `blackwell_gemm_pipelined`, `blackwell_gemm_2cta`, `blackwell_fa_ws`, `blackwell_fa_ws_persistent`, `blackwell_fa_ws_pipelined`, `blackwell_fa_ws_pipelined_persistent`, `hopper_gemm_pipelined`, `hopper_gemm_ws`, `hopper_fa_ws`, `hopper_fa_ws_pipelined`, `hopper_fa_ws_pipelined_pingpong`, `hopper_fa_ws_pipelined_pingpong_persistent`

- For other kernels: `pytest third_party/tlx/tutorials/<KERNEL.py>`

## Performance testing

**Never run performance tests unless explicitly asked.**

Performance testing: use the `kernel-perf-testing` skill.
