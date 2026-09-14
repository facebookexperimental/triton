---
globs:
  - "third_party/tlx/tutorials/**"
---

# TLX Tutorial Kernels

Python-only: no rebuild needed. Each kernel file is self-contained with its own test harness.

## Correctness testing
- All kernels: `pytest third_party/tlx/tutorials/testing/test_correctness.py`
- Single kernel: `pytest third_party/tlx/tutorials/testing/test_correctness.py::test_<kernel_name>`

Available kernels: `blackwell_gemm_clc`, `blackwell_gemm_pipelined`, `blackwell_gemm_2cta`, `blackwell_fa_ws`, `blackwell_fa_ws_persistent`, `blackwell_fa_ws_pipelined`, `blackwell_fa_clc`, `hopper_gemm_pipelined`, `hopper_gemm_ws`, `hopper_fa_ws`, `hopper_fa_ws_pipelined`, `hopper_fa_ws_pipelined_pingpong`, `hopper_fa_ws_pipelined_pingpong_persistent`

One smoke case per kernel -- a sanity check, not the authoritative suite. Ops
promoted into `tlx.ops` (`mm`, `flash_attn`, `hstu_attn`, `kimi_delta_attention`)
are tested in `python/test/unit/tlx_ops/` instead.

- For other kernels: `pytest third_party/tlx/tutorials/<KERNEL.py>`

## Performance testing

**Never run performance tests unless explicitly asked.**

Performance testing: use the `kernel-perf-testing` skill.
