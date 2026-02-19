---
globs:
  - "lib/**"
  - "include/**"
---

# Core Triton Compiler (C++)

MUST rebuild after changes: `pip install -e . --no-build-isolation`

## Testing
- `pytest python/test/unit/language/`

## Key subsystems
- `lib/Analysis/` — alias analysis, memory allocation, axis info
- `lib/Conversion/TritonToTritonGPU/` — TTIR → TTGIR lowering
- `lib/Conversion/TritonGPUToLLVM/` — TTGIR → LLVM lowering
- `lib/Dialect/Triton/` — TTIR dialect ops and transforms
- `lib/Dialect/TritonGPU/` — TTGIR dialect, pipelining, warp specialization
- `lib/Dialect/TritonNvidiaGPU/` — NVIDIA-specific passes (TMEM, TMA, fences)
- `lib/Tools/` — LinearLayout, swizzling utilities
