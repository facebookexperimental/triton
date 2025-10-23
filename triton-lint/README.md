# Triton-Lint - Static Analysis Tool for Triton Kernels
Triton-Lint detects performance anti-patterns and compiler regressions in Triton kernels through multi-level analysis. This is a proof-of-concept.

### AST Analyzer
Checks for simple authoring-time misses, such as:
- Missing `@triton.autotune` decorators
- Hardcoded block sizes
- Scalar memory access patterns
- Missing masks in memory operations

### TMA Analysis (Compiler)
- Verifies TMA operations inserted correctly
- Identifies missed TMA opportunities

### Register Spill Analysis (Compiler)
- Analyzes PTX to track spills to Python source
- Detects excessive spills


## Build and run
```bash
# Build the C++ MLIR pass
# assumes Triton build/ exists in parent dir
rm -rf build/ install/ lib/build lib/install && ./build_now.sh

# run triton-lint
python3 triton_lint_simple.py --verify-tma --analyze-spills examples/good_tma_kernel.py

================================================================================
Register spill analysis
================================================================================

File: examples/good_tma_kernel.py
Target: cuda:90

Found 3 test function(s) - running to compile kernels...
  Running test_matmul_kernel_block_ptr...   Running test_matmul_with_tma_descriptors...   Running test_optimized_matmul...
Found 2 kernel(s): matmul_kernel_block_ptr, matmul_with_tma_descriptors

Analyzing: matmul_kernel_block_ptr
  Extracting PTX from cache...
  Analyzing for spills...

   No register spills detected! Kernel is well-optimized.

Analyzing: matmul_with_tma_descriptors
  Extracting PTX from cache...
  Analyzing for spills...

   INFO: Register spills detected: 28 bytes local memory
    Local memory is ~100x slower than registers. Detected 320 spill instructions (128 stores, 192 loads)
    Pattern: balanced - Temporary spills in loops

  HOTSPOTS (locations with most spills):


  1. /data/users/pka/triton/tritlint/examples/good_tma_kernel.py:133:0
     320 register spills at this location
     Spill details: 128 stores, 192 loads, 1280 bytes

     Suggestions:
     • Reduce num_stages to decrease register pressure from pipelining
     • Break kernel into smaller functions

All kernels passed spill analysis (no errors)

============================================================
TMA Analysis
============================================================

File: examples/good_tma_kernel.py
Target: cuda:90

Found 3 test function(s) - running to compile kernels...
  Running test_matmul_kernel_block_ptr...   Running test_matmul_with_tma_descriptors...   Running test_optimized_matmul...
Found 2 kernel(s): matmul_kernel_block_ptr, matmul_with_tma_descriptors

Verifying: matmul_kernel_block_ptr

  Extracting IR from cache...
  Verifying TMA...
    TMA ops: 0
    Regular ops: 0
    Missed: 0
    Errors:
      - TMA DSL REGRESSION: Kernel uses TMA APIs (make_tensor_descriptor) but compiled IR contains NO TMA op

Verifying: matmul_with_tma_descriptors

  Extracting IR from cache...
  Verifying TMA...
    TMA ops found: 9

TMA Operations Found in Kernel Code


 Function: unknown

    ttng.async_tma_copy_global_to_local
     Total: 8 operation(s), 2 unique location(s)
     - /data/users/pka/triton/tritlint/examples/good_tma_kernel.py:128:24
     - /data/users/pka/triton/tritlint/examples/good_tma_kernel.py:129:24

    ttng.async_tma_copy_local_to_global
     Total: 1 operation(s), 1 unique location(s)
     - /data/users/pka/triton/tritlint/examples/good_tma_kernel.py:145:53

Total: 9 TMA operations, 3 unique locations, 1 function(s)

Some kernels failed TMA verification


============================================================
Fast AST linting (no compilation)
============================================================

examples/good_tma_kernel.py:14:0: warning: missing-autotune
    Kernel 'matmul_kernel_block_ptr' lacks @triton.autotune decorator
    Suggestion: Add @triton.autotune([triton.Config(...)]) to tune performance parameters
examples/good_tma_kernel.py:60:12: warning: missing-mask
    tl.load() may need masking for out-of-bounds safety
    Suggestion: Add mask= parameter: tl.load(..., mask=your_mask)
examples/good_tma_kernel.py:61:12: warning: missing-mask
    tl.load() may need masking for out-of-bounds safety
    Suggestion: Add mask= parameter: tl.load(..., mask=your_mask)
examples/good_tma_kernel.py:80:4: warning: missing-mask
    tl.store() may need masking for out-of-bounds safety
    Suggestion: Add mask= parameter: tl.store(..., mask=your_mask)
examples/good_tma_kernel.py:84:0: warning: missing-autotune
    Kernel 'matmul_with_tma_descriptors' lacks @triton.autotune decorator
    Suggestion: Add @triton.autotune([triton.Config(...)]) to tune performance parameters
examples/good_tma_kernel.py:403:12: warning: missing-mask
    tl.load() may need masking for out-of-bounds safety
    Suggestion: Add mask= parameter: tl.load(..., mask=your_mask)
examples/good_tma_kernel.py:404:12: warning: missing-mask
    tl.load() may need masking for out-of-bounds safety
    Suggestion: Add mask= parameter: tl.load(..., mask=your_mask)
examples/good_tma_kernel.py:417:4: warning: missing-mask
    tl.store() may need masking for out-of-bounds safety
    Suggestion: Add mask= parameter: tl.store(..., mask=your_mask)

8 issues found (8 warnings)
```

## Writing lint-able Kernels
Add `test_` functions to your kernel file:

```python
@triton.jit
def my_kernel(a_ptr, b_ptr, M, N, BLOCK: tl.constexpr):
    ...
# Add test function for TMA verification
def test_my_kernel():
    ...
    my_kernel[grid](a, b, M, N, BLOCK=128)
```


## Architecture

```
┌─────────────────────────────────────────┐
│  Python Source Code                     │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴─────────┐
    ↓                    ↓
┌─────────┐      ┌──────────────┐
│   AST   │      │   Compile    │
│ Analyze │      │   (Triton)   │
│ (Fast)  │      │              │
└─────────┘      └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │  MLIR (TTGIR)│
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │  Analysis    │
                 │  (C++ Pass)  │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │  PTX         │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │  PTX analysis│
                 └──────────────┘

```
