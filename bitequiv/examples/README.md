# bitequiv examples

Runnable, self-contained examples for the bitwise-equivalence project. Each script
compiles a kernel, prints the IR/PTX evidence for a specific compiler behavior, and
asserts the runtime result on a real GPU.

## layout_numerics/ — passes that CHANGE the bits (project core)

The numerics-modifying examples: data layout, reduction order, MMA precision, FMA
contraction, and approximate elementwise math. These are the behaviors the
equivalence work must reason about. Paired with
`../knowledge-base/numerics-modifying-passes.md` and
`../knowledge-base/tree-reduction-in-ptx-and-triton.md`.

| File | Pass | Bit mechanism |
|------|------|---------------|
| `b01_reduction_tree_from_layout.py` | `ReduceOpToLLVM` | `num_warps` → reduce-axis layout → tree → **bits** |
| `b02_reduction_ordering_inner_tree.py` | `inner_tree` | layout-invariant order → bitwise-equal across `num_warps` |
| `b03_mma_precision.py` | `AccelerateMatmul` | `tt.dot`→`#mma`; tf32 vs ieee → bits *(Blackwell)* |
| `b04_f32_dot_tc.py` | `F32DotTC` | `tf32x3` 3-pass: fp32 accuracy, different bits *(Blackwell)* |
| `b05_fma_contraction.py` | `DotOpToLLVM/FMA` | `a*b+c` fused → one rounding (PTX-level, invisible at TTGIR) |
| `b06_elementwise_math_precision.py` | NVIDIA `ElementwiseOpToLLVM` | `div.full` / `ex2.approx` / `cvt.rn` per-element rounding |

## ../knowledge-base/compilation-pipeline/ — bit-NEUTRAL mechanics (onboarding)

General "how the IR changes through the pipeline" tutorials (TTIR → TTGIR → LLIR →
PTX): layout assignment, coalescing/vectorization, convert_layout elimination,
software pipelining, AutoWS. None of these change the FP math; they are the contrast
to `layout_numerics/`.

## Run

```bash
# one example
python bitequiv/examples/layout_numerics/b01_reduction_tree_from_layout.py
# the layout/numerics + pipeline examples (smoke-tested)
pytest bitequiv/tests/
```

Blackwell-only examples (`b03`, `b04`, and `compilation-pipeline/06`) self-skip on
other hardware. The autotuner constraint-pruning scenarios (static IR/PTX feature
filters + reduction-order equivalence pruning) are assertion-based tests in
`../tests/test_constraint_pruning.py`.
