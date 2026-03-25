# TTGIR Miscellaneous Ops

## `ttg.fp4_to_fp`
Converts FP4 tensor to wider float type (fp16/bf16/fp32). Used for MX-format
GEMM where FP4 weights need upcasting before MMA. On Blackwell,
`tc_gen5_mma_scaled` can consume FP4 directly, potentially eliminating this op.

## `ttg.clock64`
Reads the 64-bit GPU hardware clock counter (PTX `clock64` / `%globaltimer`).
Marked with memory effects to prevent reordering/DCE. Used for cycle-level
profiling inside kernels.
