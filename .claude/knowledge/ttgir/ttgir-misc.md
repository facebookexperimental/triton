# TTGIR Miscellaneous Ops

Ops that don't fit neatly into data transfer, memory layout, tensor cores,
synchronization, or control flow categories.

## `ttg.fp4_to_fp` — FP4 to Floating-Point Conversion

Converts a tensor of FP4 (4-bit floating point) values to a wider floating-point
type (e.g., fp16, bf16, fp32). Pure operation with no memory effects.

**Dialect**: TritonGPU (`ttg.`)

```mlir
%result = ttg.fp4_to_fp %input : tensor<128x64xf4, #blocked> -> tensor<128x64xf16, #blocked>
```

**Use case**: Block-scaled GEMM (MX formats) where weights are stored in FP4
and must be upcast before MMA. Typically appears after loading FP4 data from
shared memory and before feeding it into a dot operation.

**Hardware**: Hopper and Blackwell. On Blackwell, `tc_gen5_mma_scaled` can
consume FP4 directly via hardware-supported block scaling, potentially
eliminating the need for explicit conversion.

## `ttg.clock64` — GPU Clock Counter Read

Reads the GPU's 64-bit hardware clock counter. Maps to PTX `clock64` /
`%globaltimer`. Returns an `i64` value.

**Dialect**: TritonGPU (`ttg.`)

```mlir
%cycles = ttg.clock64 : i64
```

**Memory effects**: Marked with `MemRead` and `MemWrite` on `DefaultResource`
to prevent the compiler from reordering, hoisting, or dead-code-eliminating
clock reads. This ensures timing measurements reflect the actual execution
order.

**Use case**: Cycle-level profiling and debugging inside kernels — e.g.,
measuring how many cycles an MMA operation, a barrier wait, or a TMA transfer
takes. Often used in pairs (before/after) with the difference giving elapsed
cycles.
