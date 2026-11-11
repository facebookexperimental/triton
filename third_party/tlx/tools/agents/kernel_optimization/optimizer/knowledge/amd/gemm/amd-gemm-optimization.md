# AMD GEMM Optimization

Classify each workload before changing configuration: ordinary two-dimensional
GEMM, small-M, tall-skinny, deep-K, or an irregular boundary case. Use the full
configuration sweep as evidence, not as the complete optimization space.

After two rejected parameter hypotheses, investigate a different work
decomposition. Candidate strategies include one-dimensional row tiling for
tall-skinny matrices, persistent tile scheduling, cooperative reduction for
small-M/deep-K matrices, explicit operand reuse, and a family-specific kernel
with a guarded fallback. Do not keep adding tile variants when profiles show
that launch, indexing, masking, inactive lanes, or repeated loads dominate.

Correlate end-to-end latency with MFMA utilization, memory traffic, occupancy,
registers, spills, and LDS. Inspect TTGIR/LLVM/AMDGCN when the counter evidence
suggests layout conversion, address generation, scheduling, or register
allocation overhead. A structural candidate must state why the existing work
decomposition cannot address the measured bottleneck.
