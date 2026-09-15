# AMD Attention Variant Reference

Use this reference only after the kernel has been identified as attention. Derive behavior from the supplied source and protected cases; the categories below are decision aids, not assumptions about the candidate.

## Semantic Families

- **Dense Flash Attention:** Q/K/V use normalized softmax and online maximum/sum correction. Static grids assign query tiles across batch-heads; persistent grids schedule multiple logical tiles per worker.
- **Mapped cross-attention:** the query work item may select another entity's K/V sequence through a runtime mapping. Keep this mapping separate from grouped-query head mapping.
- **Packed jagged attention:** sequence offsets determine query and KV bounds. A maximum-length grid may contain padding slots that must exit or mask safely.
- **HSTU self-attention:** scores may use SiLU scaling and a fixed normalization factor instead of softmax. Do not introduce online-softmax state into this family.
- **HSTU cross-attention:** Q and KV may use independent offsets, lengths, and dimensions. Derive each domain separately.

Classify forward, dQ, dK, dV, fused, and multi-launch backward phases independently because their ownership, reduction, and publication contracts differ.

## Grid And Bounds

For every active query tile, derive the logical grid, physical or persistent grid, exact KV lower and upper bounds, mask-free block count, boundary or diagonal block count, and reuse across heads, sequences, CTAs, CUs, and XCDs.

For unequal-length causal attention, identify the source's alignment convention explicitly. An end-aligned form commonly uses:

```text
delta = N_kv - M_q
valid(q, k) = k <= q + delta
hi(query_tile) = min(N_kv, query_tile_end + delta)
```

Do not substitute ordinary `k <= q` unless Q and KV domains and the implementation's contract make it equivalent.

## Scheduling Choices

- Start from direct loads or single-slot LDS staging when establishing correctness.
- Add double buffering only after deriving prologue, steady-state, epilogue, slot rotation, wait depth, and short-loop behavior.
- Use deeper or cluster pipelines only when the selected architecture, workload length, and LDS budget support them.
- Use persistent workers when repeated tiles amortize scheduling cost; protect small workloads and imbalanced jagged distributions.
- Separate mask-free interior work from causal, jagged, or tail boundaries when doing so preserves direct global-to-LDS movement and one continuous async ring.
- Tune backward dQ and dK/dV ownership separately. Account for reductions, atomics, accumulator residency, and any preprocessing launch.

## Online Softmax Invariants

For standard softmax attention, every streamed KV block must update the running maximum, denominator, and output accumulator consistently:

```text
m_new = max(m_old, rowmax(scores))
alpha = exp(m_old - m_new)
l_new = l_old * alpha + rowsum(exp(scores - m_new))
acc_new = acc_old * alpha + exp(scores - m_new) @ V
```

Apply log2 scaling consistently when the implementation uses `exp2`, and normalize only after the final valid block. Test large negative logits, fully masked boundaries when supported, short sequences, unequal Q/KV lengths, and non-divisible tails.

## Native MFMA And Resource Tradeoffs

A native scheduled-MFMA design requires matching accumulator and operand layouts, fragment extraction, independent accumulator-chain ordering, commit boundaries, and register residency. Port that contract as a unit. A layout or residency choice measured on one AMD architecture, head dimension, or ownership scheme is not a universal rule.

Higher unroll or prefetch depth can win long loops while raising registers or spills. Select it from correctness and stable latency over the protected distribution, not from one resource counter. Preserve a lower-resource fallback when short shapes or another dtype cannot sustain the same pipeline.
