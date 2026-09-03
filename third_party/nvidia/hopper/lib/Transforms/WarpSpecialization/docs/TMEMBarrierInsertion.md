# TMEM Barrier Insertion

`triton-nvidia-gpu-tmem-barrier-insertion` orders accesses to aliased physical
tensor memory. It normally inserts a CTA barrier for read-after-write,
write-after-read, and write-after-write hazards. A CTA barrier is unnecessary
when the hazard is entirely warp-local because program order and the
`tcgen05.wait::{ld,st}` emitted by lowering already order each warp's accesses.

## Warp-local safety rule

For two candidate accesses, let `A[w]` and `B[w]` be the sets of physical TMEM
words touched by warp `w`. The barrier can be omitted exactly when the accesses
have the same warp scope and:

```
A[i] intersect B[j] is empty for every i != j
```

The regions owned by the same warp do not need to be identical. For example,
`A[0] = [0, 64)` followed by `B[0] = [0, 32)` is safe if no other warp in `B`
touches `[0, 64)`. It is unsafe if, for example, `B[4] = [32, 64)`, because
warp 4 would overwrite words that warp 0 may still be reading.

This rule also permits completely disjoint accesses. It does not require an
overlap to justify removing a barrier; it only rejects cross-warp overlap.

## Address model

The analysis resolves each access to a statically known allocation address,
including offsets introduced by `ttng.tmem_subslice` and transparent
`ttg.memdesc_reinterpret` views. Captures into a warp-specialized partition are
traced back to the captured descriptor.

For each warp, it enumerates every physical word touched by every lowered TMEM
message. The message footprint is the instruction atom composed with its
register-vector repeat factor. This distinction matters when more than four
warps split a tensor along columns: recording only the first atom would
under-approximate the footprint and could miss a real cross-warp hazard.

The pass keeps the CTA barrier conservatively when it cannot resolve the
allocation, layout, warp count, or warp scope, and for MMA or TMEM-copy hazards
whose execution model is not represented as one independent instruction stream
per warp.

## Examples

With four warps split along rows, a full 128-column load followed by a
64-column subslice store can be warp-local:

```
load warp 0:  rows [0, 32), columns [0, 128)
store warp 0: rows [0, 32), columns [0, 64)
```

The store region is a proper subset of the same warp's load region and no
different warp overlaps it.

With eight warps split across rows and columns, the same shapes can require a
CTA barrier:

```
full load warp 0:      rows [0, 32), columns [0, 64)
subslice store warp 4: rows [0, 32), columns [32, 64)
```

The physical intersection is owned by different warps, so warp-local ordering
is insufficient.

## Tests

`test/TritonNvidiaGPU/tmem_barrier_insertion.mlir` covers same-warp equal and
proper-subset regions, the eight-warp cross-owner counterexample, descriptor
reinterpretation, and conservative fallbacks.
