# Multi-CTA grid requirements

## Summary

This document complements the
[multi-CTA control-flow contract](multi_cta_control_flow_contract.md) and the
[AutoWS requirements](multi_cta_autows_requirements.md). It defines two
additional pieces of the multi-CTA correctness contract:

1. a launcher-side verification process for checking that a launch grid is
   compatible with the compiled kernel's required cluster shape; and
2. rules for handling tiles that have no logical work but must exist to complete
   a multi-CTA participation group.

Cluster verification and dead-tile safety are separate proofs. A divisible grid
ensures that CUDA can form complete physical clusters. It does not prove that a
padded CTA uses safe data or follows the required collective trace. Conversely,
a kernel with safe dead-tile behavior is not launchable if the physical grid
cannot be partitioned into its required clusters.

For non-persistent kernels, each launched CTA has one statically determined tile
assignment, so launch-grid verification also constrains the tile schedule. For
persistent kernels, per-dimension divisibility is still required. The derived
flat-grid property that the total number of launched CTAs is divisible by
`Cx * Cy * Cz` makes explicit that a whole number of clusters is scheduled, but
it is not an alternative verification rule. Correctness of the dynamic tile
assignments remains a separate kernel-level scheduler obligation.

## Terminology and running example

Let:

- `G = (Gx, Gy, Gz)` be the final physical CTA grid passed to CUDA; and
- `C = (Cx, Cy, Cz)` be the exact required cluster shape.

For `ctas_per_cga`, `G` is already the total CTA grid. Other frontend options
may express a grid in logical programs and expand it before launch. Verification
always operates on the final physical CTA grid, after any such expansion.

A **dead tile** is a physical tile assignment with no corresponding logical
output. Dead tiles are schedule padding, not partially valid edge tiles. For
example, a 2-CTA GEMM grouped along X requires an even number of X tiles. If the
logical M-tile count is three, the kernel must launch four X CTAs:

```text
logical tiles:   [0] [1] [2]
physical CTAs:   [0] [1] [2] [dead]
2-CTA groups:     \___/     \_______/
```

The last CTA has no logical output, but it is still the second participant in
the cooperative operation for the final pair.

## Component 1: Launcher-side cluster verification

### Why the check belongs at launch

The compiler knows the required cluster shape, but a Triton launch grid may be a
runtime function of tensor shapes and autotuning parameters. Compilation alone
therefore cannot prove that every invocation supplies a compatible grid.

CUDA ultimately rejects invalid cluster launches, but relying on a driver error
provides a late and implementation-dependent diagnostic. Triton should verify
the contract before allocating launch-scoped resources or invoking CUDA.

### Required launch metadata

A compiled kernel containing a multi-CTA operation must expose one normalized,
exact cluster requirement to the launcher:

- the required physical cluster shape `C`;
- whether the user-provided grid already counts physical CTAs or must first be
  expanded; and
- whether the kernel requires exact clustering.

Preferred or fallback cluster shapes, for which the driver may select a
different physical shape, are outside this contract.

All multi-CTA operations in one kernel must be compatible with that requirement.
The compiler must reject or avoid forming an operation that requires a different
physical cluster shape.

### Verification process

For every launch of a kernel with an exact cluster requirement, the launcher
must perform the following checks:

1. Verify per-dimension divisibility:

   ```text
   Gx % Cx == 0
   Gy % Cy == 0
   Gz % Cz == 0
   ```

   For a persistent kernel, the corresponding kernel-granularity requirement
   is:

   ```text
   (Gx * Gy * Gz) % (Cx * Cy * Cz) == 0
   ```

   This flat-count check makes explicit that the launch contains an integral
   number of schedulable clusters. It does not replace the per-dimension checks
   when the required cluster shape is multidimensional.
2. Verify that the target device and launch resources support the required
   cluster shape. The CUDA driver remains the final authority, but the launcher
   should report the incompatibility as a cluster-requirement error rather than
   an unexplained launch failure.

The launcher must not silently round `G`. Rounding creates new CTA program IDs,
which is a semantic transformation: the kernel must map them to dead tiles,
make their memory accesses safe, suppress their side effects, and keep them on
the collective trace. The launcher cannot infer those properties from cluster
metadata.

Failure of either check is a launch-time error for the selected compiled
kernel. The diagnostic must identify the physical grid `G` and required cluster
shape `C`, along with any known target or resource incompatibility. Kernel
caching, recompilation, and selection of a different compiled variant are
outside this contract and must not be relied upon to bypass these checks.

### What this proves

Successful verification proves only that the physical grid can be partitioned
into complete clusters of the required shape. Because the compiler has already
proved that each multi-CTA participation group is contained within that shape,
no group is truncated at the edge of the launch grid.

It does not prove:

- that the logical tile count fills the physical grid;
- that padded program IDs are mapped to safe dead tiles;
- that dead-tile loads, stores, atomics, or other side effects are safe; or
- that participating CTAs execute compatible control flow.

Those are kernel and compiler obligations described below and in the
multi-CTA control-flow contract.

## Component 2: Dead-tile handling

### Dead-tile execution

Ordinary masking and dead-tile participation are different:

- A **partially valid tile** has some logical output. The kernel executes the
  tile and masks individual out-of-bounds elements.
- A **dead tile** has no logical output. It exists only to complete a physical
  cluster or multi-CTA participation group.

Most kernels handle partially valid edge tiles, but that does not imply that
they safely handle a fully dead tile. A kernel may fail when an operation
receives a tile with no in-bounds elements. For example, a TMA store can crash
when the tile's lower bound lies past the tensor's upper bound. It is currently
unclear whether this is an implementation bug or an API gap.

### Kernel-author responsibilities

For the initial design, the kernel author or launch wrapper must:

- round the logical grid to complete physical clusters;
- map padded program IDs to dead tiles without wrapping them onto unrelated
  live work;
- provide safe values for every load performed by a dead tile, using descriptor
  out-of-bounds fill or explicit masks as appropriate;
- ensure padded values are semantically harmless for the collective operation;
- suppress every externally visible dead-tile effect, including output stores,
  atomics, counters, queue updates, and debugging writes; and
- structure control flow and tile-validity handling so the compiler can prove
  that dead CTAs remain on the same ordered multi-CTA and barrier trace as their
  live partners.

The author must not assume that output masking makes input accesses or
intermediate side effects safe.

### Compiler support

Kernel constructs such as CTA-dependent early exits can prevent the compiler
from proving that every participant reaches an operation that could otherwise
use multiple CTAs. When an author introduces such constructs, additional kernel
structure or compiler-visible information may be required before the compiler
can safely form a multi-CTA operation. Until that participation can be proved,
the affected operation must remain single-CTA.

### 2-CTA GEMM example

For a non-persistent GEMM grouped along M, the launch wrapper can construct a
complete grid as follows:

```python
logical_m_tiles = triton.cdiv(M, BLOCK_M)
grid_m = triton.cdiv(logical_m_tiles, Cx) * Cx
```

The wrapper must apply the corresponding rounding to every clustered grid
dimension, using `Cy` and `Cz` for Y and Z.

When `logical_m_tiles` is odd, the last CTA is dead. Both CTAs in its pair still
execute the same K-loop iterations, descriptor loads, 2-CTA MMA operations, and
barrier epochs. The dead CTA's input loads return safe padding, and an explicit
output mask prevents it from writing outside the output tensor.

This is safe:

```text
both CTAs: load -> arrive/wait -> 2-CTA MMA -> repeat
dead CTA:  suppress final output writes
```

This is not safe:

```text
live CTA:  load -> arrive/wait -> 2-CTA MMA
dead CTA:  return or continue to the next tile
```

### Expected failure modes

| Contract violation | Likely symptom |
|---|---|
| Physical grid is not divisible by the exact cluster shape | Launcher cluster-requirement error before CUDA is invoked; without this verification, CUDA rejects the launch |
| A dead CTA returns or skips a collective operation | Barrier wait that never completes or a kernel hang |
| A dead AutoWS partition skips a ready/done event | Channel epoch drift, stale buffers, or an intermittent hang |
| Dead-tile inputs are not padded or masked | Out-of-bounds access, illegal memory access, or invalid MMA data |
| Padding values are not neutral for the operation | Silent corruption of a live partner's result |
| Dead-tile stores, atomics, or counters are not suppressed | Out-of-bounds writes, duplicated work, or corrupted scheduler state |

The most dangerous failures are not necessarily immediate launch errors. A
missing barrier arrival or mismatched AutoWS epoch can depend on scheduling and
appear as an intermittent hang, while a non-neutral padded operand can silently
corrupt otherwise in-bounds output.
