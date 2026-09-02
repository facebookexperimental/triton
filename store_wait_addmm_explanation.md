# Store-Wait Pipeline and the Addmm Regression

## Store-wait pipeline

A high-level descriptor store cannot write registers directly to global memory
through TMA. Early lowering converts it into:

```text
result in registers/TMEM
  -> local_store into an SMEM staging slot
  -> asynchronous TMA SMEM-to-global copy
  -> store_wait before that SMEM slot is reused
```

This is implemented in
`third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSTMAStoreLowering.cpp`.
The wait is initially tied to the token returned by the TMA copy.

For a staging ring with `K` copies, the memory planner creates:

```text
staging[K][tile_shape]
```

Stores rotate through the slots:

```text
store 0 -> slot 0
store 1 -> slot 1
...
store K -> slot 0 again
```

The token wait can therefore move from immediately after a store to immediately
before its slot is reused. It eventually becomes:

```text
cp.async.bulk.wait_group(K - 1)
```

with a final `wait_group(0)` drain. This overlap is what the later commits in
the stack implement for FA backward dQ, dK, and dV.

## Memory-planner ordering

The memory planner divides shared-memory buffers into several classes:

| Priority | Buffer class |
|---|---|
| `P0` | Innermost-loop TMA input operands |
| `P1` | Other innermost-loop buffers |
| `P2` | Inner TMA output staging, such as dQ |
| `P3` | Outer TMA output staging, such as dK/dV |
| `P4` | Other epilogue buffers |

Before `cb3ded12e`, allocation effectively happened in this order:

```text
1. Give input operands additional pipeline copies.
2. Use the remaining SMEM for output staging copies.
```

That made addmm correct, but FA backward often left dQ, dK, and dV
single-buffered because smaller input and metadata buffers had already consumed
the budget.

`cb3ded12e` moved `increaseFusedEpilogueCopies()` before the general P0/P1
copy increase:

```text
1. Reserve dQ/dK/dV output staging copies.
2. Give input operands whatever copies still fit.
```

The call now occurs in `WSMemoryPlanner.cpp` before the general Phase 4 copy
increase.

## Why addmm broke

The failing addmm configuration uses:

```python
num_stages = 3
DATA_PARTITION_FACTOR = 2
disallow_acc_multi_buffer = True
EPILOGUE_SUBTILE = 2
```

It has three relevant kinds of SMEM allocation:

```text
A operand ring
B operand ring
C output staging ring
```

Before the problematic planner ordering, the effective allocation was:

```text
A: 3 copies
B: 3 copies
C: 1 copy
```

This passes.

After `cb3ded12e`, output staging received copies first:

```text
A: 3 copies
B: 2 copies
C: 2 copies
```

The two-copy B ring is insufficient for this three-stage, data-partitioned
schedule. The producer can wrap around and overwrite a B slot while one MMA
partition is still consuming it.

The allocation interaction was confirmed with fresh-cache ablations:

```text
TRITON_WS_STAGING_COPIES=1:
A=3, B=3, C=1 -> PASS

TRITON_WS_STAGING_COPIES=2:
A=3, B=2, C=2 -> 49.5% wrong
```

Changing which input received the third copy only moved the race:

```text
A=2, B=3, C=3 -> still wrong
```

Roughly half the output was corrupted because one of the two data-partitioned
computation paths observed overwritten operand tiles. This was not a tolerance
problem or an incorrect C store wait.

## The fix

The fix identifies TMA operands that:

1. Feed an MMA.
2. Are inside a loop with `data_partition_factor > 1`.
3. Are inside a loop with `tt.disallow_acc_multi_buffer`.

Those operands receive a hard minimum of `numBuffers` before output staging is
reserved:

```cpp
buf.minCopies = std::max(buf.minCopies, numBuffers);
buf.numCopies = std::max(buf.numCopies, buf.minCopies);
```

The resulting order is conceptually:

```text
1. Reserve correctness-required operand depths.
2. Reserve dQ/dK/dV staging from the remaining budget.
3. Distribute remaining discretionary copies.
```

This condition is intentionally narrow. Pinning every MMA operand fixed addmm
but made 24 FA tests exceed the 232,448-byte SMEM limit. FA does not set
`disallow_acc_multi_buffer`, so its ordinary operands remain discretionary and
dQ/dK/dV retain their rotating store-wait staging.

This is a targeted correctness backstop, not the final general model. The real
invariant is:

> A producer must not reuse a buffer slot until every consumer of the previous
> generation assigned to that slot has completed its release.

The current condition recognizes the configuration that exposed the bug, but
similar hazards may exist in:

- A data-partitioned MMA that does not use `disallow_acc_multi_buffer`.
- A scaled MMA or another operation with multiple asynchronous consumers.
- Multiple MMAs sharing one TMA-loaded operand.
- A non-DP pipeline whose producer and consumer appear in the same reported
  stage but overlap across loop iterations.
- A circular reuse group whose members have underestimated individual floors.

## Current `minCopies` model

The generic SMEM correctness floor is computed by
`getSmemCrossStageDepth()`. It examines the `loop.stage` values of the actual
consumers and returns:

```text
maxConsumerStage - minConsumerStage + 1
```

This captures buffers consumed across multiple reported stages. It does not
fully capture:

- The producer-to-first-consumer distance.
- The last release of a generation.
- Overlap introduced when a pipeline is expanded across iterations.
- Multiple partitions progressing at different rates.
- Circular-reuse staggering between different logical buffers.

In the failing addmm case, the consumers did not expose a large enough
cross-stage span, so the generic floor remained below the three copies required
by the generated pipeline.

## FA-forward K/V circular reuse

FA forward currently has a separate effective protection through the generic
cross-stage floor and circular-group construction:

```text
v.minCopies = 2
k.minCopies = 1
```

When SMEM circular reuse groups K and V, the group starts at:

```text
groupCopies = 2 * max(k.minCopies, v.minCopies) - 1
            = 2 * 2 - 1
            = 3
```

K and V therefore share one three-slot circular buffer. This is checked by
`test/Hopper/WarpSpecialization/ws_memory_planner_fwd.mlir`.

Without circular reuse, the expected allocation is:

```text
v: 2 copies
k: 1 copy
```

With circular reuse:

```text
k/v shared ring: 3 copies
```

This works because V's cross-stage lifetime is visible to the current analysis.
It does not prove that all future circular-reuse cases are covered. If neither
member exposes the true cross-iteration lifetime through its consumer stage
span, the group floor can still be underestimated.

## Memory-planner search mode

Search mode already has mechanisms for rejecting candidate plans:

- `Packer::legalJoin` rejects illegal reuse groups.
- `Packer::feasible` rejects plans outside the SMEM or TMEM budget.
- The copy solver starts from correctness floors such as `stageSpan` and group
  entry count.
- Beam search drops infeasible branches and tries other groupings.

For SMEM, the copy solver currently uses a floor equivalent to:

```text
floor = max(stageSpan, numberOfEntries, 1)
```

It then adds discretionary copies according to a latency benefit based on the
estimated pipeline initiation interval:

```text
benefit = min(copies * estimatedII, producerLatency)
```

The exact addmm configuration contains a generated subtiled region, which is
currently an unmodeled search feature. Search mode therefore falls back to the
heuristic planner for this case. Nevertheless, a similar kernel that reaches
search mode has the same modeling gap: the plan can satisfy budget, grouping,
and `stageSpan` constraints while still reusing a slot too early.

### Why estimated II is not a correctness proof

Estimated II is appropriate for ranking performance plans. It estimates how
much latency an additional copy may hide. It is not sufficient to prove buffer
reuse safety because runtime relative progress can change due to:

- TMA latency variation.
- MMA and memory-system stalls.
- Different warp partitions progressing at different rates.
- Barrier contention and backpressure.
- One consumer partition lagging behind the others.

Even an accurate average II does not prove that the last consumer releases a
slot before a later producer overwrites it. II must remain an optimization input,
not a correctness predicate.

## Plan: static correctness validator

The robust solution is a static validator shared by heuristic and search
planning. It should prove slot-reuse safety from scheduled events and
happens-before relations rather than estimated execution time.

### 1. Normalize each channel into generation events

For every planned SMEM buffer, collect:

- The logical producer operation.
- Every actual consumer operation and consumer partition.
- Producer acquire and commit points.
- Consumer wait and release points.
- Enclosing loop and pipeline stage/cluster.
- Circular-reuse member offset, if the physical block has multiple members.

Represent one logical loop iteration as:

```text
acquire(g) -> produce(g) -> commit(g)
           -> wait(g, consumer_i) -> consume(g, consumer_i)
           -> release(g, consumer_i)
```

### 2. Define physical slot assignment

For an ordinary K-copy ring:

```text
slot(g) = g % K
```

For circular reuse with member staggering:

```text
slot(member, g) = (g + memberOffset) % K
```

The validator must use the same counter and staggering rules as code
partitioning, including subtiled-region and persistent-loop counter updates.

### 3. Build a static happens-before graph

Construct edges from facts guaranteed by the generated program:

- SSA and memory dependencies.
- Program order within one partition.
- Producer commit to consumer wait.
- Consumer release to the matching future producer acquire.
- Loop-carried token and barrier dependencies.
- Explicit TMA `wait_group` and final-drain semantics.

Do not add an edge from estimated latency, estimated II, or textual order across
independent partitions.

### 4. Validate every slot reuse

For each generation `g`, locate the next generation `g + K` assigned to the
same physical slot. Require a proven path:

```text
release(g, every consumer) -> acquire/overwrite(g + K)
```

For a shared circular group, perform this check across member boundaries as
well as across iterations. A slot can be overwritten only after all consumers
of the previous member/generation occupying that slot have released it.

If the path cannot be proven, the candidate copy count is invalid.

### 5. Integrate with search

Add a `CopySafetyValidator` after the copy solver proposes block depths and
before a complete plan is accepted:

```text
CopySolver proposes copies
  -> CopySafetyValidator validates slot reuse
  -> Packer budget check
  -> accept or reject candidate plan
```

On rejection, beam search should try:

1. A larger copy count for the affected block.
2. A different reuse grouping.
3. A separate physical allocation.

If no candidate is safe, compilation should report the unsafe buffer and
required minimum depth. It must not silently fall back to a heuristic planner
unless that planner runs the same validator and enforces the same floor.

### 6. Integrate with the heuristic planner

Run the same validator after heuristic Phase 4 and before emitting
`buffer.copy`. For every unsafe block:

1. Increase copies to the smallest statically safe depth.
2. Recompute physical SMEM usage and reuse-host capacity.
3. Reclaim only discretionary copies from other blocks if needed.
4. Emit an out-of-resources diagnostic if correctness floors cannot fit.

Correctness floors must never be reduced to satisfy the budget.

### 7. Add diagnostics

For every rejected or repaired plan, print:

```text
buffer name/id
producer partition and stage
consumer partitions and stages
proposed copy count
conflicting generations/members
missing release -> overwrite happens-before edge
minimum safe copy count found
```

This should be available in the memory-planner debug dump and in top-K plan
JSON so rejected performance candidates can be understood.

### 8. Add regression coverage

At minimum, cover:

- The addmm DP=2, `disallow_acc_multi_buffer`, three-stage failure.
- The same addmm structure with output staging depths 1, 2, and 3.
- FA-forward K/V circular reuse with the three-slot shared ring.
- A circular group whose member stage spans are both 1 but whose
  cross-iteration reuse requires more slots.
- Multiple consumer partitions where the latest consumer controls release.
- A safe plan rejected at K copies but accepted at K+1.
- Search exhaustion where every budget-fitting plan is unsafe.
- Parity between heuristic and search planners for the same IR.

### 9. Remove targeted guards only after parity

Keep the current addmm-specific `minCopies` guard until the validator:

- Derives the same three-copy minimum.
- Passes the full addmm and FA correctness suites.
- Produces identical safety decisions in heuristic and search modes.

After that, replace the targeted condition with the general derived floor.

## Regression coverage

### Implementation status

The plan is implemented by `StaticCopySafetyValidator` in memory-plan search
and the shared `getStaticSmemCopySafetyFloor` schedule analysis. Search
validates copy-solved partials and leaves, rejects plans without a proven safe
ring depth, and rechecks their copy-expanded footprint. Heuristic planning
raises `minCopies` to the same derived floor before discretionary allocation.
Estimated II is used only by the cost model.

The proven static facts currently cover consumer stage span, reuse-block entry
count, and the full-generation lifetime of TMA-fed, data-partitioned MMAv5
loops with accumulator multi-buffering disabled. Protocols not modeled by
search retain their conservative fallback into the validated heuristic path.

`test/Hopper/WarpSpecialization/ws_memory_planner_epilogue_multicopy.mlir`
covers the allocation rule.

With a large budget:

```text
MMA operands: 3 copies
output staging: 2 copies
```

With a tight budget:

```text
MMA operands: 3 copies
output staging: 1 copy
```

The complete addmm and FA device-TMA suites pass with this behavior.

## Summary

`cb3ded12e` exposed an unmodeled correctness constraint by changing allocation
priority. The store-wait mechanism itself was valid. Reserving its output
buffers first accidentally starved input rings whose depth was incorrectly
treated as optional.
