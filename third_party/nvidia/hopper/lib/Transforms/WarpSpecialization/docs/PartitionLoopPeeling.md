# Partition Loop Peeling

`peelPartitionLoops` runs in the generic software pipeliner after `lowerLoops`
and before `expandLoops`. Code partitioning has already cloned code into
physical `ttg.warp_specialize` regions. This placement matters: `lowerLoops`
must consume the original schedule before peeling moves one iteration into a
prologue. It exposes separate first-tile and remainder paths inside one
numbered partition without requiring every partition to share the same
control-flow shape.

The transformation is intentionally narrow. It recognizes an `scf.for` body
predicate of the form:

```mlir
%boundary = arith.addi %lb, %step
%masked = arith.cmpi slt, %iv, %boundary
%result = scf.if %masked { ... } else { ... }
```

when `%iv` is the loop induction variable and the comparison controls an
`scf.if`. It rewrites the loop to a zero-trip-safe outer `scf.if`, clones the
first iteration with `%masked = true`, and creates a remainder loop beginning
at `%lb + %step` with `%masked = false`. Canonicalization removes the dead side
of each activation branch later in the AutoWS pipeline.

## Matching contract

The comparison must live directly in the `scf.for` body block and must feed an
`scf.if` condition. Anything else -- a guard buried in a nested region, a
comparison that feeds arithmetic instead of a branch, or an `scf.while` (which
has no induction variable to split on, so CLC's dynamic outer loop is out of
scope) -- fails to match and the loop is left untouched. Peeling therefore never
partially rewrites a shape it does not fully recognize.

Loops are peeled in walk (post) order, innermost first. Peeling replaces a loop
with an `scf.if` and erases the original, so peeling an outer loop first would
erase the inner loops collected alongside it.

When more than one comparison matches, only the first in walk order is peeled:
the rewrite is a single first-iteration split, and peeling a second guard would
require nesting prologues. Loops without iteration arguments are supported; the
outer `scf.if` drops the terminators `scf.IfOp` auto-inserts for a result-less
op before the explicit yields are attached.

Only numbered partition regions returned by
`ttg.warp_specialize.getPartitionRegions()` are considered. The default region
is excluded, and the pass does not traverse into nested warp-specialize ops.
This placement and restriction are important: communication channels and
physical buffers have already been planned, so peeling cannot change channel
discovery or memory-planner decisions.

The HSTU self-attention backward kernel uses this pattern for its first masked
dK/dV tile. Its benchmark-derived regression is
`test/Hopper/WarpSpecialization/ws_single_partition_else_hstu_bwd.mlir`.
