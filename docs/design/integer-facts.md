# SSA Integer Ranges

Control-flow predicates become SSA refinements. Sparse dataflow propagates
signed and unsigned intervals to AxisInfo and integer narrowing.

## Refinement semantics

`tt.refine value, bound {predicate = ...}` has one result with the input type.
The predicate uses `arith.cmpi` signedness and fixed-width semantics.

Defined inputs and a true predicate produce `value`. A false predicate or a
poison input produces poison. An unused result causes no immediate UB.
The constraint applies to the result under the non-poison condition. Original
inputs and sibling uses retain their own ranges. `freeze` drops
poison-dependent constraints.

For `x1 = refine(x, limit; x < limit)`, only the value operand becomes `x1`.
The bound keeps its SSA identity. Verification checks the operation's types,
predicate, and result arity. The inserting pass establishes placement legality.

## Materialization

The pre-pass owns predicate discovery, dominance, and scoped use replacement.
The range analysis consumes SSA definitions.

| Source | Refinement scope |
| --- | --- |
| Scalar branch | True predicate at true entry; negated predicate at false entry. |
| CFG edge into a shared successor | Dedicated edge block; refined values pass through block arguments. |
| `scf.for` | Both `lower <= iv` and `iv < upper` at body entry, with the loop's signedness. |
| `llvm.assume` | Immediately after the assumption; subsequent dominated uses. |

```text
if x < 1024:
    x1 = refine(x, 1024; x < 1024)
    consume(x1)
else:
    x2 = refine(x, 1024; x >= 1024)
    consume(x2)
```

A true conjunction supplies each conjunct. A false disjunction supplies each
negated operand. Other Boolean combinations require a common consequence.
Vector predicates require an explicit element activity domain before they can
supply an all-elements constraint.

`scf.for` requires a positive step and a representable exit value
`lower + trip_count * step`; these establish non-wrapping induction.
Body bounds apply to refined induction values. Loop results join initial and
backedge values, including the zero-trip case.

Retain source assumptions and operand bundles. A false `llvm.assume` causes UB
on execution even without later uses; an unused poison result cannot replace
that constraint.

Group refinements by value and entry scope. Rewrite dominated uses, excluding
the new operation's operands. Thread new constraints through the value input
of dominated refinements; preserve bound operands. Reuse matching refinements
so repeated materialization converges from IR structure. Earlier computations
retain the ranges of their original operands.

## Range transfer

Each value has signed and unsigned intervals over the same fixed-width bit
patterns. Tensor intervals cover every element. Uninitialized, exact constant,
and full-range states are distinct.

For `refine(x, bound; x < bound)`, the bound's maximum limits the result's
maximum. Equality intersects operand intervals. Inequality can exclude a
constant at an interval endpoint. Contradictions poison the result; they supply
no constraints on sibling values or block reachability.

Arithmetic preserves source width, signedness, and definedness. Unflagged
operations can wrap. A nonnegative base does not prove that `base + lane` is
nonnegative when the sum can overflow. Signed division truncates toward zero;
signed remainder has the dividend's sign. Analyze nonzero positive and negative
divisor intervals separately.

Operation `InferIntRangeInterface` methods supply ranges to MLIR dataflow.
Constants, `make_range`, and target launch limits supply initial bounds.
Tensor shape and layout operations preserve element ranges. Joins retain ranges
that contain every executable incoming value. The MLIR range lattice owns loop
widening. Refinements in loop bodies remain range transfers after widening.

A relation between runtime values can require an explicit producer bound:

```python
column = tile // rows
# Launcher supplies 0 <= tile < rows * columns and positive dimensions.
tl.assume(column < columns)
```

Such assumptions belong at the producer that establishes the mapping contract.
A hardware program-ID limit alone cannot establish a matrix-specific grid bound.

## Consumers and IR lifetime

AxisInfo applies signed div/rem grouping rules only when the dividend's signed
interval is nonnegative. Group-base divisibility describes each contiguous
group's base; it does not constrain every element.

Integer narrowing uses the same range solver. Reconstruct operands and results
at the original width, with their signed or unsigned interpretation:

- Unsigned division, remainder, and logical right shift require zero-extension
  proofs. `remui(sext(-1 : i32), 7)` differs from `remui(-1 : i32, 7)`.
- Shift counts must stay below the new width.
- Overflow flags must hold at the new width.
- Narrowing must not introduce signed division or remainder overflow at
  `INT_MIN / -1`.

Cloning remaps refinement operands. CSE requires equal operands, result types,
and predicates. Motion preserves operand dominance and poison behavior;
memory-effect freedom alone does not establish speculation legality.

Operand, predicate, type, or CFG changes invalidate dependent analysis state.
Recompute unless the rewrite infrastructure updates all affected dependencies.
Erased values lose their ranges. Range analysis and rewriting have separate
lifetimes; canonicalization follows the range consumer.

Keep refinements through their consumers. Erasure replaces each result with its
value input and removes its poison behavior. Source assumptions retain their
execution semantics. Moving a proof-dependent computation to unrefined inputs
requires a new proof or equivalent semantic justification.

## References

- [SCF loop contract](https://github.com/facebookexperimental/llvm-project/blob/a6533ab0f5a1df16ff3dbf7358fb79097c60855e/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td#L178)
- [LLVM assume contract](https://llvm.org/docs/LangRef.html#llvm-assume-intrinsic)
- [Range and materialization tests](../../test/Analysis/test-alignment.mlir)
