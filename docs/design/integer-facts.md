# SSA Integer Facts

Control-flow predicates become SSA refinements. Sparse dataflow propagates
ranges, congruences, and relations to AxisInfo and integer narrowing.

## Refinement semantics

Notation: `refine(value, bounds...; predicates)`.

- One result, same type as `value`.
- Typed predicates reference operands; width, signedness, and division rounding
  are explicit.
- Defined inputs and true predicates produce `value`. A false predicate or
  poison input produces poison. An unused result causes no immediate UB.
- Facts apply to the result under the non-poison condition. Original inputs and
  sibling uses retain their own facts. `freeze` drops poison-dependent facts.

For `x1 = refine(x, limit; x < limit)`, the result carries `x1 < limit`.
Only the value operand becomes `x1` in the relation; `limit` keeps its SSA
identity. Refining `limit` requires another operation.

Verification checks types, operand references, predicates, and result arity.
The inserting pass establishes placement legality.

## Materialization

The pre-pass owns predicate discovery, dominance, and scoped use replacement.
The dataflow analysis consumes SSA definitions.

| Source | Refinement scope |
| --- | --- |
| Scalar branch | True predicate at true entry; negated predicate at false entry. |
| CFG edge into a shared successor | Dedicated edge block; refined values pass through block arguments. |
| `scf.for` | Both `lower <= iv` and `iv < upper` at body entry, with the loop's signedness. |
| `llvm.assume` | Immediately after the assumption; subsequent dominated uses. |

```text
if x < limit:
    x1 = refine(x, limit; x < limit)
    consume(x1)
else:
    x2 = refine(x, limit; x >= limit)
    consume(x2)
```

A true conjunction supplies each conjunct. Disjunctions require a proof over
all alternatives. Vector predicates require an explicit element activity domain;
an all-elements fact requires a reduction or a uniformity proof.

`scf.for` requires a positive step and a representable exit value
`lower + trip_count * step`; these establish non-wrapping induction.
See the [SCF contract](https://github.com/llvm/llvm-project/blob/a6533ab0f5a1df16ff3dbf7358fb79097c60855e/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td#L178).
Other recurrences need an overflow proof for ordered bounds. Wrapping
recurrences retain modular facts. Body bounds apply to refined induction values;
loop operands keep their original values. Loop results join initial and backedge
values, including the zero-trip case.

Retain source assumptions and operand bundles. A false `llvm.assume` causes UB
on execution even without later uses; an unused poison result cannot replace
that constraint. Removal requires a separate semantic justification.
See the [LLVM assume contract](https://llvm.org/docs/LangRef.html#llvm-assume-intrinsic).

Group refinements by value and entry scope. Rewrite dominated uses, excluding
the new operation's operands. Thread new facts through the value input of
dominated refinements; preserve bound operands. Reuse matching refinements so
repeated materialization converges from IR structure. Attach facts without
per-use refinements, arithmetic cloning, or pass-history markers. Earlier
computed values retain the facts from their original operands.

## Fact domain

| Component | Meaning |
| --- | --- |
| Signed/unsigned ranges | Bounds on one fixed-width bit pattern under each interpretation. |
| Congruence | `x = residue (mod modulus)`; residue zero gives divisibility. |
| Symbolic bounds | Inequalities between SSA-backed expressions. |
| Arithmetic relations | Operand/result correlations, including quotient and remainder. |

Components refine each other. Exact constants, uninitialized states, and unknown
facts are distinct. Contradictory refinements make their results poison; they
supply no facts about unrelated values or block reachability.

Expressions are typed, shared, and bounded. Variable products and divisors keep
structural representations. Congruences carry an integer interpretation.
Wrapping arithmetic reduces a modulus against `2^width`; general divisibility
survives only with an appropriate no-wrap proof.

Tensor ranges cover every element. AxisInfo group-base divisibility applies
only to each contiguous group's base.

## Proof rules

Every rule preserves source width, signedness, and definedness. Overflow flags
can establish no-wrap conditions. Unflagged arithmetic has bitvector semantics.
Lifting a source expression to mathematical integers requires a no-wrap proof;
a wider analysis representation alone is insufficient.

For defined division with `x >= 0` and `d > 0`:

```text
q = x / d, r = x % d
x = q*d + r
0 <= r < d
```

Quotient and remainder must share operands and division semantics. These
relations give bounds across runtime products and divisors:

```text
0 <= r < g*n, g > 0, n > 0  =>  0 <= r/g < n
```

Useful proof steps:

- **Product bound:** `0 <= a <= limit/b`, `limit >= 0`, `b > 0` proves that
  signed `a*b` fits when `limit` has the product's type.
- **Minimum:** `d = min(a,b)` is positive when both inputs are positive.
  For `n > 0`, proving both `x < a*n` and `x < b*n` establishes `x < d*n`.
- **Inequality combination:** positive integer scaling preserves order;
  strict integer bounds contribute at least one unit of slack.
- **Congruence gap:** if `a < b` and both are multiples of `k > 0`, then
  `a + offset < b` for `0 <= offset < k`. Non-power-of-two moduli require
  a consistent interpretation and a no-wrap proof across arithmetic.
- **Scaled coordinate:** `0 <= c <= (INT_MAX-(B-1))/B`, `0 <= lane < B`,
  `B > 0` establishes `0 <= c*B + lane <= INT_MAX`.

A query follows operand chains whose poison propagates to the queried result.
On those chains, refinements equal their value inputs. Select alternatives and
`freeze` require their own transfer rules. Bounds for `min`, `max`, and select
must hold for every feasible alternative. Local case splits intersect their
conclusions and remain within the proof budget.

Unsigned division, remainder, and minimum can use the signed mathematical rules
when operands fit the nonnegative signed range. Sign extension preserves signed
values; zero extension agrees with that interpretation for nonnegative inputs.
Normalize equivalent signed/unsigned bounds before joins.

Negative signed division truncates toward zero. Division by zero and signed
minimum divided by minus one follow the source dialect's definedness rules.
Casts preserve their specified truncation or extension semantics.

## Dataflow and IR lifetime

Sparse propagation follows SSA dependencies. Control-flow interfaces route
block arguments and region results. Joins retain facts common to every
executable incoming value, with symbolic references translated to destination
values. Relations between two merged values require matching incoming pairs on
every edge.

Calls translate actual operands to formal arguments; returns translate them
back. Drop references without a destination representation. Join all executable
call and return sites. Facts shared through a function must hold for every call.

The range lattice owns numeric intervals and widening. The symbolic lattice
holds finite sign facts and SSA bounds. Loop invariants require entry and
backedge proofs; body refinements remain transfers through widening.

- Cloning remaps predicate operands.
- CSE requires equal operands, result types, and predicates.
- Motion preserves operand dominance and poison behavior. Memory-effect freedom
  alone does not establish speculation legality.
- Operand, predicate, type, or CFG changes invalidate dependent analysis state.
  Recompute unless the rewrite infrastructure updates every affected dependency.
- Erased values lose their facts; storage reuse cannot transfer cached proofs.
- Keep refinements through their consumers. Erasure replaces each result with
  its value input, removing poison behavior. Source assumptions retain their
  execution semantics. Moving a proof-dependent computation to unrefined inputs
  requires a new proof or equivalent semantic justification.

## Producers and consumers

Operation models supply arithmetic and tensor semantics. Target interfaces
supply hardware launch bounds. Shape relations require an explicit producer
contract: a hardware program-ID bound alone cannot establish
`program_id < ceil(M/BM) * ceil(N/BN)`. Runtime dimensions need explicit facts
for positivity and overflow safety.

Queries return `Proven`, `Disproven`, or `Unknown`. Only a proof authorizes a
rewrite. Invalid IR and solver initialization failures are compiler errors.
AxisInfo consumes proofs for contiguity and constancy. Integer narrowing uses
the same range lattice. Backend cost models choose profitable legal forms.

Narrowing must reconstruct operands and results at the original width:

- Preserve signed/unsigned interpretation. Unsigned division, unsigned remainder,
  and logical right shift need zero-extension proofs. For example,
  `remui(sext(-1 : i32), 7)` differs from `remui(-1 : i32, 7)`.
- Keep shift counts below the new width.
- Re-prove overflow flags at the new width.
- Exclude newly introduced signed division/remainder overflow at `INT_MIN / -1`.

Range analysis and rewriting have separate lifetimes. Canonicalization follows
the range consumer.

## Compile-time bounds

Bound expression size, retained relations, and proof work. Budget exhaustion
returns `Unknown` and is visible in diagnostics. Retention and widening are
deterministic. Preserve input constraints while bounding derived constraints,
positive scaling, and combinations separately.

For `V` SSA states, `E` dependency edges, payload bound `K`, update bound `H`
after widening, and transfer cost `T(K)`, target storage is `O(V*K + E)` and
propagation work is `O((V + E)*H*T(K))`. Predicate operands count toward `E`;
symbolic expressions count toward `K`. All-pairs closure and exhaustive branch
path products exceed these bounds.

Use sorted, bounded polynomial terms. A local proof may cache numeric term
ranges while its input ranges remain fixed: `Q` probes of at most `T` terms
need at most `Q*T` entries. Invalidate that cache when input facts change.
Materialization accounts for inserted operands and rewritten uses; group
replacement by value and scope to bound repeated scans in nested scopes.

## Conformance

Tests cover sibling branches, shared successors, zero-trip loops, loop joins,
contradictions, poison/freeze, negative signed division, wrapping products,
non-power-of-two divisibility, and assumptions with no result users.

The production witness derives a grouped tile's column bound from source facts
and SSA arithmetic, then proves its scaled lane coordinate nonnegative.
Missing launch or shape facts yield `Unknown`. Validate numerical correctness,
assembly, compile time, and controlled performance across affected families.

## References

- [IREE integer divisibility analysis](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.cpp):
  sparse propagation, GCD joins, induction divisibility from lower bound and step.
- [IREE integer optimization](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/Util/Transforms/OptimizeIntArithmetic.cpp)
  and [integer assumptions](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/Util/IR/UtilOps.td):
  range/divisibility consumers and result-producing assumptions. Refinement
  poison semantics are defined here.
- [MLIR ValueBounds](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/ValueBoundsOpInterface.h):
  affine symbolic bounds; integer-type mode requires justified no-overflow
  assumptions. Variable products and divisors need additional rules.
