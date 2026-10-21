# SSA Integer Facts

## Purpose and ownership

Integer optimization needs ranges, divisibility, and relations between runtime
values. Independent intervals lose relations such as `tile < rows * columns`.
Those relations can determine whether signed address arithmetic wraps and
whether a tensor access is contiguous.

A fact-materialization pre-pass converts control-flow predicates and
predicate-style assumptions into result-producing SSA refinements. A sparse
dataflow analysis then derives integer facts from SSA definitions. It does not
query program points, dominance, nearby assumptions, or enclosing predicates.

TLX owns the complete implementation: the fact-materialization pre-pass,
refinement IR, integer analysis, operation models, and consumer integration.
The analysis lives in TLX compiler analysis support. Target models in TLX supply
hardware bounds. AxisInfo consumes proofs for its tensor layout rules.

Refinement operations are consumed and lowered within TLX. Arithmetic rules
apply to operation semantics, not kernel patterns.

## SSA refinement contract

Use `refine(value, bounds...; predicates)` below as notation for a typed IR
operation. Each operation has exactly one result, with the same type as its
value input. Additional SSA operands supply values referenced by the predicates.
Predicates use typed expressions over operand positions. Integer width,
comparison signedness, and division rounding are explicit. Strings are not a
predicate representation.

If all predicates hold on defined inputs, the result equals the value input.
If a predicate is false, the result is poison. Poison inputs cannot produce a
defined result. Evaluation of an unused refinement does not cause
immediate undefined behavior.

Facts constrain the result. They do not constrain the original inputs or
sibling uses. Analysis conclusions apply when the values that support the
proof are non-poison. A `freeze` operation cannot retain predicates that hold
only for non-poison refinement results.

A relational predicate can reference an unchanged bound operand:

```text
x1 = refine(x, limit; 0 <= x < limit)
```

The relation becomes `0 <= x1 < limit`. It is conditional on `x1` being
non-poison. It must not become a global constraint on `x` or `limit`. Replace
only the value input with the result when instantiating the predicate; bound
operands retain their SSA identity. A separate refinement is required to give
another value its own refined result.

The verifier checks types, operand references, predicate structure, and result
arity. It does not prove predicates by inspecting producers or control flow.
The pass that inserts the operation must justify its placement.

## Fact materialization

The pre-pass owns control-flow reasoning and scoped use replacement. It inserts
refinements only where source semantics establish the predicates. It excludes
the new operation's inputs from use replacement.

### Branches

For a scalar branch on `x < limit`, refine the constrained value at the true
successor's entry. At the false successor, use the negated comparison with the
same integer interpretation. Rewrite only uses dominated by the refinement.

```text
if x < limit:
    x1 = refine(x, limit; x < limit)
    consume(x1, limit)
else:
    x2 = refine(x, limit; x >= limit)
    consume(x2, limit)
```

A successor with other incoming edges cannot use one predecessor's predicate
unconditionally. Materialize on a dedicated edge, then pass refined values
through block arguments. At the merge, ordinary dataflow joins apply.

A true conjunction permits its individual predicates. A true disjunction does
not. Preserve a disjunction as a predicate unless its alternatives can be
handled soundly within the proof budget. Do not replace it with a conjunction.

A vector mask does not establish a predicate for every tensor element. A masked
region needs an explicit elementwise predicate and activity domain. A scalar
all-elements fact requires a justified reduction or uniformity fact.

### Loops

Materialize the loop body's guaranteed exit-test relation on a refined induction
value. For a signed upper-exclusive test, this includes `iv < upper_bound`.

Lower-bound, step, and congruence facts depend on the loop operation's semantics.
An integer recurrence that can wrap does not imply `iv >= lower_bound`.
Likewise, mathematical `iv = lower_bound + k * step` requires a valid integer
interpretation. Preserve modular relations when only bitvector arithmetic is
justified.

Use refined induction values in the body. Keep the original loop bounds and
step as inputs to the loop. Do not export body-only inequalities as facts about
the initial bound or the loop's results. Loop-carried results require a join
over their initial values and backedge values.

### Predicate-style assumptions

The same pre-pass handles `llvm.assume`-style operations. For an assumption on
`x < limit`, insert a refinement immediately after it and replace dominated
uses of the constrained values. Earlier uses and sibling regions remain
unchanged. Multiple assumptions form SSA refinement chains.

Retain the original assumption. A false `llvm.assume` causes undefined behavior
when executed, even when no later result is used. A poison-producing refinement
does not preserve that execution constraint by itself. Operand bundles can
also carry non-integer assumptions; preserve those semantics in full.
See the [LLVM assume contract](https://llvm.org/docs/LangRef.html#llvm-assume-intrinsic).

The integer analysis reads the refinement results, not the retained assumption.
This gives one SSA input form without weakening the source operation. A source
assumption may be removed only by a transformation that separately justifies
its removal or preserves its execution constraint.

### Placement and repeated execution

Reuse a refinement operation when its operands and predicates already express the
required facts. Repeated execution must converge from IR structure. Do not add
processed markers or pass-history attributes.

Group predicates by their valid entry scope. Do not insert a refinement per use
or clone arithmetic solely to attach facts. A previously computed value does
not acquire facts merely because one of its inputs has a later refinement.

## Integer fact domain

The analysis combines these components. Each component can refine the others.

| Component | Meaning |
| --- | --- |
| Signed and unsigned ranges | Bounds on the same fixed-width bit pattern under each interpretation. |
| Congruence | `x = residue (mod modulus)`; divisibility is residue zero. |
| Symbolic bounds | Inequalities between SSA-backed expressions. |
| Arithmetic relations | Correlations between operands and results, including quotient and remainder. |

Represent exact constants separately from unknown facts. Distinguish an
uninitialized state from a state with no useful bounds. An inconsistent
refinement means its result cannot be non-poison; it does not make the
enclosing block unreachable or justify facts about unrelated values.

Expression nodes are shared and typed. Preserve variable products and divisors
as structural nodes. Do not expand arbitrary polynomials. The analysis uses
bounded arithmetic rules, not an unrestricted symbolic solver.

Congruence refers to a specified integer interpretation. For wrapping
arithmetic, a divisibility fact with modulus `m` can lose factors that do not
divide `2^width`. Do not transfer general integer divisibility as if wrapping
preserved it. Power-of-two alignment is a useful special case of this rule.

Tensor range facts bound every element covered by the fact. AxisInfo's
contiguous-group base divisibility is a different property. Do not treat it as
divisibility of every tensor element.

## Arithmetic proof rules

Every rule must state its sign, width, and definedness requirements. Source
overflow flags can establish requirements; unflagged arithmetic must retain
its wrapping semantics.

For defined division with `x >= 0` and `d > 0`, retain the paired relations:

```text
q = x / d
r = x % d
x = q*d + r
0 <= r < d
```

Match the same operands, integer interpretation, and division semantics. The
relations permit generic deductions such as:

```text
0 <= r < g*n, g > 0, n > 0
    implies 0 <= r/g < n
```

These are mathematical relations. Lift a wrapping source product into them only
after proving that its relevant interpretation does not overflow. A widened
analysis expression does not prove that the original narrow operation was safe.

For a coordinate `c` and lane `l`, the following facts prove safe signed scaling:

```text
0 <= c <= (INT_MAX - (B - 1)) / B
0 <= l < B
B > 0
    implies 0 <= c*B + l <= INT_MAX
```

Rules for `min`, `max`, and select retain bounds common to all feasible
alternatives. A proof can split one local expression into cases and intersect
the conclusions. It must not enumerate the cross-product of branch paths.

Signed truncating division is not floor division for negative operands. Signed
minimum divided by minus one and division by zero require the source dialect's
definedness rules. Casts must preserve their truncation or extension semantics.

## Dataflow and lifetime

Use sparse propagation over SSA dependencies. The refinement transfer combines
incoming facts with its predicates. Arithmetic transfers propagate those facts.
The analysis has no program-point fact state and does not rescan branch tests.
Control-flow interfaces still route values to block arguments and region
results; this routing is not predicate discovery.

At a merge, retain only facts valid for every executable incoming value.
Translate symbolic references to the destination values before comparing facts.
A relation between two merged values requires agreement for the corresponding
incoming pair on every edge. Independent joins cannot establish that pairing.

Use widening for changing loop-carried facts. A body-entry refinement remains
a transfer in each iteration; widening the recurrence does not erase the
operation's semantic predicate. Prove an invariant from the entry and backedge,
not by assuming the invariant to prove itself.

Cloning remaps all predicate operands. CSE requires equal operands, result
types, and predicates. Refinements have no memory effects, but motion must
preserve poison behavior and operand dominance. Memory-effect freedom alone
does not establish speculation legality.

After operand, predicate, type, or control-flow changes, invalidate dependent
analysis state. Recompute the analysis unless the rewrite infrastructure updates
all affected dependencies. A cached proof cannot outlive its supporting IR.

Keep refinements until their proof consumers have finished. Lowering may replace
a refinement by its inputs as a removal of poison behavior; it must not export
its predicates to the original inputs. Retained source assumptions keep their
own semantics. Do not move a proof-dependent computation to unrefined operands
without a new proof or an equivalent semantic justification.

## Inputs and consumers

Operation models supply arithmetic and tensor semantics. Target interfaces
supply hardware launch bounds. A launch-to-shape relation must come from an
explicit producer contract. A hardware program-ID limit alone does not imply
`program_id < ceil(M/BM) * ceil(N/BN)`.

The runtime matrix dimensions are ordinary integer arguments unless the IR
states stronger facts. Neither positivity nor overflow safety follows from
their names or from an observed test shape.

Consumers query facts about SSA values and receive proven, disproven, or unknown
results. Unknown is an analysis result, not a compiler failure. It cannot
authorize a rewrite. Invalid IR or a failed solver initialization is a compiler
error and must be reported.

AxisInfo uses the proofs to establish tensor contiguity and constancy. Other
consumers can use them for narrowing and arithmetic simplification. Proof
availability establishes legality, not profitability. Backend cost models remain
responsible for the choice between equivalent arithmetic forms.

## Compile-time constraints

This is a compiler hot path. Bound expression size, retained relations, and
proof work. Share expression nodes and revisit only dependent states. Drop
precision soundly when a bound is reached and expose that event in diagnostics.

Let `V` be the number of SSA states, `E` the dependency edges, `K` the maximum
retained fact/expression payload per state, `H` the maximum state updates after
widening, and `T(K)` the bounded transfer cost. Target storage is
`O(V*K + E)` and propagation work is `O((V + E)*H*T(K))`. Account for predicate
operands in `E` and symbolic storage in `K`.

All-pairs relational closure requires quadratic storage and can require cubic
work. Exhaustive path splitting can require exponential work. Neither belongs
in the default analysis. Retention and widening choices must be deterministic.

The pre-pass must also account for inserted operands and rewritten uses. A
scope can refine many values, and nested scopes can multiply that work. Group
use replacement by value and scope rather than rescanning the full function
for each predicate.

## Conformance requirements

Required semantic witnesses include sibling branches, multi-predecessor merges,
zero-trip loops, loop-carried joins, contradictory predicates, poison and freeze,
negative signed division, wrapping products, and non-power-of-two divisibility.
An assumption with no result users must retain its execution semantics.

A production witness must derive a grouped tile's column bound from source
facts and SSA arithmetic, then prove the scaled lane coordinate nonnegative.
It must not inject the final coordinate bound or recognize a kernel name.
Missing launch and shape facts must remain unproved.

Acceptance requires numerical correctness and controlled performance comparison
across the affected kernel families. Compare generated assembly and compile-time
cost. A local speedup or passing proof test does not establish acceptance.

## Related infrastructure

- IREE's [integer divisibility analysis](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.cpp)
  uses sparse propagation and operation interfaces. It joins divisibility with
  GCD and derives induction-value divisibility from the lower bound and step.
- IREE's [integer optimization pass](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/Util/Transforms/OptimizeIntArithmetic.cpp)
  combines MLIR range analysis with divisibility analysis. Its result-producing
  [integer assumption operation](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/Util/IR/UtilOps.td)
  provides a useful SSA boundary. These sources inform the design; they do not
  define the refinement operation's poison semantics above.
- MLIR's [ValueBounds interface](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/ValueBoundsOpInterface.h)
  supports symbolic affine bounds. Its integer-type mode assumes no overflow.
  Use it only where that assumption is justified. Variable products and divisors
  also require rules outside the affine subset.
