# gfx950 GEMM Irregular Tiles And K Tails

Use this reference when useful work does not fit a convenient power-of-two
macro tile or when K is not divisible by the main pipeline width. Irregular
logical geometry, physical storage, transfer grouping, and boundary semantics
are independent choices; keep them separate in the proof and the plan.

## Build Logical Tiles From Native MFMAs

For a regular wave grid:

```text
BlockM = MatrixInstM * MIWaveTileM * WarpsM
BlockN = MatrixInstN * MIWaveTileN * WarpsN
```

`MIWaveTileM` and `MIWaveTileN` count native matrix instructions. They need not
be powers of two. A logical width such as 96 or 160 can be a homogeneous grid
of native MFMAs rather than a special matrix instruction.

Evaluate an irregular tile by useful and padded work, total tile count, CU
rounds, operand reuse, accumulator registers, and store efficiency. A smaller
tile is especially promising when it reduces padding without increasing the
ceiling quotient on the affected axis. Test both neighboring shapes where that
quotient changes.

## Separate Logical Extent From Physical Backing

A non-power-of-two logical tile may use a larger convenient LDS allocation:

```text
logical windows: 64 rows + 64 rows + 32 rows = 160 useful rows
physical backing: three 64-row slots = 192 allocated rows
```

This is storage padding, not compute padding. The final slot has only 32 valid
rows: global transfer, local view, MFMA ownership, and output store must retain
that logical bound. The unused backing cannot become an out-of-bounds global
read or an MFMA input containing uninitialized values.

Static subslices let several narrow transfers populate constant regions of one
wider backing allocation. Use only non-reordering views whose offset mapping is
known through lowering. If direct-to-LDS is intended, verify that recognition
follows every supported view to the root allocation and preserves the authored
offset layout. Otherwise the candidate may silently materialize through VGPRs
or address the wrong LDS region.

Treat physical padding, exact fragmented backing, and a wider shared transfer
as competing implementations. Padding may simplify indexing and vectorization;
exact backing may reduce LDS pressure; a shared transfer may improve memory
coalescing while increasing lifetime. Benchmark the final artifact.

## Distinguish M/N Boundaries From K Tails

Invalid output lanes and invalid reduction lanes have different semantics:

- For a partial M or N tile, an input lane may redirect to a valid address if
  the corresponding output is discarded by a masked store. Its arithmetic
  result is irrelevant.
- For a partial K tile, every invalid reduction lane must contribute zero.
  Redirecting to an arbitrary valid value changes the dot product.

Prove masks for A and B independently. One operand may be contiguous along K
while the other has a strided or transposed boundary. Do not infer a store mask
or reduction mask from a neighboring load.

Controlled overcompute is acceptable only when all extra reads are in bounds,
extra reduction values are zero, and extra outputs cannot escape. Account for
the added MFMA and memory work when deciding whether it beats an exact
fragmented path.

## Peel The Reduction Tail

Keep the common full-K path unmasked and at the widest proved vector width.
Handle the residual K region separately:

```text
full iterations : unmasked, aligned, steady-state pipeline
tail iteration  : operand-specific bounds, zero fill, safe vector width
epilogue        : drain every outstanding stage before reuse or return
```

The tail can be a peeled source path, a dedicated final stage, or a separate
kernel when launch cost is justified. Whichever form is used, verify K smaller
than one block, exactly one block, one block plus a tail, and multiple full
blocks plus a tail. Include tails around every supported vector-width and MFMA-K
boundary.

Do not insert a masked tail into the steady-state path and assume only the final
iteration changes. Masking can alter vectorization, direct-to-LDS legality,
instruction count, wait distance, and register allocation for the whole
specialization.

## Prove Alignment Per Operand And Operation

Separate facts for A loads, B loads, output stores, and any workspace traffic.
For each operation record base alignment, stride, offset, vector width, and
whether masking can select a misaligned address. Transposition can reverse which
operand has contiguous K access.

A narrowly scoped backend relaxation for unmasked global loads does not prove
the same width is safe for masked loads, stores, atomics, or direct-to-LDS. Keep
the relaxation close to the proved operation and fail closed outside its
contract. Recheck the emitted instruction form rather than relying on source
pointer hints alone.

## Keep Fragment Layouts Legal

Irregular compute often divides one operand into fragments while reusing the
other. For every fragment prove:

- native MFMA ownership and accumulator slice;
- global and LDS offset mapping;
- local-load layout and lane ownership;
- no overlap or gap among logical windows;
- initialization of every consumed value; and
- a store mapping that reconstructs the logical output.

Avoid layout conversions that secretly materialize through shared scratch
inside a phase-sensitive pipeline. Load into the consumer layout where possible
or preserve a proved layout through a non-reordering view.

Split counts and fragment counts need not be powers of two, but indexing and
reductions must not assume bit-mask arithmetic. Test odd counts and the last
partial fragment explicitly.

## Validate Capability, Not Just Compilation

For an irregular-tile or K-tail claim, record:

```text
problem M/N/K and layout
logical MFMA grid and macro tile
physical LDS/register backing and every static view
global and local vector widths per operand
main-loop and tail instruction forms
VGPR/AGPR/LDS/scratch and occupancy
resolved launch grid and selected public dispatch
```

Inspect final IR or AMDGCN to ensure the intended irregular grid and movement
path are present. A regular tile with boundary masks does not validate
irregular macro-tile support, and a padded allocation does not validate correct
subslice addressing.
