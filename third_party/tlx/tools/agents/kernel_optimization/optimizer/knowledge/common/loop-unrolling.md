# Loop Unrolling

Apply this guidance to Triton and TLX kernels on every target. Loop
unrolling replicates a loop body so iterations execute back-to-back
without branch and induction overhead, and so the backend optimizer sees
a larger scheduling window. Unroll only loops whose rolled form is
demonstrated in the compiled pipeline and whose unrolled form is
measured faster.

## Know What Unrolls And What Does Not

`tl.range(0, TRIP, loop_unroll_factor=N)` attaches
`tt.loop_unroll_factor` to the TTIR `scf.for`, and Triton's
`LoopUnrollPass` unrolls the body N times. A remainder epilogue loop
appears when N does not divide the trip count; use a divisor of the
trip count (or the trip count itself, including a `constexpr` such as
`loop_unroll_factor=SPLIT_K`) for complete unrolling.

Do not expect unrolling without the attribute. Plain `range()` loops
are never unrolled at TTIR level: the pass defaults to factor 1
(off). Do not expect LLVM to compensate: no `llvm.loop` metadata is
emitted, LLVM O3 full-unroll budgets (~150 instructions) reject
anything but tiny loops, and the NVPTX backend performs no unrolling.
Whatever reaches the backend assembler rolled usually stays rolled
unless the assembler itself decides to unroll.

There is no assembler-level directive to force the decision: `#pragma
unroll` in PTX input is a ptxas parse error, and
`--allow-expensive-optimizations` does not change unroll decisions. To
unroll, transform the code before the assembler.

Manual unrolling (a step-N loop with the body written N times) produces
the same code as the attribute form. Prefer the attribute form; use the
manual form only when per-copy specialization is needed.

## Explore Unrolling For Small Non-Dot Loops

Consider unrolling when all of these hold:

- The loop is small and contains no `tl.dot` (dot loops belong to the
  pipeliner; unrolling them manually usually fights it).
- The trip count is modest (tens to low hundreds) and the body is a
  few loads/stores plus elementwise math. Typical shapes: split-K
  epilogues and finish kernels, short sequential reductions, and
  prologue/epilogue blocks of pipelined main loops.
- Profiling implicates the rolled loop: SASS shows one load plus a
  back-edge branch, and the profiler reports high long-scoreboard
  stalls with low memory throughput on a streaming kernel.

Do not bother when the backend already unrolled the loop (check SASS
first), when the trip count is huge (code bloat hurts the instruction
cache), or when the loop is dominated by a pipelined dot.

## NVIDIA: Respect The ptxas Budget

On NVIDIA targets, ptxas fully unrolls only under a trip-count by
body-size budget that closes quickly as the body grows. Measured on
ptxas-blackwell targeting sm_100a for a loop shaped like one
predicated `ld.global.v2` plus packed arithmetic per load (U = loads
per iteration): U=1 unrolls through trip 132 (rolled from 134), U=2
through trip 40 (rolled from 48), U=4 through trip 8 (rolled from
12), U=8 through trip 4 (rolled from 8). Partial source unrolling
still pays while rolled through amortized branch and induction
overhead, but only complete unrolling escapes a rolled loop. These
constants are orientation for this loop shape and toolchain, not law:
re-measure rather than extrapolating to other bodies.

## Verify At Every Level

Confirm an unrolling change at all four levels before concluding:

1. **TTIR**: the dumped `.ttir` must show no `scf.for` (full unroll)
   or one `scf.for` with N loads (unroll by N). Dumping requires
   `TRITON_KERNEL_DUMP=1` together with `TRITON_DUMP_DIR` (the dump
   directory alone does not enable dumping) and a fresh compile of
   the kernel.
2. **Machine code**: disassemble the cubin. Rolled means exactly U
   loads plus a back-edge branch; unrolled means trip x U loads with
   no loop branch.
3. **Profiler**: compare duration, long-scoreboard stalls, memory
   throughput, and registers per thread against the rolled baseline,
   and confirm zero local-memory spill bytes.
4. **Numerics**: sequential accumulation order is preserved by
   unrolling, so output should be bitwise identical; still verify
   with the case accuracy gate before concluding.

## Respect The Floor

Register pressure grows with the unroll factor; confirm spill
counters rather than assuming the unrolled loop fits. After
unrolling, the floor is often a dependence chain (for example, a
sequential chain of hundreds of adds). If the unrolled kernel is
still too slow, the next transform is algorithmic (for example, a
tree reduction), not more unrolling.
