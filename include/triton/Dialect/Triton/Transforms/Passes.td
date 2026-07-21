#ifndef TRITON_PASSES
#define TRITON_PASSES

include "mlir/Pass/PassBase.td"

def TritonCombineOps : Pass</*cli-arg*/"triton-combine", /*Op*/"mlir::ModuleOp"> {
  let summary = "combine ops";
  let description = [{
    This pass aims to optimize the five following patterns:
    - `dot(a, b, 0) + c => dot(a, b, c)`

    - `addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))`

    - `select(cond, load(ptrs, broadcast(cond), ???), other) =>
         load(ptrs, broadcast(cond), other)`

    - `broadcast(constant) => reshaped_constant`
    - `torch.sum(x[:,:,None].expand(-1,-1,n) * y[None,:,:].expand(m,-1,-1),1)
       => dot(x,y,splat(0))`
  }];

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def TritonReorderBroadcast : Pass</*cli-arg*/"triton-reorder-broadcast", /*Op*/"mlir::ModuleOp"> {
  let summary = "Moves broadcast and splat after elementwise operations";
  let description = [{
    The purpose of this pass is to transform:
      - `elementwise(broadcast(a)) => broadcast(elementwise(a))`
      - `elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))`
    In the event of a match, the broadcast (or splat) operation is delayed
    and performed after the ElementWise operation.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonRewriteTensorDescriptorToPointer : Pass</*cli-arg*/"triton-rewrite-tensor-descriptor-to-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores of tensor descriptors into pointer load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_descriptor` into pointer semantics. After
    this pass, `tt.make_tensor_descriptor`  will disappear, and it generates logics to compute the pointer/mask/other
    for each load/store.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopUnroll : Pass</*cli-arg*/"triton-loop-unroll", /*Op*/"mlir::ModuleOp"> {
  let summary = "Loop unroller";
  let description = [{
    The pass unrolls a scf loop with tt.loop_unroll_factor attribute. The attribute specialises how many iterations
    the loop should be unrolled.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonSimplifySingleTripWhile : Pass</*cli-arg*/"triton-simplify-single-trip-while", /*Op*/"mlir::ModuleOp"> {
  let summary = "Replace a provably single-iteration scf.while with its body";
  let description = [{
    This pass detects an `scf.while` loop that provably executes its body exactly
    once and replaces the loop with the straight-line (inlined) body, eliminating
    the loop overhead and its loop-carried state.

    The current detection targets the "constant-flip" pattern produced by the
    non-persistent unified tile scheduler: the loop condition is a loop-carried
    `i1` value that is a constant `true` on entry (the while init) and a constant
    `false` after the body (the after-region yield). Concretely, the pass fires
    only when:
      - the before-region is side-effect-free, and
      - the `scf.condition` value is directly a before-block argument whose
        `scf.while` init folds to a constant `true` while the corresponding
        after-region yield operand folds to a constant `false`.

    The rewrite evaluates the (pure) before-region for the forwarded args, inlines
    the body once, and (only if the loop has results) evaluates the before-region
    again on the yielded values to produce them. Broader conditions (computed
    condition expressions) are not handled and could be added later via constant
    folding of the condition.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def TritonUpliftWhileToFor : Pass</*cli-arg*/"triton-uplift-while-to-for", /*Op*/"mlir::ModuleOp"> {
  let summary = "Convert a countable scf.while into an scf.for";
  let description = [{
    This pass rewrites a provably countable `scf.while` loop into the equivalent
    `scf.for` loop, so the loop exposes a real trip count to downstream
    optimizations (software pipelining, warp-specialization scheduling, loop
    unrolling).

    It reuses MLIR's `scf::populateUpliftWhileToForPatterns`, which fires only
    when the loop is a counting loop:
      - the before-region is a single `arith.cmpi` (`slt`/`sgt`) comparing a
        loop-carried induction variable against a bound that dominates the loop,
        and
      - the after-region increments that induction variable via `arith.addi`
        with a loop-invariant step.

    This targets the loop produced by the static-persistent unified tile
    scheduler (`_x < num_tiles`, `_x += num_programs`) and the hand-written
    static-persistent kernels. It intentionally leaves loops it cannot prove
    countable unchanged: the dynamic schedule (induction var advanced by an
    atomic, not `addi`) and the CLC schedule (condition is a hardware `i1`, not a
    `cmpi`). The non-persistent single-trip while is handled separately by
    `triton-simplify-single-trip-while`.

    The bound must dominate the loop for the pattern to match, so this pass is
    scheduled after LICM has hoisted any loop-invariant bound computation
    (e.g. `num_tiles`) out of the before-region.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def TritonLoopInvariantCodeMotion : Pass</*cli-arg*/"triton-licm", /*Op*/"mlir::ModuleOp"> {
  let summary = "MLIR's LICM plus hoist load ops out of loops with masks.";
  let description = [{
    This pass uses MLIR's LICM pass as base. Additionally, it hoists load ops
    out of loops that consists of pure/read-only ops. For scf.for loops, it
    generates a trip-count check. For scf.while loops, it clones the condition
    from the before body.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopAwareCSE : Pass<"triton-loop-aware-cse", "mlir::ModuleOp"> {
  let summary = "CSE within loop bodies";

  let description = [{
    The `triton-loop-aware-cse` pass performs recursive common subexpression
    elimination within loop bodies. Unlike regular CSE, which is a single-pass
    greedy algorithm, this pass can recursively eliminate loop iteration
    arguments and subcomputations that always have the same value.
  }];
}

def CudaWarnings : Pass<"test-cuda-warnings", "mlir::ModuleOp"> {
  let summary = "Emit warnings for performance-impacting patterns on CUDA targets";
  let description = [{
    This pass is intended for testing purposes only. Python code should instead call
    into the `mlir::triton::collectCudaWarnings` API instead to get warnings visible
    in Python.

    This pass analyzes TTIR for patterns that may cause performance issues
    on specific CUDA GPU architectures. Currently detects:

    - FP64 (double-precision) math operations on GB300 (SM103): GB300 has
      significantly reduced FP64 throughput (1/64th of FP32). The pass warns
      when operations like arith.addf, arith.mulf, tt.dot, math.exp, etc.
      operate on f64 types.

    The pass emits MLIR warnings that surface to the user during compilation.
    It does NOT warn on data movement operations like load/store.

    The pass uses the compute capability to determine which warnings to emit.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect"
  ];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"0",
           "Target GPU compute capability">
  ];
}

#endif
