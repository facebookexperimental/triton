#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_LOOP_PEELING_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_LOOP_PEELING_H_

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace triton {

// Peel the single last iteration of the loop.
void peelLoopEpilogue(
    scf::ForOp forOp,
    function_ref<Operation *(RewriterBase &, Operation *, bool)>
        processPeeledOp = nullptr);

// Peel the single first iteration of the loop into a prologue.
// The peeled iteration executes before the loop with iv = lowerBound.
// The loop's lower bound is adjusted to lowerBound + step, and its
// init args are replaced with the yielded values of the peeled iteration.
void peelLoopPrologue(scf::ForOp forOp);

} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITON_TRANSFORMS_LOOP_PEELING_H_
