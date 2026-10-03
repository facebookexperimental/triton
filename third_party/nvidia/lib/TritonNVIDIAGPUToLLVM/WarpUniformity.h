#ifndef TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_WARP_UNIFORMITY_H
#define TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_WARP_UNIFORMITY_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir::triton::NVIDIA {

// Conservative warp-uniformity analysis, shared by lowering steps that must
// prove warp uniformity (e.g. selecting the `.aligned` form of PTX `bar.sync`/
// `bar.arrive`, which requires all threads in a warp to execute the same
// barrier instruction).
//
// Anything these helpers cannot prove uniform is reported non-uniform; callers
// fall back to a form with no uniformity requirement.

// Returns true if `v` provably holds the same value for every thread in a
// warp: constants, CTA-uniform sources (program IDs, block/grid sizes, warp
// IDs), pure functions of uniform values, and values uniform across structured
// control flow with uniform predicates and bounds.
bool isWarpUniformValue(Value v);

// Returns true if `op` provably executes warp-uniformly: whenever one thread
// in a warp executes it, all threads in the warp execute it together. That is,
// every conditional branch that can reach it (unstructured or structured) has
// a warp-uniform condition. Warp-specialize partitions distribute whole warps,
// so ops inside a partition still execute warp-uniformly.
bool hasUniformExecution(Operation *op);

} // namespace mlir::triton::NVIDIA

#endif // TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_WARP_UNIFORMITY_H
