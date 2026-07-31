#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PARTITIONLOOPPEELING_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PARTITIONLOOPPEELING_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::gpu {

// Peel the first iteration of partition-local loops whose first-tile predicate
// has the canonical `iv < lower_bound + step` form. This runs after pipeline
// load lowering, so the peeled prologue cannot invalidate schedule analysis.
void peelPartitionLoops(ModuleOp moduleOp);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_PARTITIONLOOPPEELING_H_
