#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PARTITIONLOOPPEELING_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PARTITIONLOOPPEELING_H_

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::gpu {

// Marks the scalar first-iteration branch that peeling materializes for a
// tensor causal mask, so cloneIteration can fold it away again.
constexpr inline llvm::StringLiteral kSyntheticMaskBranchAttrName =
    "ttg.loop_peeling.synthetic_mask";

// The warp-specialization task attribute. Named here because core transforms
// cannot include the NVIDIA backend's WarpSpecialization/Utility.h, which
// declares the same name for backend passes.
constexpr inline llvm::StringLiteral kAsyncTaskIdAttrName = "async_task_id";

// Peel the first iteration of partition-local loops whose first-tile predicate
// has the canonical `iv < lower_bound + step` form. This runs after pipeline
// load lowering, so the peeled prologue cannot invalidate schedule analysis.
void peelPartitionLoops(ModuleOp moduleOp);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_PARTITIONLOOPPEELING_H_
