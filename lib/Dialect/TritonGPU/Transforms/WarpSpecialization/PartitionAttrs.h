#ifndef TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_PARTITIONATTRS_H_
#define TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_PARTITIONATTRS_H_

#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace mlir {
class Operation;
class OpOperand;
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

namespace mlir::triton::gpu {

SetVector<int> getPartitionIds(Operation *op);
SmallVector<SetVector<int>, 4> getPartitionOutputs(Operation *op);
SetVector<int> getPartitionIds(OpOperand *use);
bool hasPartition(Operation *op);
bool hasWarpSpecializeTag(Operation *op);
std::optional<int> getWarpSpecializeTag(Operation *op);

LogicalResult verifyPartitionedLoop(scf::ForOp loop);

} // namespace mlir::triton::gpu

#endif // TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_PARTITIONATTRS_H_
