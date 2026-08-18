#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_TMAMULTICAST_H
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_TMAMULTICAST_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir {
class ModuleOp;
class Value;
namespace triton {
class DescriptorLoadOp;
namespace nvidia_gpu {

struct TMAClusterGeometry {
  llvm::SmallVector<int, 3> dims;

  static FailureOr<TMAClusterGeometry> get(ModuleOp module);
  unsigned size() const;
  llvm::SmallVector<int, 3> coordinates(unsigned rank) const;
  uint16_t maskFor(unsigned rank,
                   const llvm::SmallBitVector &broadcastAxes) const;
  unsigned leaderFor(unsigned rank,
                     const llvm::SmallBitVector &broadcastAxes) const;
};

struct TMAMulticastPlan {
  TMAClusterGeometry geometry;
  llvm::SmallBitVector broadcastAxes;
};

FailureOr<TMAMulticastPlan> analyzeTMAMulticast(DescriptorLoadOp load);

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#endif
