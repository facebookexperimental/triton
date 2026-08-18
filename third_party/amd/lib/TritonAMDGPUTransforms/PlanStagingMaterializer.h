#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PLANSTAGINGMATERIALIZER_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PLANSTAGINGMATERIALIZER_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <utility>

namespace mlir::triton::amdgpu {

/// A resolved M1.5b.4 single-slot register-to-LDS staging request. The source,
/// supported derived-value DAG, and every selected consumer are direct
/// children of the same scf.for.
struct PlanRegisterToLdsStaging {
  scf::ForOp loop;
  Value source;
  std::string sourceValueId;
  RankedTensorType tensorType;
  int64_t logicalBytes = 0;
  int64_t alignment = 0;
  SmallVector<Operation *> consumers;
  SmallVector<OpOperand *> consumerOperands;
  SmallVector<Operation *> derivedOperations;
  SmallVector<std::pair<OpOperand *, Value>> preservedOperands;

  int64_t sourceLiveStartBefore = -1;
  int64_t sourceLiveEndBefore = -1;
  int64_t sourceLiveStartAfter = -1;
  int64_t sourceLiveEndAfter = -1;
  int64_t derivedOperationsCloned = 0;
  int64_t derivedOperationsPruned = 0;
  int64_t unselectedConsumersPreserved = 0;
  bool logicalLiveRangeShortened = false;

  gpu::LocalAllocOp allocation;
  gpu::LocalStoreOp store;
  gpu::LocalLoadOp load;
  gpu::BarrierOp visibilityBarrier;
  gpu::BarrierOp releaseBarrier;
  SmallVector<Operation *> clonedDerivedOperations;
  SmallVector<Value> consumerReplacementValues;
};

struct PlanStagingMaterializationResult {
  int64_t newAllocations = 0;
  int64_t newStores = 0;
  int64_t newLoads = 0;
  int64_t insertedBarriers = 0;
};

LogicalResult materializeRegisterToLdsStaging(
    MutableArrayRef<PlanRegisterToLdsStaging> staging,
    PlanStagingMaterializationResult &result, std::string &error);

} // namespace mlir::triton::amdgpu

#endif
