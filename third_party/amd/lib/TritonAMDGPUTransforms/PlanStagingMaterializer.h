#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PLANSTAGINGMATERIALIZER_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PLANSTAGINGMATERIALIZER_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <utility>

namespace mlir::triton::amdgpu {

/// A resolved M1.5b.4 register/global-to-LDS staging request. Buffered global
/// staging is expanded as a cross-iteration software pipeline; register
/// staging remains same-iteration and single-slot.
struct PlanLdsStaging {
  scf::ForOp loop;
  std::string action;
  Value source;
  std::string sourceValueId;
  std::string sourceProducerId;
  RankedTensorType tensorType;
  int64_t logicalBytes = 0;
  int64_t alignment = 0;
  int64_t distance = 0;
  int64_t bufferDepth = 1;
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
  int64_t globalLoadsEliminated = 0;
  int64_t directToLdsCopies = 0;
  int64_t asyncCommitsInserted = 0;
  int64_t asyncWaitsInserted = 0;
  bool registerSourceEliminated = false;
  bool globalAccessSemanticsPreserved = false;
  bool logicalLiveRangeShortened = false;
  bool pipelineExpanded = false;

  gpu::LocalAllocOp allocation;
  Value ringIndex;
  gpu::MemDescIndexOp bufferView;
  gpu::LocalStoreOp store;
  gpu::LocalLoadOp load;
  gpu::BarrierOp visibilityBarrier;
  gpu::BarrierOp releaseBarrier;
  Operation *globalCopy = nullptr;
  gpu::AsyncCommitGroupOp asyncCommit;
  gpu::AsyncWaitOp asyncWait;
  SmallVector<Operation *> clonedDerivedOperations;
  SmallVector<Value> consumerReplacementValues;
};

struct PlanStagingMaterializationResult {
  scf::ForOp loop;
  int64_t newAllocations = 0;
  int64_t newStores = 0;
  int64_t newLoads = 0;
  int64_t insertedBarriers = 0;
};

LogicalResult materializeLdsStaging(MutableArrayRef<PlanLdsStaging> staging,
                                    PlanStagingMaterializationResult &result,
                                    std::string &error);

} // namespace mlir::triton::amdgpu

#endif
