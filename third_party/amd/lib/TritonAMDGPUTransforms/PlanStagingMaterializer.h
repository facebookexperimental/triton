#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PLANSTAGINGMATERIALIZER_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PLANSTAGINGMATERIALIZER_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir::triton::amdgpu {

/// A resolved M1.5b.4a single-slot register-to-LDS staging request. The
/// source and every selected use are direct children of the same scf.for.
struct PlanRegisterToLdsStaging {
  scf::ForOp loop;
  Value source;
  RankedTensorType tensorType;
  int64_t logicalBytes = 0;
  int64_t alignment = 0;
  SmallVector<Operation *> consumers;
  SmallVector<OpOperand *> consumerOperands;

  gpu::LocalAllocOp allocation;
  gpu::LocalStoreOp store;
  gpu::LocalLoadOp load;
  gpu::BarrierOp visibilityBarrier;
  gpu::BarrierOp releaseBarrier;
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
