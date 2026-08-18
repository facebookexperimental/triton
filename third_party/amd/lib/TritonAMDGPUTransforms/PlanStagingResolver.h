#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PLANSTAGINGRESOLVER_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PLANSTAGINGRESOLVER_H_

#include "PlanStagingMaterializer.h"
#include "triton/Analysis/PlanPipeline.h"
#include "triton/Analysis/PlanValueGraph.h"
#include "llvm/ADT/DenseSet.h"
#include <map>

namespace mlir::triton::amdgpu {

struct PlanRegisterToLdsResolution {
  SmallVector<PlanRegisterToLdsStaging, 1> staging;
  llvm::DenseSet<Operation *> participatingOperations;
  int64_t logicalBytes = 0;
};

LogicalResult resolveRegisterToLdsStaging(
    ArrayRef<plan::PlanPipelineStagingIntent> intents, scf::ForOp loop,
    const plan::PlanValueGraph &graph,
    const std::map<std::string, Operation *> &operationById,
    const std::map<std::string, Value> &valueById,
    PlanRegisterToLdsResolution &result, std::string &error);

} // namespace mlir::triton::amdgpu

#endif
