#ifndef TRITON_ANALYSIS_PLANPIPELINE_H
#define TRITON_ANALYSIS_PLANPIPELINE_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir::triton::plan {

inline constexpr llvm::StringLiteral kPlanPipelineDeltaSchema =
    "plan-pipeline-delta/0.1";
inline constexpr llvm::StringLiteral kBeforeUpdateAsyncWaitCountPosition =
    "before_update_async_wait_count";

struct PlanPipelineTransactionIntent {
  std::string groupId;
  std::string action;
  int64_t distance = 0;
  int64_t bufferDepth = 0;
};

struct PlanLoopPipelineDelta {
  std::string loopId;
  std::vector<PlanPipelineTransactionIntent> transactions;
};

struct PlanPipelineDelta {
  std::string schemaVersion;
  std::string kernel;
  std::string inputValueGraphFingerprint;
  std::string passPosition;
  std::vector<PlanLoopPipelineDelta> loops;
};

struct PlanPipelineLoopApplyRecord {
  std::string loopId;
  int64_t initiationInterval = 0;
  int64_t operationCount = 0;
  int64_t selectedOperationCount = 0;
  int64_t movedOperationCount = 0;
  int64_t importedDependencyCount = 0;
  int64_t skippedDependencyCount = 0;
  std::vector<std::string> groups;
};

struct PlanPipelineApplyResult {
  bool accepted = false;
  std::string kernel;
  std::string inputValueGraphFingerprint;
  std::string outputValueGraphFingerprint;
  std::string error;
  int64_t movedOperationCount = 0;
  int64_t importedDependencyCount = 0;
  int64_t skippedDependencyCount = 0;
  std::vector<PlanPipelineLoopApplyRecord> loops;
};

FailureOr<PlanPipelineDelta> parsePlanPipelineDelta(llvm::StringRef payload,
                                                    std::string &error);

std::string
serializePlanPipelineApplyReport(const PlanPipelineApplyResult &result);

} // namespace mlir::triton::plan

#endif // TRITON_ANALYSIS_PLANPIPELINE_H
