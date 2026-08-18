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

struct PlanPipelineStagingIntent {
  std::string valueId;
  std::string action;
  std::vector<std::string> consumerIds;
  int64_t distance = 0;
  int64_t bufferDepth = 0;
  int64_t alignment = 0;
};

struct PlanLoopPipelineDelta {
  std::string loopId;
  std::vector<PlanPipelineTransactionIntent> transactions;
  std::vector<PlanPipelineStagingIntent> staging;
};

struct PlanPipelineStagingApplyRecord {
  std::string valueId;
  std::string action;
  int64_t distance = 0;
  int64_t bufferDepth = 0;
  int64_t derivedOperationsCloned = 0;
  int64_t derivedOperationsPruned = 0;
  int64_t selectedConsumerOperands = 0;
  int64_t unselectedConsumersPreserved = 0;
  int64_t globalLoadsEliminated = 0;
  int64_t directToLdsCopies = 0;
  int64_t asyncCommitsInserted = 0;
  int64_t asyncWaitsInserted = 0;
  int64_t sourceLiveStartBefore = -1;
  int64_t sourceLiveEndBefore = -1;
  int64_t sourceLiveStartAfter = -1;
  int64_t sourceLiveEndAfter = -1;
  bool registerSourceEliminated = false;
  bool globalAccessSemanticsPreserved = false;
  bool logicalLiveRangeShortened = false;
  bool pipelineExpanded = false;
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
  int64_t ringMutationCount = 0;
  int64_t stagingMutationCount = 0;
  int64_t rewrittenSlotIndexCount = 0;
  int64_t updatedWaitCount = 0;
  int64_t insertedBarrierCount = 0;
  int64_t logicalLdsBytesBefore = 0;
  int64_t logicalLdsBytesAfter = 0;
  bool postRewriteDdgVerified = false;
  std::vector<std::string> groups;
  std::vector<PlanPipelineStagingApplyRecord> staging;
};

struct PlanPipelineApplyResult {
  bool accepted = false;
  bool transactional = true;
  bool committed = false;
  bool rolledBack = false;
  std::string kernel;
  std::string inputValueGraphFingerprint;
  std::string candidateOutputValueGraphFingerprint;
  std::string outputValueGraphFingerprint;
  std::string failurePhase;
  std::string error;
  int64_t movedOperationCount = 0;
  int64_t importedDependencyCount = 0;
  int64_t skippedDependencyCount = 0;
  bool changesIterationStorage = false;
  bool changesSynchronization = false;
  bool changesPrefetchDistance = false;
  bool changesBufferDepth = false;
  bool changesNewStaging = false;
  bool changesGlobalStaging = false;
  bool postRewriteAuditPassed = false;
  std::vector<PlanPipelineLoopApplyRecord> loops;
};

FailureOr<PlanPipelineDelta> parsePlanPipelineDelta(llvm::StringRef payload,
                                                    std::string &error);

std::string
serializePlanPipelineApplyReport(const PlanPipelineApplyResult &result);

} // namespace mlir::triton::plan

#endif // TRITON_ANALYSIS_PLANPIPELINE_H
