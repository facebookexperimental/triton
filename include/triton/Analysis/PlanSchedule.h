#ifndef TRITON_ANALYSIS_PLANSCHEDULE_H
#define TRITON_ANALYSIS_PLANSCHEDULE_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir::triton {
class FuncOp;

namespace plan {

inline constexpr llvm::StringLiteral kPlanScheduleDeltaSchema =
    "plan-schedule-delta/0.1";
inline constexpr llvm::StringLiteral kFinalStructuredTTGIRPosition =
    "after_warp_pipeline_conversion_before_scf_to_cf";

struct PlanBlockScheduleDelta {
  std::string blockId;
  std::vector<std::string> baselineOrder;
  std::vector<std::string> desiredOrder;
  std::string reason;
};

struct PlanScheduleDelta {
  std::string schemaVersion;
  std::string kernel;
  std::string inputValueGraphFingerprint;
  std::string passPosition;
  std::vector<PlanBlockScheduleDelta> blocks;
};

struct PlanBlockScheduleApplyRecord {
  std::string blockId;
  std::string baselineOrderHash;
  std::string desiredOrderHash;
  int64_t operationCount = 0;
  int64_t movedOperationCount = 0;
};

struct PlanScheduleApplyResult {
  bool accepted = false;
  std::string kernel;
  std::string inputValueGraphFingerprint;
  std::string outputValueGraphFingerprint;
  std::string error;
  int64_t checkedDependencyCount = 0;
  int64_t anchorCount = 0;
  int64_t movedOperationCount = 0;
  std::vector<PlanBlockScheduleApplyRecord> blocks;
};

FailureOr<PlanScheduleDelta> parsePlanScheduleDelta(llvm::StringRef payload,
                                                    std::string &error);

LogicalResult applyPlanSchedule(FuncOp function, const PlanScheduleDelta &delta,
                                PlanScheduleApplyResult &result,
                                std::string &error, bool strict = true);

std::string
serializePlanScheduleApplyReport(const PlanScheduleApplyResult &result);

} // namespace plan
} // namespace mlir::triton

#endif // TRITON_ANALYSIS_PLANSCHEDULE_H
