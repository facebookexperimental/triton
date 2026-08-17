#include "triton/Analysis/PlanPipeline.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

namespace mlir::triton::plan {
namespace {

static bool readString(const llvm::json::Object &object, StringRef key,
                       std::string &destination, std::string &error) {
  std::optional<StringRef> value = object.getString(key);
  if (!value) {
    error = "pipeline delta is missing string field '" + key.str() + "'";
    return false;
  }
  destination = value->str();
  return true;
}

static bool readPositiveInteger(const llvm::json::Object &object, StringRef key,
                                int64_t &destination, std::string &error) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value || *value < 1) {
    error =
        "pipeline delta field '" + key.str() + "' must be a positive integer";
    return false;
  }
  destination = *value;
  return true;
}

} // namespace

FailureOr<PlanPipelineDelta> parsePlanPipelineDelta(StringRef payload,
                                                    std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(payload);
  if (!parsed) {
    error = llvm::toString(parsed.takeError());
    return failure();
  }
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object) {
    error = "pipeline delta root must be a JSON object";
    return failure();
  }

  PlanPipelineDelta delta;
  if (!readString(*object, "schema_version", delta.schemaVersion, error) ||
      !readString(*object, "kernel", delta.kernel, error) ||
      !readString(*object, "input_value_graph_fingerprint",
                  delta.inputValueGraphFingerprint, error) ||
      !readString(*object, "pass_position", delta.passPosition, error))
    return failure();
  if (delta.schemaVersion != kPlanPipelineDeltaSchema) {
    error = "unsupported pipeline-delta schema '" + delta.schemaVersion + "'";
    return failure();
  }
  if (delta.passPosition != kBeforeUpdateAsyncWaitCountPosition) {
    error = "pipeline delta targets unsupported pass position '" +
            delta.passPosition + "'";
    return failure();
  }

  const llvm::json::Array *loops = object->getArray("loops");
  if (!loops || loops->empty()) {
    error = "pipeline delta must contain at least one loop";
    return failure();
  }
  std::set<std::string> loopIds;
  for (const llvm::json::Value &loopValue : *loops) {
    const llvm::json::Object *loopObject = loopValue.getAsObject();
    if (!loopObject) {
      error = "pipeline delta loop must be a JSON object";
      return failure();
    }
    PlanLoopPipelineDelta loop;
    if (!readString(*loopObject, "loop", loop.loopId, error))
      return failure();
    if (!loopIds.insert(loop.loopId).second) {
      error = "pipeline delta repeats loop '" + loop.loopId + "'";
      return failure();
    }
    const llvm::json::Array *staging = loopObject->getArray("staging");
    if (staging && !staging->empty()) {
      error = "M1.5b.3 does not materialize new staging";
      return failure();
    }
    const llvm::json::Array *transactions =
        loopObject->getArray("transactions");
    if (!transactions || transactions->empty()) {
      error = "M1.5b.3 loop must contain at least one transaction group";
      return failure();
    }
    std::set<std::string> groupIds;
    for (const llvm::json::Value &transactionValue : *transactions) {
      const llvm::json::Object *transactionObject =
          transactionValue.getAsObject();
      if (!transactionObject) {
        error = "pipeline transaction must be a JSON object";
        return failure();
      }
      PlanPipelineTransactionIntent transaction;
      if (!readString(*transactionObject, "group", transaction.groupId,
                      error) ||
          !readString(*transactionObject, "action", transaction.action,
                      error) ||
          !readPositiveInteger(*transactionObject, "distance",
                               transaction.distance, error) ||
          !readPositiveInteger(*transactionObject, "buffer_depth",
                               transaction.bufferDepth, error))
        return failure();
      if (transaction.action != "set_prefetch_distance") {
        error = "unsupported pipeline transaction action '" +
                transaction.action + "'";
        return failure();
      }
      if (transaction.bufferDepth < transaction.distance) {
        error = "pipeline transaction buffer depth cannot be less than "
                "prefetch distance";
        return failure();
      }
      if (!groupIds.insert(transaction.groupId).second) {
        error =
            "pipeline delta repeats async group '" + transaction.groupId + "'";
        return failure();
      }
      loop.transactions.push_back(std::move(transaction));
    }
    delta.loops.push_back(std::move(loop));
  }
  return delta;
}

std::string
serializePlanPipelineApplyReport(const PlanPipelineApplyResult &result) {
  llvm::json::Array loops;
  for (const PlanPipelineLoopApplyRecord &loop : result.loops) {
    llvm::json::Array groups;
    for (StringRef group : loop.groups)
      groups.push_back(group);
    loops.push_back(llvm::json::Object{
        {"loop", loop.loopId},
        {"initiation_interval", loop.initiationInterval},
        {"operations", loop.operationCount},
        {"selected_operations", loop.selectedOperationCount},
        {"moved_operations", loop.movedOperationCount},
        {"imported_dependencies", loop.importedDependencyCount},
        {"skipped_inconsistent_dependencies", loop.skippedDependencyCount},
        {"ring_mutations", loop.ringMutationCount},
        {"rewritten_slot_indices", loop.rewrittenSlotIndexCount},
        {"updated_waits", loop.updatedWaitCount},
        {"inserted_barriers", loop.insertedBarrierCount},
        {"logical_lds_bytes_before", loop.logicalLdsBytesBefore},
        {"logical_lds_bytes_after", loop.logicalLdsBytesAfter},
        {"post_rewrite_ddg_verified", loop.postRewriteDdgVerified},
        {"groups", std::move(groups)},
    });
  }
  llvm::json::Object report{
      {"schema_version", "plan-pipeline-apply-report/0.1"},
      {"accepted", result.accepted},
      {"kernel", result.kernel},
      {"input_value_graph_fingerprint", result.inputValueGraphFingerprint},
      {"output_value_graph_fingerprint", result.outputValueGraphFingerprint},
      {"error", result.error.empty() ? llvm::json::Value(nullptr)
                                     : llvm::json::Value(result.error)},
      {"moved_operations", result.movedOperationCount},
      {"imported_dependencies", result.importedDependencyCount},
      {"skipped_inconsistent_dependencies", result.skippedDependencyCount},
      {"loops", std::move(loops)},
      {"changes_iteration_storage", result.changesIterationStorage},
      {"changes_synchronization", result.changesSynchronization},
      {"changes_prefetch_distance", result.changesPrefetchDistance},
      {"changes_buffer_depth", result.changesBufferDepth},
      {"changes_dot_decomposition", false},
      {"post_rewrite_audit_passed", result.postRewriteAuditPassed},
      {"materialization_scope",
       result.changesIterationStorage || result.changesSynchronization
           ? "existing_lds_ring_and_sync"
           : "existing_lds_operation_order"},
  };
  std::string payload;
  llvm::raw_string_ostream stream(payload);
  stream << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(report)));
  return payload;
}

} // namespace mlir::triton::plan
