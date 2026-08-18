#include "triton/Analysis/PlanPipeline.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
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

static bool readStringArray(const llvm::json::Object &object, StringRef key,
                            std::vector<std::string> &destination,
                            std::string &error) {
  const llvm::json::Array *values = object.getArray(key);
  if (!values || values->empty()) {
    error = "pipeline delta field '" + key.str() +
            "' must be a non-empty string array";
    return false;
  }
  std::set<std::string> unique;
  for (const llvm::json::Value &value : *values) {
    std::optional<StringRef> text = value.getAsString();
    if (!text) {
      error =
          "pipeline delta field '" + key.str() + "' must contain only strings";
      return false;
    }
    if (!unique.insert(text->str()).second) {
      error =
          "pipeline delta field '" + key.str() + "' contains a duplicate value";
      return false;
    }
    destination.push_back(text->str());
  }
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
    const llvm::json::Array *transactions =
        loopObject->getArray("transactions");
    const llvm::json::Array *staging = loopObject->getArray("staging");
    if ((!transactions || transactions->empty()) &&
        (!staging || staging->empty())) {
      error = "pipeline loop must contain a transaction or staging intent";
      return failure();
    }
    std::set<std::string> groupIds;
    if (transactions) {
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
          error = "pipeline delta repeats async group '" + transaction.groupId +
                  "'";
          return failure();
        }
        loop.transactions.push_back(std::move(transaction));
      }
    }
    std::set<std::string> stagedValues;
    if (staging) {
      for (const llvm::json::Value &stagingValue : *staging) {
        const llvm::json::Object *stagingObject = stagingValue.getAsObject();
        if (!stagingObject) {
          error = "pipeline staging intent must be a JSON object";
          return failure();
        }
        PlanPipelineStagingIntent intent;
        if (!readString(*stagingObject, "value", intent.valueId, error) ||
            !readString(*stagingObject, "action", intent.action, error) ||
            !readStringArray(*stagingObject, "consumers", intent.consumerIds,
                             error) ||
            !readPositiveInteger(*stagingObject, "buffer_depth",
                                 intent.bufferDepth, error) ||
            !readPositiveInteger(*stagingObject, "alignment", intent.alignment,
                                 error))
          return failure();
        if (intent.action != "register_to_lds" &&
            intent.action != "global_to_lds") {
          error = "M1.5b.4 supports only register_to_lds or global_to_lds "
                  "staging";
          return failure();
        }
        if (intent.bufferDepth != 1) {
          error = "M1.5b.4 supports only single-slot register staging";
          return failure();
        }
        if (!llvm::isPowerOf2_64(intent.alignment)) {
          error = "pipeline staging alignment must be a power of two";
          return failure();
        }
        if (intent.alignment > std::numeric_limits<int32_t>::max()) {
          error = "pipeline staging alignment exceeds the native attribute "
                  "range";
          return failure();
        }
        if (!stagedValues.insert(intent.valueId).second) {
          error =
              "pipeline delta repeats staged value '" + intent.valueId + "'";
          return failure();
        }
        loop.staging.push_back(std::move(intent));
      }
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
    llvm::json::Array staging;
    for (const PlanPipelineStagingApplyRecord &record : loop.staging) {
      staging.push_back(llvm::json::Object{
          {"value", record.valueId},
          {"action", record.action},
          {"derived_operations_cloned", record.derivedOperationsCloned},
          {"derived_operations_pruned", record.derivedOperationsPruned},
          {"selected_consumer_operands", record.selectedConsumerOperands},
          {"unselected_consumers_preserved",
           record.unselectedConsumersPreserved},
          {"global_loads_eliminated", record.globalLoadsEliminated},
          {"direct_to_lds_copies", record.directToLdsCopies},
          {"async_commits_inserted", record.asyncCommitsInserted},
          {"async_waits_inserted", record.asyncWaitsInserted},
          {"source_live_start_before", record.sourceLiveStartBefore},
          {"source_live_end_before", record.sourceLiveEndBefore},
          {"source_live_start_after", record.sourceLiveStartAfter},
          {"source_live_end_after", record.sourceLiveEndAfter},
          {"register_source_eliminated", record.registerSourceEliminated},
          {"global_access_semantics_preserved",
           record.globalAccessSemanticsPreserved},
          {"logical_live_range_shortened", record.logicalLiveRangeShortened},
      });
    }
    loops.push_back(llvm::json::Object{
        {"loop", loop.loopId},
        {"initiation_interval", loop.initiationInterval},
        {"operations", loop.operationCount},
        {"selected_operations", loop.selectedOperationCount},
        {"moved_operations", loop.movedOperationCount},
        {"imported_dependencies", loop.importedDependencyCount},
        {"skipped_inconsistent_dependencies", loop.skippedDependencyCount},
        {"ring_mutations", loop.ringMutationCount},
        {"staging_mutations", loop.stagingMutationCount},
        {"rewritten_slot_indices", loop.rewrittenSlotIndexCount},
        {"updated_waits", loop.updatedWaitCount},
        {"inserted_barriers", loop.insertedBarrierCount},
        {"logical_lds_bytes_before", loop.logicalLdsBytesBefore},
        {"logical_lds_bytes_after", loop.logicalLdsBytesAfter},
        {"post_rewrite_ddg_verified", loop.postRewriteDdgVerified},
        {"groups", std::move(groups)},
        {"staging", std::move(staging)},
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
      {"changes_new_staging", result.changesNewStaging},
      {"changes_global_staging", result.changesGlobalStaging},
      {"changes_dot_decomposition", false},
      {"post_rewrite_audit_passed", result.postRewriteAuditPassed},
      {"materialization_scope",
       result.changesGlobalStaging ? llvm::json::Value("global_to_lds_staging")
       : result.changesNewStaging ? llvm::json::Value("register_to_lds_staging")
       : result.changesIterationStorage || result.changesSynchronization
           ? llvm::json::Value("existing_lds_ring_and_sync")
           : llvm::json::Value("existing_lds_operation_order")},
  };
  std::string payload;
  llvm::raw_string_ostream stream(payload);
  stream << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(report)));
  return payload;
}

} // namespace mlir::triton::plan
