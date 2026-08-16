#include "triton/Analysis/PlanSchedule.h"

#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/PlanValueGraph.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>

namespace mlir::triton::plan {
namespace {

static bool readString(const llvm::json::Object &object, StringRef key,
                       std::string &destination, std::string &error) {
  std::optional<StringRef> value = object.getString(key);
  if (!value) {
    error = "schedule delta is missing string field '" + key.str() + "'";
    return false;
  }
  destination = value->str();
  return true;
}

static bool readStringArray(const llvm::json::Object &object, StringRef key,
                            std::vector<std::string> &destination,
                            std::string &error) {
  const llvm::json::Array *array = object.getArray(key);
  if (!array) {
    error = "schedule delta is missing array field '" + key.str() + "'";
    return false;
  }
  for (const llvm::json::Value &element : *array) {
    std::optional<StringRef> value = element.getAsString();
    if (!value) {
      error = "schedule delta array '" + key.str() +
              "' contains a non-string element";
      return false;
    }
    destination.push_back(value->str());
  }
  return true;
}

static std::string hashOrder(ArrayRef<std::string> order) {
  llvm::SHA256 hasher;
  for (StringRef operation : order) {
    hasher.update(operation);
    hasher.update(StringRef("\0", 1));
  }
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static bool hasUniqueValues(ArrayRef<std::string> values) {
  std::set<std::string> unique(values.begin(), values.end());
  return unique.size() == values.size();
}

static bool isExplicitlyMovableOperation(Operation *operation) {
  // These final-TTGIR operations are either reads or explicit pure compute
  // sites whose legality is fully represented by the value-graph dependencies.
  StringRef name = operation->getName().getStringRef();
  return name == "tt.load" || name == "amdg.buffer_load" ||
         name == "ttg.local_load" || name == "amdg.scheduled_mfma";
}

static bool isScheduleAnchor(Operation *operation, StringRef identityQuality) {
  if (operation->hasTrait<OpTrait::IsTerminator>() ||
      operation->getNumRegions() != 0 || identityQuality != "semantic")
    return true;
  if (isExplicitlyMovableOperation(operation))
    return false;
  return !isMemoryEffectFree(operation);
}

struct ResolvedBlockSchedule {
  Block *block = nullptr;
  std::vector<Operation *> desiredOperations;
  PlanBlockScheduleApplyRecord record;
};

static LogicalResult fail(PlanScheduleApplyResult &result, std::string &error,
                          Twine message) {
  error = message.str();
  result.error = error;
  return failure();
}

} // namespace

FailureOr<PlanScheduleDelta> parsePlanScheduleDelta(StringRef payload,
                                                    std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(payload);
  if (!parsed) {
    error = llvm::toString(parsed.takeError());
    return failure();
  }
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object) {
    error = "schedule delta root must be a JSON object";
    return failure();
  }

  PlanScheduleDelta delta;
  if (!readString(*object, "schema_version", delta.schemaVersion, error) ||
      !readString(*object, "kernel", delta.kernel, error) ||
      !readString(*object, "input_value_graph_fingerprint",
                  delta.inputValueGraphFingerprint, error) ||
      !readString(*object, "pass_position", delta.passPosition, error))
    return failure();
  if (delta.schemaVersion != kPlanScheduleDeltaSchema) {
    error = "unsupported schedule-delta schema '" + delta.schemaVersion + "'";
    return failure();
  }
  if (delta.passPosition != kFinalStructuredTTGIRPosition) {
    error = "schedule delta targets unsupported pass position '" +
            delta.passPosition + "'";
    return failure();
  }

  const llvm::json::Array *blocks = object->getArray("blocks");
  if (!blocks || blocks->empty()) {
    error = "schedule delta must contain at least one block";
    return failure();
  }
  std::set<std::string> blockIds;
  for (const llvm::json::Value &element : *blocks) {
    const llvm::json::Object *blockObject = element.getAsObject();
    if (!blockObject) {
      error = "schedule delta block must be a JSON object";
      return failure();
    }
    PlanBlockScheduleDelta block;
    if (!readString(*blockObject, "block", block.blockId, error) ||
        !readStringArray(*blockObject, "baseline_order", block.baselineOrder,
                         error) ||
        !readStringArray(*blockObject, "desired_order", block.desiredOrder,
                         error))
      return failure();
    if (std::optional<StringRef> reason = blockObject->getString("reason"))
      block.reason = reason->str();
    if (!blockIds.insert(block.blockId).second) {
      error = "schedule delta repeats block '" + block.blockId + "'";
      return failure();
    }
    if (block.baselineOrder.empty() || !hasUniqueValues(block.baselineOrder) ||
        !hasUniqueValues(block.desiredOrder) ||
        std::set<std::string>(block.baselineOrder.begin(),
                              block.baselineOrder.end()) !=
            std::set<std::string>(block.desiredOrder.begin(),
                                  block.desiredOrder.end())) {
      error = "schedule delta block '" + block.blockId +
              "' does not contain a complete unique permutation";
      return failure();
    }
    delta.blocks.push_back(std::move(block));
  }
  return delta;
}

LogicalResult applyPlanSchedule(FuncOp function, const PlanScheduleDelta &delta,
                                PlanScheduleApplyResult &result,
                                std::string &error, bool strict) {
  result.kernel = function.getName().str();
  if (delta.kernel != function.getName())
    return fail(result, error, "schedule delta kernel does not match function");

  llvm::DenseMap<Operation *, std::string> operationBindings;
  FailureOr<PlanValueGraph> graph =
      PlanValueGraph::build(function, &operationBindings);
  if (failed(graph) || (strict && failed(graph->verify(/*strict=*/true))))
    return fail(result, error,
                "failed to build a strict pre-apply value graph");
  result.inputValueGraphFingerprint = graph->getSemanticFingerprint().str();
  if (delta.inputValueGraphFingerprint != graph->getSemanticFingerprint())
    return fail(result, error,
                "schedule delta value-graph fingerprint does not match");

  std::map<std::string, Operation *> operationById;
  for (const auto &[operation, id] : operationBindings)
    operationById[id] = operation;
  std::map<std::string, StringRef> identityQuality;
  for (const PlanOperationRecord &operation : graph->getOperations())
    identityQuality[operation.id] = operation.identityQuality;
  std::map<std::string, const PlanBlockRecord *> blockById;
  for (const PlanBlockRecord &block : graph->getBlocks())
    blockById[block.id] = &block;

  SmallVector<ResolvedBlockSchedule> resolved;
  for (const PlanBlockScheduleDelta &requested : delta.blocks) {
    auto blockIt = blockById.find(requested.blockId);
    if (blockIt == blockById.end())
      return fail(result, error,
                  "schedule delta refers to unknown block '" +
                      requested.blockId + "'");
    const PlanBlockRecord &blockRecord = *blockIt->second;
    if (requested.baselineOrder != blockRecord.operations)
      return fail(result, error,
                  "schedule delta baseline order mismatch for '" +
                      requested.blockId + "'");

    ResolvedBlockSchedule current;
    current.record.blockId = requested.blockId;
    current.record.baselineOrderHash = hashOrder(requested.baselineOrder);
    current.record.desiredOrderHash = hashOrder(requested.desiredOrder);
    current.record.operationCount = requested.desiredOrder.size();

    std::map<std::string, int64_t> baselinePosition;
    std::map<std::string, int64_t> desiredPosition;
    for (auto [position, id] : llvm::enumerate(requested.baselineOrder))
      baselinePosition[id] = position;
    for (auto [position, id] : llvm::enumerate(requested.desiredOrder))
      desiredPosition[id] = position;

    for (StringRef id : requested.desiredOrder) {
      auto operationIt = operationById.find(id.str());
      if (operationIt == operationById.end())
        return fail(result, error,
                    "schedule delta refers to unknown operation '" + id.str() +
                        "'");
      Operation *operation = operationIt->second;
      if (!current.block)
        current.block = operation->getBlock();
      if (operation->getBlock() != current.block)
        return fail(result, error,
                    "schedule delta attempts cross-block movement");
      current.desiredOperations.push_back(operation);
    }
    if (!current.block)
      return fail(result, error, "schedule delta resolved an empty block");

    for (StringRef id : requested.baselineOrder) {
      auto operationIt = operationById.find(id.str());
      if (operationIt == operationById.end())
        return fail(result, error, "baseline operation cannot be resolved");
      Operation *operation = operationIt->second;
      bool anchor = isScheduleAnchor(operation, identityQuality[id.str()]);
      if (!anchor)
        continue;
      ++result.anchorCount;
      if (baselinePosition[id.str()] != desiredPosition[id.str()])
        return fail(result, error,
                    "schedule delta moves pinned operation '" + id.str() + "'");
    }

    for (const PlanDependencyEdge &dependency : graph->getDependencyEdges()) {
      if (dependency.iterationDistance != 0 ||
          dependency.sourceOperationId == dependency.destinationOperationId)
        continue;
      auto source = desiredPosition.find(dependency.sourceOperationId);
      auto destination =
          desiredPosition.find(dependency.destinationOperationId);
      if (source == desiredPosition.end() ||
          destination == desiredPosition.end())
        continue;
      ++result.checkedDependencyCount;
      if (source->second >= destination->second)
        return fail(result, error,
                    "schedule delta reverses distance-zero dependency '" +
                        dependency.id + "'");
    }

    for (size_t position = 0; position < requested.baselineOrder.size();
         ++position)
      if (requested.baselineOrder[position] != requested.desiredOrder[position])
        ++current.record.movedOperationCount;
    result.movedOperationCount += current.record.movedOperationCount;
    resolved.push_back(std::move(current));
  }

  // Every requested block is fully validated before the first mutation.
  for (ResolvedBlockSchedule &block : resolved) {
    if (block.record.movedOperationCount == 0)
      continue;
    Operation *terminator = block.block->getTerminator();
    for (Operation *operation : block.desiredOperations)
      if (operation != terminator)
        operation->moveBefore(terminator);
  }
  if (failed(mlir::verify(function)))
    return fail(result, error, "MLIR verification failed after schedule apply");

  FailureOr<PlanValueGraph> postGraph = PlanValueGraph::build(function);
  if (failed(postGraph) ||
      (strict && failed(postGraph->verify(/*strict=*/true))))
    return fail(result, error,
                "failed to build a strict post-apply value graph");
  result.outputValueGraphFingerprint =
      postGraph->getSemanticFingerprint().str();
  if (result.outputValueGraphFingerprint != result.inputValueGraphFingerprint)
    return fail(result, error,
                "stable operation/value identity changed after schedule apply");

  for (const ResolvedBlockSchedule &block : resolved)
    result.blocks.push_back(block.record);
  result.accepted = true;
  return success();
}

std::string
serializePlanScheduleApplyReport(const PlanScheduleApplyResult &result) {
  llvm::json::Array blocks;
  for (const PlanBlockScheduleApplyRecord &block : result.blocks)
    blocks.push_back(llvm::json::Object{
        {"block", block.blockId},
        {"baseline_order_hash", block.baselineOrderHash},
        {"desired_order_hash", block.desiredOrderHash},
        {"operation_count", block.operationCount},
        {"moved_operation_count", block.movedOperationCount},
    });
  llvm::json::Object report{
      {"schema_version", "plan-apply-report/0.1"},
      {"accepted", result.accepted},
      {"kernel", result.kernel},
      {"input_value_graph_fingerprint", result.inputValueGraphFingerprint},
      {"output_value_graph_fingerprint", result.outputValueGraphFingerprint},
      {"error", result.error.empty() ? llvm::json::Value(nullptr)
                                     : llvm::json::Value(result.error)},
      {"checked_distance_zero_dependencies", result.checkedDependencyCount},
      {"anchors", result.anchorCount},
      {"moved_operations", result.movedOperationCount},
      {"blocks", std::move(blocks)},
      {"changes_iteration_storage", false},
      {"changes_synchronization", false},
      {"changes_iteration_placement", false},
      {"changes_dot_decomposition", false},
  };
  std::string payload;
  llvm::raw_string_ostream stream(payload);
  stream << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(report)));
  return payload;
}

} // namespace mlir::triton::plan
