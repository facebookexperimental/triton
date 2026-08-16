#include "triton/Analysis/PlanValueGraph.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <climits>
#include <map>
#include <set>
#include <tuple>

namespace mlir::triton::plan {
namespace {

struct TemporaryOperation {
  Operation *operation = nullptr;
  unsigned ordinal = 0;
  std::string locator;
  std::string baseSignature;
  std::string digest;
};

struct TemporaryValue {
  Value value;
  std::string locator;
  std::string originKind;
  std::string argumentRole;
  Operation *definingOperation = nullptr;
  int64_t resultNumber = -1;
  int64_t argumentNumber = -1;
  std::string baseSignature;
  std::string digest;
};

struct LineageParameters {
  std::map<std::string, std::string> strings;
  std::map<std::string, int64_t> integers;
};

static std::string printType(Type type) {
  std::string result;
  llvm::raw_string_ostream os(result);
  type.print(os);
  return os.str();
}

static std::string printAttribute(Attribute attribute) {
  std::string result;
  llvm::raw_string_ostream os(result);
  attribute.print(os);
  return os.str();
}

static std::string hashText(StringRef text) {
  llvm::SHA256 hasher;
  hasher.update(text);
  auto digest = hasher.final();
  return llvm::toHex(digest, /*LowerCase=*/true);
}

static bool isPlacementOnlyAttribute(StringRef name) {
  return name.starts_with("plan.") || name.starts_with("ttg.modulo_") ||
         name.contains("schedule_cycle") || name.contains("schedule_stage") ||
         name.contains("schedule_cluster") || name == "ttg.cluster" ||
         name == "ttg.cluster_id" || name == "ttg.stage" ||
         name == "ttg.warp_group";
}

static std::string operationBaseSignature(Operation *op) {
  std::string signature;
  llvm::raw_string_ostream os(signature);
  os << op->getName().getStringRef() << "|operands=" << op->getNumOperands()
     << "|results=" << op->getNumResults()
     << "|regions=" << op->getNumRegions();
  for (auto [index, result] : llvm::enumerate(op->getResults()))
    os << "|r" << index << "=" << printType(result.getType());

  llvm::DenseSet<StringAttr> discardable;
  for (NamedAttribute attr : op->getDiscardableAttrs())
    discardable.insert(attr.getName());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (discardable.contains(attr.getName()) ||
        isPlacementOnlyAttribute(name) ||
        name == SymbolTable::getSymbolAttrName())
      continue;
    os << "|a:" << name << "=" << printAttribute(attr.getValue());
  }

  if (Operation *parent = op->getParentOp()) {
    if (!isa<FuncOp>(parent)) {
      os << "|parent=" << parent->getName().getStringRef();
      Region *region = op->getParentRegion();
      if (region)
        os << ":region=" << region->getRegionNumber();
    }
  }
  return os.str();
}

static std::string blockArgumentRole(BlockArgument argument) {
  Block *block = argument.getOwner();
  Operation *parent = block->getParentOp();
  if (auto function = dyn_cast_or_null<FuncOp>(parent))
    return "function_argument";
  if (auto loop = dyn_cast_or_null<scf::ForOp>(parent)) {
    if (argument == loop.getInductionVar())
      return "loop_induction";
    return "loop_iter_arg";
  }
  if (auto loop = dyn_cast_or_null<scf::WhileOp>(parent)) {
    if (block == loop.getBeforeBody())
      return "while_before_arg";
    if (block == loop.getAfterBody())
      return "while_after_arg";
  }
  return "block_argument";
}

static std::string valueCategory(Type type) {
  if (isa<RankedTensorType>(type))
    return "tensor_register_logical";
  if (isa<gpu::MemDescType>(type))
    return "memdesc";
  if (isa<triton::PointerType>(type))
    return "pointer";
  std::string typeText = printType(type);
  if (StringRef(typeText).contains("token"))
    return "token";
  if (type.isIntOrIndexOrFloat())
    return "scalar";
  return "other";
}

static std::optional<int64_t> getLogicalBytes(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return std::nullopt;
  Type elementType = shaped.getElementType();
  if (!elementType.isIntOrFloat())
    return std::nullopt;
  int64_t bits = elementType.getIntOrFloatBitWidth();
  if (bits <= 0)
    return std::nullopt;
  int64_t elements = shaped.getNumElements();
  if (elements < 0 || elements > (INT64_MAX - 7) / bits)
    return std::nullopt;
  return (elements * bits + 7) / 8;
}

static void fillTypeFields(PlanValueRecord &record, Type type) {
  record.type = printType(type);
  record.category = valueCategory(type);
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    record.logicalShape.assign(shaped.getShape().begin(),
                               shaped.getShape().end());
    record.elementType = printType(shaped.getElementType());
    record.logicalBytes = getLogicalBytes(type);
  }
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (tensor.getEncoding())
      record.encoding = printAttribute(tensor.getEncoding());
  } else if (auto memdesc = dyn_cast<gpu::MemDescType>(type)) {
    if (memdesc.getEncoding())
      record.encoding = printAttribute(memdesc.getEncoding());
  }
}

static void
enumerateRegion(Region &region, StringRef prefix,
                std::vector<TemporaryOperation> &operations,
                std::vector<TemporaryValue> &values,
                llvm::DenseMap<Operation *, unsigned> &operationIndex,
                llvm::DenseMap<Value, unsigned> &valueIndex) {
  for (auto [blockIndex, block] : llvm::enumerate(region.getBlocks())) {
    std::string blockPrefix = prefix.str() + "/b" + std::to_string(blockIndex);
    for (auto [argumentIndex, argument] :
         llvm::enumerate(block.getArguments())) {
      TemporaryValue temporary;
      temporary.value = argument;
      temporary.locator = blockPrefix + "/arg" + std::to_string(argumentIndex);
      temporary.originKind = "block_argument";
      temporary.argumentRole = blockArgumentRole(argument);
      temporary.argumentNumber = argumentIndex;
      std::string parentName =
          block.getParentOp()
              ? block.getParentOp()->getName().getStringRef().str()
              : "none";
      temporary.baseSignature = "block_arg|role=" + temporary.argumentRole +
                                "|parent=" + parentName +
                                "|index=" + std::to_string(argumentIndex) +
                                "|type=" + printType(argument.getType());
      valueIndex[argument] = values.size();
      values.push_back(std::move(temporary));
    }

    for (auto [opIndex, op] : llvm::enumerate(block.getOperations())) {
      TemporaryOperation temporary;
      temporary.operation = &op;
      temporary.ordinal = operations.size();
      temporary.locator = blockPrefix + "/o" + std::to_string(opIndex);
      temporary.baseSignature = operationBaseSignature(&op);
      operationIndex[&op] = operations.size();
      operations.push_back(std::move(temporary));

      for (auto [resultIndex, result] : llvm::enumerate(op.getResults())) {
        TemporaryValue value;
        value.value = result;
        value.locator =
            operations.back().locator + "/r" + std::to_string(resultIndex);
        value.originKind = "operation_result";
        value.definingOperation = &op;
        value.resultNumber = resultIndex;
        value.baseSignature =
            "op_result|op=" + operations.back().baseSignature +
            "|result=" + std::to_string(resultIndex) +
            "|type=" + printType(result.getType());
        valueIndex[result] = values.size();
        values.push_back(std::move(value));
      }

      for (auto [regionIndex, nested] : llvm::enumerate(op.getRegions()))
        enumerateRegion(nested,
                        operations.back().locator + "/r" +
                            std::to_string(regionIndex),
                        operations, values, operationIndex, valueIndex);
    }
  }
}

static llvm::json::Array intArray(ArrayRef<int64_t> values) {
  llvm::json::Array result;
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

static llvm::json::Object
slotExpressionJSON(const PlanSlotExpression &expression) {
  return llvm::json::Object{
      {"kind", expression.kind},
      {"text", expression.text},
      {"base_value", expression.baseValueId},
      {"coefficient", expression.coefficient},
      {"offset", expression.offset},
      {"modulus", expression.modulus},
      {"possible_slots", intArray(expression.possibleSlots)},
  };
}

static llvm::json::Object slotPathJSON(const PlanSlotPath &path) {
  llvm::json::Array indices;
  for (const PlanSlotExpression &expression : path.indices)
    indices.push_back(slotExpressionJSON(expression));
  return llvm::json::Object{{"root_value", path.rootValueId},
                            {"indices", std::move(indices)}};
}

static llvm::json::Object liveSegmentJSON(const PlanLiveSegment &segment) {
  llvm::json::Object object{
      {"value", segment.valueId},
      {"block", segment.blockId},
      {"start_position", segment.startPosition},
      {"end_position", segment.endPosition},
      {"live_in", segment.liveIn},
      {"live_out", segment.liveOut},
      {"crosses_backedge", segment.crossesBackedge},
      {"iteration_distance", segment.iterationDistance},
  };
  if (!segment.startOperationId.empty())
    object["start_operation"] = segment.startOperationId;
  else
    object["start_operation"] = nullptr;
  if (!segment.endOperationId.empty())
    object["end_operation"] = segment.endOperationId;
  else
    object["end_operation"] = nullptr;
  return object;
}

static LineageParameters parametersFromAttributes(Operation *op,
                                                  ArrayRef<StringRef> names) {
  LineageParameters parameters;
  for (StringRef name : names)
    if (Attribute attribute = op->getAttr(name))
      parameters.strings[name.str()] = printAttribute(attribute);
  return parameters;
}

static void addLineage(std::vector<PlanLineageEdge> &edges,
                       const llvm::DenseMap<Value, std::string> &ids,
                       Value source, Value destination, StringRef kind,
                       int64_t distance = 0,
                       LineageParameters parameters = {}) {
  auto sourceIt = ids.find(source);
  auto destinationIt = ids.find(destination);
  if (sourceIt == ids.end() || destinationIt == ids.end())
    return;
  edges.push_back(PlanLineageEdge{
      sourceIt->second, destinationIt->second, kind.str(), distance,
      std::move(parameters.strings), std::move(parameters.integers)});
}

static void
addTransformationLineage(Operation *op,
                         const llvm::DenseMap<Value, std::string> &ids,
                         std::vector<PlanLineageEdge> &edges) {
  if (op->getNumOperands() == 0 || op->getNumResults() == 0)
    return;
  StringRef name = op->getName().getStringRef();
  StringRef kind;
  LineageParameters parameters;
  if (name == "amdg.extract_slice") {
    kind = "extract_slice";
    parameters = parametersFromAttributes(op, {"static_offsets"});
  } else if (name == "tt.reshape") {
    kind = "reshape";
    parameters =
        parametersFromAttributes(op, {"allow_reorder", "efficient_layout"});
  } else if (name == "tt.trans") {
    kind = "transpose";
    parameters = parametersFromAttributes(op, {"order"});
  } else if (name == "amdg.in_thread_transpose") {
    kind = "in_thread_transpose";
  } else if (name == "ttg.convert_layout") {
    kind = "convert_layout";
  } else {
    return;
  }
  parameters.strings["source_type"] = printType(op->getOperand(0).getType());
  parameters.strings["destination_type"] =
      printType(op->getResult(0).getType());
  addLineage(edges, ids, op->getOperand(0), op->getResult(0), kind,
             /*distance=*/0, std::move(parameters));
}

static void addStructuredLineage(
    Operation *op, const llvm::DenseMap<Value, std::string> &ids,
    std::vector<PlanLineageEdge> &edges,
    std::vector<PlanDiagnostic> &diagnostics,
    const llvm::DenseMap<Operation *, std::string> &operationIds) {
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    auto iterArgs = loop.getRegionIterArgs();
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [index, init, iterArg, yielded, result] :
         llvm::enumerate(loop.getInitArgs(), iterArgs, yield.getResults(),
                         loop.getResults())) {
      LineageParameters parameters;
      parameters.integers["slot"] = index;
      addLineage(edges, ids, init, iterArg, "loop_init", 0, parameters);
      addLineage(edges, ids, yielded, iterArg, "loop_backedge", 1, parameters);
      addLineage(edges, ids, yielded, result, "loop_exit", 0, parameters);
    }
    return;
  }

  if (auto loop = dyn_cast<scf::WhileOp>(op)) {
    if (loop.getInits().size() != loop.getBeforeArguments().size() ||
        loop.getYieldOp().getResults().size() !=
            loop.getBeforeArguments().size()) {
      diagnostics.push_back(
          {"error", "unsupported_while_mapping",
           "scf.while init/yield arity does not match before arguments",
           operationIds.lookup(op), ""});
      return;
    }
    for (auto [index, init, beforeArg, yielded] :
         llvm::enumerate(loop.getInits(), loop.getBeforeArguments(),
                         loop.getYieldOp().getResults())) {
      LineageParameters parameters;
      parameters.integers["slot"] = index;
      addLineage(edges, ids, init, beforeArg, "loop_init", 0, parameters);
      addLineage(edges, ids, yielded, beforeArg, "loop_backedge", 1,
                 parameters);
    }
    auto forwarded = loop.getConditionOp().getArgs();
    if (forwarded.size() != loop.getAfterArguments().size() ||
        forwarded.size() != loop.getResults().size()) {
      diagnostics.push_back(
          {"error", "unsupported_while_forwarding",
           "scf.while condition forwarding is not one-to-one with results",
           operationIds.lookup(op), ""});
      return;
    }
    for (auto [index, value, afterArg, result] : llvm::enumerate(
             forwarded, loop.getAfterArguments(), loop.getResults())) {
      LineageParameters parameters;
      parameters.integers["slot"] = index;
      addLineage(edges, ids, value, afterArg, "loop_forward", 0, parameters);
      addLineage(edges, ids, value, result, "loop_exit", 0, parameters);
    }
    return;
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    auto addBranch = [&](Region &region, int64_t branch) {
      if (region.empty())
        return;
      auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
      if (!yield || yield.getResults().size() != ifOp.getResults().size()) {
        diagnostics.push_back({"error", "unsupported_if_yield",
                               "scf.if yield/result arity mismatch",
                               operationIds.lookup(op), ""});
        return;
      }
      for (auto [index, source, destination] :
           llvm::enumerate(yield.getResults(), ifOp.getResults())) {
        LineageParameters parameters;
        parameters.integers["branch"] = branch;
        parameters.integers["result"] = index;
        addLineage(edges, ids, source, destination, "branch_yield", 0,
                   std::move(parameters));
      }
    };
    addBranch(ifOp.getThenRegion(), 0);
    addBranch(ifOp.getElseRegion(), 1);
  }
}

} // namespace

FailureOr<PlanValueGraph> PlanValueGraph::build(FuncOp function) {
  PlanValueGraph graph;
  graph.functionName = function.getName().str();

  std::vector<TemporaryOperation> temporaryOperations;
  std::vector<TemporaryValue> temporaryValues;
  llvm::DenseMap<Operation *, unsigned> operationIndex;
  llvm::DenseMap<Value, unsigned> valueIndex;
  enumerateRegion(function.getBody(), "func/" + function.getName().str(),
                  temporaryOperations, temporaryValues, operationIndex,
                  valueIndex);

  for (TemporaryOperation &operation : temporaryOperations)
    operation.digest = hashText(operation.baseSignature);
  for (TemporaryValue &value : temporaryValues)
    value.digest = hashText(value.baseSignature);

  unsigned rounds = std::min<unsigned>(
      std::max<size_t>(temporaryOperations.size() + temporaryValues.size(), 1),
      256);
  for (unsigned round = 0; round < rounds; ++round) {
    std::vector<std::string> nextOperations(temporaryOperations.size());
    std::vector<std::string> nextValues(temporaryValues.size());
    for (auto [index, temporary] : llvm::enumerate(temporaryOperations)) {
      std::string signature = temporary.baseSignature;
      for (auto [operandNumber, operand] :
           llvm::enumerate(temporary.operation->getOperands())) {
        auto valueIt = valueIndex.find(operand);
        signature += "|operand:" + std::to_string(operandNumber) + "=" +
                     (valueIt == valueIndex.end()
                          ? ("external:" + printType(operand.getType()))
                          : temporaryValues[valueIt->second].digest);
      }
      nextOperations[index] = hashText(signature);
    }
    for (auto [index, temporary] : llvm::enumerate(temporaryValues)) {
      std::string signature = temporary.baseSignature;
      if (temporary.definingOperation) {
        auto opIt = operationIndex.find(temporary.definingOperation);
        if (opIt != operationIndex.end())
          signature += "|def=" + nextOperations[opIt->second];
      }
      std::vector<std::string> uses;
      for (OpOperand &use : temporary.value.getUses()) {
        auto opIt = operationIndex.find(use.getOwner());
        if (opIt == operationIndex.end())
          continue;
        uses.push_back(temporaryOperations[opIt->second].digest + ":" +
                       std::to_string(use.getOperandNumber()));
      }
      llvm::sort(uses);
      for (StringRef use : uses)
        signature += "|use=" + use.str();
      nextValues[index] = hashText(signature);
    }
    for (auto &&[temporary, digest] :
         llvm::zip(temporaryOperations, nextOperations))
      temporary.digest = std::move(digest);
    for (auto &&[temporary, digest] : llvm::zip(temporaryValues, nextValues))
      temporary.digest = std::move(digest);
  }

  llvm::DenseMap<Operation *, std::string> operationIds;
  llvm::DenseMap<Operation *, StringRef> operationIdentityQuality;
  std::map<std::string, std::vector<unsigned>> operationCollisions;
  for (auto [index, operation] : llvm::enumerate(temporaryOperations))
    operationCollisions[operation.digest].push_back(index);
  for (auto &[digest, indexes] : operationCollisions) {
    llvm::sort(indexes, [&](unsigned lhs, unsigned rhs) {
      return temporaryOperations[lhs].locator <
             temporaryOperations[rhs].locator;
    });
    for (auto [collisionIndex, index] : llvm::enumerate(indexes)) {
      TemporaryOperation &temporary = temporaryOperations[index];
      std::string id = "op:" + digest;
      std::string quality = "semantic";
      if (indexes.size() > 1) {
        id += ":" + std::to_string(collisionIndex);
        quality = "fallback_ordinal";
        graph.diagnostics.push_back(
            {"warning", "identity_collision",
             "structurally symmetric operations required ordinal fallback", id,
             ""});
      }
      operationIds[temporary.operation] = id;
      operationIdentityQuality[temporary.operation] =
          quality == "semantic" ? StringRef("semantic")
                                : StringRef("fallback_ordinal");
      graph.operations.push_back(
          {id, temporary.operation->getName().getStringRef().str(),
           temporary.locator, quality, temporary.ordinal,
           temporary.operation->getNumResults()});
    }
  }

  llvm::DenseMap<Value, std::string> valueIds;
  std::map<std::string, std::vector<unsigned>> valueCollisions;
  for (auto [index, value] : llvm::enumerate(temporaryValues))
    if (!value.definingOperation)
      valueCollisions[value.digest].push_back(index);

  for (TemporaryValue &temporary : temporaryValues) {
    if (!temporary.definingOperation)
      continue;
    std::string id =
        "value:" + operationIds.lookup(temporary.definingOperation) +
        ":result:" + std::to_string(temporary.resultNumber);
    valueIds[temporary.value] = id;
  }
  for (auto &[digest, indexes] : valueCollisions) {
    llvm::sort(indexes, [&](unsigned lhs, unsigned rhs) {
      return temporaryValues[lhs].locator < temporaryValues[rhs].locator;
    });
    for (auto [collisionIndex, index] : llvm::enumerate(indexes)) {
      std::string id = "value:argument:" + digest;
      if (indexes.size() > 1)
        id += ":" + std::to_string(collisionIndex);
      valueIds[temporaryValues[index].value] = id;
    }
  }

  for (TemporaryValue &temporary : temporaryValues) {
    PlanValueRecord record;
    record.id = valueIds.lookup(temporary.value);
    record.locator = temporary.locator;
    record.originKind = temporary.originKind;
    record.argumentRole = temporary.argumentRole;
    record.resultNumber = temporary.resultNumber;
    record.argumentNumber = temporary.argumentNumber;
    record.identityQuality =
        temporary.definingOperation
            ? operationIdentityQuality.lookup(temporary.definingOperation).str()
            : "semantic";
    if (!temporary.definingOperation &&
        valueCollisions[temporary.digest].size() > 1) {
      record.identityQuality = "fallback_ordinal";
      graph.diagnostics.push_back(
          {"warning", "identity_collision",
           "structurally symmetric arguments required ordinal fallback", "",
           record.id});
    }
    if (temporary.definingOperation)
      record.definingOperationId =
          operationIds.lookup(temporary.definingOperation);
    fillTypeFields(record, temporary.value.getType());
    for (OpOperand &use : temporary.value.getUses()) {
      std::string consumer = operationIds.lookup(use.getOwner());
      if (!consumer.empty())
        record.uses.push_back({consumer, use.getOperandNumber()});
    }
    llvm::sort(record.uses, [](const PlanUse &lhs, const PlanUse &rhs) {
      return std::tie(lhs.operationId, lhs.operandNumber) <
             std::tie(rhs.operationId, rhs.operandNumber);
    });
    graph.values.push_back(std::move(record));
  }

  for (TemporaryOperation &temporary : temporaryOperations) {
    addTransformationLineage(temporary.operation, valueIds, graph.lineageEdges);
    addStructuredLineage(temporary.operation, valueIds, graph.lineageEdges,
                         graph.diagnostics, operationIds);
  }

  FailureOr<PlanLivenessResult> liveness =
      analyzePlanLiveness(function, operationIds, valueIds, graph.lineageEdges);
  if (failed(liveness))
    return failure();
  graph.blocks = std::move(liveness->blocks);
  graph.liveSegments = std::move(liveness->liveSegments);
  graph.aliases = std::move(liveness->aliases);
  graph.memoryAccesses = std::move(liveness->memoryAccesses);
  graph.ldsAllocations = std::move(liveness->ldsAllocations);
  llvm::append_range(graph.diagnostics, liveness->diagnostics);

  llvm::sort(graph.operations,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(graph.values,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(graph.lineageEdges, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.destination, lhs.source, lhs.kind,
                    lhs.iterationDistance) < std::tie(rhs.destination,
                                                      rhs.source, rhs.kind,
                                                      rhs.iterationDistance);
  });
  llvm::sort(graph.diagnostics, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.severity, lhs.code, lhs.operationId, lhs.valueId,
                    lhs.message) < std::tie(rhs.severity, rhs.code,
                                            rhs.operationId, rhs.valueId,
                                            rhs.message);
  });

  std::vector<std::string> identity;
  for (const auto &operation : graph.operations)
    identity.push_back(operation.id);
  for (const auto &value : graph.values)
    identity.push_back(value.id);
  llvm::sort(identity);
  graph.semanticFingerprint = hashText(llvm::join(identity, "|"));

  if (failed(graph.verify(/*strict=*/false)))
    return failure();
  return graph;
}

LogicalResult PlanValueGraph::verify(bool strict) const {
  std::set<std::string> operationIds;
  std::set<std::string> valueIds;
  std::map<std::string, int64_t> blockSizes;
  for (const auto &operation : operations)
    if (operation.id.empty() || !operationIds.insert(operation.id).second)
      return failure();
  for (const auto &value : values)
    if (value.id.empty() || !valueIds.insert(value.id).second)
      return failure();

  for (const PlanBlockRecord &block : blocks) {
    if (block.id.empty() ||
        !blockSizes.emplace(block.id, block.operations.size()).second)
      return failure();
    if (!block.parentOperationId.empty() &&
        !operationIds.count(block.parentOperationId))
      return failure();
    for (StringRef operationId : block.operations)
      if (!operationIds.count(operationId.str()))
        return failure();
  }

  for (const auto &value : values) {
    if (!value.definingOperationId.empty() &&
        !operationIds.count(value.definingOperationId))
      return failure();
    for (const auto &use : value.uses)
      if (!operationIds.count(use.operationId))
        return failure();
  }
  for (const auto &edge : lineageEdges) {
    if (!valueIds.count(edge.source) || !valueIds.count(edge.destination))
      return failure();
    if (edge.kind == "loop_backedge") {
      if (edge.iterationDistance < 1)
        return failure();
    } else if (edge.iterationDistance != 0) {
      return failure();
    }
  }
  auto verifySegment = [&](const PlanLiveSegment &segment) {
    auto block = blockSizes.find(segment.blockId);
    if (!valueIds.count(segment.valueId) || block == blockSizes.end() ||
        segment.startPosition < 0 ||
        segment.startPosition > segment.endPosition ||
        segment.endPosition > block->second || segment.iterationDistance < 0)
      return failure();
    if (!segment.startOperationId.empty() &&
        !operationIds.count(segment.startOperationId))
      return failure();
    if (!segment.endOperationId.empty() &&
        !operationIds.count(segment.endOperationId))
      return failure();
    if (segment.crossesBackedge != (segment.iterationDistance > 0))
      return failure();
    return success();
  };
  for (const PlanLiveSegment &segment : liveSegments)
    if (failed(verifySegment(segment)))
      return failure();

  auto verifySlotPaths = [&](ArrayRef<PlanSlotPath> paths) {
    for (const PlanSlotPath &path : paths) {
      if (!valueIds.count(path.rootValueId))
        return failure();
      for (const PlanSlotExpression &index : path.indices) {
        if (!index.baseValueId.empty() && !valueIds.count(index.baseValueId))
          return failure();
        if (index.modulus < 0)
          return failure();
      }
    }
    return success();
  };
  for (const PlanAliasRecord &alias : aliases) {
    if (!valueIds.count(alias.valueId) ||
        (!alias.sourceValueId.empty() &&
         !valueIds.count(alias.sourceValueId)) ||
        failed(verifySlotPaths(alias.slotPaths)))
      return failure();
    for (StringRef root : alias.rootValueIds)
      if (!valueIds.count(root.str()))
        return failure();
  }
  for (const PlanMemoryAccess &access : memoryAccesses) {
    if (!operationIds.count(access.operationId) ||
        !valueIds.count(access.valueId) ||
        failed(verifySlotPaths(access.slotPaths)))
      return failure();
    for (StringRef root : access.rootValueIds)
      if (!valueIds.count(root.str()))
        return failure();
  }
  for (const PlanLdsAllocationRecord &allocation : ldsAllocations) {
    if (!valueIds.count(allocation.rootValueId) ||
        !operationIds.count(allocation.allocationOperationId))
      return failure();
    for (StringRef alias : allocation.aliases)
      if (!valueIds.count(alias.str()))
        return failure();
    for (const PlanLiveSegment &segment : allocation.liveSegments)
      if (segment.valueId != allocation.rootValueId ||
          failed(verifySegment(segment)))
        return failure();
  }
  if (strict)
    for (const auto &diagnostic : diagnostics)
      if (diagnostic.severity == "error")
        return failure();
  return success();
}

llvm::json::Object PlanValueGraph::toJSON() const {
  llvm::json::Array operationArray;
  for (const auto &operation : operations)
    operationArray.push_back(llvm::json::Object{
        {"id", operation.id},
        {"kind", operation.kind},
        {"locator", operation.locator},
        {"identity_quality", operation.identityQuality},
        {"ordinal", static_cast<int64_t>(operation.ordinal)},
        {"result_count", static_cast<int64_t>(operation.resultCount)},
    });

  llvm::json::Array valueArray;
  for (const auto &value : values) {
    llvm::json::Array uses;
    for (const auto &use : value.uses)
      uses.push_back(llvm::json::Object{
          {"operation", use.operationId},
          {"operand", static_cast<int64_t>(use.operandNumber)},
      });
    llvm::json::Object origin{{"kind", value.originKind}};
    if (!value.definingOperationId.empty())
      origin["operation"] = value.definingOperationId;
    if (value.resultNumber >= 0)
      origin["result"] = value.resultNumber;
    if (value.argumentNumber >= 0)
      origin["argument"] = value.argumentNumber;
    if (!value.argumentRole.empty())
      origin["role"] = value.argumentRole;

    llvm::json::Object type{{"mlir", value.type},
                            {"category", value.category},
                            {"logical_shape", intArray(value.logicalShape)}};
    if (!value.elementType.empty())
      type["element_type"] = value.elementType;
    if (!value.encoding.empty())
      type["encoding"] = value.encoding;
    if (value.logicalBytes)
      type["logical_bytes"] = *value.logicalBytes;
    else
      type["logical_bytes"] = nullptr;
    type["physical_register_bytes"] = nullptr;
    type["physical_register_interval"] = nullptr;

    valueArray.push_back(llvm::json::Object{
        {"id", value.id},
        {"locator", value.locator},
        {"identity_quality", value.identityQuality},
        {"origin", std::move(origin)},
        {"type", std::move(type)},
        {"uses", std::move(uses)},
    });
  }

  llvm::json::Array lineageArray;
  for (const auto &edge : lineageEdges) {
    llvm::json::Object parameters;
    for (const auto &[name, value] : edge.stringParameters)
      parameters[name] = value;
    for (const auto &[name, value] : edge.integerParameters)
      parameters[name] = value;
    lineageArray.push_back(llvm::json::Object{
        {"source", edge.source},
        {"destination", edge.destination},
        {"kind", edge.kind},
        {"iteration_distance", edge.iterationDistance},
        {"parameters", std::move(parameters)},
    });
  }

  llvm::json::Array diagnosticArray;
  for (const auto &diagnostic : diagnostics) {
    llvm::json::Object object{{"severity", diagnostic.severity},
                              {"code", diagnostic.code},
                              {"message", diagnostic.message}};
    if (!diagnostic.operationId.empty())
      object["operation"] = diagnostic.operationId;
    if (!diagnostic.valueId.empty())
      object["value"] = diagnostic.valueId;
    diagnosticArray.push_back(std::move(object));
  }

  llvm::json::Array blockArray;
  for (const PlanBlockRecord &block : blocks) {
    llvm::json::Array blockOperations;
    for (StringRef operation : block.operations)
      blockOperations.push_back(operation);
    blockArray.push_back(llvm::json::Object{
        {"id", block.id},
        {"parent_operation", block.parentOperationId.empty()
                                 ? llvm::json::Value(nullptr)
                                 : llvm::json::Value(block.parentOperationId)},
        {"region_number", block.regionNumber},
        {"block_number", block.blockNumber},
        {"operations", std::move(blockOperations)},
    });
  }

  llvm::json::Array liveSegmentArray;
  for (const PlanLiveSegment &segment : liveSegments)
    liveSegmentArray.push_back(liveSegmentJSON(segment));

  llvm::json::Array aliasArray;
  for (const PlanAliasRecord &alias : aliases) {
    llvm::json::Array roots;
    for (StringRef root : alias.rootValueIds)
      roots.push_back(root);
    llvm::json::Array paths;
    for (const PlanSlotPath &path : alias.slotPaths)
      paths.push_back(slotPathJSON(path));
    aliasArray.push_back(llvm::json::Object{
        {"value", alias.valueId},
        {"view_kind", alias.viewKind},
        {"source_value", alias.sourceValueId.empty()
                             ? llvm::json::Value(nullptr)
                             : llvm::json::Value(alias.sourceValueId)},
        {"root_values", std::move(roots)},
        {"static_offsets", intArray(alias.staticOffsets)},
        {"order", intArray(alias.order)},
        {"slot_paths", std::move(paths)},
        {"physical_lds_offset", nullptr},
    });
  }

  llvm::json::Array accessArray;
  for (const PlanMemoryAccess &access : memoryAccesses) {
    llvm::json::Array roots;
    for (StringRef root : access.rootValueIds)
      roots.push_back(root);
    llvm::json::Array paths;
    for (const PlanSlotPath &path : access.slotPaths)
      paths.push_back(slotPathJSON(path));
    accessArray.push_back(llvm::json::Object{
        {"operation", access.operationId},
        {"value", access.valueId},
        {"effect", access.effect},
        {"pending_async", access.pendingAsync},
        {"root_values", std::move(roots)},
        {"slot_paths", std::move(paths)},
    });
  }

  llvm::json::Array allocationArray;
  for (const PlanLdsAllocationRecord &allocation : ldsAllocations) {
    llvm::json::Array allocationAliases;
    for (StringRef alias : allocation.aliases)
      allocationAliases.push_back(alias);
    llvm::json::Array allocationSegments;
    for (const PlanLiveSegment &segment : allocation.liveSegments)
      allocationSegments.push_back(liveSegmentJSON(segment));
    llvm::json::Object object{
        {"root_value", allocation.rootValueId},
        {"allocation_operation", allocation.allocationOperationId},
        {"aliases", std::move(allocationAliases)},
        {"live_segments", std::move(allocationSegments)},
        {"physical_lds_offset", nullptr},
        {"physical_lds_size", nullptr},
    };
    object["logical_bytes"] = allocation.logicalBytes
                                  ? llvm::json::Value(*allocation.logicalBytes)
                                  : llvm::json::Value(nullptr);
    object["alignment"] = allocation.alignment
                              ? llvm::json::Value(*allocation.alignment)
                              : llvm::json::Value(nullptr);
    allocationArray.push_back(std::move(object));
  }

  return llvm::json::Object{
      {"function", functionName},
      {"semantic_fingerprint", semanticFingerprint},
      {"operations", std::move(operationArray)},
      {"values", std::move(valueArray)},
      {"lineage_edges", std::move(lineageArray)},
      {"blocks", std::move(blockArray)},
      {"live_segments", std::move(liveSegmentArray)},
      {"lds_aliases", std::move(aliasArray)},
      {"memory_accesses", std::move(accessArray)},
      {"lds_allocations", std::move(allocationArray)},
      {"diagnostics", std::move(diagnosticArray)},
  };
}

std::string serializePlanValueGraphs(ArrayRef<PlanValueGraph> graphs,
                                     ModuleOp module) {
  std::vector<const PlanValueGraph *> sorted;
  sorted.reserve(graphs.size());
  for (const auto &graph : graphs)
    sorted.push_back(&graph);
  llvm::sort(sorted, [](const auto *lhs, const auto *rhs) {
    return lhs->getFunctionName() < rhs->getFunctionName();
  });

  llvm::json::Array functions;
  std::vector<std::string> fingerprints;
  for (const PlanValueGraph *graph : sorted) {
    functions.push_back(graph->toJSON());
    fingerprints.push_back(graph->getSemanticFingerprint().str());
  }
  llvm::sort(fingerprints);
  std::string moduleFingerprint = hashText(llvm::join(fingerprints, "|"));

  std::string target;
  if (Attribute attribute = module->getAttr("ttg.target"))
    target = printAttribute(attribute);
  llvm::json::Object provenance{
      {"target", target},
      {"artifact_stage", "final_structured_ttgir"},
      {"pass_position", "after_warp_pipeline_conversion_before_scf_to_cf"},
      {"logical_tensor_bytes_are_physical_vgpr_bytes", false},
      {"static_intervals_are_physical_cycles", false},
      {"lds_logical_bytes_are_physical_allocation", false},
      {"async_lifetime_extended_through_wait", false},
  };
  llvm::json::Object root{
      {"schema_version", "plan-value-graph/0.2"},
      {"module_semantic_fingerprint", moduleFingerprint},
      {"provenance", std::move(provenance)},
      {"functions", std::move(functions)},
  };
  std::string output;
  llvm::raw_string_ostream os(output);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
  os << "\n";
  return output;
}

} // namespace mlir::triton::plan
