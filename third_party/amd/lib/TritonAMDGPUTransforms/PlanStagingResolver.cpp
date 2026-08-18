#include "PlanStagingResolver.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>

namespace mlir::triton::amdgpu {
namespace {

enum class TraceResult { NoPath, Path, Invalid };

static bool isSupportedDerivedOperation(Operation *operation) {
  if (!operation || operation->getNumRegions() != 0 ||
      operation->hasTrait<OpTrait::IsTerminator>() ||
      !isMemoryEffectFree(operation) || operation->getNumResults() == 0)
    return false;
  StringRef name = operation->getName().getStringRef();
  return llvm::is_contained(
      {StringRef("tt.reshape"), StringRef("tt.trans"),
       StringRef("ttg.convert_layout"), StringRef("amdg.extract_slice"),
       StringRef("amdg.in_thread_transpose"), StringRef("tlx.require_layout"),
       StringRef("tlx.release_layout")},
      name);
}

static bool dependsOnSource(Value value, Value source,
                            llvm::DenseSet<Value> &visited) {
  if (value == source)
    return true;
  if (!visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return dependsOnSource(operand, source, visited);
  });
}

static TraceResult
traceSupportedPath(Value value, Value source, scf::ForOp loop,
                   llvm::DenseMap<Value, TraceResult> &memo,
                   llvm::DenseSet<Operation *> &derivedOperations,
                   llvm::DenseSet<OpOperand *> &selectedEdges,
                   std::string &error) {
  if (value == source)
    return TraceResult::Path;
  auto found = memo.find(value);
  if (found != memo.end())
    return found->second;

  Operation *definition = value.getDefiningOp();
  if (!definition || !isSupportedDerivedOperation(definition)) {
    memo[value] = TraceResult::NoPath;
    return TraceResult::NoPath;
  }
  if (definition->getBlock() != loop.getBody()) {
    error = "derived register-to-LDS path leaves the selected loop body";
    memo[value] = TraceResult::Invalid;
    return TraceResult::Invalid;
  }

  OpOperand *pathOperand = nullptr;
  for (OpOperand &operand : definition->getOpOperands()) {
    TraceResult traced =
        traceSupportedPath(operand.get(), source, loop, memo, derivedOperations,
                           selectedEdges, error);
    if (traced == TraceResult::Invalid) {
      memo[value] = traced;
      return traced;
    }
    if (traced != TraceResult::Path)
      continue;
    if (pathOperand) {
      error = "derived register-to-LDS operation has ambiguous staged inputs";
      memo[value] = TraceResult::Invalid;
      return TraceResult::Invalid;
    }
    pathOperand = &operand;
  }
  if (!pathOperand) {
    memo[value] = TraceResult::NoPath;
    return TraceResult::NoPath;
  }
  derivedOperations.insert(definition);
  selectedEdges.insert(pathOperand);
  memo[value] = TraceResult::Path;
  return TraceResult::Path;
}

static const plan::PlanLiveSegment *
findUniqueLiveSegment(const plan::PlanValueGraph &graph, StringRef valueId) {
  const plan::PlanLiveSegment *found = nullptr;
  for (const plan::PlanLiveSegment &segment : graph.getLiveSegments()) {
    if (segment.valueId != valueId)
      continue;
    if (found)
      return nullptr;
    found = &segment;
  }
  return found;
}

} // namespace

LogicalResult
resolveLdsStaging(ArrayRef<plan::PlanPipelineStagingIntent> intents,
                  scf::ForOp loop, const plan::PlanValueGraph &graph,
                  const std::map<std::string, Operation *> &operationById,
                  const std::map<std::string, Value> &valueById,
                  PlanLdsStagingResolution &result, std::string &error) {
  std::map<std::string, const plan::PlanValueRecord *> valueRecordById;
  for (const plan::PlanValueRecord &value : graph.getValues())
    valueRecordById[value.id] = &value;

  for (const plan::PlanPipelineStagingIntent &intent : intents) {
    auto valueIt = valueById.find(intent.valueId);
    auto recordIt = valueRecordById.find(intent.valueId);
    if (valueIt == valueById.end() || recordIt == valueRecordById.end()) {
      error = "pipeline staging intent refers to an unknown value";
      return failure();
    }
    Value source = valueIt->second;
    auto tensorType = dyn_cast<RankedTensorType>(source.getType());
    Operation *producer = source.getDefiningOp();
    bool isGlobalToLds = intent.action == "global_to_lds";
    if (!tensorType || !producer) {
      error = isGlobalToLds
                  ? "global_to_lds staging requires a produced ranked tensor"
                  : "register-to-LDS staging requires a produced ranked tensor";
      return failure();
    }
    if (producer->getBlock() != loop.getBody()) {
      error = "register-to-LDS producer must be directly in the selected loop";
      return failure();
    }
    if (isGlobalToLds) {
      if (auto load = dyn_cast<triton::LoadOp>(producer)) {
        if (load.getIsVolatile()) {
          error = "global_to_lds does not support volatile loads";
          return failure();
        }
      } else if (!isa<BufferLoadOp>(producer)) {
        error = "global_to_lds requires tt.load or amdg.buffer_load";
        return failure();
      }
    }
    if (!recordIt->second->logicalBytes ||
        *recordIt->second->logicalBytes <= 0) {
      error = "register-to-LDS staging requires a known positive tensor size";
      return failure();
    }
    if (*recordIt->second->logicalBytes >
        std::numeric_limits<int64_t>::max() / intent.bufferDepth) {
      error = "register-to-LDS staging byte size overflows";
      return failure();
    }
    int64_t allocationBytes =
        *recordIt->second->logicalBytes * intent.bufferDepth;
    if (result.logicalBytes >
        std::numeric_limits<int64_t>::max() - allocationBytes) {
      error = "register-to-LDS staging byte size overflows";
      return failure();
    }
    result.logicalBytes += allocationBytes;

    const plan::PlanLiveSegment *live =
        findUniqueLiveSegment(graph, intent.valueId);
    if (!live) {
      error = "register-to-LDS staging requires one source live segment";
      return failure();
    }

    PlanLdsStaging staging;
    staging.loop = loop;
    staging.action = intent.action;
    staging.source = source;
    staging.sourceValueId = intent.valueId;
    staging.sourceProducerId = recordIt->second->definingOperationId;
    staging.tensorType = tensorType;
    staging.logicalBytes = *recordIt->second->logicalBytes;
    staging.alignment = intent.alignment;
    staging.distance = intent.distance;
    staging.bufferDepth = intent.bufferDepth;
    staging.sourceLiveStartBefore = live->startPosition;
    staging.sourceLiveEndBefore = live->endPosition;

    llvm::DenseSet<Operation *> requestedConsumers;
    llvm::DenseSet<Operation *> derivedOperations;
    llvm::DenseSet<OpOperand *> selectedEdges;
    llvm::DenseMap<Value, TraceResult> memo;
    for (StringRef consumerId : intent.consumerIds) {
      auto consumerIt = operationById.find(consumerId.str());
      if (consumerIt == operationById.end()) {
        error = "register-to-LDS staging refers to an unknown consumer";
        return failure();
      }
      Operation *consumer = consumerIt->second;
      if (consumer->getBlock() != loop.getBody() ||
          consumer->hasTrait<OpTrait::IsTerminator>()) {
        error =
            "register-to-LDS consumer must be directly in the selected loop";
        return failure();
      }
      if (!producer->isBeforeInBlock(consumer)) {
        error = "register-to-LDS consumer must follow its producer";
        return failure();
      }
      if (!requestedConsumers.insert(consumer).second) {
        error = "register-to-LDS staging repeats a resolved consumer";
        return failure();
      }

      bool foundPath = false;
      for (OpOperand &operand : consumer->getOpOperands()) {
        TraceResult traced =
            traceSupportedPath(operand.get(), source, loop, memo,
                               derivedOperations, selectedEdges, error);
        if (traced == TraceResult::Invalid)
          return failure();
        if (traced == TraceResult::Path) {
          staging.consumerOperands.push_back(&operand);
          selectedEdges.insert(&operand);
          foundPath = true;
          continue;
        }
        llvm::DenseSet<Value> visited;
        if (dependsOnSource(operand.get(), source, visited)) {
          error = "named register-to-LDS consumer is reached through an "
                  "unsupported derived operation";
          return failure();
        }
      }
      if (!foundPath) {
        error = "named register-to-LDS consumer does not use the staged value";
        return failure();
      }
      staging.consumers.push_back(consumer);
    }

    staging.derivedOperations.assign(derivedOperations.begin(),
                                     derivedOperations.end());
    llvm::sort(staging.derivedOperations,
               [](Operation *left, Operation *right) {
                 return left->isBeforeInBlock(right);
               });
    llvm::sort(staging.consumers, [](Operation *left, Operation *right) {
      return left->isBeforeInBlock(right);
    });

    llvm::SmallPtrSet<Operation *, 8> unselectedConsumers;
    SmallVector<Value> pathValues{source};
    for (Operation *operation : staging.derivedOperations)
      llvm::append_range(pathValues, operation->getResults());
    for (Value value : pathValues) {
      for (OpOperand &use : value.getUses()) {
        if (selectedEdges.contains(&use))
          continue;
        staging.preservedOperands.push_back({&use, value});
        unselectedConsumers.insert(use.getOwner());
      }
    }
    staging.unselectedConsumersPreserved = unselectedConsumers.size();
    if (isGlobalToLds && !staging.preservedOperands.empty()) {
      error = "global_to_lds requires the complete derived-use closure";
      return failure();
    }

    result.participatingOperations.insert(producer);
    result.participatingOperations.insert(staging.consumers.begin(),
                                          staging.consumers.end());
    result.participatingOperations.insert(staging.derivedOperations.begin(),
                                          staging.derivedOperations.end());
    result.staging.push_back(std::move(staging));
  }
  return success();
}

} // namespace mlir::triton::amdgpu
