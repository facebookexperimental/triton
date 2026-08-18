#include "PlanRingMaterializer.h"
#include "PlanStagingMaterializer.h"
#include "PlanStagingResolver.h"
#include "TritonAMDGPUTransforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/AMDLatencyModel.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/DataDependenceGraph.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/ModuloReservationTable.h"
#include "triton/Analysis/PlanPipeline.h"
#include "triton/Analysis/PlanValueGraph.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUAPPLYPLANPIPELINE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {
namespace tt = triton;
namespace ttg = triton::gpu;
namespace tta = triton::amdgpu;

class TLXAMDLatencyModel final : public ttg::AMDLatencyModel {
public:
  ttg::OpLatencyInfo getLatency(Operation *operation) const override {
    if (auto mfma = dyn_cast<tta::ScheduledMfmaOp>(operation)) {
      int64_t count = 1;
      auto result = dyn_cast<RankedTensorType>(mfma.getResult().getType());
      auto lhs = dyn_cast<RankedTensorType>(mfma.getA().getType());
      auto layout =
          result
              ? dyn_cast_or_null<ttg::AMDMfmaEncodingAttr>(result.getEncoding())
              : nullptr;
      if (result && lhs && layout && result.getRank() == 2 &&
          lhs.getRank() >= 2) {
        auto instruction = layout.getInstrShape();
        auto warps = layout.getWarpsPerCTA();
        if (instruction.size() >= 3 && warps.size() >= 2) {
          int64_t tileM = std::max<int64_t>(1, instruction[0] * warps[0]);
          int64_t tileN = std::max<int64_t>(1, instruction[1] * warps[1]);
          int64_t instructionK = std::max<int64_t>(1, instruction[2]);
          int64_t m = result.getShape()[0];
          int64_t n = result.getShape()[1];
          int64_t k = lhs.getShape()[1];
          count = std::max<int64_t>(
              1, ((m + tileM - 1) / tileM) * ((n + tileN - 1) / tileN) *
                     ((k + instructionK - 1) / instructionK));
        }
      }
      int occupancy = static_cast<int>(std::min<int64_t>(count * 4, 1 << 20));
      return {ttg::HWPipeline::MFMA, 18, 17, 1, occupancy};
    }
    if (isa<tta::BufferLoadToLocalOp>(operation))
      return {ttg::HWPipeline::GLOBAL, 790, 8, 1, 8};
    return ttg::AMDLatencyModel::getLatency(operation);
  }
};

struct ResolvedLoopPipeline {
  scf::ForOp loop;
  SmallVector<Operation *> desiredOrder;
  SmallVector<tta::PlanExistingLdsRingMutation, 1> ringMutations;
  SmallVector<tta::PlanAsyncWaitMutation, 2> waitMutations;
  SmallVector<tta::PlanRegisterToLdsStaging, 1> stagingMutations;
  tt::plan::PlanPipelineLoopApplyRecord record;
};

static LogicalResult writeReport(ModuleOp module, StringRef outputPath,
                                 StringRef payload) {
  if (outputPath.empty())
    return success();
  if (outputPath == "-") {
    llvm::outs() << payload;
    return success();
  }
  llvm::SmallString<256> path(outputPath);
  llvm::SmallString<256> parent = llvm::sys::path::parent_path(path);
  if (!parent.empty())
    if (std::error_code ec = llvm::sys::fs::create_directories(parent)) {
      module.emitError("cannot create plan-pipeline report directory '")
          << parent << "': " << ec.message();
      return failure();
    }
  std::error_code ec;
  llvm::ToolOutputFile output(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    module.emitError("cannot open plan-pipeline report '")
        << path << "': " << ec.message();
    return failure();
  }
  output.os() << payload;
  output.keep();
  return success();
}

static LogicalResult fail(tt::plan::PlanPipelineApplyResult &result,
                          std::string &error, Twine message) {
  error = message.str();
  result.error = error;
  return failure();
}

static Operation *projectToLoopBody(Operation *operation, scf::ForOp loop) {
  if (!operation)
    return nullptr;
  Block *body = loop.getBody();
  while (operation && operation->getBlock() != body) {
    operation = operation->getParentOp();
    if (operation == loop)
      return nullptr;
  }
  return operation;
}

static bool hasExactSlotDepth(const tt::plan::PlanAsyncTransaction &transaction,
                              int64_t depth) {
  bool found = false;
  for (const tt::plan::PlanSlotPath &path : transaction.slotPaths)
    for (const tt::plan::PlanSlotExpression &index : path.indices) {
      if (index.kind == "unknown")
        return false;
      if (index.modulus > 0) {
        found = true;
        if (index.modulus != depth)
          return false;
      } else if (depth == 1 && index.kind == "constant" && index.offset == 0) {
        found = true;
      }
    }
  return found;
}

static FailureOr<int64_t>
getExactConsumerDistance(const tt::plan::PlanAsyncTransaction &transaction) {
  std::optional<int64_t> distance;
  for (const tt::plan::PlanAsyncFrontier &frontier :
       transaction.consumerFrontiers) {
    if (frontier.iterationDistance <= 0)
      continue;
    if (distance && *distance != frontier.iterationDistance)
      return failure();
    distance = frontier.iterationDistance;
  }
  if (!distance)
    return failure();
  return *distance;
}

static bool
hasExactConsumerDistance(const tt::plan::PlanAsyncTransaction &transaction,
                         int64_t distance) {
  FailureOr<int64_t> current = getExactConsumerDistance(transaction);
  return succeeded(current) && *current == distance;
}

static ttg::LocalAllocOp getRootAllocation(Value value) {
  llvm::DenseSet<Value> visited;
  while (value && visited.insert(value).second) {
    if (auto allocation = value.getDefiningOp<ttg::LocalAllocOp>())
      return allocation;
    Operation *definition = value.getDefiningOp();
    if (!definition || !definition->hasTrait<OpTrait::MemDescViewTrait>() ||
        definition->getNumOperands() == 0)
      return {};
    value = definition->getOperand(0);
  }
  return {};
}

static ttg::MemDescIndexOp getRootedIndexView(Operation *operation,
                                              ttg::LocalAllocOp root) {
  ttg::MemDescIndexOp found;
  for (Value operand : operation->getOperands()) {
    if (!isa<ttg::MemDescType>(operand.getType()) ||
        getRootAllocation(operand) != root)
      continue;
    Value cursor = operand;
    while (Operation *definition = cursor.getDefiningOp()) {
      if (auto index = dyn_cast<ttg::MemDescIndexOp>(definition)) {
        if (found && found != index)
          return {};
        found = index;
        break;
      }
      if (!definition->hasTrait<OpTrait::MemDescViewTrait>() ||
          definition->getNumOperands() == 0)
        break;
      cursor = definition->getOperand(0);
    }
  }
  return found;
}

static bool containsId(ArrayRef<std::string> ids, StringRef id) {
  return llvm::is_contained(ids, id.str());
}

static const tt::plan::PlanSlotExpression *
getRingSlotExpression(ArrayRef<tt::plan::PlanSlotPath> paths, StringRef root,
                      int64_t depth) {
  const tt::plan::PlanSlotExpression *found = nullptr;
  for (const tt::plan::PlanSlotPath &path : paths) {
    if (path.rootValueId != root)
      continue;
    for (const tt::plan::PlanSlotExpression &index : path.indices) {
      if (index.kind != "modulo" || index.modulus != depth ||
          index.coefficient != 1)
        continue;
      if (found)
        return nullptr;
      found = &index;
    }
  }
  return found;
}

static std::optional<int64_t>
getSlotDistance(const tt::plan::PlanSlotExpression &producer,
                const tt::plan::PlanSlotExpression &consumer, int64_t depth) {
  if (depth <= 1 || producer.baseValueId != consumer.baseValueId ||
      producer.coefficient != consumer.coefficient ||
      producer.modulus != depth || consumer.modulus != depth)
    return std::nullopt;
  int64_t distance = (producer.offset - consumer.offset) % depth;
  if (distance < 0)
    distance += depth;
  if (distance == 0)
    return std::nullopt;
  return distance;
}

static FailureOr<int64_t> resolveExistingRingConsumers(
    const tt::plan::PlanAsyncTransaction &transaction,
    const tt::plan::PlanValueGraph &graph,
    const std::map<std::string, Operation *> &operationById, scf::ForOp loop,
    ttg::LocalAllocOp allocation, int64_t depth,
    SmallVectorImpl<Operation *> &consumers,
    SmallVectorImpl<ttg::MemDescIndexOp> &consumerViews) {
  StringRef root = transaction.rootValueIds.front();
  const tt::plan::PlanSlotExpression *producerSlot =
      getRingSlotExpression(transaction.slotPaths, root, depth);
  std::optional<int64_t> slotDistance;
  if (producerSlot) {
    for (const tt::plan::PlanMemoryAccess &access : graph.getMemoryAccesses()) {
      if (access.effect != "read" || !containsId(access.rootValueIds, root))
        continue;
      auto operationIt = operationById.find(access.operationId);
      Operation *consumer = operationIt == operationById.end()
                                ? nullptr
                                : projectToLoopBody(operationIt->second, loop);
      if (!consumer)
        continue;
      ttg::MemDescIndexOp view = getRootedIndexView(consumer, allocation);
      const tt::plan::PlanSlotExpression *consumerSlot =
          getRingSlotExpression(access.slotPaths, root, depth);
      if (!view || !consumerSlot)
        continue;
      std::optional<int64_t> distance =
          getSlotDistance(*producerSlot, *consumerSlot, depth);
      if (!distance)
        continue;
      if (slotDistance && *slotDistance != *distance)
        return failure();
      slotDistance = distance;
      if (!llvm::is_contained(consumers, consumer))
        consumers.push_back(consumer);
      if (!llvm::is_contained(consumerViews, view))
        consumerViews.push_back(view);
    }
  }
  if (slotDistance)
    return *slotDistance;

  FailureOr<int64_t> frontierDistance = getExactConsumerDistance(transaction);
  if (failed(frontierDistance))
    return failure();
  for (const tt::plan::PlanAsyncFrontier &frontier :
       transaction.consumerFrontiers) {
    if (frontier.iterationDistance != *frontierDistance)
      continue;
    auto consumerIt = operationById.find(frontier.operationId);
    Operation *consumer = consumerIt == operationById.end()
                              ? nullptr
                              : projectToLoopBody(consumerIt->second, loop);
    if (!consumer)
      continue;
    ttg::MemDescIndexOp view = getRootedIndexView(consumer, allocation);
    if (!view)
      return failure();
    if (!llvm::is_contained(consumers, consumer))
      consumers.push_back(consumer);
    if (!llvm::is_contained(consumerViews, view))
      consumerViews.push_back(view);
  }
  return consumers.empty() ? FailureOr<int64_t>(failure())
                           : FailureOr<int64_t>(*frontierDistance);
}

static int64_t getLdsCapacityBytes(ModuleOp module) {
  // M1 targets gfx950. Keep a conservative legacy fallback for tests and
  // modules whose target attribute is intentionally omitted.
  std::optional<StringRef> arch = getAMDArch(module);
  if (arch && *arch == "gfx950")
    return 160 * 1024;
  return 64 * 1024;
}

static std::string frozenDotSignature(tt::FuncOp function) {
  std::string signature;
  llvm::raw_string_ostream stream(signature);
  function.walk([&](Operation *operation) {
    if (!isa<tt::DotOp, tta::ScheduledMfmaOp>(operation))
      return;
    stream << operation->getName() << "|";
    for (Type type : operation->getOperandTypes())
      stream << type << ";";
    stream << "->";
    for (Type type : operation->getResultTypes())
      stream << type << ";";
    stream << "|" << operation->getAttrDictionary() << "\n";
  });
  return stream.str();
}

static bool isMovableSliceOperation(Operation *operation) {
  if (!operation || operation->getNumRegions() != 0 ||
      operation->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncCopyGlobalToLocalOp,
          tta::BufferLoadToLocalOp>(operation))
    return true;
  return isMemoryEffectFree(operation);
}

static void collectMovableBackwardSlice(Operation *root, scf::ForOp loop,
                                        llvm::DenseSet<Operation *> &selected) {
  SmallVector<Operation *> worklist{root};
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    operation = projectToLoopBody(operation, loop);
    if (!operation || !isMovableSliceOperation(operation) ||
        !selected.insert(operation).second)
      continue;
    for (Value operand : operation->getOperands())
      if (Operation *definition = operand.getDefiningOp())
        if (definition->getBlock() == loop.getBody() &&
            isMovableSliceOperation(definition))
          worklist.push_back(definition);
  }
}

static LogicalResult
resolveLoopPipeline(const tt::plan::PlanLoopPipelineDelta &requested,
                    scf::ForOp loop, const tt::plan::PlanValueGraph &graph,
                    const std::map<std::string, Operation *> &operationById,
                    const std::map<std::string, Value> &valueById,
                    ResolvedLoopPipeline &resolved, std::string &error) {
  resolved.loop = loop;
  resolved.record.loopId = requested.loopId;

  std::map<std::string, const tt::plan::PlanAsyncGroup *> groupById;
  for (const tt::plan::PlanAsyncGroup &group : graph.getAsyncGroups())
    groupById[group.id] = &group;
  std::map<std::string, const tt::plan::PlanAsyncTransaction *> transactionById;
  for (const tt::plan::PlanAsyncTransaction &transaction :
       graph.getAsyncTransactions())
    transactionById[transaction.id] = &transaction;

  std::set<std::string> requestedGroups;
  for (const auto &intent : requested.transactions)
    requestedGroups.insert(intent.groupId);

  if (!requested.transactions.empty() && !requested.staging.empty()) {
    error = "M1.5b.4 does not combine new staging with existing-ring "
            "transactions";
    return failure();
  }

  tta::PlanRegisterToLdsResolution stagingResolution;
  if (failed(tta::resolveRegisterToLdsStaging(requested.staging, loop, graph,
                                              operationById, valueById,
                                              stagingResolution, error)))
    return failure();
  int64_t stagingBytes = stagingResolution.logicalBytes;
  resolved.stagingMutations = std::move(stagingResolution.staging);

  std::map<std::string, const tt::plan::PlanPipelineTransactionIntent *>
      intentByGroup;
  for (const auto &intent : requested.transactions)
    intentByGroup[intent.groupId] = &intent;
  std::map<std::string, const tt::plan::PlanLdsAllocationRecord *>
      allocationByRoot;
  for (const tt::plan::PlanLdsAllocationRecord &allocation :
       graph.getLdsAllocations())
    allocationByRoot[allocation.rootValueId] = &allocation;

  llvm::DenseSet<Operation *> selected;
  selected.insert(stagingResolution.participatingOperations.begin(),
                  stagingResolution.participatingOperations.end());
  llvm::DenseSet<Operation *> mutationOperations;
  SmallVector<std::tuple<Operation *, Operation *, unsigned>> releaseToProducer;
  SmallVector<std::tuple<Operation *, Operation *, unsigned>> requestedEdges;
  std::map<Operation *, unsigned> ringByAllocation;
  std::map<std::string, SmallVector<Operation *>> consumersByGroup;
  bool hasScheduleMutation = false;
  std::set<std::string> scheduleMutationGroups;
  for (const auto &intent : requested.transactions) {
    auto groupIt = groupById.find(intent.groupId);
    if (groupIt == groupById.end()) {
      error = "pipeline delta refers to unknown async group '" +
              intent.groupId + "'";
      return failure();
    }
    const tt::plan::PlanAsyncGroup &group = *groupIt->second;
    auto commitIt = operationById.find(group.commitOperationId);
    if (commitIt == operationById.end() ||
        projectToLoopBody(commitIt->second, loop) != commitIt->second) {
      error = "async group '" + intent.groupId +
              "' is not committed directly in the selected loop";
      return failure();
    }
    if (group.transactionIds.empty()) {
      error = "async group '" + intent.groupId + "' contains no transactions";
      return failure();
    }
    collectMovableBackwardSlice(commitIt->second, loop, selected);
    resolved.record.groups.push_back(intent.groupId);
    for (StringRef transactionId : group.transactionIds) {
      auto transactionIt = transactionById.find(transactionId.str());
      if (transactionIt == transactionById.end()) {
        error = "async group '" + intent.groupId +
                "' contains an unknown transaction";
        return failure();
      }
      const tt::plan::PlanAsyncTransaction &transaction =
          *transactionIt->second;
      if (transaction.commitGroupId != intent.groupId) {
        error = "async group '" + intent.groupId +
                "' contains an inconsistently bound transaction";
        return failure();
      }
      if (transaction.rootValueIds.size() != 1) {
        error = "M1.5b.3 requires one resolved LDS root per transaction";
        return failure();
      }
      auto allocationRecord =
          allocationByRoot.find(transaction.rootValueIds.front());
      if (allocationRecord == allocationByRoot.end()) {
        error = "async transaction LDS root has no allocation record";
        return failure();
      }
      auto allocationOperation =
          operationById.find(allocationRecord->second->allocationOperationId);
      if (allocationOperation == operationById.end()) {
        error = "async transaction LDS allocation cannot be resolved";
        return failure();
      }
      auto allocation =
          dyn_cast<ttg::LocalAllocOp>(allocationOperation->second);
      auto allocationType =
          allocation ? dyn_cast<ttg::MemDescType>(allocation.getType())
                     : nullptr;
      if (!allocation || !allocationType || allocationType.getRank() < 2 ||
          allocationType.getShape().front() < 1) {
        error = "async transaction root is not a canonical existing LDS ring";
        return failure();
      }
      int64_t oldDepth = allocationType.getShape().front();
      if (!hasExactSlotDepth(transaction, oldDepth)) {
        error = "async transaction slot paths disagree with its LDS allocation "
                "depth";
        return failure();
      }
      auto producerIt = operationById.find(transaction.producerOperationId);
      if (producerIt == operationById.end() ||
          projectToLoopBody(producerIt->second, loop) != producerIt->second) {
        error = "async transaction producer is not directly in the selected "
                "loop";
        return failure();
      }
      collectMovableBackwardSlice(producerIt->second, loop, selected);

      SmallVector<Operation *> consumers;
      SmallVector<ttg::MemDescIndexOp> consumerViews;
      ttg::MemDescIndexOp producerView =
          getRootedIndexView(producerIt->second, allocation);
      if (!producerView) {
        error = "existing LDS producer does not use one direct indexed view";
        return failure();
      }
      FailureOr<int64_t> oldDistance = resolveExistingRingConsumers(
          transaction, graph, operationById, loop, allocation, oldDepth,
          consumers, consumerViews);
      if (failed(oldDistance)) {
        error = "async transaction does not have one exact positive consumer "
                "distance";
        return failure();
      }
      for (Operation *consumer : consumers)
        if (!llvm::is_contained(consumersByGroup[intent.groupId], consumer))
          consumersByGroup[intent.groupId].push_back(consumer);

      bool mutates =
          oldDepth != intent.bufferDepth || *oldDistance != intent.distance;
      if (mutates) {
        hasScheduleMutation = true;
        scheduleMutationGroups.insert(intent.groupId);
        unsigned ringIndex;
        auto existing = ringByAllocation.find(allocation);
        if (existing == ringByAllocation.end()) {
          ringIndex = resolved.ringMutations.size();
          ringByAllocation[allocation] = ringIndex;
          tta::PlanExistingLdsRingMutation ring;
          ring.loop = loop;
          ring.allocation = allocation;
          ring.oldDepth = oldDepth;
          ring.newDepth = intent.bufferDepth;
          ring.oldDistance = *oldDistance;
          ring.newDistance = intent.distance;
          resolved.ringMutations.push_back(std::move(ring));
        } else {
          ringIndex = existing->second;
          const auto &ring = resolved.ringMutations[ringIndex];
          if (ring.oldDepth != oldDepth ||
              ring.newDepth != intent.bufferDepth ||
              ring.oldDistance != *oldDistance ||
              ring.newDistance != intent.distance) {
            error = "one LDS allocation received inconsistent depth or "
                    "distance requests";
            return failure();
          }
        }
        auto &ring = resolved.ringMutations[ringIndex];
        if (!llvm::is_contained(ring.producerViews, producerView))
          ring.producerViews.push_back(producerView);
        if (!llvm::is_contained(ring.producers, producerIt->second))
          ring.producers.push_back(producerIt->second);
        for (Operation *consumer : consumers)
          if (!llvm::is_contained(ring.consumers, consumer))
            ring.consumers.push_back(consumer);
        for (ttg::MemDescIndexOp view : consumerViews)
          if (!llvm::is_contained(ring.consumerViews, view))
            ring.consumerViews.push_back(view);
        mutationOperations.insert(producerIt->second);
        mutationOperations.insert(commitIt->second);
        for (Operation *consumer : consumers)
          mutationOperations.insert(consumer);
      }

      Operation *release = nullptr;
      for (const tt::plan::PlanAsyncFrontier &frontier :
           transaction.releaseFrontiers) {
        auto releaseIt = operationById.find(frontier.operationId);
        Operation *candidate = releaseIt == operationById.end()
                                   ? nullptr
                                   : projectToLoopBody(releaseIt->second, loop);
        if (candidate && frontier.iterationDistance == *oldDistance)
          release = candidate;
      }
      if (!release)
        release = consumers.back();
      unsigned reuseDistance = static_cast<unsigned>(
          std::max<int64_t>(0, intent.bufferDepth - intent.distance));
      bool requiresScheduleMutation =
          *oldDistance != intent.distance ||
          reuseDistance < static_cast<unsigned>(*oldDistance);
      releaseToProducer.push_back({release, producerIt->second,
                                   requiresScheduleMutation
                                       ? reuseDistance
                                       : static_cast<unsigned>(*oldDistance)});
      if (requiresScheduleMutation) {
        requestedEdges.push_back({consumers.back(), release, 0});
        mutationOperations.insert(release);
      }
    }
  }

  for (const tt::plan::PlanAsyncWaitRecord &wait : graph.getAsyncWaits()) {
    std::set<std::string> family;
    bool touchesRequested = false;
    int64_t oldDistance = -1;
    for (const tt::plan::PlanAsyncWaitCompletion &completion :
         wait.completedGroups) {
      if (requestedGroups.count(completion.groupId) &&
          completion.iterationDistance > 0) {
        touchesRequested = true;
        if (oldDistance >= 0 && oldDistance != completion.iterationDistance) {
          error = "one wait completes requested groups at different distances";
          return failure();
        }
        oldDistance = completion.iterationDistance;
      }
    }
    if (!touchesRequested)
      continue;
    for (const tt::plan::PlanAsyncWaitCompletion &completion :
         wait.completedGroups) {
      if (completion.iterationDistance != oldDistance)
        continue;
      auto groupIt = groupById.find(completion.groupId);
      if (groupIt == groupById.end())
        continue;
      auto commitIt = operationById.find(groupIt->second->commitOperationId);
      if (commitIt != operationById.end() &&
          projectToLoopBody(commitIt->second, loop) == commitIt->second)
        family.insert(completion.groupId);
    }
    if (!std::includes(requestedGroups.begin(), requestedGroups.end(),
                       family.begin(), family.end())) {
      error = "pipeline delta omits a positive-distance group sharing wait '" +
              wait.operationId + "'";
      return failure();
    }

    int64_t newDistance = -1;
    bool allDepthOne = true;
    bool waitMutates = false;
    SmallVector<Operation *> waitConsumers;
    for (StringRef groupId : family) {
      auto intentIt = intentByGroup.find(groupId.str());
      if (intentIt == intentByGroup.end()) {
        error = "positive-distance wait family has no resolved intent";
        return failure();
      }
      const auto *intent = intentIt->second;
      if (newDistance >= 0 && newDistance != intent->distance) {
        error = "one wait family received inconsistent requested distances";
        return failure();
      }
      newDistance = intent->distance;
      allDepthOne &= intent->bufferDepth == 1;
      waitMutates |= intent->distance != oldDistance;
      for (Operation *consumer : consumersByGroup[groupId.str()])
        if (!llvm::is_contained(waitConsumers, consumer))
          waitConsumers.push_back(consumer);
    }
    waitMutates |= llvm::any_of(family, [&](StringRef groupId) {
      auto intentIt = intentByGroup.find(groupId.str());
      const auto *intent =
          intentIt == intentByGroup.end() ? nullptr : intentIt->second;
      auto group = groupById.find(groupId.str());
      if (!intent || group == groupById.end())
        return false;
      for (StringRef transactionId : group->second->transactionIds) {
        auto transaction = transactionById.find(transactionId.str());
        if (transaction == transactionById.end() ||
            transaction->second->rootValueIds.empty())
          continue;
        auto allocationRecord =
            allocationByRoot.find(transaction->second->rootValueIds.front());
        if (allocationRecord == allocationByRoot.end())
          continue;
        auto allocationOperation =
            operationById.find(allocationRecord->second->allocationOperationId);
        auto allocation =
            allocationOperation == operationById.end()
                ? ttg::LocalAllocOp()
                : dyn_cast<ttg::LocalAllocOp>(allocationOperation->second);
        if (allocation &&
            cast<ttg::MemDescType>(allocation.getType()).getShape().front() !=
                intent->bufferDepth)
          return true;
      }
      return false;
    });
    if (!waitMutates)
      continue;

    auto waitIt = operationById.find(wait.operationId);
    auto waitOp = waitIt == operationById.end()
                      ? ttg::AsyncWaitOp()
                      : dyn_cast<ttg::AsyncWaitOp>(waitIt->second);
    if (!waitOp || projectToLoopBody(waitOp, loop) != waitOp) {
      error = "positive-distance completion wait is not directly in the loop";
      return failure();
    }
    int64_t selectedRetained = oldDistance * family.size();
    int64_t unrelatedRetained = wait.retainedGroupCount - selectedRetained;
    if (unrelatedRetained < 0) {
      error = "wait retained count is inconsistent with its completed family";
      return failure();
    }
    int64_t newRetained =
        unrelatedRetained + (allDepthOne ? 0 : newDistance * family.size());
    resolved.waitMutations.push_back(
        {waitOp, wait.retainedGroupCount, newRetained, waitConsumers});
    bool familyScheduleMutation = llvm::any_of(family, [&](StringRef groupId) {
      return scheduleMutationGroups.count(groupId.str());
    });
    if (familyScheduleMutation) {
      mutationOperations.insert(waitOp);
      for (Operation *consumer : waitConsumers) {
        requestedEdges.push_back({waitOp, consumer, 0});
        mutationOperations.insert(consumer);
      }
      for (StringRef groupId : family) {
        auto commit =
            operationById.find(groupById[groupId.str()]->commitOperationId);
        if (commit != operationById.end()) {
          if (allDepthOne)
            requestedEdges.push_back({waitOp, commit->second, 0});
          else
            requestedEdges.push_back({commit->second, waitOp, 0});
          mutationOperations.insert(commit->second);
        }
      }
    }
  }

  int64_t logicalLdsBefore =
      graph.getResourceSummary().logicalLdsAllocationBytes;
  int64_t logicalLdsAfter = logicalLdsBefore;
  for (const tta::PlanExistingLdsRingMutation &ring : resolved.ringMutations) {
    const tt::plan::PlanLdsAllocationRecord *allocationRecord = nullptr;
    for (const auto &[root, candidate] : allocationByRoot) {
      auto operation = operationById.find(candidate->allocationOperationId);
      if (operation != operationById.end() &&
          operation->second == ring.allocation) {
        allocationRecord = candidate;
        break;
      }
    }
    if (!allocationRecord || !allocationRecord->logicalBytes ||
        ring.oldDepth <= 0 || *allocationRecord->logicalBytes % ring.oldDepth) {
      error = "cannot prove the logical byte size of a resized LDS ring";
      return failure();
    }
    int64_t slotBytes = *allocationRecord->logicalBytes / ring.oldDepth;
    if (slotBytes > std::numeric_limits<int64_t>::max() / ring.newDepth) {
      error = "resized LDS ring byte size overflows";
      return failure();
    }
    int64_t newBytes = slotBytes * ring.newDepth;
    logicalLdsAfter += newBytes - *allocationRecord->logicalBytes;

    llvm::DenseSet<Operation *> covered;
    covered.insert(ring.producers.begin(), ring.producers.end());
    covered.insert(ring.consumers.begin(), ring.consumers.end());
    for (const tt::plan::PlanMemoryAccess &access : graph.getMemoryAccesses()) {
      if (access.effect != "read" && access.effect != "write")
        continue;
      if (!containsId(access.rootValueIds, allocationRecord->rootValueId))
        continue;
      auto operation = operationById.find(access.operationId);
      Operation *inLoop = operation == operationById.end()
                              ? nullptr
                              : projectToLoopBody(operation->second, loop);
      if (inLoop && !covered.contains(inLoop)) {
        error = "resized LDS ring has an unselected in-loop reader or writer";
        return failure();
      }
    }
  }
  if (stagingBytes > std::numeric_limits<int64_t>::max() - logicalLdsAfter) {
    error = "register-to-LDS staging total byte size overflows";
    return failure();
  }
  logicalLdsAfter += stagingBytes;
  if (logicalLdsAfter >
      getLdsCapacityBytes(loop->getParentOfType<ModuleOp>())) {
    error = resolved.stagingMutations.empty()
                ? "requested LDS ring depths exceed the target LDS capacity"
                : "register-to-LDS staging exceeds the target LDS capacity";
    return failure();
  }
  resolved.record.logicalLdsBytesBefore = logicalLdsBefore;
  resolved.record.logicalLdsBytesAfter = logicalLdsAfter;
  resolved.record.ringMutationCount = resolved.ringMutations.size();
  resolved.record.stagingMutationCount = resolved.stagingMutations.size();

  TLXAMDLatencyModel latencyModel;
  llvm::DenseMap<Operation *, ttg::DataPartitionInfo> noPartition;
  ttg::DataDependenceGraph ddg = ttg::DataDependenceGraph::build(
      loop, latencyModel, noPartition, ttg::getActiveScheduleAlgo(),
      /*scheduleNestedLoops=*/false);
  SmallVector<Operation *> baselineOrder;
  std::map<Operation *, int64_t> baselinePosition;
  for (auto [position, operation] :
       llvm::enumerate(loop.getBody()->without_terminator())) {
    baselineOrder.push_back(&operation);
    baselinePosition[&operation] = position;
  }
  int64_t imported = 0;
  int64_t skipped = 0;
  for (const tt::plan::PlanDependencyEdge &dependency :
       graph.getDependencyEdges()) {
    auto sourceIt = operationById.find(dependency.sourceOperationId);
    auto destinationIt = operationById.find(dependency.destinationOperationId);
    if (sourceIt == operationById.end() || destinationIt == operationById.end())
      continue;
    Operation *source = projectToLoopBody(sourceIt->second, loop);
    Operation *destination = projectToLoopBody(destinationIt->second, loop);
    if (!source || !destination || source == destination)
      continue;
    bool replacesPipelineEdge =
        hasScheduleMutation && mutationOperations.contains(source) &&
        mutationOperations.contains(destination) &&
        llvm::is_contained(
            {StringRef("memory_raw"), StringRef("memory_war"),
             StringRef("memory_waw"), StringRef("async_completion"),
             StringRef("barrier_visibility"), StringRef("async_consumer"),
             StringRef("consumer_release"), StringRef("slot_reuse")},
            StringRef(dependency.kind));
    if (replacesPipelineEdge)
      continue;
    if (dependency.iterationDistance == 0 &&
        baselinePosition[source] > baselinePosition[destination]) {
      ++skipped;
      continue;
    }
    auto nodeIt = ddg.getOpToIdx().find(source);
    if (nodeIt == ddg.getOpToIdx().end())
      continue;
    int latency = std::max(ddg.getNode(nodeIt->second).latency, 1);
    if (succeeded(ddg.addExternalDependence(
            source, destination, latency,
            static_cast<unsigned>(dependency.iterationDistance))))
      ++imported;
  }
  for (auto [release, producer, distance] : releaseToProducer)
    if (succeeded(ddg.addExternalDependence(release, producer, 1, distance)))
      ++imported;
  for (auto [source, destination, distance] : requestedEdges)
    if (source != destination &&
        succeeded(ddg.addExternalDependence(source, destination, 1, distance)))
      ++imported;

  Operation *previousPinned = nullptr;
  for (Operation *operation : baselineOrder) {
    if (selected.contains(operation))
      continue;
    if (previousPinned &&
        succeeded(ddg.addExternalDependence(previousPinned, operation, 1, 0)))
      ++imported;
    previousPinned = operation;
  }
  if (selected.empty()) {
    error = "pipeline delta resolved no movable existing-LDS operations";
    return failure();
  }

  FailureOr<ttg::ModuloScheduleResult> schedule =
      ttg::runModuloScheduling(ddg, ttg::getActiveScheduleAlgo(),
                               /*maxII=*/0,
                               /*maxBacktracks=*/ddg.getNumNodes() * 16);
  if (failed(schedule) || !ttg::isLegalModuloSchedule(ddg, *schedule)) {
    error = "no legal constrained modulo order exists for selected pipeline";
    return failure();
  }

  resolved.desiredOrder = baselineOrder;
  llvm::stable_sort(resolved.desiredOrder,
                    [&](Operation *left, Operation *right) {
                      unsigned leftIndex = ddg.getOpToIdx().lookup(left);
                      unsigned rightIndex = ddg.getOpToIdx().lookup(right);
                      int leftCycle = schedule->nodeToCycle.lookup(leftIndex);
                      int rightCycle = schedule->nodeToCycle.lookup(rightIndex);
                      return std::tie(leftCycle, baselinePosition[left]) <
                             std::tie(rightCycle, baselinePosition[right]);
                    });
  int64_t moved = 0;
  for (auto [position, operation] : llvm::enumerate(resolved.desiredOrder))
    if (baselinePosition[operation] != static_cast<int64_t>(position))
      ++moved;

  resolved.record.initiationInterval = schedule->II;
  resolved.record.operationCount = baselineOrder.size();
  resolved.record.selectedOperationCount = selected.size();
  resolved.record.movedOperationCount = moved;
  resolved.record.importedDependencyCount = imported;
  resolved.record.skippedDependencyCount = skipped;
  return success();
}

static LogicalResult
verifyPostRewrite(const tt::plan::PlanValueGraph &postGraph,
                  const llvm::DenseMap<Operation *, std::string> &postBindings,
                  const llvm::DenseMap<Value, std::string> &postValueBindings,
                  MutableArrayRef<ResolvedLoopPipeline> resolved,
                  std::string &error) {
  for (const tt::plan::PlanUnresolvedFact &fact :
       postGraph.getUnresolvedFacts()) {
    if (fact.importance == "important" && fact.status == "open") {
      error =
          "post-rewrite audit retained an open important fact: " + fact.code;
      return failure();
    }
  }
  if (!postGraph.getLdsReuseHazards().empty()) {
    error = "post-rewrite audit found an LDS reuse hazard";
    return failure();
  }

  std::map<std::string, const tt::plan::PlanAsyncTransaction *>
      transactionByProducer;
  for (const tt::plan::PlanAsyncTransaction &transaction :
       postGraph.getAsyncTransactions())
    transactionByProducer[transaction.producerOperationId] = &transaction;
  std::map<std::string, const tt::plan::PlanAsyncWaitRecord *> waitByOperation;
  for (const tt::plan::PlanAsyncWaitRecord &wait : postGraph.getAsyncWaits())
    waitByOperation[wait.operationId] = &wait;
  std::map<std::string, const tt::plan::PlanLdsAllocationRecord *>
      allocationByOperation;
  for (const tt::plan::PlanLdsAllocationRecord &allocation :
       postGraph.getLdsAllocations())
    allocationByOperation[allocation.allocationOperationId] = &allocation;

  TLXAMDLatencyModel latencyModel;
  llvm::DenseMap<Operation *, ttg::DataPartitionInfo> noPartition;
  for (ResolvedLoopPipeline &loop : resolved) {
    bool hasStorageMutation =
        !loop.ringMutations.empty() || !loop.stagingMutations.empty();
    if (hasStorageMutation &&
        postGraph.getResourceSummary().logicalLdsAllocationBytes !=
            loop.record.logicalLdsBytesAfter) {
      error = "post-rewrite logical LDS usage does not match the capacity plan";
      return failure();
    }
    for (const tta::PlanExistingLdsRingMutation &ring : loop.ringMutations) {
      ttg::LocalAllocOp allocation = ring.allocation;
      auto type = dyn_cast<ttg::MemDescType>(allocation.getType());
      if (!type || type.getShape().empty() ||
          type.getShape().front() != ring.newDepth) {
        error = "post-rewrite LDS allocation depth does not match the plan";
        return failure();
      }
      for (Operation *producer : ring.producers) {
        auto producerId = postBindings.find(producer);
        if (producerId == postBindings.end()) {
          error = "post-rewrite producer lost stable Plan IR identity";
          return failure();
        }
        auto transaction = transactionByProducer.find(producerId->second);
        if (transaction == transactionByProducer.end() ||
            !hasExactSlotDepth(*transaction->second, ring.newDepth) ||
            !hasExactConsumerDistance(*transaction->second, ring.newDistance)) {
          error = "post-rewrite transaction does not prove the requested "
                  "depth and distance";
          return failure();
        }
        const tt::plan::PlanAsyncTransaction &post = *transaction->second;
        bool hasVisibility =
            llvm::any_of(post.visibilityFrontiers, [&](const auto &frontier) {
              return frontier.iterationDistance == ring.newDistance;
            });
        bool hasRelease =
            llvm::any_of(post.releaseFrontiers, [&](const auto &frontier) {
              return frontier.iterationDistance == ring.newDistance;
            });
        bool hasOverwrite =
            llvm::any_of(post.overwriteFrontiers, [&](const auto &frontier) {
              return frontier.iterationDistance == ring.newDepth;
            });
        if (!hasVisibility || !hasRelease || !hasOverwrite) {
          error = "post-rewrite transaction lacks a proven visibility, "
                  "release, or overwrite frontier";
          return failure();
        }
      }
    }
    for (const tta::PlanAsyncWaitMutation &mutation : loop.waitMutations) {
      auto waitId = postBindings.find(mutation.wait);
      auto wait = waitId == postBindings.end()
                      ? waitByOperation.end()
                      : waitByOperation.find(waitId->second);
      if (wait == waitByOperation.end() ||
          wait->second->retainedGroupCount != mutation.newRetainedGroupCount) {
        error = "post-rewrite async wait count does not match the plan";
        return failure();
      }
    }
    for (tta::PlanRegisterToLdsStaging &staging : loop.stagingMutations) {
      if (!staging.allocation || !staging.store || !staging.load ||
          !staging.visibilityBarrier || !staging.releaseBarrier) {
        error = "post-rewrite register-to-LDS staging is incomplete";
        return failure();
      }
      auto allocationType =
          dyn_cast<ttg::MemDescType>(staging.allocation.getType());
      if (!allocationType ||
          allocationType.getShape() != staging.tensorType.getShape() ||
          allocationType.getElementType() !=
              staging.tensorType.getElementType()) {
        error = "post-rewrite staging allocation type does not match the "
                "staged tensor";
        return failure();
      }
      auto alignment =
          staging.allocation->getAttrOfType<IntegerAttr>("alignment");
      if (!alignment || alignment.getInt() != staging.alignment) {
        error = "post-rewrite staging allocation lost its alignment";
        return failure();
      }
      auto allocationId = postBindings.find(staging.allocation);
      auto allocationRecord =
          allocationId == postBindings.end()
              ? allocationByOperation.end()
              : allocationByOperation.find(allocationId->second);
      if (allocationRecord == allocationByOperation.end() ||
          allocationRecord->second->logicalBytes != staging.logicalBytes ||
          allocationRecord->second->alignment != staging.alignment) {
        error = "post-rewrite Plan IR does not describe the requested staging "
                "allocation";
        return failure();
      }
      if (staging.store.getSrc() != staging.source ||
          staging.store.getDst() != staging.allocation.getResult() ||
          staging.load.getSrc() != staging.allocation.getResult() ||
          staging.load.getType() != staging.tensorType) {
        error = "post-rewrite register-to-LDS data path is inconsistent";
        return failure();
      }
      for (const auto &[operand, value] : staging.preservedOperands) {
        if (!operand || operand->get() != value) {
          error = "post-rewrite staging changed an unselected consumer";
          return failure();
        }
      }
      if (staging.consumerOperands.size() !=
          staging.consumerReplacementValues.size()) {
        error = "post-rewrite staging lost a named consumer replacement";
        return failure();
      }
      for (auto [operand, replacement] : llvm::zip(
               staging.consumerOperands, staging.consumerReplacementValues)) {
        if (!operand || operand->get() != replacement) {
          error = "post-rewrite named consumer does not use its staged "
                  "derived value";
          return failure();
        }
      }
      auto sourceId = postValueBindings.find(staging.source);
      if (sourceId == postValueBindings.end()) {
        error = "post-rewrite staged source lost stable Plan IR identity";
        return failure();
      }
      const tt::plan::PlanLiveSegment *sourceSegment = nullptr;
      for (const tt::plan::PlanLiveSegment &segment :
           postGraph.getLiveSegments()) {
        if (segment.valueId != sourceId->second)
          continue;
        if (sourceSegment) {
          error = "post-rewrite staged source has multiple live segments";
          return failure();
        }
        sourceSegment = &segment;
      }
      if (!sourceSegment) {
        error = "post-rewrite staged source has no live segment";
        return failure();
      }
      staging.sourceLiveStartAfter = sourceSegment->startPosition;
      staging.sourceLiveEndAfter = sourceSegment->endPosition;
      int64_t beforeLength =
          staging.sourceLiveEndBefore - staging.sourceLiveStartBefore;
      int64_t afterLength =
          staging.sourceLiveEndAfter - staging.sourceLiveStartAfter;
      staging.logicalLiveRangeShortened = afterLength < beforeLength;
      if (!staging.logicalLiveRangeShortened) {
        error = "staging_does_not_shorten_lifetime";
        return failure();
      }
      if (!staging.store->isBeforeInBlock(staging.visibilityBarrier) ||
          !staging.visibilityBarrier->isBeforeInBlock(staging.load) ||
          !staging.load->isBeforeInBlock(staging.consumers.front()) ||
          !staging.consumers.back()->isBeforeInBlock(staging.releaseBarrier)) {
        error = "post-rewrite staging synchronization order is invalid";
        return failure();
      }
    }
    if (!loop.ringMutations.empty()) {
      ttg::DataDependenceGraph ddg = ttg::DataDependenceGraph::build(
          loop.loop, latencyModel, noPartition, ttg::getActiveScheduleAlgo(),
          /*scheduleNestedLoops=*/false);
      FailureOr<ttg::ModuloScheduleResult> schedule =
          ttg::runModuloScheduling(ddg, ttg::getActiveScheduleAlgo(),
                                   /*maxII=*/0,
                                   /*maxBacktracks=*/ddg.getNumNodes() * 16);
      if (failed(schedule) || !ttg::isLegalModuloSchedule(ddg, *schedule)) {
        error = "post-rewrite DDG has no legal modulo schedule";
        return failure();
      }
      loop.record.postRewriteDdgVerified = true;
    } else if (!loop.stagingMutations.empty()) {
      // Single-slot synchronous staging does not overlap loop iterations, so
      // it needs a rebuilt DDG but not a modulo schedule. Verify that the
      // emitted order satisfies every distance-zero edge in that rebuilt DDG.
      ttg::DataDependenceGraph ddg = ttg::DataDependenceGraph::build(
          loop.loop, latencyModel, noPartition, ttg::getActiveScheduleAlgo(),
          /*scheduleNestedLoops=*/false);
      for (const ttg::DDGEdge &edge : ddg.getEdges()) {
        if (edge.distance != 0)
          continue;
        Operation *source = ddg.getNode(edge.srcIdx).op;
        Operation *destination = ddg.getNode(edge.dstIdx).op;
        if (source != destination &&
            source->getBlock() == destination->getBlock() &&
            destination->isBeforeInBlock(source)) {
          error = "post-rewrite staging order violates a distance-zero DDG "
                  "dependency";
          return failure();
        }
      }
      loop.record.postRewriteDdgVerified = true;
    }
  }
  return success();
}

static void recordStagingResults(ResolvedLoopPipeline &loop) {
  loop.record.staging.clear();
  for (const tta::PlanRegisterToLdsStaging &staging : loop.stagingMutations) {
    tt::plan::PlanPipelineStagingApplyRecord record;
    record.valueId = staging.sourceValueId;
    record.derivedOperationsCloned = staging.derivedOperationsCloned;
    record.derivedOperationsPruned = staging.derivedOperationsPruned;
    record.selectedConsumerOperands = staging.consumerOperands.size();
    record.unselectedConsumersPreserved = staging.unselectedConsumersPreserved;
    record.sourceLiveStartBefore = staging.sourceLiveStartBefore;
    record.sourceLiveEndBefore = staging.sourceLiveEndBefore;
    record.sourceLiveStartAfter = staging.sourceLiveStartAfter;
    record.sourceLiveEndAfter = staging.sourceLiveEndAfter;
    record.logicalLiveRangeShortened = staging.logicalLiveRangeShortened;
    loop.record.staging.push_back(std::move(record));
  }
}

static LogicalResult
applyPlanPipeline(tt::FuncOp function, const tt::plan::PlanPipelineDelta &delta,
                  tt::plan::PlanPipelineApplyResult &result, std::string &error,
                  bool strict) {
  result.kernel = function.getName().str();
  if (delta.kernel != function.getName())
    return fail(result, error, "pipeline delta kernel does not match function");

  llvm::DenseMap<Operation *, std::string> operationBindings;
  llvm::DenseMap<Value, std::string> valueBindings;
  FailureOr<tt::plan::PlanValueGraph> graph = tt::plan::PlanValueGraph::build(
      function, &operationBindings, &valueBindings);
  if (failed(graph) || (strict && failed(graph->verify(/*strict=*/true))))
    return fail(result, error,
                "failed to build a strict pre-apply value graph");
  result.inputValueGraphFingerprint = graph->getSemanticFingerprint().str();
  if (delta.inputValueGraphFingerprint != graph->getSemanticFingerprint())
    return fail(result, error,
                "pipeline delta value-graph fingerprint does not match");
  std::string dotSignatureBefore = frozenDotSignature(function);

  std::map<std::string, Operation *> operationById;
  for (const auto &[operation, id] : operationBindings)
    operationById[id] = operation;
  std::map<std::string, Value> valueById;
  for (const auto &[value, id] : valueBindings)
    valueById[id] = value;

  SmallVector<ResolvedLoopPipeline, 1> resolved;
  for (const tt::plan::PlanLoopPipelineDelta &requested : delta.loops) {
    auto operationIt = operationById.find(requested.loopId);
    if (operationIt == operationById.end())
      return fail(result, error, "pipeline delta refers to unknown loop");
    auto loop = dyn_cast<scf::ForOp>(operationIt->second);
    if (!loop)
      return fail(result, error, "pipeline delta target is not an scf.for");
    ResolvedLoopPipeline current;
    if (failed(resolveLoopPipeline(requested, loop, *graph, operationById,
                                   valueById, current, error))) {
      result.error = error;
      return failure();
    }
    resolved.push_back(std::move(current));
  }

  for (ResolvedLoopPipeline &loop : resolved) {
    Operation *terminator = loop.loop.getBody()->getTerminator();
    for (Operation *operation : loop.desiredOrder)
      operation->moveBefore(terminator);

    tta::PlanRingMaterializationResult materialization;
    if (failed(tta::materializeExistingLdsRings(
            loop.ringMutations, loop.waitMutations, materialization, error))) {
      result.error = error;
      return failure();
    }
    loop.record.rewrittenSlotIndexCount = materialization.rewrittenSlotIndices;
    loop.record.updatedWaitCount = materialization.updatedWaits;
    loop.record.insertedBarrierCount =
        materialization.insertedVisibilityBarriers +
        materialization.insertedReleaseBarriers;
    tta::PlanStagingMaterializationResult stagingMaterialization;
    if (failed(tta::materializeRegisterToLdsStaging(
            loop.stagingMutations, stagingMaterialization, error))) {
      result.error = error;
      return failure();
    }
    loop.record.insertedBarrierCount += stagingMaterialization.insertedBarriers;
    for (const tta::PlanExistingLdsRingMutation &ring : loop.ringMutations) {
      result.changesBufferDepth |= ring.oldDepth != ring.newDepth;
      result.changesPrefetchDistance |= ring.oldDistance != ring.newDistance;
    }
    result.changesIterationStorage |= result.changesBufferDepth;
    result.changesSynchronization |= !loop.ringMutations.empty();
    result.changesNewStaging |= !loop.stagingMutations.empty();
    result.changesIterationStorage |= !loop.stagingMutations.empty();
    result.changesSynchronization |= !loop.stagingMutations.empty();
  }
  if (failed(mlir::verify(function)))
    return fail(result, error,
                "MLIR verification failed after pipeline schedule apply");

  llvm::DenseMap<Operation *, std::string> postBindings;
  llvm::DenseMap<Value, std::string> postValueBindings;
  FailureOr<tt::plan::PlanValueGraph> postGraph =
      tt::plan::PlanValueGraph::build(function, &postBindings,
                                      &postValueBindings);
  if (failed(postGraph) ||
      (strict && failed(postGraph->verify(/*strict=*/true))))
    return fail(result, error,
                "failed to build a strict post-apply value graph");
  result.outputValueGraphFingerprint =
      postGraph->getSemanticFingerprint().str();
  bool hasStorageMutation =
      llvm::any_of(resolved, [](const ResolvedLoopPipeline &loop) {
        return !loop.ringMutations.empty() || !loop.stagingMutations.empty();
      });
  if (!hasStorageMutation) {
    if (result.outputValueGraphFingerprint != result.inputValueGraphFingerprint)
      return fail(
          result, error,
          "stable operation/value identity changed after pipeline apply");
    if (postGraph->getLdsAllocations().size() !=
            graph->getLdsAllocations().size() ||
        postGraph->getAsyncGroups().size() != graph->getAsyncGroups().size() ||
        postGraph->getAsyncTransactions().size() !=
            graph->getAsyncTransactions().size() ||
        postGraph->getAsyncWaits().size() != graph->getAsyncWaits().size() ||
        postGraph->getLdsReuseHazards().size() !=
            graph->getLdsReuseHazards().size())
      return fail(result, error,
                  "storage or synchronization structure changed after "
                  "pipeline apply");
  } else {
    if (frozenDotSignature(function) != dotSignatureBefore)
      return fail(result, error,
                  "dot decomposition or accumulator contract changed during "
                  "pipeline materialization");
    size_t newStagingCount = 0;
    for (const ResolvedLoopPipeline &loop : resolved)
      newStagingCount += loop.stagingMutations.size();
    if (postGraph->getLdsAllocations().size() !=
            graph->getLdsAllocations().size() + newStagingCount ||
        postGraph->getAsyncGroups().size() != graph->getAsyncGroups().size() ||
        postGraph->getAsyncTransactions().size() !=
            graph->getAsyncTransactions().size() ||
        postGraph->getAsyncWaits().size() != graph->getAsyncWaits().size())
      return fail(result, error,
                  "pipeline materialization changed an unrequested LDS or "
                  "async structure");
    if (failed(verifyPostRewrite(*postGraph, postBindings, postValueBindings,
                                 resolved, error))) {
      result.error = error;
      return failure();
    }
    result.postRewriteAuditPassed = true;
  }

  for (ResolvedLoopPipeline &loop : resolved) {
    recordStagingResults(loop);
    result.movedOperationCount += loop.record.movedOperationCount;
    result.importedDependencyCount += loop.record.importedDependencyCount;
    result.skippedDependencyCount += loop.record.skippedDependencyCount;
    result.loops.push_back(std::move(loop.record));
  }
  result.accepted = true;
  return success();
}

struct TritonAMDGPUApplyPlanPipelinePass
    : impl::TritonAMDGPUApplyPlanPipelineBase<
          TritonAMDGPUApplyPlanPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    tt::plan::PlanPipelineApplyResult result;
    std::string error;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
        llvm::MemoryBuffer::getFile(inputPath);
    if (!input) {
      error = "cannot read pipeline delta '" + inputPath +
              "': " + input.getError().message();
      result.error = error;
      (void)writeReport(module, reportPath,
                        tt::plan::serializePlanPipelineApplyReport(result));
      module.emitError(error);
      return signalPassFailure();
    }
    FailureOr<tt::plan::PlanPipelineDelta> delta =
        tt::plan::parsePlanPipelineDelta((*input)->getBuffer(), error);
    if (failed(delta)) {
      result.error = error;
      (void)writeReport(module, reportPath,
                        tt::plan::serializePlanPipelineApplyReport(result));
      module.emitError(error);
      return signalPassFailure();
    }

    tt::FuncOp target;
    for (tt::FuncOp function : module.getOps<tt::FuncOp>())
      if (function.getName() == delta->kernel) {
        target = function;
        break;
      }
    if (!target) {
      if (allowMissingKernel)
        return;
      result.kernel = delta->kernel;
      result.error = "pipeline delta kernel is not present in the module";
      (void)writeReport(module, reportPath,
                        tt::plan::serializePlanPipelineApplyReport(result));
      module.emitError(result.error);
      return signalPassFailure();
    }
    if (failed(applyPlanPipeline(target, *delta, result, error,
                                 /*strict=*/strict))) {
      (void)writeReport(module, reportPath,
                        tt::plan::serializePlanPipelineApplyReport(result));
      target.emitError(error);
      return signalPassFailure();
    }
    if (failed(writeReport(module, reportPath,
                           tt::plan::serializePlanPipelineApplyReport(result))))
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir
