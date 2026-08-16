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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include <algorithm>
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
      if (index.modulus <= 0)
        continue;
      found = true;
      if (index.modulus != depth)
        return false;
    }
  return found;
}

static bool
hasExactConsumerDistance(const tt::plan::PlanAsyncTransaction &transaction,
                         int64_t distance) {
  bool found = false;
  for (const tt::plan::PlanAsyncFrontier &frontier :
       transaction.consumerFrontiers) {
    if (frontier.iterationDistance <= 0)
      continue;
    found = true;
    if (frontier.iterationDistance != distance)
      return false;
  }
  return found;
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

  llvm::DenseSet<Operation *> selected;
  SmallVector<std::tuple<Operation *, Operation *, unsigned>> releaseToProducer;
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
      if (transaction.commitGroupId != intent.groupId ||
          !hasExactSlotDepth(transaction, intent.bufferDepth) ||
          !hasExactConsumerDistance(transaction, intent.distance)) {
        error = "async group '" + intent.groupId +
                "' requests a distance or buffer depth change reserved for "
                "M1.5b.3";
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
      for (const tt::plan::PlanAsyncFrontier &frontier :
           transaction.releaseFrontiers) {
        auto releaseIt = operationById.find(frontier.operationId);
        if (releaseIt == operationById.end())
          continue;
        Operation *release = projectToLoopBody(releaseIt->second, loop);
        if (release && frontier.iterationDistance > 0)
          releaseToProducer.push_back(
              {release, producerIt->second,
               static_cast<unsigned>(frontier.iterationDistance)});
      }
    }
  }

  for (const tt::plan::PlanAsyncWaitRecord &wait : graph.getAsyncWaits()) {
    std::set<std::string> family;
    bool touchesRequested = false;
    int64_t distance = -1;
    for (const tt::plan::PlanAsyncWaitCompletion &completion :
         wait.completedGroups) {
      if (requestedGroups.count(completion.groupId) &&
          completion.iterationDistance > 0) {
        touchesRequested = true;
        distance = completion.iterationDistance;
      }
    }
    if (!touchesRequested)
      continue;
    for (const tt::plan::PlanAsyncWaitCompletion &completion :
         wait.completedGroups) {
      if (completion.iterationDistance != distance)
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
  }

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
applyPlanPipeline(tt::FuncOp function, const tt::plan::PlanPipelineDelta &delta,
                  tt::plan::PlanPipelineApplyResult &result, std::string &error,
                  bool strict) {
  result.kernel = function.getName().str();
  if (delta.kernel != function.getName())
    return fail(result, error, "pipeline delta kernel does not match function");

  llvm::DenseMap<Operation *, std::string> operationBindings;
  FailureOr<tt::plan::PlanValueGraph> graph =
      tt::plan::PlanValueGraph::build(function, &operationBindings);
  if (failed(graph) || (strict && failed(graph->verify(/*strict=*/true))))
    return fail(result, error,
                "failed to build a strict pre-apply value graph");
  result.inputValueGraphFingerprint = graph->getSemanticFingerprint().str();
  if (delta.inputValueGraphFingerprint != graph->getSemanticFingerprint())
    return fail(result, error,
                "pipeline delta value-graph fingerprint does not match");

  std::map<std::string, Operation *> operationById;
  for (const auto &[operation, id] : operationBindings)
    operationById[id] = operation;

  SmallVector<ResolvedLoopPipeline> resolved;
  for (const tt::plan::PlanLoopPipelineDelta &requested : delta.loops) {
    auto operationIt = operationById.find(requested.loopId);
    if (operationIt == operationById.end())
      return fail(result, error, "pipeline delta refers to unknown loop");
    auto loop = dyn_cast<scf::ForOp>(operationIt->second);
    if (!loop)
      return fail(result, error, "pipeline delta target is not an scf.for");
    ResolvedLoopPipeline current;
    if (failed(resolveLoopPipeline(requested, loop, *graph, operationById,
                                   current, error))) {
      result.error = error;
      return failure();
    }
    resolved.push_back(std::move(current));
  }

  for (ResolvedLoopPipeline &loop : resolved) {
    Operation *terminator = loop.loop.getBody()->getTerminator();
    for (Operation *operation : loop.desiredOrder)
      operation->moveBefore(terminator);
  }
  if (failed(mlir::verify(function)))
    return fail(result, error,
                "MLIR verification failed after pipeline schedule apply");

  FailureOr<tt::plan::PlanValueGraph> postGraph =
      tt::plan::PlanValueGraph::build(function);
  if (failed(postGraph) ||
      (strict && failed(postGraph->verify(/*strict=*/true))))
    return fail(result, error,
                "failed to build a strict post-apply value graph");
  result.outputValueGraphFingerprint =
      postGraph->getSemanticFingerprint().str();
  if (result.outputValueGraphFingerprint != result.inputValueGraphFingerprint)
    return fail(result, error,
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
                "storage or synchronization structure changed after pipeline "
                "apply");

  for (ResolvedLoopPipeline &loop : resolved) {
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
