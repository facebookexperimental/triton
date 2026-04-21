// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Pass A: Modulo Schedule Pass
//
// Builds a DDG from scf.for loop bodies, computes MinII, runs Rau's iterative
// modulo scheduling, and annotates ops with loop.stage and loop.cluster
// attributes for downstream pipelining passes.

#include "DataDependenceGraph.h"
#include "LatencyModel.h"
#include "ModuloReservationTable.h"
#include "ModuloScheduleGraph.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "nvgpu-modulo-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

namespace {

// ============================================================================
// Emit loop.stage / loop.cluster attributes from modulo schedule
// ============================================================================

static void emitScheduleAttributes(scf::ForOp loop,
                                   const ttg::DataDependenceGraph &ddg,
                                   const ttg::ModuloScheduleResult &schedule) {
  const int II = schedule.II;
  const int maxStage = schedule.getMaxStage();
  auto ctx = loop.getContext();

  // Step 2.5: Compute per-stage cluster IDs from modulo cycles.
  // Ops in the same stage are ordered by cycle: lower cycle → lower cluster ID.
  // This preserves the modulo schedule's within-stage ordering for downstream
  // pipelining, instead of relying on IR program order.
  llvm::DenseMap<int, SmallVector<int>> stageToCycles;
  for (const auto &node : ddg.getNodes()) {
    auto it = schedule.nodeToCycle.find(node.idx);
    if (it == schedule.nodeToCycle.end())
      continue;
    int stage = it->second / II;
    stageToCycles[stage].push_back(it->second);
  }
  // Deduplicate and sort cycles per stage to assign dense cluster IDs.
  llvm::DenseMap<int, llvm::DenseMap<int, int>> stageAndCycleToCluster;
  for (auto &[stage, cycles] : stageToCycles) {
    llvm::sort(cycles);
    cycles.erase(llvm::unique(cycles), cycles.end());
    for (int i = 0, e = cycles.size(); i < e; ++i)
      stageAndCycleToCluster[stage][cycles[i]] = i;
  }

  for (const auto &node : ddg.getNodes()) {
    auto it = schedule.nodeToCycle.find(node.idx);
    if (it == schedule.nodeToCycle.end())
      continue;
    // For multi-stage super-nodes (prologue/kloop/epilogue sharing the same
    // Operation*), only write attrs from the node registered in opToIdx
    // (the epilogue) to avoid overwrites.
    auto opIt = ddg.getOpToIdx().find(node.op);
    if (opIt != ddg.getOpToIdx().end() && opIt->second != node.idx)
      continue;
    int stage = it->second / II;
    int cycle = it->second;
    int clusterId = stageAndCycleToCluster[stage][cycle];
    node.op->setAttr(tt::kLoopStageAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), stage));
    node.op->setAttr(tt::kLoopClusterAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), clusterId));
    // Emit raw cycle for downstream buffer depth computation (Step 3).
    node.op->setAttr("tt.modulo_cycle",
                     IntegerAttr::get(IntegerType::get(ctx, 32), cycle));
  }

  // Ensure ALL ops in the loop body have loop.stage/loop.cluster attrs.
  // Downstream passes assert every op is in the schedule.
  for (auto &op : loop.getBody()->without_terminator()) {
    if (!op.hasAttr(tt::kLoopStageAttrName))
      op.setAttr(tt::kLoopStageAttrName,
                 IntegerAttr::get(IntegerType::get(ctx, 32), 0));
    if (!op.hasAttr(tt::kLoopClusterAttrName))
      op.setAttr(tt::kLoopClusterAttrName,
                 IntegerAttr::get(IntegerType::get(ctx, 32), 0));
  }

  LDBG("Emitted schedule: II=" << II << " maxStage=" << maxStage);

  loop->setAttr("tt.modulo_ii",
                IntegerAttr::get(IntegerType::get(ctx, 32), II));
  loop->setAttr(tt::kScheduledMaxStageAttrName,
                IntegerAttr::get(IntegerType::get(ctx, 32), maxStage));
}

// ============================================================================
// Step 3: Derive per-resource buffer depths from modulo schedule
// ============================================================================

// Blackwell sm_100 SMEM budget (reserve some for barriers/scratch).
constexpr int kSmemBudgetBytes = 228 * 1024;

// Blackwell TMEM budget: 256KB total tensor memory (128 lanes × 512 cols × 4B).
constexpr int kTmemBudgetBytes = 256 * 1024;

// Fallback trip count when the loop bounds aren't constant-foldable.
// Used so kernel_time_cost can give a finite (rather than div-by-zero)
// answer for cost-based depth reduction.
constexpr int kEstimatedTripCount = 4;

// computeBufferDepths removed — buffer allocation is now done via
// allocateBuffersForLoop on the ScheduleGraph (stage-diff based).

// ============================================================================
// Phase 0d: Build ScheduleGraph from DDG + Schedule
// ============================================================================

static ttg::ScheduleNode
convertDDGNode(const ttg::DDGNode &ddgNode, unsigned nodeId,
               const ttg::ModuloScheduleResult &sched) {
  ttg::ScheduleNode sn;
  sn.id = nodeId;
  sn.op = ddgNode.op;
  sn.pipeline = ddgNode.pipeline;
  sn.latency = ddgNode.latency;
  sn.selfLatency = ddgNode.selfLatency;

  auto cycleIt = sched.nodeToCycle.find(ddgNode.idx);
  if (cycleIt != sched.nodeToCycle.end()) {
    sn.cycle = cycleIt->second;
    sn.stage = cycleIt->second / sched.II;
  }

  if (ddgNode.isSuperNode) {
    sn.prologueLatency = ddgNode.prologueLatency;
  }
  return sn;
}

/// Step 2.5: Compute dense cluster IDs within each stage.
/// Ops in the same stage are sorted by cycle; same cycle → same cluster,
/// different cycle → different cluster (lower cycle = lower cluster ID).
static void computeClusterIds(ttg::ScheduleLoop &loop) {
  // Group node indices by stage
  llvm::DenseMap<int, SmallVector<unsigned>> stageToNodes;
  for (auto &node : loop.nodes) {
    stageToNodes[node.stage].push_back(node.id);
  }

  for (auto &[stage, nodeIds] : stageToNodes) {
    // Collect unique cycles in this stage, sorted
    SmallVector<int> cycles;
    for (unsigned nid : nodeIds)
      cycles.push_back(loop.nodes[nid].cycle);
    llvm::sort(cycles);
    cycles.erase(llvm::unique(cycles), cycles.end());

    // Build cycle → dense cluster ID map
    llvm::DenseMap<int, int> cycleToCluster;
    for (int i = 0, e = cycles.size(); i < e; ++i)
      cycleToCluster[cycles[i]] = i;

    // Assign cluster IDs
    for (unsigned nid : nodeIds)
      loop.nodes[nid].cluster = cycleToCluster[loop.nodes[nid].cycle];
  }
}

/// Build a child ScheduleLoop for an inner scf.for loop (super-node).
static unsigned buildChildScheduleLoop(scf::ForOp innerLoop,
                                       ttg::ScheduleGraph &graph,
                                       const ttg::LatencyModel &model) {
  auto innerDDG = ttg::DataDependenceGraph::build(innerLoop, model);
  unsigned loopId = graph.addLoop(innerLoop);
  auto &schedLoop = graph.getLoop(loopId);

  if (innerDDG.getNumNodes() == 0)
    return loopId;

  auto innerSched = ttg::runModuloScheduling(innerDDG);
  if (failed(innerSched))
    return loopId;

  schedLoop.II = innerSched->II;
  schedLoop.maxStage = innerSched->getMaxStage();

  int tcStart = innerSched->II;
  for (const auto &node : innerDDG.getNodes()) {
    if (node.pipeline == ttg::HWPipeline::TC) {
      auto it = innerSched->nodeToCycle.find(node.idx);
      if (it != innerSched->nodeToCycle.end())
        tcStart = std::min(tcStart, it->second);
    }
  }
  schedLoop.prologueLatency = tcStart;

  schedLoop.tripCount = kEstimatedTripCount;
  schedLoop.tripCountIsEstimated = true;
  {
    auto lb = innerLoop.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
    auto ub = innerLoop.getUpperBound().getDefiningOp<arith::ConstantIntOp>();
    auto step = innerLoop.getStep().getDefiningOp<arith::ConstantIntOp>();
    if (lb && ub && step && step.value() > 0) {
      int64_t tc = (ub.value() - lb.value() + step.value() - 1) / step.value();
      if (tc > 0) {
        schedLoop.tripCount = static_cast<int>(tc);
        schedLoop.tripCountIsEstimated = false;
      }
    }
  }

  llvm::DenseMap<unsigned, unsigned> ddgToPipe;
  for (const auto &ddgNode : innerDDG.getNodes()) {
    unsigned nodeId = schedLoop.nodes.size();
    ddgToPipe[ddgNode.idx] = nodeId;
    auto sn = convertDDGNode(ddgNode, nodeId, *innerSched);

    if (ddgNode.isSuperNode) {
      if (auto nestedLoop = dyn_cast<scf::ForOp>(ddgNode.op)) {
        unsigned childId = buildChildScheduleLoop(nestedLoop, graph, model);
        sn.childPipelineId = childId;
      }
    }

    schedLoop.nodes.push_back(sn);
    schedLoop.opToNodeId[ddgNode.op] = nodeId;
  }

  for (const auto &ddgEdge : innerDDG.getEdges()) {
    auto srcIt = ddgToPipe.find(ddgEdge.srcIdx);
    auto dstIt = ddgToPipe.find(ddgEdge.dstIdx);
    if (srcIt == ddgToPipe.end() || dstIt == ddgToPipe.end())
      continue;
    ttg::ScheduleEdge se;
    se.srcId = srcIt->second;
    se.dstId = dstIt->second;
    se.latency = ddgEdge.latency;
    se.distance = ddgEdge.distance;
    schedLoop.edges.push_back(se);
  }

  // Step 2.5: compute cluster IDs
  computeClusterIds(schedLoop);

  return loopId;
}

/// Build the top-level ScheduleLoop for a scheduled loop.
static unsigned buildScheduleLoop(scf::ForOp loop,
                                  const ttg::DataDependenceGraph &ddg,
                                  const ttg::ModuloScheduleResult &sched,
                                  ttg::ScheduleGraph &graph,
                                  const ttg::LatencyModel &model) {
  unsigned loopId = graph.addLoop(loop);
  auto &schedLoop = graph.getLoop(loopId);
  schedLoop.II = sched.II;
  schedLoop.maxStage = sched.getMaxStage();

  int tcStart = sched.II;
  for (const auto &node : ddg.getNodes()) {
    if (node.pipeline == ttg::HWPipeline::TC || node.isSuperNode) {
      auto it = sched.nodeToCycle.find(node.idx);
      if (it != sched.nodeToCycle.end())
        tcStart = std::min(tcStart, it->second);
    }
  }
  schedLoop.prologueLatency = tcStart;

  // Extract trip count
  schedLoop.tripCount = kEstimatedTripCount;
  schedLoop.tripCountIsEstimated = true;
  {
    auto lb = loop.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
    auto ub = loop.getUpperBound().getDefiningOp<arith::ConstantIntOp>();
    auto step = loop.getStep().getDefiningOp<arith::ConstantIntOp>();
    if (lb && ub && step && step.value() > 0) {
      int64_t tc = (ub.value() - lb.value() + step.value() - 1) / step.value();
      if (tc > 0) {
        schedLoop.tripCount = static_cast<int>(tc);
        schedLoop.tripCountIsEstimated = false;
      }
    }
  }

  llvm::DenseMap<unsigned, unsigned> ddgToPipe;
  for (const auto &ddgNode : ddg.getNodes()) {
    unsigned nodeId = schedLoop.nodes.size();
    ddgToPipe[ddgNode.idx] = nodeId;
    auto sn = convertDDGNode(ddgNode, nodeId, sched);

    if (ddgNode.isSuperNode) {
      if (auto innerLoop = dyn_cast<scf::ForOp>(ddgNode.op)) {
        unsigned childId = buildChildScheduleLoop(innerLoop, graph, model);
        sn.childPipelineId = childId;
        // Do NOT overwrite sn.prologueLatency: it was copied from
        // ddgNode.prologueLatency by convertDDGNode and represents the
        // latency the parent scheduler actually used for this super-node's
        // edge model. The child's recomputed prologueLatency belongs to
        // the child ScheduleLoop's own metadata; for empty/unscheduled
        // children it's 0 and would underestimate the super-node here.
      }
    }

    schedLoop.nodes.push_back(sn);
    schedLoop.opToNodeId[ddgNode.op] = nodeId;
  }

  for (const auto &ddgEdge : ddg.getEdges()) {
    auto srcIt = ddgToPipe.find(ddgEdge.srcIdx);
    auto dstIt = ddgToPipe.find(ddgEdge.dstIdx);
    if (srcIt == ddgToPipe.end() || dstIt == ddgToPipe.end())
      continue;
    ttg::ScheduleEdge se;
    se.srcId = srcIt->second;
    se.dstId = dstIt->second;
    se.latency = ddgEdge.latency;
    se.distance = ddgEdge.distance;
    schedLoop.edges.push_back(se);
  }

  // Step 2.5: compute cluster IDs
  computeClusterIds(schedLoop);

  return loopId;
}

// ============================================================================
// Phase 1: Buffer Allocation
// ============================================================================

static ttg::MemoryKind classifyMemoryKind(Operation *op) {
  if (isa<ttng::TMEMAllocOp>(op))
    return ttg::MemoryKind::TMEM;
  // Both local_alloc (pre-lowering) and async_tma_copy (post-lowering)
  // produce SMEM buffers that need multi-buffering.
  if (isa<ttg::LocalAllocOp, ttng::AsyncTMACopyGlobalToLocalOp>(op))
    return ttg::MemoryKind::SMEM;
  return ttg::MemoryKind::Register;
}

static void extractBufferShape(Operation *op, ttg::ScheduleBuffer &buf) {
  Type resultType;
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
    resultType = alloc.getType();
  else if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(op))
    resultType = tmemAlloc.getType();
  else if (auto tmaCopy = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    resultType = tmaCopy.getResult().getType();
  else if (op->getNumResults() > 0)
    resultType = op->getResult(0).getType();

  auto extractFromShapedType = [&](llvm::ArrayRef<int64_t> shape, Type elemTy) {
    for (auto dim : shape) {
      if (dim <= 0 || ShapedType::isDynamic(dim))
        return;
    }
    if (!elemTy.isIntOrFloat())
      return;
    for (auto dim : shape)
      buf.shape.push_back(dim);
    buf.elementBitWidth = elemTy.getIntOrFloatBitWidth();
  };

  if (auto memDesc = dyn_cast_or_null<ttg::MemDescType>(resultType)) {
    extractFromShapedType(memDesc.getShape(), memDesc.getElementType());
  } else if (auto tensorType = dyn_cast_or_null<RankedTensorType>(resultType)) {
    extractFromShapedType(tensorType.getShape(), tensorType.getElementType());
  }
}

/// Step 3: Compute buffer count from cycle-level lifetime.
///
/// Design doc formula (§Step 3):
///   lifetime(R) = lastConsumerEnd - producerStart
///   num_buffers(R) = floor(lifetime(R) / II) + 1
///
/// For loop-carried edges (distance > 0), the consumer in iteration i+d
/// effectively ends at: consumerEnd + d * II (in absolute time).
/// This is equivalent to adding d * II to the lifetime.
///
/// Returns the absolute cycle range the buffer is live for. The hold time is
/// `selfLatency` (pipeline occupancy of the consumer) per design doc §414;
/// `latency` (when the consumer's *output* becomes available) is the edge
/// weight, not the buffer hold. We fall back to `latency` when `selfLatency`
/// is 0 (synthetic local_load nodes have selfLat=0 but still need a non-zero
/// hold time = latency).
struct LifetimeRange {
  int liveStart{0};
  int liveEnd{0};
  unsigned count{1};
};

static LifetimeRange computeLifetimeAndCount(const ttg::ScheduleLoop &loop,
                                              unsigned producerNodeId) {
  const auto &producer = loop.getNode(producerNodeId);
  int prodCycle = producer.cycle;
  int II = loop.II;
  if (II <= 0)
    return {prodCycle, prodCycle, 1};

  // Find the latest DIRECT consumer end cycle.
  int lastConsumerEnd = prodCycle;
  for (const auto &edge : loop.edges) {
    if (edge.srcId != producerNodeId)
      continue;
    const auto &consumer = loop.getNode(edge.dstId);
    int hold = consumer.selfLatency > 0 ? consumer.selfLatency : consumer.latency;
    int consumerEnd = consumer.cycle + hold + edge.distance * II;
    lastConsumerEnd = std::max(lastConsumerEnd, consumerEnd);
  }

  int lifetime = lastConsumerEnd - prodCycle;
  unsigned numBuffers = static_cast<unsigned>(std::max(lifetime / II + 1, 1));
  return {prodCycle, lastConsumerEnd, numBuffers};
}

static void allocateBuffersForLoop(ttg::ScheduleLoop &loop) {
  llvm::SmallVector<unsigned, 4> dataBufferIds;
  for (auto &node : loop.nodes) {
    if (!node.op)
      continue;

    auto kind = classifyMemoryKind(node.op);
    if (kind == ttg::MemoryKind::Register)
      continue;

    unsigned bufId = loop.buffers.size();
    ttg::ScheduleBuffer buf;
    buf.id = bufId;
    buf.kind = kind;
    buf.defOp = node.op;
    extractBufferShape(node.op, buf);

    auto life = computeLifetimeAndCount(loop, node.id);
    buf.liveStart = life.liveStart;
    buf.liveEnd = life.liveEnd;
    buf.count = life.count;

    loop.buffers.push_back(buf);
    node.producesBuffer = bufId;

    if (buf.count > 1)
      dataBufferIds.push_back(bufId);

    llvm::DenseSet<unsigned> markedConsumers;
    for (const auto &edge : loop.edges) {
      if (edge.srcId == node.id && markedConsumers.insert(edge.dstId).second)
        loop.nodes[edge.dstId].consumesBuffers.push_back(bufId);
    }
  }

  for (unsigned dataBufId : dataBufferIds) {
    unsigned barId = loop.buffers.size();
    ttg::ScheduleBuffer bar;
    bar.id = barId;
    bar.kind = ttg::MemoryKind::BARRIER;
    bar.count = loop.buffers[dataBufId].count;
    bar.liveStart = loop.buffers[dataBufId].liveStart;
    bar.liveEnd = loop.buffers[dataBufId].liveEnd;
    bar.defOp = loop.buffers[dataBufId].defOp;
    bar.pairedBufferId = dataBufId;
    loop.buffers[dataBufId].pairedBufferId = barId;
    loop.buffers.push_back(bar);
  }
}

// ============================================================================
// Step 4: SMEM/TMEM Budget Check
// ============================================================================

/// Per design doc §Step 4 (lines 1029-1063): each loop's buffer footprint
/// must fit within the per-tile SMEM/TMEM budgets. This is the per-loop
/// check; Step 4.6 (next diff) adds the kernel-wide cross-region check.
///
/// Sums `sizeBytes() × count` for each buffer (excluding mergeGroupId
/// duplicates — though Phase 1 doesn't merge yet, accept the field for
/// forward compat). BARRIER buffers are charged to SMEM (mbarriers live in
/// SMEM). Returns true if within budget.
static bool checkLoopMemoryBudget(const ttg::ScheduleLoop &loop) {
  int64_t smemBytes = 0;
  int64_t tmemBytes = 0;
  llvm::DenseSet<unsigned> seenGroups;
  for (const auto &buf : loop.buffers) {
    // For merged groups, charge once at max(size×count). Step 4.5 fills
    // mergeGroupId; before then this just walks every buffer individually.
    if (buf.mergeGroupId != UINT_MAX &&
        !seenGroups.insert(buf.mergeGroupId).second)
      continue;
    int64_t bytes = buf.totalBytes();
    if (buf.kind == ttg::MemoryKind::TMEM)
      tmemBytes += bytes;
    else // SMEM and BARRIER both use SMEM space
      smemBytes += bytes;
  }
  bool ok = true;
  if (smemBytes > kSmemBudgetBytes) {
    LDBG("[Step4] SMEM over budget: " << smemBytes << " > "
                                       << kSmemBudgetBytes << " bytes");
    ok = false;
  }
  if (tmemBytes > kTmemBudgetBytes) {
    LDBG("[Step4] TMEM over budget: " << tmemBytes << " > "
                                       << kTmemBudgetBytes << " bytes");
    ok = false;
  }
  if (ok)
    LDBG("[Step4] Budget OK: SMEM=" << smemBytes << "B, TMEM=" << tmemBytes
                                     << "B");
  return ok;
}

/// Top-level: build a ScheduleGraph from DDG + schedule result.
/// Includes Phase 0 (DDG→nodes/edges) and Phase 1 (buffer allocation).
static ttg::ScheduleGraph
buildScheduleGraph(scf::ForOp loop, const ttg::DataDependenceGraph &ddg,
                   const ttg::ModuloScheduleResult &sched,
                   const ttg::LatencyModel &model) {
  ttg::ScheduleGraph graph;
  buildScheduleLoop(loop, ddg, sched, graph, model);

  for (auto &schedLoop : graph.loops) {
    allocateBuffersForLoop(schedLoop);
    // Step 4: per-loop budget check (Step 4.5 merging + Step 4.6 cross-
    // region budget come in the next stack diff).
    checkLoopMemoryBudget(schedLoop);
  }

  return graph;
}

// ============================================================================
// Pass A: Modulo Scheduling
// ============================================================================

/// The main pass.
struct ModuloSchedulePass
    : public PassWrapper<ModuloSchedulePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuloSchedulePass)

  ModuloSchedulePass() = default;
  ModuloSchedulePass(const ModuloSchedulePass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "nvgpu-modulo-schedule"; }

  StringRef getDescription() const override {
    return "Modulo scheduling for warp specialization (Pass A)";
  }

  // Test-only knob: when set, dump the ScheduleGraph to llvm::errs()
  // unconditionally. Used by lit tests in opt builds, where `-debug-only`
  // is unavailable because LLVM_DEBUG is compiled out.
  Option<bool> printScheduleGraph{
      *this, "print-schedule-graph",
      llvm::cl::desc("Dump the ScheduleGraph to stderr unconditionally "
                     "(test-only; bypasses LLVM_DEBUG)"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    auto moduleOp = getOperation();
    ttg::LatencyModel model;


    // Find innermost loops with TMA loads or MMA ops.
    SmallVector<scf::ForOp> innerLoops;
    moduleOp.walk([&](scf::ForOp loop) {
      bool hasInnerLoop = false;
      loop.getBody()->walk([&](scf::ForOp) { hasInnerLoop = true; });
      if (hasInnerLoop)
        return;
      bool hasTMALoad = false;
      bool hasMMAv5 = false;
      loop.getBody()->walk([&](Operation *op) {
        if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
          hasTMALoad = true;
        if (isa<ttng::AsyncTMACopyGlobalToLocalOp>(op))
          hasTMALoad = true;
        if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(op))
          hasMMAv5 = true;
      });
      if (!hasTMALoad && !hasMMAv5)
        return;
      innerLoops.push_back(loop);
    });

    LDBG("Found " << innerLoops.size() << " innermost loop(s)");

    for (auto innerLoop : innerLoops) {
      // Build DDG for this inner loop.
      auto ddg = ttg::DataDependenceGraph::build(innerLoop, model);
      if (ddg.getNumNodes() == 0)
        continue;

      LDBG("DDG: " << ddg.getNumNodes() << " nodes, " << ddg.getEdges().size()
                   << " edges");

      // Run Rau's modulo scheduling.
      auto schedResult = ttg::runModuloScheduling(ddg);
      if (failed(schedResult)) {
        LDBG("Scheduling FAILED");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[PASS-A] Schedule: II=" << schedResult->II << " ResMII="
                 << ddg.computeResMII() << " RecMII=" << ddg.computeRecMII()
                 << " maxStage=" << schedResult->getMaxStage() << "\n");

      // Log per-node schedule.
      LLVM_DEBUG({
        for (const auto &node : ddg.getNodes()) {
          auto it = schedResult->nodeToCycle.find(node.idx);
          if (it == schedResult->nodeToCycle.end())
            continue;
          int cycle = it->second;
          int stage = cycle / schedResult->II;
          llvm::dbgs() << "[PASS-A]   N" << node.idx
                       << "  cycle=" << cycle << "  stage=" << stage
                       << "  " << ttg::getPipelineName(node.pipeline)
                       << "  selfLat=" << node.selfLatency << "  ";
          node.op->print(
              llvm::dbgs(),
              OpPrintingFlags().skipRegions().elideLargeElementsAttrs());
          llvm::dbgs() << "\n";
        }
      });

      // Emit schedule attributes on IR for downstream passes.
      emitScheduleAttributes(innerLoop, ddg, *schedResult);

      // Emit tt.num_stages so downstream pipelining recognises this loop
      // as scheduled. Even single-stage (maxStage=0) loops need the attr
      // present — without it, they're treated as unpipelined and skip
      // latency/buffering behaviour.
      if (!innerLoop->hasAttr(tt::kNumStagesAttrName)) {
        int numStages = schedResult->getMaxStage() + 1;
        auto ctx = innerLoop.getContext();
        innerLoop->setAttr(
            tt::kNumStagesAttrName,
            IntegerAttr::get(IntegerType::get(ctx, 32), numStages));
      }

      // Build ScheduleGraph for analysis/debug.
      auto pipelineGraph =
          buildScheduleGraph(innerLoop, ddg, *schedResult, model);

      LLVM_DEBUG({
        llvm::dbgs()
            << "[PASS-A] === Inner Loop ScheduleGraph ===\n";
        pipelineGraph.dump();
      });
      if (printScheduleGraph) {
        llvm::errs() << "[PASS-A] === Inner Loop ScheduleGraph ===\n";
        pipelineGraph.dump(llvm::errs());
      }

      // Clean up tt.modulo_cycle — internal attr, not needed downstream.
      for (auto &op : innerLoop.getBody()->without_terminator())
        op.removeAttr("tt.modulo_cycle");
    }

    // Step 2: Schedule outer loops (persistent kernels).
    SmallVector<scf::ForOp> outerLoops;
    moduleOp.walk([&](scf::ForOp loop) {
      bool hasInnerLoop = false;
      loop.getBody()->walk([&](scf::ForOp) { hasInnerLoop = true; });
      if (!hasInnerLoop)
        return;
      if (loop->getParentOfType<scf::ForOp>())
        return;
      outerLoops.push_back(loop);
    });

    LDBG("Found " << outerLoops.size() << " outer loop(s)");

    for (auto outerLoop : outerLoops) {
      auto outerDDG = ttg::DataDependenceGraph::build(outerLoop, model);
      if (outerDDG.getNumNodes() == 0)
        continue;

      LDBG("Outer DDG: " << outerDDG.getNumNodes() << " nodes, "
                         << outerDDG.getEdges().size() << " edges");

      auto outerSched = ttg::runModuloScheduling(outerDDG);
      if (failed(outerSched)) {
        LDBG("Outer scheduling FAILED");
        continue;
      }

      LDBG("Outer schedule: II=" << outerSched->II
                                 << " ResMII=" << outerDDG.computeResMII()
                                 << " RecMII=" << outerDDG.computeRecMII()
                                 << " maxStage=" << outerSched->getMaxStage());

      // Log per-node outer DDG schedule.
      LLVM_DEBUG({
        for (const auto &node : outerDDG.getNodes()) {
          auto it = outerSched->nodeToCycle.find(node.idx);
          if (it == outerSched->nodeToCycle.end())
            continue;
          int cycle = it->second;
          int stage = cycle / outerSched->II;
          llvm::dbgs() << "[PASS-A]   N" << node.idx
                       << "  cycle=" << cycle << "  stage=" << stage
                       << "  " << ttg::getPipelineName(node.pipeline)
                       << "  selfLat=" << node.selfLatency << "  "
                       << node.op->getName().getStringRef() << "\n";
        }
      });

      auto outerGraph =
          buildScheduleGraph(outerLoop, outerDDG, *outerSched, model);

      LLVM_DEBUG({
        llvm::dbgs()
            << "[PASS-A] === Outer Loop ScheduleGraph (BEFORE expand) ===\n";
        outerGraph.dump();
      });
      if (printScheduleGraph) {
        llvm::errs()
            << "[PASS-A] === Outer Loop ScheduleGraph (BEFORE expand) ===\n";
        outerGraph.dump(llvm::errs());
      }

      // Emit outer loop schedule attrs for downstream passes.
      emitScheduleAttributes(outerLoop, outerDDG, *outerSched);

      // Clean up tt.modulo_cycle — internal attr, not needed downstream.
      for (auto &op : outerLoop.getBody()->without_terminator())
        op.removeAttr("tt.modulo_cycle");
    }

  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloSchedule() {
  return std::make_unique<ModuloSchedulePass>();
}

void registerNVGPUModuloSchedule() { PassRegistration<ModuloSchedulePass>(); }
} // namespace mlir
