// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Pass A: Modulo Schedule Pass
//
// Builds a DDG from scf.for loop bodies, computes MinII, runs Rau's iterative
// modulo scheduling, and annotates ops with loop.stage, loop.cluster, and
// latency attributes for downstream pipelining passes.
//
// Outer loop wiring for persistent kernels:
//   The pass handles nested loops by modeling inner K-loops as super-nodes
//   in the outer tile loop's DDG. The outer loop schedule sets tt.modulo_ii
//   which survives through assign_latencies (per-loop skip in ScheduleLoops)
//   and through WarpSpecialization (which consumes the loops into partition
//   regions). The epilogue pipelining analysis computes overlap depth for
//   tile-to-tile pipelining.

#include "DataDependenceGraph.h"
#include "LatencyModel.h"
#include "ModuloPipelineIR.h"
#include "ModuloReservationTable.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
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

/// Step 3: Derive per-resource SMEM buffer depths from modulo schedule.
///
/// For each local_alloc (SMEM buffer) in the loop body:
///   1. Find its producer's cycle (tt.modulo_cycle on the load)
///   2. Find the last consumer's end cycle (cycle + latency)
///   3. lifetime = last_consumer_end - producer_cycle
///   4. num_buffers = floor(lifetime / II) + 1
///
/// Then check SMEM budget (228KB): if total exceeds limit, reduce depths
/// starting from the largest resource.
/// Returns the max buffer depth, or 0 if no buffers found.

// Blackwell sm_100 SMEM budget (reserve some for barriers/scratch).
constexpr int kSmemBudgetBytes = 228 * 1024;

static int getMemDescSizeBytes(ttg::MemDescType memDescType) {
  int numElements = 1;
  for (auto dim : memDescType.getShape())
    numElements *= dim;
  return numElements * memDescType.getElementType().getIntOrFloatBitWidth() / 8;
}

static int computeBufferDepths(scf::ForOp loop,
                               const ttg::LatencyModel &model) {
  auto ctx = loop.getContext();
  auto iiAttr = loop->getAttrOfType<IntegerAttr>("tt.modulo_ii");
  if (!iiAttr)
    return 0;
  int II = iiAttr.getInt();
  if (II <= 0)
    return 0;

  struct BufferInfo {
    Operation *allocOp;
    int sizeBytes;
    int numBuffers;
  };
  SmallVector<BufferInfo> buffers;

  for (auto &op : loop.getBody()->without_terminator()) {
    auto alloc = dyn_cast<ttg::LocalAllocOp>(op);
    if (!alloc || !alloc.getSrc())
      continue;

    auto memDescType = dyn_cast<ttg::MemDescType>(alloc.getType());
    if (!memDescType)
      continue;

    // Find producer cycle from the source op.
    auto *producer = alloc.getSrc().getDefiningOp();
    if (!producer)
      continue;
    auto prodCycleAttr =
        producer->getAttrOfType<IntegerAttr>("tt.modulo_cycle");
    if (!prodCycleAttr)
      continue;
    int prodCycle = prodCycleAttr.getInt();

    // Find last consumer end cycle.
    int lastConsumerEnd = prodCycle;
    for (auto *user : alloc->getUsers()) {
      auto uCycleAttr =
          user->getAttrOfType<IntegerAttr>("tt.modulo_cycle");
      if (!uCycleAttr)
        continue;
      auto info = model.getLatency(user);
      lastConsumerEnd = std::max(lastConsumerEnd,
                                 (int)(uCycleAttr.getInt() + info.latency));
    }

    int lifetime = lastConsumerEnd - prodCycle;
    int numBuffers = std::max(lifetime / II + 1, 1);
    int sizeBytes = getMemDescSizeBytes(memDescType);
    buffers.push_back({alloc, sizeBytes, numBuffers});

    LDBG("Buffer: producer_cycle=" << prodCycle
                                   << " last_consumer_end=" << lastConsumerEnd
                                   << " lifetime=" << lifetime << " II=" << II
                                   << " -> num_buffers=" << numBuffers
                                   << " (" << sizeBytes << " bytes)");
  }

  if (buffers.empty())
    return 0;

  // SMEM budget check: reduce depths if total exceeds limit.
  auto computeTotalSmem = [&]() {
    int total = 0;
    for (auto &b : buffers)
      total += b.sizeBytes * b.numBuffers;
    return total;
  };

  while (computeTotalSmem() > kSmemBudgetBytes) {
    int worstIdx = 0, worstCost = 0;
    for (int i = 0; i < (int)buffers.size(); ++i) {
      int cost = buffers[i].sizeBytes * buffers[i].numBuffers;
      if (cost > worstCost) {
        worstCost = cost;
        worstIdx = i;
      }
    }
    if (buffers[worstIdx].numBuffers <= 1)
      break;
    buffers[worstIdx].numBuffers--;
    LDBG("Reduced buffer depth for SMEM budget");
  }

  // Emit buffer.copy on each local_alloc so the WS pass's
  // ChannelPost::getNumBuffers() picks up the per-buffer depth.
  int maxNumBuffers = 1;
  for (auto &b : buffers) {
    b.allocOp->setAttr("tt.num_buffers",
                       IntegerAttr::get(IntegerType::get(ctx, 32), b.numBuffers));
    b.allocOp->setAttr("buffer.copy",
                       IntegerAttr::get(IntegerType::get(ctx, 32), b.numBuffers));
    maxNumBuffers = std::max(maxNumBuffers, b.numBuffers);
  }

  LDBG("Buffer depths: max=" << maxNumBuffers
                              << " totalSmem=" << computeTotalSmem() << "B");
  return maxNumBuffers;
}

// TODO(Phase2/3): Remove emitScheduleAttributes once ModuloExpand and
// ModuloLower consume the PipelineGraph directly. Until then, downstream
// passes (TritonGPUPipeline, WarpSpecialization) need loop.stage attrs.
static void emitScheduleAttributes(scf::ForOp loop,
                                   const ttg::DataDependenceGraph &ddg,
                                   const ttg::ModuloScheduleResult &schedule) {
  const int II = schedule.II;
  const int maxStage = schedule.getMaxStage();
  auto ctx = loop.getContext();
  auto moduleOp = loop->getParentOfType<ModuleOp>();

  // Step 2.5: Compute per-stage cluster IDs from modulo cycles.
  // Ops in the same stage are ordered by cycle: lower cycle → lower cluster ID.
  llvm::DenseMap<int, SmallVector<int>> stageToCycles;
  for (const auto &node : ddg.getNodes()) {
    auto it = schedule.nodeToCycle.find(node.idx);
    if (it == schedule.nodeToCycle.end())
      continue;
    int stage = it->second / II;
    stageToCycles[stage].push_back(it->second);
  }
  llvm::DenseMap<int, llvm::DenseMap<int, int>> stageAndCycleToCluster;
  for (auto &[stage, cycles] : stageToCycles) {
    llvm::sort(cycles);
    cycles.erase(llvm::unique(cycles), cycles.end());
    for (int i = 0, e = cycles.size(); i < e; ++i)
      stageAndCycleToCluster[stage][cycles[i]] = i;
  }

  // Collect selfLatency values for serialization.
  DenseMap<Operation *, int> selfLatencies;

  for (const auto &node : ddg.getNodes()) {
    auto it = schedule.nodeToCycle.find(node.idx);
    if (it == schedule.nodeToCycle.end())
      continue;
    int stage = it->second / II;
    int cycle = it->second;
    int clusterId = stageAndCycleToCluster[stage][cycle];
    node.op->setAttr(tt::kLoopStageAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), stage));
    node.op->setAttr(tt::kLoopClusterAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), clusterId));
    node.op->setAttr("tt.modulo_cycle",
                     IntegerAttr::get(IntegerType::get(ctx, 32), cycle));

    // Record selfLatency for ops that need it (MMAv5, loads with latency).
    // Convert from cycles to stages: lowerMMAs uses this as a stage count,
    // not a cycle count. selfLatency_stages = ceil(cycles / II).
    if (node.selfLatency > 0) {
      int selfLatStages =
          (node.selfLatency + schedule.II - 1) / schedule.II;
      selfLatencies[node.op] = selfLatStages;
    }
  }

  // Serialize selfLatency attrs so lowerMMAs/lowerLoads can read them.
  tt::serializeSelfLatencies(moduleOp, selfLatencies);

  // Ensure ALL ops in the loop body have loop.stage/loop.cluster attrs.
  // Downstream TritonGPUPipeline asserts every op is in the schedule.
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
  if (maxStage > 0) {
    loop->setAttr(tt::kScheduledMaxStageAttrName,
                  IntegerAttr::get(IntegerType::get(ctx, 32), maxStage));
  }

  // === Warp Group Partitioning analysis (informational logging) ===
  {
    llvm::DenseMap<ttg::HWPipeline, int> pipeLoad;
    llvm::DenseMap<ttg::HWPipeline, int> pipeOpCount;
    for (const auto &node : ddg.getNodes()) {
      if (node.pipeline == ttg::HWPipeline::NONE)
        continue;
      pipeLoad[node.pipeline] += node.selfLatency;
      pipeOpCount[node.pipeline]++;
    }

    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Warp group partitioning analysis (II=" << II
                 << "):\n");

    SmallVector<ttg::HWPipeline> ownGroup;
    SmallVector<ttg::HWPipeline> mergeGroup;
    for (auto pipe : {ttg::HWPipeline::MEM, ttg::HWPipeline::TC,
                      ttg::HWPipeline::CUDA, ttg::HWPipeline::SFU}) {
      int load = pipeLoad.lookup(pipe);
      int ops = pipeOpCount.lookup(pipe);
      if (load == 0 && ops == 0)
        continue;
      double util = II > 0 ? static_cast<double>(load) / II : 0.0;
      bool gets_own = util > 0.3;
      LLVM_DEBUG(llvm::dbgs() << "[PASS-A]   " << ttg::getPipelineName(pipe)
                   << ": load=" << load
                   << " util=" << llvm::format("%.1f%%", util * 100)
                   << " ops=" << ops
                   << (gets_own ? " -> OWN warp group" : " -> MERGE") << "\n");
      if (gets_own)
        ownGroup.push_back(pipe);
      else
        mergeGroup.push_back(pipe);
    }

    int numGroups = ownGroup.size();
    bool memHasOwn = false;
    for (auto p : ownGroup)
      if (p == ttg::HWPipeline::MEM)
        memHasOwn = true;
    if (!memHasOwn && pipeLoad.lookup(ttg::HWPipeline::MEM) > 0) {
      numGroups++;
      LLVM_DEBUG(llvm::dbgs() << "[PASS-A]   MEM promoted to own group "
                   << "(TMA producer needs dedicated warp)\n");
    }
    if (!mergeGroup.empty())
      numGroups++;

    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Recommended: " << numGroups << " warp groups");
    if (numGroups >= 2)
      LLVM_DEBUG(llvm::dbgs() << " (");
    bool first = true;
    for (auto p : ownGroup) {
      if (!first)
        LLVM_DEBUG(llvm::dbgs() << " + ");
      LLVM_DEBUG(llvm::dbgs() << ttg::getPipelineName(p));
      first = false;
    }
    if (!memHasOwn && pipeLoad.lookup(ttg::HWPipeline::MEM) > 0) {
      if (!first)
        LLVM_DEBUG(llvm::dbgs() << " + ");
      LLVM_DEBUG(llvm::dbgs() << "MEM(producer)");
      first = false;
    }
    if (!mergeGroup.empty()) {
      if (!first)
        LLVM_DEBUG(llvm::dbgs() << " + ");
      LLVM_DEBUG(llvm::dbgs() << "default(");
      for (unsigned i = 0; i < mergeGroup.size(); i++) {
        if (i > 0)
          LLVM_DEBUG(llvm::dbgs() << "+");
        LLVM_DEBUG(llvm::dbgs() << ttg::getPipelineName(mergeGroup[i]));
      }
      LLVM_DEBUG(llvm::dbgs() << ")");
    }
    if (numGroups >= 2)
      LLVM_DEBUG(llvm::dbgs() << ")");
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
}

// ============================================================================
// Phase 0d: Build PipelineGraph from DDG + Schedule
// ============================================================================

static ttg::PipelineNode
convertDDGNode(const ttg::DDGNode &ddgNode, unsigned pipeNodeId,
               const ttg::ModuloScheduleResult &sched) {
  ttg::PipelineNode pn;
  pn.id = pipeNodeId;
  pn.op = ddgNode.op;
  pn.pipeline = ddgNode.pipeline;
  pn.latency = ddgNode.latency;
  pn.selfLatency = ddgNode.selfLatency;

  auto cycleIt = sched.nodeToCycle.find(ddgNode.idx);
  if (cycleIt != sched.nodeToCycle.end()) {
    pn.cycle = cycleIt->second;
    pn.stage = cycleIt->second / sched.II;
  }

  if (ddgNode.isSuperNode) {
    pn.prologueLatency = ddgNode.prologueLatency;
    // childPipelineId set by caller after child loop is created
  }
  return pn;
}

/// Build a child PipelineLoop for an inner scf.for loop (super-node).
/// Recursively builds DDG, schedules it, and populates nodes/edges.
static unsigned buildChildPipelineLoop(scf::ForOp innerLoop,
                                       ttg::PipelineGraph &graph,
                                       const ttg::LatencyModel &model) {
  auto innerDDG = ttg::DataDependenceGraph::build(innerLoop, model);
  unsigned loopId = graph.addLoop(innerLoop);
  auto &pipeLoop = graph.getLoop(loopId);

  if (innerDDG.getNumNodes() == 0)
    return loopId;

  auto innerSched = ttg::runModuloScheduling(innerDDG);
  if (failed(innerSched))
    return loopId;

  pipeLoop.II = innerSched->II;
  pipeLoop.maxStage = innerSched->getMaxStage();

  // Find prologue latency (earliest TC cycle)
  int tcStart = innerSched->II;
  for (const auto &node : innerDDG.getNodes()) {
    if (node.pipeline == ttg::HWPipeline::TC) {
      auto it = innerSched->nodeToCycle.find(node.idx);
      if (it != innerSched->nodeToCycle.end())
        tcStart = std::min(tcStart, it->second);
    }
  }
  pipeLoop.prologueLatency = tcStart;

  // Extract trip count from scf.for bounds (constant or estimated).
  pipeLoop.tripCount = 4; // default estimate
  pipeLoop.tripCountIsEstimated = true;
  {
    auto lb = innerLoop.getLowerBound()
                  .getDefiningOp<arith::ConstantIntOp>();
    auto ub = innerLoop.getUpperBound()
                  .getDefiningOp<arith::ConstantIntOp>();
    auto step = innerLoop.getStep()
                    .getDefiningOp<arith::ConstantIntOp>();
    if (lb && ub && step && step.value() > 0) {
      int64_t tc =
          (ub.value() - lb.value() + step.value() - 1) / step.value();
      if (tc > 0) {
        pipeLoop.tripCount = static_cast<int>(tc);
        pipeLoop.tripCountIsEstimated = false;
      }
    }
  }

  // Convert DDG nodes → PipelineNodes
  llvm::DenseMap<unsigned, unsigned> ddgToPipe; // ddgIdx → pipeNodeId
  for (const auto &ddgNode : innerDDG.getNodes()) {
    unsigned pipeNodeId = pipeLoop.nodes.size();
    ddgToPipe[ddgNode.idx] = pipeNodeId;
    auto pn = convertDDGNode(ddgNode, pipeNodeId, *innerSched);

    // Handle nested super-nodes (arbitrary depth)
    if (ddgNode.isSuperNode) {
      if (auto nestedLoop = dyn_cast<scf::ForOp>(ddgNode.op)) {
        unsigned childId = buildChildPipelineLoop(nestedLoop, graph, model);
        pn.childPipelineId = childId;
      }
    }

    pipeLoop.nodes.push_back(pn);
    pipeLoop.opToNodeId[ddgNode.op] = pipeNodeId;
  }

  // Convert DDG edges → PipelineEdges
  for (const auto &ddgEdge : innerDDG.getEdges()) {
    auto srcIt = ddgToPipe.find(ddgEdge.srcIdx);
    auto dstIt = ddgToPipe.find(ddgEdge.dstIdx);
    if (srcIt == ddgToPipe.end() || dstIt == ddgToPipe.end())
      continue;
    ttg::PipelineEdge pe;
    pe.srcId = srcIt->second;
    pe.dstId = dstIt->second;
    pe.latency = ddgEdge.latency;
    pe.distance = ddgEdge.distance;
    pipeLoop.edges.push_back(pe);
  }

  // Populate inputs: values from outside the inner loop used by DDG nodes.
  // (1) iter_args (loop-carried values, e.g., accumulator)
  for (auto arg : innerLoop.getRegionIterArgs()) {
    for (auto *user : arg.getUsers()) {
      if (user->hasTrait<OpTrait::IsTerminator>())
        continue;
      auto it = pipeLoop.opToNodeId.find(user);
      if (it != pipeLoop.opToNodeId.end()) {
        ttg::PipelineLoop::MemPort port;
        port.op = user;
        port.isInput = true;
        pipeLoop.inputs.push_back(port);
        break; // one input entry per iter_arg
      }
    }
  }

  // (2) Captured values from outer scope (e.g., TMA descriptors, offsets).
  // Walk all DDG nodes and check if any operand is defined outside the loop.
  llvm::DenseSet<Operation *> capturedDefs;
  for (const auto &node : pipeLoop.nodes) {
    if (!node.op)
      continue;
    // Walk the op (and nested regions for scf.if) to find outer operands
    node.op->walk([&](Operation *nested) {
      for (auto operand : nested->getOperands()) {
        // Skip block arguments of the inner loop (induction var, iter_args)
        if (auto blockArg = dyn_cast<BlockArgument>(operand))
          if (blockArg.getOwner() == innerLoop.getBody())
            continue;
        auto *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        // If defOp is inside the inner loop, it's not captured
        if (innerLoop->isAncestor(defOp))
          continue;
        // Deduplicate by defining op
        if (!capturedDefs.insert(defOp).second)
          continue;
        ttg::PipelineLoop::MemPort port;
        port.op = defOp;
        port.isInput = true;
        pipeLoop.inputs.push_back(port);
      }
    });
  }

  // Populate outputs: values yielded by the loop (via scf.yield)
  auto *yieldOp = innerLoop.getBody()->getTerminator();
  for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
    auto *defOp = yieldOp->getOperand(i).getDefiningOp();
    if (!defOp)
      continue;
    auto it = pipeLoop.opToNodeId.find(defOp);
    if (it != pipeLoop.opToNodeId.end()) {
      ttg::PipelineLoop::MemPort port;
      port.op = defOp;
      port.isInput = false;
      pipeLoop.outputs.push_back(port);
    }
  }

  return loopId;
}

/// Build the top-level PipelineLoop for a scheduled outer/main loop.
static unsigned buildPipelineLoop(scf::ForOp loop,
                                  const ttg::DataDependenceGraph &ddg,
                                  const ttg::ModuloScheduleResult &sched,
                                  ttg::PipelineGraph &graph,
                                  const ttg::LatencyModel &model) {
  unsigned loopId = graph.addLoop(loop);
  auto &pipeLoop = graph.getLoop(loopId);
  pipeLoop.II = sched.II;
  pipeLoop.maxStage = sched.getMaxStage();

  // Find prologue latency (earliest TC cycle)
  int tcStart = sched.II;
  for (const auto &node : ddg.getNodes()) {
    if (node.pipeline == ttg::HWPipeline::TC || node.isSuperNode) {
      auto it = sched.nodeToCycle.find(node.idx);
      if (it != sched.nodeToCycle.end())
        tcStart = std::min(tcStart, it->second);
    }
  }
  pipeLoop.prologueLatency = tcStart;

  // Convert DDG nodes → PipelineNodes
  llvm::DenseMap<unsigned, unsigned> ddgToPipe;
  for (const auto &ddgNode : ddg.getNodes()) {
    unsigned pipeNodeId = pipeLoop.nodes.size();
    ddgToPipe[ddgNode.idx] = pipeNodeId;
    auto pn = convertDDGNode(ddgNode, pipeNodeId, sched);

    // For super-nodes, build child PipelineLoop recursively
    if (ddgNode.isSuperNode) {
      if (auto innerLoop = dyn_cast<scf::ForOp>(ddgNode.op)) {
        unsigned childId = buildChildPipelineLoop(innerLoop, graph, model);
        pn.childPipelineId = childId;
        // Copy prologue latency from child
        pn.prologueLatency = graph.getLoop(childId).prologueLatency;
      }
    }

    pipeLoop.nodes.push_back(pn);
    pipeLoop.opToNodeId[ddgNode.op] = pipeNodeId;
  }

  // Convert DDG edges → PipelineEdges
  for (const auto &ddgEdge : ddg.getEdges()) {
    auto srcIt = ddgToPipe.find(ddgEdge.srcIdx);
    auto dstIt = ddgToPipe.find(ddgEdge.dstIdx);
    if (srcIt == ddgToPipe.end() || dstIt == ddgToPipe.end())
      continue;
    ttg::PipelineEdge pe;
    pe.srcId = srcIt->second;
    pe.dstId = dstIt->second;
    pe.latency = ddgEdge.latency;
    pe.distance = ddgEdge.distance;
    pipeLoop.edges.push_back(pe);
  }

  return loopId;
}

/// Top-level: build a PipelineGraph from DDG + schedule result.
static ttg::PipelineGraph
buildPipelineGraph(scf::ForOp loop, const ttg::DataDependenceGraph &ddg,
                   const ttg::ModuloScheduleResult &sched,
                   const ttg::LatencyModel &model) {
  ttg::PipelineGraph graph;
  buildPipelineLoop(loop, ddg, sched, graph, model);
  return graph;
}

// ============================================================================
// Phase 1: Buffer Allocation
// ============================================================================

static ttg::MemoryKind classifyMemoryKind(Operation *op) {
  if (isa<ttng::TMEMAllocOp>(op))
    return ttg::MemoryKind::TMEM;
  if (isa<ttg::LocalAllocOp>(op))
    return ttg::MemoryKind::SMEM;
  return ttg::MemoryKind::Register;
}

static void extractBufferShape(Operation *op, ttg::PipelineBuffer &buf) {
  Type resultType;
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
    resultType = alloc.getType();
  else if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(op))
    resultType = tmemAlloc.getType();
  else if (op->getNumResults() > 0)
    resultType = op->getResult(0).getType();

  auto extractFromShapedType = [&](llvm::ArrayRef<int64_t> shape, Type elemTy) {
    // Guard against dynamic/unknown dimensions
    for (auto dim : shape) {
      if (dim <= 0 || ShapedType::isDynamic(dim))
        return; // leave buf.shape empty → caller will skip budget
    }
    // Guard against non-int/float element types
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

/// Compute the maximum "effective consumer stage" for a producer node.
/// For distance-0 edges: effective = consumer.stage
/// For distance-d edges: effective = consumer.stage + d * numStages
/// This ensures buffer lifetime covers cross-iteration dependencies.
/// Returns the max effective consumer stage across all outgoing edges.
static int findMaxEffectiveConsumerStage(const ttg::PipelineLoop &loop,
                                         unsigned nodeId) {
  int maxStage = loop.getNode(nodeId).stage;
  const int numStages = loop.numStages();
  for (const auto &edge : loop.edges) {
    if (edge.srcId == nodeId) {
      int effective =
          loop.getNode(edge.dstId).stage + edge.distance * numStages;
      maxStage = std::max(maxStage, effective);
    }
  }
  return maxStage;
}

static void allocateBuffersForLoop(ttg::PipelineLoop &loop) {
  // First pass: allocate data buffers (SMEM/TMEM)
  llvm::SmallVector<unsigned, 4> dataBufferIds;
  for (auto &node : loop.nodes) {
    if (!node.op)
      continue;

    auto kind = classifyMemoryKind(node.op);
    if (kind == ttg::MemoryKind::Register)
      continue;

    unsigned bufId = loop.buffers.size();
    ttg::PipelineBuffer buf;
    buf.id = bufId;
    buf.kind = kind;
    buf.defOp = node.op;
    extractBufferShape(node.op, buf);

    // Skip budget accounting for buffers with invalid shapes
    if (buf.shape.empty() || buf.elementBitWidth == 0) {
      LLVM_DEBUG(llvm::dbgs() << "[Phase1] WARNING: skipped budget for buf" << bufId
                   << ": unknown shape/type\n");
    }

    int effectiveConsumer = findMaxEffectiveConsumerStage(loop, node.id);
    int stageDiff = effectiveConsumer - node.stage;
    buf.count = static_cast<unsigned>(std::max(stageDiff + 1, 1));

    loop.buffers.push_back(buf);
    node.producesBuffer = bufId;

    // Track data buffers that need barriers (multi-buffered only)
    if (buf.count > 1)
      dataBufferIds.push_back(bufId);

    // Mark all consumers (including distance>0) — deduplicate by dstId.
    llvm::DenseSet<unsigned> markedConsumers;
    for (const auto &edge : loop.edges) {
      if (edge.srcId == node.id && markedConsumers.insert(edge.dstId).second)
        loop.nodes[edge.dstId].consumesBuffers.push_back(bufId);
    }
  }

  // Second pass: allocate a BARRIER buffer for each multi-buffered data buffer.
  // Each barrier has the same count as its paired data buffer.
  for (unsigned dataBufId : dataBufferIds) {
    unsigned barId = loop.buffers.size();
    ttg::PipelineBuffer bar;
    bar.id = barId;
    bar.kind = ttg::MemoryKind::BARRIER;
    bar.count = loop.buffers[dataBufId].count;
    bar.defOp = loop.buffers[dataBufId].defOp;
    bar.pairedBufferId = dataBufId;
    // Barriers have no shape/elementBitWidth — sizeBytes() returns 8 for BARRIER
    loop.buffers[dataBufId].pairedBufferId = barId;
    loop.buffers.push_back(bar);
  }
}

static bool checkSmemBudget(const ttg::PipelineGraph &graph) {
  constexpr int64_t kSmemBudget = 232 * 1024; // Blackwell B200
  int64_t totalSmem = 0;
  for (const auto &loop : graph.loops) {
    for (const auto &buf : loop.buffers) {
      if (buf.kind != ttg::MemoryKind::SMEM &&
          buf.kind != ttg::MemoryKind::BARRIER)
        continue;
      // Skip buffers with unknown shape (already warned in allocation)
      // Barriers have no shape but sizeBytes() handles them correctly
      if (buf.kind != ttg::MemoryKind::BARRIER &&
          (buf.shape.empty() || buf.elementBitWidth == 0))
        continue;
      totalSmem += buf.sizeBytes() * buf.count;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "[Phase1] SMEM budget: " << totalSmem << " / " << kSmemBudget
               << " bytes"
               << (totalSmem > kSmemBudget ? " EXCEEDED\n" : " OK\n"));
  return totalSmem <= kSmemBudget;
}

static bool checkTmemBudget(const ttg::PipelineGraph &graph) {
  constexpr int64_t kTmemBudget = 128 * 512 * 4;
  int64_t totalTmem = 0;
  for (const auto &loop : graph.loops) {
    for (const auto &buf : loop.buffers) {
      if (buf.kind != ttg::MemoryKind::TMEM)
        continue;
      if (buf.shape.empty() || buf.elementBitWidth == 0)
        continue;
      totalTmem += buf.sizeBytes() * buf.count;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "[Phase1] TMEM budget: " << totalTmem << " / " << kTmemBudget
               << " bytes"
               << (totalTmem > kTmemBudget ? " EXCEEDED\n" : " OK\n"));
  return totalTmem <= kTmemBudget;
}

/// Dump Phase 1 buffer declarations per loop, matching examples.txt format.
static void dumpPhase1Buffers(const ttg::PipelineGraph &graph) {
  for (unsigned loopId = 0; loopId < graph.loops.size(); ++loopId) {
    const auto &loop = graph.loops[loopId];
    if (loop.buffers.empty())
      continue;
    LLVM_DEBUG(llvm::dbgs() << "[Phase1] Loop " << loopId << " (II=" << loop.II
                 << " maxStage=" << loop.maxStage << "): "
                 << loop.buffers.size() << " buffers\n");
    for (const auto &buf : loop.buffers) {
      const char *kindStr = buf.kind == ttg::MemoryKind::SMEM      ? "SMEM"
                            : buf.kind == ttg::MemoryKind::TMEM    ? "TMEM"
                            : buf.kind == ttg::MemoryKind::BARRIER ? "BARRIER"
                                                                    : "Reg";

      if (buf.kind == ttg::MemoryKind::BARRIER) {
        // Barrier buffers: compact format matching examples.txt
        // e.g., %bar_ld = modulo.alloc BARRIER [2]
        LLVM_DEBUG(llvm::dbgs() << "[Phase1]   buf" << buf.id << " = alloc BARRIER ["
                     << buf.count << "]");
        if (buf.pairedBufferId != UINT_MAX)
          LLVM_DEBUG(llvm::dbgs() << "  // for buf" << buf.pairedBufferId);
        LLVM_DEBUG(llvm::dbgs() << ", " << buf.sizeBytes() * buf.count << " bytes total\n");
        continue;
      }

      // Find producer node for this buffer
      int producerStage = -1;
      int effectiveConsumer = -1;
      for (const auto &node : loop.nodes) {
        if (node.producesBuffer == buf.id) {
          producerStage = node.stage;
          effectiveConsumer = findMaxEffectiveConsumerStage(loop, node.id);
          break;
        }
      }
      int stageDiff =
          (producerStage >= 0 && effectiveConsumer >= 0)
              ? effectiveConsumer - producerStage
              : 0;

      LLVM_DEBUG(llvm::dbgs() << "[Phase1]   buf" << buf.id << " = alloc " << kindStr
                   << " [" << buf.count << "x");
      for (size_t i = 0; i < buf.shape.size(); ++i) {
        if (i > 0)
          LLVM_DEBUG(llvm::dbgs() << "x");
        LLVM_DEBUG(llvm::dbgs() << buf.shape[i]);
      }
      if (buf.elementBitWidth > 0) {
        LLVM_DEBUG(llvm::dbgs() << " x " << (buf.elementBitWidth <= 16 ? "f16" : "f32"));
      }
      LLVM_DEBUG(llvm::dbgs() << "]");
      LLVM_DEBUG(llvm::dbgs() << "  // stageDiff=" << stageDiff << " -> "
                   << buf.count << " bufs");
      if (buf.shape.size() > 0 && buf.elementBitWidth > 0)
        LLVM_DEBUG(llvm::dbgs() << ", " << buf.sizeBytes() * buf.count << " bytes total");
      LLVM_DEBUG(llvm::dbgs() << "\n");
    }
  }
}

/// Allocate multi-buffers for all loops. Returns false if budget exceeded.
static bool allocateBuffers(ttg::PipelineGraph &graph) {
  LLVM_DEBUG(llvm::dbgs() << "[Phase1] === Buffer Allocation ===\n");
  for (auto &loop : graph.loops)
    allocateBuffersForLoop(loop);

  // Dump detailed buffer info (compare against Phase 1 in examples.txt)
  dumpPhase1Buffers(graph);

  bool smemOk = checkSmemBudget(graph);
  bool tmemOk = checkTmemBudget(graph);
  if (!smemOk)
    LLVM_DEBUG(llvm::dbgs() << "[Phase1] ERROR: SMEM budget exceeded — reduce "
                 << "buffer depths or tile size\n");
  if (!tmemOk)
    LLVM_DEBUG(llvm::dbgs() << "[Phase1] ERROR: TMEM budget exceeded — reduce "
                 << "accumulator size or disable multi-buffering\n");
  return smemOk && tmemOk;
}

// ============================================================================
// Phase 2: Loop Expansion Plan (Software Pipelining Analysis)
// ============================================================================

static void logExpansionPlanForLoop(const ttg::PipelineLoop &loop,
                                    unsigned loopId,
                                    const ttg::PipelineGraph &graph) {
  if (loop.maxStage == 0)
    return;

  const int prologueIters = loop.maxStage;
  const int epilogueIters = loop.maxStage;

  LLVM_DEBUG(llvm::dbgs() << "[Phase2] Loop " << loopId << ": maxStage=" << loop.maxStage
               << " II=" << loop.II << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[Phase2]   Prologue: " << prologueIters
               << " iterations (stages 0.." << loop.maxStage - 1 << ")\n");

  for (int iter = 0; iter < prologueIters; ++iter) {
    LLVM_DEBUG(llvm::dbgs() << "[Phase2]     Iter " << iter << ": stages [0.." << iter
                 << "] — ");
    int opCount = 0;
    for (const auto &node : loop.nodes) {
      if (node.stage <= iter) {
        ++opCount;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << opCount << " ops, bufIdx=" << iter << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "[Phase2]   Kernel loop: all " << loop.maxStage + 1
               << " stages active\n");
  LLVM_DEBUG(llvm::dbgs() << "[Phase2]     Stage-to-buffer mapping:\n");
  for (const auto &buf : loop.buffers) {
    LLVM_DEBUG(llvm::dbgs() << "[Phase2]       buf" << buf.id << " ("
                 << (buf.kind == ttg::MemoryKind::SMEM   ? "SMEM"
                     : buf.kind == ttg::MemoryKind::TMEM ? "TMEM"
                                                         : "Reg")
                 << "): count=" << buf.count << " index=iter%" << buf.count
                 << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "[Phase2]   Epilogue: " << epilogueIters
               << " iterations (drain stages " << loop.maxStage << "..0)\n");

  for (int iter = 0; iter < epilogueIters; ++iter) {
    int minStage = iter + 1;
    LLVM_DEBUG(llvm::dbgs() << "[Phase2]     Drain " << iter << ": stages [" << minStage
                 << ".." << loop.maxStage << "] — ");
    int opCount = 0;
    for (const auto &node : loop.nodes) {
      if (node.stage >= minStage)
        ++opCount;
    }
    LLVM_DEBUG(llvm::dbgs() << opCount << " ops\n");
  }

  // Log super-node expansion order
  for (const auto &node : loop.nodes) {
    if (!node.isSuperNode())
      continue;
    const auto &child = graph.getLoop(node.childPipelineId);
    LLVM_DEBUG(llvm::dbgs() << "[Phase2]   Super-node N" << node.id << " → child loop "
                 << node.childPipelineId << " (II=" << child.II
                 << " maxStage=" << child.maxStage
                 << " prologueLat=" << child.prologueLatency << ")\n");
  }
}

static void logExpansionPlan(const ttg::PipelineGraph &graph) {
  LLVM_DEBUG(llvm::dbgs() << "[Phase2] === Loop Expansion Plan ===\n");

  auto order = graph.getBottomUpOrder();
  LLVM_DEBUG(llvm::dbgs() << "[Phase2] Processing order (bottom-up):");
  for (auto id : order)
    LLVM_DEBUG(llvm::dbgs() << " loop" << id);
  LLVM_DEBUG(llvm::dbgs() << "\n");

  for (auto id : order)
    logExpansionPlanForLoop(graph.getLoop(id), id, graph);
}

// ============================================================================
// Pass A: Modulo Scheduling
// ============================================================================

// ============================================================================
// Pipeline Graph Expansion
// ============================================================================

/// Expand a PipelineGraph: PipelineGraph in → IR transform → PipelineGraph out.
/// 1. Extract {op, stage} schedule from the input graph
/// 2. Call pipelineForLoop() to do the mechanical IR rewrite
/// 3. Build a new PipelineGraph from the expanded loop
/// Returns the new PipelineGraph and the new scf::ForOp, or failure.
static FailureOr<std::pair<ttg::PipelineGraph, scf::ForOp>>
expandPipelineGraph(const ttg::PipelineGraph &inputGraph,
                    scf::ForOp forOp, const ttg::LatencyModel &model) {
  auto &pipeLoop = inputGraph.getLoop(0);
  if (pipeLoop.numStages() <= 1)
    return failure(); // nothing to expand

  // Step 1: Build schedule from loop.stage attrs on the IR (post-lowering).
  // This ensures we pick up any ops added by lowerLoops (async copies,
  // barriers) with their correct stage assignments.
  triton::CoarseSchedule coarseSched;
  if (failed(coarseSched.deSerialize(forOp)))
    return failure(); // no schedule on this loop

  std::vector<std::pair<Operation *, unsigned>> schedule =
      coarseSched.createFinalSchedule(forOp);

  // Step 2: Call pipelineForLoop (reuse existing expander).
  triton::PipeliningOption options;
  options.supportDynamicLoops = true;
  options.peelEpilogue = false;
  options.predicateFn = triton::wrapInMaskOp;
  options.getScheduleFn =
      [&](scf::ForOp,
          std::vector<std::pair<Operation *, unsigned>> &sched) {
        sched = schedule;
      };

  IRRewriter rewriter(forOp);
  auto newForOp = triton::pipelineForLoop(rewriter, forOp, options);
  if (failed(newForOp))
    return failure();

  // Step 3: Build new PipelineGraph from expanded loop.
  auto expandedDDG = ttg::DataDependenceGraph::build(*newForOp, model);
  if (expandedDDG.getNumNodes() == 0) {
    // Expanded loop has no pipeline-relevant ops — return trivial graph.
    ttg::PipelineGraph result;
    result.addLoop(*newForOp);
    return std::make_pair(std::move(result), *newForOp);
  }

  auto expandedSched = ttg::runModuloScheduling(expandedDDG);
  if (failed(expandedSched)) {
    // Single-stage after expansion is expected — build trivial graph.
    ttg::PipelineGraph result;
    auto loopId = result.addLoop(*newForOp);
    auto &loop = result.getLoop(loopId);
    loop.II = 0;
    loop.maxStage = 0;
    // Add all DDG nodes at stage 0
    for (const auto &node : expandedDDG.getNodes()) {
      ttg::PipelineNode pn;
      pn.id = loop.nodes.size();
      pn.op = node.op;
      pn.pipeline = node.pipeline;
      pn.latency = node.latency;
      pn.selfLatency = node.selfLatency;
      pn.stage = 0;
      pn.cycle = 0;
      loop.nodes.push_back(pn);
    }
    return std::make_pair(std::move(result), *newForOp);
  }

  auto expandedGraph =
      buildPipelineGraph(*newForOp, expandedDDG, *expandedSched, model);

  // Step 4: Populate prologue/epilogue nodes by walking the parent block.
  // After pipelineForLoop, the parent block has:
  //   [original ops before inner loop]
  //   [prologue ops inserted by expander (ttg.mask, arith, etc.)]
  //   [new scf.for]
  //   [epilogue ops (if peelEpilogue=true, empty otherwise)]
  //   [original ops after inner loop]
  // We identify prologue ops as those between the last "original" op
  // before the loop and the loop itself. Since we used predicateFn,
  // the prologue ops are ttg.mask + supporting arith ops.
  auto &resultLoop = expandedGraph.getLoop(0);
  resultLoop.isExpanded = true;

  if (auto *parentBlock = (*newForOp)->getBlock()) {
    // Find prologue: ops before the new scf.for that are ttg.mask or
    // were not in the original outer body (i.e., inserted by expander).
    // Walk backwards from the scf.for to find prologue ops.
    auto *forOpPtr = (*newForOp).getOperation();
    for (auto it = Block::iterator(forOpPtr); it != parentBlock->begin();) {
      --it;
      Operation *op = &*it;
      // Stop when we hit an op that's clearly from the outer loop
      // (tmem_alloc, tmem_store, divsi, etc.)
      if (isa<ttng::TMEMAllocOp, ttng::TMEMStoreOp>(op))
        break;
      // ttg.mask and supporting arith ops are prologue
      ttg::PipelineNode pn;
      pn.id = resultLoop.prologueNodes.size();
      pn.op = op;
      auto info = model.getLatency(op);
      pn.pipeline = info.pipeline;
      pn.latency = info.latency;
      pn.selfLatency = info.selfLatency;
      resultLoop.prologueNodes.push_back(pn);
    }
    // Reverse since we walked backwards.
    std::reverse(resultLoop.prologueNodes.begin(),
                 resultLoop.prologueNodes.end());
  }

  return std::make_pair(std::move(expandedGraph), *newForOp);
}

/// The main pass.
struct ModuloSchedulePass
    : public PassWrapper<ModuloSchedulePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuloSchedulePass)

  StringRef getArgument() const override { return "nvgpu-modulo-schedule"; }

  StringRef getDescription() const override {
    return "Modulo scheduling for warp specialization (Pass A)";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    ttg::LatencyModel model;

    // Step 1: Find innermost loops (K-loops) and schedule them.
    SmallVector<scf::ForOp> innerLoops;
    moduleOp.walk([&](scf::ForOp loop) {
      // Only innermost loops (no nested scf.for inside).
      bool hasInnerLoop = false;
      loop.getBody()->walk([&](scf::ForOp) { hasInnerLoop = true; });
      if (hasInnerLoop)
        return;
      // Must have TMA loads or MMA.
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

    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] ========================================\n");
    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Step 1: Schedule INNER loops (bottom-up)\n");
    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] ========================================\n");
    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Found " << innerLoops.size()
                 << " innermost loop(s)\n");

    for (auto innerLoop : innerLoops) {
      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] --- Inner loop ---\n");

      // Build DDG for this inner loop.
      auto ddg = ttg::DataDependenceGraph::build(innerLoop, model);
      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] DDG: " << ddg.getNumNodes() << " nodes, "
                   << ddg.getEdges().size() << " edges\n");
      if (ddg.getNumNodes() == 0)
        continue;

      // Run Rau's modulo scheduling.
      auto schedResult = ttg::runModuloScheduling(ddg);
      if (failed(schedResult)) {
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Scheduling FAILED\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Schedule: II=" << schedResult->II
                   << " ResMII=" << ddg.computeResMII()
                   << " RecMII=" << ddg.computeRecMII()
                   << " maxStage=" << schedResult->getMaxStage() << "\n");

      // Log per-node schedule.
      for (const auto &node : ddg.getNodes()) {
        auto it = schedResult->nodeToCycle.find(node.idx);
        if (it == schedResult->nodeToCycle.end())
          continue;
        int cycle = it->second;
        int stage = cycle / schedResult->II;
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A]   N" << node.idx << "  cycle=" << cycle
                     << "  stage=" << stage << "  "
                     << ttg::getPipelineName(node.pipeline)
                     << "  selfLat=" << node.selfLatency << "  ");
        node.op->print(LLVM_DEBUG(llvm::dbgs(),
                       OpPrintingFlags().skipRegions().elideLargeElementsAttrs()));
        LLVM_DEBUG(llvm::dbgs() << "\n");
      }

      // Build PipelineGraph for this inner loop.
      auto pipelineGraph =
          buildPipelineGraph(innerLoop, ddg, *schedResult, model);

      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] === Inner Loop PipelineGraph (BEFORE expand) ===\n");
      pipelineGraph.dump();

      // Serialize modulo schedule as loop.stage/loop.cluster attrs on ops.
      // lowerLoops needs these to compute buffer depths and insert async ops.
      // Derive stage from modulo schedule lifetime analysis:
      //   bufferDepth = max consumer end-cycle / II
      // For loads: stage 0 (prefetched).
      // For non-loads: stage = max(consumer end-cycle) / II, at least 1.
      // This ensures lowerLoads sees the correct stageDiff for buffer depth.
      {
        triton::CoarseSchedule coarseSched;
        auto cluster = coarseSched.clusters.newAtBack();

        // First pass: compute max end-cycle across all consumers (MMA).
        int maxEndCycle = 0;
        for (const auto &node : ddg.getNodes()) {
          auto it = schedResult->nodeToCycle.find(node.idx);
          if (it == schedResult->nodeToCycle.end())
            continue;
          int endCycle = it->second + node.selfLatency;
          maxEndCycle = std::max(maxEndCycle, endCycle);
        }
        int consumerStage = std::max(1, maxEndCycle / schedResult->II);

        // Second pass: assign stages.
        for (const auto &node : ddg.getNodes()) {
          auto it = schedResult->nodeToCycle.find(node.idx);
          if (it == schedResult->nodeToCycle.end() || !node.op)
            continue;
          int stage;
          if (isa<triton::DescriptorLoadOp, triton::LoadOp>(node.op)) {
            stage = 0; // loads are prefetched
          } else {
            // All non-load ops get the consumer stage so that
            // getDefUseStageDiff sees the full buffer lifetime.
            stage = consumerStage;
          }
          coarseSched.insert(node.op, stage, cluster);
        }
        coarseSched.serialize(innerLoop);
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Serialized CoarseSchedule ("
                     << coarseSched.getNumStages() << " stages)\n");
        // Verify: check loop.stage on ops
        int opsWithStage = 0;
        innerLoop.getBody()->walk([&](Operation *op) {
          if (op->hasAttr("loop.stage"))
            opsWithStage++;
        });
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] ops with loop.stage: " << opsWithStage
                     << ", scheduled_max_stage="
                     << (innerLoop->hasAttr("tt.scheduled_max_stage") ? "yes" : "no")
                     << "\n");
      }

      // Step 1: lowerLoops — convert descriptor_load → async TMA, allocate
      // SMEM buffers. Must run before expand while loop.stage attrs are valid.
      // Note: lowerLoops may replace the ForOp, so re-find it afterward.
      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] lowerLoops (inner)\n");
      ttg::lowerLoops(moduleOp);

      // Debug: check if descriptor_load ops survive after lowerLoops
      {
        int descLoadCount = 0;
        moduleOp.walk([&](triton::DescriptorLoadOp) { descLoadCount++; });
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] descriptor_load ops remaining: "
                     << descLoadCount << "\n");
      }

      // Re-find the innermost loop (lowerLoops may have replaced the ForOp).
      innerLoop = scf::ForOp();
      moduleOp.walk([&](scf::ForOp loop) {
        bool hasChild = false;
        loop.getBody()->walk([&](scf::ForOp) { hasChild = true; });
        if (!hasChild)
          innerLoop = loop;
      });
      if (!innerLoop) {
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Lost inner loop after lowerLoops\n");
        continue;
      }

      // Step 2: Expand — prologue/kernel/epilogue via pipelineForLoop.
      auto expandResult = expandPipelineGraph(pipelineGraph, innerLoop, model);
      if (failed(expandResult)) {
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Expansion failed or not needed\n");
        continue;
      }
      auto &[expandedGraph, newForOp] = *expandResult;

      // Mark expanded loop so ScheduleLoops (inside AutoWS) skips it.
      auto builder = OpBuilder(newForOp);
      newForOp->setAttr("tt.modulo_ii",
                        builder.getI32IntegerAttr(schedResult->II));

      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] === Inner Loop PipelineGraph (AFTER expand) ===\n");
      expandedGraph.dump();

      // Clean stale inner loop scheduling attrs from the expanded IR.
      // Keep tt.modulo_ii so ScheduleLoops (inside AutoWS) skips these loops.
      if (auto parentLoop = newForOp.getOperation()->getParentOfType<scf::ForOp>()) {
        parentLoop.getBody()->walk([](Operation *op) {
          op->removeAttr("loop.stage");
          op->removeAttr("loop.cluster");
          op->removeAttr("tt.modulo_cycle");
          op->removeAttr("tt.self_latency");
        });
        parentLoop->removeAttr("tt.scheduled_max_stage");
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Cleaned stale inner loop attrs from outer body\n");
      }

      // Clean up tt.modulo_cycle — internal attr used only to pass cycle
      // info from emitScheduleAttributes to computeBufferDepths.
      for (auto &op : innerLoop.getBody()->without_terminator())
        op.removeAttr("tt.modulo_cycle");
    }

    // Step 2: Schedule OUTER loop (now with expanded inner loop).
    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] ========================================\n");
    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Step 2: Schedule OUTER loop\n");
    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] ========================================\n");

    // Find the outer loop (has an inner scf.for but is not itself nested).
    SmallVector<scf::ForOp> outerLoops;
    moduleOp.walk([&](scf::ForOp loop) {
      // Must have an inner scf.for child.
      bool hasInnerLoop = false;
      loop.getBody()->walk([&](scf::ForOp) { hasInnerLoop = true; });
      if (!hasInnerLoop)
        return;
      // Must not be inside another scf.for (top-level).
      if (loop->getParentOfType<scf::ForOp>())
        return;
      outerLoops.push_back(loop);
    });

    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Found " << outerLoops.size()
                 << " outer loop(s)\n");

    for (auto outerLoop : outerLoops) {
      // Build DDG for outer loop. Inner K-loop is now expanded, so
      // its prologue ops (ttg.mask wrapping descriptor_load) are visible.
      auto outerDDG = ttg::DataDependenceGraph::build(outerLoop, model);
      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Outer DDG: " << outerDDG.getNumNodes()
                   << " nodes, " << outerDDG.getEdges().size() << " edges\n");
      if (outerDDG.getNumNodes() == 0)
        continue;

      auto outerSched = ttg::runModuloScheduling(outerDDG);
      if (failed(outerSched)) {
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Outer scheduling FAILED\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Outer schedule: II=" << outerSched->II
                   << " ResMII=" << outerDDG.computeResMII()
                   << " RecMII=" << outerDDG.computeRecMII()
                   << " maxStage=" << outerSched->getMaxStage() << "\n");

      // Log per-node outer DDG schedule.
      for (const auto &node : outerDDG.getNodes()) {
        auto it = outerSched->nodeToCycle.find(node.idx);
        if (it == outerSched->nodeToCycle.end())
          continue;
        int cycle = it->second;
        int stage = cycle / outerSched->II;
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A]   N" << node.idx << "  cycle=" << cycle
                     << "  stage=" << stage << "  "
                     << ttg::getPipelineName(node.pipeline)
                     << "  selfLat=" << node.selfLatency << "  "
                     << node.op->getName().getStringRef() << "\n");
      }

      auto outerGraph =
          buildPipelineGraph(outerLoop, outerDDG, *outerSched, model);

      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] === Outer Loop PipelineGraph (BEFORE expand) ===\n");
      outerGraph.dump();

      // Serialize outer schedule — same lifetime-based stage assignment.
      {
        triton::CoarseSchedule coarseSched;
        auto cluster = coarseSched.clusters.newAtBack();

        int maxEndCycle = 0;
        for (const auto &node : outerDDG.getNodes()) {
          auto it = outerSched->nodeToCycle.find(node.idx);
          if (it == outerSched->nodeToCycle.end())
            continue;
          int endCycle = it->second + node.selfLatency;
          maxEndCycle = std::max(maxEndCycle, endCycle);
        }
        int consumerStage = std::max(1, maxEndCycle / outerSched->II);

        for (const auto &node : outerDDG.getNodes()) {
          auto it = outerSched->nodeToCycle.find(node.idx);
          if (it == outerSched->nodeToCycle.end() || !node.op)
            continue;
          int stage;
          if (isa<triton::DescriptorLoadOp, triton::LoadOp>(node.op)) {
            stage = 0;
          } else {
            stage = consumerStage;
          }
          coarseSched.insert(node.op, stage, cluster);
        }
        coarseSched.serialize(outerLoop);
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Serialized outer CoarseSchedule ("
                     << coarseSched.getNumStages() << " stages)\n");
      }

      // lowerLoops for outer loop (buffer alloc, async TMA).
      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] lowerLoops (outer)\n");
      ttg::lowerLoops(moduleOp);

      // Re-find the outer loop (lowerLoops may have replaced the ForOp).
      outerLoop = scf::ForOp();
      moduleOp.walk([&](scf::ForOp loop) {
        bool hasChild = false;
        loop.getBody()->walk([&](scf::ForOp) { hasChild = true; });
        if (hasChild && !loop->getParentOfType<scf::ForOp>())
          outerLoop = loop;
      });
      if (!outerLoop) {
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Lost outer loop after lowerLoops\n");
        continue;
      }

      // Expand the outer loop: PipelineGraph in → IR transform → PipelineGraph out.
      auto outerExpandResult =
          expandPipelineGraph(outerGraph, outerLoop, model);
      if (failed(outerExpandResult)) {
        LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Outer expansion failed or not needed\n");
        continue;
      }
      auto &[expandedOuterGraph, newOuterForOp] = *outerExpandResult;

      // Mark expanded outer loop so ScheduleLoops skips it.
      {
        auto builder = OpBuilder(newOuterForOp);
        newOuterForOp->setAttr("tt.modulo_ii",
                               builder.getI32IntegerAttr(outerSched->II));
      }

      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] === Outer Loop PipelineGraph (AFTER expand) ===\n");
      expandedOuterGraph.dump();

      // Dump TTGIR after full two-loop expansion.
      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] === TTGIR after outer loop expansion ===\n");
      moduleOp.print(LLVM_DEBUG(llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "\n");

      // Clean stale outer loop scheduling attrs from the expanded IR.
      // Walk the entire module because prologue ops are outside the for loop.
      // Keep tt.modulo_ii so ScheduleLoops (inside AutoWS) skips these loops.
      moduleOp->walk([](Operation *op) {
        op->removeAttr("loop.stage");
        op->removeAttr("loop.cluster");
        op->removeAttr("tt.modulo_cycle");
        op->removeAttr("tt.self_latency");
        op->removeAttr("tt.scheduled_max_stage");
      });
      LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Cleaned stale outer loop attrs\n");
    }

    // Final cleanup: remove any residual scheduling attrs from the module.
    // This ensures add_pipeline's lowerLoops/expandLoops are no-ops.
    moduleOp->walk([](Operation *op) {
      op->removeAttr("loop.stage");
      op->removeAttr("loop.cluster");
      op->removeAttr("tt.modulo_cycle");
      op->removeAttr("tt.self_latency");
      op->removeAttr("tt.scheduled_max_stage");
    });

    // Resolve ttg.mask ops created by pipelineForLoop expansion.
    DenseSet<ttg::MaskOp> emptySet;
    triton::resolveMaskOp(moduleOp);
    LLVM_DEBUG(llvm::dbgs() << "[PASS-A] Resolved mask ops\n");
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloSchedule() {
  return std::make_unique<ModuloSchedulePass>();
}
void registerNVGPUModuloSchedule() {
  PassRegistration<ModuloSchedulePass>();
}
} // namespace mlir
