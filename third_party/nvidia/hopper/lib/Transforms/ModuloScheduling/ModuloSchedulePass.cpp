// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Pass A: Modulo Schedule Pass
//
// Builds a DDG from scf.for loop bodies, computes MinII, runs Rau's iterative
// modulo scheduling, and annotates ops with loop.stage and loop.cluster
// attributes for downstream pipelining passes.

#include <cmath>

#include "DataDependenceGraph.h"
#include "LatencyModel.h"
#include "ModuloReservationTable.h"
#include "ModuloScheduleGraph.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/hopper/lib/Transforms/WarpSpecialization/CodePartitionUtility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"

namespace mlir {
int doTaskIdPropagate(triton::FuncOp &funcOp);
} // namespace mlir
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include <queue>

#include <limits>

#define DEBUG_TYPE "nvgpu-modulo-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

namespace {

// ============================================================================
// Emit all schedule annotations from the ScheduleGraph
// ============================================================================
//
// Single consolidated function that emits ALL modulo schedule annotations
// onto the IR. Everything is derived from the ScheduleGraph (which contains
// the DDG schedule + buffer allocations + budget adjustments).
//
// Per-op attrs:
//   loop.stage   — pipeline stage (cycle / II)
//   loop.cluster — within-stage ordering (dense cluster ID from cycle order)
//
// Per-loop attrs:
//   tt.modulo_ii            — initiation interval
//   tt.scheduled_max_stage  — maximum stage across all ops
//
// Per-buffer attrs (on producing local_alloc / tmem_alloc ops):
//   tt.num_buffers — buffer depth (copies needed for lifetime coverage)
//   buffer.id      — unique buffer index within the ScheduleGraph
//
static void emitScheduleFromGraph(scf::ForOp loop,
                                  const ttg::ScheduleGraph &graph,
                                  const ttg::DataDependenceGraph &ddg) {
  const auto &schedLoop = graph.getLoop(0);
  const int II = schedLoop.II;
  auto ctx = loop.getContext();

  // ── 1. Per-op: loop.stage / loop.cluster ──
  // Derive stage from cycle/II. Derive cluster from cycle ordering within
  // each stage (lower cycle → lower cluster ID).
  llvm::DenseMap<int, SmallVector<int>> stageToCycles;
  for (const auto &node : schedLoop.nodes) {
    int stage = node.cycle / II;
    stageToCycles[stage].push_back(node.cycle);
  }
  llvm::DenseMap<int, llvm::DenseMap<int, int>> stageAndCycleToCluster;
  for (auto &[stage, cycles] : stageToCycles) {
    llvm::sort(cycles);
    cycles.erase(llvm::unique(cycles), cycles.end());
    for (int i = 0, e = cycles.size(); i < e; ++i)
      stageAndCycleToCluster[stage][cycles[i]] = i;
  }

  int maxStage = 0;
  for (const auto &node : schedLoop.nodes) {
    if (!node.op)
      continue;
    // For multi-stage super-nodes sharing the same Operation*, only
    // write attrs from the canonical node (registered in opToIdx).
    auto opIt = ddg.getOpToIdx().find(node.op);
    if (opIt != ddg.getOpToIdx().end() && opIt->second != node.id)
      continue;
    int stage = node.cycle / II;
    int clusterId = stageAndCycleToCluster[stage][node.cycle];
    maxStage = std::max(maxStage, stage);
    node.op->setAttr(tt::kLoopStageAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), stage));
    node.op->setAttr(tt::kLoopClusterAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), clusterId));
  }

  // Ensure ALL ops in the loop body have loop.stage/loop.cluster.
  for (auto &op : loop.getBody()->without_terminator()) {
    if (!op.hasAttr(tt::kLoopStageAttrName))
      op.setAttr(tt::kLoopStageAttrName,
                 IntegerAttr::get(IntegerType::get(ctx, 32), 0));
    if (!op.hasAttr(tt::kLoopClusterAttrName))
      op.setAttr(tt::kLoopClusterAttrName,
                 IntegerAttr::get(IntegerType::get(ctx, 32), 0));
  }

  // ── 1.5. Per-op: ttg.partition (= our Phase B warp-group decision) ──
  // Emit the WG ID as a DenseI32ArrayAttr so the downstream WS pass
  // (PartitionSchedulingMeta) can pick it up directly instead of
  // re-deriving partitions from scratch. Skip ops with warpGroup<0
  // (unassigned NONE/infrastructure ops) — the downstream pass will
  // propagate them via SSA traversal.
  for (const auto &node : schedLoop.nodes) {
    if (!node.op || node.warpGroup < 0)
      continue;
    auto opIt = ddg.getOpToIdx().find(node.op);
    if (opIt != ddg.getOpToIdx().end() && opIt->second != node.id)
      continue; // multi-stage super-node duplicate
    int32_t wg = static_cast<int32_t>(node.warpGroup);
    node.op->setAttr(ttg::kPartitionAttrName,
                     DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>{wg}));
  }

  // ── 2. Per-loop: tt.modulo_ii, tt.scheduled_max_stage ──
  loop->setAttr("tt.modulo_ii",
                IntegerAttr::get(IntegerType::get(ctx, 32), II));
  loop->setAttr(tt::kScheduledMaxStageAttrName,
                IntegerAttr::get(IntegerType::get(ctx, 32), maxStage));

  // Override tt.num_stages with modulo's authoritative value.
  // The downstream `pipelineForLoop` reads this attribute to decide how
  // many copies of each pipelined SMEM buffer to allocate. Modulo's
  // Step 4.6 already analyzed the budget and decided per-buffer counts
  // in `tt.num_buffers`; align `tt.num_stages` to the deepest of those
  // so the pipeliner's pipelining depth matches modulo's intent.
  // See issue 001_annotation_smem_overflow.
  unsigned moduloNumStages = 1;
  for (const auto &buf : schedLoop.buffers) {
    if (buf.kind == ttg::MemoryKind::SMEM ||
        buf.kind == ttg::MemoryKind::TMEM)
      moduloNumStages = std::max(moduloNumStages, buf.count);
  }
  loop->setAttr(mlir::triton::kNumStagesAttrName,
                IntegerAttr::get(IntegerType::get(ctx, 32),
                                 static_cast<int>(moduloNumStages)));

  // ── 3. Per-buffer: tt.num_buffers, buffer.id ──
  for (const auto &buf : schedLoop.buffers) {
    if (!buf.defOp || buf.kind == ttg::MemoryKind::BARRIER)
      continue;
    buf.defOp->setAttr("tt.num_buffers",
                       IntegerAttr::get(IntegerType::get(ctx, 32), buf.count));
    buf.defOp->setAttr("buffer.id",
                       IntegerAttr::get(IntegerType::get(ctx, 32), buf.id));
  }

  // ── 4. Clean up internal attrs ──
  for (auto &op : loop.getBody()->without_terminator())
    op.removeAttr("tt.modulo_cycle");

  LLVM_DEBUG({
    llvm::dbgs() << "[MODULO] Emitted schedule: II=" << II
                 << " maxStage=" << maxStage
                 << " num_stages=" << (maxStage + 1)
                 << " buffers=" << schedLoop.buffers.size() << "\n";
    for (const auto &buf : schedLoop.buffers) {
      if (!buf.defOp || buf.kind == ttg::MemoryKind::BARRIER)
        continue;
      llvm::dbgs() << "[MODULO]   buf" << buf.id
                   << " count=" << buf.count
                   << " kind=" << (int)buf.kind
                   << " op=" << buf.defOp->getName().getStringRef() << "\n";
    }
  });
}

/// Emit tt.autows annotations on MMA ops from the modulo schedule.
/// These survive through the WS pass (which preserves discardable attrs on
/// MMA ops) and are read by scheduleKeyOpsAnnotation() inside the WS pass's
/// internal scheduleLoops call.
///
/// Format: {"stage": "N", "order": "M"} as a JSON string attribute.
/// "stage" = which SWP pipeline stage the MMA should be in.
/// "order" = relative ordering within the stage (cluster ID).
static void emitMMAAnnotations(scf::ForOp loop,
                               const ttg::DataDependenceGraph &ddg,
                               const ttg::ModuloScheduleResult &schedule) {
  const int II = schedule.II;
  auto ctx = loop.getContext();

  // Compute MMA stages from transitive MMA dependency count.
  //
  // For each MMA, walk backward through distance-0 DDG edges and count
  // how many other MMA nodes are transitively reachable. This captures
  // the data flow structure:
  //   - MMAs depending on 0-1 other MMAs → stage 0 (can be prefetched)
  //   - MMAs depending on 2+ other MMAs → stage 1 (gated on multiple
  //     prior results, natural pipeline boundary)
  //
  // Example: FA backward has 5 MMAs:
  //   qkT (0 MMA deps) → stage 0
  //   dpT (0 MMA deps) → stage 0
  //   dv  (1 MMA dep: qkT) → stage 0
  //   dq  (2 MMA deps: qkT, dpT via dsT) → stage 1
  //   dk  (2 MMA deps: qkT, dpT via dsT) → stage 1
  llvm::DenseSet<unsigned> mmaNodes;
  for (const auto &node : ddg.getNodes()) {
    if (isa<ttng::MMAv5OpInterface>(node.op) || isa<tt::DotOp>(node.op))
      mmaNodes.insert(node.idx);
  }

  // For each MMA, compute transitive MMA predecessors via backward BFS
  // through distance-0 edges only.
  llvm::DenseMap<unsigned, int> mmaStage;
  for (unsigned mmaIdx : mmaNodes) {
    llvm::DenseSet<unsigned> visited;
    llvm::SmallVector<unsigned> worklist;
    worklist.push_back(mmaIdx);
    visited.insert(mmaIdx);

    int mmaPredCount = 0;
    while (!worklist.empty()) {
      unsigned cur = worklist.pop_back_val();
      for (const auto *edge : ddg.getInEdges(cur)) {
        if (edge->distance > 0)
          continue; // skip loop-carried edges
        if (!visited.insert(edge->srcIdx).second)
          continue;
        if (mmaNodes.count(edge->srcIdx))
          mmaPredCount++;
        worklist.push_back(edge->srcIdx);
      }
    }

    // 0-1 MMA predecessors → stage 0 (prefetchable)
    // 2+  MMA predecessors → stage 1 (pipeline boundary)
    mmaStage[mmaIdx] = (mmaPredCount >= 2) ? 1 : 0;
    LDBG("MMA node " << mmaIdx << ": " << mmaPredCount
                     << " transitive MMA predecessors → stage "
                     << mmaStage[mmaIdx]);
  }

  // Collect MMA ops with their stage and cycle, then assign dense cluster IDs.
  struct MMAInfo {
    unsigned nodeIdx;
    Operation *op;
    int stage;
    int cycle;
  };
  llvm::SmallVector<MMAInfo> mmas;

  for (const auto &node : ddg.getNodes()) {
    if (!isa<ttng::MMAv5OpInterface>(node.op) && !isa<tt::DotOp>(node.op))
      continue;
    auto it = schedule.nodeToCycle.find(node.idx);
    if (it == schedule.nodeToCycle.end())
      continue;
    auto stageIt = mmaStage.find(node.idx);
    int stage = stageIt != mmaStage.end() ? stageIt->second : 0;
    mmas.push_back({node.idx, node.op, stage, it->second});
  }

  // Skip annotation if all MMAs are in the same stage — the dependency
  // analysis found no multi-MMA fan-in, so annotations won't help and
  // may break the downstream pipeliner (e.g., GEMM with 1 dot tiled
  // into 4 MMAs, or FA FWD with 2 dots tiled into 4+ MMAs).
  {
    llvm::DenseSet<int> stages;
    for (auto &mma : mmas)
      stages.insert(mma.stage);
    if (stages.size() <= 1) {
      LDBG("Skipping MMA annotations: all " << mmas.size()
                                            << " MMAs in same stage");
      return;
    }
  }

  // Assign order (cluster) within each stage based on MMA dependency depth.
  // MMAs that are independent within the same stage get the same order,
  // matching the hand-tuned convention (e.g., dpT and dv both at order 2,
  // dq and dk both at order 1).
  //
  // Depth = number of same-stage MMA predecessors in the DDG.
  // This groups independent MMAs into the same cluster.
  llvm::DenseMap<unsigned, int> mmaDepthInStage;
  for (auto &mma : mmas) {
    int depth = 0;
    for (auto &other : mmas) {
      if (other.stage != mma.stage || other.nodeIdx == mma.nodeIdx)
        continue;
      // Check if 'other' is a transitive predecessor of 'mma' (distance-0).
      llvm::DenseSet<unsigned> visited;
      llvm::SmallVector<unsigned> worklist;
      worklist.push_back(mma.nodeIdx);
      visited.insert(mma.nodeIdx);
      bool found = false;
      while (!worklist.empty() && !found) {
        unsigned cur = worklist.pop_back_val();
        for (const auto *edge : ddg.getInEdges(cur)) {
          if (edge->distance > 0)
            continue;
          if (edge->srcIdx == other.nodeIdx) {
            found = true;
            break;
          }
          if (visited.insert(edge->srcIdx).second)
            worklist.push_back(edge->srcIdx);
        }
      }
      if (found)
        depth++;
    }
    mmaDepthInStage[mma.nodeIdx] = depth;
  }

  for (auto &mma : mmas) {
    int cluster = mmaDepthInStage[mma.nodeIdx];
    std::string json = "{\"stage\": \"" + std::to_string(mma.stage) +
                       "\", \"order\": \"" + std::to_string(cluster) + "\"}";
    mma.op->setAttr("tt.autows", StringAttr::get(ctx, json));

    LDBG("MMA annotation: stage=" << mma.stage << " order=" << cluster << " on "
                                  << *mma.op);
  }

  if (!mmas.empty())
    LDBG("Emitted tt.autows on " << mmas.size() << " MMA ops");
}

// ============================================================================
// Step 3: Derive per-resource buffer depths from modulo schedule
// ============================================================================

// Blackwell sm_100 SMEM budget (reserve some for barriers/scratch).
constexpr int kSmemBudgetBytesDefault = 228 * 1024;

/// Effective SMEM budget — defaults to 228 KB (B200) but can be overridden
/// via TRITON_MODULO_SMEM_BUDGET_KB for A.7 demo / stress tests.
static int smemBudget() {
  auto env = triton::tools::getStrEnv("TRITON_MODULO_SMEM_BUDGET_KB");
  if (!env.empty()) {
    int kb = std::atoi(env.c_str());
    if (kb > 0)
      return kb * 1024;
  }
  return kSmemBudgetBytesDefault;
}

// Inline wrapper so call sites read as a constant but resolve at runtime
// (for TRITON_MODULO_SMEM_BUDGET_KB override). Replaces a preprocessor
// macro that would bypass scoping and break constexpr contexts.
static inline int kSmemBudgetBytes() { return smemBudget(); }

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
  sn.minWarps = ddgNode.minWarps;

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

/// Build a ScheduleLoop for a loop. For super-nodes (nested loops), builds
/// its own DDG and schedule recursively — works at any nesting depth.
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
        auto childDDG = ttg::DataDependenceGraph::build(innerLoop, model);
        if (childDDG.getNumNodes() > 0) {
          auto childSched = ttg::runModuloScheduling(childDDG);
          if (succeeded(childSched)) {
            unsigned childId = buildScheduleLoop(innerLoop, childDDG,
                                                 *childSched, graph, model);
            sn.childPipelineId = childId;
            sn.prologueLatency = graph.getLoop(childId).prologueLatency;
          }
        }
        if (sn.childPipelineId == UINT_MAX) {
          unsigned childId = graph.addLoop(innerLoop);
          sn.childPipelineId = childId;
        }
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
  // TMA stores need an SMEM staging buffer — the TMA engine reads from
  // SMEM, not registers. The buffer is allocated during TMA lowering but
  // must be accounted for in the SMEM budget here.
  if (isa<tt::DescriptorStoreOp, ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return ttg::MemoryKind::SMEM;
  return ttg::MemoryKind::Register;
}

/// Pass A.7-M4: pre-decide subtile factor for a descriptor_store op.
///
/// Consulted by extractBufferShape so the SMEM staging buffer starts at
/// (BM, BN/S) and the global SMEM reducer sees the shrunk size upfront —
/// otherwise the reducer would cut K-loop pipeline depth based on the full
/// store buffer, even if A.7 would later shrink it.
///
/// Today: env-override only (TRITON_MODULO_EPILOGUE_SUBTILE=2|4). The auto
/// path (estimate K-loop SMEM + outer c_smem vs budget, pick smallest S that
/// fits) is a follow-up — env override is enough for the demo.
static int getEpilogueSubtileForOp(Operation *op) {
  if (!isa<tt::DescriptorStoreOp>(op))
    return 1;
  auto storeOp = cast<tt::DescriptorStoreOp>(op);
  auto srcTy = dyn_cast<RankedTensorType>(storeOp.getSrc().getType());
  if (!srcTy || srcTy.getRank() < 2)
    return 1;
  int BN = srcTy.getShape()[1];
  auto env = triton::tools::getStrEnv("TRITON_MODULO_EPILOGUE_SUBTILE");
  int S = 0;
  if (env == "2") S = 2;
  else if (env == "4") S = 4;
  // Min sub-tile width: 32 elements = 64 bytes for fp16, which is the
  // TMA descriptor alignment minimum on Blackwell. The design doc gate
  // is 64 (better TMA throughput); loosened here so the demo can use
  // S=4 on the existing BN=128 case2 kernel.
  if (S > 1 && BN > 0 && BN % S == 0 && BN / S >= 32)
    return S;
  return 1;
}

static void extractBufferShape(Operation *op, ttg::ScheduleBuffer &buf) {
  Type resultType;
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
    resultType = alloc.getType();
  else if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(op))
    resultType = tmemAlloc.getType();
  else if (auto tmaCopy = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    resultType = tmaCopy.getResult().getType();
  else if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(op))
    resultType = storeOp.getSrc().getType();
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

  // Pass A.7-M4: when this buffer backs a descriptor_store that we'll
  // subtile, shrink its N-dimension upfront so the SMEM budget reducer
  // sees the post-subtile size and doesn't have to cut K-loop depth.
  // (The matching `count = 2` bump for double-buffered TMA-store overlap
  // is applied by the caller AFTER computeBufferCount, since that helper
  // overwrites count.)
  if (int S = getEpilogueSubtileForOp(op); S > 1 && buf.shape.size() >= 2) {
    buf.shape.back() /= S;
  }
}

/// Walk downstream from `startId`, accumulating the latest absolute-cycle
/// end of any non-transparent consumer. Transparent view ops (NONE pipeline
/// with zero latency, e.g. memdesc_trans / memdesc_subview) don't truly
/// consume the buffer — they just rebind the underlying memory descriptor —
/// so we walk through them to the real consumer (the MMA / load / store
/// that actually reads the bytes).
static int walkLastConsumerEnd(const ttg::ScheduleLoop &loop,
                               unsigned startId, int prodCycle, int II,
                               int distAcc, llvm::DenseSet<unsigned> &seen) {
  int lastEnd = prodCycle;
  for (const auto &edge : loop.edges) {
    if (edge.srcId != startId)
      continue;
    if (!seen.insert(edge.dstId).second)
      continue;
    const auto &consumer = loop.getNode(edge.dstId);
    int totalDist = distAcc + static_cast<int>(edge.distance);
    bool transparent =
        consumer.pipeline == ttg::HWPipeline::NONE && consumer.latency == 0;
    if (transparent) {
      lastEnd = std::max(lastEnd,
                         walkLastConsumerEnd(loop, consumer.id, prodCycle,
                                             II, totalDist, seen));
      continue;
    }
    int end = consumer.cycle + consumer.latency + totalDist * II;
    lastEnd = std::max(lastEnd, end);
  }
  return lastEnd;
}

/// Step 3: Compute buffer count from cycle-level lifetime.
///
/// Design doc formula:
///   lifetime(R) = lastConsumerEnd - producerStart
///   num_buffers(R) = floor(lifetime(R) / II) + 1
///
/// For loop-carried edges (distance > 0), the consumer in iteration i+d
/// effectively ends at: consumerEnd + d * II (in absolute time).
/// This is equivalent to adding d * II to the lifetime.
static unsigned computeBufferCount(const ttg::ScheduleLoop &loop,
                                   unsigned producerNodeId) {
  const auto &producer = loop.getNode(producerNodeId);
  int prodCycle = producer.cycle;
  int II = loop.II;
  if (II <= 0)
    return 1;

  llvm::DenseSet<unsigned> seen;
  seen.insert(producerNodeId);
  int lastConsumerEnd =
      walkLastConsumerEnd(loop, producerNodeId, prodCycle, II, 0, seen);

  int lifetime = lastConsumerEnd - prodCycle;
  int numBuffers = lifetime / II + 1;
  return static_cast<unsigned>(std::max(numBuffers, 1));
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

    buf.count = computeBufferCount(loop, node.id);
    // No artificial minimum: trust the lifetime-based count from
    // `computeBufferCount`. Earlier code applied `std::max(count, 3u)`
    // for SMEM ("at least 3 for async copy pipelining") which inflated
    // every load to triple-buffered, often exceeding the SMEM budget.
    // The schedule's lifetime analysis already guarantees correctness;
    // perf-vs-fit is decided by `reduceBuffersForGlobalBudget`. See
    // issue 001_annotation_smem_overflow.

    // Pass A.7-M4: descriptor_store staging buffers that we'll subtile
    // need count>=2 (double-buffered) so the emitter can issue the next
    // local_store while the prior TMA store is in flight — matches the
    // blackwell_gemm_ws tutorial overlap pattern. extractBufferShape
    // already shrank the shape; bump count here, after computeBufferCount
    // overwrote whatever it had set.
    if (getEpilogueSubtileForOp(node.op) > 1)
      buf.count = std::max<unsigned>(buf.count, 2u);

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

  // Equalize co-consumed buffer depths: buffers that feed the same
  // consumer op (e.g., A and B tiles both feeding MMA) must have the
  // same depth. Otherwise the shallower buffer limits the pipeline
  // depth and the deeper buffer wastes SMEM.
  //
  // Walk upstream from each node to collect all SMEM buffers it
  // transitively consumes (through NONE-pipeline intermediaries like
  // memdesc_trans), then equalize their depths.
  for (const auto &node : loop.nodes) {
    // Only equalize for pipeline ops that consume multiple buffers.
    if (node.pipeline == ttg::HWPipeline::NONE)
      continue;

    // Collect all SMEM buffers reachable upstream through edges.
    llvm::SmallVector<unsigned> upstreamBufs;
    llvm::SmallVector<unsigned> worklist;
    llvm::DenseSet<unsigned> visited;
    worklist.push_back(node.id);
    visited.insert(node.id);
    while (!worklist.empty()) {
      unsigned cur = worklist.pop_back_val();
      const auto &curNode = loop.nodes[cur];
      // If this node produces an SMEM buffer, collect it.
      if (curNode.producesBuffer != UINT_MAX &&
          curNode.producesBuffer < loop.buffers.size() &&
          loop.buffers[curNode.producesBuffer].kind == ttg::MemoryKind::SMEM)
        upstreamBufs.push_back(curNode.producesBuffer);
      // Walk upstream through predecessors (NONE-pipeline only, to
      // avoid crossing pipeline boundaries).
      for (const auto &edge : loop.edges) {
        if (edge.dstId != cur || edge.distance > 0)
          continue;
        const auto &pred = loop.nodes[edge.srcId];
        if (pred.pipeline != ttg::HWPipeline::NONE &&
            pred.pipeline != ttg::HWPipeline::MEM)
          continue;
        if (visited.insert(edge.srcId).second)
          worklist.push_back(edge.srcId);
      }
    }

    if (upstreamBufs.size() <= 1)
      continue;

    unsigned maxDepth = 0;
    for (unsigned bufId : upstreamBufs)
      maxDepth = std::max(maxDepth, loop.buffers[bufId].count);
    for (unsigned bufId : upstreamBufs) {
      if (loop.buffers[bufId].count != maxDepth) {
        LLVM_DEBUG(llvm::dbgs() << "[Step3] Equalized buf" << bufId
                                << " depth from " << loop.buffers[bufId].count
                                << " to " << maxDepth << " (co-consumed by "
                                << node.op->getName().getStringRef() << ")\n");
        loop.buffers[bufId].count = maxDepth;
      }
    }
  }

  for (unsigned dataBufId : dataBufferIds) {
    unsigned barId = loop.buffers.size();
    ttg::ScheduleBuffer bar;
    bar.id = barId;
    bar.kind = ttg::MemoryKind::BARRIER;
    bar.count = loop.buffers[dataBufId].count;
    bar.defOp = loop.buffers[dataBufId].defOp;
    bar.pairedBufferId = dataBufId;
    loop.buffers[dataBufId].pairedBufferId = barId;
    loop.buffers.push_back(bar);
  }
}

// ============================================================================
// Step 4.6: Global Memory Budget Check and Reduction
// ============================================================================

// Blackwell sm_100 TMEM budget. Logical capacity is 128 lanes × 512 cols ×
// 4 bytes/col = 256KB.
constexpr int64_t kTmemBudgetBytes = 128 * 512 * 4;

// Forward decl — defined under Step 4.5 below; called by reduceBuffersForBudget
// to refresh PhysicalBuffer sizes after a depth reduction.
static void buildPhysicalBuffers(ttg::ScheduleLoop &loop);

/// Compute total SMEM/TMEM usage. Buffers in the same merge group share
/// a physical allocation sized to the largest member at the deepest
/// count, so we charge each group exactly once via its PhysicalBuffer.
/// Unmerged data buffers and all BARRIER buffers (always SMEM) are
/// charged individually.
static int64_t computeTotalMemory(const ttg::ScheduleLoop &loop,
                                  ttg::MemoryKind targetKind) {
  int64_t total = 0;

  // Charge each materialized physical buffer once.
  for (const auto &pb : loop.physicalBuffers) {
    bool isTarget = (pb.kind == targetKind);
    if (targetKind == ttg::MemoryKind::SMEM &&
        pb.kind == ttg::MemoryKind::BARRIER)
      isTarget = true;
    if (isTarget)
      total += pb.totalBytes();
  }

  // Charge unmerged buffers (mergeGroupId == UINT_MAX) directly.
  for (const auto &buf : loop.buffers) {
    if (buf.mergeGroupId != UINT_MAX)
      continue;
    bool isTarget = (buf.kind == targetKind);
    if (targetKind == ttg::MemoryKind::SMEM &&
        buf.kind == ttg::MemoryKind::BARRIER)
      isTarget = true;
    if (!isTarget)
      continue;
    if (buf.kind != ttg::MemoryKind::BARRIER &&
        (buf.shape.empty() || buf.elementBitWidth == 0))
      continue;
    total += buf.totalBytes();
  }
  return total;
}

static int64_t computeTotalSmem(const ttg::ScheduleLoop &loop) {
  return computeTotalMemory(loop, ttg::MemoryKind::SMEM);
}
static int64_t computeTotalTmem(const ttg::ScheduleLoop &loop) {
  return computeTotalMemory(loop, ttg::MemoryKind::TMEM);
}

/// Compute the buffer lifetime (in cycles) for a given producer node.
/// Walks transitively through transparent view ops (same as
/// walkLastConsumerEnd) so the lifetime is consistent with the count
/// computed by computeBufferCount.
static int computeBufferLifetime(const ttg::ScheduleLoop &loop,
                                 unsigned producerNodeId) {
  const auto &producer = loop.getNode(producerNodeId);
  int prodCycle = producer.cycle;
  llvm::DenseSet<unsigned> seen;
  int lastConsumerEnd =
      walkLastConsumerEnd(loop, producerNodeId, prodCycle, loop.II, 0, seen);
  return lastConsumerEnd - prodCycle;
}

/// Cost (design doc §1437-1477): kernel time increase per byte saved by
/// reducing this buffer's depth by 1. Lower = greedily reduce first.
///
/// new_lifetime_bound = (count - 1) × II. If lifetime exceeds it, the
/// producer must stall and effective II grows; otherwise depth reduction
/// is free of latency impact (ii_increase = 0).
///
/// time_increase = ii_increase × tripCount  (loop region)
///               = ii_increase             (non-loop region — single pass)
/// cost          = time_increase / size_bytes_saved
static double kernelTimeCost(const ttg::ScheduleLoop &loop,
                             const ttg::ScheduleBuffer &buf) {
  if (buf.count <= 1 || buf.kind == ttg::MemoryKind::BARRIER)
    return std::numeric_limits<double>::infinity();
  if (loop.II <= 0)
    return std::numeric_limits<double>::infinity();
  int lifetime = buf.liveEnd - buf.liveStart;
  int newCount = static_cast<int>(buf.count) - 1;
  int newLifetimeBound = newCount * loop.II;
  int iiIncrease = 0;
  if (lifetime > newLifetimeBound) {
    int newII = (lifetime + newCount - 1) / newCount;
    iiIncrease = newII - loop.II;
  }
  int tc = loop.tripCount > 0 ? loop.tripCount : 1;
  double timeIncrease = static_cast<double>(iiIncrease) * tc;
  int64_t saved = buf.sizeBytes();
  if (saved <= 0)
    return std::numeric_limits<double>::infinity();
  return timeIncrease / static_cast<double>(saved);
}

/// Build co-consumed buffer groups: buffers that transitively feed the
/// same pipeline op must have the same depth.
static llvm::SmallVector<llvm::SmallVector<unsigned>>
buildCoConsumedGroups(const ttg::ScheduleLoop &loop) {
  // Map each SMEM buffer to a group ID via union-find.
  llvm::DenseMap<unsigned, unsigned> bufToGroup;
  unsigned nextGroup = 0;

  for (const auto &node : loop.nodes) {
    if (node.pipeline == ttg::HWPipeline::NONE)
      continue;
    // Walk upstream to collect all SMEM buffers feeding this node.
    llvm::SmallVector<unsigned> upstreamBufs;
    llvm::SmallVector<unsigned> worklist = {node.id};
    llvm::DenseSet<unsigned> visited = {node.id};
    while (!worklist.empty()) {
      unsigned cur = worklist.pop_back_val();
      const auto &curNode = loop.nodes[cur];
      if (curNode.producesBuffer != UINT_MAX &&
          curNode.producesBuffer < loop.buffers.size() &&
          loop.buffers[curNode.producesBuffer].kind == ttg::MemoryKind::SMEM)
        upstreamBufs.push_back(curNode.producesBuffer);
      for (const auto &edge : loop.edges) {
        if (edge.dstId != cur || edge.distance > 0)
          continue;
        const auto &pred = loop.nodes[edge.srcId];
        if (pred.pipeline != ttg::HWPipeline::NONE &&
            pred.pipeline != ttg::HWPipeline::MEM)
          continue;
        if (visited.insert(edge.srcId).second)
          worklist.push_back(edge.srcId);
      }
    }
    if (upstreamBufs.size() <= 1)
      continue;
    // Union all upstream buffers into the same group. Collect all
    // existing group IDs, pick the smallest, and rewrite all members
    // of every touched group to use that ID (transitive merge).
    llvm::DenseSet<unsigned> existingGroups;
    for (unsigned bufId : upstreamBufs) {
      auto it = bufToGroup.find(bufId);
      if (it != bufToGroup.end())
        existingGroups.insert(it->second);
    }
    unsigned mergedGroupId;
    if (existingGroups.empty()) {
      mergedGroupId = nextGroup++;
    } else {
      mergedGroupId =
          *std::min_element(existingGroups.begin(), existingGroups.end());
      // Rewrite all buffers in the other groups to the merged ID.
      if (existingGroups.size() > 1) {
        for (auto &[bufId, gid] : bufToGroup) {
          if (existingGroups.count(gid))
            gid = mergedGroupId;
        }
      }
    }
    for (unsigned bufId : upstreamBufs)
      bufToGroup[bufId] = mergedGroupId;
  }

  // Collect groups.
  llvm::DenseMap<unsigned, llvm::SmallVector<unsigned>> groupMap;
  for (auto &[bufId, gid] : bufToGroup)
    groupMap[gid].push_back(bufId);
  llvm::SmallVector<llvm::SmallVector<unsigned>> groups;
  for (auto &[gid, members] : groupMap)
    groups.push_back(std::move(members));
  return groups;
}

/// Reduce all buffers in a co-consumed group to the given depth.
static void reduceGroupToDepth(ttg::ScheduleLoop &loop,
                               const llvm::SmallVector<unsigned> &group,
                               unsigned newDepth) {
  for (unsigned bufId : group) {
    if (loop.buffers[bufId].count > newDepth) {
      loop.buffers[bufId].count = newDepth;
      unsigned barId = loop.buffers[bufId].pairedBufferId;
      if (barId != UINT_MAX)
        loop.buffers[barId].count = newDepth;
    }
  }
}

/// Step 4.6: If buffer allocation exceeds SMEM/TMEM budget, greedily reduce
/// buffer depths using the kernel_time_cost metric from the design doc.
/// Co-consumed buffers (feeding the same pipeline op) are reduced together.
/// After reduction, recompute II from the tightest buffer constraint:
///   new_II = max over reduced buffers of ceil(lifetime / new_depth).
/// The schedule (op placement) stays fixed — only II and buffer depths change.
static bool reduceBuffersForBudget(ttg::ScheduleLoop &loop,
                                   int64_t smemReserved = 0) {
  // Precompute buffer lifetimes (from the original schedule, before reduction).
  llvm::DenseMap<unsigned, int> bufLifetimes;
  for (unsigned i = 0; i < loop.buffers.size(); ++i) {
    auto &buf = loop.buffers[i];
    if (buf.kind == ttg::MemoryKind::BARRIER ||
        buf.kind == ttg::MemoryKind::Register)
      continue;
    for (const auto &node : loop.nodes) {
      if (node.producesBuffer == buf.id) {
        bufLifetimes[i] = computeBufferLifetime(loop, node.id);
        break;
      }
    }
  }

  // Build co-consumed groups so we reduce them together.
  auto coGroups = buildCoConsumedGroups(loop);
  // Map bufId → group index for quick lookup.
  llvm::DenseMap<unsigned, unsigned> bufToGroupIdx;
  for (unsigned g = 0; g < coGroups.size(); ++g)
    for (unsigned bufId : coGroups[g])
      bufToGroupIdx[bufId] = g;

  int originalII = loop.II;

  int64_t effectiveSmemBudget = kSmemBudgetBytes() - smemReserved;
  if (smemReserved > 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.6] SMEM reserved by other regions: " << smemReserved
               << " B, effective budget: " << effectiveSmemBudget << " B\n");
  }

  // SMEM reduction: greedily reduce the cheapest buffer first.
  // When a buffer is in a co-consumed group, reduce the entire group.
  LLVM_DEBUG({
    llvm::dbgs() << "[Step4.6] Total SMEM=" << computeTotalSmem(loop)
                 << " budget=" << effectiveSmemBudget << "\n";
    for (unsigned i = 0; i < loop.buffers.size(); ++i) {
      const auto &buf = loop.buffers[i];
      llvm::dbgs() << "[Step4.6]   buf" << i << " kind="
                   << (int)buf.kind << " count=" << buf.count
                   << " size=" << buf.sizeBytes() << "B\n";
    }
  });
  while (computeTotalSmem(loop) > effectiveSmemBudget) {
    int bestIdx = -1;
    double bestCost = std::numeric_limits<double>::infinity();
    for (unsigned i = 0; i < loop.buffers.size(); ++i) {
      const auto &buf = loop.buffers[i];
      if (buf.kind != ttg::MemoryKind::SMEM || buf.count <= 1)
        continue;
      double cost = kernelTimeCost(loop, buf);
      if (cost < bestCost) {
        bestCost = cost;
        bestIdx = i;
      }
    }
    if (bestIdx < 0)
      break;
    unsigned newDepth = loop.buffers[bestIdx].count - 1;
    // If this buffer is in a co-consumed group, reduce the whole group.
    auto groupIt = bufToGroupIdx.find(bestIdx);
    if (groupIt != bufToGroupIdx.end()) {
      reduceGroupToDepth(loop, coGroups[groupIt->second], newDepth);
      LLVM_DEBUG(llvm::dbgs()
                 << "[Step4.6] Reduced co-consumed group (buf" << bestIdx
                 << " + partners) to count=" << newDepth << "\n");
    } else {
      loop.buffers[bestIdx].count = newDepth;
      unsigned barId = loop.buffers[bestIdx].pairedBufferId;
      if (barId != UINT_MAX)
        loop.buffers[barId].count = newDepth;
      LLVM_DEBUG(llvm::dbgs() << "[Step4.6] Reduced SMEM buf" << bestIdx
                              << " to count=" << newDepth << "\n");
    }
  }

  // TMEM reduction
  while (computeTotalTmem(loop) > kTmemBudgetBytes) {
    int bestIdx = -1;
    double bestCost = std::numeric_limits<double>::infinity();
    for (unsigned i = 0; i < loop.buffers.size(); ++i) {
      auto &buf = loop.buffers[i];
      if (buf.kind != ttg::MemoryKind::TMEM || buf.count <= 1)
        continue;
      double cost = kernelTimeCost(loop, buf);
      if (cost < bestCost) {
        bestCost = cost;
        bestIdx = i;
      }
    }
    if (bestIdx < 0)
      break;
    loop.buffers[bestIdx].count--;
    unsigned barId = loop.buffers[bestIdx].pairedBufferId;
    if (barId != UINT_MAX)
      loop.buffers[barId].count = loop.buffers[bestIdx].count;
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.6] Reduced TMEM buf" << bestIdx
               << " to count=" << loop.buffers[bestIdx].count << "\n");
  }

  // Recompute II from reduced buffer depths.
  // new_II = max over all buffers of ceil(lifetime / depth).
  int newII = originalII;
  for (unsigned i = 0; i < loop.buffers.size(); ++i) {
    auto &buf = loop.buffers[i];
    if (buf.kind == ttg::MemoryKind::BARRIER ||
        buf.kind == ttg::MemoryKind::Register)
      continue;
    auto it = bufLifetimes.find(i);
    if (it == bufLifetimes.end() || buf.count <= 0)
      continue;
    int requiredII = (it->second + buf.count - 1) / buf.count;
    if (requiredII > newII) {
      LLVM_DEBUG(llvm::dbgs() << "[Step4.6] buf" << i << " lifetime="
                              << it->second << " depth=" << buf.count
                              << " → requires II=" << requiredII << "\n");
      newII = requiredII;
    }
  }

  if (newII != originalII) {
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.6] Raising II from " << originalII << " to " << newII
               << " due to buffer depth reduction\n");
    loop.II = newII;
    loop.maxStage = 0;
    // Recompute per-node stage with the new II — without this, node.stage
    // keeps the value from the initial Rau's run (computed with the old
    // smaller II), producing nonsense like stage=23 when maxStage=0.
    for (auto &node : loop.nodes) {
      node.stage = node.cycle / newII;
      loop.maxStage = std::max(loop.maxStage, node.stage);
    }
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.6] New maxStage=" << loop.maxStage << "\n");
  }

  int64_t smemUsed = computeTotalSmem(loop);
  int64_t tmemUsed = computeTotalTmem(loop);
  bool smemOk = smemUsed <= kSmemBudgetBytes();
  bool tmemOk = tmemUsed <= kTmemBudgetBytes;
  LLVM_DEBUG(llvm::dbgs() << "[Step4.6] Budget: SMEM " << smemUsed << "/"
                          << kSmemBudgetBytes() << (smemOk ? " OK" : " EXCEEDED")
                          << ", TMEM " << tmemUsed << "/" << kTmemBudgetBytes
                          << (tmemOk ? " OK" : " EXCEEDED") << "\n");
  if (!smemOk || !tmemOk) {
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.6] WARNING: memory budget exceeded"
               << " (all reducible buffers at count=1). "
               << "SMEM: " << smemUsed << "/" << kSmemBudgetBytes()
               << ", TMEM: " << tmemUsed << "/" << kTmemBudgetBytes << "\n");
  }
  return smemOk && tmemOk;
}

/// Step 4.6 (global): All loops in the function share the same SMEM pool.
/// `reduceBuffersForBudget` runs per-loop and only knows about its own
/// buffers (with optional ancestor reservation). When sibling/cousin loops
/// independently fit but their SUM exceeds the SMEM budget, every per-loop
/// check passes yet runtime fails (see issue 001_annotation_smem_overflow:
/// inner k-loop's 192 KB + outer persistent-loop's 128 KB output staging =
/// 320 KB > 232 KB).
///
/// This function runs after all per-loop checks and reduces buffer depths
/// across loops jointly until the global SMEM total fits the budget.
/// Reduction picks the cheapest buffer (lowest kernelTimeCost) from any
/// loop. After reduction the per-loop II is recomputed in the same way as
/// the per-loop reducer.
static void reduceBuffersForGlobalBudget(ttg::ScheduleGraph &graph) {
  auto totalSmem = [&]() {
    int64_t t = 0;
    for (const auto &loop : graph.loops)
      t += computeTotalSmem(loop);
    return t;
  };

  int64_t initialTotal = totalSmem();
  LLVM_DEBUG({
    llvm::dbgs() << "[Step4.6-Global] Initial total SMEM=" << initialTotal
                 << " (sum across " << graph.loops.size()
                 << " loops), budget=" << kSmemBudgetBytes() << "\n";
    for (unsigned li = 0; li < graph.loops.size(); ++li) {
      auto &loop = graph.loops[li];
      llvm::dbgs() << "[Step4.6-Global]   loop" << li
                   << " smem=" << computeTotalSmem(loop) << " buffers:\n";
      for (unsigned bi = 0; bi < loop.buffers.size(); ++bi) {
        const auto &buf = loop.buffers[bi];
        llvm::dbgs() << "[Step4.6-Global]     buf" << bi
                     << " kind=" << (int)buf.kind << " count=" << buf.count
                     << " size=" << buf.sizeBytes() << "B\n";
      }
    }
  });
  if (initialTotal <= kSmemBudgetBytes()) {
    LLVM_DEBUG(llvm::dbgs() << "[Step4.6-Global] Already fits, skipping\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "[Step4.6-Global] Applying global reduction\n");

  // Greedy: at each step, pick the cheapest reducible SMEM buffer in any
  // loop and decrement its count. Stop when fits, or when no buffer can
  // be reduced further.
  while (totalSmem() > kSmemBudgetBytes()) {
    int bestLoopIdx = -1;
    int bestBufIdx = -1;
    double bestCost = std::numeric_limits<double>::infinity();
    for (unsigned li = 0; li < graph.loops.size(); ++li) {
      auto &loop = graph.loops[li];
      for (unsigned bi = 0; bi < loop.buffers.size(); ++bi) {
        const auto &buf = loop.buffers[bi];
        if (buf.kind != ttg::MemoryKind::SMEM || buf.count <= 1)
          continue;
        double cost = kernelTimeCost(loop, buf);
        if (cost < bestCost) {
          bestCost = cost;
          bestLoopIdx = li;
          bestBufIdx = bi;
        }
      }
    }
    if (bestLoopIdx < 0)
      break; // nothing left to reduce.

    auto &loop = graph.loops[bestLoopIdx];
    auto &buf = loop.buffers[bestBufIdx];
    unsigned newDepth = buf.count - 1;
    buf.count = newDepth;
    if (buf.pairedBufferId != UINT_MAX)
      loop.buffers[buf.pairedBufferId].count = newDepth;
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.6-Global] Reduced loop" << bestLoopIdx << "/buf"
               << bestBufIdx << " to count=" << newDepth
               << " (cost=" << bestCost << ", new total=" << totalSmem()
               << ")\n");
    // Refresh PhysicalBuffers after a depth change so computeTotalSmem
    // reflects the reduction.
    buildPhysicalBuffers(loop);
  }

  // Recompute II per loop based on new buffer depths.
  for (auto &loop : graph.loops) {
    int originalII = loop.II;
    int newII = originalII;
    for (unsigned i = 0; i < loop.buffers.size(); ++i) {
      auto &buf = loop.buffers[i];
      if (buf.kind == ttg::MemoryKind::BARRIER ||
          buf.kind == ttg::MemoryKind::Register || buf.count <= 0)
        continue;
      int lifetime = -1;
      for (const auto &node : loop.nodes) {
        if (node.producesBuffer == buf.id) {
          lifetime = computeBufferLifetime(loop, node.id);
          break;
        }
      }
      if (lifetime < 0)
        continue;
      int requiredII = (lifetime + buf.count - 1) / buf.count;
      newII = std::max(newII, requiredII);
    }
    if (newII != originalII) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[Step4.6-Global] Loop " << loop.id << ": raising II from "
                 << originalII << " to " << newII
                 << " due to global buffer reduction\n");
      loop.II = newII;
      loop.maxStage = 0;
      for (auto &node : loop.nodes) {
        node.stage = node.cycle / newII;
        loop.maxStage = std::max(loop.maxStage, node.stage);
      }
    }
  }

  int64_t finalTotal = totalSmem();
  LLVM_DEBUG(llvm::dbgs() << "[Step4.6-Global] Final SMEM=" << finalTotal << "/"
                          << kSmemBudgetBytes()
                          << (finalTotal <= kSmemBudgetBytes() ? " OK"
                                                              : " STILL EXCEEDED")
                          << "\n");
}

// ============================================================================
// Step 4.5: Lifetime-Aware Buffer Merging
// ============================================================================

/// Faithful port of design doc §1156-1177 `intervals_overlap_modular`:
/// project each interval onto [0, II), split if it wraps, then test all
/// (a-half, b-half) pairs for plain interval overlap.
static bool intervalsOverlapModularSingle(int aStart, int aEnd, int bStart,
                                          int bEnd, int II) {
  if (II <= 0)
    return true;
  // Empty intervals can't overlap anything.
  if (aEnd <= aStart || bEnd <= bStart)
    return false;

  auto mod = [II](int x) {
    int r = x % II;
    return r < 0 ? r + II : r;
  };
  int aS = mod(aStart);
  int aE = mod(aEnd);
  int bS = mod(bStart);
  int bE = mod(bEnd);
  // A live interval whose duration is >= II covers the entire ring.
  if (aEnd - aStart >= II || bEnd - bStart >= II)
    return true;

  llvm::SmallVector<std::pair<int, int>, 2> aHalves;
  if (aS < aE)
    aHalves.push_back({aS, aE});
  else if (aS > aE) {
    aHalves.push_back({aS, II});
    aHalves.push_back({0, aE});
  } else {
    // aS == aE with non-empty original ⇒ wraps fully.
    return true;
  }
  llvm::SmallVector<std::pair<int, int>, 2> bHalves;
  if (bS < bE)
    bHalves.push_back({bS, bE});
  else if (bS > bE) {
    bHalves.push_back({bS, II});
    bHalves.push_back({0, bE});
  } else {
    return true;
  }
  for (auto [s1, e1] : aHalves)
    for (auto [s2, e2] : bHalves)
      if (s1 < e2 && s2 < e1)
        return true;
  return false;
}

/// Faithful port of design doc §1180-1203 `any_instances_overlap`.
/// For each (d1, d2) pair of in-flight buffer instances, shift interval B
/// by (d2 - d1) * II and test for modular overlap. Two resources can share
/// a physical buffer only if NO (d1, d2) pair produces overlap.
static bool anyInstancesOverlap(int aStart, int aEnd, int bStart, int bEnd,
                                unsigned aDepth, unsigned bDepth, int II) {
  if (II <= 0)
    return true;
  for (unsigned d1 = 0; d1 < std::max(1u, aDepth); ++d1) {
    for (unsigned d2 = 0; d2 < std::max(1u, bDepth); ++d2) {
      int offset = (static_cast<int>(d2) - static_cast<int>(d1)) * II;
      if (intervalsOverlapModularSingle(aStart, aEnd, bStart + offset,
                                        bEnd + offset, II))
        return true;
    }
  }
  return false;
}

/// Compute and store [liveStart, liveEnd) for every data buffer in the loop.
/// Lifetime is producer cycle → max(consumer.cycle + consumer.selfLatency)
/// across direct consumer edges, with loop-carried edges adjusted by
/// distance × II. Paired barriers inherit the data buffer's interval
/// (per design doc §215).
static void computeBufferLifetimes(ttg::ScheduleLoop &loop) {
  if (loop.II <= 0)
    return;
  for (auto &buf : loop.buffers) {
    if (buf.kind == ttg::MemoryKind::BARRIER ||
        buf.kind == ttg::MemoryKind::Register)
      continue;
    for (const auto &node : loop.nodes) {
      if (node.producesBuffer != buf.id)
        continue;
      buf.liveStart = node.cycle;
      // Walk transitively through transparent view ops (memdesc_trans /
      // memdesc_subview) so the buffer's live range reaches the actual
      // MMA / load / store that holds the SMEM, not just the metadata
      // rebind that the producer feeds directly.
      llvm::DenseSet<unsigned> seen;
      seen.insert(node.id);
      buf.liveEnd =
          walkLastConsumerEnd(loop, node.id, node.cycle, loop.II, 0, seen);
      break;
    }
  }
  // Mirror data-buffer intervals onto their paired barriers.
  for (auto &bar : loop.buffers) {
    if (bar.kind != ttg::MemoryKind::BARRIER)
      continue;
    if (bar.pairedBufferId == UINT_MAX)
      continue;
    const auto &data = loop.buffers[bar.pairedBufferId];
    bar.liveStart = data.liveStart;
    bar.liveEnd = data.liveEnd;
  }
}

/// Cycle-freedom check (design doc §1129-1137 / §1216): merging buffers A
/// and B adds an implicit edge "last_consumer_of_A happens-before
/// producer_of_B". Reject the merge if it would create a cycle in the
/// node-level dependency graph.
///
/// We model the merge as a candidate edge (last_consumer(B'), producer(A))
/// added per pair, where (A, B') ranges over (existing group members,
/// candidate). Run a forward reachability from producer(A) over all real
/// edges PLUS the prospective merge edges; if producer(B') is reachable
/// before the new edge is added the other direction, we'd close a cycle.
static bool mergeIntroducesCycle(const ttg::ScheduleLoop &loop,
                                 llvm::ArrayRef<unsigned> groupMembers,
                                 unsigned candidate) {
  // Collect (producer, lastConsumer) per buffer in {groupMembers + candidate}.
  auto findProducer = [&](unsigned bufId) -> unsigned {
    for (const auto &n : loop.nodes)
      if (n.producesBuffer == bufId)
        return n.id;
    return UINT_MAX;
  };
  auto lastConsumers = [&](unsigned bufId) {
    llvm::SmallVector<unsigned, 4> result;
    unsigned prodId = findProducer(bufId);
    if (prodId == UINT_MAX)
      return result;
    for (const auto &e : loop.edges)
      if (e.srcId == prodId)
        result.push_back(e.dstId);
    return result;
  };

  // Build adjacency for plain DDG (intra-iteration edges only — cross-
  // iteration edges close their own loops, which is fine).
  llvm::DenseMap<unsigned, llvm::SmallVector<unsigned, 4>> adj;
  for (const auto &e : loop.edges)
    if (e.distance == 0)
      adj[e.srcId].push_back(e.dstId);

  // Collect candidate-induced edges: for every existing member M and the
  // candidate C, both directions of "last_consumer happens-before producer"
  // are added as additional edges to test. Coloring will pick a serial
  // order, but for the cycle test, both possibilities are checked.
  llvm::SmallVector<std::pair<unsigned, unsigned>, 8> proposed;
  auto addPair = [&](unsigned aBuf, unsigned bBuf) {
    unsigned bProd = findProducer(bBuf);
    if (bProd == UINT_MAX)
      return;
    for (unsigned cons : lastConsumers(aBuf))
      proposed.push_back({cons, bProd});
  };
  for (unsigned m : groupMembers) {
    addPair(m, candidate);
    addPair(candidate, m);
  }

  // BFS from each proposed edge's source over (real edges + all proposed
  // edges except itself); a cycle exists iff we can reach back to itself.
  for (size_t i = 0; i < proposed.size(); ++i) {
    auto [src, dst] = proposed[i];
    llvm::DenseSet<unsigned> visited;
    llvm::SmallVector<unsigned, 16> stack{dst};
    while (!stack.empty()) {
      unsigned u = stack.pop_back_val();
      if (!visited.insert(u).second)
        continue;
      if (u == src)
        return true;
      for (unsigned v : adj.lookup(u))
        stack.push_back(v);
      for (size_t j = 0; j < proposed.size(); ++j)
        if (j != i && proposed[j].first == u)
          stack.push_back(proposed[j].second);
    }
  }
  return false;
}

/// Cost guard (design doc §1418-1429): merging is only beneficial when
/// max(size) × max(count) < sum(size × count). Otherwise, the physical
/// buffer (sized to the largest member with the deepest count) wastes
/// more memory than separate allocations.
static bool shouldMerge(const ttg::ScheduleLoop &loop,
                        llvm::ArrayRef<unsigned> groupMembers,
                        unsigned candidate) {
  int64_t separateCost = 0;
  int64_t maxSize = 0;
  unsigned maxCount = 0;
  auto accum = [&](unsigned bufId) {
    const auto &b = loop.buffers[bufId];
    int64_t sz = b.sizeBytes();
    separateCost += sz * static_cast<int64_t>(b.count);
    maxSize = std::max(maxSize, sz);
    maxCount = std::max(maxCount, b.count);
  };
  for (unsigned m : groupMembers)
    accum(m);
  accum(candidate);
  int64_t mergedCost = maxSize * static_cast<int64_t>(maxCount);
  return mergedCost < separateCost;
}

/// Materialize PhysicalBuffer entries from each merge group. Per design
/// doc §1140-1147: physical size = max(member.sizeBytes), physical count =
/// max(member.count).
static void buildPhysicalBuffers(ttg::ScheduleLoop &loop) {
  loop.physicalBuffers.clear();
  llvm::DenseMap<unsigned, unsigned> groupToPhys;
  for (const auto &buf : loop.buffers) {
    if (buf.mergeGroupId == UINT_MAX)
      continue;
    auto it = groupToPhys.find(buf.mergeGroupId);
    if (it == groupToPhys.end()) {
      ttg::PhysicalBuffer pb;
      pb.id = buf.mergeGroupId;
      pb.kind = buf.kind;
      pb.sizeBytes = buf.sizeBytes();
      pb.count = buf.count;
      pb.memberBufferIds.push_back(buf.id);
      groupToPhys[buf.mergeGroupId] = loop.physicalBuffers.size();
      loop.physicalBuffers.push_back(std::move(pb));
    } else {
      auto &pb = loop.physicalBuffers[it->second];
      pb.sizeBytes = std::max(pb.sizeBytes, buf.sizeBytes());
      pb.count = std::max(pb.count, buf.count);
      pb.memberBufferIds.push_back(buf.id);
    }
  }
}

/// Step 4.5: Merge buffers with non-overlapping lifetimes.
/// Greedy interval-graph coloring with three guards:
///   1. Same storage kind (SMEM only merges with SMEM).
///   2. No modular interval overlap across all (d1, d2) buffer instances.
///   3. should_merge cost guard — never inflate memory by merging.
///   4. Cycle-freedom — never introduce a deadlock-prone dependency.
static void mergeNonOverlappingBuffers(ttg::ScheduleLoop &loop) {
  if (loop.II <= 0)
    return;
  computeBufferLifetimes(loop);

  unsigned nextGroupId = 0;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 4> groups;

  for (unsigned i = 0; i < loop.buffers.size(); ++i) {
    auto &buf = loop.buffers[i];
    if (buf.kind == ttg::MemoryKind::BARRIER ||
        buf.kind == ttg::MemoryKind::Register)
      continue;
    // Skip buffers with zero-length lifetime — they have no producer/
    // consumer pattern we can reason about and shouldn't be merged blindly.
    if (buf.liveEnd == buf.liveStart)
      continue;

    bool merged = false;
    for (unsigned g = 0; g < groups.size(); ++g) {
      bool canMerge = true;
      for (unsigned memberIdx : groups[g]) {
        const auto &member = loop.buffers[memberIdx];
        if (member.kind != buf.kind) {
          canMerge = false;
          break;
        }
        // Size-compatibility: the emitter's `reuse=` only works when the
        // secondary buffer has the same allocation footprint (shape, dtype,
        // count). Without this, packing a 32KB buffer into a 16KB slot
        // (e.g., a 128x64 f32 channel into a 64x128 bf16 SMEM region) corrupts
        // memory. Saw exactly this bug with cross-WG channels in case3 FA
        // when split-pipe partitions caused incompatibly-sized buffers to
        // share the same merge group.
        if (member.shape != buf.shape ||
            member.elementBitWidth != buf.elementBitWidth ||
            member.count != buf.count) {
          canMerge = false;
          break;
        }
        if (anyInstancesOverlap(member.liveStart, member.liveEnd, buf.liveStart,
                                buf.liveEnd, member.count, buf.count,
                                loop.II)) {
          canMerge = false;
          break;
        }
      }
      if (!canMerge)
        continue;
      if (!shouldMerge(loop, groups[g], i)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[Step4.5] Skip merge buf" << i << " into group " << g
                   << " (cost guard: would inflate)\n");
        continue;
      }
      if (mergeIntroducesCycle(loop, groups[g], i)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[Step4.5] Skip merge buf" << i << " into group " << g
                   << " (would create dependency cycle)\n");
        continue;
      }
      buf.mergeGroupId = g;
      groups[g].push_back(i);
      merged = true;
      LLVM_DEBUG(llvm::dbgs()
                 << "[Step4.5] Merged buf" << i << " into group " << g
                 << " (live=[" << buf.liveStart << "," << buf.liveEnd << "), "
                 << (buf.kind == ttg::MemoryKind::SMEM ? "SMEM" : "TMEM")
                 << ")\n");
      break;
    }
    if (!merged) {
      buf.mergeGroupId = nextGroupId;
      groups.push_back({i});
      nextGroupId++;
    }
  }

  buildPhysicalBuffers(loop);

  LLVM_DEBUG(llvm::dbgs() << "[Step4.5] " << loop.buffers.size()
                          << " buffers -> " << loop.physicalBuffers.size()
                          << " physical groups\n");
}

// ============================================================================
// Step 4.7: Warp Group Partitioning (latency-aware multi-pipeline clustering)
// ============================================================================

constexpr int kBarrierOverhead = 30;

/// Per-WG warp issue cost added by the barriers each op needs. Each barrier
/// instruction (`barrier_wait`, `barrier_arrive`, `barrier_expect_bytes`,
/// `tcgen05_commit`) is real warp-issued work — not just synchronization. The
/// previous cost model only charged a global `xEdges * 30` term, which under-
/// counted merged-WG layouts: merging MEM into TC removes the cross-WG edge
/// from `xEdges` but the merged warp still issues every barrier itself, so the
/// barrier cost moves from the global penalty into one warp's instruction
/// stream. This helper estimates that per-WG barrier-issue cost so it can be
/// folded into per-WG bottleneck.
///
/// Heuristic per op IN this WG:
///   TMA load (MEM):      +2 barriers (wait_empty + expect_bytes)
///   MMA (TC):            +2 barriers (operand wait_full × 2 typical)
///   tmem_store (CUDA):   +1 barrier  (tcgen05_commit)
/// Plus, for every cross-WG edge with one endpoint in this WG:
///                        +1 barrier  (wait_full on the consumer side or
///                                     barrier_arrive on the producer side
///                                     when not folded into the MMA's
///                                     `mBarriers` HW-arrival list).
static int computeWGBarrierCost(
    const SmallVector<unsigned> &nodeIds, const ttg::ScheduleLoop &loop,
    const llvm::SmallDenseMap<unsigned, int> &nodeToWg, int thisWg) {
  int barriers = 0;
  llvm::SmallDenseSet<unsigned, 8> nodeSet;
  for (unsigned nid : nodeIds)
    nodeSet.insert(nid);
  for (unsigned nid : nodeIds) {
    const auto &n = loop.nodes[nid];
    int freq = std::max(n.frequencyMultiplier, 1);
    if (n.pipeline == ttg::HWPipeline::MEM) {
      barriers += 2 * freq; // wait_empty + expect_bytes
    } else if (n.pipeline == ttg::HWPipeline::TC) {
      barriers += 2 * freq; // operand wait_full × ~2
    } else if (n.op && llvm::StringRef(n.op->getName().getStringRef())
                            .contains("tmem_store")) {
      barriers += 1 * freq; // tcgen05_commit
    }
  }
  // Cross-WG edge: one wait/arrive on each side. Already counted above some
  // of the operand waits for TC; this adds the explicit cross-WG handshake
  // signals (sem*) that aren't naturally charged via op-pipeline counts.
  for (const auto &edge : loop.edges) {
    auto srcIt = nodeToWg.find(edge.srcId);
    auto dstIt = nodeToWg.find(edge.dstId);
    if (srcIt == nodeToWg.end() || dstIt == nodeToWg.end())
      continue;
    if (srcIt->second == dstIt->second)
      continue;
    int freq = std::max(loop.nodes[edge.srcId].frequencyMultiplier,
                        loop.nodes[edge.dstId].frequencyMultiplier);
    if (srcIt->second == thisWg || dstIt->second == thisWg)
      barriers += freq; // 1 instruction on whichever side is `thisWg`
  }
  return barriers * kBarrierOverhead;
}

/// Compute separation cost between each pair of pipelines.
/// Cost = sum of (barrier_overhead / cycle_gap) for each cross-pipeline edge.
/// High cost = tightly coupled (should stay together).
/// Low cost = loosely coupled (safe to separate).
static llvm::DenseMap<std::pair<ttg::HWPipeline, ttg::HWPipeline>, double>
computeSeparationCost(const ttg::ScheduleLoop &loop) {
  llvm::DenseMap<std::pair<ttg::HWPipeline, ttg::HWPipeline>, double> coupling;
  for (const auto &edge : loop.edges) {
    auto pSrc = loop.nodes[edge.srcId].pipeline;
    auto pDst = loop.nodes[edge.dstId].pipeline;
    if (pSrc == pDst || pSrc == ttg::HWPipeline::NONE ||
        pDst == ttg::HWPipeline::NONE)
      continue;
    int cycleGap = loop.nodes[edge.dstId].cycle - loop.nodes[edge.srcId].cycle;
    if (cycleGap <= 0)
      cycleGap = 1;
    coupling[{pSrc, pDst}] +=
        static_cast<double>(kBarrierOverhead) / cycleGap;
  }
  return coupling;
}

/// Per-op effective `selfLatency` when the containing WG has `wgWarps`
/// warps. The base `node.selfLatency` is calibrated for `node.minWarps`
/// (e.g., 4 for SFU/CUDA tile ops). Fewer warps → proportionally more
/// cycles to issue the same total warp-insts:
///
///   effective = base × min_warps / actual_warps        (when actual < min)
///   effective = base                                   (when actual ≥ min)
///
/// Linear scaling (alpha = 1.0) is the analytic assumption from "1 SFU
/// per subpartition, etc.". The actual exponent should be measured per
/// pipe — see study #71. Capping `actual` at `min` avoids over-crediting
/// extra warps beyond the hardware unit count (e.g., 8 warps don't help
/// SFU which is 1 unit per subpartition × 4 subps).
static int effectiveSelfLat(const ttg::ScheduleNode &n, int wgWarps) {
  int base = std::max(n.selfLatency, 1);
  int minW = std::max(n.minWarps, 1);
  int actW = std::max(std::min(wgWarps, minW), 1);
  return base * minW / actW;
}

/// Required WG warp count = max minWarps over the WG's ops. An async-only
/// WG (TMA/MMA, all minWarps=1) gets 1 warp; any WG containing tile-parallel
/// SFU/CUDA/TMEM work needs 4. TLX/Blackwell only allows {1, 2, 4, 8} so we
/// round up.
static int wgRequiredWarps(const SmallVector<unsigned> &nodeIds,
                           const ttg::ScheduleLoop &loop) {
  int m = 1;
  for (unsigned nid : nodeIds)
    m = std::max(m, loop.nodes[nid].minWarps);
  // Snap to TLX-allowed values.
  if (m <= 1) return 1;
  if (m <= 2) return 2;
  if (m <= 4) return 4;
  return 8;
}

/// Compute multi-pipeline makespan via list scheduling.
/// Different pipelines overlap (each tracks its own availability),
/// but data dependencies serialize.
///
/// `wgWarps` is the WG's chosen warp count; per-op `selfLat` scales when
/// the WG has fewer warps than the op's `minWarps` requirement.
static int computeMultiPipelineMakespan(
    const SmallVector<unsigned> &nodeIds, const ttg::ScheduleLoop &loop,
    int wgWarps = 4) {
  llvm::DenseMap<ttg::HWPipeline, int> pipeAvail;
  llvm::DenseMap<unsigned, int> opStart;
  // The warp inside this WG can only issue one instruction at a time. So
  // ALL in-WG ops contend on a single `warpAvail` cursor regardless of
  // which hardware pipeline they target. pipeAvail still tracks per-pipe
  // contention (e.g., two MEM ops can't share the engine), but warpAvail
  // ensures cross-pipe ops within one WG serialize on the warp.
  int warpAvail = 0;

  // Topological order: use the modulo schedule's cycle as proxy.
  SmallVector<unsigned> sorted(nodeIds);
  llvm::sort(sorted, [&](unsigned a, unsigned b) {
    return loop.nodes[a].cycle < loop.nodes[b].cycle;
  });

  for (unsigned nid : sorted) {
    const auto &node = loop.nodes[nid];
    int dataReady = 0;
    for (const auto &edge : loop.edges) {
      if (edge.dstId != nid)
        continue;
      auto it = opStart.find(edge.srcId);
      if (it == opStart.end())
        continue;
      // Use EDGE latency, not source-node latency. The modulo scheduler
      // already uses edge.latency for its placement constraints
      // (consumer.cycle ≥ producer.cycle + edge.latency). Node latency
      // can be a worst-case "result fully available" number (e.g., TMA
      // load's full latency including async overhead) while edge latency
      // captures the actual producer→consumer wait — e.g., edge
      // (descriptor_load → local_alloc).latency = TMA hardware time only,
      // because local_alloc is metadata that sits inline with the SMEM
      // commit. Using node latency here double-counts the async overhead.
      dataReady = std::max(dataReady, it->second + edge.latency);
    }
    int pipeReady = pipeAvail.lookup(node.pipeline);
    int start = std::max({dataReady, pipeReady, warpAvail});
    opStart[nid] = start;
    // Inner-loop ops fire frequencyMultiplier times per outer iter; weight
    // their pipeline occupancy accordingly. For non-flat callers (per-loop
    // makespan in computeResMII) frequencyMultiplier defaults to 1.
    // selfLat scales with the WG's warp count: an op tagged minWarps=4
    // costs ~4× more on a 1-warp WG. See effectiveSelfLat doc comment.
    int dur = effectiveSelfLat(node, wgWarps) * node.frequencyMultiplier;
    pipeAvail[node.pipeline] = start + dur;
    warpAvail = start + dur;
  }

  int makespan = 0;
  for (unsigned nid : sorted) {
    const auto &node = loop.nodes[nid];
    makespan = std::max(makespan,
                        opStart[nid] + effectiveSelfLat(node, wgWarps) *
                                           node.frequencyMultiplier);
  }
  return makespan;
}

/// Candidate warp group for agglomerative clustering.
struct WarpGroupCandidate {
  llvm::SmallDenseSet<ttg::HWPipeline, 4> pipelines;
  SmallVector<unsigned> nodeIds;
  llvm::DenseMap<ttg::HWPipeline, double> util;
};

/// Latency-aware multi-pipeline warp group partitioning.
/// Starts with one group per active pipeline, then greedily merges
/// tightly-coupled pairs validated by multi-pipeline makespan.
static void partitionIntoWarpGroups(ttg::ScheduleLoop &loop) {
  if (loop.II <= 0)
    return;

  auto coupling = computeSeparationCost(loop);

  // Compute per-pipeline utilization.
  llvm::DenseMap<ttg::HWPipeline, double> pipeUtil;
  for (const auto &node : loop.nodes) {
    if (node.pipeline == ttg::HWPipeline::NONE)
      continue;
    pipeUtil[node.pipeline] +=
        static_cast<double>(std::max(node.selfLatency, 1)) / loop.II;
  }

  // Initialize: one candidate group per active pipeline.
  SmallVector<WarpGroupCandidate> groups;
  for (auto pipe : {ttg::HWPipeline::MEM, ttg::HWPipeline::TC,
                    ttg::HWPipeline::CUDA, ttg::HWPipeline::SFU}) {
    double util = pipeUtil.lookup(pipe);
    if (util <= 0.0)
      continue;
    WarpGroupCandidate g;
    g.pipelines.insert(pipe);
    g.util[pipe] = util;
    for (unsigned i = 0; i < loop.nodes.size(); ++i) {
      if (loop.nodes[i].pipeline == pipe)
        g.nodeIds.push_back(i);
    }
    groups.push_back(std::move(g));
  }

  if (groups.size() < 2) {
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.7] < 2 active pipelines, skipping WS\n");
    return;
  }

  // Greedy agglomerative merging.
  while (groups.size() > 1) {
    int bestI = -1, bestJ = -1;
    double bestSavings = 0;

    for (unsigned i = 0; i < groups.size(); ++i) {
      for (unsigned j = i + 1; j < groups.size(); ++j) {
        double savings = 0;
        for (auto p1 : groups[i].pipelines) {
          for (auto p2 : groups[j].pipelines) {
            savings += coupling.lookup({p1, p2}) + coupling.lookup({p2, p1});
          }
        }
        if (savings <= bestSavings)
          continue;

        // Fast reject: check if any pipeline would be oversubscribed.
        bool oversubscribed = false;
        for (auto &[pipe, u] : groups[i].util) {
          double merged = u + groups[j].util.lookup(pipe);
          if (merged > 1.0) {
            oversubscribed = true;
            break;
          }
        }
        if (oversubscribed)
          continue;

        // Precise check: multi-pipeline makespan at the merged WG's
        // required warp count.
        SmallVector<unsigned> mergedNodes;
        mergedNodes.append(groups[i].nodeIds);
        mergedNodes.append(groups[j].nodeIds);
        int mergedWarps = wgRequiredWarps(mergedNodes, loop);
        int makespan = computeMultiPipelineMakespan(mergedNodes, loop,
                                                    mergedWarps);
        if (makespan > loop.II)
          continue;

        bestI = i;
        bestJ = j;
        bestSavings = savings;
      }
    }

    if (bestI < 0)
      break;

    // Execute the merge.
    auto &gi = groups[bestI];
    auto &gj = groups[bestJ];
    for (auto pipe : gj.pipelines)
      gi.pipelines.insert(pipe);
    gi.nodeIds.append(gj.nodeIds);
    for (auto &[pipe, u] : gj.util)
      gi.util[pipe] += u;
    groups.erase(groups.begin() + bestJ);
  }

  // Assign warp group IDs to nodes.
  for (unsigned gid = 0; gid < groups.size(); ++gid) {
    for (unsigned nid : groups[gid].nodeIds)
      loop.nodes[nid].warpGroup = gid;
  }
  // Infrastructure ops (NONE) are assigned later by Step 1.5.
  for (auto &node : loop.nodes) {
    if (node.pipeline == ttg::HWPipeline::NONE)
      node.warpGroup = -1;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[Step4.7] " << groups.size()
                 << " warp groups (II=" << loop.II << "):\n";
    for (unsigned gid = 0; gid < groups.size(); ++gid) {
      llvm::dbgs() << "[Step4.7]   wg" << gid << " {";
      bool first = true;
      for (auto pipe : groups[gid].pipelines) {
        if (!first)
          llvm::dbgs() << "+";
        llvm::dbgs() << ttg::getPipelineName(pipe);
        first = false;
      }
      llvm::dbgs() << "} util=(";
      first = true;
      for (auto &[pipe, u] : groups[gid].util) {
        if (!first)
          llvm::dbgs() << ", ";
        llvm::dbgs() << ttg::getPipelineName(pipe) << "="
                     << llvm::format("%.2f", u);
        first = false;
      }
      llvm::dbgs() << ") ops=" << groups[gid].nodeIds.size() << "\n";
    }
  });
  LLVM_DEBUG({
    for (const auto &node : loop.nodes) {
      llvm::dbgs() << "[PassB.1]   N" << node.id << " "
                   << node.op->getName().getStringRef() << " → wg"
                   << node.warpGroup << " ("
                   << ttg::getPipelineName(node.pipeline) << ")\n";
    }
  });
}

// ============================================================================
// Phase 4: exhaustive enumeration over pipeline partitions + scoring.
//
// Replaces the greedy `partitionIntoWarpGroups` with an exhaustive search
// over the small space of "pipeline-grouping" partitions: for each subset
// of {MEM, TC, CUDA, SFU} that's active, enumerate all ways to assign each
// active pipeline to a warp group (B(4)=15 candidates max). For each
// candidate, score using the latency-aware chain wall + barrier overhead
// + register penalty. Pick the lowest cost.
//
// Why exhaustive works here: the search space is bounded by the number of
// hardware pipelines (≤ 4), not the number of ops. Even if there are
// hundreds of ops in the loop, the partition decision is a bell-number-of-4
// problem at most.
//
// Per-pipeline op-splitting (e.g., putting loadA in one wg and loadB in
// another) is NOT enumerated yet — that's a future extension. For now we
// keep all ops on the same pipeline in the same wg.
//
// Gated by TRITON_MODULO_EXHAUSTIVE_PARTITION=1.
// ============================================================================

// ── Layer C: register-budget aware cost (slack-driven, empirically tuned) ──
//
// Per-WG register footprint = num_warps × 32 threads × per-thread regs,
// bucketed from num_warps to match the TLX emitter:
//   - 1 warp  → 24 regs   (async producer: TMA / MMA)
//   - 4 warps → 152 regs  (tile-parallel compute)
//   - 8 warps → 232 regs  (large compute, rare)
//
// MODEL OF WHAT THE COMPILER DOES (verified empirically — see perf_sweep.py
// in users/wl/wlei/autows/modulo_schedule/tlx_emitter/examples/case3_FA/):
//   1. Each TLX async_task emits `setmaxnreg = num_regs`. Total reg request
//      = Σ (num_warps × 32 × num_regs) across all explicit WGs + default.
//   2. If total ≤ 65,536, all tasks get their full reg request. Fast.
//   3. If total > 65,536, the compiler trims setmaxnreg of SOMEONE.
//      "default" has the most slack (it's typically thin epilogue work that
//      doesn't need its 232-reg request) — so it gets trimmed first, up to
//      ~152 regs/thread of slack (= kDefaultSlack reg-slots).
//   4. If the deficit exceeds default's slack, the COMPUTE WG (which is the
//      bottleneck) starts losing registers too. That's where perf collapses
//      because the compiler then has to recompute live values inside the
//      compute path → many extra cycles per inner-loop iter, NOT spills
//      (nothing in PTX shows st.local; values get re-derived instead).
//
// EMPIRICAL CALIBRATION (perf_sweep.py on B200, FA fwd inner loop):
//   V0  4+4+1+1   total 50,688  deficit  0     residual  0      → 1.0× (baseline)
//   V1  4+4+4+1   total 69,376  deficit  3,840 residual  0      → 1.0× (free)
//   V2  4+4+1+4   total 69,376  deficit  3,840 residual  0      → 1.0× (free)
//   V3  4+4+4+4   total 88,064  deficit 22,528 residual  3,072  → 3.5× slower
//   V4  4+8+1+1   total 90,624  deficit 25,088 residual  5,632  → 9× slower
// Linear penalty `kDeficitPenalty × residual` ≈ 0.5 fits V3; V4's 9× has
// non-linear amplification we approximate but don't capture exactly. Goal
// here is just to push the partitioner away from over-budget plans, not to
// predict perf to ±10%.
constexpr int kBlackwellSMRegs = 65536;
constexpr int kDefaultWGFootprint = 4 * 32 * 232;  // 29,696
// How many regs the default WG can give up before perf hurts.
// = 4 warps × 32 threads × (232 - 80) regs/thread = 19,456.
constexpr int kDefaultSlack = 4 * 32 * (232 - 80);
// Per-reg cost beyond default's slack. Tuned so V3 (~3K residual) gets
// ~1500 cost units of penalty, comparable to V0's bottleneck (~700).
constexpr double kDeficitPenalty = 0.5;
// Tiny negative tie-break: prefer MORE warp groups on exact ties so that
// when two layouts have the same bottleneck (e.g. softmax in FA fwd), the
// one that exposes more cross-pipeline parallelism wins. Positive values
// (5.0/WG was tested) pushed too hard toward 2-WG plans that collapsed
// same-pipe ops; -0.001 only matters when costs are exactly equal.
constexpr double kPerWGTieBreak = -0.001;

static int regsForWarpCount(int numWarps) {
  if (numWarps >= 8) return 232;
  if (numWarps >= 4) return 152;
  return 24;
}
static int wgFootprint(int numWarps) {
  return numWarps * 32 * regsForWarpCount(numWarps);
}

/// A "tight unit" of ops on the same hardware pipeline that should travel
/// together to the same warp group. Built by buildClusters() — two ops on
/// pipeline P are in the same cluster iff there's a path between them in
/// the dep graph where every intermediate node has pipeline NONE (i.e., is
/// an IR-level rename like local_alloc, an arithmetic helper, etc.). This
/// captures "ops connected by same-pipeline dep chains through bookkeeping
/// ops" as one atom, and is what the exhaustive search partitions over.
///
/// Why clusters and not just pipelines:
///   - Lets us split same-pipeline ops onto different WGs when they belong
///     to genuinely independent dep chains (e.g., a K-loop's TMA loads vs
///     a separate epilogue bias load — same MEM pipeline, different chains).
///   - Keeps tightly-coupled ops together so the search doesn't propose
///     splits that just add cross-WG sync without benefit.
struct OpCluster {
  int id{0};
  ttg::HWPipeline pipeline{ttg::HWPipeline::NONE};
  SmallVector<unsigned> nodeIds;
};

struct ClusterAssignment {
  // clusterToWg[clusterId] = wg id (0..numWgs-1).
  SmallVector<int> clusterToWg;
  int numWgs{0};
};

struct ScoredCandidate {
  ClusterAssignment assignment;
  int bottleneckChainWall{0};
  int crossWgEdges{0};
  int totalRegs{0};   // Σ over WGs of num_warps × 32 × regs (incl. default).
  bool feasible{true}; // false = busts SM register budget.
  double cost{0.0};   // +infinity when !feasible (excluded by min-cost pick).
};

/// Greedy agglomerative clustering. For each non-NONE op, BFS through the
/// dep graph (treating it as undirected) expanding through nodes whose
/// pipeline is NONE or matches the start op's pipeline. Other-pipeline
/// nodes block the traversal. All non-NONE ops reached in one BFS are one
/// cluster.
/// Per-op "heavy" classifier: an op is heavy if its pipeline occupancy
/// takes up a meaningful fraction of II. Heavy ops dominate per-WG cost
/// and each deserves its own cluster boundary; light ops merge freely
/// along the dep chain regardless of which pipeline they're on, so chains
/// like FA's softmax `CUDA → SFU → CUDA → SFU → CUDA` stay as ONE cluster.
///
/// Threshold = 20% of II — an op above this is "heavy enough" that
/// putting another op behind it serializes meaningfully.
///
/// Mirrors `pipelineOccupancy()` in DataDependenceGraph.h (which takes a
/// DDGNode); we read the same fields off a ScheduleNode here.
static bool isHeavyOp(const ttg::ScheduleNode &node, int II) {
  if (II <= 0 || node.pipeline == ttg::HWPipeline::NONE)
    return false;
  // Use `latency` (wall-clock to result-ready) for the heavy/light test on
  // ALL pipelines. The heavy/light test asks "does this op contribute
  // meaningful delay to its dep chain?" — that's `latency`, regardless of
  // which pipeline does the work. The previous formula used `selfLatency`
  // for CUDA/SFU on the assumption that those ops fully block the warp —
  // but that mis-classified async-flavored CUDA ops like `ttng.tmem_load`
  // (selfLat=256, lat=532): selfLat × 5 = 1280 < II=1459 looked "light"
  // while the dep edge actually carries 532 cyc, well above 0.2 × II.
  // Using `latency` everywhere lets two independent heavy chains (e.g.
  // softmax's tmem_load(qk) and the FA acc rescale's tmem_load(acc)) sit
  // in their own clusters instead of being pre-merged at clustering time.
  int occ = std::max(node.latency, 1);
  return occ * 5 >= II; // occ / II >= 0.2
}

static SmallVector<OpCluster>
buildClusters(const ttg::ScheduleLoop &loop) {
  unsigned n = loop.nodes.size();
  SmallVector<SmallVector<unsigned>> adj(n);
  for (const auto &e : loop.edges) {
    adj[e.srcId].push_back(e.dstId);
    adj[e.dstId].push_back(e.srcId);
  }
  // Pre-classify each op as heavy or light (per-op, not per-pipeline).
  SmallVector<bool> heavy(n, false);
  for (unsigned i = 0; i < n; ++i)
    heavy[i] = isHeavyOp(loop.nodes[i], loop.II);

  SmallVector<OpCluster> clusters;
  llvm::SmallDenseMap<unsigned, int> nodeToCluster;
  // Process HEAVY ops first as cluster anchors. Without this, a light BFS
  // started from one heavy op's downstream chain can claim nodes that
  // belong to a *different* heavy op's chain (because light-BFS expands
  // through any non-heavy node and the visit order is otherwise node-id
  // order). Pre-anchoring heavy ops ensures each heavy op claims its
  // same-pipeline descendants before light-BFS can absorb them — e.g. in
  // FA fwd, the rescale chain (tmem_load(acc) → mul → tmem_store(acc))
  // follows its tmem_load anchor instead of getting glued to the softmax
  // chain. Keeping the chain together avoids a 64 KB SMEM staging buffer
  // for the register-typed acc value that would otherwise cross WGs.
  SmallVector<unsigned> startOrder;
  for (unsigned i = 0; i < n; ++i)
    if (heavy[i]) startOrder.push_back(i);
  for (unsigned i = 0; i < n; ++i)
    if (!heavy[i]) startOrder.push_back(i);
  for (unsigned start : startOrder) {
    if (nodeToCluster.count(start))
      continue;
    auto startPipe = loop.nodes[start].pipeline;
    if (startPipe == ttg::HWPipeline::NONE)
      continue;

    OpCluster cluster;
    cluster.id = clusters.size();
    cluster.pipeline = startPipe;
    bool startIsHeavy = heavy[start];

    SmallVector<bool> visited(n, false);
    SmallVector<unsigned> stack;
    visited[start] = true;
    stack.push_back(start);
    while (!stack.empty()) {
      unsigned u = stack.pop_back_val();
      auto upipe = loop.nodes[u].pipeline;
      // Membership rule:
      //   Heavy start — only same-pipeline ops join (a TC MMA cluster
      //   accumulates only TC ops; MEM the same).
      //   Light start — any other LIGHT op joins, regardless of pipeline.
      //   Keeps softmax-style chains intact across SFU↔CUDA transitions.
      bool joins = startIsHeavy ? (upipe == startPipe) : !heavy[u];
      if (joins && upipe != ttg::HWPipeline::NONE && !nodeToCluster.count(u)) {
        cluster.nodeIds.push_back(u);
        nodeToCluster[u] = cluster.id;
      }
      for (unsigned v : adj[u]) {
        if (visited[v])
          continue;
        auto vpipe = loop.nodes[v].pipeline;
        // Expansion rule:
        //   Heavy start: expand only through NONE or same-pipeline ops.
        //   Light start: expand through any non-heavy op (NONE/light).
        //                Heavy ops on any pipeline block the traversal.
        bool expand = startIsHeavy
                          ? (vpipe == ttg::HWPipeline::NONE || vpipe == startPipe)
                          : !heavy[v];
        if (expand) {
          visited[v] = true;
          stack.push_back(v);
        }
      }
    }
    if (!cluster.nodeIds.empty())
      clusters.push_back(std::move(cluster));
  }

  // Post-pass: absorb "negligible" clusters into the most-connected
  // neighbor. A cluster is negligible if its total occupancy is below
  // kAbsorbThreshold — e.g. a 2-op scalar arith chain (total = 2 cyc)
  // shouldn't get its own slot in the WG partition, but the softmax chain
  // (~1300 cyc total even though each op is light) should remain its own
  // cluster. Heavy clusters always stay (BFS only put one heavy op in each
  // cluster, so total ≥ that op's heavy occupancy ≥ 0.2*II).
  constexpr int kAbsorbThreshold = 100;
  auto totalOccupancy = [&](const OpCluster &c) {
    int sum = 0;
    for (unsigned nid : c.nodeIds) {
      const auto &n = loop.nodes[nid];
      int occ = (n.pipeline == ttg::HWPipeline::MEM ||
                 n.pipeline == ttg::HWPipeline::TC)
                    ? std::max(n.latency, 1)
                    : std::max(n.selfLatency, 1);
      sum += occ;
    }
    return sum;
  };
  // Iterate until no more absorptions happen (a chain of small clusters
  // might absorb cascadingly).
  bool changed = true;
  while (changed) {
    changed = false;
    for (unsigned ci = 0; ci < clusters.size(); ++ci) {
      auto &c = clusters[ci];
      if (c.nodeIds.empty())
        continue;
      if (totalOccupancy(c) >= kAbsorbThreshold)
        continue;
      // Count edges from c's nodes to every other cluster.
      llvm::DenseMap<int, int> nbrCount;
      for (unsigned nid : c.nodeIds) {
        for (unsigned v : adj[nid]) {
          auto it = nodeToCluster.find(v);
          if (it == nodeToCluster.end() || it->second == (int)ci)
            continue;
          nbrCount[it->second] += 1;
        }
      }
      if (nbrCount.empty())
        continue; // isolated cluster — leave it alone
      int target = -1, bestCount = 0;
      for (auto &[cid, cnt] : nbrCount) {
        if (cnt > bestCount) {
          bestCount = cnt;
          target = cid;
        }
      }
      if (target < 0)
        continue;
      // Merge c into clusters[target].
      auto &dst = clusters[target];
      for (unsigned nid : c.nodeIds) {
        dst.nodeIds.push_back(nid);
        nodeToCluster[nid] = target;
      }
      c.nodeIds.clear();
      changed = true;
    }
  }
  // Compact: drop empty clusters and reassign IDs.
  SmallVector<OpCluster> compact;
  for (auto &c : clusters) {
    if (c.nodeIds.empty())
      continue;
    c.id = compact.size();
    for (unsigned nid : c.nodeIds)
      nodeToCluster[nid] = c.id;
    compact.push_back(std::move(c));
  }
  // Dump for diagnostics: each cluster's ops with their per-op cost so we can
  // reason about which clustering decisions are reasonable.
  LLVM_DEBUG({
    int heavyThr = std::max(1, loop.II / 5);  // 0.2*II — same as isHeavyOp
    for (const auto &c : compact) {
      int totalSelf = 0, totalLat = 0;
      for (unsigned nid : c.nodeIds) {
        totalSelf += std::max(loop.nodes[nid].selfLatency, 0);
        totalLat += std::max(loop.nodes[nid].latency, 0);
      }
      llvm::dbgs() << "[buildClusters] C" << c.id << " ("
                   << ttg::getPipelineName(c.pipeline)
                   << ") totalSelfLat=" << totalSelf
                   << " totalLat=" << totalLat << " (heavyThr=" << heavyThr
                   << ")\n";
      for (unsigned nid : c.nodeIds) {
        const auto &n = loop.nodes[nid];
        bool isHeavy = isHeavyOp(n, loop.II);
        llvm::StringRef opName = n.op ? n.op->getName().getStringRef()
                                      : llvm::StringRef("?");
        llvm::dbgs() << "  N" << nid << "  " << opName
                     << "  pipe=" << ttg::getPipelineName(n.pipeline)
                     << "  selfLat=" << n.selfLatency
                     << "  lat=" << n.latency
                     << (isHeavy ? "  [HEAVY]" : "  [light]")
                     << "  cyc=" << n.cycle << " stage=" << n.stage << "\n";
      }
    }
  });
  return compact;
}

/// Enumerate all set partitions of `numClusters` items in canonical form
/// (first cluster always gets wg 0; each subsequent cluster either reuses
/// an existing wg id or opens the next one). Total count = Bell(numClusters).
static SmallVector<ClusterAssignment>
enumerateClusterPartitions(int numClusters) {
  SmallVector<ClusterAssignment> result;
  ClusterAssignment current;
  current.clusterToWg.resize(numClusters, -1);
  std::function<void(int, int)> recurse = [&](int idx, int maxUsedWg) {
    if (idx == numClusters) {
      current.numWgs = maxUsedWg + 1;
      result.push_back(current);
      return;
    }
    for (int wg = 0; wg <= maxUsedWg + 1; ++wg) {
      current.clusterToWg[idx] = wg;
      recurse(idx + 1, std::max(maxUsedWg, wg));
    }
  };
  recurse(0, -1);
  return result;
}

static ScoredCandidate scoreCandidate(const ClusterAssignment &assn,
                                      const SmallVector<OpCluster> &clusters,
                                      const ttg::ScheduleLoop &loop,
                                      bool verbose = false) {
  // Group nodes by warp group via cluster → wg mapping.
  llvm::DenseMap<int, SmallVector<unsigned>> wgToNodes;
  llvm::SmallDenseMap<unsigned, int> nodeToWg;
  for (const auto &c : clusters) {
    int wg = assn.clusterToWg[c.id];
    for (unsigned nid : c.nodeIds) {
      wgToNodes[wg].push_back(nid);
      nodeToWg[nid] = wg;
    }
  }

  // Bottleneck = max over wgs of multi-pipeline makespan.
  // Each WG's makespan now depends on its chosen warp count: ops with
  // minWarps=4 (SFU/CUDA tile) cost more if the WG only gets 1 warp.
  // Layer C: also accumulate per-WG register footprint.
  int bottleneck = 0;
  int totalRegs = kDefaultWGFootprint;  // include implicit "default" WG
  for (auto &[wgId, nodes] : wgToNodes) {
    int wgWarps = wgRequiredWarps(nodes, loop);
    int ms = computeMultiPipelineMakespan(nodes, loop, wgWarps);
    int barCost = computeWGBarrierCost(nodes, loop, nodeToWg, wgId);
    int wgCost = ms + barCost;
    int fp = wgFootprint(wgWarps);
    totalRegs += fp;
    if (verbose) {
      LLVM_DEBUG({
        llvm::dbgs() << "[Score-VERBOSE]   wg" << wgId << " nodes=[";
        for (size_t i = 0; i < nodes.size(); ++i) {
          if (i)
            llvm::dbgs() << ",";
          llvm::dbgs() << "N" << nodes[i];
        }
        llvm::dbgs() << "] warps=" << wgWarps << " regs=" << fp
                     << " makespan=" << ms << " barCost=" << barCost
                     << " wgCost=" << wgCost << "\n";
      });
    }
    bottleneck = std::max(bottleneck, wgCost);
  }

  // Cross-wg edges (cost of barriers). Only count edges where both endpoints
  // are non-NONE (NONE ops aren't in any cluster). A cross-wg dep inside the
  // K-loop fires K times per outer iter, so weight by max of the endpoints'
  // frequency multipliers.
  int crossEdges = 0;
  for (const auto &edge : loop.edges) {
    auto srcIt = nodeToWg.find(edge.srcId);
    auto dstIt = nodeToWg.find(edge.dstId);
    if (srcIt == nodeToWg.end() || dstIt == nodeToWg.end())
      continue;
    if (srcIt->second != dstIt->second) {
      int freq = std::max(loop.nodes[edge.srcId].frequencyMultiplier,
                          loop.nodes[edge.dstId].frequencyMultiplier);
      crossEdges += std::max(freq, 1);
    }
  }

  // Pipeline-via-WG-separation workaround for missing software pipeline
  // lowering (see notes/pipeline_via_wg_separation.md). Penalize WGs
  // that mix SYNC-blocking work (CUDA pipe ops — the warp is stalled
  // for their full selfLatency) with ASYNC work (TC MMA, MEM TMA — the
  // warp issues and the HW completes in the background). When mixed in
  // one WG, the sync work BLOCKS the warp so the async ops can't be
  // issued back-to-back and their HW concurrency is lost. Splitting
  // them into separate WGs connected by barriers lets the async WG keep
  // issuing while the sync WG runs — the pipelined overlap we'd get
  // from a proper prologue/main/epilogue rewrite.
  //
  // Penalty per mixed WG = min(syncWork, asyncWork). That's the cycles
  // of overlap we lose by keeping them together.
  //
  // Pure-async WGs (load + MMA, MMA + MMA) get NO penalty — async ops
  // mixed in one WG already overlap via HW. Pure-sync WGs (softmax
  // chain) also get no penalty — there's no async to overlap with.
  //
  // Gated by `TRITON_MODULO_STAGE_SEPARATION=1`; disable once proper
  // `pipelineForLoop()` integration lands.
  int stageMixPenalty = 0;
  if (triton::tools::getBoolEnv("TRITON_MODULO_STAGE_SEPARATION")) {
    constexpr int kSyncThreshold = 64; // ignore tiny CUDA ops
    for (auto &[wgId, nodes] : wgToNodes) {
      int syncWork = 0;  // sum of CUDA selfLat (warp-blocking)
      int asyncWork = 0; // sum of TC/MEM pipeline occupancy
                         // (HW-overlappable per-iter work)
      for (unsigned nid : nodes) {
        const auto &n = loop.nodes[nid];
        if (n.pipeline == ttg::HWPipeline::CUDA) {
          syncWork += std::max(n.selfLatency, 0);
        } else if (n.pipeline == ttg::HWPipeline::TC ||
                   n.pipeline == ttg::HWPipeline::MEM) {
          asyncWork += std::max(n.latency, 0);
        }
      }
      if (syncWork < kSyncThreshold)
        continue; // not enough sync work to bother
      if (asyncWork == 0)
        continue; // nothing async to overlap with
      stageMixPenalty += std::min(syncWork, asyncWork);
    }
  }

  ScoredCandidate sc;
  sc.assignment = assn;
  sc.bottleneckChainWall = bottleneck;
  sc.crossWgEdges = crossEdges;
  sc.totalRegs = totalRegs;
  // Slack-aware soft penalty (see kDefaultSlack / kDeficitPenalty doc).
  // When the request exceeds the SM budget, the default WG absorbs up to
  // kDefaultSlack regs; only the residual hurts the bottleneck compute WG
  // (and that's where 3-9× perf collapses observed in perf_sweep.py).
  int deficit = std::max(0, totalRegs - kBlackwellSMRegs);
  int residual = std::max(0, deficit - kDefaultSlack);
  sc.feasible = (residual == 0);
  // Cross-WG barrier issue cost is now folded into per-WG bottleneck via
  // `computeWGBarrierCost`, so the legacy `crossEdges * kBarrierOverhead`
  // global term is dropped — keeping it would double-charge the same
  // barriers.
  sc.cost = static_cast<double>(bottleneck) +
            residual * kDeficitPenalty +
            assn.numWgs * kPerWGTieBreak +
            stageMixPenalty;
  return sc;
}

/// Format a candidate's `cluster→wg` assignment for log lines.
static void printAssignment(llvm::raw_ostream &os,
                            const ClusterAssignment &assn,
                            const SmallVector<OpCluster> &clusters) {
  os << "{";
  bool first = true;
  for (const auto &c : clusters) {
    if (!first)
      os << ", ";
    os << "C" << c.id << "(" << ttg::getPipelineName(c.pipeline) << ")→wg"
       << assn.clusterToWg[c.id];
    first = false;
  }
  os << "}";
}

// ============================================================================
// Greedy cluster partitioner — agglomerative O(N³) alternative to exhaustive.
//
// Used as a fallback when the cluster count exceeds the exhaustive cap.
// Identical scoring to partitionExhaustive (cost = bottleneck + xEdges*30 +
// numWgs*1) so the two paths are directly comparable.
//
// Algorithm:
//   1. Init: one warp group per cluster (max splitting).
//   2. Loop: for every pair (i,j) of WGs, evaluate the cost AFTER merging
//      them. Pick the merge with the largest cost reduction.
//   3. Stop when no merge reduces cost.
//
// Per-iter work is O(N² × (E + N²)) for evaluating each merge candidate;
// for N≤30 clusters this is well under a millisecond, vs Bell(N) factorial
// growth for exhaustive.
// ============================================================================

/// One candidate warp group during greedy partitioning.
struct GreedyWG {
  SmallVector<unsigned, 4> clusterIds; // member cluster ids
  SmallVector<unsigned> nodeIds;       // flattened nodeIds (cached for makespan)
};

/// Compute the cost of a partition. Mirrors `scoreCandidate`'s formula.
/// Returns +inf if the partition busts the SM register budget (Layer C).
static double evalGreedyCost(const SmallVector<GreedyWG> &wgs,
                             const ttg::ScheduleLoop &loop) {
  // Bottleneck = max per-WG makespan, computed at each WG's required warps.
  // Also accumulate per-WG register footprint (Layer C).
  int totalRegs = kDefaultWGFootprint;  // include implicit "default" WG
  llvm::SmallDenseMap<unsigned, int> nodeToWg;
  for (unsigned wgi = 0; wgi < wgs.size(); ++wgi) {
    for (unsigned nid : wgs[wgi].nodeIds)
      nodeToWg[nid] = static_cast<int>(wgi);
  }
  int bottleneck = 0;
  for (unsigned wgi = 0; wgi < wgs.size(); ++wgi) {
    int wgWarps = wgRequiredWarps(wgs[wgi].nodeIds, loop);
    int ms = computeMultiPipelineMakespan(wgs[wgi].nodeIds, loop, wgWarps);
    int barCost = computeWGBarrierCost(wgs[wgi].nodeIds, loop, nodeToWg,
                                        static_cast<int>(wgi));
    bottleneck = std::max(bottleneck, ms + barCost);
    totalRegs += wgFootprint(wgWarps);
  }
  int deficit = std::max(0, totalRegs - kBlackwellSMRegs);
  int residual = std::max(0, deficit - kDefaultSlack);
  // Cross-WG barrier-issue cost is folded into per-WG bottleneck via
  // `computeWGBarrierCost`; the legacy global `crossEdges * kBarrierOverhead`
  // term has been dropped to avoid double-charging.
  return static_cast<double>(bottleneck) +
         residual * kDeficitPenalty +
         wgs.size() * kPerWGTieBreak;
}

/// Greedy cluster-based partitioner.
static void partitionClusterGreedy(ttg::ScheduleLoop &loop) {
  if (loop.II <= 0)
    return;

  auto clusters = buildClusters(loop);
  if (clusters.size() < 2) {
    for (auto &node : loop.nodes)
      node.warpGroup =
          (node.pipeline == ttg::HWPipeline::NONE) ? -1 : 0;
    return;
  }

  // Init: one WG per cluster (most-split state).
  SmallVector<GreedyWG> wgs;
  for (const auto &c : clusters) {
    GreedyWG wg;
    wg.clusterIds.push_back(c.id);
    wg.nodeIds = c.nodeIds;
    wgs.push_back(std::move(wg));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[Greedy] Init: " << wgs.size()
                 << " WGs (one per cluster)\n";
    for (const auto &c : clusters) {
      llvm::dbgs() << "[Greedy]   C" << c.id << " ("
                   << ttg::getPipelineName(c.pipeline)
                   << ") size=" << c.nodeIds.size() << "\n";
    }
  });
  double currentCost = evalGreedyCost(wgs, loop);
  LLVM_DEBUG(llvm::dbgs() << "[Greedy] Initial cost = " << currentCost << "\n");

  // Greedy merge loop.
  int iter = 0;
  while (wgs.size() > 1) {
    int bestI = -1, bestJ = -1;
    double bestCost = currentCost;

    for (unsigned i = 0; i < wgs.size(); ++i) {
      for (unsigned j = i + 1; j < wgs.size(); ++j) {
        // Trial merge of wgs[i] and wgs[j].
        SmallVector<GreedyWG> trial = wgs;
        trial[i].clusterIds.append(trial[j].clusterIds.begin(),
                                   trial[j].clusterIds.end());
        trial[i].nodeIds.append(trial[j].nodeIds.begin(),
                                trial[j].nodeIds.end());
        trial.erase(trial.begin() + j);
        double trialCost = evalGreedyCost(trial, loop);
        if (trialCost < bestCost) {
          bestCost = trialCost;
          bestI = i;
          bestJ = j;
        }
      }
    }

    if (bestI < 0)
      break; // no merge reduces cost — local optimum reached

    LLVM_DEBUG(llvm::dbgs() << "[Greedy] iter " << iter << ": merge wg"
                            << bestI << " ⊕ wg" << bestJ << ", cost "
                            << currentCost << " → " << bestCost << "\n");
    wgs[bestI].clusterIds.append(wgs[bestJ].clusterIds.begin(),
                                 wgs[bestJ].clusterIds.end());
    wgs[bestI].nodeIds.append(wgs[bestJ].nodeIds.begin(),
                              wgs[bestJ].nodeIds.end());
    wgs.erase(wgs.begin() + bestJ);
    currentCost = bestCost;
    ++iter;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[Greedy] Final: " << wgs.size() << " WGs, cost="
                 << currentCost << "\n";
    for (unsigned wi = 0; wi < wgs.size(); ++wi) {
      llvm::dbgs() << "[Greedy]   wg" << wi << " clusters=[";
      for (size_t k = 0; k < wgs[wi].clusterIds.size(); ++k) {
        if (k)
          llvm::dbgs() << ",";
        llvm::dbgs() << "C" << wgs[wi].clusterIds[k];
      }
      llvm::dbgs() << "]\n";
    }
  });

  // Apply final assignment.
  for (auto &node : loop.nodes)
    node.warpGroup = -1;
  for (unsigned wgi = 0; wgi < wgs.size(); ++wgi) {
    for (unsigned nid : wgs[wgi].nodeIds)
      loop.nodes[nid].warpGroup = wgi;
  }
  LLVM_DEBUG({
    for (const auto &node : loop.nodes) {
      llvm::dbgs() << "[PassB.1]   N" << node.id << " "
                   << node.op->getName().getStringRef() << " → wg"
                   << node.warpGroup << " ("
                   << ttg::getPipelineName(node.pipeline) << ")\n";
    }
  });
}

/// Exhaustive partition pass: greedy agglomerative clustering followed by
/// exhaustive enumeration over the resulting clusters. Each cluster is one
/// "atom" in the partition decision; ops within a cluster always travel to
/// the same WG, but different clusters on the same pipeline can be split
/// onto different WGs.
static void partitionExhaustive(ttg::ScheduleLoop &loop) {
  if (loop.II <= 0)
    return;

  // ── Greedy agglomerative clustering ──────────────────────────────────────
  auto clusters = buildClusters(loop);
  if (clusters.size() < 2) {
    LLVM_DEBUG(llvm::dbgs() << "[Phase4] < 2 clusters, skipping\n");
    // Single cluster (or none): one WG, set everything to wg0.
    for (auto &node : loop.nodes)
      node.warpGroup =
          (node.pipeline == ttg::HWPipeline::NONE) ? -1 : 0;
    return;
  }

  // Safety cap: Bell numbers blow up fast (B(10)=115975, B(12)=4M+).
  // For N clusters each candidate also runs computeMultiPipelineMakespan
  // per WG, so total work is O(B(N) * N * |nodes|^2). At N=10 this is a
  // few seconds; at N=12 it's minutes. Fall back to the cluster-greedy
  // partitioner past the cap (same scoring, O(N^3) vs Bell(N) growth).
  // TRITON_MODULO_CLUSTER_GREEDY=1 forces greedy at any cluster count.
  constexpr unsigned kMaxClustersForExhaustive = 10;
  bool forceGreedy =
      triton::tools::getBoolEnv("TRITON_MODULO_CLUSTER_GREEDY");
  if (forceGreedy || clusters.size() > kMaxClustersForExhaustive) {
    LLVM_DEBUG(llvm::dbgs() << "[Phase4] " << clusters.size() << " clusters"
                            << (forceGreedy ? " (forced)" : " > 10")
                            << " — using cluster-greedy\n");
    partitionClusterGreedy(loop);
    return;
  }

  // ── Logging: dump nodes/edges/clusters before enumerating ────────────────
  LLVM_DEBUG({
    llvm::dbgs() << "[Phase4-VERBOSE] LOOP NODES:\n";
    for (const auto &node : loop.nodes) {
      llvm::dbgs() << "[Phase4-VERBOSE]   N" << node.id
                   << "  cycle=" << node.cycle
                   << "  pipe=" << ttg::getPipelineName(node.pipeline)
                   << "  mw=" << node.minWarps
                   << "  selfLat=" << node.selfLatency
                   << "  lat=" << node.latency;
      if (node.op)
        llvm::dbgs() << "  " << node.op->getName().getStringRef();
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "[Phase4-VERBOSE] LOOP EDGES:\n";
    for (const auto &edge : loop.edges) {
      llvm::dbgs() << "[Phase4-VERBOSE]   E N" << edge.srcId << "(pipe="
                   << ttg::getPipelineName(loop.nodes[edge.srcId].pipeline)
                   << ") -> N" << edge.dstId << "(pipe="
                   << ttg::getPipelineName(loop.nodes[edge.dstId].pipeline)
                   << ")  edge.lat=" << edge.latency
                   << "  dist=" << edge.distance << "\n";
    }
    llvm::dbgs() << "[Phase4-VERBOSE] CLUSTERS: " << clusters.size() << "\n";
    for (const auto &c : clusters) {
      llvm::dbgs() << "[Phase4-VERBOSE]   C" << c.id << " ("
                   << ttg::getPipelineName(c.pipeline)
                   << ") size=" << c.nodeIds.size() << ":\n";
      for (unsigned nid : c.nodeIds) {
        const auto &n = loop.nodes[nid];
        llvm::dbgs() << "[Phase4-VERBOSE]       N" << nid;
        if (n.op)
          llvm::dbgs() << " " << n.op->getName().getStringRef();
        llvm::dbgs() << "  (cycle=" << n.cycle << " selfLat=" << n.selfLatency
                     << " lat=" << n.latency << ")\n";
      }
    }
  });

  // ── Enumerate + score ────────────────────────────────────────────────────
  auto candidates = enumerateClusterPartitions(clusters.size());
  SmallVector<ScoredCandidate> scored;
  LLVM_DEBUG(llvm::dbgs() << "[Phase4-VERBOSE] II=" << loop.II
                          << "  numClusters=" << clusters.size()
                          << "  numCandidates=" << candidates.size() << "\n");
  for (unsigned ci = 0; ci < candidates.size(); ++ci) {
    auto sc = scoreCandidate(candidates[ci], clusters, loop, /*verbose=*/true);
    scored.push_back(sc);
    LLVM_DEBUG({
      llvm::dbgs() << "[Phase4-VERBOSE] cand[" << ci << "] ";
      printAssignment(llvm::dbgs(), sc.assignment, clusters);
      llvm::dbgs() << "  numWgs=" << sc.assignment.numWgs
                   << "  bottleneck=" << sc.bottleneckChainWall
                   << "  xEdges=" << sc.crossWgEdges << "  cost=" << sc.cost
                   << "\n";
    });
  }

  // ── Pick best ────────────────────────────────────────────────────────────
  llvm::sort(scored, [](const ScoredCandidate &a, const ScoredCandidate &b) {
    return a.cost < b.cost;
  });
  const auto &winner = scored.front();

  // Reset NONE ops to -1; cluster members get the winning WG. propagateWarp-
  // GroupToInfraOps later attaches NONE ops to their consumer's WG.
  for (auto &node : loop.nodes)
    node.warpGroup = -1;
  for (const auto &c : clusters) {
    int wg = winner.assignment.clusterToWg[c.id];
    for (unsigned nid : c.nodeIds)
      loop.nodes[nid].warpGroup = wg;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[Phase4-VERBOSE] Top-3 ranked:\n";
    for (size_t k = 0; k < std::min<size_t>(3, scored.size()); ++k) {
      const auto &c = scored[k];
      llvm::dbgs() << "[Phase4-VERBOSE]  rank " << (k + 1)
                   << ":  cost=" << c.cost
                   << " (bottleneck=" << c.bottleneckChainWall
                   << " xEdges=" << c.crossWgEdges
                   << " wgs=" << c.assignment.numWgs << ") ";
      printAssignment(llvm::dbgs(), c.assignment, clusters);
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "[Phase4] picked: cost=" << winner.cost
                 << " bottleneck=" << winner.bottleneckChainWall
                 << " xEdges=" << winner.crossWgEdges
                 << " wgs=" << winner.assignment.numWgs << "\n";
    for (const auto &node : loop.nodes) {
      llvm::dbgs() << "[PassB.1]   N" << node.id << " "
                   << node.op->getName().getStringRef() << " → wg"
                   << node.warpGroup << " ("
                   << ttg::getPipelineName(node.pipeline) << ")\n";
    }
  });
}

/// Step 1.5: Propagate warp group to infrastructure ops (pipeline=NONE).
/// If all consumers are in the same group → assign to that group.
/// If consumers span groups → mark as -2 (replicated to all groups).
/// The actual cloning happens in the downstream WSCodePartition pass.
static constexpr int kReplicatedWarpGroup = -2;

/// Post-Phase4 reassignment: scalar arith / math ops (e.g.
/// `arith.muli %k, 64` computing `offs_k`) are never anchors. The
/// CLUSTER BUILDER and Phase4 COST MODEL still see them, so cluster
/// shapes and partition decisions stay byte-identical to today. AFTER
/// Phase4 picks, we strip these scalar arith ops back to `wg = -1` so
/// `propagateWarpGroupToInfraOps` reclassifies them as infra:
/// absorbed (single consumer) or replicated (multiple). The emitter's
/// `_collect_infra_deps_recursive` then inlines them per-task, and
/// `insertCrossGroupBarriers` skips edges touching them (the
/// `src.warpGroup < 0` guard) — eliminating spurious
/// single-arrive-no-empty barriers like case2's `sem1_full`
/// (`arith.muli → tt.descriptor_load`) and case3's `sem0_full`
/// (`arith.addi → tt.descriptor_load`). Mirrors Meta's
/// `TaskIdBackwardPropagation` (`isScalarArithOrMath`) treatment.
static void demoteScalarArithToInfra(ttg::ScheduleLoop &loop) {
  for (auto &node : loop.nodes) {
    if (!node.op)
      continue;
    if (node.pipeline != ttg::HWPipeline::CUDA)
      continue;
    StringRef dialect = node.op->getDialect()->getNamespace();
    if (dialect != "arith" && dialect != "math")
      continue;
    // Tensor-result arith/math is real compute (extf, mulf, addf,
    // truncf, ...) — those ARE anchors.
    bool isScalar = true;
    for (auto t : node.op->getResultTypes())
      if (isa<RankedTensorType>(t)) {
        isScalar = false;
        break;
      }
    if (!isScalar)
      continue;
    node.warpGroup = -1; // hand off to propagateWarpGroupToInfraOps
  }
}

static void propagateWarpGroupToInfraOps(ttg::ScheduleLoop &loop) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &node : loop.nodes) {
      if (node.warpGroup != -1)
        continue; // Already assigned or replicated.

      // Collect warp groups of all consumers (outgoing edges).
      llvm::SmallSetVector<int, 4> consumerGroups;
      for (const auto &edge : loop.edges) {
        if (edge.srcId != node.id)
          continue;
        int cg = loop.nodes[edge.dstId].warpGroup;
        if (cg >= 0)
          consumerGroups.insert(cg);
      }

      if (consumerGroups.empty())
        continue; // No assigned consumers yet.

      if (consumerGroups.size() == 1) {
        node.warpGroup = *consumerGroups.begin();
      } else {
        node.warpGroup = kReplicatedWarpGroup; // Replicated.
      }
      changed = true;
    }
  }

  LLVM_DEBUG({
    int unassigned = 0, replicated = 0;
    for (const auto &node : loop.nodes) {
      if (node.warpGroup == -1)
        unassigned++;
      if (node.warpGroup == kReplicatedWarpGroup)
        replicated++;
    }
    llvm::dbgs() << "[PassB.1.5] Infra ops: " << replicated << " replicated, "
                 << unassigned << " unassigned\n";
  });
}

// ============================================================================
// Pass B Step 2: Insert Synchronization
// ============================================================================

/// Identify cross-group edges and insert barrier records.
/// SMEM transfers (MEM→TC) → mbarrier with phase cycling.
/// TMEM transfers (TC→CUDA) → named barrier.
static void insertCrossGroupBarriers(ttg::ScheduleLoop &loop) {
  llvm::DenseSet<std::pair<unsigned, unsigned>> seenBarrierPairs;
  for (const auto &edge : loop.edges) {
    const auto &src = loop.nodes[edge.srcId];
    const auto &dst = loop.nodes[edge.dstId];

    // Skip intra-group edges.
    if (src.warpGroup < 0 || dst.warpGroup < 0)
      continue;
    if (src.warpGroup == dst.warpGroup)
      continue;

    // Determine barrier kind from the producer/consumer pipeline.
    ttg::ScheduleLoop::BarrierKind kind;
    if (src.pipeline == ttg::HWPipeline::MEM) {
      // MEM → TC/CUDA: data flows through SMEM → use mbarrier.
      kind = ttg::ScheduleLoop::BarrierKind::MBARRIER;
    } else if (src.pipeline == ttg::HWPipeline::TC) {
      // TC → CUDA: data flows through TMEM → use named barrier.
      kind = ttg::ScheduleLoop::BarrierKind::NAMED;
    } else {
      // Other cross-group edges: default to mbarrier.
      kind = ttg::ScheduleLoop::BarrierKind::MBARRIER;
    }

    // Find the paired data buffer (if any) to determine depth.
    unsigned depth = 1;
    unsigned pairedBuf = UINT_MAX;
    if (src.producesBuffer != UINT_MAX) {
      pairedBuf = src.producesBuffer;
      depth = loop.buffers[pairedBuf].count;
    } else if (src.pipeline == ttg::HWPipeline::MEM && src.op) {
      // TMA load → memdesc-rebind chain (local_alloc, memdesc_trans,
      // memdesc_subview) → ... lowering step: the load's data lands in the
      // SMEM region the downstream `local_alloc` defines. There's no
      // register intermediate — the load writes directly to that SMEM (TMA
      // hardware path). So instead of synthesizing a separate staging
      // buffer for this cross-WG edge (which would double the SMEM cost
      // for K/V tiles when MEM and TC end up in different WGs), walk
      // through the metadata-rebind chain to find the downstream node that
      // owns the actual data buffer, and reuse it. The TMA's completion
      // barrier is the same mbarrier that fires when the destination SMEM
      // is full — no extra buffer required.
      llvm::SmallDenseSet<unsigned, 4> seen;
      seen.insert(edge.srcId);
      SmallVector<unsigned, 4> stack;
      stack.push_back(edge.srcId);
      while (!stack.empty() && pairedBuf == UINT_MAX) {
        unsigned cur = stack.pop_back_val();
        for (const auto &e : loop.edges) {
          if (e.srcId != cur || !seen.insert(e.dstId).second)
            continue;
          const auto &dn = loop.nodes[e.dstId];
          if (dn.producesBuffer != UINT_MAX &&
              loop.buffers[dn.producesBuffer].kind == ttg::MemoryKind::SMEM) {
            pairedBuf = dn.producesBuffer;
            depth = loop.buffers[pairedBuf].count;
            break;
          }
          // Continue walking through metadata-rebind ops (zero-latency
          // NONE pipeline whose result is a memdesc).
          if (dn.op && dn.op->getNumResults() == 1 &&
              isa<ttg::MemDescType>(dn.op->getResult(0).getType())) {
            stack.push_back(e.dstId);
          }
        }
      }
    }
    if (pairedBuf == UINT_MAX) {
      // Register-typed cross-WG flow (e.g. FA's alpha 256-vector or
      // softmax→TC P-tile bridge). The producer op holds the value in
      // registers; to ferry it to a different warp group we must stage it
      // through SMEM/TMEM with this barrier guarding the hand-off.
      // Allocate the buffer here so it's part of loop.buffers — Step 4
      // (budget) and Step 4.5 (lifetime merging) see it like any other.
      ttg::ScheduleBuffer chan;
      chan.id = loop.buffers.size();
      chan.kind = ttg::MemoryKind::SMEM;
      chan.count = 1; // single-buffered hand-off (depth>1 needs ring logic)
      chan.liveStart = src.cycle;
      chan.liveEnd = dst.cycle + std::max(dst.latency, 0);
      // Derive shape + element width from the producer's result type.
      Operation *prodOp = src.op;
      if (prodOp && prodOp->getNumResults() > 0) {
        Type resTy = prodOp->getResult(0).getType();
        auto setFromShaped = [&](llvm::ArrayRef<int64_t> shape, Type elemTy) {
          if (!elemTy.isIntOrFloat()) return;
          for (auto d : shape) {
            if (d <= 0 || ShapedType::isDynamic(d)) return;
          }
          for (auto d : shape) chan.shape.push_back(d);
          chan.elementBitWidth = elemTy.getIntOrFloatBitWidth();
        };
        if (auto memDesc = dyn_cast_or_null<ttg::MemDescType>(resTy))
          setFromShaped(memDesc.getShape(), memDesc.getElementType());
        else if (auto tt = dyn_cast_or_null<RankedTensorType>(resTy))
          setFromShaped(tt.getShape(), tt.getElementType());
      }
      // Skip if we couldn't determine a usable shape (scalar or unknown).
      if (!chan.shape.empty()) {
        loop.buffers.push_back(chan);
        pairedBuf = chan.id;
      }
    }

    // Avoid duplicate barriers for the same (producer, consumer) pair.
    if (!seenBarrierPairs.insert({edge.srcId, edge.dstId}).second)
      continue;

    ttg::ScheduleLoop::CrossGroupBarrier bar;
    bar.producerNodeId = edge.srcId;
    bar.consumerNodeId = edge.dstId;
    bar.producerWarpGroup = src.warpGroup;
    bar.consumerWarpGroup = dst.warpGroup;
    bar.kind = kind;
    bar.depth = depth;
    bar.pairedBufferId = pairedBuf;

    // Arrive/wait placement:
    // arrive AFTER the producer finishes writing (= producer node)
    // wait BEFORE the consumer starts reading (= consumer node)
    bar.arriveAfterNodeId = edge.srcId;
    bar.waitBeforeNodeId = edge.dstId;

    // For mbarrier: expected bytes from the paired buffer.
    if (kind == ttg::ScheduleLoop::BarrierKind::MBARRIER &&
        pairedBuf != UINT_MAX) {
      bar.expectBytes = loop.buffers[pairedBuf].sizeBytes();
    }

    loop.crossGroupBarriers.push_back(bar);

    LLVM_DEBUG(llvm::dbgs()
               << "[PassB.2] Barrier: N" << edge.srcId << "(wg" << src.warpGroup
               << ") → N" << edge.dstId << "(wg" << dst.warpGroup << ") "
               << (kind == ttg::ScheduleLoop::BarrierKind::MBARRIER ? "mbarrier"
                                                                    : "named")
               << " depth=" << depth << " arrive_after=N"
               << bar.arriveAfterNodeId << " wait_before=N"
               << bar.waitBeforeNodeId << " expect=" << bar.expectBytes
               << "B\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "[PassB.2] Total cross-group barriers: "
                          << loop.crossGroupBarriers.size() << "\n");
}

/// Top-level: build a ScheduleGraph from DDG + schedule result.
/// Includes Phase 0 (DDG→nodes/edges), Step 2.5 (clusters),
/// Step 3 (buffer allocation), Step 4.5 (merging), Step 4.6 (budget),
/// Pass B Steps 1-2 (warp groups + synchronization).
///
/// Cross-level SMEM propagation: parent loop SMEM is automatically
/// reserved when checking child loop budgets, so nested loops share
/// the global SMEM budget correctly at any nesting depth.
static ttg::ScheduleGraph
buildScheduleGraph(scf::ForOp loop, const ttg::DataDependenceGraph &ddg,
                   const ttg::ModuloScheduleResult &sched,
                   const ttg::LatencyModel &model) {
  ttg::ScheduleGraph graph;
  buildScheduleLoop(loop, ddg, sched, graph, model);

  for (auto &schedLoop : graph.loops) {
    allocateBuffersForLoop(schedLoop);
    mergeNonOverlappingBuffers(schedLoop);
  }

  llvm::DenseMap<unsigned, unsigned> parentMap;
  for (auto &schedLoop : graph.loops)
    for (auto &node : schedLoop.nodes)
      if (node.childPipelineId != UINT_MAX)
        parentMap[node.childPipelineId] = schedLoop.id;

  llvm::DenseMap<unsigned, int64_t> loopSmem;
  for (auto &schedLoop : graph.loops) {
    int64_t ancestorSmem = 0;
    for (unsigned id = schedLoop.id; parentMap.count(id);) {
      id = parentMap[id];
      auto it = loopSmem.find(id);
      if (it != loopSmem.end())
        ancestorSmem += it->second;
    }
    reduceBuffersForBudget(schedLoop, ancestorSmem);
    loopSmem[schedLoop.id] = computeTotalSmem(schedLoop);

    // Update maxStage to match buffer depth for prologue generation.
    int maxBufCount = 1;
    for (const auto &buf : schedLoop.buffers)
      if (buf.kind == ttg::MemoryKind::SMEM)
        maxBufCount = std::max(maxBufCount, static_cast<int>(buf.count));
    schedLoop.maxStage = std::max(schedLoop.maxStage, maxBufCount - 1);
  }

  // Per-loop reduction may pass each loop's own check yet exceed the global
  // SMEM budget when sibling/cousin loops share the same SMEM pool.
  // Run a global reduction across all loops jointly (see issue
  // 001_annotation_smem_overflow).
  reduceBuffersForGlobalBudget(graph);
  // Refresh maxStage after global reduction may have changed buffer depths.
  for (auto &schedLoop : graph.loops) {
    int maxBufCount = 1;
    for (const auto &buf : schedLoop.buffers)
      if (buf.kind == ttg::MemoryKind::SMEM)
        maxBufCount = std::max(maxBufCount, static_cast<int>(buf.count));
    schedLoop.maxStage = std::max(schedLoop.maxStage, maxBufCount - 1);
  }

  // Warp-group partition + cross-group barriers are NOT done here. Pass A
  // runs them as a single global pass over all scheduled loops
  // (`applyGlobalWarpPartition`) so cross-loop coordination — e.g., an
  // outer-loop super-node consistent with the inner loop's MMA group — is
  // possible. See `runOnOperation`.
  return graph;
}

// ============================================================================
// ============================================================================
// Schedule a single loop
// ============================================================================

struct ScheduleResult {
  ttg::ScheduleGraph graph;
  ttg::DataDependenceGraph ddg;
  ttg::ModuloScheduleResult sched; // Raw schedule (II + nodeToCycle).
                                   // Outer-loop lowering reads this.
};

/// Build the DDG, run modulo scheduling, and build the ScheduleGraph for
/// `loop`. Used for both inner and outer loops — the outer loop's super-node
/// info (printed only for `node.isSuperNode`) is naturally a no-op for
/// inner loops, so no branching is needed.
///
/// `label` is used in diagnostics ("Loop" or "Outer"). `printScheduleGraph`
/// is the test-only knob from the parent pass (dumps the graph to errs
/// unconditionally for lit tests in opt builds).
///
/// Lowering is intentionally NOT done here — `runOnOperation` runs a single
/// lower-or-emit phase after the iteration loop converges, so the schedule
/// build stays cheap and idempotent inside the iteration.
static std::optional<ScheduleResult>
scheduleOneLoop(scf::ForOp loop, const ttg::LatencyModel &model,
                triton::ModuleAxisInfoAnalysis &axisInfo, StringRef label,
                bool printScheduleGraph = false) {
  auto ddg = ttg::DataDependenceGraph::build(loop, model);
  if (ddg.getNumNodes() == 0)
    return std::nullopt;

  LDBG(label << " DDG: " << ddg.getNumNodes() << " nodes, "
             << ddg.getEdges().size() << " edges");
  // Per-pipeline node-count diagnostic (helps spot under/over-utilized
  // hardware pipelines at a glance).
  LLVM_DEBUG({
    int nMEM = 0, nTC = 0, nCUDA = 0, nSFU = 0, nNONE = 0;
    for (const auto &node : ddg.getNodes()) {
      switch (node.pipeline) {
      case ttg::HWPipeline::MEM:  ++nMEM;  break;
      case ttg::HWPipeline::TC:   ++nTC;   break;
      case ttg::HWPipeline::CUDA: ++nCUDA; break;
      case ttg::HWPipeline::SFU:  ++nSFU;  break;
      case ttg::HWPipeline::NONE: ++nNONE; break;
      }
    }
    llvm::dbgs() << "[" << label << "] Pipeline counts: MEM=" << nMEM
                 << " TC=" << nTC << " CUDA=" << nCUDA << " SFU=" << nSFU
                 << " NONE=" << nNONE << " (total=" << ddg.getNumNodes()
                 << ")\n";
  });

  // Print super-node summaries (no-op when the DDG has none, so this is
  // safe for inner loops too).
  LLVM_DEBUG({
    for (const auto &node : ddg.getNodes()) {
      if (!node.isSuperNode)
        continue;
      llvm::dbgs() << "[" << label << " DDG] Super-node N" << node.idx << " ("
                   << node.op->getName().getStringRef() << ")"
                   << " pipe=" << ttg::getPipelineName(node.pipeline)
                   << " lat=" << node.latency
                   << " innerII=" << node.innerII << "\n";
    }
  });

  auto schedResult = ttg::runModuloScheduling(ddg);
  if (failed(schedResult)) {
    LDBG(label << " scheduling FAILED");
    return std::nullopt;
  }
  LDBG(label << " scheduling SUCCESS: II=" << schedResult->II);

  LLVM_DEBUG({
    llvm::dbgs() << "[PASS-A] " << label
                 << " Schedule: II=" << schedResult->II
                 << " ResMII=" << ddg.computeResMII()
                 << " RecMII=" << ddg.computeRecMII() << " maxStage="
                 << schedResult->getMaxStage() << "\n";

    for (const auto &node : ddg.getNodes()) {
      auto it = schedResult->nodeToCycle.find(node.idx);
      if (it == schedResult->nodeToCycle.end())
        continue;
      int cycle = it->second;
      int stage = cycle / schedResult->II;
      llvm::dbgs() << "[PASS-A]   N" << node.idx << "  cycle=" << cycle
                   << "  stage=" << stage << "  "
                   << ttg::getPipelineName(node.pipeline)
                   << "  mw=" << node.minWarps
                   << "  selfLat=" << node.selfLatency
                   << "  lat=" << node.latency << "  ";
      node.op->print(llvm::dbgs(),
                     OpPrintingFlags().skipRegions().elideLargeElementsAttrs());
      llvm::dbgs() << "\n";
    }
  });

  auto graph = buildScheduleGraph(loop, ddg, *schedResult, model);

  LLVM_DEBUG({
    llvm::dbgs() << "[PASS-A] === " << label << " ScheduleGraph ===\n";
    graph.dump();
  });
  if (printScheduleGraph) {
    llvm::errs() << "[PASS-A] === " << label << " ScheduleGraph ===\n";
    graph.dump(llvm::errs());
  }

  return ScheduleResult{std::move(graph), std::move(ddg),
                        std::move(*schedResult)};
}

// Keeps graph + DDG + loop together for deferred lower-or-emit. Outer loops
// also need the raw ModuloScheduleResult because lowerOuterLoopPipeline
// reads `sched.getMaxStage()` / `sched.nodeToCycle`, which the ScheduleGraph
// alone doesn't expose.
struct ScheduledLoop {
  scf::ForOp loop;
  ttg::ScheduleGraph graph;
  ttg::DataDependenceGraph ddg;
  bool isOuter{false};
  std::optional<ttg::ModuloScheduleResult> outerSched;
};

/// Schedule `loop` and append the result to `out`. Wraps `scheduleOneLoop`
/// + the inner-vs-outer accounting needed to populate `ScheduledLoop`. A
/// failed schedule (helper returns nullopt) is silently skipped — it's
/// already been logged by the helper.
static void scheduleAndRecord(scf::ForOp loop, StringRef label, bool isOuter,
                              const ttg::LatencyModel &model,
                              triton::ModuleAxisInfoAnalysis &axisInfo,
                              bool printScheduleGraph,
                              SmallVectorImpl<ScheduledLoop> &out) {
  auto result =
      scheduleOneLoop(loop, model, axisInfo, label, printScheduleGraph);
  if (!result)
    return;
  std::optional<ttg::ModuloScheduleResult> outerSched;
  if (isOuter)
    outerSched = result->sched;
  out.push_back({loop, std::move(result->graph), std::move(result->ddg),
                 isOuter, std::move(outerSched)});
}

/// Outer loops carry per-op `loop.stage` / `loop.cluster` attrs from the
/// modulo schedule. We always clamp the stage range to [0, 1] (the outer
/// pipeliner only ever wants 2 stages) and renumber clusters in IR order
/// within each stage. Idempotent — safe to call once per scheduled outer
/// loop in the lower-or-emit phase.
static void clampOuterStagesAndClusters(scf::ForOp outerLoop) {
  auto ctx = outerLoop.getContext();
  for (auto &op : outerLoop.getBody()->without_terminator()) {
    auto stageAttr = op.getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
    if (!stageAttr)
      continue;
    if (stageAttr.getInt() > 1)
      op.setAttr(tt::kLoopStageAttrName,
                 IntegerAttr::get(IntegerType::get(ctx, 32), 1));
  }
  DenseMap<int, int> nextClusterPerStage;
  for (auto &op : outerLoop.getBody()->without_terminator()) {
    auto stageAttr = op.getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
    if (!stageAttr)
      continue;
    int stage = stageAttr.getInt();
    int cluster = nextClusterPerStage[stage]++;
    op.setAttr(tt::kLoopClusterAttrName,
               IntegerAttr::get(IntegerType::get(ctx, 32), cluster));
  }
  if (auto maxStageAttr = outerLoop->getAttrOfType<IntegerAttr>(
          tt::kScheduledMaxStageAttrName)) {
    if (maxStageAttr.getInt() > 1)
      outerLoop->setAttr(tt::kScheduledMaxStageAttrName,
                         IntegerAttr::get(IntegerType::get(ctx, 32), 1));
  }
}

/// Pass B: Per-loop warp-group partition + cross-group barriers. Each
/// ScheduleLoop gets its own Phase 4 partition run with its own II.
/// Nested kernels (case2/case5) get inner-partition (e.g., 3-WG GEMM
/// split) decided at inner II without being drowned by the outer II.
/// Single-loop kernels (case1/case3) reduce to a single partition run
/// over that one loop. The acc_tmem TC↔default hand-off is the emitter's
/// legacy carve-out — no cross-loop barrier rebuild needed.
///
/// `TRITON_MODULO_EXHAUSTIVE_PARTITION=0|off|false` opts into the greedy
/// fallback partitioner. Default is the exhaustive Phase 4 search.
static void
applyGlobalWarpPartition(MutableArrayRef<ScheduledLoop> scheduledLoops) {
  auto exhaustiveEnv =
      triton::tools::getStrEnv("TRITON_MODULO_EXHAUSTIVE_PARTITION");
  bool useGreedy = (exhaustiveEnv == "0" || exhaustiveEnv == "false" ||
                    exhaustiveEnv == "off");
  for (auto &sl : scheduledLoops) {
    for (auto &schedLoop : sl.graph.loops) {
      if (schedLoop.II <= 0)
        continue;
      if (useGreedy)
        partitionIntoWarpGroups(schedLoop);
      else
        partitionExhaustive(schedLoop);
      demoteScalarArithToInfra(schedLoop);
      propagateWarpGroupToInfraOps(schedLoop);
    }
  }

  // Run barrier insertion per-loop using the now-globally-consistent
  // warp-group IDs. Cross-loop barriers (from Phase 2 edges) are still
  // generated per-loop because the original outer loop's edges naturally
  // include the super-node edges; barrier records get attached to the
  // OUTER loop's `crossGroupBarriers`. Pass D's lowering reads them from
  // there.
  for (auto &sl : scheduledLoops)
    for (auto &schedLoop : sl.graph.loops)
      insertCrossGroupBarriers(schedLoop);
  // Step 4.5 was already run during initial buildScheduleLoop (before WG
  // partitioning), but `insertCrossGroupBarriers` may have synthesized new
  // SMEM channel buffers for register-typed cross-WG flows. Re-run the
  // lifetime-aware merger so the new buffers participate in interval-graph
  // coloring with the originals — disjoint-lifetime channels can share
  // physical storage instead of each occupying its full size.
  for (auto &sl : scheduledLoops)
    for (auto &schedLoop : sl.graph.loops) {
      // Reset mergeGroupId on every buffer so re-coloring starts fresh.
      for (auto &buf : schedLoop.buffers)
        buf.mergeGroupId = UINT_MAX;
      schedLoop.physicalBuffers.clear();
      mergeNonOverlappingBuffers(schedLoop);
    }
}

// ============================================================================
// Loop discovery
// ============================================================================

/// A scf::ForOp candidate for modulo scheduling, with its nesting context
/// and raw direct-body flags. Built once via `collectCandidates` so the rest
/// of Pass A can iterate without re-walking the module.
///
/// This is intentionally a flag bag rather than a tagged enum: a loop may
/// simultaneously carry direct compute AND wrap inner loops (e.g. an outer
/// loop that does a few non-tiling ops then enters a K-loop), and the
/// scheduling decision belongs to the caller, not to this struct. Each call
/// site filters on whichever combination of flags it cares about.
struct CandidateLoop {
  scf::ForOp op;
  scf::ForOp parent;          // null op if outermost
  unsigned depth{0};          // 0 = outermost
  bool hasMMA{false};         // direct body has tcgen5_mma{,Scaled}
  bool hasTMA{false};         // direct body has descriptor_load / async TMA copy
  bool hasInnerLoop{false};   // direct body has a nested scf::ForOp
  bool hasExistingAnnotation{false}; // tt.autows on an MMA — user-tuned, skip
};

/// Walk `moduleOp` once and collect every `scf::ForOp` with its nesting
/// context and direct-body flags.
///
/// CONTRACT: the result is sorted by `depth` non-increasing (deepest first).
/// This is a load-bearing guarantee — Pass A iterates the result in that
/// order and relies on every nested loop being scheduled before its
/// parent, because the parent's DDG promotes the child to a super-node
/// whose `innerII` field is the child's just-computed II. Use
/// `stable_sort` so loops at the same depth keep their source-order
/// relative position (deterministic across builds).
static SmallVector<CandidateLoop> collectCandidates(ModuleOp moduleOp) {
  SmallVector<CandidateLoop> result;
  moduleOp.walk([&](scf::ForOp loop) {
    CandidateLoop c;
    c.op = loop;
    c.parent = loop->getParentOfType<scf::ForOp>();
    for (auto p = c.parent; p; p = p->getParentOfType<scf::ForOp>())
      ++c.depth;
    for (auto &op : loop.getBody()->without_terminator()) {
      if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp,
              ttng::AsyncTMACopyGlobalToLocalOp>(&op))
        c.hasTMA = true;
      if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(&op)) {
        c.hasMMA = true;
        if (op.hasAttr("tt.autows"))
          c.hasExistingAnnotation = true;
      }
      if (isa<scf::ForOp>(&op))
        c.hasInnerLoop = true;
    }
    result.push_back(c);
  });
  llvm::stable_sort(result, [](const auto &a, const auto &b) {
    return a.depth > b.depth;
  });
  return result;
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

  /// DDG transformation hooks for iterative refinement.
  /// Return true if any DDG was modified (triggers re-scheduling).

  /// Pass A.5: Data partitioning — split MMA + companion loads into N
  /// parallel sub-chains so the MMA queue can issue concurrent partials
  /// (NUM_MMA_GROUPS-style on Blackwell). M1: detect candidates only.
  bool applyDataPartitioning(ModuleOp moduleOp,
                             const ttg::LatencyModel &model,
                             MutableArrayRef<ScheduledLoop> scheduledLoops) {
    // A.5 (TRITON_DATA_PARTITION_N) deferred to follow-up diff.
    return false;
  }

  // A.5 partition helpers (annotatePartition, partitionDecisions_) deferred
  // along with applyDataPartitioning above — they reference ScheduleNode /
  // ScheduleBuffer partition fields that are part of the A.5 follow-up.

  /// Pass A.7: Epilogue subtiling — split monolithic TMA stores into
  /// independent sub-chains for better pipeline interleaving.
  ///
  /// M1: detect chain. M2: pick S. M3: annotate ScheduleGraph + shrink store
  /// buffer.
  ///
  /// SINGLE-ITERATION MODE (see plan §"KNOWN LIMITATION"): this function
  /// always returns false. The Pass A iterative loop rebuilds DDG +
  /// ScheduleGraph from source TTGIR each iteration, and the global SMEM
  /// reducer runs BEFORE A.7's mutation — so re-running the loop with a
  /// memoized decision doesn't recover K-loop depth. The buffer-recovery
  /// feedback path is deferred to M4.
  bool applyEpilogueSubtiling(ModuleOp moduleOp,
                              const ttg::LatencyModel &model,
                              MutableArrayRef<ScheduledLoop> scheduledLoops) {
    // A.7 (TRITON_MODULO_EPILOGUE_SUBTILE) deferred to follow-up diff.
    return false;
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    ttg::LatencyModel model;
    triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // ================================================================
    // Iterative scheduling loop (design doc Pass A orchestrator)
    //
    // Each iteration: schedule → derive depths → check budget →
    // apply DDG transformations → re-run if any DDG changed.
    // Converges in 1-2 iterations.
    // ================================================================
    // Collect scheduling results across iterations. Only the LAST
    // iteration's results are emitted — earlier iterations are discarded
    // when DDG transformations trigger re-scheduling.
    SmallVector<ScheduledLoop, 2> scheduledLoops;

    constexpr int kMaxIterations = 3;
    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
      LDBG("=== Iterative scheduling: iteration " << iteration << " ===");
      scheduledLoops.clear();

      // Single walk: collect every scf::ForOp with its nesting context and
      // direct-body flags. Sorted deepest-first so we schedule inner loops
      // before their outer wrappers — the outer schedule consumes the inner
      // II as a super-node latency.
      auto candidates = collectCandidates(moduleOp);

      // We currently support at most a 2-level loop nest (an inner compute
      // loop optionally wrapped by an outer tile/persistent loop). Refuse
      // anything deeper rather than silently mis-scheduling — the prologue
      // expansion, super-node DDG, and outer-loop pipelining all assume
      // depth <= 1 today.
      for (const auto &c : candidates) {
        if (c.depth >= 2) {
          c.op->emitError("modulo scheduling: loop nesting depth ")
              << c.depth << " not supported (max 2 levels)";
          return signalPassFailure();
        }
      }

      // Single bottom-up pass over candidates. `collectCandidates`'s
      // contract is that the result is sorted by depth non-increasing so an
      // inner K-loop (depth=1) is visited before its outer tile loop
      // (depth=0). The outer DDG promotes the inner loop to a super-node
      // whose `innerII` is the inner schedule's II, so this order is
      // required for correctness — not just a stylistic preference. The
      // assert below makes that invariant verifiable at runtime in case
      // the sort in `collectCandidates` ever drifts.
      //
      // hasInnerLoop is the inner-vs-outer signal:
      //   * true  → wraps a nested scf.for → outer
      //   * false → leaf → inner (only worth scheduling if it has compute)
      // Inner-vs-outer differences (super-node print, retaining the raw
      // ModuloScheduleResult for lowerOuterLoopPipeline) live inside
      // scheduleAndRecord / ScheduledLoop — not the call site.
      unsigned numInner = 0, numOuter = 0;
      [[maybe_unused]] unsigned prevDepth =
          std::numeric_limits<unsigned>::max();
      for (const auto &c : candidates) {
        assert(c.depth <= prevDepth &&
               "candidates must be sorted deepest-first");
        prevDepth = c.depth;
        if (c.hasExistingAnnotation) {
          LDBG("Skipping loop with existing tt.autows annotations");
          continue;
        }
        if (c.hasInnerLoop) {
          scheduleAndRecord(c.op, "Outer", /*isOuter=*/true, model,
                            axisInfoAnalysis, printScheduleGraph,
                            scheduledLoops);
          ++numOuter;
        } else if (c.hasMMA || c.hasTMA) {
          scheduleAndRecord(c.op, "Inner", /*isOuter=*/false, model,
                            axisInfoAnalysis, printScheduleGraph,
                            scheduledLoops);
          ++numInner;
        }
      }
      LDBG("Scheduled " << numInner << " inner loop(s), " << numOuter
                        << " outer loop(s)");

      // ================================================================
      // Pass B: Global warp-group partition + cross-group barriers across
      // all scheduled loops. Replaces the per-loop call that used to live
      // inside `buildScheduleGraph` — moving it out of scheduling makes
      // cross-loop coordination possible (e.g., outer-loop super-node
      // matched to inner-loop MMA's warp group).
      // ================================================================
      applyGlobalWarpPartition(scheduledLoops);

      // ================================================================
      // Iterative refinement: apply DDG transformations and check if
      // we need to re-schedule.
      // ================================================================
      bool ddgChanged = false;
      ddgChanged |= applyDataPartitioning(moduleOp, model, scheduledLoops);
      ddgChanged |= applyEpilogueSubtiling(moduleOp, model, scheduledLoops);

      if (!ddgChanged) {
        LDBG("Converged after " << iteration + 1 << " iteration(s)");
        break;
      }

      if (iteration + 1 >= kMaxIterations) {
        LDBG("Hit iteration limit (" << kMaxIterations
                                     << ") — keeping last valid schedule");
        break;
      }

      LDBG("DDG changed by transformation — re-scheduling");
    } // end iterative loop

    // ================================================================
    // Lower-or-emit phase. Runs ONCE after convergence so the iteration
    // loop above stays a pure schedule-refinement loop (no IR rewrites
    // beyond attribute clamping). For each scheduled loop, either:
    //   * `useScheduleGraphLowering` (TRITON_MODULO_LOWER_SCHEDULE_GRAPH=1)
    //     → directly lower to multi-buffered allocs / async TMA / barriers
    //     / WS regions. compiler.py skips downstream WS+pipeliner.
    //   * otherwise → emit `loop.stage`/`loop.cluster` annotations and let
    //     the downstream WS+pipeliner consume them.
    // For outer loops, lowering is additionally gated by
    // TRITON_MODULO_OUTER_LOWERING and requires getMaxStage() >= 1.
    // ================================================================
    for (auto &sl : scheduledLoops) {
      if (sl.isOuter) {
        // Stage clamping + cluster renumbering applies to BOTH paths
        // (annotation and lowered) — it sits between the schedule and the
        // attrs that downstream consumers read, so do it here once.
        clampOuterStagesAndClusters(sl.loop);
      }
      emitScheduleFromGraph(sl.loop, sl.graph, sl.ddg);
      if (sl.isOuter) {
        // tt.modulo_cycle is a scratch attr the outer DDG builder leaves
        // behind; clean it up regardless of the lowering path.
        for (auto &op : sl.loop.getBody()->without_terminator())
          op.removeAttr("tt.modulo_cycle");
      }
    }

    // Inner scf.for ops that are super-nodes in an outer schedule
    // carry loop.stage/loop.cluster from the OUTER schedule — keep
    // them so the outer pipeliner knows where the K-loop sits.

  }
};

// ============================================================================
// Pass A.6: List scheduling for non-loop regions
// ============================================================================
//
// Degenerate Rau's algorithm — no modulo wrap, no loop-carried edges. All
// ops get stage 0; goal is minimum makespan instead of minimum II. Lives
// here (not its own file) so the ScheduleGraph is constructed in one place
// alongside the modulo case. DEBUG_TYPE is redefined for this section so
// debug output is gated by `-debug-only=nvgpu-list-schedule` per reviewer
// feedback (was previously leaking under `-debug-only=modulo-scheduling-rau`).

#undef DEBUG_TYPE
#undef DBGS
#undef LDBG
#define DEBUG_TYPE "nvgpu-list-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

/// Per-pipeline occupancy tracker without modulo wrap. Each pipeline has
/// a "next free" cycle — no fixed II, no wrap-around. Mirrors the modulo
/// reservation table for the linear (no-wrap) case.
struct PipelineTracker {
  llvm::DenseMap<ttg::HWPipeline, int> nextFree;

  /// Earliest cycle the pipeline is available. The `duration` parameter
  /// is the prospective op's hold time and is unused here (the tracker
  /// only records when the previously placed op's hold ends); kept for
  /// API symmetry with the modulo case.
  int findFreeSlot(int earliest, ttg::HWPipeline pipeline,
                   int /*duration*/) const {
    if (pipeline == ttg::HWPipeline::NONE)
      return earliest;
    auto it = nextFree.find(pipeline);
    int pipeReady = (it != nextFree.end()) ? it->second : 0;
    return std::max(earliest, pipeReady);
  }

  void reserve(int cycle, ttg::HWPipeline pipeline, int duration) {
    if (pipeline == ttg::HWPipeline::NONE)
      return;
    nextFree[pipeline] = std::max(nextFree.lookup(pipeline), cycle + duration);
  }
};

/// Earliest cycle a node may start, given predecessors already placed.
/// Predecessor result-ready time is `pred.cycle + edge.latency`; the DDG
/// builder records the producer's `latency` (result-ready) on outgoing
/// edges, so we don't add `pred.selfLatency` separately.
static int listEarliestStart(unsigned nodeIdx,
                             const ttg::DataDependenceGraph &ddg,
                             const llvm::DenseMap<unsigned, int> &scheduled) {
  int earliest = 0;
  for (const auto *edge : ddg.getInEdges(nodeIdx)) {
    auto it = scheduled.find(edge->srcIdx);
    if (it == scheduled.end())
      continue;
    earliest = std::max(earliest, it->second + edge->latency);
  }
  return earliest;
}

/// Priority-based list scheduling on the DDG. Minimises makespan rather
/// than II. Critical-path height is the priority (highest first).
static FailureOr<ttg::ListScheduleResult>
runListScheduling(const ttg::DataDependenceGraph &ddg) {
  if (ddg.getNumNodes() == 0)
    return failure();

  auto heights = ddg.computeCriticalPathHeights();

  llvm::SmallVector<unsigned> order;
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i)
    order.push_back(i);
  llvm::sort(order, [&](unsigned a, unsigned b) {
    if (heights[a] != heights[b])
      return heights[a] > heights[b];
    return a < b;
  });

  PipelineTracker tracker;
  llvm::DenseMap<unsigned, int> scheduled;

  for (unsigned nodeIdx : order) {
    const auto &node = ddg.getNode(nodeIdx);
    int duration = std::max(node.selfLatency, 1);
    if (node.pipeline == ttg::HWPipeline::NONE)
      duration = 1;

    int earliest = listEarliestStart(nodeIdx, ddg, scheduled);
    int slot = tracker.findFreeSlot(earliest, node.pipeline, duration);

    tracker.reserve(slot, node.pipeline, duration);
    scheduled[nodeIdx] = slot;

    LLVM_DEBUG(DBGS() << "  List placed N" << nodeIdx << " ("
                      << ttg::getPipelineName(node.pipeline)
                      << " dur=" << duration << ") at cycle=" << slot << "\n");
  }

  // makespan = max(start + occupancy) across all nodes.
  int makespan = 0;
  for (auto &[idx, cycle] : scheduled) {
    const auto &node = ddg.getNode(idx);
    makespan = std::max(makespan, cycle + std::max(node.selfLatency, 1));
  }

  LLVM_DEBUG(DBGS() << "List schedule: makespan=" << makespan
                    << " nodes=" << ddg.getNumNodes() << "\n");

  ttg::ListScheduleResult result;
  result.makespan = makespan;
  result.nodeToCycle = std::move(scheduled);
  return result;
}

/// Build a ScheduleGraph from a list-scheduled loop. All ops get stage 0,
/// cluster from cycle rank.
static ttg::ScheduleGraph
buildListScheduleGraph(scf::ForOp loop, const ttg::DataDependenceGraph &ddg,
                       const ttg::ListScheduleResult &result) {
  ttg::ScheduleGraph graph;
  unsigned loopId = graph.addLoop(loop);
  auto &schedLoop = graph.getLoop(loopId);
  schedLoop.II = result.makespan; // For non-loop regions, "II" = makespan
  schedLoop.maxStage = 0;

  for (const auto &ddgNode : ddg.getNodes()) {
    ttg::ScheduleNode sn;
    sn.id = schedLoop.nodes.size();
    sn.op = ddgNode.op;
    sn.pipeline = ddgNode.pipeline;
    sn.latency = ddgNode.latency;
    sn.selfLatency = ddgNode.selfLatency;
    sn.minWarps = ddgNode.minWarps;
    sn.stage = 0;

    auto cycleIt = result.nodeToCycle.find(ddgNode.idx);
    if (cycleIt != result.nodeToCycle.end())
      sn.cycle = cycleIt->second;

    schedLoop.nodes.push_back(sn);
    schedLoop.opToNodeId[ddgNode.op] = sn.id;
  }

  llvm::DenseMap<unsigned, unsigned> ddgToPipe;
  for (unsigned i = 0; i < ddg.getNodes().size(); ++i)
    ddgToPipe[ddg.getNodes()[i].idx] = i;

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

  // Cluster IDs (same logic as Step 2.5, all stage 0).
  SmallVector<int> cycles;
  for (const auto &node : schedLoop.nodes)
    cycles.push_back(node.cycle);
  llvm::sort(cycles);
  cycles.erase(llvm::unique(cycles), cycles.end());
  llvm::DenseMap<int, int> cycleToCluster;
  for (int i = 0, e = cycles.size(); i < e; ++i)
    cycleToCluster[cycles[i]] = i;
  for (auto &node : schedLoop.nodes)
    node.cluster = cycleToCluster[node.cycle];

  return graph;
}

struct ListSchedulePass
    : public PassWrapper<ListSchedulePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ListSchedulePass)

  StringRef getArgument() const override { return "nvgpu-list-schedule"; }

  StringRef getDescription() const override {
    return "List scheduling for non-loop regions (Pass A.6)";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    ttg::LatencyModel model;

    moduleOp.walk([&](scf::ForOp loop) {
      if (loop->hasAttr("tt.modulo_ii"))
        return;

      bool hasPipelineOps = false;
      loop.getBody()->walk([&](Operation *op) {
        if (isa<tt::DescriptorLoadOp, tt::DescriptorStoreOp,
                ttng::AsyncTMACopyGlobalToLocalOp, ttng::TCGen5MMAOp,
                ttng::TCGen5MMAScaledOp, ttng::TMEMLoadOp>(op))
          hasPipelineOps = true;
      });
      if (!hasPipelineOps)
        return;

      auto ddg = ttg::DataDependenceGraph::build(loop, model);
      if (ddg.getNumNodes() == 0)
        return;

      LDBG("List scheduling loop with " << ddg.getNumNodes() << " nodes");

      auto result = runListScheduling(ddg);
      if (failed(result)) {
        LDBG("List scheduling FAILED");
        return;
      }

      LDBG("List schedule: makespan=" << result->makespan);

      auto schedGraph = buildListScheduleGraph(loop, ddg, *result);

      LLVM_DEBUG({
        llvm::dbgs() << "[A.6] === List ScheduleGraph ===\n";
        schedGraph.dump();
      });

      auto ctx = loop.getContext();
      for (const auto &schedLoop : schedGraph.loops) {
        for (const auto &node : schedLoop.nodes) {
          if (!node.op)
            continue;
          node.op->setAttr(tt::kLoopStageAttrName,
                           IntegerAttr::get(IntegerType::get(ctx, 32), 0));
          node.op->setAttr(
              tt::kLoopClusterAttrName,
              IntegerAttr::get(IntegerType::get(ctx, 32), node.cluster));
        }
      }

      // Default unscheduled ops to stage 0, max cluster.
      int maxCluster = 0;
      for (const auto &schedLoop : schedGraph.loops)
        for (const auto &node : schedLoop.nodes)
          maxCluster = std::max(maxCluster, node.cluster);
      for (auto &op : loop.getBody()->without_terminator()) {
        if (!op.hasAttr(tt::kLoopStageAttrName))
          op.setAttr(tt::kLoopStageAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), 0));
        if (!op.hasAttr(tt::kLoopClusterAttrName))
          op.setAttr(tt::kLoopClusterAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), maxCluster));
      }

      // Mark the loop scheduled so downstream `processScheduledLoop`
      // (which gates on `tt.modulo_ii`) preserves the schedule attrs.
      // `tt.list_schedule_makespan` distinguishes list-scheduled loops
      // from true modulo-scheduled ones for any consumer that cares.
      loop->setAttr("tt.modulo_ii", IntegerAttr::get(IntegerType::get(ctx, 32),
                                                     result->makespan));
      loop->setAttr(
          "tt.list_schedule_makespan",
          IntegerAttr::get(IntegerType::get(ctx, 32), result->makespan));
    });
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloSchedule() {
  return std::make_unique<ModuloSchedulePass>();
}

void registerNVGPUModuloSchedule() { PassRegistration<ModuloSchedulePass>(); }

std::unique_ptr<Pass> createNVGPUListSchedule() {
  return std::make_unique<ListSchedulePass>();
}

void registerNVGPUListSchedule() { PassRegistration<ListSchedulePass>(); }
} // namespace mlir
