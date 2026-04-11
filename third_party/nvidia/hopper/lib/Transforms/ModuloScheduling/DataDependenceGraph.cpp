// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "DataDependenceGraph.h"
#include "ModuloReservationTable.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <queue>

#define DEBUG_TYPE "modulo-scheduling-ddg"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::triton::gpu {

namespace ttng = mlir::triton::nvidia_gpu;

// Default estimated trip count for inner loops with dynamic bounds.
constexpr int kDefaultInnerTripCount = 4;

// Fallback latency constants from Blackwell SM100 microbenchmarks.
// Used when inner loop scheduling fails or stageLoads are empty.
constexpr int kFallbackTMASelfLatency = 518;   // TMA load pipeline occupancy
constexpr int kFallbackTMATotalLatency = 1218; // TMA self + async overhead
constexpr int kFallbackMMALatency = 900;       // MMA 128x128x128 TC latency

/// Per-stage pipeline load summary from inner loop schedule.
struct StageLoad {
  int memSelfLatency{0};
  int tcSelfLatency{0};
  int cudaSelfLatency{0};
  int totalLatency{0};
};

/// Info extracted from inner loop modulo scheduling.
struct InnerLoopInfo {
  int II;
  int prologueLatency;
  int tripCount;
  bool tripCountIsEstimated{true};
  SmallVector<StageLoad, 4> stageLoads;
  int maxStage{0};
};

static InnerLoopInfo computeInnerLoopInfo(scf::ForOp innerLoop,
                                          const LatencyModel &model) {
  InnerLoopInfo info;
  info.tripCount = kDefaultInnerTripCount;
  info.tripCountIsEstimated = true;

  auto innerDDG = DataDependenceGraph::build(innerLoop, model);
  if (innerDDG.getNumNodes() == 0) {
    info.II = kFallbackMMALatency;
    info.prologueLatency = 0;
    return info;
  }
  auto result = runModuloScheduling(innerDDG);
  if (failed(result)) {
    info.II = std::max(innerDDG.computeResMII(), kFallbackMMALatency);
    info.prologueLatency = 0;
    return info;
  }

  info.II = result->II;
  info.maxStage = result->getMaxStage();

  auto lb = innerLoop.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
  auto ub = innerLoop.getUpperBound().getDefiningOp<arith::ConstantIntOp>();
  auto step = innerLoop.getStep().getDefiningOp<arith::ConstantIntOp>();
  if (lb && ub && step && step.value() > 0) {
    int64_t tc = (ub.value() - lb.value() + step.value() - 1) / step.value();
    if (tc > 0) {
      info.tripCount = static_cast<int>(tc);
      info.tripCountIsEstimated = false;
    }
  }

  int tcStart = result->II;
  for (const auto &node : innerDDG.getNodes()) {
    if (node.pipeline == HWPipeline::TC) {
      auto it = result->nodeToCycle.find(node.idx);
      if (it != result->nodeToCycle.end())
        tcStart = std::min(tcStart, it->second);
    }
  }
  info.prologueLatency = tcStart;

  info.stageLoads.resize(info.maxStage + 1);
  for (const auto &node : innerDDG.getNodes()) {
    auto it = result->nodeToCycle.find(node.idx);
    if (it == result->nodeToCycle.end())
      continue;
    int stage = it->second / info.II;
    if (stage > info.maxStage)
      continue;
    auto &sl = info.stageLoads[stage];
    switch (node.pipeline) {
    case HWPipeline::MEM:
      sl.memSelfLatency += node.selfLatency;
      break;
    case HWPipeline::TC:
      sl.tcSelfLatency += node.selfLatency;
      break;
    case HWPipeline::CUDA:
    case HWPipeline::SFU:
      sl.cudaSelfLatency += node.selfLatency;
      break;
    default:
      break;
    }
    sl.totalLatency = std::max(sl.totalLatency, node.latency);
  }

  return info;
}

unsigned DataDependenceGraph::addNode(Operation *op,
                                      const LatencyModel &model) {
  auto info = model.getLatency(op);
  unsigned idx = nodes.size();
  DDGNode node;
  node.op = op;
  node.idx = idx;
  node.pipeline = info.pipeline;
  node.latency = info.latency;
  node.selfLatency = info.selfLatency;
  nodes.push_back(node);
  opToIdx[op] = idx;
  return idx;
}

void DataDependenceGraph::addEdge(unsigned src, unsigned dst, int latency,
                                  unsigned distance) {
  edges.push_back(DDGEdge{src, dst, latency, distance});
  nodes[src].succs.push_back(dst);
  nodes[dst].preds.push_back(src);
}

DataDependenceGraph DataDependenceGraph::build(scf::ForOp loop,
                                               const LatencyModel &model) {
  DataDependenceGraph ddg;

  // Phase 1: Create nodes for every op in the loop body (except terminator).
  auto &body = loop.getBody()->getOperations();
  for (auto &op : body) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    // Model inner scf.for as a super-node for outer loop scheduling.
    if (auto innerLoop = dyn_cast<scf::ForOp>(op)) {
      auto info = computeInnerLoopInfo(innerLoop, model);

      if (info.maxStage == 0) {
        unsigned idx = ddg.nodes.size();
        DDGNode node;
        node.op = &op;
        node.idx = idx;
        node.pipeline = HWPipeline::TC;
        int totalLatency = info.prologueLatency + info.tripCount * info.II;
        node.latency = std::max(totalLatency, 1);
        node.selfLatency = std::max(totalLatency, 1);
        node.isSuperNode = true;
        node.innerII = info.II;
        node.prologueLatency = info.prologueLatency;
        ddg.nodes.push_back(node);
        ddg.opToIdx[&op] = idx;
        continue;
      }

      // Multi-stage: split into prologue/kloop/epilogue.
      unsigned prologueIdx = ddg.nodes.size();
      {
        DDGNode node;
        node.op = &op;
        node.idx = prologueIdx;
        node.pipeline = HWPipeline::MEM;
        int memLat = info.stageLoads.empty() ? kFallbackTMATotalLatency
                                             : info.stageLoads[0].totalLatency;
        node.latency = std::max(memLat, 1);
        node.selfLatency = info.stageLoads.empty()
                               ? kFallbackTMASelfLatency
                               : info.stageLoads[0].memSelfLatency;
        node.selfLatency = std::max(node.selfLatency, 1);
        ddg.nodes.push_back(node);
      }

      unsigned kloopIdx = ddg.nodes.size();
      {
        DDGNode node;
        node.op = &op;
        node.idx = kloopIdx;
        node.pipeline = HWPipeline::TC;
        int steadyIters = std::max(info.tripCount - info.maxStage, 1);
        node.latency = steadyIters * info.II;
        int tcPerIter = info.stageLoads.size() > (unsigned)info.maxStage
                            ? info.stageLoads[info.maxStage].tcSelfLatency
                            : kFallbackMMALatency;
        node.selfLatency = std::max(steadyIters * tcPerIter, 1);
        node.isSuperNode = true;
        node.innerII = info.II;
        node.prologueLatency = info.prologueLatency;
        ddg.nodes.push_back(node);
      }

      unsigned epilogueIdx = ddg.nodes.size();
      {
        DDGNode node;
        node.op = &op;
        node.idx = epilogueIdx;
        node.pipeline = HWPipeline::TC;
        int tcLat = info.stageLoads.size() > (unsigned)info.maxStage
                        ? info.stageLoads[info.maxStage].tcSelfLatency
                        : kFallbackMMALatency;
        node.latency = std::max(tcLat, 1);
        node.selfLatency = std::max(tcLat, 1);
        ddg.nodes.push_back(node);
      }

      ddg.opToIdx[&op] = epilogueIdx; // producer: results come from epilogue
      ddg.consumerOpToIdx[&op] =
          prologueIdx; // consumer: data enters at prologue
      ddg.addEdge(prologueIdx, kloopIdx, ddg.nodes[prologueIdx].latency,
                  /*distance=*/0);
      ddg.addEdge(kloopIdx, epilogueIdx, ddg.nodes[kloopIdx].latency,
                  /*distance=*/0);
      continue;
    }
    // Handle scf.if: walk regions to find pipeline-relevant ops.
    // Persistent kernels put TMA loads inside conditional prefetch blocks
    // (scf.if i < num_iter). Without this, those ops are invisible to
    // the scheduler.
    if (isa<scf::IfOp>(op)) {
      HWPipeline bestPipeline = HWPipeline::NONE;
      int bestLatency = 0;
      int bestSelfLatency = 0;
      op.walk([&](Operation *nested) {
        if (nested == &op)
          return;
        auto info = model.getLatency(nested);
        if (info.pipeline != HWPipeline::NONE &&
            info.selfLatency > bestSelfLatency) {
          bestPipeline = info.pipeline;
          bestLatency = info.latency;
          bestSelfLatency = info.selfLatency;
        }
      });
      if (bestPipeline != HWPipeline::NONE) {
        unsigned idx = ddg.nodes.size();
        DDGNode node;
        node.op = &op;
        node.idx = idx;
        node.pipeline = bestPipeline;
        node.latency = bestLatency;
        node.selfLatency = bestSelfLatency;
        ddg.nodes.push_back(node);
        ddg.opToIdx[&op] = idx;
        continue;
      }
    }
    ddg.addNode(&op, model);
  }

  // Phase 2: Intra-iteration edges from SSA def-use chains.
  for (auto &node : ddg.nodes) {
    for (auto operand : node.op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getNumResults() == 0)
        continue;
      auto it = ddg.opToIdx.find(defOp);
      if (it == ddg.opToIdx.end())
        continue;
      unsigned srcIdx = it->second;
      // Edge latency = producer's latency (time until result available).
      // Exception: for MEM → local_alloc edges, use selfLatency instead of
      // the full async latency. local_alloc is a format conversion (registers
      // → SMEM) that must stay at the same pipeline stage as its load.
      // The async overhead only applies to the MMA consumer, not local_alloc.
      int edgeLatency = ddg.nodes[srcIdx].latency;
      if (ddg.nodes[srcIdx].pipeline == HWPipeline::MEM &&
          isa<triton::gpu::LocalAllocOp>(node.op)) {
        edgeLatency = ddg.nodes[srcIdx].selfLatency;
      }
      ddg.addEdge(srcIdx, node.idx, edgeLatency, /*distance=*/0);
    }
  }

  // Phase 2.5: Implicit MEM-load → TC/super-node edges.
  // In persistent kernels using lowered TMA (async_tma_copy), the MEM→TC
  // dependency goes through SMEM buffers and barriers, not SSA. Without
  // these edges the scheduler places MEM and TC at the same cycle.
  {
    SmallVector<unsigned> memLoadNodes, tcNodes;
    for (const auto &node : ddg.nodes) {
      if (node.pipeline == HWPipeline::MEM) {
        bool isStore = isa<triton::DescriptorStoreOp>(node.op) ||
                       isa<ttng::AsyncTMACopyLocalToGlobalOp>(node.op);
        if (!isStore && isa<scf::IfOp>(node.op)) {
          node.op->walk([&](Operation *nested) {
            if (isa<triton::DescriptorStoreOp>(nested) ||
                isa<ttng::AsyncTMACopyLocalToGlobalOp>(nested))
              isStore = true;
          });
        }
        if (!isStore)
          memLoadNodes.push_back(node.idx);
      }
      if (node.pipeline == HWPipeline::TC || node.isSuperNode)
        tcNodes.push_back(node.idx);
    }
    for (unsigned memIdx : memLoadNodes) {
      for (unsigned tcIdx : tcNodes) {
        bool hasEdge = false;
        for (const auto &e : ddg.edges) {
          if (e.srcIdx == memIdx && e.dstIdx == tcIdx) {
            hasEdge = true;
            break;
          }
        }
        if (!hasEdge) {
          ddg.addEdge(memIdx, tcIdx, ddg.nodes[memIdx].latency,
                      /*distance=*/0);
        }
      }
    }
  }

  // Phase 3: Loop-carried edges via scf.yield → iter_args.
  auto yieldOp = loop.getBody()->getTerminator();
  auto iterArgs = loop.getRegionIterArgs();
  for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
    auto yieldVal = yieldOp->getOperand(i);
    auto *yieldDef = yieldVal.getDefiningOp();
    if (!yieldDef || yieldDef->getNumResults() == 0 ||
        ddg.opToIdx.count(yieldDef) == 0)
      continue;
    unsigned srcIdx = ddg.opToIdx[yieldDef];

    // The iter_arg at position i receives yieldVal in the next iteration.
    // Find all users of that iter_arg within the loop body.
    if (i >= iterArgs.size())
      continue;
    auto iterArg = iterArgs[i];
    for (auto *user : iterArg.getUsers()) {
      if (user->hasTrait<OpTrait::IsTerminator>())
        continue;
      // For multi-stage super-nodes, loop-carried edges should target
      // the prologue (data enters at the start), not the epilogue.
      auto consIt = ddg.consumerOpToIdx.find(user);
      unsigned dstIdx;
      if (consIt != ddg.consumerOpToIdx.end()) {
        dstIdx = consIt->second;
      } else {
        auto userIt = ddg.opToIdx.find(user);
        if (userIt == ddg.opToIdx.end())
          continue;
        dstIdx = userIt->second;
      }
      ddg.addEdge(srcIdx, dstIdx, ddg.nodes[srcIdx].latency,
                  /*distance=*/1);
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[DDG] Built DDG with " << ddg.nodes.size() << " nodes, "
                 << ddg.edges.size() << " edges\n";
    ddg.dump();
  });

  return ddg;
}

llvm::SmallVector<const DDGEdge *>
DataDependenceGraph::getInEdges(unsigned nodeIdx) const {
  llvm::SmallVector<const DDGEdge *> result;
  for (const auto &e : edges)
    if (e.dstIdx == nodeIdx)
      result.push_back(&e);
  return result;
}

llvm::SmallVector<const DDGEdge *>
DataDependenceGraph::getOutEdges(unsigned nodeIdx) const {
  llvm::SmallVector<const DDGEdge *> result;
  for (const auto &e : edges)
    if (e.srcIdx == nodeIdx)
      result.push_back(&e);
  return result;
}

llvm::DenseMap<unsigned, int>
DataDependenceGraph::computeCriticalPathHeights() const {
  llvm::DenseMap<unsigned, int> heights;
  llvm::DenseSet<unsigned> visiting; // cycle detection
  // Reverse topological order: process sinks first.
  // Use DFS-based approach since graph is small.
  std::function<int(unsigned)> computeHeight = [&](unsigned idx) -> int {
    auto it = heights.find(idx);
    if (it != heights.end())
      return it->second;
    // Guard against cycles in distance-0 edges. DDG construction guarantees
    // acyclicity, but this prevents infinite recursion if invariant is broken.
    if (!visiting.insert(idx).second)
      return 0;
    int maxSuccHeight = 0;
    for (const auto *edge : getOutEdges(idx)) {
      if (edge->distance > 0)
        continue; // skip loop-carried for critical path
      int succHeight = computeHeight(edge->dstIdx);
      maxSuccHeight = std::max(maxSuccHeight, edge->latency + succHeight);
    }
    visiting.erase(idx);
    heights[idx] = maxSuccHeight;
    return maxSuccHeight;
  };
  for (unsigned i = 0; i < nodes.size(); ++i)
    computeHeight(i);
  return heights;
}

int DataDependenceGraph::computeResMII() const {
  llvm::DenseMap<HWPipeline, int> pipeLoad;
  for (const auto &node : nodes) {
    if (node.pipeline == HWPipeline::NONE)
      continue;
    pipeLoad[node.pipeline] += node.selfLatency;
  }
  int maxLoad = 0;
  for (auto &[pipe, load] : pipeLoad) {
    LLVM_DEBUG(llvm::dbgs() << "[DDG] Pipeline " << getPipelineName(pipe)
                            << " load: " << load << " cycles\n");
    maxLoad = std::max(maxLoad, load);
  }
  return maxLoad;
}

int DataDependenceGraph::computeRecMII() const {
  // Compute RecMII = max over all recurrence circuits of ceil(sum_lat /
  // sum_dist).
  //
  // For each back-edge (distance > 0), find the longest forward path from
  // dst back to src. The recurrence latency = forward_path + back_edge_latency,
  // and distance = forward_distance + back_edge_distance. RecMII for that
  // circuit = ceil(total_lat / total_dist).
  //
  // We use Floyd-Warshall to compute longest forward paths (distance=0 edges
  // only), then combine with each back-edge.
  const unsigned N = nodes.size();
  if (N == 0)
    return 0;

  // Forward-path longest latencies (only distance=0 edges).
  constexpr int NEG_INF = -1;
  std::vector<std::vector<int>> fwdLat(N, std::vector<int>(N, NEG_INF));

  // Initialize with distance=0 edges only.
  for (const auto &e : edges) {
    if (e.distance == 0) {
      fwdLat[e.srcIdx][e.dstIdx] =
          std::max(fwdLat[e.srcIdx][e.dstIdx], e.latency);
    }
  }
  // Self-loops with distance 0.
  for (unsigned i = 0; i < N; ++i)
    fwdLat[i][i] = std::max(fwdLat[i][i], 0);

  // Floyd-Warshall on forward paths.
  for (unsigned k = 0; k < N; ++k) {
    for (unsigned i = 0; i < N; ++i) {
      for (unsigned j = 0; j < N; ++j) {
        if (fwdLat[i][k] == NEG_INF || fwdLat[k][j] == NEG_INF)
          continue;
        int newLat = fwdLat[i][k] + fwdLat[k][j];
        if (newLat > fwdLat[i][j])
          fwdLat[i][j] = newLat;
      }
    }
  }

  // For each back-edge, compute the recurrence ratio.
  int recMII = 0;
  for (const auto &e : edges) {
    if (e.distance == 0)
      continue;
    // Back-edge: src → dst with distance > 0.
    // Forward path: dst →...→ src (distance=0 edges).
    // Total recurrence: forward_lat + back_edge_lat, total_dist = e.distance.
    int forwardLat = fwdLat[e.dstIdx][e.srcIdx];
    if (forwardLat == NEG_INF)
      continue; // no forward path completes the circuit
    int totalLat = forwardLat + e.latency;
    int totalDist = e.distance;
    int rec = (totalLat + totalDist - 1) / totalDist; // ceil
    LLVM_DEBUG(llvm::dbgs() << "[DDG] Recurrence: back-edge " << e.srcIdx
                            << " -> " << e.dstIdx << " (dist=" << e.distance
                            << ") fwdLat=" << forwardLat << " totalLat="
                            << totalLat << " RecMII=" << rec << "\n");
    recMII = std::max(recMII, rec);
  }
  return recMII;
}

int DataDependenceGraph::computeMinII() const {
  int resMII = computeResMII();
  int recMII = computeRecMII();
  int minII = std::max(resMII, recMII);
  LLVM_DEBUG(llvm::dbgs() << "[DDG] ResMII=" << resMII << " RecMII=" << recMII
                          << " MinII=" << minII << "\n");
  return minII;
}

void DataDependenceGraph::dump() const {
  llvm::dbgs() << "=== DDG Dump ===\n";
  for (const auto &node : nodes) {
    llvm::dbgs() << "  Node " << node.idx
                 << ": pipeline=" << getPipelineName(node.pipeline)
                 << " latency=" << node.latency
                 << " selfLatency=" << node.selfLatency;
    if (node.isSuperNode)
      llvm::dbgs() << " [SUPER-NODE innerII=" << node.innerII << "]";
    llvm::dbgs() << " op=";
    node.op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
  }
  for (const auto &edge : edges) {
    llvm::dbgs() << "  Edge " << edge.srcIdx << " -> " << edge.dstIdx
                 << " latency=" << edge.latency << " distance=" << edge.distance
                 << "\n";
  }
  llvm::dbgs() << "=== End DDG ===\n";
}

} // namespace mlir::triton::gpu
