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
// Used to compute super-node latency = prologue + K_est * II.
constexpr int kDefaultInnerTripCount = 4;

// Fallback latency constants from Blackwell SM100 microbenchmarks.
// Used when inner loop scheduling fails or stageLoads are empty.
constexpr int kFallbackTMASelfLatency = 518;   // TMA load pipeline occupancy
constexpr int kFallbackTMATotalLatency = 1218; // TMA self + async overhead
constexpr int kFallbackMMALatency = 900;       // MMA 128x128x128 TC latency

/// Per-stage pipeline load summary from inner loop schedule.
struct StageLoad {
  int memSelfLatency{0};  // total MEM pipeline occupancy
  int tcSelfLatency{0};   // total TC pipeline occupancy
  int cudaSelfLatency{0}; // total CUDA pipeline occupancy
  int totalLatency{0};    // max end-to-end latency across all ops
};

/// Info extracted from inner loop modulo scheduling.
struct InnerLoopInfo {
  int II;
  int prologueLatency; // cycles before TC starts
  int tripCount;       // known or estimated trip count
  bool tripCountIsEstimated{true};
  // Per-stage load (stage 0 = prologue ops, stage maxStage = epilogue ops)
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

  // Try to extract constant trip count from scf.for bounds.
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

  // Find the earliest TC cycle — that's where MMA starts (= prologueLatency)
  int tcStart = result->II;
  for (const auto &node : innerDDG.getNodes()) {
    if (node.pipeline == HWPipeline::TC) {
      auto it = result->nodeToCycle.find(node.idx);
      if (it != result->nodeToCycle.end())
        tcStart = std::min(tcStart, it->second);
    }
  }
  info.prologueLatency = tcStart;

  // Collect per-stage pipeline loads from the inner schedule.
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

  LLVM_DEBUG(DBGS() << "Inner loop: II=" << info.II
                    << " maxStage=" << info.maxStage
                    << " prologueLat=" << info.prologueLatency
                    << " tripCount=" << info.tripCount
                    << (info.tripCountIsEstimated ? "(est)" : "(const)")
                    << " stages=" << info.stageLoads.size() << "\n");
  for (int s = 0; s <= info.maxStage; ++s) {
    LLVM_DEBUG(DBGS() << "  stage " << s
                      << ": MEM=" << info.stageLoads[s].memSelfLatency
                      << " TC=" << info.stageLoads[s].tcSelfLatency
                      << " CUDA=" << info.stageLoads[s].cudaSelfLatency
                      << "\n");
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
  // Inner scf.for loops become super-nodes with latency = inner loop's II.
  // scf.if ops are inspected: if they contain pipeline-relevant ops (TMA
  // loads/stores), the scf.if node inherits the dominant pipeline/latency
  // from its contents (e.g., conditional prefetch blocks).
  auto &body = loop.getBody()->getOperations();
  for (auto &op : body) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    // Handle inner scf.for as a super-node
    if (auto innerLoop = dyn_cast<scf::ForOp>(op)) {
      auto info = computeInnerLoopInfo(innerLoop, model);

      // If inner loop is single-stage (already expanded or trivial),
      // create a single super-node — no prologue/epilogue split needed.
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
        LLVM_DEBUG(DBGS() << "Single-stage inner loop: N" << idx
                         << " = super-node (TC, lat=" << node.latency
                         << " II=" << info.II << ")\n");
        continue;
      }

      // Multi-stage: split inner loop into 3 synthetic nodes in the outer DDG:
      //   prologue: MEM loads (stage 0 ops) — runs once before steady state
      //   kloop:    steady-state scf.for — TC+MEM interleaved, K iterations
      //   epilogue: last MMA (highest stage ops) — drains after loop exits
      //
      // Only kloop is a super-node (still an scf.for). Prologue/epilogue
      // are synthetic DDG nodes — they don't correspond to a real loop.

      LLVM_DEBUG({
        DBGS() << "Inner loop scheduled: II=" << info.II
               << " maxStage=" << info.maxStage
               << " tripCount=" << info.tripCount
               << (info.tripCountIsEstimated ? "(est)" : "(const)") << "\n";
        for (int s = 0; s <= info.maxStage; ++s) {
          DBGS() << "  stage " << s
                 << ": MEM=" << info.stageLoads[s].memSelfLatency
                 << " TC=" << info.stageLoads[s].tcSelfLatency
                 << " CUDA=" << info.stageLoads[s].cudaSelfLatency
                 << " totalLat=" << info.stageLoads[s].totalLatency << "\n";
        }
      });

      // --- Node 1: inner_prologue (MEM pipeline, synthetic) ---
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
        // NOT a super-node — synthetic prologue, no backing loop
        ddg.nodes.push_back(node);
        LLVM_DEBUG(DBGS() << "Split inner loop: N" << prologueIdx
                         << " = inner_prologue (MEM, lat=" << node.latency
                         << " selfLat=" << node.selfLatency << ")\n");
      }

      // --- Node 2: inner_kloop (TC pipeline, super-node) ---
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
        node.isSuperNode = true; // this IS the scf.for
        node.innerII = info.II;
        node.prologueLatency = info.prologueLatency;
        ddg.nodes.push_back(node);
        LLVM_DEBUG(DBGS() << "Split inner loop: N" << kloopIdx
                         << " = inner_kloop (TC, lat=" << node.latency
                         << " selfLat=" << node.selfLatency
                         << " steadyIters=" << steadyIters << ")\n");
      }

      // --- Node 3: inner_epilogue (TC pipeline, synthetic) ---
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
        // NOT a super-node — synthetic epilogue
        ddg.nodes.push_back(node);
        LLVM_DEBUG(DBGS() << "Split inner loop: N" << epilogueIdx
                         << " = inner_epilogue (TC, lat=" << node.latency
                         << " selfLat=" << node.selfLatency << ")\n");
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
    // TLX kernels put TMA loads inside conditional prefetch blocks
    // (scf.if i < num_iter). Without this, those ops are invisible
    // to the scheduler and the loop gets a trivial single-stage schedule.
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
        LLVM_DEBUG(DBGS() << "scf.if node " << idx
                          << ": pipeline=" << getPipelineName(bestPipeline)
                          << " latency=" << bestLatency << "\n");
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
  // TMA loads write to SMEM buffers consumed by MMA ops. This producer-consumer
  // relationship isn't visible in SSA (they communicate through barriers and
  // shared memory). Without these edges, the scheduler places MEM and TC at the
  // same cycle, missing the opportunity for multi-stage pipelining where MEM
  // ops run ahead to prefetch data for future iterations.
  // NOTE: Only MEM *load* ops get implicit edges to TC. MEM *store* ops
  // (descriptor_store) consume the K-loop result — adding store→TC edges
  // creates false cycles (store → super-node → ... → store).
  {
    SmallVector<unsigned> memLoadNodes, tcNodes;
    for (const auto &node : ddg.nodes) {
      if (node.pipeline == HWPipeline::MEM) {
        // Only loads, not stores. For scf.if nodes, check if the
        // contained op is a store.
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
        // Skip if an SSA edge already exists (avoid duplicates).
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
          LLVM_DEBUG(DBGS() << "Implicit edge: MEM node " << memIdx
                            << " -> TC node " << tcIdx << " (latency="
                            << ddg.nodes[memIdx].latency << ")\n");
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
