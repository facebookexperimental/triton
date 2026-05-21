// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "DataDependenceGraph.h"
#include "ModuloReservationTable.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  node.transferLatency = info.transferLatency;
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
    // Inner scf.for loops become super-nodes. The super-node's latency
    // is the inner loop's total execution time (II × trip_count), and
    // its pipeline is NONE (handles its own internal pipelining).
    if (auto innerLoop = dyn_cast<scf::ForOp>(op)) {
      auto innerDDG = DataDependenceGraph::build(innerLoop, model);
      auto innerSched = runModuloScheduling(innerDDG);

      DDGNode node;
      node.op = &op;
      node.idx = ddg.nodes.size();
      node.isSuperNode = true;

      if (succeeded(innerSched)) {
        node.innerII = innerSched->II;
        int tripCount = 32; // default estimate
        auto lb = innerLoop.getLowerBound()
                      .getDefiningOp<arith::ConstantIntOp>();
        auto ub = innerLoop.getUpperBound()
                      .getDefiningOp<arith::ConstantIntOp>();
        auto step = innerLoop.getStep()
                        .getDefiningOp<arith::ConstantIntOp>();
        if (lb && ub && step && step.value() > 0)
          tripCount = (ub.value() - lb.value() + step.value() - 1) /
                      step.value();
        node.latency = innerSched->II * tripCount;
        node.selfLatency = 0;
        node.pipeline = HWPipeline::NONE;
      } else {
        node.selfLatency = 10000;
        node.latency = 10000;
        node.pipeline = HWPipeline::TC;
      }

      ddg.nodes.push_back(node);
      ddg.opToIdx[&op] = node.idx;
      continue;
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
      // Exception: for MEM → local_alloc edges, use transferLatency (the TMA
      // transfer time) instead of the full async latency. local_alloc is a
      // bookkeeping op that represents data arrival — it must wait for the
      // transfer to complete, but not for the async DRAM overhead that only
      // applies to the MMA consumer.
      int edgeLatency = ddg.nodes[srcIdx].latency;
      if (ddg.nodes[srcIdx].pipeline == HWPipeline::MEM &&
          isa<triton::gpu::LocalAllocOp>(node.op)) {
        edgeLatency = ddg.nodes[srcIdx].transferLatency;
      }
      ddg.addEdge(srcIdx, node.idx, edgeLatency, /*distance=*/0);
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
      auto userIt = ddg.opToIdx.find(user);
      if (userIt == ddg.opToIdx.end())
        continue;
      // Loop-carried back-edge uses full latency so RecMII reflects
      // the true recurrence depth. For MMA with accumulator, the next
      // iteration can't read the result until the current MMA completes.
      int backEdgeLat = ddg.nodes[srcIdx].latency;
      ddg.addEdge(srcIdx, userIt->second, backEdgeLat,
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
  // Super-node latency is a hard lower bound: one outer iteration can't
  // finish faster than its inner loop. Without this, II is too small
  // and the schedule produces hundreds of stages.
  int superNodeII = 0;
  for (const auto &node : nodes)
    if (node.isSuperNode)
      superNodeII = std::max(superNodeII, node.latency);
  int minII = std::max({resMII, recMII, superNodeII});
  LLVM_DEBUG(llvm::dbgs() << "[DDG] ResMII=" << resMII << " RecMII=" << recMII
                          << " SuperNodeII=" << superNodeII
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
