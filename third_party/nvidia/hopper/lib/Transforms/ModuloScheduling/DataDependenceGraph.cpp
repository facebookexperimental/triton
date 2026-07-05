// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "DataDependenceGraph.h"
#include "ModuloReservationTable.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <queue>

#define DEBUG_TYPE "modulo-scheduling-ddg"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::triton::gpu {

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
  node.minWarps = info.minWarps;
  node.occupancy = info.occupancy;
  node.tmemAllocCols = model.getAccumulatorAllocCols(op);
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
        // Use int64_t through the computation so a large or reversed
        // iteration space can't silently overflow `int`. Clamp the
        // final value to >=1 — an empty or reversed range (`ub <= lb`
        // with positive step) would otherwise produce a non-positive
        // latency and silently break the `superNodeII` floor in
        // `computeMinII`.
        int64_t tripCount = 32; // fallback for dynamic bounds
        auto lb =
            innerLoop.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
        auto ub =
            innerLoop.getUpperBound().getDefiningOp<arith::ConstantIntOp>();
        auto step = innerLoop.getStep().getDefiningOp<arith::ConstantIntOp>();
        if (lb && ub && step && step.value() > 0) {
          int64_t lbV = lb.value();
          int64_t ubV = ub.value();
          int64_t stepV = step.value();
          tripCount = std::max<int64_t>(1, (ubV - lbV + stepV - 1) / stepV);
        } else {
          innerLoop.emitWarning(
              "modulo schedule: inner-loop bounds are not compile-time "
              "constants; using a default trip count of 32 — schedule "
              "decisions may be suboptimal");
        }
        node.latency = static_cast<int>(
            std::min<int64_t>(INT_MAX, innerSched->II * tripCount));
        node.selfLatency = 0;
        node.pipeline = HWPipeline::NONE;

        // Prologue latency: cycles before the first TC op fires inside the
        // inner loop. The outer scheduler uses this to overlap outer-loop
        // MEM ops with the inner loop's pre-MMA setup.
        int tcStart = innerSched->II;
        for (const auto &n : innerDDG.getNodes()) {
          if (n.pipeline == HWPipeline::TC) {
            auto it = innerSched->nodeToCycle.find(n.idx);
            if (it != innerSched->nodeToCycle.end())
              tcStart = std::min(tcStart, it->second);
          }
        }
        node.prologueLatency = tcStart;
      } else {
        // Inner-loop scheduling failed. Emit a diagnostic and propagate
        // a clearly-infeasible latency so the outer schedule fails too
        // instead of silently producing garbage off a fabricated
        // ~10k-cycle synthetic node.
        innerLoop.emitWarning(
            "modulo schedule: failed to schedule inner loop — outer "
            "schedule will be marked infeasible");
        node.selfLatency = INT_MAX / 2;
        node.latency = INT_MAX / 2;
        node.pipeline = HWPipeline::NONE;
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
      // Edge latency = producer's full latency (time until consumer can read
      // the result). Uniform rule with no per-pair exceptions: each op carries
      // a single `latency` that captures full delivery, and edges just
      // propagate it.
      int edgeLatency = ddg.nodes[srcIdx].latency;
      ddg.addEdge(srcIdx, node.idx, edgeLatency, /*distance=*/0);
    }
  }

  // Phase 2.5: Memory-based edges for super-nodes.
  // The DDG only captures SSA def-use edges. But super-nodes (inner
  // loops) can write to memory (TMEM via MMA) that later ops read
  // (tmem_load). Without an explicit edge, the scheduler places
  // tmem_load BEFORE the K-loop finishes, breaking the store overlap.
  //
  // Detect: if a super-node's inner loop contains MMA writing to a
  // TMEM memdesc, and a tmem_load outside the loop reads the same
  // memdesc, add an edge super-node → tmem_load.
  for (auto &node : ddg.nodes) {
    if (!node.isSuperNode)
      continue;
    auto innerLoop = dyn_cast<scf::ForOp>(node.op);
    if (!innerLoop)
      continue;
    // Find accumulator buffers written by an MMA inside the inner loop
    // (target-specific, via the latency model — e.g. Blackwell TMEM).
    llvm::DenseSet<Value> accWritten;
    innerLoop.walk([&](Operation *op) {
      if (Value w = model.getAccumulatorWrite(op))
        accWritten.insert(w);
    });
    // Find ops outside the inner loop that read the same accumulator.
    for (auto &other : ddg.nodes) {
      if (other.idx == node.idx)
        continue;
      Value accRead = model.getAccumulatorRead(other.op);
      if (accRead && accWritten.count(accRead)) {
        ddg.addEdge(node.idx, other.idx, node.latency, /*distance=*/0);
      }
    }
  }

  // Phase 2.75: Serially-dependent MMA correction. The latency model's TC
  // `occupancy` is the ISSUE-TO-ISSUE throughput of the pipelined tensor
  // core — correct for a lone MMA accumulating across iterations (GEMM),
  // where back-to-back issues genuinely overlap. But when a loop contains
  // MULTIPLE distinct MMAs connected by a dependence chain (FA: QK-MMA →
  // softmax → PV-MMA), one iteration's MMAs cannot overlap each other on the
  // TC, and the cross-iteration overlap is capped by the serial chain
  // between them. Modeling both as fully pipelined understates TC pressure,
  // drops II below what the downstream structure can exploit, and (measured
  // on FA fwd) pushes the softmax chain across a stage boundary — the
  // partitioner then cuts it with a 32KB/iter cross-WG channel, 5x slower.
  // Restore occupancy = full latency for TC nodes when any TC node reaches
  // another TC node through intra-iteration (distance-0) dependence edges.
  // Runs BEFORE Phase 3 so loop-carried TC→TC back-edges see the corrected
  // occupancy.
  {
    SmallVector<unsigned> tcNodes;
    for (const auto &node : ddg.nodes)
      if (node.pipeline == HWPipeline::TC)
        tcNodes.push_back(node.idx);
    if (tcNodes.size() > 1) {
      // Forward reachability over distance-0 edges from each TC node.
      auto reaches = [&](unsigned from, unsigned to) {
        llvm::DenseSet<unsigned> seen;
        SmallVector<unsigned> stack{from};
        seen.insert(from);
        while (!stack.empty()) {
          unsigned cur = stack.pop_back_val();
          for (const auto &e : ddg.edges) {
            if (e.srcIdx != cur || e.distance != 0)
              continue;
            if (e.dstIdx == to)
              return true;
            if (seen.insert(e.dstIdx).second)
              stack.push_back(e.dstIdx);
          }
        }
        return false;
      };
      bool serialChain = false;
      for (unsigned a : tcNodes) {
        for (unsigned b : tcNodes) {
          if (a != b && reaches(a, b)) {
            serialChain = true;
            break;
          }
        }
        if (serialChain)
          break;
      }
      if (serialChain)
        for (unsigned idx : tcNodes)
          ddg.nodes[idx].occupancy =
              std::max(ddg.nodes[idx].occupancy, ddg.nodes[idx].latency);
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
      // the true recurrence depth: a consumer that READS the value (e.g.
      // tmem_load, or a CUDA op on an iter_arg tensor) can't start until the
      // producer completes.
      //
      // Exception: TC→TC token chains. Successive MMAs accumulating into the
      // same TMEM buffer are ordered and pipelined by the tensor core itself —
      // the next MMA issues at throughput rate (occupancy), not after the
      // previous one's full completion. Charging full latency here doubled
      // GEMM RecMII (559 vs the ~256-cycle true issue interval) and, through
      // II, halved the TMA prefetch ring depth.
      int backEdgeLat = ddg.nodes[srcIdx].latency;
      const auto &srcNode = ddg.nodes[srcIdx];
      const auto &dstNode = ddg.nodes[userIt->second];
      if (srcNode.pipeline == HWPipeline::TC &&
          dstNode.pipeline == HWPipeline::TC)
        backEdgeLat = std::max(pipelineOccupancy(srcNode), 1);
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

void DataDependenceGraph::applyDataPartition(
    const llvm::DenseMap<Operation *, DataPartitionInfo> &mmaInfo) {
  if (mmaInfo.empty())
    return;
  for (auto &node : nodes) {
    if (!node.op)
      continue;
    auto it = mmaInfo.find(node.op);
    if (it == mmaInfo.end() || it->second.count <= 1)
      continue;
    const DataPartitionInfo &info = it->second;
    node.partitionCount = info.count;
    node.partitionDim = info.dim;
    node.mSize = info.mSize;
    // The bundle issues `count` MMAs back-to-back on the tensor core, so it
    // ties up the TC pipeline `count`× as long. Scale occupancy so ResMII
    // reflects the N hardware issues (the result latency is unchanged).
    node.occupancy =
        std::max(pipelineOccupancy(node), 1) * static_cast<int>(info.count);
  }
}

int DataDependenceGraph::computeResMII() const {
  // Per-pipeline busy time (cycles) bounds II from below: II ≥ pipeLoad / N
  // where N is the number of identical units on the pipeline (assumed 1).
  llvm::DenseMap<HWPipeline, int> pipeLoad;
  for (const auto &node : nodes) {
    if (node.pipeline == HWPipeline::NONE)
      continue;
    pipeLoad[node.pipeline] += pipelineOccupancy(node);
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
