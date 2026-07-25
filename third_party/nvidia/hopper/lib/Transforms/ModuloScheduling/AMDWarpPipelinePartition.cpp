// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// AMD Warp-Pipeline Partition — Steps 4.7 + 4.8 from the AMD Global
// Instruction Scheduling design doc.
//
// See AMDWarpPipelinePartition.h for the public API.

#include "AMDWarpPipelinePartition.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#define DEBUG_TYPE "amd-warp-pipeline-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {

namespace {

// AMD barrier overhead costs (cycles).
constexpr int kVmcntOverhead = 10;
constexpr int kSbarrierOverhead = 40;

// Step 4.8c tuning constants.
constexpr double kLambdaWeight = 1.5;
constexpr double kAsyncLoadPressure = 0.5;

// The AMD pipelines relevant for clustering.
constexpr HWPipeline kAMDPipelines[] = {HWPipeline::MFMA, HWPipeline::LDS,
                                        HWPipeline::GLOBAL, HWPipeline::VALU};

// ============================================================================
// Step 4.7 helpers
// ============================================================================

using PipelinePairKey = std::pair<HWPipeline, HWPipeline>;

static int barrierCostForEdge(const DDGEdge &edge,
                              const DataDependenceGraph &ddg) {
  const DDGNode &src = ddg.getNode(edge.srcIdx);
  const DDGNode &dst = ddg.getNode(edge.dstIdx);
  bool srcIsLDS = src.pipeline == HWPipeline::LDS;
  bool dstIsLDS = dst.pipeline == HWPipeline::LDS;
  if (srcIsLDS != dstIsLDS)
    return kSbarrierOverhead;
  return kVmcntOverhead;
}

static llvm::DenseMap<PipelinePairKey, double>
computeSeparationCost(const DataDependenceGraph &ddg,
                      const ModuloScheduleResult &schedule) {
  llvm::DenseMap<PipelinePairKey, double> coupling;
  int II = schedule.II;

  for (const auto &edge : ddg.getEdges()) {
    const DDGNode &src = ddg.getNode(edge.srcIdx);
    const DDGNode &dst = ddg.getNode(edge.dstIdx);

    if (src.pipeline == dst.pipeline)
      continue;
    if (src.pipeline == HWPipeline::NONE || dst.pipeline == HWPipeline::NONE)
      continue;

    auto srcIt = schedule.nodeToCycle.find(edge.srcIdx);
    auto dstIt = schedule.nodeToCycle.find(edge.dstIdx);
    if (srcIt == schedule.nodeToCycle.end() ||
        dstIt == schedule.nodeToCycle.end())
      continue;

    // Fold the loop-carried distance into the gap so a distance>=1 edge
    // reflects its true cross-iteration separation rather than collapsing to a
    // spurious
    // <=0 gap (which would clamp to 1 and over-weight the coupling).
    int cycleGap =
        dstIt->second - srcIt->second + static_cast<int>(edge.distance) * II;
    if (cycleGap <= 0)
      cycleGap = 1;

    int cost = barrierCostForEdge(edge, ddg);
    coupling[{src.pipeline, dst.pipeline}] +=
        static_cast<double>(cost) / cycleGap;
  }

  return coupling;
}

static llvm::SmallVector<unsigned>
topologicalSort(llvm::ArrayRef<unsigned> nodeIndices,
                const DataDependenceGraph &ddg) {
  llvm::DenseSet<unsigned> inSet;
  for (unsigned idx : nodeIndices)
    inSet.insert(idx);

  llvm::DenseMap<unsigned, int> inDegree;
  for (unsigned idx : nodeIndices) {
    int deg = 0;
    for (const auto *edge : ddg.getInEdges(idx)) {
      if (edge->distance == 0 && inSet.count(edge->srcIdx))
        ++deg;
    }
    inDegree[idx] = deg;
  }

  llvm::SmallVector<unsigned> queue;
  for (unsigned idx : nodeIndices) {
    if (inDegree[idx] == 0)
      queue.push_back(idx);
  }

  llvm::SmallVector<unsigned> sorted;
  sorted.reserve(nodeIndices.size());
  while (!queue.empty()) {
    unsigned cur = queue.pop_back_val();
    sorted.push_back(cur);
    for (const auto *edge : ddg.getOutEdges(cur)) {
      if (edge->distance > 0 || !inSet.count(edge->dstIdx))
        continue;
      if (--inDegree[edge->dstIdx] == 0)
        queue.push_back(edge->dstIdx);
    }
  }

  if (sorted.size() < nodeIndices.size()) {
    // A distance-0 subgraph should be acyclic; a leftover means DDG
    // construction introduced an intra-iteration cycle. Log it (downstream
    // makespan/ criticality will under-approximate deps for these nodes) and
    // append the unordered remainder so callers still see every node.
    LDBG("topologicalSort: distance-0 cycle detected, "
         << (nodeIndices.size() - sorted.size()) << " node(s) left unordered");
    llvm::DenseSet<unsigned> visited(sorted.begin(), sorted.end());
    for (unsigned idx : nodeIndices) {
      if (!visited.count(idx))
        sorted.push_back(idx);
    }
  }

  return sorted;
}

static int computeMultiPipelineMakespan(llvm::ArrayRef<unsigned> nodeIndices,
                                        const DataDependenceGraph &ddg) {
  auto sorted = topologicalSort(nodeIndices, ddg);

  llvm::DenseSet<unsigned> inSet;
  for (unsigned idx : nodeIndices)
    inSet.insert(idx);

  llvm::DenseMap<HWPipeline, int> pipeAvail;
  llvm::DenseMap<unsigned, int> opStart;

  for (unsigned idx : sorted) {
    const DDGNode &node = ddg.getNode(idx);

    int dataReady = 0;
    for (const auto *edge : ddg.getInEdges(idx)) {
      if (edge->distance > 0)
        continue;
      if (!inSet.count(edge->srcIdx))
        continue;
      auto it = opStart.find(edge->srcIdx);
      if (it != opStart.end()) {
        const DDGNode &pred = ddg.getNode(edge->srcIdx);
        dataReady = std::max(dataReady, it->second + pred.latency);
      }
    }

    int pipeReady = pipeAvail.lookup(node.pipeline);
    int start = std::max(dataReady, pipeReady);
    opStart[idx] = start;
    pipeAvail[node.pipeline] = start + pipelineOccupancy(node);
  }

  int makespan = 0;
  for (unsigned idx : nodeIndices) {
    const DDGNode &node = ddg.getNode(idx);
    auto it = opStart.find(idx);
    if (it != opStart.end())
      makespan = std::max(makespan, it->second + pipelineOccupancy(node));
  }
  return makespan;
}

static double
mergeSavings(const AMDClusterInfo &c1, const AMDClusterInfo &c2,
             const llvm::DenseMap<PipelinePairKey, double> &coupling) {
  double savings = 0;
  for (HWPipeline p1 : c1.pipelines) {
    for (HWPipeline p2 : c2.pipelines) {
      auto it1 = coupling.find({p1, p2});
      if (it1 != coupling.end())
        savings += it1->second;
      auto it2 = coupling.find({p2, p1});
      if (it2 != coupling.end())
        savings += it2->second;
    }
  }
  return savings;
}

// ============================================================================
// Step 4.8 helpers
// ============================================================================

/// Look up per-cluster per-pipeline occupancy from the cluster's utilization
/// map. Returns 0 if the pipeline isn't present.
static double getClusterOcc(const AMDClusterInfo &c, HWPipeline pipe) {
  auto it = c.utilization.find(pipe);
  return (it != c.utilization.end()) ? it->second : 0.0;
}

/// Find cluster by ID in the result vector.
static const AMDClusterInfo *
findCluster(const llvm::SmallVector<AMDClusterInfo, 4> &clusters, int id) {
  for (const auto &c : clusters) {
    if (c.clusterId == id)
      return &c;
  }
  return nullptr;
}

/// Step 4.8a: Compute per-op ASAP/ALAP slack and criticality.
static llvm::DenseMap<unsigned, double>
computeCriticality(const DataDependenceGraph &ddg,
                   const ModuloScheduleResult &schedule) {
  int II = schedule.II;

  llvm::SmallVector<unsigned> allIndices;
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i)
    allIndices.push_back(i);
  auto sorted = topologicalSort(allIndices, ddg);

  // ASAP.
  llvm::DenseMap<unsigned, int> asap;
  for (unsigned idx : sorted) {
    int earliest = 0;
    for (const auto *edge : ddg.getInEdges(idx)) {
      auto predIt = asap.find(edge->srcIdx);
      if (predIt == asap.end())
        continue;
      int predReady = predIt->second + edge->latency -
                      static_cast<int>(edge->distance) * II;
      earliest = std::max(earliest, predReady);
    }
    asap[idx] = earliest;
  }

  // ALAP.
  int maxAsap = 0;
  for (auto &[idx, val] : asap)
    maxAsap = std::max(maxAsap, val);

  llvm::DenseMap<unsigned, int> alap;
  for (auto it = sorted.rbegin(), end = sorted.rend(); it != end; ++it) {
    unsigned idx = *it;
    int latest = maxAsap;
    for (const auto *edge : ddg.getOutEdges(idx)) {
      auto succIt = alap.find(edge->dstIdx);
      if (succIt == alap.end())
        continue;
      int succStart = succIt->second - edge->latency +
                      static_cast<int>(edge->distance) * II;
      latest = std::min(latest, succStart);
    }
    alap[idx] = latest;
  }

  // Criticality.
  int maxSlack = 0;
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i)
    maxSlack = std::max(maxSlack, alap.lookup(i) - asap.lookup(i));

  constexpr double eps = 1e-6;
  llvm::DenseMap<unsigned, double> crit;
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i) {
    int slack = alap.lookup(i) - asap.lookup(i);
    crit[i] = 1.0 - static_cast<double>(slack) / (maxSlack + eps);
  }
  return crit;
}

static int
computePingpongOffset(const llvm::SmallVector<AMDClusterInfo, 4> &clusters) {
  int N = clusters.size();
  if (N <= 1)
    return 0;

  int bestDelta = 1;
  double bestContention = std::numeric_limits<double>::max();

  for (int delta = 1; delta < N; ++delta) {
    double contention = 0;
    for (const auto &c : clusters) {
      int oppositeId = (c.clusterId + delta) % N;
      const auto *opp = findCluster(clusters, oppositeId);
      if (!opp)
        continue;
      for (HWPipeline pipe : kAMDPipelines) {
        double coOcc = getClusterOcc(c, pipe) + getClusterOcc(*opp, pipe);
        if (coOcc > 1.0)
          contention += coOcc - 1.0;
      }
    }
    LDBG("Delta=" << delta << " contention=" << contention);
    if (contention < bestContention) {
      bestContention = contention;
      bestDelta = delta;
    }
  }

  LDBG("Best ping-pong offset: " << bestDelta);
  return bestDelta;
}

static llvm::SmallDenseSet<HWPipeline, 4>
findContendedPipelines(const llvm::SmallVector<AMDClusterInfo, 4> &clusters,
                       int delta) {
  int N = clusters.size();
  llvm::SmallDenseSet<HWPipeline, 4> contended;

  for (const auto &c : clusters) {
    int oppositeId = (c.clusterId + delta) % N;
    const auto *opp = findCluster(clusters, oppositeId);
    if (!opp)
      continue;
    for (HWPipeline pipe : kAMDPipelines) {
      if (getClusterOcc(c, pipe) > 0 && getClusterOcc(*opp, pipe) > 0)
        contended.insert(pipe);
    }
  }

  LLVM_DEBUG({
    DBGS() << "Contended pipelines:";
    for (HWPipeline p : contended)
      llvm::dbgs() << " " << getPipelineName(p);
    llvm::dbgs() << "\n";
  });

  return contended;
}

} // namespace

// ============================================================================
// Step 4.7: partitionForAMDWarpPipeline
// ============================================================================

AMDWarpPipelineResult
partitionForAMDWarpPipeline(const DataDependenceGraph &ddg,
                            const ModuloScheduleResult &schedule) {
  int II = schedule.II;
  if (II <= 0) {
    LDBG("II <= 0, returning single cluster");
    AMDWarpPipelineResult result;
    AMDClusterInfo single;
    single.clusterId = 0;
    for (const auto &node : ddg.getNodes())
      if (node.pipeline != HWPipeline::NONE)
        single.nodeIndices.push_back(node.idx);
    result.clusters.push_back(std::move(single));
    return result;
  }

  // Per-pipeline utilization.
  llvm::DenseMap<HWPipeline, double> pipeUtil;
  for (HWPipeline pipe : kAMDPipelines) {
    int busy = 0;
    for (const auto &node : ddg.getNodes())
      if (node.pipeline == pipe)
        busy += pipelineOccupancy(node);
    pipeUtil[pipe] = static_cast<double>(busy) / II;
  }

  // Initialize: one cluster per active pipeline.
  llvm::SmallVector<AMDClusterInfo, 4> clusters;
  for (HWPipeline pipe : kAMDPipelines) {
    AMDClusterInfo c;
    bool hasOps = false;
    for (const auto &node : ddg.getNodes()) {
      if (node.pipeline == pipe) {
        c.nodeIndices.push_back(node.idx);
        hasOps = true;
      }
    }
    if (!hasOps)
      continue;
    c.clusterId = clusters.size();
    c.pipelines.insert(pipe);
    c.utilization[pipe] = pipeUtil[pipe];
    clusters.push_back(std::move(c));
  }

  LDBG("Initial clusters: " << clusters.size());

  if (clusters.size() <= 1) {
    AMDWarpPipelineResult result;
    if (!clusters.empty())
      result.clusters.push_back(std::move(clusters[0]));
    return result;
  }

  auto coupling = computeSeparationCost(ddg, schedule);

  LLVM_DEBUG({
    for (auto &[key, cost] : coupling)
      DBGS() << "coupling(" << getPipelineName(key.first) << ", "
             << getPipelineName(key.second) << ") = " << cost << "\n";
  });

  // Greedy agglomerative merging.
  bool merged = true;
  while (merged && clusters.size() > 1) {
    merged = false;
    int bestI = -1, bestJ = -1;
    double bestSavings = 0;

    for (int i = 0, e = clusters.size(); i < e; ++i) {
      for (int j = i + 1; j < e; ++j) {
        double savings = mergeSavings(clusters[i], clusters[j], coupling);
        if (savings <= bestSavings)
          continue;

        // Fast reject: any single pipeline oversubscribed?
        bool feasible = true;
        llvm::SmallDenseMap<HWPipeline, double, 4> mergedUtil;
        for (auto &[pipe, u] : clusters[i].utilization)
          mergedUtil[pipe] += u;
        for (auto &[pipe, u] : clusters[j].utilization)
          mergedUtil[pipe] += u;
        for (auto &[pipe, u] : mergedUtil) {
          if (u > 1.0) {
            feasible = false;
            break;
          }
        }
        if (!feasible)
          continue;

        // Precise check: multi-pipeline makespan.
        llvm::SmallVector<unsigned> mergedOps;
        mergedOps.append(clusters[i].nodeIndices.begin(),
                         clusters[i].nodeIndices.end());
        mergedOps.append(clusters[j].nodeIndices.begin(),
                         clusters[j].nodeIndices.end());
        int makespan = computeMultiPipelineMakespan(mergedOps, ddg);
        if (makespan > II)
          continue;

        bestI = i;
        bestJ = j;
        bestSavings = savings;
      }
    }

    if (bestI < 0)
      break;

    LDBG("Merging clusters " << bestI << " and " << bestJ
                             << " (savings=" << bestSavings << ")");

    clusters[bestI].nodeIndices.append(clusters[bestJ].nodeIndices.begin(),
                                       clusters[bestJ].nodeIndices.end());
    for (HWPipeline p : clusters[bestJ].pipelines)
      clusters[bestI].pipelines.insert(p);
    for (auto &[pipe, u] : clusters[bestJ].utilization)
      clusters[bestI].utilization[pipe] += u;
    clusters.erase(clusters.begin() + bestJ);
    merged = true;
  }

  // Sort by earliest scheduled cycle, then assign sequential IDs. Ties break on
  // the cluster's smallest node index (unique per cluster) so the final IDs —
  // which feed ping-pong offset, priority, and the emitted op attributes — are
  // deterministic across builds even when two clusters share an earliest cycle.
  auto clusterSortKey =
      [&](const AMDClusterInfo &c) -> std::pair<int, unsigned> {
    int earliest = std::numeric_limits<int>::max();
    unsigned minIdx = std::numeric_limits<unsigned>::max();
    for (unsigned idx : c.nodeIndices) {
      auto it = schedule.nodeToCycle.find(idx);
      if (it != schedule.nodeToCycle.end())
        earliest = std::min(earliest, it->second);
      minIdx = std::min(minIdx, idx);
    }
    return {earliest, minIdx};
  };
  llvm::sort(clusters, [&](const AMDClusterInfo &a, const AMDClusterInfo &b) {
    return clusterSortKey(a) < clusterSortKey(b);
  });
  for (int i = 0, e = clusters.size(); i < e; ++i)
    clusters[i].clusterId = i;

  LDBG("Final clusters: " << clusters.size());

  AMDWarpPipelineResult result;
  result.clusters = std::move(clusters);
  return result;
}

// ============================================================================
// Step 4.8: assignAMDWarpPipelinePriorities
// ============================================================================

void assignAMDWarpPipelinePriorities(AMDWarpPipelineResult &result,
                                     const DataDependenceGraph &ddg,
                                     const ModuloScheduleResult &schedule) {
  if (result.clusters.size() < 2)
    return;

  // 4.8a: per-op criticality from ASAP/ALAP slack.
  auto crit = computeCriticality(ddg, schedule);

  // 4.8b: ping-pong offset minimizing pipeline contention.
  int delta = computePingpongOffset(result.clusters);
  result.pingpongOffset = delta;

  auto contended = findContendedPipelines(result.clusters, delta);

  // 4.8c: pscore(c) = U(c) - lambda * M(c).
  llvm::SmallVector<std::pair<int, double>> clusterScores;

  for (const auto &c : result.clusters) {
    double M = 0;
    for (HWPipeline pipe : contended)
      M += getClusterOcc(c, pipe);

    double U = 0;
    for (unsigned idx : c.nodeIndices) {
      const DDGNode &node = ddg.getNode(idx);
      double nodeCrit = crit.lookup(idx);
      if (node.pipeline == HWPipeline::GLOBAL)
        nodeCrit += kAsyncLoadPressure;
      U = std::max(U, nodeCrit);
    }

    double pscore = U - kLambdaWeight * M;
    clusterScores.push_back({c.clusterId, pscore});

    LDBG("Cluster " << c.clusterId << ": M=" << M << " U=" << U
                    << " pscore=" << pscore);
  }

  // Sort by pscore; break ties on clusterId so equal-scored clusters get a
  // stable s_setprio ranking across builds (llvm::sort is not stable).
  llvm::sort(clusterScores, [](const auto &a, const auto &b) {
    return std::tie(a.second, a.first) < std::tie(b.second, b.first);
  });

  int numLevels = std::min(static_cast<int>(clusterScores.size()), 4);
  int numClusters = clusterScores.size();

  for (int rank = 0; rank < numClusters; ++rank) {
    int clusterId = clusterScores[rank].first;
    int prio =
        (numClusters > 1) ? (rank * (numLevels - 1)) / (numClusters - 1) : 0;
    for (auto &c : result.clusters) {
      if (c.clusterId == clusterId) {
        c.sSetprio = prio;
        LDBG("Cluster " << clusterId << " -> s_setprio=" << prio);
        break;
      }
    }
  }
}

} // namespace mlir::triton::gpu
