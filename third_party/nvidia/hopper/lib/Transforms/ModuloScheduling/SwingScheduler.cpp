// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Swing Modulo Scheduling (SMS)
//
// J. Llosa, A. González, E. Ayguadé, M. Valero,
// "Swing Modulo Scheduling: A Lifetime-Sensitive Approach", PACT 1996.
//
// Simplifications relative to the paper:
//
// 1. No recurrence-aware ordering. The paper identifies SCCs, orders them
//    by RecMII contribution, and schedules the most critical recurrence
//    first. We use a simple BFS from the minimum-slack node. This works
//    for GEMM (trivial single-node recurrence) but may not prioritize
//    correctly when multiple recurrences compete (e.g., FA backward with
//    accumulator, softmax state, and pointer update recurrences).
//
// 2. Fallback on placement failure. When the directional scan (top-down
//    or bottom-up) finds no free slot, we fall back to findFreeSlot from
//    earliest. The paper would fail at this II and increment. Our fallback
//    avoids unnecessary II inflation but may place a bottom-up node early,
//    defeating the register pressure benefit.
//
// 3. The BFS swing expansion follows all DDG edges including loop-carried
//    ones (distance > 0). The paper's ordering only follows distance-0
//    edges. This may add nodes based on cross-iteration dependencies
//    rather than intra-iteration structure.
//
// These simplifications are acceptable for the current use case (GPU
// inner loops with ≤20 ops and ≤4 pipeline resources) where the graphs
// are small enough that suboptimal ordering rarely affects the achieved II.

#include "SwingScheduler.h"

#include "llvm/Support/Debug.h"
#include <climits>

#define DEBUG_TYPE "modulo-scheduling-sms"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::triton::gpu {

/// Get the duration (pipeline occupancy slots) for a DDG node.
static int getNodeDuration(const DDGNode &node) {
  if (node.pipeline == HWPipeline::NONE)
    return 1;
  return std::max(node.selfLatency, 1);
}

/// Compute the earliest start time for a node given its predecessors'
/// scheduled cycles, respecting loop-carried distances.
static int computeEarliestStart(unsigned nodeIdx,
                                const DataDependenceGraph &ddg,
                                const llvm::DenseMap<unsigned, int> &scheduled,
                                int II) {
  int earliest = 0;
  for (const auto *edge : ddg.getInEdges(nodeIdx)) {
    auto it = scheduled.find(edge->srcIdx);
    if (it == scheduled.end())
      continue;
    int constraint =
        it->second + edge->latency - static_cast<int>(edge->distance) * II;
    earliest = std::max(earliest, constraint);
  }
  return earliest;
}

/// Compute ASAP (as-soon-as-possible) times via forward relaxation.
/// Includes loop-carried edges with II-dependent bounds:
///   ASAP[dst] >= ASAP[src] + latency - distance * II
static llvm::DenseMap<unsigned, int> computeASAP(const DataDependenceGraph &ddg,
                                                 int II) {
  llvm::DenseMap<unsigned, int> asap;
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i)
    asap[i] = 0;
  bool changed = true;
  int iter = 0;
  constexpr int maxIter = 1000;
  while (changed && iter < maxIter) {
    changed = false;
    iter++;
    for (const auto &edge : ddg.getEdges()) {
      int candidate = asap[edge.srcIdx] + edge.latency -
                      static_cast<int>(edge.distance) * II;
      if (candidate > asap[edge.dstIdx]) {
        asap[edge.dstIdx] = candidate;
        changed = true;
      }
    }
  }
  if (iter >= maxIter)
    LLVM_DEBUG(DBGS() << "ASAP did not converge after " << maxIter
                      << " iterations\n");
  return asap;
}

/// Compute ALAP (as-late-as-possible) times via backward relaxation.
/// Includes loop-carried edges with II-dependent bounds:
///   ALAP[src] <= ALAP[dst] - latency + distance * II
static llvm::DenseMap<unsigned, int>
computeALAP(const DataDependenceGraph &ddg,
            const llvm::DenseMap<unsigned, int> &asap, int II) {
  int horizon = 0;
  for (auto &[idx, t] : asap)
    horizon = std::max(horizon, t);

  llvm::DenseMap<unsigned, int> alap;
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i)
    alap[i] = horizon;
  bool changed = true;
  int iter = 0;
  constexpr int maxIter = 1000;
  while (changed && iter < maxIter) {
    changed = false;
    iter++;
    for (const auto &edge : ddg.getEdges()) {
      int candidate = alap[edge.dstIdx] - edge.latency +
                      static_cast<int>(edge.distance) * II;
      if (candidate < alap[edge.srcIdx]) {
        alap[edge.srcIdx] = candidate;
        changed = true;
      }
    }
  }
  if (iter >= maxIter)
    LLVM_DEBUG(DBGS() << "ALAP did not converge after " << maxIter
                      << " iterations\n");
  return alap;
}

/// Compute the latest start for a node given already-scheduled successors.
static int computeLatestStart(unsigned nodeIdx, const DataDependenceGraph &ddg,
                              const llvm::DenseMap<unsigned, int> &scheduled,
                              int II) {
  int latest = INT_MAX;
  for (const auto *edge : ddg.getOutEdges(nodeIdx)) {
    auto it = scheduled.find(edge->dstIdx);
    if (it == scheduled.end())
      continue;
    int constraint =
        it->second - edge->latency + static_cast<int>(edge->distance) * II;
    latest = std::min(latest, constraint);
  }
  return latest;
}

FailureOr<ModuloScheduleResult> runSMS(const DataDependenceGraph &ddg,
                                       int minII, int maxII) {
  // Cap maxII to avoid spending too long on large DDGs.
  maxII = std::min(maxII, minII + 10);

  for (int II = minII; II <= maxII; ++II) {
    // Recompute ASAP/ALAP for each II — loop-carried edge constraints
    // depend on II: ASAP[v] >= ASAP[u] + latency - distance * II.
    auto asap = computeASAP(ddg, II);
    auto alap = computeALAP(ddg, asap, II);

    LLVM_DEBUG({
      DBGS() << "II=" << II << " ASAP/ALAP:\n";
      for (unsigned i = 0; i < ddg.getNumNodes(); ++i) {
        DBGS() << "  N" << i << " ASAP=" << asap[i] << " ALAP=" << alap[i]
               << " slack=" << (alap[i] - asap[i])
               << " pipeline=" << getPipelineName(ddg.getNode(i).pipeline)
               << "\n";
      }
    });

    ModuloReservationTable table{II};
    llvm::DenseMap<unsigned, int> scheduled;
    bool success = true;

    // ── Ordering phase ─────────────────────────────────────────────
    // Seed with minimum-slack node, then BFS-expand: successors
    // (top-down) then predecessors (bottom-up), sorted by slack.
    llvm::SmallVector<std::pair<unsigned, bool>> swingOrder;
    llvm::DenseSet<unsigned> inOrder;

    unsigned seed = 0;
    int seedSlack = INT_MAX;
    for (unsigned i = 0; i < ddg.getNumNodes(); ++i) {
      int slack = alap[i] - asap[i];
      if (slack < seedSlack) {
        seedSlack = slack;
        seed = i;
      }
    }
    swingOrder.push_back({seed, true});
    inOrder.insert(seed);

    for (unsigned i = 0; i < swingOrder.size(); ++i) {
      unsigned cur = swingOrder[i].first;

      // Successors → top-down
      llvm::SmallVector<std::pair<int, unsigned>> succs;
      for (unsigned s : ddg.getNode(cur).succs)
        if (inOrder.insert(s).second)
          succs.push_back({alap[s] - asap[s], s});
      llvm::sort(succs);
      for (auto &[sl, idx] : succs)
        swingOrder.push_back({idx, true});

      // Predecessors → bottom-up
      llvm::SmallVector<std::pair<int, unsigned>> preds;
      for (unsigned p : ddg.getNode(cur).preds)
        if (inOrder.insert(p).second)
          preds.push_back({alap[p] - asap[p], p});
      llvm::sort(preds);
      for (auto &[sl, idx] : preds)
        swingOrder.push_back({idx, false});
    }

    // ── Scheduling phase ────────────────────────────────────────────
    for (auto &[nodeIdx, topDown] : swingOrder) {
      const auto &node = ddg.getNode(nodeIdx);
      int duration = getNodeDuration(node);

      int earliest = computeEarliestStart(nodeIdx, ddg, scheduled, II);
      int latest = computeLatestStart(nodeIdx, ddg, scheduled, II);

      earliest = std::max(earliest, asap[nodeIdx]);
      if (latest == INT_MAX)
        latest = alap[nodeIdx] + II - 1;
      if (latest < earliest)
        latest = earliest + II - 1;

      int slot = -1;
      if (topDown) {
        slot = table.findFreeSlot(earliest, node.pipeline, duration);
      } else {
        for (int t = latest; t >= earliest; --t) {
          if (table.isIntervalFree(t, node.pipeline, duration)) {
            slot = t;
            break;
          }
        }
      }

      // Fallback: try anywhere from earliest.
      // The paper would fail at this II instead.
      if (slot < 0)
        slot = table.findFreeSlot(earliest, node.pipeline, duration);

      if (slot < 0) {
        success = false;
        LLVM_DEBUG(DBGS() << "  II=" << II << " FAILED to place N" << nodeIdx
                          << "\n");
        break;
      }

      table.reserve(slot, node.pipeline, nodeIdx, duration);
      scheduled[nodeIdx] = slot;
      LLVM_DEBUG(DBGS() << "  II=" << II << " placed N" << nodeIdx << " ("
                        << getPipelineName(node.pipeline) << " dur=" << duration
                        << ") at cycle=" << slot << " stage=" << slot / II
                        << (topDown ? " [top-down]" : " [bottom-up]") << "\n");
    }

    if (success) {
      LLVM_DEBUG(DBGS() << "SUCCESS at II=" << II << "\n");
      ModuloScheduleResult result;
      result.II = II;
      result.nodeToCycle = std::move(scheduled);
      return result;
    }

    LLVM_DEBUG(DBGS() << "FAILED at II=" << II << "\n");
  }

  return failure();
}

} // namespace mlir::triton::gpu
