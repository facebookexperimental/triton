// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ModuloReservationTable.h"

#include "ExhaustiveScheduler.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <climits>
#include <numeric>

#define DEBUG_TYPE "modulo-scheduling-rau"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::triton::gpu {

// ── ModuloReservationTable ──────────────────────────────────────────────────

ModuloReservationTable::ModuloReservationTable(int II) : II{II} {
  for (auto pipe :
       {HWPipeline::MEM, HWPipeline::TC, HWPipeline::CUDA, HWPipeline::SFU}) {
    table[pipe].assign(II, -1);
  }
}

bool ModuloReservationTable::isFree(int cycle, HWPipeline pipeline) const {
  if (pipeline == HWPipeline::NONE)
    return true;
  auto it = table.find(pipeline);
  if (it == table.end())
    return true;
  return it->second[cycle % II] < 0;
}

bool ModuloReservationTable::isIntervalFree(int cycle, HWPipeline pipeline,
                                            int duration) const {
  if (pipeline == HWPipeline::NONE)
    return true;
  for (int t = cycle; t < cycle + duration; ++t) {
    if (!isFree(t, pipeline))
      return false;
  }
  return true;
}

void ModuloReservationTable::reserve(int cycle, HWPipeline pipeline,
                                     unsigned nodeIdx, int duration) {
  if (pipeline == HWPipeline::NONE)
    return;
  for (int t = cycle; t < cycle + duration; ++t) {
    table[pipeline][t % II] = static_cast<int>(nodeIdx);
  }
}

void ModuloReservationTable::unreserve(int cycle, HWPipeline pipeline,
                                       int duration) {
  if (pipeline == HWPipeline::NONE)
    return;
  for (int t = cycle; t < cycle + duration; ++t) {
    table[pipeline][t % II] = -1;
  }
}

int ModuloReservationTable::getOccupant(int cycle, HWPipeline pipeline) const {
  if (pipeline == HWPipeline::NONE)
    return -1;
  auto it = table.find(pipeline);
  if (it == table.end())
    return -1;
  return it->second[cycle % II];
}

int ModuloReservationTable::findFreeSlot(int earliest, HWPipeline pipeline,
                                         int duration) const {
  if (pipeline == HWPipeline::NONE)
    return earliest;
  for (int t = earliest; t < earliest + II; ++t) {
    if (isIntervalFree(t, pipeline, duration))
      return t;
  }
  return -1;
}

// ── Rau's Iterative Modulo Scheduling ───────────────────────────────────────

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
    // constraint: dst_start >= src_start + latency - distance * II
    int constraint =
        it->second + edge->latency - static_cast<int>(edge->distance) * II;
    earliest = std::max(earliest, constraint);
  }
  return earliest;
}

static FailureOr<ModuloScheduleResult> runRauIMS(const DataDependenceGraph &ddg,
                                                 int minII, int maxII,
                                                 int maxBacktracks) {
  LLVM_DEBUG(DBGS() << "Computing critical path heights...\n");
  auto heights = ddg.computeCriticalPathHeights();
  LLVM_DEBUG(DBGS() << "Heights computed for " << heights.size() << " nodes\n");

  // Sort ALL nodes (including NONE-pipeline) by decreasing critical-path
  // height. NONE ops must be scheduled together with pipeline ops so that
  // dependency constraints (e.g., load → local_alloc → MMA) are respected.
  llvm::SmallVector<unsigned> priorityOrder;
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i)
    priorityOrder.push_back(i);
  llvm::sort(priorityOrder, [&](unsigned a, unsigned b) {
    if (heights[a] != heights[b])
      return heights[a] > heights[b];
    // Tiebreaker: lower index first (producers before consumers
    // in program order). This ensures that when a predecessor and
    // successor have equal heights, the predecessor is scheduled
    // first so its cycle is known when the successor is placed.
    return a < b;
  });

  LLVM_DEBUG({
    DBGS() << "MinII=" << minII << " MaxII=" << maxII
           << " Nodes=" << priorityOrder.size() << "\n";
    DBGS() << "ResMII=" << ddg.computeResMII()
           << " RecMII=" << ddg.computeRecMII() << "\n";
  });
  // Show per-pipeline resource usage for ResMII breakdown
  LLVM_DEBUG({
    llvm::DenseMap<HWPipeline, int> pipeLoad;
    for (const auto &node : ddg.getNodes()) {
      if (node.pipeline != HWPipeline::NONE)
        pipeLoad[node.pipeline] += std::max(node.selfLatency, 1);
    }
    for (auto &[pipe, load] : pipeLoad) {
      DBGS() << "  " << getPipelineName(pipe) << " total_load=" << load << "\n";
    }
  });

  for (int II = minII; II <= maxII; ++II) {
    ModuloReservationTable table{II};
    llvm::DenseMap<unsigned, int> scheduled;
    bool success = true;
    int backtracks = 0;

    // Use index-based iteration instead of range-for because ejection
    // may insert evicted nodes back into priorityOrder for re-scheduling.
    // Range-for would be UB (iterator invalidation on SmallVector insert).
    for (unsigned i = 0; i < priorityOrder.size(); ++i) {
      unsigned nodeIdx = priorityOrder[i];
      const auto &node = ddg.getNode(nodeIdx);
      int duration = std::max(node.selfLatency, 1); // at least 1 slot
      if (node.pipeline == HWPipeline::NONE)
        duration = 1; // NONE ops don't occupy any pipeline

      int earliest = computeEarliestStart(nodeIdx, ddg, scheduled, II);
      int slot = table.findFreeSlot(earliest, node.pipeline, duration);

      if (slot < 0 && backtracks < maxBacktracks) {
        // Rau's ejection: find the least-critical occupant in a
        // conflicting slot, evict it, place current node, then
        // re-schedule the evicted node later.
        int bestVictim = -1;
        int bestVictimHeight = INT_MAX;
        int currentHeight = heights.lookup(nodeIdx);
        for (int t = earliest; t < earliest + II; ++t) {
          int occupant = table.getOccupant(t, node.pipeline);
          if (occupant < 0)
            continue;
          int occHeight = heights.lookup(static_cast<unsigned>(occupant));
          // Only eject nodes with strictly lower priority (smaller height)
          // than the current node. This prevents priority inversion where
          // a less-critical node evicts a more-critical one.
          if (occHeight < currentHeight && occHeight < bestVictimHeight) {
            bestVictimHeight = occHeight;
            bestVictim = occupant;
          }
        }
        if (bestVictim >= 0) {
          // Evict the victim.
          const auto &victim = ddg.getNode(bestVictim);
          int victimDur = std::max(victim.selfLatency, 1);
          if (victim.pipeline == HWPipeline::NONE)
            victimDur = 1;
          int victimCycle = scheduled[bestVictim];
          table.unreserve(victimCycle, victim.pipeline, victimDur);
          scheduled.erase(bestVictim);

          // Place current node at the freed slot.
          slot = table.findFreeSlot(earliest, node.pipeline, duration);
          if (slot >= 0) {
            // Insert evicted node right after current position for
            // re-scheduling. Index-based iteration handles the growth
            // safely (no iterator invalidation).
            priorityOrder.insert(priorityOrder.begin() + i + 1,
                                 static_cast<unsigned>(bestVictim));
            ++backtracks;
            LLVM_DEBUG(DBGS() << "  Ejected N" << bestVictim
                              << " (height=" << bestVictimHeight
                              << ") to place N" << nodeIdx << "\n");
          } else {
            // Could not place even after ejection — restore victim.
            table.reserve(victimCycle, victim.pipeline,
                          static_cast<unsigned>(bestVictim), victimDur);
            scheduled[bestVictim] = victimCycle;
          }
        }
      }
      if (slot < 0) {
        success = false;
        break;
      }

      table.reserve(slot, node.pipeline, nodeIdx, duration);
      scheduled[nodeIdx] = slot;
      LLVM_DEBUG(DBGS() << "  II=" << II << " Placed N" << nodeIdx << " ("
                        << getPipelineName(node.pipeline) << " dur=" << duration
                        << ") at cycle=" << slot << " stage=" << slot / II
                        << "\n");
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

  LLVM_DEBUG(DBGS() << "EXHAUSTED: failed to schedule within maxII=" << maxII
                    << "\n");
  return failure();
}

// ── Swing Modulo Scheduling (SMS) ───────────────────────────────────────────
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

/// Get the duration (pipeline occupancy slots) for a DDG node.
static int getNodeDuration(const DDGNode &node) {
  if (node.pipeline == HWPipeline::NONE)
    return 1;
  return std::max(node.selfLatency, 1);
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
    LLVM_DEBUG(DBGS() << "SMS: ASAP did not converge after " << maxIter
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
    LLVM_DEBUG(DBGS() << "SMS: ALAP did not converge after " << maxIter
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

static FailureOr<ModuloScheduleResult> runSMS(const DataDependenceGraph &ddg,
                                              int minII, int maxII) {
  // Cap maxII to avoid spending too long on large DDGs.
  maxII = std::min(maxII, minII + 10);

  for (int II = minII; II <= maxII; ++II) {
    // Recompute ASAP/ALAP for each II — loop-carried edge constraints
    // depend on II: ASAP[v] >= ASAP[u] + latency - distance * II.
    auto asap = computeASAP(ddg, II);
    auto alap = computeALAP(ddg, asap, II);

    LLVM_DEBUG({
      DBGS() << "SMS: II=" << II << " ASAP/ALAP:\n";
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
    // NOTE: follows all edges including loop-carried, and does not
    // identify or prioritize SCCs/recurrences.
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
        LLVM_DEBUG(DBGS() << "  SMS: II=" << II << " FAILED to place N"
                          << nodeIdx << "\n");
        break;
      }

      table.reserve(slot, node.pipeline, nodeIdx, duration);
      scheduled[nodeIdx] = slot;
      LLVM_DEBUG(DBGS() << "  SMS: II=" << II << " placed N" << nodeIdx << " ("
                        << getPipelineName(node.pipeline) << " dur=" << duration
                        << ") at cycle=" << slot << " stage=" << slot / II
                        << (topDown ? " [top-down]" : " [bottom-up]") << "\n");
    }

    if (success) {
      LLVM_DEBUG(DBGS() << "SMS: SUCCESS at II=" << II << "\n");
      ModuloScheduleResult result;
      result.II = II;
      result.nodeToCycle = std::move(scheduled);
      return result;
    }

    LLVM_DEBUG(DBGS() << "SMS: FAILED at II=" << II << "\n");
  }

  return failure();
}

// ── Public entry point ──────────────────────────────────────────────────────

FailureOr<ModuloScheduleResult>
runModuloScheduling(const DataDependenceGraph &ddg, int maxII,
                    int maxBacktracks) {
  const int minII = ddg.computeMinII();
  if (minII <= 0)
    return failure();
  if (maxII <= 0)
    maxII = 2 * minII;

  // Cap maxII to avoid spending too long on large DDGs.
  maxII = std::min(maxII, minII + 10);

  LLVM_DEBUG({
    DBGS() << "MinII=" << minII << " MaxII=" << maxII
           << " Nodes=" << ddg.getNumNodes() << "\n";
    DBGS() << "ResMII=" << ddg.computeResMII()
           << " RecMII=" << ddg.computeRecMII() << "\n";
  });

  // TRITON_USE_MODULO_SCHEDULE selects the scheduling algorithm:
  //   "sms"        → Swing Modulo Scheduling (Llosa et al., PACT 1996)
  //   "exhaustive" → Exhaustive search with joint memory feasibility
  //   "random"     → Random sampling with greedy placement
  //   "1" or other → Rau's Iterative Modulo Scheduling (Rau, 1994)
  auto algo = mlir::triton::tools::getStrEnv("TRITON_USE_MODULO_SCHEDULE");

  if (algo == "exhaustive") {
    LLVM_DEBUG(DBGS() << "Using exhaustive search with memory feasibility\n");
    return runExhaustiveSearch(ddg, maxII);
  }

  if (algo == "random") {
    LLVM_DEBUG(DBGS() << "Using random sampling search\n");
    return runRandomSearch(ddg, maxII);
  }

  if (algo == "sms") {
    LLVM_DEBUG(DBGS() << "Using Swing Modulo Scheduling (SMS)\n");
    return runSMS(ddg, minII, maxII);
  }

  LLVM_DEBUG(DBGS() << "Using Rau's Iterative Modulo Scheduling (IMS)\n");
  return runRauIMS(ddg, minII, maxII, maxBacktracks);
}

} // namespace mlir::triton::gpu
