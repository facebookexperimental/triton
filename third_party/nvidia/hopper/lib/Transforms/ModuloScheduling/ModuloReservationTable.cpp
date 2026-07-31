// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ModuloReservationTable.h"

#include "ExhaustiveScheduler.h"
#include "SwingScheduler.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <numeric>

#define DEBUG_TYPE "modulo-scheduling-rau"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::triton::gpu {

static bool
hasValidScheduleDomain(int II, const llvm::DenseMap<unsigned, int> &nodeToCycle,
                       unsigned numNodes, llvm::ArrayRef<DDGEdge> edges) {
  if (II <= 0 || nodeToCycle.size() != numNodes)
    return false;
  for (unsigned nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
    auto it = nodeToCycle.find(nodeIdx);
    if (it == nodeToCycle.end() || it->second < 0)
      return false;
  }
  for (const auto &edge : edges)
    if (edge.srcIdx >= numNodes || edge.dstIdx >= numNodes || edge.latency < 0)
      return false;
  return true;
}

bool isValidModuloSchedule(int II,
                           const llvm::DenseMap<unsigned, int> &nodeToCycle,
                           unsigned numNodes, llvm::ArrayRef<DDGEdge> edges) {
  if (!hasValidScheduleDomain(II, nodeToCycle, numNodes, edges))
    return false;

  for (const auto &edge : edges) {
    int64_t consumerStart =
        static_cast<int64_t>(nodeToCycle.lookup(edge.dstIdx)) +
        static_cast<int64_t>(edge.distance) * static_cast<int64_t>(II);
    int64_t producerReady =
        static_cast<int64_t>(nodeToCycle.lookup(edge.srcIdx)) +
        static_cast<int64_t>(edge.latency);
    if (consumerStart < producerReady)
      return false;
  }
  return true;
}

bool isValidModuloSchedule(const DataDependenceGraph &ddg,
                           const ModuloScheduleResult &schedule) {
  return isValidModuloSchedule(schedule.II, schedule.nodeToCycle,
                               ddg.getNumNodes(), ddg.getEdges());
}

static int getReservationDuration(const DDGNode &node) {
  if (node.pipeline == HWPipeline::NONE)
    return 1;
  return std::max(node.selfLatency, 1);
}

bool tryRepairModuloSchedule(int II, llvm::DenseMap<unsigned, int> &nodeToCycle,
                             llvm::ArrayRef<DDGNode> nodes,
                             llvm::ArrayRef<DDGEdge> edges) {
  if (!hasValidScheduleDomain(II, nodeToCycle, nodes.size(), edges))
    return false;
  if (isValidModuloSchedule(II, nodeToCycle, nodes.size(), edges))
    return true;

  ModuloReservationTable table(II);
  for (unsigned nodeIdx = 0; nodeIdx < nodes.size(); ++nodeIdx) {
    int cycle = nodeToCycle.lookup(nodeIdx);
    int duration = getReservationDuration(nodes[nodeIdx]);
    if (!table.isIntervalFree(cycle, nodes[nodeIdx].pipeline, duration))
      return false;
    table.reserve(cycle, nodes[nodeIdx].pipeline, nodeIdx, duration);
  }

  for (unsigned iteration = 0; iteration < nodes.size(); ++iteration) {
    bool changed = false;
    for (unsigned dstIdx = 0; dstIdx < nodes.size(); ++dstIdx) {
      int64_t earliest = nodeToCycle.lookup(dstIdx);
      for (const auto &edge : edges) {
        if (edge.dstIdx != dstIdx)
          continue;
        int64_t required =
            static_cast<int64_t>(nodeToCycle.lookup(edge.srcIdx)) +
            static_cast<int64_t>(edge.latency) -
            static_cast<int64_t>(edge.distance) * static_cast<int64_t>(II);
        earliest = std::max(earliest, required);
      }
      if (earliest == nodeToCycle.lookup(dstIdx))
        continue;
      if (earliest > INT_MAX)
        return false;

      int64_t latest = INT_MAX;
      for (const auto &edge : edges) {
        if (edge.srcIdx != dstIdx)
          continue;
        if (edge.dstIdx == dstIdx) {
          if (static_cast<int64_t>(edge.distance) * II < edge.latency)
            return false;
          continue;
        }
        int64_t allowed =
            static_cast<int64_t>(nodeToCycle.lookup(edge.dstIdx)) +
            static_cast<int64_t>(edge.distance) * static_cast<int64_t>(II) -
            static_cast<int64_t>(edge.latency);
        latest = std::min(latest, allowed);
      }
      if (earliest > latest)
        return false;

      const auto &node = nodes[dstIdx];
      int oldCycle = nodeToCycle.lookup(dstIdx);
      int duration = getReservationDuration(node);
      table.unreserve(oldCycle, node.pipeline, duration);

      int64_t searchEnd =
          std::min<int64_t>(latest, earliest + static_cast<int64_t>(II) - 1);
      int newCycle = -1;
      for (int64_t candidate = earliest; candidate <= searchEnd; ++candidate) {
        if (table.isIntervalFree(static_cast<int>(candidate), node.pipeline,
                                 duration)) {
          newCycle = static_cast<int>(candidate);
          break;
        }
      }
      if (newCycle < 0) {
        table.reserve(oldCycle, node.pipeline, dstIdx, duration);
        return false;
      }

      nodeToCycle[dstIdx] = newCycle;
      table.reserve(newCycle, node.pipeline, dstIdx, duration);
      changed = true;
    }

    if (isValidModuloSchedule(II, nodeToCycle, nodes.size(), edges))
      return true;
    if (!changed)
      return false;
  }
  return false;
}

bool tryRepairModuloSchedule(const DataDependenceGraph &ddg,
                             ModuloScheduleResult &schedule) {
  return tryRepairModuloSchedule(schedule.II, schedule.nodeToCycle,
                                 ddg.getNodes(), ddg.getEdges());
}

// ── ModuloReservationTable ──────────────────────────────────────────────────

ModuloReservationTable::ModuloReservationTable(int II) : II{II} {
  // One row per hardware pipeline (NV + AMD). reserve()/unreserve() index
  // table[pipeline][slot] directly, so every non-NONE pipeline must have a row
  // or it would index a default-constructed empty SmallVector. NONE is handled
  // specially (no resource) by all methods.
  // MFMA is AMD's matrix engine and TC is NV's tensor core — functionally
  // analogous, but kept as distinct pipelines so each backend's LatencyModel
  // reserves its own row (a kernel only ever uses one of them).
  for (auto pipe : {HWPipeline::TMA, HWPipeline::TC, HWPipeline::CUDA,
                    HWPipeline::SFU, HWPipeline::MFMA, HWPipeline::LDS,
                    HWPipeline::GLOBAL, HWPipeline::VALU}) {
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
  llvm::SmallVector<unsigned> basePriorityOrder;
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i)
    basePriorityOrder.push_back(i);
  llvm::sort(basePriorityOrder, [&](unsigned a, unsigned b) {
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
           << " Nodes=" << basePriorityOrder.size() << "\n";
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
    auto priorityOrder = basePriorityOrder;
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
      ModuloScheduleResult result;
      result.II = II;
      result.nodeToCycle = std::move(scheduled);
      if (tryRepairModuloSchedule(ddg, result)) {
        LLVM_DEBUG(DBGS() << "SUCCESS at II=" << II << "\n");
        return result;
      }
      LLVM_DEBUG(DBGS() << "II=" << II
                        << ": rejected dependency-invalid schedule\n");
    }

    LLVM_DEBUG(DBGS() << "FAILED at II=" << II << "\n");
  }

  LLVM_DEBUG(DBGS() << "EXHAUSTED: failed to schedule within maxII=" << maxII
                    << "\n");
  return failure();
}

// runListScheduling moved to ListSchedulePass.cpp so its DEBUG_TYPE matches
// the rest of the list-scheduling pass output
// (-debug-only=nvgpu-list-schedule).

// ── Public entry point ──────────────────────────────────────────────────────

FailureOr<ModuloScheduleResult>
runModuloScheduling(const DataDependenceGraph &ddg, int maxII,
                    int maxBacktracks) {
  const int minII = ddg.computeMinII();
  if (minII <= 0)
    return failure();
  if (maxII <= 0)
    maxII = 2 * minII;

  // Cap maxII to avoid spending too long on large DDGs. The slack window
  // scales with minII: GPU inner-loop IIs are hundreds of cycles with
  // multi-hundred-cycle op durations, so a fixed +10 window (classic CPU
  // modulo-scheduling folklore) is too narrow to absorb reservation-table
  // fragmentation when one pipeline is saturated (ResMII-bound with zero
  // slack, e.g. layernorm's CUDA pipe). A complete (ILP-style) search has
  // no such fragmentation failure mode and needs no window at all.
  maxII = std::min(maxII, minII + std::max(10, minII / 8));

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
  //   "contracted" → Two-stage GEMM search on a contracted compute graph
  //   "1" or other → Rau's Iterative Modulo Scheduling (Rau, 1994)
  auto algo = mlir::triton::tools::getStrEnv("TRITON_USE_MODULO_SCHEDULE");

  auto validateResult = [&](FailureOr<ModuloScheduleResult> result)
      -> FailureOr<ModuloScheduleResult> {
    if (failed(result))
      return failure();
    if (!tryRepairModuloSchedule(ddg, *result)) {
      LLVM_DEBUG(DBGS() << "Rejecting invalid final schedule\n");
      return failure();
    }
    return std::move(result);
  };

  if (algo == "exhaustive") {
    LLVM_DEBUG(DBGS() << "Using exhaustive search with memory feasibility\n");
    return validateResult(runExhaustiveSearch(ddg, maxII));
  }

  if (algo == "random") {
    LLVM_DEBUG(DBGS() << "Using random sampling search\n");
    return validateResult(runRandomSearch(ddg, maxII));
  }

  if (algo == "contracted") {
    LLVM_DEBUG(DBGS() << "Using contracted-graph two-stage search\n");
    // Contracted mode assigns cycles with a reduced (contracted) latency model
    // and validates its own dependences (see ContractedGraphScheduler.md). The
    // full-latency validator would reject its intended stage-0→stage-1 wrap
    // schedules, so it is deliberately not applied here.
    return runContractedSearch(ddg, maxII);
  }

  if (algo == "sms") {
    LLVM_DEBUG(DBGS() << "Using Swing Modulo Scheduling (SMS)\n");
    return validateResult(runSMS(ddg, minII, maxII));
  }

  LLVM_DEBUG(DBGS() << "Using Rau's Iterative Modulo Scheduling (IMS)\n");
  return validateResult(runRauIMS(ddg, minII, maxII, maxBacktracks));
}

} // namespace mlir::triton::gpu
