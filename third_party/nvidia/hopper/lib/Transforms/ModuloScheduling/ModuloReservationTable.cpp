// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ModuloReservationTable.h"

#include "ExhaustiveScheduler.h"
#include "JointSolverScheduler.h"
#include "SwingScheduler.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <numeric>
#include <string>

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

FailureOr<ModuloScheduleResult> runRauIMS(const DataDependenceGraph &ddg,
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

std::string getActiveScheduleAlgo(llvm::StringRef forced) {
  if (!forced.empty())
    return forced.str();
  auto algo = mlir::triton::tools::getStrEnv("TRITON_USE_MODULO_SCHEDULE");
  return algo.empty() ? "rau" : algo;
}

FailureOr<ModuloScheduleResult>
runModuloScheduling(const DataDependenceGraph &ddg, llvm::StringRef algo,
                    int maxII, int maxBacktracks, int minIIOverride) {
  const int computedMinII = ddg.computeMinII();
  if (computedMinII <= 0)
    return failure();
  const int minII = std::max(computedMinII, minIIOverride);
  const std::string resolvedAlgo = getActiveScheduleAlgo(algo);

  // The complete solver computes its own true feasibility bound. Dispatch it
  // before the heuristic maxII window, which can discard feasible schedules.
  if (resolvedAlgo == "joint_solver") {
    LLVM_DEBUG(DBGS() << "Using native Z3 joint solver\n");
    auto result = runJointSolverSchedule(
        ddg, minII, /*smemBudget=*/232448, /*tmemColLimit=*/512);
    if (failed(result))
      return failure();
    if (!tryRepairModuloSchedule(ddg, *result)) {
      LLVM_DEBUG(DBGS() << "Rejecting invalid final schedule\n");
      return failure();
    }
    return result;
  }

  if (maxII <= 0)
    maxII = 2 * minII;
  else if (maxII < minII)
    return failure();

  // Cap maxII to avoid spending too long on large DDGs. The slack window
  // scales with minII: GPU inner-loop IIs are hundreds of cycles with
  // multi-hundred-cycle op durations, so a fixed +10 window (classic CPU
  // modulo-scheduling folklore) is too narrow to absorb reservation-table
  // fragmentation when one pipeline is saturated (ResMII-bound with zero
  // slack, e.g. layernorm's CUDA pipe). Applies to the heuristic paths
  // below only — the joint_solver path above needs no window (guard 2).
  maxII = std::min(maxII, minII + std::max(10, minII / 8));

  LLVM_DEBUG({
    DBGS() << "MinII=" << minII << " MaxII=" << maxII
           << " Nodes=" << ddg.getNumNodes() << "\n";
    DBGS() << "ResMII=" << ddg.computeResMII()
           << " RecMII=" << ddg.computeRecMII() << "\n";
  });

  // `algo` selects the scheduling algorithm:
  //   "joint_solver" → Native Z3 joint schedule and buffer-depth solver
  //   "sms"        → Swing Modulo Scheduling (Llosa et al., PACT 1996)
  //   "exhaustive" → Exhaustive search with joint memory feasibility
  //   "random"     → Random sampling with greedy placement
  //   "contracted" → Two-stage GEMM search on a contracted compute graph
  //   "1" or other → Rau's Iterative Modulo Scheduling (Rau, 1994)

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

  if (resolvedAlgo == "exhaustive") {
    LLVM_DEBUG(DBGS() << "Using exhaustive search with memory feasibility\n");
    return validateResult(runExhaustiveSearch(ddg, maxII, /*smemBudget=*/232448,
                               /*tmemColLimit=*/512, minII));
  }

  if (resolvedAlgo == "random") {
    LLVM_DEBUG(DBGS() << "Using random sampling search\n");
    return validateResult(runRandomSearch(ddg, maxII, /*smemBudget=*/232448,
                           /*tmemColLimit=*/512, /*numSamples=*/1000, minII));
  }

  if (resolvedAlgo == "contracted") {
    LLVM_DEBUG(DBGS() << "Using contracted-graph two-stage search\n");
    // Contracted mode assigns cycles with a reduced (contracted) latency model
    // and validates its own dependences (see ContractedGraphScheduler.md). The
    // full-latency validator would reject its intended stage-0→stage-1 wrap
    // schedules, so it is deliberately not applied here.
    return runContractedSearch(ddg, maxII);
  }

  if (resolvedAlgo == "sms") {
    LLVM_DEBUG(DBGS() << "Using Swing Modulo Scheduling (SMS)\n");
    return validateResult(runSMS(ddg, minII, maxII));
  }

  LLVM_DEBUG(DBGS() << "Using Rau's Iterative Modulo Scheduling (IMS)\n");
  return validateResult(runRauIMS(ddg, minII, maxII, maxBacktracks));
}

// ── Baseline comparison (M2 acceptance criteria) ────────────────────────────

bool isLegalModuloSchedule(const DataDependenceGraph &ddg,
                           const ModuloScheduleResult &schedule) {
  if (!isValidModuloSchedule(ddg, schedule))
    return false;
  ModuloReservationTable table(schedule.II);
  for (const auto &node : ddg.getNodes()) {
    auto cycleIt = schedule.nodeToCycle.find(node.idx);
    if (cycleIt == schedule.nodeToCycle.end() || cycleIt->second < 0)
      return false;
    if (node.pipeline == HWPipeline::NONE)
      continue;
    int duration = getReservationDuration(node);
    if (duration > schedule.II ||
        !table.isIntervalFree(cycleIt->second, node.pipeline, duration))
      return false;
    table.reserve(cycleIt->second, node.pipeline, node.idx, duration);
  }
  return true;
}

std::optional<int>
BaselineComparisonReport::baselineII(llvm::StringRef name) const {
  for (const BaselineII &entry : baselines)
    if (entry.name == name)
      return entry.ii;
  return std::nullopt;
}

bool BaselineComparisonReport::improvesOnRau() const {
  auto rau = baselineII("rau");
  return jointII && rau && *jointII < *rau;
}

bool BaselineComparisonReport::regressesOnRau() const {
  auto rau = baselineII("rau");
  return jointII && rau && *jointII > *rau;
}

bool BaselineComparisonReport::isMinIIBound() const {
  if (minII <= 0 || !jointII || *jointII != minII)
    return false;
  for (const BaselineII &entry : baselines)
    if (entry.ii && *entry.ii != minII)
      return false;
  return true;
}

bool baselineReportRequested() {
  return !mlir::triton::tools::getStrEnv("TRITON_MODULO_BASELINE_REPORT")
              .empty();
}

namespace {

/// Run one backend and keep its II only if the schedule it produced is legal.
/// Note this uses the in-tree checker rather than Z3JointSolutionValidator:
/// the two enforce the same two properties for the v1 model (dependences +
/// exclusive modular reservation), and the in-tree one keeps every baseline
/// row available in builds without Z3, where only the joint row drops out.
BaselineII runBaseline(llvm::StringRef name, const DataDependenceGraph &ddg,
                       FailureOr<ModuloScheduleResult> scheduled) {
  if (failed(scheduled))
    return BaselineII{name, std::nullopt, "no schedule found"};
  if (!isLegalModuloSchedule(ddg, *scheduled))
    return BaselineII{name, std::nullopt, "schedule failed validation"};
  return BaselineII{name, scheduled->II, ""};
}

/// Right-justify `text` in `width` columns, always leaving at least one space
/// so adjacent cells never run together when a value overflows the column.
void printCell(llvm::raw_ostream &os, llvm::StringRef text, size_t width) {
  os.indent(
      static_cast<unsigned>(width > text.size() ? width - text.size() : 1))
      << text;
}

void printCell(llvm::raw_ostream &os, std::optional<int> value, size_t width) {
  printCell(os, value ? std::to_string(*value) : std::string("-"), width);
}

constexpr size_t kFixtureWidth = 22;
constexpr size_t kNumberWidth = 12;

} // namespace

BaselineComparisonReport compareAgainstBaselines(const DataDependenceGraph &ddg,
                                                 llvm::StringRef fixture) {
  BaselineComparisonReport report;
  report.fixture = fixture.str();

  const int minII = ddg.computeMinII();
  if (minII <= 0 || ddg.getNumNodes() == 0)
    return report;
  report.minII = minII;
  // Exactly the window `runModuloScheduling` gives the heuristic backends
  // (guard 2). The whole point of the Rau row is that it is what the compiler
  // would otherwise have done, so it must be run with the production bound —
  // a roomier window would quietly flatter Rau, a tighter one would rig the
  // comparison in the solver's favour. The joint solver and the relaxed bound
  // compute their own true feasibility bounds and take no window.
  const int heuristicMaxII =
      std::min(2 * minII, minII + std::max(10, minII / 8));

  if (auto joint = runJointSolverSchedule(ddg, minII);
      succeeded(joint) && isLegalModuloSchedule(ddg, *joint))
    report.jointII = joint->II;

  if (auto bound = runJointSolverRelaxedLowerBound(ddg, minII);
      succeeded(bound))
    report.relaxedLowerBound = *bound;

  report.baselines.push_back(runBaseline(
      "rau", ddg, runRauIMS(ddg, minII, heuristicMaxII, /*maxBacktracks=*/20)));
  report.baselines.push_back(
      runBaseline("sms", ddg, runSMS(ddg, minII, heuristicMaxII)));
  // Exhaustive bails out above its node/MMA ceiling, so an absent row here is
  // expected on anything but a small loop. Reported for free context: where it
  // does terminate it brackets the optimum from the other side.
  report.baselines.push_back(runBaseline(
      "exhaustive", ddg,
      runExhaustiveSearch(ddg, heuristicMaxII, /*smemBudget=*/232448,
                          /*tmemColLimit=*/512, minII)));
  return report;
}

void printBaselineComparison(llvm::raw_ostream &os,
                             const BaselineComparisonReport &report) {
  auto printFixtureCell = [&](llvm::StringRef text) {
    os << "  " << text.substr(0, kFixtureWidth);
    os.indent(static_cast<unsigned>(
        text.size() < kFixtureWidth ? kFixtureWidth - text.size() : 1));
  };

  os << "modulo-baseline-report: " << report.fixture << "\n";
  printFixtureCell("fixture");
  printCell(os, llvm::StringRef("MinII"), kNumberWidth);
  printCell(os, llvm::StringRef("II_full"), kNumberWidth);
  printCell(os, llvm::StringRef("relaxed_LB"), kNumberWidth);
  for (const BaselineII &entry : report.baselines)
    printCell(os, "II_" + entry.name.str(), kNumberWidth);
  os << "\n";

  printFixtureCell(report.fixture);
  printCell(os, report.minII, kNumberWidth);
  printCell(os, report.jointII, kNumberWidth);
  printCell(os, report.relaxedLowerBound, kNumberWidth);
  for (const BaselineII &entry : report.baselines)
    printCell(os, entry.ii, kNumberWidth);
  os << "\n";

  for (const BaselineII &entry : report.baselines)
    if (!entry.ii && !entry.note.empty())
      os << "  note: " << entry.name << ": " << entry.note << "\n";

  // Criterion 1 — soundness. A relaxation's feasible set is a superset of the
  // full model's, so its optimum can never exceed the full model's. Violation
  // means a constraint is mis-encoded, NOT that the solver got slower.
  os << "  soundness (II_full >= relaxed_LB): ";
  if (!report.soundnessCheckable())
    os << "SKIP (joint solver or relaxed bound unavailable)\n";
  else if (report.soundnessHolds())
    os << "PASS (gap " << *report.jointII - *report.relaxedLowerBound << ")\n";
  else
    os << "FAIL\n";

  // Criterion 2 has two halves, kept separate so a tie can never be read as
  // the acceptance criterion being met. Rau is Triton's current default
  // scheduler and the path the joint solver's own fallback takes.
  auto rau = report.baselineII("rau");

  // 2a — no regression against Rau. Required on EVERY fixture.
  os << "  no-regression (II_full <= II_rau): ";
  if (!report.jointII || !rau)
    os << "SKIP (joint solver or rau unavailable)\n";
  else if (report.regressesOnRau())
    os << "FAIL (" << *report.jointII << " > " << *rau << ")\n";
  else
    os << "PASS\n";

  // 2b — the acceptance criterion proper: strict improvement on at least one
  // fixture. A single loop cannot decide "at least one", so this line reports
  // only what this fixture contributes. Reported, never tuned for: a NO is a
  // real finding to take to the team, not a reason to adjust the fixture.
  os << "  strict improvement (II_full < II_rau): ";
  if (!report.jointII || !rau)
    os << "SKIP (joint solver or rau unavailable)\n";
  else if (report.improvesOnRau())
    os << "YES (-" << *rau - *report.jointII << " vs rau)\n";
  else if (report.isMinIIBound())
    os << "NO (every backend is at MinII " << report.minII
       << " — this fixture is MinII-bound and cannot separate schedulers; it "
          "is not evidence about the solver)\n";
  else
    os << "NO (tied with rau at " << *rau << ")\n";

  // The relaxed bound is NOT a pure resource-only model. Stated at the point
  // of use so the number is never mistaken for a textbook one.
  os << "  relaxed_LB keeps resource exclusivity only; still-hard: cycle "
        "domain bounds, canonical-root pinning, SMEM/TMEM ceilings (vacuous "
        "once their gated contributors are dropped)\n";
}

} // namespace mlir::triton::gpu
