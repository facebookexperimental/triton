// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ModuloReservationTable.h"

#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <climits>
#include <numeric>
#include <optional>

#define DEBUG_TYPE "modulo-scheduling-rau"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::triton::gpu {

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

// ── Constructive joint cycle+warp scheduling ────────────────────────────────

namespace {

constexpr int kDefaultWarpGroupWarps = 4;
constexpr int kMaxHardwareWarps = 64;
constexpr int kRegisterSyncBaseCycles = 150;
constexpr int kRegisterSyncCyclesPerKB = 16;
constexpr int kOtherCrossWGSyncCycles = 60;

/// Per-warp-group circular issue stream. Every instruction contends with every
/// other instruction in the same warp group, regardless of hardware pipeline.
class WarpIssueReservationStream {
public:
  explicit WarpIssueReservationStream(int II) : II{II}, slots(II, -1) {}

  bool isIntervalFree(int cycle, int duration) const {
    for (int t = cycle; t < cycle + duration; ++t) {
      if (slots[t % II] >= 0)
        return false;
    }
    return true;
  }

  void reserve(int cycle, unsigned nodeIdx, int duration) {
    for (int t = cycle; t < cycle + duration; ++t)
      slots[t % II] = static_cast<int>(nodeIdx);
  }

  void unreserve(int cycle, int duration) {
    for (int t = cycle; t < cycle + duration; ++t)
      slots[t % II] = -1;
  }

  int getOccupant(int cycle) const { return slots[cycle % II]; }

private:
  int II{};
  llvm::SmallVector<int> slots;
};

struct JointCandidate {
  int cycle{-1};
  int warpGroup{-1};
};

static int getPipelineDuration(const DDGNode &node) {
  return pipelineOccupancy(node);
}

static int getWarpIssueDuration(const DDGNode &node) {
  return std::max(node.selfLatency, 1);
}

static int roundRequiredWarps(int minWarps) {
  if (minWarps <= 1)
    return 1;
  if (minWarps <= 2)
    return 2;
  if (minWarps <= 4)
    return 4;
  return 8;
}

static int getRequiredWarpsForNode(const DDGNode &node) {
  return roundRequiredWarps(std::max(node.minWarps, 1));
}

static int getUsedHardwareWarps(llvm::ArrayRef<int> wgRequiredWarps) {
  return kDefaultWarpGroupWarps +
         std::accumulate(wgRequiredWarps.begin(), wgRequiredWarps.end(), 0);
}

static bool canReplaceWarpGroupRequirement(llvm::ArrayRef<int> wgRequiredWarps,
                                           int warpGroup,
                                           int newRequiredWarps) {
  int usedWarps = kDefaultWarpGroupWarps;
  for (int wg = 0, e = wgRequiredWarps.size(); wg < e; ++wg)
    usedWarps += (wg == warpGroup) ? newRequiredWarps : wgRequiredWarps[wg];
  return usedWarps <= kMaxHardwareWarps;
}

static int recomputeWarpGroupRequirement(
    int warpGroup, const DataDependenceGraph &ddg,
    const llvm::DenseMap<unsigned, int> &nodeToWarpGroup) {
  int requiredWarps = 0;
  for (auto [scheduledNodeIdx, scheduledWarpGroup] : nodeToWarpGroup) {
    if (scheduledWarpGroup != warpGroup)
      continue;
    requiredWarps =
        std::max(requiredWarps, getRequiredWarpsForNode(ddg.getNode(
                                    static_cast<unsigned>(scheduledNodeIdx))));
  }
  return requiredWarps;
}

static std::optional<int64_t> getRegisterTensorBytes(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return std::nullopt;

  Type elementType = tensorType.getElementType();
  if (isa<triton::PointerType>(elementType))
    return std::nullopt;
  if (!elementType.isIntOrFloat())
    return std::nullopt;

  int64_t numElements = tensorType.getNumElements();
  if (numElements < 0)
    return std::nullopt;
  return llvm::divideCeil(numElements * tensorType.getElementTypeBitWidth(),
                          int64_t{8});
}

static int getCrossWarpGroupLatency(const DDGNode &producer) {
  if (!producer.op)
    return kOtherCrossWGSyncCycles;

  if (producer.pipeline != HWPipeline::CUDA &&
      producer.pipeline != HWPipeline::SFU &&
      producer.pipeline != HWPipeline::VALU)
    return kOtherCrossWGSyncCycles;

  int64_t resultBytes = 0;
  bool sawRegisterTensor = false;
  for (Value result : producer.op->getResults()) {
    std::optional<int64_t> bytes = getRegisterTensorBytes(result);
    if (!bytes)
      continue;
    sawRegisterTensor = true;
    resultBytes += *bytes;
  }

  if (!sawRegisterTensor)
    return kOtherCrossWGSyncCycles;

  int64_t kilobytes = llvm::divideCeil(resultBytes, int64_t{1024});
  return kRegisterSyncBaseCycles +
         static_cast<int>(kilobytes * kRegisterSyncCyclesPerKB);
}

static int computeEarliestStartForWarpGroup(
    unsigned nodeIdx, int warpGroup, const DataDependenceGraph &ddg,
    const llvm::DenseMap<unsigned, int> &scheduled,
    const llvm::DenseMap<unsigned, int> &nodeToWarpGroup, int II) {
  int earliest = 0;
  for (const auto *edge : ddg.getInEdges(nodeIdx)) {
    auto cycleIt = scheduled.find(edge->srcIdx);
    if (cycleIt == scheduled.end())
      continue;

    int latency = edge->latency;
    auto wgIt = nodeToWarpGroup.find(edge->srcIdx);
    if (wgIt != nodeToWarpGroup.end() && wgIt->second != warpGroup)
      latency += getCrossWarpGroupLatency(ddg.getNode(edge->srcIdx));

    int constraint =
        cycleIt->second + latency - static_cast<int>(edge->distance) * II;
    earliest = std::max(earliest, constraint);
  }
  return earliest;
}

static bool
hasScheduledConsumer(unsigned nodeIdx, const DataDependenceGraph &ddg,
                     const llvm::DenseMap<unsigned, int> &scheduled) {
  for (const auto *edge : ddg.getOutEdges(nodeIdx)) {
    if (scheduled.contains(edge->dstIdx))
      return true;
  }
  return false;
}

static std::optional<JointCandidate> getExactCandidate(
    unsigned nodeIdx, int warpGroup, const DataDependenceGraph &ddg,
    const ModuloReservationTable &globalTable,
    ArrayRef<WarpIssueReservationStream> issueStreams,
    ArrayRef<int> wgRequiredWarps,
    const llvm::DenseMap<unsigned, int> &scheduled,
    const llvm::DenseMap<unsigned, int> &nodeToWarpGroup, int II) {
  const DDGNode &node = ddg.getNode(nodeIdx);
  int newRequiredWarps =
      std::max(wgRequiredWarps[warpGroup], getRequiredWarpsForNode(node));
  if (!canReplaceWarpGroupRequirement(wgRequiredWarps, warpGroup,
                                      newRequiredWarps))
    return std::nullopt;
  int earliest = computeEarliestStartForWarpGroup(
      nodeIdx, warpGroup, ddg, scheduled, nodeToWarpGroup, II);
  int cycle = globalTable.findFreeSlot(earliest, node.pipeline,
                                       getPipelineDuration(node));
  if (cycle < 0)
    return std::nullopt;
  if (!issueStreams[warpGroup].isIntervalFree(cycle,
                                              getWarpIssueDuration(node)))
    return std::nullopt;
  return JointCandidate{cycle, warpGroup};
}

static JointCandidate
findCandidate(unsigned nodeIdx, const DataDependenceGraph &ddg,
              const ModuloReservationTable &globalTable,
              ArrayRef<WarpIssueReservationStream> issueStreams,
              ArrayRef<int> wgRequiredWarps,
              const llvm::DenseMap<unsigned, int> &scheduled,
              const llvm::DenseMap<unsigned, int> &nodeToWarpGroup, int II) {
  JointCandidate best;
  for (int wg = 0, e = issueStreams.size(); wg < e; ++wg) {
    std::optional<JointCandidate> candidate =
        getExactCandidate(nodeIdx, wg, ddg, globalTable, issueStreams,
                          wgRequiredWarps, scheduled, nodeToWarpGroup, II);
    if (!candidate)
      continue;
    if (best.cycle < 0 || candidate->cycle < best.cycle ||
        (candidate->cycle == best.cycle && wg < best.warpGroup)) {
      best = *candidate;
    }
  }
  return best;
}

static int
findEjectionVictim(unsigned nodeIdx, const DataDependenceGraph &ddg,
                   const ModuloReservationTable &globalTable,
                   ArrayRef<WarpIssueReservationStream> issueStreams,
                   ArrayRef<int> wgRequiredWarps,
                   const llvm::DenseMap<unsigned, int> &scheduled,
                   const llvm::DenseMap<unsigned, int> &nodeToWarpGroup,
                   const llvm::DenseMap<unsigned, int> &heights, int II) {
  int bestVictim = -1;
  int bestVictimHeight = INT_MAX;
  int currentHeight = heights.lookup(nodeIdx);
  const auto &node = ddg.getNode(nodeIdx);

  for (int wg = 0, e = issueStreams.size(); wg < e; ++wg) {
    int newRequiredWarps =
        std::max(wgRequiredWarps[wg], getRequiredWarpsForNode(node));
    if (!canReplaceWarpGroupRequirement(wgRequiredWarps, wg, newRequiredWarps))
      continue;

    int earliest = computeEarliestStartForWarpGroup(nodeIdx, wg, ddg, scheduled,
                                                    nodeToWarpGroup, II);
    int cycle = globalTable.findFreeSlot(earliest, node.pipeline,
                                         getPipelineDuration(node));
    if (cycle < 0)
      continue;

    int issueDuration = getWarpIssueDuration(node);
    llvm::SmallVector<int, 8> occupants;
    for (int t = cycle; t < cycle + issueDuration; ++t) {
      int occupant = issueStreams[wg].getOccupant(t);
      if (occupant >= 0)
        occupants.push_back(occupant);
    }

    for (int occupant : occupants) {
      unsigned occupantIdx = static_cast<unsigned>(occupant);
      // Conservatively eject only leaves: moving a node with scheduled
      // consumers would invalidate their cycle and warp-group constraints.
      if (hasScheduledConsumer(occupantIdx, ddg, scheduled))
        continue;

      int occHeight = heights.lookup(occupantIdx);
      if (occHeight < currentHeight && occHeight < bestVictimHeight) {
        bestVictimHeight = occHeight;
        bestVictim = occupant;
      }
    }
  }
  return bestVictim;
}

static void
reserveNode(ModuloReservationTable &globalTable,
            MutableArrayRef<WarpIssueReservationStream> issueStreams,
            const DDGNode &node, unsigned nodeIdx, int cycle, int warpGroup) {
  globalTable.reserve(cycle, node.pipeline, nodeIdx, getPipelineDuration(node));
  issueStreams[warpGroup].reserve(cycle, nodeIdx, getWarpIssueDuration(node));
}

static void
unreserveNode(ModuloReservationTable &globalTable,
              MutableArrayRef<WarpIssueReservationStream> issueStreams,
              const DDGNode &node, int cycle, int warpGroup) {
  globalTable.unreserve(cycle, node.pipeline, getPipelineDuration(node));
  issueStreams[warpGroup].unreserve(cycle, getWarpIssueDuration(node));
}

} // namespace

static FailureOr<ModuloScheduleResult> runRauIMS(const DataDependenceGraph &ddg,
                                                 int minII, int maxII,
                                                 int maxBacktracks) {
  LLVM_DEBUG(DBGS() << "Computing critical path heights...\n");
  auto heights = ddg.computeCriticalPathHeights();
  LLVM_DEBUG(DBGS() << "Heights computed for " << heights.size() << " nodes\n");

  llvm::SmallVector<unsigned> priorityOrder;
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i)
    priorityOrder.push_back(i);
  llvm::sort(priorityOrder, [&](unsigned a, unsigned b) {
    if (heights[a] != heights[b])
      return heights[a] > heights[b];
    return a < b;
  });

  LLVM_DEBUG({
    DBGS() << "MinII=" << minII << " MaxII=" << maxII
           << " Nodes=" << priorityOrder.size() << "\n";
    DBGS() << "ResMII=" << ddg.computeResMII()
           << " RecMII=" << ddg.computeRecMII() << "\n";
  });
  LLVM_DEBUG({
    llvm::DenseMap<HWPipeline, int> pipeLoad;
    int issueLoad = 0;
    for (const auto &node : ddg.getNodes()) {
      if (node.pipeline != HWPipeline::NONE)
        pipeLoad[node.pipeline] += getPipelineDuration(node);
      issueLoad += getWarpIssueDuration(node);
    }
    for (auto &[pipe, load] : pipeLoad)
      DBGS() << "  " << getPipelineName(pipe) << " total_load=" << load << "\n";
    DBGS() << "  warp_issue total_load=" << issueLoad << "\n";
  });

  for (int II = minII; II <= maxII; ++II) {
    ModuloReservationTable globalTable{II};
    llvm::SmallVector<WarpIssueReservationStream> issueStreams;
    llvm::SmallVector<int> wgRequiredWarps;
    llvm::DenseMap<unsigned, int> scheduled;
    llvm::DenseMap<unsigned, int> nodeToWarpGroup;
    bool success = true;
    int backtracks = 0;

    auto addWarpGroup = [&]() {
      issueStreams.emplace_back(II);
      wgRequiredWarps.push_back(0);
      return static_cast<int>(issueStreams.size()) - 1;
    };

    for (unsigned i = 0; i < priorityOrder.size(); ++i) {
      unsigned nodeIdx = priorityOrder[i];
      const auto &node = ddg.getNode(nodeIdx);
      JointCandidate candidate =
          findCandidate(nodeIdx, ddg, globalTable, issueStreams,
                        wgRequiredWarps, scheduled, nodeToWarpGroup, II);

      if (candidate.cycle < 0) {
        int newWGWarps = getRequiredWarpsForNode(node);
        if (getUsedHardwareWarps(wgRequiredWarps) + newWGWarps <=
            kMaxHardwareWarps) {
          int wg = addWarpGroup();
          wgRequiredWarps[wg] = newWGWarps;
          std::optional<JointCandidate> newCandidate = getExactCandidate(
              nodeIdx, wg, ddg, globalTable, issueStreams, wgRequiredWarps,
              scheduled, nodeToWarpGroup, II);
          if (newCandidate)
            candidate = *newCandidate;
          if (candidate.cycle < 0) {
            issueStreams.pop_back();
            wgRequiredWarps.pop_back();
          }
        }
      }

      if (candidate.cycle < 0 && backtracks < maxBacktracks &&
          !issueStreams.empty()) {
        int victim = findEjectionVictim(nodeIdx, ddg, globalTable, issueStreams,
                                        wgRequiredWarps, scheduled,
                                        nodeToWarpGroup, heights, II);
        if (victim >= 0) {
          const auto &victimNode = ddg.getNode(static_cast<unsigned>(victim));
          int victimCycle = scheduled.lookup(static_cast<unsigned>(victim));
          int victimWG = nodeToWarpGroup.lookup(static_cast<unsigned>(victim));
          unreserveNode(globalTable, issueStreams, victimNode, victimCycle,
                        victimWG);
          scheduled.erase(static_cast<unsigned>(victim));
          nodeToWarpGroup.erase(static_cast<unsigned>(victim));
          wgRequiredWarps[victimWG] =
              recomputeWarpGroupRequirement(victimWG, ddg, nodeToWarpGroup);

          candidate =
              findCandidate(nodeIdx, ddg, globalTable, issueStreams,
                            wgRequiredWarps, scheduled, nodeToWarpGroup, II);
          if (candidate.cycle >= 0) {
            priorityOrder.insert(priorityOrder.begin() + i + 1,
                                 static_cast<unsigned>(victim));
            ++backtracks;
            LLVM_DEBUG(DBGS() << "  Ejected N" << victim << " (height="
                              << heights.lookup(static_cast<unsigned>(victim))
                              << ") to place N" << nodeIdx << "\n");
          } else {
            reserveNode(globalTable, issueStreams, victimNode,
                        static_cast<unsigned>(victim), victimCycle, victimWG);
            scheduled[static_cast<unsigned>(victim)] = victimCycle;
            nodeToWarpGroup[static_cast<unsigned>(victim)] = victimWG;
            wgRequiredWarps[victimWG] =
                recomputeWarpGroupRequirement(victimWG, ddg, nodeToWarpGroup);
          }
        }
      }

      if (candidate.cycle < 0) {
        success = false;
        break;
      }

      wgRequiredWarps[candidate.warpGroup] = std::max(
          wgRequiredWarps[candidate.warpGroup], getRequiredWarpsForNode(node));
      reserveNode(globalTable, issueStreams, node, nodeIdx, candidate.cycle,
                  candidate.warpGroup);
      scheduled[nodeIdx] = candidate.cycle;
      nodeToWarpGroup[nodeIdx] = candidate.warpGroup;
      LLVM_DEBUG(DBGS() << "  II=" << II << " Placed N" << nodeIdx << " ("
                        << getPipelineName(node.pipeline)
                        << " pipe_dur=" << getPipelineDuration(node)
                        << " issue_dur=" << getWarpIssueDuration(node)
                        << ") at cycle=" << candidate.cycle
                        << " stage=" << candidate.cycle / II
                        << " wg=" << candidate.warpGroup << "\n");
    }

    if (success) {
      LLVM_DEBUG(DBGS() << "SUCCESS at II=" << II
                        << " wgs=" << issueStreams.size() << " warps="
                        << getUsedHardwareWarps(wgRequiredWarps) << "\n");

      ModuloScheduleResult result;
      result.II = II;
      result.nodeToCycle = std::move(scheduled);
      result.nodeToWarpGroup = std::move(nodeToWarpGroup);
      result.numWarpGroups = static_cast<int>(issueStreams.size());
      return result;
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

  LLVM_DEBUG(DBGS() << "Using joint Rau cycle+warp scheduling\n");
  return runRauIMS(ddg, minII, maxII, maxBacktracks);
}

} // namespace mlir::triton::gpu
