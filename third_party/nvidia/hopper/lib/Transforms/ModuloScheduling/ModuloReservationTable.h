#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_RESERVATION_TABLE_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_RESERVATION_TABLE_H

#include "DataDependenceGraph.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace mlir::triton::gpu {

/// Modulo reservation table: II time slots × one row per HWPipeline.
/// A slot [cycle % II][pipeline] holds at most one op.
class ModuloReservationTable {
public:
  explicit ModuloReservationTable(int II);

  int getII() const { return II; }

  bool isFree(int cycle, HWPipeline pipeline) const;
  bool isIntervalFree(int cycle, HWPipeline pipeline, int duration) const;
  void reserve(int cycle, HWPipeline pipeline, unsigned nodeIdx,
               int duration = 1);
  void unreserve(int cycle, HWPipeline pipeline, int duration = 1);

  /// Find earliest free slot at or after `earliest` on pipeline, within II.
  /// Checks that `duration` consecutive slots are all free.
  /// Returns -1 if no slot found.
  int findFreeSlot(int earliest, HWPipeline pipeline, int duration = 1) const;

  /// Get the node index occupying a slot, or -1 if free.
  int getOccupant(int cycle, HWPipeline pipeline) const;

private:
  int II{};
  // table[pipeline][slot] = nodeIdx or -1
  llvm::DenseMap<HWPipeline, llvm::SmallVector<int>> table;
};

/// Result of modulo scheduling for one loop.
struct ModuloScheduleResult {
  int II{};
  llvm::DenseMap<unsigned, int> nodeToCycle; // DDG node idx -> absolute cycle

  int getStage(unsigned nodeIdx) const {
    auto it = nodeToCycle.find(nodeIdx);
    return it != nodeToCycle.end() ? it->second / II : 0;
  }

  int getMaxStage() const {
    int maxStage = 0;
    for (auto &[idx, cycle] : nodeToCycle)
      maxStage = std::max(maxStage, cycle / II);
    return maxStage;
  }
};

/// Check that every DDG node has a nonnegative cycle and every dependence
/// satisfies dst_cycle + distance * II >= src_cycle + latency.
bool isValidModuloSchedule(int II,
                           const llvm::DenseMap<unsigned, int> &nodeToCycle,
                           unsigned numNodes, llvm::ArrayRef<DDGEdge> edges);
bool isValidModuloSchedule(const DataDependenceGraph &ddg,
                           const ModuloScheduleResult &schedule);

/// Move dependency-violating destinations forward when the move fits both the
/// modulo reservation table and every outgoing dependence. Returns false if
/// no local repair exists.
bool tryRepairModuloSchedule(int II, llvm::DenseMap<unsigned, int> &nodeToCycle,
                             llvm::ArrayRef<DDGNode> nodes,
                             llvm::ArrayRef<DDGEdge> edges);
bool tryRepairModuloSchedule(const DataDependenceGraph &ddg,
                             ModuloScheduleResult &schedule);

/// Run modulo scheduling on the DDG.
/// Algorithm selected by TRITON_USE_MODULO_SCHEDULE env var value:
///   "sms"        → Swing Modulo Scheduling (Llosa et al., PACT 1996)
///   "exhaustive" → Exhaustive search with joint memory feasibility
///   "random"     → Random sampling with greedy placement
///   "1" or other → Rau's Iterative Modulo Scheduling (Rau, 1994)
/// maxII defaults to 2 * MinII. maxBacktracks limits ejection in Rau's IMS.
FailureOr<ModuloScheduleResult>
runModuloScheduling(const DataDependenceGraph &ddg, int maxII = 0,
                    int maxBacktracks = 20, int minIIOverride = 0);

/// Rau's Iterative Modulo Scheduling (Rau, 1994) — Triton's default backend,
/// and the fallback the joint solver drops to. Exposed so the baseline
/// comparison can invoke it by name instead of going through the
/// TRITON_USE_MODULO_SCHEDULE dispatch in `runModuloScheduling`.
FailureOr<ModuloScheduleResult> runRauIMS(const DataDependenceGraph &ddg,
                                          int minII, int maxII,
                                          int maxBacktracks);

/// Dependence legality (`isValidModuloSchedule`) AND exclusive modular
/// reservation. Both halves are required before a scheduler's II may be
/// compared against another's: a scheduler that emits an illegal schedule can
/// report any II it likes, so an unvalidated II is meaningless.
bool isLegalModuloSchedule(const DataDependenceGraph &ddg,
                           const ModuloScheduleResult &schedule);

/// One baseline scheduler's row in the comparison report.
struct BaselineII {
  llvm::StringRef name;
  std::optional<int> ii; // nullopt: unavailable, failed, or invalid
  llvm::StringRef note;  // why it is absent, when it is
};

/// II-level comparison of the native joint solver against the in-tree
/// schedulers on one DDG, plus the relaxed lower bound. Backs the M2
/// acceptance criteria; see docs/Diff8BaselineComparison.md for what each
/// number does and does not claim.
struct BaselineComparisonReport {
  std::string fixture;
  int minII{};
  std::optional<int> jointII;
  std::optional<int> relaxedLowerBound;
  llvm::SmallVector<BaselineII, 3> baselines;

  /// Every backend landed on MinII, so the fixture is MinII-bound and cannot
  /// separate them: no scheduler can go lower, so a tie is a property of the
  /// fixture, not evidence about the solver.
  bool isMinIIBound() const;

  /// `jointII >= relaxedLowerBound` — a soundness invariant, not an
  /// improvement. Only meaningful when both numbers exist.
  bool soundnessCheckable() const {
    return jointII.has_value() && relaxedLowerBound.has_value();
  }
  bool soundnessHolds() const {
    return !soundnessCheckable() || *jointII >= *relaxedLowerBound;
  }

  /// The acceptance gate: Rau is Triton's current default and the path the
  /// joint solver's own fallback takes, so beating it is the claim that
  /// matters operationally.
  std::optional<int> baselineII(llvm::StringRef name) const;
  bool improvesOnRau() const;  // strictly better
  bool regressesOnRau() const; // strictly worse — a hard failure
};

/// True when TRITON_MODULO_BASELINE_REPORT is set. The comparison runs every
/// backend on the DDG, so it is opt-in and never on a compile path.
bool baselineReportRequested();

BaselineComparisonReport compareAgainstBaselines(const DataDependenceGraph &ddg,
                                                 llvm::StringRef fixture);

/// Emit the comparison as a fixed-width table plus one verdict line per
/// criterion. Written to `os` (the pass uses llvm::errs(), so the table does
/// not interleave with triton-opt's IR on stdout).
void printBaselineComparison(llvm::raw_ostream &os,
                             const BaselineComparisonReport &report);

/// Result of list scheduling for a non-loop region. The algorithm itself
/// lives in `ListSchedulePass.cpp` (kept there so its debug output is
/// gated by `-debug-only=nvgpu-list-schedule`).
struct ListScheduleResult {
  int makespan{}; // total cycles from first op start to last op end
  llvm::DenseMap<unsigned, int> nodeToCycle; // DDG node idx -> absolute cycle
};

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_RESERVATION_TABLE_H
