#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_RESERVATION_TABLE_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_RESERVATION_TABLE_H

#include "DataDependenceGraph.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

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

/// Run modulo scheduling on the DDG.
/// Algorithm selected by TRITON_USE_MODULO_SCHEDULE env var value:
///   "sms"        → Swing Modulo Scheduling (Llosa et al., PACT 1996)
///   "exhaustive" → Exhaustive search with joint memory feasibility
///   "random"     → Random sampling with greedy placement
///   "1" or other → Rau's Iterative Modulo Scheduling (Rau, 1994)
/// maxII defaults to 2 * MinII. maxBacktracks limits ejection in Rau's IMS.
FailureOr<ModuloScheduleResult>
runModuloScheduling(const DataDependenceGraph &ddg, int maxII = 0,
                    int maxBacktracks = 20);

/// Result of list scheduling for a non-loop region. The algorithm itself
/// lives in `ListSchedulePass.cpp` (kept there so its debug output is
/// gated by `-debug-only=nvgpu-list-schedule`).
struct ListScheduleResult {
  int makespan{}; // total cycles from first op start to last op end
  llvm::DenseMap<unsigned, int> nodeToCycle; // DDG node idx -> absolute cycle
};

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_RESERVATION_TABLE_H
