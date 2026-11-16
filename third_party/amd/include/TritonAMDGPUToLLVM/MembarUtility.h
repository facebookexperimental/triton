#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/Operation.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

namespace mlir {
class DialectRegistry;
struct AllocationSlice;
struct MembarScratchSync;
} // namespace mlir

namespace mlir::triton::AMD {

// Filter function used in the AMDGPU backend to remove dependencies that do
// not require a workgroup barrier. Two GlobalToLocalCopyTrait ops do not need
// an ordinary workgroup barrier between them: async synchronization governs
// their completion. AsyncWait synchronization only filters the
// producer-to-consumer RAW edge from an async LDS write to its LocalLoad. It
// does not filter the consumer-to-producer WAR edge when that LDS slice is
// refilled: every workgroup consumer must release a cooperatively owned slice
// before any wave overwrites it. Distinct constant memdesc_index slices are
// proven independent by the generic Membar allocation-slice analysis.
bool membarFilter(Operation *op1, Operation *op2, bool /*op1IsRead*/,
                  bool /*op2IsRead*/, Allocation *allocation,
                  const AllocationSlice &, const AllocationSlice &);

// Pair the allocated warp-local scratch reuse proof with its required leading
// wave fence. Callers using only membarFilter retain CTA synchronization for
// scratch reuse, since they have no corresponding wave-fence emitter.
MembarScratchSync getWarpLocalScratchSync();

// Registers an external interface model marking the AMD scheduling-only fences
// rocdl.sched.barrier / rocdl.sched.group.barrier with
// ttg::SchedulingBarrierOpInterface. Membar's forward scan for a sync point
// looks THROUGH ops carrying that interface (they have no cross-wave memory
// semantics), so it does not insert a redundant barrier around the real
// ttg.barrier that a tlx.workgroup_barrier interposes. Attaching via a dialect
// extension keeps the core Membar analysis free of any ROCDL dependency.
void registerSchedulingBarrierExternalModel(DialectRegistry &registry);

// Membar with the AMD split workgroup barrier: the signal/wait pair
// synchronizes shared-memory accesses issued before the signal with those
// after the wait. By convention the window between them holds no
// shared-memory access, so Membar never needs a barrier inside it.
class AMDMembarAnalysis : public MembarAnalysis {
public:
  AMDMembarAnalysis(Allocation &allocation, MembarFilterFn filter,
                    MembarScratchSync scratchSync,
                    triton::BufferRegionAnalysis &regions)
      : MembarAnalysis(allocation, std::move(filter), std::move(scratchSync),
                       regions) {}

protected:
  triton::BarrierStages getBarrierStages(Operation *op) override;
};
} // namespace mlir::triton::AMD

#endif
