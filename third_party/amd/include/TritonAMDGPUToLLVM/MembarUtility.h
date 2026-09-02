#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/Operation.h"
#include "triton/Analysis/Allocation.h"

namespace mlir {
class DialectRegistry;
}

namespace mlir::triton::AMD {

// Filter function used in the AMDGPU backend to remove dependencies that do
// not require a workgroup barrier. AsyncWait synchronization only filters the
// producer-to-consumer RAW edge from an async LDS write to its LocalLoad. It
// does not filter the consumer-to-producer WAR edge when that LDS slice is
// refilled: every workgroup consumer must release a cooperatively owned slice
// before any wave overwrites it. Distinct constant memdesc_index slices are
// proven independent by the generic Membar allocation-slice analysis.
bool membarFilter(Operation *op1, Operation *op2, bool op1IsRead,
                  bool op2IsRead, Allocation *allocation);

// Registers an external interface model marking the AMD scheduling-only fences
// rocdl.sched.barrier / rocdl.sched.group.barrier with
// ttg::SchedulingBarrierOpInterface. Membar's forward scan for a sync point
// looks THROUGH ops carrying that interface (they have no cross-wave memory
// semantics), so it does not insert a redundant barrier around the real
// ttg.barrier that a tlx.workgroup_barrier interposes. Attaching via a dialect
// extension keeps the core Membar analysis free of any ROCDL dependency.
void registerSchedulingBarrierExternalModel(DialectRegistry &registry);
} // namespace mlir::triton::AMD

#endif
