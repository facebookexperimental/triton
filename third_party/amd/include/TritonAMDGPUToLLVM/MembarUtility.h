#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/Operation.h"
#include "triton/Analysis/Allocation.h"

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
} // namespace mlir::triton::AMD

#endif
