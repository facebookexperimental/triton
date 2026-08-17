#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PLANRINGMATERIALIZER_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PLANRINGMATERIALIZER_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir::triton::amdgpu {

/// A fully resolved mutation of one existing LDS ring. This deliberately does
/// not describe creation of a new staging allocation; that is M1.5b.4.
struct PlanExistingLdsRingMutation {
  scf::ForOp loop;
  gpu::LocalAllocOp allocation;
  int64_t oldDepth = 0;
  int64_t newDepth = 0;
  int64_t oldDistance = 0;
  int64_t newDistance = 0;
  SmallVector<gpu::MemDescIndexOp> producerViews;
  SmallVector<gpu::MemDescIndexOp> consumerViews;
  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
};

/// A wait-count change derived from a complete positive-distance wait family.
struct PlanAsyncWaitMutation {
  gpu::AsyncWaitOp wait;
  int64_t oldRetainedGroupCount = 0;
  int64_t newRetainedGroupCount = 0;
  SmallVector<Operation *> consumers;
};

struct PlanRingMaterializationResult {
  int64_t resizedAllocations = 0;
  int64_t rewrittenSlotIndices = 0;
  int64_t updatedWaits = 0;
  int64_t insertedVisibilityBarriers = 0;
  int64_t insertedReleaseBarriers = 0;
};

/// Materialize already-resolved ring and synchronization mutations. All
/// validation that depends on stable Plan IR identity happens before this
/// routine. It accepts only direct, existing memdesc_index views.
LogicalResult
materializeExistingLdsRings(ArrayRef<PlanExistingLdsRingMutation> rings,
                            ArrayRef<PlanAsyncWaitMutation> waits,
                            PlanRingMaterializationResult &result,
                            std::string &error);

} // namespace mlir::triton::amdgpu

#endif
