#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <optional>

namespace mlir::triton::nvidia_gpu {

LogicalResult verifyBarrierType(Operation *op,
                                mlir::triton::gpu::MemDescType barrierType);
int allocateTMemWithInterval(
    DenseMap<Operation *, Interval<int>> &allocToIntervals,
    SmallVector<Operation *> &allocOrder);

// A match of TMemSplitLoadPattern (OptimizeTMemLayouts), which rewrites
//   split(cvt*(trans{0,2,1}(reshape(cvt*(tmem_load)))))
// into one tmem_subslice + tmem_load per split result.
struct TMemSplitLoadMatch {
  TransOp trans;
  ReshapeOp reshape;
  // The reshape's source with layout conversions stripped. The pattern only
  // rewrites the chain when this is a tmem_load (`tmemLoad`).
  Value reshapeSrc;
  TMEMLoadOp tmemLoad;
  int64_t splitNSize;
  // The layout conversions stripped above and below the reshape.
  SmallVector<mlir::triton::gpu::ConvertLayoutOp> cvts;
};

// Matches the IR-shape conditions of TMemSplitLoadPattern on `split`. A
// reshape source that is not a tmem_load is still matched (`tmemLoad` is null),
// so a caller can match a nested split before the pattern rewrites the outer
// split into a tmem_load. The warp count is checked separately with
// isTMemSplitLoadNumWarps, since it depends on where the caller runs.
std::optional<TMemSplitLoadMatch> matchTMemSplitLoad(SplitOp split);

// Whether the TMEM load layout helper the pattern uses supports `numWarps`.
bool isTMemSplitLoadNumWarps(int numWarps);

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
