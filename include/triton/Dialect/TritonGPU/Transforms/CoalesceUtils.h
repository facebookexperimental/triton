
#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {
// Compute dependency information once per function, before changing its IR.
class CoalescedEncodingAnalysis {
public:
  explicit CoalescedEncodingAnalysis(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                     unsigned maxVecBits = 128)
      : axisInfoAnalysis(axisInfoAnalysis), maxVecBits(maxVecBits) {}

  BlockedEncodingAttr getEncoding(Operation *op, int numWarps,
                                  int threadsPerWarp, CGAEncodingAttr cgaLayout,
                                  ArrayRef<int64_t> shapePerCTA);

private:
  // Whether another op with the same shape and contiguity order in op's slice
  // can use unaligned vectorized loads. Kept as a direct slice walk (see the
  // .cpp) so masked and volatile loads preserve their exact pre-refactor
  // predicate; the cached per-key summary cannot exclude the queried op.
  bool hasRelatedUnalignedVectorizedLoad(Operation *op,
                                         const SmallVector<unsigned> &order,
                                         RankedTensorType refTensorType);
  void computeSliceInfo(FunctionOpInterface func);
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
  unsigned maxVecBits;
  DenseMap<Operation *, unsigned> elementsPerThread;
};
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
