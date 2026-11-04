#ifndef TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_ALLOCATION_H
#define TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_ALLOCATION_H

#include "mlir/IR/Operation.h"
#include "triton/Analysis/Utility.h"

#include <functional>

namespace mlir {
namespace triton {
class TargetInfoBase;

namespace nvidia_gpu {
std::function<unsigned(Operation *)>
getNvidiaAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo);

// A Shared Product conversion moved as the pow2 swizzled transfers of its odd
// slabs, side by side in scratch, so each rep is one store/bar/load phase.
struct ProductSlabTransfer {
  ProductSharedTiling slabs;
  int32_t numSlabs;
  unsigned scratchBytes;
};
std::optional<ProductSlabTransfer>
getProductSlabTransfer(RankedTensorType srcTy, RankedTensorType dstTy,
                       const TargetInfoBase &targetInfo);

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
#endif // TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_ALLOCATION_H
