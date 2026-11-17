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

// A Shared mixed-radix conversion moved as the pow2 swizzled transfers of its
// odd tiles, `tilesPerRound` side by side per round, so scratch stays within
// the pow2 alloc shape's and each rep of a round is one store/bar/load.
struct MixedRadixTileTransfer {
  MixedRadixSharedTiling tiles;
  int32_t numTiles;
  int32_t tilesPerRound;
  unsigned scratchBytes;
};
std::optional<MixedRadixTileTransfer>
getMixedRadixTileTransfer(RankedTensorType srcTy, RankedTensorType dstTy,
                          const TargetInfoBase &targetInfo);

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
#endif // TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_ALLOCATION_H
