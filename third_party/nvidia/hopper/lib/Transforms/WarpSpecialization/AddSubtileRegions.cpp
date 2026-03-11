#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvgpu-add-subtile-regions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

#define GEN_PASS_DEF_NVGPUADDSUBTILEREGIONS
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUAddSubtileRegionsPass
    : public impl::NVGPUAddSubtileRegionsBase<NVGPUAddSubtileRegionsPass> {
public:
  using NVGPUAddSubtileRegionsBase::NVGPUAddSubtileRegionsBase;

  void runOnOperation() override {
    // Placeholder — subtile region generation will be added in a follow-up.
  }
};

} // namespace mlir
