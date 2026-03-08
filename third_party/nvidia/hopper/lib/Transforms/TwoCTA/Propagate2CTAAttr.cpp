//===- Propagate2CTAAttr.cpp - Propagate two_ctas attr to TCGen5MMAOp -----===//
//
// This pass propagates the two_ctas attribute from the module to TCGen5MMAOp.
// When the frontend uses tl.dot(two_ctas=True), the cluster-dim-x is set to 2.
// This pass detects that configuration and sets two_ctas=true on all
// TCGen5MMAOp operations.
//
// This approach is similar to how TLX handles 2-CTA mode - TLX directly creates
// TCGen5MMAOp with two_ctas=true. For standard Triton, AccelerateMatmul creates
// the MMA ops but doesn't set two_ctas, so we set it here based on the cluster
// configuration.
//
// NOTE: This pass is skipped when warp specialization is enabled because the
// WS passes restructure the IR in ways that are incompatible with 2-CTA sync.
// For WS+2CTA, use TLX which has native support.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_DEF_NVGPU2CTAPROPAGATEATTR
#include "nvidia/hopper/include/Transforms/Passes.h"

using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

#define DEBUG_TYPE "nvgpu-2cta-propagate-attr"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// Check if warp specialization is enabled on any loop
static bool isWarpSpecializationEnabled(ModuleOp moduleOp) {
  bool wsEnabled = false;
  moduleOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("tt.warp_specialize")) {
      wsEnabled = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return wsEnabled;
}

/// Check if the module is configured for 2-CTA mode based on cluster dimensions
static bool isModule2CTAEnabled(ModuleOp moduleOp) {
  // Check cluster-dim-x attribute
  if (auto clusterDimX = moduleOp->getAttrOfType<IntegerAttr>("ttg.cluster-dim-x")) {
    if (clusterDimX.getInt() == 2) {
      return true;
    }
  }
  return false;
}

/// Check if a TCGen5MMAOp meets the requirements for 2-CTA mode
static bool canEnable2CTA(ttng::TCGen5MMAOp mmaOp) {
  // Get the accumulator type to check dimensions
  auto accType = mmaOp.getD().getType();
  if (auto memDescType = dyn_cast<ttg::MemDescType>(accType)) {
    auto shape = memDescType.getShape();
    if (shape.size() >= 2) {
      int64_t m = shape[0];
      int64_t n = shape[1];
      // Minimum size supported by 2-CTA MMAv5
      if (m >= 64 && n >= 32) {
        return true;
      }
    }
  }
  return true; // Default to allowing if we can't determine shape
}

struct Propagate2CTAAttrPass
    : public mlir::impl::NVGPU2CTAPropagateAttrBase<Propagate2CTAAttrPass> {
  using NVGPU2CTAPropagateAttrBase::NVGPU2CTAPropagateAttrBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Skip if warp specialization is enabled
    // WS passes restructure the IR in ways that are incompatible with 2-CTA sync.
    // For WS+2CTA, use TLX which has native support.
    if (isWarpSpecializationEnabled(moduleOp)) {
      LDBG("Warp specialization enabled (tt.warp_specialize attr found)");
      LDBG("Skipping 2-CTA attribute propagation - use TLX for WS+2CTA");
      return;
    }

    // Check if module is configured for 2-CTA mode
    if (!isModule2CTAEnabled(moduleOp)) {
      LDBG("Module not configured for 2-CTA mode (cluster-dim-x != 2)");
      return;
    }

    LDBG("Module configured for 2-CTA mode, propagating attribute to MMA ops");

    int numModified = 0;

    // Find all TCGen5MMAOp operations and set two_ctas=true
    moduleOp.walk([&](ttng::TCGen5MMAOp mmaOp) {
      // Skip if already has two_ctas set
      if (mmaOp.getTwoCtas()) {
        LDBG("MMA op already has two_ctas=true, skipping");
        return;
      }

      // Check if this MMA op meets requirements for 2-CTA
      if (!canEnable2CTA(mmaOp)) {
        LDBG("MMA op does not meet 2-CTA requirements, skipping");
        return;
      }

      // Set two_ctas attribute
      mmaOp.setTwoCtas(true);
      numModified++;
      LDBG("Set two_ctas=true on TCGen5MMAOp");
    });

    LDBG("Modified " << numModified << " TCGen5MMAOp operations");
  }
};

} // namespace
