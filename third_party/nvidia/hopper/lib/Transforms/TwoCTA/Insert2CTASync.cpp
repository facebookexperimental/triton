//===- Insert2CTASync.cpp - Insert cross-CTA sync for 2-CTA mode ---------===//
//
// This pass inserts the cross-CTA synchronization pattern for 2-CTA MMA.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

// Define before including Passes.h to get the base class
#define GEN_PASS_DEF_NVGPU2CTAINSERTSYNC
#include "nvidia/hopper/include/Transforms/Passes.h"

using namespace mlir;
namespace ttng = triton::nvidia_gpu;

#define DEBUG_TYPE "nvgpu-2cta-insert-sync"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// Find all TCGen5MMAOp operations with two_ctas=true
static SmallVector<ttng::TCGen5MMAOp> find2CTAMMAOps(ModuleOp moduleOp) {
  SmallVector<ttng::TCGen5MMAOp> result;
  moduleOp.walk([&](ttng::TCGen5MMAOp mmaOp) {
    if (mmaOp.getTwoCtas()) {
      result.push_back(mmaOp);
    }
  });
  return result;
}

struct Insert2CTASyncPass
    : public mlir::impl::NVGPU2CTAInsertSyncBase<Insert2CTASyncPass> {
  using NVGPU2CTAInsertSyncBase::NVGPU2CTAInsertSyncBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Find all 2-CTA MMA operations
    SmallVector<ttng::TCGen5MMAOp> mmaOps = find2CTAMMAOps(moduleOp);

    if (mmaOps.empty()) {
      LDBG("No 2-CTA MMA operations found, nothing to do");
      return;
    }

    LDBG("Found " << mmaOps.size() << " 2-CTA MMA operations");

    // TODO: Implement cross-CTA synchronization
    // The pattern is "arrive remote, wait local":
    // 1. Get CTA rank via ClusterCTAIdOp
    // 2. Compute leader rank: leader_rank = cta_rank & ~1
    // 3. Map local barrier to leader: MapToRemoteBufferOp
    // 4. Arrive on leader's barrier: ArriveBarrierOp
    // 5. Leader waits: WaitBarrierOp with predicate
  }
};

} // namespace
