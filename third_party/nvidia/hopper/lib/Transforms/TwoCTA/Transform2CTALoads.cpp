//===- Transform2CTALoads.cpp - Transform loads for 2-CTA mode -----------===//
//
// This pass transforms B matrix loads for 2-CTA (pair CTA) mode.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

// Define before including Passes.h to get the base class
#define GEN_PASS_DEF_NVGPU2CTATRANSFORMLOADS
#include "nvidia/hopper/include/Transforms/Passes.h"

using namespace mlir;
namespace ttng = triton::nvidia_gpu;

#define DEBUG_TYPE "nvgpu-2cta-transform-loads"
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

struct Transform2CTALoadsPass
    : public mlir::impl::NVGPU2CTATransformLoadsBase<Transform2CTALoadsPass> {
  using NVGPU2CTATransformLoadsBase::NVGPU2CTATransformLoadsBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Find all 2-CTA MMA operations
    SmallVector<ttng::TCGen5MMAOp> mmaOps = find2CTAMMAOps(moduleOp);

    if (mmaOps.empty()) {
      LDBG("No 2-CTA MMA operations found, nothing to do");
      return;
    }

    LDBG("Found " << mmaOps.size() << " 2-CTA MMA operations");

    // TODO: Implement B matrix load transformation
    // 1. Insert ClusterCTAIdOp to get CTA rank
    // 2. Compute offset: cta_rank * (BLOCK_N / 2)
    // 3. Modify B loads to load BLOCK_N/2 columns with offset
  }
};

} // namespace
