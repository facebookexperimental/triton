// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Modulo Buffer Allocation Pass (placeholder)
//
// Phase boundary marker between Pass A's schedule computation and the
// loop expansion phase. Currently a no-op — the actual buffer allocation
// is performed by lowerLoops() in ModuloExpandPass, which derives
// multi-buffer depths from loop.stage differences.
//
// TODO: Move PipelineGraph-based buffer allocation here once the
// PipelineGraph expansion path replaces lowerLoops().

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvgpu-modulo-buffer-alloc"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {

struct ModuloBufferAllocPass
    : public PassWrapper<ModuloBufferAllocPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuloBufferAllocPass)

  StringRef getArgument() const override { return "nvgpu-modulo-buffer-alloc"; }

  StringRef getDescription() const override {
    return "Buffer allocation for modulo scheduling (placeholder)";
  }

  void runOnOperation() override {
    LDBG("Phase 1 analysis complete "
         "(buffer allocation deferred to ModuloExpand/lowerLoops)");
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloBufferAlloc() {
  return std::make_unique<ModuloBufferAllocPass>();
}

void registerNVGPUModuloBufferAlloc() {
  PassRegistration<ModuloBufferAllocPass>();
}
} // namespace mlir
