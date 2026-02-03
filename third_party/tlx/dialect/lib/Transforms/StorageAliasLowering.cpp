#include "IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tlx-storage-alias-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXSTORAGEALIASLOWERING

#include "tlx/dialect/include/Transforms/Passes.h.inc"

// Forward declarations of functions from the individual passes
LogicalResult computeOrValidateStorageAliasSizes(ModuleOp m);
LogicalResult materializeStorageAliasAllocations(ModuleOp m);

struct TLXStorageAliasLoweringPass
    : public impl::TLXStorageAliasLoweringBase<TLXStorageAliasLoweringPass> {
public:
  using impl::TLXStorageAliasLoweringBase<
      TLXStorageAliasLoweringPass>::TLXStorageAliasLoweringBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    LDBG("Running TLXStorageAliasLowering (combined pass)");

    // Step 1: Compute or validate storage alias sizes
    LDBG("Step 1: Computing/validating storage alias sizes");
    if (failed(computeOrValidateStorageAliasSizes(m))) {
      signalPassFailure();
      return;
    }

    // Step 2: Materialize storage alias allocations
    LDBG("Step 2: Materializing storage alias allocations");
    if (failed(materializeStorageAliasAllocations(m))) {
      signalPassFailure();
      return;
    }

    LDBG("TLXStorageAliasLowering completed successfully");
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
