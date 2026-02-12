#include "IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
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
LogicalResult processBufferOverlapOps(
    ModuleOp m, DenseMap<Value, std::pair<int64_t, int64_t>> &offsetMap);
LogicalResult materializeStorageAliasAllocations(
    ModuleOp m, const DenseMap<Value, std::pair<int64_t, int64_t>> &offsetMap,
    DenseMap<Value, std::pair<int64_t, int64_t>> &localAliasOffsetMap);

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

    // Step 2: Process buffer overlap operations (compute offsets)
    // This must run BEFORE materialization because:
    // - SetBufferOverlapOp uses StorageAliasSpecOp
    // - Materialization erases StorageAliasSpecOp
    // The computed offsets are returned in a map to be applied during
    // materialization.
    LDBG("Step 2: Processing buffer overlap operations");
    DenseMap<Value, std::pair<int64_t, int64_t>> offsetMap;
    if (failed(processBufferOverlapOps(m, offsetMap))) {
      signalPassFailure();
      return;
    }

    // Step 3: Materialize storage alias allocations
    // This creates LocalAllocOp/TMEMAllocOp and LocalAliasOp.
    // The computed offsets are stored in localAliasOffsetMap for later use.
    LDBG("Step 3: Materializing storage alias allocations");
    DenseMap<Value, std::pair<int64_t, int64_t>> localAliasOffsetMap;
    if (failed(materializeStorageAliasAllocations(m, offsetMap,
                                                  localAliasOffsetMap))) {
      signalPassFailure();
      return;
    }

    // Note: localAliasOffsetMap contains the buffer layout information for
    // LocalAliasOps that have custom offsets (from set_buffer_overlap).
    // This can be used in a future Step 4 for Phase 6 implementation.

    LDBG("TLXStorageAliasLowering completed successfully");
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
