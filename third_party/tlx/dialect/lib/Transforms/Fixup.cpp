#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "tlx/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/LogicalResult.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::tlx {

#define GEN_PASS_DEF_TRITONTLXFIXUP
#include "tlx/dialect/include/Transforms/Passes.h.inc"

class TritonTLXFixupPass : public impl::TritonTLXFixupBase<TritonTLXFixupPass> {
  using impl::TritonTLXFixupBase<TritonTLXFixupPass>::TritonTLXFixupBase;

public:
  // validate the module and error early for unsupported cases
  LogicalResult verifyModule(ModuleOp &mod, bool tlx_2cta) {
    // ws should not capture RankedTensorType
    ttg::WarpSpecializeOp invalidWSOp = nullptr;
    auto result = mod.walk([&](ttg::WarpSpecializeOp op) {
      for (auto argType : op.getPartitionOp().getOperandTypes()) {
        if (isa<RankedTensorType>(argType)) {
          invalidWSOp = op;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return invalidWSOp.emitError() << "WarpSpecializeOp should not capture "
                                        "RankedTensorType. Try moving tensor "
                                        "computation into specific async task.";
    }

    if (tlx_2cta) {
      if (numCTAs > 1) {
        return mod.emitError()
               << "num_ctas should not be set for TLX 2cta mode";
      }

      // all the async_dot ops need to be either 1cta or 2cta together
      auto walkResult = mod.walk([&](ttng::TCGen5MMAOp tcgen05MMAOp) {
        if (!tcgen05MMAOp.getTwoCtas()) {
          tcgen05MMAOp.emitError()
              << "Expecting all dot ops to be 2cta together or 1cta together";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted()) {
        return failure();
      }
    } else {
      bool isClustered = false;
      if (!clusterDims.empty()) {
        // Ensure we have exactly 3 dimensions (X, Y, Z)
        if (clusterDims.size() != 3) {
          return mod.emitError()
                 << "Expected 3 cluster dimensions, got " << clusterDims.size();
        }
        isClustered = (clusterDims[0] * clusterDims[1] * clusterDims[2]) > 1;
      }
      // There should not be a mapa in unclustered mode
      if (!isClustered) {
        if (mod.walk([&](ttng::MapToRemoteBufferOp mapaOp) {
                 mapaOp.emitError()
                     << "Unexpected buffer remote view in 1cta mode";
                 return WalkResult::interrupt();
               })
                .wasInterrupted()) {
          return failure();
        }
      }
    }
    return success();
  }

  bool isAMD() const {
    // target is set up as f"hip:{options.arch}"
    return (target.getValue().find("hip:") == 0);
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    auto hasTLXTwoCTAs =
        mod.walk([&](Operation *op) {
             if (auto tcgen05MMAOp = dyn_cast<ttng::TCGen5MMAOp>(op)) {
               if (tcgen05MMAOp.getTwoCtas()) {
                 return WalkResult::interrupt();
               }
             } else if (auto tcgen05MMAScaledOp =
                            dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
               if (tcgen05MMAScaledOp.getTwoCtas()) {
                 return WalkResult::interrupt();
               }
             }
             return WalkResult::advance();
           })
            .wasInterrupted();

    if (failed(verifyModule(mod, hasTLXTwoCTAs))) {
      return signalPassFailure();
    }

    // First check if there is any TLX related op in the module. If not, do
    // nothing.
    auto tlxDialectName = TLXDialect::getDialectNamespace();
    WalkResult result = mod.walk([&](Operation *op) {
      // Ops directly in TLX Dialect
      if (op->getDialect()->getNamespace() == tlxDialectName) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    auto hasTLXOps = result.wasInterrupted();

    auto hasExplicitLocalMemAccess =
        mod.walk([&](Operation *op) {
             // Ops that should not be in TTIR unless introduced by TLX
             if (isa<ttg::LocalLoadOp, ttg::LocalStoreOp,
                     ttng::AsyncTMACopyGlobalToLocalOp,
                     ttng::AsyncTMACopyLocalToGlobalOp, ttng::TMEMAllocOp,
                     ttng::TMEMLoadOp, ttng::TMEMStoreOp, ttng::TCGen5MMAOp>(
                     op)) {
               return WalkResult::interrupt();
             }
             return WalkResult::advance();
           })
            .wasInterrupted();

    auto hasWarpSpecOps = mod.walk([&](Operation *op) {
                               if (isa<ttg::WarpSpecializeOp, ttg::WarpYieldOp,
                                       ttg::WarpReturnOp>(op)) {
                                 return WalkResult::interrupt();
                               }
                               return WalkResult::advance();
                             })
                              .wasInterrupted();

    if (!hasTLXOps && !hasExplicitLocalMemAccess && !hasWarpSpecOps &&
        !hasTLXTwoCTAs) {
      return;
    }

    // Attach metadata to the module.
    Builder b(&getContext());
    mod->setAttr(ttg::AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(ttg::AttrNumThreadsPerWarp,
                 b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(ttg::AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    mod->setAttr(ttg::AttrTargetName, b.getStringAttr(this->target.getValue()));
    if (hasTLXOps)
      mod->setAttr(AttrHasTLXOpsName, b.getBoolAttr(true));
    if (hasExplicitLocalMemAccess)
      mod->setAttr(AttrHasExplicitLocalMemAccessName, b.getBoolAttr(true));
    if (hasWarpSpecOps)
      mod->setAttr(AttrHasWarpSpecOpsName, b.getBoolAttr(true));
    if (hasTLXTwoCTAs) {
      mod->setAttr(AttrTLXEnablePairedCTAMMAName, b.getBoolAttr(true));
    }

    // Propagate the `exclusive` marker from `tlx.async_tasks(exclusive=True)`:
    // it enables the single-warp-specialize lowering, which requires the module
    // to contain exactly one warp_specialize op. Only NVIDIA consumes this; on
    // AMD `exclusive` is ignored.
    if (!isAMD()) {
      int numWarpSpecializeOps = 0;
      bool hasExclusiveWS = false;
      bool hasNoEndingClusterSync = false;
      mod.walk([&](ttg::WarpSpecializeOp op) {
        ++numWarpSpecializeOps;
        if (op->hasAttr("tlx.exclusive"))
          hasExclusiveWS = true;
        if (op->hasAttr("tlx.no_ending_cluster_sync"))
          hasNoEndingClusterSync = true;
      });
      if (hasExclusiveWS) {
        if (numWarpSpecializeOps != 1) {
          mod.emitError()
              << "tlx.async_tasks(exclusive=True) requires exactly one "
                 "warp_specialize op in the module, but found "
              << numWarpSpecializeOps;
          return signalPassFailure();
        }
        ttg::setHasSingleWarpSpecialize(mod, /*value=*/true);
      }
      // `no_ending_cluster_sync`: the user handles the post-warp-specialize
      // sync, so mark the module to skip the compiler's cluster arrive/wait
      // before TMEM dealloc.
      if (hasNoEndingClusterSync)
        setUserPostWsSyncOnMod(mod, /*value=*/true);
    }
  }
};

} // namespace mlir::triton::tlx
