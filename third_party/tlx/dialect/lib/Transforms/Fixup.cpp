#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "tlx/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <mlir/Analysis/SliceAnalysis.h>

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::tlx {

#define GEN_PASS_DEF_TRITONTLXFIXUP
#include "tlx/dialect/include/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tlx-fixup"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class TritonTLXFixupPass : public impl::TritonTLXFixupBase<TritonTLXFixupPass> {
  using impl::TritonTLXFixupBase<TritonTLXFixupPass>::TritonTLXFixupBase;

public:
  // validate the module and error early for unsupported cases
  LogicalResult verifyModule(ModuleOp &mod) {
    // ws should not capture RankedTensorType
    ttg::WarpSpecializeOp invalidWSOp = nullptr;
    auto result = mod.walk([&](ttg::WarpSpecializeOp op) {
      for (auto argType : op.getOperandTypes()) {
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
    return success();
  }

  LogicalResult insertInvalBarrier(ModuleOp &mod) {
    auto hasInvalBarrierOp = mod.walk([&](Operation *op) {
                                  if (isa<ttng::InvalBarrierOp>(op)) {
                                    return WalkResult::interrupt();
                                  }
                                  return WalkResult::advance();
                                })
                                 .wasInterrupted();
    if (hasInvalBarrierOp) {
      return mod.emitError() << "InvalBarrierOp unexpectedly found. Unable to "
                                "auto insert InvalBarrierOp.";
    }

    // Find all barrier init ops in the module
    std::vector<Value> barriers;
    mod.walk(
        [&](ttng::InitBarrierOp op) { barriers.push_back(op.getAlloc()); });

    // Find the entry funcOp
    triton::FuncOp funcOp = nullptr;
    mod.walk([&](triton::FuncOp op) {
      if (triton::isKernel(op)) {
        funcOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // todo: consider removing all the inval op that's located right before
    // return in a later pass to save a few cycles.
    // Insert InvalBarrierOp before returnOp of
    // entry funcOp
    funcOp.walk([&](triton::ReturnOp op) {
      OpBuilder builder(op); // Insert *before* returnOp
      Location loc = op.getLoc();
      for (auto barrier : barriers) {
        builder.create<ttng::InvalBarrierOp>(loc, barrier);
      }
    });

    return success();
  }

  LogicalResult insertClusterSync(ModuleOp &mod) {
    SetVector<Operation *> barAllocOps;
    // Find all bar alloc op in the back slice of mapa ops
    mod.walk([&](ttng::MapToRemoteBufferOp mapaOp) {
      SetVector<Operation *> ops;
      if (failed(getBackwardSlice(mapaOp.getOperation(), &ops))) {
        llvm_unreachable("Failed to find bar alloc for MapToRemoteBufferOp");
      }
      for (auto op : ops) {
        if (isa<ttg::LocalAllocOp>(op)) {
          barAllocOps.insert(op);
        }
      }
    });

    // Find all the bar init op from the alloc ops who had a mapa usage later
    SetVector<Operation *> barInitOps;
    for (auto barAllocOp : barAllocOps) {
      SetVector<Operation *> ops;
      getForwardSlice(barAllocOp, &ops);
      for (auto op : ops) {
        if (isa<ttng::InitBarrierOp>(op)) {
          barInitOps.insert(op);
        }
      }
    }

    if (barInitOps.empty()) {
      return success();
    }

    // follow the program order and identify the last bar init op, assuming
    // all bar init happens at the first block of the proram for simplicity
    // TODO: add a module level check to make sure the assumption is true
    mod.walk<WalkOrder::PreOrder>([&](ttng::InitBarrierOp initBarOp) {
      if (barInitOps.size() <= 1) {
        return WalkResult::interrupt();
      }
      barInitOps.remove(initBarOp);
      return WalkResult::advance();
    });

    assert(barInitOps.size() == 1 &&
           "expecting only 1 bar init op left for cluster sync op insertion");
    auto lastBarInitOp = cast<ttng::InitBarrierOp>(barInitOps.back());
    OpBuilder builder(lastBarInitOp);
    builder.setInsertionPointAfter(lastBarInitOp);
    // need to insert cluster arrive and wait to prevent CTA_X from arriving
    // CTA_Y's bar before CTA_Y inits it, as shown in ptx doc examples:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
    builder.create<ttng::ClusterArriveOp>(lastBarInitOp.getLoc(),
                                          /*relaxed*/ false);
    builder.create<ttng::ClusterWaitOp>(lastBarInitOp.getLoc());

    return success();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (failed(verifyModule(mod))) {
      return signalPassFailure();
    }

    if (failed(insertInvalBarrier(mod))) {
      return signalPassFailure();
    }

    if (failed(insertClusterSync(mod))) {
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
    auto hasTLXTwoCTAs = mod.walk([&](ttng::TCGen5MMAOp tcgen05MMAOp) {
                              if (tcgen05MMAOp.getTwoCtas()) {
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
      assert(numCTAs <= 1 && "num_ctas should not be set for TLX 2cta mode");
      mod->setAttr(AttrIsTLXTwoCTAModeName, b.getBoolAttr(true));
    }
  }
};

} // namespace mlir::triton::tlx
