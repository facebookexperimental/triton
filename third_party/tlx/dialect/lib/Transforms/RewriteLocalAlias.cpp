#include "IR/Dialect.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tlx-rewrite-local-alias"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXREWRITELOCALALIAS

#include "tlx/dialect/include/Transforms/Passes.h.inc"

LogicalResult rewriteLocalAlias(ModuleOp m) {
  // Build a closure of all local_alloc and local_alias ops that share the same
  // physical memory
  LDBG("rewriteLocalAlias\n");

  // Forward map: alloc op -> alias ops
  DenseMap<Operation *, SmallVector<tlx::LocalAliasOp, 4>> aliasClasses;
  // Reverse map: alias op -> base alloc op
  DenseMap<tlx::LocalAliasOp, Operation *> aliasToAlloc;

  // Collect alias ops and bucket them by their base local alloc.
  WalkResult result = m.walk([&](Operation *op) -> WalkResult {
    if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op)) {
      assert(aliasClasses.count(op) == 0 && "Duplicate alloc op");
      aliasClasses[op] = {};
    } else if (auto aliasOp = dyn_cast<tlx::LocalAliasOp>(op)) {
      auto alias = aliasOp.getSrc();
      auto srcOp = alias.getDefiningOp();
      if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(srcOp)) {
        assert(aliasClasses.count(srcOp) && "Base alloc op not in map");
        aliasClasses[srcOp].push_back(aliasOp);
        aliasToAlloc[aliasOp] = srcOp;
      } else if (auto srcAliasOp = dyn_cast<tlx::LocalAliasOp>(srcOp)) {
        srcOp = aliasToAlloc[srcAliasOp];
        assert(srcOp && "Alias op must refer to a local alloc");
        assert(aliasClasses.count(srcOp) && "Base alloc not in map");
        aliasClasses[srcOp].push_back(aliasOp);
        aliasToAlloc[aliasOp] = srcOp;
      } else {
        op->emitError(
            "LocalAliasOp must refer to a local_alloc or local_alias op");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }

  LLVM_DEBUG({
    LDBG("Alias classes");
    for (auto &kv : aliasClasses) {
      Operation *allocOp = kv.first;
      auto &aliases = kv.second;
      DBGS() << "  Base alloc: ";
      allocOp->dump();
      for (auto alias : aliases) {
        DBGS() << "     aliases: ";
        alias->dump();
      }
      llvm::dbgs() << "\n";
    }
  });

  if (aliasToAlloc.empty()) {
    LDBG("No LocalAliasOp");
    return success();
  }

  // Compute the max shape of an alias class
  DenseMap<Operation *, ttg::MemDescType> allocToMaxStorageType;
  for (auto &kv : aliasClasses) {
    auto allocOp = kv.first;
    auto &aliases = kv.second;
    auto allocType =
        dyn_cast<ttg::MemDescType>(allocOp->getResult(0).getType());
    auto maxStorageType = allocType;
    auto maxStorageSize =
        allocType.getNumElements() * allocType.getElementTypeBitWidth();
    for (tlx::LocalAliasOp alias : aliases) {
      auto aliasType = dyn_cast<ttg::MemDescType>(alias.getResult().getType());
      auto aliasStorageSize =
          aliasType.getNumElements() * aliasType.getElementTypeBitWidth();
      if (aliasStorageSize > maxStorageSize) {
        maxStorageType = aliasType;
        maxStorageSize = aliasStorageSize;
      }
    }
    allocToMaxStorageType[allocOp] = maxStorageType;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n\n";
    LDBG("Max storage type for each alloc op");
    for (auto &kv : allocToMaxStorageType) {
      auto allocOp = kv.first;
      auto maxStorageType = kv.second;
      DBGS() << "  alloc: ";
      allocOp->dump();
      DBGS() << "     max storage type: ";
      maxStorageType.dump();
      llvm::dbgs() << "\n";
    }
  });

  // Create a new local_alloc op for each alias class if the max storage type
  // isn't the same as the base alloc type
  DenseMap<Operation *, Operation *> allocToNewAlloc;
  OpBuilder builder(m.getContext());
  for (auto &kv : aliasClasses) {
    Operation *baseAllocOp = kv.first;
    auto baseAllocType =
        dyn_cast<ttg::MemDescType>(baseAllocOp->getResult(0).getType());

    ttg::MemDescType maxType = allocToMaxStorageType[baseAllocOp];
    if (maxType != baseAllocType) {
      // Need a new alloc with the larger type.
      builder.setInsertionPoint(baseAllocOp);
      auto newAllocOp =
          builder.create<ttg::LocalAllocOp>(baseAllocOp->getLoc(), maxType);
      // Save mapping so we can rewrite uses later.
      allocToNewAlloc[baseAllocOp] = newAllocOp;
    }
  }

  // Rewrite uses of local_alias ops to use the new local_alloc op.
  for (auto &kv : aliasClasses) {
    // Replace the base alloc op with the new one if it exists.
    Operation *baseAllocOp = kv.first;
    if (allocToNewAlloc.count(baseAllocOp)) {
      auto newAllocOp = allocToNewAlloc[baseAllocOp];
      // Create a memdesc reinterpret op to convert the new alloc to the base
      // alloc
      LLVM_DEBUG({
        llvm::dbgs() << "\n";
        DBGS() << "Rewrite base alloc: ";
        baseAllocOp->dump();
        DBGS() << "  to: ";
        newAllocOp->dump();
      });

      builder.setInsertionPoint(baseAllocOp);
      auto newAllocType =
          dyn_cast<ttg::MemDescType>(newAllocOp->getResult(0).getType());
      auto baseAllocType =
          dyn_cast<ttg::MemDescType>(baseAllocOp->getResult(0).getType());
      auto newAllocToBaseAllocOp = builder.create<ttg::MemDescReinterpretOp>(
          baseAllocOp->getLoc(), baseAllocType, newAllocOp->getResult(0));
      baseAllocOp->getResult(0).replaceAllUsesWith(
          newAllocToBaseAllocOp.getResult());
      baseAllocOp->erase();
      baseAllocOp = newAllocOp;
    }

    // Rewrite all alias ops in the class to use the new/base alloc op.
    auto &aliases = kv.second;
    for (tlx::LocalAliasOp aliasOp : aliases) {
      LLVM_DEBUG({
        llvm::dbgs() << "\n";
        DBGS() << "Rewrite alias: ";
        aliasOp->dump();
      });
      builder.setInsertionPoint(aliasOp);
      auto aliasType = aliasOp.getResult().getType();
      auto baseAllocToAliasOp = builder.create<ttg::MemDescReinterpretOp>(
          baseAllocOp->getLoc(), aliasType, baseAllocOp->getResult(0));
      aliasOp.getResult().replaceAllUsesWith(baseAllocToAliasOp.getResult());
      aliasOp->erase();
    }
  }

  return success();
}

struct TLXRewriteLocalAliasPass
    : public impl::TLXRewriteLocalAliasBase<TLXRewriteLocalAliasPass> {
public:
  using impl::TLXRewriteLocalAliasBase<
      TLXRewriteLocalAliasPass>::TLXRewriteLocalAliasBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (failed(tlx::rewriteLocalAlias(m))) {
      signalPassFailure();
    }
  }
};
} // namespace tlx
} // namespace triton
} // namespace mlir
