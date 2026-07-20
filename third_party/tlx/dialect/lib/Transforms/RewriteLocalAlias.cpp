#include "IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
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

namespace {

static ttg::MemDescReinterpretOp createAliasReinterpret(OpBuilder &builder,
                                                        Location loc,
                                                        ttg::MemDescType type,
                                                        Value source) {
  auto op = ttg::MemDescReinterpretOp::create(builder, loc, type, source);
  auto sourceType = cast<ttg::MemDescType>(source.getType());
  if (ttg::isPaddedEncoding(sourceType.getEncoding()) ||
      ttg::isPaddedEncoding(type.getEncoding()))
    op->setAttr("tlx.allow_different_padding_pattern", builder.getUnitAttr());
  return op;
}

// Total logical storage bits for a memdesc view. This deliberately mirrors
// MemDescReinterpretOp::verify so aliases are constructed from physical
// storage sizes rather than logical tensor element counts.
int64_t getMemDescStorageBits(ttg::MemDescType ty) {
  auto rank = cast<ttg::LayoutEncodingTrait>(ty.getEncoding()).getRank();
  auto shape = ty.getAllocShape().take_back(rank);
  LinearLayout layout = ttg::isPaddedEncoding(ty.getEncoding())
                            ? ttg::paddedLinearLayout(shape, ty.getEncoding())
                            : ttg::toLinearLayout(shape, ty.getEncoding());
  int64_t numLayoutCopies = 1;
  for (int64_t dim : ty.getAllocShape().drop_back(rank))
    numLayoutCopies *= dim;
  auto *ctx = ty.getContext();
  bool isSharedMemory = isa<ttg::SharedMemorySpaceAttr>(ty.getMemorySpace());
  auto dim = StringAttr::get(ctx, isSharedMemory ? "offset" : "col");
  return numLayoutCopies * layout.getInDimSize(dim) *
         ty.getElementTypeBitWidth();
}

// Produce a descriptor of dstTy that views base's backing allocation. A
// smaller alias cannot be represented by a size-changing reinterpret, so
// reinterpret the whole allocation as repeated destination-layout slots and
// select slot zero structurally with memdesc_index.
FailureOr<Value> emitAliasView(OpBuilder &builder, Operation *errorOp,
                               Location loc, Value base,
                               ttg::MemDescType dstTy) {
  auto srcTy = cast<ttg::MemDescType>(base.getType());
  int64_t srcBits = getMemDescStorageBits(srcTy);
  int64_t dstBits = getMemDescStorageBits(dstTy);

  if (srcBits == dstBits)
    return createAliasReinterpret(builder, loc, dstTy, base).getResult();

  if (srcBits < dstBits || srcBits % dstBits != 0)
    return errorOp->emitError()
           << "TLXRewriteLocalAlias cannot view a " << srcBits
           << "-bit allocation as a " << dstBits << "-bit alias";
  int64_t ratio = srcBits / dstBits;
  unsigned encRank =
      cast<ttg::LayoutEncodingTrait>(dstTy.getEncoding()).getRank();
  int64_t dstRank = dstTy.getRank();
  assert((dstRank == static_cast<int64_t>(encRank) ||
          dstRank == static_cast<int64_t>(encRank) + 1) &&
         "MemDescType rank must equal encoding rank or be one greater");
  int64_t dstBatch =
      dstRank == static_cast<int64_t>(encRank) + 1 ? dstTy.getShape()[0] : 1;
  if (dstBatch != 1)
    return errorOp->emitError()
           << "TLXRewriteLocalAlias cannot shrink a size-mismatched alias "
              "with leading batch dim "
           << dstBatch << " (only unit batch dim is supported)";

  auto layoutShape = dstTy.getShape().take_back(encRank);
  SmallVector<int64_t> intermediateShape;
  intermediateShape.reserve(encRank + 1);
  intermediateShape.push_back(ratio);
  intermediateShape.append(layoutShape.begin(), layoutShape.end());
  auto intermediateTy = ttg::MemDescType::get(
      intermediateShape, dstTy.getElementType(), dstTy.getEncoding(),
      dstTy.getMemorySpace(), dstTy.getMutableMemory(), intermediateShape);
  Value intermediate =
      createAliasReinterpret(builder, loc, intermediateTy, base).getResult();

  SmallVector<int64_t> slotShape(layoutShape.begin(), layoutShape.end());
  auto slotTy = ttg::MemDescType::get(
      slotShape, dstTy.getElementType(), dstTy.getEncoding(),
      dstTy.getMemorySpace(), dstTy.getMutableMemory(), slotShape);
  Value zero = arith::ConstantIntOp::create(builder, loc, /*value=*/0,
                                            /*width=*/32);
  Value slot =
      ttg::MemDescIndexOp::create(builder, loc, slotTy, intermediate, zero)
          .getResult();

  if (dstRank == static_cast<int64_t>(encRank))
    return slot;
  return Value(createAliasReinterpret(builder, loc, dstTy, slot).getResult());
}

} // namespace

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
      Operation *newAllocOp = nullptr;
      if (isa<ttg::LocalAllocOp>(baseAllocOp)) {
        newAllocOp =
            ttg::LocalAllocOp::create(builder, baseAllocOp->getLoc(), maxType);
      } else {
        assert(isa<ttng::TMEMAllocOp>(baseAllocOp) && "Unexpected alloc op");
        newAllocOp = ttng::TMEMAllocOp::create(builder, baseAllocOp->getLoc(),
                                               maxType, nullptr);
      }
      // Save mapping so we can rewrite uses later.
      allocToNewAlloc[baseAllocOp] = newAllocOp;
    }
  }

  // Rewrite uses of local_alias ops to use the new local_alloc op.
  for (auto &kv : aliasClasses) {
    // Replace the base alloc op with the new one if it exists.
    Operation *baseAllocOp = kv.first;
    if (Operation *newAllocOp = allocToNewAlloc.lookup(baseAllocOp)) {
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
      auto baseAllocType =
          dyn_cast<ttg::MemDescType>(baseAllocOp->getResult(0).getType());
      FailureOr<Value> newAllocToBaseAlloc =
          emitAliasView(builder, baseAllocOp, baseAllocOp->getLoc(),
                        newAllocOp->getResult(0), baseAllocType);
      if (failed(newAllocToBaseAlloc))
        return failure();
      baseAllocOp->getResult(0).replaceAllUsesWith(*newAllocToBaseAlloc);
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
      auto aliasType = cast<ttg::MemDescType>(aliasOp.getResult().getType());
      FailureOr<Value> baseAllocToAlias =
          emitAliasView(builder, aliasOp, baseAllocOp->getLoc(),
                        baseAllocOp->getResult(0), aliasType);
      if (failed(baseAllocToAlias))
        return failure();
      aliasOp.getResult().replaceAllUsesWith(*baseAllocToAlias);
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
