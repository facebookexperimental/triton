#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUMMACOLLECTORREUSEPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace ttg = triton::gpu;

namespace {

// K extent of one f16/bf16 tcgen05.mma instruction.
constexpr int64_t kInstK = 16;

bool isEligible(TCGen5MMAOp op) {
  if (!op.getIsAsync() || op.getAccDep() || op.getToken())
    return false;
  auto aTy = op.getA().getType();
  auto bTy = op.getB().getType();
  auto dTy = op.getD().getType();
  if (!isa<ttg::SharedEncodingTrait>(aTy.getEncoding()))
    return false;
  if (aTy.getElementTypeBitWidth() != 16 || bTy.getElementTypeBitWidth() != 16)
    return false;
  int64_t k = aTy.getDimSize(1);
  if (k % kInstK != 0)
    return false;
  // Collector reuse is per instruction, so each op must lower to a single
  // M x N instruction per K slice.
  auto tmemEnc = dyn_cast<TensorMemoryEncodingAttr>(dTy.getEncoding());
  if (!tmemEnc)
    return false;
  SmallVector<int64_t> dPerCTA = ttg::getShapePerCTA(dTy);
  return dPerCTA[0] == tmemEnc.getBlockM() && dPerCTA[1] <= tmemEnc.getBlockN();
}

bool sameIssue(TCGen5MMAOp a, TCGen5MMAOp b) {
  return a.getA() == b.getA() && a.getPred() == b.getPred() &&
         a.getTwoCtas() == b.getTwoCtas() &&
         a.getMulticast() == b.getMulticast() &&
         a.getIsUnsigned() == b.getIsUnsigned() &&
         a->getDiscardableAttrDictionary() == b->getDiscardableAttrDictionary();
}

// Slice `v` along `dim` to [off, off + size). A transposed view is sliced
// through its source so the slice stays on the allocated layout.
Value sliceDim(OpBuilder &b, Location loc, Value v, int dim, int64_t off,
               int64_t size) {
  if (auto trans = v.getDefiningOp<ttg::MemDescTransOp>()) {
    ArrayRef<int32_t> order = trans.getOrder();
    Value src = sliceDim(b, loc, trans.getSrc(), order[dim], off, size);
    return ttg::MemDescTransOp::create(b, loc, src, order);
  }
  auto ty = cast<ttg::MemDescType>(v.getType());
  SmallVector<int64_t> shape(ty.getShape());
  SmallVector<int32_t> offsets(shape.size(), 0);
  shape[dim] = size;
  offsets[dim] = off;
  auto sliceTy =
      ttg::MemDescType::get(shape, ty.getElementType(), ty.getEncoding(),
                            ty.getMemorySpace(), ty.getMutableMemory(),
                            ty.getAllocShape());
  return ttg::MemDescSubsliceOp::create(b, loc, sliceTy, v, offsets);
}

// Re-issue `group` (MMAs sharing A) K slice by K slice, with the ops inner, so
// the instructions that read the same A slice are adjacent and can share it
// through the TensorCore collector buffer. Each accumulator still sums its K
// slices in the original order, so results are unchanged. All completion
// barriers move to the last instruction; a commit tracks every prior MMA, so
// they only fire later, never earlier.
void reissue(MutableArrayRef<TCGen5MMAOp> group) {
  TCGen5MMAOp last = group.back();
  OpBuilder b(last);
  Location loc = last.getLoc();
  MLIRContext *ctx = b.getContext();
  int64_t numK = group.front().getA().getType().getDimSize(1) / kInstK;

  SmallVector<Value> barriers, barrierPreds;
  for (TCGen5MMAOp op : group) {
    llvm::append_range(barriers, op.getBarriers());
    llvm::append_range(barrierPreds, op.getBarrierPreds());
  }
  Value trueVal = arith::ConstantIntOp::create(b, loc, 1, 1);

  for (int64_t k = 0; k < numK; ++k) {
    Value aK = sliceDim(b, loc, group.front().getA(), 1, k * kInstK, kInstK);
    for (auto [j, op] : llvm::enumerate(group)) {
      Value bK = sliceDim(b, loc, op.getB(), 0, k * kInstK, kInstK);
      bool isLastInst = k == numK - 1 && j == group.size() - 1;
      auto newOp = TCGen5MMAOp::create(
          b, op.getLoc(), /*token=*/Type(), aK, bK, op.getD(),
          /*acc_dep=*/Value(), k == 0 ? op.getUseD() : trueVal, op.getPred(),
          op.getTwoCtas(), op.getMulticast(),
          isLastInst ? ValueRange(barriers) : ValueRange(),
          isLastInst ? ValueRange(barrierPreds) : ValueRange(),
          /*is_async=*/true, op.getIsUnsigned());
      for (NamedAttribute attr : op->getDiscardableAttrs())
        newOp->setAttr(attr.getName(), attr.getValue());
      StringRef usage = j == 0                    ? "fill"
                        : j == group.size() - 1 ? "lastuse"
                                                : "use";
      newOp->setAttr(kCollectorAAttrName, StringAttr::get(ctx, usage));
    }
  }
  for (TCGen5MMAOp op : group)
    op->erase();
}

void processBlock(Block &block) {
  SmallVector<SmallVector<TCGen5MMAOp>> groups;
  for (auto it = block.begin(); it != block.end(); ++it) {
    auto first = dyn_cast<TCGen5MMAOp>(&*it);
    if (!first || !isEligible(first))
      continue;
    SmallVector<TCGen5MMAOp> group{first};
    auto scan = std::next(it);
    for (; scan != block.end(); ++scan) {
      Operation *op = &*scan;
      if (auto mma = dyn_cast<TCGen5MMAOp>(op)) {
        if (!isEligible(mma) || !sameIssue(first, mma))
          break;
        group.push_back(mma);
        continue;
      }
      // Ops in between are only moved past by the re-issue, so they must not
      // order with the MMAs.
      if (op->hasTrait<OpTrait::IsTerminator>() || !isMemoryEffectFree(op))
        break;
    }
    if (group.size() > 1) {
      it = std::prev(scan);
      groups.push_back(std::move(group));
    }
  }
  for (auto &group : groups)
    reissue(group);
}

} // namespace

class TritonNvidiaGPUMMACollectorReusePass
    : public impl::TritonNvidiaGPUMMACollectorReusePassBase<
          TritonNvidiaGPUMMACollectorReusePass> {
public:
  using TritonNvidiaGPUMMACollectorReusePassBase::
      TritonNvidiaGPUMMACollectorReusePassBase;

  void runOnOperation() override {
    SmallVector<Block *> blocks;
    getOperation().walk([&](Block *block) { blocks.push_back(block); });
    for (Block *block : blocks)
      processBlock(*block);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
