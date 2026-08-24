#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/JSON.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPROMOTELHSTOTMEMPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

enum class OpndAMemType { Unspecified, SMem, TMem };

/// Extract the memory type for opndA from a tt.autows annotation.
static OpndAMemType getOpndAMemType(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("tt.autows");
  if (!attr)
    return OpndAMemType::Unspecified;
  auto parsed = llvm::json::parse(attr.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return OpndAMemType::Unspecified;
  }
  auto *obj = parsed->getAsObject();
  if (!obj)
    return OpndAMemType::Unspecified;
  auto *channelsArr = obj->getArray("channels");
  if (!channelsArr)
    return OpndAMemType::Unspecified;
  for (auto &elem : *channelsArr) {
    auto str = elem.getAsString();
    if (!str)
      continue;
    StringRef channel = *str;
    if (!channel.consume_front("opndA,"))
      continue;
    StringRef memType = channel.take_front(channel.find(','));
    if (memType == "smem")
      return OpndAMemType::SMem;
    if (memType == "tmem")
      return OpndAMemType::TMem;
  }
  return OpndAMemType::Unspecified;
}

template <class MMAOpTy>
Attribute getLHSTMemLayout(MMAOpTy tcGen5MMAOp, gpu::MemDescType lhsTMEMType) {
  int numWarps = ttg::lookupNumWarps(tcGen5MMAOp);
  return nvidia_gpu::getDefaultLayoutForTmemLdSt(lhsTMEMType, numWarps);
}

template <class MMAOpTy> class LHSToTMem : public OpRewritePattern<MMAOpTy> {
public:
  using OpRewritePattern<MMAOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(MMAOpTy tcGen5MMAOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = tcGen5MMAOp->getContext();
    Location loc = tcGen5MMAOp.getLoc();
    auto lhs = tcGen5MMAOp.getA();
    auto localAllocOp = lhs.template getDefiningOp<ttg::LocalAllocOp>();
    if (!localAllocOp)
      return failure();
    // Limit the liverange of the TMem allocations to single block.
    if (localAllocOp->getParentRegion() != tcGen5MMAOp->getParentRegion())
      return failure();
    Value src = localAllocOp.getSrc();
    // Check tt.autows annotation for explicit opndA memory type.
    // If annotated as "smem", skip promotion. If "tmem", promote directly
    // (skip the transposed-shared-source heuristic). If no annotation,
    // fall through to the heuristic.
    const OpndAMemType opndAMem = getOpndAMemType(tcGen5MMAOp);
    if (opndAMem == OpndAMemType::SMem)
      return failure();
    const bool annotatedTmem = opndAMem == OpndAMemType::TMem;

    // If the same source value is also allocated and transposed for use as
    // operand A of another gen5 MMA, skip promotion. The transposed path
    // cannot be promoted to tmem, so keeping both in smem avoids a redundant
    // tmem allocation and copy for the same data. This covers both:
    //   1. Same local_alloc used directly + through memdesc_trans
    //   2. Separate local_allocs from the same src, one transposed
    if (!annotatedTmem) {
      for (Operation *srcUser : src.getUsers()) {
        auto otherAlloc = dyn_cast<ttg::LocalAllocOp>(srcUser);
        if (!otherAlloc)
          continue;
        for (Operation *allocUser : otherAlloc->getResult(0).getUsers()) {
          if (auto transOp = dyn_cast<ttg::MemDescTransOp>(allocUser)) {
            for (Operation *transUser : transOp->getResult(0).getUsers()) {
              if (auto mmaOp = dyn_cast<TCGen5MMAOp>(transUser)) {
                if (mmaOp.getA() == transOp->getResult(0))
                  return failure();
              } else if (auto mmaScaledOp =
                             dyn_cast<TCGen5MMAScaledOp>(transUser)) {
                if (mmaScaledOp.getA() == transOp->getResult(0))
                  return failure();
              }
            }
          }
        }
      }
    }
    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcLayout = srcType.getEncoding();
    auto accTMemEncoding = dyn_cast<TensorMemoryEncodingAttr>(
        tcGen5MMAOp.getD().getType().getEncoding());
    auto cgaLayout = triton::gpu::getCGALayout(srcLayout);
    // TMem encoding for A operand is the same as for D (Acc), with colStride 1,
    // i.e. densely packed in TMEM.
    unsigned elemBitWidth =
        lhs.getType().getElementType().getIntOrFloatBitWidth();
    if (!llvm::is_contained({8, 16, 32}, elemBitWidth)) {
      return failure();
    }
    // Padded fp4 operand cannot be trivially promoted to TMEM.
    if (isFp4Padded(lhs.getType().getEncoding())) {
      return failure();
    }
    const unsigned colStride = 1;
    auto aTMemEncoding = TensorMemoryEncodingAttr::get(
        context, accTMemEncoding.getBlockM(), lhs.getType().getShape()[1],
        colStride, cgaLayout, accTMemEncoding.getTwoCTAs(),
        accTMemEncoding.getCtaMode() == TensorMemoryCTAMode::TwoCTA_RHS
            ? TensorMemoryCTAMode::TwoCTA_LHS
            : accTMemEncoding.getCtaMode());
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    ttg::MemDescType lhsMemDescType = ttg::MemDescType::get(
        lhs.getType().getShape(), lhs.getType().getElementType(), aTMemEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/false);
    bool layoutTmemCompatible =
        isDistributedLayoutTMemCompatible(tcGen5MMAOp, srcType, lhsMemDescType);
    Attribute newLayout = srcLayout;
    if (!layoutTmemCompatible) {
      if (!comesFromLoadOrBlockArg(src) ||
          triton::tools::getBoolEnv("ALLOW_LHS_TMEM_LAYOUT_CONVERSION")) {
        newLayout = getLHSTMemLayout(tcGen5MMAOp, lhsMemDescType);
      } else {
        return failure();
      }
    }
    rewriter.setInsertionPointAfter(localAllocOp);
    if (newLayout != srcLayout) {
      auto ty = cast<RankedTensorType>(src.getType());
      auto newTy = ty.cloneWithEncoding(newLayout);
      src = ttg::ConvertLayoutOp::create(rewriter, loc, newTy, src);
    }
    Value tMemAlloc = TMEMAllocOp::create(rewriter, loc, lhsMemDescType, src);
    tcGen5MMAOp.getAMutable().assign(tMemAlloc);
    return success();
  }
};
} // namespace

class TritonNvidiaGPUPromoteLHSToTMemPass
    : public impl::TritonNvidiaGPUPromoteLHSToTMemPassBase<
          TritonNvidiaGPUPromoteLHSToTMemPass> {
public:
  using TritonNvidiaGPUPromoteLHSToTMemPassBase<
      TritonNvidiaGPUPromoteLHSToTMemPass>::
      TritonNvidiaGPUPromoteLHSToTMemPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<LHSToTMem<TCGen5MMAOp>>(context);
    patterns.add<LHSToTMem<TCGen5MMAScaledOp>>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
