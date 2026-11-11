#include <algorithm>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUCOALESCEASYNCCOPY
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static Value convertValueLayout(Value src, Attribute enc,
                                PatternRewriter &rewriter) {
  auto ty = cast<RankedTensorType>(src.getType());
  auto newTy = ty.cloneWithEncoding(enc);
  auto cvt = ConvertLayoutOp::create(rewriter, src.getLoc(), newTy, src);
  return cvt.getResult();
}

static void retargetCopyOperandsToEncoding(
    AsyncCopyGlobalToLocalOp copyOp, Attribute newEncoding,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternRewriter &rewriter) {
  Value src = copyOp.getSrc();
  Value mask = copyOp.getMask();
  Value other = copyOp.getOther();

  // insert cvt's after src, mask, and other
  src = convertValueLayout(src, newEncoding, rewriter);
  if (mask)
    mask = convertValueLayout(mask, newEncoding, rewriter);
  if (other)
    other = convertValueLayout(other, newEncoding, rewriter);

  unsigned contiguity = axisInfoAnalysis.getContiguity(src);
  if (mask)
    contiguity =
        std::min<unsigned>(contiguity, axisInfoAnalysis.getMaskAlignment(mask));

  rewriter.modifyOpInPlace(copyOp, [&]() {
    copyOp.getSrcMutable().assign(src);
    if (mask)
      copyOp.getMaskMutable().assign(mask);
    if (other)
      copyOp.getOtherMutable().assign(other);
    copyOp.setContiguity(contiguity);
  });
}

// This pass currently applies in two cases.
//
// Case A: all of the following are true:
//   1) Operand A for WGMMA is to be loaded in registers
//   2) We upcast operand A in registers before the WGMMA
//      (downcasting is not yet supported)
//   3) Pipelining is enabled for loading A
//
// In this case, the AsyncCopyGlobalToLocal op's SharedEncoding vec is smaller
// than the BlockedEncoding sizePerThread for the k dimension. For example, when
// upcasting from int8 to bf16, the shared vec is 8 and sizePerThread for k is
// 16. AsyncCopyGlobalToLocal then generates two 8-byte cp.async operations for
// each contiguous 16B global-data chunk owned by a thread, which breaks
// coalescing and doubles the minimum required transactions.
//
// This issue occurs for cp.async because it combines load and store into one
// instruction. Additionally, cp.async only supports copy sizes up to 16B. The
// fix is to clip each dim of sizePerThread by shared vec and the cp.async copy
// size limit, so that the vectorization of load and store are equal along the
// contiguous dimension. In the above example, each thread will then only own 8B
// contiguous global data.
//
// Case B: user hints or analysis can infer alignment that allows 256-bit
// vectorized load/store operations for regular instructions. In that case,
// sizePerThread may exceed the 128-bit cp.async hardware limit and break
// coalescing. Again, the fix is to cap sizePerThread.
struct ClipAsyncCopySizePerThread
    : public OpRewritePattern<AsyncCopyGlobalToLocalOp> {
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
  using OpRewritePattern::OpRewritePattern;
  ClipAsyncCopySizePerThread(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                             MLIRContext *context)
      : OpRewritePattern(context), axisInfoAnalysis(axisInfoAnalysis) {}

  LogicalResult matchAndRewrite(AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    // Bulk copies use a single instruction; coalescing is not applicable.
    if (copyOp.getUseBulk())
      return rewriter.notifyMatchFailure(copyOp, "bulk copy, skip");
    Value src = copyOp.getSrc();
    Value mask = copyOp.getMask();
    Value other = copyOp.getOther();
    auto srcTy = dyn_cast<RankedTensorType>(src.getType());
    if (!srcTy)
      return failure();
    auto dstTy = cast<MemDescType>(copyOp.getResult().getType());
    auto blockedEnc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(copyOp,
                                         "src must be of blocked encoding");
    auto sharedEnc = dyn_cast<SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    if (!sharedEnc)
      return failure();
    auto sharedVec = sharedEnc.getVec();

    // obtain max contiguous copy size
    // Note this can be further optimized, as copyContigSize can be even
    // smaller when lowering, depending on contiguity and mask alignment
    // (see AsyncCopyGlobalToLocalOpConversion)
    LinearLayout regLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout = triton::gpu::toLinearLayout(dstTy);
    unsigned copyContigSize =
        regLayout.invertAndCompose(sharedLayout).getNumConsecutiveInOut();
    unsigned cpAsyncContigSize =
        std::max(1u, 128u / dstTy.getElementTypeBitWidth());
    copyContigSize = std::min<unsigned>(copyContigSize, cpAsyncContigSize);

    // obtain block sizePerThread along contig dim
    auto contigPerThread = getContigPerThread(srcTy);
    auto blockContigSize = contigPerThread[blockedEnc.getOrder()[0]];

    if (blockContigSize <= copyContigSize)
      return rewriter.notifyMatchFailure(
          copyOp,
          "blocked sizePerThread along contiguous dim must be greater than the "
          "max contiguous copy size ");

    contigPerThread[blockedEnc.getOrder()[0]] = copyContigSize;

    // obtain new blockedEnc based on clipped sizePerThread
    auto mod = copyOp->getParentOfType<ModuleOp>();
    int numWarps = lookupNumWarps(copyOp);
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    auto newBlockEnc = BlockedEncodingAttr::get(
        copyOp.getContext(), srcTy.getShape(), contigPerThread,
        blockedEnc.getOrder(), numWarps, threadsPerWarp,
        blockedEnc.getCGALayout());

    retargetCopyOperandsToEncoding(copyOp, newBlockEnc, axisInfoAnalysis,
                                   rewriter);

    return success();
  }
};

// Assume the expensive copies are already coalesced.
// Skip dtype smaller than 32 bits to avoid problems with contiguity.
static bool isCheapAsyncCopy(AsyncCopyGlobalToLocalOp copyOp) {
  auto srcTy = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
  if (!srcTy)
    return false;
  auto dstTy = cast<MemDescType>(copyOp.getResult().getType());
  int numWarps = lookupNumWarps(copyOp);
  auto mod = copyOp->getParentOfType<ModuleOp>();
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
  return srcTy.getNumElements() < numWarps * threadsPerWarp &&
         dstTy.getElementTypeBitWidth() >= 32;
}

// For cheap loads we usually pick the layout based on users but when converting
// to async_cp the layout of the copy is independent of the layout of the users
// so picking a coalesced layout is better.
struct CoalesceCheapAsyncCopyGlobalToLocal
    : public OpRewritePattern<AsyncCopyGlobalToLocalOp> {
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
  DenseMap<AsyncCopyGlobalToLocalOp, Attribute> &coalescedAsyncCopyMap;
  using OpRewritePattern::OpRewritePattern;
  CoalesceCheapAsyncCopyGlobalToLocal(
      ModuleAxisInfoAnalysis &axisInfoAnalysis,
      DenseMap<AsyncCopyGlobalToLocalOp, Attribute> &coalescedAsyncCopyMap,
      MLIRContext *context)
      : OpRewritePattern(context), axisInfoAnalysis(axisInfoAnalysis),
        coalescedAsyncCopyMap(coalescedAsyncCopyMap) {}

  LogicalResult matchAndRewrite(AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value src = copyOp.getSrc();
    Value mask = copyOp.getMask();
    Value other = copyOp.getOther();
    auto srcTy = dyn_cast<RankedTensorType>(src.getType());
    if (!srcTy)
      return failure();
    auto dstTy = cast<MemDescType>(copyOp.getResult().getType());
    int numCTAs = triton::gpu::getNumCTAs(dstTy.getEncoding());
    if (!isCheapAsyncCopy(copyOp))
      return failure();
    auto shapePerCTA = triton::gpu::getShapePerCTA(dstTy);
    auto cgaLayout = triton::gpu::getCGALayout(dstTy.getEncoding());

    auto newEnc = coalescedAsyncCopyMap[copyOp];
    if (newEnc == nullptr || newEnc == srcTy.getEncoding())
      return failure();

    retargetCopyOperandsToEncoding(copyOp, newEnc, axisInfoAnalysis, rewriter);

    return success();
  }
};

// Whether every program reads the same addresses through `ptr`: nothing that
// computes it reads tt.get_program_id. Follows scf.for loop-carried values;
// any other region result or block argument counts as varying.
static bool isProgramInvariant(Value ptr) {
  SmallVector<Value> worklist{ptr};
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      Operation *parent = arg.getOwner()->getParentOp();
      if (isa<FunctionOpInterface>(parent))
        continue;
      auto forOp = dyn_cast<scf::ForOp>(parent);
      if (!forOp)
        return false;
      if (arg == forOp.getInductionVar()) {
        worklist.append(
            {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()});
        continue;
      }
      worklist.push_back(forOp.getTiedLoopInit(arg)->get());
      worklist.push_back(forOp.getTiedLoopYieldedValue(arg)->get());
      continue;
    }
    Operation *def = v.getDefiningOp();
    if (isa<triton::GetProgramIdOp>(def))
      return false;
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      unsigned i = cast<OpResult>(v).getResultNumber();
      worklist.push_back(forOp.getInitArgs()[i]);
      worklist.push_back(forOp.getYieldedValues()[i]);
      continue;
    }
    if (def->getNumRegions() != 0)
      return false;
    worklist.append(def->operand_begin(), def->operand_end());
  }
  return true;
}

// A mixed-radix load keeps the vector that adds no registers (Coalesce), but
// its async copy predicates off duplicate threads, so it takes the wider
// vector AxisInfo proves.
struct WidenMixedRadixAsyncCopy
    : public OpRewritePattern<AsyncCopyGlobalToLocalOp> {
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
  DenseMap<AsyncCopyGlobalToLocalOp, Attribute> &widenedMap;
  DenseSet<AsyncCopyGlobalToLocalOp> &keepL1;
  WidenMixedRadixAsyncCopy(
      ModuleAxisInfoAnalysis &axisInfoAnalysis,
      DenseMap<AsyncCopyGlobalToLocalOp, Attribute> &widenedMap,
      DenseSet<AsyncCopyGlobalToLocalOp> &keepL1, MLIRContext *context)
      : OpRewritePattern(context), axisInfoAnalysis(axisInfoAnalysis),
        widenedMap(widenedMap), keepL1(keepL1) {}

  LogicalResult matchAndRewrite(AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    Attribute newEnc = widenedMap.lookup(copyOp);
    if (!newEnc ||
        newEnc ==
            cast<RankedTensorType>(copyOp.getSrc().getType()).getEncoding())
      return failure();
    retargetCopyOperandsToEncoding(copyOp, newEnc, axisInfoAnalysis, rewriter);
    // A copy under 16 bytes lowers to cp.async.ca (L1); at 16 bytes it lowers
    // to .cg (L2 only). Programs sharing an SM hit L1 on a copy every program
    // makes from the same addresses, so that copy keeps .ca.
    auto policy =
        dyn_cast_or_null<triton::CachePolicyAttr>(copyOp.getCachePolicyAttr());
    if (keepL1.contains(copyOp) &&
        (!copyOp.getCachePolicyAttr() ||
         (policy && policy.getCacheModifier() == triton::CacheModifier::NONE)))
      rewriter.modifyOpInPlace(copyOp, [&]() {
        copyOp.setCachePolicyAttr(triton::CachePolicyAttr::get(
            getContext(), triton::CacheModifier::CA,
            policy ? policy.getEvictionPolicy()
                   : triton::EvictionPolicy::NORMAL));
      });
    return success();
  }
};

static FailureOr<LinearEncodingAttr>
getWidenedMixedRadixEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                             AsyncCopyGlobalToLocalOp copyOp, int numWarps,
                             int threadsPerWarp, bool &underSixteenBytes) {
  auto type = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
  if (!type || isRadix2Encoding(type.getEncoding()))
    return failure();
  MLIRContext *ctx = copyOp.getContext();
  AxisInfo &info = *axisInfoAnalysis.getAxisInfo(copyOp.getSrc());
  SmallVector<unsigned> order = getOrderFromContiguity(info.getContiguity());
  unsigned vector = getMixedRadixVector(
      ctx, type.getShape(),
      getAxisInfoVector(info, type, type.getShape(), order[0],
                        numWarps * threadsPerWarp),
      order, numWarps, threadsPerWarp, /*predicated=*/true);
  // A layout already as wide (e.g. a scale tile's TMEM store layout) stays.
  unsigned contig = getContigPerThread(type)[order[0]];
  if (vector <= contig)
    return failure();
  unsigned bits = cast<triton::PointerType>(type.getElementType())
                      .getPointeeType()
                      .getIntOrFloatBitWidth();
  underSixteenBytes = contig * bits < 128;
  return getMixedRadixEncoding(ctx, type.getShape(), numWarps, threadsPerWarp,
                               /*numCTAs=*/1, vector, order);
}

struct CoalesceAsyncCopyPass
    : impl::TritonGPUCoalesceAsyncCopyBase<CoalesceAsyncCopyPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    triton::ModuleAxisInfoAnalysis axisInfoAnalysis(m);
    // Collect the coalesced encoding first as changing the IR invalidates the
    // axis analysis.
    DenseMap<AsyncCopyGlobalToLocalOp, Attribute> coalescedAsyncCopyMap;
    DenseMap<AsyncCopyGlobalToLocalOp, Attribute> widenedMap;
    DenseSet<AsyncCopyGlobalToLocalOp> keepL1;
    m.walk([&](AsyncCopyGlobalToLocalOp copyOp) {
      if (copyOp.getUseBulk())
        return;
      bool underSixteenBytes = false;
      FailureOr<LinearEncodingAttr> widened = getWidenedMixedRadixEncoding(
          axisInfoAnalysis, copyOp, triton::gpu::lookupNumWarps(copyOp),
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(m),
          underSixteenBytes);
      if (succeeded(widened)) {
        widenedMap[copyOp] = *widened;
        if (underSixteenBytes && isProgramInvariant(copyOp.getSrc()))
          keepL1.insert(copyOp);
        return;
      }
      // Only cheap copies read the map. An NPOT copy with 3 elements per
      // thread has no power-of-two blocked layout, so don't build one.
      if (!isCheapAsyncCopy(copyOp))
        return;
      auto dstTy = cast<MemDescType>(copyOp.getResult().getType());
      int numWarps = triton::gpu::lookupNumWarps(copyOp);
      int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(m);
      int numCTAs = triton::gpu::getNumCTAs(dstTy.getEncoding());
      auto cgaLayout = triton::gpu::getCGALayout(dstTy.getEncoding());
      auto shapePerCTA = triton::gpu::getShapePerCTA(dstTy);
      coalescedAsyncCopyMap[copyOp] =
          buildCoalescedEncoding(axisInfoAnalysis, copyOp, numWarps,
                                 threadsPerWarp, cgaLayout, shapePerCTA);
    });

    MLIRContext *context = &getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<CoalesceCheapAsyncCopyGlobalToLocal>(
        axisInfoAnalysis, coalescedAsyncCopyMap, context);
    patterns.add<ClipAsyncCopySizePerThread>(axisInfoAnalysis, context);
    patterns.add<WidenMixedRadixAsyncCopy>(axisInfoAnalysis, widenedMap, keepL1,
                                           context);

    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
