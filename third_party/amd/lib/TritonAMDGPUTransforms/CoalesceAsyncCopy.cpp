#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-coalesce-async-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = triton::gpu;
using mlir::triton::amdgpu::ISAFamily;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCOALESCEASYNCCOPY
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// On gfx9 global and buffer loads directly to shared memory need to write
// coalesced. This pattern converts the layout of the src, mask and other to
// ensure the owned data per thread is contiguous and does not exceed the
// supported load vector size.
//
// Works for both ttg::AsyncCopyGlobalToLocalOp and
// triton::amdgpu::BufferLoadToLocalOp via template specialisation of a few
// op-specific helpers gathered in OpTraits<OpTy>.
//
// Key differences between the two op types:
//   AsyncCopyGlobalToLocalOp  — source is a tensor-of-pointers (srcTy encodes
//     both pointer element type and the distributed layout).  Contiguity is
//     bounded by both the axis-info result AND the consecutive-in-out property
//     of the reg→shared mapping.  canLoadDirectToLDS is called with srcTy
//     directly.  The swizzled branch asserts sizePerThread >= loadContig.
//
//   BufferLoadToLocalOp       — source is a scalar pointer + offset tensor
//     (i32). The true contiguity comes from pointer/offset divisibility and
//     must NOT be additionally capped by the current reg→shared consecutive-
//     in-out (which would clamp it to 1 when sizePerThread=[1]).
//     canLoadDirectToLDS must be called with a reconstructed effective
//     pointer tensor type.  The swizzled branch does not assert srcElemContig.

// ---------------------------------------------------------------------------
// Op-specific traits (specialised below)
// ---------------------------------------------------------------------------
template <typename OpTy>
struct OpTraits;

template <>
struct OpTraits<ttg::AsyncCopyGlobalToLocalOp> {
  // Return the tensor whose encoding drives the distributed layout.
  static Value getSourceTensor(ttg::AsyncCopyGlobalToLocalOp op) {
    return op.getSrc();
  }

  // Return the destination MemDesc value.
  static Value getDestValue(ttg::AsyncCopyGlobalToLocalOp op) {
    return op.getResult();
  }

  // Cap by regToSharedLayout?
  static constexpr bool capByRegToShared = true;

  // Build the effective pointer tensor type for canLoadDirectToLDS.
  static RankedTensorType
  getEffectivePtrType(ttg::AsyncCopyGlobalToLocalOp op,
                      RankedTensorType srcTy) {
    return srcTy; // srcTy is already a tensor-of-pointers
  }

  // Assert srcElemContig >= loadContig in the swizzled branch?
  static constexpr bool assertSrcElemContig = true;

  // Assign the (possibly converted) source operand back to the op.
  static void assignSource(ttg::AsyncCopyGlobalToLocalOp op,
                           PatternRewriter &rewriter, Value newSrc) {
    op.getSrcMutable().assign(newSrc);
  }
};

template <>
struct OpTraits<triton::amdgpu::BufferLoadToLocalOp> {
  // The "source tensor" whose encoding we manipulate is the offsets tensor.
  static Value getSourceTensor(triton::amdgpu::BufferLoadToLocalOp op) {
    return op.getOffsets();
  }

  // Return the destination MemDesc value.
  static Value getDestValue(triton::amdgpu::BufferLoadToLocalOp op) {
    return op.getDest();
  }

  static constexpr bool capByRegToShared = false;

  // Build the effective pointer tensor type for canLoadDirectToLDS.
  // BufferLoadToLocalOp uses a scalar ptr + offset tensor, so we reconstruct
  // the pointer tensor type the same way the lowering does.
  static RankedTensorType
  getEffectivePtrType(triton::amdgpu::BufferLoadToLocalOp op,
                      RankedTensorType /*offsetTy*/) {
    return cast<RankedTensorType>(
        LLVM::AMD::getPointerTypeWithShape(op.getPtr(), op.getOffsets()));
  }

  // Do NOT assert srcElemContig >= loadContig: the offsets tensor may have
  // sizePerThread=1 even when the true data contiguity is larger.
  static constexpr bool assertSrcElemContig = false;

  static void assignSource(triton::amdgpu::BufferLoadToLocalOp op,
                           PatternRewriter &rewriter, Value newOffsets) {
    op.getOffsetsMutable().assign(newOffsets);
  }
};

// ---------------------------------------------------------------------------
// Combined rewrite pattern
// ---------------------------------------------------------------------------
template <typename OpTy>
struct CoalesceAsyncCopyToLocal : public OpRewritePattern<OpTy> {
  using Traits = OpTraits<OpTy>;

  CoalesceAsyncCopyToLocal(const triton::AMD::TargetInfo &targetInfo,
                         const DenseMap<OpTy, unsigned> &contiguityMap,
                         MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx), targetInfo{targetInfo},
        contiguityMap{contiguityMap} {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value srcTensor = Traits::getSourceTensor(op);
    Value dst = Traits::getDestValue(op);
    Value mask = op.getMask();
    Value other = op.getOther();

    auto srcTy = cast<RankedTensorType>(srcTensor.getType());
    auto dstTy = cast<ttg::MemDescType>(dst.getType());

    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(op, "src encoding must be #blocked");

    if (!isa<ttg::SwizzledSharedEncodingAttr, ttg::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      return rewriter.notifyMatchFailure(
          op, "dst encoding must be #swizzled or #padded");
    }

    // Retrieve the precomputed contiguity.
    auto it = contiguityMap.find(op);
    if (it == contiguityMap.end())
      return op->emitError() << "No contiguity information about the copy op";
    unsigned loadContig = it->second;
    assert(loadContig > 0);

    // Build the reg→shared layout mapping (used for cap and padded branch).
    LinearLayout regLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout;
    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(dstTy.getEncoding());
    if (paddedEnc) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
    auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);

    // AsyncCopyGlobalToLocalOp caps loadContig by the reg→shared consecutive-
    // in-out, which bounds the real global-memory contiguity through the
    // tensor-of-pointers encoding. BufferLoadToLocalOp must NOT apply this cap
    // because the offsets tensor sizePerThread may be 1 even when the true
    // contiguity is larger; we rely solely on ptr/offset divisibility.
    if (Traits::capByRegToShared)
      loadContig = std::min<unsigned>(
          loadContig, regToSharedLayout.getNumConsecutiveInOut());

    // Select the largest supported load width ≤ loadContig.
    auto elemBitWidth = dstTy.getElementTypeBitWidth();
    loadContig =
        fitToValidDirectToLdsVecSize(loadContig, elemBitWidth, targetInfo);
    if (loadContig == 0)
      return rewriter.notifyMatchFailure(
          op, "could not find layout config to create coalesced writes");

    auto mod = op->template getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::lookupNumWarps(op);
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

    // Check whether the op already writes coalesced at the target contiguity.
    RankedTensorType effectivePtrTy =
        Traits::getEffectivePtrType(op, srcTy);
    if (LLVM::AMD::canLoadDirectToLDS(targetInfo, effectivePtrTy,
                                      dstTy.getEncoding(),
                                      dstTy.getAllocShape(), loadContig))
      return rewriter.notifyMatchFailure(op, "already writes coalesced");

    if (!targetInfo.supportsDirectToLdsLoadBitWidth(loadContig * elemBitWidth))
      return rewriter.notifyMatchFailure(op,
                                        "unable to find supported vector size "
                                        "based on src and dst encodings");

    ttg::DistributedEncodingTrait newDistEnc;

    if (isa<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding())) {
      // For swizzled layouts the swizzling is applied during lowering, so we
      // only adjust sizePerThread to avoid strided writes into LDS.
      auto contigPerThread = ttg::getContigPerThread(srcTy);
      if (Traits::assertSrcElemContig) {
        auto srcElemContig = contigPerThread[blockedEnc.getOrder()[0]];
        assert(srcElemContig >= loadContig);
      }
      contigPerThread[blockedEnc.getOrder()[0]] = loadContig;
      newDistEnc = BlockedEncodingAttr::get(
          op.getContext(), srcTy.getShape(), contigPerThread,
          blockedEnc.getOrder(), numWarps, threadsPerWarp,
          blockedEnc.getCGALayout());
    } else if (paddedEnc) {
      // For padded layouts we build a linear encoding whose reg/lane/warp bases
      // are taken from the shared layout's offset bases so that each warp maps
      // to consecutive LDS offsets. Steps:
      //   1) log2(loadContig)    bases → register (contiguous elements/load)
      //   2) log2(threadsPerWarp) bases → lane    (warp-coalesced writes)
      //   3) log2(numWarps)       bases → warp    (pad with broadcast if needed)
      //   4) remaining bases → additional register bases

      auto *ctx = srcTy.getContext();
      StringAttr kOffset = StringAttr::get(ctx, "offset");
      auto rank = srcTy.getRank();
      auto offsetBases = sharedLayout.getBases().lookup(kOffset);

      int log2LoadContig = llvm::Log2_32(loadContig);
      int log2ThreadsPerWarp = llvm::Log2_32(threadsPerWarp);
      int log2NumWarps = llvm::Log2_32(numWarps);

      if ((int)offsetBases.size() < log2LoadContig + log2ThreadsPerWarp)
        return rewriter.notifyMatchFailure(
            op, "dst shape is too small. We require at least loadContig * "
                "threadsPerWarp elements");

      auto remainingBases = ArrayRef(offsetBases);
      auto takeN = [&remainingBases](size_t n) {
        auto take = std::min(remainingBases.size(), n);
        auto v = remainingBases.take_front(take).vec();
        remainingBases = remainingBases.drop_front(take);
        return v;
      };

      auto regBases = takeN(log2LoadContig);
      auto laneBases = takeN(log2ThreadsPerWarp);
      auto warpBases = takeN(log2NumWarps);
      warpBases.resize(log2NumWarps, std::vector<int32_t>(rank, 0));
      append_range(regBases, remainingBases);

      triton::LinearLayout newRegLayout(
          {
              {StringAttr::get(ctx, "register"), regBases},
              {StringAttr::get(ctx, "lane"), laneBases},
              {StringAttr::get(ctx, "warp"), warpBases},
          },
          triton::standardOutDimNames(ctx, rank));

      newRegLayout = triton::gpu::combineCtaCgaWithShape(
          newRegLayout, blockedEnc.getCGALayout(), srcTy.getShape());

      auto newRegToShared = newRegLayout.invertAndCompose(sharedLayout);
      if (newRegToShared.getNumConsecutiveInOut() < loadContig)
        return rewriter.notifyMatchFailure(
            op, "could not coalesce global addresses based on the linear "
                "component of the padded encoding");

      newDistEnc = ttg::LinearEncodingAttr::get(ctx, std::move(newRegLayout));
    } else {
      assert(false && "Unsupported layout");
    }

    if (newDistEnc == srcTy.getEncoding())
      return rewriter.notifyMatchFailure(
          op, "Unable to find a new src layout to coalesce writes to LDS");

    // Convert the source tensor (and optionally mask/other) to the new encoding.
    auto convertLayout = [&rewriter](auto loc, Value old, auto newEnc) {
      auto oldTy = cast<RankedTensorType>(old.getType());
      RankedTensorType newTy = oldTy.cloneWithEncoding(newEnc);
      return ttg::ConvertLayoutOp::create(rewriter, loc, newTy, old);
    };

    auto loc = op->getLoc();
    Value newSrc = convertLayout(loc, srcTensor, newDistEnc);
    if (mask)
      mask = convertLayout(loc, mask, newDistEnc);
    if (other)
      other = convertLayout(loc, other, newDistEnc);

    rewriter.modifyOpInPlace(op, [&]() {
      Traits::assignSource(op, rewriter, newSrc);
      if (mask)
        op.getMaskMutable().assign(mask);
      if (other)
        op.getOtherMutable().assign(other);
      op.setContiguity(loadContig);
    });
    return success();
  }

private:
  const triton::AMD::TargetInfo &targetInfo;
  const DenseMap<OpTy, unsigned> &contiguityMap;
};

// Convenience aliases.
using CoalesceAsyncCopyWrites =
    CoalesceAsyncCopyToLocal<ttg::AsyncCopyGlobalToLocalOp>;
using CoalesceBufferLoadToLocal =
    CoalesceAsyncCopyToLocal<triton::amdgpu::BufferLoadToLocalOp>;

} // anonymous namespace

class TritonAMDGPUCoalesceAsyncCopyPass
    : public impl::TritonAMDGPUCoalesceAsyncCopyBase<
          TritonAMDGPUCoalesceAsyncCopyPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    triton::AMD::TargetInfo targetInfo(gfxArch);

    mlir::RewritePatternSet patterns(context);

    if (!llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                            targetInfo.getISAFamily()))
      return; // This pass is CDNA3 and CDNA4 specific.

    // Precompute the contiguity of all async-copy ops before any IR changes to
    // avoid rebuilding ModuleAxisInfoAnalysis on every pattern application.
    AMD::ModuleAxisInfoAnalysis axisAnalysis(m);
    DenseMap<ttg::AsyncCopyGlobalToLocalOp, unsigned> asyncCopyContiguity;
    DenseMap<triton::amdgpu::BufferLoadToLocalOp, unsigned>
        bufferOpsToLocalContiguity;

    m->walk([&](Operation *op) {
      if (auto globalCopyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
        unsigned contiguity =
            mlir::LLVM::AMD::getContiguity(globalCopyOp.getSrc(), axisAnalysis);
        if (auto mask = globalCopyOp.getMask())
          contiguity = std::min<unsigned>(
              contiguity, axisAnalysis.getMaskAlignment(mask));
        asyncCopyContiguity.insert({globalCopyOp, contiguity});
      }

      if (auto bufferCopyOp =
              dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(op)) {
        Value ptr = bufferCopyOp.getPtr();
        Value offsets = bufferCopyOp.getOffsets();

        unsigned elemBitWidth = triton::getPointeeBitWidth(ptr.getType());
        unsigned elemNumBytes = std::max(elemBitWidth / 8, 1u);

        // Alignment from the scalar base pointer divisibility.
        unsigned ptrAlign = 1;
        if (auto *ptrInfo = axisAnalysis.getAxisInfo(ptr)) {
          unsigned ptrDivisibility = ptrInfo->getDivisibility(0);
          ptrAlign = std::max(ptrDivisibility / elemNumBytes, 1u);
        }

        // Alignment from the offset tensor's innermost (fast-varying) dimension,
        // derived from axis-info divisibility — NOT capped by sizePerThread.
        unsigned offsetAlign = 1;
        if (auto *offsetInfo = axisAnalysis.getAxisInfo(offsets)) {
          auto contiguityVec = offsetInfo->getContiguity();
          SmallVector<unsigned> offsetOrder =
              getOrderFromContiguity(contiguityVec);
          unsigned innerDim = offsetOrder[0];
          unsigned divisibility = offsetInfo->getDivisibility(innerDim);
          offsetAlign = std::max(divisibility / elemNumBytes, 1u);
        }

        // Cap to the widest vectorised load (128 bits).
        unsigned maxVec = 128 / elemBitWidth;
        unsigned contiguity = std::min({ptrAlign, offsetAlign, maxVec});

        // NOTE: We intentionally do NOT cap by getMaskAlignment(mask). The
        // mask for buffer_load_to_local may have a different (smaller) shape
        // than the offsets tensor (e.g., mask_n[:, None] is [BLOCK_N, 1] while
        // offsets is [BLOCK_N, BLOCK_D_Q]). getMaskAlignment would return 1
        // for such a 1-wide mask, incorrectly capping contiguity along the
        // fast (column) dimension.
        bufferOpsToLocalContiguity.insert({bufferCopyOp, contiguity});
      }
    });

    patterns.add<CoalesceAsyncCopyWrites>(targetInfo, asyncCopyContiguity,
                                          context);
    patterns.add<CoalesceBufferLoadToLocal>(
        targetInfo, bufferOpsToLocalContiguity, context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
