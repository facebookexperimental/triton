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
// ensure the owned data per thread is contiguous and does no exceed the
// supported load vector size.
struct CoalesceAsyncCopyWrites
    : public OpRewritePattern<ttg::AsyncCopyGlobalToLocalOp> {
  CoalesceAsyncCopyWrites(const triton::AMD::TargetInfo &targetInfo,
                          const DenseMap<ttg::AsyncCopyGlobalToLocalOp,
                                         unsigned> &asyncCopyContiguity,
                          MLIRContext *ctx)
      : OpRewritePattern(ctx), targetInfo{targetInfo},
        asyncCopyContiguity{std::move(asyncCopyContiguity)} {}

  LogicalResult matchAndRewrite(ttg::AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto src = copyOp.getSrc();
    auto dst = copyOp.getResult();
    Value mask = copyOp.getMask();
    Value other = copyOp.getOther();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<ttg::MemDescType>(dst.getType());

    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(copyOp,
                                         "src encoding must be #blocked");

    if (!isa<ttg::SwizzledSharedEncodingAttr, ttg::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      return rewriter.notifyMatchFailure(
          copyOp, "dst encoding must be #swizzled or #padded");
    }

    // We start from the precomputed contiguity we got from AxisAnalysis.
    unsigned loadContig = 0;
    if (auto it = asyncCopyContiguity.find(copyOp);
        it != asyncCopyContiguity.end())
      loadContig = it->second;
    else
      return copyOp->emitError()
             << "No contiguity information about the copy op";
    assert(loadContig > 0);

    // Further restrict the contiguity based on the contiguity of the src to dst
    // layout e.g. if the order of the blocked and shared encoding is different
    // we can only load one element at a time or if the shared encoding is
    // swizzled we cannot exceed the vector size of the swizzling pattern
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
    loadContig = std::min<unsigned>(loadContig,
                                    regToSharedLayout.getNumConsecutiveInOut());

    // Select the largest supported load width equal or smaller than loadContig
    auto elemBitWidth = dstTy.getElementTypeBitWidth();
    loadContig =
        fitToValidDirectToLdsVecSize(loadContig, elemBitWidth, targetInfo);

    if (loadContig == 0) {
      return rewriter.notifyMatchFailure(
          copyOp, "could not find layout config to create coalesced writes");
    }

    // Do not rewrite if we already use the correct contiguity (could be from a
    // previous rewrite)
    auto mod = copyOp->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::lookupNumWarps(copyOp);
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

    ttg::DistributedEncodingTrait newDistEnc;

    if (LLVM::AMD::canLoadDirectToLDS(targetInfo, srcTy, dstTy.getEncoding(),
                                      dstTy.getAllocShape(), loadContig)) {
      return rewriter.notifyMatchFailure(copyOp, "already writes coalesced");
    }
    // Check if we support load contig because canLoadDirectToLds can change it
    if (!targetInfo.supportsDirectToLdsLoadBitWidth(loadContig * elemBitWidth))
      return rewriter.notifyMatchFailure(copyOp,
                                         "unable to find supported vector size "
                                         "based on src and dst encodings");

    if (isa<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding())) {
      // For swizzled layouts we apply the swizzling during lowering so we only
      // adjust the sizePerThread of the blocked encoding to avoid strided
      // writes into LDS
      auto contigPerThread = ttg::getContigPerThread(srcTy);
      auto srcElemContig = contigPerThread[blockedEnc.getOrder()[0]];
      assert(srcElemContig >= loadContig);
      contigPerThread[blockedEnc.getOrder()[0]] = loadContig;
      newDistEnc = BlockedEncodingAttr::get(
          copyOp.getContext(), srcTy.getShape(), contigPerThread,
          blockedEnc.getOrder(), numWarps, threadsPerWarp,
          blockedEnc.getCGALayout());
    } else if (paddedEnc) {
      // For padded layouts the linear_component maps from LDS offsets to n-D
      // tensor indices. This mapping might reorder elements resulting in
      // scattered writes into LDS which is not supported on GFX9. To ensure
      // coalesced writes we change the src layout to a linear encoding which
      // effectivly copies/mimicks the linear_component so each warp (reg+lane
      // bases) map to consecutive LDS offsets resulting in coalesced writes
      // The new linear encoding is build by taking bases from the
      // linear_component and assigning them to reg/lane/warp bases in the
      // following steps:
      // 1) Take log2(loadContig) bases as reg bases to ensure our registers per
      // load instruction point to contiguous elements in LDS.
      // 2) Take log2(threadsPerWarp) as lane bases to ensure lanes write
      // contiguous into LDS.
      // 3) Take log2(numWarps) as warp bases or add braodcasting bases if we
      // run out of bases
      // 4) Take any remaining bases as additional reg bases

      auto *ctx = srcTy.getContext();
      StringAttr kOffset = StringAttr::get(ctx, "offset");

      auto rank = srcTy.getRank();

      auto offsetBases = sharedLayout.getBases().lookup(kOffset);

      int log2LoadContig = llvm::Log2_32(loadContig);
      int log2ThreadsPerWarp = llvm::Log2_32(threadsPerWarp);
      int log2NumWarps = llvm::Log2_32(numWarps);

      if (offsetBases.size() < log2LoadContig + log2ThreadsPerWarp) {
        return rewriter.notifyMatchFailure(
            copyOp, "dst shape is too small. We require at least loadContig * "
                    "threadsPerWarp elements");
      }

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
      if (newRegToShared.getNumConsecutiveInOut() < loadContig) {
        return rewriter.notifyMatchFailure(
            copyOp, "could not coalesce global addresses based on the linear "
                    "component of the padded encoding");
      }

      newDistEnc = ttg::LinearEncodingAttr::get(ctx, std::move(newRegLayout));
    } else {
      assert(false && "Unsupported layout");
    }

    if (newDistEnc == srcTy.getEncoding()) {
      return rewriter.notifyMatchFailure(
          copyOp, "Unable to find a new src layout to coalesce writes to LDS");
    }

    // Convert layout of src, mask and other to new encoding
    auto convertLayout = [&rewriter](auto loc, Value old, auto newEnc) {
      auto oldTy = cast<RankedTensorType>(old.getType());
      RankedTensorType newSrcTy = oldTy.cloneWithEncoding(newEnc);
      return ttg::ConvertLayoutOp::create(rewriter, loc, newSrcTy, old);
    };

    auto loc = copyOp->getLoc();
    Value cvtSrc = convertLayout(loc, src, newDistEnc);

    if (mask)
      mask = convertLayout(loc, mask, newDistEnc);
    if (other)
      other = convertLayout(loc, other, newDistEnc);

    rewriter.modifyOpInPlace(copyOp, [&]() {
      copyOp.getSrcMutable().assign(cvtSrc);
      if (mask)
        copyOp.getMaskMutable().assign(mask);
      if (other)
        copyOp.getOtherMutable().assign(other);
      copyOp.setContiguity(loadContig);
    });
    return success();
  }

private:
  const triton::AMD::TargetInfo &targetInfo;
  const DenseMap<ttg::AsyncCopyGlobalToLocalOp, unsigned> &asyncCopyContiguity;
};

struct CoalesceBufferLoadToLocal
    : public OpRewritePattern<triton::amdgpu::BufferLoadToLocalOp> {
  CoalesceBufferLoadToLocal(const triton::AMD::TargetInfo &targetInfo,
                          const DenseMap<triton::amdgpu::BufferLoadToLocalOp,
                                         unsigned> &bufferLoadToLocalContiguity,
                          MLIRContext *ctx)
      : OpRewritePattern(ctx), targetInfo{targetInfo},
        bufferLoadToLocalContiguity{std::move(bufferLoadToLocalContiguity)} {}

  LogicalResult matchAndRewrite(triton::amdgpu::BufferLoadToLocalOp bufOp,
                                PatternRewriter &rewriter) const override {
    auto ptr = bufOp.getPtr();
    // for buffer ops, ptr input is a scalar pointer, offsets have the same layout
    // as the result
    auto offsets = bufOp.getOffsets();
    auto dst = bufOp.getDest();
    Value mask = bufOp.getMask();
    Value other = bufOp.getOther();

    auto srcTy = cast<RankedTensorType>(offsets.getType());
    auto dstTy = cast<ttg::MemDescType>(dst.getType());
    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(bufOp,
                                         "src encoding must be #blocked");

    if (!isa<ttg::SwizzledSharedEncodingAttr, ttg::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      return rewriter.notifyMatchFailure(
          bufOp, "dst encoding must be #swizzled or #padded");
    }

    // We start from the precomputed contiguity we got from AxisAnalysis.
    unsigned loadContig = 0;
    if (auto it = bufferLoadToLocalContiguity.find(bufOp);
        it != bufferLoadToLocalContiguity.end()) {
      loadContig = it->second;
    } else {
      return bufOp->emitError()
             << "No contiguity information about the copy op";
    }
    assert(loadContig > 0);

    // Further restrict the contiguity based on the contiguity of the src to dst
    // layout e.g. if the order of the blocked and shared encoding is different
    // we can only load one element at a time or if the shared encoding is
    // swizzled we cannot exceed the vector size of the swizzling pattern
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
    // For buffer_load_to_local, we do NOT cap loadContig by
    // regToSharedLayout.getNumConsecutiveInOut() (which uses the CURRENT src
    // encoding). Unlike AsyncCopyGlobalToLocalOp (where the tensor-pointer
    // encoding bounds the real global-memory contiguity), buffer_load_to_local
    // uses a scalar ptr + offset tensor; the true contiguity comes from
    // ptr/offset DIVISIBILITY, already computed above. Capping by the current
    // encoding's consecutive-in-out would reduce loadContig to 1 (when
    // sizePerThread=[1,1]), preventing the rewrite. The new encoding built
    // below sets contigPerThread = loadContig, satisfying the property.
    //
    // regToSharedLayout is still used in the padded-encoding branch below.
    // Select the largest supported load width equal or smaller than loadContig
    auto elemBitWidth = dstTy.getElementTypeBitWidth();
    loadContig =
        fitToValidDirectToLdsVecSize(loadContig, elemBitWidth, targetInfo);
    if (loadContig == 0) {
      return rewriter.notifyMatchFailure(
          bufOp, "could not find layout config to create coalesced writes");
    }

    // Do not rewrite if we already use the correct contiguity (could be from a
    // previous rewrite)
    auto mod = bufOp->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::lookupNumWarps(bufOp);
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

    ttg::DistributedEncodingTrait newDistEnc;
    {
      unsigned tmpVec = loadContig;
      // canLoadDirectToLDS calls getPointeeBitWidth(srcTy), which requires
      // srcTy to be a tensor-of-pointers. For buffer_load_to_local, srcTy is
      // the offset tensor (tensor<...xi32>), NOT a pointer tensor. We must
      // reconstruct the effective pointer type (tensor<...x!tt.ptr<elem>>) the
      // same way BufferLoadToLocalOpConversion does in LoadStoreOpToLLVM.cpp:730.
      auto effectivePtrTy = cast<RankedTensorType>(
          LLVM::AMD::getPointerTypeWithShape(bufOp.getPtr(), offsets));
      bool already = LLVM::AMD::canLoadDirectToLDS(targetInfo, effectivePtrTy, dstTy.getEncoding(),
                                                   dstTy.getAllocShape(), tmpVec);
      if (already) {
        return rewriter.notifyMatchFailure(bufOp, "already writes coalesced");
      }
    }
    // Check if we support load contig because canLoadDirectToLds can change it
    if (!targetInfo.supportsDirectToLdsLoadBitWidth(loadContig * elemBitWidth)) {
      return rewriter.notifyMatchFailure(bufOp,
                                         "unable to find supported vector size "
                                         "based on src and dst encodings");
    }

    if (isa<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding())) {
      // For swizzled layouts we apply the swizzling during lowering so we only
      // adjust the sizePerThread of the blocked encoding to avoid strided
      // writes into LDS.
      // For buffer_load_to_local, the offsets tensor sizePerThread may be
      // smaller than loadContig (since the scalar ptr + offsets pattern can
      // address contiguous elements even when sizePerThread=1). Unlike
      // CoalesceAsyncCopyWrites, we do NOT assert srcElemContig >= loadContig
      // and instead let BlockedEncodingAttr::get compute a fresh valid layout
      // with the widened sizePerThread.
      auto contigPerThread = ttg::getContigPerThread(srcTy);
      contigPerThread[blockedEnc.getOrder()[0]] = loadContig;
      newDistEnc = BlockedEncodingAttr::get(
          bufOp.getContext(), srcTy.getShape(), contigPerThread,
          blockedEnc.getOrder(), numWarps, threadsPerWarp,
          blockedEnc.getCGALayout());
    } else if (paddedEnc) {
      // For padded layouts the linear_component maps from LDS offsets to n-D
      // tensor indices. This mapping might reorder elements resulting in
      // scattered writes into LDS which is not supported on GFX9. To ensure
      // coalesced writes we change the src layout to a linear encoding which
      // effectivly copies/mimicks the linear_component so each warp (reg+lane
      // bases) map to consecutive LDS offsets resulting in coalesced writes
      // The new linear encoding is build by taking bases from the
      // linear_component and assigning them to reg/lane/warp bases in the
      // following steps:
      // 1) Take log2(loadContig) bases as reg bases to ensure our registers per
      // load instruction point to contiguous elements in LDS.
      // 2) Take log2(threadsPerWarp) as lane bases to ensure lanes write
      // contiguous into LDS.
      // 3) Take log2(numWarps) as warp bases or add braodcasting bases if we
      // run out of bases
      // 4) Take any remaining bases as additional reg bases

      auto *ctx = srcTy.getContext();
      StringAttr kOffset = StringAttr::get(ctx, "offset");

      auto rank = srcTy.getRank();

      auto offsetBases = sharedLayout.getBases().lookup(kOffset);

      int log2LoadContig = llvm::Log2_32(loadContig);
      int log2ThreadsPerWarp = llvm::Log2_32(threadsPerWarp);
      int log2NumWarps = llvm::Log2_32(numWarps);

      if (offsetBases.size() < log2LoadContig + log2ThreadsPerWarp) {
        return rewriter.notifyMatchFailure(
            bufOp, "dst shape is too small. We require at least loadContig * "
                    "threadsPerWarp elements");
      }

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
      if (newRegToShared.getNumConsecutiveInOut() < loadContig) {
        return rewriter.notifyMatchFailure(
            bufOp, "could not coalesce global addresses based on the linear "
                    "component of the padded encoding");
      }

      newDistEnc = ttg::LinearEncodingAttr::get(ctx, std::move(newRegLayout));
    } else {
      assert(false && "Unsupported layout");
    }

    if (newDistEnc == srcTy.getEncoding()) {
      return rewriter.notifyMatchFailure(
          bufOp, "Unable to find a new src layout to coalesce writes to LDS");
    }

    // Convert layout of src, mask and other to new encoding
    auto convertLayout = [&rewriter](auto loc, Value old, auto newEnc) {
      auto oldTy = cast<RankedTensorType>(old.getType());
      RankedTensorType newSrcTy = oldTy.cloneWithEncoding(newEnc);
      return ttg::ConvertLayoutOp::create(rewriter, loc, newSrcTy, old);
    };

    auto loc = bufOp->getLoc();
    // ptr is a scalar pointer (!tt.ptr<elem>), not a tensor — do NOT convert it.
    // Only offsets (and optionally mask/other) are tensors that need layout change.
    Value cvtOffs = convertLayout(loc, offsets, newDistEnc);

    if (mask)
      mask = convertLayout(loc, mask, newDistEnc);
    if (other)
      other = convertLayout(loc, other, newDistEnc);

    rewriter.modifyOpInPlace(bufOp, [&]() {
      bufOp.getOffsetsMutable().assign(cvtOffs);
      if (mask)
        bufOp.getMaskMutable().assign(mask);
      if (other)
        bufOp.getOtherMutable().assign(other);
      bufOp.setContiguity(loadContig);
    });
    return success();
  }

private:
  const triton::AMD::TargetInfo &targetInfo;
  const DenseMap<triton::amdgpu::BufferLoadToLocalOp, unsigned> &bufferLoadToLocalContiguity;
};

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

    // Precompute the contiguity of all AsyncCopy ops based on the src and
    // mask contiguity/alignment to avoid rebuilding ModuleAxisInfoAnalysis
    // after every IR change.
    AMD::ModuleAxisInfoAnalysis axisAnalysis(m);
    DenseMap<ttg::AsyncCopyGlobalToLocalOp, unsigned> asyncCopyContiguity;
    DenseMap<triton::amdgpu::BufferLoadToLocalOp, unsigned> bufferOpsToLocalContiguity;
    m->walk([&](Operation* op) {
      if (auto globalCopyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
        unsigned contiguity =
            mlir::LLVM::AMD::getContiguity(globalCopyOp.getSrc(), axisAnalysis);
        if (auto mask = globalCopyOp.getMask()) {
          contiguity =
              std::min<unsigned>(contiguity, axisAnalysis.getMaskAlignment(mask));
        }
        asyncCopyContiguity.insert({globalCopyOp, contiguity});
      }

      if (auto bufferCopyOp = dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(op)) {
        Value ptr = bufferCopyOp.getPtr();
        Value offsets = bufferCopyOp.getOffsets();

        auto offsetTy = cast<RankedTensorType>(offsets.getType());
        unsigned elemBitWidth = triton::getPointeeBitWidth(ptr.getType());
        unsigned elemNumBytes = std::max(elemBitWidth / 8, 1u);

        // Alignment from the scalar base pointer divisibility.
        unsigned ptrAlign = 1;
        auto *ptrInfo = axisAnalysis.getAxisInfo(ptr);
        if (ptrInfo) {
          unsigned ptrDivisibility = ptrInfo->getDivisibility(0);
          ptrAlign = std::max(ptrDivisibility / elemNumBytes, 1u);
        }

        // Alignment from the offset tensor's innermost (fast-varying) dimension,
        // derived from axis-info divisibility — NOT capped by sizePerThread.
        unsigned offsetAlign = 1;
        auto *offsetInfo = axisAnalysis.getAxisInfo(offsets);
        if (offsetInfo) {
          auto contiguityVec = offsetInfo->getContiguity();
          SmallVector<unsigned> offsetOrder = getOrderFromContiguity(contiguityVec);
          unsigned innerDim = offsetOrder[0];
          unsigned divisibility = offsetInfo->getDivisibility(innerDim);
          offsetAlign = std::max(divisibility / elemNumBytes, 1u);
        }

        // Cap to the widest vectorized load (128 bits).
        unsigned maxVec = 128 / elemBitWidth;
        unsigned contiguity = std::min({ptrAlign, offsetAlign, maxVec});

        // NOTE: For buffer_load_to_local, we intentionally do NOT cap by
        // getMaskAlignment(mask). The mask for these ops may have a different
        // (smaller) shape than the offsets tensor (e.g., mask_n[:, None] is
        // [BLOCK_N, 1] while offsets is [BLOCK_N, BLOCK_D_Q]). getMaskAlignment
        // would return 1 for such a 1-wide mask, incorrectly capping contiguity.
        // The mask is applied per-row (one bit covers all BLOCK_D_Q elements),
        // so it does not limit vectorization along the column (fast) dimension.
        bufferOpsToLocalContiguity.insert({bufferCopyOp, contiguity});
      }
    });

    patterns.add<CoalesceAsyncCopyWrites>(targetInfo, asyncCopyContiguity,
                                          context);
    patterns.add<CoalesceBufferLoadToLocal>(targetInfo, bufferOpsToLocalContiguity,
                                          context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
