#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
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
        it != bufferLoadToLocalContiguity.end())
      loadContig = it->second;
    else
      return bufOp->emitError()
             << "No contiguity information about the copy op";
    assert(loadContig > 0);

llvm::outs() << "loc4, loadContig = " << loadContig << "\n";
    // Further restrict the contiguity based on the contiguity of the src to dst
    // layout e.g. if the order of the blocked and shared encoding is different
    // we can only load one element at a time or if the shared encoding is
    // swizzled we cannot exceed the vector size of the swizzling pattern
    LinearLayout regLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout;
llvm::outs() << "loc4.1\n";
    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(dstTy.getEncoding());
    if (paddedEnc) {
llvm::outs() << "loc4.2\n";
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
llvm::outs() << "loc4.3\n";
    auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);
    loadContig = std::min<unsigned>(loadContig,
                                    regToSharedLayout.getNumConsecutiveInOut());
llvm::outs() << "regToSharedLayout = " << regToSharedLayout.getNumConsecutiveInOut() << "\n";
llvm::outs() << "loc4.4, loadContig = " << loadContig << "\n";
    // Select the largest supported load width equal or smaller than loadContig
    auto elemBitWidth = dstTy.getElementTypeBitWidth();
    loadContig =
        fitToValidDirectToLdsVecSize(loadContig, elemBitWidth, targetInfo);
llvm::outs() << "loc4.5, loadContig = " << loadContig << "\n";
    if (loadContig == 0) {
      return rewriter.notifyMatchFailure(
          bufOp, "could not find layout config to create coalesced writes");
    }
llvm::outs() << "loc5\n";

    // Do not rewrite if we already use the correct contiguity (could be from a
    // previous rewrite)
    auto mod = bufOp->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::lookupNumWarps(bufOp);
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

    ttg::DistributedEncodingTrait newDistEnc;
llvm::outs() << "loc6\n";

    if (LLVM::AMD::canLoadDirectToLDS(targetInfo, srcTy, dstTy.getEncoding(),
                                      dstTy.getAllocShape(), loadContig)) {
      return rewriter.notifyMatchFailure(bufOp, "already writes coalesced");
    }
    // Check if we support load contig because canLoadDirectToLds can change it
    if (!targetInfo.supportsDirectToLdsLoadBitWidth(loadContig * elemBitWidth))
      return rewriter.notifyMatchFailure(bufOp,
                                         "unable to find supported vector size "
                                         "based on src and dst encodings");

llvm::outs() << "loc7\n";
    if (isa<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding())) {
      // For swizzled layouts we apply the swizzling during lowering so we only
      // adjust the sizePerThread of the blocked encoding to avoid strided
      // writes into LDS
      auto contigPerThread = ttg::getContigPerThread(srcTy);
      auto srcElemContig = contigPerThread[blockedEnc.getOrder()[0]];
      assert(srcElemContig >= loadContig);
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

llvm::outs() << "loc8\n";
      int log2LoadContig = llvm::Log2_32(loadContig);
      int log2ThreadsPerWarp = llvm::Log2_32(threadsPerWarp);
      int log2NumWarps = llvm::Log2_32(numWarps);

      if (offsetBases.size() < log2LoadContig + log2ThreadsPerWarp) {
        return rewriter.notifyMatchFailure(
            bufOp, "dst shape is too small. We require at least loadContig * "
                    "threadsPerWarp elements");
      }

llvm::outs() << "loc9\n";
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
llvm::outs() << "loc10\n";

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
llvm::outs() << "loc11\n";

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
    Value cvtPtr = convertLayout(loc, ptr, newDistEnc);
    Value cvtOffs = convertLayout(loc, offsets, newDistEnc);

    if (mask)
      mask = convertLayout(loc, mask, newDistEnc);
    if (other)
      other = convertLayout(loc, other, newDistEnc);
llvm::outs() << "loc12\n";

    rewriter.modifyOpInPlace(bufOp, [&]() {
      bufOp.getPtrMutable().assign(cvtPtr);
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
    m->walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
      unsigned contiguity =
          mlir::LLVM::AMD::getContiguity(copyOp.getSrc(), axisAnalysis);
      if (auto mask = copyOp.getMask()) {
        llvm::outs() << "global_mask, cont = " << axisAnalysis.getMaskAlignment(mask) << "\n";
        contiguity =
            std::min<unsigned>(contiguity, axisAnalysis.getMaskAlignment(mask));
      }
      llvm::outs() << "global_cont = " << contiguity << "\n";
      asyncCopyContiguity.insert({copyOp, contiguity});
    });

    // Precompute the contiguity of all BufferLoadToLocal ops based on the ptr, offset, and
    // mask contiguity/alignment to avoid rebuilding ModuleAxisInfoAnalysis
    // after every IR change.    
    DenseMap<triton::amdgpu::BufferLoadToLocalOp, unsigned> bufferOpsToLocalContiguity;
    m->walk([&](triton::amdgpu::BufferLoadToLocalOp bufOp) {
      Value ptr = bufOp.getPtr();
      Value offsets = bufOp.getOffsets();
      unsigned contiguity =
             mlir::LLVM::AMD::getContiguity(ptr, axisAnalysis);
      llvm::outs() << "offset_cont = " << contiguity << "\n";

      if (auto mask = bufOp.getMask()) {
        contiguity =
            std::min<unsigned>(contiguity, axisAnalysis.getMaskAlignment(mask));
      llvm::outs() << "mask_cont = " << axisAnalysis.getMaskAlignment(mask) << "\n";
      }

      bufferOpsToLocalContiguity.insert({bufOp, contiguity});
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
