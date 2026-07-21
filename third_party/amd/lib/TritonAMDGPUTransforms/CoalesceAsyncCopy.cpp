#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-coalesce-async-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCOALESCEASYNCCOPY
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Fallback for a direct-to-LDS async copy that cannot be lowered on this target
// (e.g. a masked partial-K fp16 load whose per-thread vector width collapses to
// a bitwidth CDNA does not support for direct-to-LDS: 16-bit). Rather than
// leave an op that later fails to legalize in ConvertTritonAMDGPUToLLVM
// (`unrealized_conversion_cast` / "failed to legalize
// async_copy_global_to_local"), rewrite it into a synchronous
// tt.load + ttg.local_store. This mirrors the non-async load->local_store
// pipeline; the Membar pass inserts the LDS barrier before the consuming
// local_load, so it is correct (it only loses the async prefetch overlap for
// this copy).
static LogicalResult
decomposeAsyncCopyToSync(ttg::AsyncCopyGlobalToLocalOp copyOp,
                         PatternRewriter &rewriter) {
  Location loc = copyOp.getLoc();
  Value src = copyOp.getSrc();
  Value dst = copyOp.getResult();
  Value mask = copyOp.getMask();
  Value other = copyOp.getOther();

  rewriter.setInsertionPoint(copyOp);

  // Synchronous global load into registers.
  Value loaded;
  if (mask)
    loaded = triton::LoadOp::create(rewriter, loc, src, mask, copyOp.getCache(),
                                    copyOp.getEvict(), copyOp.getIsVolatile());
  else
    loaded = triton::LoadOp::create(rewriter, loc, src, copyOp.getCache(),
                                    copyOp.getEvict(), copyOp.getIsVolatile());

  // Preserve `other` (the fill value for masked-out lanes); a masked tt.load
  // leaves those lanes undefined, so select them explicitly.
  if (other && mask)
    loaded = arith::SelectOp::create(rewriter, loc, mask, loaded, other);

  // Store into the shared-memory buffer. Membar adds the barrier before the
  // consuming local_load, exactly as for the non-async pipeline.
  ttg::LocalStoreOp::create(rewriter, loc, loaded, dst);

  // The synchronous copy is not part of any async group: drop its token from
  // every async_commit_group / async_wait that consumed it (both accept zero
  // tokens).
  Value token = copyOp.getToken();
  SmallVector<Operation *> users(token.getUsers().begin(),
                                 token.getUsers().end());
  for (Operation *user : users) {
    SmallVector<Value> kept;
    for (Value t : user->getOperands())
      if (t != token)
        kept.push_back(t);
    rewriter.setInsertionPoint(user);
    if (auto commit = dyn_cast<ttg::AsyncCommitGroupOp>(user)) {
      auto n = ttg::AsyncCommitGroupOp::create(rewriter, commit.getLoc(), kept);
      rewriter.replaceOp(commit, n.getResult());
    } else if (auto wait = dyn_cast<ttg::AsyncWaitOp>(user)) {
      auto n = ttg::AsyncWaitOp::create(rewriter, wait.getLoc(), kept,
                                        wait.getNum());
      rewriter.replaceOp(wait, n.getResult());
    }
  }

  rewriter.eraseOp(copyOp);
  return success();
}

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
      // A padded dst whose per-thread width collapses below a supported
      // direct-to-LDS bitwidth (e.g. 16-bit partial-K masked load on CDNA4)
      // cannot be lowered as a direct-to-LDS copy and would crash later in
      // ConvertTritonAMDGPUToLLVM. Fall back to a synchronous load +
      // local_store. Swizzled dsts are left unchanged (the lowering applies
      // swizzling to the src pointers and can still handle them).
      if (paddedEnc)
        return decomposeAsyncCopyToSync(copyOp, rewriter);
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
      auto newRegLayout = triton::AMD::deduceRegLayoutFromPaddedShared(
          sharedLayout, loadContig, threadsPerWarp, numWarps, srcTy.getShape(),
          blockedEnc.getCGALayout(), ctx);
      if (failed(newRegLayout)) {
        return rewriter.notifyMatchFailure(
            copyOp,
            "could not derive a coalesced direct-to-LDS register layout "
            "from the linear component of the padded encoding (dst shape "
            "too small or reg->shared consecutiveness < loadContig)");
      }

      newDistEnc = ttg::LinearEncodingAttr::get(ctx, std::move(*newRegLayout));
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

    if (!llvm::is_contained({AMD::ISAFamily::CDNA3, AMD::ISAFamily::CDNA4},
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
        contiguity =
            std::min<unsigned>(contiguity, axisAnalysis.getMaskAlignment(mask));
      }
      asyncCopyContiguity.insert({copyOp, contiguity});
    });
    patterns.add<CoalesceAsyncCopyWrites>(targetInfo, asyncCopyContiguity,
                                          context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
