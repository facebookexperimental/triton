#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/TMAMulticast.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAMulticast.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMALOWERINGPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

static void
lowerTMALoad(Operation *op, RankedTensorType tensorType, Value desc,
             function_ref<void(Value, Value, Value, Value)> createLoad,
             PatternRewriter &rewriter) {
  MLIRContext *ctx = op->getContext();
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto loc = op->getLoc();
  auto encoding = getEncodingFromDescriptor(op, tensorType, desc);
  gpu::MemDescType memDescType = gpu::MemDescType::get(
      tensorType.getShape(), tensorType.getElementType(), encoding,
      sharedMemorySpace, /*mutableMemory=*/true);
  auto alloc =
      gpu::LocalAllocOp::create(rewriter, loc, memDescType).getResult();
  bool multicast = op->hasAttr(::mlir::triton::kMulticastAxesAttrName);
  auto numCTAs = gpu::lookupNumCTAs(op);
  auto barrierCGALayout =
      gpu::CGAEncodingAttr::get1DLayout(tensorType.getContext(), numCTAs);
  auto numBarrierSlots = product(barrierCGALayout.getCTASplitNum());
  auto barrierEncoding = gpu::SwizzledSharedEncodingAttr::get(
      tensorType.getContext(), 1, 1, 1, {0}, barrierCGALayout);
  gpu::MemDescType barrierMemDescType =
      gpu::MemDescType::get({numBarrierSlots}, rewriter.getI64Type(),
                            barrierEncoding, sharedMemorySpace,
                            /*mutableMemory=*/true);
  Value barrierAlloc =
      gpu::LocalAllocOp::create(rewriter, loc, barrierMemDescType);
  InitBarrierOp::create(rewriter, loc, barrierAlloc, 1);
  auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
  int sizeInBytes = product(shapePerCTA) *
                    tensorType.getElementType().getIntOrFloatBitWidth() / 8;
  Value pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  triton::nvidia_gpu::BarrierExpectOp::create(rewriter, loc, barrierAlloc,
                                              sizeInBytes, pred);
  createLoad(desc, barrierAlloc, alloc, pred);
  Value phase = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  WaitBarrierOp::create(rewriter, loc, barrierAlloc, phase);
  if (multicast)
    ClusterBarrierOp::create(rewriter, loc);
  InvalBarrierOp::create(rewriter, loc, barrierAlloc);
  replaceUsesWithLocalLoad(rewriter, op->getResult(0), alloc);
  op->erase();
}

class TMALoadLowering : public OpRewritePattern<DescriptorLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto createLoad = [&](Value desc, Value barrierAlloc, Value alloc,
                          Value pred) {
      if (op->hasAttr(::mlir::triton::kMulticastAxesAttrName))
        triton::nvidia_gpu::ClusterBarrierOp::create(rewriter, op.getLoc());
      auto copy = triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp::create(
          rewriter, op.getLoc(), desc, op.getIndices(), barrierAlloc, alloc,
          pred);
      if (Attribute axes =
              op->getAttr(::mlir::triton::kMulticastAxesAttrName))
        copy->setAttr(::mlir::triton::kMulticastAxesAttrName, axes);
    };
    lowerTMALoad(op, op.getType(), op.getDesc(), createLoad, rewriter);
    return success();
  }
};

class MaterializeTMAMulticastTargets
    : public OpRewritePattern<AsyncTMACopyGlobalToLocalOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AsyncTMACopyGlobalToLocalOp op,
                                PatternRewriter &rewriter) const override {
    auto axesAttr = op->getAttrOfType<DenseI32ArrayAttr>(
        ::mlir::triton::kMulticastAxesAttrName);
    if (!axesAttr || op.getMulticastTargets())
      return failure();

    llvm::SmallBitVector axes(3);
    for (int32_t axis : axesAttr.asArrayRef()) {
      if (axis < 0 || axis >= 3)
        return op.emitOpError("tt.multicast_axes values must be in [0, 2]");
      axes.set(axis);
    }
    if (axes.none())
      return op.emitOpError("tt.multicast_axes must not be empty");

    auto geometry =
        TMAClusterGeometry::get(op->getParentOfType<ModuleOp>());
    if (failed(geometry))
      return op.emitOpError(
          "tt.multicast_axes requires exact physical cluster geometry");

    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    Value rank = triton::nvgpu::ClusterCTAIdOp::create(rewriter, loc, i32);
    Value mask = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value leaderRank = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    for (unsigned candidate = 0; candidate < geometry->size(); ++candidate) {
      Value candidateRank =
          arith::ConstantIntOp::create(rewriter, loc, candidate, 32);
      Value isCandidate = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, rank, candidateRank);
      Value candidateMask = arith::ConstantIntOp::create(
          rewriter, loc, geometry->maskFor(candidate, axes), 32);
      Value candidateLeader = arith::ConstantIntOp::create(
          rewriter, loc, geometry->leaderFor(candidate, axes), 32);
      mask = arith::SelectOp::create(rewriter, loc, isCandidate, candidateMask,
                                     mask);
      leaderRank = arith::SelectOp::create(
          rewriter, loc, isCandidate, candidateLeader, leaderRank);
    }

    Value isLeader = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, rank, leaderRank);
    Value pred = arith::AndIOp::create(rewriter, loc, op.getPred(), isLeader);
    rewriter.modifyOpInPlace(op, [&] {
      op.getMulticastTargetsMutable().assign(mask);
      op.getPredMutable().assign(pred);
    });
    return success();
  }
};

struct TMAGatherLowering : public OpRewritePattern<DescriptorGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorGatherOp op,
                                PatternRewriter &rewriter) const override {
    Value xOffsets =
        sextI16ToI32Indices(op.getXOffsets(), rewriter, op.getLoc());

    auto createLoad = [&](Value desc, Value barrierAlloc, Value alloc,
                          Value pred) {
      triton::nvidia_gpu::AsyncTMAGatherOp::create(rewriter, op.getLoc(), desc,
                                                   xOffsets, op.getYOffset(),
                                                   barrierAlloc, alloc, pred);
    };
    lowerTMALoad(op, op.getType(), op.getDesc(), createLoad, rewriter);
    return success();
  }
};

static void lowerTMAStore(Operation *op, mlir::TypedValue<RankedTensorType> src,
                          Value desc,
                          function_ref<void(Value, Value)> createStore,
                          PatternRewriter &rewriter) {
  MLIRContext *ctx = op->getContext();
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto loc = op->getLoc();
  auto tensorType = src.getType();
  auto encoding = getEncodingFromDescriptor(op, src.getType(), desc);
  assert(isa<gpu::SharedEncodingTrait>(encoding));
  gpu::MemDescType memDescType = gpu::MemDescType::get(
      tensorType.getShape(), tensorType.getElementType(), encoding,
      sharedMemorySpace, /*mutableMemory=*/false);
  // If there is a local_load for src and there are no intervening instructions,
  // then we can safely reuse the allocation being loaded from as the source of
  // the TMA store.
  Value alloc;
  if (auto localLoad =
          dyn_cast_or_null<gpu::LocalLoadOp>(src.getDefiningOp())) {
    bool interfere = false;
    if (localLoad->getBlock() == op->getBlock()) {
      for (Operation *it = localLoad->getNextNode(); it && it != op;
           it = it->getNextNode()) {
        // Check op cannot update SMEM
        if (isa<gpu::LocalStoreOp, DescriptorLoadOp>(it)) {
          interfere = true;
          break;
        }
      }
    }

    if (!interfere) {
      alloc = localLoad.getSrc();
    }
  }

  if (!alloc) {
    alloc = gpu::LocalAllocOp::create(rewriter, loc, memDescType, src);
  }
  triton::nvidia_gpu::FenceAsyncSharedOp::create(rewriter, loc, false);
  createStore(desc, alloc);
  triton::nvidia_gpu::TMAStoreWaitOp::create(rewriter, loc, 0,
                                             /*read_only=*/false);
  rewriter.eraseOp(op);
}

struct TMAStoreLowering : public OpRewritePattern<DescriptorStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto createStore = [&](Value desc, Value alloc) {
      triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp::create(
          rewriter, op.getLoc(), desc, op.getIndices(), alloc);
    };
    lowerTMAStore(op, op.getSrc(), op.getDesc(), createStore, rewriter);
    return success();
  }
};

struct TMAReduceLowering : public OpRewritePattern<DescriptorReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto createStore = [&](Value desc, Value alloc) {
      triton::nvidia_gpu::AsyncTMAReduceOp::create(
          rewriter, op.getLoc(), op.getKind(), desc, op.getIndices(), alloc);
    };
    lowerTMAStore(op, op.getSrc(), op.getDesc(), createStore, rewriter);
    return success();
  }
};

struct TMAScatterLowering : public OpRewritePattern<DescriptorScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorScatterOp op,
                                PatternRewriter &rewriter) const override {
    Value xOffsets =
        sextI16ToI32Indices(op.getXOffsets(), rewriter, op.getLoc());
    auto createStore = [&](Value desc, Value alloc) {
      triton::nvidia_gpu::AsyncTMAScatterOp::create(
          rewriter, op.getLoc(), desc, xOffsets, op.getYOffset(), alloc);
    };
    lowerTMAStore(op, op.getSrc(), op.getDesc(), createStore, rewriter);
    return success();
  }
};

class TMACreateDescLowering : public OpRewritePattern<MakeTensorDescOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeTensorDescOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value descPtr;
    // If desc_ptr is provided, use it directly without creating global scratch
    if (op.getDescPtr()) {
      descPtr = op.getDescPtr();
    } else {
      // Create global scratch allocation when desc_ptr is not provided
      auto alloc = triton::gpu::GlobalScratchAllocOp::create(
          rewriter, loc, getPointerType(rewriter.getI8Type()), TMA_SIZE_BYTES,
          TMA_ALIGN, UnitAttr());
      descPtr = alloc.getResult();
    }

    if (failed(createTMADesc(descPtr, op, rewriter))) {
      return failure();
    }
    TensormapFenceproxyAcquireOp::create(rewriter, loc, descPtr);
    auto newDesc =
        ReinterpretTensorDescOp::create(rewriter, loc, op.getType(), descPtr);
    rewriter.replaceOp(op, newDesc);
    return success();
  }
};

} // anonymous namespace

class TritonNvidiaGPUTMALoweringPass
    : public impl::TritonNvidiaGPUTMALoweringPassBase<
          TritonNvidiaGPUTMALoweringPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TMALoadLowering, MaterializeTMAMulticastTargets,
                 TMAGatherLowering, TMAStoreLowering, TMAScatterLowering,
                 TMAReduceLowering, TMACreateDescLowering>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
