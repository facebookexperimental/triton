#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUMMALOWERINGPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

class SyncMMALowering : public OpInterfaceRewritePattern<MMAv5OpInterface> {
public:
  using OpInterfaceRewritePattern<MMAv5OpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(MMAv5OpInterface op,
                                PatternRewriter &rewriter) const override {
    // If the op doesn't have synchronous semantic skip the pattern.
    if (op.isAsync())
      return failure();
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
    auto numCTAs = gpu::lookupNumCTAs(op);
    auto barrierCGALayout = ttg::CGAEncodingAttr::get1DLayout(ctx, numCTAs);
    auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
        ctx, 1, 1, 1, {0}, barrierCGALayout);
    ttg::MemDescType barrierMemDescType =
        ttg::MemDescType::get({numCTAs}, rewriter.getI64Type(), barrierEncoding,
                              sharedMemorySpace, /*mutableMemory=*/true);
    Value barrierAlloc =
        ttg::LocalAllocOp::create(rewriter, loc, barrierMemDescType, Value());
    InitBarrierOp::create(rewriter, loc, barrierAlloc, 1);
    op.addCompletionBarrier(barrierAlloc,
                            arith::ConstantIntOp::create(rewriter, loc, 1, 1));
    op.setIsAsync(true);

    rewriter.setInsertionPointAfter(op);
    Value phase = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    WaitBarrierOp::create(rewriter, loc, barrierAlloc, phase,
                          op.getPredicate());
    InvalBarrierOp::create(rewriter, loc, barrierAlloc);
    return success();
  }
};

static bool areEquivalentBarrierViews(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto lhsIndex = lhs.getDefiningOp<ttg::MemDescIndexOp>();
  auto rhsIndex = rhs.getDefiningOp<ttg::MemDescIndexOp>();
  if (!lhsIndex || !rhsIndex ||
      !areEquivalentBarrierViews(lhsIndex.getSrc(), rhsIndex.getSrc()))
    return false;
  APInt lhsValue;
  APInt rhsValue;
  return matchPattern(lhsIndex.getIndex(), m_ConstantInt(&lhsValue)) &&
         matchPattern(rhsIndex.getIndex(), m_ConstantInt(&rhsValue)) &&
         lhsValue == rhsValue;
}

static bool isConstantZero(Value value) {
  APInt constant;
  return value && matchPattern(value, m_ConstantInt(&constant)) &&
         constant.isZero();
}

static LogicalResult lowerTwoCTARHSScaleCopy(Value src, Value dst,
                                             Operation *anchor,
                                             PatternRewriter &rewriter) {
  Location loc = src.getLoc();
  auto srcType = cast<ttg::MemDescType>(src.getType());
  auto dstType = cast<ttg::MemDescType>(dst.getType());
  int numWarps = ttg::lookupNumWarps(anchor);
  if (numWarps < 4 || !llvm::isPowerOf2_32(numWarps)) {
    return emitError(loc)
           << "two-CTA M64 scaled MMA scale lowering requires a power-of-two "
              "MMA partition with at least 4 warps; got "
           << numWarps;
  }

  MLIRContext *context = src.getContext();
  auto shape = dstType.getShape();
  Type elType = dstType.getElementType();
  auto sharedLayout = ttg::getScaleSmemLayoutForTMEMCopy(
      context, shape, ttg::CGAEncodingAttr::get1CTALayout(context, 2));
  auto sharedEncoding = ttg::SharedLinearEncodingAttr::get(
      context, std::move(sharedLayout), /*alignment=*/128);
  auto sharedViewType = ttg::MemDescType::get(shape, elType, sharedEncoding,
                                              srcType.getMemorySpace(),
                                              srcType.getMutableMemory());
  Value sharedView =
      ttg::MemDescReinterpretOp::create(rewriter, loc, sharedViewType, src);
  auto registerEncoding = getDefaultLayoutForTmemLdSt(dstType, numWarps);
  auto registerType = RankedTensorType::get(shape, elType, registerEncoding);
  Value scale = ttg::LocalLoadOp::create(rewriter, loc, registerType,
                                         sharedView, Value());
  Value pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  TMEMStoreOp::create(rewriter, loc, dst, scale, pred);
  return success();
}

struct TCGen5MMAScaleSharedToTmemConversion
    : public OpRewritePattern<TCGen5MMAScaledOp> {
  using OpRewritePattern<TCGen5MMAScaledOp>::OpRewritePattern;

  // Create a tmem_copy of scales from shared memory to tmem. `rows` is the M or
  // N of the MMA operation (for LHS or RHS respectively).
  FailureOr<bool> lowerScaleToTmem(
      OpOperand &operand, PatternRewriter &rewriter, int rows,
      TensorMemoryScalesBlockRepOrder blockRepOrder,
      TensorMemoryCTAMode ctaMode = TensorMemoryCTAMode::DEFAULT) const {
    Location loc = operand.getOwner()->getLoc();
    MLIRContext *context = operand.getOwner()->getContext();
    Attribute tensorMemorySpace = TensorMemorySpaceAttr::get(context);
    auto oldType = cast<ttg::MemDescType>(operand.get().getType());
    int numWarps = 0;
    if (ctaMode == TensorMemoryCTAMode::TwoCTA_RHS) {
      numWarps = ttg::lookupNumWarps(operand.getOwner());
      if (numWarps < 4 || !llvm::isPowerOf2_32(numWarps)) {
        operand.getOwner()->emitError()
            << "two-CTA M64 scaled MMA scale lowering requires a power-of-two "
               "MMA partition with at least 4 warps; got "
            << numWarps;
        return failure();
      }
    }
    auto numElems = product(oldType.getShape());
    Type elType = oldType.getElementType();
    // The scales SMEM source may use a flexible multi-dimensional layout (e.g.
    // a 5D `1x2x32x4x4` shape), whose CGALayout has the source's rank. The
    // scales TMEM encoding is always rank 2, so derive a rank-2 CGALayout from
    // the source's CTA split rather than reusing the (possibly higher-rank)
    // source CGALayout directly.
    SmallVector<unsigned> srcCTAsPerCGA =
        ttg::getCTAsPerCGA(oldType.getEncoding());
    ttg::CGAEncodingAttr CGALayout;
    if (product<unsigned>(srcCTAsPerCGA) == 1) {
      CGALayout = ttg::CGAEncodingAttr::get1CTALayout(context, /*rank=*/2);
    } else {
      SmallVector<unsigned> srcCTASplitNum =
          ttg::getCTASplitNum(oldType.getEncoding());
      unsigned ctasPerCGA = product<unsigned>(srcCTAsPerCGA);
      unsigned ctaSplitNum = product<unsigned>(srcCTASplitNum);
      CGALayout = ttg::CGAEncodingAttr::fromSplitParams(
          context, /*CTAsPerCGA=*/{ctasPerCGA, 1u},
          /*CTASplitNum=*/{ctaSplitNum, 1u}, /*CTAOrder=*/{0u, 1u});
    }
    // Distribute the scales across the rows of the MMA operation.
    SmallVector<int64_t> shape = {rows, numElems / rows};
    Attribute scaleEncoding = TensorMemoryScalesEncodingAttr::get(
        context, CGALayout, blockRepOrder, ctaMode);
    Type scaleAType =
        ttg::MemDescType::get(shape, elType, scaleEncoding, tensorMemorySpace,
                              /*mutableMemory=*/true);
    auto tmemAlloc = TMEMAllocOp::create(rewriter, loc, scaleAType, Value());
    if (ctaMode == TensorMemoryCTAMode::TwoCTA_RHS) {
      // The M=128 cta_group::2 Layout-B RHS consumes the high N half from
      // row partition 64. Reinterpret the packed scale SMEM as its canonical
      // 2D tensor and let the TMEM store layout move that basis from columns
      // to rows. Cross-CTA publication remains an explicit user barrier.
      if (failed(lowerTwoCTARHSScaleCopy(operand.get(), tmemAlloc,
                                         operand.getOwner(), rewriter)))
        return failure();
    } else {
      TMEMCopyOp::create(rewriter, loc, operand.get(), tmemAlloc);
    }
    operand.set(tmemAlloc);
    return true;
  }

  LogicalResult matchAndRewrite(TCGen5MMAScaledOp op,
                                PatternRewriter &rewriter) const override {
    auto aScaleType = op.getAScale().getType();
    auto bScaleType = op.getBScale().getType();
    if (aScaleType.getShape() != aScaleType.getAllocShape() ||
        bScaleType.getShape() != bScaleType.getAllocShape()) {
      op.emitError("subviews NYI");
      return failure();
    }
    int blockM = op.getBlockM();
    int blockN = op.getBlockN();
    auto dEncoding =
        cast<TensorMemoryEncodingAttr>(op.getD().getType().getEncoding());
    bool isTwoCTAM64 =
        op.getTwoCtas() && dEncoding.getBlockM() == 64 &&
        dEncoding.getCtaMode() == TensorMemoryCTAMode::TwoCTA_RHS;
    if (isTwoCTAM64) {
      // A scales remain per-CTA along M. B scales describe the complete N
      // dimension and use the Layout-B row partition selected by TwoCTA_RHS.
      blockN *= 2;
    }
    auto aScaleBlockRepOrder = getTensorMemoryScalesBlockRepOrder(
        op, /*isA=*/true, op.getAType(), op.getBType(),
        aScaleType.getElementType(), bScaleType.getElementType());
    auto bScaleBlockRepOrder = getTensorMemoryScalesBlockRepOrder(
        op, /*isA=*/false, op.getAType(), op.getBType(),
        aScaleType.getElementType(), bScaleType.getElementType());
    Operation *rhsPublicationBarrier = nullptr;
    if (isTwoCTAM64) {
      for (Operation *wait = op->getPrevNode(); wait;
           wait = wait->getPrevNode()) {
        if (isa<ClusterBarrierOp>(wait)) {
          rhsPublicationBarrier = wait;
          break;
        }
        if (isa<ClusterWaitOp>(wait)) {
          for (Operation *arrive = wait->getPrevNode(); arrive;
               arrive = arrive->getPrevNode()) {
            if (isa<ClusterArriveOp>(arrive)) {
              rhsPublicationBarrier = arrive;
              break;
            }
            if (isa<ClusterWaitOp, ClusterBarrierOp, WaitBarrierOp,
                    MMAv5OpInterface>(arrive))
              break;
          }
          break;
        }
        if (auto waitBarrier = dyn_cast<WaitBarrierOp>(wait)) {
          if (!isConstantZero(waitBarrier.getPred())) {
            for (Operation *arrive = wait->getPrevNode(); arrive;
                 arrive = arrive->getPrevNode()) {
              if (auto arriveBarrier = dyn_cast<ArriveBarrierOp>(arrive)) {
                auto remote = arriveBarrier.getBarrier()
                                  .getDefiningOp<MapToRemoteBufferOp>();
                if (remote && !isConstantZero(arriveBarrier.getPred()) &&
                    areEquivalentBarrierViews(remote.getSrc(),
                                              waitBarrier.getBarrier())) {
                  rhsPublicationBarrier = arrive;
                  break;
                }
              }
              if (isa<ClusterWaitOp, ClusterBarrierOp, WaitBarrierOp,
                      MMAv5OpInterface>(arrive))
                break;
            }
          }
          break;
        }
        if (isa<MMAv5OpInterface>(wait))
          break;
      }
      if (!rhsPublicationBarrier) {
        return op.emitError(
            "two-CTA blockM=64 scaled MMA requires a matched inter-CTA "
            "arrive/wait barrier after publishing its scales");
      }
      DominanceInfo dominance(op->getParentOp());
      auto sharedScaleDoesNotDominate = [&](Value scale) {
        auto type = cast<ttg::MemDescType>(scale.getType());
        return isa<ttg::SharedMemorySpaceAttr>(type.getMemorySpace()) &&
               !dominance.dominates(scale, rhsPublicationBarrier);
      };
      if (sharedScaleDoesNotDominate(op.getAScale()) ||
          sharedScaleDoesNotDominate(op.getBScale())) {
        return op.emitError(
            "two-CTA blockM=64 shared-memory scale operands must be defined "
            "before their publication barrier");
      }
    }

    bool anyChanged = false;
    if (isa<ttg::SharedMemorySpaceAttr>(aScaleType.getMemorySpace())) {
      if (rhsPublicationBarrier)
        rewriter.setInsertionPoint(rhsPublicationBarrier);
      FailureOr<bool> changed = lowerScaleToTmem(
          op.getAScaleMutable(), rewriter, blockM, aScaleBlockRepOrder);
      rewriter.setInsertionPoint(op);
      if (failed(changed))
        return failure();
      anyChanged = *changed || anyChanged;
    }
    if (isa<ttg::SharedMemorySpaceAttr>(bScaleType.getMemorySpace())) {
      auto bScaleCTAMode = isTwoCTAM64 ? TensorMemoryCTAMode::TwoCTA_RHS
                                       : TensorMemoryCTAMode::DEFAULT;
      if (rhsPublicationBarrier)
        rewriter.setInsertionPoint(rhsPublicationBarrier);
      FailureOr<bool> changed =
          lowerScaleToTmem(op.getBScaleMutable(), rewriter, blockN,
                           bScaleBlockRepOrder, bScaleCTAMode);
      rewriter.setInsertionPoint(op);
      if (failed(changed))
        return failure();
      anyChanged = *changed || anyChanged;
    }
    return LogicalResult::success(anyChanged);
  }
};

struct TwoCTARHSScaleCopyConversion : public OpRewritePattern<TMEMCopyOp> {
  using OpRewritePattern<TMEMCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto encoding = dyn_cast<TensorMemoryScalesEncodingAttr>(
        op.getDst().getType().getEncoding());
    if (!encoding || encoding.getCtaMode() != TensorMemoryCTAMode::TwoCTA_RHS)
      return failure();
    if (failed(lowerTwoCTARHSScaleCopy(op.getSrc(), op.getDst(), op, rewriter)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

std::pair<SmallVector<TCGen5CommitOp>, SmallVector<Value>>
collectCommitOpsAfter(MMAv5OpInterface mmaOp) {
  auto isConstTrue = [](Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<BoolAttr>(constOp.getValueAttr())) {
        return attr.getValue();
      }
    }
    return false;
  };

  SmallVector<TCGen5CommitOp> commitOps;
  SmallVector<Value> commitPredicates;
  auto mmaPred = mmaOp.getPredicate();
  Operation *nextOp = mmaOp->getNextNode();
  SmallVector<Value> mmaDescs = mmaOp.getCompletionDescs();

  while (nextOp) {
    if (auto commit = dyn_cast<TCGen5CommitOp>(nextOp)) {
      // If the mma predicate is true, or mma and commit ops use the same
      // predicate, it is safe to merge them. Otherwise, keep commit order by
      // not merging later commits across this one.
      if (!isConstTrue(mmaPred) && mmaPred != commit.getPred())
        break;
      if (!llvm::equal(mmaDescs, commit.getDescs()))
        break;
      commitOps.push_back(commit);
      commitPredicates.push_back(commit.getPred());
    } else if (!isPure(nextOp)) {
      // Only move commits across pure ops. We also bail here when encountering
      // another MMAv5 op.
      break;
    }
    nextOp = nextOp->getNextNode();
  }

  return {commitOps, commitPredicates};
}

// Return false if defining ops cannot be moved above the target op
bool moveDefiningOpsBefore(Value val, Operation *target) {
  SetVector<Operation *> toMove;

  std::function<bool(Value)> collectOpsToMove = [&](Value val) {
    if (auto defOp = val.getDefiningOp()) {
      if (defOp->getBlock() == target->getBlock() &&
          target->isBeforeInBlock(defOp)) {
        if (!isPure(defOp)) {
          // This defOp needs to move above the target op, but it is unsafe due
          // to impurity.
          return false;
        }
        for (Value operand : defOp->getOperands()) {
          if (!collectOpsToMove(operand)) {
            return false;
          }
        }
        toMove.insert(defOp);
      }
    }
    return true;
  };

  if (!collectOpsToMove(val)) {
    return false;
  }

  for (Operation *op : toMove) {
    op->moveBefore(target);
  }

  return true;
}

class MergeCommitIntoMMA : public OpInterfaceRewritePattern<MMAv5OpInterface> {
public:
  using OpInterfaceRewritePattern<MMAv5OpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(MMAv5OpInterface op,
                                PatternRewriter &rewriter) const override {
    auto [commitOps, predicates] = collectCommitOpsAfter(op);
    if (commitOps.empty()) {
      return llvm::failure();
    }
    for (auto [commit, pred] : llvm::zip(commitOps, predicates)) {
      if (!pred) {
        pred = arith::ConstantIntOp::create(rewriter, op.getLoc(), true, 1);
      }
      Value barrier = commit.getBarrier();
      if (!moveDefiningOpsBefore(barrier, op) ||
          !moveDefiningOpsBefore(pred, op)) {
        // Give up merging a commit if its defining ops cannot be moved above
        // the mma op.
        break;
      }
      op.addCompletionBarrier(barrier, pred);
      rewriter.eraseOp(commit);
    }
    return success();
  }
};

} // anonymous namespace

class TritonNvidiaGPUMMALoweringPass
    : public impl::TritonNvidiaGPUMMALoweringPassBase<
          TritonNvidiaGPUMMALoweringPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<SyncMMALowering, TCGen5MMAScaleSharedToTmemConversion,
                 TwoCTARHSScaleCopyConversion, MergeCommitIntoMMA>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
