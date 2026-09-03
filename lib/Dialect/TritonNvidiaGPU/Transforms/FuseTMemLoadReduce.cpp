#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TargetFeatures.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUFUSETMEMLOADREDUCEPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Strip away all intermediate ttg.convert_layout ops to reach the true
// producer.
static Value stripConvertLayout(Value v) {
  while (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>())
    v = cvt.getSrc();
  return v;
}

// Combine "ttng.tmem_load" and "tt.reduce" into "ttng.tmem_load" if it
// has the `redOp` attribute.  This targets the PTX `tcgen05.ld.red`
// instruction on Blackwell (sm103+).
//
// Match:
//
//   %v = ttng.tmem_load %tmem :
//        !ttg.memdesc<MxNxf32, #tmem, ...> -> tensor<MxNxf32, #blocked>
//   [ %cvt = ttg.convert_layout %v ... ] // optional
//   %r  = "tt.reduce"(%cvt or %v) ({...max/min combiner...}) {axis = 1}
//
// And rewrite this to:
//
//   %v, %r' = ttng.tmem_load %tmem {redOp = #ttng.redOp<max|min>, NaN = ...}
//             : ... -> tensor<MxNxf32, #blocked>, tensor<Mxf32,
//             slice(#blocked)>
//   [ %r = ttg.convert_layout %r' ]
//
// I.e., the fused load operation additionally performs an
// element-wise reduction along the N-dimension of the input and produces a
// second result tensor %r'. For a input of shape [M, N], the
// reduced result has shape [M], containing one reduced value per "slice"
// of the N-dimension.

class FuseTMemLoadReducePattern : public OpRewritePattern<triton::ReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    // Instruction `tcgen05.ld.red` is only available on Blackwell sm103+.
    auto targetFeatures =
        TargetFeatures::fromModuleOp(reduceOp->getParentOfType<ModuleOp>());
    if (!targetFeatures.supportLdRed())
      return failure();

    // Bail if the region isn't a trivial shape, it should have exactly one
    // operand and one result.
    Operation *combiner = reduceOp.getSingleCombiner();
    if (!combiner)
      return failure();

    // Only support reduction along the N-dimension.
    if (reduceOp.getAxis() != 1)
      return failure();

    if (reduceOp.hasDefinedOrdering())
      return failure();

    // Look through "convert_layout" to find the "tmem_load", which is
    // guaranteed to produce a result with rank 2.
    auto tmemLoad = stripConvertLayout(reduceOp.getOperands()[0])
                        .getDefiningOp<TMEMLoadOp>();
    if (!tmemLoad)
      return failure();

    // Skip if already a fused load.
    if (tmemLoad.getRedOp())
      return failure();

    // This is not a HW/PTX restriction, tcgen05.ld.red supports integer integer
    // types, but this is a Triton limitation: the reduction is restricted to
    // f32, element types, see the definition of TTNG_TMEMLoadOp.
    if (!tmemLoad.getType().getElementType().isF32())
      return failure();

    TMEMLoadReduceModifier redOpKind;
    bool propagateNaN;
    if (isa<arith::MaxNumFOp>(combiner)) {
      // MaxNumFOp: if one of the arguments is NaN, the result is also NaN.
      redOpKind = TMEMLoadReduceModifier::MAX;
      propagateNaN = false;
    } else if (isa<arith::MaximumFOp>(combiner)) {
      // MaximumFOp: if one of the arguments is NaN, the result is the other
      // argument.
      redOpKind = TMEMLoadReduceModifier::MAX;
      propagateNaN = true;
    } else if (isa<arith::MinNumFOp>(combiner)) {
      redOpKind = TMEMLoadReduceModifier::MIN;
      propagateNaN = false;
    } else if (isa<arith::MinimumFOp>(combiner)) {
      redOpKind = TMEMLoadReduceModifier::MIN;
      propagateNaN = true;
    } else {
      return failure();
    }

    // Verify the layout supports the fused "tcgen05.ld.red": the load must be
    // packed, each thread's register bases must span the full N axis and must
    // not advance M.
    auto maxnreg = getContextualMaxNReg(tmemLoad);
    if (!supportsTMemLoadReduce(tmemLoad.getType(), tmemLoad.getSrc().getType(),
                                maxnreg))
      return failure();

    // Allow an `arrive`/`barrier` immediately after the tmem_load on a warp-
    // spec partition boundary. The pass is expected to hoist the fusion across
    // that barrier by pushing the arrive past the reduction, provided there are
    // no intervening memory ops. This mirrors the tmem_load splitting analysis
    // which operates per subtile.
    SmallVector<Operation *> barriersToMove;
    auto usesTMemLoad = [&](Operation *op) {
      return llvm::any_of(op->getOperands(), [&](Value operand) {
        return stripConvertLayout(operand) == tmemLoad.getResult() ||
            (tmemLoad.getToken() && operand == tmemLoad.getToken());
      });
    };
    if (tmemLoad->getBlock() == reduceOp->getBlock()) {
      for (Operation *op = tmemLoad->getNextNode(); op != reduceOp.getOperation();
           op = op->getNextNode()) {
        if (isa<ttg::ConvertLayoutOp>(op)) {
          // The convert_layout on the tmem_load -> reduce edge is already
          // handled via stripConvertLayout; allow it.
          continue;
        }
        // Allow mbarrier/cluster/named arrives that constitute the warp-spec
        // partition edge. Use operator info (isa) rather than name string
        // so only true arrive ops are considered. Wait barriers must not be
        // hoisted past the reduction.
        if (isa<ArriveBarrierOp, ClusterArriveOp, NamedBarrierArriveOp,
                AsyncCopyMbarrierArriveOp>(op)) {
          // Reject if the barrier uses the tmem_load result as an operand
          // (should not happen) or has a memory effect that would prevent
          // reordering past the reduction.
          if (usesTMemLoad(op))
            return failure();
          // Only pure barrier ops (no additional memory ops) are allowed to be
          // pushed past the reduction. Query MemoryEffectOpInterface directly
          // (arrive ops are not HasRecursiveMemoryEffects).
          if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
            SmallVector<MemoryEffects::EffectInstance> effs;
            effects.getEffects(effs);
            bool onlyBarrier = true;
            for (auto &eff : effs) {
              if (!isa<MemoryEffects::Read, MemoryEffects::Write>(
                      eff.getEffect()))
                continue;
              // Only an effect on the shared-memory mbarrier state is safe to
              // reorder past the reduction. Compare against the SharedMemory
              // resource singleton rather than its stringified name. A null
              // resource is unmodeled/unknown memory and must block hoisting.
              if (eff.getResource() != ttg::SharedMemory::get())
                onlyBarrier = false;
            }
            if (!onlyBarrier)
              return failure();
          }
          barriersToMove.push_back(op);
          continue;
        }
        // Pure, memory-effect-free bookkeeping may be inserted on the
        // partition edge (for example, memdesc_index selects the barrier
        // slot). It is safe to cross as long as it is independent of the
        // tmem_load result and token.
        if (isPure(op) && isMemoryEffectFree(op) && !usesTMemLoad(op))
          continue;
        // Any other op between load and reduce blocks fusion. This ensures we
        // do not reorder past real memory operations; only the partition
        // barrier is hoisted.
        return failure();
      }
    } else {
      // Cross-block / cross-region (e.g. different warp_specialize partitions)
      // is not handled yet. The TLX explicit arrive case and the AutoWS
      // same-partition case are both same-block, so reject otherwise.
      // Future per-subtile splitting would need to handle channels.
      return failure();
    }

    // Now build the fused load.
    auto *ctx = tmemLoad.getContext();
    auto redOpAttr = TMEMLoadReduceModifierAttr::get(ctx, redOpKind);
    BoolAttr nanAttr = propagateNaN ? rewriter.getBoolAttr(true) : BoolAttr();
    Type tokenTy = tmemLoad.getToken() ? tmemLoad.getToken().getType() : Type();
    rewriter.setInsertionPoint(tmemLoad);
    auto newLoad = TMEMLoadOp::create(rewriter, tmemLoad.getLoc(),
                                      /*result=*/tmemLoad.getType(),
                                      /*token=*/tokenTy,
                                      /*src=*/tmemLoad.getSrc(),
                                      /*dep=*/tmemLoad.getDep(), redOpAttr,
                                      /*abs=*/BoolAttr(), nanAttr);

    // Replace original load uses (result + optional token).
    SmallVector<Value> loadReplacements{newLoad.getResult()};
    if (tmemLoad.getToken())
      loadReplacements.push_back(newLoad.getToken());
    rewriter.replaceOp(tmemLoad, loadReplacements);

    // Splice the reduce into the fused-load `red` result, inserting a layout
    // conversion if the slice encodings differ.
    Value redResult = newLoad.getRed();
    Type expectedTy = reduceOp->getResult(0).getType();
    if (redResult.getType() != expectedTy) {
      rewriter.setInsertionPoint(reduceOp);
      redResult = ttg::ConvertLayoutOp::create(rewriter, reduceOp.getLoc(),
                                               expectedTy, redResult);
    }
    // Push the warp-spec partition barrier(s) past the reduction so the
    // fused tcgen05.ld.red can be formed. The barrier is independent of the
    // reduction's data flow and is moved to after the reduce (now the fused
    // load's red result) per subtile. Reverse iteration preserves relative
    // order of multiple arrives.
    for (Operation *barrier : llvm::reverse(barriersToMove)) {
      rewriter.moveOpAfter(barrier, reduceOp.getOperation());
    }
    rewriter.replaceOp(reduceOp, redResult);
    return success();
  }
};

} // anonymous namespace

class TritonNvidiaGPUFuseTMEMLoadReducePass
    : public impl::TritonNvidiaGPUFuseTMEMLoadReducePassBase<
          TritonNvidiaGPUFuseTMEMLoadReducePass> {
public:
  using TritonNvidiaGPUFuseTMEMLoadReducePassBase<
      TritonNvidiaGPUFuseTMEMLoadReducePass>::
      TritonNvidiaGPUFuseTMEMLoadReducePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FuseTMemLoadReducePattern>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
