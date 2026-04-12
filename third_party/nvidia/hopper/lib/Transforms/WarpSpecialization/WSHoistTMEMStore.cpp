#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

// Hoist a loop-invariant TMEMStore out of an outer ForOp when an inner loop's
// MMA uses useAccum=False on its first iteration, making the per-iteration
// store redundant.
class HoistLoopInvariantTMEMStore : public OpRewritePattern<ttng::TMEMStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMStoreOp store,
                                PatternRewriter &rewriter) const override {
    // 1. Store must have a token.
    if (!store.getDep())
      return failure();

    // 2. Store must be directly inside a scf::ForOp (the outer loop).
    auto outerFor = dyn_cast<scf::ForOp>(store->getParentOp());
    if (!outerFor)
      return failure();

    // 3-5. Source, predicate, and destination must be loop-invariant.
    if (!outerFor.isDefinedOutsideOfLoop(store.getSrc()) ||
        !outerFor.isDefinedOutsideOfLoop(store.getPred()) ||
        !outerFor.isDefinedOutsideOfLoop(store.getDst()))
      return failure();

    // 6. Store's input token must either be a block argument of the outer loop
    //    body (loop-carried) or be defined outside the loop (loop-invariant).
    auto depArg = dyn_cast<BlockArgument>(store.getDep());
    bool depIsLoopCarried = depArg && depArg.getOwner() == outerFor.getBody();
    bool depIsLoopInvariant = outerFor.isDefinedOutsideOfLoop(store.getDep());
    if (!depIsLoopCarried && !depIsLoopInvariant)
      return failure();

    // 7. Find all users of the TMEM buffer inside the outer loop and classify
    //    them: this store, an MMA inside a single nested ForOp, and optionally
    //    a TMEMLoadOp at the outer loop level.
    Value tmemBuf = store.getDst();
    scf::ForOp innerFor;
    ttng::MMAv5OpInterface mmaOp;
    ttng::TMEMLoadOp tmemLoad;

    for (Operation *user : tmemBuf.getUsers()) {
      // Skip users outside the outer loop.
      if (!outerFor->isAncestor(user))
        continue;

      if (user == store.getOperation())
        continue;

      if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(user)) {
        if (mmaOp)
          return failure(); // multiple MMAs
        mmaOp = mma;
        auto parentFor = dyn_cast<scf::ForOp>(mma->getParentOp());
        if (!parentFor || parentFor->getParentOp() != outerFor)
          return failure(); // MMA not in a direct child ForOp
        if (innerFor && innerFor != parentFor)
          return failure(); // multiple inner loops
        innerFor = parentFor;
      } else if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
        if (tmemLoad)
          return failure(); // multiple loads
        if (load->getParentOp() != outerFor)
          return failure(); // load not at outer loop level
        tmemLoad = load;
      } else {
        return failure(); // unexpected user
      }
    }

    if (!mmaOp || !innerFor)
      return failure();

    // Inner loop bounds must be loop-invariant (defined outside outer loop).
    if (!outerFor.isDefinedOutsideOfLoop(innerFor.getLowerBound()) ||
        !outerFor.isDefinedOutsideOfLoop(innerFor.getUpperBound()) ||
        !outerFor.isDefinedOutsideOfLoop(innerFor.getStep()))
      return failure();

    // 8. The MMA must have useAccum=False on the first iteration of the inner
    //    loop.
    Value accUseFlag = mmaOp.useAccumulator();
    bool firstIterFalse = false;
    if (matchPattern(accUseFlag, m_Zero())) {
      firstIterFalse = true;
    } else if (auto blockArg = dyn_cast<BlockArgument>(accUseFlag)) {
      // If useAccum is a block arg of the inner loop, check that its init
      // value is false.
      if (blockArg.getOwner() == innerFor.getBody()) {
        Value initVal = innerFor.getInitArgs()[blockArg.getArgNumber() - 1];
        firstIterFalse = matchPattern(initVal, m_Zero());
      }
    }
    if (!firstIterFalse)
      return failure();

    // 9. The store must precede the inner loop in program order.
    if (store->isBeforeInBlock(innerFor) == false)
      return failure();

    // 10. If a TMEMLoad exists, it must follow the inner loop.
    if (tmemLoad && innerFor->isBeforeInBlock(tmemLoad) == false)
      return failure();

    // === Transformation: hoist the store before the outer loop ===
    auto tokType = rewriter.getType<ttg::AsyncTokenType>();

    auto copyAttrs = [&](ttng::TMEMStoreOp hoistedStore) {
      for (auto attr : store->getAttrs())
        if (!hoistedStore->hasAttr(attr.getName()))
          hoistedStore->setAttr(attr.getName(), attr.getValue());
    };

    if (depIsLoopCarried) {
      int tokArgNo = depArg.getArgNumber() - 1; // arg 0 is induction var

      rewriter.setInsertionPoint(outerFor);
      auto hoistedStore = ttng::TMEMStoreOp::create(
          rewriter, store.getLoc(), tokType, store.getDst(),
          outerFor.getInitArgs()[tokArgNo], store.getSrc(), store.getPred());
      copyAttrs(hoistedStore);

      // Wire hoisted store's output as the outer loop's token init arg.
      outerFor.getInitArgsMutable()[tokArgNo].assign(hoistedStore.getToken());

      // Inside loop body: replace store's token with the region iter arg.
      store.getToken().replaceAllUsesWith(outerFor.getRegionIterArg(tokArgNo));
    } else {
      // Dep is defined outside the loop — just move the store before the loop.
      rewriter.setInsertionPoint(outerFor);
      auto hoistedStore = ttng::TMEMStoreOp::create(
          rewriter, store.getLoc(), tokType, store.getDst(), store.getDep(),
          store.getSrc(), store.getPred());
      copyAttrs(hoistedStore);

      store.getToken().replaceAllUsesWith(hoistedStore.getToken());
    }

    // Erase the original store.
    rewriter.eraseOp(store);
    return success();
  }
};

} // namespace

namespace mlir {

void doHoistLoopInvariantTMEMStore(triton::FuncOp &funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<HoistLoopInvariantTMEMStore>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    llvm_unreachable("Failed to hoist loop-invariant TMEM store");
  }
}

#define GEN_PASS_DEF_NVGPUTESTWSHOISTTMEMSTORE
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSHoistTMEMStorePass
    : public impl::NVGPUTestWSHoistTMEMStoreBase<
          NVGPUTestWSHoistTMEMStorePass> {
public:
  using impl::NVGPUTestWSHoistTMEMStoreBase<
      NVGPUTestWSHoistTMEMStorePass>::NVGPUTestWSHoistTMEMStoreBase;

  void runOnOperation() override {
    getOperation()->walk(
        [&](triton::FuncOp funcOp) { doHoistLoopInvariantTMEMStore(funcOp); });
  }
};

} // namespace mlir
