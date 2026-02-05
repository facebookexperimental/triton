//===----------------------------------------------------------------------===//
// SelectToBranch Pass
//===----------------------------------------------------------------------===//
// This pass identifies patterns like:
//   acc = tmem_load(subslice)
//   scaled_acc = mul(acc, alpha)
//   result = select(ballot_cond, scaled_acc, acc)
//   tmem_store(subslice, result)
//
// And wraps them in ttng.if_from_where to enable conditional execution:
//   ttng.if_from_where %ballot_cond {
//     %loaded = tmem_load(subslice)
//     %scaled = mul(%loaded, alpha)
//     tmem_store(subslice, %scaled)
//     ttng.if_from_where_yield
//   } : tensor<...>
//
// SAFETY INVARIANT:
// This optimization is safe because the pattern matching GUARANTEES that:
//   1. falseVal comes from a TMEMLoadOp
//   2. The load and store operate on the SAME memory descriptor
//
// Therefore, when the condition is false:
//   result = select(false, trueVal, load(mem))
//          = load(mem)
//   store(mem, result) = store(mem, load(mem))  // NO-OP!
//
// Since storing the same value back is a no-op, skipping the entire block
// when the condition is false produces identical memory state.
//
// This invariant makes the optimization safe for BOTH forward and backward
// kernels, as long as the backward kernel follows the same pattern where
// the false-value is the loaded value from the same memory location.
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUSELECTTOBRANCHPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

/// Check if a value is uniform across all threads in a warp.
/// A value is warp-uniform if it's derived from `vote_ballot_sync`.
static bool isWarpUniform(Value val, int depth = 0) {
  if (depth > 10)
    return false;

  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return false;

  if (isa<ttng::VoteBallotSyncOp>(defOp))
    return true;

  if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
    return isWarpUniform(cmpOp.getLhs(), depth + 1) &&
           isWarpUniform(cmpOp.getRhs(), depth + 1);
  }
  if (auto cmpOp = dyn_cast<arith::CmpFOp>(defOp)) {
    return isWarpUniform(cmpOp.getLhs(), depth + 1) &&
           isWarpUniform(cmpOp.getRhs(), depth + 1);
  }

  if (isa<arith::ConstantOp>(defOp))
    return true;

  if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(defOp))
    return isWarpUniform(broadcastOp.getSrc(), depth + 1);

  if (auto expandOp = dyn_cast<triton::ExpandDimsOp>(defOp))
    return isWarpUniform(expandOp.getSrc(), depth + 1);

  if (auto splatOp = dyn_cast<triton::SplatOp>(defOp))
    return isWarpUniform(splatOp.getSrc(), depth + 1);

  if (auto convertOp = dyn_cast<ttg::ConvertLayoutOp>(defOp))
    return isWarpUniform(convertOp.getSrc(), depth + 1);

  return false;
}

/// Pattern to match the full load→compute→select→store sequence and wrap in
/// if_from_where. Matches:
///   acc = tmem_load(subslice)
///   scaled = compute(acc, ...)
///   result = select(ballot_cond, scaled, acc)
///   tmem_store(subslice, result)
///
/// When ballot_cond is false, the store writes back the same value that was
/// loaded, which is a no-op. So we can skip the entire sequence.
struct TMEMSelectStorePattern : public OpRewritePattern<ttng::TMEMStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the stored value comes from a select
    Value storedVal = storeOp.getSrc();
    auto selectOp = storedVal.getDefiningOp<arith::SelectOp>();
    if (!selectOp)
      return failure();

    Value cond = selectOp.getCondition();
    Value trueVal = selectOp.getTrueValue();
    Value falseVal = selectOp.getFalseValue();

    // Check that condition is warp-uniform (derived from ballot)
    if (!isWarpUniform(cond))
      return failure();

    // Check that condition is a tensor type (not scalar)
    if (!isa<RankedTensorType>(cond.getType()))
      return failure();

    // falseVal should come from a tmem_load on the same memory location
    auto loadOp = falseVal.getDefiningOp<ttng::TMEMLoadOp>();
    if (!loadOp)
      return failure();

    // Check that load and store operate on the same memory (same memdesc)
    if (loadOp.getSrc() != storeOp.getDst())
      return failure();

    // Check that trueVal's computation depends on the load
    Operation *trueDefOp = trueVal.getDefiningOp();
    if (!trueDefOp)
      return failure();

    // Collect the slice of operations between load and select that contribute
    // to trueVal. These will be moved inside the if_from_where body.
    llvm::SetVector<Operation *> opsToMove;

    // Use a worklist to find all ops in the backward slice from trueVal
    // that are between loadOp and selectOp
    llvm::SmallVector<Operation *> worklist;
    worklist.push_back(trueDefOp);

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();

      // Skip if already processed
      if (opsToMove.contains(op))
        continue;

      // Skip if this is the load op or before it
      if (op == loadOp)
        continue;
      if (!loadOp->isBeforeInBlock(op))
        continue;

      // Skip if this is the select or after it
      if (selectOp->isBeforeInBlock(op) || op == selectOp.getOperation())
        continue;

      opsToMove.insert(op);

      // Add operand-defining ops to worklist
      for (Value operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp())
          worklist.push_back(defOp);
      }
    }

    // Check that the load is only used by ops we're moving (and the select)
    for (Operation *user : loadOp->getUsers()) {
      if (user != selectOp.getOperation() && !opsToMove.contains(user))
        return failure();
    }

    // Check that the select result is only used by the store
    if (!selectOp->hasOneUse())
      return failure();

    Location loc = storeOp.getLoc();

    // Create if_from_where with no results (side-effect only)
    // The body contains: load, compute, store
    auto ifOp = rewriter.create<ttng::IfFromWhereOp>(loc, TypeRange{}, cond,
                                                     ValueRange{});

    // Build the body region - need to create a block first
    Block *thenBlock = rewriter.createBlock(&ifOp.getThenRegion());
    rewriter.setInsertionPointToStart(thenBlock);

    // Clone the load op inside the body
    IRMapping mapping;
    Operation *clonedLoad = rewriter.clone(*loadOp, mapping);

    // Clone all the ops that compute trueVal in topological order
    SmallVector<Operation *> sortedOps(opsToMove.begin(), opsToMove.end());
    llvm::sort(sortedOps, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });

    for (Operation *op : sortedOps) {
      // Before cloning, update mapping for any operands that come from the load
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        Value operand = op->getOperand(i);
        if (operand.getDefiningOp() == loadOp) {
          mapping.map(operand, clonedLoad->getResult(0));
        }
      }
      rewriter.clone(*op, mapping);
    }

    // Get the cloned trueVal (the computed value to store)
    Value clonedTrueVal = mapping.lookupOrDefault(trueVal);

    // Clone the store op, storing the computed value
    // We need to update the src operand to use clonedTrueVal
    rewriter.create<ttng::TMEMStoreOp>(loc, storeOp.getDst(), clonedTrueVal,
                                       storeOp.getPred());

    // Add the yield (no values since no results)
    rewriter.create<ttng::IfFromWhereYieldOp>(loc, ValueRange{});

    // Erase the original store, select, computation ops, and load
    rewriter.eraseOp(storeOp);
    rewriter.eraseOp(selectOp);
    for (Operation *op : llvm::reverse(sortedOps)) {
      if (op->use_empty())
        rewriter.eraseOp(op);
    }
    if (loadOp->use_empty())
      rewriter.eraseOp(loadOp);

    return success();
  }
};

struct SelectToBranchPass
    : public impl::TritonNvidiaGPUSelectToBranchPassBase<SelectToBranchPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TMEMSelectStorePattern>(ctx);

    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
