//===----------------------------------------------------------------------===//
// IfFromWhereOp to LLVM Conversion
//===----------------------------------------------------------------------===//
// Converts ttng.if_from_where operations to LLVM CFG with conditional branches.
// The condition is a tensor (typically from vote_ballot_sync), and we extract
// element 0 as the scalar branch condition since all elements are identical
// (warp-uniform).
//
// SAFETY INVARIANT:
// This lowering is used by the select_to_branch pass which GUARANTEES that:
//   1. The false-value of the original select comes from a TMEMLoadOp
//   2. The load and store operate on the SAME memory descriptor
//
// Therefore, when the condition is false:
//   result = select(false, trueVal, load(mem)) = load(mem)
//   store(mem, result) = store(mem, load(mem))  // NO-OP!
//
// Since storing the same value back is a no-op, skipping the entire block
// when the condition is false produces identical memory state. This makes
// the optimization safe for BOTH forward and backward kernels.
//===----------------------------------------------------------------------===//

#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;

/// Convert IfFromWhereOp to LLVM control flow.
/// This creates:
///   - A conditional branch based on the first element of the condition tensor
///   - A "then" block containing the converted body operations
///   - A merge block with phi nodes to select between then/else values
struct IfFromWhereOpConversion
    : public ConvertOpToLLVMPattern<ttng::IfFromWhereOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::IfFromWhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the converted condition (now an LLVM struct of i1 values)
    Value convertedCond = adaptor.getCondition();

    // Extract the first element from the condition tensor
    // Since the condition is warp-uniform, all elements have the same value
    Value scalarCond;
    if (auto structTy =
            dyn_cast<LLVM::LLVMStructType>(convertedCond.getType())) {
      // Extract element 0 from the struct
      scalarCond = rewriter.create<LLVM::ExtractValueOp>(loc, convertedCond, 0);
    } else {
      // Already a scalar i1
      scalarCond = convertedCond;
    }

    // Get the current block and function
    Block *currentBlock = rewriter.getInsertionBlock();
    Region *parentRegion = currentBlock->getParent();

    // Split the current block at the if_from_where op
    // Operations after the if will go to the merge block
    Block *mergeBlock = rewriter.splitBlock(currentBlock, op->getIterator());

    // Create the then block by inlining the op's then region
    Block *thenBlock = &op.getThenRegion().front();

    // Move the then block before the merge block
    parentRegion->getBlocks().splice(mergeBlock->getIterator(),
                                     op.getThenRegion().getBlocks());

    // Get the yield op and its values
    auto yieldOp = cast<ttng::IfFromWhereYieldOp>(thenBlock->getTerminator());
    SmallVector<Value> thenValues(yieldOp.getValues());

    // Replace the yield with a branch to merge block
    rewriter.setInsertionPointToEnd(thenBlock);
    rewriter.create<cf::BranchOp>(loc, mergeBlock, thenValues);
    rewriter.eraseOp(yieldOp);

    // Create the else block that just branches to merge with else values
    Block *elseBlock = rewriter.createBlock(mergeBlock);
    rewriter.create<cf::BranchOp>(loc, mergeBlock, adaptor.getElseValues());

    // Add block arguments to merge block for phi values
    SmallVector<Location> locs(op.getNumResults(), loc);
    SmallVector<Value> mergeBlockArgs;
    for (auto [idx, resultType] : llvm::enumerate(op.getResultTypes())) {
      Type convertedType = getTypeConverter()->convertType(resultType);
      mergeBlock->insertArgument(idx, convertedType, loc);
      mergeBlockArgs.push_back(mergeBlock->getArgument(idx));
    }

    // Add conditional branch at the end of the original block
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<cf::CondBranchOp>(loc, scalarCond, thenBlock, elseBlock);

    // Replace the op's results with the merge block arguments
    rewriter.replaceOp(op, mergeBlockArgs);

    return success();
  }
};

/// Convert IfFromWhereYieldOp - this is handled inline during IfFromWhereOp
/// conversion, but we still need a pattern to mark it as legal/handled
struct IfFromWhereYieldOpConversion
    : public ConvertOpToLLVMPattern<ttng::IfFromWhereYieldOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::IfFromWhereYieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Yield ops are erased during IfFromWhereOp conversion
    // This pattern should never actually match since yields are removed
    return failure();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateIfFromWhereOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<IfFromWhereOpConversion>(typeConverter, benefit);
  patterns.add<IfFromWhereYieldOpConversion>(typeConverter, benefit);
}
