#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_CANONICALIZELLVMIR
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
class SelectConstantConditionPattern : public OpRewritePattern<LLVM::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SelectOp op,
                                PatternRewriter &b) const override {
    BoolAttr cond;
    if (!matchPattern(op.getCondition(), m_Constant(&cond)))
      return failure();
    Value val = cond.getValue() ? op.getTrueValue() : op.getFalseValue();
    b.replaceOp(op, ValueRange{val});
    return success();
  }
};

class ElideFullClusterRankMaskPattern : public OpRewritePattern<LLVM::AndOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AndOp op,
                                PatternRewriter &rewriter) const override {
    APInt mask;
    Value rank = op.getLhs();
    if (!matchPattern(op.getRhs(), m_ConstantInt(&mask))) {
      if (!matchPattern(op.getLhs(), m_ConstantInt(&mask)))
        return failure();
      rank = op.getRhs();
    }

    if (!rank.getDefiningOp<triton::nvgpu::ClusterCTAIdOp>())
      return failure();

    unsigned numCTAs = triton::gpu::lookupPhysicalNumCTAs(op);
    if (mask.countr_one() < llvm::Log2_32_Ceil(numCTAs))
      return failure();

    rewriter.replaceOp(op, rank);
    return success();
  }
};

class BranchConstantConditionPattern : public OpRewritePattern<LLVM::CondBrOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CondBrOp op,
                                PatternRewriter &b) const override {
    BoolAttr cond;
    if (!matchPattern(op.getCondition(), m_Constant(&cond)))
      return failure();
    Block *dest = cond.getValue() ? op.getTrueDest() : op.getFalseDest();
    ValueRange args =
        cond.getValue() ? op.getTrueDestOperands() : op.getFalseDestOperands();
    b.replaceOpWithNewOp<LLVM::BrOp>(op, args, dest);
    return success();
  }
};

template <typename Op>
class BooleanConstantPattern : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &b) const override {
    if (!op.getType().isInteger(1))
      return failure();
    for (unsigned i = 0; i != 2; ++i) {
      BoolAttr constant;
      if (!matchPattern(op->getOperand(i), m_Constant(&constant)))
        continue;
      bool value = constant.getValue();
      if constexpr (std::is_same_v<Op, LLVM::AndOp>) {
        b.replaceOp(op, op->getOperand(value ? 1 - i : i));
      } else if constexpr (std::is_same_v<Op, LLVM::OrOp>) {
        b.replaceOp(op, op->getOperand(value ? i : 1 - i));
      } else {
        if (value) {
          BoolAttr other;
          if (!matchPattern(op->getOperand(1 - i), m_Constant(&other)))
            continue;
          b.replaceOpWithNewOp<LLVM::ConstantOp>(
              op, b.getI1Type(), b.getBoolAttr(!other.getValue()));
        } else {
          b.replaceOp(op, op->getOperand(1 - i));
        }
      }
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
struct CanonicalizeLLVMIR
    : public mlir::triton::gpu::impl::CanonicalizeLLVMIRBase<
          CanonicalizeLLVMIR> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<
        SelectConstantConditionPattern, ElideFullClusterRankMaskPattern,
        BranchConstantConditionPattern, BooleanConstantPattern<LLVM::AndOp>,
        BooleanConstantPattern<LLVM::OrOp>,
        BooleanConstantPattern<LLVM::XOrOp>>(&getContext());

    getContext()
        .getLoadedDialect<LLVM::LLVMDialect>()
        ->getCanonicalizationPatterns(patterns);
    for (mlir::RegisteredOperationName op :
         getContext().getRegisteredOperationsByDialect(
             LLVM::LLVMDialect::getDialectNamespace()))
      op.getCanonicalizationPatterns(patterns, &getContext());

    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace
