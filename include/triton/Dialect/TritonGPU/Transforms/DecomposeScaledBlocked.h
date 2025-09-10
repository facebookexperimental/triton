#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::gpu {

<<<<<<< HEAD
=======
class DecomposeScaledBlocked : public OpRewritePattern<DotScaledOp> {
public:
  DecomposeScaledBlocked(MLIRContext *context, PatternBenefit benefit)
      : OpRewritePattern<DotScaledOp>(context, benefit) {}

  LogicalResult matchAndRewrite(DotScaledOp scaledDotOp,
                                PatternRewriter &rewriter) const override;

protected:
  FloatType getComputeType(ScaleDotElemType aType, ScaleDotElemType bType,
                           PatternRewriter &rewriter) const;
  TypedValue<RankedTensorType> scaleTo16(PatternRewriter &rewriter,
                                         TypedValue<RankedTensorType> scale,
                                         FloatType computeType) const;
  TypedValue<RankedTensorType>
  broadcastScale(PatternRewriter &rewriter, DotScaledOp scaledDotOp,
                 ModuleOp mod, TypedValue<RankedTensorType> scale,
                 int dim) const;
  TypedValue<RankedTensorType> maskNan(PatternRewriter &rewriter,
                                       DotScaledOp scaledDotOp,
                                       TypedValue<RankedTensorType> mxfp,
                                       TypedValue<RankedTensorType> scale,
                                       int dim) const;
  virtual TypedValue<RankedTensorType> scaleArg(PatternRewriter &rewriter,
                                                DotScaledOp scaledDotOp,
                                                int opIdx,
                                                FloatType computeType) const;
  TypedValue<RankedTensorType>
  cvtDotOperand(PatternRewriter &rewriter, DotScaledOp scaledDotOp, int opIdx,
                TypedValue<RankedTensorType> v) const;
  TypedValue<RankedTensorType>
  extendAndBroadcastScale(PatternRewriter &rewriter, DotScaledOp scaledDotOp,
                          TypedValue<RankedTensorType> &scale,
                          FloatType computeType, RankedTensorType dstType,
                          int opIdx) const;
  static SmallVector<int, 2> getTransposeOrder(int rank);
};

>>>>>>> 85e99d639 ([AMD] Enable f16 * mxfp scaled dot decomposition for gfx950 (#7839))
void populateDecomposeScaledBlockedPatterns(mlir::RewritePatternSet &patterns,
                                            int benefit);

} // namespace mlir::triton::gpu
