#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::gpu {

class DecomposeScaledBlocked : public OpRewritePattern<DotScaledOp> {
public:
  DecomposeScaledBlocked(MLIRContext *context, PatternBenefit benefit,
                         bool allowUE5M3Scales = false)
      : OpRewritePattern<DotScaledOp>(context, benefit),
        allowUE5M3Scales(allowUE5M3Scales) {}

  LogicalResult matchAndRewrite(DotScaledOp scaledDotOp,
                                PatternRewriter &rewriter) const override;

protected:
  // UE5M3 (int8 block-16) scales exist only on NV: AMD treats int8 scales as
  // E8M0 at any block size. Only the NV populate call site opts in.
  bool allowUE5M3Scales;

  FloatType getComputeType(ScaleDotElemType aType, ScaleDotElemType bType,
                           PatternRewriter &rewriter) const;
  TypedValue<RankedTensorType> scaleTo16(PatternRewriter &rewriter,
                                         TypedValue<RankedTensorType> scale,
                                         FloatType computeType) const;
  TypedValue<RankedTensorType>
  decodeUE5M3To16(PatternRewriter &rewriter, TypedValue<RankedTensorType> scale,
                  FloatType computeType) const;
  TypedValue<RankedTensorType>
  broadcastScale(PatternRewriter &rewriter, DotScaledOp scaledDotOp,
                 TypedValue<RankedTensorType> scale, int dim,
                 Attribute dstEncoding) const;
  TypedValue<RankedTensorType> maskNan(PatternRewriter &rewriter,
                                       DotScaledOp scaledDotOp,
                                       TypedValue<RankedTensorType> mxfp,
                                       TypedValue<RankedTensorType> scale,
                                       int dim) const;
  virtual TypedValue<RankedTensorType> scaleArg(PatternRewriter &rewriter,
                                                DotScaledOp scaledDotOp,
                                                int opIdx,
                                                FloatType computeType) const;
  TypedValue<RankedTensorType> upcastFp6(PatternRewriter &rewriter,
                                         DotScaledOp scaledDotOp, int opIdx,
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

void populateDecomposeScaledBlockedPatterns(mlir::RewritePatternSet &patterns,
                                            int benefit,
                                            bool allowUE5M3Scales = false);

} // namespace mlir::triton::gpu
