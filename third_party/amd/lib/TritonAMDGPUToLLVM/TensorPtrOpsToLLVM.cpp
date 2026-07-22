#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TDMUtility.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
// Validates that the tensor descriptor's strides and shared layout are
// compatible with TDM. Requirements:
//  - The shared order must be [rank-1, rank-2, ..., 0].
//  - All stride-1 dimensions must be consecutive trailing dims.
// Additionally, a single stride-1 dimension may appear at the rank-2
// position (col-major) if the shared order has rank-2 and rank-1 swapped.
LogicalResult validateStridesAndSharedOrder(triton::MakeTensorDescOp op,
                                            Attribute sharedEnc,
                                            ArrayRef<int64_t> shape,
                                            ValueRange strides) {
  int rank = shape.size();
  auto sharedOrder = triton::gpu::getOrder(
      cast<triton::gpu::SharedEncodingTrait>(sharedEnc), shape);

  SmallVector<unsigned> strideOneDims;
  for (auto [dim, strideVal] : llvm::enumerate(strides)) {
    if (getConstantIntValue(getAsOpFoldResult(strideVal)).value_or(0) == 1)
      strideOneDims.push_back(dim);
  }

  if (strideOneDims.empty())
    return op.emitError() << "requires at least one dimension to have stride 1";

  // If the only stride-1 dim is the second-to-last dimension (col-major) we can
  // safely reorder the dimensions during lowering.
  bool isColMajor =
      strideOneDims.size() == 1 && strideOneDims.front() == rank - 2;

  SmallVector<unsigned> expectedOrder(llvm::reverse(llvm::seq<unsigned>(rank)));
  if (isColMajor)
    std::swap(expectedOrder[0], expectedOrder[1]);

  if (sharedOrder != ArrayRef(expectedOrder)) {
    if (isColMajor)
      return op.emitError()
             << "requires shared order [rank-2, rank-1, rank-3, "
                "rank-4, ..., 0] because dim[rank-2] has stride 1";
    return op.emitError() << "requires shared order [rank-1, rank-2, ..., 0]";
  }

  if (strideOneDims.size() > 1) {
    unsigned numStride1Dims = strideOneDims.size();
    for (unsigned i = 0; i < numStride1Dims; ++i) {
      if (strideOneDims[i] != rank - numStride1Dims + i)
        return op.emitError() << "requires all stride 1 dimensions to be "
                                 "consecutive starting from the last dimension";
    }
  }

  return success();
}

// Collects all users of the value beyond the basic block boundaries
// defining a given value.
void collectUsers(Value value, llvm::SetVector<Operation *> &users) {
  for (OpOperand &use : value.getUses()) {
    Operation *userOp = use.getOwner();
    if (users.contains(userOp)) {
      // stop recursion; avoid loops
      return;
    }
    users.insert(userOp);
    const unsigned argIdx = use.getOperandNumber();

    if (auto unrealCast = dyn_cast<mlir::UnrealizedConversionCastOp>(userOp)) {
      collectUsers(unrealCast->getResult(argIdx), users);
    }

    if (auto branch = dyn_cast<mlir::BranchOpInterface>(userOp)) {
      auto successors = branch->getSuccessors();
      for (auto [idx, successor] : llvm::enumerate(successors)) {
        auto operands = branch.getSuccessorOperands(idx);
        if (argIdx < operands.size()) {
          collectUsers(successor->getArgument(argIdx), users);
        }
      }
    }
  }
}

struct MakeTensorDescOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorDescOp> {
  using ConvertOpToLLVMPattern<
      triton::MakeTensorDescOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto basePtr = adaptor.getBase();
    auto tensorShape = llvm::to_vector(adaptor.getShape());
    auto tensorStride = llvm::to_vector(adaptor.getStrides());
    auto result = op.getResult();

    auto tensorDescTy = result.getType();
    auto sharedEnc = tensorDescTy.getSharedLayout();
    if (!sharedEnc) {
      if (!sharedEnc)
        return rewriter.notifyMatchFailure(
            op, "Descriptor has no shared memory layout assigned.");
    }
    unsigned padInterval = 0;
    unsigned padAmount = 0;
    if (auto padEnc = getPaddedEncoding(sharedEnc)) {
      if (padEnc.getIntervals().size() != 1 || padEnc.getPaddings().size() != 1)
        return rewriter.notifyMatchFailure(
            op, "NYI: Multiple interval-padding pairs in TDM.");
      padInterval = padEnc.getIntervals()[0];
      padAmount = padEnc.getPaddings()[0];
    }

    Type elementType =
        getTypeConverter()->convertType(tensorDescTy.getElementType());
    SmallVector<int64_t> blockShape = to_vector(tensorDescTy.getShape());
    int numWarps = lookupNumWarps(op);
    auto shapePerCTA = triton::gpu::getShapePerCTA(sharedEnc, blockShape);

    if (failed(validateStridesAndSharedOrder(op, sharedEnc, shapePerCTA,
                                             tensorStride))) {
      return failure();
    }
    auto sharedOrder = triton::gpu::getOrder(
        cast<triton::gpu::SharedEncodingTrait>(sharedEnc), shapePerCTA);
    bool isRowMajor = sharedOrder[0] == (sharedOrder.size() - 1);
    // Create TDM descriptor for 2D-5D tensors
    auto tdmDesc = LLVM::AMD::createTDMDescriptor(
        rewriter, loc, getTypeConverter(), elementType, shapePerCTA, numWarps,
        padInterval, padAmount, tensorShape, tensorStride, basePtr, isRowMajor,
        sharedEnc);

    SmallVector<Value> groups = tdmDesc.getAllGroups();

    auto desc =
        packLLElements(loc, getTypeConverter(), groups, rewriter, tensorDescTy);

    rewriter.replaceOp(op, desc);
    return success();
  }
};

struct UpdateTensorDescriptorOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::UpdateTensorDescriptorOp> {
  using ConvertOpToLLVMPattern<
      triton::amdgpu::UpdateTensorDescriptorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::UpdateTensorDescriptorOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tensorDescTy = op.getDesc().getType();
    auto blockTy = tensorDescTy.getBlockType();
    Type elementType =
        getTypeConverter()->convertType(blockTy.getElementType());
    SmallVector<int64_t> blockShape = to_vector(blockTy.getShape());

    if (blockShape.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "UpdateTensorDescriptorOp lowering currently supports 2D only");
    }

    // Unpack the input descriptor into vector groups (group0: <4 x i32>,
    // group1: <8 x i32> for 2D).
    SmallVector<Value> groups =
        mlir::LLVM::AMD::unpackTDMDescriptor(rewriter, loc, adaptor.getDesc());
    assert(groups.size() == 2 && "2D descriptor expects 2 vector groups");
    Value group0 = groups[0];
    Value group1 = groups[1];

    SmallVector<Value> addOffsets = llvm::to_vector(adaptor.getAddOffsets());
    SmallVector<Value> setBounds = llvm::to_vector(adaptor.getSetBounds());

    Value destPtr;
    if (op.getDest()) {
      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, adaptor.getDest(), elementType, rewriter);
      destPtr = smemObj.getBase();
    }
    Value barrierPtr;
    if (op.getBarrier()) {
      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, adaptor.getBarrier(),
          getTypeConverter()->convertType(
              op.getBarrier().getType().getElementType()),
          rewriter);
      barrierPtr = smemObj.getBase();
    }
    Value pred = adaptor.getPred();

    mlir::LLVM::AMD::updateTensorDescriptor(
        rewriter, loc, elementType, blockShape, group0, group1, addOffsets,
        setBounds, destPtr, pred, barrierPtr);

    // Re-pack the mutated groups back into the flat MLIR struct that
    // matches convertTensorDescType / the host-side TDMDescriptor ABI.
    SmallVector<Value> mutated = {group0, group1};
    SmallVector<Value> scalars =
        mlir::LLVM::AMD::scalarizeTDMDescriptor(rewriter, loc, mutated);
    Value newDesc = packLLElements(loc, getTypeConverter(), scalars, rewriter,
                                   tensorDescTy);

    rewriter.replaceOp(op, newDesc);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateTensorPtrOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MakeTensorDescOpConversion>(typeConverter, benefit);
  patterns.add<UpdateTensorDescriptorOpConversion>(typeConverter, benefit);
  return;
}
