#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/Transforms/InliningUtils.h"
#include "triton/Dialect/Triton/IR/AttrInterfaces.cpp.inc"
#include "triton/Dialect/Triton/IR/Dialect.cpp.inc"
#include "triton/Dialect/Triton/IR/OpInterfaces.cpp.inc"

using namespace mlir;
using namespace mlir::triton;

//===----------------------------------------------------------------------===//
// TritonDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TritonInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    auto funcOp = dyn_cast<triton::FuncOp>(callable);
    if (!funcOp)
      return true;
    if (funcOp->hasAttr("noinline"))
      return !funcOp->getAttrOfType<BoolAttr>("noinline").getValue();
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       IRMapping &) const final {
    return true;
  }
  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, Block *newDest) const final {
    // Only return needs to be handled here.
    auto returnOp = dyn_cast<triton::ReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    OpBuilder builder(op);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest,
                                       returnOp.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // Only return needs to be handled here.
    auto returnOp = cast<triton::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};

} // namespace

///--- SparseDotOp ---
// TODO(sparsity) maybe replace with linear layouts, or otherwise clean up
namespace {
// Implied properties of 2:4 sparse dots.
constexpr int kContractingFactor = 2;
constexpr int kMetadataElementsPerPackedValue = 8;
constexpr int kMetadataElementsPerWarp = 16;
} // namespace

mlir::LogicalResult SparseDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc = cast<RankedTensorType>(operands[0].getType()).getEncoding();
  auto bEnc = cast<RankedTensorType>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();

  return success();
}

// TODO: had to disable verifier b/c TensorOrMemDesc is TTGIR-only
/*
LogicalResult SparseDotOp::verify() {
  // Verify operand A.
  auto aTensorTy = cast<TensorOrMemDesc>(getOperand(0).getType());
  auto aElemTy = aTensorTy.getElementType();
  if (!aElemTy.isF16() && !aElemTy.isBF16())
    return emitError("element type of operand A is not supported");
  auto aShape = aTensorTy.getShape();
  if (aShape.size() != 2)
    return emitError("shape of operand A is incorrect");

  // Verify operand B.
  auto bTensorTy = cast<TensorOrMemDesc>(getOperand(1).getType());
  auto bElemTy = bTensorTy.getElementType();
  if (!bElemTy.isF16() && !bElemTy.isBF16())
    return emitError("element type of operand B is not supported");
  auto bShape = bTensorTy.getShape();
  if (bShape.size() != 2)
    return emitError("shape of operand B is incorrect");

  // Verify operand C.
  auto cTensorTy = cast<RankedTensorType>(getOperand(2).getType());
  auto cElemTy = cTensorTy.getElementType();
  if (!cElemTy.isF32())
    return emitError("element type of operand C is not supported");
  auto cShape = cTensorTy.getShape();
  if (cShape.size() != 2)
    return emitError("shape of operand C is incorrect");

  // Check operand dependencies.
  if (aShape[0] != cShape[0] || bShape[1] != cShape[1] ||
      bShape[0] != aShape[1] * kContractingFactor)
    return emitError("operand shape dimensions are incorrect");
  if (aElemTy != bElemTy)
    return emitError("operand element types do not match");

  // Verify sparse metadata.
  auto metaTy = cast<RankedTensorType>(getOperand(3).getType());
  auto metaShape = metaTy.getShape();
  if (!metaTy.getElementType().isInteger(16) || metaShape.size() != 2)
    return emitError("sparse metadata tensor is invalid");
  if (metaShape[0] != aShape[0] ||
      metaShape[1] * kMetadataElementsPerPackedValue != aShape[1])
    return emitError("sparse metadata shape dimensions are incorrect");

  // Verify tensor encoding.
  auto aEncoding = aTensorTy.getEncoding();
  auto bEncoding = bTensorTy.getEncoding();
  if (!aEncoding && !bEncoding)
    return mlir::success();
  if (!aEncoding || !bEncoding)
    return emitError("mismatching encoding between A and B operands");

  Dialect &dialect = aEncoding.getDialect();
  auto interface = cast<DialectInferLayoutInterface>(&dialect);
  return interface->verifyDotOpEncodingCompatibility(getOperation(), aEncoding,
                                                     bEncoding);
}
*/

void TritonDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"
      >();

  // We can also add interface here.
  addInterfaces<TritonInlinerInterface>();
}

Operation *TritonDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
