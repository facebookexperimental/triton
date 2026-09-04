#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeRange.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/NVWS/IR/NVWSAttrEnums.cpp.inc"

#include "Dialect/NVWS/IR/NVWSOpInterfaces.cpp.inc"

namespace mlir::triton::nvws {

LogicalResult ArefCreateOp::verify() {
  SmallVector<int> dims;
  for (auto operand : getOperands()) {
    SmallVector<Operation *> users(operand.user_begin(), operand.user_end());
    if (!llvm::all_of(users, [](Operation *op) {
          return isa<ArefCreateOp, gpu::LocalDeallocOp>(op);
        }))
      return emitError("Aref buffer is used elsewhere, Aref cannot guarantee "
                       "async safety");
    auto type = operand.getType();
    if (auto mType = dyn_cast<gpu::MemDescType>(type)) {
      dims.push_back(mType.getShape()[0]);
    } else if (auto rType = dyn_cast<RankedTensorType>(type)) {
      dims.push_back(rType.getShape()[0]);
    } else {
      return emitError("Aref is sliced, but input type isn't supported.");
    }
  }
  if (!llvm::all_equal(dims))
    return emitError("Leading dims of sliced aref inputs don't match.");

  return success();
}

template <typename T>
static std::optional<Twine> verifySlice(T &origType, T &newType) {
  if (!origType || !newType)
    return "MLIR Types don't match";
  if (isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
          origType.getEncoding())) {
    if (origType.getElementType() != newType.getElementType() ||
        origType.getRank() != newType.getRank()) {
      return "Ranks don't match for TensorMemoryScalesEncodingAttr";
    }
    for (size_t i = 0, e = newType.getShape().size(); i < e; i++) {
      if (origType.getShape()[i] != newType.getShape()[i])
        return "Dimensions don't match for TensorMemoryScalesEncodingAttr";
    }
  } else {
    if (origType.getElementType() != newType.getElementType() ||
        origType.getRank() - 1 != newType.getRank()) {
      return "Ranks don't match";
    }
    for (size_t i = 0, e = newType.getShape().size(); i < e; i++) {
      if (origType.getShape()[i + 1] != newType.getShape()[i])
        return "Dimensions don't match";
    }
  }
  return std::nullopt;
}

std::optional<Twine> static arefEnterVerify(
    ArefType aref, mlir::ValueTypeRange<ResultRange> resultTypes) {
  auto typeArray = aref.getBaseType();
  if (typeArray.size() != resultTypes.size())
    return "Aref has different number of arguments than enter";
  // This should probably rely on the memdescSubsliceOp verifier?
  for (auto [orig, arg] : llvm::zip(typeArray, resultTypes)) {
    if (auto origT = dyn_cast<RankedTensorType>(orig)) {
      auto argT = dyn_cast<RankedTensorType>(arg);
      if (auto result = verifySlice(origT, argT))
        return result;
    } else if (auto origT = dyn_cast<triton::gpu::MemDescType>(orig)) {
      auto argT = dyn_cast<triton::gpu::MemDescType>(arg);
      if (auto result = verifySlice(origT, argT))
        return result;
    } else {
      return "Slicing not Implemented for this type";
    }
  }
  return std::nullopt;
}

LogicalResult ArefPutEnterOp::verify() {
  if (auto result =
          arefEnterVerify(getAref().getType(), getBuffers().getType()))
    return emitError(*result);
  return success();
}

LogicalResult ArefGetEnterOp::verify() {
  if (auto result =
          arefEnterVerify(getAref().getType(), getBuffers().getType()))
    return emitError(*result);
  return success();
}

LogicalResult WarpGroupOp::verify() {
  auto numWarps = getNumWarps();
  auto regions = getRegions();
  if (numWarps.size() != regions.size())
    return emitError("Must supply numWarps for each Warp Group.");
  if (getResults().size() > 0) {
    if (regions.size() == 0) {
      return emitError("Must have at least one region when there are results.");
    }
    if (!isa<nvws::WarpGroupYieldOp>(
            regions.front()->front().getTerminator())) {
      return emitError("When nvws.warp_group op has results, the first region "
                       "should be terminated by nvws.warp_group.yield op.");
    }
    auto yieldOp =
        cast<nvws::WarpGroupYieldOp>(regions.front()->front().getTerminator());
    if (getResults().size() != yieldOp.getNumOperands()) {
      return emitError(
          "Mismatch in the number of results returned by nvws.warp_group op "
          "and the number of the operands of the corresponding "
          "nvws.warp_group.yield op in the first region.");
    }
  }
  return success();
}

ParseResult WarpGroupOp::parse(OpAsmParser &p, OperationState &result) {
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  SmallVector<int32_t> partitionNumWarps;
  while (succeeded(p.parseOptionalKeyword(
      ("partition" + Twine(partitionNumWarps.size()).str())))) {
    if (p.parseKeyword("num_warps") || p.parseLParen() ||
        p.parseInteger(partitionNumWarps.emplace_back()) || p.parseRParen() ||
        p.parseRegion(*result.addRegion()))
      return failure();
  }

  result.addAttribute(getNumWarpsAttrName(result.name),
                      p.getBuilder().getDenseI32ArrayAttr(partitionNumWarps));

  return success();
}

void WarpGroupOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     {getNumWarpsAttrName()});

  for (auto [i, region, numWarps] :
       llvm::enumerate(getPartitionRegions(), getNumWarps())) {
    p.printNewline();
    p << "partition" << i;
    p << " num_warps(" << numWarps << ") ";
    p.printRegion(region, /*printEntryBlockArgs=*/false);
  }
}

void CreateTokenOp::build(::mlir::OpBuilder &builder,
                          ::mlir::OperationState &state, uint32_t num,
                          TokenLoadType loadType) {
  auto tokenType = TokenType::get(builder.getContext());
  auto resultType = RankedTensorType::get({num}, tokenType);
  build(builder, state, resultType, num, loadType);
}

void ArefPutEnterOp::setStage(Value stage) { getStageMutable().assign(stage); }
void ArefPutExitOp::setStage(Value stage) { getStageMutable().assign(stage); }
void ArefGetExitOp::setStage(Value stage) { getStageMutable().assign(stage); }
void ArefGetEnterOp::setStage(Value stage) { getStageMutable().assign(stage); }
void ArefBufferOp::setStage(Value stage) { getStageMutable().assign(stage); }

void TMAStoreWaitOp::addBarrier(Value barrier, Value pred) {
  getBarriersMutable().append(barrier);
  getBarrierPredsMutable().append(pred);
}

void TMAStoreWaitOp::addToken(Value token, Value idx) {
  getNvwsTokensMutable().append(token);
  getNvwsTokenIndicesMutable().append(idx);
}

// barriers-and-preds := (`,` ssa-value `[` ssa-value `]`)*
static ParseResult
parseBarriersAndPreds(OpAsmParser &p,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &barriers,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &preds) {
  while (succeeded(p.parseOptionalComma())) {
    if (p.parseOperand(barriers.emplace_back()) || p.parseLSquare() ||
        p.parseOperand(preds.emplace_back()) || p.parseRSquare())
      return failure();
  }
  return success();
}

static void printBarriersAndPreds(OpAsmPrinter &p, Operation *op,
                                  OperandRange barriers, OperandRange preds) {
  assert(barriers.size() == preds.size());
  for (auto [barrier, pred] : llvm::zip(barriers, preds))
    p << ", " << barrier << '[' << pred << ']';
}

// nvws-tokens-and-indices := (`nvws_token` ssa-value `[` ssa-value `]`)*
static ParseResult parseNvwsTokensAndIndices(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &tokens,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indices) {
  while (succeeded(p.parseOptionalKeyword("nvws_token"))) {
    if (p.parseOperand(tokens.emplace_back()) || p.parseLSquare() ||
        p.parseOperand(indices.emplace_back()) || p.parseRSquare())
      return failure();
  }
  return success();
}

static void printNvwsTokensAndIndices(OpAsmPrinter &p, Operation *op,
                                      OperandRange tokens,
                                      OperandRange indices) {
  assert(tokens.size() == indices.size());
  for (auto [token, index] : llvm::zip(tokens, indices))
    p << " nvws_token " << token << '[' << index << ']';
}

ParseResult TMAStoreWaitOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  SmallVector<OpAsmParser::UnresolvedOperand> barriers;
  SmallVector<OpAsmParser::UnresolvedOperand> barrierPreds;
  SmallVector<OpAsmParser::UnresolvedOperand> nvwsTokens;
  SmallVector<OpAsmParser::UnresolvedOperand> nvwsTokenIndices;
  Type srcType;
  SmallVector<Type> barrierTypes;
  SmallVector<Type> nvwsTokenTypes;

  if (parser.parseOperand(src) ||
      parseBarriersAndPreds(parser, barriers, barrierPreds) ||
      parseNvwsTokensAndIndices(parser, nvwsTokens, nvwsTokenIndices) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(srcType))
    return failure();

  for (size_t i = 0; i < barriers.size(); ++i) {
    if (parser.parseComma() || parser.parseType(barrierTypes.emplace_back()))
      return failure();
  }
  for (size_t i = 0; i < nvwsTokens.size(); ++i) {
    if (parser.parseComma() || parser.parseType(nvwsTokenTypes.emplace_back()))
      return failure();
  }

  auto loc = parser.getCurrentLocation();
  auto &builder = parser.getBuilder();
  if (parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperands(barriers, barrierTypes, loc, result.operands) ||
      parser.resolveOperands(barrierPreds, builder.getI1Type(),
                             result.operands) ||
      parser.resolveOperands(nvwsTokens, nvwsTokenTypes, loc,
                             result.operands) ||
      parser.resolveOperands(nvwsTokenIndices, builder.getI32Type(),
                             result.operands))
    return failure();

  result.addAttribute(TMAStoreWaitOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(barriers.size()),
                           static_cast<int32_t>(barrierPreds.size()),
                           static_cast<int32_t>(nvwsTokens.size()),
                           static_cast<int32_t>(nvwsTokenIndices.size())}));
  return success();
}

void TMAStoreWaitOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getSrc();
  printBarriersAndPreds(printer, *this, getBarriers(), getBarrierPreds());
  printNvwsTokensAndIndices(printer, *this, getNvwsTokens(),
                            getNvwsTokenIndices());
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {getOperandSegmentSizeAttr()});
  printer << " : " << getSrc().getType();
  for (Value barrier : getBarriers())
    printer << ", " << barrier.getType();
  for (Value token : getNvwsTokens())
    printer << ", " << token.getType();
}

} // namespace mlir::triton::nvws

#define GET_OP_CLASSES
#include "Dialect/NVWS/IR/Ops.cpp.inc"
