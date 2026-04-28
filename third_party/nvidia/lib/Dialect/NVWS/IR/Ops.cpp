#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeRange.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/NVWS/IR/NVWSAttrEnums.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/NVWS/IR/NVWSOpInterfaces.cpp.inc"
#include "Dialect/NVWS/IR/Ops.cpp.inc"

namespace mlir::triton::nvws {

static LogicalResult verifyNoDuplicateAsyncOps(Operation *op,
                                               ArrayAttr asyncOps) {
  llvm::DenseSet<AsyncOp> seen;
  for (Attribute attr : asyncOps) {
    auto asyncAttr = dyn_cast<AsyncOpAttr>(attr);
    if (!asyncAttr)
      return op->emitError("async_ops must be an array of #nvws.async_op");
    if (!seen.insert(asyncAttr.getValue()).second)
      return op->emitError("async_ops contains duplicate async kind");
  }
  return success();
}

static bool hasProtocolUsers(SemaphoreCreateOp semaphoreCreate) {
  return !semaphoreCreate.getResult().use_empty();
}

static bool allLegalSemaphoreBackingUses(Value value,
                                         llvm::DenseSet<Operation *> &seen);

static bool isLegalSemaphoreBackingUse(Operation *user,
                                       llvm::DenseSet<Operation *> &seen) {
  if (isa<SemaphoreCreateOp, gpu::LocalDeallocOp>(user))
    return true;
  if (!user->hasTrait<OpTrait::MemDescViewTrait>() &&
      !isa<triton::nvidia_gpu::TMEMSubSliceOp>(user))
    return false;
  if (!seen.insert(user).second)
    return true;
  return llvm::all_of(user->getResults(), [&](Value result) {
    return allLegalSemaphoreBackingUses(result, seen);
  });
}

static bool allLegalSemaphoreBackingUses(Value value,
                                         llvm::DenseSet<Operation *> &seen) {
  return llvm::all_of(value.getUsers(), [&](Operation *user) {
    return isLegalSemaphoreBackingUse(user, seen);
  });
}

static LogicalResult
verifySharedBufferPeerTupleInvariant(SemaphoreCreateOp semaphoreCreate) {
  if (!hasProtocolUsers(semaphoreCreate))
    return success();

  SmallVector<Value> buffers(semaphoreCreate.getBuffers().begin(),
                             semaphoreCreate.getBuffers().end());
  int numStages = semaphoreCreate.getType().getNumStages();
  llvm::DenseSet<Operation *> seenPeers;

  for (Value buffer : buffers) {
    for (Operation *user : buffer.getUsers()) {
      auto peer = dyn_cast<SemaphoreCreateOp>(user);
      if (!peer || peer == semaphoreCreate)
        continue;
      if (!seenPeers.insert(user).second)
        continue;

      auto peerBuffers = peer.getBuffers();
      if (peerBuffers.size() != buffers.size()) {
        return semaphoreCreate.emitError(
            "semaphores sharing a backing buffer must use identical ordered "
            "buffer operands");
      }
      for (auto [lhs, rhs] : llvm::zip(buffers, peerBuffers)) {
        if (lhs != rhs) {
          return semaphoreCreate.emitError(
              "semaphores sharing a backing buffer must use identical ordered "
              "buffer operands");
        }
      }
    }
  }

  return success();
}

LogicalResult SemaphoreReleaseOp::verify() {
  if (auto count = getArriveCountAttr())
    if (count.getInt() < 1)
      return emitError("arrive_count must be >= 1, got ") << count.getInt();
  return verifyNoDuplicateAsyncOps(getOperation(), getAsyncOps());
}

LogicalResult SemaphoreCreateOp::verify() {
  SmallVector<int64_t> dims;

  for (auto operand : getOperands()) {
    llvm::DenseSet<Operation *> seen;
    if (!allLegalSemaphoreBackingUses(operand, seen)) {
      return emitError("Semaphore buffer is used elsewhere, Semaphore cannot "
                       "guarantee async safety");
    }

    Type type = operand.getType();
    if (auto memTy = dyn_cast<triton::gpu::MemDescType>(type)) {
      auto shape = memTy.getShape();
      if (shape.empty())
        return emitError("Semaphore is sliced, but input type has empty shape");
      dims.push_back(shape.front());
    } else if (auto rankedTy = dyn_cast<RankedTensorType>(type)) {
      auto shape = rankedTy.getShape();
      if (shape.empty())
        return emitError("Semaphore is sliced, but input type has empty shape");
      dims.push_back(shape.front());
    } else {
      return emitError("Semaphore is sliced, but input type isn't supported");
    }
  }

  if (!dims.empty() && !llvm::all_equal(dims))
    return emitError("Leading dims of sliced semaphore inputs don't match");

  if (failed(verifySharedBufferPeerTupleInvariant(*this)))
    return failure();

  for (Operation *user : getResult().getUsers()) {
    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp)
      continue;

    if (failed(verifyNoDuplicateAsyncOps(releaseOp, releaseOp.getAsyncOps())))
      return failure();
  }

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

static std::optional<Twine>
verifySemaphoreBuffer(SemaphoreType semaphore,
                      mlir::ValueTypeRange<ResultRange> resultTypes) {
  auto typeArray = semaphore.getBaseType();
  if (typeArray.size() != resultTypes.size())
    return "Semaphore has different number of arguments than buffer";

  for (auto [orig, resultTy] : llvm::zip(typeArray, resultTypes)) {
    if (auto origT = dyn_cast<RankedTensorType>(orig)) {
      auto resultT = dyn_cast<RankedTensorType>(resultTy);
      if (auto verifyResult = verifySlice(origT, resultT))
        return verifyResult;
    } else if (auto origT = dyn_cast<triton::gpu::MemDescType>(orig)) {
      auto resultT = dyn_cast<triton::gpu::MemDescType>(resultTy);
      if (auto verifyResult = verifySlice(origT, resultT))
        return verifyResult;
      if (!resultT.getMutableMemory())
        return "Semaphore buffer result memdesc must be mutable";
    } else {
      return "Slicing not implemented for this type";
    }
  }

  return std::nullopt;
}

LogicalResult SemaphoreBufferOp::verify() {
  if (auto verifyResult = verifySemaphoreBuffer(getSemaphore().getType(),
                                                getBuffers().getType()))
    return emitError(*verifyResult);
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
  auto ctx = p.getBuilder().getContext();

  SMLoc operandLoc = p.getCurrentLocation();
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  SmallVector<int32_t> partitionNumWarps;
  while (succeeded(p.parseOptionalKeyword(
      ("partition" + Twine(partitionNumWarps.size()).str())))) {
    SMLoc regionLoc = p.getCurrentLocation();
    if (p.parseKeyword("num_warps") || p.parseLParen() ||
        p.parseInteger(partitionNumWarps.emplace_back()) || p.parseRParen() ||
        p.parseRegion(*result.addRegion()))
      return failure();
  }

  result.addAttribute(getNumWarpsAttrName(result.name),
                      p.getBuilder().getDenseI32ArrayAttr(partitionNumWarps));

  if (!result.regions.empty() && !result.regions.front()->empty()) {
    Operation *terminator = result.regions.front()->front().getTerminator();
    if (auto yieldOp = dyn_cast<WarpGroupYieldOp>(terminator))
      result.addTypes(yieldOp.getOperandTypes());
  }

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

ParseResult SemaphoreAcquireOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::UnresolvedOperand semaphore;
  OpAsmParser::UnresolvedOperand stage;
  OpAsmParser::UnresolvedOperand phase;
  bool hasStage = false;
  bool hasPhase = false;
  SemaphoreType semaphoreType;
  ::mlir::triton::gpu::AsyncTokenType tokenType;

  if (parser.parseOperand(semaphore))
    return failure();
  if (succeeded(parser.parseOptionalLSquare())) {
    hasStage = true;
    if (parser.parseOperand(stage))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      hasPhase = true;
      if (parser.parseOperand(phase))
        return failure();
    }
    if (parser.parseRSquare())
      return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(semaphoreType) ||
      parser.parseArrow() || parser.parseCustomTypeWithFallback(tokenType))
    return failure();

  Builder &builder = parser.getBuilder();
  if (parser.resolveOperand(semaphore, semaphoreType, result.operands))
    return failure();
  Type i32Type = builder.getI32Type();
  if (hasStage &&
      parser.resolveOperand(stage, i32Type, result.operands))
    return failure();
  if (hasPhase &&
      parser.resolveOperand(phase, i32Type, result.operands))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      builder.getDenseI32ArrayAttr(
                          {1, hasStage ? 1 : 0, hasPhase ? 1 : 0}));
  result.addTypes(tokenType);
  return success();
}

void SemaphoreAcquireOp::print(OpAsmPrinter &p) {
  p << " " << getSemaphore();
  if (getStage()) {
    p << "[" << getStage();
    if (getPhase())
      p << ", " << getPhase();
    p << "]";
  }
  p.printOptionalAttrDict((*this)->getAttrs(), {getOperandSegmentSizesAttrName()});
  p << " : ";
  Type semaphoreType = getSemaphore().getType();
  if (auto validType = dyn_cast<SemaphoreType>(semaphoreType))
    p.printStrippedAttrOrType(validType);
  else
    p << semaphoreType;
  p << " -> ";
  Type tokenType = getToken().getType();
  if (auto validType = dyn_cast<::mlir::triton::gpu::AsyncTokenType>(tokenType))
    p.printStrippedAttrOrType(validType);
  else
    p << tokenType;
}

void SemaphoreAcquireOp::setStage(Value stage) {
  getStageMutable().assign(stage);
}
void SemaphoreReleaseOp::setStage(Value stage) {
  getStageMutable().assign(stage);
}
void SemaphoreBufferOp::setStage(Value stage) {
  getStageMutable().assign(stage);
}

} // namespace mlir::triton::nvws
