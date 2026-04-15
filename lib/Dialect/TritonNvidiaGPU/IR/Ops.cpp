/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOpInterfaces.cpp.inc"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

LogicalResult MapToRemoteBufferOp::verify() {
  // src and result should have the same type except MemorySpace
  MemDescType localType = getSrc().getType();
  MemDescType remoteType = getResult().getType();
  if (!(localType.getShape() == remoteType.getShape() &&
        localType.getElementType() == remoteType.getElementType() &&
        localType.getEncoding() == remoteType.getEncoding() &&
        localType.getMutableMemory() == remoteType.getMutableMemory() &&
        localType.getAllocShape() == remoteType.getAllocShape())) {
    return emitOpError() << "Local MemDesc not matching Remote MemDesc: "
                         << localType << " vs " << remoteType;
  }
  if (!isa<SharedMemorySpaceAttr>(localType.getMemorySpace())) {
    return emitOpError() << "Invalid memory space for local MemDesc: "
                         << localType;
  }
  if (!isa<SharedClusterMemorySpaceAttr>(remoteType.getMemorySpace())) {
    return emitOpError() << "Invalid memory space for remote MemDesc: "
                         << remoteType;
  }
  return success();
}

// -- WarpGroupDotOp --
LogicalResult WarpGroupDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc = cast<TensorOrMemDesc>(operands[0].getType()).getEncoding();
  auto bEnc = cast<MemDescType>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    Dialect &dialect = aEnc.getDialect();
    auto interface = cast<DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return failure();
  }
  return success();
}

LogicalResult WarpGroupDotOp::verify() {
  auto resTy = getD().getType();
  auto nvmmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(resTy.getEncoding());
  if (!nvmmaEnc || !nvmmaEnc.isHopper())
    return emitOpError("WGMMA result layout must be Hopper NVMMA");

  if (!isa<NVMMASharedEncodingAttr, DotOperandEncodingAttr,
           SharedLinearEncodingAttr>(getA().getType().getEncoding()))
    return emitOpError("WGMMA A operand must have NVMMA shared or dot layout");
  if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(
          getB().getType().getEncoding()))
    return emitOpError("WGMMA B operand must have NVMMA shared layout");

  auto numWarps = gpu::lookupNumWarps(getOperation());
  if (numWarps % 4)
    return emitOpError("WGMMA requires num_warps to be divisible by 4");

  auto retShapePerCTA = getShapePerCTA(resTy);
  int rank = retShapePerCTA.size();
  if (rank != 2)
    return emitOpError("WGMMA result shape must be 2D");
  if (retShapePerCTA[0] % 64 != 0)
    return emitOpError("WGMMA result M dimension must be divisible by 64");
  if (retShapePerCTA[1] % 8 != 0)
    return emitOpError("WGMMA result N dimension must be divisible by 8");

  // Verify MMA version is supported for operands.
  int mmaVersion = nvmmaEnc.getVersionMajor();
  if (!supportMMA(getA(), mmaVersion) || !supportMMA(getB(), mmaVersion))
    return emitOpError("unsupported MMA version for the given operands");

  auto aElemTy = getA().getType().getElementType();
  if (getMaxNumImpreciseAcc() < 32 &&
      (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(aElemTy)) &&
      resTy.getElementType().isF32()) {
    return emitOpError("Cannot use F32 as the accumulator element type when "
                       "the max_num_imprecise_acc is less than 32");
  }

  if (auto aTensorTy = dyn_cast<RankedTensorType>(getA().getType())) {
    auto aDotOpEnc = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    unsigned kWidth = 32 / aTensorTy.getElementTypeBitWidth();
    if (aDotOpEnc.getKWidth() != kWidth) {
      return emitOpError("in-register LHS operand must have a kWidth of ")
             << kWidth << " but got " << aDotOpEnc.getKWidth();
    }
  }

  return success();
}

void WarpGroupDotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto &a = getAMutable();
  auto &b = getBMutable();
  if (isa<MemDescType>(a.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &a, SharedMemory::get());
  if (isa<MemDescType>(b.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &b, SharedMemory::get());
}

bool WarpGroupDotOp::needsPartialAccumulator() {
  const auto &a = getA();
  const auto &d = getD();
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto aElTy = cast<triton::gpu::TensorOrMemDesc>(a.getType()).getElementType();
  bool isFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                         Float8E4M3FNUZType>(aElTy);
  bool accFP32 =
      cast<triton::gpu::TensorOrMemDesc>(d.getType()).getElementType().isF32();
  uint32_t maxNumImpreciseAcc = getMaxNumImpreciseAcc();
  return isFP8 && accFP32 && maxNumImpreciseAcc <= aTensorTy.getShape()[1];
}

bool WarpGroupDotOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

// -- WarpGroupDotWaitOp --
LogicalResult WarpGroupDotWaitOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  for (Value operand : operands)
    inferredReturnTypes.push_back(operand.getType());
  return success();
}

LogicalResult WarpGroupDotWaitOp::verify() {
  if (getOperands().empty())
    return emitOpError("expected to be waiting on at least one dependency");
  return success();
}

// -- InitBarrierOp --
LogicalResult InitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- InvalBarrierOp --
LogicalResult InvalBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- FenceMBarrierInitReleaseClusterOp --
LogicalResult FenceMBarrierInitReleaseClusterOp::verify() {
  // FB: comment out these because we allow the op in frontend/ttir, where the
  // ir does not have tlx cluster dim yet int numCTAs =
  // triton::gpu::lookupNumCTAs(getOperation()); if (numCTAs <= 1)
  //   return emitOpError("requires ttg.num-ctas > 1");
  return success();
}

// -- ClusterArriveOp --
LogicalResult ClusterArriveOp::verify() {
  // FB: comment out these because we allow the op in frontend/ttir, where the
  // ir does not have tlx cluster dim yet int numCTAs =
  // triton::gpu::lookupNumCTAs(getOperation()); if (numCTAs <= 1)
  //   return emitOpError("requires ttg.num-ctas > 1");
  return success();
}

// -- ClusterWaitOp --
LogicalResult ClusterWaitOp::verify() {
  // FB: comment out these because we allow the op in frontend/ttir, where the
  // ir does not have tlx cluster dim yet int numCTAs =
  // triton::gpu::lookupNumCTAs(getOperation()); if (numCTAs <= 1)
  //   return emitOpError("requires ttg.num-ctas > 1");
  return success();
}

// -- BarrierExpectOp --
LogicalResult BarrierExpectOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- WaitBarrierOp --
LogicalResult WaitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- ArriveBarrierOp --
LogicalResult ArriveBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  return success();
}

// -- VoteBallotSyncOp --
LogicalResult VoteBallotSyncOp::verify() {
  Type predType = getPred().getType();
  Type resultType = getResult().getType();

  bool predIsTensor = isa<RankedTensorType>(predType);
  bool resultIsTensor = isa<RankedTensorType>(resultType);

  // Both must be scalars or both must be tensors
  if (predIsTensor != resultIsTensor) {
    return emitOpError("predicate and result must both be scalars or both be "
                       "tensors, got pred=")
           << predType << " and result=" << resultType;
  }

  if (predIsTensor) {
    auto predTensorType = cast<RankedTensorType>(predType);
    auto resultTensorType = cast<RankedTensorType>(resultType);

    // Check element types
    if (!predTensorType.getElementType().isInteger(1)) {
      return emitOpError("tensor predicate must have i1 element type, got ")
             << predTensorType.getElementType();
    }
    if (!resultTensorType.getElementType().isInteger(32)) {
      return emitOpError("tensor result must have i32 element type, got ")
             << resultTensorType.getElementType();
    }

    // Shapes must match
    if (predTensorType.getShape() != resultTensorType.getShape()) {
      return emitOpError("predicate and result tensor shapes must match, got ")
             << predTensorType.getShape() << " vs "
             << resultTensorType.getShape();
    }

    // Encodings must match (if present)
    if (predTensorType.getEncoding() != resultTensorType.getEncoding()) {
      return emitOpError(
                 "predicate and result tensor encodings must match, got ")
             << predTensorType.getEncoding() << " vs "
             << resultTensorType.getEncoding();
    }
  } else {
    // Scalar case
    if (!predType.isInteger(1)) {
      return emitOpError("scalar predicate must be i1, got ") << predType;
    }
    if (!resultType.isInteger(32)) {
      return emitOpError("scalar result must be i32, got ") << resultType;
    }
  }

  return success();
}

template <typename TOp>
LogicalResult verifyTMAEncoding(TOp *op, Value desc, Attribute enc) {
  auto nvmma = dyn_cast<NVMMASharedEncodingAttr>(enc);
  if (!nvmma)
    return op->emitOpError("TMA descriptor must have NVMMA shared layout");
  auto descTy = cast<TensorDescType>(desc.getType());
  auto descBlockEnc = descTy.getBlockType().getEncoding();
  // If the descriptor has no encoding yet (e.g., before
  // optimize-descriptor-encoding pass), skip the match check.
  if (descBlockEnc) {
    auto descEnc = dyn_cast<NVMMASharedEncodingAttr>(descBlockEnc);
    // NOTE: Cannot do descEnc != enc as the encodings may differ in rank for
    // rank-reducing loads
    if (!descEnc || descEnc.getTransposed() != nvmma.getTransposed() ||
        descEnc.getSwizzlingByteWidth() != nvmma.getSwizzlingByteWidth() ||
        descEnc.getElementBitWidth() != nvmma.getElementBitWidth() ||
        descEnc.getFp4Padded() != nvmma.getFp4Padded())
      return op->emitOpError("TMA descriptor layout must match shared layout");
  }
  if (nvmma.getTransposed())
    return op->emitOpError("TMA descriptor layout must not be transposed");
  return success();
}

// -- AsyncTMACopyGlobalToLocalOp --
LogicalResult AsyncTMACopyGlobalToLocalOp::verify() {
  if (failed(verifyBarrierType(*this, getBarrier().getType())))
    return failure();
  if (getCoord().size() < 1 || getCoord().size() > 5)
    return emitOpError("TMA copies must have between 1 and 5 coordinates");
  auto resultType = getResult().getType();
  if (!resultType.getMutableMemory())
    return emitOpError("Cannot store into immutable memory");
  return verifyTMAEncoding(this, getDesc(), resultType.getEncoding());
}

// -- AsyncTMACopyLocalToGlobalOp --
LogicalResult AsyncTMACopyLocalToGlobalOp::verify() {
  return verifyTMAEncoding(this, getDesc(), getSrc().getType().getEncoding());
}

// -- AsyncTMAGatherOp --
LogicalResult AsyncTMAGatherOp::verify() {
  if (failed(verifyBarrierType(*this, getBarrier().getType())))
    return failure();

  triton::gpu::MemDescType resultType = getResult().getType();
  if (!resultType.getMutableMemory())
    return emitOpError("cannot store into immutable memory");
  if (failed(verifyTMAEncoding(this, getDesc(), resultType.getEncoding())))
    return failure();
  return DescriptorGatherOp::verifyResultType(*this, resultType,
                                              getXOffsets().getType());
}

// -- AsyncTMAScatter --
LogicalResult AsyncTMAScatterOp::verify() {
  auto srcType = getSrc().getType();
  if (failed(verifyTMAEncoding(this, getDesc(), srcType.getEncoding())))
    return failure();
  return DescriptorGatherOp::verifyResultType(*this, srcType,
                                              getXOffsets().getType());
}

// -- TCGen5MMAOp --

// barrier-and-pred := `,` ssa-value `[` ssa-value `]`
// barriers-and-preds := (barrier-and-pred)*
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
  for (auto [barrier, pred] : llvm::zip(barriers, preds)) {
    p << ", " << barrier << '[' << pred << ']';
  }
}

// token := `[` (ssa-value (`,` ssa-value)*)? `]`
// dep-operand := token?
static ParseResult
parseToken(OpAsmParser &p, std::optional<OpAsmParser::UnresolvedOperand> &dep,
           Type &token) {
  if (failed(p.parseOptionalLSquare()))
    return success();
  token = p.getBuilder().getType<AsyncTokenType>();
  if (succeeded(p.parseOptionalRSquare()))
    return success();
  if (p.parseOperand(dep.emplace()) || p.parseRSquare())
    return failure();
  return success();
}
static void printToken(OpAsmPrinter &p, Operation *op, Value dep, Type token) {
  if (!token)
    return;
  p << '[';
  if (dep)
    p << dep;
  p << ']';
}

namespace {
enum class MMADTypeKind { tf32, f16, f8f6f4, i8 };
} // namespace

static std::string strMMADTypeKind(MMADTypeKind kind) {
  switch (kind) {
  case MMADTypeKind::tf32:
    return "tf32";
  case MMADTypeKind::f16:
    return "f16";
  case MMADTypeKind::f8f6f4:
    return "f8f6f4";
  case MMADTypeKind::i8:
    return "i8";
  }
  llvm_unreachable("unknown mma dtype kind");
}

static std::optional<std::pair<MMADTypeKind, SmallVector<Type>>>
getMMAv5DTypeKindAndAcc(Type t) {
  MLIRContext *ctx = t.getContext();
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-kind-shapes
  if (t.isF32()) {
    return {{MMADTypeKind::tf32, {Float32Type::get(ctx)}}};
  }
  if (t.isF16()) {
    return {
        {MMADTypeKind::f16, {Float16Type::get(ctx), Float32Type::get(ctx)}}};
  }
  if (t.isBF16()) {
    return {{MMADTypeKind::f16, {Float32Type::get(ctx)}}};
  }
  // TODO: float6 and explicit float4 types are not supported yet.
  // TODO: tcgen05.mma supports ui8/si8 -> s32 MMA, but Triton does not.
  // FIXME: i8 is used to represent float4 types.
  if (isa<Float8E4M3FNType, Float8E5M2Type>(t) || t.isInteger(8)) {
    return {
        {MMADTypeKind::f8f6f4, {Float16Type::get(ctx), Float32Type::get(ctx)}}};
  }
  return std::nullopt;
}

static LogicalResult verifyMMADType(Operation *op, Type a, Type b, Type d) {
  auto akind = getMMAv5DTypeKindAndAcc(a);
  auto bkind = getMMAv5DTypeKindAndAcc(b);
  if (!akind)
    return op->emitOpError("unsupported LHS operand dtype: ") << a;
  if (!bkind)
    return op->emitOpError("unsupported RHS operand dtype: ") << b;
  if (akind->first != bkind->first) {
    return op->emitOpError(
               "LHS and RHS operand dtypes kinds don't match: LHS kind is ")
           << strMMADTypeKind(akind->first) << " but RHS kind is "
           << strMMADTypeKind(bkind->first);
  }
  if (!llvm::is_contained(akind->second, d) ||
      !llvm::is_contained(bkind->second, d)) {
    InFlightDiagnostic diag =
        op->emitOpError("unsupported accumulator dtype for operand types ")
        << a << " and " << b << ", accumulator dtype is " << d
        << " but must be one of [";
    llvm::interleaveComma(akind->second, diag, [&](Type t) { diag << t; });
    diag << "]";
    return diag;
  }
  return success();
}

LogicalResult TCGen5MMAOp::verify() {
  if (!getIsAsync() && !getBarriers().empty()) {
    return emitOpError("The op is synchronous but a barrier is present.");
  }
  Type atype = getA().getType().getElementType();
  Type btype = getB().getType().getElementType();
  Type dtype = getD().getType().getElementType();
  if (failed(verifyMMADType(*this, atype, btype, dtype)))
    return failure();

  auto aEnc = getA().getType().getEncoding();
  if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr,
           TensorMemoryEncodingAttr>(aEnc))
    return emitOpError(
        "LHS operand must have a NVMMAShared or TensorMemory encoding");
  auto bEnc = getB().getType().getEncoding();
  if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(bEnc))
    return emitOpError("RHS operand must have a NVMMAShared encoding");
  auto retType = getD().getType();
  auto retEnc = dyn_cast<TensorMemoryEncodingAttr>(retType.getEncoding());
  if (!retEnc)
    return emitOpError("Return operand must have a TensorMemory encoding");

  // Check colStride of TMEM operands
  if (auto tmem = dyn_cast<TensorMemoryEncodingAttr>(aEnc)) {
    if (tmem.getColStride() != 1)
      return emitOpError("The col stride of the LHS operand must be 1");
  }
  if (retEnc.getColStride() != 32 / retType.getElementTypeBitWidth())
    return emitOpError("The col stride of the return operand must be 32 / ")
           << retType.getElementTypeBitWidth() << " but got "
           << retEnc.getColStride();
  // The maximum size of a MMA instruction is 128x256
  if (retEnc.getBlockN() > 256)
    return emitOpError("The block size of the return operand must be less than "
                       "or equal to 256");

  // if (getTwoCtas()) {
  // Once we have a `block` dimension in TMEM, we can look at this via the
  // associated LL
  // NOTE(TLX): CTASplitNum verification is disabled because TLX two-CTA
  // mode intentionally keeps shared memory CTASplitNum as [1,1] to avoid
  // triggering upstream CTA distribution passes (PlanCTA, AccelerateMatmul).
  // The upstream checks require {2,1} for LHS, {1,2} for RHS, and {2,1}
  // for the return value, which is incompatible with TLX's approach.
  // TODO: Re-enable once TLX adopts upstream's CGAEncodingAttr convention.
  //
  // auto checkSplitNum = [&](ArrayRef<unsigned> splitNum,
  //                          std::string_view name,
  //                          ArrayRef<unsigned> expected) -> LogicalResult {
  //   if (splitNum != expected) {
  //     return emitOpError("The op is two CTAs but the split num of the ")
  //            << name << " is not " << expected << ". Got " << splitNum;
  //   }
  //   return success();
  // };
  // if (failed(checkSplitNum(getCTASplitNum(aEnc), "LHS", {2, 1})))
  //   return failure();
  // if (failed(checkSplitNum(getCTASplitNum(bEnc), "RHS", {1, 2})))
  //   return failure();
  // if (failed(checkSplitNum(getCTASplitNum(retEnc), "returned value",
  //                          {2, 1})))
  //   return failure();

  // NOTE(TLX): twoCTAs encoding checks disabled — TLX does not propagate
  // twoCTAs into TensorMemoryEncodingAttr. See comment above.
  // if (!retEnc.getTwoCTAs())
  //   return emitOpError(
  //       "The returned value's encoding must have twoCTA=true to be used "
  //       "in a twoCTA matmul");
  // if (auto tmemEnc = dyn_cast<TensorMemoryEncodingAttr>(aEnc)) {
  //   if (!tmemEnc.getTwoCTAs())
  //     return emitOpError(
  //         "The LHS operand's encoding must have twoCTA=true to be used "
  //         "in a twoCTA matmul");
  // }
  // }

  return success();
}

void TCGen5MMAOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The op reads the accumulator if `useD` is not known to be false.
  APInt useD;
  if (!matchPattern(getUseD(), m_ConstantInt(&useD)) || !useD.isZero()) {
    effects.emplace_back(MemoryEffects::Read::get(), &getDMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), &getDMutable(),
                       TensorMemory::get());

  if (isa<SharedMemorySpaceAttr>(getA().getType().getMemorySpace())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         SharedMemory::get());

  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
                       SharedMemory::get());
}

bool TCGen5MMAOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

bool TCGen5MMAOp::verifyOutputDims() {

  if (getTwoCtas()) {
    // Here we have to relax the verification to support two possibilities
    // - For TLX 2CTA:
    //  - Full MMA shape: [2M, K] x [K, N] -> [2M, N]
    //  - Each CTA: [M, K] x [K, N/2] -> [M, N]. We're verifying each CTA here.
    // - For non TLX 2CTA: each CTA has [M, K] x [K, N] -> [M, N]
    // We cannot rely on module attr to differentiate them here because this
    // verification can run before Fixup pass. If we want to be as accurate as
    // possible, we should have a tlxTwoCTAs flag on MMA Op in the future
    auto aShape = getA().getType().getShape();
    auto bShape = getB().getType().getShape();
    auto dShape = getD().getType().getShape();
    return dShape[dShape.size() - 2] == aShape[aShape.size() - 2] &&
           (dShape[dShape.size() - 1] == bShape[bShape.size() - 1] /* non TLX*/
            || dShape[dShape.size() - 1] ==
                   2 * bShape[bShape.size() - 1] /* TLX 2CTA*/);
  }
  // 1cta case still delegates to default verifiers
  return DotOpInterfaceTrait::verifyOutputDims();
}

Value TCGen5MMAOp::useAccumulator() { return getUseD(); }

void TCGen5MMAOp::setUseAccumulator(Value flag) {
  getUseDMutable().assign(flag);
}

void TCGen5MMAOp::addCompletionBarrier(Value barrier, Value pred) {
  getBarrierPredsMutable().append(pred);
  getBarriersMutable().append(barrier);
}

void TMAStoreTokenWaitOp::addBarrier(Value barrier, Value pred) {
  getBarriersMutable().append(barrier);
  getBarrierPredsMutable().append(pred);
}

void TMAStoreTokenWaitOp::addToken(Value token, Value idx) {
  getNvwsTokensMutable().append(token);
  getNvwsTokenIndicesMutable().append(idx);
}

// nvws-tokens-and-indices := (`nvws_token` ssa-value `[` ssa-value `]`)*
static ParseResult parseNvwsTokensAndIndices(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nvwsTokens,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nvwsTokenIndices) {
  while (succeeded(p.parseOptionalKeyword("nvws_token"))) {
    if (p.parseOperand(nvwsTokens.emplace_back()) || p.parseLSquare() ||
        p.parseOperand(nvwsTokenIndices.emplace_back()) || p.parseRSquare())
      return failure();
  }
  return success();
}

static void printNvwsTokensAndIndices(OpAsmPrinter &p, Operation *op,
                                      OperandRange nvwsTokens,
                                      OperandRange nvwsTokenIndices) {
  assert(nvwsTokens.size() == nvwsTokenIndices.size());
  for (auto [tok, idx] : llvm::zip(nvwsTokens, nvwsTokenIndices)) {
    p << " nvws_token " << tok << '[' << idx << ']';
  }
}

TypedValue<MemDescType> TCGen5MMAOp::getAccumulator() { return getD(); }

void TCGen5MMAOp::setAccumulator(Value accum) { getDMutable().assign(accum); }

Value TCGen5MMAOp::getPredicate() { return getPred(); }

void TCGen5MMAOp::setPredicate(Value pred) { getPredMutable().assign(pred); }

void TCGen5MMAOp::build(OpBuilder &builder, OperationState &state, Type token,
                        Value a, Value b, Value d, Value accDep, Value useD,
                        Value pred, bool useTwoCTAs, ValueRange barriers,
                        ValueRange barrierPreds, bool isAsync) {
  if (!barriers.empty()) {
    isAsync = true;
  }
  build(builder, state, token, a, b, d, accDep, useD, pred, barriers,
        barrierPreds, isAsync ? builder.getUnitAttr() : UnitAttr(),
        useTwoCTAs ? builder.getUnitAttr() : UnitAttr());
}

bool TCGen5MMAOp::isAsync() { return getIsAsync(); }

// -- TCGen5MMAScaledOp --
LogicalResult TCGen5MMAScaledOp::verify() {
  Type atype = getA().getType().getElementType();
  Type btype = getB().getType().getElementType();
  Type dtype = getD().getType().getElementType();
  if (failed(verifyMMADType(*this, atype, btype, dtype)))
    return failure();
  return success();
  return success();
}

void TCGen5MMAScaledOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The op reads the accumulator if `useD` is not known to be false.
  APInt useD;
  if (!matchPattern(getUseD(), m_ConstantInt(&useD)) || !useD.isZero()) {
    effects.emplace_back(MemoryEffects::Read::get(), &getDMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), &getDMutable(),
                       TensorMemory::get());

  if (isa<SharedMemorySpaceAttr>(getA().getType().getMemorySpace())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         SharedMemory::get());

  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
                       SharedMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getAScaleMutable(),
                       TensorMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getBScaleMutable(),
                       TensorMemory::get());
}

bool TCGen5MMAScaledOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  auto aKdim = aShape[aShape.size() - 1];
  auto bKdim = bShape[aShape.size() - 2];
  if (this->getAType() == ScaleDotElemType::E2M1 && !transA)
    aKdim *= 2;
  if (this->getBType() == ScaleDotElemType::E2M1 && !transB)
    bKdim *= 2;

  return aKdim == bKdim;
}

bool TCGen5MMAScaledOp::verifyOutputDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();
  auto cShape = this->getD().getType().getShape();
  auto oMdim = cShape[cShape.size() - 2];
  auto oNdim = cShape[cShape.size() - 1];

  int aMdim = aShape[aShape.size() - 2];
  int bNdim = bShape[bShape.size() - 1];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && transA)
    aMdim *= 2;
  if (this->getBType() == ScaleDotElemType::E2M1 && transB)
    bNdim *= 2;

  if (aMdim != oMdim)
    return false;

  // For 2-CTA TLX mode, output N should be 2 * B's N dimension
  if (getTwoCtas()) {
    return oNdim == bNdim || oNdim == 2 * bNdim;
  }
  return bNdim == oNdim;
}

Value TCGen5MMAScaledOp::useAccumulator() { return getUseD(); }

void TCGen5MMAScaledOp::setUseAccumulator(Value flag) {
  getUseDMutable().assign(flag);
}

void TCGen5MMAScaledOp::addCompletionBarrier(Value barrier, Value pred) {
  getBarrierPredsMutable().append(pred);
  getBarriersMutable().append(barrier);
}

TypedValue<MemDescType> TCGen5MMAScaledOp::getAccumulator() { return getD(); }

void TCGen5MMAScaledOp::setAccumulator(Value accum) {
  getDMutable().assign(accum);
}

Value TCGen5MMAScaledOp::getPredicate() { return getPred(); }

void TCGen5MMAScaledOp::setPredicate(Value pred) {
  getPredMutable().assign(pred);
}

int64_t TCGen5MMAScaledOp::getBlockM() {
  ArrayRef<int64_t> shape = getA().getType().getShape();
  int64_t blockM = shape[shape.size() - 2];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && transA)
    blockM *= 2;
  return blockM;
}

int64_t TCGen5MMAScaledOp::getBlockN() {
  ArrayRef<int64_t> shape = getB().getType().getShape();
  int64_t blockN = shape[shape.size() - 1];
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  if (this->getBType() == ScaleDotElemType::E2M1 && transB)
    blockN *= 2;
  return blockN;
}

int64_t TCGen5MMAScaledOp::getBlockK() {
  ArrayRef<int64_t> shape = getA().getType().getShape();
  int64_t blockK = shape[shape.size() - 1];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && !transA)
    blockK *= 2;
  return blockK;
}

void TCGen5MMAScaledOp::build(OpBuilder &builder, OperationState &state,
                              Type token, Value a, Value b, Value d,
                              Value accDep, Value aScale, Value bScale,
                              ScaleDotElemType aType, ScaleDotElemType bType,
                              Value useD, Value pred, bool twoCTAs,
                              ValueRange barriers, ValueRange barrierPreds,
                              bool isAsync) {
  MLIRContext *ctx = builder.getContext();
  if (!barriers.empty()) {
    isAsync = true;
  }
  build(builder, state, token, a, b, d, accDep, aScale, bScale,
        ScaleDotElemTypeAttr::get(ctx, aType),
        ScaleDotElemTypeAttr::get(ctx, bType), useD, pred, barriers,
        barrierPreds, isAsync ? builder.getUnitAttr() : UnitAttr(),
        twoCTAs ? builder.getUnitAttr() : UnitAttr());
}

bool TCGen5MMAScaledOp::isAsync() { return getIsAsync(); }

// -- TMEMStoreOp --
static LogicalResult verifyTMEMOperand(Operation *op, RankedTensorType type,
                                       MemDescType memdesc, StringRef regName) {
  if (type.getRank() != 2)
    return op->emitOpError(regName) << " must be a 2D tensor";
  // Skip verification for placeholder layouts - they will be resolved later
  if (isa<triton::tlx::DummyTMEMLayoutAttr>(memdesc.getEncoding()))
    return success();
  if (!type.getEncoding())
    return success();
  // Skip verification for placeholder layouts - they will be resolved later
  if (isa<triton::tlx::DummyRegisterLayoutAttr>(type.getEncoding()))
    return success();

  if (isDistributedLayoutTMemCompatible(op, type, memdesc))
    return success();

  // isDistributedLayoutTMemCompatible has a coverage gap for
  // getTmemLoadLayoutSplitLongM layouts. Fall back to checking if the current
  // layout matches any of the compatible layouts enumerated by
  // getTmemCompatibleLayouts.
  SmallVector<DistributedEncodingTrait> layouts =
      getTmemCompatibleLayouts(op, type, memdesc);
  auto encoding =
      dyn_cast<triton::gpu::LayoutEncodingTrait>(type.getEncoding());
  if (encoding) {
    for (auto &layout : layouts) {
      if (triton::gpu::areLayoutsEquivalent(
              type.getShape(), encoding,
              cast<triton::gpu::LayoutEncodingTrait>(layout)))
        return success();
    }
  }

  // If it failed, give the user a hint
  InFlightDiagnostic diag = op->emitOpError(regName);
  diag.attachNote() << "Got: " << type.getEncoding();
  for (Attribute layout : layouts)
    diag.attachNote() << "potential TMEM layout: " << layout;
  return diag;
}

LogicalResult TMEMStoreOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr,
           TensorMemoryScalesEncodingAttr, triton::tlx::DummyTMEMLayoutAttr>(
          getDst().getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  if (!getDst().getType().getMutableMemory()) {
    return emitOpError("Cannot store into an immutable alloc");
  }
  if (failed(verifyTMEMOperand(*this, getSrc().getType(), getDst().getType(),
                               "source")))
    return failure();
  return triton::gpu::verifyMemoryOpTypes(*this, getSrc().getType(),
                                          getDst().getType());
}

// -- TMEMLoadOp --
LogicalResult TMEMLoadOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("source must be a tensor memory buffer.");
  if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
          getSrc().getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  if (failed(verifyTMEMOperand(*this, getType(), getSrc().getType(), "result")))
    return failure();
  return triton::gpu::verifyMemoryOpTypes(*this, getSrc().getType(), getType());
}

// -- TMEMAllocOp --
LogicalResult TMEMAllocOp::verify() {
  // Accept TensorMemoryEncodingAttr, TensorMemoryScalesEncodingAttr,
  // or DummyTMEMLayoutAttr (placeholder for deferred layout resolution)
  if (!isa<TensorMemoryEncodingAttr, TensorMemoryScalesEncodingAttr,
           triton::tlx::DummyTMEMLayoutAttr>(getType().getEncoding()))
    return emitOpError("should use tensor memory encoding");
  if (getSrc() &&
      failed(verifyTMEMOperand(*this, getSrc().getType(), getType(), "source")))
    return failure();
  return triton::gpu::verifyAllocOp(*this, getSrc(), getType());
}

void TMEMAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  Operation *op = getOperation();
  // If allocation is immutable, mark it as no side effect allow things like
  // CSE, DCE to work in early compiler passes.
  // After the memory offset is computed, we attach the true side effect to the
  // op.
  if (!getType().getMutableMemory() && !op->hasAttr("tensor_memory_col_offset"))
    return;
  OpResult alloc = getOperation()->getOpResult(0);
  effects.emplace_back(MemoryEffects::Allocate::get(), alloc,
                       TensorMemory::get());
  if (getSrc())
    effects.emplace_back(MemoryEffects::Write::get(), alloc,
                         TensorMemory::get());
}

// -- TMEMCopyOp --
LogicalResult TMEMCopyOp::verify() {
  if (!isa<triton::gpu::SharedMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("The source must be a shared memory buffer");

  auto srcTy = cast<triton::gpu::MemDescType>(getSrc().getType());
  auto dstTy = cast<triton::gpu::MemDescType>(getDst().getType());

  if (getBarrier() && !isa<triton::gpu::SharedMemorySpaceAttr>(
                          getBarrier().getType().getMemorySpace())) {
    return emitOpError("The optional barrier should be a shared memory buffer");
  }
  if (!getDst().getType().getMutableMemory()) {
    return emitOpError("Cannot copy into an immutable alloc");
  }
  auto sharedEnc =
      dyn_cast<triton::gpu::SharedEncodingTrait>(srcTy.getEncoding());
  if (sharedEnc.getAlignment() < 16) {
    return emitOpError("Source must have at least 16-byte alignment to be "
                       "representable in a matrix descriptor.");
  }

  auto mod = getOperation()->getParentOfType<ModuleOp>();
  unsigned numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
  if (numCTAs != 1)
    return emitOpError("NYI: Only one CTA is supported for now.");

  // Fp4 we could lift if we needed
  auto nvmmaEnc =
      dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(srcTy.getEncoding());
  if (nvmmaEnc && (nvmmaEnc.getTransposed() || nvmmaEnc.getFp4Padded())) {
    return emitOpError("The source should not be transposed or padded");
  }
  if (isa<triton::tlx::DummyTMEMLayoutAttr>(getDst().getType().getEncoding())) {
    return success();
  } else if (isa<TensorMemoryScalesEncodingAttr>(
                 getDst().getType().getEncoding())) {
    if (nvmmaEnc && nvmmaEnc.getSwizzlingByteWidth() != 0) {
      return emitOpError("The source should not be swizzled for now");
    }
  } else {
    if (getSrc().getType().getShape() != getDst().getType().getShape()) {
      return emitOpError(
          "The source and destination must have the same shape.");
    }
    auto tmemEnc = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
        getDst().getType().getEncoding());
    if (!tmemEnc) {
      return emitOpError("Incorrect tmem layout.");
    }
    if (tmemEnc.getBlockM() != 128) {
      return emitOpError("Tmem layout ahouls have M=128.");
    }
    if (nvmmaEnc && nvmmaEnc.getSwizzlingByteWidth() == 0) {
      return emitOpError("Source layout should be swizzled.");
    }
    // When we lift this, we should make sure we handle unpacked cleanly
    if (srcTy.getElementType().getIntOrFloatBitWidth() != 32) {
      return emitOpError("Source element type should be 32-bit.");
    }
  }
  // Given that we want to support flexible input SMEM shapes, kinds of shape
  // checking we can do here are limited. For simplicity, shape checking is
  // omitted.
  return success();
}

// -- TMEMSubSliceOp --
LogicalResult TMEMSubSliceOp::verify() {
  auto srcTy = cast<triton::gpu::MemDescType>(getSrc().getType());
  auto encoding = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
      srcTy.getEncoding());
  if (!encoding)
    return emitOpError("The source must be a tensor memory buffer.");
  if (!llvm::is_contained({64, 128}, encoding.getBlockM())) {
    return emitOpError("The source tensor memory descriptor must have a 128xN "
                       "or 64xN layout, got block_m=")
           << encoding.getBlockM();
  }
  auto dstTy = cast<triton::gpu::MemDescType>(getResult().getType());
  auto dstEncoding = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
      dstTy.getEncoding());
  if (!dstEncoding)
    return emitOpError("The destination must be a tensor memory buffer.");
  if (dstEncoding.getBlockM() != encoding.getBlockM() ||
      dstEncoding.getCTASplitM() != encoding.getCTASplitM() ||
      dstEncoding.getCTASplitN() != encoding.getCTASplitN() ||
      dstEncoding.getColStride() != encoding.getColStride())
    return emitOpError("The destination must have the same block size and "
                       "CTASplit size as the source.");
  return mlir::success();
}

void TMEMSubSliceOp::build(OpBuilder &builder, OperationState &state,
                           Value alloc, int offset, int size) {
  auto allocTy = cast<triton::gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape(allocTy.getShape());
  shape.back() = size;
  auto encoding =
      cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(allocTy.getEncoding());
  unsigned newBlockN = std::min<unsigned>(encoding.getBlockN(), size);
  auto newEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
      builder.getContext(), encoding.getBlockM(), newBlockN,
      encoding.getColStride(), encoding.getCTASplitM(), encoding.getCTASplitN(),
      encoding.getTwoCTAs());
  auto subsliceType = gpu::MemDescType::get(
      shape, allocTy.getElementType(), newEncoding, allocTy.getMemorySpace(),
      allocTy.getMutableMemory(), allocTy.getAllocShape());
  build(builder, state, subsliceType, alloc, offset);
}

// -- TensormapCreateOp --
LogicalResult TensormapCreateOp::verify() {
  auto rank = getBoxDim().size();
  if (getGlobalDim().size() != rank) {
    return emitError("Rank mismatch for global dim. Got ")
           << getGlobalDim().size() << " but expected " << rank;
  }
  if (getGlobalStride().size() + 1 != rank) {
    return emitError("Rank mismatch for global stride. Got ")
           << getGlobalStride().size() << " but expected " << rank - 1;
  }
  if (getElementStride().size() != rank) {
    return emitError("Rank mismatch for element stride. Got ")
           << getElementStride().size() << " but expected " << rank;
  }
  return success();
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#define GET_OP_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.cpp.inc"
