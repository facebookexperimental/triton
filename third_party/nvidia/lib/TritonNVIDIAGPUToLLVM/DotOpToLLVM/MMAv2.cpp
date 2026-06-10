#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

namespace {

using ValueTableV2 = std::map<std::array<int, 3>, Value>;

Value loadC(Value tensor, Value llTensor,
            const LLVMTypeConverter *typeConverter, Location loc,
            ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = tensor.getContext();
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  size_t fcSize = triton::gpu::getTotalElemsPerThread(tensor.getType());

  assert(isa<NvidiaMmaEncodingAttr>(tensorTy.getEncoding()) &&
         "Currently, we only support $c with a mma layout.");
  // Load a normal C tensor with mma layout, that should be a
  // LLVM::struct with fcSize elements.
  auto structTy = cast<LLVM::LLVMStructType>(llTensor.getType());
  assert(structTy.getBody().size() == fcSize &&
         "DotOp's $c operand should pass the same number of values as $d in "
         "mma layout.");

  auto numMmaRets = tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
  assert(numMmaRets == 8 || numMmaRets == 4 || numMmaRets == 2);
  if (numMmaRets == 8 || numMmaRets == 4) {
    return llTensor;
  } else if (numMmaRets == 2) {
    auto cPack = SmallVector<Value>();
    auto cElemTy = tensorTy.getElementType();
    int numCPackedElem = 4 / numMmaRets;
    Type cPackTy = vec_ty(cElemTy, numCPackedElem);
    for (int i = 0; i < fcSize; i += numCPackedElem) {
      Value pack = LLVM::UndefOp::create(rewriter, loc, cPackTy);
      for (int j = 0; j < numCPackedElem; ++j) {
        pack = b.insert_element(cPackTy, pack,
                                b.extract_val(cElemTy, llTensor, i + j),
                                b.i32_val(j));
      }
      cPack.push_back(pack);
    }

    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(cPack.size(), cPackTy));
    Value result =
        packLLElements(loc, typeConverter, cPack, rewriter, structTy);
    return result;
  }

  return llTensor;
}

// The number of i32 registers owned by each thread along m, n, k dimensions.
// For example, for m16n8k32 with i8 inputs, a thread owns 2, 1, and 2 registers
// along m, n, k respectively.
struct NumRegisters {
  int m;
  int n;
  int k;
};

// Base indices into the per-thread A/B tiles for one MMA.
// BaseOffset::m = NumRegisters.m * m where 0 <= m < repM.
// (Similarly for n and k.)
struct BaseOffset {
  int m;
  int n;
  int k;
};

ValueTableV2 getValuesFromDotOperandLayoutStruct(
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter, Value value, RankedTensorType type,
    const NumRegisters &numRegisters, int repBatch, int repOuter, int repK) {
  // Use the LinearLayout to map register indices to element coordinates,
  // then derive value table keys from coordinates. This naturally handles
  // NPOT rep counts (via modularIdentity1D in the layout) and largeK
  // (by computing the correct sub-MMA index from K coordinates), without
  // any pow2 rounding or manual permutation.
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto elems = unpackLLElements(loc, value, rewriter);
  auto eltTy = typeConverter->convertType(type.getElementType());
  ValueTableV2 vals;
  auto bitwidth = eltTy.getIntOrFloatBitWidth();
  auto numElemsPerVec = std::max(32 / bitwidth, 1u);
  auto vecTy = vec_ty(eltTy, numElemsPerVec);

  auto dot = cast<DotOperandEncodingAttr>(type.getEncoding());
  auto kWidth = dot.getKWidth();
  bool isA = (dot.getOpIdx() == 0);
  auto rank = type.getRank();

  // Get the LinearLayout for this dot operand encoding.
  auto layout = triton::gpu::toLinearLayout(type);
  auto *ctx = type.getContext();
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");

  // Dimension indices in the layout output.
  int kDimIdx = isA ? rank - 1 : rank - 2;
  int outerDimIdx = isA ? rank - 2 : rank - 1;
  int kDimSize = triton::gpu::getShapePerCTA(type)[kDimIdx];

  // Get warp tiling info to compute coordinate-to-rep-index mapping.
  auto mma = cast<NvidiaMmaEncodingAttr>(dot.getParent());
  auto warpsPerCTA = mma.getWarpsPerCTA();
  // For dot operands, the K dimension warps are broadcast (set to 1).
  // The outer dimension (M for A, N for B) uses the remaining warps.
  int warpsOuter = warpsPerCTA[outerDimIdx];

  // The batch dimension (3D dots) is also tiled across warps.  Each MMA
  // handles a single batch element (tile size 1), so consecutive batch reps
  // for one warp are strided by the number of warps tiling the batch dim.
  int warpsBatch = (rank == 3) ? warpsPerCTA[0] : 1;
  int batchRepStride = 1 * warpsBatch;

  // MMA tile sizes per warp in element coordinates.
  int mmaTileOuter = isA ? 16 : 8; // M=16 for A, N=8 for B

  // Stride between consecutive outer reps for one warp: the MMA tile
  // times the number of warps tiling that dimension, since warps
  // interleave across the outer dimension.
  int outerRepStride = mmaTileOuter * warpsOuter;

  // Number of outer registers per MMA tile (within-tile register groups).
  int numRegOuter = isA ? numRegisters.m : numRegisters.n;
  // Stride between consecutive within-tile register positions.
  int innerOuterStride = mmaTileOuter / numRegOuter; // always 8

  // K tile sizes and sub-MMA splitting parameters.
  int dotTileK = kWidth * 8;
  int numSubMmaPerDotTile = kWidth / numElemsPerVec;

  auto numVecs = static_cast<int>(elems.size()) / numElemsPerVec;
  for (int vecIdx = 0; vecIdx < numVecs; ++vecIdx) {
    int regIdx = vecIdx * numElemsPerVec;

    // Query the layout for this register's element coordinates.
    auto coords = layout.apply(
        {{kRegister, regIdx}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});

    int outerCoord = coords[outerDimIdx].second;
    int kCoord = coords[kDimIdx].second;
    // Map the absolute batch coordinate to a per-thread batch rep index
    // (consumers index the value table by rep index, not absolute coord).
    int batchCoord = (rank == 3) ? coords[0].second / batchRepStride : 0;

    // Outer register index (M or N direction).
    // outerRep = which rep of the MMA tile this coordinate belongs to.
    // innerOuter = which register within the MMA tile.
    int outerRep = outerCoord / outerRepStride;
    int innerOuter = (outerCoord % mmaTileOuter) / innerOuterStride;
    int outerReg = outerRep * numRegOuter + innerOuter;

    // K register index.
    int kReg;
    if (bitwidth == 64) {
      // fp64 is never largeK: K-registers are uniformly K-strided, so the
      // emitter indexes them contiguously. Map kCoord back to its linear index.
      int numKRegs = std::max(repK, 1) * numRegisters.k;
      int kStep = std::max<int>(kDimSize / numKRegs, 1);
      kReg = kCoord / kStep;
    } else {
      // Packed/largeK: account for sub-MMA splitting.
      int dotTileIdx = kCoord / dotTileK;
      int kInDotTile = kCoord % dotTileK;
      int kWidthGroupIdx = kInDotTile / (kWidth * 4); // 0 or 1
      int kInGroup = kInDotTile % kWidth;
      int subMmaIdx = kInGroup / numElemsPerVec;
      int kRepInTile = subMmaIdx * numRegisters.k + kWidthGroupIdx;
      kReg = dotTileIdx * numSubMmaPerDotTile * numRegisters.k + kRepInTile;
    }

    // Pack numElemsPerVec elements into one i32 (or keep as-is for fp64).
    Value vec = b.undef(vecTy);
    for (unsigned i = 0; i < numElemsPerVec; ++i) {
      vec = b.insert_element(vec, b.bitcast(elems[regIdx + i], eltTy),
                             b.i32_val(i));
    }
    if (bitwidth == 64) {
      vals[{batchCoord, outerReg, kReg}] = vec;
    } else {
      vals[{batchCoord, outerReg, kReg}] = b.bitcast(vec, i32_ty);
    }
  }

  // Fill broadcast slots.  When the outer dimension (M for A, N for B) is
  // smaller than the MMA fragment extent (e.g. M=1 < 16, N=2 < 8) the layout
  // collapses the broadcast registers, so layout.apply() above only produces
  // outer-register positions for the distinct outer coordinates.  The MMA
  // emitters, however, unconditionally index every fragment slot the
  // instruction requires (e.g. ha[{b, base.m, k}] AND ha[{b, base.m + 1, k}]).
  // Any slot that was not produced above must be aliased to the broadcast
  // source (the same value, replicated), otherwise the operand is a null Value
  // and the resulting inline asm has fewer parameters than constraints
  // (cantFail abort).  Mirror the layout broadcast by wrapping the missing
  // outer-register index modulo the number of distinct outer registers that
  // were actually populated.
  int numOuterRegs = repOuter * numRegOuter;
  for (int bIdx = 0; bIdx < std::max(repBatch, 1); ++bIdx) {
    for (int kReg = 0; kReg < std::max(repK, 1) * numRegisters.k; ++kReg) {
      // Collect the outer-register slots that were actually produced for this
      // (b, kReg).  Broadcast collapses to the low coordinates, so these form a
      // contiguous prefix, but we don't rely on that here.
      SmallVector<int> present;
      for (int oReg = 0; oReg < numOuterRegs; ++oReg)
        if (vals.count({bIdx, oReg, kReg}))
          present.push_back(oReg);
      if (present.empty())
        continue; // nothing to broadcast from (handled elsewhere)
      for (int oReg = 0; oReg < numOuterRegs; ++oReg) {
        if (!vals.count({bIdx, oReg, kReg}))
          vals[{bIdx, oReg, kReg}] =
              vals[{bIdx, present[oReg % present.size()], kReg}];
      }
    }
  }
  return vals;
}

enum class TensorCoreType : uint8_t {
  // floating-point tensor core instr
  FP32_FP16_FP16_FP32 = 0, // default
  FP32_BF16_BF16_FP32,
  FP32_TF32_TF32_FP32,
  FP16_FP16_FP16_FP16,
  // fp32 accumulator, fp8 operand
  FP32_FP8E5M2_FP8E5M2_FP32,
  FP32_FP8E5M2_FP8E4M3FN_FP32,
  FP32_FP8E4M3FN_FP8E5M2_FP32,
  FP32_FP8E4M3FN_FP8E4M3FN_FP32,
  // fp16 accumulator, fp8 operand
  FP16_FP8E5M2_FP8E5M2_FP16,
  FP16_FP8E5M2_FP8E4M3FN_FP16,
  FP16_FP8E4M3FN_FP8E5M2_FP16,
  FP16_FP8E4M3FN_FP8E4M3FN_FP16,
  // integer tensor core instr
  INT32_INT1_INT1_INT32, // Not implemented
  INT32_INT4_INT4_INT32, // Not implemented
  INT32_INT8_INT8_INT32, // Not implemented
  // double precision tensor core instr
  FP64_FP64_FP64_FP64,
  // scaled mxfp8 x mxfp8 matmul
  FP32_FP8E5M2_FP8E5M2_FP32_SCALE_VEC_1X,
  FP32_FP8E5M2_FP8E4M3FN_FP32_SCALE_VEC_1X,
  FP32_FP8E4M3FN_FP8E5M2_FP32_SCALE_VEC_1X,
  FP32_FP8E4M3FN_FP8E4M3FN_FP32_SCALE_VEC_1X,
  //
  FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X,
  FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X,
  //
  NOT_APPLICABLE,
};

static Type getMmaRetType(TensorCoreType mmaType, MLIRContext *ctx) {
  Type fp64Ty = type::f64Ty(ctx);
  Type fp32Ty = type::f32Ty(ctx);
  Type fp16Ty = type::f16Ty(ctx);
  Type i32Ty = type::i32Ty(ctx);
  Type fp64x4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp64Ty));
  Type fp32x4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp32Ty));
  Type i32x4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, i32Ty));
  Type fp16x2Pack2Ty = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(2, vec_ty(fp16Ty, 2)));
  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP16_FP16_FP16_FP16:
    return fp16x2Pack2Ty;
  case TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32:
  case TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32:
  case TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32:
  case TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP16_FP8E5M2_FP8E5M2_FP16:
  case TensorCoreType::FP16_FP8E5M2_FP8E4M3FN_FP16:
  case TensorCoreType::FP16_FP8E4M3FN_FP8E5M2_FP16:
  case TensorCoreType::FP16_FP8E4M3FN_FP8E4M3FN_FP16:
    return fp16x2Pack2Ty;
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return i32x4Ty;
  case TensorCoreType::FP64_FP64_FP64_FP64:
    return fp64x4Ty;
  case TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X:
  case TensorCoreType::FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X:
    return fp32x4Ty;
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }

  return Type{};
}

static TensorCoreType getMmaTypeDotScaled(DotScaledOp op, RankedTensorType aTy,
                                          RankedTensorType bTy,
                                          RankedTensorType dTy) {
  if (dTy.getElementType().isF32()) {
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType())) {
      return TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32_SCALE_VEC_1X;
    }
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType())) {
      return TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32_SCALE_VEC_1X;
    }
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType())) {
      return TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32_SCALE_VEC_1X;
    }
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType())) {
      return TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32_SCALE_VEC_1X;
    }
    if (op.getBElemType() == ScaleDotElemType::E2M1 &&
        op.getAElemType() == ScaleDotElemType::E2M1) {
      if (isa<mlir::Float8E4M3FNType>(
              op.getBScale().getType().getElementType())) {
        return TensorCoreType::FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X;
      } else {
        return TensorCoreType::FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X;
      }
    }
  }
  return TensorCoreType::NOT_APPLICABLE;
}

static TensorCoreType getMmaTypeDot(DotOp op, RankedTensorType aTy,
                                    RankedTensorType bTy,
                                    RankedTensorType dTy) {
  if (dTy.getElementType().isF32()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP32_FP16_FP16_FP32;
    if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
      return TensorCoreType::FP32_BF16_BF16_FP32;
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType()))
      return TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32;
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType()))
      return TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32;
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType()))
      return TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32;
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType()))
      return TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32;
    if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
        op.getInputPrecision() == InputPrecision::TF32)
      return TensorCoreType::FP32_TF32_TF32_FP32;
  } else if (dTy.getElementType().isInteger(32)) {
    if (aTy.getElementType().isInteger(8) && bTy.getElementType().isInteger(8))
      return TensorCoreType::INT32_INT8_INT8_INT32;
  } else if (dTy.getElementType().isF16()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP16_FP16_FP16_FP16;
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType()))
      return TensorCoreType::FP16_FP8E5M2_FP8E5M2_FP16;
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType()))
      return TensorCoreType::FP16_FP8E5M2_FP8E4M3FN_FP16;
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType()))
      return TensorCoreType::FP16_FP8E4M3FN_FP8E5M2_FP16;
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType()))
      return TensorCoreType::FP16_FP8E4M3FN_FP8E4M3FN_FP16;
  } else if (dTy.getElementType().isF64()) {
    if (aTy.getElementType().isF64() && bTy.getElementType().isF64())
      return TensorCoreType::FP64_FP64_FP64_FP64;
  }

  return TensorCoreType::NOT_APPLICABLE;
}

inline static const std::map<TensorCoreType, std::string> mmaInstrPtxTuring = {
    {TensorCoreType::FP32_FP16_FP16_FP32,
     "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"},

    {TensorCoreType::INT32_INT8_INT8_INT32,
     "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32"},

    {TensorCoreType::FP16_FP16_FP16_FP16,
     "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"},
};

inline static const std::map<TensorCoreType, std::string> mmaInstrPtxAmpere = {
    {TensorCoreType::FP32_FP16_FP16_FP32,
     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"},
    {TensorCoreType::FP32_BF16_BF16_FP32,
     "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"},
    {TensorCoreType::FP32_TF32_TF32_FP32,
     "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"},

    {TensorCoreType::INT32_INT1_INT1_INT32,
     "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc"},
    {TensorCoreType::INT32_INT4_INT4_INT32,
     "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32"},
    {TensorCoreType::INT32_INT8_INT8_INT32,
     "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"},

    {TensorCoreType::FP16_FP16_FP16_FP16,
     "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"},

    {TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32,
     "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"},
    {TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32,
     "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32"},
    {TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32,
     "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32"},
    {TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32,
     "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"},

    {TensorCoreType::FP16_FP8E5M2_FP8E5M2_FP16,
     "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16"},
    {TensorCoreType::FP16_FP8E5M2_FP8E4M3FN_FP16,
     "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e4m3.f16"},
    {TensorCoreType::FP16_FP8E4M3FN_FP8E5M2_FP16,
     "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16"},
    {TensorCoreType::FP16_FP8E4M3FN_FP8E4M3FN_FP16,
     "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16"},

    {TensorCoreType::FP64_FP64_FP64_FP64,
     "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"},
};

inline static const std::map<TensorCoreType, std::string> mmaInstrPtxHopper = {
    {TensorCoreType::FP64_FP64_FP64_FP64,
     "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64"},
};

inline static const std::map<TensorCoreType, std::string> mmaInstrPtxScaled = {
    {TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32_SCALE_VEC_1X,
     "mma.sync.aligned.m16n8k32.row.col."
     "kind::mxf8f6f4.block_scale.scale_vec::"
     "1X.f32.e5m2.e5m2.f32.ue8m0"},
    {TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32_SCALE_VEC_1X,
     "mma.sync.aligned.m16n8k32.row.col."
     "kind::mxf8f6f4.block_scale.scale_vec::"
     "1X.f32.e5m2.e4m3.f32.ue8m0"},
    {TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32_SCALE_VEC_1X,
     "mma.sync.aligned.m16n8k32.row.col."
     "kind::mxf8f6f4.block_scale.scale_vec::"
     "1X.f32.e4m3.e5m2.f32.ue8m0"},
    {TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32_SCALE_VEC_1X,
     "mma.sync.aligned.m16n8k32.row.col."
     "kind::mxf8f6f4.block_scale.scale_vec::"
     "1X.f32.e4m3.e4m3.f32.ue8m0"},
    {TensorCoreType::FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X,
     "mma.sync.aligned.m16n8k64.row.col."
     "kind::mxf4nvf4.block_scale.scale_vec::"
     "2X.f32.e2m1.e2m1.f32.ue8m0"},
    {TensorCoreType::FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X,
     "mma.sync.aligned.m16n8k64.row.col."
     "kind::mxf4nvf4.block_scale.scale_vec::"
     "4X.f32.e2m1.e2m1.f32.ue4m3"},
};

static void callMmaTuringInt8(PTXBuilder &builder, int b,
                              const BaseOffset &base,
                              mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                              unsigned colsPerThread, int numCPackedElem,
                              ValueTableV2 &ha, ValueTableV2 &hb,
                              const SmallVector<Value> &fc) {
  auto retArgs1 = builder.newListOperand(numMmaRets / 2, "=r");
  auto retArgs2 = builder.newListOperand(numMmaRets / 2, "=r");
  auto cArgs1 = builder.newListOperand();
  for (int i = 0; i < numMmaRets / 2; ++i) {
    cArgs1->listAppend(builder.newOperand(
        fc[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i],
        std::to_string(i)));
    // reuse the output registers
  }
  auto cArgs2 = builder.newListOperand();
  for (int i = numMmaRets / 2; i < numMmaRets; ++i) {
    cArgs2->listAppend(builder.newOperand(
        fc[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i],
        std::to_string(i)));
    // reuse the output registers
  }
  auto aArgs1 = builder.newListOperand({
      {ha[{b, base.m, base.k}], "r"},
  });
  auto bArgs1 = builder.newListOperand({
      {hb[{b, base.n, base.k}], "r"},
  });
  auto aArgs2 = builder.newListOperand({
      {ha[{b, base.m, base.k + 1}], "r"},
  });
  auto bArgs2 = builder.newListOperand({{hb[{b, base.n, base.k + 1}], "r"}});
  auto aArgs3 = builder.newListOperand({
      {ha[{b, base.m + 1, base.k}], "r"},
  });
  auto bArgs3 = builder.newListOperand({
      {hb[{b, base.n, base.k}], "r"},
  });
  auto aArgs4 = builder.newListOperand({
      {ha[{b, base.m + 1, base.k + 1}], "r"},
  });
  auto bArgs4 = builder.newListOperand({{hb[{b, base.n, base.k + 1}], "r"}});
  mma(retArgs1, aArgs1, bArgs1, cArgs1);
  mma(retArgs1, aArgs2, bArgs2, cArgs1);
  mma(retArgs2, aArgs3, bArgs3, cArgs2);
  mma(retArgs2, aArgs4, bArgs4, cArgs2);
}

static void callMmaTuringFp16(PTXBuilder &builder, int b,
                              const BaseOffset &base,
                              mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                              unsigned colsPerThread, int numCPackedElem,
                              ValueTableV2 &ha, ValueTableV2 &hb,
                              const SmallVector<Value> &fc, bool isAccF16) {
  auto retArgs = builder.newListOperand(numMmaRets, isAccF16 ? "=r" : "=f");
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i) {
    cArgs->listAppend(builder.newOperand(
        fc[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i],
        std::to_string(i)));
    // reuse the output registers
  }
  auto aArgs1 = builder.newListOperand({
      {ha[{b, base.m, base.k}], "r"},
      {ha[{b, base.m + 1, base.k}], "r"},
  });
  auto bArgs1 = builder.newListOperand({{hb[{b, base.n, base.k}], "r"}});
  auto aArgs2 = builder.newListOperand({
      {ha[{b, base.m, base.k + 1}], "r"},
      {ha[{b, base.m + 1, base.k + 1}], "r"},
  });
  auto bArgs2 = builder.newListOperand({{hb[{b, base.n, base.k + 1}], "r"}});
  mma(retArgs, aArgs1, bArgs1, cArgs);
  mma(retArgs, aArgs2, bArgs2, cArgs);
}

// Repeat m8n8k4 (2, 1, 4) times, as m16n8k16 on hopper.
static void callMmaAmpereFp64(PTXBuilder &builder, int b,
                              const BaseOffset &base,
                              mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                              unsigned colsPerThread, int numCPackedElem,
                              unsigned batchOffset, ValueTableV2 &ha,
                              ValueTableV2 &hb, const SmallVector<Value> &fc,
                              int kRegs) {
  auto retArgs1 = builder.newListOperand(numMmaRets / 2, "=d");
  auto retArgs2 = builder.newListOperand(numMmaRets / 2, "=d");
  auto cArgs1 = builder.newListOperand();
  for (int i = 0; i < numMmaRets / 2; ++i) {
    cArgs1->listAppend(builder.newOperand(
        fc[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i +
           batchOffset * b],
        std::to_string(i)));
    // reuse the output registers
  }
  auto cArgs2 = builder.newListOperand();
  for (int i = numMmaRets / 2; i < numMmaRets; ++i) {
    cArgs2->listAppend(builder.newOperand(
        fc[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i +
           batchOffset * b],
        std::to_string(i)));
    // reuse the output registers
  }
  for (int vk = 0; vk < kRegs; ++vk) {
    auto aArgs1 = builder.newListOperand({
        {ha[{b, base.m, base.k + vk}], "d"},
    });
    auto bArgs = builder.newListOperand({{hb[{b, base.n, base.k + vk}], "d"}});
    auto aArgs2 = builder.newListOperand({
        {ha[{b, base.m + 1, base.k + vk}], "d"},
    });
    mma(retArgs1, aArgs1, bArgs, cArgs1);
    mma(retArgs2, aArgs2, bArgs, cArgs2);
  }
}

// Unified MMAV2 function for Ampere and HopperF64 architectures
static void callMmaV2(PTXBuilder &builder, int b, const BaseOffset &base,
                      mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                      unsigned colsPerThread, int numCPackedElem,
                      unsigned batchOffset, ValueTableV2 &ha, ValueTableV2 &hb,
                      const SmallVector<Value> &fc,
                      const std::string &constraintRet,
                      const std::string &constraintAB, int kRegs) {
  auto retArgs = builder.newListOperand(numMmaRets, constraintRet);
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i) {
    cArgs->listAppend(builder.newOperand(
        fc[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i +
           batchOffset * b],
        std::to_string(i)));
    // reuse the output registers
  }

  auto aArgs = builder.newListOperand();
  for (int vk = 0; vk < kRegs; ++vk) {
    aArgs->listAppend(
        builder.newOperand(ha[{b, base.m, base.k + vk}], constraintAB));
    aArgs->listAppend(
        builder.newOperand(ha[{b, base.m + 1, base.k + vk}], constraintAB));
  }

  auto bArgs = builder.newListOperand();
  for (int vk = 0; vk < kRegs; ++vk) {
    bArgs->listAppend(
        builder.newOperand(hb[{b, base.n, base.k + vk}], constraintAB));
  }

  mma(retArgs, aArgs, bArgs, cArgs);
}

static void callMmaScaled(PTXBuilder &builder, int b, const BaseOffset &base,
                          mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                          unsigned colsPerThread, ValueTableV2 &aTable,
                          ValueTableV2 &bTable,
                          const SmallVector<Value> &cValues, Value aScaleValue,
                          Value bScaleValue, int kRegs) {
  int numCPackedElem = 4 / static_cast<int>(numMmaRets);
  auto retArgs = builder.newListOperand(numMmaRets, "=f");
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i)
    cArgs->listAppend(builder.newOperand(
        cValues[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i],
        std::to_string(i)));

  auto aArgs = builder.newListOperand();
  for (int vk = 0; vk < kRegs; ++vk) {
    aArgs->listAppend(
        builder.newOperand(aTable[{b, base.m, base.k + vk}], "r"));
    aArgs->listAppend(
        builder.newOperand(aTable[{b, base.m + 1, base.k + vk}], "r"));
  }

  auto bArgs = builder.newListOperand();
  for (int vk = 0; vk < kRegs; ++vk)
    bArgs->listAppend(
        builder.newOperand(bTable[{b, base.n, base.k + vk}], "r"));

  SmallVector<PTXBuilder::Operand *> ops{retArgs, aArgs, bArgs, cArgs};

  auto appendScale = [&](Value scale, unsigned byteId, unsigned threadId) {
    ops.push_back(builder.newOperand(scale, "r"));
    auto sel = builder.newListOperand();
    sel->listAppend(builder.newConstantOperand(std::to_string(byteId)));
    sel->listAppend(builder.newConstantOperand(std::to_string(threadId)));
    ops.push_back(sel);
  };

  // Use only byteId=0 since each thread sign-extends a single i8 scale
  // into i32 instead of packing 4 bytes.
  appendScale(aScaleValue, 0, 0);
  appendScale(bScaleValue, 0, 0);

  mma(ops);
}

using EmitMmaCallback = std::function<void(
    PTXBuilder &builder, int b, int m, int n, int k,
    mlir::triton::PTXInstr &mma, unsigned numMmaRets, unsigned colsPerThread,
    unsigned batchOffset, ValueTableV2 &ha, ValueTableV2 &hb,
    const SmallVector<Value> &fc, RankedTensorType dTensorTy, int repK)>;

LogicalResult
convertMMAImpl(DotOpInterface op, Value llvmA, Value llvmB, Value llvmC,
               const LLVMTypeConverter *typeConverter,
               ConversionPatternRewriter &rewriter, TensorCoreType mmaType,
               const NumRegisters &numRegisters,
               const std::map<TensorCoreType, std::string> &mmaInstructions,
               const EmitMmaCallback &emitMma) {
  auto loc = op.getLoc();
  auto aType = cast<RankedTensorType>(op.getA().getType());
  auto bType = cast<RankedTensorType>(op.getB().getType());
  assert(mlir::isa<DotOperandEncodingAttr>(aType.getEncoding()) &&
         mlir::isa<DotOperandEncodingAttr>(bType.getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  Value cOperand = op->getOperand(2);
  Value loadedC = loadC(cOperand, llvmC, typeConverter, loc, rewriter);

  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = op->getContext();

  auto aTensorTy = cast<RankedTensorType>(op.getA().getType());
  auto bTensorTy = cast<RankedTensorType>(op.getB().getType());
  auto dTensorTy = cast<RankedTensorType>(op.getD().getType());

  auto aShapePerCTA = triton::gpu::getShapePerCTA(aTensorTy);
  auto bShapePerCTA = triton::gpu::getShapePerCTA(bTensorTy);
  auto dShapePerCTA = triton::gpu::getShapePerCTA(dTensorTy);

  int bitwidth = aTensorTy.getElementType().getIntOrFloatBitWidth();
  auto dotOpA = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  int kWidth = dotOpA.getKWidth();
  auto repA =
      cast<NvidiaMmaEncodingAttr>(dotOpA.getParent())
          .getRepForOperand(aShapePerCTA, bitwidth, kWidth, dotOpA.getOpIdx());
  auto dotOpB = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  auto repB =
      cast<NvidiaMmaEncodingAttr>(dotOpB.getParent())
          .getRepForOperand(bShapePerCTA, bitwidth, kWidth, dotOpB.getOpIdx());

  assert(repA[2] == repB[1]);
  assert(repA[0] == repB[0]);
  int repM = repA[1], repN = repB[2], repK = repA[2];
  int repBatch = repA[0];

  auto ha = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                llvmA, aTensorTy, numRegisters,
                                                repBatch, repM, repK);

  auto hb = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                llvmB, bTensorTy, numRegisters,
                                                repBatch, repN, repK);

  auto fc = unpackLLElements(loc, loadedC, rewriter);

  int bitwidthRet = dTensorTy.getElementType().getIntOrFloatBitWidth();
  auto numMmaRets = bitwidthRet == 64 ? 4 : bitwidthRet / 8;
  int numCPackedElem = 4 / numMmaRets;

  if (mmaInstructions.find(mmaType) == mmaInstructions.end()) {
    return emitError(loc, "Unsupported MMA instruction for the given mma type");
  }
  auto rank = dTensorTy.getRank();
  auto elemsPerThread = triton::gpu::getElemsPerThread(dTensorTy);
  auto batchOffset =
      elemsPerThread[rank - 2] * elemsPerThread[rank - 1] / numCPackedElem;
  // The register layout allocates nextPow2(repN) positions per N-rep set
  // (ceil(log2(repN)) register bits), even when repN is NPOT.  Dead register
  // positions at n >= repN hold aliased values that are never touched by the
  // MMA loop.  Using the pow2-rounded stride keeps the formula consistent
  // with the LinearLayout's register ordering.
  unsigned colsPerThread = (1u << llvm::Log2_64_Ceil(std::max(repN, 1))) * 2;
  auto callMma = [&](unsigned b, unsigned m, unsigned n, unsigned k) {
    PTXBuilder builder;
    auto &mma = *builder.create(mmaInstructions.at(mmaType));
    emitMma(builder, b, static_cast<int>(m), static_cast<int>(n),
            static_cast<int>(k), mma, numMmaRets, colsPerThread, batchOffset,
            ha, hb, fc, dTensorTy, repK);

    Value mmaOut =
        builder.launch(rewriter, loc, getMmaRetType(mmaType, op->getContext()));

    Type elemTy = cast<LLVM::LLVMStructType>(mmaOut.getType()).getBody()[0];
    for (int i = 0; i < numMmaRets; ++i) {
      fc[(numRegisters.m * static_cast<int>(m) * colsPerThread +
          4 * numRegisters.n * static_cast<int>(n)) /
             numCPackedElem +
         i + batchOffset * b] = tb.extract_val(elemTy, mmaOut, i);
    }
  };

  for (int b = 0; b < repBatch; ++b)
    for (int k = 0; k < repK; ++k)
      for (int m = 0; m < repM; ++m)
        for (int n = 0; n < repN; ++n) {
          callMma(b, m, n, k);
        }

  Type resElemTy = dTensorTy.getElementType();

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(fc.size() * numCPackedElem, resElemTy));
  SmallVector<Value> results(fc.size() * numCPackedElem);
  for (int i = 0; i < fc.size(); ++i) {
    for (int j = 0; j < numCPackedElem; ++j) {
      results[i * numCPackedElem + j] =
          numCPackedElem > 1
              ? tb.bitcast(tb.extract_element(fc[i], tb.i32_val(j)), resElemTy)
              : tb.bitcast(fc[i], resElemTy);
    }
  }
  Value res = packLLElements(loc, typeConverter, results, rewriter, structTy);

  rewriter.replaceOp(op, res);

  return success();
}

} // namespace

LogicalResult convertMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                         const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring,
                         bool isHopperF64) {
  auto aTensorTy = op.getA().getType();
  auto bTensorTy = op.getB().getType();
  auto dTensorTy = op.getD().getType();

  TensorCoreType mmaType = getMmaTypeDot(op, aTensorTy, bTensorTy, dTensorTy);

  bool isFp64Path = (mmaType == TensorCoreType::FP64_FP64_FP64_FP64);
  NumRegisters numRegisters = {2, 1, isFp64Path ? 4 : 2};

  const auto &instrMap =
      isTuring ? mmaInstrPtxTuring
               : (isHopperF64 ? mmaInstrPtxHopper : mmaInstrPtxAmpere);
  EmitMmaCallback emit = [&](PTXBuilder &builder, int b, int m, int n, int k,
                             mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                             unsigned colsPerThread, unsigned batchOffset,
                             ValueTableV2 &ha, ValueTableV2 &hb,
                             const SmallVector<Value> &fc, RankedTensorType dTy,
                             int /*repK*/) {
    const unsigned numCPackedElem = 4u / numMmaRets;
    bool isIntMMA = dTy.getElementType().isInteger(32);
    bool isAccF16 = dTy.getElementType().isF16();
    bool isFp64MMA = dTy.getElementType().isF64();
    BaseOffset base{numRegisters.m * m, numRegisters.n * n, numRegisters.k * k};
    if (isTuring) {
      assert(b == 0 && "Turing only supports batch size 1");
      if (isIntMMA)
        callMmaTuringInt8(builder, b, base, mma, numMmaRets, colsPerThread,
                          numCPackedElem, ha, hb, fc);
      else
        callMmaTuringFp16(builder, b, base, mma, numMmaRets, colsPerThread,
                          numCPackedElem, ha, hb, fc, isAccF16);
    } else {
      if (isFp64MMA) {
        if (!isHopperF64) {
          callMmaAmpereFp64(builder, b, base, mma, numMmaRets, colsPerThread,
                            numCPackedElem, batchOffset, ha, hb, fc,
                            /*kRegs*/ 4);
        } else {
          callMmaV2(builder, b, base, mma, numMmaRets, colsPerThread,
                    numCPackedElem, batchOffset, ha, hb, fc, "=d", "d",
                    /*kRegs*/ 4);
        }
      } else {
        callMmaV2(builder, b, base, mma, numMmaRets, colsPerThread,
                  numCPackedElem, batchOffset, ha, hb, fc,
                  isIntMMA || isAccF16 ? "=r" : "=f", "r", numRegisters.k);
      }
    }
  };

  return convertMMAImpl(op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
                        typeConverter, rewriter, mmaType, numRegisters,
                        instrMap, emit);
}

LogicalResult convertMMADotScaled(triton::DotScaledOp op,
                                  triton::DotScaledOp::Adaptor adaptor,
                                  const LLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter) {
  auto aTensorTy = cast<RankedTensorType>(op.getA().getType());
  auto bTensorTy = cast<RankedTensorType>(op.getB().getType());
  auto dTensorTy = cast<RankedTensorType>(op.getD().getType());

  TensorCoreType mmaType =
      getMmaTypeDotScaled(op, aTensorTy, bTensorTy, dTensorTy);

  SmallVector<Value> unpackedAScale =
      unpackLLElements(op.getLoc(), adaptor.getAScale(), rewriter);
  SmallVector<Value> unpackedBScale =
      unpackLLElements(op.getLoc(), adaptor.getBScale(), rewriter);

  NumRegisters numRegisters = {2, 1, 2};
  EmitMmaCallback emit = [&](PTXBuilder &builder, int b, int m, int n, int k,
                             mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                             unsigned colsPerThread, unsigned batchOffset,
                             ValueTableV2 &aTable, ValueTableV2 &bTable,
                             const SmallVector<Value> &cValues,
                             RankedTensorType dTy, int repK) {
    auto tb = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    auto i32 = IntegerType::get(op->getContext(), 32);

    auto packElements = [&](ArrayRef<Value> bytes, int loc,
                            int numBytes) -> Value {
      Value packed = tb.zext(i32, bytes[loc]);
      for (int i = 1; i < numBytes; ++i) {
        Value byte = tb.zext(i32, bytes[loc + i]);
        Value shifted = tb.shl(byte, tb.i32_val(i * 8));
        packed = tb.or_(packed, shifted);
      }
      return packed;
    };

    int scaleVecMode;
    if (mmaInstrPtxScaled.at(mmaType).find("1X") != std::string::npos) {
      scaleVecMode = 1;
    } else if (mmaType ==
               TensorCoreType::FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X) {
      scaleVecMode = 2;
    } else if (mmaType == TensorCoreType::FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X) {
      scaleVecMode = 4;
    } else {
      llvm_unreachable("Unsupported scale vector mode!");
    }
    Value aScaleValue =
        packElements(unpackedAScale, m * repK * scaleVecMode + k * scaleVecMode,
                     scaleVecMode);
    Value bScaleValue =
        packElements(unpackedBScale, n * repK * scaleVecMode + k * scaleVecMode,
                     scaleVecMode);

    BaseOffset base{numRegisters.m * m, numRegisters.n * n, numRegisters.k * k};
    callMmaScaled(builder, b, base, mma, numMmaRets, colsPerThread, aTable,
                  bTable, cValues, aScaleValue, bScaleValue, numRegisters.k);
  };

  return convertMMAImpl(op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
                        typeConverter, rewriter, mmaType, numRegisters,
                        mmaInstrPtxScaled, emit);
}
