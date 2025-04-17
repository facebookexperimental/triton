#include "Dialect/NVGPU/IR/Dialect.h"
#include "DotOpToLLVM/MMAHelpers.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;

namespace ttg = ::mlir::triton::gpu;
namespace ttn = ::mlir::triton::nvgpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::SwizzledSharedEncodingAttr;
using ttn::OperandsAndConstraints;

// The functions below are defined in AccelerateMatmul.cpp.
namespace mlir::triton::gpu {
SmallVector<unsigned, 3>
getWarpsPerTile(Operation *dotOp, ArrayRef<int64_t> shape, int version,
                int numWarps, const SmallVector<unsigned, 3> &instrShape);
int computeOrigBitWidth(Value x);
Value getSharedMemMMAOperand(Value v, mlir::PatternRewriter &rewriter,
                             int opIdx, bool allowTranspose);
} // namespace mlir::triton::gpu

// The functions below are defined in MMAv2.cpp.
using ValueTableV2 = std::map<std::array<int, 3>, Value>;
ValueTableV2 getValuesFromDotOperandLayoutStruct(
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter, Value value, int batch, int repOuter,
    int repK, RankedTensorType type);

namespace {
// Copied from WGMMA.cpp.
// TODO(sparsity):
// Originally there was just a definition for this (and expected to link with
// WGMMA.cpp); but in WGMMA.cpp it's marked as static; how did that work?
static Value createDescriptor(ConversionPatternRewriter &rewriter, Location loc,
                              int64_t swizzling, uint32_t stride) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  static_assert(sizeof(NVIDIA::SMEMDescriptor) == 8,
                "Descriptor size should be 64 bits.");
  NVIDIA::SMEMDescriptor desc;
  desc.descriptor = 0;
  switch (swizzling) {
  case 0:
    desc.swizzlingMode = 0;
    break;
  case 32:
    desc.swizzlingMode = 3;
    break;
  case 64:
    desc.swizzlingMode = 2;
    break;
  case 128:
    desc.swizzlingMode = 1;
    break;
  default:
    llvm::report_fatal_error("Unsupported swizzling size.");
  }
  desc.strideDimensionBaseOffset = swizzling >> 1;
  desc.leadDimensionBaseOffset = (swizzling * stride) >> 4;
  return b.int_val(64, desc.descriptor);
}
} // namespace

// The functions below are defined in WGMMA.cpp.
int64_t
getSwizzlingFromLayout(const triton::gpu::NVMMASharedEncodingAttr &layout,
                       uint32_t widthInByte);
ttn::WGMMAEltType getMmaRetType(Value);
ttn::WGMMAEltType getMmaOperandType(Value, bool);

namespace {

// Returns true if the `op` contains an operation in it's regions that satisfies
// the `fn`.
inline bool ContainsOp(mlir::Operation *op,
                       llvm::function_ref<bool(mlir::Operation *)> fn) {
  auto visitor = [&](mlir::Operation *nested_op) {
    return fn(nested_op) ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
  };
  return op->walk(visitor).wasInterrupted();
}
} // namespace

namespace mlir::triton {
#define GEN_PASS_DEF_SPARSEDOTOPTOLLVMPASS
#define GEN_PASS_DEF_SPARSELOCALLOADTOLLVMPASS
#define GEN_PASS_DEF_SPARSEWGMMAOPTOLLVMPASS
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace mlir {
namespace triton {
namespace gpu {

constexpr int kThreadsPerWarp = 32;
// Each 16x16 original sparse matrix tile requires 16 metadata values of
// 16-bit size, where the first thread (T0) in each 4-thread group holds two
// such values in a register (32-bit).
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#sparse-matrix-storage
constexpr int kTileSize = 16;
constexpr int kMetaElementsBitSize = 2;
// Metadata elements are packed into 16-bits values.
constexpr int kMetaElementsPerPackedValue = 16 / kMetaElementsBitSize;
constexpr int kColumnsPerCtaTile = kTileSize / kMetaElementsPerPackedValue;

class SparseLocalLoadToLLVM
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    triton::gpu::MemDescType src_ty = op.getSrc().getType();
    if (!isa<triton::gpu::SharedEncodingTrait>(src_ty.getEncoding()))
      return failure();
    RankedTensorType dst_ty = op.getType();
    if (!isa<SparseDotMetaEncodingAttr>(dst_ty.getEncoding()))
      return failure();
    return lowerSharedToSparseMeta(op, adaptor, rewriter);
  }

private:
  // lowering metadata (local_load: shared -> sparse dot meta) to LLVM
  LogicalResult
  lowerSharedToSparseMeta(triton::gpu::LocalLoadOp op,
                          triton::gpu::LocalLoadOpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    llvm::outs() << " SparseLocalLoadToLLVM\n";
    llvm::outs().flush();

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto load_sparse_encoding = cast<SparseDotMetaEncodingAttr>(
        cast<RankedTensorType>(op.getResult().getType()).getEncoding());

    // Calculate tile size as number of mask elements (4xi4).
    NvidiaMmaEncodingAttr mma_layout =
        cast<NvidiaMmaEncodingAttr>(load_sparse_encoding.getParent());
    SmallVector<unsigned> warps_per_cta = mma_layout.getWarpsPerCTA();

    // Calculate offset in the tile for the current thread.
    Value threads_per_warp = b.i32_val(kThreadsPerWarp);
    Value thread_id = getThreadId(rewriter, loc);
    Value warp_id = b.udiv(thread_id, threads_per_warp);
    Value warp_group_id;
    if (mma_layout.isHopper()) {
      // Hopper MMA instructions force a warp order of [0, 1]. See docs:
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-wgmma-mma-async-m64nnk8
      warp_group_id = b.urem(warp_id, b.i32_val(warps_per_cta[0]));
    } else {
      assert(mma_layout.isAmpere() &&
             "SparseDot is only supported on Ampere and Hopper");
      warp_group_id = b.udiv(warp_id, b.i32_val(warps_per_cta[1]));
    }
    // Calculate row and column id, based on mma.sp.sync.aligned.m16n8k32:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#sparse-mma-metadata-16832-f16bf16.
    // column-id takes into consideration that we pack elements for metadata.
    constexpr int kThreadsInGroup = 4;
    constexpr int kMetadataLineOffset = kThreadsPerWarp / kThreadsInGroup;
    Value lane_id = b.urem(thread_id, threads_per_warp);
    Value lane_group_id = b.udiv(lane_id, b.i32_val(kThreadsInGroup));
    Value row_id =
        b.add(b.mul(warp_group_id, b.i32_val(kTileSize)), lane_group_id);
    SmallVector<unsigned> shape_per_cta_tile = {kTileSize * warps_per_cta[0],
                                                kColumnsPerCtaTile};
    Value column_id = b.urem(lane_id, b.i32_val(shape_per_cta_tile[1]));

    // Calculate number of tile repetitions.
    Value tensor = op.getSrc();
    auto mem_desc = cast<triton::gpu::MemDescType>(tensor.getType());
    auto shape = mem_desc.getShape();
    int rep_m = shape[0] / shape_per_cta_tile[0];
    int rep_k = shape[1] / shape_per_cta_tile[1];
    assert(rep_m > 0 && rep_k > 0);

    // Load sparse metadata from shared memory.
    auto elem_ty = getTypeConverter()->convertType(mem_desc.getElementType());
    auto s_mem_obj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), elem_ty, rewriter);
    const SmallVector<Value> strides =
        s_mem_obj.getStrides(mem_desc, loc, rewriter);
    Value stride_m = strides[0];
    Value stride_k = strides[1];
    MLIRContext *ctx = tensor.getContext();
    Type ptr_ty = ptr_ty(ctx, 3);
    Value base = b.gep(ptr_ty, i16_ty, s_mem_obj.getBase(), b.i32_val(0));
    SmallVector<Value> values;

    for (int k = 0; k < rep_k; ++k) {
      for (int m = 0; m < rep_m; ++m) {
        // Each thread processes two different rows.
        Value row_lower = b.add(row_id, b.i32_val(m * shape_per_cta_tile[0]));
        Value row_upper = b.add(row_lower, b.i32_val(kMetadataLineOffset));
        Value column = b.add(column_id, b.i32_val(k * shape_per_cta_tile[1]));
        Value offset_lower =
            b.add(b.mul(row_lower, stride_m), b.mul(column, stride_k));
        Value offset_upper =
            b.add(b.mul(row_upper, stride_m), b.mul(column, stride_k));
        Value lower = b.load(i16_ty, b.gep(ptr_ty, i16_ty, base, offset_lower));
        Value upper = b.load(i16_ty, b.gep(ptr_ty, i16_ty, base, offset_upper));
        values.push_back(lower);
        values.push_back(upper);
      }
    }

    // Pack resulting values as LLVM struct.
    Type struct_ty = struct_ty(SmallVector<Type>(values.size(), i16_ty));
    Value res =
        packLLElements(loc, getTypeConverter(), values, rewriter, struct_ty);

    auto unpacked = unpackLLElements(loc, res, rewriter);

    rewriter.replaceOp(op, res);
    return success();
  }
};

bool IsLocalLoadWithSparseEncoding(Operation *op) {
  auto local_load = mlir::dyn_cast<triton::gpu::LocalLoadOp>(op);
  if (!local_load)
    return false;
  return isa<SparseDotMetaEncodingAttr>(local_load.getType().getEncoding());
}

/*
struct SparseLocalLoadToLLVMPass
    : public impl::SparseLocalLoadToLLVMPassBase<SparseLocalLoadToLLVMPass> {
  void runOnOperation() override {
    // Exit early if there are no sparse ops.
    ModuleOp mod = getOperation();
    if (!ContainsOp(mod, IsLocalLoadWithSparseEncoding))
      return;

    // Allocate shared memory and set barrier
    // This is also done in the TritonGPUToLLVMPass but we need to do it before
    // we write the local load op to LLVM to have barriers in the right place.
    // See b/358375493.
    ModuleAllocation allocation(getOperation());
    ModuleMembarAnalysis membar_pass(&allocation);
    membar_pass.run();

    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect, mlir::gpu::GPUDialect,
                           arith::ArithDialect>();
    target.addDynamicallyLegalOp<triton::gpu::LocalLoadOp>(
        [](triton::gpu::LocalLoadOp op) {
          return !isa<SparseDotMetaEncodingAttr>(op.getType().getEncoding());
        });
    LowerToLLVMOptions option(context);
    auto compute_capability = getNVIDIAComputeCapability(mod);
    int ptx_version = 0; // TritonGPUToLLVMTypeConverter does not use this.
    mlir::triton::NVIDIA::TargetInfo target_info(compute_capability,
                                                 ptx_version);
    TritonGPUToLLVMTypeConverter type_converter(context, option, target_info);
    auto pattern = std::make_unique<SparseLocalLoadToLLVM>(type_converter);
    RewritePatternSet patterns(context, std::move(pattern));
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
*/

constexpr int kContractingFactor = 2; // implied by N:M (2:4)
constexpr int kCore = 2;              // number of core matrices per batch
constexpr int kCoreTile = kCore * kContractingFactor;

// ----- Ampere implementation.
std::string getMmaSpPtxInstruction(Type type) {
  if (type.isF16()) {
    return "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32";
  } else if (type.isBF16()) {
    return "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32";
  }
  llvm::report_fatal_error("Unsupported SparseDotOp operand type");
}

LogicalResult convertSparseMMA(SparseDotOp op, SparseDotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // Get number of repetitions across the dimensions.
  auto aTensorTy = cast<RankedTensorType>(op.getA().getType());
  auto bTensorTy = cast<RankedTensorType>(op.getB().getType());

  auto layoutA = dyn_cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  auto layoutB = dyn_cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  assert(layoutA != nullptr && layoutB != nullptr);

  int bitwidth = aTensorTy.getElementType().getIntOrFloatBitWidth();
  int kWidth = layoutA.getKWidth();
  auto mmaEnc = cast<NvidiaMmaEncodingAttr>(layoutA.getParent());
  auto repA = mmaEnc.getRepForOperand(triton::gpu::getShapePerCTA(aTensorTy),
                                      bitwidth, kWidth, layoutA.getOpIdx());
  auto repB = mmaEnc.getRepForOperand(triton::gpu::getShapePerCTA(bTensorTy),
                                      bitwidth, kWidth, layoutB.getOpIdx());

  assert(repA[0] == 1 && repB[0] == 1); // batch size
  assert(repB[1] == repA[2] * kContractingFactor);
  int repM = repA[1], repN = repB[2], repK = repB[1];
  int repBatch = repA[0];
  // Arrange loaded values into positions.

  auto ha = getValuesFromDotOperandLayoutStruct(
      typeConverter, loc, rewriter, adaptor.getA(), repBatch, repM,
      repK / kContractingFactor, aTensorTy);
  auto hb = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                adaptor.getB(), repBatch, repN,
                                                repK, bTensorTy);

  // Combine loaded metadata values.
  auto hMeta = unpackLLElements(loc, adaptor.getAMeta(), rewriter);
  SmallVector<Value> hMetaPacked;
  for (int i = 0; i < hMeta.size(); i += kCore) {
    Value lower = b.zext(i32_ty, hMeta[i]);
    Value upper = b.zext(i32_ty, hMeta[i + 1]);
    Value packed = b.or_(b.shl(upper, b.i32_val(16)), lower);
    hMetaPacked.push_back(packed);
  }

  // Flatten accumulator values.
  auto fc = unpackLLElements(loc, adaptor.getC(), rewriter);

  // Create `mma.sp` instruction for 4/8 core matrices.
  auto callMma = [&](int m, int n, int k) {
    triton::PTXBuilder builder;
    auto &mma =
        *builder.create(getMmaSpPtxInstruction(aTensorTy.getElementType()));

    auto retArgs = builder.newListOperand(kCoreTile, "=f");
    auto cArgs = builder.newListOperand();
    int baseIdx = m * repN * kCore + n * kCoreTile;
    for (int i = 0; i < kCoreTile; ++i) {
      cArgs->listAppend(builder.newOperand(fc[baseIdx + i], std::to_string(i)));
    }
    int b_ = 0;
    int i = k / kContractingFactor;
    auto aArgs = builder.newListOperand({
        {ha.at({b_, m, i}), "r"},
        {ha.at({b_, m + 1, i}), "r"},
        {ha.at({b_, m, i + 1}), "r"},
        {ha.at({b_, m + 1, i + 1}), "r"},
    });
    auto bArgs = builder.newListOperand({
        {hb.at({b_, n, k}), "r"},
        {hb.at({b_, n, k + 1}), "r"},
        {hb.at({b_, n, k + 2}), "r"},
        {hb.at({b_, n, k + 3}), "r"},
    });
    auto metaArg =
        builder.newOperand(hMetaPacked[k / kCoreTile * repM + m / kCore], "r");
    auto selector = builder.newConstantOperand(0);
    mma(retArgs, aArgs, bArgs, cArgs, metaArg, selector);

    Type fp32x4Ty = LLVM::LLVMStructType::getLiteral(
        op.getContext(), SmallVector<Type>(kCoreTile, f32_ty));
    Value mmaOut = builder.launch(rewriter, loc, fp32x4Ty);
    for (int i = 0; i < kCoreTile; ++i) {
      fc[baseIdx + i] = b.extract_val(f32_ty, mmaOut, i);
    }
  };

  for (int k = 0; k < repK; k += kContractingFactor)
    for (int m = 0; m < repM; ++m)
      for (int n = 0; n < repN; ++n)
        callMma(kCore * m, n, kCore * k);

  // Replace with new packed result.
  Type structTy = LLVM::LLVMStructType::getLiteral(
      op.getContext(), SmallVector<Type>(fc.size(), f32_ty));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);
  rewriter.replaceOp(op, res);

  return success();
}

// ----- Hopper implementation.

constexpr int kWarpsInGroup = 4;
constexpr int kMmaAccumulatorCount = 2;
constexpr int kMmaLineSize = 128;
constexpr int kMmaAlignment = 16;

// TODO(sparsity): what is this even used for and is it correct
constexpr int kMmaPerPhase = 1;

// Shared memory descriptor builder for WGMMA.
Value smemDescriptor(int A, int B, ConversionPatternRewriter &rewriter,
                     Location loc, std::vector<unsigned int> instrShape,
                     bool trans, int dimWpt, Value warpId,
                     triton::gpu::MemDescType tensorTy, Value baseDesc,
                     int minor) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto sharedLayout = cast<NVMMASharedEncodingAttr>(tensorTy.getEncoding());
  int elemBytes = tensorTy.getElementTypeBitWidth() / 8;
  int elemsPerSwizzlingRow = kMmaLineSize / kMmaPerPhase / elemBytes;
  Value elemsPerSwizzlingRowVal = b.i32_val(elemsPerSwizzlingRow);

  Value k = b.i32_val(B * instrShape[1]);
  Value m = b.add(b.i32_val(A * dimWpt * instrShape[0]),
                  b.mul(warpId, b.i32_val(instrShape[0])));
  if (trans) {
    std::swap(k, m);
  }
  Value leading_offset = b.mul(b.udiv(k, elemsPerSwizzlingRowVal),
                               b.i32_val(minor * elemsPerSwizzlingRow));
  Value stride_offset = b.mul(m, elemsPerSwizzlingRowVal);
  Value offset = b.add(b.add(leading_offset, stride_offset),
                       b.urem(k, elemsPerSwizzlingRowVal));
  Value off1 = b.mul(b.i32_val(elemBytes), offset);
  Value off_ = b.zext(i64_ty, b.udiv(off1, b.i32_val(kMmaAlignment)));

  return b.add(baseDesc, off_);
}

LogicalResult convertSparseWGMMA(ttng::SparseWarpGroupDotOp op,
                                 ttng::SparseWarpGroupDotOp::Adaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter,
                                 Value thread) {
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // Get number of repetitions across the dimensions.
  auto aTensorTy = cast<triton::gpu::MemDescType>(op.getA().getType());
  auto bTensorTy = cast<triton::gpu::MemDescType>(op.getB().getType());
  auto dTensorTy = cast<RankedTensorType>(op.getD().getType());
  auto mmaEnc = cast<NvidiaMmaEncodingAttr>(dTensorTy.getEncoding());

  auto shapePerCTA = getShapePerCTA(dTensorTy);
  auto instrMNK = mmaEnc.getInstrShape();
  auto warpSize = mmaEnc.getWarpsPerCTA();
  auto shapePerCTATile = SmallVector<unsigned>{instrMNK[0] * warpSize[0],
                                               instrMNK[1] * warpSize[1]};
  auto instrShape = mmaEnc.getInstrShape();
  // TODO(sparsity): use repPerOperand or whatever it's called
  int repM = ceil<unsigned>(shapePerCTA[0], shapePerCTATile[0]);
  int repN = ceil<unsigned>(shapePerCTA[1], shapePerCTATile[1]);
  int repK = ceil<unsigned>(bTensorTy.getShape()[0],
                            instrShape[2] * kContractingFactor);

  // Flatten accumulator values.
  auto fc = unpackLLElements(loc, adaptor.getC(), rewriter);
  int accSize = kMmaAccumulatorCount * (instrShape[1] / kWarpsInGroup);
  assert(fc.size() == repM * repN * accSize);

  // Get warp ID.
  auto wpt = mmaEnc.getWarpsPerCTA();
  Value warp =
      b.and_(b.udiv(thread, b.i32_val(kThreadsPerWarp)), b.i32_val(0xFFFFFFFC));
  Value warpM = b.urem(warp, b.i32_val(wpt[0]));
  Value warpMN = b.udiv(warp, b.i32_val(wpt[0]));
  Value warpN = b.urem(warpMN, b.i32_val(wpt[1]));

  // Create descriptor.
  auto getSharedData = [&](Value arg, triton::gpu::MemDescType tensorTy) {
    auto sharedObj = getSharedMemoryObjectFromStruct(
        loc, arg, typeConverter->convertType(tensorTy.getElementType()),
        rewriter);
    auto sharedLayout = cast<NVMMASharedEncodingAttr>(tensorTy.getEncoding());
    auto shape = getShapePerCTA(tensorTy);
    auto ord = sharedLayout.getOrder();
    int byteSize = aTensorTy.getElementTypeBitWidth() / 8;
    int64_t swizzling =
        getSwizzlingFromLayout(sharedLayout, shape[ord[0]] * byteSize);
    Value baseDesc = createDescriptor(rewriter, loc, swizzling, shape[ord[1]]);
    baseDesc = b.add(baseDesc, b.lshr(b.ptrtoint(i64_ty, sharedObj.getBase()),
                                      b.int_val(64, 4)));
    return std::make_tuple(shape, ord, baseDesc);
  };

  // Create descriptor for loading A from shared memory.
  auto tA = getSharedData(adaptor.getA(), aTensorTy);
  Value warpA = b.urem(warpM, b.i32_val(std::get<0>(tA)[0] / instrShape[0]));
  bool transA = std::get<1>(tA)[0] == 0;
  auto loadA = [&](int m, int k) {
    return smemDescriptor(m, k, rewriter, loc, {instrShape[0], instrShape[2]},
                          transA, wpt[0], warpA, aTensorTy, std::get<2>(tA),
                          std::get<0>(tA)[std::get<1>(tA)[1]]);
  };

  // Create descriptor for loading B from shared memory.
  auto tB = getSharedData(adaptor.getB(), bTensorTy);
  Value warpB = b.urem(warpN, b.i32_val(std::get<0>(tB)[1] / instrShape[1]));
  bool transB = std::get<1>(tB)[0] == 1;
  auto loadB = [&](int n, int k) {
    return smemDescriptor(n, k, rewriter, loc,
                          {instrShape[1], instrShape[2] * kContractingFactor},
                          transB, wpt[1], warpB, bTensorTy, std::get<2>(tB),
                          std::get<0>(tB)[std::get<1>(tB)[1]]);
  };

  // Load metadata from shared memory.
  // TODO(sparsity): remove this debug
  auto aMetaTensor = cast<RankedTensorType>(op.getAMeta().getType());
  auto aMetaTensorShape = aMetaTensor.getShape();
  auto linearEncoding = toLinearEncoding(aMetaTensor.getEncoding(), aMetaTensor.getShape());
  auto linearElemsPerThread = linearEncoding.getElemsPerThread(aMetaTensorShape);
  auto linearLayout = linearEncoding.toLinearLayout(aMetaTensorShape);
  auto dim0Size = linearLayout.getOutDimSize(StringAttr::get(op.getContext(), "dim0"));
  auto dim1Size = linearLayout.getOutDimSize(StringAttr::get(op.getContext(), "dim1"));
  auto expectedTotalElemsPerThread = getTotalElemsPerThread(op.getAMeta().getType());
  auto hMeta = unpackLLElements(loc, adaptor.getAMeta(), rewriter);

  op->getParentOp()->dump();
  llvm::outs().flush();

  SmallVector<Value> hMetaPacked;
  for (int i = 0; i < hMeta.size(); i += kCore) {
    Value lower = b.zext(i32_ty, hMeta[i]);
    Value upper = b.zext(i32_ty, hMeta[i + 1]);
    Value packed = b.or_(b.shl(upper, b.i32_val(16)), lower);
    hMetaPacked.push_back(packed);
  }
  if (hMetaPacked.size() != repM * repK) {
    llvm::outs() << "  hMetaPacked.size() wrong " << "\n";
    llvm::outs().flush();
  }
  assert(hMetaPacked.size() == repM * repK);

  // Generate prologue.
  ttn::WGMMAEltType eltTypeA = getMmaOperandType(op.getA(), false);
  ttn::WGMMAEltType eltTypeB = getMmaOperandType(op.getB(), false);
  ttn::WGMMAEltType eltTypeC = getMmaRetType(op.getD());

  ttn::WGMMALayout layoutA =
      transA ? ttn::WGMMALayout::col : ttn::WGMMALayout::row;
  ttn::WGMMALayout layoutB =
      transB ? ttn::WGMMALayout::row : ttn::WGMMALayout::col;

  rewriter.create<ttn::FenceAsyncSharedOp>(loc, 0);
  rewriter.create<ttn::WGMMAFenceOp>(loc);

  // Generate main loop.
  for (int m = 0; m < repM; ++m) {
    for (int n = 0; n < repN; ++n) {
      llvm::MutableArrayRef acc(&fc[(m * repN + n) * accSize], accSize);
      auto accTy = LLVM::LLVMStructType::getLiteral(
          op.getContext(), SmallVector<Type>(accSize, f32_ty));
      Value d = packLLElements(loc, typeConverter, acc, rewriter, accTy);
      for (int k = 0; k < repK; ++k) {
        Value a = loadA(m, k);
        Value b = loadB(n, k);
        Value meta = hMetaPacked[k * repM + m];
        d = rewriter.create<ttn::SparseWGMMAOp>(
            loc, accTy, a, meta, b, d, kWarpsInGroup * instrShape[0],
            instrShape[1], kContractingFactor * instrShape[2], eltTypeC,
            eltTypeA, eltTypeB, layoutA, layoutB);
      }
      auto res = unpackLLElements(loc, d, rewriter);
      for (int i = 0; i < res.size(); ++i) {
        acc[i] = res[i];
      }
    }
  }

  // Replace with new packed result.
  Type structTy = LLVM::LLVMStructType::getLiteral(
      op.getContext(), SmallVector<Type>(fc.size(), f32_ty));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

  rewriter.create<ttn::WGMMACommitGroupOp>(loc);
  res = rewriter.create<ttn::WGMMAWaitGroupOp>(loc, res, 0);
  rewriter.replaceOp(op, res);

  return success();
}

// ----- Dispatch based on architecture.

LogicalResult rewriteSparseDotOp(SparseDotOp op, SparseDotOp::Adaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) {
  auto resultTy = cast<RankedTensorType>(op.getResult().getType());
  NvidiaMmaEncodingAttr mmaLayout =
      cast<NvidiaMmaEncodingAttr>(resultTy.getEncoding());

  assert(mmaLayout.isAmpere() &&
         "Only Ampere Sparse MMA (vs. WGMMA) is supported");
  return convertSparseMMA(op, adaptor, typeConverter, rewriter);
}

struct SparseDotOpConversion : public ConvertOpToLLVMPattern<SparseDotOp> {
  using ConvertOpToLLVMPattern<SparseDotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SparseDotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteSparseDotOp(op, adaptor, getTypeConverter(), rewriter);
  }
};

LogicalResult
rewriteSparseWarpGroupDotOp(ttng::SparseWarpGroupDotOp op,
                            ttng::SparseWarpGroupDotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto resultTy = cast<RankedTensorType>(op.getResult().getType());
  NvidiaMmaEncodingAttr mmaLayout =
      cast<NvidiaMmaEncodingAttr>(resultTy.getEncoding());

  assert(mmaLayout.isHopper() && "Only Hopper Sparse WGMMA is supported");
  return convertSparseWGMMA(op, adaptor, typeConverter, rewriter,
                            getThreadId(rewriter, op.getLoc()));
}

struct SparseWarpGroupDotOpConversion
    : public ConvertOpToLLVMPattern<ttng::SparseWarpGroupDotOp> {
  using ConvertOpToLLVMPattern<
      ttng::SparseWarpGroupDotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::SparseWarpGroupDotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "  do SparseWarpGroupDotOpConversion\n";
    llvm::outs().flush();
    return rewriteSparseWarpGroupDotOp(op, adaptor, getTypeConverter(),
                                       rewriter);
  }
};

/*
struct SparseDotOpToLLVMPass
    : public impl::SparseDotOpToLLVMPassBase<SparseDotOpToLLVMPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect, NVVM::NVVMDialect,
                           arith::ArithDialect, ttn::NVGPUDialect>();
    target.addIllegalOp<SparseDotOp>();
    target.addIllegalDialect<mlir::gpu::GPUDialect>();
    LowerToLLVMOptions option(context);
    ModuleOp module = getOperation();
    auto computeCapability = getNVIDIAComputeCapability(module);
    int ptx_version = 0; // TritonGPUToLLVMTypeConverter does not use this.
    mlir::triton::NVIDIA::TargetInfo targetInfo(computeCapability, ptx_version);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    RewritePatternSet patterns(context);
    patterns.add<SparseDotOpConversion,
SparseWarpGroupDotOpConversion>(typeConverter);
    // TODO(b/358375493): Remove this once TritonGPUToLLVMTypeConverter is
    // splitted into smaller passes.
    populateGpuToNVVMConversionPatterns(typeConverter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
*/

// TODO(sparsity): move this probably?
class SparseWGMMAOpPattern : public OpRewritePattern<ttn::SparseWGMMAOp> {
public:
  using OpRewritePattern<ttn::SparseWGMMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::SparseWGMMAOp op,
                                PatternRewriter &rewriter) const override {
    return rewriteAsPtxAsm(op, rewriter, getPtxAsm(op),
                           getOperandsAndConstraints(op),
                           getOutputConstraints(op));
  }

  std::vector<std::string> getOutputConstraints(ttn::SparseWGMMAOp op) const {
    auto outputStructType = cast<LLVM::LLVMStructType>(op.getType());
    uint32_t numOutputRegs = outputStructType.getBody().size();
    std::string output =
        outputStructType.getBody().front().isF32() ? "=f" : "=r";
    return std::vector<std::string>(numOutputRegs, output);
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::SparseWGMMAOp op) const {
    return {{op.getOpC(), "0"},
            {op.getOpA(), "l"},
            {op.getOpB(), "l"},
            {op.getMetaA(), "r"}};
  }

  std::string getPtxAsm(ttn::SparseWGMMAOp op) const {
    auto m = op.getM();
    auto n = op.getN();
    auto k = op.getK();
    auto eltTypeC = op.getEltTypeC();
    auto eltTypeA = op.getEltTypeA();
    auto eltTypeB = op.getEltTypeB();
    auto layoutA = op.getLayoutA();
    auto layoutB = op.getLayoutB();

    // Only f16/bf16 variant is supported.
    using WGMMAEltType = ttn::WGMMAEltType;
    [[maybe_unused]] bool supported =
        eltTypeC == WGMMAEltType::f32 &&
        ((eltTypeA == WGMMAEltType::f16 && eltTypeB == WGMMAEltType::f16) ||
         (eltTypeA == WGMMAEltType::bf16 && eltTypeB == WGMMAEltType::bf16)) &&
        (m == 64 && 8 <= n && n <= 256 && n % 8 == 0 && k == 32);
    assert(supported && "Sparse WGMMA type or shape is not supported");

    // Operands
    uint32_t asmOpIdx = 0;
    std::string args = "";

    // Output and operand C
    uint32_t numCRegs =
        cast<LLVM::LLVMStructType>(op.getType()).getBody().size();
    args += "{";
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    args += "}, ";
    asmOpIdx += numCRegs;

    // Operands A and B (must be `desc`)
    args += "$" + std::to_string(asmOpIdx++) + ", ";
    args += "$" + std::to_string(asmOpIdx++) + ", ";

    // Metadata for A
    args += "$" + std::to_string(asmOpIdx++) + ", 0, ";

    // `scale-d`, `imm-scale-a`, and `imm-scale-b` are 1 by default
    args += "1, 1, 1";

    // `trans-a` and `trans-b`
    using WGMMALayout = ttn::WGMMALayout;
    args += ", " + std::to_string(layoutA == WGMMALayout::col);
    args += ", " + std::to_string(layoutB == WGMMALayout::row);

    auto ptxAsm = "wgmma.mma_async.sp.sync.aligned"
                  ".m" +
                  std::to_string(m) + "n" + std::to_string(n) + "k" +
                  std::to_string(k) + "." + stringifyEnum(eltTypeC).str() +
                  "." + stringifyEnum(eltTypeA).str() + "." +
                  stringifyEnum(eltTypeB).str() + " " + args + ";";
    return ptxAsm;
  }
};

struct SparseWGMMAOpToLLVMPass
    : public impl::SparseWGMMAOpToLLVMPassBase<SparseWGMMAOpToLLVMPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto pattern = std::make_unique<SparseWGMMAOpPattern>(context);
    RewritePatternSet patterns(context, std::move(pattern));
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace gpu

namespace NVIDIA {
void populateSparseDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit) {
  patterns.add<ttg::SparseLocalLoadToLLVM>(typeConverter, benefit);
  patterns.add<ttg::SparseDotOpConversion>(typeConverter, benefit);
  patterns.add<ttg::SparseWarpGroupDotOpConversion>(typeConverter, benefit);
}
} // namespace NVIDIA

/*
std::unique_ptr<Pass> CreateSparseLocalLoadToLLVMPass() {
  return std::make_unique<gpu::SparseLocalLoadToLLVMPass>();
}

std::unique_ptr<Pass> CreateSparseDotOpToLLVMPass() {
  return std::make_unique<gpu::SparseDotOpToLLVMPass>();
}
*/

std::unique_ptr<Pass> CreateSparseWGMMAOpToLLVMPass() {
  return std::make_unique<gpu::SparseWGMMAOpToLLVMPass>();
}

} // namespace triton
} // namespace mlir
