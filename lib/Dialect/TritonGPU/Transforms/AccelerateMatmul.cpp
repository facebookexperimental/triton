#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/NvmmaSmemAttrs.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace triton {
namespace gpu {

namespace {

static bool isUnsupportedMMAv5Int8Dot(int computeCapability, DotOp op) {
  if (nvidia_gpu::TargetFeatures(computeCapability).supportsI8Tcgen05MMA())
    return false;
  auto aElemTy = op.getA().getType().getElementType();
  auto bElemTy = op.getB().getType().getElementType();
  return aElemTy.isInteger(8) && bElemTy.isInteger(8);
}

// List supported mma version in order of preference.
static SmallVector<int> getMMAVersionsSupported(int computeCapability,
                                                DotOp op) {
  SmallVector<int> versionsSupported;
  if (computeCapability < 75) {
    versionsSupported = {1};
  } else if (computeCapability < 90) {
    versionsSupported = {2};
  } else if (computeCapability < 100) {
    versionsSupported = {3, 2};
  } else if (computeCapability < 120) {
    // Exclude consumer Blackwell (sm120)
    if (isUnsupportedMMAv5Int8Dot(computeCapability, op)) {
      versionsSupported = {2};
    } else {
      versionsSupported = {5, 2};
    }
  } else if (computeCapability < 130) {
    versionsSupported = {2};
  } else {
    assert(false && "computeCapability not supported");
  }
  return versionsSupported;
}

// Get the highest version supported for the hardware and the dot.
static int getMMAVersionSafe(int computeCapability, DotOp op) {
  for (int baseVersion : getMMAVersionsSupported(computeCapability, op)) {
    if (supportMMA(op, baseVersion))
      return baseVersion;
    if (baseVersion == 3) {
      auto remark = op.emitRemark()
                    << "MMA version 3 acceleration not applied due to "
                       "unsupported shapes or data types.";
      remark.attachNote() << "Target compute capability (" << computeCapability
                          << ") supports MMA v3.";
    }

    if (baseVersion == 5) {
      auto remark = op.emitRemark()
                    << "MMA version 5 acceleration not applied due to "
                       "unsupported shapes or data types.";
      remark.attachNote() << "Target compute capability (" << computeCapability
                          << ") supports MMA v5.";
    }
  }
  return 0;
}

SmallVector<unsigned> warpsPerTileV2(DotOpInterface dotOp,
                                     const ArrayRef<int64_t> shape,
                                     int numWarps) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return getMmaV2WarpsPerCTA(shape, numWarps);

  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion() &&
           !isa<TransOp>(op);
  };
  auto slices = mlir::getSlice(dotOp, {filter}, {filter});
  bool hasChainedDot = false;
  for (Operation *op : slices) {
    if (isa<DotOp, DotScaledOp>(op) && (op != dotOp)) {
      auto resTy = cast<RankedTensorType>(op->getResult(0).getType());
      if (resTy.getRank() != rank) {
        continue;
      }
      if (auto mmaEncoding =
              dyn_cast<NvidiaMmaEncodingAttr>(resTy.getEncoding())) {
        return to_vector(mmaEncoding.getWarpsPerCTA());
      }
      hasChainedDot = true;
    }
  }
  if (hasChainedDot) {
    if (shape[0] >= shape[1]) {
      return {(unsigned)numWarps, 1};
    } else {
      return {1, (unsigned)numWarps};
    }
  }

  return getMmaV2WarpsPerCTA(shape, numWarps);
}
SmallVector<unsigned, 2>
warpsPerTileV3(DotOpInterface dotOp, const ArrayRef<int64_t> shape,
               int numWarps, const SmallVector<unsigned, 3> &instrShape) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getD(), &slices);
  // Contains a chained dot. We prefer to assign warps to one axis
  // to facilitate use cases like flash attention, allowing reductions within
  // the same warp.
  if (llvm::find_if(slices, [](Operation *op) {
        return isa<mlir::triton::DotOpInterface>(op);
      }) != slices.end())
    return {(unsigned)numWarps, 1};
  if (hasMmaV3WarpgroupsOnN(shape, numWarps))
    return {4, (unsigned)numWarps / 4};

  // For MMAv3, the smallest indivisible unit of warp shape is (4, 1).
  SmallVector<unsigned, 2> ret = {4, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, instrShape[1]};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] > shapePerWarp[0] * ret[0]) {
      ret[0] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

// Returns a shared memory allocation that can be used by a dotMMA op for the
// given value.
static Value
getSharedMemoryMMAOperand(Value v, mlir::PatternRewriter &rewriter, int opIdx,
                          bool allowTranspose, bool isMMAv5Fp4Padded = false,
                          bool forceTranspose = false,
                          Operation *op = nullptr /*only for diagnostic*/) {
  OpBuilder::InsertionGuard g(rewriter);
  Value arg = v;
  while (auto cvtOp = arg.getDefiningOp<ConvertLayoutOp>())
    arg = cvtOp.getSrc();
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto order = getOrderForMemory(argType);

  // If the MMA op doesn't support transpose pick the layout expected by the MMA
  // op.
  llvm::SmallVector<unsigned> newOrder = order;
  if (!allowTranspose) {
    if (opIdx == 1) {
      newOrder = {0, 1};
    } else {
      newOrder = {1, 0};
    }
    if (forceTranspose)
      std::swap(newOrder[0], newOrder[1]);
  }

  if (newOrder != order && op) {
    op->emitWarning("Warning: Forcing a different order [")
        << newOrder[0] << ", " << newOrder[1]
        << "] on SMEM than the register order for the operand " << opIdx
        << ". Registers will be transposed before SMEM store and the pipelined "
           "load for this operand will be disabled, so poor performance is "
           "expected. Recommendation: consider transposing the operand in "
           "global "
           "memory to remove the need to transpose the tensor in registers.";
  }

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CGALayout = getCGALayout(argType.getEncoding());
  auto newLayout = NVMMASharedEncodingAttr::get(
      argType.getContext(), argType.getShape(), newOrder, CGALayout,
      argType.getElementType(), isMMAv5Fp4Padded);
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return LocalAllocOp::create(rewriter, arg.getLoc(), newType, arg);
}

static bool isTmemCopyCompatibleScale(RankedTensorType argType,
                                      bool usesTMAload) {
  assert(argType.getEncoding() && "unexpected tensor type");
  {
    auto shmemLayout = NVMMASharedEncodingAttr::get(
        argType.getContext(), /*swizzlingByteWidth=*/0,
        /*transposed=*/false,
        /*elementBitWidth=*/8,
        /*fp4Padded=*/false, getCGALayout(argType.getEncoding()));
    auto type = MemDescType::get(
        argType.getShape(), argType.getElementType(), shmemLayout,
        SharedMemorySpaceAttr::get(argType.getContext()));
    if (!isInnermostContiguous(type, 512)) {
      // TMEM copy expects metadata "chunks" in SMEM to be stored contiguously
      // in the innermost axes.
      return false;
    }
  }

  if (usesTMAload)
    return true;

  if (argType.getRank() != 2) {
    // TODO: Add support for higher rank when 5D coalesced load is fixed.
    return false;
  }

  auto innermostBits = argType.getDimSize(argType.getRank() - 1) * 8;
  return innermostBits % (32 * 128) == 0;
}

template <typename OpTy>
static OpTy getDefiningOpSkippingConvertLayout(Value value) {
  while (auto cvtOp = value.getDefiningOp<ConvertLayoutOp>())
    value = cvtOp.getSrc();
  return value.getDefiningOp<OpTy>();
}

static Value stripConvertLayout(Value value) {
  while (auto cvtOp = value.getDefiningOp<ConvertLayoutOp>())
    value = cvtOp.getSrc();
  return value;
}

static Value
tryCreateTmemCopyCompatibleScaleOperand(Value scale,
                                        mlir::PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  // The final reshape to conform to the logical shape requirement of
  // dot_scaled.
  auto reshape2D = getDefiningOpSkippingConvertLayout<ReshapeOp>(scale);
  if (!reshape2D)
    return Value();

  auto scaleType = dyn_cast<RankedTensorType>(reshape2D.getResult().getType());
  if (!scaleType || scaleType.getShape().size() != 2)
    return Value();

  auto scale2DShape = scaleType.getShape();
  auto blockMN = scale2DShape[0];
  auto numScales = scale2DShape[1];
  if (blockMN % 128 != 0 || numScales % 4 != 0)
    return Value();

  // Permute to expose the 32x4x4 interleaving used by scale blocks in TMEM.
  auto transOp =
      getDefiningOpSkippingConvertLayout<TransOp>(reshape2D.getSrc());
  if (!transOp || transOp.getOrder() != ArrayRef<int32_t>({0, 3, 2, 1, 4}))
    return Value();

  // A logical scale tensor has shape [block_mn, num_scales]. The tensor view
  // chain in a user kernel signals TMEMCopy compatibility by exposing the
  // underlying 128x4 scale block as
  //     --reshape
  //   [mn_copy_tiles, k_copy_tiles, 32, 4, 4]
  //     --trans {0, 3, 2, 1, 4}-->
  //   [mn_copy_tiles, 4, 32, k_copy_tiles, 4]
  //     --flatten-->
  //   [block_mn, num_scales]
  //
  // The block shape with which the scale tensor is loaded is flexible:
  // For example, [.., 2, 256], [.., 512], and [.., 32, 4, 4] are all acceptable
  // as long as a reshape exposes the same 32x4x4 blocks before the transpose.
  auto mnCopyTiles = blockMN / 128;
  auto kCopyTiles = numScales / 4;
  SmallVector<int64_t> tiledScaleShape = {mnCopyTiles, kCopyTiles, 32, 4, 4};

  Value tiledScale = stripConvertLayout(transOp.getSrc());
  auto reshapeTiled = tiledScale.getDefiningOp<ReshapeOp>();
  if (!reshapeTiled ||
      reshapeTiled.getType().getShape() != ArrayRef<int64_t>(tiledScaleShape))
    return Value();

  Value packedScale = stripConvertLayout(reshapeTiled.getSrc());
  auto packedScaleType = dyn_cast<RankedTensorType>(packedScale.getType());
  if (!packedScaleType)
    return Value();

  auto *loadOp = packedScale.getDefiningOp();
  if (!isa_and_nonnull<LoadOp, DescriptorLoadLikeOpInterface>(loadOp))
    return Value();

  bool usesTMAload = isa<DescriptorLoadLikeOpInterface>(loadOp);
  if (!isTmemCopyCompatibleScale(packedScaleType, usesTMAload))
    return Value();

  auto sharedMemory = SharedMemorySpaceAttr::get(scaleType.getContext());
  auto sinkLayout = getScaleSmemLayoutForTMEMCopy(
      scaleType.getContext(), scaleType.getShape(),
      getCGALayout(scaleType.getEncoding()));
  auto layout = SharedLinearEncodingAttr::get(scaleType.getContext(),
                                              std::move(sinkLayout),
                                              /*alignment=*/128);
  auto sinkType =
      MemDescType::get(scaleType.getShape(), scaleType.getElementType(), layout,
                       sharedMemory, /*mutableMemory=*/false);
  rewriter.setInsertionPointAfterValue(reshape2D.getResult());
  return LocalAllocOp::create(rewriter, reshape2D.getLoc(), sinkType,
                              reshape2D.getResult());
}

static Value createTmemScaleOperand(Value tensor, Attribute tmemEncoding,
                                    Attribute tensorMemorySpace, int numWarps,
                                    Location loc, PatternRewriter &rewriter) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto tmemType =
      MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                       tmemEncoding, tensorMemorySpace,
                       /*mutableMemory=*/false);
  auto tmemLayout = nvidia_gpu::getDefaultLayoutForTmemLdSt(tmemType, numWarps);
  auto stagedType = tensorType.cloneWithEncoding(tmemLayout);
  Value converted = ConvertLayoutOp::create(rewriter, loc, stagedType, tensor);
  return triton::nvidia_gpu::TMEMAllocOp::create(rewriter, loc, tmemType,
                                                 /*token=*/Type(), converted);
}

SmallVector<unsigned, 3>
getWarpsPerTile(DotOpInterface dotOp, const ArrayRef<int64_t> shape,
                int version, int numWarps,
                const SmallVector<unsigned, 3> &instrShape) {
  switch (version) {
  case 2:
    return warpsPerTileV2(dotOp, shape, numWarps);
  case 3:
    return warpsPerTileV3(dotOp, shape, numWarps, instrShape);
  default:
    assert(false && "not supported version");
    return {0, 0};
  }
}

static bool bwdFilter(Operation *op) {
  return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
         isView(op) ||
         isa<Fp4ToFpOp, LoadOp, DescriptorLoadLikeOpInterface, BroadcastOp,
             ConvertLayoutOp>(op);
}

static bool joinPacksLastDim(JoinOp join) {
  auto type = cast<RankedTensorType>(join.getResult().getType());
  unsigned rank = type.getRank();
  if (rank < 2)
    return false;

  auto order = getOrder(type);
  // Check that the appended join axis and the previous last axis are the two
  // fastest-changing axes, so the join packs the last dimension contiguously.
  return order[0] == rank - 1 && order[1] == rank - 2;
}

static bool canWidenKWidthForJoin(int loadBitWidth,
                                  const SetVector<Operation *> &slice) {
  return llvm::any_of(slice, [loadBitWidth](Operation *op) {
    auto join = dyn_cast<JoinOp>(op);
    if (!join)
      return false;
    auto type = cast<RankedTensorType>(join.getResult().getType());
    return type.getElementTypeBitWidth() == loadBitWidth * 2 &&
           joinPacksLastDim(join);
  });
}

// Finds the bitwidth with which the value x is loaded
static int computeOrigBitWidth(Value x) {
  SetVector<Operation *> slice;
  mlir::BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.filter = bwdFilter;
  (void)getBackwardSlice(x, &slice, opt);

  // TODO: This heuristic may be a bit too coarse and may need improving
  // If the chain contains a fp4 to fp16/bf16 conversion, then the original
  // bitwidth is 4.
  if (llvm::any_of(slice, [](Operation *op) { return isa<Fp4ToFpOp>(op); }))
    return 4;

  int origBitWidth = getElementTypeOrSelf(x).getIntOrFloatBitWidth();
  for (auto op : slice) {
    if (isa<LoadOp, DescriptorLoadLikeOpInterface>(op)) {
      if (auto tensorTy =
              dyn_cast<RankedTensorType>(op->getResultTypes().front())) {
        origBitWidth =
            std::min<int>(origBitWidth, tensorTy.getElementTypeBitWidth());
      }
    }
  }

  // If JoinOp occurred at least once, in backward layout propagation,
  // the kWidth will be split in half as we pass through the JoinOp.
  // Hence we divide origBitWidth by 2 here to compensate for that and
  // improve our load width.
  // This won't be optimal if there is a tree of multiple JoinOps, which
  // would require counting the max number of JoinOp's along any path.
  //
  // In the future we might want to do something like trying a large kWidth,
  // run layout backpropagation and see what's the contiguity that you
  // get at the loads that feed into it.
  //
  // This heuristic is intended for packed-K values of shape [..., K / 2]
  // joined into [..., K / 2, 2]. We also check that the bitwidth of the joined
  // value is twice the original bitwidth, meaning "one loaded storage element
  // contributes two logical K values after unpack/join".
  if (canWidenKWidthForJoin(origBitWidth, slice))
    origBitWidth /= 2;

  return origBitWidth;
}

namespace {

// Common MMA encoding creation
struct MMAEncodingResult {
  NvidiaMmaEncodingAttr mmaEnc;
  RankedTensorType newRetType;
  Value newAcc;
  int versionMajor;
  int versionMinor;
};

// Unified implementation for DotOpInterface
static MMAEncodingResult createMMAEncodingForDot(DotOpInterface dotOp,
                                                 PatternRewriter &rewriter,
                                                 int computeCapability,
                                                 int versionMajor) {
  auto oldRetType = cast<RankedTensorType>(dotOp.getD().getType());
  auto oldAType = cast<RankedTensorType>(dotOp.getA().getType());

  int numWarps = lookupNumWarps(dotOp);

  int versionMinor = computeCapability == 75 ? 1 : 0;
  // Only MMAv2 and MMAv3 rely on computing instrShape/warpsPerTile here.
  if (!(versionMajor == 2 || versionMajor == 3)) {
    return {nullptr, RankedTensorType(), Value(), versionMajor, versionMinor};
  }

  auto CGALayout = getCGALayout(oldRetType.getEncoding());
  auto retShapePerCTA = getShapePerCTA(oldRetType);
  auto instrShape = mmaVersionToInstrShape(versionMajor, retShapePerCTA,
                                           oldAType.getElementType(), numWarps);
  auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                      numWarps, instrShape);

  auto mmaEnc = NvidiaMmaEncodingAttr::get(oldRetType.getContext(),
                                           versionMajor, versionMinor,
                                           warpsPerTile, CGALayout, instrShape);
  auto newRetType = oldRetType.cloneWithEncoding(mmaEnc);

  auto oldAcc = dotOp->getOperand(2);
  auto newAcc =
      ConvertLayoutOp::create(rewriter, oldAcc.getLoc(), newRetType, oldAcc);

  return {mmaEnc, newRetType, newAcc, versionMajor, versionMinor};
}

// Common operand conversion
static Value convertDotOperandForMMA(Value v, int opIdx, int bitwidth,
                                     RankedTensorType newRetType,
                                     PatternRewriter &rewriter) {
  auto minType = bitwidth > 0 ? rewriter.getIntegerType(bitwidth) : v.getType();
  auto vType = cast<RankedTensorType>(v.getType());
  auto newVEncoding = DotOperandEncodingAttr::get(
      v.getContext(), opIdx, newRetType.getEncoding(), minType);
  auto newVType = vType.cloneWithEncoding(newVEncoding);
  return ConvertLayoutOp::create(rewriter, v.getLoc(), newVType, v);
}

} // namespace

class BlockedToMMA : public mlir::OpRewritePattern<DotOp> {
  int computeCapability;
  mutable llvm::DenseMap<Operation *, unsigned> dotOpInstNs;

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability, int benefit)
      : OpRewritePattern<DotOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    if (computeCapability < 80) {
      dotOp.emitRemark()
          << "Dot op using MMA for compute capability " << computeCapability
          << " has been deprecated. It falls back to the FMA path.";
      return failure();
    }
    // TODO: Check data-types and SM compatibility
    auto retType = dotOp.getType();
    if (!retType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(retType.getEncoding()))
      return failure();

    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto oldRetType = cast<RankedTensorType>(dotOp.getType());

    // Enable F64 MMA only on targets with high performance F64 tensor cores.
    // Otherwise, fallback to F64 FMA for better performance.
    if ((oldAType.getElementType().isF64() ||
         oldBType.getElementType().isF64() ||
         oldRetType.getElementType().isF64()) &&
        !(computeCapability == 80 || computeCapability == 90 ||
          (computeCapability >= 100 && computeCapability < 120))) {
      return failure();
    }

    auto mmaVersion = getMMAVersionSafe(computeCapability, dotOp);
    auto mmaResult =
        createMMAEncodingForDot(dotOp, rewriter, computeCapability, mmaVersion);
    if (!(mmaResult.versionMajor >= 1 && mmaResult.versionMajor <= 3))
      return failure();

    Operation *newDot = nullptr;
    bool aFromLoad = comesFromLoadOrBlockArg(a);
    bool bFromLoad = comesFromLoadOrBlockArg(b);

    if (mmaResult.versionMajor == 3) {
      auto eltType = cast<RankedTensorType>(a.getType()).getElementType();
      bool allowTranspose = eltType.isF16() || eltType.isBF16();
      if (!aFromLoad) {
        int bitwidth = getElementTypeOrSelf(a).getIntOrFloatBitWidth();
        a = convertDotOperandForMMA(a, 0, bitwidth, mmaResult.newRetType,
                                    rewriter);
      } else {
        a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose,
                                      /*isMMAv5Fp4Padded=*/false,
                                      /*forceTranspose=*/false, dotOp);
      }
      b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose,
                                    /*isMMAv5Fp4Padded=*/false,
                                    /*forceTranspose=*/false, dotOp);

      newDot = triton::nvidia_gpu::WarpGroupDotOp::create(
          rewriter, dotOp.getLoc(), mmaResult.newRetType, a, b,
          mmaResult.newAcc, nullptr, dotOp.getInputPrecision(),
          dotOp.getMaxNumImpreciseAcc(), false);
      // Propagate discardable attributes (e.g. tt.autows) from the original
      // dot.
      for (auto attr : dotOp->getDiscardableAttrs())
        newDot->setDiscardableAttr(attr.getName(), attr.getValue());
    } else {
      int minBitwidth =
          std::min(computeOrigBitWidth(a), computeOrigBitWidth(b));
      // Let K = getShapePerCTA(oldAType).back() and computeBitwidth be the
      // dot operand bitwidth. Since kWidth = max(32 / minBitwidth, 1),
      // MMAv2's four K lanes require 4 * max(32 / minBitwidth, 1) <= K.
      // We have:
      //   (a) K >= 4;
      //   (b) minBitwidth >= 128 / K.
      //
      // Native operand packing is preserved by the invariant:
      //   (c) minBitwidth <= computeBitwidth.
      // Raising it for (b) could violate (c) only if 128 / K > computeBitwidth,
      // hence K < 128 / computeBitwidth. This contradicts the CUDA frontend's
      // requirement K >= 256 / computeBitwidth, which also implies (a).
      minBitwidth = std::max<int64_t>(minBitwidth,
                                      4 * 32 / getShapePerCTA(oldAType).back());
      assert(minBitwidth <= oldAType.getElementTypeBitWidth() &&
             "minBitwidth must not exceed the dot operand bitwidth");
      a = convertDotOperandForMMA(a, 0, minBitwidth, mmaResult.newRetType,
                                  rewriter);
      b = convertDotOperandForMMA(b, 1, minBitwidth, mmaResult.newRetType,
                                  rewriter);
      newDot = DotOp::create(rewriter, dotOp.getLoc(), mmaResult.newRetType, a,
                             b, mmaResult.newAcc, dotOp.getInputPrecision(),
                             dotOp.getMaxNumImpreciseAcc());
    }

    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, dotOp.getType(),
                                                 newDot->getResult(0));
    return success();
  }
};

static bool canUseTwoCTAs(triton::DotOp dotOp) {
  RankedTensorType retType = dotOp.getType();
  auto retShapePerCTA = getShapePerCTA(retType);
  // TODO: we could support 2 CTAs matmul with numCTAs > 2.
  SmallVector<unsigned> splitNum = getCTASplitNum(retType.getEncoding());
  if (splitNum.size() != 2 || splitNum[0] != 2 || splitNum[1] != 1)
    return false;
  int m = retShapePerCTA[0];
  int n = retShapePerCTA[1];
  // minimum size supported by 2CTAs mmav5.
  if (m < 64 || n < 32)
    return false;
  Value b = dotOp.getB();
  // Skip convert layouts.
  while (auto cvtOp = b.getDefiningOp<ConvertLayoutOp>())
    b = cvtOp.getSrc();
  return llvm::isa_and_nonnull<triton::LoadOp,
                               triton::DescriptorLoadLikeOpInterface>(
      b.getDefiningOp());
}

static DistributedEncodingTrait
replaceCGALayout(DistributedEncodingTrait layout,
                 const triton::gpu::CGAEncodingAttr &newCGALayout) {
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(layout)) {
    return BlockedEncodingAttr::get(
        layout.getContext(), blockedLayout.getSizePerThread(),
        blockedLayout.getThreadsPerWarp(), blockedLayout.getWarpsPerCTA(),
        blockedLayout.getOrder(), newCGALayout);
  } else if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    return SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        replaceCGALayout(sliceLayout.getParent(), newCGALayout));
  } else {
    llvm::report_fatal_error("not implemented");
    return layout;
  }
}

static Value splitBOperand(Value b, mlir::PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  MLIRContext *ctx = b.getContext();
  while (auto cvtOp = b.getDefiningOp<ConvertLayoutOp>())
    b = cvtOp.getSrc();
  auto loadOp = b.getDefiningOp();
  assert((isa<triton::LoadOp, triton::DescriptorLoadLikeOpInterface>(loadOp)) &&
         "expected LoadOp");
  RankedTensorType bType = cast<RankedTensorType>(b.getType());
  auto currentLayout = cast<DistributedEncodingTrait>(bType.getEncoding());
  auto kBlock = StringAttr::get(ctx, "block");
  auto dims = standardOutDimNames(ctx, 2);
  auto newCGALayout =
      CGAEncodingAttr::get(ctx, LinearLayout({{kBlock, {{0, 1}}}}, dims));
  Attribute newLayout = replaceCGALayout(currentLayout, newCGALayout);
  rewriter.setInsertionPoint(loadOp);
  for (OpOperand &operand : loadOp->getOpOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!tensorType)
      continue;
    Value newOperand = ConvertLayoutOp::create(
        rewriter, operand.get().getLoc(),
        tensorType.cloneWithEncoding(newLayout), operand.get());
    loadOp->setOperand(operand.getOperandNumber(), newOperand);
  }
  loadOp->getResult(0).setType(bType.cloneWithEncoding(newLayout));
  Value newB = loadOp->getResult(0);
  rewriter.setInsertionPointAfter(loadOp);
  auto cvt = ConvertLayoutOp::create(rewriter, b.getLoc(), bType, newB);
  rewriter.replaceAllUsesExcept(newB, cvt.getResult(), cvt);
  return newB;
}

class BlockedToMMAv5 : public mlir::OpRewritePattern<DotOp> {
  int computeCapability;

public:
  BlockedToMMAv5(mlir::MLIRContext *context, int computeCapability, int benefit)
      : OpRewritePattern<DotOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = lookupNumWarps(dotOp);
    auto CGALayout = getCGALayout(oldRetType.getEncoding());

    int versionMajor = getMMAVersionSafe(computeCapability, dotOp);
    if (versionMajor != 5)
      return failure();
    Location loc = dotOp.getLoc();
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    if (std::min(computeOrigBitWidth(a), computeOrigBitWidth(b)) >= 32 &&
        dotOp.getInputPrecision() != InputPrecision::TF32)
      return failure();
    auto oldAType = dotOp.getA().getType();
    auto oldBType = dotOp.getB().getType();
    bool useTwoCTAs;
    auto tmemCTAMode = triton::nvidia_gpu::TensorMemoryCTAMode::DEFAULT;
    if (dotOp.getTwoCtas()) {
      auto mod = dotOp->getParentOfType<ModuleOp>();
      auto clusterDims = triton::gpu::TritonGPUDialect::getClusterDims(mod);
      if (clusterDims[0] < 2) {
        dotOp.emitWarning()
            << "two_ctas=True requires ctas_per_cga=(2,1,1) or larger; "
               "cluster-dim-x is "
            << clusterDims[0] << ". Falling back to 1-CTA MMA.";
        useTwoCTAs = false;
      } else if (oldRetType.getShape()[0] < 64) {
        dotOp.emitWarning()
            << "two_ctas=True with BLOCK_M < 64 is not supported. "
               "Falling back to 1-CTA MMA.";
        useTwoCTAs = false;
      } else {
        useTwoCTAs = true;
        if (oldRetType.getShape()[0] == 64)
          tmemCTAMode = triton::nvidia_gpu::TensorMemoryCTAMode::TwoCTA_RHS;
      }
    } else {
      // NYI: PTX 13+ requires all tcgen instructions in a kernel to have a
      // consistent CTA mode, disabling compiler-driven 2CTA mode for now.
      // To re-enable, change the line below to:
      //   useTwoCTAs = canUseTwoCTAs(dotOp);
      useTwoCTAs = false;
      if (useTwoCTAs) {
        b = splitBOperand(b, rewriter);
      }
    }
    // TF32 transpose is only supported with 128 swizzle mode with 32B
    // atomicity. As we currently don't support this layout we disallow
    // transpose for TF32 inputs.
    bool allowTranspose = !dotOp.getA().getType().getElementType().isF32();
    a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose);
    b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose);
    MLIRContext *context = dotOp->getContext();
    auto instrShape = mmaVersionToInstrShape(
        versionMajor, retShapePerCTA, oldAType.getElementType(), numWarps);
    auto bitwidth = oldRetType.getElementType().getIntOrFloatBitWidth();
    unsigned colStride = 32 / bitwidth;
    // An NPOT N accumulates in TMEM blocks of its pow2 alloc shape; tcgen05
    // still writes only N columns.
    int64_t instrN = instrShape[1];
    Attribute accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
        context, instrShape[0], getNpotPow2AllocShape(instrN)[0], colStride,
        CGALayout, useTwoCTAs, tmemCTAMode);
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    MemDescType accMemDescType =
        MemDescType::get(oldRetType.getShape(), oldRetType.getElementType(),
                         accEncoding, tensorMemorySpace, /*mutableMemory=*/true,
                         getNpotPow2AllocShape(oldRetType.getShape()));
    auto newDistributedEncoding =
        nvidia_gpu::getDefaultLayoutForTmemLdSt(accMemDescType, numWarps);
    auto newAccType = oldRetType.cloneWithEncoding(newDistributedEncoding);
    Value cvtAcc =
        ConvertLayoutOp::create(rewriter, loc, newAccType, dotOp.getOperand(2));
    auto tokType = rewriter.getType<AsyncTokenType>();
    auto acc = triton::nvidia_gpu::TMEMAllocOp::create(
        rewriter, loc, accMemDescType, tokType, cvtAcc);
    auto vTrue = arith::ConstantIntOp::create(rewriter, dotOp.getLoc(), 1, 1);
    auto mma = triton::nvidia_gpu::TCGen5MMAOp::create(
        rewriter, loc, tokType, a, b, acc, acc.getToken(), /*useD=*/vTrue,
        /*pred=*/vTrue);
    mma.setTwoCtas(useTwoCTAs);
    // Propagate discardable attributes (e.g. tt.autows) from the original dot.
    for (auto attr : dotOp->getDiscardableAttrs())
      mma->setDiscardableAttr(attr.getName(), attr.getValue());

    auto ld = triton::nvidia_gpu::TMEMLoadOp::create(
        rewriter, loc, newAccType, tokType, acc, /*dep=*/mma.getToken());
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType, ld);
    return success();
  }
};

class ScaledBlockedToMMA : public mlir::OpRewritePattern<triton::DotScaledOp> {
  int computeCapability;

public:
  ScaledBlockedToMMA(mlir::MLIRContext *context, int computeCapability,
                     int benefit)
      : mlir::OpRewritePattern<triton::DotScaledOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotScaledOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability / 10 != 12)
      return failure();

    auto numCTAs = lookupNumCTAs(rewriter);
    if (numCTAs != 1) {
      return failure();
    }
    // Skip if any scale is missing. This pattern requires both scales.
    if (!dotOp.getAScale() || !dotOp.getBScale())
      return failure();

    auto aScaleType = dotOp.getAScale().getType();
    auto bScaleType = dotOp.getBScale().getType();

    if (mlir::isa<LinearEncodingAttr>(aScaleType.getEncoding()) ||
        mlir::isa<LinearEncodingAttr>(bScaleType.getEncoding())) {
      return failure();
    }
    auto aElemType = dotOp.getAElemType();
    auto bElemType = dotOp.getBElemType();
    auto isFP8 = [&](ScaleDotElemType elemType) -> bool {
      return elemType == ScaleDotElemType::E4M3 ||
             elemType == ScaleDotElemType::E5M2;
    };
    auto isFP4 = [&](ScaleDotElemType elemType) -> bool {
      return elemType == ScaleDotElemType::E2M1;
    };

    bool isFP4xFP4 = isFP4(aElemType) && isFP4(bElemType);
    // TODO: Enable mixed-precision mxfp for sm120
    if (!((isFP8(aElemType) && isFP8(bElemType)) || isFP4xFP4)) {
      return rewriter.notifyMatchFailure(
          dotOp, "only FP8xFP8 and FP4xFP4 are supported on sm120");
    }
    if (isFP4xFP4 && (!dotOp.getLhsKPack() || !dotOp.getRhsKPack())) {
      return rewriter.notifyMatchFailure(
          dotOp, "SM120 native FP4xFP4 requires K-packed operands");
    }

    auto scaleElemType = dotOp.getAScale().getType().getElementType();
    if (scaleElemType != dotOp.getBScale().getType().getElementType()) {
      return failure();
    }

    // UE5M3 (int8 block-16) scales have no native sm120 lowering: the MMAv2
    // backend keys the scale kind off the element type alone and would
    // encode them as UE8M0/2X, reading the bytes wrong. Fail so the op falls
    // back to the UE5M3-aware decompose path.
    if (scaleElemType.isInteger(8) && dotOp.deduceScaleFactor() == 16) {
      return rewriter.notifyMatchFailure(
          dotOp, "sm120 native scaled dot does not support UE5M3 (int8 "
                 "block-16) scales");
    }

    // Common MMA encoding creation
    auto mmaResult =
        createMMAEncodingForDot(dotOp, rewriter, computeCapability, 2);

    // Operand processing
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    Operation *newDot = nullptr;

    // ScaledBlockedToMMA logic
    int bitwidthA = oldAType.getElementType().getIntOrFloatBitWidth();
    int bitwidthB = oldBType.getElementType().getIntOrFloatBitWidth();
    int minBitwidth = std::min(bitwidthA, bitwidthB);

    Value newA = convertDotOperandForMMA(a, 0, minBitwidth,
                                         mmaResult.newRetType, rewriter);
    Value newB = convertDotOperandForMMA(b, 1, minBitwidth,
                                         mmaResult.newRetType, rewriter);
    const auto mmaWarps = mmaResult.mmaEnc.getWarpsPerCTA(); // [wM, wN]
    // Convert scales to Linear layout
    auto convertScale = [&](Value scale, int opIdx) -> Value {
      auto ty = cast<RankedTensorType>(scale.getType());
      SmallVector<int64_t> shape = llvm::to_vector(ty.getShape());
      MLIRContext *ctx = ty.getContext();
      auto blocked = cast<triton::gpu::BlockedEncodingAttr>(ty.getEncoding());

      auto ll = triton::gpu::getSM120DotScaledScaleLayout(
          ctx, shape, opIdx, mmaWarps, blocked.getCGALayout());
      auto newEnc = triton::gpu::LinearEncodingAttr::get(ctx, std::move(ll));
      auto newTy = RankedTensorType::get(shape, ty.getElementType(), newEnc);
      return ConvertLayoutOp::create(rewriter, scale.getLoc(), newTy, scale);
    };
    Value aScale = convertScale(dotOp.getAScale(), /*opIdx=*/0);
    Value bScale = convertScale(dotOp.getBScale(), /*opIdx=*/1);

    newDot = triton::DotScaledOp::create(
        rewriter, dotOp.getLoc(), mmaResult.newRetType, newA, newB,
        mmaResult.newAcc, aScale, bScale, dotOp.getAElemType(),
        dotOp.getBElemType(), dotOp.getFastMath(), dotOp.getLhsKPack(),
        dotOp.getRhsKPack());
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, dotOp.getType(),
                                                 newDot->getResult(0));
    return success();
  }
};

// Whether ScaledBlockedToMMAv5 lowers `dotOp` to tcgen05 rather than leaving
// it to the decompose pattern.
static bool isTcgen05ScaledDot(DotScaledOp dotOp, int computeCapability,
                               int numWarps) {
  if (dotOp.getAScale() == nullptr || dotOp.getBScale() == nullptr)
    return false;
  auto retShapePerCTA = getShapePerCTA(dotOp.getType());
  if (computeCapability < 100 || computeCapability >= 120)
    return false;
  if (numWarps != 4 && numWarps != 8)
    return false;
  if (retShapePerCTA[0] < 128 || retShapePerCTA[1] < 16)
    return false;
  // fp6 (groups of 16 values per 12 bytes) cannot tile the tcgen05 K
  // dimension today; route it to the decompose (upcast) path instead
  // of native MMAv5.
  if (isMXFP6ScaleType(dotOp.getAElemType()) ||
      isMXFP6ScaleType(dotOp.getBElemType()))
    return false;
  bool isAFP4 = dotOp.getAElemType() == ScaleDotElemType::E2M1;
  bool isFp4MMA = isAFP4 && dotOp.getBElemType() == ScaleDotElemType::E2M1;
  // An MN-packed fp4 A is packed along M, so its last dim is already K.
  auto blockK = dotOp.getA().getType().getShape().back() *
                (isAFP4 && dotOp.getLhsKPack() ? 2 : 1);
  bool hasMNMajorFp4Operand =
      isFp4MMA && (!dotOp.getLhsKPack() || !dotOp.getRhsKPack());
  auto aScaleType = dotOp.getAScale().getType();
  auto bScaleType = dotOp.getBScale().getType();
  auto aScaleElemType = aScaleType.getElementType();
  auto bScaleElemType = bScaleType.getElementType();
  auto isBlock16Scale = [blockK](RankedTensorType scaleType) {
    return scaleType.getShape().back() * 16 == blockK;
  };
  bool hasUE4M3Scale = isa<Float8E4M3FNType>(aScaleElemType) ||
                       isa<Float8E4M3FNType>(bScaleElemType);
  bool hasUE5M3Scale =
      (aScaleElemType.isInteger(8) && isBlock16Scale(aScaleType)) ||
      (bScaleElemType.isInteger(8) && isBlock16Scale(bScaleType));
  // UE5M3 scales are native only on sm107; sm100/sm103 trap, so decompose.
  if (hasUE5M3Scale &&
      !nvidia_gpu::TargetFeatures(computeCapability).supportsUE5M3Tcgen05MMA())
    return false;
  // mxf4nvf4 supports UE4M3/UE5M3 scales but not MN-major operands, while
  // the mxf8f6f4 fallback for MN-major FP4 only supports E8M0 scales.
  if (hasMNMajorFp4Operand && (hasUE4M3Scale || hasUE5M3Scale))
    return false;
  return true;
}

class ScaledBlockedToMMAv5
    : public mlir::OpRewritePattern<triton::DotScaledOp> {
  int computeCapability;

public:
  ScaledBlockedToMMAv5(mlir::MLIRContext *context, int computeCapability,
                       int benefit)
      : mlir::OpRewritePattern<triton::DotScaledOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotScaledOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    int numWarps = lookupNumWarps(dotOp);
    if (!isTcgen05ScaledDot(dotOp, computeCapability, numWarps))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    auto CGALayout = getCGALayout(oldRetType.getEncoding());
    Location loc = dotOp.getLoc();
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();

    bool IsAMixedPrecFp4 = false;
    bool IsBMixedPrecFp4 = false;
    bool isAFP4 = dotOp.getAElemType() == ScaleDotElemType::E2M1;
    bool isBFP4 = dotOp.getBElemType() == ScaleDotElemType::E2M1;

    if (dotOp.getAElemType() != dotOp.getBElemType()) {
      if (isAFP4)
        IsAMixedPrecFp4 = true;
      else if (isBFP4)
        IsBMixedPrecFp4 = true;
    }

    bool isFp4MMA = isAFP4 && isBFP4;
    bool requiresFp4Padding =
        nvidia_gpu::TargetFeatures(computeCapability).requiresFp4Padding();

    // On Blackwell, if we use mixed-precision MMA we need to pad the fp4
    // operand
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-smem
    // On Rubin, the fp4 operand must be packed in SMEM if MMA_K = 64 is used.
    // However, MMA_K = 64 with MN-major fp4 is not supported. In this case,
    // the fp4 operand must be padded and MMA_K = 32 must be used.
    // MMA_K = 64 is also not supported when BLOCK_M = 64.
    // An MN-packed fp4 A is packed along M, so its last dim is already K.
    auto blockK = dotOp.getA().getType().getShape().back() *
                  (isAFP4 && dotOp.getLhsKPack() ? 2 : 1);
    auto blockM = dotOp.getA().getType().getShape()[0];
    bool hasMNMajorFp4Operand =
        isFp4MMA && (!dotOp.getLhsKPack() || !dotOp.getRhsKPack());
    // mxf4 does not support MN-major operands, so mxfp4 x mxfp4 falls back to
    // mxf8f6f4 if either operand is not K-packed. The mxf8f6f4 shared-memory
    // packing format requires padding for every fp4 operand, even if the
    // operand is K packed.
    bool isMMAv5Fp4PaddedLhs =
        hasMNMajorFp4Operand ||
        (IsAMixedPrecFp4 && (requiresFp4Padding || blockM == 64 ||
                             blockK == 32 || !dotOp.getLhsKPack()));
    bool isMMAv5Fp4PaddedRhs =
        hasMNMajorFp4Operand ||
        (IsBMixedPrecFp4 &&
         (requiresFp4Padding || blockK == 32 || !dotOp.getRhsKPack()));

    // For mixed-precision fp4 operands, set allowTranspose = false, to force
    // the packed axis, K, to be contiguous in SMEM
    a = getSharedMemoryMMAOperand(a, rewriter, 0,
                                  /*allowTranspose=*/!isAFP4,
                                  /*isMMAv5Fp4Padded=*/isMMAv5Fp4PaddedLhs,
                                  /*forceTranspose=*/!dotOp.getLhsKPack(),
                                  dotOp);
    b = getSharedMemoryMMAOperand(b, rewriter, 1,
                                  /*allowTranspose=*/!isBFP4,
                                  /*isMMAv5Fp4Padded=*/isMMAv5Fp4PaddedRhs,
                                  /*forceTranspose=*/!dotOp.getRhsKPack(),
                                  dotOp);

    MLIRContext *context = dotOp->getContext();
    unsigned m = 128;
    unsigned n = retShapePerCTA[1] >= 256 ? 256 : retShapePerCTA[1];

    // Honor tt.dot_scaled's two_ctas flag. Needs a cluster of at least 2 CTAs;
    // otherwise warn and fall back to 1-CTA so the kernel still compiles.
    // BLOCK_M >= 128 is already guaranteed by the retShapePerCTA bail above.
    bool useTwoCTAs = false;
    if (dotOp.getTwoCtas()) {
      auto mod = dotOp->getParentOfType<ModuleOp>();
      auto clusterDims = triton::gpu::TritonGPUDialect::getClusterDims(mod);
      if (clusterDims[0] < 2) {
        dotOp.emitWarning()
            << "two_ctas=True requires ctas_per_cga=(2,1,1) or larger; "
               "cluster-dim-x is "
            << clusterDims[0] << ". Falling back to 1-CTA scaled MMA.";
      } else {
        useTwoCTAs = true;
      }
    }

    auto bitwidth = oldRetType.getElementType().getIntOrFloatBitWidth();
    unsigned colStride = 32 / bitwidth;
    Attribute accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
        context, m, n, colStride, CGALayout, useTwoCTAs,
        triton::nvidia_gpu::TensorMemoryCTAMode::DEFAULT, /*fp4Padded=*/false);
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    MemDescType accMemDescType =
        MemDescType::get(oldRetType.getShape(), oldRetType.getElementType(),
                         accEncoding, tensorMemorySpace,
                         /*mutableMemory=*/true);
    auto newDistributedEncoding =
        nvidia_gpu::getDefaultLayoutForTmemLdSt(accMemDescType, numWarps);
    auto newAccType = oldRetType.cloneWithEncoding(newDistributedEncoding);
    Value cvtAcc =
        ConvertLayoutOp::create(rewriter, loc, newAccType, dotOp.getOperand(2));
    auto tokType = rewriter.getType<AsyncTokenType>();
    auto acc = triton::nvidia_gpu::TMEMAllocOp::create(
        rewriter, loc, accMemDescType, tokType, cvtAcc);

    RankedTensorType oldScaleAType = dotOp.getAScale().getType();
    RankedTensorType oldScaleBType = dotOp.getBScale().getType();

    auto aScaleBlockRepOrder =
        triton::nvidia_gpu::getTensorMemoryScalesBlockRepOrder(
            dotOp, /*isA=*/true, dotOp.getAElemType(), dotOp.getBElemType(),
            oldScaleAType.getElementType(), oldScaleBType.getElementType());
    auto bScaleBlockRepOrder =
        triton::nvidia_gpu::getTensorMemoryScalesBlockRepOrder(
            dotOp, /*isA=*/false, dotOp.getAElemType(), dotOp.getBElemType(),
            oldScaleAType.getElementType(), oldScaleBType.getElementType());
    Attribute scaleAEncoding =
        triton::nvidia_gpu::TensorMemoryScalesEncodingAttr::get(
            context, CGALayout, aScaleBlockRepOrder,
            triton::nvidia_gpu::TensorMemoryCTAMode::DEFAULT);
    Attribute scaleBEncoding =
        triton::nvidia_gpu::TensorMemoryScalesEncodingAttr::get(
            context, CGALayout, bScaleBlockRepOrder,
            triton::nvidia_gpu::TensorMemoryCTAMode::DEFAULT);
    Value scaleA =
        tryCreateTmemCopyCompatibleScaleOperand(dotOp.getAScale(), rewriter);
    if (!scaleA) {
      scaleA =
          createTmemScaleOperand(dotOp.getAScale(), scaleAEncoding,
                                 tensorMemorySpace, numWarps, loc, rewriter);
    }
    Value scaleB =
        tryCreateTmemCopyCompatibleScaleOperand(dotOp.getBScale(), rewriter);
    if (!scaleB) {
      scaleB =
          createTmemScaleOperand(dotOp.getBScale(), scaleBEncoding,
                                 tensorMemorySpace, numWarps, loc, rewriter);
    }

    auto vTrue = arith::ConstantIntOp::create(rewriter, dotOp.getLoc(), 1, 1);
    auto mmaOp = triton::nvidia_gpu::TCGen5MMAScaledOp::create(
        rewriter, loc, tokType, a, b, acc.getResult(), acc.getToken(), scaleA,
        scaleB, dotOp.getAElemType(), dotOp.getBElemType(),
        /*useD=*/vTrue, /*pred=*/vTrue, /*two_ctas=*/useTwoCTAs);
    // Propagate discardable attributes (e.g. tt.autows) from the original dot.
    for (auto attr : dotOp->getDiscardableAttrs())
      mmaOp->setDiscardableAttr(attr.getName(), attr.getValue());

    auto ld = triton::nvidia_gpu::TMEMLoadOp::create(
        rewriter, loc, newAccType, tokType, acc, mmaOp.getToken());
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType, ld);
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  Type operandElType =
      cast<RankedTensorType>(operand.getType()).getElementType();
  if (type::isFloat8(operandElType)) {
    return FpToFpOp::create(builder, loc, tensorPromotedType, operand);
  }
  return arith::ExtFOp::create(builder, loc, tensorPromotedType, operand);
}

static bool mmav2SupportsFp8Operands(int computeCapability) {
  // promote operands for sm < 89 since fp8 mma is not natively supported
  // although PTX instructions for mma v2 w/ fp8 operands exist for sm90 and
  // sm100, they are emulated as fp16 upcasts + fp16 HMMA in SASS. sm120 has
  // hardware support for fp8 operands w/ mmav2.
  return computeCapability == 89 || computeCapability / 10 == 12;
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod, int computeCapability) {
  mod.walk([=](DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    NvidiaMmaEncodingAttr mmaLayout =
        dyn_cast<NvidiaMmaEncodingAttr>(D.getType().getEncoding());
    if (mmaLayout) {
      bool isNativeFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType>(AElType);
      // promote to f16 unless there's hardware support for fp8 operands
      if (!isNativeFP8 ||
          (isNativeFP8 && (mmav2SupportsFp8Operands(computeCapability) ||
                           mmaLayout.isHopper())))
        return;
      promoteType = builder.getF16Type();
    } else {
      // FMA case.
      Type AElType = dotOp.getA().getType().getElementType();
      Type DElType = D.getType().getElementType();
      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

// Transpose scaled_dot ops that have a scale on lhs.
static void transposeDotOp(DotScaledOp dotOp) {
  // two_ctas means the CTA pair splits M, so it cannot be carried across a
  // transpose that swaps M and N. Drop it, but say so: every other unsupported
  // 2-CTA config warns before falling back, and silently ignoring the flag
  // would leave the user wondering why no cta_group::2 was emitted.
  if (dotOp.getTwoCtas())
    dotOp.emitWarning() << "two_ctas=True is not supported when only the rhs "
                           "is scaled, because the dot is transposed here. "
                           "Falling back to 1-CTA scaled MMA.";
  OpBuilder builder(dotOp);
  Value lhs = dotOp.getA();
  std::array<int, 2> transOrder = {1, 0};
  Value lhsTransposed = TransOp::create(builder, lhs.getLoc(), lhs, transOrder);
  Value rhs = dotOp.getB();
  Value rhsTransposed = TransOp::create(builder, rhs.getLoc(), rhs, transOrder);
  Value c = dotOp.getC();
  Value cTransposed = TransOp::create(builder, c.getLoc(), c, transOrder);
  Value result = DotScaledOp::create(
      builder, dotOp.getLoc(), cTransposed.getType(), rhsTransposed,
      lhsTransposed, cTransposed, dotOp.getBScale(), dotOp.getAScale(),
      dotOp.getBElemType(), dotOp.getAElemType(), dotOp.getFastMath(),
      /*lhs_k_pack=*/dotOp.getRhsKPack(),
      /*rhs_k_pack=*/dotOp.getLhsKPack());
  Operation *transposedResult =
      TransOp::create(builder, result.getLoc(), result, transOrder);
  dotOp.replaceAllUsesWith(transposedResult);
  dotOp.erase();
}

static void transposeDots(ModuleOp m) {
  // TODO: extend to regular dot when it is profitable. For instance when we may
  // want to use rhs from register for mmav3.
  SmallVector<DotScaledOp> toTranspose;
  m.walk([&](DotScaledOp dotOp) -> void {
    if (dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr)
      toTranspose.push_back(dotOp);
  });
  for (DotScaledOp dotOp : toTranspose) {
    transposeDotOp(dotOp);
  }
}

#define GEN_PASS_DEF_TRITONGPUACCELERATEMATMUL
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUAccelerateMatmulPass
    : public impl::TritonGPUAccelerateMatmulBase<
          TritonGPUAccelerateMatmulPass> {
public:
  using impl::TritonGPUAccelerateMatmulBase<
      TritonGPUAccelerateMatmulPass>::TritonGPUAccelerateMatmulBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    auto computeCapability = getNVIDIAComputeCapability(m);
    // We could do this generically if we manage to improve the heuristics
    // reverted in these two PRs https://github.com/triton-lang/triton/pull/5834
    // https://github.com/triton-lang/triton/pull/5837
    transposeDots(m);

    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;
    constexpr int benefitMMAv5 = 10;
    constexpr int benefitSM120 = 10;

    patterns.add<BlockedToMMA>(context, computeCapability, benefitDefault);
    patterns.add<ScaledBlockedToMMA>(context, computeCapability, benefitSM120);
    populateDecomposeScaledBlockedPatterns(patterns, benefitDefault,
                                           /*allowUE5M3Scales=*/true);
    patterns.add<BlockedToMMAv5, ScaledBlockedToMMAv5>(
        context, computeCapability, benefitMMAv5);

    // A pattern that rejects an op emits a named error and fails to match,
    // which still lets the greedy driver converge. Fail the pass on any
    // error so the rejected op never reaches downstream passes.
    bool emittedError = false;
    ScopedDiagnosticHandler errorHandler(context, [&](Diagnostic &diag) {
      if (diag.getSeverity() == DiagnosticSeverity::Error)
        emittedError = true;
      return failure();
    });
    if (applyPatternsGreedily(m, std::move(patterns)).failed() || emittedError)
      return signalPassFailure();
    // Now that we have picked the mma type, decompose dot that are not natively
    // supported.
    decomposeMixedModeDotOp(m, computeCapability);
  }
};

namespace {
// Whether the lowering's descriptor match finds a core-matrix form for the B
// memdesc the MMA patterns alloc, with dim `fast` contiguous.
bool hasCoreMatrixForm(RankedTensorType bTy, unsigned fast) {
  MLIRContext *ctx = bTy.getContext();
  auto enc = NVMMASharedEncodingAttr::get(
      ctx, bTy.getShape(), {fast, 1 - fast},
      CGAEncodingAttr::get1CTALayout(ctx, bTy.getRank()), bTy.getElementType(),
      /*fp4Padded=*/false);
  auto memTy = MemDescType::get(bTy.getShape(), bTy.getElementType(), enc,
                                SharedMemorySpaceAttr::get(ctx));
  return nvidia_gpu::getMmaOperandSmemAttrs(memTy).has_value();
}

// Why BlockedToMMA can't issue `dot` exactly on wgmma, or std::nullopt. Its
// warps must sit on each dim's power-of-two part, as the mixed-radix layout
// puts them; then exact issues no more atoms per warp than padded.
std::optional<std::string> getMmaUnsupportedReason(DotOp dot, int numWarps) {
  ArrayRef<int64_t> shape = dot.getType().getShape();
  auto instrShape = mmaVersionToInstrShape(
      3, shape, dot.getA().getType().getElementType(), numWarps);
  int64_t k = dot.getA().getType().getShape().back();
  if (!isMmaV3NpotKLegal(k, instrShape[2]))
    return llvm::formatv("NPOT dot is not yet supported on wgmma with K {0}, "
                         "whose power-of-two part {1} is less than k{2}",
                         k, LinearLayout::twoPartOf(static_cast<int32_t>(k)),
                         instrShape[2])
        .str();
  SmallVector<unsigned> warps =
      to_vector(warpsPerTileV3(dot, shape, numWarps, instrShape));
  if (!llvm::isPowerOf2_32(instrShape[1]) &&
      !isMmaV3NpotNLegal(warps, instrShape))
    return llvm::formatv(
               "NPOT wgmma N {0} is not yet supported with {1} warpgroups on N",
               instrShape[1], warps[1])
        .str();
  for (auto [dim, size] : llvm::enumerate(shape)) {
    int64_t twoPart = LinearLayout::twoPartOf(static_cast<int32_t>(size));
    if (twoPart % (warps[dim] * LinearLayout::twoPartOf(instrShape[dim])) != 0)
      return llvm::formatv("NPOT dot is not yet supported with {0} warps of "
                           "{1} on dim {2}, whose power-of-two part is {3}",
                           warps[dim], instrShape[dim], dim, twoPart)
          .str();
  }
  // B's order isn't known yet: take the N-major order of a row-major load
  // unless the dtype forces K-major.
  Type elemTy = dot.getA().getType().getElementType();
  if (!hasCoreMatrixForm(dot.getB().getType(),
                         elemTy.isF16() || elemTy.isBF16() ? 1 : 0))
    return llvm::formatv("NPOT wgmma N {0} is not yet supported: B's shared "
                         "layout has no core-matrix form",
                         shape[1])
        .str();
  return std::nullopt;
}

// Asks the TCGen5MMAOp verifier's rule about the tiles BlockedToMMAv5 issues,
// then the lowering's descriptor match about the B memdesc it allocs. B's order
// isn't known yet: take the stricter MN-major form, which f32 can't take.
std::optional<std::string>
getTcgen05UnsupportedReason(DotOp dot, nvidia_gpu::Tcgen05Kind kind,
                            int numWarps) {
  Type elemTy = dot.getA().getType().getElementType();
  ArrayRef<int64_t> shape = dot.getType().getShape();
  auto instrShape = mmaVersionToInstrShape(5, shape, elemTy, numWarps);
  if (auto reason = nvidia_gpu::getNpotMMAv5TileUnsupportedReason(
          kind, /*aInTmem=*/false, /*bMNMajor=*/true, dot.getTwoCtas(),
          instrShape[0], instrShape[1], shape,
          dot.getA().getType().getShape().back(),
          elemTy.getIntOrFloatBitWidth()))
    return reason;
  if (!hasCoreMatrixForm(dot.getB().getType(), elemTy.isF32() ? 0 : 1))
    return llvm::formatv("NPOT MMAv5 N {0} is not yet supported: B's shared "
                         "layout has no core-matrix form",
                         shape[1])
        .str();
  return std::nullopt;
}

struct NvidiaNpotTargetLegality : public NpotTargetLegality {
  using NpotTargetLegality::NpotTargetLegality;
  StringRef getTargetPrefix() const override { return "cuda"; }
  std::optional<std::string>
  getDotUnsupportedReason(Operation *op, const LinearLayout &, StringRef target,
                          int numWarps) const override {
    // Exact only where BlockedToMMA picks wgmma; f64 takes FMA on some
    // targets.
    auto dot = dyn_cast<DotOp>(op);
    int computeCapability = 0;
    target.split(':').second.getAsInteger(10, computeCapability);
    if (dot && computeCapability >= 80 &&
        !dot.getA().getType().getElementType().isF64()) {
      SmallVector<int> versions =
          getMMAVersionsSupported(computeCapability, dot);
      if (versions == SmallVector<int>{3, 2} && supportMMA(dot, 3, numWarps))
        return getMmaUnsupportedReason(dot, numWarps);
      if (versions == SmallVector<int>{5, 2} && supportMMA(dot, 5, numWarps))
        if (auto kind = nvidia_gpu::getTcgen05Kind(
                dot.getA().getType().getElementType()))
          return getTcgen05UnsupportedReason(dot, *kind, numWarps);
    }
    return ("NPOT dot is not yet supported on " + target).str();
  }
  std::optional<std::string>
  getAsyncCopyUnsupportedReason(Operation *, const LinearLayout &,
                                StringRef) const override {
    return std::nullopt;
  }
};
} // namespace

void registerNpotTargetLegality(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *, nvidia_gpu::TritonNvidiaGPUDialect *dialect) {
        dialect->addInterfaces<NvidiaNpotTargetLegality>();
      });
}

} // namespace gpu
} // namespace triton
} // namespace mlir
