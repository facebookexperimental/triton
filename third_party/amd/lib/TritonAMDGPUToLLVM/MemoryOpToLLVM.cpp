#include "AsyncUtility.h"
#include "AtomicRMWOpsEmitter.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include <type_traits>

using mlir::triton::amdgpu::ISAFamily;
using ::mlir::triton::gpu::MemDescType;

namespace {

static LLVM::FenceOp createAMDGPUMemoryFence(OpBuilder &builder, Location loc,
                                             LLVM::AtomicOrdering ordering,
                                             StringRef synchronizeAddrSpace) {
  auto fence =
      LLVM::FenceOp::create(builder, loc, ordering, /*syncscope=*/"workgroup");
  if (!synchronizeAddrSpace.empty()) {
    Attribute mmra = builder.getAttr<LLVM::MMRATagAttr>("amdgpu-synchronize-as",
                                                        synchronizeAddrSpace);
    fence->setDiscardableAttr(LLVM::LLVMDialect::getMmraAttrName(), mmra);
  }
  return fence;
}

// Creates and returns the result Value of a single ds_read_tr* op for the
// given (isaFamily, logicalBitWidth).
static Value createDsReadTr(Operation *op, RewriterBase &rewriter, Location loc,
                            Value vecAddr, VectorType vTy, ISAFamily isaFamily,
                            unsigned logicalBitWidth) {
  // tr16 instructions return vectors of bf16/f16 while tr8 and tr4
  // instructions return vectors of i32. Generate the corresponding i32 vector
  // type.
  const auto physicalBitWidth =
      getIntOrFloatOrPtrBitWidth(vTy.getElementType());
  const auto numElemsI32 = (vTy.getNumElements() * physicalBitWidth / 32);
  const auto vTyI32 = VectorType::get(numElemsI32, i32_ty);

  // GFX1250 uses opaque LLVM intrinsic calls; their results cannot be cast to
  // AliasAnalysisOpInterface, so no no-alias scope is attached.
  auto callIntrinsic = [&](StringRef name, VectorType retTy) -> Value {
    return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, {retTy},
                                           {vecAddr})
        .getResult(0);
  };

  switch (isaFamily) {
  case ISAFamily::GFX1250:
    if (logicalBitWidth == 16)
      return callIntrinsic("llvm.amdgcn.ds.load.tr16.b128", vTy);
    if (logicalBitWidth == 8)
      return callIntrinsic("llvm.amdgcn.ds.load.tr8.b64", vTyI32);
    if (logicalBitWidth == 4)
      return callIntrinsic("llvm.amdgcn.ds.load.tr4.b64", vTyI32);
    return {};
  case ISAFamily::CDNA4: {
    Value dsReadTr;
    if (logicalBitWidth == 16)
      dsReadTr = ROCDL::ds_read_tr16_b64::create(rewriter, loc, vTy, vecAddr);
    else if (logicalBitWidth == 8)
      dsReadTr = ROCDL::ds_read_tr8_b64::create(rewriter, loc, vTyI32, vecAddr);
    else if (logicalBitWidth == 4)
      dsReadTr = ROCDL::ds_read_tr4_b64::create(rewriter, loc, vTyI32, vecAddr);
    else
      return {};
    AMD::addLocalLoadNoAliasScope(
        op, cast<LLVM::AliasAnalysisOpInterface>(dsReadTr.getDefiningOp()));
    return dsReadTr;
  }
  default:
    return {};
  }
}

// Emits a single ds_read_tr* operation at `vecAddr` and unpacks the loaded
// vector into individual element Values. Returns an empty vector if the ISA
// family does not support a ds_read_tr* instruction.
SmallVector<Value> emitDsReadTr(Operation *op, Location loc, Value vecAddr,
                                VectorType vTy, Type llvmElemTy,
                                unsigned logicalBitWidth,
                                ConversionPatternRewriter &rewriter,
                                const ::triton::AMD::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  const auto physicalBitWidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
  assert(physicalBitWidth == 16 || physicalBitWidth == 8);

  Value dsReadTr = createDsReadTr(op, rewriter, loc, vecAddr, vTy,
                                  targetInfo.getISAFamily(), logicalBitWidth);
  if (!dsReadTr)
    return {};

  Value vecVal = b.bitcast(dsReadTr, vTy);
  SmallVector<Value> loadedVals;
  for (int v = 0; v < vTy.getNumElements(); v++)
    loadedVals.push_back(b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
  return loadedVals;
}

LogicalResult lowerDsReadTr(
    Operation *op, ::triton::AMD::TargetInfo::LDSTransLoadParams ldsParams,
    Location loc, LinearLayout cvt, unsigned logicalBitWidth,
    SmallVector<Value> &vals, ArrayRef<Value> smemBases, Value affineOffset,
    uint64_t maskSpanAffineOffset,
    ArrayRef<std::pair<unsigned, unsigned>> paddingShifts, Type llvmElemTy,
    const std::shared_ptr<DistributedCoordinateGroups> &coordinateGroups,
    ConversionPatternRewriter &rewriter,
    const ::triton::AMD::TargetInfo &targetInfo) {

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();

  auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kOffset = S("offset");
  auto kBlock = S("block");
  auto kAddr = S("addr");
  auto kPartition = S("partition");
  auto smemPtrTy = ptr_ty(ctx, 3);
  auto bitWidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
  auto logicalMaskSpanAffineOffset =
      maskSpanAffineOffset * (bitWidth / logicalBitWidth);

  assert(!smemBases.empty() && "expected at least one smem base");
  LinearLayout cvtLayout = cvt;
  LinearLayout partitionLayout;
  Value basesVec;
  const bool isPartitioned = smemBases.size() > 1;

  if (isPartitioned) {
    assert(cvtLayout.hasOutDim(kPartition) &&
           cvtLayout.getOutDimSize(kPartition) ==
               static_cast<int32_t>(smemBases.size()) &&
           "smemBases size must match partition dimension size");
    auto inDimNames = llvm::to_vector(cvtLayout.getInDimNames());
    partitionLayout = cvtLayout.sublayout(inDimNames, {kPartition});
    SmallVector<StringAttr> outDims =
        llvm::to_vector(cvtLayout.getOutDimNames());
    llvm::erase(outDims, kPartition);
    cvtLayout = cvtLayout.sublayout(inDimNames, outDims);
    basesVec = LLVM::buildBasePtrVector(loc, rewriter, smemBases);
  }

  // A ds_read_trK_bN instruction takes one LDS base address from each of the
  // lanes in a warp, reads N / K contiguous K-bit elements from each base, and
  // distributes the elements along the lanes in a manner that can be described
  // by a bijective linear map
  //
  //                           F: R ⊕ L  ->  C ⊕ A,
  //
  // where R and L are the destination register and lane index spaces, and C and
  // A are the contiguous-offset and address-providing lane index spaces.
  //
  // The LinearLayout `fullTile` describes this map. Its `tile` factor maps the
  // first t := log2(N / K) bases of L to C. The bases mapped to A are an order-
  // preserving interleaving of the bases of R and the remaining bases of L:
  //
  //                   R[0:a], L[t:t + b], R[a:t], L[t + b:p],
  //
  // where p := log2(lanesPerWarp), and a and b are instruction-specific
  // parameters.
  auto tile = LinearLayout::identity1D(ldsParams.tileSize, kLane, kOffset);
  const unsigned numInstrRegBits = llvm::Log2_32(ldsParams.tileSize);
  const unsigned numAddrLaneBits =
      llvm::Log2_32(targetInfo.getWarpSize()) - numInstrRegBits;
  auto fullTile =
      tile *
      LinearLayout::identity1D(1 << ldsParams.leadingRegBases, kReg, kAddr) *
      LinearLayout::identity1D(1 << ldsParams.leadingLaneBases, kLane, kAddr) *
      LinearLayout::identity1D(
          1 << (numInstrRegBits - ldsParams.leadingRegBases), kReg, kAddr) *
      LinearLayout::identity1D(
          1 << (numAddrLaneBits - ldsParams.leadingLaneBases), kLane, kAddr) *
      LinearLayout::identity1D(1, kWarp, kAddr);

  if (cvtLayout.getInDimSize(kReg) < fullTile.getInDimSize(kReg)) {
    return failure();
  }

  auto maybeQuot = divideLeft(cvtLayout, tile);
  if (!maybeQuot.has_value()) {
    return failure();
  }

  // From here on we perform the lowering
  auto reps = zerosLike(tile) * maybeQuot.value();

  // Sanity check
  assert(fullTile.getInDimSize(kReg) * logicalBitWidth ==
         ldsParams.instBitWidth);

  // If we are lowering a subslice, the subslice offsets shall not touch the
  // contiguous part of the tile
  if (logicalMaskSpanAffineOffset & (tile.getOutDimSize(kOffset) - 1)) {
    return failure();
  }

  // fullTile.invert() is a map from kOffset, kAddr into kReg, kLane, kWarp
  // addrToOffset gives us a map from kAddr into kOffset, which is the map of
  // the addresses each lane should hold
  auto addrToOffset = fullTile.invert().compose(reps);
  // sanity check
  assert(addrToOffset.getInDimSizeLog2(kAddr) >= 3 &&
         addrToOffset.getInDimSizeLog2(kAddr) <= 6);

  // ds_read_tr* shuffles data across lanes so the lane issuing the load
  // matches the kAddr decomposition of fullTile. Using addrToOffset's
  // kAddr bases as the kLane bases of this layout lets us use laneId
  // to get the LDS offset each lane should read.
  LinearLayout addrLayout =
      LinearLayout({{kLane, addrToOffset.getBases().lookup(kAddr)},
                    {kWarp, reps.getBases().lookup(kWarp)}},
                   {{kOffset, reps.getOutDimSize(kOffset)}}, false);

  if (logicalBitWidth == 4) {
    // Writing out the corresponding abstract maps for the above LinearLayouts:
    //
    // `cvtLayout`:                    G    :         D      ->     S
    // `tile * maybeQuot`:          T ⊕ Q  :     L_C ⊕ D'  ->   C ⊕ S'
    // `reps`:                      0 ⊕ Q  :     L_C ⊕ D'  ->   C ⊕ S'
    // `addrLayout`: (0 ⊕ Q) o F^{-1} o i_A:         A      ->   C ⊕ S',
    //
    // we see that `reps` and `addrLayout`, though constructed using nibble
    // coordinates, only have byte-aligned outputs. This allows us to safely
    // convert to byte coordinates by halving the nibble-offset values with
    // `logicalToI8` and dropping the low `kReg` basis in our layouts.
    auto logicalToI8 = LinearLayout::zeros1D(2, kOffset, kOffset) *
                       LinearLayout::identity1D(reps.getOutDimSize(kOffset) / 2,
                                                kOffset, kOffset);
    reps = reps.compose(logicalToI8);
    addrLayout = addrLayout.compose(logicalToI8);

    auto numLogicalRegBases = reps.getInDimSizeLog2(kReg);
    ColumnAction dropNibbleBasis(
        llvm::to_vector(llvm::seq<size_t>(1, numLogicalRegBases)), kReg,
        numLogicalRegBases);
    reps = dropNibbleBasis.apply(reps);
    if (isPartitioned)
      partitionLayout = dropNibbleBasis.apply(partitionLayout);
  }

  // Matrix accesses are CTA-local. Model that with a trivial block output so
  // additive stride analysis always compares (offset, block) components.
  reps =
      reps.reshapeOuts({{kOffset, reps.getOutDimSize(kOffset)}, {kBlock, 1}});
  addrLayout = addrLayout.reshapeOuts(reps.getOutDims());

  // Compute the bits that are moved by one instruction
  // Compute elements for which we can swap the xor by an add
  auto elemsPerInstr = ldsParams.instBitWidth / bitWidth;
  auto [nAdditive, permStrides] =
      actionAdditiveStrides(reps, addrLayout, maskSpanAffineOffset,
                            /*maskSpanBlocks=*/0, elemsPerInstr);
  reps = permStrides.apply(reps);
  if (isPartitioned) {
    partitionLayout = permStrides.apply(partitionLayout);

    // One ds_read_tr* instruction produces `elemsPerInstr` consecutive
    // physical values along kReg from a single LDS base pointer. We only
    // select a partition once per instruction, so all of those register
    // positions must map to the same partition. For a LinearLayout that holds
    // iff the low log2(elemsPerInstr) register bases contribute 0 to
    // kPartition. Bail out if not, so a generic lowering can take over.
    for (unsigned pos = 0; pos < llvm::Log2_32(elemsPerInstr); ++pos) {
      if (partitionLayout.getBasis(kReg, pos, kPartition) != 0)
        return failure();
    }

    // partitionLayout's kLane is the destination lane which is the lane that
    // owns the loaded data in the destination tensor. The laneId is the
    // source lane issuing the load. For ds_read_tr* the hardware shuffles
    // data across lanes, so the two differ: we need to remap.
    //
    // Example: ds_load_tr8_b64 on gfx1250, from the test
    // `ds_transpose_partitioned_remaps_lane`.
    //
    //  fullTile:
    //   - lane=1 -> (1, 0)
    //     lane=2 -> (2, 0)
    //     lane=4 -> (4, 0)
    //     lane=8 -> (0, 4)
    //     lane=16 -> (0, 16)
    //   - register=1 -> (0, 1)
    //     register=2 -> (0, 2)
    //     register=4 -> (0, 8)
    //   where out dims are: [offset (size 8), addr (size 32)]
    //
    // `addr` is the non-contiguous part of the source lane's access.
    // `lane` in the inverse tile is the destination lane after the hardware
    // transpose. `fullTile.invert().sublayout({kAddr}, {kLane})` gives:
    //
    //   - addr=1 -> (0)
    //     addr=2 -> (0)
    //     addr=4 -> (8)
    //     addr=8 -> (0)
    //     addr=16 -> (16)
    //   where out dims are: [lane (size 32)]
    //
    // Then rename the input dimension from `addr` to `lane` so the map can
    // compose with partitionLayout.
    //
    // For this test, partitionLayout would choose the partition from the
    // destination-lane basis `lane=8`:
    //
    //   - register=1 -> (0)
    //     ...
    //     register=32 -> (0)
    //   - lane=1 -> (0)
    //     lane=2 -> (0)
    //     lane=4 -> (0)
    //     lane=8 -> (1)
    //     lane=16 -> (0)
    //   - warp=1 -> (0)
    //     warp=2 -> (0)
    //   where out dims are: [partition (size 2)]
    //
    // Querying this with the runtime source lane asks for the partition of
    // the wrong lane. Composing with laneRemap rewrites the partition basis
    // through the transpose:
    //
    //   - register=1 -> (0)
    //     ...
    //     register=32 -> (0)
    //   - lane=1 -> (0)
    //     lane=2 -> (0)
    //     lane=4 -> (1)
    //     lane=8 -> (0)
    //     lane=16 -> (0)
    //   - warp=1 -> (0)
    //     warp=2 -> (0)
    //   where out dims are: [partition (size 2)]
    //
    // Destination basis `lane=8` is reached from source basis `addr=4`, so
    // each source lane selects the LDS base expected by its destination lane.

    auto regIdentity = LinearLayout::identity1D(
        partitionLayout.getInDimSize(kReg), kReg, kReg);
    auto srcToDstLaneMap =
        fullTile.invert().sublayout({kAddr}, {kLane}).renameInDim(kAddr, kLane);
    auto warpIdentity = LinearLayout::identity1D(
        partitionLayout.getInDimSize(kWarp), kWarp, kWarp);
    auto laneRemap = regIdentity * srcToDstLaneMap * warpIdentity;
    partitionLayout = laneRemap.compose(partitionLayout);
  }

  // Perform computation in bytes, LLVM optimises this better
  assert(bitWidth >= 8);
  auto i8Tile =
      zerosLike(LinearLayout::identity1D(bitWidth / 8, kReg, kOffset));
  auto i8AddrLayout = i8Tile * addrLayout;

  auto outDims = llvm::to_vector(i8AddrLayout.getOutDimNames());
  bool rematerializeLane = i8AddrLayout.hasInDim(kLane) &&
                           !i8AddrLayout.sublayoutIsZero({kLane}, outDims);
  bool rematerializeWarp = i8AddrLayout.hasInDim(kWarp) &&
                           !i8AddrLayout.sublayoutIsZero({kWarp}, outDims);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  if (auto group = op->getAttrOfType<IntegerAttr>(
          "tlx.rematerialize_coordinates_group")) {
    std::tie(laneId, warpId) =
        coordinateGroups->getOrCreate(op, group.getInt(), rematerializeLane,
                                      rematerializeWarp, rewriter, targetInfo);
  } else if (op->hasAttr("tlx.rematerialize_coordinates")) {
    if (rematerializeLane)
      laneId =
          targetInfo.rematerializeDistributedCoordinate(rewriter, loc, laneId);
    if (rematerializeWarp)
      warpId =
          targetInfo.rematerializeDistributedCoordinate(rewriter, loc, warpId);
  }
  auto regBase =
      applyLinearLayout(
          loc, rewriter, i8AddrLayout,
          {{kReg, b.i32_val(0)}, {kLane, laneId}, {kWarp, warpId}})[0]
          .second;

  // It's fine that we don't compute the offset in bytes as affineOffset
  // will be folded into a constant
  auto affineOffsetI8 = b.mul(affineOffset, b.i32_val(bitWidth / 8));
  bool hasPadding = !paddingShifts.empty();
  Value paddedAffineOffsetI8 = b.i32_val(0);
  if (hasPadding && maskSpanAffineOffset != 0) {
    // `maskSpanAffineOffset != 0` indicates the affine offsets come from
    // MemDescSubsliceOp, whose verifier guarantees that the affine offsets
    // are bitwise disjoint from other offset contributors. Padding can thus
    // be applied separately. This helps LLVM reuse base pointers.
    paddedAffineOffsetI8 =
        applyPadding(loc, rewriter, affineOffsetI8, paddingShifts);
  } else {
    regBase = b.xor_(regBase, affineOffsetI8);
  }

  auto vecTy = vec_ty(llvmElemTy, elemsPerInstr);
  for (int i = 0; i < reps.getInDimSize(kReg); i += nAdditive) {
    auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
    auto regIdxI8 = regIdx * (bitWidth / 8);
    Value offset = b.xor_(regBase, b.i32_val(regIdxI8));

    if (hasPadding) {
      offset = applyPadding(loc, rewriter, offset, paddingShifts);
      if (maskSpanAffineOffset != 0)
        offset = b.add(offset, paddedAffineOffsetI8);
    }

    for (int i2 = 0; i2 < nAdditive; i2 += elemsPerInstr) {
      // all these constants will go as immediate values to ds_read_tr
      auto regIdxAdd =
          reps.apply({{kReg, i2}, {kLane, 0}, {kWarp, 0}})[0].second;
      auto regIdxAddI8 = regIdxAdd * (bitWidth / 8);
      // `actionAdditiveStrides` forces `regIdxAddI8` and `offset` to be
      // bitwise disjoint, so we can calculate their padding contributions
      // separately.
      regIdxAddI8 = applyPadding(regIdxAddI8, paddingShifts);
      Value innerOffset = b.add(offset, b.i32_val(regIdxAddI8));
      Value smemBaseVal = smemBases[0];
      if (isPartitioned) {
        auto partOut = applyLinearLayout(
            loc, rewriter, partitionLayout,
            {{kReg, b.i32_val(i + i2)}, {kLane, laneId}, {kWarp, warpId}});
        smemBaseVal = b.extract_element(basesVec, partOut[0].second);
      }
      auto vecAddr = b.gep(smemPtrTy, i8_ty, smemBaseVal, innerOffset,
                           LLVM::GEPNoWrapFlags::inbounds);
      llvm::append_range(vals,
                         emitDsReadTr(op, loc, vecAddr, vecTy, llvmElemTy,
                                      logicalBitWidth, rewriter, targetInfo));
    }
  }
  // apply all the inverse permutations in the reverse order
  assert(vals.size() == reps.getInDimSize(kReg));
  vals = permStrides.inverse().apply(vals);

  return success();
}

template <typename OpTy>
class TransLocalLoadOpConversion : public ConvertOpToLLVMPattern<OpTy> {
  static constexpr bool isPackedTransposed =
      std::is_same_v<OpTy, triton::amdgpu::LocalLoadPackedTransposedOp>;

public:
  TransLocalLoadOpConversion(
      const LLVMTypeConverter &converter, const AMD::TargetInfo &targetInfo,
      PatternBenefit benefit,
      std::shared_ptr<DistributedCoordinateGroups> coordinateGroups)
      : ConvertOpToLLVMPattern<OpTy>(converter, benefit),
        targetInfo(targetInfo), coordinateGroups(std::move(coordinateGroups)) {}
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();

    auto typeConverter = this->getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    unsigned bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    unsigned logicalBitWidth = bitWidth;
    if constexpr (isPackedTransposed) {
      // FP4 is represented as packed elements inside i8 values.
      if (bitWidth != 8)
        return failure();
      logicalBitWidth = 4;
    } else {
      // FP4 is represented as i8 and, when packed along K, can be
      // transposed using ds_read_tr8 which doesn't change packing.
      if (bitWidth != 16 && bitWidth != 8)
        return failure();
    }
    auto ldsParamsVec = targetInfo.queryLDSTransLoadParams(logicalBitWidth);
    if (ldsParamsVec.empty())
      return failure();
    if (SharedMemoryObject::getMaskSpanOffsetsAndBlocks(srcTy).second != 0)
      return failure();

    auto dstLL = triton::gpu::toLinearLayout(dstTy);
    LinearLayout sharedLL = triton::gpu::toLinearLayoutIgnoringPadding(srcTy);

    if constexpr (isPackedTransposed) {
      // Perform factorization and address routing in logical fp4 coordinates.
      std::optional<StringAttr> srcPackedDim;
      std::optional<StringAttr> dstPackedDim;
      auto srcShape = srcTy.getShape();
      auto dstShape = dstTy.getShape();
      assert(srcShape.size() == dstShape.size());
      auto outDimNames = llvm::to_vector(sharedLL.getOutDimNames());

      for (unsigned dim = 0; dim < srcShape.size(); ++dim) {
        if (srcShape[dim] * 2 == dstShape[dim]) {
          srcPackedDim = outDimNames[dim];
          continue;
        }
        if (dstShape[dim] * 2 == srcShape[dim]) {
          dstPackedDim = outDimNames[dim];
          continue;
        }
        if (srcShape[dim] != dstShape[dim])
          return failure();
      }
      if (!srcPackedDim || !dstPackedDim)
        return failure();

      auto kReg = str_attr("register");
      auto kOffset = str_attr("offset");
      sharedLL = LinearLayout::identity1D(2, kOffset, *srcPackedDim) * sharedLL;
      dstLL = LinearLayout::identity1D(2, kReg, *dstPackedDim) * dstLL;
    }

    auto cvtDstLL = dstLL.invertAndCompose(sharedLL);
    auto kBlock = StringAttr::get(ctx, "block");
    auto maybeSublayout = cvtDstLL.quotient({kBlock});
    if (!maybeSublayout)
      return failure();
    cvtDstLL = maybeSublayout.value();

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    SmallVector<Value> smemBases = llvm::to_vector(smemObj.getBases());
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
    auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);
    auto paddingShifts = getPaddedSharedShifts(srcTy.getEncoding(),
                                               srcTy.getElementTypeBitWidth(),
                                               /*offsetInBytes=*/true);

    for (const auto &ldsParams : ldsParamsVec) {
      if (triton::gpu::isPaddedEncoding(srcTy.getEncoding()) &&
          triton::gpu::getMinInterval(srcTy.getEncoding()) <
              ldsParams.instBitWidth / bitWidth) {
        continue;
      }

      SmallVector<Value> values;
      auto result = lowerDsReadTr(
          op, ldsParams, loc, cvtDstLL, logicalBitWidth, values, smemBases,
          affineOffset, maskSpanAffineOffset, paddingShifts, llvmElemTy,
          coordinateGroups, rewriter, targetInfo);
      if (failed(result))
        continue;

      auto value =
          packTensorElements(loc, typeConverter, values, rewriter, dstTy);

      rewriter.replaceOp(op, value);
      return success();
    }
    return failure();
  }

private:
  const AMD::TargetInfo &targetInfo;
  std::shared_ptr<DistributedCoordinateGroups> coordinateGroups;
};

struct LocalAtomicScatterRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAtomicScatterRMWOp> {

  LocalAtomicScatterRMWOpConversion(const LLVMTypeConverter &converter,
                                    const AMD::TargetInfo &targetInfo,
                                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAtomicScatterRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto lowering = prepareLocalAtomicScatterRMW(
        op, adaptor.getDst(), adaptor.getIndices(), adaptor.getValues(),
        op.getMask() ? adaptor.getMask() : Value(), rewriter, targetInfo,
        getTypeConverter());
    if (failed(lowering))
      return failure();
    LocalAtomicScatterRMWInfo &info = *lowering;

    auto binOp = matchAtomicOp(op.getAtomicRmwOp());
    if (!binOp)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW operation");

    // Lower to per-element llvm.atomicrmw on addrspace(3) with
    // syncscope("workgroup") monotonic.
    const auto memOrder = LLVM::AtomicOrdering::monotonic;
    const StringRef scope = "workgroup";
    LLVM::AMD::AtomicRMWEmitter emitter(targetInfo, *binOp, memOrder, scope);

    bool returnOld = !op.getResult().use_empty();

    if (llvm::any_of(info.addrs, [](const LocalSharedMemoryAddress &addr) {
          return bool(addr.ctaId);
        })) {
      return rewriter.notifyMatchFailure(
          op, "cross-CTA shared atomics are not supported on AMDGPU");
    }

    SmallVector<Value> results;
    if (returnOld)
      results.reserve(info.addrs.size());

    for (auto [i, addrAndValue] :
         llvm::enumerate(llvm::zip(info.addrs, info.values))) {
      auto [addr, value] = addrAndValue;
      Value rmwMask = triton::gpu::maybeAnd(
          rewriter, loc, info.threadPred,
          info.maskValues.empty() ? Value() : info.maskValues[i]);
      // emitAtomicRMW requires a non-null predicate, default to true if null.
      if (!rmwMask)
        rmwMask = b.true_val();

      Value old = emitter.emitAtomicRMW(rewriter, addr.ptr, value, rmwMask,
                                        /*sharedMemBase=*/std::nullopt,
                                        /*enableIntraWaveReduce=*/false);
      if (returnOld)
        results.push_back(old);
    }

    if (!returnOld) {
      rewriter.eraseOp(op);
      return success();
    }

    finalizeTensorAtomicResults(op, info.valuesTy, rewriter, results,
                                info.llvmElemTy, b, info.threadPred, targetInfo,
                                getTypeConverter());
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

static FailureOr<SmallVector<Value>>
packMfmaDotOperandFragments(Value value, RankedTensorType tensorTy,
                            unsigned opIdx, ArrayRef<int64_t> expectedRep,
                            int64_t kBase,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter, Location loc) {
  auto dotEncoding =
      dyn_cast<triton::gpu::DotOperandEncodingAttr>(tensorTy.getEncoding());
  auto mfmaEncoding =
      dotEncoding
          ? dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(dotEncoding.getParent())
          : triton::gpu::AMDMfmaEncodingAttr();
  if (!mfmaEncoding || dotEncoding.getOpIdx() != opIdx ||
      !llvm::is_contained({4u, 8u}, dotEncoding.getKWidth()))
    return failure();

  SmallVector<int64_t> rep = mfmaEncoding.getRepForOperand(
      tensorTy.getShape(), dotEncoding.getKWidth(), opIdx);
  if (rep != expectedRep)
    return failure();

  int64_t batch = rep[0];
  int64_t nonKRep = rep[opIdx == 0 ? 1 : 2];
  int64_t kRep = rep[opIdx == 0 ? 2 : 1];
  int64_t numKVec = kRep * dotEncoding.getKWidth() / kBase;
  if (numKVec <= 0)
    return failure();

  SmallVector<Value> elems =
      unpackTensorElements(loc, value, rewriter, tensorTy);
  SmallVector<int64_t> strides =
      computeStrides({batch, nonKRep, numKVec, kBase});
  if (elems.size() != static_cast<size_t>(batch * nonKRep * numKVec * kBase))
    return failure();

  Type elemTy = typeConverter->convertType(tensorTy.getElementType());
  auto vecTy = vec_ty(elemTy, kBase);
  TritonLLVMOpBuilder b(loc, rewriter);
  SmallVector<Value> fragments;
  for (int64_t batchIdx = 0; batchIdx < batch; ++batchIdx) {
    for (int64_t nonKIdx = 0; nonKIdx < nonKRep; ++nonKIdx) {
      for (int64_t kVecIdx = 0; kVecIdx < numKVec; ++kVecIdx) {
        Value fragment = b.undef(vecTy);
        for (int64_t k = 0; k < kBase; ++k) {
          int64_t index = linearize({batchIdx, nonKIdx, kVecIdx, k}, strides);
          fragment =
              b.insert_element(vecTy, fragment, elems[index], b.i32_val(k));
        }
        fragments.push_back(fragment);
      }
    }
  }
  return fragments;
}

// Passes an MFMA occupies the matrix pipeline for, per LLVM's schedule model
static FailureOr<int> getMfmaNumPasses(ArrayRef<unsigned> instrShape) {
  if (instrShape == ArrayRef<unsigned>({32, 32, 8}) ||
      instrShape == ArrayRef<unsigned>({32, 32, 16}))
    return 16;
  if (instrShape == ArrayRef<unsigned>({16, 16, 16}) ||
      instrShape == ArrayRef<unsigned>({16, 16, 32}))
    return 8;
  return failure();
}

// Wait states between an MFMA writing its destination and any consumer reading
// it
static FailureOr<int> getMfmaDrainWaitStates(ISAFamily isaFamily,
                                             ArrayRef<unsigned> instrShape) {
  FailureOr<int> numPasses = getMfmaNumPasses(instrShape);
  if (failed(numPasses))
    return failure();
  if (isaFamily == ISAFamily::CDNA3)
    return *numPasses + 3;
  // CDNA4 adds one wait state, except for 2-pass instructions.
  if (isaFamily == ISAFamily::CDNA4)
    return *numPasses + 3 + (*numPasses != 2 ? 1 : 0);
  return failure();
}

struct ScheduledMfmaLoweringInfo {
  StringRef intrinsicName;
  // K elements one lane feeds into a single MFMA; sets fragment width.
  int64_t kBase;
  // The gfx90a+ bf16 `_1k` intrinsics take packed i16 vectors.
  bool intrinsicOperandsAreI16;
};

static FailureOr<ScheduledMfmaLoweringInfo>
getScheduledMfmaLoweringInfo(Location loc,
                             triton::gpu::AMDMfmaEncodingAttr mfma,
                             Type aElemType, Type bElemType) {
  // Reuse the backend-wide intrinsic table so this path cannot drift from the
  // intrinsic the ordinary dot lowering picks for the same layout.
  ArrayRef<unsigned> instrShape = mfma.getInstrShape();
  FailureOr<MfmaIntrinsic> intrinsic = MfmaIntrinsic::get(
      loc, mfma.getVersion(), instrShape[0], instrShape[1], instrShape[2],
      aElemType, bElemType, /*withScale=*/false, /*useTF32=*/false);
  if (failed(intrinsic))
    return failure();

  bool intrinsicOperandsAreI16 =
      intrinsic->name == ROCDL::mfma_f32_32x32x8bf16_1k::getOperationName() ||
      intrinsic->name == ROCDL::mfma_f32_16x16x16bf16_1k::getOperationName();
  return ScheduledMfmaLoweringInfo{intrinsic->name,
                                   static_cast<int64_t>(intrinsic->kBase),
                                   intrinsicOperandsAreI16};
}

static FailureOr<Value>
constrainMfmaFragmentRegisterClass(Value fragment, StringRef registerClass,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc, bool hasSideEffects = false,
                                   LLVM::InlineAsmOp *pin = nullptr) {
  auto fragmentTy = cast<VectorType>(fragment.getType());
  unsigned elementBitWidth =
      getIntOrFloatOrPtrBitWidth(fragmentTy.getElementType());
  int64_t totalBitWidth = fragmentTy.getNumElements() * elementBitWidth;
  if (totalBitWidth <= 0 || totalBitWidth % 32 != 0)
    return failure();

  int64_t registerCount = totalBitWidth / 32;
  auto registerVectorTy = vec_ty(i32_ty, registerCount);
  TritonLLVMOpBuilder b(loc, rewriter);
  Value packed = b.bitcast(fragment, registerVectorTy);
  auto *ctx = rewriter.getContext();
  auto asmDialect = LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
  auto operandAttrs = ArrayAttr::get(ctx, {});
  StringRef outputConstraint = registerClass == "agpr" ? "=a" : "=v";
  std::string constraints = outputConstraint.str() + ",0";
  auto identity = LLVM::InlineAsmOp::create(
      rewriter, loc, registerVectorTy, ValueRange{packed}, "", constraints,
      hasSideEffects,
      /*is_align_stack=*/false, LLVM::TailCallKind::None, asmDialect,
      operandAttrs);
  if (pin)
    *pin = identity;
  Value constrained = b.bitcast(identity->getResult(0), fragmentTy);
  return constrained;
}

// Build an asm snippet providing exactly `waitStates` MFMA wait states.
//
// `s_nop N` provides N+1 wait states, and N is a 4-bit field, so a single
// instruction covers at most 16. Two are always enough here: the largest
// requirement LLVM models for gfx950 is 20 wait states
// (GFX940_XDL_N_PassWritesVGPROverlappedSrcABWaitStates for a 16-pass MFMA).
//
//    1  ->  "s_nop 0"
//    4  ->  "s_nop 3"
//   16  ->  "s_nop 15"
//   18  ->  "s_nop 15\ns_nop 1"
//   20  ->  "s_nop 15\ns_nop 3"
static std::string mfmaWaitStateAsm(int waitStates) {
  // A non-positive count would silently emit no padding at all, which is the
  // exact hazard this padding exists to prevent.
  assert(waitStates > 0 && waitStates <= 32 &&
         "MFMA wait states must be positive and fit in two s_nops");
  if (waitStates <= 16)
    return "s_nop " + std::to_string(waitStates - 1);
  return "s_nop 15\ns_nop " + std::to_string(waitStates - 16 - 1);
}

// LLVM can forward a stored MFMA destination into a later load, including by
// promoting private allocations. Preserve that dependency in the consumer
// analysis even though the lowered IR has no SSA edge between store and load.
class MfmaMemoryForwarding {
public:
  explicit MfmaMemoryForwarding(LLVM::LLVMFuncOp function)
      : function(function), aliases(function) {
    function.walk([&](Operation *op) {
      if (!isa_and_nonnull<LLVM::LLVMDialect>(op->getDialect()) ||
          op->getNumRegions() || isa<LLVM::InlineAsmOp>(op))
        return;
      // Inline assembly's hidden memory accesses cannot be promoted into SSA;
      // its explicit register operands are handled by the normal value walk.
      // Missing effect interfaces (e.g. masked loads) remain conservative.
      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      if (!effects || effects.hasEffect<MemoryEffects::Read>())
        readers.push_back(op);
    });
  }

  // Returns true for a memory escape that needs completion, otherwise appends
  // possible reloads. Cache edges once per store, shared by all result pins.
  bool appendReloads(LLVM::StoreOp store, SmallVectorImpl<Value> &worklist) {
    auto [it, inserted] = forwarding.try_emplace(store);
    auto &uses = it->second;
    if (inserted) {
      // Kernels are unreferenced external definitions. Callable helpers can
      // expose stored values to an inlined caller outside this local walk.
      uses.escapes =
          function.getLinkage() != LLVM::Linkage::External ||
          !SymbolTable::symbolKnownUseEmpty(function, function->getParentOp());
      for (Operation *reader : readers) {
        if (!mayExecuteAfter(reader, store))
          continue;
        if (auto load = dyn_cast<LLVM::LoadOp>(reader)) {
          if (!disjointAddressSpaces(store.getAddr(), load.getAddr()) &&
              !aliases.alias(store.getAddr(), load.getAddr()).isNo())
            uses.loads.push_back(load.getResult());
        } else if (aliases.getModRef(reader, store.getAddr()).isRef()) {
          // Calls, copies, and unmodeled reads can expose the stored bytes
          // after inlining or memory optimization. Do not lose that path.
          uses.escapes = true;
        }
      }
    }
    llvm::append_range(worklist, uses.loads);
    return uses.escapes;
  }

private:
  bool mayExecuteAfter(Operation *reader, Operation *store) {
    // Flat CFG reachability does not model region exits or enclosing loops.
    if (reader->getParentRegion() != &function.getBody() ||
        store->getParentRegion() != &function.getBody())
      return true;
    Block *from = store->getBlock();
    Block *to = reader->getBlock();
    // isReachable starts at successors, so reaching the same block requires
    // a real backedge. Acyclic epilogue stores cannot feed earlier loads.
    return (from == to && store->isBeforeInBlock(reader)) ||
           from->isReachable(to);
  }

  static bool disjointAddressSpaces(Value lhs, Value rhs) {
    unsigned a = cast<LLVM::LLVMPointerType>(lhs.getType()).getAddressSpace();
    unsigned b = cast<LLVM::LLVMPointerType>(rhs.getType()).getAddressSpace();
    // Only global, shared, and private are pairwise disjoint. Flat, constant,
    // and buffer/resource spaces must not be excluded by their number alone.
    auto isDistinctSpace = [](unsigned space) {
      return llvm::is_contained({llvm::AMDGPUAS::GLOBAL_ADDRESS,
                                 llvm::AMDGPUAS::LOCAL_ADDRESS,
                                 llvm::AMDGPUAS::PRIVATE_ADDRESS},
                                space);
    };
    return a != b && isDistinctSpace(a) && isDistinctSpace(b);
  }

  struct ForwardedUses {
    SmallVector<Value> loads;
    bool escapes = false;
  };
  LLVM::LLVMFuncOp function;
  AliasAnalysis aliases;
  SmallVector<Operation *> readers;
  DenseMap<Operation *, ForwardedUses> forwarding;
};

// Native consumers are visible to LLVM's hazard recognizer, but an instruction
// in inline asm is not. Follow result identities through the lowered IR because
// LLVM can fold even arithmetic (for example, adding zero) back to the MFMA
// destination. A later MFMA starts a new chain with its own result pin.
// Strengthen an explicit commit reached by a persistent result instead of
// draining every producer separately. Keep its register constraints: a live
// operand handoff can complete the results without moving them into AGPRs.
static bool needsOpaqueConsumerDrain(
    Operation *pin, int requiredWait,
    DenseMap<Operation *, int> &commitWaitStates,
    DenseMap<Operation *, std::unique_ptr<MfmaMemoryForwarding>>
        &memoryForwarding) {
  SmallVector<Value> worklist(pin->getResults());
  llvm::SmallDenseSet<Value, 16> visited;
  // Visit all paths even after finding an escape, so commit strengthening
  // does not depend on use-list order. The escape still guards this root.
  bool needsDrain = false;
  auto appendRegionSuccessors = [&](RegionBranchOpInterface branch,
                                    RegionBranchPoint point, OpOperand &use) {
    RegionBranchSuccessorMapping mapping;
    branch.getSuccessorOperandInputMapping(mapping, point);
    auto found = mapping.find(&use);
    if (found != mapping.end())
      llvm::append_range(worklist, found->second);
  };

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *consumer = use.getOwner();
      if (consumer->getName().getStringRef().starts_with("rocdl.mfma."))
        continue;

      if (auto inlineAsm = dyn_cast<LLVM::InlineAsmOp>(consumer)) {
        auto commit = commitWaitStates.find(consumer);
        if (commit != commitWaitStates.end()) {
          if (commit->second < requiredWait) {
            inlineAsm.setAsmString(mfmaWaitStateAsm(requiredWait));
            commit->second = requiredWait;
          }
          // Only this value path is complete. Other uses can bypass the
          // boundary and still require a guard at the producer.
          continue;
        } else if (!inlineAsm.getAsmString().empty()) {
          needsDrain = true;
          continue;
        }
        llvm::append_range(worklist, consumer->getResults());
        continue;
      }

      // Calls and returns can hide a consumer outside this conversion. Invoke
      // also implements BranchOpInterface, so check it before CFG forwarding.
      if (isa<LLVM::CallOp, LLVM::CallIntrinsicOp, LLVM::InvokeOp,
              LLVM::ReturnOp>(consumer)) {
        needsDrain = true;
        continue;
      }

      if (auto store = dyn_cast<LLVM::StoreOp>(consumer)) {
        if (use.getOperandNumber() != 0)
          continue;
        auto function = store->getParentOfType<LLVM::LLVMFuncOp>();
        if (!function) {
          needsDrain = true;
          continue;
        }
        auto &memory = memoryForwarding[function];
        if (!memory)
          memory = std::make_unique<MfmaMemoryForwarding>(function);
        if (memory->appendReloads(store, worklist))
          needsDrain = true;
        continue;
      }

      if (auto branch = dyn_cast<BranchOpInterface>(consumer)) {
        for (unsigned i = 0; i < consumer->getNumSuccessors(); ++i) {
          SuccessorOperands operands = branch.getSuccessorOperands(i);
          OperandRange forwarded = operands.getForwardedOperands();
          if (forwarded.empty())
            continue;
          unsigned begin = forwarded.getBeginOperandIndex();
          unsigned index = use.getOperandNumber();
          if (index >= begin && index - begin < forwarded.size())
            worklist.push_back(consumer->getSuccessor(i)->getArgument(
                operands.getProducedOperandCount() + index - begin));
        }
        continue;
      }

      // warp_predicate has capture-only regions and no region branch
      // interface. Its init values bypass the body on inactive lanes.
      if (auto predicate = dyn_cast<triton::gpu::WarpPredicateOp>(consumer)) {
        unsigned index = use.getOperandNumber();
        if (index != 0)
          worklist.push_back(predicate.getResult(index - 1));
        continue;
      }
      if (auto yield = dyn_cast<triton::gpu::PredicateYieldOp>(consumer)) {
        worklist.push_back(
            yield->getParentOp()->getResult(use.getOperandNumber()));
        continue;
      }

      // This includes warp_specialize partition captures and warp_yield, as
      // well as SCF loop-carried values and conditional results.
      if (auto branch = dyn_cast<RegionBranchOpInterface>(consumer)) {
        appendRegionSuccessors(branch, RegionBranchPoint::parent(), use);
        continue;
      }
      if (auto terminator =
              dyn_cast<RegionBranchTerminatorOpInterface>(consumer)) {
        auto parent =
            dyn_cast<RegionBranchOpInterface>(consumer->getParentOp());
        if (!parent) {
          needsDrain = true;
          continue;
        }
        appendRegionSuccessors(parent, RegionBranchPoint(terminator), use);
        continue;
      }

      // Aggregates, vector operations, casts, selects, and native arithmetic
      // can all preserve the destination through LLVM optimization.
      if (consumer->getNumRegions() == 0 &&
          (isa_and_nonnull<LLVM::LLVMDialect, ROCDL::ROCDLDialect>(
               consumer->getDialect()) ||
           isa<UnrealizedConversionCastOp>(consumer))) {
        if (isa_and_nonnull<LLVM::LLVMDialect>(consumer->getDialect())) {
          auto effects = dyn_cast<MemoryEffectOpInterface>(consumer);
          // Masked/scattered stores and other unmodeled writers cannot be
          // treated as dead ends just because they have no SSA results.
          if (!effects || effects.hasEffect<MemoryEffects::Write>()) {
            needsDrain = true;
            continue;
          }
        }
        llvm::append_range(worklist, consumer->getResults());
        continue;
      }

      // An unmodeled operation or region may let the destination escape.
      needsDrain = true;
    }
  }
  return needsDrain;
}

class RematerializedRangeOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::RematerializedRangeOp> {
public:
  RematerializedRangeOpConversion(const LLVMTypeConverter &converter,
                                  const AMD::TargetInfo &targetInfo,
                                  PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::amdgpu::RematerializedRangeOp>(converter,
                                                                      benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = triton::amdgpu::RematerializedRangeOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::amdgpu::RematerializedRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto tensorTy = cast<RankedTensorType>(op.getResult().getType());
    TritonLLVMOpBuilder b(loc, rewriter);

    // Start a fresh machine live range for the thread coordinates at each
    // source location. The empty tied inline asm emits no instruction, but its
    // side effect prevents LLVM from CSEing the derived range arithmetic back
    // into an earlier location and recreating the lifetime this op is meant to
    // split.
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    LinearLayout layout = triton::gpu::toLinearLayout(tensorTy);
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    auto outDims = llvm::to_vector(layout.getOutDimNames());
    if (layout.hasInDim(kLane) && !layout.sublayoutIsZero({kLane}, outDims))
      laneId =
          targetInfo.rematerializeDistributedCoordinate(rewriter, loc, laneId);
    if (layout.hasInDim(kWarp) && !layout.sublayoutIsZero({kWarp}, outDims))
      warpId =
          targetInfo.rematerializeDistributedCoordinate(rewriter, loc, warpId);
    Value blockId = targetInfo.getClusterCTAId(rewriter, loc);

    SmallVector<Value> values;
    values.reserve(layout.getInDimSize(kRegister));
    for (unsigned reg = 0; reg < layout.getInDimSize(kRegister); ++reg) {
      auto indices = applyLinearLayout(loc, rewriter, layout,
                                       {{kRegister, b.i32_val(reg)},
                                        {kLane, laneId},
                                        {kWarp, warpId},
                                        {kBlock, blockId}});
      if (indices.size() != 1)
        return rewriter.notifyMatchFailure(
            op, "rank-one range layout produced multiple coordinates");
      values.push_back(b.add(indices.front().second, b.i32_val(op.getStart())));
    }

    Value result =
        packTensorElements(loc, getTypeConverter(), values, rewriter, tensorTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

class RegisterResidentOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::RegisterResidentOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::amdgpu::RegisterResidentOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = triton::amdgpu::RegisterResidentOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::amdgpu::RegisterResidentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();
    auto tensorTy = cast<RankedTensorType>(op.getInput().getType());
    Type elemTy = typeConverter->convertType(tensorTy.getElementType());
    unsigned bitWidth = getIntOrFloatOrPtrBitWidth(elemTy);
    unsigned registersPerGroup = op.getRegistersPerGroup();
    unsigned elementsPerGroup = registersPerGroup * 32 / bitWidth;
    SmallVector<Value> elements =
        unpackTensorElements(loc, adaptor.getInput(), rewriter, tensorTy);
    if (elements.empty() || elements.size() % elementsPerGroup != 0)
      return rewriter.notifyMatchFailure(
          op, "native tuple does not divide the per-thread elements");

    StringRef registerClass = op.getRegisterClass();
    StringRef outputConstraint = registerClass == "agpr" ? "=a" : "=v";
    auto asmDialect = LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
    auto operandAttrs = ArrayAttr::get(ctx, {});
    auto elementVectorTy = vec_ty(elemTy, elementsPerGroup);
    auto registerVectorTy = vec_ty(i32_ty, registersPerGroup);
    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> registerGroups;

    for (unsigned begin = 0; begin < elements.size();
         begin += elementsPerGroup) {
      Value elementVector = b.undef(elementVectorTy);
      for (unsigned index = 0; index < elementsPerGroup; ++index)
        elementVector =
            b.insert_element(elementVectorTy, elementVector,
                             elements[begin + index], b.i32_val(index));
      registerGroups.push_back(b.bitcast(elementVector, registerVectorTy));
    }

    std::string constraints;
    for (unsigned index = 0; index < registerGroups.size(); ++index) {
      if (!constraints.empty())
        constraints += ",";
      constraints += outputConstraint;
    }
    for (unsigned index = 0; index < registerGroups.size(); ++index)
      constraints += "," + std::to_string(index);
    Type asmResultTy = registerVectorTy;
    if (registerGroups.size() != 1)
      asmResultTy = LLVM::LLVMStructType::getLiteral(
          ctx, SmallVector<Type>(registerGroups.size(), registerVectorTy));
    Value asmResult = LLVM::InlineAsmOp::create(
                          rewriter, loc, asmResultTy, registerGroups,
                          /*asm_string=*/"", constraints,
                          /*has_side_effects=*/false,
                          /*is_align_stack=*/false, LLVM::TailCallKind::None,
                          asmDialect, operandAttrs)
                          .getRes();

    SmallVector<Value> constrainedElements;
    constrainedElements.reserve(elements.size());
    for (unsigned group = 0; group < registerGroups.size(); ++group) {
      Value constrained = registerGroups.size() == 1
                              ? asmResult
                              : b.extract_val(asmResult, group);
      Value restored = b.bitcast(constrained, elementVectorTy);
      for (unsigned index = 0; index < elementsPerGroup; ++index)
        constrainedElements.push_back(
            b.extract_element(elemTy, restored, b.i32_val(index)));
    }

    Value result = packTensorElements(loc, typeConverter, constrainedElements,
                                      rewriter, op.getResult().getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

class RegisterClassAnchorOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::RegisterClassAnchorOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::amdgpu::RegisterClassAnchorOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = triton::amdgpu::RegisterClassAnchorOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::amdgpu::RegisterClassAnchorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();
    auto tensorTy = cast<RankedTensorType>(op.getInput().getType());
    Type elemTy = typeConverter->convertType(tensorTy.getElementType());
    unsigned bitWidth = getIntOrFloatOrPtrBitWidth(elemTy);
    unsigned elementsPerRegister = 32 / bitWidth;
    SmallVector<Value> elements =
        unpackTensorElements(loc, adaptor.getInput(), rewriter, tensorTy);
    if (elements.empty() || elements.size() % elementsPerRegister != 0)
      return rewriter.notifyMatchFailure(
          op, "native register does not divide the per-thread elements");

    StringRef outputConstraint = op.getRegisterClass() == "agpr" ? "=a" : "=v";
    std::string constraints = outputConstraint.str() + ",0";
    auto asmDialect = LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
    auto operandAttrs = ArrayAttr::get(ctx, {});
    Type registerTy = elementsPerRegister == 1
                          ? elemTy
                          : Type(vec_ty(elemTy, elementsPerRegister));
    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> constrainedElements;
    constrainedElements.reserve(elements.size());

    for (unsigned begin = 0; begin < elements.size();
         begin += elementsPerRegister) {
      Value registerValue = elements[begin];
      if (elementsPerRegister != 1) {
        registerValue = b.undef(registerTy);
        for (unsigned index = 0; index < elementsPerRegister; ++index)
          registerValue =
              b.insert_element(registerTy, registerValue,
                               elements[begin + index], b.i32_val(index));
      }

      Value asmResult = LLVM::InlineAsmOp::create(
                            rewriter, loc, registerTy, registerValue,
                            /*asm_string=*/"", constraints,
                            /*has_side_effects=*/true,
                            /*is_align_stack=*/false, LLVM::TailCallKind::None,
                            asmDialect, operandAttrs)
                            .getRes();
      if (elementsPerRegister == 1) {
        constrainedElements.push_back(asmResult);
        continue;
      }
      for (unsigned index = 0; index < elementsPerRegister; ++index)
        constrainedElements.push_back(
            b.extract_element(elemTy, asmResult, b.i32_val(index)));
    }

    Value result = packTensorElements(loc, typeConverter, constrainedElements,
                                      rewriter, op.getResult().getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Validate the encoding version against targetInfo in the lowering
static LogicalResult verifyMfmaVersionMatchesTarget(
    Operation *op, triton::gpu::AMDMfmaEncodingAttr mfma, ISAFamily isaFamily) {
  if (!llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4}, isaFamily))
    return op->emitOpError(
        "is supported only on CDNA3 (gfx942) and CDNA4 (gfx950)");
  unsigned expected = isaFamily == ISAFamily::CDNA3 ? 3 : 4;
  if (mfma.getVersion() != expected)
    return op->emitOpError() << "carries a version " << mfma.getVersion()
                             << " MFMA layout, which does not match the CDNA"
                             << expected << " target";
  return success();
}

// `auto` derives storage from the role alone, identically on every target.
// Targets that cannot honor it reject it in the verifier.
static StringRef resolveAccumulatorStorage(triton::amdgpu::ScheduledMfmaOp op) {
  StringRef storage = op.getAccumulatorRegisterClass();
  if (storage != "auto")
    return storage;
  return op.getAccumulatorRole() == "persistent" ? "agpr" : "vgpr";
}

// Return the first accumulator reaching this boundary that its producer pinned
// into AGPRs, or null if none is provably AGPR-resident.
static triton::amdgpu::ScheduledMfmaOp
findAgprResidentAccumulator(triton::amdgpu::MfmaCommitOp op, size_t &index) {
  for (auto [inputIndex, input] : llvm::enumerate(op.getInputs())) {
    if (!cast<RankedTensorType>(input.getType()).getElementType().isF32())
      continue;
    auto producer = input.getDefiningOp<triton::amdgpu::ScheduledMfmaOp>();
    if (producer && resolveAccumulatorStorage(producer) == "agpr") {
      index = inputIndex;
      return producer;
    }
  }
  return nullptr;
}

// Prove register-class-stable inputs without looking through an uncompleted
// native MFMA. Persistent result pins terminate the walk, including at loop
// backedges; unknown producers and cycles consisting only of forwarding fail
// closed. Constants can seed an accumulator before its first update.
static bool hasPinnedCommitInput(Value value, StringRef outputConstraint,
                                 const AMD::ScheduledMfmaLoweringState &state,
                                 ConversionPatternRewriter &rewriter,
                                 bool accumulator,
                                 llvm::DenseSet<Value> &active, bool &sawPin) {
  if (!active.insert(value).second)
    return false;
  auto finish = [&](bool result) {
    active.erase(value);
    return result;
  };
  auto recurse = [&](Value input) {
    return hasPinnedCommitInput(input, outputConstraint, state, rewriter,
                                accumulator, active, sawPin);
  };
  // CFG operands can still reference source values pending dialect-conversion
  // replacement. Follow the compiler's mapping, never infer a result pin from
  // a source op or an IR attribute. An unmapped materialization fails closed.
  if (isa<RankedTensorType>(value.getType()) &&
      !value.getDefiningOp<UnrealizedConversionCastOp>()) {
    Value converted = rewriter.getRemappedValue(value);
    return finish(converted && converted != value && recurse(converted));
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    bool sawIncoming = false;
    for (Block *predecessor : argument.getOwner()->getPredecessors()) {
      auto branch = dyn_cast<BranchOpInterface>(predecessor->getTerminator());
      if (!branch)
        return finish(false);
      for (auto [index, successor] : llvm::enumerate(branch->getSuccessors())) {
        if (successor != argument.getOwner())
          continue;
        SuccessorOperands operands = branch.getSuccessorOperands(index);
        unsigned arg = argument.getArgNumber();
        unsigned produced = operands.getProducedOperandCount();
        ValueRange forwarded = operands.getForwardedOperands();
        if (arg < produced || arg - produced >= forwarded.size() ||
            !recurse(forwarded[arg - produced]))
          return finish(false);
        sawIncoming = true;
      }
    }
    return finish(sawIncoming);
  }
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return finish(false);
  if (auto assembly = dyn_cast<LLVM::InlineAsmOp>(producer)) {
    if (!assembly.getAsmString().empty())
      return finish(false);
    SmallVector<StringRef> constraints;
    assembly.getConstraints().split(constraints, ',');
    unsigned groups = 1;
    if (auto tuple = dyn_cast<LLVM::LLVMStructType>(value.getType()))
      groups = tuple.getBody().size();
    if (constraints.size() != 2 * groups)
      return finish(false);
    for (unsigned index = 0; index < groups; ++index)
      if (constraints[index] != outputConstraint ||
          constraints[groups + index] != std::to_string(index))
        return finish(false);
    if (accumulator && !llvm::any_of(state.resultPins, [&](auto pin) {
          return pin.first == producer;
        }))
      return finish(false);
    sawPin = true;
    return finish(true);
  }
  if (auto extract = dyn_cast<LLVM::ExtractValueOp>(producer)) {
    Value container = extract.getContainer();
    while (auto insert = container.getDefiningOp<LLVM::InsertValueOp>()) {
      if (insert.getPosition() == extract.getPosition())
        return finish(recurse(insert.getValue()));
      container = insert.getContainer();
    }
    return finish(recurse(container));
  }
  if (auto extract = dyn_cast<LLVM::ExtractElementOp>(producer))
    return finish(recurse(extract.getVector()));
  if (auto insert = dyn_cast<LLVM::InsertValueOp>(producer))
    return finish(recurse(insert.getContainer()) && recurse(insert.getValue()));
  if (auto insert = dyn_cast<LLVM::InsertElementOp>(producer))
    return finish(recurse(insert->getOperand(0)) &&
                  recurse(insert->getOperand(1)));
  if (isa<LLVM::BitcastOp, UnrealizedConversionCastOp>(producer))
    return finish(producer->getNumOperands() == 1 &&
                  recurse(producer->getOperand(0)));
  return finish(isa<LLVM::ConstantOp, LLVM::UndefOp>(producer));
}

static bool canAnchorAdjacentCommit(
    triton::amdgpu::MfmaCommitOp first, triton::amdgpu::MfmaCommitOp second,
    ValueRange firstInputs, ValueRange secondInputs,
    const AMD::ScheduledMfmaLoweringState &state,
    ConversionPatternRewriter &rewriter, const AMD::TargetInfo &targetInfo) {
  if (targetInfo.getArch() != "gfx950" || !second ||
      first->getNextNode() != second.getOperation())
    return false;
  int firstWait = 0;
  int secondWait = 0;
  bool secondHasAccumulator = false;
  bool secondHasOperand = false;
  DominanceInfo dominance(first->getParentOp());
  auto prove = [&](Value source, Value converted, bool isSecond) {
    auto tensor = cast<RankedTensorType>(source.getType());
    bool accumulator = tensor.getElementType().isF32();
    auto mfma =
        dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(tensor.getEncoding());
    if (accumulator) {
      if (!mfma || mfma.getVersion() != 4)
        return false;
      FailureOr<int> wait = getMfmaDrainWaitStates(targetInfo.getISAFamily(),
                                                   mfma.getInstrShape());
      if (failed(wait))
        return false;
      int &required = isSecond ? secondWait : firstWait;
      required = std::max(required, *wait);
      secondHasAccumulator |= isSecond;
    } else {
      auto dot =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(tensor.getEncoding());
      auto parent =
          dot ? dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(dot.getParent())
              : triton::gpu::AMDMfmaEncodingAttr();
      if (!isSecond || !parent || parent.getVersion() != 4)
        return false;
      secondHasOperand = true;
    }
    if (!dominance.dominates(converted, first.getOperation()))
      return false;
    llvm::DenseSet<Value> active;
    bool sawPin = false;
    StringRef constraint = accumulator && isSecond ? "=v" : "=a";
    return hasPinnedCommitInput(converted, constraint, state, rewriter,
                                accumulator, active, sawPin) &&
           sawPin;
  };
  for (auto [source, converted] : llvm::zip(first.getInputs(), firstInputs))
    if (!prove(source, converted, false))
      return false;
  for (auto [source, converted] : llvm::zip(second.getInputs(), secondInputs))
    if (!prove(source, converted, true))
      return false;
  return secondHasAccumulator && secondHasOperand && firstWait >= secondWait;
}

// Recognize only a compiler-owned boundary whose complete input tuple is the
// register-class-identical suffix returned by an earlier combined boundary.
// This is an SSA dependency through the full delay, not an adjacency fact.
static LLVM::InlineAsmOp
getAnchoringCommit(LLVM::InlineAsmOp second,
                   const DenseMap<Operation *, int> &commits) {
  LLVM::InlineAsmOp first;
  unsigned begin = 0;
  for (auto [index, input] : llvm::enumerate(second.getOperands())) {
    auto extract = input.getDefiningOp<LLVM::ExtractValueOp>();
    if (!extract || extract.getPosition().size() != 1)
      return nullptr;
    auto producer = extract.getContainer().getDefiningOp<LLVM::InlineAsmOp>();
    if (!producer || !commits.contains(producer))
      return nullptr;
    if (!first) {
      first = producer;
      begin = extract.getPosition().front();
    }
    if (producer != first || extract.getPosition().front() != begin + index)
      return nullptr;
  }
  if (!first || begin == 0 || first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second) ||
      begin + second.getNumOperands() != first.getNumOperands())
    return nullptr;
  for (Operation *op = first->getNextNode(); op != second.getOperation();
       op = op->getNextNode())
    if (!isa<LLVM::ExtractValueOp>(op))
      return nullptr;
  SmallVector<StringRef> firstConstraints;
  SmallVector<StringRef> secondConstraints;
  first.getConstraints().split(firstConstraints, ',');
  second.getConstraints().split(secondConstraints, ',');
  auto isTiedCommit = [](ArrayRef<StringRef> constraints, unsigned count) {
    if (constraints.size() != 2 * count + 1 ||
        constraints.back() != "~{memory}")
      return false;
    for (unsigned index = 0; index < count; ++index)
      if (constraints[count + index] != std::to_string(index))
        return false;
    return true;
  };
  if (!isTiedCommit(firstConstraints, first.getNumOperands()) ||
      !isTiedCommit(secondConstraints, second.getNumOperands()))
    return nullptr;
  for (unsigned index = 0; index < begin; ++index)
    if (firstConstraints[index] != "=a")
      return nullptr;
  bool sawVgpr = false;
  bool sawAgpr = false;
  for (unsigned index = 0; index < second.getNumOperands(); ++index) {
    StringRef constraint = secondConstraints[index];
    if ((constraint != "=v" && constraint != "=a") ||
        firstConstraints[begin + index] != constraint)
      return nullptr;
    sawVgpr |= constraint == "=v";
    sawAgpr |= constraint == "=a";
  }
  return sawVgpr && sawAgpr ? first : nullptr;
}

class MfmaCommitOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::MfmaCommitOp> {
public:
  using OpAdaptor = triton::amdgpu::MfmaCommitOp::Adaptor;

  MfmaCommitOpConversion(const LLVMTypeConverter &converter,
                         const AMD::TargetInfo &targetInfo,
                         AMD::ScheduledMfmaLoweringState &scheduledMfmaState,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::amdgpu::MfmaCommitOp>(converter,
                                                             benefit),
        targetInfo(targetInfo), scheduledMfmaState(scheduledMfmaState) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::MfmaCommitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                            targetInfo.getISAFamily()))
      return op.emitOpError(
          "is supported only on CDNA3 (gfx942) and CDNA4 (gfx950)");
    for (Value input : op.getInputs()) {
      auto tensorTy = cast<RankedTensorType>(input.getType());
      Attribute encoding = tensorTy.getEncoding();
      auto mfma = dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(encoding);
      if (auto dot = dyn_cast<triton::gpu::DotOperandEncodingAttr>(encoding))
        mfma = dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(dot.getParent());
      if (mfma && failed(verifyMfmaVersionMatchesTarget(
                      op, mfma, targetInfo.getISAFamily())))
        return failure();
    }
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();
    TritonLLVMOpBuilder b(loc, rewriter);

    // Capture adjacency before replacing either source operation. Anchor the
    // complete second tuple as tied outputs of the first full-delay boundary;
    // adjacency alone cannot stop its pure native MFMA producers from sinking.
    auto second =
        dyn_cast_or_null<triton::amdgpu::MfmaCommitOp>(op->getNextNode());
    DominanceInfo dominance(op->getParentOp());
    bool canRemapSecond =
        targetInfo.getArch() == "gfx950" && second &&
        llvm::all_of(op.getInputs(),
                     [](Value input) {
                       return cast<RankedTensorType>(input.getType())
                           .getElementType()
                           .isF32();
                     }) &&
        llvm::any_of(second.getInputs(),
                     [](Value input) {
                       return !cast<RankedTensorType>(input.getType())
                                   .getElementType()
                                   .isF32();
                     }) &&
        llvm::all_of(second.getInputs(), [&](Value input) {
          return dominance.dominates(input, op.getOperation());
        });
    SmallVector<Value> secondInputs;
    bool shareDrain =
        canRemapSecond &&
        succeeded(
            rewriter.getRemappedValues(second.getInputs(), secondInputs)) &&
        canAnchorAdjacentCommit(op, second, adaptor.getInputs(), secondInputs,
                                scheduledMfmaState, rewriter, targetInfo);
    SmallVector<Value> sources(op.getInputs());
    SmallVector<Value> convertedInputs(adaptor.getInputs());
    unsigned firstInputCount = sources.size();
    if (shareDrain) {
      llvm::append_range(sources, second.getInputs());
      llvm::append_range(convertedInputs, secondInputs);
    }

    auto packGroups =
        [&](Value value, RankedTensorType tensorTy, unsigned registersPerGroup,
            Type &elementVectorTy,
            Type &registerVectorTy) -> FailureOr<SmallVector<Value>> {
      Type elemTy = typeConverter->convertType(tensorTy.getElementType());
      unsigned bitWidth = getIntOrFloatOrPtrBitWidth(elemTy);
      if (registersPerGroup == 0 || bitWidth == 0 ||
          (registersPerGroup * 32) % bitWidth != 0)
        return failure();
      unsigned elementsPerGroup = registersPerGroup * 32 / bitWidth;
      if (elementsPerGroup == 0)
        return failure();
      SmallVector<Value> elements =
          unpackTensorElements(loc, value, rewriter, tensorTy);
      if (elements.empty() || elements.size() % elementsPerGroup != 0)
        return failure();

      elementVectorTy = vec_ty(elemTy, elementsPerGroup);
      registerVectorTy = vec_ty(i32_ty, registersPerGroup);
      SmallVector<Value> groups;
      for (unsigned begin = 0; begin < elements.size();
           begin += elementsPerGroup) {
        Value elementVector = b.undef(elementVectorTy);
        for (unsigned index = 0; index < elementsPerGroup; ++index)
          elementVector =
              b.insert_element(elementVectorTy, elementVector,
                               elements[begin + index], b.i32_val(index));
        groups.push_back(b.bitcast(elementVector, registerVectorTy));
      }
      return groups;
    };

    SmallVector<Type> elementVectorTypes;
    SmallVector<SmallVector<Value>> inputGroups;
    SmallVector<size_t> firstGroupIndices;
    SmallVector<Value> operands;
    SmallVector<Type> outputTypes;
    std::string constraints;
    constexpr unsigned warpSize = 64;
    bool hasLiveDependency = llvm::any_of(op.getInputs(), [](Value input) {
      return !cast<RankedTensorType>(input.getType()).getElementType().isF32();
    });
    size_t agprInputIndex = 0;
    if (hasLiveDependency && findAgprResidentAccumulator(op, agprInputIndex))
      return op.emitOpError()
             << "input " << agprInputIndex
             << " is an AGPR-resident accumulator committed alongside a live "
                "dot operand. The AGPR read is materialized ahead of this "
                "boundary's hazard padding; pin the accumulator with "
                "accumulator_register_class=\"vgpr\"";
    for (auto [inputIndex, source] : llvm::enumerate(sources)) {
      Value converted = convertedInputs[inputIndex];
      auto tensorTy = cast<RankedTensorType>(source.getType());
      Type elementVectorTy;
      Type registerVectorTy;
      unsigned registersPerGroup = 0;
      StringRef outputConstraint;

      if (tensorTy.getElementType().isF32()) {
        auto mfma =
            cast<triton::gpu::AMDMfmaEncodingAttr>(tensorTy.getEncoding());
        ArrayRef<unsigned> instr = mfma.getInstrShape();
        registersPerGroup = instr[0] * instr[1] / warpSize;
        outputConstraint =
            hasLiveDependency || inputIndex >= firstInputCount ? "=v" : "=a";
      } else {
        auto dot =
            cast<triton::gpu::DotOperandEncodingAttr>(tensorTy.getEncoding());
        auto mfma = cast<triton::gpu::AMDMfmaEncodingAttr>(dot.getParent());
        ArrayRef<unsigned> instr = mfma.getInstrShape();
        unsigned fragmentElements =
            dot.getOpIdx() == 0 ? instr[0] * instr[2] : instr[2] * instr[1];
        unsigned bitWidth = getIntOrFloatOrPtrBitWidth(
            typeConverter->convertType(tensorTy.getElementType()));
        if (fragmentElements == 0 || fragmentElements % warpSize != 0) {
          return rewriter.notifyMatchFailure(
              op, "native dot fragment has a fractional per-lane width");
        }
        unsigned fragmentBitWidth = fragmentElements / warpSize * bitWidth;
        if (fragmentBitWidth == 0 || fragmentBitWidth % 32 != 0) {
          return rewriter.notifyMatchFailure(
              op, "native dot fragment does not fill complete registers");
        }
        registersPerGroup = fragmentBitWidth / 32;
        outputConstraint = "=a";
      }

      FailureOr<SmallVector<Value>> maybeGroups =
          packGroups(converted, tensorTy, registersPerGroup, elementVectorTy,
                     registerVectorTy);
      if (failed(maybeGroups))
        return rewriter.notifyMatchFailure(
            op, "native fragments do not divide an input");

      elementVectorTypes.push_back(elementVectorTy);
      firstGroupIndices.push_back(operands.size());
      inputGroups.push_back(std::move(*maybeGroups));
      for (Value group : inputGroups.back()) {
        if (!constraints.empty())
          constraints += ",";
        constraints += outputConstraint;
        operands.push_back(group);
        outputTypes.push_back(registerVectorTy);
      }
    }
    for (size_t index = 0; index < outputTypes.size(); ++index)
      constraints += "," + std::to_string(index);
    constraints += ",~{memory}";

    // gfx950 uses the established six-state transient handoff when this
    // boundary also carries a live dot operand. CDNA3 still needs its full,
    // layout-specific result-read delay. A persistent epilogue likewise uses
    // the largest target-specific delay carried by the boundary.
    bool useGfx950LiveDependencyHandoff =
        targetInfo.getISAFamily() == ISAFamily::CDNA4 && hasLiveDependency;
    int waitStates = 6;
    if (!useGfx950LiveDependencyHandoff) {
      waitStates = 0;
      for (Value input : op.getInputs()) {
        auto tensorTy = cast<RankedTensorType>(input.getType());
        if (!tensorTy.getElementType().isF32())
          continue;
        auto mfma =
            cast<triton::gpu::AMDMfmaEncodingAttr>(tensorTy.getEncoding());
        FailureOr<int> drainWaitStates = getMfmaDrainWaitStates(
            targetInfo.getISAFamily(), mfma.getInstrShape());
        if (failed(drainWaitStates))
          return rewriter.notifyMatchFailure(
              op, "commit boundary carries an MFMA layout with no modeled "
                  "result-read hazard requirement");
        waitStates = std::max(waitStates, *drainWaitStates);
      }
    }
    std::string waitAsm = mfmaWaitStateAsm(waitStates);

    Type resultTy = outputTypes.front();
    if (outputTypes.size() != 1)
      resultTy = LLVM::LLVMStructType::getLiteral(ctx, outputTypes);
    auto asmDialect = LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
    auto operandAttrs = ArrayAttr::get(ctx, {});
    auto inlineAsm = LLVM::InlineAsmOp::create(
        rewriter, loc, resultTy, operands, waitAsm, constraints,
        /*has_side_effects=*/true,
        /*is_align_stack=*/false, LLVM::TailCallKind::None, asmDialect,
        operandAttrs);
    Value constrained = inlineAsm->getResult(0);

    auto getConstrainedGroup = [&](size_t index) {
      return outputTypes.size() == 1
                 ? constrained
                 : b.extract_val(outputTypes[index], constrained, index);
    };

    // Retain the second boundary's original register classes, tuple, memory
    // clobber, and side effects. Its operands are exclusively values returned
    // after the first delay, so no copy can read an uncompleted original dV.
    SmallVector<Value> secondGroups;
    Value secondConstrained;
    size_t secondFirstGroup =
        shareDrain ? firstGroupIndices[firstInputCount] : 0;
    if (shareDrain) {
      SmallVector<Type> secondOutputTypes;
      std::string secondConstraints;
      for (size_t input = firstInputCount; input < sources.size(); ++input) {
        StringRef constraint = cast<RankedTensorType>(sources[input].getType())
                                       .getElementType()
                                       .isF32()
                                   ? "=v"
                                   : "=a";
        for (size_t group = 0; group < inputGroups[input].size(); ++group) {
          size_t index = firstGroupIndices[input] + group;
          if (!secondConstraints.empty())
            secondConstraints += ",";
          secondConstraints += constraint;
          secondGroups.push_back(getConstrainedGroup(index));
          secondOutputTypes.push_back(outputTypes[index]);
        }
      }
      for (size_t index = 0; index < secondGroups.size(); ++index)
        secondConstraints += "," + std::to_string(index);
      secondConstraints += ",~{memory}";
      Type secondResultTy = secondOutputTypes.front();
      if (secondOutputTypes.size() != 1)
        secondResultTy =
            LLVM::LLVMStructType::getLiteral(ctx, secondOutputTypes);
      auto secondAsm = LLVM::InlineAsmOp::create(
          rewriter, second.getLoc(), secondResultTy, secondGroups, waitAsm,
          secondConstraints, /*has_side_effects=*/true,
          /*is_align_stack=*/false, LLVM::TailCallKind::None, asmDialect,
          operandAttrs);
      secondConstrained = secondAsm.getRes();
      // Keep this boundary recognizable until every reaching root has been
      // analyzed and both delays have been strengthened by the finalizer.
      scheduledMfmaState.commitWaitStates[secondAsm] = waitStates;
    }

    SmallVector<Value> results;
    for (size_t inputIndex = 0; inputIndex < inputGroups.size(); ++inputIndex) {
      SmallVector<Value> elements;
      Type elementVectorTy = elementVectorTypes[inputIndex];
      auto vectorTy = cast<VectorType>(elementVectorTy);
      for (size_t group = 0; group < inputGroups[inputIndex].size(); ++group) {
        size_t groupIndex = firstGroupIndices[inputIndex] + group;
        Value registerGroup;
        if (shareDrain && inputIndex >= firstInputCount) {
          size_t index = groupIndex - secondFirstGroup;
          registerGroup = secondGroups.size() == 1
                              ? secondConstrained
                              : b.extract_val(outputTypes[groupIndex],
                                              secondConstrained, index);
        } else {
          registerGroup = getConstrainedGroup(groupIndex);
        }
        Value elementGroup = b.bitcast(registerGroup, elementVectorTy);
        for (int64_t index = 0; index < vectorTy.getNumElements(); ++index)
          elements.push_back(b.extract_element(vectorTy.getElementType(),
                                               elementGroup, b.i32_val(index)));
      }
      results.push_back(packTensorElements(
          loc, typeConverter, elements, rewriter,
          cast<RankedTensorType>(sources[inputIndex].getType())));
    }
    if (shareDrain)
      rewriter.replaceOp(second,
                         ValueRange(results).drop_front(firstInputCount));
    rewriter.replaceOp(op, ValueRange(results).take_front(firstInputCount));
    scheduledMfmaState.commitWaitStates[inlineAsm] = waitStates;
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
  AMD::ScheduledMfmaLoweringState &scheduledMfmaState;
};

class ScheduledMfmaOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ScheduledMfmaOp> {
public:
  using OpAdaptor = triton::amdgpu::ScheduledMfmaOp::Adaptor;

  ScheduledMfmaOpConversion(const LLVMTypeConverter &converter,
                            const AMD::TargetInfo &targetInfo,
                            AMD::ScheduledMfmaLoweringState &scheduledMfmaState,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::amdgpu::ScheduledMfmaOp>(converter,
                                                                benefit),
        targetInfo(targetInfo), scheduledMfmaState(scheduledMfmaState) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::ScheduledMfmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    auto aTy = cast<RankedTensorType>(op.getA().getType());
    auto bTy = cast<RankedTensorType>(op.getB().getType());
    auto accTy = cast<RankedTensorType>(op.getAcc().getType());
    auto mfma = cast<triton::gpu::AMDMfmaEncodingAttr>(accTy.getEncoding());
    if (failed(verifyMfmaVersionMatchesTarget(op, mfma,
                                              targetInfo.getISAFamily())))
      return failure();
    ArrayRef<unsigned> instrShape = mfma.getInstrShape();
    FailureOr<ScheduledMfmaLoweringInfo> maybeInfo =
        getScheduledMfmaLoweringInfo(loc, mfma, aTy.getElementType(),
                                     bTy.getElementType());
    if (failed(maybeInfo))
      return op.emitOpError(
          "has no supported native lowering for this target, element type, "
          "and instruction shape");
    const ScheduledMfmaLoweringInfo &info = *maybeInfo;
    auto aDot = cast<triton::gpu::DotOperandEncodingAttr>(aTy.getEncoding());
    auto bDot = cast<triton::gpu::DotOperandEncodingAttr>(bTy.getEncoding());

    SmallVector<int64_t> aRep =
        mfma.getRepForOperand(aTy.getShape(), aDot.getKWidth(), 0);
    SmallVector<int64_t> bRep =
        mfma.getRepForOperand(bTy.getShape(), bDot.getKWidth(), 1);
    FailureOr<SmallVector<Value>> maybeA =
        packMfmaDotOperandFragments(adaptor.getA(), aTy, /*opIdx=*/0, aRep,
                                    info.kBase, typeConverter, rewriter, loc);
    FailureOr<SmallVector<Value>> maybeB =
        packMfmaDotOperandFragments(adaptor.getB(), bTy, /*opIdx=*/1, bRep,
                                    info.kBase, typeConverter, rewriter, loc);
    int64_t numRepM = aRep[1];
    int64_t numRepN = bRep[2];
    int64_t numRepK = aRep[2] * aDot.getKWidth() / info.kBase;
    int64_t numRepKB = bRep[1] * bDot.getKWidth() / info.kBase;
    if (failed(maybeA) || failed(maybeB) || numRepK <= 0 ||
        numRepK != numRepKB ||
        maybeA->size() != static_cast<size_t>(numRepM * numRepK) ||
        maybeB->size() != static_cast<size_t>(numRepN * numRepK))
      return rewriter.notifyMatchFailure(
          op, "operands do not match the verified native MFMA grid");

    constexpr int64_t warpSize = 64;
    int64_t elemsPerFragment = instrShape[0] * instrShape[1] / warpSize;
    SmallVector<int64_t> strides =
        computeStrides({1, numRepM, numRepN, elemsPerFragment});
    SmallVector<Value> elements =
        unpackTensorElements(loc, adaptor.getAcc(), rewriter, accTy);
    if (elements.size() !=
        static_cast<size_t>(numRepM * numRepN * elemsPerFragment))
      return rewriter.notifyMatchFailure(
          op, "accumulator element count does not match its MFMA grid");

    Type accElemTy = typeConverter->convertType(accTy.getElementType());
    auto fragmentTy = vec_ty(accElemTy, elemsPerFragment);
    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> accumulatorFragments;
    accumulatorFragments.reserve(numRepM * numRepN);
    for (int64_t m = 0; m < numRepM; ++m) {
      for (int64_t n = 0; n < numRepN; ++n) {
        Value fragment = b.undef(fragmentTy);
        for (int64_t index = 0; index < elemsPerFragment; ++index) {
          int64_t linearIndex = linearize({0, m, n, index}, strides);
          fragment = b.insert_element(fragmentTy, fragment,
                                      elements[linearIndex], b.i32_val(index));
        }
        accumulatorFragments.push_back(fragment);
      }
    }

    StringRef aStorage = op.getResidentOperand() == "lhs" ? "agpr" : "vgpr";
    StringRef bStorage = op.getResidentOperand() == "rhs" ? "agpr" : "vgpr";
    StringRef accumulatorStorage = resolveAccumulatorStorage(op);

    Value zeroFragment;
    if (op.getInitialize()) {
      Attribute zeroAttr = rewriter.getZeroAttr(accElemTy);
      auto zeroElements =
          DenseElementsAttr::get(cast<ShapedType>(fragmentTy), zeroAttr);
      zeroFragment =
          LLVM::ConstantOp::create(rewriter, loc, fragmentTy, zeroElements);
    }
    bool isPersistent = op.getAccumulatorRole() == "persistent";
    bool orderOperandPins =
        scheduledMfmaState.operandOrderEligibleOps.contains(op.getOperation());
    int drainWaitStates = 0;
    if (isPersistent) {
      FailureOr<int> requiredWait =
          getMfmaDrainWaitStates(targetInfo.getISAFamily(), instrShape);
      if (failed(requiredWait))
        return rewriter.notifyMatchFailure(
            op, "persistent MFMA layout has no modeled result-read hazard "
                "requirement");
      drainWaitStates = *requiredWait;
    }
    bool pinAccumulatorInput = isPersistent && !op.getInitialize() &&
                               !matchPattern(op.getAcc(), m_Constant());

    SmallVector<Value> updatedFragments = accumulatorFragments;
    if (pinAccumulatorInput) {
      for (Value &fragment : updatedFragments) {
        FailureOr<Value> constrainedC = constrainMfmaFragmentRegisterClass(
            fragment, accumulatorStorage, rewriter, loc,
            /*hasSideEffects=*/true);
        if (failed(constrainedC))
          return rewriter.notifyMatchFailure(
              op, "native MFMA accumulator must pack into complete 32-bit "
                  "registers");
        fragment = *constrainedC;
      }
    }

    // Keep one SSA chain per output fragment and retain K/N/M source order.
    // Native intrinsics expose every chain to AMDGPU's hazard recognizer and
    // scheduler. Empty C/D pins constrain persistent storage at the boundaries
    // of the grid; the arithmetic itself stays visible to LLVM.
    for (int64_t k = 0; k < numRepK; ++k) {
      for (int64_t n = 0; n < numRepN; ++n) {
        for (int64_t m = 0; m < numRepM; ++m) {
          int64_t accumulatorIndex = m * numRepN + n;
          Value current = updatedFragments[accumulatorIndex];
          Value operandA = (*maybeA)[m * numRepK + k];
          Value operandB = (*maybeB)[n * numRepK + k];
          if (isPersistent) {
            LLVM::InlineAsmOp pinA, pinB;
            FailureOr<Value> constrainedA = constrainMfmaFragmentRegisterClass(
                operandA, aStorage, rewriter, loc, /*hasSideEffects=*/false,
                &pinA);
            FailureOr<Value> constrainedB = constrainMfmaFragmentRegisterClass(
                operandB, bStorage, rewriter, loc, /*hasSideEffects=*/false,
                &pinB);
            if (failed(constrainedA) || failed(constrainedB))
              return rewriter.notifyMatchFailure(
                  op, "native MFMA operands must pack into complete 32-bit "
                      "registers");
            // Share repeated fragments through CSE, then order the surviving
            // pins only when both original operands came entirely from LDS.
            if (orderOperandPins)
              for (LLVM::InlineAsmOp pin : {pinA, pinB})
                pin->setAttr(triton::AMD::kScheduledMfmaOperandPinAttrName,
                             rewriter.getUnitAttr());
            operandA = *constrainedA;
            operandB = *constrainedB;
          }
          if (mfma.getIsTransposed())
            std::swap(operandA, operandB);

          if (info.intrinsicOperandsAreI16) {
            auto packedTy = vec_ty(i16_ty, info.kBase);
            operandA = b.bitcast(operandA, packedTy);
            operandB = b.bitcast(operandB, packedTy);
          }
          OperationState loweredOp(loc, info.intrinsicName);
          loweredOp.addTypes(fragmentTy);
          bool zeroThisInstruction = op.getInitialize() && k == 0;
          Value intrinsicAcc = zeroThisInstruction ? zeroFragment : current;
          loweredOp.addOperands({operandA, operandB, intrinsicAcc});
          loweredOp.addAttribute("cbsz", rewriter.getI32IntegerAttr(0));
          loweredOp.addAttribute("abid", rewriter.getI32IntegerAttr(0));
          // For `blgp`: f64 MFMA uses negation flags, while other MFMA ops
          // use B-lane permutation flags.
          Attribute blgpAttr =
              cast<VectorType>(fragmentTy).getElementType().isF64()
                  ? Attribute(ROCDL::MFMANegModifierAttr::get(
                        rewriter.getContext(), ROCDL::MFMANegModifier::none))
                  : Attribute(ROCDL::MFMAPermBAttr::get(
                        rewriter.getContext(), ROCDL::MFMAPermB::none));
          loweredOp.addAttribute("blgp", blgpAttr);
          updatedFragments[accumulatorIndex] =
              rewriter.create(loweredOp)->getResult(0);
        }
      }
    }

    SmallVector<LLVM::InlineAsmOp> resultPins;
    if (isPersistent) {
      for (Value &fragment : updatedFragments) {
        // Anchor each completed K chain to side-effect ordering while keeping
        // its arithmetic in native MFMAs.
        LLVM::InlineAsmOp pin;
        FailureOr<Value> constrainedD = constrainMfmaFragmentRegisterClass(
            fragment, accumulatorStorage, rewriter, loc,
            /*hasSideEffects=*/true, &pin);
        if (failed(constrainedD))
          return rewriter.notifyMatchFailure(
              op,
              "native MFMA result must pack into complete 32-bit registers");
        fragment = *constrainedD;
        resultPins.push_back(pin);
      }
    }

    for (int64_t m = 0; m < numRepM; ++m) {
      for (int64_t n = 0; n < numRepN; ++n) {
        Value fragment = updatedFragments[m * numRepN + n];
        for (int64_t index = 0; index < elemsPerFragment; ++index) {
          int64_t linearIndex = linearize({0, m, n, index}, strides);
          elements[linearIndex] =
              b.extract_element(accElemTy, fragment, b.i32_val(index));
        }
      }
    }
    Value result = packTensorElements(loc, typeConverter, elements, rewriter,
                                      op.getResult().getType());
    rewriter.replaceOp(op, result);
    for (LLVM::InlineAsmOp pin : resultPins)
      scheduledMfmaState.resultPins.emplace_back(pin, drainWaitStates);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
  AMD::ScheduledMfmaLoweringState &scheduledMfmaState;
};

class BarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
public:
  BarrierOpConversion(const LLVMTypeConverter &converter,
                      const AMD::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::BarrierOp>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename triton::gpu::BarrierOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!mlir::triton::amdgpu::isCDNA(targetInfo.getISAFamily()))
      return failure();
    // Check no other memory addrspaces are selected.
    // TensorRead/Write are allowed but noop.
    auto mask = triton::gpu::AddrSpace::Local |
                triton::gpu::AddrSpace::GlobalRead |
                triton::gpu::AddrSpace::GlobalWrite |
                triton::gpu::AddrSpace::TensorRead |
                triton::gpu::AddrSpace::TensorWrite;
    if ((op.getAddrSpace() & ~mask) != triton::gpu::AddrSpace::None)
      return failure();
    bool localBarrier = op.hasLocal();
    bool globalBarrier = op.hasGlobalRead() || op.hasGlobalWrite();
    if (localBarrier || globalBarrier) {
      StringRef mmraAddrSpace = "";
      if (localBarrier && !globalBarrier)
        mmraAddrSpace = "local";
      else if (!localBarrier && globalBarrier)
        mmraAddrSpace = "global";

      // Local/global barriers use LLVM fences so the AMDGPU memory legalizer
      // selects target-specific waits. Mixed local+global barriers are left
      // untagged so LLVM conservatively synchronizes every relevant space.
      createAMDGPUMemoryFence(rewriter, op->getLoc(),
                              LLVM::AtomicOrdering::release, mmraAddrSpace);
      ROCDL::SBarrierOp::create(rewriter, op->getLoc());
      createAMDGPUMemoryFence(rewriter, op->getLoc(),
                              LLVM::AtomicOrdering::acquire, mmraAddrSpace);
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOpWithNewOp<ROCDL::SBarrierOp>(op);

    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

/// Encodes the waitcnt value for AMDGPU architectures.
///
/// Note: This function duplicates the bitpacking logic from AMDGPU backend
/// (llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.h), as it's not accessible from
/// llvm/include. The logic handles different encoding schemes across
/// various GPU architecture versions (pre-gfx9 to gfx11).
///
/// The waitcnt encoding uses different bit positions for each counter
/// based on the ISA version:
/// - Vmcnt (vector memory counter): tracks pending vector memory operations
/// - Expcnt (export counter): tracks pending export operations
/// - Lgkmcnt (LDS/GDS/scalar memory counter): tracks pending LDS/GDS/scalar
/// memory ops
///
/// Each architecture version has its own bit layout, Vmcnt, Expcnt and Lgkmcnt
/// are decoded as follows:
///     Vmcnt = Waitcnt[3:0]        (pre-gfx9)
///     Vmcnt = Waitcnt[15:14,3:0]  (gfx9,10)
///     Vmcnt = Waitcnt[15:10]      (gfx11)
///     Expcnt = Waitcnt[6:4]       (pre-gfx11)
///     Expcnt = Waitcnt[2:0]       (gfx11)
///     Lgkmcnt = Waitcnt[11:8]     (pre-gfx10)
///     Lgkmcnt = Waitcnt[13:8]     (gfx10)
///     Lgkmcnt = Waitcnt[9:4]      (gfx11)
static FailureOr<unsigned> encodeWaitcnt(llvm::AMDGPU::IsaVersion isaVersion,
                                         unsigned vmcnt, unsigned lgkmcnt) {
  if (isaVersion.Major == 9) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(15u, lgkmcnt);
    unsigned lowBits = vmcnt & 0xF;
    unsigned highBits = (vmcnt >> 4) << 14;
    unsigned otherCnts = (expcnt << 4) | (lgkmcnt << 8);
    return lowBits | highBits | otherCnts;
  }
  if (isaVersion.Major == 10) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(63u, lgkmcnt);
    unsigned lowBits = vmcnt & 0xF;
    unsigned highBits = (vmcnt >> 4) << 14;
    unsigned otherCnts = (expcnt << 4) | (lgkmcnt << 8);
    return lowBits | highBits | otherCnts;
  }
  if (isaVersion.Major == 11) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(63u, lgkmcnt);
    return (vmcnt << 10) | expcnt | (lgkmcnt << 4);
  }
  return failure();
}

struct MemoryCounterWaitOpConversion
    : public ConvertOpToLLVMPattern<amdgpu::MemoryCounterWaitOp> {
  MemoryCounterWaitOpConversion(const LLVMTypeConverter &converter,
                                const AMD::TargetInfo &targetInfo,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(amdgpu::MemoryCounterWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // amdgpu::MemoryCounterWaitOp supports gfx9 onwards
    auto isaVersion = targetInfo.getIsaVersion();

    /// If major version >= gfx12, lower to
    ///   * ROCDL::WaitDscntOp if ds is present
    ///   * ROCDL::WaitLoadcntOp if load is present
    ///   * ROCDL::WaitStorecntOp if store is present
    if (isaVersion.Major >= 12) {
      Location loc = op.getLoc();
      if (std::optional<int> ds = adaptor.getDs())
        ROCDL::WaitDscntOp::create(rewriter, loc, *ds);

      if (std::optional<int> load = adaptor.getLoad())
        ROCDL::WaitLoadcntOp::create(rewriter, loc, *load);

      if (std::optional<int> store = adaptor.getStore())
        ROCDL::WaitStorecntOp::create(rewriter, loc, *store);

      rewriter.eraseOp(op);
      return success();
    }

    /// Otherwise, lower to ROCDL::SWaitcntOp
    auto getVal = [](Attribute attr) -> unsigned {
      if (attr)
        return cast<IntegerAttr>(attr).getInt();

      // This value will be clamped to the maximum value for the target version.
      return 1024;
    };
    unsigned ds = getVal(adaptor.getDsAttr());

    unsigned vmcnt = 1024;
    Attribute load = adaptor.getLoadAttr();
    Attribute store = adaptor.getStoreAttr();
    if (load && store) {
      vmcnt = getVal(load) + getVal(store);
    } else if (load) {
      vmcnt = getVal(load);
    } else if (store) {
      vmcnt = getVal(store);
    }

    FailureOr<unsigned> waitcnt = encodeWaitcnt(isaVersion, vmcnt, ds);
    if (failed(waitcnt))
      return op.emitOpError("unsupported chipset");

    rewriter.replaceOpWithNewOp<ROCDL::SWaitcntOp>(op, *waitcnt);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::AMD::finalizeScheduledMfmaLowering(
    const ScheduledMfmaLoweringState &state) {
  SmallVector<std::pair<Operation *, int>> needsDrain;
  DenseMap<Operation *, std::unique_ptr<MfmaMemoryForwarding>> memoryForwarding;
  DenseMap<Operation *, int> functionWaitStates;
  DenseMap<Operation *, bool> functionNeedsDrain;
  // Shared commits take the largest requirement among their reaching roots.
  // Transient-only handoffs have no persistent result pin and keep their
  // original delay. This deliberately also strengthens persistent commits
  // whose outputs have only native consumers or no consumers.
  DenseMap<Operation *, int> commitWaitStates = state.commitWaitStates;
  auto isNativeInstruction = [](StringRef name) {
    // These lower to native instructions on the supported targets. Exp2 is
    // also emitted as a direct LLVM call by the elementwise conversion.
    return llvm::is_contained(
        {"llvm.amdgcn.perm", "llvm.amdgcn.permlane16.swap",
         "llvm.amdgcn.permlane32.swap", "llvm.amdgcn.readfirstlane",
         "llvm.amdgcn.ds.permute", "llvm.amdgcn.ds.bpermute",
         "llvm.amdgcn.raw.ptr.buffer.atomic.fadd", "llvm.amdgcn.wave.barrier",
         "llvm.amdgcn.exp2.f32", "llvm.exp2.f32"},
        name);
  };
  for (auto [pin, requiredWait] : state.resultPins) {
    auto function = pin->getParentOfType<LLVM::LLVMFuncOp>();
    bool crossesCallBoundary = !function;
    if (function) {
      auto entry = functionNeedsDrain.try_emplace(function);
      bool &needsFunctionDrain = entry.first->second;
      if (entry.second) {
        // Callable helpers must complete before returning. Kernels have no
        // symbol uses and external linkage at this conversion stage.
        needsFunctionDrain = function.getLinkage() != LLVM::Linkage::External ||
                             !SymbolTable::symbolKnownUseEmpty(
                                 function, function->getParentOp());
        function.walk([&](Operation *op) {
          // Coroutine intrinsics can introduce calls or helper functions in
          // LLVM's later coroutine lowering despite having dedicated MLIR ops.
          if (isa<LLVM::InvokeOp>(op) ||
              op->getName().getStringRef().starts_with("llvm.intr.coro."))
            needsFunctionDrain = true;
          if (auto call = dyn_cast<LLVM::CallOp>(op)) {
            auto callee = call.getCallee();
            if (!callee || !isNativeInstruction(*callee))
              needsFunctionDrain = true;
          }
          if (auto call = dyn_cast<LLVM::CallIntrinsicOp>(op)) {
            // Unknown intrinsics may lower to calls (including some amdgcn
            // intrinsics), so keep their boundary covered conservatively.
            if (!isNativeInstruction(call.getIntrin()))
              needsFunctionDrain = true;
          }
        });
      }
      crossesCallBoundary = needsFunctionDrain;
      int &waitStates = functionWaitStates[function];
      waitStates = std::max(waitStates, requiredWait);
    }
    // A call can reuse registers even without consuming the MFMA result. Keep
    // its full destination tuple live until completion on either side of a
    // call/return, including when LLVM moves otherwise independent operations.
    if (crossesCallBoundary ||
        needsOpaqueConsumerDrain(pin, requiredWait, commitWaitStates,
                                 memoryForwarding))
      needsDrain.emplace_back(pin, requiredWait);
  }

  // A dead destination can be reused by an unrelated inline-assembly output
  // while its native MFMA is still in flight. SSA consumer analysis cannot see
  // that physical-register WAW hazard (or a reused source's WAR hazard).
  // Keep completion inside the opaque instruction sequence: pure assembly can
  // move across other operations during LLVM optimization. Function scope also
  // covers backedges and values or partial fragments eliminated later by LLVM.
  // Empty register pins and compiler-owned commits emit no register writes.
  // The latter retain their separately modeled completion contract.
  for (auto [function, waitStates] : functionWaitStates) {
    std::string completion = mfmaWaitStateAsm(waitStates) + "\n";
    function->walk([&](LLVM::InlineAsmOp assembly) {
      if (assembly.getAsmString().empty() ||
          commitWaitStates.contains(assembly))
        return;
      assembly.setAsmString(completion + assembly.getAsmString().str());
    });
  }

  // Analyze every root before changing result pins: otherwise an upgraded pin
  // could appear to be an opaque consumer of a different root. Commits remain
  // recognizable by their map entries while their delays are strengthened.
  for (auto [pin, requiredWait] : needsDrain)
    cast<LLVM::InlineAsmOp>(pin).setAsmString(mfmaWaitStateAsm(requiredWait));

  // Consumer analysis and opaque/call guards are unchanged. Only now can an
  // explicitly anchored second boundary share the finalized first delay.
  for (auto [op, waitStates] : commitWaitStates) {
    auto second = cast<LLVM::InlineAsmOp>(op);
    auto first = getAnchoringCommit(second, commitWaitStates);
    if (first && commitWaitStates.lookup(first) >= waitStates)
      second.setAsmString("");
  }
}

void mlir::triton::AMD::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit,
    std::shared_ptr<DistributedCoordinateGroups> coordinateGroups,
    ScheduledMfmaLoweringState &scheduledMfmaState) {
  PatternBenefit transBenefit = PatternBenefit(benefit.getBenefit() + 1);
  PatternBenefit barrierBenefit = PatternBenefit(benefit.getBenefit() + 1);

  patterns.add<TransLocalLoadOpConversion<triton::gpu::LocalLoadOp>>(
      typeConverter, targetInfo, transBenefit, coordinateGroups);
  patterns.add<
      TransLocalLoadOpConversion<triton::amdgpu::LocalLoadPackedTransposedOp>>(
      typeConverter, targetInfo, benefit, coordinateGroups);
  patterns.add<LocalAtomicScatterRMWOpConversion>(typeConverter, targetInfo,
                                                  benefit.getBenefit() + 1);
  patterns.add<RematerializedRangeOpConversion>(typeConverter, targetInfo,
                                                transBenefit);
  patterns.add<RegisterResidentOpConversion, RegisterClassAnchorOpConversion>(
      typeConverter, transBenefit);
  patterns.add<MfmaCommitOpConversion, ScheduledMfmaOpConversion>(
      typeConverter, targetInfo, scheduledMfmaState, transBenefit);
  patterns.add<BarrierOpConversion, MemoryCounterWaitOpConversion>(
      typeConverter, targetInfo, barrierBenefit);
}
