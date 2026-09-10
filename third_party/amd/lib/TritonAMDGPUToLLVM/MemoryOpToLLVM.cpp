#include "AsyncUtility.h"
#include "AtomicRMWOpsEmitter.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

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

class TransLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  TransLocalLoadOpConversion(
      const LLVMTypeConverter &converter, const AMD::TargetInfo &targetInfo,
      PatternBenefit benefit,
      std::shared_ptr<DistributedCoordinateGroups> coordinateGroups)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo), coordinateGroups(std::move(coordinateGroups)) {}
  using OpAdaptor = typename triton::gpu::LocalLoadOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();

    auto typeConverter = this->getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    unsigned bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    // FP4 is represented as i8 and, when packed along K, can be
    // transposed using ds_read_tr8 which doesn't change packing.
    if (bitWidth != 16 && bitWidth != 8) {
      return failure();
    }
    auto ldsParamsVec = targetInfo.queryLDSTransLoadParams(bitWidth);
    if (ldsParamsVec.empty())
      return failure();
    if (SharedMemoryObject::getMaskSpanOffsetsAndBlocks(srcTy).second != 0)
      return failure();

    LinearLayout sharedLL;
    if (triton::gpu::isPaddedEncoding(srcTy.getEncoding())) {
      sharedLL = triton::gpu::paddedLinearLayout(srcTy);
    } else {
      sharedLL = triton::gpu::toLinearLayout(srcTy);
    }
    LinearLayout cvtDstLL =
        triton::gpu::toLinearLayout(dstTy).invertAndCompose(sharedLL);
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
              ldsParams.tileSize) {
        continue;
      }

      llvm::SmallVector<Value> values;
      auto result =
          lowerDsReadTr(op, ldsParams, loc, cvtDstLL, values, smemBases,
                        affineOffset, maskSpanAffineOffset, paddingShifts,
                        llvmElemTy, rewriter, targetInfo);
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
  LogicalResult
  lowerDsReadTr(triton::gpu::LocalLoadOp op,
                ::triton::AMD::TargetInfo::LDSTransLoadParams ldsParams,
                Location loc, LinearLayout cvt, SmallVector<Value> &vals,
                ArrayRef<Value> smemBases, Value affineOffset,
                uint64_t maskSpanAffineOffset,
                ArrayRef<std::pair<unsigned, unsigned>> paddingShifts,
                Type llvmElemTy, ConversionPatternRewriter &rewriter,
                const ::triton::AMD::TargetInfo &targetInfo) const {

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

    // Map onto offsets (contiguous part) and addr (non-contiguous part)
    LinearLayout fullTile;
    // Contiguous tile
    LinearLayout tile;
    // ds_read_tr*_b64 performs a cooperative transposed load across 16
    // threads. The instruction processes an Nx16 tile (N=4 for 16-bit, N=8 for
    // 8-bit). The loaded tile is re-packed/transposed where lane i will
    // receive the i-th column.
    //
    // Loaded tile layout (input):     Register layout (output after transpose):
    //     K0  K1  ... K15               R0  R1  R2  R3
    // M0[ ............... ]    =>  T0 [ .   .   .   . ]
    // M1[ ............... ]        T1 [ .   .   .   . ]
    // M2[ ............... ]        ...
    // M3[ ............... ]        T15[ .   .   .   . ]
    //
    // Each lane loads 64 contiguous bits from LDS. After the transpose,
    // lane i receives column i from the input (elements strided by 16
    // the loaded tile).
    //
    // For example with N=4 (16-bit):
    // - Lane 0 receives elements from column 0: originally at [t0,t4,t8,t12]
    // - Lane 1 receives elements from column 1: originally at [t0,t4,t8,t12]
    //   These are the second 16 bits loaded by the same lanes before repacking
    // - Lane 4 receives elements from column 4: originally at [t1,t5,t9,t13]
    //
    // Note that there is no restriction on where elements are loaded
    // from, only that each lane needs to load 64 contiguous bits from shared
    // memory. We require N number of lanes to be contiguous since they read
    // consecutive 64 bits loaded from the same lanes.
    tile = LinearLayout::identity1D(ldsParams.tileSize, kLane, kOffset);
    const auto isaFamily = targetInfo.getISAFamily();
    const unsigned missingLanes =
        targetInfo.getWarpSize() / tile.getInDimSize(kLane);
    unsigned otherLanes = 1;
    if (isaFamily == ISAFamily::CDNA4) {
      otherLanes = (bitWidth == 8) ? 2 : 4;
    } else if (ldsParams.tileKind ==
               AMD::TargetInfo::TileKind::DoubleContiguity) {
      otherLanes = 2;
    }

    switch (ldsParams.tileKind) {
    case AMD::TargetInfo::TileKind::DoubleContiguity:
      fullTile =
          tile * LinearLayout::identity1D(ldsParams.tileSize / 2, kReg, kAddr) *
          LinearLayout::identity1D(otherLanes, kLane, kAddr) *
          LinearLayout::identity1D(2, kReg, kAddr) *
          LinearLayout::identity1D(missingLanes / otherLanes, kLane, kAddr);
      break;
    case AMD::TargetInfo::TileKind::Standard:
      fullTile =
          tile * LinearLayout::identity1D(otherLanes, kLane, kAddr) *
          LinearLayout::identity1D(ldsParams.tileSize, kReg, kAddr) *
          LinearLayout::identity1D(missingLanes / otherLanes, kLane, kAddr);
      break;
    }
    // Add warp dimension so we can invert and compose with reps later
    fullTile *= LinearLayout::identity1D(1, kWarp, kAddr);

    if (cvtLayout.getInDimSize(kReg) < fullTile.getInDimSize(kReg))
      return failure();

    auto maybeQuot = divideLeft(cvtLayout, tile);
    if (!maybeQuot.has_value())
      return failure();

    // From here on we perform the lowering
    auto reps = zerosLike(tile) * maybeQuot.value();

    // Sanity check
    assert(fullTile.getInDimSize(kReg) * bitWidth == ldsParams.instBitWidth);

    // If we are lowering a subslice, the subslice offsets shall not touch the
    // contiguous part of the tile
    if (maskSpanAffineOffset & (tile.getOutDimSize(kOffset) - 1))
      return failure();

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

    // Matrix accesses are CTA-local. Model that with a trivial block output so
    // additive stride analysis always compares (offset, block) components.
    reps =
        reps.reshapeOuts({{kOffset, reps.getOutDimSize(kOffset)}, {kBlock, 1}});
    addrLayout = addrLayout.reshapeOuts(reps.getOutDims());

    // Compute the bits that are moved by one instruction
    // Compute elements for which we can swap the xor by an add
    auto [nAdditive, permStrides] = actionAdditiveStrides(
        reps, addrLayout, maskSpanAffineOffset, /*maskSpanBlocks=*/0,
        fullTile.getInDimSize(kReg));
    reps = permStrides.apply(reps);
    if (isPartitioned) {
      partitionLayout = permStrides.apply(partitionLayout);

      // One ds_read_tr* instruction produces `fullTile.getInDimSize(kReg)`
      // consecutive register values from a single LDS base pointer. We only
      // select a partition once per instruction, so all of those register
      // positions must map to the same partition. For a LinearLayout that holds
      // iff the low log2(elemsPerInstr) register bases contribute 0 to
      // kPartition. Bail out if not, so a generic lowering can take over.
      const unsigned numInstrRegBits =
          llvm::Log2_32(fullTile.getInDimSize(kReg));
      for (unsigned pos = 0; pos < numInstrRegBits; ++pos) {
        if (partitionLayout.getBasis(kReg, pos, kPartition) != 0)
          return failure();
      }

      // partitionLayout's kLane is the destination lane which is the lane that
      // owns the loaded data in the destination tensor. The laneId is the
      // source lane issuing the load. For ds_read_tr* the hardware shuffles
      // data across lanes, so the two differ: we need to remap.
      //
      // Example: ds_load_tr8_b64 on gfx1250 (DoubleContiguity), from the test
      // `ds_transpose_partitioned_uses_double_contiguity`.
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
      auto srcToDstLaneMap = fullTile.invert()
                                 .sublayout({kAddr}, {kLane})
                                 .renameInDim(kAddr, kLane);
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
      std::tie(laneId, warpId) = coordinateGroups->getOrCreate(
          op, group.getInt(), rematerializeLane, rematerializeWarp, rewriter,
          targetInfo);
    } else if (op->hasAttr("tlx.rematerialize_coordinates")) {
      if (rematerializeLane)
        laneId = targetInfo.rematerializeDistributedCoordinate(rewriter, loc,
                                                               laneId);
      if (rematerializeWarp)
        warpId = targetInfo.rematerializeDistributedCoordinate(rewriter, loc,
                                                               warpId);
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

    // Elements per op
    auto elemsPerInstr = fullTile.getInDimSize(kReg);
    auto elemsPerVec = ldsParams.instBitWidth / bitWidth;
    auto vecTy = vec_ty(llvmElemTy, elemsPerVec);
    for (int i = 0; i < cvtLayout.getInDimSize(kReg); i += nAdditive) {
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
                                        rewriter, targetInfo));
      }
    }
    // apply all the inverse permutations in the reverse order
    assert(vals.size() == cvtLayout.getInDimSize(kReg));
    vals = permStrides.inverse().apply(vals);

    return success();
  }

  // Emits a single ds_read_tr* operation at `vecAddr` and unpacks the loaded
  // vector into individual element Values. Returns an empty vector if the ISA
  // family does not support a ds_read_tr* instruction.
  SmallVector<Value>
  emitDsReadTr(triton::gpu::LocalLoadOp op, Location loc, Value vecAddr,
               VectorType vTy, Type llvmElemTy,
               ConversionPatternRewriter &rewriter,
               const ::triton::AMD::TargetInfo &targetInfo) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    const auto bitWidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
    assert(bitWidth == 16 || bitWidth == 8);

    Value dsReadTr = createDsReadTr(op, rewriter, loc, vecAddr, vTy,
                                    targetInfo.getISAFamily(), bitWidth);
    if (!dsReadTr)
      return {};

    Value vecVal = b.bitcast(dsReadTr, vTy);
    SmallVector<Value> loadedVals;
    for (int v = 0; v < vTy.getNumElements(); v++)
      loadedVals.push_back(b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
    return loadedVals;
  }

  // Creates and returns the result Value of a single ds_read_tr* op for the
  // given (isaFamily, bitWidth).
  static Value createDsReadTr(triton::gpu::LocalLoadOp op,
                              RewriterBase &rewriter, Location loc,
                              Value vecAddr, VectorType vTy,
                              ISAFamily isaFamily, unsigned bitWidth) {
    // tr16 instructions return vectors of bf16/f16 while "tr8" instructions
    // return vectors of i32. Generate the corresponding i32 vector type.
    const auto numElemsI32 = (vTy.getNumElements() * bitWidth / 32);
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
      if (bitWidth == 16)
        return callIntrinsic("llvm.amdgcn.ds.load.tr16.b128", vTy);
      return callIntrinsic("llvm.amdgcn.ds.load.tr8.b64", vTyI32);
    case ISAFamily::CDNA4: {
      Value dsReadTr;
      if (bitWidth == 16)
        dsReadTr = ROCDL::ds_read_tr16_b64::create(rewriter, loc, vTy, vecAddr);
      else
        dsReadTr =
            ROCDL::ds_read_tr8_b64::create(rewriter, loc, vTyI32, vecAddr);
      AMD::addLocalLoadNoAliasScope(
          op, cast<LLVM::AliasAnalysisOpInterface>(dsReadTr.getDefiningOp()));
      return dsReadTr;
    }
    default:
      return {};
    }
  }

private:
  const AMD::TargetInfo &targetInfo;
  std::shared_ptr<DistributedCoordinateGroups> coordinateGroups;
};

class LocalLoadPackedTransposedOpConversion
    : public ConvertOpToLLVMPattern<
          triton::amdgpu::LocalLoadPackedTransposedOp> {
public:
  LocalLoadPackedTransposedOpConversion(const LLVMTypeConverter &converter,
                                        const AMD::TargetInfo &targetInfo,
                                        PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<triton::amdgpu::LocalLoadPackedTransposedOp>(
            converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor =
      typename triton::amdgpu::LocalLoadPackedTransposedOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::amdgpu::LocalLoadPackedTransposedOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    auto typeConverter = this->getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    unsigned bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    // FP4 is represented as i8 and
    if (bitWidth != 8) {
      return failure();
    }
    // FP4 packed along M/N are not supported yet on GFX1250
    if (targetInfo.getISAFamily() == ISAFamily::GFX1250) {
      return failure();
    }

    return lowerSharedToDotOperandTransLL(op, adaptor, typeConverter, rewriter);
  }

private:
  LogicalResult
  lowerSharedToDotOperandTransLL(triton::amdgpu::LocalLoadPackedTransposedOp op,
                                 OpAdaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kBlock = str_attr("block");
    auto kOffset = str_attr("offset");
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    if (SharedMemoryObject::getMaskSpanOffsetsAndBlocks(srcTy).second != 0)
      return failure();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitWidth = llvmElemTy.getIntOrFloatBitWidth();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    mlir::Type retTy = dstTy;
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
    auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);
    auto paddingShifts = getPaddedSharedShifts(srcTy.getEncoding(), bitWidth,
                                               /*offsetInBytes=*/true);

    auto shape = srcTy.getShape();
    auto ldsParamsVec = targetInfo.queryLDSTransLoadParams(bitWidth);
    if (ldsParamsVec.size() != 1)
      return failure();
    const auto ldsTransLoadParams = &ldsParamsVec[0];
    // FP4 are packed into i8 so the real bitWidth is different
    auto llBitWidth = 4;
    auto ldsTransLayout = triton::gpu::chooseDsReadTrLayout(
        dstTy.getEncoding(), shape, llBitWidth,
        ldsTransLoadParams->instBitWidth,
        ldsTransLoadParams->numLanesInShuffleGroup);

    // Check that we have computed a layout
    if (!ldsTransLayout) {
      return failure();
    }
    auto regLayout =
        ldsTransLayout->removeZeroBasesAlongDim(str_attr("register"));

    auto smemPtrTy = ptr_ty(ctx, 3);
    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(srcTy.getEncoding());
    LinearLayout cvt = LinearLayout::empty();
    if (paddedEnc) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvt = regLayout.invertAndCompose(sharedLL);
    } else {
      auto sharedLL = triton::gpu::toLinearLayout(srcTy);
      cvt = regLayout.invertAndCompose(sharedLL);
    }
    // Check that we will be able to vectorize the load.
    // Need to have exactly ldsTransLoadParams->tileSize,
    // otherwise we can't use ds_read_tr
    auto [elemsPerVec, permutation] =
        largestVectorisation(ctx, cvt, bitWidth, ldsTransLoadParams->tileSize);

    if (paddedEnc)
      elemsPerVec = std::min<int>(elemsPerVec, paddedEnc.getMinInterval());

    if (elemsPerVec != ldsTransLoadParams->tileSize)
      return failure();

    assert(cvt.isTrivialOver({kBlock}) && "NYI");
    auto lowerInst = [&](RewriterBase &rewriter, Location loc,
                         ArrayRef<Value> inVals, Value vecAddr, int idx,
                         VectorType vTy, Value ctaId) -> SmallVector<Value> {
      assert(!ctaId && "NYI");
      auto numElemsI32 = (vTy.getNumElements() * bitWidth / 32);
      auto vTyI32 = VectorType::get(numElemsI32, i32_ty);
      Value dsReadTr =
          ROCDL::ds_read_tr4_b64::create(rewriter, loc, vTyI32, vecAddr);
      Value vecVal = b.bitcast(dsReadTr, vTy);
      SmallVector<Value> loadedVals;
      for (int v = 0; v < vTy.getNumElements(); v++) {
        loadedVals.push_back(
            b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
      }

      return loadedVals;
    };

    SmallVector<Value> outVals = lowerLdSt(
        loc, rewriter.getContext(), cvt, {}, // Input for store, output for load
        llvmElemTy, smemObj.getBase(), paddingShifts, affineOffset,
        maskSpanAffineOffset, /*affineBlockOffset=*/Value(),
        /*maskSpanAffineBlock=*/0, laneId, warpId, rewriter, targetInfo,
        ldsTransLoadParams->tileSize, lowerInst);
    Value result =
        packUniqueTensorElements(loc, typeConverter, outVals, rewriter, retTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
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

struct ScheduledMfmaAsmInfo {
  StringRef asmMnemonic;
  // `_1k`: the gfx90a+ bf16 set taking 4 bf16/lane instead of 2, declared as
  // packed i16, so operands need a bitcast.
  bool intrinsicOperandsAreI16;
};

static FailureOr<ScheduledMfmaAsmInfo>
getScheduledMfmaAsmInfo(StringRef intrinsicName) {
  if (intrinsicName == ROCDL::mfma_f32_32x32x16_f16::getOperationName())
    return ScheduledMfmaAsmInfo{"v_mfma_f32_32x32x16_f16", false};
  if (intrinsicName == ROCDL::mfma_f32_32x32x16_bf16::getOperationName())
    return ScheduledMfmaAsmInfo{"v_mfma_f32_32x32x16_bf16", false};
  if (intrinsicName == ROCDL::mfma_f32_16x16x32_f16::getOperationName())
    return ScheduledMfmaAsmInfo{"v_mfma_f32_16x16x32_f16", false};
  if (intrinsicName == ROCDL::mfma_f32_16x16x32_bf16::getOperationName())
    return ScheduledMfmaAsmInfo{"v_mfma_f32_16x16x32_bf16", false};
  if (intrinsicName == ROCDL::mfma_f32_32x32x8f16::getOperationName())
    return ScheduledMfmaAsmInfo{"v_mfma_f32_32x32x8_f16", false};
  if (intrinsicName == ROCDL::mfma_f32_32x32x8bf16_1k::getOperationName())
    return ScheduledMfmaAsmInfo{"v_mfma_f32_32x32x8_bf16", true};
  if (intrinsicName == ROCDL::mfma_f32_16x16x16f16::getOperationName())
    return ScheduledMfmaAsmInfo{"v_mfma_f32_16x16x16_f16", false};
  if (intrinsicName == ROCDL::mfma_f32_16x16x16bf16_1k::getOperationName())
    return ScheduledMfmaAsmInfo{"v_mfma_f32_16x16x16_bf16", true};
  return failure();
}

struct ScheduledMfmaLoweringInfo {
  // Mnemonic only; the inline-asm path appends the operand list.
  StringRef asmMnemonic;
  StringRef intrinsicName;
  // K elements one lane feeds into a single MFMA; sets fragment width.
  int64_t kBase;
  // Padding before each inline-asm MFMA
  int inputWaitStates;
  // Padding after the last MFMA
  int drainWaitStates;
  // Operands must be bitcast to i16 vectors for the `_1k` intrinsics.
  bool intrinsicOperandsAreI16;
};

static FailureOr<ScheduledMfmaLoweringInfo>
getScheduledMfmaLoweringInfo(Location loc, ISAFamily isaFamily,
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

  FailureOr<ScheduledMfmaAsmInfo> asmInfo =
      getScheduledMfmaAsmInfo(intrinsic->name);
  if (failed(asmInfo))
    return failure();

  FailureOr<int> drainWaitStates =
      getMfmaDrainWaitStates(isaFamily, instrShape);
  if (failed(drainWaitStates))
    return failure();

  return ScheduledMfmaLoweringInfo{asmInfo->asmMnemonic,
                                   intrinsic->name,
                                   static_cast<int64_t>(intrinsic->kBase),
                                   /*inputWaitStates=*/4,
                                   *drainWaitStates,
                                   asmInfo->intrinsicOperandsAreI16};
}

static FailureOr<Value>
constrainMfmaFragmentRegisterClass(Value fragment, StringRef registerClass,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc) {
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
      /*has_side_effects=*/false,
      /*is_align_stack=*/false, LLVM::TailCallKind::None, asmDialect,
      operandAttrs);
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

// Drain the MFMA pipeline before `fragment` is read by anything else.
//
// The scheduled-MFMA lowering emits `v_mfma_*` inside an `asm sideeffect`
// block so it can pin the accumulator's register class. AMDGPU's hazard
// recognizer matches on `SIInstrInfo::isMAI()` and therefore cannot see an
// MFMA hidden inside `INLINEASM`: it never inserts the mandatory wait states
// between the MFMA writing its destination and the first consumer reading it.
// (The `transient` path lowers to the ROCDL intrinsic and gets them for free --
// LLVM emits e.g. `s_nop 7` before a `buffer_store` of a 16x16x32 result.)
// Without this drain the consumer reads the destination registers before the
// MFMA has written them, yielding garbage or NaN.
//
// This is emitted once per accumulator chain, not per MFMA: back-to-back MFMAs
// forwarding srcC need no padding, so the chain itself is not serialized.
//
// `waitStates` mirrors LLVM's target-specific
// MFMA*WritesAGPRAccVgprReadWaitStates requirement.
static FailureOr<Value>
drainMfmaPipeline(Value fragment, StringRef registerClass, int waitStates,
                  ConversionPatternRewriter &rewriter, Location loc) {
  auto fragmentTy = cast<VectorType>(fragment.getType());
  unsigned elementBitWidth =
      getIntOrFloatOrPtrBitWidth(fragmentTy.getElementType());
  int64_t totalBitWidth = fragmentTy.getNumElements() * elementBitWidth;
  if (totalBitWidth <= 0 || totalBitWidth % 32 != 0)
    return failure();

  std::string waitAsm = mfmaWaitStateAsm(waitStates);
  auto registerVectorTy = vec_ty(i32_ty, totalBitWidth / 32);
  TritonLLVMOpBuilder b(loc, rewriter);
  Value packed = b.bitcast(fragment, registerVectorTy);
  auto *ctx = rewriter.getContext();
  auto asmDialect = LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
  // Tie the result to the input so the drain stays on the accumulator's SSA
  // chain and keeps the accumulator in its pinned register class.
  std::string constraints =
      (registerClass == "agpr" ? std::string("=a") : std::string("=v")) + ",0";
  auto drain = LLVM::InlineAsmOp::create(
      rewriter, loc, registerVectorTy, ValueRange{packed}, waitAsm, constraints,
      /*has_side_effects=*/true,
      /*is_align_stack=*/false, LLVM::TailCallKind::None, asmDialect,
      ArrayAttr::get(ctx, {}));
  Value drained = b.bitcast(drain->getResult(0), fragmentTy);
  return drained;
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

class MfmaCommitOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::MfmaCommitOp> {
public:
  using OpAdaptor = triton::amdgpu::MfmaCommitOp::Adaptor;

  MfmaCommitOpConversion(const LLVMTypeConverter &converter,
                         const AMD::TargetInfo &targetInfo,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::amdgpu::MfmaCommitOp>(converter,
                                                             benefit),
        targetInfo(targetInfo) {}

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
    for (auto [source, converted] :
         llvm::zip(op.getInputs(), adaptor.getInputs())) {
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
        outputConstraint = hasLiveDependency ? "=v" : "=a";
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

    // Preserve the established gfx950 boundary. On gfx942, use the largest
    // result-read delay among the native layouts carried by this boundary;
    // `getMfmaDrainWaitStates` is the same requirement the scheduled-MFMA
    // lowering pads for, so the two stay in sync.
    int waitStates = 6;
    if (targetInfo.getISAFamily() == ISAFamily::CDNA3) {
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

    SmallVector<Value> results;
    for (size_t inputIndex = 0; inputIndex < inputGroups.size(); ++inputIndex) {
      SmallVector<Value> elements;
      Type elementVectorTy = elementVectorTypes[inputIndex];
      auto vectorTy = cast<VectorType>(elementVectorTy);
      for (size_t group = 0; group < inputGroups[inputIndex].size(); ++group) {
        Value registerGroup =
            getConstrainedGroup(firstGroupIndices[inputIndex] + group);
        Value elementGroup = b.bitcast(registerGroup, elementVectorTy);
        for (int64_t index = 0; index < vectorTy.getNumElements(); ++index)
          elements.push_back(b.extract_element(vectorTy.getElementType(),
                                               elementGroup, b.i32_val(index)));
      }
      results.push_back(packTensorElements(
          loc, typeConverter, elements, rewriter,
          cast<RankedTensorType>(op.getOutputs()[inputIndex].getType())));
    }
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

class ScheduledMfmaOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ScheduledMfmaOp> {
public:
  using OpAdaptor = triton::amdgpu::ScheduledMfmaOp::Adaptor;

  ScheduledMfmaOpConversion(const LLVMTypeConverter &converter,
                            const AMD::TargetInfo &targetInfo,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::amdgpu::ScheduledMfmaOp>(converter,
                                                                benefit),
        targetInfo(targetInfo) {}

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
        getScheduledMfmaLoweringInfo(loc, targetInfo.getISAFamily(), mfma,
                                     aTy.getElementType(),
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

    auto inputConstraint = [](StringRef registerClass) -> StringRef {
      return registerClass == "agpr" ? "a" : "v";
    };
    StringRef outputConstraint = accumulatorStorage == "agpr" ? "=a" : "=&v";
    Value zeroFragment;
    if (op.getAccumulatorRole() == "transient" && op.getInitialize()) {
      Attribute zeroAttr = rewriter.getZeroAttr(accElemTy);
      auto zeroElements =
          DenseElementsAttr::get(cast<ShapedType>(fragmentTy), zeroAttr);
      zeroFragment =
          LLVM::ConstantOp::create(rewriter, loc, fragmentTy, zeroElements);
    }
    auto *ctx = rewriter.getContext();
    auto asmDialect = LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
    auto operandAttrs = ArrayAttr::get(ctx, {});
    bool useLatencyAwareIntrinsic = op.getAccumulatorRole() == "transient";

    SmallVector<Value> updatedFragments = accumulatorFragments;
    // Keep one SSA chain per output fragment while making source order
    // explicit across the grid. Round-robin the K slices over independent
    // output fragments so persistent inline-assembly chains expose enough
    // distance between dependent MFMAs. Native intrinsics expose transient
    // chains to AMDGPU's MFMA hazard recognizer and machine scheduler. Direct
    // inline assembly preserves the requested register class for persistent
    // chains.
    for (int64_t k = 0; k < numRepK; ++k) {
      for (int64_t n = 0; n < numRepN; ++n) {
        for (int64_t m = 0; m < numRepM; ++m) {
          int64_t accumulatorIndex = m * numRepN + n;
          Value current = updatedFragments[accumulatorIndex];
          Value operandA = (*maybeA)[m * numRepK + k];
          Value operandB = (*maybeB)[n * numRepK + k];
          // The MFMA inline asm below already constrains ordinary operands to
          // VGPRs.  Pre-constrain only the resident-operand path, where one
          // input must be moved to AGPRs before issuing the instruction.
          if (!useLatencyAwareIntrinsic &&
              (aStorage == "agpr" || bStorage == "agpr")) {
            FailureOr<Value> constrainedA = constrainMfmaFragmentRegisterClass(
                operandA, aStorage, rewriter, loc);
            FailureOr<Value> constrainedB = constrainMfmaFragmentRegisterClass(
                operandB, bStorage, rewriter, loc);
            if (failed(constrainedA) || failed(constrainedB))
              return rewriter.notifyMatchFailure(
                  op, "native MFMA operands must pack into complete 32-bit "
                      "registers");
            operandA = *constrainedA;
            operandB = *constrainedB;
          }
          StringRef aRegisterClass = aStorage;
          StringRef bRegisterClass = bStorage;
          if (mfma.getIsTransposed()) {
            std::swap(operandA, operandB);
            std::swap(aRegisterClass, bRegisterClass);
          }

          bool zeroThisInstruction = op.getInitialize() && k == 0;
          if (useLatencyAwareIntrinsic) {
            if (info.intrinsicOperandsAreI16) {
              auto packedTy = vec_ty(i16_ty, info.kBase);
              operandA = b.bitcast(operandA, packedTy);
              operandB = b.bitcast(operandB, packedTy);
            }
            OperationState loweredOp(loc, info.intrinsicName);
            loweredOp.addTypes(fragmentTy);
            Value intrinsicAcc = zeroThisInstruction ? zeroFragment : current;
            loweredOp.addOperands({operandA, operandB, intrinsicAcc});
            loweredOp.addAttribute("cbsz", rewriter.getI32IntegerAttr(0));
            loweredOp.addAttribute("abid", rewriter.getI32IntegerAttr(0));
            // For `blgp`: f64 MFMA uses negation flags, while other MFMA ops
            // use B-lane permutation flags.
            MLIRContext *ctx = rewriter.getContext();
            if (cast<VectorType>(fragmentTy).getElementType().isF64()) {
              loweredOp.addAttribute("blgp",
                                     ROCDL::MFMANegModifierAttr::get(
                                         ctx, ROCDL::MFMANegModifier::none));
            } else {
              loweredOp.addAttribute("blgp", ROCDL::MFMAPermBAttr::get(
                                                 ctx, ROCDL::MFMAPermB::none));
            }
            current = rewriter.create(loweredOp)->getResult(0);
          } else {
            std::string constraints = outputConstraint.str();
            constraints += "," + inputConstraint(aRegisterClass).str();
            constraints += "," + inputConstraint(bRegisterClass).str();
            SmallVector<Value> asmOperands{operandA, operandB};
            if (!zeroThisInstruction) {
              asmOperands.push_back(current);
              constraints += ",0";
            }
            // The hazard recognizer cannot see this MFMA (it lives inside an
            // `asm sideeffect` block), so it will not pad a preceding VALU
            // write of srcA/srcB or of EXEC. Per LLVM's checkMAIHazards90A
            // that needs `LegacyVALUNotDotWritesVGPRWaitStates` (2) and
            // `VALUWritesExecWaitStates` (4) respectively; 4 covers both. An
            // exact same-register srcC forward from the previous MFMA in the
            // chain is explicitly not a hazard, so this does not serialize
            // the accumulation chain.
            std::string mfmaAsm = mfmaWaitStateAsm(info.inputWaitStates) +
                                  "\n" + info.asmMnemonic.str() +
                                  " $0, $1, $2, ";
            mfmaAsm += zeroThisInstruction ? "0" : "$0";
            auto inlineAsm = LLVM::InlineAsmOp::create(
                rewriter, loc, fragmentTy, asmOperands, mfmaAsm, constraints,
                /*has_side_effects=*/true,
                /*is_align_stack=*/false, LLVM::TailCallKind::None, asmDialect,
                operandAttrs);
            current = inlineAsm->getResult(0);
          }
          updatedFragments[accumulatorIndex] = current;
        }
      }
    }

    if (!useLatencyAwareIntrinsic) {
      // The consumer is unknown at this point, so use the target-specific
      // result-read requirement from `getMfmaDrainWaitStates`. Sizing the drain
      // for the worst consumer keeps it sufficient on its own, rather than
      // relying on the next MFMA's input padding to make up a shortfall.
      for (int64_t n = 0; n < numRepN; ++n) {
        for (int64_t m = 0; m < numRepM; ++m) {
          int64_t accumulatorIndex = m * numRepN + n;
          FailureOr<Value> drained = drainMfmaPipeline(
              updatedFragments[accumulatorIndex], accumulatorStorage,
              info.drainWaitStates, rewriter, loc);
          if (failed(drained))
            return rewriter.notifyMatchFailure(
                op, "MFMA accumulator fragment must pack into complete 32-bit "
                    "registers");
          updatedFragments[accumulatorIndex] = *drained;
        }
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
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
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
    if (targetInfo.getIsaVersion().Major < 9)
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

void mlir::triton::AMD::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit,
    std::shared_ptr<DistributedCoordinateGroups> coordinateGroups) {
  PatternBenefit transBenefit = PatternBenefit(benefit.getBenefit() + 1);
  PatternBenefit barrierBenefit = PatternBenefit(benefit.getBenefit() + 1);

  patterns.add<TransLocalLoadOpConversion>(typeConverter, targetInfo,
                                           transBenefit, coordinateGroups);
  patterns.add<LocalLoadPackedTransposedOpConversion>(typeConverter, targetInfo,
                                                      benefit);
  patterns.add<LocalAtomicScatterRMWOpConversion>(typeConverter, targetInfo,
                                                  benefit.getBenefit() + 1);
  patterns.add<RematerializedRangeOpConversion>(typeConverter, targetInfo,
                                                transBenefit);
  patterns.add<RegisterResidentOpConversion, RegisterClassAnchorOpConversion>(
      typeConverter, transBenefit);
  patterns.add<MfmaCommitOpConversion, ScheduledMfmaOpConversion>(
      typeConverter, targetInfo, transBenefit);
  patterns.add<BarrierOpConversion, MemoryCounterWaitOpConversion>(
      typeConverter, targetInfo, barrierBenefit);
}
