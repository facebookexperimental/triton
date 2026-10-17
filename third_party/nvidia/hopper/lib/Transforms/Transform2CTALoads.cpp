// Transform B matrix descriptor loads for 2-CTA MMA operations.
//
// When non-TLX TCGen5MMAOp has two_ctas=true, this pass splits B loads so each
// CTA loads half of B:
//   CTA 0 loads B[:, 0 : BLOCK_N/2]
//   CTA 1 loads B[:, BLOCK_N/2 : BLOCK_N]
//
// The pass:
//   1. Traces B operand from MMA back to its DescriptorLoadOp
//   2. Clones the MakeTensorDescOp with half-width block shape
//   3. Adds CTA-based offset to the load's N-dimension index
//   4. Creates a new DescriptorLoadOp with half-width result
//   5. Creates a new LocalAllocOp with half-width SMEM allocation
//
// This pass is needed for the ctas_per_cga=(2,1,1) approach where num_ctas=1,
// because splitBOperand (used by PlanCTA path) requires CTASplitNum=[2,1]
// which doesn't exist with num_ctas=1.
//
// Must run after AccelerateMatmul (which creates TCGen5MMAOp) and before
// Insert2CTASync (which adds cross-CTA barriers).

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace nvgpu = triton::nvgpu;

#define DEBUG_TYPE "nvgpu-2cta-transform-loads"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_NVGPU2CTATRANSFORMLOADS
#include "nvidia/hopper/include/Transforms/Passes.h.inc"
} // namespace mlir

namespace {

// Create a BlockedEncodingAttr compatible with the given shape.
// If the original encoding's tile size exceeds the new shape in the N
// dimension, adjust threadsPerWarp to fit.
static ttg::BlockedEncodingAttr
getCompatibleEncoding(ttg::BlockedEncodingAttr origEncoding,
                      ArrayRef<int64_t> shape, unsigned splitDim,
                      MLIRContext *ctx) {
  auto spt = SmallVector<unsigned>(origEncoding.getSizePerThread());
  auto tpw = SmallVector<unsigned>(origEncoding.getThreadsPerWarp());
  auto wpc = SmallVector<unsigned>(origEncoding.getWarpsPerCTA());
  auto order = SmallVector<unsigned>(origEncoding.getOrder());
  auto ctaLayout = origEncoding.getCGALayout();

  bool compatible = true;
  for (auto dimAndSize : llvm::enumerate(shape)) {
    unsigned dim = dimAndSize.index();
    int64_t size = dimAndSize.value();
    unsigned tile = spt[dim] * tpw[dim] * wpc[dim];
    compatible &= size % tile == 0;
  }
  if (compatible)
    return origEncoding;

  // Reduce the split dimension's tile size. Move threads to the other
  // dimension when possible to keep the total threadsPerWarp unchanged.
  unsigned otherDim = 1 - splitDim;
  while (spt[splitDim] * tpw[splitDim] * wpc[splitDim] >
             static_cast<unsigned>(shape[splitDim]) &&
         tpw[splitDim] > 1) {
    tpw[splitDim] /= 2;
    tpw[otherDim] *= 2;
  }
  // If still too large, reduce sizePerThread in the split dimension.
  while (spt[splitDim] * tpw[splitDim] * wpc[splitDim] >
             static_cast<unsigned>(shape[splitDim]) &&
         spt[splitDim] > 1) {
    spt[splitDim] /= 2;
  }

  return ttg::BlockedEncodingAttr::get(ctx, spt, tpw, wpc, order, ctaLayout);
}

// Halving the block along the contiguous dimension can leave the original
// swizzle byte width illegal: an NVMMASharedLayout requires the contiguous
// dimension to hold at least 8 * swizzleByteWidth / elementBitWidth elements.
// A 128-byte-swizzled fp16 B tile that is 64 elements wide (HEAD_DIM=64) is
// legal, but its 32-element half is not. Recompute the widest swizzle that is
// legal for the halved shape instead of carrying the original one over.
static Attribute shrinkSwizzleForShape(Attribute layout,
                                       ArrayRef<int64_t> newBlockShape) {
  auto nvmma = dyn_cast_if_present<ttg::NVMMASharedEncodingAttr>(layout);
  if (!nvmma || nvmma.getSwizzlingByteWidth() == 0 || newBlockShape.size() < 2)
    return layout;

  unsigned contigDim = nvmma.getTransposed() ? 0 : newBlockShape.size() - 1;
  unsigned eltBitWidth = nvmma.getElementBitWidth();
  int64_t packingFactor = nvmma.getFp4Padded() ? 2 : 1;
  int64_t contigBytes =
      newBlockShape[contigDim] * packingFactor * eltBitWidth / 8;

  unsigned swizzle = nvmma.getSwizzlingByteWidth();
  while (swizzle >= 32 && contigBytes < static_cast<int64_t>(swizzle))
    swizzle /= 2;
  if (swizzle < 32)
    swizzle = 0;
  if (swizzle == nvmma.getSwizzlingByteWidth())
    return layout;

  return ttg::NVMMASharedEncodingAttr::get(
      layout.getContext(), swizzle, nvmma.getTransposed(), eltBitWidth,
      nvmma.getFp4Padded(), nvmma.getCGALayout());
}

struct BLoadTrace {
  tt::DescriptorLoadOp descLoad;
  ttg::LocalAllocOp localAlloc;
  ttg::MemDescTransOp memDescTrans;
  tt::TransOp trans;
  unsigned splitDim = 1;
};

// Descriptor loads may drop leading unit block dimensions from their tensor
// result.  BMM uses this to keep one rank-3 [B, K, N] descriptor while feeding
// the rank-2 [K, N] tile expected by dot.  Map the MMA/result dimension back
// to the descriptor dimension, accepting only this unambiguous prefix-unit
// form.
static FailureOr<unsigned> mapResultDimToDescriptor(tt::TensorDescType descType,
                                                    RankedTensorType resultType,
                                                    unsigned resultDim) {
  auto blockShape = descType.getBlockType().getShape();
  auto resultShape = resultType.getShape();
  if (resultDim >= resultShape.size() || blockShape.size() < resultShape.size())
    return failure();

  unsigned rankDelta = blockShape.size() - resultShape.size();
  if (!llvm::all_of(blockShape.take_front(rankDelta),
                    [](int64_t dim) { return dim == 1; }))
    return failure();
  if (!llvm::equal(blockShape.drop_front(rankDelta), resultShape))
    return failure();
  return rankDelta + resultDim;
}

// Halve dimension splitDim of a blocked layout's tile when the halved shape no
// longer holds it: move threads from splitDim to the largest other dimension,
// then shrink sizePerThread. Returns null if the layout is not blocked.
static ttg::BlockedEncodingAttr halveBlockedEncoding(Attribute encoding,
                                                     ArrayRef<int64_t> shape,
                                                     unsigned splitDim) {
  auto blocked = dyn_cast_if_present<ttg::BlockedEncodingAttr>(encoding);
  if (!blocked)
    return {};
  auto spt = SmallVector<unsigned>(blocked.getSizePerThread());
  auto tpw = SmallVector<unsigned>(blocked.getThreadsPerWarp());
  auto wpc = SmallVector<unsigned>(blocked.getWarpsPerCTA());
  unsigned otherDim = splitDim;
  for (unsigned d = 0; d < shape.size(); ++d)
    if (d != splitDim && (otherDim == splitDim || shape[d] > shape[otherDim]))
      otherDim = d;
  auto tile = [&](unsigned d) { return spt[d] * tpw[d] * wpc[d]; };
  while (otherDim != splitDim &&
         tile(splitDim) > static_cast<unsigned>(shape[splitDim]) &&
         tpw[splitDim] > 1) {
    tpw[splitDim] /= 2;
    tpw[otherDim] *= 2;
  }
  while (tile(splitDim) > static_cast<unsigned>(shape[splitDim]) &&
         spt[splitDim] > 1)
    spt[splitDim] /= 2;
  return ttg::BlockedEncodingAttr::get(blocked.getContext(), spt, tpw, wpc,
                                       blocked.getOrder(),
                                       blocked.getCGALayout());
}

// For a reshape that only inserts or removes unit dimensions, map a
// non-unit result dimension to the matching source dimension.
static std::optional<unsigned> mapSqueezeReshapeDim(ArrayRef<int64_t> src,
                                                    ArrayRef<int64_t> dst,
                                                    unsigned dstDim) {
  SmallVector<unsigned> srcDims, dstDims;
  for (unsigned d = 0; d < src.size(); ++d)
    if (src[d] != 1)
      srcDims.push_back(d);
  for (unsigned d = 0; d < dst.size(); ++d)
    if (dst[d] != 1)
      dstDims.push_back(d);
  if (srcDims.size() != dstDims.size())
    return std::nullopt;
  for (unsigned i = 0; i < srcDims.size(); ++i) {
    if (src[srcDims[i]] != dst[dstDims[i]])
      return std::nullopt;
    if (dstDims[i] == dstDim)
      return srcDims[i];
  }
  return std::nullopt;
}

// Trace B operand from MMA back through LocalAllocOp and cheap layout/view ops
// to find the DescriptorLoadOp. When B is transposed, either before allocation
// with tt.trans or after allocation with ttg.memdesc_trans, split the
// descriptor dimension that becomes the MMA N dimension after transpose.
static FailureOr<BLoadTrace> traceToDescriptorLoad(Value bMemDesc) {
  ttg::MemDescTransOp memDescTrans;
  unsigned splitDim = 1;
  if (auto transOp = bMemDesc.getDefiningOp<ttg::MemDescTransOp>()) {
    memDescTrans = transOp;
    if (memDescTrans.getOrder().size() != 2)
      return failure();
    // MMA B's N dimension is result dimension 1. Map it back to the source
    // descriptor/local_alloc dimension through the memdesc transpose order.
    splitDim = memDescTrans.getOrder()[1];
    bMemDesc = memDescTrans.getSrc();
  }

  auto localAlloc = bMemDesc.getDefiningOp<ttg::LocalAllocOp>();
  if (!localAlloc)
    return failure();

  Value tensor = localAlloc.getSrc();
  // Skip convert_layout ops.
  while (auto cvt = tensor.getDefiningOp<ttg::ConvertLayoutOp>())
    tensor = cvt.getSrc();

  tt::TransOp trans;
  if (auto transOp = tensor.getDefiningOp<tt::TransOp>()) {
    trans = transOp;
    if (trans.getOrder().size() != 2)
      return failure();
    // MMA B's N dimension is result dimension 1. Map it back to the
    // descriptor-load source dimension through the transpose order.
    splitDim = trans.getOrder()[splitDim];
    tensor = trans.getSrc();
    while (auto cvt = tensor.getDefiningOp<ttg::ConvertLayoutOp>())
      tensor = cvt.getSrc();
  }

  auto descLoad = tensor.getDefiningOp<tt::DescriptorLoadOp>();
  if (!descLoad)
    return failure();

  return BLoadTrace{descLoad, localAlloc, memDescTrans, trans, splitDim};
}

struct Transform2CTALoads
    : public impl::NVGPU2CTATransformLoadsBase<Transform2CTALoads> {
  DenseMap<Value, tt::TensorDescType> originalDescTypes;
  // B operands (local_alloc / memdesc_trans results) already split by an
  // earlier MMA that shares them.
  DenseSet<Value> splitBOperands;
  // Half-width B allocs written from registers. Each CTA writes its own half,
  // so the MMAs reading them need a cluster-scoped fence.
  DenseSet<Operation *> registerSplitBAllocs;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (!ttg::isPhysicalCluster(moduleOp))
      return;

    // TLX kernels manage their own 2-CTA load splitting and synchronization.
    if (moduleOp->hasAttr("tlx.has_tlx_ops"))
      return;

    originalDescTypes.clear();
    moduleOp.walk([&](tt::DescriptorLoadOp descLoad) {
      auto descType = cast<tt::TensorDescType>(descLoad.getDesc().getType());
      originalDescTypes.try_emplace(descLoad.getDesc(), descType);
    });

    // Collect 2-CTA MMA ops. Walk the shared MMAv5 interface so both the plain
    // (tc_gen5_mma) and scaled (tc_gen5_mma_scaled) ops are handled by one
    // path.
    SmallVector<ttng::MMAv5OpInterface> twoCTAMMAOps;
    moduleOp->walk([&](ttng::MMAv5OpInterface mma) {
      if (mma.getTwoCtas())
        twoCTAMMAOps.push_back(mma);
    });

    if (twoCTAMMAOps.empty())
      return;

    LDBG("Found " << twoCTAMMAOps.size() << " 2-CTA MMA ops to transform");

    DenseSet<Operation *> splitTransformedMMAs;
    for (auto mma : twoCTAMMAOps) {
      Operation *op = mma.getOperation();
      if (splitTransformedMMAs.contains(op) ||
          splitBOperands.contains(mma.getB()))
        continue;
      // Both paths halve the B descriptor; they differ in how many MMAs consume
      // the load. transformSplitBLoad handles a source-level tt.split feeding
      // two MMAs by making two subslice views of one buffer, and only a plain
      // MMA has that split. A scaled MMA has a single consumer and uses
      // transformBLoad below.
      if (auto plain = dyn_cast<ttng::TCGen5MMAOp>(op)) {
        if (succeeded(transformSplitBLoad(plain, splitTransformedMMAs)))
          continue;
      }
      if (succeeded(transformBLoad(mma.getB(), op)))
        continue;
      if (succeeded(transformRegisterBuiltB(mma.getB(), op)))
        continue;
      // A 2-CTA MMA reads half of B from each CTA of the pair. Leaving the
      // full B tile in both CTAs silently computes with the wrong half.
      op->emitError("two_ctas dot: cannot split the B operand across the CTA "
                    "pair (B must be built from descriptor loads through "
                    "elementwise, broadcast, reshape or transpose ops, with "
                    "BLOCK_N / 2 >= 16)");
      signalPassFailure();
      return;
    }

    for (auto mma : twoCTAMMAOps) {
      Value b = mma.getB();
      while (Operation *def = b.getDefiningOp()) {
        if (!def->hasTrait<OpTrait::MemDescViewTrait>())
          break;
        b = def->getOperand(0);
      }
      if (registerSplitBAllocs.contains(b.getDefiningOp()))
        mma->setAttr(ttng::AttrTwoCTARegisterBName,
                     UnitAttr::get(mma->getContext()));
    }

    // Host-side descriptor arguments are retyped to the half-width block in
    // place. Any load of such an argument that was not replaced by a
    // half-width load would read the wrong tile.
    bool unsplitHostLoad = false;
    moduleOp.walk([&](tt::DescriptorLoadOp descLoad) {
      auto desc = descLoad.getDesc();
      if (desc.getDefiningOp())
        return;
      auto orig = originalDescTypes.lookup(desc);
      if (orig && orig != desc.getType() && !descLoad->hasAttr("two_cta_b")) {
        descLoad.emitError("two_ctas dot: this load shares a descriptor "
                           "argument with a 2-CTA B operand that was split "
                           "across the CTA pair, but is not split itself");
        unsplitHostLoad = true;
      }
    });
    if (unsplitHostLoad) {
      signalPassFailure();
      return;
    }

    // A 2-CTA TMA load is issued by both CTAs as one hardware CTA-group
    // transaction. Mark every descriptor load producing a rank-2 MMA tile in
    // the cooperative kernel, including rank-3 descriptors with a leading
    // unit block dimension and A operands (K/V) that are not visited by B
    // splitting. Rank-1 metadata loads remain ordinary per-CTA loads, matching
    // TLX's raw M/D bulk-copy path.
    //
    // Cooperative marking is a performance choice, not a correctness
    // requirement: leaving a load unmarked keeps the pre-existing per-CTA
    // path, where each CTA issues its own transaction. Setting
    // TRITON_DISABLE_2CTA_COOPERATIVE_LOADS=1 selects that path for the whole
    // module so the two protocols can be benchmarked against each other on a
    // given kernel. Marking stays all-or-nothing per module because barrier
    // fusion requires a homogeneous group; optimizeTMALoads rejects a mixed
    // group rather than silently miscounting expected bytes.
    if (!triton::tools::getBoolEnv("TRITON_DISABLE_2CTA_COOPERATIVE_LOADS")) {
      moduleOp.walk([&](tt::DescriptorLoadOp descLoad) {
        auto resultTy = dyn_cast<RankedTensorType>(descLoad.getType());
        if (resultTy && resultTy.getRank() == 2)
          descLoad->setAttr(ttng::AttrTwoCTALoadName,
                            UnitAttr::get(moduleOp.getContext()));
      });
    }
  }

  // A source-level split of one KxN V tile normally becomes
  //
  //   descriptor_load -> reshape -> trans -> split -> local_alloc x2
  //
  // before this pass. Transforming either leaf independently would retain a
  // register-mediated shared-to-shared copy. Instead, make the cooperative
  // descriptor load half-width once, allocate its complete 2Kx(N/2) tile in
  // shared memory, and feed the two MMAs with zero-copy Kx(N/2) subslices.
  LogicalResult transformSplitBLoad(ttng::TCGen5MMAOp mma,
                                    DenseSet<Operation *> &transformedMMAs) {
    auto localAlloc = mma.getB().getDefiningOp<ttg::LocalAllocOp>();
    if (!localAlloc)
      return failure();
    auto split = localAlloc.getSrc().getDefiningOp<tt::SplitOp>();
    if (!split || !split.getOutLHS().hasOneUse() ||
        !split.getOutRHS().hasOneUse())
      return failure();

    auto lhsAlloc =
        dyn_cast<ttg::LocalAllocOp>(*split.getOutLHS().getUsers().begin());
    auto rhsAlloc =
        dyn_cast<ttg::LocalAllocOp>(*split.getOutRHS().getUsers().begin());
    if (!lhsAlloc || !rhsAlloc || !lhsAlloc.getResult().hasOneUse() ||
        !rhsAlloc.getResult().hasOneUse())
      return failure();
    auto lhsMMA =
        dyn_cast<ttng::TCGen5MMAOp>(*lhsAlloc.getResult().getUsers().begin());
    auto rhsMMA =
        dyn_cast<ttng::TCGen5MMAOp>(*rhsAlloc.getResult().getUsers().begin());
    if (!lhsMMA || !rhsMMA || !lhsMMA.getTwoCtas() || !rhsMMA.getTwoCtas() ||
        lhsMMA.getB() != lhsAlloc.getResult() ||
        rhsMMA.getB() != rhsAlloc.getResult())
      return failure();

    auto trans = split.getSrc().getDefiningOp<tt::TransOp>();
    if (!trans || trans.getOrder().size() != 3 || trans.getOrder()[0] != 1 ||
        trans.getOrder()[1] != 2 || trans.getOrder()[2] != 0)
      return failure();
    auto reshape = trans.getSrc().getDefiningOp<tt::ReshapeOp>();
    if (!reshape)
      return failure();
    auto descLoad = reshape.getSrc().getDefiningOp<tt::DescriptorLoadOp>();
    if (!descLoad)
      return failure();

    auto descType = originalDescTypes.lookup(descLoad.getDesc());
    auto loadType = dyn_cast<RankedTensorType>(descLoad.getType());
    auto lhsType = dyn_cast<ttg::MemDescType>(lhsAlloc.getType());
    auto rhsType = dyn_cast<ttg::MemDescType>(rhsAlloc.getType());
    if (!descType || !loadType || !lhsType || !rhsType ||
        loadType.getRank() != 2 || lhsType != rhsType)
      return failure();

    constexpr unsigned splitDim = 1;
    auto blockShape = descType.getBlockType().getShape();
    auto loadShape = loadType.getShape();
    auto leafShape = lhsType.getShape();
    if (blockShape.size() != 2 || blockShape[splitDim] % 2 != 0 ||
        loadShape.size() != 2 || leafShape.size() != 2 ||
        loadShape[0] != 2 * leafShape[0] || loadShape[1] != leafShape[1])
      return failure();

    int64_t halfN = blockShape[splitDim] / 2;
    if (halfN < 16)
      return failure();
    SmallVector<int64_t> newBlockShape(blockShape.begin(), blockShape.end());
    newBlockShape[splitDim] = halfN;

    MLIRContext *ctx = mma.getContext();
    auto elemType = descType.getElementType();
    auto sharedLayout =
        shrinkSwizzleForShape(descType.getSharedLayout(), newBlockShape);
    auto newDescType =
        tt::TensorDescType::get(newBlockShape, elemType, sharedLayout);

    Value newDesc;
    auto makeDesc = descLoad.getDesc().getDefiningOp<tt::MakeTensorDescOp>();
    if (makeDesc) {
      OpBuilder descBuilder(makeDesc);
      IRMapping mapper;
      auto *clonedOp = descBuilder.clone(*makeDesc.getOperation(), mapper);
      auto newMakeDesc = cast<tt::MakeTensorDescOp>(clonedOp);
      newMakeDesc.getResult().setType(newDescType);
      newDesc = newMakeDesc.getResult();
    } else {
      auto descVal = descLoad.getDesc();
      // The makeDesc path above clones before retyping. Here there is nothing
      // to clone -- the descriptor is typically a func argument -- so the type
      // is mutated in place, which is only sound if this load is its sole
      // consumer. Otherwise an unrelated descriptor_load sharing the value
      // would silently inherit the half-width type and address the wrong tile.
      if (!descVal.hasOneUse())
        return failure();
      descVal.setType(newDescType);
      newDesc = descVal;
      if (auto funcOp = descLoad->getParentOfType<triton::FuncOp>()) {
        auto &entryBlock = funcOp.getBlocks().front();
        SmallVector<Type> argTys(entryBlock.getArgumentTypes());
        funcOp.setFunctionType(FunctionType::get(
            ctx, argTys, funcOp.getFunctionType().getResults()));
      }
    }

    OpBuilder builder(descLoad);
    Location loc = descLoad.getLoc();
    auto i32Ty = builder.getI32Type();
    Value ctaRank = nvgpu::ClusterCTAIdOp::create(builder, loc, i32Ty);
    Value two = arith::ConstantIntOp::create(builder, loc, 2, 32);
    Value ctaMod2 = arith::RemSIOp::create(builder, loc, ctaRank, two);
    Value halfNVal = arith::ConstantIntOp::create(builder, loc, halfN, 32);
    Value offset = arith::MulIOp::create(builder, loc, ctaMod2, halfNVal);
    SmallVector<Value> newIndices(descLoad.getIndices());
    newIndices[splitDim] =
        arith::AddIOp::create(builder, loc, newIndices[splitDim], offset);

    auto origEncoding = cast<ttg::BlockedEncodingAttr>(loadType.getEncoding());
    auto newEncoding =
        getCompatibleEncoding(origEncoding, newBlockShape, splitDim, ctx);
    auto halfResultType =
        RankedTensorType::get(newBlockShape, elemType, newEncoding);
    auto newDescLoad = tt::DescriptorLoadOp::create(
        builder, loc, halfResultType, newDesc, newIndices);
    newDescLoad->setAttr("two_cta_b", builder.getUnitAttr());

    builder.setInsertionPoint(lhsAlloc);
    auto fullMemDescEncoding =
        shrinkSwizzleForShape(lhsType.getEncoding(), newBlockShape);
    auto fullMemDescType = ttg::MemDescType::get(
        newBlockShape, elemType, fullMemDescEncoding, lhsType.getMemorySpace(),
        lhsType.getMutableMemory());
    auto fullAlloc = ttg::LocalAllocOp::create(
        builder, lhsAlloc.getLoc(), fullMemDescType, newDescLoad.getResult());

    SmallVector<int64_t> slicedShape(newBlockShape.begin(),
                                     newBlockShape.end());
    slicedShape[0] /= 2;
    auto slicedType = ttg::MemDescType::get(
        slicedShape, elemType, fullMemDescEncoding, lhsType.getMemorySpace(),
        lhsType.getMutableMemory(), newBlockShape);
    SmallVector<int32_t> lhsOffsets = {0, 0};
    SmallVector<int32_t> rhsOffsets = {static_cast<int32_t>(slicedShape[0]), 0};
    Value lhsSlice = ttg::MemDescSubsliceOp::create(
        builder, lhsAlloc.getLoc(), slicedType, fullAlloc, lhsOffsets);
    Value rhsSlice = ttg::MemDescSubsliceOp::create(
        builder, rhsAlloc.getLoc(), slicedType, fullAlloc, rhsOffsets);
    lhsAlloc.getResult().replaceAllUsesWith(lhsSlice);
    rhsAlloc.getResult().replaceAllUsesWith(rhsSlice);
    lhsAlloc.erase();
    rhsAlloc.erase();

    if (split->use_empty())
      split.erase();
    if (trans->use_empty())
      trans.erase();
    if (reshape->use_empty())
      reshape.erase();
    if (descLoad->use_empty())
      descLoad.erase();
    if (makeDesc && makeDesc->use_empty())
      makeDesc.erase();

    transformedMMAs.insert(lhsMMA);
    transformedMMAs.insert(rhsMMA);
    LDBG("Transformed split B load for 2-CTA MMAs at "
         << lhsMMA.getLoc() << " and " << rhsMMA.getLoc());
    return success();
  }

  // Split a B tile built in registers from descriptor loads (e.g. a dtype
  // cast, a scale by an N-independent vector, a product of two tiles, or a
  // rank-3 load reshaped to 2-D). Every value on the way from the descriptor
  // loads to the local_alloc carries the N dimension; each load is replaced by
  // its CTA's half (as in transformBLoad) and the chain is rebuilt at half
  // width. Values that do not carry N (broadcast sources, splats, scalars)
  // are reused as they are.
  LogicalResult transformRegisterBuiltB(Value b, Operation *mma) {
    if (!b)
      return failure();
    ttg::MemDescTransOp memDescTrans;
    unsigned splitDim = 1;
    Value bMemDesc = b;
    if (auto transOp = bMemDesc.getDefiningOp<ttg::MemDescTransOp>()) {
      if (transOp.getOrder().size() != 2)
        return failure();
      memDescTrans = transOp;
      splitDim = transOp.getOrder()[1];
      bMemDesc = transOp.getSrc();
    }
    auto localAlloc = bMemDesc.getDefiningOp<ttg::LocalAllocOp>();
    if (!localAlloc || !localAlloc.getSrc())
      return failure();
    Value root = localAlloc.getSrc();
    int64_t fullN = cast<RankedTensorType>(root.getType()).getShape()[splitDim];
    if (fullN % 2 != 0 || fullN / 2 < 16)
      return failure();
    int64_t halfN = fullN / 2;

    // Collect the N-carrying slice in post-order (operands first), with the
    // dimension of each value that holds N.
    DenseMap<Value, unsigned> splitDimOf;
    SmallVector<Value> slice;
    SmallVector<tt::DescriptorLoadOp> loads;
    std::function<bool(Value, unsigned)> visit = [&](Value v,
                                                     unsigned d) -> bool {
      if (auto it = splitDimOf.find(v); it != splitDimOf.end())
        return it->second == d;
      auto ty = dyn_cast<RankedTensorType>(v.getType());
      Operation *op = v.getDefiningOp();
      if (!ty || d >= ty.getRank() || ty.getShape()[d] != fullN || !op ||
          op->getNumResults() != 1 ||
          !halveBlockedEncoding(ty.getEncoding(), ty.getShape(), d))
        return false;
      splitDimOf[v] = d;
      bool ok = false;
      if (auto load = dyn_cast<tt::DescriptorLoadOp>(op)) {
        auto descType = originalDescTypes.lookup(load.getDesc());
        ok = descType && succeeded(mapResultDimToDescriptor(descType, ty, d)) &&
             (load.getDesc().getDefiningOp<tt::MakeTensorDescOp>() ||
              llvm::all_of(load.getDesc().getUsers(), [&](Operation *user) {
                // A descriptor argument is retyped in place; every user must
                // be a load (checked to be split at the end of the pass).
                return isa<tt::DescriptorLoadOp>(user);
              }));
        if (ok)
          loads.push_back(load);
      } else if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
        ok = visit(cvt.getSrc(), d);
      } else if (auto trans = dyn_cast<tt::TransOp>(op)) {
        ok = visit(trans.getSrc(), trans.getOrder()[d]);
      } else if (auto reshape = dyn_cast<tt::ReshapeOp>(op)) {
        auto srcDim = mapSqueezeReshapeDim(
            reshape.getSrc().getType().getShape(), ty.getShape(), d);
        ok = !reshape.getAllowReorder() && srcDim &&
             visit(reshape.getSrc(), *srcDim);
      } else if (auto bcast = dyn_cast<tt::BroadcastOp>(op)) {
        ok = bcast.getSrc().getType().getShape()[d] == 1 ||
             visit(bcast.getSrc(), d);
      } else if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
        ok = isa<SplatElementsAttr>(cst.getValue());
      } else if (isa<tt::SplatOp>(op)) {
        ok = true;
      } else if (op->hasTrait<OpTrait::Elementwise>() &&
                 op->getNumRegions() == 0 &&
                 !isa<tt::ElementwiseInlineAsmOp>(op)) {
        ok = llvm::all_of(op->getOperands(), [&](Value operand) {
          return !isa<RankedTensorType>(operand.getType()) || visit(operand, d);
        });
      }
      if (ok)
        slice.push_back(v);
      return ok;
    };
    if (!visit(root, splitDim) || loads.empty())
      return failure();
    // A descriptor argument is retyped in place, so it can only be halved
    // along one dimension.
    DenseMap<Value, unsigned> hostDescDim;
    for (auto load : loads) {
      if (load.getDesc().getDefiningOp())
        continue;
      unsigned descDim =
          *mapResultDimToDescriptor(originalDescTypes.lookup(load.getDesc()),
                                    cast<RankedTensorType>(load.getType()),
                                    splitDimOf.lookup(load.getResult()));
      auto [it, inserted] = hostDescDim.try_emplace(load.getDesc(), descDim);
      if (!inserted && it->second != descDim)
        return failure();
    }

    MLIRContext *ctx = mma->getContext();
    auto halfType = [&](Value v) {
      auto ty = cast<RankedTensorType>(v.getType());
      unsigned d = splitDimOf.lookup(v);
      SmallVector<int64_t> shape(ty.getShape());
      shape[d] = halfN;
      return RankedTensorType::get(
          shape, ty.getElementType(),
          halveBlockedEncoding(ty.getEncoding(), shape, d));
    };
    auto castTo = [](OpBuilder &builder, Location loc, Value v,
                     RankedTensorType ty) -> Value {
      if (v.getType() == ty)
        return v;
      return ttg::ConvertLayoutOp::create(builder, loc, ty, v);
    };

    DenseMap<Value, Value> halfOf;
    DenseMap<std::pair<Value, unsigned>, Value> halfDescOf;
    for (Value v : slice) {
      Operation *op = v.getDefiningOp();
      unsigned d = splitDimOf.lookup(v);
      RankedTensorType newTy = halfType(v);
      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);
      Location loc = op->getLoc();
      Value newV;
      if (auto load = dyn_cast<tt::DescriptorLoadOp>(op)) {
        auto descType = originalDescTypes.lookup(load.getDesc());
        unsigned descDim = *mapResultDimToDescriptor(
            descType, cast<RankedTensorType>(v.getType()), d);
        Value &newDesc = halfDescOf[{load.getDesc(), descDim}];
        if (!newDesc) {
          SmallVector<int64_t> blockShape(descType.getBlockType().getShape());
          blockShape[descDim] /= 2;
          auto newDescType = tt::TensorDescType::get(
              blockShape, descType.getElementType(),
              shrinkSwizzleForShape(descType.getSharedLayout(), blockShape));
          if (auto makeDesc =
                  load.getDesc().getDefiningOp<tt::MakeTensorDescOp>()) {
            OpBuilder descBuilder(makeDesc);
            auto newMakeDesc =
                cast<tt::MakeTensorDescOp>(descBuilder.clone(*makeDesc));
            newMakeDesc.getResult().setType(newDescType);
            newDesc = newMakeDesc.getResult();
          } else {
            Value descVal = load.getDesc();
            descVal.setType(newDescType);
            if (auto funcOp = load->getParentOfType<triton::FuncOp>()) {
              auto &entryBlock = funcOp.getBlocks().front();
              SmallVector<Type> argTys(entryBlock.getArgumentTypes());
              funcOp.setFunctionType(FunctionType::get(
                  ctx, argTys, funcOp.getFunctionType().getResults()));
            }
            newDesc = descVal;
          }
        }
        builder.setInsertionPoint(load);
        auto i32Ty = builder.getI32Type();
        Value ctaRank = nvgpu::ClusterCTAIdOp::create(builder, loc, i32Ty);
        Value two = arith::ConstantIntOp::create(builder, loc, 2, 32);
        Value ctaMod2 = arith::RemSIOp::create(builder, loc, ctaRank, two);
        Value halfNVal = arith::ConstantIntOp::create(builder, loc, halfN, 32);
        Value offset = arith::MulIOp::create(builder, loc, ctaMod2, halfNVal);
        SmallVector<Value> indices(load.getIndices());
        indices[descDim] =
            arith::AddIOp::create(builder, loc, indices[descDim], offset);
        auto newLoad =
            tt::DescriptorLoadOp::create(builder, loc, newTy, newDesc, indices);
        newLoad->setAttr("two_cta_b", builder.getUnitAttr());
        newV = newLoad.getResult();
      } else if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
        newV = castTo(builder, loc, halfOf.lookup(cvt.getSrc()), newTy);
      } else if (auto trans = dyn_cast<tt::TransOp>(op)) {
        Value t = tt::TransOp::create(
            builder, loc, halfOf.lookup(trans.getSrc()), trans.getOrder());
        newV = castTo(builder, loc, t, newTy);
      } else if (auto reshape = dyn_cast<tt::ReshapeOp>(op)) {
        Value r = tt::ReshapeOp::create(builder, loc, newTy.getShape(),
                                        halfOf.lookup(reshape.getSrc()),
                                        /*allowReorder=*/false);
        newV = castTo(builder, loc, r, newTy);
      } else if (auto bcast = dyn_cast<tt::BroadcastOp>(op)) {
        Value src = bcast.getSrc();
        RankedTensorType srcTy;
        if (cast<RankedTensorType>(src.getType()).getShape()[d] != 1) {
          srcTy = halfType(src);
          src = halfOf.lookup(src);
        } else {
          auto origSrcTy = cast<RankedTensorType>(src.getType());
          srcTy = RankedTensorType::get(origSrcTy.getShape(),
                                        origSrcTy.getElementType(),
                                        newTy.getEncoding());
        }
        newV = tt::BroadcastOp::create(builder, loc, newTy,
                                       castTo(builder, loc, src, srcTy));
      } else if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
        auto splat = cast<SplatElementsAttr>(cst.getValue());
        newV = arith::ConstantOp::create(
            builder, loc, newTy,
            SplatElementsAttr::get(newTy, splat.getSplatValue<Attribute>()));
      } else if (auto splat = dyn_cast<tt::SplatOp>(op)) {
        newV = tt::SplatOp::create(builder, loc, newTy, splat.getSrc());
      } else {
        IRMapping mapping;
        for (Value operand : op->getOperands())
          if (halfOf.count(operand))
            mapping.map(operand, castTo(builder, loc, halfOf.lookup(operand),
                                        halfType(operand)));
        Operation *clone = builder.clone(*op, mapping);
        clone->getResult(0).setType(newTy);
        newV = clone->getResult(0);
      }
      halfOf[v] = newV;
    }

    // Allocate the half tile and swap it in.
    OpBuilder builder(localAlloc);
    Value allocSrc = halfOf.lookup(root);
    auto allocSrcType = cast<RankedTensorType>(allocSrc.getType());
    auto origMemDescType = cast<ttg::MemDescType>(localAlloc.getType());
    auto newMemDescType = ttg::MemDescType::get(
        allocSrcType.getShape(), allocSrcType.getElementType(),
        shrinkSwizzleForShape(origMemDescType.getEncoding(),
                              allocSrcType.getShape()),
        origMemDescType.getMemorySpace(), origMemDescType.getMutableMemory());
    auto newLocalAlloc = ttg::LocalAllocOp::create(builder, localAlloc.getLoc(),
                                                   newMemDescType, allocSrc);
    registerSplitBAllocs.insert(newLocalAlloc);
    if (memDescTrans) {
      auto newMemDescTrans = ttg::MemDescTransOp::create(
          builder, memDescTrans.getLoc(), newLocalAlloc.getResult(),
          memDescTrans.getOrder());
      newMemDescTrans->setAttrs(memDescTrans->getAttrs());
      memDescTrans.getResult().replaceAllUsesWith(newMemDescTrans.getResult());
      memDescTrans.erase();
      splitBOperands.insert(newMemDescTrans.getResult());
    } else {
      localAlloc.getResult().replaceAllUsesWith(newLocalAlloc.getResult());
      splitBOperands.insert(newLocalAlloc.getResult());
    }
    if (localAlloc.getResult().use_empty())
      localAlloc.erase();
    // Erase the full-width chain where nothing else uses it.
    for (Value v : llvm::reverse(slice))
      if (v.use_empty())
        v.getDefiningOp()->erase();
    SetVector<Operation *> oldMakeDescs;
    for (auto &[key, newDesc] : halfDescOf)
      if (auto makeDesc = key.first.getDefiningOp<tt::MakeTensorDescOp>())
        oldMakeDescs.insert(makeDesc);
    for (Operation *makeDesc : oldMakeDescs)
      if (makeDesc->use_empty())
        makeDesc->erase();

    LDBG("Transformed register-built B for 2-CTA MMA at " << mma->getLoc());
    return success();
  }

  LogicalResult transformBLoad(Value b, Operation *mma) {
    if (!b)
      return failure();
    // Trace B operand back to DescriptorLoadOp.
    FailureOr<BLoadTrace> trace = traceToDescriptorLoad(b);
    if (failed(trace))
      return failure();
    auto descLoad = trace->descLoad;
    auto localAlloc = trace->localAlloc;
    auto memDescTrans = trace->memDescTrans;
    auto trans = trace->trans;
    unsigned splitDim = trace->splitDim;

    // Use the descriptor's original type. Host-side TMA descriptor arguments
    // are mutated in-place below, so reused descriptors must not read the
    // already-halved type on later MMA users.
    auto descType = originalDescTypes.lookup(descLoad.getDesc());
    assert(descType && "expected descriptor load type to be captured before "
                       "2-CTA load transformation");
    auto blockShape = descType.getBlockType().getShape();
    auto origResultType =
        cast<RankedTensorType>(descLoad.getResult().getType());
    auto descriptorSplitDim =
        mapResultDimToDescriptor(descType, origResultType, splitDim);
    if (failed(descriptorSplitDim)) {
      LDBG("descriptor/result ranks do not have a supported unit-prefix "
           "relationship");
      return failure();
    }
    SmallVector<int64_t> newBlockShape(blockShape.begin(), blockShape.end());
    int64_t blockN = blockShape[*descriptorSplitDim];
    assert(blockN % 2 == 0 && "BLOCK_N must be even for 2-CTA B splitting");
    int64_t halfN = blockN / 2;
    newBlockShape[*descriptorSplitDim] = halfN;

    if (halfN < 16) {
      LDBG("halfN=" << halfN << " too small, skipping");
      return failure();
    }

    MLIRContext *ctx = mma->getContext();
    auto elemType = descType.getElementType();
    auto sharedLayout =
        shrinkSwizzleForShape(descType.getSharedLayout(), newBlockShape);
    auto newDescType =
        tt::TensorDescType::get(newBlockShape, elemType, sharedLayout);

    // --- Step 1: Create half-width descriptor ---
    Value newDesc;
    auto makeDesc = descLoad.getDesc().getDefiningOp<tt::MakeTensorDescOp>();
    if (makeDesc) {
      // Device-side TMA: clone MakeTensorDescOp with half-width block shape.
      OpBuilder descBuilder(makeDesc);
      IRMapping mapper;
      auto *clonedOp = descBuilder.clone(*makeDesc.getOperation(), mapper);
      auto newMakeDesc = cast<tt::MakeTensorDescOp>(clonedOp);
      newMakeDesc.getResult().setType(newDescType);
      newDesc = newMakeDesc.getResult();
    } else {
      // Host-side TMA: the descriptor is a function argument. Update its type
      // to half-width block shape. The runtime (getTensorDescMetadata +
      // fillTMADescriptorTiled) reads the block shape from the final IR type
      // and creates the CuTensorMap with the correct half-width box_dim.
      // This follows the same pattern as Data Partitioning (WSDataPartition).
      auto descVal = descLoad.getDesc();
      descVal.setType(newDescType);
      newDesc = descVal;
      // Update the function signature to match.
      if (auto funcOp = descLoad->getParentOfType<triton::FuncOp>()) {
        auto &entryBlock = funcOp.getBlocks().front();
        SmallVector<Type> argTys(entryBlock.getArgumentTypes());
        funcOp.setFunctionType(FunctionType::get(
            ctx, argTys, funcOp.getFunctionType().getResults()));
      }
    }

    LDBG("Created half-width descriptor");

    // --- Step 2: Compute CTA-based offset ---
    OpBuilder builder(descLoad);
    Location loc = descLoad.getLoc();
    auto i32Ty = builder.getI32Type();

    Value ctaRank = nvgpu::ClusterCTAIdOp::create(builder, loc, i32Ty);
    Value two = arith::ConstantIntOp::create(builder, loc, 2, 32);
    Value ctaMod2 = arith::RemSIOp::create(builder, loc, ctaRank, two);
    Value halfNVal = arith::ConstantIntOp::create(builder, loc, halfN, 32);
    Value offset = arith::MulIOp::create(builder, loc, ctaMod2, halfNVal);

    // New N-dimension index = original + CTA offset.
    SmallVector<Value> newIndices(descLoad.getIndices());
    newIndices[*descriptorSplitDim] = arith::AddIOp::create(
        builder, loc, newIndices[*descriptorSplitDim], offset);

    // --- Step 3: Create new DescriptorLoadOp with half-width result ---
    auto origEncoding =
        cast<ttg::BlockedEncodingAttr>(origResultType.getEncoding());
    SmallVector<int64_t> newResultShape(origResultType.getShape().begin(),
                                        origResultType.getShape().end());
    newResultShape[splitDim] = halfN;
    auto newEncoding =
        getCompatibleEncoding(origEncoding, newResultShape, splitDim, ctx);
    auto halfResultType =
        RankedTensorType::get(newResultShape, elemType, newEncoding);

    auto newDescLoad = tt::DescriptorLoadOp::create(
        builder, loc, halfResultType, newDesc, newIndices);
    // Mark as a 2-CTA B-operand load so WS passes can identify it
    // without complex value tracing through pipeline buffers.
    newDescLoad->setAttr("two_cta_b", builder.getUnitAttr());

    LDBG("Created half-width load");

    // --- Step 4: Create new LocalAllocOp with half-width SMEM ---
    Value allocSrc = newDescLoad.getResult();
    if (trans) {
      builder.setInsertionPoint(localAlloc);
      allocSrc = tt::TransOp::create(builder, trans.getLoc(), allocSrc,
                                     trans.getOrder());
    }

    auto origMemDescType = cast<ttg::MemDescType>(localAlloc.getType());
    auto allocSrcType = cast<RankedTensorType>(allocSrc.getType());
    auto newMemDescEncoding = shrinkSwizzleForShape(
        origMemDescType.getEncoding(), allocSrcType.getShape());
    auto newMemDescType = ttg::MemDescType::get(
        allocSrcType.getShape(), elemType, newMemDescEncoding,
        origMemDescType.getMemorySpace(), origMemDescType.getMutableMemory());

    builder.setInsertionPoint(localAlloc);
    auto newLocalAlloc = ttg::LocalAllocOp::create(builder, localAlloc.getLoc(),
                                                   newMemDescType, allocSrc);
    // A register transpose makes the alloc a generic write.
    if (trans)
      registerSplitBAllocs.insert(newLocalAlloc);

    // --- Step 5: Replace uses and clean up ---
    if (memDescTrans) {
      auto newMemDescTrans = ttg::MemDescTransOp::create(
          builder, memDescTrans.getLoc(), newLocalAlloc.getResult(),
          memDescTrans.getOrder());
      newMemDescTrans->setAttrs(memDescTrans->getAttrs());
      memDescTrans.getResult().replaceAllUsesWith(newMemDescTrans.getResult());
      memDescTrans.erase();
      splitBOperands.insert(newMemDescTrans.getResult());
      if (localAlloc.getResult().use_empty())
        localAlloc.erase();
    } else {
      localAlloc.getResult().replaceAllUsesWith(newLocalAlloc.getResult());
      splitBOperands.insert(newLocalAlloc.getResult());
      localAlloc.erase();
    }

    // Clean up old transpose if no other users.
    if (trans && trans.getResult().use_empty())
      trans.erase();

    // Clean up old descriptor_load if no other users.
    if (descLoad.getResult().use_empty())
      descLoad.erase();

    // Clean up old MakeTensorDescOp if no other users (device-side only).
    if (makeDesc && makeDesc.getResult().use_empty())
      makeDesc.erase();

    LDBG("Transformed B load for 2-CTA MMA at " << mma->getLoc());
    return success();
  }
};

} // namespace
