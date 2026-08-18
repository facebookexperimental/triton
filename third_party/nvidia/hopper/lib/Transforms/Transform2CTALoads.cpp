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

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (!ttng::is2CTA(moduleOp))
      return;

    // TLX kernels manage their own 2-CTA load splitting and synchronization.
    if (moduleOp->hasAttr("tlx.has_tlx_ops"))
      return;

    originalDescTypes.clear();
    moduleOp.walk([&](tt::DescriptorLoadOp descLoad) {
      auto descType = cast<tt::TensorDescType>(descLoad.getDesc().getType());
      originalDescTypes.try_emplace(descLoad.getDesc(), descType);
    });

    // Collect 2-CTA MMA ops.
    SmallVector<ttng::TCGen5MMAOp> twoCTAMMAOps;
    moduleOp->walk([&](ttng::TCGen5MMAOp mma) {
      if (mma.getTwoCtas())
        twoCTAMMAOps.push_back(mma);
    });

    if (twoCTAMMAOps.empty())
      return;

    LDBG("Found " << twoCTAMMAOps.size() << " 2-CTA MMA ops to transform");

    DenseSet<Operation *> splitTransformedMMAs;
    for (auto mma : twoCTAMMAOps) {
      if (splitTransformedMMAs.contains(mma))
        continue;
      if (succeeded(transformSplitBLoad(mma, splitTransformedMMAs)))
        continue;
      if (failed(transformBLoad(mma)))
        LDBG("Skipped MMA at " << mma.getLoc()
                               << " (B not from descriptor load)");
    }

    // A 2-CTA TMA load is issued by both CTAs as one hardware CTA-group
    // transaction. Mark every rank-2 descriptor load in the cooperative
    // kernel, including A operands (K/V) that are not visited by B splitting.
    // Rank-1 metadata loads remain ordinary per-CTA loads, matching TLX's raw
    // M/D bulk-copy path.
    moduleOp.walk([&](tt::DescriptorLoadOp descLoad) {
      auto resultTy = dyn_cast<RankedTensorType>(descLoad.getType());
      if (resultTy && resultTy.getRank() == 2)
        descLoad->setAttr("two_cta_load", UnitAttr::get(moduleOp.getContext()));
    });
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

  LogicalResult transformBLoad(ttng::TCGen5MMAOp mma) {
    // Trace B operand back to DescriptorLoadOp.
    FailureOr<BLoadTrace> trace = traceToDescriptorLoad(mma.getB());
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
    assert(blockShape.size() == 2 && "Expected 2D block shape");
    SmallVector<int64_t> newBlockShape(blockShape.begin(), blockShape.end());
    int64_t blockN = blockShape[splitDim];
    assert(blockN % 2 == 0 && "BLOCK_N must be even for 2-CTA B splitting");
    int64_t halfN = blockN / 2;
    newBlockShape[splitDim] = halfN;

    if (halfN < 16) {
      LDBG("halfN=" << halfN << " too small, skipping");
      return failure();
    }

    MLIRContext *ctx = mma.getContext();
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
    newIndices[splitDim] =
        arith::AddIOp::create(builder, loc, newIndices[splitDim], offset);

    // --- Step 3: Create new DescriptorLoadOp with half-width result ---
    auto origResultType =
        cast<RankedTensorType>(descLoad.getResult().getType());
    auto origEncoding =
        cast<ttg::BlockedEncodingAttr>(origResultType.getEncoding());
    auto newEncoding =
        getCompatibleEncoding(origEncoding, newBlockShape, splitDim, ctx);
    auto halfResultType =
        RankedTensorType::get(newBlockShape, elemType, newEncoding);

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

    // --- Step 5: Replace uses and clean up ---
    if (memDescTrans) {
      auto newMemDescTrans = ttg::MemDescTransOp::create(
          builder, memDescTrans.getLoc(), newLocalAlloc.getResult(),
          memDescTrans.getOrder());
      newMemDescTrans->setAttrs(memDescTrans->getAttrs());
      memDescTrans.getResult().replaceAllUsesWith(newMemDescTrans.getResult());
      memDescTrans.erase();
      if (localAlloc.getResult().use_empty())
        localAlloc.erase();
    } else {
      localAlloc.getResult().replaceAllUsesWith(newLocalAlloc.getResult());
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

    LDBG("Transformed B load for 2-CTA MMA at " << mma.getLoc());
    return success();
  }
};

} // namespace
