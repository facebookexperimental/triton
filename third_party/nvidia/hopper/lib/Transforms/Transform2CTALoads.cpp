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

    for (auto mma : twoCTAMMAOps) {
      if (failed(transformBLoad(mma)))
        LDBG("Skipped MMA at " << mma.getLoc()
                               << " (B not from descriptor load)");
    }
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
    auto elemType = descType.getBlockType().getElementType();
    auto blockEncoding = descType.getBlockType().getEncoding();
    auto halfBlockType =
        RankedTensorType::get(newBlockShape, elemType, blockEncoding);
    auto newDescType = tt::TensorDescType::get(ctx, halfBlockType);

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
    auto newMemDescType = ttg::MemDescType::get(
        allocSrcType.getShape(), elemType, origMemDescType.getEncoding(),
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
