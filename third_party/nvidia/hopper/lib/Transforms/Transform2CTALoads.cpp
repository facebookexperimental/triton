// Transform B matrix descriptor loads for 2-CTA MMA operations.
//
// When TCGen5MMAOp has two_ctas=true and is not async (i.e., pure Triton,
// not TLX), this pass splits B loads so each CTA loads half of B:
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
getCompatibleEncoding(ttg::BlockedEncodingAttr origEncoding, int64_t M,
                      int64_t N, MLIRContext *ctx) {
  auto spt = SmallVector<unsigned>(origEncoding.getSizePerThread());
  auto tpw = SmallVector<unsigned>(origEncoding.getThreadsPerWarp());
  auto wpc = SmallVector<unsigned>(origEncoding.getWarpsPerCTA());
  auto order = SmallVector<unsigned>(origEncoding.getOrder());
  auto ctaLayout = origEncoding.getCGALayout();

  unsigned tileN = spt[1] * tpw[1] * wpc[1];
  if (M % (spt[0] * tpw[0] * wpc[0]) == 0 && N % tileN == 0)
    return origEncoding;

  // Reduce N-tile by halving threadsPerWarp[1] and doubling
  // threadsPerWarp[0] to keep total threads = 32.
  while (spt[1] * tpw[1] * wpc[1] > static_cast<unsigned>(N) && tpw[1] > 1) {
    tpw[1] /= 2;
    tpw[0] *= 2;
  }
  // If still too large, reduce sizePerThread[1].
  while (spt[1] * tpw[1] * wpc[1] > static_cast<unsigned>(N) && spt[1] > 1) {
    spt[1] /= 2;
  }

  return ttg::BlockedEncodingAttr::get(ctx, spt, tpw, wpc, order, ctaLayout);
}

// Trace B operand from MMA back through LocalAllocOp and ConvertLayoutOps
// to find the DescriptorLoadOp.
static tt::DescriptorLoadOp
traceToDescriptorLoad(Value bMemDesc, ttg::LocalAllocOp &outLocalAlloc) {
  auto localAlloc = bMemDesc.getDefiningOp<ttg::LocalAllocOp>();
  if (!localAlloc)
    return nullptr;
  outLocalAlloc = localAlloc;

  Value tensor = localAlloc.getSrc();
  // Skip convert_layout ops.
  while (auto cvt = tensor.getDefiningOp<ttg::ConvertLayoutOp>())
    tensor = cvt.getSrc();

  return tensor.getDefiningOp<tt::DescriptorLoadOp>();
}

struct Transform2CTALoads
    : public impl::NVGPU2CTATransformLoadsBase<Transform2CTALoads> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Check if any cluster dimension >= 2.
    bool hasCluster = false;
    for (auto attr :
         {"ttg.cluster-dim-x", "ttg.cluster-dim-y", "ttg.cluster-dim-z"}) {
      if (auto intAttr = moduleOp->getAttrOfType<IntegerAttr>(attr)) {
        if (intAttr.getInt() >= 2)
          hasCluster = true;
      }
    }
    if (!hasCluster)
      return;

    // Collect 2-CTA MMA ops (skip async/TLX-managed).
    SmallVector<ttng::TCGen5MMAOp> twoCTAMMAOps;
    moduleOp->walk([&](ttng::TCGen5MMAOp mma) {
      if (mma.getTwoCtas() && !mma.getIsAsync())
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
    ttg::LocalAllocOp localAlloc;
    auto descLoad = traceToDescriptorLoad(mma.getB(), localAlloc);
    if (!descLoad)
      return failure();

    // Find the MakeTensorDescOp that created the descriptor.
    auto makeDesc = descLoad.getDesc().getDefiningOp<tt::MakeTensorDescOp>();
    if (!makeDesc) {
      LDBG("B descriptor is not from MakeTensorDescOp, skipping");
      return failure();
    }

    // Get block shape from descriptor type.
    auto descType = cast<tt::TensorDescType>(makeDesc.getType());
    auto blockShape = descType.getBlockType().getShape();
    assert(blockShape.size() == 2 && "Expected 2D block shape");
    int64_t blockK = blockShape[0];
    int64_t blockN = blockShape[1];
    assert(blockN % 2 == 0 && "BLOCK_N must be even for 2-CTA B splitting");
    int64_t halfN = blockN / 2;

    if (halfN < 16) {
      LDBG("halfN=" << halfN << " too small, skipping");
      return failure();
    }

    MLIRContext *ctx = mma.getContext();
    auto elemType = descType.getBlockType().getElementType();

    // --- Step 1: Clone MakeTensorDescOp with half-width block shape ---
    OpBuilder builder(makeDesc);
    IRMapping mapper;
    auto *clonedOp = builder.clone(*makeDesc.getOperation(), mapper);
    auto newMakeDesc = cast<tt::MakeTensorDescOp>(clonedOp);

    // Change result type to half-width descriptor.
    auto halfBlockType = RankedTensorType::get({blockK, halfN}, elemType);
    auto newDescType = tt::TensorDescType::get(ctx, halfBlockType);
    newMakeDesc.getResult().setType(newDescType);

    LDBG("Created half-width descriptor: " << blockK << "x" << halfN);

    // --- Step 2: Compute CTA-based offset ---
    builder.setInsertionPoint(descLoad);
    Location loc = descLoad.getLoc();
    auto i32Ty = builder.getI32Type();

    Value ctaRank = builder.create<nvgpu::ClusterCTAIdOp>(loc, i32Ty);
    Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
    Value ctaMod2 = builder.create<arith::RemSIOp>(loc, ctaRank, two);
    Value halfNVal = builder.create<arith::ConstantIntOp>(loc, halfN, 32);
    Value offset = builder.create<arith::MulIOp>(loc, ctaMod2, halfNVal);

    // New N-dimension index = original + CTA offset.
    SmallVector<Value> newIndices(descLoad.getIndices());
    newIndices.back() =
        builder.create<arith::AddIOp>(loc, newIndices.back(), offset);

    // --- Step 3: Create new DescriptorLoadOp with half-width result ---
    auto origResultType =
        cast<RankedTensorType>(descLoad.getResult().getType());
    auto origEncoding =
        cast<ttg::BlockedEncodingAttr>(origResultType.getEncoding());
    auto newEncoding = getCompatibleEncoding(origEncoding, blockK, halfN, ctx);
    auto halfResultType =
        RankedTensorType::get({blockK, halfN}, elemType, newEncoding);

    auto newDescLoad = builder.create<tt::DescriptorLoadOp>(
        loc, halfResultType, newMakeDesc.getResult(), newIndices);
    // Mark as a 2-CTA B-operand load so WS passes can identify it
    // without complex value tracing through pipeline buffers.
    newDescLoad->setAttr("two_cta_b", builder.getUnitAttr());

    LDBG("Created half-width load: " << blockK << "x" << halfN);

    // --- Step 4: Create new LocalAllocOp with half-width SMEM ---
    auto origMemDescType = cast<ttg::MemDescType>(localAlloc.getType());
    auto newMemDescType = ttg::MemDescType::get(
        {blockK, halfN}, elemType, origMemDescType.getEncoding(),
        origMemDescType.getMemorySpace(), origMemDescType.getMutableMemory());

    builder.setInsertionPoint(localAlloc);
    auto newLocalAlloc = builder.create<ttg::LocalAllocOp>(
        localAlloc.getLoc(), newMemDescType, newDescLoad.getResult());

    // --- Step 5: Replace uses and clean up ---
    localAlloc.getResult().replaceAllUsesWith(newLocalAlloc.getResult());
    localAlloc.erase();

    // Clean up old descriptor_load if no other users.
    if (descLoad.getResult().use_empty())
      descLoad.erase();

    // Clean up old MakeTensorDescOp if no other users.
    if (makeDesc.getResult().use_empty())
      makeDesc.erase();

    LDBG("Transformed B load for 2-CTA MMA at " << mma.getLoc());
    return success();
  }
};

} // namespace
