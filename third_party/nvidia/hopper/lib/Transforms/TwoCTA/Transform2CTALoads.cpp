//===- Transform2CTALoads.cpp - Transform loads for 2-CTA mode -----------===//
//
// This pass transforms B matrix loads for 2-CTA (pair CTA) mode.
// Each CTA loads BLOCK_N/2 columns of B with an offset based on CTA rank.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

// Define before including Passes.h to get the base class
#define GEN_PASS_DEF_NVGPU2CTATRANSFORMLOADS
#include "nvidia/hopper/include/Transforms/Passes.h"

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace nvgpu = triton::nvgpu;

#define DEBUG_TYPE "nvgpu-2cta-transform-loads"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// Find all TCGen5MMAOp operations with two_ctas=true
static SmallVector<ttng::TCGen5MMAOp> find2CTAMMAOps(ModuleOp moduleOp) {
  SmallVector<ttng::TCGen5MMAOp> result;
  moduleOp.walk([&](ttng::TCGen5MMAOp mmaOp) {
    if (mmaOp.getTwoCtas()) {
      result.push_back(mmaOp);
    }
  });
  return result;
}

/// Get the defining operation for B operand, tracing through local_alloc
static Operation *getBDefiningOp(ttng::TCGen5MMAOp mmaOp) {
  Value bOp = mmaOp.getB();
  // B is a memdesc, typically from ttg.local_alloc
  if (auto localAlloc = bOp.getDefiningOp<ttg::LocalAllocOp>()) {
    // The source of local_alloc is typically a load
    return localAlloc.getSrc().getDefiningOp();
  }
  return bOp.getDefiningOp();
}

/// Find the descriptor load that feeds into B operand
static tt::DescriptorLoadOp findBDescriptorLoad(ttng::TCGen5MMAOp mmaOp) {
  Operation *defOp = getBDefiningOp(mmaOp);
  if (auto descLoad = dyn_cast_or_null<tt::DescriptorLoadOp>(defOp)) {
    return descLoad;
  }
  return nullptr;
}

/// Get the N dimension from B's type (B is K x N)
static int64_t getBLoadN(tt::DescriptorLoadOp loadOp) {
  auto resultType = loadOp.getResult().getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
    // B is K x N, so N is the second dimension
    return tensorType.getShape()[1];
  }
  return 0;
}

/// Insert ClusterCTAIdOp at the function entry if not already present
static Value getOrCreateClusterCTAId(tt::FuncOp funcOp, OpBuilder &builder) {
  // Check if we already have a ClusterCTAIdOp at the entry
  Block &entryBlock = funcOp.getBody().front();
  for (auto &op : entryBlock) {
    if (auto ctaIdOp = dyn_cast<nvgpu::ClusterCTAIdOp>(&op)) {
      return ctaIdOp.getResult();
    }
  }

  // Insert at the beginning of the function
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entryBlock);
  auto ctaIdOp = builder.create<nvgpu::ClusterCTAIdOp>(funcOp.getLoc());
  return ctaIdOp.getResult();
}

struct Transform2CTALoadsPass
    : public mlir::impl::NVGPU2CTATransformLoadsBase<Transform2CTALoadsPass> {
  using NVGPU2CTATransformLoadsBase::NVGPU2CTATransformLoadsBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Find all 2-CTA MMA operations
    SmallVector<ttng::TCGen5MMAOp> mmaOps = find2CTAMMAOps(moduleOp);

    if (mmaOps.empty()) {
      LDBG("No 2-CTA MMA operations found, nothing to do");
      return;
    }

    LDBG("Found " << mmaOps.size() << " 2-CTA MMA operations");

    // Track which descriptor loads have been modified to avoid double-modification
    DenseSet<Operation *> modifiedLoads;

    for (auto mmaOp : mmaOps) {
      // Find the B descriptor load
      auto bLoadOp = findBDescriptorLoad(mmaOp);
      if (!bLoadOp) {
        LDBG("Could not find B descriptor load for MMA op");
        continue;
      }

      // Skip if already modified
      if (modifiedLoads.contains(bLoadOp)) {
        LDBG("B load already modified, skipping");
        continue;
      }

      // Get the containing function
      auto funcOp = mmaOp->getParentOfType<tt::FuncOp>();
      if (!funcOp) {
        LDBG("MMA op not inside a function");
        continue;
      }

      // Get or create ClusterCTAIdOp
      OpBuilder builder(moduleOp.getContext());
      Value ctaRank = getOrCreateClusterCTAId(funcOp, builder);

      // Get the N dimension of B load
      int64_t blockN = getBLoadN(bLoadOp);
      if (blockN == 0) {
        LDBG("Could not determine B load N dimension");
        continue;
      }

      // For 2-CTA mode, we need to:
      // 1. Compute offset: (cta_rank % 2) * (BLOCK_N / 2)
      // 2. Add this offset to the N index of the descriptor load

      builder.setInsertionPoint(bLoadOp);
      Location loc = bLoadOp.getLoc();

      // cta_rank % 2 (get whether CTA0 or CTA1 in the pair)
      Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
      Value ctaInPair = builder.create<arith::RemSIOp>(loc, ctaRank, two);

      // BLOCK_N / 2
      int64_t halfBlockN = blockN / 2;
      Value halfN = builder.create<arith::ConstantIntOp>(loc, halfBlockN, 32);

      // offset = (cta_rank % 2) * (BLOCK_N / 2)
      Value offset = builder.create<arith::MulIOp>(loc, ctaInPair, halfN);

      // Get the current indices (K index and N index)
      SmallVector<Value> indices(bLoadOp.getIndices());
      if (indices.size() != 2) {
        LDBG("B load doesn't have 2 indices, unexpected shape");
        continue;
      }

      // Add offset to the N index (second index for B which is K x N)
      Value newNIndex = builder.create<arith::AddIOp>(loc, indices[1], offset);
      indices[1] = newNIndex;

      // Create new descriptor load with modified indices
      // Note: The descriptor's block_shape should already be [BLOCK_K, BLOCK_N/2]
      // as set by the frontend
      auto newLoad = builder.create<tt::DescriptorLoadOp>(
          loc, bLoadOp.getResult().getType(), bLoadOp.getDesc(), indices);

      // Replace uses of old load with new load
      bLoadOp.getResult().replaceAllUsesWith(newLoad.getResult());
      bLoadOp.erase();

      modifiedLoads.insert(newLoad);

      LDBG("Transformed B load for 2-CTA mode with offset calculation");
      LDBG("  BLOCK_N = " << blockN << ", offset = (cta_rank % 2) * "
                          << halfBlockN);
    }

    LDBG("Finished transforming " << modifiedLoads.size() << " B loads");
  }
};

} // namespace
