//===- Transform2CTALoads.cpp - Transform loads for 2-CTA mode -----------===//
//
// This pass transforms B matrix loads for 2-CTA (pair CTA) mode.
// Each CTA loads BLOCK_N/2 columns of B with an offset based on CTA rank.
//
// The pass:
// 1. Updates the B TensorDescType block shape from [K, N] to [K, N/2]
// 2. Adds CTA rank-based offset to the B load indices
// 3. Updates the B load result type to [K, N/2]
//
// Note: The accumulator and output remain at full [M, N] size because
// TMEM allocation requires at least 128x128. The 2-CTA MMA instruction
// internally handles the coordination between the two CTAs.
//
// NOTE: This pass is skipped when warp specialization is enabled because the
// WS passes restructure the IR in ways that are incompatible with 2-CTA sync.
// For WS+2CTA, use TLX which has native support.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

/// Check if warp specialization is enabled on any loop
static bool isWarpSpecializationEnabled(ModuleOp moduleOp) {
  bool wsEnabled = false;
  moduleOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("tt.warp_specialize")) {
      wsEnabled = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return wsEnabled;
}

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

/// Get the local_alloc that uses the B load result
static ttg::LocalAllocOp findBLocalAlloc(ttng::TCGen5MMAOp mmaOp) {
  Value bOp = mmaOp.getB();
  if (auto localAlloc = bOp.getDefiningOp<ttg::LocalAllocOp>()) {
    return localAlloc;
  }
  return nullptr;
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

/// Create a new TensorDescType with halved N dimension
static tt::TensorDescType createHalfNDescType(tt::TensorDescType descType) {
  auto blockType = descType.getBlockType();
  SmallVector<int64_t> shape(blockType.getShape());
  // Halve the last dimension (N)
  shape.back() /= 2;
  auto newBlockType = RankedTensorType::get(
      shape, blockType.getElementType(), blockType.getEncoding());
  return tt::TensorDescType::get(descType.getContext(), newBlockType);
}

/// Create a new RankedTensorType with halved N dimension
static RankedTensorType createHalfNTensorType(RankedTensorType tensorType) {
  SmallVector<int64_t> shape(tensorType.getShape());
  // Halve the last dimension (N)
  shape.back() /= 2;
  return RankedTensorType::get(shape, tensorType.getElementType(),
                               tensorType.getEncoding());
}

/// Create a new MemDescType with halved N dimension
static ttg::MemDescType createHalfNMemDescType(ttg::MemDescType memDescType) {
  SmallVector<int64_t> shape(memDescType.getShape());
  // Halve the last dimension (N)
  shape.back() /= 2;
  return ttg::MemDescType::get(shape, memDescType.getElementType(),
                               memDescType.getEncoding(),
                               memDescType.getMemorySpace(),
                               memDescType.getMutableMemory());
}

struct Transform2CTALoadsPass
    : public mlir::impl::NVGPU2CTATransformLoadsBase<Transform2CTALoadsPass> {
  using NVGPU2CTATransformLoadsBase::NVGPU2CTATransformLoadsBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Skip if warp specialization is enabled
    // WS passes restructure the IR in ways that are incompatible with 2-CTA sync.
    // For WS+2CTA, use TLX which has native support.
    if (isWarpSpecializationEnabled(moduleOp)) {
      LDBG("Warp specialization enabled (tt.warp_specialize attr found)");
      LDBG("Skipping 2-CTA load transformation - use TLX for WS+2CTA");
      return;
    }

    // Find all 2-CTA MMA operations
    SmallVector<ttng::TCGen5MMAOp> mmaOps = find2CTAMMAOps(moduleOp);

    if (mmaOps.empty()) {
      LDBG("No 2-CTA MMA operations found, nothing to do");
      return;
    }

    LDBG("Found " << mmaOps.size() << " 2-CTA MMA operations");

    // Track which descriptor args have been modified
    DenseSet<unsigned> modifiedDescArgs;
    // Track which loads have been modified
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

      // Get the descriptor (should be a function argument)
      Value desc = bLoadOp.getDesc();
      auto descType = dyn_cast<tt::TensorDescType>(desc.getType());
      if (!descType) {
        LDBG("Descriptor is not TensorDescType");
        continue;
      }

      // Get block shape
      auto blockType = descType.getBlockType();
      auto shape = blockType.getShape();
      if (shape.size() != 2) {
        LDBG("B descriptor block shape is not 2D");
        continue;
      }

      int64_t blockK = shape[0];
      int64_t blockN = shape[1];

      LDBG("B descriptor block shape: [" << blockK << ", " << blockN << "]");

      // Get or create ClusterCTAIdOp
      OpBuilder builder(moduleOp.getContext());
      Value ctaRank = getOrCreateClusterCTAId(funcOp, builder);

      int64_t halfBlockN = blockN / 2;

      // Update B descriptor argument type if it's a function argument
      if (auto bbArg = dyn_cast<BlockArgument>(desc)) {
        unsigned argIndex = bbArg.getArgNumber();
        if (!modifiedDescArgs.contains(argIndex)) {
          // Create new descriptor type with halved N
          auto newDescType = createHalfNDescType(descType);
          bbArg.setType(newDescType);
          modifiedDescArgs.insert(argIndex);

          LDBG("Updated B descriptor arg " << argIndex << " type to ["
                                           << blockK << ", " << halfBlockN
                                           << "]");
        }
      }

      // Compute CTA offset for B load
      builder.setInsertionPoint(bLoadOp);
      Location loc = bLoadOp.getLoc();

      // cta_rank % 2 (get whether CTA0 or CTA1 in the pair)
      Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
      Value ctaInPair = builder.create<arith::RemSIOp>(loc, ctaRank, two);

      // BLOCK_N / 2
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

      // Create new result type with halved N
      auto oldResultType =
          cast<RankedTensorType>(bLoadOp.getResult().getType());
      auto newResultType = createHalfNTensorType(oldResultType);

      // Create new descriptor load with modified indices and result type
      auto newLoad = builder.create<tt::DescriptorLoadOp>(
          loc, newResultType, bLoadOp.getDesc(), indices);

      // Update the local_alloc that uses this load
      auto localAlloc = findBLocalAlloc(mmaOp);
      if (localAlloc && localAlloc.getSrc() == bLoadOp.getResult()) {
        // Create new local_alloc with updated types
        auto oldMemDescType = cast<ttg::MemDescType>(localAlloc.getType());
        auto newMemDescType = createHalfNMemDescType(oldMemDescType);

        builder.setInsertionPointAfter(newLoad);
        auto newLocalAlloc = builder.create<ttg::LocalAllocOp>(
            localAlloc.getLoc(), newMemDescType, newLoad.getResult());

        // Update the MMA op's B operand
        mmaOp.getBMutable().assign(newLocalAlloc.getResult());

        // Remove old local_alloc
        localAlloc.erase();

        LDBG("Updated B local_alloc type to [" << blockK << ", " << halfBlockN
                                               << "]");
      } else {
        // Just replace the load result uses
        bLoadOp.getResult().replaceAllUsesWith(newLoad.getResult());
      }

      // Remove old load
      bLoadOp.erase();

      modifiedLoads.insert(newLoad);

      LDBG("Transformed B load for 2-CTA mode:");
      LDBG("  BLOCK_N = " << blockN << " -> " << halfBlockN);
    }

    // Update function signature to match the new argument types
    for (auto funcOp : moduleOp.getOps<tt::FuncOp>()) {
      auto &entryBlock = funcOp.getBlocks().front();
      SmallVector<Type> argTys(entryBlock.getArgumentTypes());
      funcOp.setFunctionType(FunctionType::get(
          funcOp.getContext(), argTys,
          funcOp.getFunctionType().getResults()));
    }

    LDBG("Finished transforming " << modifiedLoads.size() << " B loads");
    LDBG("Modified " << modifiedDescArgs.size() << " descriptor arguments");
  }
};

} // namespace
