#include "CodePartitionUtility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-tma-store-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

void doTMAStoreLowering(triton::FuncOp &funcOp) {
  SmallVector<tt::DescriptorStoreOp> storeOps;
  funcOp.walk([&](tt::DescriptorStoreOp op) {
    // Skip stores with non-trivial reduce semantics.
    if (op.getReduceKind() != tt::DescriptorReduceKind::NONE)
      return;
    storeOps.push_back(op);
  });

  if (storeOps.empty())
    return;

  LDBG("Lowering " << storeOps.size() << " DescriptorStoreOp(s)");

  MLIRContext *ctx = funcOp.getContext();
  Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);

  for (auto storeOp : storeOps) {
    auto loc = storeOp.getLoc();
    auto asyncTaskIds = getAsyncTaskIds(storeOp);

    OpBuilderWithAsyncTaskIds builder(storeOp);
    builder.setInsertionPoint(storeOp);

    auto src = storeOp.getSrc();
    auto desc = storeOp.getDesc();
    auto tensorType = src.getType();

    // Compute shared encoding from the descriptor.
    auto encoding = ttng::getEncodingFromDescriptor(storeOp, tensorType, desc);
    ttg::MemDescType memDescType = ttg::MemDescType::get(
        tensorType.getShape(), tensorType.getElementType(), encoding,
        sharedMemorySpace, /*mutableMemory=*/true);

    // Create empty LocalAllocOp (no src) so createChannelPost can find
    // the separate LocalStoreOp as the producer.
    auto alloc = builder.create<ttg::LocalAllocOp>(loc, memDescType);

    // Copy register data → SMEM.
    builder.create<ttg::LocalStoreOp>(loc, src, alloc);

    // Fence for ordering between software write and TMA hardware read.
    builder.create<ttng::FenceAsyncSharedOp>(loc, /*bCluster=*/false);

    // Translate indices for TMA.
    auto indices = ttng::translateTMAIndices(
        builder, loc, desc.getType().getBlockType().getEncoding(),
        storeOp.getIndices());

    // Async TMA copy from local (SMEM) to global.
    builder.create<ttng::AsyncTMACopyLocalToGlobalOp>(
        loc, desc, indices, alloc, tt::EvictionPolicy::NORMAL);

    // Wait for the TMA store to complete.
    builder.create<ttng::TMAStoreWaitOp>(loc, /*pendings=*/0);

    storeOp.erase();
  }
}

} // namespace mlir
