#include "IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tlx-storage-alias-allocation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace tlx {

LogicalResult materializeStorageAliasAllocations(ModuleOp m) {
  LDBG("materializeStorageAliasAllocations");

  OpBuilder builder(m.getContext());

  // Map from storage_alias_spec SSA value to its materialized allocation
  DenseMap<Value, Value> specToAlloc;

  // Collect all storage_alias_spec operations
  SmallVector<StorageAliasSpecOp> specOps;
  m.walk([&](StorageAliasSpecOp specOp) { specOps.push_back(specOp); });

  // First pass: create LocalAllocOp/TMEMAllocOp for each storage_alias_spec
  for (auto specOp : specOps) {
    builder.setInsertionPoint(specOp);

    auto bufferShapeAttr = specOp.getBufferShapeAttr();
    if (!bufferShapeAttr) {
      specOp.emitError("storage_alias_spec has no shape set; "
                       "run TLXStorageAliasSizeDefinition pass first");
      return failure();
    }

    auto bufferShape = bufferShapeAttr.asArrayRef();
    auto storage = specOp.getStorage();

    Value allocResult;
    if (storage == StorageKind::smem) {
      // SMEM: 1D allocation
      if (bufferShape.size() != 1) {
        specOp.emitError("SMEM storage_alias_spec must have 1D shape, got ")
            << bufferShape.size() << "D";
        return failure();
      }

      int64_t bufferSizeBytes = bufferShape[0];
      LDBG("Creating SMEM allocation with size " << bufferSizeBytes);

      // Create a 1D byte buffer type for the allocation
      auto elemType = IntegerType::get(m.getContext(), 8);
      SmallVector<int64_t> shape{bufferSizeBytes};

      // Create a shared encoding with default parameters
      auto ctaLayout = ttg::CTALayoutAttr::get(m.getContext(), {1}, {1}, {0});
      auto sharedEncoding = ttg::SwizzledSharedEncodingAttr::get(
          m.getContext(), /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1,
          /*order=*/{0}, ctaLayout);

      auto memorySpace = ttg::SharedMemorySpaceAttr::get(m.getContext());
      auto memDescType =
          ttg::MemDescType::get(shape, elemType, sharedEncoding, memorySpace,
                                /*mutableMemory=*/true);
      auto allocOp =
          builder.create<ttg::LocalAllocOp>(specOp.getLoc(), memDescType);
      allocResult = allocOp.getResult();
    } else {
      // TMEM: 2D allocation
      assert(storage == StorageKind::tmem && "Unexpected storage kind");

      if (bufferShape.size() != 2) {
        specOp.emitError("TMEM storage_alias_spec must have 2D shape, got ")
            << bufferShape.size() << "D";
        return failure();
      }

      int64_t blockM = bufferShape[0];
      int64_t blockN = bufferShape[1];
      LDBG("Creating TMEM allocation with shape [" << blockM << ", " << blockN
                                                   << "]");

      auto tmemElemType = IntegerType::get(m.getContext(), 32);
      SmallVector<int64_t> tmemShape{blockM, blockN};
      auto memorySpace = ttng::TensorMemorySpaceAttr::get(m.getContext());
      auto tmemEncoding = ttng::TensorMemoryEncodingAttr::get(
          m.getContext(), blockM, blockN,
          /*colStride=*/1, /*CTASplitM=*/1, /*CTASplitN=*/1);
      auto memDescType =
          ttg::MemDescType::get(tmemShape, tmemElemType, tmemEncoding,
                                memorySpace, /*mutableMemory=*/true);
      auto allocOp = builder.create<ttng::TMEMAllocOp>(specOp.getLoc(),
                                                       memDescType, nullptr);
      allocResult = allocOp.getResult();
    }

    specToAlloc[specOp.getResult()] = allocResult;
  }

  // Second pass: replace storage_alias_local_alloc with LocalAliasOp
  SmallVector<StorageAliasLocalAllocOp> allocOpsToReplace;
  m.walk([&](StorageAliasLocalAllocOp allocOp) {
    allocOpsToReplace.push_back(allocOp);
  });

  for (auto allocOp : allocOpsToReplace) {
    Value storageAlias = allocOp.getStorageAlias();
    auto it = specToAlloc.find(storageAlias);
    if (it == specToAlloc.end()) {
      allocOp.emitError("storage_alias_spec not found for this allocation");
      return failure();
    }

    LDBG("Replacing storage_alias_local_alloc with LocalAliasOp");

    builder.setInsertionPoint(allocOp);

    // Create a LocalAliasOp to reinterpret the allocation with the correct type
    auto resultType = allocOp.getResult().getType();
    auto localAliasOp =
        builder.create<LocalAliasOp>(allocOp.getLoc(), resultType, it->second);

    // Replace all uses and erase the old operation
    allocOp.getResult().replaceAllUsesWith(localAliasOp.getResult());
    allocOp.erase();
  }

  // Third pass: erase storage_alias_spec operations
  for (auto specOp : specOps) {
    // Check if the spec still has uses (it shouldn't at this point)
    if (!specOp.getResult().use_empty()) {
      specOp.emitError(
          "storage_alias_spec still has uses after allocation materialization");
      return failure();
    }
    specOp.erase();
  }

  return success();
}

} // namespace tlx
} // namespace triton
} // namespace mlir
