#include "IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
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

#define GEN_PASS_DEF_TLXSTORAGEALIASALLOCATION

#include "tlx/dialect/include/Transforms/Passes.h.inc"

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

    auto bufferSizeAttr = specOp.getBufferSizeBytesAttr();
    if (!bufferSizeAttr) {
      specOp.emitError("storage_alias_spec has no size set; "
                       "run TLXStorageAliasSizeDefinition pass first");
      return failure();
    }

    int64_t bufferSizeBytes = bufferSizeAttr.getInt();
    auto storage = specOp.getStorage();

    LDBG("Creating allocation for storage_alias_spec with size "
         << bufferSizeBytes);

    // Create a 1D byte buffer type for the allocation
    auto elemType = IntegerType::get(m.getContext(), 8);
    SmallVector<int64_t> shape{bufferSizeBytes};

    // Create a shared encoding with default parameters
    auto ctaLayout = ttg::CTALayoutAttr::get(m.getContext(), {1}, {1}, {0});
    auto sharedEncoding = ttg::SwizzledSharedEncodingAttr::get(
        m.getContext(), /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1,
        /*order=*/{0}, ctaLayout);

    Value allocResult;
    if (storage == StorageKind::smem) {
      // Create LocalAllocOp for shared memory
      auto memorySpace = ttg::SharedMemorySpaceAttr::get(m.getContext());
      auto memDescType =
          ttg::MemDescType::get(shape, elemType, sharedEncoding, memorySpace,
                                /*mutableMemory=*/true);
      auto allocOp =
          builder.create<ttg::LocalAllocOp>(specOp.getLoc(), memDescType);
      allocResult = allocOp.getResult();
    } else {
      // Create TMEMAllocOp for tensor memory
      // Follow TLX make_default pattern: use shape from the largest allocation
      // referencing this spec to determine block dimensions
      assert(storage == StorageKind::tmem && "Unexpected storage kind");

      // Find the shape from the single largest allocation (by total size)
      // to determine the TMEM block dimensions (following make_default pattern)
      // Note: For 3D+ allocations, we flatten all dimensions except the last two
      // into a single "buffer count" and multiply it into the allocation size
      int64_t maxBlockM = 0;
      int64_t maxBlockN = 0;
      int64_t largestAllocSize = 0;
      for (auto user : specOp.getResult().getUsers()) {
        if (auto allocOp = dyn_cast<StorageAliasLocalAllocOp>(user)) {
          auto resultType = cast<ttg::MemDescType>(allocOp.getType());
          auto allocShape = resultType.getShape();
          // Calculate total allocation size in bytes
          int64_t allocSize = 1;
          for (int64_t dim : allocShape) {
            allocSize *= dim;
          }
          auto elemType = resultType.getElementType();
          int64_t elemBits = elemType.getIntOrFloatBitWidth();
          allocSize *= (elemBits + 7) / 8;

          if (allocSize > largestAllocSize) {
            largestAllocSize = allocSize;
            // Use last two dimensions as blockM and blockN (standard 2D layout)
            // For 3D+ allocations, the leading dimensions are buffer counts
            // which will be handled by using the full bufferSizeBytes
            maxBlockM =
                allocShape.size() >= 2 ? allocShape[allocShape.size() - 2] : 1;
            maxBlockN =
                allocShape.size() >= 1 ? allocShape[allocShape.size() - 1] : 1;
          }
        }
      }

      // For TMEM, we need to ensure the allocation is large enough for ALL users
      // The bufferSizeBytes already accounts for the largest user, so we compute
      // blockN to ensure blockM × blockN × sizeof(i32) >= bufferSizeBytes
      int64_t i32SizeBytes = 4;
      int64_t requiredElements = (bufferSizeBytes + i32SizeBytes - 1) / i32SizeBytes;

      // Adjust blockN to ensure we have enough total elements
      // blockM is kept from the largest allocation, blockN is scaled up as needed
      if (maxBlockM > 0) {
        int64_t minBlockN = (requiredElements + maxBlockM - 1) / maxBlockM;
        if (minBlockN > maxBlockN) {
          maxBlockN = minBlockN;
        }
      }

      // Validate block dimensions meet TMEM constraints
      // blockM must be 64 or 128, blockN must be power of 2
      if (maxBlockM != 64 && maxBlockM != 128) {
        // Round up to nearest valid blockM
        maxBlockM = maxBlockM <= 64 ? 64 : 128;
      }

      // Use i32 (4 bytes) with unpacked=true (following make_default pattern)
      auto tmemElemType = IntegerType::get(m.getContext(), 32);
      SmallVector<int64_t> tmemShape{maxBlockM, maxBlockN};
      auto memorySpace = ttng::TensorMemorySpaceAttr::get(m.getContext());
      auto tmemEncoding = ttng::TensorMemoryEncodingAttr::get(
          m.getContext(), maxBlockM, maxBlockN,
          /*unpacked=*/true, /*CTASplitM=*/1, /*CTASplitN=*/1);
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

struct TLXStorageAliasAllocationPass
    : public impl::TLXStorageAliasAllocationBase<
          TLXStorageAliasAllocationPass> {
public:
  using impl::TLXStorageAliasAllocationBase<
      TLXStorageAliasAllocationPass>::TLXStorageAliasAllocationBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (failed(materializeStorageAliasAllocations(m))) {
      signalPassFailure();
    }
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
