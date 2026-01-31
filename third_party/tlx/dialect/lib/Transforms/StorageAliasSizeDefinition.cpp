#include "IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tlx-storage-alias-size-definition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace tlx {

// Helper to get element type bit width, handling pointer types specially
static int64_t getElementBitWidth(ttg::MemDescType memDescType) {
  Type elemType = memDescType.getElementType();
  // Pointer types are 64-bit (8 bytes)
  if (isa<triton::PointerType>(elemType)) {
    return 64;
  }
  return elemType.getIntOrFloatBitWidth();
}

LogicalResult computeOrValidateStorageAliasSizes(ModuleOp m) {
  LDBG("computeOrValidateStorageAliasSizes");

  // Map from storage_alias_spec SSA value to list of referencing allocations
  DenseMap<Value, SmallVector<StorageAliasLocalAllocOp, 4>> storageAliasUsers;

  // Collect all storage_alias_local_alloc operations
  m.walk([&](StorageAliasLocalAllocOp allocOp) {
    Value storageAlias = allocOp.getStorageAlias();
    storageAliasUsers[storageAlias].push_back(allocOp);
  });

  bool hasError = false;

  // Process each storage_alias_spec
  m.walk([&](StorageAliasSpecOp specOp) {
    Value specValue = specOp.getResult();
    auto &users = storageAliasUsers[specValue];
    auto storage = specOp.getStorage();

    if (users.empty()) {
      // Warn: storage_alias_spec has no users
      specOp.emitWarning(
          "storage_alias_spec has no referencing storage_alias_local_alloc "
          "operations");
      return;
    }

    SmallVector<int64_t> bufferShape;

    int64_t totalSizeBytes;

    if (storage == StorageKind::smem) {
      // SMEM: Compute 1D shape (total size in bytes)
      int64_t maxRequiredSize = 0;
      for (auto allocOp : users) {
        auto memDescType =
            cast<ttg::MemDescType>(allocOp.getResult().getType());
        int64_t size =
            memDescType.getNumElements() * getElementBitWidth(memDescType) / 8;
        if (size < 0) {
          allocOp.emitError()
              << "unsupported element type for storage alias allocation: "
              << memDescType.getElementType();
          hasError = true;
          return;
        }
        LDBG("  SMEM allocation requires " << size << " bytes");
        maxRequiredSize = std::max(maxRequiredSize, size);
      }
      LDBG(
          "Max required size for SMEM storage_alias_spec: " << maxRequiredSize);
      bufferShape.push_back(maxRequiredSize);
      totalSizeBytes = maxRequiredSize;
    } else {
      // TMEM: Compute 2D shape based on maximum dimensions across all users
      // Note: TMEM allocations may be 2D or 3D (with leading NUM_MMA_GROUPS
      // dim) For all shapes, we scale blockN by dividing by max(1,
      // 4/elementBytes) to convert to i32 units. For larger types (>4 bytes),
      // we scale blockM.
      assert(storage == StorageKind::tmem && "Unexpected storage kind");

      int64_t maxBlockM = 0;
      int64_t maxBlockN = 0;
      for (auto allocOp : users) {
        auto memDescType =
            cast<ttg::MemDescType>(allocOp.getResult().getType());
        auto shape = memDescType.getShape();
        if (shape.size() < 2) {
          allocOp.emitError()
              << "TMEM allocations must be at least 2D, got rank "
              << shape.size();
          hasError = true;
          return;
        }

        int64_t elementBits = getElementBitWidth(memDescType);
        int64_t elementBytes = elementBits / 8;

        // Get base blockM and blockN from the last two dimensions
        int64_t blockM = shape[shape.size() - 2];
        int64_t blockN = shape[shape.size() - 1];

        // Multiply in any leading dimensions (NUM_MMA_GROUPS, etc.)
        for (size_t i = 0; i < shape.size() - 2; ++i) {
          blockN *= shape[i];
        }

        // Scale for element size relative to i32 (4 bytes)
        // All scaling happens on N dimension:
        // - For larger types (> 4 bytes), scale N up
        // - For smaller types (< 4 bytes), scale N down
        if (elementBytes > 4) {
          blockN *= (elementBytes / 4);
        } else if (elementBytes < 4) {
          // Divide N by (4 / elementBytes), rounding up
          int64_t scaleFactor = 4 / elementBytes;
          blockN = (blockN + scaleFactor - 1) / scaleFactor;
        }

        LDBG("  TMEM allocation: ");
        for (size_t i = 0; i < shape.size(); ++i) {
          LDBG(shape[i] << (i < shape.size() - 1 ? "x" : ""));
        }
        LDBG(" with " << elementBytes << "-byte elements -> shape [" << blockM
                      << ", " << blockN << "]");

        maxBlockM = std::max(maxBlockM, blockM);
        maxBlockN = std::max(maxBlockN, blockN);
      }

      // Ensure blockM is valid (64 or 128 for TMEM)
      if (maxBlockM != 64 && maxBlockM != 128) {
        maxBlockM = maxBlockM <= 64 ? 64 : 128;
      }

      LDBG("Max required shape for TMEM storage_alias_spec: ["
           << maxBlockM << ", " << maxBlockN << "]");
      bufferShape.push_back(maxBlockM);
      bufferShape.push_back(maxBlockN);
      // TMEM uses i32 elements (4 bytes each)
      totalSizeBytes = maxBlockM * maxBlockN * 4;
    }

    OpBuilder builder(specOp);

    // Validate or set the size and update shape if explicit size is larger
    if (auto explicitSizeAttr = specOp.getBufferSizeBytesAttr()) {
      int64_t explicitSize = explicitSizeAttr.getInt();
      if (explicitSize < totalSizeBytes) {
        specOp.emitError()
            << "storage_alias_spec buffer_size_bytes " << explicitSize
            << " is too small for computed shape, requires at least "
            << totalSizeBytes << " bytes";
        hasError = true;
        return;
      }
      LDBG("Explicit size " << explicitSize
                            << " is sufficient for shape (needs "
                            << totalSizeBytes << ")");
      // Update shape to reflect the explicit (larger) size
      if (storage == StorageKind::smem) {
        bufferShape[0] = explicitSize;
      } else {
        // For TMEM, pad blockN to accommodate the larger explicit size
        // TMEM uses i32 elements (4 bytes each)
        int64_t blockM = bufferShape[0];
        int64_t requiredBlockN = explicitSize / (blockM * 4);
        if (requiredBlockN > bufferShape[1]) {
          LDBG("Padding TMEM blockN from " << bufferShape[1] << " to "
                                           << requiredBlockN);
          bufferShape[1] = requiredBlockN;
        }
      }
    } else {
      LDBG("Setting computed size: " << totalSizeBytes);
      specOp.setBufferSizeBytesAttr(builder.getI64IntegerAttr(totalSizeBytes));
    }

    // Set the computed buffer shape on the operation
    specOp.setBufferShapeAttr(builder.getDenseI64ArrayAttr(bufferShape));
  });

  return hasError ? failure() : success();
}

} // namespace tlx
} // namespace triton
} // namespace mlir
