#include "IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
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

#define GEN_PASS_DEF_TLXSTORAGEALIASSIZEDEFINITION

#include "tlx/dialect/include/Transforms/Passes.h.inc"

namespace {

/// Compute the size in bytes of an allocation based on its MemDescType.
/// Returns -1 if the element type is not supported.
int64_t computeAllocationSizeBytes(ttg::MemDescType memDescType) {
  Type elemType = memDescType.getElementType();
  int64_t elemBitWidth = 0;

  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    elemBitWidth = floatType.getWidth();
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    elemBitWidth = intType.getWidth();
  } else if (isa<triton::PointerType>(elemType)) {
    elemBitWidth = 64;
  } else {
    // Unsupported element type - return -1 to signal error
    return -1;
  }

  // Compute total number of elements
  int64_t numElements = 1;
  for (int64_t dim : memDescType.getShape()) {
    numElements *= dim;
  }

  // Round up to bytes
  return (numElements * elemBitWidth + 7) / 8;
}

} // namespace

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

    if (users.empty()) {
      // Warn: storage_alias_spec has no users
      specOp.emitWarning(
          "storage_alias_spec has no referencing storage_alias_local_alloc "
          "operations");
      return;
    }

    // Compute max required size
    int64_t maxRequiredSize = 0;
    for (auto allocOp : users) {
      auto memDescType = cast<ttg::MemDescType>(allocOp.getResult().getType());
      int64_t size = computeAllocationSizeBytes(memDescType);
      if (size < 0) {
        allocOp.emitError()
            << "unsupported element type for storage alias allocation: "
            << memDescType.getElementType();
        hasError = true;
        return;
      }
      LDBG("  Allocation requires " << size << " bytes");
      maxRequiredSize = std::max(maxRequiredSize, size);
    }

    LDBG("Max required size for storage_alias_spec: " << maxRequiredSize);

    // Validate or set the size
    if (auto explicitSizeAttr = specOp.getBufferSizeBytesAttr()) {
      int64_t explicitSize = explicitSizeAttr.getInt();
      if (explicitSize < maxRequiredSize) {
        specOp.emitError()
            << "storage_alias_spec size " << explicitSize
            << " is too small, requires at least " << maxRequiredSize
            << " bytes";
        hasError = true;
        return;
      }
      LDBG("Explicit size " << explicitSize << " is sufficient");
      // Explicit size is sufficient, keep it (allows padding)
    } else {
      // Set computed size on the operation
      LDBG("Setting computed size: " << maxRequiredSize);
      OpBuilder builder(specOp);
      specOp.setBufferSizeBytesAttr(
          builder.getI64IntegerAttr(maxRequiredSize));
    }
  });

  return hasError ? failure() : success();
}

struct TLXStorageAliasSizeDefinitionPass
    : public impl::TLXStorageAliasSizeDefinitionBase<
          TLXStorageAliasSizeDefinitionPass> {
public:
  using impl::TLXStorageAliasSizeDefinitionBase<
      TLXStorageAliasSizeDefinitionPass>::TLXStorageAliasSizeDefinitionBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (failed(computeOrValidateStorageAliasSizes(m))) {
      signalPassFailure();
    }
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
