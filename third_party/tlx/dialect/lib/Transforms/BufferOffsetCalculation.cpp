#include "IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tlx-buffer-offset-calculation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace tlx {

// Recursively compute the size of an element in the reuse group tree
// For allocations: size is the per-buffer allocation size
// For shared groups: size is the max of children
// For distinct groups: size is the sum of children
static int64_t getElementSize(Value element) {
  if (auto allocOp = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
    auto memDescType = cast<ttg::MemDescType>(allocOp.getResult().getType());
    return getAllocationSizePerBuffer(memDescType);
  }

  if (auto reuseGroupOp = element.getDefiningOp<ReuseGroupOp>()) {
    auto groupKind = reuseGroupOp.getGroupKind();
    auto elements = reuseGroupOp.getElements();

    if (groupKind == ReuseGroupKind::shared) {
      int64_t maxSize = 0;
      for (auto child : elements) {
        maxSize = std::max(maxSize, getElementSize(child));
      }
      return maxSize;
    } else { // distinct
      int64_t totalSize = 0;
      for (auto child : elements) {
        totalSize += getElementSize(child);
      }
      return totalSize;
    }
  }

  llvm_unreachable("unexpected element type in reuse group");
}

// Recursively collect offsets for StorageAliasLocalAllocOp values
static LogicalResult
collectOffsets(Value element, int64_t currentOffset,
               int64_t bytesBetweenBuffers,
               DenseMap<Value, std::pair<int64_t, int64_t>> &offsetMap) {
  if (auto allocOp = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
    LDBG("  Recording buffer_offset="
         << currentOffset << ", bytes_between_buffers=" << bytesBetweenBuffers
         << " for StorageAliasLocalAllocOp");
    offsetMap[element] = {currentOffset, bytesBetweenBuffers};
    return success();
  }

  if (auto reuseGroupOp = element.getDefiningOp<ReuseGroupOp>()) {
    auto groupKind = reuseGroupOp.getGroupKind();
    auto elements = reuseGroupOp.getElements();

    if (groupKind == ReuseGroupKind::shared) {
      LDBG("  Processing shared group at offset " << currentOffset);
      // All children start at the same offset
      for (auto child : elements) {
        if (failed(collectOffsets(child, currentOffset, bytesBetweenBuffers,
                                  offsetMap)))
          return failure();
      }
    } else { // distinct
      LDBG("  Processing distinct group at offset " << currentOffset);
      // Children are placed sequentially
      int64_t runningOffset = currentOffset;
      for (auto child : elements) {
        if (failed(collectOffsets(child, runningOffset, bytesBetweenBuffers,
                                  offsetMap)))
          return failure();
        int64_t childSize = getElementSize(child);
        LDBG("    Child size: " << childSize << ", next offset: "
                                << runningOffset + childSize);
        runningOffset += childSize;
      }

      // Verify we have enough space
      int64_t totalSize = runningOffset - currentOffset;
      if (totalSize > bytesBetweenBuffers) {
        return reuseGroupOp.emitError()
               << "not enough space for distinct allocations: need "
               << totalSize << " bytes, have " << bytesBetweenBuffers
               << " bytes";
      }
    }
    return success();
  }

  llvm_unreachable("unexpected element type in reuse group");
}

// Clean up unused ReuseGroupOp operations after processing
// Uses worklist algorithm to handle nested groups
static void cleanupReuseGroupOps(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<ReuseGroupOp> toErase;
    module.walk([&](ReuseGroupOp op) {
      if (op.getResult().use_empty()) {
        toErase.push_back(op);
        changed = true;
      }
    });
    for (auto op : toErase) {
      LDBG("Erasing unused ReuseGroupOp");
      op.erase();
    }
  }
}

LogicalResult processBufferOverlapOps(
    ModuleOp module, DenseMap<Value, std::pair<int64_t, int64_t>> &offsetMap) {
  LDBG("processBufferOverlapOps");

  // Track which storage_alias_specs have been processed
  DenseSet<Value> processedSpecs;

  // Collect all SetBufferOverlapOps
  SmallVector<SetBufferOverlapOp> overlapOps;
  module.walk([&](SetBufferOverlapOp op) { overlapOps.push_back(op); });

  LDBG("Found " << overlapOps.size() << " SetBufferOverlapOp(s)");

  // Process each SetBufferOverlapOp
  for (auto overlapOp : overlapOps) {
    Value overlapDef = overlapOp.getOverlapDef();
    Value specValue = overlapOp.getStorageAliasSpec();

    LDBG("Processing SetBufferOverlapOp");

    // Check for duplicate set_buffer_overlap on same spec
    if (processedSpecs.contains(specValue)) {
      return overlapOp.emitError(
          "storage_alias_spec already has a set_buffer_overlap defined; "
          "each spec can only have one overlap definition");
    }

    // Find any allocation to get the num_buffers
    int64_t numBuffers = 1;
    std::function<bool(Value)> findNumBuffers = [&](Value element) -> bool {
      if (auto allocOp = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
        auto memDescType =
            cast<ttg::MemDescType>(allocOp.getResult().getType());
        numBuffers = memDescType.getShape()[0];
        return true;
      }
      if (auto reuseGroupOp = element.getDefiningOp<ReuseGroupOp>()) {
        for (auto child : reuseGroupOp.getElements()) {
          if (findNumBuffers(child))
            return true;
        }
      }
      return false;
    };

    if (!findNumBuffers(overlapDef)) {
      return overlapOp.emitError(
          "could not find StorageAliasLocalAllocOp in overlap definition");
    }

    // Compute total size from the reuse group tree
    int64_t sizePerBuffer = getElementSize(overlapDef);
    int64_t bytesBetweenBuffers = sizePerBuffer;

    LDBG("  numBuffers=" << numBuffers << ", sizePerBuffer=" << sizePerBuffer
                         << ", bytesBetweenBuffers=" << bytesBetweenBuffers);

    // Recursively collect offsets starting at offset 0
    if (failed(collectOffsets(overlapDef, /*currentOffset=*/0,
                              bytesBetweenBuffers, offsetMap))) {
      return failure();
    }

    // Mark spec as processed
    processedSpecs.insert(specValue);

    // Erase the SetBufferOverlapOp
    LDBG("Erasing SetBufferOverlapOp");
    overlapOp.erase();
  }

  // Clean up unused ReuseGroupOp operations
  cleanupReuseGroupOps(module);

  LDBG("processBufferOverlapOps completed successfully");
  return success();
}

} // namespace tlx
} // namespace triton
} // namespace mlir
