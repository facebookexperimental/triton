#include "IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tlx-buffer-offset-calculation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace tlx {

// Recursively collect offsets for StorageAliasLocalAllocOp values
// The offsetMap stores (buffer_offset, units_between_buffer_groups, group_size)
// tuples. Units are bytes for SMEM, or TMEM columns when useTmemColumns=true.
static LogicalResult collectOffsets(
    Value element, int64_t currentOffset, int64_t bytesBetweenBufferGroups,
    int64_t alignment, int64_t currentGroupSize,
    DenseMap<Value, std::tuple<int64_t, int64_t, int64_t>> &offsetMap,
    bool useTmemColumns = false) {
  if (auto allocOp = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
    LDBG("  Recording buffer_offset="
         << currentOffset << ", bytes_between_buffer_groups="
         << bytesBetweenBufferGroups << ", group_size=" << currentGroupSize
         << " for StorageAliasLocalAllocOp");
    offsetMap[element] = std::make_tuple(
        currentOffset, bytesBetweenBufferGroups, currentGroupSize);
    return success();
  }

  if (auto reuseGroupOp = element.getDefiningOp<ReuseGroupOp>()) {
    auto groupKind = reuseGroupOp.getGroupKind();
    auto elements = reuseGroupOp.getElements();
    int64_t groupSize = reuseGroupOp.getGroupSize();

    if (groupKind == ReuseGroupKind::shared) {
      LDBG("  Processing shared group at offset "
           << currentOffset << " with group_size=" << groupSize);
      // For subtiling: divide bytesBetweenBufferGroups by group_size
      // This means each subtile buffer gets bytesBetweenBufferGroups/groupSize
      // spacing
      int64_t childBytesBetween = bytesBetweenBufferGroups / groupSize;
      // Multiply the group_size to propagate to children
      int64_t childGroupSize = currentGroupSize * groupSize;
      // All children start at the same offset
      for (auto child : elements) {
        if (failed(collectOffsets(child, currentOffset, childBytesBetween,
                                  alignment, childGroupSize, offsetMap,
                                  useTmemColumns)))
          return failure();
      }
    } else { // distinct
      LDBG("  Processing distinct group at offset " << currentOffset);
      // Children are placed sequentially, each aligned
      int64_t runningOffset = currentOffset;
      for (auto child : elements) {
        // For TMEM columns, align each child to its own column count
        // to ensure offsets are divisible by each buffer's column width.
        int64_t childAlignment =
            useTmemColumns ? getElementAlignment(child, true) : alignment;
        runningOffset = alignUp(runningOffset, childAlignment);
        if (failed(collectOffsets(child, runningOffset,
                                  bytesBetweenBufferGroups, alignment,
                                  currentGroupSize, offsetMap, useTmemColumns)))
          return failure();
        int64_t childSize = getElementSize(child, alignment, useTmemColumns);
        LDBG("    Child size: " << childSize << ", next offset: "
                                << runningOffset + childSize);
        runningOffset += childSize;
      }

      // Verify we have enough space
      int64_t totalSize = runningOffset - currentOffset;
      if (totalSize > bytesBetweenBufferGroups) {
        return reuseGroupOp.emitError()
               << "not enough space for distinct allocations: need "
               << totalSize << " bytes, have " << bytesBetweenBufferGroups
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
    ModuleOp module,
    DenseMap<Value, std::tuple<int64_t, int64_t, int64_t>> &offsetMap) {
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

    // Check if this overlap group uses TMEM storage. For TMEM, we compute
    // sizes in column units instead of bytes, because memdesc_index lowering
    // multiplies the index by numCols (from getTmemAllocSizes), and different
    // TMEM buffer types have different bytes-per-column ratios.
    bool isTmem = containsTmemAllocation(overlapDef);

    // Compute alignment from the reuse group tree.
    // For TMEM, alignment is 1 column (columns are the atomic unit).
    int64_t alignment = getElementAlignment(overlapDef, isTmem);

    // Compute total size from the reuse group tree.
    // For TMEM, sizes are in column units; for SMEM, in bytes.
    int64_t sizePerBufferGroup = getElementSize(overlapDef, alignment, isTmem);
    int64_t bytesBetweenBufferGroups = alignUp(sizePerBufferGroup, alignment);

    LDBG("  numBuffers=" << numBuffers << ", sizePerBufferGroup="
                         << sizePerBufferGroup << ", bytesBetweenBufferGroups="
                         << bytesBetweenBufferGroups
                         << ", alignment=" << alignment);

    // Recursively collect offsets starting at offset 0 with group_size 1
    if (failed(collectOffsets(overlapDef, /*currentOffset=*/0,
                              bytesBetweenBufferGroups, alignment,
                              /*currentGroupSize=*/1, offsetMap, isTmem))) {
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
