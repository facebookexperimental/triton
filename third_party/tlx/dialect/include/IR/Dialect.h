#ifndef TRITON_DIALECT_TLX_IR_DIALECT_H_
#define TRITON_DIALECT_TLX_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "tlx/dialect/include/IR/Dialect.h.inc"
#include "tlx/dialect/include/IR/Traits.h"
#define GET_ATTRDEF_CLASSES
#include "tlx/dialect/include/IR/TLXAttrDefs.h.inc"

#include "tlx/dialect/include/IR/Types.h"

#define GET_OP_CLASSES
#include "tlx/dialect/include/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace tlx {
constexpr static char AttrHasExplicitLocalMemAccessName[] =
    "tlx.has_explicit_local_mem_access";
constexpr static char AttrHasTLXOpsName[] = "tlx.has_tlx_ops";
constexpr static char AttrHasWarpSpecOpsName[] = "tlx.has_warp_spec_ops";
constexpr static char AttrTLXEnablePairedCTAMMAName[] =
    "tlx.enable_paired_cta_mma";

bool tlxEnablePairedMMA(Operation *op);

// Get element size in bytes for a type, handling pointer types (8 bytes)
// and using ceiling division for sub-byte types.
inline int64_t getElementBytes(mlir::Type elemType) {
  int64_t elemBits = isa<triton::PointerType>(elemType)
                         ? 64
                         : elemType.getIntOrFloatBitWidth();
  return (elemBits + 7) / 8;
}

// Compute the size of one buffer in an allocation (excluding the num
// dimension). For a shape like [num, d1, d2, ...], returns d1 * d2 * ... *
// elemBytes.
inline int64_t
getAllocationSizePerBuffer(triton::gpu::MemDescType memDescType) {
  int64_t totalBytes = memDescType.getNumElements() *
                       getElementBytes(memDescType.getElementType());
  return totalBytes / memDescType.getShape()[0];
}

// Check if a MemDescType uses TMEM storage.
inline bool isTmemMemDesc(triton::gpu::MemDescType memDescType) {
  return isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
      memDescType.getMemorySpace());
}

// Compute the number of TMEM columns for one buffer in a multi-buffered
// allocation. For a shape like [numBuf, d1, d2, ...], strips the leading
// dimension and computes the per-buffer TMEM column count.
inline int64_t
getAllocationColumnsPerBuffer(triton::gpu::MemDescType memDescType) {
  auto shape = memDescType.getShape();
  assert(shape.size() >= 2 && "TMEM allocation must be at least 2D");
  auto encoding = memDescType.getEncoding();

  // Strip leading num_buffers dimension
  SmallVector<int64_t> perBufferShape(shape.begin() + 1, shape.end());

  if (isa<DummyTMEMLayoutAttr>(encoding)) {
    // DummyTMEMLayoutAttr is a placeholder for sub-16-bit types that will
    // resolve to TensorMemoryScalesEncodingAttr after layout propagation.
    // Compute columns using the scales formula directly since
    // getTmemAllocSizes doesn't handle placeholder encodings.
    int64_t m = perBufferShape[perBufferShape.size() - 2];
    int64_t k = perBufferShape[perBufferShape.size() - 1];
    return ((m + 31) / 32) * ((k + 3) / 4);
  }

  // For resolved encodings (TensorMemoryEncodingAttr,
  // TensorMemoryScalesEncodingAttr), delegate to getTmemAllocSizes.
  auto perBufferType = triton::gpu::MemDescType::get(
      perBufferShape, memDescType.getElementType(), encoding,
      memDescType.getMemorySpace(), memDescType.getMutableMemory());
  auto tmemAlloc = triton::nvidia_gpu::getTmemAllocSizes(perBufferType);
  return tmemAlloc.numCols;
}

// Check if an element in the reuse group tree contains TMEM allocations.
inline bool containsTmemAllocation(Value element) {
  if (auto allocOp = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
    auto memDescType =
        cast<triton::gpu::MemDescType>(allocOp.getResult().getType());
    return isTmemMemDesc(memDescType);
  }
  if (auto reuseGroupOp = element.getDefiningOp<ReuseGroupOp>()) {
    for (auto child : reuseGroupOp.getElements()) {
      if (containsTmemAllocation(child))
        return true;
    }
  }
  return false;
}

// TODO: We currently force data to be 128-byte aligned for SMEM (TMA) and
// 32-byte aligned for TMEM, but we may want to consider relaxing this in the
// future by examining the full IR.
constexpr int64_t kSmemAlignment = 128;
constexpr int64_t kTmemAlignment = 32;

inline int64_t alignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Get the alignment requirement for a single allocation.
// The alignment is the max of the storage type alignment (SMEM or TMEM)
// and the element type alignment.
inline int64_t getAllocAlignment(triton::gpu::MemDescType memDescType) {
  int64_t storageAlignment = isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
                                 memDescType.getMemorySpace())
                                 ? kTmemAlignment
                                 : kSmemAlignment;
  int64_t elemAlignment = getElementBytes(memDescType.getElementType());
  return std::max(storageAlignment, elemAlignment);
}

// Recursively compute the alignment requirement for an element in the
// reuse group tree. For allocations: alignment is determined by the memory
// space and element type. For groups (both shared and distinct): alignment
// is the max of all children's alignments.
// When useTmemColumns is true, returns the buffer's column count for leaf
// allocations (ensures offsets within distinct groups are divisible by
// each buffer's column width).
inline int64_t getElementAlignment(Value element, bool useTmemColumns = false) {
  if (auto allocOp = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
    auto memDescType =
        cast<triton::gpu::MemDescType>(allocOp.getResult().getType());
    if (useTmemColumns)
      return getAllocationColumnsPerBuffer(memDescType);
    return getAllocAlignment(memDescType);
  }

  if (auto reuseGroupOp = element.getDefiningOp<ReuseGroupOp>()) {
    int64_t maxAlignment = 1;
    for (auto child : reuseGroupOp.getElements()) {
      maxAlignment =
          std::max(maxAlignment, getElementAlignment(child, useTmemColumns));
    }
    return maxAlignment;
  }

  llvm_unreachable("unexpected element type in reuse group");
}

// Recursively compute the size of an element in the reuse group tree.
// For allocations: size is the per-buffer allocation size (in bytes, or in
// TMEM columns when useTmemColumns is true).
// For shared groups: size is the max of children.
// For distinct groups: size is the sum of children (with alignment padding).
inline int64_t getElementSize(Value element, int64_t alignment,
                              bool useTmemColumns = false) {
  if (auto allocOp = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
    auto memDescType =
        cast<triton::gpu::MemDescType>(allocOp.getResult().getType());
    if (useTmemColumns)
      return getAllocationColumnsPerBuffer(memDescType);
    return getAllocationSizePerBuffer(memDescType);
  }

  if (auto reuseGroupOp = element.getDefiningOp<ReuseGroupOp>()) {
    auto groupKind = reuseGroupOp.getGroupKind();
    auto elements = reuseGroupOp.getElements();
    int64_t groupSize = reuseGroupOp.getGroupSize();

    if (groupKind == ReuseGroupKind::shared) {
      int64_t maxSize = 0;
      for (auto child : elements) {
        maxSize =
            std::max(maxSize, getElementSize(child, alignment, useTmemColumns));
      }
      // Multiply by group_size for subtiling
      return maxSize * groupSize;
    } else { // distinct
      int64_t totalSize = 0;
      for (auto child : elements) {
        // For TMEM columns, align each child to its own column count
        // to ensure offsets are divisible by each buffer's column width.
        int64_t childAlignment =
            useTmemColumns ? getElementAlignment(child, true) : alignment;
        totalSize = alignUp(totalSize, childAlignment);
        totalSize += getElementSize(child, alignment, useTmemColumns);
      }
      return totalSize;
    }
  }

  llvm_unreachable("unexpected element type in reuse group");
}

} // namespace tlx
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TLX_IR_DIALECT_H_
