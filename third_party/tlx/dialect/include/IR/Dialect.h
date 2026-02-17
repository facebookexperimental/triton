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
inline int64_t getElementAlignment(Value element) {
  if (auto allocOp = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
    auto memDescType =
        cast<triton::gpu::MemDescType>(allocOp.getResult().getType());
    return getAllocAlignment(memDescType);
  }

  if (auto reuseGroupOp = element.getDefiningOp<ReuseGroupOp>()) {
    int64_t maxAlignment = 1;
    for (auto child : reuseGroupOp.getElements()) {
      maxAlignment = std::max(maxAlignment, getElementAlignment(child));
    }
    return maxAlignment;
  }

  llvm_unreachable("unexpected element type in reuse group");
}

// Recursively compute the size of an element in the reuse group tree.
// For allocations: size is the per-buffer allocation size.
// For shared groups: size is the max of children.
// For distinct groups: size is the sum of children (with alignment padding).
inline int64_t getElementSize(Value element, int64_t alignment) {
  if (auto allocOp = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
    auto memDescType =
        cast<triton::gpu::MemDescType>(allocOp.getResult().getType());
    return getAllocationSizePerBuffer(memDescType);
  }

  if (auto reuseGroupOp = element.getDefiningOp<ReuseGroupOp>()) {
    auto groupKind = reuseGroupOp.getGroupKind();
    auto elements = reuseGroupOp.getElements();

    if (groupKind == ReuseGroupKind::shared) {
      int64_t maxSize = 0;
      for (auto child : elements) {
        maxSize = std::max(maxSize, getElementSize(child, alignment));
      }
      return maxSize;
    } else { // distinct
      int64_t totalSize = 0;
      for (auto child : elements) {
        totalSize = alignUp(totalSize, alignment);
        totalSize += getElementSize(child, alignment);
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
