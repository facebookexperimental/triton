#ifndef TRITON_DIALECT_TLX_IR_DIALECT_H_
#define TRITON_DIALECT_TLX_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

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

} // namespace tlx
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TLX_IR_DIALECT_H_
