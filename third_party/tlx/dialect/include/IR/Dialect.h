#ifndef TRITON_DIALECT_TLX_IR_DIALECT_H_
#define TRITON_DIALECT_TLX_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "tlx/dialect/include/IR/Dialect.h.inc"
#include "tlx/dialect/include/IR/OpsEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "tlx/dialect/include/IR/TLXAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "tlx/dialect/include/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace tlx {
constexpr static char AttrHasTLXOpsName[] = "tlx.has_tlx_ops";
} // namespace tlx
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TLX_IR_DIALECT_H_
