#ifndef DIALECT_TLX_IR_DIALECT_H_
#define DIALECT_TLX_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace tlx {} // namespace proton
} // namespace triton
} // namespace mlir

#endif // DIALECT_TLX_IR_DIALECT_H_
