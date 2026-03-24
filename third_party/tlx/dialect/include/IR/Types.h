#ifndef TRITON_DIALECT_TLX_IR_TYPES_H_
#define TRITON_DIALECT_TLX_IR_TYPES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "tlx/dialect/include/IR/TLXTypesEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tlx/dialect/include/IR/TLXTypes.h.inc"

#endif // TRITON_DIALECT_TLX_IR_TYPES_H_
