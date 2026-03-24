#ifndef TRITON_TLX_PASSES_H
#define TRITON_TLX_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton::tlx {

#define GEN_PASS_DECL
#include "tlx/dialect/include/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "tlx/dialect/include/Transforms/Passes.h.inc"

} // namespace mlir::triton::tlx

#endif
