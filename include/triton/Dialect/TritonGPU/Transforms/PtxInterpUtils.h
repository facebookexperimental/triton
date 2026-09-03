#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PTXINTERPUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PTXINTERPUTILS_H_

#include "mlir/IR/Operation.h"

// Helpers that interpret the contents of an inline-PTX asm block well enough to
// answer a specific structural question about it. The instruction mnemonics
// recognized here are NVIDIA-specific; the callers live in target-independent
// TritonGPU transforms, which is why these are declared alongside them rather
// than in third_party/nvidia.

namespace mlir {

// Return true if the op multiplies its two operands and does nothing else, so
// that a zero operand forces a zero result. Covers arith.mulf and the packed
// mul.f32x2 emitted as elementwise inline PTX (see
// third_party/nvidia/language/cuda/inline_ptx_lib.py). Callers depend on the
// zero-propagation property, so an asm block that multiplies and then performs
// further arithmetic must NOT match.
bool isMulOp(Operation *op);

} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_PTXINTERPUTILS_H_
