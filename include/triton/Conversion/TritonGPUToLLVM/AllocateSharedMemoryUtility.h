#ifndef TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ALLOCATE_UTILITY_H_
#define TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ALLOCATE_UTILITY_H_

#include "mlir/IR/BuiltinOps.h"
#include "triton/Analysis/Allocation.h"

namespace mlir::triton::gpu {

/// Attach shared memory related attributes to module and operations inside it.
/// This includes total shared memory consumption in module and shared memory
/// offsets of buffers associated with operations.
void attachAllocationSizeAndOffsetAttr(ModuleOp mod,
                                       ModuleAllocation &allocation);

/// Add shared memory access annotations to all operations that use shared
/// memory Only adds annotations when MLIR_ENABLE_DUMP=1 is set.
void addSharedMemoryAnnotations(ModuleOp mod);

/// Under TRITON_ALLOW_NPOT, rejects every tt.reduce, tt.scan and tt.gather
/// with a non-power-of-two operand or result dimension, and every tt.histogram
/// with a non-power-of-two input, as each is wrong on a modular layout:
///  - tt.reduce folds the wrapped slots into the combine (an NPOT sum counts
///    cover - extent elements twice);
///  - tt.reduce and tt.scan scratch sizing builds a LinearEncodingAttr the
///    layout cannot satisfy;
///  - tt.histogram counts the wrapped slots (96 inputs give 128 counts);
///  - tt.gather returns wrong elements (over 95% of a 96x32 result).
/// Run before ModuleAllocation; emits one error per op and never mutates the
/// IR.
LogicalResult verifyNpotModularOpsSupported(ModuleOp mod);

} // namespace mlir::triton::gpu

#endif // TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ALLOCATE_UTILITY_H_
