#ifndef TRITON_THIRD_PARTY_NVIDIA_LIB_DIALECT_NVWS_TRANSFORMS_ASSIGNSTAGEPHASE_H_
#define TRITON_THIRD_PARTY_NVIDIA_LIB_DIALECT_NVWS_TRANSFORMS_ASSIGNSTAGEPHASE_H_

#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::triton::nvws_semas {

bool isFirstUseFreshWriteAfterAcquire(
    nvws::SemaphoreAcquireOp acquireOp, llvm::ArrayRef<Value> semaphores);

} // namespace mlir::triton::nvws_semas

#endif
