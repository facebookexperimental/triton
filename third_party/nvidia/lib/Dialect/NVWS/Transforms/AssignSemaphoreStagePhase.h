#ifndef NVWS_TRANSFORMS_ASSIGN_SEMAPHORE_STAGE_PHASE_H_
#define NVWS_TRANSFORMS_ASSIGN_SEMAPHORE_STAGE_PHASE_H_

#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::triton::nvws_semas {

bool isFirstUseFreshWriteAfterAcquire(nvws::SemaphoreAcquireOp acquireOp,
                                      llvm::ArrayRef<Value> semaphores);

} // namespace mlir::triton::nvws_semas

#endif // NVWS_TRANSFORMS_ASSIGN_SEMAPHORE_STAGE_PHASE_H_
