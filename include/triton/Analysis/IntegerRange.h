#ifndef TRITON_ANALYSIS_INTEGER_RANGE_H
#define TRITON_ANALYSIS_INTEGER_RANGE_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton {

/// Find signed div/rem operations whose dividends are nonnegative at use.
LogicalResult collectNonNegativeDividends(Operation *root,
                                          DenseSet<Operation *> &operations);

} // namespace mlir::triton

#endif
