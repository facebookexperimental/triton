#ifndef TRITON_ANALYSIS_INTEGER_RANGE_H
#define TRITON_ANALYSIS_INTEGER_RANGE_H

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton {

/// Create a solver for SSA integer ranges and target launch bounds.
std::unique_ptr<DataFlowSolver> createIntegerRangeSolver(Operation *root);

/// Find signed div/rem operations with nonnegative SSA dividends.
LogicalResult collectNonNegativeDividends(Operation *root,
                                          DenseSet<Operation *> &operations);

} // namespace mlir::triton

#endif
