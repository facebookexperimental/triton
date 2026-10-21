#ifndef TRITON_ANALYSIS_INTEGER_RANGE_H
#define TRITON_ANALYSIS_INTEGER_RANGE_H

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton {

enum class IntegerProof { Proven, Disproven, Unknown };

/// Create a solver for SSA integer facts and target launch bounds.
std::unique_ptr<DataFlowSolver> createIntegerFactSolver(Operation *root);

class IntegerFactAnalysis {
public:
  IntegerFactAnalysis();
  ~IntegerFactAnalysis();

  LogicalResult initialize(Operation *root);
  IntegerValueRange getRange(Value value) const;
  /// Query predicates on non-poison values.
  IntegerProof prove(arith::CmpIPredicate predicate, Value lhs,
                     Value rhs) const;

private:
  std::unique_ptr<DataFlowSolver> solver;
};

/// Find signed div/rem operations with nonnegative SSA dividends.
LogicalResult collectNonNegativeDividends(Operation *root,
                                          DenseSet<Operation *> &operations);

} // namespace mlir::triton

#endif
