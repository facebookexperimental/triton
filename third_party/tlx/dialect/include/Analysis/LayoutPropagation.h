#ifndef TLX_ANALYSIS_LAYOUTPROPAGATION_H
#define TLX_ANALYSIS_LAYOUTPROPAGATION_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <optional>

using namespace mlir::dataflow;

namespace mlir::triton::tlx {

//===----------------------------------------------------------------------===//
// LayoutEncoding
//===----------------------------------------------------------------------===//

class LayoutEncoding {
public:
  /// Construct a LayoutEncoding value as uninitialized.
  explicit LayoutEncoding() = default;

  /// Construct a LayoutEncoding value with a known constant.
  LayoutEncoding(Attribute encoding) : encoding(std::move(encoding)) {}

  bool operator==(const LayoutEncoding &rhs) const {
    return encoding == rhs.encoding;
  }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !encoding.has_value(); }

  /// Whether the state is unknown.
  bool isUnknown() const { return encoding == nullptr; }

  Attribute getLayoutEncoding() const {
    assert(!isUninitialized());
    return *encoding;
  }

  void print(raw_ostream &os) const;
  static LayoutEncoding meet(const LayoutEncoding &lhs,
                             const LayoutEncoding &rhs);
  static LayoutEncoding join(const LayoutEncoding &lhs,
                             const LayoutEncoding &rhs);
  static LayoutEncoding getUnknownLayout() {
    return LayoutEncoding{/*layoutEncoding=*/nullptr};
  }

private:
  std::optional<Attribute> encoding;
};

//===----------------------------------------------------------------------===//
// LayoutEncodingLattice
//===----------------------------------------------------------------------===//

class LayoutEncodingLattice : public Lattice<LayoutEncoding> {
public:
  using Lattice::Lattice;
};

//===----------------------------------------------------------------------===//
// LayoutPropagation
//===----------------------------------------------------------------------===//

class LayoutPropagation
    : public SparseBackwardDataFlowAnalysis<LayoutEncodingLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<LayoutEncodingLattice *> operands,
                 ArrayRef<const LayoutEncodingLattice *> results) override;

  void visitBranchOperand(OpOperand &operand) override;

  void visitCallOperand(OpOperand &operand) override;

  void setToExitState(LayoutEncodingLattice *lattice) override;
};

class LayoutForwardPropagation
    : public SparseForwardDataFlowAnalysis<LayoutEncodingLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const LayoutEncodingLattice *> operands,
                 ArrayRef<LayoutEncodingLattice *> results) override;

  void setToEntryState(LayoutEncodingLattice *lattice) override;
};

} // namespace mlir::triton::tlx

#endif // TLX_ANALYSIS_LAYOUTPROPAGATION_H
