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
  ///
  /// Unknown is the conservative "conflicting or unsupported" state for layout
  /// propagation: forward joins widen to unknown when distinct concrete
  /// encodings meet at a merge point, and backward meets also collapse to
  /// unknown when any side is already unknown.
  bool isUnknown() const { return encoding == nullptr; }

  Attribute getLayoutEncoding() const {
    assert(!isUninitialized());
    return *encoding;
  }

  void print(raw_ostream &os) const;
  static LayoutEncoding meet(const LayoutEncoding &lhs,
                             const LayoutEncoding &rhs);
  /// Lattice join used by forward propagation.
  ///
  /// Uninitialized yields to any concrete state, unknown dominates, equal
  /// concrete encodings stay concrete, and conflicting concrete encodings widen
  /// to unknown so region merges stay conservative instead of asserting.
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
// LayoutBackwardPropagation
//===----------------------------------------------------------------------===//

class LayoutBackwardPropagation
    : public SparseBackwardDataFlowAnalysis<LayoutEncodingLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<LayoutEncodingLattice *> operands,
                 ArrayRef<const LayoutEncodingLattice *> results) override;

  void visitBranchOperand(OpOperand &operand) override;

  void visitCallOperand(OpOperand &operand) override;

  void setToExitState(LayoutEncodingLattice *lattice) override;

  void visitNonControlFlowArguments(
      RegionSuccessor &successor,
      ArrayRef<BlockArgument> arguments) override {
    // Default: do nothing
  }
  LogicalResult visitRegionInReverse(Operation *op);

  void visitWarpSpecRegionArgs(Operation *op, Value opnd,
                               const LayoutEncoding &resultEncoding);
};

//===----------------------------------------------------------------------===//
// LayoutForwardPropagation
//===----------------------------------------------------------------------===//

class LayoutForwardPropagation
    : public SparseForwardDataFlowAnalysis<LayoutEncodingLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const LayoutEncodingLattice *> operands,
                 ArrayRef<LayoutEncodingLattice *> results) override;


  void setToEntryState(LayoutEncodingLattice *lattice) override;

  LogicalResult visitRegion(Operation *op);

  LogicalResult visitWarpSpecRegionArgs(Operation *op, Value opnd,
                                        const LayoutEncoding &opndEncoding);
};

//===----------------------------------------------------------------------===//
// TensorLayout
//===----------------------------------------------------------------------===//

class TensorLayout {
public:
  /// Construct a TensorLayout value as uninitialized.
  explicit TensorLayout() = default;

  /// Construct a TensorLayout value with a known constant.
  TensorLayout(Attribute encoding) : encoding(std::move(encoding)) {}

  bool operator==(const TensorLayout &rhs) const {
    return encoding == rhs.encoding;
  }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !encoding.has_value(); }

  /// Whether the state is unknown.
  bool isUnknown() const { return encoding == nullptr; }

  Attribute getLayoutEncoding() const {
    assert(!isUninitialized());
    assert(!isUnknown());
    return *encoding;
  }

  void print(raw_ostream &os) const;
  static TensorLayout meet(const TensorLayout &lhs, const TensorLayout &rhs);
  static TensorLayout join(const TensorLayout &lhs, const TensorLayout &rhs);
  static TensorLayout getUnknownLayout() {
    return TensorLayout{/*layoutEncoding=*/nullptr};
  }

private:
  std::optional<Attribute> encoding;
};

//===----------------------------------------------------------------------===//
// TensorLayoutLattice
//===----------------------------------------------------------------------===//

class TensorLayoutLattice : public Lattice<TensorLayout> {
public:
  using Lattice::Lattice;
};

//===----------------------------------------------------------------------===//
// TensorBackwardPropagation
//===----------------------------------------------------------------------===//

class TensorBackwardPropagation
    : public SparseBackwardDataFlowAnalysis<TensorLayoutLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<TensorLayoutLattice *> operands,
                 ArrayRef<const TensorLayoutLattice *> results) override;

  void visitBranchOperand(OpOperand &operand) override;

  void visitCallOperand(OpOperand &operand) override;

  void setToExitState(TensorLayoutLattice *lattice) override;

  void visitNonControlFlowArguments(
      RegionSuccessor &successor,
      ArrayRef<BlockArgument> arguments) override {
    // Default: do nothing
  }
};

} // namespace mlir::triton::tlx

#endif // TLX_ANALYSIS_LAYOUTPROPAGATION_H
