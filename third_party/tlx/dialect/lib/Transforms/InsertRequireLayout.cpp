#include "IR/Dialect.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tlx-amd-insert-require-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace tlx = ::mlir::triton::tlx;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXINSERTREQUIRELAYOUT
#include "tlx/dialect/include/Transforms/Passes.h.inc"

namespace {

// ============================================================================
// Backward dataflow analysis: propagate DotOperandEncodingAttr from tt.DotOp
// operands backward through convert_layout chains and scf.for iter_args to
// local_load ops.
//
// MLIR's SparseBackwardDataFlowAnalysis automatically handles scf.for
// (via RegionBranchOpInterface), so the analysis naturally sees through
// yield -> body_arg -> init_value chains without any manual iter_arg tracing.
// ============================================================================

class DotEncoding {
public:
  DotEncoding() = default;
  explicit DotEncoding(Attribute enc) : encoding(enc) {}

  bool operator==(const DotEncoding &rhs) const {
    return encoding == rhs.encoding;
  }

  bool isUninitialized() const { return !encoding.has_value(); }
  bool isUnknown() const {
    return encoding.has_value() && *encoding == nullptr;
  }

  Attribute getEncoding() const {
    assert(!isUninitialized() && !isUnknown());
    return *encoding;
  }

  void print(raw_ostream &os) const {
    if (isUninitialized())
      os << "<uninitialized>";
    else if (isUnknown())
      os << "<unknown>";
    else
      encoding->print(os);
  }

  static DotEncoding meet(const DotEncoding &lhs, const DotEncoding &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    return DotEncoding(nullptr); // conflict
  }

  static DotEncoding join(const DotEncoding &lhs, const DotEncoding &rhs) {
    return meet(lhs, rhs);
  }

private:
  std::optional<Attribute> encoding;
};

class DotEncodingLattice : public Lattice<DotEncoding> {
public:
  using Lattice::Lattice;
};

class DotEncodingBackward
    : public SparseBackwardDataFlowAnalysis<DotEncodingLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<DotEncodingLattice *> operands,
                 ArrayRef<const DotEncodingLattice *> results) override {
    // Seed from tt.DotOp: propagate the required dot-operand encoding to
    // the values that define operands A and B.
    if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
      for (unsigned i = 0; i < 2; ++i) {
        auto type = cast<RankedTensorType>(dotOp.getOperand(i).getType());
        if (auto dotEnc =
                dyn_cast<ttg::DotOperandEncodingAttr>(type.getEncoding())) {
          ChangeResult changed = operands[i]->meet(DotEncoding(dotEnc));
          propagateIfChanged(operands[i], changed);
        }
      }
      return success();
    }

    // Passthrough for ConvertLayoutOp: the source should eventually carry
    // the same dot-operand encoding (the convert becomes redundant).
    if (isa<ttg::ConvertLayoutOp>(op)) {
      auto resultEnc = results[0]->getValue();
      if (!resultEnc.isUninitialized() && !resultEnc.isUnknown()) {
        ChangeResult changed = operands[0]->meet(resultEnc);
        propagateIfChanged(operands[0], changed);
      }
      return success();
    }

    return success();
  }

  void visitBranchOperand(OpOperand &operand) override {}
  void visitCallOperand(OpOperand &operand) override {}
  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
                               ArrayRef<BlockArgument> arguments) override {}
  void setToExitState(DotEncodingLattice *lattice) override {}
};

// ============================================================================
// Rewrite helpers
// ============================================================================

static ttg::SwizzledSharedEncodingAttr
computeSharedEncFromDotEnc(ttg::DotOperandEncodingAttr dotEnc,
                           ttg::LocalLoadOp localLoadOp) {
  auto resultType = cast<RankedTensorType>(localLoadOp.getType());
  auto order = ttg::getOrderForMemory(resultType);
  auto ctaLayout = ttg::getCGALayout(resultType.getEncoding());
  unsigned bitWidth = resultType.getElementType().getIntOrFloatBitWidth();
  return ttg::SwizzledSharedEncodingAttr::get(localLoadOp->getContext(), dotEnc,
                                              resultType.getShape(), order,
                                              ctaLayout, bitWidth,
                                              /*needTrans=*/false);
}

static void applyRequireLayout(ttg::SwizzledSharedEncodingAttr encoding,
                               ttg::LocalLoadOp localLoadOp,
                               OpBuilder &builder) {
  auto loadMemDesc = localLoadOp->getOperand(0);

  if (loadMemDesc.getDefiningOp<tlx::RequireLayoutOp>())
    return;

  // Respect user-specified order on the source memdesc.
  if (auto srcType = dyn_cast<ttg::MemDescType>(loadMemDesc.getType())) {
    if (auto srcEnc =
            dyn_cast<ttg::SwizzledSharedEncodingAttr>(srcType.getEncoding())) {
      if (srcEnc.getOrder() != encoding.getOrder()) {
        LDBG("Respecting user-specified order "
             << srcEnc << " instead of derived " << encoding);
        encoding = ttg::SwizzledSharedEncodingAttr::get(
            encoding.getContext(), encoding.getVec(), encoding.getPerPhase(),
            encoding.getMaxPhase(), srcEnc.getOrder(), encoding.getCGALayout());
      }
    }
  }

  builder.setInsertionPoint(localLoadOp);
  if (auto type = dyn_cast<ttg::MemDescType>(loadMemDesc.getType())) {
    auto newType = ttg::MemDescType::get(
        type.getShape(), type.getElementType(), mlir::cast<Attribute>(encoding),
        type.getMemorySpace(), type.getMutableMemory());
    auto requireOp = tlx::RequireLayoutOp::create(
        builder, localLoadOp->getLoc(), newType, loadMemDesc);
    localLoadOp->setOperand(0, requireOp.getResult());
  }
}

static void
collectRegionBranchSuccessors(RegionBranchOpInterface branchOp,
                              SmallVectorImpl<RegionSuccessor> &successors) {
  auto appendUniqueSuccessors = [&](ArrayRef<RegionSuccessor> newSuccessors) {
    for (RegionSuccessor successor : newSuccessors) {
      if (!llvm::is_contained(successors, successor))
        successors.push_back(successor);
    }
  };

  SmallVector<RegionSuccessor> newSuccessors;
  branchOp.getSuccessorRegions(RegionBranchPoint::parent(), newSuccessors);
  appendUniqueSuccessors(newSuccessors);
  for (Region &region : branchOp->getRegions()) {
    newSuccessors.clear();
    branchOp.getSuccessorRegions(region, newSuccessors);
    appendUniqueSuccessors(newSuccessors);
  }
}

static std::optional<Type> getConsensusType(ValueRange values) {
  if (values.empty())
    return std::nullopt;

  Type consensusType = values.front().getType();
  for (Value value : values.drop_front()) {
    if (value.getType() != consensusType)
      return std::nullopt;
  }
  return consensusType;
}

static void updateRegionBranchTypes(RegionBranchOpInterface branchOp) {
  SmallVector<RegionSuccessor> successors;
  collectRegionBranchSuccessors(branchOp, successors);

  bool changed = true;
  while (changed) {
    changed = false;
    for (RegionSuccessor successor : successors) {
      ValueRange successorInputs = branchOp.getSuccessorInputs(successor);
      for (auto [index, successorInput] : llvm::enumerate(successorInputs)) {
        SmallVector<Value> predecessorValues;
        branchOp.getPredecessorValues(successor, index, predecessorValues);
        std::optional<Type> consensusType =
            getConsensusType(ValueRange(predecessorValues));
        if (!consensusType || successorInput.getType() == *consensusType)
          continue;

        LDBG("Updating region-branch type for "
             << successorInput << " from " << successorInput.getType() << " to "
             << *consensusType);
        successorInput.setType(*consensusType);
        changed = true;
      }
    }
  }
}

} // namespace

// ============================================================================
// Main pass logic
// ============================================================================

LogicalResult insertRequireLayout(ModuleOp m) {
  OpBuilder builder(m.getContext());
  LDBG("insertRequireLayout");

  // --- Run backward dataflow analysis ---
  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  loadBaselineAnalyses(solver);
  solver.load<DotEncodingBackward>(symbolTable);
  if (failed(solver.initializeAndRun(m)))
    return failure();

  // --- Rewrite local_loads whose results require a dot-operand encoding ---
  m.walk([&](ttg::LocalLoadOp localLoadOp) {
    auto *lattice =
        solver.lookupState<DotEncodingLattice>(localLoadOp.getResult());
    if (!lattice || lattice->getValue().isUninitialized() ||
        lattice->getValue().isUnknown())
      return;

    auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
        lattice->getValue().getEncoding());
    if (!dotEnc)
      return;

    LDBG("local_load needs dot encoding: " << dotEnc);

    // Insert RequireLayoutOp for memdesc swizzling.
    auto sharedEnc = computeSharedEncFromDotEnc(dotEnc, localLoadOp);
    applyRequireLayout(sharedEnc, localLoadOp, builder);

    // Set local_load output type to #dot_op.
    auto resultType = cast<RankedTensorType>(localLoadOp.getType());
    auto newType = RankedTensorType::get(resultType.getShape(),
                                         resultType.getElementType(), dotEnc);
    localLoadOp->getResult(0).setType(newType);
  });

  // --- Fix region-branch result and block-argument types ---
  // After rewriting local_load results, region-branch edges may no longer have
  // matching source and destination types. Recompute those destination types
  // from the RegionBranch interfaces and only retag slots whose predecessors
  // all agree on the same type.
  m.walk<WalkOrder::PostOrder>([&](RegionBranchOpInterface branchOp) {
    updateRegionBranchTypes(branchOp);
  });

  // --- Clean up redundant convert_layout ops ---

  // Shortcut convert chains where the source already has the target type
  // (e.g. body_arg(#dot_op) -> cvt(#other) -> cvt(#dot_op) -> dot can be
  // replaced by body_arg -> dot).
  m.walk([&](tt::DotOp dotOp) {
    for (unsigned i = 0; i < 2; ++i) {
      Value operand = dotOp.getOperand(i);
      Value src = operand;
      while (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>())
        src = cvt.getSrc();
      if (src != operand && src.getType() == operand.getType())
        dotOp->setOperand(i, src);
    }
  });

  // Remove identity converts and DCE dead ones. Post-order walk allows
  // erasing the current op inside the callback.
  m.walk([&](ttg::ConvertLayoutOp cvt) {
    if (cvt.getSrc().getType() == cvt.getType()) {
      cvt.replaceAllUsesWith(cvt.getSrc());
      cvt.erase();
    } else if (cvt.getResult().use_empty()) {
      cvt.erase();
    }
  });

  return success();
}

struct TLXInsertRequireLayoutPass
    : public impl::TLXInsertRequireLayoutBase<TLXInsertRequireLayoutPass> {
public:
  using impl::TLXInsertRequireLayoutBase<
      TLXInsertRequireLayoutPass>::TLXInsertRequireLayoutBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (failed(tlx::insertRequireLayout(m)))
      signalPassFailure();
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
