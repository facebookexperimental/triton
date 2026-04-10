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
// Backward dataflow analysis: propagate the required dot-operand encoding and
// rewrite legality from tt.DotOp operands backward through convert_layout
// chains and region-branch carriers to local_load ops.
//
// The analysis tracks both the desired dot encoding and whether rewriting the
// value is still legal. We union convert_layout source/result anchors so mixed
// uses that branch through sibling convert chains share the same legality
// state.
// ============================================================================

class DotRewriteState {
public:
  enum class Kind {
    Uninitialized,
    Required,
    Conflict,
    Illegal,
  };

  DotRewriteState() = default;
  explicit DotRewriteState(Attribute enc)
      : kind(Kind::Required), encoding(enc) {}

  static DotRewriteState getConflict() {
    DotRewriteState state;
    state.kind = Kind::Conflict;
    return state;
  }

  static DotRewriteState getIllegal() {
    DotRewriteState state;
    state.kind = Kind::Illegal;
    return state;
  }

  bool operator==(const DotRewriteState &rhs) const {
    return kind == rhs.kind && encoding == rhs.encoding;
  }

  bool isUninitialized() const { return kind == Kind::Uninitialized; }
  bool isRequired() const { return kind == Kind::Required; }
  bool isConflict() const { return kind == Kind::Conflict; }
  bool isIllegal() const { return kind == Kind::Illegal; }

  Attribute getEncoding() const {
    assert(isRequired() && "expected required dot encoding state");
    return *encoding;
  }

  void print(raw_ostream &os) const {
    if (isUninitialized()) {
      os << "<uninitialized>";
      return;
    }
    if (isConflict()) {
      os << "<conflict>";
      return;
    }
    if (isIllegal()) {
      os << "<illegal>";
      return;
    }
    if (isRequired()) {
      encoding->print(os);
      return;
    }
    llvm_unreachable("unknown dot rewrite state");
  }

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const DotRewriteState &state) {
    state.print(os);
    return os;
  }

  static DotRewriteState meet(const DotRewriteState &lhs,
                              const DotRewriteState &rhs) {
    if (lhs.isIllegal() || rhs.isIllegal())
      return getIllegal();
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    if (lhs.isConflict() || rhs.isConflict())
      return getConflict();
    return getConflict();
  }

  static DotRewriteState join(const DotRewriteState &lhs,
                              const DotRewriteState &rhs) {
    return meet(lhs, rhs);
  }

private:
  Kind kind = Kind::Uninitialized;
  std::optional<Attribute> encoding;
};

class DotRewriteLattice : public Lattice<DotRewriteState> {
public:
  using Lattice::Lattice;
};

static bool isTrackedDotValue(Value value) {
  return isa<RankedTensorType>(value.getType());
}

static bool isAllowedDotOperandUser(Operation *op, unsigned operandIndex) {
  if (auto dotOp = dyn_cast<tt::DotOp>(op))
    return operandIndex < 2 && operandIndex < dotOp->getNumOperands();

  return isa<ttg::ConvertLayoutOp, RegionBranchOpInterface,
             RegionBranchTerminatorOpInterface>(op);
}

class DotRewriteBackward
    : public SparseBackwardDataFlowAnalysis<DotRewriteLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  void initializeEquivalentLatticeAnchor(Operation *top) override {
    top->walk([&](ttg::ConvertLayoutOp cvt) {
      if (!isTrackedDotValue(cvt.getSrc()) ||
          !isTrackedDotValue(cvt.getResult()))
        return;
      unionLatticeAnchors<DotRewriteLattice>(cvt.getSrc(), cvt.getResult());
    });
  }

  LogicalResult
  visitOperation(Operation *op, ArrayRef<DotRewriteLattice *> operands,
                 ArrayRef<const DotRewriteLattice *> results) override {
    // Seed from tt.DotOp: propagate the required dot-operand encoding to
    // the values that define operands A and B.
    if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
      for (unsigned i = 0; i < 2; ++i) {
        auto type = cast<RankedTensorType>(dotOp.getOperand(i).getType());
        if (auto dotEnc =
                dyn_cast<ttg::DotOperandEncodingAttr>(type.getEncoding())) {
          ChangeResult changed = operands[i]->meet(DotRewriteState(dotEnc));
          propagateIfChanged(operands[i], changed);
        }
      }
      return success();
    }

    // If a tracked tensor value is used by an unsupported operation, the
    // require_layout rewrite is no longer legal for that entire carrier chain.
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!isTrackedDotValue(operand))
        continue;
      if (isAllowedDotOperandUser(op, index))
        continue;

      DotRewriteState operandState = operands[index]->getValue();
      if (operandState.isUninitialized())
        continue;

      ChangeResult changed =
          operands[index]->meet(DotRewriteState::getIllegal());
      propagateIfChanged(operands[index], changed);
    }

    return success();
  }

  void visitBranchOperand(OpOperand &operand) override {
    if (!isTrackedDotValue(operand.get()))
      return;

    auto *lattice = getLatticeElement(operand.get());
    DotRewriteState state = lattice->getValue();
    if (state.isUninitialized())
      return;

    ChangeResult changed = lattice->meet(DotRewriteState::getIllegal());
    propagateIfChanged(lattice, changed);
  }

  void visitCallOperand(OpOperand &operand) override {
    if (!isTrackedDotValue(operand.get()))
      return;

    auto *lattice = getLatticeElement(operand.get());
    DotRewriteState state = lattice->getValue();
    if (state.isUninitialized())
      return;

    ChangeResult changed = lattice->meet(DotRewriteState::getIllegal());
    propagateIfChanged(lattice, changed);
  }

  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
                               ArrayRef<BlockArgument> arguments) override {}
  void setToExitState(DotRewriteLattice *lattice) override {}
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
  solver.load<DotRewriteBackward>(symbolTable);
  if (failed(solver.initializeAndRun(m)))
    return failure();

  // Proposed long-term flow.
  // 1. InsertRequireLayout should discover dot-fed local_load ops and insert
  //    the missing memdesc-side tlx.require_layout constraints for shared
  //    memory.
  // 2. Tensor-side tlx.require_layout and tlx.release_layout constraints
  //    should then be propagated by tlx-propagate-layout, which already owns
  //    TLX layout propagation and lowering to ttg.convert_layout.
  // 3. The direct local_load result retagging below is a temporary bridge
  //    until tlx-propagate-layout owns register-side propagation for this
  //    AMD local_load-to-dot path.
  // --- Rewrite local_loads whose results require a dot-operand encoding ---
  m.walk([&](ttg::LocalLoadOp localLoadOp) {
    auto *lattice =
        solver.lookupState<DotRewriteLattice>(localLoadOp.getResult());
    if (!lattice || lattice->getValue().isUninitialized())
      return;

    if (lattice->getValue().isIllegal() || lattice->getValue().isConflict()) {
      LDBG("Skipping local_load rewrite due to state: " << lattice->getValue());
      return;
    }

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
