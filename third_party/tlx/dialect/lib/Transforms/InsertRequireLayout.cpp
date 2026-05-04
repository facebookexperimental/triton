#include "IR/Dialect.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "tlx/dialect/include/Analysis/LayoutPropagation.h"
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

static bool
isTransparentDotUserBeforeConstraintMaterialization(Operation *op,
                                                    unsigned operandIndex) {
  // This is the pre-materialization half of the shared dot-layout policy. The
  // insert pass sees raw tt.dot users and the convert_layout chain that still
  // connects them to local_load. After those converts are rewritten into
  // explicit tlx.require_layout anchors, tlx-propagate-layout enforces the same
  // transparent-carrier policy from the tlx.require_layout anchors instead.
  if (auto dotOp = dyn_cast<tt::DotOp>(op))
    return operandIndex < 2 && operandIndex < dotOp->getNumOperands();

  return isa<ttg::ConvertLayoutOp>(op) || isTransparentLayoutCarrierOp(op);
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
      if (isTransparentDotUserBeforeConstraintMaterialization(op, index))
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
    // For RegionBranchTerminatorOpInterface (scf.yield) and
    // RegionBranchOpInterface (scf.for init args), allow the dot encoding
    // to propagate backward so local_load ops feeding loop-carried values
    // produce dot_op layout directly.
    if (isTransparentLayoutCarrierOp(operand.getOwner()))
      return;
    poisonUnhandledCase(operand);
  }

  void visitCallOperand(OpOperand &operand) override {
    poisonUnhandledCase(operand);
  }

  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
                               ArrayRef<BlockArgument> arguments) override {}
  void setToExitState(DotRewriteLattice *lattice) override {}

private:
  void poisonUnhandledCase(OpOperand &operand) {
    if (!isTrackedDotValue(operand.get()))
      return;

    auto *lattice = getLatticeElement(operand.get());
    DotRewriteState state = lattice->getValue();
    if (state.isUninitialized())
      return;

    ChangeResult changed = lattice->meet(DotRewriteState::getIllegal());
    propagateIfChanged(lattice, changed);
  }
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

static void materializeTensorRequireLayout(tt::DotOp dotOp,
                                           unsigned operandIndex,
                                           OpBuilder &builder) {
  Value operand = dotOp.getOperand(operandIndex);
  auto cvt = operand.getDefiningOp<ttg::ConvertLayoutOp>();
  if (!cvt)
    return;

  auto dstType = dyn_cast<RankedTensorType>(cvt.getType());
  if (!dstType || !isSupportedDotConstraintEncoding(dstType.getEncoding()))
    return;

  builder.setInsertionPoint(cvt);
  auto requireOp = tlx::RequireLayoutOp::create(builder, cvt.getLoc(),
                                                cvt.getType(), cvt.getSrc());
  dotOp->setOperand(operandIndex, requireOp.getResult());
  if (cvt.getResult().use_empty())
    cvt.erase();
}

static void materializeDotUserTensorConstraints(ModuleOp m,
                                                OpBuilder &builder) {
  m.walk([&](tt::DotOp dotOp) {
    for (unsigned i = 0; i < 2; ++i)
      materializeTensorRequireLayout(dotOp, i, builder);
  });
}

} // namespace

// ============================================================================
// Main pass logic
// ============================================================================

LogicalResult insertRequireLayout(ModuleOp m) {
  OpBuilder builder(m.getContext());
  LDBG("insertRequireLayout");

  // --- Run backward dataflow analysis ---
  // SparseBackwardDataFlowAnalysis requires a SymbolTableCollection even though
  // this analysis does not query symbol tables directly.
  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  loadBaselineAnalyses(solver);
  solver.load<DotRewriteBackward>(symbolTable);
  if (failed(solver.initializeAndRun(m)))
    return failure();

  // InsertRequireLayout owns constraint synthesis only:
  // 1. Discover dot-fed local_load ops and add the missing memdesc-side
  //    tlx.require_layout constraints for shared memory.
  // 2. Rewrite matched dot-path ttg.convert_layout ops into explicit tensor
  //    tlx.require_layout constraints.
  // 3. Leave tensor/register propagation, region-branch retagging, and final
  //    convert cleanup to tlx-propagate-layout and downstream cleanup passes.
  m.walk([&](ttg::LocalLoadOp localLoadOp) {
    auto *lattice =
        solver.lookupState<DotRewriteLattice>(localLoadOp.getResult());
    if (!lattice) {
      LDBG("local_load has NO lattice: " << localLoadOp);
      return;
    }
    if (lattice->getValue().isUninitialized()) {
      LDBG("local_load lattice UNINITIALIZED: " << localLoadOp);
      return;
    }

    if (lattice->getValue().isIllegal() || lattice->getValue().isConflict()) {
      LDBG("Skipping local_load rewrite due to state: " << lattice->getValue());
      localLoadOp->emitRemark()
          << "dot operand layout constraint cannot be folded into local_load "
             "because the value has incompatible users or conflicting dot "
             "requirements";
      return;
    }

    auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
        lattice->getValue().getEncoding());
    if (!dotEnc)
      return;

    LDBG("local_load needs dot encoding: " << dotEnc << " for: " << localLoadOp);

    // Insert RequireLayoutOp for memdesc swizzling.
    auto sharedEnc = computeSharedEncFromDotEnc(dotEnc, localLoadOp);
    applyRequireLayout(sharedEnc, localLoadOp, builder);
  });

  materializeDotUserTensorConstraints(m, builder);

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
