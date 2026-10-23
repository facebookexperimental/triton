#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/IntegerRange.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include <array>

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONMATERIALIZEINTEGERFACTS
#define GEN_PASS_DEF_TRITONERASEINTEGERREFINEMENTS
#define GEN_PASS_DEF_TRITONNARROWINTEGERRANGES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
class IntegerRangeRewriteListener : public RewriterBase::Listener {
public:
  explicit IntegerRangeRewriteListener(DataFlowSolver &solver)
      : solver(solver) {}

  void notifyOperationReplaced(Operation *op,
                               ValueRange replacements) override {
    auto overflow = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(op);
    if (!overflow || replacements.size() != 1)
      return;
    Operation *extension = replacements.front().getDefiningOp();
    if (!isa_and_nonnull<arith::ExtSIOp, arith::ExtUIOp>(extension))
      return;
    Operation *narrow = extension->getOperand(0).getDefiningOp();
    if (!narrow || narrow->getName() != op->getName())
      return;
    unsigned width = getElementTypeOrSelf(narrow->getResult(0).getType())
                         .getIntOrFloatBitWidth();
    auto flags = overflow.getOverflowAttr().getValue();
    for (Value value :
         llvm::concat<Value>(op->getOperands(), op->getResults())) {
      const auto *lattice =
          solver.lookupState<dataflow::IntegerValueRangeLattice>(value);
      if (!lattice || lattice->getValue().isUninitialized()) {
        flags = arith::IntegerOverflowFlags::none;
        break;
      }
      const auto &range = lattice->getValue().getValue();
      if (!range.smin().isSignedIntN(width) ||
          !range.smax().isSignedIntN(width))
        flags = arith::bitEnumClear(flags, arith::IntegerOverflowFlags::nsw);
      if (!range.umax().isIntN(width))
        flags = arith::bitEnumClear(flags, arith::IntegerOverflowFlags::nuw);
    }
    if (flags != overflow.getOverflowAttr().getValue())
      narrowedFlags[narrow] = flags;
  }

  void notifyOperationErased(Operation *op) override {
    narrowedFlags.erase(op);
    solver.eraseState(solver.getProgramPointAfter(op));
    for (Value result : op->getResults())
      solver.eraseState(result);
  }

  void updateNarrowedOverflowFlags() {
    // Wide no-wrap guarantees do not imply no-wrap at the narrowed width.
    for (auto [op, flags] : narrowedFlags) {
      auto overflow = cast<arith::ArithIntegerOverflowFlagsInterface>(op);
      op->setAttr(
          overflow.getIntegerOverflowAttrName(),
          arith::IntegerOverflowFlagsAttr::get(op->getContext(), flags));
    }
  }

private:
  DenseMap<Operation *, arith::IntegerOverflowFlags> narrowedFlags;
  DataFlowSolver &solver;
};

static bool preservesIntegerWidth(Operation *op, DataFlowSolver &solver) {
  if (isa<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp>(op)) {
    const auto *shift = solver.lookupState<dataflow::IntegerValueRangeLattice>(
        op->getOperand(1));
    // A valid wide shift must not become an out-of-range narrow shift.
    if (!shift || shift->getValue().isUninitialized() ||
        shift->getValue().getValue().umax().uge(32))
      return false;
  }
  if (!isa<arith::DivUIOp, arith::RemUIOp, arith::CeilDivUIOp, arith::ShRUIOp>(
          op))
    return true;
  // Sign-extension changes unsigned division/remainder and logical shifts.
  for (Value operand : op->getOperands()) {
    const auto *range =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(operand);
    if (!range || range->getValue().isUninitialized() ||
        range->getValue().getValue().umax().getActiveBits() > 32)
      return false;
  }
  return true;
}

struct TritonNarrowIntegerRangesPass
    : impl::TritonNarrowIntegerRangesBase<TritonNarrowIntegerRangesPass> {
  void runOnOperation() override {
    auto solver = createIntegerFactSolver(getOperation());
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();
    IntegerRangeRewriteListener listener(*solver);
    RewritePatternSet patterns(&getContext());
    arith::populateIntRangeNarrowingPatterns(patterns, *solver, {32});
    arith::populateControlFlowValuesNarrowingPatterns(patterns, *solver, {32});
    SmallVector<Operation *> candidates;
    getOperation()->walk([&](Operation *op) {
      if ((isa<arith::ArithDialect>(op->getDialect()) ||
           isa<LoopLikeOpInterface>(op)) &&
          preservesIntegerWidth(op, *solver))
        candidates.push_back(op);
    });
    // Keep block-argument storage stable while range facts are live.
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(false)
        .enableFolding(false)
        .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled)
        .setStrictness(GreedyRewriteStrictness::ExistingOps)
        .setListener(&listener);
    if (failed(
            applyOpPatternsGreedily(candidates, std::move(patterns), config)))
      return signalPassFailure();
    // New ops stay unvisited and unfolded until their flags are repaired.
    listener.updateNarrowedOverflowFlags();
  }
};

struct Predicate {
  Value value;
  Value bound;
  arith::CmpIPredicate comparison;
};

static arith::CmpIPredicate swapPredicate(arith::CmpIPredicate predicate) {
  using P = arith::CmpIPredicate;
  switch (predicate) {
  case P::eq:
    return P::eq;
  case P::ne:
    return P::ne;
  case P::slt:
    return P::sgt;
  case P::sle:
    return P::sge;
  case P::sgt:
    return P::slt;
  case P::sge:
    return P::sle;
  case P::ult:
    return P::ugt;
  case P::ule:
    return P::uge;
  case P::ugt:
    return P::ult;
  case P::uge:
    return P::ule;
  }
  llvm_unreachable("invalid integer comparison predicate");
}

// Only conjunctions implied by the selected edge can be split.
static void collectPredicates(Value condition, bool truth,
                              SmallVectorImpl<Predicate> &predicates) {
  SmallVector<std::pair<Value, bool>> pending{{condition, truth}};
  DenseSet<std::pair<Value, unsigned>> visited;
  while (!pending.empty()) {
    auto [value, expected] = pending.pop_back_val();
    if (!visited.insert({value, expected}).second)
      continue;
    Operation *op = value.getDefiningOp();
    if (!op)
      continue;
    if ((expected && isa<arith::AndIOp>(op)) ||
        (!expected && isa<arith::OrIOp>(op))) {
      pending.emplace_back(op->getOperand(0), expected);
      pending.emplace_back(op->getOperand(1), expected);
    } else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      if (!isa<IntegerType>(cmp.getLhs().getType()))
        continue;
      arith::CmpIPredicate pred = cmp.getPredicate();
      if (!expected)
        pred = arith::invertPredicate(pred);
      predicates.push_back({cmp.getLhs(), cmp.getRhs(), pred});
      predicates.push_back({cmp.getRhs(), cmp.getLhs(), swapPredicate(pred)});
    }
  }
}

static Value refinementIdentity(Value value) {
  while (auto refine = value.getDefiningOp<RefineOp>())
    value = refine.getValue();
  return value;
}

static void materialize(ArrayRef<Predicate> predicates, Block *block,
                        Block::iterator position, DominanceInfo &dominance) {
  // Existing entry refinements already dominate the remaining scope.
  while (position != block->end() && isa<RefineOp>(*position))
    ++position;
  OpBuilder builder(block, position);
  for (Predicate predicate : predicates) {
    if (matchPattern(predicate.value, m_Constant()))
      continue;
    auto matches = [&](RefineOp refine) {
      return refinementIdentity(refine.getBound()) ==
                 refinementIdentity(predicate.bound) &&
             refine.getPredicate() == predicate.comparison;
    };
    bool known = false;
    for (Value value = predicate.value;
         auto refine = value.getDefiningOp<RefineOp>();
         value = refine.getValue())
      known |= matches(refine);
    SmallVector<std::pair<Value, bool>> pending{{predicate.value, known}};
    SmallVector<std::pair<Value, bool>> inputs;
    DenseSet<Value> visited;
    while (!pending.empty()) {
      auto [input, known] = pending.pop_back_val();
      if (!visited.insert(input).second)
        continue;
      inputs.emplace_back(input, known);
      for (Operation *user : input.getUsers()) {
        auto candidate = dyn_cast<RefineOp>(user);
        if (!candidate || candidate.getValue() != input ||
            position == block->end() ||
            !dominance.properlyDominates(user, &*position))
          continue;
        pending.emplace_back(candidate.getResult(),
                             known || matches(candidate));
      }
    }
    for (auto [input, known] : inputs) {
      if (known)
        continue;
      auto inScope = [&](OpOperand &use) {
        if (auto refine = dyn_cast<RefineOp>(use.getOwner())) {
          // Chain value facts; bound operands keep their SSA identity.
          if (use.getOperandNumber() != 0 || matches(refine))
            return false;
        }
        return position != block->end() &&
               dominance.dominates(&*position, use.getOwner());
      };
      if (!llvm::any_of(input.getUses(), inScope))
        continue;
      auto refinement = RefineOp::create(builder, predicate.value.getLoc(),
                                         predicate.value.getType(), input,
                                         predicate.bound, predicate.comparison);
      // A sibling refinement can be poison even when this input is defined.
      input.replaceUsesWithIf(refinement.getResult(), inScope);
    }
  }
}

struct TritonMaterializeIntegerFactsPass
    : impl::TritonMaterializeIntegerFactsBase<
          TritonMaterializeIntegerFactsPass> {
  void runOnOperation() override {
    SmallVector<Operation *> sources;
    getOperation()->walk([&](Operation *op) {
      if (isa<LLVM::AssumeOp, scf::IfOp, scf::ForOp, cf::CondBranchOp>(op))
        sources.push_back(op);
    });
    DominanceInfo dominance(getOperation());
    for (Operation *source : sources) {
      if (auto assume = dyn_cast<LLVM::AssumeOp>(source)) {
        SmallVector<Predicate> predicates;
        collectPredicates(assume.getCond(), true, predicates);
        materialize(predicates, source->getBlock(),
                    std::next(source->getIterator()), dominance);
      } else if (auto branch = dyn_cast<scf::IfOp>(source)) {
        for (bool truth : {true, false}) {
          Region &region =
              truth ? branch.getThenRegion() : branch.getElseRegion();
          if (region.empty())
            continue;
          SmallVector<Predicate> predicates;
          collectPredicates(branch.getCondition(), truth, predicates);
          materialize(predicates, &region.front(), region.front().begin(),
                      dominance);
        }
      } else if (auto loop = dyn_cast<scf::ForOp>(source)) {
        if (!isa<IntegerType>(loop.getInductionVar().getType()))
          continue;
        // scf.for requires a positive step and a representable exit value.
        std::array predicates{
            Predicate{loop.getInductionVar(), loop.getLowerBound(),
                      loop.getUnsignedCmp() ? arith::CmpIPredicate::uge
                                            : arith::CmpIPredicate::sge},
            Predicate{loop.getInductionVar(), loop.getUpperBound(),
                      loop.getUnsignedCmp() ? arith::CmpIPredicate::ult
                                            : arith::CmpIPredicate::slt}};
        materialize(predicates, loop.getBody(), loop.getBody()->begin(),
                    dominance);
      } else if (auto branch = dyn_cast<cf::CondBranchOp>(source)) {
        for (unsigned edge : {0u, 1u}) {
          SmallVector<Predicate> predicates;
          collectPredicates(branch.getCondition(), edge == 0, predicates);
          if (predicates.empty())
            continue;
          Block *destination = branch->getSuccessor(edge);
          ValueRange operands = edge == 0 ? branch.getTrueDestOperands()
                                          : branch.getFalseDestOperands();
          if (!destination->getSinglePredecessor() ||
              branch.getTrueDest() == branch.getFalseDest()) {
            SmallVector<Value> arguments(operands);
            OpBuilder builder(source);
            Block *edgeBlock = builder.createBlock(destination);
            cf::BranchOp::create(builder, source->getLoc(), destination,
                                 arguments);
            branch->setSuccessor(edgeBlock, edge);
            branch.getSuccessorOperands(edge).erase(0, arguments.size());
            destination = edgeBlock;
            dominance.invalidate();
          } else if (dominance.properlyDominates(branch->getBlock(),
                                                 destination)) {
            // Unique incoming values dominate every successor argument use.
            for (auto [operand, argument] :
                 llvm::zip_equal(operands, destination->getArguments()))
              argument.replaceAllUsesWith(operand);
          }
          materialize(predicates, destination, destination->begin(), dominance);
        }
      }
    }
  }
};

struct TritonEraseIntegerRefinementsPass
    : impl::TritonEraseIntegerRefinementsBase<
          TritonEraseIntegerRefinementsPass> {
  void runOnOperation() override {
    getOperation()->walk([](RefineOp op) {
      op.getResult().replaceAllUsesWith(op.getValue());
      op.erase();
    });
  }
};
} // namespace
} // namespace mlir::triton
