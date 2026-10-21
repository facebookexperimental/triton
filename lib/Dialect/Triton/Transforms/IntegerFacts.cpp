#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONMATERIALIZEINTEGERFACTS
#define GEN_PASS_DEF_TRITONERASEINTEGERREFINEMENTS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
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

static void materialize(ArrayRef<Predicate> predicates, Block *block,
                        Block::iterator position, DominanceInfo &dominance) {
  OpBuilder builder(block, position);
  for (Predicate predicate : predicates) {
    if (matchPattern(predicate.value, m_Constant()))
      continue;
    SmallVector<Value> pending{predicate.value};
    SmallVector<Value> inputs;
    DenseSet<Value> visited;
    while (!pending.empty()) {
      Value input = pending.pop_back_val();
      if (!visited.insert(input).second)
        continue;
      inputs.push_back(input);
      for (Operation *user : input.getUsers()) {
        auto candidate = dyn_cast<RefineOp>(user);
        if (!candidate || candidate.getValue() != input ||
            position == block->end() ||
            !dominance.properlyDominates(user, &*position))
          continue;
        pending.push_back(candidate.getResult());
      }
    }
    for (Value input : inputs) {
      auto existing = input.getDefiningOp<RefineOp>();
      if (existing && existing.getBound() == predicate.bound &&
          existing.getPredicate() == predicate.comparison)
        continue;
      auto inScope = [&](OpOperand &use) {
        if (auto refine = dyn_cast<RefineOp>(use.getOwner())) {
          // Chain value facts; bound operands keep their SSA identity.
          if (use.getOperandNumber() != 0 ||
              (refine.getBound() == predicate.bound &&
               refine.getPredicate() == predicate.comparison))
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
        Predicate predicate{loop.getInductionVar(), loop.getUpperBound(),
                            loop.getUnsignedCmp() ? arith::CmpIPredicate::ult
                                                  : arith::CmpIPredicate::slt};
        materialize({predicate}, loop.getBody(), loop.getBody()->begin(),
                    dominance);
      } else if (auto branch = dyn_cast<cf::CondBranchOp>(source)) {
        for (unsigned edge : {0u, 1u}) {
          SmallVector<Predicate> predicates;
          collectPredicates(branch.getCondition(), edge == 0, predicates);
          if (predicates.empty())
            continue;
          Block *destination = branch->getSuccessor(edge);
          if (!destination->getSinglePredecessor() ||
              branch.getTrueDest() == branch.getFalseDest()) {
            ValueRange operands = edge == 0 ? branch.getTrueDestOperands()
                                            : branch.getFalseDestOperands();
            SmallVector<Value> arguments(operands);
            OpBuilder builder(source);
            Block *edgeBlock = builder.createBlock(destination);
            cf::BranchOp::create(builder, source->getLoc(), destination,
                                 arguments);
            branch->setSuccessor(edgeBlock, edge);
            branch.getSuccessorOperands(edge).erase(0, arguments.size());
            destination = edgeBlock;
            dominance.invalidate();
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
