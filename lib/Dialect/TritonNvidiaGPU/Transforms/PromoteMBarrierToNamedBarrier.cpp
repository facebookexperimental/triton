#include "mlir/IR/Matchers.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/NamedBarrier.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPROMOTEMBARRIERTONAMEDBARRIERPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

template <typename OpTy> struct IndexedBarrierUse {
  OpTy op;
  Value index;
};

struct BarrierCandidate {
  BarrierCandidate(ttg::LocalAllocOp alloc, unsigned numBarriers)
      : alloc(alloc), numBarriers(numBarriers) {}

  ttg::LocalAllocOp alloc;
  unsigned numBarriers;
  SmallVector<IndexedBarrierUse<ttng::InitBarrierOp>> inits;
  SmallVector<IndexedBarrierUse<ttng::InvalBarrierOp>> invalidations;
  SmallVector<IndexedBarrierUse<ttng::ArriveBarrierOp>> arrives;
  SmallVector<IndexedBarrierUse<ttng::WaitBarrierOp>> waits;
  SmallVector<ttg::LocalDeallocOp> deallocations;
  SmallVector<Operation *> views;
  SmallVector<std::pair<ttg::WarpSpecializePartitionsOp, Value>> captures;
  llvm::SmallDenseSet<Value> visitedValues;
  bool hasUnknownUse = false;
};

std::optional<unsigned> getBarrierCount(ttg::LocalAllocOp alloc) {
  ttg::MemDescType type = alloc.getType();
  if (alloc.getSrc() || !type.getElementType().isInteger(64))
    return std::nullopt;
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() == 1)
    return shape.front() == 1 ? std::optional<unsigned>(1) : std::nullopt;
  if (shape.size() != 2 || shape.back() != 1 || shape.front() <= 0)
    return std::nullopt;
  return static_cast<unsigned>(shape.front());
}

Region *getWarpSpecializeTaskRegion(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    // A top-level region (ModuleOp's) has no parent op. Any barrier not
    // nested in a warp-specialize partition reaches this null rather than a
    // partitions op, so return null here instead of dereferencing it in the
    // isa<> and the getParentRegion() below.
    if (!parent)
      return nullptr;
    if (isa<ttg::WarpSpecializePartitionsOp>(parent))
      return region;
    if (auto warpSpecialize = dyn_cast<ttg::WarpSpecializeOp>(parent)) {
      if (region == &warpSpecialize.getDefaultRegion())
        return region;
    }
    region = parent->getParentRegion();
  }
  return nullptr;
}

ttg::WarpSpecializeOp getWarpSpecializeOwner(Region *taskRegion) {
  if (auto warpSpecialize =
          dyn_cast<ttg::WarpSpecializeOp>(taskRegion->getParentOp()))
    return warpSpecialize;
  auto partitions =
      cast<ttg::WarpSpecializePartitionsOp>(taskRegion->getParentOp());
  return partitions.getParentOp();
}

using ttg::resolveWarpSpecializeCapture;

bool isWarpUniformValue(Value value, llvm::SmallDenseSet<Value> &visiting) {
  if (!visiting.insert(value).second)
    return true;
  if (Operation *def = value.getDefiningOp()) {
    if (def->hasTrait<OpTrait::ConstantLike>())
      return true;
    if (def->getName().getDialectNamespace() == "arith")
      return llvm::all_of(def->getOperands(), [&](Value operand) {
        return isWarpUniformValue(operand, visiting);
      });
    return isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(def);
  }

  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return false;
  Operation *parent = arg.getOwner()->getParentOp();
  if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent))
    return isWarpUniformValue(
        partitions.getExplicitCaptures()[arg.getArgNumber()], visiting);
  if (isa<triton::FuncOp>(parent))
    return true;
  if (auto loop = dyn_cast<scf::ForOp>(parent)) {
    if (arg != loop.getInductionVar())
      return false;
    return isWarpUniformValue(loop.getLowerBound(), visiting) &&
           isWarpUniformValue(loop.getUpperBound(), visiting) &&
           isWarpUniformValue(loop.getStep(), visiting);
  }
  return false;
}

// A value is CTA-uniform if every thread in the CTA computes the same value.
// Loop-carried values are accepted only under the induction hypothesis in
// `assumedUniform` (see proveWhileUniform); anything else unprovable --
// thread-indexed values, loads, non-induction scf.for carriers -- returns
// false. Conservative: rejection only misses promotions, never miscompiles.
bool isCTAUniformValue(Value value, llvm::SmallDenseSet<Value> &visiting,
                       const llvm::SmallDenseSet<Value> &assumedUniform) {
  if (!visiting.insert(value).second)
    return true;
  if (assumedUniform.contains(value))
    return true;
  if (Operation *def = value.getDefiningOp()) {
    if (def->hasTrait<OpTrait::ConstantLike>())
      return true;
    if (def->getName().getDialectNamespace() == "arith")
      return llvm::all_of(def->getOperands(), [&](Value operand) {
        return isCTAUniformValue(operand, visiting, assumedUniform);
      });
    return isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(def);
  }

  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return false;
  Operation *parent = arg.getOwner()->getParentOp();
  if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent))
    return isCTAUniformValue(
        partitions.getExplicitCaptures()[arg.getArgNumber()], visiting,
        assumedUniform);
  if (isa<triton::FuncOp>(parent))
    return true;
  if (auto loop = dyn_cast<scf::ForOp>(parent)) {
    if (arg != loop.getInductionVar())
      return false;
    return isCTAUniformValue(loop.getLowerBound(), visiting, assumedUniform) &&
           isCTAUniformValue(loop.getUpperBound(), visiting, assumedUniform) &&
           isCTAUniformValue(loop.getStep(), visiting, assumedUniform);
  }
  return false;
}

// Prove an scf.while executes uniformly (same trip count on every warp) by
// induction over its loop-carried values: the inits are uniform with no
// assumptions (base case); the forwarded values, the yielded updates and the
// condition are uniform assuming the carried values are (inductive step).
// After-region args are justified by the forwarded values proven just above:
// they are exactly the condition's operands, and dominance guarantees the
// before region cannot reference after-region args. On success the carried
// values and results are added to `provenUniform`: they are warp-uniform on
// every iteration and may be used by enclosing guards.
bool proveWhileUniform(scf::WhileOp loop,
                       llvm::SmallDenseSet<Value> &provenUniform) {
  auto condOp =
      dyn_cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
  auto yieldOp =
      dyn_cast<scf::YieldOp>(loop.getAfter().front().getTerminator());
  if (!condOp || !yieldOp ||
      loop.getBeforeArguments().size() != loop.getInits().size() ||
      loop.getAfterArguments().size() != condOp.getArgs().size() ||
      yieldOp.getResults().size() != loop.getInits().size())
    return false;

  llvm::SmallDenseSet<Value> assumed = provenUniform;
  for (Value arg : loop.getBeforeArguments())
    assumed.insert(arg);
  for (Value arg : loop.getAfterArguments())
    assumed.insert(arg);

  for (Value init : loop.getInits()) {
    llvm::SmallDenseSet<Value> visiting;
    if (!isCTAUniformValue(init, visiting, provenUniform))
      return false;
  }
  for (Value forwarded : condOp.getArgs()) {
    llvm::SmallDenseSet<Value> visiting;
    if (!isCTAUniformValue(forwarded, visiting, assumed))
      return false;
  }
  for (Value updated : yieldOp.getResults()) {
    llvm::SmallDenseSet<Value> visiting;
    if (!isCTAUniformValue(updated, visiting, assumed))
      return false;
  }
  {
    llvm::SmallDenseSet<Value> visiting;
    if (!isCTAUniformValue(condOp.getCondition(), visiting, assumed))
      return false;
  }
  for (Value arg : loop.getBeforeArguments())
    provenUniform.insert(arg);
  for (Value arg : loop.getAfterArguments())
    provenUniform.insert(arg);
  for (Value result : loop.getResults())
    provenUniform.insert(result);
  return true;
}

bool isWarpUniform(Operation *op) {
  Region *taskRegion = getWarpSpecializeTaskRegion(op);
  if (!taskRegion)
    return false;
  // First pass, outside in: prove every enclosing scf.while by induction,
  // outermost first so inner inits may reference proven outer carriers (an
  // outer loop's inits are defined before it and can never use inner
  // results). Carried uniformity is what lets enclosing guards reference loop
  // counters.
  llvm::SmallDenseSet<Value> provenUniform;
  SmallVector<scf::WhileOp> whileAncestors;
  for (Operation *parent = op->getParentOp();
       parent && parent != taskRegion->getParentOp();
       parent = parent->getParentOp()) {
    if (auto loop = dyn_cast<scf::WhileOp>(parent))
      whileAncestors.push_back(loop);
  }
  for (scf::WhileOp loop : llvm::reverse(whileAncestors)) {
    if (!proveWhileUniform(loop, provenUniform))
      return false;
  }
  // Second pass: every ancestor must preserve uniform execution. scf.for
  // repeats uniformly with uniform bounds; scf.if runs one branch uniformly
  // with a uniform condition; scf.while was proven above. Anything else --
  // scf.index_switch, an unstructured cf.cond_br region -- may run for a
  // subset, and a promoted named barrier would then wait on warps that never
  // arrive. Allow-list rather than deny-list: a deny-list silently
  // over-promotes every time a new region op appears.
  //
  // Uniformity here is intra-task only: a guard uniform within each task may
  // still take different branches across tasks. Cross-task agreement is
  // proven separately by guardsAgree, not here.
  for (Operation *parent = op->getParentOp();
       parent && parent != taskRegion->getParentOp();
       parent = parent->getParentOp()) {
    if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      // The trip count must also be the same for every warp, or the arrivals
      // will not pair up across iterations. Cross-task trip agreement comes
      // from the iteration-space equivalence proof, not from here.
      llvm::SmallDenseSet<Value> visiting;
      if (!isCTAUniformValue(loop.getLowerBound(), visiting, provenUniform) ||
          !isCTAUniformValue(loop.getUpperBound(), visiting, provenUniform) ||
          !isCTAUniformValue(loop.getStep(), visiting, provenUniform))
        return false;
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      llvm::SmallDenseSet<Value> visiting;
      if (!isCTAUniformValue(ifOp.getCondition(), visiting, provenUniform))
        return false;
      continue;
    }
    if (isa<scf::WhileOp>(parent))
      continue;
    return false;
  }
  return true;
}

bool haveMatchingLoopNests(Operation *arrive, Operation *wait) {
  Region *arrivePartition = getWarpSpecializeTaskRegion(arrive);
  Region *waitPartition = getWarpSpecializeTaskRegion(wait);
  if (!arrivePartition || !waitPartition || arrivePartition == waitPartition ||
      getWarpSpecializeOwner(arrivePartition) !=
          getWarpSpecializeOwner(waitPartition))
    return false;

  auto getLoops = [](Operation *op, Region *partition) {
    SmallVector<Operation *> loops;
    for (Operation *parent = op->getParentOp();
         parent != partition->getParentOp(); parent = parent->getParentOp())
      if (isa<scf::ForOp, scf::WhileOp>(parent))
        loops.push_back(parent);
    return loops;
  };
  auto arriveLoops = getLoops(arrive, arrivePartition);
  auto waitLoops = getLoops(wait, waitPartition);
  // Repeated waits may observe one completed mbarrier phase. A named wait
  // consumes a new arrival, so uniform execution alone does not establish
  // matching numbers of arrivals and waits.
  if (arriveLoops.size() != waitLoops.size())
    return false;
  SmallVector<scf::ForOp> arriveForLoops;
  SmallVector<scf::ForOp> waitForLoops;
  for (auto [arriveParent, waitParent] : llvm::zip(arriveLoops, waitLoops)) {
    if (arriveParent->getName() != waitParent->getName())
      return false;
    // The guard-chain check separately proves while-condition equivalence.
    if (isa<scf::WhileOp>(arriveParent))
      continue;
    arriveForLoops.push_back(cast<scf::ForOp>(arriveParent));
    waitForLoops.push_back(cast<scf::ForOp>(waitParent));
  }
  return ttg::haveEquivalentWarpSpecializeLoopNests(arriveForLoops,
                                                    waitForLoops);
}

// Null (unpredicated) counts as always-true, preserving the previous behavior
// of rejecting only present predicates.
bool isAlwaysTrue(Value pred) {
  if (!pred)
    return true;
  auto constant = pred.getDefiningOp<arith::ConstantOp>();
  auto value = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : nullptr;
  return value && !value.getValue().isZero();
}

bool matchConstantInt(Value value, int64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) && expected >= 0 &&
         constant.getLimitedValue() == expected;
}

Value stripIntegerCasts(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (!isa<arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp>(def))
      break;
    value = def->getOperand(0);
  }
  return value;
}

Value matchBinaryOperandWithConstant(Operation *op, int64_t expected) {
  if (matchConstantInt(op->getOperand(0), expected))
    return op->getOperand(1);
  if (matchConstantInt(op->getOperand(1), expected))
    return op->getOperand(0);
  return {};
}

struct IterationCounter {
  scf::ForOp loop;
  Value value;
  uint64_t initialValue;
  uint64_t increment;
};

std::optional<IterationCounter> matchIterationCounter(Value counter) {
  counter = stripIntegerCasts(counter);
  if (auto div = counter.getDefiningOp<arith::DivUIOp>()) {
    auto sub = stripIntegerCasts(div.getLhs()).getDefiningOp<arith::SubIOp>();
    if (!sub)
      return std::nullopt;
    Value induction = stripIntegerCasts(sub.getLhs());
    auto arg = dyn_cast<BlockArgument>(induction);
    auto loop = arg ? dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())
                    : scf::ForOp();
    if (!loop || induction != loop.getInductionVar() ||
        stripIntegerCasts(sub.getRhs()) !=
            stripIntegerCasts(loop.getLowerBound()) ||
        stripIntegerCasts(div.getRhs()) != stripIntegerCasts(loop.getStep()))
      return std::nullopt;
    // (iv - lb) / step counts 0, 1, 2, ... only for a positive step: scf.for
    // permits a negative step, for which iv - lb is negative and the unsigned
    // division wraps instead of tracking the iteration.
    APInt step;
    if (!matchPattern(resolveWarpSpecializeCapture(loop.getStep()),
                      m_ConstantInt(&step)) ||
        !step.isStrictlyPositive())
      return std::nullopt;
    return IterationCounter{loop, counter, /*initialValue=*/0,
                            /*increment=*/1};
  }

  auto arg = dyn_cast<BlockArgument>(counter);
  if (!arg)
    return std::nullopt;
  auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!loop)
    return std::nullopt;

  auto iterArgs = loop.getRegionIterArgs();
  auto it = llvm::find(iterArgs, counter);
  if (it == iterArgs.end())
    return std::nullopt;
  unsigned index = std::distance(iterArgs.begin(), it);
  APInt initialValue;
  if (!matchPattern(loop.getInitArgs()[index], m_ConstantInt(&initialValue)) ||
      initialValue.isNegative() || initialValue.getActiveBits() > 32)
    return std::nullopt;

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto add = yield.getResults()[index].getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;
  Value incrementValue;
  if (add.getLhs() == counter)
    incrementValue = add.getRhs();
  else if (add.getRhs() == counter)
    incrementValue = add.getLhs();
  else
    return std::nullopt;
  APInt increment;
  if (!matchPattern(incrementValue, m_ConstantInt(&increment)) ||
      increment.isZero() || increment.isNegative() ||
      increment.getActiveBits() > 32)
    return std::nullopt;
  return IterationCounter{loop, counter, initialValue.getZExtValue(),
                          increment.getZExtValue()};
}

struct CounterExpression {
  IterationCounter counter;
  Value value;
  uint64_t offset;
};

std::optional<CounterExpression> matchCounterExpression(Value value) {
  value = stripIntegerCasts(value);
  Value counter = value;
  uint64_t offset = 0;
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    APInt constant;
    if (matchPattern(add.getLhs(), m_ConstantInt(&constant)))
      counter = add.getRhs();
    else if (matchPattern(add.getRhs(), m_ConstantInt(&constant)))
      counter = add.getLhs();
    else
      return std::nullopt;
    if (constant.isNegative() || constant.getActiveBits() > 32)
      return std::nullopt;
    offset = constant.getZExtValue();
  }
  std::optional<IterationCounter> iteration = matchIterationCounter(counter);
  if (!iteration)
    return std::nullopt;
  return CounterExpression{*iteration, value, offset};
}

bool areEquivalentLoopValues(
    Value lhs, Value rhs,
    llvm::SmallVectorImpl<std::pair<Value, Value>> &assumed, unsigned depth,
    const llvm::DenseMap<Value, Value> *positionalIVs = nullptr);

// Loop-carried values in different loops are equivalent when the loops
// produce the same value sequences: same bounds for scf.for induction
// variables; same inits, forwarded values and updates (by induction) for
// scf.while carriers. Pairs under comparison are recorded in `assumed` so
// the recursion terminates. Only same-role, same-index pairs are considered:
// cross-role pairs carry different values in general.
bool areEquivalentLoopCarriedValues(
    BlockArgument lhs, BlockArgument rhs,
    llvm::SmallVectorImpl<std::pair<Value, Value>> &assumed, unsigned depth,
    const llvm::DenseMap<Value, Value> *positionalIVs = nullptr) {
  if (llvm::is_contained(assumed, std::make_pair(Value(lhs), Value(rhs))))
    return true;
  auto lhsFor = dyn_cast<scf::ForOp>(lhs.getOwner()->getParentOp());
  auto rhsFor = dyn_cast<scf::ForOp>(rhs.getOwner()->getParentOp());
  if (lhsFor && rhsFor) {
    if (lhs != lhsFor.getInductionVar() || rhs != rhsFor.getInductionVar())
      return false;
    // With a positional map (guard comparison), equal bounds are not enough:
    // counters from different nest positions fire different numbers of times
    // (an outer-loop guard fires per inner iteration; triangular inner bounds
    // skew the counts further), so the induction variables must correspond
    // positionally. Counters outside the map only agree when identical, which
    // the lhs == rhs check in areEquivalentLoopValues already handled.
    if (positionalIVs && positionalIVs->lookup(lhs) != rhs)
      return false;
    return areEquivalentLoopValues(lhsFor.getLowerBound(),
                                   rhsFor.getLowerBound(), assumed, depth + 1,
                                   positionalIVs) &&
           areEquivalentLoopValues(lhsFor.getUpperBound(),
                                   rhsFor.getUpperBound(), assumed, depth + 1,
                                   positionalIVs) &&
           areEquivalentLoopValues(lhsFor.getStep(), rhsFor.getStep(), assumed,
                                   depth + 1, positionalIVs);
  }
  auto lhsWhile = dyn_cast<scf::WhileOp>(lhs.getOwner()->getParentOp());
  auto rhsWhile = dyn_cast<scf::WhileOp>(rhs.getOwner()->getParentOp());
  if (!lhsWhile || !rhsWhile)
    return false;
  bool lhsBefore = lhs.getOwner() == &lhsWhile.getBefore().front();
  bool rhsBefore = rhs.getOwner() == &rhsWhile.getBefore().front();
  bool lhsAfter = lhs.getOwner() == &lhsWhile.getAfter().front();
  bool rhsAfter = rhs.getOwner() == &rhsWhile.getAfter().front();
  if (lhs.getArgNumber() != rhs.getArgNumber() || lhsBefore != rhsBefore ||
      lhsAfter != rhsAfter || (!lhsBefore && !lhsAfter) ||
      lhsWhile.getInits().size() != rhsWhile.getInits().size())
    return false;
  auto lhsCond =
      dyn_cast<scf::ConditionOp>(lhsWhile.getBefore().front().getTerminator());
  auto rhsCond =
      dyn_cast<scf::ConditionOp>(rhsWhile.getBefore().front().getTerminator());
  auto lhsYield =
      dyn_cast<scf::YieldOp>(lhsWhile.getAfter().front().getTerminator());
  auto rhsYield =
      dyn_cast<scf::YieldOp>(rhsWhile.getAfter().front().getTerminator());
  if (!lhsCond || !rhsCond || !lhsYield || !rhsYield ||
      lhsCond.getArgs().size() != rhsCond.getArgs().size() ||
      lhsYield.getResults().size() != rhsYield.getResults().size() ||
      lhsCond.getArgs().size() != lhsWhile.getInits().size() ||
      lhsYield.getResults().size() != lhsWhile.getInits().size())
    return false;
  // The induction hypothesis is scoped to this proof: `assumed` is shared by
  // reference across the whole top-level query, so hypotheses left behind --
  // by a failed proof or a successful one -- would short-circuit unrelated
  // later comparisons to true.
  size_t assumedSize = assumed.size();
  for (unsigned i = 0, e = lhsWhile.getInits().size(); i < e; ++i) {
    assumed.push_back(
        {lhsWhile.getBeforeArguments()[i], rhsWhile.getBeforeArguments()[i]});
    assumed.push_back(
        {lhsWhile.getAfterArguments()[i], rhsWhile.getAfterArguments()[i]});
  }
  auto equivalent = [&](Value lhsValue, Value rhsValue) {
    return areEquivalentLoopValues(lhsValue, rhsValue, assumed, depth + 1,
                                   positionalIVs);
  };
  bool result =
      llvm::all_of(llvm::zip(lhsWhile.getInits(), rhsWhile.getInits()),
                   [&](auto operands) {
                     return equivalent(std::get<0>(operands),
                                       std::get<1>(operands));
                   }) &&
      llvm::all_of(llvm::zip(lhsCond.getArgs(), rhsCond.getArgs()),
                   [&](auto operands) {
                     return equivalent(std::get<0>(operands),
                                       std::get<1>(operands));
                   }) &&
      llvm::all_of(llvm::zip(lhsYield.getResults(), rhsYield.getResults()),
                   [&](auto operands) {
                     return equivalent(std::get<0>(operands),
                                       std::get<1>(operands));
                   });
  assumed.resize(assumedSize);
  return result;
}

bool areEquivalentLoopValues(
    Value lhs, Value rhs,
    llvm::SmallVectorImpl<std::pair<Value, Value>> &assumed, unsigned depth,
    const llvm::DenseMap<Value, Value> *positionalIVs) {
  if (depth > 32)
    return false;
  lhs = resolveWarpSpecializeCapture(lhs);
  rhs = resolveWarpSpecializeCapture(rhs);
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;

  auto lhsArg = dyn_cast<BlockArgument>(lhs);
  auto rhsArg = dyn_cast<BlockArgument>(rhs);
  if (lhsArg || rhsArg) {
    if (lhsArg && rhsArg &&
        areEquivalentLoopCarriedValues(lhsArg, rhsArg, assumed, depth,
                                       positionalIVs))
      return true;
    // Same block as well as same op: an scf.while's before[i] and after[i]
    // share a parent op and argument number but carry different values (the
    // before region also executes one extra time).
    return lhsArg && rhsArg && lhsArg.getArgNumber() == rhsArg.getArgNumber() &&
           lhsArg.getOwner() == rhsArg.getOwner() &&
           lhsArg.getOwner()->getParentOp() == rhsArg.getOwner()->getParentOp();
  }

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef || lhsDef->getName() != rhsDef->getName() ||
      lhsDef->getNumOperands() != rhsDef->getNumOperands())
    return false;

  if (auto lhsConstant = dyn_cast<arith::ConstantOp>(lhsDef)) {
    auto rhsConstant = dyn_cast<arith::ConstantOp>(rhsDef);
    return rhsConstant && lhsConstant.getValue() == rhsConstant.getValue();
  }
  if (auto lhsProgramId = dyn_cast<triton::GetProgramIdOp>(lhsDef)) {
    auto rhsProgramId = dyn_cast<triton::GetProgramIdOp>(rhsDef);
    return rhsProgramId && lhsProgramId.getAxis() == rhsProgramId.getAxis();
  }
  if (auto lhsNumPrograms = dyn_cast<triton::GetNumProgramsOp>(lhsDef)) {
    auto rhsNumPrograms = dyn_cast<triton::GetNumProgramsOp>(rhsDef);
    return rhsNumPrograms &&
           lhsNumPrograms.getAxis() == rhsNumPrograms.getAxis();
  }
  if (auto lhsCmp = dyn_cast<arith::CmpIOp>(lhsDef)) {
    auto rhsCmp = cast<arith::CmpIOp>(rhsDef);
    if (lhsCmp.getPredicate() != rhsCmp.getPredicate())
      return false;
  } else if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
                  arith::DivUIOp>(lhsDef)) {
    return false;
  }

  return llvm::all_of(llvm::zip(lhsDef->getOperands(), rhsDef->getOperands()),
                      [&](auto operands) {
                        return areEquivalentLoopValues(
                            std::get<0>(operands), std::get<1>(operands),
                            assumed, depth + 1, positionalIVs);
                      });
}

bool areEquivalentLoopValues(
    Value lhs, Value rhs, unsigned depth = 0,
    const llvm::DenseMap<Value, Value> *positionalIVs = nullptr) {
  llvm::SmallVector<std::pair<Value, Value>, 4> assumed;
  return areEquivalentLoopValues(lhs, rhs, assumed, depth, positionalIVs);
}

bool haveEquivalentIterationSpaces(scf::ForOp lhs, scf::ForOp rhs) {
  return areEquivalentLoopValues(lhs.getLowerBound(), rhs.getLowerBound()) &&
         areEquivalentLoopValues(lhs.getUpperBound(), rhs.getUpperBound()) &&
         areEquivalentLoopValues(lhs.getStep(), rhs.getStep());
}

std::optional<Value> findIterationCounter(scf::ForOp loop,
                                          uint64_t initialValue,
                                          uint64_t increment) {
  Value result;
  for (Value iterArg : loop.getRegionIterArgs()) {
    std::optional<IterationCounter> counter = matchIterationCounter(iterArg);
    if (!counter || counter->loop != loop ||
        counter->initialValue != initialValue ||
        counter->increment != increment)
      continue;
    if (result)
      return std::nullopt;
    result = iterArg;
  }
  if (!result)
    return std::nullopt;
  return result;
}

std::optional<Value> matchSlotCounter(Value index, unsigned numBarriers) {
  if (!index)
    return std::nullopt;
  index = stripIntegerCasts(resolveWarpSpecializeCapture(index));
  if (auto rem = index.getDefiningOp<arith::RemUIOp>()) {
    if (matchConstantInt(rem.getRhs(), numBarriers))
      return stripIntegerCasts(rem.getLhs());
    return std::nullopt;
  }

  auto sub = index.getDefiningOp<arith::SubIOp>();
  if (!sub)
    return std::nullopt;
  Value counter = stripIntegerCasts(sub.getLhs());
  auto mul = stripIntegerCasts(sub.getRhs()).getDefiningOp<arith::MulIOp>();
  if (!mul)
    return std::nullopt;
  Value quotient = matchBinaryOperandWithConstant(mul, numBarriers);
  if (!quotient)
    return std::nullopt;
  auto div = stripIntegerCasts(quotient).getDefiningOp<arith::DivUIOp>();
  if (!div || !matchConstantInt(div.getRhs(), numBarriers) ||
      stripIntegerCasts(div.getLhs()) != counter)
    return std::nullopt;
  return counter;
}

struct WaitPhasePlan {
  ttng::WaitBarrierOp wait;
  Value iteration;
  Value loopCounter;
  scf::ForOp loop;
  unsigned iterationsPerGeneration = 0;
  unsigned initialIteration = 0;
  unsigned iterationIncrement = 0;
  unsigned iterationOffset = 0;
  std::optional<unsigned> constantPhase;
};

Value getWaitPredicate(const WaitPhasePlan &plan) {
  ttng::WaitBarrierOp wait = plan.wait;
  return wait.getPred();
}

std::optional<WaitPhasePlan> getWaitPhasePlan(ttng::WaitBarrierOp wait) {
  APInt constant;
  if (matchPattern(wait.getPhase(), m_ConstantInt(&constant))) {
    if (constant.getActiveBits() > 1)
      return std::nullopt;
    WaitPhasePlan plan{wait};
    plan.constantPhase = constant.getZExtValue();
    return plan;
  }

  Value phase = stripIntegerCasts(wait.getPhase());
  Value generation;
  if (auto andOp = phase.getDefiningOp<arith::AndIOp>()) {
    generation = matchBinaryOperandWithConstant(andOp, 1);
  } else if (auto remOp = phase.getDefiningOp<arith::RemUIOp>()) {
    if (!matchConstantInt(remOp.getRhs(), 2))
      return std::nullopt;
    generation = remOp.getLhs();
  }
  if (!generation)
    return std::nullopt;

  generation = stripIntegerCasts(generation);
  unsigned iterationsPerGeneration = 1;
  std::optional<CounterExpression> iteration =
      matchCounterExpression(generation);
  if (!iteration) {
    auto divOp = generation.getDefiningOp<arith::DivUIOp>();
    if (!divOp)
      return std::nullopt;
    APInt divisor;
    if (!matchPattern(divOp.getRhs(), m_ConstantInt(&divisor)) ||
        divisor.isZero() || divisor.getActiveBits() > 32)
      return std::nullopt;
    iterationsPerGeneration = divisor.getZExtValue();
    generation = stripIntegerCasts(divOp.getLhs());
    iteration = matchCounterExpression(generation);
  }
  if (!iteration || wait->getBlock() != iteration->counter.loop.getBody())
    return std::nullopt;
  return WaitPhasePlan{wait,
                       iteration->value,
                       iteration->counter.value,
                       iteration->counter.loop,
                       iterationsPerGeneration,
                       static_cast<unsigned>(iteration->counter.initialValue),
                       static_cast<unsigned>(iteration->counter.increment),
                       static_cast<unsigned>(iteration->offset)};
}

void traceBarrierUses(Value value, Value index, BarrierCandidate &candidate) {
  if (!candidate.visitedValues.insert(value).second)
    return;

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(user)) {
      if (index) {
        candidate.hasUnknownUse = true;
        continue;
      }
      candidate.views.push_back(user);
      traceBarrierUses(indexOp.getResult(), indexOp.getIndex(), candidate);
      continue;
    }
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      candidate.views.push_back(user);
      traceBarrierUses(user->getResult(0), index, candidate);
      continue;
    }
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(user)) {
      unsigned operandIdx = use.getOperandNumber();
      candidate.captures.push_back({partitions, value});
      for (Region &region : partitions.getPartitionRegions())
        traceBarrierUses(region.getArgument(operandIdx), index, candidate);
      continue;
    }
    if (auto init = dyn_cast<ttng::InitBarrierOp>(user)) {
      candidate.inits.push_back({init, index});
      continue;
    }
    if (auto inval = dyn_cast<ttng::InvalBarrierOp>(user)) {
      candidate.invalidations.push_back({inval, index});
      continue;
    }
    if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(user)) {
      candidate.arrives.push_back({arrive, index});
      continue;
    }
    if (auto wait = dyn_cast<ttng::WaitBarrierOp>(user)) {
      candidate.waits.push_back({wait, index});
      continue;
    }
    if (auto dealloc = dyn_cast<ttg::LocalDeallocOp>(user)) {
      candidate.deallocations.push_back(dealloc);
      continue;
    }
    candidate.hasUnknownUse = true;
  }
}

std::optional<unsigned> getStaticSlot(Value index, unsigned numBarriers) {
  if (!index)
    return numBarriers == 1 ? std::optional<unsigned>(0) : std::nullopt;
  index = resolveWarpSpecializeCapture(index);
  APInt value;
  if (!matchPattern(index, m_ConstantInt(&value)))
    return std::nullopt;
  int64_t slot = value.getSExtValue();
  if (slot < 0 || slot >= static_cast<int64_t>(numBarriers))
    return std::nullopt;
  return static_cast<unsigned>(slot);
}

template <typename RangeT>
bool coversEverySlot(const RangeT &uses, unsigned numBarriers) {
  llvm::SmallDenseSet<unsigned> slots;
  for (const auto &use : uses) {
    std::optional<unsigned> slot = getStaticSlot(use.index, numBarriers);
    if (!slot || !slots.insert(*slot).second)
      return false;
  }
  return slots.size() == numBarriers;
}

struct DynamicIndexSummary {
  Value first;
  bool allEqual = true;
  bool hasStatic = false;
};

template <typename RangeT>
bool mapUsesToSlots(const RangeT &uses, unsigned numBarriers,
                    SmallVectorImpl<SmallVector<Operation *>> &slotOps,
                    DynamicIndexSummary &dynamicIndex) {
  slotOps.clear();
  slotOps.resize(numBarriers);
  SmallVector<Operation *> dynamicOps;
  for (const auto &use : uses) {
    std::optional<unsigned> slot = getStaticSlot(use.index, numBarriers);
    auto typedOp = use.op;
    Operation *op = typedOp.getOperation();
    if (slot) {
      dynamicIndex.hasStatic = true;
      slotOps[*slot].push_back(op);
      continue;
    }
    // A null index names no slot on a multi-slot allocation; getStaticSlot
    // already accepted the single-slot case above. Unindexed multi-slot uses
    // are also verifier-illegal, so this is defense in depth.
    if (!use.index)
      return false;
    Value resolvedIndex = resolveWarpSpecializeCapture(use.index);
    if (!dynamicIndex.first)
      dynamicIndex.first = resolvedIndex;
    else if (dynamicIndex.first != resolvedIndex)
      dynamicIndex.allEqual = false;
    dynamicOps.push_back(op);
  }
  if (!dynamicOps.empty()) {
    for (auto &ops : slotOps)
      ops.append(dynamicOps);
  }
  return true;
}

// An arrive with no enclosing loop executes at most once per task execution,
// like a priming arrive. (A warp-uniform guard may still skip it, but then it
// is skipped the same way before and after promotion.)
static bool isPrimingArrive(Operation *arrive) {
  Region *task = getWarpSpecializeTaskRegion(arrive);
  if (!task)
    return false;
  for (Operation *parent = arrive->getParentOp();
       parent && parent != task->getParentOp();
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp>(parent))
      return false;
  }
  return true;
}

struct BarrierPlan {
  unsigned numThreads;
  SmallVector<unsigned> activeSlots;
};

bool hasWarpUniformPredicate(Value predicate) {
  if (!predicate)
    return true;
  llvm::SmallDenseSet<Value> visiting;
  return isWarpUniformValue(predicate, visiting);
}

// One gating ancestor (scf.if / scf.while) between an arrive/wait and its
// task: the op kind, which region holds the op, and the gating condition.
// scf.for repeats but never gates, so loops are skipped: trip agreement is
// the iteration-space equivalence proof's job, not this one's.
struct GuardEdge {
  OperationName name;
  unsigned regionNumber;
  Value condition;
};

std::optional<SmallVector<GuardEdge>> getGuardChain(Operation *op) {
  Region *taskRegion = getWarpSpecializeTaskRegion(op);
  if (!taskRegion)
    return std::nullopt;
  SmallVector<GuardEdge> chain;
  for (Operation *child = op, *parent = op->getParentOp();
       parent && parent != taskRegion->getParentOp();
       child = parent, parent = parent->getParentOp()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      chain.push_back({parent->getName(),
                       child->getParentRegion()->getRegionNumber(),
                       ifOp.getCondition()});
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      // The before region always executes, so its condition is not a gate in
      // the strict sense -- but the edge still marks trip counts (once per
      // iteration plus the exit check). Dropping it would admit unbalanced
      // before-vs-bare pairings, so before-region ops keep their edge.
      auto condOp = dyn_cast<scf::ConditionOp>(
          whileOp.getBefore().front().getTerminator());
      if (!condOp)
        return std::nullopt;
      chain.push_back({parent->getName(),
                       child->getParentRegion()->getRegionNumber(),
                       condOp.getCondition()});
    }
  }
  return chain;
}

// The scf.for loops enclosing `op` up to the task boundary, innermost first.
// Used to map guard induction variables positionally across tasks.
SmallVector<scf::ForOp> getEnclosingForChain(Operation *op) {
  SmallVector<scf::ForOp> chain;
  Region *taskRegion = getWarpSpecializeTaskRegion(op);
  Operation *stop = taskRegion ? taskRegion->getParentOp() : nullptr;
  for (Operation *parent = op->getParentOp(); parent && parent != stop;
       parent = parent->getParentOp())
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      chain.push_back(forOp);
  return chain;
}

// Positional guard agreement: same number of gating ancestors, same op kind
// and region at each position, provably equal conditions. Same values imply
// same branches and trip counts, so agreeing ops execute in lockstep across
// tasks. Runs after the per-op uniformity checks, which have already
// rejected thread-dependent conditions and non-allow-listed ancestors.
//
// Guard conditions may reference loop counters, which are compared
// positionally: the enclosing for nests map outermost-first (like
// haveEquivalentWarpSpecializeLoopNests), and counters correspond only at the
// same position. Equal bounds alone would equate counters from different
// loops whose guards fire different numbers of times. Only same-size nests
// map; anything else (including counters captured from outside the task)
// agrees only when identical. Cross-task pairs with different nest sizes are
// rejected downstream by haveMatchingLoopNests regardless.
bool guardsAgree(Operation *lhsOp, ArrayRef<GuardEdge> lhs, Operation *rhsOp,
                 ArrayRef<GuardEdge> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  SmallVector<scf::ForOp> lhsFors = getEnclosingForChain(lhsOp);
  SmallVector<scf::ForOp> rhsFors = getEnclosingForChain(rhsOp);
  llvm::DenseMap<Value, Value> positionalIVs;
  if (lhsFors.size() == rhsFors.size())
    for (auto [lhsLoop, rhsLoop] :
         llvm::zip(llvm::reverse(lhsFors), llvm::reverse(rhsFors)))
      positionalIVs[lhsLoop.getInductionVar()] = rhsLoop.getInductionVar();
  return llvm::all_of(llvm::zip(lhs, rhs), [&](const auto &pair) {
    const auto &[l, r] = pair;
    return l.name == r.name && l.regionNumber == r.regionNumber &&
           areEquivalentLoopValues(l.condition, r.condition, /*depth=*/0,
                                   &positionalIVs);
  });
}

bool hasProvenStaticSlotSchedule(const BarrierCandidate &candidate,
                                 ArrayRef<WaitPhasePlan> waitPhases) {
  SmallVector<scf::ForOp> waitLoops(candidate.numBarriers);
  for (auto [use, plan] : llvm::zip(candidate.waits, waitPhases)) {
    std::optional<unsigned> slot =
        getStaticSlot(use.index, candidate.numBarriers);
    if (!slot || waitLoops[*slot] || !plan.iteration ||
        plan.iterationsPerGeneration != 1 || plan.initialIteration != 0 ||
        plan.iterationIncrement != 1 || plan.iterationOffset != 0 ||
        !isAlwaysTrue(getWaitPredicate(plan)))
      return false;
    waitLoops[*slot] = plan.loop;
  }

  SmallVector<SmallVector<std::pair<Region *, scf::ForOp>>> loopArrives(
      candidate.numBarriers);
  SmallVector<std::pair<ttng::ArriveBarrierOp, unsigned>> initialArrives;
  for (const auto &use : candidate.arrives) {
    std::optional<unsigned> slot =
        getStaticSlot(use.index, candidate.numBarriers);
    if (!slot || !waitLoops[*slot])
      return false;
    Region *task = getWarpSpecializeTaskRegion(use.op);
    if (!task)
      return false;
    scf::ForOp loop = use.op->getParentOfType<scf::ForOp>();
    if (!loop) {
      if (use.op->getParentRegion() != task ||
          llvm::any_of(initialArrives, [&](const auto &entry) {
            return entry.second == *slot &&
                   getWarpSpecializeTaskRegion(entry.first) == task;
          }))
        return false;
      initialArrives.push_back({use.op, *slot});
      continue;
    }
    if (use.op->getBlock() != loop.getBody() ||
        !haveEquivalentIterationSpaces(loop, waitLoops[*slot]) ||
        llvm::any_of(loopArrives[*slot],
                     [task](const auto &entry) { return entry.first == task; }))
      return false;
    loopArrives[*slot].push_back({task, loop});
  }

  for (unsigned slot = 0; slot < candidate.numBarriers; ++slot) {
    if (waitLoops[slot] && loopArrives[slot].empty())
      return false;
  }
  for (auto [arrive, slot] : initialArrives) {
    Region *task = getWarpSpecializeTaskRegion(arrive);
    auto loopIt = llvm::find_if(loopArrives[slot], [&](const auto &entry) {
      return entry.first == task;
    });
    if (loopIt == loopArrives[slot].end() ||
        arrive->getBlock() != loopIt->second->getBlock() ||
        !arrive->isBeforeInBlock(loopIt->second))
      return false;
  }
  return true;
}

bool hasProvenDynamicSchedule(const BarrierCandidate &candidate,
                              ArrayRef<WaitPhasePlan> waitPhases) {
  if (hasProvenStaticSlotSchedule(candidate, waitPhases))
    return true;

  auto firstDynamic = llvm::find_if(
      waitPhases, [](const WaitPhasePlan &plan) { return plan.iteration; });
  if (firstDynamic == waitPhases.end()) {
    return llvm::all_of(waitPhases, [](const WaitPhasePlan &plan) {
      return plan.constantPhase == 0 && isAlwaysTrue(getWaitPredicate(plan));
    });
  }
  // A mix of constant-phase (peeled prologue) and dynamic-phase waits is a
  // conditional schedule: reject it (see the phase-offset design doc).
  if (llvm::any_of(waitPhases,
                   [](const WaitPhasePlan &plan) { return !plan.iteration; }))
    return false;

  if (firstDynamic->initialIteration != 0 ||
      firstDynamic->iterationIncrement != 1 ||
      firstDynamic->iterationOffset != 0)
    return false;

  for (auto [use, plan] : llvm::zip(candidate.waits, waitPhases)) {
    if (plan.iterationsPerGeneration != candidate.numBarriers ||
        plan.initialIteration != firstDynamic->initialIteration ||
        plan.iterationIncrement != firstDynamic->iterationIncrement ||
        plan.iterationOffset != firstDynamic->iterationOffset ||
        plan.loopCounter != firstDynamic->loopCounter ||
        !isAlwaysTrue(getWaitPredicate(plan)) ||
        !haveEquivalentIterationSpaces(plan.loop, firstDynamic->loop))
      return false;
    if (candidate.numBarriers > 1) {
      std::optional<Value> slotCounter =
          matchSlotCounter(use.index, candidate.numBarriers);
      if (!slotCounter || stripIntegerCasts(*slotCounter) != plan.iteration)
        return false;
    }
  }
  for (const auto &use : candidate.arrives) {
    scf::ForOp loop = use.op->getParentOfType<scf::ForOp>();
    if (!loop || use.op->getBlock() != loop.getBody() ||
        !haveEquivalentIterationSpaces(loop, firstDynamic->loop))
      return false;
    std::optional<Value> counter;
    if (candidate.numBarriers == 1)
      counter = findIterationCounter(loop, /*initialValue=*/0,
                                     /*increment=*/1);
    else
      counter = matchSlotCounter(use.index, candidate.numBarriers);
    if (!counter)
      return false;
    std::optional<IterationCounter> iteration = matchIterationCounter(*counter);
    if (!iteration || iteration->loop != loop || iteration->initialValue != 0 ||
        iteration->increment != 1)
      return false;
  }
  return true;
}

std::optional<BarrierPlan> getBarrierPlan(BarrierCandidate &candidate) {
  if (candidate.hasUnknownUse ||
      !coversEverySlot(candidate.inits, candidate.numBarriers) ||
      (!candidate.invalidations.empty() &&
       !coversEverySlot(candidate.invalidations, candidate.numBarriers)))
    return std::nullopt;

  for (const auto &use : candidate.arrives) {
    ttng::ArriveBarrierOp arrive = use.op;
    if (arrive.getPerThread() || arrive.isMulticast() ||
        !isAlwaysTrue(arrive.getPred()) || !isWarpUniform(arrive))
      return std::nullopt;
  }
  SmallVector<WaitPhasePlan> waitPhases;
  for (const auto &use : candidate.waits) {
    ttng::WaitBarrierOp wait = use.op;
    if (!wait.getDeps().empty() || !isWarpUniform(wait) ||
        !hasWarpUniformPredicate(wait.getPred()))
      return std::nullopt;
    std::optional<WaitPhasePlan> phasePlan = getWaitPhasePlan(wait);
    if (!phasePlan)
      return std::nullopt;
    waitPhases.push_back(*phasePlan);
  }
  // Cross-task guard agreement: intra-task uniformity does not imply that
  // guards take the same branches in different tasks. Every arrive and every
  // wait must agree with all the others: a named barrier releases only when
  // every counted thread rendezvouses, and the count includes the waiting
  // tasks' threads, so a skipped wait (or two waits under disagreeing guards)
  // starves the barrier just as surely as a missing arrival. In particular
  // there is no exemption for unguarded arrives: unlike an mbarrier phase,
  // which a stray arrival merely advances, a named-barrier generation
  // accumulates arrivals toward the thread count, so an arrival without its
  // paired wait skews the generation.
  SmallVector<std::pair<Operation *, SmallVector<GuardEdge>>> arriveGuards;
  for (const auto &use : candidate.arrives) {
    std::optional<SmallVector<GuardEdge>> chain = getGuardChain(use.op);
    if (!chain)
      return std::nullopt;
    arriveGuards.push_back({use.op, std::move(*chain)});
  }
  SmallVector<std::pair<Operation *, SmallVector<GuardEdge>>> waitGuards;
  for (const auto &use : candidate.waits) {
    std::optional<SmallVector<GuardEdge>> chain = getGuardChain(use.op);
    if (!chain)
      return std::nullopt;
    waitGuards.push_back({use.op, std::move(*chain)});
  }
  for (unsigned i = 0; i < arriveGuards.size(); ++i)
    for (unsigned j = i + 1; j < arriveGuards.size(); ++j)
      if (!guardsAgree(arriveGuards[i].first, arriveGuards[i].second,
                       arriveGuards[j].first, arriveGuards[j].second))
        return std::nullopt;
  for (unsigned i = 0; i < waitGuards.size(); ++i)
    for (unsigned j = i + 1; j < waitGuards.size(); ++j)
      if (!guardsAgree(waitGuards[i].first, waitGuards[i].second,
                       waitGuards[j].first, waitGuards[j].second))
        return std::nullopt;
  for (const auto &waitChain : waitGuards)
    for (const auto &arriveChain : arriveGuards)
      if (!guardsAgree(waitChain.first, waitChain.second, arriveChain.first,
                       arriveChain.second))
        return std::nullopt;
  if (!hasProvenDynamicSchedule(candidate, waitPhases))
    return std::nullopt;

  SmallVector<SmallVector<Operation *>> arrivesBySlot;
  SmallVector<SmallVector<Operation *>> waitsBySlot;
  DynamicIndexSummary arriveIndex;
  DynamicIndexSummary waitIndex;
  bool hasDynamicPhase = llvm::any_of(
      waitPhases, [](const WaitPhasePlan &plan) { return plan.iteration; });
  if (!mapUsesToSlots(candidate.arrives, candidate.numBarriers, arrivesBySlot,
                      arriveIndex) ||
      !mapUsesToSlots(candidate.waits, candidate.numBarriers, waitsBySlot,
                      waitIndex) ||
      static_cast<bool>(arriveIndex.first) !=
          static_cast<bool>(waitIndex.first))
    return std::nullopt;
  if (!hasDynamicPhase && arriveIndex.first &&
      (!arriveIndex.allEqual || !waitIndex.allEqual || arriveIndex.hasStatic ||
       waitIndex.hasStatic || arriveIndex.first != waitIndex.first))
    return std::nullopt;

  SmallVector<uint32_t> initCounts(candidate.numBarriers);
  for (const auto &use : candidate.inits) {
    auto init = use.op;
    initCounts[*getStaticSlot(use.index, candidate.numBarriers)] =
        init.getCount();
  }

  auto isCTALocal = [](auto op) {
    ttg::MemDescType type = op.getAlloc().getType();
    // Ordinary shared memory can hold a barrier broadcast across CTAs.
    return !isa<ttng::SharedClusterMemorySpaceAttr>(type.getMemorySpace()) &&
           type.getShape().size() == 1 &&
           type.getShape().front() == ttg::lookupNumCTAs(op);
  };
  auto areCTALocal = [&](const auto &uses) {
    return llvm::all_of(uses,
                        [&](const auto &use) { return isCTALocal(use.op); });
  };
  if (!areCTALocal(candidate.inits) || !areCTALocal(candidate.arrives) ||
      !areCTALocal(candidate.waits))
    return std::nullopt;

  unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(
      candidate.alloc->getParentOfType<ModuleOp>());
  unsigned numThreads = 0;
  SmallVector<unsigned> activeSlots;
  // One owner for the whole allocation: checking per slot would let different
  // slots resolve to different `warp_specialize` ops as long as their thread
  // counts coincide.
  ttg::WarpSpecializeOp warpSpecialize;
  auto checkWarpSpecialize = [&](Region *task) {
    ttg::WarpSpecializeOp owner = getWarpSpecializeOwner(task);
    if (!warpSpecialize)
      warpSpecialize = owner;
    return owner == warpSpecialize;
  };
  for (unsigned slot = 0; slot < candidate.numBarriers; ++slot) {
    const auto &arrives = arrivesBySlot[slot];
    const auto &waits = waitsBySlot[slot];
    if (arrives.empty() && waits.empty())
      continue;
    if (arrives.empty() || waits.empty())
      return std::nullopt;

    struct TaskArrives {
      uint32_t count;
      bool hasLooped = false;
      bool hasPriming = false;
    };
    SmallVector<std::pair<Region *, TaskArrives>> arriveTasks;
    for (Operation *arriveOp : arrives) {
      Region *task = getWarpSpecializeTaskRegion(arriveOp);
      if (!task)
        return std::nullopt;
      uint32_t count = cast<ttng::ArriveBarrierOp>(arriveOp).getCount();
      // Same-task arrives share one accounting entry, so a second loop-nested
      // arrive would contribute uncounted arrivals and over-release the named
      // barrier. Allow only the priming shape alongside it: one unlooped
      // arrive, which executes exactly once.
      bool priming = isPrimingArrive(arriveOp);
      auto it = llvm::find_if(
          arriveTasks, [task](auto &entry) { return entry.first == task; });
      if (it == arriveTasks.end()) {
        arriveTasks.push_back({task, {count, !priming, priming}});
        continue;
      }
      if (it->second.count != count)
        return std::nullopt;
      if (priming ? it->second.hasPriming : it->second.hasLooped)
        return std::nullopt;
      if (priming)
        it->second.hasPriming = true;
      else
        it->second.hasLooped = true;
    }
    uint32_t totalArrivalCount = 0;
    llvm::SmallDenseSet<Region *> arriveTaskSet;
    for (const auto &[task, info] : arriveTasks) {
      arriveTaskSet.insert(task);
      totalArrivalCount += info.count;
    }
    if (totalArrivalCount != initCounts[slot])
      return std::nullopt;

    llvm::SmallDenseSet<Region *> waitTaskSet;
    for (Operation *wait : waits) {
      Region *task = getWarpSpecializeTaskRegion(wait);
      // An mbarrier phase can be waited on more than once, but every named
      // barrier wait participates in a new generation. SWP may clone a wait
      // into several sites with different phase/slot expressions; matching
      // each site to an arrive loop does not prove one wait per generation.
      // Count sites per slot, not per allocation: separate static slots can
      // legitimately have a wait in the same task. Dynamic indices were
      // conservatively mapped to every slot by mapUsesToSlots above.
      if (!task || arriveTaskSet.contains(task) ||
          !waitTaskSet.insert(task).second)
        return std::nullopt;
      for (Region *producer : arriveTaskSet) {
        if (llvm::none_of(arrives, [&](Operation *arrive) {
              return getWarpSpecializeTaskRegion(arrive) == producer &&
                     haveMatchingLoopNests(arrive, wait);
            }))
          return std::nullopt;
      }
    }

    unsigned slotThreads = 0;
    for (Region *task : arriveTaskSet) {
      if (!checkWarpSpecialize(task))
        return std::nullopt;
      slotThreads += ttg::lookupNumWarps(task) * threadsPerWarp;
    }
    for (Region *task : waitTaskSet) {
      if (!checkWarpSpecialize(task))
        return std::nullopt;
      slotThreads += ttg::lookupNumWarps(task) * threadsPerWarp;
    }
    if (slotThreads == 0 || (numThreads != 0 && slotThreads != numThreads))
      return std::nullopt;
    numThreads = slotThreads;
    activeSlots.push_back(slot);
  }
  if (activeSlots.empty())
    return std::nullopt;
  return BarrierPlan{numThreads, std::move(activeSlots)};
}

void eraseBarrierStorage(BarrierCandidate &candidate) {
  for (auto use : candidate.inits)
    use.op.erase();
  for (auto use : candidate.invalidations)
    use.op.erase();
  for (ttg::LocalDeallocOp op : candidate.deallocations)
    op.erase();

  // Two passes: this one drops views the promotion just made dead, the one
  // below drops views that only stayed live as capture operands. `views` is
  // walked twice, so null each entry as it goes -- the second pass would
  // otherwise dereference an operation this pass already erased.
  for (Operation *&view : llvm::reverse(candidate.views)) {
    if (view->getResult(0).use_empty()) {
      view->erase();
      view = nullptr;
    }
  }

  // Erase every dead capture of a partitions op in one pass, scanning operand
  // positions rather than searching for each captured value. Two things make
  // the position the thing to key on: erasing an operand shifts the higher
  // ones down, so a stored index goes stale; and a value can be captured into
  // more than one operand, where searching by value keeps finding the first of
  // them and a later dead slot is never reached.
  llvm::MapVector<ttg::WarpSpecializePartitionsOp, llvm::SmallDenseSet<Value>>
      capturedByPartitions;
  for (auto [partitions, captured] : candidate.captures)
    capturedByPartitions[partitions].insert(captured);

  for (auto &[partitions, capturedValues] : capturedByPartitions) {
    llvm::BitVector toRemove(partitions.getNumOperands());
    for (unsigned idx = 0; idx < partitions->getNumOperands(); ++idx) {
      if (!capturedValues.contains(partitions->getOperand(idx)))
        continue;
      bool unused =
          llvm::all_of(partitions.getPartitionRegions(), [idx](Region &region) {
            return region.getArgument(idx).use_empty();
          });
      if (unused)
        toRemove.set(idx);
    }
    if (toRemove.none())
      continue;
    for (Region &region : partitions.getPartitionRegions())
      region.front().eraseArguments(toRemove);
    partitions->eraseOperands(toRemove);
  }

  for (Operation *view : llvm::reverse(candidate.views)) {
    if (view && view->getResult(0).use_empty())
      view->erase();
  }

  if (candidate.alloc.use_empty())
    candidate.alloc.erase();
}

// Loop depth is partition-relative: only scf.for loops between the op and its
// warp-specialize partition count. A loop around the whole warp_specialize op
// is invisible here, so such a candidate can tie with a genuinely unnested one.
unsigned getLoopDepth(const BarrierCandidate &candidate) {
  auto depth = [](Operation *op) {
    unsigned result = 0;
    Region *partition = getWarpSpecializeTaskRegion(op);
    if (!partition)
      return result;
    for (Operation *parent = op->getParentOp();
         parent && parent != partition->getParentOp();
         parent = parent->getParentOp()) {
      if (isa<scf::ForOp>(parent))
        ++result;
    }
    return result;
  };

  unsigned result = 0;
  for (const auto &use : candidate.arrives)
    result = std::max(result, depth(use.op));
  for (const auto &use : candidate.waits)
    result = std::max(result, depth(use.op));
  return result;
}

Value createSelectedNamedBarrierId(OpBuilder &builder, Location loc,
                                   ArrayRef<std::optional<int32_t>> idsBySlot,
                                   Value index) {
  // A single handle needs no selection: the loop below would start at index
  // -1, emit nothing, and return it, but say so directly.
  if (idsBySlot.size() == 1) {
    assert(idsBySlot.front() && "expected an ID for every active slot");
    return createCompilerNamedBarrierId(builder, loc, *idsBySlot.front());
  }
  if (std::optional<unsigned> slot = getStaticSlot(index, idsBySlot.size())) {
    assert(idsBySlot[*slot] && "expected an ID for every active slot");
    return createCompilerNamedBarrierId(builder, loc, *idsBySlot[*slot]);
  }

  SmallVector<Value> handles;
  for (std::optional<int32_t> id : idsBySlot) {
    assert(id && "dynamic indices require every slot to be active");
    handles.push_back(createCompilerNamedBarrierId(builder, loc, *id));
  }
  Value selected = handles.back();
  for (int32_t slot = static_cast<int32_t>(idsBySlot.size()) - 2; slot >= 0;
       --slot) {
    Value slotValue =
        arith::ConstantIntOp::create(builder, loc, index.getType(), slot);
    Value isSlot = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         index, slotValue);
    selected =
        arith::SelectOp::create(builder, loc, isSlot, handles[slot], selected);
  }
  return selected;
}

void promoteBarrier(BarrierCandidate &candidate,
                    ArrayRef<std::optional<int32_t>> idsBySlot,
                    unsigned numThreads) {
  // `createSelectedNamedBarrierId` only asserts that a referenced slot has an
  // ID, which is compiled out in release builds and would dereference an empty
  // optional. Check every use up front: this rewrite mutates as it walks, so
  // bailing partway would leave the barrier half promoted.
  auto slotIdsAvailable = [&](Value index) {
    if (std::optional<unsigned> slot = getStaticSlot(index, idsBySlot.size()))
      return idsBySlot[*slot].has_value();
    return llvm::all_of(
        idsBySlot, [](std::optional<int32_t> id) { return id.has_value(); });
  };
  for (const auto &use : candidate.arrives)
    if (!slotIdsAvailable(use.index))
      return;
  for (const auto &use : candidate.waits)
    if (!slotIdsAvailable(use.index))
      return;

  for (const auto &use : candidate.arrives) {
    ttng::ArriveBarrierOp arrive = use.op;
    OpBuilder builder(arrive);
    Value namedId = createSelectedNamedBarrierId(builder, arrive.getLoc(),
                                                 idsBySlot, use.index);
    Value count =
        arith::ConstantIntOp::create(builder, arrive.getLoc(), numThreads, 32);
    ttng::NamedBarrierArriveOp::create(builder, arrive.getLoc(), namedId,
                                       count);
    arrive.erase();
  }
  for (const auto &use : candidate.waits) {
    ttng::WaitBarrierOp wait = use.op;
    OpBuilder builder(wait);
    Value namedId = createSelectedNamedBarrierId(builder, wait.getLoc(),
                                                 idsBySlot, use.index);
    Value count =
        arith::ConstantIntOp::create(builder, wait.getLoc(), numThreads, 32);
    // Every promoted wait is unconditional: the schedule provers only accept
    // waits whose predicate is provably always-true, and phase plans with an
    // initial skip are never built.
    ttng::NamedBarrierWaitOp::create(builder, wait.getLoc(), namedId, count);
    wait.erase();
  }
  eraseBarrierStorage(candidate);
}

} // namespace

class TritonNvidiaGPUPromoteMBarrierToNamedBarrierPass
    : public impl::TritonNvidiaGPUPromoteMBarrierToNamedBarrierPassBase<
          TritonNvidiaGPUPromoteMBarrierToNamedBarrierPass> {
public:
  using TritonNvidiaGPUPromoteMBarrierToNamedBarrierPassBase::
      TritonNvidiaGPUPromoteMBarrierToNamedBarrierPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // TLX fixup marks modules with source-level TLX ops, explicit memory
    // accesses, or manual warp specialization before any of those ops are
    // lowered. Their barriers have source-defined synchronization semantics;
    // only promote barriers generated by the Triton compiler.
    if (module->hasAttr(triton::kSkipGenericPipelineAttrName))
      return;

    NamedBarrierIdAllocator allocator(module);
    if (failed(tryEnsureWarpSpecializeBarrierIds(module, allocator)))
      return;

    struct RankedCandidate {
      BarrierCandidate candidate;
      unsigned numThreads;
      SmallVector<unsigned> activeSlots;
      unsigned loopDepth;
      unsigned ordinal;
    };

    SmallVector<RankedCandidate, 0> candidates;
    unsigned ordinal = 0;
    module.walk([&](ttg::LocalAllocOp alloc) {
      std::optional<unsigned> numBarriers = getBarrierCount(alloc);
      if (!numBarriers)
        return;
      BarrierCandidate candidate(alloc, *numBarriers);
      traceBarrierUses(alloc.getResult(), Value(), candidate);
      std::optional<BarrierPlan> plan = getBarrierPlan(candidate);
      if (!plan)
        return;
      unsigned loopDepth = getLoopDepth(candidate);
      candidates.push_back({std::move(candidate), plan->numThreads,
                            std::move(plan->activeSlots), loopDepth,
                            ordinal++});
    });

    llvm::stable_sort(
        candidates, [](const RankedCandidate &lhs, const RankedCandidate &rhs) {
          if (lhs.loopDepth != rhs.loopDepth)
            return lhs.loopDepth > rhs.loopDepth;
          if (lhs.activeSlots.size() != rhs.activeSlots.size())
            return lhs.activeSlots.size() < rhs.activeSlots.size();
          return lhs.ordinal < rhs.ordinal;
        });

    // Candidates are validated before any promotion runs and promoted later in
    // ranked order. This is safe because each candidate's recorded IR hangs off
    // its own alloc, so one candidate's cleanup cannot erase IR another
    // recorded; shared partition ops only lose capture operands, which cleanup
    // resolves by value rather than by stored index.
    for (RankedCandidate &ranked : candidates) {
      std::optional<SmallVector<int32_t>> ids =
          allocator.allocate(ranked.activeSlots.size());
      if (!ids)
        continue;
      SmallVector<std::optional<int32_t>> idsBySlot(
          ranked.candidate.numBarriers);
      for (auto [slot, id] : llvm::zip(ranked.activeSlots, *ids))
        idsBySlot[slot] = id;
      promoteBarrier(ranked.candidate, idsBySlot, ranked.numThreads);
    }
  }
};

#define GEN_PASS_DEF_TRITONNVIDIAGPUVERIFYNAMEDBARRIERCOUNTSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

class TritonNvidiaGPUVerifyNamedBarrierCountsPass
    : public impl::TritonNvidiaGPUVerifyNamedBarrierCountsPassBase<
          TritonNvidiaGPUVerifyNamedBarrierCountsPass> {
public:
  using TritonNvidiaGPUVerifyNamedBarrierCountsPassBase::
      TritonNvidiaGPUVerifyNamedBarrierCountsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);

    WalkResult result = module.walk([&](ttg::WarpSpecializeOp wsOp) {
      // Compiler ids are module-unique, so grouping within one
      // warp_specialize op recovers each promoted barrier exactly.
      DenseMap<int32_t, SmallVector<Operation *>> groups;
      wsOp.walk([&](Operation *op) {
        Value bar;
        if (auto arrive = dyn_cast<ttng::NamedBarrierArriveOp>(op))
          bar = arrive.getBar();
        else if (auto wait = dyn_cast<ttng::NamedBarrierWaitOp>(op))
          bar = wait.getBar();
        else
          return;
        auto idOp = bar.getDefiningOp<ttng::CompilerNamedBarrierIdOp>();
        if (!idOp)
          return;
        APInt id;
        if (!matchPattern(idOp.getValue(), m_ConstantInt(&id)))
          return;
        groups[static_cast<int32_t>(id.getSExtValue())].push_back(op);
      });

      for (auto &[id, ops] : groups) {
        (void)id;
        llvm::SmallDenseSet<Region *, 4> tasks;
        bool allInTasks = true;
        for (Operation *op : ops) {
          Region *task = getWarpSpecializeTaskRegion(op);
          if (!task) {
            allInTasks = false;
            break;
          }
          tasks.insert(task);
        }
        if (!allInTasks)
          continue;
        unsigned expected = 0;
        for (Region *task : tasks)
          expected += ttg::lookupNumWarps(task) * threadsPerWarp;
        for (Operation *op : ops) {
          Value count =
              isa<ttng::NamedBarrierArriveOp>(op)
                  ? cast<ttng::NamedBarrierArriveOp>(op).getNumThreads()
                  : cast<ttng::NamedBarrierWaitOp>(op).getNumThreads();
          APInt baked;
          if (!matchPattern(count, m_ConstantInt(&baked)))
            continue;
          if (baked.getZExtValue() == expected)
            continue;
          op->emitOpError()
              << "named barrier participant count mismatch: baked "
              << baked.getZExtValue() << " threads but participating tasks "
              << "now require " << expected;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace mlir::triton::nvidia_gpu
