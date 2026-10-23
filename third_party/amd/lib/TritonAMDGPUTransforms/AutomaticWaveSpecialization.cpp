#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "triton/Analysis/IntegerRange.h"
#include "triton/Analysis/Uniformity.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "tritonamdgpu-automatic-wave-specialization"

namespace mlir {
#define GEN_PASS_DEF_TRITONAMDGPUAUTOMATICWAVESPECIALIZATION
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {
constexpr StringLiteral specializedAttr = "tritonamdgpu.wave_specialized";

// Triton's kernel ABI uses a linear X workitem coordinate. Explicit warp
// specialization can allocate more waves than the default partition, so use
// the physical total when present. Do not infer bounds from absent metadata.
void attachIDBounds(LLVM::LLVMFuncOp function) {
  auto module = function->getParentOfType<ModuleOp>();
  if (!module)
    return;
  auto warps = module->getAttrOfType<IntegerAttr>("ttg.total-num-warps");
  if (!warps)
    warps = module->getAttrOfType<IntegerAttr>(triton::gpu::AttrNumWarpsName);
  auto size =
      module->getAttrOfType<IntegerAttr>(triton::gpu::AttrNumThreadsPerWarp);
  if (!warps || warps.getValue().getActiveBits() > 31 || warps.getInt() <= 0)
    return;
  uint64_t numWarps = warps.getInt();
  auto attach = [&](auto id, uint64_t bound) {
    if (!id.getRangeAttr())
      id.setRangeAttr(
          LLVM::ConstantRangeAttr::get(function.getContext(), 32, 0, bound));
  };
  function.walk([&](ROCDL::WaveId id) { attach(id, numWarps); });
  if (!size || size.getValue().getActiveBits() > 31 || size.getInt() <= 0 ||
      !size.getValue().isPowerOf2() || numWarps * size.getInt() > UINT32_MAX)
    return;
  function.walk(
      [&](ROCDL::ThreadIdXOp id) { attach(id, numWarps * size.getInt()); });
}

std::optional<APInt> integerConstant(Value value) {
  IntegerAttr attr;
  if (auto type = dyn_cast<IntegerType>(value.getType());
      type && matchPattern(value, m_Constant(&attr)))
    return attr.getValue().sextOrTrunc(type.getWidth());
  return std::nullopt;
}

// Normalize a scalar bit test to (input & mask) == 0, with optional negation.
// Range analysis proves when a lookup in a low-ones bit mask is the same test
// on a bit of the index. The algebra lives here, not in lowering patterns or a
// new dataflow lattice.
bool normalizeBitTest(LLVM::ICmpOp cmp, DataFlowSolver &solver,
                      IRRewriter &rewriter) {
  if (!cmp.getType().isInteger(1) ||
      (cmp.getPredicate() != LLVM::ICmpPredicate::eq &&
       cmp.getPredicate() != LLVM::ICmpPredicate::ne))
    return false;
  Value tested = cmp.getLhs();
  auto rhs = integerConstant(cmp.getRhs());
  if (!rhs) {
    rhs = integerConstant(tested);
    tested = cmp.getRhs();
  }
  if (!rhs)
    return false;
  auto bitAnd = tested.getDefiningOp<LLVM::AndOp>();
  if (!bitAnd)
    return false;
  Value input = bitAnd.getLhs();
  auto mask = integerConstant(bitAnd.getRhs());
  if (!mask) {
    mask = integerConstant(input);
    input = bitAnd.getRhs();
  }
  if (!mask || !mask->isPowerOf2() || (!rhs->isZero() && *rhs != *mask))
    return false;
  bool negate =
      (cmp.getPredicate() == LLVM::ICmpPredicate::ne) ^ !rhs->isZero();
  bool changed =
      !rhs->isZero() || cmp.getPredicate() == LLVM::ICmpPredicate::ne;
  if (auto shift = input.getDefiningOp<LLVM::LShrOp>();
      shift && mask->isOne()) {
    auto table = integerConstant(shift.getLhs());
    unsigned ones = table ? table->countr_one() : 0;
    auto *range =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(shift.getRhs());
    if (table && table->isMask() && llvm::isPowerOf2_32(ones) && range &&
        !range->getValue().isUninitialized()) {
      const APInt &max = range->getValue().getValue().umax();
      // Both bounds matter: out-of-width LLVM shifts produce poison, and
      // index 4 is already a counterexample for the table 0b0011.
      if (max.ult(table->getBitWidth()) && max.ult(2 * ones)) {
        input = shift.getRhs();
        mask = APInt(mask->getBitWidth(), ones);
        negate = !negate;
        changed = true;
      }
    }
  }
  if (auto xored = input.getDefiningOp<LLVM::XOrOp>()) {
    Value base = xored.getLhs();
    auto bits = integerConstant(xored.getRhs());
    if (!bits) {
      bits = integerConstant(base);
      base = xored.getRhs();
    }
    if (bits) {
      negate ^= !(*bits & *mask).isZero();
      input = base;
      changed = true;
    }
  }
  if (!changed)
    return false;
  rewriter.setInsertionPoint(cmp);
  auto loc = cmp.getLoc();
  Value maskValue =
      LLVM::ConstantOp::create(rewriter, loc, input.getType(),
                               rewriter.getIntegerAttr(input.getType(), *mask));
  Value zero = LLVM::ConstantOp::create(rewriter, loc, input.getType(),
                                        rewriter.getZeroAttr(input.getType()));
  Value bit = LLVM::AndOp::create(rewriter, loc, input, maskValue);
  Value result =
      LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq, bit, zero);
  if (negate) {
    Value one = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                         rewriter.getBoolAttr(true));
    result = LLVM::XOrOp::create(rewriter, loc, result, one);
  }
  // Keep analysis anchors alive until all range queries have finished.
  rewriter.replaceAllUsesWith(cmp.getResult(), result);
  return true;
}

LogicalResult canonicalizePredicates(LLVM::LLVMFuncOp function) {
  attachIDBounds(function);
  IRRewriter rewriter(function.getContext());
  {
    DataFlowSolver solver{DataFlowConfig().setInterprocedural(false)};
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(function)))
      return failure();
    SmallVector<LLVM::ICmpOp> comparisons;
    function.walk([&](LLVM::ICmpOp cmp) { comparisons.push_back(cmp); });
    for (auto cmp : comparisons) {
      if (cmp.getResult().use_empty() || !cmp.getType().isInteger(1))
        continue;
      auto *range = solver.lookupState<dataflow::IntegerValueRangeLattice>(
          cmp.getResult());
      if (range && !range->getValue().isUninitialized()) {
        if (auto value = range->getValue().getValue().getConstantValue()) {
          rewriter.setInsertionPoint(cmp);
          Value constant = LLVM::ConstantOp::create(
              rewriter, cmp.getLoc(), cmp.getType(),
              rewriter.getIntegerAttr(cmp.getType(), *value));
          rewriter.replaceAllUsesWith(cmp.getResult(), constant);
          continue;
        }
      }
      normalizeBitTest(cmp, solver, rewriter);
    }
  }
  DominanceInfo dominance(function);
  eliminateCommonSubExpressions(rewriter, dominance, function);
  return success();
}

struct Plan {
  Value predicate;
  // The bool records whether this equivalent definition is complemented.
  SmallVector<std::pair<Value, bool>> equivalents;
  SmallVector<Operation *> prefix;
  unsigned score = 0;
};

// Restrict noncontiguous extraction to deterministic per-thread computations.
// In particular, do not move convergent broadcasts or inline assembly across
// retained work, even if they carry no memory effects.
bool movable(Operation *op) {
  if (isa<ROCDL::ThreadIdXOp, ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp,
          ROCDL::BlockIdXOp, ROCDL::BlockIdYOp, ROCDL::BlockIdZOp,
          ROCDL::WaveId>(op))
    return true;
  return isa_and_nonnull<LLVM::LLVMDialect, arith::ArithDialect,
                         math::MathDialect, index::IndexDialect>(
             op->getDialect()) &&
         !isa<LLVM::CallOp, LLVM::CallIntrinsicOp, LLVM::InlineAsmOp>(op) &&
         isPure(op);
}

std::optional<Plan> planFor(Value predicate, Block &entry, unsigned bodySize,
                            unsigned maxExtraOps, DominanceInfo &dominance) {
  llvm::SmallPtrSet<Operation *, 32> slice;
  SmallVector<Value> worklist{predicate};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() != &entry)
        return std::nullopt;
      continue;
    }
    Operation *def = value.getDefiningOp();
    if (def->getNumRegions() || def->hasTrait<OpTrait::IsTerminator>())
      return std::nullopt;
    // Every non-entry dependency must be safe to execute before any original
    // control flow. Non-entry block arguments reject loop-carried recurrences.
    if (def->getBlock() != &entry && !movable(def))
      return std::nullopt;
    if (slice.insert(def).second)
      llvm::append_range(worklist, def->getOperands());
  }

  Plan plan;
  plan.predicate = predicate;
  plan.equivalents.emplace_back(predicate, false);
  bool crossedRetained = false;
  for (Operation &op : entry) {
    if (!slice.contains(&op)) {
      crossedRetained = true;
      continue;
    }
    if (crossedRetained && !movable(&op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot extract dependency across retained work: " << op
                 << "\n");
      return std::nullopt;
    }
    plan.prefix.push_back(&op);
  }
  for (Block &block : *entry.getParent()) {
    if (&block == &entry)
      continue;
    for (Operation &op : block)
      if (slice.contains(&op))
        plan.prefix.push_back(&op);
  }
  // With no non-entry block arguments, the slice lies on the definition's
  // dominance chain. Preserve order within each block, and order those blocks
  // by dominance rather than their arbitrary textual layout.
  llvm::stable_sort(plan.prefix, [&](Operation *a, Operation *b) {
    if (a->getBlock() == b->getBlock())
      return a->isBeforeInBlock(b);
    return dominance.dominates(a->getBlock(), b->getBlock());
  });
  // One extra body, up to four constants (including complements),
  // entry/leaf/exit branches, common return, and four dispatch operations.
  if (bodySize - plan.prefix.size() + 13 > maxExtraOps) {
    LLVM_DEBUG(llvm::dbgs()
               << "Code-growth limit rejects " << predicate << "\n");
    return std::nullopt;
  }
  for (OpOperand &use : predicate.getUses()) {
    Operation *user = use.getOwner();
    if (slice.contains(user))
      continue;
    // Prefer repeated uses beyond entry, typically masks in original loops.
    unsigned weight = user->getBlock() == &entry ? 1 : 4;
    if (isa<LLVM::SelectOp, LLVM::CondBrOp>(user) &&
        use.getOperandNumber() == 0)
      weight *= 2;
    if (auto select = dyn_cast<LLVM::SelectOp>(user))
      if (auto type = dyn_cast<VectorType>(select.getType()))
        weight *= type.getNumElements();
    plan.score += weight;
  }
  return plan;
}

// Repeated lowered fused loads build the same selector in different blocks.
// Group their invariant expression trees without moving all their definitions
// to the prefix. Only the representative slice is extracted; every equivalent
// result is substituted in each clone and ordinary DCE removes the duplicates.
bool equivalent(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto l = dyn_cast<OpResult>(lhs);
  auto r = dyn_cast<OpResult>(rhs);
  if (!l || !r || l.getResultNumber() != r.getResultNumber() ||
      !movable(l.getOwner()) || !movable(r.getOwner()))
    return false;
  return OperationEquivalence::isEquivalentTo(
      l.getOwner(), r.getOwner(),
      [](Value a, Value b) { return success(equivalent(a, b)); }, nullptr,
      OperationEquivalence::IgnoreLocations);
}

std::pair<Value, bool> stripNegation(Value value) {
  bool negate = false;
  while (auto op = value.getDefiningOp<LLVM::XOrOp>()) {
    if (auto rhs = integerConstant(op.getRhs()); rhs && rhs->isOne())
      value = op.getLhs();
    else if (auto lhs = integerConstant(op.getLhs()); lhs && lhs->isOne())
      value = op.getRhs();
    else
      break;
    negate = !negate;
  }
  return {value, negate};
}

void specialize(LLVM::LLVMFuncOp function, const Plan &plan) {
  Region &body = function.getBody();
  Block &entry = body.front();
  // The new entry keeps the function arguments. Move only the planned slice
  // back into it, in source order; all other operations keep their order.
  entry.splitBlock(entry.begin());
  for (Operation *op : plan.prefix)
    op->moveBefore(&entry, entry.end());

  OpBuilder builder(function.getContext());
  Location loc = function.getLoc();
  builder.setInsertionPointToEnd(&entry);
  // Speculating a pure predicate can produce poison on a path that originally
  // did not use it. Freeze before branching, and broadcast that choice so all
  // lanes still enter the same version. Defined uniform predicates are
  // unchanged.
  Value dispatch =
      LLVM::ZExtOp::create(builder, loc, builder.getI32Type(), plan.predicate);
  dispatch = LLVM::FreezeOp::create(builder, loc, dispatch);
  dispatch = ROCDL::ReadfirstlaneOp::create(builder, loc, builder.getI32Type(),
                                            dispatch);
  dispatch = LLVM::TruncOp::create(builder, loc, builder.getI1Type(), dispatch);

  Region original;
  original.getBlocks().splice(original.end(), body.getBlocks(),
                              std::next(body.begin()), body.end());
  Block *exit = new Block;
  body.push_back(exit);
  builder.setInsertionPointToEnd(exit);
  LLVM::ReturnOp::create(builder, loc, ValueRange{});

  Block *leaves[2];
  for (unsigned version = 0; version != 2; ++version) {
    Block *leaf = new Block;
    body.getBlocks().insert(exit->getIterator(), leaf);
    leaves[version] = leaf;
    builder.setInsertionPointToEnd(leaf);
    Value constant = LLVM::ConstantOp::create(
        builder, loc, builder.getI1Type(), builder.getBoolAttr(version == 0));
    Value complement;
    if (llvm::any_of(plan.equivalents, [](auto entry) { return entry.second; }))
      complement = LLVM::ConstantOp::create(builder, loc, builder.getI1Type(),
                                            builder.getBoolAttr(version != 0));
    IRMapping mapping;
    for (auto [predicate, negate] : plan.equivalents)
      mapping.map(predicate, negate ? complement : constant);
    // Region::cloneInto creates all blocks and remaps forward edges, backedges,
    // arguments and SSA references, independent of source block layout.
    original.cloneInto(&body, exit->getIterator(), mapping);
    // cloneInto creates new results for equivalent definitions retained in
    // the body. Substitute those results after cloning has installed its map.
    for (auto [predicate, negate] : plan.equivalents)
      if (Value cloned = mapping.lookupOrNull(predicate);
          cloned && cloned != (negate ? complement : constant))
        cloned.replaceAllUsesWith(negate ? complement : constant);
    LLVM::BrOp::create(builder, loc, ValueRange{},
                       mapping.lookup(&original.front()));
    for (Block &block : original) {
      Block *cloned = mapping.lookup(&block);
      if (auto ret = dyn_cast<LLVM::ReturnOp>(cloned->getTerminator())) {
        builder.setInsertionPoint(ret);
        LLVM::BrOp::create(builder, ret.getLoc(), ValueRange{}, exit);
        ret.erase();
      }
    }
  }
  builder.setInsertionPointToEnd(&entry);
  LLVM::CondBrOp::create(builder, loc, dispatch, leaves[0], leaves[1]);
  function->setAttr(specializedAttr, builder.getUnitAttr());
}

struct AutomaticWaveSpecialization
    : impl::TritonAMDGPUAutomaticWaveSpecializationBase<
          AutomaticWaveSpecialization> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    triton::registerIntegerRangeExternalModels(registry);
  }

  void runOnOperation() override {
    SmallVector<LLVM::LLVMFuncOp> functions;
    getOperation().walk(
        [&](LLVM::LLVMFuncOp function) { functions.push_back(function); });
    for (auto function : functions) {
      // Triton's AMD lowering marks kernel entries external and device helpers
      // internal. Only kernel arguments have the uniform ABI assumed here.
      if (function.isExternal() ||
          function.getLinkage() != LLVM::Linkage::External ||
          !isa<LLVM::LLVMVoidType>(
              function.getFunctionType().getReturnType()) ||
          function->hasAttr(specializedAttr))
        continue;
      bool supported = true;
      for (Block &block : function.getBody())
        for (Operation &op : block) {
          supported &= !op.getNumRegions();
          if (op.hasTrait<OpTrait::IsTerminator>())
            supported &= isa<LLVM::BrOp, LLVM::CondBrOp, LLVM::SwitchOp,
                             LLVM::ReturnOp, LLVM::UnreachableOp>(op);
        }
      if (!supported) {
        LLVM_DEBUG(llvm::dbgs()
                   << function.getSymName() << ": unsupported lowered CFG\n");
        continue;
      }
      if (failed(canonicalizePredicates(function))) {
        function.emitError("integer range analysis failed");
        return signalPassFailure();
      }
      unsigned bodySize = 0;
      for (Block &block : function.getBody())
        bodySize += block.getOperations().size();
      triton::UniformityAnalysis::Options options;
      options.isUniformArgument = [](BlockArgument) { return true; };
      options.linearThreadIdX = true;
      triton::UniformityAnalysis analysis(function, options);
      if (failed(analysis.getStatus())) {
        function.emitError("wave uniformity analysis failed");
        return signalPassFailure();
      }
      Block &entry = function.getBody().front();
      DominanceInfo dominance(function);
      SmallVector<Plan> groups;
      auto consider = [&](Value value) {
        Attribute attr;
        if (!value.getType().isInteger(1) ||
            matchPattern(value, m_Constant(&attr)))
          return;
        if (!analysis.isUniform(value)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Not proven wave-uniform: " << value << "\n");
          return;
        }
        auto plan = planFor(value, entry, bodySize, maxExtraOps, dominance);
        if (!plan)
          return;
        for (Plan &group : groups) {
          auto [base, negate] = stripNegation(value);
          auto [groupBase, groupNegate] = stripNegation(group.predicate);
          if (!equivalent(groupBase, base))
            continue;
          group.score += plan->score;
          group.equivalents.emplace_back(value, negate != groupNegate);
          return;
        }
        groups.push_back(std::move(*plan));
      };
      for (BlockArgument arg : entry.getArguments())
        consider(arg);
      for (Block &block : function.getBody())
        for (Operation &op : block)
          for (Value result : op.getResults())
            consider(result);
      Plan *best = nullptr;
      for (Plan &group : groups)
        if (group.score >= 2 && (!best || group.score > best->score ||
                                 (group.score == best->score &&
                                  group.prefix.size() < best->prefix.size())))
          best = &group;
      if (!best) {
        LLVM_DEBUG(
            llvm::dbgs()
            << function.getSymName()
            << ": no useful invariant uniform predicate within extraction "
               "and code-growth limits\n");
        continue;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << function.getSymName() << ": specializing "
                 << best->predicate << ", prefix " << best->prefix.size()
                 << ", extra body " << bodySize - best->prefix.size()
                 << " ops, " << best->equivalents.size()
                 << " equivalent predicates\n");
      specialize(function, *best);
    }
  }
};
} // namespace
} // namespace mlir
