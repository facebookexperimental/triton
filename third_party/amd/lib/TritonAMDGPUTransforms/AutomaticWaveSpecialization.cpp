#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Uniformity.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-automatic-wave-specialization"

namespace mlir {
#define GEN_PASS_DEF_TRITONAMDGPUAUTOMATICWAVESPECIALIZATION
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {
constexpr StringLiteral specializedAttr = "tritonamdgpu.wave_specialized";

// AMD kernels use a one-dimensional workgroup. Clearing low lane bits and
// dividing/shifting by the configured wave size removes lane dependence.
bool isWaveId(Operation *op, unsigned waveSize) {
  if (!isa<LLVM::UDivOp, LLVM::LShrOp>(op))
    return false;
  APInt amount;
  if (!matchPattern(op->getOperand(1), m_ConstantInt(&amount)))
    return false;
  unsigned shift = llvm::Log2_32(waveSize);
  if ((isa<LLVM::UDivOp>(op) && amount != waveSize) ||
      (isa<LLVM::LShrOp>(op) && amount != shift))
    return false;
  Value thread = op->getOperand(0);
  if (auto mask = thread.getDefiningOp<LLVM::AndOp>()) {
    APInt bits;
    if (!matchPattern(mask.getRhs(), m_ConstantInt(&bits)))
      return false;
    thread = mask.getLhs();
  }
  return isa_and_nonnull<ROCDL::ThreadIdXOp>(thread.getDefiningOp());
}

struct Plan {
  Value predicate;
  SmallVector<Value> equivalents;
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
  plan.equivalents.push_back(predicate);
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
  // One extra body, two constants, entry/leaf/exit branches, common return,
  // and four operations to normalize the dispatch predicate.
  if (bodySize - plan.prefix.size() + 11 > maxExtraOps) {
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
    IRMapping mapping;
    mapping.map(plan.predicate, constant);
    // Region::cloneInto creates all blocks and remaps forward edges, backedges,
    // arguments and SSA references, independent of source block layout.
    original.cloneInto(&body, exit->getIterator(), mapping);
    // cloneInto creates new results for equivalent definitions retained in
    // the body. Substitute those results after cloning has installed its map.
    for (Value predicate : plan.equivalents)
      if (Value cloned = mapping.lookupOrNull(predicate);
          cloned && cloned != constant)
        cloned.replaceAllUsesWith(constant);
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

  void runOnOperation() override {
    if (waveSize != 32 && waveSize != 64) {
      getOperation().emitError("wave-size must be 32 or 64");
      return signalPassFailure();
    }
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
      unsigned bodySize = 0;
      bool supported = true;
      for (Block &block : function.getBody())
        for (Operation &op : block) {
          ++bodySize;
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
      triton::UniformityAnalysis::Options options;
      options.isUniformArgument = [](BlockArgument) { return true; };
      options.getRule =
          [&](Operation *op,
              unsigned) -> std::optional<triton::UniformityAnalysis::Rule> {
        if (isWaveId(op, waveSize))
          return triton::UniformityAnalysis::Rule::Uniform;
        return std::nullopt;
      };
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
          if (!equivalent(group.predicate, value))
            continue;
          group.score += plan->score;
          group.equivalents.push_back(value);
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
