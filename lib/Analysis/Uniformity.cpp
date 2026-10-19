#include "triton/Analysis/Uniformity.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::dataflow;

namespace {
using Rule = UniformityAnalysis::Rule;

unsigned getSubgroupSize(Operation *function) {
  auto module = function->getParentOfType<ModuleOp>();
  if (!module)
    return 0;
  auto size =
      module->getAttrOfType<IntegerAttr>(triton::gpu::AttrNumThreadsPerWarp);
  if (!size || !size.getValue().isPowerOf2() ||
      size.getValue().getActiveBits() > 32)
    return 0;
  return size.getValue().getZExtValue();
}

Rule defaultRule(Operation *op, const UniformityAnalysis::Options &options) {
  if (op->getNumRegions())
    return Rule::Unknown;
  // Distributed tensor values require a different, layout-aware analysis.
  if (llvm::any_of(op->getOperandTypes(),
                   [](Type t) { return isa<TensorType>(t); }) ||
      llvm::any_of(op->getResultTypes(),
                   [](Type t) { return isa<TensorType>(t); }))
    return Rule::Unknown;
  // ConstantLike also includes undef/poison and address-of operations. It is
  // not a uniformity contract (e.g. thread-local addresses can differ).
  if (isa<arith::ConstantOp, index::ConstantOp, LLVM::ConstantOp, LLVM::ZeroOp>(
          op))
    return Rule::Uniform;
  if (isa<mlir::gpu::LaneIdOp>(op))
    return Rule::ThreadId;
  if (options.linearThreadIdX) {
    if (isa<ROCDL::ThreadIdXOp>(op))
      return Rule::ThreadId;
    if (auto thread = dyn_cast<mlir::gpu::ThreadIdOp>(op))
      if (thread.getDimension() == mlir::gpu::Dimension::x)
        return Rule::ThreadId;
  }
  if (isa<mlir::gpu::BlockIdOp, mlir::gpu::BlockDimOp, mlir::gpu::GridDimOp,
          mlir::gpu::SubgroupIdOp, mlir::gpu::SubgroupSizeOp, ROCDL::BlockIdXOp,
          ROCDL::BlockIdYOp, ROCDL::BlockIdZOp, ROCDL::WaveId,
          ROCDL::ReadfirstlaneOp, ROCDL::ReadlaneOp>(op))
    return Rule::Uniform;
  // These dialects' effect-free, region-free operations are deterministic
  // per-thread computations. Do not generalize this to arbitrary Pure ops:
  // thread IDs, undef, and communication can also be Pure.
  if (isa_and_nonnull<arith::ArithDialect, math::MathDialect,
                      index::IndexDialect, vector::VectorDialect>(
          op->getDialect()) &&
      isMemoryEffectFree(op))
    return Rule::FromOperands;
  if (isa<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::UDivOp, LLVM::SDivOp,
          LLVM::URemOp, LLVM::SRemOp, LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp,
          LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp, LLVM::ICmpOp, LLVM::FCmpOp,
          LLVM::SelectOp, LLVM::FAddOp, LLVM::FSubOp, LLVM::FMulOp,
          LLVM::FDivOp, LLVM::FRemOp, LLVM::FNegOp, LLVM::FPExtOp,
          LLVM::FPTruncOp, LLVM::SExtOp, LLVM::ZExtOp, LLVM::TruncOp,
          LLVM::SIToFPOp, LLVM::UIToFPOp, LLVM::FPToSIOp, LLVM::FPToUIOp,
          LLVM::BitcastOp, LLVM::PtrToIntOp, LLVM::IntToPtrOp, LLVM::GEPOp,
          LLVM::ExtractElementOp, LLVM::InsertElementOp, LLVM::ShuffleVectorOp,
          LLVM::ExtractValueOp, LLVM::InsertValueOp>(op))
    return Rule::FromOperands;
  // Includes loads, atomics, calls, inline assembly, poison/undef/freeze,
  // address-space casts, and unmodeled intrinsics.
  return Rule::Unknown;
}

Attribute constant(Value value) {
  Attribute attr;
  if (value)
    matchPattern(value, m_Constant(&attr));
  return attr;
}

bool sameValue(Value a, Value b) {
  if (a == b)
    return true;
  Attribute attr = constant(a);
  return attr && a.getType() == b.getType() && attr == constant(b);
}

struct Uniformity {
  enum Kind { Bottom, Uniform, ThreadId, Unknown } kind = Bottom;
  // For ThreadId<N>, all bits above the low log2(N) bits agree across lanes.
  // N is a power of two, not a count of distinct values. Uniform is N == 1.
  unsigned threadIdSize = 1;
  Uniformity() = default;
  Uniformity(Kind kind) : kind(kind) {}
  static Uniformity threadId(unsigned size) {
    Uniformity result(size == 1 ? Uniform : ThreadId);
    result.threadIdSize = size;
    return result;
  }
  static Uniformity join(Uniformity a, Uniformity b) {
    if (a.kind == Bottom)
      return b;
    if (b.kind == Bottom)
      return a;
    if (a.kind == Unknown || b.kind == Unknown)
      return Unknown;
    return threadId(std::max(a.threadIdSize, b.threadIdSize));
  }
  // Ordinary computations and control decisions cannot use the low-bit bound.
  Uniformity withoutThreadId() const {
    return kind == ThreadId ? Uniformity(Unknown) : *this;
  }
  bool operator==(const Uniformity &other) const {
    return kind == other.kind && threadIdSize == other.threadIdSize;
  }
  void print(raw_ostream &os) const {
    if (kind == ThreadId) {
      os << "ThreadId<" << threadIdSize << ">";
      return;
    }
    os << (kind == Bottom ? "bottom" : kind == Uniform ? "uniform" : "unknown");
  }
};
struct UniformityLattice : Lattice<Uniformity> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UniformityLattice)
  using Lattice::Lattice;
};

class UniformityDataFlow
    : public SparseForwardDataFlowAnalysis<UniformityLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UniformityDataFlow)
  using Base = SparseForwardDataFlowAnalysis<UniformityLattice>;
  UniformityDataFlow(DataFlowSolver &solver, Operation *function,
                     UniformityAnalysis::Options options)
      : Base(solver), solver(solver), function(function),
        options(std::move(options)), subgroupSize(getSubgroupSize(function)) {}

  LogicalResult initialize(Operation *top) override {
    if (failed(Base::initialize(top)))
      return failure();
    // SparseAnalysis initializes blocks through its own private visitBlock.
    // Also install our CFG join dependencies before the solver starts.
    top->walk([&](Operation *op) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          if (!block.isEntryBlock())
            visitCFGBlock(&block);
    });
    return success();
  }

  LogicalResult visit(ProgramPoint *point) override {
    if (point->isBlockStart() && !point->getBlock()->isEntryBlock()) {
      visitCFGBlock(point->getBlock());
      return success();
    }
    return Base::visit(point);
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const UniformityLattice *> operands,
                               ArrayRef<UniformityLattice *> results) override {
    for (auto [index, result] : llvm::enumerate(results)) {
      Rule rule = defaultRule(op, options);
      if (options.getRule)
        if (auto model = options.getRule(op, index))
          rule = *model;
      Uniformity state = Uniformity::Unknown;
      if (rule == Rule::Uniform) {
        state = Uniformity::Uniform;
      } else if (rule == Rule::ThreadId) {
        if (subgroupSize)
          state = Uniformity::threadId(subgroupSize);
      } else if (rule == Rule::FromOperands) {
        if (auto reduced = reduceThreadId(op)) {
          update(result, *reduced);
          continue;
        }
        state = Uniformity::Uniform;
        ValueRange inputs = op->getOperands();
        if (isa<arith::SelectOp, LLVM::SelectOp>(op) &&
            sameValue(inputs[1], inputs[2]))
          inputs = inputs.slice(1, 1);
        bool pending = false;
        for (Value input : inputs) {
          Uniformity next =
              atUse(input, op->getBlock(), getProgramPointAfter(op));
          pending |= next.kind == Uniformity::Bottom;
          state = Uniformity::join(state, next.withoutThreadId());
        }
        if (pending && state.kind != Uniformity::Unknown)
          state = Uniformity::Bottom;
      }
      update(result, state);
    }
    return success();
  }

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const UniformityLattice *> arguments,
                         ArrayRef<UniformityLattice *> results) override {
    (void)visitOperation(call, arguments, results);
  }

  // A cross-block use may observe a different iteration of its definition.
  // For the basic CFG analysis, require uniform control throughout the region.
  // Values defined at region entry and literal constants need no such check.
  Uniformity atUse(Value value, Block *use, ProgramPoint *point) {
    Uniformity state = valueState(value, point);
    Block *def = value.getParentBlock();
    if (!def || def->isEntryBlock() || constant(value))
      return state;
    Block *localUse = def->getParent()->findAncestorBlockInRegion(*use);
    if (localUse != def &&
        cfgControl(def->getParent(), point).kind == Uniformity::Unknown)
      return Uniformity::Unknown;
    return state;
  }

private:
  std::optional<Uniformity> reduceThreadId(Operation *op) {
    bool mask = isa<arith::AndIOp, index::AndOp, LLVM::AndOp>(op);
    bool divide = isa<arith::DivUIOp, index::DivUOp, LLVM::UDivOp>(op);
    bool shift = isa<arith::ShRUIOp, index::ShrUOp, LLVM::LShrOp>(op);
    if ((!mask && !divide && !shift) ||
        !isa<IntegerType, IndexType>(op->getResult(0).getType()))
      return std::nullopt;
    Value input = op->getOperand(0);
    APInt amount;
    if (!matchPattern(op->getOperand(1), m_ConstantInt(&amount))) {
      if (!mask || !matchPattern(input, m_ConstantInt(&amount)))
        return std::nullopt;
      input = op->getOperand(1);
    }
    Uniformity state = atUse(input, op->getBlock(), getProgramPointAfter(op));
    if (state.kind == Uniformity::Bottom)
      return state;
    if (state.kind != Uniformity::ThreadId)
      return std::nullopt;
    if (mask) {
      APInt varying(amount.getBitWidth(), state.threadIdSize - 1);
      return Uniformity::threadId(1u << (varying & amount).getActiveBits());
    }
    unsigned bits;
    if (divide) {
      // Other divisors can cross a quotient boundary within the subgroup.
      if (!amount.isPowerOf2())
        return Uniformity::Unknown;
      bits = amount.logBase2();
    } else {
      // Index may lower to i32 or i64; only accept shifts valid for both.
      auto type = dyn_cast<IntegerType>(input.getType());
      unsigned width = type ? type.getWidth() : 32;
      if (amount.uge(width))
        return Uniformity::Unknown;
      bits = amount.getZExtValue();
    }
    if (bits >= llvm::Log2_32(state.threadIdSize))
      return Uniformity::Uniform;
    return Uniformity::threadId(state.threadIdSize >> bits);
  }

  Uniformity valueState(Value value, ProgramPoint *point) {
    const auto *lattice = point ? getLatticeElementFor(point, value)
                                : solver.lookupState<UniformityLattice>(value);
    return lattice ? lattice->getValue() : Uniformity::Unknown;
  }

  void update(UniformityLattice *lattice, Uniformity state) {
    propagateIfChanged(lattice, lattice->join(state));
  }

  void setToEntryState(UniformityLattice *lattice) override {
    auto arg = dyn_cast<BlockArgument>(lattice->getAnchor());
    bool uniform = arg && arg.getOwner()->isEntryBlock() &&
                   arg.getOwner()->getParentOp() == function &&
                   options.isUniformArgument && options.isUniformArgument(arg);
    update(lattice, uniform ? Uniformity::Uniform : Uniformity::Unknown);
  }

  // Joining uniform values across edges is valid only if the edge selection
  // agrees across lanes, or every edge forwards the same value. Inspect all
  // branch decisions in a CFG region: this deliberately gives up precision
  // after reconvergence, but also handles diamonds with unconditional incoming
  // edges and irreducible CFGs without a separate control-dependence analysis.
  Uniformity cfgControl(Region *region, ProgramPoint *point) {
    Uniformity state = Uniformity::Uniform;
    for (Block &block : *region) {
      auto *blockPoint = getProgramPointBefore(&block);
      const auto *live = point ? getOrCreateFor<Executable>(point, blockPoint)
                               : solver.lookupState<Executable>(blockPoint);
      if (!live || !live->isLive())
        continue;
      Operation *term = block.getTerminator();
      if (term->getNumSuccessors() <= 1)
        continue;
      // The interface identifies forwarded operands, but is not a contract
      // that an arbitrary operation selects its edge deterministically.
      if (!isa<cf::CondBranchOp, cf::SwitchOp, LLVM::CondBrOp, LLVM::SwitchOp>(
              term))
        return Uniformity::Unknown;
      auto branch = dyn_cast<BranchOpInterface>(term);
      if (!branch)
        return Uniformity::Unknown;
      llvm::SmallBitVector forwarded(term->getNumOperands());
      for (unsigned i = 0; i < term->getNumSuccessors(); ++i) {
        OperandRange args =
            branch.getSuccessorOperands(i).getForwardedOperands();
        for (unsigned j = 0; j < args.size(); ++j)
          forwarded.set(args.getBeginOperandIndex() + j);
      }
      for (auto [i, operand] : llvm::enumerate(term->getOperands()))
        if (!forwarded[i])
          state = Uniformity::join(
              state, valueState(operand, point).withoutThreadId());
    }
    return state;
  }

  void visitCFGBlock(Block *block) {
    ProgramPoint *point = getProgramPointBefore(block);
    if (!getOrCreateFor<Executable>(point, point)->isLive())
      return;
    Uniformity control = cfgControl(block->getParent(), point);
    for (BlockArgument arg : block->getArguments()) {
      SmallVector<Value> incoming;
      for (auto it = block->pred_begin(); it != block->pred_end(); ++it) {
        if (!getOrCreateFor<Executable>(point,
                                        getLatticeAnchor<CFGEdge>(*it, block))
                 ->isLive())
          continue;
        auto branch = dyn_cast<BranchOpInterface>((*it)->getTerminator());
        if (!branch) {
          update(getLatticeElement(arg), Uniformity::Unknown);
          continue;
        }
        incoming.push_back(branch.getSuccessorOperands(
            it.getSuccessorIndex())[arg.getArgNumber()]);
      }
      joinInputs(arg, incoming, control, block, point);
    }
  }

  void joinInputs(Value result, ArrayRef<Value> incoming, Uniformity control,
                  Block *use, ProgramPoint *point) {
    Uniformity state;
    bool same = !incoming.empty();
    for (Value input : incoming) {
      if (!input) {
        state = Uniformity::Unknown;
        same = false;
        continue;
      }
      same &= incoming.front() && sameValue(input, incoming.front());
      state = Uniformity::join(state, atUse(input, use, point));
    }
    if (!same && control.kind == Uniformity::Unknown)
      state = Uniformity::Unknown;
    update(getLatticeElement(result), state);
  }

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor, ValueRange inputs,
      ArrayRef<UniformityLattice *> lattices) override {
    auto loop = dyn_cast<scf::ForOp>(op);
    for (auto [input, lattice] : llvm::zip(inputs, lattices)) {
      if (!loop || input != loop.getInductionVar()) {
        update(lattice, Uniformity::Unknown);
        continue;
      }
      ProgramPoint *point = getProgramPointBefore(loop.getBody());
      Uniformity lower =
          atUse(loop.getLowerBound(), op->getBlock(), point).withoutThreadId();
      Uniformity step =
          atUse(loop.getStep(), op->getBlock(), point).withoutThreadId();
      Uniformity state = Uniformity::join(lower, step);
      if ((lower.kind == Uniformity::Bottom ||
           step.kind == Uniformity::Bottom) &&
          state.kind != Uniformity::Unknown)
        state = Uniformity::Bottom;
      update(lattice, state);
    }
  }

  void
  visitRegionSuccessors(ProgramPoint *point, RegionBranchOpInterface branch,
                        RegionSuccessor successor,
                        ArrayRef<AbstractSparseLattice *> lattices) override {
    Operation *op = branch;
    if (!isa<scf::IfOp, scf::ForOp, scf::WhileOp, scf::ExecuteRegionOp,
             scf::IndexSwitchOp>(op)) {
      for (auto *lattice : lattices)
        update(static_cast<UniformityLattice *>(lattice), Uniformity::Unknown);
      return;
    }
    const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
    if (!predecessors->allPredecessorsKnown())
      return;
    Uniformity control = Uniformity::Uniform;
    Block *use = point->isBlockStart() ? point->getBlock() : op->getBlock();
    // SCF recurrences agree within an iteration even with a varying trip
    // count. Apply the loop's exit decision when propagating its results.
    if (!point->isBlockStart()) {
      for (Value operand : op->getOperands())
        control = Uniformity::join(
            control, atUse(operand, op->getBlock(), point).withoutThreadId());
      for (Region &region : op->getRegions()) {
        control = Uniformity::join(control, cfgControl(&region, point));
        for (Block &block : region)
          if (auto cond = dyn_cast<scf::ConditionOp>(block.getTerminator()))
            control = Uniformity::join(
                control,
                atUse(cond.getCondition(), &block, point).withoutThreadId());
      }
    }
    DenseMap<Value, SmallVector<Value>> incoming;
    for (Operation *pred : predecessors->getKnownPredecessors()) {
      std::optional<OperandRange> operands;
      if (pred == op)
        operands = branch.getEntrySuccessorOperands(successor);
      else if (auto term = dyn_cast<RegionBranchTerminatorOpInterface>(pred))
        operands = term.getSuccessorOperands(successor);
      if (!operands) {
        for (auto *lattice : lattices)
          update(static_cast<UniformityLattice *>(lattice),
                 Uniformity::Unknown);
        return;
      }
      for (auto [input, operand] :
           llvm::zip(predecessors->getSuccessorInputs(pred), *operands))
        incoming[input].push_back(operand);
    }
    for (auto &[result, values] : incoming)
      joinInputs(result, values, control, use, point);
    // Even a single exiting edge may carry values from different iterations.
    if (!point->isBlockStart() && isa<scf::ForOp, scf::WhileOp>(op) &&
        control.kind == Uniformity::Unknown)
      for (auto *lattice : lattices)
        update(static_cast<UniformityLattice *>(lattice), Uniformity::Unknown);
    SmallVector<Value> inputs = branch.getNonSuccessorInputs(successor);
    SmallVector<UniformityLattice *> nonSuccessorLattices;
    for (Value input : inputs)
      nonSuccessorLattices.push_back(getLatticeElement(input));
    visitNonControlFlowArguments(op, successor, inputs, nonSuccessorLattices);
  }

  DataFlowSolver &solver;
  Operation *function;
  UniformityAnalysis::Options options;
  unsigned subgroupSize;
};
} // namespace

struct UniformityAnalysis::Impl {
  DataFlowSolver solver{DataFlowConfig().setInterprocedural(false)};
  UniformityDataFlow *analysis = nullptr;
  LogicalResult status = failure();

  Impl(Operation *function, Options options) {
    auto callable = dyn_cast<FunctionOpInterface>(function);
    if (!callable || callable.getFunctionBody().empty())
      return;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    analysis = solver.load<UniformityDataFlow>(function, std::move(options));
    status = solver.initializeAndRun(function);
  }

  bool uniform(Value value) const {
    auto *lattice = solver.lookupState<UniformityLattice>(value);
    return succeeded(status) && lattice &&
           lattice->getValue().kind == Uniformity::Uniform;
  }
};

UniformityAnalysis::UniformityAnalysis(Operation *function)
    : UniformityAnalysis(function, Options{}) {}
UniformityAnalysis::UniformityAnalysis(Operation *function, Options options)
    : impl(std::make_unique<Impl>(function, std::move(options))) {}
UniformityAnalysis::~UniformityAnalysis() = default;
LogicalResult UniformityAnalysis::getStatus() const { return impl->status; }
bool UniformityAnalysis::isUniform(Value value) const {
  return impl->uniform(value);
}
bool UniformityAnalysis::isUniform(Value value, Operation *user) const {
  return impl->uniform(value) && user && user->getBlock() &&
         impl->analysis->atUse(value, user->getBlock(), nullptr).kind ==
             Uniformity::Uniform;
}
bool UniformityAnalysis::isUniform(const OpOperand &use) const {
  return isUniform(use.get(), use.getOwner());
}
