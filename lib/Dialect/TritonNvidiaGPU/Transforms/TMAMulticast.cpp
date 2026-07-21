#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAMulticast.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

constexpr StringLiteral kAxesAttr = "tt.multicast_axes";

bool isPolicyEnabled(tt::DescriptorLoadOp load) {
  if (auto local = load->getAttrOfType<BoolAttr>("tt.multicast"))
    return local.getValue();
  auto mod = load->getParentOfType<ModuleOp>();
  if (auto global = mod->getAttrOfType<BoolAttr>("ttg.multicast"))
    return global.getValue();
  return false;
}

class ProgramIdDependencyAnalysis {
public:
  FailureOr<llvm::SmallBitVector> get(Value value) {
    if (auto it = cache.find(value); it != cache.end())
      return it->second;
    // Back-edge: `value` is already on the active stack (a dependency cycle,
    // e.g. a loop-carried arg reached through its own yield). This is a monotone
    // union (join) analysis, so a value's program-id axes are the union of all
    // its contributions and the self-referential back-edge is the join identity
    // (the empty set) — it adds nothing at the fixpoint. Any real pid-dependence
    // still enters through the loop init or a direct (non-cyclic) operand of the
    // recurrence (e.g. `t' = t + step` keeps t's pid axis via the init `t`), so
    // returning the empty set here is sound. Returning failure() instead would
    // conservatively reject every loop-carried index and disable multicast for
    // persistent kernels, which is the primary use case.
    if (!active.insert(value).second)
      return llvm::SmallBitVector(3);

    FailureOr<llvm::SmallBitVector> result = analyze(value);
    active.erase(value);
    if (succeeded(result))
      cache.try_emplace(value, *result);
    return result;
  }

private:
  FailureOr<llvm::SmallBitVector> analyze(Value value) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      Operation *parent = arg.getOwner()->getParentOp();
      if (isa<tt::FuncOp>(parent))
        return llvm::SmallBitVector(3);
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (arg == forOp.getInductionVar()) {
          auto lowerDeps = get(forOp.getLowerBound());
          auto stepDeps = get(forOp.getStep());
          if (failed(lowerDeps) || failed(stepDeps))
            return failure();
          *lowerDeps |= *stepDeps;
          return *lowerDeps;
        }
        unsigned index = arg.getArgNumber() - 1;
        return mergeLoopValues(
            forOp.getInitArgs()[index],
            forOp.getBody()->getTerminator()->getOperand(index));
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
        unsigned index = arg.getArgNumber();
        if (arg.getOwner() == whileOp.getBeforeBody())
          return mergeLoopValues(
              whileOp.getInits()[index],
              whileOp.getAfterBody()->getTerminator()->getOperand(index));
        return get(
            whileOp.getBeforeBody()->getTerminator()->getOperand(index + 1));
      }
      return failure();
    }

    Operation *op = value.getDefiningOp();
    if (!op)
      return failure();
    if (auto pid = dyn_cast<tt::GetProgramIdOp>(op)) {
      llvm::SmallBitVector deps(3);
      deps.set(static_cast<unsigned>(pid.getAxis()));
      return deps;
    }
    if (isa<arith::ConstantOp, tt::GetNumProgramsOp>(op))
      return llvm::SmallBitVector(3);

    StringRef name = op->getName().getStringRef();
    if (name.contains("clc") ||
        isa<tt::AtomicRMWOp, tt::AtomicCASOp, tt::LoadOp>(op))
      return failure();
    if (!isa<arith::ArithDialect>(op->getDialect()) &&
        !isa<tt::SplatOp, tt::BroadcastOp, tt::ExpandDimsOp>(op))
      return failure();
    if (!isMemoryEffectFree(op))
      return failure();

    llvm::SmallBitVector deps(3);
    for (Value operand : op->getOperands()) {
      auto operandDeps = get(operand);
      if (failed(operandDeps))
        return failure();
      deps |= *operandDeps;
    }
    return deps;
  }

  FailureOr<llvm::SmallBitVector> mergeLoopValues(Value init, Value next) {
    auto initDeps = get(init);
    auto nextDeps = get(next);
    if (failed(initDeps) || failed(nextDeps))
      return failure();
    *initDeps |= *nextDeps;
    return *initDeps;
  }

  DenseMap<Value, llvm::SmallBitVector> cache;
  llvm::SmallPtrSet<Value, 16> active;
};

} // namespace

namespace mlir::triton::nvidia_gpu {

FailureOr<TMAClusterGeometry> TMAClusterGeometry::get(ModuleOp module) {
  auto usesPhysicalClusters =
      module->getAttrOfType<BoolAttr>("ttg.ctas-per-cga");
  if (!usesPhysicalClusters || !usesPhysicalClusters.getValue())
    return failure();
  TMAClusterGeometry geometry{ttg::TritonGPUDialect::getClusterDims(module)};
  if (geometry.dims.size() != 3 ||
      llvm::any_of(geometry.dims, [](int dim) { return dim <= 0; }) ||
      llvm::any_of(geometry.dims,
                   [](int dim) { return !llvm::isPowerOf2_32(dim); }) ||
      geometry.size() <= 1 || geometry.size() > 16)
    return failure();
  return geometry;
}

unsigned TMAClusterGeometry::size() const {
  return static_cast<unsigned>(dims[0] * dims[1] * dims[2]);
}

llvm::SmallVector<int, 3> TMAClusterGeometry::coordinates(unsigned rank) const {
  llvm::SmallVector<int, 3> coord(3);
  coord[0] = rank % dims[0];
  rank /= dims[0];
  coord[1] = rank % dims[1];
  coord[2] = rank / dims[1];
  return coord;
}

uint16_t
TMAClusterGeometry::maskFor(unsigned rank,
                            const llvm::SmallBitVector &broadcastAxes) const {
  auto source = coordinates(rank);
  uint16_t mask = 0;
  for (unsigned candidate = 0; candidate < size(); ++candidate) {
    auto target = coordinates(candidate);
    bool sameGroup = true;
    for (unsigned axis = 0; axis < 3; ++axis)
      if (!broadcastAxes.test(axis) && source[axis] != target[axis])
        sameGroup = false;
    if (sameGroup)
      mask |= uint16_t(1u << candidate);
  }
  return mask;
}

unsigned
TMAClusterGeometry::leaderFor(unsigned rank,
                              const llvm::SmallBitVector &broadcastAxes) const {
  uint16_t mask = maskFor(rank, broadcastAxes);
  return llvm::countr_zero(static_cast<unsigned>(mask));
}

FailureOr<TMAMulticastPlan> analyzeTMAMulticast(tt::DescriptorLoadOp load) {
  if (!isPolicyEnabled(load))
    return failure();
  auto module = load->getParentOfType<ModuleOp>();
  if (auto metaWS = module->getAttrOfType<BoolAttr>("ttg.use-meta-ws");
      metaWS && metaWS.getValue())
    return failure();
  auto geometry = TMAClusterGeometry::get(module);
  if (failed(geometry))
    return failure();

  ProgramIdDependencyAnalysis analysis;
  llvm::SmallBitVector varyingAxes(3);
  for (Value index : load.getIndices()) {
    auto deps = analysis.get(index);
    if (failed(deps))
      return failure();
    varyingAxes |= *deps;
  }

  // Pid-invariant indices are not sufficient: a descriptor whose base / shape /
  // strides derive from program_id (e.g. `base + pid * stride`) can be loaded at
  // identical indices across CTAs yet address different tiles, so multicasting
  // the leader's tile would corrupt the others. Fold the descriptor operand's
  // pid-dependence into varyingAxes as well; a computed descriptor whose
  // invariance the analysis cannot prove fails here and the load is rejected
  // (no multicast) rather than being assumed broadcastable.
  auto descDeps = analysis.get(load.getDesc());
  if (failed(descDeps))
    return failure();
  varyingAxes |= *descDeps;

  llvm::SmallBitVector broadcastAxes(3);
  for (unsigned axis = 0; axis < 3; ++axis)
    if (geometry->dims[axis] > 1 && !varyingAxes.test(axis))
      broadcastAxes.set(axis);

  for (Operation *parent = load->getParentOp();
       parent && !isa<tt::FuncOp>(parent); parent = parent->getParentOp()) {
    SmallVector<Value> controls;
    if (auto ifOp = dyn_cast<scf::IfOp>(parent))
      controls.push_back(ifOp.getCondition());
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      controls.append(
          {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()});
    if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      auto condition =
          cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator());
      controls.push_back(condition.getCondition());
    }
    for (Value control : controls) {
      auto deps = analysis.get(control);
      if (failed(deps) || deps->anyCommon(broadcastAxes))
        return failure();
    }
  }

  if (broadcastAxes.none())
    return failure();
  return TMAMulticastPlan{*geometry, broadcastAxes};
}

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMAMULTICASTPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

class TritonNvidiaGPUTMAMulticastPass
    : public impl::TritonNvidiaGPUTMAMulticastPassBase<
          TritonNvidiaGPUTMAMulticastPass> {
  void runOnOperation() override {
    getOperation().walk([](tt::DescriptorLoadOp load) {
      load->removeAttr(kAxesAttr);
      auto plan = analyzeTMAMulticast(load);
      if (failed(plan))
        return;
      SmallVector<int32_t> axes;
      for (int axis : plan->broadcastAxes.set_bits())
        axes.push_back(axis);
      load->setAttr(kAxesAttr, DenseI32ArrayAttr::get(load.getContext(), axes));
    });
  }
};

} // namespace mlir::triton::nvidia_gpu
