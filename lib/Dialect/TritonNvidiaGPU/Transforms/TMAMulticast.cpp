#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAMulticast.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/TMAMulticast.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

class ProgramIdDependencyAnalysis {
public:
  explicit ProgramIdDependencyAnalysis(ArrayRef<int> clusterDims)
      : clusterDims(clusterDims) {}

  FailureOr<llvm::SmallBitVector> get(Value value) {
    if (auto it = cache.find(value); it != cache.end())
      return it->second;
    bool isRoot = active.empty();
    if (isRoot)
      backEdgeEpoch = 0;
    unsigned epochAtEntry = backEdgeEpoch;
    // A loop-carried back-edge contributes no new dependency to this monotone
    // union. Do not cache intermediate values while the cycle is open: their
    // result does not yet include the loop argument's initial dependencies.
    if (!active.insert(value).second) {
      ++backEdgeEpoch;
      return llvm::SmallBitVector(3);
    }

    FailureOr<llvm::SmallBitVector> result = analyze(value);
    active.erase(value);
    if (succeeded(result) && (isRoot || epochAtEntry == backEdgeEpoch))
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
    if (isa<ttng::CLCAdvanceOp, ttng::CLCReadOp>(op)) {
      unsigned resultNumber = cast<OpResult>(value).getResultNumber();
      llvm::SmallBitVector deps(3);
      // CLC operation semantics define a clustered claim as one cluster-scoped
      // request whose response has a group-uniform validity bit followed by
      // per-CTA X/Y/Z coordinates reconstructed from the claimed cluster base.
      // The identity of the elected issuing CTA is a lowering detail.
      if (resultNumber != 0)
        deps.set(resultNumber - 1);
      return deps;
    }
    if (isa<arith::ConstantOp, tt::GetNumProgramsOp>(op))
      return llvm::SmallBitVector(3);

    auto analyzeClusterQuotient =
        [&](Value lhs, Value rhs) -> FailureOr<llvm::SmallBitVector> {
      auto pid = lhs.getDefiningOp<tt::GetProgramIdOp>();
      auto constant = rhs.getDefiningOp<arith::ConstantOp>();
      auto divisor =
          constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
      if (!pid || !divisor)
        return failure();
      unsigned axis = static_cast<unsigned>(pid.getAxis());
      if (divisor.getInt() != clusterDims[axis])
        return failure();
      // Physical clusters are aligned to their shape, so dividing a physical
      // program coordinate by the exact cluster dimension yields the same
      // cluster coordinate for every CTA in that cluster.
      return llvm::SmallBitVector(3);
    };
    if (auto div = dyn_cast<arith::DivSIOp>(op)) {
      auto deps = analyzeClusterQuotient(div.getLhs(), div.getRhs());
      if (succeeded(deps))
        return deps;
    }
    if (auto div = dyn_cast<arith::DivUIOp>(op)) {
      auto deps = analyzeClusterQuotient(div.getLhs(), div.getRhs());
      if (succeeded(deps))
        return deps;
    }

    // TODO: Model CLC and mutable-memory dependencies using the latest
    // schedule instead of rejecting them conservatively.
    if (isa<ttng::CLCTryCancelOp, ttng::CLCLoadResultOp, ttng::CLCIsCanceledOp,
            ttng::CLCGetProgramIdOp, ttng::CLCTryCancelAsyncOp,
            ttng::CLCQueryCancelOp, tt::AtomicRMWOp, tt::AtomicCASOp,
            tt::LoadOp>(op))
      return failure();
    if (!isa<arith::ArithDialect>(op->getDialect()) &&
        !isa<tt::SplatOp, tt::BroadcastOp, tt::ExpandDimsOp,
             tt::MakeTensorDescOp, ttng::ReinterpretTensorDescOp>(op))
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
  llvm::SmallVector<int, 3> clusterDims;
  unsigned backEdgeEpoch = 0;
};

} // namespace

namespace mlir::triton::nvidia_gpu {

FailureOr<TMAClusterGeometry> TMAClusterGeometry::get(ModuleOp module) {
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

static FailureOr<TMAMulticastPlan>
analyzeTMAMulticast(tt::DescriptorLoadOp load) {
  if (!load.getMulticast())
    return failure();
  auto module = load->getParentOfType<ModuleOp>();
  auto func = load->getParentOfType<tt::FuncOp>();
  if (!func || !tt::isKernel(func) || !func.getBody().hasOneBlock())
    return failure();
  auto geometry = TMAClusterGeometry::get(module);
  if (failed(geometry))
    return failure();

  ProgramIdDependencyAnalysis analysis(geometry->dims);
  llvm::SmallBitVector varyingAxes(3);
  for (Value index : load.getIndices()) {
    auto deps = analysis.get(index);
    if (failed(deps))
      return failure();
    varyingAxes |= *deps;
  }

  // Pid-invariant indices are not sufficient: a descriptor whose base / shape /
  // strides derive from program_id (e.g. `base + pid * stride`) can be loaded
  // at identical indices across CTAs yet address different tiles, so
  // multicasting the leader's tile would corrupt the others. Fold the
  // descriptor operand's pid-dependence into varyingAxes as well; a computed
  // descriptor whose invariance the analysis cannot prove fails here and the
  // load is rejected (no multicast) rather than being assumed broadcastable.
  auto descDeps = analysis.get(load.getDesc());
  if (failed(descDeps))
    return failure();
  varyingAxes |= *descDeps;

  llvm::SmallBitVector broadcastAxes(3);
  llvm::SmallBitVector synchronizationAxes(3);
  for (unsigned axis = 0; axis < 3; ++axis)
    if (geometry->dims[axis] > 1) {
      synchronizationAxes.set(axis);
      if (!varyingAxes.test(axis))
        broadcastAxes.set(axis);
    }

  for (Operation *parent = load->getParentOp();
       parent && !isa<tt::FuncOp>(parent); parent = parent->getParentOp()) {
    SmallVector<Value> controls;
    if (auto ifOp = dyn_cast<scf::IfOp>(parent))
      controls.push_back(ifOp.getCondition());
    else if (auto forOp = dyn_cast<scf::ForOp>(parent))
      controls.append(
          {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()});
    else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      auto condition =
          cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator());
      controls.push_back(condition.getCondition());
    } else if (parent->getNumRegions() != 0)
      return failure();
    for (Value control : controls) {
      auto deps = analysis.get(control);
      // Lowering brackets multicast with full-cluster barriers, including axes
      // outside a data recipient subgroup.
      if (failed(deps) || deps->anyCommon(synchronizationAxes))
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
      load->removeAttr(tt::kMulticastAxesAttrName);
      if (!load.getMulticast())
        return;
      auto plan = analyzeTMAMulticast(load);
      if (failed(plan))
        return;
      SmallVector<int32_t> axes;
      for (int axis : plan->broadcastAxes.set_bits())
        axes.push_back(axis);
      load->setAttr(tt::kMulticastAxesAttrName,
                    DenseI32ArrayAttr::get(load.getContext(), axes));
    });
  }
};

} // namespace mlir::triton::nvidia_gpu
