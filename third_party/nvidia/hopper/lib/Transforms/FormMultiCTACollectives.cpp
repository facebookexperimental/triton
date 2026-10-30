#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

namespace mlir {

#define GEN_PASS_DEF_NVGPUFORMMULTICTACOLLECTIVES
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

struct PhysicalCluster {
  SmallVector<int> dims;
  int64_t size = 1;
  int axis = -1;
  unsigned nonUnitAxes = 0;
};

FailureOr<PhysicalCluster> getPhysicalCluster(ModuleOp module) {
  PhysicalCluster cluster;
  cluster.dims = ttg::TritonGPUDialect::getClusterDims(module);
  for (auto [axis, dim] : llvm::enumerate(cluster.dims)) {
    if (dim < 1)
      return module.emitError("physical cluster dimensions must be positive");
    cluster.size *= dim;
    if (dim != 1) {
      ++cluster.nonUnitAxes;
      cluster.axis = axis;
    }
  }
  return cluster;
}

bool isSupportedScalarCombiner(triton::ReduceOp reduce) {
  if (reduce.getNumResults() != 1)
    return false;
  Type resultType = reduce->getResult(0).getType();
  auto tensorType = dyn_cast<RankedTensorType>(resultType);
  if (!resultType.isF32() &&
      !(tensorType && tensorType.hasStaticShape() &&
        tensorType.getElementType().isF32() && tensorType.getNumElements() > 0))
    return false;
  Operation *combiner = reduce.getSingleCombiner();
  return combiner && isa<arith::AddFOp, arith::MaxNumFOp, arith::MaximumFOp,
                         arith::MinNumFOp, arith::MinimumFOp>(combiner);
}

bool hasOnlyConstantCaptures(triton::ReduceOp reduce) {
  Block &body = reduce.getCombineOp().front();
  return llvm::all_of(body.without_terminator(), [&](Operation &op) {
    return llvm::all_of(op.getOperands(), [&](Value operand) {
      if (auto argument = dyn_cast<BlockArgument>(operand))
        return argument.getOwner() == &body;
      Operation *def = operand.getDefiningOp();
      return def && (def->getBlock() == &body ||
                     def->hasTrait<OpTrait::ConstantLike>());
    });
  });
}

LogicalResult validateFunction(triton::FuncOp function, int axis, int size) {
  bool sawLocalRank = false;
  WalkResult result = function.walk([&](Operation *op) {
    if (isa<triton::CallOp>(op)) {
      op->emitError("multi_cta does not support calls in v1");
      return WalkResult::interrupt();
    }
    if (isa<ttg::WarpSpecializeOp>(op)) {
      op->emitError("multi_cta does not support warp specialization in v1");
      return WalkResult::interrupt();
    }
    auto pid = dyn_cast<triton::GetProgramIdOp>(op);
    if (!pid || pid.getAxisAsInt() != axis)
      return WalkResult::advance();
    if (pid->getResult(0).use_empty()) {
      pid.emitError("cluster-axis program ID must define local rank");
      return WalkResult::interrupt();
    }
    for (OpOperand &use : pid->getResult(0).getUses()) {
      Operation *owner = use.getOwner();
      Value divisor;
      if (auto rem = dyn_cast<arith::RemSIOp>(owner);
          rem && rem.getLhs() == pid->getResult(0))
        divisor = rem.getRhs();
      else if (auto rem = dyn_cast<arith::RemUIOp>(owner);
               rem && rem.getLhs() == pid->getResult(0))
        divisor = rem.getRhs();
      else {
        pid.emitError(
            "cluster-axis program ID may only be used as local_rank = "
            "program_id % P");
        return WalkResult::interrupt();
      }
      APInt divisorValue;
      if (!matchPattern(divisor, m_ConstantInt(&divisorValue)) ||
          divisorValue.getSExtValue() != size) {
        owner->emitError("local-rank divisor must equal physical cluster size ")
            << size;
        return WalkResult::interrupt();
      }
      sawLocalRank = true;
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  if (!sawLocalRank)
    return function.emitError(
        "multi_cta requires local_rank = program_id(cluster_axis) % P");
  return success();
}

std::optional<int64_t> getConstantInt(Value value) {
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)))
    return std::nullopt;
  return constant.getSExtValue();
}

std::optional<int64_t> getRankBlock(Value lowerBound,
                                    ArrayRef<Value> localRanks) {
  auto mul = lowerBound.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return std::nullopt;
  auto match = [&](Value rank, Value block) -> std::optional<int64_t> {
    if (!llvm::is_contained(localRanks, rank))
      return std::nullopt;
    auto value = getConstantInt(block);
    return value && *value > 0 ? value : std::nullopt;
  };
  if (auto block = match(mul.getLhs(), mul.getRhs()))
    return block;
  return match(mul.getRhs(), mul.getLhs());
}

bool dependsOnLoop(Value value, ArrayRef<scf::ForOp> loops,
                   llvm::SmallPtrSetImpl<Operation *> &visited) {
  Operation *def = value.getDefiningOp();
  if (!def || !visited.insert(def).second)
    return false;
  if (auto loop = dyn_cast<scf::ForOp>(def))
    return llvm::is_contained(loops, loop);
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return dependsOnLoop(operand, loops, visited);
  });
}

LogicalResult validateCanonicalOwnership(triton::ReduceOp reduce, int axis,
                                         int size) {
  auto function = reduce->getParentOfType<triton::FuncOp>();
  SmallVector<Value> localRanks;
  function.walk([&](triton::GetProgramIdOp pid) {
    if (pid.getAxisAsInt() != axis)
      return;
    for (OpOperand &use : pid->getResult(0).getUses())
      if (auto rem = dyn_cast<arith::RemSIOp>(use.getOwner()))
        localRanks.push_back(rem.getResult());
      else if (auto rem = dyn_cast<arith::RemUIOp>(use.getOwner()))
        localRanks.push_back(rem.getResult());
  });

  SmallVector<scf::ForOp> shardLoops;
  function.walk([&](scf::ForOp loop) {
    auto block = getRankBlock(loop.getLowerBound(), localRanks);
    auto step = getConstantInt(loop.getStep());
    if (block && step && *step == *block * size)
      shardLoops.push_back(loop);
  });

  bool hasMemory = false;
  WalkResult memoryResult = function.walk([&](Operation *op) {
    if (!isa<triton::LoadOp, triton::StoreOp>(op))
      return WalkResult::advance();
    hasMemory = true;
    auto loop = op->getParentOfType<scf::ForOp>();
    if (!loop || !llvm::is_contained(shardLoops, loop)) {
      op->emitError("multi_cta memory accesses must be in canonical "
                    "rank-strided loops");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (memoryResult.wasInterrupted())
    return failure();
  if (!hasMemory)
    return success();
  for (Value partial : reduce.getOperands()) {
    llvm::SmallPtrSet<Operation *, 16> visited;
    if (!dependsOnLoop(partial, shardLoops, visited))
      return reduce.emitError(
          "multi_cta partial must depend on a canonical rank-strided loop");
  }
  return success();
}

LogicalResult formCollective(triton::ReduceOp reduce, int axis, int size) {
  SmallVector<SmallVector<OpOperand *>> uses;
  for (Value result : reduce.getResults()) {
    SmallVector<OpOperand *> resultUses;
    for (OpOperand &use : result.getUses())
      resultUses.push_back(&use);
    uses.push_back(std::move(resultUses));
  }

  OpBuilder builder(reduce);
  builder.setInsertionPointAfter(reduce);
  auto collective = ttng::ClusterReduceOp::create(
      builder, reduce.getLoc(), TypeRange(reduce.getResults()),
      reduce.getResults(), axis, size);

  Block &source = reduce.getCombineOp().front();
  Block *target = builder.createBlock(
      &collective.getCombineOp(), {}, source.getArgumentTypes(),
      SmallVector<Location>(source.getNumArguments(), reduce.getLoc()));
  builder.setInsertionPointToStart(target);
  IRMapping mapping;
  for (auto [from, to] :
       llvm::zip(source.getArguments(), target->getArguments()))
    mapping.map(from, to);
  for (Operation &op : source.without_terminator()) {
    for (Value operand : op.getOperands()) {
      if (mapping.contains(operand))
        continue;
      Operation *constant = operand.getDefiningOp();
      if (!constant || !constant->hasTrait<OpTrait::ConstantLike>())
        return reduce.emitError(
            "multi_cta combine region may capture only constants");
      Operation *clone = builder.clone(*constant);
      for (auto [original, copy] :
           llvm::zip(constant->getResults(), clone->getResults()))
        mapping.map(original, copy);
    }
    builder.clone(op, mapping);
  }
  SmallVector<Value> results;
  for (Value result : source.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(result));
  ttng::ClusterReduceReturnOp::create(builder, reduce.getLoc(), results);

  reduce->removeAttr("tt.multi_cta");
  for (auto [resultUses, result] : llvm::zip(uses, collective.getResults()))
    for (OpOperand *use : resultUses)
      use->set(result);
  return success();
}

class NVGPUFormMultiCTACollectivesPass
    : public impl::NVGPUFormMultiCTACollectivesBase<
          NVGPUFormMultiCTACollectivesPass> {
public:
  using impl::NVGPUFormMultiCTACollectivesBase<
      NVGPUFormMultiCTACollectivesPass>::NVGPUFormMultiCTACollectivesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<triton::ReduceOp> reductions;
    WalkResult scan = module.walk([&](Operation *op) {
      if (!op->hasAttr("tt.multi_cta"))
        return WalkResult::advance();
      auto reduce = dyn_cast<triton::ReduceOp>(op);
      if (!reduce) {
        op->emitError("tt.multi_cta is supported only on tt.reduce");
        return WalkResult::interrupt();
      }
      reductions.push_back(reduce);
      return WalkResult::advance();
    });
    if (scan.wasInterrupted())
      return signalPassFailure();
    if (reductions.empty())
      return;

    auto cluster = getPhysicalCluster(module);
    if (failed(cluster))
      return signalPassFailure();
    if (cluster->size == 1) {
      for (triton::ReduceOp reduce : reductions)
        reduce->removeAttr("tt.multi_cta");
      return;
    }
    if (computeCapability != 90 && computeCapability != 100 &&
        computeCapability != 103) {
      module.emitError(
          "multi_cta requires NVIDIA compute capability 90, 100, or 103");
      return signalPassFailure();
    }
    if (ttg::TritonGPUDialect::getNumCTAs(module) != 1) {
      module.emitError("multi_cta requires logical ttg.num-ctas to equal 1");
      return signalPassFailure();
    }
    if (cluster->nonUnitAxes != 1 ||
        (cluster->size != 2 && cluster->size != 4)) {
      module.emitError("multi_cta requires a one-dimensional physical cluster "
                       "of size 2 or 4");
      return signalPassFailure();
    }

    llvm::SmallPtrSet<Operation *, 4> validatedFunctions;
    for (triton::ReduceOp reduce : reductions) {
      auto function = reduce->getParentOfType<triton::FuncOp>();
      if (!function || reduce->getBlock() != &function.getBody().front()) {
        reduce.emitError("multi_cta reduce must be in the tt.func entry block");
        return signalPassFailure();
      }
      if (validatedFunctions.insert(function).second &&
          failed(validateFunction(function, cluster->axis, cluster->size)))
        return signalPassFailure();
      if (failed(
              validateCanonicalOwnership(reduce, cluster->axis, cluster->size)))
        return signalPassFailure();
      if (reduce.hasDefinedOrdering() || !isSupportedScalarCombiner(reduce)) {
        reduce.emitError(
            "multi_cta requires an unordered f32 add, max, or min reduction");
        return signalPassFailure();
      }
      if (!hasOnlyConstantCaptures(reduce)) {
        reduce.emitError("multi_cta combine region may capture only constants");
        return signalPassFailure();
      }
    }
    for (triton::ReduceOp reduce : reductions)
      if (failed(formCollective(reduce, cluster->axis, cluster->size)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace mlir
