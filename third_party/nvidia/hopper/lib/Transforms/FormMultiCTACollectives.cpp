#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/ADT/SmallPtrSet.h"

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
  if (reduce.getNumResults() != 1 || !reduce->getResult(0).getType().isF32())
    return false;
  Operation *combiner = reduce.getSingleCombiner();
  return combiner && isa<arith::AddFOp, arith::MaxNumFOp, arith::MaximumFOp,
                         arith::MinNumFOp, arith::MinimumFOp>(combiner);
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
  for (Operation &op : source.without_terminator())
    builder.clone(op, mapping);
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
      if (reduce.hasDefinedOrdering() || !isSupportedScalarCombiner(reduce)) {
        reduce.emitError("multi_cta requires an unordered scalar f32 add, max, "
                         "or min reduction");
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
