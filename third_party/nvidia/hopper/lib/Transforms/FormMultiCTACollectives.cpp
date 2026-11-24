#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <limits>
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
    if (cluster.size > std::numeric_limits<int64_t>::max() / dim)
      return module.emitError("physical cluster size overflows int64_t");
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
      !(tensorType && tensorType.getRank() > 0 && tensorType.hasStaticShape() &&
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
    // SM90 specialization may carry these annotations without enabling the
    // Meta AutoWS option. Every physical CTA must execute the collective with
    // the same warps, so reject annotated and materialized specialization.
    if (op->hasAttr("tt.warp_specialize") || op->hasAttr("async_task_id") ||
        op->hasAttr("ttg.partition")) {
      op->emitError("multi_cta does not support warp specialization in v1");
      return WalkResult::interrupt();
    }
    if (isa<ttg::WarpSpecializeOp>(op)) {
      op->emitError("multi_cta does not support warp specialization in v1");
      return WalkResult::interrupt();
    }
    if (isa<triton::nvgpu::ClusterCTAIdOp>(op)) {
      sawLocalRank = true;
      return WalkResult::advance();
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
        "multi_cta requires local_rank = program_id(cluster_axis) % P or "
        "cluster_cta_rank()");
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

bool dependsOnAny(Value value, ArrayRef<Value> roots,
                  llvm::DenseSet<Value> &visited) {
  if (llvm::is_contained(roots, value))
    return true;
  if (!visited.insert(value).second)
    return false;

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop =
        dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!loop)
      return !isa_and_nonnull<triton::FuncOp>(
          argument.getOwner()->getParentOp());
    if (argument == loop.getInductionVar())
      return dependsOnAny(loop.getLowerBound(), roots, visited) ||
             dependsOnAny(loop.getUpperBound(), roots, visited) ||
             dependsOnAny(loop.getStep(), roots, visited);
    unsigned index = argument.getArgNumber() - 1;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    return dependsOnAny(loop.getLowerBound(), roots, visited) ||
           dependsOnAny(loop.getUpperBound(), roots, visited) ||
           dependsOnAny(loop.getStep(), roots, visited) ||
           dependsOnAny(loop.getInitArgs()[index], roots, visited) ||
           dependsOnAny(yield.getResults()[index], roots, visited);
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (auto loop = dyn_cast<scf::ForOp>(def)) {
    auto result = cast<OpResult>(value);
    unsigned index = result.getResultNumber();
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    return dependsOnAny(loop.getLowerBound(), roots, visited) ||
           dependsOnAny(loop.getUpperBound(), roots, visited) ||
           dependsOnAny(loop.getStep(), roots, visited) ||
           dependsOnAny(loop.getInitArgs()[index], roots, visited) ||
           dependsOnAny(yield.getResults()[index], roots, visited);
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    unsigned index = cast<OpResult>(value).getResultNumber();
    if (dependsOnAny(ifOp.getCondition(), roots, visited))
      return true;
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      if (region->empty())
        continue;
      auto yield = cast<scf::YieldOp>(region->front().getTerminator());
      if (dependsOnAny(yield.getResults()[index], roots, visited))
        return true;
    }
  }
  // Region captures are not explicit operands. Fail closed for region
  // results whose control flow and block arguments are not modeled above.
  if (def->getNumRegions() != 0 && !isa<scf::IfOp>(def))
    return true;
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return dependsOnAny(operand, roots, visited);
  });
}

bool isZeroInteger(Value value) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) && constant.isZero();
}

// Return the largest nonnegative offset. Bounds accumulate across the entire
// IV-dependent projection: independently bounded additions can still overlap
// the next CTA's tile when their sum reaches BLOCK.
std::optional<int64_t> getLaneOffsetUpperBound(Value value, int64_t block) {
  if (isZeroInteger(value))
    return 0;
  if (auto constant = getConstantInt(value))
    return *constant >= 0 && *constant < block ? constant : std::nullopt;
  if (auto range = value.getDefiningOp<triton::MakeRangeOp>()) {
    int64_t start = range.getStartAttr().getInt();
    int64_t end = range.getEndAttr().getInt();
    if (start >= 0 && end > start && end <= block)
      return end - 1;
    return std::nullopt;
  }
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto lhs = getLaneOffsetUpperBound(add.getLhs(), block);
    auto rhs = getLaneOffsetUpperBound(add.getRhs(), block);
    if (lhs && rhs && *lhs <= block - 1 - *rhs)
      return *lhs + *rhs;
    return std::nullopt;
  }
  if (def->getNumOperands() != 1)
    return std::nullopt;
  StringRef name = def->getName().getStringRef();
  if (name != "tt.splat" && name != "tt.broadcast" &&
      name != "tt.expand_dims" && name != "arith.extsi" &&
      name != "arith.extui" && name != "arith.index_cast" &&
      name != "ttg.convert_layout")
    return std::nullopt;
  return getLaneOffsetUpperBound(def->getOperand(0), block);
}

// A rank-independent pointer can still move between shard iterations.
// Only accept values defined outside the loop, or pure regionless expressions
// of such values. In-loop splats of kernel/row pointers remain admissible.
bool isLoopInvariantBase(Value value, scf::ForOp loop,
                         llvm::DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = argument.getOwner()->getParentOp();
    return parent != loop && !loop->isAncestor(parent);
  }
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (!loop->isAncestor(def))
    return true;
  if (def->getNumRegions() != 0 || !isPure(def))
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    llvm::DenseSet<Value> operandVisited;
    return isLoopInvariantBase(operand, loop, operandVisited);
  });
}

// Identify the varying lane dimension through logical address projections.
// Ambiguous or absent dimensions use a conservative bound over all dimensions.
struct LaneProjection {
  bool known;
  std::optional<unsigned> axis;
};

LaneProjection getLaneProjection(Value value) {
  if (!isa<RankedTensorType>(value.getType()))
    return {true, std::nullopt};
  if (value.getDefiningOp<triton::MakeRangeOp>())
    return {true, 0};
  Attribute constant;
  if (matchPattern(value, m_Constant(&constant))) {
    auto dense = dyn_cast<DenseElementsAttr>(constant);
    return {dense && dense.isSplat(), std::nullopt};
  }
  Operation *def = value.getDefiningOp();
  if (!def)
    return {false, std::nullopt};
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto lhs = getLaneProjection(add.getLhs());
    auto rhs = getLaneProjection(add.getRhs());
    if (!lhs.known || !rhs.known ||
        (lhs.axis && rhs.axis && lhs.axis != rhs.axis))
      return {false, std::nullopt};
    return {true, lhs.axis ? lhs.axis : rhs.axis};
  }
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
    auto projection = getLaneProjection(expand.getSrc());
    if (projection.axis &&
        *projection.axis >= static_cast<unsigned>(expand.getAxis()))
      ++*projection.axis;
    return projection;
  }
  StringRef name = def->getName().getStringRef();
  if (def->getNumOperands() == 1 &&
      (name == "tt.splat" || name == "tt.broadcast" ||
       name == "ttg.convert_layout" || name == "arith.extsi" ||
       name == "arith.extui" || name == "arith.index_cast"))
    return getLaneProjection(def->getOperand(0));
  return {false, std::nullopt};
}

bool isLaneUniform(Value value, std::optional<unsigned> axis) {
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor || tensor.getRank() == 0)
    return true;
  if (axis && *axis < static_cast<unsigned>(tensor.getRank()) &&
      tensor.getShape()[*axis] == 1)
    return true;
  Attribute constant;
  if (matchPattern(value, m_Constant(&constant))) {
    auto dense = dyn_cast<DenseElementsAttr>(constant);
    return dense && dense.isSplat();
  }
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (isa<triton::SplatOp>(def))
    return true;
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
    if (axis && *axis > static_cast<unsigned>(expand.getAxis()))
      --*axis;
    return isLaneUniform(expand.getSrc(), axis);
  }
  if (isa<triton::BroadcastOp, ttg::ConvertLayoutOp>(def))
    return isLaneUniform(def->getOperand(0), axis);
  // Arithmetic and pointer addition have explicit pointwise semantics.
  // Unknown operations, including opaque assembly, cannot prove uniformity.
  auto *dialect = def->getDialect();
  bool pointwise = isa<triton::AddPtrOp>(def) ||
                   (dialect && dialect->getNamespace() == "arith" &&
                    def->hasTrait<OpTrait::Elementwise>());
  return pointwise && isPure(def) && def->getNumOperands() > 0 &&
         llvm::all_of(def->getOperands(), [&](Value operand) {
           return isLaneUniform(operand, axis);
         });
}

std::optional<int64_t>
getBaseLaneOffsetUpperBound(Value value, int64_t block,
                            std::optional<unsigned> axis) {
  if (isLaneUniform(value, axis))
    return 0;
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto lhs = getBaseLaneOffsetUpperBound(add.getLhs(), block, axis);
    auto rhs = getBaseLaneOffsetUpperBound(add.getRhs(), block, axis);
    if (lhs && rhs && *lhs <= block - 1 - *rhs)
      return *lhs + *rhs;
    return std::nullopt;
  }
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
    if (axis && *axis > static_cast<unsigned>(expand.getAxis()))
      --*axis;
    return getBaseLaneOffsetUpperBound(expand.getSrc(), block, axis);
  }
  if (isa<triton::BroadcastOp, ttg::ConvertLayoutOp>(def))
    return getBaseLaneOffsetUpperBound(def->getOperand(0), block, axis);
  return getLaneOffsetUpperBound(value, block);
}

// Account for lane offsets in an invariant base as well as in the IV path.
// Orthogonal row bases remain covered by the caller's disjoint-row contract.
std::optional<int64_t>
getPointerBaseOffsetUpperBound(Value value, int64_t block,
                               std::optional<unsigned> axis) {
  if (isLaneUniform(value, axis))
    return 0;
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto add = dyn_cast<triton::AddPtrOp>(def)) {
    auto base = getPointerBaseOffsetUpperBound(add.getPtr(), block, axis);
    auto offset = getBaseLaneOffsetUpperBound(add.getOffset(), block, axis);
    if (base && offset && *base <= block - 1 - *offset)
      return *base + *offset;
    return std::nullopt;
  }
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
    if (axis && *axis > static_cast<unsigned>(expand.getAxis()))
      --*axis;
    return getPointerBaseOffsetUpperBound(expand.getSrc(), block, axis);
  }
  if (isa<triton::BroadcastOp, ttg::ConvertLayoutOp>(def))
    return getPointerBaseOffsetUpperBound(def->getOperand(0), block, axis);
  return std::nullopt;
}

std::optional<int64_t>
getUnitStrideOffsetUpperBound(Value value, Value induction, int64_t block,
                              ArrayRef<Value> localRanks,
                              llvm::DenseSet<Value> &visited) {
  if (value == induction)
    return 0;
  if (!visited.insert(value).second)
    return std::nullopt;

  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  StringRef name = def->getName().getStringRef();
  bool isAddPtr = name == "tt.addptr";
  bool isAdd = name == "arith.addi";
  bool isUnary = name == "tt.splat" || name == "tt.broadcast" ||
                 name == "tt.expand_dims" || name == "arith.extsi" ||
                 name == "arith.extui" || name == "arith.index_cast" ||
                 name == "ttg.convert_layout";
  if (!isAddPtr && !isAdd && !isUnary)
    return std::nullopt;

  std::optional<unsigned> dependentIndex;
  for (auto [index, operand] : llvm::enumerate(def->getOperands())) {
    llvm::DenseSet<Value> dependencyVisited;
    if (!dependsOnAny(operand, ArrayRef<Value>{induction}, dependencyVisited))
      continue;
    if (dependentIndex)
      return std::nullopt;
    dependentIndex = index;
  }
  if (!dependentIndex || (isUnary && def->getNumOperands() != 1))
    return std::nullopt;
  int64_t addedOffset = 0;
  if (isAdd || (isAddPtr && *dependentIndex == 0)) {
    auto bound =
        getLaneOffsetUpperBound(def->getOperand(1 - *dependentIndex), block);
    if (!bound)
      return std::nullopt;
    addedOffset = *bound;
  }
  if (isAddPtr) {
    if (*dependentIndex == 1) {
      auto loop = dyn_cast<scf::ForOp>(
          cast<BlockArgument>(induction).getOwner()->getParentOp());
      llvm::DenseSet<Value> invariantVisited;
      if (!loop ||
          !isLoopInvariantBase(def->getOperand(0), loop, invariantVisited))
        return std::nullopt;
      auto projection = getLaneProjection(def->getOperand(1));
      auto axis = projection.known ? projection.axis : std::nullopt;
      auto baseOffset =
          getPointerBaseOffsetUpperBound(def->getOperand(0), block, axis);
      if (!baseOffset)
        return std::nullopt;
      addedOffset = *baseOffset;
    }
    llvm::DenseSet<Value> rankVisited;
    if (dependsOnAny(def->getOperand(1 - *dependentIndex), localRanks,
                     rankVisited))
      return std::nullopt;
  }
  auto bound = getUnitStrideOffsetUpperBound(
      def->getOperand(*dependentIndex), induction, block, localRanks, visited);
  // Subtraction checks both the tile limit and signed addition overflow.
  if (!bound || *bound > block - 1 - addedOffset)
    return std::nullopt;
  return *bound + addedOffset;
}

bool isCanonicalUnitStrideAddress(Value value, Value induction, int64_t block,
                                  ArrayRef<Value> localRanks,
                                  llvm::DenseSet<Value> &visited) {
  return getUnitStrideOffsetUpperBound(value, induction, block, localRanks,
                                       visited)
      .has_value();
}

LogicalResult validateCollectivePlacement(triton::ReduceOp reduce, int axis) {
  auto function = reduce->getParentOfType<triton::FuncOp>();
  if (!function ||
      !triton::isKernel(cast<FunctionOpInterface>(function.getOperation())))
    return reduce.emitError(
        "multi_cta reduce must be contained in a kernel tt.func");
  Block *entry = &function.getBody().front();
  if (reduce->getBlock() == entry)
    return success();

  auto loop = dyn_cast_or_null<scf::ForOp>(reduce->getBlock()->getParentOp());
  if (!loop || loop->getBlock() != entry)
    return reduce.emitError(
        "multi_cta reduce must be in the function entry block or one directly "
        "nested scf.for");
  for (Value bound :
       {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}) {
    if (ttng::mayDependOnClusterRank(bound, axis))
      return reduce.emitError(
          "multi_cta loop bounds and step must not depend on cluster rank");
  }
  return success();
}

bool hasRankZeroGuard(Value value, ArrayRef<Value> localRanks,
                      llvm::DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  if (auto compare = value.getDefiningOp<arith::CmpIOp>()) {
    if (compare.getPredicate() != arith::CmpIPredicate::eq)
      return false;
    return (llvm::is_contained(localRanks, compare.getLhs()) &&
            isZeroInteger(compare.getRhs())) ||
           (llvm::is_contained(localRanks, compare.getRhs()) &&
            isZeroInteger(compare.getLhs()));
  }
  if (auto conjunction = value.getDefiningOp<arith::AndIOp>())
    return hasRankZeroGuard(conjunction.getLhs(), localRanks, visited) ||
           hasRankZeroGuard(conjunction.getRhs(), localRanks, visited);
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    return hasRankZeroGuard(splat.getSrc(), localRanks, visited);
  return false;
}

// Trace address-only views to catch obvious aliasing within one SSA buffer.
// Runtime aliasing between distinct kernel arguments, and disjoint row/base
// slices supplied by the caller, remain part of the kernel's ownership
// contract.
Value getBufferRoot(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    StringRef name = def->getName().getStringRef();
    if (name != "tt.addptr" && name != "tt.splat" && name != "tt.broadcast" &&
        name != "tt.expand_dims" && name != "ttg.convert_layout" &&
        name != "tt.make_tensor_descriptor")
      break;
    value = def->getOperand(0);
  }
  return value;
}

LogicalResult validateMemorySafety(triton::ReduceOp reduce, int axis,
                                   int size) {
  auto function = reduce->getParentOfType<triton::FuncOp>();
  SmallVector<Value> localRanks;
  function.walk([&](triton::nvgpu::ClusterCTAIdOp rank) {
    localRanks.push_back(rank.getResult());
  });
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
    if (block && step && *block <= std::numeric_limits<int64_t>::max() / size &&
        *step == *block * size)
      shardLoops.push_back(loop);
  });

  llvm::DenseSet<Value> shardedRoots;
  SmallVector<std::pair<Value, Operation *>> rankZeroStores;
  WalkResult memoryResult = function.walk([&](Operation *op) {
    if (isa<triton::AtomicRMWOp, triton::AtomicCASOp,
            triton::DescriptorReduceOp, triton::DescriptorScatterOp>(op)) {
      op->emitError(
          "multi_cta does not support atomic or reducing memory writes in v1");
      return WalkResult::interrupt();
    }

    if (auto store = dyn_cast<triton::StoreOp>(op)) {
      auto loop = store->getParentOfType<scf::ForOp>();
      if (!loop || !llvm::is_contained(shardLoops, loop)) {
        llvm::DenseSet<Value> visited;
        if (hasRankZeroGuard(store.getMask(), localRanks, visited)) {
          rankZeroStores.emplace_back(getBufferRoot(store.getPtr()), op);
          return WalkResult::advance();
        }
        store.emitError("multi_cta stores must be rank-strided or guarded by "
                        "local_rank == 0");
        return WalkResult::interrupt();
      }
      llvm::DenseSet<Value> visited;
      auto block = getRankBlock(loop.getLowerBound(), localRanks);
      if (!block) {
        store.emitError("multi_cta rank-strided loop lost its validated block");
        return WalkResult::interrupt();
      }
      if (!isCanonicalUnitStrideAddress(store.getPtr(), loop.getInductionVar(),
                                        *block, localRanks, visited)) {
        store.emitError(
            "multi_cta store address must be a unit-stride projection of the "
            "canonical rank-strided loop induction variable");
        return WalkResult::interrupt();
      }
      shardedRoots.insert(getBufferRoot(store.getPtr()));
      return WalkResult::advance();
    }
    if (auto store = dyn_cast<triton::DescriptorStoreOp>(op)) {
      if (store.getReduceKind() != triton::DescriptorReduceKind::NONE) {
        store.emitError(
            "multi_cta does not support reducing descriptor stores in v1");
        return WalkResult::interrupt();
      }
      auto sourceType = dyn_cast<RankedTensorType>(store.getSrc().getType());
      llvm::DenseSet<Value> descriptorVisited;
      if (dependsOnAny(store.getDesc(), localRanks, descriptorVisited)) {
        store.emitError(
            "multi_cta descriptor store base must not depend on local rank");
        return WalkResult::interrupt();
      }
      bool hasDisjointRankDimension = sourceType && sourceType.hasStaticShape();
      unsigned rankDependentDimensions = 0;
      if (sourceType) {
        for (auto [dim, index] : llvm::enumerate(store.getIndices())) {
          llvm::DenseSet<Value> visited;
          if (!dependsOnAny(index, localRanks, visited))
            continue;
          ++rankDependentDimensions;
          auto block = getRankBlock(index, localRanks);
          if (!block || dim >= static_cast<unsigned>(sourceType.getRank()) ||
              *block < sourceType.getShape()[dim])
            hasDisjointRankDimension = false;
        }
      }
      hasDisjointRankDimension &= rankDependentDimensions == 1;
      if (!hasDisjointRankDimension) {
        store.emitError(
            "multi_cta descriptor store requires exactly one rank * block "
            "index whose block covers the stored tile extent");
        return WalkResult::interrupt();
      }
      shardedRoots.insert(getBufferRoot(store.getDesc()));
      return WalkResult::advance();
    }
    // Formation runs before its own remote exchanges are introduced. Remote
    // stores/copies and every other unproven write must fail closed here.
    // Only these two operations have intrinsically CTA-local write semantics.
    if (isa<ttg::LocalAllocOp, ttg::LocalStoreOp>(op))
      return WalkResult::advance();
    // Walk all leaves of containers; the function itself has no effect model.
    if (op == function.getOperation() ||
        op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects) {
      op->emitError("multi_cta cannot prove memory effects for this operation");
      return WalkResult::interrupt();
    }
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    if (llvm::any_of(instances, [](const auto &effect) {
          return isa<MemoryEffects::Write>(effect.getEffect());
        })) {
      op->emitError(
          "multi_cta memory writes require a proven ownership pattern");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (memoryResult.wasInterrupted())
    return failure();
  for (auto [root, store] : rankZeroStores) {
    if (shardedRoots.contains(root))
      return store->emitError(
          "multi_cta rank-zero side stores must not share a buffer with "
          "rank-sharded stores");
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
    if (autoWarpSpecialization) {
      module.emitError(
          "multi_cta does not support automatic warp specialization in v1");
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
      if (!function ||
          failed(validateCollectivePlacement(reduce, cluster->axis)))
        return signalPassFailure();
      if (validatedFunctions.insert(function).second &&
          failed(validateFunction(function, cluster->axis, cluster->size)))
        return signalPassFailure();
      if (failed(validateMemorySafety(reduce, cluster->axis, cluster->size)))
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
