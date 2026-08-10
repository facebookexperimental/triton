// Backend-only lowering for clustered DynamicPersistent1DScheduler claims.
//
// The frontend intentionally emits the ordinary single-CTA shape:
//
//   tile = program_id(0)
//   while tile < num_tiles:
//     ...
//     tile = atomic_add(counter, 1)
//
// The prepare pass recognizes that canonical shape in a required multi-CTA
// kernel, linearizes the initial physical CTA coordinate, and marks the atomic.
// After warp specialization has assigned and materialized the run-once owner,
// the late materialization pass turns the marked atomic into one cluster-leader
// reservation of K consecutive PIDs and distributes the base PID through DSM.
// The same late pass handles kernels that are not warp specialized.
// The public scheduler and its TTIR remain unchanged.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include <numeric>

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUATOMICTILESCHEDULERPREPAREPASS
#define GEN_PASS_DEF_TRITONNVIDIAGPUATOMICTILESCHEDULERMATERIALIZEPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace nvgpu = mlir::triton::nvgpu;

namespace {

constexpr StringLiteral kClusterClaimAttr = "ttng.cluster_atomic_tile_claim";
constexpr StringLiteral kClusterSizeAttr =
    "ttng.cluster_atomic_tile_cluster_size";

static LogicalResult materializeClusterAtomicTileScheduler(Operation *root);

struct PreparedClaim {
  scf::WhileOp loop;
  tt::AtomicRMWOp atomic;
  unsigned carriedIndex;
};

static FailureOr<SmallVector<int, 3>> getExplicitClusterDims(ModuleOp mod) {
  if (ttg::TritonGPUDialect::getNumCTAs(mod) != 1)
    return failure();
  auto dims = ttg::TritonGPUDialect::getClusterDims(mod);
  if (dims.size() != 3)
    return failure();
  return SmallVector<int, 3>(dims.begin(), dims.end());
}

static int getClusterSize(ArrayRef<int> dims) {
  return dims[0] * dims[1] * dims[2];
}

static bool hasTwoCTADot(scf::WhileOp loop) {
  bool found = false;
  loop.getAfter().walk([&](tt::DotOp dot) {
    if (dot.getTwoCtas())
      found = true;
  });
  return found;
}

static bool isConstOne(Value value) {
  return value && matchPattern(value, m_One());
}

static bool reachesYield(Value value, scf::YieldOp yield) {
  SmallVector<Value> worklist{value};
  DenseSet<Value> visited;
  visited.insert(value);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (Operation *user : current.getUsers()) {
      if (user == yield)
        return true;
      for (Value result : user->getResults()) {
        if (visited.insert(result).second)
          worklist.push_back(result);
      }
    }
  }
  return false;
}

static FailureOr<unsigned> getDirectYieldIndex(tt::AtomicRMWOp atomic,
                                               scf::WhileOp loop) {
  if (!atomic.getResult().hasOneUse())
    return failure();
  OpOperand &use = *atomic.getResult().getUses().begin();
  auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
  if (!yield || yield != loop.getYieldOp())
    return failure();
  return use.getOperandNumber();
}

// Return a conservative divisor known to divide value, capped at target.  The
// clustered loop condition must be uniform, so proving the trip bound is a
// multiple of K is part of recognizing the scheduler rather than a runtime
// assumption.  This intentionally handles only the arithmetic shapes emitted
// by padded tile-count calculations.
static int64_t getKnownDivisor(Value value, int64_t target,
                               unsigned depth = 0) {
  if (depth > 32)
    return 1;
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return std::gcd<int64_t>(constant.value() % target, target);
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto func = dyn_cast_or_null<tt::FuncOp>(arg.getOwner()->getParentOp());
    if (!func || arg.getOwner() != &func.getBody().front())
      return 1;
    auto attr = func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                   "tt.divisibility");
    return attr ? std::gcd<int64_t>(attr.getInt(), target) : 1;
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    int64_t lhs = getKnownDivisor(mul.getLhs(), target, depth + 1);
    int64_t rhs = getKnownDivisor(mul.getRhs(), target, depth + 1);
    return std::gcd<int64_t>(lhs * rhs, target);
  }
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return std::gcd(getKnownDivisor(add.getLhs(), target, depth + 1),
                    getKnownDivisor(add.getRhs(), target, depth + 1));
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return std::gcd(getKnownDivisor(sub.getLhs(), target, depth + 1),
                    getKnownDivisor(sub.getRhs(), target, depth + 1));
  if (auto select = value.getDefiningOp<arith::SelectOp>())
    return std::gcd(getKnownDivisor(select.getTrueValue(), target, depth + 1),
                    getKnownDivisor(select.getFalseValue(), target, depth + 1));
  return 1;
}

static bool isClusterUniform(Value value, scf::WhileOp loop,
                             unsigned depth = 0) {
  if (depth > 32)
    return false;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto func = dyn_cast_or_null<tt::FuncOp>(arg.getOwner()->getParentOp());
    return func && arg.getOwner() == &func.getBody().front();
  }
  Operation *def = value.getDefiningOp();
  if (!def || loop->isAncestor(def) || isa<tt::GetProgramIdOp>(def) ||
      isa<nvgpu::ClusterCTAIdOp>(def))
    return false;
  if (isa<tt::GetNumProgramsOp>(def))
    return true;
  if (def->getDialect()->getNamespace() !=
      arith::ArithDialect::getDialectNamespace())
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return isClusterUniform(operand, loop, depth + 1);
  });
}

static LogicalResult validateCanonicalClaim(scf::WhileOp loop,
                                            tt::AtomicRMWOp atomic,
                                            unsigned carriedIndex,
                                            int clusterSize) {
  if (atomic->getBlock() != loop.getAfterBody())
    return atomic.emitError(
        "clustered atomic tile claim must be directly in the persistent "
        "scf.while after-region");
  if (atomic.getAtomicRmwOp() != tt::RMWOp::ADD ||
      !atomic.getResult().getType().isInteger(32) ||
      isa<RankedTensorType>(atomic.getPtr().getType()) ||
      !isConstOne(atomic.getVal()) ||
      (atomic.getMask() && !isConstOne(atomic.getMask())))
    return atomic.emitError(
        "clustered dynamic scheduling requires a scalar i32 atomic_add of "
        "constant 1 with a true mask");
  if (atomic.getScope() != tt::MemSyncScope::GPU &&
      atomic.getScope() != tt::MemSyncScope::SYSTEM)
    return atomic.emitError(
        "clustered dynamic scheduler atomic must have GPU or system scope");

  auto counterArg = dyn_cast<BlockArgument>(atomic.getPtr());
  auto func = atomic->getParentOfType<tt::FuncOp>();
  if (!counterArg || !func || counterArg.getOwner() != &func.getBody().front())
    return atomic.emitError(
        "clustered dynamic scheduler counter must be a uniform kernel "
        "argument");

  if (carriedIndex >= loop.getInits().size() ||
      carriedIndex >= loop.getBeforeArguments().size() ||
      carriedIndex >= loop.getAfterArguments().size())
    return atomic.emitError("invalid clustered scheduler carry index");

  auto seed = loop.getInits()[carriedIndex].getDefiningOp<tt::GetProgramIdOp>();
  if (!seed || seed.getAxisAsInt() != 0)
    return atomic.emitError(
        "clustered dynamic scheduler must be initialized directly from "
        "program_id(0)");

  auto condition = loop.getConditionOp();
  if (carriedIndex >= condition.getArgs().size() ||
      condition.getArgs()[carriedIndex] !=
          loop.getBeforeArguments()[carriedIndex])
    return atomic.emitError(
        "clustered scheduler tile must be forwarded directly through "
        "scf.condition");

  auto cmp = condition.getCondition().getDefiningOp<arith::CmpIOp>();
  Value tile = loop.getBeforeArguments()[carriedIndex];
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
      cmp.getLhs() != tile)
    return atomic.emitError(
        "clustered scheduler condition must be the canonical "
        "tile_id < num_tiles comparison");
  if (!isClusterUniform(cmp.getRhs(), loop))
    return cmp.emitError(
        "clustered scheduler num_tiles must be cluster-uniform");
  if (getKnownDivisor(cmp.getRhs(), clusterSize) != clusterSize)
    return cmp.emitError("cannot prove num_tiles is divisible by physical "
                         "cluster size ")
           << clusterSize
           << "; pad the 1-D scheduled tile space to a full cluster";
  return success();
}

// Form an abstract 1-D program id whose K consecutive values correspond to one
// physical cluster.  PTX cluster ranks are X-major, so use the same order for
// both cluster coordinates and the local rank.
static Value createLinearClusterProgramId(OpBuilder &builder, Location loc,
                                          Value pidX,
                                          ArrayRef<int> clusterDims) {
  SmallVector<Value, 3> pid(3), numPrograms(3), clusterCoord(3), numClusters(3);
  pid[0] = pidX;
  for (int dim = 1; dim < 3; ++dim)
    pid[dim] = tt::GetProgramIdOp::create(builder, loc, dim);
  for (int dim = 0; dim < 3; ++dim) {
    numPrograms[dim] = tt::GetNumProgramsOp::create(builder, loc, dim);
    Value dimSize =
        arith::ConstantIntOp::create(builder, loc, clusterDims[dim], 32);
    clusterCoord[dim] = arith::DivUIOp::create(builder, loc, pid[dim], dimSize);
    numClusters[dim] =
        arith::DivUIOp::create(builder, loc, numPrograms[dim], dimSize);
  }

  Value clusterLinear = clusterCoord[2];
  clusterLinear = arith::AddIOp::create(
      builder, loc,
      arith::MulIOp::create(builder, loc, clusterLinear, numClusters[1]),
      clusterCoord[1]);
  clusterLinear = arith::AddIOp::create(
      builder, loc,
      arith::MulIOp::create(builder, loc, clusterLinear, numClusters[0]),
      clusterCoord[0]);

  Value size = arith::ConstantIntOp::create(builder, loc,
                                            getClusterSize(clusterDims), 32);
  Value rank =
      nvgpu::ClusterCTAIdOp::create(builder, loc, builder.getI32Type());
  return arith::AddIOp::create(
      builder, loc, arith::MulIOp::create(builder, loc, clusterLinear, size),
      rank);
}

class AtomicTileSchedulerPreparePass
    : public impl::TritonNvidiaGPUAtomicTileSchedulerPreparePassBase<
          AtomicTileSchedulerPreparePass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!ttng::is2CTA(mod))
      return;

    auto clusterDims = getExplicitClusterDims(mod);
    if (failed(clusterDims)) {
      mod.emitError("clustered atomic tile scheduling supports only explicit "
                    "ctas_per_cga clusters with ttg.num-ctas == 1");
      return signalPassFailure();
    }
    int clusterSize = getClusterSize(*clusterDims);
    if (clusterSize <= 1)
      return;

    SmallVector<PreparedClaim> claims;
    bool invalid = false;
    mod.walk([&](scf::WhileOp loop) {
      if (invalid || !hasTwoCTADot(loop))
        return;

      SmallVector<tt::AtomicRMWOp> carriedAtomics;
      loop.getAfterBody()->walk([&](tt::AtomicRMWOp atomic) {
        if (atomic->getParentOfType<scf::WhileOp>() != loop)
          return;
        if (reachesYield(atomic.getResult(), loop.getYieldOp()))
          carriedAtomics.push_back(atomic);
      });
      if (carriedAtomics.empty())
        return; // Static/non-atomic persistent schedule.
      if (carriedAtomics.size() != 1) {
        loop.emitError("clustered dynamic scheduler requires exactly one "
                       "direct loop-carried atomic tile claim");
        invalid = true;
        return;
      }
      tt::AtomicRMWOp atomic = carriedAtomics.front();
      auto index = getDirectYieldIndex(atomic, loop);
      if (failed(index)) {
        atomic.emitError("clustered dynamic scheduler atomic must be forwarded "
                         "directly through scf.yield");
        invalid = true;
        return;
      }
      if (failed(validateCanonicalClaim(loop, atomic, *index, clusterSize))) {
        invalid = true;
        return;
      }
      claims.push_back({loop, atomic, *index});
    });
    if (invalid)
      return signalPassFailure();

    for (PreparedClaim claim : claims) {
      OpBuilder builder(claim.loop);
      Location loc = claim.loop.getLoc();
      Value seed = createLinearClusterProgramId(
          builder, loc, claim.loop.getInits()[claim.carriedIndex],
          *clusterDims);
      claim.loop->setOperand(claim.carriedIndex, seed);
      claim.atomic->setAttr(kClusterClaimAttr, builder.getUnitAttr());
      claim.atomic->setAttr(kClusterSizeAttr,
                            builder.getI32IntegerAttr(clusterSize));
    }
  }
};

static Value mapBarrierToRank(OpBuilder &builder, Location loc, Value barrier,
                              Value rank) {
  auto localTy = cast<ttg::MemDescType>(barrier.getType());
  auto remoteTy = ttg::MemDescType::get(
      localTy.getShape(), localTy.getElementType(), localTy.getEncoding(),
      SharedClusterMemorySpaceAttr::get(builder.getContext()),
      localTy.getMutableMemory(), localTy.getAllocShape());
  return MapToRemoteBufferOp::create(builder, loc, remoteTy, barrier, rank);
}

static Value createPidSlot(OpBuilder &builder, Location loc) {
  MLIRContext *ctx = builder.getContext();
  auto cga = ttg::CGAEncodingAttr::get1CTALayout(ctx, /*rank=*/1);
  auto encoding = ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, cga);
  auto type = ttg::MemDescType::get({1}, builder.getI32Type(), encoding,
                                    ttg::SharedMemorySpaceAttr::get(ctx),
                                    /*mutableMemory=*/true);
  return ttg::LocalAllocOp::create(builder, loc, type);
}

// A marked claim may already live inside an AutoWS owner partition. Keep its
// DSM state at function lifetime so every CTA initializes the same captures
// before the partition starts using them.
static Value createPersistentBarrierAlloc(ImplicitLocOpBuilder &builder,
                                          int arriveCount) {
  Value alloc = createScalarAlloc(builder, builder.getI64Type(), 1);
  Value barrier = createSingleBufferView(builder, alloc, 0);
  InitBarrierOp::create(builder, barrier, arriveCount);
  return alloc;
}

// Partition regions are isolated from above. Late materialization therefore
// has to extend the already-created warp_specialize capture list explicitly.
static Value captureInWarpPartition(Value value, Operation *user) {
  auto wsOp = user->getParentOfType<ttg::WarpSpecializeOp>();
  if (!wsOp || wsOp.getDefaultRegion().isAncestor(user->getParentRegion()))
    return value;

  Value captured;
  auto partOp = wsOp.getPartitionOp();
  partOp->insertOperands(partOp.getNumOperands(), value);
  for (Region *region : wsOp.getPartitionRegions()) {
    BlockArgument arg = region->addArgument(value.getType(), value.getLoc());
    if (region->isAncestor(user->getParentRegion()))
      captured = arg;
  }
  assert(captured && "operation not found in a warp partition region");
  return captured;
}

static LogicalResult materializeClaim(tt::AtomicRMWOp atomic, int clusterSize) {
  auto loop = atomic->getParentOfType<scf::WhileOp>();
  if (!loop)
    return atomic.emitError(
        "prepared clustered atomic tile claim is not inside scf.while");

  Location loc = atomic.getLoc();
  Attribute taskIds = atomic->getAttr("async_task_id");
  auto tagOwner = [&](Operation *op) {
    if (taskIds)
      op->setAttr("async_task_id", taskIds);
  };
  auto func = atomic->getParentOfType<tt::FuncOp>();
  int numWarps = ttg::lookupNumWarps(atomic);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(
      atomic->getParentOfType<ModuleOp>());
  OpBuilder loopBuilder(loop);
  Value zero = arith::ConstantIntOp::create(loopBuilder, loc, 0, 32);
  scf::WhileOp newLoop = replaceWhileOpWithNewSignature(
      loopBuilder, loop, {zero}, {loopBuilder.getI32Type()});
  loop->erase();

  ImplicitLocOpBuilder allocBuilder(loc, func);
  allocBuilder.setInsertionPointToStart(&func.getBody().front());
  Value pidSlot = createPidSlot(allocBuilder, loc);
  Value fullAlloc =
      createPersistentBarrierAlloc(allocBuilder, /*arriveCount=*/1);
  Value emptyAlloc =
      createPersistentBarrierAlloc(allocBuilder, /*arriveCount=*/clusterSize);
  Value fullBar = createSingleBufferView(allocBuilder, fullAlloc, 0);
  Value emptyBar = createSingleBufferView(allocBuilder, emptyAlloc, 0);
  pidSlot = captureInWarpPartition(pidSlot, atomic);
  fullBar = captureInWarpPartition(fullBar, atomic);
  emptyBar = captureInWarpPartition(emptyBar, atomic);

  Value phaseBefore = newLoop.getBeforeArguments().back();
  Value phaseAfter = newLoop.getAfterArguments().back();
  auto condition = newLoop.getConditionOp();
  condition.getArgsMutable().append(phaseBefore);

  OpBuilder builder(atomic);
  Type i32 = builder.getI32Type();
  Value rank = nvgpu::ClusterCTAIdOp::create(builder, loc, i32);
  Value zeroI32 = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value oneI32 = arith::ConstantIntOp::create(builder, loc, 1, 32);
  Value isLeader = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         rank, zeroI32);
  tagOwner(rank.getDefiningOp());
  tagOwner(zeroI32.getDefiningOp());
  tagOwner(oneI32.getDefiningOp());
  tagOwner(isLeader.getDefiningOp());

  // The first opposite-phase wait passes immediately.  Later iterations wait
  // until every CTA has consumed the previous PID before rank zero overwrites
  // any slot.
  Value emptyPhase = arith::XOrIOp::create(builder, loc, phaseAfter, oneI32);
  tagOwner(emptyPhase.getDefiningOp());
  auto emptyWait =
      WaitBarrierOp::create(builder, loc, emptyBar, emptyPhase, isLeader);
  emptyWait->setAttr("acquireCluster", builder.getUnitAttr());
  tagOwner(emptyWait);

  auto ifLeader = scf::IfOp::create(builder, loc, TypeRange{}, isLeader,
                                    /*withElseRegion=*/false);
  OpBuilder leaderBuilder = OpBuilder::atBlockBegin(ifLeader.thenBlock());
  Value increment =
      arith::ConstantIntOp::create(leaderBuilder, loc, clusterSize, 32);
  auto leaderAtomic = tt::AtomicRMWOp::create(
      leaderBuilder, loc, atomic.getResult().getType(), atomic.getAtomicRmwOp(),
      atomic.getPtr(), increment, atomic.getMask(), atomic.getSem(),
      atomic.getScope());

  auto valueEncoding =
      ttg::getDefaultBlockedEncoding(atomic.getContext(), {1}, numWarps,
                                     threadsPerWarp, ttg::lookupNumCTAs(func));
  auto valueType = RankedTensorType::get({1}, i32, valueEncoding);
  Value baseTensor =
      tt::SplatOp::create(leaderBuilder, loc, valueType, leaderAtomic);

  // Rank zero writes and signals its own slot locally. Under AutoWS the normal
  // arrive rendezvous is scoped to the owner partition.
  ttg::LocalStoreOp::create(leaderBuilder, loc, baseTensor, pidSlot);
  ArriveBarrierOp::create(leaderBuilder, loc, fullBar, /*count=*/1);

  // Every other CTA receives the same base through generic-proxy DSM.  The
  // release arrive and acquire wait order the store/load, so no async-proxy
  // fence is required.
  for (int target = 1; target < clusterSize; ++target) {
    Value targetRank =
        arith::ConstantIntOp::create(leaderBuilder, loc, target, 32);
    ttg::RemoteShmemStoreOp::create(leaderBuilder, loc, baseTensor, pidSlot,
                                    targetRank);
    Value remoteFull =
        mapBarrierToRank(leaderBuilder, loc, fullBar, targetRank);
    ArriveBarrierOp::create(leaderBuilder, loc, remoteFull, /*count=*/1);
  }
  ifLeader.walk(tagOwner);

  builder.setInsertionPointAfter(ifLeader);
  tagOwner(ifLeader);
  auto fullWait = WaitBarrierOp::create(builder, loc, fullBar, phaseAfter);
  fullWait->setAttr("acquireCluster", builder.getUnitAttr());
  tagOwner(fullWait);
  Value loaded =
      ttg::LocalLoadOp::create(builder, loc, valueType, pidSlot, Value());
  Value base = tt::UnsplatOp::create(builder, loc, i32, loaded);
  Value pid = arith::AddIOp::create(builder, loc, base, rank);
  tagOwner(loaded.getDefiningOp());
  tagOwner(base.getDefiningOp());
  tagOwner(pid.getDefiningOp());

  // Signal consumption to rank zero.  The remote mbarrier release is ordered
  // after the generic-proxy load and rank zero's next opposite-phase wait is
  // the matching acquire.
  Value remoteEmpty = mapBarrierToRank(builder, loc, emptyBar, zeroI32);
  tagOwner(remoteEmpty.getDefiningOp());
  auto emptyArrive =
      ArriveBarrierOp::create(builder, loc, remoteEmpty, /*count=*/1);
  tagOwner(emptyArrive);

  atomic.getResult().replaceAllUsesWith(pid);
  atomic.erase();

  auto yield = newLoop.getYieldOp();
  OpBuilder yieldBuilder(yield);
  Attribute yieldTaskIds = yield->getAttr("async_task_id");
  Value onePhase =
      arith::ConstantIntOp::create(yieldBuilder, yield.getLoc(), 1, 32);
  Value toggled =
      arith::XOrIOp::create(yieldBuilder, yield.getLoc(), phaseAfter, onePhase);
  if (yieldTaskIds) {
    onePhase.getDefiningOp()->setAttr("async_task_id", yieldTaskIds);
    toggled.getDefiningOp()->setAttr("async_task_id", yieldTaskIds);
  }
  yield.getResultsMutable().append(toggled);
  return success();
}

class AtomicTileSchedulerMaterializePass
    : public impl::TritonNvidiaGPUAtomicTileSchedulerMaterializePassBase<
          AtomicTileSchedulerMaterializePass> {
public:
  void runOnOperation() override {
    if (failed(materializeClusterAtomicTileScheduler(getOperation())))
      return signalPassFailure();
  }
};

static LogicalResult materializeClusterAtomicTileScheduler(Operation *root) {
  SmallVector<tt::AtomicRMWOp> claims;
  root->walk([&](tt::AtomicRMWOp atomic) {
    if (atomic->hasAttr(kClusterClaimAttr))
      claims.push_back(atomic);
  });
  if (claims.empty())
    return success();

  ModuleOp mod = dyn_cast<ModuleOp>(root);
  if (!mod)
    mod = root->getParentOfType<ModuleOp>();
  auto clusterDims = getExplicitClusterDims(mod);
  if (failed(clusterDims))
    return mod.emitError("prepared clustered atomic tile claim requires "
                         "explicit ctas_per_cga semantics");
  int clusterSize = getClusterSize(*clusterDims);

  for (tt::AtomicRMWOp atomic : claims) {
    auto preparedSize = atomic->getAttrOfType<IntegerAttr>(kClusterSizeAttr);
    if (!preparedSize || preparedSize.getInt() != clusterSize)
      return atomic.emitError(
          "prepared cluster size does not match module cluster dimensions");
    if (failed(materializeClaim(atomic, clusterSize)))
      return failure();
  }
  return success();
}

} // namespace

} // namespace mlir::triton::nvidia_gpu
