#include "CodePartitionUtility.h"
#include "WarpSpecializationPipeline.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <optional>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-tma-store-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static void copyLoopScheduleAttrs(Operation *from, Operation *to) {
  if (auto attr = from->getAttr(tt::kLoopStageAttrName))
    to->setAttr(tt::kLoopStageAttrName, attr);
  if (auto attr = from->getAttr(tt::kLoopClusterAttrName))
    to->setAttr(tt::kLoopClusterAttrName, attr);
}

void doTMAStoreLowering(triton::FuncOp &funcOp) {
  SmallVector<tt::DescriptorStoreOp> storeOps;
  funcOp.walk([&](tt::DescriptorStoreOp op) {
    // Skip stores with non-trivial reduce semantics.
    if (op.getReduceKind() != tt::DescriptorReduceKind::NONE)
      return;
    storeOps.push_back(op);
  });

  if (storeOps.empty())
    return;

  LDBG("Lowering " << storeOps.size() << " DescriptorStoreOp(s)");

  MLIRContext *ctx = funcOp.getContext();
  Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);

  for (auto storeOp : storeOps) {
    auto loc = storeOp.getLoc();
    auto partitionIds = getWSPartitionIds(storeOp);

    OpBuilderWithPartitionIds builder(storeOp);
    builder.setInsertionPoint(storeOp);

    auto src = storeOp.getSrc();
    auto desc = storeOp.getDesc();
    auto tensorType = src.getType();

    // Compute shared encoding from the descriptor.
    auto encoding = ttng::getEncodingFromDescriptor(storeOp, tensorType, desc);
    ttg::MemDescType memDescType = ttg::MemDescType::get(
        tensorType.getShape(), tensorType.getElementType(), encoding,
        sharedMemorySpace, /*mutableMemory=*/true);

    // Derive a NameLoc for the staging alloc from the descriptor's name.
    auto allocLoc = loc;
    Location descLoc = desc.getLoc();
    if (auto *defOp = desc.getDefiningOp())
      descLoc = defOp->getLoc();
    // Walk through CallSiteLoc/FusedLoc to find a NameLoc.
    auto findName = [](Location l, auto &self) -> StringRef {
      if (auto nameLoc = dyn_cast<NameLoc>(l))
        return nameLoc.getName().strref();
      if (auto callSiteLoc = dyn_cast<CallSiteLoc>(l))
        return self(callSiteLoc.getCallee(), self);
      if (auto fusedLoc = dyn_cast<FusedLoc>(l))
        for (Location sub : fusedLoc.getLocations()) {
          auto s = self(sub, self);
          if (!s.empty())
            return s;
        }
      return {};
    };
    auto descName = findName(descLoc, findName);
    if (!descName.empty()) {
      auto stagingName = (descName + "_staging").str();
      allocLoc = NameLoc::get(StringAttr::get(ctx, stagingName), loc);
    }
    auto alloc = builder.create<ttg::LocalAllocOp>(allocLoc, memDescType, src);

    // Async TMA copy from local (SMEM) to global, producing a token.
    auto tokenType = ttg::AsyncTokenType::get(ctx);
    auto tmaStore = builder.create<ttng::AsyncTMACopyLocalToGlobalOp>(
        loc, tokenType, desc, storeOp.getIndices(), alloc,
        tt::EvictionPolicy::NORMAL);
    copyLoopScheduleAttrs(storeOp, tmaStore);

    // Wait for this specific TMA store to finish reading from SMEM.
    auto waitOp = builder.create<ttng::TMAStoreTokenWaitOp>(
        loc, tmaStore.getToken(), ValueRange{}, ValueRange{}, ValueRange{},
        ValueRange{});
    copyLoopScheduleAttrs(storeOp, waitOp);

    storeOp.erase();
  }

  // Also lower DescriptorReduceOp → local_alloc + AsyncTMAReduceOp (with token)
  // + TMAStoreTokenWaitOp, matching the early TMA store pattern.
  SmallVector<tt::DescriptorReduceOp> reduceOps;
  funcOp.walk([&](tt::DescriptorReduceOp op) { reduceOps.push_back(op); });

  if (!reduceOps.empty())
    LDBG("Lowering " << reduceOps.size() << " DescriptorReduceOp(s)");

  for (auto reduceOp : reduceOps) {
    auto loc = reduceOp.getLoc();
    OpBuilderWithPartitionIds builder(reduceOp);
    builder.setInsertionPoint(reduceOp);

    auto src = reduceOp.getSrc();
    auto desc = reduceOp.getDesc();
    auto tensorType = src.getType();

    auto encoding = ttng::getEncodingFromDescriptor(reduceOp, tensorType, desc);
    ttg::MemDescType memDescType = ttg::MemDescType::get(
        tensorType.getShape(), tensorType.getElementType(), encoding,
        sharedMemorySpace, /*mutableMemory=*/true);

    // Derive a NameLoc for the staging alloc from the descriptor's name.
    auto allocLoc = loc;
    Location descLoc = desc.getLoc();
    if (auto *defOp = desc.getDefiningOp())
      descLoc = defOp->getLoc();
    auto findName = [](Location l, auto &self) -> StringRef {
      if (auto nameLoc = dyn_cast<NameLoc>(l))
        return nameLoc.getName().strref();
      if (auto callSiteLoc = dyn_cast<CallSiteLoc>(l))
        return self(callSiteLoc.getCallee(), self);
      if (auto fusedLoc = dyn_cast<FusedLoc>(l))
        for (Location sub : fusedLoc.getLocations()) {
          auto s = self(sub, self);
          if (!s.empty())
            return s;
        }
      return {};
    };
    auto descName = findName(descLoc, findName);
    if (!descName.empty()) {
      auto stagingName = (descName + "_reduce_staging").str();
      allocLoc = NameLoc::get(StringAttr::get(ctx, stagingName), loc);
    }
    auto alloc = builder.create<ttg::LocalAllocOp>(allocLoc, memDescType, src);

    auto tokenType = ttg::AsyncTokenType::get(ctx);
    auto tmaReduce = builder.create<ttng::AsyncTMAReduceOp>(
        loc, tokenType, reduceOp.getKind(), desc, reduceOp.getIndices(), alloc,
        tt::EvictionPolicy::NORMAL);
    copyLoopScheduleAttrs(reduceOp, tmaReduce);

    auto waitOp = builder.create<ttng::TMAStoreTokenWaitOp>(
        loc, tmaReduce.getToken(), ValueRange{}, ValueRange{}, ValueRange{},
        ValueRange{});
    copyLoopScheduleAttrs(reduceOp, waitOp);

    reduceOp.erase();
  }
}

// ---------------------------------------------------------------------------
// Standalone pass wrapper
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUWSTMASTORELOWERING
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

struct NVGPUWSTMAStoreLoweringPass
    : public impl::NVGPUWSTMAStoreLoweringBase<NVGPUWSTMAStoreLoweringPass> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!mod->hasAttr("ttg.early_tma_store_lowering"))
      return;
    mod->walk([&](triton::FuncOp funcOp) { doTMAStoreLowering(funcOp); });
  }
};

// ---------------------------------------------------------------------------

// Annotate TMA store waits with can_rotate_by_buffer_count
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTESTANNOTATETMASTOREWAITS
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

static constexpr const char *kCanRotateByBufferCount =
    "can_rotate_by_buffer_count";

static bool isTMAStoreLikeOp(Operation *op) {
  return isa<ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAReduceOp>(op);
}

static Value getTMAStoreSource(Operation *op) {
  if (auto copyOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return copyOp.getSrc();
  if (auto reduceOp = dyn_cast<ttng::AsyncTMAReduceOp>(op))
    return reduceOp.getSrc();
  return {};
}

// Trace the token back to the defining TMA store-like op
// (AsyncTMACopyLocalToGlobalOp or AsyncTMAReduceOp), handling both direct
// definitions and loop-carried block arguments. Returns the SMEM source
// buffer and the defining op.
static Operation *getDefiningTMAStoreOp(ttng::TMAStoreTokenWaitOp waitOp,
                                        Value &buffer) {
  Value token = waitOp.getToken();

  // Direct case: token defined by AsyncTMACopyLocalToGlobalOp.
  if (auto defOp = token.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>()) {
    buffer = defOp.getSrc();
    return defOp;
  }

  // Direct case: token defined by AsyncTMAReduceOp.
  if (auto defOp = token.getDefiningOp<ttng::AsyncTMAReduceOp>()) {
    buffer = defOp.getSrc();
    return defOp;
  }

  // Loop-carried case: token is a block argument of an scf.for body.
  if (auto blockArg = dyn_cast<BlockArgument>(token)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return nullptr;
    unsigned iterArgIdx = blockArg.getArgNumber() - 1;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yieldedVal = yieldOp.getOperand(iterArgIdx);
    if (auto defOp =
            yieldedVal.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>()) {
      buffer = defOp.getSrc();
      return defOp;
    }
    if (auto defOp = yieldedVal.getDefiningOp<ttng::AsyncTMAReduceOp>()) {
      buffer = defOp.getSrc();
      return defOp;
    }
  }

  return nullptr;
}

static bool samePureValue(Value lhs, Value rhs, unsigned depth = 0) {
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType() || depth > 8)
    return false;

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef)
    return false;
  if (lhsDef->getName() != rhsDef->getName())
    return false;
  if (cast<OpResult>(lhs).getResultNumber() !=
      cast<OpResult>(rhs).getResultNumber())
    return false;
  if (!isMemoryEffectFree(lhsDef) || !isMemoryEffectFree(rhsDef))
    return false;
  if (lhsDef->getNumRegions() || rhsDef->getNumRegions())
    return false;

  return OperationEquivalence::isEquivalentTo(
      lhsDef, rhsDef,
      [depth](Value lhsOperand, Value rhsOperand) {
        return success(samePureValue(lhsOperand, rhsOperand, depth + 1));
      },
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
}

static bool sameMemDescValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef)
    return false;
  if (lhsDef->getName() != rhsDef->getName())
    return false;
  if (!isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp,
           ttg::MemDescReinterpretOp>(lhsDef))
    return false;
  if (lhsDef->getNumOperands() != rhsDef->getNumOperands())
    return false;

  for (unsigned i = 0; i < lhsDef->getNumOperands(); ++i) {
    if (!samePureValue(lhsDef->getOperand(i), rhsDef->getOperand(i)))
      return false;
  }
  return true;
}

static Operation *
findLocalStoreWritingBuffer(scf::ForOp forOp, Value buffer,
                            const tt::CoarseSchedule &schedule) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    auto localStore = dyn_cast<ttg::LocalStoreOp>(&op);
    if (!localStore || !schedule.count(&op))
      continue;
    if (sameMemDescValue(localStore.getDst(), buffer))
      return &op;
  }
  return nullptr;
}

void doAnnotateTMAStoreWaits(triton::FuncOp &funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  // Use walk to find TMAStoreTokenWaitOp ops inside ForOp bodies, including
  // those nested inside SubtiledRegionOp regions.
  funcOp.walk([&](scf::ForOp forOp) {
    forOp.walk([&](ttng::TMAStoreTokenWaitOp waitOp) {
      Value buffer;
      auto *tmaOp = getDefiningTMAStoreOp(waitOp, buffer);
      if (!tmaOp)
        return;

      auto allocOp = buffer.getDefiningOp<ttg::LocalAllocOp>();
      if (!allocOp)
        return;

      // Only annotate buffers that have buffer.copy from the memory planner.
      // Buffers without buffer.copy were not planned and cannot be rotated.
      auto bufferCopy = allocOp->getAttrOfType<IntegerAttr>("buffer.copy");
      if (!bufferCopy)
        return;

      int k = bufferCopy.getInt();
      if (k <= 0)
        return;

      waitOp->setAttr(kCanRotateByBufferCount,
                      IntegerAttr::get(IntegerType::get(ctx, 32), k));
    });
  });
}

struct NVGPUTestAnnotateTMAStoreWaitsPass
    : public impl::NVGPUTestAnnotateTMAStoreWaitsBase<
          NVGPUTestAnnotateTMAStoreWaitsPass> {
  void runOnOperation() override {
    getOperation()->walk(
        [&](triton::FuncOp funcOp) { doAnnotateTMAStoreWaits(funcOp); });
  }
};

// ---------------------------------------------------------------------------
// Validate TMA store annotations (safety checks)
// ---------------------------------------------------------------------------

void doValidateTMAStoreAnnotations(triton::FuncOp &funcOp) {
  funcOp.walk([&](scf::ForOp forOp) {
    forOp.walk([&](ttng::TMAStoreTokenWaitOp waitOp) {
      if (!waitOp->hasAttr(kCanRotateByBufferCount))
        return;

      Value buffer;
      auto *tmaOp = getDefiningTMAStoreOp(waitOp, buffer);
      if (!tmaOp) {
        waitOp->removeAttr(kCanRotateByBufferCount);
        return;
      }

      auto allocOp = buffer.getDefiningOp<ttg::LocalAllocOp>();
      if (!allocOp) {
        waitOp->removeAttr(kCanRotateByBufferCount);
        return;
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Reschedule TMA store waits using the SWP CoarseSchedule
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTESTTMASTORETOKENWAITREORDER
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

static Operation *
findScheduledWaitBarrierBetween(Operation *producer, Operation *insertionTarget,
                                const tt::CoarseSchedule &schedule,
                                bool includeBarrierBeforeProducer) {
  auto findBefore = [&](Operation *op, Operation *stopOp) -> Operation * {
    for (auto revIt = Block::reverse_iterator(op->getIterator());
         revIt != op->getBlock()->rend(); ++revIt) {
      Operation *candidate = &*revIt;
      if (candidate == stopOp)
        return nullptr;
      if (isa<ttng::WaitBarrierOp>(candidate) && schedule.count(candidate))
        return candidate;
    }
    return nullptr;
  };

  if (producer->isBeforeInBlock(insertionTarget))
    return findBefore(insertionTarget, producer);

  if (!includeBarrierBeforeProducer)
    return nullptr;

  Operation *wraparoundTarget =
      insertionTarget->isBeforeInBlock(producer) ? insertionTarget : producer;
  for (auto revIt = Block::reverse_iterator(wraparoundTarget->getIterator());
       revIt != producer->getBlock()->rend(); ++revIt) {
    Operation *candidate = &*revIt;
    if (isTMAStoreLikeOp(candidate))
      return nullptr;
    if (isa<ttng::WaitBarrierOp>(candidate) && schedule.count(candidate))
      return candidate;
  }

  return nullptr;
}

// Best-effort reordering of annotated TMA store waits. Every case where a
// wait cannot be repositioned leaves the schedule unchanged for that wait and
// logs (see the LDBG sites below); there is no failure mode, so this returns
// void rather than a LogicalResult.
void doTMAStoreWaitReorder(triton::FuncOp &funcOp) {
  funcOp.walk([&](scf::ForOp forOp) {
    bool hasNestedFor = false;
    forOp.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
    if (hasNestedFor)
      return;

    // Deserialize the SWP schedule. If there is no schedule, create a basic
    // single-stage schedule so the reorder logic can still work.
    tt::CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp))) {
      schedule.setNumStages(1);
      auto cluster = schedule.clusters.newAtBack();
      for (auto &op : forOp.getBody()->without_terminator())
        schedule.insert(&op, 0, cluster);
    }

    // Bail out if the loop body contains any allocation ops. Reordering
    // waits in such loops would serialize a multi-stage schedule that
    // covers only a subset of the body ops, causing the pipeliner to fail
    // on the unscheduled allocations.
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op))
        return;
    }

    // Collect annotated TMA store waits that are direct children of this
    // loop and whose defining TMA store is in the same loop.
    SmallVector<ttng::TMAStoreTokenWaitOp> waits;
    for (auto &op : forOp.getBody()->without_terminator()) {
      auto waitOp = dyn_cast<ttng::TMAStoreTokenWaitOp>(&op);
      if (!waitOp || !waitOp->hasAttr(kCanRotateByBufferCount))
        continue;
      Value buffer;
      auto *tmaStore = getDefiningTMAStoreOp(waitOp, buffer);
      if (!tmaStore || tmaStore->getParentOp() != forOp)
        continue;
      waits.push_back(waitOp);
    }
    if (waits.empty())
      return;

    int numTMAStores = 0;
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (isTMAStoreLikeOp(&op))
        ++numTMAStores;
    }

    bool changed = false;
    for (auto waitOp : waits) {
      auto attr = waitOp->getAttrOfType<IntegerAttr>(kCanRotateByBufferCount);
      if (!attr)
        continue;
      int k = attr.getInt();

      // Find the defining TMA store op.
      Value buffer;
      auto *tmaStore = getDefiningTMAStoreOp(waitOp, buffer);
      if (!tmaStore)
        continue;

      // The defining op must be in the schedule for the LinearizedIterator.
      if (!schedule.count(tmaStore))
        continue;

      // Walk the linearized schedule from the TMA store, counting K
      // TMA store-like ops. The wait must be placed before the K-th op to
      // ensure the buffer slot is not overwritten.
      auto it = schedule.linearized(forOp, tmaStore);
      it.setMaxStages(schedule.getNumStages() + k);

      // Skip past the starting TMA store itself.
      ++it;

      Operation *insertionTarget = nullptr;
      int targetStage = 0;
      int storeCount = 0;

      while (!it.isEnd()) {
        Operation *op = *it;
        int stageAtOp = it.currStage();
        ++it;
        if (isTMAStoreLikeOp(op)) {
          ++storeCount;
          if (storeCount == k) {
            insertionTarget = op;
            targetStage = stageAtOp;
            break;
          }
        }
      }

      if (insertionTarget) {
        Operation *targetTMAStore = insertionTarget;
        int numPrevTMAStores = 0;
        for (auto &op : forOp.getBody()->without_terminator()) {
          if (&op == tmaStore)
            break;
          if (isTMAStoreLikeOp(&op))
            ++numPrevTMAStores;
        }

        Value targetBuffer = getTMAStoreSource(targetTMAStore);
        Operation *targetWriter =
            targetBuffer
                ? findLocalStoreWritingBuffer(forOp, targetBuffer, schedule)
                : nullptr;
        if (targetWriter) {
          insertionTarget = targetWriter;
        } else {
          // If the buffer is updated by a different partition, the TMA store
          // must be guarded by that partition's wait_barrier. Reorder before
          // the barrier so the token wait completes before the target buffer
          // can be updated.
          Operation *waitBarrier = findScheduledWaitBarrierBetween(
              tmaStore, targetTMAStore, schedule,
              k >= (numTMAStores - numPrevTMAStores));
          if (!waitBarrier) {
            LDBG("failed to find wait_barrier guarding target TMA store for "
                 "loop at "
                 << forOp.getLoc() << "; leaving TMA store wait unchanged");
            continue;
          }
          insertionTarget = waitBarrier;
        }

        // Split the cluster at the insertion target: ops before it remain
        // in the original cluster, the target and subsequent ops stay in
        // the returned cluster.
        auto targetCluster =
            schedule.splitClusterBefore(insertionTarget, forOp);
        // Insert a new cluster for our wait between the split halves.
        auto waitCluster = schedule.clusters.newBefore(targetCluster);
        schedule.insert(waitOp, targetStage, waitCluster);
      } else {
        // Target not found; leave the schedule unchanged for this wait.
        LDBG("no reorder target found for TMA store wait in loop at "
             << forOp.getLoc() << "; leaving TMA store wait unchanged");
        continue;
      }

      waitOp->removeAttr(kCanRotateByBufferCount);
      changed = true;
    }

    if (changed)
      schedule.serialize(forOp);
  });
}

struct NVGPUTestTMAStoreTokenWaitReorderPass
    : public impl::NVGPUTestTMAStoreTokenWaitReorderBase<
          NVGPUTestTMAStoreTokenWaitReorderPass> {
  void runOnOperation() override {
    getOperation()->walk(
        [&](triton::FuncOp funcOp) { doTMAStoreWaitReorder(funcOp); });
  }
};

// ---------------------------------------------------------------------------
// Lower TMAStoreTokenWaitOp with barriers into TMAStoreWaitOp + ArriveBarrierOp
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTMASTORETOKENWAITLOWERING
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

// Count TMA store-like ops (AsyncTMACopyLocalToGlobalOp and AsyncTMAReduceOp)
// in [from, to) within a block. An scf.if contributes the minimum store count
// across its branches; if there is no else region, the else contribution is 0.
struct ConditionBranch {
  Value condition;
  bool takeThen;
};

using ConditionContext = SmallVector<ConditionBranch, 4>;

static bool isInRegion(Operation *op, Region &region) {
  return region.findAncestorOpInRegion(*op) != nullptr;
}

static ConditionContext getConditionContext(Operation *op) {
  ConditionContext context;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(parent);
    if (!ifOp)
      continue;

    if (isInRegion(op, ifOp.getThenRegion())) {
      context.push_back({ifOp.getCondition(), /*takeThen=*/true});
    } else if (isInRegion(op, ifOp.getElseRegion())) {
      context.push_back({ifOp.getCondition(), /*takeThen=*/false});
    }
  }
  return context;
}

static std::optional<bool>
getConsistentBranch(scf::IfOp ifOp, const ConditionContext &context) {
  for (const ConditionBranch &branch : context) {
    if (samePureValue(ifOp.getCondition(), branch.condition))
      return branch.takeThen;
  }
  return std::nullopt;
}

static int countTMAStoresInRange(Block::iterator from, Block::iterator to,
                                 const ConditionContext &context);

static int countTMAStoresInRegion(Region &region,
                                  const ConditionContext &context) {
  if (region.empty())
    return 0;

  Block &block = region.front();
  return countTMAStoresInRange(block.begin(), block.end(), context);
}

static int countTMAStoreContribution(Operation *op,
                                     const ConditionContext &context) {
  if (isTMAStoreLikeOp(op))
    return 1;

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (std::optional<bool> branch = getConsistentBranch(ifOp, context)) {
      Region &selectedRegion =
          *branch ? ifOp.getThenRegion() : ifOp.getElseRegion();
      return countTMAStoresInRegion(selectedRegion, context);
    }

    int thenCount = countTMAStoresInRegion(ifOp.getThenRegion(), context);
    int elseCount = ifOp.getElseRegion().empty()
                        ? 0
                        : countTMAStoresInRegion(ifOp.getElseRegion(), context);
    return std::min(thenCount, elseCount);
  }

  return 0;
}

static int countTMAStoresInRange(Block::iterator from, Block::iterator to,
                                 const ConditionContext &context) {
  int count = 0;
  for (auto it = from; it != to; ++it) {
    count += countTMAStoreContribution(&*it, context);
  }
  return count;
}

static std::optional<int>
countTMAStoresUntilWait(Block *block, Block::iterator from,
                        ttng::TMAStoreTokenWaitOp waitOp,
                        const ConditionContext &context) {
  if (waitOp->getBlock() == block)
    return countTMAStoresInRange(from, waitOp->getIterator(), context);

  int count = 0;
  for (auto it = from; it != block->end(); ++it) {
    Operation *op = &*it;
    if (!op->isProperAncestor(waitOp)) {
      count += countTMAStoreContribution(op, context);
      continue;
    }

    for (Region &region : op->getRegions()) {
      if (!isInRegion(waitOp, region))
        continue;
      if (region.empty())
        return count;
      if (std::optional<int> inner = countTMAStoresUntilWait(
              &region.front(), region.front().begin(), waitOp, context))
        return count + *inner;
      return std::nullopt;
    }
    return std::nullopt;
  }

  return std::nullopt;
}

static std::optional<int>
computePendingsFromToken(Value token, ttng::TMAStoreTokenWaitOp waitOp,
                         const ConditionContext &context, unsigned depth = 0) {
  if (depth > 8)
    return std::nullopt;

  // Direct case: token defined by a TMA store-like op in same block.
  auto directDef = token.getDefiningOp();
  if (directDef && isTMAStoreLikeOp(directDef)) {
    return countTMAStoresUntilWait(directDef->getBlock(),
                                   std::next(directDef->getIterator()), waitOp,
                                   context);
  }

  // If-result case: count after the defining if, excluding the defining if
  // itself to match direct-token semantics.
  if (auto ifOp = token.getDefiningOp<scf::IfOp>()) {
    return countTMAStoresUntilWait(
        ifOp->getBlock(), std::next(ifOp->getIterator()), waitOp, context);
  }

  // Loop result case: the result may come from the loop body yield, or from the
  // initial iter_arg if the loop executes zero times. Use the conservative
  // minimum across both paths.
  if (auto forOp = token.getDefiningOp<scf::ForOp>()) {
    auto result = cast<OpResult>(token);
    unsigned iterArgIdx = result.getResultNumber();
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (iterArgIdx >= yieldOp.getNumOperands() ||
        iterArgIdx >= forOp.getInitArgs().size())
      return std::nullopt;

    Value yieldedVal = yieldOp.getOperand(iterArgIdx);
    auto yieldDef = yieldedVal.getDefiningOp();
    if (!yieldDef || !isTMAStoreLikeOp(yieldDef) ||
        yieldDef->getBlock() != forOp.getBody())
      return std::nullopt;

    Block *body = forOp.getBody();
    std::optional<int> afterLoop = countTMAStoresUntilWait(
        forOp->getBlock(), std::next(forOp->getIterator()), waitOp, context);
    if (!afterLoop)
      return std::nullopt;

    int loopPath = countTMAStoresInRange(std::next(yieldDef->getIterator()),
                                         body->end(), context) +
                   *afterLoop;

    std::optional<int> initPath = computePendingsFromToken(
        forOp.getInitArgs()[iterArgIdx], waitOp, context, depth + 1);
    if (!initPath)
      return std::nullopt;

    return std::min(loopPath, *initPath);
  }

  // Loop-carried case: token is a block argument of an scf.for body.
  if (auto blockArg = dyn_cast<BlockArgument>(token)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return std::nullopt;

    unsigned iterArgIdx = blockArg.getArgNumber() - 1;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yieldedVal = yieldOp.getOperand(iterArgIdx);

    // Trace the yielded value to its defining TMA store-like op.
    auto defOp = yieldedVal.getDefiningOp();
    if (!defOp || !isTMAStoreLikeOp(defOp) ||
        defOp->getBlock() != forOp.getBody())
      return std::nullopt;

    Block *body = forOp.getBody();

    // Stores after the def until end of loop body (excluding yield).
    int storesAfterDef = countTMAStoresInRange(std::next(defOp->getIterator()),
                                               body->end(), context);

    // Stores from start of loop body until the wait.
    int storesBeforeWait =
        countTMAStoresInRange(body->begin(), waitOp->getIterator(), context);

    return storesAfterDef + storesBeforeWait;
  }

  return std::nullopt;
}

// Compute the pendings value for a TMAStoreTokenWaitOp.
// pendings = number of TMA store-like ops issued after the token's defining
// store and before this wait, in program execution order.
static int computePendings(ttng::TMAStoreTokenWaitOp waitOp) {
  ConditionContext context = getConditionContext(waitOp);
  if (std::optional<int> count =
          computePendingsFromToken(waitOp.getToken(), waitOp, context))
    return *count;

  // Fallback: unknown pattern, drain all stores.
  return 0;
}

struct NVGPUTMAStoreTokenWaitLoweringPass
    : public impl::NVGPUTMAStoreTokenWaitLoweringBase<
          NVGPUTMAStoreTokenWaitLoweringPass> {
  void runOnOperation() override {
    SmallVector<ttng::TMAStoreTokenWaitOp> opsToLower;
    getOperation()->walk(
        [&](ttng::TMAStoreTokenWaitOp op) { opsToLower.push_back(op); });
    for (auto op : opsToLower) {
      OpBuilder builder(op);
      auto loc = op.getLoc();
      int pendings = computePendings(op);
      ttng::TMAStoreWaitOp::create(builder, loc, pendings);
      for (auto barrier : op.getBarriers()) {
        ttng::ArriveBarrierOp::create(builder, loc, barrier, /*count=*/1);
      }
      op.erase();
    }
  }
};

} // namespace mlir
