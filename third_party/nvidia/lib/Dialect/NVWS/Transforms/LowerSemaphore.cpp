/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "Utilities.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

#define DEBUG_TYPE "nvws-lower-semaphore"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERSEMAPHORE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

// Lowering contract and the egx/nvws-semaphore delta:
// sema-docs/assign-stage-phase-and-lower-semaphores.md.
// ----------------------------------------------------------------------------

struct PartitionWsTagIds {
  std::optional<int> wsTag;
  SetVector<int> partitionIds;
};
std::optional<PartitionWsTagIds> getPartitionWsTagIds(Operation *op) {
  std::optional<PartitionWsTagIds> partitionWsTagIds;
  if (hasPartition(op)) {
    partitionWsTagIds =
        PartitionWsTagIds{std::nullopt, triton::gpu::getPartitionIds(op)};
    if (auto wsTag = getWarpSpecializeTag(op)) {
      partitionWsTagIds->wsTag = *wsTag;
    }
  }
  return partitionWsTagIds;
}

void assignStageCluster(Operation *op,
                        std::optional<PartitionWsTagIds> partitionWsTagIds,
                        StageCluster stageCluster, OpBuilder &builder) {
  if (partitionWsTagIds) {
    setPartition(op, partitionWsTagIds->partitionIds);
    if (auto wsTag = partitionWsTagIds->wsTag) {
      setWarpSpecializeTag(op, *wsTag);
    }
    setStageCluster(builder, op, stageCluster);
  }
}

bool isOperandPipelineable(Value v, scf::ForOp forOp) {
  auto isPipelineable = [](Operation *op) {
    return isa<SemaphoreAcquireOp, SemaphoreBufferOp>(op);
  };

  Operation *foundDef = nullptr;
  return triton::nvidia_gpu::isOperandPipelineableBase(v, forOp, foundDef,
                                                       isPipelineable);
}

void setIsAsync(triton::nvidia_gpu::MMAv5OpInterface mmaOp,
                unsigned defaultNumStages) {
  bool isAsync = true;
  auto forOp = mmaOp->getParentOfType<scf::ForOp>();
  if (!forOp)
    return;

  unsigned numStages = getNumStagesOrDefault(forOp, defaultNumStages);
  if (numStages <= 1)
    return;

  if (auto scaledOp = dyn_cast<triton::nvidia_gpu::TCGen5MMAScaledOp>(
          mmaOp.getOperation())) {
    if (!triton::nvidia_gpu::areScalesPipelineable(scaledOp, forOp)) {
      isAsync = false;
    }
    if (!isOperandPipelineable(scaledOp.getAScale(), forOp) ||
        !isOperandPipelineable(scaledOp.getBScale(), forOp)) {
      isAsync = false;
    }
  }
  mmaOp.setIsAsync(isAsync);
}

// CONTRACT: pending_count is AUTHORED by the producing pass and REQUIRED here.
// LowerSemaphore uses it verbatim; folded circular semaphore IR is intentionally
// not re-derived from the post-fold physical release stream here.
FailureOr<int> getPendingCount(SemaphoreCreateOp op) {
  auto authored = op.getPendingCountAttr();
  if (!authored)
    return op.emitError(
        "semaphore.create reached nvws-lower-semaphore without a "
        "pending_count; the producing pass must author it");
  return authored.getInt();
}

FailureOr<Value> createAndInitMbar(SemaphoreCreateOp op,
                                   PatternRewriter &rewriter) {
  int numStages = op.getType().getNumStages();
  auto pendingCountOr = getPendingCount(op);
  if (failed(pendingCountOr))
    return failure();
  int pendingCount = *pendingCountOr;

  rewriter.setInsertionPoint(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto mbars = createScalarAlloc(b, b.getI64Type(), numStages);
  for (int i = 0; i < numStages; ++i) {
    auto view = createSingleBufferView(b, mbars, i);
    InitBarrierOp::create(b, view, pendingCount);
  }

  return mbars;
}

SmallVector<AsyncOp> castAsyncOpAttrs(ArrayAttr opAttrs) {
  SmallVector<AsyncOp> kinds;
  for (auto asyncKind : opAttrs) {
    kinds.push_back(cast<AsyncOpAttr>(asyncKind).getValue());
  }
  return kinds;
}

void createTMALoad(triton::nvws::DescriptorLoadOp op, PatternRewriter &rewriter,
                   Value barrierAlloc, Value pred) {
  auto newLoadOp = triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp::create(
      rewriter, op.getLoc(), op.getDesc(), op.getIndices(), barrierAlloc,
      op.getResult(), pred);
  assignStageCluster(newLoadOp, getPartitionWsTagIds(op), getStageCluster(op),
                     rewriter);
}

void createTMAGather(triton::nvws::DescriptorGatherOp op,
                     PatternRewriter &rewriter, Value barrierAlloc,
                     Value pred) {
  auto newGatherOp = triton::nvidia_gpu::AsyncTMAGatherOp::create(
      rewriter, op.getLoc(), op.getDesc(), op.getXOffsets(), op.getYOffset(),
      barrierAlloc, op.getResult(), pred);
  assignStageCluster(newGatherOp, getPartitionWsTagIds(op), getStageCluster(op),
                     rewriter);
}

FailureOr<Value> rematerializeStageBefore(Value stage,
                                          Operation *insertionPoint,
                                          PatternRewriter &rewriter) {
  if (!stage)
    return failure();
  rewriter.setInsertionPoint(insertionPoint);
  DominanceInfo domInfo;
  if (domInfo.properlyDominates(stage, insertionPoint))
    return stage;

  // AssignStagePhase materializes authored release offsets immediately before
  // the release. TMA lowering moves the barrier view to the earlier descriptor
  // load, so reproduce the pure scalar stage computation there as well.
  SetVector<Operation *> slice;
  auto canRematerialize = [](Operation *op) {
    return op->getNumRegions() == 0 && isPureScalarOp(op);
  };
  if (!getDominatingValueSetOpsToHoist(domInfo, insertionPoint, {stage}, slice,
                                       canRematerialize))
    return failure();

  IRMapping mapping;
  for (Operation *sliceOp : topologicalSort(slice))
    rewriter.clone(*sliceOp, mapping);
  return mapping.lookupOrDefault(stage);
}

LogicalResult lowerTMALoad(SemaphoreReleaseOp op,
                           PatternRewriter &rewriter, Value mbars) {
  auto kinds = castAsyncOpAttrs(op.getAsyncOps());
  if (!llvm::any_of(kinds,
                    [](AsyncOp kind) { return kind == AsyncOp::TMALoad; }))
    return success();

  auto loc = op.getLoc();
  int txCount = 0;
  SmallVector<Operation *> loadOps;
  for (auto tokUser : op.getToken().getUsers()) {
    auto bufOp = dyn_cast<SemaphoreBufferOp>(tokUser);
    if (!bufOp)
      continue;

    for (auto buffer : bufOp.getBuffers()) {
      for (auto user : buffer.getUsers()) {
        if (auto loadOp =
                dyn_cast<triton::nvws::DescriptorLoadOpInterface>(user)) {
          loadOps.push_back(loadOp);
          txCount += loadOp.getTxCount();
        }
      }
    }
  }
  assert(
      loadOps.size() <=
      op.getSemaphore().getDefiningOp<SemaphoreCreateOp>().getBuffers().size());
  if (loadOps.empty())
    return success();

  auto topo = topologicalSort({loadOps.begin(), loadOps.end()});
  loadOps.assign(topo.begin(), topo.end());

  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  auto stageOr =
      rematerializeStageBefore(op.getStage(), loadOps.front(), rewriter);
  if (failed(stageOr))
    return op.emitError("cannot rematerialize TMA release stage before its "
                        "descriptor load");
  auto fullBarrier = createSingleBufferView(rewriter, mbars, *stageOr);
  assignStageCluster(fullBarrier.getDefiningOp(), partitionWsTagIds,
                     stageCluster, rewriter);

  auto pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  assignStageCluster(pred, partitionWsTagIds, stageCluster, rewriter);
  auto expectOp = triton::nvidia_gpu::BarrierExpectOp::create(
      rewriter, loc, fullBarrier, txCount, pred);
  assignStageCluster(expectOp, partitionWsTagIds, stageCluster, rewriter);

  for (auto loadOp : loadOps) {
    rewriter.setInsertionPoint(loadOp);
    if (auto descLoad = dyn_cast<triton::nvws::DescriptorLoadOp>(loadOp)) {
      createTMALoad(descLoad, rewriter, fullBarrier, pred);
    } else if (auto descGather =
                   dyn_cast<triton::nvws::DescriptorGatherOp>(loadOp)) {
      createTMAGather(descGather, rewriter, fullBarrier, pred);
    } else {
      llvm_unreachable("Unknown load op");
    }
    loadOp->erase();
  }
  return success();
}

LogicalResult lowerTMALoads(SemaphoreCreateOp op, PatternRewriter &rewriter,
                            Value mbars) {
  for (auto user : op->getUsers()) {
    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp)
      continue;
    if (failed(lowerTMALoad(releaseOp, rewriter, mbars)))
      return failure();
  }
  return success();
}

void rewriteAcquire(SemaphoreAcquireOp op, PatternRewriter &rewriter,
                    Value mbars) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
  assignStageCluster(mbar.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);

  auto waitOp = WaitBarrierOp::create(rewriter, loc, mbar, op.getPhase());
  assignStageCluster(waitOp, partitionWsTagIds, stageCluster, rewriter);
}

LogicalResult rewriteRelease(
    SemaphoreCreateOp semaOp, SemaphoreReleaseOp op, PatternRewriter &rewriter,
    Value mbars, const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema) {
  auto loc = op.getLoc();
  auto asyncKinds = castAsyncOpAttrs(op.getAsyncOps());
  // CONTRACT: arrive_count is authored by the producing pass and required.
  auto countAttr = op.getArriveCountAttr();
  if (!countAttr)
    return op.emitError(
        "semaphore.release reached nvws-lower-semaphore without an "
        "arrive_count; the producing pass must author it");
  int arriveCount = countAttr.getInt();
  rewriter.setInsertionPointAfter(op);
  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  bool needFence = [&]() {
    bool isGenericProxy = llvm::any_of(
        asyncKinds, [](AsyncOp kind) { return kind == AsyncOp::NONE; });
    if (!isGenericProxy)
      return false;

    // Currently we assume that an semaphore buffer does not contain both SMEM
    // and TMEM. So checking only the first buffer is fine.
    auto semaType = cast<SemaphoreType>(semaOp.getType());
    auto semaBufType = cast<MemDescType>(semaType.getBaseType()[0]);
    auto tmem = TensorMemorySpaceAttr::get(semaOp.getContext());
    if (semaBufType.getMemorySpace() == tmem)
      return false;

    // Fence decision depends on other semaphores grouped with this semaphore
    // by the first backing buffer. For a generic release (async_ops=[none]),
    // we need a fence if some other semaphore in that group has an async
    // release with either TC5MMA or TMALoad. Those async releases can make a
    // generic release on this semaphore need fence_async_shared before
    // arriving on its mbarrier.
    auto it = hasAsyncPeerBySema.find(semaOp.getOperation());
    return it != hasAsyncPeerBySema.end() && it->second;
  }();

  if (needFence) {
    auto fence = FenceAsyncSharedOp::create(rewriter, loc, /*bCluster=*/false);
    assignStageCluster(fence, partitionWsTagIds, stageCluster, rewriter);
  }

  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
  assignStageCluster(mbar.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);

  for (auto asyncKind : asyncKinds) {
    Operation *arriveOp = nullptr;
    switch (asyncKind) {
    case AsyncOp::NONE:
    case AsyncOp::WGMMA:
      arriveOp = ArriveBarrierOp::create(rewriter, loc, mbar, arriveCount);
      break;
    case AsyncOp::TC5MMA:
    case AsyncOp::TMEMCopy:
    case AsyncOp::TMALoad:
      // Commit/TMA-completed arrivals cannot arrive N times (user ruling,
      // fable/integrate-pending-count-plan.md).
      if (arriveCount > 1)
        return op.emitError("arrive_count > 1 is only lowerable for "
                            "none/wgmma async kinds");
      if (asyncKind != AsyncOp::TMALoad)
        arriveOp = TCGen5CommitOp::create(rewriter, loc, mbar, Value(),
                                          ValueRange{});
      break;
    case AsyncOp::CpAsync:
    default:
      llvm_unreachable("unsupported async op");
    }
    if (arriveOp)
      assignStageCluster(arriveOp, partitionWsTagIds, stageCluster, rewriter);
  }
  return success();
}

static MemDescType getAsMutable(MemDescType type) {
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          /*mutableMemory=*/true);
}

static void propagateMutability(Value value) {
  for (Operation *user : value.getUsers()) {
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      user->getResult(0).setType(
          getAsMutable(cast<MemDescType>(user->getResult(0).getType())));
      propagateMutability(user->getResult(0));
    }
  }
}

void rewriteBuffer(SemaphoreBufferOp op, PatternRewriter &rewriter,
                   ArrayRef<Value> buffers) {
  auto loc = op.getLoc();
  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  for (auto [i, buffer] : llvm::enumerate(buffers)) {
    // replacement helper may erase ops adjacent to this insertion point,
    // so refresh it for each buffer result before creating new view ops.
    rewriter.setInsertionPointAfter(op);

    auto memDesc = cast<MemDescType>(buffer.getType());
    if (isa<TensorMemoryScalesEncodingAttr>(memDesc.getEncoding())) {
      op.getBuffers()[i].replaceAllUsesWith(buffer);
      continue;
    }

    auto shape = memDesc.getShape();
    assert(shape.size() > 1 && "expected multi-buffered semaphore buffer");
    SmallVector<int64_t> viewShape(shape.begin() + 1, shape.end());
    auto viewType =
        MemDescType::get(viewShape, memDesc.getElementType(),
                         memDesc.getEncoding(), memDesc.getMemorySpace(),
                         /*mutableMemory=*/true);
    auto view =
        MemDescIndexOp::create(rewriter, loc, viewType, buffer, op.getStage());
    assignStageCluster(view, partitionWsTagIds, stageCluster, rewriter);
    op.getBuffers()[i].replaceAllUsesWith(view);
    // Before lowering, memdesc_trans consumes an immutable buffer.
    // After lowering, all buffers are mutable.
    propagateMutability(view);
  }
}

DenseSet<MMAv5OpInterface> getAsyncMMAv5Consumers(Value semaphore) {
  DenseSet<MMAv5OpInterface> mmav5Ops;
  for (auto semaUser : semaphore.getUsers()) {
    auto acquireOp = dyn_cast<SemaphoreAcquireOp>(semaUser);
    if (!acquireOp)
      continue;
    if (hasPartition(acquireOp) && getPartitionIds(acquireOp).front() == 0) {
      // Ignore MMAv5 ops in the default partition. They are not warp
      // specialized.
      continue;
    }

    for (auto tokUser : acquireOp.getToken().getUsers()) {
      auto bufferOp = dyn_cast<SemaphoreBufferOp>(tokUser);
      if (!bufferOp)
        continue;

      for (auto consumer : bufferOp->getUsers()) {
        if (auto mmav5 = dyn_cast<MMAv5OpInterface>(consumer)) {
          mmav5Ops.insert(mmav5);
        } else if (auto forOp = consumer->getParentOfType<scf::ForOp>()) {
          auto users =
              getTopLevelUsersInLoop(consumer, forOp, [](Operation *user) {
                return isa<MMAv5OpInterface>(user);
              });
          for (auto user : users) {
            mmav5Ops.insert(cast<MMAv5OpInterface>(user));
          }
        }
      }
    }
  }
  return mmav5Ops;
}

// PartitionLoops moves partitioned siblings carrying a warp-specialization tag
// into the matching loop's warp group. Semaphore cleanup remains unpartitioned,
// so a tagged user before the loop is semantically live through that loop even
// though its current lexical position precedes it.
void addWarpSpecializationCleanupAnchors(
    SemaphoreCreateOp op, SetVector<Operation *> &users) {
  Block *block = op->getBlock();
  DenseSet<int> movedUserTags;
  for (Operation *user : users) {
    Operation *anchor = block->findAncestorOpInBlock(*user);
    if (!anchor || !hasPartition(anchor))
      continue;
    if (auto tag = getWarpSpecializeTag(anchor))
      movedUserTags.insert(*tag);
  }
  if (movedUserTags.empty())
    return;

  for (Operation &candidate : *block) {
    auto loop = dyn_cast<scf::ForOp>(&candidate);
    if (!loop)
      continue;
    auto stages = loop->getAttrOfType<ArrayAttr>(kPartitionStagesAttrName);
    if (!stages || stages.size() <= 1)
      continue;
    if (auto tag = getWarpSpecializeTag(loop);
        tag && movedUserTags.contains(*tag))
      users.insert(loop);
  }
}

class LowerSemaphoreCreate : public OpRewritePattern<SemaphoreCreateOp> {
public:
  LowerSemaphoreCreate(
      MLIRContext *ctx,
      const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema,
      unsigned defaultNumStages)
      : OpRewritePattern<SemaphoreCreateOp>(ctx),
        hasAsyncPeerBySema(hasAsyncPeerBySema),
        defaultNumStages(defaultNumStages) {}

  LogicalResult matchAndRewrite(SemaphoreCreateOp op,
                                PatternRewriter &rewriter) const override {
    for (auto user : op->getUsers()) {
      auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
      if (!releaseOp)
        continue;
      auto kinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
      if (llvm::any_of(kinds, [](AsyncOp kind) {
            return kind == AsyncOp::TMALoad || kind == AsyncOp::CpAsync;
          })) {
        // the semaphore release op is async, so we need to setIsAsync(true)
        // if the peer semaphore consumes data via mmav5 ops.
        for (auto mma : getAsyncMMAv5Consumers(op.getResult()))
          setIsAsync(mma, defaultNumStages);
        break;
      }
    }

    auto mbarsOr = createAndInitMbar(op, rewriter);
    if (failed(mbarsOr))
      return failure();
    Value mbars = *mbarsOr;
    SmallVector<Value> buffers(op.getBuffers().begin(), op.getBuffers().end());

    // Load TMA loads before erasing/rewriting semaphore users.
    if (failed(lowerTMALoads(op, rewriter, mbars)))
      return failure();

    SetVector<Operation *> opToDelete;
    opToDelete.insert(op.getOperation());

    SetVector<Operation *> allUsers;
    for (Operation *user : op->getUsers())
      allUsers.insert(user);
    addWarpSpecializationCleanupAnchors(op, allUsers);

    Operation *cleanupAnchor = op.getOperation();
    if (!allUsers.empty()) {
      auto sortedUsers = topologicalSort(allUsers);
      cleanupAnchor =
          op->getBlock()->findAncestorOpInBlock(*sortedUsers.back());
    }

    {
      ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      b.setInsertionPointAfter(cleanupAnchor);
      int numStages = op.getType().getNumStages();
      for (int i = 0; i < numStages; ++i) {
        auto view = createSingleBufferView(b, mbars, i);
        InvalBarrierOp::create(b, view);
      }
      LocalDeallocOp::create(b, mbars);
    }

    SmallVector<Operation *> users(op->getUsers().begin(),
                                   op->getUsers().end());
    for (auto userOp : users) {
      opToDelete.insert(userOp);
      if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(userOp)) {
        rewriteAcquire(acquireOp, rewriter, mbars);
      } else if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(userOp)) {
        if (failed(rewriteRelease(op, releaseOp, rewriter, mbars,
                                  hasAsyncPeerBySema)))
          return failure();
      } else if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(userOp)) {
        rewriteBuffer(bufferOp, rewriter, buffers);
      } else {
        llvm_unreachable("unexpected semaphore user");
      }
    }

    auto sorted = topologicalSort(opToDelete);
    OpBuilder b(op);
    auto replToken =
        ub::PoisonOp::create(b, op.getLoc(), b.getType<AsyncTokenType>());
    // Poison tokens may be yielded by ws-loops and PartitionLoops requires
    // all ops to carry partition annotations.  Copy from the semaphore.
    if (hasPartition(op))
      setPartition(replToken, getPartitionIds(op));
    for (auto candidate : sorted) {
      if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(candidate))
        acquireOp.getToken().replaceAllUsesWith(replToken);
    }
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it)
      rewriter.eraseOp(*it);

    return success();
  }

private:
  const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema;
  unsigned defaultNumStages;
};

// Precompute cross-semaphore async relationships before any rewrite:
//
// hasAsyncPeerBySema[S] == true iff some other semaphore grouped with S by the
// first backing buffer has an async release with either TC5MMA or TMALoad.
// rewriteRelease uses that precomputed fact together with the current release
// kind and memory space to decide whether to insert fence_async_shared before
// arriving on S's mbarrier.
//
// This is required for deterministic/correct fence lowering. Greedy rewrites
// process semaphores independently and may erase one semaphore before
// rewriting its peer; computing this relation ahead of time avoids
// rewrite-order-dependent fence decisions.
llvm::DenseMap<Operation *, bool> computeHasAsyncPeerBySema(
    llvm::DenseMap<Value, SmallVector<SemaphoreCreateOp>> semaGroups) {
  llvm::DenseMap<Operation *, bool> hasAsyncPeerBySema;
  for (auto &[_, semas] : semaGroups) {
    llvm::DenseMap<Operation *, bool> hasAsyncRelease;
    for (auto semaOp : semas) {
      bool hasAsync = false;
      for (Operation *user : semaOp->getUsers()) {
        auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
        if (!releaseOp)
          continue;
        auto kinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
        bool hasAsyncConsumer = llvm::any_of(
            kinds, [](AsyncOp kind) { return kind == AsyncOp::TC5MMA; });
        bool hasAsyncProducer = llvm::any_of(
            kinds, [](AsyncOp kind) { return kind == AsyncOp::TMALoad; });
        if (hasAsyncConsumer || hasAsyncProducer) {
          hasAsync = true;
          break;
        }
      }
      hasAsyncRelease[semaOp.getOperation()] = hasAsync;
    }

    for (auto semaOp : semas) {
      bool hasAsyncPeer = llvm::any_of(semas, [&](SemaphoreCreateOp otherSema) {
        return otherSema != semaOp &&
               hasAsyncRelease.lookup(otherSema.getOperation());
      });
      hasAsyncPeerBySema[semaOp.getOperation()] = hasAsyncPeer;
    }
  }

  return hasAsyncPeerBySema;
}

void hoistPoisonOps(triton::FuncOp funcOp) {
  auto block = &funcOp.getBody().front();
  funcOp.walk([&](ub::PoisonOp op) { op->moveBefore(&block->front()); });
}

LogicalResult verifyLoweringPreconditions(ModuleOp module) {
  LogicalResult result = success();
  module.walk([&](SemaphoreAcquireOp op) {
    if (!op.getStage() || !op.getPhase()) {
      op.emitError("requires stage and phase operands; run "
                   "nvws-assign-semaphore-stage-phase first");
      result = failure();
    }
  });
  module.walk([&](SemaphoreBufferOp op) {
    if (!op.getStage()) {
      op.emitError("requires a stage operand; run "
                   "nvws-assign-semaphore-stage-phase first");
      result = failure();
    }
  });
  module.walk([&](SemaphoreReleaseOp op) {
    if (!op.getStage()) {
      op.emitError("requires a stage operand; run "
                   "nvws-assign-semaphore-stage-phase first");
      result = failure();
    }
  });
  return result;
}

} // anonymous namespace

class NVWSLowerSemaphore
    : public impl::NVWSLowerSemaphoreBase<NVWSLowerSemaphore> {
  using impl::NVWSLowerSemaphoreBase<
      NVWSLowerSemaphore>::NVWSLowerSemaphoreBase;

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    if (failed(verifyLoweringPreconditions(m)))
      return signalPassFailure();

    auto getSemaGroups = [&]() {
      llvm::DenseMap<Value, SmallVector<SemaphoreCreateOp>> semaGroups;
      m.walk([&](SemaphoreCreateOp semaOp) {
        semaGroups[semaOp.getBuffers().front()].push_back(semaOp);
      });
      return semaGroups;
    };

    mlir::RewritePatternSet patterns(context);
    // Precompute peer information before rewriting: lowering one semaphore
    // erases it, but peer relationships must not depend on rewrite order.
    auto hasAsyncPeerBySema = computeHasAsyncPeerBySema(getSemaGroups());
    patterns.add<LowerSemaphoreCreate>(context, hasAsyncPeerBySema, numStages);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(m, std::move(patterns), config)))
      return signalPassFailure();

    // Poison tokens replace erased semaphore tokens. Keep them outside
    // partitioned regions for the downstream partitioning passes.
    m.walk([&](triton::FuncOp funcOp) { hoistPoisonOps(funcOp); });
  }
};

} // namespace triton
} // namespace mlir
