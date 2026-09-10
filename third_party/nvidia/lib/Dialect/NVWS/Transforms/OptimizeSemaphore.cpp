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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSSEMAPHOREOPTIMIZE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

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

SmallVector<AsyncOp> castAsyncOpAttrs(ArrayAttr opAttrs) {
  SmallVector<AsyncOp> kinds;
  for (auto asyncKind : opAttrs) {
    kinds.push_back(cast<AsyncOpAttr>(asyncKind).getValue());
  }
  return kinds;
}

bool hasProducerLoad(SemaphoreCreateOp semaOp) {
  for (auto user : semaOp->getUsers()) {
    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp)
      continue;
    auto asyncKinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
    if (llvm::any_of(asyncKinds,
                     [](AsyncOp kind) { return kind == AsyncOp::TMALoad; })) {
      return true;
    }
  }
  return false;
}

int getSemaphoreGroupNumStages(ArrayRef<SemaphoreCreateOp> semas,
                               int defaultNumStages) {
  std::optional<int> groupNumStages;
  for (SemaphoreCreateOp semaOp : semas) {
    for (Operation *user : semaOp->getUsers()) {
      auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
      if (!releaseOp)
        continue;
      auto asyncKinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
      if (!llvm::is_contained(asyncKinds, AsyncOp::TMALoad))
        continue;

      auto innerLoop = releaseOp->getParentOfType<scf::ForOp>();
      auto wsLoop = innerLoop ? getOuterWSLoop(innerLoop) : scf::ForOp{};
      if (!wsLoop)
        continue;

      int loopNumStages = getNumStagesOrDefault(wsLoop, defaultNumStages);
      // A physical semaphore group has one depth. If it is shared by multiple
      // WS loops, provision it for the largest authored loop depth.
      groupNumStages = std::max(groupNumStages.value_or(0), loopNumStages);
    }
  }
  return groupNumStages.value_or(defaultNumStages);
}

void multiBufferSemaphore(
    llvm::DenseMap<Value, SmallVector<SemaphoreCreateOp>> semaGroups,
    int defaultNumStages) {
  SetVector<Operation *> allocsToErase;
  for (auto &[_, semas] : semaGroups) {
    if (!llvm::any_of(semas, hasProducerLoad)) {
      continue;
    }

    // The command-line value is only a default. An explicit tt.num_stages on
    // the owning WS loop must control both software pipelining and the buffer
    // ring allocated for its TMA semaphore protocol.
    int numStages = getSemaphoreGroupNumStages(semas, defaultNumStages);
    if (numStages <= 1)
      continue;

    bool eligible = true;
    for (auto opnd : semas.front().getBuffers()) {
      Operation *defOp = opnd.getDefiningOp();
      auto localAlloc = dyn_cast_or_null<LocalAllocOp>(defOp);
      if (!localAlloc || localAlloc->hasAttr("buffer.copy")) {
        eligible = false;
        break;
      }
    }

    if (!eligible) {
      continue;
    }

    OpBuilder builder(semas.front());
    SmallVector<Value> newBuffers;
    SmallVector<Type> newBufferTypes;
    newBuffers.reserve(semas.front().getBuffers().size());
    newBufferTypes.reserve(semas.front().getBuffers().size());

    for (auto opnd : semas.front().getBuffers()) {
      auto oldAlloc = opnd.getDefiningOp();
      auto oldBufType = cast<MemDescType>(opnd.getType());
      auto newBufType =
          getMultiBufferedType(getBufferViewType(oldBufType, true), numStages);
      Operation *newAlloc = triton::nvws::createAlloc(
          builder, oldAlloc->getLoc(), newBufType, Value());
      newBuffers.push_back(newAlloc->getResult(0));
      newBufferTypes.push_back(newBufType);
      oldAlloc->replaceAllUsesWith(newAlloc);
      allocsToErase.insert(oldAlloc);
    }

    for (auto semaOp : semas) {
      OpBuilder semaBuilder(semaOp);
      auto semaTy = SemaphoreType::get(
          semaBuilder.getContext(),
          TypeArrayAttr::get(semaBuilder.getContext(), newBufferTypes));
      uint32_t releasedMask = resizeReleasedMask(
          getReleasedMask(semaOp), semaOp.getType().getNumStages(),
          semaTy.getNumStages());
      auto newSema =
          SemaphoreCreateOp::create(semaBuilder, semaOp.getLoc(), semaTy,
                                    newBuffers, releasedMask);
      newSema->setAttrs(semaOp->getAttrs());
      if (releasedMask)
        newSema->setAttr("released_mask",
                         semaBuilder.getI32IntegerAttr(releasedMask));
      else
        newSema->removeAttr("released_mask");
      semaOp.getResult().replaceAllUsesWith(newSema.getResult());
      semaOp.erase();
    }
  }

  for (auto alloc : allocsToErase) {
    alloc->erase();
  }
}

// ---------------------------------------------------------------------------
// combineSemaphores: coalesce multiple semaphore pairs that feed the same
// dominant consumer in a warp-specialize for-loop.
// ---------------------------------------------------------------------------

struct CombinedReleasePlan {
  SmallVector<Attribute> asyncOps;
  int pendingCount;
};

FailureOr<CombinedReleasePlan>
planCombinedRelease(ArrayRef<SemaphoreReleaseOp> releaseOps) {
  assert(!releaseOps.empty());
  llvm::SmallSetVector<Attribute, 5> asyncOpsSet;
  for (SemaphoreReleaseOp releaseOp : releaseOps) {
    for (Attribute attr : releaseOp.getAsyncOps()) {
      auto kind = cast<AsyncOpAttr>(attr).getValue();
      if (kind == AsyncOp::CpAsync) {
        releaseOp.emitError(
            "cannot combine semaphore release with unsupported cp_async kind");
        return failure();
      }
      asyncOpsSet.insert(attr);
    }
  }
  if (asyncOpsSet.empty()) {
    SemaphoreReleaseOp releaseOp = releaseOps.front();
    releaseOp.emitError("cannot combine semaphore releases with empty "
                        "async_ops");
    return failure();
  }

  // Combining collapses all releases on one side to one release site. Each
  // distinct completion kind contributes one arrival at that site, so both
  // arrive_count and the barrier threshold are normalized to that topology.
  SmallVector<Attribute> asyncOps(asyncOpsSet.begin(), asyncOpsSet.end());
  return CombinedReleasePlan{std::move(asyncOps),
                             static_cast<int>(asyncOpsSet.size())};
}

void createCombinedSemaphoreOps(ArrayRef<SemaphoreAcquireOp> acquireOps,
                                ArrayRef<SemaphoreBufferOp> bufferOps,
                                ArrayRef<SemaphoreReleaseOp> releaseOps,
                                SemaphoreCreateOp acquireSema,
                                SemaphoreCreateOp releaseSema,
                                const CombinedReleasePlan &releasePlan,
                                OpBuilder &builder) {
  assert(!acquireOps.empty() && !bufferOps.empty() && !releaseOps.empty());

  auto firstAcquire = *llvm::min_element(acquireOps, [](auto a, auto b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });
  auto lastRelease = *llvm::max_element(releaseOps, [](auto a, auto b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  auto partition = getPartitionWsTagIds(firstAcquire);
  auto stage = getStageCluster(firstAcquire);

  builder.setInsertionPoint(firstAcquire);
  auto combinedAcquire =
      SemaphoreAcquireOp::create(builder, firstAcquire.getLoc(), acquireSema,
                                 builder.getType<AsyncTokenType>());
  assignStageCluster(combinedAcquire, partition, stage, builder);

  SmallVector<Type> bufferResultTypes;
  for (auto bufferOp : bufferOps) {
    for (auto res : bufferOp.getBuffers())
      bufferResultTypes.push_back(res.getType());
  }

  builder.setInsertionPointAfter(combinedAcquire);
  auto combinedBuffer = SemaphoreBufferOp::create(
      builder, firstAcquire.getLoc(), acquireSema, TypeRange(bufferResultTypes),
      combinedAcquire.getToken());
  assignStageCluster(combinedBuffer, partition, stage, builder);

  std::function<void(Operation *, Operation *)> moveUserAfter =
      [&](Operation *op, Operation *target) {
        auto curBlock = target->getBlock();
        for (auto user : op->getUsers()) {
          auto userOp = curBlock->findAncestorOpInBlock(*user);
          if (userOp->isBeforeInBlock(target)) {
            userOp->moveAfter(target);
            moveUserAfter(userOp, userOp);
          }
        }
      };

  int bufOffset = 0;
  for (auto bufferOp : bufferOps) {
    moveUserAfter(bufferOp, combinedBuffer);
    for (auto [j, oldBuf] : llvm::enumerate(bufferOp.getBuffers()))
      oldBuf.replaceAllUsesWith(combinedBuffer.getBuffers()[bufOffset + j]);
    bufOffset += bufferOp.getBuffers().size();
  }

  builder.setInsertionPoint(lastRelease);
  auto combinedRelease = SemaphoreReleaseOp::create(
      builder, lastRelease.getLoc(), releaseSema, combinedAcquire.getToken(),
      builder.getArrayAttr(releasePlan.asyncOps));
  combinedRelease.setArriveCountAttr(builder.getI32IntegerAttr(1));
  releaseSema.setPendingCountAttr(
      builder.getI32IntegerAttr(releasePlan.pendingCount));
  assignStageCluster(combinedRelease, getPartitionWsTagIds(lastRelease),
                     getStageCluster(lastRelease), builder);
}

SmallVector<Operation *> findSharedMemorySinkOps(Value value) {
  SmallVector<Operation *> sinkOps;
  for (Operation *user : value.getUsers()) {
    if (isa<MMAv5OpInterface, LocalLoadOp>(user)) {
      sinkOps.push_back(user);
    } else if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      auto rec = findSharedMemorySinkOps(user->getResult(0));
      sinkOps.insert(sinkOps.end(), rec.begin(), rec.end());
    }
  }
  return sinkOps;
}

// 2-hop traversal: acquireOp → token → SemaphoreBufferOp → buffer results
// → findSharedMemorySinkOps → findNearestCommonDominator.
Operation *getDominantConsumer(SemaphoreAcquireOp acquireOp, Block &container,
                               DominanceInfo &domInfo) {
  SmallVector<Operation *> sinkOps;
  for (auto tokUser : acquireOp.getToken().getUsers()) {
    auto bufferOp = dyn_cast<SemaphoreBufferOp>(tokUser);
    if (!bufferOp)
      continue;
    for (auto buf : bufferOp.getBuffers()) {
      auto ops = findSharedMemorySinkOps(buf);
      sinkOps.insert(sinkOps.end(), ops.begin(), ops.end());
    }
  }
  if (sinkOps.empty()) {
    return nullptr;
  }
  Operation *liveBeforeOp = findNearestCommonDominator(sinkOps, domInfo);
  return container.findAncestorOpInBlock(*liveBeforeOp);
}

struct SemaToCombineInfo {
  SemaphoreCreateOp emptySema;
  SemaphoreCreateOp fullSema;
  SemaphoreBufferOp consBufferOp;
  SemaphoreReleaseOp consReleaseOp;
  SemaphoreAcquireOp prodAcquireOp;
  SemaphoreBufferOp prodBufferOp;
  SemaphoreReleaseOp prodReleaseOp;
};

SmallVector<SemaToCombineInfo>
analyzeCombinedSemaphoreGroup(ArrayRef<SemaphoreAcquireOp> acquireGroup) {
  SmallVector<SemaToCombineInfo> combinedInfos;
  SmallVector<int> producerPartitionIds;

  for (auto consAcquire : acquireGroup) {
    SemaToCombineInfo info;

    // Acquire-token lineage: consumer acquires FULL; consumer release targets
    // EMPTY (cross-release). Follow consumer acquire -> token ->
    // SemaphoreReleaseOp -> getSemaphore() to find that partner EMPTY
    // semaphore.
    for (Operation *tokUser : consAcquire.getToken().getUsers()) {
      if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(tokUser)) {
        if (info.consReleaseOp)
          return {};
        info.consReleaseOp = releaseOp;
      } else if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(tokUser)) {
        if (info.consBufferOp)
          return {};
        info.consBufferOp = bufferOp;
      }
    }

    // Skip groups whose consumer acquire does not have the canonical
    // FULL acquire -> buffer -> cross-release EMPTY protocol shape.
    if (!info.consBufferOp || !info.consReleaseOp)
      return {};

    auto fullSema =
        consAcquire.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    auto emptySema =
        info.consReleaseOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    info.emptySema = emptySema;
    info.fullSema = fullSema;

    // Collect producer partition IDs from EMPTY semaphore users.
    for (auto user : emptySema->getUsers()) {
      auto prodAcquire = dyn_cast<SemaphoreAcquireOp>(user);
      if (!prodAcquire)
        continue;
      if (!hasPartition(prodAcquire))
        return {};
      auto partitionIds = getPartitionIds(prodAcquire);
      if (partitionIds.size() != 1)
        return {};
      producerPartitionIds.push_back(partitionIds.front());
      if (info.prodAcquireOp)
        return {};
      info.prodAcquireOp = prodAcquire;
      for (Operation *tokUser : prodAcquire.getToken().getUsers()) {
        if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(tokUser)) {
          if (info.prodBufferOp)
            return {};
          info.prodBufferOp = bufferOp;
        }
        if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(tokUser)) {
          if (info.prodReleaseOp)
            return {};
          info.prodReleaseOp = releaseOp;
        }
      }
    }

    if (!info.prodAcquireOp || !info.prodBufferOp || !info.prodReleaseOp)
      return {};

    // Only combine pairs whose users are covered by this rewrite.
    for (Operation *user : fullSema->getUsers())
      if (user != consAcquire && user != info.consBufferOp &&
          user != info.prodReleaseOp)
        return {};

    for (Operation *user : emptySema->getUsers())
      if (user != info.prodAcquireOp && user != info.prodBufferOp &&
          user != info.consReleaseOp)
        return {};

    combinedInfos.push_back(info);
  }

  // All producers must be in the same partition.
  if (!producerPartitionIds.empty() &&
      llvm::any_of(producerPartitionIds,
                   [&](int id) { return id != producerPartitionIds[0]; })) {
    // The combine rewrite assumes one producer partition for the whole group.
    // Mixed producer partitions need a different protocol reconstruction.
    return {};
  }

  // A combined semaphore has one stage cursor and indexes every backing buffer
  // with it, so all component semaphore depths must agree.
  std::optional<int> combinedDepth;
  std::optional<uint32_t> emptyReleasedMask, fullReleasedMask;
  for (SemaToCombineInfo info : combinedInfos) {
    int depth = info.fullSema.getType().getNumStages();
    if (combinedDepth && *combinedDepth != depth)
      return {};
    combinedDepth = depth;
    uint32_t emptyMask = getReleasedMask(info.emptySema);
    uint32_t fullMask = getReleasedMask(info.fullSema);
    if ((emptyReleasedMask && *emptyReleasedMask != emptyMask) ||
        (fullReleasedMask && *fullReleasedMask != fullMask))
      return {};
    emptyReleasedMask = emptyMask;
    fullReleasedMask = fullMask;
  }

  return combinedInfos;
}

struct CombinedSemaPair {
  SemaphoreCreateOp empty;
  SemaphoreCreateOp full;
};

CombinedSemaPair createCombinedSemaphores(ArrayRef<SemaToCombineInfo> infos,
                                          scf::ForOp loop) {
  SmallVector<Value> allBufs;
  SmallVector<Type> allBufTypes;
  for (auto info : infos) {
    for (Value buf : info.fullSema.getBuffers()) {
      allBufs.push_back(buf);
      allBufTypes.push_back(buf.getType());
    }
  }

  auto lastInfo = *llvm::max_element(infos, [](auto a, auto b) {
    assert(a.fullSema->getBlock() == b.fullSema->getBlock());
    return a.fullSema->isBeforeInBlock(b.fullSema);
  });
  auto lastCreate = lastInfo.fullSema;

  auto *ctx = loop->getContext();
  auto combinedType =
      SemaphoreType::get(ctx, TypeArrayAttr::get(ctx, allBufTypes));

  OpBuilder builder(lastCreate);
  builder.setInsertionPointAfter(lastCreate);
  // EMPTY must appear before FULL in IR so that the greedy rewriter
  // processes FULL first. lowerTMALoads on the FULL semaphore follows
  // the producer-release token chain through the EMPTY semaphore's
  // acquire/buffer ops; those must still be live at that point.
  auto combinedEmpty =
      SemaphoreCreateOp::create(builder, lastCreate->getLoc(), combinedType,
                                allBufs,
                                getReleasedMask(infos.front().emptySema));
  auto combinedFull =
      SemaphoreCreateOp::create(builder, lastCreate->getLoc(), combinedType,
                                allBufs,
                                getReleasedMask(infos.front().fullSema));
  return {combinedEmpty, combinedFull};
}

void combineConsumerSide(ArrayRef<SemaphoreAcquireOp> acquireGroup,
                         ArrayRef<SemaToCombineInfo> infos,
                         CombinedSemaPair combinedPair,
                         const CombinedReleasePlan &releasePlan,
                         OpBuilder &builder) {
  // Consumer acquires FULL, buffers from it, then cross-releases EMPTY.
  SmallVector<SemaphoreBufferOp> consBufferOps;
  SmallVector<SemaphoreReleaseOp> consReleaseOps;
  for (auto info : infos) {
    consBufferOps.push_back(info.consBufferOp);
    consReleaseOps.push_back(info.consReleaseOp);
  }
  createCombinedSemaphoreOps(acquireGroup, consBufferOps, consReleaseOps,
                             combinedPair.full, combinedPair.empty, releasePlan,
                             builder);
}

void combineProducerSide(ArrayRef<SemaToCombineInfo> infos,
                         CombinedSemaPair combinedPair,
                         const CombinedReleasePlan &releasePlan,
                         OpBuilder &builder) {
  // Producer acquires EMPTY, buffers from it, then cross-releases FULL.
  SmallVector<SemaphoreAcquireOp> prodAcquireOps;
  SmallVector<SemaphoreBufferOp> prodBufferOps;
  SmallVector<SemaphoreReleaseOp> prodReleaseOps;
  for (auto info : infos) {
    prodAcquireOps.push_back(info.prodAcquireOp);
    prodBufferOps.push_back(info.prodBufferOp);
    prodReleaseOps.push_back(info.prodReleaseOp);
  }
  createCombinedSemaphoreOps(prodAcquireOps, prodBufferOps, prodReleaseOps,
                             combinedPair.empty, combinedPair.full, releasePlan,
                             builder);
}

void eraseSemaToCombineGroup(ArrayRef<SemaphoreAcquireOp> acquireGroup,
                             ArrayRef<SemaToCombineInfo> infos) {
  for (auto info : infos)
    info.consReleaseOp->erase();
  for (auto info : infos)
    info.consBufferOp->erase();
  for (auto acquireOp : acquireGroup)
    acquireOp->erase();
  for (auto info : infos)
    info.prodReleaseOp->erase();
  for (auto info : infos)
    info.prodBufferOp->erase();
  for (auto info : infos)
    info.prodAcquireOp->erase();
  for (auto info : infos)
    info.fullSema->erase();
  for (auto info : infos)
    info.emptySema->erase();
}

LogicalResult combineSemaphores(scf::ForOp loop) {
  // 1. Find consumer acquire ops (consumer = acquires a semaphore with no
  //    initially released stages). Skip TMEM.
  SmallVector<SemaphoreAcquireOp> consumerAcquires;
  auto tmem = TensorMemorySpaceAttr::get(loop.getContext());
  for (auto acquireOp : loop.getOps<SemaphoreAcquireOp>()) {
    auto semaCreate =
        acquireOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    if (getReleasedMask(semaCreate))
      continue;
    bool isTMEM = llvm::any_of(semaCreate.getBuffers(), [&](Value buf) {
      return cast<MemDescType>(buf.getType()).getMemorySpace() == tmem;
    });
    if (isTMEM)
      continue;
    consumerAcquires.push_back(acquireOp);
  }

  // 2. Group by (dominant consumer, partition ID).
  DominanceInfo domInfo(loop);
  llvm::DenseMap<std::pair<Operation *, int>, SmallVector<SemaphoreAcquireOp>>
      groups;
  for (auto acquireOp : consumerAcquires) {
    auto liveBeforeOp =
        getDominantConsumer(acquireOp, *loop.getBody(), domInfo);
    if (!liveBeforeOp)
      continue;
    assert(hasPartition(acquireOp));
    auto partitionIds = getPartitionIds(acquireOp);
    assert(partitionIds.size() == 1);
    groups[{liveBeforeOp, partitionIds.front()}].push_back(acquireOp);
  }

  // 3. Combine each group with size > 1.
  for (auto &[key, acquireGroup] : groups) {
    if (acquireGroup.size() <= 1)
      continue;

    auto groupInfo = analyzeCombinedSemaphoreGroup(acquireGroup);
    if (groupInfo.empty())
      continue;

    SmallVector<SemaphoreReleaseOp> consumerReleases;
    SmallVector<SemaphoreReleaseOp> producerReleases;
    for (SemaToCombineInfo info : groupInfo) {
      consumerReleases.push_back(info.consReleaseOp);
      producerReleases.push_back(info.prodReleaseOp);
    }
    auto consumerPlan = planCombinedRelease(consumerReleases);
    if (failed(consumerPlan))
      return failure();
    auto producerPlan = planCombinedRelease(producerReleases);
    if (failed(producerPlan))
      return failure();

    auto combinedPair = createCombinedSemaphores(groupInfo, loop);
    OpBuilder builder(loop.getContext());
    combineConsumerSide(acquireGroup, groupInfo, combinedPair, *consumerPlan,
                        builder);
    combineProducerSide(groupInfo, combinedPair, *producerPlan, builder);
    eraseSemaToCombineGroup(acquireGroup, groupInfo);
  }
  return success();
}

} // anonymous namespace

class NVWSSemaphoreOptimize
    : public impl::NVWSSemaphoreOptimizeBase<NVWSSemaphoreOptimize> {
  using impl::NVWSSemaphoreOptimizeBase<
      NVWSSemaphoreOptimize>::NVWSSemaphoreOptimizeBase;

public:
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    SmallVector<scf::ForOp> loops;
    m.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(triton::kWarpSpecializeAttrName))
        loop->walk([&](scf::ForOp op) { loops.push_back(op); });
    });
    for (scf::ForOp loop : loops)
      if (failed(combineSemaphores(loop)))
        return signalPassFailure();

    // Combining replaces semaphore.create operations, so form groups only
    // after all combine rewrites have completed.
    llvm::DenseMap<Value, SmallVector<SemaphoreCreateOp>> semaGroups;
    m.walk([&](SemaphoreCreateOp semaOp) {
      semaGroups[semaOp.getBuffers().front()].push_back(semaOp);
    });
    multiBufferSemaphore(std::move(semaGroups), numStages);
  }
};

} // namespace triton
} // namespace mlir
