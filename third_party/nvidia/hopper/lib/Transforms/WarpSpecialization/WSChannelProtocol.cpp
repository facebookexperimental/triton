//===- WSChannelProtocol.cpp - Planned channel endpoints -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WSChannelProtocol.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <tuple>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace ttnvws = mlir::triton::nvws;

namespace mlir {

static Operation *getEffectiveProtocolParent(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent && isa<ttng::SubtiledRegionOp>(parent))
    parent = parent->getParentOp();
  return parent;
}

static Operation *getCadenceParentOutsideSubtiledRegions(Operation *op) {
  Operation *parent = op->getParentOp();
  while (auto subtiled = op->getParentOfType<ttng::SubtiledRegionOp>()) {
    parent = subtiled->getParentOp();
    op = subtiled;
  }
  return parent;
}

Operation *getProtocolSameLevelOp(Operation *producer, Operation *consumer) {
  Operation *op = consumer;
  while (!isa<triton::FuncOp>(op)) {
    if (getEffectiveProtocolParent(op) ==
        getEffectiveProtocolParent(producer)) {
      while (auto subtiled = op->getParentOfType<ttng::SubtiledRegionOp>()) {
        if (subtiled->getParentOp() == getEffectiveProtocolParent(producer))
          op = subtiled;
        else
          break;
      }
      return op;
    }
    op = op->getParentOp();
  }

  op = producer;
  while (!isa<triton::FuncOp>(op)) {
    if (getEffectiveProtocolParent(consumer) ==
        getEffectiveProtocolParent(op)) {
      while (auto subtiled =
                 consumer->getParentOfType<ttng::SubtiledRegionOp>()) {
        if (subtiled->getParentOp() == getEffectiveProtocolParent(op))
          consumer = subtiled.getOperation();
        else
          break;
      }
      return consumer;
    }
    op = op->getParentOp();
  }
  llvm_unreachable("failed to find same-level channel protocol endpoint");
}

const ChannelConsumerProtocolPlan *
ChannelProtocolPlan::findConsumer(AsyncTaskId task) const {
  auto it = llvm::find_if(consumers, [task](const auto &consumer) {
    return consumer.task == task;
  });
  return it == consumers.end() ? nullptr : &*it;
}

bool taskUsesOnlyGen5Consumers(ArrayRef<Channel *> channels,
                               AsyncTaskId consumerTaskId) {
  bool foundConsumer = false;
  for (Channel *channel : channels) {
    if (!llvm::is_contained(channel->relation.second, consumerTaskId))
      continue;

    SmallVector<Operation *> dstOps;
    if (channel->channelKind == DataChannelKind::SMEMAlloc)
      static_cast<AllocChannel *>(channel)->getDstOps(dstOps);
    else
      dstOps.push_back(channel->getDstOp());

    for (Operation *dst : dstOps) {
      for (Operation *consumer : getActualConsumers(dst)) {
        if (!llvm::is_contained(getAsyncTaskIds(consumer), consumerTaskId))
          continue;
        foundConsumer = true;
        if (!isa<ttng::MMAv5OpInterface>(consumer))
          return false;
      }
    }
  }
  assert(foundConsumer && "expected a consumer for the channel task");
  return true;
}

static Operation *findFirstProtocolOp(const DenseSet<Operation *> &ops,
                                      Block *block) {
  for (Operation &op : block->getOperations())
    if (ops.contains(&op))
      return &op;
  return nullptr;
}

static Operation *findLastProtocolOp(const DenseSet<Operation *> &ops,
                                     Block *block) {
  for (Operation &op : llvm::reverse(block->getOperations()))
    if (ops.contains(&op))
      return &op;
  return nullptr;
}

Operation *getProtocolConsumerReleaseAnchor(PostDominanceInfo &postDominance,
                                            Operation *producer,
                                            Operation *consumer,
                                            AsyncTaskId consumerTask) {
  if (consumer->getBlock() != producer->getBlock())
    return getProtocolSameLevelOp(producer, consumer);

  auto actualConsumers = getActualConsumers(consumer);
  std::unordered_set<Operation *> mutuallyNonDominatingUsers;
  for (Operation *user : actualConsumers) {
    auto it = mutuallyNonDominatingUsers.begin();
    while (it != mutuallyNonDominatingUsers.end()) {
      if (postDominance.properlyPostDominates(user, *it)) {
        it = mutuallyNonDominatingUsers.erase(it);
      } else if (postDominance.properlyPostDominates(*it, user)) {
        break;
      } else {
        ++it;
      }
    }
    if (it == mutuallyNonDominatingUsers.end())
      mutuallyNonDominatingUsers.insert(user);
  }

  if (mutuallyNonDominatingUsers.size() == 1) {
    Operation *user = *mutuallyNonDominatingUsers.begin();
    while (user && user->getParentOp() != consumer->getParentOp())
      user = user->getParentOp();
    assert(user && "failed to find common consumer parent");
    return user;
  }

  for (Operation &op : llvm::reverse(consumer->getBlock()->getOperations())) {
    auto taskIds = getAsyncTaskIds(&op);
    if (taskIds.size() == 1 && taskIds[0] == consumerTask)
      return &op;
  }
  return nullptr;
}

struct ProtocolCadence {
  ChannelProtocolCadence kind = ChannelProtocolCadence::Unsupported;
  Operation *scope = nullptr;
  Operation *innerScope = nullptr;
};

static ProtocolCadence classifyProtocolCadence(Operation *producer,
                                               Operation *consumer) {
  auto producerFor = producer->getParentOfType<scf::ForOp>();
  auto consumerFor = consumer->getParentOfType<scf::ForOp>();
  if (producerFor && producerFor == consumerFor)
    return {ChannelProtocolCadence::Loop, producerFor.getOperation()};
  if (!producerFor && consumerFor &&
      producer->getBlock() == consumerFor->getBlock())
    return {ChannelProtocolCadence::OutsideToInnerLoop,
            consumerFor->getParentOp(), consumerFor.getOperation()};
  if (producerFor && !consumerFor &&
      producerFor->getBlock() == consumer->getBlock())
    return {ChannelProtocolCadence::InnerToOutsideLoop,
            producerFor->getParentOp(), producerFor.getOperation()};
  if (producerFor && consumerFor && producerFor->isProperAncestor(consumerFor))
    return {ChannelProtocolCadence::OuterToInnerLoop,
            producerFor.getOperation(), consumerFor.getOperation()};
  if (producerFor && consumerFor && consumerFor->isProperAncestor(producerFor))
    return {ChannelProtocolCadence::InnerToOuterLoop,
            consumerFor.getOperation(), producerFor.getOperation()};

  auto producerWhile = producer->getParentOfType<scf::WhileOp>();
  auto consumerWhile = consumer->getParentOfType<scf::WhileOp>();
  if (producerWhile && producerWhile == consumerWhile)
    return {ChannelProtocolCadence::WhileLoop, producerWhile.getOperation()};

  if (!producerFor && !consumerFor && !producerWhile && !consumerWhile &&
      getCadenceParentOutsideSubtiledRegions(producer) ==
          getCadenceParentOutsideSubtiledRegions(consumer))
    return {ChannelProtocolCadence::StraightLine,
            getCadenceParentOutsideSubtiledRegions(producer)};
  return {};
}

ChannelProtocolPlan buildChannelProtocolPlan(ArrayRef<Channel *> channels,
                                             PostDominanceInfo &postDominance) {
  assert(!channels.empty() && "expected a non-empty channel consumer group");
  ChannelProtocolPlan plan;
  plan.masterChannel = channels.front();
  plan.channels.append(channels.begin(), channels.end());
  plan.copies = plan.masterChannel->getNumBuffers();

  DenseSet<Operation *> producerOps;
  for (Channel *channel : channels) {
    if (Operation *producer = channel->getSrcOp())
      producerOps.insert(producer);
    if (channel->channelKind == DataChannelKind::SMEMAlloc) {
      SmallVector<Operation *> dstOps;
      static_cast<AllocChannel *>(channel)->getDstOps(dstOps);
      for (Operation *dst : dstOps) {
        plan.consumerOps.insert(dst);
        for (Operation *consumer : getActualConsumers(dst)) {
          plan.consumerOps.insert(consumer);
          plan.actualConsumerOps.insert(consumer);
        }

        if (!isa<ttg::LocalLoadOp>(dst))
          continue;
        for (Operation *user : dst->getUsers()) {
          while (isa<ttg::ConvertLayoutOp>(user) && user->hasOneUse())
            user = *user->getUsers().begin();
          if (isa<tt::DescriptorStoreOp, ttng::AsyncTMACopyLocalToGlobalOp,
                  ttng::AsyncTMAReduceOp>(user)) {
            plan.consumerOps.insert(user);
            plan.actualConsumerOps.insert(user);
          }
        }
      }
    } else if (Operation *dst = channel->getDstOp()) {
      plan.consumerOps.insert(dst);
      auto actualConsumers = getActualConsumers(dst);
      Operation *actualConsumer =
          actualConsumers.size() == 1 ? actualConsumers.front() : dst;
      plan.consumerOps.insert(actualConsumer);
      plan.actualConsumerOps.insert(actualConsumer);
    }

    if (Operation *producer = channel->getSrcOp())
      if (isa<ttnvws::DescriptorLoadOp>(producer))
        plan.tmaProducers.push_back(producer);
  }

  SmallVector<Operation *> additionalConsumers;
  for (Operation *consumer : plan.actualConsumerOps) {
    if (!isa<ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAReduceOp>(
            consumer))
      continue;
    for (Operation *user : consumer->getUsers())
      if (isa<ttng::TMAStoreTokenWaitOp>(user))
        additionalConsumers.push_back(user);
  }
  for (Operation *consumer : additionalConsumers) {
    plan.consumerOps.insert(consumer);
    plan.actualConsumerOps.insert(consumer);
  }

  Operation *frontProducer = channels.front()->getSrcOp();
  Operation *frontConsumer = channels.front()->getDstOp();
  if (!frontProducer || !frontConsumer)
    return plan;
  plan.headProducer =
      findFirstProtocolOp(producerOps, frontProducer->getBlock());
  plan.tailProducer =
      findLastProtocolOp(producerOps, frontProducer->getBlock());
  plan.headConsumer =
      findFirstProtocolOp(plan.consumerOps, frontConsumer->getBlock());
  plan.tailConsumer =
      findLastProtocolOp(plan.consumerOps, frontConsumer->getBlock());
  if (!plan.headProducer || !plan.tailProducer || !plan.headConsumer ||
      !plan.tailConsumer)
    return plan;

  DenseSet<Operation *> tmaAndHead(plan.tmaProducers.begin(),
                                   plan.tmaProducers.end());
  tmaAndHead.insert(plan.headProducer);
  plan.tmaHeadProducer =
      findFirstProtocolOp(tmaAndHead, plan.headProducer->getBlock());
  plan.producerAcquireAnchor =
      getProtocolSameLevelOp(plan.headConsumer, plan.tmaHeadProducer);
  plan.producerReadyAnchor =
      getProtocolSameLevelOp(plan.headConsumer, plan.tailProducer);
  // A TMA ready event and an MMAv5 completion event are both asynchronous:
  // their issue point does not serialize later task instructions. This was
  // immaterial while the planned validator admitted SMEM only, but is required
  // when modeling A5 TMEM channels produced by MMAv5.
  plan.producerReadyIsAsync = !plan.tmaProducers.empty() ||
                              isa<ttng::MMAv5OpInterface>(plan.tailProducer);
  plan.tmaConsumerWaitAnchor =
      getProtocolSameLevelOp(plan.tmaHeadProducer, plan.headConsumer);
  ProtocolCadence cadence =
      classifyProtocolCadence(plan.headProducer, plan.headConsumer);
  plan.cadence = cadence.kind;
  plan.cadenceScope = cadence.scope;
  plan.innerCadenceScope = cadence.innerScope;

  SmallVector<AsyncTaskId> consumerTasks;
  for (Channel *channel : channels)
    for (AsyncTaskId task : channel->relation.second)
      if (!llvm::is_contained(consumerTasks, task))
        consumerTasks.push_back(task);
  llvm::sort(consumerTasks);
  for (AsyncTaskId task : consumerTasks) {
    Operation *head = plan.headConsumer;
    Operation *tail = plan.tailConsumer;
    for (Operation &op : plan.headConsumer->getBlock()->getOperations()) {
      if (!plan.consumerOps.contains(&op))
        continue;
      auto taskIds = getAsyncTaskIds(&op);
      if (llvm::is_contained(taskIds, task)) {
        head = &op;
        break;
      }
    }
    for (Operation &op :
         llvm::reverse(plan.tailConsumer->getBlock()->getOperations())) {
      if (!plan.consumerOps.contains(&op))
        continue;
      auto taskIds = getAsyncTaskIds(&op);
      if (llvm::is_contained(taskIds, task)) {
        tail = &op;
        break;
      }
    }
    Operation *waitAnchor = getProtocolSameLevelOp(plan.headProducer, head);
    auto actualConsumers = getActualConsumers(waitAnchor);
    Operation *waitScheduleAnchor =
        actualConsumers.size() == 1 ? actualConsumers.front() : waitAnchor;
    plan.consumers.push_back({task, head, tail, waitAnchor, waitScheduleAnchor,
                              getProtocolConsumerReleaseAnchor(
                                  postDominance, plan.tailProducer, tail, task),
                              taskUsesOnlyGen5Consumers(channels, task)});
  }
  return plan;
}

StagingReuseProtocolPlan
buildStagingReuseProtocolPlan(triton::FuncOp funcOp,
                              ArrayRef<Channel *> orderedChannels) {
  StagingReuseProtocolPlan plan;
  DenseMap<int64_t, Channel *> bufferIdToChannel;
  for (Channel *channel : orderedChannels) {
    Operation *alloc = channel->getAllocOp();
    if (!alloc)
      continue;
    if (auto bufferId = alloc->getAttrOfType<IntegerAttr>("buffer.id"))
      bufferIdToChannel[bufferId.getInt()] = channel;
  }

  struct ReusePair {
    int64_t targetBufferId;
    AsyncTaskId drainedStoreTask;
    Operation *firstStore;
  };
  SmallVector<ReusePair> pairs;
  auto unsupported = [&](StringRef reason) {
    plan.complete = false;
    if (plan.unsupportedReason.empty())
      plan.unsupportedReason = reason.str();
  };

  funcOp.walk([&](ttg::LocalAllocOp alloc) {
    auto reuseTarget =
        alloc->getAttrOfType<IntegerAttr>("allocation.reuseTarget");
    if (!reuseTarget)
      return;
    ++plan.reuseTargetCount;
    plan.affectedBufferIds.insert(reuseTarget.getInt());

    auto staging = alloc->getAttrOfType<IntegerAttr>("buffer.tmaStaging");
    auto bufferId = alloc->getAttrOfType<IntegerAttr>("buffer.id");
    if (bufferId)
      plan.affectedBufferIds.insert(bufferId.getInt());
    if (!staging || !bufferId) {
      unsupported("staging reuse is missing buffer metadata");
      return;
    }

    Operation *firstStore = nullptr;
    SmallVector<AsyncTaskId> drainedStoreTasks;
    for (Operation *user : alloc->getUsers()) {
      if (isa<ttg::LocalStoreOp>(user)) {
        if (!firstStore) {
          firstStore = user;
        } else if (user->getBlock() != firstStore->getBlock()) {
          unsupported("staging reuse stores span multiple blocks");
          return;
        } else if (user->isBeforeInBlock(firstStore)) {
          firstStore = user;
        }
      }
      if (isa<ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAReduceOp>(
              user)) {
        auto taskIds = getAsyncTaskIds(user);
        drainedStoreTasks.append(taskIds.begin(), taskIds.end());
      }
    }
    if (!firstStore || drainedStoreTasks.empty() ||
        !llvm::all_of(drainedStoreTasks, [&](AsyncTaskId task) {
          return task == drainedStoreTasks.front();
        })) {
      unsupported(
          "staging reuse requires one store block and drained-store task");
      return;
    }
    pairs.push_back(
        {reuseTarget.getInt(), drainedStoreTasks.front(), firstStore});
  });

  LoopLikeOpInterface outerLoop;
  std::optional<bool> finiteSingleTile;
  for (const ReusePair &pair : pairs) {
    auto target = bufferIdToChannel.find(pair.targetBufferId);
    if (target == bufferIdToChannel.end() || !target->second->getSrcOp()) {
      unsupported("staging reuse target has no operand-load channel");
      continue;
    }
    LoopLikeOpInterface loop = getParentPersistentLoop(pair.firstStore);
    auto loadTasks = getAsyncTaskIds(target->second->getSrcOp());
    if (loadTasks.empty() || !llvm::all_of(loadTasks, [&](AsyncTaskId task) {
          return task == loadTasks.front();
        })) {
      unsupported("staging reuse lacks one load task");
      continue;
    }

    bool pairFiniteSingleTile = !loop;
    AsyncTaskId loadTask = loadTasks.front();
    ++plan.matchedPairCount;
    if (plan.matchedPairCount == 1) {
      outerLoop = loop;
      finiteSingleTile = pairFiniteSingleTile;
      plan.loadTask = loadTask;
      plan.drainedStoreTask = pair.drainedStoreTask;
      plan.diagnosticChannelId = target->second->uniqID;
    } else {
      plan.diagnosticChannelId =
          std::min(plan.diagnosticChannelId, target->second->uniqID);
      if (pairFiniteSingleTile != *finiteSingleTile || loop != outerLoop ||
          loadTask != plan.loadTask ||
          pair.drainedStoreTask != plan.drainedStoreTask)
        plan.consistent = false;
    }
  }

  if (!plan.hasReuseTargets())
    return plan;
  if (!plan.hasMatchedPairs()) {
    unsupported("staging reuse has no matched operand channel");
    return plan;
  }
  if (!plan.consistent) {
    if (plan.unsupportedReason.empty())
      plan.unsupportedReason =
          "staging reuse pairs span multiple loops or tasks";
    return plan;
  }

  plan.finiteSingleTile = *finiteSingleTile;
  if (plan.finiteSingleTile)
    return plan;

  plan.outerLoop = outerLoop.getOperation();
  Block *body = getPersistentLoopBody(outerLoop);
  if (!body) {
    unsupported("staging reuse has no persistent-loop body");
    return plan;
  }
  plan.acquireAnchor = &body->front();
  plan.releaseAnchor = body->getTerminator();
  return plan;
}

} // namespace mlir
