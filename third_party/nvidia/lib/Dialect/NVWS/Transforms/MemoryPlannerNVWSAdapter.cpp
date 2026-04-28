/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "MemoryPlannerNVWSAdapter.h"
#include "WSUtility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <memory>

#define DEBUG_TYPE "nvws-memory-planner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::nvws::planner {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

static Operation *skipIdxOp(Operation *op) {
  if (auto idx = dyn_cast_or_null<ttg::MemDescIndexOp>(op)) {
    Operation *first = nullptr;
    unsigned numUsers = 0;
    for (Operation *user : idx->getUsers()) {
      first = user;
      ++numUsers;
    }
    if (numUsers <= 1)
      return first;
  }
  return op;
}

static bool isConstFalse(Value v) {
  if (!v)
    return false;
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    Attribute value = constOp.getValue();
    if (auto boolAttr = dyn_cast<BoolAttr>(value))
      return !boolAttr.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(value))
      return intAttr.getInt() == 0;
  }
  return false;
}

static bool isLoopCarriedInitConstFalse(Value v, scf::ForOp forOp) {
  // A loop-carried use_acc flag initialized to false means the first MMA
  // iteration fully overwrites operand D, so it can seed producer tracking.
  auto blockArg = dyn_cast<BlockArgument>(v);
  if (!blockArg || blockArg.getOwner() != forOp.getBody())
    return false;

  unsigned argNum = blockArg.getArgNumber();
  unsigned numInductionVars = forOp.getNumInductionVars();
  if (argNum < numInductionVars)
    return false;

  return isConstFalse(forOp.getInitArgs()[argNum - numInductionVars]);
}

static Value getTmemAllocValue(Operation *allocOp) {
  auto alloc = cast<ttng::TMEMAllocOp>(allocOp);
  return alloc.getResult();
}

static bool isMmaAccumulatorUse(Value allocOrSubview, Operation *user) {
  auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user);
  return mmaOp && mmaOp.getAccumulator() == allocOrSubview;
}

static bool isTmemProducer(Value allocOrSubview, Operation *user) {
  if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user))
    return mmaOp.getAccumulator() == allocOrSubview;
  return isa<ttng::TMEMStoreOp>(user);
}

static SmallVector<int> getTaskIds(Operation *op) {
  return nvws::getAsyncTaskIds(op);
}

static bool needsChannel(int producer, ArrayRef<int> consumers) {
  return !llvm::all_of(consumers, [producer](int consumerId) {
    return consumerId == producer;
  });
}

static SmallVector<int> getUniqueTaskIds(ArrayRef<Operation *> ops) {
  SmallVector<int> taskIds;
  DenseSet<int> seenTaskIds;
  for (Operation *op : ops) {
    for (int id : getTaskIds(op)) {
      if (seenTaskIds.insert(id).second)
        taskIds.push_back(id);
    }
  }
  return taskIds;
}

static void setTmemChannelAttr(Operation *op, int channelId,
                               StringRef attrName) {
  SmallVector<int> ids;
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(attrName))
    ids.append(attr.asArrayRef().begin(), attr.asArrayRef().end());
  ids.push_back(channelId);
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  op->setAttr(attrName, DenseI32ArrayAttr::get(op->getContext(), ids));
}

static Operation *findTmemStartEnd(const TmemDataChannelPost *ch,
                                   StringRef attrName) {
  auto alloc = cast<ttng::TMEMAllocOp>(ch->allocOp);
  for (Operation *usr : alloc.getResult().getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    DenseSet<int> channelIds;
    if (auto attr = user->getAttrOfType<DenseI32ArrayAttr>(attrName)) {
      for (int asyncTaskId : attr.asArrayRef())
        channelIds.insert(asyncTaskId);
      if (channelIds.contains(ch->uniqID))
        return user;
    }
  }
  return nullptr;
}

Operation *TmemDataChannelPost::getSrcOp() const {
  if (explicitSrcOp)
    return explicitSrcOp;

  if (isOperandD)
    return findTmemStartEnd(this, "tmem.start");

  auto alloc = cast<ttng::TMEMAllocOp>(allocOp);
  for (Operation *usr : alloc.getResult().getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    Value producerValue = user == usr ? getTmemAllocValue(allocOp)
                                      : usr->getResult(0);
    if (isTmemProducer(producerValue, user))
      return user;
  }
  return nullptr;
}

static void getAllConsumers(const TmemDataChannelPost *ch,
                            SmallVectorImpl<Operation *> &consumers) {
  auto alloc = cast<ttng::TMEMAllocOp>(ch->allocOp);
  for (Operation *usr : alloc.getResult().getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    Value producerValue = user == usr ? getTmemAllocValue(ch->allocOp)
                                      : usr->getResult(0);
    if (!isTmemProducer(producerValue, user))
      consumers.push_back(user);
  }
}

Operation *TmemDataChannelPost::getDstOp() const {
  if (!explicitDstOps.empty())
    return explicitDstOps.back();

  if (isOperandD)
    return findTmemStartEnd(this, "tmem.end");

  SmallVector<Operation *> allConsumers;
  getAllConsumers(this, allConsumers);
  if (allConsumers.empty())
    return nullptr;
  return allConsumers.back();
}

void TmemDataChannelPost::getDstOps(
    SmallVectorImpl<Operation *> &dsts) const {
  if (!explicitDstOps.empty()) {
    dsts.append(explicitDstOps.begin(), explicitDstOps.end());
    return;
  }

  if (isOperandD) {
    if (Operation *dst = getDstOp())
      dsts.push_back(dst);
    return;
  }
  getAllConsumers(this, dsts);
}

static void
createChannelsForProducers(SmallVectorImpl<Operation *> &currentProds,
                           int producerTaskId, ArrayRef<int> consumerIds,
                           Operation *allocOp, Operation *consumerOp,
                           SmallVectorImpl<
                               std::unique_ptr<TmemDataChannelPost>> &channels,
                           bool isSameIterGuard = false) {
  for (Operation *prod : currentProds) {
    auto channelId = channels.size();
    auto channel = std::make_unique<TmemDataChannelPost>(
        producerTaskId, consumerIds, allocOp, true /*isOperandD*/,
        true /*isOperandDNoAcc*/, channelId);
    channel->isSameIterGuard = isSameIterGuard;
    channels.push_back(std::move(channel));
    setTmemChannelAttr(prod, channelId, "tmem.start");
    setTmemChannelAttr(consumerOp, channelId, "tmem.end");
  }
}

static LogicalResult
handleOperandD(ttng::TMEMAllocOp tmemAllocOp,
               ttng::MMAv5OpInterface representativeMma,
               SmallVectorImpl<std::unique_ptr<TmemDataChannelPost>>
                   &channels) {
  DenseSet<Operation *> users;
  DenseSet<Operation *> handledUsers;
  for (Operation *user : tmemAllocOp.getResult().getUsers())
    users.insert(skipIdxOp(user));

  auto forOp = representativeMma->getParentOfType<scf::ForOp>();
  if (!forOp)
    return representativeMma->emitError(
        "NVWS memory planner expected operand-D MMA inside scf.for");

  SmallVector<Operation *> currentProds;
  SmallVector<int> channelsToUpdate;
  Operation *firstProducer = nullptr;
  Operation *lastConsumer = nullptr;
  unsigned numChannelsCreated = 0;

  for (Operation *user : tmemAllocOp.getResult().getUsers()) {
    Operation *actual = skipIdxOp(user);
    if (auto storeOp = dyn_cast_or_null<ttng::TMEMStoreOp>(actual)) {
      if (!forOp->isProperAncestor(storeOp)) {
        currentProds.clear();
        currentProds.push_back(storeOp);
        handledUsers.insert(storeOp);
      }
    }
  }

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!users.contains(&op))
      continue;
    handledUsers.insert(&op);

    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(&op)) {
      if (mmaOp.getAccumulator() == tmemAllocOp.getResult()) {
        if (currentProds.empty() &&
            (isConstFalse(mmaOp.useAccumulator()) ||
             isLoopCarriedInitConstFalse(mmaOp.useAccumulator(), forOp))) {
          currentProds.push_back(&op);
          continue;
        }
        if (currentProds.empty())
          return op.emitError("NVWS memory planner found no producer for MMA "
                              "operand-D accumulator");

        SmallVector<int> producerIds = getTaskIds(currentProds.front());
        SmallVector<int> consumerIds = getTaskIds(&op);
        if (producerIds.size() != 1) {
          // NVWS may retain several partition IDs on a producer that Meta's
          // monolithic code partitioner represents as one channel owner. Do
          // not invent an owner. When no channel can be modeled, the planner's
          // unknown-channel path keeps the allocation function-live.
          currentProds.push_back(&op);
          continue;
        }

        int producerId = producerIds.front();
        if (needsChannel(producerId, consumerIds)) {
          if (!firstProducer)
            firstProducer = currentProds.front();
          lastConsumer = &op;
          ++numChannelsCreated;
          createChannelsForProducers(currentProds, producerId, consumerIds,
                                     tmemAllocOp.getOperation(), &op,
                                     channels);
          currentProds.clear();
          currentProds.push_back(&op);
        } else {
          currentProds.push_back(&op);
        }
      } else {
        if (currentProds.empty())
          return op.emitError("NVWS memory planner found no producer for "
                              "TMEM MMA consumer");

        SmallVector<int> producerIds = getTaskIds(currentProds.front());
        SmallVector<int> consumerIds = getTaskIds(&op);
        if (producerIds.size() != 1) {
          currentProds.push_back(&op);
          continue;
        }

        int producerId = producerIds.front();
        if (needsChannel(producerId, consumerIds)) {
          if (!firstProducer)
            firstProducer = currentProds.front();
          lastConsumer = &op;
          ++numChannelsCreated;
          createChannelsForProducers(currentProds, producerId, consumerIds,
                                     tmemAllocOp.getOperation(), &op,
                                     channels);
        } else {
          currentProds.push_back(&op);
        }
      }
    } else if (isa<ttng::TMEMStoreOp>(&op)) {
      currentProds.clear();
      currentProds.push_back(&op);
    } else if (isa<ttng::TMEMLoadOp>(&op)) {
      if (!currentProds.empty()) {
        SmallVector<int> producerIds = getTaskIds(currentProds.front());
        SmallVector<int> consumerIds = getTaskIds(&op);
        if (producerIds.size() != 1) {
          currentProds.push_back(&op);
          continue;
        }
        int producerId = producerIds.front();
        if (needsChannel(producerId, consumerIds)) {
          if (!firstProducer)
            firstProducer = currentProds.front();
          lastConsumer = &op;
          ++numChannelsCreated;
          createChannelsForProducers(currentProds, producerId, consumerIds,
                                     tmemAllocOp.getOperation(), &op,
                                     channels);
        } else {
          currentProds.push_back(&op);
        }
      } else {
        unsigned channelId = channels.size();
        channelsToUpdate.push_back(channelId);
        channels.push_back(std::make_unique<TmemDataChannelPost>(
            -1, getTaskIds(&op), tmemAllocOp.getOperation(),
            true /*isOperandD*/, true /*isOperandDNoAcc*/, channelId));
        setTmemChannelAttr(&op, channelId, "tmem.end");
      }
    } else {
      return op.emitError("NVWS memory planner found unsupported TMEM user");
    }
  }

  for (int idx : channelsToUpdate) {
    if (currentProds.empty())
      return representativeMma->emitError(
          "NVWS memory planner found no producer for deferred TMEM channel");
    Operation *lastProd = currentProds.back();
    SmallVector<int> producerIds = getTaskIds(lastProd);
    if (producerIds.size() != 1)
      continue;
    channels[idx]->producer = producerIds.front();
    setTmemChannelAttr(lastProd, channels[idx]->uniqID, "tmem.start");
  }

  for (Operation *user : users) {
    if (handledUsers.contains(user))
      continue;
    if (!isa_and_nonnull<ttng::TMEMLoadOp>(user))
      return user->emitError(
          "NVWS memory planner found unsupported post-loop TMEM user");
    if (currentProds.empty())
      return user->emitError(
          "NVWS memory planner found no producer for post-loop TMEM load");

    SmallVector<int> producerIds = getTaskIds(currentProds.front());
    SmallVector<int> consumerIds = getTaskIds(user);
    if (producerIds.size() != 1)
      continue;
    int producerId = producerIds.front();
    if (needsChannel(producerId, consumerIds)) {
      if (!firstProducer)
        firstProducer = currentProds.front();
      lastConsumer = user;
      ++numChannelsCreated;
      createChannelsForProducers(currentProds, producerId, consumerIds,
                                 tmemAllocOp.getOperation(), user, channels);
    }
  }

  if (numChannelsCreated >= 2 && firstProducer && lastConsumer &&
      firstProducer->getBlock() == lastConsumer->getBlock()) {
    SmallVector<int> firstProducerIds = getTaskIds(firstProducer);
    SmallVector<int> lastConsumerIds = getTaskIds(lastConsumer);
    if (firstProducerIds.size() == 1 &&
        needsChannel(firstProducerIds.front(), lastConsumerIds)) {
      SmallVector<Operation *> producers = {firstProducer};
      createChannelsForProducers(producers, firstProducerIds.front(),
                                 lastConsumerIds, tmemAllocOp.getOperation(),
                                 lastConsumer, channels);
    }

    if (lastConsumerIds.size() == 1 && isa<ttng::TMEMLoadOp>(lastConsumer) &&
        isa<ttng::TMEMStoreOp>(firstProducer) &&
        needsChannel(lastConsumerIds.front(), firstProducerIds)) {
      unsigned channelId = channels.size();
      auto guard = std::make_unique<TmemDataChannelPost>(
          lastConsumerIds.front(), firstProducerIds, tmemAllocOp.getOperation(),
          true /*isOperandD*/, false /*isOperandDNoAcc*/, channelId);
      guard->isSameIterGuard = true;
      channels.push_back(std::move(guard));
      setTmemChannelAttr(lastConsumer, channelId, "tmem.start");
      setTmemChannelAttr(firstProducer, channelId, "tmem.end");
    }
  }

  return success();
}

static LogicalResult
createTmemChannelPost(ttng::TMEMAllocOp alloc,
                      SmallVectorImpl<std::unique_ptr<TmemDataChannelPost>>
                          &channels) {
  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
  ttng::MMAv5OpInterface operandDMma;
  bool isOperandD = false;
  bool isOperandDNoAcc = false;

  for (Operation *usr : alloc.getResult().getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    Value accessValue =
        user == usr ? Value(alloc.getResult()) : Value(usr->getResult(0));
    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      if (mmaOp.getAccumulator() == accessValue) {
        if (user != usr)
          return user->emitError("NVWS memory planner does not support "
                                 "partial-view TMEM producer modeling");
        if (!isConstFalse(mmaOp.useAccumulator())) {
          operandDMma = mmaOp;
          isOperandD = true;
        } else {
          isOperandDNoAcc = true;
          producers.push_back(user);
        }
      } else {
        consumers.push_back(user);
      }
    } else if (isa<ttng::TMEMStoreOp>(user)) {
      if (user != usr)
        return user->emitError("NVWS memory planner does not support "
                               "partial-view TMEM producer modeling");
      producers.push_back(user);
    } else if (isa<ttng::TMEMLoadOp>(user)) {
      consumers.push_back(user);
    } else {
      return user->emitError("NVWS memory planner found unsupported TMEM user");
    }
  }

  if (isOperandD)
    return handleOperandD(alloc, operandDMma, channels);

  if (producers.empty()) {
    if (!alloc.getSrc())
      return success();
    if (consumers.empty())
      return success();

    SmallVector<int> producerIds = getTaskIds(alloc.getOperation());
    if (producerIds.size() != 1)
      return alloc.emitError("NVWS memory planner expected sourceful "
                             "ttng.tmem_alloc to have exactly one partition");

    SmallVector<int> consumerTaskIds = getUniqueTaskIds(consumers);
    consumerTaskIds.erase(std::remove(consumerTaskIds.begin(),
                                      consumerTaskIds.end(),
                                      producerIds.front()),
                          consumerTaskIds.end());

    // Sourceful ttng.tmem_alloc is planner-equivalent to a hoisted storage
    // allocation plus an init store at the real alloc op. Keep that producer
    // record internal to the planner; InsertTmemSemaphore must not see it as an
    // extra real channel.
    channels.push_back(std::make_unique<TmemDataChannelPost>(
        producerIds.front(), consumerTaskIds, alloc.getOperation(),
        alloc.getOperation(), consumers, false /*isOperandD*/,
        false /*isOperandDNoAcc*/, true /*isPlannerOnly*/, channels.size()));
    return success();
  }

  Operation *producerOp = producers.front();
  if (producers.size() > 1 && !consumers.empty()) {
    producerOp = nullptr;
    for (Operation *prod : producers) {
      if (prod->getBlock() == consumers.front()->getBlock()) {
        if (producerOp)
          return prod->emitError(
              "NVWS memory planner found ambiguous TMEM producers");
        producerOp = prod;
      }
    }
    if (!producerOp)
      producerOp = producers.front();
  }

  SmallVector<int> producerIds = getTaskIds(producerOp);
  if (producerIds.size() != 1)
    return success();
  int producerId = producerIds.front();

  SmallVector<int> consumerTaskIds = getUniqueTaskIds(consumers);
  consumerTaskIds.erase(
      std::remove(consumerTaskIds.begin(), consumerTaskIds.end(), producerId),
      consumerTaskIds.end());

  if (needsChannel(producerId, consumerTaskIds)) {
    channels.push_back(std::make_unique<TmemDataChannelPost>(
        producerId, consumerTaskIds, alloc.getOperation(),
        false /*isOperandD*/, isOperandDNoAcc, channels.size()));
  } else if (!consumers.empty()) {
    channels.push_back(std::make_unique<TmemDataChannelPost>(
        producerId, consumerTaskIds, alloc.getOperation(), producerOp,
        consumers, false /*isOperandD*/, isOperandDNoAcc,
        true /*isPlannerOnly*/, channels.size()));
  }

  return success();
}

static LogicalResult collectTmemPostChannels(
    SmallVectorImpl<std::unique_ptr<TmemDataChannelPost>> &channels,
    FuncOp funcOp) {
  WalkResult result = funcOp.walk([&](ttng::TMEMAllocOp alloc) {
    if (failed(createTmemChannelPost(alloc, channels)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static bool isTransparentMemdescViewOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.memdesc_index" || name == "ttg.memdesc_subview" ||
         name == "ttg.memdesc_trans" || name == "ttg.memdesc_reinterpret" ||
         name == "ttg.memdesc_reshape";
}

static void collectLocalMemdescUsers(Value value,
                                     DenseSet<Operation *> &users) {
  SmallVector<Value> worklist = {value};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      Operation *actual = skipIdxOp(user);
      if (!actual)
        continue;
      users.insert(actual);

      if (actual == user && isTransparentMemdescViewOp(actual)) {
        for (Value result : actual->getResults())
          worklist.push_back(result);
        continue;
      }

      if (isa<nvws::DescriptorLoadOp, nvws::DescriptorGatherOp>(actual)) {
        for (Value result : actual->getResults())
          if (isa<ttg::MemDescType>(result.getType()))
            worklist.push_back(result);
      }
    }
  }
}

static bool isLocalProducer(Operation *op) {
  return isa<ttg::LocalStoreOp, nvws::DescriptorLoadOp,
             nvws::DescriptorGatherOp>(op);
}

static Operation *producerForSourcefulLocalAlloc(ttg::LocalAllocOp alloc) {
  if (!alloc.getSrc())
    return nullptr;
  return alloc.getOperation();
}

static LogicalResult createLocalChannelPost(
    ttg::LocalAllocOp alloc,
    SmallVectorImpl<std::unique_ptr<LocalDataChannelPost>> &channels) {
  if (!alloc.isSharedMemoryAlloc())
    return success();

  DenseSet<Operation *> users;
  collectLocalMemdescUsers(alloc.getResult(), users);

  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
  if (Operation *producer = producerForSourcefulLocalAlloc(alloc))
    producers.push_back(producer);

  for (Operation *user : users) {
    if (isTransparentMemdescViewOp(user))
      continue;
    if (isLocalProducer(user)) {
      producers.push_back(user);
      continue;
    }
    consumers.push_back(user);
  }

  if (producers.empty() && consumers.empty())
    return success();

  Operation *producerOp =
      producers.empty() ? alloc.getOperation() : producers.front();
  SmallVector<int> producerIds = getTaskIds(producerOp);
  int producerId = producerIds.size() == 1 ? producerIds.front() : -1;

  SmallVector<int> consumerTaskIds = getUniqueTaskIds(consumers);
  if (producerId >= 0)
    consumerTaskIds.erase(std::remove(consumerTaskIds.begin(),
                                      consumerTaskIds.end(), producerId),
                          consumerTaskIds.end());

  channels.push_back(std::make_unique<LocalDataChannelPost>(
      producerId, consumerTaskIds, alloc.getOperation(), producerOp, consumers,
      channels.size()));
  return success();
}

static LogicalResult collectLocalPostChannels(
    SmallVectorImpl<std::unique_ptr<LocalDataChannelPost>> &channels,
    FuncOp funcOp) {
  WalkResult result = funcOp.walk([&](ttg::LocalAllocOp alloc) {
    if (failed(createLocalChannelPost(alloc, channels)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult collectPostChannels(
    SmallVectorImpl<std::unique_ptr<Channel>> &channels, FuncOp funcOp) {
  SmallVector<std::unique_ptr<TmemDataChannelPost>> tmemChannels;
  if (failed(collectTmemPostChannels(tmemChannels, funcOp)))
    return failure();

  SmallVector<std::unique_ptr<LocalDataChannelPost>> localChannels;
  if (failed(collectLocalPostChannels(localChannels, funcOp)))
    return failure();

  for (auto &channel : tmemChannels)
    channels.push_back(std::move(channel));
  for (auto &channel : localChannels) {
    channel->uniqID += tmemChannels.size();
    channels.push_back(std::move(channel));
  }
  return success();
}

LogicalResult emitSmemPlanAnnotations(
    FuncOp funcOp, ArrayRef<Channel *> channels, int smemAllocAlgo,
    bool smemCircularReuse, const DenseSet<Operation *> &eligibleAllocs) {
  llvm::MapVector<int64_t, SmallVector<ttg::LocalAllocOp>> groups;
  funcOp->walk<WalkOrder::PreOrder>([&](ttg::LocalAllocOp alloc) {
    alloc->removeAttr("buffer.circular");
    alloc->removeAttr("buffer.start");
    if (!alloc.isSharedMemoryAlloc())
      return;
    if (auto id = alloc->getAttrOfType<IntegerAttr>("buffer.id"))
      groups[id.getInt()].push_back(alloc);
  });

  // Algorithm 0 always creates Meta shared-id pools. Algorithm 1 creates a
  // circular reuse group only when Meta's smemCircularReuse policy selected
  // it. Keep that policy bit separate from the NVWS-only representation.
  if (smemAllocAlgo == 1 && !smemCircularReuse)
    return success();

  auto i32Type = IntegerType::get(funcOp.getContext(), 32);
  for (auto &[bufferId, group] : groups) {
    (void)bufferId;
    if (group.size() < 2)
      continue;

    bool selectedReuseGroup = llvm::all_of(group, [&](ttg::LocalAllocOp alloc) {
      return eligibleAllocs.contains(alloc.getOperation());
    });
    if (smemAllocAlgo == 1)
      selectedReuseGroup &= group.size() == 2;
    if (!selectedReuseGroup)
      continue;

    // Meta skips physical SMEM folding for unlike memdesc types. Preserve the
    // planner id but leave that fallback group non-circular.
    Type groupType = group.front().getType();
    SmallVector<std::pair<Operation *, ttg::LocalAllocOp>> ordered;
    Block *producerBlock = nullptr;
    for (ttg::LocalAllocOp alloc : group) {
      if (alloc.getType() != groupType) {
        ordered.clear();
        break;
      }
      Channel *channel = nullptr;
      for (Channel *candidate : channels) {
        if (candidate->getAllocOp() == alloc) {
          channel = candidate;
          break;
        }
      }
      Operation *producer = channel ? channel->getSrcOp() : nullptr;
      if (!producer ||
          (producerBlock && producer->getBlock() != producerBlock)) {
        ordered.clear();
        break;
      }
      producerBlock = producer->getBlock();
      ordered.emplace_back(producer, alloc);
    }
    if (ordered.size() != group.size())
      continue;

    llvm::sort(ordered, [](const auto &lhs, const auto &rhs) {
      return lhs.first->isBeforeInBlock(rhs.first);
    });
    for (auto [start, item] : llvm::enumerate(ordered)) {
      ttg::LocalAllocOp alloc = item.second;
      alloc->setAttr("buffer.circular", UnitAttr::get(funcOp.getContext()));
      alloc->setAttr("buffer.start", IntegerAttr::get(i32Type, start));
    }
  }

  for (auto &[bufferId, group] : groups) {
    auto firstCopy = group.front()->getAttrOfType<IntegerAttr>("buffer.copy");
    if (!firstCopy)
      return group.front().emitError(
          "NVWS memory planner omitted buffer.copy from planned local group");

    SmallVector<ttg::LocalAllocOp> circular;
    DenseSet<int64_t> starts;
    for (ttg::LocalAllocOp alloc : group) {
      auto copy = alloc->getAttrOfType<IntegerAttr>("buffer.copy");
      if (!copy || copy.getInt() != firstCopy.getInt())
        return alloc.emitError()
               << "NVWS memory planner assigned inconsistent buffer.copy "
                  "values to local buffer.id "
               << bufferId;
      if (!alloc->hasAttr("buffer.circular"))
        continue;
      circular.push_back(alloc);
      auto start = alloc->getAttrOfType<IntegerAttr>("buffer.start");
      if (!start || start.getInt() < 0 ||
          start.getInt() >= firstCopy.getInt())
        return alloc.emitError(
            "NVWS circular local allocation has invalid buffer.start");
      if (!starts.insert(start.getInt()).second)
        return alloc.emitError(
            "NVWS circular local group has duplicate buffer.start");
    }

    if (circular.empty())
      continue;
    if (circular.size() != group.size())
      return circular.front().emitError(
          "NVWS memory planner partially marked a local group circular");
    if (smemAllocAlgo == 1 &&
        (circular.size() != 2 || !starts.contains(0) || !starts.contains(1)))
      return circular.front().emitError(
          "NVWS algorithm-1 circular local group must have two starts");
    if (smemAllocAlgo == 0 &&
        firstCopy.getInt() < static_cast<int64_t>(circular.size()))
      return circular.front().emitError(
          "NVWS algorithm-0 circular local group has fewer copies than "
          "members");
  }
  return success();
}

void emitTmemOwnerOffsets(FuncOp funcOp) {
  auto i32Type = IntegerType::get(funcOp.getContext(), 32);
  funcOp->walk([&](ttng::TMEMAllocOp alloc) {
    if (alloc->hasAttr("buffer.id") && !alloc->hasAttr("buffer.offset"))
      alloc->setAttr("buffer.offset", IntegerAttr::get(i32Type, 0));
  });
}

Operation *getSameLevelOp(Operation *producer, Operation *consumer) {
  Operation *op = consumer;
  while (!isa<FuncOp>(op)) {
    if (op->getParentOp() == producer->getParentOp())
      return op;
    op = op->getParentOp();
  }

  op = producer;
  while (!isa<FuncOp>(op)) {
    if (consumer->getParentOp() == op->getParentOp())
      return consumer;
    op = op->getParentOp();
  }
  return nullptr;
}

SmallVector<Operation *> getActualConsumers(Operation *consumerOp) {
  auto collectThroughTranspose = [&](Operation *root) {
    DenseSet<Operation *> users;
    DenseSet<Operation *> visited;
    SmallVector<Operation *> worklist = {root};
    while (!worklist.empty()) {
      Operation *user = worklist.pop_back_val();
      if (!visited.insert(user).second)
        continue;
      if (isa<tt::TransOp, ttg::MemDescTransOp>(user)) {
        worklist.append(user->getUsers().begin(), user->getUsers().end());
      } else {
        users.insert(user);
      }
    }
    return users;
  };

  if (isa<ttg::MemDescTransOp>(consumerOp)) {
    auto users = collectThroughTranspose(consumerOp);
    return SmallVector<Operation *>(users.begin(), users.end());
  }

  if (isa<ttg::LocalAllocOp>(consumerOp)) {
    DenseSet<Operation *> users;
    for (Operation *user : consumerOp->getUsers()) {
      if (isa<tt::TransOp, ttg::MemDescTransOp>(user)) {
        auto transUsers = collectThroughTranspose(user);
        users.insert(transUsers.begin(), transUsers.end());
      } else {
        users.insert(user);
      }
    }
    return SmallVector<Operation *>(users.begin(), users.end());
  }
  return {consumerOp};
}

bool hasLoopCarriedAccToken(Operation *tmemAlloc, scf::ForOp forOp) {
  for (Operation *directUser : tmemAlloc->getResult(0).getUsers()) {
    Operation *user = skipIdxOp(directUser);
    auto mmaOp = dyn_cast_or_null<ttng::MMAv5OpInterface>(user);
    if (!mmaOp || !forOp->isProperAncestor(mmaOp))
      continue;

    auto blockArg = dyn_cast_or_null<BlockArgument>(mmaOp.getAccDep());
    if (!blockArg || blockArg.getOwner() != forOp.getBody())
      continue;
    unsigned argIdx = blockArg.getArgNumber() - forOp.getNumInductionVars();
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value token = mmaOp.getToken();
    if (token && yieldOp.getOperand(argIdx) == token)
      return true;
  }
  return false;
}

void dumpCombinedGraph(SmallVector<std::unique_ptr<Channel>> &, FuncOp,
                       llvm::raw_ostream &) {}

void dumpTmemBufferLiveness(
    SmallVector<ttng::TMEMAllocOp> &,
    DenseMap<Operation *, Interval<size_t>> &,
    DenseMap<Operation *, ttng::TMemAllocation> &,
    DenseMap<Operation *, TmemDataChannelPost *> &, SmallVector<Channel *> &,
    llvm::raw_ostream &) {}

void dumpSmemBufferLiveness(
    llvm::MapVector<Allocation::BufferId, std::pair<Interval<size_t>, size_t>> &,
    DenseMap<Allocation::BufferId, Operation *> &, SmallVector<Channel *> &,
    llvm::raw_ostream &) {}

} // namespace mlir::triton::nvws::planner
