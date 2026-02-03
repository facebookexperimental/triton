#include "CodePartitionUtility.h"
#include "TMEMUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/MapVector.h"
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace ttnvws = ::mlir::triton::nvws;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-code-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static unsigned getNumBuffersOrDefault(scf::ForOp forOp, unsigned numBuffers) {
  // Use the attribute attached to the loop if it exists otherwise use the
  // global control.
  if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
    return numBuffers;
  return mlir::cast<IntegerAttr>(
             forOp->getAttr(mlir::triton::kNumStagesAttrName))
      .getInt();
}

// Get the bufferIdx and phase for the last iteration of the immediate scope.
std::pair<Value, Value>
getOutOfScopeBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                               Operation *op, unsigned numBuffers,
                               const DenseSet<Operation *> &regionsWithChannels,
                               ReuseConfig *config, int reuseGroupIdx) {
  // Get the current in-scope accumulation count for op.
  Value accumCnt =
      getAccumCount(builder, op, regionsWithChannels, config, reuseGroupIdx);

  // Get the out-of-scope accumulation count.
  assert(isa<BlockArgument>(accumCnt) &&
         "Expected accumCnt to be a block argument");
  auto bbArg = dyn_cast<BlockArgument>(accumCnt);
  Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
  if (auto forOp = dyn_cast<scf::ForOp>(bbAargOwner)) {
    accumCnt = forOp.getResult(bbArg.getArgNumber() - 1);
  } else {
    llvm_unreachable("Unexpected block argument owner");
  }

  // The accumulation count is one past the last iteration. Subtract one to get
  // the last valid iteration index.
  auto loc = bbAargOwner->getLoc();
  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
  accumCnt = builder.createWithAsyncTaskIds<arith::SubIOp>(loc, accumCnt, one);

  return getBufferIdxAndPhase(builder, op->getLoc(), accumCnt, numBuffers);
}

// Find transitive users of the root op. Track through control flow ops (such as
// yield) to get to the real users.
void getTransitiveUsers(Value root,
                        SetVector<std::pair<Operation *, unsigned>> &users) {
  for (Operation *userOp : root.getUsers()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(userOp)) {
      for (OpOperand &operand : yieldOp->getOpOperands()) {
        if (operand.get() == root) {
          auto result =
              yieldOp->getParentOp()->getResult(operand.getOperandNumber());
          getTransitiveUsers(result, users);
        }
      }
    } else {
      // find operand index of root
      unsigned operandIndex = 0;
      for (OpOperand &operand : userOp->getOpOperands()) {
        if (operand.get() == root) {
          break;
        }
        operandIndex++;
      }
      assert(operandIndex < userOp->getNumOperands() &&
             "root is not an operand of userOp");
      users.insert({userOp, operandIndex});
    }
  }
}

// When traversing gen5, producerOp can be either the defining op of operand
// A or the accumulator.
static void createChannel(Operation *producerOp, mlir::DominanceInfo &dom,
                          SmallVector<std::unique_ptr<Channel>> &channels,
                          bool opndAOfGen5, unsigned producerNumBuffers) {
  // For TMEM channels, op is Gen5 op, producerOp can be either A operand
  // or accumulator.
  auto producerTaskIds = getAsyncTaskIds(producerOp);
  auto producerTaskId = producerTaskIds.front();
  for (auto result : producerOp->getResults()) {
    if (result.use_empty()) {
      continue;
    }

    SetVector<std::pair<Operation *, unsigned>> users;
    getTransitiveUsers(result, users);
    LDBG("getTransitiveUsers returns " << users.size());
    for (auto user : users) {
      auto userOp = user.first;
      if (producerOp == userOp && !opndAOfGen5)
        continue;
      // rule out users that are not dominated by op
      if (producerOp->getBlock() != userOp->getBlock()) {
        if (!dom.properlyDominates(producerOp->getParentOp(), userOp)) {
          continue;
        }
      } else {
        if (!dom.properlyDominates(producerOp, userOp) && producerOp != userOp)
          continue;
      }

      auto consumerTaskIds = getAsyncTaskIds(userOp);
      if (consumerTaskIds.empty())
        continue;
      // Remove producer task id from consumerTaskIds.
      auto iter = std::remove(consumerTaskIds.begin(), consumerTaskIds.end(),
                              producerTaskId);
      consumerTaskIds.erase(iter, consumerTaskIds.end());

      // Add a channel from the single producer task to consumerTaskIds.
      if (consumerTaskIds.size() > 0) {
        DataChannelKind channelKind = DataChannelKind::SMEM;
        if (isa<ttng::TMEMAllocOp, ttng::TMEMStoreOp, ttng::TCGen5MMAOp>(
                producerOp)) {
          channelKind = DataChannelKind::TMEM;
        } else if (auto tAllocOp = dyn_cast<ttg::LocalAllocOp>(producerOp)) {
          channelKind = DataChannelKind::SMEM;
        } else {
          channelKind = DataChannelKind::REG;
        }

        if (isa<scf::ForOp>(userOp)) {
          LDBG("createChannel with dstOp ForOp: producerId "
               << producerTaskId << " number of consumerIds "
               << consumerTaskIds.size());
          auto *tSrc = userOp->getOperand(user.second).getDefiningOp();
          if (isa<scf::ForOp>(tSrc)) {
            LDBG("createChannel with srcOp ForOp");
            continue;
          }
        }
        channels.push_back(std::make_unique<Channel>(
            producerTaskId, consumerTaskIds, userOp, user.second,
            producerNumBuffers, channels.size(), channelKind));
      }
    }
  }
}

// Can be one end of the channel.
static bool isChannelAnchorOp(Operation *op) {
  if (isa<tt::LoadOp, tt::DescriptorLoadOp>(op) ||
      isa<mlir::triton::DotOpInterface, ttng::TMEMStoreOp>(op))
    return true;
  // Local alloc op with a register operand can be the producer of a channel.
  if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
    if (allocOp.getSrc())
      return true;
  }
  if (auto allocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
    if (allocOp.getSrc())
      return true;
  }
  // Any computation tensor op?
  if (dyn_cast<arith::ConstantOp>(op) || dyn_cast<scf::IfOp>(op) ||
      dyn_cast<scf::ForOp>(op))
    return false;
  for (auto result : op->getResults()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(result.getType()))
      return true;
  }
  return false;
}

// Loads will be in producer warp groups. For now, we only allow a single
// warp group/task for a producer. For each LoadOp, create a channel from it
// to any direct user which belongs to a different taskId.
void collectAsyncChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                          triton::FuncOp &funcOp, unsigned numBuffers) {
  mlir::DominanceInfo dom(funcOp);
  funcOp.walk([&](Operation *producerOp) {
    // FIXME: It is possible that a local_alloc can start a channel, when a
    // gemm's operand is in smem and comes from local_alloc.
    if (isChannelAnchorOp(producerOp)) {
      auto producerTaskIds = getAsyncTaskIds(producerOp);
      if (producerTaskIds.empty() || producerTaskIds.size() > 1) {
        LLVM_DEBUG({
          LDBG(" ignoring ops without async task id or with multiple task "
               "ids: ");
          producerOp->dump();
        });
        return;
      }
      auto producerTaskId = producerTaskIds.front();
      unsigned producerNumBuffers = numBuffers;
      if (auto forOp = producerOp->getParentOfType<scf::ForOp>()) {
        producerNumBuffers = getNumBuffersOrDefault(forOp, numBuffers);
      }

      // If the consumer is in a different task, create a channel.
      createChannel(producerOp, dom, channels, false, producerNumBuffers);
    }
  });

  LLVM_DEBUG({
    LDBG("\n\n");
    LDBG(channels.size() << " async channels:");
    for (unsigned i = 0; i < channels.size(); i++) {
      const auto &channel = channels[i];
      LDBG("channel [" << i << "]  " << to_string(channel->channelKind));
      LDBG("producer op: " << channel->relation.first);
      channel->getSrcOp()->dump();
      for (auto &asyncTaskId : channel->relation.second)
        LDBG("consumer: " << asyncTaskId);
      channel->getDstOp()->dump();
      LDBG("numBuffers: " << channel->getNumBuffers() << "\n");
    }
  });
}

static Operation *getUniqueActualConsumer(Operation *consumerOp) {
  auto consumers = getActualConsumers(consumerOp);
  return consumers.size() == 1 ? consumers[0] : consumerOp;
}

static Operation *getUniqueActualConsumer(Operation *consumerOp,
                                          AsyncTaskId taskId) {
  auto consumers = getActualConsumers(consumerOp);
  if (consumers.size() == 1)
    return consumers[0];
  // Check to see if there is only one consumer with the specific taskId.
  Operation *uniqOp = nullptr;
  for (auto *op : consumers) {
    SmallVector<AsyncTaskId> asyncTasks = getAsyncTaskIds(op);
    assert(asyncTasks.size() > 0);
    if (asyncTasks.size() > 1)
      return consumerOp;
    if (asyncTasks[0] == taskId) {
      if (uniqOp)
        return consumerOp;
      uniqOp = op;
    }
  }
  return uniqOp ? uniqOp : consumerOp;
}

static Operation *getLastOpInBlock(DenseSet<Operation *> &ops) {
  Operation *tailConsumer = nullptr;
  Operation *first = *(ops.begin());
  auto cBlock = first->getParentOp();
  bool inOneBlock = true;
  DenseSet<Operation *> blocks;
  for (auto *op : ops) {
    blocks.insert(op->getParentOp());
    if (op->getParentOp() != cBlock) {
      inOneBlock = false;
      break;
    }
  }
  if (inOneBlock) {
    assert(isa<scf::ForOp>(cBlock));
    scf::ForOp cFor = cast<scf::ForOp>(cBlock);
    for (auto &op : reverse(cFor.getBody()->getOperations())) {
      if (ops.count(&op)) {
        tailConsumer = &op;
        break;
      }
    }
    return tailConsumer;
  }
  // Handle ops in different blocks: find the last op in the last block.
  // find the last block in blocks
  auto *lastB = *(blocks.begin());
  for (auto *block : blocks) {
    if (block == lastB)
      continue;
    if (appearsBefore(lastB, block))
      lastB = block;
  }
  assert(isa<scf::ForOp>(lastB));
  scf::ForOp lastFor = cast<scf::ForOp>(lastB);
  for (auto &op : reverse(lastFor.getBody()->getOperations())) {
    if (ops.count(&op)) {
      tailConsumer = &op;
      break;
    }
  }
  return tailConsumer;
}

// Group channels in two ways:
//  - by producer ops. One producer corresponds to multiple channels. This
//    grouping will be used to create buffers per shared producer.
//  - by consumer ops. One consumer corresponds to multiple channels. This
//  grouping will be used to create barriers per shared consumer.
// Also compute orderedChannels, which will be keyed by getDstOp() of channels,
// to enforce deterministic order for map.
void groupChannels(
    SmallVector<Channel *> &channels,
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByProducers,
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByConsumers,
    SmallVector<Channel *> &orderedChannels) {

  // Group channels by producer op.
  DenseMap<Operation *, SmallVector<Channel *>> producerChannels;
  for (auto channel : channels) {
    producerChannels[channel->getSrcOp()].push_back(channel);
  }

#ifndef NDEBUG
  // Some sanity checks.
  for (auto &item : producerChannels) {
    auto &channels = item.second;
    unsigned numBuffers = channels.front()->getNumBuffers();
    for (auto c : channels) {
      assert(c->getNumBuffers() == numBuffers && "Unmatched number of buffers");
    }
  }
#endif

  // Two channels can be combined if
  //   src1 and src2 are in the same block and
  //   (dst1 == dst2 or
  //    (dst1 and dst2 are in the same block, both have a single user, and
  //     dst1User == dst2User and dst1User is in the same block as dst1))
  auto channelCanBeMerged = [](Channel *c1, Channel *c2) -> bool {
    if (c1->getSrcOp()->getBlock() != c2->getSrcOp()->getBlock())
      return false;
    Operation *dst1 = c1->getDstOp(), *dst2 = c2->getDstOp();
    if (dst1 == dst2)
      return true;
    // We only have one CommChannel for channels in channelsGroupedByConsumers.
    // A CommChannel can have multiple tokens, one for each consumer taskId.
    // Consider the case where channel v is between producer
    // task 0 and consumer task 1, while channel p is between producer task 2
    // and consumer task 1, but in createToken, we only consider the first
    // channel in the group.
    if (getAsyncTaskIds(c1->getSrcOp()) != getAsyncTaskIds(c2->getSrcOp()))
      return false;
    // Check taskIds on dstOps.
    if (getAsyncTaskIds(dst1) != getAsyncTaskIds(dst2))
      return false;
    auto dst1User = getUniqueActualConsumer(dst1);
    auto dst2User = getUniqueActualConsumer(dst2);
    if (!dst1User || !dst2User)
      return false;
    return dst1User == dst2User && dst1User->getBlock() == dst1->getBlock();
  };

  // Group channels by consumer if they can be merged.
  SmallVector<SmallVector<Channel *>> consumerChannels;

  assert(channels.size() > 0 && "channel size is zero");
  // Compare with existing channels in the consumerChannels to see if
  // it can be combined.
  for (auto *c0 : channels) {
    bool merged = false;
    for (auto &c : consumerChannels) {
      if (channelCanBeMerged(c0, c.front())) {
        c.push_back(c0);
        merged = true;
        break;
      }
    }
    if (!merged) { // Create a new entry.
      orderedChannels.push_back(c0);
      // TODO: Even if the channels fail the channelCanBeMerged check, there may
      // be some benefit to tracking the channels that have the same consumer op
      // so they can share the same arrive op.
      consumerChannels.push_back({c0});
    }
  }

  // Reorder channels associated with one entry based on program order of the
  // producers.
  for (auto &group : consumerChannels) {
    auto &allOps = group.front()->getSrcOp()->getBlock()->getOperations();
    DenseMap<Operation *, size_t> opIdx;
    opIdx.reserve(allOps.size());
    for (auto [idx, op] : enumerate(allOps)) {
      opIdx[&op] = idx;
    }
    sort(group, [&](Channel *a, Channel *b) {
      return opIdx[a->getSrcOp()] < opIdx[b->getSrcOp()];
    });
  }

  // Switch to using channel as the key instead of ops as ops can be volatile.
  for (auto &kv : producerChannels) {
    channelsGroupedByProducers[kv.second.front()] = kv.second;
  }
  for (auto &c : consumerChannels) {
    auto *keyChannel = c.front();
    auto [it, inserted] =
        channelsGroupedByConsumers.try_emplace(keyChannel, std::move(c));
    assert(inserted && "Channel in multiple groups");
  }

  LLVM_DEBUG({
    DBGS() << "\n\n";
    LDBG("Grouped channels by producer:");
    unsigned i = 0;
    for (auto &kv : channelsGroupedByProducers) {
      DBGS() << "Channel  " << ++i << ":\n";
      DBGS() << "producer:  ";
      kv.getFirst()->getSrcOp()->dump();
      for (auto &channel : kv.second) {
        DBGS() << "consumer: ";
        channel->getDstOp()->dump();
        DBGS() << "] ";
        LDBG("numBuffers: " << channel->getNumBuffers());
        DBGS() << "\n";
      }
    }

    DBGS() << "\n\n";
    LDBG("Grouped channels by consumer:");
    i = 0;
    for (auto &kv : channelsGroupedByConsumers) {
      DBGS() << "Channel  " << ++i << ":\n";
      DBGS() << "consumer:  ";
      kv.getFirst()->getDstOp()->dump();
      for (auto &channel : kv.second) {
        DBGS() << "producer: ";
        channel->getSrcOp()->dump();
        for (auto &asyncTaskId : channel->relation.second)
          DBGS() << asyncTaskId << ", ";
        DBGS() << "] ";
        LDBG("numBuffers: " << channel->getNumBuffers());
        DBGS() << "\n";
      }
      DBGS() << "\n";
    }
  });
}

// Reorder producer ops to unblock consumers interleavingly.
void reorderProducerOps(SmallVector<Channel *> &channels) {
  if (channels.size() <= 1)
    return;

  // Bail out if channels are not in the same block
  auto block = channels.front()->getSrcOp()->getBlock();
  for (auto &channel : channels) {
    if (channel->getSrcOp()->getBlock() != block) {
      return;
    }
  }

  // Group channels by the first consumer taskId of each channel. Smaller taskId
  // has higher priority.
  // TODO: consider consumer priority
  std::map<AsyncTaskId, SmallVector<Channel *>> groupedProducerOps;
  for (auto &channel : channels) {
    auto asyncTaskId = channel->relation.second.front();
    groupedProducerOps[asyncTaskId].push_back(channel);
  }

  // No need to reorder if all channels are in the same group.
  if (groupedProducerOps.size() <= 1)
    return;

  // Sort each group by number of consumers.
  for (auto &group : groupedProducerOps) {
    std::sort(group.second.begin(), group.second.end(),
              [&](Channel *a, Channel *b) {
                return a->relation.second.size() < b->relation.second.size();
              });
  }

  // Start from the first producer in channels. Iterate through the groups
  // which are ordered by the first consumer taskId. Within each group, channels
  // are ordered by number of consumers.
  Operation *currOp = channels.front()->getSrcOp();
  for (auto &group : groupedProducerOps) {
    for (auto &channel : group.second) {
      channel->getSrcOp()->moveAfter(currOp);
      currOp = channel->getSrcOp();
    }
  }

  // Move backward dependency slice close to producer ops.
  // Start from the last producer op backwards and move backward slice to
  // before each op. This guarantees that the backward slice of each op is
  // scheduled as late as possible.
  for (auto &group : reverse(groupedProducerOps)) {
    for (auto &channel : reverse(group.second)) {
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      SetVector<Operation *> backwardSlice;
      (void)getBackwardSlice(channel->getSrcOp(), &backwardSlice, opt);
      for (auto &op : backwardSlice) {
        if (op->getBlock() == block)
          op->moveBefore(channel->getSrcOp());
      }
    }
  }

  LLVM_DEBUG({
    LDBG("\n");
    LDBG("after reordering producer ops");
    currOp->getParentOfType<triton::FuncOp>().dump();
    LDBG("\n");
  });
}

// Reorder operations in epilogs to pack ops on a dependency chain as close as
// possible.
void reorderEpilogOps(const SmallVector<Channel *> &channels,
                      triton::FuncOp funcOp) {

  llvm::SetVector<Block *> epliogBlocks;
  funcOp->walk([&](Operation *op) {
    if (isa<tt::DescriptorStoreOp, tt::StoreOp>(op)) {
      epliogBlocks.insert(op->getBlock());
    }
  });

  auto lastInBlockOperand = [](Operation *op) {
    Operation *lastOperandOp = nullptr;
    for (auto opnd : op->getOperands()) {
      if (auto defOp = opnd.getDefiningOp()) {
        if (defOp->getBlock() != op->getBlock())
          continue;
        if (!lastOperandOp || !defOp->isBeforeInBlock(lastOperandOp))
          lastOperandOp = defOp;
      }
    }
    return lastOperandOp;
  };

  auto firstInBlockUser = [](Operation *op) {
    Operation *firstUser = nullptr;
    for (Operation *user : op->getUsers()) {
      if (user->getBlock() != op->getBlock())
        continue;
      if (!firstUser || user->isBeforeInBlock(firstUser))
        firstUser = user;
    }
    return firstUser;
  };

  for (auto block : epliogBlocks) {
    LLVM_DEBUG({
      LDBG("\n");
      LDBG("reordering epilog block");
      block->dump();
      LDBG("\n");
    });
    // Find the last scf::ForOp in the block
    SetVector<Operation *> epilogOps;
    std::map<AsyncTaskId, SmallVector<Operation *>> channelOps;
    for (Operation &op : reverse(*block)) {
      if (isa<scf::ForOp, scf::IfOp>(op))
        break;
      epilogOps.insert(&op);
    }

    // Bail out if there's any barrier ops in epilogOps
    bool hasBarrierOps = false;
    for (auto op : epilogOps) {
      if (isa<ttng::WaitBarrierOp, ttng::ArriveBarrierOp,
              ttng::NamedBarrierArriveOp, ttng::NamedBarrierWaitOp,
              ttng::AsyncCopyMbarrierArriveOp, gpu::BarrierOp>(op)) {
        hasBarrierOps = true;
        break;
      }
    }

    if (hasBarrierOps)
      continue;

    for (auto channel : channels) {
      if (epilogOps.contains(channel->getDstOp()))
        channelOps[channel->relation.first].push_back(channel->getDstOp());
    }

    // Streamline ops on a channel chain.
    // Starting with producers with smaller task ids, moving forward
    // dependencies of the consumer ops close to the them.
    for (auto item : channelOps) {
      for (auto op : item.second) {
        SetVector<Operation *> forwardSlice;
        (void)getForwardSlice(op, &forwardSlice);
        for (auto &depOp : reverse(forwardSlice)) {
          if (!epilogOps.contains(depOp))
            continue;
          // push depOp to be right after its operands
          auto lastOpndOp = lastInBlockOperand(depOp);
          if (lastOpndOp)
            depOp->moveAfter(lastOpndOp);
        }
      }
    }

    // Group store ops based on types.
    SmallVector<SmallVector<Operation *, 2>, 2> storeBuckets(2);
    for (auto op : reverse(epilogOps)) {
      if (isa<tt::DescriptorStoreOp>(op))
        storeBuckets[0].push_back(op);
      if (isa<tt::StoreOp>(op))
        storeBuckets[1].push_back(op);
    }

    if (storeBuckets[0].size() != storeBuckets[1].size())
      continue;

    // Reorder store operations in the sequence:
    //   bucket[0][N], bucket[1][N],
    //   bucket[0][N-1], bucket[1][N-1],
    //   ...
    //   bucket[0][0], bucket[1][0].
    //
    // This ordering aligns with the expected producer pattern, where
    // producers of bucket[0][0], bucket[1][0], ... complete earlier than
    // those of bucket[0][1], bucket[1][1], and so on. By reordering the
    // stores in this manner, we ensure that operations finish as early as
    // possible overall.
    SmallVector<Operation *> storeOps;
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &store : storeBuckets) {
        if (!store.empty()) {
          storeOps.push_back(store.back());
          store.pop_back();
          changed = true;
        }
      }
    }

    assert(storeBuckets[0].empty() && storeBuckets[1].empty() &&
           "All stores must have been processed");

    // Reorder stores op physically based on the computed
    for (unsigned i = 1; i < storeOps.size(); i++) {
      storeOps[i]->moveBefore(storeOps[i - 1]);
    }

    // Streamline ops on a store chain
    // For each store op, move backward dependencies close to the op.
    // Start from the last store op backwards and move backward slice to
    // before each op. This guarantees that the backward slice of each op is
    // scheduled as late as possible.
    for (auto storeOp : storeOps) {
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      SetVector<Operation *> backwardSlice;
      (void)getBackwardSlice(storeOp, &backwardSlice, opt);
      for (auto &depOp : reverse(backwardSlice)) {
        if (!epilogOps.contains(depOp))
          continue;
        // push depOp to be right before its first user
        auto firstUser = firstInBlockUser(depOp);
        if (firstUser)
          depOp->moveBefore(firstUser);
      }
    }

    LLVM_DEBUG({
      LDBG("\n");
      LDBG("reordered epilog block");
      block->dump();
      LDBG("\n");
    });
  }

  LLVM_DEBUG({
    LDBG("\n");
    LDBG("after reordering epilog ops");
    funcOp.dump();
    LDBG("\n");
  });
}

// Find top-level ops which contain at least one channel. If a channel's
// getSrcOp() and getDstOp() belong to the inner loop, the outer loop will be
// part of asyncTaskOps.
SmallVector<Operation *>
getTaskTopRegion(triton::FuncOp funcOp,
                 const SmallVector<Channel *> &channels) {
  SmallVector<Operation *> asyncTaskOps;
  auto isAsyncTaskTopOp = [&](Operation *taskTopOp) -> bool {
    for (auto c : channels) {
      Operation *producer = c->getSrcOp(), *consumer = c->getDstOp();
      while (producer && !isa<triton::FuncOp>(producer->getParentOp())) {
        producer = producer->getParentOp();
      }
      while (consumer && !isa<triton::FuncOp>(consumer->getParentOp())) {
        consumer = consumer->getParentOp();
      }
      if (producer == taskTopOp && consumer == taskTopOp)
        return true;
    }
    return false;
  };
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &bodyOp : block.getOperations()) {
      Operation *op = &bodyOp;
      if (op->getNumRegions() <= 0)
        continue;
      // If this op does not contain both a producer taskId and a consumer
      // taskId, continue.
      if (getAsyncTaskIds(op).size() == 1)
        continue;
      if (isAsyncTaskTopOp(op))
        asyncTaskOps.push_back(op);
    }
  }

  LLVM_DEBUG({
    LDBG("\nTop Task Bodies");
    for (auto op : asyncTaskOps) {
      LDBG("\nTask Body:");
      op->dump();
    }
  });
  return asyncTaskOps;
}

// Create an allocation to hold the mbarriers.
static Value createBarrierAlloc(triton::FuncOp funcOp, unsigned distance) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(funcOp.getContext());
  Location loc = funcOp.getLoc();
  auto context = funcOp.getContext();
  auto barrierCTALayout =
      ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                              /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
      context, 1, 1, 1, {0}, barrierCTALayout);
  ttg::MemDescType barrierMemDescType = ttg::MemDescType::get(
      {distance, 1}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Type singleBarrierMemDescType = ttg::MemDescType::get(
      {1}, builder.getI64Type(), barrierEncoding,
      barrierMemDescType.getMemorySpace(), /*mutableMemory=*/true);
  Value barrierAlloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loc, barrierMemDescType, Value());
  for (unsigned i = 0; i < distance; i++) {
    Value idx = builder.create<arith::ConstantIntOp>(loc, i, 32);
    Value barrierView = builder.create<ttg::MemDescIndexOp>(
        loc, singleBarrierMemDescType, barrierAlloc, idx);
    builder.create<ttng::InitBarrierOp>(funcOp->getLoc(), barrierView, 1);
  }
  return barrierAlloc;
}

static Operation *ProducerIsGen5(Operation *producerOp) {
  if (isa<ttng::TCGen5MMAOp>(producerOp))
    return producerOp;
  Operation *allocOp = producerOp;
  if (auto tmSt = dyn_cast<ttng::TMEMStoreOp>(producerOp)) {
    allocOp = tmSt.getDst().getDefiningOp();
  }
  for (auto user : allocOp->getUsers()) {
    if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(user)) {
      if (mmaOp.getD() == allocOp->getResult(0))
        return user;
    }
  }
  return nullptr;
}

// channelsGroupedByConsumers: channels are grouped together.
// Go through each group, check the first channel in the group, create a token
// for each consumer taskId. Return a map that maps each channel + consumer
// taskId to a token. Also update barrierAllocMap that maps each channel +
// consumer taskId to a BarrierAlloc.
void createToken(
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByConsumers,
    const SmallVector<Channel *> &orderedChannels, triton::FuncOp funcOp,
    const DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseMap<Channel *, CommChannel> &tokenMap, ReuseConfig *config) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  DenseMap<ttng::TCGen5MMAOp, Channel *> gen5Barriers;
  for (auto *key : orderedChannels) {
    auto it = channelsGroupedByConsumers.find(key);
    LLVM_DEBUG({
      LDBG("createToken key:");
      LDBG("consumer: ");
      key->getDstOp()->dump();

      LDBG("createToken channelsGroupedByConsumers:");
      for (auto map_key : make_first_range(channelsGroupedByConsumers)) {
        LDBG("representative consumer: ");
        map_key->getDstOp()->dump();
      }
    });
    assert(it != channelsGroupedByConsumers.end());
    Channel *channel = it->second.front();
    // For each reuse group, choose a representative channel.
    int reuseGrp = channelInReuseGroup(channel, config);
    if (reuseGrp >= 0) {
      if (channel != config->getGroup(reuseGrp)->channels[0])
        continue;
    }

    CommChannel commChannel;
    auto producerOp = it->second.front()->getSrcOp();
    auto dstOp = it->second.front()->getDstOp();

    // Pre-allocate TMA barrier if ANY channel in the group has a TMA producer.
    // insertAsyncComm may be called with different isPost values,
    // so check both direct DescriptorLoadOp and the post case
    // (LocalStoreOp with DescriptorLoadOp source) to ensure we catch all TMA
    // loads.
    bool hasTMAProducer = false;
    for (auto *c : it->second) {
      // Check for direct DescriptorLoadOp (isPost=false case)
      if (isa<tt::DescriptorLoadOp>(c->getSrcOp())) {
        hasTMAProducer = true;
        break;
      }
      // Check for LocalStoreOp with DescriptorLoadOp source (isPost=true case)
      if (auto ls = dyn_cast<ttg::LocalStoreOp>(c->getSrcOp())) {
        if (auto def = ls.getSrc().getDefiningOp()) {
          if (isa<tt::DescriptorLoadOp>(def)) {
            hasTMAProducer = true;
            break;
          }
        }
      }
    }
    if (hasTMAProducer) {
      commChannel.producerBarrier =
          createBarrierAlloc(funcOp, channel->getNumBuffers());
    }
    // Pattern matching for tmem_store --> getD --> tmem_load (gen5 is the
    // actual producer) or gen5 --> tmem_load
    if (ProducerIsGen5(producerOp))
      commChannel.producerBarrier =
          createBarrierAlloc(funcOp, channel->getNumBuffers());

    for (auto consumerAsyncTaskId : channel->relation.second) {
      // It is possible that this channel has two consumer taskIds.
      Operation *consumerOp =
          getUniqueActualConsumer(dstOp, consumerAsyncTaskId);

      // For channels associated with acc of gen5, consumerOp is not the gen5,
      // it is usually tmem_load.
      bool useGen5Barrier = isa<ttng::TCGen5MMAOp>(consumerOp) &&
                            producerOp->getBlock() == consumerOp->getBlock();
      LLVM_DEBUG({
        LDBG("-- createToken: useGen5Barrier = " << useGen5Barrier);
        producerOp->dump();
        dstOp->dump();
        consumerOp->dump();
      });
      if (useGen5Barrier) {
        auto mmaOp = cast<ttng::TCGen5MMAOp>(consumerOp);
        // If the gen5 barrier for this mmaOp is already used for another
        // channel, do not use it for this channel.
        if (gen5Barriers.count(mmaOp) && gen5Barriers[mmaOp] != channel) {
          // useGen5Barrier = false; // FIXME
          LDBG("-- mmaOp already has a channel associated");
        }
      }

      // No token is needed for a TMA <-> TCGen5MMAOp channel
      if (!isa<tt::DescriptorLoadOp>(producerOp) ||
          !useGen5Barrier) { // isa<ttng::TCGen5MMAOp>(consumerOp)) {
        ttnvws::TokenLoadType tokenLoadType;
        assert(copyOpMap.count(channel));
        auto copyOp = copyOpMap.find(channel)->second.first;
        if (isa<ttg::AsyncCopyGlobalToLocalOp>(copyOp)) {
          tokenLoadType = ttnvws::TokenLoadType::AsyncLoadOp;
        } else if (isa<tt::DescriptorLoadOp>(copyOp)) {
          tokenLoadType = ttnvws::TokenLoadType::TMALoadOp;
        } else if (isa<ttg::LocalStoreOp>(copyOp)) {
          tokenLoadType = ttnvws::TokenLoadType::LocalStoreOp;
        } else if (isa<ttng::TMEMLoadOp>(consumerOp)) {
          tokenLoadType = ttnvws::TokenLoadType::TmemLoadOp;
        } else if (isa<ttng::TCGen5MMAOp>(consumerOp)) {
          // For operand A of gen5, we have tmem_store + gen5.
          tokenLoadType = ttnvws::TokenLoadType::TmemLoadOp;
        } else {
          llvm_unreachable("Unexpected load type");
        }
        Value v;
        if (it->second.front()->getSrcOp()->getParentOfType<scf::ForOp>())
          v = builder.create<ttnvws::CreateTokenOp>(
              funcOp.getLoc(), channel->getNumBuffers(), tokenLoadType);
        else
          v = builder.create<ttnvws::CreateTokenOp>(funcOp.getLoc(), 1,
                                                    tokenLoadType);
        commChannel.tokens[consumerAsyncTaskId] = v;
      }

      if (useGen5Barrier) {
        Value v = createBarrierAlloc(funcOp, channel->getNumBuffers());
        commChannel.consumerBarriers[consumerAsyncTaskId] = v;
        gen5Barriers[cast<ttng::TCGen5MMAOp>(consumerOp)] = channel;
      }
    }

    // Channels in the group share the same set of tokens.
    for (auto &c : it->second) {
      tokenMap[c] = commChannel;
    }
    // For channels in the same reuse group as channel, use the same token.
    if (reuseGrp >= 0) {
      for (auto *reuse : config->getGroup(reuseGrp)->channels)
        tokenMap[reuse] = commChannel;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Communication Channels: \n";
    for (auto &item : tokenMap) {
      llvm::dbgs() << "\ndata channel: \n";
      llvm::dbgs() << *item.first->getSrcOp() << "\n";
      llvm::dbgs() << *item.first->getDstOp() << "\n";
      llvm::dbgs() << "communication channel: \n";
      for (auto &kv : item.second.tokens) {
        llvm::dbgs() << "token: " << kv.first << " " << kv.second << "\n";
      }
      if (item.second.producerBarrier)
        llvm::dbgs() << "producer barrier: " << *item.second.producerBarrier
                     << "\n";
      for (auto &kv : item.second.consumerBarriers)
        llvm::dbgs() << "consumer barrier: " << kv.first << " " << kv.second
                     << "\n";
    }
  });
}

static Operation *isProducerTMA(Channel *ch, bool isPost) {
  if (!isPost && isa<tt::DescriptorLoadOp>(ch->getSrcOp()))
    return ch->getSrcOp();
  if (!isPost)
    return nullptr;
  auto producerOp = ch->getSrcOp();
  // Pre-allocate TMA barrier, do not use token for producer.
  // We have a chain of descriptor_load -> local_store.
  if (auto ls = dyn_cast<ttg::LocalStoreOp>(producerOp)) {
    Operation *def = ls.getSrc().getDefiningOp();
    if (isa<tt::DescriptorLoadOp>(def))
      return def;
  }
  return nullptr;
}

// Handle buffer index and phase computation for operations outside loops
// (epilogue/prologue). Returns a pair of (bufferIdx, phase).
static std::pair<Value, Value> getBufferIdxAndPhaseForOutsideLoopOps(
    OpBuilderWithAsyncTaskIds &builder, Operation *user, Channel *channel,
    Operation *oldAllocOp, unsigned numBuffers,
    const DenseSet<Operation *> &regionsWithChannels, ReuseConfig *config,
    int reuseGrp) {
  Value bufferIdx;
  Value _phase;

  // For operations outside loops (epilogue), compute the
  // correct bufferIdx and phase based on the parent loop's final
  // iteration. Find the parent loop that this
  // operation came from by walking up the IR.
  Operation *opInsideLoop = nullptr;

  // Look at the channel's source operation, which is where
  // the data was produced, to find the
  // loop that produced the data being consumed in the epilogue.
  if (channel) {
    if (auto srcOp = channel->getSrcOp()) {
      if (srcOp->getParentOfType<scf::ForOp>()) {
        opInsideLoop = srcOp;
      }
    }
  }

  // If channel doesn't have a source in a loop, try the
  // allocation's operand
  if (!opInsideLoop && oldAllocOp->getNumOperands() > 0) {
    if (auto defOp = oldAllocOp->getOperand(0).getDefiningOp()) {
      if (defOp->getParentOfType<scf::ForOp>()) {
        opInsideLoop = defOp;
      }
    }
  }

  if (opInsideLoop) {
    // Determine if this is a prologue or epilogue operation
    bool isPrologue = false;

    // Check if this is an initialization operation (prologue)
    // TMEMAlloc without src operand indicates the buffer needs
    // initialization from a constant (like tl.zeros()), which should
    // happen before the loop
    if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(oldAllocOp)) {
      if (!tmemAlloc.getSrc()) {
        // No src means this needs explicit initialization before the loop
        isPrologue = true;
      }
    }

    auto parentLoop = opInsideLoop->getParentOfType<scf::ForOp>();
    if (isPrologue) {
      // For prologue operations (initialization), use initial values
      // and place before the loop
      if (parentLoop) {
        builder.setInsertionPoint(parentLoop);
      }
      bufferIdx = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          user->getLoc(), 0, 32);
      _phase = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          user->getLoc(), 0, 1);
    } else {
      // For epilogue operations, compute final loop values
      // and place after the loop to avoid forward references
      if (parentLoop) {
        builder.setInsertionPointAfter(parentLoop);
      }
      std::tie(bufferIdx, _phase) =
          getOutOfScopeBufferIdxAndPhase(builder, opInsideLoop, numBuffers,
                                         regionsWithChannels, config, reuseGrp);
    }
    // Restore insertion point to user
    builder.setInsertionPoint(user);
  } else {
    // Fallback: if we can't find a parent loop, use constant 0
    // (this should only happen for operations truly outside any loop)
    bufferIdx = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        user->getLoc(), 0, 32);
    _phase = builder.createWithAsyncTaskIds<arith::ConstantIndexOp>(
        user->getLoc(), 0);
  }

  return {bufferIdx, _phase};
}

// Check if a channel needs token-based synchronization by examining if
// actual consumers are inside loops when endpoints are outside loops
static bool checkConsumersInLoops(Channel *channel) {
  auto *srcOp = channel->getSrcOp();
  auto *dstOp = channel->getDstOp();

  // Special case when srcOp or dstOp is scf.for;
  // we need to check if operations inside the loop need sync
  bool srcIsLoop = isa<scf::ForOp>(srcOp);
  bool dstIsLoop = isa<scf::ForOp>(dstOp);

  if (srcIsLoop || dstIsLoop) {
    // When the channel endpoints are loop operations themselves,
    // we need to look inside the loops to determine if sync is needed
    LDBG("createToken: channel "
         << channel->uniqID << " has loop as endpoint (srcIsLoop=" << srcIsLoop
         << ", dstIsLoop=" << dstIsLoop
         << ") - proceeding with token creation");
    // Fall through to create tokens
    return false;
  }

  // Normal case: check if ops are outside loops
  bool producerOutsideLoop = srcOp && !srcOp->getParentOfType<scf::ForOp>();
  bool consumerOutsideLoop = dstOp && !dstOp->getParentOfType<scf::ForOp>();

  // If both producer and consumer ops are outside loops, check if actual
  // consumers are inside loops. This handles both cases:
  // 1. Multiple consumer task IDs in different loops
  // 2. Single consumer task ID but actual consumer is inside a loop
  if (producerOutsideLoop && consumerOutsideLoop) {
    // Collect all destination operations
    SmallVector<Operation *> dstOps;
    if (channel->channelKind == DataChannelKind::SMEMPost) {
      auto *cPost = static_cast<ChannelPost *>(channel);
      cPost->getDstOps(dstOps);
    } else {
      dstOps.push_back(dstOp);
    }

    // Check if actual consumers (with the consumer task IDs) are inside
    // loops
    bool hasConsumersInLoops = false;

    // For each consumer task ID, check if operations with that task ID are
    // in loops
    for (auto consumerTaskId : channel->relation.second) {
      // Check actual consumers from dstOps
      for (auto *dst : dstOps) {
        auto consumers = getActualConsumers(dst);
        for (auto *consumer : consumers) {
          auto consumerTasks = getAsyncTaskIds(consumer);
          // Check if this consumer has the task ID we're looking for
          if (std::find(consumerTasks.begin(), consumerTasks.end(),
                        consumerTaskId) != consumerTasks.end()) {
            // Check if this consumer is inside a loop
            if (consumer->getParentOfType<scf::ForOp>()) {
              hasConsumersInLoops = true;
              LDBG("createToken: found consumer with task "
                   << consumerTaskId << " inside loop for channel "
                   << channel->uniqID);
              break;
            }
          }
        }
        if (hasConsumersInLoops)
          break;
      }
      if (hasConsumersInLoops)
        break;
    }
    return hasConsumersInLoops;
  }

  return false;
}

void createTokenPost(
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByConsumers,
    const SmallVector<Channel *> &orderedChannels, triton::FuncOp funcOp,
    const DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseMap<Channel *, CommChannel> &tokenMap, ReuseConfig *config) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));

  // First pass: ensure all representative channels are processed first
  // This prevents issues where non-representative channels are processed
  // before their representative, leaving them without CommChannels
  SmallVector<Channel *> processOrder;
  DenseSet<Channel *> processed;

  // Add all representative channels first
  for (auto *key : orderedChannels) {
    auto it = channelsGroupedByConsumers.find(key);
    if (it == channelsGroupedByConsumers.end())
      continue;
    Channel *channel = it->second.front();
    int reuseGrp = channelInReuseGroup(channel, config);
    if (reuseGrp >= 0) {
      auto *repChannel = config->getGroup(reuseGrp)->channels[0];
      if (channel == repChannel && !processed.count(channel)) {
        processOrder.push_back(channel);
        processed.insert(channel);
      }
    } else if (!processed.count(channel)) {
      // Not in a reuse group, process normally
      processOrder.push_back(channel);
      processed.insert(channel);
    }
  }

  // Add non-representative channels
  for (auto *key : orderedChannels) {
    auto it = channelsGroupedByConsumers.find(key);
    if (it == channelsGroupedByConsumers.end())
      continue;
    Channel *channel = it->second.front();
    if (!processed.count(channel)) {
      processOrder.push_back(channel);
      processed.insert(channel);
    }
  }

  for (auto *channel : processOrder) {
    auto it = channelsGroupedByConsumers.find(channel);
    LLVM_DEBUG({
      LDBG("createToken key:");
      LDBG("consumer: ");
      channel->getDstOp()->dump();
      LDBG("producer: ");
      channel->getSrcOp()->dump();
    });
    assert(it != channelsGroupedByConsumers.end());

    // For each reuse group, choose a representative channel.
    int reuseGrp = channelInReuseGroup(channel, config);
    if (reuseGrp >= 0) {
      // FIXME: check that the other channels in the reuse group have the same
      // choice about producerBarrier, and consumerBarriers. If not, we should
      // not set producerBarrier, and consumerBarriers.
      auto *repChannel = config->getGroup(reuseGrp)->channels[0];
      if (channel != repChannel) {
        // This channel is in a reuse group but is not the representative.
        // The representative should have already been processed in the first
        // pass.
        auto repIt = tokenMap.find(repChannel);
        assert(repIt != tokenMap.end() &&
               "Representative channel should have been processed first");
        // Share the representative's CommChannel
        tokenMap[channel] = repIt->second;
        LDBG("createToken: channel "
             << channel->uniqID
             << " shares CommChannel from representative channel "
             << repChannel->uniqID);
        continue;
      }
    }

    CommChannel commChannel;
    auto producerOp = it->second.front()->getSrcOp();
    auto dstOp = it->second.front()->getDstOp();

    // Pre-allocate TMA barrier if any channel in the group has a TMA producer.
    // insertAsyncComm is called with both isPost=false and
    // isPost=true, so we must check both to ensure we catch all TMA loads.
    // Also check all channels in the reuse group, not just the consumer group.
    bool hasTMAProducer = false;
    // First check channels grouped by consumer
    for (auto *c : it->second) {
      if (isProducerTMA(c, true) || isProducerTMA(c, false)) {
        hasTMAProducer = true;
        break;
      }
    }
    // Also check all channels in the reuse group (if applicable)
    if (!hasTMAProducer && reuseGrp >= 0) {
      for (auto *c : config->getGroup(reuseGrp)->channels) {
        if (isProducerTMA(c, true) || isProducerTMA(c, false)) {
          hasTMAProducer = true;
          break;
        }
      }
    }
    if (hasTMAProducer) {
      commChannel.producerBarrier =
          createBarrierAlloc(funcOp, channel->getNumBuffers());
    }
    // If channel is from a gen5, pre-allocate gen5 barrier.
    bool hasProdBar = false;
    if (isa<ttng::TCGen5MMAOp>(producerOp)) {
      commChannel.producerBarrier =
          createBarrierAlloc(funcOp, channel->getNumBuffers());
      hasProdBar = true;
    }
    // Check if this channel needs token-based synchronization.
    // When srcOp and dstOp are both outside loops, we need to check if the
    // actual consumers are inside loops. This can happen with both single and
    // multiple consumer task IDs.
    checkConsumersInLoops(channel);
    for (auto consumerAsyncTaskId : channel->relation.second) {
      // It is possible that this channel has two consumer taskIds.
      // We can have multiple consumer ops for ChannelPost, or one consumer op
      // has multiple actual consumers. Here we collect all consumer ops.
      DenseSet<Operation *> actualConsumers;
      SmallVector<Operation *> dstOps;
      if (channel->channelKind == DataChannelKind::SMEMPost) {
        auto *cPost = static_cast<ChannelPost *>(channel);
        cPost->getDstOps(dstOps);
      } else {
        dstOps.push_back(dstOp);
      }
      // If it is used by gen5, we can create a gen5 barrier for consumer
      // release.
      bool useGen5Barrier = true;
      for (auto *dst : dstOps) {
        auto consumers = getActualConsumers(dst);
        for (auto *t : consumers) {
          SmallVector<AsyncTaskId> asyncTasks = getAsyncTaskIds(t);

          // Handle operations that belong to multiple tasks (e.g., boundary
          // ops) Only include if this consumer belongs to the task we're
          // processing
          if (asyncTasks.empty()) {
            LLVM_DEBUG({
              LDBG("Skipping operation with no async tasks");
              t->dump();
            });
            continue;
          }

          if (std::find(asyncTasks.begin(), asyncTasks.end(),
                        consumerAsyncTaskId) != asyncTasks.end()) {
            actualConsumers.insert(t);
            // XXX: Op can have multiple async tasks

            // If consumer and producer are not in the same block, but
            // as long as all consumers are gen5, we can use a gen5 related
            // barrier such as gen5.commit. Remove producerOp->getBlock() !=
            // t->getBlock()
            if (!isa<ttng::TCGen5MMAOp>(t))
              useGen5Barrier = false;
          }
        }
      }
      assert(!actualConsumers.empty());
      Operation *consumerOp =
          *actualConsumers.begin(); // getLastOpInBlock(actualConsumers);

      LLVM_DEBUG({
        LDBG("-- createToken: useGen5Barrier = "
             << useGen5Barrier << " channel " << channel->uniqID);
        producerOp->dump();
        dstOp->dump();
        consumerOp->dump();
      });
      // Need token only when we are not using inline barriers
      if (!hasProdBar || !useGen5Barrier) {
        ttnvws::TokenLoadType tokenLoadType;
        auto copyOp = channel->getSrcOp();
        if (isa<ttg::AsyncCopyGlobalToLocalOp>(copyOp)) {
          tokenLoadType = ttnvws::TokenLoadType::AsyncLoadOp;
        } else if (isProducerTMA(channel, true)) {
          tokenLoadType = ttnvws::TokenLoadType::TMALoadOp;
        } else if (isa<ttg::LocalStoreOp>(copyOp)) {
          tokenLoadType = ttnvws::TokenLoadType::LocalStoreOp;
        } else if (isa<ttng::TMEMLoadOp>(consumerOp)) {
          tokenLoadType = ttnvws::TokenLoadType::TmemLoadOp;
        } else if (isa<ttng::TCGen5MMAOp>(consumerOp)) {
          // For operand A of gen5, we have tmem_store + gen5.
          tokenLoadType = ttnvws::TokenLoadType::TmemLoadOp;
        } else {
          llvm_unreachable("Unexpected load type");
        }
        Value v;
        v = builder.create<ttnvws::CreateTokenOp>(
            funcOp.getLoc(), channel->getNumBuffers(), tokenLoadType);
        commChannel.tokens[consumerAsyncTaskId] = v;
      }

      if (useGen5Barrier) {
        Value v = createBarrierAlloc(funcOp, channel->getNumBuffers());
        commChannel.consumerBarriers[consumerAsyncTaskId] = v;
      }
    }

    // Channels in the group share the same set of tokens.
    for (auto &c : it->second) {
      tokenMap[c] = commChannel;
    }
    // For channels in the same reuse group as channel, use the same token.
    // If the channel has a single buffer, still uses different tokens.
    if (reuseGrp >= 0) {
      for (auto *reuse : config->getGroup(reuseGrp)->channels)
        tokenMap[reuse] = commChannel;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Communication Channels: \n";
    for (auto &item : tokenMap) {
      llvm::dbgs() << "\ndata channel: \n";
      llvm::dbgs() << *item.first->getSrcOp() << "\n";
      llvm::dbgs() << *item.first->getDstOp() << "\n";
      llvm::dbgs() << "communication channel: \n";
      for (auto &kv : item.second.tokens) {
        llvm::dbgs() << "token: " << kv.first << " " << kv.second << "\n";
      }
      if (item.second.producerBarrier)
        llvm::dbgs() << "producer barrier: " << *item.second.producerBarrier
                     << "\n";
      for (auto &kv : item.second.consumerBarriers)
        llvm::dbgs() << "consumer barrier: " << kv.first << " " << kv.second
                     << "\n";
    }
  });
}

static Value hoistLocalAlloc(OpBuilderWithAsyncTaskIds &builder,
                             Operation *oldAlloc) {

  Type oldAllocType;

  if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(oldAlloc)) {
    oldAllocType = localAlloc.getType();
  } else if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(oldAlloc)) {
    oldAllocType = tmemAlloc.getType();
  } else {
    llvm_unreachable("Unexpected alloc type");
  }

  // If the alloc is already hoisted, return the buffer.
  if (isa<triton::FuncOp>(oldAlloc->getParentOp())) {
    return oldAlloc->getResult(0);
  }

  auto allocDescType = cast<triton::gpu::MemDescType>(oldAllocType);
  SmallVector<int64_t> shape(allocDescType.getShape());
  Type memdescType = ttg::MemDescType::get(
      shape, allocDescType.getElementType(), allocDescType.getEncoding(),
      allocDescType.getMemorySpace(), /*mutableMemory*/ true);
  Operation *newAlloc;
  if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(oldAlloc)) {
    newAlloc =
        builder.create<ttg::LocalAllocOp>(oldAlloc->getLoc(), memdescType);
  } else if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(oldAlloc)) {
    if (tmemAlloc.getToken()) {
      newAlloc = builder.create<ttng::TMEMAllocOp>(
          oldAlloc->getLoc(), memdescType, tmemAlloc.getToken().getType(),
          Value());
    } else {
      newAlloc = builder.create<ttng::TMEMAllocOp>(
          oldAlloc->getLoc(), memdescType, mlir::Type(), Value());
    }
  } else {
    llvm_unreachable("Unexpected alloc type");
  }

  auto newBuf = newAlloc->getResult(0);
  auto originTaskIds = builder.getAsyncTaskIds();
  auto originLoopScheduleInfo = builder.getLoopScheduleInfo();
  builder.setAsyncTaskIdsFromOp(oldAlloc);
  if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(oldAlloc)) {
    builder.setLoopScheduleInfoFromOp(oldAlloc);
    if (localAlloc.getSrc() != nullptr) {
      auto storeOp = builder.createWithAsyncTaskIds<ttg::LocalStoreOp>(
          oldAlloc->getLoc(), localAlloc.getSrc(), newBuf);
      storeOp->moveBefore(oldAlloc);
    }
    mlir::triton::replaceUsesAndPropagateType(builder, oldAlloc, newBuf);
  } else if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(oldAlloc)) {
    builder.setLoopScheduleInfoFromOp(tmemAlloc);
    if (tmemAlloc.getSrc() != nullptr) {
      auto pred = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          oldAlloc->getLoc(), 1, 1);
      auto storeOp = builder.createWithAsyncTaskIds<ttng::TMEMStoreOp>(
          oldAlloc->getLoc(), newBuf, tmemAlloc.getSrc(), pred);
      pred->moveBefore(oldAlloc);
      storeOp->moveBefore(oldAlloc);
    }
    oldAlloc->replaceAllUsesWith(newAlloc);
  }
  builder.setAsynTaskIdsFromArray(originTaskIds);
  builder.setLoopScheduleInfoFromInfo(originLoopScheduleInfo);
  oldAlloc->erase();
  return newBuf;
}

// Create a local buffer for register channels. Return the allocated buffer and
// the new producer (reloaded value).
static std::pair<Value, Value>
createLocalAlloc(OpBuilderWithAsyncTaskIds &builder, Channel *channel,
                 bool useTMEM, bool isPost) {
  auto srcResult = channel->getSrcOperand();
  auto srcOp = channel->getSrcOp();
  auto dstOp = channel->getDstOp();
  auto tensorType = dyn_cast<RankedTensorType>(srcResult.getType());
  auto context = builder.getContext();

  // Get basic information from tensorType
  auto order = ttg::getOrderForMemory(tensorType);
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Check the consumer type
  auto actualConsumers = getActualConsumers(dstOp);
  LLVM_DEBUG({
    DBGS() << "actual consumers: \n";
    for (auto consumerOp : actualConsumers) {
      DBGS() << *consumerOp << "\n";
    }
  });

  Value buffer;
  Value newProducer;

  if (useTMEM) {
    // Get shape, layout and type of the complete buffer
    auto shape = tensorType.getShape();
    SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
    bufferShape.push_back(1);
    Attribute tensorMemorySpace = ttng::TensorMemorySpaceAttr::get(context);
    auto blockM = bufferShape[0];
    auto elemType = tensorType.getElementType();
    unsigned elemBitWidth = elemType.getIntOrFloatBitWidth();
    bool unpacked = elemBitWidth != 16;
    auto encoding = ttng::TensorMemoryEncodingAttr::get(
        context, blockM, bufferShape[1],
        /*unpacked=*/unpacked, /*CTASplitM=*/1, /*CTASplitN=*/1);
    Type memdescType =
        ttg::MemDescType::get(bufferShape, elemType, encoding,
                              tensorMemorySpace, /*mutableMemory*/ true);
    auto allocOp = builder.create<ttng::TMEMAllocOp>(
        srcOp->getLoc(), memdescType, builder.getType<ttg::AsyncTokenType>(),
        /*src=*/Value());
    newProducer = TMEM1DAllocator(builder).replaceWith1DTMEM(
        dyn_cast<mlir::OpResult>(srcResult), channel->relation.first, dstOp,
        allocOp);
    buffer = allocOp->getResult(0);
  } else {
    auto originTaskIds = builder.getAsyncTaskIds();
    auto originLoopScheduleInfo = builder.getLoopScheduleInfo();
    if (isPost)
      builder.setAsyncTaskIdsFromOp(srcOp);
    tt::DescriptorStoreOp tmaStore;
    bool requireMMASharedEncoding =
        llvm::any_of(actualConsumers, [&](Operation *op) {
          // convert_layout
          if (isa<ttg::ConvertLayoutOp>(op)) {
            for (auto *user : op->getUsers()) {
              // Do not reuse the current order for TMA store desc. Subsequent
              // codegen for TMA store does not handle mismatching order well.
              if ((tmaStore = dyn_cast<tt::DescriptorStoreOp>(user))) {
                return false;
              }
            }
          }
          // Do not reuse the current order for TMA store desc. Subsequent
          // codegen for TMA store does not handle mismatching order well.
          if ((tmaStore = dyn_cast<tt::DescriptorStoreOp>(op))) {
            return false;
          }
          return isa<mlir::triton::DotOpInterface>(op);
        });

    // Get shape, layout and type of a slice
    auto sliceShape = tensorType.getShape();
    Attribute sharedLayout;
    if (requireMMASharedEncoding) {
      sharedLayout = ttg::NVMMASharedEncodingAttr::get(
          context, sliceShape, order, CTALayout, elemType,
          /*fp4Padded*/ false);
    } else if (tmaStore) {
      sharedLayout = ttng::getEncodingFromDescriptor(tmaStore, tensorType,
                                                     tmaStore.getDesc());
    } else if (auto tmaLoad = dyn_cast<tt::DescriptorLoadOp>(srcOp)) {
      sharedLayout = ttng::getEncodingFromDescriptor(tmaLoad, tmaLoad.getType(),
                                                     tmaLoad.getDesc());
    } else {
      // Create an unswizzled layout for now.
      // TODO: optimize it based on the consumer.
      sharedLayout = ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1,
                                                          order, CTALayout);
    }

    // Get shape, layout and type of the complete buffer
    SmallVector<int64_t> bufferShape(sliceShape.begin(), sliceShape.end());
    if (srcOp->getParentOfType<scf::ForOp>())
      bufferShape.insert(bufferShape.begin(), channel->getNumBuffers());
    else
      bufferShape.insert(bufferShape.begin(), 1);

    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);
    Type memdescType = ttg::MemDescType::get(
        isPost ? sliceShape : bufferShape, elemType, sharedLayout,
        sharedMemorySpace, /*mutableMemory*/ true);
    auto allocOp =
        builder.create<ttg::LocalAllocOp>(srcOp->getLoc(), memdescType);
    buffer = allocOp->getResult(0);

    if (isPost) {
      // Generate the local store
      builder.setLoopScheduleInfoFromOp(srcOp);
      auto storeOp = builder.createWithAsyncTaskIds<ttg::LocalStoreOp>(
          srcOp->getLoc(), srcResult, allocOp);
      storeOp->moveAfter(srcOp);

      // local load
      builder.setAsyncTaskIdsFromOp(dstOp);
      builder.setLoopScheduleInfoFromOp(dstOp);
      auto loadOp = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
          srcOp->getLoc(), srcResult.getType(), allocOp, Value());
      loadOp->moveBefore(dstOp);
      dstOp->replaceUsesOfWith(srcResult, loadOp->getResult(0));
      newProducer = loadOp->getResult(0);
      builder.setAsynTaskIdsFromArray(originTaskIds);
      builder.setLoopScheduleInfoFromInfo(originLoopScheduleInfo);
    }
  }

  return {buffer, newProducer};
}

static ttg::LocalAllocOp hoistLocalAllocPost(OpBuilder &builder,
                                             ttg::LocalAllocOp oldAlloc,
                                             int numBuffers) {
  auto oldRetType = oldAlloc.getType();
  auto allocDescType = cast<triton::gpu::MemDescType>(oldRetType);
  SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                oldRetType.getShape().end()};
  if (numBuffers >= 1) {
    shape.insert(shape.begin(), numBuffers);
  }

  Type memdescType = ttg::MemDescType::get(
      shape, allocDescType.getElementType(), allocDescType.getEncoding(),
      allocDescType.getMemorySpace(), allocDescType.getMutableMemory());
  return builder.create<ttg::LocalAllocOp>(oldAlloc.getLoc(), memdescType);
}

static ttng::TMEMAllocOp createTMemAllocPost(OpBuilder &builder,
                                             ttng::TMEMAllocOp oldTMemAllocOp,
                                             int numBuffers) {
  Location loc = oldTMemAllocOp.getLoc();
  auto oldRetType = oldTMemAllocOp.getType();
  SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                oldRetType.getShape().end()};
  // We can still use subView in createTMEMCopy even if numBuffers is 1.
  if (numBuffers >= 1) {
    shape.insert(shape.begin(), numBuffers);
  }
  Type accMemDescType = triton::gpu::MemDescType::get(
      shape, oldRetType.getElementType(), oldRetType.getEncoding(),
      oldRetType.getMemorySpace(), /*mutableMemory=*/true);
  return builder.create<ttng::TMEMAllocOp>(
      oldTMemAllocOp.getLoc(), accMemDescType,
      builder.getType<ttg::AsyncTokenType>(), /*src=*/Value());
}

// Create a buffer array for each producer op, if the producer is in a ForOp,
// the buffer array will contain numBuffers.
DenseMap<Channel *, Value> createBuffer(const SmallVector<Channel *> &channels,
                                        triton::FuncOp funcOp, bool isPost) {

  DenseMap<Channel *, Value> bufferMap;
  MLIRContext *context = funcOp.getContext();

  // Sort channels by the positions of producer op.
  llvm::DenseMap<Operation *, uint64_t> order;
  uint64_t nextId = 0;
  funcOp->walk<WalkOrder::PreOrder>(
      [&](Operation *op) { order[op] = nextId++; });

  SmallVector<Channel *> orderedChannels = channels;
  // Reorder channels associated with one entry based on program order of the
  // producers.
  llvm::sort(orderedChannels, [&](Channel *a, Channel *b) {
    auto resultA = dyn_cast<mlir::OpResult>(a->getSrcOperand());
    auto resultB = dyn_cast<mlir::OpResult>(b->getSrcOperand());
    auto srcOpA = resultA.getDefiningOp();
    auto srcOpB = resultB.getDefiningOp();
    if (srcOpA != srcOpB)
      return order[srcOpA] < order[srcOpB]; // program order
    return resultA.getResultNumber() <
           resultB.getResultNumber(); // tie-break within same op
  });

  LLVM_DEBUG({
    LDBG("\n\n");
    LDBG(orderedChannels.size() << " ordered channels:");
    for (unsigned i = 0; i < orderedChannels.size(); i++) {
      const auto &channel = orderedChannels[i];
      LDBG("ordered channel [" << i << "]  "
                               << to_string(channel->channelKind));
    }
  });

  OpBuilderWithAsyncTaskIds builder(funcOp->getContext());
  llvm::MapVector<Channel *, SmallVector<Channel *>> channelsGroupedByProducers;

  // Group channels by source values
  // Do not group if they are in different blocks.
  llvm::MapVector<Value, SmallVector<Channel *>> repChannelsForValue;
  for (auto *channelInOrder : orderedChannels) {
    auto srcValue = channelInOrder->getSrcOperand();
    // Find the repChannel for channelInOrder, by checking srcValue and block.
    Channel *repCh = nullptr;
    if (repChannelsForValue.count(srcValue)) {
      for (auto *tCh : repChannelsForValue[srcValue]) {
        if (tCh->getDstOp()->getBlock() ==
            channelInOrder->getDstOp()->getBlock()) {
          repCh = tCh;
          break;
        }
      }
      if (repCh)
        channelsGroupedByProducers[repCh].push_back(channelInOrder);
    }
    // create a new entry
    if (!repCh) {
      repChannelsForValue[srcValue].push_back(channelInOrder);
      channelsGroupedByProducers[channelInOrder].push_back(channelInOrder);
    }
  }

  mlir::DominanceInfo dom(funcOp);
  LDBG("channels in group");
  for (auto &[repChannel, channels] : channelsGroupedByProducers) {
    auto srcValue = repChannel->getSrcOperand();
    // Find a common place for all users of the producer, which would be the
    // common dominator.
    std::unordered_set<Channel *> mutuallyNonDominatingUsers;
    for (auto user : channels) {
      LLVM_DEBUG(user->getDstOp()->dump());
      auto it = mutuallyNonDominatingUsers.begin();
      while (it != mutuallyNonDominatingUsers.end()) {
        if (dom.properlyDominates(user->getDstOp(), (*it)->getDstOp())) {
          it = mutuallyNonDominatingUsers.erase(it);
        } else if (dom.properlyDominates((*it)->getDstOp(), user->getDstOp())) {
          break;
        } else {
          ++it;
        }
      }
      if (it == mutuallyNonDominatingUsers.end())
        mutuallyNonDominatingUsers.insert(user);
    }

    auto *channel = channels.front();
    if (mutuallyNonDominatingUsers.size() == 1) {
      // Find the common parent of this user and c
      channel = *mutuallyNonDominatingUsers.begin();
    } else {
      // Check if this is a static allocation outside loops
      auto *allocOp = channel->getAllocOp();
      if (!allocOp) {
        // Try to get alloc from srcOp for SMEM/TMEM channels
        auto srcOp = channel->getSrcOp();
        if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(srcOp)) {
          allocOp = localAlloc;
        } else if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(srcOp)) {
          allocOp = tmemAlloc;
        }
      }
      bool isOutsideLoop = allocOp && !allocOp->getParentOfType<scf::ForOp>();

      if (isOutsideLoop) {
        // Static allocation outside loops - multiple consumers in different
        // sequential loops can share this buffer without pipelining.
        // Just pick the first channel, no special handling needed.
        LLVM_DEBUG({
          LDBG("Non-dominating consumers for static allocation outside loops");
          LDBG("Allocation: ");
          allocOp->dump();
          LDBG("Using first channel without pipelining");
        });
        channel = channels.front();
      } else {
        assert(false && "Non-dominating consumers unsupported");
      }
    }

    auto srcOp = channel->getSrcOp();
    auto dstOp = channel->getDstOp();
    unsigned numBuffers = channel->getNumBuffers();
    Value buffer;

    LLVM_DEBUG({
      DBGS() << "\n";
      LDBG("Creating buffers for channel [" << channel->uniqID << "] "
                                            << to_string(channel->channelKind));
      LDBG("Producer:");
      DBGS() << *srcOp << "\n";
      LDBG("Consumer:");
      DBGS() << *dstOp << "\n";
    });

    builder.setInsertionPointToStart(&(funcOp.getBody().front()));

    Value newProducer;

    // For TMEM channel, multi-buffer TMEM alloc
    if (channel->channelKind == DataChannelKind::TMEM) {
      // Move TMEM alloc to the beginning of the function.
      if (auto oldAlloc = dyn_cast<ttng::TMEMAllocOp>(srcOp)) {
        buffer = hoistLocalAlloc(builder, oldAlloc);
      } else if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(srcOp)) {
        auto oldAlloc = mmaOp.getAccumulator().getDefiningOp();
        buffer = hoistLocalAlloc(builder, oldAlloc);
      } else if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(srcOp)) {
        auto oldAlloc = storeOp.getDst().getDefiningOp();
        buffer = hoistLocalAlloc(builder, oldAlloc);
      } else if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(dstOp)) {
        auto oldAlloc = loadOp.getSrc().getDefiningOp();
        buffer = hoistLocalAlloc(builder, oldAlloc);
      } else
        llvm_unreachable("Unexpected srcOp type");
    } else if (channel->channelKind == DataChannelKind::SMEM) {
      // Move LocalAlloc to the beginning of the function.
      if (auto oldAlloc = dyn_cast<ttg::LocalAllocOp>(srcOp)) {
        buffer = hoistLocalAlloc(builder, oldAlloc);
      } else
        llvm_unreachable("Unexpected srcOp type");
    } else if (auto tensorType =
                   dyn_cast<RankedTensorType>(srcValue.getType())) {
      auto res = createLocalAlloc(
          builder, channel, isPost ? tensorType.getShape().size() == 1 : false,
          isPost);
      buffer = res.first;
      newProducer = res.second;
    } else {
      llvm_unreachable("Unexpected result type");
    }

    LLVM_DEBUG({
      LDBG("resulting buffer:");
      DBGS() << buffer << "\n";
    });

    // Channels in the group share the same buffer.
    for (auto c : channels) {
      bufferMap[c] = buffer;
    }

    // Replace all rest consumers with the loadOp
    if (newProducer) {
      for (auto c : channels) {
        auto dstOp = c->getDstOp();
        assert(c->relation.second == channel->relation.second &&
               "channels sharing the same producer must be in the same task");
        dstOp->replaceUsesOfWith(dstOp->getOperand(c->getDstOperandIdx()),
                                 newProducer);
      }
    }
  }
  return bufferMap;
}

// Update bufferMap and allocOp of channels.
static void updateChannelSharingAlloc(
    DenseMap<Channel *, SmallVector<Channel *>> &channelsSharingAlloc,
    Value buffer, Channel *channel, DenseMap<Channel *, Value> &bufferMap) {
  for (auto &kv : channelsSharingAlloc) {
    bool found = false;
    for (auto *tCh : kv.second) {
      if (tCh == channel) {
        found = true;
        break;
      }
    }
    if (found) {
      for (auto *tCh : kv.second) {
        if (tCh == channel)
          continue;
        // Update other channels in the group.
        if (tCh->channelKind == DataChannelKind::TMEMPost) {
          ttng::TmemDataChannelPost *tmemChannel =
              static_cast<ttng::TmemDataChannelPost *>(tCh);
          tmemChannel->allocOp = buffer.getDefiningOp();
        } else {
          ChannelPost *smemChannel = static_cast<ChannelPost *>(tCh);
          smemChannel->allocOp = buffer.getDefiningOp();
        }
        bufferMap[tCh] = buffer;
      }
      break;
    }
  }
}

// Need to rewrite type of the buffers to contain copies. Also all uses
// of the buffers need bufferIdx.
DenseMap<Channel *, Value> createBufferPost(
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByProducers,
    const SmallVector<Channel *> &orderedChannels, triton::FuncOp funcOp,
    ReuseConfig *config, DenseSet<Operation *> &regionsWithChannels) {

  DenseMap<Channel *, Value> bufferMap;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
#if 0
  DenseSet<Channel *> visited;
  for (auto &item : channelsGroupedByProducers) {
    auto &channels = item.second;
    for (auto c : channels) {
      assert(!visited.count(c));
      visited.insert(c);
    }
  }
#endif
  DenseMap<Channel *, SmallVector<Channel *>> channelsSharingAlloc;
  DenseSet<Channel *> handled;
  for (unsigned i = 0; i < orderedChannels.size(); ++i) {
    auto *ch = orderedChannels[i];
    if (handled.count(ch))
      continue;
    auto *alloc = ch->getAllocOp();
    assert(alloc);
    channelsSharingAlloc[ch].push_back(ch);
    handled.insert(ch);
    for (unsigned j = i + 1; j < orderedChannels.size(); ++j) {
      if (orderedChannels[j]->getAllocOp() == alloc) {
        channelsSharingAlloc[ch].push_back(orderedChannels[j]);
        handled.insert(orderedChannels[j]);
      }
    }
  }
  for (auto *channelInOrder : orderedChannels) {
    if (channelsGroupedByProducers.find(channelInOrder) ==
        channelsGroupedByProducers.end())
      continue;
    auto &channels = channelsGroupedByProducers[channelInOrder];
    auto *channel = channels.front();
    // Check to see if we have handled the allocOp.
    if (bufferMap.count(channel))
      continue;

    unsigned numBuffers = channel->getNumBuffers();
    Value buffer;
    Operation *oldAllocOp = nullptr;

    // Create multi-buffer allocs here. Do not modify channel yet.
    if (channel->channelKind == DataChannelKind::TMEMPost) {
      ttng::TmemDataChannelPost *tmemChannel =
          static_cast<ttng::TmemDataChannelPost *>(channel);
      oldAllocOp = tmemChannel->allocOp;
      OpBuilderWithAsyncTaskIds builder(oldAllocOp);
      buffer = createTMemAllocPost(
          builder, cast<ttng::TMEMAllocOp>(tmemChannel->allocOp), numBuffers);
    } else { // must be SMEMPost
      ChannelPost *smemChannel = static_cast<ChannelPost *>(channel);
      oldAllocOp = smemChannel->allocOp;
      OpBuilderWithAsyncTaskIds builder(oldAllocOp);
      buffer = hoistLocalAllocPost(
          builder, cast<ttg::LocalAllocOp>(smemChannel->allocOp), numBuffers);
    }
    buffer.getDefiningOp()->setAttr("buffer.copy",
                                    oldAllocOp->getAttr("buffer.copy"));
    buffer.getDefiningOp()->setAttr("buffer.id",
                                    oldAllocOp->getAttr("buffer.id"));
    if (oldAllocOp->getAttr("buffer.offset"))
      buffer.getDefiningOp()->setAttr("buffer.offset",
                                      oldAllocOp->getAttr("buffer.offset"));
    SmallVector<Operation *> users;
    for (auto *user : oldAllocOp->getResult(0).getUsers())
      users.push_back(user);
    DenseMap<Operation *, Value> userToBufIdx;
    int reuseGrp = channelInReuseGroup(channel, config);
    for (auto *user : users) {
      Value bufferIdx;
      Value _phase = Value();
      OpBuilderWithAsyncTaskIds builder(user);
      builder.clearLoopScheduleInfo();
      if (auto forOp = user->getParentOfType<scf::ForOp>()) {
        // Goes through channels here. Make sure the channel is not partilly
        // mutated.
        getBufferIdxAndPhase(builder, user, numBuffers, regionsWithChannels,
                             bufferIdx, _phase, config, reuseGrp, channel);
      } else {
        // For operations outside loops (epilogue), compute the
        // correct bufferIdx and phase based on the parent loop's final
        // iteration. Find the parent loop that this
        // operation came from by walking up the IR.
        std::tie(bufferIdx, _phase) = getBufferIdxAndPhaseForOutsideLoopOps(
            builder, user, channel, oldAllocOp, numBuffers, regionsWithChannels,
            config, reuseGrp);
      }
      userToBufIdx[user] = bufferIdx;
    }
    for (auto *user : oldAllocOp->getResult(0).getUsers()) {
      LLVM_DEBUG({
        LDBG("\nuser for oldAlloc ");
        user->dump();
      });
    }
    // Make modifications to IR and channels.
    for (auto *user : users) {
      Value bufferIdx = userToBufIdx[user];
      OpBuilderWithAsyncTaskIds builder(user);
      // Replace TMEM accesses.
      if (channel->channelKind == DataChannelKind::TMEMPost) {
        auto newTMemAllocOp = cast<ttng::TMEMAllocOp>(buffer.getDefiningOp());
        auto srcView = createBufferView(builder, newTMemAllocOp, bufferIdx);
        auto oldTMemAllocOp = cast<ttng::TMEMAllocOp>(oldAllocOp);
        user->replaceUsesOfWith(oldTMemAllocOp->getResult(0), srcView);
      } else {
        auto newSAllocOp = cast<ttg::LocalAllocOp>(buffer.getDefiningOp());
        auto srcView = createBufferView(builder, newSAllocOp, bufferIdx);
        auto oldSAllocOp = cast<ttg::LocalAllocOp>(oldAllocOp);
        user->replaceUsesOfWith(oldSAllocOp->getResult(0), srcView);
      }
    }
    // There is a special case where channels can share the same allocOp.
    if (channel->channelKind == DataChannelKind::TMEMPost) {
      ttng::TmemDataChannelPost *tmemChannel =
          static_cast<ttng::TmemDataChannelPost *>(channel);
      tmemChannel->allocOp = buffer.getDefiningOp();

      auto oldTMemAllocOp = cast<ttng::TMEMAllocOp>(oldAllocOp);
      auto newTMemAllocOp = cast<ttng::TMEMAllocOp>(buffer.getDefiningOp());
      if (oldTMemAllocOp.getToken())
        oldTMemAllocOp.getToken().replaceAllUsesWith(newTMemAllocOp.getToken());
    } else {
      ChannelPost *smemChannel = static_cast<ChannelPost *>(channel);
      smemChannel->allocOp = buffer.getDefiningOp();
    }
    updateChannelSharingAlloc(channelsSharingAlloc, buffer, channel, bufferMap);
    oldAllocOp->erase();
    for (auto *user : buffer.getDefiningOp()->getResult(0).getUsers()) {
      LLVM_DEBUG({
        LDBG("\nuser for newAlloc ");
        user->dump();
      });
    }
    // Channels in the group share the same buffer.
    for (auto c : channels) {
      bufferMap[c] = buffer;

      LLVM_DEBUG({
        LDBG("\nchannel after BufferPost: " << static_cast<int>(c->channelKind)
                                            << " ");
        c->getAllocOp()->dump();
      });

      if (c->getSrcOp()) {
        LLVM_DEBUG(c->getSrcOp()->dump());
      } else
        LDBG("no SrcOp");
      if (c->getDstOp()) {
        LLVM_DEBUG(c->getDstOp()->dump());
      } else
        LDBG("no DstOp");
    }
  }
  unsigned groupId = 0;
  for (unsigned idx = 0; idx < config->getGroupSize(); ++idx) {
    // TODO: add reinterpret logic
    for (auto *c : config->getGroup(idx)->channels) {
      bufferMap[c].getDefiningOp()->setAttr(
          "allocation.shareGroup",
          IntegerAttr::get(IntegerType::get(context, 32), groupId));
    }
    ++groupId;
  }
  return bufferMap;
}

// Make TCGen5MMAOp fully asynchronous by de-synchronizing it. This leverages
// its inline barrier to synchronize with both the producer (TMA load) and the
// consumer (TMEM load). Return the WaitBarrierOp inserted before the consumer
// (TMEM load). If the inline barrier is used for A/B operands of gen5,
// insert WaitBarrier as ProducerAquire; If it is used for D operand, insert
// WaitBarrier as ConsumerWait.
// Set up inline barrier for gen5 based on barrierAlloc. When asProducerAcquire
// is false, mmaOp is the producer, producerOrConsumer is the consumer, and
// we will add WaitBarrier as consumerWait in the same partition as
// producerOrConsumer. When asProducerAcquire is true, mmaOp is the consumer,
// producerOrConsumer is the producer.
// addCompletionBarrier is the logic for deciding if the barrier should be
// directly set by the MMA operation. If False we should have generated
// a tcgen05.commit Operation instead.
ttng::WaitBarrierOp
desyncTCGen5MMAOp(OpBuilderWithAsyncTaskIds &builder, ttng::TCGen5MMAOp mmaOp,
                  Value barrierAlloc, Value bufferIdx, Value inPhase,
                  unsigned numBuffers, Operation *producerOrConsumer,
                  DenseSet<Operation *> &regionsWithChannels,
                  mlir::DominanceInfo &dom, bool asProducerAcquire,
                  ReuseConfig *config, bool addCompletionBarrier) {
  // Attach the barrier as an operand of the mma op, either as producerCommit
  // or consumerRelease.
  builder.setInsertionPoint(mmaOp);
  builder.setAsyncTaskIdsFromOp(mmaOp);
  builder.setLoopScheduleInfoFromOp(mmaOp);
  if (addCompletionBarrier) {
    auto consumerBarrier =
        getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
    // assert(mmaOp.getBarriers().empty() && "mmaOp should not have barriers");
    auto pred = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        mmaOp->getLoc(), true, 1);
    mmaOp.addCompletionBarrier(consumerBarrier, pred);
  }
  mmaOp.setIsAsync(true);

  // Create a wait_barrier before producerOrConsumer. When asProducerAcquire is
  // true this wait_barrier serves as producer_acquire. When asProducerAcquire
  // is false this wait_barrier serves as consumer_wait.
  builder.setInsertionPoint(producerOrConsumer);
  builder.setAsyncTaskIdsFromOp(producerOrConsumer);
  builder.setLoopScheduleInfoFromOp(producerOrConsumer);
  auto producerBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  // curPhase = curPhase xor True for emptyBarrier.
  Value phase = inPhase;
  auto loc = producerOrConsumer->getLoc();
  if (asProducerAcquire) {
    Value _1_1b =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
    // Creating phase for producerOrConsumer.
    phase = builder.createWithAsyncTaskIds<mlir::arith::XOrIOp>(loc, inPhase,
                                                                _1_1b);
  }
  // Use zero extension (ExtUIOp) instead of sign extension (ExtSIOp)
  // When phase is i1 with value 1, ExtSIOp produces -1 (all bits set)
  // because the sign bit is 1. ExtUIOp correctly produces 1.
  phase = builder.createWithAsyncTaskIds<arith::ExtUIOp>(
      loc, builder.getI32Type(), phase);
  auto waitOp = builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
      loc, producerBarrier, phase);
  builder.clearLoopScheduleInfo();
  return waitOp;

  LLVM_DEBUG({
    LDBG("desync: create wait_barrier for producer ");
    producerBarrier.dump();
  });
#if 0
  // Create a wait_barrier before the tmem load.
  SetVector<std::pair<Operation *, unsigned>> users;
  getTransitiveUsers(mmaOp.getD(), users);
  for (auto item : users) {
    auto user = item.first;
    if (user == mmaOp)
      continue;
    // TODO: identify the real consumer of the mma op.
    // rule out users that are not dominated by op
    if (mmaOp->getBlock() != user->getBlock()) {
      if (!dom.properlyDominates(mmaOp->getParentOp(), user))
        continue;
    } else {
      if (!dom.properlyDominates(mmaOp, user))
        continue;
    }
    builder.setInsertionPoint(user);
    builder.setAsyncTaskIdsFromOp(mmaOp);
    builder.setLoopScheduleInfoFromOp(user);
    // If user and mmaOp are in the same block, we can use the same barrier.
    if (user->getBlock() != mmaOp->getBlock()) {
      // Compute the barrier from the last consumer instance
      // Extract the accum count from the consumer block.
      builder.clearLoopScheduleInfo();
      std::tie(bufferIdx, phase) = getOutOfScopeBufferIdxAndPhase(
          builder, mmaOp, numBuffers, regionsWithChannels, config, -1);
      builder.setLoopScheduleInfoFromOp(user);
      // Use zero extension (ExtUIOp) instead of sign extension (ExtSIOp)
      phase = builder.createWithAsyncTaskIds<arith::ExtUIOp>(
          user->getLoc(), builder.getI32Type(), phase);
      consumerBarrier =
          getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
    } else {
      // mmaOp can be in a different task from headProducer. Even if user and
      // mma are in the same block and they share the same barrier, but the
      // phases should be offset by 1.
      auto loc = user->getLoc();
      Value _1_1b =
          builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
      phase = builder.createWithAsyncTaskIds<mlir::arith::XOrIOp>(loc, inPhase,
                                                                  _1_1b);
      // Use zero extension (ExtUIOp) instead of sign extension (ExtSIOp)
      phase = builder.createWithAsyncTaskIds<arith::ExtUIOp>(
          loc, builder.getI32Type(), phase);
    }

    // TODO: if there are multiple users of the mma op, we need to barrier
    // before the first user.
    auto waitOp = builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
        user->getLoc(), consumerBarrier, phase);
    builder.clearLoopScheduleInfo();
    return waitOp;
  }

  llvm_unreachable("Failed to find the consumer of the mma op");
#endif
}

void replaceBufferReuse(triton::FuncOp funcOp,
                        const DenseMap<Channel *, SmallVector<Channel *>>
                            &channelsGroupedByConsumers,
                        const SmallVector<Channel *> &orderedChannels,
                        ReuseConfig *config) {
  // Multiple channels can associate with the same alloc.
  DenseSet<Operation *> handledAllocs;
  for (auto *key : orderedChannels) {
    auto it = channelsGroupedByConsumers.find(key);
    assert(it != channelsGroupedByConsumers.end());
    Channel *channel = it->second.front();
    // For each reuse group, choose a representative channel.
    int reuseGrp = channelInReuseGroup(channel, config, false /*reuseBarrier*/);
    if (reuseGrp < 0)
      continue;
    // The biggest type should be the representative.
    auto *repCh = config->getGroup(reuseGrp)->channels[0];
    if (channel != repCh && channel->getAllocOp() != repCh->getAllocOp()) {
      if (handledAllocs.count(channel->getAllocOp()))
        continue;
      LLVM_DEBUG({
        LDBG("replace users for channel with alloc "
             << channel->getAllocOp() << " in reuseGrp " << reuseGrp);
        channel->getAllocOp()->dump();
      });
      handledAllocs.insert(channel->getAllocOp());
      if (channel->channelKind == DataChannelKind::SMEMPost) {
        if (channel->getAllocOp()->getResult(0).getType() ==
            repCh->getAllocOp()->getResult(0).getType()) {
          // Types match - can do simple replacement
          SmallVector<Operation *> users;
          for (auto *user : channel->getAllocOp()->getResult(0).getUsers()) {
            users.push_back(user);
          }
          for (auto *user : users) {
            user->replaceUsesOfWith(channel->getAllocOp()->getResult(0),
                                    repCh->getAllocOp()->getResult(0));
          }
          channel->getAllocOp()->erase();
          continue;
        }
        // Types don't match for SMEM - cannot reinterpret SMEM like TMEM
        // Skip buffer reuse for this SMEM channel
        LLVM_DEBUG({
          LDBG("Type mismatch in SMEM reuse group "
               << reuseGrp << " - skipping buffer reuse");
          LDBG("Channel " << channel->uniqID << " type: "
                          << channel->getAllocOp()->getResult(0).getType());
          LDBG("Representative channel "
               << repCh->uniqID
               << " type: " << repCh->getAllocOp()->getResult(0).getType());
        });
        continue;
      }

      // Only TMEM channels reach here
      if (channel->channelKind != DataChannelKind::TMEMPost) {
        LDBG("Skipping non-TMEM channel " << channel->uniqID
                                          << " in buffer reuse");
        assert(false && "Only TMEMPost channels should reach this point");
      }

      // Verify that both channel and representative allocations are TMEM
      // sliceAndReinterpretMDTMEM only works with TMEM allocations
      bool channelIsTMEM = isa<ttng::TMEMAllocOp>(channel->getAllocOp());
      bool repChIsTMEM = isa<ttng::TMEMAllocOp>(repCh->getAllocOp());

      assert(channelIsTMEM && repChIsTMEM && "TMEM allocations required");

      // Collect all users of the allocation
      SmallVector<Operation *> users;
      for (auto *user : channel->getAllocOp()->getResult(0).getUsers()) {
        users.push_back(user);
      }

      // Single pass: create reinterpret ops and replace uses
      for (auto *user : users) {
        OpBuilderWithAsyncTaskIds builder(user->getContext());
        builder.setInsertionPoint(user);
        builder.setAsyncTaskIdsFromOp(user);
        auto bufferOff =
            channel->getAllocOp()->getAttrOfType<IntegerAttr>("buffer.offset");
        int64_t offset = bufferOff ? bufferOff.getInt() : 0;

        // Try primary representative
        auto reinter = sliceAndReinterpretMDTMEM(
            builder, repCh->getAllocOp(), channel->getAllocOp(), user, offset);

        // If primary fails, try alternative representatives
        if (!reinter) {
          for (unsigned groupIdx = 0; groupIdx < config->getGroupSize();
               ++groupIdx) {
            auto *altRepCh = config->getGroup(groupIdx)->channels[0];
            if (altRepCh == repCh)
              continue;

            reinter =
                sliceAndReinterpretMDTMEM(builder, altRepCh->getAllocOp(),
                                          channel->getAllocOp(), user, offset);

            if (reinter) {
              LDBG("Using alternative TMEM allocation from group " << groupIdx);
              break;
            }
          }
        }

        // If all representatives fail, emit error and crash
        if (!reinter) {
          channel->getAllocOp()->emitError(
              "Failed to allocate TMEM buffer: out of bounds. "
              "Cannot fall back to SMEM for TMEM/TMA allocations. "
              "Channel ID: ")
              << channel->uniqID << ", offset: " << offset;
          repCh->getAllocOp()->emitRemark(
              "Representative channel that caused the failure");
          llvm_unreachable(
              "TMEM allocation out of bounds - no SMEM fallback available");
        }

        LLVM_DEBUG({
          LDBG("replace users for channel user ");
          user->dump();
        });
        user->replaceUsesOfWith(channel->getAllocOp()->getResult(0), reinter);
        LLVM_DEBUG({
          LDBG("replace users for channel user after replacing ");
          user->dump();
        });
      }

      // All users were successfully replaced, safe to erase
      channel->getAllocOp()->erase();
    }
  }
}

// Lower producers for channels. Here channels are grouped in
// "channelsGroupedByConsumers". tokenMap tracks the set of tokens for each
// channel.
void insertAsyncComm(
    triton::FuncOp funcOp,
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByConsumers,
    const SmallVector<Channel *> &orderedChannels,
    const DenseMap<Channel *, CommChannel> &tokenMap,
    const DenseMap<Channel *, DenseMap<int, Value>> &barrierAllocMap,
    const DenseMap<Channel *, Value> &bufferMap,
    const DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseSet<Operation *> &regionsWithChannels, ReuseConfig *config,
    bool isPost) {

  // Find the operation that is along producer's parent chain, and its parent
  // is the same op as producer's parent. Here p is producer, and c is consumer.
  auto getSameLevelOp = [](Operation *p, Operation *c) -> Operation * {
    Operation *op = c;
    // Go along consumer's parent chain until it is in the same scope as
    // producer, return the current scope of consumer.
    while (!isa<triton::FuncOp>(op)) {
      if (op->getParentOp() == p->getParentOp()) {
        // consumer is in the nested region.
        return op;
      }
      op = op->getParentOp();
    }
    op = p;
    // Go along producer's parent chain until it is in the same scope as
    // consumer, return the current scope of producer.
    while (!isa<triton::FuncOp>(op)) {
      if (c->getParentOp() == op->getParentOp()) {
        return c;
      }
      op = op->getParentOp();
    }
    llvm_unreachable("Failed to find consumer's same level Op with producer");
  };

  // 0: same scope, -1: A in nested scope, 1: B in nested scope
  auto isAinNestedRegion = [](Operation *A, Operation *B) -> int {
    if (A->getBlock() == B->getBlock())
      return 0;
    Operation *op = A;
    while (!isa<triton::FuncOp>(op)) {
      if (op->getParentOp() == B->getParentOp()) {
        // A is in the nested region.
        return -1;
      }
      op = op->getParentOp();
    }
    op = B;
    while (!isa<triton::FuncOp>(op)) {
      if (op->getParentOp() == A->getParentOp()) {
        // B is in the nested region.
        return 1;
      }
      op = op->getParentOp();
    }
    llvm_unreachable("error in isAinNestedRegion");
  };

  mlir::DominanceInfo dom(funcOp);
  mlir::PostDominanceInfo pdom(funcOp);
  auto consumerReleaseHeuristic = [&](Operation *p, Operation *c,
                                      int consumerAsyncTaskId) -> Operation * {
    if (c->getBlock() != p->getBlock())
      return getSameLevelOp(p, c);

    // Find a common place for all users of the consumer, which would be the
    // common post dominator.
    auto actualConsumers = getActualConsumers(c);
    std::unordered_set<Operation *> mutuallyNonDominatingUsers;
    for (auto user : actualConsumers) {
      auto it = mutuallyNonDominatingUsers.begin();
      while (it != mutuallyNonDominatingUsers.end()) {
        if (pdom.properlyPostDominates(user, *it)) {
          it = mutuallyNonDominatingUsers.erase(it);
        } else if (pdom.properlyPostDominates(*it, user)) {
          break;
        } else {
          ++it;
        }
      }
      if (it == mutuallyNonDominatingUsers.end())
        mutuallyNonDominatingUsers.insert(user);
    }

    if (mutuallyNonDominatingUsers.size() == 1) {
      // Find the common parent of this user and c
      auto user = *mutuallyNonDominatingUsers.begin();
      while (user && user->getParentOp() != c->getParentOp())
        user = user->getParentOp();
      assert(user && "Failed to find common parent of this user and c");
      return user;
    }

    for (auto &op : reverse(c->getBlock()->getOperations())) {
      auto asyncTasks = getAsyncTaskIds(&op);
      if (asyncTasks.size() == 1 && asyncTasks[0] == consumerAsyncTaskId)
        return &op;
    }

    return nullptr;
  };

  DenseMap<ttng::TCGen5MMAOp, ttng::WaitBarrierOp> tmemWaitBarriers;

  // Postpone TMEM channels until all SMEM channels are processed.
  // TODO: Reorder the channels in channelsGroupedByConsumers in dependency
  // order. This is to ensure that we insert the synchronization primitives for
  // dependent before using it.
  SmallVector<std::pair<Channel *, SmallVector<Channel *>>>
      orderedChannelsGroupedByConsumers;
  for (auto *key : orderedChannels) {
    if (key->channelKind == DataChannelKind::SMEMPost ||
        key->channelKind == DataChannelKind::SMEM ||
        key->channelKind == DataChannelKind::REG) {
      auto kv = channelsGroupedByConsumers.find(key);
      orderedChannelsGroupedByConsumers.push_back({key, kv->second});
    }
  }
  for (auto *key : orderedChannels) {
    if (key->channelKind == DataChannelKind::TMEMPost) {
      auto kv = channelsGroupedByConsumers.find(key);
      orderedChannelsGroupedByConsumers.push_back({key, kv->second});
    }
  }

  // Go through each channel group.
  for (auto kv : orderedChannelsGroupedByConsumers) {
    // Find head and tail ops.
    DenseSet<Operation *> producerOps;
    DenseSet<Operation *> consumerOps;
    DenseSet<Operation *> actualConsumerOps;
    for (auto &c : kv.second) {
      if (isPost) {
        producerOps.insert(c->getSrcOp());
      } else {
        auto pcOp = copyOpMap.find(c)->second;
        producerOps.insert(pcOp.first);
        consumerOps.insert(pcOp.second);
      }
      if (c->channelKind == DataChannelKind::SMEMPost) {
        auto *cPost = static_cast<ChannelPost *>(c);
        SmallVector<Operation *> dsts;
        cPost->getDstOps(dsts);
        for (auto *dst : dsts) {
          consumerOps.insert(dst);
          auto consumers = getActualConsumers(dst);
          for (auto *t : consumers) {
            consumerOps.insert(t);
            actualConsumerOps.insert(t);
          }

          // If the consumer is subsequently used to perform a TMA store, we
          // would like to skip actually loading the value and just directly
          // copy it from SMEM to global memory. To make this possible, the TMA
          // store should be treated as a consumer of the channel, so that the
          // consumer release barrier is placed after the TMA store is
          // completed. Note that this is best effort, if we miss the TMA store,
          // the result will incur a performance hit, but still be correct.
          if (llvm::isa<ttg::LocalLoadOp>(dst)) {
            for (auto user : dst->getUsers()) {
              // Advance past any layout conversions, because we will be storing
              // directly from memory anyway.
              while (llvm::isa<ttg::ConvertLayoutOp>(user) && user->hasOneUse())
                user = *user->getUsers().begin();
              if (llvm::isa<tt::DescriptorStoreOp>(user)) {
                consumerOps.insert(user);
                actualConsumerOps.insert(user);
              }
            }
          }
        }
      } else {
        consumerOps.insert(c->getDstOp());
        consumerOps.insert(getUniqueActualConsumer(c->getDstOp()));
        actualConsumerOps.insert(getUniqueActualConsumer(c->getDstOp()));
      }
    }

    // Assuming all ops are under the same block.
    auto getFirstOpInBlock =
        [&](const DenseSet<Operation *> &ops) -> Operation * {
      Operation *first = *(ops.begin());
      auto block = first->getBlock();
      Operation *headOp = nullptr;
      for (auto &op : block->getOperations()) {
        if (ops.count(&op)) {
          headOp = &op;
          break;
        }
      }
      return headOp;
    };
    auto appearsBefore = [&](Operation *A, Operation *B) -> bool {
      assert(A->getBlock() == B->getBlock());
      auto block = A->getBlock();
      int AIdx = -1, BIdx = -1, cnt = 0;
      for (auto &op : block->getOperations()) {
        if (&op == A) {
          AIdx = cnt;
        }
        if (&op == B) {
          BIdx = cnt;
        }
        ++cnt;
      }
      assert(AIdx >= 0 && BIdx >= 0);
      return AIdx < BIdx;
    };

    // Find head producer
    auto producerBlock = kv.second.front()->getSrcOp()->getBlock();
    Operation *headProducer = nullptr;
    for (auto &op : producerBlock->getOperations()) {
      if (producerOps.count(&op)) {
        headProducer = &op;
        break;
      }
    }
    // Find tail producer
    Operation *tailProducer = nullptr;
    for (auto &op : reverse(producerBlock->getOperations())) {
      if (producerOps.count(&op)) {
        tailProducer = &op;
        break;
      }
    }

    // Find head consumer and tail consumer
    auto consumerBlock = kv.second.front()->getDstOp()->getBlock();
    Operation *headConsumer = nullptr;
    for (auto &op : consumerBlock->getOperations()) {
      if (consumerOps.count(&op)) {
        headConsumer = &op;
        break;
      }
    }
    Operation *tailConsumer = nullptr;
    for (auto &op : reverse(consumerBlock->getOperations())) {
      if (consumerOps.count(&op)) {
        tailConsumer = &op;
        break;
      }
    }

    // We have one set of tokens for each channel group.
    // Check if token exists (may not exist for channels we skipped in
    // createToken)
    auto tokenIt = tokenMap.find(kv.second.front());
    if (tokenIt == tokenMap.end()) {
      // Token doesn't exist - this is expected for allocations outside loops
      // that don't need async synchronization. Skip comm insertion.
      LDBG("insertAsyncComm: skipping channel group (no token) for "
           << kv.first->getAllocOp() << " - likely allocation outside loop");
      continue;
    }
    auto &commChannel = tokenIt->second;
    auto masterChannel = kv.first;

    SmallVector<AsyncTaskId> asyncTaskP;
    asyncTaskP.push_back(masterChannel->relation.first);
    SmallVector<AsyncTaskId> &asyncTaskC = masterChannel->relation.second;
    SmallVector<AsyncTaskId> asyncTasksPC = asyncTaskP;
    asyncTasksPC.insert(asyncTasksPC.end(), asyncTaskC.begin(),
                        asyncTaskC.end());

    OpBuilderWithAsyncTaskIds builder(headProducer->getContext());
    if (auto funcOp = dyn_cast<triton::FuncOp>(headProducer->getParentOp())) {
      builder.setInsertionPointToStart(&(funcOp.getBody().front()));
    } else {
      builder.setInsertionPoint(headProducer->getParentOp());
    }
    builder.setAsynTaskIdsFromArray(asyncTasksPC);

    SmallVector<tt::DescriptorLoadOp> tmaLoads;
    SmallVector<Value> buffers;
    // Go through all channels in this channel group.
    for (auto &c : kv.second) {
      if (auto *tmaLoadOp = isProducerTMA(c, isPost)) {
        auto tmaLoad = cast<tt::DescriptorLoadOp>(tmaLoadOp);
        tmaLoads.push_back(tmaLoad);
        buffers.push_back(bufferMap.find(c)->second);
      }
    }

    Value bufferIdx;
    Value phase = Value();
    Operation *tmaHeadProducer = headProducer;
    {
      DenseSet<Operation *> tOps;
      for (auto tOp : tmaLoads)
        tOps.insert(tOp.getOperation());
      tOps.insert(headProducer);
      tmaHeadProducer = getFirstOpInBlock(tOps);
    }

    auto withSameTask = [&](Operation *A, Operation *B) -> bool {
      auto aTasks = getAsyncTaskIds(A);
      auto bTasks = getAsyncTaskIds(B);
      return aTasks == bTasks;
    };

    // Return the backward channel if found.
    // Assume chF is a forward channel where producer and consumer are in the
    // same block.
    auto isForwardOfChannelLoop = [&](Channel *chF) -> Channel * {
      if (chF->channelKind != DataChannelKind::TMEMPost)
        return nullptr;
      ttng::TmemDataChannelPost *tmemChannel =
          static_cast<ttng::TmemDataChannelPost *>(chF);
      if (!tmemChannel->isOperandD)
        return nullptr;
      // Check for a cycle, a channel from chF->getDstOp to an op prior to
      // chF->getSrcOp and all users are in the same block.
      for (auto *ch : orderedChannels) {
        if (ch == chF)
          continue;
        if (withSameTask(ch->getDstOp(), chF->getSrcOp()) &&
            ch->getAllocOp() == chF->getAllocOp() &&
            ch->getSrcOp() == chF->getDstOp() &&
            chF->getSrcOp()->getBlock() == ch->getSrcOp()->getBlock() &&
            chF->getSrcOp()->getBlock() == ch->getDstOp()->getBlock()) {
          if (appearsBefore(ch->getDstOp(), chF->getSrcOp()))
            return ch;
        }
      }
      return nullptr;
    };
    // Assume chB is a backward channel where producer and consumer are in the
    // same block.
    auto isBackwardOfChannelLoop = [&](Channel *chB) -> bool {
      if (chB->channelKind != DataChannelKind::TMEMPost)
        return false;
      ttng::TmemDataChannelPost *tmemChannel =
          static_cast<ttng::TmemDataChannelPost *>(chB);
      if (!tmemChannel->isOperandD)
        return false;
      // Check for a cycle, a channel from an op after chB->getDstOp to
      // chB->getSrcOp and all users are in the same block.
      for (auto *ch : orderedChannels) {
        if (ch == chB)
          continue;
        if (withSameTask(ch->getSrcOp(), chB->getDstOp()) &&
            ch->getAllocOp() == chB->getAllocOp() &&
            ch->getDstOp() == chB->getSrcOp() &&
            chB->getSrcOp()->getBlock() == ch->getSrcOp()->getBlock() &&
            chB->getSrcOp()->getBlock() == ch->getDstOp()->getBlock()) {
          if (appearsBefore(chB->getDstOp(), ch->getSrcOp()))
            return true;
        }
      }
      return false;
    };
    Operation *nestedInsertionTarget = nullptr;
    // Check to see if producer and consumer are in the same block.
    bool producerInNestedRegion = false, consumerInNestedRegion = false;
    if (headProducer->getBlock() != headConsumer->getBlock()) {
      LDBG("different blocks for channel " << masterChannel->uniqID);
      int regionCmp = isAinNestedRegion(headProducer, headConsumer);
      if (regionCmp < 0) {
        // A/producer in nested region. Lift up headProducer till it is
        // in the same scope as headConsumer.
        assert(isa<ttng::TCGen5MMAOp>(headProducer) &&
               "Only TCGen5MMAOp supported");
        nestedInsertionTarget = getSameLevelOp(headConsumer, headProducer);
        producerInNestedRegion = true;
      } else if (regionCmp > 0) {
        // B/consumer in nested region. Lift up headConsumer till it is
        // in the same scope as headProducer.
        nestedInsertionTarget = getSameLevelOp(tmaHeadProducer, headConsumer);
        consumerInNestedRegion = true;
      }
    } else {
      // Check to see if consumer appears later than producer (loop-carried).
      if (!appearsBefore(headProducer, headConsumer)) {
        // We will combine this channel with the other channel associated with
        // the same value (gen5 operandD).
        // -- Both channels are in the same block
        // -- One channel is a forward edge, the other is a back edge.
        // When handling the forward edge, we put a consumer release with gen5
        // and a consumer wait prior to gen5, we also put a producer acquire
        // before the srcOp of the channel and a producer commit after the
        // srcOp. Instead, we need to move the producer acquire to be prior to
        // the dstOp of the backward channel. We will have:
        //   tmem_load(dstOp of channel B) ...
        //   tmem_store(srcOp of channel F) ...
        //   gen5(srcOp of channel B, dstOp of channel F)
        // We should emit:
        //   producer_acquire
        //   tmem_load(dstOp of channel B) ...
        //   tmem_store(srcOp of channel F)
        //   producer_commit ...
        //   consumer_wait (gen5 partition)
        //   gen5 consumer_release (srcOp of channel B, dstOp of channel F)
        assert(isBackwardOfChannelLoop(masterChannel));
        LDBG("Skip consumer before producer for channel "
             << masterChannel->uniqID);
        continue;
      }
    }
    Operation *producerAcquireForChannelLoop = nullptr;
    if (headProducer->getBlock() == headConsumer->getBlock()) {
      auto *bwdCh = isForwardOfChannelLoop(masterChannel);
      if (bwdCh)
        producerAcquireForChannelLoop = bwdCh->getDstOp();
    }
    int reuseGrp = channelInReuseGroup(masterChannel, config);
    builder.clearLoopScheduleInfo();
    if (nestedInsertionTarget) {
      // If the producer is nested we need to pull the buffer + index
      // calculation to the lift-up headProducer.
      if (producerInNestedRegion) {
        builder.setInsertionPoint(nestedInsertionTarget);
      } else {
        assert(consumerInNestedRegion);
        builder.setInsertionPoint(tmaHeadProducer);
      }
      LLVM_DEBUG({
        LDBG("call getBufferIdxAndPhase3 ");
        nestedInsertionTarget->dump();
      });
      getBufferIdxAndPhase(builder, nestedInsertionTarget,
                           kv.second.front()->getNumBuffers(),
                           regionsWithChannels, bufferIdx, phase, config,
                           reuseGrp, masterChannel);
    } else if (auto forOp = headProducer->getParentOfType<scf::ForOp>()) {
      // headProducer can be local_store but bufferIdx will be used
      // by tmaLoad as well.
      if (producerAcquireForChannelLoop) {
        builder.setInsertionPoint(producerAcquireForChannelLoop);
      } else {
        builder.setInsertionPoint(tmaHeadProducer);
      }
      LLVM_DEBUG({
        LDBG("call getBufferIdxAndPhase2 ");
        headProducer->dump();
      });
      getBufferIdxAndPhase(builder, headProducer,
                           kv.second.front()->getNumBuffers(),
                           regionsWithChannels, bufferIdx, phase, config,
                           reuseGrp, masterChannel);
    } else {
      // Producer is not in a ForOp, create phase and bufferIdx here.
      bufferIdx = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 32);
      phase = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 1);
    }

    // Lower TMA loads and TCGen5MMAOp first before inserting synchronization
    // primitives to avoid displacement.

    LLVM_DEBUG({
      LDBG("SrcOp of master Channel " << masterChannel->uniqID << " ");
      masterChannel->getSrcOp()->dump();
      LDBG("DstOp of master Channel ");
      masterChannel->getDstOp()->dump();
      LDBG("headProducer ");
      headProducer->dump();
      LDBG("tailProducer ");
      tailProducer->dump();
      LDBG("headConsumer ");
      headConsumer->dump();
      LDBG("tailConsumer ");
      tailConsumer->dump();
    });

    builder.setAsynTaskIdsFromArray(masterChannel->relation.first);

    if (commChannel.producerBarrier) {
      // If we are using producer barrier, it is either TMA or gen5. Handle gen5
      // here, TMA will be handled later.
      Operation *mmaOp = dyn_cast<ttng::TCGen5MMAOp>(headProducer);
      if (mmaOp) {
        // Add one barrier to gen5 for producer_commit, also insert WaitBarrier
        // (consumer_wait) at headConsumer to wait till gen5 is done so we can
        // start using the output (D operand).
        LLVM_DEBUG({
          LDBG("channel has gen5 mma as producer " << masterChannel->uniqID
                                                   << " ");
        });
        // If we have a nested target we cannot use the barrier in the
        // TCGen5MMAOp directly and instead need a tcgen05.commit.
        bool addCompletionBarrier = nestedInsertionTarget == nullptr;
        if (!addCompletionBarrier) {
          // We need to place the commit after the for loop.
          builder.setInsertionPointAfter(nestedInsertionTarget);
          builder.setLoopScheduleInfoFromOp(nestedInsertionTarget);
          builder.setAsyncTaskIdsFromOp(mmaOp);
          builder.createWithAsyncTaskIds<ttng::TCGen5CommitOp>(
              mmaOp->getLoc(), *commChannel.producerBarrier);
          builder.clearLoopScheduleInfo();
        }
        // Still call desyncTCGen5MMAOp to handle the consumer.
        desyncTCGen5MMAOp(builder, cast<ttng::TCGen5MMAOp>(mmaOp),
                          *commChannel.producerBarrier, bufferIdx, phase,
                          masterChannel->getNumBuffers(), headConsumer,
                          regionsWithChannels, dom, false, config,
                          addCompletionBarrier);
      }
    }
    // Channel can have multiple consumers.
    for (auto &consumerTaskId : masterChannel->relation.second) {
      // Set up consumer release and producer acquire for channel where consumer
      // is gen5.
      if (commChannel.consumerBarriers.count(consumerTaskId)) {
        // filter with consumerTaskId
        DenseSet<Operation *> filteredOps;
        for (auto *tCon : actualConsumerOps) {
          SmallVector<AsyncTaskId> asyncTasks = getAsyncTaskIds(tCon);

          // Handle operations that belong to multiple tasks (e.g., boundary
          // ops) Only include if this consumer belongs to the task we're
          // processing
          if (asyncTasks.empty()) {
            LLVM_DEBUG({
              LDBG("Skipping operation with no async tasks");
              tCon->dump();
            });
            continue;
          }

          if (std::find(asyncTasks.begin(), asyncTasks.end(), consumerTaskId) !=
              asyncTasks.end()) {
            filteredOps.insert(tCon);
            // XXX: Op can have multiple async tasks
          }
        }
        // Get the last mmaOp.
        auto *lastConsumer = getLastOpInBlock(filteredOps);
        auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(lastConsumer);
        if (!mmaOp)
          continue;
        // Assume a single task for mmaOp.
        SmallVector<AsyncTaskId> asyncTasksMma = getAsyncTaskIds(mmaOp);
        assert(asyncTasksMma.size() == 1 && asyncTasksMma[0] == consumerTaskId);
        LLVM_DEBUG({
          LDBG("unique actual consumer is gen5 mma " << masterChannel->uniqID
                                                     << " ");
          mmaOp->dump();
        });
        auto iter = commChannel.consumerBarriers.find(consumerTaskId);
        Value consumerBarrier = iter->second;
        // Use consumerBarrier as gen5 inline barrier.
        // Correctly set the insertion point for producerAcquire when there is a
        // tma/gen5 channel.
        Operation *producerAcquirePoint = headProducer;
        if (isProducerTMA(masterChannel, isPost))
          producerAcquirePoint = tmaHeadProducer;
        if (producerAcquireForChannelLoop) {
          LLVM_DEBUG({
            LDBG("move producer acquire for inline barrier "
                 << masterChannel->uniqID << " ");
            producerAcquireForChannelLoop->dump();
          });
          producerAcquirePoint = producerAcquireForChannelLoop;
        }
        bool addCompletionBarrier = nestedInsertionTarget == nullptr;
        if (!addCompletionBarrier) {
          // We need to place the commit after the for loop.
          builder.setInsertionPointAfter(nestedInsertionTarget);
          builder.setLoopScheduleInfoFromOp(nestedInsertionTarget);
          builder.setAsyncTaskIdsFromOp(mmaOp);
          builder.createWithAsyncTaskIds<ttng::TCGen5CommitOp>(mmaOp->getLoc(),
                                                               consumerBarrier);
          builder.clearLoopScheduleInfo();
        }
        auto tmemWaitBarrier = desyncTCGen5MMAOp(
            builder, mmaOp, consumerBarrier, bufferIdx, phase,
            masterChannel->getNumBuffers(), producerAcquirePoint,
            regionsWithChannels, dom, true, config, addCompletionBarrier);
        tmemWaitBarriers[mmaOp] = tmemWaitBarrier;
      }
    }

    for (const auto &token : commChannel.tokens) {
      // Use token for producer acquire and consumer release.
      if (commChannel.consumerBarriers.empty()) {
        // Insert ProducerAcquireOp before the producer.
        // Even when A is nested inside B we still need to place
        // the acquire right before the head producer to avoid
        // reordering the barriers incorrectly. This acquire will
        // be idemponent in the loop because we don't flip the phase.
        auto producerAcquirePoint =
            getSameLevelOp(headConsumer, tmaHeadProducer); // tmaHeadProducer;
        builder.setAsynTaskIdsFromArray(masterChannel->relation.first);
        if (producerAcquireForChannelLoop) {
          builder.setInsertionPoint(producerAcquireForChannelLoop);
          builder.setLoopScheduleInfoFromOp(producerAcquireForChannelLoop);
        } else {
          builder.setInsertionPoint(producerAcquirePoint);
          builder.setLoopScheduleInfoFromOp(producerAcquirePoint);
        }
        auto acquireOp =
            builder.createWithAsyncTaskIds<ttnvws::ProducerAcquireOp>(
                headProducer->getLoc(), token.second, bufferIdx, phase);
        LLVM_DEBUG({
          LDBG("Insert ProducerAcquireOp " << masterChannel->uniqID << " ");
          producerAcquirePoint->dump();
        });
      }

      if (!commChannel.producerBarrier) {
        // When there is no producer barrier, we will emit both ProducerCommit
        // and ConsumerWait. Otherwise, there is no explicit ProducerCommit,
        // and ConsumerWait will be on the producerBarrier via WaitBarrierOp
        // which is handled else where.
        Operation *producerCommitPoint;
        if (masterChannel->channelKind == DataChannelKind::TMEM) {
          // There is one case where gen5 takes an input acc and an input for
          // operand A from the same task. Delay the commit.
          ttng::TmemDataChannel *tmemChannel =
              static_cast<ttng::TmemDataChannel *>(masterChannel);
#if 0
          assert(tmemWaitBarriers.count(tmemChannel->tmemMmaOp) &&
                 "Failed to find tmemWaitBarriers");
          producerCommitPoint = tmemWaitBarriers[tmemChannel->tmemMmaOp];
#endif
          bool handled = false;
          // This TMEM channel's producer is TMEMStore, and it feeds into
          // operand A of gen5.
          if (auto producerSt = dyn_cast<ttng::TMEMStoreOp>(tailProducer)) {
            auto producerAllocOp = producerSt.getDst().getDefiningOp();
            if (producerAllocOp->getResult(0) ==
                tmemChannel->tmemMmaOp.getA()) {
              // Check for operand D of tmemMmaOp.
              Value dOpnd = tmemChannel->tmemMmaOp.getD();
              // Check for tmem_store of operand D.
              auto allocOp = dOpnd.getDefiningOp();
              for (auto user : allocOp->getUsers()) {
                if (auto tmSt = dyn_cast<ttng::TMEMStoreOp>(user)) {
                  if (user->getBlock() != tailProducer->getBlock())
                    break;

                  Operation *laterSt = nullptr;
                  for (auto &op : reverse(user->getBlock()->getOperations())) {
                    if (&op == tmSt || &op == tailProducer) {
                      laterSt = &op;
                      break;
                    }
                  }
                  producerCommitPoint =
                      laterSt; // later point of tailProducer or tmemStore.
                  handled = true;
                  LDBG("Insert ProducerCommitOp at the later tmem_store"
                       << masterChannel->uniqID << " ");
                  break;
                }
              }
            }
          }
          if (!handled)
            producerCommitPoint = getSameLevelOp(headConsumer, tailProducer);
        } else {
          producerCommitPoint = getSameLevelOp(headConsumer, tailProducer);
        }
        LLVM_DEBUG({
          LDBG("Insert ProducerCommitOp " << masterChannel->uniqID << " ");
          producerCommitPoint->dump();
        });
        builder.setInsertionPointAfter(producerCommitPoint);
        builder.setLoopScheduleInfoFromOp(producerCommitPoint);
        auto commitOp =
            builder.createWithAsyncTaskIds<ttnvws::ProducerCommitOp>(
                tailProducer->getLoc(), token.second, bufferIdx);
      }
    }

    for (const auto &token : commChannel.tokens) {
      builder.setAsynTaskIdsFromArray(token.first);
      // Insert ConsumerWaitOp
      if (!commChannel.producerBarrier) {
        auto consumerWaitPoint = getSameLevelOp(headProducer, headConsumer);
        builder.setInsertionPoint(consumerWaitPoint);
        builder.setLoopScheduleInfoFromOp(consumerWaitPoint);
        auto waitOp = builder.createWithAsyncTaskIds<ttnvws::ConsumerWaitOp>(
            headConsumer->getLoc(), token.second, bufferIdx, phase);
        LDBG("create ConsumerWait " << masterChannel->uniqID << " ");
      }

      // Insert ConsumerReleaseOp, if consumer is not a TCGen5MMAOp. For
      // TCGen5MMAOp, TCGen5MMAOp lowering will handle the ConsumerReleaseOp.
      if (commChannel.consumerBarriers.empty()) {
        auto consumerReleasePoint =
            consumerReleaseHeuristic(tailProducer, tailConsumer, token.first);
        builder.setInsertionPointAfter(consumerReleasePoint);
        builder.setLoopScheduleInfoFromOp(consumerReleasePoint);
        auto releaseOp =
            builder.createWithAsyncTaskIds<ttnvws::ConsumerReleaseOp>(
                consumerReleasePoint->getLoc(), token.second, bufferIdx);
        LLVM_DEBUG({
          LDBG("create ConsumerRelease " << masterChannel->uniqID << " ");
          token.second.dump();
        });
      }
    }

    // Optimize TMA loads.
    if (tmaLoads.size() > 0) {
      // Instead of headConsumer, need to lift out to the same scope.
      auto consumerWaitPoint = getSameLevelOp(tmaHeadProducer, headConsumer);
      optimizeTMALoads(builder, tmaLoads, buffers, *commChannel.producerBarrier,
                       bufferIdx, bufferIdx, phase, tmaHeadProducer,
                       headConsumer, consumerWaitPoint, isPost);
    }
  }

  // Clean up tokens that are not used anymore.
  // Remove an LocalAllocOp op if it is only used by
  // MemDescIndexOp/InitBarrierOp
  DenseSet<Value> removedBarriers;
  auto removeTokenfNotUsed = [&](Value barrier) {
    if (removedBarriers.count(barrier))
      return;
    if (barrier.use_empty()) {
      barrier.getDefiningOp()->erase();
      removedBarriers.insert(barrier);
      return;
    }

    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(barrier.getDefiningOp())) {
      // Check: alloc result is only used once
      if (!alloc->hasOneUse())
        return;

      Operation *memDescUser = *alloc->user_begin();
      auto memDesc = dyn_cast<ttg::MemDescIndexOp>(memDescUser);
      if (!memDesc || !memDesc->hasOneUse())
        return;

      Operation *idxUser = *memDesc->user_begin();
      if (isa<ttng::InitBarrierOp>(idxUser)) {
        // Safe to erase: drop uses first then erase ops
        idxUser->erase();
        memDesc->dropAllUses();
        memDesc->erase();
        alloc->erase();
        removedBarriers.insert(barrier);
      }
    }
  };

  for (auto commChannel : tokenMap) {
    if (commChannel.second.producerBarrier)
      removeTokenfNotUsed(*commChannel.second.producerBarrier);

    for (auto &barrier : commChannel.second.consumerBarriers)
      removeTokenfNotUsed(barrier.second);

    for (auto &token : commChannel.second.tokens)
      removeTokenfNotUsed(token.second);
  }
}

void foldLocalLoads(triton::FuncOp funcOp) {
  // If loadResult has a single use which is LocalAlloc, we can get rid of
  // sharedLoad and replace all uses of LocalAlloc with viewLoad.
  DenseMap<Operation *, Value> opsToReplace;
  funcOp.walk([&](ttg::LocalAllocOp localAlloc) {
    if (auto src = localAlloc.getSrc()) {
      if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(src.getDefiningOp())) {
        // Only fold within the same tasks
        if (getAsyncTaskIds(localLoad) == getAsyncTaskIds(localAlloc)) {
          opsToReplace[localAlloc] = localLoad.getSrc();
        }
      }
    }
  });
  OpBuilderWithAsyncTaskIds builder(funcOp.getContext());
  for (auto kv : opsToReplace)
    mlir::triton::replaceUsesAndPropagateType(builder, kv.getFirst(),
                                              kv.getSecond());
}

// Compare against TritonNvidiaGPURemoveTMEMTokensPass.
static void cleanupTmemTokens(triton::FuncOp funcOp) {
  auto b = OpBuilder::atBlockBegin(&funcOp.getBody().front());
  Value replTok =
      b.create<ub::PoisonOp>(funcOp.getLoc(), b.getType<ttg::AsyncTokenType>());
  funcOp.walk([&](Operation *op) {
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      storeOp.getDepMutable().clear();
      if (storeOp.getToken())
        storeOp.getToken().replaceAllUsesWith(replTok);
    } else if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      loadOp.getDepMutable().clear();
      if (loadOp.getToken())
        loadOp.getToken().replaceAllUsesWith(replTok);
    } else if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(op)) {
      mmaOp.getAccDepMutable().clear();
      if (mmaOp.getToken())
        mmaOp.getToken().replaceAllUsesWith(replTok);
    } else if (auto alloc = dyn_cast<ttng::TMEMAllocOp>(op)) {
      if (alloc.getToken())
        alloc.getToken().replaceAllUsesWith(replTok);
    }
  });
}

void doBufferAllocation(triton::FuncOp &funcOp) {
  // Step 1: collect all communications between producers and consumers.
  SmallVector<std::unique_ptr<Channel>> channelsOrigin;
  collectAsyncChannels(channelsOrigin, funcOp, 1 /*numBuffers*/);
  SmallVector<Channel *> channels;
  for (const auto &c : channelsOrigin) {
    channels.push_back(c.get());
  }
  if (channels.empty()) {
    return;
  }

  // Step 2: Reorder ops based on channel information.
  reorderEpilogOps(channels, funcOp);

  // Step 3: Create buffers. A buffer for each channel.
  createBuffer(channels, funcOp, true);
}

void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers) {
  // Step 1: collect all communications between producers and consumers.
  SmallVector<std::unique_ptr<Channel>> channelsOrigin;
  collectAsyncChannels(channelsOrigin, funcOp, numBuffers);
  SmallVector<Channel *> channels;
  for (const auto &c : channelsOrigin) {
    channels.push_back(c.get());
  }
  if (channels.empty()) {
    return;
  }

  // Step 2: group channels
  // -  each entry of the channelsGroupedByProducers is keyed by the srcOp.
  // -  each entry of the channelsGroupedByConsumers is keyed by the dstOp.
  DenseMap<Channel *, SmallVector<Channel *>> channelsGroupedByProducers;
  DenseMap<Channel *, SmallVector<Channel *>> channelsGroupedByConsumers;
  SmallVector<Channel *> orderedChannels;
  groupChannels(channels, channelsGroupedByProducers,
                channelsGroupedByConsumers, orderedChannels);

  // Step 3: Create buffers. An array of buffers for each channel.
  DenseMap<Channel *, Value> bufferMap = createBuffer(channels, funcOp, false);
  LLVM_DEBUG({
    LDBG("\n\nafter createBuffer");
    funcOp.dump();
  });

  // Step 4: reorder producer ops and the backward slices of the producer ops.
  reorderProducerOps(channels);

  // Step 5: find top-level ops that contain a channel, also create new ForOps
  // by adding phase and bufferIdx to the original ForOps, erase the original
  // ForOps.
  SmallVector<Operation *> asyncTaskTopOps = getTaskTopRegion(funcOp, channels);
  SmallVector<Operation *> opList;
  for (auto &op : asyncTaskTopOps) {
    if (auto origIfOp = dyn_cast<scf::IfOp>(op)) {
      opList.push_back(op);
    }
    if (auto origForOp = dyn_cast<scf::ForOp>(op))
      opList.push_back(op);
  }
  DenseSet<Operation *> regionsWithChannels;
  collectRegionsWithChannels(channels, regionsWithChannels);
  ReuseConfig config;
  appendAccumCntsForOps(asyncTaskTopOps, channels, regionsWithChannels,
                        &config);
  LLVM_DEBUG({
    LDBG("\n\nafter appendAccumCntsForOps");
    funcOp.dump();
  });

  // Step 6: Lower the loads. Also add local copy ops for non-load
  // producers.
  DenseMap<Channel *, std::pair<Operation *, Operation *>> copyOpMap;
  insertAsyncCopy(funcOp, channelsGroupedByProducers, bufferMap, copyOpMap,
                  regionsWithChannels, &config);
  LLVM_DEBUG({
    LDBG("\n\nwith async copy");
    funcOp.dump();
  });

  // Step 7: Create tokens. A set of tokens for each group of channels for
  // each channel.
  DenseMap<Channel *, DenseMap<int, Value>> barrierAllocMap;
  DenseMap<Channel *, CommChannel> tokenMap;
  createToken(channelsGroupedByConsumers, orderedChannels, funcOp, copyOpMap,
              tokenMap, &config);
  LLVM_DEBUG({
    LDBG("\n\nafter createToken");
    funcOp.dump();
  });

  // Step 8: add async communication ops (ProducerAcquire etc). Also lower
  // TMA loads.
  insertAsyncComm(funcOp, channelsGroupedByConsumers, orderedChannels, tokenMap,
                  barrierAllocMap, bufferMap, copyOpMap, regionsWithChannels,
                  &config, false);
  LLVM_DEBUG({
    LDBG("\n\nwith SyncOps");
    funcOp.dump();
  });

  // If loadResult has a single use which is LocalAlloc, we can get rid of
  // sharedLoad and replace all uses of LocalAlloc with viewLoad.
  foldLocalLoads(funcOp);
  LLVM_DEBUG({
    LDBG("\n\nsimplify localLoad + localAlloc");
    funcOp.dump();
  });

  specializeRegion(funcOp, 0 /*requestedRegisters*/);
  LLVM_DEBUG({
    LDBG("\n\nwith specializeRegion");
    funcOp.dump();
  });
}

void doCodePartitionPost(triton::FuncOp &funcOp, unsigned numBuffers) {
  // Step 1: collect all communications between producers and consumers.
  SmallVector<std::unique_ptr<Channel>> channelsOrigin;
  collectPostChannels(channelsOrigin, funcOp);
  SmallVector<Channel *> channels;
  for (const auto &c : channelsOrigin) {
    channels.push_back(c.get());
  }
  if (channels.empty()) {
    return;
  }
  SmallVector<Channel *> orderedChannels;
  orderedChannels = channels;
  std::sort(orderedChannels.begin(), orderedChannels.end(),
            [&](Channel *a, Channel *b) { return a->uniqID < b->uniqID; });
  DenseMap<Channel *, SmallVector<Channel *>> channelsGroupedByProducers;
  DenseMap<Channel *, SmallVector<Channel *>> channelsGroupedByConsumers;
  for (auto *ch : orderedChannels) {
    channelsGroupedByProducers[ch].push_back(ch);
  }
  for (auto *ch : orderedChannels) {
    channelsGroupedByConsumers[ch].push_back(ch);
  }
  // Step 2: find top-level ops that contain a channel, also create new ForOps
  // by adding phase and bufferIdx to the original ForOps, erase the original
  // ForOps.
  SmallVector<Operation *> asyncTaskTopOps = getTaskTopRegion(funcOp, channels);
  SmallVector<Operation *> opList;
  for (auto &op : asyncTaskTopOps) {
    if (auto origIfOp = dyn_cast<scf::IfOp>(op)) {
      opList.push_back(op);
    }
    if (auto origForOp = dyn_cast<scf::ForOp>(op))
      opList.push_back(op);
  }
  DenseSet<Operation *> regionsWithChannels;
  collectRegionsWithChannelsPost(channels, regionsWithChannels);
  ReuseConfig config;
  DenseMap<unsigned, std::vector<Channel *>> bufferIdToChannels;
  for (auto *ch : orderedChannels) {
    Operation *allocOp;
    if (ch->channelKind == DataChannelKind::TMEMPost) {
      ttng::TmemDataChannelPost *tmemChannel =
          static_cast<ttng::TmemDataChannelPost *>(ch);
      allocOp = tmemChannel->allocOp;
    } else {
      ChannelPost *smemChannel = static_cast<ChannelPost *>(ch);
      allocOp = smemChannel->allocOp;
    }
    if (auto bufferId = allocOp->getAttrOfType<IntegerAttr>("buffer.id")) {
      bufferIdToChannels[bufferId.getInt()].push_back(ch);
      LLVM_DEBUG({
        LDBG("\nchannel with allocOp: " << static_cast<int>(ch->channelKind)
                                        << " " << ch->uniqID << " ");
        allocOp->dump();
      });
    } else
      assert(false);
  }
  for (auto kv : bufferIdToChannels) {
    if (kv.second.size() > 1) {
      ReuseGroup group;
      // make sure the channel without buffer.offset is the first one (i.e the
      // representative channel)
      std::vector<Channel *> ordered(kv.second);
      std::stable_partition(ordered.begin(), ordered.end(), [](Channel *ch) {
        auto bufferOffset =
            ch->getAllocOp()->getAttrOfType<IntegerAttr>("buffer.offset");
        if (bufferOffset)
          return false;
        return true;
      });
      group.channels = ordered;
      LDBG("ReuseGroup with size " << kv.second.size() << " buffer.id "
                                   << kv.first << "\n");
      config.groups.push_back(group);
    }
  }
  appendAccumCntsForOps(asyncTaskTopOps, channels, regionsWithChannels,
                        &config);
  LLVM_DEBUG({
    LDBG("\n\nafter appendAccumCntsForOps");
    funcOp.dump();
  });
  // Step 5: Create buffers. An array of buffers for each channel.
  DenseMap<Channel *, Value> bufferMap =
      createBufferPost(channelsGroupedByProducers, channels, funcOp, &config,
                       regionsWithChannels);
  LLVM_DEBUG({
    LDBG("\n\nafter createBuffer");
    funcOp.dump();
  });

  // Step 6: Lower the loads. Local copy ops for non-load
  // producers should have been handled prior.
  DenseMap<Channel *, std::pair<Operation *, Operation *>> copyOpMap;
#if 0
  insertAsyncCopy(funcOp, channelsGroupedByProducers, bufferMap, copyOpMap,
                  regionsWithChannels, &config, true /*isPost*/);
  LLVM_DEBUG({
    LDBG("\n\nwith async copy");
    funcOp.dump();
  });
#endif

  // Step 7: Create tokens. A set of tokens for each group of channels for
  // each channel.
  DenseMap<Channel *, DenseMap<int, Value>> barrierAllocMap;
  DenseMap<Channel *, CommChannel> tokenMap;
  createTokenPost(channelsGroupedByConsumers, orderedChannels, funcOp,
                  copyOpMap, tokenMap, &config);
  LLVM_DEBUG({
    LDBG("\n\nafter createToken");
    funcOp.dump();
  });

  // Step 8: add async communication ops (ProducerAcquire etc). Also lower
  // TMA loads.
  insertAsyncComm(funcOp, channelsGroupedByConsumers, orderedChannels, tokenMap,
                  barrierAllocMap, bufferMap, copyOpMap, regionsWithChannels,
                  &config, true);
  LLVM_DEBUG({
    LDBG("\n\nwith SyncOps");
    funcOp.dump();
  });

  // Prune any unnecessary barriers related to tgen05.commit
  fuseTcgen05CommitBarriers(funcOp);
  LLVM_DEBUG({
    LDBG("\n\nPruned tcgen05 commit barriers");
    funcOp.dump();
  });

  // If loadResult has a single use which is LocalAlloc, we can get rid of
  // sharedLoad and replace all uses of LocalAlloc with viewLoad.
  foldLocalLoads(funcOp);
  LLVM_DEBUG({
    LDBG("\n\nsimplify localLoad + localAlloc");
    funcOp.dump();
  });

  // Clean up Tokens for tmem, tokens should be threaded within the partitions.
  // This should also clean up tokens in the ForOp arguments.
  cleanupTmemTokens(funcOp);
  LLVM_DEBUG({
    LDBG("\n\nclean up tmem tokens");
    funcOp.dump();
  });

  // Replace buffer reuses
  replaceBufferReuse(funcOp, channelsGroupedByConsumers, orderedChannels,
                     &config);
  LLVM_DEBUG({
    LDBG("\n\nreplace buffer reuse");
    funcOp.dump();
  });

  specializeRegion(funcOp, 0 /*requestedRegisters*/);
  LLVM_DEBUG({
    LDBG("\n\nwith specializeRegion");
    funcOp.dump();
  });
}

#define GEN_PASS_DEF_NVGPUTESTWSCODEPARTITION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSCodePartitionPass
    : public impl::NVGPUTestWSCodePartitionBase<NVGPUTestWSCodePartitionPass> {
public:
  using impl::NVGPUTestWSCodePartitionBase<
      NVGPUTestWSCodePartitionPass>::NVGPUTestWSCodePartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    // Disable code partitioning when numBuffers is 0.
    if (numBuffers > 0) {
      if (postChannelCreation > 0)
        doCodePartitionPost(funcOp, numBuffers);
      else
        doCodePartition(funcOp, numBuffers);
    }
  }
  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
    LLVM_DEBUG({
      LDBG("post pass");
      getOperation()->dump();
    });
    return;
  }
};

#define GEN_PASS_DEF_NVGPUTESTWSBUFFERALLOCATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSBufferAllocationPass
    : public impl::NVGPUTestWSBufferAllocationBase<
          NVGPUTestWSBufferAllocationPass> {
public:
  using impl::NVGPUTestWSBufferAllocationBase<
      NVGPUTestWSBufferAllocationPass>::NVGPUTestWSBufferAllocationBase;

  void runOnFuncOp(triton::FuncOp funcOp) { doBufferAllocation(funcOp); }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
