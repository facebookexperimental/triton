#include "CodePartitionUtility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-utility"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Check to see if op is enclosed under ifOp.
bool enclosing(scf::IfOp ifOp, Operation *op) {
  return ifOp->isProperAncestor(op);
}

bool enclosing(scf::ForOp forOp, Operation *op) {
  return forOp->isProperAncestor(op);
}

// After createBufferPost, MemDescIndexOp will be used.
Operation *skipIdxOp(Operation *op) {
  if (auto idx = dyn_cast<triton::gpu::MemDescIndexOp>(op)) {
    unsigned numUsers = 0;
    Operation *first = nullptr;
    for (auto *user : idx.getOperation()->getUsers()) {
      ++numUsers;
      first = user;
    }
    assert(numUsers <= 1);
    return first;
  }
  return op;
}

Operation *ChannelPost::getSrcOp() {
  for (auto usr : allocOp->getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(user))
      return user;
    if (isa<ttng::AsyncTMACopyGlobalToLocalOp>(user))
      return user;
  }
  return nullptr;
}

static void getAllConsumers(ChannelPost *ch,
                            SmallVector<Operation *> &consumers) {
  for (auto usr : ch->allocOp->getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    if (!isa<ttg::LocalStoreOp>(user) &&
        !isa<ttng::AsyncTMACopyGlobalToLocalOp>(user))
      consumers.push_back(user);
  }
  // assume all consumers are in the same block, with same taskId
  auto taskIds = getAsyncTaskIds(consumers[0]);
  for (unsigned i = 1; i < consumers.size(); ++i) {
    auto taskIds2 = getAsyncTaskIds(consumers[i]);
    assert(taskIds == taskIds2 &&
           consumers[i]->getBlock() == consumers[0]->getBlock());
  }
}

static bool appearsBefore(Operation *A, Operation *B) {
  assert(A->getBlock() == B->getBlock());
  auto block = A->getBlock();
  for (auto &op : block->getOperations()) {
    if (&op == A) {
      // A appears first.
      return true;
    }
    if (&op == B) {
      return false;
    }
  }
  llvm_unreachable("appearsBefore");
}

// A few assumptions, a channel can have multiple consumers, but the consumers
// must be in the same region and the taskIds must be the same. We can have
// a representative consumer in the channel.
Operation *ChannelPost::getDstOp() {
  SmallVector<Operation *> consumers;
  getAllConsumers(this, consumers);
  if (consumers.size() == 1)
    return consumers[0];
  assert(consumers.size() != 0);
  Operation *head = consumers[0];
  for (unsigned i = 1; i < consumers.size(); ++i) {
    if (appearsBefore(consumers[i], head))
      head = consumers[i];
  }
  return head;
}

Operation *ChannelPost::getDstOpLast() {
  SmallVector<Operation *> consumers;
  getAllConsumers(this, consumers);
  if (consumers.size() == 1)
    return consumers[0];
  assert(consumers.size() != 0);
  Operation *tail = consumers[0];
  for (unsigned i = 1; i < consumers.size(); ++i) {
    if (!appearsBefore(consumers[i], tail))
      tail = consumers[i];
  }
  return tail;
}

void ChannelPost::getDstOps(SmallVector<Operation *> &dsts) {
  getAllConsumers(this, dsts);
}

static bool isTmemProducer(Operation *allocOp, Operation *user) {
  if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(user)) {
    if (mmaOp.getD() == allocOp->getResult(0))
      return true;
  }
  if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(user))
    return true;
  return false;
}

static Operation *findTmemStartEnd(ttng::TmemDataChannelPost *ch,
                                   std::string attrName) {
  for (auto usr : ch->allocOp->getResult(0).getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    DenseSet<int> channelIds;
    if (auto attr = user->getAttrOfType<DenseI32ArrayAttr>(attrName)) {
      for (AsyncTaskId asyncTaskId : attr.asArrayRef()) {
        channelIds.insert(asyncTaskId);
      }
      if (channelIds.count(ch->uniqID))
        return user;
    }
  }
  return nullptr;
}

Operation *ttng::TmemDataChannelPost::getSrcOp() {
  if (isOperandD) { // is inout
    // Find tmem.start for this channel ID.
    return findTmemStartEnd(this, "tmem.start");
  }
  for (auto usr : cast<ttng::TMEMAllocOp>(allocOp).getResult().getUsers()) {
    // If there is no subview, user will be the same as usr and we check if opnd
    // D of user is from alloc If there is a subview, alloc -> subview -> user,
    // we check if opnd D of user is from subview.
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    if (isTmemProducer(user == usr ? allocOp : usr, user))
      return user;
  }
  return nullptr;
}

static void getAllConsumers(ttng::TmemDataChannelPost *ch,
                            SmallVector<Operation *> &consumers) {
  auto *allocOp = ch->getAllocOp();
  for (auto usr : cast<ttng::TMEMAllocOp>(allocOp).getResult().getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    if (!isTmemProducer(user == usr ? allocOp : usr, user))
      consumers.push_back(user);
  }
  // assume all consumers are in the same block, with same taskId
  auto taskIds = getAsyncTaskIds(consumers[0]);
  for (unsigned i = 1; i < consumers.size(); ++i) {
    auto taskIds2 = getAsyncTaskIds(consumers[i]);
    assert(taskIds == taskIds2 &&
           consumers[i]->getBlock() == consumers[0]->getBlock());
  }
}

Operation *ttng::TmemDataChannelPost::getDstOp() {
  if (isOperandD) {
    // Find tmem.end for this channel ID.
    return findTmemStartEnd(this, "tmem.end");
  }
  SmallVector<Operation *> consumers;
  getAllConsumers(this, consumers);
  if (consumers.size() == 1)
    return consumers[0];
  assert(consumers.size() != 0);
  return consumers.back();
}

Operation *ttng::TmemDataChannelPost::getDstOpLast() {
  assert(!isOperandD);
  SmallVector<Operation *> consumers;
  getAllConsumers(this, consumers);
  if (consumers.size() == 1)
    return consumers[0];
  assert(consumers.size() != 0);
  Operation *tail = consumers[0];
  for (unsigned i = 1; i < consumers.size(); ++i) {
    if (!appearsBefore(consumers[i], tail))
      tail = consumers[i];
  }
  return tail;
}

void ttng::TmemDataChannelPost::getDstOps(SmallVector<Operation *> &dsts) {
  assert(!isOperandD);
  getAllConsumers(this, dsts);
}

unsigned ChannelPost::getNumBuffers() {
  // get buffer.copy
  if (auto copy = allocOp->getAttrOfType<IntegerAttr>("buffer.copy"))
    return copy.getInt();
  return 1;
}

unsigned ttng::TmemDataChannelPost::getNumBuffers() {
  // get buffer.copy
  if (auto copy = allocOp->getAttrOfType<IntegerAttr>("buffer.copy"))
    return copy.getInt();
  return 1;
}

// Check to see if there is no outer loop that is enclosed under ifOp.
bool immediateEnclosing(scf::IfOp ifOp, Operation *subOp) {
  auto pOp = subOp->getParentOfType<scf::ForOp>();
  if (!pOp)
    return true;
  return !enclosing(ifOp, pOp.getOperation());
}

// Control Ops can be replaced during the pass, but channel srcOp/dstOp should
// be valid.
static bool needAccumCntForReuse(Operation *ctrlOp, ReuseGroup *group) {
  if (group->channels[0]->getNumBuffers() <= 1)
    return false;
  // Goes through each channel in the ResuseGroup, check srcOp and dstOp to
  // see if it is inside ctrlOp.
  for (auto *ch : group->channels) {
    if (auto forOp = dyn_cast<scf::ForOp>(ctrlOp)) {
      if (enclosing(forOp, ch->getSrcOp()))
        return true;
      if (enclosing(forOp, ch->getDstOp()))
        return true;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(ctrlOp)) {
      if (enclosing(ifOp, ch->getSrcOp()))
        return true;
      if (enclosing(ifOp, ch->getDstOp()))
        return true;
    }
  }
  return false;
}

// Return number of AccumCnts for the given ctrlOp. We need one for each nested
// region that contains a channel. Also add accumCnt for each ReuseGroup. We can
// use a simplify pass later on to remove redundant accumCnt.
unsigned getAccumCnts(Operation *ctrlOp,
                      const DenseSet<Operation *> &regionsWithChannels,
                      ReuseConfig *config) {
  unsigned cnt = 0;
  LDBG("getAccumCnts: " << ctrlOp);
  for (auto *op : regionsWithChannels) {
    LDBG("-- getAccumCnts: " << ctrlOp << " regionsWithChannels " << op);
    if (ctrlOp == op) {
      ++cnt;
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(ctrlOp)) {
      if (enclosing(forOp, op))
        ++cnt;
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(ctrlOp)) {
      if (enclosing(ifOp, op))
        ++cnt;
      continue;
    }
    llvm_unreachable("region op other than If/For is not supported");
  }
  if (!config)
    return cnt;
  // Go through each ReuseGroup, and see if we need accumCnt for the given
  // ctrlOp. We need one for a given ReuseGroup when ctrlOp encloses an op from
  // the ReuseGroup.
  for (auto &group : config->groups)
    if (needAccumCntForReuse(ctrlOp, &group))
      ++cnt;
  return cnt;
}

// Figure out the argument index for parentForOp, associated with either
// ctrlOp or with the reuse group. For the latter, we ignore ctrlOp,
// get numbers of arguments for unique channels in parentForOp, then
// decide accumCnts for reuse groups. When reuseGroupIdx is negative,
// we find the argument index associated with unique channels inside
// ctrlOp.
unsigned getAccumArgIdx(scf::ForOp parentForOp, Operation *ctrlOp,
                        const DenseSet<Operation *> &regionsWithChannels,
                        ReuseConfig *config, int reuseGroupIdx) {
  if (reuseGroupIdx >= 0) {
    auto cnts = getAccumCnts(parentForOp, regionsWithChannels, nullptr);
    for (unsigned idx = 0; idx < reuseGroupIdx; ++idx) {
      if (needAccumCntForReuse(parentForOp.getOperation(),
                               config->getGroup(idx)))
        ++cnts;
      return cnts;
    }
  }
  // Walk parentForOp in preorder.
  unsigned preOrderId = 0, ctrlId = 0;
  bool found = false;
  parentForOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    // This will walk parentForOp.
    if (subOp == ctrlOp) {
      ctrlId = preOrderId;
      found = true;
    }
    for (auto *op : regionsWithChannels) {
      if (op == subOp) {
        LDBG("getAccumArgIdx: saw ctrlOp enclosing channel " << subOp);
        ++preOrderId;
      }
    }
  });
  assert(found && "error in getAccumArgIdx");
  LDBG("getAccumArgIdx: " << parentForOp.getOperation() << " " << ctrlOp << " "
                          << ctrlId);
  return ctrlId;
}

// Find channels of reuse group that are inside regionOp. If the channel is
// directly in regionOp, add the channel's DstOp, otherwise add the region Op
// that is directly in regionOp and encloses the channel.
void getReuseChannels(ReuseGroup *group, Operation *regionOp,
                      SmallVector<Operation *> &chList) {
  if (!isa<scf::ForOp>(regionOp) && !isa<scf::IfOp>(regionOp))
    return;
  if (group->channels.size() <= 1 || group->channels[0]->getNumBuffers() <= 1)
    return;
  // Goes through body of regionOp, if the body op is a regionOp, check
  // to see if it contains a channel in the reuse group.
  auto parentForOp = regionOp->getParentOfType<scf::ForOp>();
  if (!parentForOp)
    LDBG("getReuseChannels for group: " << group->channels.size()
                                        << " no outer for");
  else
    LDBG("getReuseChannels for group: " << group->channels.size()
                                        << " with outer for");
  if (auto ifOp = dyn_cast<scf::IfOp>(regionOp)) {
    for (Operation &op : ifOp.thenBlock()->getOperations()) {
      if (isa<scf::ForOp>(&op) || isa<scf::IfOp>(&op)) {
        if (needAccumCntForReuse(&op, group)) {
          chList.push_back(&op);
        }
      } else {
        // Check if op is dstOp of a channel in reuse group. Assume srcOp and
        // dstOp has the same enclosing parentOp.
        for (auto *ch : group->channels) {
          if (&op == ch->getDstOp()) {
            LLVM_DEBUG({
              LDBG("\nchannel with DstOp: ");
              op.dump();
            });
            chList.push_back(&op);
          }
        }
      }
    }
    return;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(regionOp)) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (isa<scf::ForOp>(&op) || isa<scf::IfOp>(&op)) {
        if (needAccumCntForReuse(&op, group)) {
          LDBG("\ninserting ctrlOp in chList");
          chList.push_back(&op);
        }
      } else {
        // Check if op is dstOp of a channel in reuse group. Assume srcOp and
        // dstOp has the same enclosing parentOp.
        for (auto *ch : group->channels) {
          if (&op == ch->getDstOp()) {
            LLVM_DEBUG({
              LDBG("\nchannel with DstOp: ");
              op.dump();
            });
            chList.push_back(&op);
          }
        }
      }
    }
    return;
  }
  assert(false);
}

// regionOp must contains channels in config[idx].
unsigned getReuseAccumArgIdx(Operation *regionOp,
                             const DenseSet<Operation *> &regionsWithChannels,
                             ReuseConfig *config, int reuseGroupIdx) {
  auto cnts = getAccumCnts(regionOp, regionsWithChannels, nullptr);
  unsigned argIdx = 0;
  assert(reuseGroupIdx >= 0 && reuseGroupIdx < config->getGroupSize());
  for (unsigned idx = 0; idx < reuseGroupIdx; ++idx) {
    if (needAccumCntForReuse(regionOp, config->getGroup(reuseGroupIdx)))
      ++argIdx;
  }
  return cnts + argIdx - 1;
}

// Compute and return the buffer index and phase for a given accumulate count.
std::pair<Value, Value> getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                                             Location loc, Value accumCnt,
                                             unsigned numBuffers) {
  Value numBuffersVal =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, numBuffers, 32);
  numBuffersVal = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
      loc, builder.getI64Type(), numBuffersVal);
  // Calculate accumCnt / numBuffers
  // initBufferIdx = accumCnt - accumCnt / numBuffers * numBuffers
  // initPhase = (accumCnt / numBuffers) & 1
  Value bufferIdx = builder.createWithAsyncTaskIds<arith::DivUIOp>(
      loc, accumCnt, numBuffersVal);
  Value initBufferIdx = builder.createWithAsyncTaskIds<arith::SubIOp>(
      loc, accumCnt,
      builder.createWithAsyncTaskIds<arith::MulIOp>(loc, bufferIdx,
                                                    numBuffersVal));
  initBufferIdx = builder.createWithAsyncTaskIds<arith::TruncIOp>(
      loc, builder.getI32Type(), initBufferIdx);

  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
  bufferIdx =
      builder.createWithAsyncTaskIds<arith::AndIOp>(loc, bufferIdx, one);
  Value initPhase = builder.createWithAsyncTaskIds<arith::TruncIOp>(
      loc, builder.getI1Type(), bufferIdx);
  return {initBufferIdx, initPhase};
}

// Get the current accumulation count for the given op within its immediate
// scope.
// ForA (accumForA, accumIfA, accumForB, accumIfB)
//   IfA (accumIfA, accumForB)
//     Channel A --> uses ForA.arg[accumIfA]
//     ForB (accumForB)
//       Channel B --> uses ForB.arg[accumForB]
//   ThenYield ForA.arg[accumIfA] + 1, ForB.res[accumForB]
//   ElseYield ForA.arg[accumIfA], ForA.arg[accumForB]
//   ForC (accumForC, accumIfB)
//     IfB
//       Channel C --> uses ForC.arg[accumIfB]
//     ThenYield ForC.arg[accumIfB] + 1
//     ElseYield ForC.arg[accumIfB]
//   Channel D --> uses ForA.arg[accumForA]
Value getAccumCount(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                    const DenseSet<Operation *> &regionsWithChannels,
                    ReuseConfig *config, int reuseGroupIdx) {
  auto parentForOp = op->getParentOfType<scf::ForOp>();
  auto *pOp = op->getParentOp();
  // Get parentForOp.arg[pOp]
  unsigned tSize = parentForOp.getBody()->getArguments().size();
  unsigned parentTCnts = getAccumCnts(parentForOp, regionsWithChannels, config);
  unsigned accumArgId = getAccumArgIdx(parentForOp, pOp, regionsWithChannels,
                                       config, reuseGroupIdx);
  Value accumCnt =
      parentForOp.getBody()->getArgument(tSize - parentTCnts + accumArgId);

  LDBG("getAccumCount: parentForOp " << parentForOp.getOperation() << " pOp "
                                     << pOp << " " << tSize << " "
                                     << parentTCnts << " " << accumArgId);
  return accumCnt;
}

void getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                          unsigned numBuffers,
                          const DenseSet<Operation *> &regionsWithChannels,
                          Value &bufferIdx, Value &phase, ReuseConfig *config,
                          int reuseGroupIdx) {
  Value accumCnt =
      getAccumCount(builder, op, regionsWithChannels, config, reuseGroupIdx);
  std::tie(bufferIdx, phase) =
      getBufferIdxAndPhase(builder, op->getLoc(), accumCnt, numBuffers);
}

Value getBarrierForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                 Value barrierAlloc, Value bufferIdx) {
  auto context = barrierAlloc.getContext();
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType barrierTy = ttg::MemDescType::get(
      {1}, builder.getI64Type(),
      cast<ttg::MemDescType>(barrierAlloc.getType()).getEncoding(),
      sharedMemorySpace,
      /*mutableMemory=*/true);

  // Create barrierForTMA from barrierAlloc.
  return builder.createWithAsyncTaskIds<ttg::MemDescIndexOp>(
      barrierAlloc.getLoc(), barrierTy, barrierAlloc, bufferIdx);
}

static void setTmemChannelAttr(Operation *op, int channelId,
                               std::string attrName) {
  SmallVector<int> asyncTaskIds;
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(attrName)) {
    for (AsyncTaskId asyncTaskId : attr.asArrayRef()) {
      asyncTaskIds.push_back(asyncTaskId);
    }
  }
  asyncTaskIds.push_back(channelId);
  SmallVector<int> sortedAsyncTaskIds(asyncTaskIds.begin(), asyncTaskIds.end());
  sort(sortedAsyncTaskIds);
  auto i32Ty = IntegerType::get(op->getContext(), 32);
  auto size = static_cast<int64_t>(sortedAsyncTaskIds.size());
  auto vecTy = VectorType::get(size, i32Ty);
  op->setAttr(attrName,
              DenseI32ArrayAttr::get(op->getContext(), sortedAsyncTaskIds));
}

static void handleOperandD(ttng::TMEMAllocOp tmemAllocOp,
                           ttng::TCGen5MMAOp mmaOp,
                           SmallVector<std::unique_ptr<Channel>> &channels) {
  SmallVector<Operation *> consumers;
  SmallVector<Operation *> producers;
  // Go through ops in the body to figure out producer/consumer of the tmem.
  // FIXME: assuming mmaOp is inside a ForOp.
  DenseSet<Operation *> users;
  DenseSet<Operation *> handledUsers;
  for (auto user : tmemAllocOp.getResult().getUsers()) {
    users.insert(user);
  }
  auto forOp = mmaOp->getParentOfType<scf::ForOp>();
  Operation *currentProd = nullptr;
  auto ctx = forOp.getContext();
  SmallVector<int> channelsToBeUpdate;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!users.count(&op))
      continue;
    handledUsers.insert(&op);
    if (auto mmaOpT = dyn_cast<ttng::TCGen5MMAOp>(&op)) {
      if (&op == mmaOp.getOperation()) {
        // This uses and defines D. Will be both producer and consumer.
        assert(currentProd != nullptr);
        // Start a channel from currentProd to op
        auto producerTaskIds = getAsyncTaskIds(currentProd);
        auto consumerIds = getAsyncTaskIds(&op);
        assert(producerTaskIds.size() == 1);
        auto producerTaskId = producerTaskIds.front();
        auto channelID = channels.size();
        channels.push_back(std::make_unique<ttng::TmemDataChannelPost>(
            producerTaskId, consumerIds, tmemAllocOp.getOperation(),
            true /*isOperandD*/, channels.size()));
        // Mark producer and consumer.
        setTmemChannelAttr(currentProd, channelID, "tmem.start");
        setTmemChannelAttr(&op, channelID, "tmem.end");
        currentProd = &op;
      } else {
        assert(mmaOpT.getD() != tmemAllocOp.getResult());
        // This uses tmem. mark as tmem.end = channel_id
        assert(currentProd != nullptr);
        // Start a channel from currentProd to op
        auto producerTaskIds = getAsyncTaskIds(currentProd);
        assert(producerTaskIds.size() == 1);
        auto producerTaskId = producerTaskIds.front();
        auto channelID = channels.size();
        auto consumerIds = getAsyncTaskIds(&op);
        channels.push_back(std::make_unique<ttng::TmemDataChannelPost>(
            producerTaskId, consumerIds, tmemAllocOp.getOperation(),
            true /*isOperandD*/, channels.size()));
        // Mark producer and consumer.
        setTmemChannelAttr(currentProd, channelID, "tmem.start");
        setTmemChannelAttr(&op, channelID, "tmem.end");
      }
    } else if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(&op)) {
      currentProd = &op; // mark as tmem.start = channel_id
    } else if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(&op)) {
      if (currentProd) {
        // Start a channel from currentProd to op
        auto producerTaskIds = getAsyncTaskIds(currentProd);
        assert(producerTaskIds.size() == 1);
        auto producerTaskId = producerTaskIds.front();
        auto channelID = channels.size();
        auto consumerIds = getAsyncTaskIds(&op);
        channels.push_back(std::make_unique<ttng::TmemDataChannelPost>(
            producerTaskId, consumerIds, tmemAllocOp.getOperation(),
            true /*isOperandD*/, channels.size()));
        // Mark producer and consumer.
        setTmemChannelAttr(currentProd, channelID, "tmem.start");
        setTmemChannelAttr(&op, channelID, "tmem.end");
      } else {
        channelsToBeUpdate.push_back(channels.size());
        auto channelID = channels.size();
        auto consumerIds = getAsyncTaskIds(&op);
        channels.push_back(std::make_unique<ttng::TmemDataChannelPost>(
            -1, consumerIds, tmemAllocOp.getOperation(), true /*isOperandD*/,
            channels.size()));
        // Mark producer and consumer.
        setTmemChannelAttr(&op, channelID, "tmem.end");
      }
    } else {
      assert(0);
    }
  }
  // Update channel's producer here.
  for (auto idx : channelsToBeUpdate) {
    assert(currentProd); // assuming ForOp runs at least one iteration.
    channels[idx]->relation.first = getAsyncTaskIds(currentProd).front();
    setTmemChannelAttr(currentProd, channels[idx]->uniqID, "tmem.start");
  }
  // For consumers outside of ForOp.
  for (auto *user : users) {
    if (handledUsers.count(user))
      continue;
    // only handle tmem_load. FIXME: check if it is after the ForOp
    if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(user)) {
      assert(currentProd);
      // Start a channel from currentProd to user
      auto producerTaskIds = getAsyncTaskIds(currentProd);
      assert(producerTaskIds.size() == 1);
      auto producerTaskId = producerTaskIds.front();
      auto channelID = channels.size();
      auto consumerIds = getAsyncTaskIds(user);
      channels.push_back(std::make_unique<ttng::TmemDataChannelPost>(
          producerTaskId, consumerIds, tmemAllocOp.getOperation(),
          true /*isOperandD*/, channels.size()));
      // Mark producer and consumer.
      setTmemChannelAttr(currentProd, channelID, "tmem.start");
      setTmemChannelAttr(user, channelID, "tmem.end");
    }
  }
}

static void createChannelPost(Operation *allocOp, mlir::DominanceInfo &dom,
                              SmallVector<std::unique_ptr<Channel>> &channels) {
  // source can be local_store, consumer can be gen5, ttg.memdesc_trans,
  // local_load Can be produced by tmem_store or gen5, consumed by tmem_load or
  // gen5
  Operation *producerOp = nullptr;
  SmallVector<Operation *> consumers;
  SmallVector<Operation *> producers;
  auto isConstFalse = [](Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<BoolAttr>(constOp.getValueAttr())) {
        return !attr.getValue();
      }
    }
    return false;
  };
  if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(allocOp)) {
    bool isOperandD = false;
    ttng::TCGen5MMAOp mmaOp;
    // Go through users of the first result (i.e exclude token).
    for (auto user : tmemAllocOp.getResult().getUsers()) {
      if (auto mmaOpT = dyn_cast<ttng::TCGen5MMAOp>(user)) {
        if (mmaOpT.getD() == allocOp->getResult(0)) {
          if (!isConstFalse(mmaOpT.useAccumulator())) {
            mmaOp = mmaOpT;
            isOperandD = true;
          } else
            producers.push_back(user);
        } else // other operands are consumers
          consumers.push_back(user);
      } else if (isa<ttng::TMEMStoreOp>(user)) {
        producers.push_back(user);
      } else if (isa<ttng::TMEMLoadOp>(user)) {
        consumers.push_back(user);
      } else
        assert(0);
    }
    if (isOperandD) {
      // Create a list of virtual channels for this case. Each virtual channel
      // has a single producer.
      handleOperandD(tmemAllocOp, mmaOp, channels);
      return;
    }

    producerOp = producers[0];
    if (producers.size() > 1) {
      assert(consumers.size() == 1);
      producerOp = nullptr;
      for (auto *prod : producers) {
        // Ignore the one that is not in the same block as consumer.
        if (prod->getBlock() != consumers[0]->getBlock())
          continue;
        assert(producerOp == nullptr);
        producerOp = prod;
      }
    }
  } else {
    assert(isa<ttg::LocalAllocOp>(allocOp));
    for (auto user : allocOp->getUsers()) {
      if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(user)) {
        // Alloc associated with operand D can have multiple producers.
        assert(mmaOp.getD() != allocOp->getResult(0));
        consumers.push_back(user);
      } else if (isa<ttg::LocalStoreOp>(user)) {
        assert(producerOp == nullptr);
        producerOp = user;
      } else
        consumers.push_back(user);
    }
  }
  auto producerTaskIds = getAsyncTaskIds(producerOp);
  assert(producerTaskIds.size() == 1);
  auto producerTaskId = producerTaskIds.front();
  // Either a single consumer op with multiple taskIds, or multiple consumer ops
  // with the same taskId.
  auto consumerTaskIds = getAsyncTaskIds(consumers[0]);
  if (consumerTaskIds.size() > 1)
    assert(consumers.size() == 1);
  // Remove producer task id from consumerTaskIds.
  auto iter = std::remove(consumerTaskIds.begin(), consumerTaskIds.end(),
                          producerTaskId);
  consumerTaskIds.erase(iter, consumerTaskIds.end());

  if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(allocOp))
    channels.push_back(std::make_unique<ttng::TmemDataChannelPost>(
        producerTaskIds.front(), consumerTaskIds, allocOp, false,
        channels.size()));
  else
    channels.push_back(std::make_unique<ChannelPost>(
        producerTaskIds.front(), consumerTaskIds, allocOp, channels.size()));
}

void collectPostChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                         triton::FuncOp &funcOp) {
  mlir::DominanceInfo dom(funcOp);
  funcOp.walk([&](Operation *op) {
    // FIXME: It is possible that a local_alloc can start a channel, when a
    // gemm's operand is in smem and comes from local_alloc.
    // All buffers have been allocated, a channel will be created based on
    // the alloc.
    if (dyn_cast<ttng::TMEMAllocOp>(op)) {
      createChannelPost(op, dom, channels);
    } else if (dyn_cast<ttg::LocalAllocOp>(op)) {
      createChannelPost(op, dom, channels);
    }
  });
}

// Find the operation that is along producer's parent chain, and its parent
// is the same op as producer's parent. Here p is producer, and c is consumer.
Operation *getSameLevelOp(Operation *p, Operation *c) {
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

// When the consumer is a local_alloc loading from shared memory to registers,
// look ahead for the actual consumers, usually dot ops, that can directly
// use shared memory. The local_alloc will be removed later.
SmallVector<Operation *> getActualConsumers(Operation *consumerOp) {
  // TransOp is not a real consumer. It caculates the shared memory
  // address for the real consumer. Continue to find its transitive users
  // recursively. Return all transitive users;
  auto goThroughTrans = [&](Operation *user) -> DenseSet<Operation *> {
    DenseSet<Operation *> users;
    DenseSet<Operation *> visited;
    SmallVector<Operation *> transUsers;
    transUsers.push_back(user);
    while (!transUsers.empty()) {
      auto transUser = transUsers.pop_back_val();
      visited.insert(transUser);
      if (isa<tt::TransOp, ttg::MemDescTransOp>(transUser)) {
        for (auto transitiveUser : transUser->getUsers()) {
          if (!visited.count(transitiveUser))
            transUsers.push_back(transitiveUser);
        }
      } else {
        users.insert(transUser);
      }
    }
    return users;
  };
  if (isa<ttg::MemDescTransOp>(consumerOp)) {
    auto users = goThroughTrans(consumerOp);
    return SmallVector<Operation *>(users.begin(), users.end());
  }
  if (isa<ttg::LocalAllocOp>(consumerOp)) {
    DenseSet<Operation *> users;
    for (auto user : consumerOp->getUsers()) {
      if (isa<tt::TransOp, ttg::MemDescTransOp>(user)) {
        auto transUsers = goThroughTrans(user);
        for (auto *tUsr : transUsers)
          users.insert(tUsr);
      } else {
        users.insert(user);
      }
    }

    return SmallVector<Operation *>(users.begin(), users.end());
  }
  return {consumerOp};
}

} // namespace mlir
