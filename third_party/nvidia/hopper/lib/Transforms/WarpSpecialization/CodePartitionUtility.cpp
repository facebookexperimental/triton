#include "CodePartitionUtility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-utility"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Helper function to check if a channel is needed between producer and
// consumers. Returns false if the producer task ID matches all consumer task
// IDs (no cross-warp synchronization needed).
static bool needsChannel(int producer, const SmallVector<int> &consumers) {
  return !llvm::all_of(
      consumers, [producer](int consumerId) { return consumerId == producer; });
}

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
                            SmallVector<Operation *> &consumers,
                            bool sameBlock = true) {
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
    assert(taskIds == taskIds2);
    if (sameBlock)
      assert(consumers[i]->getBlock() == consumers[0]->getBlock());
  }
}

// Return an op that encloses both a and b
static Operation *getCommonScope(Operation *a, Operation *b) {
  DenseSet<Operation *> parentScopes;
  Operation *op = a;
  while (!isa<triton::FuncOp>(op)) {
    parentScopes.insert(op);
    op = op->getParentOp();
  }
  // Worst case the function should enclose both A and B.
  parentScopes.insert(op);
  op = b;
  while (!isa<triton::FuncOp>(op)) {
    if (parentScopes.count(op))
      return op;
    op = op->getParentOp();
  }
  return parentScopes.count(op) ? op : nullptr;
}

// Return the lifted "op" that is directly under scope.
static Operation *getLiftedOp(Operation *op, Operation *scope) {
  if (op == scope)
    return op;
  Operation *liftedUser = nullptr;
  while (!isa<triton::FuncOp>(op)) {
    if (op->getParentOp() == scope) {
      return op;
    }
    op = op->getParentOp();
  }
  return nullptr;
}

bool appearsBefore(Operation *A, Operation *B) {
  // A and B can be from different blocks.
  if (A->getBlock() != B->getBlock()) {
    auto *outScope = getCommonScope(A, B);
    return appearsBefore(getLiftedOp(A, outScope), getLiftedOp(B, outScope));
  }
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
  getAllConsumers(this, consumers, false);
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
  getAllConsumers(this, dsts, false);
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
    }
    return cnts;
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
    if (needAccumCntForReuse(regionOp, config->getGroup(idx)))
      ++argIdx;
  }
  assert(needAccumCntForReuse(regionOp, config->getGroup(reuseGroupIdx)));
  return cnts + argIdx;
}

// Compute and return the buffer index and phase for a given accumulate count.
std::pair<Value, Value> getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                                             Location loc, Value accumCnt,
                                             unsigned numBuffers) {
  // ensure type compatibility
  Value numBuffersVal;
  if (accumCnt.getType().isIndex()) {
    // accumCnt is index type, create an index constant
    numBuffersVal =
        builder.createWithAsyncTaskIds<arith::ConstantIndexOp>(loc, numBuffers);
  } else {
    // accumCnt is integer type, create a matching integer constant
    auto intType = llvm::cast<IntegerType>(accumCnt.getType());
    numBuffersVal = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        loc, numBuffers, intType.getWidth());
  }
  // Calculate accumCnt / numBuffers
  // initBufferIdx = accumCnt - accumCnt / numBuffers * numBuffers
  // initPhase = (accumCnt / numBuffers) & 1
  Value bufferIdx = builder.createWithAsyncTaskIds<arith::DivUIOp>(
      loc, accumCnt, numBuffersVal);
  auto mulOp = builder.createWithAsyncTaskIds<arith::MulIOp>(loc, bufferIdx,
                                                             numBuffersVal);
  Value initBufferIdx =
      builder.createWithAsyncTaskIds<arith::SubIOp>(loc, accumCnt, mulOp);

  // Convert to i32 for buffer indexing
  if (initBufferIdx.getType().isIndex()) {
    // For index type, use index_cast to convert to i32
    initBufferIdx = builder.createWithAsyncTaskIds<arith::IndexCastOp>(
        loc, builder.getI32Type(), initBufferIdx);
  } else {
    // For integer types, truncate to i32
    initBufferIdx = builder.createWithAsyncTaskIds<arith::TruncIOp>(
        loc, builder.getI32Type(), initBufferIdx);
  }

  // ensure type compatibility
  Value one;
  if (bufferIdx.getType().isIndex()) {
    // For index type, create a constant index
    one = builder.createWithAsyncTaskIds<arith::ConstantIndexOp>(loc, 1);
  } else if (auto intType = llvm::dyn_cast<IntegerType>(bufferIdx.getType())) {
    // For integer types, create a constant with matching bit width
    one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        loc, 1, intType.getWidth());
  } else {
    llvm_unreachable("bufferIdx must be either index or integer type");
  }
  bufferIdx =
      builder.createWithAsyncTaskIds<arith::AndIOp>(loc, bufferIdx, one);

  // Convert to i1 for phase
  Value initPhase;
  if (bufferIdx.getType().isIndex()) {
    // For index type, first cast to i32, then truncate to i1
    Value bufferIdxI32 = builder.createWithAsyncTaskIds<arith::IndexCastOp>(
        loc, builder.getI32Type(), bufferIdx);
    initPhase = builder.createWithAsyncTaskIds<arith::TruncIOp>(
        loc, builder.getI1Type(), bufferIdxI32);
  } else {
    // For integer types, truncate to i1
    initPhase = builder.createWithAsyncTaskIds<arith::TruncIOp>(
        loc, builder.getI1Type(), bufferIdx);
  }
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

  // Handle operations outside loops (e.g., epilogue operations).
  // These operations don't participate in buffer cycling, so return constant 0.
  if (!parentForOp) {
    LDBG("getAccumCount: operation outside loop, returning constant 0");
    return builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
  }

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

int channelInReuseGroup(Channel *channel, ReuseConfig *config,
                        bool reuseBarrier) {
  for (unsigned idx = 0; idx < config->getGroupSize(); idx++) {
    // Reuse the same barriers when numBuffers > 1.
    if (config->getGroup(idx)->channels[0]->getNumBuffers() <= 1 &&
        reuseBarrier)
      continue;
    for (auto *ch : config->getGroup(idx)->channels) {
      if (channel == ch)
        return idx;
    }
  }
  return -1;
}

void getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                          unsigned numBuffers,
                          const DenseSet<Operation *> &regionsWithChannels,
                          Value &bufferIdx, Value &phase, ReuseConfig *config,
                          int reuseGroupIdx, Channel *ch) {
  Value accumCnt =
      getAccumCount(builder, op, regionsWithChannels, config, reuseGroupIdx);
  if (reuseGroupIdx < 0) {
    std::tie(bufferIdx, phase) =
        getBufferIdxAndPhase(builder, op->getLoc(), accumCnt, numBuffers);
    return;
  }
  // op is a user of the channel. accumCnt is the corresponding argument of the
  // parentForOp.
  // Go through chList in the parentForOp, assume ch is directly in parentForOp.
  // FIXME: handle the case where ch is inside in IfOp.
  SmallVector<Operation *> chList;
  auto parentForOp = op->getParentOfType<scf::ForOp>();
  getReuseChannels(config->getGroup(reuseGroupIdx), parentForOp.getOperation(),
                   chList);
  assert(chList.size() >= 1);
  int vecIdx = 0, theIdx = -1;
  for (auto *tCh : chList) {
    if (tCh == ch->getDstOp()) {
      theIdx = vecIdx;
      break;
    }
    ++vecIdx;
  }
  assert(theIdx >= 0);
  if (theIdx == 0) {
    std::tie(bufferIdx, phase) =
        getBufferIdxAndPhase(builder, op->getLoc(), accumCnt, numBuffers);
    return;
  }
  // Increment accumCnt if there are multiple channels in the reuseGroup in this
  // region.
  // Create idxVal with the same type as accumCnt to ensure type compatibility
  Value idxVal;
  if (accumCnt.getType().isIndex()) {
    idxVal = builder.createWithAsyncTaskIds<arith::ConstantIndexOp>(
        op->getLoc(), theIdx);
  } else {
    auto intType = llvm::cast<IntegerType>(accumCnt.getType());
    idxVal = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        op->getLoc(), theIdx, intType.getWidth());
  }
  Value addRes = builder.createWithAsyncTaskIds<arith::AddIOp>(
      op->getLoc(), accumCnt, idxVal);

  std::tie(bufferIdx, phase) =
      getBufferIdxAndPhase(builder, op->getLoc(), addRes, numBuffers);
}

Value getBarrierForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                 Value barrierAlloc, Value bufferIdx) {
  ttg::MemDescType allocType = cast<ttg::MemDescType>(barrierAlloc.getType());
  ttg::MemDescType barrierTy =
      ttg::MemDescType::get({1}, builder.getI64Type(), allocType.getEncoding(),
                            allocType.getMemorySpace(),
                            /*mutableMemory=*/true);

  // Create barrierForTMA from barrierAlloc.
  auto output = builder.createWithAsyncTaskIds<ttg::MemDescIndexOp>(
      barrierAlloc.getLoc(), barrierTy, barrierAlloc, bufferIdx);
  return output;
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

// Helper function to create channels from multiple producers to a single
// consumer. Creates one channel per producer in the currentProds vector.
// @param currentProds Vector of producer operations
// @param producerTaskId Task ID of the producers (must all be the same)
// @param consumerIds Consumer task IDs
// @param allocOp The TMEM allocation operation
// @param consumerOp The consumer operation
// @param channels Output vector to add created channels to
static void
createChannelsForProducers(SmallVector<Operation *> &currentProds,
                           int producerTaskId, SmallVector<int> &consumerIds,
                           Operation *allocOp, Operation *consumerOp,
                           SmallVector<std::unique_ptr<Channel>> &channels) {
  for (auto *prod : currentProds) {
    auto channelID = channels.size();
    channels.push_back(std::make_unique<ttng::TmemDataChannelPost>(
        producerTaskId, consumerIds, allocOp, true /*isOperandD*/, true,
        channelID));
    setTmemChannelAttr(prod, channelID, "tmem.start");
    setTmemChannelAttr(consumerOp, channelID, "tmem.end");
  }
}

/// Dump information about a single channel for debugging.
static void dumpChannel(Channel *ch, llvm::raw_ostream &os) {
  os << "  Channel ID: " << ch->uniqID << "\n";
  os << "    Kind: " << to_string(ch->channelKind) << "\n";
  os << "    Producer Task ID: " << ch->relation.first << "\n";
  os << "    Consumer Task IDs: [";
  for (size_t i = 0; i < ch->relation.second.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << ch->relation.second[i];
  }
  os << "]\n";
  os << "    NumBuffers: " << ch->getNumBuffers() << "\n";
  if (auto *allocOp = ch->getAllocOp()) {
    os << "    AllocOp: ";
    allocOp->print(os, OpPrintingFlags().skipRegions());
    os << "\n";
  }
  if (auto *srcOp = ch->getSrcOp()) {
    os << "    SrcOp: ";
    srcOp->print(os, OpPrintingFlags().skipRegions());
    os << "\n";
  }
  if (auto *dstOp = ch->getDstOp()) {
    os << "    DstOp: ";
    dstOp->print(os, OpPrintingFlags().skipRegions());
    os << "\n";
  }
  // For TmemDataChannelPost, dump additional info
  if (ch->channelKind == DataChannelKind::TMEMPost) {
    auto *tmemCh = static_cast<ttng::TmemDataChannelPost *>(ch);
    os << "    isOperandD: " << (tmemCh->isOperandD ? "true" : "false") << "\n";
    os << "    isOperandDNoAcc: "
       << (tmemCh->isOperandDNoAcc ? "true" : "false") << "\n";
  }
}

/// Dump all channels associated with an OperandD (same allocOp).
static void
dumpChannelsForOperandD(ttng::TMEMAllocOp tmemAllocOp,
                        SmallVector<std::unique_ptr<Channel>> &channels,
                        llvm::raw_ostream &os) {
  os << "\n=== Channels for OperandD ===\n";
  os << "TMEMAllocOp: ";
  tmemAllocOp.getOperation()->print(os, OpPrintingFlags().skipRegions());
  os << "\n";
  os << "Number of channels: ";
  size_t count = 0;
  for (auto &ch : channels) {
    if (ch->getAllocOp() == tmemAllocOp.getOperation()) {
      ++count;
    }
  }
  os << count << "\n";
  for (auto &ch : channels) {
    if (ch->getAllocOp() == tmemAllocOp.getOperation()) {
      dumpChannel(ch.get(), os);
    }
  }
  os << "=== End Channels for OperandD ===\n\n";
}

/// Dump all channels in the channel collection for debugging.
static void dumpAllChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                            llvm::raw_ostream &os) {
  os << "\n=== All Channels ===\n";
  os << "Total channel count: " << channels.size() << "\n\n";
  for (auto &ch : channels) {
    dumpChannel(ch.get(), os);
  }
  os << "=== End All Channels ===\n\n";
}

/// Get a short name for an operation for display in the graph.
static std::string getOpShortName(Operation *op) {
  if (!op)
    return "null";
  std::string name = op->getName().getStringRef().str();
  // Remove dialect prefix for brevity
  size_t dotPos = name.find('.');
  if (dotPos != std::string::npos && dotPos + 1 < name.size()) {
    name = name.substr(dotPos + 1);
  }
  return name;
}

/// Get operation_id attribute value, or -1 if not present.
static int getOperationId(Operation *op) {
  if (!op)
    return -1;
  if (auto opIdAttr = op->getAttrOfType<IntegerAttr>("operation_id")) {
    return opIdAttr.getInt();
  }
  return -1;
}

/// Get buffer.id attribute value, or -1 if not present.
static int getBufferId(Operation *op) {
  if (!op)
    return -1;
  if (auto bufIdAttr = op->getAttrOfType<IntegerAttr>("buffer.id")) {
    return bufIdAttr.getInt();
  }
  return -1;
}

/// Get named location string from an operation, or empty string if not present.
/// Supports NameLoc, FusedLoc, FileLineColLoc, and CallSiteLoc.
static std::string getNamedLoc(Operation *op) {
  if (!op)
    return "";
  Location loc = op->getLoc();

  // Try to get NameLoc (e.g., loc("myName"))
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    return nameLoc.getName().str();
  }
  // Try FusedLoc which may contain a NameLoc or FileLineColLoc
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (Location subLoc : fusedLoc.getLocations()) {
      if (auto nameLoc = dyn_cast<NameLoc>(subLoc)) {
        return nameLoc.getName().str();
      }
    }
    // If no NameLoc found, try to get FileLineColLoc
    for (Location subLoc : fusedLoc.getLocations()) {
      if (auto fileLoc = dyn_cast<FileLineColLoc>(subLoc)) {
        std::string filename = fileLoc.getFilename().str();
        // Extract just the filename without path
        size_t lastSlash = filename.rfind('/');
        if (lastSlash != std::string::npos) {
          filename = filename.substr(lastSlash + 1);
        }
        return filename + ":" + std::to_string(fileLoc.getLine());
      }
    }
  }
  // Try FileLineColLoc directly (e.g., "file.py":42:0)
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    std::string filename = fileLoc.getFilename().str();
    // Extract just the filename without path
    size_t lastSlash = filename.rfind('/');
    if (lastSlash != std::string::npos) {
      filename = filename.substr(lastSlash + 1);
    }
    return filename + ":" + std::to_string(fileLoc.getLine());
  }
  // Try CallSiteLoc - extract location from callee
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    // Get the callee location (where the function is defined)
    Location calleeLoc = callSiteLoc.getCallee();
    if (auto fileLoc = dyn_cast<FileLineColLoc>(calleeLoc)) {
      std::string filename = fileLoc.getFilename().str();
      size_t lastSlash = filename.rfind('/');
      if (lastSlash != std::string::npos) {
        filename = filename.substr(lastSlash + 1);
      }
      return filename + ":" + std::to_string(fileLoc.getLine());
    }
    if (auto nameLoc = dyn_cast<NameLoc>(calleeLoc)) {
      return nameLoc.getName().str();
    }
    // Try FusedLoc within callee
    if (auto fusedLoc = dyn_cast<FusedLoc>(calleeLoc)) {
      for (Location subLoc : fusedLoc.getLocations()) {
        if (auto nameLoc = dyn_cast<NameLoc>(subLoc)) {
          return nameLoc.getName().str();
        }
      }
      for (Location subLoc : fusedLoc.getLocations()) {
        if (auto fileLoc = dyn_cast<FileLineColLoc>(subLoc)) {
          std::string filename = fileLoc.getFilename().str();
          size_t lastSlash = filename.rfind('/');
          if (lastSlash != std::string::npos) {
            filename = filename.substr(lastSlash + 1);
          }
          return filename + ":" + std::to_string(fileLoc.getLine());
        }
      }
    }
  }
  return "";
}

/// Get a unique node ID for an operation.
static std::string getNodeId(Operation *op) {
  if (!op)
    return "null";
  std::stringstream ss;
  // Use operation_id if available for more readable graph
  int opId = getOperationId(op);
  if (opId >= 0) {
    ss << "op_" << opId;
  } else {
    // Use a hash of the pointer for consistent IDs
    ss << "op_" << (reinterpret_cast<uintptr_t>(op) % 100000);
  }
  return ss.str();
}

/// Check if an operation is a key operation (GEMM, load/store, or tensor
/// computation).
static bool isKeyOp(Operation *op) {
  // GEMM operations
  if (isa<ttng::TCGen5MMAOp>(op))
    return true;

  // Load operations
  if (isa<tt::DescriptorLoadOp, tt::LoadOp, ttng::TMEMLoadOp, ttg::LocalLoadOp>(
          op))
    return true;

  // Store operations
  if (isa<tt::DescriptorStoreOp, tt::StoreOp, ttng::TMEMStoreOp,
          ttg::LocalStoreOp, tt::DescriptorReduceOp>(op))
    return true;

  // Tensor computation operations (arithmetic and math on tensors)
  if (op->getNumResults() > 0) {
    if (auto resultType = op->getResult(0).getType()) {
      if (isa<RankedTensorType>(resultType)) {
        if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                arith::MaxNumFOp, arith::MinNumFOp, arith::TruncFOp,
                math::ExpOp, math::Exp2Op, math::LogOp, math::Log2Op,
                math::SqrtOp, math::RsqrtOp, math::TanhOp>(op))
          return true;
      }
    }
  }

  return false;
}

/// Get NamedLoc from a Value's defining operation, if available.
static std::string getValueName(Value val) {
  if (!val)
    return "";
  if (auto *defOp = val.getDefiningOp()) {
    std::string locName = getNamedLoc(defOp);
    if (!locName.empty())
      return locName;
  }
  // For block arguments, try to get a meaningful name
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    return "arg" + std::to_string(blockArg.getArgNumber());
  }
  return "";
}

/// Get a simple shape string from a type (e.g., "128x128xf32").
static std::string getShapeStr(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    std::string result;
    llvm::raw_string_ostream ss(result);
    for (int64_t dim : tensorType.getShape()) {
      ss << dim << "x";
    }
    ss << tensorType.getElementType();
    return result;
  }
  if (auto memDescType = dyn_cast<ttg::MemDescType>(type)) {
    std::string result;
    llvm::raw_string_ostream ss(result);
    for (int64_t dim : memDescType.getShape()) {
      ss << dim << "x";
    }
    ss << memDescType.getElementType();
    return result;
  }
  // Fallback: just print the type without layout details
  std::string result;
  llvm::raw_string_ostream ss(result);
  ss << type;
  return result;
}

/// Get a simplified operation description focusing on shapes and variable
/// names.
static std::string getKeyOpDescription(Operation *op) {
  std::string result;
  llvm::raw_string_ostream ss(result);

  std::string opName = getOpShortName(op);

  // Helper lambda to format input variable with name if available
  auto formatInput = [](Value val) -> std::string {
    std::string name = getValueName(val);
    if (!name.empty())
      return name;
    return getShapeStr(val.getType());
  };

  // Helper lambda to format output variable with shape
  auto formatOutput = [](Value val) -> std::string {
    return getShapeStr(val.getType());
  };

  // For GEMM, show operand names/shapes: A @ B -> D
  if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(op)) {
    ss << opName << " " << formatInput(mmaOp.getA()) << " @ "
       << formatInput(mmaOp.getB()) << " -> " << formatInput(mmaOp.getD());
    return result;
  }

  // For loads, show source and result
  if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(op)) {
    ss << opName << " " << formatInput(loadOp.getDesc()) << " -> "
       << formatOutput(loadOp.getResult());
    return result;
  }
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    ss << opName << " " << formatInput(loadOp.getPtr()) << " -> "
       << formatOutput(loadOp.getResult());
    return result;
  }
  if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
    ss << opName << " " << formatInput(loadOp.getSrc()) << " -> "
       << formatOutput(loadOp.getResult());
    return result;
  }
  if (auto loadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
    ss << opName << " " << formatInput(loadOp.getSrc()) << " -> "
       << formatOutput(loadOp.getResult());
    return result;
  }

  // For stores, show source and destination
  if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(op)) {
    ss << opName << " " << formatInput(storeOp.getSrc()) << " -> "
       << formatInput(storeOp.getDesc());
    return result;
  }
  if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
    ss << opName << " " << formatInput(storeOp.getValue()) << " -> "
       << formatInput(storeOp.getPtr());
    return result;
  }
  if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
    ss << opName << " " << formatInput(storeOp.getSrc()) << " -> "
       << formatInput(storeOp.getDst());
    return result;
  }
  if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
    ss << opName << " " << formatInput(storeOp.getSrc()) << " -> "
       << formatInput(storeOp.getDst());
    return result;
  }
  if (auto reduceOp = dyn_cast<tt::DescriptorReduceOp>(op)) {
    ss << opName << " " << formatInput(reduceOp.getSrc()) << " -> "
       << formatInput(reduceOp.getDesc());
    return result;
  }

  // For arithmetic/math ops, show inputs and output
  if (op->getNumResults() > 0) {
    ss << opName << " ";
    bool first = true;
    for (Value operand : op->getOperands()) {
      if (!first)
        ss << ", ";
      ss << formatInput(operand);
      first = false;
    }
    ss << " -> " << formatOutput(op->getResult(0));
    return result;
  }

  ss << opName;
  return result;
}

/// Check if an operation or its nested regions contain any key operations.
static bool containsKeyOps(Operation *op) {
  if (isKeyOp(op))
    return true;

  // Check nested regions
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &innerOp : block) {
        if (containsKeyOps(&innerOp))
          return true;
      }
    }
  }
  return false;
}

/// Simplify a name that may be in filename:linenumber format.
/// If the name matches "filename.py:123" pattern, return just "L123"
static std::string simplifyName(const std::string &name) {
  if (name.empty())
    return name;

  // Check if name contains a colon (file:line format)
  size_t colonPos = name.rfind(':');
  if (colonPos != std::string::npos && colonPos + 1 < name.size()) {
    // Check if what follows the colon is a number
    std::string afterColon = name.substr(colonPos + 1);
    bool isNumber =
        !afterColon.empty() &&
        std::all_of(afterColon.begin(), afterColon.end(), ::isdigit);
    if (isNumber) {
      return "L" + afterColon;
    }
  }
  return name;
}

/// Get the loop depth of an operation (number of enclosing scf.for loops)
static int getLoopDepth(Operation *op) {
  int depth = 0;
  Operation *parent = op->getParentOp();
  while (parent) {
    if (isa<scf::ForOp>(parent)) {
      depth++;
    }
    parent = parent->getParentOp();
  }
  return depth;
}

/// Get the name of a value for display purposes.
/// Returns named location if available, otherwise a placeholder.
static std::string getValueDisplayName(Value val) {
  if (Operation *defOp = val.getDefiningOp()) {
    std::string name = getNamedLoc(defOp);
    if (!name.empty())
      return simplifyName(name);
  }
  return "?";
}

/// Generate a compact label for a key operation.
/// Format:
/// Line 1: [opId] output = operator(inputs)
/// Line 2: shape, Ln (loop depth)
static std::string getKeyOpLabel(Operation *op) {
  std::string label;

  // Add operation ID
  int opId = getOperationId(op);
  if (opId >= 0) {
    label = "[" + std::to_string(opId) + "] ";
  }

  std::string opName = getOpShortName(op);
  std::string locName = getNamedLoc(op);
  std::string outputName = locName.empty() ? "?" : simplifyName(locName);

  // Helper to get tensor input names (skip non-tensor operands)
  auto getTensorInputs = [](Operation *op) -> std::string {
    std::string inputs;
    bool first = true;
    for (Value operand : op->getOperands()) {
      Type type = operand.getType();
      // Check if it's a tensor-like type
      if (isa<RankedTensorType>(type) || isa<triton::gpu::MemDescType>(type) ||
          isa<triton::PointerType>(type)) {
        if (!first)
          inputs += ", ";
        inputs += getValueDisplayName(operand);
        first = false;
      }
    }
    return inputs;
  };

  // Helper to get only the source tensor name for store operations
  auto getStoreSrcName = [](Operation *op) -> std::string {
    if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(op)) {
      return getValueDisplayName(storeOp.getSrc());
    }
    if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
      return getValueDisplayName(storeOp.getValue());
    }
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      return getValueDisplayName(storeOp.getSrc());
    }
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
      return getValueDisplayName(storeOp.getSrc());
    }
    if (auto reduceOp = dyn_cast<tt::DescriptorReduceOp>(op)) {
      return getValueDisplayName(reduceOp.getSrc());
    }
    return "?";
  };

  // Helper to get output shape (excluding !ttg.async.token)
  auto getOutputShape = [](Operation *op) -> std::string {
    if (op->getNumResults() > 0) {
      std::string shape = getShapeStr(op->getResult(0).getType());
      // Remove !ttg.async.token
      if (shape.find("!ttg.async.token") != std::string::npos) {
        return "";
      }
      return shape;
    }
    // For store ops, get shape from the stored value
    if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(op)) {
      return getShapeStr(storeOp.getSrc().getType());
    }
    if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
      return getShapeStr(storeOp.getValue().getType());
    }
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      return getShapeStr(storeOp.getSrc().getType());
    }
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
      return getShapeStr(storeOp.getSrc().getType());
    }
    return "";
  };

  // Build the label based on operation type
  if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(op)) {
    // GEMM: D = mma(A, B)
    std::string aName = getValueDisplayName(mmaOp.getA());
    std::string bName = getValueDisplayName(mmaOp.getB());
    label += outputName + " = " + opName + "(" + aName + ", " + bName + ")";
  } else if (isa<tt::DescriptorLoadOp, tt::LoadOp, ttng::TMEMLoadOp,
                 ttg::LocalLoadOp>(op)) {
    // Load: out = load(src)
    std::string inputs = getTensorInputs(op);
    label += outputName + " = " + opName + "(" + inputs + ")";
  } else if (isa<tt::DescriptorStoreOp, tt::StoreOp, ttng::TMEMStoreOp,
                 ttg::LocalStoreOp, tt::DescriptorReduceOp>(op)) {
    // Store: store(src) - only show the source tensor, not the destination
    std::string srcName = getStoreSrcName(op);
    label += opName + "(" + srcName + ")";
  } else {
    // Generic: out = op(inputs)
    std::string inputs = getTensorInputs(op);
    if (op->getNumResults() > 0) {
      label += outputName + " = " + opName + "(" + inputs + ")";
    } else {
      label += opName + "(" + inputs + ")";
    }
  }

  // Add shape and loop depth on second line
  std::string shape = getOutputShape(op);
  int loopDepth = getLoopDepth(op);

  std::string secondLine;
  if (!shape.empty()) {
    secondLine = shape;
  }
  if (loopDepth > 0) {
    if (!secondLine.empty())
      secondLine += ", ";
    secondLine += "L" + std::to_string(loopDepth);
  }
  if (!secondLine.empty()) {
    label += "\\n" + secondLine;
  }

  return label;
}

/// Generate a DOT subgraph for key operations with control flow structure.
/// This creates a vertical flow showing the execution order of key ops.
static void dumpKeyOpsSubgraph(triton::FuncOp funcOp, llvm::raw_ostream &os,
                               const std::string &subgraphName) {
  os << "  subgraph cluster_" << subgraphName << " {\n";
  os << "    label=\"Key Operations\";\n";
  os << "    style=filled;\n";
  os << "    fillcolor=lightyellow;\n";
  os << "    node [shape=box, fontsize=9, style=filled];\n\n";

  int nodeCounter = 0;
  int clusterCounter = 0;

  // Recursive function to walk operations and create nested clusters
  std::function<void(Operation *, int, llvm::raw_ostream &, std::string &)>
      walkOp = [&](Operation *op, int depth, llvm::raw_ostream &clusterOs,
                   std::string &prevNodeId) {
        // Handle control flow operations - create nested clusters
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          if (!containsKeyOps(op))
            return;

          std::string clusterId =
              subgraphName + "_cluster_for_" + std::to_string(clusterCounter++);
          std::string forNodeId =
              subgraphName + "_for_" + std::to_string(nodeCounter++);

          // Start a new subgraph cluster for this for loop
          clusterOs << "    subgraph cluster_" << clusterId << " {\n";
          clusterOs << "      label=\"scf.for\";\n";
          clusterOs << "      style=rounded;\n";
          clusterOs << "      color=blue;\n";
          clusterOs << "      bgcolor=lightcyan;\n";

          std::string innerPrevId = "";
          for (Operation &innerOp : forOp.getBody()->getOperations()) {
            walkOp(&innerOp, depth + 1, clusterOs, innerPrevId);
          }

          clusterOs << "    }\n";

          // Connect previous node to first node in this cluster (if any)
          if (!prevNodeId.empty() && !innerPrevId.empty()) {
            // We'll handle this with ltail/lhead later if needed
          }
          if (!innerPrevId.empty()) {
            prevNodeId = innerPrevId;
          }
          return;
        }

        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          if (!containsKeyOps(op))
            return;

          std::string clusterId =
              subgraphName + "_cluster_if_" + std::to_string(clusterCounter++);

          // Start a new subgraph cluster for this if statement
          clusterOs << "    subgraph cluster_" << clusterId << " {\n";
          clusterOs << "      label=\"scf.if\";\n";
          clusterOs << "      style=rounded;\n";
          clusterOs << "      color=magenta;\n";
          clusterOs << "      bgcolor=mistyrose;\n";

          std::string innerPrevId = "";
          for (Operation &innerOp :
               ifOp.getThenRegion().front().getOperations()) {
            walkOp(&innerOp, depth + 1, clusterOs, innerPrevId);
          }
          if (!ifOp.getElseRegion().empty()) {
            for (Operation &innerOp :
                 ifOp.getElseRegion().front().getOperations()) {
              walkOp(&innerOp, depth + 1, clusterOs, innerPrevId);
            }
          }

          clusterOs << "    }\n";

          if (!innerPrevId.empty()) {
            prevNodeId = innerPrevId;
          }
          return;
        }

        // Check if this is a key operation
        if (isKeyOp(op)) {
          std::string nodeId = subgraphName + "_" + getNodeId(op);

          // Build label using the new format
          std::string label = getKeyOpLabel(op);

          // Color based on partition number (async_task_id)
          // Color palette for different partitions
          static const std::vector<std::string> partitionColors = {
              "lightblue",   // Partition 0
              "lightgreen",  // Partition 1
              "lightsalmon", // Partition 2
              "lightyellow", // Partition 3
              "lightpink",   // Partition 4
              "lightcyan",   // Partition 5
              "lavender",    // Partition 6
              "wheat",       // Partition 7
          };

          std::string fillcolor = "white";
          auto taskIds = getAsyncTaskIds(op);
          if (!taskIds.empty()) {
            int partitionNum = taskIds.front();
            fillcolor = partitionColors[partitionNum % partitionColors.size()];
          }

          clusterOs << "      " << nodeId << " [label=\"" << label
                    << "\", fillcolor=" << fillcolor << "];\n";

          // Connect to previous node for vertical ordering
          if (!prevNodeId.empty()) {
            clusterOs << "      " << prevNodeId << " -> " << nodeId
                      << " [style=invis];\n";
          }
          prevNodeId = nodeId;
        }
      };

  // Walk through the function body
  std::string prevNodeId = "";
  for (Operation &op : funcOp.getBody().front().getOperations()) {
    walkOp(&op, 0, os, prevNodeId);
  }

  os << "  }\n\n";
}

/// Generate a combined DOT graph showing key ops and channels side by side.
/// Left side: Key operations with control flow
/// Right side: Channel connections between partitions
void dumpCombinedGraph(SmallVector<std::unique_ptr<Channel>> &channels,
                       triton::FuncOp funcOp, llvm::raw_ostream &os) {
  os << "\n=== Combined Key Ops + Channel Graph (DOT format) ===\n";
  os << "// Render with: dot -Tpng <file>.dot -o graph.png\n";
  os << "digraph CombinedGraph {\n";
  os << "  rankdir=TB;\n";
  os << "  compound=true;\n";
  os << "  node [shape=box, style=filled, fontsize=9];\n";
  os << "  edge [fontsize=7];\n\n";

  // Color palette for different partitions
  static const std::vector<std::string> partitionColors = {
      "lightblue",   // Partition 0
      "lightgreen",  // Partition 1
      "lightsalmon", // Partition 2
      "lightyellow", // Partition 3
      "lightpink",   // Partition 4
      "lightcyan",   // Partition 5
      "lavender",    // Partition 6
      "wheat",       // Partition 7
  };

  // Collect all key operations and channel operations, grouped by partition
  DenseMap<int, SmallVector<Operation *>> partitionOps;
  DenseSet<Operation *> channelOps; // Track ops that are in channels

  // First, collect operations from channels
  for (auto &ch : channels) {
    Operation *srcOp = ch->getSrcOp();
    Operation *dstOp = ch->getDstOp();
    int producerId = ch->relation.first;

    if (srcOp) {
      channelOps.insert(srcOp);
      // Add to partition if not already there
      auto &ops = partitionOps[producerId];
      if (std::find(ops.begin(), ops.end(), srcOp) == ops.end()) {
        ops.push_back(srcOp);
      }
    }

    for (int consumerId : ch->relation.second) {
      if (dstOp) {
        channelOps.insert(dstOp);
        auto &ops = partitionOps[consumerId];
        if (std::find(ops.begin(), ops.end(), dstOp) == ops.end()) {
          ops.push_back(dstOp);
        }
      }
    }
  }

  // Now collect all key operations and add those not in channels
  std::function<void(Operation *)> collectKeyOps = [&](Operation *op) {
    // Recurse into nested regions
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Operation &innerOp : block) {
          collectKeyOps(&innerOp);
        }
      }
    }

    // Check if this is a key operation
    if (isKeyOp(op)) {
      // Get partition from async_task_id
      auto taskIds = getAsyncTaskIds(op);
      if (!taskIds.empty()) {
        int partitionId = taskIds.front();
        auto &ops = partitionOps[partitionId];
        if (std::find(ops.begin(), ops.end(), op) == ops.end()) {
          ops.push_back(op);
        }
      }
    }
  };

  // Collect key ops from function body
  for (Operation &op : funcOp.getBody().front().getOperations()) {
    collectKeyOps(&op);
  }

  // Sort partition IDs
  SmallVector<int> sortedPartitions;
  for (auto &kv : partitionOps) {
    sortedPartitions.push_back(kv.first);
  }
  llvm::sort(sortedPartitions);

  // Create nested subgraphs for each partition with nodes in program order
  for (int partId : sortedPartitions) {
    // Sort operations by operation_id (program order)
    auto &ops = partitionOps[partId];
    llvm::sort(ops, [](Operation *a, Operation *b) {
      return getOperationId(a) < getOperationId(b);
    });

    std::string fillcolor = partitionColors[partId % partitionColors.size()];
    // Use a lighter version of the color for the cluster background
    // Graphviz uses #RRGGBBAA format for transparency
    std::string bgColor = fillcolor;

    os << "  subgraph cluster_partition_" << partId << " {\n";
    os << "    label=\"Partition " << partId << "\";\n";
    os << "    style=filled;\n";
    os << "    fillcolor=\"" << bgColor << "\";\n";
    os << "    color=blue;\n";

    std::string prevNodeId = "";
    for (Operation *op : ops) {
      std::string nodeId = "op_" + getNodeId(op);

      // Use key op label format for all nodes
      std::string label = getKeyOpLabel(op);

      // Color node based on partition
      std::string nodeFillColor = fillcolor;

      // Add border color based on channel type
      std::string borderColor = "black";
      bool inChannel = channelOps.contains(op);
      if (inChannel) {
        for (auto &ch : channels) {
          if (ch->getSrcOp() == op || ch->getDstOp() == op) {
            if (ch->channelKind == DataChannelKind::TMEMPost) {
              borderColor = "red";
              break;
            } else if (ch->channelKind == DataChannelKind::SMEMPost) {
              borderColor = "darkgreen";
            }
          }
        }
      }

      os << "    " << nodeId << " [label=\"" << label << "\", fillcolor=\""
         << nodeFillColor << "\", color=" << borderColor << "];\n";

      // Add invisible edge for vertical ordering within partition
      if (!prevNodeId.empty()) {
        os << "    " << prevNodeId << " -> " << nodeId << " [style=invis];\n";
      }
      prevNodeId = nodeId;
    }
    os << "  }\n\n";
  }

  // Channel edges
  os << "  // Channel edges\n";
  for (auto &ch : channels) {
    Operation *srcOp = ch->getSrcOp();
    Operation *dstOp = ch->getDstOp();

    if (!srcOp || !dstOp)
      continue;

    std::string srcId = "op_" + getNodeId(srcOp);
    std::string dstId = "op_" + getNodeId(dstOp);

    std::string style = "solid";
    std::string color = "black";
    std::string edgeLabel = "ch" + std::to_string(ch->uniqID);

    // Add buffer ID if available
    Operation *allocOp = ch->getAllocOp();
    int bufferId = getBufferId(allocOp);
    if (bufferId >= 0) {
      edgeLabel += " B" + std::to_string(bufferId);
    }

    std::string locName = getNamedLoc(srcOp);
    if (locName.empty()) {
      locName = getNamedLoc(allocOp);
    }
    if (!locName.empty()) {
      edgeLabel += "\\n\\\"" + locName + "\\\"";
    }

    if (ch->channelKind == DataChannelKind::TMEMPost) {
      color = "red";
      edgeLabel += "\\n(TMEM)";
      auto *tmemCh = static_cast<ttng::TmemDataChannelPost *>(ch.get());
      if (tmemCh->isOperandD) {
        style = "bold";
        edgeLabel += " [D]";
      }
    } else if (ch->channelKind == DataChannelKind::SMEMPost) {
      color = "darkgreen";
      edgeLabel += "\\n(SMEM)";
    }

    // Mark cross-stage channels: source and destination in different
    // pipeline stages.
    auto srcStageAttr =
        srcOp->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
    auto dstStageAttr =
        dstOp->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
    if (srcStageAttr && dstStageAttr) {
      int srcStage = srcStageAttr.getValue().getSExtValue();
      int dstStage = dstStageAttr.getValue().getSExtValue();
      if (srcStage != dstStage) {
        edgeLabel += "\\n[CROSS-STAGE s" + std::to_string(srcStage) + "->s" +
                     std::to_string(dstStage) + "]";
        style = "dashed";
        color = "purple";
      }
    }

    os << "  " << srcId << " -> " << dstId << " [label=\"" << edgeLabel
       << "\", color=" << color << ", style=" << style << "];\n";

    // Draw additional edges for other destinations that are in a different
    // stage from the source (cross-stage consumers not already shown).
    if (srcOp) {
      auto srcSA = srcOp->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
      if (srcSA) {
        int srcS = srcSA.getValue().getSExtValue();
        SmallVector<Operation *> allDsts;
        bool canGetDsts = true;
        if (ch->channelKind == DataChannelKind::TMEMPost) {
          auto *tmCh = static_cast<ttng::TmemDataChannelPost *>(ch.get());
          if (tmCh->isOperandD)
            canGetDsts = false;
        }
        if (canGetDsts)
          ch->getDstOps(allDsts);
        for (Operation *extraDst : allDsts) {
          if (extraDst == dstOp)
            continue;
          auto dstSA =
              extraDst->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
          if (!dstSA)
            continue;
          int dstS = dstSA.getValue().getSExtValue();
          if (srcS != dstS) {
            std::string extraDstId = "op_" + getNodeId(extraDst);
            std::string extraLabel =
                "ch" + std::to_string(ch->uniqID) + "\\n[CROSS-STAGE s" +
                std::to_string(srcS) + "->s" + std::to_string(dstS) + "]";
            os << "  " << srcId << " -> " << extraDstId << " [label=\""
               << extraLabel << "\", color=purple, style=dashed];\n";
          }
        }
      }
    }
  }

  os << "}\n";
  os << "=== End Combined Graph ===\n";
}

/// Generate a buffer liveness visualization for TMEM allocations using
/// pre-calculated liveness intervals from the memory planner.
void dumpTmemBufferLiveness(
    SmallVector<ttng::TMEMAllocOp> &allocs,
    DenseMap<Operation *, Interval<size_t>> &allocToIntervals,
    DenseMap<Operation *, ttng::TMemAllocation> &allocToSize,
    DenseMap<Operation *, ttng::TmemDataChannelPost *> &allocToChannel,
    SmallVector<Channel *> &channels, llvm::raw_ostream &os) {
  os << "=== TMEM Buffer Liveness Graph ===\n";
  os << "digraph TmemBufferLiveness {\n";
  os << "  rankdir=LR;\n";
  os << "  node [shape=record, fontsize=9];\n";
  os << "  edge [style=invis];\n\n";

  if (allocs.empty()) {
    os << "  empty [label=\"No TMEM allocations\"];\n";
    os << "}\n";
    os << "=== End TMEM Buffer Liveness Graph ===\n";
    return;
  }

  // Find all channels for each alloc (handles OperandD case with multiple
  // channels)
  DenseMap<Operation *, SmallVector<Channel *>> allocToAllChannels;
  for (auto *ch : channels) {
    if (ch->channelKind != DataChannelKind::TMEMPost)
      continue;
    Operation *allocOp = ch->getAllocOp();
    if (allocOp)
      allocToAllChannels[allocOp].push_back(ch);
  }

  // Find global min/max for axis
  size_t globalMin = std::numeric_limits<size_t>::max();
  size_t globalMax = 0;
  for (auto &alloc : allocs) {
    auto it = allocToIntervals.find(alloc.getOperation());
    if (it != allocToIntervals.end()) {
      globalMin = std::min(globalMin, it->second.start());
      globalMax = std::max(globalMax, it->second.end());
    }
  }

  if (globalMin == std::numeric_limits<size_t>::max()) {
    os << "  empty [label=\"No liveness intervals\"];\n";
    os << "}\n";
    os << "=== End TMEM Buffer Liveness Graph ===\n";
    return;
  }

  // Create a time axis at the top
  os << "  // Time axis\n";
  os << "  subgraph cluster_axis {\n";
  os << "    label=\"Operation ID\";\n";
  os << "    style=invis;\n";
  os << "    axis [shape=none, label=\"";
  size_t step = std::max((globalMax - globalMin) / 10, (size_t)1);
  for (size_t i = globalMin; i <= globalMax; i += step) {
    os << i;
    if (i + step <= globalMax)
      os << "  |  ";
  }
  os << "\"];\n";
  os << "  }\n\n";

  // Color palette for buffers
  static const std::vector<std::string> tmemColors = {
      "lightpink",   "lavender",  "peachpuff", "thistle",
      "lightyellow", "lightcyan", "wheat",     "lightgreen"};

  // Create a subgraph for each TMEM alloc
  int allocIdx = 0;
  std::string prevAllocNode;

  for (auto &alloc : allocs) {
    auto intervalIt = allocToIntervals.find(alloc.getOperation());
    if (intervalIt == allocToIntervals.end())
      continue;

    Interval<size_t> interval = intervalIt->second;
    std::string color = tmemColors[allocIdx % tmemColors.size()];
    std::string allocNode = "tmem_" + std::to_string(allocIdx);

    // Get buffer name from location
    std::string bufferName = getNamedLoc(alloc.getOperation());
    if (bufferName.empty())
      bufferName = "alloc" + std::to_string(allocIdx);

    // Get row x col size
    std::string sizeStr;
    auto sizeIt = allocToSize.find(alloc.getOperation());
    if (sizeIt != allocToSize.end()) {
      sizeStr = std::to_string(sizeIt->second.numRows) + "x" +
                std::to_string(sizeIt->second.numCols);
    }

    // Get all channels for this alloc
    auto &allocChannels = allocToAllChannels[alloc.getOperation()];

    // Count OperandD channels
    int operandDCount = 0;
    for (auto *ch : allocChannels) {
      auto *tmemCh = static_cast<ttng::TmemDataChannelPost *>(ch);
      if (tmemCh->isOperandD)
        operandDCount++;
    }

    // Build label with row x col size
    std::string bufLabel = bufferName;
    if (!sizeStr.empty())
      bufLabel += " " + sizeStr;
    bufLabel += " [" + std::to_string(interval.start()) + "-" +
                std::to_string(interval.end()) + ")";
    if (operandDCount > 0) {
      bufLabel += " [" + std::to_string(operandDCount) + " OperandD]";
    }

    os << "  // TMEM Alloc: " << bufferName << "\n";
    os << "  subgraph cluster_" << allocNode << " {\n";
    os << "    label=\"" << bufLabel << "\";\n";
    os << "    style=filled;\n";
    os << "    fillcolor=\"" << color << "\";\n";
    os << "    color=black;\n\n";

    // Create a node for each channel in this alloc
    std::string prevChNode;
    for (auto *ch : allocChannels) {
      auto *tmemCh = static_cast<ttng::TmemDataChannelPost *>(ch);
      std::string chNode = allocNode + "_ch" + std::to_string(ch->uniqID);

      // Get src/dst operation IDs if available
      std::string label = "ch" + std::to_string(ch->uniqID);
      if (tmemCh->isOperandD) {
        label += " [D]";
      }

      // Add src->dst info
      Operation *srcOp = ch->getSrcOp();
      Operation *dstOp = ch->getDstOp();
      if (srcOp && dstOp) {
        int srcId = getOperationId(srcOp);
        int dstId = getOperationId(dstOp);
        if (srcId >= 0 && dstId >= 0) {
          label += " (" + std::to_string(srcId) + " to " +
                   std::to_string(dstId) + ")";
        }
      }

      os << "    " << chNode << " [label=\"" << label
         << "\", style=filled, fillcolor=white];\n";

      if (!prevChNode.empty()) {
        os << "    " << prevChNode << " -> " << chNode << " [style=invis];\n";
      }
      prevChNode = chNode;
    }

    // If no channels, show the liveness interval
    if (allocChannels.empty()) {
      std::string infoNode = allocNode + "_info";
      os << "    " << infoNode << " [label=\"no channels\", style=filled, "
         << "fillcolor=white];\n";
      prevChNode = infoNode;
    }

    os << "  }\n\n";

    // Link allocs to maintain order
    if (!prevAllocNode.empty() && !prevChNode.empty()) {
      os << "  " << prevAllocNode << " -> "
         << (allocChannels.empty()
                 ? allocNode + "_info"
                 : allocNode + "_ch" + std::to_string(allocChannels[0]->uniqID))
         << " [style=invis];\n";
    }
    if (!prevChNode.empty()) {
      prevAllocNode = prevChNode;
    }
    allocIdx++;
  }

  // Create a summary table
  os << "\n  // Summary table\n";
  os << "  subgraph cluster_summary {\n";
  os << "    label=\"TMEM Allocation Summary\";\n";
  os << "    style=filled;\n";
  os << "    fillcolor=white;\n";
  os << "    summary [shape=none, label=<\n";
  os << "      <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">\n";
  os << "        "
        "<TR><TD><B>Name</B></TD><TD><B>Size</B></TD><TD><B>Channels</B></"
        "TD><TD><B>"
        "Liveness</B></TD><TD><B>OperandD</B></TD></TR>\n";

  for (auto &alloc : allocs) {
    auto intervalIt = allocToIntervals.find(alloc.getOperation());
    if (intervalIt == allocToIntervals.end())
      continue;

    std::string bufferName = getNamedLoc(alloc.getOperation());
    if (bufferName.empty())
      bufferName = "alloc";

    // Get row x col size for summary
    std::string sizeStr = "-";
    auto sizeIt = allocToSize.find(alloc.getOperation());
    if (sizeIt != allocToSize.end()) {
      sizeStr = std::to_string(sizeIt->second.numRows) + "x" +
                std::to_string(sizeIt->second.numCols);
    }

    auto &allocChannels = allocToAllChannels[alloc.getOperation()];
    int operandDCount = 0;
    for (auto *ch : allocChannels) {
      auto *tmemCh = static_cast<ttng::TmemDataChannelPost *>(ch);
      if (tmemCh->isOperandD)
        operandDCount++;
    }

    os << "        <TR><TD>" << bufferName << "</TD><TD>" << sizeStr
       << "</TD><TD>" << allocChannels.size() << "</TD><TD>["
       << intervalIt->second.start() << "-" << intervalIt->second.end()
       << ")</TD><TD>" << operandDCount << "</TD></TR>\n";
  }

  os << "      </TABLE>\n";
  os << "    >];\n";
  os << "  }\n";

  os << "}\n";
  os << "=== End TMEM Buffer Liveness Graph ===\n";
}

void dumpSmemBufferLiveness(
    llvm::MapVector<Allocation::BufferId, std::pair<Interval<size_t>, size_t>>
        &bufferInfo,
    DenseMap<Allocation::BufferId, Operation *> &bufferOwners,
    SmallVector<Channel *> &channels, llvm::raw_ostream &os) {
  os << "=== SMEM Buffer Liveness Graph ===\n";
  os << "digraph SmemBufferLiveness {\n";
  os << "  rankdir=LR;\n";
  os << "  node [shape=record, fontsize=9];\n";
  os << "  edge [style=invis];\n\n";

  if (bufferInfo.empty()) {
    os << "  empty [label=\"No SMEM allocations\"];\n";
    os << "}\n";
    os << "=== End SMEM Buffer Liveness Graph ===\n";
    return;
  }

  // Find all SMEM channels for each alloc
  DenseMap<Operation *, SmallVector<Channel *>> allocToAllChannels;
  for (auto *ch : channels) {
    if (ch->channelKind != DataChannelKind::SMEMPost)
      continue;
    Operation *allocOp = ch->getAllocOp();
    if (allocOp)
      allocToAllChannels[allocOp].push_back(ch);
  }

  // Find global min/max for axis
  size_t globalMin = std::numeric_limits<size_t>::max();
  size_t globalMax = 0;
  for (auto &[bufferId, info] : bufferInfo) {
    auto &interval = info.first;
    if (interval.start() == 0 && interval.end() == 0)
      continue;
    globalMin = std::min(globalMin, interval.start());
    globalMax = std::max(globalMax, interval.end());
  }

  if (globalMin == std::numeric_limits<size_t>::max()) {
    os << "  empty [label=\"No liveness intervals\"];\n";
    os << "}\n";
    os << "=== End SMEM Buffer Liveness Graph ===\n";
    return;
  }

  // Create a time axis at the top
  os << "  // Time axis\n";
  os << "  subgraph cluster_axis {\n";
  os << "    label=\"Operation ID\";\n";
  os << "    style=invis;\n";
  os << "    axis [shape=none, label=\"";
  size_t step = std::max((globalMax - globalMin) / 10, (size_t)1);
  for (size_t i = globalMin; i <= globalMax; i += step) {
    os << i;
    if (i + step <= globalMax)
      os << "  |  ";
  }
  os << "\"];\n";
  os << "  }\n\n";

  // Color palette for buffers
  static const std::vector<std::string> smemColors = {
      "lightblue",   "lightgreen", "lightyellow", "lightcoral",
      "lightsalmon", "lightcyan",  "lavender",    "peachpuff"};

  // Create a subgraph for each SMEM buffer
  int bufferIdx = 0;
  std::string prevBufferNode;

  for (auto &[bufferId, info] : bufferInfo) {
    auto &interval = info.first;
    auto bufferSize = info.second;

    if (interval.start() == 0 && interval.end() == 0)
      continue;

    Operation *owner = bufferOwners.lookup(bufferId);
    std::string color = smemColors[bufferIdx % smemColors.size()];
    std::string bufferNode = "smem_" + std::to_string(bufferIdx);

    // Get buffer name from location
    std::string bufferName = owner ? getNamedLoc(owner) : "";
    if (bufferName.empty())
      bufferName = "alloc" + std::to_string(bufferIdx);

    // Get all channels for this alloc
    auto &allocChannels =
        owner ? allocToAllChannels[owner] : allocToAllChannels[nullptr];

    // Build label with buffer ID and size
    std::string bufLabel = bufferName + " B" + std::to_string(bufferId) + " [" +
                           std::to_string(interval.start()) + "-" +
                           std::to_string(interval.end()) + ")";
    bufLabel += " size=" + std::to_string(bufferSize);

    os << "  // SMEM Buffer: " << bufferName << "\n";
    os << "  subgraph cluster_" << bufferNode << " {\n";
    os << "    label=\"" << bufLabel << "\";\n";
    os << "    style=filled;\n";
    os << "    fillcolor=\"" << color << "\";\n";
    os << "    color=black;\n\n";

    // Create a node for each channel in this buffer
    std::string prevChNode;
    for (auto *ch : allocChannels) {
      std::string chNode = bufferNode + "_ch" + std::to_string(ch->uniqID);

      // Get src/dst operation IDs if available
      std::string label = "ch" + std::to_string(ch->uniqID);

      // Add src->dst info
      Operation *srcOp = ch->getSrcOp();
      Operation *dstOp = ch->getDstOp();
      if (srcOp && dstOp) {
        int srcId = getOperationId(srcOp);
        int dstId = getOperationId(dstOp);
        if (srcId >= 0 && dstId >= 0) {
          label += " (" + std::to_string(srcId) + " to " +
                   std::to_string(dstId) + ")";
        }
      }

      os << "    " << chNode << " [label=\"" << label
         << "\", style=filled, fillcolor=white];\n";

      if (!prevChNode.empty()) {
        os << "    " << prevChNode << " -> " << chNode << " [style=invis];\n";
      }
      prevChNode = chNode;
    }

    // If no channels, show the liveness interval
    if (allocChannels.empty()) {
      std::string infoNode = bufferNode + "_info";
      os << "    " << infoNode << " [label=\"no channels\", style=filled, "
         << "fillcolor=white];\n";
      prevChNode = infoNode;
    }

    os << "  }\n\n";

    // Link buffers to maintain order
    if (!prevBufferNode.empty() && !prevChNode.empty()) {
      os << "  " << prevBufferNode << " -> "
         << (allocChannels.empty()
                 ? bufferNode + "_info"
                 : bufferNode + "_ch" +
                       std::to_string(allocChannels[0]->uniqID))
         << " [style=invis];\n";
    }
    if (!prevChNode.empty()) {
      prevBufferNode = prevChNode;
    }
    bufferIdx++;
  }

  // Create a summary table
  os << "\n  // Summary table\n";
  os << "  subgraph cluster_summary {\n";
  os << "    label=\"SMEM Buffer Summary\";\n";
  os << "    style=filled;\n";
  os << "    fillcolor=white;\n";
  os << "    summary [shape=none, label=<\n";
  os << "      <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">\n";
  os << "        <TR><TD><B>Name</B></TD><TD><B>BufferID</B></TD><TD><B>"
        "Size</B></TD><TD><B>Channels</B></TD><TD><B>Liveness</B></TD></TR>\n";

  bufferIdx = 0;
  for (auto &[bufferId, info] : bufferInfo) {
    auto &interval = info.first;
    auto bufferSize = info.second;

    if (interval.start() == 0 && interval.end() == 0)
      continue;

    Operation *owner = bufferOwners.lookup(bufferId);
    std::string bufferName = owner ? getNamedLoc(owner) : "";
    if (bufferName.empty())
      bufferName = "alloc" + std::to_string(bufferIdx);

    auto &allocChannels =
        owner ? allocToAllChannels[owner] : allocToAllChannels[nullptr];

    os << "        <TR><TD>" << bufferName << "</TD><TD>" << bufferId
       << "</TD><TD>" << bufferSize << "</TD><TD>" << allocChannels.size()
       << "</TD><TD>[" << interval.start() << "-" << interval.end()
       << ")</TD></TR>\n";
    bufferIdx++;
  }

  os << "      </TABLE>\n";
  os << "    >];\n";
  os << "  }\n";

  os << "}\n";
  os << "=== End SMEM Buffer Liveness Graph ===\n";
}
///
/// This function creates producer-consumer channels for a TMEM allocation that
/// is used as the accumulator (operand D) of a TCGen5MMA operation. The
/// accumulator follows a read-modify-write pattern where:
///   1. A producer writes to the TMEM (either a tmem_store or an MMA)
///   2. The MMA reads the accumulator, performs computation, and writes back
///
/// The function handles several cases for finding the initial producer:
///   - TMEMStoreOp outside the loop: Initialization before the loop starts
///   - MMA with use_acc=false: The MMA overwrites (doesn't accumulate), so it
///     becomes the first producer without needing a prior value
///   - TMEMStoreOp inside the loop: Re-initialization within the loop
///
/// For each producer-consumer pair, a TmemDataChannelPost is created to track
/// the data dependency for warp specialization scheduling.
///
/// @param tmemAllocOp The TMEM allocation used as operand D
/// @param mmaOp The MMA operation that uses this TMEM as its accumulator
/// @param channels Output vector to collect the created channels
/// @return success() if channels were created successfully, failure() otherwise
static LogicalResult
handleOperandD(ttng::TMEMAllocOp tmemAllocOp, ttng::TCGen5MMAOp mmaOp,
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
  if (!forOp) {
    return mmaOp.emitError(
        "handleOperandD: MMA operation is not inside a scf.for loop");
  }
  // Track multiple producers when channels are skipped (same task IDs).
  // All producers in the vector must share the exact same task IDs.
  SmallVector<Operation *> currentProds;
  auto ctx = forOp.getContext();
  SmallVector<int> channelsToBeUpdate;

  // Check for producers outside the loop body (e.g., tmem_store before the
  // loop that initializes the accumulator). These producers dominate the loop.
  for (auto user : tmemAllocOp.getResult().getUsers()) {
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(user)) {
      // Check if this store is outside the loop (not nested under forOp)
      if (!forOp->isProperAncestor(storeOp)) {
        currentProds.clear();
        currentProds.push_back(storeOp);
        handledUsers.insert(storeOp);
      }
    }
  }

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!users.count(&op))
      continue;
    handledUsers.insert(&op);
    if (auto mmaOpT = dyn_cast<ttng::TCGen5MMAOp>(&op)) {
      if (&op == mmaOp.getOperation()) {
        // This uses and defines D. Will be both producer and consumer.
        // If useAcc is false, the MMA doesn't read the accumulator - it
        // overwrites it completely. In this case, the MMA is the first
        // producer and doesn't need a prior producer.
        if (currentProds.empty()) {
          Value useAccFlag = mmaOpT.useAccumulator();
          bool useAccIsFalse = false;
          if (useAccFlag) {
            // If useAccFlag is a block argument of the loop, trace it back
            // to its init value. Even if useAccFlag may be true, we don't
            // need a producer if useAcc = False for the first iteration.
            if (auto blockArg = dyn_cast<BlockArgument>(useAccFlag)) {
              if (blockArg.getOwner() == forOp.getBody()) {
                // Block arg 0 is the induction variable, so iter args start
                // at index 1.
                unsigned argNum = blockArg.getArgNumber();
                if (argNum > 0) {
                  useAccFlag = forOp.getInitArgs()[argNum - 1];
                }
              }
            }
            if (auto constOp = useAccFlag.getDefiningOp<arith::ConstantOp>()) {
              if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
                useAccIsFalse = !boolAttr.getValue();
              } else if (auto intAttr =
                             dyn_cast<IntegerAttr>(constOp.getValue())) {
                useAccIsFalse = intAttr.getInt() == 0;
              }
            }
          }
          if (useAccIsFalse) {
            // MMA with use_acc=false is the first producer
            currentProds.clear();
            currentProds.push_back(&op);
            continue;
          }
        }
        if (currentProds.empty()) {
          mmaOp.emitError(
              "handleOperandD: no producer found for MMA operand D. "
              "Expected a tmem_store before the loop or use_acc=false.");
          return failure();
        }
        // Start a channel from currentProds to op
        auto producerTaskIds = getAsyncTaskIds(currentProds.front());
        auto consumerIds = getAsyncTaskIds(&op);
        if (producerTaskIds.size() != 1) {
          mmaOp.emitError(
              "handleOperandD: expected exactly one producer task ID, got ")
              << producerTaskIds.size();
          return failure();
        }
        int producerTaskId = producerTaskIds.front();
        if (needsChannel(producerTaskId, consumerIds)) {
          createChannelsForProducers(currentProds, producerTaskId, consumerIds,
                                     tmemAllocOp.getOperation(), &op, channels);
          currentProds.clear();
          currentProds.push_back(&op);
        } else {
          // Channel skipped - append to producers vector
          currentProds.push_back(&op);
        }
      } else {
        if (mmaOpT.getD() == tmemAllocOp.getResult()) {
          mmaOp.emitError(
              "handleOperandD: unexpected MMA using same TMEM as operand D");
          return failure();
        }
        // This uses tmem. mark as tmem.end = channel_id
        if (currentProds.empty()) {
          mmaOpT.emitError(
              "handleOperandD: no producer found for MMA consumer");
          return failure();
        }
        // Start a channel from currentProds to op
        auto producerTaskIds = getAsyncTaskIds(currentProds.front());
        if (producerTaskIds.size() != 1) {
          mmaOpT.emitError(
              "handleOperandD: expected exactly one producer task ID, got ")
              << producerTaskIds.size();
          return failure();
        }
        auto producerTaskId = producerTaskIds.front();
        auto consumerIds = getAsyncTaskIds(&op);
        if (needsChannel(producerTaskId, consumerIds)) {
          createChannelsForProducers(currentProds, producerTaskId, consumerIds,
                                     tmemAllocOp.getOperation(), &op, channels);
        } else {
          // Channel skipped - append to producers vector
          currentProds.push_back(&op);
        }
      }
    } else if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(&op)) {
      currentProds.clear();
      currentProds.push_back(&op); // mark as tmem.start = channel_id
    } else if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(&op)) {
      if (!currentProds.empty()) {
        // Start a channel from currentProds to op
        auto producerTaskIds = getAsyncTaskIds(currentProds.front());
        if (producerTaskIds.size() != 1) {
          loadOp.emitError("handleOperandD: expected exactly one producer task "
                           "ID for TMEMLoad, got ")
              << producerTaskIds.size();
          return failure();
        }
        auto producerTaskId = producerTaskIds.front();
        auto consumerIds = getAsyncTaskIds(&op);
        if (needsChannel(producerTaskId, consumerIds)) {
          createChannelsForProducers(currentProds, producerTaskId, consumerIds,
                                     tmemAllocOp.getOperation(), &op, channels);
        } else {
          // Channel skipped - append to producers vector
          currentProds.push_back(&op);
        }
      } else {
        channelsToBeUpdate.push_back(channels.size());
        auto channelID = channels.size();
        auto consumerIds = getAsyncTaskIds(&op);
        channels.push_back(std::make_unique<ttng::TmemDataChannelPost>(
            -1, consumerIds, tmemAllocOp.getOperation(), true /*isOperandD*/,
            true, channels.size()));
        // Mark producer and consumer.
        setTmemChannelAttr(&op, channelID, "tmem.end");
      }
    } else {
      // Unexpected operation type using the TMEM
      return op.emitError(
          "handleOperandD: unexpected operation type using TMEM");
    }
  }
  // Update channel's producer here.
  for (auto idx : channelsToBeUpdate) {
    if (currentProds.empty()) {
      // This can happen if ForOp never produces - should not occur in valid IR
      return mmaOp.emitError(
          "handleOperandD: no producer found for deferred channel update");
    }
    // For deferred channels, we only have one channel per consumer, so use
    // the last producer in the vector (which should be the most recent).
    auto *lastProd = currentProds.back();
    channels[idx]->relation.first = getAsyncTaskIds(lastProd).front();
    setTmemChannelAttr(lastProd, channels[idx]->uniqID, "tmem.start");
  }
  // For consumers outside of ForOp.
  for (auto *user : users) {
    if (handledUsers.count(user))
      continue;
    // only handle tmem_load. FIXME: check if it is after the ForOp
    if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (currentProds.empty()) {
        return loadOp.emitError(
            "handleOperandD: no producer found for TMEMLoad outside loop");
      }
      // Start a channel from currentProds to user
      auto producerTaskIds = getAsyncTaskIds(currentProds.front());
      if (producerTaskIds.size() != 1) {
        return loadOp.emitError("handleOperandD: expected exactly one producer "
                                "task ID, got ")
               << producerTaskIds.size();
      }
      auto producerTaskId = producerTaskIds.front();
      auto consumerIds = getAsyncTaskIds(user);
      if (needsChannel(producerTaskId, consumerIds)) {
        createChannelsForProducers(currentProds, producerTaskId, consumerIds,
                                   tmemAllocOp.getOperation(), user, channels);
      } else {
        assert(false && "Unexpected Producer Found");
      }
    }
  }
  LLVM_DEBUG({
    llvm::dbgs() << "\n[handleOperandD] Completed channel creation\n";
    dumpChannelsForOperandD(tmemAllocOp, channels, llvm::dbgs());
  });
  return success();
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
  bool isOperandDNoAcc = false;
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
          } else {
            isOperandDNoAcc = true;
            producers.push_back(user);
          }
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
      if (failed(handleOperandD(tmemAllocOp, mmaOp, channels))) {
        // Error already emitted by handleOperandD
        return;
      }
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
    auto localAlloc = cast<ttg::LocalAllocOp>(allocOp);
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
    // If no LocalStoreOp user but the alloc has a tensor source,
    // the local_alloc itself is the producer (direct alloc+store).
    if (!producerOp && localAlloc.getSrc())
      producerOp = allocOp;
  }
  // FIXME: If we couldn't find a valid producer (e.g., for allocs outside the
  // loop), skip creating a channel for this allocation.
  if (!producerOp)
    return;
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

  if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(allocOp)) {
    if (needsChannel(producerTaskId, consumerTaskIds)) {
      channels.push_back(std::make_unique<ttng::TmemDataChannelPost>(
          producerTaskId, consumerTaskIds, allocOp, false, isOperandDNoAcc,
          channels.size()));
    }
  } else
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
  LLVM_DEBUG({
    llvm::dbgs() << "\n[collectPostChannels] Completed channel collection\n";
    dumpAllChannels(channels, llvm::dbgs());
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
  return nullptr;
  // llvm_unreachable("Failed to find consumer's same level Op with producer");
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

struct CommitOpSubgroupInfo {
  // Arrive value from the init Barrier
  int initCount;
  SmallVector<Operation *> bufferAllocs;
  SmallVector<Operation *> bufferConsumers;
  SmallVector<ttng::WaitBarrierOp> barrierWaiters;
  SmallVector<ttng::TCGen5CommitOp> commits;
};

// Check if two values are certain to match given the assumption.
// that the original value are located in the same block and therefore
// occur with the same frequency.
bool valuesMatch(Value v1, Value v2) {
  if (v1 == v2) {
    return true;
  }
  auto *op1 = v1.getDefiningOp();
  auto *op2 = v2.getDefiningOp();
  if (!op1 || !op2) {
    return false;
  }
  // Verify the op types match
  if ((op1->getName() != op2->getName()) ||
      (op1->getNumOperands() != op2->getNumOperands())) {
    return false;
  }

  // Special case on constants
  if (auto const1 = dyn_cast<mlir::arith::ConstantOp>(op1)) {
    auto const2 = cast<mlir::arith::ConstantOp>(op2);
    return const1.getValue() == const2.getValue();
  }
  // Check all operands
  for (unsigned i = 0; i < op1->getNumOperands(); ++i) {
    if (!valuesMatch(op1->getOperand(i), op2->getOperand(i))) {
      return false;
    }
  }
  // If all operands match and we have the same exact op type then
  // this op matches.
  return true;
}

// Return True if the two ttng::WaitBarrierOp will either have
// exactly the same value or exactly the opposite value in
// every iteration of the loop. If so, then these are safe to fuse.
bool hasMatchingPhase(ttng::WaitBarrierOp wait1, ttng::WaitBarrierOp wait2) {
  return valuesMatch(wait1.getPhase(), wait2.getPhase());
}

void mergeSubgroups(std::vector<CommitOpSubgroupInfo> &subgroups, int initCount,
                    Operation *bufferAllocOp, ttng::TCGen5CommitOp commit,
                    SmallVector<Operation *> &consumers,
                    SmallVector<ttng::WaitBarrierOp> &barrierWaiters) {
  assert(consumers.size() == barrierWaiters.size());
  if (barrierWaiters.empty()) {
    return;
  }
  // Validate the inputs. All consumers must go to the same subgroup
  // to remove a barrier.
  auto initWaiter = barrierWaiters[0];
  for (size_t i = 1; i < consumers.size(); i++) {
    auto nextWaiter = barrierWaiters[i];
    if ((initWaiter->getParentOp() != nextWaiter->getParentOp()) &&
        hasMatchingPhase(initWaiter, nextWaiter)) {
      // Unsupported commit.
      return;
    }
  }
  bool found = false;
  auto insertIntoSubgroup =
      ([](CommitOpSubgroupInfo &subgroup, int initCount,
          Operation *bufferAllocOp, ttng::TCGen5CommitOp commit,
          SmallVector<Operation *> &consumers,
          SmallVector<ttng::WaitBarrierOp> &barrierWaiters) {
        subgroup.initCount = initCount;
        subgroup.bufferConsumers.insert(subgroup.bufferConsumers.end(),
                                        consumers.begin(), consumers.end());
        subgroup.barrierWaiters.insert(subgroup.barrierWaiters.end(),
                                       barrierWaiters.begin(),
                                       barrierWaiters.end());
        for (size_t j = 0; j < consumers.size(); j++) {
          subgroup.bufferAllocs.push_back(bufferAllocOp);
          subgroup.commits.push_back(commit);
        }
      });
  for (auto &subgroup : subgroups) {
    if (subgroup.initCount == initCount) {
      // Select a represetentive for comparison.
      auto groupWaiter = subgroup.barrierWaiters.front();
      // Require matching parent ops.
      if ((groupWaiter->getParentOp() == initWaiter->getParentOp()) &&
          hasMatchingPhase(groupWaiter, initWaiter)) {
        insertIntoSubgroup(subgroup, initCount, bufferAllocOp, commit,
                           consumers, barrierWaiters);
        found = true;
        break;
      }
    }
  }
  if (!found) {
    CommitOpSubgroupInfo subgroup;
    insertIntoSubgroup(subgroup, initCount, bufferAllocOp, commit, consumers,
                       barrierWaiters);
    subgroups.push_back(subgroup);
  }
}

void updateSubgroup(CommitOpSubgroupInfo &subgroup) {
  Operation *keptAlloc = nullptr;
  ttng::TCGen5CommitOp keptCommit = nullptr;
  // Track consumers + waiters we are planning to keep.
  // This is important because if we find two waiters
  // in the same task id we need to select the first one
  // in program order.
  SmallVector<Operation *> processedConsumers;
  SmallVector<ttng::WaitBarrierOp> processedWaiters;
  // Track alloc + commit which could be duplicated.
  DenseSet<Operation *> deletedOps;
  for (size_t i = 0; i < subgroup.bufferAllocs.size(); i++) {
    auto alloc = subgroup.bufferAllocs[i];
    auto commit = subgroup.commits[i];
    auto consumer = subgroup.bufferConsumers[i];
    auto waiter = subgroup.barrierWaiters[i];
    // Keep exactly one allocation and commit.
    // We know we are going to fuse all barriers together.
    if (keptAlloc == nullptr) {
      keptAlloc = alloc;
      keptCommit = commit;
      processedConsumers.push_back(consumer);
      processedWaiters.push_back(waiter);
      continue;
    }
    // If a barrier has already been fused its possible
    // multiple consumers share an alloc/commit.
    if (alloc != keptAlloc) {
      deletedOps.insert(alloc);
    }
    if (commit != keptCommit) {
      deletedOps.insert(commit);
    }
    // Check all existing operations for a matching task id.
    // Within the same task we will pick the earliest by
    // program order.
    auto taskId = waiter->getAttr("async_task_id");
    bool matched = false;
    bool keptWait = true;
    for (size_t j = 0; j < processedConsumers.size(); j++) {
      auto existingConsumer = processedConsumers[j];
      auto existingWaiter = processedWaiters[j];
      auto existingTaskID = existingWaiter->getAttr("async_task_id");
      if (taskId == existingTaskID) {
        // If task ids match we should delete whichever one comes later
        // in program order.
        if (existingWaiter->isBeforeInBlock(waiter)) {
          deletedOps.insert(waiter);
          deletedOps.insert(consumer);
          keptWait = false;
        } else {
          deletedOps.insert(existingWaiter);
          deletedOps.insert(existingConsumer);
          // Replace the existing consumer in place.
          processedConsumers[j] = consumer;
          processedWaiters[j] = waiter;
        }
        matched = true;
        break;
      }
    }
    if (!matched) {
      // If we only have a new task ID we must keep the wait.
      processedConsumers.push_back(consumer);
      processedWaiters.push_back(waiter);
    }
    if (keptWait) {
      // If we kept the wait then we should update
      // the allocation being used.
      consumer->replaceUsesOfWith(alloc->getResult(0), keptAlloc->getResult(0));
    }
  }
  // Remove the deleted ops.
  DenseSet<Operation *> erasedOps;
  std::function<void(Operation *)> eraseOp = [&](Operation *op) {
    if (erasedOps.count(op)) {
      return;
    }
    for (auto user : op->getUsers()) {
      eraseOp(user);
    }
    erasedOps.insert(op);
    op->erase();
  };
  for (auto op : deletedOps) {
    eraseOp(op);
  }
}

// Find all ttng::TCGen5CommitOp that could be theoritically
// fused together if the consumers are compatible.
SmallVector<ttng::TCGen5CommitOp>
collectCommitGroup(ttng::TCGen5CommitOp &commitOp,
                   DenseSet<ttng::TCGen5CommitOp> &seenCommits) {
  SmallVector<ttng::TCGen5CommitOp> commitGroup;
  auto block = commitOp->getBlock();
  auto startit = mlir::Block::iterator(commitOp);
  for (auto it = startit; it != block->end(); it++) {
    if (auto op = dyn_cast<ttng::TCGen5CommitOp>(*it)) {
      if (!seenCommits.count(op)) {
        seenCommits.insert(op);
        commitGroup.push_back(op);
      }
    } else {
      // We currently only support all ttng::TCGen5CommitOp
      // being grouped together.
      break;
    }
  }
  return commitGroup;
}

// Fuse together the barriers used by repeated
// tcgen05.commit operations. This works with the following
// setup:
// 1, Collect all tcgen05.commit operations that logically occur
// "concurrently" and especially without any intermediate mma ops.
// Right now we only support commit operations that are placed next
// to each other in the IR, but in theory this can be extended.
//
// 2. For each candidate group, group together barriers based on the
// underlying consumer(s). We will form a subgroup if the barrier:
//    a. Has no pipelining state. In the future this can be extended
//       to matching, but we don't want to worry about cluster reordering.
//    b. Has the same nesting level.
//    c. Has the same expected phase value.
//    d. Has the same expected arrival count (init count).
//
// 3. For each subgroup, update the barriers based on the consumer's location.
//    a. With the same async task id, eliminate all but the first barrier.
//    b. With different async task ids, use the same allocation.
//
// 4. Cleanup the code to remove the unused barriers.
//
// Note: This is run before warp specialization to simplify the
// transformation.
void fuseTcgen05CommitBarriers(tt::FuncOp &funcOp) {
  DenseSet<ttng::TCGen5CommitOp> seenCommits;
  SmallVector<SmallVector<ttng::TCGen5CommitOp>> commitGroups;
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](ttng::TCGen5CommitOp commitOp) {
    if (!seenCommits.count(commitOp)) {
      auto commitGroup = collectCommitGroup(commitOp, seenCommits);
      if (commitGroup.size() > 1) {
        commitGroups.push_back(commitGroup);
      }
    }
  });
  for (auto &commitGroup : commitGroups) {
    std::vector<CommitOpSubgroupInfo> subgroups;
    for (auto &commitOp : commitGroup) {
      auto barrier = commitOp.getBarrier();
      auto barrierAllocOp = barrier.getDefiningOp();
      // For each barrier that are 3 types of operations:
      // 1. Initializer: This should immediately follow the alloc.
      // 2. Producer: This should only be the tcgen05.commit op.
      // 3. Consumer: 1 or more ops.
      // We want to collect all of the consumers.
      SmallVector<Operation *> bufferConsumers;
      SmallVector<ttng::WaitBarrierOp> consumers;
      bool safe = true;
      int initCount = -1;
      for (auto user : barrier.getUsers()) {
        // We have found the consumer.
        if (user == commitOp) {
          continue;
        }
        // Track the operation for replacing buffers.
        Operation *bufferConsumer = user;
        // Find the actual barrier using op.
        if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(user)) {
          Operation *nextConsumer = nullptr;
          for (auto indexUser : indexOp->getUsers()) {
            if (nextConsumer) {
              safe = false;
              break;
            }
            nextConsumer = indexUser;
          }
          if (!nextConsumer) {
            safe = false;
          } else {
            user = nextConsumer;
          }
        }
        if (auto initBarrier = dyn_cast<ttng::InitBarrierOp>(user)) {
          if (initCount == -1) {
            initCount = initBarrier.getCount();
          } else {
            // Multiple inits. This is not safe.
            safe = false;
          }
        } else if (auto barrierOp = dyn_cast<ttng::WaitBarrierOp>(user)) {
          // We don't support pipelining state yet.
          if (barrierOp->hasAttr(tt::kLoopStageAttrName)) {
            safe = false;
          } else {
            consumers.push_back(barrierOp);
            bufferConsumers.push_back(bufferConsumer);
          }
        } else {
          // Unexpected barrier op.
          safe = false;
        }
        if (!safe) {
          break;
        }
      }
      // Cannot group this commit. Unsupport operations.
      if (!safe || initCount == -1) {
        continue;
      }
      mergeSubgroups(subgroups, initCount, barrierAllocOp, commitOp,
                     bufferConsumers, consumers);
    }
    for (auto &subgroup : subgroups) {
      updateSubgroup(subgroup);
    }
  }
}

} // namespace mlir
