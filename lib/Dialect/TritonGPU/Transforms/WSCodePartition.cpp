#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUWSCODEPARTITION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-warp-spec-code-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

std::pair<int, bool> scanRegUsage(Block *block, AsyncTaskId asyncTaskId,
                                  int regDecProducer, int regIncConsumer) {
  // TODO: scan ops to estimate register usage
  if (asyncTaskId == 0) {
    // deallocate registers
    return {regDecProducer == 0 ? 40 : regDecProducer, false};
  } else {
    // allocate registers
    return {regIncConsumer == 0 ? 232 : regIncConsumer, true};
  }
}

unsigned getNumBuffersOrDefault(scf::ForOp forOp, unsigned numBuffers) {
  // Use the attribute attached to the loop if it exists otherwise use the
  // global control.
  if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
    return numBuffers;
  return mlir::cast<IntegerAttr>(
             forOp->getAttr(mlir::triton::kNumStagesAttrName))
      .getInt();
}

// Collect argument indices that are used by the specific taskId.
static SmallVector<unsigned> collectBlockArgsForTask(
    scf::ForOp forOp, int asyncTaskId,
    DenseMap<BlockArgument, Value> &blockArgToYieldOperand) {

  // Collect argument indices that can be reached along the definition chain.
  SetVector<unsigned> argIndices;
  std::function<void(Value, unsigned)> dfs = [&](Value arg, unsigned argIdx) {
    for (auto user : arg.getUsers()) {
      // Skip ops that are not in the same async task
      if (!hasAsyncTaskId(user, asyncTaskId))
        continue;

      // Skip control flow ops that are shared by all async tasks
      if (isa<scf::YieldOp>(user))
        continue;

      // Found a real user, the arg is needed
      if (user->getNumRegions() == 0) {
        argIndices.insert(argIdx);
        return;
      }

      // Iterate through all regions of the user operation
      for (auto &region : user->getRegions()) {
        for (auto regionArg : region.getArguments()) {
          if (arg == regionArg)
            dfs(regionArg, argIdx);
        }
      }
    }
  };

  // check dependency with DFS traversal for loop args and results.
  mlir::Block &block = forOp.getRegion().front();
  for (unsigned i = forOp.getNumInductionVars(); i < block.getNumArguments();
       ++i) {
    auto arg = block.getArgument(i);
    dfs(arg, i - forOp.getNumInductionVars());
  }
  for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
    auto result = forOp->getResult(i);
    dfs(result, i);
  }

  SmallVector<unsigned> args(argIndices.begin(), argIndices.end());
  llvm::sort(args);
  return args;
}

Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilderWithAsyncTaskIds &builder,
                        AsyncTaskId asyncTaskId);

Operation *SpecializeIfOp(scf::IfOp ifOp, IRMapping &mapping,
                          OpBuilderWithAsyncTaskIds &builder,
                          AsyncTaskId asyncTaskId) {
  LLVM_DEBUG({
    LDBG("specialize ifOp ");
    ifOp.dump();
  });

  auto newIfOp = builder.createWithAsyncTaskIds<scf::IfOp>(
      ifOp.getLoc(), ifOp->getResultTypes(),
      mapping.lookup(ifOp.getCondition()), true, ifOp.elseBlock());

  OpBuilderWithAsyncTaskIds ifBuilder(ifOp.getContext());
  ifBuilder.setAsynTaskIdsFromArray({asyncTaskId});

  // Handle thenRegion of this IfOp.
  ifBuilder.setInsertionPointToEnd(newIfOp.thenBlock());
  for (Operation &thenOp : ifOp.thenBlock()->getOperations()) {
    SpecializeOp(&thenOp, mapping, ifBuilder, asyncTaskId);
  }

  // Handle elseRegion of the IfOp.
  if (ifOp.elseBlock()) {
    ifBuilder.setInsertionPointToEnd(newIfOp.elseBlock());
    for (Operation &elseOp : ifOp.elseBlock()->getOperations()) {
      SpecializeOp(&elseOp, mapping, ifBuilder, asyncTaskId);
    }
  }

  for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
    mapping.map(ifOp.getResult(i), newIfOp.getResult(i));
  return newIfOp;
}

Operation *SpecializeForOp(scf::ForOp forOp, IRMapping &mapping,
                           OpBuilderWithAsyncTaskIds &builder,
                           AsyncTaskId asyncTaskId) {
  // Prepare blockArgToYieldOperand mapping.
  DenseMap<BlockArgument, Value> blockArgToYieldOperand;
  auto yieldOp = llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  assert(yieldOp.getNumOperands() == forOp.getNumRegionIterArgs());
  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
    blockArgToYieldOperand[forOp.getRegionIterArg(i)] = yieldOp.getOperand(i);

  // Create newForOp for each task Id.
  auto usedArgs =
      collectBlockArgsForTask(forOp, asyncTaskId, blockArgToYieldOperand);

  // Prepare newLoopArgs.
  SmallVector<Value> newLoopArgs;
  for (unsigned argNumber : usedArgs) {
    auto arg = forOp.getInitArgs()[argNumber];
    auto newArg = mapping.lookupOrDefault(arg);
    assert(newArg && "Unexpected missing mapping");
    newLoopArgs.push_back(newArg);
  }

  // Prepare loop bounds.
  auto newLowerBound = mapping.lookupOrDefault(forOp.getLowerBound());
  auto newUpperBound = mapping.lookupOrDefault(forOp.getUpperBound());
  auto newStep = mapping.lookupOrDefault(forOp.getStep());

  // Create newForOp.
  auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
      forOp.getLoc(), newLowerBound, newUpperBound, newStep, newLoopArgs);
  if (forOp->getAttr("tt.loop_schedule"))
    newForOp->setAttr("tt.loop_schedule", forOp->getAttr("tt.loop_schedule"));

  // Initialize Value mapping from forOp to newForOp
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < usedArgs.size(); ++i) {
    auto oldArg = forOp.getRegionIterArgs()[usedArgs[i]];
    auto newArg = newForOp.getRegionIterArgs()[i];
    mapping.map(oldArg, newArg);
  }

  // Recursively clone all operations with this asyncTaskId to newForOp.
  OpBuilderWithAsyncTaskIds forBuilder(forOp.getContext());
  forBuilder.setAsynTaskIdsFromArray({asyncTaskId});
  forBuilder.setInsertionPointToStart(newForOp.getBody());
  for (Operation &op : forOp.getBody()->without_terminator()) {
    SpecializeOp(&op, mapping, forBuilder, asyncTaskId);
  }

  // Create YieldOp for newForOp.
  SmallVector<Value> newYieldOperands;
  for (unsigned i : usedArgs)
    newYieldOperands.push_back(mapping.lookup(yieldOp.getOperand(i)));

  bool createNewYield = true;
  if (newForOp.getBody()->mightHaveTerminator()) {
    auto initialYield =
        llvm::cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    if (newYieldOperands.size() == 0) {
      setAsyncTaskIds(initialYield, {asyncTaskId});
      createNewYield = false;
    }
  }
  if (createNewYield) {
    auto newYieldOp =
        forBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    setAsyncTaskIds(newYieldOp, {asyncTaskId});
  }

  // Replace results of forOp with results of newForOp.
  for (unsigned i = 0; i < usedArgs.size(); ++i) {
    auto oldResult = forOp.getResult(usedArgs[i]);
    auto newResult = newForOp.getResult(i);
    mapping.map(oldResult, newResult);
  }

  return newForOp;
}

Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilderWithAsyncTaskIds &builder,
                        AsyncTaskId asyncTaskId) {
  auto taskIds = getAsyncTaskIds(op);
  // yieldOp are sometimes implict, meaning they do not necessarily have a task
  // id, but they should be shared by all async tasks.
  if (!hasAsyncTaskId(op, asyncTaskId) && !isa<scf::YieldOp>(op))
    return nullptr;

  if (op->getNumRegions() == 0) {
    Operation *newOp = builder.clone(*op, mapping);
    setAsyncTaskIds(newOp, asyncTaskId);
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      mapping.map(op->getResult(i), newOp->getResult(i));
    return newOp;
  } else {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return SpecializeIfOp(ifOp, mapping, builder, asyncTaskId);
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return SpecializeForOp(forOp, mapping, builder, asyncTaskId);
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      Operation *newOp = builder.clone(*op, mapping);
      // recursively set async task ids for child ops
      newOp->walk(
          [&](Operation *childOp) { setAsyncTaskIds(childOp, asyncTaskId); });
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
      return newOp;
    } else {
      llvm_unreachable("Unexpected Op with regions");
    }
  }

  return nullptr;
}

// Create IfOp for each ayncTaskId.
DenseMap<AsyncTaskId, scf::IfOp> SpecializeRegion(triton::FuncOp funcOp,
                                                  int regDecProducer,
                                                  int regIncConsumer) {

  LLVM_DEBUG({
    LDBG("\n\n");
    LDBG("Start specializing region");
  });

  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  auto loc = funcOp.getLoc();

  // Collect original operations
  SmallVector<Operation *> opList;
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &op : block.getOperations()) {
      auto taskIds = getAsyncTaskIds(&op);
      if (!taskIds.empty())
        opList.push_back(&op);
    }
  }

  LLVM_DEBUG({
    LDBG("ops to be specialized: ");
    for (Operation *op : opList) {
      op->dump();
    }
  });

  // Create GetAsyncTaskIdOp.
  Block *lastBlock = &funcOp.getBody().back();
  auto returnOp = llvm::cast<triton::ReturnOp>(lastBlock->getTerminator());
  builder.setInsertionPoint(returnOp);
  Value curAsyncTaskId = builder.create<ttng::GetAsyncTaskIdOp>(loc);

  DenseMap<AsyncTaskId, scf::IfOp> tasksToIfOp;

  // Clone all operations into the corresponding if blocks. If the operation
  // has multiple taskIds, it will be cloned for multiple if blocks.
  // If the original code has an IfOp, we should only clone its
  // body with the right asyncTaskId, instead of cloning the IfOp.
  for (AsyncTaskId asyncTaskId : getNestedAsyncTaskIds(funcOp)) {
    // Create IfOp for each asyncTaskId.
    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, curAsyncTaskId,
        builder.create<arith::ConstantIntOp>(loc, asyncTaskId, 32));

    auto ifOp = builder.create<scf::IfOp>(loc, cond);
    tasksToIfOp[asyncTaskId] = ifOp;
    setAsyncTaskIds(ifOp, {asyncTaskId});

    OpBuilderWithAsyncTaskIds taskBuilder(context);
    taskBuilder.setAsynTaskIdsFromArray({asyncTaskId});

    // Set insertion point before yieldOp.
    auto yieldOp = ifOp.thenYield();
    setAsyncTaskIds(yieldOp, {asyncTaskId});
    taskBuilder.setInsertionPoint(yieldOp);

    IRMapping mapping;
    for (Operation *op : opList) {
      SpecializeOp(op, mapping, taskBuilder, asyncTaskId);
    }
  }

  // Decide if this taskId is a producer or a consumer, and create either
  // RegAllocOp or RegDeallocOp accordingly.
  for (auto ifOps : tasksToIfOp) {
    AsyncTaskId asyncTaskId = ifOps.first;
    auto ifOp = ifOps.second;
    OpBuilderWithAsyncTaskIds taskBuilder(ifOp.getContext());
    taskBuilder.setAsynTaskIdsFromArray({asyncTaskId});
    auto regAlloc = scanRegUsage(ifOp.thenBlock(), asyncTaskId, regDecProducer,
                                 regIncConsumer);
    taskBuilder.setInsertionPointToStart(&(ifOp.getThenRegion().front()));
    if (regAlloc.second)
      taskBuilder.create<ttng::RegAllocOp>(
          loc, taskBuilder.getI32IntegerAttr(regAlloc.first));
    else
      taskBuilder.create<ttng::RegDeallocOp>(
          loc, taskBuilder.getI32IntegerAttr(regAlloc.first));
  }

  LLVM_DEBUG({
    LDBG("\n\nWith task Id checks");
    funcOp.dump();
  });

  // Remove original operations that have been cloned in reverse order.
  for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
    Operation *op = *it;
    LLVM_DEBUG({
      LDBG("erasing op ");
      op->dump();
    });
    // For debugging purposes, check to see if the original op is still in use.
    bool hasUse = false;
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      for (Operation *user : op->getResult(i).getUsers()) {
        hasUse = true;
        LLVM_DEBUG({
          LDBG("op has use ");
          user->dump();
        });
      }
    }
    op->erase();
  }
  return tasksToIfOp;
}

struct Channel {
public:
  using Relation = std::pair<int, SmallVector<int>>;

  Channel(int producer, SmallVector<int> &consumers, Operation *op,
          unsigned operandIdx, unsigned numBuffers)
      : relation(producer, consumers), op(op), operandIdx(operandIdx),
        numBuffers(numBuffers) {}

  bool operator==(const Channel &c) {
    return relation == c.relation && operandIdx == c.operandIdx && op == c.op;
  }

  Operation *getDstOp() { return op; }
  unsigned getDstOperandIdx() { return operandIdx; }
  Value getSrcOperand() { return op->getOperand(operandIdx); }
  Operation *getSrcOp() { return getSrcOperand().getDefiningOp(); }

  Relation relation; // producer task Id, a list of consumer task Ids
  Operation *op;
  unsigned operandIdx;
  unsigned numBuffers;
};

// Find transitive users of the root op. Ignore control flow ops such as yield
// in between.
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

// Value findTransitiveProducer(Value root) {
//   SetVector<Operation *> users;
//   getTransitiveUsers(root, users);
//   assert(!users.empty() && "producer op has no consumers");
//   if (users.size() == 1) {
//     return root;
//   }

//   // Reconsile the user list. If there are multiple users, we need to
//   // find a global producer in the same scope with the outtermost-scoped
//   user. Region *currentScope = root.getDefiningOp()->getParentRegion();
//   Region *userScope = nullptr;

//   // check all users have same scope
//   for (Operation *user : users) {
//     auto currentScope = user->getParentRegion();
//     if (userScope)

//       auto parentRegions = getAllParentRegions(user);
//     if (parentRegions.count(currentScope) == 0) {
//       llvm::errs() << "user is not in the same scope as the producer\n";
//       llvm_unreachable("user is not in the same scope as the producer");
//     }

//     if (userScope->getParentOfType<scf::ForOp>() == nullptr) {
//       newScope = userScope;
//     }
//   }

//   auto getAllParentRegions = [](Operation *op) {
//     SetVector<Region *> parentRegions;
//     Region *region = op->getParentRegion();
//     while (region) {
//       parentRegions.insert(region);
//       mlir::Operation *parentOp = region->getParentOp();
//       if (!parentOp)
//         break;
//       region = parentOp->getParentRegion();
//     }

//     return parentRegions;
//   };

//   // Find the outtermost-scoped user.
//   for (Operation *user : users) {
//     auto parentRegions = getAllParentRegions(user);
//     if (parentRegions.count(currentScope) == 0) {
//       llvm::errs() << "user is not in the same scope as the producer\n";
//       llvm_unreachable("user is not in the same scope as the producer");
//     }

//     if (userScope->getParentOfType<scf::ForOp>() == nullptr) {
//       newScope = userScope;
//     }
//   }

//   // Find the global producer in the same scope with the outtermost-scoped
//   user.

//   // Promote producer to the same level as the consumer.

//   return root;
// }

// Loads will be in producer warp groups. For now, we only allow a single
// warp group/task for a producer. For each LoadOp, create a channel from it
// to any direct user which belongs to a different taskId.
void collectAsyncChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                          triton::FuncOp &funcOp, unsigned numBuffers) {
  funcOp.walk([&](Operation *op) {
    if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op) ||
        op->hasTrait<OpTrait::DotLike>()) {
      auto producerTaskIds = getAsyncTaskIds(op);
      if (producerTaskIds.empty() || producerTaskIds.size() > 1) {
        LLVM_DEBUG({
          LDBG(" ignoring load ops without async task id or with multiple task "
               "ids: ");
          op->dump();
        });
        return;
      }
      auto producerTaskId = producerTaskIds.front();
      unsigned producerNumBuffers = numBuffers;
      if (auto forOp = op->getParentOfType<scf::ForOp>()) {
        producerNumBuffers = getNumBuffersOrDefault(forOp, numBuffers);
      }

      for (auto result : op->getResults()) {
        if (result.use_empty()) {
          continue;
        }

        SetVector<std::pair<Operation *, unsigned>> users;
        getTransitiveUsers(result, users);
        for (auto pc : users) {
          auto userOp = pc.first;
          auto consumerTaskIds = getAsyncTaskIds(userOp);
          if (consumerTaskIds.empty())
            continue;
          // Remove producer task id from consumerTaskIds.
          auto iter = std::remove(consumerTaskIds.begin(),
                                  consumerTaskIds.end(), producerTaskId);
          consumerTaskIds.erase(iter, consumerTaskIds.end());
          // Add a channel from the single producer task to consumerTaskIds.
          if (consumerTaskIds.size() > 0) {
            channels.push_back(std::make_unique<Channel>(
                producerTaskId, consumerTaskIds, userOp, pc.second,
                producerNumBuffers));
          }
        }
      }
    }
  });

  LLVM_DEBUG({
    LDBG("Async channels:");
    for (auto &channel : channels) {
      LDBG("producer op: " << channel->relation.first);
      channel->getSrcOp()->dump();
      for (auto &asyncTaskId : channel->relation.second)
        LDBG("consumer: " << asyncTaskId);
      channel->getDstOp()->dump();
      LDBG("numBuffers: " << channel->numBuffers);
    }
  });
}

// Update map, which will be keyed by getDstOp() of the channel. Use mapKeyVec
// to enforce deterministic order for map.
void groupChannels(
    SmallVector<Channel *> &channels,
    DenseMap<Operation *, SmallVector<Channel *>> &groupedChannels,
    SmallVector<Operation *> &mapKeyVec) {
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
    return dst1->getBlock() == dst2->getBlock();
  };
  assert(channels.size() > 0 && "channel size is zero");
  // Compare with existing channels in the groupedChannels to see if it can be
  // combined.
  for (auto *c0 : channels) {
    bool merged = false;
    for (auto &kv : groupedChannels) {
      if (kv.second.size() > 0 && channelCanBeMerged(c0, kv.second.front())) {
        kv.second.push_back(c0);
        merged = true;
        break;
      }
    }
    if (!merged) { // Create a new entry.
      auto *keyOp = c0->getDstOp();
      if (!groupedChannels.count(keyOp))
        mapKeyVec.push_back(keyOp);
      groupedChannels[keyOp].push_back(c0);
    }
  }

  // Reorder channels associated with one entry based on program order of the
  // producers.
  for (auto &kv : groupedChannels) {
    if (kv.second.size() > 1) {
      auto &allOps = kv.second.front()->getSrcOp()->getBlock()->getOperations();
      std::sort(
          kv.second.begin(), kv.second.end(), [&](Channel *a, Channel *b) {
            auto itrA =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == a->getSrcOp();
                });
            auto itrB =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == b->getSrcOp();
                });
            assert(itrA != allOps.end() && itrB != allOps.end());
            return std::distance(itrA, itrB) < 0;
          });
    }
  }

  LLVM_DEBUG({
    LDBG("\n\nGrouped channels:");
    for (auto &kv : groupedChannels) {
      DBGS() << "consumer op:  ";
      kv.getFirst()->dump();
      for (auto &channel : kv.second) {
        DBGS() << "producer: ";
        channel->getSrcOp()->dump();
        for (auto &asyncTaskId : channel->relation.second)
          DBGS() << asyncTaskId << ", ";
        DBGS() << "] ";
        LDBG("numBuffers: " << channel->numBuffers);
      }
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
      getBackwardSlice(channel->getSrcOp(), &backwardSlice, opt);
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

bool isInnermostLoop(scf::ForOp forOp) {
  for (Operation &nestedOp : forOp.getBody()->getOperations()) {
    if (isa<scf::ForOp>(nestedOp)) {
      return false;
    }
  }
  return true;
}

// Add phase and bufferIndex to be used when lowering the producer.
scf::ForOp createNewLoop(scf::ForOp forOp, int numBuffers,
                         scf::ForOp &parentForOp) {
  auto loc = forOp.getLoc();
  Block *body = forOp.getBody();

  OpBuilderWithAsyncTaskIds builder(forOp.getContext());
  builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(forOp));
  builder.setInsertionPoint(forOp);

  Value numBuffersVal =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, numBuffers, 32);

  // Step 1: Append bufferIdx and phase as forOp arguments.
  Value phase =
      body->insertArgument(body->getNumArguments(), builder.getI1Type(), loc);
  Value bufferIdx =
      body->insertArgument(body->getNumArguments(), builder.getI32Type(), loc);

  // Step 2: Generate bufferIdx and phase for next iteration:
  //   nextBufferIdx = bufferIdx + 1
  //   nextPhase = ((nextBufferIdx < numBuffers && curPhase) ||
  //                (nextBufferIdx >= numBuffers && curPhase^1))
  //   nextBufferIdx = nextBufferIdx >= numBuffers ? 0 : nextBufferIdx
  auto yieldOp = llvm::cast<scf::YieldOp>(body->getTerminator());
  builder.setInsertionPoint(yieldOp);
  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 32);
  Value _1_1b = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  //   nextBufferIdx = bufferIdx + 1
  Value nextBufferIdx =
      builder.createWithAsyncTaskIds<arith::AddIOp>(loc, bufferIdx, one);
  Value bufferGECond = builder.createWithAsyncTaskIds<arith::CmpIOp>(
      loc, arith::CmpIPredicate::uge, nextBufferIdx, numBuffersVal);
  Value bufferLTCond = builder.createWithAsyncTaskIds<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, nextBufferIdx, numBuffersVal);
  // nextBufferIdx >= numBuffers ? nextBufferIdx - numBuffers : nextBufferIdx
  Value moduloBufferIdx = builder.createWithAsyncTaskIds<arith::SubIOp>(
      loc, nextBufferIdx, numBuffersVal);
  nextBufferIdx = builder.createWithAsyncTaskIds<mlir::arith::SelectOp>(
      loc, bufferGECond, moduloBufferIdx, nextBufferIdx);

  // nextPhase = ((nextBufferIdx < numBuffers && curPhase) ||
  //              (nextBufferIdx >= numBuffers && curPhase^1))
  Value flipPhase =
      builder.createWithAsyncTaskIds<mlir::arith::XOrIOp>(loc, phase, _1_1b);
  Value cond0 = builder.createWithAsyncTaskIds<mlir::arith::AndIOp>(
      loc, bufferGECond, flipPhase);
  Value cond1 = builder.createWithAsyncTaskIds<mlir::arith::AndIOp>(
      loc, bufferLTCond, phase);
  Value nextPhase =
      builder.createWithAsyncTaskIds<mlir::arith::OrIOp>(loc, cond0, cond1);

  // Step 3: Add nextBufferIdx and nextPhase to yieldOp.
  yieldOp->insertOperands(yieldOp.getNumOperands(), {nextPhase, nextBufferIdx});

  // Step 4: Create loop arguments for the new ForOp.
  SmallVector<Value> newLoopArgs;
  for (auto operand : forOp.getInitArgs())
    newLoopArgs.push_back(operand);

  builder.setInsertionPoint(forOp);
  Value initBufferIdx, initPhase;
  // Set initial values for bufferIdx and phase.
  if (parentForOp) {
    // It is possible that parent loop induction variable has different type.
    // Here we promote to 64 bit.
    // numSteps = ((upperBound - lowerBound) + forOpStep - 1) / forOpStep
    Value numSteps = builder.createWithAsyncTaskIds<arith::SubIOp>(
        loc, forOp.getUpperBound(), forOp.getLowerBound());
    numSteps = builder.createWithAsyncTaskIds<arith::AddIOp>(loc, numSteps,
                                                             forOp.getStep());
    if (forOp.getStep().getType() != builder.getI64Type())
      numSteps = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
          loc, builder.getI64Type(), numSteps);

    Value one =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
    numSteps =
        builder.createWithAsyncTaskIds<arith::SubIOp>(loc, numSteps, one);
    Value innerForStep = forOp.getStep();
    if (forOp.getStep().getType() != builder.getI64Type())
      innerForStep = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
          loc, builder.getI64Type(), forOp.getStep());
    numSteps = builder.createWithAsyncTaskIds<arith::DivUIOp>(loc, numSteps,
                                                              innerForStep);

    // TODO: use a global flattened iteration space index for multi-dim loops.
    // tmpIdx = parentForOp.iterationIdx * numSteps
    // initBufferIdx = tmpIdx - tmpIdx / numBuffers * numBuffers
    // initPhase = (tmpIdx / numBuffers) & 1
    // Assume parent ForOp has bufferIdx as the last argument.
    Value parentIterIdx = builder.createWithAsyncTaskIds<arith::SubIOp>(
        loc, parentForOp.getInductionVar(), parentForOp.getLowerBound());
    parentIterIdx = builder.createWithAsyncTaskIds<arith::DivUIOp>(
        loc, parentIterIdx, parentForOp.getStep());
    if (parentForOp.getStep().getType() != builder.getI64Type())
      parentIterIdx = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
          loc, builder.getI64Type(), parentIterIdx);

    initBufferIdx = builder.createWithAsyncTaskIds<arith::MulIOp>(
        loc, parentIterIdx, numSteps);
    numBuffersVal = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
        loc, builder.getI64Type(), numBuffersVal);
    Value bufferIdx = builder.createWithAsyncTaskIds<arith::DivUIOp>(
        loc, initBufferIdx, numBuffersVal);
    initBufferIdx = builder.createWithAsyncTaskIds<arith::SubIOp>(
        loc, initBufferIdx,
        builder.createWithAsyncTaskIds<arith::MulIOp>(loc, bufferIdx,
                                                      numBuffersVal));
    initBufferIdx = builder.createWithAsyncTaskIds<arith::TruncIOp>(
        loc, builder.getI32Type(), initBufferIdx);

    bufferIdx =
        builder.createWithAsyncTaskIds<arith::AndIOp>(loc, bufferIdx, one);
    initPhase = builder.createWithAsyncTaskIds<arith::TruncIOp>(
        loc, builder.getI1Type(), bufferIdx);
  } else {
    // Set initial phase to false, and initial bufferIdx to 0.
    initBufferIdx =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 32);
    initPhase = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 1);
  }
  newLoopArgs.append({initPhase, initBufferIdx});

  // Step 5: Create newForOp and take the region of the original forOp.
  auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      newLoopArgs);
  if (forOp->getAttr("tt.loop_schedule"))
    newForOp->setAttr("tt.loop_schedule", forOp->getAttr("tt.loop_schedule"));
  newForOp.getRegion().takeBody(forOp.getRegion());

  // Step 6: Replace forOp with newForOp.
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();

  return newForOp;
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

// For ForOps in taskTopOps, create new ForOp for each by adding phase,
// bufferIdx to the arguments.
void appendBufferIdxArgs(SmallVector<Operation *> &taskTopOps,
                         unsigned numBuffers) {
  SmallVector<scf::ForOp> orderedForOps;
  for (auto &op : taskTopOps) {
    op->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
      if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
        orderedForOps.push_back(forOp);
      }
    });
  }

  for (auto &origForOp : orderedForOps) {
    scf::ForOp parentForOp = origForOp->getParentOfType<scf::ForOp>();
    scf::ForOp newForOp;
    // for(...) -> for(..., phase, bufferIdx)
    unsigned loopNumBuffers = getNumBuffersOrDefault(origForOp, numBuffers);
    newForOp = createNewLoop(origForOp, loopNumBuffers, parentForOp);
    // origForOp is erased in createNewLoop. If origForOp is a top operation
    // (i.e in taskTopOps), make sure taskTopOps is updated with the newForOp.
    auto asyncTaskLoopForItr = std::find(taskTopOps.begin(), taskTopOps.end(),
                                         origForOp.getOperation());
    if (asyncTaskLoopForItr != taskTopOps.end()) {
      // Update taskTopOps.
      *asyncTaskLoopForItr = newForOp.getOperation();
    }
  }

  LLVM_DEBUG({
    LDBG("\n\nafter appendBufferIdxArgs");
    taskTopOps[0]->getParentOfType<triton::FuncOp>().dump();
  });
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
  auto barrierEncoding =
      ttg::SharedEncodingAttr::get(context, 1, 1, 1, {0}, barrierCTALayout);
  Type barrierMemDescType = tt::MemDescType::get(
      {distance}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Type singleBarrierMemDescType =
      tt::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                           sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loc, barrierMemDescType, Value());
  for (unsigned i = 0; i < distance; i++) {
    Value idx = builder.create<arith::ConstantIntOp>(loc, i, 32);
    Value barrierView = builder.create<ttg::MemDescSubviewOp>(
        loc, singleBarrierMemDescType, barrierAlloc, idx);
    builder.create<ttng::InitBarrierOp>(funcOp->getLoc(), barrierView, 1);
  }
  return barrierAlloc;
}

// groupedChannels: channels are grouped together.
// Go through each group, check the first channel in the group, create a token
// for each consumer taskId. Return a map that maps each channel + consumer
// taskId to a token. Also update barrierAllocMap that maps each channel +
// consumer taskId to a BarrierAlloc.
DenseMap<Channel *, DenseMap<int, Value>> createToken(
    const DenseMap<Operation *, SmallVector<Channel *>> &groupedChannels,
    const SmallVector<Operation *> &mapKeyVec, triton::FuncOp funcOp,
    int numConsumerGroups,
    DenseMap<Channel *, DenseMap<int, Value>> &barrierAllocMap) {
  DenseMap<Channel *, DenseMap<int, Value>> ret;
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  for (auto *key : mapKeyVec) {
    auto it = groupedChannels.find(key);
    Channel *channel = it->second.front();
    for (auto consumerAsyncTaskId : channel->relation.second) {
      Value v;
      if (it->second.front()->getSrcOp()->getParentOfType<scf::ForOp>()) {
        v = builder.create<ttng::CreateTokenOp>(funcOp.getLoc(),
                                                channel->numBuffers);
      } else {
        v = builder.create<ttng::CreateTokenOp>(funcOp.getLoc(), 1);
      }
      // Channels in the group share the same set of tokens.
      for (auto &c : it->second)
        ret[c][consumerAsyncTaskId] = v;

      auto producerOp = it->second.front()->getSrcOp();
      if (isa<tt::ExperimentalDescriptorLoadOp>(producerOp)) {
        Value bAlloc = createBarrierAlloc(funcOp, channel->numBuffers);
        // Channels in the group share the same set of tokens.
        for (auto &c : it->second) {
          ret[c][consumerAsyncTaskId] = v;
          barrierAllocMap[c][consumerAsyncTaskId] = bAlloc;
        }
      }
    }
  }
  return ret;
}

// Create a buffer array for each producer op, if the producer is in a ForOp,
// the buffer array will contain numBuffers.
DenseMap<Channel *, Value> createBuffer(const SmallVector<Channel *> &channels,
                                        triton::FuncOp funcOp,
                                        int numConsumerGroups) {

  // Group channels by producer op.
  DenseMap<Value, SmallVector<Channel *>> groupedChannels;
  for (auto channel : channels) {
    groupedChannels[channel->getSrcOperand()].push_back(channel);
  }

#ifndef NDEBUG
  // Some sanity checks.
  for (auto &item : groupedChannels) {
    auto &channels = item.second;
    unsigned numBuffers = channels.front()->numBuffers;
    for (auto c : channels) {
      assert(c->numBuffers == numBuffers && "Unmatched number of buffers");
    }
  }
#endif

  DenseMap<Channel *, Value> bufferMap;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(funcOp);
  for (auto &item : groupedChannels) {
    auto &channels = item.second;
    auto srcValue = item.first;
    auto srcOp = srcValue.getDefiningOp();
    unsigned numBuffers = channels.front()->numBuffers;

    if (auto tensorType = dyn_cast<RankedTensorType>(srcValue.getType())) {
      // Get basic information from tensorType
      auto order = ttg::getOrder(tensorType.getEncoding());
      auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
      auto elemType = tensorType.getElementType();

      // Get shape, layout and type of a slice
      auto sliceShape = tensorType.getShape();
      auto sharedLayout = ttg::SharedEncodingAttr::get(
          context, sliceShape, order, CTALayout, elemType);
      auto sliceType =
          RankedTensorType::get(sliceShape, elemType, sharedLayout);

      // Get shape, layout and type of the complete buffer
      SmallVector<int64_t> bufferShape(sliceShape.begin(), sliceShape.end());
      if (srcOp->getParentOfType<scf::ForOp>())
        bufferShape.insert(bufferShape.begin(), numBuffers);
      else
        bufferShape.insert(bufferShape.begin(), 1);
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(context);
      auto bufferType =
          RankedTensorType::get(bufferShape, elemType, sharedLayout);
      Type memdescType =
          tt::MemDescType::get(bufferShape, elemType, sharedLayout,
                               sharedMemorySpace, /*mutableMemory*/ true);
      builder.setInsertionPointToStart(&(funcOp.getBody().front()));
      Value buffer =
          builder.create<ttg::LocalAllocOp>(funcOp.getLoc(), memdescType);

      // Channels in the group share the same buffer.
      for (auto c : channels)
        bufferMap[c] = buffer;
    } else {
      llvm_unreachable("Unexpected result type");
    }
  }
  return bufferMap;
}

static Operation *createAsyncCopy(const DenseMap<Channel *, Value> &bufferMap,
                                  Channel *c, Operation *op,
                                  SmallVector<AsyncTaskId> &asyncTasksPC,
                                  Value bufferIdx, Value bufferIdxExtract) {
  auto loadOp = cast<triton::LoadOp>(op);
  auto buffer = bufferMap.find(c)->second;
  MLIRContext *context = loadOp->getContext();
  OpBuilderWithAsyncTaskIds builder(context);
  builder.setInsertionPoint(loadOp->getParentOp());
  builder.setAsynTaskIdsFromArray(asyncTasksPC);

  builder.setInsertionPoint(loadOp);
  Value loadResult = loadOp.getResult();
  auto tensorType = dyn_cast<RankedTensorType>(loadResult.getType());
  if (!tensorType)
    return nullptr;
  // Get basic information from tensorType
  auto order = ttg::getOrder(tensorType.getEncoding());
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout = ttg::SharedEncodingAttr::get(context, sliceShape, order,
                                                   CTALayout, elemType);
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  tt::MemDescType subviewTy =
      tt::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                           sliceType.getEncoding(), sharedMemorySpace,
                           /*mutableMemory=*/true);
  Value zero = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
      loadOp.getLoc(), 0, 32);
  SmallVector<Value> copyOffsets(sliceType.getRank() + 1, zero);
  copyOffsets[0] = bufferIdx;
  builder.setAsyncTaskIdsFromOp(loadOp);
  builder.setInsertionPointAfter(loadOp);
  auto view = builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      loadOp.getLoc(), subviewTy, buffer, copyOffsets);
  // Create cp.async
  Operation *copy =
      builder.createWithAsyncTaskIds<ttg::AsyncCopyGlobalToLocalOp>(
          loadOp.getLoc(), loadOp.getPtr(), view, loadOp.getMask(),
          loadOp.getOther(), loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile());

  // Extract part.
  builder.setAsyncTaskIdsFromValueUsers(loadResult);
  builder.setInsertionPoint(c->getDstOp());
  SmallVector<Value> loadOffsets(sliceType.getRank() + 1, zero);
  loadOffsets[0] = bufferIdxExtract;
  auto viewLoad = builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      loadOp.getLoc(), subviewTy, buffer, loadOffsets);
  auto sharedLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
      loadOp.getLoc(), loadOp.getType(), viewLoad /*,wait->getResult(0)*/);
  // Replace all uses of loadResult
  // TODO: replace the real consumer
  loadResult.replaceAllUsesWith(sharedLoad.getResult());
  loadOp.erase();
  return copy;
}

static Operation *createLocalCopy(const DenseMap<Channel *, Value> &bufferMap,
                                  Channel *channel, Value srcBufferIdx,
                                  Value dstBufferIdx) {
  Operation *srcOp = channel->getSrcOp();
  Operation *dstOp = channel->getDstOp();
  MLIRContext *context = srcOp->getContext();
  auto buffer = bufferMap.find(channel)->second;

  Value srcValue = channel->getSrcOperand();
  auto tensorType = dyn_cast<RankedTensorType>(srcValue.getType());
  if (!tensorType)
    return nullptr;
  // Get basic information from tensorType
  auto order = ttg::getOrder(tensorType.getEncoding());
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout = ttg::SharedEncodingAttr::get(context, sliceShape, order,
                                                   CTALayout, elemType);
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  tt::MemDescType subviewTy =
      tt::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                           sliceType.getEncoding(), sharedMemorySpace,
                           /*mutableMemory=*/true);

  // Consumer part.
  OpBuilderWithAsyncTaskIds dstBuilder(dstOp);
  dstBuilder.setInsertionPoint(dstOp);
  Value zero = dstBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(
      srcOp->getLoc(), 0, 32);
  SmallVector<Value> loadOffsets(sliceType.getRank() + 1, zero);
  loadOffsets[0] = dstBufferIdx;
  auto dstView = dstBuilder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      srcOp->getLoc(), subviewTy, buffer, loadOffsets);
  auto sharedLoad = dstBuilder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
      srcOp->getLoc(), srcValue.getType(), dstView);
  srcValue.replaceAllUsesWith(sharedLoad.getResult());

  // Producer part.
  OpBuilderWithAsyncTaskIds srcBuilder(srcOp);
  srcBuilder.setInsertionPoint(srcOp->getParentOp());
  zero = srcBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(
      srcOp->getLoc(), 0, 32);
  SmallVector<Value> storeOffsets(sliceType.getRank() + 1, zero);
  storeOffsets[0] = srcBufferIdx;
  srcBuilder.setInsertionPointAfter(srcOp);
  auto srcView = srcBuilder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      srcOp->getLoc(), subviewTy, buffer, storeOffsets);
  // Create local_alloc
  Operation *copy = srcBuilder.createWithAsyncTaskIds<ttg::LocalStoreOp>(
      srcOp->getLoc(), srcValue, srcView);

  return copy;
}

static int getTMALoadSize(tt::ExperimentalDescriptorLoadOp &tmaLoad) {
  auto tensorTy = cast<RankedTensorType>(tmaLoad->getResult(0).getType());
  int loadSize = product(tensorTy.getShape());
  return loadSize * tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
}

Value getBarrierForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                 Value barrierAlloc, Value bufferIdx) {
  auto context = barrierAlloc.getContext();
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  tt::MemDescType barrierTy = tt::MemDescType::get(
      {1}, builder.getI64Type(),
      cast<tt::MemDescType>(barrierAlloc.getType()).getEncoding(),
      sharedMemorySpace,
      /*mutableMemory=*/true);

  // Create barrierForTMA from barrierAlloc.
  return builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      barrierAlloc.getLoc(), barrierTy, barrierAlloc,
      ArrayRef<Value>({bufferIdx}));
}

Value getBufferForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                Type loadType, Value buffer, Value bufferIdx,
                                bool mutableMem) {
  auto context = buffer.getContext();
  auto tensorType = dyn_cast<RankedTensorType>(loadType);
  assert(tensorType);

  auto order = ttg::getOrder(tensorType.getEncoding());
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout = ttg::SharedEncodingAttr::get(context, sliceShape, order,
                                                   CTALayout, elemType);
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  tt::MemDescType subviewTy =
      tt::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                           sliceType.getEncoding(), sharedMemorySpace,
                           /*mutableMemOry=*/mutableMem);

  Value zero = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
      buffer.getLoc(), 0, 32);
  SmallVector<Value> copyOffsets(sliceType.getRank() + 1, zero);
  copyOffsets[0] = bufferIdx;

  return builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      buffer.getLoc(), subviewTy, buffer, copyOffsets);
}

Operation *
optimizeTMALoads(OpBuilderWithAsyncTaskIds &builder,
                 SmallVector<tt::ExperimentalDescriptorLoadOp> &tmaLoads,
                 SmallVector<Value> &buffers, Value barrierAlloc,
                 Value bufferIdx, Value bufferIdxExtract, Value phase,
                 Operation *headProducer, Operation *headConsumer) {
  auto loc = barrierAlloc.getLoc();

  // Compute the total size of the loads.
  int sizeInBytes = 0;
  for (auto &tmaLoad : tmaLoads) {
    sizeInBytes += getTMALoadSize(tmaLoad);
  }

  // For each of the following ops, we will operate on a subview of each value
  // according to the pipeline stage.

  // Create a barrier_expect with the appropriate size and insert it before the
  // first load.
  builder.setInsertionPoint(headProducer);
  builder.setAsyncTaskIdsFromOp(headProducer);
  auto prodBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  auto pred = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  auto expect = builder.createWithAsyncTaskIds<ttng::BarrierExpectOp>(
      loc, prodBarrier, sizeInBytes, pred);

  // Convert all the producers to async_tma_copy_global_to_local
  Operation *copy = nullptr;
  for (auto [tmaLoad, buffer] : zip(tmaLoads, buffers)) {
    auto pipelineBuffer = getBufferForPipelineStage(builder, tmaLoad.getType(),
                                                    buffer, bufferIdx, true);
    copy = builder.createWithAsyncTaskIds<ttng::AsyncTMACopyGlobalToLocalOp>(
        loc, tmaLoad.getDescPtr(), tmaLoad.getIndices(), prodBarrier,
        pipelineBuffer, pred);
  }

  // Create a wait_barrier before the first consumer.
  builder.setInsertionPoint(headConsumer);
  builder.setAsyncTaskIdsFromOp(headConsumer);
  auto consBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdxExtract);
  phase = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
      loc, builder.getI32Type(), phase);
  auto wait = builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
      loc, consBarrier, phase);

  // Convert all the consumers to local_load
  for (auto [tmaLoad, buffer] : zip(tmaLoads, buffers)) {
    auto pipelineBuffer = getBufferForPipelineStage(
        builder, tmaLoad.getType(), buffer, bufferIdxExtract, false);
    auto sharedLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
        loc, tmaLoad.getType(), pipelineBuffer);

    Value loadResult = tmaLoad.getResult();
    tmaLoad.getResult().replaceAllUsesWith(sharedLoad.getResult());
    tmaLoad.erase();
  }
  return copy;
}

// Lower producers for channels. Here channels are grouped in "groupedChannels".
// tokenMap tracks the set of tokens for each channel.
void buildAsyncComm(
    const DenseMap<Operation *, SmallVector<Channel *>> &groupedChannels,
    const DenseMap<Channel *, DenseMap<int, Value>> &tokenMap,
    const DenseMap<Channel *, DenseMap<int, Value>> &barrierAllocMap,
    const DenseMap<Channel *, Value> &bufferMap, int numConsumerGroups) {

  // Find the operation that is along producer's parent chain, and its parent
  // is the same op as producer's parent. Here p is producer, and c is consumer.
  auto getSameLevelOp = [](Operation *p, Operation *c) -> Operation * {
    while (!isa<triton::FuncOp>(c)) {
      if (c->getParentOp() == p->getParentOp()) {
        return c;
      }
      c = c->getParentOp();
    }
    llvm_unreachable("Failed to find consumer's same level Op with producer");
  };

  auto consumerReleaseHeutistic = [&](Operation *p, Operation *c,
                                      int consumerAsyncTaskId) -> Operation * {
    if (c->getBlock() != p->getBlock())
      return getSameLevelOp(p, c);
    for (auto it = c->getBlock()->rbegin(); it != c->getBlock()->rend(); ++it) {
      if (!it->hasAttr("async_task_id"))
        continue;
      auto asyncAttr = it->getAttrOfType<DenseIntElementsAttr>("async_task_id")
                           .getValues<int>();
      if (asyncAttr.size() == 1 && asyncAttr[0] == consumerAsyncTaskId)
        return &(*it);
    }
    return nullptr;
  };

  auto getAsyncTasks = [&](Operation *p, Operation *c,
                           SmallVector<AsyncTaskId> &asyncTaskP,
                           SmallVector<AsyncTaskId> &asyncTaskC,
                           SmallVector<AsyncTaskId> &asyncTasksPC) -> void {
    asyncTaskP = getNestedAsyncTaskIds(p);
    asyncTaskC = getNestedAsyncTaskIds(c);
    asyncTasksPC.reserve(asyncTaskP.size() + asyncTaskC.size());
    asyncTasksPC.insert(asyncTasksPC.end(), asyncTaskP.begin(),
                        asyncTaskP.end());
    asyncTasksPC.insert(asyncTasksPC.end(), asyncTaskC.begin(),
                        asyncTaskC.end());
  };

  // Go through each channel group.
  for (auto kv : groupedChannels) {
    auto headProducer = kv.second.front()->getSrcOp();
    auto tailProducer = kv.second.back()->getSrcOp();
    auto headConsumer = kv.second.front()->getDstOp();
    auto tailConsumer = kv.second.back()->getDstOp();
    // We have one set of tokens for each channel group.
    auto tokens = tokenMap.find(kv.second.front())->second;

    SmallVector<AsyncTaskId> asyncTaskP, asyncTaskC, asyncTasksPC;
    getAsyncTasks(headProducer, headConsumer, asyncTaskP, asyncTaskC,
                  asyncTasksPC);
    OpBuilderWithAsyncTaskIds builder(headProducer->getContext());
    if (auto funcOp = dyn_cast<triton::FuncOp>(headProducer->getParentOp())) {
      builder.setInsertionPointToStart(&(funcOp.getBody().front()));
    } else {
      builder.setInsertionPoint(headProducer->getParentOp());
    }
    builder.setAsynTaskIdsFromArray(asyncTasksPC);

    Value bufferIdx;
    Value phase = Value();
    if (auto forOp = headProducer->getParentOfType<scf::ForOp>()) {
      // We already added phase, bufferIdx to the ForOp.
      auto tSize = forOp.getBody()->getArguments().size();
      assert(tSize >= 2);
      bufferIdx = forOp.getBody()->getArguments().back();
      phase = forOp.getBody()->getArgument(tSize - 2); // next to last argument
    } else {
      // Producer is not in a ForOp, create phase and bufferIdx here.
      bufferIdx = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 32);
      phase = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 1);
    }

    builder.setAsynTaskIdsFromArray(asyncTaskP);
    for (auto token : tokens) {
      // Insert ProducerAcquireOp before the producer.
      builder.setInsertionPoint(headProducer);
      builder.createWithAsyncTaskIds<ttng::ProducerAcquireOp>(
          headProducer->getLoc(), token.second, bufferIdx, phase);

      // Insert ProducerCommitOp if producer is LoadOp. For TMA, TMA lowering
      // will handle the ProducerCommit.
      if (!isa<tt::ExperimentalDescriptorLoadOp>(headProducer)) {
        builder.setInsertionPointAfter(tailProducer);
        builder.createWithAsyncTaskIds<ttng::ProducerCommitOp>(
            tailProducer->getLoc(), token.second, bufferIdx);
      }
    }

    for (auto token : tokens) {
      builder.setAsynTaskIdsFromArray(token.first);
      // Insert ConsumerWaitOp
      if (!isa<tt::ExperimentalDescriptorLoadOp>(headProducer)) {
        auto consumerWaitPoint = getSameLevelOp(headProducer, headConsumer);
        builder.setInsertionPoint(consumerWaitPoint);
        builder.createWithAsyncTaskIds<ttng::ConsumerWaitOp>(
            headConsumer->getLoc(), token.second, bufferIdx, phase);
      }

      // Insert ConsumerReleaseOp.
      auto consumerReleasePoint =
          consumerReleaseHeutistic(tailProducer, tailConsumer, token.first);
      builder.setInsertionPointAfter(consumerReleasePoint);
      builder.createWithAsyncTaskIds<ttng::ConsumerReleaseOp>(
          consumerReleasePoint->getLoc(), token.second, bufferIdx);
    }

    SmallVector<tt::ExperimentalDescriptorLoadOp> tmaLoads;
    SmallVector<Value> buffers;
    // Go through all channels in this channel group.
    for (auto &c : kv.second) {
      if (isa<triton::LoadOp>(c->getSrcOp())) {
        // After createAsyncCopy, c->getSrcOp()/headProducer are no longer
        // valid.
        createAsyncCopy(bufferMap, c, c->getSrcOp(), asyncTasksPC, bufferIdx,
                        bufferIdx);
      } else if (auto tmaLoad = dyn_cast<tt::ExperimentalDescriptorLoadOp>(
                     c->getSrcOp())) {
        tmaLoads.push_back(tmaLoad);
        buffers.push_back(bufferMap.find(c)->second);
      } else {
        createLocalCopy(bufferMap, c, bufferIdx, bufferIdx);
      }
    }

    // Optimize TMA loads.
    if (tmaLoads.size() > 0) {
      auto barrierAllocs = barrierAllocMap.find(kv.second.front())->second;
      // TODO: we created one Alloc for each consumer taskId, but here, we
      // only use the first Alloc.
      auto barrierAlloc = barrierAllocs.begin()->second;
      optimizeTMALoads(builder, tmaLoads, buffers, barrierAlloc, bufferIdx,
                       bufferIdx, phase, headProducer, headConsumer);
    }
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
    replaceUsesAndPropagateType(builder, kv.getFirst(), kv.getSecond());
}

class TritonGPUWSCodePartitionPass
    : public impl::TritonGPUWSCodePartitionBase<TritonGPUWSCodePartitionPass> {
public:
  using impl::TritonGPUWSCodePartitionBase<
      TritonGPUWSCodePartitionPass>::TritonGPUWSCodePartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    // Disable code partitioning when numBuffers is 0.
    if (numBuffers == 0)
      return;

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

    // Step 2: group channels where each entry of the groupedChannels is keyed
    // by the dstOp.
    DenseMap<Operation *, SmallVector<Channel *>> groupedChannels;
    SmallVector<Operation *> mapKeyVec;
    groupChannels(channels, groupedChannels, mapKeyVec);

    // Step 3: reorder producer ops and the backward slices of the producer ops.
    reorderProducerOps(channels);

    // Step 4: find top-level ops that contain a channel, also create new ForOps
    // by adding phase and bufferIdx to the original ForOps, erase the original
    // ForOps.
    SmallVector<Operation *> asyncTaskTopOps =
        getTaskTopRegion(funcOp, channels);
    appendBufferIdxArgs(asyncTaskTopOps, numBuffers);
    LLVM_DEBUG({
      LDBG("\n\nafter appendBufferIdxArgs");
      funcOp.dump();
    });

    // Step 5: Create tokens, and buffers. A set of tokens for each group of
    // channels and an array of buffers for each channel.
    DenseMap<Channel *, DenseMap<int, Value>> barrierAllocMap;
    DenseMap<Channel *, DenseMap<int, Value>> tokenMap = createToken(
        groupedChannels, mapKeyVec, funcOp, numConsumerGroups, barrierAllocMap);
    DenseMap<Channel *, Value> bufferMap =
        createBuffer(channels, funcOp, numConsumerGroups);
    LLVM_DEBUG({
      LDBG("\n\nafter createBuffer");
      funcOp.dump();
    });

    // Step 6: add async communication ops (ProducerAcquire etc). Also lower the
    // loads.
    buildAsyncComm(groupedChannels, tokenMap, barrierAllocMap, bufferMap,
                   numConsumerGroups);
    LLVM_DEBUG({
      LDBG("\n\nwith SyncOps");
      funcOp.dump();
    });

    // If loadResult has a single use which is LocalAlloc, we can get rid of
    // sharedLoad and replace all uses of LocalAlloc with viewLoad.
    DenseMap<Operation *, Value> opsToReplace;
    foldLocalLoads(funcOp);
    LLVM_DEBUG({
      LDBG("\n\nsimplify localLoad + localAlloc");
      funcOp.dump();
    });

    auto ret = SpecializeRegion(funcOp, regDecProducer, regIncConsumer);
    LLVM_DEBUG({
      LDBG("\n\nwith SpecializeRegion");
      funcOp.dump();
    });
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

} // namespace gpu
} // namespace triton
} // namespace mlir
