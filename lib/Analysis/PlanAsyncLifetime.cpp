#include "triton/Analysis/PlanValueGraph.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include <algorithm>
#include <deque>
#include <numeric>
#include <set>
#include <tuple>

namespace mlir::triton::plan {
namespace {

struct OperationPoint {
  std::string blockId;
  int64_t position = -1;
};

struct GroupInstance {
  unsigned group = 0;
  int64_t iteration = 0;
  bool crossRegion = false;
};

struct SimulationState {
  std::deque<GroupInstance> outstanding;
};

static bool isCommit(Operation *op) {
  return op->getName().getStringRef() == "ttg.async_commit_group";
}

static bool isWait(Operation *op) {
  return llvm::is_contained(
      {StringRef("ttg.async_wait"), StringRef("amdg.async_wait")},
      op->getName().getStringRef());
}

static bool isLocalBarrier(Operation *op) {
  return llvm::is_contained(
      {StringRef("ttg.barrier"), StringRef("gpu.barrier")},
      op->getName().getStringRef());
}

static int64_t getRetainedGroupCount(Operation *op) {
  if (IntegerAttr count = op->getAttrOfType<IntegerAttr>("num"))
    return std::max<int64_t>(0, count.getInt());
  if (IntegerAttr count =
          op->getAttrOfType<IntegerAttr>("ttg.num_commit_groups"))
    return std::max<int64_t>(0, count.getInt());
  return 0;
}

static std::string pathKey(const PlanSlotPath &path) {
  std::string key = path.rootValueId;
  for (const PlanSlotExpression &index : path.indices)
    key += "|" + index.text;
  return key;
}

static bool rootsOverlap(ArrayRef<std::string> lhs, ArrayRef<std::string> rhs) {
  return llvm::any_of(
      lhs, [&](StringRef left) { return llvm::is_contained(rhs, left); });
}

enum class AliasRelation { Disjoint, Same, MayAlias };

static AliasRelation compareExpression(const PlanSlotExpression &lhs,
                                       const PlanSlotExpression &rhs) {
  if (lhs.text == rhs.text)
    return AliasRelation::Same;
  if (!lhs.possibleSlots.empty() && !rhs.possibleSlots.empty()) {
    bool intersects = llvm::any_of(lhs.possibleSlots, [&](int64_t slot) {
      return llvm::is_contained(rhs.possibleSlots, slot);
    });
    if (!intersects)
      return AliasRelation::Disjoint;
  }
  if (lhs.kind == "constant" && rhs.kind == "constant")
    return AliasRelation::Disjoint;
  return AliasRelation::MayAlias;
}

static AliasRelation comparePath(const PlanSlotPath &lhs,
                                 const PlanSlotPath &rhs) {
  if (lhs.rootValueId != rhs.rootValueId)
    return AliasRelation::Disjoint;
  AliasRelation result = AliasRelation::Same;
  size_t common = std::min(lhs.indices.size(), rhs.indices.size());
  for (size_t index = 0; index < common; ++index) {
    AliasRelation relation =
        compareExpression(lhs.indices[index], rhs.indices[index]);
    if (relation == AliasRelation::Disjoint)
      return relation;
    if (relation == AliasRelation::MayAlias)
      result = relation;
  }
  if (lhs.indices.size() != rhs.indices.size())
    result = AliasRelation::MayAlias;
  return result;
}

static AliasRelation comparePaths(ArrayRef<PlanSlotPath> lhs,
                                  ArrayRef<PlanSlotPath> rhs) {
  if (lhs.empty() || rhs.empty())
    return AliasRelation::MayAlias;
  AliasRelation result = AliasRelation::Disjoint;
  for (const PlanSlotPath &left : lhs)
    for (const PlanSlotPath &right : rhs) {
      AliasRelation relation = comparePath(left, right);
      if (relation == AliasRelation::Same)
        return relation;
      if (relation == AliasRelation::MayAlias)
        result = relation;
    }
  return result;
}

static bool accessesMayAlias(ArrayRef<std::string> lhsRoots,
                             ArrayRef<PlanSlotPath> lhsPaths,
                             ArrayRef<std::string> rhsRoots,
                             ArrayRef<PlanSlotPath> rhsPaths) {
  return rootsOverlap(lhsRoots, rhsRoots) &&
         comparePaths(lhsPaths, rhsPaths) != AliasRelation::Disjoint;
}

static void collectProducerOperations(
    Value value, const llvm::DenseMap<Operation *, unsigned> &producerIndices,
    llvm::SmallSetVector<Operation *, 4> &producers,
    llvm::DenseSet<Value> &visited, unsigned depth = 0) {
  if (!value || depth > 32 || !visited.insert(value).second)
    return;
  if (Operation *definition = value.getDefiningOp()) {
    if (producerIndices.count(definition)) {
      producers.insert(definition);
      return;
    }
    for (Value operand : definition->getOperands())
      collectProducerOperations(operand, producerIndices, producers, visited,
                                depth + 1);
    return;
  }

  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return;
  Operation *parent = argument.getOwner()->getParentOp();
  if (auto loop = dyn_cast_or_null<scf::ForOp>(parent)) {
    if (argument == loop.getInductionVar())
      return;
    unsigned index = argument.getArgNumber() - 1;
    if (index >= loop.getInitArgs().size())
      return;
    collectProducerOperations(loop.getInitArgs()[index], producerIndices,
                              producers, visited, depth + 1);
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    collectProducerOperations(yield.getResults()[index], producerIndices,
                              producers, visited, depth + 1);
  } else if (auto loop = dyn_cast_or_null<scf::WhileOp>(parent)) {
    unsigned index = argument.getArgNumber();
    if (argument.getOwner() == loop.getBeforeBody()) {
      if (index < loop.getInits().size())
        collectProducerOperations(loop.getInits()[index], producerIndices,
                                  producers, visited, depth + 1);
      if (index < loop.getYieldOp().getResults().size())
        collectProducerOperations(loop.getYieldOp().getResults()[index],
                                  producerIndices, producers, visited,
                                  depth + 1);
    } else if (index < loop.getConditionOp().getArgs().size()) {
      collectProducerOperations(loop.getConditionOp().getArgs()[index],
                                producerIndices, producers, visited, depth + 1);
    }
  }
}

static void addFrontier(std::vector<PlanAsyncFrontier> &frontiers,
                        PlanAsyncFrontier frontier) {
  auto key = [](const PlanAsyncFrontier &item) {
    return std::tie(item.operationId, item.kind, item.iterationDistance,
                    item.precision);
  };
  if (llvm::none_of(frontiers, [&](const auto &item) {
        return key(item) == key(frontier);
      }))
    frontiers.push_back(std::move(frontier));
}

static void addCompletion(PlanAsyncWaitRecord &wait,
                          PlanAsyncWaitCompletion completion) {
  if (llvm::none_of(wait.completedGroups, [&](const auto &item) {
        return std::tie(item.groupId, item.iterationDistance, item.precision) ==
               std::tie(completion.groupId, completion.iterationDistance,
                        completion.precision);
      }))
    wait.completedGroups.push_back(std::move(completion));
}

class AsyncLifetimeAnalysis {
public:
  AsyncLifetimeAnalysis(
      FuncOp function,
      const llvm::DenseMap<Operation *, std::string> &operationIds,
      const llvm::DenseMap<Value, std::string> &valueIds,
      ArrayRef<PlanBlockRecord> blocks, ArrayRef<PlanAliasRecord> aliases,
      ArrayRef<PlanMemoryAccess> memoryAccesses)
      : function(function), operationIds(operationIds), valueIds(valueIds),
        blocks(blocks), aliases(aliases), memoryAccesses(memoryAccesses) {}

  PlanAsyncLifetimeResult run() {
    indexOperations();
    buildTransactions();
    buildGroups();
    SimulationState state;
    simulateBlock(function.getBody().front(), state, /*iteration=*/0,
                  /*loopDepth=*/0, "exact");
    connectCompletions();
    connectConsumersAndReuse();
    diagnoseUnsupportedDomains();
    sortResult();
    return std::move(result);
  }

private:
  void indexOperations() {
    for (const PlanBlockRecord &block : blocks)
      for (auto [position, operationId] : llvm::enumerate(block.operations))
        points[operationId] = {block.id, static_cast<int64_t>(position)};
    for (const auto &[operation, id] : operationIds) {
      operations[id] = operation;
      if (isWait(operation)) {
        PlanAsyncWaitRecord wait;
        wait.operationId = id;
        wait.retainedGroupCount = getRetainedGroupCount(operation);
        wait.precision = "exact";
        waitIndices[operation] = result.waits.size();
        result.waits.push_back(std::move(wait));
      }
    }
  }

  void buildTransactions() {
    for (const PlanMemoryAccess &access : memoryAccesses) {
      if (!access.pendingAsync ||
          !llvm::is_contained({StringRef("read"), StringRef("write")},
                              access.effect))
        continue;
      PlanAsyncTransaction transaction;
      transaction.id = "async-tx:" + access.operationId + ":" +
                       std::to_string(result.transactions.size());
      transaction.producerOperationId = access.operationId;
      transaction.destinationValueId = access.valueId;
      transaction.direction =
          access.effect == "write" ? "lds_write" : "lds_read";
      transaction.precision = "exact";
      transaction.rootValueIds = access.rootValueIds;
      transaction.slotPaths = access.slotPaths;
      Operation *operation = operations.lookup(access.operationId);
      if (operation) {
        producerTransactions[operation].push_back(result.transactions.size());
        producerIndices[operation] = result.transactions.size();
      }
      result.transactions.push_back(std::move(transaction));
    }
  }

  void buildGroups() {
    function.walk([&](Operation *operation) {
      if (!isCommit(operation))
        return;
      PlanAsyncGroup group;
      group.commitOperationId = operationIds.lookup(operation);
      group.id = "async-group:" + group.commitOperationId;

      llvm::SmallSetVector<Operation *, 4> producers;
      llvm::DenseSet<Value> visited;
      for (Value operand : operation->getOperands())
        collectProducerOperations(operand, producerIndices, producers, visited);
      for (Operation *producer : producers)
        for (unsigned transaction : producerTransactions.lookup(producer)) {
          group.transactionIds.push_back(result.transactions[transaction].id);
          result.transactions[transaction].commitGroupId = group.id;
        }
      llvm::sort(group.transactionIds);
      group.transactionIds.erase(
          std::unique(group.transactionIds.begin(), group.transactionIds.end()),
          group.transactionIds.end());
      groupIndices[operation] = result.groups.size();
      result.groups.push_back(std::move(group));
    });

    // Tokenless commits group the pending async operations since the previous
    // commit in their direct block.
    for (Block &block : function.getBody().getBlocks())
      assignTokenlessGroups(block);
  }

  void assignTokenlessGroups(Block &block) {
    SmallVector<unsigned> pending;
    for (Operation &operation : block) {
      for (unsigned transaction : producerTransactions.lookup(&operation))
        if (result.transactions[transaction].commitGroupId.empty())
          pending.push_back(transaction);
      if (isCommit(&operation) && operation.getNumOperands() == 0) {
        unsigned groupIndex = groupIndices.lookup(&operation);
        for (unsigned transaction : pending) {
          result.groups[groupIndex].transactionIds.push_back(
              result.transactions[transaction].id);
          result.transactions[transaction].commitGroupId =
              result.groups[groupIndex].id;
        }
        pending.clear();
      }
      for (Region &region : operation.getRegions())
        for (Block &nested : region)
          assignTokenlessGroups(nested);
    }
  }

  void recordCompleted(Operation *waitOperation, GroupInstance instance,
                       int64_t iteration, StringRef precision) {
    unsigned waitIndex = waitIndices.lookup(waitOperation);
    PlanAsyncWaitRecord &wait = result.waits[waitIndex];
    int64_t distance = std::max<int64_t>(0, iteration - instance.iteration);
    std::string completionPrecision =
        instance.crossRegion ? "conservative_cross_region" : precision.str();
    addCompletion(wait, {result.groups[instance.group].id, distance,
                         completionPrecision});
    if (completionPrecision != "exact" &&
        completionPrecision != "structured_exact")
      wait.precision = completionPrecision;
    else if (completionPrecision == "structured_exact" &&
             wait.precision == "exact")
      wait.precision = completionPrecision;
  }

  void processWait(Operation *operation, SimulationState &state,
                   int64_t iteration, StringRef precision) {
    SmallVector<unsigned> tokenGroups;
    for (Value operand : operation->getOperands()) {
      llvm::SmallSetVector<Operation *, 4> commits;
      llvm::DenseSet<Value> visited;
      collectCommitOperations(operand, commits, visited);
      for (Operation *commit : commits)
        if (groupIndices.count(commit))
          tokenGroups.push_back(groupIndices.lookup(commit));
    }

    std::deque<GroupInstance> remaining;
    if (!tokenGroups.empty()) {
      for (GroupInstance instance : state.outstanding) {
        if (llvm::is_contained(tokenGroups, instance.group))
          recordCompleted(operation, instance, iteration, precision);
        else
          remaining.push_back(instance);
      }
    } else {
      int64_t complete =
          std::max<int64_t>(0, static_cast<int64_t>(state.outstanding.size()) -
                                   getRetainedGroupCount(operation));
      for (int64_t index = 0; index < complete; ++index) {
        recordCompleted(operation, state.outstanding.front(), iteration,
                        precision);
        state.outstanding.pop_front();
      }
      remaining = std::move(state.outstanding);
    }
    state.outstanding = std::move(remaining);

    PlanAsyncWaitRecord &wait = result.waits[waitIndices.lookup(operation)];
    for (const GroupInstance &instance : state.outstanding)
      if (!llvm::is_contained(wait.possiblyOutstandingGroups,
                              result.groups[instance.group].id))
        wait.possiblyOutstandingGroups.push_back(
            result.groups[instance.group].id);
  }

  void collectCommitOperations(Value value,
                               llvm::SmallSetVector<Operation *, 4> &commits,
                               llvm::DenseSet<Value> &visited,
                               unsigned depth = 0) {
    if (!value || depth > 32 || !visited.insert(value).second)
      return;
    if (Operation *definition = value.getDefiningOp()) {
      if (isCommit(definition)) {
        commits.insert(definition);
        return;
      }
      for (Value operand : definition->getOperands())
        collectCommitOperations(operand, commits, visited, depth + 1);
      return;
    }
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument)
      return;
    if (auto loop =
            dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())) {
      if (argument == loop.getInductionVar())
        return;
      unsigned index = argument.getArgNumber() - 1;
      if (index < loop.getInitArgs().size()) {
        collectCommitOperations(loop.getInitArgs()[index], commits, visited,
                                depth + 1);
        auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
        collectCommitOperations(yield.getResults()[index], commits, visited,
                                depth + 1);
      }
    }
  }

  void simulateBlock(Block &block, SimulationState &state, int64_t iteration,
                     unsigned loopDepth, StringRef precision) {
    for (Operation &operation : block) {
      if (isCommit(&operation)) {
        state.outstanding.push_back(
            {groupIndices.lookup(&operation), iteration, false});
      } else if (isWait(&operation)) {
        processWait(&operation, state, iteration,
                    loopDepth ? "structured_exact" : precision);
      } else if (auto loop = dyn_cast<scf::ForOp>(&operation)) {
        bool containsQueueOperation = false;
        loop.getRegion().walk([&](Operation *nested) {
          containsQueueOperation |= isCommit(nested) || isWait(nested);
        });
        if (!containsQueueOperation)
          continue;
        // Iterate until the outstanding group templates and their relative
        // ages repeat. This is a symbolic fixed point, not a runtime unroll.
        std::set<std::string> seenStates;
        bool converged = false;
        for (int64_t offset = 0; offset < 64; ++offset) {
          simulateBlock(*loop.getBody(), state, offset, loopDepth + 1,
                        "structured_exact");
          std::string signature;
          for (const GroupInstance &instance : state.outstanding)
            signature += std::to_string(instance.group) + ":" +
                         std::to_string(offset - instance.iteration) + ";";
          if (!seenStates.insert(signature).second) {
            converged = true;
            break;
          }
        }
        if (!converged)
          result.diagnostics.push_back(
              {"warning", "async_loop_fixed_point_limit",
               "async commit queue did not stabilize after 64 symbolic "
               "iterations",
               operationIds.lookup(&operation), ""});
        for (GroupInstance &instance : state.outstanding)
          instance.crossRegion = true;
      } else if (auto loop = dyn_cast<scf::WhileOp>(&operation)) {
        result.diagnostics.push_back(
            {"warning", "conservative_async_while",
             "async lifetime used two symbolic while-loop iterations",
             operationIds.lookup(&operation), ""});
        for (int64_t offset = 0; offset < 2; ++offset) {
          simulateBlock(*loop.getBeforeBody(), state, offset, loopDepth + 1,
                        "conservative_control_flow");
          simulateBlock(*loop.getAfterBody(), state, offset, loopDepth + 1,
                        "conservative_control_flow");
        }
      } else if (auto ifOp = dyn_cast<scf::IfOp>(&operation)) {
        SimulationState thenState = state;
        SimulationState elseState = state;
        simulateBlock(ifOp.getThenRegion().front(), thenState, iteration,
                      loopDepth, "conservative_control_flow");
        if (!ifOp.getElseRegion().empty())
          simulateBlock(ifOp.getElseRegion().front(), elseState, iteration,
                        loopDepth, "conservative_control_flow");
        state = joinStates(thenState, elseState);
      }
    }
  }

  SimulationState joinStates(const SimulationState &lhs,
                             const SimulationState &rhs) {
    SimulationState resultState = lhs;
    for (GroupInstance instance : rhs.outstanding) {
      bool present =
          llvm::any_of(resultState.outstanding, [&](const auto &item) {
            return item.group == instance.group &&
                   item.iteration == instance.iteration;
          });
      if (!present) {
        instance.crossRegion = true;
        resultState.outstanding.push_back(instance);
      }
    }
    for (GroupInstance &instance : resultState.outstanding)
      instance.crossRegion = true;
    return resultState;
  }

  void connectCompletions() {
    std::map<std::string, unsigned> transactionById;
    for (auto [index, transaction] : llvm::enumerate(result.transactions))
      transactionById[transaction.id] = index;
    for (const PlanAsyncWaitRecord &wait : result.waits) {
      Operation *waitOperation = operations.lookup(wait.operationId);
      for (const PlanAsyncWaitCompletion &completion : wait.completedGroups) {
        auto group = llvm::find_if(result.groups, [&](const auto &candidate) {
          return candidate.id == completion.groupId;
        });
        if (group == result.groups.end())
          continue;
        for (const std::string &transactionId : group->transactionIds) {
          PlanAsyncTransaction &transaction =
              result.transactions[transactionById[transactionId]];
          addFrontier(transaction.completionFrontiers,
                      makeFrontier(waitOperation, "completion_wait",
                                   completion.iterationDistance,
                                   completion.precision));
          if (Operation *barrier = findBarrierAfter(waitOperation))
            addFrontier(transaction.visibilityFrontiers,
                        makeFrontier(barrier, "visibility_barrier",
                                     completion.iterationDistance,
                                     completion.precision));
        }
      }
    }
  }

  PlanAsyncFrontier makeFrontier(Operation *operation, StringRef kind,
                                 int64_t distance, StringRef precision) {
    std::string operationId = operationIds.lookup(operation);
    OperationPoint point = points[operationId];
    return {operationId,     point.blockId,  kind.str(),
            precision.str(), point.position, distance};
  }

  Operation *findBarrierAfter(Operation *operation) {
    for (Operation *next = operation ? operation->getNextNode() : nullptr; next;
         next = next->getNextNode())
      if (isLocalBarrier(next))
        return next;
    return nullptr;
  }

  Operation *findReleaseBarrier(Operation *consumer) {
    if (Operation *barrier = findBarrierAfter(consumer))
      return barrier;
    Operation *parent = consumer ? consumer->getParentOp() : nullptr;
    if (isa_and_nonnull<scf::ForOp, scf::WhileOp>(parent))
      for (Operation &candidate : *consumer->getBlock())
        if (isLocalBarrier(&candidate))
          return &candidate;
    return nullptr;
  }

  bool isAfter(Operation *anchor, Operation *candidate) {
    if (!anchor || !candidate)
      return false;
    Operation *cursor = candidate;
    while (cursor && cursor->getBlock() != anchor->getBlock())
      cursor = cursor->getParentOp();
    return cursor && anchor->isBeforeInBlock(cursor);
  }

  std::optional<int64_t>
  slotReusePeriod(const PlanAsyncTransaction &transaction) {
    int64_t period = 1;
    bool foundModulo = false;
    for (const PlanSlotPath &path : transaction.slotPaths)
      for (const PlanSlotExpression &index : path.indices) {
        if (index.kind == "unknown" ||
            (index.kind == "induction" && index.modulus == 0))
          return std::nullopt;
        if (index.modulus > 0) {
          int64_t coefficient = std::abs(index.coefficient);
          period = std::lcm(period, index.modulus /
                                        std::gcd(coefficient, index.modulus));
          foundModulo = true;
        }
      }
    return foundModulo ? period : std::optional<int64_t>(1);
  }

  bool isInStructuredLoop(Operation *operation) {
    return operation && operation->getParentOfType<scf::ForOp>();
  }

  void connectConsumersAndReuse() {
    for (PlanAsyncTransaction &transaction : result.transactions) {
      if (transaction.direction != "lds_write")
        continue;
      Operation *producer = operations.lookup(transaction.producerOperationId);
      for (const PlanMemoryAccess &access : memoryAccesses) {
        if (access.effect != "read" ||
            access.operationId == transaction.producerOperationId ||
            !accessesMayAlias(transaction.rootValueIds, transaction.slotPaths,
                              access.rootValueIds, access.slotPaths))
          continue;
        Operation *consumer = operations.lookup(access.operationId);
        for (const PlanAsyncFrontier &visibility :
             transaction.visibilityFrontiers) {
          Operation *barrier = operations.lookup(visibility.operationId);
          int64_t distance = visibility.iterationDistance;
          std::string precision = visibility.precision;
          if (!isAfter(barrier, consumer)) {
            if (!barrier->getParentOfType<scf::ForOp>() ||
                barrier->getBlock() != consumer->getBlock())
              continue;
            ++distance;
            precision = "conservative_slot_alias";
          }
          addFrontier(
              transaction.consumerFrontiers,
              makeFrontier(consumer, "lds_consumer", distance, precision));
          if (Operation *release = findReleaseBarrier(consumer)) {
            int64_t releaseDistance = distance;
            if (!isAfter(consumer, release))
              ++releaseDistance;
            addFrontier(transaction.releaseFrontiers,
                        makeFrontier(release, "reuse_release_barrier",
                                     releaseDistance, precision));
          }
        }
      }

      // The same static producer in a loop overwrites the same modulo slot
      // after its symbolic period. This captures ping-pong and ring-buffer
      // reuse without unrolling the runtime loop.
      if (isInStructuredLoop(producer)) {
        if (std::optional<int64_t> period = slotReusePeriod(transaction))
          addFrontier(transaction.overwriteFrontiers,
                      makeFrontier(producer, "slot_overwrite", *period,
                                   transaction.slotPaths.empty()
                                       ? "conservative_slot_alias"
                                       : "structured_exact"));
      }

      // Also record statically later writes which may alias the destination.
      for (const PlanMemoryAccess &access : memoryAccesses) {
        if (access.effect != "write" ||
            access.operationId == transaction.producerOperationId ||
            !accessesMayAlias(transaction.rootValueIds, transaction.slotPaths,
                              access.rootValueIds, access.slotPaths))
          continue;
        Operation *overwrite = operations.lookup(access.operationId);
        if (isAfter(producer, overwrite))
          addFrontier(transaction.overwriteFrontiers,
                      makeFrontier(overwrite, "slot_overwrite", 0,
                                   "conservative_slot_alias"));
      }
      diagnoseTransaction(transaction);
    }
  }

  void diagnoseTransaction(const PlanAsyncTransaction &transaction) {
    auto addHazard = [&](StringRef severity, StringRef code, StringRef message,
                         StringRef operationId = "") {
      std::string root = transaction.rootValueIds.empty()
                             ? ""
                             : transaction.rootValueIds.front();
      result.hazards.push_back({severity.str(), code.str(), message.str(),
                                transaction.id, operationId.str(), root});
    };
    if (transaction.commitGroupId.empty())
      addHazard("warning", "async_write_without_commit",
                "async LDS access is not associated with a commit group",
                transaction.producerOperationId);
    if (transaction.completionFrontiers.empty())
      addHazard("warning", "async_write_without_completion",
                "no completing wait was proven for this async LDS access",
                transaction.producerOperationId);
    if (!transaction.completionFrontiers.empty() &&
        transaction.visibilityFrontiers.empty())
      addHazard(
          "warning", "async_write_without_visibility_barrier",
          "completion was proven but no following local barrier was found");
    if (!transaction.overwriteFrontiers.empty() &&
        transaction.releaseFrontiers.empty())
      addHazard("warning", "lds_overwrite_without_release",
                "a possible overwrite has no proven consumer-release barrier");
  }

  void diagnoseUnsupportedDomains() {
    function.walk([&](Operation *operation) {
      StringRef name = operation->getName().getStringRef();
      if ((name.contains("tdm") || name.contains("mbarrier")) &&
          (name.contains("async") || name.contains("wait") ||
           name.contains("arrive")))
        result.diagnostics.push_back(
            {"warning", "unsupported_async_domain",
             "M1.4c commit-count analysis does not model TDM/mbarrier epochs",
             operationIds.lookup(operation), ""});
    });
  }

  void sortResult() {
    auto sortFrontiers = [](auto &frontiers) {
      llvm::sort(frontiers, [](const auto &lhs, const auto &rhs) {
        return std::tie(lhs.iterationDistance, lhs.blockId, lhs.position,
                        lhs.operationId, lhs.kind, lhs.precision) <
               std::tie(rhs.iterationDistance, rhs.blockId, rhs.position,
                        rhs.operationId, rhs.kind, rhs.precision);
      });
    };
    for (PlanAsyncTransaction &transaction : result.transactions) {
      llvm::sort(transaction.rootValueIds);
      llvm::sort(transaction.slotPaths, [](const auto &lhs, const auto &rhs) {
        return pathKey(lhs) < pathKey(rhs);
      });
      sortFrontiers(transaction.completionFrontiers);
      sortFrontiers(transaction.visibilityFrontiers);
      sortFrontiers(transaction.consumerFrontiers);
      sortFrontiers(transaction.releaseFrontiers);
      sortFrontiers(transaction.overwriteFrontiers);
      bool conservative = false;
      bool structured = false;
      bool exact = false;
      auto updatePrecision = [&](ArrayRef<PlanAsyncFrontier> frontiers) {
        for (const PlanAsyncFrontier &frontier : frontiers) {
          conservative |=
              StringRef(frontier.precision).starts_with("conservative");
          structured |= frontier.precision == "structured_exact";
          exact |= frontier.precision == "exact";
        }
      };
      updatePrecision(transaction.completionFrontiers);
      updatePrecision(transaction.visibilityFrontiers);
      updatePrecision(transaction.consumerFrontiers);
      updatePrecision(transaction.releaseFrontiers);
      updatePrecision(transaction.overwriteFrontiers);
      transaction.precision =
          conservative && (structured || exact)
              ? "mixed"
              : (conservative ? "conservative"
                              : (structured ? "structured_exact" : "exact"));
    }
    llvm::sort(result.transactions, [](const auto &lhs, const auto &rhs) {
      return lhs.id < rhs.id;
    });
    llvm::sort(result.groups, [](const auto &lhs, const auto &rhs) {
      return lhs.id < rhs.id;
    });
    for (PlanAsyncWaitRecord &wait : result.waits) {
      llvm::sort(wait.completedGroups, [](const auto &lhs, const auto &rhs) {
        return std::tie(lhs.groupId, lhs.iterationDistance, lhs.precision) <
               std::tie(rhs.groupId, rhs.iterationDistance, rhs.precision);
      });
      llvm::sort(wait.possiblyOutstandingGroups);
    }
    llvm::sort(result.waits, [](const auto &lhs, const auto &rhs) {
      return lhs.operationId < rhs.operationId;
    });
    llvm::sort(result.hazards, [](const auto &lhs, const auto &rhs) {
      return std::tie(lhs.severity, lhs.code, lhs.transactionId,
                      lhs.operationId) < std::tie(rhs.severity, rhs.code,
                                                  rhs.transactionId,
                                                  rhs.operationId);
    });
    llvm::sort(result.diagnostics, [](const auto &lhs, const auto &rhs) {
      return std::tie(lhs.severity, lhs.code, lhs.operationId, lhs.valueId,
                      lhs.message) < std::tie(rhs.severity, rhs.code,
                                              rhs.operationId, rhs.valueId,
                                              rhs.message);
    });
  }

  FuncOp function;
  const llvm::DenseMap<Operation *, std::string> &operationIds;
  const llvm::DenseMap<Value, std::string> &valueIds;
  ArrayRef<PlanBlockRecord> blocks;
  ArrayRef<PlanAliasRecord> aliases;
  ArrayRef<PlanMemoryAccess> memoryAccesses;
  PlanAsyncLifetimeResult result;
  llvm::StringMap<Operation *> operations;
  llvm::StringMap<OperationPoint> points;
  llvm::DenseMap<Operation *, SmallVector<unsigned>> producerTransactions;
  llvm::DenseMap<Operation *, unsigned> producerIndices;
  llvm::DenseMap<Operation *, unsigned> groupIndices;
  llvm::DenseMap<Operation *, unsigned> waitIndices;
};

} // namespace

FailureOr<PlanAsyncLifetimeResult> analyzePlanAsyncLifetimes(
    FuncOp function,
    const llvm::DenseMap<Operation *, std::string> &operationIds,
    const llvm::DenseMap<Value, std::string> &valueIds,
    ArrayRef<PlanBlockRecord> blocks, ArrayRef<PlanAliasRecord> aliases,
    ArrayRef<PlanMemoryAccess> memoryAccesses) {
  return AsyncLifetimeAnalysis(function, operationIds, valueIds, blocks,
                               aliases, memoryAccesses)
      .run();
}

} // namespace mlir::triton::plan
