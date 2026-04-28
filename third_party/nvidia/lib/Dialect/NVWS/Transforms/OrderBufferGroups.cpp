#include "InsertSemas.h"
#include "MetaToNVWSConvert.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/LatencyModel.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include <cstdlib>
#include <limits>
#include <map>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvws_semas;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton {
#define GEN_PASS_DEF_NVWSORDERBUFFERGROUPS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
} // namespace mlir::triton

namespace {

constexpr int64_t kMinReadinessGap = 32;
constexpr int64_t kMaxReadiness = std::numeric_limits<int32_t>::max();

using Predecessors = DenseMap<Operation *, SmallVector<Operation *, 2>>;
using IncomingByOp = DenseMap<Operation *, SmallVector<Operation *, 2>>;

static bool isSourceFreeAlloc(Operation *op) {
  if (auto alloc = dyn_cast<ttng::TMEMAllocOp>(op))
    return !alloc.getSrc();
  if (auto alloc = dyn_cast<LocalAllocOp>(op))
    return !alloc.getSrc();
  return false;
}

static bool isEligibleGroup(const GroupDag &group, Block *entry) {
  if (!group.isTmem() || group.isCircular() ||
      group.pieceTable.members.empty() || group.bufferId < 0)
    return false;
  for (const Member &member : group.pieceTable.members) {
    auto alloc = dyn_cast_or_null<ttng::TMEMAllocOp>(member.allocOp);
    if (!alloc || alloc.getSrc() || alloc->getBlock() != entry ||
        !alloc->hasAttr(kBufferCopyAttrName))
      return false;
  }
  return true;
}

static void addUniquePredecessor(Predecessors &predecessors, Operation *op,
                                 Operation *predecessor) {
  if (!op || !predecessor || op == predecessor)
    return;
  auto &values = predecessors[op];
  if (!llvm::is_contained(values, predecessor))
    values.push_back(predecessor);
}

// Add the simple same-block memory dependencies that are absent from SSA.
// The access DAG supplies exact alias pieces, so disjoint members of one
// physical buffer group do not acquire false dependencies.
static IncomingByOp collectIncomingWriters(GroupDag &group,
                                           Predecessors &predecessors) {
  DenseMap<Block *, SmallVector<Node *>> nodesByBlock;
  for (Node *node : group.nodesOfKind(Node::Access))
    if (node->op)
      nodesByBlock[node->op->getBlock()].push_back(node);

  IncomingByOp incoming;
  for (auto &[block, nodes] : nodesByBlock) {
    (void)block;
    llvm::sort(nodes, [](Node *lhs, Node *rhs) {
      return lhs->op != rhs->op && lhs->op->isBeforeInBlock(rhs->op);
    });
    struct PieceState {
      Node *lastWriter = nullptr;
      SetVector<Node *> readers;
    };
    DenseMap<PieceId, PieceState> state;
    for (Node *node : nodes) {
      DenseMap<PieceId, Effect> effects;
      forEachTouchedPiece(group, node, [&](PieceId piece, Effect effect) {
        mergeEffect(effects, piece, effect);
      });
      SetVector<Operation *> incomingSources;
      auto recordSource = [&](Node *source) {
        if (!source)
          return;
        addUniquePredecessor(predecessors, node->op, source->op);
        if (!sameOwner(source->owner, node->owner))
          incomingSources.insert(source->op);
      };
      for (auto [piece, effect] : effects) {
        PieceState &pieceState = state[piece];
        recordSource(pieceState.lastWriter);
        if (effect == Effect::R) {
          pieceState.readers.insert(node);
          continue;
        }
        for (Node *reader : pieceState.readers)
          recordSource(reader);
        pieceState.readers.clear();
        pieceState.lastWriter = node;
      }
      if (!incomingSources.empty())
        incoming[node->op].assign(incomingSources.begin(),
                                  incomingSources.end());
    }
  }
  return incoming;
}

class ReadinessScorer {
public:
  explicit ReadinessScorer(const Predecessors &memoryPredecessors)
      : memoryPredecessors(memoryPredecessors) {}

  std::optional<int64_t> score(Operation *op) {
    if (!op)
      return std::nullopt;
    if (auto it = memo.find(op); it != memo.end())
      return it->second;
    if (!active.insert(op).second)
      return std::nullopt;

    int64_t start = 0;
    bool complete = true;
    auto include = [&](Operation *predecessor) {
      std::optional<int64_t> value = score(predecessor);
      if (!value) {
        complete = false;
        return;
      }
      start = std::max(start, *value);
    };
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        include(def);
    if (auto it = memoryPredecessors.find(op);
        it != memoryPredecessors.end())
      for (Operation *predecessor : it->second)
        include(predecessor);

    active.erase(op);
    if (!complete)
      return std::nullopt;
    int64_t latency = model.getLatency(op).latency;
    int64_t ready = std::min(
        kMaxReadiness, start + std::max<int64_t>(latency, 0));
    memo[op] = ready;
    return ready;
  }

private:
  const Predecessors &memoryPredecessors;
  NVLatencyModel model;
  DenseMap<Operation *, int64_t> memo;
  DenseSet<Operation *> active;
};

struct PairPreference {
  std::optional<unsigned> before;
  bool conflict = false;
};

static bool canMoveEarlier(Operation *op, Operation *before) {
  if (!op || !before || op->getBlock() != before->getBlock() ||
      op->isBeforeInBlock(before) || !isSourceFreeAlloc(op))
    return false;
  for (Operation *cursor = before; cursor && cursor != op;
       cursor = cursor->getNextNode())
    if (!isSourceFreeAlloc(cursor))
      return false;
  return true;
}

static LogicalResult orderGroupsInBlock(FuncOp func, Block &functionBlock) {
  FailureOr<SmallVector<GroupDag, 0>> groupsOr =
      collectGroups(func, &functionBlock);
  if (failed(groupsOr))
    return failure();
  SmallVector<GroupDag, 0> &groups = *groupsOr;
  if (groups.size() < 2)
    return success();
  for (GroupDag &group : groups)
    if (failed(buildAccessDag(group, func, functionBlock)))
      return failure();

  SmallVector<bool> eligible(groups.size(), false);
  SmallVector<Operation *> representatives(groups.size(), nullptr);
  for (auto [index, group] : llvm::enumerate(groups)) {
    eligible[index] = isEligibleGroup(group, &functionBlock);
    if (eligible[index])
      representatives[index] = group.pieceTable.members.front().allocOp;
  }

  Predecessors memoryPredecessors;
  SmallVector<IncomingByOp> incomingByGroup;
  incomingByGroup.reserve(groups.size());
  for (GroupDag &group : groups)
    incomingByGroup.push_back(
        collectIncomingWriters(group, memoryPredecessors));

  DenseMap<Operation *, SmallVector<unsigned>> groupsByConsumer;
  for (unsigned index = 0; index < groups.size(); ++index) {
    if (!eligible[index])
      continue;
    for (auto &[consumer, writers] : incomingByGroup[index])
      if (!writers.empty())
        groupsByConsumer[consumer].push_back(index);
  }

  ReadinessScorer scorer(memoryPredecessors);
  auto summarizeReadiness = [&](unsigned group,
                                Operation *consumer)
      -> std::optional<std::pair<int64_t, StageCluster>> {
    auto it = incomingByGroup[group].find(consumer);
    if (it == incomingByGroup[group].end() || it->second.empty())
      return std::nullopt;
    int64_t readyTime = 0;
    StageCluster commonSchedule;
    for (Operation *writer : it->second) {
      std::optional<int64_t> ready = scorer.score(writer);
      StageCluster schedule = getStageCluster(writer);
      if (!ready || !schedule ||
          (commonSchedule && commonSchedule != schedule))
        return std::nullopt;
      commonSchedule = schedule;
      readyTime = std::max(readyTime, *ready);
    }
    return std::make_pair(readyTime, commonSchedule);
  };
  std::map<std::pair<unsigned, unsigned>, PairPreference> preferences;
  for (auto &[consumer, consumerGroups] : groupsByConsumer) {
    llvm::sort(consumerGroups);
    consumerGroups.erase(
        std::unique(consumerGroups.begin(), consumerGroups.end()),
        consumerGroups.end());
    for (unsigned lhsPos = 0; lhsPos < consumerGroups.size(); ++lhsPos) {
      unsigned lhs = consumerGroups[lhsPos];
      auto lhsReadiness = summarizeReadiness(lhs, consumer);
      for (unsigned rhsPos = lhsPos + 1; rhsPos < consumerGroups.size();
           ++rhsPos) {
        unsigned rhs = consumerGroups[rhsPos];
        auto rhsReadiness = summarizeReadiness(rhs, consumer);
        if (!lhsReadiness || !rhsReadiness ||
            lhsReadiness->second != rhsReadiness->second ||
            std::abs(lhsReadiness->first - rhsReadiness->first) <
                kMinReadinessGap)
          continue;
        unsigned before =
            lhsReadiness->first > rhsReadiness->first ? lhs : rhs;
        auto key = std::minmax(lhs, rhs);
        PairPreference &preference = preferences[key];
        if (!preference.before)
          preference.before = before;
        else if (*preference.before != before)
          preference.conflict = true;
      }
    }
  }

  SmallVector<unsigned> eligibleGroups;
  DenseMap<unsigned, unsigned> localIndex;
  for (unsigned index = 0; index < groups.size(); ++index)
    if (eligible[index]) {
      localIndex[index] = eligibleGroups.size();
      eligibleGroups.push_back(index);
    }
  if (eligibleGroups.size() < 2)
    return success();

  SmallVector<SmallVector<unsigned>> edges(eligibleGroups.size());
  SmallVector<unsigned> indegree(eligibleGroups.size(), 0);
  for (auto &[pair, preference] : preferences) {
    if (preference.conflict || !preference.before)
      continue;
    unsigned before = *preference.before;
    unsigned after = before == pair.first ? pair.second : pair.first;
    unsigned from = localIndex.lookup(before);
    unsigned to = localIndex.lookup(after);
    if (!llvm::is_contained(edges[from], to)) {
      edges[from].push_back(to);
      ++indegree[to];
    }
  }

  SmallVector<unsigned> ordered;
  SmallVector<bool> emitted(eligibleGroups.size(), false);
  while (ordered.size() != eligibleGroups.size()) {
    std::optional<unsigned> next;
    for (unsigned index = 0; index < eligibleGroups.size(); ++index)
      if (!emitted[index] && indegree[index] == 0) {
        next = index;
        break;
      }
    if (!next)
      return success(); // Cyclic preferences: preserve authored order.
    emitted[*next] = true;
    ordered.push_back(eligibleGroups[*next]);
    for (unsigned successor : edges[*next])
      --indegree[successor];
  }

  // Realize the stable topological order with earlier-only moves. Moving the
  // first member is sufficient: collectGroups keys a group by first sighting.
  Operation *firstRepresentative = representatives[eligibleGroups.front()];
  Operation *lastRepresentative = firstRepresentative;
  for (unsigned group : eligibleGroups) {
    Operation *representative = representatives[group];
    if (representative->isBeforeInBlock(firstRepresentative))
      firstRepresentative = representative;
    if (lastRepresentative->isBeforeInBlock(representative))
      lastRepresentative = representative;
  }
  for (Operation *cursor = firstRepresentative;;
       cursor = cursor->getNextNode()) {
    if (!isSourceFreeAlloc(cursor))
      return success();
    if (cursor == lastRepresentative)
      break;
  }
  for (int index = static_cast<int>(ordered.size()) - 2; index >= 0; --index) {
    Operation *op = representatives[ordered[index]];
    Operation *before = representatives[ordered[index + 1]];
    if (!op->isBeforeInBlock(before)) {
      if (!canMoveEarlier(op, before))
        return success();
      op->moveBefore(before);
    }
  }
  return success();
}

static LogicalResult orderGroups(FuncOp func) {
  bool hasWarpSpecializeLoop = false;
  func.walk([&](scf::ForOp loop) {
    hasWarpSpecializeLoop |= loop->hasAttr(kWarpSpecializeAttrName);
  });
  if (!hasWarpSpecializeLoop)
    return success();
  if (failed(validateNVWSManagedAllocationLocality(
          func, "nvws-order-buffer-groups")))
    return failure();
  for (Block &functionBlock : func.getBody())
    if (failed(orderGroupsInBlock(func, functionBlock)))
      return failure();
  return success();
}

class NVWSOrderBufferGroups
    : public mlir::triton::impl::NVWSOrderBufferGroupsBase<
          NVWSOrderBufferGroups> {
public:
  using mlir::triton::impl::NVWSOrderBufferGroupsBase<
      NVWSOrderBufferGroups>::NVWSOrderBufferGroupsBase;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](FuncOp func) {
      if (failed(orderGroups(func))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    (void)result;
  }
};

} // namespace
