#ifndef NV_DIALECT_HOPPER_TRANSFORMS_WSBARRIERANALYSIS_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_WSBARRIERANALYSIS_H_

#include "CodePartitionUtility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <functional>
#include <optional>

namespace mlir {

// Standard representation of a WS barrier constraint.
//
// The source task is always the partition where the barrier op lives (available
// from async_task_id). The destination is the partition on the other side of
// the channel that this barrier communicates with.
//
// WS barrier metadata is stored under a top-level constraints.WSBarrier key so
// generic barrier constraints can coexist without being treated as WS barriers.
//
// All fields are optional — unknown information is left null and filled in
// by later passes.
struct WSBarrierAttr {
  static constexpr llvm::StringLiteral kKey = "WSBarrier";

  // Destination task ID — the foreign partition this barrier communicates with.
  // Set during insertAsyncComm.
  IntegerAttr dstTask;

  // Task IDs reachable from the destination through the channel adjacency
  // graph (excluding the source). Set after code partitioning via
  // buildChannelGraph() + injectChannelGraph().
  DenseI32ArrayAttr channelGraph;

  // V2 ordered-region metadata. These fields are optional and only used to
  // relax overlapping channelGraph checks when both barriers came from the same
  // parent region and the wait is known to be earlier than the arrive.
  // TODO: Refine parent/region construction to match the region-based design
  // doc exactly; the initial implementation only provides the metadata shape
  // and a conservative same-parent ordering hook.
  IntegerAttr parentId;
  IntegerAttr minRegionId;
  IntegerAttr maxRegionId;

  // Build a constraints DictionaryAttr from the populated fields. Null fields
  // are omitted from the nested WSBarrier dictionary.
  DictionaryAttr build(MLIRContext *ctx) const {
    SmallVector<NamedAttribute> entries;
    if (channelGraph)
      entries.emplace_back(StringAttr::get(ctx, "channelGraph"), channelGraph);
    if (dstTask)
      entries.emplace_back(StringAttr::get(ctx, "dstTask"), dstTask);
    if (parentId)
      entries.emplace_back(StringAttr::get(ctx, "parentId"), parentId);
    if (minRegionId)
      entries.emplace_back(StringAttr::get(ctx, "minRegionId"), minRegionId);
    if (maxRegionId)
      entries.emplace_back(StringAttr::get(ctx, "maxRegionId"), maxRegionId);
    if (entries.empty())
      return {};
    auto wsBarrier = DictionaryAttr::get(ctx, entries);
    SmallVector<NamedAttribute> topLevel;
    topLevel.emplace_back(StringAttr::get(ctx, kKey), wsBarrier);
    return DictionaryAttr::get(ctx, topLevel);
  }

  // Parse from an existing constraints DictionaryAttr.
  static WSBarrierAttr parse(DictionaryAttr dict) {
    WSBarrierAttr attr;
    if (!dict)
      return attr;
    auto wsBarrier = dict.getAs<DictionaryAttr>(kKey);
    if (!wsBarrier)
      return attr;
    attr.dstTask = wsBarrier.getAs<IntegerAttr>("dstTask");
    attr.channelGraph = wsBarrier.getAs<DenseI32ArrayAttr>("channelGraph");
    attr.parentId = wsBarrier.getAs<IntegerAttr>("parentId");
    attr.minRegionId = wsBarrier.getAs<IntegerAttr>("minRegionId");
    attr.maxRegionId = wsBarrier.getAs<IntegerAttr>("maxRegionId");
    return attr;
  }

  // Convenience: create with only dstTask set.
  static WSBarrierAttr forDstTask(MLIRContext *ctx, int taskId) {
    WSBarrierAttr attr;
    attr.dstTask = IntegerAttr::get(IntegerType::get(ctx, 32), taskId);
    return attr;
  }
};

struct WSBarrierRegionInfo {
  int parentId = -1;
  int minRegionId = -1;
  int maxRegionId = -1;
};

struct WSBarrierOpRegionInfo {
  Operation *parent = nullptr;
  int parentId = -1;
  int regionId = -1;
  int numRegions = -1;

  bool isValid() const {
    return parent && parentId >= 0 && regionId >= 0 && numRegions > 0;
  }
};

// Build the WS barrier channel graph for all channels.
//
// For each directed (src, dst) task pair, returns the set of foreign task IDs
// that could interfere with barrier reordering. This is computed as the set of
// task IDs reachable from dst through the channel adjacency graph, excluding
// src (the partition where the barrier lives).
//
// Uses the mapping: default partition = 0, partition p = p + 1.
//
// Example for a GEMM with channels (1<->2), (2<->0), (0<->3):
//   (0, 2) -> [1, 2]     (0, 3) -> [3]
//   (2, 0) -> [0, 3]     (3, 0) -> [0, 1, 2]
static inline DenseMap<std::pair<int, int>, SmallVector<int>>
buildChannelGraph(ArrayRef<Channel *> channels) {
  DenseMap<int, SetVector<int>> adj;
  DenseSet<std::pair<int, int>> channelPairs;
  for (auto *ch : channels) {
    int src = ch->relation.first;
    for (int dst : ch->relation.second) {
      adj[src].insert(dst);
      adj[dst].insert(src);
      channelPairs.insert({src, dst});
      channelPairs.insert({dst, src});
    }
  }

  DenseMap<std::pair<int, int>, SmallVector<int>> result;
  for (auto &pair : channelPairs) {
    int src = pair.first;
    int dst = pair.second;
    if (result.count(pair))
      continue;

    // BFS from dst through the channel adjacency graph, excluding src.
    SetVector<int> reachable;
    SmallVector<int> worklist = {dst};
    while (!worklist.empty()) {
      int cur = worklist.pop_back_val();
      if (cur == src || !reachable.insert(cur))
        continue;
      if (auto it = adj.find(cur); it != adj.end()) {
        for (int neighbor : it->second)
          worklist.push_back(neighbor);
      }
    }

    SmallVector<int> sorted(reachable.begin(), reachable.end());
    llvm::sort(sorted);
    result[pair] = std::move(sorted);
  }
  return result;
}

// Inject the channelGraph into a WSBarrierAttr stored in a constraints dict.
static inline DictionaryAttr
injectChannelGraph(MLIRContext *ctx, DictionaryAttr existing,
                   ArrayRef<int> graphTaskIds,
                   std::optional<int> parentId = std::nullopt,
                   std::optional<int> minRegionId = std::nullopt,
                   std::optional<int> maxRegionId = std::nullopt) {
  auto attr = WSBarrierAttr::parse(existing);
  attr.channelGraph = DenseI32ArrayAttr::get(ctx, graphTaskIds);
  if (parentId)
    attr.parentId = IntegerAttr::get(IntegerType::get(ctx, 32), *parentId);
  if (minRegionId)
    attr.minRegionId =
        IntegerAttr::get(IntegerType::get(ctx, 32), *minRegionId);
  if (maxRegionId)
    attr.maxRegionId =
        IntegerAttr::get(IntegerType::get(ctx, 32), *maxRegionId);
  auto updated = attr.build(ctx);
  if (!existing)
    return updated;

  SmallVector<NamedAttribute> merged;
  bool replaced = false;
  for (NamedAttribute namedAttr : existing) {
    if (namedAttr.getName().getValue() == WSBarrierAttr::kKey) {
      merged.emplace_back(StringAttr::get(ctx, WSBarrierAttr::kKey),
                          updated.getAs<DictionaryAttr>(WSBarrierAttr::kKey));
      replaced = true;
    } else {
      merged.push_back(namedAttr);
    }
  }
  if (!replaced)
    merged.emplace_back(StringAttr::get(ctx, WSBarrierAttr::kKey),
                        updated.getAs<DictionaryAttr>(WSBarrierAttr::kKey));
  return DictionaryAttr::get(ctx, merged);
}

static inline bool isWSBarrierBackwardToken(Operation *op) {
  return isa<triton::nvws::ProducerAcquireOp, triton::nvws::ConsumerReleaseOp>(
      op);
}

static inline bool isWSBarrierForwardToken(Operation *op) {
  return isa<triton::nvws::ProducerCommitOp, triton::nvws::ConsumerWaitOp>(op);
}

static inline Operation *getNearestWSBarrierParent(Operation *op) {
  for (Operation *cur = op->getParentOp(); cur; cur = cur->getParentOp()) {
    if (isa<scf::IfOp>(cur))
      return nullptr;
    if (isa<scf::ForOp, scf::WhileOp>(cur))
      return cur;
    if (isa<triton::FuncOp>(cur))
      return cur;
  }
  return nullptr;
}

static inline Operation *getDirectChildInParent(Operation *op,
                                                Operation *parent) {
  Operation *child = op;
  while (child && child->getParentOp() != parent)
    child = child->getParentOp();
  return child;
}

static inline int getNumOrderedRegions(Block &block) {
  int count = 1;
  for (Operation &op : block.without_terminator())
    if (op.getNumRegions() != 0)
      ++count;
  return count;
}

static inline DenseMap<Block *, int>
getOrderedRegionBlockOffsets(Operation *parent, int &numRegions) {
  DenseMap<Block *, int> blockOffsets;
  numRegions = 0;
  for (Region &region : parent->getRegions()) {
    for (Block &block : region) {
      blockOffsets[&block] = numRegions;
      numRegions += getNumOrderedRegions(block);
    }
  }
  return blockOffsets;
}

static inline std::optional<int>
getBaseRegionInParent(Operation *parent, Operation *op,
                      const DenseMap<Block *, int> &blockOffsets) {
  Operation *child = getDirectChildInParent(op, parent);
  if (!child || child->getBlock()->getParentOp() != parent)
    return std::nullopt;

  auto offsetIt = blockOffsets.find(child->getBlock());
  if (offsetIt == blockOffsets.end())
    return std::nullopt;
  int baseRegion = offsetIt->second;
  for (Operation &cur : child->getBlock()->without_terminator()) {
    if (&cur == child)
      return baseRegion;
    if (cur.getNumRegions() != 0)
      ++baseRegion;
  }
  return std::nullopt;
}

// Build deterministic ordered-region metadata for every operation under
// `scope`. Parent IDs are assigned in DFS order. Region IDs are one-based and
// split each ordered parent into basic-block regions separated by direct child
// region-bearing ops. Ops nested under scf.if get invalid metadata so V2 falls
// back to the V1 channelGraph rule for conditional channels.
static inline DenseMap<Operation *, WSBarrierOpRegionInfo>
buildWSBarrierOpRegionInfo(Operation *scope) {
  struct ParentInfo {
    int parentId = -1;
    int numRegions = -1;
    DenseMap<Block *, int> blockOffsets;
  };

  DenseMap<Operation *, WSBarrierOpRegionInfo> result;
  DenseMap<Operation *, int> parentIds;
  DenseMap<Operation *, ParentInfo> parentInfo;

  auto getParentInfo = [&](Operation *parent) -> ParentInfo & {
    auto [idIt, insertedId] = parentIds.try_emplace(parent, parentIds.size());
    auto [infoIt, insertedInfo] = parentInfo.try_emplace(parent);
    if (insertedInfo) {
      infoIt->second.parentId = idIt->second;
      infoIt->second.blockOffsets =
          getOrderedRegionBlockOffsets(parent, infoIt->second.numRegions);
    }
    return infoIt->second;
  };

  std::function<void(Block &)> assignBlock = [&](Block &block) {
    for (Operation &op : block.without_terminator()) {
      Operation *parent = getNearestWSBarrierParent(&op);
      if (parent) {
        ParentInfo &info = getParentInfo(parent);
        auto baseRegionInfo =
            getBaseRegionInParent(parent, &op, info.blockOffsets);
        if (baseRegionInfo) {
          result[&op] = {parent, info.parentId, *baseRegionInfo + 1,
                         info.numRegions};
        } else {
          result[&op] = {};
        }
      } else {
        result[&op] = {};
      }
      for (Region &region : op.getRegions())
        for (Block &childBlock : region)
          assignBlock(childBlock);
    }
  };

  for (Region &region : scope->getRegions())
    for (Block &block : region)
      assignBlock(block);

  return result;
}

// Initial ordered-region ID assignment for token ops. The parent is the nearest
// valid ordered parent op: scf.for, scf.while, or the containing tt.func.
// Backward resource-reuse tokens are shifted into the second half of the
// parent's region ID space so forward and backward edges are never confused.
static inline DenseMap<Operation *, WSBarrierRegionInfo>
buildWSBarrierRegionInfo(Operation *scope) {
  auto opRegionInfo = buildWSBarrierOpRegionInfo(scope);
  DenseMap<Operation *, WSBarrierRegionInfo> result;
  for (auto &it : opRegionInfo) {
    Operation *op = it.first;
    const WSBarrierOpRegionInfo &info = it.second;
    if (!isWSBarrierForwardToken(op) && !isWSBarrierBackwardToken(op))
      continue;
    if (!info.isValid()) {
      result[op] = {-1, -1};
      continue;
    }
    int regionId = info.regionId;
    if (isWSBarrierBackwardToken(op))
      regionId += info.numRegions;
    result[op] = {info.parentId, regionId, regionId};
  }
  return result;
}

// Complete the V2 ordered-region summary for each WS token. The V1
// channelGraph remains a partition reachability set. V2 only adds ordered
// parent/range information used when two overlapping V1 graphs are compared.
//
// The range is intentionally same-iteration only:
//   1. Start from the token's own ordered region.
//   2. Group sibling channels in the same parent, direction, and region. This
//      makes equivalent same-region channel relationships visible from either
//      participating partition.
//   3. For waits, also union later regions in the waiting partition for the
//      same direction. This evaluates an arrive-past-wait move from the delayed
//      arrive's perspective and prevents proving a move that would cross later
//      logical work in the waiting partition.
// Channels under scf.if keep invalid metadata and therefore use the V1
// disjoint-channelGraph fallback.
static inline DenseMap<Operation *, WSBarrierRegionInfo>
buildWSBarrierOrderedRegionRanges(
    Operation *scope,
    const DenseMap<std::pair<int, int>, SmallVector<int>> &channelGraph) {
  struct TokenNode {
    Operation *op = nullptr;
    int srcTask = -1;
    int dstTask = -1;
    int parentId = -1;
    int regionId = -1;
    bool isBackward = false;
    bool isWait = false;
  };

  auto tokenRegionInfo = buildWSBarrierRegionInfo(scope);
  SmallVector<TokenNode> nodes;
  scope->walk([&](Operation *op) {
    if (!isWSBarrierForwardToken(op) && !isWSBarrierBackwardToken(op))
      return;
    auto constraints = op->getAttrOfType<DictionaryAttr>("constraints");
    auto attr = WSBarrierAttr::parse(constraints);
    if (!attr.dstTask)
      return;
    auto taskIds = getAsyncTaskIds(op);
    if (taskIds.size() != 1)
      return;
    auto regionIt = tokenRegionInfo.find(op);
    if (regionIt == tokenRegionInfo.end())
      return;
    nodes.push_back(
        {op, taskIds[0], static_cast<int>(attr.dstTask.getInt()),
         regionIt->second.parentId, regionIt->second.minRegionId,
         isWSBarrierBackwardToken(op),
         isa<triton::nvws::ProducerAcquireOp, triton::nvws::ConsumerWaitOp>(
             op)});
  });

  DenseMap<Operation *, WSBarrierRegionInfo> result;
  for (const TokenNode &node : nodes) {
    if (node.parentId < 0 || node.regionId < 0) {
      result[node.op] = {-1, -1, -1};
      continue;
    }

    int minRegionId = node.regionId;
    int maxRegionId = node.regionId;
    DenseSet<int> reachableTasks;
    if (auto it = channelGraph.find({node.srcTask, node.dstTask});
        it != channelGraph.end()) {
      reachableTasks.insert(it->second.begin(), it->second.end());
    }

    for (const TokenNode &other : nodes) {
      if (other.parentId != node.parentId ||
          other.isBackward != node.isBackward || other.regionId < 0)
        continue;

      bool sameRegionPeer = other.regionId == node.regionId &&
                            (other.srcTask == node.srcTask ||
                             reachableTasks.contains(other.srcTask) ||
                             reachableTasks.contains(other.dstTask));
      bool laterWaitInSamePartition = node.isWait &&
                                      other.srcTask == node.srcTask &&
                                      other.regionId >= node.regionId;
      if (!sameRegionPeer && !laterWaitInSamePartition)
        continue;

      minRegionId = std::min(minRegionId, other.regionId);
      maxRegionId = std::max(maxRegionId, other.regionId);
    }

    result[node.op] = {node.parentId, minRegionId, maxRegionId};
  }
  return result;
}

// canAdvanceWSBarrier, sinkWSArrives, raiseWSWaits are defined in
// nvidia/hopper/include/Transforms/WSBarrierReorder.h and used by
// the InterleaveTMem pass.

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_WSBARRIERANALYSIS_H_
