#ifndef NV_DIALECT_HOPPER_TRANSFORMS_WSBARRIERANALYSIS_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_WSBARRIERANALYSIS_H_

#include "CodePartitionUtility.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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

  // Build a constraints DictionaryAttr from the populated fields. Null fields
  // are omitted from the nested WSBarrier dictionary.
  DictionaryAttr build(MLIRContext *ctx) const {
    SmallVector<NamedAttribute> entries;
    if (channelGraph)
      entries.emplace_back(StringAttr::get(ctx, "channelGraph"), channelGraph);
    if (dstTask)
      entries.emplace_back(StringAttr::get(ctx, "dstTask"), dstTask);
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
    return attr;
  }

  // Convenience: create with only dstTask set.
  static WSBarrierAttr forDstTask(MLIRContext *ctx, int taskId) {
    WSBarrierAttr attr;
    attr.dstTask = IntegerAttr::get(IntegerType::get(ctx, 32), taskId);
    return attr;
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
static inline DictionaryAttr injectChannelGraph(MLIRContext *ctx,
                                                DictionaryAttr existing,
                                                ArrayRef<int> graphTaskIds) {
  auto attr = WSBarrierAttr::parse(existing);
  attr.channelGraph = DenseI32ArrayAttr::get(ctx, graphTaskIds);
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

// canAdvanceWSBarrier, sinkWSArrives, raiseWSWaits are defined in
// nvidia/hopper/include/Transforms/WSBarrierReorder.h and used by
// the InterleaveTMem pass.

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_WSBARRIERANALYSIS_H_
