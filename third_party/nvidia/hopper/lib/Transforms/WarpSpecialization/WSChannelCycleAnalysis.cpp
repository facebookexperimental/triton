//===- WSChannelCycleAnalysis.cpp - Channel progress analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WSChannelCycleAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>

namespace mlir {

ProtocolEventId ProtocolGraph::addEvent(ProtocolEventKind kind,
                                        const std::string &label) {
  ProtocolEventId id = events.size();
  events.push_back({id, kind, label});
  return id;
}

void ProtocolGraph::addEdge(ProtocolEventId from, ProtocolEventId to,
                            int64_t iterationDistance, ProtocolEdgeKind kind,
                            unsigned channelId) {
  edges.push_back({from, to, iterationDistance, kind, channelId});
}

namespace {

struct SCCState {
  explicit SCCState(const ProtocolGraph &graph)
      : graph(graph), index(graph.events.size(), -1),
        lowLink(graph.events.size(), -1), onStack(graph.events.size(), false),
        outgoing(graph.events.size()) {
    for (auto [edgeId, edge] : llvm::enumerate(graph.edges))
      outgoing[edge.from].push_back(edgeId);
  }

  void visit(ProtocolEventId event) {
    index[event] = nextIndex;
    lowLink[event] = nextIndex++;
    stack.push_back(event);
    onStack[event] = true;

    for (unsigned edgeId : outgoing[event]) {
      ProtocolEventId successor = graph.edges[edgeId].to;
      if (index[successor] == -1) {
        visit(successor);
        lowLink[event] = std::min(lowLink[event], lowLink[successor]);
      } else if (onStack[successor]) {
        lowLink[event] = std::min(lowLink[event], index[successor]);
      }
    }

    if (lowLink[event] != index[event])
      return;

    llvm::SmallVector<ProtocolEventId> component;
    while (true) {
      ProtocolEventId member = stack.pop_back_val();
      onStack[member] = false;
      component.push_back(member);
      if (member == event)
        break;
    }
    llvm::sort(component);
    components.push_back(std::move(component));
  }

  const ProtocolGraph &graph;
  llvm::SmallVector<int> index;
  llvm::SmallVector<int> lowLink;
  llvm::SmallVector<bool> onStack;
  llvm::SmallVector<ProtocolEventId> stack;
  llvm::SmallVector<llvm::SmallVector<unsigned>> outgoing;
  llvm::SmallVector<llvm::SmallVector<ProtocolEventId>> components;
  int nextIndex = 0;
};

static llvm::SmallVector<llvm::SmallVector<ProtocolEventId>>
computeSCCs(const ProtocolGraph &graph) {
  SCCState state(graph);
  for (ProtocolEventId event = 0; event < graph.events.size(); ++event)
    if (state.index[event] == -1)
      state.visit(event);
  llvm::sort(state.components, [](const auto &lhs, const auto &rhs) {
    return lhs.front() < rhs.front();
  });
  return state.components;
}

static void canonicalizeCycle(llvm::SmallVectorImpl<unsigned> &cycle) {
  if (cycle.empty())
    return;
  auto first = std::min_element(cycle.begin(), cycle.end());
  std::rotate(cycle.begin(), first, cycle.end());
}

static ProtocolValidation
validateSCC(const ProtocolGraph &graph,
            llvm::ArrayRef<ProtocolEventId> component) {
  llvm::SmallDenseSet<ProtocolEventId> members(component.begin(),
                                               component.end());
  llvm::SmallVector<unsigned> internalEdges;
  for (auto [edgeId, edge] : llvm::enumerate(graph.edges))
    if (members.contains(edge.from) && members.contains(edge.to))
      internalEdges.push_back(edgeId);

  bool hasCycle = component.size() > 1;
  if (!hasCycle)
    hasCycle = llvm::any_of(internalEdges, [&](unsigned edgeId) {
      const ProtocolEdge &edge = graph.edges[edgeId];
      return edge.from == edge.to;
    });
  if (!hasCycle)
    return {};

  const int64_t eventCount = component.size();
  const int64_t coefficient = eventCount + 1;
  llvm::SmallVector<int64_t> weights;
  weights.reserve(internalEdges.size());
  for (unsigned edgeId : internalEdges) {
    int64_t product;
    if (llvm::MulOverflow(graph.edges[edgeId].iterationDistance, coefficient,
                          product) ||
        product == std::numeric_limits<int64_t>::min()) {
      return {ProtocolStatus::Unsupported,
              {},
              0,
              "iteration distance is too large for cycle analysis"};
    }
    weights.push_back(product - 1);
  }

  llvm::SmallVector<int64_t> distance(graph.events.size(), 0);
  llvm::SmallVector<std::optional<unsigned>> predecessor(graph.events.size());
  std::optional<ProtocolEventId> changed;
  for (int64_t iteration = 0; iteration < eventCount; ++iteration) {
    changed.reset();
    for (auto [localEdgeId, edgeId] : llvm::enumerate(internalEdges)) {
      const ProtocolEdge &edge = graph.edges[edgeId];
      int64_t candidate;
      if (llvm::AddOverflow(distance[edge.from], weights[localEdgeId],
                            candidate)) {
        return {ProtocolStatus::Unsupported,
                {},
                0,
                "cycle-distance accumulation overflowed"};
      }
      if (candidate >= distance[edge.to])
        continue;
      distance[edge.to] = candidate;
      predecessor[edge.to] = edgeId;
      changed = edge.to;
    }
    if (!changed)
      return {};
  }

  ProtocolEventId cycleEvent = *changed;
  for (int64_t i = 0; i < eventCount; ++i) {
    if (!predecessor[cycleEvent])
      return {ProtocolStatus::Unsupported,
              {},
              0,
              "failed to reconstruct a detected channel cycle"};
    cycleEvent = graph.edges[*predecessor[cycleEvent]].from;
  }

  llvm::SmallVector<unsigned> cycle;
  ProtocolEventId cursor = cycleEvent;
  do {
    if (!predecessor[cursor])
      return {ProtocolStatus::Unsupported,
              {},
              0,
              "failed to reconstruct a detected channel cycle"};
    unsigned edgeId = *predecessor[cursor];
    cycle.push_back(edgeId);
    cursor = graph.edges[edgeId].from;
  } while (cursor != cycleEvent && cycle.size() <= component.size());
  if (cursor != cycleEvent)
    return {ProtocolStatus::Unsupported,
            {},
            0,
            "detected channel cycle exceeds its strongly connected component"};

  std::reverse(cycle.begin(), cycle.end());
  canonicalizeCycle(cycle);
  int64_t totalDistance = 0;
  for (unsigned edgeId : cycle) {
    if (llvm::AddOverflow(totalDistance, graph.edges[edgeId].iterationDistance,
                          totalDistance)) {
      return {ProtocolStatus::Unsupported,
              {},
              0,
              "cycle witness distance overflowed"};
    }
  }
  return {ProtocolStatus::Unsafe, std::move(cycle), totalDistance,
          "channel protocol contains a non-positive-distance cycle"};
}

static StringRef stringifyStatus(ProtocolStatus status) {
  switch (status) {
  case ProtocolStatus::Safe:
    return "safe";
  case ProtocolStatus::Unsafe:
    return "unsafe";
  case ProtocolStatus::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown protocol status");
}

} // namespace

ProtocolValidation validateProtocolCycles(const ProtocolGraph &graph) {
  for (auto [eventId, event] : llvm::enumerate(graph.events)) {
    if (event.id != eventId)
      return {ProtocolStatus::Unsupported,
              {},
              0,
              "protocol event IDs must be contiguous"};
  }
  for (const ProtocolEdge &edge : graph.edges) {
    if (edge.from >= graph.events.size() || edge.to >= graph.events.size())
      return {ProtocolStatus::Unsupported,
              {},
              0,
              "protocol edge references an unknown event"};
  }

  std::optional<ProtocolValidation> unsupported;
  for (const auto &component : computeSCCs(graph)) {
    ProtocolValidation result = validateSCC(graph, component);
    if (result.status == ProtocolStatus::Unsafe)
      return result;
    if (result.status == ProtocolStatus::Unsupported && !unsupported)
      unsupported = std::move(result);
  }
  if (unsupported)
    return std::move(*unsupported);
  return {};
}

// This pass is intentionally only a direct lit harness for the common solver.
// Production graph builders do not consume these attributes. The edge array is
// a flat sequence of (from, to, iteration-distance, channel-id) records.
#define GEN_PASS_DEF_NVGPUTESTWSCHANNELCYCLEANALYSIS
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSChannelCycleAnalysisPass
    : public impl::NVGPUTestWSChannelCycleAnalysisBase<
          NVGPUTestWSChannelCycleAnalysisPass> {
public:
  using impl::NVGPUTestWSChannelCycleAnalysisBase<
      NVGPUTestWSChannelCycleAnalysisPass>::NVGPUTestWSChannelCycleAnalysisBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto eventCount =
        module->getAttrOfType<IntegerAttr>("nvws.test.protocol_event_count");
    auto edgeValues =
        module->getAttrOfType<DenseI64ArrayAttr>("nvws.test.protocol_edges");
    if (!eventCount || eventCount.getInt() < 0 || !edgeValues ||
        edgeValues.size() % 4 != 0) {
      module->setAttr("nvws.test.protocol_status",
                      StringAttr::get(module.getContext(), "unsupported"));
      module->setAttr("nvws.test.protocol_reason",
                      StringAttr::get(module.getContext(),
                                      "malformed protocol test graph"));
      return;
    }

    ProtocolGraph graph;
    for (int64_t i = 0; i < eventCount.getInt(); ++i)
      graph.addEvent(ProtocolEventKind::Acquire);
    ArrayRef<int64_t> values = edgeValues.asArrayRef();
    for (size_t i = 0; i < values.size(); i += 4) {
      if (values[i] < 0 || values[i + 1] < 0 || values[i + 3] < 0) {
        module->setAttr("nvws.test.protocol_status",
                        StringAttr::get(module.getContext(), "unsupported"));
        module->setAttr("nvws.test.protocol_reason",
                        StringAttr::get(module.getContext(),
                                        "negative protocol test identifier"));
        return;
      }
      graph.addEdge(values[i], values[i + 1], values[i + 2],
                    ProtocolEdgeKind::TaskOrder, values[i + 3]);
    }

    ProtocolValidation result = validateProtocolCycles(graph);
    module->setAttr(
        "nvws.test.protocol_status",
        StringAttr::get(module.getContext(), stringifyStatus(result.status)));
    if (!result.reason.empty())
      module->setAttr("nvws.test.protocol_reason",
                      StringAttr::get(module.getContext(), result.reason));
    if (!result.cycleEdgeIds.empty()) {
      llvm::SmallVector<int64_t> cycle(result.cycleEdgeIds.begin(),
                                       result.cycleEdgeIds.end());
      module->setAttr("nvws.test.protocol_cycle_edges",
                      DenseI64ArrayAttr::get(module.getContext(), cycle));
      module->setAttr(
          "nvws.test.protocol_cycle_distance",
          IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                           result.cycleIterationDistance));
    }
  }
};

} // namespace mlir
