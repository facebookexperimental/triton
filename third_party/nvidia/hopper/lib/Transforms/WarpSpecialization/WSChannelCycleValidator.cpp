//===- WSChannelCycleValidator.cpp - Post-memory validation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WSChannelCycleValidator.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-channel-cycle-validator"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

enum class ProtocolEventSide {
  BeforeAnchor,
  AfterAnchor,
};

struct ScheduledProtocolEvent {
  ProtocolEventId id;
  AsyncTaskId task;
  Operation *scope;
  Operation *anchor;
  ProtocolEventSide side;
  int64_t stage;
  int64_t cluster;
  unsigned channelId;
  bool ordersFollowingEvents;
};

struct ProtocolTaskTimeline {
  AsyncTaskId task;
  Operation *scope;
  bool cyclic;
  SmallVector<ScheduledProtocolEvent> events;
};

enum class SmemReuseGroupKind {
  Circular,
  DependencyPair,
  LinearChain,
};

struct SmemReuseGroupPlan {
  SmallVector<Channel *> transactions;
  unsigned copies;
  bool finite;
  SmemReuseGroupKind kind;
};

struct TmemCrossPartitionReuseGroupPlan {
  SmallVector<Channel *> chain;
};

struct TmemReuseGroupPlan {
  SmallVector<Channel *> transactions;
  bool finite;
};

struct ChannelEventSet {
  ProtocolEventId acquire;
  ProtocolEventId ready;
  std::optional<ProtocolEventId> reuseWait;
  SmallVector<ProtocolEventId> releases;
};

struct MappedProtocolGraph {
  ProtocolGraph graph;
  SmallVector<unsigned> originalEdgeIds;
};

static bool getScheduleCoordinate(Operation *op, int64_t &stage,
                                  int64_t &cluster) {
  if (!op)
    return false;
  auto stageAttr = op->getAttrOfType<IntegerAttr>("loop.stage");
  auto clusterAttr = op->getAttrOfType<IntegerAttr>("loop.cluster");
  if (!stageAttr || !clusterAttr)
    return false;
  stage = stageAttr.getInt();
  cluster = clusterAttr.getInt();
  return true;
}

static ProtocolTaskTimeline &
getTaskTimeline(SmallVectorImpl<ProtocolTaskTimeline> &timelines,
                AsyncTaskId task, Operation *scope, bool cyclic = true) {
  auto it = llvm::find_if(timelines, [&](const ProtocolTaskTimeline &timeline) {
    return timeline.task == task && timeline.scope == scope;
  });
  if (it != timelines.end()) {
    assert(it->cyclic == cyclic && "inconsistent protocol cadence for task");
    return *it;
  }
  timelines.push_back({task, scope, cyclic, {}});
  return timelines.back();
}

static bool sameEventClass(const ScheduledProtocolEvent &lhs,
                           const ScheduledProtocolEvent &rhs) {
  return lhs.anchor == rhs.anchor && lhs.side == rhs.side;
}

static bool addTaskOrderEdges(ProtocolGraph &graph,
                              ProtocolTaskTimeline &timeline) {
  Block *block =
      timeline.events.empty() ? nullptr : timeline.events[0].anchor->getBlock();
  if (llvm::any_of(timeline.events,
                   [block](const ScheduledProtocolEvent &event) {
                     return event.anchor->getBlock() != block;
                   }))
    return false;
  llvm::stable_sort(timeline.events, [](const ScheduledProtocolEvent &lhs,
                                        const ScheduledProtocolEvent &rhs) {
    if (lhs.cluster != rhs.cluster)
      return lhs.cluster < rhs.cluster;
    if (lhs.anchor != rhs.anchor)
      return lhs.anchor->isBeforeInBlock(rhs.anchor);
    return lhs.side < rhs.side;
  });

  SmallVector<ArrayRef<ScheduledProtocolEvent>> classes;
  for (size_t begin = 0; begin < timeline.events.size();) {
    size_t end = begin + 1;
    while (end < timeline.events.size() &&
           sameEventClass(timeline.events[begin], timeline.events[end]))
      ++end;
    classes.push_back(ArrayRef(timeline.events).slice(begin, end - begin));
    begin = end;
  }
  if (classes.empty())
    return true;

  auto connect = [&](ArrayRef<ScheduledProtocolEvent> from,
                     ArrayRef<ScheduledProtocolEvent> to, bool wraps) {
    int64_t distance = to.front().stage - from.front().stage + (wraps ? 1 : 0);
    for (const ScheduledProtocolEvent &src : from) {
      if (!src.ordersFollowingEvents)
        continue;
      for (const ScheduledProtocolEvent &dst : to)
        graph.addEdge(src.id, dst.id, distance,
                      wraps ? ProtocolEdgeKind::TaskWrap
                            : ProtocolEdgeKind::TaskOrder,
                      src.channelId);
    }
  };
  auto hasOrderingSource = [](ArrayRef<ScheduledProtocolEvent> eventClass) {
    return llvm::any_of(eventClass, [](const ScheduledProtocolEvent &event) {
      return event.ordersFollowingEvents;
    });
  };
  size_t firstTarget = timeline.cyclic ? 0 : 1;
  for (size_t target = firstTarget; target < classes.size(); ++target) {
    size_t source = target == 0 ? classes.size() - 1 : target - 1;
    while (source != target && !hasOrderingSource(classes[source])) {
      if (!timeline.cyclic && source == 0)
        break;
      source = source == 0 ? classes.size() - 1 : source - 1;
    }
    if (!hasOrderingSource(classes[source]))
      continue;
    connect(classes[source], classes[target],
            timeline.cyclic && source >= target);
  }
  return true;
}

static std::optional<int64_t> getConstantTripCount(scf::ForOp loop) {
  APInt lowerBound;
  APInt upperBound;
  APInt stepValue;
  if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lowerBound)) ||
      !matchPattern(loop.getUpperBound(), m_ConstantInt(&upperBound)) ||
      !matchPattern(loop.getStep(), m_ConstantInt(&stepValue)))
    return std::nullopt;

  int64_t lower = lowerBound.getSExtValue();
  int64_t upper = upperBound.getSExtValue();
  int64_t step = stepValue.getSExtValue();
  if (step <= 0)
    return std::nullopt;
  if (upper <= lower)
    return 0;

  int64_t distance;
  if (llvm::SubOverflow(upper, lower, distance))
    return std::nullopt;
  int64_t roundedDistance;
  if (llvm::AddOverflow(distance, step - 1, roundedDistance))
    return std::nullopt;
  return roundedDistance / step;
}

// A nested software pipeline re-enters through its prologue on every
// invocation of the enclosing loop. An async producer and its consumer wait
// in the same early stage can therefore complete the first transaction before
// a later-stage event in that task exists. A negative task-order edge ending
// at that wait carries exactly that prologue credit; treating it as an
// ordinary steady-state backedge creates a false zero-credit cycle.
//
// An acquire normally does not gain this credit. The exception is a task-wrap
// edge into an acquire that has an incoming positive-distance slot-reuse edge:
// the first execution consumes that physical slot's initial empty state. This
// covers an A2 async-reader WAR without crediting D120's non-wrap
// producer-acquire cycle.
static bool hasNestedPrologueCredit(
    const ProtocolGraph &graph, const ProtocolEdge &edge, Operation *scope,
    const DenseMap<ProtocolEventId, ScheduledProtocolEvent> &eventSchedules) {
  bool isNegativeTaskOrder =
      edge.kind == ProtocolEdgeKind::TaskOrder && edge.iterationDistance < 0;
  bool isNonPositiveTaskWrap =
      edge.kind == ProtocolEdgeKind::TaskWrap && edge.iterationDistance <= 0;
  if (!isNegativeTaskOrder && !isNonPositiveTaskWrap)
    return false;
  if (graph.events[edge.to].kind == ProtocolEventKind::Acquire) {
    return edge.kind == ProtocolEdgeKind::TaskWrap &&
           llvm::any_of(graph.edges, [&](const ProtocolEdge &candidate) {
             return candidate.to == edge.to &&
                    candidate.kind == ProtocolEdgeKind::SlotReuse &&
                    candidate.iterationDistance > 0;
           });
  }
  if (graph.events[edge.to].kind != ProtocolEventKind::Wait)
    return false;
  auto waitIt = eventSchedules.find(edge.to);
  if (waitIt == eventSchedules.end() || waitIt->second.scope != scope)
    return false;

  const ScheduledProtocolEvent &wait = waitIt->second;
  return llvm::any_of(eventSchedules, [&](const auto &entry) {
    const ScheduledProtocolEvent &event = entry.second;
    return event.scope == scope && event.channelId == wait.channelId &&
           graph.events[event.id].kind == ProtocolEventKind::Ready &&
           !event.ordersFollowingEvents && event.stage == wait.stage;
  });
}

static int64_t getScopeEdgeDistance(
    const ProtocolGraph &graph, const ProtocolEdge &edge, Operation *scope,
    bool isNested,
    const DenseMap<ProtocolEventId, ScheduledProtocolEvent> &eventSchedules) {
  if (isNested && hasNestedPrologueCredit(graph, edge, scope, eventSchedules))
    return edge.iterationDistance + 1;
  return edge.iterationDistance;
}

static ProtocolValidation
mapValidationToOriginalEdges(ProtocolValidation validation,
                             ArrayRef<unsigned> originalEdgeIds,
                             const ProtocolGraph &originalGraph) {
  if (validation.cycleEdgeIds.empty())
    return validation;
  int64_t totalDistance = 0;
  for (unsigned &edgeId : validation.cycleEdgeIds) {
    assert(edgeId < originalEdgeIds.size() && "missing original edge mapping");
    edgeId = originalEdgeIds[edgeId];
    if (llvm::AddOverflow(totalDistance,
                          originalGraph.edges[edgeId].iterationDistance,
                          totalDistance)) {
      return {ProtocolStatus::Unsupported,
              {},
              0,
              "cycle witness distance overflowed after boundary expansion"};
    }
  }
  validation.cycleIterationDistance = totalDistance;
  return validation;
}

static MappedProtocolGraph buildScopeGraph(
    const ProtocolGraph &graph,
    const DenseMap<ProtocolEventId, Operation *> &eventScopes,
    const DenseMap<ProtocolEventId, ScheduledProtocolEvent> &eventSchedules,
    Operation *scope, bool isNested) {
  MappedProtocolGraph mapped;
  DenseMap<ProtocolEventId, ProtocolEventId> eventMap;
  for (const ProtocolEvent &event : graph.events) {
    auto scopeIt = eventScopes.find(event.id);
    if (scopeIt == eventScopes.end() || scopeIt->second != scope)
      continue;
    eventMap[event.id] = mapped.graph.addEvent(event.kind, event.label);
  }
  for (auto [edgeId, edge] : llvm::enumerate(graph.edges)) {
    auto from = eventMap.find(edge.from);
    auto to = eventMap.find(edge.to);
    if (from == eventMap.end() || to == eventMap.end())
      continue;
    mapped.graph.addEdge(
        from->second, to->second,
        getScopeEdgeDistance(graph, edge, scope, isNested, eventSchedules),
        edge.kind, edge.channelId);
    mapped.originalEdgeIds.push_back(edgeId);
  }
  return mapped;
}

// A scheduled inner loop is a finite transaction domain. Its software
// pipeline enters through a prologue and leaves through a drain before the
// enclosing loop starts the next invocation. Expand that finite domain so an
// edge from event(i) to event(i + distance) exists only when both transaction
// indices are in the same invocation. This removes negative-distance walks
// that terminate at the prologue while retaining zero-distance wait-for
// cycles that fit in the loop body.
static ProtocolValidation validateFiniteNestedScope(
    const ProtocolGraph &graph,
    const DenseMap<ProtocolEventId, Operation *> &eventScopes,
    const DenseMap<ProtocolEventId, ScheduledProtocolEvent> &eventSchedules,
    Operation *scope, int64_t tripCount) {
  if (tripCount <= 0)
    return {};

  SmallVector<ProtocolEventId> scopeEvents;
  DenseMap<ProtocolEventId, unsigned> localEventIds;
  for (const ProtocolEvent &event : graph.events) {
    auto scopeIt = eventScopes.find(event.id);
    if (scopeIt == eventScopes.end() || scopeIt->second != scope)
      continue;
    localEventIds[event.id] = scopeEvents.size();
    scopeEvents.push_back(event.id);
  }

  // Keep the exact domain comfortably above production FA's 128 inner
  // transactions. Adding another supported channel must not silently turn a
  // previously proved cycle into Unsupported merely because the normalized
  // graph acquired a few more endpoint events.
  constexpr int64_t maxExpandedEvents = 16384;
  if (scopeEvents.empty() ||
      tripCount > maxExpandedEvents / int64_t(scopeEvents.size())) {
    return {ProtocolStatus::Unsupported,
            {},
            0,
            "nested loop is too large for exact boundary expansion"};
  }

  MappedProtocolGraph expanded;
  for (int64_t iteration = 0; iteration < tripCount; ++iteration)
    for (ProtocolEventId eventId : scopeEvents) {
      const ProtocolEvent &event = graph.events[eventId];
      expanded.graph.addEvent(event.kind, event.label);
    }

  auto getExpandedEvent = [&](ProtocolEventId eventId, int64_t iteration) {
    return ProtocolEventId(iteration * scopeEvents.size() +
                           localEventIds.lookup(eventId));
  };
  for (auto [edgeId, edge] : llvm::enumerate(graph.edges)) {
    auto from = localEventIds.find(edge.from);
    auto to = localEventIds.find(edge.to);
    if (from == localEventIds.end() || to == localEventIds.end())
      continue;
    // Exact expansion already models the prologue and drain boundaries: a
    // negative-distance edge simply has no target in the first iterations.
    // Do not collapse a prologue-credited -1 edge to zero here, or an A5
    // intra-transaction reuse edge can turn that boundary-crossing dependence
    // into a fabricated zero-distance cycle for a one-iteration loop.
    int64_t iterationDistance = edge.iterationDistance;
    // A zero-distance task wrap into an asynchronously seeded consumer wait
    // executes only after the prologue has issued that channel's first ready
    // event. Account for that one unit of startup credit. Keep negative edges
    // unchanged: the finite-domain boundary itself removes their nonexistent
    // prologue instances (notably for A5).
    if (iterationDistance == 0 &&
        hasNestedPrologueCredit(graph, edge, scope, eventSchedules))
      iterationDistance = 1;
    for (int64_t iteration = 0; iteration < tripCount; ++iteration) {
      int64_t targetIteration;
      if (llvm::AddOverflow(iteration, iterationDistance, targetIteration) ||
          targetIteration < 0 || targetIteration >= tripCount)
        continue;
      expanded.graph.addEdge(getExpandedEvent(edge.from, iteration),
                             getExpandedEvent(edge.to, targetIteration),
                             /*iterationDistance=*/0, edge.kind,
                             edge.channelId);
      expanded.originalEdgeIds.push_back(edgeId);
    }
  }

  return mapValidationToOriginalEdges(validateProtocolCycles(expanded.graph),
                                      expanded.originalEdgeIds, graph);
}

static ProtocolValidation validateProtocolScopes(
    const ProtocolGraph &graph,
    const DenseMap<ProtocolEventId, Operation *> &eventScopes,
    const DenseMap<ProtocolEventId, ScheduledProtocolEvent> &eventSchedules) {
  SmallVector<Operation *> scopes;
  for (const ProtocolEvent &event : graph.events) {
    auto it = eventScopes.find(event.id);
    if (it != eventScopes.end() && !llvm::is_contained(scopes, it->second))
      scopes.push_back(it->second);
  }

  std::optional<ProtocolValidation> unsupported;
  for (Operation *scope : scopes) {
    ProtocolValidation result;
    if (auto loop = dyn_cast<scf::ForOp>(scope)) {
      if (std::optional<int64_t> tripCount = getConstantTripCount(loop)) {
        result = validateFiniteNestedScope(graph, eventScopes, eventSchedules,
                                           scope, *tripCount);
      } else {
        MappedProtocolGraph mapped = buildScopeGraph(
            graph, eventScopes, eventSchedules, scope, /*isNested=*/true);
        result = mapValidationToOriginalEdges(
            validateBoundaryProtocolCycles(mapped.graph),
            mapped.originalEdgeIds, graph);
      }
      // Prologue-credit normalization can lift an original strictly negative
      // recurrence to zero in either the symbolic or exact-expanded scoped
      // graph. Preserve the one-sided boundary meaning after mapping the
      // witness back: the original recurrence still terminates before
      // iteration zero and is not a realizable circular wait.
      if (result.status == ProtocolStatus::Unsafe &&
          result.cycleIterationDistance < 0)
        result = {};
    } else {
      MappedProtocolGraph mapped = buildScopeGraph(
          graph, eventScopes, eventSchedules, scope, /*isNested=*/false);
      result = mapValidationToOriginalEdges(
          validateProtocolCycles(mapped.graph), mapped.originalEdgeIds, graph);
      if (result.status == ProtocolStatus::Unsafe &&
          result.cycleIterationDistance < 0) {
        result.status = ProtocolStatus::Unsupported;
        result.reason =
            "negative-distance cycles require boundary-aware validation";
      }
    }

    if (result.status == ProtocolStatus::Unsafe)
      return result;
    if (result.status == ProtocolStatus::Unsupported && !unsupported)
      unsupported = std::move(result);
  }
  if (unsupported)
    return std::move(*unsupported);
  return {};
}

static SmallVector<SmemReuseGroupPlan>
collectSupportedSmemReuseGroups(ArrayRef<ChannelProtocolPlan> plans,
                                ReuseConfig *reuseConfig,
                                DenseMap<Channel *, unsigned> &groupByChannel) {
  SmallVector<SmemReuseGroupPlan> groups;
  if (!reuseConfig)
    return groups;

  DenseMap<Channel *, const ChannelProtocolPlan *> planByChannel;
  DenseSet<Channel *> ambiguousChannels;
  for (const ChannelProtocolPlan &plan : plans) {
    for (Channel *channel : plan.channels) {
      if (!planByChannel.try_emplace(channel, &plan).second)
        ambiguousChannels.insert(channel);
    }
  }

  for (unsigned groupIndex = 0; groupIndex < reuseConfig->getGroupSize();
       ++groupIndex) {
    ReuseGroup *group = reuseConfig->getGroup(groupIndex);
    if (group->channels.size() <= 1)
      continue;
    Channel *representative = group->channels.front();
    unsigned copies = representative->getNumBuffers();
    if (llvm::any_of(group->channels, [](Channel *channel) {
          return channel->channelKind != DataChannelKind::SMEMAlloc ||
                 channelIsSubtiled(channel);
        }))
      continue;

    SmemReuseGroupKind kind;
    SmallVector<Channel *> transactions;
    if (copies > 1) {
      if (!verifyReuseGroup1(group))
        continue;
      kind = SmemReuseGroupKind::Circular;
      transactions.append(group->channels.begin(), group->channels.end());
      // A1 assigns each member an accumCnt offset in consumer program order.
      // Use that exact logical transaction order for physical-slot ownership.
      llvm::stable_sort(transactions, [](Channel *lhs, Channel *rhs) {
        return lhs->getDstOp()->isBeforeInBlock(rhs->getDstOp());
      });
    } else {
      if (llvm::any_of(group->channels, [](Channel *channel) {
            return channel->getNumBuffers() != 1;
          }))
        continue;
      if (group->channels.size() == 2 && verifyReuseGroup2(group)) {
        kind = SmemReuseGroupKind::DependencyPair;
        auto [early, late] = orderReuseGroup2(group);
        transactions.assign({early, late});
      } else if (verifyReuseGroupN(group)) {
        kind = SmemReuseGroupKind::LinearChain;
        transactions = orderReuseGroupN(group);
        bool consumerOrderMatches = llvm::all_of(
            llvm::seq<size_t>(1, transactions.size()), [&](size_t index) {
              Operation *previous = transactions[index - 1]->getDstOp();
              Operation *current = transactions[index]->getDstOp();
              return previous && current &&
                     previous->getBlock() == current->getBlock() &&
                     previous->isBeforeInBlock(current);
            });
        if (!consumerOrderMatches)
          continue;
      } else {
        continue;
      }
    }

    Operation *scope = nullptr;
    Block *transactionBlock = nullptr;
    std::optional<bool> finite;
    bool supported = true;
    for (Channel *channel : transactions) {
      auto planIt = planByChannel.find(channel);
      if (channel->getNumBuffers() != copies ||
          ambiguousChannels.contains(channel) ||
          planIt == planByChannel.end() ||
          planIt->second->channels.size() != 1) {
        supported = false;
        break;
      }
      const ChannelProtocolPlan &plan = *planIt->second;
      bool planIsFinite = plan.cadence == ChannelProtocolCadence::StraightLine;
      bool planIsLoop = plan.cadence == ChannelProtocolCadence::Loop &&
                        isa_and_nonnull<scf::ForOp>(plan.cadenceScope);
      Operation *source = channel->getSrcOp();
      Operation *destination = channel->getDstOp();
      if ((!planIsFinite && !planIsLoop) || !plan.cadenceScope || !source ||
          !destination || source->getBlock() != destination->getBlock()) {
        supported = false;
        break;
      }
      if (!finite) {
        finite = planIsFinite;
        scope = plan.cadenceScope;
        transactionBlock = destination->getBlock();
      } else if (*finite != planIsFinite || scope != plan.cadenceScope ||
                 transactionBlock != destination->getBlock()) {
        supported = false;
        break;
      }
    }
    if (!supported)
      continue;

    if (llvm::adjacent_find(transactions, [](Channel *lhs, Channel *rhs) {
          return lhs->getDstOp() == rhs->getDstOp();
        }) != transactions.end())
      continue;

    unsigned supportedIndex = groups.size();
    groups.push_back({std::move(transactions), copies, *finite, kind});
    for (Channel *channel : groups.back().transactions)
      groupByChannel[channel] = supportedIndex;
  }
  return groups;
}

static SmallVector<TmemCrossPartitionReuseGroupPlan>
collectSupportedTmemCrossPartitionReuseGroups(
    ArrayRef<ChannelProtocolPlan> plans, ReuseConfig *reuseConfig,
    DenseMap<Channel *, unsigned> &groupByChannel) {
  SmallVector<TmemCrossPartitionReuseGroupPlan> groups;
  if (!reuseConfig)
    return groups;

  DenseMap<Channel *, const ChannelProtocolPlan *> planByChannel;
  DenseSet<Channel *> ambiguousChannels;
  for (const ChannelProtocolPlan &plan : plans) {
    for (Channel *channel : plan.channels) {
      if (!planByChannel.try_emplace(channel, &plan).second)
        ambiguousChannels.insert(channel);
    }
  }

  for (unsigned groupIndex = 0; groupIndex < reuseConfig->getGroupSize();
       ++groupIndex) {
    ReuseGroup *group = reuseConfig->getGroup(groupIndex);
    if (group->channels.size() < 3 ||
        llvm::any_of(group->channels, [](Channel *channel) {
          return channel->channelKind != DataChannelKind::TMEMAlloc ||
                 channel->getNumBuffers() != 1 || channelIsSubtiled(channel);
        }))
      continue;

    // A5 is full-overlap temporal reuse. Distinct starting columns describe
    // A4/A6 spatial packing and require their own protocol models.
    int64_t offset = getTmemBufferOffset(group->channels.front()->getAllocOp());
    if (llvm::any_of(group->channels,
                     [&](Channel *channel) {
                       return getTmemBufferOffset(channel->getAllocOp()) !=
                              offset;
                     }) ||
        !verifyReuseGroupCrossPartition(group))
      continue;

    SmallVector<Channel *> chain = orderReuseGroupChain(group);
    if (chain.size() != group->channels.size())
      continue;

    Operation *scope = nullptr;
    Block *transactionBlock = nullptr;
    bool supported = true;
    for (Channel *channel : chain) {
      auto planIt = planByChannel.find(channel);
      if (ambiguousChannels.contains(channel) ||
          planIt == planByChannel.end() ||
          planIt->second->channels.size() != 1) {
        supported = false;
        break;
      }
      const ChannelProtocolPlan &plan = *planIt->second;
      Operation *source = channel->getSrcOp();
      Operation *destination = channel->getDstOp();
      if (plan.cadence != ChannelProtocolCadence::Loop ||
          !isa_and_nonnull<scf::ForOp>(plan.cadenceScope) || !source ||
          !destination || source->getBlock() != destination->getBlock() ||
          plan.copies != 1 || plan.consumers.empty() ||
          !plan.producerAcquireAnchor || !plan.producerReadyAnchor ||
          triton::gpu::isPhysicalCluster(plan.cadenceScope)) {
        supported = false;
        break;
      }
      auto isScheduledInScope = [&](Operation *anchor) {
        int64_t stage;
        int64_t cluster;
        return anchor &&
               anchor->getParentOfType<scf::ForOp>().getOperation() ==
                   plan.cadenceScope &&
               getScheduleCoordinate(anchor, stage, cluster);
      };
      if (!isScheduledInScope(plan.producerAcquireAnchor) ||
          !isScheduledInScope(plan.producerReadyAnchor) ||
          llvm::any_of(
              plan.consumers, [&](const ChannelConsumerProtocolPlan &consumer) {
                return !isScheduledInScope(consumer.waitScheduleAnchor) ||
                       !isScheduledInScope(consumer.releaseAnchor);
              })) {
        supported = false;
        break;
      }
      if (!scope) {
        scope = plan.cadenceScope;
        transactionBlock = destination->getBlock();
      } else if (scope != plan.cadenceScope ||
                 transactionBlock != destination->getBlock()) {
        supported = false;
        break;
      }
    }
    if (!supported)
      continue;

    unsigned supportedIndex = groups.size();
    groups.push_back({std::move(chain)});
    for (Channel *channel : groups.back().chain)
      groupByChannel[channel] = supportedIndex;
  }
  return groups;
}

static SmallVector<TmemReuseGroupPlan>
collectSupportedTmemReuseGroups(ArrayRef<ChannelProtocolPlan> plans,
                                ReuseConfig *reuseConfig,
                                DenseMap<Channel *, unsigned> &groupByChannel) {
  SmallVector<TmemReuseGroupPlan> groups;
  if (!reuseConfig)
    return groups;

  DenseMap<Channel *, const ChannelProtocolPlan *> planByChannel;
  DenseSet<Channel *> ambiguousChannels;
  for (const ChannelProtocolPlan &plan : plans) {
    for (Channel *channel : plan.channels) {
      if (!planByChannel.try_emplace(channel, &plan).second)
        ambiguousChannels.insert(channel);
    }
  }

  for (unsigned groupIndex = 0; groupIndex < reuseConfig->getGroupSize();
       ++groupIndex) {
    ReuseGroup *group = reuseConfig->getGroup(groupIndex);
    if (group->channels.size() != 2 ||
        llvm::any_of(group->channels,
                     [](Channel *channel) {
                       return channel->channelKind !=
                                  DataChannelKind::TMEMAlloc ||
                              channel->getNumBuffers() != 1 ||
                              channelIsSubtiled(channel);
                     }) ||
        !verifyReuseGroup2(group))
      continue;

    auto [early, late] = orderReuseGroup2(group);
    SmallVector<Channel *> transactions{early, late};

    Operation *scope = nullptr;
    Block *transactionBlock = nullptr;
    std::optional<bool> finite;
    bool supported = true;
    for (Channel *channel : transactions) {
      auto planIt = planByChannel.find(channel);
      if (ambiguousChannels.contains(channel) ||
          planIt == planByChannel.end() ||
          planIt->second->channels.size() != 1) {
        supported = false;
        break;
      }
      const ChannelProtocolPlan &plan = *planIt->second;
      bool planIsFinite = plan.cadence == ChannelProtocolCadence::StraightLine;
      bool planIsLoop = plan.cadence == ChannelProtocolCadence::Loop &&
                        isa_and_nonnull<scf::ForOp>(plan.cadenceScope);
      Operation *source = channel->getSrcOp();
      Operation *destination = channel->getDstOp();
      if ((!planIsFinite && !planIsLoop) || !plan.cadenceScope || !source ||
          !destination || source->getBlock() != destination->getBlock() ||
          plan.copies != 1 || plan.consumers.empty() ||
          !plan.producerAcquireAnchor || !plan.producerReadyAnchor) {
        supported = false;
        break;
      }
      if (!finite) {
        finite = planIsFinite;
        scope = plan.cadenceScope;
        transactionBlock = destination->getBlock();
      } else if (*finite != planIsFinite || scope != plan.cadenceScope ||
                 transactionBlock != destination->getBlock()) {
        supported = false;
        break;
      }
    }
    if (!supported)
      continue;

    unsigned supportedIndex = groups.size();
    groups.push_back({std::move(transactions), *finite});
    for (Channel *channel : groups.back().transactions)
      groupByChannel[channel] = supportedIndex;
  }
  return groups;
}

static void attachAuditAttributes(triton::FuncOp funcOp,
                                  const PostMemoryProtocolAnalysis &analysis) {
  MLIRContext *context = funcOp.getContext();
  funcOp->setAttr("nvws.test.channel_cycle_status",
                  StringAttr::get(context, stringifyProtocolStatus(
                                               analysis.validation.status)));
  funcOp->setAttr("nvws.test.channel_cycle_event_count",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.graph.events.size()));
  funcOp->setAttr("nvws.test.channel_cycle_edge_count",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.graph.edges.size()));
  funcOp->setAttr("nvws.test.channel_cycle_supported_channels",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.supportedChannelCount));
  funcOp->setAttr("nvws.test.channel_cycle_ignored_intra_task_channels",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.ignoredIntraTaskChannelCount));
  funcOp->setAttr("nvws.test.channel_cycle_ignored_finite_guard_channels",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.ignoredFiniteGuardChannelCount));
  funcOp->setAttr("nvws.test.channel_cycle_unsupported_channels",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.unsupportedChannelCount));
  funcOp->setAttr(
      "nvws.test.channel_cycle_staging_reuse_protocols",
      IntegerAttr::get(IntegerType::get(context, 64),
                       analysis.supportedStagingReuseProtocolCount));
  funcOp->setAttr(
      "nvws.test.channel_cycle_finite_staging_reuse_protocols",
      IntegerAttr::get(IntegerType::get(context, 64),
                       analysis.supportedFiniteStagingReuseProtocolCount));
  funcOp->setAttr("nvws.test.channel_cycle_smem_reuse_groups",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.supportedSmemReuseGroupCount));
  funcOp->setAttr("nvws.test.channel_cycle_smem_a1_groups",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.supportedSmemA1GroupCount));
  funcOp->setAttr("nvws.test.channel_cycle_smem_a2_groups",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.supportedSmemA2GroupCount));
  funcOp->setAttr("nvws.test.channel_cycle_smem_a3_groups",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.supportedSmemA3GroupCount));
  funcOp->setAttr("nvws.test.channel_cycle_tmem_a2_groups",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.supportedTmemA2GroupCount));
  funcOp->setAttr("nvws.test.channel_cycle_tmem_a5_groups",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.supportedTmemA5GroupCount));
  if (!analysis.validation.reason.empty())
    funcOp->setAttr("nvws.test.channel_cycle_reason",
                    StringAttr::get(context, analysis.validation.reason));
  if (analysis.validation.cycleEdgeIds.empty())
    return;

  SmallVector<int64_t> cycleEdges(analysis.validation.cycleEdgeIds.begin(),
                                  analysis.validation.cycleEdgeIds.end());
  SmallVector<int64_t> cycleChannels;
  SmallVector<int64_t> cycleDistances;
  for (unsigned edgeId : analysis.validation.cycleEdgeIds) {
    const ProtocolEdge &edge = analysis.graph.edges[edgeId];
    cycleChannels.push_back(edge.channelId);
    cycleDistances.push_back(edge.iterationDistance);
  }
  funcOp->setAttr("nvws.test.channel_cycle_edges",
                  DenseI64ArrayAttr::get(context, cycleEdges));
  funcOp->setAttr("nvws.test.channel_cycle_channels",
                  DenseI64ArrayAttr::get(context, cycleChannels));
  funcOp->setAttr("nvws.test.channel_cycle_edge_distances",
                  DenseI64ArrayAttr::get(context, cycleDistances));
  funcOp->setAttr("nvws.test.channel_cycle_distance",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.validation.cycleIterationDistance));
}

static void
appendValidationManifest(triton::FuncOp funcOp,
                         const PostMemoryProtocolAnalysis &analysis) {
  auto path = triton::tools::getStrEnv("TRITON_WS_SEARCH_MANIFEST");
  if (path.empty())
    return;
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Append);
  if (error)
    return;

  llvm::json::Array cycleChannels;
  llvm::json::Array cycleDistances;
  for (unsigned edgeId : analysis.validation.cycleEdgeIds) {
    const ProtocolEdge &edge = analysis.graph.edges[edgeId];
    cycleChannels.push_back(int64_t(edge.channelId));
    cycleDistances.push_back(edge.iterationDistance);
  }
  llvm::json::Object record{
      {"kind", "validation"},
      {"function", funcOp.getName().str()},
      {"status", stringifyProtocolStatus(analysis.validation.status).str()},
      {"reason", analysis.validation.reason},
      {"event_count", int64_t(analysis.graph.events.size())},
      {"edge_count", int64_t(analysis.graph.edges.size())},
      {"supported_channels", int64_t(analysis.supportedChannelCount)},
      {"ignored_intra_task_channels",
       int64_t(analysis.ignoredIntraTaskChannelCount)},
      {"ignored_finite_guard_channels",
       int64_t(analysis.ignoredFiniteGuardChannelCount)},
      {"unsupported_channels", int64_t(analysis.unsupportedChannelCount)},
      {"finite_staging_reuse_protocols",
       int64_t(analysis.supportedFiniteStagingReuseProtocolCount)},
      {"cycle_distance", analysis.validation.cycleIterationDistance},
      {"cycle_channels", std::move(cycleChannels)},
      {"cycle_edge_distances", std::move(cycleDistances)},
  };
  output << llvm::json::Value(std::move(record)) << "\n";
}

} // namespace

PostMemoryProtocolAnalysis analyzePostMemoryChannelProtocols(
    ArrayRef<ChannelProtocolPlan> plans, ReuseConfig *reuseConfig,
    const StagingReuseProtocolPlan &stagingReusePlan) {
  PostMemoryProtocolAnalysis analysis;
  SmallVector<ProtocolTaskTimeline> timelines;
  DenseMap<ProtocolEventId, Operation *> eventCadenceScopes;
  DenseMap<ProtocolEventId, ScheduledProtocolEvent> eventSchedules;
  DenseMap<Channel *, ChannelEventSet> channelEvents;
  DenseMap<Channel *, unsigned> smemReuseGroupByChannel;
  SmallVector<SmemReuseGroupPlan> smemReuseGroups =
      collectSupportedSmemReuseGroups(plans, reuseConfig,
                                      smemReuseGroupByChannel);
  DenseMap<Channel *, unsigned> tmemReuseGroupByChannel;
  SmallVector<TmemReuseGroupPlan> tmemReuseGroups =
      collectSupportedTmemReuseGroups(plans, reuseConfig,
                                      tmemReuseGroupByChannel);
  DenseMap<Channel *, unsigned> tmemA5GroupByChannel;
  SmallVector<TmemCrossPartitionReuseGroupPlan> tmemA5Groups =
      collectSupportedTmemCrossPartitionReuseGroups(plans, reuseConfig,
                                                    tmemA5GroupByChannel);
  DenseMap<Channel *, const ChannelProtocolPlan *> planByChannel;
  for (const ChannelProtocolPlan &plan : plans)
    for (Channel *member : plan.channels)
      planByChannel.try_emplace(member, &plan);

  // A2 and A5 do not materialize their two physical-slot edges at one
  // acquire. When an explicit wait is needed, code partitioning relocates the
  // late channel's ordinary EMPTY acquire before the early producer (the
  // cross-iteration edge), then inserts a separate wait on the early
  // channel's completion immediately before the late producer (the
  // intra-iteration edge). Preserve those two distinct anchors here; folding
  // both dependencies onto the default late acquire creates a false cycle in
  // pipelined FA backward.
  DenseMap<Channel *, Operation *> relocatedAcquireByChannel;
  DenseMap<Channel *, Channel *> explicitEarlyByLateChannel;
  auto planExplicitEndpointReuse = [&](Channel *early, Channel *late) {
    if (!needExplicitReuseWait(early, late))
      return;
    auto latePlanIt = planByChannel.find(late);
    if (latePlanIt == planByChannel.end())
      return;
    Operation *earlyProducer = early->getSrcOp();
    Operation *lateProducer = latePlanIt->second->headProducer;
    Operation *lateConsumer = late->getDstOp();
    if (!earlyProducer || !lateProducer || !lateConsumer ||
        earlyProducer->getBlock() != lateProducer->getBlock() ||
        lateConsumer->getBlock() != earlyProducer->getBlock() ||
        !appearsBefore(earlyProducer, lateProducer))
      return;
    relocatedAcquireByChannel[late] = earlyProducer;
    explicitEarlyByLateChannel[late] = early;
  };
  for (const SmemReuseGroupPlan &group : smemReuseGroups)
    if (group.kind == SmemReuseGroupKind::DependencyPair)
      planExplicitEndpointReuse(group.transactions.front(),
                                group.transactions.back());
  for (const TmemReuseGroupPlan &group : tmemReuseGroups)
    planExplicitEndpointReuse(group.transactions.front(),
                              group.transactions.back());
  for (const TmemCrossPartitionReuseGroupPlan &group : tmemA5Groups)
    planExplicitEndpointReuse(group.chain.front(), group.chain.back());
  std::string unsupportedReason;
  bool hasUnsupportedProtocol =
      stagingReusePlan.hasReuseTargets() && !stagingReusePlan.isSupported();
  if (hasUnsupportedProtocol)
    unsupportedReason = stagingReusePlan.unsupportedReason;

  for (const ChannelProtocolPlan &plan : plans) {
    Channel *channel = plan.masterChannel;
    Operation *producerAcquireAnchor = plan.producerAcquireAnchor;
    if (auto it = relocatedAcquireByChannel.find(channel);
        it != relocatedAcquireByChannel.end())
      producerAcquireAnchor = it->second;
    auto unsupported = [&](StringRef reason) {
      ++analysis.unsupportedChannelCount;
      LDBG("channel " << (channel ? int64_t(channel->uniqID) : -1)
                      << " unsupported: " << reason << ", cadence="
                      << static_cast<unsigned>(plan.cadence) << ", source="
                      << (channel && channel->getSrcOp()
                              ? channel->getSrcOp()->getName().getStringRef()
                              : StringRef("<none>"))
                      << ", destination="
                      << (channel && channel->getDstOp()
                              ? channel->getDstOp()->getName().getStringRef()
                              : StringRef("<none>")));
      if (unsupportedReason.empty())
        unsupportedReason = reason.str();
    };
    if (!channel || plan.channels.empty() || plan.copies == 0) {
      unsupported("channel plan has incomplete identity or copy depth");
      continue;
    }
    // collectAllocChannels intentionally retains same-task TMA staging rings
    // so memory allocation and store-wait rotation can use their metadata.
    // An empty consumer-task set means code partitioning creates no token or
    // cross-partition wait for this logical channel. It therefore contributes
    // no wait-for edge and is outside the channel-cycle graph, not an
    // unsupported synchronization protocol.
    if (llvm::all_of(plan.channels, [](Channel *member) {
          return member->relation.second.empty();
        })) {
      ++analysis.ignoredIntraTaskChannelCount;
      LDBG("channel " << channel->uniqID
                      << " ignored: no cross-partition consumer task");
      continue;
    }
    // A direct-grid operand-D lifecycle may carry a reverse
    // tmem_load->tmem_store guard even though the CTA executes only one tile.
    // Code partitioning uses its initially-empty token to admit the first
    // store and has no next transaction to wait for the final release. The
    // forward lifecycle channels carry all realizable finite dependencies.
    if (plan.channels.size() == 1 &&
        channel->channelKind == DataChannelKind::TMEMAlloc &&
        static_cast<ttng::TmemAllocChannel *>(channel)->isSameIterGuard &&
        plan.cadence == ChannelProtocolCadence::StraightLine) {
      ++analysis.ignoredFiniteGuardChannelCount;
      LDBG("channel " << channel->uniqID
                      << " ignored: finite operand-D guard has no wrap");
      continue;
    }
    bool isStagingReuseChannel =
        llvm::any_of(plan.channels, [&](Channel *member) {
          Operation *alloc = member->getAllocOp();
          if (!alloc)
            return false;
          auto bufferId = alloc->getAttrOfType<IntegerAttr>("buffer.id");
          return bufferId &&
                 stagingReusePlan.affectedBufferIds.contains(bufferId.getInt());
        });
    if (isStagingReuseChannel && !stagingReusePlan.isSupported()) {
      unsupported(stagingReusePlan.unsupportedReason);
      continue;
    }
    bool isInnerLoopCadence = plan.cadence == ChannelProtocolCadence::Loop &&
                              isa_and_nonnull<scf::ForOp>(plan.cadenceScope);
    bool isOuterToInnerCadence =
        plan.cadence == ChannelProtocolCadence::OuterToInnerLoop &&
        isa_and_nonnull<scf::ForOp>(plan.cadenceScope) &&
        isa_and_nonnull<scf::ForOp>(plan.innerCadenceScope);
    bool isInnerToOuterCadence =
        plan.cadence == ChannelProtocolCadence::InnerToOuterLoop &&
        isa_and_nonnull<scf::ForOp>(plan.cadenceScope) &&
        isa_and_nonnull<scf::ForOp>(plan.innerCadenceScope);
    bool isNestedBoundaryCadence =
        isOuterToInnerCadence || isInnerToOuterCadence;
    bool isOutsideToInnerCadence =
        plan.cadence == ChannelProtocolCadence::OutsideToInnerLoop &&
        isa_and_nonnull<triton::FuncOp>(plan.cadenceScope) &&
        isa_and_nonnull<scf::ForOp>(plan.innerCadenceScope);
    bool isInnerToOutsideCadence =
        plan.cadence == ChannelProtocolCadence::InnerToOutsideLoop &&
        isa_and_nonnull<triton::FuncOp>(plan.cadenceScope) &&
        isa_and_nonnull<scf::ForOp>(plan.innerCadenceScope);
    bool isDirectBoundaryCadence =
        isOutsideToInnerCadence || isInnerToOutsideCadence;
    auto smemReuseGroupIt = smemReuseGroupByChannel.find(channel);
    const SmemReuseGroupPlan *smemReuseGroup =
        smemReuseGroupIt == smemReuseGroupByChannel.end()
            ? nullptr
            : &smemReuseGroups[smemReuseGroupIt->second];
    auto tmemA5GroupIt = tmemA5GroupByChannel.find(channel);
    const TmemCrossPartitionReuseGroupPlan *tmemA5Group =
        tmemA5GroupIt == tmemA5GroupByChannel.end()
            ? nullptr
            : &tmemA5Groups[tmemA5GroupIt->second];
    auto tmemReuseGroupIt = tmemReuseGroupByChannel.find(channel);
    const TmemReuseGroupPlan *tmemReuseGroup =
        tmemReuseGroupIt == tmemReuseGroupByChannel.end()
            ? nullptr
            : &tmemReuseGroups[tmemReuseGroupIt->second];
    bool isFiniteSmemReuseCadence =
        smemReuseGroup && smemReuseGroup->finite &&
        plan.cadence == ChannelProtocolCadence::StraightLine;
    bool isFiniteTmemReuseCadence =
        tmemReuseGroup && tmemReuseGroup->finite &&
        plan.cadence == ChannelProtocolCadence::StraightLine;
    bool isFiniteStraightLineCadence =
        plan.cadence == ChannelProtocolCadence::StraightLine &&
        plan.cadenceScope && !isa<scf::ForOp, scf::WhileOp>(plan.cadenceScope);
    bool isFiniteCadence =
        isFiniteSmemReuseCadence || isFiniteTmemReuseCadence ||
        isFiniteStraightLineCadence || isDirectBoundaryCadence;
    if (!isInnerLoopCadence && !isNestedBoundaryCadence &&
        !isDirectBoundaryCadence && !isFiniteSmemReuseCadence &&
        !isFiniteTmemReuseCadence && !isFiniteStraightLineCadence) {
      unsupported("only scf.for loop-cadence channels are supported");
      continue;
    }
    if (triton::gpu::isPhysicalCluster(plan.cadenceScope) ||
        (plan.innerCadenceScope &&
         triton::gpu::isPhysicalCluster(plan.innerCadenceScope))) {
      unsupported("multi-CTA channel protocols are not yet supported");
      continue;
    }
    if (isOuterToInnerCadence &&
        (!producerAcquireAnchor ||
         plan.innerCadenceScope->getBlock() !=
             producerAcquireAnchor->getBlock() ||
         llvm::any_of(plan.actualConsumerOps, [&](Operation *consumer) {
           return consumer->getParentOfType<scf::ForOp>().getOperation() !=
                  plan.innerCadenceScope;
         }))) {
      unsupported(
          "outer-to-inner channels require one direct nested consumer loop");
      continue;
    }
    if (isInnerToOuterCadence &&
        (!producerAcquireAnchor || !plan.producerReadyAnchor ||
         plan.innerCadenceScope->getBlock() !=
             producerAcquireAnchor->getBlock() ||
         plan.innerCadenceScope->getBlock() !=
             plan.producerReadyAnchor->getBlock() ||
         llvm::any_of(
             plan.channels,
             [&](Channel *member) {
               Operation *producer = member->getSrcOp();
               return !producer ||
                      producer->getParentOfType<scf::ForOp>().getOperation() !=
                          plan.innerCadenceScope;
             }) ||
         llvm::any_of(plan.actualConsumerOps, [&](Operation *consumer) {
           return consumer->getParentOfType<scf::ForOp>().getOperation() !=
                  plan.cadenceScope;
         }))) {
      unsupported(
          "inner-to-outer channels require one direct nested producer loop");
      continue;
    }
    if (isDirectBoundaryCadence &&
        (!producerAcquireAnchor || !plan.producerReadyAnchor ||
         plan.innerCadenceScope->getBlock() !=
             producerAcquireAnchor->getBlock() ||
         plan.innerCadenceScope->getBlock() !=
             plan.producerReadyAnchor->getBlock())) {
      unsupported("direct-grid boundary channel has inconsistent endpoints");
      continue;
    }
    bool isSmemChannel = channel->channelKind == DataChannelKind::SMEMAlloc;
    bool isInnerToOuterAccumulator =
        isInnerToOuterCadence &&
        channel->channelKind == DataChannelKind::TMEMAlloc &&
        plan.channels.size() == 1 && plan.copies == 1 &&
        isa<ttng::MMAv5OpInterface>(channel->getSrcOp()) &&
        isa<ttng::TMEMLoadOp>(channel->getDstOp()) &&
        static_cast<ttng::TmemAllocChannel *>(channel)->isOperandD;
    bool isOrdinaryLoopTmemResult =
        isInnerLoopCadence &&
        channel->channelKind == DataChannelKind::TMEMAlloc &&
        plan.channels.size() == 1 && plan.copies == 1 &&
        isa<ttng::MMAv5OpInterface>(channel->getSrcOp()) &&
        isa<ttng::TMEMLoadOp>(channel->getDstOp());
    bool isFiniteOperandD =
        (isDirectBoundaryCadence || isFiniteStraightLineCadence) &&
        channel->channelKind == DataChannelKind::TMEMAlloc &&
        plan.channels.size() == 1 && plan.copies == 1 &&
        static_cast<ttng::TmemAllocChannel *>(channel)->isOperandD;
    bool isTmemChannel =
        channel->channelKind == DataChannelKind::TMEMAlloc &&
        (tmemReuseGroup || tmemA5Group || isInnerToOuterAccumulator ||
         isOrdinaryLoopTmemResult || isFiniteOperandD);
    if (!isSmemChannel && !isTmemChannel) {
      unsupported("ordinary TMEM protocols are not yet supported");
      continue;
    }
    if (llvm::any_of(plan.channels, [&](Channel *member) {
          return member->channelKind != channel->channelKind ||
                 member->relation.first != channel->relation.first ||
                 member->getNumBuffers() != plan.copies;
        })) {
      unsupported("grouped producers do not share one channel protocol");
      continue;
    }
    if (reuseConfig && llvm::any_of(plan.channels, [&](Channel *member) {
          return channelInReuseGroup(member, reuseConfig,
                                     /*reuseBarrier=*/false) >= 0 &&
                 !smemReuseGroupByChannel.contains(member) &&
                 !tmemReuseGroupByChannel.contains(member) &&
                 !tmemA5GroupByChannel.contains(member);
        })) {
      unsupported("physical reuse-group protocols are not yet supported");
      continue;
    }
    if (plan.consumers.empty() || !producerAcquireAnchor ||
        !plan.producerReadyAnchor) {
      unsupported("channel plan has incomplete protocol endpoints");
      continue;
    }

    unsigned channelId = channel->uniqID;
    std::string label = "channel " + std::to_string(channelId);
    LDBG(label << " buffer " << channel->getAllocOp()->getAttr("buffer.id")
               << " copies " << plan.copies << " producer task "
               << channel->relation.first);
    size_t eventStart = analysis.graph.events.size();
    ProtocolEventId acquire =
        analysis.graph.addEvent(ProtocolEventKind::Acquire, label + " acquire");
    ProtocolEventId ready =
        analysis.graph.addEvent(ProtocolEventKind::Ready, label + " ready");

    SmallVector<ScheduledProtocolEvent> pendingEvents;
    auto stageEvent = [&](ProtocolEventId id, AsyncTaskId task,
                          Operation *anchor, ProtocolEventSide side,
                          bool ordersFollowingEvents = true) {
      int64_t stage;
      int64_t cluster;
      if (!anchor)
        return false;
      if (isNestedBoundaryCadence || isDirectBoundaryCadence ||
          isFiniteSmemReuseCadence || isFiniteTmemReuseCadence ||
          isFiniteStraightLineCadence) {
        Block *expectedBlock = nullptr;
        if (isNestedBoundaryCadence || isDirectBoundaryCadence)
          expectedBlock = plan.innerCadenceScope->getBlock();
        else if (smemReuseGroup)
          expectedBlock =
              smemReuseGroup->transactions.front()->getDstOp()->getBlock();
        else if (tmemReuseGroup)
          expectedBlock =
              tmemReuseGroup->transactions.front()->getDstOp()->getBlock();
        else
          expectedBlock = producerAcquireAnchor->getBlock();
        if (anchor->getBlock() != expectedBlock)
          return false;
        // Cross-scope events form an outer-iteration protocol. A finite A1
        // group instead uses its straight-line block as one transaction
        // sequence. Neither shape has inner-pipeline schedule coordinates at
        // these anchors.
        stage = 0;
        cluster = 0;
      } else {
        if (anchor->getParentOfType<scf::ForOp>().getOperation() !=
                plan.cadenceScope ||
            !getScheduleCoordinate(anchor, stage, cluster))
          return false;
      }
      LDBG("channel " << channelId << " event " << id << " task " << task
                      << " stage " << stage << " cluster " << cluster
                      << (side == ProtocolEventSide::BeforeAnchor ? " before "
                                                                  : " after ")
                      << anchor->getName());
      pendingEvents.push_back({id, task, plan.cadenceScope, anchor, side, stage,
                               cluster, channelId, ordersFollowingEvents});
      return true;
    };
    if (!stageEvent(acquire, channel->relation.first, producerAcquireAnchor,
                    ProtocolEventSide::BeforeAnchor) ||
        !stageEvent(ready, channel->relation.first, plan.producerReadyAnchor,
                    ProtocolEventSide::AfterAnchor,
                    !plan.producerReadyIsAsync)) {
      analysis.graph.events.resize(eventStart);
      unsupported("producer endpoint lacks a loop schedule coordinate");
      continue;
    }

    // A2/A5 insertion materializes a second wait immediately before the late
    // producer. It is distinct from both the late channel's relocated EMPTY
    // acquire and its asynchronous ready completion. Keeping this as an
    // explicit scheduled event is essential: attaching the dependency to the
    // ready event loses the producer-side blocking point and can hide a real
    // cross-task cycle.
    std::optional<ProtocolEventId> reuseWait;
    if (explicitEarlyByLateChannel.contains(channel)) {
      reuseWait = analysis.graph.addEvent(ProtocolEventKind::Wait,
                                          label + " reuse wait");
      if (!stageEvent(*reuseWait, channel->relation.first, plan.headProducer,
                      ProtocolEventSide::BeforeAnchor)) {
        analysis.graph.events.resize(eventStart);
        unsupported("reuse wait lacks a loop schedule coordinate");
        continue;
      }
    }

    struct ConsumerEvents {
      ProtocolEventId wait;
      ProtocolEventId release;
    };
    SmallVector<ConsumerEvents> consumerEvents;
    bool complete = true;
    for (const ChannelConsumerProtocolPlan &consumer : plan.consumers) {
      ProtocolEventId wait = analysis.graph.addEvent(
          ProtocolEventKind::Wait,
          label + " wait task " + std::to_string(consumer.task));
      ProtocolEventId release = analysis.graph.addEvent(
          ProtocolEventKind::Release,
          label + " release task " + std::to_string(consumer.task));
      Operation *waitAnchor = (isOuterToInnerCadence || isOutsideToInnerCadence)
                                  ? plan.innerCadenceScope
                                  : consumer.waitScheduleAnchor;
      Operation *releaseAnchor =
          (isOuterToInnerCadence || isOutsideToInnerCadence)
              ? plan.innerCadenceScope
              : consumer.releaseAnchor;
      if (!stageEvent(wait, consumer.task, waitAnchor,
                      ProtocolEventSide::BeforeAnchor) ||
          !stageEvent(release, consumer.task, releaseAnchor,
                      ProtocolEventSide::AfterAnchor,
                      !consumer.releaseIsAsync)) {
        complete = false;
        break;
      }
      consumerEvents.push_back({wait, release});
    }
    if (!complete) {
      analysis.graph.events.resize(eventStart);
      unsupported("consumer endpoint lacks a loop schedule coordinate");
      continue;
    }

    for (ProtocolEventId event = eventStart;
         event < analysis.graph.events.size(); ++event)
      eventCadenceScopes[event] = plan.cadenceScope;

    auto getEventStage = [&](ProtocolEventId id) {
      auto event = llvm::find_if(pendingEvents,
                                 [id](const ScheduledProtocolEvent &candidate) {
                                   return candidate.id == id;
                                 });
      assert(event != pendingEvents.end() &&
             "missing scheduled protocol event");
      return event->stage;
    };
    auto getTaskSpanDistance = [&](ProtocolEventId from, ProtocolEventId to) {
      return getEventStage(to) - getEventStage(from);
    };

    for (const ScheduledProtocolEvent &event : pendingEvents) {
      getTaskTimeline(timelines, event.task, event.scope,
                      /*cyclic=*/!isFiniteCadence)
          .events.push_back(event);
      eventSchedules.try_emplace(event.id, event);
    }
    analysis.graph.addEdge(acquire, ready, getTaskSpanDistance(acquire, ready),
                           ProtocolEdgeKind::TaskOrder, channelId);
    ChannelEventSet &events = channelEvents[channel];
    events.acquire = acquire;
    events.ready = ready;
    events.reuseWait = reuseWait;
    for (const ConsumerEvents &consumer : consumerEvents) {
      analysis.graph.addEdge(ready, consumer.wait, 0,
                             ProtocolEdgeKind::DataReady, channelId);
      analysis.graph.addEdge(
          consumer.wait, consumer.release,
          getTaskSpanDistance(consumer.wait, consumer.release),
          (isNestedBoundaryCadence || isDirectBoundaryCadence)
              ? ProtocolEdgeKind::ControlFlow
              : ProtocolEdgeKind::TaskOrder,
          channelId);
      events.releases.push_back(consumer.release);
      // A2/A3 retain their ordinary per-channel one-copy tokens. Their extra
      // physical-alias dependencies are added below. A1 shares the circular
      // barrier itself, so its member-local edge is replaced completely.
      if (!isFiniteCadence &&
          (!smemReuseGroup ||
           smemReuseGroup->kind != SmemReuseGroupKind::Circular))
        analysis.graph.addEdge(consumer.release, acquire, plan.copies,
                               ProtocolEdgeKind::SlotReuse, channelId);
    }
    ++analysis.supportedChannelCount;
  }

  for (const SmemReuseGroupPlan &group : smemReuseGroups) {
    if (llvm::any_of(group.transactions, [&](Channel *channel) {
          return !channelEvents.contains(channel);
        }))
      continue;

    ++analysis.supportedSmemReuseGroupCount;
    switch (group.kind) {
    case SmemReuseGroupKind::Circular:
      ++analysis.supportedSmemA1GroupCount;
      break;
    case SmemReuseGroupKind::DependencyPair:
      ++analysis.supportedSmemA2GroupCount;
      break;
    case SmemReuseGroupKind::LinearChain:
      ++analysis.supportedSmemA3GroupCount;
      break;
    }
    if (group.kind == SmemReuseGroupKind::DependencyPair) {
      Channel *early = group.transactions.front();
      Channel *late = group.transactions.back();
      if (explicitEarlyByLateChannel.lookup(late) == early) {
        const ChannelEventSet &earlyEvents = channelEvents.lookup(early);
        const ChannelEventSet &lateEvents = channelEvents.lookup(late);
        assert(lateEvents.reuseWait &&
               "explicit SMEM A2 dependency is missing its producer wait");
        for (ProtocolEventId release : earlyEvents.releases)
          analysis.graph.addEdge(release, *lateEvents.reuseWait,
                                 /*iterationDistance=*/0,
                                 ProtocolEdgeKind::SlotReuse, late->uniqID);
      }
      continue;
    }
    size_t transactionCount = group.transactions.size();
    for (size_t target = 0; target < transactionCount; ++target) {
      if (group.finite && target < group.copies)
        continue;

      size_t predecessor;
      int64_t iterationDistance;
      if (group.finite) {
        // There is no predecessor for the first K direct-grid transactions:
        // they consume the ring's initially empty slots. Later transactions
        // reuse the slot held by the transaction exactly K positions earlier.
        predecessor = target - group.copies;
        iterationDistance = 0;
      } else {
        // For a cyclic N-transaction sequence, solve
        //   predecessorIteration * N + predecessor + K
        //       = targetIteration * N + target
        // for the previous logical transaction that owned this physical slot.
        int64_t unwrappedPredecessor = int64_t(target) - int64_t(group.copies);
        int64_t predecessorRemainder =
            unwrappedPredecessor % int64_t(transactionCount);
        if (predecessorRemainder < 0)
          predecessorRemainder += transactionCount;
        predecessor = predecessorRemainder;
        iterationDistance =
            (int64_t(predecessor) + int64_t(group.copies) - int64_t(target)) /
            int64_t(transactionCount);
      }

      Channel *sourceChannel = group.transactions[predecessor];
      Channel *targetChannel = group.transactions[target];
      const ChannelEventSet &sourceEvents = channelEvents.lookup(sourceChannel);
      const ChannelEventSet &targetEvents = channelEvents.lookup(targetChannel);
      for (ProtocolEventId release : sourceEvents.releases)
        analysis.graph.addEdge(release, targetEvents.acquire, iterationDistance,
                               ProtocolEdgeKind::SlotReuse,
                               targetChannel->uniqID);
    }
  }

  for (const TmemReuseGroupPlan &group : tmemReuseGroups) {
    if (llvm::any_of(group.transactions, [&](Channel *channel) {
          return !channelEvents.contains(channel);
        }))
      continue;

    ++analysis.supportedTmemA2GroupCount;
    Channel *early = group.transactions.front();
    Channel *late = group.transactions.back();
    const ChannelEventSet &earlyEvents = channelEvents.lookup(early);
    const ChannelEventSet &lateEvents = channelEvents.lookup(late);
    if (explicitEarlyByLateChannel.lookup(late) == early) {
      assert(lateEvents.reuseWait &&
             "explicit TMEM A2 dependency is missing its producer wait");
      for (ProtocolEventId release : earlyEvents.releases)
        analysis.graph.addEdge(release, *lateEvents.reuseWait,
                               /*iterationDistance=*/0,
                               ProtocolEdgeKind::SlotReuse, late->uniqID);
    }
  }

  for (const TmemCrossPartitionReuseGroupPlan &group : tmemA5Groups) {
    if (llvm::any_of(group.chain, [&](Channel *channel) {
          return !channelEvents.contains(channel);
        }))
      continue;

    ++analysis.supportedTmemA5GroupCount;
    Channel *first = group.chain.front();
    Channel *last = group.chain.back();
    const ChannelEventSet &firstEvents = channelEvents.lookup(first);
    const ChannelEventSet &lastEvents = channelEvents.lookup(last);

    // Insertion applies A2 synchronization to the A5 chain endpoints. The
    // first channel's read gates the last writer within a transaction, and the
    // last channel's read gates the first writer of the next transaction.
    // Intermediate ownership transitions are already represented by the
    // ordinary task timelines and data-ready edges that proved the chain.
    if (explicitEarlyByLateChannel.lookup(last) == first) {
      assert(lastEvents.reuseWait &&
             "explicit TMEM A5 dependency is missing its producer wait");
      for (ProtocolEventId release : firstEvents.releases)
        analysis.graph.addEdge(release, *lastEvents.reuseWait,
                               /*iterationDistance=*/0,
                               ProtocolEdgeKind::SlotReuse, last->uniqID);
    }
  }

  if (stagingReusePlan.isSupported()) {
    analysis.supportedStagingReuseProtocolCount = 1;
    if (stagingReusePlan.finiteSingleTile)
      analysis.supportedFiniteStagingReuseProtocolCount = 1;
    if (stagingReusePlan.needsCrossTaskWar()) {
      unsigned channelId = stagingReusePlan.diagnosticChannelId;
      ProtocolEventId acquire = analysis.graph.addEvent(
          ProtocolEventKind::Acquire, "staging reuse acquire");
      ProtocolEventId release = analysis.graph.addEvent(
          ProtocolEventKind::Release, "staging reuse release");
      ScheduledProtocolEvent acquireEvent{acquire,
                                          stagingReusePlan.loadTask,
                                          stagingReusePlan.outerLoop,
                                          stagingReusePlan.acquireAnchor,
                                          ProtocolEventSide::BeforeAnchor,
                                          /*stage=*/0,
                                          /*cluster=*/0,
                                          channelId,
                                          /*ordersFollowingEvents=*/true};
      ScheduledProtocolEvent releaseEvent{release,
                                          stagingReusePlan.drainedStoreTask,
                                          stagingReusePlan.outerLoop,
                                          stagingReusePlan.releaseAnchor,
                                          ProtocolEventSide::BeforeAnchor,
                                          /*stage=*/0,
                                          /*cluster=*/0,
                                          channelId,
                                          /*ordersFollowingEvents=*/true};
      getTaskTimeline(timelines, acquireEvent.task, acquireEvent.scope)
          .events.push_back(acquireEvent);
      getTaskTimeline(timelines, releaseEvent.task, releaseEvent.scope)
          .events.push_back(releaseEvent);
      eventCadenceScopes[acquire] = stagingReusePlan.outerLoop;
      eventCadenceScopes[release] = stagingReusePlan.outerLoop;
      eventSchedules[acquire] = acquireEvent;
      eventSchedules[release] = releaseEvent;
      analysis.graph.addEdge(release, acquire, /*iterationDistance=*/1,
                             ProtocolEdgeKind::SlotReuse, channelId);
    }
  }

  for (ProtocolTaskTimeline &timeline : timelines) {
    if (addTaskOrderEdges(analysis.graph, timeline))
      continue;
    ++analysis.unsupportedChannelCount;
    if (unsupportedReason.empty())
      unsupportedReason = "task protocol events span multiple blocks";
  }
  LLVM_DEBUG({
    for (auto [edgeId, edge] : llvm::enumerate(analysis.graph.edges)) {
      DBGS() << "protocol edge " << edgeId << " "
             << analysis.graph.events[edge.from].label << " -> "
             << analysis.graph.events[edge.to].label << " kind "
             << static_cast<unsigned>(edge.kind) << " distance "
             << edge.iterationDistance << " channel " << edge.channelId << "\n";
    }
  });
  analysis.validation = validateProtocolScopes(
      analysis.graph, eventCadenceScopes, eventSchedules);
  if (analysis.validation.status != ProtocolStatus::Unsafe &&
      hasUnsupportedProtocol) {
    // A malformed physical-alias protocol is more actionable than a separate
    // supported component whose dynamic boundary analysis is incomplete.
    analysis.validation.status = ProtocolStatus::Unsupported;
    analysis.validation.reason = stagingReusePlan.unsupportedReason;
  } else if (analysis.validation.status == ProtocolStatus::Safe &&
             analysis.unsupportedChannelCount != 0) {
    analysis.validation.status = ProtocolStatus::Unsupported;
    analysis.validation.reason = unsupportedReason;
  }
  return analysis;
}

LogicalResult validatePostMemoryChannelProtocols(
    triton::FuncOp funcOp, ArrayRef<Channel *> orderedChannels,
    const DenseMap<Channel *, SmallVector<Channel *>> &consumerGroups,
    ReuseConfig *reuseConfig, bool emitAuditAttributes) {
  PostDominanceInfo postDominance(funcOp);
  SmallVector<ChannelProtocolPlan, 0> plans;
  for (Channel *channel : orderedChannels) {
    auto groupIt = consumerGroups.find(channel);
    if (groupIt == consumerGroups.end())
      continue;
    plans.push_back(buildChannelProtocolPlan(groupIt->second, postDominance));
  }

  StagingReuseProtocolPlan stagingReusePlan =
      buildStagingReuseProtocolPlan(funcOp, orderedChannels);

  PostMemoryProtocolAnalysis analysis =
      analyzePostMemoryChannelProtocols(plans, reuseConfig, stagingReusePlan);
  appendValidationManifest(funcOp, analysis);
  LDBG("post-memory channel-cycle audit: "
       << stringifyProtocolStatus(analysis.validation.status)
       << ", events=" << analysis.graph.events.size()
       << ", edges=" << analysis.graph.edges.size()
       << ", supported=" << analysis.supportedChannelCount
       << ", ignored-intra-task=" << analysis.ignoredIntraTaskChannelCount
       << ", ignored-finite-guard=" << analysis.ignoredFiniteGuardChannelCount
       << ", unsupported=" << analysis.unsupportedChannelCount
       << ", staging-reuse=" << analysis.supportedStagingReuseProtocolCount
       << ", finite-staging-reuse="
       << analysis.supportedFiniteStagingReuseProtocolCount
       << ", smem-reuse-groups=" << analysis.supportedSmemReuseGroupCount
       << ", tmem-a2-groups=" << analysis.supportedTmemA2GroupCount
       << ", tmem-a5-groups=" << analysis.supportedTmemA5GroupCount);
  LLVM_DEBUG({
    for (unsigned edgeId : analysis.validation.cycleEdgeIds) {
      const ProtocolEdge &edge = analysis.graph.edges[edgeId];
      DBGS() << "cycle edge " << edgeId << " "
             << analysis.graph.events[edge.from].label << " -> "
             << analysis.graph.events[edge.to].label << " kind "
             << static_cast<unsigned>(edge.kind) << " distance "
             << edge.iterationDistance << "\n";
    }
  });
  if (emitAuditAttributes)
    attachAuditAttributes(funcOp, analysis);
  if (analysis.validation.status != ProtocolStatus::Unsafe)
    return success();

  InFlightDiagnostic diagnostic = funcOp.emitError(
      "warp specialization rejected an unsafe post-memory channel protocol");
  diagnostic << ": total iteration distance "
             << analysis.validation.cycleIterationDistance << ", channels [";
  for (auto [index, edgeId] :
       llvm::enumerate(analysis.validation.cycleEdgeIds)) {
    if (index)
      diagnostic << ", ";
    diagnostic << analysis.graph.edges[edgeId].channelId;
  }
  diagnostic << "], edge distances [";
  for (auto [index, edgeId] :
       llvm::enumerate(analysis.validation.cycleEdgeIds)) {
    if (index)
      diagnostic << ", ";
    diagnostic << analysis.graph.edges[edgeId].iterationDistance;
  }
  diagnostic << "]";
  return failure();
}

} // namespace mlir
