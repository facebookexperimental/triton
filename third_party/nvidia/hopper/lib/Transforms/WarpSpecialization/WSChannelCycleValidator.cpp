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
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

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
  SmallVector<ScheduledProtocolEvent> events;
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
                AsyncTaskId task, Operation *scope) {
  auto it = llvm::find_if(timelines, [&](const ProtocolTaskTimeline &timeline) {
    return timeline.task == task && timeline.scope == scope;
  });
  if (it != timelines.end())
    return *it;
  timelines.push_back({task, scope, {}});
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
        graph.addEdge(src.id, dst.id, distance, ProtocolEdgeKind::TaskOrder,
                      src.channelId);
    }
  };
  auto hasOrderingSource = [](ArrayRef<ScheduledProtocolEvent> eventClass) {
    return llvm::any_of(eventClass, [](const ScheduledProtocolEvent &event) {
      return event.ordersFollowingEvents;
    });
  };
  for (size_t target = 0; target < classes.size(); ++target) {
    size_t source = (target + classes.size() - 1) % classes.size();
    while (source != target && !hasOrderingSource(classes[source]))
      source = (source + classes.size() - 1) % classes.size();
    if (!hasOrderingSource(classes[source]))
      continue;
    connect(classes[source], classes[target], source >= target);
  }
  return true;
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
  funcOp->setAttr("nvws.test.channel_cycle_unsupported_channels",
                  IntegerAttr::get(IntegerType::get(context, 64),
                                   analysis.unsupportedChannelCount));
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

} // namespace

PostMemoryProtocolAnalysis analyzePostMemoryChannelProtocols(
    ArrayRef<ChannelProtocolPlan> plans, ReuseConfig *reuseConfig,
    const DenseSet<int64_t> &specializedBufferIds) {
  PostMemoryProtocolAnalysis analysis;
  SmallVector<ProtocolTaskTimeline> timelines;
  DenseSet<ProtocolEventId> nestedCadenceEvents;
  std::string unsupportedReason;

  for (const ChannelProtocolPlan &plan : plans) {
    Channel *channel = plan.masterChannel;
    auto unsupported = [&](StringRef reason) {
      ++analysis.unsupportedChannelCount;
      if (unsupportedReason.empty())
        unsupportedReason = reason.str();
    };
    if (!channel || plan.channels.empty() || plan.copies == 0) {
      unsupported("channel plan has incomplete identity or copy depth");
      continue;
    }
    if (plan.cadence != ChannelProtocolCadence::Loop ||
        !isa_and_nonnull<scf::ForOp>(plan.cadenceScope)) {
      unsupported("only scf.for loop-cadence channels are supported");
      continue;
    }
    if (triton::gpu::isPhysicalCluster(plan.cadenceScope)) {
      unsupported("multi-CTA channel protocols are not yet supported");
      continue;
    }
    if (channel->channelKind != DataChannelKind::SMEMAlloc) {
      unsupported("only ordinary SMEM channels are supported");
      continue;
    }
    if (llvm::any_of(plan.channels, [&](Channel *member) {
          return member->channelKind != DataChannelKind::SMEMAlloc ||
                 member->relation.first != channel->relation.first ||
                 member->getNumBuffers() != plan.copies;
        })) {
      unsupported("grouped producers do not share one SMEM protocol");
      continue;
    }
    if (llvm::any_of(plan.channels, [&](Channel *member) {
          Operation *alloc = member->getAllocOp();
          if (!alloc)
            return false;
          auto bufferId = alloc->getAttrOfType<IntegerAttr>("buffer.id");
          return bufferId && specializedBufferIds.contains(bufferId.getInt());
        })) {
      unsupported("staging-reuse channel protocols are not yet supported");
      continue;
    }
    if (reuseConfig && llvm::any_of(plan.channels, [&](Channel *member) {
          return channelInReuseGroup(member, reuseConfig,
                                     /*reuseBarrier=*/false) >= 0;
        })) {
      unsupported("physical reuse-group protocols are not yet supported");
      continue;
    }
    if (plan.consumers.empty() || !plan.producerAcquireAnchor ||
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
      if (!anchor ||
          anchor->getParentOfType<scf::ForOp>().getOperation() !=
              plan.cadenceScope ||
          !getScheduleCoordinate(anchor, stage, cluster))
        return false;
      LDBG("channel " << channelId << " event " << id << " task " << task
                      << " stage " << stage << " cluster " << cluster
                      << (side == ProtocolEventSide::BeforeAnchor ? " before "
                                                                  : " after ")
                      << anchor->getName());
      pendingEvents.push_back({id, task, plan.cadenceScope, anchor, side, stage,
                               cluster, channelId, ordersFollowingEvents});
      return true;
    };
    if (!stageEvent(acquire, channel->relation.first,
                    plan.producerAcquireAnchor,
                    ProtocolEventSide::BeforeAnchor) ||
        !stageEvent(ready, channel->relation.first, plan.producerReadyAnchor,
                    ProtocolEventSide::AfterAnchor,
                    !plan.producerReadyIsAsync)) {
      analysis.graph.events.resize(eventStart);
      unsupported("producer endpoint lacks a loop schedule coordinate");
      continue;
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
      if (!stageEvent(wait, consumer.task, consumer.waitScheduleAnchor,
                      ProtocolEventSide::BeforeAnchor) ||
          !stageEvent(release, consumer.task, consumer.releaseAnchor,
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

    if (plan.cadenceScope->getParentOfType<scf::ForOp>() ||
        plan.cadenceScope->getParentOfType<scf::WhileOp>()) {
      for (ProtocolEventId event = eventStart;
           event < analysis.graph.events.size(); ++event)
        nestedCadenceEvents.insert(event);
    }

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

    for (const ScheduledProtocolEvent &event : pendingEvents)
      getTaskTimeline(timelines, event.task, event.scope)
          .events.push_back(event);
    analysis.graph.addEdge(acquire, ready, getTaskSpanDistance(acquire, ready),
                           ProtocolEdgeKind::TaskOrder, channelId);
    for (const ConsumerEvents &consumer : consumerEvents) {
      analysis.graph.addEdge(ready, consumer.wait, 0,
                             ProtocolEdgeKind::DataReady, channelId);
      analysis.graph.addEdge(
          consumer.wait, consumer.release,
          getTaskSpanDistance(consumer.wait, consumer.release),
          ProtocolEdgeKind::TaskOrder, channelId);
      analysis.graph.addEdge(consumer.release, acquire, plan.copies,
                             ProtocolEdgeKind::SlotReuse, channelId);
    }
    ++analysis.supportedChannelCount;
  }

  for (ProtocolTaskTimeline &timeline : timelines) {
    if (addTaskOrderEdges(analysis.graph, timeline))
      continue;
    ++analysis.unsupportedChannelCount;
    if (unsupportedReason.empty())
      unsupportedReason = "task protocol events span multiple blocks";
  }
  analysis.validation = validateProtocolCycles(analysis.graph);
  bool hasNegativeNestedEdge =
      analysis.validation.status == ProtocolStatus::Unsafe &&
      llvm::any_of(analysis.validation.cycleEdgeIds, [&](unsigned edgeId) {
        const ProtocolEdge &edge = analysis.graph.edges[edgeId];
        return edge.iterationDistance < 0 &&
               (nestedCadenceEvents.contains(edge.from) ||
                nestedCadenceEvents.contains(edge.to));
      });
  if (hasNegativeNestedEdge) {
    analysis.validation.status = ProtocolStatus::Unsupported;
    analysis.validation.cycleEdgeIds.clear();
    analysis.validation.reason =
        "mixed-distance cycles in nested loops require outer-boundary "
        "validation";
  }
  if (analysis.validation.status == ProtocolStatus::Unsafe &&
      analysis.validation.cycleIterationDistance < 0) {
    analysis.validation.status = ProtocolStatus::Unsupported;
    analysis.validation.cycleEdgeIds.clear();
    analysis.validation.reason =
        "negative-distance cycles require boundary-aware validation";
  }
  if (analysis.validation.status == ProtocolStatus::Safe &&
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

  DenseSet<int64_t> specializedBufferIds;
  funcOp.walk([&](triton::gpu::LocalAllocOp alloc) {
    auto reuseTarget =
        alloc->getAttrOfType<IntegerAttr>("allocation.reuseTarget");
    if (!reuseTarget)
      return;
    specializedBufferIds.insert(reuseTarget.getInt());
    if (auto bufferId = alloc->getAttrOfType<IntegerAttr>("buffer.id"))
      specializedBufferIds.insert(bufferId.getInt());
  });

  PostMemoryProtocolAnalysis analysis = analyzePostMemoryChannelProtocols(
      plans, reuseConfig, specializedBufferIds);
  LDBG("post-memory channel-cycle audit: "
       << stringifyProtocolStatus(analysis.validation.status)
       << ", events=" << analysis.graph.events.size()
       << ", edges=" << analysis.graph.edges.size()
       << ", supported=" << analysis.supportedChannelCount
       << ", unsupported=" << analysis.unsupportedChannelCount);
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
