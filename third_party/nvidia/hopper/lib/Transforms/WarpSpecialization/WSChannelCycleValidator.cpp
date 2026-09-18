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
    for (const ScheduledProtocolEvent &src : from)
      for (const ScheduledProtocolEvent &dst : to)
        graph.addEdge(src.id, dst.id, distance, ProtocolEdgeKind::TaskOrder,
                      src.channelId);
  };
  for (size_t i = 1; i < classes.size(); ++i)
    connect(classes[i - 1], classes[i], /*wraps=*/false);
  connect(classes.back(), classes.front(), /*wraps=*/true);
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

PostMemoryProtocolAnalysis
analyzePostMemoryChannelProtocols(ArrayRef<ChannelProtocolPlan> plans,
                                  ReuseConfig *reuseConfig) {
  PostMemoryProtocolAnalysis analysis;
  SmallVector<ProtocolTaskTimeline> timelines;
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
    size_t eventStart = analysis.graph.events.size();
    ProtocolEventId acquire =
        analysis.graph.addEvent(ProtocolEventKind::Acquire, label + " acquire");
    ProtocolEventId ready =
        analysis.graph.addEvent(ProtocolEventKind::Ready, label + " ready");

    SmallVector<ScheduledProtocolEvent> pendingEvents;
    auto stageEvent = [&](ProtocolEventId id, AsyncTaskId task,
                          Operation *anchor, ProtocolEventSide side) {
      int64_t stage;
      int64_t cluster;
      if (!anchor ||
          anchor->getParentOfType<scf::ForOp>().getOperation() !=
              plan.cadenceScope ||
          !getScheduleCoordinate(anchor, stage, cluster))
        return false;
      pendingEvents.push_back({id, task, plan.cadenceScope, anchor, side, stage,
                               cluster, channelId});
      return true;
    };
    if (!stageEvent(acquire, channel->relation.first,
                    plan.producerAcquireAnchor,
                    ProtocolEventSide::BeforeAnchor) ||
        !stageEvent(ready, channel->relation.first, plan.producerReadyAnchor,
                    ProtocolEventSide::AfterAnchor)) {
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
                      ProtocolEventSide::AfterAnchor)) {
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

    for (const ScheduledProtocolEvent &event : pendingEvents)
      getTaskTimeline(timelines, event.task, event.scope)
          .events.push_back(event);
    analysis.graph.addEdge(acquire, ready, 0, ProtocolEdgeKind::DataReady,
                           channelId);
    for (const ConsumerEvents &consumer : consumerEvents) {
      analysis.graph.addEdge(ready, consumer.wait, 0,
                             ProtocolEdgeKind::DataReady, channelId);
      analysis.graph.addEdge(consumer.wait, consumer.release, 0,
                             ProtocolEdgeKind::ControlFlow, channelId);
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
  if (analysis.validation.status == ProtocolStatus::Safe &&
      analysis.unsupportedChannelCount != 0) {
    analysis.validation.status = ProtocolStatus::Unsupported;
    analysis.validation.reason = unsupportedReason;
  }
  return analysis;
}

void auditPostMemoryChannelProtocols(
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

  PostMemoryProtocolAnalysis analysis =
      analyzePostMemoryChannelProtocols(plans, reuseConfig);
  LDBG("post-memory channel-cycle audit: "
       << stringifyProtocolStatus(analysis.validation.status)
       << ", events=" << analysis.graph.events.size()
       << ", edges=" << analysis.graph.edges.size()
       << ", supported=" << analysis.supportedChannelCount
       << ", unsupported=" << analysis.unsupportedChannelCount);
  if (emitAuditAttributes)
    attachAuditAttributes(funcOp, analysis);
}

} // namespace mlir
