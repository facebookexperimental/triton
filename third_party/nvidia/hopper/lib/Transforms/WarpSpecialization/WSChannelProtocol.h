//===- WSChannelProtocol.h - Planned channel endpoints ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NV_DIALECT_HOPPER_TRANSFORMS_WARPSPECIALIZATION_WSCHANNELPROTOCOL_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_WARPSPECIALIZATION_WSCHANNELPROTOCOL_H_

#include "CodePartitionUtility.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {

class PostDominanceInfo;

/// Transaction cadence understood by the planned channel-protocol builder.
/// The first validator slice accepts only Loop cadence; the remaining values
/// are explicit so an unsupported shape is never mistaken for a simple loop.
enum class ChannelProtocolCadence {
  StraightLine,
  Loop,
  WhileLoop,
  Subtiled,
  Unsupported,
};

struct ChannelConsumerProtocolPlan {
  AsyncTaskId task;
  Operation *head = nullptr;
  Operation *tail = nullptr;
  Operation *waitAnchor = nullptr;
  Operation *waitScheduleAnchor = nullptr;
  Operation *releaseAnchor = nullptr;
  bool releaseIsAsync = false;
};

/// Non-owning, non-mutating description of the synchronization endpoints for
/// one consumer-group representative after memory planning. This is shared by
/// synchronization insertion and the planned-protocol validator so both use
/// exactly the same endpoint and cadence decisions.
struct ChannelProtocolPlan {
  Channel *masterChannel = nullptr;
  SmallVector<Channel *> channels;
  Operation *headProducer = nullptr;
  Operation *tailProducer = nullptr;
  Operation *headConsumer = nullptr;
  Operation *tailConsumer = nullptr;
  Operation *tmaHeadProducer = nullptr;
  Operation *producerAcquireAnchor = nullptr;
  Operation *producerReadyAnchor = nullptr;
  bool producerReadyIsAsync = false;
  Operation *tmaConsumerWaitAnchor = nullptr;
  Operation *cadenceScope = nullptr;
  ChannelProtocolCadence cadence = ChannelProtocolCadence::Unsupported;
  unsigned copies = 0;
  DenseSet<Operation *> consumerOps;
  DenseSet<Operation *> actualConsumerOps;
  SmallVector<Operation *> tmaProducers;
  SmallVector<ChannelConsumerProtocolPlan> consumers;

  const ChannelConsumerProtocolPlan *findConsumer(AsyncTaskId task) const;
};

/// Return true when every consumer in `consumerTaskId` uses an MMAv5 inline
/// completion barrier. This classification is shared by token creation and
/// protocol validation.
bool taskUsesOnlyGen5Consumers(ArrayRef<Channel *> channels,
                               AsyncTaskId consumerTaskId);

/// Same-level lookup used by channel protocol placement. SubtiledRegionOp is a
/// sequencing marker rather than a control-flow boundary, so this variant
/// treats it as transparent while walking parent scopes.
Operation *getProtocolSameLevelOp(Operation *producer, Operation *consumer);

Operation *getProtocolConsumerReleaseAnchor(PostDominanceInfo &postDominance,
                                            Operation *producer,
                                            Operation *consumer,
                                            AsyncTaskId consumerTask);

/// Build the shared, non-mutating endpoint plan for one consumer group. The
/// caller may subsequently relocate producerAcquireAnchor for operand-D or
/// physical-reuse hazards, but all ordinary endpoint decisions come from this
/// plan.
ChannelProtocolPlan buildChannelProtocolPlan(ArrayRef<Channel *> channels,
                                             PostDominanceInfo &postDominance);

} // namespace mlir

#endif
