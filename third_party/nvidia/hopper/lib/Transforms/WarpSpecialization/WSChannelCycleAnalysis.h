//===- WSChannelCycleAnalysis.h - Channel progress analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NV_DIALECT_HOPPER_TRANSFORMS_WARPSPECIALIZATION_WSCHANNELCYCLEANALYSIS_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_WARPSPECIALIZATION_WSCHANNELCYCLEANALYSIS_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace mlir {

using ProtocolEventId = unsigned;

enum class ProtocolEventKind {
  Acquire,
  Ready,
  Wait,
  Release,
};

enum class ProtocolEdgeKind {
  TaskOrder,
  DataReady,
  SlotReuse,
  ControlFlow,
  TaskWrap,
};

/// One event in a normalized channel protocol. The post-memory and
/// post-insertion graph builders attach their own IR metadata to these stable
/// event IDs; the cycle solver deliberately depends only on the protocol
/// topology and iteration distances.
struct ProtocolEvent {
  ProtocolEventId id;
  ProtocolEventKind kind;
  std::string label;
};

/// `from(i) -> to(i + iterationDistance)`.
struct ProtocolEdge {
  ProtocolEventId from;
  ProtocolEventId to;
  int64_t iterationDistance;
  ProtocolEdgeKind kind;
  unsigned channelId;
};

struct ProtocolGraph {
  ProtocolEventId addEvent(ProtocolEventKind kind,
                           const std::string &label = {});
  void addEdge(ProtocolEventId from, ProtocolEventId to,
               int64_t iterationDistance, ProtocolEdgeKind kind,
               unsigned channelId);

  llvm::SmallVector<ProtocolEvent> events;
  llvm::SmallVector<ProtocolEdge> edges;
};

enum class ProtocolStatus {
  Safe,
  Unsafe,
  Unsupported,
};

struct ProtocolValidation {
  ProtocolStatus status = ProtocolStatus::Safe;
  llvm::SmallVector<unsigned> cycleEdgeIds;
  int64_t cycleIterationDistance = 0;
  std::string reason;
};

/// Detect a directed cycle whose iteration-distance sum is non-positive.
/// Such a cycle has no startup credit and therefore cannot make progress.
///
/// For an SCC with N events, each distance d is transformed to
/// `d * (N + 1) - 1`. A cycle of length L then has transformed weight
/// `(N + 1) * sum(d) - L`, which is negative exactly when `sum(d) <= 0`.
/// Bellman-Ford supplies both the decision and a deterministic edge witness.
ProtocolValidation validateProtocolCycles(const ProtocolGraph &graph);

llvm::StringRef stringifyProtocolStatus(ProtocolStatus status);

} // namespace mlir

#endif
