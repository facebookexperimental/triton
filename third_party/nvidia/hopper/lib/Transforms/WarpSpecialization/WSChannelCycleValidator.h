//===- WSChannelCycleValidator.h - Post-memory validation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NV_DIALECT_HOPPER_TRANSFORMS_WARPSPECIALIZATION_WSCHANNELCYCLEVALIDATOR_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_WARPSPECIALIZATION_WSCHANNELCYCLEVALIDATOR_H_

#include "WSChannelCycleAnalysis.h"
#include "WSChannelProtocol.h"

namespace mlir {

/// Result of lowering post-memory channel plans into the normalized protocol
/// graph. Unsupported plans are omitted from the graph and reported
/// separately, allowing an unsafe supported component to take precedence.
struct PostMemoryProtocolAnalysis {
  ProtocolGraph graph;
  ProtocolValidation validation;
  unsigned supportedChannelCount = 0;
  unsigned ignoredIntraTaskChannelCount = 0;
  unsigned ignoredFiniteGuardChannelCount = 0;
  unsigned unsupportedChannelCount = 0;
  unsigned supportedStagingReuseProtocolCount = 0;
  unsigned supportedFiniteStagingReuseProtocolCount = 0;
  unsigned supportedSmemReuseGroupCount = 0;
  unsigned supportedSmemA1GroupCount = 0;
  unsigned supportedSmemA2GroupCount = 0;
  unsigned supportedSmemA3GroupCount = 0;
  unsigned supportedTmemA2GroupCount = 0;
  unsigned supportedTmemA5GroupCount = 0;
};

/// Build and validate ordinary loop-cadence SMEM channel protocols,
/// inner-to-outer operand-D TMEM drains, A1/A2/A3 SMEM reuse groups, A2/A5
/// TMEM reuse groups, and the coalesced staging-to-operand reuse WAR protocol.
/// Other specialized TMEM shapes remain conservatively unsupported.
PostMemoryProtocolAnalysis analyzePostMemoryChannelProtocols(
    ArrayRef<ChannelProtocolPlan> plans, ReuseConfig *reuseConfig,
    const StagingReuseProtocolPlan &stagingReusePlan);

/// Re-derive endpoint plans from post-memory channels and run the common
/// solver. Unsafe cycles fail validation; safe and not-yet-supported protocol
/// components continue. Test-only audit attributes are optional.
LogicalResult validatePostMemoryChannelProtocols(
    triton::FuncOp funcOp, ArrayRef<Channel *> orderedChannels,
    const DenseMap<Channel *, SmallVector<Channel *>> &consumerGroups,
    ReuseConfig *reuseConfig, bool emitAuditAttributes);

} // namespace mlir

#endif
