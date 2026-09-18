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
  unsigned unsupportedChannelCount = 0;
};

/// Build and validate the ordinary loop-cadence channel protocols represented
/// by post-memory endpoint plans. Reuse groups and other specialized protocol
/// shapes are conservatively reported as unsupported by this first builder.
PostMemoryProtocolAnalysis
analyzePostMemoryChannelProtocols(ArrayRef<ChannelProtocolPlan> plans,
                                  ReuseConfig *reuseConfig);

/// Re-derive endpoint plans from post-memory channels, run the common solver,
/// and optionally attach test-only audit attributes to the function.
void auditPostMemoryChannelProtocols(
    triton::FuncOp funcOp, ArrayRef<Channel *> orderedChannels,
    const DenseMap<Channel *, SmallVector<Channel *>> &consumerGroups,
    ReuseConfig *reuseConfig, bool emitAuditAttributes);

} // namespace mlir

#endif
