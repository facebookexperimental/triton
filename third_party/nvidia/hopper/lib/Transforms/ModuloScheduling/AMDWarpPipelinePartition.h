// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// AMD Warp-Pipeline Partition (Steps 4.7 + 4.8)
//
// Step 4.7: Latency-aware multi-pipeline cluster partitioning for AMD
//           warp-pipelining. Starts with one cluster per active pipeline,
//           then greedily merges tightly-coupled pairs.
//
// Step 4.8: Derive s_setprio priorities from the modulo reservation table
//           (MRT occupancy, monopolization, urgency).
//
// The output is a set of clusters, each becoming a warp_pipeline_stage
// region. Both warp groups run ALL clusters, phase-offset by one stage
// (the symmetric ping-pong model).

#ifndef TRITON_GPU_MODULO_SCHEDULING_AMD_WARP_PIPELINE_PARTITION_H
#define TRITON_GPU_MODULO_SCHEDULING_AMD_WARP_PIPELINE_PARTITION_H

#include "DataDependenceGraph.h"
#include "ModuloReservationTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu {

/// A cluster of ops that will become a warp_pipeline_stage region.
struct AMDClusterInfo {
  int clusterId{0};
  llvm::SmallVector<unsigned, 16> nodeIndices;
  llvm::SmallDenseSet<HWPipeline, 4> pipelines;
  llvm::SmallDenseMap<HWPipeline, double, 4> utilization;
  int sSetprio{0};
};

/// Result of the warp-pipeline partition (Steps 4.7 + 4.8).
struct AMDWarpPipelineResult {
  llvm::SmallVector<AMDClusterInfo, 4> clusters;
  int pingpongOffset{1};

  bool isWarpPipelineCandidate() const { return clusters.size() >= 2; }
};

/// Step 4.7: Partition DDG nodes into clusters for AMD warp-pipelining.
///
/// Uses separation cost (barrier overhead / cycle gap) and multi-pipeline
/// makespan validation to decide which pipelines merge. Produces >= 2
/// clusters for warp-pipelining to be profitable; returns a single-cluster
/// result if merging collapses everything (caller should skip warp-pipeline).
AMDWarpPipelineResult
partitionForAMDWarpPipeline(const DataDependenceGraph &ddg,
                            const ModuloScheduleResult &schedule);

/// Step 4.8: Derive s_setprio priorities from MRT occupancy and slack.
///
/// Populates sSetprio on each cluster in `result` and sets pingpongOffset.
/// Only fires on warp-pipelined loops (>= 2 clusters).
void assignAMDWarpPipelinePriorities(AMDWarpPipelineResult &result,
                                     const DataDependenceGraph &ddg,
                                     const ModuloScheduleResult &schedule);

} // namespace mlir::triton::gpu

#endif // TRITON_GPU_MODULO_SCHEDULING_AMD_WARP_PIPELINE_PARTITION_H
