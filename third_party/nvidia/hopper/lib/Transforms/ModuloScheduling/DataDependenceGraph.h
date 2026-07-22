#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_DDG_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_DDG_H

#include "LatencyModel.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu {

struct DDGEdge {
  unsigned srcIdx{};
  unsigned dstIdx{};
  int latency{};
  unsigned distance{}; // 0 = intra-iteration, 1+ = loop-carried
};

/// Pass A.5 data-partition descriptor for one MMA bundle (or its accumulator
/// buffer). `count` = number of partitions N, `dim` = 0 (M) / 1 (N), `mSize` =
/// per-partition size along `dim` (e.g. BM/N for an M-split).
struct DataPartitionInfo {
  unsigned count{1};
  unsigned dim{0};
  unsigned mSize{0};
};

struct DDGNode {
  Operation *op{};
  unsigned idx{};
  HWPipeline pipeline{HWPipeline::NONE};
  int latency{};
  int selfLatency{};
  // How long this op ties up its hardware engine — i.e. when the NEXT op of the
  // same kind can start. Different from `latency`, which is how long until this
  // op's RESULT is ready for a consumer to use.
  // They're equal for most ops, but differ for an async TMA load: the load
  // engine is free again after ~bytes/bandwidth and can start the next load,
  // while the loaded data isn't ready for ~700 more cycles — so occupancy is
  // much smaller than latency. (TMA store: occupancy ≈ latency; TC = latency;
  // CUDA/SFU = selfLatency.)
  // See OpLatencyInfo::occupancy. 0 = unset → pipelineOccupancy() falls back.
  int occupancy{};
  // Min warps assumed by the modeled `selfLatency`. If the containing WG has
  // fewer warps, effective selfLat scales up by minWarps/actualWarps. See
  // `notes/latency_vs_selflatency.md` and `LatencyModel::getMinWarps`.
  int minWarps{1};
  // True hardware floor on WG width (TMEM = 4) — see
  // OpLatencyInfo::hardMinWarps. minWarps above is only the calibration
  // anchor; the width search may narrow below it, never below this.
  int hardMinWarps{1};
  // Cross-warp reduce facts for the width search (0 = not a reduce). See
  // OpLatencyInfo::reduceAxisWarps / reduceSyncSelfLat1w.
  int reduceAxisWarps{0};
  int reduceSyncSelfLat1w{0};
  bool isSuperNode{false}; // True if this node represents an inner loop
  int innerII{0};          // If super-node, the inner loop's II
  int prologueLatency{0};  // If super-node, cycles before TC starts (MEM busy)
  // Target-specific accumulator-buffer size (e.g. Blackwell TMEM columns),
  // precomputed via LatencyModel so the schedulers stay HW-agnostic. 0 = none.
  // AMDGPU has no separate accumulator memory (MFMA accumulates in VGPRs), so
  // AMDLatencyModel leaves this 0 — it is NV-only today but kept on the shared
  // node so the schedulers need no per-backend field.
  int64_t tmemAllocCols{0};

  // Pass A.5 data partitioning. When partitionCount > 1 this node is a
  // partition "bundle": the emitter fans it into partitionCount parallel ops
  // (an MMA → N async_dots, each handling mSize rows along partitionDim into
  // its own accumulator). The bundle stays a single scheduled node occupying
  // max(full-tile occupancy, partitionCount x issue cost) — the M-split
  // conserves MAC area, so only the per-sub-MMA issue floor scales with N
  // (see applyDataPartition). Default 1 = unpartitioned.
  unsigned partitionCount{1};
  unsigned partitionDim{0}; // 0 = M, 1 = N (emitter supports M only today)
  unsigned mSize{0};        // per-partition size along partitionDim
  llvm::SmallVector<unsigned> succs;
  llvm::SmallVector<unsigned> preds;
};

/// How many cycles this op holds its pipeline resource. Used by ResMII and the
/// modulo reservation table when placing ops.
///
/// Prefer the per-op `occupancy` computed by the LatencyModel (validated on
/// B200): TMA *store* and TC are bandwidth/serial-bound (occupancy ≈ latency),
/// but a TMA *load* is multi-outstanding — it occupies the engine only
/// ~bytes/bandwidth, far less than its round-trip latency. CUDA/SFU use the
/// per-op pipe slot count (selfLatency).
///
/// Fallback (occupancy unset, e.g. manually-built super-nodes): the old rule —
/// full `latency` for async TMA/TC, `selfLatency` otherwise.
inline int pipelineOccupancy(const DDGNode &node) {
  if (node.occupancy > 0)
    return node.occupancy;
  if (node.pipeline == HWPipeline::NONE)
    return 1;
  if (node.pipeline == HWPipeline::TMA || node.pipeline == HWPipeline::TC)
    return std::max(node.latency, 1);
  return std::max(node.selfLatency, 1);
}

/// Data Dependence Graph for one scf.for loop body.
/// Captures both intra-iteration and loop-carried (distance-1) edges.
class DataDependenceGraph {
public:
  /// `partition` (Pass A.5) is applied to every inner super-node's DDG before
  /// it is modulo-scheduled, so the super-node's `innerII` reflects the split
  /// (an M-partitioned inner MMA is scheduled at its partitioned ResMII, not
  /// the unpartitioned one). Empty by default = no partitioning.
  static DataDependenceGraph
  build(scf::ForOp loop, const LatencyModel &model,
        const llvm::DenseMap<Operation *, DataPartitionInfo> &partition =
            llvm::DenseMap<Operation *, DataPartitionInfo>());

  llvm::ArrayRef<DDGNode> getNodes() const { return nodes; }
  llvm::ArrayRef<DDGEdge> getEdges() const { return edges; }
  const DDGNode &getNode(unsigned idx) const { return nodes[idx]; }
  unsigned getNumNodes() const { return nodes.size(); }
  const llvm::DenseMap<Operation *, unsigned> &getOpToIdx() const {
    return opToIdx;
  }

  /// Get all incoming edges for a node.
  llvm::SmallVector<const DDGEdge *> getInEdges(unsigned nodeIdx) const;

  /// Get all outgoing edges for a node.
  llvm::SmallVector<const DDGEdge *> getOutEdges(unsigned nodeIdx) const;

  /// Compute critical-path height (bottom-up) from each node to any sink.
  llvm::DenseMap<unsigned, int> computeCriticalPathHeights() const;

  /// Compute ResMII: max over all pipelines of total self-latency.
  int computeResMII() const;

  /// Compute RecMII: max over all recurrence circuits of sum_lat / sum_dist.
  int computeRecMII() const;

  /// Compute MinII = max(ResMII, RecMII).
  int computeMinII() const;

  /// Pass A.5: tag the MMA nodes named in `mmaInfo` as partition bundles —
  /// copy their partition fields onto the node and scale TC pipeline occupancy
  /// by `count`, so ResMII reflects the N hardware MMA issues. Nodes not in the
  /// map are untouched.
  void applyDataPartition(
      const llvm::DenseMap<Operation *, DataPartitionInfo> &mmaInfo);

  /// Dump the DDG to llvm::dbgs() for debugging.
  void dump() const;

private:
  llvm::SmallVector<DDGNode> nodes;
  llvm::SmallVector<DDGEdge> edges;
  llvm::DenseMap<Operation *, unsigned> opToIdx;
  // For multi-stage super-nodes (prologue/kloop/epilogue sharing the same
  // Operation*), opToIdx maps to the epilogue (producer). consumerOpToIdx
  // maps to the prologue so loop-carried edges target the correct node.
  llvm::DenseMap<Operation *, unsigned> consumerOpToIdx;

  unsigned addNode(Operation *op, const LatencyModel &model);
  void addEdge(unsigned src, unsigned dst, int latency, unsigned distance);
};

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_DDG_H
