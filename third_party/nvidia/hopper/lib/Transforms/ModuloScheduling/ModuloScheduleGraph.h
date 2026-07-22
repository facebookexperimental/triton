// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// ModuloScheduleGraph — abstract representation of a modulo-scheduled
// loop nest with multi-buffered memory, pipeline stages, and optional
// warp specialization.
//
// The graph is a side data structure (not MLIR ops). It references MLIR
// Operations but adds scheduling metadata (cycles, stages, buffers,
// edges) that drive the lowering passes.
//
// Transformation phases:
//   Phase 0: SCHEDULE  — DDG + Rau's → populate ScheduleNode cycle/stage
//   Phase 1: BUFFERS   — stage diffs → populate ScheduleBuffer count
//   Phase 1.5: WS      — utilization → assign warp_group per stage
//   Phase 2: EXPAND    — bottom-up prologue/kernel/epilogue per loop
//   Phase 3: LOWER     — replace MLIR ops with async copies + barriers

#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULE_GRAPH_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULE_GRAPH_H

#include "LatencyModel.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>

namespace mlir::triton::gpu {

// ============================================================================
// Memory abstraction
// ============================================================================

enum class MemoryKind { SMEM, TMEM, Register, BARRIER };

/// A multi-buffered memory allocation.
/// Represents SMEM or TMEM that needs multiple copies for pipelining.
struct ScheduleBuffer {
  unsigned id{};
  MemoryKind kind{MemoryKind::SMEM};
  llvm::SmallVector<int64_t, 4> shape; // e.g., {128, 64}
  unsigned elementBitWidth{16};        // e.g., 16 for f16
  unsigned count{1};                   // number of buffers (from stageDiff + 1)
  // Depth the schedule's lifetime analysis asked for, snapshotted before any
  // SMEM-budget reduction ever decrements `count`. count < requestedCount
  // means this ring runs shallower than the schedule wants — the A.5 auto
  // search uses the gap (in bytes) as its SMEM-pressure tie-breaker.
  unsigned requestedCount{1};

  // For data buffers: index of the corresponding BARRIER buffer (UINT_MAX if
  // none) For barrier buffers: index of the data buffer this barrier guards
  unsigned pairedBufferId{UINT_MAX};

  // Step 4.5: Buffer merging. Buffers with the same mergeGroupId share a
  // physical allocation. UINT_MAX = not merged (own physical buffer).
  unsigned mergeGroupId{UINT_MAX};

  // Live interval (cycle-level, for merging analysis)
  int liveStart{0}; // producer cycle
  int liveEnd{0};   // last consumer end cycle

  // The MLIR op that originally defines this buffer (e.g., local_alloc)
  Operation *defOp{nullptr};

  // Pass A.5 data partitioning. When partitionCount > 1 the emitter splits this
  // buffer into partitionCount per-group allocations, each replacing
  // shape[partitionDim] with mSize (used for the TMEM accumulator: full
  // (BM, BN) shape here, emitted as N groups of (mSize, BN)). Default 1 =
  // unpartitioned. The shared A/B SMEM operands stay unpartitioned.
  unsigned partitionCount{1};
  unsigned partitionDim{0}; // 0 = M, 1 = N
  unsigned mSize{0};        // per-partition size along partitionDim

  int64_t sizeBytes() const {
    if (kind == MemoryKind::BARRIER)
      return 8; // mbarrier object is 8 bytes in SMEM
    int64_t elems = 1;
    for (auto d : shape)
      elems *= d;
    return elems * elementBitWidth / 8;
  }
  int64_t totalBytes() const { return sizeBytes() * count; }
};

/// A physical buffer materialized from one or more logical ScheduleBuffers
/// that share storage via lifetime-aware merging (Step 4.5 / 4.6).
///
/// Per design doc §1140-1147: physical size = max(member.sizeBytes),
/// physical count = max(member.count). Shape is opaque (we only track
/// bytes — the lowering pass will allocate uint8 storage and reinterpret).
struct PhysicalBuffer {
  unsigned id{};
  MemoryKind kind{MemoryKind::SMEM};
  int64_t sizeBytes{0}; // max over members
  unsigned count{1};    // max over members
  llvm::SmallVector<unsigned, 4> memberBufferIds;

  int64_t totalBytes() const { return sizeBytes * static_cast<int64_t>(count); }
};

// ============================================================================
// Pipeline node — a scheduled operation
// ============================================================================

/// A node in the pipeline graph. Wraps an MLIR Operation with scheduling info.
struct ScheduleNode {
  unsigned id{};
  Operation *op{nullptr};

  // Schedule assignment (from Phase 0 + Step 2.5)
  HWPipeline pipeline{HWPipeline::NONE};
  int cycle{0};   // absolute cycle within the II
  int stage{0};   // cycle / II
  int cluster{0}; // dense rank of cycle within stage (Step 2.5)
  int latency{0}; // cycles until result available
  int selfLatency{
      0};           // warp-issue cost (cycles the warp is blocked dispatching)
  int occupancy{0}; // cycles this op holds its hardware pipeline (for ResMII
                    // / makespan). For TMA: bytes/bandwidth (≫ selfLatency for
                    // BW-bound stores); for TC: latency; CUDA/SFU: selfLatency.
                    // See OpLatencyInfo::occupancy. 0 = unset (fall back).
  int minWarps{1};  // warp count assumed by selfLatency; effective scales
                    // up by minWarps/actualWarps when WG has fewer warps

  // Frequency this op fires per outer-loop iteration. Outer-loop ops have
  // frequencyMultiplier = 1; inner K-loop ops have frequencyMultiplier =
  // K_TRIPS so the flat-view warp-makespan and barrier costs reflect that
  // an inner op runs K times per outer iteration. Set by buildFlatView
  // based on each ScheduledLoop's nesting depth. Default 1 (single-loop
  // kernels and non-flat callers don't need it).
  int frequencyMultiplier{1};

  // Super-node: if this node represents a child pipeline (inner loop)
  unsigned childPipelineId{UINT_MAX}; // index into ScheduleGraph::pipelines
  int prologueLatency{0};             // cycles before TC starts in child

  // Buffer references
  unsigned producesBuffer{UINT_MAX}; // index into ScheduleLoop::buffers
  llvm::SmallVector<unsigned, 2> consumesBuffers; // indices into buffers

  // Warp specialization (from Phase 1.5)
  int warpGroup{-1}; // -1 = unassigned, -2 = replicated (see replicatedGroups)
  // When warpGroup == -2 (replicated infra op whose consumers span multiple
  // warp groups), the set of consumer warp groups it must be cloned into.
  // Emitted as a multi-id ttg.partition so the WS pass replicates it. Empty
  // unless replicated.
  llvm::SmallVector<int, 4> replicatedGroups;

  // Pass A.5 data partitioning (carried from DDGNode). partitionCount > 1 marks
  // a partitioned MMA bundle: the emitter fans it into partitionCount
  // async_dots slicing mSize rows along partitionDim into per-group
  // accumulators. Default 1 = unpartitioned.
  unsigned partitionCount{1};
  unsigned partitionDim{0}; // 0 = M, 1 = N
  unsigned mSize{0};        // per-partition size along partitionDim
  // Pass A.7 epilogue subtiling. When subtileCount > 1, this op belongs to the
  // epilogue chain (tmem_load → ... → descriptor_store) that the sched2tlx
  // emitter renders as a `for sub_n in range(S)` loop, each iteration storing a
  // (BM, nSize) sub-tile along N. Defaults mean "not subtiled" (full BN tile).
  int subtileIndex{-1};
  int subtileCount{1};
  int nOffset{0};
  int nSize{0}; // 0 = full BN

  bool isSuperNode() const { return childPipelineId != UINT_MAX; }
  bool hasBuffer() const {
    return producesBuffer != UINT_MAX || !consumesBuffers.empty();
  }
};

// ============================================================================
// Pipeline edge — producer-consumer dependency
// ============================================================================

struct ScheduleEdge {
  unsigned srcId{};
  unsigned dstId{};
  int latency{};
  unsigned distance{}; // 0 = intra-iteration, 1+ = loop-carried
};

// ============================================================================
// Pipeline loop — a single pipelined scf.for
// ============================================================================

/// A pipelined loop with its schedule, nodes, edges, and buffers.
/// Analogous to a function: has inputs (consumed from outer scope),
/// outputs (produced for outer scope), and a body (nodes + edges).
struct ScheduleLoop {
  unsigned id{};
  scf::ForOp forOp;

  // Schedule parameters
  int II{0};
  int maxStage{0};
  int prologueLatency{0}; // cycles before TC starts (for parent's super-node)
  int tripCount{0};       // loop trip count (0 = unknown/not set)
  bool tripCountIsEstimated{
      false}; // true if tripCount is estimated, not constant

  // Body (kernel loop steady state)
  llvm::SmallVector<ScheduleNode, 16> nodes;
  llvm::SmallVector<ScheduleEdge, 16> edges;

  // Expanded structure (populated after expansion, empty before)
  // Prologue: ops cloned before the loop (stage 0 of first iterations)
  // Epilogue: ops cloned after the loop (drain of last stage)
  llvm::SmallVector<ScheduleNode, 8> prologueNodes;
  llvm::SmallVector<ScheduleNode, 8> epilogueNodes;
  bool isExpanded{false}; // true after expandScheduleGraph

  // Memory interface (inputs/outputs crossing loop boundary)
  // These drive multi-buffering at the parent level.
  //
  // isInput is intentionally kept alongside the separate inputs/outputs
  // vectors: it allows generic iteration over all ports (e.g., when building
  // the parent's buffer map) without needing to know which vector a port came
  // from.
  struct MemPort {
    unsigned bufferId{UINT_MAX}; // index into parent's buffers
    Operation *op{nullptr};      // the MLIR op at the boundary
    bool isInput{true};
  };
  llvm::SmallVector<MemPort, 4> inputs;  // consumed from outer scope
  llvm::SmallVector<MemPort, 4> outputs; // produced for outer scope

  // Multi-buffered allocations within this loop
  llvm::SmallVector<ScheduleBuffer, 4> buffers;

  // Physical buffers materialized from merge groups (populated by Step 4.5).
  // Each PhysicalBuffer's id matches the mergeGroupId of its member buffers.
  llvm::SmallVector<PhysicalBuffer, 4> physicalBuffers;

  // Absolute kernel-timeline interval for this loop region (Step 4.6).
  // 0 = unset; populated by computeRegionIntervals before kernel-wide
  // budget checks. For a non-persistent kernel: prologue + steady-state +
  // epilogue (all in cycles).
  int64_t regionStart{0};
  int64_t regionEnd{0};

  // Cross-group synchronization (from Pass B Step 2)
  enum class BarrierKind { MBARRIER, NAMED };
  struct CrossGroupBarrier {
    unsigned producerNodeId{};
    unsigned consumerNodeId{};
    int producerWarpGroup{};
    int consumerWarpGroup{};
    BarrierKind kind{BarrierKind::MBARRIER};
    unsigned depth{1};                 // number of barrier phases
    unsigned pairedBufferId{UINT_MAX}; // data buffer this barrier guards

    // Arrive/wait placement: explicit node IDs for Phase 3 lowering.
    // arriveAfterNodeId: emit mbarrier.arrive AFTER this node completes.
    // waitBeforeNodeId: emit mbarrier.wait BEFORE this node starts.
    unsigned arriveAfterNodeId{UINT_MAX};
    unsigned waitBeforeNodeId{UINT_MAX};

    // For mbarrier: expected bytes the producer writes (for expect_tx).
    int64_t expectBytes{0};
  };
  llvm::SmallVector<CrossGroupBarrier, 4> crossGroupBarriers;

  // Top-N alternative warp-group partitions for multi-graph autotuning dump.
  // Each entry is node-indexed (size == nodes.size()) giving the raw
  // cluster-applied warpGroup per node (NONE ops = -1), BEFORE infra-op
  // propagation / barrier synthesis. [0] is the committed winner; [1..] are
  // the next-best candidates from the Phase-4 cost model. Populated only when
  // TRITON_MODULO_DUMP_TOPN > 1; empty otherwise. Re-finalized + dumped into
  // one pluralized file (schedule_graphs.json) so an external harness can
  // autotune over them.
  llvm::SmallVector<llvm::SmallVector<int>, 3> topPartitions;

  // Phase-4 cost-model cost for THIS loop's committed/dumped partition (lower
  // == predicted faster). Dumped as `partition_cost` so an autotuning harness
  // can check the cost-model ranking against measured perf, and tell whether
  // equal-cost variants actually run equally fast. 0 = unset.
  double partitionCost{0.0};
  // Costs parallel to `topPartitions[k]` (best-first), populated with it.
  llvm::SmallVector<double, 3> topPartitionCosts;

  // Lookup
  llvm::DenseMap<Operation *, unsigned> opToNodeId;

  // Helpers
  const ScheduleNode &getNode(unsigned id) const {
    assert(id < nodes.size() && "node id out of range");
    return nodes[id];
  }
  /// Find the node for an MLIR op, or nullptr if not in this loop.
  const ScheduleNode *findNode(Operation *op) const {
    auto it = opToNodeId.find(op);
    if (it == opToNodeId.end())
      return nullptr;
    return &nodes[it->second];
  }
  int numStages() const { return maxStage + 1; }

  /// Get all nodes in a given stage.
  llvm::SmallVector<const ScheduleNode *> getNodesInStage(int stage) const {
    llvm::SmallVector<const ScheduleNode *> result;
    for (const auto &n : nodes)
      if (n.stage == stage)
        result.push_back(&n);
    return result;
  }
};

// ============================================================================
// Pipeline graph — the top-level container
// ============================================================================

/// The complete pipeline graph for a kernel. Contains all pipelined loops
/// (potentially nested) and their relationships.
struct ScheduleGraph {
  llvm::SmallVector<ScheduleLoop, 4> loops;

  /// Add a new loop and return its id.
  unsigned addLoop(scf::ForOp forOp) {
    unsigned id = loops.size();
    ScheduleLoop loop;
    loop.id = id;
    loop.forOp = forOp;
    loops.push_back(std::move(loop));
    return id;
  }

  ScheduleLoop &getLoop(unsigned id) {
    assert(id < loops.size() && "loop id out of range");
    return loops[id];
  }
  const ScheduleLoop &getLoop(unsigned id) const {
    assert(id < loops.size() && "loop id out of range");
    return loops[id];
  }

  /// Find the innermost loops (leaves) — process these first (bottom-up).
  llvm::SmallVector<unsigned> getInnermostLoops() const {
    llvm::SmallVector<unsigned> result;
    for (const auto &loop : loops) {
      bool isInnermost = true;
      for (const auto &node : loop.nodes) {
        if (node.isSuperNode()) {
          isInnermost = false;
          break;
        }
      }
      // A loop with no super-nodes is innermost
      // (but it might still not be a leaf if it has no nodes at all)
      if (isInnermost && !loop.nodes.empty())
        result.push_back(loop.id);
    }
    return result;
  }

  /// Get loops in bottom-up order (innermost first, outermost last).
  llvm::SmallVector<unsigned> getBottomUpOrder() const {
    llvm::SmallVector<unsigned> order;
    llvm::DenseSet<unsigned> visited;

    std::function<void(unsigned)> visit = [&](unsigned id) {
      if (visited.count(id))
        return;
      // Visit children first
      for (const auto &node : loops[id].nodes) {
        if (node.isSuperNode()) {
          assert(node.childPipelineId < loops.size() &&
                 "childPipelineId out of range");
          visit(node.childPipelineId);
        }
      }
      visited.insert(id);
      order.push_back(id);
    };

    for (unsigned i = 0; i < loops.size(); ++i)
      visit(i);
    return order;
  }

  /// Dump the graph for debugging. The no-arg overload writes to
  /// llvm::dbgs() (gated by `-debug-only=...`); the ostream overload
  /// writes unconditionally and is used by passes that expose a
  /// `print-schedule-graph` option (lit tests rely on this since
  /// `-debug-only` is debug-build only).
  void dump() const;
  void dump(llvm::raw_ostream &os) const;
};

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULE_GRAPH_H
