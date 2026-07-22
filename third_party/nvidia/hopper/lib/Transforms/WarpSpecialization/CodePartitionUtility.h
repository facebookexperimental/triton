#ifndef NV_DIALECT_HOPPER_TRANSFORMS_CODEPARTITIONUTILITY_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_CODEPARTITIONUTILITY_H_

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "Utility.h"
#include <algorithm>
#include <numeric>

namespace mlir {

namespace tt = mlir::triton;

// Discardable i32 attribute stamped by WSAtomicBroadcast on the
// dynamic-persistent tile-id broadcast slot's `local_alloc` to request a
// specific multi-buffer depth (tile prefetch depth). `allocateSmemBuffers`
// (WSMemoryPlanner) pins the buffer's copy count to this value so the requested
// depth is honored end-to-end (buffer shape + accumCnt-driven slot/phase),
// rather than leaving this non-innermost broadcast channel single-buffered.
// Defined here so the producer (WSAtomicBroadcast) and the consumer
// (WSMemoryPlanner) share one source of truth for the name.
constexpr llvm::StringLiteral kAtomicBroadcastCopiesAttrName =
    "ttg.atomic_broadcast_copies";

// Strip every warp-specialization metadata attribute that AutoWS stamps on
// ops/loops. Every graceful-reject path must call this so the downstream
// tritongpu-pipeline pass sees a plain, compilable non-WS kernel; a leftover WS
// attribute produces the "half-tagged" state the pipeliner crashes on. This is
// the single canonical set consumed by all reject paths.
void removeWarpSpecMetadata(triton::FuncOp funcOp);

enum class DataChannelKind : int {
  SMEM = 0,
  TMEM = 1,
  REG = 2,
  SMEMAlloc = 3,
  TMEMAlloc = 4
};

static inline std::string to_string(DataChannelKind k) {
  switch (k) {
  case DataChannelKind::SMEM:
    return "smem";
  case DataChannelKind::TMEM:
    return "tmem";
  case DataChannelKind::REG:
    return "reg";
  case DataChannelKind::SMEMAlloc:
    return "smem_alloc";
  case DataChannelKind::TMEMAlloc:
    return "tmem_alloc";
  }
  return "Unknown";
}

struct Channel {
public:
  using Relation = std::pair<int, SmallVector<int>>;

  Channel(int producer, SmallVector<int> &consumers, Operation *op,
          unsigned operandIdx, unsigned numBuffers, unsigned ID,
          DataChannelKind channelKind = DataChannelKind::SMEM)
      : relation(producer, consumers), op(op), operandIdx(operandIdx),
        _numBuffers(numBuffers), uniqID(ID), channelKind(channelKind) {}

  bool operator==(const Channel &c) {
    return relation == c.relation && operandIdx == c.operandIdx && op == c.op;
  }
  virtual ~Channel() = default;

  virtual Operation *getDstOp() { return op; }
  unsigned getDstOperandIdx() { return operandIdx; }
  Value getSrcOperand() { return op->getOperand(operandIdx); }
  virtual Operation *getSrcOp() { return getSrcOperand().getDefiningOp(); }
  virtual Operation *getAllocOp() { return nullptr; }
  virtual unsigned getNumBuffers() { return _numBuffers; }
  virtual Operation *getDstOpLast() { return nullptr; }
  virtual void getDstOps(SmallVector<Operation *> &dsts) {}

  Relation relation; // producer task Id, a list of consumer task Ids
  Operation *op;
  unsigned operandIdx;
  unsigned _numBuffers;
  DataChannelKind channelKind = DataChannelKind::SMEM;
  unsigned uniqID;
  std::string srcName; // Producer name captured at channel creation
};

// A few assumptions, a channel can have multiple consumers, but the consumers
// must be in the same region and the taskIds must be the same. We can have
// a representative consumer in the channel.
struct AllocChannel : Channel {
public:
  using Relation = std::pair<int, SmallVector<int>>;

  // source can be local_store, consumer can be gen5, ttg.memdesc_trans,
  // local_load
  AllocChannel(int producer, SmallVector<int> &consumers, Operation *allocOp,
               unsigned ID)
      : Channel(producer, consumers, nullptr, 0 /*operandIdx*/, 0, ID),
        allocOp(allocOp) {
    channelKind = DataChannelKind::SMEMAlloc;
  }

  bool operator==(const AllocChannel &c) {
    return relation == c.relation && allocOp == c.allocOp;
  }
  virtual ~AllocChannel() = default;

  virtual Operation *getSrcOp();
  virtual Operation *getDstOp();
  virtual Operation *getDstOpLast();
  virtual void getDstOps(SmallVector<Operation *> &dsts);
  virtual Operation *getAllocOp() { return allocOp; }
  virtual unsigned getNumBuffers();

  Operation *allocOp;

  // Producer op (ttg.local_store / async_tma_copy / the in-body template store
  // of a ttng.subtiled_region), resolved and cached in createAllocChannel at
  // channel-creation time, before any buffer rewriting. getSrcOp() returns this
  // when set. Caching is required for subtiled-region producers: when a
  // *sibling* channel that shares the same template store is lowered,
  // insertAsyncComm rewires the in-body store and removes the shared per-tile
  // buffer position, so a later alloc-walk in getSrcOp() would find nothing and
  // return null (SIGSEGV at the getSrcOp()->getBlock() deref). The template op
  // itself survives, so the cached pointer stays valid. Null for direct
  // alloc-with-src producers.
  Operation *cachedSrcOp = nullptr;

  // For a collapsed both-endpoints-subtiled channel, the per-tile staging
  // allocs of this (producer region, consumer region) pair other than the
  // representative (allocOp). collectAllocChannels records them here instead of
  // creating duplicate channels; after the in-body view rewire removes their
  // per-tile positions they become use_empty and are erased in insertAsyncComm.
  // Empty for every non-collapsed channel.
  SmallVector<Operation *> collapsedSiblingAllocs;

  // True only for the representative of a both-endpoints-subtiled collapse
  // (producer AND consumer in different-task ttng.subtiled_regions), set in
  // collectAllocChannels. This — NOT the broad channelIsSubtiled (which is also
  // true for a consumer-only-subtiled channel such as an epilogue bias load) —
  // gates the size-1 subtiled reuse-group / in-body slot-rotation machinery at
  // buffer.copy == 1. See channelIsCollapsedBothSubtiled.
  bool isCollapsedBothSubtiled = false;
};

struct ReuseGroup {
  std::vector<unsigned> channelIDs;
  std::vector<Channel *> channels;
};

struct ReuseConfig {
  // Each ReuseGroup
  std::vector<ReuseGroup> groups;
  unsigned getGroupSize() { return groups.size(); }
  ReuseGroup *getGroup(unsigned idx) {
    assert(idx < groups.size());
    return &groups[idx];
  }
};

struct CommChannel {
  DenseMap<int, Value> tokens;
  // Producer barrier is only needed when the producer op itself can update the
  // barrier inline, such as the TMA load.
  std::optional<Value> producerBarrier;
  // Consumer barrier is only needed when the consumer op itself can update the
  // barrier inline, such as MMAv5 ops.
  DenseMap<int, Value> consumerBarriers;
};

namespace ttng = ::mlir::triton::nvidia_gpu;
namespace triton {
namespace nvidia_gpu {
struct TmemAllocChannel : Channel {
  bool isOperandD;
  bool isOperandDNoAcc;
  // When true, this channel is a same-iteration resource-hazard guard:
  // tmem_load (producer) → tmem_store (consumer). It ensures the tmem_load
  // finishes reading before the next iteration's tmem_store overwrites.
  // This is the reverse direction of the wrap-around data-flow channel.
  bool isSameIterGuard = false;
  // For a collapsed chained-accumulator channel (multiple same-task MMA writers
  // into one operand-D tile, first writer use_acc=false): the channel's forward
  // commit comes from the LAST writer (getSrcOp), but the reuse/empty
  // producer_acquire must be placed before the FIRST (fresh-overwrite) writer
  // so the whole chain waits for the consumer's read of the previous iteration
  // (mirrors TLX attn_bwd_ws_2kv: one dq_empties acquire before n0, one
  // dq_fulls from n1). When set, insertAsyncComm relocates producer_acquire
  // here.
  Operation *acquireBeforeOp = nullptr;
  Operation *allocOp;

  // Can be produced by tmem_store or operand D of gen5, consumed by tmem_load
  // or gen5
  TmemAllocChannel(int producer, SmallVector<int> &consumers,
                   Operation *allocOp, bool isOperandD, bool isOperandDNoAcc,
                   unsigned uniqID)
      : Channel(producer, consumers, nullptr, 0 /*operandIdx*/, 0, uniqID),
        isOperandD(isOperandD), isOperandDNoAcc(isOperandDNoAcc),
        allocOp(allocOp) {
    assert(consumers.size() == 1 &&
           "TmemAllocChannel must have a single consumer partition");
    channelKind = DataChannelKind::TMEMAlloc;
  }

  virtual Operation *getSrcOp();
  virtual Operation *getDstOp();
  virtual unsigned getNumBuffers();
  virtual Operation *getAllocOp() { return allocOp; }
  virtual Operation *getDstOpLast();
  virtual void getDstOps(SmallVector<Operation *> &dsts);
};
} // namespace nvidia_gpu
} // namespace triton

constexpr static char kWarpSpecializeGeneratedBarrierAttrName[] =
    "ttg.ws_generated_barrier";

bool enclosing(scf::IfOp ifOp, Operation *op);
bool enclosing(scf::ForOp forOp, Operation *op);
bool enclosing(scf::WhileOp whileOp, Operation *op);

/// Returns true if \p tmemAlloc has a MMAv5OpInterface user inside \p forOp
/// whose acc_dep token is a loop iter_arg of \p forOp and whose output
/// token is yielded back to the same iter_arg position. This indicates
/// the accumulator is reused across iterations and the buffer index
/// should not rotate within this loop.
bool hasLoopCarriedAccToken(Operation *tmemAlloc, scf::ForOp forOp);

// Return number of AccumCnts for the given ctrlOp. AccumCnts due to reuses
// will be at the end, we go through all ReuseGroups and if any channel in
// the group is nested under ctrlOp, we add one accumCnt for this group.
unsigned getAccumCnts(Operation *ctrlOp,
                      const DenseSet<Operation *> &regionsWithChannels,
                      ReuseConfig *config);

// We pass in groupIdx, if it is -1, we are getting accumCnt for a channel
// not in a reuse group, directly in ctrlOp. ctrlOp can be null if
// reuseGroupIdx >= 0. parentOp is the control-flow op (scf.for or the after
// region of an scf.while) whose accumCnt arguments we are indexing into.
unsigned getAccumArgIdx(Operation *parentOp, Operation *ctrlOp,
                        const DenseSet<Operation *> &regionsWithChannels,
                        ReuseConfig *config, int reuseGroupIdx);

void getReuseChannels(ReuseGroup *gruop, Operation *regionOp,
                      SmallVector<Operation *> &chList);

// True when the channel's producer or consumer op is inside (or is) a
// ttng.subtiled_region. A collapsed both-endpoints-subtiled channel is the sole
// member of its reuse group, so reuse-group machinery must treat it specially
// even at size 1.
bool channelIsSubtiled(Channel *ch);
// Narrow form of channelIsSubtiled: true only for the representative of a
// both-endpoints-subtiled collapse (AllocChannel::isCollapsedBothSubtiled).
// Used to extend the subtiled reuse-group / in-body rotation machinery to
// buffer.copy == 1 without misfiring on consumer-only-subtiled channels.
bool channelIsCollapsedBothSubtiled(Channel *ch);

// Skip the accumCnt for unique channels.
unsigned getReuseAccumArgIdx(Operation *regionOp,
                             const DenseSet<Operation *> &regionsWithChannels,
                             ReuseConfig *config, int reuseGroupIdx);

SmallVector<Operation *>
getTaskTopRegion(triton::FuncOp funcOp, const SmallVector<Channel *> &channels);

void appendAccumCntsForOps(SmallVector<Operation *> &taskTopOps,
                           const SmallVector<Channel *> &channels,
                           DenseSet<Operation *> &regionsWithChannels,
                           ReuseConfig *config);

void collectRegionsWithChannels(const SmallVector<Channel *> &channels,
                                DenseSet<Operation *> &regionsWithChannels);

Value getAccumCount(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                    const DenseSet<Operation *> &regionsWithChannels,
                    ReuseConfig *config, int reuseGroupIdx);
std::pair<Value, Value> getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                                             Location loc, Value accumCnt,
                                             unsigned numBuffers);
void getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                          unsigned numBuffers,
                          const DenseSet<Operation *> &regionsWithChannels,
                          Value &bufferIdx, Value &phase, ReuseConfig *config,
                          int reuseGroupIdx, Channel *ch);

Value getBarrierForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                 Value barrierAlloc, Value bufferIdx);

Operation *optimizeTMALoads(OpBuilderWithAsyncTaskIds &builder,
                            SmallVector<tt::DescriptorLoadOp> &tmaLoads,
                            SmallVector<Value> &buffers, Value barrierAlloc,
                            Value bufferIdx, Value bufferIdxExtract,
                            Value phase, Operation *headProducer,
                            Operation *headConsumer,
                            Operation *headConsumerSameLevel,
                            ArrayRef<int> additionalConsumerTaskIds = {},
                            DictionaryAttr consumerWaitConstraints = {});
void specializeRegion(triton::FuncOp funcOp, unsigned requestedRegisters);
Value createBufferView(OpBuilderWithAsyncTaskIds &builder, Value alloc,
                       Value idx);
// Same-task SMEM records are useful for memory-planner bookkeeping, but code
// partitioning should only consume cross-partition communication channels.
void collectAllocChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                          triton::FuncOp &funcOp,
                          bool includeSameTaskSmemChannels = true);

/// Generate a combined DOT graph showing key ops and channels side by side.
/// Left subgraph: Key operations with control flow structure.
/// Right subgraph: Channel connections between partitions.
/// Output can be rendered with Graphviz: dot -Tpng graph.dot -o graph.png
void dumpCombinedGraph(SmallVector<std::unique_ptr<Channel>> &channels,
                       triton::FuncOp funcOp, llvm::raw_ostream &os);

/// Generate a buffer liveness visualization for TMEM allocations using
/// pre-calculated liveness intervals from the memory planner.
/// @param allocs List of TMEM allocation operations
/// @param allocToIntervals Map from alloc operation to liveness interval
/// @param allocToChannel Map from alloc operation to associated channel
/// @param channels List of all channels (for finding all channels per alloc)
/// @param os Output stream for DOT format
void dumpTmemBufferLiveness(
    SmallVector<triton::nvidia_gpu::TMEMAllocOp> &allocs,
    DenseMap<Operation *, Interval<size_t>> &allocToIntervals,
    DenseMap<Operation *, triton::nvidia_gpu::TMemAllocation> &allocToSize,
    DenseMap<Operation *, triton::nvidia_gpu::TmemAllocChannel *>
        &allocToChannel,
    SmallVector<Channel *> &channels, llvm::raw_ostream &os);

/// Generate a buffer liveness visualization for SMEM allocations using
/// pre-calculated liveness intervals from the memory planner.
/// @param bufferRange Map from buffer to liveness interval
/// @param channels List of all channels (for finding associated channels)
/// @param os Output stream for DOT format
void dumpSmemBufferLiveness(
    llvm::MapVector<Allocation::BufferId, std::pair<Interval<size_t>, size_t>>
        &bufferInfo,
    DenseMap<Allocation::BufferId, Operation *> &bufferOwners,
    SmallVector<Channel *> &channels, llvm::raw_ostream &os);

Operation *getSameLevelOp(Operation *p, Operation *c);
SmallVector<Operation *> getActualConsumers(Operation *consumerOp);
int channelInReuseGroup(Channel *channel, ReuseConfig *config,
                        bool reuseBarrier = true);
void fuseTcgen05CommitBarriers(triton::FuncOp &funcOp);
void doTMAStoreLowering(triton::FuncOp &funcOp);
bool appearsBefore(Operation *A, Operation *B);

// Shared reuse-legality primitive: is `dstOp` in the forward slice of `srcOp`,
// following SSA results AND memory (store -> buffer -> load)?  Single source of
// truth for "one op's value flows into another"; used by both the memory
// planner (hasPotentialReuse) and code partitioning (hasDependencyChain).
//
// `followBufferReuse` widens the memory hop from the store's own slot to any
// view/slot of the root buffer (see the .cpp). Use it only to ORDER an
// already-decided reuse (code partitioning), never to gate the planner's
// packing decision -- the wider walk over-forms reuse groups.
bool dependsThroughMemory(Operation *srcOp, Operation *dstOp,
                          bool followBufferReuse = false);

// Verify that an A1 (SMEM circular reuse) group is well-formed:
// - Multi-buffered (channels[0]->getNumBuffers() > 1).
// - Every producer/consumer of every channel lives in one common basic block.
// Returns true if valid, false otherwise.
bool verifyReuseGroup1(ReuseGroup *group);

// Verify that a 2-buffer reuse group is well-formed:
// - Exactly 2 channels, each with a single copy (getNumBuffers() == 1).
// - A dependency chain exists from one channel's consumer to the other's
//   producer.
// Returns true if valid; asserts on violations.
bool verifyReuseGroup2(ReuseGroup *group);

// For a verified 2-buffer reuse group, determine which channel is early (A)
// and which is late (B). Channel A is early if there is a data dependency
// chain from A's consumer to B's producer (A.consumer -> ... -> B.producer).
// Returns {earlyChannel, lateChannel}.
std::pair<Channel *, Channel *> orderReuseGroup2(ReuseGroup *group);

// Order an N-channel single-copy reuse group into one dependency chain
// (channel i's consumer reaches channel i+1's producer). Generalizes
// orderReuseGroup2 to N channels and across partitions. Returns the ordered
// channels, or an empty vector if no unique total chain order exists.
SmallVector<Channel *> orderReuseGroupChain(ReuseGroup *group);

// Verify that a reuse group with N channels (N >= 2) is well-formed:
// - At least 2 channels, each with a single copy (getNumBuffers() == 1).
// - All producers are in the same block (so program order gives a total order).
bool verifyReuseGroupN(ReuseGroup *group);

// For a verified N-channel reuse group, order channels by program order of
// their producer ops (getSrcOp()). Returns a sorted vector where channels[0]
// is earliest and channels[N-1] is latest in program order.
SmallVector<Channel *> orderReuseGroupN(ReuseGroup *group);

// Given ordered channels {early, late} in a reuse group, determine
// whether we need to explicitly move late's producer_acquire to before early's
// producer.
// Returns false when late's consumer and early's producer are in the same
// partition AND early's producer appears before late's consumer in program
// order (partition-internal ordering guarantees correctness).
// Returns true otherwise (explicit synchronization needed).
bool needExplicitReuseWait(Channel *earlyChannel, Channel *lateChannel);

// Returns true when `ownerCh` is the space owner of a reuse group and its
// producer overwrites the whole physical allocation before writing, not just
// its logical slice. Packed sibling slices can be clobbered by such a producer,
// so the owner needs back-edges to sibling consumers that are live across the
// repeated overwrite. Detection: the channel is the representative (its alloc
// has no `buffer.offset`) and is a `TmemAllocChannel` with
// `isOperandDNoAcc`.
bool isWholeAllocationOverwriteReuseOwner(Channel *ownerCh);
void invalidateWarpSpecializeBarriers(triton::FuncOp funcOp);

// Verify an A5 (cross-partition) reuse group: a single-copy group of >= 3
// channels whose producers span more than one block, where only a 2-channel
// "core" (producers co-located in one block) needs an explicit reuse barrier
// and every "extra" channel elides against the core via needExplicitReuseWait
// (same-partition implicit ordering enforces the WAR). Realized case: FA-bwd
// `_BWD_DOT_ATTRS_TMEM` group {dpT, dq, dsT}. Returns true iff treatable.
bool verifyReuseGroupCrossPartition(ReuseGroup *group);

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_CODEPARTITIONUTILITY_H_
