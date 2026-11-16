#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"
#include "BufferIndexAnalysis.h"
#include "BufferRegion.h"
#include "CallGraph.h"
#include "Function.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

namespace mlir {

class OpBuilder;
struct AllocationSlice;

/// Callback to allow backend to provide more information on whether a barrier
/// is needed between two operations. Even though two operations access the same
/// shared memory they may not require a barrier in between them.
using MembarFilterFn = std::function<bool(
    Operation *, Operation *, bool /*lhsIsRead*/, bool /*rhsIsRead*/,
    Allocation *, const AllocationSlice &, const AllocationSlice &)>;

/// Slice-level filter to allow backends to ignore specific aliasing cases.
using MembarSliceFilterFn =
    std::function<bool(const AllocationSlice &, const AllocationSlice &,
                       bool /*lhsIsRead*/, bool /*rhsIsRead*/, Allocation *)>;

/// Optional backend policy for reusing scratch without a CTA rendezvous.
/// canReuse must be a pure proof that both operations use the same disjoint
/// per-warp scratch partitions and that only prior reads need ordering before
/// the next write. It is queried only for intersecting allocated scratch
/// effects, after the ordinary filter. emitBefore must order those reads before
/// the supplied operation's writes and be idempotent across repeated analysis.
/// Membar emits it only for an actual WAR and when no incoming hazard needs a
/// CTA barrier. Neither callback may discard pending BlockInfo dependencies.
/// Omitting either callback conservatively disables scratch reuse.
struct MembarScratchSync {
  std::function<bool(Operation *, Operation *, Allocation *)> canReuse;
  std::function<void(Operation *, OpBuilder &)> emitBefore;

  explicit operator bool() const { return canReuse && emitBefore; }
};

// Represents the access to a slice of an allocation
// It contains information both on physical memory (the interval) and a
// logical view on it (layout, subslice offsets and shape for the access)
struct AllocationSlice {
public:
  // Create allocation slice from a value, collecting subslice offsets.
  // BufferIndexAnalysis attaches dynamic buffer-index information to the
  // stage-aware slice before it is inserted into BlockInfo.
  AllocationSlice(Value value, Interval<size_t> allocationInterval,
                  Allocation::BufferId bufferId,
                  Allocation *allocation = nullptr, Value stageBasis = {});

  // Builder for accesses that represent accesses to the whole
  // allocation (scratch buffers, ArriveBarrierOp, ..)
  AllocationSlice(Interval<size_t> interval)
      : allocationInterval(interval), accessTy(nullptr),
        bufferId(Allocation::InvalidBufferId) {}

  bool operator<(const AllocationSlice &other) const {
    return asTuple() < other.asTuple();
  }

  bool operator==(const AllocationSlice &other) const {
    return asTuple() == other.asTuple();
  }

  // Check if a AllocationSlice intersects with another other.
  // This happens if their subslice regions intersect in all dimensions.
  // Returns true if it can't prove the AllocationSlices are disjoint.
  bool intersects(const AllocationSlice &other) const;

  Allocation::BufferId getBufferId() const { return bufferId; }

  // Transport a pending access into the counted loop's current-IV coordinates.
  AllocationSlice enterStageLoop(Value induction, unsigned initialValue) const;
  AllocationSlice advanceStageLoop(Value induction) const;
  AllocationSlice forgetStageLoop() const;

  /// Translate shared-memory geometry and discard callee-local index facts.
  AllocationSlice
  translateToCallsite(CallOpInterface call, FunctionOpInterface callee,
                      triton::BufferRegionAnalysis &regions) const;

  void print(raw_ostream &os) const;

  void invalidateIterationInfo() {
    bufferIndexExpr = nullptr;
    subsliceSource = {};
  }

  // Immutable MAY-addresses covering every dynamic instance of this access.
  // They remain valid across backedges, unlike the epoch-relative facts above.
  // The owning BufferRegionAnalysis outlives every slice using this pointer.
  const triton::BufferRegionFootprint *physicalFootprint = nullptr;

  // Effect classification is preserved when geometry is widened or translated.
  // Unknown for implicit scratch and non-shared accesses.
  std::optional<triton::gpu::SharedKind> sharedKind;

  // Buffer-index expression attached by BufferIndexAnalysis. It participates
  // in ordering/equality so accesses to different slots remain separate.
  // Must not be mutated after the slice is inserted into a sorted container
  // (e.g. BlockInfo::SliceMapT); rebuild the container instead, as
  // BlockInfo::invalidateIterationInfo does.
  const BufferIndexExpr *bufferIndexExpr = nullptr;

  // Parameters have no callee buffer ID. Bind this argument's effects to the
  // caller's allocation when importing the function summary.
  std::optional<unsigned> argumentIndex;

private:
  // An empty basis denotes a literal stage. A nonempty basis denotes
  // (basis + offset) % numStages. Only an eligible counted loop supplies a
  // basis.
  struct StageInfo {
    Value parent;
    Value basis;
    Interval<size_t> parentInterval{0, 0};
    size_t stride = 0;
    unsigned numStages = 0;
    unsigned offset = 0;
  } stage;
  using StageKey = std::tuple<const void *, const void *, Interval<size_t>,
                              size_t, unsigned, unsigned>;

  std::tuple<Interval<size_t>, Allocation::BufferId, const void *,
             llvm::ArrayRef<int32_t>, StageKey, const BufferIndexExpr *,
             const void *, const void *, std::optional<unsigned>,
             std::optional<triton::gpu::SharedKind>>
  asTuple() const {
    return {allocationInterval,
            bufferId,
            accessTy.getAsOpaquePointer(),
            subsliceOffsets,
            {stage.parent.getAsOpaquePointer(),
             stage.basis.getAsOpaquePointer(), stage.parentInterval,
             stage.stride, stage.numStages, stage.offset},
            bufferIndexExpr,
            subsliceSource.getAsOpaquePointer(),
            physicalFootprint,
            argumentIndex,
            sharedKind};
  }
  // Offsets from subslice, borrowed from its immutable context-owned attribute.
  // Empty when offsets are unknown.
  llvm::ArrayRef<int32_t> subsliceOffsets;
  // The source descriptor supplying the coordinates for subslice offsets.
  Value subsliceSource;
  // The allocated interval for this buffer
  Interval<size_t> allocationInterval;
  // Type of the memory descriptor for this access
  triton::gpu::MemDescType accessTy;
  // Buffer id for partial sync on wait_barrier deps.
  Allocation::BufferId bufferId;
};

struct ThreadIssuer {
  enum Kind {
    Unknown,
    // Thread zero of the current execution partition.
    Thread0,
    // A fixed leader of partition-relative warp zero.
    // The leader's lane may be nonzero.
    Warp0Leader
  };
  Kind kind = Unknown;
  Region *region = nullptr;

  bool operator==(const ThreadIssuer &) const = default;
};

// Empty summaries contain no operations. Nonempty summaries preserve an issuer
// only when every operation has the same known kind and execution region.
class IssuerSummary {
public:
  IssuerSummary() = default;
  explicit IssuerSummary(ThreadIssuer issuer) : issuer(issuer) {}

  bool empty() const { return !issuer; }
  bool hasSameKnownIssuer(const IssuerSummary &other) const;
  void join(const IssuerSummary &other);

  bool operator==(const IssuerSummary &) const = default;

private:
  std::optional<ThreadIssuer> issuer;
};

enum class CompletionSync : uint8_t {
  None,
  // Independent global reads may proceed before the rendezvous.
  Shared,
  All,
};

struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  SliceMapT syncReadSlices;
  SliceMapT syncWriteSlices;

  // Thread ordering is independent of which physical addresses are accessed.
  struct ThreadSyncState {
    IssuerSummary effectIssuers;
    CompletionSync completion = CompletionSync::None;
    bool peerWaitNeedsSync = false;

    bool operator==(const ThreadSyncState &) const = default;
  } threadSync;
  struct ThreadDemandState {
    bool hasDemand = false;
    // Base membar demands not proven to have only global-read effects.
    bool hasDemandBeyondGlobalReads = false;
    // Only publication demands contribute to this issuer summary.
    IssuerSummary publicationIssuers;
    bool mayNotifyPeer = false;

    bool operator==(const ThreadDemandState &) const = default;
  } threadDemands;

  BlockInfo() = default;

  bool hasEffects() const {
    return !threadSync.effectIssuers.empty() || !syncReadSlices.empty() ||
           !syncWriteSlices.empty();
  }

  /// Unions two BlockInfo objects.
  BlockInfo &join(const BlockInfo &other) {
    for (auto &slice : other.syncReadSlices)
      syncReadSlices[slice.first].insert(slice.second.begin(),
                                         slice.second.end());

    for (auto &slice : other.syncWriteSlices)
      syncWriteSlices[slice.first].insert(slice.second.begin(),
                                          slice.second.end());
    joinThreadState(other);
    return *this;
  }

  BlockInfo
  mapSlices(const std::function<AllocationSlice(const AllocationSlice &)> &map)
      const {
    BlockInfo result;
    // Stage transport re-coordinates slices only; thread ordering is
    // coordinate-independent and must survive edge traversal and summaries.
    result.threadSync = threadSync;
    result.threadDemands = threadDemands;
    auto transfer = [&](const SliceMapT &source, SliceMapT &destination) {
      for (const auto &[slice, ops] : source) {
        auto &mappedOps = destination[map(slice)];
        mappedOps.insert(ops.begin(), ops.end());
      }
    };
    transfer(syncReadSlices, result.syncReadSlices);
    transfer(syncWriteSlices, result.syncWriteSlices);
    return result;
  }

  template <typename Transform> void transformSlices(Transform transform) {
    for (auto *slices : {&syncReadSlices, &syncWriteSlices}) {
      SliceMapT transformed;
      for (const auto &[slice, ops] : *slices)
        for (const AllocationSlice &mapped : transform(slice)) {
          auto &dst = transformed[mapped];
          dst.insert(ops.begin(), ops.end());
        }
      *slices = std::move(transformed);
    }
  }

  /// Clears the buffer index and access origin of every slice, rebuilding both
  /// maps. Used at loop backedges where the same SSA value can denote a value
  /// from a different dynamic iteration, and before storing function summaries
  /// where per-function SSA identity is no longer meaningful.
  void invalidateIterationInfo();

  void dump() {
    auto &err = llvm::errs();
    err << "Block Interval:\n";
    err << "  Read Intervals:\n";
    for (auto &[slice, ops] : syncReadSlices) {
      err << "    ";
      slice.print(err);
      err << " ";
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
    err << "  Write Intervals:\n";
    for (auto &[slice, ops] : syncWriteSlices) {
      err << "    ";
      slice.print(err);
      err << " ";
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
  }

  /// Returns true if Slices in two BlockInfo objects are intersected.
  bool isIntersected(const BlockInfo &other, MembarFilterFn filter,
                     Allocation *allocation,
                     MembarSliceFilterFn sliceFilter = nullptr) const {
    return /*RAW*/ isIntersected(syncWriteSlices, other.syncReadSlices,
                                 /*lhsIsRead=*/false, /*rhsIsRead=*/true,
                                 filter, sliceFilter, allocation) ||
           /*WAR*/
           isIntersected(syncReadSlices, other.syncWriteSlices,
                         /*lhsIsRead=*/true, /*rhsIsRead=*/false, filter,
                         sliceFilter, allocation) ||
           /*WAW*/
           isIntersected(syncWriteSlices, other.syncWriteSlices,
                         /*lhsIsRead=*/false, /*rhsIsRead=*/false, filter,
                         sliceFilter, allocation);
  }

  /// Clears the effects because a barrier is inserted.
  void sync() {
    syncReadSlices.clear();
    syncWriteSlices.clear();
    threadSync = {};
    threadDemands = {};
  }

  bool operator==(const BlockInfo &) const = default;

  /// Checks one pair of access maps with the same alias and operation filters.
  static bool isIntersected(const SliceMapT &lhsSlices,
                            const SliceMapT &rhsSlices, bool lhsIsRead,
                            bool rhsIsRead, MembarFilterFn filter,
                            MembarSliceFilterFn sliceFilter,
                            Allocation *allocation) {
    for (auto &lhs : lhsSlices)
      for (auto &rhs : rhsSlices)
        if (lhs.first.intersects(rhs.first))
          if (!sliceFilter || !sliceFilter(lhs.first, rhs.first, lhsIsRead,
                                           rhsIsRead, allocation))
            for (auto lhsOp : lhs.second)
              for (auto rhsOp : rhs.second)
                if (!filter || !filter(lhsOp, rhsOp, lhsIsRead, rhsIsRead,
                                       allocation, lhs.first, rhs.first))
                  return true;
    return false;
  }

private:
  void joinThreadState(const BlockInfo &other);
};

/// Classify the barriers that synchronize local memory accesses in `op`
/// relative to its memory effects.
triton::BarrierStages getLocalBarrierStages(Operation *op,
                                            Allocation *allocation);

/// Tracks memory and thread-ordering state at the current program point and at
/// function boundaries.
struct MembarInfo {
  /// Effects since the most recent whole-region synchronization.
  BlockInfo pending;

  /// Effects reachable from the function entry block before the first
  /// whole-region synchronization.
  /// It keeps incrementing during the iterative algorithm until
  ///  we note all paths to a basic block has synchronized.
  BlockInfo entryBlockInfo;

  /// Whether every path from the function entry block to the current program
  /// point has synchronized the whole execution region.
  bool allPathsFromEntrySynced = false;

  /// Every warp has synchronized since the last memory or thread effect.
  /// This does not discharge dependencies between different warps.
  bool warpsSynced = false;

  MembarInfo &join(const MembarInfo &other) {
    pending.join(other.pending);
    entryBlockInfo.join(other.entryBlockInfo);
    allPathsFromEntrySynced &= other.allPathsFromEntrySynced;
    warpsSynced &= other.warpsSynced;
    return *this;
  }

  void addBlockInfo(const BlockInfo &blockInfo) {
    if (!allPathsFromEntrySynced)
      entryBlockInfo.join(blockInfo);
    pending.join(blockInfo);
    if (blockInfo.hasEffects())
      warpsSynced = false;
  }

  void syncWarps() { warpsSynced = true; }

  void sync() {
    pending.sync();
    allPathsFromEntrySynced = true;
    syncWarps();
  }

  void applyCallSummary(const MembarInfo &callee) {
    if (!allPathsFromEntrySynced)
      entryBlockInfo.join(callee.entryBlockInfo);
    if (callee.allPathsFromEntrySynced)
      sync();
    pending.join(callee.pending);
    // Keep warp coverage local to the current execution region.
    warpsSynced = false;
  }

  template <typename Transform> void transformSlices(Transform transform) {
    pending.transformSlices(transform);
    entryBlockInfo.transformSlices(transform);
  }

  bool operator==(const MembarInfo &) const = default;
};

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

// Shared stage-loop discovery and slice transport for membar and fence
// analyses.
class MembarStageAnalysis {
protected:
  // A deliberately bounded CF loop: preheader -> header -> body -> header,
  // with a signed exclusive test, constant nonnegative start and unit step.
  struct StageLoop {
    Block *entry;
    Block *header;
    Block *body;
    Value induction;
    unsigned initialValue;
  };
  SmallVector<StageLoop> stageLoops;
  void discoverStageLoops(FunctionOpInterface function);
  Value getStageBasis(Operation *operation) const;
  BlockInfo transferStageEdge(const BlockInfo &info, Block *from,
                              Block *to) const;
};

// Common class to analyze fence placement with BlockInfo state.
class MembarOrFenceAnalysis
    : public triton::PostOrderFunctionAnalysis<BlockInfo>,
      protected MembarStageAnalysis {
public:
  MembarOrFenceAnalysis(Allocation &allocation, MembarFilterFn filter,
                        MembarScratchSync scratchSync = {})
      : allocation(allocation), filter(std::move(filter)),
        scratchSync(std::move(scratchSync)) {}

  void run(FunctionOpInterface function, FuncMapT &funcMap);

protected:
  BlockInfo transferEdge(const BlockInfo &info, Block *from,
                         Block *to) const override;

  Allocation &allocation;
  MembarFilterFn filter;
  MembarScratchSync scratchSync;
};

class MembarAnalysis : public triton::PostOrderFunctionAnalysis<MembarInfo>,
                       protected MembarStageAnalysis {
public:
  enum class AccessMode { AllSharedAccesses, AllocatorAliasesOnly };

  /// Creates a new Membar analysis that generates the shared memory barrier
  /// in the following circumstances:
  /// - RAW: If a shared memory write is followed by a shared memory read, and
  /// their addresses are intersected, a barrier is inserted.
  /// - WAR: If a shared memory read is followed by a shared memory write, and
  /// their addresses are intersected, a barrier is inserted.
  /// The following circumstances do not require a barrier:
  /// - WAW: not possible because overlapped memory allocation is not allowed.
  /// - RAR: no write is performed.
  /// Temporary storage of operations such as Reduce are considered as both
  /// a shared memory read. If the temporary storage is written but not read,
  /// it is considered as the problem of the operation itself but not the membar
  /// analysis.
  MembarAnalysis(Allocation &allocation, MembarFilterFn filter,
                 MembarScratchSync scratchSync,
                 triton::BufferRegionAnalysis &regions,
                 MembarSliceFilterFn sliceFilter = nullptr,
                 AccessMode accessMode = AccessMode::AllSharedAccesses)
      : allocation(allocation), filter(std::move(filter)), regions(regions),
        scratchSync(std::move(scratchSync)),
        sliceFilter(std::move(sliceFilter)), accessMode(accessMode),
        bufferIndexAnalysis(
            cast<FunctionOpInterface>(allocation.getOperation())) {}

  void run(FunctionOpInterface function, FuncMapT &funcMap);

private:
  MembarInfo transferEdge(const MembarInfo &info, Block *from,
                          Block *to) const override;

  void updateSuccessor(Operation *terminator, Block *successor,
                       MembarInfo *membarInfo) override;

  void updateExitState(MembarInfo *membarInfo) override;

protected:
  /// Shared-memory lifetime markers do not issue memory accesses.
  static bool hasThreadEffects(Operation *operation);

  /// A null builder analyzes existing synchronization without inserting any.
  void update(Operation *operation, MembarInfo *membarInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;

  void updateMemoryEffects(Operation *operation, MembarInfo *membarInfo,
                           FuncMapT *funcMap, OpBuilder *builder,
                           bool cluster = false, BlockInfo effects = {});
  /// Returns the inserted barrier, or nullptr when none is inserted.
  Operation *syncIfNeeded(Operation *operation, const BlockInfo &effects,
                          MembarInfo *membarInfo, OpBuilder *builder,
                          bool cluster = false);
  virtual Operation *insertBarrier(Operation *operation, OpBuilder *builder,
                                   bool cluster = false);
  virtual triton::BarrierStages getBarrierStages(Operation *operation);
  /// Returns true if the same block has a later wait or barrier, as classified
  /// by getBarrierStages, before any memory effect or nested control flow.
  bool hasSyncPointBeforeMemoryEffect(Operation *operation);

  /// Whether pending thread effects must rendezvous before upcoming demands.
  virtual bool requiresThreadSync(const BlockInfo &pending,
                                  const BlockInfo &effects);

  Allocation &allocation;
  MembarFilterFn filter;
  triton::BufferRegionAnalysis &regions;

private:
  MembarScratchSync scratchSync;
  // Materialize only the final decisions, after CFG analysis has converged.
  llvm::SmallPtrSet<Operation *, 16> scratchSyncOps;
  SmallVector<AllocationSlice>
  getAllocationSlices(Value value, Value stageBasis = {},
                      std::optional<triton::gpu::SharedKind> sharedKind = {});
  bool isRegionLocal(Value value);
  bool mayNotifyPeer(Operation *op);
  BlockInfo getThreadEffects(Operation *op);
  void addThreadDemand(BlockInfo &effects, Operation *op);

  MembarSliceFilterFn sliceFilter;
  AccessMode accessMode;
  BufferIndexAnalysis bufferIndexAnalysis;
  DenseMap<Value, bool> regionLocalAllocations;
};

/// Inserts shared-memory and operation rendezvous barriers across a module,
/// before async-completion analyses consume the synchronization points.
/// Function summaries retain entry-prefix and pending exit states for calls.
class ModuleMembarAnalysis {
public:
  ModuleMembarAnalysis(ModuleAllocation &moduleAllocation,
                       MembarFilterFn filter = nullptr,
                       MembarScratchSync scratchSync = {})
      : moduleAllocation(moduleAllocation), filter(std::move(filter)),
        scratchSync(std::move(scratchSync)) {}

  void run();

  template <typename AnalysisT> void run() {
    // Geometry and interval slices use the same completed allocation, including
    // backend scratch sizes and shared-memory partition offsets.
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<triton::BufferRegionAnalysis>(
        triton::BufferRegionAnalysis::Mode::AllMemory, &moduleAllocation);
    if (failed(solver->initializeAndRun(moduleAllocation.getModuleOp())))
      llvm::report_fatal_error("failed to analyze allocated buffer regions");
    runAnalysis<AnalysisT>(*regions);
  }

  template <typename AnalysisT>
  void runAnalysis(triton::BufferRegionAnalysis &regions) {
    AnalysisT::runModule(
        moduleAllocation.getModuleOp(), [&](FunctionOpInterface function) {
          auto &allocation = *moduleAllocation.getFuncData(function);
          return AnalysisT(allocation, filter, scratchSync, regions);
        });
  }

private:
  ModuleAllocation &moduleAllocation;
  MembarFilterFn filter;
  MembarScratchSync scratchSync;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
