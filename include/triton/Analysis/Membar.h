#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"
#include "BufferIndexAnalysis.h"
#include "CallGraph.h"
#include "Function.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <set>
#include <tuple>
#include <utility>

namespace mlir {

class OpBuilder;
struct AllocationSlice;

/// Callback to allow backend to provide more information on whether a barrier
/// is needed between two operations. Even though two operations access the same
/// shared memory they may not require a barrier in between them.
using MembarFilterFn =
    std::function<bool(Operation *, Operation *, bool /*lhsIsRead*/,
                       bool /*rhsIsRead*/, Allocation *)>;

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

  AllocationSlice translated(size_t offset,
                             bool invalidateBufferId = false) const {
    AllocationSlice shifted = invalidateBufferId ? forgetStageLoop() : *this;
    shifted.allocationInterval =
        Interval<size_t>(shifted.allocationInterval.start() + offset,
                         shifted.allocationInterval.end() + offset);
    if (invalidateBufferId) {
      shifted.bufferId = Allocation::InvalidBufferId;
      shifted.stage = {};
      // Per-function SSA identities cannot be compared across call sites.
      shifted.bufferIndexExpr = nullptr;
    } else if (shifted.stage.parent) {
      shifted.stage.parentInterval =
          Interval<size_t>(shifted.stage.parentInterval.start() + offset,
                           shifted.stage.parentInterval.end() + offset);
    }
    return shifted;
  }

  void print(raw_ostream &os) const;

  // Buffer-index expression attached by BufferIndexAnalysis. It participates
  // in ordering/equality so accesses to different slots remain separate.
  // Must not be mutated after the slice is inserted into a sorted container
  // (e.g. BlockInfo::SliceMapT); rebuild the container instead, as
  // BufferIndexAnalysis::invalidateBufferIndices does.
  const BufferIndexExpr *bufferIndexExpr = nullptr;

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
             llvm::ArrayRef<int64_t>, StageKey, const BufferIndexExpr *>
  asTuple() const {
    return {allocationInterval,
            bufferId,
            accessTy.getAsOpaquePointer(),
            subsliceOffsets,
            {stage.parent.getAsOpaquePointer(),
             stage.basis.getAsOpaquePointer(), stage.parentInterval,
             stage.stride, stage.numStages, stage.offset},
            bufferIndexExpr};
  }
  // Offsets from subslice. Empty when offsets are unknown
  SmallVector<int64_t> subsliceOffsets;
  // The allocated interval for this buffer
  Interval<size_t> allocationInterval;
  // Type of the memory descriptor for this access
  triton::gpu::MemDescType accessTy;
  // Buffer id for partial sync on wait_barrier deps.
  Allocation::BufferId bufferId;
};

struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  SliceMapT syncReadSlices;
  SliceMapT syncWriteSlices;

  BlockInfo() = default;

  /// Unions two BlockInfo objects.
  BlockInfo &join(const BlockInfo &other) {
    for (auto &slice : other.syncReadSlices)
      syncReadSlices[slice.first].insert(slice.second.begin(),
                                         slice.second.end());

    for (auto &slice : other.syncWriteSlices)
      syncWriteSlices[slice.first].insert(slice.second.begin(),
                                          slice.second.end());
    return *this;
  }

  BlockInfo
  mapSlices(const std::function<AllocationSlice(const AllocationSlice &)> &map)
      const {
    BlockInfo result;
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

  /// Clears the slices because a barrier is inserted.
  void sync() {
    syncReadSlices.clear();
    syncWriteSlices.clear();
  }

  /// Compares two BlockInfo objects.
  bool operator==(const BlockInfo &other) const {
    return syncReadSlices == other.syncReadSlices &&
           syncWriteSlices == other.syncWriteSlices;
  }

  bool operator!=(const BlockInfo &other) const { return !(*this == other); }

private:
  bool isIntersected(const SliceMapT &lhsSlices, const SliceMapT &rhsSlices,
                     bool lhsIsRead, bool rhsIsRead, MembarFilterFn filter,
                     MembarSliceFilterFn sliceFilter,
                     Allocation *allocation) const {
    for (auto &lhs : lhsSlices)
      for (auto &rhs : rhsSlices)
        if (lhs.first.intersects(rhs.first))
          if (!sliceFilter || !sliceFilter(lhs.first, rhs.first, lhsIsRead,
                                           rhsIsRead, allocation))
            for (auto lhsOp : lhs.second)
              for (auto rhsOp : rhs.second)
                if (!filter ||
                    !filter(lhsOp, rhsOp, lhsIsRead, rhsIsRead, allocation))
                  return true;
    return false;
  }
};

/// Returns true if `op` synchronizes local memory accesses for membar-style
/// analyses.
bool containsLocalBarrier(Operation *op);

inline BlockInfo translateBlockInfoToCallsite(const BlockInfo &calleeBlockInfo,
                                              size_t callOffset) {
  BlockInfo translatedBlockInfo;
  auto translateSlices = [&](const BlockInfo::SliceMapT &srcSlices,
                             BlockInfo::SliceMapT &dstSlices) {
    for (const auto &[slice, ops] : srcSlices) {
      auto translatedSlice =
          slice.translated(callOffset, /*invalidateBufferId=*/true);
      auto &dstOps = dstSlices[translatedSlice];
      dstOps.insert(ops.begin(), ops.end());
    }
  };

  translateSlices(calleeBlockInfo.syncReadSlices,
                  translatedBlockInfo.syncReadSlices);
  translateSlices(calleeBlockInfo.syncWriteSlices,
                  translatedBlockInfo.syncWriteSlices);
  return translatedBlockInfo;
}

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

// Common class to analyze membar and fence placement.
class MembarOrFenceAnalysis
    : public triton::PostOrderFunctionAnalysis<BlockInfo> {
public:
  MembarOrFenceAnalysis(Allocation &allocation, MembarFilterFn filter,
                        MembarScratchSync scratchSync = {})
      : allocation(allocation), filter(std::move(filter)),
        scratchSync(std::move(scratchSync)) {}

  void run(FunctionOpInterface function, FuncMapT &funcMap);

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
  BlockInfo transferEdge(const BlockInfo &info, Block *from,
                         Block *to) const override;

  Allocation &allocation;
  MembarFilterFn filter;
  MembarScratchSync scratchSync;
};

class MembarAnalysis : public MembarOrFenceAnalysis {
public:
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
                 MembarScratchSync scratchSync = {})
      : MembarOrFenceAnalysis(allocation, std::move(filter),
                              std::move(scratchSync)),
        bufferIndexAnalysis(
            cast<FunctionOpInterface>(allocation.getOperation())) {}

  void run(FunctionOpInterface function, FuncMapT &funcMap);

private:
  /// Updates the BlockInfo operation based on the operation.
  void update(Operation *operation, BlockInfo *blockInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;

  void updateSuccessor(Operation *terminator, Block *successor,
                       BlockInfo *blockInfo) override;

  void updateExitState(BlockInfo *blockInfo) override;

  void insertBarrier(Operation *operation, OpBuilder *builder);

  // Materialize only the final decisions, after CFG analysis has converged.
  llvm::SmallPtrSet<Operation *, 16> scratchSyncOps;
  BufferIndexAnalysis bufferIndexAnalysis;
};

/// Postorder traversal on the callgraph to insert membar instructions
/// of each function.
/// Each function maintains a BlockInfo map that includes all potential buffers
/// after returning. This way users do not have to explicitly insert membars
/// before and after function calls, but might be a bit conservative.
template <typename AnalysisT>
class ModuleMembarOrFenceAnalysis : public triton::CallGraph<BlockInfo> {
public:
  ModuleMembarOrFenceAnalysis(ModuleAllocation &moduleAllocation,
                              MembarFilterFn filter = nullptr,
                              MembarScratchSync scratchSync = {})
      : triton::CallGraph<BlockInfo>(moduleAllocation.getModuleOp()),
        moduleAllocation(moduleAllocation), filter(std::move(filter)),
        scratchSync(std::move(scratchSync)) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order walk callback
        [&](FunctionOpInterface funcOp) {
          auto &allocation = *moduleAllocation.getFuncData(funcOp);
          if (!funcMap.try_emplace(funcOp).second)
            return;
          AnalysisT(allocation, filter, scratchSync).run(funcOp, funcMap);
        });
  }

private:
  ModuleAllocation &moduleAllocation;
  MembarFilterFn filter;
  MembarScratchSync scratchSync;
};

using ModuleMembarAnalysis = ModuleMembarOrFenceAnalysis<MembarAnalysis>;

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
