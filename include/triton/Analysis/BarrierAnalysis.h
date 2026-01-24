#ifndef TRITON_ANALYSIS_BARRIERANALYSIS_H
#define TRITON_ANALYSIS_BARRIERANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// Barrier Operation Types
//===----------------------------------------------------------------------===//

/// Types of barrier operations in TTGIR
enum class BarrierOpKind {
  // Memory barriers (mbarrier)
  InitBarrier,   // ttng.init_barrier - allocate and initialize mbarrier
  InvalBarrier,  // ttng.inval_barrier - invalidate mbarrier for reuse
  BarrierExpect, // ttng.barrier_expect - set expected transaction bytes
  WaitBarrier,   // ttng.wait_barrier - wait for phase flip
  ArriveBarrier, // ttng.arrive_barrier - signal arrival (non-blocking)

  // Named barriers (hardware pre-allocated 0-15)
  NamedBarrierArrive, // ttng.arrive_barrier_named
  NamedBarrierWait,   // ttng.wait_barrier_named

  // Async operations that interact with barriers
  AsyncCopy, // Async copy that may arrive a barrier
  TMALoad,   // TMA load with implicit barrier arrive
  TMAStore,  // TMA store with implicit barrier arrive
};

/// Convert BarrierOpKind to string for printing
StringRef barrierOpKindToString(BarrierOpKind kind);

//===----------------------------------------------------------------------===//
// Barrier Operation Info
//===----------------------------------------------------------------------===//

/// Information about a single barrier operation
struct BarrierOpInfo {
  /// The operation itself
  Operation *op = nullptr;

  /// Kind of barrier operation
  BarrierOpKind kind;

  /// Barrier ID (for named barriers) or allocation pointer identity
  /// For mbarriers, this is derived from the allocation's defining op
  /// For named barriers, this is the literal barrier number (0-15)
  int64_t barrierId = -1;

  /// For memory barriers: the phase being waited on or arrived
  /// -1 if not applicable or dynamic
  int64_t phase = -1;

  /// For named barriers: number of threads participating
  int64_t numThreads = -1;

  /// For arrive operations: the arrive count
  int64_t arriveCount = 1;

  /// For expect operations: expected transaction bytes
  int64_t expectedBytes = -1;

  /// Async task ID (warp group) this operation belongs to
  /// -1 if not in an async_task region
  int64_t asyncTaskId = -1;

  /// Order index within the function (topological order)
  size_t orderIndex = 0;

  /// Print barrier info
  void print(llvm::raw_ostream &os) const;
};

//===----------------------------------------------------------------------===//
// Barrier Dependency
//===----------------------------------------------------------------------===//

/// Represents a producer-consumer dependency between barrier operations
struct BarrierDependency {
  /// Producer operation (e.g., arrive, expect, TMA load)
  const BarrierOpInfo *producer = nullptr;

  /// Consumer operation (e.g., wait)
  const BarrierOpInfo *consumer = nullptr;

  /// Type of dependency
  enum class Kind {
    ArriveThenWait, // Arrive signals wait
    ExpectThenWait, // Expect bytes + TMA load satisfies wait
    InitThenUse,    // Init must precede all barrier ops
  };
  Kind kind;

  /// Whether this is a cross-warp-group dependency
  bool isCrossWarpGroup() const {
    if (!producer || !consumer)
      return false;
    return producer->asyncTaskId != consumer->asyncTaskId &&
           producer->asyncTaskId != -1 && consumer->asyncTaskId != -1;
  }

  void print(llvm::raw_ostream &os) const;
};

//===----------------------------------------------------------------------===//
// Warp Group Info
//===----------------------------------------------------------------------===//

/// Information about a warp group (async task)
struct WarpGroupInfo {
  int64_t taskId = -1;
  std::string taskName;
  bool isProducer = false;
  bool isConsumer = false;

  /// Barrier operations belonging to this warp group
  SmallVector<const BarrierOpInfo *> barrierOps;

  /// Operations ordered by execution order
  SmallVector<const BarrierOpInfo *> orderedOps;
};

//===----------------------------------------------------------------------===//
// Barrier Execution Order Analysis
//===----------------------------------------------------------------------===//

/// Analysis pass that examines barrier operations in TTGIR to understand
/// execution order and producer/consumer relationships between warp groups.
///
/// This analysis:
/// 1. Collects all barrier operations (mbarrier and named barriers)
/// 2. Groups them by warp group (async task)
/// 3. Identifies producer/consumer dependencies
/// 4. Detects potential execution orderings and race conditions
/// 5. Generates execution trace visualization
class BarrierExecutionOrderAnalysis {
public:
  explicit BarrierExecutionOrderAnalysis(FunctionOpInterface funcOp);

  /// Run the analysis
  void run();

  /// Get all barrier operations
  ArrayRef<BarrierOpInfo> getBarrierOps() const { return barrierOps; }

  /// Get barrier operations for a specific warp group
  ArrayRef<const BarrierOpInfo *>
  getBarrierOpsForWarpGroup(int64_t taskId) const;

  /// Get all identified dependencies
  ArrayRef<BarrierDependency> getDependencies() const { return dependencies; }

  /// Get warp group info
  const WarpGroupInfo *getWarpGroupInfo(int64_t taskId) const;

  /// Get all warp groups
  const std::map<int64_t, WarpGroupInfo> &getWarpGroups() const {
    return warpGroups;
  }

  /// Check for potential issues
  struct AnalysisIssue {
    enum class Kind {
      MissingArrive, // Wait without matching arrive
      MissingWait,   // Arrive without matching wait
      PhaseConflict, // Inconsistent phase usage
      DeadlockRisk,  // Potential deadlock detected
      BarrierReuse,  // Barrier reused without invalidation
    };
    Kind kind;
    SmallVector<const BarrierOpInfo *> involvedOps;
    std::string message;
  };
  ArrayRef<AnalysisIssue> getIssues() const { return issues; }

  /// Print analysis results
  void print(llvm::raw_ostream &os) const;

  /// Print execution trace visualization (ASCII art timeline)
  void printExecutionTrace(llvm::raw_ostream &os) const;

  /// Print dependency graph in DOT format for visualization
  void printDependencyGraph(llvm::raw_ostream &os) const;

private:
  /// Collect all barrier operations from the function
  void collectBarrierOps();

  /// Assign barrier IDs based on allocations
  void assignBarrierIds();

  /// Group operations by warp group
  void groupByWarpGroup();

  /// Analyze dependencies between barrier operations
  void analyzeDependencies();

  /// Detect potential issues
  void detectIssues();

  /// Get the async task ID for an operation
  int64_t getAsyncTaskId(Operation *op) const;

  /// Get barrier allocation identity (for matching arrive/wait pairs)
  std::optional<int64_t> getBarrierAllocId(Operation *op) const;

  /// Trace a value through block arguments to find its origin
  /// This handles WarpSpecializeOp explicit captures
  Value traceValueThroughBlockArgs(Value value) const;

  FunctionOpInterface funcOp;
  SmallVector<BarrierOpInfo> barrierOps;
  std::map<int64_t, WarpGroupInfo> warpGroups;
  SmallVector<BarrierDependency> dependencies;
  SmallVector<AnalysisIssue> issues;

  /// Map from (local_alloc operation, index value) to barrier ID
  /// This helps match barriers when new memdesc_index ops are created
  DenseMap<std::pair<Operation *, int64_t>, int64_t> allocIndexToBarrierId;

  /// Also keep a simple map for direct lookups (memdesc_index defOp → barrier
  /// ID)
  DenseMap<Operation *, int64_t> defOpToBarrierId;
  int64_t nextBarrierId = 0;
};

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_BARRIERANALYSIS_H
