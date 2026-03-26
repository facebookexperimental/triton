#ifndef TRITON_ANALYSIS_BARRIERANALYSIS_H
#define TRITON_ANALYSIS_BARRIERANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::scf {
class ForOp;
} // namespace mlir::scf

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// Barrier Operation Kinds (for the concrete trace)
//===----------------------------------------------------------------------===//

/// Classification of barrier operations following the design doc.
enum class BarrierOpKind {
  ArriveSync,  // ttng.arrive_barrier — pending_arrives -= cnt
  ExpectBytes, // ttng.barrier_expect — arrive (cnt=1) + pending_bytes += sz
  TmaLoad,     // ttng.async_tma_copy_global_to_local — pending_bytes -= xfer
  Wait,        // ttng.wait_barrier — blocks while phase == phi
};

/// Convert BarrierOpKind to string for printing.
StringRef barrierOpKindToString(BarrierOpKind kind);

//===----------------------------------------------------------------------===//
// Concrete Operation Instance (after loop unrolling)
//===----------------------------------------------------------------------===//

/// A single barrier operation after loop unrolling with all parameters
/// resolved to concrete values.
struct ConcreteBarrierOp {
  BarrierOpKind kind;

  /// Which task (warp group) this belongs to.
  int64_t taskId = -1;
  std::string taskName;

  /// Barrier identity: (allocName, slotIndex) uniquely identifies a barrier.
  std::string allocName;
  int64_t slotIndex = -1;

  /// Phase for wait operations (-1 if unresolved).
  int64_t phase = -1;

  /// Arrive count: 1 for ArriveSync/ExpectBytes, 0 for TmaLoad.
  int64_t arriveCount = 1;

  /// Expected bytes for ExpectBytes operations.
  int64_t expectedBytes = -1;

  /// Loop iteration this came from (-1 if not in a loop).
  int64_t iteration = -1;

  /// Position within the task's operation sequence.
  size_t position = 0;

  /// Source location string for diagnostics.
  std::string locInfo;
};

/// Per-task operation trace after loop unrolling.
struct TaskTrace {
  int64_t taskId = -1;
  std::string taskName;
  std::vector<ConcreteBarrierOp> ops;
};

/// Barrier allocation info (from local_alloc + init_barrier).
struct BarrierAllocInfo {
  std::string name;
  int64_t numSlots = 0;
  int64_t arriveCount = 1;
};

//===----------------------------------------------------------------------===//
// BarrierDeadlockAnalysis
//===----------------------------------------------------------------------===//

/// Extracts concrete barrier operation traces from TTGIR by walking
/// warp_specialize regions and unrolling scf.for loops.
///
/// This is Step 1 of the deadlock detection pipeline: building the program
/// model. Step 2 (Z3 constraint encoding) is added in a follow-up PR.
class BarrierDeadlockAnalysis {
public:
  explicit BarrierDeadlockAnalysis(FunctionOpInterface funcOp,
                                   int unrollBound = 0);

  /// Run the full trace extraction pipeline.
  void run();

  /// Print a human-readable summary of the extracted traces.
  void printSummary(llvm::raw_ostream &os) const;

  /// Get the task traces.
  const std::vector<TaskTrace> &getTaskTraces() const { return taskTraces; }

  /// Get barrier allocations.
  const std::vector<BarrierAllocInfo> &getBarrierAllocs() const {
    return barrierAllocs;
  }

private:
  /// Extract barrier allocations (local_alloc ops with i64 element type).
  void collectBarrierAllocs();

  /// Build concrete operation traces by walking warp_specialize regions.
  void buildTaskTraces();

  /// Unroll an scf.for loop body and append concrete ops to trace.
  void unrollLoop(mlir::scf::ForOp forOp, int64_t taskId,
                  const std::string &taskName, TaskTrace &trace,
                  OperandRange captures);

  /// Resolve a barrier Value to (allocName, slotIndex) by tracing def-use
  /// chains through memdesc_index and warp_specialize captures.
  std::pair<std::string, int64_t> resolveBarrier(Value barrierVal,
                                                 OperandRange captures);

  /// Evaluate an integer SSA value to a concrete int at a given loop
  /// iteration. Returns nullopt if the value cannot be resolved.
  std::optional<int64_t> tryEvalInt(Value val, int64_t loopVar,
                                    Value loopInductionVar);

  FunctionOpInterface funcOp;
  int unrollBound;
  std::vector<TaskTrace> taskTraces;
  std::vector<BarrierAllocInfo> barrierAllocs;

  /// Map from local_alloc Operation* to barrier alloc name.
  DenseMap<Operation *, std::string> allocOpToName;

  /// Map capture index to the original value's alloc name.
  std::map<unsigned, std::string> captureIndexToAllocName;
};

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_BARRIERANALYSIS_H
