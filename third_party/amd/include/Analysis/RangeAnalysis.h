#ifndef TRITONAMD_ANALYSIS_RANGE_ANALYSIS_H
#define TRITONAMD_ANALYSIS_RANGE_ANALYSIS_H

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::triton {
class FuncOp;
}

namespace mlir::triton::AMD {

struct IntegerAssumption {
  ConstantIntRanges range;
  Operation *owner;
};
using IntegerAssumptions = DenseMap<Value, SmallVector<IntegerAssumption>>;

/// Integer ranges with scoped assumptions and bounded loop simulation.
struct TritonIntegerRangeAnalysis : dataflow::IntegerRangeAnalysis {
  using dataflow::IntegerRangeAnalysis::IntegerRangeAnalysis;
  using Base = dataflow::IntegerRangeAnalysis;
  TritonIntegerRangeAnalysis(DataFlowSolver &solver,
                             const IntegerAssumptions &assumptions,
                             DominanceInfo *dominanceInfo,
                             bool assumeNoArithOverflow_ = false)
      : dataflow::IntegerRangeAnalysis(solver), assumptions(assumptions),
        domInfo(dominanceInfo), assumeNoArithOverflow(assumeNoArithOverflow_) {}

  /// Set the maximum PID value for a given axis. When set, GetProgramIdOp
  /// for that axis will use [0, maxPID] instead of the default range.
  void setPidBound(int axis, int64_t maxPID) { pidBounds[axis] = maxPID; }

  void setToEntryState(dataflow::IntegerValueRangeLattice *lattice) override;

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) override;

  std::optional<int64_t> maybeGetTripCount(LoopLikeOpInterface loop);

  /// This method (which overloads
  /// AbstractSparseForwardDataFlowAnalysis::visitRegionSuccessors)
  /// implements "abstract interpretation" of loops with statically known bounds
  /// in order to infer tight ranges for loop carried values (and therefore loop
  /// body values). By "abstract interpretation" we mean lattice states are
  /// propagated to all region successors N times, where N is the total trip
  /// count of the loop. Recall for scf.for, both the loop itself and the users
  /// of the loop successors. Thus, after N propagations both loop body values
  /// and users of loop results will have accurate ranges (assuming we have
  /// implemented support for range analysis on the ops).
  /// *Note*, this implementation is majority similar to
  /// AbstractSparseForwardDataFlowAnalysis::visitRegionSuccessors
  /// (so check there for more explanation/insight) and basically only does two
  /// things differently:
  ///
  /// 1. If the branch op is a loop (LoopLikeOpInterface) then we attempt to
  /// compute its total trip count (nested loop trip counts multiply) and
  /// initialize a visit count to 0. Note, due to how Dataflow analysis works we
  /// have to actually visit the loop N times for each iter_arg (each argument
  /// lattice) so we actually track visit count for (loop, arg) not just (loop).
  ///
  /// 2. Before propagating along a backedge, we check if we have propagated for
  /// (loop, arg) >= N - 1 times. Propagation from the loop op itself (the init
  /// args) is never blocked: it seeds the first iteration and, for loops which
  /// may run zero times.
  ///
  /// Note, for loops where the trip count cannot be inferred *and* loops with a
  /// total trip count larger than `kDefaultMaxTripCount`, fallback to
  /// upstream's conservative inference (i.e., we infer [min_int, max_int]) for
  /// the loop operands and all users and all users of the results of the loop.
  void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionSuccessor successor,
      ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) override;

  /// Collect constant comparison bounds and their assumption sites.
  static IntegerAssumptions collectAssumptions(Operation *rootOp,
                                               bool filterConstants = true);

  /// Intersect inferred ranges with assumptions that dominate this use.
  IntegerValueRange getRangeAt(Value value, Operation *use,
                               const IntegerValueRange &inferred) const;

  int64_t getTotalLoopTripCount(LoopLikeOpInterface loop);

  /// The number of times the lattices of `loop` may have to be updated while
  /// simulating it.
  int64_t getLoopSimulationSteps(LoopLikeOpInterface loop);

  /// Simulation steps (see getLoopSimulationSteps) of all loops contained under
  /// the root operation being analyzed. Note, nested loops have trip counts
  /// computed as a product of enclosing loops; i.e. for
  ///   scf.for i = 1 to 10
  ///     scf.for j = 1 to 10
  /// the trip count of the outer loop (on i) is 10 but the trip count of the
  /// inner loop (on j) is 100.
  llvm::SmallDenseMap<LoopLikeOpInterface, int64_t> loopSimulationSteps;

  /// Visit counts tabulating how many times each lattice has been propagated
  /// through each loop. This is used in visitRegionSuccessors to end
  /// propagation when loopVisits[loop, ...] reaches loopSimulationSteps[loop].
  llvm::SmallDenseMap<
      std::pair<LoopLikeOpInterface, dataflow::IntegerValueRangeLattice *>,
      int64_t>
      loopVisits;

  IntegerAssumptions assumptions;

  /// The defaultTransferFunc is the default transfer function for this dataflow
  /// problem.
  /// @param[in] op: the Operation in question
  /// @param[in] result: a particular value defined by this op. Note that op
  ///            may define multiple values.
  /// @param[in] srcLattices: lattices of all source operands
  /// @param[in] destLattices: lattices all all result values
  /// @param[in] incomingRange: the value-range inffered for result
  void defaultTransferFunc(
      Operation *op, Value result,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> srcLattices,
      ArrayRef<dataflow::IntegerValueRangeLattice *> destLattices,
      const IntegerValueRange &incomingRange);

private:
  LogicalResult visitOperationHelper(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices);

  DominanceInfo *domInfo = nullptr;
  bool assumeNoArithOverflow = false;

  /// Optional per-axis PID bounds. When set via setPidBound(), these override
  /// the default kDefaultMaxPrograms for GetProgramIdOp on the given axis.
  llvm::SmallDenseMap<int, int64_t> pidBounds;
};

std::optional<SmallVector<std::optional<ConstantIntRanges>>>
collectRanges(const DataFlowSolver &solver, ValueRange values);

std::optional<bool> evaluateCmpI(const DataFlowSolver &solver,
                                 arith::CmpIOp cmpOp);

bool isEmptyInitializedRange(ConstantIntRanges rv);

void populateFoldTrueCmpIOpPatterns(RewritePatternSet &patterns,
                                    DataFlowSolver *solver);

} // namespace mlir::triton::AMD

#endif
