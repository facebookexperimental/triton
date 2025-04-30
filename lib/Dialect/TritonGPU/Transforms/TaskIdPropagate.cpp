#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Analysis/TaskIdPropagation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/DataFlowFramework.h"

#define DEBUG_TYPE "triton-gpu-taskid-propagate"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUTASKIDPROPAGATE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#ifndef NDEBUG
static bool oneVecCoversTheOther(SmallVector<AsyncTaskId> &one,
                                 SmallVector<AsyncTaskId> &other) {
  // Every element of other appears in one.
  for (AsyncTaskId t : other) {
    // If t doesn't appear in one, return false.
    bool found = false;
    for (AsyncTaskId t2 : one) {
      if (t2 == t) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

struct AsyncTaskIdsCompare {
  static SmallVector<AsyncTaskId> getEmptyKey() {
    SmallVector<AsyncTaskId> V;
    V.push_back(reinterpret_cast<AsyncTaskId>(-1));
    return V;
  }

  static SmallVector<AsyncTaskId> getTombstoneKey() {
    SmallVector<AsyncTaskId> V;
    V.push_back(reinterpret_cast<AsyncTaskId>(-2));
    return V;
  }

  static unsigned getHashValue(const SmallVector<AsyncTaskId> &V) {
    return static_cast<unsigned>(llvm::hash_combine_range(V.begin(), V.end()));
  }

  static bool isEqual(const SmallVector<AsyncTaskId> &LHS,
                      const SmallVector<AsyncTaskId> &RHS) {
    return LHS == RHS;
  }
};

// Make sure the def chain contains the right taskId.
bool verifyTaskId(DataFlowSolver &solver, triton::FuncOp &funcOp,
                  const llvm::DenseSet<Operation *> &anchorOps) {
  bool retCode = true;
  DenseSet<SmallVector<AsyncTaskId>, AsyncTaskIdsCompare> anchorAsyncTasks;
  for (auto anchorOp : anchorOps) {
    anchorAsyncTasks.insert(getAsyncTaskIds(anchorOp));
  }

  funcOp.walk([&](Operation *op) {
    // Skip control ops
    if (llvm::isa<ReturnOp, FuncOp, scf::YieldOp, scf::ForOp>(op))
      return;

    auto deadCodeLattice =
        solver.lookupState<Executable>(solver.getProgramPointBefore(op));
    auto asyncTaskIds = getAsyncTaskIds(op);
    // If it is not a dead code, it should have a taskId.
    if (asyncTaskIds.empty() && deadCodeLattice) {
      LLVM_DEBUG({
        LDBG("Op does not have task id");
        op->dump();
      });
      llvm_unreachable("Op does not have task id");
    }

    auto partitionShouldBeUsedSpecified = [](Operation *op) {
      if (isa<StoreOp, ExperimentalDescriptorLoadOp>(op))
        return true;
      if (isa<AtomicRMWOp, AtomicCASOp>(op))
        return true;
      if (op->hasTrait<OpTrait::DotLike>())
        return true;
      return false;
    };

    if (!anchorAsyncTasks.contains(asyncTaskIds)) {
      if (partitionShouldBeUsedSpecified(op)) {
        LLVM_DEBUG({
          LDBG("async tasks not specified by user");
          op->dump();
        });
        llvm_unreachable("async tasks not specified by user");
      }
    }

    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (llvm::isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(defOp))
        continue;
      auto defTaskIds = getAsyncTaskIds(defOp);
      // Make sure defTaskIds cover asyncTaskIds. Call addAsyncTaskIds if
      // necessary.
      LLVM_DEBUG({
        if (!oneVecCoversTheOther(defTaskIds, asyncTaskIds)) {
          // print defOp and op
          LDBG("Def op does not cover op");
          LDBG("Def op");
          defOp->dump();
          LDBG("op");
          op->dump();
        }
      });
      assert(oneVecCoversTheOther(defTaskIds, asyncTaskIds) &&
             "defTaskIds should cover asyncTaskIds");
    }
  });
  return retCode;
}
#endif

class TritonGPUTaskIdPropagatePass
    : public impl::TritonGPUTaskIdPropagateBase<TritonGPUTaskIdPropagatePass> {
public:
  using impl::TritonGPUTaskIdPropagateBase<
      TritonGPUTaskIdPropagatePass>::TritonGPUTaskIdPropagateBase;

  void labelParentOps(Operation *op) {
    auto asyncTaskIds = getAsyncTaskIds(op);
    auto parent = op->getParentOp();
    while (parent && !isa<triton::FuncOp>(parent)) {
      addAsyncTaskIds(parent, asyncTaskIds);
      parent = parent->getParentOp();
    }
  }

  void runOnOperation() override {

    llvm::DenseSet<Operation *> anchorOps;
    getOperation()->walk([&](triton::FuncOp funcOp) {
      funcOp.walk([&](mlir::Operation *op) {
        auto asyncTasks = getAsyncTaskIds(op);
        if (!asyncTasks.empty()) {
          std::sort(asyncTasks.begin(), asyncTasks.end());
          setAsyncTaskIds(op, asyncTasks);
          if (!isa<arith::ConstantOp, arith::ConstantIntOp>(op))
            anchorOps.insert(op);
          if (numConsumerGroups == 0)
            op->removeAttr("async_task_id");
        }
      });
    });
    if (numConsumerGroups == 0 || anchorOps.empty())
      return;

    SymbolTableCollection symbolTable;
    Operation *op = getOperation();
    DataFlowSolver solver;

    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<TaskIdBackwardPropagation>(symbolTable);
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    // Annotate the ops with the results from the dataflow analysis.
    getOperation()->walk([&](triton::FuncOp funcOp) {
      funcOp.walk([&](mlir::Operation *op) {
        auto taskIds = TaskId::getUninitialized();
        // Get the union of the results
        for (auto result : op->getResults()) {
          auto *lattice = solver.lookupState<TaskIdLattice>(result);
          if (!lattice)
            llvm_unreachable("Lattice not found.");
          taskIds = taskIds.meet(taskIds, lattice->getValue());
        }
        // Get the union of the operands
        if (op->getNumResults() == 0) {
          for (auto operand : op->getOperands()) {
            auto *lattice = solver.lookupState<TaskIdLattice>(operand);
            if (!lattice)
              llvm_unreachable("Lattice not found.");
            taskIds = taskIds.meet(taskIds, lattice->getValue());
          }
        }
        // TODO(Arda): Ideally front-end should not allow constant ops to be
        // annotated. Anchor constants cause problems.
        if (!taskIds.isUninitialized() &&
            (isa<arith::ConstantOp>(op) || !op->hasAttr("async_task_id"))) {
          op->setAttr("async_task_id", taskIds.getTaskIds());
          labelParentOps(op);
        }
      });
#ifndef NDEBUG
      verifyTaskId(solver, funcOp, anchorOps);
#endif
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
