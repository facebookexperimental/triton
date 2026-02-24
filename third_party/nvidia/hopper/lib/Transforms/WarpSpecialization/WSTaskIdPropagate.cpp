#include "CodePartitionUtility.h"
#include "TaskIdPropagation.h"
#include "Utility.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "nvgpu-ws-task-id-propagate"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::dataflow;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {

/// Given a TMEMStoreOp, check its source value for async_task_id.
/// Traverse back through the def chain looking for an operation with
/// async_task_id set.
static SmallVector<AsyncTaskId>
findAsyncIdFromTMEMStoreSource(ttng::TMEMStoreOp storeOp) {
  Value src = storeOp.getSrc();
  SmallVector<Value> workList;
  DenseSet<Value> visited;
  workList.push_back(src);

  while (!workList.empty()) {
    Value current = workList.pop_back_val();
    if (visited.contains(current))
      continue;
    visited.insert(current);

    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    auto taskIds = getAsyncTaskIds(defOp);
    if (!taskIds.empty()) {
      return taskIds;
    }

    // Continue traversing backward through operands
    for (Value operand : defOp->getOperands()) {
      workList.push_back(operand);
    }
  }
  return {};
}

/// Handle operand D for MMA ops with task_id set.
/// This function finds TMEMStoreOp (initialization) before the loop
/// containing the MMA and assigns async_task_id to it if not already set.
static void handleOperandDTaskIdPropagation(triton::FuncOp &funcOp) {
  funcOp.walk([&](ttng::TCGen5MMAOp mmaOp) {
    // Step 1: Check if the MMA op has a task_id set.
    auto mmaTaskIds = getAsyncTaskIds(mmaOp);
    if (mmaTaskIds.empty())
      return;

    LDBG("Found MMA op with task_id: " << mmaOp);

    // Step 2: Traverse operand D to find the TMEM alloc.
    Value dOperand = mmaOp.getD();
    auto *allocOp = dOperand.getDefiningOp();
    if (!allocOp)
      return;

    auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(allocOp);
    if (!tmemAllocOp) {
      // Try to trace through subview or similar
      return;
    }

    // Find the for loop containing the MMA
    auto forOp = mmaOp->getParentOfType<scf::ForOp>();
    if (!forOp) {
      LDBG("MMA op is not inside a scf.for loop");
      return;
    }

    // Step 3: Find the TMEMStoreOp before the loop
    for (auto user : tmemAllocOp.getResult().getUsers()) {
      auto storeOp = dyn_cast<ttng::TMEMStoreOp>(user);
      if (!storeOp)
        continue;

      // Check if this store is outside and before the loop
      if (forOp->isProperAncestor(storeOp) || !appearsBefore(storeOp, forOp))
        continue;

      // Find the earliest user with an async task ID to use as the source.
      Operation *taskIdSource = mmaOp;
      for (auto otherUser : tmemAllocOp.getResult().getUsers()) {
        if (otherUser == storeOp || otherUser == taskIdSource)
          continue;
        auto otherTaskIds = getAsyncTaskIds(otherUser);
        if (otherTaskIds.empty())
          continue;
        // Check if this user is earlier than the current taskIdSource
        if (!taskIdSource || appearsBefore(otherUser, taskIdSource)) {
          taskIdSource = otherUser;
        }
      }

      // Step 4: Check if the TMEMStoreOp already has a task_id
      auto storeTaskIds = getAsyncTaskIds(storeOp);
      if (!storeTaskIds.empty()) {
        LDBG("TMEMStoreOp already has task_id: " << storeOp);
        continue;
      }

      // Step 5: Look for async_id along the initialization value's creation
      SmallVector<AsyncTaskId> srcAsyncId =
          findAsyncIdFromTMEMStoreSource(storeOp);

      if (!srcAsyncId.empty()) {
        LDBG("Found async_id from source: assigning to TMEMStoreOp");
        setAsyncTaskIds(storeOp, srcAsyncId);
      } else {
        // Step 6: If no async_id found, assign the async_id from the earliest
        // matching user
        LDBG("No async_id from source, using task_id from earliest user");
        // Get the task IDs from the earliest matching user
        auto taskIdsToPropagate = getAsyncTaskIds(taskIdSource);
        setAsyncTaskIds(storeOp, taskIdsToPropagate);
      }
    }
  });
}

int doTaskIdPropagate(triton::FuncOp &funcOp) {
  // Compute the min partition to normalize to 0
  int64_t minPartition = INT64_MAX;
  funcOp.walk([&](mlir::Operation *op) {
    if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(kPartitionAttrName)) {
      assert(attr.size() == 1 && "expected exactly 1 partition element");
      int64_t idx = attr[0];
      assert(idx >= 0);
      minPartition = std::min(idx, minPartition);
    }
  });
  DenseSet<AsyncTaskId> totalTaskIds;
  // Convert ttg.partition to async_task_id
  funcOp.walk([&](mlir::Operation *op) {
    if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(kPartitionAttrName)) {
      assert(attr.size() == 1 && "expected exactly 1 partition element");
      int64_t idx = attr[0] - minPartition;
      totalTaskIds.insert(idx);
      assert(idx >= 0);
      setAsyncTaskIds(op, idx);
      op->removeAttr(kPartitionAttrName);
    }
  });

  // Handle operand D for MMA ops - propagate task_id to initialization
  // TMEMStoreOps before loops.
  handleOperandDTaskIdPropagation(funcOp);

  std::vector<int> allTasksVec(totalTaskIds.begin(), totalTaskIds.end());
  ArrayRef<AsyncTaskId> allTasks(allTasksVec);

  // Hack: set async_task_id to all tasks for all assume ops.
  // This is not necesssarily generally desirable because it could
  // force data into multiple partitions. However, for now we will
  // assume this is for the inputs and can state this as needed.
  funcOp.walk([&](LLVM::AssumeOp op) { setAsyncTaskIds(op, allTasks); });

  // Mark all forOps with all async tasks. We assume DCE can
  // prune any unused loops. Also propagate to loop bounds (start, stop, step).
  funcOp.walk([&](scf::ForOp op) {
    setAsyncTaskIds(op, allTasks);
    if (auto *defOp = op.getLowerBound().getDefiningOp())
      addAsyncTaskIds(defOp, allTasks);
    if (auto *defOp = op.getUpperBound().getDefiningOp())
      addAsyncTaskIds(defOp, allTasks);
    if (auto *defOp = op.getStep().getDefiningOp())
      addAsyncTaskIds(defOp, allTasks);
  });

  SymbolTableCollection symbolTable;
  Operation *op = funcOp.getOperation();
  DataFlowSolver solver;

  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<ttg::TaskIdBackwardPropagation>(symbolTable);
  if (failed(solver.initializeAndRun(op)))
    return -1;

  funcOp.walk([&](mlir::Operation *op) {
    auto taskIds = ttg::TaskId::getUninitialized();
    // Get the union of the results
    for (auto result : op->getResults()) {
      auto *lattice = solver.lookupState<ttg::TaskIdLattice>(result);
      if (!lattice)
        llvm_unreachable("Lattice not found.");
      taskIds = taskIds.meet(taskIds, lattice->getValue());
    }
    // Get the union of the operands
    if (op->getNumResults() == 0) {
      for (auto operand : op->getOperands()) {
        auto *lattice = solver.lookupState<ttg::TaskIdLattice>(operand);
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
    }
  });
  // The parent operations must have the union of their children's operations.
  // We do this in a separate walk to avoid having a parent operation treated
  // like an anchor op and skipped by the first walk.
  funcOp.walk([&](mlir::Operation *op) { labelParentOps(op); });
  return 0;
}

#define GEN_PASS_DEF_NVGPUTESTWSTASKIDPROPAGATE
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSTaskIdPropagatePass
    : public impl::NVGPUTestWSTaskIdPropagateBase<
          NVGPUTestWSTaskIdPropagatePass> {
public:
  using impl::NVGPUTestWSTaskIdPropagateBase<
      NVGPUTestWSTaskIdPropagatePass>::NVGPUTestWSTaskIdPropagateBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    llvm::DenseSet<Operation *> anchorOps;
    funcOp.walk([&](mlir::Operation *op) {
      auto asyncTasks = getAsyncTaskIds(op);
      if (!asyncTasks.empty()) {
        std::sort(asyncTasks.begin(), asyncTasks.end());
        setAsyncTaskIds(op, asyncTasks);
        if (!isa<arith::ConstantOp, arith::ConstantIntOp>(op))
          anchorOps.insert(op);
        if (numWarpGroups == 0)
          op->removeAttr("async_task_id");
      }
    });
    if (numWarpGroups == 0 || anchorOps.empty())
      return;
    int retCode = doTaskIdPropagate(funcOp);
    if (retCode != 0)
      signalPassFailure();
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
