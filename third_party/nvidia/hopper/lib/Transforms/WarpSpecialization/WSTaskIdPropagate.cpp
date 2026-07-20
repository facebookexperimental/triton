#include "CodePartitionUtility.h"
#include "TaskIdPropagation.h"
#include "Utility.h"
#include "WarpSpecializationPipeline.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "nvgpu-ws-partition-id-propagate"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::dataflow;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {

/// Given a TMEMStoreOp, check its source value for ttg.partition.
/// Traverse back through the def chain looking for an operation with
/// ttg.partition set.
static SmallVector<WSPartitionId>
findPartitionIdsFromTMEMStoreSource(ttng::TMEMStoreOp storeOp) {
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

    auto partitionIds = getWSPartitionIds(defOp);
    if (!partitionIds.empty()) {
      return partitionIds;
    }

    // Continue traversing backward through operands
    for (Value operand : defOp->getOperands()) {
      workList.push_back(operand);
    }
  }
  return {};
}

static ttng::TMEMAllocOp findBaseTMEMAlloc(Value desc) {
  SmallVector<Value> workList;
  DenseSet<Value> visited;
  workList.push_back(desc);

  while (!workList.empty()) {
    Value current = workList.pop_back_val();
    if (visited.contains(current))
      continue;
    visited.insert(current);

    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    if (auto allocOp = dyn_cast<ttng::TMEMAllocOp>(defOp))
      return allocOp;

    if (auto subSliceOp = dyn_cast<ttng::TMEMSubSliceOp>(defOp)) {
      workList.push_back(subSliceOp.getSrc());
      continue;
    }

    if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(defOp)) {
      workList.push_back(indexOp.getSrc());
      continue;
    }

    if (auto reinterpretOp = dyn_cast<ttg::MemDescReinterpretOp>(defOp)) {
      workList.push_back(reinterpretOp.getSrc());
      continue;
    }

    if (auto subSliceOp = dyn_cast<ttg::MemDescSubsliceOp>(defOp)) {
      workList.push_back(subSliceOp.getSrc());
      continue;
    }
  }
  return nullptr;
}

/// Handle operand D for MMA ops with ttg.partition set.
/// This function finds TMEMStoreOp (initialization) before the loop
/// containing the MMA and assigns ttg.partition to it if not already set.
static void handleOperandDPartitionPropagation(triton::FuncOp &funcOp) {
  funcOp.walk([&](ttng::MMAv5OpInterface mmaOp) {
    // Step 1: Check if the MMA op has ttg.partition set.
    auto mmaPartitionIds = getWSPartitionIds(mmaOp);
    if (mmaPartitionIds.empty())
      return;

    LDBG("Found MMA op with ttg.partition: " << mmaOp);

    // Step 2: Traverse the accumulator operand to find the TMEM alloc.
    Value dOperand = mmaOp.getAccumulator();
    auto tmemAllocOp = findBaseTMEMAlloc(dOperand);
    if (!tmemAllocOp)
      return;

    // Find the loop containing the MMA
    auto loop = mmaOp->getParentOfType<LoopLikeOpInterface>();
    if (!loop) {
      LDBG("MMA op is not inside a loop");
      return;
    }

    // Step 3: Find the TMEMStoreOp before the loop
    for (auto user : dOperand.getUsers()) {
      auto storeOp = dyn_cast<ttng::TMEMStoreOp>(user);
      if (!storeOp)
        continue;

      // Check if this store is outside and before the loop
      if (loop->isProperAncestor(storeOp) || !appearsBefore(storeOp, loop))
        continue;

      // Find the earliest user with a partition ID to use as the source.
      Operation *partitionIdSource = mmaOp;
      for (auto otherUser : dOperand.getUsers()) {
        if (otherUser == storeOp || otherUser == partitionIdSource)
          continue;
        auto otherPartitionIds = getWSPartitionIds(otherUser);
        if (otherPartitionIds.empty())
          continue;
        // Check if this user is earlier than the current partitionIdSource.
        if (!partitionIdSource || appearsBefore(otherUser, partitionIdSource)) {
          partitionIdSource = otherUser;
        }
      }

      // Step 4: Check if the TMEMStoreOp already has ttg.partition.
      auto storePartitionIds = getWSPartitionIds(storeOp);
      if (!storePartitionIds.empty()) {
        LDBG("TMEMStoreOp already has ttg.partition: " << storeOp);
        continue;
      }

      // Step 5: Look for ttg.partition along the initialization value's
      // creation.
      SmallVector<WSPartitionId> srcPartitionIds =
          findPartitionIdsFromTMEMStoreSource(storeOp);

      if (!srcPartitionIds.empty()) {
        LDBG("Found ttg.partition from source: assigning to TMEMStoreOp");
        setWSPartitionIds(storeOp, srcPartitionIds);
      } else {
        // Step 6: If no source partition is found, assign the partition from
        // the earliest matching user.
        LDBG("No ttg.partition from source, using earliest user's partition");
        // Get the partition IDs from the earliest matching user
        auto partitionIdsToPropagate = getWSPartitionIds(partitionIdSource);
        setWSPartitionIds(storeOp, partitionIdsToPropagate);
      }
    }
  });
}

LogicalResult doTaskIdPropagate(triton::FuncOp funcOp) {
  // Compute the min partition to normalize to 0.
  int64_t minPartition = INT64_MAX;
  funcOp.walk([&](mlir::Operation *op) {
    if (auto attr =
            op->getAttrOfType<DenseI32ArrayAttr>(ttg::kPartitionAttrName)) {
      for (int64_t idx : attr.asArrayRef()) {
        assert(idx >= 0);
        minPartition = std::min(idx, minPartition);
      }
    }
  });
  DenseSet<WSPartitionId> totalPartitionIds;
  // Normalize ttg.partition indices to start at 0, in place. ttg.partition is
  // the single partition representation for the whole WS pipeline.
  funcOp.walk([&](mlir::Operation *op) {
    if (auto attr =
            op->getAttrOfType<DenseI32ArrayAttr>(ttg::kPartitionAttrName)) {
      SmallVector<WSPartitionId> ids;
      for (int64_t rawIdx : attr.asArrayRef()) {
        WSPartitionId idx = static_cast<WSPartitionId>(rawIdx - minPartition);
        assert(idx >= 0);
        totalPartitionIds.insert(idx);
        ids.push_back(idx);
      }
      setWSPartitionIds(op, ids);
    }
  });

  // Handle operand D for MMA ops - propagate partition IDs to initialization
  // TMEMStoreOps before loops.
  handleOperandDPartitionPropagation(funcOp);

  // Existing ttg.partition anchors also contribute to the global partition
  // union. In partition-only inputs there may be no ttg.partition attrs to
  // normalize, but loops, assumes, and loop bounds still need to be visible to
  // all partitions.
  funcOp.walk([&](mlir::Operation *op) {
    for (WSPartitionId partitionId : getWSPartitionIds(op))
      totalPartitionIds.insert(partitionId);
  });

  std::vector<int> allPartitionsVec(totalPartitionIds.begin(),
                                    totalPartitionIds.end());
  ArrayRef<WSPartitionId> allPartitions(allPartitionsVec);

  // Hack: set ttg.partition to all partitions for all assume ops.
  // This is not necesssarily generally desirable because it could
  // force data into multiple partitions. However, for now we will
  // assume this is for the inputs and can state this as needed.
  funcOp.walk([&](LLVM::AssumeOp op) { setWSPartitionIds(op, allPartitions); });

  // Mark all loops with all partitions. We assume DCE can prune any unused
  // loops. Also propagate to scf.for loop bounds (start, stop, step) since
  // they are outside the loop body.
  funcOp.walk([&](scf::ForOp op) {
    setWSPartitionIds(op, allPartitions);
    if (auto *defOp = op.getLowerBound().getDefiningOp())
      addWSPartitionIds(defOp, allPartitions);
    if (auto *defOp = op.getUpperBound().getDefiningOp())
      addWSPartitionIds(defOp, allPartitions);
    if (auto *defOp = op.getStep().getDefiningOp())
      addWSPartitionIds(defOp, allPartitions);
  });
  funcOp.walk([&](scf::WhileOp op) { setWSPartitionIds(op, allPartitions); });

  SymbolTableCollection symbolTable;
  Operation *op = funcOp.getOperation();
  DataFlowSolver solver;

  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<ttg::TaskIdBackwardPropagation>(symbolTable);
  if (failed(solver.initializeAndRun(op)))
    return failure();

  funcOp.walk([&](mlir::Operation *op) {
    auto propagatedPartitionIds = ttg::TaskId::getUninitialized();
    // Get the union of the results
    for (auto result : op->getResults()) {
      auto *lattice = solver.lookupState<ttg::TaskIdLattice>(result);
      if (!lattice)
        llvm_unreachable("Lattice not found.");
      propagatedPartitionIds = propagatedPartitionIds.meet(
          propagatedPartitionIds, lattice->getValue());
    }
    // Get the union of the operands
    if (op->getNumResults() == 0) {
      for (auto operand : op->getOperands()) {
        auto *lattice = solver.lookupState<ttg::TaskIdLattice>(operand);
        if (!lattice)
          llvm_unreachable("Lattice not found.");
        propagatedPartitionIds = propagatedPartitionIds.meet(
            propagatedPartitionIds, lattice->getValue());
      }
    }
    // TODO(Arda): Ideally front-end should not allow constant ops to be
    // annotated. Anchor constants cause problems.
    bool isScalarArithOrMath =
        isa<arith::ArithDialect, math::MathDialect>(op->getDialect()) &&
        llvm::none_of(op->getResultTypes(),
                      [](Type t) { return isa<RankedTensorType>(t); });
    bool isAnchor =
        !isScalarArithOrMath && op->hasAttr(ttg::kPartitionAttrName);
    if (!propagatedPartitionIds.isUninitialized() &&
        (isa<arith::ConstantOp>(op) || !isAnchor)) {
      // For non-anchor ops with existing annotations, merge the lattice
      // value with the annotation to preserve the original partition
      // assignment.
      if (auto existing =
              op->getAttrOfType<DenseI32ArrayAttr>(ttg::kPartitionAttrName)) {
        propagatedPartitionIds =
            ttg::TaskId::meet(propagatedPartitionIds, ttg::TaskId(existing));
      }
      op->setAttr(ttg::kPartitionAttrName, propagatedPartitionIds.getTaskIds());
    }
  });
  // Re-propagate all partitions to ForOp loop bounds after the solver. The
  // solver may have overridden constants with a narrower set of partitions. We
  // also do this before the solver in case the bounds are not constants.
  funcOp.walk([&](scf::ForOp op) {
    if (auto *defOp = op.getLowerBound().getDefiningOp())
      addWSPartitionIds(defOp, allPartitions);
    if (auto *defOp = op.getUpperBound().getDefiningOp())
      addWSPartitionIds(defOp, allPartitions);
    if (auto *defOp = op.getStep().getDefiningOp())
      addWSPartitionIds(defOp, allPartitions);
  });
  // The parent operations must have the union of their children's operations.
  // We do this in a separate walk to avoid having a parent operation treated
  // like an anchor op and skipped by the first walk.
  funcOp.walk([&](mlir::Operation *op) { labelParentOps(op); });
  return success();
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
      auto partitionIds = getWSPartitionIds(op);
      if (!partitionIds.empty()) {
        std::sort(partitionIds.begin(), partitionIds.end());
        setWSPartitionIds(op, partitionIds);
        if (!isa<arith::ConstantOp, arith::ConstantIntOp>(op))
          anchorOps.insert(op);
        if (numWarpGroups == 0)
          op->removeAttr(ttg::kPartitionAttrName);
      }
    });
    if (numWarpGroups == 0 || anchorOps.empty())
      return;
    if (failed(doTaskIdPropagate(funcOp)))
      signalPassFailure();
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
