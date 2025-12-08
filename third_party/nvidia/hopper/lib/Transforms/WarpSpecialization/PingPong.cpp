#include "Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <unordered_set>

#define DEBUG_TYPE "nvgpu-ping-pong-sync"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

// Manages expensive operations for critical region identification and
// assigns unique barrier IDs to each operation type.
class CriticalRegionManager {
private:
  // Map from operation name to assigned barrier ID
  llvm::StringMap<unsigned> opNameToBarrierId;

  // Map from expensive operation name to the set of operation names that mark
  // the end of its critical region
  llvm::StringMap<SmallVector<std::string>> expensiveOpToEndOps;

  // Next barrier ID to assign (wraps around in range [5, 15])
  unsigned nextBarrierId = 5;

  // Barrier ID range constants
  static constexpr unsigned MIN_BARRIER_ID = 0;
  static constexpr unsigned MAX_BARRIER_ID = 15;

public:
  // Map from expensive operation name to the set of operations that mark
  // the critical region's start and end
  llvm::StringMap<SmallVector<Operation *>> expensiveOpToPingBoundaryOps;
  llvm::StringMap<SmallVector<Operation *>> expensiveOpToPongBoundaryOps;

  CriticalRegionManager() {
    // Register default expensive operations
    // IMPORTANT: Assume operations are expensive only for 2D tensors
    registerExpensiveOp("ttng.warp_group_dot"); // GEMM/Dot operation on Hopper
    registerExpensiveOp("math.exp");            // Exponential
    registerExpensiveOp("math.exp2");           // Exponential base 2

    // For exp2 operations, critical region might end at tmem_store
    registerEndOfCriticalOpType("math.exp2", "ttng.tmem_store");
  }

  // Register a new expensive operation TYPE and assign it a unique barrier ID.
  unsigned registerExpensiveOp(const std::string &opTypeName) {
    auto it = opNameToBarrierId.find(opTypeName);
    if (it != opNameToBarrierId.end()) {
      return it->second;
    }

    unsigned barrierId = nextBarrierId;
    opNameToBarrierId[opTypeName] = barrierId;

    LDBG("Registered expensive op type '" << opTypeName << "' with barrier ID "
                                          << barrierId);

    // Increment and wrap around
    nextBarrierId += 2;
    if (nextBarrierId > MAX_BARRIER_ID) {
      nextBarrierId = MIN_BARRIER_ID;
    }

    return barrierId;
  }

  // Register an operation TYPE that marks the end of a critical region
  void registerEndOfCriticalOpType(const std::string &expensiveOpTypeName,
                                   const std::string &endOpTypeName) {
    if (opNameToBarrierId.count(expensiveOpTypeName) == 0) {
      registerExpensiveOp(expensiveOpTypeName);
    }

    auto &endOps = expensiveOpToEndOps[expensiveOpTypeName];
    if (llvm::find(endOps, endOpTypeName) == endOps.end()) {
      endOps.push_back(endOpTypeName);
      LDBG("Registered '" << endOpTypeName << "' as end-of-critical-op for '"
                          << expensiveOpTypeName << "'");
    }
  }

  // Check if the given operation marks the end of a critical region
  // for ANY registered expensive operation. Returns the expensive op name if
  // found.
  std::optional<std::string> findExpensiveOpForEndOp(Operation *op) const {
    std::string opName = op->getName().getStringRef().str();

    for (const auto &entry : expensiveOpToEndOps) {
      if (llvm::find(entry.second, opName) != entry.second.end()) {
        return entry.first().str(); // Convert StringRef to std::string
      }
    }
    return std::nullopt;
  }

  // Check if an operation is registered as an expensive operation
  bool isExpensiveOp(Operation *op) const {
    // First check by name
    std::string opName = op->getName().getStringRef().str();
    return opNameToBarrierId.count(opName) > 0;
  }

  // Get the barrier ID assigned to an operation.
  // Returns std::nullopt if the operation is not registered.
  std::optional<unsigned> getBarrierId(std::string opName) const {
    // std::string opName = op->getName().getStringRef().str();
    auto it = opNameToBarrierId.find(opName);
    if (it != opNameToBarrierId.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  // Add an operation to the ping boundary list for a specific expensive
  // operation
  void addPingBoundaryOp(Operation *expensiveOp, Operation *boundaryOp) {
    std::string expensiveOpName = expensiveOp->getName().getStringRef().str();

    // Ensure the expensive op is registered first
    if (opNameToBarrierId.count(expensiveOpName) == 0) {
      registerExpensiveOp(expensiveOpName);
    }

    auto &boundaryOps = expensiveOpToPingBoundaryOps[expensiveOpName];
    boundaryOps.push_back(boundaryOp);
    LDBG("Added ping boundary op for '" << expensiveOpName
                                        << "' in Ping region.");
  }

  // Add an operation to the pong boundary list for a specific expensive
  // operation
  void addPongBoundaryOp(Operation *expensiveOp, Operation *boundaryOp) {
    std::string expensiveOpName = expensiveOp->getName().getStringRef().str();

    // Ensure the expensive op is registered first
    if (opNameToBarrierId.count(expensiveOpName) == 0) {
      registerExpensiveOp(expensiveOpName);
    }

    auto &boundaryOps = expensiveOpToPongBoundaryOps[expensiveOpName];
    boundaryOps.push_back(boundaryOp);
    LDBG("Added pong boundary op for '" << expensiveOpName
                                        << "' in Pong region.");
  }

  // Get the ping barrier ID for an expensive operation (barrierId)
  std::optional<unsigned> getPingBarrierId(std::string opName) const {
    auto barrierId = getBarrierId(opName);
    if (barrierId) {
      return *barrierId; // Ping uses the base barrier ID
    }
    return std::nullopt;
  }

  // Get the pong barrier ID for an expensive operation (barrierId + 1)
  std::optional<unsigned> getPongBarrierId(std::string opName) const {
    auto barrierId = getBarrierId(opName);
    if (barrierId) {
      return *barrierId + 1; // Pong uses barrierId + 1
    }
    return std::nullopt;
  }

  bool hasPingBoundarySetup(std::string expensiveOpName) const {
    // std::string expensiveOpName =
    // expensiveOp->getName().getStringRef().str();
    return (expensiveOpToPingBoundaryOps.count(expensiveOpName) > 0) and
           (expensiveOpToPingBoundaryOps.at(expensiveOpName).size() == 2);
  }

  bool hasPingPongBoundarySetup(std::string expensiveOpName) const {
    // std::string expensiveOpName =
    // expensiveOp->getName().getStringRef().str();
    return (expensiveOpToPingBoundaryOps.count(expensiveOpName) > 0) and
           (expensiveOpToPingBoundaryOps.at(expensiveOpName).size() == 2) and
           (expensiveOpToPongBoundaryOps.count(expensiveOpName) > 0) and
           (expensiveOpToPongBoundaryOps.at(expensiveOpName).size() == 2);
  }

  void dumpBoundaryOps() const {
    LDBG("===== Critical Region Manager Dump =====");
    LDBG("expensiveOpToPingBoundaryOps");
    for (const auto &entry : expensiveOpToPingBoundaryOps) {
      LDBG("expensiveOp: " << entry.first().str());
      for (const auto &op : entry.second) {
        LDBG("  ping boundary op:" << op->getName().getStringRef().str());
      }
    }
    LDBG("expensiveOpToPongBoundaryOps");
    for (const auto &entry : expensiveOpToPongBoundaryOps) {
      LDBG("expensiveOp: " << entry.first().str());
      for (const auto &op : entry.second) {
        LDBG("  pong boundary op:" << op->getName().getStringRef().str());
      }
    }
  }

  // Clear all registered operations and reset barrier ID counter
  void clear() {
    opNameToBarrierId.clear();
    expensiveOpToEndOps.clear();
    expensiveOpToPingBoundaryOps.clear();
    expensiveOpToPongBoundaryOps.clear();
    nextBarrierId = MIN_BARRIER_ID;
  }
};

// Returns the taskId if op has a single taskId, otherwise, returns -1.
static int getSingleTaskId(Operation *op) {
  auto asyncTasks = getAsyncTaskIds(op);
  if (asyncTasks.size() != 1)
    return -1;
  return asyncTasks[0];
}

static unsigned getLoopDepth(Operation *op) {
  unsigned depth = 0;
  auto pOp = op->getParentOfType<scf::ForOp>();
  while (pOp) {
    ++depth;
    pOp = pOp->getParentOfType<scf::ForOp>();
  }
  return depth;
}

void getNestedFor(Region *partition,
                  DenseMap<unsigned, SmallVector<Operation *>> &loopDepthMap) {
  partition->walk([&](Operation *subOp) {
    if (dyn_cast<scf::ForOp>(subOp)) {
      unsigned tDepth = getLoopDepth(subOp);
      loopDepthMap[tDepth].push_back(subOp);
    }
  });
}

static void handleWarpSpec(ttg::WarpSpecializeOp wsOp) {
  // Store loops and loop depths of each partition.
  SmallVector<DenseMap<unsigned, SmallVector<Operation *>>> partitionLoopDepths;
  unsigned partitionId = 0;
  SmallVector<Region *> computeRegions;

  // Collect all compute regions and their loop depths.
  for (Region *region : wsOp.getPartitionRegions()) {
    computeRegions.push_back(region);
    DenseMap<unsigned, SmallVector<Operation *>> loopDepths;
    getNestedFor(region, loopDepths);
    partitionLoopDepths.push_back(loopDepths);
    // Dump partitionLoopDepths
    LDBG("partition " << partitionId << " has " << loopDepths.size());
    for (auto &loopDepth : loopDepths) {
      LDBG("loop depth " << loopDepth.first << " has "
                         << loopDepth.second.size());
    }
    ++partitionId;
  }

  LDBG("Found " << partitionLoopDepths.size() << " compute regions");

  unsigned numPartitionWithLoops = 0;
  bool hasSingleOuterLoop = true;
  bool hasPersistent = false;
  for (auto &loopDepth : partitionLoopDepths) {
    // Check the partition has at lease a loop
    if (!loopDepth.empty()) {
      numPartitionWithLoops += 1;
    }
    // Check the partition a single outer loop
    if (loopDepth[0].size() != 1) {
      hasSingleOuterLoop = false;
    }
    // TODO: should better check before code parition if the kernel is
    // persistent. The kernel is persistent if it has a ForOp at depth of 1.
    if (loopDepth.find(1) != loopDepth.end()) {
      hasPersistent = true;
    }
  }
  if (numPartitionWithLoops < 2 || hasSingleOuterLoop == false)
    return;

  // Find the critical region boundaries
  // Initialize the critical region manager
  CriticalRegionManager crManager;
  // Process each partition to find expensive operations and their boundaries
  for (unsigned iter = 0; iter < computeRegions.size(); ++iter) {
    Region *region = computeRegions[iter];
    LDBG("Processing partition " << iter);

    // Pattern 1: Find ttng.warp_group_dot operations that are OUT of any loops
    // Walk the region at the top level (not inside loops)
    region->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<scf::ForOp>(op)) {
        return; // Skip if this is a ForOp - we only want ops outside loops
      }

      // Check if this is a warp_group_dot operation
      if (isa<ttng::WarpGroupDotOp>(op)) {
        LDBG("Found warp_group_dot outside loops in partition " << iter);
        if (crManager.hasPingPongBoundarySetup(
                op->getName().getStringRef().str())) {
          return; // Already setup, skip
        } else if (crManager.hasPingBoundarySetup(
                       op->getName().getStringRef().str())) {
          crManager.addPongBoundaryOp(
              op, op->getNextNode()); // Start and end are the same op
          crManager.addPongBoundaryOp(op, op->getNextNode());
        } else {
          crManager.addPingBoundaryOp(
              op, op->getNextNode()); // Start and end are the same op
          crManager.addPingBoundaryOp(op, op->getNextNode());
        }
      }
    });

    // Pattern 2: Find exp/exp2 operations in the INNERMOST loop
    if (partitionLoopDepths[iter].empty()) {
      LDBG("No loops in partition " << iter << ", skipping exp search");
      continue;
    }

    // Find the innermost loop (maximum depth)
    auto &loopDepth = partitionLoopDepths[iter];
    auto maxIt = std::max_element(
        loopDepth.begin(), loopDepth.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });
    if (maxIt->second.empty())
      continue;

    auto innermostForOp = dyn_cast<scf::ForOp>(maxIt->second[0]);
    if (!innermostForOp)
      continue;

    LDBG("Searching for exp ops in innermost loop at depth " << maxIt->first);
    // Walk through the innermost loop to find exp operations
    Operation *expOp = nullptr;
    Operation *expStartOp = nullptr; // exp critical region start
    Operation *expEndOp = nullptr;   // exp critical region end
    SmallVector<Operation *> expOps;
    for (auto &op : innermostForOp.getBody()->without_terminator()) {
      // Check for exp or exp2 operations
      if (isa<math::Exp2Op, math::ExpOp>(op)) {
        auto tensorTy = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
        if (tensorTy && tensorTy.getRank() > 1)
          expOps.push_back(&op);
      }
    }

    if (expOps.empty())
      continue;

    expOp = expOps.front();
    expStartOp = expOps.front();

    // Detect the next barrier arrive operation after tmem_store
    Operation *currentOp = expOps.back();
    bool foundTmemStore = false;
    // Walk forward from tmem_store to find the next barrier arrive
    while (currentOp) {
      currentOp = currentOp->getNextNode();
      // Stop at terminator
      if (!currentOp ||
          currentOp == innermostForOp.getBody()->getTerminator()) {
        break;
      }
      // Check the end of critical region, either a tma copy or tmem_store
      if (isa<ttng::AsyncTMACopyGlobalToLocalOp,
              ttng::AsyncTMACopyLocalToGlobalOp>(currentOp)) {
        expEndOp = currentOp->getNextNode();
        break;
      }
      if (isa<ttng::TMEMStoreOp>(currentOp)) {
        foundTmemStore = true;
        continue;
      }
      if (foundTmemStore && (isa<ttng::NamedBarrierArriveOp>(currentOp) ||
                             isa<ttng::ArriveBarrierOp>(currentOp))) {
        expEndOp = currentOp->getNextNode();
        break;
      }
    }

    if (!expEndOp) {
      expEndOp = expOps.back()->getNextNode();
    }

    if (crManager.hasPingPongBoundarySetup(
            expOp->getName().getStringRef().str())) {
      return; // Already setup, skip
    } else if (crManager.hasPingBoundarySetup(
                   expOp->getName().getStringRef().str())) {
      crManager.addPongBoundaryOp(expOp,
                                  expStartOp); // Start and end are the same op
      crManager.addPongBoundaryOp(expOp, expEndOp);
    } else {
      crManager.addPingBoundaryOp(expOp,
                                  expStartOp); // Start and end are the same op
      crManager.addPingBoundaryOp(expOp, expEndOp);
    }
  }

  auto &opToPingBoundary = crManager.expensiveOpToPingBoundaryOps;
  auto &opToPongBoundary = crManager.expensiveOpToPongBoundaryOps;
  crManager.dumpBoundaryOps();
  for (auto &pingEntry : opToPingBoundary) {
    StringRef keyOp = pingEntry.first();
    if (!crManager.hasPingPongBoundarySetup(keyOp.str()))
      continue;
    auto pingBarrierId = crManager.getPingBarrierId(keyOp.str());
    auto pongBarrierId = crManager.getPongBarrierId(keyOp.str());
    if (!pingBarrierId || !pongBarrierId)
      continue;
    // Insert barriers for the ping partition
    Operation *pingStart = pingEntry.second[0];
    Operation *pingEnd = pingEntry.second[1];

    // Get the partition region
    Region *partitionRegion = pingStart->getParentRegion();
    while (partitionRegion) {
      Operation *parentOp = partitionRegion->getParentOp();
      if (isa<ttg::WarpSpecializePartitionsOp>(parentOp)) {
        break;
      }
      partitionRegion = parentOp->getParentRegion();
    }

    if (!partitionRegion) {
      LDBG("No region found for ping partition.");
      continue;
    }

    Block &pingRegionBlock = partitionRegion->front();
    OpBuilder builder(&pingRegionBlock, pingRegionBlock.begin());
    auto pingRegionLoc = pingRegionBlock.front().getLoc();
    Value pingBarrier =
        builder.create<arith::ConstantIntOp>(pingRegionLoc, *pingBarrierId, 32);
    Value pongBarrier =
        builder.create<arith::ConstantIntOp>(pingRegionLoc, *pongBarrierId, 32);
    Value pingNumThreads =
        builder.create<arith::ConstantIntOp>(pingRegionLoc, 256, 32);
    builder.create<ttng::NamedBarrierArriveOp>(pingRegionLoc, pongBarrier,
                                               pingNumThreads);
    LDBG("pingBarrier: " << pingBarrier.getLoc() << " " << pingBarrier);
    LDBG("pongBarrier: " << pongBarrier.getLoc() << " " << pongBarrier);

    builder.setInsertionPoint(pingStart);
    builder.create<ttng::NamedBarrierWaitOp>(pingStart->getLoc(), pingBarrier,
                                             pingNumThreads);
    builder.setInsertionPoint(pingEnd);
    builder.create<ttng::NamedBarrierArriveOp>(pingEnd->getLoc(), pongBarrier,
                                               pingNumThreads);

    // Insert barriers for the pong partition
    Operation *pongStart = opToPongBoundary[keyOp][0];
    Operation *pongEnd = opToPongBoundary[keyOp][1];
    Region *pongRegion = pongStart->getParentRegion();
    Block &pongRegionBlock = pongRegion->front();
    OpBuilder builder2(&pongRegionBlock, pongRegionBlock.begin());
    auto pongRegionLoc = pongRegionBlock.front().getLoc();
    Value pingBarrier2 = builder2.create<arith::ConstantIntOp>(
        pongRegionLoc, *pingBarrierId, 32);
    Value pongBarrier2 = builder2.create<arith::ConstantIntOp>(
        pongRegionLoc, *pongBarrierId, 32);
    Value pingNumThreads2 =
        builder2.create<arith::ConstantIntOp>(pongRegionLoc, 256, 32);
    builder2.setInsertionPoint(pongStart);
    builder2.create<ttng::NamedBarrierWaitOp>(pongStart->getLoc(), pongBarrier2,
                                              pingNumThreads2);
    builder2.setInsertionPoint(pongEnd);
    builder2.create<ttng::NamedBarrierArriveOp>(pongEnd->getLoc(), pingBarrier2,
                                                pingNumThreads2);
  }
}

void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability) {
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &bodyOp : block.getOperations()) {
      Operation *op = &bodyOp;
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        handleWarpSpec(wsOp);
      }
    }
  }
}

Operation* lookUpPingPongOp(Operation *currentOp, const llvm::DenseSet<Operation *> &opSet) {
  for (Operation *refOp : opSet) {
    if (currentOp->getName() == refOp->getName())
      return refOp;
  }
  return nullptr;
}

bool areControlFlowEquivalent(Operation *op1, Operation *op2,
                              DominanceInfo &domInfo,
                              PostDominanceInfo &postDomInfo) {
    if (op1->getBlock() != op2->getBlock())
        return false;

    // Check if op1 dominates op2 AND op2 post-dominates op1
    if (domInfo.dominates(op1, op2) && postDomInfo.postDominates(op2, op1))
        return true;

    // Check the reverse (op2 dominates op1 AND op1 post-dominates op2)
    if (domInfo.dominates(op2, op1) && postDomInfo.postDominates(op1, op2))
        return true;

    return false;
}

void doPingPongPrep(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability) {
  CriticalRegionManager crManager;

  mlir::DominanceInfo domInfo(funcOp);
  mlir::PostDominanceInfo postDomInfo(funcOp);

  // pingpong attribute id
  unsigned pingpongRID = 0;

  // Set of expensive ops that have been not paired
  llvm::DenseSet<Operation*> expensiveOps;

  // Scan all operations in the function to find expensive ops
  funcOp.walk([&](Operation *op) {
    if (!crManager.isExpensiveOp(op))
      return;

    if (op->getNumOperands() == 0)
      return;

    // Only 2D tensor operations are considered expensive
    auto tensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (tensorTy && tensorTy.getRank() > 1) {
      Operation *lookUpOp = lookUpPingPongOp(op, expensiveOps);
      LDBG("Found expensive op: " << op->getName().getStringRef().str());
      if (lookUpOp == nullptr) {
        expensiveOps.insert(op);
        LDBG("No paired expensive op found, insert op " << op->getName().getStringRef().str() << " to expensiveOps set");
      } else {
        LDBG("Found paired expensive op, op name: " << op->getName().getStringRef().str() << ", lookUpOp name: " << lookUpOp->getName().getStringRef().str());
        if (areControlFlowEquivalent(lookUpOp, op, domInfo, postDomInfo)) {
          int thisOpTaskId = getSingleTaskId(op);
          int pairOpTaskId = getSingleTaskId(lookUpOp);
          assert(thisOpTaskId >= 0 && pairOpTaskId >= 0);

          // Make sure the two ops are in different partitions
          if (thisOpTaskId == pairOpTaskId)
            return;

          op->setAttr("pingpong_id",
                        IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                        pingpongRID));
          lookUpOp->setAttr("pingpong_id",
                        IntegerAttr::get(IntegerType::get(lookUpOp->getContext(), 32),
                                        pingpongRID));
          LDBG("Assign pingpong_id " << pingpongRID << " to op '"
                                        << lookUpOp->getName().getStringRef().str()
                                        << "' with task_id " << pairOpTaskId
                                        << " and " << thisOpTaskId);
          pingpongRID += 1;
          expensiveOps.erase(lookUpOp);
         }
      }
    }
  });
}

#define GEN_PASS_DEF_NVGPUTESTPINGPONGSYNC
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestPingPongSyncPass
    : public impl::NVGPUTestPingPongSyncBase<NVGPUTestPingPongSyncPass> {
public:
  using impl::NVGPUTestPingPongSyncBase<
      NVGPUTestPingPongSyncPass>::NVGPUTestPingPongSyncBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
      doPingPongSync(funcOp, numWarpGroups, capability);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
