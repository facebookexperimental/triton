#include "Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

enum class CriticalOpType : uint8_t {
  /// Operations that cannot be reordered by ptxas (warp_group_dot, etc.)
  NonReorderable = 0,
  /// Pure arithmetic operations that can be reordered (exp, exp2, etc.)
  // The ops should have memory ops as ping-pong boundary
  PureArithmetic = 1,
};

// Manages expensive operations for critical region identification and
// assigns unique barrier IDs to each operation type.
class CriticalRegionManager {
private:
  // Barrier ID range constants
  static constexpr unsigned MIN_BARRIER_ID = 0;
  static constexpr unsigned MAX_BARRIER_ID = 15;

public:
  // Current barrier ID to assign (wraps around in range [0, 15])
  int barrierId;

  // Map from critical operation name to its operation type
  llvm::StringMap<CriticalOpType> keyOpNameToType;

  // Map from pingpong region id to its barrier ID
  llvm::DenseMap<int, std::pair<int, int>> pingpongIdToBarrierId;

  // Map from pingpong region id to its critical operations
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToKeyOps;

  // Map from pingpong region id to operations that mark
  // the critical region's start and end
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToPingBoundaryOps;
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToPongBoundaryOps;

  CriticalRegionManager() {
    // Randomly initialize barrier ID to a value in range [0, 15]
    barrierId = MIN_BARRIER_ID + 5;

    // Register default expensive operations
    // IMPORTANT: Assume operations are expensive only for 2D tensors
    registerCriticalOp("ttng.warp_group_dot", CriticalOpType::NonReorderable); // GEMM/Dot operation on Hopper
    registerCriticalOp("math.exp", CriticalOpType::PureArithmetic);            // Exponential
    registerCriticalOp("math.exp2", CriticalOpType::PureArithmetic);           // Exponential base 2
  }

  // Register a new expensive operation TYPE and assign it a unique barrier ID.
  void registerCriticalOp(const std::string &opTypeName, CriticalOpType type) {
    // Ensure the operation type is not already registered
    if (keyOpNameToType.count(opTypeName) > 0) {
      LDBG("Operation type '" << opTypeName << "' already registered.");
      return;
    }

    // Register a new critical operation
    keyOpNameToType[opTypeName] = type;
    LDBG("Registered '" << opTypeName << "' as expensive operation.");
  }

  // Check if an operation is registered as an expensive operation
  bool isExpensiveOp(Operation *op) const {
    std::string opName = op->getName().getStringRef().str();
    return keyOpNameToType.count(opName) > 0;
  }

  // Get the barrier ID assigned to an operation.
  // Returns std::nullopt if the operation is not registered.
  void assignBarrierId(int pingpongId) {
    if (pingpongIdToBarrierId.count(pingpongId) > 0) {
      LDBG("Barrier ID already assigned for pingpong region '" << pingpongId
                                                               << "'.");
      return;
    }

    // Assign barrier ID to the pingpong region
    int barrierId = this->barrierId;
    int barrierId_1 = this->barrierId + 1;
    if (barrierId_1 > MAX_BARRIER_ID) {
      barrierId_1 = MIN_BARRIER_ID;
    }
    pingpongIdToBarrierId[pingpongId] = {barrierId, barrierId_1};
    LDBG("Assigned barrier ID {" << barrierId << ", " << barrierId_1 << "} to pingpong region '"
                                 << pingpongId << "'.");

    // Increment the barrier ID counter
    this->barrierId = (barrierId + 2 > MAX_BARRIER_ID) ? MIN_BARRIER_ID : barrierId + 2;
  }

  bool hasPingBoundary(int pingpongRegionId) const {
    return (pingpongIdToPingBoundaryOps.count(pingpongRegionId) > 0) and
           (pingpongIdToPingBoundaryOps.at(pingpongRegionId).size() == 2);
  }

  bool hasPongBoundary(int pingpongRegionId) const {
    return (pingpongIdToPongBoundaryOps.count(pingpongRegionId) > 0) and
           (pingpongIdToPongBoundaryOps.at(pingpongRegionId).size() == 2);
  }

  bool hasPingPongBoundarySetup(int pingpongRegionId) const {
    return (pingpongIdToPingBoundaryOps.count(pingpongRegionId) > 0) and
           (pingpongIdToPingBoundaryOps.at(pingpongRegionId).size() == 2) and
           (pingpongIdToPongBoundaryOps.count(pingpongRegionId) > 0) and
           (pingpongIdToPongBoundaryOps.at(pingpongRegionId).size() == 2);
  }

  void dumpBoundaryOps() const {
    LDBG("===== Critical Region Manager Dump =====");
    LDBG("pingpongIdToPingBoundaryOps");
    for (const auto &entry : pingpongIdToPingBoundaryOps) {
      LDBG("pingpongId: " << entry.first);
      for (const auto &op : entry.second) {
        LDBG("  ping boundary op:" << op->getName().getStringRef().str());
      }
    }
    LDBG("pingpongIdToPongBoundaryOps");
    for (const auto &entry : pingpongIdToPongBoundaryOps) {
      LDBG("pingpongId: " << entry.first);
      for (const auto &op : entry.second) {
        LDBG("  pong boundary op:" << op->getName().getStringRef().str());
      }
    }
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

// Operation* findEndOp(CriticalRegionManager &crManager, Operation *op, mlir::DominanceInfo &domInfo, mlir::PostDominanceInfo &postDomInfo) {
//   std::string opName = op->getName().getStringRef().str();
//   assert(crManager.expensiveOpToEndOps.count(opName) && "Critical region end ops not registered for op");
//   SmallVector<std::string> endOps = crManager.expensiveOpToEndOps[opName];
//   Operation* curOp = op;
//   while (curOp) {
//     std::string curOpName = curOp->getName().getStringRef().str();
//     if (llvm::find(endOps, curOpName) != endOps.end())
//       return curOp;
//     // Set end op to the end of the control flow equivalent region
//     if (!areControlFlowEquivalent(curOp, curOp->getNextNode(), domInfo, postDomInfo))
//       return curOp;
//     curOp = curOp->getNextNode();
//   }
//   return nullptr;
// }

// Operation* unionOfEndOps(SmallVector<Operation *> &endOps) {

//   Operation* curOp = endOps[0];
//   return curOp;
// }

// static void handleWarpSpec(ttg::WarpSpecializeOp wsOp) {
//   auto funcOp = wsOp->getParentOfType<triton::FuncOp>();
//   assert(funcOp != nullptr);
//   // Construct dominance info from the function
//   mlir::DominanceInfo domInfo(funcOp);
//   mlir::PostDominanceInfo postDomInfo(funcOp);

//   // Store loops and loop depths of each partition.
//   SmallVector<DenseMap<unsigned, SmallVector<Operation *>>> partitionLoopDepths;
//   unsigned partitionId = 0;
//   SmallVector<Region *> computeRegions;

//   // Collect all compute regions and their loop depths.
//   for (Region *region : wsOp.getPartitionRegions()) {
//     computeRegions.push_back(region);
//     DenseMap<unsigned, SmallVector<Operation *>> loopDepths;
//     getNestedFor(region, loopDepths);
//     partitionLoopDepths.push_back(loopDepths);
//     // Dump partitionLoopDepths
//     LDBG("partition " << partitionId << " has " << loopDepths.size());
//     for (auto &loopDepth : loopDepths) {
//       LDBG("loop depth " << loopDepth.first << " has "
//                          << loopDepth.second.size());
//     }
//     ++partitionId;
//   }

//   LDBG("Found " << partitionLoopDepths.size() << " compute regions");

//   unsigned numPartitionWithLoops = 0;
//   bool hasSingleOuterLoop = true;
//   for (auto &loopDepth : partitionLoopDepths) {
//     // Check the partition has at lease a loop
//     if (!loopDepth.empty()) {
//       numPartitionWithLoops += 1;
//     }
//     // Check that every partition should have a single outer loop, i.e. loop of depth 0
//     if (loopDepth[0].size() != 1) {
//       hasSingleOuterLoop = false;
//     }
//   }
//   if (numPartitionWithLoops < 2 || hasSingleOuterLoop == false)
//     return;

//   // Initialize the critical region manager
//   CriticalRegionManager crManager;
//   // Step 1: Process each partition to find expensive operations and their boundaries

//   for (unsigned iter = 0; iter < computeRegions.size(); ++iter) {
//     Region *region = computeRegions[iter];
//     LDBG("Processing partition " << iter);
//     // Map pingpong region id to the expensive ops in the partition
//     llvm::DenseMap<int, SmallVector<Operation *>> keyOps;
//     region->walk<WalkOrder::PreOrder>([&](Operation *op) {
//       // Check if this is a warp_group_dot operation
//       if (auto pingpongIdAttr = op->getAttrOfType<IntegerAttr>("pingpong_id")) {
//         LDBG("Found op with pingpong id " << pingpongIdAttr.getInt());
//         keyOps[pingpongIdAttr.getInt()].push_back(op);
//       }
//     });

//     if (keyOps.empty())
//       continue;

//     for (auto &keyEntry: keyOps) {
//       int pingpongRegionId = keyEntry.first;
//       // TODO: choose the end of region op to be closest to the terminator
//       SmallVector<Operation*> endOps;
//       for (auto &keyOp : keyEntry.second) {
//         Operation* endOp = findEndOp(crManager, keyOp, domInfo, postDomInfo);
//         endOps.push_back(endOp);
//       }
//       // Operation* endOp = findEndOp(crManager, keyOp, domInfo, postDomInfo);
//       bool hasPingBoundarySetup = crManager.hasPingBoundarySetup(pingpongRegionId);
//       bool hasPongBoundarySetup = crManager.hasPongBoundarySetup(pingpongRegionId);
//       LDBG("parition " << iter << " has ping boundary setup: " << hasPingBoundarySetup << " pong boundary setup: " << hasPongBoundarySetup);
//       Operation* keyOp = keyEntry.second[0];
//       Operation* commonEndOp = unionOfEndOps(endOps);
//       if (!hasPingBoundarySetup) {
//         crManager.addPingBoundaryOp(pingpongRegionId, keyOp); // push back key op
//         crManager.addPingBoundaryOp(pingpongRegionId, keyOp); // Start and end are the same op
//         crManager.addPingBoundaryOp(pingpongRegionId, commonEndOp);
//       }
//       if (hasPingBoundarySetup && !hasPongBoundarySetup) {
//         crManager.addPongBoundaryOp(pingpongRegionId, keyOp); // push back key op
//         crManager.addPongBoundaryOp(pingpongRegionId, keyOp); // Start and end are the same op
//         crManager.addPongBoundaryOp(pingpongRegionId, commonEndOp);
//       }
//     }
//   }

//   // Step 2: Insert pingpong barriers to the IR
//   auto &opToPingBoundary = crManager.expensiveOpToPingBoundaryOps;
//   auto &opToPongBoundary = crManager.expensiveOpToPongBoundaryOps;
//   crManager.dumpBoundaryOps();
//   for (auto &pingEntry : opToPingBoundary) {
//     int pingpongRegionId = pingEntry.first;
//     Operation* keyOp = pingEntry.second[0];
//     if (!crManager.hasPingPongBoundarySetup(pingpongRegionId))
//       continue;
//     auto pingBarrierId = crManager.getPingBarrierId(keyOp->getName().getStringRef().str());
//     auto pongBarrierId = crManager.getPongBarrierId(keyOp->getName().getStringRef().str());
//     if (!pingBarrierId || !pongBarrierId)
//       continue;
//     // Insert barriers for the ping partition
//     Operation *pingStart = pingEntry.second[1];
//     Operation *pingEnd = pingEntry.second[2];
//     Region *pingRegion = pingStart->getParentRegion();
//     Block &pingRegionBlock = pingRegion->front();
//     OpBuilder builder(&pingRegionBlock, pingRegionBlock.begin());
//     auto pingRegionLoc = pingRegionBlock.front().getLoc();
//     Value pingBarrier =
//         builder.create<arith::ConstantIntOp>(pingRegionLoc, *pingBarrierId, 32);
//     Value pongBarrier =
//         builder.create<arith::ConstantIntOp>(pingRegionLoc, *pongBarrierId, 32);
//     Value pingNumThreads =
//         builder.create<arith::ConstantIntOp>(pingRegionLoc, 256, 32);
//     builder.create<ttng::NamedBarrierArriveOp>(pingRegionLoc, pongBarrier,
//                                                pingNumThreads);
//     builder.setInsertionPoint(pingStart);
//     builder.create<ttng::NamedBarrierWaitOp>(pingStart->getLoc(), pingBarrier,
//                                              pingNumThreads);
//     // Insert AFTER the pingEnd op
//     builder.setInsertionPointAfter(pingEnd);
//     builder.create<ttng::NamedBarrierArriveOp>(pingEnd->getLoc(), pongBarrier,
//                                                pingNumThreads);

//     // Insert barriers for the pong partition
//     Operation *pongStart = opToPongBoundary[pingpongRegionId][1];
//     Operation *pongEnd = opToPongBoundary[pingpongRegionId][2];
//     Region *pongRegion = pongStart->getParentRegion();
//     Block &pongRegionBlock = pongRegion->front();
//     OpBuilder builder2(&pongRegionBlock, pongRegionBlock.begin());
//     auto pongRegionLoc = pongRegionBlock.front().getLoc();
//     Value pingBarrier2 = builder2.create<arith::ConstantIntOp>(
//         pongRegionLoc, *pingBarrierId, 32);
//     Value pongBarrier2 = builder2.create<arith::ConstantIntOp>(
//         pongRegionLoc, *pongBarrierId, 32);
//     Value pingNumThreads2 =
//         builder2.create<arith::ConstantIntOp>(pongRegionLoc, 256, 32);
//     builder2.setInsertionPoint(pongStart);
//     builder2.create<ttng::NamedBarrierWaitOp>(pongStart->getLoc(), pongBarrier2,
//                                               pingNumThreads2);
//     // Insert AFTER the pongEnd op
//     builder2.setInsertionPointAfter(pongEnd);
//     builder2.create<ttng::NamedBarrierArriveOp>(pongEnd->getLoc(), pingBarrier2,
//                                                 pingNumThreads2);
//   }
// }

void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability) {
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &bodyOp : block.getOperations()) {
      Operation *op = &bodyOp;
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        // handleWarpSpec(wsOp);
      }
    }
  }
}


static Operation *getSplitOp(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (auto result = dyn_cast<OpResult>(operand)) {
      if (isa<tt::SplitOp>(result.getOwner())) {
        return result.getOwner();
      }
    }
  }
  return nullptr;
}

static std::optional<NameLoc> getNameLoc(Operation *op) {
  Location loc = op->getLoc();
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    if (auto nameLoc = dyn_cast<NameLoc>(callSiteLoc.getCallee())) {
      return nameLoc;
    }
  }
  return std::nullopt;
}

void doPingPongPrep(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability) {
  CriticalRegionManager crManager;

  // Initialize the dominance and post-dominance info
  mlir::DominanceInfo domInfo(funcOp);
  mlir::PostDominanceInfo postDomInfo(funcOp);

  // A list of expensive ops grouped in vectors.
  // Each vector contains ops that belong to the same pingpong region.
  llvm::SmallVector<llvm::SmallVector<Operation *, 4>> expensiveOps;

  // Scan all operations in the function to find expensive ops
  funcOp.walk([&](Operation *op) {
    if (!crManager.isExpensiveOp(op))
      return;

    if (op->getNumOperands() == 0)
      return;

    // Only 2D tensor operations are considered expensive
    auto tensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (tensorTy && tensorTy.getRank() > 1) {
      LDBG("Found expensive op " << op->getName().getStringRef().str()
                                 << ", location" << op->getLoc());

      bool foundGroup = false;
      for (auto &group : expensiveOps) {
        bool matchType = true;
        bool matchVar = false;
        for (auto &refOp : group) {
          // Check 1: Same Operation Name
          if (op->getName() != refOp->getName()) {
            matchType = false;
            break;
          }
          // Check 2: Control Flow Equivalent
          if (!areControlFlowEquivalent(op, refOp, domInfo, postDomInfo)) {
            matchType = false;
            break;
          }

          // Check 3: If in the same partition, the op is dependent on the same
          // tt.split
          int opTaskId = getSingleTaskId(op);
          int refTaskId = getSingleTaskId(refOp);
          if (opTaskId == refTaskId) {
            Operation *opSplit = getSplitOp(op);
            Operation *refSplit = getSplitOp(refOp);
            // Are from the same data split op
            if (opSplit == refSplit) {
              LDBG("Same split op");
              matchVar = true;
            }
          } else { // Check 4: If in different partitions, the op is called from
                   // the same source var
            std::optional<NameLoc> opLoc = getNameLoc(op);
            std::optional<NameLoc> refLoc = getNameLoc(refOp);
            if (opLoc.has_value() && refLoc.has_value()) {
              LDBG("op loc: " << opLoc << ", refLoc" << refLoc);
              if (opLoc.value() == refLoc.value()) {
                LDBG("Same source var");
                matchVar = true;
              }
            }
          }
        }
        foundGroup = matchType && matchVar;
        if (foundGroup) {
          LDBG("Insert to ref op group "
               << group[0]->getName().getStringRef().str());
          group.push_back(op);
          break;
        }
      }

      if (!foundGroup) {
        LDBG("Create new group for op " << op->getName().getStringRef().str());
        expensiveOps.push_back({op});
      }
    }
  });

  // pingpong region id
  unsigned pingpongID = 0;

  // Assign pingpong IDs to groups
  for (auto &group : expensiveOps) {
    if (group.size() < 2)
      continue;

    // Check if ops are in different partitions
    bool differentPartitions = false;
    int firstTaskId = getSingleTaskId(group[0]);
    for (size_t i = 1; i < group.size(); ++i) {
      if (getSingleTaskId(group[i]) != firstTaskId) {
        differentPartitions = true;
        break;
      }
    }

    if (!differentPartitions)
      continue;

    for (auto *op : group) {
      op->setAttr("pingpong_id",
                  IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                   pingpongID));
      LDBG("Assign pingpong_id " << pingpongID << " to op '"
                                 << op->getName().getStringRef().str()
                                 << "' with task_id " << getSingleTaskId(op));
    }
    pingpongID++;
  }
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
