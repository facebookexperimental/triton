//===----------------------------------------------------------------------===//
// PingPong Barrier Insertion Pass
//
// Enforce pingpong around expensive ops (warp_group_dot, math.exp)
// across warp partitions by inserting named barriers.
//
// Two passes:
//   1. doPingPongPrep: Preprocess to group expensive ops that
//      i) of the same type,
//      ii) in the same control flow, and
//      iii) operate on the same or subtiled variables
//      into pingpong regions and assign a unique pingpong_id.
//
//   2. doPingPongSync: For each pingpong region, identify start and end
//      boundaries, and insert arrive/wait named barriers to the IR.
//
// Barrier pattern:
//   Ping: arrive(pong) at entry, wait(ping) before op, arrive(pong) after op
//   Pong: wait(pong) before op, arrive(ping) after op
//
// Critical op types:
//   - NonReorderable (warp_group_dot): has memory effects, boundary is the op
//   - PureArithmetic (math.exp): boundary extends to next memory op
//===----------------------------------------------------------------------===//

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
  static constexpr unsigned MIN_BARRIER_ID = 9;
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

  // Map from pingpong region id to the participating thread number
  llvm::DenseMap<int, int> pingpongIdToThreadNum;

  CriticalRegionManager() {
    // Initialize barrier ID to be 9
    barrierId = MIN_BARRIER_ID;

    // Register default expensive operations
    // IMPORTANT: Assume operations are expensive only for 2D tensors
    registerCriticalOp(
        "ttng.warp_group_dot",
        CriticalOpType::NonReorderable); // GEMM/Dot operation on Hopper
    registerCriticalOp("math.exp",
                       CriticalOpType::PureArithmetic); // Exponential
    registerCriticalOp("math.exp2",
                       CriticalOpType::PureArithmetic); // Exponential base 2
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
      LDBG("Barrier ID {" << pingpongIdToBarrierId[pingpongId].first << ", "
                          << pingpongIdToBarrierId[pingpongId].second
                          << "} already assigned for pingpong region '"
                          << pingpongId << "'.");
      return;
    }

    // Assign barrier ID to the pingpong region
    int barrierId = this->barrierId;
    int barrierId_1 = this->barrierId + 1;
    if (barrierId_1 > MAX_BARRIER_ID) {
      barrierId_1 = MIN_BARRIER_ID;
    }
    pingpongIdToBarrierId[pingpongId] = {barrierId, barrierId_1};
    LDBG("Assigned barrier ID {" << barrierId << ", " << barrierId_1
                                 << "} to pingpong region '" << pingpongId
                                 << "'.");

    // Increment the barrier ID counter
    this->barrierId =
        (barrierId + 2 > MAX_BARRIER_ID) ? MIN_BARRIER_ID : barrierId + 2;
  }

  bool hasPingBoundary(int pingpongRegionId) const {
    return (pingpongIdToPingBoundaryOps.count(pingpongRegionId) > 0) and
           (pingpongIdToPingBoundaryOps.at(pingpongRegionId).size() == 2);
  }

  bool hasPongBoundary(int pingpongRegionId) const {
    return (pingpongIdToPongBoundaryOps.count(pingpongRegionId) > 0) and
           (pingpongIdToPongBoundaryOps.at(pingpongRegionId).size() == 2);
  }

  bool hasPingPongBoundary(int pingpongRegionId) const {
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
        LDBG("  ping boundary op: " << op->getName().getStringRef().str());
      }
    }
    LDBG("pingpongIdToPongBoundaryOps");
    for (const auto &entry : pingpongIdToPongBoundaryOps) {
      LDBG("pingpongId: " << entry.first);
      for (const auto &op : entry.second) {
        LDBG("  pong boundary op: " << op->getName().getStringRef().str());
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

// Return a map of loop depth to the loop ops in the partition.
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

Operation *findStartOp(CriticalRegionManager &crManager, Operation *keyOp,
                       mlir::DominanceInfo &domInfo,
                       mlir::PostDominanceInfo &postDomInfo) {
  // Set the start op of this pingpong region to be the critical op
  return keyOp;
}

/// Dump memory effects of an operation for debugging
void dumpMemoryEffects(Operation *op) {
  auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memInterface) {
    LDBG("  Op '" << op->getName().getStringRef().str()
                  << "' does not implement MemoryEffectOpInterface");
    return;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memInterface.getEffects(effects);

  if (effects.empty()) {
    LDBG("  Op '" << op->getName().getStringRef().str()
                  << "' has no memory effects");
    return;
  }

  for (const auto &effect : effects) {
    std::string effectType;
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      effectType = "Read";
    else if (isa<MemoryEffects::Write>(effect.getEffect()))
      effectType = "Write";
    else if (isa<MemoryEffects::Allocate>(effect.getEffect()))
      effectType = "Allocate";
    else if (isa<MemoryEffects::Free>(effect.getEffect()))
      effectType = "Free";
    else
      effectType = "Unknown";

    std::string resourceName = effect.getResource()->getName().str();
    LDBG("  Op '" << op->getName().getStringRef().str() << "' has effect: "
                  << effectType << " on resource: " << resourceName);
  }
}

Operation *findEndOp(CriticalRegionManager &crManager, Operation *keyOp,
                     mlir::DominanceInfo &domInfo,
                     mlir::PostDominanceInfo &postDomInfo) {
  // Set the end op of this pingpong region to be the first op with memory side
  // effect after this critical op
  std::string opName = keyOp->getName().getStringRef().str();
  if (crManager.keyOpNameToType.count(opName) == 0) {
    LDBG("Operation " << opName
                      << " is not registered as an expensive operation.");
    return nullptr;
  }
  CriticalOpType opType = crManager.keyOpNameToType[opName];
  if (opType == CriticalOpType::NonReorderable) {
    LDBG("Operation " << opName << " is not reorderable.");
    Operation *curOp = keyOp;
    while (curOp) {
      if (!isMemoryEffectFree(curOp)) {
        LDBG("Found op with memory effects:");
        dumpMemoryEffects(curOp);
        return curOp;
      }
      // Set end op to the end of the control flow equivalent region
      if (!areControlFlowEquivalent(curOp, curOp->getNextNode(), domInfo,
                                    postDomInfo))
        return curOp;
      curOp = curOp->getNextNode();
    }
  } else if (opType == CriticalOpType::PureArithmetic) {
    LDBG("Operation " << opName << " is pure arithmetic.");
    Operation *curOp = keyOp;
    while (curOp) {
      if (!isMemoryEffectFree(curOp)) {
        LDBG("Found op with memory effects:");
        dumpMemoryEffects(curOp);
        return curOp;
      }
      // Set end op to the end of the control flow equivalent region
      if (!areControlFlowEquivalent(curOp, curOp->getNextNode(), domInfo,
                                    postDomInfo))
        return curOp;
      curOp = curOp->getNextNode();
    }
  }
  return nullptr;
}

/// Returns the operation from startOps that is closest to the entry
/// (executed earliest). All ops must be in the same block.
Operation *unionOfStartOps(SmallVector<Operation *> &startOps) {
  if (startOps.empty())
    return nullptr;

  Operation *earliestOp = startOps[0];
  Block *block = earliestOp->getBlock();

  for (size_t i = 1; i < startOps.size(); ++i) {
    Operation *op = startOps[i];
    // Verify all ops are in the same block
    if (op->getBlock() != block) {
      LDBG("Warning: unionOfStartOps called with ops in different blocks");
      return nullptr;
    }
    // If op is before earliestOp, then op is closer to entry
    if (op->isBeforeInBlock(earliestOp)) {
      earliestOp = op;
    }
  }
  return earliestOp;
}

/// Returns the operation from endOps that is closest to the terminator
/// (executed latest). All ops must be in the same block.
Operation *unionOfEndOps(SmallVector<Operation *> &endOps) {
  if (endOps.empty())
    return nullptr;

  Operation *latestOp = endOps[0];
  Block *block = latestOp->getBlock();

  for (size_t i = 1; i < endOps.size(); ++i) {
    Operation *op = endOps[i];
    // Verify all ops are in the same block
    if (op->getBlock() != block) {
      LDBG("Warning: unionOfEndOps called with ops in different blocks");
      return nullptr;
    }
    // If latestOp is before op, then op is closer to terminator
    if (latestOp->isBeforeInBlock(op)) {
      latestOp = op;
    }
  }
  return latestOp;
}

static void handleWarpSpec(ttg::WarpSpecializeOp wsOp) {
  // Get the function op
  auto funcOp = wsOp->getParentOfType<triton::FuncOp>();
  assert(funcOp != nullptr);

  // Construct dominance info from the function
  mlir::DominanceInfo domInfo(funcOp);
  mlir::PostDominanceInfo postDomInfo(funcOp);

  // Store loops and loop depths of each partition.
  SmallVector<DenseMap<unsigned, SmallVector<Operation *>>> partitionLoopDepths;
  SmallVector<Region *> computeRegions;

  // Collect all compute regions and their loop depths.
  for (Region *region : wsOp.getPartitionRegions()) {
    computeRegions.push_back(region);
    DenseMap<unsigned, SmallVector<Operation *>> loopDepths;
    getNestedFor(region, loopDepths);
    partitionLoopDepths.push_back(loopDepths);
    // Dump partitionLoopDepths
    for (auto &loopDepth : loopDepths) {
      LDBG("loop depth " << loopDepth.first << " has "
                         << loopDepth.second.size());
    }
  }

  LDBG("Found " << partitionLoopDepths.size() << " compute regions");

  // Check if at least two partitions have loops and
  // each partition has a single outer loop
  unsigned numPartitionWithLoops = 0;
  bool hasSingleOuterLoop = true;
  for (auto &loopDepth : partitionLoopDepths) {
    // Check the partition has at lease a loop
    if (!loopDepth.empty()) {
      numPartitionWithLoops += 1;
    }
    // Check that every partition should have a single outer loop, i.e. loop of
    // depth 0
    if (loopDepth[0].size() != 1) {
      hasSingleOuterLoop = false;
    }
  }
  if (numPartitionWithLoops < 2 || hasSingleOuterLoop == false)
    return;

  // Initialize the critical region manager
  CriticalRegionManager crManager;

  // Step 1: Process each partition to find expensive operations and their
  // boundaries
  for (unsigned iter = 0; iter < computeRegions.size(); ++iter) {
    Region *region = computeRegions[iter];
    // Walk through the region to find operations that have pingpong_id
    // attribute
    region->walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Check if this is a warp_group_dot operation
      if (auto pingpongIdAttr = op->getAttrOfType<IntegerAttr>("pingpong_id")) {
        int pingpongId = pingpongIdAttr.getInt();
        LDBG("Found op " << op->getName().getStringRef().str()
                         << " with pingpong id " << pingpongId);
        // Prepare CriticalRegionManager for this pingpong region
        crManager.pingpongIdToKeyOps[pingpongId].push_back(op);
        crManager.assignBarrierId(pingpongId);
      }
    });
  }

  // Step 2: For each pingpong region,
  //         i) find the boundaries and
  //         ii) calculate the participating thread number
  for (auto &[pingpongId, keyOps] : crManager.pingpongIdToKeyOps) {
    // Map from the ping and pong partition id to the start and end ops
    llvm::DenseMap<int, SmallVector<Operation *>> startOps;
    llvm::DenseMap<int, SmallVector<Operation *>> endOps;

    // Map from the ping and pong partition id to its number of warps
    llvm::DenseMap<int, int> numWarps;

    // Find the start and end ops for each key operation in the pingpong region
    for (auto &keyOp : keyOps) {
      int partitionId = getSingleTaskId(keyOp);
      Operation *startOp = findStartOp(crManager, keyOp, domInfo, postDomInfo);
      Operation *endOp = findEndOp(crManager, keyOp, domInfo, postDomInfo);
      startOps[partitionId].push_back(startOp);
      endOps[partitionId].push_back(endOp);
      // Look up the number of warps for each partition
      if (numWarps.count(partitionId) == 0) {
        numWarps[partitionId] = mlir::triton::gpu::lookupNumWarps(keyOp);
        LDBG("numWarps of " << partitionId << " is " << numWarps[partitionId]);
      }
    }

    if (startOps.size() != 2 || endOps.size() != 2 || numWarps.size() != 2) {
      LDBG("pingpong ops are not in two partitions");
      continue;
    }

    int numberOfThreads = 0;
    for (auto [partitionId, startOp] : startOps) {
      // The start and end ops are unioned for each partition to find the
      // boundary ops
      Operation *unionStartOp = unionOfStartOps(startOp);
      Operation *unionEndOp = unionOfEndOps(endOps[partitionId]);
      if (!crManager.hasPingBoundary(pingpongId)) {
        crManager.pingpongIdToPingBoundaryOps[pingpongId].push_back(
            unionStartOp);
        crManager.pingpongIdToPingBoundaryOps[pingpongId].push_back(unionEndOp);
      } else {
        crManager.pingpongIdToPongBoundaryOps[pingpongId].push_back(
            unionStartOp);
        crManager.pingpongIdToPongBoundaryOps[pingpongId].push_back(unionEndOp);
      }
      // The number of participating threads is summed up from ping and pong
      // partitions
      numberOfThreads += numWarps[partitionId] * 32; // 32 threads per warp
      LDBG("numberOfThreads " << numberOfThreads);
    }

    if (crManager.pingpongIdToThreadNum.count(pingpongId) == 0) {
      crManager.pingpongIdToThreadNum[pingpongId] = numberOfThreads;
      LDBG("pingpongId " << pingpongId << " has " << numberOfThreads);
    }

    crManager.dumpBoundaryOps();
  }

  // Step 3: Insert pingpong barriers to the IR
  for (auto &[pingpongId, keyOps] : crManager.pingpongIdToKeyOps) {
    if (!crManager.hasPingPongBoundary(pingpongId))
      continue;
    SmallVector<Operation *> pingBoundOps =
        crManager.pingpongIdToPingBoundaryOps[pingpongId];
    SmallVector<Operation *> pongBoundOps =
        crManager.pingpongIdToPongBoundaryOps[pingpongId];

    int pingBarrierId = crManager.pingpongIdToBarrierId[pingpongId].first;
    int pongBarrierId = crManager.pingpongIdToBarrierId[pingpongId].second;

    int numThreads = crManager.pingpongIdToThreadNum[pingpongId];

    // Insert barriers for the ping partition
    Operation *pingStart = pingBoundOps[0];
    Operation *pingEnd = pingBoundOps[1];
    Region *pingRegion = pingStart->getParentRegion();
    // walk up to the partition region of the warp_spec op
    while (pingRegion) {
      Operation *parentOp = pingRegion->getParentOp();
      if (isa<ttg::WarpSpecializePartitionsOp>(parentOp)) {
        break;
      }
      pingRegion = parentOp->getParentRegion();
    }
    if (!pingRegion) {
      LDBG("No region found for ping partition.");
      continue;
    }
    Block &pingRegionBlock = pingRegion->front();
    OpBuilder builder(&pingRegionBlock, pingRegionBlock.begin());
    auto pingRegionLoc = pingRegionBlock.front().getLoc();
    // Prepare values
    Value pingBarrier =
        builder.create<arith::ConstantIntOp>(pingRegionLoc, pingBarrierId, 32);
    Value pongBarrier =
        builder.create<arith::ConstantIntOp>(pingRegionLoc, pongBarrierId, 32);
    Value pingNumThreads =
        builder.create<arith::ConstantIntOp>(pingRegionLoc, numThreads, 32);
    // Insert arrive barrier for the ping partition to allow the initial entry
    builder.create<ttng::NamedBarrierArriveOp>(pingRegionLoc, pongBarrier,
                                               pingNumThreads);
    builder.setInsertionPoint(pingStart);
    builder.create<ttng::NamedBarrierWaitOp>(pingStart->getLoc(), pingBarrier,
                                             pingNumThreads);
    // Insert AFTER the pingEnd op
    builder.setInsertionPointAfter(pingEnd);
    builder.create<ttng::NamedBarrierArriveOp>(pingEnd->getLoc(), pongBarrier,
                                               pingNumThreads);

    // Insert barriers for the pong partition
    Operation *pongStart = pongBoundOps[0];
    Operation *pongEnd = pongBoundOps[1];
    Region *pongRegion = pongStart->getParentRegion();
    Block &pongRegionBlock = pongRegion->front();
    OpBuilder builder2(&pongRegionBlock, pongRegionBlock.begin());
    auto pongRegionLoc = pongRegionBlock.front().getLoc();
    Value pingBarrier2 =
        builder2.create<arith::ConstantIntOp>(pongRegionLoc, pingBarrierId, 32);
    Value pongBarrier2 =
        builder2.create<arith::ConstantIntOp>(pongRegionLoc, pongBarrierId, 32);
    Value pingNumThreads2 =
        builder2.create<arith::ConstantIntOp>(pongRegionLoc, numThreads, 32);
    builder2.setInsertionPoint(pongStart);
    builder2.create<ttng::NamedBarrierWaitOp>(pongStart->getLoc(), pongBarrier2,
                                              pingNumThreads2);
    // Insert AFTER the pongEnd op
    builder2.setInsertionPointAfter(pongEnd);
    builder2.create<ttng::NamedBarrierArriveOp>(pongEnd->getLoc(), pingBarrier2,
                                                pingNumThreads2);
  }
}

/// doPingPongPrep pass: Insert pingpong barriers to the IR
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

/// doPingPongPrep pass: Group expensive ops into pingpong regions
void doPingPongPrep(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability) {
  CriticalRegionManager crManager;

  // Initialize the dominance and post-dominance info
  mlir::DominanceInfo domInfo(funcOp);
  mlir::PostDominanceInfo postDomInfo(funcOp);

  // A list of expensive op groups.
  // Each group contains ops at the same pingpong region.
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

      // Check if the expensive op belongs to an existing group
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

  // pingpong region ID
  unsigned pingpongID = 0;

  // Assign pingpong region IDs to groups
  for (auto &group : expensiveOps) {
    if (group.size() < 2)
      continue;

    // Check if ops are from different partitions
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
      op->setAttr(
          "pingpong_id",
          IntegerAttr::get(IntegerType::get(op->getContext(), 32), pingpongID));
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
