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
  static constexpr unsigned MIN_BARRIER_ID = 7;
  static constexpr unsigned MAX_BARRIER_ID = 15;

public:
  // Current barrier ID to assign (wraps around in range [0, 15])
  int barrierId;

  // Map from compute capability to a map of critical operation name to its
  // operation type
  llvm::DenseMap<int, llvm::StringMap<CriticalOpType>> keyOpNameToType;

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
    // Initialize barrier ID to be 7
    barrierId = MIN_BARRIER_ID;

    /// Register default expensive operations
    // Hopper (compute capability 9)
    registerCriticalOp(
        9, "ttng.warp_group_dot",
        CriticalOpType::NonReorderable); // GEMM/Dot operation on Hopper
    // Blackwell (compute capability 10)
    registerCriticalOp(10, "math.exp",
                       CriticalOpType::PureArithmetic); // Exponential
    registerCriticalOp(10, "math.exp2",
                       CriticalOpType::PureArithmetic); // Exponential base 2
  }

  // Register a new expensive operation TYPE and assign it a unique barrier ID.
  void registerCriticalOp(int computeCapability, const std::string &opTypeName,
                          CriticalOpType type) {
    // Initialize the inner map for this compute capability if it doesn't exist
    if (keyOpNameToType.count(computeCapability) == 0) {
      keyOpNameToType[computeCapability] = llvm::StringMap<CriticalOpType>();
    }

    // Ensure the operation type is not already registered for this compute
    // capability
    if (keyOpNameToType[computeCapability].count(opTypeName) > 0) {
      LDBG("Operation type '" << opTypeName
                              << "' already registered for compute capability "
                              << computeCapability << ".");
      return;
    }

    // Register a new critical operation for this compute capability
    keyOpNameToType[computeCapability][opTypeName] = type;
    LDBG("Registered '" << opTypeName
                        << "' as expensive operation for compute capability "
                        << computeCapability << ".");
  }

  // Check if an operation is registered as an expensive operation for the given
  // compute capability
  bool isExpensiveOp(Operation *op, int computeCapability) const {
    std::string opName = op->getName().getStringRef().str();
    auto it = keyOpNameToType.find(computeCapability);
    if (it == keyOpNameToType.end()) {
      return false;
    }
    return it->second.count(opName) > 0;
  }

  // Get the CriticalOpType for an operation (for the given compute capability)
  std::optional<CriticalOpType> getCriticalOpType(Operation *op,
                                                  int computeCapability) const {
    std::string opName = op->getName().getStringRef().str();
    auto it = keyOpNameToType.find(computeCapability);
    if (it == keyOpNameToType.end()) {
      return std::nullopt;
    }
    auto opIt = it->second.find(opName);
    if (opIt == it->second.end()) {
      return std::nullopt;
    }
    return opIt->second;
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
  if (asyncTasks.size() > 1)
    return -1;
  if (asyncTasks.empty()) {
    // No async_task_id or ttg.partition
    // Fall back to find warp_spec op

    Region *curRegion = op->getParentRegion();
    Region *partitionRegion = nullptr;
    ttg::WarpSpecializeOp wsOp = nullptr;
    // walk up to the partition region of the warp_spec op
    while (curRegion) {
      Operation *parentOp = curRegion->getParentOp();
      if (isa<ttg::WarpSpecializePartitionsOp>(parentOp)) {
        partitionRegion = curRegion;
      } else if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(parentOp)) {
        wsOp = ws;
        break;
      }
      curRegion = parentOp->getParentRegion();
    }
    if (!partitionRegion || !wsOp) {
      LDBG("No partition region or warp_spec op found.");
      return -1;
    }
    if (partitionRegion == &wsOp.getDefaultRegion()) {
      return 0;
    }
    auto partitionRegions = wsOp.getPartitionRegions();
    for (auto [idx, region] : llvm::enumerate(partitionRegions)) {
      if (partitionRegion == region) {
        return idx + 1; // partition 1, 2, ... for partition regions
      }
    }
    return -1; // Should not reach here
  }
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

/// Check if there's a loop operation between two operations in the same block
static bool hasInterveningLoop(Operation *earlier, Operation *later) {
  // Walk from earlier to later, checking for loop operations
  Operation *current = earlier->getNextNode();
  while (current && current != later) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(current)) {
      return true;
    }
    current = current->getNextNode();
  }
  return false;
}

bool areControlFlowEquivalent(Operation *op1, Operation *op2,
                              DominanceInfo &domInfo,
                              PostDominanceInfo &postDomInfo) {
  if (op1->getBlock() != op2->getBlock())
    return false;

  // Check if op1 dominates op2 AND op2 post-dominates op1
  if (domInfo.dominates(op1, op2) && postDomInfo.postDominates(op2, op1)) {
    // Additional check: ensure no loop separates them
    if (hasInterveningLoop(op1, op2)) {
      LDBG("Ops separated by a loop, not control flow equivalent");
      return false;
    }
    return true;
  }

  // Check the reverse (op2 dominates op1 AND op1 post-dominates op2)
  if (domInfo.dominates(op2, op1) && postDomInfo.postDominates(op1, op2)) {
    // Additional check: ensure no loop separates them
    if (hasInterveningLoop(op2, op1)) {
      LDBG("Ops separated by a loop, not control flow equivalent");
      return false;
    }
    return true;
  }

  return false;
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

Operation *findStartOp(CriticalRegionManager &crManager, Operation *keyOp,
                       mlir::DominanceInfo &domInfo,
                       mlir::PostDominanceInfo &postDomInfo) {
  // Set the start op of this pingpong region to be the critical op
  return keyOp;
}

/// Find the end boundary op for the critical region.
/// Rules are specific to the CriticalOpType.
/// For now, we apply the same rule for NonReorderable and PureArithmetic --
/// the end op is the first op with memory side effect after the critical op.
Operation *findEndOp(CriticalRegionManager &crManager, Operation *keyOp,
                     int computeCapability, mlir::DominanceInfo &domInfo,
                     mlir::PostDominanceInfo &postDomInfo) {
  // Set the end op of this pingpong region to be the first op with memory side
  // effect after this critical op
  std::string opName = keyOp->getName().getStringRef().str();
  auto opTypeOpt = crManager.getCriticalOpType(keyOp, computeCapability);
  if (!opTypeOpt.has_value()) {
    LDBG("Operation " << opName
                      << " is not registered as an expensive operation for "
                         "compute capability "
                      << computeCapability << ".");
    return nullptr;
  }
  CriticalOpType opType = opTypeOpt.value();
  if (opType == CriticalOpType::NonReorderable) {
    LDBG("Operation " << opName << " is not reorderable.");
    // Find the first op with memory side effect after this critical op
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
    // Find the first op with memory side effect after this critical op
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

static void handleWarpSpec(ttg::WarpSpecializeOp wsOp, int computeCapability) {
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
      Operation *endOp =
          findEndOp(crManager, keyOp, computeCapability, domInfo, postDomInfo);
      startOps[partitionId].push_back(startOp);
      endOps[partitionId].push_back(endOp);
      // Look up the number of warps for each partition
      if (numWarps.count(partitionId) == 0) {
        numWarps[partitionId] = ttg::lookupNumWarps(keyOp);
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

static int getCapability(ModuleOp moduleOp, int defaultCapability) {
  LDBG("defaultCapability: " << defaultCapability);
  if (auto targetAttr =
          moduleOp->getAttrOfType<StringAttr>(ttg::AttrTargetName)) {
    StringRef ref = targetAttr.strref();
    if (ref.starts_with("cuda:")) {
      StringRef capabilityStr = ref.drop_front(5); // drop the "cuda:"
      int parsedCapability;
      if (!capabilityStr.getAsInteger(10, parsedCapability)) {
        LDBG("Using compute capability from module: " << parsedCapability);
        return parsedCapability / 10;
      }
    }
  }
  return defaultCapability;
}

/// doPingPongSync pass: Insert pingpong barriers to the IR
void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  capability = getCapability(moduleOp, capability);
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &bodyOp : block.getOperations()) {
      Operation *op = &bodyOp;
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        handleWarpSpec(wsOp, capability);
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
  // Check if loc is directly a NameLoc
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    return nameLoc;
  }
  // Check if loc is a CallSiteLoc with a NameLoc callee
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    if (auto nameLoc = dyn_cast<NameLoc>(callSiteLoc.getCallee())) {
      return nameLoc;
    }
  }
  // Check if loc is a FusedLoc and extract NameLoc from it
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (Location subLoc : fusedLoc.getLocations()) {
      if (auto nameLoc = dyn_cast<NameLoc>(subLoc)) {
        return nameLoc;
      }
      if (auto callSiteLoc = dyn_cast<CallSiteLoc>(subLoc)) {
        if (auto nameLoc = dyn_cast<NameLoc>(callSiteLoc.getCallee())) {
          return nameLoc;
        }
      }
    }
  }
  return std::nullopt;
}

/// Check if two operations have equivalent locations
/// Returns true if they have matching NameLoc or equivalent raw Location
static bool areLocationsEquivalent(Operation *op1, Operation *op2) {
  // First, try to compare using NameLoc
  std::optional<NameLoc> nameLoc1 = getNameLoc(op1);
  std::optional<NameLoc> nameLoc2 = getNameLoc(op2);

  if (nameLoc1.has_value() && nameLoc2.has_value()) {
    LDBG("op1 nameLoc: " << nameLoc1.value()
                         << ", op2 nameLoc: " << nameLoc2.value());
    return nameLoc1.value() == nameLoc2.value();
  }

  // Fallback: compare raw Location objects for equivalence
  Location loc1 = op1->getLoc();
  Location loc2 = op2->getLoc();

  // FileLineColLoc comparison
  if (auto fileLoc1 = dyn_cast<FileLineColLoc>(loc1)) {
    if (auto fileLoc2 = dyn_cast<FileLineColLoc>(loc2)) {
      LDBG("Comparing FileLineColLoc: " << fileLoc1 << " vs " << fileLoc2);
      return fileLoc1.getFilename() == fileLoc2.getFilename() &&
             fileLoc1.getLine() == fileLoc2.getLine() &&
             fileLoc1.getColumn() == fileLoc2.getColumn();
    }
  }

  // Direct Location comparison (works for simple cases)
  LDBG("Comparing raw locations: " << loc1 << " vs " << loc2);
  return loc1 == loc2;
}

/// doPingPongPrep pass: Group expensive ops into pingpong regions
void doPingPongPrep(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  capability = getCapability(moduleOp, capability);

  // Initialize the critical region manager
  CriticalRegionManager crManager;

  // Initialize the dominance and post-dominance info
  mlir::DominanceInfo domInfo(funcOp);
  mlir::PostDominanceInfo postDomInfo(funcOp);

  // A list of expensive op groups.
  // Each group contains ops at the same pingpong region.
  llvm::SmallVector<llvm::SmallVector<Operation *, 4>> expensiveOps;

  // Scan all operations in the function to find expensive ops
  funcOp.walk([&](Operation *op) {
    if (!crManager.isExpensiveOp(op, capability))
      return;

    if (op->getNumOperands() == 0)
      return;

    // IMPORTANT: Assume operations are expensive only for 2D shaped types
    Type operandType = op->getOperand(0).getType();
    int64_t rank = -1;

    if (auto tensorTy = dyn_cast<RankedTensorType>(operandType)) {
      rank = tensorTy.getRank();
      LDBG("RankedTensorType: " << tensorTy << ", rank " << rank);
    } else if (auto memDescTy = dyn_cast<ttg::MemDescType>(operandType)) {
      rank = memDescTy.getRank();
      LDBG("MemDescType: " << memDescTy << ", rank " << rank);
    } else if (auto shapedTy = dyn_cast<ShapedType>(operandType)) {
      // Fallback for other ShapedTypes
      if (shapedTy.hasRank()) {
        rank = shapedTy.getRank();
        LDBG("ShapedType: " << shapedTy << ", rank " << rank);
      }
    }

    if (rank < 2) {
      LDBG("Operand type is not a 2D shaped type, skipping");
      return;
    }

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

        int opTaskId = getSingleTaskId(op);
        int refTaskId = getSingleTaskId(refOp);
        if (opTaskId == -1 || refTaskId == -1) {
          continue;
        }
        if (opTaskId == refTaskId) {
          // Check 2: If in the same partition, they should be control Flow
          // Equivalent
          if (!areControlFlowEquivalent(op, refOp, domInfo, postDomInfo)) {
            matchType = false;
            break;
          }
          // Check 3: the op is dependent on the same tt.split
          Operation *opSplit = getSplitOp(op);
          Operation *refSplit = getSplitOp(refOp);
          // Are from the same data split op
          if (opSplit && refSplit && (opSplit == refSplit)) {
            LDBG("Same split op");
            matchVar = true;
          }
        } else {
          // Check 4: If in different partitions, the op is called from
          // the same source var
          if (areLocationsEquivalent(op, refOp)) {
            LDBG("Same source var (locations equivalent)");
            matchVar = true;
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

#define GEN_PASS_DEF_NVGPUTESTPINGPONGPREP
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestPingPongPrepPass
    : public impl::NVGPUTestPingPongPrepBase<NVGPUTestPingPongPrepPass> {
public:
  using impl::NVGPUTestPingPongPrepBase<
      NVGPUTestPingPongPrepPass>::NVGPUTestPingPongPrepBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    doPingPongPrep(funcOp, numWarpGroups, capability);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

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
