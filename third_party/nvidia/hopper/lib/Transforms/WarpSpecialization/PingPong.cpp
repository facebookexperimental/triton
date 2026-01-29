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
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <unordered_set>

#define DEBUG_TYPE "nvgpu-ping-pong-sync"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

namespace { // anonymous namespace
/// Manages expensive operations for critical region identification and
/// assigns unique barrier IDs to each operation type.
class CriticalRegionManager {
private:
  /// Barrier ID range constants
  /// This pass only uses named barriers 7 - 15 and reserves 0 - 6 for other
  /// uses.
  static constexpr unsigned MIN_BARRIER_ID = 7;
  static constexpr unsigned MAX_BARRIER_ID = 15;

public:
  /// Current barrier ID to assign (range [MIN_BARRIER_ID, MAX_BARRIER_ID])
  int barrierId;

  /// Map from pingpong region id to its barrier ID
  llvm::DenseMap<int, std::pair<int, int>> pingpongIdToBarrierId;

  /// Map from pingpong region id to its critical operations
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToKeyOps;

  /// Map from pingpong region id to operations that mark
  /// the critical region's start and end
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToPingBoundaryOps;
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToPongBoundaryOps;

  /// Map from pingpong region id to the participating thread number
  llvm::DenseMap<int, int> pingpongIdToThreadNum;

  CriticalRegionManager() {
    // Initialize barrier ID to be MIN_BARRIER_ID
    barrierId = MIN_BARRIER_ID;
  }

  /// Check if an operation is registered as an expensive operation for the
  /// given compute capability. Only considers ops with 2D+ shaped operands.
  bool isExpensiveOp(Operation *op, int computeCapability) const {
    switch (computeCapability) {
    case 90: // Hopper
      // On Hopper, wgmma is expensive
      if (isa<ttng::WarpGroupDotOp>(op)) {
        // WarpGroupDotOp has its own verifier that checks the tensor shapes
        // so we can directly put a WarpGroupDotOp into pingpong region
        LDBG("Encounter a " << op->getName() << " op on Hopper.");
        return true;
      }
      break;
    case 100: // Blackwell
      // On Blackwell, exp/exp2 uses SFU which can be expensive for multi-dim
      // tensors Blackwell increases performance for GEMM which is no longer a
      // bottleneck
      if (isa<math::ExpOp, math::Exp2Op>(op)) {
        LDBG("Encounter a " << op->getName() << " op on Blackwell.");
        Type resultType = op->getResult(0).getType();
        if (auto tensorTy = dyn_cast<RankedTensorType>(resultType))
          return tensorTy.getRank() > 1;
      }
      break;
    default:
      break;
    }
    return false;
  }

  /// Assign barrier IDs for a pingpong region.
  /// Sets barrier IDs to -1 if we have exhausted available barriers.
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

    // Check if we would exceed the maximum barrier ID
    if (barrierId > MAX_BARRIER_ID || barrierId_1 > MAX_BARRIER_ID) {
      pingpongIdToBarrierId[pingpongId] = {-1, -1};
      LDBG("Barrier IDs exhausted, assigning {-1, -1} to pingpong region '"
           << pingpongId << "'.");
      return;
    }

    pingpongIdToBarrierId[pingpongId] = {barrierId, barrierId_1};
    LDBG("Assigned barrier ID {" << barrierId << ", " << barrierId_1
                                 << "} to pingpong region '" << pingpongId
                                 << "'.");

    // Increment the barrier ID counter
    this->barrierId = barrierId + 2;
  }

  bool hasPingBoundary(int pingpongRegionId) const {
    return (pingpongIdToPingBoundaryOps.count(pingpongRegionId) > 0) &&
           (pingpongIdToPingBoundaryOps.at(pingpongRegionId).size() == 2);
  }

  bool hasPongBoundary(int pingpongRegionId) const {
    return (pingpongIdToPongBoundaryOps.count(pingpongRegionId) > 0) &&
           (pingpongIdToPongBoundaryOps.at(pingpongRegionId).size() == 2);
  }

  bool hasPingPongBoundary(int pingpongRegionId) const {
    return (pingpongIdToPingBoundaryOps.count(pingpongRegionId) > 0) &&
           (pingpongIdToPingBoundaryOps.at(pingpongRegionId).size() == 2) &&
           (pingpongIdToPongBoundaryOps.count(pingpongRegionId) > 0) &&
           (pingpongIdToPongBoundaryOps.at(pingpongRegionId).size() == 2);
  }

  void dumpBoundaryOps() const {
    LDBG("===== Critical Region Manager Dump =====");
    LDBG("pingpongIdToPingBoundaryOps");
    for (const auto &entry : pingpongIdToPingBoundaryOps) {
      LDBG("pingpongId: " << entry.first);
      for (const auto &op : entry.second) {
        LDBG("  ping boundary op: " << op->getName().getStringRef());
      }
    }
    LDBG("pingpongIdToPongBoundaryOps");
    for (const auto &entry : pingpongIdToPongBoundaryOps) {
      LDBG("pingpongId: " << entry.first);
      for (const auto &op : entry.second) {
        LDBG("  pong boundary op: " << op->getName().getStringRef());
      }
    }
  }
};

/// Returns the taskId if op has a single taskId, otherwise, returns -1.
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

/// Return a map of loop depth to the loop ops in the partition.
void getNestedFor(Region *partition,
                  DenseMap<unsigned, SmallVector<Operation *>> &loopDepthMap) {
  partition->walk([&](Operation *subOp) {
    if (dyn_cast<scf::ForOp>(subOp)) {
      unsigned tDepth = getLoopDepth(subOp);
      loopDepthMap[tDepth].push_back(subOp);
    }
  });
}

/// Returns true if both operations are in the same block with no intervening
/// control flow operations. False otherwise.
bool areControlFlowEquivalent(Operation *op1, Operation *op2) {
  if (!op1 || !op2)
    return false;

  if (op1->getBlock() != op2->getBlock())
    return false;

  // Determine which op comes first
  Operation *earlier = op1;
  Operation *later = op2;
  if (later->isBeforeInBlock(earlier))
    std::swap(earlier, later);

  // Check for intervening control flow operations
  for (Operation *cur = earlier->getNextNode(); cur && cur != later;
       cur = cur->getNextNode()) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(cur))
      return false;
  }

  return true;
}

/// Dump memory effects of an operation for debugging
void dumpMemoryEffects(Operation *op) {
  auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memInterface) {
    LDBG("  Op '" << op->getName().getStringRef()
                  << "' does not implement MemoryEffectOpInterface");
    return;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memInterface.getEffects(effects);

  if (effects.empty()) {
    LDBG("  Op '" << op->getName().getStringRef() << "' has no memory effects");
    return;
  }

  for (const auto &effect : effects) {
    llvm::StringRef effectType;
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

    llvm::StringRef resourceName = effect.getResource()->getName();
    LDBG("  Op '" << op->getName().getStringRef() << "' has effect: "
                  << effectType << " on resource: " << resourceName);
  }
}

/// Find the start boundary op for the critical region.
Operation *findStartOp(CriticalRegionManager &crManager, Operation *keyOp) {
  // Set the start op of this pingpong region to be the critical op
  return keyOp;
}

/// Find the end boundary op for the critical region.
/// Scans from keyOp until it finds an op with memory side effects,
/// a control flow break, or reaches stopOp (if provided).
/// Returns nullptr if stopOp is reached without finding a valid end boundary.
Operation *findEndOp(CriticalRegionManager &crManager, Operation *keyOp,
                     Operation *stopOp = nullptr) {
  Operation *curOp = keyOp;
  Operation *later = stopOp;

  // Determine which op comes first
  if (stopOp) {
    if (later->isBeforeInBlock(curOp))
      std::swap(curOp, later);
  }

  // Set the end op of this pingpong region to be the first op with memory side
  // effect after this critical op
  while (curOp) {
    if (!isMemoryEffectFree(curOp)) {
      LDBG("Found op with memory effects:");
      dumpMemoryEffects(curOp);
      return curOp;
    }
    // If we've reached the stop op, there's no memory effect between them
    if (stopOp && curOp == stopOp) {
      return nullptr;
    }
    // Check if we've hit a control flow boundary
    // Set end op to the end of the control flow equivalent region
    Operation *nextOp = curOp->getNextNode();
    if (!nextOp || !areControlFlowEquivalent(curOp, nextOp))
      return curOp;
    curOp = nextOp;
  }
  return nullptr;
}

/// Returns the operation from startOps that is closest to the entry
/// (executed earliest). All ops must be in the same block.
Operation *firstOpInBlock(llvm::ArrayRef<Operation *> startOps) {
  if (startOps.empty())
    return nullptr;

  assert(llvm::all_of(startOps,
                      [&](Operation *op) {
                        return op->getBlock() == startOps[0]->getBlock();
                      }) &&
         "firstOpInBlock called with ops in different blocks");

  auto it = llvm::min_element(startOps, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  return *it;
}

/// Returns the operation from endOps that is closest to the terminator
/// (executed latest). All ops must be in the same block.
Operation *lastOpInBlock(llvm::ArrayRef<Operation *> endOps) {
  if (endOps.empty())
    return nullptr;

  assert(llvm::all_of(endOps,
                      [&](Operation *op) {
                        return op->getBlock() == endOps[0]->getBlock();
                      }) &&
         "lastOpInBlock called with ops in different blocks");

  auto it = llvm::max_element(
      endOps, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return *it;
}

/// Returns the partition ID that contains the keyOp that occurs first.
/// Ordering is determined by:
///   1. Stage number (lower stage executes first)
///   2. Cluster number (lower cluster executes first if same stage)
///   3. Program order (isBeforeInBlock if same stage and cluster)
int arrivesFirst(llvm::ArrayRef<Operation *> keyOps) {
  assert(llvm::all_of(
             keyOps, [&](Operation *op) { return ttg::getStageCluster(op); }) &&
         "Loop stage and cluster not found for all key ops");

  auto it = llvm::min_element(keyOps, [](Operation *a, Operation *b) {
    auto scA = ttg::getStageCluster(a);
    auto scB = ttg::getStageCluster(b);

    int stageA = scA->first;
    int stageB = scB->first;
    int clusterA = scA->second;
    int clusterB = scB->second;
    if (stageA == stageB) {
      if (clusterA == clusterB) {
        return a->isBeforeInBlock(b);
      } else {
        return clusterA < clusterB;
      }
    } else {
      return stageA < stageB;
    }
  });
  return getSingleTaskId(*it);
}

/// Process a WarpSpecializeOp to insert pingpong barriers for critical regions.
/// Finds ops with pingpong_id attributes, computes their boundaries, assigns
/// named barrier IDs, and inserts arrive/wait barriers to enforce mutual
/// exclusion between ping and pong partitions.
static void handleWarpSpec(ttg::WarpSpecializeOp wsOp, int computeCapability) {
  // Get the function op
  auto funcOp = wsOp->getParentOfType<triton::FuncOp>();
  assert(funcOp != nullptr);

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
        LDBG("Found op " << op->getName().getStringRef() << " with pingpong id "
                         << pingpongId);
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
      if (partitionId != -1) {
        Operation *startOp = findStartOp(crManager, keyOp);
        Operation *endOp = findEndOp(crManager, keyOp, nullptr);
        startOps[partitionId].push_back(startOp);
        endOps[partitionId].push_back(endOp);
        // Look up the number of warps for each partition
        if (numWarps.count(partitionId) == 0) {
          numWarps[partitionId] = ttg::lookupNumWarps(keyOp);
          LDBG("numWarps of " << partitionId << " is "
                              << numWarps[partitionId]);
        }
      }
    }

    if (startOps.size() != 2 || endOps.size() != 2 || numWarps.size() != 2) {
      LDBG("pingpong ops are not in two partitions");
      continue;
    }

    // Find which partition arrives first
    int arrivesFirstPartitionId = arrivesFirst(keyOps);
    LDBG("arrivesFirstPartitionId " << arrivesFirstPartitionId);

    int numberOfThreads = 0;
    for (auto [partitionId, startOp] : startOps) {
      // The start and end ops are unioned for each partition to find the
      // boundary ops
      Operation *unionStartOp = firstOpInBlock(startOp);
      Operation *unionEndOp = lastOpInBlock(endOps[partitionId]);

      // The pong partition is the partition that arrives first
      if (partitionId != arrivesFirstPartitionId) {
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

    if (pingBarrierId == -1 || pongBarrierId == -1) {
      LDBG("Named barriers have run out for the pingpong region " << pingpongId
                                                                  << ".");
      continue;
    }

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
} // anonymous namespace

/// doPingPongSync pass: Insert pingpong barriers to the IR
void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  capability = getNVIDIAComputeCapability(moduleOp);
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &bodyOp : block.getOperations()) {
      Operation *op = &bodyOp;
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        handleWarpSpec(wsOp, capability);
      }
    }
  }
}

/// doPingPongPrep pass: Group expensive ops into pingpong regions
void doPingPongPrep(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  capability = getNVIDIAComputeCapability(moduleOp);

  // Initialize the critical region manager
  CriticalRegionManager crManager;

  // A list of expensive op groups.
  // Each group contains ops at the same pingpong region.
  llvm::SmallVector<llvm::SmallVector<Operation *, 4>> expensiveOps;

  // Scan all operations in the function to find expensive ops
  funcOp.walk([&](Operation *op) {
    if (!crManager.isExpensiveOp(op, capability))
      return;

    // Check if the expensive op belongs to an existing group
    bool foundGroup = false;
    for (auto &group : expensiveOps) {
      bool matchType = true;
      // bool matchVar = false;
      for (auto &refOp : group) {
        // Check 1: Same Operation Name
        if (op->getName() != refOp->getName()) {
          matchType = false;
          break;
        }

        // Check 2: Same block with no intervening control flow ops
        if (!areControlFlowEquivalent(op, refOp)) {
          matchType = false;
          break;
        }

        // Check 3: no memory side effect ops between two ops
        int opTaskId = getSingleTaskId(op);
        int refTaskId = getSingleTaskId(refOp);
        if (opTaskId == -1 || refTaskId == -1) {
          continue;
        }
        if (opTaskId == refTaskId) {
          // If findEndOp returns nullptr when stopOp is provided,
          // there's no memory effect between keyOp and stopOp
          bool hasMemEffects = (findEndOp(crManager, op, refOp) != nullptr);
          LDBG("op in partition " << opTaskId
                                  << " has memory effects: " << hasMemEffects);
          if (hasMemEffects)
            matchType = false;
        }
      }
      foundGroup = matchType;
      if (foundGroup) {
        LDBG("Insert to ref op group " << group[0]->getName().getStringRef());
        group.push_back(op);
        break;
      }
    }

    if (!foundGroup) {
      LDBG("Create new group for op " << op->getName().getStringRef());
      expensiveOps.push_back({op});
    }
  });

  // pingpong region ID
  unsigned pingpongID = 0;

  // Assign pingpong region IDs to groups
  for (auto &group : expensiveOps) {
    if (group.size() < 2)
      continue;

    // Count the number of distinct partitions in the group
    llvm::SmallDenseSet<int, 4> partitionIds;
    for (auto *op : group) {
      int taskId = getSingleTaskId(op);
      if (taskId != -1)
        partitionIds.insert(taskId);
    }

    // Only handle pingpong for the case of 2 different partitions
    if (partitionIds.size() != 2)
      continue;

    for (auto *op : group) {
      op->setAttr(
          "pingpong_id",
          IntegerAttr::get(IntegerType::get(op->getContext(), 32), pingpongID));
      LDBG("Assign pingpong_id " << pingpongID << " to op '"
                                 << op->getName().getStringRef()
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
