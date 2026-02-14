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
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
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
  unsigned barrierId = MIN_BARRIER_ID;

  /// Map from pingpong region id to its barrier ID
  llvm::DenseMap<int, std::pair<unsigned, unsigned>> pingpongIdToBarrierId;

  /// Map from pingpong region id to its critical operations
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToKeyOps;

  /// Map from pingpong region id to operations that mark
  /// the critical region's start and end
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToPingBoundaryOps;
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToPongBoundaryOps;

  /// Map from pingpong region id to the participating thread number
  llvm::DenseMap<int, int> pingpongIdToThreadNum;

  CriticalRegionManager() = default;

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
    unsigned barrierId = this->barrierId;
    unsigned barrierId_1 = this->barrierId + 1;

    // Check if we would exceed the maximum barrier ID
    if (this->barrierId + 1 > MAX_BARRIER_ID) {
      LDBG("Barrier IDs exhausted for pingpong region '" << pingpongId << "'.");
      return;
    }

    pingpongIdToBarrierId[pingpongId] = {barrierId, barrierId_1};
    LDBG("Assigned barrier ID {" << barrierId << ", " << barrierId_1
                                 << "} to pingpong region '" << pingpongId
                                 << "'.");

    // Increment the barrier ID counter
    this->barrierId = barrierId + 2;
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
        LDBG("  ping boundary op: " << op->getName());
      }
    }
    LDBG("pingpongIdToPongBoundaryOps");
    for (const auto &entry : pingpongIdToPongBoundaryOps) {
      LDBG("pingpongId: " << entry.first);
      for (const auto &op : entry.second) {
        LDBG("  pong boundary op: " << op->getName());
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
  assert(op1 && op2 &&
         "Both input ops of areControlFlowEquivalent must be non-null.");

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
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    LDBG("  Op '" << op->getName()
                  << "' does not implement MemoryEffectOpInterface.");
    SmallVector<MemoryEffects::EffectInstance> effects;
    memInterface.getEffects(effects);
    if (effects.empty()) {
      LDBG("  Op '" << op->getName() << "' has no memory effects.");
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
      LDBG("  Op '" << op->getName() << "' has effect: " << effectType
                    << " on resource: " << resourceName);
    }
  } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    LDBG("  Op '" << op->getName() << "' may have recursive memory effects.");
  }
  return;
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
    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(curOp)) {
      LDBG("Found control flow op " << curOp->getName());
      return nullptr;
    }
    if (!isMemoryEffectFree(curOp)) {
      LDBG("Found op with memory effects: " << curOp->getName());
      dumpMemoryEffects(curOp);
      return curOp;
    }
    // If we've reached the stop op, there's no memory effect between them
    if (curOp == stopOp) {
      return nullptr;
    }
    // Check if we've hit a control flow boundary
    // Set end op to the end of the control flow equivalent region
    Operation *nextOp = curOp->getNextNode();
    if (!nextOp || !areControlFlowEquivalent(curOp, nextOp))
      return nullptr;
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

/// Validate that critical ops alternate between partitions in contiguous blocks
/// and return the partition ID that arrives first. Returns -1 if the schedule
/// is invalid (ops have interleaved schedule order or don't alternate properly).
///
/// Uses the linearized schedule to walk from the first critical op and verify
/// the pattern:
///   [partition A ops] [partition B ops] [partition A ops] [partition B ops] ...
int arrivesFirst(
    scf::ForOp forOp, const triton::CoarseSchedule &schedule,
    const llvm::DenseMap<int, SmallVector<Operation *>> &partitionOps) {
  // Collect all critical ops across partitions
  SmallVector<Operation *> allOps;
  for (auto &[partitionId, ops] : partitionOps) {
    allOps.append(ops.begin(), ops.end());
  }

  assert(llvm::all_of(
             allOps, [&](Operation *op) { return ttg::getStageCluster(op); }) &&
         "Loop stage and cluster not found for all key ops");

  // Build a set of critical ops for fast lookup
  llvm::SmallDenseSet<Operation *, 8> criticalOps(allOps.begin(), allOps.end());

  // Find the earliest critical op in the schedule
  Operation *firstOp =
      *llvm::min_element(allOps, [&](Operation *a, Operation *b) {
        return schedule.isOpBefore(a, b);
      });
  assert(firstOp && "Failed to find the earliest op in the schedule");

  // Validate that the schedule alternates between partitions
  int curPartitionId = getSingleTaskId(firstOp);
  int curSeenOps = 1;

  auto linearized = schedule.linearized(forOp, firstOp);

  while (auto nextOp = linearized.findNext(
        [&](Operation *op) { return criticalOps.contains(op); })) {
    int nextPartitionId = getSingleTaskId(*nextOp);
    if (nextPartitionId == curPartitionId) {
      curSeenOps++;
      if (curSeenOps > partitionOps.lookup(curPartitionId).size()) {
        LDBG("Partition " << curPartitionId << " have scheduled "
              << curSeenOps << " ops consecutively, not alternating.");
        return -1;
      }
    } else {
      if (curSeenOps != partitionOps.lookup(curPartitionId).size()) {
        LDBG("Partition " << curPartitionId << " scheduled "
                          << curSeenOps << " before the next partition, not alternating.");
        return -1;
      }
      curSeenOps = 1;
    }
    curPartitionId = nextPartitionId;
  }

  if (curSeenOps != partitionOps.lookup(curPartitionId).size()) {
    LDBG("Partition " << curPartitionId << " scheduled "
                      << curSeenOps << " before the next partition, not alternating.");
    return -1;
  }

  return getSingleTaskId(firstOp);
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
      if (auto pingpongIdAttr = op->getAttrOfType<IntegerAttr>("pingpong_id")) {
        int pingpongId = pingpongIdAttr.getInt();
        LDBG("Found op " << op->getName() << " with pingpong id "
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
    bool foundNullEndOp = false;
    int arrivesFirstPartitionId = -1;
    for (auto &keyOp : keyOps) {
      int partitionId = getSingleTaskId(keyOp);
      if (partitionId != -1) {
        Operation *startOp = keyOp;
        Operation *endOp = findEndOp(crManager, keyOp, nullptr);
        if (!endOp) {
          foundNullEndOp = true;
          break;
        }
        startOps[partitionId].push_back(startOp);
        endOps[partitionId].push_back(endOp);
        // Look up the number of warps for each partition
        if (numWarps.count(partitionId) == 0) {
          numWarps[partitionId] = ttg::lookupNumWarps(keyOp);
          LDBG("numWarps of " << partitionId << " is "
                              << numWarps[partitionId]);
        }
        // Get the first partition id from the attribute
        if (auto attr = keyOp->getAttrOfType<IntegerAttr>(
                "pingpong_first_partition_id")) {
          arrivesFirstPartitionId = attr.getInt();
        }
      }
    }
    if (foundNullEndOp)
      continue;

    if (arrivesFirstPartitionId == -1) {
      LDBG("pingpong_first_partition_id attribute not found");
      continue;
    }
    LDBG("arrivesFirstPartitionId " << arrivesFirstPartitionId);

    if (startOps.size() != 2 || endOps.size() != 2 || numWarps.size() != 2) {
      LDBG("pingpong ops are not in two partitions");
      continue;
    }

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
  for (auto &[pingpongId, pingBoundOps] :
       crManager.pingpongIdToPingBoundaryOps) {
    if (!crManager.hasPingPongBoundary(pingpongId))
      continue;
    const SmallVector<Operation *> &pongBoundOps =
        crManager.pingpongIdToPongBoundaryOps[pingpongId];

    if (crManager.pingpongIdToBarrierId.count(pingpongId) == 0) {
      LDBG("Named barriers have run out for the pingpong region " << pingpongId
                                                                  << ".");
      continue;
    }

    auto [pingBarrierId, pongBarrierId] =
        crManager.pingpongIdToBarrierId[pingpongId];

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
                    int capability, int defaultNumStages) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  capability = getNVIDIAComputeCapability(moduleOp);

  // Initialize the critical region manager
  CriticalRegionManager crManager;

  // A list of expensive op groups.
  // Each group contains ops at the same pingpong region.
  llvm::SmallVector<llvm::SmallVector<Operation *, 4>> expensiveOps;

  // Step 1: Group find expensive ops into pingpong regions
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
        LDBG("Insert to ref op group " << group[0]->getName());
        group.push_back(op);
        break;
      }
    }

    if (!foundGroup) {
      LDBG("Create new group for op " << op->getName());
      expensiveOps.push_back({op});
    }
  });

  // pingpong region ID
  unsigned pingpongID = 0;

  // Step 2: Assign pingpong region ID to each group
  for (auto &group : expensiveOps) {
    // Categorize ops into ping and pong partitions
    llvm::DenseMap<int, SmallVector<Operation *>> partitionOps;
    // The parent scf::ForOp for the critical ops
    scf::ForOp forOp = nullptr;
    for (auto *op : group) {
      int taskId = getSingleTaskId(op);
      if (taskId != -1)
        partitionOps[taskId].push_back(op);
      // ops share control flow, so taking the last parent ForOp is safe
      if (auto parentFor = op->getParentOfType<scf::ForOp>())
        forOp = parentFor;
    }

    // Only handle pingpong for the case of 2 different partitions
    if (partitionOps.size() != 2)
      continue;

    // Only handle pingpong when inside loops
    if (!forOp) {
      LDBG("No parent ForOp found, skipping this critical region.");
      continue;
    }

    // Extract the schedule from the parent ForOp
    triton::CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp))) {
      LDBG("No schedule found, re-running scheduleLoops");

      // Get the parent ModuleOp
      auto moduleOp = forOp->getParentOfType<ModuleOp>();

      // Re-run scheduling for all loops in the module
      int numStages = triton::getNumStagesOrDefault(forOp, defaultNumStages);
      bool useMetaWS = true; // or false depending on your use case
      triton::gpu::scheduleLoops(moduleOp, numStages, useMetaWS);

      // Now try deserializing again
      if (failed(schedule.deSerialize(forOp))) {
        LDBG("Still failed after re-running scheduleLoops, skipping");
        continue;
      }
    }

    // Find which partition arrives first and validate alternation pattern.
    // Returns -1 if the schedule is invalid (ops interleave or don't alternate).
    int arrivesFirstPartitionId = arrivesFirst(forOp, schedule, partitionOps);
    if (arrivesFirstPartitionId == -1) {
      LDBG("Skipping group due to invalid pingpong schedule pattern");
      continue;
    }

    for (auto *op : group) {
      op->setAttr(
          "pingpong_id",
          IntegerAttr::get(IntegerType::get(op->getContext(), 32), pingpongID));
      op->setAttr("pingpong_first_partition_id",
                  IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                   arrivesFirstPartitionId));
      LDBG("Assign pingpong_id " << pingpongID << " to op '" << op->getName()
                                 << "' with task_id " << getSingleTaskId(op)
                                 << ", first_partition_id "
                                 << arrivesFirstPartitionId);
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
    doPingPongPrep(funcOp, numWarpGroups, capability, numStages);
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
