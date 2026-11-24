//===----------------------------------------------------------------------===//
// PingPong Barrier Insertion Pass
//
// Enforce pingpong around expensive ops (warp_group_dot, math.exp,
// tanh.approx inline asm) across warp partitions by inserting mbarriers.
//
// Two passes:
//   1. doPingPongPrep: Preprocess to group expensive ops that
//      i) of the same type,
//      ii) in the same control flow, and
//      iii) operate on the same or subtiled variables
//      into pingpong regions and assign a unique pingpong_id.
//
//   2. doPingPongSync: For each pingpong region, identify start and end
//      boundaries, and insert arrive/wait mbarriers to the IR.
//
// Barrier pattern:
//   Ping: arrive(pong) at entry, wait(ping) before op, arrive(pong) after op
//   Pong: wait(pong) before op, arrive(ping) after op
//
// Critical op types:
//   - NonReorderable (warp_group_dot): has memory effects, boundary is the op
//   - PureArithmetic (math.exp, tanh.approx inline asm): boundary extends to
//     next memory op
//===----------------------------------------------------------------------===//

#include "Utility.h"
#include "WarpSpecializationPipeline.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "nvgpu-ping-pong-sync"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

namespace { // anonymous namespace
/// SFU-backed math ops. On NVPTX, exp2 lowers directly to ex2.approx;
/// sin/cos/tanh lower to their .approx forms when approximations are allowed
/// (else MUFU-seeded libdevice); sqrt/rsqrt lower through rsqrt.approx (plus
/// refinement unless approximations are allowed). Deliberately excludes log2
/// (libdevice does not use lg2.approx; the approx lowering is disabled by
/// default) and erf (no approx path).
static bool isSFUMathOp(Operation *op) {
  return isa<math::ExpOp, math::Exp2Op, math::SinOp, math::CosOp, math::TanhOp,
             math::SqrtOp, math::RsqrtOp>(op);
}

/// SFU-backed inline asm: pure elementwise asm blocks invoking PTX
/// instructions that execute on the special-function unit. Per the PTX ISA
/// (floating-point §9.7.3 and half-precision §9.7.4 instructions), the
/// single-precision transcendental/reciprocal approximations -- sin, cos,
/// ex2, lg2, tanh, rcp, rsqrt, sqrt and div .approx -- lower to SFU/MUFU
/// hardware at a fraction of FFMA throughput. Deliberately excludes the
/// correctly-rounded/full-range spellings (div.full, .rn), which lower
/// differently. Matches the asm template text since the op carries no finer
/// opcode.
///
/// The template must also be straight-line register arithmetic of the kind
/// TLX kernels emit (e.g. tanh.approx, f32x2 fma/mul blocks): any barrier,
/// synchronization, memory, control-flow or collective operation in the
/// template disqualifies the whole op, since the ping-pong region must not
/// wrap such effects. Tokens are matched as instruction prefixes (e.g.
/// "bar." covers bar.sync/bar.warp.sync/bar.cluster.*) rather than
/// enumerating every PTX variant. Matching is intentionally conservative and
/// may reject on comments; conversely, unlisted spellings may slip through,
/// so prefer prefix tokens when extending this list.
static bool isSFUInlineAsmOp(Operation *op) {
  auto asmOp = dyn_cast<tt::ElementwiseInlineAsmOp>(op);
  if (!asmOp || !asmOp.getPure())
    return false;
  StringRef asmStr = asmOp.getAsmString();
  static constexpr StringRef kSFUApproxMnemonics[] = {
      "sin.approx", "cos.approx",   "ex2.approx",  "lg2.approx", "tanh.approx",
      "rcp.approx", "rsqrt.approx", "sqrt.approx", "div.approx"};
  bool hasSFU = llvm::any_of(kSFUApproxMnemonics, [&](StringRef mnemonic) {
    return asmStr.contains(mnemonic);
  });
  if (!hasSFU)
    return false;
  static constexpr StringRef kUnsafeSubstrings[] = {
      // Barriers and synchronization.
      "bar.",
      "barrier",
      "mbarrier",
      // Memory ordering.
      "fence",
      "membar",
      // Memory access.
      "ld.",
      "st.",
      "cp.",
      "atom.",
      "red.",
      "prefetch",
      // Control flow ("bra " matches branches, not identifiers like bravo).
      "bra ",
      "bra\t",
      "bra.",
      "call",
      "ret;",
      "exit",
      "trap",
      // Async / collective operations.
      "wgmma",
      "tcgen05",
      "shfl",
      "vote",
  };
  return !llvm::any_of(kUnsafeSubstrings, [&](StringRef substr) {
    return asmStr.contains(substr);
  });
}

/// Any op that exercises the SFU on NVIDIA GPUs.
static bool isSFUOp(Operation *op) {
  return isSFUMathOp(op) || isSFUInlineAsmOp(op);
}

/// Manages expensive operations for critical region identification and
/// assigns unique barrier IDs to each operation type.
class CriticalRegionManager {
public:
  /// Map from pingpong region id to its critical operations
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToKeyOps;

  /// Map from pingpong region id to operations that mark
  /// the critical region's start and end
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToPingBoundaryOps;
  llvm::DenseMap<int, SmallVector<Operation *>> pingpongIdToPongBoundaryOps;

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
      // On Blackwell, SFU-backed ops are expensive for multi-dim tensors.
      // Blackwell increases performance for GEMM which is no longer a
      // bottleneck
      if (isSFUOp(op)) {
        LDBG("Encounter a " << op->getName() << " op on Blackwell.");
        Type resultType = op->getResult(0).getType();
        if (auto tensorTy = dyn_cast<RankedTensorType>(resultType))
          return tensorTy.getRank() > 1;
      }
      break;
    }
    return false;
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

// Thread a 0-based iteration counter through `nest` (innermost first, as
// collected by `collectDirectLoopNest`) and return the value visible inside
// the innermost loop. The innermost loop advances it, each enclosing loop
// carries out what its inner loop produced, and each inner loop resumes from
// the value its parent is carrying -- so the count runs across the whole nest
// rather than restarting.
//
// This replaces deriving the count from the induction variables. Doing that
// arithmetically needs the loop bounds to cooperate: it divides `iv - lb`
// unsigned (wrong for the negative step `scf.for` permits) and multiplies by a
// trip count read once (wrong when an inner bound depends on an outer
// induction variable, as in a causal/triangular nest). A carried counter is
// exact for any bounds, and it is what the rest of warp specialization already
// does -- see `getAccumCount` in `CodePartitionUtility.cpp`, which reads its
// count out of a loop-carried argument for exactly this reason.
static Value createIterationCounter(SmallVector<scf::ForOp> nest) {
  if (nest.empty())
    return {};
  std::reverse(nest.begin(), nest.end());

  // Give every loop a counter, seeded with zero for now. Rewriting a loop
  // moves its body rather than cloning it, so the inner entries stay valid.
  SmallVector<BlockArgument> counters;
  for (scf::ForOp &loop : nest) {
    OpBuilder builder(loop);
    Value zero = arith::ConstantIntOp::create(builder, loop.getLoc(), 0, 32);
    loop = addIterArgsToLoop(builder, loop, {zero});
    counters.push_back(loop.getRegionIterArgs().back());
  }

  // Now wire the counters together, innermost first: only the innermost loop
  // increments, and an inner loop's starting value is its parent's counter.
  for (int i = nest.size() - 1; i >= 0; --i) {
    scf::ForOp loop = nest[i];
    if (i + 1 == static_cast<int>(nest.size())) {
      OpBuilder builder(loop.getBody()->getTerminator());
      Value one = arith::ConstantIntOp::create(builder, loop.getLoc(), 1, 32);
      appendToForOpYield(loop, {arith::AddIOp::create(builder, loop.getLoc(),
                                                      counters[i], one)});
    } else {
      appendToForOpYield(loop, {nest[i + 1].getResults().back()});
    }
    if (i > 0)
      loop.getInitArgsMutable()[loop.getInitArgs().size() - 1].set(
          counters[i - 1]);
  }
  return counters.back();
}

// The mbarrier flips once per arrival, and each iteration arrives once, so the
// phase a wait expects is the parity of the iteration count.
static Value getBarrierPhase(OpBuilder &builder, Location loc, Value counter) {
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  return arith::AndIOp::create(builder, loc, counter, one);
}

static unsigned captureInPartitions(ttg::WarpSpecializeOp wsOp, Value value) {
  ttg::WarpSpecializePartitionsOp partitions = wsOp.getPartitionOp();
  unsigned captureIdx = partitions.getNumOperands();
  for (auto [idx, capture] :
       llvm::enumerate(partitions.getExplicitCaptures())) {
    if (capture == value) {
      captureIdx = idx;
      break;
    }
  }
  if (captureIdx == partitions.getNumOperands()) {
    partitions->insertOperands(captureIdx, value);
    for (Region *region : wsOp.getPartitionRegions())
      region->addArgument(value.getType(), value.getLoc());
  }
  return captureIdx;
}

static Region *getPartitionRegion(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    // A top-level region has no parent op.
    if (!parent)
      return nullptr;
    if (isa<ttg::WarpSpecializePartitionsOp>(parent))
      return region;
    region = parent->getParentRegion();
  }
  return nullptr;
}

// The scf.for loops enclosing `op`, innermost first. Empty when no loop
// encloses `op`, when the chain crosses the partition region boundary, or
// when any enclosing region op is not a loop: a guard can skip arrivals on
// one side, and threading a counter through it can violate dominance.
static SmallVector<scf::ForOp> collectDirectLoopNest(Operation *op) {
  Region *partition = getPartitionRegion(op);
  if (!partition)
    return {};
  for (Operation *parent = op->getParentOp();
       parent != partition->getParentOp(); parent = parent->getParentOp()) {
    if (!isa<scf::ForOp>(parent))
      return {};
  }
  SmallVector<scf::ForOp> nest;
  for (auto loop = op->getParentOfType<scf::ForOp>(); loop;
       loop = loop->getParentOfType<scf::ForOp>())
    nest.push_back(loop);
  if (nest.empty())
    return nest;
  for (size_t i = 1; i < nest.size(); ++i)
    if (nest[i - 1]->getBlock() != nest[i].getBody())
      return {};
  // The nest must stay inside the partition region, which is isolated from
  // above: a loop outside it cannot carry a counter into it.
  if (getPartitionRegion(nest.back()) != partition)
    return {};
  return nest;
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
    LDBG("  Op '" << op->getName() << "' implements MemoryEffectOpInterface.");
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

/// Returns true if there is a region-splitting memory side effect strictly
/// between two same-partition expensive ops.
///
/// This is the grouping-time question: can two expensive ops live in the same
/// ping-pong region? It differs from findEndOp (which finds a region's *end*
/// boundary and, for NonReorderable ops, returns the op itself) in two ways:
///   - The two endpoint ops are excluded. An expensive op such as
///   WarpGroupDotOp
///     is NonReorderable: its own SMEM-operand read is the op itself, not an
///     intervening effect, so it must not reject grouping (B-4 / T273470439).
///   - Other expensive ops between the endpoints are skipped. They are peers in
///     the same ping-pong region (managed by their own barriers), so they do
///     not split the region either; skipping them lets a run of N expensive ops
///     union into a single region.
///
/// Callers must ensure the two ops are control-flow equivalent
/// (areControlFlowEquivalent), which guarantees they share a block with no
/// intervening control flow, so a simple forward scan terminates at `b`.
bool hasInterveningMemEffect(CriticalRegionManager &crManager, Operation *a,
                             Operation *b, int capability) {
  Operation *earlier = a;
  Operation *later = b;
  if (later->isBeforeInBlock(earlier))
    std::swap(earlier, later);

  for (Operation *curOp = earlier->getNextNode(); curOp && curOp != later;
       curOp = curOp->getNextNode()) {
    // Peer expensive ops are part of the same ping-pong region, not a barrier.
    if (crManager.isExpensiveOp(curOp, capability))
      continue;
    if (!isMemoryEffectFree(curOp)) {
      LDBG("Found intervening op with memory effects: " << curOp->getName());
      dumpMemoryEffects(curOp);
      return true;
    }
  }
  return false;
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
/// is invalid (ops have interleaved schedule order or don't alternate
/// properly).
///
/// Uses the linearized schedule to walk from the first critical op and verify
/// the pattern:
///   [partition A ops] [partition B ops] [partition A ops] [partition B ops]
///   ...
int arrivesFirst(
    scf::ForOp forOp, const triton::CoarseSchedule &schedule,
    const llvm::DenseMap<int, SmallVector<Operation *>> &partitionOps) {
  // Collect all critical ops across partitions
  llvm::SmallDenseSet<Operation *, 8> criticalOps;
  for (auto &[partitionId, ops] : partitionOps) {
    criticalOps.insert(ops.begin(), ops.end());
  }

  assert(
      llvm::all_of(criticalOps,
                   [&](Operation *op) { return ttg::getStageCluster(op); }) &&
      "Loop stage and cluster not found for all key ops");

  // Step 1: Find the earliest critical op by linearizing from the start of the
  // loop
  auto linearized = schedule.linearized(
      forOp, &*forOp.getBody()->without_terminator().begin());
  auto firstCriticalOp = linearized.findNext(
      [&](Operation *op) { return criticalOps.contains(op); });
  if (!firstCriticalOp) {
    LDBG("Failed to find the earliest critical op in the schedule");
    return -1;
  }
  Operation *firstOp = *firstCriticalOp;

  // Step 2: Validate that the schedule alternates between partitions
  //         - Correct alternation means: after all ops in one partition
  //         execute, the next scheduled op must be in the other partition
  //         - Check correct alternation until we reach the end of linearized
  //         schedule
  int curPartitionId = getSingleTaskId(firstOp);
  int curSeenOps = 1;

  while (auto nextOp = linearized.findNext(
             [&](Operation *op) { return criticalOps.contains(op); })) {
    int nextPartitionId = getSingleTaskId(*nextOp);
    if (nextPartitionId == curPartitionId) {
      // Check if operations in the same partition get scheduled consecutively
      // more than once
      curSeenOps++;
      if (curSeenOps > partitionOps.lookup(curPartitionId).size()) {
        LDBG("Partition " << curPartitionId << " have scheduled " << curSeenOps
                          << " ops consecutively, not alternating.");
        return -1;
      }
    } else {
      // Check if operations in the other partition get scheduled after ALL
      // operations in the current partition are scheduled
      if (curSeenOps != partitionOps.lookup(curPartitionId).size()) {
        LDBG("Partition " << curPartitionId << " scheduled " << curSeenOps
                          << " before the next partition, not alternating.");
        return -1;
      }
      curSeenOps = 1;
    }
    curPartitionId = nextPartitionId;
  }

  if (curSeenOps != partitionOps.lookup(curPartitionId).size()) {
    LDBG("Partition " << curPartitionId << " scheduled " << curSeenOps
                      << " before the next partition, not alternating.");
    return -1;
  }

  return getSingleTaskId(firstOp);
}

/// Process a WarpSpecializeOp to insert pingpong barriers for critical regions.
/// Finds ops with pingpong_id attributes, computes their boundaries, allocates
/// an mbarrier pair, derives wait phases from the enclosing loops, and inserts
/// mbarrier arrive/wait ops to enforce mutual exclusion between ping and pong
/// partitions.
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

    if (startOps.size() != 2 || endOps.size() != 2) {
      LDBG("pingpong ops are not in two partitions");
      continue;
    }

    for (auto [partitionId, startOp] : startOps) {
      // The start and end ops are unioned for each partition to find the
      // boundary ops
      Operation *unionStartOp = firstOpInBlock(startOp);
      Operation *unionEndOp = lastOpInBlock(endOps[partitionId]);

      // The pong partition goes first and ping waits
      if (partitionId != arrivesFirstPartitionId) {
        crManager.pingpongIdToPingBoundaryOps[pingpongId].push_back(
            unionStartOp);
        crManager.pingpongIdToPingBoundaryOps[pingpongId].push_back(unionEndOp);
      } else {
        crManager.pingpongIdToPongBoundaryOps[pingpongId].push_back(
            unionStartOp);
        crManager.pingpongIdToPongBoundaryOps[pingpongId].push_back(unionEndOp);
      }
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

    // Insert barriers for the ping partition
    Operation *pingStart = pingBoundOps[0];
    Operation *pingEnd = pingBoundOps[1];
    Region *pingRegion = getPartitionRegion(pingStart);
    if (!pingRegion) {
      LDBG("No region found for ping partition.");
      continue;
    }
    Block &pingRegionBlock = pingRegion->front();
    Operation *pongStart = pongBoundOps[0];
    Operation *pongEnd = pongBoundOps[1];
    Region *pongRegion = getPartitionRegion(pongStart);
    if (!pongRegion) {
      LDBG("No region found for pong partition.");
      continue;
    }

    // Both phases come from a loop-carried counter, so both sides need a
    // directly-nested loop chain inside their partition to carry it. Check
    // that before touching anything: everything below mutates the IR, and
    // bailing partway would leave a half-built rendezvous.
    SmallVector<scf::ForOp> pingNest = collectDirectLoopNest(pingStart);
    SmallVector<scf::ForOp> pongNest = collectDirectLoopNest(pongStart);
    if (pingNest.empty() || pongNest.empty()) {
      LDBG("No directly-nested enclosing loop for the ping-pong region.");
      continue;
    }
    // Exact counters on each side are not enough if one side executes fewer
    // iterations: the other side would eventually wait on a missing arrival.
    if (!ttg::haveEquivalentWarpSpecializeLoopNests(pingNest, pongNest)) {
      LDBG("Ping-pong partitions have different enclosing loop nests.");
      continue;
    }
    // Thread the counters first -- this rewrites the loops, and everything
    // below inserts into them.
    Value pingCounter = createIterationCounter(pingNest);
    Value pongCounter = createIterationCounter(pongNest);

    Value barrierAlloc = triton::createBarrierAlloc(wsOp, 2, 1);
    OpBuilder allocBuilder(wsOp);
    Value pingBarrier =
        triton::createSingleBufferView(allocBuilder, barrierAlloc, 0);
    Value pongBarrier =
        triton::createSingleBufferView(allocBuilder, barrierAlloc, 1);
    unsigned pingBarrierIdx = captureInPartitions(wsOp, pingBarrier);
    unsigned pongBarrierIdx = captureInPartitions(wsOp, pongBarrier);
    Value pingBarrierInPing = pingRegion->getArgument(pingBarrierIdx);
    Value pongBarrierInPing = pingRegion->getArgument(pongBarrierIdx);
    Value pingBarrierInPong = pongRegion->getArgument(pingBarrierIdx);
    Value pongBarrierInPong = pongRegion->getArgument(pongBarrierIdx);

    OpBuilder builder(&pingRegionBlock, pingRegionBlock.begin());
    auto pingRegionLoc = pingRegionBlock.front().getLoc();
    // Insert arrive barrier for the ping partition to allow the initial entry
    ttng::ArriveBarrierOp::create(builder, pingRegionLoc, pongBarrierInPing,
                                  1u);
    builder.setInsertionPoint(pingStart);
    Value pingPhase =
        getBarrierPhase(builder, pingStart->getLoc(), pingCounter);
    ttng::WaitBarrierOp::create(builder, pingStart->getLoc(), pingBarrierInPing,
                                pingPhase);
    // Insert AFTER the pingEnd op
    builder.setInsertionPointAfter(pingEnd);
    ttng::ArriveBarrierOp::create(builder, pingEnd->getLoc(), pongBarrierInPing,
                                  1u);

    // Insert barriers for the pong partition
    Block &pongRegionBlock = pongRegion->front();
    OpBuilder builder2(&pongRegionBlock, pongRegionBlock.begin());
    builder2.setInsertionPoint(pongStart);
    Value pongPhase =
        getBarrierPhase(builder2, pongStart->getLoc(), pongCounter);
    ttng::WaitBarrierOp::create(builder2, pongStart->getLoc(),
                                pongBarrierInPong, pongPhase);
    // Insert AFTER the pongEnd op
    builder2.setInsertionPointAfter(pongEnd);
    ttng::ArriveBarrierOp::create(builder2, pongEnd->getLoc(),
                                  pingBarrierInPong, 1u);
  }
}
} // anonymous namespace

/// doPingPongSync pass: Insert pingpong barriers to the IR
void doPingPongSync(triton::FuncOp funcOp, unsigned numWarpGroups,
                    int capability) {
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
void doPingPongPrep(triton::FuncOp funcOp, unsigned numWarpGroups,
                    int capability, int defaultNumStages) {

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
          // Reject grouping only if a real (non-expensive) memory side effect
          // sits strictly between the two ops. The endpoints themselves and any
          // peer expensive ops in between are part of the same ping-pong region
          // and must not split it (B-4 / T273470439).
          bool hasMemEffects =
              hasInterveningMemEffect(crManager, op, refOp, capability);
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

    // Ensure the schedule is available for this loop. scheduleLoops is a no-op
    // if the schedule is already complete.
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    int numStages = triton::getNumStagesOrDefault(forOp, defaultNumStages);
    triton::gpu::scheduleLoops(moduleOp, numStages, /*useMetaWS=*/true);

    triton::CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp))) {
      LDBG("Failed to deserialize schedule, skipping");
      continue;
    }

    // Find which partition arrives first and validate alternation pattern.
    // Returns -1 if the schedule is invalid (ops interleave or don't
    // alternate).
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
