#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/hopper/lib/Transforms/WarpSpecialization/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

// Safe wrapper around getPartitionIds that handles ops without partition attrs.
static SetVector<int> safeGetPartitionIds(Operation *op) {
  if (!op->hasAttr(kPartitionAttrName))
    return {};
  return getPartitionIds(op);
}

namespace ttng = triton::nvidia_gpu;

#define DEBUG_TYPE "tritongpu-partition-scheduling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
namespace {

inline bool isEpilogueStoreOp(Operation *op) {
  return isa<DescriptorStoreOp, ttng::AsyncTMACopyLocalToGlobalOp,
             ttng::TMAStoreTokenWaitOp>(op);
}

/// Check if an operation is an MMA-like operation (MMAv5 or WarpGroupDot).
/// Used for backward slice analysis and data partition detection.
inline bool isMMAOp(Operation *op) {
  return isa<ttng::MMAv5OpInterface>(op) || isa<ttng::WarpGroupDotOp>(op);
}

//===----------------------------------------------------------------------===//
// Op Categories and Scheduling Template Infrastructure
//===----------------------------------------------------------------------===//
//
// This section defines the categorization framework for partition scheduling.
// The goal is to categorize ops first, then apply templated scheduling rules.
// Currently this is used for analysis/logging only - the actual scheduling
// logic is unchanged.

/// Categories of operations for partition scheduling.
enum class OpCategory {
  Load,          // TMA loads
  MMA,           // MMA operations
  MemDescView,   // Memory descriptor views
  EpilogueStore, // Descriptor stores
  TMAReduction,  // TMA reduction operations
  DataPartition, // Ops exclusive to one MMA's slice
  Correction,    // Cross-iteration MMA users
  Default        // Everything else
};

/// Sentinel value for ops shared across multiple data partition groups.
static constexpr unsigned SHARED_DPID = UINT_MAX;

/// Get a string representation of an OpCategory.
static llvm::StringRef toString(OpCategory category) {
  switch (category) {
  case OpCategory::Load:
    return "Load";
  case OpCategory::MMA:
    return "MMA";
  case OpCategory::MemDescView:
    return "MemDescView";
  case OpCategory::EpilogueStore:
    return "EpilogueStore";
  case OpCategory::TMAReduction:
    return "TMAReduction";
  case OpCategory::Correction:
    return "Correction";
  case OpCategory::DataPartition:
    return "DataPartition";
  case OpCategory::Default:
    return "Default";
  }
  llvm_unreachable("Unknown OpCategory");
}

//===----------------------------------------------------------------------===//
// Data Partition Detection
//===----------------------------------------------------------------------===//

/// Collect backward slice for an MMA operation.
/// Enhanced to enter scf.if regions: when an scf.if op is in the slice,
/// follow yield operands in the then/else blocks backward. This captures
/// ops like tmem_load QK and mulf(QK*scale) in flex attention that feed
/// into scf.if yield operands but are missed by standard getBackwardSlice.
static SetVector<Operation *> collectMMABackwardSlice(scf::ForOp loop,
                                                      Operation *mmaOp) {
  SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  options.inclusive = false;
  options.filter = [&](Operation *op) {
    return loop->isAncestor(op) && !isMMAOp(op);
  };
  for (Value operand : mmaOp->getOperands()) {
    (void)getBackwardSlice(operand, &slice, options);
  }

  // Enter scf.if regions: follow yield operands backward until fixpoint.
  // getBackwardSlice adds scf.if ops to the slice but does NOT enter their
  // regions. Only follow yield operands that correspond to scf.if results
  // actually consumed by ops already in the slice. This prevents pulling in
  // ops from other data partitions (e.g., in flex attention, scf.if yields
  // values for both dp0 and dp1 — we only want the one used by this MMA).
  DenseSet<Operation *> visitedIfs;
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : llvm::to_vector(slice)) {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp || !visitedIfs.insert(ifOp).second)
        continue;
      // Find which scf.if results are actually used by ops in the slice.
      DenseSet<unsigned> usedResultIndices;
      for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
        for (Operation *user : ifOp.getResult(i).getUsers()) {
          if (slice.contains(user) ||
              llvm::any_of(mmaOp->getOperands(), [&](Value v) {
                return v.getDefiningOp() == user;
              })) {
            usedResultIndices.insert(i);
            break;
          }
        }
      }
      // Follow only the yield operands for used results.
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        if (region->empty())
          continue;
        auto *yieldOp = region->front().getTerminator();
        for (unsigned idx : usedResultIndices) {
          if (idx < yieldOp->getNumOperands()) {
            unsigned prevSize = slice.size();
            (void)getBackwardSlice(yieldOp->getOperand(idx), &slice, options);
            if (slice.size() > prevSize)
              changed = true;
          }
        }
      }
    }
  }

  return slice;
}

//===----------------------------------------------------------------------===//
// Debug Utilities
//==-----------------------------------------------------------------====//

/// Get the loop depth of an operation.
static unsigned getLoopDepth(Operation *op) {
  unsigned depth = 0;
  Operation *parent = op->getParentOp();
  while (parent) {
    if (isa<scf::ForOp, scf::WhileOp>(parent))
      depth++;
    parent = parent->getParentOp();
  }
  return depth;
}

/// Get a one-line pretty representation of an operation for debug printing.
/// Format: "op_name <shape> (depth=N)"
static std::string prettyOp(Operation *op) {
  std::string result;
  llvm::raw_string_ostream os(result);

  // Op name (short form without dialect prefix)
  StringRef opName = op->getName().getStringRef();
  size_t dotPos = opName.rfind('.');
  if (dotPos != StringRef::npos)
    os << opName.substr(dotPos + 1);
  else
    os << opName;

  // Result type info (shape + element type for tensors/memdescs)
  if (op->getNumResults() > 0) {
    SmallVector<std::string> typeStrs;
    for (Value r : op->getResults()) {
      Type ty = r.getType();
      std::string ts;
      llvm::raw_string_ostream tos(ts);
      if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
        tos << "<";
        for (unsigned d = 0; d < tensorTy.getRank(); d++) {
          if (d > 0)
            tos << "x";
          tos << tensorTy.getDimSize(d);
        }
        tos << "x" << tensorTy.getElementType() << ">";
      } else if (auto memDescTy = dyn_cast<MemDescType>(ty)) {
        tos << "<memdesc ";
        for (unsigned d = 0; d < memDescTy.getRank(); d++) {
          if (d > 0)
            tos << "x";
          tos << memDescTy.getShape()[d];
        }
        tos << ">";
      }
      if (!ts.empty())
        typeStrs.push_back(std::move(ts));
    }
    if (!typeStrs.empty()) {
      os << " ";
      for (unsigned i = 0; i < typeStrs.size(); i++) {
        if (i > 0)
          os << ", ";
        os << typeStrs[i];
      }
    }
  }

  os << " (depth=" << getLoopDepth(op) << ")";
  return result;
}

//===----------------------------------------------------------------------===//
// Scheduling Options and Partition Layout
//===----------------------------------------------------------------------===//
//
// Tuning knobs control how categories map to partitions.
// The partition layout is determined by the categorizer results + options.

/// Tuning knobs for partition scheduling.
struct SchedulingOptions {
  bool mergeCorrection = false;
  bool mergeEpilogue = false;
  bool mergeEpilogueToComputation = false;
  bool mergeReduction = false;
  bool separateEpilogueStore = false;
};

/// Holds all partition pointers created by createPartitionLayout.
struct PartitionLayout {
  Partition *correctionPartition = nullptr;
  Partition *reductionPartition = nullptr;
  Partition *gemmPartition = nullptr;
  Partition *loadPartition = nullptr;
  Partition *epiloguePartition = nullptr;
  Partition *epilogueStorePartition = nullptr;
  Partition *defaultPartition = nullptr; // computed alias
  SmallVector<Partition *, 2> computationPartitions;

  /// Fallback: correction -> reduction -> epilogue -> first computation.
  Partition *getDefaultPartition() const {
    if (correctionPartition)
      return correctionPartition;
    if (reductionPartition)
      return reductionPartition;
    if (epiloguePartition)
      return epiloguePartition;
    if (!computationPartitions.empty())
      return computationPartitions.back();
    return nullptr;
  }

  /// Get the opToDpId map (for schedulePostLoopOps).
  bool hasGemm() const { return gemmPartition != nullptr; }
};

//===----------------------------------------------------------------------===//
// OpCategorizer - Categorizes operations for scheduling
//===----------------------------------------------------------------------===//

/// Information about a categorized operation.
struct CategorizedOp {
  Operation *op;
  OpCategory category;
  unsigned dataPartitionId = 0;
  Operation *parentMMA = nullptr;
};

/// Categorizes operations in a loop for partition scheduling.
class OpCategorizer {
public:
  OpCategorizer(scf::ForOp mainLoop, ArrayRef<Operation *> mmaOps)
      : mainLoop(mainLoop), mmas(mmaOps.begin(), mmaOps.end()) {
    // Collect all loops (nested + main)
    for (auto nestedLoop : mainLoop.getOps<scf::ForOp>())
      loops.push_back(nestedLoop);
    loops.push_back(mainLoop);
  }

  /// Categorize all operations in the loop.
  void categorize() {
    collectMMABackwardSlices();
    categorizeLoads();
    categorizeMMAs();
    categorizeEpilogueStores();
    categorizeTMAReductions();
    categorizeCorrectionOps(); // Before DataPartition to prevent stealing
    categorizeDataPartitionOps();
  }

  /// Get operations in a specific category.
  SmallVector<CategorizedOp> getOpsInCategory(OpCategory cat) const {
    SmallVector<CategorizedOp> result;
    for (auto &[op, catOp] : opCategories) {
      if (catOp.category == cat)
        result.push_back(catOp);
    }
    return result;
  }

  /// Get the detected data partition factor.
  unsigned getDataPartitionFactor() const { return dataPartitionFactor; }

  /// Get all MMAs.
  ArrayRef<Operation *> getMMAs() const { return mmas; }

  /// Check if any MMAs are MMAv5 (Blackwell).
  bool hasMMAv5() const {
    return llvm::any_of(
        mmas, [](Operation *op) { return isa<ttng::MMAv5OpInterface>(op); });
  }

  /// Get the shared ops (ops appearing in multiple MMA backward slices).
  const DenseSet<Operation *> &getSharedOps() const { return sharedOps; }

  /// Get the dpId for an op. Returns SHARED_DPID if the op is shared across
  /// groups, or 0 if the op has no dpId assigned.
  unsigned getDpId(Operation *op) const {
    auto it = opToDpId.find(op);
    return it != opToDpId.end() ? it->second : 0;
  }

  const DenseMap<Operation *, unsigned> &getOpToDpIdMap() const {
    return opToDpId;
  }

  /// Pretty-print all categorized ops grouped by category.
  void printCategorizedOps(llvm::raw_ostream &os) const {
    os << "=== OpCategorizer Results ===\n";
    os << "  Loops: " << loops.size() << ", MMAs: " << mmas.size()
       << ", dpFactor: " << dataPartitionFactor << "\n";

    // Group ops by category in deterministic order
    constexpr OpCategory categoryOrder[] = {
        OpCategory::MMA,           OpCategory::Load,
        OpCategory::MemDescView,   OpCategory::EpilogueStore,
        OpCategory::TMAReduction,  OpCategory::Correction,
        OpCategory::DataPartition, OpCategory::Default};

    for (OpCategory cat : categoryOrder) {
      SmallVector<const CategorizedOp *> ops;
      for (auto &[op, catOp] : opCategories) {
        if (catOp.category == cat)
          ops.push_back(&catOp);
      }
      if (ops.empty())
        continue;

      os << "  [" << toString(cat) << "] (" << ops.size() << " ops):\n";
      for (const auto *catOp : ops) {
        os << "    " << prettyOp(catOp->op);
        if (catOp->dataPartitionId > 0 ||
            catOp->category == OpCategory::DataPartition)
          os << " [dp=" << catOp->dataPartitionId << "]";
        os << "\n";
      }
    }
  }

private:
  void collectMMABackwardSlices() {
    // Only process innermost loop's MMAs for data partitioning
    scf::ForOp innermostLoop = loops.empty() ? mainLoop : loops[0];

    SmallVector<Operation *> loopMmas;
    for (auto mmaOp : mmas) {
      if (mmaOp->getParentOp() == innermostLoop.getOperation())
        loopMmas.push_back(mmaOp);
    }

    if (loopMmas.size() < 2) {
      dataPartitionFactor = 1;
      return;
    }

    // Collect backward slice for each MMA
    for (auto mmaOp : loopMmas) {
      mmaToSlice[mmaOp] = collectMMABackwardSlice(innermostLoop, mmaOp);
    }

    // Find shared ops (appear in multiple slices)
    DenseMap<Operation *, unsigned> opCount;
    for (auto &[mma, slice] : mmaToSlice) {
      for (Operation *op : slice)
        opCount[op]++;
    }
    for (auto &[op, count] : opCount) {
      if (count > 1)
        sharedOps.insert(op);
    }

    // Group dependent MMAs using union-find.
    // MMA B depends on MMA A if A's result feeds (directly or via iter args
    // and intermediate ops) into B's operands.
    // Strategy: For each MMA, collect its forward user set (excluding other
    // MMAs). If that forward set overlaps with another MMA's backward slice,
    // they are dependent.
    unsigned n = loopMmas.size();
    SmallVector<unsigned> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<unsigned(unsigned)> find = [&](unsigned x) -> unsigned {
      return parent[x] == x ? x : parent[x] = find(parent[x]);
    };
    auto unite = [&](unsigned a, unsigned b) { parent[find(a)] = find(b); };

    // Build forward reachability from each MMA result (through iter args too)
    for (unsigned i = 0; i < n; ++i) {
      Operation *mmaOp = loopMmas[i];
      // Collect all ops reachable from this MMA's results
      DenseSet<Operation *> forwardSet;
      SmallVector<Value> worklist;
      for (Value result : mmaOp->getResults())
        worklist.push_back(result);

      // Also follow cross-iteration paths: MMA result → yield → iter arg
      for (OpOperand &use : mmaOp->getUses()) {
        if (use.getOwner() == innermostLoop.getBody()->getTerminator()) {
          worklist.push_back(
              innermostLoop.getRegionIterArg(use.getOperandNumber()));
        }
      }

      while (!worklist.empty()) {
        Value val = worklist.pop_back_val();
        for (Operation *user : val.getUsers()) {
          if (!innermostLoop->isAncestor(user))
            continue;
          if (isMMAOp(user))
            continue; // Don't traverse through other MMAs
          if (!forwardSet.insert(user).second)
            continue; // Already visited
          for (Value result : user->getResults())
            worklist.push_back(result);
        }
      }

      // Check if any other MMA's backward slice overlaps with this forward set
      for (unsigned j = 0; j < n; ++j) {
        if (i == j || find(i) == find(j))
          continue;
        auto &slice = mmaToSlice[loopMmas[j]];
        for (Operation *op : slice) {
          if (forwardSet.contains(op)) {
            unite(i, j);
            break;
          }
        }
      }
    }

    // Count distinct groups that have exclusive (non-shared) ops
    DenseSet<unsigned> groupsWithExclusiveOps;
    for (unsigned i = 0; i < n; ++i) {
      auto &slice = mmaToSlice[loopMmas[i]];
      for (Operation *op : slice) {
        if (!sharedOps.contains(op) && !isa<arith::ConstantOp>(op)) {
          groupsWithExclusiveOps.insert(find(i));
          break;
        }
      }
    }

    dataPartitionFactor =
        groupsWithExclusiveOps.size() > 1 ? groupsWithExclusiveOps.size() : 1;

    LLVM_DEBUG(llvm::dbgs() << "[data-partition] " << n << " MMAs → "
                            << groupsWithExclusiveOps.size()
                            << " independent groups (dpFactor="
                            << dataPartitionFactor << ")\n");

    // Build opToDpId map for ALL ops reachable from MMAs.
    // This is the single source of truth for data partition ID assignment.
    if (dataPartitionFactor > 1) {
      // Normalize group IDs to contiguous 0..dpFactor-1 range.
      DenseMap<unsigned, unsigned> rootToGroupId;
      unsigned nextGroupId = 0;
      for (unsigned i = 0; i < n; ++i) {
        unsigned root = find(i);
        if (!rootToGroupId.count(root))
          rootToGroupId[root] = nextGroupId++;
      }

      // Assign dpId to MMAs themselves.
      for (unsigned i = 0; i < n; ++i) {
        unsigned groupId = rootToGroupId[find(i)];
        opToDpId[loopMmas[i]] = groupId;
      }

      // Assign dpId to all backward slice ops.
      for (unsigned i = 0; i < n; ++i) {
        unsigned groupId = rootToGroupId[find(i)];
        for (Operation *op : mmaToSlice[loopMmas[i]]) {
          auto it = opToDpId.find(op);
          if (it == opToDpId.end()) {
            opToDpId[op] = groupId;
          } else if (it->second != groupId) {
            it->second = SHARED_DPID;
          }
        }
      }

      // Assign dpId to pre-loop ops: follow MMA operands backward across
      // the loop boundary. Ops defined outside the innermost loop that
      // feed exclusively into one MMA group get that group's dpId.
      for (unsigned i = 0; i < n; ++i) {
        unsigned groupId = rootToGroupId[find(i)];
        Operation *mmaOp = loopMmas[i];
        SmallVector<Operation *> worklist;
        for (Value operand : mmaOp->getOperands()) {
          if (auto *defOp = operand.getDefiningOp()) {
            if (!innermostLoop->isAncestor(defOp))
              worklist.push_back(defOp);
          }
        }
        // Also follow pre-loop ops from the backward slice.
        for (Operation *op : mmaToSlice[mmaOp]) {
          for (Value operand : op->getOperands()) {
            if (auto *defOp = operand.getDefiningOp()) {
              if (!innermostLoop->isAncestor(defOp) &&
                  mainLoop->isAncestor(defOp))
                worklist.push_back(defOp);
            }
          }
        }
        DenseSet<Operation *> visited;
        while (!worklist.empty()) {
          Operation *op = worklist.pop_back_val();
          if (!visited.insert(op).second)
            continue;
          auto it = opToDpId.find(op);
          if (it == opToDpId.end()) {
            opToDpId[op] = groupId;
          } else if (it->second != groupId) {
            it->second = SHARED_DPID;
          }
          for (Value operand : op->getOperands()) {
            if (auto *defOp = operand.getDefiningOp()) {
              if (mainLoop->isAncestor(defOp) &&
                  !innermostLoop->isAncestor(defOp))
                worklist.push_back(defOp);
            }
          }
        }
      }

      // Assign dpId to post-loop ops: follow loop results forward.
      // Each loop result traces back to a specific MMA group's yield.
      auto yieldOp = innermostLoop.getBody()->getTerminator();
      // Helper: find dpId for an in-loop op by walking backward through its
      // operand chain until we find an op in opToDpId. This handles ops like
      // l_i0 (softmax sum accumulation) that are not in any MMA's backward
      // slice but whose operands (e.g., alpha from the correction chain) are.
      auto findDpIdBackward = [&](Operation *startOp) -> unsigned {
        SmallVector<Operation *> bwWorklist;
        DenseSet<Operation *> bwVisited;
        bwWorklist.push_back(startOp);
        while (!bwWorklist.empty()) {
          Operation *op = bwWorklist.pop_back_val();
          if (!bwVisited.insert(op).second)
            continue;
          auto foundIt = opToDpId.find(op);
          if (foundIt != opToDpId.end())
            return foundIt->second;
          for (Value operand : op->getOperands())
            if (auto *defOp = operand.getDefiningOp())
              if (innermostLoop->isAncestor(defOp))
                bwWorklist.push_back(defOp);
        }
        return SHARED_DPID;
      };
      for (unsigned argIdx = 0; argIdx < innermostLoop.getNumResults();
           ++argIdx) {
        Value yieldVal = yieldOp->getOperand(argIdx);
        Operation *yieldDef = yieldVal.getDefiningOp();
        if (!yieldDef)
          continue;
        auto it = opToDpId.find(yieldDef);
        // If the yield def is not directly in opToDpId (e.g., softmax sum
        // accumulation ops that don't feed any MMA), walk backward through
        // its operand chain to find an ancestor with a known dpId.
        if (it == opToDpId.end()) {
          unsigned backwardDpId = findDpIdBackward(yieldDef);
          if (backwardDpId == SHARED_DPID)
            continue;
          opToDpId[yieldDef] = backwardDpId;
          it = opToDpId.find(yieldDef);
        }
        unsigned yieldDpId = it->second;
        if (yieldDpId == SHARED_DPID)
          continue;
        // Follow the loop result to post-loop consumers.
        Value loopResult = innermostLoop.getResult(argIdx);
        SmallVector<Operation *> postWorklist;
        for (Operation *user : loopResult.getUsers())
          postWorklist.push_back(user);
        DenseSet<Operation *> postVisited;
        while (!postWorklist.empty()) {
          Operation *op = postWorklist.pop_back_val();
          if (!postVisited.insert(op).second)
            continue;
          if (innermostLoop->isAncestor(op))
            continue;
          auto pit = opToDpId.find(op);
          if (pit == opToDpId.end()) {
            opToDpId[op] = yieldDpId;
          } else if (pit->second != yieldDpId) {
            pit->second = SHARED_DPID;
          }
          for (Value result : op->getResults())
            for (Operation *user : result.getUsers())
              postWorklist.push_back(user);
        }
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[data-partition] opToDpId map (" << opToDpId.size()
                     << " entries):\n";
        unsigned sharedCount = 0;
        for (auto &[op, dpId] : opToDpId) {
          if (dpId == SHARED_DPID)
            sharedCount++;
        }
        llvm::dbgs() << "  shared ops: " << sharedCount << "\n";
      });
    }
  }

  void categorizeLoads() {
    for (auto loop : loops) {
      for (Operation &op : loop.getOps()) {
        if (!isa<DescriptorLoadOp, DescriptorGatherOp>(op))
          continue;

        addCategorizedOp(&op, OpCategory::Load);
      }
    }
  }

  void categorizeMMAs() {
    for (auto mmaOp : mmas) {
      unsigned dpId = getDpId(mmaOp);
      addCategorizedOp(mmaOp, OpCategory::MMA, dpId, mmaOp);

      // Categorize memory descriptor views feeding into MMA
      SmallVector<Operation *> worklist;
      for (Value operand : mmaOp->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp())
          worklist.push_back(defOp);
      }
      while (!worklist.empty()) {
        Operation *op = worklist.pop_back_val();
        if (!op->hasTrait<OpTrait::MemDescViewTrait>())
          continue;
        if (opCategories.contains(op))
          continue;
        addCategorizedOp(op, OpCategory::MemDescView, getDpId(op), mmaOp);
        if (Operation *defOp = op->getOperand(0).getDefiningOp())
          worklist.push_back(defOp);
      }
    }
  }

  void categorizeEpilogueStores() {
    // Collect stores inside the loops.
    for (auto loop : loops) {
      loop.walk([&](Operation *op) {
        if (isEpilogueStoreOp(op))
          addCategorizedOp(op, OpCategory::EpilogueStore);
      });
    }
    // Also collect stores AFTER the main loop in the parent block (e.g., bwd
    // epilogue stores that write gradients after the loop completes).
    Operation *loopOp = mainLoop.getOperation();
    Block *parentBlock = loopOp->getBlock();
    if (!parentBlock)
      return;
    bool afterLoop = false;
    for (Operation &op : *parentBlock) {
      if (&op == loopOp) {
        afterLoop = true;
        continue;
      }
      if (afterLoop && isEpilogueStoreOp(&op)) {
        addCategorizedOp(&op, OpCategory::EpilogueStore);
      }
    }
  }

  void categorizeDataPartitionOps() {
    if (dataPartitionFactor <= 1)
      return;

    // Map exclusive ops to their MMA group's dpId using opToDpId.
    for (auto &[mma, slice] : mmaToSlice) {
      for (Operation *op : slice) {
        if (!sharedOps.contains(op) && !opCategories.contains(op) &&
            !isa<arith::ConstantOp>(op)) {
          unsigned dpId = getDpId(op);
          if (dpId != SHARED_DPID)
            addCategorizedOp(op, OpCategory::DataPartition, dpId, mma);
        }
      }
    }
  }

  void categorizeCorrectionOps() {
    for (auto mmaOp : mmas) {
      scf::ForOp loop = mmaOp->getParentOfType<scf::ForOp>();
      unsigned dpId = getDpId(mmaOp);
      for (OpOperand &use : mmaOp->getUses()) {
        if (use.getOwner() != loop.getBody()->getTerminator())
          continue;
        // MMA result is yielded - find users in next iteration
        for (OpOperand &iterUse :
             loop.getRegionIterArg(use.getOperandNumber()).getUses()) {
          Operation *user = iterUse.getOwner();
          if (!opCategories.contains(user)) {
            addCategorizedOp(user, OpCategory::Correction, dpId, mmaOp);
          }
        }
        break;
      }
    }
  }

  /// Categorize TMA reduction operations (descriptor_reduce and
  /// async_tma_reduce).
  void categorizeTMAReductions() {
    auto isReductionOp = [](Operation *op) {
      return isa<DescriptorReduceOp, ttng::AsyncTMAReduceOp>(op);
    };
    for (scf::ForOp loop : loops) {
      loop.walk([&](Operation *op) {
        if (isReductionOp(op))
          addCategorizedOp(op, OpCategory::TMAReduction);
      });
    }
    // Also check the main loop if not in loops
    if (loops.empty()) {
      mainLoop.walk([&](Operation *op) {
        if (isReductionOp(op))
          addCategorizedOp(op, OpCategory::TMAReduction);
      });
    }
  }

  void addCategorizedOp(Operation *op, OpCategory cat,
                        unsigned dataPartitionId = 0,
                        Operation *parentMMA = nullptr) {
    // If no explicit dpId provided, look up from opToDpId map.
    if (dataPartitionId == 0) {
      auto it = opToDpId.find(op);
      if (it != opToDpId.end() && it->second != SHARED_DPID)
        dataPartitionId = it->second;
    }
    opCategories[op] = CategorizedOp{op, cat, dataPartitionId, parentMMA};
  }

  scf::ForOp mainLoop;
  SmallVector<scf::ForOp> loops;
  SmallVector<Operation *> mmas;
  DenseMap<Operation *, CategorizedOp> opCategories;
  DenseMap<Operation *, SetVector<Operation *>> mmaToSlice;
  DenseSet<Operation *> sharedOps;
  DenseMap<Operation *, unsigned> opToDpId;
  unsigned dataPartitionFactor = 1;
};

/// Create partitions based on the categorizer results and scheduling options.
/// This replaces the old template system (UnifiedFATemplate, GEMMTemplate,
/// selectTemplate).
static PartitionLayout createPartitionLayout(PartitionSet &schedule,
                                             const OpCategorizer &categorizer,
                                             const SchedulingOptions &options) {
  PartitionLayout layout;
  unsigned dpFactor = categorizer.getDataPartitionFactor();
  bool hasCorrection =
      !categorizer.getOpsInCategory(OpCategory::Correction).empty();
  bool hasReduction =
      !categorizer.getOpsInCategory(OpCategory::TMAReduction).empty();
  bool hasEpilogue =
      !categorizer.getOpsInCategory(OpCategory::EpilogueStore).empty();
  bool hasMMAv5 = categorizer.hasMMAv5();

  LLVM_DEBUG(llvm::dbgs() << "[partition-layout] dpFactor=" << dpFactor
                          << ", hasCorrection=" << hasCorrection
                          << ", hasReduction=" << hasReduction
                          << ", hasEpilogue=" << hasEpilogue
                          << ", hasMMAv5=" << hasMMAv5 << "\n");

  // Correction partition: needed when we have correction ops and not merging.
  if (hasCorrection && !options.mergeCorrection) {
    layout.correctionPartition = schedule.addPartition(0);
    layout.correctionPartition->setType("correction");
  }

  // Reduction partition: for bwd.
  if (hasReduction && !options.mergeReduction) {
    layout.reductionPartition = schedule.addPartition(0);
    layout.reductionPartition->setType("reduction");
  }

  // Gemm partition: only when MMAv5 ops exist.
  if (hasMMAv5) {
    layout.gemmPartition = schedule.addPartition(1);
    layout.gemmPartition->setType("gemm");
  }

  // Epilogue partition: for non-store epilogue ops when not merging.
  if (hasEpilogue && !options.mergeEpilogue &&
      !options.mergeEpilogueToComputation) {
    layout.epiloguePartition = schedule.addPartition(0);
    layout.epiloguePartition->setType("epilogue");
  }

  // Epilogue store partition: dedicated 1-warp partition for epilogue stores.
  if (options.separateEpilogueStore && hasEpilogue) {
    layout.epilogueStorePartition = schedule.addPartition(0);
    layout.epilogueStorePartition->setType("epilogue_store");
  }

  // Load partition: always created last so it gets the highest partition index,
  // which maps to the default (producer) warp group at runtime.
  layout.loadPartition = schedule.addPartition(0);
  layout.loadPartition->setType("load");

  // Set default partition alias using fallback chain.
  layout.defaultPartition = layout.getDefaultPartition();

  LLVM_DEBUG({
    llvm::dbgs() << "[partition-layout] Created partitions:";
    for (Partition &p : schedule.getPartitions())
      llvm::dbgs() << " " << p.getType() << "(" << p.getIndex() << ")";
    llvm::dbgs() << "\n";
  });

  return layout;
}

} // namespace

//===----------------------------------------------------------------------===//
// assignPartitions
//===----------------------------------------------------------------------===//

// Find the last operation in the loop body that defined this value, with a
// maximum of distance 1.
static Operation *findDefOpInLoop(scf::ForOp loop, Value value,
                                  int distance = 0) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getParentBlock() != loop.getBody())
      return {};
    // Don't look back more than distance 1.
    if (distance == 1)
      return {};
    return findDefOpInLoop(
        loop, loop.getYieldedValues()[arg.getArgNumber() - 1], distance + 1);
  }
  Operation *defOp = value.getDefiningOp();
  if (!loop.getBodyRegion().isAncestor(defOp->getParentRegion()))
    return {};
  return defOp;
}

// For `op`, invoke `callback` on all the definitions of its inputs from within
// `loop`, which might not be in the same iteration.
static void iterateDefs(scf::ForOp loop, Operation *op,
                        function_ref<void(OpResult)> callback) {
  visitNestedOperands(op, [&](OpOperand &operand) {
    Value value = operand.get();
    if (value.getParentBlock() != loop.getBody())
      return;
    auto arg = dyn_cast<BlockArgument>(value);
    if (arg == loop.getInductionVar())
      return;
    auto [def, distance] = getDefinitionAndDistance(loop, operand.get());
    if (def && def.getParentBlock() == loop.getBody())
      callback(def);
  });
}

// For `op`, invoke `callback` on all its transitive users within `loop`, which
// may be in a future iteration.
static void iterateUsers(scf::ForOp loop, Operation *op,
                         function_ref<void(Operation *)> callback) {
  SmallVector<OpOperand *> uses;
  DenseSet<OpOperand *> visited;
  for (OpOperand &use : op->getUses())
    uses.push_back(&use);
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    if (!visited.insert(use).second)
      continue;
    Operation *owner = loop.getBody()->findAncestorOpInBlock(*use->getOwner());
    if (auto nestedFor = dyn_cast<scf::ForOp>(owner)) {
      // For captured values used inside nested loops, walk the use
      // chain inside the loop to find partitioned consumers.
      SmallVector<Operation *> innerWorklist;
      DenseSet<Operation *> innerVisited;
      for (OpOperand &innerUse : use->get().getUses())
        if (nestedFor->isAncestor(innerUse.getOwner()))
          innerWorklist.push_back(innerUse.getOwner());
      while (!innerWorklist.empty()) {
        Operation *innerOp = innerWorklist.pop_back_val();
        if (!innerVisited.insert(innerOp).second)
          continue;
        if (hasPartition(innerOp)) {
          callback(innerOp);
        } else {
          for (Value result : innerOp->getResults())
            for (OpOperand &u : result.getUses())
              if (nestedFor->isAncestor(u.getOwner()))
                innerWorklist.push_back(u.getOwner());
        }
      }
      continue;
    }
    if (!isa<scf::YieldOp>(owner)) {
      callback(owner);
      continue;
    }
    BlockArgument arg = loop.getRegionIterArg(use->getOperandNumber());
    for (OpOperand &use : arg.getUses())
      uses.emplace_back(&use);
  }
}

// Helper: schedule an operation to a partition if it is not already scheduled.
static bool tryScheduleOp(Partition *partition, Operation *op) {
  if (hasPartition(op))
    return false;
  setPartition(op, partition);
  return true;
}

// Check if any of the inputs to `op` are reachable from a non-null partition.
static bool hasDefPartition(scf::ForOp loop, Operation *op,
                            PartitionSet &schedule) {
  SmallVector<Operation *> worklist{op};
  DenseSet<Operation *> seen;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!seen.insert(op).second)
      continue;
    if (hasPartition(op))
      return true;
    iterateDefs(loop, op,
                [&](OpResult def) { worklist.push_back(def.getDefiningOp()); });
  }
  return false;
}

// Recursively schedule the users of an operation, stopping when
// encountering an operation that is already assigned.
// If \p partition is null, a new partition will be created if needed.
static Partition *scheduleUsers(scf::ForOp loop, PartitionSet &schedule,
                                Partition *partition, Operation *op) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : op->getUses())
    uses.push_back(&use);
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    Operation *user = loop.getBody()->findAncestorOpInBlock(*use->getOwner());

    if (user == loop.getBody()->getTerminator()) {
      for (OpOperand &use :
           loop.getRegionIterArg(use->getOperandNumber()).getUses())
        uses.push_back(&use);
      continue;
    }

    if (hasPartition(user))
      continue;
    if (!partition) {
      partition = schedule.addPartition(/* stage is unused */ 0);
      partition->setType("computation");
    }
    tryScheduleOp(partition, user);
    for (OpOperand &use : user->getUses())
      uses.push_back(&use);
  }
  return partition;
}

// Schedule post-loop operations (operations outside and after the loop) into
// the appropriate partition. Epilogue store ops and their transitive users
// (e.g., TMAStoreTokenWaitOp) go to the epilogue partition. All other post-loop
// ops (e.g., tmem_load for accumulator reads, arithmetic for normalization) go
// to the default partition. This prevents TMEM ops from landing in the
// epilogue, which would force it to use 4 warps (TMEM lane coverage hardware

static void
schedulePostLoopOps(scf::ForOp loop, PartitionSet &schedule,
                    const PartitionLayout &layout,
                    const SchedulingOptions &options,
                    const DenseMap<Operation *, unsigned> &opToDpId,
                    const DenseMap<unsigned, Partition *> &dpIdToPartition) {
  auto findDpId = [&](Operation *op) -> unsigned {
    auto it = opToDpId.find(op);
    if (it != opToDpId.end())
      return it->second;
    for (Value operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        auto defIt = opToDpId.find(defOp);
        if (defIt != opToDpId.end())
          return defIt->second;
      }
    }
    return SHARED_DPID;
  };

  // Deterministic fallback: pick the partition with the smallest dpId key.
  // DenseMap iteration order is non-deterministic, so .begin() can return
  // different entries across builds. Use min_element on the key instead.
  auto dpIdFallbackPartition = [&]() -> Partition * {
    if (dpIdToPartition.empty())
      return nullptr;
    auto minIt = std::min_element(
        dpIdToPartition.begin(), dpIdToPartition.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });
    return minIt->second;
  };

  auto getEpilogueTarget = [&](Operation *op) -> Partition * {
    if (options.mergeEpilogueToComputation) {
      unsigned dpId = findDpId(op);
      if (dpId != SHARED_DPID) {
        auto it = dpIdToPartition.find(dpId);
        if (it != dpIdToPartition.end())
          return it->second;
      }
      if (auto *p = dpIdFallbackPartition())
        return p;
    }
    if (options.mergeEpilogue) {
      if (layout.correctionPartition)
        return layout.correctionPartition;
      if (layout.reductionPartition)
        return layout.reductionPartition;
      // When no correction/reduction partition exists (e.g., mergeCorrection +
      // mergeEpilogue on Hopper), route epilogue ops to their dpId-based
      // computation partition so each data partition's epilogue stays local.
      unsigned dpId = findDpId(op);
      if (dpId != SHARED_DPID) {
        auto it = dpIdToPartition.find(dpId);
        if (it != dpIdToPartition.end())
          return it->second;
      }
      if (auto *p = dpIdFallbackPartition())
        return p;
    }
    if (layout.epiloguePartition)
      return layout.epiloguePartition;
    return layout.defaultPartition;
  };

  auto getStoreTarget = [&](Operation *op) -> Partition * {
    if (layout.epilogueStorePartition)
      return layout.epilogueStorePartition;
    return getEpilogueTarget(op);
  };

  SmallVector<OpOperand *> uses;
  // For persistent kernels, seed from nested inner loop results.
  for (auto &op : loop.getOps())
    if (auto innerLoop = dyn_cast<scf::ForOp>(op))
      for (OpResult result : innerLoop.getResults())
        for (OpOperand &use : result.getUses())
          uses.push_back(&use);
  for (OpResult result : loop.getResults())
    for (OpOperand &use : result.getUses())
      uses.push_back(&use);

  DenseSet<Operation *> visited;
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    Operation *user = use->getOwner();
    if (!visited.insert(user).second)
      continue;
    // Skip ops inside nested inner loops. Ops directly in the ws-loop
    // body (post-inner-loop) or outside the ws-loop are processed.
    if (auto parentLoop = user->getParentOfType<scf::ForOp>())
      if (parentLoop != loop)
        continue;

    { // Schedule post-loop op (override earlier phase assignments)
      Partition *target = nullptr;
      if (isEpilogueStoreOp(user)) {
        target = getStoreTarget(user);
      } else {
        bool hasStoreInput = false;
        if (layout.epilogueStorePartition) {
          hasStoreInput = llvm::any_of(user->getOperands(), [&](Value v) {
            if (auto *defOp = v.getDefiningOp()) {
              auto ids = safeGetPartitionIds(defOp);
              return !ids.empty() &&
                     llvm::is_contained(
                         ids, layout.epilogueStorePartition->getIndex());
            }
            return false;
          });
        }
        if (hasStoreInput)
          target = layout.epilogueStorePartition;
        else
          target = getEpilogueTarget(user);
      }
      if (target)
        setPartition(user, target);
    }

    for (OpResult result : user->getResults())
      for (OpOperand &nextUse : result.getUses())
        uses.push_back(&nextUse);
  }
}
// Result of getInitialSchedule.
struct ScheduleResult {
  PartitionSet schedule;
  PartitionLayout layout;
  SchedulingOptions options;
  DenseMap<Operation *, unsigned> opToDpId;
  DenseMap<unsigned, Partition *> dpIdToPartition;
  bool createComputePartitions;
};

// Given a partitioning scheme, determine an initial schedule by performing a
// first-order partition assignment to the operations in the scheme and its
// users and/or dependencies. This sets up the initial partitioning of the ops.
static std::optional<ScheduleResult>
getInitialSchedule(scf::ForOp mainLoop, const SchedulingOptions &schedOpts) {
  // Check for an existing schedule.
  if (FailureOr<PartitionSet> scheduleOr = PartitionSet::fromLoop(mainLoop);
      succeeded(scheduleOr))
    // Deserialized schedule: layout/options unknown, use defaults.
    return ScheduleResult{std::move(*scheduleOr),
                          PartitionLayout{},
                          schedOpts,
                          DenseMap<Operation *, unsigned>(),
                          DenseMap<unsigned, Partition *>(),
                          /*createComputePartitions=*/true};

  // Collect all loops (nested + main)
  SmallVector<scf::ForOp> loops{mainLoop.getOps<scf::ForOp>()};
  loops.push_back(mainLoop);

  // Collect all MMAs
  SmallVector<Operation *> mmas;
  for (auto loop : loops) {
    for (auto &op : loop.getOps()) {
      if (isMMAOp(&op))
        mmas.push_back(&op);
    }
  }

  //===--------------------------------------------------------------------===//
  // Phase 1: Categorize all operations using OpCategorizer
  //===--------------------------------------------------------------------===//
  OpCategorizer categorizer(mainLoop, mmas);
  categorizer.categorize();

  LLVM_DEBUG(categorizer.printCategorizedOps(llvm::dbgs()));

  unsigned dataPartitionFactor = categorizer.getDataPartitionFactor();
  LLVM_DEBUG(
      llvm::dbgs() << "[tritongpu-partition-scheduling] Using template-based "
                      "scheduling with data partition factor: "
                   << dataPartitionFactor << "\n");

  //===--------------------------------------------------------------------===//
  // Phase 2: Create partition layout using tuning knobs
  //===--------------------------------------------------------------------===//
  PartitionSet schedule;
  PartitionLayout layout =
      createPartitionLayout(schedule, categorizer, schedOpts);

  // Extract partition references from layout
  Partition *defaultPartition = layout.defaultPartition;
  Partition *mmaPartition = layout.gemmPartition;
  Partition *loadPartition = layout.loadPartition;
  Partition *epiloguePartition = layout.epiloguePartition;
  Partition *correctionPartition = layout.correctionPartition;
  Partition *reductionPartition = layout.reductionPartition;

  // For backward compatibility: use default as fallback
  if (!correctionPartition)
    correctionPartition = defaultPartition;
  if (!reductionPartition)
    reductionPartition = defaultPartition;

  //===--------------------------------------------------------------------===//
  // Phase 3: Schedule ops using template-based partition assignment
  //===--------------------------------------------------------------------===//

  // Schedule loads and their associated allocs (both in-loop and pre-loop)
  SmallVector<Operation *> loadsAndAllocs;

  // Pre-loop descriptor_loads (e.g., k and v loads in bwd attention)
  if (!loops.empty()) {
    Operation *loopOp = loops[0].getOperation();
    for (Operation &op : *loopOp->getBlock()) {
      if (&op == loopOp)
        break; // Stop at the loop itself.
      if (!isa<DescriptorLoadOp, DescriptorGatherOp>(op))
        continue;
      tryScheduleOp(loadPartition, &op);
      loadsAndAllocs.push_back(&op);
      // Local alloc users of the load with matching encoding
      SharedEncodingTrait sharedEnc = getSharedEncoding(&op);
      for (Operation *user : op.getUsers()) {
        if (auto alloc = dyn_cast<LocalAllocOp>(user)) {
          if (sharedEnc == alloc.getType().getEncoding()) {
            tryScheduleOp(loadPartition, alloc);
            loadsAndAllocs.push_back(alloc);
          }
        } else if (isa<ttng::TMEMAllocOp>(user)) {
          tryScheduleOp(loadPartition, user);
          loadsAndAllocs.push_back(user);
        }
      }
    }

    // For BWD (hasReduction): tag pre-loop TMEMStoreOp with the reduction
    // partition index. These ops initialize accumulators (e.g., zeroing dK/dV)
    // before the loop. Without explicit assignment, they would get pulled
    // into the gemm partition via token chains to the in-loop MMA, causing
    // gemm to require >=4 warps (TMEM ops need 4 warps).
    // We set the attribute directly rather than using schedule.trySchedule
    // because pre-loop ops must not be added to the partition's ops list
    // (optimizeSchedule only handles in-loop ops).
    if (reductionPartition != nullptr) {
      for (Operation &op : *loopOp->getBlock()) {
        if (&op == loopOp)
          break;
        if (isa<ttng::TMEMStoreOp>(op))
          setPartition(&op, ArrayRef<int>{static_cast<int>(
                                reductionPartition->getIndex())});
      }
    }
  }

  // In-loop loads
  for (auto loop : loops) {
    for (Operation &op : loop.getOps()) {
      if (!isa<DescriptorLoadOp, DescriptorGatherOp>(op))
        continue;
      tryScheduleOp(loadPartition, &op);
      loadsAndAllocs.push_back(&op);

      // Local alloc users of the load with matching encoding
      SharedEncodingTrait sharedEnc = getSharedEncoding(&op);
      for (Operation *user : op.getUsers()) {
        if (auto alloc = dyn_cast<LocalAllocOp>(user)) {
          if (sharedEnc == alloc.getType().getEncoding()) {
            tryScheduleOp(loadPartition, alloc);
            loadsAndAllocs.push_back(alloc);
          }
        } else if (isa<ttng::TMEMAllocOp>(user)) {
          tryScheduleOp(loadPartition, user);
          loadsAndAllocs.push_back(user);
        }
      }
    }
  }

  // Schedule epilogue stores (both inside loops AND post-loop stores)
  // Also schedule the backward slice of post-loop epilogue stores (tmem_load,
  // truncf, etc.)
  if (epiloguePartition) {
    // Stores inside loops (both pre-lowering DescriptorStoreOp and
    // post-lowering AsyncTMACopyLocalToGlobalOp)
    for (auto loop : loops) {
      loop.walk([&](Operation *op) {
        if (isEpilogueStoreOp(op))
          tryScheduleOp(epiloguePartition, op);
      });
    }

    // Also schedule categorized epilogue stores (includes post-loop stores for
    // bwd) and their backward slice (tmem_load, truncf that feed into them)
    auto epilogueStoreOps =
        categorizer.getOpsInCategory(OpCategory::EpilogueStore);
    for (const auto &catOp : epilogueStoreOps) {
      tryScheduleOp(epiloguePartition, catOp.op);

      // Only schedule backward slice for post-loop stores (not inside any loop)
      // This captures ops like tmem_load, truncf that prepare data for storing
      bool isPostLoop = !catOp.op->getParentOfType<scf::ForOp>();
      if (isPostLoop) {
        SetVector<Operation *> slice;
        BackwardSliceOptions options;
        options.omitBlockArguments = true;
        // Only include ops in the same block AND that are not loops or
        // scheduled
        options.filter = [&](Operation *op) {
          // Must be in the same block as the store (post-loop region)
          if (op->getBlock() != catOp.op->getBlock())
            return false;
          // Skip scf.for and other control flow - we only want data-producing
          // ops
          if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op))
            return false;
          // Skip ops that are already scheduled
          if (hasPartition(op))
            return false;
          return true;
        };
        (void)getBackwardSlice(catOp.op, &slice, options);
        for (Operation *op : slice) {
          // Skip constants - they can be shared across partitions
          if (isa<arith::ConstantOp>(op))
            continue;
          tryScheduleOp(epiloguePartition, op);
        }
      }
    }

    // Schedule regular StoreOps to epilogue only when the epilogue partition
    // is otherwise empty (no DescriptorStoreOps or categorized epilogue stores
    // were scheduled above). When epilogue already has stores (e.g., FA kernels
    // with TMA output stores), additional StoreOps should stay in the
    // computation partition to avoid cross-partition TMEM overhead.
    if (epiloguePartition->getOps().empty()) {
      for (auto loop : loops)
        for (StoreOp op : loop.getOps<StoreOp>())
          tryScheduleOp(epiloguePartition, op);
    }
  }

  // Schedule MMAs and their associated stores
  for (auto loop : loops) {
    for (auto &op : loop.getOps()) {
      if (!isMMAOp(&op))
        continue;
      if (mmaPartition)
        tryScheduleOp(mmaPartition, &op);

      // For MMAv5: if the store is unrelated to the use of the MMA, place
      // in MMA partition. Exception: in BWD (hasReduction), keep TMEMStoreOp
      // out of the gemm partition so that gemm can run with fewer warps.
      if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(&op)) {
        auto storeOp = dyn_cast_or_null<ttng::TMEMStoreOp>(
            findDefOpInLoop(loop, mmaOp.getAccDep()));
        if (mmaPartition && reductionPartition == nullptr &&
            !ttng::hasAccReadModifyWrite(mmaOp, loop) && storeOp &&
            loop.isDefinedOutsideOfLoop(storeOp.getSrc()))
          tryScheduleOp(mmaPartition, storeOp);
      }
    }
  }

  // Schedule memory descriptor views feeding into MMAs (MMAv5 only —
  // memdesc views are a Blackwell TMEM concept, not used on Hopper).
  if (mmaPartition) {
    for (auto loop : loops) {
      for (auto &mmaOp : loop.getOps()) {
        if (!isMMAOp(&mmaOp))
          continue;
        SmallVector<Operation *> operandViews;
        for (Value operand : mmaOp.getOperands()) {
          if (Operation *defOp = operand.getDefiningOp())
            operandViews.push_back(defOp);
        }
        while (!operandViews.empty()) {
          Operation *op = operandViews.pop_back_val();
          if (!op->hasTrait<OpTrait::MemDescViewTrait>())
            continue;

          // Duplicate the op if necessary to ensure MMA partition is only user
          if (!llvm::all_of(op->getUsers(), [&](Operation *user) {
                auto ids = safeGetPartitionIds(user);
                return !ids.empty() && ids.contains(mmaPartition->getIndex());
              })) {
            Operation *newOp = OpBuilder(op).clone(*op);
            op->replaceUsesWithIf(newOp->getResults(), [&](OpOperand &use) {
              auto ids = safeGetPartitionIds(use.getOwner());
              return !ids.empty() && ids.contains(mmaPartition->getIndex());
            });
            op = newOp;
          }

          tryScheduleOp(mmaPartition, op);
          if (Operation *defOp = op->getOperand(0).getDefiningOp())
            operandViews.push_back(defOp);
        }
      }
    }
  } // if (mmaPartition)

  // If there are no loads or MMAs, don't warp specialize.
  if (loadsAndAllocs.empty() && mmas.empty())
    return std::nullopt;

  //===--------------------------------------------------------------------===//
  // Phase 4: Propagate users (load users, correction, reductions)
  //===--------------------------------------------------------------------====//

  // Load users go to default partition (shared computation).
  // When default is absent (e.g., bwd), skip — MMA user propagation in
  // Phase 5 will capture these ops through the use chain.
  if (defaultPartition) {
    for (Operation *loadOrAlloc : loadsAndAllocs) {
      scf::ForOp parentLoop = loadOrAlloc->getParentOfType<scf::ForOp>();
      if (!parentLoop) {
        // Skip pre-loop ops that don't have a parent loop
        continue;
      }
      scheduleUsers(parentLoop, schedule, defaultPartition, loadOrAlloc);
    }
  }

  // Correction ops (cross-iteration MMA users) go to correction partition
  // (which is aliased to default for fwd).
  // Skip entirely when no correction partition is available.
  Partition *corrDest =
      correctionPartition ? correctionPartition : defaultPartition;
  if (corrDest) {
    for (auto mmaOp : mmas) {
      for (OpOperand &use : mmaOp->getUses()) {
        auto loop = mmaOp->getParentOfType<scf::ForOp>();
        if (use.getOwner() != loop.getBody()->getTerminator())
          continue;
        for (OpOperand &use :
             loop.getRegionIterArg(use.getOperandNumber()).getUses()) {
          tryScheduleOp(corrDest, use.getOwner());
          scheduleUsers(loop, schedule, corrDest, use.getOwner());
        }
        break;
      }
    }
  }

  // TMA reduction ops go to reduction partition, along with their producers
  // (e.g., tmem_load, mulf that compute the value being reduced).
  Partition *reductionDest =
      reductionPartition ? reductionPartition : defaultPartition;
  auto tmaReductionOps = categorizer.getOpsInCategory(OpCategory::TMAReduction);
  for (const auto &catOp : tmaReductionOps) {
    tryScheduleOp(reductionDest, catOp.op);
    // Also schedule the backward slice (producers) of the reduction value.
    // The reduction op typically has operands: descriptor, indices, value.
    // We want to schedule the ops that produce the value being reduced.
    for (Value operand : catOp.op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        // Walk backward through the def chain to schedule producers.
        SmallVector<Operation *> worklist;
        worklist.push_back(defOp);
        DenseSet<Operation *> visited;
        while (!worklist.empty()) {
          Operation *op = worklist.pop_back_val();
          if (!visited.insert(op).second)
            continue;
          // Skip ops that are already scheduled to a different partition
          // (like MMA ops in gemm partition).
          if (hasPartition(op))
            continue;
          // Skip ops outside the loop.
          if (!op->getParentOfType<scf::ForOp>())
            continue;
          tryScheduleOp(reductionDest, op);
          // Add operand definitions to worklist.
          for (Value opnd : op->getOperands()) {
            if (Operation *def = opnd.getDefiningOp())
              worklist.push_back(def);
          }
        }
      }
    }

    // For BWD (hasReduction): tag pre-loop TMEMStoreOp with the reduction
    // partition index. These ops initialize accumulators (e.g., zeroing
    // dK/dV) before the loop. Without explicit assignment, they would get
    // pulled into the gemm partition via token chains to the in-loop MMA,
    // causing gemm to require >=4 warps (TMEM ops need 4 warps). We set the
    // attribute directly rather than using schedule.trySchedule because
    // pre-loop ops must not be added to the partition's ops list
    // (optimizeSchedule only handles in-loop ops).
    if (reductionPartition) {
      Builder b(mainLoop->getContext());
      for (Operation &op : *mainLoop->getBlock()) {
        if (&op == mainLoop)
          break;
        if (isa<ttng::TMEMStoreOp>(op))
          op.setAttr(kPartitionAttrName,
                     b.getDenseI32ArrayAttr({reductionPartition->getIndex()}));
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Phase 5: Create per-MMA computation partitions
  //===--------------------------------------------------------------------===//
  // MMA users create computation partitions. This runs AFTER correction/load
  // user propagation so that shared ops are already claimed, leaving only
  // per-MMA-exclusive ops for the computation partitions.
  //
  // When dpFactor > 1 (fwd): each independent MMA group gets its own
  //   dynamic partition via scheduleUsers(nullptr).
  // When dpFactor == 1 (bwd): all MMA users share a single computation
  //   partition to avoid creating too many partitions.

  DenseMap<Operation *, Partition *> mmaToPartition;
  SmallVector<Operation *> inFirstLoop;

  // For dpFactor==1, pre-create a single shared computation partition.
  // For dpFactor>1, let scheduleUsers(nullptr) create per-group partitions.
  Partition *sharedComputePartition = nullptr;
  if (dataPartitionFactor <= 1) {
    // All MMA users go to one computation partition (bwd pattern).
    sharedComputePartition = nullptr; // lazy-created on first use
  }

  // When dpFactor > 1 and there is no defaultPartition, pre-assign
  // DataPartition-categorized ops to their respective computation partitions
  // BEFORE scheduleUsers runs. Without a defaultPartition, Phase 4 (load user
  // propagation) is skipped, so shared/correction ops are not pre-claimed.
  // This causes the first MMA's scheduleUsers to greedily absorb all
  // computation ops into a single partition (e.g., when an scf.if merges both
  // partitions' values). The categorizer already knows which ops belong to
  // which data partition via backward slice analysis.
  //
  // When defaultPartition exists (e.g., FA with epilogue stores), the
  // original Phase 4 → Phase 5 flow works correctly — skip pre-assignment to
  // avoid changing the partition creation order that downstream passes
  // expect.
  DenseMap<unsigned, Partition *> dpIdToPartition;
  DenseMap<Operation *, Partition *> mmaToPreassignedPartition;
  if (dataPartitionFactor > 1) {
    // Pre-assign exclusive DataPartition ops to per-dpId computation
    // partitions.
    auto dpOps = categorizer.getOpsInCategory(OpCategory::DataPartition);

    // Collect which dpIds actually have ops.
    DenseSet<unsigned> usedDpIds;
    for (const auto &catOp : dpOps)
      usedDpIds.insert(catOp.dataPartitionId);

    // Pre-create computation partitions in reverse dpId order to match
    // the original partition creation order.
    SmallVector<unsigned> sortedDpIds(usedDpIds.begin(), usedDpIds.end());
    llvm::sort(sortedDpIds, std::greater<unsigned>());
    for (unsigned dpId : sortedDpIds) {
      dpIdToPartition[dpId] = schedule.addPartition(0);
      dpIdToPartition[dpId]->setType("computation");
    }

    for (const auto &catOp : dpOps) {
      unsigned dpId = catOp.dataPartitionId;
      auto it = dpIdToPartition.find(dpId);
      if (it != dpIdToPartition.end()) {
        tryScheduleOp(it->second, catOp.op);
        if (catOp.parentMMA)
          mmaToPreassignedPartition[catOp.parentMMA] = it->second;
      }
    }
    // Pre-assign shared ops (ops appearing in multiple MMA backward slices,
    // e.g., scf.if for masking) to the default partition. Without this,
    // propagatePartitions forms clusters around the unscheduled scf.if with
    // multiple sink partitions (one per data partition), collapsing them
    // into a single partition.
    if (defaultPartition) {
      for (Operation *sharedOp : categorizer.getSharedOps()) {
        if (!isa<arith::ConstantOp>(sharedOp))
          tryScheduleOp(defaultPartition, sharedOp);
      }
    }
  }

  for (auto mmaOp : llvm::reverse(mmas)) {
    if (mmaOp->getParentOfType<scf::ForOp>() == loops[0]) {
      Partition *targetPart = nullptr;
      LDBG("[phase5] Processing MMA: "
           << prettyOp(mmaOp) << " dpId=" << categorizer.getDpId(mmaOp)
           << " inPreassigned=" << mmaToPreassignedPartition.count(mmaOp));
      if (dataPartitionFactor > 1) {
        // Check if this MMA has a pre-assigned partition (flex path).
        auto it = mmaToPreassignedPartition.find(mmaOp);
        if (it != mmaToPreassignedPartition.end()) {
          targetPart = it->second;
        } else {
          // This MMA (e.g., a QK MMA) has no pre-assigned partition, but
          // its users may already be pre-assigned to a computation partition
          // (e.g., tmem_load and mulf(QK*scale) are DataPartition ops).
          // Use that existing partition to avoid creating extra computation
          // partitions that inflate TMEM channel count.
          for (OpOperand &use : mmaOp->getUses()) {
            Operation *user = use.getOwner();
            {
              auto ids = safeGetPartitionIds(user);
              for (int id : ids) {
                Partition *p = schedule.getPartition(id);
                if (p && p->getType() == "computation") {
                  targetPart = p;
                  break;
                }
              }
            }
            if (targetPart)
              break;
          }
          // If no user has a computation partition, look up the MMA's dpId
          // and use the corresponding pre-created computation partition.
          // This handles the case where the MMA itself has a dpId but its
          // users aren't pre-assigned (e.g., Hopper QK MMA whose users are
          // softmax ops that will be scheduled later by scheduleUsers).
          if (!targetPart) {
            unsigned dpId = categorizer.getDpId(mmaOp);
            LDBG("[phase5]   dpId fallback: dpId="
                 << dpId
                 << " dpIdToPartition.count=" << dpIdToPartition.count(dpId));
            if (dpId != SHARED_DPID) {
              auto it = dpIdToPartition.find(dpId);
              if (it != dpIdToPartition.end())
                targetPart = it->second;
            }
          }
          LDBG("[phase5]   targetPart after fallback: "
               << (targetPart ? targetPart->getType() : "null"));
          // If we found a pre-assigned computation partition, skip
          // scheduleUsers entirely — all MMA users are already pre-assigned
          // and calling scheduleUsers would create extra partitions from
          // unscheduled transitive users (yield ops, loop-carried args).
          if (targetPart) {
            // For non-MMAv5 ops without a gemm partition, also schedule the
            // MMA op itself into the computation partition.
            if (!mmaPartition)
              tryScheduleOp(targetPart, mmaOp);
            mmaToPartition[mmaOp] = targetPart;
            inFirstLoop.push_back(mmaOp);
            continue;
          }
        }
        // Otherwise nullptr → scheduleUsers creates a new partition (FA
        // path).
      } else {
        // bwd: all MMA users share one partition
        targetPart = sharedComputePartition;
      }
      auto part = scheduleUsers(mmaOp->getParentOfType<scf::ForOp>(), schedule,
                                targetPart, mmaOp);
      if (dataPartitionFactor <= 1 && !sharedComputePartition)
        sharedComputePartition = part;
      if (!part)
        part = targetPart;
      // For non-MMAv5 ops without a gemm partition, also schedule the
      // MMA op itself into the computation partition.
      if (!mmaPartition && part)
        tryScheduleOp(part, mmaOp);
      mmaToPartition[mmaOp] = part;
      inFirstLoop.push_back(mmaOp);
    }
  }

  // For dpFactor<=1 (BWD), populate dpIdToPartition so
  // schedulePostLoopOps can route via mergeEpilogueToComputation.
  if (dataPartitionFactor <= 1 && dpIdToPartition.empty()) {
    if (sharedComputePartition) {
      dpIdToPartition[0] = sharedComputePartition;
    } else {
      // Fallback: find any computation partition in the schedule.
      for (Partition &p : schedule.getPartitions()) {
        if (p.getType() == "computation") {
          dpIdToPartition[0] = &p;
          break;
        }
      }
    }
  }

  // For causal attention with 3 loops, match MMAs in second loop to first
  // loop
  unsigned Idx = 0;
  for (auto mmaOp : llvm::reverse(mmas)) {
    if (loops.size() == 3 && mmaOp->getParentOfType<scf::ForOp>() == loops[1]) {
      auto *part = mmaToPartition[inFirstLoop[Idx]];
      scheduleUsers(mmaOp->getParentOfType<scf::ForOp>(), schedule, part,
                    mmaOp);
      ++Idx;
    }
  }

  // Assign remaining unscheduled inner-loop ops using their dpId.
  // Only assign to computation partitions that already exist in
  // dpIdToPartition (don't create new ones).
  // For ops not in opToDpId (e.g., l_i update chain: l_i*alpha, l_i+l_ij),
  // trace through operands to find the dpId from an operand that IS in
  // opToDpId.
  if (dataPartitionFactor > 1 && !dpIdToPartition.empty()) {
    // Helper to find dpId by tracing operands.
    auto findDpIdFromOperands = [&](Operation *op) -> unsigned {
      unsigned dpId = categorizer.getDpId(op);
      if (dpId != 0 && dpId != SHARED_DPID)
        return dpId;
      // Trace through operands to find a non-zero dpId.
      SmallVector<Operation *> worklist;
      DenseSet<Operation *> visited;
      for (Value operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp())
          worklist.push_back(defOp);
      }
      while (!worklist.empty()) {
        Operation *curr = worklist.pop_back_val();
        if (!visited.insert(curr).second)
          continue;
        unsigned currDpId = categorizer.getDpId(curr);
        if (currDpId != 0 && currDpId != SHARED_DPID)
          return currDpId;
        // Also check if the op has a partition assignment that maps to
        // a computation partition.
        {
          auto ids = safeGetPartitionIds(curr);
          if (!ids.empty())
            for (int id : ids) {
              Partition *p = schedule.getPartition(id);
              if (p && p->getType() == "computation") {
                // Find which dpId maps to this partition.
                for (auto &[did, part] : dpIdToPartition) {
                  if (part == p)
                    return did;
                }
              }
            }
        }
        for (Value operand : curr->getOperands()) {
          if (auto *defOp = operand.getDefiningOp())
            worklist.push_back(defOp);
        }
      }
      return dpId; // fallback to original (may be 0)
    };

    scf::ForOp innermostLoop = loops[0];
    for (Operation &op : innermostLoop.getOps()) {
      if (hasPartition(&op))
        continue;
      if (isa<arith::ConstantOp, scf::YieldOp>(&op))
        continue;
      // Skip loop counter increment ops (scalar integer arithmetic that
      // feeds the yield). These are loop-control ops, not data-partition
      // computation ops.
      if (op.getNumResults() == 1 && op.getResult(0).getType().isIntOrIndex() &&
          !isa<RankedTensorType>(op.getResult(0).getType()))
        continue;
      unsigned dpId = findDpIdFromOperands(&op);
      if (dpId != SHARED_DPID) {
        auto it = dpIdToPartition.find(dpId);
        if (it != dpIdToPartition.end())
          tryScheduleOp(it->second, &op);
      }
    }
  }

  // Pre-schedule post-loop ops before propagatePartitions claims them.
  schedulePostLoopOps(mainLoop, schedule, layout, schedOpts,
                      categorizer.getOpToDpIdMap(), dpIdToPartition);

  // Update defaultPartition after computation partitions are created.
  layout.defaultPartition = layout.getDefaultPartition();

  bool createComputePartitions =
      (layout.correctionPartition != nullptr ||
       layout.reductionPartition != nullptr || dataPartitionFactor > 1) &&
      layout.defaultPartition != nullptr;

  return ScheduleResult{std::move(schedule),
                        layout,
                        schedOpts,
                        categorizer.getOpToDpIdMap(),
                        std::move(dpIdToPartition),
                        createComputePartitions};
}

namespace {
// This data structure represents a cluster of operations that have not been
// assigned to a stage. Operations form a cluster when:
//
// - they are adjacent in the SSA use def graph
// - they are not already assigned to a partition
// - at least one of their inputs is reachable from a definition partition
//
struct OpCluster {
  // These are the operations in the cluster.
  SetVector<Operation *> ops;
  // The definition partitions are the partitions from which inputs of the
  // operation are reachable. When the cluster is fully formed, the defining
  // op in the loop of any input to any operation in the cluster is either in
  // the root partition or one of these partitions.
  SetVector<Partition *> defPartitions;
  // The sink partitions which consume the outputs of operations in this
  // cluster. When the cluster is fully formed, all uses in the loop of
  // outputs of any operation in the cluster belong to one of these
  // partitions.
  SetVector<Partition *> sinkPartitions;
};

// Owning class for a bunch of clusters. This class manages the lifetimes of
// the clusters and has some helper functions.
struct OpClusters : public llvm::MapVector<Operation *, OpCluster *> {
  using MapVector::MapVector;

  // Create a new cluster that contains only the given operation, a return a
  // cluster that already contains the operation.
  OpCluster *getOrCreate(Operation *op) {
    OpCluster *&cluster = (*this)[op];
    if (!cluster) {
      cluster = clusters.emplace_back(new OpCluster).get();
      cluster->ops.insert(op);
    }
    return cluster;
  }
  // Merge two clusters by merging their sets and clearing the other cluster,
  // marking it as dead.
  void merge(OpCluster *dst, OpCluster *src) {
    dst->ops.insert_range(src->ops);
    dst->defPartitions.insert_range(src->defPartitions);
    dst->sinkPartitions.insert_range(src->sinkPartitions);
    for (Operation *op : src->ops)
      (*this)[op] = dst;
    src->ops.clear();
    src->defPartitions.clear();
    src->sinkPartitions.clear();
  }

  SmallVector<std::unique_ptr<OpCluster>> clusters;
};
} // namespace

namespace {

// Operations that require partition assignment are those reachable from an
// operation in a partition. This function propagates partitions by first
// forming contiguous clusters from the unassigned operations and then
// deciding what to do with the operations in that cluster.
// Check if an op produces only scalar results (can be rematerialized).
static bool isScalarOp(Operation *op) {
  if (op->getNumResults() == 0)
    return false;
  return llvm::all_of(op->getResults(), [](Value v) {
    return !isa<RankedTensorType, triton::gpu::MemDescType>(v.getType());
  });
}

void propagatePartitions(scf::ForOp loop, PartitionSet &schedule,
                         bool createComputePartitions) {
  OpClusters opClusters;

  for (Partition &partition : schedule.getPartitions()) {
    // For each partition, check if any of their inputs are reachable from
    // another partition and spawn a single cluster at that operation.
    auto defCallback = [&](OpResult result, unsigned distance) {
      Operation *defOp = result.getDefiningOp();
      if (!hasPartition(defOp) && hasDefPartition(loop, defOp, schedule)) {
        // Add the current partition as a sink to the cluster.
        opClusters.getOrCreate(defOp)->sinkPartitions.insert(&partition);
      }
    };
    partition.iterateDefs(loop, defCallback);

    // For each partition, place users of its outputs in a cluster if it is
    // not already assigned to a partition.
    auto useCallback = [&](OpResult result, OpOperand &use, unsigned distance) {
      Operation *user = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      // Skip users outside the loop — they are handled by
      // schedulePostLoopOps.
      if (!user)
        return;
      if (!hasPartition(user)) {
        // Add the current partition as a def to the cluster.
        opClusters.getOrCreate(user)->defPartitions.insert(&partition);
      }
    };
    partition.iterateUses(loop, useCallback);
  }

  // Now we have a pile of single-operation clusters directly adjacent to the
  // operations in a partition. Grow the clusters by adding adjacent
  // operations clusters and merging clusters when possible.
  SmallVector<Operation *> worklist =
      llvm::to_vector(llvm::make_first_range(opClusters));
  while (!worklist.empty()) {
    // Grab an op off the worklist. We know it has a cluster already.
    Operation *op = worklist.pop_back_val();
    OpCluster *cluster = opClusters.find(op)->second;
    // Look at the definitions directly feeding into this operation.
    iterateDefs(loop, op, [&](OpResult def) {
      Operation *defOp = def.getDefiningOp();
      if (hasPartition(defOp)) {
        // The input originates from an operation already assigned to a
        // partition. Add this as a def partition.
        for (auto id : safeGetPartitionIds(defOp)) {
          cluster->defPartitions.insert(schedule.getPartition(id));
        }
      } else {
        // If the input is not reachable from a partition, ignore it.
        if (!hasDefPartition(loop, defOp, schedule))
          return;
        // This operation is not assigned to a partition.
        OpCluster *&defCluster = opClusters[defOp];
        if (!defCluster) {
          // This operation has not yet been added to a cluster. Add it to the
          // current cluster and recurse on it.
          defCluster = cluster;
          cluster->ops.insert(defOp);
          worklist.push_back(defOp);
        } else if (defCluster != cluster) {
          // This operation is part of another cluster. Merge the two clusters
          // together and continue.
          opClusters.merge(cluster, defCluster);
        }
      }
    });
    // Check the users of the operation.
    iterateUsers(loop, op, [&](Operation *user) {
      if (hasPartition(user)) {
        // If the user is already assigned to a partition, add that partition
        // as one of the sink partitions.
        for (auto id : safeGetPartitionIds(user)) {
          cluster->sinkPartitions.insert(schedule.getPartition(id));
        }
        return;
      }
      // If the user does not already have a cluster, add it to the current
      // cluster. We don't have to handle merging here because when the user
      // visits the current op, it will trigger the merge.
      OpCluster *&userCluster = opClusters[user];
      if (userCluster)
        return;
      userCluster = cluster;
      cluster->ops.insert(user);
      worklist.push_back(user);
    });
  }

  // We have clustered unassigned ops in the liveouts of ops in assigned
  // partitions and in the critical paths between ops in different partitions.
  // Ops that are next to each other are placed in the same cluster. Now the
  // task is to figure out how to assign partitions to the ops in each cluster
  // based on the def and sink partitions, which is very non-trivial.
  for (OpCluster &cluster : llvm::make_pointee_range(opClusters.clusters)) {
    // Skip dead clusters.
    if (cluster.ops.empty())
      continue;
    // Skip clusters with no def partitions (all scalar ops).
    if (cluster.defPartitions.empty())
      continue;
    assert(llvm::all_of(cluster.ops,
                        [&](Operation *op) { return !hasPartition(op); }));

    // If there are multiple def or sink partitions, don't know what to do.
    // Assign the whole cluster to its own partition.
    if (cluster.defPartitions.size() > 1 || cluster.sinkPartitions.size() > 1) {
      // For BWD-like kernels (has reduction partition, no epilogue
      // partition), avoid creating extra partitions which can split
      // pointer-typed ops across partitions and crash createLocalAlloc. Reuse
      // the existing computation partition instead.
      Partition *existingComputation = nullptr;
      bool hasReduction = false;
      bool hasEpilogue = false;
      for (Partition &p : schedule.getPartitions()) {
        if (p.getType() == "reduction")
          hasReduction = true;
        if (p.getType() == "epilogue")
          hasEpilogue = true;
        if (p.getType() == "computation")
          existingComputation = &p;
      }
      if (hasReduction && !hasEpilogue && existingComputation) {
        for (Operation *op : cluster.ops) {
          if (isScalarOp(op))
            continue;
          setPartition(op, existingComputation);
        }
        continue;
      }
      // For GEMM with data partitioning, merge into the default partition
      // instead of creating a separate computation partition.
      // TODO: Fix issues with DataPartitioning.
      if (!createComputePartitions) {
        Partition *fallbackPartition = nullptr;
        for (Partition &p : schedule.getPartitions()) {
          if (p.getType() == "default") {
            fallbackPartition = &p;
            break;
          }
        }
        // When no default partition exists (e.g., Hopper with all categories
        // merged), use the first computation partition as fallback.
        if (!fallbackPartition) {
          for (Partition &p : schedule.getPartitions()) {
            if (p.getType() == "computation") {
              fallbackPartition = &p;
              break;
            }
          }
        }
        if (fallbackPartition) {
          for (Operation *op : cluster.ops) {
            if (isScalarOp(op))
              continue;
            setPartition(op, fallbackPartition);
          }
          continue;
        }
      }
      // For data-partitioned kernels: if a single computation partition is
      // in the sinks, assign the cluster there instead of creating extra
      // computation partitions. This prevents partition inflation (e.g., 4
      // computation partitions instead of 2) when intermediate ops between
      // the gemm and computation partitions form a cluster.
      if (cluster.sinkPartitions.size() == 1 &&
          cluster.sinkPartitions.front()->getType() == "computation") {
        for (Operation *op : cluster.ops) {
          if (isScalarOp(op))
            continue;
          setPartition(op, cluster.sinkPartitions.front());
        }
        continue;
      }
      Partition *newPartition = schedule.addPartition(0);
      newPartition->setType("computation");
      for (Operation *op : cluster.ops) {
        if (isScalarOp(op))
          continue;
        setPartition(op, newPartition);
      }
      continue;
    }

    // If there is no sink partition, this means there is a backedge
    // somewhere, for now assign the cluster to the def partition.
    Partition *defPartition = cluster.defPartitions.front();
    if (cluster.sinkPartitions.empty()) {
      for (Operation *op : cluster.ops) {
        if (isScalarOp(op))
          continue;
        setPartition(op, defPartition);
      }
      continue;
    }

    // Find the critical path between the def partition and sink partition.
    Partition *sinkPartition = cluster.sinkPartitions.front();
    SetVector<Operation *> critPath;
    DenseSet<Operation *> opsInCluster(cluster.ops.begin(), cluster.ops.end());
    auto callback = [&](OpResult result, unsigned distance) {
      Operation *defOp = result.getDefiningOp();
      if (opsInCluster.contains(defOp))
        critPath.insert(defOp);
    };
    sinkPartition->iterateDefs(loop, callback);
    for (unsigned i = 0; i < critPath.size(); ++i) {
      Operation *op = critPath[i];
      iterateDefs(loop, op, [&](OpResult def) {
        Operation *defOp = def.getDefiningOp();
        if (opsInCluster.contains(defOp))
          critPath.insert(defOp);
      });
    }

    // If all ops are on the critical path, assign them to the def partition.
    if (critPath.size() == cluster.ops.size()) {
      for (Operation *op : cluster.ops) {
        if (isScalarOp(op))
          continue;
        setPartition(op, defPartition);
      }
      continue;
    }

    // Some ops are on the critical path, and there is also a backedge.
    // Rematerialize the critical path ops into the sink partition. Leave the
    // rest in the def partition and rely on DCE to remove them.
    critPath = topologicalSort(critPath);
    DenseSet<Operation *> sinkOps(sinkPartition->getOps().begin(),
                                  sinkPartition->getOps().end());
    for (Operation *op : llvm::reverse(critPath)) {
      OpBuilder b(op);
      Operation *clone = b.clone(*op);
      op->replaceUsesWithIf(clone->getResults(), [&](OpOperand &use) {
        return sinkOps.contains(use.getOwner());
      });
      sinkOps.insert(clone);
      setPartition(clone, sinkPartition);
    }
    for (Operation *op : cluster.ops) {
      if (isScalarOp(op))
        continue;
      setPartition(op, defPartition);
    }
  }
}

/// Walk over \p loop and clone Broadcast/ExpandDims ops into each
/// partition that they have users in. This reduces the amount of data that
/// needs to be transferred through memory.
///
/// When a ConvertLayoutOp sits between an ExpandDimsOp/BroadcastOp and its
/// consumer (e.g., due to upstream layout choices producing different
/// encodings), also walk backward and clone the operand chain
/// (ConvertLayoutOp, ExpandDimsOp, BroadcastOp) to avoid creating an
/// unintended cross-partition boundary.
void optimizeSchedule(scf::ForOp loop, PartitionSet &schedule) {
  // Helper to get partition for an op, returning null if unscheduled.
  auto getPartition = [&](Operation *op) -> Partition * {
    if (!hasPartition(op))
      return nullptr;
    auto ids = safeGetPartitionIds(op);
    if (ids.size() != 1)
      return nullptr;
    return schedule.getPartition(static_cast<unsigned>(ids[0]));
  };

  // After cloning a BroadcastOp/ExpandDimsOp into a user partition, walk
  // backward through the cloned op's operand chain and also clone any
  // ConvertLayoutOp/BroadcastOp/ExpandDimsOp that feeds it from a different
  // partition. This handles the pattern where upstream layout passes insert
  // a ConvertLayoutOp between ExpandDimsOp and BroadcastOp, which would
  // otherwise break the cloning chain and create a cross-partition boundary.
  auto cloneOperandChain = [&](Operation *clonedOp, Partition *userPartition) {
    Operation *current = clonedOp;
    while (true) {
      Operation *toPull = nullptr;
      unsigned operandIdx = 0;
      for (auto [idx, operand] : llvm::enumerate(current->getOperands())) {
        auto *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        Partition *defPartition = getPartition(defOp);
        if (!defPartition || defPartition == userPartition)
          continue;
        if (!isa<ConvertLayoutOp, BroadcastOp, ExpandDimsOp>(defOp))
          continue;
        toPull = defOp;
        operandIdx = idx;
        break;
      }
      if (!toPull)
        break;
      Operation *pullClone = OpBuilder(toPull).clone(*toPull);
      setPartition(pullClone, userPartition);
      current->setOperand(operandIdx, pullClone->getResult(0));
      current = pullClone;
    }
  };

  // Walk everything in reverse so that operations are visited before their
  // operands.
  loop.walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    if (!isa<MemDescTransOp, ConvertLayoutOp, BroadcastOp, ExpandDimsOp>(op))
      // if (op->getNumResults() != 1 || !isPure(op) ||
      //     isa<scf::YieldOp, scf::ForOp, scf::IfOp>(op))
      return;

    Partition *partition = getPartition(op);
    if (!partition)
      return;

    // Record all the other partitions in which we have users.
    llvm::SmallDenseSet<Partition *, 2> userPartitions;
    for (OpOperand &use : op->getUses()) {
      Partition *userPartition = getPartition(use.getOwner());
      if (!userPartition || userPartition == partition)
        continue;
      userPartitions.insert(userPartition);
    }

    for (auto *userPartition : userPartitions) {
      // Clone the instruction into each user partition.
      Operation *clone = OpBuilder(op).clone(*op);
      setPartition(clone, userPartition);
      // Replace all users in that partition with the clone.
      op->replaceUsesWithIf(clone->getResults(), [&](OpOperand &otherUse) {
        return getPartition(otherUse.getOwner()) == userPartition;
      });
      // Walk backward and clone any cheap layout ops feeding the clone.
      cloneOperandChain(clone, userPartition);
    }
  });
}

/// Split scf.if ops whose results feed different computation partitions
/// into separate per-partition scf.if ops. This is needed for
/// data-partitioned kernels (like flex attention) where an scf.if for masking
/// returns both data partitions' results as a tuple. Without splitting, the
/// downstream WSCodePartition pass creates channels from the single scf.if
/// producer to consumers in different tasks, violating the "channels sharing
/// the same producer must be in the same task" invariant.
///
/// Before:
///   %r:2 = scf.if %cond -> (T, T) {
///     yield %a, %b          // %a for dp0, %b for dp1
///   } else {
///     yield %c, %d          // %c for dp0, %d for dp1
///   } {ttg.partition = [0]}  // default partition
///   use(%r#0) {ttg.partition = [3]}  // computation partition dp0
///   use(%r#1) {ttg.partition = [4]}  // computation partition dp1
///
/// After:
///   %r0 = scf.if %cond -> (T) {
///     yield %a
///   } else {
///     yield %c
///   } {ttg.partition = [3]}  // dp0 computation partition
///   %r1 = scf.if %cond -> (T) {
///     yield %b
///   } else {
///     yield %d
///   } {ttg.partition = [4]}  // dp1 computation partition
///   use(%r0) {ttg.partition = [3]}
///   use(%r1) {ttg.partition = [4]}
void splitDataPartitionedIfOps(scf::ForOp loop, PartitionSet &schedule) {
  SmallVector<scf::IfOp> ifsToSplit;

  loop.walk([&](scf::IfOp ifOp) {
    if (ifOp.getNumResults() < 2)
      return;

    // Check if results feed different partitions.
    DenseSet<int> resultPartitions;
    bool hasDifferentPartitions = false;
    for (OpResult result : ifOp.getResults()) {
      for (Operation *user : result.getUsers()) {
        auto ids = safeGetPartitionIds(user);
        for (int id : ids)
          resultPartitions.insert(id);
      }
    }
    // Only split if results feed more than one computation partition.
    unsigned computationCount = 0;
    for (Partition &p : schedule.getPartitions()) {
      if (p.getType() == "computation" &&
          resultPartitions.contains(p.getIndex()))
        computationCount++;
    }
    if (computationCount >= 2)
      ifsToSplit.push_back(ifOp);
  });

  for (scf::IfOp ifOp : ifsToSplit) {
    unsigned numResults = ifOp.getNumResults();
    OpBuilder builder(ifOp);

    // For each result, determine which computation partition its users belong
    // to, then find which yield operands in the then/else blocks map to it.
    // Group results by their consumer partition.
    DenseMap<int, SmallVector<unsigned>> partitionToResultIndices;
    for (unsigned i = 0; i < numResults; i++) {
      int partId = -1;
      for (Operation *user : ifOp.getResult(i).getUsers()) {
        auto ids = safeGetPartitionIds(user);
        if (!ids.empty()) {
          // Find a computation partition among the user's partitions.
          for (int id : ids) {
            Partition *p = schedule.getPartition(id);
            if (p && p->getType() == "computation") {
              partId = id;
              break;
            }
          }
        }
        if (partId >= 0)
          break;
      }
      if (partId >= 0)
        partitionToResultIndices[partId].push_back(i);
    }

    // Only split if we have at least 2 groups.
    if (partitionToResultIndices.size() < 2)
      continue;

    // Create one scf.if per partition group.
    for (auto &entry : partitionToResultIndices) {
      auto &partId = entry.first;
      auto &resultIndices = entry.second;
      auto *origThenBlock = ifOp.thenBlock();
      auto *origElseBlock = ifOp.elseBlock();
      auto *origThenYield = origThenBlock->getTerminator();
      auto *origElseYield =
          origElseBlock ? origElseBlock->getTerminator() : nullptr;

      // Collect needed ops for the else block via backward reachability.
      DenseSet<Operation *> neededElseOps;
      if (origElseBlock) {
        for (unsigned idx : resultIndices) {
          SmallVector<Operation *> worklist;
          if (auto *def = origElseYield->getOperand(idx).getDefiningOp())
            worklist.push_back(def);
          while (!worklist.empty()) {
            Operation *curr = worklist.pop_back_val();
            if (!curr || curr->getBlock() != origElseBlock)
              continue;
            if (!neededElseOps.insert(curr).second)
              continue;
            for (Value operand : curr->getOperands())
              if (auto *def = operand.getDefiningOp())
                worklist.push_back(def);
          }
        }
      }

      // Build result types for this split.
      SmallVector<Type> splitResultTypes;
      for (unsigned idx : resultIndices)
        splitResultTypes.push_back(ifOp.getResult(idx).getType());

      // Use the callback-based builder to populate then/else blocks.
      auto thenBuilder = [&](OpBuilder &b, Location loc) {
        IRMapping mapping;
        for (Operation &op : origThenBlock->without_terminator()) {
          bool needed = false;
          for (unsigned idx : resultIndices) {
            if (origThenYield->getOperand(idx).getDefiningOp() == &op)
              needed = true;
          }
          if (needed)
            b.clone(op, mapping);
        }
        SmallVector<Value> yieldVals;
        for (unsigned idx : resultIndices)
          yieldVals.push_back(
              mapping.lookupOrDefault(origThenYield->getOperand(idx)));
        scf::YieldOp::create(b, loc, yieldVals);
      };

      auto elseBuilder = [&](OpBuilder &b, Location loc) {
        IRMapping mapping;
        for (Operation &op : origElseBlock->without_terminator()) {
          if (neededElseOps.contains(&op))
            b.clone(op, mapping);
        }
        SmallVector<Value> yieldVals;
        for (unsigned idx : resultIndices)
          yieldVals.push_back(
              mapping.lookupOrDefault(origElseYield->getOperand(idx)));
        scf::YieldOp::create(b, loc, yieldVals);
      };

      auto newIf = scf::IfOp::create(
          builder, ifOp.getLoc(), ifOp.getCondition(), thenBuilder,
          origElseBlock
              ? elseBuilder
              : static_cast<function_ref<void(OpBuilder &, Location)>>(
                    nullptr));

      // Assign the new scf.if to this computation partition.
      setPartition(newIf, schedule.getPartition(partId));

      // Replace uses of the original results with the new scf.if results.
      for (unsigned i = 0; i < resultIndices.size(); i++) {
        ifOp.getResult(resultIndices[i]).replaceAllUsesWith(newIf.getResult(i));
      }
    }

    // Erase the original scf.if (all uses should be replaced).
    ifOp.erase();
  }
}

} // namespace

namespace mlir {
#define GEN_PASS_DEF_NVGPUPARTITIONSCHEDULINGMETA
#include "nvidia/hopper/include/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
struct PartitionSchedulingMeta
    : public mlir::impl::NVGPUPartitionSchedulingMetaBase<
          PartitionSchedulingMeta> {
  using NVGPUPartitionSchedulingMetaBase::NVGPUPartitionSchedulingMetaBase;

  void runOnOperation() override;
};
} // namespace

void PartitionSchedulingMeta::runOnOperation() {
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName))
      loops.push_back(loop);
  });
  for (auto [idx, loop] : llvm::enumerate(loops)) {
    // Build SchedulingOptions from pass options and per-loop attributes.
    SchedulingOptions schedOpts;
    schedOpts.mergeEpilogue = mergeEpilogue;
    schedOpts.mergeEpilogueToComputation = mergeEpilogueToComputation;
    schedOpts.mergeCorrection = mergeCorrection;
    schedOpts.mergeReduction = mergeReduction;
    schedOpts.separateEpilogueStore = separateEpilogueStore;

    // Per-loop tt.merge_epilogue_to_computation overrides pass option.
    if (auto attr =
            loop->getAttrOfType<BoolAttr>("tt.merge_epilogue_to_computation"))
      schedOpts.mergeEpilogueToComputation = attr.getValue();

    // Per-loop tt.separate_epilogue_store overrides pass option.
    if (auto attr = loop->getAttrOfType<BoolAttr>("tt.separate_epilogue_store"))
      schedOpts.separateEpilogueStore = attr.getValue();

    // Per-loop tt.merge_correction overrides pass option.
    if (auto attr = loop->getAttrOfType<BoolAttr>("tt.merge_correction"))
      schedOpts.mergeCorrection = attr.getValue();

    // Per-loop tt.merge_epilogue overrides pass option.
    if (auto attr2 = loop->getAttrOfType<BoolAttr>("tt.merge_epilogue"))
      schedOpts.mergeEpilogue = attr2.getValue();

    if (std::optional<ScheduleResult> result =
            getInitialSchedule(loop, schedOpts)) {
      PartitionSet &schedule = result->schedule;
      propagatePartitions(loop, schedule, result->createComputePartitions);

      optimizeSchedule(loop, schedule);

      // Split scf.if ops whose results feed different computation partitions.
      // This must run after all partition assignments are finalized (after
      // propagatePartitions + optimizeSchedule) but before serialization.
      splitDataPartitionedIfOps(loop, schedule);

      schedule.serialize(loop);
      loop->setAttr(
          kWarpSpecializeTagAttrName,
          IntegerAttr::get(IntegerType::get(loop.getContext(), 32), idx));
      // Clean Broadcast/ExpandDims that were left with no users
      // after optimizeSchedule. We wait until after the schedule is
      // serialized to avoid invalidating pointers stored in the schedule.
      loop.walk<WalkOrder::PostOrder, ReverseIterator>([](Operation *op) {
        // By default, the walk is in postorder so it is safe to delete ops
        // while we walk.
        if (op->use_empty() && isPure(op) && op->getNumResults() == 1 &&
            !isa<scf::YieldOp, scf::ForOp, scf::IfOp>(op))
          op->erase();
      });
    }
  }
}
