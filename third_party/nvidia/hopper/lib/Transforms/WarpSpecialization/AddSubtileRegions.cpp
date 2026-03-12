#include "Utility.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvgpu-add-subtile-regions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return true if \p op is a reshape or trans (shape-adjusting, no compute).
static bool isShapeOp(Operation *op) { return isa<ReshapeOp, TransOp>(op); }

/// Return true if \p op loads from TMEM/SMEM into registers.
/// Note: We don't have an explicit SMEM load op yet, so this
/// is just TMEM.
static bool isMemoryToRegistersLoad(Operation *op) {
  return isa<TMEMLoadOp>(op);
}

/// Trace backward from \p splitOp through single-use reshape/trans chains.
/// Returns the earliest op in the chain (may be a load or the first shape op).
/// If the backward trace reaches another SplitOp output, returns nullptr
/// (indicating this is a nested split, not a root).
static Operation *findSplitTreeRoot(SplitOp splitOp) {
  Value current = splitOp.getSrc();
  Operation *earliest = splitOp;

  while (auto *defOp = current.getDefiningOp()) {
    if (isShapeOp(defOp)) {
      earliest = defOp;
      current = defOp->getOperand(0);
      continue;
    }
    if (isMemoryToRegistersLoad(defOp)) {
      earliest = defOp;
      break;
    }
    // If we reach another SplitOp's output, this is nested.
    if (isa<SplitOp>(defOp))
      return nullptr;
    // Unknown op — start from the earliest shape op we found.
    break;
  }

  return earliest;
}

/// Recursively collect leaf values from a split tree.
/// Each SplitOp output that feeds through reshape→trans→split recurses;
/// otherwise it's a leaf.
static void collectLeaves(SplitOp splitOp, SmallVectorImpl<Value> &leaves) {
  for (Value result : splitOp.getResults()) {
    // Check if this result feeds into another split via reshape/trans chain.
    Operation *user = nullptr;
    Value traced = result;

    // Follow single-use reshape/trans chain.
    while (traced.hasOneUse()) {
      Operation *singleUser = *traced.getUsers().begin();
      if (isShapeOp(singleUser)) {
        traced = singleUser->getResult(0);
        continue;
      }
      if (isa<SplitOp>(singleUser)) {
        user = singleUser;
      }
      break;
    }

    if (user) {
      // Recurse into the nested split.
      collectLeaves(cast<SplitOp>(user), leaves);
    } else {
      // This is a leaf subtile value.
      leaves.push_back(result);
    }
  }
}

/// Collect unvisited ops in the backward def chain starting from \p start,
/// and append them in topological order (def before use) to \p setupOps.
static void collectIntermediateOps(Value start, DenseSet<Operation *> &visited,
                                   SmallVectorImpl<Operation *> &setupOps) {
  SmallVector<Operation *> intermediates;
  Value src = start;
  while (auto *defOp = src.getDefiningOp()) {
    if (visited.count(defOp))
      break;
    intermediates.push_back(defOp);
    if (defOp->getNumOperands() > 0)
      src = defOp->getOperand(0);
    else
      break;
  }
  for (auto it = intermediates.rbegin(); it != intermediates.rend(); ++it) {
    if (visited.insert(*it).second)
      setupOps.push_back(*it);
  }
}

/// Collect all ops in the setup chain: from \p root through the split tree
/// (inclusive). The ops are collected in topological order (def before use).
static void collectSetupOps(Operation *root, SplitOp rootSplit,
                            SmallVectorImpl<Operation *> &setupOps) {
  DenseSet<Operation *> visited;

  // First, collect the backward chain from rootSplit to root.
  SmallVector<Operation *> chain;
  Value v = rootSplit.getSrc();
  while (auto *defOp = v.getDefiningOp()) {
    if (defOp == root) {
      chain.push_back(defOp);
      break;
    }
    chain.push_back(defOp);
    if (defOp->getNumOperands() > 0)
      v = defOp->getOperand(0);
    else
      break;
  }

  // Add chain in reverse (topological) order.
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    if (visited.insert(*it).second)
      setupOps.push_back(*it);
  }

  // Now collect the split tree using DFS.
  SmallVector<SplitOp> splitWorklist;
  splitWorklist.push_back(rootSplit);

  while (!splitWorklist.empty()) {
    SplitOp split = splitWorklist.pop_back_val();
    if (!visited.insert(split).second)
      continue;
    // Add intermediate shape ops between parent split output and this split.
    collectIntermediateOps(split.getSrc(), visited, setupOps);
    setupOps.push_back(split);

    // Check each result for nested splits.
    for (Value result : split.getResults()) {
      Value traced = result;
      while (traced.hasOneUse()) {
        Operation *singleUser = *traced.getUsers().begin();
        if (isShapeOp(singleUser)) {
          traced = singleUser->getResult(0);
          continue;
        }
        if (auto nestedSplit = dyn_cast<SplitOp>(singleUser)) {
          splitWorklist.push_back(nestedSplit);
        }
        break;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Forward op matching
//===----------------------------------------------------------------------===//

/// A matched operation across all subtiles.
struct MatchedOp {
  /// The representative operation (from subtile 0).
  Operation *repOp;
  /// The per-subtile operations (one per leaf).
  SmallVector<Operation *> perTileOps;
  /// For each operand of repOp: if the operand is the same across all subtiles,
  /// store the value (it's "captured" from outer scope). If it differs, store
  /// nullptr (it will become a tile block argument fed from the previous step).
  SmallVector<Value> capturedOperands;
  /// Whether this is the last matched op (a memory op, inclusive).
  bool isTerminal;
};

/// Given N leaf subtile values, walk their users in lockstep, matching
/// structurally identical ops. Returns a list of MatchedOp descriptors.
/// Stops at the first op with memory effects (inclusive).
static SmallVector<MatchedOp> matchForwardOps(ArrayRef<Value> subtileLeaves) {
  unsigned numTiles = subtileLeaves.size();
  SmallVector<MatchedOp> matched;

  // Current "frontier" values for each subtile — start with the leaves.
  SmallVector<SmallVector<Value>> frontiers(1);
  frontiers[0].assign(subtileLeaves.begin(), subtileLeaves.end());

  bool keepGoing = true;
  do {
    auto &currentValues = frontiers.back();

    // Each current value should have exactly one user for matching.
    SmallVector<Operation *> userOps;
    for (unsigned i = 0; i < numTiles; ++i) {
      Value val = currentValues[i];
      if (!val.hasOneUse())
        return matched; // Can't match further.
      userOps.push_back(*val.getUsers().begin());
    }

    // Check structural identity: same op name, same number of results,
    // same result types, same attributes.
    Operation *rep = userOps[0];
    for (unsigned i = 1; i < numTiles; ++i) {
      Operation *other = userOps[i];
      if (rep->getName() != other->getName())
        return matched;
      if (rep->getNumResults() != other->getNumResults())
        return matched;
      for (unsigned r = 0; r < rep->getNumResults(); ++r) {
        if (rep->getResult(r).getType() != other->getResult(r).getType())
          return matched;
      }
      if (rep->getAttrs() != other->getAttrs())
        return matched;
      if (rep->getNumOperands() != other->getNumOperands())
        return matched;
    }

    // Classify operands.
    MatchedOp m;
    m.repOp = rep;
    m.perTileOps.assign(userOps.begin(), userOps.end());
    m.isTerminal = !isMemoryEffectFree(rep);

    for (unsigned opIdx = 0; opIdx < rep->getNumOperands(); ++opIdx) {
      // Check if the operand is the same value across all subtiles.
      Value repOperand = rep->getOperand(opIdx);
      bool allSame = true;
      for (unsigned i = 1; i < numTiles; ++i) {
        if (userOps[i]->getOperand(opIdx) != repOperand) {
          allSame = false;
          break;
        }
      }
      // Also check if the operand comes from the tile body chain
      // (i.e., from currentValues or a previous matched op result).
      // For now, operands that come from the current frontier are "varying"
      // (they map to per-tile block args). Operands that are the same across
      // all subtiles are "captured".
      if (allSame) {
        // Check if it's from the frontier (varying within tiles).
        bool fromFrontier = false;
        for (unsigned i = 0; i < numTiles; ++i) {
          if (currentValues[i] == repOperand) {
            fromFrontier = true;
            break;
          }
        }
        if (fromFrontier) {
          // It's varying (happens to be same value but comes from tile chain).
          m.capturedOperands.push_back(nullptr);
        } else {
          m.capturedOperands.push_back(repOperand);
        }
      } else {
        m.capturedOperands.push_back(nullptr); // varying
      }
    }

    matched.push_back(std::move(m));

    keepGoing = !matched.back().isTerminal && rep->getNumResults() == 1;
    if (keepGoing) {
      // Advance frontier: results of matched ops become new current values.
      SmallVector<Value> nextValues;
      for (unsigned i = 0; i < numTiles; ++i)
        nextValues.push_back(userOps[i]->getResult(0));
      frontiers.push_back(std::move(nextValues));
    }
  } while (keepGoing);

  return matched;
}

//===----------------------------------------------------------------------===//
// Async partition consistency
//===----------------------------------------------------------------------===//

/// Verify that all setup and matched ops share the same async_task_id
/// partition. Emits an error on \p rootSplit and returns failure if not.
static LogicalResult
verifyAsyncPartitionConsistency(Operation *setupRoot, SplitOp rootSplit,
                                ArrayRef<MatchedOp> matchedOps) {
  SmallVector<Operation *> setupOps;
  collectSetupOps(setupRoot, rootSplit, setupOps);

  SmallVector<AsyncTaskId> referenceIds;
  bool foundReference = false;

  auto checkOp = [&](Operation *op) -> bool {
    auto ids = getAsyncTaskIds(op);
    if (ids.empty())
      return true;
    if (!foundReference) {
      referenceIds = std::move(ids);
      foundReference = true;
      return true;
    }
    return ids == referenceIds;
  };

  for (Operation *op : setupOps)
    if (!checkOp(op))
      return rootSplit.emitError(
          "ops in subtile region have inconsistent async_task_id partitions");
  for (auto &m : matchedOps)
    for (Operation *op : m.perTileOps)
      if (!checkOp(op))
        return rootSplit.emitError(
            "ops in subtile region have inconsistent async_task_id partitions");
  return success();
}

//===----------------------------------------------------------------------===//
// SubtiledRegion construction
//===----------------------------------------------------------------------===//

/// Build a SubtiledRegionOp from the identified pattern.
static void buildSubtiledRegion(Operation *setupRoot, SplitOp rootSplit,
                                ArrayRef<Value> subtileLeaves,
                                ArrayRef<MatchedOp> matchedOps) {
  unsigned numTiles = subtileLeaves.size();
  if (matchedOps.empty())
    return;

  OpBuilder builder(matchedOps.front().perTileOps[0]);
  Location loc = setupRoot->getLoc();

  // Determine tile block arguments: for each matched op, collect the varying
  // operand values per tile. These become setup yields and tile block args.
  //
  // tileArgValues[tileIdx][argIdx] = the value for that tile's block arg.
  // We also track which (matchedOpIdx, operandIdx) each tile arg corresponds
  // to.
  struct TileArgSource {
    unsigned matchedOpIdx;
    unsigned operandIdx;
  };
  SmallVector<TileArgSource> tileArgSources;

  // First tile arg is always the subtile leaf value itself.
  // But actually, the subtile leaf feeds as input to the first matched op.
  // We need to figure out which operands of matched ops are "varying".

  // Build tile arg sources: for the first matched op, the varying operand
  // is the one that comes from subtileLeaves. For subsequent ops, varying
  // operands come from the previous matched op's result.
  //
  // We distinguish two kinds of varying operands:
  // 1. Those that map to the result of a previous matched op (intra-tile flow)
  // 2. Those that are genuinely different values per tile (need setup yields)
  //
  // For kind 2, we need to add them to the setup region yields.

  // Track which values in each tile flow from the previous matched op result.
  // prevResults[tileIdx] = result of the previous matched op for that tile.
  SmallVector<Value> prevResults(subtileLeaves.begin(), subtileLeaves.end());

  // Collect per-tile varying operand values that need to come from setup.
  // setupYieldValues[tileIdx] = values to yield for that tile.
  SmallVector<SmallVector<Value>> setupYieldValues(numTiles);

  for (unsigned mIdx = 0; mIdx < matchedOps.size(); ++mIdx) {
    auto &m = matchedOps[mIdx];
    for (unsigned opIdx = 0; opIdx < m.repOp->getNumOperands(); ++opIdx) {
      if (m.capturedOperands[opIdx])
        continue; // Captured from outer scope, not a tile arg.

      // Check if this varying operand comes from prevResults (intra-tile flow).
      bool isIntraTile = true;
      for (unsigned t = 0; t < numTiles; ++t) {
        if (m.perTileOps[t]->getOperand(opIdx) != prevResults[t]) {
          isIntraTile = false;
          break;
        }
      }

      if (!isIntraTile) {
        // This is a genuinely varying operand — needs a setup yield.
        tileArgSources.push_back({mIdx, opIdx});
        for (unsigned t = 0; t < numTiles; ++t) {
          setupYieldValues[t].push_back(m.perTileOps[t]->getOperand(opIdx));
        }
      }
    }

    // Update prevResults with this op's results.
    if (m.repOp->getNumResults() == 1) {
      for (unsigned t = 0; t < numTiles; ++t)
        prevResults[t] = m.perTileOps[t]->getResult(0);
    }
  }

  // The subtile leaves themselves are always the first tile arg.
  // Insert them at the beginning of setupYieldValues.
  for (unsigned t = 0; t < numTiles; ++t) {
    setupYieldValues[t].insert(setupYieldValues[t].begin(), subtileLeaves[t]);
  }

  // Number of tile block args = 1 (subtile leaf) + extra varying operands.
  unsigned numTileArgs = 1 + tileArgSources.size();

  // Build the flat setup yield values in tile-major order.
  SmallVector<Value> flatSetupYields;
  for (unsigned t = 0; t < numTiles; ++t) {
    for (auto &v : setupYieldValues[t])
      flatSetupYields.push_back(v);
  }

  // Collect types for tile block args.
  SmallVector<Type> tileArgTypes;
  tileArgTypes.push_back(subtileLeaves[0].getType());
  for (auto &src : tileArgSources) {
    auto &m = matchedOps[src.matchedOpIdx];
    tileArgTypes.push_back(
        m.perTileOps[0]->getOperand(src.operandIdx).getType());
  }

  // Create the SubtiledRegionOp.
  auto subtileOp = builder.create<SubtiledRegionOp>(
      loc, /*resultTypes=*/TypeRange{},
      /*barriers=*/ValueRange{}, /*barrierPhases=*/ValueRange{},
      /*barrierAnnotations=*/builder.getArrayAttr({}));

  // --- Setup region ---
  {
    Block *setupBlock = builder.createBlock(&subtileOp.getSetupRegion());
    builder.setInsertionPointToStart(setupBlock);

    // Collect all setup ops.
    SmallVector<Operation *> setupOps;
    collectSetupOps(setupRoot, rootSplit, setupOps);

    // Clone setup ops into the setup region.
    IRMapping setupMapping;
    for (Operation *op : setupOps)
      builder.clone(*op, setupMapping);

    // Build setup yield operands: remap the flatSetupYields.
    SmallVector<Value> remappedYields;
    for (Value v : flatSetupYields)
      remappedYields.push_back(setupMapping.lookupOrDefault(v));

    builder.create<SubtiledRegionYieldOp>(loc, remappedYields);
  }

  // --- Tile region ---
  {
    Block *tileBlock = builder.createBlock(&subtileOp.getTileRegion());

    // Add block arguments.
    SmallVector<Location> argLocs(numTileArgs, loc);
    tileBlock->addArguments(tileArgTypes, argLocs);

    builder.setInsertionPointToStart(tileBlock);

    // Clone matched ops into tile region, substituting:
    // - The subtile leaf operand → block arg 0
    // - Intra-tile flow (previous result) → result of previous cloned op
    // - Extra varying operands → block args 1..N
    // - Captured operands → outer-scope values (referenced directly)

    Value prevTileResult = tileBlock->getArgument(0);
    unsigned extraArgIdx = 1; // index into tile block args for extra varying

    for (unsigned mIdx = 0; mIdx < matchedOps.size(); ++mIdx) {
      auto &m = matchedOps[mIdx];
      Operation *rep = m.repOp;

      // Build operand list for the cloned op.
      SmallVector<Value> operands;
      for (unsigned opIdx = 0; opIdx < rep->getNumOperands(); ++opIdx) {
        if (m.capturedOperands[opIdx]) {
          // Captured from outer scope.
          operands.push_back(m.capturedOperands[opIdx]);
          continue;
        }

        // Check if this is an intra-tile operand (from prevTileResult).
        bool isIntraTile = true;
        for (unsigned t = 0; t < numTiles; ++t) {
          Value expected;
          if (mIdx == 0)
            expected = subtileLeaves[t];
          else if (matchedOps[mIdx - 1].repOp->getNumResults() == 1)
            expected = matchedOps[mIdx - 1].perTileOps[t]->getResult(0);
          else
            expected = nullptr;
          if (m.perTileOps[t]->getOperand(opIdx) != expected) {
            isIntraTile = false;
            break;
          }
        }

        if (isIntraTile) {
          operands.push_back(prevTileResult);
        } else {
          // Extra varying operand — use the next tile block arg.
          operands.push_back(tileBlock->getArgument(extraArgIdx));
          extraArgIdx++;
        }
      }

      // Clone the op with the new operands.
      OperationState state(loc, rep->getName(), operands, rep->getResultTypes(),
                           rep->getAttrs());
      auto *cloned = builder.create(state);

      if (cloned->getNumResults() == 1)
        prevTileResult = cloned->getResult(0);
    }

    builder.create<SubtiledRegionYieldOp>(loc, ValueRange{});
  }

  // --- Teardown region (empty) ---
  // The teardown region is left empty (AnyRegion with 0 blocks).

  // --- Erase original ops ---
  // Erase per-tile ops in reverse order (last matched op first) so that
  // uses are removed before defs. Skip ops that still have external uses.
  for (int mIdx = matchedOps.size() - 1; mIdx >= 0; --mIdx) {
    for (auto *op : matchedOps[mIdx].perTileOps) {
      if (op->use_empty())
        op->erase();
    }
  }

  // Erase setup ops in reverse topological order. Skip ops that still have
  // uses outside the setup chain (e.g. a tmem_load whose token result is
  // consumed elsewhere).
  SmallVector<Operation *> setupOps;
  collectSetupOps(setupRoot, rootSplit, setupOps);
  for (auto it = setupOps.rbegin(); it != setupOps.rend(); ++it) {
    if ((*it)->use_empty())
      (*it)->erase();
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

} // anonymous namespace

namespace mlir {

void doAddSubtileRegions(triton::FuncOp &funcOp) {
  // Step 1: Find all SplitOps and identify root splits.
  SmallVector<SplitOp> rootSplits;
  funcOp.walk([&](SplitOp splitOp) {
    Operation *root = findSplitTreeRoot(splitOp);
    if (root) // nullptr means nested split, skip.
      rootSplits.push_back(splitOp);
  });

  LDBG("Found " << rootSplits.size() << " root split(s)");

  for (SplitOp rootSplit : rootSplits) {
    // Step 1: Find setup root.
    Operation *setupRoot = findSplitTreeRoot(rootSplit);
    if (!setupRoot)
      continue;

    LDBG("Root split: " << *rootSplit);
    LDBG("Setup root: " << *setupRoot);

    // Step 2: Collect subtile leaves.
    SmallVector<Value> leaves;
    collectLeaves(rootSplit, leaves);

    LDBG("Found " << leaves.size() << " subtile leaves");
    if (leaves.size() < 2)
      continue;

    // Step 3: Match forward ops.
    auto matchedOps = matchForwardOps(leaves);

    LDBG("Matched " << matchedOps.size() << " forward op(s)");
    if (matchedOps.empty())
      continue;

    // Step 3.5: Verify async partition consistency.
    if (failed(
            verifyAsyncPartitionConsistency(setupRoot, rootSplit, matchedOps)))
      continue;

    // Step 4: Build SubtiledRegionOp.
    buildSubtiledRegion(setupRoot, rootSplit, leaves, matchedOps);
  }
}

#define GEN_PASS_DEF_NVGPUTESTADDSUBTILEREGIONS
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestAddSubtileRegionsPass
    : public impl::NVGPUTestAddSubtileRegionsBase<
          NVGPUTestAddSubtileRegionsPass> {
public:
  using NVGPUTestAddSubtileRegionsBase::NVGPUTestAddSubtileRegionsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](triton::FuncOp funcOp) { doAddSubtileRegions(funcOp); });
  }
};

} // namespace mlir
