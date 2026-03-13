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
// TMA store matching
//===----------------------------------------------------------------------===//

/// Describes a matched TMA store sequence across all subtiles.
struct TMAStoreMatch {
  SmallVector<Operation *> perTileTMACopyOps;
  SmallVector<Operation *> perTileTokenWaitOps;
  /// For each non-src operand of the tma_copy (desc, coords):
  /// if the same across all subtiles, store the value (captured);
  /// if varying, store nullptr (needs setup yield / block arg).
  SmallVector<Value> capturedOperands;
  SmallVector<AsyncTaskId> asyncTaskIds;
};

/// Return the SMEM buffer type from the terminal op in a TMA store pattern.
/// The terminal is either a LocalAllocOp (with src) or a LocalStoreOp.
static Type getTerminalBufferType(Operation *termOp) {
  if (auto alloc = dyn_cast<LocalAllocOp>(termOp))
    return alloc.getResult().getType();
  return cast<LocalStoreOp>(termOp).getDst().getType();
}

/// Check whether the terminal ops from the matched pattern have TMA store
/// consumers (tma_copy → token_wait). The terminal can be a LocalAllocOp
/// (with src) or a LocalStoreOp.
static std::optional<TMAStoreMatch>
matchTMAStoreOps(ArrayRef<MatchedOp> matchedOps) {
  if (matchedOps.empty())
    return std::nullopt;

  auto &terminal = matchedOps.back();
  if (!terminal.isTerminal)
    return std::nullopt;

  // The terminal must be a store to SMEM: either a LocalAllocOp with src
  // or a LocalStoreOp.
  bool isAllocTerminal = isa<LocalAllocOp>(terminal.repOp);
  bool isStoreTerminal = isa<LocalStoreOp>(terminal.repOp);
  if (!isAllocTerminal && !isStoreTerminal)
    return std::nullopt;
  if (isAllocTerminal && !cast<LocalAllocOp>(terminal.repOp).getSrc())
    return std::nullopt;

  unsigned numTiles = terminal.perTileOps.size();
  TMAStoreMatch match;

  for (unsigned t = 0; t < numTiles; ++t) {
    auto *termOp = terminal.perTileOps[t];

    // Find the AsyncTMACopyLocalToGlobalOp that consumes the SMEM buffer.
    AsyncTMACopyLocalToGlobalOp tmaCopy;
    if (isAllocTerminal) {
      // LocalAllocOp: the memdesc result should have a single tma_copy user.
      Value allocResult = termOp->getResult(0);
      if (!allocResult.hasOneUse())
        return std::nullopt;
      tmaCopy = dyn_cast<AsyncTMACopyLocalToGlobalOp>(
          *allocResult.getUsers().begin());
    } else {
      // LocalStoreOp: find the tma_copy that uses the same dst memdesc.
      Value memdesc = cast<LocalStoreOp>(termOp).getDst();
      for (auto *user : memdesc.getUsers()) {
        if (user == termOp)
          continue;
        tmaCopy = dyn_cast<AsyncTMACopyLocalToGlobalOp>(user);
        if (tmaCopy)
          break;
      }
    }
    if (!tmaCopy)
      return std::nullopt;

    // Find TMAStoreTokenWaitOp that uses the tma_copy token.
    Value token = tmaCopy.getToken();
    if (!token || !token.hasOneUse())
      return std::nullopt;
    auto tokenWait = dyn_cast<TMAStoreTokenWaitOp>(*token.getUsers().begin());
    if (!tokenWait)
      return std::nullopt;

    // Must have no barriers.
    if (!tokenWait.getBarriers().empty())
      return std::nullopt;

    match.perTileTMACopyOps.push_back(tmaCopy);
    match.perTileTokenWaitOps.push_back(tokenWait);
  }

  // Verify structural identity across all subtiles.
  for (unsigned t = 1; t < numTiles; ++t) {
    if (match.perTileTMACopyOps[0]->getAttrs() !=
        match.perTileTMACopyOps[t]->getAttrs())
      return std::nullopt;
    if (match.perTileTMACopyOps[0]->getNumOperands() !=
        match.perTileTMACopyOps[t]->getNumOperands())
      return std::nullopt;
    if (match.perTileTokenWaitOps[0]->getAttrs() !=
        match.perTileTokenWaitOps[t]->getAttrs())
      return std::nullopt;
  }

  // Classify non-src operands (desc, coords) as captured vs varying.
  auto repCopy = cast<AsyncTMACopyLocalToGlobalOp>(match.perTileTMACopyOps[0]);

  // desc
  {
    Value repDesc = repCopy.getDesc();
    bool allSame = true;
    for (unsigned t = 1; t < numTiles; ++t) {
      auto otherCopy =
          cast<AsyncTMACopyLocalToGlobalOp>(match.perTileTMACopyOps[t]);
      if (otherCopy.getDesc() != repDesc) {
        allSame = false;
        break;
      }
    }
    match.capturedOperands.push_back(allSame ? repDesc : nullptr);
  }

  // coords
  for (unsigned c = 0; c < repCopy.getCoord().size(); ++c) {
    Value repCoord = repCopy.getCoord()[c];
    bool allSame = true;
    for (unsigned t = 1; t < numTiles; ++t) {
      auto otherCopy =
          cast<AsyncTMACopyLocalToGlobalOp>(match.perTileTMACopyOps[t]);
      if (otherCopy.getCoord()[c] != repCoord) {
        allSame = false;
        break;
      }
    }
    match.capturedOperands.push_back(allSame ? repCoord : nullptr);
  }

  // Collect async_task_ids from the TMA store ops and verify consistency.
  match.asyncTaskIds = getAsyncTaskIds(match.perTileTMACopyOps[0]);
  for (unsigned t = 1; t < numTiles; ++t) {
    if (getAsyncTaskIds(match.perTileTMACopyOps[t]) != match.asyncTaskIds)
      return std::nullopt;
  }

  LDBG("Matched TMA store sequence for " << numTiles << " subtile(s)");
  return match;
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
///
/// When \p tmaMatch is provided (same-task case), the tile body is extended
/// to include TMA store ops with SMEM buffer reuse: the terminal local_alloc
/// is replaced by local_store + tma_copy + wait.
static void buildSubtiledRegion(Operation *setupRoot, SplitOp rootSplit,
                                ArrayRef<Value> subtileLeaves,
                                ArrayRef<MatchedOp> matchedOps,
                                const TMAStoreMatch *tmaMatch = nullptr) {
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

  // Track which values in each tile flow from the previous matched op result.
  // prevResults[tileIdx] = result of the previous matched op for that tile.
  SmallVector<Value> prevResults(subtileLeaves.begin(), subtileLeaves.end());

  // Collect per-tile varying operand values that need to come from setup.
  // setupYieldValues[tileIdx] = values to yield for that tile.
  SmallVector<SmallVector<Value>> setupYieldValues(numTiles);

  // Number of matched ops to include in the tile body clone loop.
  // When TMA match is present, we exclude the terminal SMEM write op.
  unsigned matchedOpsToClone =
      tmaMatch ? matchedOps.size() - 1 : matchedOps.size();

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

  // When TMA match is present, add the SMEM buffer and varying TMA operands
  // to the tile args. The buffer is shared across tiles (same local_alloc),
  // represented as a null placeholder in setupYieldValues (replaced during
  // setup region construction with the actual alloc result).
  if (tmaMatch) {
    for (unsigned t = 0; t < numTiles; ++t) {
      // Buffer memdesc placeholder (null — created inside setup region).
      setupYieldValues[t].push_back(Value());

      // Varying desc/coord operands.
      auto tmaCopy =
          cast<AsyncTMACopyLocalToGlobalOp>(tmaMatch->perTileTMACopyOps[t]);
      unsigned capturedIdx = 0;
      // desc
      if (!tmaMatch->capturedOperands[capturedIdx])
        setupYieldValues[t].push_back(tmaCopy.getDesc());
      capturedIdx++;
      // coords
      for (unsigned c = 0; c < tmaCopy.getCoord().size(); ++c) {
        if (!tmaMatch->capturedOperands[capturedIdx])
          setupYieldValues[t].push_back(tmaCopy.getCoord()[c]);
        capturedIdx++;
      }
    }
  }

  // Number of tile block args.
  unsigned numTileArgs = setupYieldValues[0].size();

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

  if (tmaMatch) {
    // Buffer memdesc type.
    tileArgTypes.push_back(getTerminalBufferType(matchedOps.back().repOp));

    // Varying TMA operand types.
    auto repCopy =
        cast<AsyncTMACopyLocalToGlobalOp>(tmaMatch->perTileTMACopyOps[0]);
    unsigned capturedIdx = 0;
    if (!tmaMatch->capturedOperands[capturedIdx])
      tileArgTypes.push_back(repCopy.getDesc().getType());
    capturedIdx++;
    for (unsigned c = 0; c < repCopy.getCoord().size(); ++c) {
      if (!tmaMatch->capturedOperands[capturedIdx])
        tileArgTypes.push_back(repCopy.getCoord()[c].getType());
      capturedIdx++;
    }
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

    // If TMA match, create the mutable buffer local_alloc (no src).
    Value bufAllocResult;
    if (tmaMatch) {
      auto allocType = getTerminalBufferType(matchedOps.back().repOp);
      bufAllocResult = builder.create<LocalAllocOp>(loc, allocType);
    }

    // Build setup yield operands: remap the flatSetupYields.
    // Null values are buffer placeholders (replaced with bufAllocResult).
    SmallVector<Value> remappedYields;
    for (Value v : flatSetupYields) {
      if (!v)
        remappedYields.push_back(bufAllocResult);
      else
        remappedYields.push_back(setupMapping.lookupOrDefault(v));
    }

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

    for (unsigned mIdx = 0; mIdx < matchedOpsToClone; ++mIdx) {
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

    // When TMA match is present, emit the TMA store sequence instead of
    // the terminal local_alloc.
    if (tmaMatch) {
      // Buffer block arg comes after subtile leaf + extra varying args.
      unsigned bufArgIdx = 1 + tileArgSources.size();
      Value bufArg = tileBlock->getArgument(bufArgIdx);

      // local_store %prev_result, %buf_arg
      builder.create<LocalStoreOp>(loc, prevTileResult, bufArg);

      // Build tma_copy operands (desc, coords).
      auto repCopy =
          cast<AsyncTMACopyLocalToGlobalOp>(tmaMatch->perTileTMACopyOps[0]);

      unsigned tmaArgIdx = bufArgIdx + 1;
      unsigned capturedIdx = 0;

      // desc
      Value desc;
      if (tmaMatch->capturedOperands[capturedIdx]) {
        desc = tmaMatch->capturedOperands[capturedIdx];
      } else {
        desc = tileBlock->getArgument(tmaArgIdx++);
      }
      capturedIdx++;

      // coords
      SmallVector<Value> coords;
      for (unsigned c = 0; c < repCopy.getCoord().size(); ++c) {
        if (tmaMatch->capturedOperands[capturedIdx]) {
          coords.push_back(tmaMatch->capturedOperands[capturedIdx]);
        } else {
          coords.push_back(tileBlock->getArgument(tmaArgIdx++));
        }
        capturedIdx++;
      }

      // async_tma_copy_local_to_global
      auto tokenType = AsyncTokenType::get(builder.getContext());
      auto tmaCopy = builder.create<AsyncTMACopyLocalToGlobalOp>(
          loc, tokenType, desc, coords, bufArg, repCopy.getEvict());

      // tma_store_token_wait
      builder.create<TMAStoreTokenWaitOp>(loc, tmaCopy.getToken(),
                                          /*barriers=*/ValueRange{},
                                          /*barrier_preds=*/ValueRange{},
                                          /*nvws_tokens=*/ValueRange{},
                                          /*nvws_token_indices=*/ValueRange{});
    }

    builder.create<SubtiledRegionYieldOp>(loc, ValueRange{});
  }

  // --- Teardown region (empty) ---
  // The teardown region is left empty (AnyRegion with 0 blocks).

  // --- Erase original ops ---
  // When TMA match is present (same-task), erase all TMA store ops.
  if (tmaMatch) {
    for (auto *op : tmaMatch->perTileTokenWaitOps)
      if (op->use_empty())
        op->erase();
    for (auto *op : tmaMatch->perTileTMACopyOps)
      if (op->use_empty())
        op->erase();
  }

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

/// Build a second SubtiledRegionOp for TMA stores when they belong to a
/// different async task than the epilogue compute ops.
static void buildTMAStoreSubtiledRegion(ArrayRef<MatchedOp> matchedOps,
                                        const TMAStoreMatch &tmaMatch) {
  unsigned numTiles = matchedOps.back().perTileOps.size();

  // Insert after the last TMA token wait op so that all local_alloc results
  // (which the setup region references) dominate the subtile region.
  Operation *lastTMAOp = tmaMatch.perTileTokenWaitOps[0];
  for (auto *op : tmaMatch.perTileTokenWaitOps) {
    if (lastTMAOp->isBeforeInBlock(op))
      lastTMAOp = op;
  }
  OpBuilder builder(lastTMAOp);
  builder.setInsertionPointAfter(lastTMAOp);
  Location loc = matchedOps.back().repOp->getLoc();

  // Setup yields per tile: local_alloc memdesc + varying TMA operands.
  SmallVector<SmallVector<Value>> setupYieldValues(numTiles);
  SmallVector<Type> tileArgTypes;

  // First arg: memdesc from the original (surviving) terminal op.
  auto getTerminalMemdesc = [](Operation *op) -> Value {
    if (auto alloc = dyn_cast<LocalAllocOp>(op))
      return alloc.getResult();
    return cast<LocalStoreOp>(op).getDst();
  };
  tileArgTypes.push_back(
      getTerminalMemdesc(matchedOps.back().perTileOps[0]).getType());
  for (unsigned t = 0; t < numTiles; ++t)
    setupYieldValues[t].push_back(
        getTerminalMemdesc(matchedOps.back().perTileOps[t]));

  // Varying TMA operands.
  auto repCopy =
      cast<AsyncTMACopyLocalToGlobalOp>(tmaMatch.perTileTMACopyOps[0]);
  unsigned capturedIdx = 0;
  // desc
  if (!tmaMatch.capturedOperands[capturedIdx]) {
    tileArgTypes.push_back(repCopy.getDesc().getType());
    for (unsigned t = 0; t < numTiles; ++t) {
      auto copy =
          cast<AsyncTMACopyLocalToGlobalOp>(tmaMatch.perTileTMACopyOps[t]);
      setupYieldValues[t].push_back(copy.getDesc());
    }
  }
  capturedIdx++;
  // coords
  for (unsigned c = 0; c < repCopy.getCoord().size(); ++c) {
    if (!tmaMatch.capturedOperands[capturedIdx]) {
      tileArgTypes.push_back(repCopy.getCoord()[c].getType());
      for (unsigned t = 0; t < numTiles; ++t) {
        auto copy =
            cast<AsyncTMACopyLocalToGlobalOp>(tmaMatch.perTileTMACopyOps[t]);
        setupYieldValues[t].push_back(copy.getCoord()[c]);
      }
    }
    capturedIdx++;
  }

  unsigned numTileArgs = tileArgTypes.size();

  // Build flat setup yields (tile-major order).
  SmallVector<Value> flatSetupYields;
  for (unsigned t = 0; t < numTiles; ++t)
    for (auto &v : setupYieldValues[t])
      flatSetupYields.push_back(v);

  // Create SubtiledRegionOp.
  auto subtileOp = builder.create<SubtiledRegionOp>(
      loc, /*resultTypes=*/TypeRange{},
      /*barriers=*/ValueRange{}, /*barrierPhases=*/ValueRange{},
      /*barrierAnnotations=*/builder.getArrayAttr({}));

  // --- Setup region ---
  {
    Block *setupBlock = builder.createBlock(&subtileOp.getSetupRegion());
    builder.setInsertionPointToStart(setupBlock);
    // No ops to clone — just yield the values directly from outer scope.
    builder.create<SubtiledRegionYieldOp>(loc, flatSetupYields);
  }

  // --- Tile region ---
  {
    Block *tileBlock = builder.createBlock(&subtileOp.getTileRegion());
    SmallVector<Location> argLocs(numTileArgs, loc);
    tileBlock->addArguments(tileArgTypes, argLocs);
    builder.setInsertionPointToStart(tileBlock);

    // arg 0 = memdesc from local_alloc
    Value bufArg = tileBlock->getArgument(0);
    unsigned argIdx = 1;

    // desc
    capturedIdx = 0;
    Value desc;
    if (tmaMatch.capturedOperands[capturedIdx]) {
      desc = tmaMatch.capturedOperands[capturedIdx];
    } else {
      desc = tileBlock->getArgument(argIdx++);
    }
    capturedIdx++;

    // coords
    SmallVector<Value> coords;
    for (unsigned c = 0; c < repCopy.getCoord().size(); ++c) {
      if (tmaMatch.capturedOperands[capturedIdx]) {
        coords.push_back(tmaMatch.capturedOperands[capturedIdx]);
      } else {
        coords.push_back(tileBlock->getArgument(argIdx++));
      }
      capturedIdx++;
    }

    // async_tma_copy_local_to_global
    auto tokenType = AsyncTokenType::get(builder.getContext());
    auto tmaCopy = builder.create<AsyncTMACopyLocalToGlobalOp>(
        loc, tokenType, desc, coords, bufArg, repCopy.getEvict());

    // tma_store_token_wait
    builder.create<TMAStoreTokenWaitOp>(loc, tmaCopy.getToken(),
                                        /*barriers=*/ValueRange{},
                                        /*barrier_preds=*/ValueRange{},
                                        /*nvws_tokens=*/ValueRange{},
                                        /*nvws_token_indices=*/ValueRange{});

    builder.create<SubtiledRegionYieldOp>(loc, ValueRange{});
  }

  // Erase original TMA store ops.
  for (auto *op : tmaMatch.perTileTokenWaitOps)
    op->erase();
  for (auto *op : tmaMatch.perTileTMACopyOps)
    op->erase();
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

    // Step 3.6: Check for TMA store consumers of the terminal local_alloc.
    auto tmaMatch = matchTMAStoreOps(matchedOps);
    if (tmaMatch) {
      // Determine reference task IDs from the matched ops.
      SmallVector<AsyncTaskId> referenceIds;
      for (auto &m : matchedOps) {
        auto ids = getAsyncTaskIds(m.perTileOps[0]);
        if (!ids.empty()) {
          referenceIds = ids;
          break;
        }
      }

      bool sameTask = tmaMatch->asyncTaskIds.empty() || referenceIds.empty() ||
                      tmaMatch->asyncTaskIds == referenceIds;

      if (sameTask) {
        // Build single extended subtile region with TMA store + buffer reuse.
        buildSubtiledRegion(setupRoot, rootSplit, leaves, matchedOps,
                            &*tmaMatch);
      } else {
        // Build first subtile region (local_allocs survive).
        buildSubtiledRegion(setupRoot, rootSplit, leaves, matchedOps);
        // Build second subtile region for TMA stores.
        buildTMAStoreSubtiledRegion(matchedOps, *tmaMatch);
      }
    } else {
      // No TMA store match — existing behavior.
      buildSubtiledRegion(setupRoot, rootSplit, leaves, matchedOps);
    }
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
