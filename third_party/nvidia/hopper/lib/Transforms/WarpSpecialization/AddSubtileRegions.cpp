#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
  auto repAlloc = dyn_cast<LocalAllocOp>(terminal.repOp);
  bool isAllocTerminal = repAlloc && repAlloc.getSrc();
  if (!isAllocTerminal && !isa<LocalStoreOp>(terminal.repOp))
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

  // Returns repVal if the same across all subtiles, nullptr otherwise.
  auto classifyOperand = [&](Value repVal,
                             function_ref<Value(Operation *)> getVal) -> Value {
    for (unsigned t = 1; t < numTiles; ++t) {
      if (getVal(match.perTileTMACopyOps[t]) != repVal)
        return nullptr;
    }
    return repVal;
  };

  match.capturedOperands.push_back(
      classifyOperand(repCopy.getDesc(), [](Operation *op) {
        return cast<AsyncTMACopyLocalToGlobalOp>(op).getDesc();
      }));
  for (unsigned c = 0; c < repCopy.getCoord().size(); ++c) {
    match.capturedOperands.push_back(
        classifyOperand(repCopy.getCoord()[c], [c](Operation *op) {
          return cast<AsyncTMACopyLocalToGlobalOp>(op).getCoord()[c];
        }));
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
// Partition and schedule consistency
//===----------------------------------------------------------------------===//

/// Verify that all setup and matched ops share the same async_task_id,
/// loop.stage, and loop.cluster. Emits an error on \p rootSplit and returns
/// failure if not.
static LogicalResult
verifyPartitionAndScheduleConsistency(Operation *setupRoot, SplitOp rootSplit,
                                      ArrayRef<MatchedOp> matchedOps) {
  SmallVector<Operation *> setupOps;
  collectSetupOps(setupRoot, rootSplit, setupOps);

  SmallVector<AsyncTaskId> referenceIds;
  bool foundAsyncRef = false;

  IntegerAttr referenceStage;
  bool foundStageRef = false;

  IntegerAttr referenceCluster;
  bool foundClusterRef = false;

  auto checkOp = [&](Operation *op) -> std::optional<std::string> {
    // Check async_task_id.
    auto ids = getAsyncTaskIds(op);
    if (!ids.empty()) {
      if (!foundAsyncRef) {
        referenceIds = std::move(ids);
        foundAsyncRef = true;
      } else if (ids != referenceIds) {
        return std::string(
            "ops in subtile region have inconsistent async_task_id partitions");
      }
    }

    // Check loop.stage.
    if (op->hasAttr(tt::kLoopStageAttrName)) {
      auto stage = op->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
      if (!foundStageRef) {
        referenceStage = stage;
        foundStageRef = true;
      } else if (stage != referenceStage) {
        return std::string(
            "ops in subtile region have inconsistent loop.stage attributes");
      }
    }

    // Check loop.cluster.
    if (op->hasAttr(tt::kLoopClusterAttrName)) {
      auto cluster = op->getAttrOfType<IntegerAttr>(tt::kLoopClusterAttrName);
      if (!foundClusterRef) {
        referenceCluster = cluster;
        foundClusterRef = true;
      } else if (cluster != referenceCluster) {
        return std::string(
            "ops in subtile region have inconsistent loop.cluster attributes");
      }
    }

    return std::nullopt;
  };

  for (Operation *op : setupOps)
    if (auto err = checkOp(op))
      return rootSplit.emitError(*err);
  for (auto &m : matchedOps)
    for (Operation *op : m.perTileOps)
      if (auto err = checkOp(op))
        return rootSplit.emitError(*err);
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

  // Collect reference attributes (async_task_id, loop.stage, loop.cluster)
  // from the first element that has them.
  SmallVector<AsyncTaskId> refAsyncTaskIds;
  IntegerAttr refLoopStage;
  IntegerAttr refLoopCluster;
  SmallVector<Operation *> refOps;
  collectSetupOps(setupRoot, rootSplit, refOps);
  for (auto &m : matchedOps)
    refOps.append(m.perTileOps.begin(), m.perTileOps.end());
  for (Operation *op : refOps) {
    if (refAsyncTaskIds.empty()) {
      auto ids = getAsyncTaskIds(op);
      if (!ids.empty())
        refAsyncTaskIds = std::move(ids);
    }
    if (!refLoopStage && op->hasAttr(tt::kLoopStageAttrName))
      refLoopStage = op->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
    if (!refLoopCluster && op->hasAttr(tt::kLoopClusterAttrName))
      refLoopCluster = op->getAttrOfType<IntegerAttr>(tt::kLoopClusterAttrName);
  }

  // Helper to apply reference attributes to a newly created op.
  auto applyRefAttrs = [&](Operation *op) {
    if (!refAsyncTaskIds.empty())
      setAsyncTaskIds(op, refAsyncTaskIds);
    if (refLoopStage)
      op->setAttr(tt::kLoopStageAttrName, refLoopStage);
    if (refLoopCluster)
      op->setAttr(tt::kLoopClusterAttrName, refLoopCluster);
  };

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
  applyRefAttrs(subtileOp);

  // --- Setup region ---
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
    auto allocOp = builder.create<LocalAllocOp>(loc, allocType);
    applyRefAttrs(allocOp);
    bufAllocResult = allocOp;
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

  applyRefAttrs(builder.create<SubtiledRegionYieldOp>(loc, remappedYields));

  // --- Tile region ---
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
  // the terminal SMEM write op.
  if (tmaMatch) {
    // Buffer block arg comes after subtile leaf + extra varying args.
    unsigned bufArgIdx = 1 + tileArgSources.size();
    Value bufArg = tileBlock->getArgument(bufArgIdx);

    // local_store %prev_result, %buf_arg
    applyRefAttrs(builder.create<LocalStoreOp>(loc, prevTileResult, bufArg));

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
    applyRefAttrs(tmaCopy);

    // tma_store_token_wait
    auto tokenWait = builder.create<TMAStoreTokenWaitOp>(
        loc, tmaCopy.getToken(),
        /*barriers=*/ValueRange{},
        /*barrier_preds=*/ValueRange{},
        /*nvws_tokens=*/ValueRange{},
        /*nvws_token_indices=*/ValueRange{});
    applyRefAttrs(tokenWait);
  }

  applyRefAttrs(builder.create<SubtiledRegionYieldOp>(loc, ValueRange{}));

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
  SmallVector<Operation *> setupOpsToErase;
  collectSetupOps(setupRoot, rootSplit, setupOpsToErase);
  for (auto it = setupOpsToErase.rbegin(); it != setupOpsToErase.rend(); ++it) {
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

  // Collect reference attributes from TMA store ops.
  SmallVector<AsyncTaskId> refAsyncTaskIds = tmaMatch.asyncTaskIds;
  IntegerAttr refLoopStage;
  IntegerAttr refLoopCluster;
  for (auto *op : tmaMatch.perTileTMACopyOps) {
    if (!refLoopStage && op->hasAttr(tt::kLoopStageAttrName))
      refLoopStage = op->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
    if (!refLoopCluster && op->hasAttr(tt::kLoopClusterAttrName))
      refLoopCluster = op->getAttrOfType<IntegerAttr>(tt::kLoopClusterAttrName);
    if (refLoopStage && refLoopCluster)
      break;
  }

  // Helper to apply reference attributes to a newly created op.
  auto applyRefAttrs = [&](Operation *op) {
    if (!refAsyncTaskIds.empty())
      setAsyncTaskIds(op, refAsyncTaskIds);
    if (refLoopStage)
      op->setAttr(tt::kLoopStageAttrName, refLoopStage);
    if (refLoopCluster)
      op->setAttr(tt::kLoopClusterAttrName, refLoopCluster);
  };

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
  applyRefAttrs(subtileOp);

  // --- Setup region ---
  Block *setupBlock = builder.createBlock(&subtileOp.getSetupRegion());
  builder.setInsertionPointToStart(setupBlock);
  // No ops to clone — just yield the values directly from outer scope.
  applyRefAttrs(builder.create<SubtiledRegionYieldOp>(loc, flatSetupYields));

  // --- Tile region ---
  Block *tileBlock = builder.createBlock(&subtileOp.getTileRegion());
  SmallVector<Location> argLocs(numTileArgs, loc);
  tileBlock->addArguments(tileArgTypes, argLocs);
  builder.setInsertionPointToStart(tileBlock);

  // arg 0 = memdesc from terminal op
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
  applyRefAttrs(tmaCopy);

  // tma_store_token_wait
  auto tokenWait =
      builder.create<TMAStoreTokenWaitOp>(loc, tmaCopy.getToken(),
                                          /*barriers=*/ValueRange{},
                                          /*barrier_preds=*/ValueRange{},
                                          /*nvws_tokens=*/ValueRange{},
                                          /*nvws_token_indices=*/ValueRange{});
  applyRefAttrs(tokenWait);

  applyRefAttrs(builder.create<SubtiledRegionYieldOp>(loc, ValueRange{}));

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

    // Step 3.5: Verify partition and schedule consistency.
    if (failed(verifyPartitionAndScheduleConsistency(setupRoot, rootSplit,
                                                     matchedOps)))
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

//===----------------------------------------------------------------------===//
// doFuseSubtileRegions
//===----------------------------------------------------------------------===//

/// Return the number of tile arguments for a SubtiledRegionOp by inspecting
/// the tile region's block arguments.
static unsigned getNumTileArgs(SubtiledRegionOp op) {
  Block &tileBlock = op.getTileRegion().front();
  return tileBlock.getNumArguments();
}

/// Return the number of tiles by dividing setup yields by tile args.
static unsigned getNumTiles(SubtiledRegionOp op) {
  auto yieldOp =
      cast<SubtiledRegionYieldOp>(op.getSetupRegion().front().getTerminator());
  unsigned numTileArgs = getNumTileArgs(op);
  assert(numTileArgs > 0 && "tile region must have at least one argument");
  return yieldOp.getNumOperands() / numTileArgs;
}

/// Check whether two SubtiledRegionOps are fusible.
static bool areFusible(SubtiledRegionOp a, SubtiledRegionOp b) {
  // Must be in the same block.
  if (a->getBlock() != b->getBlock())
    return false;

  // Must have the same number of tiles.
  if (getNumTiles(a) != getNumTiles(b))
    return false;

  // Both must have empty barriers (we run before doAnnotateSubtileBarriers).
  if (a.getBarriers().size() != 0 || b.getBarriers().size() != 0)
    return false;

  // Both must have empty teardown regions.
  if (!a.getTeardownRegion().empty() || !b.getTeardownRegion().empty())
    return false;

  // Compatible async_task_id.
  if (getAsyncTaskIds(a) != getAsyncTaskIds(b))
    return false;

  // Compatible loop.stage.
  auto stageA = a->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
  auto stageB = b->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
  if (stageA != stageB)
    return false;

  // Compatible loop.cluster.
  auto clusterA = a->getAttrOfType<IntegerAttr>(tt::kLoopClusterAttrName);
  auto clusterB = b->getAttrOfType<IntegerAttr>(tt::kLoopClusterAttrName);
  if (clusterA != clusterB)
    return false;

  // Must be adjacent: no intervening ops with memory effects.
  Operation *cursor = a->getNextNode();
  while (cursor && cursor != b.getOperation()) {
    if (!isMemoryEffectFree(cursor))
      return false;
    cursor = cursor->getNextNode();
  }
  // b must actually follow a in the same block.
  if (cursor != b.getOperation())
    return false;

  return true;
}

/// Fuse two adjacent SubtiledRegionOps into one.
static SubtiledRegionOp fuseSubtiledRegions(SubtiledRegionOp a,
                                            SubtiledRegionOp b) {
  unsigned numTiles = getNumTiles(a);
  unsigned nA = getNumTileArgs(a);
  unsigned nB = getNumTileArgs(b);
  unsigned nFused = nA + nB;

  OpBuilder builder(b);
  Location loc = a.getLoc();

  // Create the fused SubtiledRegionOp.
  auto fusedOp = builder.create<SubtiledRegionOp>(
      loc, /*resultTypes=*/TypeRange{},
      /*barriers=*/ValueRange{}, /*barrierPhases=*/ValueRange{},
      /*barrierAnnotations=*/builder.getArrayAttr({}));

  // Copy attributes from A.
  for (auto attr : a->getAttrs()) {
    if (attr.getName() == "barrierAnnotations" ||
        attr.getName() == "operandSegmentSizes")
      continue;
    fusedOp->setAttr(attr.getName(), attr.getValue());
  }

  // --- Fused setup region ---
  Block *setupBlock = builder.createBlock(&fusedOp.getSetupRegion());
  builder.setInsertionPointToStart(setupBlock);

  // Clone A's setup ops (except terminator).
  IRMapping mappingA;
  Block &setupA = a.getSetupRegion().front();
  for (auto &op : setupA.without_terminator())
    builder.clone(op, mappingA);

  // Clone B's setup ops (except terminator).
  IRMapping mappingB;
  Block &setupB = b.getSetupRegion().front();
  for (auto &op : setupB.without_terminator())
    builder.clone(op, mappingB);

  // Build interleaved yield: for each tile t, yield A's args then B's args.
  auto yieldA = cast<SubtiledRegionYieldOp>(setupA.getTerminator());
  auto yieldB = cast<SubtiledRegionYieldOp>(setupB.getTerminator());

  SmallVector<Value> fusedYieldValues;
  for (unsigned t = 0; t < numTiles; ++t) {
    for (unsigned i = 0; i < nA; ++i)
      fusedYieldValues.push_back(
          mappingA.lookupOrDefault(yieldA.getOperand(t * nA + i)));
    for (unsigned i = 0; i < nB; ++i)
      fusedYieldValues.push_back(
          mappingB.lookupOrDefault(yieldB.getOperand(t * nB + i)));
  }

  auto fusedSetupYield =
      builder.create<SubtiledRegionYieldOp>(loc, fusedYieldValues);
  // Copy attributes from A's yield.
  for (auto attr : yieldA->getAttrs())
    fusedSetupYield->setAttr(attr.getName(), attr.getValue());

  // --- Fused tile region ---
  Block *tileBlock = builder.createBlock(&fusedOp.getTileRegion());
  Block &tileA = a.getTileRegion().front();
  Block &tileB = b.getTileRegion().front();

  // Add nA + nB block arguments.
  SmallVector<Type> tileArgTypes;
  SmallVector<Location> tileArgLocs;
  for (unsigned i = 0; i < nA; ++i) {
    tileArgTypes.push_back(tileA.getArgument(i).getType());
    tileArgLocs.push_back(tileA.getArgument(i).getLoc());
  }
  for (unsigned i = 0; i < nB; ++i) {
    tileArgTypes.push_back(tileB.getArgument(i).getType());
    tileArgLocs.push_back(tileB.getArgument(i).getLoc());
  }
  tileBlock->addArguments(tileArgTypes, tileArgLocs);

  builder.setInsertionPointToStart(tileBlock);

  // Clone A's tile body, mapping A's block args to first nA args.
  IRMapping tileMapA;
  for (unsigned i = 0; i < nA; ++i)
    tileMapA.map(tileA.getArgument(i), tileBlock->getArgument(i));
  for (auto &op : tileA.without_terminator())
    builder.clone(op, tileMapA);

  // Clone B's tile body, mapping B's block args to next nB args.
  IRMapping tileMapB;
  for (unsigned i = 0; i < nB; ++i)
    tileMapB.map(tileB.getArgument(i), tileBlock->getArgument(nA + i));
  for (auto &op : tileB.without_terminator())
    builder.clone(op, tileMapB);

  // Terminate with empty yield.
  auto fusedTileYield =
      builder.create<SubtiledRegionYieldOp>(loc, ValueRange{});
  // Copy attributes from A's tile yield.
  auto tileYieldA = cast<SubtiledRegionYieldOp>(tileA.getTerminator());
  for (auto attr : tileYieldA->getAttrs())
    fusedTileYield->setAttr(attr.getName(), attr.getValue());

  // Erase originals.
  a.erase();
  b.erase();

  return fusedOp;
}

void doFuseSubtileRegions(triton::FuncOp &funcOp) {
  SmallVector<SubtiledRegionOp> subtileOps;
  funcOp.walk([&](SubtiledRegionOp op) { subtileOps.push_back(op); });

  if (subtileOps.size() < 2)
    return;

  // Iterate through adjacent pairs, fusing when possible.
  unsigned i = 0;
  while (i + 1 < subtileOps.size()) {
    SubtiledRegionOp a = subtileOps[i];
    SubtiledRegionOp b = subtileOps[i + 1];

    if (areFusible(a, b)) {
      SubtiledRegionOp fused = fuseSubtiledRegions(a, b);
      // Replace both with the fused op in the list.
      subtileOps[i] = fused;
      subtileOps.erase(subtileOps.begin() + i + 1);
      // Don't advance i — allow chain-fusion.
    } else {
      ++i;
    }
  }
}

#define GEN_PASS_DEF_NVGPUTESTFUSESUBTILEREGIONS
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestFuseSubtileRegionsPass
    : public impl::NVGPUTestFuseSubtileRegionsBase<
          NVGPUTestFuseSubtileRegionsPass> {
public:
  using NVGPUTestFuseSubtileRegionsBase::NVGPUTestFuseSubtileRegionsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](triton::FuncOp funcOp) { doFuseSubtileRegions(funcOp); });
  }
};

void doAnnotateSubtileBarriers(triton::FuncOp &funcOp) {
  SmallVector<SubtiledRegionOp> subtileOps;
  funcOp.walk([&](SubtiledRegionOp op) { subtileOps.push_back(op); });

  for (auto subtileOp : subtileOps) {
    Block *parentBlock = subtileOp->getBlock();
    if (!parentBlock)
      continue;

    SmallVector<Value> newBarriers;
    SmallVector<Value> newPhases;
    SmallVector<Attribute> newAnnotations;

    // Scan backward for adjacent WaitBarrierOps.
    SmallVector<WaitBarrierOp> waitOps;
    Operation *prev = subtileOp->getPrevNode();
    while (prev && isa<WaitBarrierOp>(prev)) {
      waitOps.push_back(cast<WaitBarrierOp>(prev));
      prev = prev->getPrevNode();
    }
    // Reverse so they appear in original order.
    std::reverse(waitOps.begin(), waitOps.end());

    // Scan forward for adjacent ArriveBarrierOps.
    SmallVector<ArriveBarrierOp> arriveOps;
    Operation *next = subtileOp->getNextNode();
    while (next && isa<ArriveBarrierOp>(next)) {
      arriveOps.push_back(cast<ArriveBarrierOp>(next));
      next = next->getNextNode();
    }

    if (waitOps.empty() && arriveOps.empty())
      continue;

    // Determine the target op index for annotations. We use index 0
    // (first op in tile body) for BEFORE annotations and the last op
    // index for AFTER annotations.
    Block &tileBlock = subtileOp.getTileRegion().front();
    unsigned numTileOps = 0;
    for (auto &op : tileBlock.without_terminator())
      ++numTileOps;
    unsigned lastOpIdx = numTileOps > 0 ? numTileOps - 1 : 0;

    OpBuilder builder(subtileOp->getContext());

    // Absorb wait barriers.
    for (auto waitOp : waitOps) {
      unsigned barrierIdx = newBarriers.size();
      newBarriers.push_back(waitOp.getAlloc());
      newPhases.push_back(waitOp.getPhase());
      newAnnotations.push_back(BarrierAnnotationAttr::get(
          subtileOp->getContext(), barrierIdx, BarrierPlacement::BEFORE,
          /*targetOpIdx=*/0, builder.getStringAttr("wait_barrier"),
          /*count=*/1));
    }

    // Absorb arrive barriers.
    for (auto arriveOp : arriveOps) {
      unsigned barrierIdx = newBarriers.size();
      newBarriers.push_back(arriveOp.getAlloc());
      // Arrive doesn't use phase; use a placeholder constant.
      auto phaseConst = builder.create<arith::ConstantOp>(
          arriveOp.getLoc(), builder.getI32IntegerAttr(0));
      phaseConst->moveBefore(subtileOp);
      newPhases.push_back(phaseConst);
      newAnnotations.push_back(BarrierAnnotationAttr::get(
          subtileOp->getContext(), barrierIdx, BarrierPlacement::AFTER,
          /*targetOpIdx=*/lastOpIdx, builder.getStringAttr("arrive_barrier"),
          /*count=*/arriveOp.getCount()));
    }

    // Merge with any existing barriers/annotations on the op.
    SmallVector<Value> allBarriers(subtileOp.getBarriers().begin(),
                                   subtileOp.getBarriers().end());
    SmallVector<Value> allPhases(subtileOp.getBarrierPhases().begin(),
                                 subtileOp.getBarrierPhases().end());
    SmallVector<Attribute> allAnnotations;
    for (auto attr : subtileOp.getBarrierAnnotations())
      allAnnotations.push_back(attr);

    // Adjust barrier indices for new annotations (offset by existing count).
    unsigned existingCount = allBarriers.size();
    for (auto &ann : newAnnotations) {
      auto a = cast<BarrierAnnotationAttr>(ann);
      ann = BarrierAnnotationAttr::get(subtileOp->getContext(),
                                       a.getBarrierIdx() + existingCount,
                                       a.getPlacement(), a.getTargetOpIdx(),
                                       a.getBarrierOpKind(), a.getCount());
    }

    allBarriers.append(newBarriers.begin(), newBarriers.end());
    allPhases.append(newPhases.begin(), newPhases.end());
    allAnnotations.append(newAnnotations.begin(), newAnnotations.end());

    // Replace the SubtiledRegionOp with a new one that has the barriers.
    OpBuilder replaceBuilder(subtileOp);
    auto newOp = replaceBuilder.create<SubtiledRegionOp>(
        subtileOp.getLoc(), subtileOp.getResultTypes(), allBarriers, allPhases,
        replaceBuilder.getArrayAttr(allAnnotations));

    // Copy attributes (async_task_id, loop.stage, etc).
    for (auto attr : subtileOp->getAttrs()) {
      if (attr.getName() == "barrierAnnotations" ||
          attr.getName() == "operandSegmentSizes")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }

    // Move regions from old op to new op.
    newOp.getSetupRegion().takeBody(subtileOp.getSetupRegion());
    newOp.getTileRegion().takeBody(subtileOp.getTileRegion());
    if (!subtileOp.getTeardownRegion().empty())
      newOp.getTeardownRegion().takeBody(subtileOp.getTeardownRegion());

    // Replace results.
    subtileOp.replaceAllUsesWith(newOp.getResults());
    subtileOp.erase();

    // Erase original barrier ops.
    for (auto waitOp : waitOps)
      waitOp.erase();
    for (auto arriveOp : arriveOps)
      arriveOp.erase();
  }
}

} // namespace mlir
