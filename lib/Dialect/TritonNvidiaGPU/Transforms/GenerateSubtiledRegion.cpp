#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTESTGENERATESUBTILEDREGIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

/// Get the async task IDs from an operation.
static SmallVector<int32_t> getOpAsyncTaskIds(Operation *op) {
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id"))
    return SmallVector<int32_t>(attr.asArrayRef());
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(gpu::kPartitionAttrName))
    return SmallVector<int32_t>(attr.asArrayRef());
  return {};
}

/// A segment of structurally equivalent per-tile chain ops with a uniform
/// async task set. opsPerTile[t] holds the ops for tile t.
struct ChainSegment {
  SmallVector<SmallVector<Operation *>> opsPerTile;
  SmallVector<int32_t> taskIds;
};

/// Strip convert_layout ops wrapping a value.
static Value stripConvertLayout(Value v) {
  while (auto cvt = v.getDefiningOp<gpu::ConvertLayoutOp>())
    v = cvt.getSrc();
  return v;
}

/// Trace the setup chain backward from a SplitOp:
///   split <- trans{[0,2,1]} <- reshape <- (convert_layout)* <- tmem_load
/// Returns the tmem_load op, or nullptr if the pattern doesn't match.
static TMEMLoadOp traceSetupChain(triton::SplitOp splitOp) {
  Value src = stripConvertLayout(splitOp.getSrc());
  auto transOp = src.getDefiningOp<triton::TransOp>();
  if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
    return nullptr;
  auto reshapeOp = transOp.getSrc().getDefiningOp<triton::ReshapeOp>();
  if (!reshapeOp)
    return nullptr;
  Value reshapeSrc = stripConvertLayout(reshapeOp.getSrc());
  return reshapeSrc.getDefiningOp<TMEMLoadOp>();
}

/// Result of structural equivalence check between two per-tile op chains.
struct EquivalenceResult {
  /// Operands that differ between the two chains: (chain0 value, chain1 value).
  SmallVector<std::pair<Value, Value>> differingOperands;

  /// Index of the chain that should be used as the tile body template (0 or 1).
  /// When one chain has extra identity-compatible ops, this is the longer chain
  /// so that the tile body includes those ops.
  unsigned templateChainIdx = 0;

  /// Identity-compatible ops present in the template chain but absent from the
  /// other chain. For each, the builder must create an integer constant with
  /// `identityVal` (0 for add/sub, 1 for mul) and add it as a differing
  /// operand paired with `varyingOperand`.
  struct IdentityOp {
    Value
        varyingOperand;  // the non-pass-through operand from the template chain
    int64_t identityVal; // 0 for addi/subi, 1 for muli
  };
  SmallVector<IdentityOp> identityOps;

  /// The actual operations in the template chain that are identity-inserted
  /// (no counterpart in the other chain). Used by groupByContiguousTaskSet
  /// to align segments.
  DenseSet<Operation *> identityOpSet;
};

/// Return true if `op` is an integer address computation op that can act as
/// an identity when one operand is the identity element (0 for add/sub, 1 for
/// mul).
static bool isIdentityCompatibleOp(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op);
}

/// For an identity-compatible op, return the identity element value
/// (0 for add/sub, 1 for mul).
static int64_t getIdentityValue(Operation *op) {
  if (isa<arith::MulIOp>(op))
    return 1;
  return 0; // addi, subi
}

/// Try to match two ops as structurally equivalent (same name, same attrs,
/// same result types). If they match, update the value map and record
/// differing operands. Returns false if the ops don't match.
static bool matchOps(Operation *op0, Operation *op1,
                     llvm::DenseMap<Value, Value> &valueMap,
                     SmallVector<std::pair<Value, Value>> &differingOperands) {
  if (op0->getName() != op1->getName())
    return false;
  if (op0->getNumOperands() != op1->getNumOperands())
    return false;
  if (op0->getNumResults() != op1->getNumResults())
    return false;
  for (auto [r0, r1] : llvm::zip(op0->getResults(), op1->getResults()))
    if (r0.getType() != r1.getType())
      return false;
  if (op0->getAttrDictionary() != op1->getAttrDictionary())
    return false;

  for (auto [v0, v1] : llvm::zip(op0->getOperands(), op1->getOperands())) {
    auto it = valueMap.find(v0);
    if (it != valueMap.end()) {
      if (it->second != v1)
        return false;
      continue;
    }
    if (v0 == v1)
      continue;
    differingOperands.push_back({v0, v1});
    valueMap[v0] = v1;
  }
  for (auto [r0, r1] : llvm::zip(op0->getResults(), op1->getResults()))
    valueMap[r0] = r1;
  return true;
}

/// Check if two per-tile op chains are structurally equivalent, allowing
/// identity-compatible integer address ops (addi, subi, muli) to be present
/// in one chain but absent in the other.
///
/// When chains have the same length, this performs exact matching (like the
/// original checkStructuralEquivalence). When they differ, a two-pointer
/// alignment is used: extra ops in the longer chain are accepted if they are
/// identity-compatible, and their results are mapped to their pass-through
/// operand in the shorter chain's value space.
static std::optional<EquivalenceResult>
checkStructuralEquivalence(ArrayRef<Operation *> chain0,
                           ArrayRef<Operation *> chain1) {
  // Determine which chain is the template (longer or chain0 if same length).
  unsigned tplIdx = (chain1.size() > chain0.size()) ? 1 : 0;
  ArrayRef<Operation *> tplChain = (tplIdx == 0) ? chain0 : chain1;
  ArrayRef<Operation *> otherChain = (tplIdx == 0) ? chain1 : chain0;

  EquivalenceResult result;
  result.templateChainIdx = tplIdx;

  // Value map: template chain values → other chain values.
  llvm::DenseMap<Value, Value> valueMap;
  SmallVector<std::pair<Value, Value>> differingOperands;

  unsigned ti = 0, oi = 0;
  while (ti < tplChain.size() && oi < otherChain.size()) {
    Operation *tOp = tplChain[ti];
    Operation *oOp = otherChain[oi];

    if (matchOps(tOp, oOp, valueMap, differingOperands)) {
      ti++;
      oi++;
      continue;
    }

    // Ops don't match. Check if the template op is identity-compatible and
    // can be skipped (i.e., its result can be treated as equal to one of its
    // operands in the other chain).
    if (isIdentityCompatibleOp(tOp) && tOp->getNumResults() == 1) {
      // Try each operand as the pass-through. The pass-through operand's
      // mapped value (in the other chain) replaces the template op's result.
      // For subi, only operand 0 can be the pass-through (x - 0 = x, but
      // 0 - x != x).
      bool skipped = false;
      unsigned numCandidates = isa<arith::SubIOp>(tOp) ? 1 : 2;
      for (unsigned opIdx = 0; opIdx < numCandidates; ++opIdx) {
        Value passThrough = tOp->getOperand(opIdx);
        Value varying = tOp->getOperand(1 - opIdx);

        // Resolve the pass-through operand to the other chain's value.
        Value otherVal = valueMap.lookup(passThrough);
        if (!otherVal)
          otherVal = passThrough; // external value, same in both chains

        // Map the template op's result to the other chain's pass-through.
        valueMap[tOp->getResult(0)] = otherVal;

        int64_t identityVal = getIdentityValue(tOp);
        result.identityOps.push_back({varying, identityVal});
        result.identityOpSet.insert(tOp);
        ti++;
        skipped = true;
        break;
      }
      if (skipped)
        continue;
    }

    // Can't align — not structurally equivalent.
    return std::nullopt;
  }

  // Handle remaining ops in the template chain.
  while (ti < tplChain.size()) {
    Operation *tOp = tplChain[ti];
    if (!isIdentityCompatibleOp(tOp) || tOp->getNumResults() != 1)
      return std::nullopt;

    Value passThrough = tOp->getOperand(0);
    Value varying = tOp->getOperand(1);
    Value otherVal = valueMap.lookup(passThrough);
    if (!otherVal)
      otherVal = passThrough;
    valueMap[tOp->getResult(0)] = otherVal;
    result.identityOps.push_back({varying, getIdentityValue(tOp)});
    result.identityOpSet.insert(tOp);
    ti++;
  }

  // All other-chain ops must be consumed.
  if (oi != otherChain.size())
    return std::nullopt;

  // Normalize differing operands: always (chain0 value, chain1 value).
  if (tplIdx == 0) {
    result.differingOperands = std::move(differingOperands);
  } else {
    // Template is chain1, so valueMap is chain1→chain0. Swap pairs.
    for (auto &[v0, v1] : differingOperands)
      result.differingOperands.push_back({v1, v0});
  }

  return result;
}

/// Result of N-way structural equivalence check.
struct NWayEquivalenceResult {
  /// differingOperands[i][t] is the value for tile t at differing position i.
  SmallVector<SmallVector<Value>> differingOperands;
  unsigned templateChainIdx = 0;
  SmallVector<EquivalenceResult::IdentityOp> identityOps;
  DenseSet<Operation *> identityOpSet;
};

/// Check structural equivalence across N chains. Finds the longest chain
/// as the template and compares all others against it pairwise.
static std::optional<NWayEquivalenceResult>
checkStructuralEquivalenceN(ArrayRef<SmallVector<Operation *>> chains) {
  assert(chains.size() >= 2);
  unsigned numTiles = chains.size();

  // Find the longest chain as template.
  unsigned tplIdx = 0;
  for (unsigned t = 1; t < numTiles; ++t) {
    if (chains[t].size() > chains[tplIdx].size())
      tplIdx = t;
  }

  // Compare each non-template chain against the template.
  SmallVector<EquivalenceResult> pairResults(numTiles);
  for (unsigned t = 0; t < numTiles; ++t) {
    if (t == tplIdx)
      continue;
    auto res = checkStructuralEquivalence(chains[tplIdx], chains[t]);
    if (!res)
      return std::nullopt;
    if (res->templateChainIdx != 0)
      return std::nullopt;
    pairResults[t] = std::move(*res);
  }

  // All pairs must have the same number of differing operands and identity ops.
  unsigned numDiff = 0;
  unsigned numIdentity = 0;
  bool first = true;
  for (unsigned t = 0; t < numTiles; ++t) {
    if (t == tplIdx)
      continue;
    if (first) {
      numDiff = pairResults[t].differingOperands.size();
      numIdentity = pairResults[t].identityOps.size();
      first = false;
    } else {
      if (pairResults[t].differingOperands.size() != numDiff)
        return std::nullopt;
      if (pairResults[t].identityOps.size() != numIdentity)
        return std::nullopt;
    }
  }

  // Find the first non-template index for reference.
  unsigned refIdx = (tplIdx == 0) ? 1 : 0;

  NWayEquivalenceResult result;
  result.templateChainIdx = tplIdx;
  result.identityOps = pairResults[refIdx].identityOps;
  result.identityOpSet = pairResults[refIdx].identityOpSet;

  for (unsigned i = 0; i < numDiff; ++i) {
    SmallVector<Value> perTile(numTiles);
    // The template chain's value is .first from any pair result.
    perTile[tplIdx] = pairResults[refIdx].differingOperands[i].first;
    for (unsigned t = 0; t < numTiles; ++t) {
      if (t == tplIdx)
        continue;
      perTile[t] = pairResults[t].differingOperands[i].second;
    }
    result.differingOperands.push_back(std::move(perTile));
  }

  return result;
}

/// Check if a split result feeds into another reshape → trans → split chain.
/// If so, return the inner split op; otherwise return nullptr.
static triton::SplitOp getInnerSplit(Value splitResult) {
  for (Operation *user : splitResult.getUsers()) {
    auto reshapeOp = dyn_cast<triton::ReshapeOp>(user);
    if (!reshapeOp)
      continue;
    for (Operation *reshapeUser : reshapeOp.getResult().getUsers()) {
      auto transOp = dyn_cast<triton::TransOp>(reshapeUser);
      if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
        continue;
      Value transSrc = stripConvertLayout(transOp.getResult());
      for (Operation *transUser : transSrc.getUsers()) {
        if (auto innerSplit = dyn_cast<triton::SplitOp>(transUser))
          return innerSplit;
      }
    }
  }
  return nullptr;
}

/// Walk a tree of nested splits rooted at `rootSplit` and collect all leaf
/// values (split results that don't feed into further splits). Also collects
/// all intermediate ops (reshape, trans, inner splits) as setup ops.
/// Leaf values are ordered left-to-right in the tree.
static void
collectSplitTreeLeaves(triton::SplitOp rootSplit,
                       SmallVectorImpl<Value> &leafValues,
                       SmallVectorImpl<Operation *> &innerSetupOps) {
  SmallVector<Value> worklist = {rootSplit.getOutLHS(), rootSplit.getOutRHS()};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    auto innerSplit = getInnerSplit(v);
    if (innerSplit) {
      // Collect the intermediate ops (reshape, trans, split) as setup.
      for (Operation *user : v.getUsers()) {
        if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(user)) {
          innerSetupOps.push_back(reshapeOp);
          for (Operation *ru : reshapeOp.getResult().getUsers()) {
            if (auto transOp = dyn_cast<triton::TransOp>(ru)) {
              innerSetupOps.push_back(transOp);
            }
          }
        }
      }
      innerSetupOps.push_back(innerSplit);
      // Push RHS first so LHS is processed first (stack order).
      worklist.push_back(innerSplit.getOutRHS());
      worklist.push_back(innerSplit.getOutLHS());
    } else {
      leafValues.push_back(v);
    }
  }
}

/// Collect the per-tile op chain for a split result: all ops in the block
/// that transitively depend on `splitResult`.
/// When `includeAuxiliary` is true, also collects ops that are needed by the
/// chain but don't depend on the split result (e.g., address offset
/// computations like arith.addi). This is used for the 2-tile path where
/// identity insertion handles these ops. For the N-tile path, auxiliary ops
/// are left out and treated as differing operands.
static SmallVector<Operation *>
collectPerTileChain(Value splitResult, Operation *splitOp, Block *block,
                    bool includeAuxiliary = true) {
  SmallVector<Operation *> chain;
  llvm::DenseSet<Operation *> visited;
  SmallVector<Value> worklist;
  worklist.push_back(splitResult);

  // Forward walk: find all transitive users of the split result.
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (user->getBlock() != block)
        continue;
      if (user->isBeforeInBlock(splitOp) || user == splitOp)
        continue;
      if (!visited.insert(user).second)
        continue;
      chain.push_back(user);
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }

  if (includeAuxiliary) {
    llvm::DenseSet<Operation *> chainSet(chain.begin(), chain.end());
    SmallVector<Operation *> auxWorklist(chain.begin(), chain.end());
    while (!auxWorklist.empty()) {
      Operation *op = auxWorklist.pop_back_val();
      for (Value operand : op->getOperands()) {
        auto defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() != block)
          continue;
        if (defOp->isBeforeInBlock(splitOp) || defOp == splitOp)
          continue;
        if (!chainSet.contains(defOp) && visited.insert(defOp).second) {
          chainSet.insert(defOp);
          auxWorklist.push_back(defOp);
        }
      }
    }
    SmallVector<Operation *> fullChain;
    for (Operation *op : chainSet)
      fullChain.push_back(op);
    llvm::sort(fullChain, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    return fullChain;
  }

  llvm::sort(chain,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return chain;
}

/// Group structurally equivalent chain ops by contiguous async task set.
/// Ops without task IDs are merged into the current segment.
/// Returns nullopt if corresponding ops in chain0/chain1 have different task
/// sets.
static std::optional<SmallVector<ChainSegment>>
groupByContiguousTaskSet(ArrayRef<Operation *> chain0,
                         ArrayRef<Operation *> chain1) {
  assert(chain0.size() == chain1.size());
  if (chain0.empty())
    return std::nullopt;

  SmallVector<ChainSegment> segments;
  for (size_t i = 0; i < chain0.size(); ++i) {
    auto taskIds0 = getOpAsyncTaskIds(chain0[i]);
    auto taskIds1 = getOpAsyncTaskIds(chain1[i]);

    if (taskIds0 != taskIds1)
      return std::nullopt;

    if (taskIds0.empty() && !segments.empty()) {
      segments.back().opsPerTile[0].push_back(chain0[i]);
      segments.back().opsPerTile[1].push_back(chain1[i]);
      continue;
    }

    if (segments.empty() || segments.back().taskIds != taskIds0) {
      ChainSegment seg;
      seg.opsPerTile.resize(2);
      seg.taskIds = taskIds0;
      segments.push_back(std::move(seg));
    }

    segments.back().opsPerTile[0].push_back(chain0[i]);
    segments.back().opsPerTile[1].push_back(chain1[i]);
  }
  return segments;
}

/// Group N chains by contiguous async task set. All chains must have the
/// same length (no identity-compatible ops — the N-tile path excludes
/// auxiliary ops so chains are uniform).
static std::optional<SmallVector<ChainSegment>>
groupByContiguousTaskSetN(ArrayRef<SmallVector<Operation *>> chains) {
  unsigned numTiles = chains.size();
  assert(numTiles >= 2);
  size_t chainLen = chains[0].size();
  for (auto &c : chains) {
    if (c.size() != chainLen)
      return std::nullopt;
  }
  if (chainLen == 0)
    return std::nullopt;

  SmallVector<ChainSegment> segments;
  for (size_t i = 0; i < chainLen; ++i) {
    auto taskIds = getOpAsyncTaskIds(chains[0][i]);
    for (unsigned t = 1; t < numTiles; ++t) {
      if (getOpAsyncTaskIds(chains[t][i]) != taskIds)
        return std::nullopt;
    }

    if (taskIds.empty() && !segments.empty()) {
      for (unsigned t = 0; t < numTiles; ++t)
        segments.back().opsPerTile[t].push_back(chains[t][i]);
      continue;
    }

    if (segments.empty() || segments.back().taskIds != taskIds) {
      ChainSegment seg;
      seg.opsPerTile.resize(numTiles);
      seg.taskIds = taskIds;
      segments.push_back(std::move(seg));
    }

    for (unsigned t = 0; t < numTiles; ++t)
      segments.back().opsPerTile[t].push_back(chains[t][i]);
  }
  return segments;
}

/// Group chains by contiguous async task set when the chains have different
/// lengths (due to identity-compatible ops). Uses the template chain from the
/// equivalence result for task set boundaries. Identity ops (present only in
/// the template chain) are placed in both opsPerTile[0] and [1] of their
/// segment.
static std::optional<SmallVector<ChainSegment>>
groupByContiguousTaskSetWithIdentity(ArrayRef<Operation *> chain0,
                                     ArrayRef<Operation *> chain1,
                                     const EquivalenceResult &equiv) {
  ArrayRef<Operation *> tplChain =
      (equiv.templateChainIdx == 0) ? chain0 : chain1;
  ArrayRef<Operation *> otherChain =
      (equiv.templateChainIdx == 0) ? chain1 : chain0;

  if (tplChain.empty())
    return std::nullopt;

  // Two-pointer alignment: walk the template chain and pair with the other
  // chain, skipping identity ops.
  SmallVector<ChainSegment> segments;
  unsigned oi = 0;
  for (size_t ti = 0; ti < tplChain.size(); ++ti) {
    auto taskIds = getOpAsyncTaskIds(tplChain[ti]);

    if (taskIds.empty() && !segments.empty()) {
      // Ops without task IDs join the current segment.
    } else if (segments.empty() || segments.back().taskIds != taskIds) {
      ChainSegment seg;
      seg.opsPerTile.resize(2);
      seg.taskIds = taskIds;
      segments.push_back(std::move(seg));
    }

    bool isIdentity = equiv.identityOpSet.count(tplChain[ti]);

    if (isIdentity) {
      segments.back().opsPerTile[0].push_back(tplChain[ti]);
      segments.back().opsPerTile[1].push_back(tplChain[ti]);
    } else {
      assert(oi < otherChain.size());
      Operation *op0 =
          (equiv.templateChainIdx == 0) ? tplChain[ti] : otherChain[oi];
      Operation *op1 =
          (equiv.templateChainIdx == 0) ? otherChain[oi] : tplChain[ti];
      segments.back().opsPerTile[0].push_back(op0);
      segments.back().opsPerTile[1].push_back(op1);
      ++oi;
    }
  }
  return segments;
}

/// Build a single SubtiledRegionOp for N tiles (generalized).
/// `leafValues` has one value per tile (the split leaf result).
/// `chains` has one chain per tile.
/// `equiv` is the N-way equivalence result.
/// `setupOps` includes all ops from tmem_load through the split tree.
static void buildSingleSubtiledRegionN(
    OpBuilder &builder, Location loc, ArrayRef<Operation *> setupOps,
    ArrayRef<Value> leafValues, ArrayRef<SmallVector<Operation *>> chains,
    const NWayEquivalenceResult &equiv) {
  MLIRContext *ctx = builder.getContext();
  unsigned numTiles = leafValues.size();
  auto &differing = equiv.differingOperands;
  ArrayRef<Operation *> tplChain = chains[equiv.templateChainIdx];

  // Tile arg types and per-tile mappings.
  SmallVector<Type> tileArgTypes;
  SmallVector<SmallVector<int32_t>> tileMappings(numTiles);

  // Tile arg 0: the leaf split result (same type for all tiles).
  tileArgTypes.push_back(leafValues[0].getType());
  for (unsigned t = 0; t < numTiles; ++t)
    tileMappings[t].push_back(t); // yield slot t → tile t's leaf value

  // Differing operands: one tile arg per differing position.
  unsigned yieldIdx = numTiles;
  for (auto &perTile : differing) {
    tileArgTypes.push_back(perTile[0].getType());
    for (unsigned t = 0; t < numTiles; ++t) {
      tileMappings[t].push_back(yieldIdx + t);
    }
    yieldIdx += numTiles;
  }

  // Identity insertions: one tile arg per identity op.
  // Yield 2 values per identity op: (varying, identity_const).
  // Template tile maps to varying; all other tiles map to identity_const.
  for (auto [i, id] : llvm::enumerate(equiv.identityOps)) {
    tileArgTypes.push_back(id.varyingOperand.getType());
    unsigned varyingSlot = yieldIdx;
    unsigned constSlot = yieldIdx + 1;
    for (unsigned t = 0; t < numTiles; ++t) {
      if (t == equiv.templateChainIdx)
        tileMappings[t].push_back(varyingSlot);
      else
        tileMappings[t].push_back(constSlot);
    }
    yieldIdx += 2;
  }

  SmallVector<Attribute> mappingAttrs;
  for (auto &mapping : tileMappings)
    mappingAttrs.push_back(DenseI32ArrayAttr::get(ctx, mapping));
  auto tileMappingsAttr = builder.getArrayAttr(mappingAttrs);
  auto barrierAnnotationsAttr = builder.getArrayAttr({});
  auto tokenAnnotationsAttr = builder.getArrayAttr({});

  auto regionOp = SubtiledRegionOp::create(
      builder, loc, TypeRange{}, ValueRange{}, ValueRange{}, ValueRange{},
      tileMappingsAttr, barrierAnnotationsAttr, tokenAnnotationsAttr);

  // --- Setup Region ---
  Block *setupBlock = &regionOp.getSetupRegion().emplaceBlock();
  OpBuilder setupBuilder = OpBuilder::atBlockEnd(setupBlock);
  IRMapping setupMapping;
  for (Operation *op : setupOps)
    setupBuilder.clone(*op, setupMapping);

  SmallVector<Value> setupYieldValues;
  // Yield the N leaf values.
  for (Value leaf : leafValues)
    setupYieldValues.push_back(setupMapping.lookupOrDefault(leaf));
  // Yield N-way differing operands.
  for (auto &perTile : differing) {
    for (Value v : perTile)
      setupYieldValues.push_back(setupMapping.lookupOrDefault(v));
  }
  // Yield identity insertion operands.
  for (auto &id : equiv.identityOps) {
    Value varying = setupMapping.lookupOrDefault(id.varyingOperand);
    Value identityConst = arith::ConstantOp::create(
        setupBuilder, loc,
        setupBuilder.getIntegerAttr(id.varyingOperand.getType(),
                                    id.identityVal));
    setupYieldValues.push_back(varying);
    setupYieldValues.push_back(identityConst);
  }
  SubtiledRegionYieldOp::create(setupBuilder, loc, setupYieldValues);

  // --- Tile Region ---
  Block *tileBlock = &regionOp.getTileRegion().emplaceBlock();
  for (Type ty : tileArgTypes)
    tileBlock->addArgument(ty, loc);
  tileBlock->addArgument(builder.getI32Type(), loc); // tile index

  OpBuilder tileBuilder = OpBuilder::atBlockEnd(tileBlock);
  IRMapping tileMapping;
  // Map template chain's leaf value to tile arg 0.
  Value tplLeaf = leafValues[equiv.templateChainIdx];
  tileMapping.map(tplLeaf, tileBlock->getArgument(0));
  unsigned argIdx = 1;
  // Map differing operands.
  for (auto &perTile : differing) {
    Value tplVal = perTile[equiv.templateChainIdx];
    tileMapping.map(tplVal, tileBlock->getArgument(argIdx++));
  }
  // Map identity operands.
  for (auto &id : equiv.identityOps)
    tileMapping.map(id.varyingOperand, tileBlock->getArgument(argIdx++));

  for (Operation *op : tplChain)
    tileBuilder.clone(*op, tileMapping);
  SubtiledRegionYieldOp::create(tileBuilder, loc, ValueRange{});

  // --- Teardown Region ---
  Block *teardownBlock = &regionOp.getTeardownRegion().emplaceBlock();
  OpBuilder teardownBuilder = OpBuilder::atBlockEnd(teardownBlock);
  SubtiledRegionYieldOp::create(teardownBuilder, loc, ValueRange{});
}

/// Build a single SubtiledRegionOp (2-tile path).
static void buildSingleSubtiledRegion(OpBuilder &builder, Location loc,
                                      ArrayRef<Operation *> setupOps, Value lhs,
                                      Value rhs, ArrayRef<Operation *> chain0,
                                      ArrayRef<Operation *> chain1,
                                      const EquivalenceResult &equiv) {
  MLIRContext *ctx = builder.getContext();
  auto &differing = equiv.differingOperands;
  ArrayRef<Operation *> tplChain =
      (equiv.templateChainIdx == 0) ? chain0 : chain1;

  // Tile arg types and mappings.
  SmallVector<Type> tileArgTypes;
  SmallVector<int32_t> tile0Mapping, tile1Mapping;

  // Tile arg 0: split result.
  tileArgTypes.push_back(lhs.getType());
  tile0Mapping.push_back(0);
  tile1Mapping.push_back(1);

  // Additional tile args from differing operands.
  for (auto [i, pair] : llvm::enumerate(differing)) {
    tileArgTypes.push_back(pair.first.getType());
    tile0Mapping.push_back(2 + 2 * i);
    tile1Mapping.push_back(2 + 2 * i + 1);
  }

  // Additional tile args from identity insertions.
  unsigned identityBase = 2 + 2 * differing.size();
  for (auto [i, id] : llvm::enumerate(equiv.identityOps)) {
    tileArgTypes.push_back(id.varyingOperand.getType());
    // For the template chain's tile, use the varying operand.
    // For the other tile, use the identity constant.
    unsigned tplSlot = identityBase + 2 * i;
    unsigned otherSlot = identityBase + 2 * i + 1;
    if (equiv.templateChainIdx == 0) {
      tile0Mapping.push_back(tplSlot);
      tile1Mapping.push_back(otherSlot);
    } else {
      tile0Mapping.push_back(otherSlot);
      tile1Mapping.push_back(tplSlot);
    }
  }

  auto tileMappingsAttr =
      builder.getArrayAttr({DenseI32ArrayAttr::get(ctx, tile0Mapping),
                            DenseI32ArrayAttr::get(ctx, tile1Mapping)});
  auto barrierAnnotationsAttr = builder.getArrayAttr({});
  auto tokenAnnotationsAttr = builder.getArrayAttr({});

  auto regionOp = SubtiledRegionOp::create(
      builder, loc, /*resultTypes=*/TypeRange{},
      /*barriers=*/ValueRange{}, /*accumCnts=*/ValueRange{},
      /*tokenValues=*/ValueRange{}, tileMappingsAttr, barrierAnnotationsAttr,
      tokenAnnotationsAttr);

  // --- Setup Region ---
  Block *setupBlock = &regionOp.getSetupRegion().emplaceBlock();
  OpBuilder setupBuilder = OpBuilder::atBlockEnd(setupBlock);
  IRMapping setupMapping;
  for (Operation *op : setupOps)
    setupBuilder.clone(*op, setupMapping);

  SmallVector<Value> setupYieldValues;
  setupYieldValues.push_back(setupMapping.lookupOrDefault(lhs));
  setupYieldValues.push_back(setupMapping.lookupOrDefault(rhs));
  for (auto [v0, v1] : differing) {
    setupYieldValues.push_back(setupMapping.lookupOrDefault(v0));
    setupYieldValues.push_back(setupMapping.lookupOrDefault(v1));
  }
  // Yield identity insertion operands: (varying, identity_const) pairs.
  for (auto &id : equiv.identityOps) {
    Value varying = setupMapping.lookupOrDefault(id.varyingOperand);
    Value identityConst = arith::ConstantOp::create(
        setupBuilder, loc,
        setupBuilder.getIntegerAttr(id.varyingOperand.getType(),
                                    id.identityVal));
    // Template side gets the varying operand, other side gets the constant.
    setupYieldValues.push_back(varying);
    setupYieldValues.push_back(identityConst);
  }
  SubtiledRegionYieldOp::create(setupBuilder, loc, setupYieldValues);

  // --- Tile Region ---
  Block *tileBlock = &regionOp.getTileRegion().emplaceBlock();
  for (Type ty : tileArgTypes)
    tileBlock->addArgument(ty, loc);
  tileBlock->addArgument(builder.getI32Type(), loc);

  OpBuilder tileBuilder = OpBuilder::atBlockEnd(tileBlock);
  IRMapping tileMapping;
  Value tplSplitResult = (equiv.templateChainIdx == 0) ? lhs : rhs;
  tileMapping.map(tplSplitResult, tileBlock->getArgument(0));
  unsigned argIdx = 1;
  for (auto [i, pair] : llvm::enumerate(differing)) {
    Value tplVal = (equiv.templateChainIdx == 0) ? pair.first : pair.second;
    tileMapping.map(tplVal, tileBlock->getArgument(argIdx++));
  }

  // Map identity insertion operands: the template chain's op references the
  // varying operand, which is mapped to the tile arg.
  for (auto &id : equiv.identityOps)
    tileMapping.map(id.varyingOperand, tileBlock->getArgument(argIdx++));

  // Clone from the template chain (which has all ops including identity ones).
  for (Operation *op : tplChain)
    tileBuilder.clone(*op, tileMapping);
  SubtiledRegionYieldOp::create(tileBuilder, loc, ValueRange{});

  // --- Teardown Region ---
  Block *teardownBlock = &regionOp.getTeardownRegion().emplaceBlock();
  OpBuilder teardownBuilder = OpBuilder::atBlockEnd(teardownBlock);
  SubtiledRegionYieldOp::create(teardownBuilder, loc, ValueRange{});
}

/// Create a mutable MemDescType with a trivial shared encoding for buffering
/// a tensor value through SMEM.
static gpu::MemDescType createBufferMemDescType(MLIRContext *ctx,
                                                RankedTensorType tensorType) {
  SmallVector<unsigned> order;
  for (int i = tensorType.getRank() - 1; i >= 0; --i)
    order.push_back(static_cast<unsigned>(i));
  auto cgaLayout =
      gpu::CGAEncodingAttr::get1CTALayout(ctx, tensorType.getRank());
  auto sharedEncoding = gpu::SwizzledSharedEncodingAttr::get(
      ctx, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, order, cgaLayout);
  auto sharedMemorySpace = gpu::SharedMemorySpaceAttr::get(ctx);
  return gpu::MemDescType::get(tensorType.getShape(),
                               tensorType.getElementType(), sharedEncoding,
                               sharedMemorySpace, /*mutableMemory=*/true);
}

/// Build multiple SubtiledRegionOps for a chain that spans multiple contiguous
/// async task sets.
///
/// Two transition types are handled:
///   Option 1 (explicit store): The last op of a segment is a local_alloc with
///     data. It is split into an empty outer-scope alloc + local_store.
///   Option 2 (implicit buffer): No memory op at the boundary. Cross-segment
///     tensor values are buffered through SMEM via local_store + local_load.
static void buildMultiTaskSubtiledRegions(OpBuilder &outerBuilder, Location loc,
                                          ArrayRef<Operation *> setupOps,
                                          Value lhs, Value rhs,
                                          ArrayRef<ChainSegment> segments) {
  MLIRContext *ctx = outerBuilder.getContext();

  // --- Transition analysis ---
  // For each transition i between segments[i] and segments[i+1], collect
  // buffer info.  A buffer entry describes one value that needs to be stored
  // to SMEM in the producing segment and (optionally) loaded in the consuming
  // segment.
  struct BufferEntry {
    Value chain0Val;     // value in chain0 being buffered
    Value chain1Val;     // corresponding value in chain1
    Value smem0;         // outer-scope empty alloc for tile 0
    Value smem1;         // outer-scope empty alloc for tile 1
    bool needsLocalLoad; // true for option 2 (consuming segment needs load)
  };

  struct TransitionInfo {
    // Non-null for option 1 (explicit store at local_alloc).
    gpu::LocalAllocOp alloc0;
    gpu::LocalAllocOp alloc1;

    SmallVector<BufferEntry> buffers;

    bool isExplicitStore() const { return alloc0 != nullptr; }
  };

  SmallVector<TransitionInfo> transitions;

  for (size_t i = 0; i + 1 < segments.size(); ++i) {
    TransitionInfo tr;
    Operation *lastOp0 = segments[i].opsPerTile[0].back();
    auto alloc0 = dyn_cast<gpu::LocalAllocOp>(lastOp0);

    if (alloc0 && alloc0.getSrc()) {
      // Option 1: explicit memory store at local_alloc.
      auto alloc1 = cast<gpu::LocalAllocOp>(segments[i].opsPerTile[1].back());
      tr.alloc0 = alloc0;
      tr.alloc1 = alloc1;

      auto memDescType = cast<gpu::MemDescType>(alloc0.getType());
      if (!memDescType.getMutableMemory()) {
        memDescType = gpu::MemDescType::get(
            memDescType.getShape(), memDescType.getElementType(),
            memDescType.getEncoding(), memDescType.getMemorySpace(),
            /*mutableMemory=*/true, memDescType.getAllocShape());
      }

      auto e0 =
          gpu::LocalAllocOp::create(outerBuilder, loc, memDescType, Value{});
      auto e1 =
          gpu::LocalAllocOp::create(outerBuilder, loc, memDescType, Value{});
      // The alloc result (memdesc) is consumed directly by the next segment
      // (e.g., async_tma_copy), so no local_load is needed.
      tr.buffers.push_back({alloc0.getResult(), alloc1.getResult(),
                            e0.getResult(), e1.getResult(),
                            /*needsLocalLoad=*/false});
    } else {
      // Option 2: implicit buffer. Find cross-segment tensor values.
      DenseSet<Value> seg0Results;
      for (auto *op : segments[i].opsPerTile[0])
        for (Value r : op->getResults())
          seg0Results.insert(r);

      llvm::MapVector<Value, Value> seen; // chain0Val -> chain1Val
      for (auto [op0, op1] : llvm::zip(segments[i + 1].opsPerTile[0],
                                       segments[i + 1].opsPerTile[1])) {
        for (auto [v0, v1] : llvm::zip(op0->getOperands(), op1->getOperands()))
          if (seg0Results.contains(v0) && !seen.count(v0))
            seen[v0] = v1;
      }

      for (auto &[v0, v1] : seen) {
        auto tensorTy = dyn_cast<RankedTensorType>(v0.getType());
        if (!tensorTy)
          continue; // skip tokens, scalars — only buffer tensors
        auto memDescType = createBufferMemDescType(ctx, tensorTy);
        auto e0 =
            gpu::LocalAllocOp::create(outerBuilder, loc, memDescType, Value{});
        auto e1 =
            gpu::LocalAllocOp::create(outerBuilder, loc, memDescType, Value{});
        tr.buffers.push_back({v0, v1, e0.getResult(), e1.getResult(),
                              /*needsLocalLoad=*/true});
      }
    }
    transitions.push_back(std::move(tr));
  }

  // --- Generate a SubtiledRegionOp for each segment ---
  for (size_t segIdx = 0; segIdx < segments.size(); ++segIdx) {
    const auto &seg = segments[segIdx];
    bool isFirstSegment = (segIdx == 0);
    bool hasOutgoingTransition = (segIdx < transitions.size());
    bool hasIncomingTransition = (segIdx > 0);

    // Build the sub-chain for structural equivalence.
    // For option 1, exclude the transition local_alloc (replaced by
    // local_store).
    SmallVector<Operation *> subOps0(seg.opsPerTile[0]);
    SmallVector<Operation *> subOps1(seg.opsPerTile[1]);
    if (hasOutgoingTransition && transitions[segIdx].isExplicitStore()) {
      subOps0.pop_back(); // remove local_alloc
      subOps1.pop_back();
    }

    // Compute per-segment differing operands.
    auto segEquiv = checkStructuralEquivalence(subOps0, subOps1);
    if (!segEquiv)
      return;
    auto &segDiff = segEquiv->differingOperands;
    auto &segIdentity = segEquiv->identityOps;
    ArrayRef<Operation *> tplOps =
        (segEquiv->templateChainIdx == 0) ? subOps0 : subOps1;

    // Resolve cross-segment operands: replace original values with outer-scope
    // SMEM values.  Track which entries need a local_load in the tile body.
    struct DiffEntry {
      Value chain0Val; // original value in chain0 ops (for tileMapping)
      Value setupVal0; // value to yield in setup for tile 0
      Value setupVal1; // value to yield in setup for tile 1
      bool needsLocalLoad = false;
    };
    SmallVector<DiffEntry> resolvedDiff;
    for (auto &[v0, v1] : segDiff) {
      Value setupV0 = v0, setupV1 = v1;
      bool needsLoad = false;
      for (auto &tr : transitions) {
        for (auto &buf : tr.buffers) {
          if (v0 == buf.chain0Val) {
            setupV0 = buf.smem0;
            needsLoad = buf.needsLocalLoad;
          }
          if (v1 == buf.chain1Val)
            setupV1 = buf.smem1;
        }
      }
      resolvedDiff.push_back({v0, setupV0, setupV1, needsLoad});
    }

    // Build tile arg types and mappings.
    SmallVector<Type> tileArgTypes;
    SmallVector<int32_t> tile0Map, tile1Map;
    int32_t yieldIdx = 0;

    if (isFirstSegment) {
      tileArgTypes.push_back(lhs.getType());
      tile0Map.push_back(0);
      tile1Map.push_back(1);
      yieldIdx = 2;
    }

    for (auto &entry : resolvedDiff) {
      // For implicit-buffer entries the tile arg is a memdesc, not the
      // original tensor type.
      Type argType = entry.needsLocalLoad ? entry.setupVal0.getType()
                                          : entry.chain0Val.getType();
      tileArgTypes.push_back(argType);
      tile0Map.push_back(yieldIdx);
      tile1Map.push_back(yieldIdx + 1);
      yieldIdx += 2;
    }

    // Identity insertion tile args: (varying, identity_const) pairs.
    for (auto &id : segIdentity) {
      tileArgTypes.push_back(id.varyingOperand.getType());
      int32_t tplSlot = yieldIdx;
      int32_t otherSlot = yieldIdx + 1;
      if (segEquiv->templateChainIdx == 0) {
        tile0Map.push_back(tplSlot);
        tile1Map.push_back(otherSlot);
      } else {
        tile0Map.push_back(otherSlot);
        tile1Map.push_back(tplSlot);
      }
      yieldIdx += 2;
    }

    // Outgoing SMEM args (for local_store at the end of this segment).
    // Collect the buffer entries for the outgoing transition so we can add
    // tile args for the SMEM destinations.
    SmallVector<BufferEntry *> outgoingBuffers;
    if (hasOutgoingTransition) {
      for (auto &buf : transitions[segIdx].buffers) {
        tileArgTypes.push_back(buf.smem0.getType());
        tile0Map.push_back(yieldIdx);
        tile1Map.push_back(yieldIdx + 1);
        yieldIdx += 2;
        outgoingBuffers.push_back(&buf);
      }
    }

    auto tileMappingsAttr =
        outerBuilder.getArrayAttr({DenseI32ArrayAttr::get(ctx, tile0Map),
                                   DenseI32ArrayAttr::get(ctx, tile1Map)});
    auto barrierAnnotationsAttr = outerBuilder.getArrayAttr({});
    auto tokenAnnotationsAttr = outerBuilder.getArrayAttr({});

    auto regionOp = SubtiledRegionOp::create(
        outerBuilder, loc, TypeRange{}, ValueRange{}, ValueRange{},
        /*tokenValues=*/ValueRange{}, tileMappingsAttr, barrierAnnotationsAttr,
        tokenAnnotationsAttr);

    // --- Setup Region ---
    Block *setupBlock = &regionOp.getSetupRegion().emplaceBlock();
    OpBuilder setupBuilder = OpBuilder::atBlockEnd(setupBlock);

    SmallVector<Value> setupYields;
    if (isFirstSegment) {
      IRMapping setupMapping;
      for (Operation *op : setupOps)
        setupBuilder.clone(*op, setupMapping);
      setupYields.push_back(setupMapping.lookupOrDefault(lhs));
      setupYields.push_back(setupMapping.lookupOrDefault(rhs));

      for (auto &entry : resolvedDiff) {
        setupYields.push_back(setupMapping.lookupOrDefault(entry.setupVal0));
        setupYields.push_back(setupMapping.lookupOrDefault(entry.setupVal1));
      }
      for (auto &id : segIdentity) {
        setupYields.push_back(setupMapping.lookupOrDefault(id.varyingOperand));
        setupYields.push_back(arith::ConstantOp::create(
            setupBuilder, loc,
            setupBuilder.getIntegerAttr(id.varyingOperand.getType(),
                                        id.identityVal)));
      }
    } else {
      for (auto &entry : resolvedDiff) {
        setupYields.push_back(entry.setupVal0);
        setupYields.push_back(entry.setupVal1);
      }
      for (auto &id : segIdentity) {
        setupYields.push_back(id.varyingOperand);
        setupYields.push_back(arith::ConstantOp::create(
            setupBuilder, loc,
            setupBuilder.getIntegerAttr(id.varyingOperand.getType(),
                                        id.identityVal)));
      }
    }

    // Yield SMEM values for outgoing stores.
    for (auto *buf : outgoingBuffers) {
      setupYields.push_back(buf->smem0);
      setupYields.push_back(buf->smem1);
    }

    SubtiledRegionYieldOp::create(setupBuilder, loc, setupYields);

    // --- Tile Region ---
    Block *tileBlock = &regionOp.getTileRegion().emplaceBlock();
    for (Type ty : tileArgTypes)
      tileBlock->addArgument(ty, loc);
    tileBlock->addArgument(outerBuilder.getI32Type(), loc);

    OpBuilder tileBuilder = OpBuilder::atBlockEnd(tileBlock);
    IRMapping tileMapping;
    unsigned argIdx = 0;

    if (isFirstSegment)
      tileMapping.map(lhs, tileBlock->getArgument(argIdx++));

    for (auto &entry : resolvedDiff) {
      Value tileArg = tileBlock->getArgument(argIdx++);
      if (entry.needsLocalLoad) {
        // Option 2: tile arg is a memdesc — emit local_load to get the tensor.
        auto loaded = gpu::LocalLoadOp::create(
            tileBuilder, loc, entry.chain0Val.getType(), tileArg);
        tileMapping.map(entry.chain0Val, loaded.getResult());
      } else {
        tileMapping.map(entry.chain0Val, tileArg);
      }
    }

    // Map identity insertion operands: the template chain's identity op
    // references the varying operand, which is mapped to the tile arg.
    for (auto &id : segIdentity)
      tileMapping.map(id.varyingOperand, tileBlock->getArgument(argIdx++));

    // Collect outgoing SMEM tile args.
    SmallVector<Value> outgoingSmemArgs;
    for (size_t i = 0; i < outgoingBuffers.size(); ++i)
      outgoingSmemArgs.push_back(tileBlock->getArgument(argIdx++));

    // Clone segment ops into the tile body (from the template chain which
    // includes identity ops).
    for (Operation *op : tplOps)
      tileBuilder.clone(*op, tileMapping);

    // Emit outgoing stores. Use the template chain's value for lookup since
    // the tile body was cloned from the template chain.
    if (hasOutgoingTransition) {
      auto &tr = transitions[segIdx];
      bool tplIs1 = (segEquiv->templateChainIdx == 1);
      if (tr.isExplicitStore()) {
        // Option 1: store the local_alloc's source data.
        auto srcAlloc =
            tplIs1
                ? cast<gpu::LocalAllocOp>(segments[segIdx].opsPerTile[1].back())
                : tr.alloc0;
        Value data = tileMapping.lookupOrDefault(srcAlloc.getSrc());
        gpu::LocalStoreOp::create(tileBuilder, loc, data, outgoingSmemArgs[0]);
      } else {
        // Option 2: store each cross-segment value.
        for (auto [buf, smemArg] : llvm::zip(tr.buffers, outgoingSmemArgs)) {
          Value bufVal = tplIs1 ? buf.chain1Val : buf.chain0Val;
          Value data = tileMapping.lookupOrDefault(bufVal);
          gpu::LocalStoreOp::create(tileBuilder, loc, data, smemArg);
        }
      }
    }

    SubtiledRegionYieldOp::create(tileBuilder, loc, ValueRange{});

    // --- Teardown Region ---
    Block *teardownBlock = &regionOp.getTeardownRegion().emplaceBlock();
    OpBuilder teardownBuilder = OpBuilder::atBlockEnd(teardownBlock);
    SubtiledRegionYieldOp::create(teardownBuilder, loc, ValueRange{});
  }
}

/// Build multiple SubtiledRegionOps for N-tile chains spanning multiple
/// async task sets. Uses implicit buffering (Option 2) at segment
/// transitions — cross-segment tensor values are communicated through SMEM.
static bool buildMultiTaskSubtiledRegionsN(OpBuilder &outerBuilder,
                                           Location loc,
                                           ArrayRef<Operation *> setupOps,
                                           ArrayRef<Value> leafValues,
                                           ArrayRef<ChainSegment> segments) {
  MLIRContext *ctx = outerBuilder.getContext();
  unsigned numTiles = leafValues.size();

  // --- Transition analysis ---
  // For each transition between segments[i] and segments[i+1], find
  // cross-segment tensor values and create SMEM buffers for them.
  struct BufferEntryN {
    SmallVector<Value> chainVals; // one per tile
    SmallVector<Value> smemVals;  // one per tile
    bool needsLocalLoad;
  };

  SmallVector<SmallVector<BufferEntryN>> transitions; // one per transition

  for (size_t i = 0; i + 1 < segments.size(); ++i) {
    Operation *lastOp0 = segments[i].opsPerTile[0].back();
    auto alloc0 = dyn_cast<gpu::LocalAllocOp>(lastOp0);

    if (alloc0 && alloc0.getSrc()) {
      // Option 1: explicit memory store at local_alloc.
      // Not yet supported for N-tile multi-task.
      return false;
    }

    // Option 2: implicit buffer.
    // Find cross-segment values: results of segment i ops used by segment i+1.
    DenseSet<Value> seg0Results;
    for (auto *op : segments[i].opsPerTile[0])
      for (Value r : op->getResults())
        seg0Results.insert(r);

    // Use MapVector for deterministic ordering.
    llvm::MapVector<Value, SmallVector<Value>> seen;
    for (size_t opIdx = 0; opIdx < segments[i + 1].opsPerTile[0].size();
         ++opIdx) {
      for (unsigned t = 0; t < numTiles; ++t) {
        for (Value v : segments[i + 1].opsPerTile[t][opIdx]->getOperands()) {
          if (t == 0 && seg0Results.contains(v) && !seen.count(v)) {
            SmallVector<Value> perTile(numTiles);
            perTile[0] = v;
            seen[v] = std::move(perTile);
          }
        }
      }
    }
    // Fill in non-zero tiles by matching operand position.
    for (size_t opIdx = 0; opIdx < segments[i + 1].opsPerTile[0].size();
         ++opIdx) {
      auto *op0 = segments[i + 1].opsPerTile[0][opIdx];
      for (auto [oprIdx, v0] : llvm::enumerate(op0->getOperands())) {
        auto it = seen.find(v0);
        if (it == seen.end())
          continue;
        for (unsigned t = 1; t < numTiles; ++t) {
          Value vt = segments[i + 1].opsPerTile[t][opIdx]->getOperand(oprIdx);
          it->second[t] = vt;
        }
      }
    }

    // Bail if any cross-segment value is not a tensor (e.g., pre-allocated
    // SMEM memdesc from the memory planner). These need to be passed through
    // as differing operands without re-buffering, which requires the
    // per-segment refactor.
    for (auto &[v0, perTile] : seen) {
      if (!isa<RankedTensorType>(v0.getType()))
        return false;
    }

    SmallVector<BufferEntryN> bufs;
    for (auto &[v0, perTile] : seen) {
      auto tensorTy = cast<RankedTensorType>(v0.getType());
      auto memDescType = createBufferMemDescType(ctx, tensorTy);
      SmallVector<Value> smems;
      for (unsigned t = 0; t < numTiles; ++t) {
        auto alloc =
            gpu::LocalAllocOp::create(outerBuilder, loc, memDescType, Value{});
        smems.push_back(alloc.getResult());
      }
      bufs.push_back({perTile, smems, /*needsLocalLoad=*/true});
    }
    transitions.push_back(std::move(bufs));
  }

  // --- Generate a SubtiledRegionOp per segment ---
  for (size_t segIdx = 0; segIdx < segments.size(); ++segIdx) {
    auto &seg = segments[segIdx];
    bool isFirstSegment = (segIdx == 0);
    bool hasOutgoing = (segIdx < transitions.size());
    bool hasIncoming = (segIdx > 0);

    SmallVector<SmallVector<Operation *>> segChains;
    for (auto &ops : seg.opsPerTile)
      segChains.push_back(SmallVector<Operation *>(ops));

    auto segEquiv = checkStructuralEquivalenceN(segChains);
    if (!segEquiv)
      return false;
    auto &segDiff = segEquiv->differingOperands;

    // Resolve cross-segment operands.
    struct DiffEntryN {
      SmallVector<Value> chainVals;
      SmallVector<Value> setupVals;
      bool needsLocalLoad = false;
    };
    SmallVector<DiffEntryN> resolvedDiff;
    for (auto &perTile : segDiff) {
      SmallVector<Value> setupVals = perTile;
      bool needsLoad = false;
      if (hasIncoming) {
        for (auto &buf : transitions[segIdx - 1]) {
          for (unsigned t = 0; t < numTiles; ++t) {
            if (perTile[t] == buf.chainVals[t]) {
              setupVals[t] = buf.smemVals[t];
              needsLoad = true;
            }
          }
        }
      }
      resolvedDiff.push_back({perTile, setupVals, needsLoad});
    }

    // Build tile arg types and N-way mappings.
    SmallVector<Type> tileArgTypes;
    SmallVector<SmallVector<int32_t>> tileMaps(numTiles);
    int32_t yieldIdx = 0;

    if (isFirstSegment) {
      tileArgTypes.push_back(leafValues[0].getType());
      for (unsigned t = 0; t < numTiles; ++t)
        tileMaps[t].push_back(yieldIdx + t);
      yieldIdx += numTiles;
    }

    for (auto &entry : resolvedDiff) {
      Type argType = entry.needsLocalLoad ? entry.setupVals[0].getType()
                                          : entry.chainVals[0].getType();
      tileArgTypes.push_back(argType);
      for (unsigned t = 0; t < numTiles; ++t)
        tileMaps[t].push_back(yieldIdx + t);
      yieldIdx += numTiles;
    }

    // Outgoing SMEM args.
    SmallVector<BufferEntryN *> outBufs;
    if (hasOutgoing) {
      for (auto &buf : transitions[segIdx]) {
        tileArgTypes.push_back(buf.smemVals[0].getType());
        for (unsigned t = 0; t < numTiles; ++t)
          tileMaps[t].push_back(yieldIdx + t);
        yieldIdx += numTiles;
        outBufs.push_back(&buf);
      }
    }

    SmallVector<Attribute> mapAttrs;
    for (auto &m : tileMaps)
      mapAttrs.push_back(DenseI32ArrayAttr::get(ctx, m));
    auto tileMappingsAttr = outerBuilder.getArrayAttr(mapAttrs);
    auto barrierAnnotationsAttr = outerBuilder.getArrayAttr({});
    auto tokenAnnotationsAttr = outerBuilder.getArrayAttr({});

    auto regionOp =
        SubtiledRegionOp::create(outerBuilder, loc, TypeRange{}, ValueRange{},
                                 ValueRange{}, ValueRange{}, tileMappingsAttr,
                                 barrierAnnotationsAttr, tokenAnnotationsAttr);

    // --- Setup Region ---
    Block *setupBlock = &regionOp.getSetupRegion().emplaceBlock();
    OpBuilder setupBuilder = OpBuilder::atBlockEnd(setupBlock);

    SmallVector<Value> setupYields;
    if (isFirstSegment) {
      IRMapping setupMapping;
      for (Operation *op : setupOps)
        setupBuilder.clone(*op, setupMapping);
      for (Value leaf : leafValues)
        setupYields.push_back(setupMapping.lookupOrDefault(leaf));
      for (auto &entry : resolvedDiff) {
        for (unsigned t = 0; t < numTiles; ++t)
          setupYields.push_back(
              setupMapping.lookupOrDefault(entry.setupVals[t]));
      }
    } else {
      for (auto &entry : resolvedDiff) {
        for (unsigned t = 0; t < numTiles; ++t)
          setupYields.push_back(entry.setupVals[t]);
      }
    }
    for (auto *buf : outBufs) {
      for (unsigned t = 0; t < numTiles; ++t)
        setupYields.push_back(buf->smemVals[t]);
    }
    SubtiledRegionYieldOp::create(setupBuilder, loc, setupYields);

    // --- Tile Region ---
    Block *tileBlock = &regionOp.getTileRegion().emplaceBlock();
    for (Type ty : tileArgTypes)
      tileBlock->addArgument(ty, loc);
    tileBlock->addArgument(outerBuilder.getI32Type(), loc);

    OpBuilder tileBuilder = OpBuilder::atBlockEnd(tileBlock);
    IRMapping tileMapping;
    unsigned argIdx = 0;

    if (isFirstSegment)
      tileMapping.map(leafValues[0], tileBlock->getArgument(argIdx++));

    for (auto &entry : resolvedDiff) {
      Value tileArg = tileBlock->getArgument(argIdx++);
      if (entry.needsLocalLoad) {
        auto loaded = gpu::LocalLoadOp::create(
            tileBuilder, loc, entry.chainVals[0].getType(), tileArg);
        tileMapping.map(entry.chainVals[0], loaded.getResult());
      } else {
        tileMapping.map(entry.chainVals[0], tileArg);
      }
    }

    SmallVector<Value> outSmemArgs;
    for (size_t i = 0; i < outBufs.size(); ++i)
      outSmemArgs.push_back(tileBlock->getArgument(argIdx++));

    for (Operation *op : segChains[0])
      tileBuilder.clone(*op, tileMapping);

    if (hasOutgoing) {
      for (auto [buf, smemArg] : llvm::zip(transitions[segIdx], outSmemArgs)) {
        Value data = tileMapping.lookupOrDefault(buf.chainVals[0]);
        gpu::LocalStoreOp::create(tileBuilder, loc, data, smemArg);
      }
    }
    SubtiledRegionYieldOp::create(tileBuilder, loc, ValueRange{});

    // --- Teardown Region ---
    Block *teardownBlock = &regionOp.getTeardownRegion().emplaceBlock();
    OpBuilder teardownBuilder = OpBuilder::atBlockEnd(teardownBlock);
    SubtiledRegionYieldOp::create(teardownBuilder, loc, ValueRange{});
  }
  return true;
}

/// Return true if any op across the N chains has a different async_task_id
/// than the first task-annotated op.
static bool isMultiTask(ArrayRef<SmallVector<Operation *>> chains) {
  SmallVector<int32_t> firstTaskIds;
  for (auto &chain : chains) {
    for (auto *op : chain) {
      auto taskIds = getOpAsyncTaskIds(op);
      if (taskIds.empty())
        continue;
      if (firstTaskIds.empty())
        firstTaskIds = taskIds;
      else if (taskIds != firstTaskIds)
        return true;
    }
  }
  return false;
}

/// Find the latest op in `block` among all chain ops and differing operand
/// definitions, then return the op immediately after it (the insertion point
/// for the SubtiledRegionOp).
static Operation *
findInsertionPoint(Block *block, Operation *anchor,
                   ArrayRef<SmallVector<Operation *>> chains,
                   ArrayRef<SmallVector<Value>> differingOperands = {},
                   ArrayRef<EquivalenceResult::IdentityOp> identityOps = {}) {
  Operation *latest = anchor;
  auto updateLatest = [&](Operation *op) {
    if (op && op->getBlock() == block && latest->isBeforeInBlock(op))
      latest = op;
  };
  for (auto &chain : chains)
    for (auto *op : chain)
      updateLatest(op);
  for (auto &perTile : differingOperands)
    for (Value v : perTile)
      if (auto defOp = v.getDefiningOp())
        updateLatest(defOp);
  for (auto &id : identityOps)
    if (auto defOp = id.varyingOperand.getDefiningOp())
      updateLatest(defOp);
  return latest->getNextNode();
}

/// Return true if any task ID set appears non-contiguously in the segment
/// list (e.g., task A → B → A).
static bool hasNonContiguousTaskIds(ArrayRef<ChainSegment> segments) {
  SmallVector<SmallVector<int32_t>> seenTaskSets;
  for (auto &seg : segments) {
    if (seenTaskSets.empty() || seenTaskSets.back() != seg.taskIds) {
      for (size_t i = 0; i + 1 < seenTaskSets.size(); ++i) {
        if (seenTaskSets[i] == seg.taskIds)
          return true;
      }
      seenTaskSets.push_back(seg.taskIds);
    }
  }
  return false;
}

} // anonymous namespace

void tryGenerateForSplit(triton::SplitOp splitOp) {
  auto tmemLoad = traceSetupChain(splitOp);
  if (!tmemLoad)
    return;

  Block *block = splitOp->getBlock();
  Value lhs = splitOp.getOutLHS();
  Value rhs = splitOp.getOutRHS();

  // Check for nested split tree (4-tile, 8-tile, etc.).
  SmallVector<Value> leafValues;
  SmallVector<Operation *> innerSetupOps;
  collectSplitTreeLeaves(splitOp, leafValues, innerSetupOps);

  // If any leaf feeds into yet another split (not caught by the tree walker),
  // bail out — we only support complete trees.
  unsigned numTiles = leafValues.size();

  if (numTiles > 2) {
    // --- N-tile path (4, 8, ...) ---
    // Collect per-tile chains for each leaf value. The "barrier" for chain
    // collection is the last split in the tree, not the root split.
    Operation *lastSetupOp = innerSetupOps.empty()
                                 ? static_cast<Operation *>(splitOp)
                                 : innerSetupOps.back();
    SmallVector<SmallVector<Operation *>> chains;
    for (Value leaf : leafValues) {
      auto chain = collectPerTileChain(leaf, lastSetupOp, block,
                                       /*includeAuxiliary=*/false);
      if (chain.empty())
        return;
      chains.push_back(std::move(chain));
    }

    auto equiv = checkStructuralEquivalenceN(chains);
    if (!equiv)
      return;

    bool multiTask = isMultiTask(chains);

    // Collect setup ops: tmemLoad → root split + inner setup ops.
    SmallVector<Operation *> setupOps;
    for (auto it = Block::iterator(tmemLoad); it != Block::iterator(splitOp);
         ++it)
      setupOps.push_back(&*it);
    setupOps.push_back(splitOp);
    llvm::sort(innerSetupOps, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    for (auto *op : innerSetupOps)
      setupOps.push_back(op);

    Operation *insertBefore =
        findInsertionPoint(block, lastSetupOp, chains, equiv->differingOperands,
                           equiv->identityOps);
    OpBuilder builder(insertBefore);
    Location loc = splitOp.getLoc();

    bool built = false;
    if (multiTask) {
      auto segments = groupByContiguousTaskSetN(chains);
      if (!segments || segments->empty())
        return;
      if (segments->size() == 1) {
        buildSingleSubtiledRegionN(builder, loc, setupOps, leafValues, chains,
                                   *equiv);
        built = true;
      } else {
        built = buildMultiTaskSubtiledRegionsN(builder, loc, setupOps,
                                               leafValues, *segments);
      }
    } else {
      buildSingleSubtiledRegionN(builder, loc, setupOps, leafValues, chains,
                                 *equiv);
      built = true;
    }
    if (!built)
      return;

    // Erase original ops (reverse program order).
    // Chains first, then setup (which includes inner setup ops).
    for (auto &chain : llvm::reverse(chains)) {
      for (Operation *op : llvm::reverse(chain)) {
        if (op->use_empty())
          op->erase();
      }
    }
    for (Operation *op : llvm::reverse(setupOps)) {
      if (op->use_empty())
        op->erase();
    }
    return;
  }

  // --- 2-tile path (existing) ---
  SmallVector<Operation *> chain0 = collectPerTileChain(lhs, splitOp, block);
  SmallVector<Operation *> chain1 = collectPerTileChain(rhs, splitOp, block);

  if (chain0.empty() || chain1.empty())
    return;

  auto differing = checkStructuralEquivalence(chain0, chain1);
  if (!differing)
    return;

  bool hasIdentityInsertions = !differing->identityOps.empty();

  SmallVector<Operation *> setupOps;
  for (auto it = Block::iterator(tmemLoad); it != Block::iterator(splitOp);
       ++it)
    setupOps.push_back(&*it);
  setupOps.push_back(splitOp);

  SmallVector<SmallVector<Operation *>> twoChains = {chain0, chain1};
  Operation *insertBefore = findInsertionPoint(block, splitOp, twoChains);
  OpBuilder builder(insertBefore);
  Location loc = splitOp.getLoc();

  std::optional<SmallVector<ChainSegment>> segments;
  if (hasIdentityInsertions)
    segments = groupByContiguousTaskSetWithIdentity(chain0, chain1, *differing);
  else
    segments = groupByContiguousTaskSet(chain0, chain1);
  if (!segments || segments->empty())
    return;

  if (segments->size() == 1) {
    buildSingleSubtiledRegion(builder, loc, setupOps, lhs, rhs, chain0, chain1,
                              *differing);
  } else {
    // Check if task IDs form non-contiguous groups (e.g., task A → B → A).
    // This happens in addmm where the bias load (task 3) is interleaved
    // between compute ops (task 2). Merge segments with the same task ID
    // and reorder by data dependency to produce contiguous task groups.
    bool hasNonContiguousTasks = hasNonContiguousTaskIds(*segments);

    if (hasNonContiguousTasks) {
      // Merge segments with the same task ID.
      SmallVector<ChainSegment> merged;
      for (auto &seg : *segments) {
        ChainSegment *target = nullptr;
        for (auto &m : merged) {
          if (m.taskIds == seg.taskIds) {
            target = &m;
            break;
          }
        }
        if (target) {
          for (size_t t = 0; t < seg.opsPerTile.size(); ++t)
            target->opsPerTile[t].append(seg.opsPerTile[t].begin(),
                                         seg.opsPerTile[t].end());
        } else {
          merged.push_back(seg);
        }
      }

      // Topological sort by data dependency: if segment A produces values
      // consumed by segment B, A must come before B.
      unsigned n = merged.size();
      SmallVector<DenseSet<Value>> segResults(n);
      for (unsigned i = 0; i < n; ++i)
        for (auto *op : merged[i].opsPerTile[0])
          for (Value r : op->getResults())
            segResults[i].insert(r);

      SmallVector<unsigned> inDegree(n, 0);
      SmallVector<SmallVector<unsigned>> adj(n);
      for (unsigned i = 0; i < n; ++i) {
        DenseSet<unsigned> deps;
        for (auto *op : merged[i].opsPerTile[0])
          for (Value v : op->getOperands())
            for (unsigned j = 0; j < n; ++j)
              if (j != i && segResults[j].contains(v) && deps.insert(j).second)
                adj[j].push_back(i);
        inDegree[i] = deps.size();
      }

      SmallVector<unsigned> order;
      SmallVector<unsigned> worklist;
      for (unsigned i = 0; i < n; ++i)
        if (inDegree[i] == 0)
          worklist.push_back(i);
      while (!worklist.empty()) {
        unsigned u = worklist.pop_back_val();
        order.push_back(u);
        for (unsigned v : adj[u])
          if (--inDegree[v] == 0)
            worklist.push_back(v);
      }

      SmallVector<ChainSegment> reordered;
      for (unsigned idx : order)
        reordered.push_back(std::move(merged[idx]));

      // Strip identity ops from the non-template side so that per-segment
      // checkStructuralEquivalence correctly detects identity insertions.
      // Without this, both sides have the same Operation* and the identity
      // op becomes dead code in the tile body.
      unsigned nonTpl = (differing->templateChainIdx == 0) ? 1 : 0;
      for (auto &seg : reordered) {
        SmallVector<Operation *> filtered;
        for (auto *op : seg.opsPerTile[nonTpl]) {
          if (!differing->identityOpSet.count(op))
            filtered.push_back(op);
        }
        seg.opsPerTile[nonTpl] = std::move(filtered);
      }

      buildMultiTaskSubtiledRegions(builder, loc, setupOps, lhs, rhs,
                                    reordered);
    } else {
      for (size_t i = 0; i + 1 < segments->size(); ++i) {
        Operation *lastOp = (*segments)[i].opsPerTile[0].back();
        auto alloc = dyn_cast<gpu::LocalAllocOp>(lastOp);

        if (alloc && alloc.getSrc()) {
          for (size_t j = 0; j + 1 < (*segments)[i].opsPerTile[0].size(); ++j) {
            for (Value operand :
                 (*segments)[i].opsPerTile[0][j]->getOperands()) {
              if (operand == alloc.getResult())
                return;
            }
          }
        }
      }

      buildMultiTaskSubtiledRegions(builder, loc, setupOps, lhs, rhs,
                                    *segments);
    }
  }

  for (Operation *op : llvm::reverse(chain1)) {
    if (op->use_empty())
      op->erase();
  }
  for (Operation *op : llvm::reverse(chain0)) {
    if (op->use_empty())
      op->erase();
  }
  for (Operation *op : llvm::reverse(setupOps)) {
    if (op->use_empty())
      op->erase();
  }
}

namespace {

class TritonNvidiaGPUTestGenerateSubtiledRegionPass
    : public impl::TritonNvidiaGPUTestGenerateSubtiledRegionPassBase<
          TritonNvidiaGPUTestGenerateSubtiledRegionPass> {
public:
  using TritonNvidiaGPUTestGenerateSubtiledRegionPassBase::
      TritonNvidiaGPUTestGenerateSubtiledRegionPassBase;

  void runOnOperation() override {
    // Collect root splits (those tracing to tmem_load) in function bodies.
    // Process them one at a time, re-walking after each success to avoid
    // dangling pointers from erased inner splits. Track failed splits to
    // avoid infinite loops on splits that can't be processed (e.g.,
    // multi-task N-tile).
    DenseSet<Operation *> failedSplits;
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<triton::SplitOp> splitOps;
      getOperation().walk([&](triton::FuncOp funcOp) {
        for (auto &block : funcOp.getBody()) {
          for (auto &op : block) {
            if (auto splitOp = dyn_cast<triton::SplitOp>(&op))
              if (!failedSplits.contains(&op))
                splitOps.push_back(splitOp);
          }
        }
      });
      for (auto splitOp : splitOps) {
        auto tmemLoad = traceSetupChain(splitOp);
        if (!tmemLoad)
          continue;
        unsigned opCountBefore = 0;
        getOperation().walk([&](Operation *) { opCountBefore++; });
        tryGenerateForSplit(splitOp);
        unsigned opCountAfter = 0;
        getOperation().walk([&](Operation *) { opCountAfter++; });
        if (opCountBefore != opCountAfter) {
          changed = true;
        } else {
          failedSplits.insert(splitOp);
        }
        break;
      }
    }
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
