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
  /// to align segments when chains have different lengths.
  DenseSet<Operation *> identityOpSet;

  /// When forcedIdentityOps is used: for each identity op, the other chain's
  /// varying operand if the op matched (non-null), or null if the other chain
  /// lacked the op (use identity constant). Empty when forcedIdentityOps is
  /// not used.
  SmallVector<Value> identityMatchedVarying;
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
///
/// When `forcedIdentityOps` is provided, those template ops are ALWAYS
/// treated as identity — even if they match the other chain's op. When a
/// forced-identity op matches, both pointers advance but the match is
/// recorded as identity with the other chain's varying operand stored in
/// `identityMatchedVarying` for per-tile value tracking.
static std::optional<EquivalenceResult> checkStructuralEquivalence(
    ArrayRef<Operation *> chain0, ArrayRef<Operation *> chain1,
    const DenseSet<Operation *> *forcedIdentityOps = nullptr) {
  // Determine which chain is the template (longer or chain0 if same length).
  unsigned tplIdx = (chain1.size() > chain0.size()) ? 1 : 0;
  ArrayRef<Operation *> tplChain = (tplIdx == 0) ? chain0 : chain1;
  ArrayRef<Operation *> otherChain = (tplIdx == 0) ? chain1 : chain0;

  EquivalenceResult result;
  result.templateChainIdx = tplIdx;

  // Value map: template chain values → other chain values.
  llvm::DenseMap<Value, Value> valueMap;
  SmallVector<std::pair<Value, Value>> differingOperands;

  auto handleIdentityOp = [&](Operation *tOp) -> bool {
    if (!isIdentityCompatibleOp(tOp) || tOp->getNumResults() != 1)
      return false;
    bool skipped = false;
    unsigned numCandidates = isa<arith::SubIOp>(tOp) ? 1 : 2;
    for (unsigned opIdx = 0; opIdx < numCandidates; ++opIdx) {
      Value passThrough = tOp->getOperand(opIdx);
      Value varying = tOp->getOperand(1 - opIdx);
      Value otherVal = valueMap.lookup(passThrough);
      if (!otherVal)
        otherVal = passThrough;
      valueMap[tOp->getResult(0)] = otherVal;
      result.identityOps.push_back({varying, getIdentityValue(tOp)});
      result.identityOpSet.insert(tOp);
      skipped = true;
      break;
    }
    return skipped;
  };

  unsigned ti = 0, oi = 0;
  while (ti < tplChain.size() && oi < otherChain.size()) {
    Operation *tOp = tplChain[ti];
    Operation *oOp = otherChain[oi];

    // If this template op is forced to be identity, handle it without
    // calling matchOps (which would mutate valueMap/differingOperands).
    // Check if the other chain has a corresponding op (same name) — if
    // yes, it's a "matched identity" and we extract its varying operand.
    if (forcedIdentityOps && forcedIdentityOps->count(tOp)) {
      unsigned varIdx = (getIdentityValue(tOp) == 0) ? 1 : 0;
      unsigned passIdx = 1 - varIdx;

      if (oOp->getName() == tOp->getName()) {
        // Other chain has the matching op. Map the template's result to
        // the other chain's result so that downstream matchOps comparisons
        // resolve correctly (valueMap maps template → other).
        valueMap[tOp->getResult(0)] = oOp->getResult(0);

        Value otherVarying = oOp->getOperand(varIdx);
        result.identityOps.push_back(
            {tOp->getOperand(varIdx), getIdentityValue(tOp)});
        result.identityOpSet.insert(tOp);
        result.identityMatchedVarying.push_back(otherVarying);
        ti++;
        oi++;
        continue;
      }
      // Other chain lacks the op — pure identity.
      if (handleIdentityOp(tOp)) {
        result.identityMatchedVarying.push_back(Value());
        ti++;
        continue;
      }
      return std::nullopt;
    }

    if (matchOps(tOp, oOp, valueMap, differingOperands)) {
      ti++;
      oi++;
      continue;
    }

    // Ops don't match. Try identity insertion.
    if (handleIdentityOp(tOp)) {
      ti++;
      continue;
    }

    return std::nullopt;
  }

  // Handle remaining ops in the template chain.
  while (ti < tplChain.size()) {
    Operation *tOp = tplChain[ti];
    if (!handleIdentityOp(tOp))
      return std::nullopt;
    if (forcedIdentityOps)
      result.identityMatchedVarying.push_back(Value());
    ti++;
  }

  if (oi != otherChain.size())
    return std::nullopt;

  // Normalize differing operands: always (chain0 value, chain1 value).
  if (tplIdx == 0) {
    result.differingOperands = std::move(differingOperands);
  } else {
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

  /// Per-tile varying values for each identity op. identityPerTile[i][t] is
  /// the varying value for identity op i at tile t. A null Value means the
  /// tile needs the identity constant (the op was missing from that tile's
  /// chain). A non-null Value means the tile had a matching op with its own
  /// varying operand value.
  SmallVector<SmallVector<Value>> identityPerTile;
};

/// Check structural equivalence across N chains. Finds the longest chain
/// as the template and compares all others against it pairwise.
///
/// Uses a "canonical identity set" approach for identity ops: first compares
/// the template against the shortest non-template chain to discover all
/// identity ops (the canonical identity set). Then re-compares all other chains
/// with the canonical identity set forced, so identity counts are consistent
/// across all pairs. Per-tile varying values are recorded in `identityPerTile`.
static std::optional<NWayEquivalenceResult> checkStructuralEquivalenceN(
    ArrayRef<SmallVector<Operation *>> chains,
    const DenseSet<Operation *> *outerIdentityOps = nullptr) {
  assert(chains.size() >= 2);
  unsigned numTiles = chains.size();

  // Find the longest chain as template.
  unsigned tplIdx = 0;
  for (unsigned t = 1; t < numTiles; ++t) {
    if (chains[t].size() > chains[tplIdx].size())
      tplIdx = t;
  }

  // Find the shortest non-template chain — comparing template against this
  // chain discovers the maximum set of identity ops (the "canonical identity
  // set").
  unsigned shortIdx = 0;
  bool firstNonTpl = true;
  for (unsigned t = 0; t < numTiles; ++t) {
    if (t == tplIdx)
      continue;
    if (firstNonTpl || chains[t].size() < chains[shortIdx].size()) {
      shortIdx = t;
      firstNonTpl = false;
    }
  }

  // Step 1: Compare template vs shortest chain to build the canonical identity
  // set. If an outer identity set is provided (from the full-chain
  // equivalence), use it to force consistent identity detection.
  auto canonicalRes = checkStructuralEquivalence(
      chains[tplIdx], chains[shortIdx], outerIdentityOps);
  if (!canonicalRes || canonicalRes->templateChainIdx != 0)
    return std::nullopt;

  // Step 2: Re-compare all other chains with forcedIdentityOps so identity
  // counts are consistent. Use whichever identity set is larger: the one
  // discovered in step 1 or the outer one.
  SmallVector<EquivalenceResult> pairResults(numTiles);
  pairResults[shortIdx] = std::move(*canonicalRes);

  DenseSet<Operation *> &canonicalIdentity =
      pairResults[shortIdx].identityOpSet;
  const DenseSet<Operation *> *forcedSet = &canonicalIdentity;
  if (outerIdentityOps && outerIdentityOps->size() > canonicalIdentity.size())
    forcedSet = outerIdentityOps;

  for (unsigned t = 0; t < numTiles; ++t) {
    if (t == tplIdx || t == shortIdx)
      continue;
    auto res = checkStructuralEquivalence(chains[tplIdx], chains[t], forcedSet);
    if (!res || res->templateChainIdx != 0)
      return std::nullopt;
    pairResults[t] = std::move(*res);
  }

  // Validate consistent counts.
  unsigned numDiff = pairResults[shortIdx].differingOperands.size();
  unsigned numIdentity = pairResults[shortIdx].identityOps.size();
  for (unsigned t = 0; t < numTiles; ++t) {
    if (t == tplIdx)
      continue;
    if (pairResults[t].differingOperands.size() != numDiff ||
        pairResults[t].identityOps.size() != numIdentity)
      return std::nullopt;
  }

  NWayEquivalenceResult result;
  result.templateChainIdx = tplIdx;
  result.identityOps = pairResults[shortIdx].identityOps;
  result.identityOpSet = pairResults[shortIdx].identityOpSet;

  // Build differing operands.
  for (unsigned i = 0; i < numDiff; ++i) {
    SmallVector<Value> perTile(numTiles);
    perTile[tplIdx] = pairResults[shortIdx].differingOperands[i].first;
    for (unsigned t = 0; t < numTiles; ++t) {
      if (t == tplIdx)
        continue;
      perTile[t] = pairResults[t].differingOperands[i].second;
    }
    result.differingOperands.push_back(std::move(perTile));
  }

  // Build per-tile identity varying values.
  for (unsigned i = 0; i < numIdentity; ++i) {
    SmallVector<Value> perTile(numTiles);
    perTile[tplIdx] = result.identityOps[i].varyingOperand;
    for (unsigned t = 0; t < numTiles; ++t) {
      if (t == tplIdx)
        continue;
      if (!pairResults[t].identityMatchedVarying.empty())
        perTile[t] = pairResults[t].identityMatchedVarying[i]; // may be null
      // else: null (identity constant)
    }
    result.identityPerTile.push_back(std::move(perTile));
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
/// that transitively depend on `splitResult`, plus auxiliary ops that are
/// needed by the chain but don't depend on the split result (e.g., address
/// offset computations like arith.addi).
///
/// `excludeOps` allows callers to prevent specific ops (e.g., inner setup
/// ops from nested split trees) from being captured by the auxiliary walk.
static SmallVector<Operation *>
collectPerTileChain(Value splitResult, Operation *splitOp, Block *block,
                    const DenseSet<Operation *> &excludeOps = {}) {
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
      if (excludeOps.contains(user))
        continue;
      if (!visited.insert(user).second)
        continue;
      chain.push_back(user);
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }

  // Auxiliary walk: recursively collect ops that chain ops depend on but
  // that don't themselves depend on the split result.
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
      if (excludeOps.contains(defOp))
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
  llvm::sort(fullChain,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return fullChain;
}

/// Group N chains by contiguous async task set, handling identity-compatible
/// ops that may make the template chain longer than the others.
///
/// Walks the template chain (identified by `equiv.templateChainIdx`) and
/// pairs each non-identity op with the corresponding op in every other chain.
/// Identity ops (present only in the template) are placed in all tiles'
/// opsPerTile. When chains have equal length (no identity ops), this
/// degenerates to simple positional pairing.
static std::optional<SmallVector<ChainSegment>>
groupByContiguousTaskSet(ArrayRef<SmallVector<Operation *>> chains,
                         const NWayEquivalenceResult &equiv) {
  unsigned numTiles = chains.size();
  assert(numTiles >= 2);
  unsigned tplIdx = equiv.templateChainIdx;
  ArrayRef<Operation *> tplChain = chains[tplIdx];
  if (tplChain.empty())
    return std::nullopt;

  // All non-template chains have the same length (template length minus
  // identity ops). Use a single "other" pointer that advances for all.
  SmallVector<ChainSegment> segments;
  unsigned oi = 0;
  for (size_t ti = 0; ti < tplChain.size(); ++ti) {
    auto taskIds = getOpAsyncTaskIds(tplChain[ti]);

    if (taskIds.empty() && !segments.empty()) {
      // Ops without task IDs join the current segment.
    } else if (segments.empty() || segments.back().taskIds != taskIds) {
      ChainSegment seg;
      seg.opsPerTile.resize(numTiles);
      seg.taskIds = taskIds;
      segments.push_back(std::move(seg));
    }

    if (equiv.identityOpSet.count(tplChain[ti])) {
      for (unsigned t = 0; t < numTiles; ++t)
        segments.back().opsPerTile[t].push_back(tplChain[ti]);
    } else {
      for (unsigned t = 0; t < numTiles; ++t) {
        if (t == tplIdx)
          segments.back().opsPerTile[t].push_back(tplChain[ti]);
        else
          segments.back().opsPerTile[t].push_back(chains[t][oi]);
      }
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
  // When identityPerTile is populated (mixed identity — some tiles have
  // the op, others don't), yield N values per identity op (one per tile).
  // Otherwise, yield 2 values (varying + identity_const).
  bool hasMixedIdentity = !equiv.identityPerTile.empty();
  for (auto [i, id] : llvm::enumerate(equiv.identityOps)) {
    tileArgTypes.push_back(id.varyingOperand.getType());
    if (hasMixedIdentity) {
      for (unsigned t = 0; t < numTiles; ++t)
        tileMappings[t].push_back(yieldIdx + t);
      yieldIdx += numTiles;
    } else {
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
  if (!equiv.identityPerTile.empty()) {
    // Mixed identity: yield N values per identity op (per-tile varying
    // values, with identity constants for tiles missing the op).
    for (auto [i, id] : llvm::enumerate(equiv.identityOps)) {
      for (unsigned t = 0; t < numTiles; ++t) {
        Value v = equiv.identityPerTile[i][t];
        if (v)
          setupYieldValues.push_back(setupMapping.lookupOrDefault(v));
        else
          setupYieldValues.push_back(arith::ConstantOp::create(
              setupBuilder, loc,
              setupBuilder.getIntegerAttr(id.varyingOperand.getType(),
                                          id.identityVal)));
      }
    }
  } else {
    for (auto &id : equiv.identityOps) {
      Value varying = setupMapping.lookupOrDefault(id.varyingOperand);
      Value identityConst = arith::ConstantOp::create(
          setupBuilder, loc,
          setupBuilder.getIntegerAttr(id.varyingOperand.getType(),
                                      id.identityVal));
      setupYieldValues.push_back(varying);
      setupYieldValues.push_back(identityConst);
    }
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

/// Build multiple SubtiledRegionOps for N-tile chains spanning multiple
/// async task sets. Uses implicit buffering (Option 2) at segment
/// transitions — cross-segment tensor values are communicated through SMEM.
static bool buildMultiTaskSubtiledRegionsN(
    OpBuilder &outerBuilder, Location loc, ArrayRef<Operation *> setupOps,
    ArrayRef<Value> leafValues, ArrayRef<ChainSegment> segments,
    const DenseSet<Operation *> *canonicalIdentityOps = nullptr) {
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

  // --- Validation pass: check all bail conditions and segment equivalence
  // before creating any IR. This prevents partial modifications that would
  // change the op count and cause the fixpoint loop to retry indefinitely.

  // Detect cross-segment tensor values for each transition.
  struct TransitionInfoN {
    llvm::MapVector<Value, SmallVector<Value>> crossSegVals;
  };
  SmallVector<TransitionInfoN> transitionInfos;

  for (size_t i = 0; i + 1 < segments.size(); ++i) {
    DenseSet<Value> seg0Results;
    for (auto *op : segments[i].opsPerTile[0])
      for (Value r : op->getResults())
        seg0Results.insert(r);

    TransitionInfoN info;
    for (size_t opIdx = 0; opIdx < segments[i + 1].opsPerTile[0].size();
         ++opIdx) {
      for (unsigned t = 0; t < numTiles; ++t) {
        for (Value v : segments[i + 1].opsPerTile[t][opIdx]->getOperands()) {
          if (t == 0 && seg0Results.contains(v) &&
              !info.crossSegVals.count(v)) {
            SmallVector<Value> perTile(numTiles);
            perTile[0] = v;
            info.crossSegVals[v] = std::move(perTile);
          }
        }
      }
    }
    for (size_t opIdx = 0; opIdx < segments[i + 1].opsPerTile[0].size();
         ++opIdx) {
      auto *op0 = segments[i + 1].opsPerTile[0][opIdx];
      for (auto [oprIdx, v0] : llvm::enumerate(op0->getOperands())) {
        auto it = info.crossSegVals.find(v0);
        if (it == info.crossSegVals.end())
          continue;
        for (unsigned t = 1; t < numTiles; ++t)
          it->second[t] =
              segments[i + 1].opsPerTile[t][opIdx]->getOperand(oprIdx);
      }
    }
    for (auto &[v0, perTile] : info.crossSegVals) {
      if (!isa<RankedTensorType>(v0.getType()))
        return false;
    }
    transitionInfos.push_back(std::move(info));
  }

  SmallVector<NWayEquivalenceResult, 4> segEquivs;
  for (size_t segIdx = 0; segIdx < segments.size(); ++segIdx) {
    SmallVector<SmallVector<Operation *>> segChains;
    for (auto &ops : segments[segIdx].opsPerTile)
      segChains.push_back(SmallVector<Operation *>(ops));
    auto segEquiv =
        checkStructuralEquivalenceN(segChains, canonicalIdentityOps);
    if (!segEquiv)
      return false;
    segEquivs.push_back(std::move(*segEquiv));
  }

  // --- All checks passed. Now create SMEM allocs and SubtiledRegionOps.

  SmallVector<SmallVector<BufferEntryN>> transitions;
  for (size_t i = 0; i < transitionInfos.size(); ++i) {
    SmallVector<BufferEntryN> bufs;
    for (auto &[v0, perTile] : transitionInfos[i].crossSegVals) {
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
    bool hasOutgoing = (segIdx < transitions.size());
    bool hasIncoming = (segIdx > 0);

    auto &segEquiv = segEquivs[segIdx];
    auto &segDiff = segEquiv.differingOperands;

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

    // Check if this segment's task IDs match any setup op's task IDs.
    // If so, this segment "owns" the setup — it clones the setup ops and
    // gets the leaf values as tile args.
    bool ownsSetup = false;
    for (auto *op : setupOps) {
      auto opTasks = getOpAsyncTaskIds(op);
      if (!opTasks.empty() && opTasks == seg.taskIds) {
        ownsSetup = true;
        break;
      }
    }

    // Build tile arg types and N-way mappings.
    SmallVector<Type> tileArgTypes;
    SmallVector<SmallVector<int32_t>> tileMaps(numTiles);
    int32_t yieldIdx = 0;

    if (ownsSetup) {
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
    if (ownsSetup) {
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

    if (ownsSetup)
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

    // Clone from tile 0's ops (the template chain for this segment).
    for (Operation *op : seg.opsPerTile[0])
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

  // Check for nested split tree (4-tile, 8-tile, etc.).
  SmallVector<Value> leafValues;
  SmallVector<Operation *> innerSetupOps;
  collectSplitTreeLeaves(splitOp, leafValues, innerSetupOps);
  unsigned numTiles = leafValues.size();

  // Collect per-tile chains. For nested splits, exclude inner setup ops
  // from auxiliary collection to avoid capturing other branches.
  Operation *lastSetupOp = innerSetupOps.empty()
                               ? static_cast<Operation *>(splitOp)
                               : innerSetupOps.back();
  DenseSet<Operation *> excludeOps(innerSetupOps.begin(), innerSetupOps.end());
  SmallVector<SmallVector<Operation *>> chains;
  for (Value leaf : leafValues) {
    auto chain = collectPerTileChain(leaf, lastSetupOp, block, excludeOps);
    if (chain.empty())
      return;
    chains.push_back(std::move(chain));
  }

  auto equiv = checkStructuralEquivalenceN(chains);
  if (!equiv)
    return;

  // Collect setup ops: tmemLoad → root split + inner setup ops.
  SmallVector<Operation *> setupOps;
  for (auto it = Block::iterator(tmemLoad); it != Block::iterator(splitOp);
       ++it)
    setupOps.push_back(&*it);
  setupOps.push_back(splitOp);
  llvm::sort(innerSetupOps,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (auto *op : innerSetupOps)
    setupOps.push_back(op);

  Operation *insertBefore = findInsertionPoint(
      block, lastSetupOp, chains, equiv->differingOperands, equiv->identityOps);
  OpBuilder builder(insertBefore);
  Location loc = splitOp.getLoc();

  bool built = false;
  bool multiTask = isMultiTask(chains);

  if (!multiTask) {
    buildSingleSubtiledRegionN(builder, loc, setupOps, leafValues, chains,
                               *equiv);
    built = true;
  } else {
    auto segments = groupByContiguousTaskSet(chains, *equiv);
    if (!segments || segments->empty())
      return;

    if (segments->size() == 1) {
      buildSingleSubtiledRegionN(builder, loc, setupOps, leafValues, chains,
                                 *equiv);
      built = true;
    } else if (hasNonContiguousTaskIds(*segments)) {
      // Merge segments with the same task ID and topologically sort by
      // data dependency to produce contiguous task groups.
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
      for (auto &seg : reordered) {
        for (unsigned t = 0; t < seg.opsPerTile.size(); ++t) {
          if (t == equiv->templateChainIdx)
            continue;
          SmallVector<Operation *> filtered;
          for (auto *op : seg.opsPerTile[t]) {
            if (!equiv->identityOpSet.count(op))
              filtered.push_back(op);
          }
          seg.opsPerTile[t] = std::move(filtered);
        }
      }

      built = buildMultiTaskSubtiledRegionsN(builder, loc, setupOps, leafValues,
                                             reordered, &equiv->identityOpSet);
    } else {
      // Strip identity ops from non-template tiles so per-segment
      // equivalence detects them correctly (otherwise all tiles have the
      // same Operation* and identity is invisible).
      SmallVector<ChainSegment> strippedSegments(segments->begin(),
                                                 segments->end());
      for (auto &seg : strippedSegments) {
        for (unsigned t = 0; t < seg.opsPerTile.size(); ++t) {
          if (t == equiv->templateChainIdx)
            continue;
          SmallVector<Operation *> filtered;
          for (auto *op : seg.opsPerTile[t]) {
            if (!equiv->identityOpSet.count(op))
              filtered.push_back(op);
          }
          seg.opsPerTile[t] = std::move(filtered);
        }
      }
      built = buildMultiTaskSubtiledRegionsN(builder, loc, setupOps, leafValues,
                                             strippedSegments,
                                             &equiv->identityOpSet);
    }
  }

  if (!built)
    return;

  // Erase original ops (reverse program order).
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
                // Skip splits inside SubtiledRegionOp regions (cloned setup
                // chains) to avoid infinite re-processing.
                if (!op.getParentOfType<SubtiledRegionOp>())
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
