#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUGENERATESUBTILEDREGIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

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

/// Check if two per-tile op chains are structurally equivalent.
/// Returns a list of (operand_from_chain0, operand_from_chain1) pairs
/// for operands that differ between the two chains.
/// Returns nullopt if the chains are not structurally equivalent.
static std::optional<SmallVector<std::pair<Value, Value>>>
checkStructuralEquivalence(ArrayRef<Operation *> chain0,
                           ArrayRef<Operation *> chain1) {
  if (chain0.size() != chain1.size())
    return std::nullopt;

  llvm::DenseMap<Value, Value> valueMap;
  SmallVector<std::pair<Value, Value>> differingOperands;

  for (auto [op0, op1] : llvm::zip(chain0, chain1)) {
    if (op0->getName() != op1->getName())
      return std::nullopt;
    if (op0->getNumOperands() != op1->getNumOperands())
      return std::nullopt;
    if (op0->getNumResults() != op1->getNumResults())
      return std::nullopt;
    for (auto [r0, r1] : llvm::zip(op0->getResults(), op1->getResults())) {
      if (r0.getType() != r1.getType())
        return std::nullopt;
    }
    if (op0->getAttrDictionary() != op1->getAttrDictionary())
      return std::nullopt;

    for (auto [v0, v1] : llvm::zip(op0->getOperands(), op1->getOperands())) {
      auto it = valueMap.find(v0);
      if (it != valueMap.end()) {
        if (it->second != v1)
          return std::nullopt;
        continue;
      }
      if (v0 == v1)
        continue;
      differingOperands.push_back({v0, v1});
      valueMap[v0] = v1;
    }

    for (auto [r0, r1] : llvm::zip(op0->getResults(), op1->getResults()))
      valueMap[r0] = r1;
  }

  return differingOperands;
}

/// Collect the per-tile op chain for a split result: all ops in the block
/// that transitively depend on `splitResult`, plus any ops needed exclusively
/// by those (e.g., offset computations).
static SmallVector<Operation *>
collectPerTileChain(Value splitResult, Operation *splitOp, Block *block) {
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

  // Also collect ops that are needed by the chain but don't depend on the
  // split result (e.g., offset computations like arith.addi %col, %half_n).
  llvm::DenseSet<Operation *> chainSet(chain.begin(), chain.end());
  for (Operation *op : chain) {
    for (Value operand : op->getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != block)
        continue;
      if (defOp->isBeforeInBlock(splitOp) || defOp == splitOp)
        continue;
      if (!chainSet.contains(defOp) && visited.insert(defOp).second)
        chainSet.insert(defOp);
    }
  }

  // Rebuild as sorted list.
  SmallVector<Operation *> fullChain;
  for (Operation *op : chainSet)
    fullChain.push_back(op);
  llvm::sort(fullChain,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return fullChain;
}

} // anonymous namespace

void tryGenerateForSplit(triton::SplitOp splitOp) {
  auto tmemLoad = traceSetupChain(splitOp);
  if (!tmemLoad)
    return;

  Block *block = splitOp->getBlock();
  Value lhs = splitOp.getOutLHS();
  Value rhs = splitOp.getOutRHS();

  SmallVector<Operation *> chain0 = collectPerTileChain(lhs, splitOp, block);
  SmallVector<Operation *> chain1 = collectPerTileChain(rhs, splitOp, block);

  if (chain0.empty() || chain1.empty())
    return;

  auto differing = checkStructuralEquivalence(chain0, chain1);
  if (!differing)
    return;

  // --- Build the SubtiledRegionOp ---
  OpBuilder builder(tmemLoad);
  Location loc = splitOp.getLoc();

  // Collect setup ops: everything from tmemLoad to splitOp (inclusive).
  SmallVector<Operation *> setupOps;
  for (auto it = Block::iterator(tmemLoad); it != Block::iterator(splitOp);
       ++it)
    setupOps.push_back(&*it);
  setupOps.push_back(splitOp);

  // Tile arg types and mappings.
  SmallVector<Type> tileArgTypes;
  SmallVector<int32_t> tile0Mapping, tile1Mapping;

  // Tile arg 0: split result.
  tileArgTypes.push_back(lhs.getType());
  tile0Mapping.push_back(0);
  tile1Mapping.push_back(1);

  // Additional tile args from differing operands.
  for (auto [i, pair] : llvm::enumerate(*differing)) {
    tileArgTypes.push_back(pair.first.getType());
    tile0Mapping.push_back(2 + 2 * i);
    tile1Mapping.push_back(2 + 2 * i + 1);
  }

  auto tileMappingsAttr = builder.getArrayAttr(
      {DenseI32ArrayAttr::get(builder.getContext(), tile0Mapping),
       DenseI32ArrayAttr::get(builder.getContext(), tile1Mapping)});
  auto barrierAnnotationsAttr = builder.getArrayAttr({});

  auto regionOp = SubtiledRegionOp::create(
      builder, loc, /*resultTypes=*/TypeRange{},
      /*barriers=*/ValueRange{}, /*barrierPhases=*/ValueRange{},
      tileMappingsAttr, barrierAnnotationsAttr);

  // --- Setup Region ---
  Block *setupBlock = &regionOp.getSetupRegion().emplaceBlock();
  OpBuilder setupBuilder = OpBuilder::atBlockEnd(setupBlock);
  IRMapping setupMapping;
  for (Operation *op : setupOps)
    setupBuilder.clone(*op, setupMapping);

  SmallVector<Value> setupYieldValues;
  setupYieldValues.push_back(setupMapping.lookupOrDefault(lhs));
  setupYieldValues.push_back(setupMapping.lookupOrDefault(rhs));
  for (auto [v0, v1] : *differing) {
    setupYieldValues.push_back(setupMapping.lookupOrDefault(v0));
    setupYieldValues.push_back(setupMapping.lookupOrDefault(v1));
  }
  SubtiledRegionYieldOp::create(setupBuilder, loc, setupYieldValues);

  // --- Tile Region ---
  Block *tileBlock = &regionOp.getTileRegion().emplaceBlock();
  for (Type ty : tileArgTypes)
    tileBlock->addArgument(ty, loc);

  OpBuilder tileBuilder = OpBuilder::atBlockEnd(tileBlock);
  IRMapping tileMapping;
  tileMapping.map(lhs, tileBlock->getArgument(0));
  for (auto [i, pair] : llvm::enumerate(*differing))
    tileMapping.map(pair.first, tileBlock->getArgument(1 + i));

  for (Operation *op : chain0)
    tileBuilder.clone(*op, tileMapping);
  SubtiledRegionReturnOp::create(tileBuilder, loc);

  // --- Teardown Region ---
  Block *teardownBlock = &regionOp.getTeardownRegion().emplaceBlock();
  OpBuilder teardownBuilder = OpBuilder::atBlockEnd(teardownBlock);
  SubtiledRegionTeardownOp::create(teardownBuilder, loc, ValueRange{});

  // --- Erase original ops (reverse program order) ---
  for (Operation *op : llvm::reverse(chain1))
    op->erase();
  for (Operation *op : llvm::reverse(chain0))
    op->erase();
  for (Operation *op : llvm::reverse(setupOps)) {
    if (op->use_empty())
      op->erase();
  }
}

namespace {

class TritonNvidiaGPUGenerateSubtiledRegionPass
    : public impl::TritonNvidiaGPUGenerateSubtiledRegionPassBase<
          TritonNvidiaGPUGenerateSubtiledRegionPass> {
public:
  using TritonNvidiaGPUGenerateSubtiledRegionPassBase::
      TritonNvidiaGPUGenerateSubtiledRegionPassBase;

  void runOnOperation() override {
    SmallVector<triton::SplitOp> splitOps;
    getOperation().walk([&](triton::SplitOp op) { splitOps.push_back(op); });
    for (auto splitOp : splitOps)
      tryGenerateForSplit(splitOp);
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
