#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <deque>

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUREMOVELAYOUTCONVERSIONS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-remove-layout-conversions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// The current algorithm works by analyzing the IR and doing a one-shot rewrite
// based on the analysis. The algorithm is as follows.
//
// 1. Find all the anchor ops. These are ops that have a layout we want to
//    preserve.
//
// 2. For each anchor, propagate its layout to all its descendants.
//    An op can have multiple ancestors that are anchors, so at this stage an op
//    may have multiple layouts associated with it.
//
// 3. Resolve conflicts by deciding which of the multiple layouts the op should
//    keep, inserting convert-layout ops to resolve conflicts.  After this
//    stage, each value has only one layout associated with it.
//
// 4. Rewrite the IR by walking the function in dominance order. Since we
//    assume the IR is structured we just need to process the regions in the
//    correct order. For each op, rewrite it using the layout decided by the
//    analysis phase.
class LayoutPropagation {
public:
  // Structure to keep track of the layout associated to a value.
  struct LayoutInfo {
    LayoutInfo(Attribute encoding) { encodings.insert(encoding); }
    LayoutInfo() {}
    llvm::SmallSetVector<Attribute, 8> encodings;
  };
  LayoutPropagation(FuncOp F, unsigned smemBudget = 0)
      : funcOp(F), smemBudget(smemBudget) {}
  // Find the anchor ops and set their layout in the data structure.
  void initAnchorLayout();
  // Recursively Propagate the layout to all the users of the anchor ops until
  // we reach a fix point.
  void propagateLayout();
  // Add layouts given in `Info` to the uses of `value`.
  SmallVector<Value> propagateToUsers(Value value, LayoutInfo &info);
  // Set the encoding to all the values and fill out the values with new layout
  // in `changed`.
  void setEncoding(ValueRange values, LayoutInfo &info,
                   SmallVector<Value> &changed, Operation *op);
  // Resolve cases where a value has multiple layouts associated to it.
  void resolveConflicts();
  // Rewrite the IR for the full module.
  void rewrite();
  // Rewrite the IR for a region.
  void rewriteRegion(Region &R);
  // Rewrite an op based on the layout picked by the analysis.
  void rewriteOp(Operation *op);
  // Rewrite a for op based on the layout picked by the analysis.
  void rewriteForOp(scf::ForOp forOp);
  void rewriteWhileOp(scf::WhileOp whileOp);
  void rewriteIfOp(scf::IfOp ifOp);
  void rewriteYieldOp(scf::YieldOp yieldOp);
  void rewriteConditionOp(scf::ConditionOp conditionOp);
  void rewriteReduceToScalar(Operation *reduceOp);
  void rewriteAssertOp(AssertOp assertOp);
  Attribute getEncodingBeforeRewrite(Value value) const;
  void setEncodingInPlace(Value value, Attribute encoding);
  void rewriteGenericOpInPlace(Operation *op, Attribute encoding);
  // Return the mapped value in the given encoding. This will insert a convert
  // if the encoding is different than the encoding decided at resolve time.
  Value getValueAs(Value value, Attribute encoding);
  // Dump the current stage of layout information.
  void dump();

private:
  // map from value to layout information.
  llvm::MapVector<Value, LayoutInfo> layouts;
  // original encodings of tensor values rewritten in place.
  DenseMap<Value, Attribute> originalEncodings;
  FuncOp funcOp;
  unsigned smemBudget;
};

class LayoutRematerialization {
public:
  LayoutRematerialization(FuncOp F, unsigned smemBudget = 0)
      : funcOp(F), smemBudget(smemBudget) {}

  // Map the original value to the remat'ed one.
  void addRematValue(Value old, Attribute encoding, Value newV);
  // Get the remat'ed value in the given encoding, if one already exists and
  // is different then the layout conversion root.
  Value getRematValue(Value value, Attribute encoding) const {
    return rematMapping.lookup({value, encoding});
  }

  bool backwardRematerialization();

  /// Rematerialize the backward slice leading up to \p convertOp to produce the
  /// result layout directly if it is possible and profitable to do so.
  /// \return true if \p convertOp was eliminated, false otherwise.
  bool backwardRematerialization(ConvertLayoutOp convertOp);

  // TODO: Merge the three hoistConvert*(); functions as they are duplicate code
  void hoistConvertDotOperand();
  void hoistConvertOnTopOfExtOrBroadcast();
  void hoistConvertIntoConditionals();

  /// Attempt to hoist \p convertOp above operations that make the tensor larger
  /// and costlier to convert (e.g. ExtFOp and BroadcastOp). If this is
  /// possible, rematerialize the slice between the convert and that operation
  /// and hoist the convert above it.
  /// \return true if \p convertOp was hoisted, false otherwise.
  bool hoistConvertOnTopOfExtOrBroadcast(ConvertLayoutOp convertOp);

  /// Attempt to hoist \p convertOp into conditionals so the conversion is only
  /// conditionally executed. If this is possible, rematerialize the slice
  /// between the convert and the conditional and move the convert inside.
  /// \return true if \p convertOp was hoisted, false otherwise.
  bool hoistConvertIntoConditionals(ConvertLayoutOp convertOp);

  bool hoistConvertDotOperand(ConvertLayoutOp convertOp);

  void rewriteSlice(
      SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
      const DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
      ConvertLayoutOp convertOp, IRMapping &mapping);
  void rewriteSlice(
      SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
      const DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
      ConvertLayoutOp convertOp);

  /// Invokes the utility function getConvertBackwardSlice with a callback for
  /// checking whether a rematerialization for a particular value already
  /// exists. Any value that has an existing rematerialization for all of its
  /// uses will have that rematerialization inserted in \p existingRemats, and
  /// will not have its operands traversed for inclusion in \p slice.
  LogicalResult getConvertBackwardSlice(
      OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
      DenseMap<Value, Attribute> &layout,
      DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
      std::function<bool(Operation *)> stopPropagation);

  LogicalResult getRematerializableSlice(
      OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
      DenseMap<Value, Attribute> &layout,
      DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
      std::function<bool(Operation *)> stopPropagation = nullptr);

private:
  void updateRematMapping(SmallVector<std::tuple<Value, Value>> &values);
  // Existing tuples of (value, layout) that needs to be updated when recreating
  // scf ops. This prevents keeping track of Values that have been delete when
  // rewriting slices.
  DenseMap<Value, Attribute> mappedValues;
  // map of the values remat based on encoding.
  DenseMap<std::pair<Value, Attribute>, Value> rematMapping;
  FuncOp funcOp;
  // Meta extension: when > 0, the remove-layout pass is operating under a SMEM
  // budget and convert-elimination hoists bypass the upstream perf cost gate.
  unsigned smemBudget;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
};

void LayoutRematerialization::addRematValue(Value old, Attribute encoding,
                                            Value newV) {
  LDBG("addRematValue " << old << " encoding " << encoding << " " << newV);
  rematMapping[{old, encoding}] = newV;
  mappedValues[old] = encoding;
}

// Facebook begin
// Look ahead to at the transitive uses and see if there is a convert to mma
// operations.
static bool hasConvertToMMATransisitiveUse(Operation *op, Attribute encoding) {
  SmallVector<Value> queue = {op->getResult(0)};
  SetVector<Operation *> forwardSlice;
  llvm::SmallDenseSet<Value> seen;
  while (!queue.empty()) {
    Value currentValue = queue.back();
    queue.pop_back();
    getForwardSlice(currentValue, &forwardSlice);
    for (Operation *op : forwardSlice) {
      // HACK: Stop propagation if the ReduceOp is using mma layout but is
      // producing tensor smaller than the layout we would like to propagate.
      // This is to avoid stepping into the known bug.
      if (isa<mlir::triton::ReduceOp>(op)) {
        auto tensorType =
            dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        if (tensorType &&
            isa<NvidiaMmaEncodingAttr>(tensorType.getEncoding())) {
          auto mmaInstrShape =
              cast<NvidiaMmaEncodingAttr>(encoding).getInstrShape();
          if (tensorType.getShape()[tensorType.getRank() - 2] <
                  mmaInstrShape[0] ||
              tensorType.getShape()[tensorType.getRank() - 1] <
                  mmaInstrShape[1]) {
            return false;
          }
        }
      }

      if (auto convertOp = dyn_cast<ConvertLayoutOp>(op)) {
        Attribute dstEncoding = convertOp.getType().getEncoding();
        if (auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(dstEncoding))
          return (mmaLayout.getVersionMajor() > 1) ? true
                                                   : mmaLayout == encoding;
        if (isa<triton::gpu::AMDMfmaEncodingAttr,
                triton::gpu::AMDWmmaEncodingAttr>(dstEncoding))
          return true;
        if (isa<triton::gpu::DotOperandEncodingAttr>(dstEncoding)) {
          if (auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(encoding)) {
            return mmaLayout.getVersionMajor() > 1;
          } else {
            assert((mlir::isa<triton::gpu::AMDMfmaEncodingAttr,
                              triton::gpu::AMDWmmaEncodingAttr>(encoding)));
            return true;
          }
        }
      }
      bool isMMAV3 =
          isa<NvidiaMmaEncodingAttr>(encoding) &&
          cast<NvidiaMmaEncodingAttr>(encoding).getVersionMajor() == 3;
      if (isMMAV3 && (isa<LocalAllocOp>(op) || isa<LocalStoreOp>(op)))
        return true;
      auto yield = dyn_cast<scf::YieldOp>(op);
      if (!yield)
        continue;
      if (auto ifOp = dyn_cast<scf::IfOp>(yield->getParentOp())) {
        for (OpOperand &operand : yield->getOpOperands()) {
          Operation *def = operand.get().getDefiningOp();
          if (def &&
              (forwardSlice.count(def) || operand.get() == currentValue) &&
              (seen.insert(operand.get()).second == true))
            queue.push_back(ifOp.getResult(operand.getOperandNumber()));
        }
      }
      auto forOp = dyn_cast<scf::ForOp>(yield.getOperation()->getParentOp());
      if (!forOp)
        continue;
      for (OpOperand &operand : yield->getOpOperands()) {
        Operation *def = operand.get().getDefiningOp();
        if (def && (forwardSlice.count(def) || operand.get() == currentValue) &&
            (seen.insert(operand.get()).second == true))
          queue.push_back(forOp.getRegionIterArg(operand.getOperandNumber()));
      }
    }
  }
  return false;
}
// Facebook end

// Return true if the op is an op with a layout we don't want to change. We will
// propagate the layout starting from anchor ops.
bool isLayoutAnchor(Operation *op) {
  // A user-pinned result (an encoding carrying PinnedEncodingTrait, e.g. TLX's
  // #tlx.user_layout) is a hard anchor regardless of the producing op: the user
  // explicitly chose that layout, so layout optimization must not rewrite it.
  for (Value result : op->getResults())
    if (auto rankedTy = dyn_cast<RankedTensorType>(result.getType()))
      if (isa_and_nonnull<PinnedEncodingTrait>(rankedTy.getEncoding()))
        return true;

  if (isa<DescriptorOpInterface>(op))
    return true;
  if (isa<LoadOp, StoreOp>(op))
    return isExpensiveLoadOrStore(op);
  // local_load is expensive as it reads from shared memory with specific layout
  if (isa<triton::gpu::LocalLoadOp>(op))
    return isExpensiveLocalLoad(op);
  if (isa<DotOp, DotScaledOp, nvidia_gpu::WarpGroupDotOp, AtomicRMWOp,
          AtomicCASOp, triton::nvidia_gpu::TMEMLoadOp>(op))
    return true;
  if (auto gatherOp = dyn_cast<GatherOp>(op))
    return gatherOp.getEfficientLayout();

  // Heuristic: Mark permuting reshape as a layout anchor.  Its dst can be
  // anything, so it stops forward-propagation of layouts.  We rely on the
  // backwards pass to fix it up if necessary.  (If we didn't do this, then
  // anything following the reshape won't be covered by the forward pass at
  // all.)
  if (auto reshape = dyn_cast<ReshapeOp>(op))
    return reshape.getAllowReorder();

  return false;
}

void LayoutPropagation::initAnchorLayout() {
  auto addAnchor = [&](Value v) {
    if (auto tensorType = dyn_cast<RankedTensorType>(v.getType())) {
      // Facebook begin
      // Workaround, don't popagate MMA layout unless there is a convert
      // back to mma further down to avoid generating reduction with MMA
      // layout that may have lower performance.
      // This can be improved with more aggressive backward propagation.
      if (isa<MmaEncodingTrait>(tensorType.getEncoding()) &&
          v.getDefiningOp() &&
          !hasConvertToMMATransisitiveUse(v.getDefiningOp(),
                                          tensorType.getEncoding())) {
        return;
      }
      // Facebook end
      layouts.insert({v, LayoutInfo(tensorType.getEncoding())});
    }
  };

  // Consider function args as anchors.  This makes it easier to write tests --
  // you can pass a tensor with an encoding as an arg, instead of explicitly
  // calling tt.load.
  for (auto arg : funcOp.getArguments()) {
    addAnchor(arg);
  }

  funcOp.walk([&](Operation *op) {
    if (isLayoutAnchor(op)) {
      for (auto result : op->getResults()) {
        addAnchor(result);
      }
    }
  });
}

void LayoutPropagation::setEncoding(ValueRange values, LayoutInfo &info,
                                    SmallVector<Value> &changed,
                                    Operation *op) {
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    bool hasChanged = false;
    for (auto encoding : info.encodings) {
      Attribute dstEncoding;
      if (isa<ConvertLayoutOp>(op)) {
        // Try to remove the convert by making the dst encoding match the source
        // encoding.
        dstEncoding = encoding;
      } else {
        dstEncoding = inferDstEncoding(op, encoding);
      }
      if (dstEncoding)
        hasChanged |= layouts[value].encodings.insert(dstEncoding);
    }
    if (hasChanged)
      changed.push_back(value);
  }
}

SmallVector<Value> LayoutPropagation::propagateToUsers(Value value,
                                                       LayoutInfo &info) {
  SmallVector<Value> changed;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      setEncoding({arg, result}, info, changed, user);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      Value arg = whileOp.getBeforeArguments()[use.getOperandNumber()];
      setEncoding({arg}, info, changed, user);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto parent = yieldOp->getParentOp();
      SmallVector<Value> valuesToPropagate;
      // For scf.ForOp / scf.IfOp the yield's operand i corresponds 1:1 to
      // the parent's result i, so we can propagate directly.
      // For scf.WhileOp, the do-region yield feeds the before-block args
      // (handled via `whileOp.getBeforeArguments()[i]` below), NOT the
      // parent results. The parent result mapping is determined by
      // scf.condition and is handled correctly via the scf::ConditionOp
      // branch later in this function. Indexing parent results by the yield
      // operand number is incorrect for WhileOp: the index can be OOB (init
      // count > result count when scf.condition drops operands), and even
      // when in-bounds it may point to the wrong result because scf.condition
      // can drop/reorder operands relative to the before-arg order.
      if (isa<scf::ForOp, scf::IfOp>(parent))
        valuesToPropagate.push_back(parent->getResult(use.getOperandNumber()));
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        valuesToPropagate.push_back(
            forOp.getRegionIterArg(use.getOperandNumber()));
      if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
        valuesToPropagate.push_back(
            whileOp.getBeforeArguments()[use.getOperandNumber()]);
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
        setEncoding(valuesToPropagate, info, changed, user);
      continue;
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      // Skip arg 0 as it is the condition.
      unsigned argIndex = use.getOperandNumber() - 1;
      Value afterArg = whileOp.getAfterArguments()[argIndex];
      Value result = whileOp->getResult(argIndex);
      setEncoding({afterArg, result}, info, changed, user);
      continue;
    }
    if (auto dotWaitOp = dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(user)) {
      unsigned opIndex = use.getOperandNumber();
      Value result = dotWaitOp->getResult(opIndex);
      setEncoding(result, info, changed, user);
      continue;
    }
    if (auto gatherOp = dyn_cast<GatherOp>(user)) {
      // Propagate the layout through the indices only, and if the layout does
      // not have an efficient layout set.
      if (!gatherOp.getEfficientLayout() &&
          &use == &gatherOp.getIndicesMutable()) {
        setEncoding(gatherOp.getResult(), info, changed, user);
        continue;
      }
    }
    if (user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
        user->hasTrait<OpTrait::Elementwise>() ||
        isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
            ConvertLayoutOp>(user)) {
      setEncoding(user->getResults(), info, changed, user);
      continue;
    }
  }
  return changed;
}

void LayoutPropagation::propagateLayout() {
  SmallVector<Value> queue;
  for (auto it : layouts) {
    queue.push_back(it.first);
  }
  while (!queue.empty()) {
    Value currentValue = queue.back();
    LayoutInfo info = layouts[currentValue];
    queue.pop_back();
    SmallVector<Value> changed = propagateToUsers(currentValue, info);

    LLVM_DEBUG({
      DBGS() << "propagateLayout considering " << currentValue << ", which has "
             << info.encodings.size() << " candidate encoding(s):\n";
      for (Attribute encoding : info.encodings)
        DBGS() << "  " << encoding << "\n";
      DBGS() << "changed: " << changed.size() << "\n";
    });

    queue.insert(queue.end(), changed.begin(), changed.end());
  }
}

// Compute the base shared memory usage from all existing local_alloc ops in the
// function. This accounts for explicit buffers (data tiles, mbarriers) but not
// scratch buffers from convert_layout ops, which are what we're trying to
// eliminate.
static unsigned computeBaseSmem(FuncOp funcOp) {
  unsigned total = 0;
  funcOp->walk([&](LocalAllocOp alloc) {
    if (!alloc.isSharedMemoryAlloc())
      return;
    auto allocType = alloc.getType();
    int64_t numElems;
    if (auto paddedEnc =
            dyn_cast<PaddedSharedEncodingAttr>(allocType.getEncoding())) {
      SmallVector<int64_t> unpaddedShape = getShapePerCTA(allocType);
      numElems = paddedEnc.getPaddedSize(unpaddedShape);
    } else {
      auto shapePerCTA = getAllocationShapePerCTA(allocType);
      numElems = product<int64_t>(shapePerCTA);
    }
    total += numElems * allocType.getElementTypeBitWidth() / 8;
  });
  return total;
}

// Estimate the scratch buffer cost (in bytes) that would result from choosing
// `encoding` for `value`. This checks each operand of value's defining op: if
// an operand is an anchor with a different layout, a convert_layout will be
// needed, and we estimate its scratch size.
static unsigned estimateConvertScratchCost(Value value, Attribute encoding) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return 0;
  auto encTrait = dyn_cast<LayoutEncodingTrait>(encoding);
  unsigned cost = 0;
  for (Value operand : op->getOperands()) {
    auto srcTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!srcTy)
      continue;
    Attribute srcEnc = srcTy.getEncoding();
    if (!srcEnc || srcEnc == encoding)
      continue;
    if (encTrait && srcTy.getRank() != encTrait.getRank())
      continue;
    auto dstTy = srcTy.cloneWithEncoding(encoding);
    if (cvtNeedsSharedMemory(srcTy, dstTy)) {
      unsigned elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy);
      cost += elems * getElementBitWidth(srcTy) / 8;
    }
  }
  return cost;
}

// Compute a score for a layout to guide conflict resolution.
// Based on sizePerThread (vectorization) for both blocked and linear encodings.
// Higher score is preferred — layouts with more elements per thread allow
// better vectorized memory access (ld.shared, st.shared).
static int64_t getLayoutScore(Attribute encoding) {
  SmallVector<unsigned> sizePerThread;
  if (auto blocked = dyn_cast<BlockedEncodingAttr>(encoding)) {
    sizePerThread = SmallVector<unsigned>(blocked.getSizePerThread());
  } else if (auto linear = dyn_cast<LinearEncodingAttr>(encoding)) {
    sizePerThread = linear.getSizePerThread();
  }
  if (sizePerThread.empty())
    return 0;
  int64_t score = 1;
  for (auto size : sizePerThread) {
    score *= size;
  }
  return score;
}

void LayoutPropagation::resolveConflicts() {
  for (auto &it : layouts) {
    Operation *op = it.first.getDefiningOp();
    LayoutInfo &info = it.second;
    if (info.encodings.size() <= 1)
      continue;
    // Hacky resolve, prefer block encoding.
    // TODO: add a proper heuristic.
    Attribute encoding = *info.encodings.begin();
    bool isLoadOrStore =
        op && isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(op);
    // Pick the layout with maximum score.
    // This prefers layouts with larger sizePerThread values for better
    // vectorized memory access. Both blocked and linear encodings are scored,
    // so e.g. a linear layout from TMEMLoadOp (sizePerThread=[1,32]) beats
    // a blocked layout from local_load (sizePerThread=[1,8]).
    int64_t bestScore = getLayoutScore(encoding);
    for (Attribute e : info.encodings) {
      int64_t score = getLayoutScore(e);
      if (score > bestScore) {
        bestScore = score;
        encoding = e;
      }
    }
    // If no layout with vectorization found, fall back to the original
    // heuristic (prefer blocked for load/store, MMA for compute).
    if (bestScore == 0) {
      for (Attribute e : info.encodings) {
        if ((isLoadOrStore && isa<BlockedEncodingAttr>(e)) ||
            (!isLoadOrStore && isa<MmaEncodingTrait>(e))) {
          encoding = e;
          break;
        }
      }
    }
    // Budget-aware override: if the chosen encoding would introduce a
    // convert_layout whose scratch buffer pushes SMEM over budget, pick the
    // candidate with the lowest scratch cost instead.
    if (smemBudget > 0) {
      unsigned baseCost = computeBaseSmem(funcOp);
      unsigned scratchCost = estimateConvertScratchCost(it.first, encoding);
      if (baseCost + scratchCost > smemBudget) {
        LDBG("Budget override: base=" << baseCost << " scratch=" << scratchCost
                                      << " total=" << (baseCost + scratchCost)
                                      << " budget=" << smemBudget);
        // Try each candidate and pick the one with lowest scratch cost.
        Attribute bestEncoding = encoding;
        unsigned bestScratchCost = scratchCost;
        for (Attribute e : info.encodings) {
          unsigned cost = estimateConvertScratchCost(it.first, e);
          if (cost < bestScratchCost) {
            bestScratchCost = cost;
            bestEncoding = e;
          }
        }
        if (bestEncoding != encoding) {
          LDBG("  Overriding to encoding with scratch=" << bestScratchCost);
          encoding = bestEncoding;
        }
      }
    }

    info.encodings.clear();
    info.encodings.insert(encoding);
  }
}

void LayoutPropagation::dump() {
  for (auto it : layouts) {
    llvm::errs() << "Value: ";
    OpPrintingFlags flags;
    flags.skipRegions();
    it.first.print(llvm::errs(), flags);
    llvm::errs() << " \n encoding:\n";
    for (auto encoding : it.second.encodings) {
      encoding.print(llvm::errs());
      llvm::errs() << "\n";
    }
    llvm::errs() << "--\n";
  }
}

void LayoutPropagation::rewrite() { rewriteRegion(funcOp->getRegion(0)); }

bool reduceToScalar(Operation *op) {
  // For reductions returning a scalar we can change the src encoding without
  // affecting the output.
  return isa<ReduceOp>(op) && !isa<RankedTensorType>(op->getResultTypes()[0]);
}

void LayoutPropagation::rewriteRegion(Region &region) {
  std::deque<Region *> queue = {&region};
  while (!queue.empty()) {
    Region *currentRegion = queue.front();
    queue.pop_front();
    for (Operation &op : currentRegion->getOps()) {
      bool needRewrite = false;
      SmallVector<Value> results = op.getResults();
      for (Value result : results) {
        auto it = layouts.find(result);
        // If we haven't mapped this value skip.
        if (it == layouts.end())
          continue;
        LayoutInfo &info = it->second;
        assert(info.encodings.size() == 1 &&
               "we should have resolved to a single encoding");
        auto encoding = cast<RankedTensorType>(result.getType()).getEncoding();
        // If the encoding is already what we want skip.
        if (encoding == *info.encodings.begin())
          continue;
        needRewrite = true;
      }
      if (needRewrite) {
        rewriteOp(&op);
        for (Region &R : op.getRegions())
          queue.push_back(&R);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
        rewriteYieldOp(yieldOp);
      } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(&op)) {
        rewriteConditionOp(conditionOp);
      } else if (reduceToScalar(&op)) {
        rewriteReduceToScalar(&op);
      } else if (auto assertOp = dyn_cast<AssertOp>(&op)) {
        rewriteAssertOp(assertOp);
      } else {
        // If we don't need to rewrite the op we still need to remap the
        // operands.
        for (OpOperand &operand : op.getOpOperands()) {
          auto it = layouts.find(operand.get());
          if (it == layouts.end())
            continue;
          Attribute encoding = getEncodingBeforeRewrite(operand.get());
          Value newOperand = getValueAs(operand.get(), encoding);
          op.setOperand(operand.getOperandNumber(), newOperand);
        }
        for (Region &R : op.getRegions())
          queue.push_back(&R);
      }
    }
  }
}

Value LayoutPropagation::getValueAs(Value value, Attribute encoding) {
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    if (cast<RankedTensorType>(value.getType()).getEncoding() == encoding)
      return value;
    OpBuilder rewriter(value.getContext());
    rewriter.setInsertionPointAfterValue(value);
    auto tmpType = tensorType.cloneWithEncoding(encoding);
    Value converted =
        ConvertLayoutOp::create(rewriter, value.getLoc(), tmpType, value);
    // TODO: we could cache the conversion.
    return converted;
  }
  return value;
}

Attribute LayoutPropagation::getEncodingBeforeRewrite(Value value) const {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return {};
  if (auto it = originalEncodings.find(value); it != originalEncodings.end())
    return it->second;
  return tensorType.getEncoding();
}

void LayoutPropagation::setEncodingInPlace(Value value, Attribute encoding) {
  auto tensorType = cast<RankedTensorType>(value.getType());
  if (!originalEncodings.count(value))
    originalEncodings[value] = tensorType.getEncoding();
  value.setType(tensorType.cloneWithEncoding(encoding));
}

void LayoutPropagation::rewriteGenericOpInPlace(Operation *op,
                                                Attribute encoding) {
  Attribute operandEnc;
  if (op->getNumOperands() > 0) {
    for (Value operand : op->getOperands()) {
      auto it = layouts.find(operand);
      if (it == layouts.end())
        continue;
      Attribute enc = it->second.encodings[0];
      if (inferDstEncoding(op, enc) == encoding) {
        operandEnc = enc;
        break;
      }
    }
    if (!operandEnc)
      operandEnc = inferSrcEncoding(op, encoding);
    assert(operandEnc);
  }
  for (OpOperand &operand : op->getOpOperands()) {
    op->setOperand(operand.getOperandNumber(),
                   getValueAs(operand.get(), operandEnc));
  }
  for (Value result : op->getResults()) {
    auto tensorType = dyn_cast<RankedTensorType>(result.getType());
    if (!tensorType)
      continue;
    setEncodingInPlace(result, encoding);
  }
}

void LayoutPropagation::rewriteForOp(scf::ForOp forOp) {
  for (auto [i, operand, result, regionArg] :
       llvm::enumerate(forOp.getInitArgs(), forOp.getResults(),
                       forOp.getRegionIterArgs())) {
    auto resultTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resultTy)
      continue;
    auto it = layouts.find(result);
    if (it == layouts.end())
      continue;
    Attribute encoding = it->second.encodings[0];
    Value convertedOperand = getValueAs(operand, encoding);
    forOp.getInitArgsMutable()[i].assign(convertedOperand);
    setEncodingInPlace(result, encoding);
    setEncodingInPlace(regionArg, encoding);
  }
}

void LayoutPropagation::rewriteWhileOp(scf::WhileOp whileOp) {
  for (auto [i, operand, beforeArg] :
       llvm::enumerate(whileOp->getOperands(), whileOp.getBeforeArguments())) {
    auto it = layouts.find(beforeArg);
    if (it == layouts.end())
      continue;
    Attribute encoding = it->second.encodings[0];
    Value convertedOperand = getValueAs(operand, encoding);
    whileOp->setOperand(i, convertedOperand);
    setEncodingInPlace(beforeArg, encoding);
  }

  for (auto [result, afterArg] :
       llvm::zip(whileOp.getResults(), whileOp.getAfterArguments())) {
    auto it = layouts.find(result);
    if (it == layouts.end())
      continue;
    Attribute encoding = it->second.encodings[0];
    setEncodingInPlace(result, encoding);
    setEncodingInPlace(afterArg, encoding);
  }
}

void LayoutPropagation::rewriteIfOp(scf::IfOp ifOp) {
  for (unsigned i = 0, e = ifOp->getNumResults(); i < e; ++i) {
    auto it = layouts.find(ifOp.getResult(i));
    if (it == layouts.end())
      continue;
    Attribute encoding = *(it->second.encodings.begin());
    setEncodingInPlace(ifOp.getResult(i), encoding);
  }
}

void LayoutPropagation::rewriteYieldOp(scf::YieldOp yieldOp) {
  Operation *parentOp = yieldOp->getParentOp();
  for (OpOperand &operand : yieldOp->getOpOperands()) {
    Type yieldType = operand.get().getType();
    if (isa<scf::ForOp, scf::IfOp>(parentOp))
      yieldType = parentOp->getResult(operand.getOperandNumber()).getType();
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
      yieldType =
          whileOp.getBeforeArguments()[operand.getOperandNumber()].getType();
    auto tensorType = dyn_cast<RankedTensorType>(yieldType);
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    yieldOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteConditionOp(scf::ConditionOp conditionOp) {
  scf::WhileOp whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
  for (unsigned i = 1; i < conditionOp->getNumOperands(); ++i) {
    OpOperand &operand = conditionOp->getOpOperand(i);
    Type argType = whileOp->getResult(operand.getOperandNumber() - 1).getType();
    auto tensorType = dyn_cast<RankedTensorType>(argType);
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    conditionOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteReduceToScalar(Operation *reduceOp) {
  OpBuilder rewriter(reduceOp);
  Attribute srcEncoding;
  // Since all the operands need to have the same encoding pick the first one
  // and use it for all the operands.
  for (Value operand : reduceOp->getOperands()) {
    auto it = layouts.find(operand);
    if (it != layouts.end()) {
      srcEncoding = it->second.encodings[0];
      break;
    }
  }
  if (!srcEncoding)
    return;
  for (OpOperand &operand : reduceOp->getOpOperands()) {
    Value newOperand = getValueAs(operand.get(), srcEncoding);
    reduceOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteAssertOp(AssertOp assertOp) {
  Attribute srcEncoding;
  // Only need to deal with the first operand which is the condition tensor.
  Value operand = assertOp->getOperand(0);
  auto it = layouts.find(operand);
  if (it == layouts.end())
    return;
  srcEncoding = it->second.encodings[0];
  Value newOperand = getValueAs(operand, srcEncoding);
  assertOp->setOperand(0, newOperand);
}

void LayoutPropagation::rewriteOp(Operation *op) {
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    rewriteForOp(forOp);
  else if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    rewriteWhileOp(whileOp);
  else if (auto ifOp = dyn_cast<scf::IfOp>(op))
    rewriteIfOp(ifOp);
  else {
    Attribute encoding = *layouts[op->getResult(0)].encodings.begin();
    if (canUseResultEncoding(op, encoding)) {
      setEncodingInPlace(op->getResult(0), encoding);
    } else if (op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
               op->hasTrait<OpTrait::Elementwise>() ||
               isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
                   GatherOp, ConvertLayoutOp, nvidia_gpu::WarpGroupDotWaitOp>(
                   op)) {
      rewriteGenericOpInPlace(op, encoding);
    } else {
      llvm::report_fatal_error("unexpected op in rewrite");
    }
  }
}

bool canBeRemat(Operation *op) {
  if (isa<LoadOp, StoreOp>(op))
    return !isExpensiveLoadOrStore(op);
  if (isa<triton::gpu::LocalLoadOp>(op))
    return !isExpensiveLocalLoad(op);
  if (isa<AtomicRMWOp, AtomicCASOp, DotOp>(op))
    return false;
  if (auto gather = dyn_cast<GatherOp>(op))
    return !gather.getEfficientLayout();
  if (auto reshape = dyn_cast<ReshapeOp>(op))
    return !reshape.getEfficientLayout();

  if (isa<scf::WhileOp, scf::ConditionOp>(op))
    return false;

  return true;
}

void LayoutRematerialization::updateRematMapping(
    SmallVector<std::tuple<Value, Value>> &values) {
  for (auto [old, newV] : values) {
    auto it = mappedValues.find(old);
    if (it != mappedValues.end()) {
      Attribute encoding = it->second;
      auto rematIt = rematMapping.find({old, it->second});
      assert(rematIt != rematMapping.end());
      Value replacedValue = rematIt->second;
      rematMapping.erase(rematIt);
      mappedValues.erase(it);
      // Loop through the replacement value to find the new version of remat
      // value. This should be okay as the number of values should be small.
      for (auto [before, after] : values) {
        if (before == replacedValue) {
          replacedValue = after;
          break;
        }
      }
      rematMapping[{newV, encoding}] = replacedValue;
      mappedValues[newV] = encoding;
    }
  }
}

void LayoutRematerialization::rewriteSlice(
    SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
    const DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
    ConvertLayoutOp convertOp, IRMapping &mapping) {
  SetVector<Operation *> opsToRewrite;
  // Keep track of yield operands that need to be duplicated.
  DenseMap<Operation *, SmallVector<int>> yieldOperandsMap;
  // Keep these around to remove them from the slice after our collection pass
  // This ensures we don't duplicate them during an for rewrite or causing the
  // for/yield to fall out of sync
  SetVector<Value> valuesWithExistingRemat;
  for (Value v : slice) {
    auto layoutIt = layout.find(v);
    assert(layoutIt != layout.end());
    // If we found a valid rematerialization for this value while constructing
    // the slice, use that.
    if (Value remat = existingRemats.lookup({v, layoutIt->second})) {
      assert(getRematValue(v, layoutIt->second) == remat && "remat mismatch");
      mapping.map(v, remat);
      valuesWithExistingRemat.insert(v);
      continue;
    }
    if (v.getDefiningOp()) {
      opsToRewrite.insert(v.getDefiningOp());
      if (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
        unsigned operandIdx = cast<OpResult>(v).getResultNumber();
        opsToRewrite.insert(ifOp.thenYield().getOperation());
        yieldOperandsMap[ifOp.thenYield()].push_back(operandIdx);
        opsToRewrite.insert(ifOp.elseYield().getOperation());
        yieldOperandsMap[ifOp.elseYield()].push_back(operandIdx);
      }
    } else {
      BlockArgument blockArg = cast<BlockArgument>(v);
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (auto loopOp = cast<LoopLikeOpInterface>(parentOp)) {
        opsToRewrite.insert(loopOp.getOperation());
        OpOperand *operand = loopOp.getTiedLoopYieldedValue(blockArg);
        auto yieldOp = blockArg.getOwner()->getTerminator();
        yieldOperandsMap[yieldOp].push_back(operand->getOperandNumber());
        opsToRewrite.insert(yieldOp);
      }
    }
  }
  slice.set_subtract(valuesWithExistingRemat);
  opsToRewrite = mlir::topologicalSort(opsToRewrite);

  // replaceAllUsesWith calls delayed until after initial rewrite.
  // This is required for slice.count(value) to work mid rewrite.
  SmallVector<std::tuple<Value, Value>> replacements;

  SmallVector<Operation *> deadOps;
  IRRewriter builder(slice.begin()->getContext());
  for (Operation *op : opsToRewrite) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Keep a mapping of the operands index to the new operands index.
      SmallVector<std::pair<size_t, size_t>> argMapping;
      SmallVector<Value> newOperands;
      for (auto arg : forOp.getRegionIterArgs()) {
        if (slice.count(arg)) {
          OpOperand &initVal = *forOp.getTiedLoopInit(arg);
          argMapping.push_back(std::make_pair(
              forOp.getTiedLoopResult(&initVal).getResultNumber(),
              forOp.getInitArgs().size() + newOperands.size()));
          newOperands.push_back(mapping.lookup(initVal.get()));
        }
      }
      // Create a new for loop with the new operands.
      scf::ForOp newForOp = replaceForOpWithNewSignature(
          builder, forOp, newOperands, replacements);
      deadOps.push_back(forOp.getOperation());
      Block &loopBody = *newForOp.getBody();
      for (auto m : argMapping) {
        mapping.map(forOp.getResult(m.first), newForOp.getResult(m.second));
        int numIndVars = newForOp.getNumInductionVars();
        mapping.map(loopBody.getArgument(m.first + numIndVars),
                    loopBody.getArgument(m.second + numIndVars));
        LLVM_DEBUG({
          DBGS() << "mapping forOp "
                 << loopBody.getArgument(m.first + numIndVars) << " to "
                 << loopBody.getArgument(m.second + numIndVars) << '\n';
        });
        // The result is not in the layout/slice, the argument is.
        Value oldArg = loopBody.getArgument(m.first + numIndVars);
        addRematValue(newForOp.getResult(m.first), layout[oldArg],
                      newForOp.getResult(m.second));
        addRematValue(oldArg, layout[oldArg],
                      loopBody.getArgument(m.second + numIndVars));
      }
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Type> newTypes;
      for (auto res : ifOp.getResults()) {
        if (slice.count(res)) {
          auto it = layout.find(res);
          assert(it != layout.end());

          auto oldType = cast<RankedTensorType>(res.getType());
          auto newType = oldType.cloneWithEncoding(it->second);
          newTypes.push_back(newType);
        }
      }
      scf::IfOp newIfOp =
          replaceIfOpWithNewSignature(builder, ifOp, newTypes, replacements);
      unsigned oldIdx = 0;
      unsigned newIdx = ifOp.getNumResults();
      for (auto res : ifOp.getResults()) {
        if (slice.count(res)) {
          // Why can't we use res instead of ifOp.getResult(oldIdx)?
          mapping.map(ifOp.getResult(oldIdx), newIfOp.getResult(newIdx));
          addRematValue(ifOp.getResult(oldIdx), layout[res],
                        newIfOp.getResult(newIdx));
          ++newIdx;
        }
        ++oldIdx;
      }
      deadOps.push_back(ifOp.getOperation());
      continue;
    }
    builder.setInsertionPoint(op);
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      auto yieldOperands = llvm::to_vector(yieldOp.getOperands());
      SmallVector<int> operandsToRewrite = yieldOperandsMap[op];
      // Sort so that operands are added in the same order as the new scf
      // results/arguments.
      std::sort(operandsToRewrite.begin(), operandsToRewrite.end());
      for (int operandIdx : operandsToRewrite) {
        yieldOperands.push_back(mapping.lookup(yieldOp.getOperand(operandIdx)));
      }
      scf::YieldOp::create(builder, op->getLoc(), yieldOperands);
      op->erase();
      continue;
    }
    if (isa<arith::ConstantOp>(op)) {
      Operation *newOp = builder.clone(*op);
      auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
      auto newType = tensorType.cloneWithEncoding(layout[op->getResult(0)]);
      auto cvt = ConvertLayoutOp::create(builder, op->getLoc(), newType,
                                         newOp->getResult(0));
      mapping.map(op->getResult(0), cvt.getResult());
      addRematValue(op->getResult(0), layout[op->getResult(0)],
                    cvt.getResult());
      continue;
    }
    Operation *newOp = builder.clone(*op, mapping);
    for (auto [old, newV] : llvm::zip(op->getResults(), newOp->getResults())) {
      auto it = layout.find(old);
      if (it == layout.end())
        continue;
      auto newType =
          cast<RankedTensorType>(old.getType()).cloneWithEncoding(it->second);
      newV.setType(newType);
      addRematValue(old, it->second, newV);
    }
  }
  // Check mapping and see if there are existing convertOps on the old Argument
  convertOp.replaceAllUsesWith(mapping.lookup(convertOp.getSrc()));

  updateRematMapping(replacements);
  for (auto &kv : replacements) {
    builder.replaceAllUsesWith(std::get<0>(kv), std::get<1>(kv));
  }

  convertOp->erase();
  for (Operation *op : deadOps)
    op->erase();
}

void LayoutRematerialization::rewriteSlice(
    SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
    const DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
    ConvertLayoutOp convertOp) {
  IRMapping mapping;
  rewriteSlice(slice, layout, existingRemats, convertOp, mapping);
}

LogicalResult LayoutRematerialization::getConvertBackwardSlice(
    OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
    DenseMap<Value, Attribute> &layout,
    DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
    std::function<bool(Operation *)> stopPropagation) {
  // Allow re-using existing conversions for a value if it dominates the use.
  auto getExistingConversion = [&](OpOperand &value, Attribute encoding) {
    Value remat = getRematValue(value.get(), encoding);
    if (!remat)
      return Value();
    // `value` can be replaced with an existing rematerialization if it
    // dominates the current use of value.
    Operation *user = value.getOwner();
    if (domInfo.properlyDominates(remat, user)) {
      existingRemats.try_emplace({value.get(), encoding}, remat);
      return remat;
    }
    // FIXME: If the current user is a conversion, then we know it will become
    // a no-op when its operand is replaced with `remat`, but we need to check
    // that its users are all dominated by `remat` so the IR is valid.
    // if (isa<ConvertLayoutOp>(user) && remat.getDefiningOp() &&
    //     domInfo.properlyDominates(user, remat.getDefiningOp())) {
    //   for (Operation *op : user->getUsers()) {
    //     if (!domInfo.dominates(remat, op))
    //       return Value();
    //   }
    //   return remat;
    // }

    // There is an existing rematerialization, but it doesn't dominate all the
    // uses we care about, so ensure it isn't used.
    existingRemats[{value.get(), encoding}] = Value();
    return Value();
  };

  return mlir::getConvertBackwardSlice(root, slice, rootEncoding, layout,
                                       stopPropagation, getExistingConversion);
}

LogicalResult LayoutRematerialization::getRematerializableSlice(
    OpOperand &root, Attribute rootEncoding, SetVector<Value> &sliceArg,
    DenseMap<Value, Attribute> &layoutArg,
    DenseMap<std::pair<Value, Attribute>, Value> &existingRematsArg,
    std::function<bool(Operation *)> stopPropagation) {
  // Operate on copies of the input, we do not want to modify them unless we
  // have succeeded.
  auto slice = sliceArg;
  auto layout = layoutArg;
  auto existingRemats = existingRematsArg;
  LogicalResult result = getConvertBackwardSlice(
      root, rootEncoding, slice, layout, existingRemats, stopPropagation);
  if (result.failed() || slice.empty())
    return failure();

  // Check if all the operations in the slice can be rematerialized.
  for (Value v : slice) {
    if (Operation *op = v.getDefiningOp()) {
      if (!canBeRemat(op))
        return failure();
    }
  }
  sliceArg = std::move(slice);
  layoutArg = std::move(layout);
  existingRematsArg = std::move(existingRemats);
  return success();
}

bool LayoutRematerialization::backwardRematerialization() {
  bool changed = false;
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    if (!backwardRematerialization(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    } else {
      changed = true;
    }
  }
  return changed;
}

void LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    if (!hoistConvertOnTopOfExtOrBroadcast(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

void LayoutRematerialization::hoistConvertIntoConditionals() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    if (!hoistConvertIntoConditionals(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

static bool isExpensiveMathOp(Operation *op) {
  // These operations are either multiple instructions or have throughput
  // lower than 16 according to the arithmetic instructions table in:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
  return isa<arith::DivFOp, math::ErfcOp, math::SinhOp, math::CoshOp,
             math::TanhOp, math::AsinhOp, math::AcoshOp, math::AtanhOp,
             math::CtPopOp, math::CountLeadingZerosOp,
             math::CountTrailingZerosOp, math::ExpOp, math::Exp2Op,
             math::ExpM1Op, math::LogOp, math::Log2Op, math::Log10Op,
             math::Log1pOp, math::SinOp, math::CosOp, math::TanOp, math::AsinOp,
             math::AcosOp, math::AtanOp, math::Atan2Op, math::PowFOp,
             math::SqrtOp, math::RsqrtOp, math::ErfOp, math::CbrtOp>(op);
}

static int64_t getByteCount(Value result, int64_t minElementCount = 0,
                            int64_t minBitWidth = 0) {
  int64_t elementCount = 0;
  int64_t dtypeBitWidth = 0;
  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    elementCount = tensorTy.getNumElements();
    auto elemType = tensorTy.getElementType();
    if (elemType.isIntOrFloat()) {
      dtypeBitWidth = elemType.getIntOrFloatBitWidth();
    }
  }
  if (elementCount < minElementCount) {
    elementCount = minElementCount;
  }
  if (dtypeBitWidth < minBitWidth) {
    dtypeBitWidth = minBitWidth;
  }
  return (elementCount * dtypeBitWidth) >> 3;
}

/// Compute the cost of a ConvertLayoutOp with source \p convertSrc and result
/// encoding \p resultEncoding.
int64_t getConvertCost(Value convertSrc, Attribute resultEncoding) {
  auto srcType = cast<RankedTensorType>(convertSrc.getType());
  auto resultType = srcType.cloneWithEncoding(resultEncoding);
  if (cvtReordersRegisters(srcType, resultType))
    return 0;

  // Measure the number of bytes that we're manipulating with the
  // ConvertLayoutOp. We pessimistically assume that we round-trip
  // through shared memory and that we cannot vectorise sub-register
  // loads/stores, so we set a minimum element count of 32 (the warp
  // size and number of shared memory banks) and minimum bitwidth of
  // 32 (the width per bank of the shared memory load/store unit).
  auto convertLayoutBytes = getByteCount(convertSrc, 32, 32);
  // We measure costs in standardised milli-SM-cycles. The smem load
  // and store each cost 8 * convertLayoutBytes, and then we double
  // it to account for extra cost due to synchronisation.
  return 32 * convertLayoutBytes;
}

static unsigned getCostFactor(Value result, Attribute rematEncoding) {
  auto tensorType = cast<RankedTensorType>(result.getType());
  unsigned oldElemsPerThread = getUniqueElemsPerThread(tensorType);
  unsigned newElemsPerThread =
      getUniqueElemsPerThread(rematEncoding, tensorType.getShape());
  return std::max(1u, newElemsPerThread / oldElemsPerThread);
}

/// Determine whether rematerializing \p slice is beneficial given that it will
/// eliminate \p convertOp and require creating new convert ops with cost \p
/// newCvtCost.
bool isRematBeneficial(ConvertLayoutOp convertOp, const SetVector<Value> &slice,
                       const DenseMap<Value, Attribute> &layout,
                       int64_t newCvtCost) {
  // Identify all operations in the slice
  SetVector<Operation *> sliceOps;
  for (Value v : slice) {
    if (Operation *op = v.getDefiningOp()) {
      sliceOps.insert(op);
    }
  }

  // Determine which values used by operations outside the slice. We can use
  // this to determine whether they will actually survive and therefore need to
  // contribute to the cost.
  SetVector<Value> nonSliceOnlyValues;

  // Identify values that directly have uses outside the slice.
  for (Value v : slice) {
    for (auto &use : v.getUses()) {
      auto *user = use.getOwner();
      if (user == convertOp || sliceOps.contains(user))
        continue;
      // For region branch ops, check whether the values they flow into are in
      // the slice or unused instead.
      if (isa<RegionBranchTerminatorOpInterface>(user))
        user = user->getParentOp();
      if (auto rbi = dyn_cast<RegionBranchOpInterface>(user)) {
        RegionBranchSuccessorMapping mapping;
        rbi.getSuccessorOperandInputMapping(mapping);
        auto it = mapping.find(&use);
        if (it != mapping.end()) {
          // We have found the values this use flows into, check if they are
          // used outside the slice.
          bool isSliceOnly = llvm::all_of(it->second, [&](Value v) {
            return slice.contains(v) || v.use_empty();
          });
          if (isSliceOnly)
            continue;
        }
      }
      nonSliceOnlyValues.insert(v);
      break;
    }
  }

  // Expand the set to all transitive operands in the slice.
  for (size_t i = 0; i < nonSliceOnlyValues.size(); ++i) {
    Value v = nonSliceOnlyValues[i];
    auto *op = v.getDefiningOp();
    // If the operand is a block argument, get the enclosing op.
    op = op ? op : v.getParentBlock()->getParentOp();
    if (auto rbi = dyn_cast<RegionBranchOpInterface>(op)) {
      // Try to determine the operands that flow into this value, and mark them
      // as being used outside the slice.
      RegionBranchInverseSuccessorMapping mapping;
      rbi.getSuccessorInputOperandMapping(mapping);
      auto it = mapping.find(v);
      if (it != mapping.end()) {
        for (auto tiedOperand : it->second)
          if (slice.contains(tiedOperand->get()))
            nonSliceOnlyValues.insert(tiedOperand->get());
        continue;
      }
    }
    // In the general case, propagate to all operands of the op.
    for (auto operand : op->getOperands())
      if (slice.contains(operand))
        nonSliceOnlyValues.insert(operand);
  }

  int64_t convertLayoutCost =
      getConvertCost(convertOp.getSrc(), convertOp.getType().getEncoding());
  int64_t rematerialisationCost = newCvtCost;

  // Evaluate single-use status for every operation in slice
  for (Operation *op : sliceOps) {
    auto dialect = op->getDialect();
    bool isOpUsedOutsideSlice = llvm::any_of(op->getResults(), [&](Value v) {
      return nonSliceOnlyValues.contains(v);
    });

    if (isa<arith::ConstantOp>(op)) {
      // special-case: arith.constant has zero cost
      continue;
    } else if (isa<arith::ArithDialect, math::MathDialect>(dialect)) {
      // this is an arithmetic operation; we distinguish between cheap
      // operations (such as floating point add/mul which can be fused
      // as halves of a single-cycle FMA instruction) and expensive
      // operations which use the special function unit and/or involve
      // multiple instructions.
      int64_t multiplier = isExpensiveMathOp(op) ? 8 : 1;
      for (Value result : op->getResults()) {
        Attribute rematEncoding = layout.lookup(result);
        int64_t cost = multiplier * getByteCount(result);
        // If the new layout increases the amount of work that needs to happen
        // on each thread, account for that.
        unsigned factor = getCostFactor(result, rematEncoding);
        if (!isOpUsedOutsideSlice)
          factor -= 1;
        rematerialisationCost += cost * factor;
      }
      continue;
    }

    // If all of the results of the op are only used within the slice, when we
    // rematerialise, this operation does not get duplicated so it does not
    // contribute to our cost model.
    if (!isOpUsedOutsideSlice)
      continue;

    if (isa<LoadOp>(op) || isa<LocalLoadOp>(op)) {
      // optimistically assume L1-cached:
      for (Value result : op->getResults()) {
        rematerialisationCost += 8 * getByteCount(result);
      }
    } else if (isa<ReduceOp>(op)) {
      // Reduce op introduce much cost.
      auto reduceOp = dyn_cast<ReduceOp>(op);
      ReduceOpHelper helper(reduceOp);
      if (!helper.isAssociative()) {
        // We shouldn't rematerize a no associative reduce op if it has multiple
        // use chain.
        LDBG("  skipped rematerialization due to non-associative reduce in the "
             "slice");
        return false;
      }
      rematerialisationCost += helper.getIntraWarpSizeWithUniqueData();
      rematerialisationCost += 8 * helper.getInterWarpSizeWithUniqueData();
    }
  }

  LLVM_DEBUG({
    DBGS() << "  convert layout cost: " << convertLayoutCost << "\n";
    DBGS() << "  rematerialisation cost: " << rematerialisationCost << "\n";
  });

  return convertLayoutCost >= rematerialisationCost;
}

bool LayoutRematerialization::backwardRematerialization(
    ConvertLayoutOp convertOp) {
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding())) {
    // DotOperand is hoisted by hoistDotOperand for pipelining purposes.
    if (auto parentForOp = convertOp->getParentOfType<scf::ForOp>()) {
      if (getNumStagesOrDefault(parentForOp, 3) > 1) {
        return false;
      }
    }
  }
  Value oldV = convertOp.getSrc();
  LDBG("check backward remat with source " << oldV << " encoding "
                                           << targetType.getEncoding());
  // Check to see if there are existing remat'ed values for the pair of oldValue
  // and encoding. Make sure it dominates the current conversion.
  Value newV = getRematValue(oldV, targetType.getEncoding());
  if (newV && domInfo.properlyDominates(newV, convertOp)) {
    // Replace it with the remat'ed value.
    convertOp.replaceAllUsesWith(newV);
    convertOp->erase();
    LDBG("found remat'ed value" << newV);
    return true;
  }

  // 1. Take a backward slice of all the tensor dependencies that can be
  // rematerialized.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  DenseMap<std::pair<Value, Attribute>, Value> existingRemats;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout,
      existingRemats);
  if (result.failed()) {
    LDBG("  getRematerializableSlice failed");
    return false;
  }

  // 2. Determine whether rematerialisation is beneficial.
  if (!isRematBeneficial(convertOp, slice, layout, /*newCvtCost=*/0)) {
    LDBG("  skipped rematerialization due to higher cost");
    return false;
  }

  LLVM_DEBUG({
    DBGS() << "  remat convert op " << convertOp << '\n';
    for (Value v : slice)
      DBGS() << "    " << v << '\n';
  });

  // 3. Rewrite the slice.
  rewriteSlice(slice, layout, existingRemats, convertOp);
  return true;
}

void LayoutRematerialization::hoistConvertDotOperand() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    if (!hoistConvertDotOperand(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

bool LayoutRematerialization::hoistConvertDotOperand(
    ConvertLayoutOp convertOp) {
  auto targetType = convertOp.getType();
  // The pass is targeted to MMA dot operands

  auto canBePipelined = [&](ConvertLayoutOp convertOp) {
    // FIXME: Check that the parent is a for loop
    auto parent = convertOp->getParentOp();
    if (!parent)
      return false;

    // Find all the dot-like ops in the for loop that have a dot operand
    // encoding on the lhs and check if any of them post-dominates the load +
    // cvt
    SmallVector<Operation *> dotLikeOps;
    parent->walk([&](Operation *op) {
      if (!isa<mlir::triton::DotOpInterface>(op))
        return;
      auto opType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!opType)
        return;
      auto dotEnc = dyn_cast<DotOperandEncodingAttr>(opType.getEncoding());
      if (!dotEnc)
        return;
      if (isa<MmaEncodingTrait>(dotEnc.getParent()))
        dotLikeOps.push_back(op);
    });
    if (dotLikeOps.empty())
      return false;
    return llvm::any_of(dotLikeOps, [&](Operation *dot) {
      return postDomInfo.postDominates(dot, convertOp);
    });
  };

  // We move convert #dot_operand next to their loads. This is done
  // so that it's then easy to pipeline these loads
  if (!canBePipelined(convertOp))
    return false;

  // We hoist over any operation that can be done without data movement between
  // threads We do views and elementwise pure ops for now
  auto noDataMovement = [](Operation *op) {
    return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
           isa<BroadcastOp, Fp4ToFpOp, ConvertLayoutOp, UpcastFpOpInterface>(
               op) ||
           isView(op);
  };
  // Stop the slice as soon as we find an operation that cannot be done without
  // data movement between threads
  auto stop = std::not_fn(noDataMovement);

  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  DenseMap<std::pair<Value, Attribute>, Value> existingRemats;
  // Set-up the conversion "cache"
  LogicalResult result = getConvertBackwardSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout,
      existingRemats, stop);
  if (result.failed())
    return false;

  IRMapping mapping;
  OpBuilder builder(convertOp.getContext());
  SetVector<Value> innerSlice;
  for (Value v : slice) {
    if (!v.getDefiningOp()) {
      LLVM_DEBUG(
          { DBGS() << "  Block arguments not supported. Got " << v << "\n"; });
      return false;
    }

    // We expect the leaves of the slice to be Load, descriptor load-like ops,
    // or arith::Constant. This could be generalised if necessary.
    if (!isa<LoadOp, DescriptorLoadLikeOpInterface>(v.getDefiningOp())) {
      auto op = v.getDefiningOp();
      if (isa<arith::ConstantOp>(op) || noDataMovement(op)) {
        innerSlice.insert(v);
        continue;
      } else {
        LLVM_DEBUG({
          DBGS() << "  Leaves must be Load, descriptor load-like ops, or "
                    "Constant. Got "
                 << v << "\n";
        });
        return false;
      }
    }
    Operation *loadOp = v.getDefiningOp();
    builder.setInsertionPointAfter(loadOp);
    auto type = dyn_cast<RankedTensorType>(loadOp->getResult(0).getType());
    if (!type)
      continue;
    auto newType = type.cloneWithEncoding(layout[loadOp->getResult(0)]);
    auto newConvertOp = ConvertLayoutOp::create(builder, convertOp.getLoc(),
                                                newType, loadOp->getResult(0));
    mapping.map(loadOp->getResult(0), newConvertOp.getResult());
  }

  if (innerSlice.empty()) {
    return false;
  }

  LLVM_DEBUG({
    DBGS() << "  Hoisting " << convertOp << '\n';
    for (Value v : innerSlice)
      DBGS() << "    " << v << '\n';
  });

  rewriteSlice(innerSlice, layout, existingRemats, convertOp, mapping);
  return true;
}

// For convert left we try to hoist them above type extension to reduce the cost
// of the convert.
bool LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast(
    ConvertLayoutOp convertOp) {
  // DotOperand is hoisted by hoistDotOperand
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding()))
    return false;

  auto isExtOrBroadcastOp = [](Operation *op) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, BroadcastOp,
            ExpandDimsOp>(op)) {
      return true;
    }
    if (auto fpToFpOp = dyn_cast<FpToFpOp>(op)) {
      auto srcType = cast<RankedTensorType>(fpToFpOp.getSrc().getType());
      return getElementBitWidth(srcType) <
             getElementBitWidth(cast<RankedTensorType>(fpToFpOp.getType()));
    }
    return false;
  };
  // 1. Take a backward slice of all the tensor dependencies.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  DenseMap<std::pair<Value, Attribute>, Value> existingRemats;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout,
      existingRemats, isExtOrBroadcastOp);
  if (result.failed())
    return false;

  Operation *extOrBroadcastOp = nullptr;
  unsigned sliceSize = slice.size();
  for (unsigned i = 0; i < sliceSize; i++) {
    Value v = slice[i];
    Operation *op = v.getDefiningOp();
    if (!op || !isExtOrBroadcastOp(op))
      continue;

    Attribute srcEncoding = inferSrcEncoding(op, layout[v]);
    if (!srcEncoding)
      return false;

    // If we can rematerialize the rest of the ext slice we can ignore this ext
    // as it won't need a convert.
    if (succeeded(getRematerializableSlice(op->getOpOperand(0), srcEncoding,
                                           slice, layout, existingRemats)))
      continue;

    // Only apply it if there is a single ext op otherwise we would have to
    // duplicate the convert.
    if (extOrBroadcastOp != nullptr)
      return false;
    extOrBroadcastOp = op;
  }

  if (extOrBroadcastOp == nullptr)
    return false;
  Attribute dstEncoding = layout[extOrBroadcastOp->getResult(0)];
  Attribute srcEncoding = inferSrcEncoding(extOrBroadcastOp, dstEncoding);
  if (!srcEncoding)
    return false;
  // Under a SMEM budget (a Meta extension) eliminating convert-layout scratch
  // takes priority over the upstream perf-cost heuristic (#9194), so force the
  // hoist. Without a budget, keep the upstream cost gate.
  if (smemBudget == 0) {
    int64_t newCvtCost =
        getConvertCost(extOrBroadcastOp->getOperand(0), srcEncoding);
    if (!isRematBeneficial(convertOp, slice, layout, newCvtCost))
      return false;
  }
  // Move the convert before the ext op and rewrite the slice.
  OpBuilder builder(extOrBroadcastOp);
  auto tensorType =
      cast<RankedTensorType>(extOrBroadcastOp->getOperand(0).getType());
  auto newType = tensorType.cloneWithEncoding(srcEncoding);
  auto newConvertOp = ConvertLayoutOp::create(
      builder, convertOp.getLoc(), newType, extOrBroadcastOp->getOperand(0));
  Operation *newExtOrBroadcast = builder.clone(*extOrBroadcastOp);
  newExtOrBroadcast->setOperand(0, newConvertOp.getResult());
  auto oldExtOrBroadcastType =
      cast<RankedTensorType>(extOrBroadcastOp->getResult(0).getType());
  Type newExtOrBroadcastType =
      oldExtOrBroadcastType.cloneWithEncoding(dstEncoding);
  newExtOrBroadcast->getResult(0).setType(newExtOrBroadcastType);
  IRMapping mapping;
  mapping.map(extOrBroadcastOp->getResult(0), newExtOrBroadcast->getResult(0));
  slice.remove(extOrBroadcastOp->getResult(0));
  // 3. Rewrite the slice.
  rewriteSlice(slice, layout, existingRemats, convertOp, mapping);
  return true;
}

bool LayoutRematerialization::hoistConvertIntoConditionals(
    ConvertLayoutOp convertOp) {
  // Take the backward slice of tensor dependencies rooted at the conversion,
  // stopping at conditionals. This subslice is used to initialize the analysis.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  DenseMap<std::pair<Value, Attribute>, Value> existingRemats;
  auto isIfOp = [](Operation *op) { return isa<scf::IfOp>(op); };
  if (failed(getRematerializableSlice(convertOp.getSrcMutable(),
                                      convertOp.getType().getEncoding(), slice,
                                      layout, existingRemats, isIfOp)))
    return false;

  // These are the conditional edges above which conversions should be hoisted.
  // The value represents the `scf.if` op result and the operand represents the
  // edge into one of the branches.
  SmallVector<std::pair<Value, OpOperand *>> hoistAbove;

  // The list of `scf.if` op results in the slice that are not rematerializable.
  // Hoisting is terminated at these values.
  SmallVector<OpResult> terminals;

  // This loop recurses through the subslices of the backwards dependencies, so
  // re-query the size of `slice`.
  for (unsigned i = 0; i != slice.size(); ++i) {
    Value v = slice[i];
    auto ifOp = v.getDefiningOp<scf::IfOp>();
    if (!ifOp)
      continue;

    Attribute rootLayout = layout.at(v);
    unsigned resIdx = cast<OpResult>(v).getResultNumber();

    // Take the backward slice along each branch.
    auto thenYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
    auto elseYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

    OpOperand &thenRes = thenYield.getResultsMutable()[resIdx];
    OpOperand &elseRes = elseYield.getResultsMutable()[resIdx];

    auto newSlice = slice;
    auto newLayout = layout;
    auto newExistingRemats = existingRemats;

    LogicalResult thenResult = getRematerializableSlice(
        thenRes, rootLayout, newSlice, newLayout, newExistingRemats, isIfOp);
    LogicalResult elseResult = getRematerializableSlice(
        elseRes, rootLayout, newSlice, newLayout, newExistingRemats, isIfOp);

    // If propagation across both edges of this conditional succeeded, then we
    // don't need to hoist across it. Merge into the current slice.
    if (succeeded(thenResult) && succeeded(elseResult)) {
      slice = std::move(newSlice);
      layout = std::move(newLayout);
      existingRemats = std::move(newExistingRemats);
      continue;
    }

    // If propagation across both edges failed, then this conditional
    // terminates backwards rematerialization.
    if (failed(thenResult) && failed(elseResult)) {
      terminals.push_back(cast<OpResult>(v));
      continue;
    }

    // Only hoist into conditionals inside loops. The assumption is that an if
    // inside a loop executes fewer than the total number of loop iterations,
    // making this hoist profitable.
    if (!isa<scf::ForOp>(ifOp->getParentOp())) {
      terminals.push_back(cast<OpResult>(v));
      continue;
    }

    slice = std::move(newSlice);
    layout = std::move(newLayout);
    existingRemats = std::move(newExistingRemats);
    // The layout conversion can be rematerialized along one edge but not the
    // other. We can hoist the conversion into the other branch. Push this
    // into the subslice list for analysis.
    if (succeeded(thenResult)) {
      hoistAbove.emplace_back(v, &elseRes);
    } else {
      hoistAbove.emplace_back(v, &thenRes);
    }
  }

  // Exit early if there is nothing to do.
  if (hoistAbove.empty())
    return false;

  // Rematerialize failed hoists right before the condtional, and hoist those
  // that succeeded into the branch and then rewrite the slice.
  IRMapping mapping;
  auto hoistRemat = [&](OpBuilder &b, Value v, Attribute encoding) {
    auto tensorType = cast<RankedTensorType>(v.getType());
    auto newType = tensorType.cloneWithEncoding(encoding);
    Value newCvt = ConvertLayoutOp::create(b, convertOp.getLoc(), newType, v);

    mapping.map(v, newCvt);
    slice.remove(v);
  };
  for (Value v : terminals) {
    OpBuilder b(v.getContext());
    b.setInsertionPointAfter(v.getDefiningOp());
    hoistRemat(b, v, layout.at(v));
  }
  for (auto [result, edge] : hoistAbove) {
    OpBuilder b(edge->getOwner());
    hoistRemat(b, edge->get(), layout.at(result));
  }
  rewriteSlice(slice, layout, existingRemats, convertOp, mapping);
  return true;
}

bool backwardRematerialization(ModuleOp module) {
  bool changed = false;
  module.walk([&](FuncOp funcOp) {
    LayoutRematerialization layoutRemat(funcOp);
    changed |= layoutRemat.backwardRematerialization();
  });
  return changed;
}

void hoistConvert(ModuleOp module, unsigned smemBudget = 0) {
  SmallVector<ConvertLayoutOp> convertOps;
  module.walk([smemBudget](FuncOp funcOp) {
    LayoutRematerialization layoutRemat(funcOp, smemBudget);
    layoutRemat.hoistConvertOnTopOfExtOrBroadcast();

    layoutRemat = LayoutRematerialization(funcOp);
    layoutRemat.hoistConvertIntoConditionals();

    layoutRemat = LayoutRematerialization(funcOp);
    layoutRemat.hoistConvertDotOperand();
  });
}

// Prune a store-only layout conversion through a non-reordering reshape:
//
//   local_store(reshape(convert_layout(x)), dst)
//
// becomes:
//
//   local_store(reshape(x), dst)
//
// This is intentionally narrow and runs after layout decisions have already
// been made. The rewrite relies on local_store being a layout-flexible sink:
// the source register encoding may change, but the destination memdesc still
// defines the logical memory layout.
struct PruneLocalStoreOfReshapeConvert : public OpRewritePattern<LocalStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalStoreOp store,
                                PatternRewriter &rewriter) const override {
    auto reshape = store.getSrc().getDefiningOp<ReshapeOp>();
    if (!reshape || reshape.getAllowReorder())
      return failure();
    if (!reshape.getResult().hasOneUse())
      return failure();

    auto convert = reshape.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert || !convert.getResult().hasOneUse())
      return failure();

    auto srcType = dyn_cast<RankedTensorType>(convert.getSrc().getType());
    auto storeSrcType = dyn_cast<RankedTensorType>(store.getSrc().getType());
    if (!srcType || !storeSrcType || !srcType.getEncoding())
      return failure();
    if (srcType.getElementType() != storeSrcType.getElementType())
      return failure();
    if (srcType.getNumElements() != storeSrcType.getNumElements())
      return failure();

    Attribute reshapeEncoding;
    auto *layoutInterface = dyn_cast<DialectInferLayoutInterface>(
        &srcType.getEncoding().getDialect());
    if (!layoutInterface)
      return failure();
    if (failed(layoutInterface->inferReshapeOpEncoding(
            srcType.getShape(), srcType.getEncoding(), storeSrcType.getShape(),
            reshapeEncoding, /*allowReorder=*/false, store.getLoc())))
      return failure();

    auto reshapeType =
        RankedTensorType::get(storeSrcType.getShape(),
                              storeSrcType.getElementType(), reshapeEncoding);
    rewriter.setInsertionPoint(store);
    auto newReshape = ReshapeOp::create(
        rewriter, reshape.getLoc(), reshapeType, convert.getSrc(),
        reshape.getAllowReorder(), reshape.getEfficientLayout());
    auto newStore = LocalStoreOp::create(
        rewriter, store.getLoc(), newReshape.getResult(), store.getDst());
    newStore->setAttrs(store->getAttrs());
    rewriter.eraseOp(store);
    rewriter.eraseOp(reshape);
    rewriter.eraseOp(convert);
    return success();
  }
};
} // namespace

class TritonGPURemoveLayoutConversionsPass
    : public impl::TritonGPURemoveLayoutConversionsBase<
          TritonGPURemoveLayoutConversionsPass> {
public:
  using Base = impl::TritonGPURemoveLayoutConversionsBase<
      TritonGPURemoveLayoutConversionsPass>;
  using Base::Base;
  // Cleanup convert ops.
  void cleanupConvertOps() {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    RewritePatternSet cleanUpPatterns(context);
    ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns, context);
    if (applyPatternsGreedily(m, std::move(cleanUpPatterns)).failed()) {
      signalPassFailure();
    }

    LLVM_DEBUG({
      DBGS() << "Module after canonicalizing:\n";
      m.dump();
    });
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // 1. Propagate layout forward starting from "anchor" ops.
    m.walk([this](FuncOp funcOp) {
      LayoutPropagation layoutPropagation(funcOp, smemBudget);
      layoutPropagation.initAnchorLayout();
      layoutPropagation.propagateLayout();
      layoutPropagation.resolveConflicts();
      layoutPropagation.rewrite();
    });

    LLVM_DEBUG({
      DBGS() << "Module after propagating layouts forward:\n";
      m.dump();
    });

    cleanupConvertOps();

    bool changed = false;
    do {
      changed = false;
      // 2. For remaining convert ops, try to rematerialize the slice of
      // producer operation to avoid having to convert.
      changed = backwardRematerialization(m);
      LLVM_DEBUG({
        DBGS() << "Module after backward remat:\n";
        m.dump();
      });

      // Cleanup dummy converts created during backward remat.
      cleanupConvertOps();
    } while (changed);
    // 3. For remaining converts, try to hoist them above cast generating larger
    // size types in order to reduce the cost of the convert op.
    hoistConvert(m, smemBudget);
    LLVM_DEBUG({
      DBGS() << "Module after hoisting converts:\n";
      m.dump();
    });

    // 4. Prepare dead iter args to be cleaned up by dead code elimination in
    // the pattern rewriter below.
    runDeadIterArgElimination(m);

    // 5. Apply clean up patterns to remove dead convert and dead code generated
    // by the previous transformations.
    RewritePatternSet convertCleanup(context);
    ConvertLayoutOp::getCanonicalizationPatterns(convertCleanup, context);
    convertCleanup.add<PruneLocalStoreOfReshapeConvert>(context);
    if (applyPatternsGreedily(m, std::move(convertCleanup)).failed()) {
      signalPassFailure();
    }

    RewritePatternSet scfCleanup(context);
    scf::ForOp::getCanonicalizationPatterns(scfCleanup, context);
    scf::IfOp::getCanonicalizationPatterns(scfCleanup, context);
    if (applyPatternsGreedily(m, std::move(scfCleanup)).failed()) {
      LLVM_DEBUG(DBGS() << "scf cleanup did not converge\n");
    }

    LLVM_DEBUG({
      DBGS() << "Module after final cleanups:\n";
      m.dump();
    });

    // 5. Budget-aware convert elimination. If smemBudget is set, find remaining
    // convert_layout ops whose scratch would push SMEM over budget, and try to
    // eliminate them by propagating the source encoding through their users.
    if (smemBudget > 0) {
      eliminateOverBudgetConverts(m);
      hoistConvert(m, smemBudget);
      cleanupConvertOps();
    }
  }

  // Find convert_layout ops that need SMEM scratch and would push total SMEM
  // over budget. For each such convert, if the source is an anchor (like
  // tmem_load) and the users are elementwise ops feeding into local_store/
  // local_load (which can accept any layout), propagate the source layout
  // through the convert's users and erase the convert.
  void eliminateOverBudgetConverts(ModuleOp m) {
    m.walk([this](FuncOp funcOp) {
      unsigned baseSmem = computeBaseSmem(funcOp);

      // Collect converts whose scratch would push SMEM over budget.
      SmallVector<ConvertLayoutOp> overBudgetConverts;
      funcOp->walk([&](ConvertLayoutOp cvt) {
        auto srcTy = cvt.getSrc().getType();
        auto dstTy = cvt.getType();
        if (!cvtNeedsSharedMemory(srcTy, dstTy))
          return;
        unsigned scratchBytes = getNumScratchElemsSwizzledCvt(srcTy, dstTy) *
                                getElementBitWidth(srcTy) / 8;
        if (baseSmem + scratchBytes > smemBudget) {
          overBudgetConverts.push_back(cvt);
        }
      });

      for (ConvertLayoutOp cvt : overBudgetConverts) {
        Attribute srcEnc = cvt.getSrc().getType().getEncoding();
        if (canPropagateSrcEncodingThroughUsers(cvt, srcEnc)) {
          propagateSrcEncodingAndErase(cvt, srcEnc);
        }
      }
    });
  }

  // Check whether we can propagate srcEnc through all transitive users of the
  // convert result until we hit local_store or local_load (which accept any
  // layout) or the value dies. Returns false if any user requires a specific
  // layout that doesn't match srcEnc.
  bool canPropagateSrcEncodingThroughUsers(ConvertLayoutOp cvt,
                                           Attribute srcEnc) {
    unsigned srcEncRank = 0;
    if (auto encTrait = dyn_cast<LayoutEncodingTrait>(srcEnc))
      srcEncRank = encTrait.getRank();

    SmallVector<Value> worklist;
    worklist.push_back(cvt.getResult());
    DenseSet<Value> visited;

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!visited.insert(v).second)
        continue;

      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        // local_store accepts any register layout — it's a sink.
        if (isa<LocalStoreOp>(user))
          continue;
        // Elementwise ops are layout-transparent — propagate through them.
        if (user->hasTrait<OpTrait::Elementwise>() ||
            isa<arith::ExtFOp, arith::TruncFOp, arith::ExtUIOp, arith::ExtSIOp,
                arith::TruncIOp, arith::SIToFPOp, arith::FPToSIOp,
                arith::BitcastOp>(user)) {
          for (Value result : user->getResults()) {
            auto rtt = dyn_cast<RankedTensorType>(result.getType());
            if (!rtt)
              continue;
            if (srcEncRank > 0 && rtt.getRank() != srcEncRank)
              return false;
            worklist.push_back(result);
          }
          continue;
        }
        // scf.yield passes values through to the parent op's results.
        // For ForOp/WhileOp, the parent results are tied to block arguments
        // and init operands via loop-carried dependencies — in-place type
        // rewriting cannot safely update all of them, so block propagation.
        // For IfOp, the results are simple branches with no loop-carried
        // deps, so propagation is safe if we also follow the IfOp results.
        if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
          Operation *parent = yieldOp->getParentOp();
          if (isa<scf::ForOp, scf::WhileOp>(parent))
            return false;
          if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
            for (Value result : ifOp.getResults()) {
              if (isa<RankedTensorType>(result.getType()))
                worklist.push_back(result);
            }
            continue;
          }
          return false;
        }
        // Any other user (dot, reduce, another convert, etc.) blocks
        // propagation.
        return false;
      }
    }
    return true;
  }

  // Propagate the source encoding through all users of the convert result,
  // rewriting types in place, then erase the convert. For elementwise ops
  // whose other operands have a different encoding, change their local_load
  // to produce the new encoding directly (local_load can produce any layout).
  // If a non-local_load operand has a mismatched encoding, insert a
  // convert_layout on it.
  void propagateSrcEncodingAndErase(ConvertLayoutOp cvt, Attribute srcEnc) {
    IRRewriter rewriter(cvt.getContext());
    Value src = cvt.getSrc();
    Value dst = cvt.getResult();

    // Collect all ops that need type rewriting (forward from convert users).
    SmallVector<Operation *> opsToRewrite;
    SetVector<Operation *> ifOpsToRewrite;
    SmallVector<Value> worklist = {dst};
    DenseSet<Value> visited;

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!visited.insert(v).second)
        continue;
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        if (isa<LocalStoreOp>(user))
          continue;
        // For scf.yield under scf.if, follow through to the IfOp results.
        // ForOp/WhileOp yields are blocked by
        // canPropagateSrcEncodingThroughUsers.
        if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
          if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
            ifOpsToRewrite.insert(ifOp.getOperation());
            for (Value result : ifOp.getResults()) {
              if (isa<RankedTensorType>(result.getType()))
                worklist.push_back(result);
            }
          }
          continue;
        }
        opsToRewrite.push_back(user);
        for (Value result : user->getResults()) {
          if (isa<RankedTensorType>(result.getType()))
            worklist.push_back(result);
        }
      }
    }

    // Build a set of ops being rewritten for fast lookup.
    DenseSet<Operation *> rewriteSet(opsToRewrite.begin(), opsToRewrite.end());

    // For each op we're rewriting, fix up any operands that aren't in srcEnc.
    // When an operand comes through a chain of elementwise ops from a
    // local_load, rewrite the entire chain to srcEnc.
    for (Operation *op : opsToRewrite) {
      for (OpOperand &operand : op->getOpOperands()) {
        auto ty = dyn_cast<RankedTensorType>(operand.get().getType());
        if (!ty || ty.getEncoding() == srcEnc)
          continue;
        // Walk backward through elementwise ops to find a local_load.
        // Rewrite each op's result type along the way.
        SmallVector<Operation *> backwardChain;
        Value current = operand.get();
        bool foundLocalLoad = false;
        while (auto defOp = current.getDefiningOp()) {
          if (auto localLoad = dyn_cast<LocalLoadOp>(defOp)) {
            foundLocalLoad = true;
            break;
          }
          if (defOp->hasTrait<OpTrait::Elementwise>() ||
              isa<arith::ExtFOp, arith::TruncFOp, arith::ExtUIOp,
                  arith::ExtSIOp, arith::TruncIOp>(defOp)) {
            backwardChain.push_back(defOp);
            // Elementwise ops have one primary input.
            current = defOp->getOperand(0);
            continue;
          }
          break;
        }
        if (foundLocalLoad) {
          // Before mutating types in place, verify that every value in the
          // backward chain (and the local_load) is only used by ops that
          // are also being rewritten or are in the backward chain itself.
          // If a value has users outside this set, in-place mutation would
          // create an encoding mismatch for those external users.
          bool safeToMutate = true;
          auto isSafeUser = [&](Operation *user) {
            return rewriteSet.contains(user) ||
                   llvm::is_contained(backwardChain, user);
          };
          // Check the local_load's result.
          for (Operation *user : current.getUsers()) {
            if (!isSafeUser(user)) {
              safeToMutate = false;
              break;
            }
          }
          // Check each op in the backward chain.
          if (safeToMutate) {
            for (Operation *chainOp : backwardChain) {
              for (Value result : chainOp->getResults()) {
                for (Operation *user : result.getUsers()) {
                  if (!isSafeUser(user)) {
                    safeToMutate = false;
                    break;
                  }
                }
                if (!safeToMutate)
                  break;
              }
              if (!safeToMutate)
                break;
            }
          }
          if (safeToMutate) {
            // Safe: mutate the local_load and chain in place.
            current.setType(cast<RankedTensorType>(current.getType())
                                .cloneWithEncoding(srcEnc));
            for (Operation *chainOp : backwardChain) {
              for (Value result : chainOp->getResults()) {
                if (auto rty = dyn_cast<RankedTensorType>(result.getType()))
                  result.setType(rty.cloneWithEncoding(srcEnc));
              }
            }
          } else {
            // Unsafe: fall back to inserting a convert on this operand.
            rewriter.setInsertionPoint(op);
            auto newTy = ty.cloneWithEncoding(srcEnc);
            auto newCvt = ConvertLayoutOp::create(rewriter, op->getLoc(), newTy,
                                                  operand.get());
            operand.set(newCvt);
          }
        } else {
          // Fallback: insert a convert_layout on this operand.
          rewriter.setInsertionPoint(op);
          auto newTy = ty.cloneWithEncoding(srcEnc);
          auto newCvt = ConvertLayoutOp::create(rewriter, op->getLoc(), newTy,
                                                operand.get());
          operand.set(newCvt);
        }
      }
    }

    // Rewrite result types to use srcEnc.
    unsigned srcEncRank = 0;
    if (auto encTrait = dyn_cast<LayoutEncodingTrait>(srcEnc))
      srcEncRank = encTrait.getRank();
    for (Operation *op : opsToRewrite) {
      for (Value result : op->getResults()) {
        if (auto ty = dyn_cast<RankedTensorType>(result.getType())) {
          if (srcEncRank > 0 && ty.getRank() != srcEncRank)
            continue;
          result.setType(ty.cloneWithEncoding(srcEnc));
        }
      }
    }
    // Rewrite IfOp result types that we propagated through.
    for (Operation *op : ifOpsToRewrite) {
      for (Value result : op->getResults()) {
        if (auto ty = dyn_cast<RankedTensorType>(result.getType())) {
          result.setType(ty.cloneWithEncoding(srcEnc));
        }
      }
    }

    // Replace all uses of the convert result with the convert source.
    dst.replaceAllUsesWith(src);
    cvt.erase();

    LLVM_DEBUG({
      DBGS() << "Eliminated over-budget convert_layout, propagated " << srcEnc
             << " through " << opsToRewrite.size() << " ops\n";
    });
  }
};

} // namespace mlir::triton::gpu
