#include "IR/Dialect.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tlx/dialect/include/Analysis/LayoutPropagation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/DataFlowFramework.h"
#define DEBUG_TYPE "tlx-propagate-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXPROPAGATELAYOUT
#include "tlx/dialect/include/Transforms/Passes.h.inc"

class RequireLayoutPattern : public mlir::OpRewritePattern<RequireLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(RequireLayoutOp requireLayoutOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(requireLayoutOp.getSrc().getType()))
      return failure();
    auto resultType = cast<RankedTensorType>(requireLayoutOp.getType());
    if (containsPinnedEncoding(resultType.getEncoding())) {
      auto boundary = ttg::RequireLayoutOp::create(
          rewriter, requireLayoutOp.getLoc(), requireLayoutOp.getType(),
          requireLayoutOp.getSrc());
      if (requireLayoutOp->hasAttr("tlx.rematerialize_coordinates"))
        boundary->setAttr("tlx.rematerialize_coordinates",
                          rewriter.getUnitAttr());
      rewriter.replaceOp(requireLayoutOp, boundary);
      return success();
    }
    bool rematerializeCoordinates =
        requireLayoutOp->hasAttr("tlx.rematerialize_coordinates");
    if (requireLayoutOp.getSrc().getType() == requireLayoutOp.getType() &&
        !rematerializeCoordinates) {
      rewriter.replaceOp(requireLayoutOp, requireLayoutOp.getSrc());
      return success();
    }

    auto convert = ttg::ConvertLayoutOp::create(
        rewriter, requireLayoutOp.getLoc(), requireLayoutOp.getType(),
        requireLayoutOp.getSrc());
    if (rematerializeCoordinates)
      convert->setAttr("tlx.rematerialize_coordinates", rewriter.getUnitAttr());
    rewriter.replaceOp(requireLayoutOp, convert);
    return success();
  }
};

class ReleaseLayoutPattern : public mlir::OpRewritePattern<ReleaseLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReleaseLayoutOp releaseLayoutOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (releaseLayoutOp.getSrc().getType() == releaseLayoutOp.getType()) {
      rewriter.replaceOp(releaseLayoutOp, releaseLayoutOp.getSrc());
      return success();
    }
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
        releaseLayoutOp, releaseLayoutOp.getType(), releaseLayoutOp.getSrc());
    return success();
  }
};

// Late AMD passes can express a tensor layout conversion as a spill through
// immutable local memory:
//   tensor -> ttg.local_alloc -> ttg.local_load(dot)
// Fold it back to either an identity or an explicit convert_layout so the
// fallback does not survive to LLVM as LDS traffic.
// A local_alloc whose encoding is the user-pinned wrapper is a hard constraint:
// it must not be retagged or folded away (the user explicitly asked for this
// buffer with this layout). These checks run while the wrapper is still present
// -- the unwrap to the concrete layout happens after the greedy patterns.
static bool isUserPinnedAlloc(ttg::LocalAllocOp allocOp) {
  return isa_and_nonnull<UserLayoutAttr>(allocOp.getType().getEncoding());
}

class FoldRetaggedLocalAllocLoad
    : public mlir::OpRewritePattern<ttg::LocalLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ttg::LocalLoadOp localLoadOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto allocOp = localLoadOp.getSrc().getDefiningOp<ttg::LocalAllocOp>();
    if (!allocOp || !allocOp.getSrc())
      return failure();
    if (isUserPinnedAlloc(allocOp))
      return failure();
    if (localLoadOp.getToken())
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(localLoadOp.getType());
    if (!resultType ||
        !isSupportedDotConstraintEncoding(resultType.getEncoding()))
      return failure();

    if (allocOp.getSrc().getType() == localLoadOp.getType()) {
      rewriter.replaceOp(localLoadOp, allocOp.getSrc());
      return success();
    }

    // The matched local_alloc -> local_load pair is always an LDS round-trip.
    // Replace it with a layout conversion: it lowers to register shuffles for
    // the common encoding pairs and is no worse than the spill in the few
    // cases where the conversion itself still goes through LDS.
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
        localLoadOp, localLoadOp.getType(), allocOp.getSrc());
    return success();
  }
};

class FoldLocalAllocLoadFallback
    : public mlir::OpRewritePattern<ttg::LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ttg::LocalAllocOp allocOp,
                  mlir::PatternRewriter &rewriter) const override {
    Value src = allocOp.getSrc();
    if (!src || !isa<RankedTensorType>(src.getType()))
      return failure();
    if (isUserPinnedAlloc(allocOp))
      return failure();
    SmallVector<ttg::LocalLoadOp> loads;
    for (Operation *user : allocOp->getUsers()) {
      auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(user);
      if (!localLoadOp || localLoadOp.getToken())
        return failure();
      auto resultType = dyn_cast<RankedTensorType>(localLoadOp.getType());
      if (!resultType ||
          !isSupportedDotConstraintEncoding(resultType.getEncoding()))
        return failure();
      loads.push_back(localLoadOp);
    }
    if (loads.empty())
      return failure();

    for (ttg::LocalLoadOp localLoadOp : loads) {
      rewriter.setInsertionPoint(localLoadOp);
      Value replacement = src;
      if (src.getType() != localLoadOp.getType())
        replacement = ttg::ConvertLayoutOp::create(
            rewriter, localLoadOp.getLoc(), localLoadOp.getType(), src);
      rewriter.replaceOp(localLoadOp, replacement);
    }
    if (allocOp->use_empty())
      rewriter.eraseOp(allocOp);
    return success();
  }
};

// `ttg.warp_predicate` restricts EXEC before entering its region. A layout
// conversion that redistributes values across waves therefore cannot remain in
// the region: a skipped wave would not participate in the shuffle. Layout
// inference commonly introduces exactly that shape around an MFMA body:
//
//   old init -> warp_predicate { ... MFMA value -> old layout -> yield }
//
// Move captured conversions before the EXEC restriction and move conversions
// on yielded values across the region boundary. The latter changes the carried
// type to the body's native layout and converts back after all waves have
// reconverged. Since convert_layout preserves the logical tensor, the
// old->new->old round trip also preserves every inactive lane's init value.
class HoistWarpPredicateLayoutConversions
    : public mlir::OpRewritePattern<ttg::WarpPredicateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ttg::WarpPredicateOp predicateOp,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;

    auto yieldOp = dyn_cast<ttg::PredicateYieldOp>(
        predicateOp.getRegion().front().getTerminator());
    if (!yieldOp)
      return failure();

    auto predicateType =
        dyn_cast<RankedTensorType>(predicateOp.getPredicate().getType());
    std::optional<Attribute> predicateEncoding;
    bool conflictingPredicateEncoding = false;
    // Determine the physical row ownership before moving or retyping anything.
    // One execution predicate cannot safely control carried row values whose
    // native layouts assign those rows to different lanes.
    for (auto [result, init, yielded] :
         llvm::zip(predicateOp.getResults(), predicateOp.getInits(),
                   yieldOp.getValues())) {
      auto oldType = dyn_cast<RankedTensorType>(result.getType());
      auto boundaryConvert = yielded.getDefiningOp<ttg::ConvertLayoutOp>();
      auto bodyType =
          boundaryConvert
              ? dyn_cast<RankedTensorType>(boundaryConvert.getSrc().getType())
              : RankedTensorType();
      bool hasHoistableBoundary =
          boundaryConvert && boundaryConvert->hasOneUse() && oldType &&
          bodyType && oldType != bodyType &&
          oldType.getShape() == bodyType.getShape() &&
          oldType.getElementType() == bodyType.getElementType() &&
          (init.getType() == oldType || init.getType() == bodyType) &&
          yielded.getType() == oldType;
      RankedTensorType nativeType = hasHoistableBoundary ? bodyType : oldType;
      if (predicateType && nativeType &&
          nativeType.getRank() >= predicateType.getRank() &&
          llvm::equal(
              predicateType.getShape(),
              nativeType.getShape().take_front(predicateType.getRank()))) {
        Attribute encoding = nativeType.getEncoding();
        if (!isa<ttg::DistributedEncodingTrait>(encoding))
          return predicateOp.emitError(
              "expected distributed encodings on warp_predicate carried "
              "tensors");
        for (int rank = nativeType.getRank(); rank > predicateType.getRank();
             --rank)
          encoding = ttg::SliceEncodingAttr::get(
              predicateOp.getContext(), rank - 1,
              cast<ttg::DistributedEncodingTrait>(encoding));
        if (!predicateEncoding) {
          predicateEncoding = encoding;
        } else {
          auto selectedType = RankedTensorType::get(
              predicateType.getShape(), predicateType.getElementType(),
              *predicateEncoding);
          auto candidateType =
              RankedTensorType::get(predicateType.getShape(),
                                    predicateType.getElementType(), encoding);
          if (!ttg::isLayoutEquivalentIgnoringRegisterOrder(
                  ttg::toLinearLayout(selectedType),
                  ttg::toLinearLayout(candidateType)))
            conflictingPredicateEncoding = true;
        }
      }
    }
    if (conflictingPredicateEncoding)
      return predicateOp.emitError(
          "carried row values require conflicting predicate layouts");

    // Captured tensor conversions (for example Q -> dot operand layout) must
    // execute with the full CTA active. Collect first because moving while
    // walking the region invalidates the walk.
    SmallVector<ttg::ConvertLayoutOp> capturedConversions;
    predicateOp.getRegion().walk([&](ttg::ConvertLayoutOp convertOp) {
      Region *sourceRegion = convertOp.getSrc().getParentRegion();
      if (sourceRegion != &predicateOp.getRegion() &&
          !predicateOp.getRegion().isAncestor(sourceRegion))
        capturedConversions.push_back(convertOp);
    });
    for (ttg::ConvertLayoutOp convertOp : capturedConversions) {
      rewriter.moveOpBefore(convertOp, predicateOp);
      changed = true;
    }

    for (unsigned index = 0, e = predicateOp.getNumResults(); index < e;
         ++index) {
      OpResult result = cast<OpResult>(predicateOp.getResult(index));
      Value init = predicateOp.getInits()[index];
      Value yielded = yieldOp.getValues()[index];
      auto boundaryConvert = yielded.getDefiningOp<ttg::ConvertLayoutOp>();
      if (!boundaryConvert || !boundaryConvert->hasOneUse())
        continue;

      auto oldType = dyn_cast<RankedTensorType>(result.getType());
      auto bodyType =
          dyn_cast<RankedTensorType>(boundaryConvert.getSrc().getType());
      if (!oldType || !bodyType || oldType == bodyType ||
          oldType.getShape() != bodyType.getShape() ||
          oldType.getElementType() != bodyType.getElementType() ||
          (init.getType() != oldType && init.getType() != bodyType) ||
          yielded.getType() != oldType)
        continue;

      // Convert the false-path value before EXEC is restricted.
      Value convertedInit = init;
      if (init.getType() == oldType) {
        rewriter.setInsertionPoint(predicateOp);
        convertedInit = ttg::ConvertLayoutOp::create(
            rewriter, predicateOp.getLoc(), bodyType, init);
      }

      // Consecutive predicated regions can keep their carried values in the
      // native body layout. Record ordinary consumers that still require the
      // old type, and recognize an old->body bridge inserted when the sibling
      // region happened to be rewritten first.
      auto siblingAcceptsBodyType = [&](OpOperand &use) {
        auto sibling = dyn_cast<ttg::WarpPredicateOp>(use.getOwner());
        // Operand zero is the predicate, not a carried value.
        if (!sibling || use.getOperandNumber() == 0)
          return false;
        unsigned siblingIndex = use.getOperandNumber() - 1;
        if (siblingIndex >= sibling.getNumResults())
          return false;
        if (sibling.getResult(siblingIndex).getType() == bodyType)
          return true;
        if (sibling.getResult(siblingIndex).getType() != oldType ||
            !sibling.getRegion().hasOneBlock())
          return false;
        auto siblingYield = dyn_cast<ttg::PredicateYieldOp>(
            sibling.getRegion().front().getTerminator());
        if (!siblingYield || siblingIndex >= siblingYield.getNumOperands())
          return false;
        auto siblingBoundary = siblingYield.getValues()[siblingIndex]
                                   .getDefiningOp<ttg::ConvertLayoutOp>();
        return siblingBoundary && siblingBoundary->hasOneUse() &&
               siblingBoundary.getSrc().getType() == bodyType;
      };
      SmallVector<OpOperand *> oldLayoutUses;
      SmallVector<ttg::ConvertLayoutOp> siblingBridges;
      for (OpOperand &use : result.getUses()) {
        if (siblingAcceptsBodyType(use))
          continue;
        auto convert = dyn_cast<ttg::ConvertLayoutOp>(use.getOwner());
        if (convert && convert.getSrc() == result &&
            convert.getType() == bodyType) {
          siblingBridges.push_back(convert);
          continue;
        }
        oldLayoutUses.push_back(&use);
      }

      rewriter.modifyOpInPlace(predicateOp, [&] {
        predicateOp->setOperand(index + 1, convertedInit);
        result.setType(bodyType);
      });
      rewriter.modifyOpInPlace(yieldOp, [&] {
        yieldOp->setOperand(index, boundaryConvert.getSrc());
      });

      for (ttg::ConvertLayoutOp bridge : siblingBridges) {
        rewriter.replaceAllUsesWith(bridge.getResult(), result);
        rewriter.eraseOp(bridge);
      }

      // Restore the original type only for non-predicate consumers. A sibling
      // warp_predicate is rewritten to the same body type and consumes the
      // result directly, avoiding a redundant new->old->new round trip.
      if (!oldLayoutUses.empty()) {
        rewriter.setInsertionPointAfter(predicateOp);
        Value convertedResult = ttg::ConvertLayoutOp::create(
            rewriter, predicateOp.getLoc(), oldType, result);
        for (OpOperand *use : oldLayoutUses)
          use->set(convertedResult);
      }

      rewriter.eraseOp(boundaryConvert);
      changed = true;
    }

    if (predicateType && predicateEncoding) {
      auto wavePredicateType = RankedTensorType::get(
          predicateType.getShape(), predicateType.getElementType(),
          *predicateEncoding);
      if (wavePredicateType != predicateType) {
        rewriter.setInsertionPoint(predicateOp);
        Value wavePredicate = ttg::ConvertLayoutOp::create(
            rewriter, predicateOp.getLoc(), wavePredicateType,
            predicateOp.getPredicate());
        rewriter.modifyOpInPlace(
            predicateOp, [&] { predicateOp->setOperand(0, wavePredicate); });
        changed = true;
      }
    }

    return success(changed);
  }
};

static RankedTensorType getNewTensorType(RankedTensorType origType,
                                         Attribute encoding) {
  return RankedTensorType::get(origType.getShape(), origType.getElementType(),
                               encoding);
}

static bool isRetaggableTensorProducerValue(Value value) {
  if (!isa<RankedTensorType>(value.getType()))
    return false;

  Operation *definingOp = value.getDefiningOp();
  return isa_and_nonnull<ttg::LocalLoadOp>(definingOp);
}

static Type getTensorCandidateType(Value value, DataFlowSolver &solver,
                                   const llvm::DenseSet<Value> &blockedValues) {
  auto tensorType = cast<RankedTensorType>(value.getType());
  if (blockedValues.contains(value))
    return tensorType;

  // A user-pinned register layout is a hard constraint: never retag it.
  if (isa_and_nonnull<UserLayoutAttr>(tensorType.getEncoding()))
    return tensorType;

  auto *lattice = solver.lookupState<TensorLayoutLattice>(value);
  if (!lattice || lattice->getValue().isUninitialized() ||
      lattice->getValue().isUnknown())
    return tensorType;

  return getNewTensorType(tensorType, lattice->getValue().getLayoutEncoding());
}

static bool isRetaggableLocalAllocLoadFallback(ttg::LocalAllocOp allocOp) {
  if (!allocOp.getSrc() || !isa<RankedTensorType>(allocOp.getSrc().getType()))
    return false;
  if (allocOp->use_empty())
    return false;

  for (Operation *user : allocOp->getUsers()) {
    auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(user);
    if (!localLoadOp)
      return false;
    auto resultType = dyn_cast<RankedTensorType>(localLoadOp.getType());
    if (!resultType ||
        !isSupportedDotConstraintEncoding(resultType.getEncoding()))
      return false;
  }
  return true;
}

static void
rewriteTensorValueFromLattice(Value value, DataFlowSolver &solver,
                              const llvm::DenseSet<Value> &blockedValues) {
  if (!isRetaggableTensorProducerValue(value))
    return;

  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return;
  auto newType = cast<RankedTensorType>(
      getTensorCandidateType(value, solver, blockedValues));
  if (newType != tensorType)
    value.setType(newType);
}

static ttg::MemDescType getNewMemDescType(ttg::MemDescType origType,
                                          Attribute encoding) {
  return ttg::MemDescType::get(origType.getShape(), origType.getElementType(),
                               encoding, origType.getMemorySpace(),
                               origType.getMutableMemory(),
                               origType.getAllocShape());
}

static FailureOr<const LayoutEncodingLattice *>
lookupMemDescLatticeOrEmitError(Value value, DataFlowSolver &solver,
                                Operation *diagnosticOp) {
  auto *lattice = solver.lookupState<LayoutEncodingLattice>(value);
  if (lattice)
    return lattice;

  diagnosticOp->emitError()
      << "expected memdesc layout lattice for value " << value;
  return failure();
}

static FailureOr<LayoutEncoding>
getMemDescConsensusLayout(ArrayRef<Value> values, DataFlowSolver &solver,
                          Operation *diagnosticOp) {
  LayoutEncoding consensus;
  for (Value value : values) {
    FailureOr<const LayoutEncodingLattice *> lattice =
        lookupMemDescLatticeOrEmitError(value, solver, diagnosticOp);
    if (failed(lattice))
      return failure();
    consensus = LayoutEncoding::join(consensus, (*lattice)->getValue());
  }
  return consensus;
}

static LogicalResult rewriteMemDescValueFromLattice(Value value,
                                                    DataFlowSolver &solver,
                                                    Operation *diagnosticOp) {
  auto origType = dyn_cast<ttg::MemDescType>(value.getType());
  if (!origType)
    return success();

  // A user-pinned shared layout is a hard constraint: never retag it to satisfy
  // a consumer. Leave the wrapper in place; tlx-resolve-placeholder-layouts
  // unwraps it to the concrete layout the user asked for.
  if (isa<UserLayoutAttr>(origType.getEncoding()))
    return success();

  FailureOr<const LayoutEncodingLattice *> lattice =
      lookupMemDescLatticeOrEmitError(value, solver, diagnosticOp);
  if (failed(lattice))
    return failure();

  LayoutEncoding layout = (*lattice)->getValue();
  if (layout.isUninitialized())
    return success();
  if (layout.isUnknown()) {
    LDBG("Leaving memdesc value unchanged due to unknown layout: " << value);
    return success();
  }

  auto newType = getNewMemDescType(origType, layout.getLayoutEncoding());
  if (newType != origType)
    value.setType(newType);
  return success();
}

static void
collectRegionBranchSuccessors(RegionBranchOpInterface branchOp,
                              SmallVectorImpl<RegionSuccessor> &successors) {
  auto appendUniqueSuccessors = [&](ArrayRef<RegionSuccessor> newSuccessors) {
    for (RegionSuccessor successor : newSuccessors) {
      if (!llvm::is_contained(successors, successor))
        successors.push_back(successor);
    }
  };

  SmallVector<RegionSuccessor> newSuccessors;
  branchOp.getSuccessorRegions(RegionBranchPoint::parent(), newSuccessors);
  appendUniqueSuccessors(newSuccessors);
  for (Region &region : branchOp->getRegions()) {
    newSuccessors.clear();
    branchOp.getSuccessorRegions(region, newSuccessors);
    appendUniqueSuccessors(newSuccessors);
  }
}

using TensorTypeMap = llvm::DenseMap<Value, Type>;

struct TensorRegionInfo {
  llvm::DenseSet<Value> blockedValues;
  TensorTypeMap regionTypes;
};

// The type an incoming region edge value actually contributes to the consensus.
static Type getTensorEdgeType(Value value, DataFlowSolver &solver,
                              const llvm::DenseSet<Value> &blockedValues,
                              const TensorTypeMap &regionTypes) {
  auto tensorType = cast<RankedTensorType>(value.getType());
  // Region carriers must use the type realizable by their nested input edges.
  if (auto it = regionTypes.find(value); it != regionTypes.end())
    return it->second;
  if (!isRetaggableTensorProducerValue(value))
    return tensorType;
  return getTensorCandidateType(value, solver, blockedValues);
}

static std::optional<Type>
getTensorConsensusType(ValueRange values, DataFlowSolver &solver,
                       const llvm::DenseSet<Value> &blockedValues,
                       const TensorTypeMap &regionTypes) {
  if (values.empty())
    return std::nullopt;

  std::optional<Type> consensusType;
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      return std::nullopt;

    Type candidateType =
        getTensorEdgeType(value, solver, blockedValues, regionTypes);
    if (!consensusType) {
      consensusType = candidateType;
      continue;
    }
    if (*consensusType != candidateType)
      return std::nullopt;
  }
  return consensusType;
}

static TensorTypeMap
computeTensorRegionTypes(triton::FuncOp funcOp, DataFlowSolver &solver,
                         const llvm::DenseSet<Value> &blockedValues) {
  TensorTypeMap regionTypes;
  funcOp.walk([&](RegionBranchOpInterface branchOp) {
    SmallVector<RegionSuccessor> successors;
    collectRegionBranchSuccessors(branchOp, successors);
    for (RegionSuccessor successor : successors) {
      for (Value input : branchOp.getSuccessorInputs(successor)) {
        if (isa<RankedTensorType>(input.getType()))
          regionTypes.try_emplace(
              input, getTensorCandidateType(input, solver, blockedValues));
      }
    }
  });

  bool changed = true;
  while (changed) {
    changed = false;
    funcOp.walk<WalkOrder::PostOrder>([&](RegionBranchOpInterface branchOp) {
      SmallVector<RegionSuccessor> successors;
      collectRegionBranchSuccessors(branchOp, successors);
      for (RegionSuccessor successor : successors) {
        ValueRange inputs = branchOp.getSuccessorInputs(successor);
        for (auto [index, input] : llvm::enumerate(inputs)) {
          auto tensorType = dyn_cast<RankedTensorType>(input.getType());
          if (!tensorType)
            continue;

          SmallVector<Value> predecessors;
          branchOp.getPredecessorValues(successor, index, predecessors);
          if (predecessors.empty())
            continue;

          std::optional<Type> consensus = getTensorConsensusType(
              ValueRange(predecessors), solver, blockedValues, regionTypes);
          Type type = consensus.value_or(input.getType());
          if (blockedValues.contains(input) ||
              isa_and_nonnull<UserLayoutAttr>(tensorType.getEncoding()))
            type = input.getType();
          auto it = regionTypes.find(input);
          assert(it != regionTypes.end() && "expected seeded region type");
          if (it == regionTypes.end())
            continue;
          if (it->second == type)
            continue;
          it->second = type;
          changed = true;
        }
      }
    });
  }
  return regionTypes;
}

static TensorRegionInfo computeTensorRegionInfo(triton::FuncOp funcOp,
                                                DataFlowSolver &solver) {
  TensorRegionInfo info;
  bool changed = true;
  while (changed) {
    changed = false;
    info.regionTypes =
        computeTensorRegionTypes(funcOp, solver, info.blockedValues);
    funcOp.walk([&](RegionBranchOpInterface branchOp) {
      SmallVector<RegionSuccessor> successors;
      collectRegionBranchSuccessors(branchOp, successors);

      for (RegionSuccessor successor : successors) {
        ValueRange successorInputs = branchOp.getSuccessorInputs(successor);
        for (auto [index, successorInput] : llvm::enumerate(successorInputs)) {
          if (!isa<RankedTensorType>(successorInput.getType()))
            continue;

          SmallVector<Value> predecessorValues;
          branchOp.getPredecessorValues(successor, index, predecessorValues);
          if (predecessorValues.empty())
            continue;

          bool successorBlocked = info.blockedValues.contains(successorInput);
          if (!successorBlocked &&
              getTensorConsensusType(ValueRange(predecessorValues), solver,
                                     info.blockedValues, info.regionTypes))
            continue;

          if (!successorBlocked)
            LDBG("Blocking tensor carrier value due to inconsistent "
                 "predecessor layouts at "
                 << branchOp->getName());
          changed |= info.blockedValues.insert(successorInput).second;
          for (Value predecessorValue : predecessorValues) {
            if (!isa<RankedTensorType>(predecessorValue.getType()))
              continue;
            changed |= info.blockedValues.insert(predecessorValue).second;
          }
        }
      }
    });
  }

  return info;
}

static void updateTensorRegionBranchTypes(triton::FuncOp funcOp,
                                          const TensorTypeMap &regionTypes) {
  funcOp.walk<WalkOrder::PostOrder>([&](RegionBranchOpInterface branchOp) {
    SmallVector<RegionSuccessor> successors;
    collectRegionBranchSuccessors(branchOp, successors);

    for (RegionSuccessor successor : successors) {
      for (Value input : branchOp.getSuccessorInputs(successor)) {
        auto it = regionTypes.find(input);
        if (it != regionTypes.end() && input.getType() != it->second)
          input.setType(it->second);
      }
    }
  });
}

// Retire user-pinned *shared* layouts: replace every #tlx.user_layout<L>
// encoding on a MemDescType (results and block arguments) with its wrapped
// concrete layout L. Runs after the dataflow rewrite has refused to retag these
// buffers, so the user's choice has been honored and the marker is no longer
// needed. Done here (not only in tlx-resolve-placeholder-layouts) because the
// AMD pipeline does not run that pass, but always runs this one.
//
// Register (RankedTensorType) user layouts are intentionally left wrapped: they
// must survive as anchors through remove-layout-conversions and the other
// layout passes, and are unwrapped later by tlx-finalize-user-layouts.
static void unwrapUserLayoutEncodings(Operation *root) {
  auto rewrite = [](Value v) {
    if (auto md = dyn_cast<ttg::MemDescType>(v.getType())) {
      if (auto w = dyn_cast_or_null<UserLayoutAttr>(md.getEncoding()))
        v.setType(getNewMemDescType(md, w.getLayout()));
    }
  };
  root->walk([&](Operation *op) {
    for (Value result : op->getResults())
      rewrite(result);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          rewrite(arg);
  });
}

class TlxPropagateLayoutPass
    : public impl::TlxPropagateLayoutBase<TlxPropagateLayoutPass> {
public:
  using impl::TlxPropagateLayoutBase<
      TlxPropagateLayoutPass>::TlxPropagateLayoutBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    // We can terminate early if we don't have a layout constraint.
    WalkResult walkResult = funcOp.walk([&](mlir::Operation *op) {
      if (isa<tlx::RequireLayoutOp, tlx::ReleaseLayoutOp>(op))
        return WalkResult::interrupt();
      if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
        if (isRetaggableLocalAllocLoadFallback(allocOp))
          return WalkResult::interrupt();
      }
      if (auto copyOp = dyn_cast<ttng::TMEMCopyOp>(op)) {
        auto dstType = cast<ttg::MemDescType>(copyOp.getDst().getType());
        if (isa_and_nonnull<DummyTMEMLayoutAttr>(dstType.getEncoding()))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!walkResult.wasInterrupted())
      return;

    SymbolTableCollection symbolTable;
    Operation *op = getOperation();
    DataFlowSolver solver;

    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<LayoutBackwardPropagation>(symbolTable);
    solver.load<LayoutForwardPropagation>();
    solver.load<TensorBackwardPropagation>(symbolTable);
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    TensorRegionInfo tensorRegionInfo = computeTensorRegionInfo(funcOp, solver);

    WalkResult typeRewriteWalk = funcOp.walk([&](mlir::Operation *op) {
      if (isa<tlx::RequireLayoutOp>(op))
        return WalkResult::advance();

      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        for (auto [i, capture] :
             llvm::enumerate(wsOp.getPartitionOp().getExplicitCaptures())) {
          auto captureType = dyn_cast<ttg::MemDescType>(capture.getType());
          if (!captureType)
            continue;
          // User-pinned shared layouts are fixed; don't retag WS captures.
          if (isa<UserLayoutAttr>(captureType.getEncoding()))
            continue;

          SmallVector<Value> relatedValues;
          relatedValues.push_back(capture);
          for (Region *partitionRegion : wsOp.getPartitionRegions())
            relatedValues.push_back(partitionRegion->getArgument(i));

          FailureOr<LayoutEncoding> consensus =
              getMemDescConsensusLayout(relatedValues, solver, wsOp);
          if (failed(consensus))
            return WalkResult::interrupt();
          if (consensus->isUninitialized())
            continue;
          if (consensus->isUnknown()) {
            LDBG("Leaving warp_specialize capture #" << i
                                                     << " unchanged due to "
                                                        "non-concrete "
                                                        "partition consensus");
            continue;
          }

          auto newType =
              getNewMemDescType(captureType, consensus->getLayoutEncoding());
          if (capture.getType() != newType)
            capture.setType(newType);
          for (Region *partitionRegion : wsOp.getPartitionRegions()) {
            if (partitionRegion->getArgument(i).getType() != newType)
              partitionRegion->getArgument(i).setType(newType);
          }
        }
        return WalkResult::advance();
      }

      for (Value result : op->getResults()) {
        if (!isa<ttg::MemDescType>(result.getType())) {
          rewriteTensorValueFromLattice(result, solver,
                                        tensorRegionInfo.blockedValues);
          continue;
        }

        if (failed(rewriteMemDescValueFromLattice(result, solver, op)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (typeRewriteWalk.wasInterrupted())
      return signalPassFailure();

    updateTensorRegionBranchTypes(funcOp, tensorRegionInfo.regionTypes);

    // Verify that no DummyTMEMLayoutAttr remains after layout propagation.
    bool hasDummyLayout = false;
    funcOp.walk([&](ttng::TMEMAllocOp allocOp) {
      auto encoding = allocOp.getType().getEncoding();
      if (isa_and_nonnull<DummyTMEMLayoutAttr>(encoding)) {
        allocOp.emitError(
            "DummyTMEMLayoutAttr was not resolved during layout propagation");
        hasDummyLayout = true;
      }
      return WalkResult::advance();
    });
    if (hasDummyLayout)
      return signalPassFailure();
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RequireLayoutPattern>(context);
    patterns.add<ReleaseLayoutPattern>(context);
    patterns.add<FoldRetaggedLocalAllocLoad>(context);
    patterns.add<FoldLocalAllocLoadFallback>(context);
    patterns.add<HoistWarpPredicateLayoutConversions>(context);

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      signalPassFailure();

    // Honor-then-retire user-pinned layouts. The dataflow rewrite and the fold
    // patterns above all skip values whose encoding is the wrapper (so the
    // user's choice is never retagged or folded away); only now, after they
    // have run, do we unwrap the marker back to the concrete layout the user
    // asked for.
    unwrapUserLayoutEncodings(getOperation());
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
