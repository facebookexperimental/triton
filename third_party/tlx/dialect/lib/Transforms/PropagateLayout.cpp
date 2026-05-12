#include "IR/Dialect.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tlx/dialect/include/Analysis/LayoutPropagation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/DataFlowFramework.h"
#define DEBUG_TYPE "tlx-propagate-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;
namespace tt = ::mlir::triton;
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
    if (requireLayoutOp.getSrc().getType() == requireLayoutOp.getType()) {
      rewriter.replaceOp(requireLayoutOp, requireLayoutOp.getSrc());
      return success();
    }
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
        requireLayoutOp, requireLayoutOp.getType(), requireLayoutOp.getSrc());
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

static RankedTensorType getNewTensorType(RankedTensorType origType,
                                         Attribute encoding) {
  return RankedTensorType::get(origType.getShape(), origType.getElementType(),
                               encoding);
}

static bool isRetaggableTensorProducerValue(Value value) {
  if (!isa<RankedTensorType>(value.getType()))
    return false;

  Operation *definingOp = value.getDefiningOp();
  // Sparse dataflow handles the RegionBranchOpInterface edge consistency; if
  // all incoming values agree, retagging the region result is safe.
  return isa_and_nonnull<ttg::LocalLoadOp, RegionBranchOpInterface>(definingOp);
}

static Type getTensorCandidateType(Value value, DataFlowSolver &solver,
                                   const llvm::DenseSet<Value> &blockedValues) {
  auto tensorType = cast<RankedTensorType>(value.getType());
  if (blockedValues.contains(value))
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

static std::optional<Type>
getTensorConsensusType(ValueRange values, DataFlowSolver &solver,
                       const llvm::DenseSet<Value> &blockedValues) {
  if (values.empty())
    return std::nullopt;

  std::optional<Type> consensusType;
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      return std::nullopt;

    Type candidateType = getTensorCandidateType(value, solver, blockedValues);
    if (!consensusType) {
      consensusType = candidateType;
      continue;
    }
    if (*consensusType != candidateType)
      return std::nullopt;
  }
  return consensusType;
}

static llvm::DenseSet<Value>
computeBlockedTensorValues(triton::FuncOp funcOp, DataFlowSolver &solver) {
  llvm::DenseSet<Value> blockedValues;
  bool changed = true;
  while (changed) {
    changed = false;
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

          if (getTensorConsensusType(ValueRange(predecessorValues), solver,
                                     blockedValues))
            continue;

          LDBG("Blocking tensor carrier value due to inconsistent predecessor "
               "layouts at "
               << branchOp->getName());
          changed |= blockedValues.insert(successorInput).second;
          for (Value predecessorValue : predecessorValues) {
            if (!isa<RankedTensorType>(predecessorValue.getType()))
              continue;
            changed |= blockedValues.insert(predecessorValue).second;
          }
        }
      }

      return WalkResult::advance();
    });
  }

  return blockedValues;
}

static void
updateTensorRegionBranchTypes(triton::FuncOp funcOp, DataFlowSolver &solver,
                              const llvm::DenseSet<Value> &blockedValues) {
  funcOp.walk<WalkOrder::PostOrder>([&](RegionBranchOpInterface branchOp) {
    SmallVector<RegionSuccessor> successors;
    collectRegionBranchSuccessors(branchOp, successors);

    bool changed = true;
    while (changed) {
      changed = false;
      for (RegionSuccessor successor : successors) {
        ValueRange successorInputs = branchOp.getSuccessorInputs(successor);
        for (auto [index, successorInput] : llvm::enumerate(successorInputs)) {
          if (!isa<RankedTensorType>(successorInput.getType()))
            continue;

          SmallVector<Value> predecessorValues;
          branchOp.getPredecessorValues(successor, index, predecessorValues);
          std::optional<Type> consensusType = getTensorConsensusType(
              ValueRange(predecessorValues), solver, blockedValues);
          if (!consensusType || successorInput.getType() == *consensusType)
            continue;

          successorInput.setType(*consensusType);
          changed = true;
        }
      }
    }
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

    llvm::DenseSet<Value> blockedTensorValues =
        computeBlockedTensorValues(funcOp, solver);

    WalkResult typeRewriteWalk = funcOp.walk([&](mlir::Operation *op) {
      if (isa<tlx::RequireLayoutOp>(op))
        return WalkResult::advance();

      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        for (auto [i, capture] :
             llvm::enumerate(wsOp.getPartitionOp().getExplicitCaptures())) {
          auto captureType = dyn_cast<ttg::MemDescType>(capture.getType());
          if (!captureType)
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

      for (auto [i, result] : llvm::enumerate(op->getResults())) {
        if (!isa<ttg::MemDescType>(result.getType())) {
          rewriteTensorValueFromLattice(result, solver, blockedTensorValues);
          continue;
        }

        if (failed(rewriteMemDescValueFromLattice(result, solver, op)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (typeRewriteWalk.wasInterrupted())
      return signalPassFailure();

    updateTensorRegionBranchTypes(funcOp, solver, blockedTensorValues);

    // Verify that no DummyTMEMLayoutAttr remains after layout propagation
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

    return;
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RequireLayoutPattern>(context);
    patterns.add<ReleaseLayoutPattern>(context);
    patterns.add<FoldRetaggedLocalAllocLoad>(context);
    patterns.add<FoldLocalAllocLoadFallback>(context);

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
