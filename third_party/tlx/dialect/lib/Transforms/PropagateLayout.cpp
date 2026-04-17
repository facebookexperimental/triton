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

  auto *lattice = solver.lookupState<TensorLayoutLattice>(value);
  if (!lattice || lattice->getValue().isUninitialized() ||
      lattice->getValue().isUnknown())
    return tensorType;

  return getNewTensorType(tensorType, lattice->getValue().getLayoutEncoding());
}

static void rewriteTensorValueFromLattice(
    Value value, DataFlowSolver &solver,
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

static void updateTensorRegionBranchTypes(
    triton::FuncOp funcOp, DataFlowSolver &solver,
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
      return WalkResult::advance();
    });
    if (!walkResult.wasInterrupted())
      return;

    PatternRewriter rewriter(&getContext());
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

    auto getNewMemDescType = [&](ttg::MemDescType origType,
                                 Attribute encoding) {
      return ttg::MemDescType::get(
          origType.getShape(), origType.getElementType(), encoding,
          origType.getMemorySpace(), origType.getMutableMemory());
    };

    funcOp.walk([&](mlir::Operation *op) {
      if (isa<tlx::RequireLayoutOp>(op))
        return WalkResult::advance();

      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        Region *firstRegion = wsOp.getPartitionRegions()[0];
        for (auto [i, blockArg] :
             llvm::enumerate(firstRegion->getArguments())) {
          if (!isa<ttg::MemDescType>(blockArg.getType()))
            continue;
          auto lattice = solver.lookupState<LayoutEncodingLattice>(blockArg);
          if (!lattice)
            llvm_unreachable("Lattice not found.");
          if (lattice->getValue().isUninitialized() ||
              lattice->getValue().isUnknown())
            continue;
          for (Region *partitionRegion : wsOp.getPartitionRegions()) {
            if (auto origType =
                    dyn_cast<ttg::MemDescType>(blockArg.getType())) {
              auto newType = getNewMemDescType(
                  origType, lattice->getValue().getLayoutEncoding());
              partitionRegion->getArgument(i).setType(newType);
            }
          }
        }
        return WalkResult::advance();
      }

      for (auto [i, result] : llvm::enumerate(op->getResults())) {
        if (!isa<ttg::MemDescType>(result.getType()))
          rewriteTensorValueFromLattice(result, solver, blockedTensorValues);
        else {
          auto *lattice = solver.lookupState<LayoutEncodingLattice>(result);
          if (!lattice)
            llvm_unreachable("Lattice not found.");
          if (lattice->getValue().isUninitialized() ||
              lattice->getValue().isUnknown())
            continue;
          if (auto origType = dyn_cast<ttg::MemDescType>(result.getType())) {
            auto newType = getNewMemDescType(
                origType, lattice->getValue().getLayoutEncoding());
            op->getResult(i).setType(newType);
          }
        }
      }
      return WalkResult::advance();
    });

    updateTensorRegionBranchTypes(funcOp, solver, blockedTensorValues);

    // Fix up RequireLayoutOps feeding into TMEMStoreOps with scales encoding.
    // ResolvePlaceholderLayouts assigned a generic TMEM-compatible register
    // layout, but for scales the register layout must use
    // getScaleTMEMStoreLinearLayout.
    funcOp.walk([&](ttng::TMEMStoreOp storeOp) {
      auto memTy = storeOp.getDst().getType();
      if (!isa<ttng::TensorMemoryScalesEncodingAttr>(memTy.getEncoding()))
        return WalkResult::advance();

      auto requireOp = storeOp.getSrc().getDefiningOp<RequireLayoutOp>();
      if (!requireOp)
        return WalkResult::advance();

      auto srcTy = cast<RankedTensorType>(requireOp.getResult().getType());
      int numWarps = ttg::lookupNumWarps(storeOp);
      // TODO: port getScaleTMEMStoreLinearLayout to upstream
      // // TODO: port getScaleTMEMStoreLinearLayout
      // auto scalesLL = ttg::getScaleTMEMStoreLinearLayout(srcTy, numWarps);
      // auto newEncoding = ttg::LinearEncodingAttr::get(srcTy.getContext(), scalesLL);
      // auto newType = RankedTensorType::get(srcTy.getShape(),
      //                                      srcTy.getElementType(), newEncoding);
      // requireOp->getResult(0).setType(newType);
      return WalkResult::advance();
    });

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

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
