#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "tritonamdgpu-warp-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUPRESERVEWARPPIPELINEOWNERS
#define GEN_PASS_DEF_TRITONAMDGPUNORMALIZEWARPPIPELINE
#define GEN_PASS_DEF_TRITONAMDGPUWARPPIPELINE
#include "TritonAMDGPUTransforms/Passes.h.inc"

static constexpr llvm::StringLiteral kOwnerBeginAttr =
    "triton.warp_pipeline.owner_begin";
static constexpr llvm::StringLiteral kOwnerEndAttr =
    "triton.warp_pipeline.owner_end";
static constexpr llvm::StringLiteral kOwnerStageCountAttr =
    "triton.warp_pipeline.owner_stage_count";
static constexpr llvm::StringLiteral kStageBeginAttr =
    "triton.warp_pipeline.stage_begin";
static constexpr llvm::StringLiteral kFlatStageAttr =
    "triton.warp_pipeline.flat";

// Ops that may appear between pipeline stages but never inside one.  Pre-
// existing memory-fence/wait ops at cluster boundaries are tolerated so that
// prefetch patterns continue to work; encountering one mid-cluster is treated
// as malformed input by the callers.
static bool canSitBetweenStages(Operation *op) {
  return isa<ttg::AsyncWaitOp, gpu::BarrierOp, triton::gpu::BarrierOp,
             tt::amdgpu::AsyncTDMWait>(op);
}

// True if `op` carries the cluster-end marker emitted by the frontend.
static bool isPipelineBorder(Operation *op) {
  return op->hasAttr("triton.warp_pipeline.border");
}

static std::optional<int64_t> getOwnerMarkerId(Operation *op,
                                               StringRef attrName) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return std::nullopt;
}

static Operation *createOwnerMarker(OpBuilder &b, Location loc,
                                    StringRef attrName, int64_t ownerId) {
  auto marker =
      ROCDL::SchedBarrier::create(b, loc, ROCDL::SchedGroupMask::none);
  marker->setAttr(attrName, b.getI64IntegerAttr(ownerId));
  return marker;
}

// True if `op` is a structured loop (scf.for / scf.while).  Pipeline clusters
// are straight-line scheduling units, so loops remain boundaries instead of
// being absorbed.  This also rejects nested warp-pipelined scf.for ops.
static bool isLoopOp(Operation *op) {
  return isa<scf::ForOp, scf::WhileOp>(op);
}

// Outcome of attempting to build a pipeline from a region.
//   NotApplicable: no border markers were present (the region opted out).
//   Created:       a pipeline was successfully materialized.
//   Malformed:     border markers were present but the pipeline could not be
//                  built; an error has been emitted at the offending op.
enum class PipelineResult { NotApplicable, Created, Malformed };

struct BorderMarker {
  StringAttr cluster;
  int priority;
  bool allowMemoryReorder;
};

// Read the per-stage policy from a border marker op. Priority defaults to -1
// when the marker doesn't carry the optional priority attribute.
static BorderMarker readBorderMarker(Operation *op) {
  StringAttr clusterStr =
      op->getAttrOfType<StringAttr>("triton.warp_pipeline.border");
  int priority = -1;
  if (auto intAttr =
          op->getAttrOfType<IntegerAttr>("triton.warp_pipeline.priority"))
    priority = intAttr.getInt();
  return {clusterStr, priority,
          op->hasAttr("triton.warp_pipeline.allow_memory_reorder")};
}

// If `cluster` is empty, materialize a dummy SchedBarrier so the cluster is
// non-empty.  This lets users deliberately request a pipeline bubble by
// emitting a begin/end pair with no body between them.
static void addDummyOpIfEmptyCluster(OpBuilder &b, Location loc,
                                     Operation *insertBefore,
                                     SmallVectorImpl<Operation *> &cluster) {
  if (!cluster.empty())
    return;
  b.setInsertionPoint(insertBefore);
  auto dummyOp =
      ROCDL::SchedBarrier::create(b, loc, ROCDL::SchedGroupMask::none);
  dummyOp->setAttr("triton.warp_pipeline.empty_cluster", b.getUnitAttr());
  cluster.push_back(dummyOp);
}

// Create a scf.execute_region op representing a pipeline cluster.
static void createClusterOp(OpBuilder &b, Location loc,
                            SmallVector<Operation *> &ops, BorderMarker marker,
                            bool isFlat = false) {
  assert(!ops.empty() && "empty stage");

  // Insert the execute_region before the first op in the cluster.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(ops.front());

  // Build fast ops lookup for the cluster.
  llvm::SmallPtrSet<Operation *, 32> opsLookup(ops.begin(), ops.end());

  // Determine which results have users outside the cluster.
  SmallVector<Type> yieldedTypes;
  SmallVector<std::pair<OpResult, unsigned>>
      resultToYieldIdx; // (orig result, idx in yields)

  for (Operation *op : ops) {
    for (OpResult r : op->getResults()) {
      bool hasExternalUse = llvm::any_of(r.getUsers(), [&](Operation *u) {
        return !opsLookup.contains(op->getBlock()->findAncestorOpInBlock(*u));
      });
      if (hasExternalUse) {
        yieldedTypes.push_back(r.getType());
        resultToYieldIdx.emplace_back(
            r, static_cast<unsigned>(yieldedTypes.size() - 1));
      }
    }
  }

  // Create the execute_region with the final result types.
  auto exec = scf::ExecuteRegionOp::create(b, loc, yieldedTypes);
  Block *body = &exec.getRegion().emplaceBlock();
  b.setInsertionPointToStart(body);

  // Clone ops in order, remapping intra-cluster defs to their clones.
  IRMapping mapping;
  for (Operation *op : ops) {
    Operation *clone = b.clone(*op, mapping);
    // Map each result so subsequent clones use the cloned defs.
    for (auto [origRes, clonedRes] :
         llvm::zip(op->getResults(), clone->getResults()))
      mapping.map(origRes, clonedRes);
  }

  // Build the yield values.
  SmallVector<Value> yieldVals(yieldedTypes.size());
  for (auto [origRes, yieldIdx] : resultToYieldIdx) {
    Value mapped = mapping.lookupOrNull(origRes);
    assert(mapped && "mapped result missing");
    yieldVals[yieldIdx] = mapped;
  }
  scf::YieldOp::create(b, loc, yieldVals);

  // Replace external uses of original results with exec results.
  // Internal uses were already remapped when cloning.
  for (auto [origRes, yieldIdx] : resultToYieldIdx) {
    Value repl = exec.getResult(yieldIdx);
    origRes.replaceUsesWithIf(repl, [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      return owner && owner->getParentOp() != exec;
    });
  }

  // Erase original ops now that their external uses are redirected.
  std::reverse(ops.begin(), ops.end());
  for (Operation *op : ops)
    op->erase();

  // Keep the region structured for later conversion.
  exec.setNoInline(true);
  exec->setAttr("triton.warp_pipeline.stage", marker.cluster);
  if (isFlat)
    exec->setAttr(kFlatStageAttr, b.getUnitAttr());
  if (marker.priority > -1) {
    exec->setAttr("triton.warp_pipeline.priority",
                  b.getI32IntegerAttr(marker.priority));
  }
  if (marker.allowMemoryReorder)
    exec->setAttr("triton.warp_pipeline.allow_memory_reorder", b.getUnitAttr());

  LLVM_DEBUG(llvm::dbgs() << "[warp-pipeline] created stage with " << ops.size()
                          << " ops and " << yieldedTypes.size() << " yields\n");
  return;
}

// Stage markers survive layout propagation and loop unrolling. Only operations
// between a matched pair belong to a fixed stage. Labels need not be unique:
// unrolling copies both markers, and each pair remains in the same block.
struct Stage {
  Operation *begin;
  Operation *end;
};

static LogicalResult collectStages(Operation *begin, Operation *end,
                                   SmallVectorImpl<Stage> &stages) {
  Operation *active = nullptr;
  for (Operation *op = begin; op && op != end; op = op->getNextNode()) {
    if (op->hasAttr(kStageBeginAttr)) {
      if (active)
        return op->emitError(
            "nested warp_pipeline_stage regions are not supported");
      active = op;
    } else if (isPipelineBorder(op)) {
      if (!active || active->getAttr(kStageBeginAttr) !=
                         op->getAttr("triton.warp_pipeline.border"))
        return op->emitError("unmatched warp_pipeline_stage end marker");
      stages.push_back({active, op});
      active = nullptr;
    } else if (active) {
      if (canSitBetweenStages(op))
        return op->emitError("barrier or wait op cannot appear inside a "
                             "warp_pipeline_stage region");
      if (isLoopOp(op))
        return op->emitError(
            "loop op cannot appear inside a warp_pipeline_stage "
            "region; place stage blocks inside the loop body");
    }
  }
  if (active)
    return active->emitError("unmatched warp_pipeline_stage begin marker");
  return success();
}

static bool canPlaceInStage(Operation *op) {
  if (op->getNumRegions() == 0 && triton::isPureScalarOp(op))
    return true;
  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    // Register reorders and warp shuffles need no shared memory or CTA barrier.
    // The op verifier also checks that forced shuffles stay inside one warp.
    return !cvtNeedsSharedMemory(cvt);
  }
  return false;
}

// This is a cold compiler path. Plan placement before changing the block.
// Reverse SSA order gives each unassigned user a stage before its producers.
// In a straight-line scope, the earliest use sets the latest legal stage.
// A fixed use (for example a wait) requires the producer's stage to end before
// that use. Loop yields are uses at the end of this iteration, not backedges.
static LogicalResult normalizeStageInputs(Operation *begin, Operation *end,
                                          bool hasOwner) {
  SmallVector<Stage> stages;
  if (failed(collectStages(begin, end, stages)))
    return failure();
  if (stages.empty())
    return success();

  if (hasOwner) {
    auto originalCount =
        begin->getPrevNode()->getAttrOfType<IntegerAttr>(kOwnerStageCountAttr);
    // Owner recovery removes annotations when folding leaves one iteration.
    // Such a scope does not execute as a pipeline and needs no placement.
    if (originalCount && stages.size() == originalCount.getInt())
      return success();
  }

  if (!hasOwner && !isa<scf::ForOp>(begin->getParentOp())) {
    if (!isa<tt::FuncOp>(begin->getParentOp()))
      return stages.front().begin->emitError(
          "warp_pipeline_stage requires a loop body or a function block");
    // Explicit boundaries keep a flat pipeline's prologue and epilogue out.
    begin = stages.front().begin;
    end = stages.back().end->getNextNode();
  }

  DenseMap<Operation *, unsigned> placement;
  for (auto [index, stage] : llvm::enumerate(stages))
    for (Operation *op = stage.begin; op != stage.end->getNextNode();
         op = op->getNextNode())
      placement[op] = index;

  SmallVector<Operation *> inputs;
  for (Operation *op = begin; op && op != end; op = op->getNextNode()) {
    if (placement.contains(op) || canSitBetweenStages(op) ||
        op->hasTrait<OpTrait::IsTerminator>())
      continue;
    if (!canPlaceInStage(op))
      return op->emitError(
          "cannot place operation outside warp_pipeline_stage: "
          "expected pure scalar computation or a wave-local "
          "layout conversion");
    inputs.push_back(op);
  }

  Block *block = begin->getBlock();
  for (Operation *op : llvm::reverse(inputs)) {
    unsigned latest = stages.size() - 1;
    Operation *limitingUser = nullptr;
    for (Operation *user : op->getUsers()) {
      Operation *ancestor = block->findAncestorOpInBlock(*user);
      if (!ancestor)
        continue; // A use outside the scope follows it by SSA dominance.
      auto it = placement.find(ancestor);
      if (it != placement.end()) {
        if (it->second <= latest) {
          latest = it->second;
          limitingUser = ancestor;
        }
      } else {
        // Fixed operations cannot move. Find the last stage before this use.
        while (!stages[latest].end->isBeforeInBlock(ancestor)) {
          if (latest == 0) {
            auto diag = op->emitError("no legal warp-pipeline stage for input");
            diag.attachNote(user->getLoc())
                << "this fixed use precedes every available stage";
            return failure();
          }
          --latest;
        }
        limitingUser = ancestor;
      }
    }
    // Sink only. This preserves operand dominance and execution multiplicity.
    // Producers assigned to the same stage will retain their original order.
    if (!op->isBeforeInBlock(stages[latest].begin)) {
      auto diag = op->emitError("no legal warp-pipeline stage for input");
      if (limitingUser)
        diag.attachNote(limitingUser->getLoc())
            << "this use requires a stage before the input's original position";
      return failure();
    }
    placement[op] = latest;
  }

  SmallVector<Operation *> anchors;
  for (Stage stage : stages)
    anchors.push_back(stage.begin);
  for (Operation *op : inputs) {
    Operation *&anchor = anchors[placement.lookup(op)];
    op->moveAfter(anchor);
    anchor = op;
  }
  return success();
}

struct TritonAMDGPUNormalizeWarpPipelinePass
    : impl::TritonAMDGPUNormalizeWarpPipelineBase<
          TritonAMDGPUNormalizeWarpPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([](Block *block) -> WalkResult {
      if (block->empty())
        return WalkResult::advance();
      Operation *begin = &block->front();
      bool hasOwner = false;
      for (Operation &op : *block) {
        bool isBegin = op.hasAttr(kOwnerBeginAttr);
        if (!isBegin && !op.hasAttr(kOwnerEndAttr))
          continue;
        if (failed(normalizeStageInputs(begin, &op, hasOwner)))
          return WalkResult::interrupt();
        begin = op.getNextNode();
        hasOwner = isBegin;
      }
      return failed(normalizeStageInputs(begin, nullptr, hasOwner))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

// Construct only explicit stages. An unassigned operation in a loop or an
// owner scope means normalization did not establish the placement contract.
static PipelineResult createStages(OpBuilder &b, Operation *begin,
                                   Operation *end, bool isFlat, bool bounded) {
  SmallVector<Stage> stages;
  if (failed(collectStages(begin, end, stages)))
    return PipelineResult::Malformed;
  if (stages.empty())
    return PipelineResult::NotApplicable;
  if (stages.size() < 2) {
    stages.front().begin->emitError(
        "warp_pipeline_stage requires at least two stages");
    return PipelineResult::Malformed;
  }
  if (!bounded) {
    begin = stages.front().begin;
    end = stages.back().end->getNextNode();
  }
  unsigned index = 0;
  for (Operation *op = begin; op && op != end;) {
    if (index < stages.size() && op == stages[index].begin) {
      op = stages[index++].end->getNextNode();
      continue;
    }
    if (!canSitBetweenStages(op) && !op->hasTrait<OpTrait::IsTerminator>()) {
      op->emitError(
          "unassigned operation outside warp_pipeline_stage; "
          "run warp-pipeline normalization before stage construction");
      return PipelineResult::Malformed;
    }
    op = op->getNextNode();
  }
  for (Stage stage : stages) {
    SmallVector<Operation *> ops;
    for (Operation *op = stage.begin->getNextNode(); op != stage.end;
         op = op->getNextNode())
      ops.push_back(op);
    addDummyOpIfEmptyCluster(b, stage.begin->getLoc(), stage.end, ops);
    createClusterOp(b, stage.begin->getLoc(), ops, readBorderMarker(stage.end),
                    isFlat);
    stage.begin->erase();
    stage.end->erase();
  }
  return PipelineResult::Created;
}

static PipelineResult createPipeline(OpBuilder &b, scf::ForOp forOp) {
  PipelineResult result = createStages(b, &forOp.getBody()->front(), nullptr,
                                       /*isFlat=*/false, /*bounded=*/true);
  if (result == PipelineResult::Created)
    forOp->setAttr("triton.warp_pipeline.pipelined_for", b.getUnitAttr());
  return result;
}

static PipelineResult createFlatPipeline(OpBuilder &b, Block &block,
                                         Operation *scopeBegin = nullptr,
                                         Operation *scopeEnd = nullptr) {
  return createStages(
      b, scopeBegin ? scopeBegin->getNextNode() : &block.front(), scopeEnd,
      /*isFlat=*/true, /*bounded=*/scopeBegin != nullptr);
}

// Preserve the identity of the loop that directly owns stage borders before
// canonicalization can fold or fully unroll it.  The boundary markers live in
// the parent block, so loop-body splicing leaves them around exactly the code
// that used to be the owner loop.
struct TritonAMDGPUPreserveWarpPipelineOwnersPass
    : impl::TritonAMDGPUPreserveWarpPipelineOwnersBase<
          TritonAMDGPUPreserveWarpPipelineOwnersPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<scf::ForOp> owners;
    module.walk([&](scf::ForOp forOp) {
      if (llvm::any_of(*forOp.getBody(),
                       [](Operation &op) { return isPipelineBorder(&op); }))
        owners.push_back(forOp);
    });

    OpBuilder builder(module);
    int64_t nextOwnerId = 0;
    for (scf::ForOp owner : owners) {
      SmallVector<Operation *> borders;
      for (Operation &op : *owner.getBody())
        if (isPipelineBorder(&op))
          borders.push_back(&op);

      if (std::optional<APInt> tripCount = owner.getStaticTripCount();
          tripCount && (tripCount->isZero() || tripCount->isOne())) {
        // A zero/single-trip loop cannot overlap different loop iterations.
        // Keep the kernel as an autotuning candidate, but make its warp-stage
        // annotations a no-op instead of phase-shifting an enclosing loop.
        for (Operation &op : llvm::make_early_inc_range(*owner.getBody()))
          if (isPipelineBorder(&op) || op.hasAttr(kStageBeginAttr))
            op.erase();
        continue;
      }

      int64_t ownerId = nextOwnerId++;
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(owner);
      Operation *begin =
          createOwnerMarker(builder, owner.getLoc(), kOwnerBeginAttr, ownerId);
      begin->setAttr(kOwnerStageCountAttr,
                     builder.getI64IntegerAttr(borders.size()));
      builder.setInsertionPointAfter(owner);
      createOwnerMarker(builder, owner.getLoc(), kOwnerEndAttr, ownerId);
    }
  }
};

struct OwnerScope {
  Operation *begin;
  Operation *end;
};

// Consume owner markers before scanning scf.for ops.  If the owner loop still
// exists, its stage borders remain nested and the normal loop path handles it.
// If the loop was flattened, direct borders appear between the markers and are
// reconstructed as a bounded flat pipeline.  In neither case can an enclosing
// persistent loop accidentally become the pipeline owner.
static LogicalResult recoverFlattenedOwnerScopes(OpBuilder &builder,
                                                 tt::FuncOp funcOp) {
  SmallVector<Block *> blocks;
  funcOp.walk([&](Block *block) { blocks.push_back(block); });

  for (Block *block : blocks) {
    SmallVector<OwnerScope> scopes;
    Operation *activeBegin = nullptr;
    int64_t activeId = -1;

    for (Operation &op : *block) {
      if (auto beginId = getOwnerMarkerId(&op, kOwnerBeginAttr)) {
        if (activeBegin)
          return op.emitError("nested warp-pipeline owner scopes are not "
                              "supported");
        activeBegin = &op;
        activeId = *beginId;
        continue;
      }
      if (auto endId = getOwnerMarkerId(&op, kOwnerEndAttr)) {
        if (!activeBegin || *endId != activeId)
          return op.emitError("unmatched warp-pipeline owner end marker");
        scopes.push_back({activeBegin, &op});
        activeBegin = nullptr;
        activeId = -1;
      }
    }
    if (activeBegin)
      return activeBegin->emitError(
          "unmatched warp-pipeline owner begin marker");

    for (OwnerScope scope : scopes) {
      SmallVector<Operation *> directBorders;
      for (Operation *op = scope.begin->getNextNode(); op && op != scope.end;
           op = op->getNextNode())
        if (isPipelineBorder(op))
          directBorders.push_back(op);

      auto originalStageCount =
          scope.begin->getAttrOfType<IntegerAttr>(kOwnerStageCountAttr);
      if (!directBorders.empty() && originalStageCount &&
          static_cast<int64_t>(directBorders.size()) ==
              originalStageCount.getInt()) {
        // Canonicalization flattened exactly one copy of the owner body. It
        // cannot overlap loop iterations, so retain the operations but erase
        // the scheduling-only stage borders.
        for (Operation *op = scope.begin->getNextNode(); op != scope.end;) {
          Operation *next = op->getNextNode();
          if (isPipelineBorder(op) || op->hasAttr(kStageBeginAttr))
            op->erase();
          op = next;
        }
        scope.begin->erase();
        scope.end->erase();
        continue;
      }

      PipelineResult result =
          createFlatPipeline(builder, *block, scope.begin, scope.end);
      scope.begin->erase();
      scope.end->erase();
      if (result == PipelineResult::Malformed)
        return failure();
    }
  }
  return success();
}

struct TritonAMDGPUWarpPipelinePass
    : impl::TritonAMDGPUWarpPipelineBase<TritonAMDGPUWarpPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m);
    bool malformed = false;
    for (auto funcOp : m.getOps<tt::FuncOp>()) {
      if (failed(recoverFlattenedOwnerScopes(builder, funcOp))) {
        malformed = true;
        continue;
      }
      funcOp.walk([&](scf::ForOp forOp) {
        switch (createPipeline(builder, forOp)) {
        case PipelineResult::NotApplicable:
          LDBG("scf.for has no warp_pipeline_stage borders; skipping");
          break;
        case PipelineResult::Created:
          break;
        case PipelineResult::Malformed:
          malformed = true;
          break;
        }
      });

      // Process remaining border markers in flat (non-loop) code.  Only the
      // function's top-level blocks are visited; borders inside nested
      // non-loop regions (e.g. scf.if bodies) are not handled here.
      for (Block &block : funcOp.getBody()) {
        switch (createFlatPipeline(builder, block)) {
        case PipelineResult::NotApplicable:
          break;
        case PipelineResult::Created:
          break;
        case PipelineResult::Malformed:
          malformed = true;
          break;
        }
      }
    }
    if (malformed)
      signalPassFailure();
  }
};

} // namespace mlir
