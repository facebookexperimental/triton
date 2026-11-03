#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"

#include "Allocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/NamedBarrier.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONNVIDIAGPUMEMBAR
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace mlir::triton {
namespace {

namespace ttng = mlir::triton::nvidia_gpu;

struct TritonNvidiaGPUMembar
    : public impl::TritonNvidiaGPUMembarBase<TritonNvidiaGPUMembar> {
  using TritonNvidiaGPUMembarBase::TritonNvidiaGPUMembarBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);

    // Assign warp-specialization barrier IDs before shared-memory allocation.
    ttng::NamedBarrierIdAllocator barrierAllocator(mod);
    if (failed(ttng::ensureWarpSpecializeBarrierIds(mod, barrierAllocator))) {
      signalPassFailure();
      return;
    }

    // Hoisting must precede allocation so liveness sees the entry prologue.
    SetVector<Operation *> coveredInitOps;
    if (failed(maybeInsertClusterSync(mod, coveredInitOps))) {
      signalPassFailure();
      return;
    }

    // Single-WS lowering handles exactly one warp_specialize region, but
    // lowering adds a synthetic region for each cluster sync that worker
    // warps must join. Those regions need the multi-region dispatch loop,
    // whose shared-memory state allocation must reserve. The cluster barrier
    // insertion passes below run after allocation and add syncs only on SM90+
    // in logical multi-CTA modules, so conservatively drop single-WS for
    // those.
    bool droppedSingleWS = false;
    if (triton::gpu::hasSingleWarpSpecialize(mod) &&
        (NVIDIA::needsSyntheticWarpSpecialize(mod) ||
         (computeCapability >= 90 &&
          triton::gpu::TritonGPUDialect::getNumCTAs(mod) > 1))) {
      mod->removeAttr(triton::gpu::AttrSingleWarpSpecializeName);
      droppedSingleWS = true;
    }

    ModuleAllocation allocation(
        mod, ttng::getNvidiaAllocationAnalysisScratchSizeFn(targetInfo));
    if (!coveredInitOps.empty() || droppedSingleWS)
      triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);

    ttng::runClusterBarrierInsertion(allocation, computeCapability);
    if (failed(ttng::runCrossCTAMBarrierInitSyncInsertion(
            allocation, computeCapability, coveredInitOps.getArrayRef()))) {
      signalPassFailure();
      return;
    }

    ModuleMembarAnalysis membarPass(allocation, NVIDIA::canSkipBarSync);
    membarPass.run();
  }

private:
  static bool isMovableInitDependency(Operation *op) {
    if (auto alloc = dyn_cast<triton::gpu::LocalAllocOp>(op))
      return !alloc.getSrc();
    return op->getNumRegions() == 0 && isPure(op);
  }

  static bool isEntryInitPrologueOp(Operation *op) {
    if (isa<ttng::InitBarrierOp, ttng::TCGen5GlobalAllocOp>(op))
      return true;
    if (auto alloc = dyn_cast<ttng::TMEMAllocOp>(op))
      return !alloc.getSrc();
    return isMovableInitDependency(op);
  }

  LogicalResult validateExplicitEntryBarInit(ttng::InitBarrierOp init) {
    Block *entry = init->getBlock();
    SetVector<Operation *> dependencies;
    SmallVector<Value> worklist(init->getOperands());
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        if (arg.getOwner() != entry)
          return init.emitOpError("explicit cluster initialization has a "
                                  "non-entry dependency");
        continue;
      }
      Operation *def = value.getDefiningOp();
      if (!dependencies.insert(def))
        continue;
      if (def->getBlock() != entry || !isMovableInitDependency(def))
        return init.emitOpError("explicit cluster initialization cannot safely "
                                "hoist an effectful or non-entry dependency");
      llvm::append_range(worklist, def->getOperands());
    }
    for (Operation &op : *entry) {
      if (&op == init.getOperation())
        break;
      if (!isEntryInitPrologueOp(&op))
        return init.emitOpError(
            "explicit cluster initialization must precede "
            "effectful execution; later-phase initialization "
            "is unsupported");
      if (auto previous = dyn_cast<ttng::InitBarrierOp>(op);
          previous && previous.getAlloc() == init.getAlloc())
        return init.emitOpError("explicit cluster barrier reinitialization "
                                "is unsupported");
    }
    return success();
  }

  // Push entry-block mbarrier init ops early enough to preserve the
  // mbarrier-init -> cluster-sync -> TMEM-alloc sequence. Barrier init ops in
  // WS partition bodies are local pipeline barriers; they do not participate in
  // this kernel-entry cluster synchronization point.
  LogicalResult ensureEarlyEntryBarInit(ModuleOp &mod,
                                        SetVector<Operation *> &barInitOps,
                                        bool validateMotion) {
    FunctionOpInterface funcOp = nullptr;
    mod.walk([&](FunctionOpInterface op) {
      if (triton::isKernel(op) && !op.isExternal()) {
        funcOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    assert(funcOp && "Expecting to find a kernel func but got none.");
    Block *entryBlock = &funcOp.getFunctionBody().front();

    // Move all mbar init ops (and their deps) to the beginning of the block
    SetVector<Operation *> opsToMove;
    SmallVector<Operation *> worklist(barInitOps.begin(), barInitOps.end());
    while (!worklist.empty()) {
      auto *op = worklist.pop_back_val();
      if (!opsToMove.insert(op))
        continue;
      for (Value operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (validateMotion && (defOp->getBlock() != entryBlock ||
                                 !isMovableInitDependency(defOp)))
            return op->emitOpError("cluster initialization cannot safely hoist "
                                   "an effectful or non-entry dependency");
          if (defOp->getBlock() == entryBlock)
            worklist.push_back(defOp);
        }
      }
    }
    SmallVector<Operation *> opsInBlockOrder;
    for (auto &op : *entryBlock) {
      if (opsToMove.contains(&op))
        opsInBlockOrder.push_back(&op);
    }
    Operation *insertPt = nullptr;
    for (auto *op : opsInBlockOrder) {
      if (!insertPt)
        op->moveBefore(entryBlock, entryBlock->begin());
      else
        op->moveAfter(insertPt);
      insertPt = op;
    }

    // Check the block again to make sure all entry-block mbar init ops are
    // earlier than the first entry-block TMEM allocation op.
    Operation *firstTMEMAlloc = nullptr;
    for (auto &op : *entryBlock) {
      if (isa<ttng::TCGen5GlobalAllocOp, ttng::TMEMAllocOp>(&op)) {
        firstTMEMAlloc = &op;
        break;
      }
    }
    if (firstTMEMAlloc) {
      for (auto *op : barInitOps) {
        if (!op->isBeforeInBlock(firstTMEMAlloc)) {
          op->emitError() << "Barrier init is not before TMEM allocation. "
                             "Cannot insert cluster sync between them.";
          return failure();
        }
      }
    }

    return success();
  }

  // Return the operand or result Value of a given op if the Value is used for
  // cross CTA mbarrier arrival. This function assumes the kernel has cluster
  // size larger than 1.
  std::optional<SetVector<Value>> getRemoteBarrier(Operation *op) {
    if (auto mapaOp = llvm::dyn_cast<ttng::MapToRemoteBufferOp>(op)) {
      // plain cross CTA mbarrier arrive and cross CTA DSMEM store/copy need
      // mapa to map mbarrier addr explicitly
      llvm::SetVector<Value> bars;
      bars.insert(mapaOp.getResult());
      return bars;
    } else if (auto tmaLoadOp =
                   llvm::dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      // If it's a TMA load with multicast, the mbar signal is multicasted too
      if (tmaLoadOp.getMulticastTargets()) {
        llvm::SetVector<Value> bars;
        bars.insert(tmaLoadOp.getBarrier());
        return bars;
      }
    } else if (auto asyncCLCTryCancelOp =
                   llvm::dyn_cast<ttng::AsyncCLCTryCancelOp>(op)) {
      // If it's AsyncCLCTryCancelOp, the signal will be broadcasted to other
      // CTAs only when .multicast::cluster::all is specified, which is true now
      // no matter what cluster size is. Since we're assuming cluster size > 1,
      // we should consider the barrier here as remote barrier.
      llvm::SetVector<Value> bars;
      bars.insert(asyncCLCTryCancelOp.getMbarAlloc());
      return bars;
    } else if (auto clcTryCancelOp = llvm::dyn_cast<ttng::CLCTryCancelOp>(op)) {
      // The core Triton CLC scheduler also multicasts its response and
      // completion signal when an explicit physical cluster is configured.
      // All CTAs must finish initializing their local completion barriers
      // before the lead CTA can issue the request.
      llvm::SetVector<Value> bars;
      bars.insert(clcTryCancelOp.getMbarrier());
      return bars;
    } else if (auto tcgen5CommitOp = llvm::dyn_cast<ttng::TCGen5CommitOp>(op)) {
      // As of now, there're only three sources to have a tcgen05.commit
      // instruction:
      // 1. Front end supplied a TCGen5CommitOp directly
      // 2. When lowering gen5 TMEMCopy to llvm, compiler inserts inline ptx
      // 3. When lowering gen5 MMA to llvm, compiler inserts inline ptx
      // And the eventual tcgen05.commit has .multicast::cluster to broadcast
      // mbar signals to multiple CTAs only under 2cta mode.
      // https://github.com/facebookexperimental/triton/blob/70d488dc45ca7e75432b0352cb9dd07b602a82cf/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp#L327
      // Although it's valid
      // to have .multicast::cluster for 1cta mode too, there's currently no
      // support for it.

      // Cases 1 and 2 will read module attribute for 2cta mode, case 3 will
      // read module attr or op arg for 2cta mode, which are equivalent since
      // all tcgen05 ops have to be consistent with module attr on this.

      // Case 1: explicit TCGen5CommitOp from front end or earlier passes
      if (!tcgen5CommitOp.getDescs().empty()) {
        llvm::SetVector<Value> bars;
        bars.insert(tcgen5CommitOp.getBarrier());
        return bars;
      }
    } else if (llvm::isa<ttng::MMAv5OpInterface>(op)) {
      // case 3 for gen5 commit: a commit inline ptx will be generated for each
      // barrier on the gen5 MMA op. If the mod is in 2cta mode, the commit op
      // can multicast bar signals.
      if (tlx::tlxEnablePairedMMA(op)) {
        llvm::SetVector<Value> bars;
        // TODO: move getBarriers() into MMAv5OpInterface to simplify this
        if (auto mma = llvm::dyn_cast<ttng::TCGen5MMAOp>(op)) {
          for (auto bar : mma.getBarriers()) {
            bars.insert(bar);
          }
        } else {
          // "assert" it's a scaled MMA op so that we crash explicitly if new
          // MMAv5OpInterface is added
          auto scaledMMA = llvm::cast<ttng::TCGen5MMAScaledOp>(op);
          for (auto bar : scaledMMA.getBarriers()) {
            bars.insert(bar);
          }
        }
        return bars;
      }
    }

    return std::nullopt;
  }

  // If the kernel is clustered, insert cluster sync properly to
  // bootstrap remote bars or tmem
  LogicalResult maybeInsertClusterSync(ModuleOp &mod,
                                       SetVector<Operation *> &coveredInitOps) {
    SetVector<Operation *> explicitInitOps;
    LogicalResult status = success();
    mod.walk([&](ttng::InitBarrierOp init) {
      if (!init.getBCluster())
        return;
      explicitInitOps.insert(init);
      auto func = init->getParentOfType<triton::FuncOp>();
      if (computeCapability < 90) {
        init.emitOpError(
            "explicit cluster initialization requires NVIDIA SM90+");
        status = failure();
      } else if (!func || !triton::isKernel(func) ||
                 init->getBlock() != &func.front()) {
        init.emitOpError(
            "explicit cluster initialization requires unconditional "
            "kernel-entry placement after inlining");
        status = failure();
      } else if (failed(validateExplicitEntryBarInit(init))) {
        status = failure();
      }
    });
    if (failed(status))
      return failure();
    if (triton::gpu::lookupPhysicalNumCTAs(mod) <= 1) {
      return success();
    }

    // Explicit cluster scope owns the bootstrap independently of consumers.
    // Only default-scope kernels need the legacy consumer-based inference.
    if (explicitInitOps.empty()) {
      bool hasRemoteBar = false;
      bool hasMulticastArrive = false;
      mod.walk([&](Operation *op) {
        if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op);
            arrive && arrive.isMulticast()) {
          hasMulticastArrive = true;
          return WalkResult::interrupt();
        }
        if (getRemoteBarrier(op).has_value()) {
          hasRemoteBar = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!hasRemoteBar && !hasMulticastArrive && !tlx::tlxEnablePairedMMA(mod))
        return success();
    }

    // Find the kernel entry block and collect entry-block barrier inits.
    // Only entry-block barriers need the cluster fence+sync for cross-CTA
    // visibility. Barriers inside WS partition bodies are local pipeline
    // barriers that don't participate in cross-CTA communication.
    // Legacy lowering tests can already contain LLVM kernels.
    FunctionOpInterface funcOp = nullptr;
    mod.walk([&](FunctionOpInterface op) {
      if (triton::isKernel(op) && !op.isExternal()) {
        funcOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    assert(funcOp && "Expecting to find a kernel func.");
    Block *entryBlock = &funcOp.getFunctionBody().front();

    SetVector<Operation *> entryBarInitOps;
    for (auto &op : *entryBlock) {
      // Do not pull later local initialization into an explicit bootstrap.
      // Legacy callers retain their existing entry-block selection behavior.
      if (!explicitInitOps.empty() && !isEntryInitPrologueOp(&op))
        break;
      if (isa<ttng::InitBarrierOp>(op))
        entryBarInitOps.insert(&op);
    }

    for (Operation *init : explicitInitOps) {
      if (!entryBarInitOps.contains(init))
        return init->emitOpError("explicit cluster initialization requires the "
                                 "designated kernel-entry bootstrap owner");
    }
    if (entryBarInitOps.empty())
      return success();

    if (failed(ensureEarlyEntryBarInit(mod, entryBarInitOps,
                                       !explicitInitOps.empty())))
      return failure();

    ttng::InitBarrierOp lastBarInitOp;
    for (auto it = entryBlock->rbegin(), e = entryBlock->rend(); it != e;
         ++it) {
      if (entryBarInitOps.contains(&*it)) {
        lastBarInitOp = cast<ttng::InitBarrierOp>(*it);
        break;
      }
    }

    OpBuilder builder(lastBarInitOp);
    builder.setInsertionPointAfter(lastBarInitOp);
    // need to insert fence to make mbar init visible to cluster
    auto fence = ttng::FenceMBarrierInitReleaseClusterOp::create(
        builder, lastBarInitOp.getLoc());
    tlx::setClusterSyncBootstrap(fence);
    // need to insert cluster arrive and wait to prevent CTA_X from arriving
    // CTA_Y's bar before CTA_Y inits it, as shown in ptx doc examples:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
    auto arrive = ttng::ClusterArriveOp::create(builder, lastBarInitOp.getLoc(),
                                                /*relaxed=*/true);
    auto wait = ttng::ClusterWaitOp::create(builder, lastBarInitOp.getLoc());
    tlx::setClusterSyncBootstrap(arrive);
    tlx::setClusterSyncBootstrap(wait);
    coveredInitOps.insert(entryBarInitOps.begin(), entryBarInitOps.end());
    // mark mod attr so that WS lowering is aware of this cluster sync point
    tlx::setClusterSyncKernelInitOnMod(mod, true);
    return success();
  }
};

} // namespace

bool NVIDIA::canSkipBarSync(Operation *before, Operation *after,
                            bool /*beforeIsRead*/, bool /*afterIsRead*/,
                            Allocation * /*allocation*/) {
  // These mbarrier ops are single threaded, so are always synchronized wrt.
  // each other.
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::BarrierExpectOp>(
          before) &&
      isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::BarrierExpectOp>(
          after))
    return true;

  // wait_barrier will never run ahead of the load it's waiting on
  if (isa<ttng::TMALoadLikeOpInterface>(before) &&
      isa<ttng::WaitBarrierOp>(after))
    return true;

  // Identical same-width commutative atomics can be freely reordered.
  auto beforeAtomic = dyn_cast<triton::gpu::LocalAtomicScatterRMWOp>(before);
  auto afterAtomic = dyn_cast<triton::gpu::LocalAtomicScatterRMWOp>(after);
  return beforeAtomic && afterAtomic && beforeAtomic.isCommutative() &&
         afterAtomic.isCommutative() &&
         beforeAtomic.getAtomicRmwOp() == afterAtomic.getAtomicRmwOp() &&
         beforeAtomic.getDst().getType().getElementType() ==
             afterAtomic.getDst().getType().getElementType();
}

} // namespace mlir::triton
