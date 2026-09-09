#include "mlir/Dialect/Arith/IR/Arith.h"
#include "nvidia/hopper/include/Transforms/WSBarrierReorder.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-nvidia-unify-ws-barrier-locations"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUUNIFYWSBARRIERLOCATIONSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

bool isAllowedRegionOp(Operation *op) {
  return isa<ttg::LocalLoadOp, TMEMLoadOp, ttg::ConvertLayoutOp,
             tt::ExpandDimsOp, tt::BroadcastOp, arith::ExtFOp, arith::TruncFOp>(
      op);
}

bool isBarrierBookkeepingOp(Operation *op) {
  if (isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp,
          ttg::MemDescReinterpretOp, ttg::MemDescTransOp,
          ttg::MemDescReshapeOp>(op))
    return true;
  if (!isa<arith::ConstantOp, arith::ExtUIOp, arith::TruncIOp, arith::XOrIOp,
           arith::AndIOp>(op) ||
      op->getNumResults() != 1)
    return false;
  return isa<IntegerType, IndexType>(op->getResult(0).getType());
}

// Both wait unification and operand ordering trade overlap for register
// relief. Keep one size policy for both transformations.
constexpr unsigned kMinRegisterHeavyBroadcastElemsPerThread = 32;

bool isRegisterHeavyBroadcast(Operation *op) {
  auto broadcast = dyn_cast<tt::BroadcastOp>(op);
  if (!broadcast)
    return false;
  auto type = dyn_cast<RankedTensorType>(broadcast.getResult().getType());
  if (!type || !type.getEncoding())
    return false;
  return ttg::getTotalElemsPerThread(type) >=
         kMinRegisterHeavyBroadcastElemsPerThread;
}

bool canUnifyWaitLocations(WaitBarrierOp earlier, WaitBarrierOp later) {
  if (earlier->getBlock() != later->getBlock() ||
      !earlier->isBeforeInBlock(later))
    return false;

  Operation *insertPt = earlier->getNextNode();
  if (!insertPt || wouldBreakOperandDominance(later, insertPt))
    return false;

  bool containsRegisterHeavyBroadcast = false;
  for (Operation *op = insertPt; op != later.getOperation();
       op = op->getNextNode()) {
    if (!op || !canRaiseWSWaitPast(later, op))
      return false;

    if (isBarrierLikeOp(op))
      continue;
    if (!isAllowedRegionOp(op) && !isBarrierBookkeepingOp(op))
      return false;
    containsRegisterHeavyBroadcast |= isRegisterHeavyBroadcast(op);
  }
  return containsRegisterHeavyBroadcast;
}

bool unifyOneWaitPair(Block &block,
                      DenseMap<Operation *, Operation *> &unifiedLocalLoads) {
  SmallVector<WaitBarrierOp> waits;
  for (Operation &op : block) {
    auto wait = dyn_cast<WaitBarrierOp>(&op);
    if (wait && hasWSBarrierConstraints(wait.getConstraints()))
      waits.push_back(wait);
  }

  for (unsigned i = 0; i + 1 < waits.size(); ++i) {
    WaitBarrierOp earlier = waits[i];
    WaitBarrierOp later = waits[i + 1];
    if (!canUnifyWaitLocations(earlier, later))
      continue;
    for (Operation *op = earlier->getNextNode(); op != later.getOperation();
         op = op->getNextNode())
      if (isa<ttg::LocalLoadOp>(op))
        unifiedLocalLoads.try_emplace(op, earlier);
    LLVM_DEBUG(llvm::dbgs() << "unifying adjacent WS wait regions\n");
    later->moveAfter(earlier);
    return true;
  }
  return false;
}

bool unifyBarrierLocations(
    Block &block, DenseMap<Operation *, Operation *> &unifiedLocalLoads) {
  bool changed = false;
  while (unifyOneWaitPair(block, unifiedLocalLoads))
    changed = true;
  return changed;
}

bool isCheapSMEMOperandPreparation(Operation *op) {
  return isa<tt::ExpandDimsOp, tt::BroadcastOp, ttg::ConvertLayoutOp,
             arith::ExtFOp, arith::TruncFOp>(op);
}

bool isProfitableSMEMOperandChain(ArrayRef<Operation *> reverseChain) {
  return llvm::all_of(reverseChain, isCheapSMEMOperandPreparation) &&
         llvm::any_of(reverseChain, isRegisterHeavyBroadcast);
}

// Put an indivisible TMEM operand before a streamable SMEM broadcast operand.
// When wait unification is enabled, keep both waits at their unified location
// and move only the SMEM load/release/preparation chain. This changes register
// materialization order without moving the TMEM acquire earlier. When wait
// unification is disabled, move the complete SMEM channel as before.
bool prioritizeTMemOperand(
    Block &block, const DenseMap<Operation *, Operation *> &unifiedLocalLoads,
    bool requireUnifiedLoad) {
  bool changed = false;
  SmallVector<TMEMLoadOp> tmemLoads;
  for (Operation &op : block)
    if (auto load = dyn_cast<TMEMLoadOp>(&op))
      tmemLoads.push_back(load);

  for (TMEMLoadOp tmemLoad : tmemLoads) {
    Value tmemValue = tmemLoad.getResult();
    Operation *commonUser = nullptr;
    while (tmemValue.hasOneUse()) {
      Operation *user = *tmemValue.user_begin();
      if (!isPure(user))
        break;
      if (user->getNumOperands() != 1 || user->getNumResults() != 1) {
        commonUser = user;
        break;
      }
      tmemValue = user->getResult(0);
    }
    if (!commonUser || !isPure(commonUser) || commonUser->getBlock() != &block)
      continue;

    for (Value operand : commonUser->getOperands()) {
      if (operand == tmemValue)
        continue;

      SmallVector<Operation *> reverseChain;
      Value current = operand;
      ttg::LocalLoadOp localLoad;
      while (Operation *def = current.getDefiningOp()) {
        if (def->getBlock() != &block || !def->hasOneUse())
          break;
        if (auto load = dyn_cast<ttg::LocalLoadOp>(def)) {
          localLoad = load;
          break;
        }
        if (!isPure(def) || def->getNumOperands() != 1 ||
            def->getNumResults() != 1)
          break;
        reverseChain.push_back(def);
        current = def->getOperand(0);
      }
      if (!localLoad || !localLoad->isBeforeInBlock(tmemLoad) ||
          !isProfitableSMEMOperandChain(reverseChain))
        continue;

      WaitBarrierOp acquire;
      bool moveAcquire = false;
      if (auto it = unifiedLocalLoads.find(localLoad);
          it != unifiedLocalLoads.end()) {
        acquire = dyn_cast<WaitBarrierOp>(it->second);
      } else if (!requireUnifiedLoad) {
        acquire = dyn_cast_or_null<WaitBarrierOp>(localLoad->getPrevNode());
        moveAcquire = true;
      }
      if (!acquire || !hasWSBarrierConstraints(acquire.getConstraints()))
        continue;

      SmallVector<Operation *> acquirePrefix;
      if (moveAcquire) {
        llvm::SmallPtrSet<Operation *, 8> movingOps{acquire, localLoad};
        for (Operation *op = acquire->getPrevNode(); op && isPure(op);
             op = op->getPrevNode()) {
          bool usedOnlyByMovingOps =
              llvm::all_of(op->getUsers(), [&](Operation *user) {
                return movingOps.contains(user);
              });
          if (!usedOnlyByMovingOps)
            break;
          acquirePrefix.push_back(op);
          movingOps.insert(op);
        }
      }

      SmallVector<Operation *> releasePrefix;
      Operation *releaseCandidate = localLoad->getNextNode();
      while (releaseCandidate && isPure(releaseCandidate)) {
        releasePrefix.push_back(releaseCandidate);
        releaseCandidate = releaseCandidate->getNextNode();
      }
      auto release = dyn_cast_or_null<ArriveBarrierOp>(releaseCandidate);
      if (!release || !hasWSBarrierConstraints(release.getConstraints()))
        continue;

      DictionaryAttr acquireWS =
          getWSBarrierConstraints(acquire.getConstraints());
      DictionaryAttr releaseWS =
          getWSBarrierConstraints(release.getConstraints());
      if (!hasOrderedWSBarrierInfo(acquireWS) ||
          !hasOrderedWSBarrierInfo(releaseWS) ||
          acquireWS.getAs<IntegerAttr>("parentId").getInt() !=
              releaseWS.getAs<IntegerAttr>("parentId").getInt())
        continue;

      bool safe = true;
      for (Operation *op = release->getNextNode(); op && op != commonUser;
           op = op->getNextNode()) {
        if (auto arrive = dyn_cast<ArriveBarrierOp>(op)) {
          if (!hasWSBarrierConstraints(arrive.getConstraints()) ||
              !canAdvanceWSBarrierArrivePastWait(arrive.getConstraints(),
                                                 acquire.getConstraints())) {
            safe = false;
            break;
          }
          continue;
        }
        if (auto wait = dyn_cast<WaitBarrierOp>(op)) {
          if (!hasWSBarrierConstraints(wait.getConstraints()) ||
              !canAdvanceWSBarrierArrivePastWait(release.getConstraints(),
                                                 wait.getConstraints())) {
            safe = false;
            break;
          }
          continue;
        }
        if (!canAdvanceWSBarrier(release.getConstraints(), op)) {
          safe = false;
          break;
        }
      }
      if (!safe)
        continue;

      if (moveAcquire) {
        for (Operation *op : llvm::reverse(acquirePrefix))
          op->moveBefore(commonUser);
        acquire->moveBefore(commonUser);
      }
      localLoad->moveBefore(commonUser);
      for (Operation *op : releasePrefix)
        op->moveBefore(commonUser);
      release->moveBefore(commonUser);
      for (Operation *op : llvm::reverse(reverseChain))
        op->moveBefore(commonUser);
      changed = true;
      break;
    }
  }
  return changed;
}

} // namespace

struct TritonNvidiaGPUUnifyWSBarrierLocationsPass
    : public impl::TritonNvidiaGPUUnifyWSBarrierLocationsPassBase<
          TritonNvidiaGPUUnifyWSBarrierLocationsPass> {
  using impl::TritonNvidiaGPUUnifyWSBarrierLocationsPassBase<
      TritonNvidiaGPUUnifyWSBarrierLocationsPass>::
      TritonNvidiaGPUUnifyWSBarrierLocationsPassBase;

  void runOnOperation() override {
    bool unifyWaits =
        !triton::tools::getBoolEnv("TRITON_DISABLE_WSBARRIER_REORDER");

    getOperation().walk([&](Block *block) {
      if (block->empty())
        return;
      DenseMap<Operation *, Operation *> unifiedLocalLoads;
      if (unifyWaits)
        unifyBarrierLocations(*block, unifiedLocalLoads);
      prioritizeTMemOperand(*block, unifiedLocalLoads,
                            /*requireUnifiedLoad=*/unifyWaits);
    });
  }
};

} // namespace mlir::triton::nvidia_gpu
