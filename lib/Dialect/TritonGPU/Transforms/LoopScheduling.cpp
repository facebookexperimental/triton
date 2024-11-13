#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#define DEBUG_TYPE "triton-loop-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

// Begin __FACEBOOK__ CompPipe
static void scheduleLoads(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                          DenseSet<Operation *> &rootUsers, int numStages) {
  // Get all loads that are (transitively) used by dot ops and their distance
  // to the dot op.
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse =
          mlir::triton::loadOpsToIndirectionLevelAndUse(forOp);
  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
    for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
      LDBG("  - load: " << *l);
      LDBG("    at indirection level: " << i);
      LDBG("    used by op: " << *u);
    }
  });
  if (loadOpToIndLevelAndUse.empty())
    return;

  // Calculate the stage distance between applicable loads.
  int maxIndirectionLevel = -1;
  for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse) {
    maxIndirectionLevel = std::max(maxIndirectionLevel, dist);
  }
  unsigned stagesBetweenLoads =
      ceil<unsigned>(numStages - 2, maxIndirectionLevel + 1);

  tt::CoarseSchedule::Cluster rootUsersCluster = schedule.clusters.newAtFront();
  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    // Non-LoadOp(s) are the root uses of all LoadOp(s) and should be
    // always present in the opInfo
    if (!isa<tt::LoadOp>(use)) {
      rootUsers.insert(use);
      schedule.insert(use, numStages - 1, rootUsersCluster);
    }
  }

  SmallVector<tt::CoarseSchedule::Cluster> loadsClusters;
  for (int i = 0; i < maxIndirectionLevel + 1; i++) {
    loadsClusters.push_back(schedule.clusters.newAtBack());
  }
  // Assign stages to the loads.
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    schedule.insert(loadOp, stage, loadsClusters[indLevel]);
  }
}

static tt::CoarseSchedule::Cluster
schedulePrologueAndEpilogue(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                            DenseSet<Operation *> &rootUsers, int numStages) {
  // afterPrologue : first cluster curently but we will add a cluster at front
  // and a cluster at back
  tt::CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  // Look for the IfOp that is in the backward slice any of the currently
  // scheduled ops and put it at the beginning of the loop.
  DenseMap<scf::IfOp, int> ifsToStage;
  // Go stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : schedule.getOpsInOrder(forOp)) {
      if (stage_ != stage)
        continue;
      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      getBackwardSlice((Operation *)op, &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  tt::CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
  for (auto [ifOp, stage] : ifsToStage) {
    schedule.insert(ifOp, stage, prologueCluster);
  }
  // Look for the IfOp that is in the forward slice of the root users and put it
  // at the end of the loop.
  tt::CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto rootUser : rootUsers) {
    SetVector<Operation *> forwardSlice;
    getForwardSlice(rootUser, &forwardSlice);

    int stage = schedule[rootUser].first;
    for (auto op : forwardSlice) {
      scf::IfOp ifOp = dyn_cast<scf::IfOp>(op);
      if (ifOp == nullptr) {
        // check if the op is in the body of an if op that's part of the loop
        auto parentOp = op->getParentOp();
        if (parentOp != nullptr &&
            parentOp->getParentOp() == forOp.getOperation()) {
          ifOp = dyn_cast<scf::IfOp>(parentOp);
        }
      }
      if (ifOp) {
        schedule.insertIfAbsent(ifOp, stage,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

static const char *kLoopScheduleAttrName = "tt.loop_schedule";
std::string getLoopScheduleOrDefault(scf::ForOp forOp) {
  if (!forOp->hasAttr(kLoopScheduleAttrName))
    return "default";
  return (cast<StringAttr>(forOp->getAttr(kLoopScheduleAttrName))).str();
}
// End __FACEBOOK__ CompPipe

static bool isHeavyComputation(Operation *op) {
  // include exp2, mulf, addf 1D. Somehow we don't go through reduction
  // when checking dependencies
  if (!isa<arith::MulFOp>(op) && !isa<math::Exp2Op>(op) &&
      !isa<arith::AddFOp>(op))
    return false;
  auto tensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!tensorTy)
    return false;
  if (tensorTy.getRank() < 1)
    return false;
  return true;
}

// Find all consumer_waits needed for a given dot. Assume we have this sequence
// consumer_wait -> subview -> local_load -> dot
// or
// consumer_wait -> subview -> dot
// with TMA
// wait_barrier -> subview -> trans -> dot
// We assume consumer_wait and subview are right next to each other. We start
// from consumer_wait or wait_barrier, find the subview and check if the subview
// feeds into the dot.
static DenseSet<Operation *> getConsumerWaits(Operation *dot,
                                              scf::ForOp forOp) {
  llvm::SmallVector<Operation *> deps;
  DenseSet<Operation *> seen;
  // Get dependencies of the DotOp, stop when hitting Subview or another Dot
  std::function<void(Operation *, Operation *)> dfs = [&](Operation *op,
                                                          Operation *baseOp) {
    if (!seen.insert(op).second)
      return;
    if (op != baseOp &&
        op->hasTrait<OpTrait::DotLike>()) // do not go through Dots
      return;
    if (isa<triton::gpu::MemDescSubviewOp>(op)) {
      deps.push_back(op);
      return;
    }
    if (isa<ttg::LocalLoadOp>(op) || op->hasTrait<OpTrait::DotLike>())
      deps.push_back(op);

    for (Value operand : op->getOperands()) {
      Value v = operand;
      Operation *defOp = v.getDefiningOp();
      if (defOp && defOp->getBlock() == op->getBlock()) {
        dfs(defOp, baseOp);
      }
    }
  };
  dfs(dot, dot);
  DenseSet<Operation *> depSet;
  for (auto *op : deps) {
    depSet.insert(op);
  }
  // Go through loop body, check for the sequence.
  Operation *currentWait = nullptr;
  unsigned seqNum = 0;
  DenseSet<Operation *> waits;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto wait = dyn_cast<ttng::ConsumerWaitOp>(op)) {
      currentWait = &op;
      seqNum = 1;
      continue;
    }
    if (auto wait = dyn_cast<ttng::WaitBarrierOp>(op)) {
      currentWait = &op;
      seqNum = 1;
      continue;
    }
    if (currentWait && seqNum == 1) {
      if (isa<arith::ConstantOp>(op))
        continue;
      // subview must be next to wait minus some constants
      // we should try to associate a barrier with a buffer
      if (auto view = dyn_cast<triton::gpu::MemDescSubviewOp>(op)) {
        seqNum = 2;
        if (depSet.count(&op))
          waits.insert(currentWait);
      } else {
        currentWait = nullptr;
        seqNum = 0;
      }
      continue;
    }
  }
  return waits;
}

static void
getListOfProducerAcquires(scf::ForOp forOp,
                          SmallVector<Operation *> &producerAquires) {
  auto funcOp = forOp->getParentOfType<FunctionOpInterface>();
  funcOp.walk([&](scf::ForOp forOp) {
    auto taskArr = mlir::getAsyncTaskIds(forOp);
    if (taskArr.size() == 1 && taskArr[0] == 0) {
      // Producer warp group ForOp.
      forOp.walk([&](Operation *op) {
        if (isa<ttng::ProducerAcquireOp>(op))
          producerAquires.push_back(op);
      });
    }
  });
}

// FIXME: need to know the corresponding wait/release for a given load.
static Operation *
getConsumerReleaseForWait(Operation *wait, scf::ForOp forOp,
                          SmallVector<Operation *> &producerAquires,
                          bool firstLoad) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto release = dyn_cast<ttng::ConsumerReleaseOp>(op)) {
      if (isa<ttng::WaitBarrierOp>(wait)) {
        // TMA case, only match with producerAquires (1st operand).
        // For data partitioning, 4 tokens inside the loop. First 2
        // producerAcquires correspond to firstLoad (loadK). Last 2 correspond
        // to secondLoad (loadV).
        assert(producerAquires.size() == 4);
        if (release->getOperand(0) ==
            producerAquires[firstLoad ? 0 : 2]->getOperand(0))
          return release;
        if (release->getOperand(0) ==
            producerAquires[firstLoad ? 1 : 3]->getOperand(0))
          return release;
        continue;
      }
      bool isMatch = true;
      unsigned i = 0;
      for (Value operand : wait->getOperands()) {
        if (i >= release->getNumOperands())
          break;
        if (operand != release->getOperand(i)) {
          isMatch = false;
          break;
        }
        ++i;
      }
      if (isMatch)
        return release;
    }
  }
  return nullptr;
}

#define GEN_PASS_DEF_TRITONGPULOOPSCHEDULING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPULoopSchedulingPass
    : public impl::TritonGPULoopSchedulingBase<TritonGPULoopSchedulingPass> {
public:
  using impl::TritonGPULoopSchedulingBase<
      TritonGPULoopSchedulingPass>::TritonGPULoopSchedulingBase;

  // Begin __FACEBOOK__ CompPipe
  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
      return numStages;
    return mlir::cast<IntegerAttr>(
               forOp->getAttr(mlir::triton::kNumStagesAttrName))
        .getInt();
  }

  tt::CoarseSchedule::Cluster
  getDefaultLoopSchedule(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                         int numStages) {
    DenseSet<Operation *> rootUsers;
    scheduleLoads(forOp, schedule, rootUsers, numStages);
    return schedulePrologueAndEpilogue(forOp, schedule, rootUsers, numStages);
  }

  // Check for warp spec consumer group. Assume two dots.
  bool
  isFlashAttention(scf::ForOp forOp,
                   llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
                       &loadOpToIndLevelAndUse,
                   SmallVector<Operation *> &keyOps,
                   DenseSet<Operation *> &heavyCompOps) {
    SmallVector<Operation *> loads;
    SmallVector<Operation *> dots;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      // Check for loop-carried dependencies.
      // We have two loadOps, one feeding the first dot, and the other feeding
      // the second dot.
      if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op)) {
        loads.push_back(&op);
      }
      if (op.hasTrait<OpTrait::DotLike>()) {
        dots.push_back(&op);
      }
    }
    // Check for async_task_id.
    auto taskArr = mlir::getAsyncTaskIds(forOp);
    bool isConsumerWG = taskArr.size() != 1 ? false : taskArr[0] != 0;
    if (dots.size() != 2 || (loads.size() != 2 && !isConsumerWG))
      return false;

    Operation *secondDot = dots[1];
    DenseSet<Operation *> seen;
    DenseSet<Operation *> tracedDots;
    // Make sure there is a dependency path from firstDot to secondDot.
    // This means we need to do computation pipelining to break the dependency.
    std::function<void(Operation * op)> dfs = [&](Operation *op) {
      if (!seen.insert(op).second)
        return;
      for (Value operand : op->getOperands()) {
        Value v = operand;
        Operation *defOp = v.getDefiningOp();
        if (defOp && defOp->getBlock() == op->getBlock()) {
          if (defOp->hasTrait<OpTrait::DotLike>()) {
            // Stop tracing when hitting a dot.
            tracedDots.insert(defOp);
          } else {
            if (isHeavyComputation(defOp))
              heavyCompOps.insert(defOp);
            dfs(defOp);
          }
        }
      }
    };
    dfs(secondDot);
    if (tracedDots.size() != 1)
      return false;

    for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse) {
      if (dist != 0)
        return false;
    }

    keyOps.push_back(loads.size() == 0 ? nullptr : loads[0]); // FIXME
    keyOps.push_back(loads.size() == 0 ? nullptr : loads[1]);
    keyOps.push_back(dots[0]);
    keyOps.push_back(secondDot);
    return true;
  }

  tt::CoarseSchedule::Cluster
  getFAFirstDotSchedule(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                        int numStages) {
    llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
        loadOpToIndLevelAndUse =
            mlir::triton::loadOpsToIndirectionLevelAndUse(forOp);
    LLVM_DEBUG({
      LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
      for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
        LDBG("  - load: " << *l);
        LDBG("    at indirection level: " << i);
        LDBG("    used by op: " << *u);
      }
    });
    // if (loadOpToIndLevelAndUse.empty())
    //   return schedule.clusters.begin();

    // Check to see if the for loop matches the pattern for flash attention.
    // If yes, move the first dot to its own stage (numStages - 2), the
    // rest of the computation will be in stage (numStages - 1). The two loads
    // will be in stage 0 and 1.
    SmallVector<Operation *> keyOps;
    DenseSet<Operation *> heavyCompOps;
    if (!isFlashAttention(forOp, loadOpToIndLevelAndUse, keyOps,
                          heavyCompOps)) {
      LDBG("isFlashAttention returns false");
      return schedule.clusters.begin();
    }

    // firstLoad: keyOps[0]
    tt::CoarseSchedule::Cluster rootUsersCluster =
        schedule.clusters.newAtFront();
    tt::CoarseSchedule::Cluster loadCluster = schedule.clusters.newAtBack();
    bool isConsumerWG = keyOps[0] == nullptr;
    if (!isConsumerWG) {
      schedule.insert(keyOps[0], 0, loadCluster);
      schedule.insert(keyOps[1], 1, loadCluster);
    } else {
      // Check producer warp group to get the list of ProducerAcquires (assume
      // they are in order matching firstLoad and secondLoad). Then match
      // ConsumerReleases with them. With TMA, align consumerRleases with
      // consumerWaits, assuming consumerWaits happen in order matching
      // firstLoad and secondLoad.
      SmallVector<Operation *> producerAquires;
      getListOfProducerAcquires(forOp, producerAquires);
      // dependency from consumer_wait to subview, then to consumer_release
      // Assume this group of ops: consumer_wait, subview, local_load. Find the
      // corresponding consumer_release for the consumer_wait by checking the
      // operands. The local_load needed by firstDot will be in the same stage
      // cluseter as firstDot.
      DenseSet<Operation *> ConsumerWaitsForDot1 =
          getConsumerWaits(keyOps[2], forOp);
      for (auto *op : ConsumerWaitsForDot1) {
        schedule.insert(op, isConsumerWG ? 0 : numStages - 2, rootUsersCluster);
        Operation *consumerRelease =
            getConsumerReleaseForWait(op, forOp, producerAquires, true);
        schedule.insert(consumerRelease, isConsumerWG ? 0 : numStages - 2,
                        rootUsersCluster);
        LLVM_DEBUG({
          LDBG("firstDot wait ");
          op->dump();
          LDBG("firstDot release ");
          consumerRelease->dump();
        });
      }
      DenseSet<Operation *> ConsumerWaitsForDot2 =
          getConsumerWaits(keyOps[3], forOp);
      for (auto *op : ConsumerWaitsForDot2) {
        schedule.insert(op, numStages - 1, rootUsersCluster);
        Operation *consumerRelease =
            getConsumerReleaseForWait(op, forOp, producerAquires, false);
        schedule.insert(consumerRelease, numStages - 1, rootUsersCluster);
        LLVM_DEBUG({
          LDBG("secondDot wait ");
          op->dump();
          LDBG("secondDot release ");
          consumerRelease->dump();
        });
      }
    }
    schedule.insert(keyOps[2], isConsumerWG ? 0 : numStages - 2,
                    rootUsersCluster);
    schedule.insert(keyOps[3], numStages - 1, rootUsersCluster);
    return schedule.clusters.begin();
  }

  tt::CoarseSchedule::Cluster
  getFASecondDotSchedule(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                         int numStages) {
    llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
        loadOpToIndLevelAndUse =
            mlir::triton::loadOpsToIndirectionLevelAndUse(forOp);
    LLVM_DEBUG({
      LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
      for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
        LDBG("  - load: " << *l);
        LDBG("    at indirection level: " << i);
        LDBG("    used by op: " << *u);
      }
    });
    // if (loadOpToIndLevelAndUse.empty())
    //   return schedule.clusters.begin();

    // Check to see if the for loop matches the pattern for flash attention.
    // If yes, move the second dot to its own stage (numStages - 1), the
    // rest of the computation will be in stage (numStages - 2). The two loads
    // will be in stage 0 and 1.
    SmallVector<Operation *> keyOps;
    DenseSet<Operation *> heavyCompOps;
    if (!isFlashAttention(forOp, loadOpToIndLevelAndUse, keyOps,
                          heavyCompOps)) {
      LDBG("isFlashAttention returns false");
      return schedule.clusters.begin();
    }
    // Go through loop body
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (isHeavyComputation(&op))
        heavyCompOps.insert(&op);
    }
    // keyOps: load0, load1, dot0, dot1
    //   Dot0(i+1)
    //   Dot1(i)
    //   Softmax(i+1): includes MUL0(i+1)
    //   MUL1(i+1)
    tt::CoarseSchedule::Cluster rootUsersCluster =
        schedule.clusters.newAtFront();
    tt::CoarseSchedule::Cluster nextCluster = schedule.clusters.newAtBack();
    tt::CoarseSchedule::Cluster nextNextCluster = schedule.clusters.newAtBack();
    tt::CoarseSchedule::Cluster loadCluster = schedule.clusters.newAtBack();
    bool isConsumerWG = keyOps[0] == nullptr;
    if (!isConsumerWG) {
      schedule.insert(keyOps[0], 0, loadCluster);
      schedule.insert(keyOps[1], 1, loadCluster);
    } else {
      SmallVector<Operation *> producerAquires;
      getListOfProducerAcquires(forOp, producerAquires);

      DenseSet<Operation *> ConsumerWaitsForDot1 =
          getConsumerWaits(keyOps[2], forOp);
      for (auto *op : ConsumerWaitsForDot1) {
        schedule.insert(op, isConsumerWG ? 0 : numStages - 2, rootUsersCluster);
        Operation *consumerRelease =
            getConsumerReleaseForWait(op, forOp, producerAquires, true);
        assert(consumerRelease);
        schedule.insert(consumerRelease, isConsumerWG ? 0 : numStages - 2,
                        rootUsersCluster);
        LLVM_DEBUG({
          LDBG("firstDot wait ");
          op->dump();
          LDBG("firstDot release ");
          consumerRelease->dump();
        });
      }
      DenseSet<Operation *> ConsumerWaitsForDot2 =
          getConsumerWaits(keyOps[3], forOp);
      for (auto *op : ConsumerWaitsForDot2) {
        schedule.insert(op, numStages - 1, nextCluster);
        Operation *consumerRelease =
            getConsumerReleaseForWait(op, forOp, producerAquires, false);
        schedule.insert(consumerRelease, numStages - 1, nextCluster);
        LLVM_DEBUG({
          LDBG("secondDot wait ");
          op->dump();
          LDBG("secondDot release ");
          consumerRelease->dump();
        });
      }
    }
    schedule.insert(keyOps[2], isConsumerWG ? 0 : numStages - 2,
                    rootUsersCluster);
    schedule.insert(keyOps[3], numStages - 1, nextCluster);
    // Softmax(i+1), MUL1(i+1) in nextNextCluster
    for (auto *heavyOp : heavyCompOps)
      schedule.insert(heavyOp, isConsumerWG ? 0 : numStages - 2,
                      nextNextCluster);
    return schedule.clusters.begin();
  }
  // End __FACEBOOK__ CompPipe

  void runOnOperation() override {
    // Begin __FACEBOOK__ CompPipe
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1 or loop without loop_schedule
      if (getNumStagesOrDefault(forOp) > 1 &&
          forOp->hasAttr(kLoopScheduleAttrName))
        loops.push_back(forOp);
    });

    if (loops.empty())
      return;
    for (scf::ForOp forOp : loops) {
      int loopNumStages = getNumStagesOrDefault(forOp);
      tt::CoarseSchedule coarseSchedule(loopNumStages);
      tt::CoarseSchedule::Cluster afterPrologue;

      std::string loopSchedule = getLoopScheduleOrDefault(forOp);
      if (loopSchedule == "default") {
        afterPrologue =
            getDefaultLoopSchedule(forOp, coarseSchedule, loopNumStages);
      } else if (loopSchedule == "FA_firstDot") {
        afterPrologue =
            getFAFirstDotSchedule(forOp, coarseSchedule, loopNumStages);
      } else if (loopSchedule == "FA_secondDot") {
        afterPrologue =
            getFASecondDotSchedule(forOp, coarseSchedule, loopNumStages);
      } else {
        assert(false && "unrecognized loop schedule");
      }
      // Go through schedule and assign (stage, cluster).
      // shift so afterPrologue will be at clusterId 0
      auto ctx = forOp.getContext();
      for (auto [op, stage_, cluster] : coarseSchedule.getOpsInOrder(forOp)) {
        op->setAttr("loop.stage",
                    IntegerAttr::get(IntegerType::get(ctx, 32), stage_));
        op->setAttr("loop.cluster",
                    IntegerAttr::get(IntegerType::get(ctx, 32),
                                     *cluster - *afterPrologue));
        LLVM_DEBUG({
          LDBG("set stage " << stage_ << " cluster " << (*cluster));
          op->dump();
        });
      }
    }
    // End __FACEBOOK__ CompPipe
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
