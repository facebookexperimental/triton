#include "mlir/IR/Dominance.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace mlir::triton::gpu {

//===----------------------------------------------------------------------===//
// scheduleLoops
//===----------------------------------------------------------------------===//

bool hasGpuBarriers(scf::ForOp forOp) {
  WalkResult result = forOp.walk(
      [&](mlir::gpu::BarrierOp barrier) { return WalkResult::interrupt(); });
  return result.wasInterrupted();
}

// Return true if the preconditions for pipelining the loop are met.
bool isSafeToPipeline(scf::ForOp forOp) {
  // Skip loop with distance > 1.
  if (loopHasDistGreaterThanOne(forOp))
    return false;
  // Don't pipeline outer loops.
  if (isOuterLoop(forOp))
    return false;
  // Skip loops with barriers.
  if (hasGpuBarriers(forOp))
    return false;
  return true;
}

// Process an inner loop inside a warp-specialized loop. This validates
// the preconditions for finding the inner most loop.
void preprocesssWarpSpecializedInnerLoop(scf::ForOp &forOp, Builder &builder) {
  // Only update the innermost loop.
  if (!isOuterLoop(forOp)) {
    // Check that this is a loop that already ran loop scheduling once.
    // If so apply the same attribute to the inner loop.
    if (forOp->hasAttr(kScheduledMaxStageAttrName)) {
      forOp->setAttr(kWarpSpecializeAttrName, builder.getUnitAttr());
    }
  }
}

// Process the given function to propagate the warp-specialize attribute
// from the outer loop to the inner loops. This is done to enable the loop
// scheduler to run on the inner loops after we have finished warp
// specialization.
void preprocesssWarpSpecializedOuterLoop(scf::ForOp &forOp, Builder &builder) {
  if (!isOuterLoop(forOp))
    return;
  // We reuse the same attribute because nothing in the compiler depends on
  // it after loop scheduling as warp specialization is already done. In the
  // future we should make this more robust by using a separate attribute
  // to verify that the loop is already warp-specialized.
  bool hasWarpSpecializeAttr = forOp->hasAttr(kWarpSpecializeAttrName);
  if (hasWarpSpecializeAttr) {
    forOp.walk([&](scf::ForOp innerLoop) {
      preprocesssWarpSpecializedInnerLoop(innerLoop, builder);
    });
  }
}

void doLoopSchedulePreprocessing(ModuleOp moduleOp, Builder &builder) {
  // Process the given function to propagate the warp-specialize attribute
  // from the outer loop to the inner loops. This is done to enable the loop
  // scheduler to run on the inner loops after we have finished warp
  // specialization.
  //
  // To avoid issues with the first invocation, we only propagate the
  // attribute when the inner loop already has the max stage count.
  moduleOp.walk([&](scf::ForOp forOp) {
    preprocesssWarpSpecializedOuterLoop(forOp, builder);
  });
}

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
void scheduleDistanceOneDependencies(scf::ForOp forOp,
                                     CoarseSchedule &schedule) {
  int numStages = schedule.getNumStages();

  // Mapping from the cluster to the cluster before it.
  DenseMap<CoarseSchedule::ClusterHash, CoarseSchedule::Cluster> dist1Cluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op.getBlock()) {
          auto yieldOp = op.getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && schedule.count(defOp) == 0) {
            if (isa<tt::LoadOp>(defOp)) {
              // Exception: Schedule loads with a distance of 1 together
              // with the current op.
              schedule.insertIfAbsent(defOp, stage, cluster);
              schedule.insertDepsOfOp(defOp, stage, cluster,
                                      /*includeArg=*/true,
                                      /*insertIfEarlier=*/true);
            } else {
              CoarseSchedule::ClusterHash clusterHash =
                  CoarseSchedule::hashCluster(cluster);
              if (dist1Cluster.count(clusterHash) == 0) {
                dist1Cluster[clusterHash] =
                    schedule.clusters.newBefore(cluster);
              }
              schedule.insertIfAbsent(defOp, stage + 1,
                                      dist1Cluster[clusterHash]);
              schedule.insertDepsOfOp(defOp, stage + 1,
                                      dist1Cluster[clusterHash],
                                      /*includeArg=*/true,
                                      /*includeIfEarlier=*/true);
            }
          }
        }
      }
    }
  }
}

void scheduleRemainingToLastStage(scf::ForOp forOp, CoarseSchedule &schedule,
                                  CoarseSchedule::Cluster afterPrologue) {
  int numStages = schedule.getNumStages();
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  DenseMap<Operation *, CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0) {
      opToCluster[&op] = afterPrologue;
    }
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == numStages - 1) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        CoarseSchedule::Cluster userCluster = opToCluster[user];
        CoarseSchedule::Cluster opCluster;
        if (schedule.count(op))
          opCluster = schedule[op].second;
        else
          opCluster = opToCluster[op];
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, numStages - 1, cluster);
  }
}

namespace {
bool hasLatenciesAssigned(scf::ForOp forOp,
                          const DenseMap<Operation *, int> &opLatency) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opLatency.count(&op))
      return true;
  }
  return false;
}

// Determine the chain of dots in the given set of users for a dot.
std::tuple<SmallVector<ttng::MMAv5OpInterface>, bool>
computeDotChain(ttng::MMAv5OpInterface dotOp,
                DenseSet<ttng::MMAv5OpInterface> &seenDots) {
  SmallVector<ttng::MMAv5OpInterface> chain;
  std::optional<ttng::MMAv5OpInterface> nextDotOp = dotOp;
  while (nextDotOp.has_value()) {
    ttng::MMAv5OpInterface activeDotOp = nextDotOp.value();
    chain.push_back(activeDotOp);
    seenDots.insert(activeDotOp);
    nextDotOp = std::nullopt;
    DenseSet<Operation *> seenOps;
    SmallVector<Operation *> users;
    auto addUsers = [&](Operation *op) {
      for (mlir::Value result : op->getResults()) {
        for (mlir::OpOperand &use : result.getUses()) {
          auto owner = use.getOwner();
          if (owner && !isa<scf::YieldOp>(owner) && !seenOps.count(owner)) {
            users.push_back(use.getOwner());
          }
        }
      }
    };
    addUsers(activeDotOp);
    while (!users.empty()) {
      auto nextOp = users.pop_back_val();
      if (auto newDotOp = dyn_cast<ttng::MMAv5OpInterface>(nextOp)) {
        if (seenDots.count(newDotOp)) {
          // Already seen dot, not support
          return {chain, false};
        }
        if (nextDotOp.has_value() && nextDotOp != newDotOp) {
          // Not a linear chain
          return {chain, false};
        }
        nextDotOp = newDotOp;
      } else {
        seenOps.insert(nextOp);
        addUsers(nextOp);
      }
    }
  }
  return {chain, true};
}

// Determine the chain of independent dot ops that are present in the body
// of the loop. This will be used to influence the cluster decisions for placing
// the dot ops at a maximum distance from each other. This returns a "success"
// value with the following possible reasons for failure:
// 1. The loop has <= 1 chain of dot ops. This is not helpful for scheduling
// decisions.
// 2. All dots are independent (longest chain is length 1). This is not helpful
// for scheduling decisions.
// 3. The chain of dots is not a line (e.g. A->B and A->C or A->C and B->C).
// This case is too complicated
//    to currently suppport.
// 4. A dot is gated under additional control flow. This is not currently
// supported.
// 5. Any type of dot is present that is not a MMAv5OpInterface.
std::tuple<SmallVector<SmallVector<ttng::MMAv5OpInterface>>, bool>
determineIndependentDotChains(scf::ForOp forOp, int maxStages) {
  DenseSet<ttng::MMAv5OpInterface> seenDots;
  SmallVector<SmallVector<ttng::MMAv5OpInterface>> dotChains;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op)) {
      if (seenDots.count(mmaOp)) {
        // If we have already seen this Dot then we can just skip
        // forward in program order. computeDotChain will detect
        // any non-chain patterns.
        continue;
      }
      auto [dotChain, success] = computeDotChain(mmaOp, seenDots);
      if (!success) {
        return {dotChains, false};
      }
      dotChains.push_back(dotChain);
    } else if (isa<tt::DotOpInterface>(op)) {
      // Cluster decisions require MMAv5OpInterface
      return {dotChains, false};
    } else if (isa<scf::IfOp, scf::ForOp>(op)) {
      // Exit with unsupported control flow.
      bool found = false;
      op.walk([&](tt::DotOpInterface op) {
        found = true;
        // Interrupt the walk early if found
        return mlir::WalkResult::interrupt();
      });
      if (found) {
        return {dotChains, false};
      }
    }
  }
  if (dotChains.size() < 2) {
    // Only 1 chain, ignore.
    return {dotChains, false};
  }
  size_t maxChainLength = 0;
  for (auto &chain : dotChains) {
    maxChainLength = std::max(maxChainLength, chain.size());
  }
  if (maxChainLength < 2) {
    // All chains are length 1. Ignore.
    return {dotChains, false};
  }
  if (maxChainLength > maxStages) {
    // Not enough stages to pipeline
    // out the longest chain.
    return {dotChains, false};
  }
  return {dotChains, true};
}

CoarseSchedule scheduleKeyOps(scf::ForOp forOp,
                              const DenseMap<Operation *, int> &opLatency,
                              int defaultNumStages) {
  llvm::MapVector<Operation *, int> opToStage;
  // Find terminator for later reference
  auto terminator = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  // Determine all operations that have a non-zero latency
  SmallVector<Operation *> latOps;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opLatency.count(&op))
      latOps.push_back(&op);
  }
  // If no latency ops, nothing to schedule
  if (latOps.empty())
    return CoarseSchedule(0);

  // Schedule parallel dot pattern.
  int maxStages = getNumStagesOrDefault(forOp, defaultNumStages);
  // Compute the longest path to the yield for each operation reachable
  // from any latency operation. We also use this to embed stage information
  // for mmas.
  DenseMap<Operation *, int> distance;
  // Track the MMA cluster information for the independent dot chain path.
  DenseMap<Operation *, int> clusterMap;
  auto [chains, success] = determineIndependentDotChains(forOp, maxStages);
  size_t maxChainLength = 0;
  size_t numDots = 0;
  SmallVector<int> maxClusterPerDistance(maxStages, -1);
  if (success) {
    for (auto &chain : chains) {
      maxChainLength = std::max(maxChainLength, chain.size());
      numDots += chain.size();
    }
    // Assign each chain in order. Any time we wrap around to the
    // next stage we assign that op to a later stage. When we can
    // get the same dot distance with a later stage (but an earlier cluster),
    // then we will.
    int startingOffset = 0;
    int incrementValue = (numDots - (maxChainLength - 1));
    for (auto &chain : chains) {
      int lastOffset = startingOffset - incrementValue;
      // Distance is maxStage - stage.
      // We initialize the distance to (chain_length - 1)
      // and decrement to 0.
      // Note the max stage is numStages - 1.
      int currDistance = (chain.size() - 1);
      for (auto &op : chain) {
        int nextOffset = (lastOffset + incrementValue) % numDots;
        if (nextOffset < lastOffset) {
          currDistance--;
        }
        // Update the distance to impact the stage of the MMA
        // and its dependent operations.
        distance[op] = currDistance;
        // Use mmaClusters to encode the ordering of the underlying clusters.
        // This alters the simple heuristic later that cluster = max_stages -
        // stage. To address this we leverage the follow details:
        //
        // 1. Every MMA operand will be at a distance >= MMA distance.
        //    This is because the calculation for distance is distance + .
        // 2. Every user will be at a distance <= MMA distance. This is because
        //    the only ops that have defined distance are MMAs and loads. Since
        //    MMAs are ordered (and guarenteed to be at a smaller distance), the
        //    only way the distance could increase is if the MMA is an input to
        //    to the load, requiring it to be either address, offset, or mask,
        //    all of which are non-sense.
        //
        // As a result, when analyzing distance. We can safely assign each op to
        // a cluster based on its distance as well as already assigned clusters.
        // Anything that comes after an MMA (e.g. no known cluster) but has a
        // computed distance placed in the last cluster for a given stage.
        clusterMap[op] = nextOffset;
        maxClusterPerDistance[currDistance] =
            std::max(maxClusterPerDistance[currDistance], nextOffset);
        lastOffset = nextOffset;
      }
      startingOffset += maxChainLength;
    }
  }
  // Initialize the cluster information for anything
  // not covered by the dots.
  int offset = -1;
  // Assign ops to the clusters in reverse-stage order;
  // ops with higher stage numbers are assigned first. This way we will
  // end up with roughly reverse program order in the clusters.
  for (int i = 0; i < maxStages; i++) {
    if (maxClusterPerDistance[i] == -1) {
      maxClusterPerDistance[i] = numDots + offset++;
    }
  }

  DominanceInfo domInfo(forOp);
  std::function<std::tuple<int, int>(Operation *)> computeDistance =
      [&](Operation *op) -> std::tuple<int, int> {
    auto it = distance.find(op);
    if (it != distance.end()) {
      int cluster = -1;
      auto clusterInfo = clusterMap.find(op);
      if (clusterInfo != clusterMap.end()) {
        cluster = clusterInfo->second;
      }
      return {it->second, cluster};
    }
    // Compute max distance among all users that are inside the loop body
    int maxDist = -1;
    int currCluster = -1;
    for (Operation *user : op->getUsers()) {
      // Only consider users inside the same block and not the terminator
      Operation *inBlockUser = forOp.getBody()->findAncestorOpInBlock(*user);
      if (!inBlockUser || inBlockUser == terminator)
        continue;
      auto [distUser, clusterUser] = computeDistance(inBlockUser);
      if (distUser > maxDist)
        maxDist = distUser;
    }
    int lat = 0;
    if (opLatency.count(op))
      lat = opLatency.lookup(op);
    // If an op has no users (maxDist == -1) but has latency, we include its
    // latency otherwise it contributes 0 to the distance.
    //
    // The maximum distance allowed is the maxmium number of stages.
    int d = std::min(lat + (maxDist < 0 ? 0 : maxDist), maxStages - 1);
    distance[op] = d;
    int c = -1;
    // We must always be scheduled as early as our earliest user for the same
    // distance. If we are at a larger distance (e.g. earlier stage), then we
    // can/should be scheduled to a later cluster. Default to -1 here.
    if (d == maxDist) {
      if (currCluster == -1) {
        c = currCluster;
      } else {
        currCluster = std::min(c, currCluster);
      }
    }
    if (c != -1) {
      clusterMap[op] = c;
    }
    return {d, c};
  };

  // Compute distances for all latency-starting ops
  int maxDistance = 0;
  for (Operation *latOp : latOps) {
    auto [d, _] = computeDistance(latOp);
    if (d > maxDistance)
      maxDistance = d;
  }

  // Assign stage to each op reachable from a latency op
  for (auto [op, dist] : distance) {
    // We only schedule ops that are downstream of a latency op
    // (had a non-negative distance due to a latency op).
    if (dist >= 0)
      opToStage[op] = maxDistance - dist;
  }
  auto stages = llvm::make_second_range(opToStage);
  int maxStage = *llvm::max_element(stages);
  int droppedStages = (maxStages - maxDistance);
  int numClusters = maxClusterPerDistance[(maxClusterPerDistance.size() - 1) -
                                          droppedStages] +
                    1;
  CoarseSchedule schedule(maxStage + 1);
  SmallVector<CoarseSchedule::Cluster> clusters(numClusters);
  for (int i = 0; i < numClusters; i++) {
    clusters[i] = schedule.clusters.newAtBack();
  }

  for (auto [op, stage] : opToStage) {
    auto mappedClusterIdx = clusterMap.find(op);
    int clusterIdx;
    if (mappedClusterIdx != clusterMap.end()) {
      clusterIdx = mappedClusterIdx->second;
    } else {
      auto dist = maxDistance - stage;
      clusterIdx = maxClusterPerDistance[dist];
    }
    schedule.insert(op, stage, clusters[clusterIdx]);
  }

  // Move `scf.if` ops in the current schedule (forward slice of the latency
  // ops) into a new epilogue cluster at the end of the schedule, pushing them
  // as close to the end of the loop body as possible.
  CoarseSchedule::Cluster epilogue = schedule.clusters.newAtBack();
  for (auto [op, stage] : opToStage) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp)
      continue;
    // If the `scf.if` op itself is a latency op, skip it.
    if (opLatency.contains(ifOp))
      continue;
    // Ensure this does not create scheduling conflicts by ensuring the forward
    // slice of the `scf.if` does not contain ops that are already scheduled, as
    // this will cause the `scf.if` to be scheduled after its dependents.
    SetVector<Operation *> slice;
    getForwardSlice(ifOp, &slice);
    if (llvm::any_of(slice, [&](Operation *op) { return opToStage.count(op); }))
      continue;
    schedule.insert(ifOp, stage, epilogue);
  }

  return schedule;
}

// Get an initial schedule for the loop. This is the base schedule from which
// the rest of the pass will backward propagate dependencies.
CoarseSchedule getInitialSchedule(scf::ForOp forOp,
                                  const DenseMap<Operation *, int> &opLatency,
                                  int defaultNumStages) {
  if (!isSafeToPipeline(forOp))
    return CoarseSchedule(0);

  // If the loop has assigned latencies, use them to determine the initial
  // schedule.
  if (hasLatenciesAssigned(forOp, opLatency))
    return scheduleKeyOps(forOp, opLatency, defaultNumStages);

  // If the loop has an existing schedule, use it as the base schedule.
  CoarseSchedule schedule;
  if (forOp->hasAttr(kWarpSpecializeAttrName) &&
      succeeded(schedule.deSerialize(forOp))) {
    // The loop was partitioned from a warp-specialized loop, meaning it can
    // have a partial view of the original loop stages. Re-schedule the loop
    // root at the stages of the latency ops to prune unnecessary stages.
    auto isLatencyOp = [&](Operation &op) {
      return opLatency.count(&op) ||
             isa<LoadOp, DescriptorLoadOp, DescriptorGatherOp, LocalStoreOp,
                 LocalLoadOp, ttng::TMEMLoadOp, ttng::TMEMStoreOp,
                 AsyncCopyGlobalToLocalOp, ttng::AsyncTMACopyGlobalToLocalOp,
                 ttng::AsyncTMAGatherOp, ttng::MMAv5OpInterface,
                 ttng::WaitBarrierOp, ttng::ArriveBarrierOp>(op);
    };

    // If there are no latency ops or all latency ops are in the same stage, we
    // don't need to pipeline the loop. Return a new schedule with everything
    // assigned to the same stage.
    DenseSet<int> latencyStages;
    auto ops = forOp.getBody()->without_terminator();
    for (Operation &op : llvm::make_filter_range(ops, isLatencyOp)) {
      // FIXME: This should assert all latency ops have an assigned stage.
      if (schedule.count(&op))
        latencyStages.insert(schedule[&op].first);
    }
    if (latencyStages.size() <= 1) {
      CoarseSchedule normalized(/*numStages=*/1);
      auto cluster = normalized.clusters.newAtFront();
      for (Operation &op : ops)
        normalized.insert(&op, 0, cluster);
      return normalized;
    }

    schedule.shrinkToFit();
    return schedule;
  }

  return CoarseSchedule(0);
}

// Schedule the prologue and epilogue `if` ops in the loop, pushing them as
// close to the loop boundaries as possible. Return the cluster after the
// prologue (or the beginning of the loop if there is no prologue).
CoarseSchedule::Cluster schedulePrologueAndEpilogue(scf::ForOp forOp,
                                                    CoarseSchedule &schedule) {
  int numStages = schedule.getNumStages();
  CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

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
      opt.omitUsesFromAbove = false;
      (void)getBackwardSlice((Operation *)op, &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  if (!ifsToStage.empty()) {
    CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
    for (auto [ifOp, stage] : ifsToStage) {
      schedule.insertIfAbsent(ifOp, stage, prologueCluster);
    }
  }

  // Other IfOps should be pushed to the end.
  CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (ifsToStage.count(ifOp) == 0) {
        schedule.insertIfAbsent(ifOp, numStages - 1,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

void scheduleLoop(scf::ForOp forOp, const DenseMap<Operation *, int> &opLatency,
                  int defaultNumStages) {
  // Based on the latencies, schedule the key ops to the stages.
  CoarseSchedule schedule =
      getInitialSchedule(forOp, opLatency, defaultNumStages);
  if (schedule.empty())
    return;
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Initial coarse schedule:\n" << forOp << "\n";
  });
  // Schedule the dependencies
  CoarseSchedule::Cluster afterPrologue =
      schedulePrologueAndEpilogue(forOp, schedule);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Coarse schedule with prologue and epilogue:\n" << forOp << "\n";
  });
  scheduleDependencies(forOp, schedule);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Coarse schedule with dependencies:\n" << forOp << "\n";
  });
  scheduleDistanceOneDependencies(forOp, schedule);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Coarse schedule with dist 1:\n" << forOp << "\n";
  });
  scheduleRemainingToLastStage(forOp, schedule, afterPrologue);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Final coarse schedule:\n" << forOp << "\n";
  });

  // Write the schedule to the IR
  schedule.serialize(forOp);
}
} // namespace

/// Schedule the loops based on the latencies assigned to the operations.
void scheduleLoops(ModuleOp moduleOp, int defaultNumStages) {
  DenseMap<Operation *, int> opLatency = deserializeLatencies(moduleOp);
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    scheduleLoop(forOp, opLatency, defaultNumStages);
  }
}
//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_TRITONGPUSCHEDULELOOPS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct ScheduleLoops : public impl::TritonGPUScheduleLoopsBase<ScheduleLoops> {
  using TritonGPUScheduleLoopsBase::TritonGPUScheduleLoopsBase;

  void runOnOperation() override { scheduleLoops(getOperation(), numStages); }
};

} // namespace mlir::triton::gpu
