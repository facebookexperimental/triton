#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// assignPartitions
//===----------------------------------------------------------------------===//

// Find the last operation in the loop body that defined this value, with a
// maximum of distance 1.
static Operation *findDefOpInLoop(scf::ForOp loop, Value value,
                                  int distance = 0) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getParentBlock() != loop.getBody())
      return {};
    // Don't look back more than distance 1.
    if (distance == 1)
      return {};
    return findDefOpInLoop(
        loop, loop.getYieldedValues()[arg.getArgNumber() - 1], distance + 1);
  }
  Operation *defOp = value.getDefiningOp();
  if (!loop.getBodyRegion().isAncestor(defOp->getParentRegion()))
    return {};
  return defOp;
}

// For `op`, invoke `callback` on all the definitions of its inputs from within
// `loop`, which might not be in the same iteration.
static void iterateDefs(scf::ForOp loop, Operation *op,
                        function_ref<void(OpResult)> callback) {
  visitNestedOperands(op, [&](OpOperand &operand) {
    Value value = operand.get();
    if (value.getParentBlock() != loop.getBody())
      return;
    auto arg = dyn_cast<BlockArgument>(value);
    if (arg == loop.getInductionVar())
      return;
    auto [def, distance] = getDefinitionAndDistance(loop, operand.get());
    if (def && def.getParentBlock() == loop.getBody())
      callback(def);
  });
}

// For `op`, invoke `callback` on all its transitive users within `loop`, which
// may be in a future iteration.
static void iterateUsers(scf::ForOp loop, Operation *op,
                         function_ref<void(Operation *)> callback) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : op->getUses())
    uses.push_back(&use);
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    Operation *owner = loop.getBody()->findAncestorOpInBlock(*use->getOwner());
    if (!isa<scf::YieldOp>(owner)) {
      callback(owner);
      continue;
    }
    BlockArgument arg = loop.getRegionIterArg(use->getOperandNumber());
    for (OpOperand &use : arg.getUses())
      uses.emplace_back(&use);
  }
}

// Check if any of the inputs to `op` are reachable from a non-null partition.
static bool hasDefPartition(scf::ForOp loop, Operation *op,
                            WarpSchedule &schedule) {
  SmallVector<Operation *> worklist{op};
  DenseSet<Operation *> seen;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!seen.insert(op).second)
      continue;
    Partition *p = schedule.getPartition(op);
    if (p && p != schedule.getRootPartition())
      return true;
    iterateDefs(loop, op,
                [&](OpResult def) { worklist.push_back(def.getDefiningOp()); });
  }
  return false;
}

// Recursively schedule the dependencies of an operation, stopping when
// encountering an operation that is already assigned.
static void scheduleDependencies(scf::ForOp loop, WarpSchedule &schedule,
                                 Partition *partition, Operation *op) {
  SmallVector<Value> deps;
  for (Value value : getNestedOperands(op)) {
    if (isa<RankedTensorType, MemDescType>(value.getType()))
      deps.push_back(value);
  }

  while (!deps.empty()) {
    Value dep = deps.pop_back_val();

    if (auto arg = dyn_cast<BlockArgument>(dep)) {
      if (arg.getOwner() == loop.getBody() && arg != loop.getInductionVar())
        deps.push_back(loop.getYieldedValues()[arg.getArgNumber() - 1]);
      continue;
    }

    Operation *defOp =
        loop.getBody()->findAncestorOpInBlock(*dep.getDefiningOp());
    if (!defOp || !hasDefPartition(loop, defOp, schedule) ||
        !schedule.trySchedule(partition, defOp))
      continue;
    llvm::append_range(deps, getNestedOperands(defOp));
  }
}

// Recursively schedule the users of an operation, stopping when
// encountering an operation that is already assigned.
// If \p partition is null, a new partition will be created if needed.
static void scheduleUsers(scf::ForOp loop, WarpSchedule &schedule,
                          Partition *partition, Operation *op) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : op->getUses())
    uses.push_back(&use);
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    Operation *user = loop.getBody()->findAncestorOpInBlock(*use->getOwner());

    if (user == loop.getBody()->getTerminator()) {
      for (OpOperand &use :
           loop.getRegionIterArg(use->getOperandNumber()).getUses())
        uses.push_back(&use);
      continue;
    }

    if (schedule.isScheduled(user))
      continue;
    if (!partition)
      partition = schedule.addPartition(/* stage is unused */ 0);
    schedule.trySchedule(partition, user);
    for (OpOperand &use : user->getUses())
      uses.push_back(&use);
  }
}

// Given a partitioning scheme, determine an initial schedule by performing a
// first-order partition assignment to the operations in the scheme and its
// users and/or dependencies. This sets up the initial partitioning of the ops.
static std::optional<WarpSchedule> getInitialSchedule(scf::ForOp mainLoop) {
  // Check for an existing schedule.
  if (FailureOr<WarpSchedule> scheduleOr = WarpSchedule::deserialize(mainLoop);
      succeeded(scheduleOr))
    return {std::move(*scheduleOr)};

  // Start by creating the default partition, a partition for for all loads, and
  // a partition for all MMAs.
  WarpSchedule schedule;
  Partition *defaultPartition = schedule.addPartition(0);
  Partition *mmaPartition = schedule.addPartition(1);
  Partition *loadPartition = schedule.addPartition(0);

  SmallVector<scf::ForOp> loops{mainLoop.getOps<scf::ForOp>()};
  loops.push_back(mainLoop);

  // Find loads to pipeline.
  SmallVector<Operation *> loadsAndAllocs;
  for (auto loop : loops) {
    for (Operation &op : loop.getOps()) {
      // Only TMA loads are supported at the moment.
      if (!isa<DescriptorLoadOp, DescriptorGatherOp>(op))
        continue;
      schedule.trySchedule(loadPartition, &op);
      loadsAndAllocs.push_back(&op);

      // Local alloc users of the load with matching encoding will cause the
      // underlying buffer to be pass through. Keep track of them.
      SharedEncodingTrait sharedEnc = getSharedEncoding(&op);
      for (Operation *user : op.getUsers()) {
        if (auto alloc = dyn_cast<LocalAllocOp>(user)) {
          if (sharedEnc == alloc.getType().getEncoding()) {
            schedule.trySchedule(loadPartition, alloc);
            loadsAndAllocs.push_back(alloc);
          }
        } else if (isa<ttng::TMEMAllocOp>(user)) {
          schedule.trySchedule(loadPartition, user);
          loadsAndAllocs.push_back(user);
        }
      }
    }
  }

  // Ensure the epilogue stores are in a separate partition.
  auto epiloguePartition = schedule.addPartition(/* stage is unused */ 0);
  for (auto loop : loops)
    for (DescriptorStoreOp op : loop.getOps<DescriptorStoreOp>())
      schedule.trySchedule(epiloguePartition, op);

  // Find MMAs to pipeline.
  SmallVector<ttng::MMAv5OpInterface> mmas;
  for (auto loop : loops) {
    for (auto mmaOp : loop.getOps<ttng::MMAv5OpInterface>()) {
      schedule.trySchedule(mmaPartition, mmaOp);
      mmas.push_back(mmaOp);

      // If the store is unrelated to the use of the MMA, then it gets placed in
      // the MMA partition.
      auto storeOp = dyn_cast_or_null<ttng::TMEMStoreOp>(
          findDefOpInLoop(loop, mmaOp.getAccDep()));
      if (!ttng::hasAccReadModifyWrite(mmaOp, loop) && storeOp &&
          loop.isDefinedOutsideOfLoop(storeOp.getSrc()))
        schedule.trySchedule(mmaPartition, storeOp);
    }
    for (auto mmaOp : mmas) {
      // Look for views into the operands.
      SmallVector<Operation *> operandViews;
      for (Value operand : mmaOp->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp())
          operandViews.push_back(defOp);
      }
      while (!operandViews.empty()) {
        Operation *op = operandViews.pop_back_val();
        if (!op->hasTrait<OpTrait::MemDescViewTrait>())
          continue;

        // Duplicate the op if necessary to ensure that the MMA partition is the
        // only user.
        if (!llvm::all_of(op->getUsers(), [&](Operation *user) {
              return schedule.getPartition(user) == mmaPartition;
            })) {
          Operation *newOp = OpBuilder(op).clone(*op);
          op->replaceUsesWithIf(newOp->getResults(), [&](OpOperand &use) {
            return schedule.getPartition(use.getOwner()) == mmaPartition;
          });
          op = newOp;
        }

        schedule.trySchedule(mmaPartition, op);
        if (Operation *defOp = op->getOperand(0).getDefiningOp())
          operandViews.push_back(defOp);
      }
    }
  }

  // If there are no loads or MMAs, don't warp specialize.
  if (loadsAndAllocs.empty() && mmas.empty())
    return std::nullopt;

  // Disable scheduling exp on the default partition for now, because we want
  // softmax on separate partitions.
  // Propagate defs of exp.
  // for (scf::ForOp loop : loops) {
  //   for (Operation &op : loop.getOps()) {
  //     if (!isa<math::Exp2Op, ElementwiseInlineAsmOp>(op))
  //       continue;
  //     int elementCount = 0;
  //     for (Type type : op.getResultTypes()) {
  //       if (auto tensorTy = dyn_cast<RankedTensorType>(type))
  //         elementCount += tensorTy.getNumElements();
  //     }
  //     if (elementCount > 256) {
  //       schedule.trySchedule(defaultPartition, &op);
  //       scheduleDependencies(loop, schedule, defaultPartition, &op);
  //     }
  //   }
  // }

  // Propagate users of loads and MMAs.
  // Load users go to the default partition.
  for (Operation *loadOrAlloc : loadsAndAllocs)
    scheduleUsers(loadOrAlloc->getParentOfType<scf::ForOp>(), schedule,
                  defaultPartition, loadOrAlloc);

  // HACK: If the user is in the next iteration, put it in the default
  // partition. This ensures that the correction goes to the default partition,
  // because it uses acc.
  for (auto mmaOp : mmas) {
    for (OpOperand &use : mmaOp->getUses()) {
      auto loop = mmaOp->getParentOfType<scf::ForOp>();
      if (use.getOwner() != loop.getBody()->getTerminator())
        continue;
      for (OpOperand &use :
           loop.getRegionIterArg(use.getOperandNumber()).getUses()) {
        schedule.trySchedule(defaultPartition, use.getOwner());
        scheduleUsers(loop, schedule, defaultPartition, use.getOwner());
      }
      break;
    }
  }

  // The users of MMAs go to a new partition for each MMA.
  for (auto mmaOp : llvm::reverse(mmas)) {
    scheduleUsers(mmaOp->getParentOfType<scf::ForOp>(), schedule, nullptr,
                  mmaOp);
  }

  // Handle operations outside the loop (both prologue and epilogue).
  // This ensures all operations, including TMA loads/stores outside
  // the loop, get partition assignments.
  Partition *outsideLoopPartition = schedule.getRootPartition();

  // Walk all operations in the parent block (both before and after the loop)
  Block *parentBlock = mainLoop->getBlock();
  for (Operation &op : *parentBlock) {
    // Skip the loop itself
    if (&op == &(*mainLoop))
      continue;

    // Skip operations that are already scheduled
    if (schedule.isScheduled(&op))
      continue;

    // Skip terminators
    if (isa<triton::ReturnOp>(op))
      continue;

    // Assign this operation and all nested operations to the root partition
    op.walk([&](Operation *nestedOp) {
      if (!schedule.isScheduled(nestedOp) && !isa<triton::ReturnOp>(nestedOp)) {
        schedule.insert(outsideLoopPartition, nestedOp);
        // Manually set the async_task_id attribute since serialize() won't
        // reach operations outside the loop.
        //
        // ALL operations outside loops belong to the root partition (index 0).
        // Even though allocations outside loops may be used by multiple
        // partitions inside the loop, they should only have ONE task ID.
        // The compiler's existing mechanisms handle cross-partition visibility
        // for operations in partition 0.
        SmallVector<int32_t> taskIds{0};
        nestedOp->setAttr(
            "async_task_id",
            DenseI32ArrayAttr::get(nestedOp->getContext(), taskIds));
      }
    });
  }

  return schedule;
}

namespace {
// This data structure represents a cluster of operations that have not been
// assigned to a stage. Operations form a cluster when:
//
// - they are adjacent in the SSA use def graph
// - they are not already assigned to a partition
// - at least one of their inputs is reachable from a definition partition
//
struct OpCluster {
  // These are the operations in the cluster.
  SetVector<Operation *> ops;
  // The definition partitions are the partitions from which inputs of the
  // operation are reachable. When the cluster is fully formed, the defining op
  // in the loop of any input to any operation in the cluster is either in the
  // root partition or one of these partitions.
  SetVector<Partition *> defPartitions;
  // The sink partitions which consume the outputs of operations in this
  // cluster. When the cluster is fully formed, all uses in the loop of outputs
  // of any operation in the cluster belong to one of these partitions.
  SetVector<Partition *> sinkPartitions;
};

// Owning class for a bunch of clusters. This class manages the lifetimes of the
// clusters and has some helper functions.
struct OpClusters : public llvm::MapVector<Operation *, OpCluster *> {
  using MapVector::MapVector;

  // Create a new cluster that contains only the given operation, a return a
  // cluster that already contains the operation.
  OpCluster *getOrCreate(Operation *op) {
    OpCluster *&cluster = (*this)[op];
    if (!cluster) {
      cluster = clusters.emplace_back(new OpCluster).get();
      cluster->ops.insert(op);
    }
    return cluster;
  }
  // Merge two clusters by merging their sets and clearing the other cluster,
  // marking it as dead.
  void merge(OpCluster *dst, OpCluster *src) {
    dst->ops.insert_range(src->ops);
    dst->defPartitions.insert_range(src->defPartitions);
    dst->sinkPartitions.insert_range(src->sinkPartitions);
    for (Operation *op : src->ops)
      (*this)[op] = dst;
    src->ops.clear();
    src->defPartitions.clear();
    src->sinkPartitions.clear();
  }

  SmallVector<std::unique_ptr<OpCluster>> clusters;
};
} // namespace

// Operations that require partition assignment are those reachable from an
// operation in a partition. This function propagates partitions by first
// forming contiguous clusters from the unassigned operations and then deciding
// what to do with the operations in that cluster.
void propagatePartitions(scf::ForOp loop, WarpSchedule &schedule) {
  OpClusters opClusters;

  for (Partition &partition : schedule.getPartitions()) {
    // For each partition, check if any of their inputs are reachable from
    // another partition and spawn a single cluster at that operation.
    auto defCallback = [&](OpResult result, unsigned distance) {
      Operation *defOp = result.getDefiningOp();
      if (!schedule.isScheduled(defOp) &&
          hasDefPartition(loop, defOp, schedule)) {
        // Add the current partition as a sink to the cluster.
        opClusters.getOrCreate(defOp)->sinkPartitions.insert(&partition);
      }
    };
    schedule.iterateDefs(loop, &partition, defCallback);

    // For each partition, place users of its outputs in a cluster if it is not
    // already assigned to a partition.
    auto useCallback = [&](OpResult result, OpOperand &use, unsigned distance) {
      Operation *user = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      if (!schedule.isScheduled(user)) {
        // Add the current partition as a def to the cluster.
        opClusters.getOrCreate(user)->defPartitions.insert(&partition);
      }
    };
    schedule.iterateUses(loop, &partition, useCallback);
  }

  // Now we have a pile of single-operation clusters directly adjacent to the
  // operations in a partition. Grow the clusters by adding adjacent operations
  // clusters and merging clusters when possible.
  SmallVector<Operation *> worklist =
      llvm::to_vector(llvm::make_first_range(opClusters));
  while (!worklist.empty()) {
    // Grab an op off the worklist. We know it has a cluster already.
    Operation *op = worklist.pop_back_val();
    OpCluster *cluster = opClusters.find(op)->second;
    // Look at the definitions directly feeding into this operation.
    iterateDefs(loop, op, [&](OpResult def) {
      Operation *defOp = def.getDefiningOp();
      if (schedule.isScheduled(defOp)) {
        // The input originates from an operation already assigned to a
        // partition. Add this as a def partition.
        cluster->defPartitions.insert(schedule.getPartition(defOp));
      } else {
        // If the input is not reachable from a partition, ignore it.
        if (!hasDefPartition(loop, defOp, schedule))
          return;
        // This operation is not assigned to a partition.
        OpCluster *&defCluster = opClusters[defOp];
        if (!defCluster) {
          // This operation has not yet been added to a cluster. Add it to the
          // current cluster and recurse on it.
          defCluster = cluster;
          cluster->ops.insert(defOp);
          worklist.push_back(defOp);
        } else if (defCluster != cluster) {
          // This operation is part of another cluster. Merge the two clusters
          // together and continue.
          opClusters.merge(cluster, defCluster);
        }
      }
    });
    // Check the users of the operation.
    iterateUsers(loop, op, [&](Operation *user) {
      if (schedule.isScheduled(user)) {
        // If the user is already assigned to a partition, add that partition as
        // one of the sink partitions.
        Partition *userPartition = schedule.getPartition(user);
        cluster->sinkPartitions.insert(userPartition);
        return;
      }
      // If the user does not already have a cluster, add it to the current
      // cluster. We don't have to handle merging here because when the user
      // visits the current op, it will trigger the merge.
      OpCluster *&userCluster = opClusters[user];
      if (userCluster)
        return;
      userCluster = cluster;
      cluster->ops.insert(user);
      worklist.push_back(user);
    });
  }

  // We have clustered unassigned ops in the liveouts of ops in assigned
  // partitions and in the critical paths between ops in different partitions.
  // Ops that are next to each other are placed in the same cluster. Now the
  // task is to figure out how to assign partitions to the ops in each cluster
  // based on the def and sink partitions, which is very non-trivial.
  for (OpCluster &cluster : llvm::make_pointee_range(opClusters.clusters)) {
    // Skip dead clusters.
    if (cluster.ops.empty())
      continue;
    assert(!cluster.defPartitions.empty());
    assert(llvm::all_of(
        cluster.ops, [&](Operation *op) { return !schedule.isScheduled(op); }));

    // If there are multiple def or sink partitions, don't know what to do.
    // Assign the whole cluster to its own partition.
    if (cluster.defPartitions.size() > 1 || cluster.sinkPartitions.size() > 1) {
      Partition *newPartition = schedule.addPartition(0);
      for (Operation *op : cluster.ops)
        schedule.insert(newPartition, op);
      continue;
    }

    // If there is no sink partition, this means there is a backedge somewhere,
    // for now assign the cluster to the def partition.
    Partition *defPartition = cluster.defPartitions.front();
    if (cluster.sinkPartitions.empty()) {
      for (Operation *op : cluster.ops)
        schedule.insert(defPartition, op);
      continue;
    }

    // Find the critical path between the def partition and sink partition.
    Partition *sinkPartition = cluster.sinkPartitions.front();
    SetVector<Operation *> critPath;
    DenseSet<Operation *> opsInCluster(cluster.ops.begin(), cluster.ops.end());
    auto callback = [&](OpResult result, unsigned distance) {
      Operation *defOp = result.getDefiningOp();
      if (opsInCluster.contains(defOp))
        critPath.insert(defOp);
    };
    schedule.iterateDefs(loop, sinkPartition, callback);
    for (unsigned i = 0; i < critPath.size(); ++i) {
      Operation *op = critPath[i];
      iterateDefs(loop, op, [&](OpResult def) {
        Operation *defOp = def.getDefiningOp();
        if (opsInCluster.contains(defOp))
          critPath.insert(defOp);
      });
    }

    // If all ops are on the critical path, assign them to the def partition.
    if (critPath.size() == cluster.ops.size()) {
      for (Operation *op : cluster.ops)
        schedule.insert(defPartition, op);
      continue;
    }

    // Some ops are on the critical path, and there is also a backedge.
    // Rematerialize the critical path ops into the sink partition. Leave the
    // rest in the def partition and rely on DCE to remove them.
    critPath = topologicalSort(critPath);
    DenseSet<Operation *> sinkOps(sinkPartition->getOps().begin(),
                                  sinkPartition->getOps().end());
    for (Operation *op : llvm::reverse(critPath)) {
      OpBuilder b(op);
      Operation *clone = b.clone(*op);
      op->replaceUsesWithIf(clone->getResults(), [&](OpOperand &use) {
        return sinkOps.contains(use.getOwner());
      });
      sinkOps.insert(clone);
      schedule.insert(sinkPartition, clone);
    }
    for (Operation *op : cluster.ops)
      schedule.insert(defPartition, op);
  }
}

// Rematerialize chains of broadcasts where the user is in a different partition
// than the broadcast to reduce the amount of data that needs to be transferred.
void rematerializeBroadcasts(WarpSchedule &schedule, OpOperand *use) {
  static_assert(
      std::is_base_of_v<OpTrait::OneResult<BroadcastOp>, BroadcastOp> &&
      std::is_base_of_v<OpTrait::OneResult<ExpandDimsOp>, ExpandDimsOp> &&
      std::is_base_of_v<OpTrait::OneResult<ConvertLayoutOp>, ConvertLayoutOp>);

  Operation *defOp = use->get().getDefiningOp();
  while (isa_and_nonnull<BroadcastOp, ExpandDimsOp, ConvertLayoutOp>(defOp)) {
    Operation *clone = OpBuilder(defOp).clone(*defOp);
    Partition *userPartition = schedule.getPartition(use->getOwner());
    assert(userPartition && "user not scheduled");
    schedule.insert(userPartition, clone);
    use->set(clone->getResult(0));

    defOp = clone->getOperand(0).getDefiningOp();
    use = &clone->getOpOperand(0);
  }
}

/// Walk over \p loop and clone Broadcast/ExpandDims/ConvertLayout ops into each
/// partition that they have users in. This reduces the amount of data that
/// needs to be transferred through memory.
void optimizeSchedule(scf::ForOp loop, WarpSchedule &schedule) {
  // Walk everything in reverse so that operations are visited before their
  // operands.
  loop.walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    if (!isa<BroadcastOp, ExpandDimsOp, ConvertLayoutOp>(op))
      return;

    Partition *partition = schedule.getPartition(op);
    if (!partition)
      return;

    // Record all the other partitions in which we have users.
    llvm::SmallDenseSet<Partition *, 2> userPartitions;
    for (OpOperand &use : op->getUses()) {
      Partition *userPartition = schedule.getPartition(use.getOwner());
      if (!userPartition || userPartition == partition)
        continue;
      userPartitions.insert(userPartition);
    }

    for (auto *userPartition : userPartitions) {
      // Clone the instruction into each user partition.
      Operation *clone = OpBuilder(op).clone(*op);
      schedule.insert(userPartition, clone);
      // Replace all users in that partition with the clone.
      op->replaceUsesWithIf(clone->getResults(), [&](OpOperand &otherUse) {
        return schedule.getPartition(otherUse.getOwner()) == userPartition;
      });
    }
  });
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUPARTITIONSCHEDULING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct PartitionScheduling
    : public triton::gpu::impl::TritonGPUPartitionSchedulingBase<
          PartitionScheduling> {
  using TritonGPUPartitionSchedulingBase::TritonGPUPartitionSchedulingBase;

  void runOnOperation() override;
};
} // namespace

void PartitionScheduling::runOnOperation() {
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName))
      loops.push_back(loop);
  });
  for (auto [idx, loop] : llvm::enumerate(loops)) {
    if (std::optional<WarpSchedule> schedule = getInitialSchedule(loop)) {
      propagatePartitions(loop, *schedule);
      optimizeSchedule(loop, *schedule);
      schedule->serialize(loop);
      loop->setAttr(
          kWarpSpecializeTagAttrName,
          IntegerAttr::get(IntegerType::get(loop.getContext(), 32), idx));
      // Clean Broadcast/ExpandDims/ConvertLayout that were left with no users
      // after optimizeSchedule. We wait until after the schedule is serialized
      // to avoid invalidating pointers stored in the schedule.
      loop.walk<WalkOrder::PostOrder, ReverseIterator>([](Operation *op) {
        // By default, the walk is in postorder so it is safe to delete ops
        // while we walk.
        if (isa<BroadcastOp, ExpandDimsOp, ConvertLayoutOp>(op))
          if (op->use_empty())
            op->erase();
      });
    }
  }
}
