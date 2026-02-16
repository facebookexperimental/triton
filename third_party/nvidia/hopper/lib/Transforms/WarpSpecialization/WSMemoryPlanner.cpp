#include "CodePartitionUtility.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "llvm/Support/raw_os_ostream.h"

#define DEBUG_TYPE "nvgpu-ws-memory-planner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Environment variable to dump DOT files: TRITON_DUMP_WS_GRAPHS
// When set to a directory path, dumps visualization files there.
// Example: TRITON_DUMP_WS_GRAPHS=/tmp/graphs
static std::optional<std::string> getGraphDumpDir() {
  if (const char *env = std::getenv("TRITON_DUMP_WS_GRAPHS")) {
    return std::string(env);
  }
  return std::nullopt;
}

// Counter for unique file names when multiple kernels are compiled
static std::atomic<int> graphDumpCounter{0};

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

using OperationListT = std::vector<Operation *>;

//===----------------------------------------------------------------------===//
// reorderOpsBySchedule - Reorder ops in pipelined loops by cluster assignment
//===----------------------------------------------------------------------===//

/// Helper to read the loop.stage attribute from an op. Returns -1 if absent.
static int getLoopStage(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
  return attr ? attr.getValue().getSExtValue() : -1;
}

/// Helper to read the loop.cluster attribute from an op. Returns -1 if absent.
static int getLoopCluster(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>(tt::kLoopClusterAttrName);
  return attr ? attr.getValue().getSExtValue() : -1;
}

/// Returns true if `op` has a data dependency on any result of `defOp`
/// (i.e. `op` uses a value defined by `defOp`).
static bool dependsOn(Operation *op, Operation *defOp) {
  for (Value operand : op->getOperands()) {
    if (Operation *d = operand.getDefiningOp()) {
      if (d == defOp)
        return true;
    }
  }
  return false;
}

/// For each innermost scf::ForOp that carries loop.stage / loop.cluster
/// attributes, reorder the operations so that within the same stage,
/// ops with smaller cluster IDs appear before ops with larger cluster IDs.
/// Within the same (stage, cluster), original program order is preserved.
///
/// Algorithm (per ForOp):
///   For each stage x (ascending):
///     Find the smallest cluster number for this stage.
///     For each cluster y from smallest to largest:
///       If y is the smallest cluster, set lastOp to the last op in this
///         cluster (they are already in program order) and continue.
///       Let A = the last op with cluster y-1 (i.e. lastOp from previous
///         cluster).
///       For each op in [stage x, cluster y] in program order:
///         If the op is already after A, update lastOp and continue.
///         Otherwise, move it to be right after A (if dependency constraints
///           are satisfied).
///
/// A move is skipped if it would place an op before an op it depends on.
static void reorderOpsBySchedule(Operation *funcOp) {
  funcOp->walk([](scf::ForOp forOp) {
    Block *body = forOp.getBody();

    // ---- 1. Collect scheduled ops, grouped by (stage, cluster) ----
    // stageMap[stage][cluster] = list of ops in program order
    std::map<int, std::map<int, SmallVector<Operation *>>> stageMap;
    for (Operation &op : body->without_terminator()) {
      int stage = getLoopStage(&op);
      int cluster = getLoopCluster(&op);
      if (stage < 0 || cluster < 0)
        continue;
      stageMap[stage][cluster].push_back(&op);
    }

    if (stageMap.empty())
      return;

    // ---- 2. For each stage, reorder clusters ----
    for (auto &[stage, clusterMap] : stageMap) {
      Operation *lastOp = nullptr;

      // clusterMap is std::map<int,...> so iteration is ascending by cluster.
      bool isFirstCluster = true;
      for (auto &[cluster, ops] : clusterMap) {
        if (isFirstCluster) {
          // Smallest cluster: ops stay in place; just record the last one.
          lastOp = ops.back();
          isFirstCluster = false;
          continue;
        }

        // For each op in this cluster (in program order), ensure it comes
        // after lastOp (the last op from the previous cluster).
        Operation *A = lastOp;
        for (Operation *op : ops) {
          if (A->isBeforeInBlock(op)) {
            // Already after A — nothing to do.
            lastOp = op;
          } else {
            // op is before A: need to move op to right after A.
            // Check that none of op's operands are defined by an op that
            // would end up after the insertion point.
            bool safe = true;
            for (Value operand : op->getOperands()) {
              Operation *defOp = operand.getDefiningOp();
              if (!defOp || defOp->getBlock() != body)
                continue;
              // defOp must be before or equal to A for the move to be safe.
              if (A->isBeforeInBlock(defOp) || defOp == op) {
                LDBG("reorderOpsBySchedule: cannot move "
                     << *op << " after " << *A << " — depends on " << *defOp);
                safe = false;
                break;
              }
            }
            if (safe) {
              op->moveAfter(A);
              lastOp = op;
            } else {
              // Cannot move; leave in place but still track for subsequent ops.
              lastOp = op;
            }
          }
          // Update A so subsequent ops in this cluster go after the latest one.
          A = lastOp;
        }
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// MemoryPlannerBase - Abstract base class for memory planners
//===----------------------------------------------------------------------===//

/// Abstract base class for memory planners in warp-specialized kernels.
/// Provides common functionality for both SMEM and TMEM memory planning,
/// including operation ID mapping, channel lookup, and liveness computation.
/// Subclasses implement memory-type-specific allocation strategies.
class MemoryPlannerBase {
public:
  MemoryPlannerBase(Operation *operation, Allocation *allocation,
                    SmallVector<Channel *> *channels)
      : operation(operation), allocation(allocation), channels(channels) {}

  virtual ~MemoryPlannerBase() = default;

  /// Run the memory planner with the given number of buffers.
  /// @param numBuffers Number of buffers for multi-buffering (SMEM) or
  ///                   starting buffer ID (TMEM)
  /// @return LogicalResult indicating success or failure.
  virtual LogicalResult run(unsigned numBuffers) = 0;

protected:
  Operation *operation;
  Allocation *allocation;
  SmallVector<Channel *> *channels;
  DenseMap<Operation *, size_t> operationId;

  /// Build the operation ID map by walking the operation tree.
  /// Assigns monotonically increasing IDs to operations in post-order.
  void buildOperationIdMap() {
    operation->walk<WalkOrder::PostOrder>([&](Operation *op) {
      LLVM_DEBUG(
          op->setAttr("operation_id",
                      IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                       operationId.size())));
      operationId[op] = operationId.size();
    });
  }

  /// Get the channel kind this planner handles.
  /// @return DataChannelKind::SMEMPost or DataChannelKind::TMEMPost
  virtual DataChannelKind getChannelKind() const = 0;

  /// Compute the liveness interval for a value.
  /// @param value The allocation value to compute liveness for
  /// @return Interval representing the live range in operation IDs
  virtual Interval<size_t> computeLivenessInterval(Value value) = 0;

  /// Compute the interval for the liveness operations.
  /// @param liveOps The vector of live operations
  /// @return Interval representing the live range in operation IDs
  Interval<size_t> computeIntervalFromOps(const OperationListT &liveOps) {
    if (liveOps.empty()) {
      return Interval<size_t>(0, 0);
    }
    auto minId = std::numeric_limits<size_t>::max();
    auto maxId = std::numeric_limits<size_t>::min();
    for (Operation *liveOp : liveOps) {
      if (operationId[liveOp] < minId) {
        minId = operationId[liveOp];
      }
      if ((operationId[liveOp] + 1) > maxId) {
        maxId = operationId[liveOp] + 1;
      }
    }
    return Interval(minId, maxId);
  }

  /// Get the interval for a control operation (ForOp).
  /// @param ctrlOp The control operation (typically a scf::ForOp)
  /// @return Interval from first instruction to the control op
  Interval<size_t> getIntervalForCtrlOp(Operation *ctrlOp) {
    auto forOp = dyn_cast<scf::ForOp>(ctrlOp);
    if (!forOp) {
      return Interval<size_t>(0, 0);
    }
    for (Operation &op : forOp.getBody()->without_terminator()) {
      return Interval(operationId[&op], operationId[ctrlOp]);
    }
    return Interval(operationId[ctrlOp], operationId[ctrlOp]);
  }
};

/// Check if a ForOp is an innermost loop (contains no nested ForOps).
/// @param forOp The loop operation to check
/// @return true if the loop has no nested ForOp, false otherwise
static bool isInnermostLoop(scf::ForOp forOp) {
  for (Operation &nestedOp : forOp.getBody()->getOperations()) {
    if (isa<scf::ForOp>(nestedOp)) {
      return false;
    }
  }
  return true;
}

/// Find the channel associated with a given allocation operation.
/// @param op The operation to find a channel for (typically an allocation op)
/// @param channels The list of channels to search through
/// @return Pointer to the matching Channel, or nullptr if not found
static Channel *findChannelForOp(Operation *op,
                                 SmallVector<Channel *> &channels) {
  Channel *TheCh = nullptr;
  for (auto *ch : channels) {
    Operation *alloc = ch->getAllocOp();
    if (alloc == op) {
      TheCh = ch;
      break;
    }
  }
  return TheCh;
}

/// Find the channel associated with a value's defining allocation operation.
/// Convenience wrapper around findChannelForOp.
/// @param value The value whose defining operation to find a channel for
/// @param channels The list of channels to search through
/// @return Pointer to the matching Channel, or nullptr if not found
static Channel *findChannelForAlloc(Value value,
                                    SmallVector<Channel *> &channels) {
  return findChannelForOp(value.getDefiningOp(), channels);
}

/// Collect all actual users (consumers) of a channel.
/// For a channel, this includes the source operation and the actual consumers
/// derived from the destination operations.
/// @param TheCh The channel to get users for (may be nullptr)
/// @param users Output set to collect all user operations
/// @param alloc Optional allocation operation for validation
/// @return success() if users were collected, failure() if validation failed
static LogicalResult getAllAcutalUsersForChannel(Channel *TheCh,
                                                 DenseSet<Operation *> &users,
                                                 Operation *alloc = nullptr) {
  // Skip null channels
  if (!TheCh) {
    // Allocations inside loops should have associated channels
    // For outside loop ops, channels are not created when there is
    // no valid producer or outside loop op has no task IDs (e.g., store)
    if (alloc && alloc->getParentOfType<scf::ForOp>()) {
      return alloc->emitError(
          "getAllAcutalUsersForChannel: expected channel for allocation "
          "inside loop");
    }
    return success();
  }
  Operation *src = TheCh->getSrcOp();
  // Skip channels without valid source operations (e.g., allocations outside
  // loops)
  if (!src)
    return success();
  SmallVector<Operation *> dsts;
  TheCh->getDstOps(dsts);
  users.insert(src);
  for (auto *op : dsts) {
    auto actual = getActualConsumers(op);
    for (auto *tOp : actual)
      users.insert(tOp);
  }
  return success();
}

/// Find the lowest common ancestor scope that contains both operations.
/// Walks up the parent hierarchy of operation 'a' to collect all ancestor
/// scopes, then walks up 'b' until it finds a matching scope.
/// @param a The first operation to find common scope for
/// @param b The second operation to lift until it reaches the common scope
/// @return The common ancestor Operation, or nullptr if no common scope found
///         (other than FuncOp which is not returned)
static Operation *getLiftedScope(Operation *a, Operation *b) {
  DenseSet<Operation *> parentScopes;
  Operation *op = a;
  while (!isa<triton::FuncOp>(op)) {
    parentScopes.insert(op);
    op = op->getParentOp();
  }
  op = b;
  while (!isa<triton::FuncOp>(op)) {
    if (parentScopes.count(op))
      return op;
    op = op->getParentOp();
  }
  return nullptr;
}

/// Normalize a set of user operations to be at the same scope level.
/// Takes a set of user operations that may be at different nesting levels
/// and lifts them to be direct children of their lowest common ancestor scope.
/// This ensures all operations can be compared in program order within a block.
/// @param users Input set of user operations to normalize
/// @param userScopes Output set of operations lifted to the same scope level
/// @return success() if normalization succeeded, failure() otherwise
static LogicalResult getUserScopes(DenseSet<Operation *> &users,
                                   DenseSet<Operation *> &userScopes) {
  // Skip if users is empty (e.g., channels without valid operations)
  if (users.empty())
    return success();

  bool first = true;
  for (auto user : users) {
    if (first) {
      userScopes.insert(user);
    } else {
      // We may need to lift the scopes in userScopes.
      auto *scope = *(userScopes.begin());
      // If we can reach the same scope when lifting up "scope", return the
      // lifted "scope". Otherwise, we can lift up "user" to be in the same
      // scope as "scope", return scope.
      auto *sameLevel = getSameLevelOp(user, scope);
      if (sameLevel && sameLevel != scope) {
        // user stays unchanged, scope gets lifted to sameLevel.
        userScopes.clear();
        userScopes.insert(sameLevel);
        userScopes.insert(user);
      } else if (sameLevel) {
        // scope stays unchanged, user gets lifted.
        userScopes.insert(getSameLevelOp(scope, user));
      } else { // user and scope in different blocks, lift both.
        // find the parent scope that include both scope and user
        auto *parentScope = getLiftedScope(scope, user);
        userScopes.clear();
        if (!parentScope) {
          return failure();
        }
        Operation *op = user;
        Operation *liftedUser = nullptr;
        while (!isa<triton::FuncOp>(op)) {
          if (op->getParentOp() == parentScope) {
            liftedUser = op;
            break;
          }
          op = op->getParentOp();
        }
        if (!liftedUser) {
          return failure();
        }
        userScopes.insert(liftedUser);
        op = scope;
        Operation *liftedScope = nullptr;
        while (!isa<triton::FuncOp>(op)) {
          if (op->getParentOp() == parentScope) {
            liftedScope = op;
            break;
          }
          op = op->getParentOp();
        }
        if (!liftedScope) {
          return failure();
        }
        userScopes.insert(liftedScope);
      }
    }
    first = false;
  }
  return success();
}

/// Collect all live operations between the first and last user operations.
/// First normalizes users to the same scope level, then walks through all
/// operations (including nested ones) between the first and last user in
/// program order.
/// @param users Set of user operations to find live range for
/// @param liveOps Output vector to collect all live operations
/// @return success() if live ops were collected, failure() otherwise
static LogicalResult updateLiveOpsAcrossScopes(DenseSet<Operation *> &users,
                                               OperationListT &liveOps) {
  DenseSet<Operation *> userScopes;
  if (failed(getUserScopes(users, userScopes))) {
    return failure();
  }
  // Return early if no user scopes (e.g., when users is empty)
  if (userScopes.empty())
    return success();
  // Find the block that contains all users
  bool foundStart = false;
  auto *scope = *(userScopes.begin());
  if (!scope || !scope->getBlock()) {
    return success();
  }
  Operation *lastDst = nullptr;
  for (auto &op : scope->getBlock()->getOperations()) {
    if (userScopes.count(&op)) {
      lastDst = &op;
    }
  }
  for (auto &op : scope->getBlock()->getOperations()) {
    if (userScopes.count(&op) || foundStart) {
      foundStart = true;
      // Goes through nested regions.
      op.walk<WalkOrder::PostOrder>(
          [&](Operation *nestedOp) { liveOps.push_back(nestedOp); });
    }
    if (&op == lastDst) {
      break;
    }
  }
  return success();
}

namespace triton {

/// Memory planner for shared memory (SMEM) allocations in warp-specialized
/// kernels. Analyzes liveness of SMEM buffers based on channel producer/
/// consumer relationships and assigns buffer IDs and copy counts for
/// multi-buffering optimization. Buffers used in innermost loops with 2D+
/// shapes are candidates for multi-buffering with the specified numBuffers.
class MemoryPlanner : public MemoryPlannerBase {
public:
  MemoryPlanner(Operation *operation, Allocation *allocation,
                SmallVector<Channel *> *channels)
      : MemoryPlannerBase(operation, allocation, channels), lastBufferId(0) {}

  /// Get the next available buffer ID after running the planner.
  unsigned getLastBufferId() const { return lastBufferId; }

protected:
  DataChannelKind getChannelKind() const override {
    return DataChannelKind::SMEMPost;
  }

  Interval<size_t> computeLivenessInterval(Value value) override {
    auto liveOps = livenessForSmemChannel(value);
    if (liveOps.empty()) {
      return Interval<size_t>(0, 0);
    }
    return computeIntervalFromOps(liveOps);
  }

private:
  bool usersInInnermostLoop(Operation *alloc) {
    Channel *ch = findChannelForOp(alloc, *channels);
    if (!ch || ch->channelKind != getChannelKind()) {
      return false;
    }
    DenseSet<Operation *> users;
    (void)getAllAcutalUsersForChannel(ch, users, alloc);
    if (users.empty())
      return false;
    auto *first = *(users.begin());
    for (auto *user : users) {
      if (user->getBlock() != first->getBlock())
        return false;
    }
    auto parentLoop = first->getParentOfType<scf::ForOp>();
    if (!parentLoop)
      return false;
    return isInnermostLoop(parentLoop);
  }

  void getExplicitValueSize(Operation *op) {
    auto alloc = dyn_cast<ttg::LocalAllocOp>(op);
    if (!alloc || !alloc.isSharedMemoryAlloc())
      return;
    auto allocType = alloc.getType();
    int64_t numElems = 0;
    if (auto paddedEnc =
            dyn_cast<ttg::PaddedSharedEncodingAttr>(allocType.getEncoding())) {
      SmallVector<int64_t> unpaddedShape = ttg::getShapePerCTA(allocType);
      numElems = paddedEnc.getPaddedSize(unpaddedShape);
    } else {
      auto shapePerCTA = ttg::getAllocationShapePerCTA(allocType);
      numElems = product<int64_t>(shapePerCTA);
    }
    int64_t bytes = numElems * allocType.getElementTypeBitWidth() / 8;

    auto alignment = alloc.getAlignmentOrDefault();
    allocation->addBuffer<BufferT::BufferKind::Explicit>(alloc, bytes,
                                                         alignment);
  }

  void getValuesAndSizes() {
    operation->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { getExplicitValueSize(op); });
  }

  void resolveExplicitBufferLiveness(
      function_ref<Interval<size_t>(Value value)> getLiveness) {
    for (auto valueBufferIter : allocation->valueBuffer) {
      auto value = valueBufferIter.first;
      auto *buffer = valueBufferIter.second;
      bufferRange[buffer] = getLiveness(value);
      LLVM_DEBUG({
        llvm::dbgs() << "-- buffer " << buffer->id << "; value: ";
        value.dump();
      });
    }
  }

  OperationListT livenessForSmemChannel(Value value) {
    Operation *alloc = value.getDefiningOp();
    Channel *ch = findChannelForAlloc(value, *channels);
    ChannelPost *TheCh = nullptr;
    if (ch && ch->channelKind == DataChannelKind::SMEMPost) {
      TheCh = static_cast<ChannelPost *>(ch);
    }
    std::vector<Operation *> liveOps;
    DenseSet<Operation *> users;
    (void)getAllAcutalUsersForChannel(TheCh, users, alloc);
    (void)updateLiveOpsAcrossScopes(users, liveOps);
    return liveOps;
  }

  void resolveLiveness() {
    buildOperationIdMap();

    Liveness liveness(operation);
    auto getValueLivenessRange = [&](Value value) {
      Operation *defOp = value.getDefiningOp();
      LLVM_DEBUG({
        llvm::dbgs() << "-- getValueLivenessRange \n";
        value.dump();
      });
      auto liveOperations = livenessForSmemChannel(value);

      if (liveOperations.empty()) {
        return Interval<size_t>(0, 0);
      }

      auto minId = std::numeric_limits<size_t>::max();
      auto maxId = std::numeric_limits<size_t>::min();
      llvm::for_each(liveOperations, [&](Operation *liveOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "---- liveOp " << operationId[liveOp] << "\n");
        if (defOp && isa<mlir::triton::gpu::WarpSpecializeOp>(defOp)) {
          minId = 0;
          maxId = operationId.size();
          return;
        }
        if (operationId[liveOp] < minId) {
          minId = operationId[liveOp];
        }
        if ((operationId[liveOp] + 1) > maxId) {
          maxId = operationId[liveOp] + 1;
        }
      });
      return Interval(minId, maxId);
    };

    resolveExplicitBufferLiveness(getValueLivenessRange);
  }

public:
  LogicalResult run(unsigned numBuffers) override {
    getValuesAndSizes();
    resolveLiveness();

    // Dump SMEM buffer liveness using pre-calculated intervals
    // Create public data structures from private bufferRange
    llvm::MapVector<Allocation::BufferId, std::pair<Interval<size_t>, size_t>>
        bufferInfo;
    DenseMap<Allocation::BufferId, Operation *> bufferOwners;
    for (auto &bufferIter : bufferRange) {
      auto *buffer = bufferIter.first;
      auto &interval = bufferIter.second;
      bufferInfo[buffer->id] = std::make_pair(interval, buffer->size);
      bufferOwners[buffer->id] = buffer->owner;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "\n[MemoryPlanner] SMEM buffer liveness:\n";
      dumpSmemBufferLiveness(bufferInfo, bufferOwners, *channels, llvm::dbgs());
    });

    // Dump to file if TRITON_DUMP_WS_GRAPHS is set
    if (auto dumpDir = getGraphDumpDir()) {
      int id = graphDumpCounter++;
      std::string filename =
          *dumpDir + "/smem_liveness_" + std::to_string(id) + ".dot";
      std::ofstream ofs(filename);
      if (ofs.is_open()) {
        llvm::raw_os_ostream os(ofs);
        dumpSmemBufferLiveness(bufferInfo, bufferOwners, *channels, os);
        llvm::errs() << "Dumped SMEM liveness to: " << filename << "\n";
      }
    }

    unsigned bufferId = 0;
    int bufferIdInnermost = -1;

    DenseMap<int, Type> idTypes;
    for (auto bufferIter : bufferRange) {
      Operation *owner = bufferIter.first->owner;
      auto sAlloc = cast<ttg::LocalAllocOp>(owner);
      auto aType = sAlloc.getType();
      auto allocDescType = cast<triton::gpu::MemDescType>(aType);
      auto elemType = aType.getElementType();
      unsigned numD = 0;
      for (int shape : allocDescType.getShape()) {
        if (shape > 1)
          ++numD;
      }
      if (usersInInnermostLoop(owner) && numD >= 2) {
        if (bufferIdInnermost < 0) {
          bufferIdInnermost = bufferId;
          ++bufferId;
        }
        if (idTypes.count(bufferIdInnermost) == 0) {
          idTypes[bufferIdInnermost] = elemType;
        }
        if (idTypes[bufferIdInnermost] != elemType) {
          bufferIdInnermost = bufferId;
          idTypes[bufferIdInnermost] = elemType;
          ++bufferId;
        }
        owner->setAttr(
            "buffer.id",
            IntegerAttr::get(IntegerType::get(owner->getContext(), 32),
                             bufferIdInnermost));
        owner->setAttr(
            "buffer.copy",
            IntegerAttr::get(IntegerType::get(owner->getContext(), 32),
                             numBuffers));
      } else {
        if (idTypes.count(bufferId) == 0) {
          idTypes[bufferId] = elemType;
        }
        owner->setAttr(
            "buffer.id",
            IntegerAttr::get(IntegerType::get(owner->getContext(), 32),
                             bufferId));
        owner->setAttr(
            "buffer.copy",
            IntegerAttr::get(IntegerType::get(owner->getContext(), 32), 1));
        ++bufferId;
      }
    }
    lastBufferId = bufferId;
    return success();
  }

  void dumpBuffers() const {
    LDBG("Dump bufferRange: id size offset ---------");
    for (auto bufferIter : bufferRange) {
      llvm::dbgs() << "-- " << bufferIter.first->id << " "
                   << bufferIter.first->size << " " << bufferIter.first->offset;
      llvm::dbgs() << " interval " << bufferIter.second.start() << " "
                   << bufferIter.second.end() << "\n";
      bufferIter.first->owner->dump();
    }
  }

private:
  using BufferT = Allocation::BufferT;
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;

  BufferRangeMapT bufferRange;
  unsigned lastBufferId;
};
} // namespace triton

/// Collect all users of a TMEM allocation from its channel.
/// For operand D allocations (accumulator), collects all direct users.
/// For other allocations, delegates to getAllAcutalUsersForChannel.
/// @param TheCh The TMEM data channel post to get users for
/// @param users Output set to collect all user operations
/// @return success() if users were collected, failure() if TheCh is null
static LogicalResult getAllTmemUsers(ttng::TmemDataChannelPost *TheCh,
                                     DenseSet<Operation *> &users) {
  if (!TheCh) {
    return failure();
  }
  auto *allocOp = TheCh->getAllocOp();
  if (!allocOp) {
    return failure();
  }
  auto tmemAllocOp = llvm::dyn_cast<ttng::TMEMAllocOp>(allocOp);
  if (!tmemAllocOp) {
    return failure();
  }
  if (TheCh->isOperandD) {
    for (auto user : tmemAllocOp.getResult().getUsers()) {
      users.insert(user);
    }
  } else {
    if (failed(getAllAcutalUsersForChannel(TheCh, users))) {
      return failure();
    }
  }
  return success();
}

/// Compute the list of operations where a TMEM value is live.
/// Uses the channel's producer/consumer information to determine the live
/// range, which spans from the first user to the last user in program order.
/// @param value The TMEM allocation value to compute liveness for
/// @param channels The list of channels to search for the allocation's channel
/// @return Vector of operations where the value is live (empty on failure)
OperationListT livenessForTmemChannel(Value value,
                                      SmallVector<Channel *> &channels) {
  std::vector<Operation *> liveOps;
  // Find the channel for value in channels.
  Channel *ch = findChannelForAlloc(value, channels);
  if (!ch || ch->channelKind != DataChannelKind::TMEMPost) {
    return liveOps;
  }
  ttng::TmemDataChannelPost *TheCh =
      static_cast<ttng::TmemDataChannelPost *>(ch);
  DenseSet<Operation *> users;
  if (failed(getAllTmemUsers(TheCh, users))) {
    return liveOps;
  }
  (void)updateLiveOpsAcrossScopes(users, liveOps);

  return liveOps;
}

namespace triton {

/// Memory planner for tensor memory (TMEM) allocations in warp-specialized
/// kernels. Handles allocation of TMEM buffers used for Blackwell TCGen5MMA
/// operations. Computes liveness intervals based on channel relationships
/// and performs memory reuse optimization by allowing non-interfering buffers
/// to share TMEM space. Prioritizes operand D (accumulator) allocations and
/// larger buffers when assigning memory locations.
class MemoryPlannerTmem : public MemoryPlannerBase {
public:
  MemoryPlannerTmem(Operation *operation, Allocation *allocation,
                    SmallVector<Channel *> *channels)
      : MemoryPlannerBase(operation, allocation, channels) {}

protected:
  DataChannelKind getChannelKind() const override {
    return DataChannelKind::TMEMPost;
  }

  Interval<size_t> computeLivenessInterval(Value value) override {
    auto liveOps = livenessForTmemChannel(value, *channels);
    if (liveOps.empty()) {
      return Interval<size_t>(0, 0);
    }
    return computeIntervalFromOps(liveOps);
  }

private:
  using BufferT = Allocation::BufferT;
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;
  using GraphT = DenseMap<BufferT *, DenseSet<BufferT *>>;

  BufferRangeMapT bufferRange;

  // Data collected by collectTMemAllocsAndLiveness().
  SmallVector<triton::nvidia_gpu::TMEMAllocOp> allocs;
  SmallVector<BufferT *> buffers;
  Allocation tmemAllocation;
  DenseMap<Operation *, Interval<size_t>> allocToIntervals;
  DenseMap<Operation *, ttng::TMemAllocation> allocToSize;
  DenseMap<Operation *, ttng::TmemDataChannelPost *> allocToChannel;

  Interval<size_t> getLiveIntervals(Value value, Liveness &liveness,
                                    DenseMap<Operation *, size_t> &opId,
                                    SmallVector<Channel *> &chans) {
    auto liveOperations = livenessForTmemChannel(value, chans);
    SmallVector<Operation *> users(value.getUsers());
    while (!users.empty()) {
      Operation *user = users.pop_back_val();
      if (!isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp>(user))
        continue;
      auto usersLivness = livenessForTmemChannel(user->getResult(0), chans);
      liveOperations.insert(liveOperations.end(), usersLivness.begin(),
                            usersLivness.end());
      users.append(user->getResult(0).getUsers().begin(),
                   user->getResult(0).getUsers().end());
    }
    auto minId = std::numeric_limits<size_t>::max();
    auto maxId = std::numeric_limits<size_t>::min();
    std::for_each(liveOperations.begin(), liveOperations.end(),
                  [&](Operation *liveOp) {
                    if (opId[liveOp] < minId) {
                      minId = opId[liveOp];
                    }
                    if ((opId[liveOp] + 1) > maxId) {
                      maxId = opId[liveOp] + 1;
                    }
                  });
    return Interval(minId, maxId);
  }

  unsigned getLoopDepth(Operation *op) {
    unsigned depth = 0;
    auto pOp = op->getParentOfType<scf::ForOp>();
    while (pOp) {
      ++depth;
      pOp = pOp->getParentOfType<scf::ForOp>();
    }
    return depth;
  }

public:
  /// Phase 1: Collect TMEM allocs, compute liveness intervals, sort by
  /// priority, and create buffer objects. Must be called before
  /// allocateBuffers() or before verifying deserialized decisions.
  LogicalResult collectTMemAllocsAndLiveness() {
    Operation *parentOp = operation;
    buildOperationIdMap();
    parentOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto alloc = dyn_cast<triton::nvidia_gpu::TMEMAllocOp>(op)) {
        allocs.push_back(alloc);
      }
    });
    Liveness liveness(parentOp);
    for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
      ttng::TMEMAllocOp alloc = *it;
      Interval<size_t> liveInterval =
          getLiveIntervals(alloc, liveness, operationId, *channels);
      auto memDescType = alloc.getType();
      ttng::TMemAllocation allocSize = ttng::getTmemAllocSizes(memDescType);
      LLVM_DEBUG(alloc.dump());
      LDBG("tmem liveness: " << liveInterval.start() << " "
                             << liveInterval.end());
      LDBG("tmem allocSize: " << allocSize.numCols << " " << allocSize.numRows);

      ttng::TmemDataChannelPost *TheCh = nullptr;
      Channel *chBase = findChannelForAlloc(alloc, *channels);
      if (chBase && chBase->channelKind == DataChannelKind::TMEMPost) {
        TheCh = static_cast<ttng::TmemDataChannelPost *>(chBase);
      }
      allocToIntervals[alloc.getOperation()] = liveInterval;
      allocToSize.insert(
          {alloc.getOperation(),
           ttng::TMemAllocation(allocSize.numCols, allocSize.numRows)});
      allocToChannel[alloc.getOperation()] = TheCh;
    }
    // Sort allocs according to isOperandD, size, live interval.
    sort(allocs, [&](ttng::TMEMAllocOp a, ttng::TMEMAllocOp b) {
      Channel *aChBase = findChannelForAlloc(a, *channels);
      Channel *bChBase = findChannelForAlloc(b, *channels);
      ttng::TmemDataChannelPost *aCh = nullptr;
      ttng::TmemDataChannelPost *bCh = nullptr;
      if (aChBase && aChBase->channelKind == DataChannelKind::TMEMPost) {
        aCh = static_cast<ttng::TmemDataChannelPost *>(aChBase);
      }
      if (bChBase && bChBase->channelKind == DataChannelKind::TMEMPost) {
        bCh = static_cast<ttng::TmemDataChannelPost *>(bChBase);
      }
      // Handle null channels - put them at the end
      if (!aCh && !bCh)
        return false;
      if (!aCh)
        return false;
      if (!bCh)
        return true;
      if (aCh->isOperandD && !bCh->isOperandD)
        return true;
      if (bCh->isOperandD && !aCh->isOperandD)
        return false;
      auto iter1 = allocToSize.find(a.getOperation());
      auto iter2 = allocToSize.find(b.getOperation());
      if (iter1 == allocToSize.end() || iter2 == allocToSize.end())
        return false;
      if (iter1->second.numRows == iter2->second.numRows &&
          iter1->second.numCols == iter2->second.numCols) {
        // check live interval length and offset.
        auto intv1 = allocToIntervals[a.getOperation()];
        auto intv2 = allocToIntervals[b.getOperation()];
#if 0
        // larger interval has higher priority
        if (intv1.size() > intv2.size())
          return true;
        if (intv1.size() < intv2.size())
          return false;
#endif
        // early interval has higher priority
        if (intv1.start() < intv2.start())
          return true;
        if (intv1.start() > intv2.start())
          return false;
        // Equal intervals - maintain stable sort
        return false;
      }
      if (iter1->second.numRows == iter2->second.numRows)
        return iter1->second.numCols > iter2->second.numCols;
      if (iter1->second.numCols == iter2->second.numCols)
        return iter1->second.numRows > iter2->second.numRows;
      // Default comparison by total size
      return (iter1->second.numRows * iter1->second.numCols) >
             (iter2->second.numRows * iter2->second.numCols);
    });
    for (auto alloc : allocs) {
      // size is 0, alignment is default, offset is default
      tmemAllocation.addBuffer<BufferT::BufferKind::Explicit>(alloc, 0);
      BufferT *tBuf = tmemAllocation.valueBuffer[alloc];
      auto iter1 = allocToSize.find(alloc.getOperation());
      tBuf->rowSize = iter1->second.numRows;
      tBuf->colSize = iter1->second.numCols;
      tBuf->rowOffset = std::numeric_limits<size_t>::max();
      tBuf->colOffset = std::numeric_limits<size_t>::max();
      tBuf->isOwnerOfSpace = false;
      tBuf->reuseOwner = nullptr;
      buffers.emplace_back(tBuf);
    }

    // Dump TMEM buffer liveness using pre-calculated intervals
    LLVM_DEBUG({
      llvm::dbgs() << "\n[MemoryPlannerTmem] TMEM buffer liveness:\n";
      dumpTmemBufferLiveness(allocs, allocToIntervals, allocToSize,
                             allocToChannel, *channels, llvm::dbgs());
    });

    // Dump to file if TRITON_DUMP_WS_GRAPHS is set
    if (auto dumpDir = getGraphDumpDir()) {
      int id = graphDumpCounter++;
      std::string filename =
          *dumpDir + "/tmem_liveness_" + std::to_string(id) + ".dot";
      std::ofstream ofs(filename);
      if (ofs.is_open()) {
        llvm::raw_os_ostream os(ofs);
        dumpTmemBufferLiveness(allocs, allocToIntervals, allocToSize,
                               allocToChannel, *channels, os);
        llvm::errs() << "Dumped TMEM liveness to: " << filename << "\n";
      }
    }

    for (auto valueBufferIter : tmemAllocation.valueBuffer) {
      auto *buffer = valueBufferIter.second;
      Operation *alloc = valueBufferIter.first.getDefiningOp();
      bufferRange[buffer] = allocToIntervals[alloc];
    }
    return success();
  }

  /// Phase 2: Run the per-loop TMEM allocation heuristic using collected
  /// liveness data. collectTMemAllocsAndLiveness() must have been called first.
  LogicalResult allocateBuffers(unsigned bufferId) {
    Operation *parentOp = operation;
    SmallVector<Operation *> innermostLoops;
    parentOp->walk([&](Operation *subOp) {
      if (auto theForOp = dyn_cast<scf::ForOp>(subOp))
        if (isInnermostLoop(theForOp))
          innermostLoops.push_back(subOp);
    });
    DenseSet<Operation *> handledAllocs;
    unsigned ctrlIdx = 0;
    for (auto *ctrlOp : innermostLoops) {
      SmallVector<triton::nvidia_gpu::TMEMAllocOp> allocsForThisLoop;
      unsigned allocIdx = 0;
      auto ctrlInt = getIntervalForCtrlOp(ctrlOp);
      for (auto alloc : allocs) {
        auto allocInt = bufferRange.lookup(buffers[allocIdx]);
        ++allocIdx;
        if (!handledAllocs.count(alloc.getOperation()) &&
            (ctrlInt.intersects(allocInt) ||
             ctrlIdx == innermostLoops.size() - 1)) {
          allocsForThisLoop.push_back(alloc);
          handledAllocs.insert(alloc.getOperation());
        }
      }
      LDBG("run allocation on innermost loop "
           << allocsForThisLoop.size() << " allocs " << ctrlInt.start() << " "
           << ctrlInt.end());
      for (auto t : allocsForThisLoop)
        LLVM_DEBUG(t.getOperation()->dump());
      auto result = allocateTMemAllocs(
          allocsForThisLoop, buffers, // allocToIntervals,
          /*allocToSize,*/ allocToChannel, operationId, ctrlOp, bufferId);
      if (failed(result))
        return failure();
      bufferId = *result;
      ++ctrlIdx;
    }
    SmallVector<triton::nvidia_gpu::TMEMAllocOp> lastAllocs;
    for (auto alloc : allocs) {
      if (!handledAllocs.count(alloc)) {
        LDBG("Warning: allocation not handled in any innermost loop");
      }
    }
    if (!lastAllocs.empty()) {
      auto result = allocateTMemAllocs(lastAllocs, buffers, // allocToIntervals,
                                       /*allocToSize,*/ allocToChannel,
                                       operationId, nullptr, bufferId);
      if (failed(result))
        return failure();
      bufferId = *result;
    }
    return success();
  }

  /// Override pure virtual — combines both phases.
  LogicalResult run(unsigned bufferId) override {
    if (failed(collectTMemAllocsAndLiveness()))
      return failure();
    return allocateBuffers(bufferId);
  }

  // Accessors for verification of deserialized decisions.
  const SmallVector<triton::nvidia_gpu::TMEMAllocOp> &getAllocs() const {
    return allocs;
  }
  const DenseMap<Operation *, Interval<size_t>> &getAllocIntervals() const {
    return allocToIntervals;
  }
  const DenseMap<Operation *, ttng::TMemAllocation> &getAllocSizes() const {
    return allocToSize;
  }

  /// Verify that the buffer decisions (buffer.id, buffer.offset, buffer.copy)
  /// currently set on the TMEM alloc ops are feasible.  This checks hard
  /// constraints only — not heuristic quality signals (dependency chains,
  /// partition matching).
  ///
  /// Must be called after collectTMemAllocsAndLiveness() and after the
  /// decisions have been applied to the alloc ops (e.g. via
  /// readDecisionsFromFile).
  LogicalResult verifyBufferDecisions() {
    LDBG("verifyBufferDecisions: checking " << allocs.size() << " allocs");

    // Step 1: Read decisions from alloc op attributes and group by buffer.id.
    struct AllocInfo {
      Operation *op;
      int bufferId;
      int bufferCopy;
      int bufferOffset; // column offset
      size_t rowSize;
      size_t colSize;
      Interval<size_t> liveness;
    };

    SmallVector<AllocInfo> infos;
    for (auto alloc : allocs) {
      Operation *op = alloc.getOperation();
      auto idAttr = op->getAttrOfType<IntegerAttr>("buffer.id");
      auto copyAttr = op->getAttrOfType<IntegerAttr>("buffer.copy");
      auto offsetAttr = op->getAttrOfType<IntegerAttr>("buffer.offset");

      if (!idAttr) {
        return op->emitError("verifyBufferDecisions: missing buffer.id");
      }
      int bid = idAttr.getInt();
      int bcopy = copyAttr ? copyAttr.getInt() : 1;
      int boffset = offsetAttr ? offsetAttr.getInt() : 0;

      auto sizeIt = allocToSize.find(op);
      if (sizeIt == allocToSize.end()) {
        return op->emitError(
            "verifyBufferDecisions: alloc not found in allocToSize");
      }
      auto intIt = allocToIntervals.find(op);
      if (intIt == allocToIntervals.end()) {
        return op->emitError(
            "verifyBufferDecisions: alloc not found in allocToIntervals");
      }

      infos.push_back(
          {op, bid, bcopy, boffset, static_cast<size_t>(sizeIt->second.numRows),
           static_cast<size_t>(sizeIt->second.numCols), intIt->second});
      LDBG("  alloc buffer.id="
           << bid << " buffer.offset=" << boffset << " buffer.copy=" << bcopy
           << " rows=" << sizeIt->second.numRows
           << " cols=" << sizeIt->second.numCols << " live=["
           << intIt->second.start() << "," << intIt->second.end() << ")");
    }

    // Group by buffer.id.
    DenseMap<int, SmallVector<size_t>> idToIndices;
    for (size_t i = 0; i < infos.size(); ++i) {
      idToIndices[infos[i].bufferId].push_back(i);
    }

    // Step 2: Check buffer.copy consistency — TMEM buffers must be 1.
    for (auto &info : infos) {
      if (info.bufferCopy != 1) {
        return info.op->emitError("verifyBufferDecisions: TMEM buffer.copy "
                                  "must be 1, got ")
               << info.bufferCopy;
      }
    }

    // Step 3: Per buffer.id group checks.
    size_t totalRowUsage = 0;
    for (auto &[bid, indices] : idToIndices) {
      LDBG("  group buffer.id=" << bid << " with " << indices.size()
                                << " allocs");

      // 3a. Find the owner — the alloc with offset 0 and the largest colSize.
      size_t ownerIdx = indices[0];
      for (size_t idx : indices) {
        if (infos[idx].bufferOffset == 0 &&
            infos[idx].colSize >= infos[ownerIdx].colSize) {
          ownerIdx = idx;
        }
      }
      auto &owner = infos[ownerIdx];
      LDBG("    owner: rows=" << owner.rowSize << " cols=" << owner.colSize);

      // 3b. Row size consistency — all allocs in the group must match.
      for (size_t idx : indices) {
        if (infos[idx].rowSize != owner.rowSize) {
          return infos[idx].op->emitError(
                     "verifyBufferDecisions: row size mismatch in buffer.id=")
                 << bid << ": expected " << owner.rowSize << " got "
                 << infos[idx].rowSize;
        }
      }

      // 3c. Column capacity — offset + colSize must fit within owner.
      for (size_t idx : indices) {
        size_t endCol = infos[idx].bufferOffset + infos[idx].colSize;
        if (endCol > owner.colSize) {
          return infos[idx].op->emitError(
                     "verifyBufferDecisions: column overflow in buffer.id=")
                 << bid << ": offset(" << infos[idx].bufferOffset << ") + cols("
                 << infos[idx].colSize << ") = " << endCol << " > owner.cols("
                 << owner.colSize << ")";
        }
      }

      // 3d. Liveness × column conflict — within the same buffer.id, any two
      //     allocs with overlapping liveness must have non-overlapping columns.
      for (size_t i = 0; i < indices.size(); ++i) {
        for (size_t j = i + 1; j < indices.size(); ++j) {
          auto &a = infos[indices[i]];
          auto &b = infos[indices[j]];
          if (a.liveness.intersects(b.liveness)) {
            Interval<size_t> aCol(a.bufferOffset, a.bufferOffset + a.colSize);
            Interval<size_t> bCol(b.bufferOffset, b.bufferOffset + b.colSize);
            if (aCol.intersects(bCol)) {
              return a.op->emitError(
                         "verifyBufferDecisions: liveness+column conflict in "
                         "buffer.id=")
                     << bid << ": alloc at offset " << a.bufferOffset
                     << " (cols=" << a.colSize << ", live=["
                     << a.liveness.start() << "," << a.liveness.end()
                     << ")) conflicts with alloc at offset " << b.bufferOffset
                     << " (cols=" << b.colSize << ", live=["
                     << b.liveness.start() << "," << b.liveness.end() << "))";
            }
          }
        }
      }

      totalRowUsage += owner.rowSize;
    }

    // Step 4: Total row capacity — must not exceed 512.
    LDBG("  total TMEM row usage: " << totalRowUsage << " / 512");
    if (totalRowUsage > 512) {
      return allocs[0].emitError("verifyBufferDecisions: total TMEM row usage ")
             << totalRowUsage << " exceeds 512-row hardware limit";
    }

    LDBG("verifyBufferDecisions: all checks passed");
    return success();
  }

  // ---------------------------------------------------------------
  // Helper methods factored out of allocateTMemAllocs.
  // These operate on the member state populated by
  // collectTMemAllocsAndLiveness() and are reusable by
  // verifyBufferDecisions().
  // ---------------------------------------------------------------

  /// Check whether dstOp is in the forward SSA slice of srcOp,
  /// i.e. dstOp transitively uses a result of srcOp.  Also follows
  /// memory dependencies (local_store, tmem_store).
  static bool isDataDependent(Operation *srcOp, Operation *dstOp) {
    SmallVector<Operation *, 16> worklist;
    DenseSet<Operation *> visited;
    auto enqueueUsers = [&](Operation *op) {
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (visited.insert(user).second)
            worklist.push_back(user);
        }
      }
      if (isa<triton::gpu::LocalStoreOp>(op) ||
          isa<triton::nvidia_gpu::TMEMStoreOp>(op)) {
        for (Value operand : op->getOperands()) {
          if (isa<triton::gpu::MemDescType>(operand.getType())) {
            for (Operation *user : operand.getUsers()) {
              if (user != op && visited.insert(user).second)
                worklist.push_back(user);
            }
          }
        }
      }
    };
    enqueueUsers(srcOp);
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (op == dstOp)
        return true;
      enqueueUsers(op);
    }
    return false;
  }

  /// Check whether any TMEM channel forms a valid transitive dependency
  /// from consumerOp to producerOp across partition boundaries.
  bool hasTransitiveDependency(Operation *consumerOp, Operation *producerOp,
                               const SmallVector<AsyncTaskId> &consumerTasks,
                               const SmallVector<AsyncTaskId> &producerTasks) {
    DenseSet<AsyncTaskId> consumerSet(consumerTasks.begin(),
                                      consumerTasks.end());
    DenseSet<AsyncTaskId> producerSet(producerTasks.begin(),
                                      producerTasks.end());
    for (auto *ch : *channels) {
      if (ch->channelKind != DataChannelKind::TMEMPost)
        continue;
      auto *tmemCh = static_cast<ttng::TmemDataChannelPost *>(ch);
      Operation *chSrcOp = tmemCh->getSrcOp();
      Operation *chDstOp = tmemCh->getDstOp();
      auto chSrcTasks = getAsyncTaskIds(chSrcOp);
      auto chDstTasks = getAsyncTaskIds(chDstOp);
      bool srcMatch = llvm::any_of(
          chSrcTasks, [&](AsyncTaskId t) { return consumerSet.contains(t); });
      bool dstMatch = llvm::any_of(
          chDstTasks, [&](AsyncTaskId t) { return producerSet.contains(t); });
      if (!srcMatch || !dstMatch)
        continue;
      if (isDataDependent(consumerOp, chSrcOp) &&
          isDataDependent(chDstOp, producerOp))
        return true;
    }
    return false;
  }

  /// Check if src and dst are along a dependency chain (same partition,
  /// direct data dep, or transitive dep via a channel).
  bool alongDependencyChain(Operation *src, Operation *dst,
                            unsigned depChainCondition) {
    auto *srcCh = allocToChannel[src];
    auto *dstCh = allocToChannel[dst];
    Operation *consumerOp = srcCh->getDstOp();
    Operation *producerOp = dstCh->getSrcOp();
    auto consumerTasks = getAsyncTaskIds(consumerOp);
    auto producerTasks = getAsyncTaskIds(producerOp);
    if (consumerTasks == producerTasks)
      return true;
    if (isDataDependent(consumerOp, producerOp))
      return true;
    if (hasTransitiveDependency(consumerOp, producerOp, consumerTasks,
                                producerTasks))
      return true;
    return false;
  }

  /// Check if alloc's liveness intersects the given control op's interval.
  bool isInSameLoop(BufferT *alloc, Operation *ctrlOp) {
    if (ctrlOp) {
      auto ctrlInt = getIntervalForCtrlOp(ctrlOp);
      return bufferRange[alloc].intersects(ctrlInt);
    }
    return false;
  }

  /// Get combined async task IDs from all users of the alloc's channel.
  SmallVector<AsyncTaskId> getCombinedTasks(BufferT *alloc) {
    Channel *chBase = findChannelForOp(alloc->owner, *channels);
    ttng::TmemDataChannelPost *TheCh = nullptr;
    if (chBase && chBase->channelKind == DataChannelKind::TMEMPost) {
      TheCh = static_cast<ttng::TmemDataChannelPost *>(chBase);
    }
    SmallVector<AsyncTaskId> combinedTasks;
    if (!TheCh) {
      return combinedTasks;
    }
    DenseSet<Operation *> users;
    if (failed(getAllTmemUsers(TheCh, users))) {
      return combinedTasks;
    }
    DenseSet<AsyncTaskId> combinedSet;
    for (auto *user : users) {
      auto asyncTasksVec = getAsyncTaskIds(user);
      for (auto t : asyncTasksVec) {
        if (!combinedSet.count(t)) {
          combinedSet.insert(t);
          combinedTasks.push_back(t);
        }
      }
    }
    std::sort(combinedTasks.begin(), combinedTasks.end());
    return combinedTasks;
  }

  /// Check partition matching between two buffers with configurable
  /// strictness: 0=any, 1=src/dst match, 2=combined tasks match.
  bool samePartition(BufferT *alloc, BufferT *cand,
                     unsigned partitionCondition) {
    if (partitionCondition == 0)
      return true;
    if (partitionCondition == 1) {
      auto *srcCh = allocToChannel[alloc->owner];
      auto *dstCh = allocToChannel[cand->owner];
      auto dstChPart = getAsyncTaskIds(dstCh->getSrcOp());
      auto srcChPart = getAsyncTaskIds(srcCh->getDstOp());
      LLVM_DEBUG(llvm::dbgs() << "Check partitions\n");
      for (auto t : dstChPart) {
        LLVM_DEBUG(llvm::dbgs() << t << " ");
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
      for (auto t : srcChPart) {
        LLVM_DEBUG(llvm::dbgs() << t << " ");
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
      return getAsyncTaskIds(dstCh->getSrcOp()) ==
             getAsyncTaskIds(srcCh->getDstOp());
    }
    auto aTasks = getCombinedTasks(alloc);
    auto bTasks = getCombinedTasks(cand);
    LLVM_DEBUG(llvm::dbgs() << "Check combined partitions\n");
    for (auto t : aTasks) {
      LLVM_DEBUG(llvm::dbgs() << t << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
    for (auto t : bTasks) {
      LLVM_DEBUG(llvm::dbgs() << t << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
    return aTasks == bTasks;
  }

  /// Verify no column/liveness overlap between cand and other buffers
  /// reusing reuseOwner at the given colOffset.
  bool checkOtherReuses(BufferT *cand, BufferT *reuseOwner, size_t colOffset) {
    for (auto *alloc : buffers) {
      if (!alloc->isOwnerOfSpace && alloc->reuseOwner == reuseOwner) {
        Interval candSizeRange = {colOffset, colOffset + cand->colSize};
        Interval allocSizeRange = {alloc->colOffset,
                                   alloc->colOffset + alloc->colSize};
        if (bufferRange[alloc].intersects(bufferRange[cand]) &&
            allocSizeRange.intersects(candSizeRange)) {
          LLVM_DEBUG({
            LDBG("checkOtherReuses conflict "
                 << colOffset << " " << alloc->colOffset << " " << cand->colSize
                 << " " << alloc->colSize);
            alloc->owner->dump();
          });
          return false;
        }
      }
    }
    return true;
  }

  /// Check if cand's liveness overlaps with ALL currently allocated buffers.
  bool allInterfere(BufferT *cand) {
    for (auto *alloc : buffers) {
      if (alloc->rowOffset != std::numeric_limits<size_t>::max()) {
        if (!bufferRange[alloc].intersects(bufferRange[cand]))
          return false;
      }
    }
    return true;
  }

  /// Compute the current total row usage across all allocated buffers.
  size_t computeTotalRowUsage() {
    size_t maxRowOffset = 0;
    for (auto *alloc : buffers) {
      if (alloc->rowOffset != std::numeric_limits<size_t>::max()) {
        maxRowOffset =
            std::max(maxRowOffset, alloc->rowOffset + alloc->rowSize);
      }
    }
    return maxRowOffset;
  }

  /// Check if allocating cand at maxRowOffset would exceed 512 rows.
  /// If allocate is true, also assigns the row/col offset and buffer.id.
  bool allocateNewSpace(BufferT *cand, bool allocate, unsigned &bufferId) {
    size_t maxRowOffset = 0;
    for (auto *alloc : buffers) {
      if (alloc->rowOffset != std::numeric_limits<size_t>::max()) {
        maxRowOffset =
            std::max(maxRowOffset, alloc->rowOffset + alloc->rowSize);
        LLVM_DEBUG({
          LDBG("\nbuffer is allocated "
               << alloc->rowOffset << " " << alloc->rowSize << " "
               << alloc->colOffset << " " << alloc->isOwnerOfSpace);
          alloc->owner->dump();
        });
      }
    }
    if (allocate) {
      cand->rowOffset = maxRowOffset;
      cand->colOffset = 0;
      cand->isOwnerOfSpace = true;
      cand->reuseOffset = 0;
      cand->reuseOwner = cand;
      cand->owner->setAttr(
          "buffer.id",
          IntegerAttr::get(IntegerType::get(cand->owner->getContext(), 32),
                           bufferId));
      ++bufferId;
    }
    if (maxRowOffset + cand->rowSize > 512)
      return false;
    return true;
  }

  /// Look up the BufferT for a given op.
  BufferT *getBuffer(Operation *candAlloc) {
    for (auto *alloc : buffers) {
      if (alloc->owner == candAlloc)
        return alloc;
    }
    return nullptr;
  }

  /// Read the buffer.id attribute from an op.
  static int getBufferId(Operation *op) {
    auto stageAttr = op->getAttrOfType<IntegerAttr>("buffer.id");
    return stageAttr.getInt();
  }

  // ---------------------------------------------------------------
  // allocateTMemAllocs — main allocation heuristic loop.
  // Uses the helper methods above.
  // ---------------------------------------------------------------

  FailureOr<unsigned> allocateTMemAllocs(
      SmallVector<triton::nvidia_gpu::TMEMAllocOp> &allocs,
      SmallVector<BufferT *> &buffers,
      DenseMap<Operation *, ttng::TmemDataChannelPost *> &allocToChannel,
      DenseMap<Operation *, size_t> &operationId, Operation *ctrlOp,
      unsigned bufferId) {
    // Lambdas that depend on the per-call ctrlOp parameter.
    auto sameLoop = [&](BufferT *alloc) -> bool {
      return isInSameLoop(alloc, ctrlOp);
    };
    auto findUsesInCtrlOp = [&](BufferT *buf, BufferT *cand) -> size_t {
      assert(buf->colOffset == 0);
      size_t maxColOffset = 0;
      for (auto *alloc : buffers) {
        if (!alloc->isOwnerOfSpace && alloc->reuseOwner == buf->reuseOwner &&
            alloc != buf &&
            (sameLoop(alloc) ||
             bufferRange[alloc].intersects(bufferRange[cand]))) {
          maxColOffset =
              std::max(maxColOffset, alloc->colOffset + alloc->colSize);
        }
      }
      return maxColOffset;
    };
    auto findReuseSpace = [&](BufferT *cand, BufferT *reuseOwner,
                              unsigned depChainCondition) -> size_t {
      // Pass 1: find the first free column offset after all live reusers.
      size_t maxColOffset = 0;
      for (auto *alloc : buffers) {
        if (alloc->isOwnerOfSpace || alloc->reuseOwner != reuseOwner)
          continue;
        if (!bufferRange[alloc].intersects(bufferRange[cand]))
          continue;
        maxColOffset =
            std::max(alloc->colOffset + alloc->colSize, maxColOffset);
      }
      LDBG("findReuseSpace first pass maxColOffset " << maxColOffset);
      if (maxColOffset + cand->colSize <= reuseOwner->colSize)
        return maxColOffset;
      // Pass 2: if the owner is in a different loop, try to overlap with
      // an existing reuser at column 0 that is in the same loop as cand
      // and connected by a dependency chain.
      if (sameLoop(reuseOwner))
        return std::numeric_limits<size_t>::max();
      for (auto *alloc : buffers) {
        if (alloc->isOwnerOfSpace || alloc->reuseOwner != reuseOwner)
          continue;
        if (alloc->colOffset != 0 || !sameLoop(alloc))
          continue;
        bool onChain = bufferRange[cand].start() < bufferRange[alloc].start()
                           ? alongDependencyChain(cand->owner, alloc->owner,
                                                  depChainCondition)
                           : alongDependencyChain(alloc->owner, cand->owner,
                                                  depChainCondition);
        if (!onChain)
          continue;
        auto tOffset = findUsesInCtrlOp(alloc, cand);
        LLVM_DEBUG({
          LDBG("findUsesInCtrlOp returns " << tOffset);
          alloc->owner->dump();
        });
        if (tOffset + cand->colSize <= alloc->colSize)
          return tOffset;
      }
      return std::numeric_limits<size_t>::max();
    };
    auto firstReuseOfBuffer = [&](BufferT *cand) -> bool {
      for (auto alloc : allocs) {
        if (cand->owner == alloc.getOperation()) {
          break;
        }
        auto *allocBuf = getBuffer(alloc.getOperation());
        if (allocBuf->reuseOwner == cand->reuseOwner)
          return false;
      }
      return true;
    };
    auto findReuseChannel = [&](BufferT *cand, unsigned partitionCondition,
                                unsigned depChainCondition) -> BufferT * {
      for (auto *alloc : buffers) {
        if (!alloc->isOwnerOfSpace)
          continue;
        LLVM_DEBUG({
          LDBG("check to reuse buffer owned by " << bufferRange[alloc].start()
                                                 << " "
                                                 << bufferRange[alloc].end());
          alloc->owner->dump();
        });
        // Skip allocs not live-range disjoint or too small to fit cand.
        if (bufferRange[alloc].intersects(bufferRange[cand]))
          continue;
        if (alloc->colSize < cand->colSize)
          continue;
        // Reuse eligibility depends on whether alloc and cand are in the
        // same loop.  Inside the same loop, reuse requires a dependency
        // chain between the two (the earlier-live one must feed into the
        // later-live one) so the compiler can guarantee non-overlapping
        // lifetimes.  Across different loops, a weaker partition-matching
        // check suffices because the loops are sequenced by control flow.
        bool inSameLoop = sameLoop(alloc);
        bool canReuse =
            inSameLoop ? (bufferRange[cand].start() < bufferRange[alloc].start()
                              ? alongDependencyChain(cand->owner, alloc->owner,
                                                     depChainCondition)
                              : alongDependencyChain(alloc->owner, cand->owner,
                                                     depChainCondition))
                       : samePartition(alloc, cand, partitionCondition);
        if (!canReuse) {
          LLVM_DEBUG({
            LDBG("can't reuse owner — partition/dep check failed");
            alloc->owner->dump();
          });
          continue;
        }
        auto colOffset = findReuseSpace(cand, alloc, depChainCondition);
        if (colOffset == std::numeric_limits<size_t>::max()) {
          LDBG("-- findReuseSpace fails");
          continue;
        }
        if (!checkOtherReuses(cand, alloc, colOffset)) {
          LDBG("-- checkOtherReuses fails");
          continue;
        }
        cand->isOwnerOfSpace = false;
        cand->rowOffset = alloc->rowOffset;
        cand->colOffset = colOffset;
        cand->reuseOwner = alloc;
        LLVM_DEBUG({
          LDBG("set offset to " << cand->rowOffset << " " << cand->colOffset
                                << " sameLoop " << inSameLoop << ":");
          cand->owner->dump();
        });
        return alloc;
      }
      return nullptr;
    };

    // Main allocation loop.
    DenseMap<Operation *, Interval<size_t>> bufferSet;
    Operation *candidateAlloc = nullptr;
    SmallVector<Operation *> allocOrder;
    for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
      ttng::TMEMAllocOp alloc = *it;
      auto *candBuf = getBuffer(alloc.getOperation());
      LLVM_DEBUG({
        LDBG("\ntry tmem allocation size "
             << candBuf->rowSize << " " << bufferRange[candBuf].start() << " "
             << bufferRange[candBuf].end());
        alloc.getOperation()->dump();
      });
      if (allInterfere(candBuf)) {
        LDBG("\nallInterfere");
        bool hasSpace = allocateNewSpace(candBuf, true, bufferId);
        if (!hasSpace) {
          return alloc.emitError("can't find tmem space: no new space for "
                                 "tmem alloc when all buffers interfere");
        }
      } else {
        // Two-pass reuse search with decreasing strictness:
        //   partitionCondition=2: combined tasks of src+dst must match
        //                         (strictest — both sides touch same
        //                         partitions)
        //   partitionCondition=1: channel endpoints must match
        //                         (relaxed — only checks src.dst vs dst.src
        //                         tasks)
        //   depChainCondition=1:  currently unused inside alongDependencyChain,
        //                         reserved for future multi-level dependency
        //                         checks
        auto *reuseBuf = findReuseChannel(candBuf, 2 /*partitionCondition*/,
                                          1 /*depChainCondition*/);
        if (!reuseBuf)
          reuseBuf = findReuseChannel(candBuf, 1 /*partitionCondition*/,
                                      1 /*depChainCondition*/);
        if (reuseBuf) {
          alloc.getOperation()->setAttr(
              "buffer.id",
              IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                               getBufferId(reuseBuf->owner)));
          alloc.getOperation()->setAttr(
              "buffer.offset",
              IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                               candBuf->colOffset));
        } else {
          if (allocateNewSpace(candBuf, false, bufferId))
            allocateNewSpace(candBuf, true, bufferId);
          else {
            return alloc.emitError(
                "can't find tmem space: failed to allocate new space");
          }
        }
      }
      LLVM_DEBUG({
        LDBG("\ntmem allocation " << candBuf->rowOffset << " "
                                  << candBuf->colOffset << " "
                                  << candBuf->isOwnerOfSpace);
        alloc.getOperation()->dump();
      });

      // FIXME: heuristics
      alloc.getOperation()->setAttr(
          "buffer.copy",
          IntegerAttr::get(IntegerType::get(alloc->getContext(), 32), 1));
    }
    return bufferId;
  }
};
} // namespace triton

//===----------------------------------------------------------------------===//
// Buffer Decision Serialization/Deserialization
//===----------------------------------------------------------------------===//

struct BufferDecision {
  unsigned channelId;
  unsigned bufferId;
  unsigned bufferCopy;
  unsigned bufferOffset;

  bool operator==(const BufferDecision &other) const {
    return channelId == other.channelId && bufferId == other.bufferId &&
           bufferCopy == other.bufferCopy && bufferOffset == other.bufferOffset;
  }

  bool operator!=(const BufferDecision &other) const {
    return !(*this == other);
  }
};

struct BufferDecisionList {
  SmallVector<BufferDecision> decisions;

  bool operator==(const BufferDecisionList &other) const {
    if (decisions.size() != other.decisions.size())
      return false;
    for (size_t i = 0; i < decisions.size(); ++i) {
      if (decisions[i] != other.decisions[i])
        return false;
    }
    return true;
  }

  bool operator!=(const BufferDecisionList &other) const {
    return !(*this == other);
  }
};

static void sortChannelsByProgramOrder(SmallVector<Channel *> &channels) {
  llvm::sort(channels, [](Channel *a, Channel *b) {
    Operation *allocA = a->getAllocOp();
    Operation *allocB = b->getAllocOp();
    if (!allocA || !allocB)
      return a->uniqID < b->uniqID;
    return allocA->isBeforeInBlock(allocB) ||
           (allocA->getBlock() != allocB->getBlock() && a->uniqID < b->uniqID);
  });
}

static BufferDecision extractBufferDecision(Channel *ch) {
  BufferDecision decision;
  decision.channelId = ch->uniqID;
  decision.bufferId = 0;
  decision.bufferCopy = 1;
  decision.bufferOffset = 0;

  Operation *allocOp = ch->getAllocOp();
  if (!allocOp)
    return decision;

  if (auto attr = allocOp->getAttrOfType<IntegerAttr>("buffer.id"))
    decision.bufferId = attr.getInt();
  if (auto attr = allocOp->getAttrOfType<IntegerAttr>("buffer.copy"))
    decision.bufferCopy = attr.getInt();
  if (auto attr = allocOp->getAttrOfType<IntegerAttr>("buffer.offset"))
    decision.bufferOffset = attr.getInt();

  return decision;
}

static void applyBufferDecision(Channel *ch, const BufferDecision &decision) {
  Operation *allocOp = ch->getAllocOp();
  if (!allocOp)
    return;

  auto ctx = allocOp->getContext();
  auto i32Type = IntegerType::get(ctx, 32);

  allocOp->setAttr("buffer.id", IntegerAttr::get(i32Type, decision.bufferId));
  allocOp->setAttr("buffer.copy",
                   IntegerAttr::get(i32Type, decision.bufferCopy));
  allocOp->setAttr("buffer.offset",
                   IntegerAttr::get(i32Type, decision.bufferOffset));
}

BufferDecisionList serializeBufferDecisions(SmallVector<Channel *> &channels) {
  SmallVector<Channel *> sortedChannels(channels.begin(), channels.end());
  sortChannelsByProgramOrder(sortedChannels);

  BufferDecisionList result;
  for (Channel *ch : sortedChannels) {
    result.decisions.push_back(extractBufferDecision(ch));
  }
  return result;
}

LogicalResult deserializeBufferDecisions(SmallVector<Channel *> &channels,
                                         const BufferDecisionList &decisions) {
  SmallVector<Channel *> sortedChannels(channels.begin(), channels.end());
  sortChannelsByProgramOrder(sortedChannels);

  if (sortedChannels.size() != decisions.decisions.size()) {
    LDBG("deserialize failed: channel count mismatch ("
         << sortedChannels.size() << " vs " << decisions.decisions.size()
         << ")");
    return failure();
  }

  for (size_t i = 0; i < sortedChannels.size(); ++i) {
    Channel *ch = sortedChannels[i];
    const BufferDecision &decision = decisions.decisions[i];

    if (ch->uniqID != decision.channelId) {
      LDBG("deserialize failed: channel id mismatch at index "
           << i << " (" << ch->uniqID << " vs " << decision.channelId << ")");
      return failure();
    }

    applyBufferDecision(ch, decision);
  }
  return success();
}

std::string serializeBufferDecisionsToString(const BufferDecisionList &list) {
  llvm::json::Array decisionsArray;
  for (const auto &decision : list.decisions) {
    llvm::json::Object obj;
    obj["channelId"] = static_cast<int64_t>(decision.channelId);
    obj["bufferId"] = static_cast<int64_t>(decision.bufferId);
    obj["bufferCopy"] = static_cast<int64_t>(decision.bufferCopy);
    obj["bufferOffset"] = static_cast<int64_t>(decision.bufferOffset);
    decisionsArray.push_back(std::move(obj));
  }

  llvm::json::Object root;
  root["version"] = 1;
  root["decisions"] = std::move(decisionsArray);

  std::string result;
  llvm::raw_string_ostream os(result);
  os << llvm::json::Value(std::move(root));
  return result;
}

std::optional<BufferDecisionList>
deserializeBufferDecisionsFromString(StringRef jsonStr) {
  auto parsed = llvm::json::parse(jsonStr);
  if (!parsed) {
    LDBG("JSON parse error: " << llvm::toString(parsed.takeError()));
    return std::nullopt;
  }

  auto *root = parsed->getAsObject();
  if (!root) {
    LDBG("JSON root is not an object");
    return std::nullopt;
  }

  auto version = root->getInteger("version");
  if (!version || *version != 1) {
    LDBG("Unsupported version: " << (version ? *version : -1));
    return std::nullopt;
  }

  auto *decisionsArray = root->getArray("decisions");
  if (!decisionsArray) {
    LDBG("Missing 'decisions' array");
    return std::nullopt;
  }

  BufferDecisionList result;
  for (const auto &item : *decisionsArray) {
    auto *obj = item.getAsObject();
    if (!obj) {
      LDBG("Decision item is not an object");
      return std::nullopt;
    }

    BufferDecision decision;
    auto channelId = obj->getInteger("channelId");
    auto bufferId = obj->getInteger("bufferId");
    auto bufferCopy = obj->getInteger("bufferCopy");
    auto bufferOffset = obj->getInteger("bufferOffset");

    if (!channelId.has_value() || !bufferId.has_value() ||
        !bufferCopy.has_value() || !bufferOffset.has_value()) {
      LDBG("Missing required field in decision");
      return std::nullopt;
    }

    decision.channelId = static_cast<unsigned>(*channelId);
    decision.bufferId = static_cast<unsigned>(*bufferId);
    decision.bufferCopy = static_cast<unsigned>(*bufferCopy);
    decision.bufferOffset = static_cast<unsigned>(*bufferOffset);
    result.decisions.push_back(decision);
  }

  return result;
}

//===----------------------------------------------------------------------===//

LogicalResult writeDecisionsToFile(SmallVector<Channel *> &channels,
                                   StringRef filePath) {
  BufferDecisionList decisions = serializeBufferDecisions(channels);
  std::string json = serializeBufferDecisionsToString(decisions);

  std::error_code ec;
  llvm::raw_fd_ostream os(filePath, ec);
  if (ec) {
    LDBG("Failed to open file for writing: " << filePath << " - "
                                             << ec.message());
    return failure();
  }

  os << json;
  LDBG("Wrote buffer decisions to: " << filePath);
  return success();
}

LogicalResult readDecisionsFromFile(SmallVector<Channel *> &channels,
                                    StringRef filePath) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (!bufferOrErr) {
    LDBG("Failed to open file for reading: "
         << filePath << " - " << bufferOrErr.getError().message());
    return failure();
  }

  StringRef content = (*bufferOrErr)->getBuffer();
  auto decisions = deserializeBufferDecisionsFromString(content);
  if (!decisions) {
    LDBG("Failed to parse decisions from file: " << filePath);
    return failure();
  }

  if (failed(deserializeBufferDecisions(channels, *decisions))) {
    LDBG("Failed to apply decisions from file: " << filePath);
    return failure();
  }

  LDBG("Applied buffer decisions from: " << filePath);
  return success();
}

LogicalResult doMemoryPlanner(triton::FuncOp &funcOp, unsigned numBuffers,
                              StringRef readDecisionFile = "",
                              StringRef writeDecisionFile = "") {

  // Step 0: reorder ops within pipelined loops so that within each stage,
  // ops with smaller cluster IDs appear before ops with larger cluster IDs.
  reorderOpsBySchedule(funcOp);

  // Step 1: collect all communications between producers and consumers.
  SmallVector<std::unique_ptr<Channel>> channelsOrigin;
  collectPostChannels(channelsOrigin, funcOp);
  SmallVector<Channel *> channels;
  for (const auto &c : channelsOrigin) {
    channels.push_back(c.get());
  }
  if (channels.empty()) {
    return success();
  }
  for (auto *ch : channels) {
    LLVM_DEBUG({
      LDBG("\nchannel with allocOp: " << static_cast<int>(ch->channelKind)
                                      << " " << ch->uniqID << " ");
      ch->getAllocOp()->dump();
    });
    if (ch->channelKind == DataChannelKind::TMEMPost) {
      ttng::TmemDataChannelPost *TheCh =
          static_cast<ttng::TmemDataChannelPost *>(ch);
      LDBG("channel type TMEM" << TheCh->isOperandD << " "
                               << TheCh->isOperandDNoAcc);
    }
  }

  // Step 1.5: Identify channels whose source and destination(s) are in
  // different pipeline stages.  This information is useful for SWP-aware
  // memory planning (e.g. determining multi-buffer depth).
  for (auto *ch : channels) {
    Operation *srcOp = ch->getSrcOp();
    if (!srcOp)
      continue;
    int srcStage = getLoopStage(srcOp);
    if (srcStage < 0)
      continue;

    SmallVector<Operation *> dstOps;
    // TmemDataChannelPost::getDstOps asserts !isOperandD, so for TMEM
    // accumulator channels we fall back to getDstOp/getDstOpLast.
    bool isTmemOperandD = false;
    if (ch->channelKind == DataChannelKind::TMEMPost) {
      auto *tmemCh = static_cast<ttng::TmemDataChannelPost *>(ch);
      isTmemOperandD = tmemCh->isOperandD;
    }
    if (isTmemOperandD) {
      if (Operation *dst = ch->getDstOp())
        dstOps.push_back(dst);
    } else {
      ch->getDstOps(dstOps);
      if (dstOps.empty()) {
        if (Operation *dst = ch->getDstOp())
          dstOps.push_back(dst);
      }
    }

    for (Operation *dstOp : dstOps) {
      int dstStage = getLoopStage(dstOp);
      if (dstStage < 0)
        continue;
      if (srcStage != dstStage) {
        LDBG("cross-stage channel " << ch->uniqID << ": src stage=" << srcStage
                                    << " dst stage=" << dstStage);
        LLVM_DEBUG({
          llvm::dbgs() << "  src: ";
          srcOp->dump();
          llvm::dbgs() << "  dst: ";
          dstOp->dump();
          if (ch->getAllocOp()) {
            llvm::dbgs() << "  alloc: ";
            ch->getAllocOp()->dump();
          }
        });
      }
    }
  }

  // Step 2: figure out smem/tmem sizes and liveness.
  // If two buffers are sharing a multi-staged alloc, the liveness can overlap,
  // otherwise, the liveness can't overlap.
  Allocation allocation;
  triton::MemoryPlanner planner(funcOp, &allocation, &channels);
  if (failed(planner.run(numBuffers)))
    return failure();
  unsigned bufferId = planner.getLastBufferId();
  LLVM_DEBUG(funcOp.dump());
  LLVM_DEBUG(planner.dumpBuffers());

  // Dump combined key ops + channel graph (side by side visualization)
  // Note: Placed before MemoryPlannerTmem to visualize state even if TMEM
  // allocation fails
  LLVM_DEBUG({
    llvm::dbgs() << "\n[doMemoryPlanner] Combined visualization:\n";
    dumpCombinedGraph(channelsOrigin, funcOp, llvm::dbgs());
  });

  // Dump to file if TRITON_DUMP_WS_GRAPHS is set
  if (auto dumpDir = getGraphDumpDir()) {
    int id = graphDumpCounter++;
    std::string filename =
        *dumpDir + "/combined_graph_" + std::to_string(id) + ".dot";
    std::ofstream ofs(filename);
    if (ofs.is_open()) {
      llvm::raw_os_ostream os(ofs);
      dumpCombinedGraph(channelsOrigin, funcOp, os);
      llvm::errs() << "Dumped combined graph to: " << filename << "\n";
    }
  }

  // Step 3: Collect TMEM allocs and liveness BEFORE the decision branch so
  // that we have liveness data available for verifying deserialized decisions.
  Allocation tmemAllocation;
  triton::MemoryPlannerTmem tmemPlanner(funcOp, &tmemAllocation, &channels);
  if (failed(tmemPlanner.collectTMemAllocsAndLiveness()))
    return failure();

  // If a read decision file is provided (via argument or env var), apply
  // decisions from file instead of running the planner.
  StringRef effectiveReadFile = readDecisionFile;
  std::string envReadFile;
  if (effectiveReadFile.empty()) {
    if (const char *envVal = std::getenv("TRITON_WS_DECISION_FILE")) {
      envReadFile = envVal;
      effectiveReadFile = envReadFile;
    }
  }
  if (!effectiveReadFile.empty()) {
    if (failed(readDecisionsFromFile(channels, effectiveReadFile)))
      return failure();
    LDBG("Applied decisions from: " << effectiveReadFile);
    if (failed(tmemPlanner.verifyBufferDecisions()))
      return failure();
    return success();
  }

  // Step 4: Run TMEM allocation heuristic.
  if (failed(tmemPlanner.allocateBuffers(bufferId)))
    return failure();

  // If a write decision file is provided, serialize decisions to file.
  if (!writeDecisionFile.empty()) {
    if (failed(writeDecisionsToFile(channels, writeDecisionFile))) {
      return failure();
    }
  }

  // allocateTMem(funcOp, channels, bufferId);
  return success();
}

#define GEN_PASS_DEF_NVGPUTESTWSMEMORYPLANNER
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSMemoryPlannerPass
    : public impl::NVGPUTestWSMemoryPlannerBase<NVGPUTestWSMemoryPlannerPass> {
public:
  using impl::NVGPUTestWSMemoryPlannerBase<
      NVGPUTestWSMemoryPlannerPass>::NVGPUTestWSMemoryPlannerBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numBuffers >= 1 || !readDecisionFile.empty()) {
      if (failed(doMemoryPlanner(funcOp, numBuffers, readDecisionFile,
                                 writeDecisionFile)))
        signalPassFailure();
    }
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
