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
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
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

/// Given a value, walk backwards through the SSA def-use chain, passing
/// through "transparent" ops that don't generate new data (split, reshape,
/// trans, type casts, layout conversions), and return the root tmem_load
/// operation that originally produced the data. Returns nullptr if the chain
/// doesn't trace back to a tmem_load (e.g., block arguments or other sources).
///
/// This is used to identify SMEM buffers that originate from the same
/// tmem_load (e.g., its result is split into multiple sub-tiles, each
/// stored to a separate SMEM buffer). Such buffers are candidates for
/// buffer ID sharing when they have disjoint liveness.
static Operation *findOriginalLoadOp(Value value) {
  Operation *op = value.getDefiningOp();
  // Currently we only support TMEMLoadOp.
  while (op && !isa<ttng::TMEMLoadOp>(op)) {
    // TODO: Generalize to support addmm.
    // The SubtileOperator should hopefully simplify this work.
    // Transparent ops: trace through to their single tensor input.
    if (isa<tt::SplitOp, tt::ReshapeOp, tt::TransOp, ttg::ConvertLayoutOp,
            arith::TruncFOp, arith::ExtFOp, arith::SIToFPOp, arith::FPToSIOp,
            arith::UIToFPOp, arith::FPToUIOp, arith::TruncIOp, arith::ExtSIOp,
            arith::ExtUIOp, arith::BitcastOp>(op)) {
      op = op->getOperand(0).getDefiningOp();
    } else {
      // Unknown op — Don't support
      op = nullptr;
    }
  }
  return op;
}

/// Given a channel, find the original load operation that produced the data
/// stored into the channel's SMEM buffer. Returns nullptr if the channel has
/// no valid source or the source can't be traced to a load.
static Operation *findOriginalLoadForChannel(Channel *ch) {
  if (!ch || ch->channelKind != DataChannelKind::SMEMPost)
    return nullptr;
  Operation *srcOp = ch->getSrcOp();
  if (!srcOp)
    return nullptr;
  if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(srcOp))
    return findOriginalLoadOp(storeOp.getSrc());
  return nullptr;
}

/// Check if a group of alloc ops all have the same element type and SMEM size.
static bool allAllocsCompatible(ArrayRef<Operation *> allocs,
                                ArrayRef<unsigned> sizes) {
  assert(allocs.size() == sizes.size());
  auto firstAlloc = cast<ttg::LocalAllocOp>(allocs[0]);
  auto firstElemType = firstAlloc.getType().getElementType();
  unsigned firstSize = sizes[0];
  for (unsigned i = 1; i < allocs.size(); ++i) {
    auto alloc = cast<ttg::LocalAllocOp>(allocs[i]);
    if (alloc.getType().getElementType() != firstElemType ||
        sizes[i] != firstSize)
      return false;
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
      // Skip guard channels (isSameIterGuard) — they are auxiliary
      // synchronization channels and should not influence memory planning.
      if (ch->channelKind == DataChannelKind::TMEMPost) {
        auto *tmemCh = static_cast<ttng::TmemDataChannelPost *>(ch);
        if (tmemCh->isSameIterGuard)
          continue;
      }
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
    if (alloc && alloc->getParentOfType<scf::ForOp>() &&
        !alloc->hasAttr("tma_store_buffer")) {
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

    // Enforce minimum buffer.copy >= number of entries sharing each
    // buffer.id. When buffers are shared (e.g. Data Partition) they
    // must be completely disjoin based on the barrier handling. Rather
    // than enforce/optimize that, we ensure we can store 1 of each
    // buffer.
    enforceMinBufferCopy();

    // Phase 2: Merge non-innermost-loop buffers with disjoint liveness
    // and shared data generation step (same original load op).
    // This handles epilogue buffers that come from splitting a single
    // tmem_load result into multiple sub-tiles stored to separate SMEM
    // buffers. Since they are used sequentially, their liveness is disjoint
    // and they can share the same buffer.id to save SMEM.
    //
    // Note: This doesn't yet provide the ability to increase the buffer count
    // in the epilogue.
    fuseEpilogueBuffers();

    lastBufferId = bufferId;
    return success();
  }

  /// Group non-innermost-loop buffers by their original load op and assign
  /// the same buffer.id to buffers within each group that have compatible
  /// types/sizes and pairwise disjoint liveness intervals.
  void enforceMinBufferCopy() {
    DenseMap<int, unsigned> idCounts;
    for (auto bufferIter : bufferRange) {
      Operation *owner = bufferIter.first->owner;
      if (auto id = owner->getAttrOfType<IntegerAttr>("buffer.id"))
        idCounts[id.getInt()]++;
    }
    for (auto bufferIter : bufferRange) {
      Operation *owner = bufferIter.first->owner;
      auto id = owner->getAttrOfType<IntegerAttr>("buffer.id");
      auto copy = owner->getAttrOfType<IntegerAttr>("buffer.copy");
      if (id && copy) {
        unsigned minCopy = idCounts[id.getInt()];
        if (static_cast<unsigned>(copy.getInt()) < minCopy) {
          owner->setAttr(
              "buffer.copy",
              IntegerAttr::get(IntegerType::get(owner->getContext(), 32),
                               minCopy));
        }
      }
    }
  }

  void fuseEpilogueBuffers() {
    DenseMap<Operation *, SmallVector<BufferT *>> loadGroups;
    for (auto &bufferIter : bufferRange) {
      BufferT *buffer = bufferIter.first;
      Operation *owner = buffer->owner;
      if (usersInInnermostLoop(owner))
        continue;
      Channel *ch = findChannelForOp(owner, *channels);
      Operation *origLoad = findOriginalLoadForChannel(ch);
      if (!origLoad)
        continue;
      loadGroups[origLoad].push_back(buffer);
    }

    for (auto &[origLoad, group] : loadGroups) {
      if (group.size() < 2)
        continue;

      SmallVector<Operation *> allocs;
      SmallVector<unsigned> sizes;
      for (auto *buf : group) {
        allocs.push_back(buf->owner);
        sizes.push_back(buf->size);
      }
      if (!allAllocsCompatible(allocs, sizes))
        continue;

      // Sort by liveness start for greedy interval packing.
      llvm::sort(group, [&](BufferT *a, BufferT *b) {
        return bufferRange[a].start() < bufferRange[b].start();
      });

      // Verify all liveness intervals are pairwise disjoint.
      bool disjoint = true;
      for (unsigned i = 0; i < group.size() && disjoint; ++i) {
        for (unsigned j = i + 1; j < group.size(); ++j) {
          if (bufferRange[group[i]].intersects(bufferRange[group[j]])) {
            disjoint = false;
            break;
          }
        }
      }
      if (!disjoint)
        continue;

      // All buffers share the first buffer's ID.
      auto firstId = group[0]->owner->getAttrOfType<IntegerAttr>("buffer.id");
      if (!firstId)
        continue;
      unsigned sharedId = firstId.getValue().getZExtValue();
      auto i32Type = IntegerType::get(group[0]->owner->getContext(), 32);
      for (unsigned i = 1; i < group.size(); ++i) {
        group[i]->owner->setAttr("buffer.id",
                                 IntegerAttr::get(i32Type, sharedId));
      }
      LDBG("Phase 2 (epilogue fusion): merged "
           << group.size() << " buffers into buffer.id=" << sharedId);
    }
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

//===----------------------------------------------------------------------===//
// New SMEM Allocation — WSBuffer-based approach (Phases 1–3)
//===----------------------------------------------------------------------===//

namespace {

/// Priority levels for SMEM multi-buffering candidates.
enum class WSBufferPriority {
  P0_InnermostTMA = 0, // innermost loop + TMA channel
  P1_InnermostNonTMA,  // innermost loop, non-TMA
  P2_Other,            // outside loop / non-innermost (never increased)
};

/// A wrapper around one ttg.local_alloc op for the new SMEM allocation.
struct WSBuffer {
  Operation *allocOp;
  unsigned sizeBytes;
  Interval<size_t> liveness;
  bool isInnermost;
  bool isTMA;
  bool isCrossStage;
  unsigned bufferId;
  unsigned numCopies;
  WSBufferPriority priority;
  bool isPinned = false; // Set by user annotation; skips heuristic phases.
};

/// Parsed channel annotation from tt.autows JSON on an MMA op.
/// Format: "opndA,smem,2,0" → operand=opndA, memType=smem, numCopies=2,
/// bufferId=0.
struct ChannelAnnotation {
  std::string operand; // "opndA", "opndB", "opndD"
  std::string memType; // "smem", "tmem"
  unsigned numCopies;
  unsigned bufferId;
};

/// Parse tt.autows channel annotations from all MMA ops in parentOp.
/// Returns a map from (mmaOp, operandIdx) → ChannelAnnotation, where
/// operandIdx is 0=opndA, 1=opndB, 2=opndD.
/// Detects and warns about conflicting annotations.
static std::map<std::pair<Operation *, unsigned>, ChannelAnnotation>
parseChannelAnnotations(Operation *parentOp) {
  std::map<std::pair<Operation *, unsigned>, ChannelAnnotation> result;
  // Track bufferId → (numCopies, sourceOp) for cross-MMA consistency checks.
  std::map<unsigned, std::pair<unsigned, Operation *>> bufferIdToInfo;

  parentOp->walk([&](Operation *op) {
    if (!isa<ttng::MMAv5OpInterface>(op))
      return;
    auto attr = op->getAttrOfType<StringAttr>("tt.autows");
    if (!attr)
      return;
    auto parsed = llvm::json::parse(attr.getValue());
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return;
    }
    auto *obj = parsed->getAsObject();
    if (!obj)
      return;
    auto *channelsArr = obj->getArray("channels");
    if (!channelsArr)
      return;
    for (auto &elem : *channelsArr) {
      auto str = elem.getAsString();
      if (!str)
        continue;
      SmallVector<StringRef, 4> parts;
      StringRef(*str).split(parts, ',');
      if (parts.size() != 4)
        continue;
      ChannelAnnotation ann;
      ann.operand = parts[0].str();
      ann.memType = parts[1].str();
      ann.numCopies = std::stoi(parts[2].str());
      ann.bufferId = std::stoi(parts[3].str());

      // Validate operand name.
      if (ann.operand != "opndA" && ann.operand != "opndB" &&
          ann.operand != "opndD") {
        LDBG("WARNING: invalid operand name '"
             << ann.operand << "' in channel annotation, skipping");
        continue;
      }
      // Validate memType.
      if (ann.memType != "smem" && ann.memType != "tmem") {
        LDBG("WARNING: invalid memType '"
             << ann.memType << "' in channel annotation, skipping");
        continue;
      }

      unsigned opIdx = ann.operand == "opndA"   ? 0
                       : ann.operand == "opndB" ? 1
                                                : 2; // opndD

      // Check for duplicate operand annotation on the same MMA.
      auto key = std::make_pair(op, opIdx);
      auto dupIt = result.find(key);
      if (dupIt != result.end()) {
        auto &prev = dupIt->second;
        LDBG("WARNING: duplicate annotation for "
             << ann.operand << " on same MMA op — overwriting " << prev.memType
             << "," << prev.numCopies << "," << prev.bufferId << " with "
             << ann.memType << "," << ann.numCopies << "," << ann.bufferId);
      }

      // Check for same bufferId with conflicting numCopies across all MMA ops.
      auto bufIt = bufferIdToInfo.find(ann.bufferId);
      if (bufIt != bufferIdToInfo.end()) {
        if (bufIt->second.first != ann.numCopies) {
          LDBG("WARNING: bufferId="
               << ann.bufferId
               << " has conflicting numCopies: " << bufIt->second.first
               << " vs " << ann.numCopies << " — using max("
               << bufIt->second.first << ", " << ann.numCopies << ")");
          unsigned maxCopies = std::max(bufIt->second.first, ann.numCopies);
          ann.numCopies = maxCopies;
          bufIt->second.first = maxCopies;
        }
      } else {
        bufferIdToInfo[ann.bufferId] = {ann.numCopies, op};
      }

      // Check for operand D annotated as SMEM (always TMEM).
      if (ann.operand == "opndD" && ann.memType != "tmem") {
        LDBG("WARNING: opndD must be tmem, got '" << ann.memType
                                                  << "' — correcting to tmem");
        ann.memType = "tmem";
      }

      result[key] = ann;
      LDBG("parseChannelAnnotations: MMA op has annotation: "
           << ann.operand << "," << ann.memType << "," << ann.numCopies << ","
           << ann.bufferId);
    }
  });
  return result;
}

/// Trace an MMA operand value back to its defining alloc op (local_alloc or
/// tmem_alloc), following through memdesc_trans, MemDescIndex, etc.
static Operation *traceBackToAlloc(Value v) {
  DenseSet<Value> visited;
  SmallVector<Value> worklist = {v};
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;
    Operation *defOp = cur.getDefiningOp();
    if (!defOp)
      continue;
    if (isa<ttg::LocalAllocOp>(defOp) || isa<ttng::TMEMAllocOp>(defOp))
      return defOp;
    // Follow through memdesc_trans, MemDescIndex, memdesc_reinterpret, etc.
    for (auto operand : defOp->getOperands())
      worklist.push_back(operand);
  }
  return nullptr;
}

/// Build a mapping from alloc ops → ChannelAnnotation using a top-down
/// approach: iterate over annotated MMA ops, trace each operand back to its
/// defining alloc op, and associate the annotation.
///
/// This is more robust than the old bottom-up approach (alloc → trace users →
/// find MMA) because it directly uses the MMA's operand accessors (getA(),
/// getB(), getD()) to identify which alloc feeds which operand.
///
/// Detects and warns about conflicting annotations:
///   - Duplicate allocOp mapping (same alloc gets annotations from multiple
///   MMAs)
///   - memType mismatch (SMEM alloc annotated as tmem, or vice versa)
static DenseMap<Operation *, ChannelAnnotation> buildAllocToAnnotationMap(
    SmallVector<Channel *> &channels,
    const std::map<std::pair<Operation *, unsigned>, ChannelAnnotation>
        &annotations) {
  DenseMap<Operation *, ChannelAnnotation> result;

  if (annotations.empty())
    return result;

  for (auto &[key, ann] : annotations) {
    auto [mmaOp, opIdx] = key;
    auto tcMma = dyn_cast<ttng::TCGen5MMAOp>(mmaOp);
    if (!tcMma)
      continue;

    // Get the MMA operand value for this annotation.
    Value operandVal;
    if (opIdx == 0)
      operandVal = tcMma.getA();
    else if (opIdx == 1)
      operandVal = tcMma.getB();
    else if (opIdx == 2)
      operandVal = tcMma.getD();
    else
      continue;

    // Trace back to the defining alloc op.
    Operation *allocOp = traceBackToAlloc(operandVal);
    if (!allocOp) {
      LDBG("buildAllocToAnnotationMap: could not trace "
           << ann.operand << " back to alloc op, skipping");
      continue;
    }

    // Validate memType matches the actual alloc type.
    bool isSmemAlloc = isa<ttg::LocalAllocOp>(allocOp);
    bool isTmemAlloc = isa<ttng::TMEMAllocOp>(allocOp);
    if (isSmemAlloc && ann.memType != "smem") {
      LDBG("WARNING: SMEM alloc annotated with memType='"
           << ann.memType << "' — expected 'smem', skipping annotation for "
           << ann.operand);
      continue;
    }
    if (isTmemAlloc && ann.memType != "tmem") {
      LDBG("WARNING: TMEM alloc annotated with memType='"
           << ann.memType << "' — expected 'tmem', skipping annotation for "
           << ann.operand);
      continue;
    }

    // Check for duplicate allocOp mapping.
    auto dupIt = result.find(allocOp);
    if (dupIt != result.end()) {
      auto &prev = dupIt->second;
      if (prev.bufferId != ann.bufferId || prev.numCopies != ann.numCopies) {
        LDBG("WARNING: allocOp has conflicting annotations: "
             << prev.operand << "," << prev.memType << "," << prev.numCopies
             << "," << prev.bufferId << " vs " << ann.operand << ","
             << ann.memType << "," << ann.numCopies << "," << ann.bufferId
             << " — using earlier annotation");
      }
      continue;
    }

    result[allocOp] = ann;
    LDBG("buildAllocToAnnotationMap: " << ann.operand << "," << ann.memType
                                       << "," << ann.numCopies << ","
                                       << ann.bufferId);
  }
  return result;
}

/// Check if all users of a channel are in the same innermost loop and the
/// alloc type has at least 2 non-trivial dimensions.
static bool isInnermostSmemChannel(Operation *alloc,
                                   SmallVector<Channel *> &channels) {
  Channel *ch = findChannelForOp(alloc, channels);
  if (!ch || ch->channelKind != DataChannelKind::SMEMPost)
    return false;
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
  if (!isInnermostLoop(parentLoop))
    return false;

  // Check 2D+ shape.
  auto sAlloc = cast<ttg::LocalAllocOp>(alloc);
  auto allocDescType = cast<ttg::MemDescType>(sAlloc.getType());
  unsigned numD = 0;
  for (int64_t shape : allocDescType.getShape()) {
    if (shape > 1)
      ++numD;
  }
  return numD >= 2;
}

/// Check if a channel's producer is a TMA operation.
static bool isSmemTMAChannel(Operation *alloc,
                             SmallVector<Channel *> &channels) {
  Channel *ch = findChannelForOp(alloc, channels);
  if (!ch || ch->channelKind != DataChannelKind::SMEMPost)
    return false;
  auto *chPost = static_cast<ChannelPost *>(ch);
  Operation *srcOp = chPost->getSrcOp();
  if (!srcOp)
    return false;
  if (isa<ttng::AsyncTMACopyGlobalToLocalOp>(srcOp))
    return true;
  if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(srcOp)) {
    Value stored = storeOp.getSrc();
    if (auto *defOp = stored.getDefiningOp())
      return isa<tt::DescriptorLoadOp>(defOp);
  }
  return false;
}

/// Helper to read the loop.stage attribute from an op. Returns -1 if absent.
static int getLoopStage(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
  return attr ? attr.getValue().getSExtValue() : -1;
}

/// Check if a channel's actual consumers are in different loop.stage values.
/// The producer stage is not considered because it may be in a different
/// partition. We follow through memdesc_trans operations to find the actual
/// consumers. Only returns true if the buffer is updated inside the innermost
/// loop (srcOp has loop.stage).
static bool isSmemCrossStage(Operation *alloc,
                             SmallVector<Channel *> &channels) {
  Channel *ch = findChannelForOp(alloc, channels);
  if (!ch || ch->channelKind != DataChannelKind::SMEMPost)
    return false;

  // Check that the source (producer) is inside the innermost loop.
  // If srcOp doesn't have loop.stage, the buffer is written outside the loop
  // and doesn't need double-buffering.
  Operation *srcOp = ch->getSrcOp();
  if (!srcOp)
    return false;
  int srcStage = getLoopStage(srcOp);
  if (srcStage < 0)
    return false;

  SmallVector<Operation *> dstOps;
  ch->getDstOps(dstOps);
  if (dstOps.empty()) {
    if (Operation *dst = ch->getDstOp())
      dstOps.push_back(dst);
  }

  // Collect all actual consumers by following through memdesc_trans operations.
  SmallVector<Operation *> actualConsumers;
  for (Operation *dstOp : dstOps) {
    auto consumers = getActualConsumers(dstOp);
    for (auto *consumer : consumers)
      actualConsumers.push_back(consumer);
  }

  // Check if actual consumers are in different stages.
  int firstConsumerStage = -1;
  for (Operation *consumer : actualConsumers) {
    int stage = getLoopStage(consumer);
    if (stage >= 0) {
      if (firstConsumerStage < 0) {
        firstConsumerStage = stage;
      } else if (stage != firstConsumerStage) {
        return true;
      }
    }
  }
  return false;
}

/// Compute the byte size for a local_alloc op.
static unsigned getSmemAllocSizeBytes(ttg::LocalAllocOp alloc) {
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
  return static_cast<unsigned>(numElems * allocType.getElementTypeBitWidth() /
                               8);
}

/// Compute total SMEM usage in bytes across all WSBuffers.
/// Buffers sharing the same buffer.id (reuse group) contribute
/// max(sizes) * copies instead of sum(sizes) * copies.
static unsigned computeTotalSmem(const SmallVector<WSBuffer> &wsBuffers) {
  DenseMap<unsigned, std::pair<unsigned, unsigned>>
      idInfo; // id -> (maxSize, copies)
  for (const auto &buf : wsBuffers) {
    auto it = idInfo.find(buf.bufferId);
    if (it == idInfo.end()) {
      idInfo[buf.bufferId] = {buf.sizeBytes, buf.numCopies};
    } else {
      it->second.first = std::max(it->second.first, buf.sizeBytes);
      it->second.second = std::max(it->second.second, buf.numCopies);
    }
  }
  unsigned total = 0;
  for (auto &kv : idInfo)
    total += kv.second.first * kv.second.second;
  return total;
}

/// Group P2_Other WSBuffers by their original load op and assign the same
/// buffer.id to buffers within each group that have compatible types/sizes.
static void fuseEpilogueWSBuffers(SmallVector<WSBuffer> &wsBuffers,
                                  SmallVector<Channel *> &channels) {
  DenseMap<Operation *, SmallVector<unsigned>> loadGroups;
  for (unsigned i = 0; i < wsBuffers.size(); ++i) {
    auto &buf = wsBuffers[i];
    if (buf.priority != WSBufferPriority::P2_Other)
      continue;
    Channel *ch = findChannelForOp(buf.allocOp, channels);
    Operation *origLoad = findOriginalLoadForChannel(ch);
    if (!origLoad)
      continue;
    loadGroups[origLoad].push_back(i);
  }

  for (auto &[origLoad, indices] : loadGroups) {
    if (indices.size() < 2)
      continue;

    SmallVector<Operation *> allocs;
    SmallVector<unsigned> sizes;
    for (unsigned idx : indices) {
      allocs.push_back(wsBuffers[idx].allocOp);
      sizes.push_back(wsBuffers[idx].sizeBytes);
    }
    if (!allAllocsCompatible(allocs, sizes))
      continue;

    unsigned sharedId = wsBuffers[indices[0]].bufferId;
    for (unsigned k = 1; k < indices.size(); ++k)
      wsBuffers[indices[k]].bufferId = sharedId;
    LDBG("Phase 3.5 (epilogue fusion): merged "
         << indices.size() << " P2_Other buffers into bufferId=" << sharedId);
  }
}

/// Phase 4.5: Iterative copy increase for fused P2_Other groups.
/// Epilogue buffers merged in Phase 3.5 share a single bufferId but are
/// left at numCopies=1 by Phase 4. Increase copies uniformly for each
/// fused group while staying within the SMEM budget.
static void increaseFusedEpilogueCopies(SmallVector<WSBuffer> &wsBuffers,
                                        unsigned numBuffers,
                                        unsigned smemBudget) {
  // Collect fused P2_Other groups by bufferId.
  DenseMap<unsigned, SmallVector<unsigned>> epilogueGroups;
  for (unsigned i = 0; i < wsBuffers.size(); ++i) {
    auto &buf = wsBuffers[i];
    if (buf.isPinned || buf.priority != WSBufferPriority::P2_Other)
      continue;
    epilogueGroups[buf.bufferId].push_back(i);
  }

  for (auto &[bufferId, indices] : epilogueGroups) {
    if (indices.size() < 2)
      continue;

    // Determine current copies (should be uniform within a fused group).
    unsigned currentCopies = wsBuffers[indices[0]].numCopies;

    // Respect cross-stage minimum from Phase 2.
    unsigned minCopies = currentCopies;
    for (unsigned idx : indices) {
      if (wsBuffers[idx].isCrossStage)
        minCopies = std::max(minCopies, 2u);
    }
    if (minCopies > currentCopies)
      currentCopies = minCopies;

    // Iteratively increase numCopies up to numBuffers.
    unsigned tryCopies = currentCopies + 1;
    while (tryCopies <= numBuffers) {
      // Tentatively set all buffers in the group.
      SmallVector<unsigned> saved;
      for (unsigned idx : indices)
        saved.push_back(wsBuffers[idx].numCopies);

      for (unsigned idx : indices)
        wsBuffers[idx].numCopies = tryCopies;

      unsigned totalSmem = computeTotalSmem(wsBuffers);
      if (totalSmem <= smemBudget) {
        LDBG("Phase 4.5: epilogue group bufferId="
             << bufferId << " copies=" << tryCopies
             << " totalSmem=" << totalSmem << " ≤ " << smemBudget);
        tryCopies++;
      } else {
        // Revert and stop.
        for (unsigned k = 0; k < indices.size(); ++k)
          wsBuffers[indices[k]].numCopies = saved[k];
        LDBG("Phase 4.5: epilogue group bufferId="
             << bufferId << " copies=" << tryCopies << " totalSmem="
             << totalSmem << " > " << smemBudget << " — budget exhausted");
        break;
      }
    }
  }
}

/// New SMEM allocation: Phases 1–5.
///
/// Phase 1: Create one WSBuffer per local_alloc, all copy=1, unique IDs.
/// Phase 2: Enforce cross-stage minimum (copy >= 2).
/// Phase 3: Classify into priority levels P0/P1/P2.
/// Phase 4: Iterative copy increase within SMEM budget.
/// Phase 5: Emit buffer.id and buffer.copy attributes.
///
/// Returns the next available buffer ID after the SMEM allocations.
static unsigned allocateSmemBuffers(
    triton::FuncOp funcOp, SmallVector<Channel *> &channels,
    unsigned numBuffers, unsigned smemBudget, bool smemCircularReuse,
    const DenseMap<Operation *, ChannelAnnotation> &allocToAnnotation) {
  // ── Phase 1: Create WSBuffers ───────────────────────────────────────
  SmallVector<WSBuffer> wsBuffers;
  unsigned nextBufferId = 0;

  funcOp->walk<WalkOrder::PreOrder>([&](ttg::LocalAllocOp alloc) {
    if (!alloc.isSharedMemoryAlloc())
      return;

    WSBuffer buf;
    buf.allocOp = alloc;
    buf.sizeBytes = getSmemAllocSizeBytes(alloc);
    buf.isInnermost = isInnermostSmemChannel(alloc, channels);
    buf.isTMA = buf.isInnermost && isSmemTMAChannel(alloc, channels);
    buf.isCrossStage = isSmemCrossStage(alloc, channels);
    buf.bufferId = nextBufferId++;
    buf.numCopies = 1;
    buf.priority = WSBufferPriority::P2_Other;

    // Check for annotation-based pre-assignment.
    auto it = allocToAnnotation.find(alloc.getOperation());
    if (it != allocToAnnotation.end() && it->second.memType == "smem") {
      buf.bufferId = it->second.bufferId;
      buf.numCopies = it->second.numCopies;
      buf.isPinned = true;
      LDBG("Phase 1: WSBuffer pinned by annotation: bufferId="
           << buf.bufferId << " numCopies=" << buf.numCopies);
    }

    wsBuffers.push_back(buf);

    LDBG("Phase 1: WSBuffer["
         << buf.bufferId << "] " << buf.sizeBytes << " bytes"
         << " innermost=" << buf.isInnermost << " TMA=" << buf.isTMA
         << " crossStage=" << buf.isCrossStage << " pinned=" << buf.isPinned);
  });

  if (wsBuffers.empty())
    return nextBufferId;

  // Ensure heuristic-assigned IDs don't collide with annotated IDs.
  for (auto &buf : wsBuffers)
    if (buf.isPinned)
      nextBufferId = std::max(nextBufferId, buf.bufferId + 1);

  // ── Phase 2: Enforce cross-stage minimum ────────────────────────────
  // Budget-aware: only set copy=2 if the total SMEM stays within budget.
  for (auto &buf : wsBuffers) {
    if (buf.isPinned) {
      if (buf.isCrossStage && buf.numCopies < 2) {
        LDBG("WARNING: pinned WSBuffer["
             << buf.bufferId << "] is cross-stage but has numCopies="
             << buf.numCopies << " — this may cause correctness issues"
             << " (producer may overwrite before consumer reads)");
      }
      continue;
    }
    if (buf.isCrossStage && numBuffers >= 2) {
      unsigned saved = buf.numCopies;
      buf.numCopies = 2;
      unsigned totalSmem = computeTotalSmem(wsBuffers);
      if (totalSmem <= smemBudget) {
        LDBG("Phase 2: WSBuffer[" << buf.bufferId
                                  << "] cross-stage → numCopies=2"
                                  << " (totalSmem=" << totalSmem << ")");
      } else {
        buf.numCopies = saved;
        LDBG("Phase 2: WSBuffer["
             << buf.bufferId << "] cross-stage copy=2 skipped"
             << " (would exceed budget: " << totalSmem << " > " << smemBudget
             << ")");
      }
    }
  }

  // ── Phase 3: Classify and prioritize ────────────────────────────────
  for (auto &buf : wsBuffers) {
    if (buf.isPinned)
      continue;
    if (buf.isInnermost && buf.isTMA) {
      buf.priority = WSBufferPriority::P0_InnermostTMA;
    } else if (buf.isInnermost) {
      buf.priority = WSBufferPriority::P1_InnermostNonTMA;
    } else {
      buf.priority = WSBufferPriority::P2_Other;
    }
    LDBG("Phase 3: WSBuffer["
         << buf.bufferId << "] priority=" << static_cast<int>(buf.priority));
  }

  // ── Phase 3.5: Merge P2_Other buffers from the same original load ───
  // Epilogue buffers (e.g., from splitting a tmem_load result into sub-tiles
  // stored to separate SMEM buffers) have disjoint liveness and can share
  // the same buffer.id to reduce SMEM usage before the copy increase pass.
  fuseEpilogueWSBuffers(wsBuffers, channels);

  // ── Phase 4: Iterative copy increase ────────────────────────────────
  // Process P0 then P1. P2 is never increased.
  for (auto priority : {WSBufferPriority::P0_InnermostTMA,
                        WSBufferPriority::P1_InnermostNonTMA}) {
    // Collect candidate indices at this priority.
    SmallVector<unsigned> candidateIndices;
    for (unsigned i = 0; i < wsBuffers.size(); ++i) {
      if (wsBuffers[i].isPinned)
        continue;
      if (wsBuffers[i].priority == priority)
        candidateIndices.push_back(i);
    }
    if (candidateIndices.empty())
      continue;

    LDBG("Phase 4: processing priority=" << static_cast<int>(priority)
                                         << " with " << candidateIndices.size()
                                         << " candidates");

    // Step 0: Decide grouping upfront.
    bool isReuseGroup = false;
    if (smemCircularReuse && candidateIndices.size() == 2) {
      isReuseGroup = true;
      auto &bufA = wsBuffers[candidateIndices[0]];
      auto &bufB = wsBuffers[candidateIndices[1]];

      // B shares A's buffer.id.
      bufB.bufferId = bufA.bufferId;

      // Compute starting copies for the group based on cross-stage.
      unsigned maxCrossStageMin = 1;
      if (bufA.isCrossStage)
        maxCrossStageMin = std::max(maxCrossStageMin, 2u);
      if (bufB.isCrossStage)
        maxCrossStageMin = std::max(maxCrossStageMin, 2u);

      unsigned groupStart = 1;
      if (maxCrossStageMin >= 2)
        groupStart = maxCrossStageMin * 2 - 1; // e.g., 3

      // Clamp to num_buffers.
      if (groupStart > numBuffers)
        groupStart = numBuffers;

      bufA.numCopies = groupStart;
      bufB.numCopies = groupStart;

      LDBG("Phase 4: formed reuse group ["
           << bufA.bufferId << "] with startCopies=" << groupStart
           << " (crossStageMin=" << maxCrossStageMin << ")");
    }

    // Step 1: Incremental loop.
    unsigned currentGroupCopies;
    if (isReuseGroup) {
      currentGroupCopies = wsBuffers[candidateIndices[0]].numCopies;
    } else {
      // Start at the minimum numCopies across candidates (may be > 1
      // after Phase 2 cross-stage enforcement).
      currentGroupCopies = numBuffers; // will be lowered
      for (unsigned idx : candidateIndices)
        currentGroupCopies =
            std::min(currentGroupCopies, wsBuffers[idx].numCopies);
    }

    bool foundValidSolution = false;

    while (currentGroupCopies <= numBuffers) {
      if (isReuseGroup) {
        // Reuse group path: set group copies and check budget.
        auto &bufA = wsBuffers[candidateIndices[0]];
        auto &bufB = wsBuffers[candidateIndices[1]];
        unsigned savedA = bufA.numCopies, savedB = bufB.numCopies;
        bufA.numCopies = currentGroupCopies;
        bufB.numCopies = currentGroupCopies;

        unsigned totalSmem = computeTotalSmem(wsBuffers);
        if (totalSmem <= smemBudget) {
          foundValidSolution = true;
          LDBG("Phase 4: reuse group copies=" << currentGroupCopies
                                              << " totalSmem=" << totalSmem
                                              << " ≤ " << smemBudget);
          currentGroupCopies++;
        } else {
          bufA.numCopies = savedA;
          bufB.numCopies = savedB;
          LDBG("Phase 4: reuse group copies="
               << currentGroupCopies << " totalSmem=" << totalSmem << " > "
               << smemBudget << " — budget exhausted");
          break;
        }
      } else {
        // Individual path: bring each pending candidate to currentGroupCopies.
        SmallVector<unsigned> pending;
        for (unsigned idx : candidateIndices) {
          if (wsBuffers[idx].numCopies < currentGroupCopies)
            pending.push_back(idx);
        }

        if (pending.empty()) {
          currentGroupCopies++;
          continue;
        }

        bool advancedAny = false;
        for (unsigned idx : pending) {
          auto &buf = wsBuffers[idx];
          unsigned saved = buf.numCopies;
          buf.numCopies = currentGroupCopies;

          unsigned totalSmem = computeTotalSmem(wsBuffers);
          if (totalSmem <= smemBudget) {
            advancedAny = true;
            foundValidSolution = true;
            LDBG("Phase 4: WSBuffer["
                 << buf.bufferId << "] copies=" << currentGroupCopies
                 << " totalSmem=" << totalSmem << " ≤ " << smemBudget);
          } else {
            buf.numCopies = saved;
            LDBG("Phase 4: WSBuffer["
                 << buf.bufferId << "] copies=" << currentGroupCopies
                 << " totalSmem=" << totalSmem << " > " << smemBudget
                 << " — skipped");
          }
        }

        if (!advancedAny)
          break;

        currentGroupCopies++;
      }
    }

    // Step 2: Finalize reuse decision.
    // If final copies is even, split the group back.
    if (isReuseGroup) {
      auto &bufA = wsBuffers[candidateIndices[0]];
      auto &bufB = wsBuffers[candidateIndices[1]];
      if (bufA.numCopies % 2 == 0) {
        unsigned half = bufA.numCopies / 2;
        bufA.numCopies = half;
        bufB.numCopies = half;
        bufB.bufferId = nextBufferId++;
        isReuseGroup = false;
        LDBG("Phase 4: split reuse group — even copies="
             << (half * 2) << " → each gets " << half);
      }
    }

    // Step 3: Validate.
    if (!foundValidSolution) {
      LDBG("Phase 4: WARNING — no valid SMEM allocation found for priority="
           << static_cast<int>(priority));
    }
  }

  LDBG("Phase 4 complete: totalSmem=" << computeTotalSmem(wsBuffers));

  // ── Phase 4.5: Iterative copy increase for fused P2_Other groups ────
  increaseFusedEpilogueCopies(wsBuffers, numBuffers, smemBudget);

  LDBG("Phase 4.5 complete: totalSmem=" << computeTotalSmem(wsBuffers));

  // ── Phase 5: Emit buffer.id and buffer.copy attributes ──────────────
  auto i32Type = IntegerType::get(funcOp.getContext(), 32);
  for (auto &buf : wsBuffers) {
    buf.allocOp->setAttr("buffer.id", IntegerAttr::get(i32Type, buf.bufferId));
    buf.allocOp->setAttr("buffer.copy",
                         IntegerAttr::get(i32Type, buf.numCopies));
    LDBG("Phase 5: WSBuffer[" << buf.bufferId << "] buffer.id=" << buf.bufferId
                              << " buffer.copy=" << buf.numCopies);
  }

  return nextBufferId;
}

} // anonymous namespace

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
struct TMemAllocInfo {
  ttng::TMEMAllocOp alloc;
  unsigned baseCols;
  unsigned copy;
};

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

  SmallVector<BufferT *> buffers;
  DenseMap<Operation *, ttng::TmemDataChannelPost *> allocToChannel;

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

  /// Look up the BufferT for a given alloc operation.
  BufferT *getBuffer(Operation *candAlloc) {
    for (auto *alloc : buffers) {
      if (alloc->owner == candAlloc)
        return alloc;
    }
    return nullptr;
  }

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
  LogicalResult run(unsigned bufferId) override {
    Operation *parentOp = operation;
    SmallVector<triton::nvidia_gpu::TMEMAllocOp> allocs;
    buildOperationIdMap();
    parentOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto alloc = dyn_cast<triton::nvidia_gpu::TMEMAllocOp>(op)) {
        allocs.push_back(alloc);
      }
    });
    Liveness liveness(parentOp);
    DenseMap<Operation *, Interval<size_t>> allocToIntervals;
    DenseMap<Operation *, ttng::TMemAllocation> allocToSize;
    allocToChannel.clear();
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
           ttng::TMemAllocation(allocSize.numRows, allocSize.numCols)});
      allocToChannel[alloc.getOperation()] = TheCh;
    }
    // Sort allocs according to isOperandD, size, live interval.
    // This can be adjusted later on.
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
    Allocation allocation;
    this->buffers.clear();
    for (auto alloc : allocs) {
      // size is 0, alignment is default, offset is default
      allocation.addBuffer<BufferT::BufferKind::Explicit>(alloc, 0);
      BufferT *tBuf = allocation.valueBuffer[alloc];
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

    for (auto valueBufferIter : allocation.valueBuffer) {
      auto *buffer = valueBufferIter.second;
      // valueBuffer maps value to BufferT
      Operation *alloc = valueBufferIter.first.getDefiningOp();
      // bufferRange maps BufferT to interval
      bufferRange[buffer] = allocToIntervals[alloc];
    }
    // For each innermost loop according to program order (via
    // getIntervalForCtrlOp)
    //   Go through all buffers that are live in the loop
    //   Start with buffers with longest span within the loop
    //   For each buffer
    //     either allocate new space (owner of a set of rows)
    //     or reuse an existing buffer's space
    //     if this buffer interferes with all allocated buffers, allocate new
    //     space if this buffer is along the dependency chain, reuse space if
    //     there is enough space, allocate new space otherwise, reuse space

    // Use BufferT to track rowSize/colSize/rowOffset etc, use bufferRange to
    // track intervals.
    SmallVector<Operation *> innermostLoops;
    parentOp->walk([&](Operation *subOp) {
      if (auto theForOp = dyn_cast<scf::ForOp>(subOp))
        if (isInnermostLoop(theForOp))
          innermostLoops.push_back(subOp);
    });
    DenseSet<Operation *> handledAllocs;
    unsigned ctrlIdx = 0;

    // ── Pre-assignment: parse annotations and partition annotated TMEM allocs.
    auto annotations = parseChannelAnnotations(parentOp);
    DenseMap<Operation *, ChannelAnnotation> tmemAllocAnnotations;
    if (!annotations.empty())
      tmemAllocAnnotations = buildAllocToAnnotationMap(*channels, annotations);
    // Filter to only tmem annotations.
    DenseMap<Operation *, ChannelAnnotation> tmemAnnotations;
    for (auto &[op, ann] : tmemAllocAnnotations) {
      if (ann.memType == "tmem")
        tmemAnnotations[op] = ann;
    }

    // Pre-assign annotated TMEM allocs before heuristic.
    if (!tmemAnnotations.empty()) {
      auto i32Type = IntegerType::get(parentOp->getContext(), 32);

      // Group annotated allocs by bufferId.
      std::map<unsigned, SmallVector<ttng::TMEMAllocOp>> annotatedGroups;
      for (auto alloc : allocs) {
        auto it = tmemAnnotations.find(alloc.getOperation());
        if (it != tmemAnnotations.end())
          annotatedGroups[it->second.bufferId].push_back(alloc);
      }

      // For each group: first alloc is owner, rest are reusers.
      // Validate reuse legality and compute buffer.offset.
      size_t preAssignRowOffset = 0;
      for (auto &[bid, group] : annotatedGroups) {
        // Owner: first alloc in the group.
        auto ownerAlloc = group[0];
        auto *ownerBuf = getBuffer(ownerAlloc.getOperation());
        ownerBuf->rowOffset = preAssignRowOffset;
        ownerBuf->colOffset = 0;
        ownerBuf->isOwnerOfSpace = true;
        ownerBuf->reuseOwner = ownerBuf;
        ownerAlloc->setAttr("buffer.id", IntegerAttr::get(i32Type, bid));
        ownerAlloc->setAttr("buffer.copy", IntegerAttr::get(i32Type, 1));
        preAssignRowOffset += ownerBuf->rowSize;
        handledAllocs.insert(ownerAlloc.getOperation());
        LDBG("TMEM pre-assign: owner alloc buffer.id="
             << bid << " rows=" << ownerBuf->rowSize << "x"
             << ownerBuf->colSize);

        // Reusers: subsequent allocs in the group.
        size_t nextColOffset = 0;
        for (size_t i = 1; i < group.size(); ++i) {
          auto reuserAlloc = group[i];
          auto *reuserBuf = getBuffer(reuserAlloc.getOperation());

          // Validate: reuser columns must fit in owner.
          if (reuserBuf->colSize > ownerBuf->colSize) {
            LDBG("WARNING: annotated TMEM reuse buffer.id="
                 << bid << " reuser colSize=" << reuserBuf->colSize
                 << " > owner colSize=" << ownerBuf->colSize
                 << " — skipping reuse, treating as separate owner");
            reuserBuf->rowOffset = preAssignRowOffset;
            reuserBuf->colOffset = 0;
            reuserBuf->isOwnerOfSpace = true;
            reuserBuf->reuseOwner = reuserBuf;
            reuserAlloc->setAttr("buffer.id", IntegerAttr::get(i32Type, bid));
            reuserAlloc->setAttr("buffer.copy", IntegerAttr::get(i32Type, 1));
            preAssignRowOffset += reuserBuf->rowSize;
            handledAllocs.insert(reuserAlloc.getOperation());
            continue;
          }

          // Validate: liveness non-overlap.
          if (bufferRange[ownerBuf].intersects(bufferRange[reuserBuf])) {
            LDBG("WARNING: annotated TMEM reuse buffer.id="
                 << bid << " has overlapping liveness between owner and reuser"
                 << " — skipping reuse, treating as separate owner");
            reuserBuf->rowOffset = preAssignRowOffset;
            reuserBuf->colOffset = 0;
            reuserBuf->isOwnerOfSpace = true;
            reuserBuf->reuseOwner = reuserBuf;
            reuserAlloc->setAttr("buffer.id", IntegerAttr::get(i32Type, bid));
            reuserAlloc->setAttr("buffer.copy", IntegerAttr::get(i32Type, 1));
            preAssignRowOffset += reuserBuf->rowSize;
            handledAllocs.insert(reuserAlloc.getOperation());
            continue;
          }

          // Assign reuser at nextColOffset within owner's column space.
          reuserBuf->rowOffset = ownerBuf->rowOffset;
          reuserBuf->colOffset = nextColOffset;
          reuserBuf->isOwnerOfSpace = false;
          reuserBuf->reuseOwner = ownerBuf;
          reuserAlloc->setAttr("buffer.id", IntegerAttr::get(i32Type, bid));
          reuserAlloc->setAttr("buffer.copy", IntegerAttr::get(i32Type, 1));
          reuserAlloc->setAttr("buffer.offset",
                               IntegerAttr::get(i32Type, nextColOffset));
          handledAllocs.insert(reuserAlloc.getOperation());
          LDBG("TMEM pre-assign: reuser buffer.id="
               << bid << " colOffset=" << nextColOffset
               << " size=" << reuserBuf->rowSize << "x" << reuserBuf->colSize);
          nextColOffset += reuserBuf->colSize;
        }
      }

      // Ensure heuristic buffer IDs don't collide with annotated IDs.
      for (auto &[bid, _] : annotatedGroups)
        bufferId = std::max(bufferId, bid + 1);
    }

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
      // Check for per-loop tt.tmem_alloc_algo attribute on the forOp
      // or its parent ForOps (e.g., the WS loop wrapping the innermost
      // scheduled loop in persistent kernels).
      // 1 = greedy (allocateTMemAllocs), 2 = backtracking
      // (allocateTMemAllocs2). Default is 1 (greedy).
      int tmemAllocAlgo = 1;
      if (auto attr = ctrlOp->getAttrOfType<IntegerAttr>("tt.tmem_alloc_algo"))
        tmemAllocAlgo = attr.getInt();
      // Walk parent ForOps: outermost sets the default, innermost wins.
      for (auto parent = ctrlOp->getParentOfType<scf::ForOp>(); parent;
           parent = parent->getParentOfType<scf::ForOp>()) {
        if (auto attr =
                parent->getAttrOfType<IntegerAttr>("tt.tmem_alloc_algo")) {
          // Only override if the innermost (ctrlOp) didn't set it.
          if (!ctrlOp->getAttrOfType<IntegerAttr>("tt.tmem_alloc_algo"))
            tmemAllocAlgo = attr.getInt();
        }
      }

      FailureOr<unsigned> result;
      if (tmemAllocAlgo == 1) {
        LDBG("using tmem allocation algorithm 1 (greedy)");
        result = allocateTMemAllocs(allocsForThisLoop, buffers, allocToChannel,
                                    operationId, ctrlOp, bufferId);
      } else {
        LDBG("using tmem allocation algorithm 2 (backtracking)");
        result = allocateTMemAllocs2(allocsForThisLoop, buffers, allocToChannel,
                                     operationId, ctrlOp, bufferId);
      }
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
    // TODO: Remove this when the memory planner has the logic for allocating
    // multi-buffer TMEM fully working.
    // Post-processing: maximize TMEM utilization by increasing buffer.copy
    // for TMEM allocs in round-robin until we approach the 512-column limit.
    // Only applies to persistent kernels where CTAs process multiple tiles.
    constexpr unsigned tmemColLimit = 512;

    SmallVector<TMemAllocInfo> allocInfos;

    unsigned totalCols = 0;
    for (auto alloc : allocs) {
      ttng::TMemAllocation allocSize = ttng::getTmemAllocSizes(alloc.getType());
      unsigned baseCols = allocSize.numCols;
      unsigned copy = 1;
      if (auto copyAttr = alloc->getAttrOfType<IntegerAttr>("buffer.copy"))
        copy = copyAttr.getInt();
      totalCols += baseCols * copy;
      // TODO: Remove this restriction once buffer index constraints are
      // tested for TMEM allocs that are not loop-carried MMA accumulators.
      // Currently only allocs with a loop-carried acc token have correct
      // multi-buffer index logic in createBufferPost.
      bool hasLoopCarriedMMA = false;
      for (auto *user : alloc.getResult().getUsers()) {
        if (auto forOp = user->getParentOfType<scf::ForOp>()) {
          if (hasLoopCarriedAccToken(alloc, forOp)) {
            hasLoopCarriedMMA = true;
            break;
          }
        }
      }
      if (!hasLoopCarriedMMA)
        continue;
      allocInfos.push_back({alloc, baseCols, copy});
    }

    while (totalCols < tmemColLimit && !allocInfos.empty()) {
      bool added = false;
      for (unsigned idx = 0; idx < allocInfos.size(); idx++) {
        auto &info = allocInfos[idx];
        if (totalCols + info.baseCols <= tmemColLimit) {
          info.copy += 1;
          totalCols += info.baseCols;
          added = true;
        }
      }
      if (!added)
        break;
    }

    for (auto &info : allocInfos) {
      info.alloc->setAttr(
          "buffer.copy",
          IntegerAttr::get(IntegerType::get(info.alloc->getContext(), 32),
                           info.copy));
    }

    LLVM_DEBUG({
      DBGS() << "TMEM multi-buffering post-processing: totalCols = "
             << totalCols << " / " << tmemColLimit << "\n";
      for (auto &info : allocInfos) {
        DBGS() << "  baseCols=" << info.baseCols << " copy=" << info.copy
               << ": ";
        info.alloc->dump();
      }
    });

    return success();
  }

  // ---------------------------------------------------------------
  // allocateTMemAllocs2 — backtracking search allocation algorithm.
  // ---------------------------------------------------------------

  /// State for backtracking search.
  struct AllocationState {
    /// For each buffer, stores (reuseOwner, colOffset). nullptr means owner.
    DenseMap<BufferT *, std::pair<BufferT *, size_t>> assignment;
    /// Set of buffers that own their space.
    DenseSet<BufferT *> owners;
    /// Total rows used.
    size_t usedRows = 0;
  };

  /// Check if candidate can potentially reuse owner's space.
  /// Returns priority: 0 = cannot reuse, 1 = can reuse, 2 = exact size match.
  /// Uses bidirectional data dependency via SSA def-use chain walk (primary),
  /// with samePartition fallback for cross-loop buffers where SSA chains may
  /// be broken by loop-carried values.
  int hasPotentialReuse(BufferT *owner, BufferT *candidate, Operation *ctrlOp) {
    // Size check: candidate must fit in owner's columns
    if (candidate->colSize > owner->colSize)
      return 0;

    // Liveness check: must not overlap (would need same space at same time)
    if (bufferRange[owner].intersects(bufferRange[candidate]))
      return 0;

    // Bidirectional data dependency check via channels (SSA def-use walk).
    auto *srcCh = allocToChannel[owner->owner];
    auto *dstCh = allocToChannel[candidate->owner];
    auto hasDependency = [&]() -> bool {
      if (!srcCh || !dstCh)
        return false;
      if (isDataDependent(srcCh->getDstOp(), dstCh->getSrcOp()) ||
          isDataDependent(dstCh->getDstOp(), srcCh->getSrcOp()))
        return true;
      return false;
    };

    if (!hasDependency())
      return 0;

    // Priority: prefer exact size matches
    if (candidate->colSize == owner->colSize)
      return 2;
    return 1;
  }

  /// Compute column offset for candidate in owner's reuse group.
  /// Returns INVALID (max size_t) if can't fit.
  /// Uses hasPotentialReuse to determine if buffers can share columns.
  size_t computeColOffset(BufferT *candidate, BufferT *owner,
                          const AllocationState &state, Operation *ctrlOp) {
    size_t maxColOffset = 0;

    // Check compatibility with existing reusers using hasPotentialReuse.
    // If hasPotentialReuse returns > 0 in either direction, they can share
    // the same column space. Otherwise, they need different columns.
    for (auto &[reuser, assignment] : state.assignment) {
      auto [reuseOwner, reuserColOffset] = assignment;
      if (reuseOwner != owner)
        continue;

      // Check if reuser and candidate can share columns
      bool canShareColumns =
          (hasPotentialReuse(reuser, candidate, ctrlOp) > 0 ||
           hasPotentialReuse(candidate, reuser, ctrlOp) > 0);
      if (!canShareColumns) {
        // They can't share - place candidate after reuser's column range
        maxColOffset =
            std::max(maxColOffset, reuserColOffset + reuser->colSize);
      }
    }

    // Check if candidate fits
    if (maxColOffset + candidate->colSize > owner->colSize)
      return std::numeric_limits<size_t>::max();

    return maxColOffset;
  }

  /// Recursive backtracking search for buffer allocation.
  bool tryAllocate(SmallVectorImpl<ttng::TMEMAllocOp> &allocs, size_t idx,
                   AllocationState &state, size_t maxRows, Operation *ctrlOp) {
    // Base case: all buffers allocated
    if (idx == allocs.size())
      return true;

    BufferT *buf = getBuffer(allocs[idx].getOperation());

    // Collect reuse candidates sorted by priority (descending)
    SmallVector<std::pair<BufferT *, int>> candidates;
    for (BufferT *owner : state.owners) {
      int priority = hasPotentialReuse(owner, buf, ctrlOp);
      if (priority > 0)
        candidates.push_back({owner, priority});
    }
    // Sort by priority descending
    llvm::sort(candidates, [](const auto &a, const auto &b) {
      return a.second > b.second;
    });

    // Try each reuse candidate
    for (auto &[owner, priority] : candidates) {
      size_t colOffset = computeColOffset(buf, owner, state, ctrlOp);
      if (colOffset == std::numeric_limits<size_t>::max())
        continue; // Can't fit or dependency check failed

      // Tentatively assign
      AllocationState newState = state;
      newState.assignment[buf] = {owner, colOffset};

      LLVM_DEBUG({
        LDBG("tryAllocate: trying reuse ["
             << bufferRange[buf].start() << "-" << bufferRange[buf].end()
             << ") in owner [" << bufferRange[owner].start() << "-"
             << bufferRange[owner].end() << ") at col " << colOffset);
      });

      // Recurse
      if (tryAllocate(allocs, idx + 1, newState, maxRows, ctrlOp)) {
        state = newState;
        return true;
      }
      // Backtrack: try next candidate
      LLVM_DEBUG({
        LDBG("tryAllocate: backtracking from reuse ["
             << bufferRange[buf].start() << "-" << bufferRange[buf].end()
             << ") in owner [" << bufferRange[owner].start() << "-"
             << bufferRange[owner].end() << ")");
      });
    }

    // Try allocating new space
    if (state.usedRows + buf->rowSize <= maxRows) {
      AllocationState newState = state;
      newState.owners.insert(buf);
      newState.usedRows += buf->rowSize;

      LLVM_DEBUG({
        LDBG("tryAllocate: trying new space for ["
             << bufferRange[buf].start() << "-" << bufferRange[buf].end()
             << ") at row " << state.usedRows);
      });

      if (tryAllocate(allocs, idx + 1, newState, maxRows, ctrlOp)) {
        state = newState;
        return true;
      }
      LLVM_DEBUG({
        LDBG("tryAllocate: backtracking from new space for ["
             << bufferRange[buf].start() << "-" << bufferRange[buf].end()
             << ")");
      });
    }

    return false; // No valid allocation, backtrack
  }

  /// Apply the allocation state to the actual buffers.
  void applyAllocationState(SmallVectorImpl<ttng::TMEMAllocOp> &allocs,
                            const AllocationState &state, unsigned &bufferId) {
    // First pass: assign owners
    size_t rowOffset = 0;
    DenseMap<BufferT *, unsigned> ownerToBufferId;
    for (auto alloc : allocs) {
      BufferT *buf = getBuffer(alloc.getOperation());
      if (state.owners.contains(buf)) {
        buf->rowOffset = rowOffset;
        buf->colOffset = 0;
        buf->isOwnerOfSpace = true;
        buf->reuseOwner = buf;
        ownerToBufferId[buf] = bufferId;
        alloc.getOperation()->setAttr(
            "buffer.id",
            IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                             bufferId));
        ++bufferId;
        rowOffset += buf->rowSize;
      }
    }

    // Second pass: assign reusers
    for (auto alloc : allocs) {
      BufferT *buf = getBuffer(alloc.getOperation());
      if (!state.owners.contains(buf)) {
        auto it = state.assignment.find(buf);
        assert(it != state.assignment.end());
        auto [owner, colOffset] = it->second;
        buf->rowOffset = owner->rowOffset;
        buf->colOffset = colOffset;
        buf->isOwnerOfSpace = false;
        buf->reuseOwner = owner;
        alloc.getOperation()->setAttr(
            "buffer.id",
            IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                             ownerToBufferId[owner]));
        alloc.getOperation()->setAttr(
            "buffer.offset",
            IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                             colOffset));
      }
      // Set buffer.copy attribute
      alloc.getOperation()->setAttr(
          "buffer.copy",
          IntegerAttr::get(IntegerType::get(alloc->getContext(), 32), 1));
    }
  }

  FailureOr<unsigned> allocateTMemAllocs2(
      SmallVector<ttng::TMEMAllocOp> &allocs, SmallVector<BufferT *> &buffers,
      DenseMap<Operation *, ttng::TmemDataChannelPost *> &allocToChannel,
      DenseMap<Operation *, size_t> &operationId, Operation *ctrlOp,
      unsigned bufferId) {

    LDBG("allocateTMemAllocs2: starting with " << allocs.size() << " allocs");

    // Debug: dump allocation order and liveness
    LLVM_DEBUG({
      llvm::dbgs()
          << "\n=== allocateTMemAllocs2: Buffer Allocation Order ===\n";
      size_t idx = 0;
      for (auto alloc : allocs) {
        auto *buf = getBuffer(alloc.getOperation());
        llvm::dbgs() << "  [" << idx++ << "] liveness=["
                     << bufferRange[buf].start() << "-"
                     << bufferRange[buf].end() << ") size=" << buf->rowSize
                     << "x" << buf->colSize << "\n";
      }
      llvm::dbgs() << "\n=== hasPotentialReuse Matrix ===\n";
      for (auto alloc_i : allocs) {
        for (auto alloc_j : allocs) {
          if (alloc_i.getOperation() != alloc_j.getOperation()) {
            auto *buf_i = getBuffer(alloc_i.getOperation());
            auto *buf_j = getBuffer(alloc_j.getOperation());
            int priority = hasPotentialReuse(buf_i, buf_j, ctrlOp);
            if (priority > 0) {
              llvm::dbgs() << "  hasPotentialReuse(["
                           << bufferRange[buf_i].start() << "-"
                           << bufferRange[buf_i].end() << "), ["
                           << bufferRange[buf_j].start() << "-"
                           << bufferRange[buf_j].end() << ")) = " << priority
                           << "\n";
            }
          }
        }
      }
      llvm::dbgs() << "=== End hasPotentialReuse ===\n\n";
    });

    // Initialize state and run backtracking search
    AllocationState state;
    constexpr size_t maxRows = 512; // TMEM has 512 rows

    if (!tryAllocate(allocs, 0, state, maxRows, ctrlOp)) {
      return allocs[0].emitError(
          "allocateTMemAllocs2: failed to allocate TMEM buffers");
    }

    // Apply the final allocation state
    applyAllocationState(allocs, state, bufferId);

    LLVM_DEBUG({
      llvm::dbgs() << "\n=== allocateTMemAllocs2: Final Allocation ===\n";
      for (auto alloc : allocs) {
        auto *buf = getBuffer(alloc.getOperation());
        llvm::dbgs() << "  [" << bufferRange[buf].start() << "-"
                     << bufferRange[buf].end() << ") -> row=" << buf->rowOffset
                     << " col=" << buf->colOffset
                     << " owner=" << buf->isOwnerOfSpace << "\n";
      }
      llvm::dbgs() << "=== End Final Allocation ===\n\n";
    });

    return bufferId;
  }

  FailureOr<unsigned> allocateTMemAllocs(
      SmallVector<triton::nvidia_gpu::TMEMAllocOp> &allocs,
      SmallVector<BufferT *> &buffers,
      DenseMap<Operation *, ttng::TmemDataChannelPost *> &allocToChannel,
      DenseMap<Operation *, size_t> &operationId, Operation *ctrlOp,
      unsigned bufferId) {
    auto alongDependencyChain = [&](Operation *src, Operation *dst,
                                    unsigned depChainCondition) -> bool {
      // consumer of srcAlloc --> producer of dstAlloc
      // consumer partition of srcAllc vs. producer partition of dstAlloc
      auto *srcCh = allocToChannel[src];
      auto *dstCh = allocToChannel[dst];
      if (!srcCh || !dstCh)
        return false;
      if (getAsyncTaskIds(dstCh->getSrcOp()) ==
          getAsyncTaskIds(srcCh->getDstOp()))
        return true;
      return false;
    };
    auto sameLoop = [&](BufferT *alloc) -> bool {
      // cand belongs to ctrlOp.
      if (ctrlOp) {
        auto ctrlInt = getIntervalForCtrlOp(ctrlOp);
        // If alloc also belongs to ctrlOp, return true.
        return bufferRange[alloc].intersects(ctrlInt);
      }
      // For allocs not in an innermost loop
      return false;
    };
    auto getCombinedTasks = [&](BufferT *alloc) -> SmallVector<AsyncTaskId> {
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
    };
    // Should we check source partitions and dst partitions separately?
    auto samePartition = [&](BufferT *alloc, BufferT *cand,
                             unsigned partitionCondition) -> bool {
      if (partitionCondition == 0)
        return true;
      if (partitionCondition == 1) {
        // Check dstPartition of alloc with srcPartiton of cand
        auto *srcCh = allocToChannel[alloc->owner];
        auto *dstCh = allocToChannel[cand->owner];
        if (!srcCh || !dstCh)
          return false;
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
    };

    // buf and cand belong to the same ctrlOp
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
    // Make sure we can place cand at colOffset in the buffer owned by
    // reuseOwner.
    auto checkOtherReuses = [&](BufferT *cand, BufferT *reuseOwner,
                                size_t colOffset) -> bool {
      for (auto *alloc : buffers) {
        if (!alloc->isOwnerOfSpace && alloc->reuseOwner == reuseOwner) {
          Interval candSizeRange = {colOffset, colOffset + cand->colSize};
          Interval allocSizeRange = {alloc->colOffset,
                                     alloc->colOffset + alloc->colSize};
          if (bufferRange[alloc].intersects(bufferRange[cand]) &&
              allocSizeRange.intersects(candSizeRange)) {
            LLVM_DEBUG({
              LDBG("checkOtherReuses conflict "
                   << colOffset << " " << alloc->colOffset << " "
                   << cand->colSize << " " << alloc->colSize);
              alloc->owner->dump();
            });
            return false;
          }
        }
      }
      return true;
    };
    auto findReuseSpace = [&](BufferT *cand, BufferT *reuseOwner,
                              unsigned depChainCondition) -> size_t {
      size_t maxColOffset = 0;
      // Try to find the colOffset in this reuseOwner. If there is already a
      // reuse in the same loop, move up colOffset.
      for (auto *alloc : buffers) {
        if (!alloc->isOwnerOfSpace && alloc->reuseOwner == reuseOwner) {
          if (sameLoop(alloc) ||
              bufferRange[alloc].intersects(bufferRange[cand]))
            maxColOffset =
                std::max(alloc->colOffset + alloc->colSize, maxColOffset);
        }
      }
      LDBG("findReuseSpace first pass maxColOffset " << maxColOffset);
      if (maxColOffset + cand->colSize <= reuseOwner->colSize)
        return maxColOffset;
      if (!sameLoop(reuseOwner)) {
        // owner is not live in this ctrlOp
        // If owner is in a different loop, try to find a buffer in this loop
        // where
        // -- colOffset == 0, in this loop, and along the dependency chain
        for (auto *alloc : buffers) {
          if (!alloc->isOwnerOfSpace && alloc->reuseOwner == reuseOwner &&
              alloc->colOffset == 0 && sameLoop(alloc) &&
              alongDependencyChain(alloc->owner, cand->owner,
                                   depChainCondition)) {
            auto tOffset = findUsesInCtrlOp(alloc, cand);
            LLVM_DEBUG({
              LDBG("findUsesInCtrlOp returns " << tOffset);
              alloc->owner->dump();
            });
            if (tOffset + cand->colSize <= alloc->colSize)
              return tOffset;
          }
        }
      }
      return std::numeric_limits<size_t>::max();
    };
    auto getBuffer = [&](Operation *candAlloc) -> BufferT * {
      for (auto *alloc : buffers) {
        if (alloc->owner == candAlloc)
          return alloc;
      }
      return nullptr;
    };
    // Return true if this is the first reuse of a buffer in "ctrlOp" while the
    // owner of the buffer is in a different ctrlOp.
    auto firstReuseOfBuffer = [&](BufferT *cand) -> bool {
      for (auto alloc : allocs) {
        if (cand->owner == alloc.getOperation()) {
          // later allocs are not handled yet.
          break;
        }
        auto *allocBuf = getBuffer(alloc.getOperation());
        if (allocBuf->reuseOwner == cand->reuseOwner)
          return false;
      }
      return true;
    };
    // partitionCondition: used when buffer owner is in different loop
    // depChainCondition: used when buffer owner is in the same loop
    auto findReuseChannel = [&](BufferT *cand, unsigned partitionCondition,
                                unsigned depChainCondition) -> BufferT * {
      for (auto *alloc : buffers) {
        if (alloc->isOwnerOfSpace) {
          LLVM_DEBUG({
            LDBG("check to reuse buffer owned by " << bufferRange[alloc].start()
                                                   << " "
                                                   << bufferRange[alloc].end());
            alloc->owner->dump();
          });
          // The buffer owner owns a set of rows.
          // If alloc and cand are in different loops, we can reuse as
          // long as they have the same partitions.
          // Otherwise, reuse when there is a dependency chain.
          if (!bufferRange[alloc].intersects(bufferRange[cand]) &&
              alloc->colSize >= cand->colSize &&
              ((!sameLoop(alloc) &&
                samePartition(alloc, cand, partitionCondition)) ||
               (sameLoop(alloc) &&
                alongDependencyChain(alloc->owner, cand->owner,
                                     depChainCondition)))) {
            // Make sure there is no liveness overlap with other buffers using
            // the space.
            auto colOffset = findReuseSpace(cand, alloc, depChainCondition);
            if (colOffset == std::numeric_limits<size_t>::max()) {
              LDBG("-- findReuseSpace fails");
              continue;
            }
            if (!checkOtherReuses(cand, alloc, colOffset)) {
              LDBG("-- checkOtherReuses fails");
              continue;
            }
            cand->isOwnerOfSpace = false; // redundant with reuseOwner?
            cand->rowOffset = alloc->rowOffset;
            cand->colOffset = colOffset;
            cand->reuseOwner = alloc;
            LLVM_DEBUG({
              LDBG("set offset to " << cand->rowOffset << " " << cand->colOffset
                                    << " sameLoop " << sameLoop(alloc) << ":");
              cand->owner->dump();
            });
            return alloc;
          }
          LLVM_DEBUG({
            LDBG("can't reuse owner "
                 << bufferRange[alloc].intersects(bufferRange[cand]));
            alloc->owner->dump();
          });
        }
      }
      return nullptr;
    };
    // interferes with all allocated buffers
    auto allInterfere = [&](BufferT *cand) -> bool {
      for (auto *alloc : buffers) {
        if (alloc->rowOffset != std::numeric_limits<size_t>::max()) {
          if (!bufferRange[alloc].intersects(bufferRange[cand]))
            return false;
        }
      }
      return true;
    };
    auto allocateNewSpace = [&](BufferT *cand, bool allocate) -> bool {
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
    };
    auto getBufferId = [&](Operation *op) -> int {
      auto stageAttr = op->getAttrOfType<IntegerAttr>("buffer.id");
      return stageAttr.getInt();
    };

    // Heuristics: num_buffers is one for each alloc
    // If liveness overlaps, we can't reuse the buffer.
    // Heuristics:
    //   if this buffer interferes with all allocated buffers, allocate new
    //   space; reuse buffers
    //   if belongs to the same loop and along the dependency chain
    //   or belongs to different loops and have the same partitions
    //   if there is enough space, allocate new space otherwise, reuse space
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
      // if this is the first buffer to be allocated, allocate new space.
      // get a list of allocated buffers, check if it interferes
      if (allInterfere(candBuf)) {
        LDBG("\nallInterfere");
        bool hasSpace = allocateNewSpace(candBuf, true);
        if (!hasSpace) {
          return alloc.emitError("can't find tmem space: no new space for "
                                 "tmem alloc when all buffers interfere");
        }
      } else {
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
          if (allocateNewSpace(candBuf, false))
            allocateNewSpace(candBuf, true);
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

      // Initial buffer.copy = 1; post-processing in run() may increase this.
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

    if (!channelId || !bufferId || !bufferCopy || !bufferOffset) {
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
                              StringRef writeDecisionFile = "",
                              int smemAllocAlgo = 0, unsigned smemBudget = 0,
                              bool smemCircularReuse = false) {

  // Step 1: collect all communications between producers and consumers.
  SmallVector<std::unique_ptr<Channel>> channelsOrigin;
  collectPostChannels(channelsOrigin, funcOp);
  SmallVector<Channel *> channels;
  for (const auto &c : channelsOrigin) {
    // Skip guard channels (isSameIterGuard) — they are auxiliary
    // synchronization channels used by the code partition pass and
    // should not influence memory planning decisions.
    if (c->channelKind == DataChannelKind::TMEMPost) {
      auto *tmemCh = static_cast<ttng::TmemDataChannelPost *>(c.get());
      if (tmemCh->isSameIterGuard)
        continue;
    }
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

  // If a read decision file is provided, apply decisions from file instead of
  // running the planner.
  if (!readDecisionFile.empty()) {
    if (failed(readDecisionsFromFile(channels, readDecisionFile))) {
      return failure();
    }
    LDBG("Skipping memory planner - using decisions from file");
    return success();
  }

  // Step 2: figure out smem/tmem sizes and liveness.
  // If two buffers are sharing a multi-staged alloc, the liveness can overlap,
  // otherwise, the liveness can't overlap.

  // Check for per-loop SMEM allocation attributes on the WS ForOp.
  // These override the pass-level defaults, following the same pattern
  // as tt.tmem_alloc_algo.
  int effectiveSmemAllocAlgo = smemAllocAlgo;
  unsigned effectiveSmemBudget = smemBudget;
  bool effectiveSmemCircularReuse = smemCircularReuse;
  funcOp->walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr("tt.warp_specialize"))
      return;
    // Walk from the WS ForOp up through parent ForOps, collecting
    // attributes. The innermost (WS) loop has highest priority.
    SmallVector<scf::ForOp> loopChain;
    loopChain.push_back(forOp);
    for (auto parent = forOp->getParentOfType<scf::ForOp>(); parent;
         parent = parent->getParentOfType<scf::ForOp>()) {
      loopChain.push_back(parent);
    }
    // Apply from outermost to innermost (innermost wins).
    for (auto it = loopChain.rbegin(); it != loopChain.rend(); ++it) {
      auto loop = *it;
      if (auto attr = loop->getAttrOfType<IntegerAttr>("tt.smem_alloc_algo"))
        effectiveSmemAllocAlgo = attr.getInt();
      if (auto attr = loop->getAttrOfType<IntegerAttr>("tt.smem_budget"))
        effectiveSmemBudget = static_cast<unsigned>(attr.getInt());
      if (auto attr = loop->getAttrOfType<BoolAttr>("tt.smem_circular_reuse"))
        effectiveSmemCircularReuse = attr.getValue();
    }
  });

  unsigned bufferId;
  if (effectiveSmemAllocAlgo == 1) {
    // New WSBuffer-based SMEM allocation (Phases 1-5).
    LDBG("using SMEM allocation algorithm 1 (WSBuffer-based)"
         << " smemBudget=" << effectiveSmemBudget
         << " smemCircularReuse=" << effectiveSmemCircularReuse);
    assert(effectiveSmemBudget != 0 && "smem budget is not set");
    // Parse channel annotations from MMA ops for SMEM pre-assignment.
    auto mmaAnnotations = parseChannelAnnotations(funcOp);
    DenseMap<Operation *, ChannelAnnotation> smemAllocAnnotations;
    if (!mmaAnnotations.empty())
      smemAllocAnnotations =
          buildAllocToAnnotationMap(channels, mmaAnnotations);

    bufferId =
        allocateSmemBuffers(funcOp, channels, numBuffers, effectiveSmemBudget,
                            effectiveSmemCircularReuse, smemAllocAnnotations);
  } else {
    // Original SMEM allocation.
    LDBG("using SMEM allocation algorithm 0 (original)");
    Allocation allocation;
    triton::MemoryPlanner planner(funcOp, &allocation, &channels);
    if (failed(planner.run(numBuffers)))
      return failure();
    bufferId = planner.getLastBufferId();
    LLVM_DEBUG(funcOp.dump());
    LLVM_DEBUG(planner.dumpBuffers());
  }

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

  {
    Allocation allocation;
    triton::MemoryPlannerTmem planner(funcOp, &allocation, &channels);
    if (failed(planner.run(bufferId)))
      return failure();
  }

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
                                 writeDecisionFile, smemAllocAlgo, smemBudget,
                                 smemCircularReuse)))
        signalPassFailure();
    }
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
