#include "InsertSemas.h"
#include "MetaToNVWSConvert.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir::triton {
#define GEN_PASS_DEF_NVWSMETATONVWSCONVERT
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
} // namespace mlir::triton

namespace {

static bool isWarpSpecializeLoop(Operation *op) {
  auto loop = dyn_cast<scf::ForOp>(op);
  return loop && loop->hasAttr(kWarpSpecializeAttrName);
}

static bool isNestedInWarpSpecializeLoop(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isWarpSpecializeLoop(parent))
      return true;
  }
  return false;
}

static FailureOr<SmallVector<int>> getNormalizedTaskIds(Operation *op) {
  auto taskIds = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id");
  if (!taskIds)
    taskIds = op->getAttrOfType<DenseI32ArrayAttr>(kPartitionAttrName);
  // A zero-operand scf.yield may be left unassigned by Meta task propagation.
  // It has no value-specific ownership; it terminates its parent region and
  // therefore inherits that region operation's complete task assignment.
  if (!taskIds && isa<scf::YieldOp>(op)) {
    Operation *parent = op->getParentOp();
    taskIds = parent->getAttrOfType<DenseI32ArrayAttr>("async_task_id");
    if (!taskIds)
      taskIds =
          parent->getAttrOfType<DenseI32ArrayAttr>(kPartitionAttrName);
  }
  if (!taskIds)
    return failure();

  SmallVector<int> ids(taskIds.asArrayRef().begin(),
                       taskIds.asArrayRef().end());
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  if (ids.empty() || llvm::any_of(ids, [](int id) { return id < 0; }))
    return failure();
  return ids;
}

static LogicalResult materializePartition(Operation *op) {
  FailureOr<SmallVector<int>> ids = getNormalizedTaskIds(op);
  if (failed(ids))
    return op->emitError(
        "MetaToNVWSConvert requires a non-empty, non-negative "
        "async_task_id or ttg.partition assignment on ")
           << op->getName() << " with attributes " << op->getAttrDictionary();

  auto expected = DenseI32ArrayAttr::get(op->getContext(), *ids);
  if (Attribute existing = op->getAttr(kPartitionAttrName)) {
    if (existing != expected)
      return op->emitError(
          "MetaToNVWSConvert found conflicting async_task_id and "
          "ttg.partition assignments");
  }
  op->setAttr(kPartitionAttrName, expected);
  return success();
}

// Meta uses tt.warp_specialize to identify the inner loop whose schedule is
// analyzed, then specializes the enclosing task-bearing loop nest. NVWS uses
// the attribute as the physical specialization root. Promote Meta's root
// metadata to the outermost enclosing scf.for with the same authored task set;
// leave the inner loop's ordinary pipeline metadata untouched.
static LogicalResult promoteMetaWarpSpecializeRoots(FuncOp func) {
  SmallVector<scf::ForOp> scheduledLoops;
  func.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName))
      scheduledLoops.push_back(loop);
  });

  for (scf::ForOp scheduled : scheduledLoops) {
    FailureOr<SmallVector<int>> scheduledIds =
        getNormalizedTaskIds(scheduled);
    if (failed(scheduledIds))
      return scheduled.emitError(
          "MetaToNVWSConvert cannot determine the task domain of the Meta "
          "warp-specialize loop");

    scf::ForOp root = scheduled;
    for (auto parent = scheduled->getParentOfType<scf::ForOp>(); parent;
         parent = parent->getParentOfType<scf::ForOp>()) {
      if (!parent->hasAttr("async_task_id"))
        break;
      FailureOr<SmallVector<int>> parentIds = getNormalizedTaskIds(parent);
      if (failed(parentIds) || *parentIds != *scheduledIds)
        return parent.emitError(
            "MetaToNVWSConvert found an enclosing loop with a mismatched "
            "async_task_id task domain");
      root = parent;
    }

    if (root == scheduled)
      continue;
    if (root->hasAttr(kWarpSpecializeAttrName))
      return root.emitError(
          "MetaToNVWSConvert found multiple Meta warp-specialize loops for "
          "one promoted root");

    for (StringRef name : {kWarpSpecializeAttrName,
                           kWarpSpecializeTagAttrName,
                           kPartitionStagesAttrName,
                           kPartitionTypesAttrName}) {
      if (Attribute attr = scheduled->getAttr(name)) {
        root->setAttr(name, attr);
        scheduled->removeAttr(name);
      }
    }
  }
  return success();
}

// Work around the tokenized representation produced by early Meta TMA-reduce
// lowering. A direct async token cannot remain an SSA edge between two
// physical NVWS partitions: PartitionLoops would clone the reduce into its
// partition and the wait into another partition, leaving the wait with a use
// of the erased original producer. Keep the drain wait with the operation that
// issued the reduce.
//
// This should eventually be replaced by an `nvws.descriptor_reduce` operation,
// analogous to `nvws.descriptor_load`, so the NVWS pipeline can retain the
// descriptor-level operation through partitioning and lower it afterward
// without introducing a cross-partition async token.
static LogicalResult coLocateDirectTmaReduceWaits(FuncOp func) {
  LogicalResult result = success();
  func.walk([&](TMAStoreTokenWaitOp wait) {
    if (failed(result))
      return;
    auto reduce = wait.getToken().getDefiningOp<AsyncTMAReduceOp>();
    if (!reduce)
      return;
    FailureOr<SmallVector<int>> ids = getNormalizedTaskIds(reduce);
    if (failed(ids)) {
      result = reduce.emitError(
          "MetaToNVWSConvert cannot determine ownership of the direct "
          "async_tma_reduce token producer");
      return;
    }
    auto ownership = DenseI32ArrayAttr::get(func.getContext(), *ids);
    // Set both forms so this is idempotent across the early ownership
    // conversion and the post-planner conversion.
    wait->setAttr("async_task_id", ownership);
    wait->setAttr(kPartitionAttrName, ownership);
  });
  return result;
}

static void insertAll(SetVector<int> &dst, const SetVector<int> &src) {
  dst.insert(src.begin(), src.end());
}

static void collectYieldedValues(Operation *op, unsigned resultIndex,
                                 SmallVectorImpl<Value> &values) {
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    Operation *terminator = loop.getBody()->getTerminator();
    if (resultIndex < terminator->getNumOperands())
      values.push_back(terminator->getOperand(resultIndex));
    return;
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      if (region->empty())
        continue;
      Operation *terminator = region->front().getTerminator();
      if (resultIndex < terminator->getNumOperands())
        values.push_back(terminator->getOperand(resultIndex));
    }
    return;
  }

  if (auto reduce = dyn_cast<ReduceOp>(op)) {
    Region &region = reduce.getCombineOp();
    if (region.empty())
      return;
    Operation *terminator = region.front().getTerminator();
    if (resultIndex < terminator->getNumOperands())
      values.push_back(terminator->getOperand(resultIndex));
  }
}

static void insertPartitionForUser(SetVector<int> &result, Operation *user,
                                   Block *scope = nullptr) {
  if (hasPartition(user)) {
    insertAll(result, getPartitionIds(user));
    return;
  }
  if (!scope)
    return;
  if (Operation *ancestor = scope->findAncestorOpInBlock(*user)) {
    if (hasPartition(ancestor))
      insertAll(result, getPartitionIds(ancestor));
  }
}

static SetVector<int> inferAsyncTokenResultPartition(Operation *op,
                                                     unsigned resultIndex) {
  SetVector<int> result;
  if (resultIndex >= op->getNumResults() ||
      !isa<AsyncTokenType>(op->getResult(resultIndex).getType()))
    return result;

  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    if (resultIndex >= loop.getNumRegionIterArgs())
      return result;
    BlockArgument arg = loop.getRegionIterArg(resultIndex);
    for (OpOperand &use : arg.getUses())
      insertPartitionForUser(result, use.getOwner(), loop.getBody());
    return result;
  }

  SmallVector<Value> yieldedValues;
  collectYieldedValues(op, resultIndex, yieldedValues);
  for (Value value : yieldedValues) {
    if (Operation *def = value.getDefiningOp())
      insertPartitionForUser(result, def);
  }
  return result;
}

static void insertYieldedProducerPartitions(
    SetVector<int> &result, Value value,
    DenseSet<Operation *> &consumedExternalTaskIds) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return;

  if (def->getNumRegions() != 0 && def->hasAttr(kPartitionOutputsAttrName)) {
    auto it = llvm::find(def->getResults(), value);
    if (it != def->result_end()) {
      unsigned index = std::distance(def->result_begin(), it);
      auto outputs = getPartitionOutputs(def);
      if (index < outputs.size()) {
        insertAll(result, outputs[index]);
        return;
      }
    }
  }

  if (hasPartition(def))
    insertAll(result, getPartitionIds(def));
  else if (FailureOr<SmallVector<int>> ids = getNormalizedTaskIds(def);
           succeeded(ids)) {
    result.insert(ids->begin(), ids->end());
    consumedExternalTaskIds.insert(def);
  }
}

static SetVector<int>
inferResultPartition(Operation *op, unsigned resultIndex,
                     DenseSet<Operation *> &consumedExternalTaskIds) {
  SetVector<int> result;
  SmallVector<Value> yieldedValues;
  collectYieldedValues(op, resultIndex, yieldedValues);
  for (Value value : yieldedValues)
    insertYieldedProducerPartitions(result, value, consumedExternalTaskIds);

  insertAll(result, inferAsyncTokenResultPartition(op, resultIndex));
  return result;
}

static LogicalResult materializePartitionOutputs(
    Operation *op, DenseSet<Operation *> &consumedExternalTaskIds) {
  if (!isa<scf::ForOp, scf::IfOp, ReduceOp>(op) || op->getNumResults() == 0)
    return success();
  // A second conversion after NVWSInsertAllocas and memory planning must retain
  // the ownership policy established by the first conversion. Allocation
  // materialization rewrites values but does not change which partitions own
  // region results, and re-inferring from the rewritten access graph can lose
  // the original yielded-producer association.
  if (op->hasAttr(kPartitionOutputsAttrName))
    return success();

  SmallVector<SetVector<int>> outputs;
  outputs.reserve(op->getNumResults());
  SetVector<int> opPartitions = getPartitionIds(op);
  for (unsigned index = 0; index < op->getNumResults(); ++index) {
    SetVector<int> ids =
        inferResultPartition(op, index, consumedExternalTaskIds);
    if (ids.empty())
      return op->emitError(
          "MetaToNVWSConvert cannot determine partition.outputs for result ")
             << index;
    for (int id : ids) {
      if (!opPartitions.contains(id))
        return op->emitError("MetaToNVWSConvert result partition ")
               << id << " is absent from the Meta region assignment";
    }
    outputs.push_back(std::move(ids));
  }
  setPartitionOutputs(op, outputs);
  return success();
}

static int getTxCount(Operation *descOp) {
  RankedTensorType tensorType;
  Value desc;
  if (auto load = dyn_cast<DescriptorLoadOp>(descOp)) {
    tensorType = load.getType();
    desc = load.getDesc();
  } else {
    auto gather = cast<DescriptorGatherOp>(descOp);
    tensorType = gather.getType();
    desc = gather.getDesc();
  }
  Attribute encoding = getEncodingFromDescriptor(descOp, tensorType, desc);
  auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
  return product(shapePerCTA) *
         getIntOrFloatOrPtrBitWidth(tensorType.getElementType()) / 8;
}

static std::optional<int> getEnclosingWarpSpecializeTag(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isWarpSpecializeLoop(parent))
      return getWarpSpecializeTag(parent);
  }
  return std::nullopt;
}

static std::optional<int> getAssociatedWarpSpecializeTag(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (isWarpSpecializeLoop(def))
    return getWarpSpecializeTag(def);
  if (auto tag = getWarpSpecializeTag(def))
    return tag;
  return getEnclosingWarpSpecializeTag(def);
}

static SetVector<int> inferAssociatedWarpSpecializeTags(Operation *op) {
  SetVector<int> tags;
  for (Value operand : op->getOperands())
    if (auto tag = getAssociatedWarpSpecializeTag(operand))
      tags.insert(*tag);
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (auto tag = getWarpSpecializeTag(user))
        tags.insert(*tag);
      if (auto tag = getEnclosingWarpSpecializeTag(user))
        tags.insert(*tag);
    }
  }
  return tags;
}

static bool isManagedMemdescValue(Value value, DenseSet<Value> &seen) {
  if (!seen.insert(value).second)
    return false;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (isa<TMEMAllocOp>(def))
    return true;
  if (auto alloc = dyn_cast<LocalAllocOp>(def))
    return cast<MemDescType>(alloc.getType()).getMutableMemory();
  if (!nvws_semas::isSupportedAliasOp(def))
    return false;
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return isa<MemDescType>(operand.getType()) &&
           isManagedMemdescValue(operand, seen);
  });
}

// InsertSemas replays aliases of managed storage at the partitioned access
// that consumes them. Preserve ownership for an alias defined outside a WS
// loop when its value flows exclusively into that loop; otherwise the replayed
// alias would be a child of the partitioned loop without ttg.partition. Walk
// backwards through complete alias chains, but do not convert unrelated
// external Meta ops or aliases rooted at function arguments.
static FailureOr<DenseSet<Operation *>>
collectExternalAliasesFeedingWarpSpecialize(FuncOp func) {
  DenseSet<Operation *> aliases;
  SmallVector<Operation *, 8> worklist;
  auto enqueueExternalAlias = [&](Value value) {
    Operation *def = value.getDefiningOp();
    if (def && nvws_semas::isSupportedAliasOp(def) &&
        !isWarpSpecializeLoop(def) && !isNestedInWarpSpecializeLoop(def))
      worklist.push_back(def);
  };

  func.walk([&](Operation *op) {
    if (!isWarpSpecializeLoop(op) && !isNestedInWarpSpecializeLoop(op))
      return;
    for (Value operand : op->getOperands())
      enqueueExternalAlias(operand);
  });

  while (!worklist.empty()) {
    Operation *alias = worklist.pop_back_val();
    if (!aliases.insert(alias).second)
      continue;
    for (Value operand : alias->getOperands())
      if (isa<MemDescType>(operand.getType()))
        enqueueExternalAlias(operand);
  }

  for (Operation *alias : llvm::make_early_inc_range(aliases)) {
    DenseSet<Value> seen;
    bool managed = llvm::any_of(alias->getOperands(), [&](Value operand) {
      return isa<MemDescType>(operand.getType()) &&
             isManagedMemdescValue(operand, seen);
    });
    if (!managed)
      aliases.erase(alias);
  }

  for (Operation *alias : aliases) {
    for (Value result : alias->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (aliases.contains(user) || isWarpSpecializeLoop(user) ||
            isNestedInWarpSpecializeLoop(user))
          continue;
        return alias->emitError(
            "MetaToNVWSConvert cannot partition an external managed "
            "memdesc alias that also has a root-scope use");
      }
    }
  }
  return aliases;
}

// PartitionLoops treats partitioned siblings of a WS loop as part of that
// loop's specialization unit. Materialize the tag association already implied
// by their def-use closure, exactly as the former NVWS scheduler finalizer did.
static LogicalResult tagExternalPartitionedOps(FuncOp func,
                                               ArrayRef<scf::ForOp> wsLoops) {
  SetVector<int> loopTags;
  for (scf::ForOp loop : wsLoops) {
    auto tag = getWarpSpecializeTag(loop);
    if (!tag)
      return loop.emitError(
          "MetaToNVWSConvert requires a tag on every Meta WS loop");
    loopTags.insert(*tag);
  }

  bool changed = false;
  do {
    changed = false;
    Operation *ambiguous = nullptr;
    SetVector<int> ambiguousTags;
    func.walk([&](Operation *op) {
      if (!hasPartition(op) || isNestedInWarpSpecializeLoop(op) ||
          isWarpSpecializeLoop(op))
        return WalkResult::advance();

      SetVector<int> tags = inferAssociatedWarpSpecializeTags(op);
      if (tags.empty() && loopTags.size() == 1)
        tags.insert(loopTags.front());
      if (tags.size() > 1) {
        ambiguous = op;
        ambiguousTags = tags;
        return WalkResult::interrupt();
      }
      if (tags.size() == 1) {
        int inferred = tags.front();
        if (auto existing = getWarpSpecializeTag(op)) {
          if (*existing != inferred) {
            ambiguous = op;
            ambiguousTags.insert(*existing);
            ambiguousTags.insert(inferred);
            return WalkResult::interrupt();
          }
        } else {
          setWarpSpecializeTag(op, inferred);
          changed = true;
        }
      }
      return WalkResult::advance();
    });
    if (ambiguous)
      return ambiguous->emitError(
          "MetaToNVWSConvert found an ambiguous WS tag association for an "
          "external partitioned operation");
  } while (changed);

  Operation *untagged = nullptr;
  func.walk([&](Operation *op) {
    if (!hasPartition(op) || isNestedInWarpSpecializeLoop(op) ||
        isWarpSpecializeLoop(op))
      return WalkResult::advance();
    if (!hasWarpSpecializeTag(op)) {
      untagged = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (untagged)
    return untagged->emitError(
        "MetaToNVWSConvert cannot associate external partitioned operation "
        "with a unique WS loop");
  return success();
}

static LogicalResult convertDescriptorStores(FuncOp func) {
  SmallVector<LocalStoreOp> stores;
  func.walk([&](LocalStoreOp store) {
    if (isa_and_nonnull<DescriptorLoadOp, DescriptorGatherOp>(
            store.getSrc().getDefiningOp()))
      stores.push_back(store);
  });

  for (LocalStoreOp store : stores) {
    Operation *descOp = store.getSrc().getDefiningOp();
    FailureOr<SmallVector<int>> descIds = getNormalizedTaskIds(descOp);
    if (failed(descIds))
      return descOp->emitError(
          "MetaToNVWSConvert descriptor producer is missing its Meta task "
          "assignment");
    SetVector<int> expectedPartitions;
    expectedPartitions.insert(descIds->begin(), descIds->end());
    SetVector<int> storePartitions;
    if (hasPartition(store)) {
      storePartitions = getPartitionIds(store);
    } else if (FailureOr<SmallVector<int>> storeIds =
                   getNormalizedTaskIds(store);
               succeeded(storeIds)) {
      storePartitions.insert(storeIds->begin(), storeIds->end());
    }
    if (storePartitions != expectedPartitions)
      return store.emitError(
          "MetaToNVWSConvert descriptor producer and planned local_store "
          "have different task assignments");

    OpBuilder builder(store);
    int txCount = getTxCount(descOp);
    Operation *newOp = nullptr;
    if (auto load = dyn_cast<DescriptorLoadOp>(descOp)) {
      newOp = triton::nvws::DescriptorLoadOp::create(
          builder, load.getLoc(), load.getDesc(), load.getIndices(), txCount,
          store.getDst(), load.getCache(), load.getEvict());
    } else {
      auto gather = cast<DescriptorGatherOp>(descOp);
      newOp = triton::nvws::DescriptorGatherOp::create(
          builder, gather.getLoc(), gather.getDesc(), gather.getXOffsets(),
          gather.getYOffset(), txCount, store.getDst());
    }
    newOp->setAttrs(descOp->getAttrs());
    newOp->setAttr(kPartitionAttrName,
                   DenseI32ArrayAttr::get(store.getContext(), *descIds));
    newOp->removeAttr("async_task_id");

    store.erase();
    if (descOp->use_empty())
      descOp->erase();
  }
  return success();
}

struct SmemPlanPolicy {
  int allocAlgo = 1;
  bool circularReuse = false;
};

static SmemPlanPolicy getEffectiveSmemPlanPolicy(FuncOp func) {
  SmemPlanPolicy policy;
  func.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr(kWarpSpecializeAttrName))
      return;
    SmallVector<scf::ForOp> loopChain{forOp};
    for (auto parent = forOp->getParentOfType<scf::ForOp>(); parent;
         parent = parent->getParentOfType<scf::ForOp>())
      loopChain.push_back(parent);
    for (scf::ForOp loop : llvm::reverse(loopChain)) {
      if (auto attr =
              loop->getAttrOfType<IntegerAttr>("tt.smem_alloc_algo"))
        policy.allocAlgo = attr.getInt();
      if (auto attr =
              loop->getAttrOfType<BoolAttr>("tt.smem_circular_reuse"))
        policy.circularReuse = attr.getValue();
    }
  });
  return policy;
}

static unsigned computeMemDescBytes(MemDescType type) {
  int64_t numElements = 0;
  if (auto padded =
          dyn_cast<PaddedSharedEncodingAttr>(type.getEncoding())) {
    SmallVector<int64_t> unpaddedShape = getShapePerCTA(type);
    numElements = padded.getPaddedSize(unpaddedShape);
  } else {
    numElements = product<int64_t>(getAllocationShapePerCTA(type));
  }
  return static_cast<unsigned>(numElements * type.getElementTypeBitWidth() /
                               8);
}

static bool areReuseTypesCompatible(MemDescType host, MemDescType reuser) {
  return host.getEncoding() == reuser.getEncoding() &&
         host.getMemorySpace() == reuser.getMemorySpace() &&
         host.getElementType() == reuser.getElementType();
}

// Convert Meta's reuse relation to the canonical buffer.* contract consumed by
// InsertSemas. Keep every allocation: same buffer.id forms one logical reuse
// group, while buffer.start selects each reuser's authored logical copy.
static LogicalResult translateReuseTargets(FuncOp func) {
  DenseMap<int64_t, LocalAllocOp> hosts;
  llvm::MapVector<int64_t, SmallVector<LocalAllocOp>> reusers;
  LogicalResult result = success();

  func.walk<WalkOrder::PreOrder>([&](LocalAllocOp alloc) {
    if (!alloc.isSharedMemoryAlloc())
      return;
    if (auto target =
            alloc->getAttrOfType<IntegerAttr>("allocation.reuseTarget")) {
      if (target.getInt() < 0) {
        result = alloc.emitError(
            "MetaToNVWSConvert requires allocation.reuseTarget to be a "
            "non-negative integer buffer id");
        return;
      }
      reusers[target.getInt()].push_back(alloc);
      return;
    }
    if (alloc->hasAttr("buffer.tmaStaging"))
      return;
    if (auto id = alloc->getAttrOfType<IntegerAttr>("buffer.id"))
      hosts.try_emplace(id.getInt(), alloc);
  });
  if (failed(result))
    return failure();

  auto i32 = IntegerType::get(func.getContext(), 32);
  for (auto &[targetId, candidates] : reusers) {
    auto hostIt = hosts.find(targetId);
    if (hostIt == hosts.end()) {
      for (LocalAllocOp candidate : candidates)
        candidate->removeAttr("allocation.reuseTarget");
      continue;
    }

    LocalAllocOp host = hostIt->second;
    auto hostType = cast<MemDescType>(host.getType());
    unsigned hostBytes = computeMemDescBytes(hostType);
    DenseMap<int64_t, int64_t> nextStart;
    for (LocalAllocOp candidate : candidates) {
      auto candidateType = cast<MemDescType>(candidate.getType());
      auto copies = candidate->getAttrOfType<IntegerAttr>("buffer.copy");
      auto originalId = candidate->getAttrOfType<IntegerAttr>("buffer.id");
      bool dominates = host->getBlock() == candidate->getBlock() &&
                       host->isBeforeInBlock(candidate);
      if (!copies || !originalId || copies.getInt() < 1 || !dominates ||
          copies.getInt() * computeMemDescBytes(candidateType) > hostBytes ||
          !areReuseTypesCompatible(hostType, candidateType)) {
        candidate->removeAttr("allocation.reuseTarget");
        continue;
      }
      candidate->setAttr("buffer.id", IntegerAttr::get(i32, targetId));
      candidate->setAttr(
          "buffer.start",
          IntegerAttr::get(
              i32, nextStart[originalId.getInt()]++ % copies.getInt()));
      candidate->removeAttr("allocation.reuseTarget");
    }
  }
  return success();
}

static LogicalResult translateBufferPlan(FuncOp func) {
  SmemPlanPolicy policy = getEffectiveSmemPlanPolicy(func);

  if (policy.allocAlgo != 1 || policy.circularReuse) {
    // Algorithm 0 and explicit circular-reuse plans use repeated IDs as
    // distinct logical ring entries. Preserve the already selected ID/copy
    // policy and add only NVWS's representation attributes.
    llvm::MapVector<int64_t, SmallVector<LocalAllocOp>> circularGroups;
    func.walk<WalkOrder::PreOrder>([&](LocalAllocOp alloc) {
      if (!alloc.isSharedMemoryAlloc() ||
          alloc->hasAttr("allocation.reuseTarget"))
        return;
      if (auto id = alloc->getAttrOfType<IntegerAttr>("buffer.id"))
        circularGroups[id.getInt()].push_back(alloc);
    });
    auto i32 = IntegerType::get(func.getContext(), 32);
    for (auto &[id, group] : circularGroups) {
      (void)id;
      if (group.size() < 2)
        continue;
      for (auto [start, alloc] : llvm::enumerate(group)) {
        alloc->setAttr("buffer.circular", UnitAttr::get(func.getContext()));
        alloc->setAttr("buffer.start",
                       IntegerAttr::get(i32, static_cast<int64_t>(start)));
      }
    }
  }

  return translateReuseTargets(func);
}

static bool isSourceFreeManagedAlloc(Operation *op) {
  if (auto alloc = dyn_cast<LocalAllocOp>(op))
    return !alloc.getSrc();
  if (auto alloc = dyn_cast<TMEMAllocOp>(op))
    return !alloc.getSrc();
  return false;
}

static Block *getTopLevelFunctionBlock(Operation *op, FuncOp func) {
  Block *block = op->getBlock();
  while (block && block->getParent() != &func.getBody()) {
    Operation *parent = block->getParentOp();
    block = parent ? parent->getBlock() : nullptr;
  }
  return block;
}

static InFlightDiagnostic managedBlockError(Operation *op,
                                            StringRef diagnosticPrefix) {
  return op->emitError() << diagnosticPrefix << ": ";
}

static FailureOr<Block *>
getManagedGroupUseBlock(const nvws_semas::GroupDag &group, FuncOp func,
                        StringRef diagnosticPrefix) {
  Operation *anchor = group.pieceTable.members.front().allocOp;
  Block *fallbackBlock = nullptr;
  Block *useBlock = nullptr;
  SmallVector<Value, 8> worklist;
  DenseSet<Value> seen;
  for (const nvws_semas::Member &member : group.pieceTable.members) {
    Block *block = getTopLevelFunctionBlock(member.allocOp, func);
    if (!block)
      return managedBlockError(member.allocOp, diagnosticPrefix)
             << "managed allocation is not nested in the function body";
    if (!fallbackBlock)
      fallbackBlock = block;
    for (Value result : member.allocOp->getResults())
      worklist.push_back(result);
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (isa<BranchOpInterface>(user))
        return managedBlockError(anchor, diagnosticPrefix)
               << "managed memdesc flow through function CFG block arguments "
                  "is unsupported";
      Block *block = getTopLevelFunctionBlock(user, func);
      if (!block)
        return managedBlockError(anchor, diagnosticPrefix)
               << "managed use is not nested in the function body";
      if (useBlock && useBlock != block)
        return managedBlockError(anchor, diagnosticPrefix)
               << "managed memdesc flow across function CFG blocks is "
                  "unsupported";
      useBlock = block;

      if (nvws_semas::isSupportedAliasOp(user))
        for (Value result : user->getResults())
          if (isa<MemDescType>(result.getType()))
            worklist.push_back(result);
      for (Value result : user->getResults())
        if (isa<AsyncTokenType>(result.getType()))
          worklist.push_back(result);

      if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        Operation *parent = yield->getParentOp();
        unsigned index = use.getOperandNumber();
        if (index < parent->getNumResults() &&
            isa<AsyncTokenType>(parent->getResult(index).getType()))
          worklist.push_back(parent->getResult(index));
      } else if (auto condition = dyn_cast<scf::ConditionOp>(user)) {
        unsigned index = use.getOperandNumber();
        auto whileOp = dyn_cast<scf::WhileOp>(condition->getParentOp());
        if (whileOp && index > 0 && index - 1 < whileOp->getNumResults() &&
            isa<AsyncTokenType>(whileOp->getResult(index - 1).getType()))
          worklist.push_back(whileOp->getResult(index - 1));
      }
    }
  }
  return useBlock ? useBlock : fallbackBlock;
}

static LogicalResult verifyManagedGroupBlockLocality(
    const nvws_semas::GroupDag &group, FuncOp func, Block *expectedBlock,
    StringRef diagnosticPrefix) {
  Operation *anchor = group.pieceTable.members.front().allocOp;
  Block *definitionBlock = nullptr;
  for (const nvws_semas::Member &member : group.pieceTable.members) {
    Block *block = getTopLevelFunctionBlock(member.allocOp, func);
    if (!block)
      return managedBlockError(member.allocOp, diagnosticPrefix)
             << "managed allocation is not nested in the function body";
    if (definitionBlock && definitionBlock != block)
      return managedBlockError(anchor, diagnosticPrefix)
             << "one buffer group spans function CFG blocks";
    definitionBlock = block;
  }
  if (definitionBlock != expectedBlock)
    return managedBlockError(anchor, diagnosticPrefix)
           << "managed memdesc flow across function CFG blocks is unsupported";
  return success();
}

static LogicalResult validateManagedAllocationLocalityImpl(
    FuncOp func, StringRef diagnosticPrefix) {
  FailureOr<SmallVector<nvws_semas::GroupDag, 0>> groupsOr =
      nvws_semas::collectGroups(func);
  if (failed(groupsOr))
    return failure();
  for (const nvws_semas::GroupDag &group : *groupsOr) {
    FailureOr<Block *> useBlock =
        getManagedGroupUseBlock(group, func, diagnosticPrefix);
    if (failed(useBlock) ||
        failed(verifyManagedGroupBlockLocality(
            group, func, *useBlock, diagnosticPrefix)))
      return failure();
  }
  return success();
}

// Canonical Meta materializes communication buffers at function entry. NVWS
// semaphore planning intentionally requires each managed buffer lifetime to be
// contained in one top-level function CFG block. Localize only complete,
// source-free final buffer groups whose entire memdesc/token use closure has a
// unique block; retain the strict shared validator for every other shape.
static LogicalResult localizeManagedAllocGroups(FuncOp func) {
  bool hasCompletedPlan = false;
  func.walk([&](Operation *op) {
    if (isa<LocalAllocOp, TMEMAllocOp>(op) &&
        op->hasAttr(nvws_semas::kBufferIdAttrName))
      hasCompletedPlan = true;
  });
  // The bridge may run once before NVWSInsertAllocas and again after memory
  // planning. Only the latter has the final buffer groups to normalize.
  if (!hasCompletedPlan)
    return success();

  FailureOr<SmallVector<nvws_semas::GroupDag, 0>> groupsOr =
      nvws_semas::collectGroups(func);
  if (failed(groupsOr))
    return failure();
  auto &groups = *groupsOr;

  DenseMap<Operation *, unsigned> operationOrder;
  unsigned nextOrder = 0;
  func.walk<WalkOrder::PreOrder>(
      [&](Operation *op) { operationOrder[op] = nextOrder++; });

  llvm::MapVector<Block *, SmallVector<Operation *, 4>> movesByBlock;
  DenseSet<Operation *> scheduled;
  for (const nvws_semas::GroupDag &group : groups) {
    FailureOr<Block *> useBlock =
        getManagedGroupUseBlock(group, func, "MetaToNVWSConvert");
    if (failed(useBlock))
      return failure();

    Block *definitionBlock = nullptr;
    bool movable = true;
    for (const nvws_semas::Member &member : group.pieceTable.members) {
      Block *block = getTopLevelFunctionBlock(member.allocOp, func);
      if (!block || (definitionBlock && definitionBlock != block) ||
          member.allocOp->getBlock() != block ||
          !isSourceFreeManagedAlloc(member.allocOp)) {
        movable = false;
        break;
      }
      definitionBlock = block;
    }
    if (!movable || definitionBlock == *useBlock)
      continue;
    for (const nvws_semas::Member &member : group.pieceTable.members)
      if (scheduled.insert(member.allocOp).second)
        movesByBlock[*useBlock].push_back(member.allocOp);
  }

  for (auto &[targetBlock, moves] : movesByBlock) {
    llvm::sort(moves, [&](Operation *lhs, Operation *rhs) {
      return operationOrder.lookup(lhs) < operationOrder.lookup(rhs);
    });
    Operation *anchor = targetBlock->empty() ? nullptr : &targetBlock->front();
    for (Operation *op : moves) {
      if (anchor)
        op->moveBefore(anchor);
      else
        op->moveBefore(targetBlock, targetBlock->end());
    }
  }

  return validateManagedAllocationLocalityImpl(func, "MetaToNVWSConvert");
}

// Meta buffer allocation rewrites a sourceful local_alloc into a source-free
// local_alloc plus local_store.  The store keeps the task assignment, while
// the newly materialized allocation does not.  Restore that mechanical
// association for that exact one-store representation.
static LogicalResult completePlannedLocalAllocTaskIds(FuncOp func) {
  LogicalResult result = success();
  func.walk([&](LocalAllocOp alloc) {
    if (failed(result) || !isNestedInWarpSpecializeLoop(alloc) ||
        alloc.getSrc() || succeeded(getNormalizedTaskIds(alloc)))
      return;

    std::optional<SmallVector<int>> producerIds;
    for (Operation *user : alloc->getUsers()) {
      auto store = dyn_cast<LocalStoreOp>(user);
      if (!store || store.getDst() != alloc.getResult())
        continue;
      FailureOr<SmallVector<int>> storeIds = getNormalizedTaskIds(store);
      if (failed(storeIds))
        continue;
      if (producerIds) {
        result = alloc.emitError(
            "MetaToNVWSConvert requires exactly one assigned producer store "
            "for a planned local_alloc");
        return;
      }
      producerIds = *storeIds;
    }

    if (producerIds)
      alloc->setAttr("async_task_id",
                     DenseI32ArrayAttr::get(func.getContext(), *producerIds));
  });
  return result;
}

// A source-free TMEM allocation is likewise only a storage handle.  If Meta
// leaves it unassigned inside a WS loop, use the first assigned direct store
// as the deterministic bookkeeping partition.
static void completePlannedTmemAllocTaskIds(FuncOp func) {
  DenseMap<Operation *, SmallVector<int>> storeTaskIds;
  func.walk<WalkOrder::PreOrder>([&](TMEMStoreOp store) {
    auto alloc = store.getDst().getDefiningOp<TMEMAllocOp>();
    if (!alloc || alloc.getSrc() || !isNestedInWarpSpecializeLoop(alloc) ||
        succeeded(getNormalizedTaskIds(alloc)) ||
        storeTaskIds.contains(alloc))
      return;
    FailureOr<SmallVector<int>> ids = getNormalizedTaskIds(store);
    if (succeeded(ids))
      storeTaskIds.try_emplace(alloc, *ids);
  });

  for (auto &[op, ids] : storeTaskIds)
    op->setAttr("async_task_id",
                DenseI32ArrayAttr::get(func.getContext(), ids));
}

static LogicalResult convertFunc(FuncOp func) {
  if (failed(promoteMetaWarpSpecializeRoots(func)))
    return failure();

  SmallVector<scf::ForOp> wsLoops;
  func.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName))
      wsLoops.push_back(loop);
  });
  if (wsLoops.empty())
    return success();

  for (scf::ForOp loop : wsLoops) {
    if (!loop->hasAttr(kWarpSpecializeTagAttrName))
      return loop.emitError(
          "MetaToNVWSConvert requires the Meta warp-specialize tag");
  }

  if (failed(coLocateDirectTmaReduceWaits(func)))
    return failure();

  if (failed(completePlannedLocalAllocTaskIds(func)))
    return failure();
  completePlannedTmemAllocTaskIds(func);

  FailureOr<DenseSet<Operation *>> externalAliases =
      collectExternalAliasesFeedingWarpSpecialize(func);
  if (failed(externalAliases))
    return failure();
  DenseSet<Operation *> consumedExternalTaskIds;
  WalkResult partitionResult = func.walk([&](Operation *op) {
    if (!isWarpSpecializeLoop(op) && !isNestedInWarpSpecializeLoop(op)) {
      if (externalAliases->contains(op)) {
        if (failed(materializePartition(op)))
          return WalkResult::interrupt();
        consumedExternalTaskIds.insert(op);
        return WalkResult::advance();
      }
      op->removeAttr(kPartitionAttrName);
      op->removeAttr(kPartitionOutputsAttrName);
      op->removeAttr(kPartitionStagesAttrName);
      op->removeAttr(kWarpSpecializeTagAttrName);
      return WalkResult::advance();
    }
    if (isa<ub::PoisonOp>(op))
      return WalkResult::advance();
    if (failed(materializePartition(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (partitionResult.wasInterrupted())
    return failure();

  WalkResult outputResult = func.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (!isWarpSpecializeLoop(op) && !isNestedInWarpSpecializeLoop(op))
      return WalkResult::advance();
    if (failed(
            materializePartitionOutputs(op, consumedExternalTaskIds)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (outputResult.wasInterrupted())
    return failure();

  if (failed(convertDescriptorStores(func)) ||
      failed(tagExternalPartitionedOps(func, wsLoops)) ||
      failed(translateBufferPlan(func)) ||
      failed(localizeManagedAllocGroups(func)))
    return failure();

  // Remove each Meta ownership attribute that was consumed into NVWS
  // representation. Function-scope attributes not consulted by conversion
  // remain as unconsumed metadata.
  func.walk([&](Operation *op) {
    if (isWarpSpecializeLoop(op) || isNestedInWarpSpecializeLoop(op) ||
        consumedExternalTaskIds.contains(op))
      op->removeAttr("async_task_id");
  });
  return success();
}

class NVWSMetaToNVWSConvert
    : public mlir::triton::impl::NVWSMetaToNVWSConvertBase<
          NVWSMetaToNVWSConvert> {
public:
  using mlir::triton::impl::NVWSMetaToNVWSConvertBase<
      NVWSMetaToNVWSConvert>::NVWSMetaToNVWSConvertBase;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](FuncOp func) {
      if (failed(convertFunc(func))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    (void)result;
  }
};

} // namespace

LogicalResult mlir::triton::validateNVWSManagedAllocationLocality(
    FuncOp func, StringRef diagnosticPrefix) {
  return validateManagedAllocationLocalityImpl(func, diagnosticPrefix);
}
