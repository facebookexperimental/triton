#include "BufferGroups.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::triton::nvws::buffer {

namespace gpu = triton::gpu;
namespace nvidia_gpu = triton::nvidia_gpu;

std::optional<int64_t> getI64Attr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

bool isSupportedAliasOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.memdesc_index" || name == "ttg.memdesc_subview" ||
         name == "ttg.memdesc_subslice" || name == "ttg.memdesc_trans" ||
         name == "ttg.memdesc_reinterpret" || name == "ttg.memdesc_reshape";
}

static InFlightDiagnostic bufferError(Operation *op,
                                      StringRef diagnosticPrefix) {
  return op->emitError() << diagnosticPrefix << ": ";
}

FailureOr<SmallVector<Group, 0>>
collectGroups(FuncOp func, Block *functionBlock, StringRef diagnosticPrefix) {
  using Buckets = llvm::MapVector<int64_t, SmallVector<Operation *, 2>>;
  Buckets tmemBuckets, localBuckets;
  SmallVector<Operation *, 4> circularLocals;
  int64_t nextSynthetic = -1;
  auto add = [&](Buckets &buckets, Operation *op, std::optional<int64_t> id) {
    int64_t key = id ? *id : nextSynthetic--;
    buckets[key].push_back(op);
  };
  LogicalResult result = success();
  auto collect = [&](Operation *op) {
    std::optional<int64_t> id = getI64Attr(op, kBufferIdAttrName);
    if (isa<nvidia_gpu::TMEMAllocOp>(op)) {
      add(tmemBuckets, op, id);
      return;
    }
    auto alloc = dyn_cast<gpu::LocalAllocOp>(op);
    if (!alloc || !cast<gpu::MemDescType>(alloc.getType()).getMutableMemory())
      return;
    if (!op->hasAttr(kBufferCircularAttrName)) {
      add(localBuckets, op, id);
      return;
    }
    if (!id) {
      result = bufferError(op, diagnosticPrefix)
               << "circular local alloc requires buffer.id";
      return;
    }
    for (StringRef name : {kBufferCopyAttrName, kBufferStartAttrName})
      if (!op->hasAttr(name)) {
        result = bufferError(op, diagnosticPrefix)
                 << "circular local alloc requires " << name;
        return;
      }
    if (op->hasAttr(kBufferOffsetAttrName)) {
      result = bufferError(op, diagnosticPrefix)
               << "circular local alloc must not carry buffer.offset";
      return;
    }
    circularLocals.push_back(op);
  };
  if (functionBlock) {
    for (Operation &op : *functionBlock)
      op.walk(collect);
  } else {
    func.walk(collect);
  }
  if (failed(result))
    return failure();

  SmallVector<Group, 0> groups;
  auto makeGroup = [&](MemoryKind memory, int64_t id,
                       ArrayRef<Operation *> allocations,
                       bool circular = false) {
    Group &group = groups.emplace_back();
    group.bufferId = id;
    group.memory = memory;
    group.circular = circular;
    group.allocations.assign(allocations.begin(), allocations.end());
  };
  for (auto &[id, allocations] : tmemBuckets) {
    std::optional<int64_t> expectedCopy;
    Operation *expectedCopyOp = nullptr;
    for (Operation *op : allocations) {
      std::optional<int64_t> copy = getI64Attr(op, kBufferCopyAttrName);
      if (!copy)
        continue;
      if (!expectedCopy) {
        expectedCopy = copy;
        expectedCopyOp = op;
        continue;
      }
      if (copy == expectedCopy)
        continue;
      InFlightDiagnostic diag = bufferError(op, diagnosticPrefix)
                                << "TMEM allocations sharing buffer.id " << id
                                << " have conflicting buffer.copy values "
                                << *expectedCopy << " and " << *copy;
      diag.attachNote(expectedCopyOp->getLoc())
          << "first buffer.copy value is " << *expectedCopy;
      return failure();
    }
    makeGroup(MemoryKind::Tmem, id, allocations);
  }
  for (auto &[id, allocations] : localBuckets)
    makeGroup(MemoryKind::Local, id, allocations);
  for (Operation *op : circularLocals)
    makeGroup(MemoryKind::Local, *getI64Attr(op, kBufferIdAttrName),
              ArrayRef<Operation *>(op), true);
  return groups;
}

Block *getTopLevelFunctionBlock(Operation *op, FuncOp func) {
  Block *block = op->getBlock();
  while (block && block->getParent() != &func.getBody()) {
    Operation *parent = block->getParentOp();
    block = parent ? parent->getBlock() : nullptr;
  }
  return block;
}

FailureOr<Block *> getManagedGroupUseBlock(const Group &group, FuncOp func,
                                           StringRef diagnosticPrefix) {
  Operation *anchor = group.allocations.front();
  Block *fallbackBlock = nullptr;
  Block *useBlock = nullptr;
  SmallVector<Value, 8> worklist;
  DenseSet<Value> seen;
  for (Operation *allocation : group.allocations) {
    Block *block = getTopLevelFunctionBlock(allocation, func);
    if (!block)
      return bufferError(allocation, diagnosticPrefix)
             << "managed allocation is not nested in the function body";
    if (!fallbackBlock)
      fallbackBlock = block;
    for (Value result : allocation->getResults())
      worklist.push_back(result);
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (isa<BranchOpInterface>(user))
        return bufferError(anchor, diagnosticPrefix)
               << "managed memdesc flow through function CFG block arguments "
                  "is unsupported";
      Block *block = getTopLevelFunctionBlock(user, func);
      if (!block)
        return bufferError(anchor, diagnosticPrefix)
               << "managed use is not nested in the function body";
      if (useBlock && useBlock != block)
        return bufferError(anchor, diagnosticPrefix)
               << "managed memdesc flow across function CFG blocks is "
                  "unsupported";
      useBlock = block;

      if (isSupportedAliasOp(user))
        for (Value result : user->getResults())
          if (isa<gpu::MemDescType>(result.getType()))
            worklist.push_back(result);
      for (Value result : user->getResults())
        if (isa<gpu::AsyncTokenType>(result.getType()))
          worklist.push_back(result);

      if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        Operation *parent = yield->getParentOp();
        unsigned index = use.getOperandNumber();
        if (index < parent->getNumResults() &&
            isa<gpu::AsyncTokenType>(parent->getResult(index).getType()))
          worklist.push_back(parent->getResult(index));
      } else if (auto condition = dyn_cast<scf::ConditionOp>(user)) {
        unsigned index = use.getOperandNumber();
        auto whileOp = dyn_cast<scf::WhileOp>(condition->getParentOp());
        if (whileOp && index > 0 && index - 1 < whileOp->getNumResults() &&
            isa<gpu::AsyncTokenType>(whileOp->getResult(index - 1).getType()))
          worklist.push_back(whileOp->getResult(index - 1));
      }
    }
  }
  return useBlock ? useBlock : fallbackBlock;
}

static LogicalResult
verifyManagedGroupBlockLocality(const Group &group, FuncOp func,
                                Block *expectedBlock,
                                StringRef diagnosticPrefix) {
  Operation *anchor = group.allocations.front();
  Block *definitionBlock = nullptr;
  for (Operation *allocation : group.allocations) {
    Block *block = getTopLevelFunctionBlock(allocation, func);
    if (!block)
      return bufferError(allocation, diagnosticPrefix)
             << "managed allocation is not nested in the function body";
    if (definitionBlock && definitionBlock != block)
      return bufferError(anchor, diagnosticPrefix)
             << "one buffer group spans function CFG blocks";
    definitionBlock = block;
  }
  if (definitionBlock != expectedBlock)
    return bufferError(anchor, diagnosticPrefix)
           << "managed memdesc flow across function CFG blocks is unsupported";
  return success();
}

LogicalResult validateManagedAllocationLocality(FuncOp func,
                                                StringRef diagnosticPrefix) {
  FailureOr<SmallVector<Group, 0>> groupsOr = collectGroups(func);
  if (failed(groupsOr))
    return failure();
  for (const Group &group : *groupsOr) {
    FailureOr<Block *> useBlock =
        getManagedGroupUseBlock(group, func, diagnosticPrefix);
    if (failed(useBlock) || failed(verifyManagedGroupBlockLocality(
                                group, func, *useBlock, diagnosticPrefix)))
      return failure();
  }
  return success();
}

} // namespace mlir::triton::nvws::buffer
