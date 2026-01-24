#include "triton/Analysis/BarrierAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

StringRef barrierOpKindToString(BarrierOpKind kind) {
  switch (kind) {
  case BarrierOpKind::InitBarrier:
    return "init_barrier";
  case BarrierOpKind::InvalBarrier:
    return "inval_barrier";
  case BarrierOpKind::BarrierExpect:
    return "barrier_expect";
  case BarrierOpKind::WaitBarrier:
    return "wait_barrier";
  case BarrierOpKind::ArriveBarrier:
    return "arrive_barrier";
  case BarrierOpKind::NamedBarrierArrive:
    return "named_barrier_arrive";
  case BarrierOpKind::NamedBarrierWait:
    return "named_barrier_wait";
  case BarrierOpKind::AsyncCopy:
    return "async_copy";
  case BarrierOpKind::TMALoad:
    return "tma_load";
  case BarrierOpKind::TMAStore:
    return "tma_store";
  }
  llvm_unreachable("Unknown BarrierOpKind");
}

static StringRef dependencyKindToString(BarrierDependency::Kind kind) {
  switch (kind) {
  case BarrierDependency::Kind::ArriveThenWait:
    return "arrive->wait";
  case BarrierDependency::Kind::ExpectThenWait:
    return "expect->wait";
  case BarrierDependency::Kind::InitThenUse:
    return "init->use";
  }
  llvm_unreachable("Unknown dependency kind");
}

static StringRef
issueKindToString(BarrierExecutionOrderAnalysis::AnalysisIssue::Kind kind) {
  switch (kind) {
  case BarrierExecutionOrderAnalysis::AnalysisIssue::Kind::MissingArrive:
    return "MISSING_ARRIVE";
  case BarrierExecutionOrderAnalysis::AnalysisIssue::Kind::MissingWait:
    return "MISSING_WAIT";
  case BarrierExecutionOrderAnalysis::AnalysisIssue::Kind::PhaseConflict:
    return "PHASE_CONFLICT";
  case BarrierExecutionOrderAnalysis::AnalysisIssue::Kind::DeadlockRisk:
    return "DEADLOCK_RISK";
  case BarrierExecutionOrderAnalysis::AnalysisIssue::Kind::BarrierReuse:
    return "BARRIER_REUSE";
  }
  llvm_unreachable("Unknown issue kind");
}

//===----------------------------------------------------------------------===//
// BarrierOpInfo
//===----------------------------------------------------------------------===//

void BarrierOpInfo::print(llvm::raw_ostream &os) const {
  os << "  [" << orderIndex << "] " << barrierOpKindToString(kind);
  os << " (bar=" << barrierId;
  if (phase >= 0)
    os << ", phase=" << phase;
  if (numThreads >= 0)
    os << ", threads=" << numThreads;
  if (arriveCount > 1)
    os << ", count=" << arriveCount;
  if (expectedBytes >= 0)
    os << ", bytes=" << expectedBytes;
  os << ")";
  if (asyncTaskId >= 0)
    os << " [WG" << asyncTaskId << "]";
  if (op)
    os << " @ " << op->getLoc();
  os << "\n";
}

//===----------------------------------------------------------------------===//
// BarrierDependency
//===----------------------------------------------------------------------===//

void BarrierDependency::print(llvm::raw_ostream &os) const {
  os << "  ";
  if (producer) {
    os << barrierOpKindToString(producer->kind) << "[" << producer->orderIndex
       << "]";
    if (producer->asyncTaskId >= 0)
      os << "(WG" << producer->asyncTaskId << ")";
  }
  os << " --[" << dependencyKindToString(kind) << "]--> ";
  if (consumer) {
    os << barrierOpKindToString(consumer->kind) << "[" << consumer->orderIndex
       << "]";
    if (consumer->asyncTaskId >= 0)
      os << "(WG" << consumer->asyncTaskId << ")";
  }
  if (isCrossWarpGroup())
    os << " [CROSS-WG]";
  os << "\n";
}

//===----------------------------------------------------------------------===//
// BarrierExecutionOrderAnalysis
//===----------------------------------------------------------------------===//

BarrierExecutionOrderAnalysis::BarrierExecutionOrderAnalysis(
    FunctionOpInterface funcOp)
    : funcOp(funcOp) {}

void BarrierExecutionOrderAnalysis::run() {
  collectBarrierOps();
  assignBarrierIds();
  groupByWarpGroup();
  analyzeDependencies();
  detectIssues();
}

int64_t BarrierExecutionOrderAnalysis::getAsyncTaskId(Operation *op) const {
  // First check the operation itself for async_task_id attribute
  // This is the standard way async task IDs are assigned in Triton
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id")) {
    auto arr = attr.asArrayRef();
    if (!arr.empty())
      return arr[0]; // Return first task ID if multiple
  }

  // Check for ttg.partition attribute (used for partition assignments)
  if (auto attr = op->getAttrOfType<IntegerAttr>("ttg.partition")) {
    int64_t idx = attr.getInt();
    if (idx >= 0)
      return idx;
  }

  // Walk up the parent operations to find warp partition region
  Operation *current = op->getParentOp();
  while (current) {
    // Check for async_task_id on parent ops (DenseI32ArrayAttr)
    if (auto attr =
            current->getAttrOfType<DenseI32ArrayAttr>("async_task_id")) {
      auto arr = attr.asArrayRef();
      if (!arr.empty())
        return arr[0];
    }

    // Check for ttg.partition attribute
    if (auto attr = current->getAttrOfType<IntegerAttr>("ttg.partition")) {
      int64_t idx = attr.getInt();
      if (idx >= 0)
        return idx;
    }

    // Check if we're inside a WarpSpecializePartitionsOp region
    if (auto partitionsOp =
            dyn_cast<gpu::WarpSpecializePartitionsOp>(current)) {
      // Find which partition region the operation is in
      Region *opRegion = op->getParentRegion();
      while (opRegion && opRegion->getParentOp() != partitionsOp) {
        opRegion = opRegion->getParentOp()
                       ? opRegion->getParentOp()->getParentRegion()
                       : nullptr;
      }
      if (opRegion) {
        // Find the index of this region in the partitions
        int64_t idx = 0;
        for (Region &region : partitionsOp.getPartitionRegions()) {
          if (&region == opRegion)
            return idx;
          idx++;
        }
      }
    }

    // Check if we're inside a WarpSpecializeOp
    if (auto wsOp = dyn_cast<gpu::WarpSpecializeOp>(current)) {
      // Check if we're in the default region (task ID = -1 or 0 depending on
      // convention)
      if (op->getParentRegion() == &wsOp.getDefaultRegion()) {
        return 0; // Default region is task 0
      }
      // Otherwise we're in a partition region, handled by
      // WarpSpecializePartitionsOp above
    }

    current = current->getParentOp();
  }

  return -1; // Not in an async task region
}

/// Helper to extract a constant index from a memdesc_index operation
static std::optional<int64_t> getConstantIndex(gpu::MemDescIndexOp indexOp) {
  if (!indexOp)
    return 0;

  // Get the index value (single integer index)
  Value indexVal = indexOp.getIndex();
  if (!indexVal)
    return 0;

  // Try to get the index as a constant
  if (auto constOp = indexVal.getDefiningOp<arith::ConstantIntOp>()) {
    return constOp.value();
  }
  if (auto constOp = indexVal.getDefiningOp<arith::ConstantIndexOp>()) {
    return constOp.value();
  }
  return std::nullopt; // Dynamic index
}

/// Helper to trace a barrier value back to its (local_alloc, index) pair
static std::pair<Operation *, int64_t>
getBarrierAllocAndIndex(Value barrierVal,
                        std::function<Value(Value)> traceBlockArgs) {
  // Trace through block arguments first
  barrierVal = traceBlockArgs(barrierVal);

  Operation *defOp = barrierVal.getDefiningOp();
  if (!defOp)
    return {nullptr, -1};

  // If this is a memdesc_index, get the index and trace to source
  if (auto indexOp = dyn_cast<gpu::MemDescIndexOp>(defOp)) {
    auto idx = getConstantIndex(indexOp);
    if (!idx)
      return {nullptr, -1}; // Dynamic index, can't track

    Value src = indexOp.getSrc();
    src = traceBlockArgs(src);

    // Continue tracing if nested memdesc_index
    while (auto nestedIndex =
               dyn_cast_or_null<gpu::MemDescIndexOp>(src.getDefiningOp())) {
      src = traceBlockArgs(nestedIndex.getSrc());
    }

    Operation *allocOp = src.getDefiningOp();
    if (allocOp)
      return {allocOp, *idx};
  }

  return {nullptr, -1};
}

Value BarrierExecutionOrderAnalysis::traceValueThroughBlockArgs(
    Value value) const {
  // Trace a value through block arguments to find its origin
  // This handles the case where barriers are passed as explicit captures
  // to WarpSpecializeOp and used as block arguments in partition regions

  while (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *block = blockArg.getOwner();
    Operation *parentOp = block->getParentOp();

    // Check if we're in a partition region of WarpSpecializePartitionsOp
    if (auto partitionsOp =
            dyn_cast<gpu::WarpSpecializePartitionsOp>(parentOp)) {
      // Find which region this block belongs to
      unsigned argIndex = blockArg.getArgNumber();

      // Get the parent WarpSpecializeOp
      if (auto wsOp =
              dyn_cast<gpu::WarpSpecializeOp>(partitionsOp->getParentOp())) {
        // The explicit captures of WarpSpecializeOp are passed to all
        // partition regions
        if (argIndex < wsOp.getExplicitCaptures().size()) {
          value = wsOp.getExplicitCaptures()[argIndex];
          continue;
        }
      }
    }

    // Cannot trace further
    break;
  }

  return value;
}

std::optional<int64_t>
BarrierExecutionOrderAnalysis::getBarrierAllocId(Operation *op) const {
  // Get the barrier allocation value and trace it back to its defining op
  Value barrierAlloc;

  using nvidia_gpu::ArriveBarrierOp;
  using nvidia_gpu::BarrierExpectOp;
  using nvidia_gpu::InitBarrierOp;
  using nvidia_gpu::InvalBarrierOp;
  using nvidia_gpu::WaitBarrierOp;

  if (auto initOp = dyn_cast<InitBarrierOp>(op)) {
    barrierAlloc = initOp.getAlloc();
  } else if (auto invalOp = dyn_cast<InvalBarrierOp>(op)) {
    barrierAlloc = invalOp.getAlloc();
  } else if (auto expectOp = dyn_cast<BarrierExpectOp>(op)) {
    barrierAlloc = expectOp.getAlloc();
  } else if (auto waitOp = dyn_cast<WaitBarrierOp>(op)) {
    barrierAlloc = waitOp.getAlloc();
  } else if (auto arriveOp = dyn_cast<ArriveBarrierOp>(op)) {
    barrierAlloc = arriveOp.getAlloc();
  }

  if (!barrierAlloc)
    return std::nullopt;

  // Create a lambda wrapper for traceValueThroughBlockArgs
  auto traceBlockArgs = [this](Value v) -> Value {
    return this->traceValueThroughBlockArgs(v);
  };

  // Try to get (alloc, index) pair first - this is the most reliable method
  auto [allocOp, idx] = getBarrierAllocAndIndex(barrierAlloc, traceBlockArgs);
  if (allocOp) {
    auto key = std::make_pair(allocOp, idx);
    auto it = allocIndexToBarrierId.find(key);
    if (it != allocIndexToBarrierId.end()) {
      return it->second;
    }
  }

  // Fall back to direct def op lookup
  barrierAlloc = traceValueThroughBlockArgs(barrierAlloc);
  Operation *defOp = barrierAlloc.getDefiningOp();
  while (defOp) {
    // Check if this is a local_alloc or similar
    if (defOp->hasTrait<OpTrait::IsIsolatedFromAbove>())
      break;

    auto it = defOpToBarrierId.find(defOp);
    if (it != defOpToBarrierId.end()) {
      return it->second;
    }

    // Try to trace through memdesc_index operations
    if (auto indexOp = dyn_cast<gpu::MemDescIndexOp>(defOp)) {
      Value src = indexOp.getSrc();
      src = traceValueThroughBlockArgs(src);
      defOp = src.getDefiningOp();
      continue;
    }

    // Try to get parent allocation
    if (defOp->getNumOperands() > 0) {
      Value operand = defOp->getOperand(0);
      operand = traceValueThroughBlockArgs(operand);
      defOp = operand.getDefiningOp();
    } else {
      break;
    }
  }

  return std::nullopt;
}

void BarrierExecutionOrderAnalysis::collectBarrierOps() {
  size_t orderIndex = 0;

  funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    BarrierOpInfo info;
    info.op = op;
    info.orderIndex = orderIndex++;
    info.asyncTaskId = getAsyncTaskId(op);

    using nvidia_gpu::ArriveBarrierOp;
    using nvidia_gpu::BarrierExpectOp;
    using nvidia_gpu::InitBarrierOp;
    using nvidia_gpu::InvalBarrierOp;
    using nvidia_gpu::NamedBarrierArriveOp;
    using nvidia_gpu::NamedBarrierWaitOp;
    using nvidia_gpu::WaitBarrierOp;

    bool isBarrierOp =
        llvm::TypeSwitch<Operation *, bool>(op)
            .Case<InitBarrierOp>([&](auto initOp) {
              info.kind = BarrierOpKind::InitBarrier;
              info.arriveCount = initOp.getCount();
              return true;
            })
            .Case<InvalBarrierOp>([&](auto) {
              info.kind = BarrierOpKind::InvalBarrier;
              return true;
            })
            .Case<BarrierExpectOp>([&](auto expectOp) {
              info.kind = BarrierOpKind::BarrierExpect;
              info.expectedBytes = expectOp.getSize();
              return true;
            })
            .Case<WaitBarrierOp>([&](auto waitOp) {
              info.kind = BarrierOpKind::WaitBarrier;
              // Try to get static phase value
              if (auto phaseOp =
                      waitOp.getPhase()
                          .template getDefiningOp<arith::ConstantIntOp>()) {
                info.phase = phaseOp.value();
              }
              return true;
            })
            .Case<ArriveBarrierOp>([&](auto arriveOp) {
              info.kind = BarrierOpKind::ArriveBarrier;
              info.arriveCount = arriveOp.getCount();
              return true;
            })
            .Case<NamedBarrierArriveOp>([&](auto namedArriveOp) {
              info.kind = BarrierOpKind::NamedBarrierArrive;
              // Get barrier ID from constant if possible
              if (auto barOp =
                      namedArriveOp.getBar()
                          .template getDefiningOp<arith::ConstantIntOp>()) {
                info.barrierId = barOp.value();
              }
              if (auto threadsOp =
                      namedArriveOp.getNumThreads()
                          .template getDefiningOp<arith::ConstantIntOp>()) {
                info.numThreads = threadsOp.value();
              }
              return true;
            })
            .Case<NamedBarrierWaitOp>([&](auto namedWaitOp) {
              info.kind = BarrierOpKind::NamedBarrierWait;
              if (auto barOp =
                      namedWaitOp.getBar()
                          .template getDefiningOp<arith::ConstantIntOp>()) {
                info.barrierId = barOp.value();
              }
              if (auto threadsOp =
                      namedWaitOp.getNumThreads()
                          .template getDefiningOp<arith::ConstantIntOp>()) {
                info.numThreads = threadsOp.value();
              }
              return true;
            })
            .Default([](Operation *) { return false; });

    if (isBarrierOp) {
      barrierOps.push_back(info);
    }
  });
}

void BarrierExecutionOrderAnalysis::assignBarrierIds() {
  using nvidia_gpu::InitBarrierOp;

  // Create a lambda wrapper for traceValueThroughBlockArgs
  auto traceBlockArgs = [this](Value v) -> Value {
    return this->traceValueThroughBlockArgs(v);
  };

  // First pass: assign IDs to init operations
  // We use (local_alloc, index) pairs to match barriers across different
  // memdesc_index operations
  for (auto &info : barrierOps) {
    if (info.kind == BarrierOpKind::InitBarrier) {
      if (auto initOp = dyn_cast<InitBarrierOp>(info.op)) {
        Value alloc = initOp.getAlloc();

        // Get the (alloc, index) pair for this barrier
        auto [allocOp, idx] = getBarrierAllocAndIndex(alloc, traceBlockArgs);

        if (allocOp) {
          auto key = std::make_pair(allocOp, idx);
          if (allocIndexToBarrierId.find(key) == allocIndexToBarrierId.end()) {
            allocIndexToBarrierId[key] = nextBarrierId++;
          }
          info.barrierId = allocIndexToBarrierId[key];
        }

        // Also register the immediate defining op for direct lookups
        alloc = traceValueThroughBlockArgs(alloc);
        Operation *defOp = alloc.getDefiningOp();
        if (defOp && defOpToBarrierId.find(defOp) == defOpToBarrierId.end()) {
          if (info.barrierId >= 0) {
            defOpToBarrierId[defOp] = info.barrierId;
          } else {
            info.barrierId = nextBarrierId;
            defOpToBarrierId[defOp] = nextBarrierId++;
          }
        } else if (defOp) {
          info.barrierId = defOpToBarrierId[defOp];
        }
      }
    }
  }

  // Second pass: assign IDs to other barrier ops based on their allocation
  for (auto &info : barrierOps) {
    if (info.barrierId < 0 && info.kind != BarrierOpKind::NamedBarrierArrive &&
        info.kind != BarrierOpKind::NamedBarrierWait) {
      if (auto allocId = getBarrierAllocId(info.op)) {
        info.barrierId = *allocId;
      }
    }
  }
}

void BarrierExecutionOrderAnalysis::groupByWarpGroup() {
  for (const auto &info : barrierOps) {
    int64_t taskId = info.asyncTaskId;

    if (warpGroups.find(taskId) == warpGroups.end()) {
      WarpGroupInfo wgInfo;
      wgInfo.taskId = taskId;
      if (taskId < 0) {
        wgInfo.taskName = "main";
      } else {
        wgInfo.taskName = "WG" + std::to_string(taskId);
      }
      warpGroups[taskId] = wgInfo;
    }

    warpGroups[taskId].barrierOps.push_back(&info);
    warpGroups[taskId].orderedOps.push_back(&info);

    // Determine if producer or consumer
    switch (info.kind) {
    case BarrierOpKind::InitBarrier:
    case BarrierOpKind::BarrierExpect:
    case BarrierOpKind::ArriveBarrier:
    case BarrierOpKind::NamedBarrierArrive:
    case BarrierOpKind::TMALoad:
    case BarrierOpKind::TMAStore:
    case BarrierOpKind::AsyncCopy:
      warpGroups[taskId].isProducer = true;
      break;
    case BarrierOpKind::WaitBarrier:
    case BarrierOpKind::NamedBarrierWait:
      warpGroups[taskId].isConsumer = true;
      break;
    default:
      break;
    }
  }

  // Sort operations within each warp group by order index
  for (auto &[taskId, wgInfo] : warpGroups) {
    llvm::sort(wgInfo.orderedOps,
               [](const BarrierOpInfo *a, const BarrierOpInfo *b) {
                 return a->orderIndex < b->orderIndex;
               });
  }
}

void BarrierExecutionOrderAnalysis::analyzeDependencies() {
  // Build maps for fast lookup
  std::map<int64_t, SmallVector<const BarrierOpInfo *>> barrierIdToArrives;
  std::map<int64_t, SmallVector<const BarrierOpInfo *>> barrierIdToWaits;
  std::map<int64_t, SmallVector<const BarrierOpInfo *>> barrierIdToExpects;
  std::map<int64_t, const BarrierOpInfo *> barrierIdToInit;

  for (const auto &info : barrierOps) {
    switch (info.kind) {
    case BarrierOpKind::InitBarrier:
      barrierIdToInit[info.barrierId] = &info;
      break;
    case BarrierOpKind::ArriveBarrier:
    case BarrierOpKind::NamedBarrierArrive:
      barrierIdToArrives[info.barrierId].push_back(&info);
      break;
    case BarrierOpKind::WaitBarrier:
    case BarrierOpKind::NamedBarrierWait:
      barrierIdToWaits[info.barrierId].push_back(&info);
      break;
    case BarrierOpKind::BarrierExpect:
      barrierIdToExpects[info.barrierId].push_back(&info);
      break;
    default:
      break;
    }
  }

  // Create init->use dependencies
  for (const auto &[barrierId, initInfo] : barrierIdToInit) {
    for (const auto *arriveInfo : barrierIdToArrives[barrierId]) {
      BarrierDependency dep;
      dep.producer = initInfo;
      dep.consumer = arriveInfo;
      dep.kind = BarrierDependency::Kind::InitThenUse;
      dependencies.push_back(dep);
    }
    for (const auto *waitInfo : barrierIdToWaits[barrierId]) {
      BarrierDependency dep;
      dep.producer = initInfo;
      dep.consumer = waitInfo;
      dep.kind = BarrierDependency::Kind::InitThenUse;
      dependencies.push_back(dep);
    }
  }

  // Create arrive->wait dependencies
  for (const auto &[barrierId, arrives] : barrierIdToArrives) {
    for (const auto *arriveInfo : arrives) {
      for (const auto *waitInfo : barrierIdToWaits[barrierId]) {
        // Only create dependency if arrive could potentially satisfy wait
        // (based on ordering and phase)
        BarrierDependency dep;
        dep.producer = arriveInfo;
        dep.consumer = waitInfo;
        dep.kind = BarrierDependency::Kind::ArriveThenWait;
        dependencies.push_back(dep);
      }
    }
  }

  // Create expect->wait dependencies (for TMA)
  for (const auto &[barrierId, expects] : barrierIdToExpects) {
    for (const auto *expectInfo : expects) {
      for (const auto *waitInfo : barrierIdToWaits[barrierId]) {
        BarrierDependency dep;
        dep.producer = expectInfo;
        dep.consumer = waitInfo;
        dep.kind = BarrierDependency::Kind::ExpectThenWait;
        dependencies.push_back(dep);
      }
    }
  }
}

void BarrierExecutionOrderAnalysis::detectIssues() {
  // Check for waits without corresponding arrives
  std::map<int64_t, SmallVector<const BarrierOpInfo *>> barrierIdToArrives;
  std::map<int64_t, SmallVector<const BarrierOpInfo *>> barrierIdToWaits;

  for (const auto &info : barrierOps) {
    if (info.kind == BarrierOpKind::ArriveBarrier ||
        info.kind == BarrierOpKind::NamedBarrierArrive) {
      barrierIdToArrives[info.barrierId].push_back(&info);
    } else if (info.kind == BarrierOpKind::WaitBarrier ||
               info.kind == BarrierOpKind::NamedBarrierWait) {
      barrierIdToWaits[info.barrierId].push_back(&info);
    }
  }

  // Check for orphaned waits
  for (const auto &[barrierId, waits] : barrierIdToWaits) {
    if (barrierIdToArrives.find(barrierId) == barrierIdToArrives.end() ||
        barrierIdToArrives[barrierId].empty()) {
      AnalysisIssue issue;
      issue.kind = AnalysisIssue::Kind::MissingArrive;
      issue.involvedOps.append(waits.begin(), waits.end());
      issue.message = "Wait on barrier " + std::to_string(barrierId) +
                      " without matching arrive";
      issues.push_back(issue);
    }
  }

  // Check for orphaned arrives (less critical, might be intentional)
  for (const auto &[barrierId, arrives] : barrierIdToArrives) {
    if (barrierIdToWaits.find(barrierId) == barrierIdToWaits.end() ||
        barrierIdToWaits[barrierId].empty()) {
      AnalysisIssue issue;
      issue.kind = AnalysisIssue::Kind::MissingWait;
      issue.involvedOps.append(arrives.begin(), arrives.end());
      issue.message = "Arrive on barrier " + std::to_string(barrierId) +
                      " without matching wait (might be intentional)";
      issues.push_back(issue);
    }
  }

  // TODO: Add more sophisticated deadlock detection
  // This would require building a wait-for graph and checking for cycles
}

ArrayRef<const BarrierOpInfo *>
BarrierExecutionOrderAnalysis::getBarrierOpsForWarpGroup(int64_t taskId) const {
  auto it = warpGroups.find(taskId);
  if (it != warpGroups.end()) {
    return it->second.barrierOps;
  }
  return {};
}

const WarpGroupInfo *
BarrierExecutionOrderAnalysis::getWarpGroupInfo(int64_t taskId) const {
  auto it = warpGroups.find(taskId);
  if (it != warpGroups.end()) {
    return &it->second;
  }
  return nullptr;
}

void BarrierExecutionOrderAnalysis::print(llvm::raw_ostream &os) const {
  os << "========================================\n";
  os << " Barrier Execution Order Analysis\n";
  os << "========================================\n\n";

  // Print barrier operations grouped by warp group
  os << "Barrier Operations by Warp Group:\n";
  os << "─────────────────────────────────\n";

  for (const auto &[taskId, wgInfo] : warpGroups) {
    os << "\n[" << wgInfo.taskName << "]";
    if (wgInfo.isProducer)
      os << " (Producer)";
    if (wgInfo.isConsumer)
      os << " (Consumer)";
    os << "\n";

    for (const auto *opInfo : wgInfo.orderedOps) {
      opInfo->print(os);
    }
  }

  // Print dependencies
  os << "\nDependencies:\n";
  os << "─────────────\n";
  for (const auto &dep : dependencies) {
    dep.print(os);
  }

  // Print cross-warp-group dependencies
  os << "\nCross-Warp-Group Dependencies:\n";
  os << "──────────────────────────────\n";
  bool hasCrossWG = false;
  for (const auto &dep : dependencies) {
    if (dep.isCrossWarpGroup()) {
      dep.print(os);
      hasCrossWG = true;
    }
  }
  if (!hasCrossWG) {
    os << "  (none)\n";
  }

  // Print issues
  if (!issues.empty()) {
    os << "\nPotential Issues:\n";
    os << "─────────────────\n";
    for (const auto &issue : issues) {
      os << "  [" << issueKindToString(issue.kind) << "] " << issue.message
         << "\n";
    }
  }
}

void BarrierExecutionOrderAnalysis::printExecutionTrace(
    llvm::raw_ostream &os) const {
  os << "\n========================================\n";
  os << " Execution Trace Visualization\n";
  os << "========================================\n\n";

  if (barrierOps.empty()) {
    os << "No barrier operations found.\n";
    return;
  }

  // Find the range of order indices
  size_t maxOrder = 0;
  for (const auto &info : barrierOps) {
    maxOrder = std::max(maxOrder, info.orderIndex);
  }

  // Create timeline for each warp group
  const int scale = 3;
  const int width = (maxOrder + 1) * scale + 10;

  // Print time ruler
  os << "Order:  ";
  for (size_t i = 0; i <= maxOrder; i += 2) {
    os << llvm::format("%-6zu", i);
  }
  os << "\n        │";
  for (size_t i = 0; i < maxOrder * scale; i++) {
    os << "─";
  }
  os << "\n";

  // Symbols for different operations
  auto getSymbol = [](BarrierOpKind kind) -> char {
    switch (kind) {
    case BarrierOpKind::InitBarrier:
      return 'I';
    case BarrierOpKind::InvalBarrier:
      return 'X';
    case BarrierOpKind::BarrierExpect:
      return 'E';
    case BarrierOpKind::WaitBarrier:
    case BarrierOpKind::NamedBarrierWait:
      return 'W';
    case BarrierOpKind::ArriveBarrier:
    case BarrierOpKind::NamedBarrierArrive:
      return 'A';
    case BarrierOpKind::TMALoad:
      return 'L';
    case BarrierOpKind::TMAStore:
      return 'S';
    case BarrierOpKind::AsyncCopy:
      return 'C';
    default:
      return '?';
    }
  };

  // Print each warp group's timeline
  for (const auto &[taskId, wgInfo] : warpGroups) {
    std::string line(width, ' ');

    for (const auto *opInfo : wgInfo.orderedOps) {
      int pos = 8 + opInfo->orderIndex * scale;
      if (pos < (int)line.size()) {
        line[pos] = getSymbol(opInfo->kind);
      }
    }

    os << llvm::format("%-6s", wgInfo.taskName.c_str()) << "│" << line << "\n";
  }

  // Print legend
  os << "\nLegend:\n";
  os << "  I=init_barrier  X=inval_barrier  E=barrier_expect\n";
  os << "  A=arrive        W=wait           L=TMA_load\n";
  os << "  S=TMA_store     C=async_copy\n";
}

void BarrierExecutionOrderAnalysis::printDependencyGraph(
    llvm::raw_ostream &os) const {
  os << "digraph BarrierDependencies {\n";
  os << "  rankdir=TB;\n";
  os << "  node [shape=box];\n\n";

  // Create nodes for each barrier operation
  for (const auto &info : barrierOps) {
    os << "  op" << info.orderIndex << " [label=\""
       << barrierOpKindToString(info.kind) << "\\nbar=" << info.barrierId;
    if (info.asyncTaskId >= 0)
      os << "\\nWG" << info.asyncTaskId;
    os << "\"];\n";
  }

  os << "\n";

  // Create edges for dependencies
  for (const auto &dep : dependencies) {
    if (dep.producer && dep.consumer) {
      os << "  op" << dep.producer->orderIndex << " -> op"
         << dep.consumer->orderIndex << " [label=\""
         << dependencyKindToString(dep.kind) << "\"";
      if (dep.isCrossWarpGroup()) {
        os << ", color=red, penwidth=2";
      }
      os << "];\n";
    }
  }

  os << "}\n";
}

} // namespace mlir::triton
