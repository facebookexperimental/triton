//===-- BarrierAnalysis.cpp - Barrier trace extraction ----------*- C++ -*-===//
//
// Extracts concrete barrier operation traces from TTGIR warp-specialized
// kernels. This is Step 1 of the deadlock detection pipeline (program model
// extraction). See docs/barrier_deadlock_detection_design.md.
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/BarrierAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::triton {

namespace gpu = triton::gpu;
namespace nvidia_gpu = triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// BarrierOpKind string conversion
//===----------------------------------------------------------------------===//

StringRef barrierOpKindToString(BarrierOpKind kind) {
  switch (kind) {
  case BarrierOpKind::ArriveSync:
    return "arrive_barrier";
  case BarrierOpKind::ExpectBytes:
    return "barrier_expect";
  case BarrierOpKind::TmaLoad:
    return "tma_load";
  case BarrierOpKind::Wait:
    return "wait_barrier";
  }
  llvm_unreachable("Unknown BarrierOpKind");
}

//===----------------------------------------------------------------------===//
// BarrierDeadlockAnalysis
//===----------------------------------------------------------------------===//

BarrierDeadlockAnalysis::BarrierDeadlockAnalysis(FunctionOpInterface funcOp,
                                                 int unrollBound)
    : funcOp(funcOp), unrollBound(unrollBound) {}

void BarrierDeadlockAnalysis::run() {
  collectBarrierAllocs();
  buildTaskTraces();
}

//===----------------------------------------------------------------------===//
// Barrier allocation collection
//===----------------------------------------------------------------------===//

void BarrierDeadlockAnalysis::collectBarrierAllocs() {
  funcOp.walk([&](gpu::LocalAllocOp allocOp) {
    auto result = allocOp.getResult();
    auto memDescType = dyn_cast<gpu::MemDescType>(result.getType());
    if (!memDescType)
      return;

    // Barrier allocations have i64 element type
    if (!memDescType.getElementType().isInteger(64))
      return;

    // Try to get a name from the location
    std::string name;
    if (auto nameLoc = dyn_cast<NameLoc>(allocOp.getLoc())) {
      name = nameLoc.getName().str();
    } else if (auto fusedLoc = dyn_cast<FusedLoc>(allocOp.getLoc())) {
      for (auto loc : fusedLoc.getLocations()) {
        if (auto nl = dyn_cast<NameLoc>(loc)) {
          name = nl.getName().str();
          break;
        }
      }
    }
    if (name.empty())
      name = "bar_" + std::to_string(allocOpToName.size());

    allocOpToName[allocOp] = name;

    // Number of slots from the shape
    auto shape = memDescType.getShape();
    int64_t numSlots = shape.empty() ? 1 : shape[0];

    // Find arrive count from init_barrier ops that use indexed slots
    int64_t arriveCount = 1;
    for (auto *user : result.getUsers()) {
      if (auto indexOp = dyn_cast<gpu::MemDescIndexOp>(user)) {
        for (auto *indexUser : indexOp.getResult().getUsers()) {
          if (auto initOp = dyn_cast<nvidia_gpu::InitBarrierOp>(indexUser)) {
            arriveCount = initOp.getCount();
            break;
          }
        }
      }
    }

    barrierAllocs.push_back({name, numSlots, arriveCount});
  });
}

//===----------------------------------------------------------------------===//
// Barrier resolution via def-use chains
//===----------------------------------------------------------------------===//

std::pair<std::string, int64_t>
BarrierDeadlockAnalysis::resolveBarrier(Value barrierVal,
                                        OperandRange captures) {
  // Trace through memdesc_index to find (allocName, slotIndex)
  auto indexOp =
      dyn_cast_or_null<gpu::MemDescIndexOp>(barrierVal.getDefiningOp());
  if (!indexOp)
    return {"", -1};

  Value src = indexOp.getSrc();

  // Try to get static slot index
  int64_t slotIdx = -1;
  if (auto constOp = indexOp.getIndex().getDefiningOp<arith::ConstantIntOp>())
    slotIdx = constOp.value();

  // Trace source through block arguments (warp_specialize captures)
  Value tracedSrc = src;
  while (auto blockArg = dyn_cast<BlockArgument>(tracedSrc)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();

    if (auto partitionsOp =
            dyn_cast<gpu::WarpSpecializePartitionsOp>(parentOp)) {
      unsigned argIndex = blockArg.getArgNumber();
      if (auto wsOp = dyn_cast<gpu::WarpSpecializeOp>(
              partitionsOp->getParentOp())) {
        if (argIndex < wsOp.getExplicitCaptures().size()) {
          tracedSrc = wsOp.getExplicitCaptures()[argIndex];
          continue;
        }
      }
    }
    break;
  }

  // Look up alloc name from the traced source
  std::string allocName;
  if (auto *defOp = tracedSrc.getDefiningOp()) {
    auto it = allocOpToName.find(defOp);
    if (it != allocOpToName.end())
      allocName = it->second;
  }

  return {allocName, slotIdx};
}

//===----------------------------------------------------------------------===//
// Concrete integer expression evaluation
//===----------------------------------------------------------------------===//

std::optional<int64_t>
BarrierDeadlockAnalysis::tryEvalInt(Value val, int64_t loopVar,
                                    Value loopInductionVar) {
  if (val == loopInductionVar)
    return loopVar;

  if (auto constOp = val.getDefiningOp<arith::ConstantIntOp>())
    return constOp.value();

  if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
    return constOp.value();

  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  // Handle unary ops first (1 operand) — must come before the binary check
  if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(defOp))
    return tryEvalInt(defOp->getOperand(0), loopVar, loopInductionVar);

  // Binary ops require 2 operands
  if (defOp->getNumOperands() < 2)
    return std::nullopt;

  auto lhs = tryEvalInt(defOp->getOperand(0), loopVar, loopInductionVar);
  auto rhs = tryEvalInt(defOp->getOperand(1), loopVar, loopInductionVar);
  if (!lhs || !rhs)
    return std::nullopt;

  if (isa<arith::AddIOp>(defOp))
    return *lhs + *rhs;
  if (isa<arith::SubIOp>(defOp))
    return *lhs - *rhs;
  if (isa<arith::MulIOp>(defOp))
    return *lhs * *rhs;
  if (isa<arith::RemSIOp>(defOp))
    return *rhs != 0 ? std::optional(*lhs % *rhs) : std::nullopt;
  if (isa<arith::DivSIOp>(defOp))
    return *rhs != 0 ? std::optional(*lhs / *rhs) : std::nullopt;
  if (isa<arith::XOrIOp>(defOp))
    return *lhs ^ *rhs;
  if (isa<arith::AndIOp>(defOp))
    return *lhs & *rhs;
  if (isa<arith::OrIOp>(defOp))
    return *lhs | *rhs;
  if (isa<arith::MinSIOp>(defOp))
    return std::min(*lhs, *rhs);

  if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
    switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return (*lhs == *rhs) ? 1 : 0;
    case arith::CmpIPredicate::ne:
      return (*lhs != *rhs) ? 1 : 0;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return (*lhs < *rhs) ? 1 : 0;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return (*lhs <= *rhs) ? 1 : 0;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return (*lhs > *rhs) ? 1 : 0;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return (*lhs >= *rhs) ? 1 : 0;
    }
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Loop unrolling and trace construction
//===----------------------------------------------------------------------===//

void BarrierDeadlockAnalysis::unrollLoop(mlir::scf::ForOp forOp,
                                         int64_t taskId,
                                         const std::string &taskName,
                                         TaskTrace &trace,
                                         OperandRange captures) {
  auto lbConst = forOp.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
  auto stepConst = forOp.getStep().getDefiningOp<arith::ConstantIntOp>();

  int64_t lb = lbConst ? lbConst.value() : 0;
  int64_t step = stepConst ? stepConst.value() : 1;
  int64_t ub = unrollBound > 0 ? unrollBound : 5;

  Value inductionVar = forOp.getInductionVar();

  // Resolve slot index dynamically when it depends on the loop variable
  auto resolveDynSlot = [&](Value alloc, int64_t staticSlot,
                            int64_t k) -> int64_t {
    if (staticSlot >= 0)
      return staticSlot;
    if (auto indexOp =
            dyn_cast_or_null<gpu::MemDescIndexOp>(alloc.getDefiningOp())) {
      if (auto evalIdx = tryEvalInt(indexOp.getIndex(), k, inductionVar))
        return *evalIdx;
    }
    return staticSlot;
  };

  // Track iter_args (e.g., phase variable) across iterations
  SmallVector<int64_t> iterArgValues;
  for (auto initVal : forOp.getInitArgs()) {
    if (auto constOp = initVal.getDefiningOp<arith::ConstantIntOp>())
      iterArgValues.push_back(constOp.value());
    else
      iterArgValues.push_back(0);
  }

  auto &bodyBlock = forOp.getRegion().front();
  // Block args: [inductionVar, iterArg0, iterArg1, ...]
  SmallVector<Value> iterArgBlockArgs;
  for (unsigned i = 1; i < bodyBlock.getNumArguments(); ++i)
    iterArgBlockArgs.push_back(bodyBlock.getArgument(i));

  for (int64_t k = lb; k < lb + ub * step; k += step) {
    for (auto &op : bodyBlock) {
      ConcreteBarrierOp concreteOp;
      concreteOp.taskId = taskId;
      concreteOp.taskName = taskName;
      concreteOp.iteration = k;

      bool isBarrierOp = false;

      if (auto waitOp = dyn_cast<nvidia_gpu::WaitBarrierOp>(&op)) {
        concreteOp.kind = BarrierOpKind::Wait;
        auto [name, slot] = resolveBarrier(waitOp.getAlloc(), captures);
        concreteOp.allocName = name;
        concreteOp.slotIndex =
            resolveDynSlot(waitOp.getAlloc(), slot, k);

        // Resolve phase: try direct eval, then iter_args
        auto phaseVal = tryEvalInt(waitOp.getPhase(), k, inductionVar);
        if (!phaseVal) {
          for (unsigned i = 0; i < iterArgBlockArgs.size(); ++i) {
            if (waitOp.getPhase() == iterArgBlockArgs[i]) {
              phaseVal = iterArgValues[i];
              break;
            }
          }
        }
        concreteOp.phase = phaseVal.value_or(-1);
        isBarrierOp = true;

      } else if (auto arriveOp =
                     dyn_cast<nvidia_gpu::ArriveBarrierOp>(&op)) {
        concreteOp.kind = BarrierOpKind::ArriveSync;
        auto [name, slot] = resolveBarrier(arriveOp.getAlloc(), captures);
        concreteOp.allocName = name;
        concreteOp.slotIndex = resolveDynSlot(arriveOp.getAlloc(), slot, k);
        concreteOp.arriveCount = arriveOp.getCount();
        isBarrierOp = true;

      } else if (auto expectOp =
                     dyn_cast<nvidia_gpu::BarrierExpectOp>(&op)) {
        concreteOp.kind = BarrierOpKind::ExpectBytes;
        auto [name, slot] = resolveBarrier(expectOp.getAlloc(), captures);
        concreteOp.allocName = name;
        concreteOp.slotIndex = resolveDynSlot(expectOp.getAlloc(), slot, k);
        concreteOp.expectedBytes = expectOp.getSize();
        // barrier_expect does arrive (PTX mbarrier.arrive.expect_tx)
        concreteOp.arriveCount = 1;
        isBarrierOp = true;

      } else if (auto tmaOp =
                     dyn_cast<nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(&op)) {
        concreteOp.kind = BarrierOpKind::TmaLoad;
        auto [name, slot] = resolveBarrier(tmaOp.getBarrier(), captures);
        concreteOp.allocName = name;
        concreteOp.slotIndex = resolveDynSlot(tmaOp.getBarrier(), slot, k);
        // TMA load does NOT arrive; only contributes bytes
        concreteOp.arriveCount = 0;
        isBarrierOp = true;
      }

      if (isBarrierOp) {
        std::string locStr;
        llvm::raw_string_ostream locOs(locStr);
        op.getLoc()->print(locOs);
        concreteOp.locInfo = locStr;
        concreteOp.position = trace.ops.size();
        trace.ops.push_back(concreteOp);
      }
    }

    // Update iter_arg values for next iteration
    auto *terminator = bodyBlock.getTerminator();
    if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
      for (unsigned i = 0;
           i < yieldOp.getNumOperands() && i < iterArgValues.size(); ++i) {
        auto newVal = tryEvalInt(yieldOp.getOperand(i), k, inductionVar);

        // If direct eval fails, try resolving through iter_args
        if (!newVal) {
          for (unsigned j = 0; j < iterArgBlockArgs.size(); ++j) {
            if (yieldOp.getOperand(i) == iterArgBlockArgs[j]) {
              newVal = iterArgValues[j];
              break;
            }
          }
        }

        // Handle XOR pattern: phase = old_phase ^ (buf == NUM_STAGES-1)
        if (!newVal) {
          if (auto xorOp =
                  yieldOp.getOperand(i).getDefiningOp<arith::XOrIOp>()) {
            auto lhsVal = tryEvalInt(xorOp.getLhs(), k, inductionVar);
            auto rhsVal = tryEvalInt(xorOp.getRhs(), k, inductionVar);
            if (!lhsVal) {
              for (unsigned j = 0; j < iterArgBlockArgs.size(); ++j) {
                if (xorOp.getLhs() == iterArgBlockArgs[j]) {
                  lhsVal = iterArgValues[j];
                  break;
                }
              }
            }
            if (lhsVal && rhsVal)
              newVal = *lhsVal ^ *rhsVal;
          }
        }

        if (newVal)
          iterArgValues[i] = *newVal;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Build task traces from warp_specialize regions
//===----------------------------------------------------------------------===//

void BarrierDeadlockAnalysis::buildTaskTraces() {
  funcOp.walk([&](gpu::WarpSpecializeOp wsOp) {
    auto explicitCaptures = wsOp.getExplicitCaptures();

    // Build capture name map
    for (unsigned i = 0; i < explicitCaptures.size(); ++i) {
      Value cap = explicitCaptures[i];
      if (auto *defOp = cap.getDefiningOp()) {
        auto it = allocOpToName.find(defOp);
        if (it != allocOpToName.end())
          captureIndexToAllocName[i] = it->second;
      }
    }

    // Process default region (typically the producer, task 0)
    {
      TaskTrace trace;
      trace.taskId = 0;
      trace.taskName = "default";

      for (auto &op : wsOp.getDefaultRegion().front()) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op))
          unrollLoop(forOp, 0, "default", trace, explicitCaptures);
      }

      if (!trace.ops.empty())
        taskTraces.push_back(std::move(trace));
    }

    // Process partition regions (typically consumers)
    int partIdx = 0;
    for (Region *region : wsOp.getPartitionRegions()) {
      std::string taskName = "partition" + std::to_string(partIdx);
      int64_t taskId = partIdx + 1;

      TaskTrace trace;
      trace.taskId = taskId;
      trace.taskName = taskName;

      for (auto &op : region->front()) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op))
          unrollLoop(forOp, taskId, taskName, trace, explicitCaptures);
      }

      if (!trace.ops.empty())
        taskTraces.push_back(std::move(trace));
      partIdx++;
    }
  });
}

//===----------------------------------------------------------------------===//
// Summary printing
//===----------------------------------------------------------------------===//

void BarrierDeadlockAnalysis::printSummary(llvm::raw_ostream &os) const {
  os << "Barrier allocations:\n";
  for (const auto &alloc : barrierAllocs) {
    os << "  " << alloc.name << ": " << alloc.numSlots
       << " slots, arrive_count=" << alloc.arriveCount << "\n";
  }

  os << "\nTask traces:\n";
  for (const auto &task : taskTraces) {
    os << "  " << task.taskName << " (task " << task.taskId << "): "
       << task.ops.size() << " barrier ops\n";
    for (const auto &op : task.ops) {
      os << "    [" << op.position << "] " << barrierOpKindToString(op.kind)
         << " " << op.allocName << "[" << op.slotIndex << "]";
      if (op.phase >= 0)
        os << " phase=" << op.phase;
      if (op.arriveCount > 1)
        os << " count=" << op.arriveCount;
      if (op.expectedBytes > 0)
        os << " bytes=" << op.expectedBytes;
      os << " (iter " << op.iteration << ")\n";
    }
  }
}

} // namespace mlir::triton
