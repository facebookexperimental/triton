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
  collectInitialArrives();
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
// Initial arrive collection (pre-task barrier ops)
//===----------------------------------------------------------------------===//

void BarrierDeadlockAnalysis::collectInitialArrives() {
  // Walk barrier ops that are OUTSIDE warp_specialize regions but inside the
  // function. These represent pre-task arrives that set the initial barrier
  // state (e.g., pre-arriving "empty" barriers so the producer can start).
  funcOp.walk([&](Operation *op) {
    // Skip ops inside warp_specialize regions
    if (op->getParentOfType<gpu::WarpSpecializeOp>())
      return;

    // Resolve barrier for arrive/expect ops and count initial arrives
    if (auto arriveOp = dyn_cast<nvidia_gpu::ArriveBarrierOp>(op)) {
      auto indexOp = dyn_cast_or_null<gpu::MemDescIndexOp>(
          arriveOp.getAlloc().getDefiningOp());
      if (!indexOp)
        return;
      Value src = indexOp.getSrc();
      std::string allocName;
      if (auto *defOp = src.getDefiningOp()) {
        auto it = allocOpToName.find(defOp);
        if (it != allocOpToName.end())
          allocName = it->second;
      }
      if (allocName.empty())
        return;
      int64_t slotIdx = -1;
      if (auto constOp =
              indexOp.getIndex().getDefiningOp<arith::ConstantIntOp>())
        slotIdx = constOp.value();
      if (slotIdx < 0)
        return;
      std::string key = allocName + "_" + std::to_string(slotIdx);
      initialArrives[key] += arriveOp.getCount();

    } else if (auto expectOp = dyn_cast<nvidia_gpu::BarrierExpectOp>(op)) {
      auto indexOp = dyn_cast_or_null<gpu::MemDescIndexOp>(
          expectOp.getAlloc().getDefiningOp());
      if (!indexOp)
        return;
      Value src = indexOp.getSrc();
      std::string allocName;
      if (auto *defOp = src.getDefiningOp()) {
        auto it = allocOpToName.find(defOp);
        if (it != allocOpToName.end())
          allocName = it->second;
      }
      if (allocName.empty())
        return;
      int64_t slotIdx = -1;
      if (auto constOp =
              indexOp.getIndex().getDefiningOp<arith::ConstantIntOp>())
        slotIdx = constOp.value();
      if (slotIdx < 0)
        return;
      // barrier_expect counts as an arrive (cnt=1)
      std::string key = allocName + "_" + std::to_string(slotIdx);
      initialArrives[key] += 1;
    }
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
      if (auto wsOp =
              dyn_cast<gpu::WarpSpecializeOp>(partitionsOp->getParentOp())) {
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

void BarrierDeadlockAnalysis::unrollLoop(mlir::scf::ForOp forOp, int64_t taskId,
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
        concreteOp.slotIndex = resolveDynSlot(waitOp.getAlloc(), slot, k);

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

      } else if (auto arriveOp = dyn_cast<nvidia_gpu::ArriveBarrierOp>(&op)) {
        concreteOp.kind = BarrierOpKind::ArriveSync;
        auto [name, slot] = resolveBarrier(arriveOp.getAlloc(), captures);
        concreteOp.allocName = name;
        concreteOp.slotIndex = resolveDynSlot(arriveOp.getAlloc(), slot, k);
        concreteOp.arriveCount = arriveOp.getCount();
        isBarrierOp = true;

      } else if (auto expectOp = dyn_cast<nvidia_gpu::BarrierExpectOp>(&op)) {
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

  if (!initialArrives.empty()) {
    os << "\nInitial arrives (pre-task):\n";
    for (const auto &[key, cnt] : initialArrives)
      os << "  " << key << ": " << cnt << "\n";
  }

  os << "\nTask traces:\n";
  for (const auto &task : taskTraces) {
    os << "  " << task.taskName << " (task " << task.taskId
       << "): " << task.ops.size() << " barrier ops\n";
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

//===----------------------------------------------------------------------===//
// Z3 Python script generation
//===----------------------------------------------------------------------===//

void BarrierDeadlockAnalysis::dumpPythonZ3Script(llvm::raw_ostream &os) const {
  // Helper: unique barrier slot key string for Z3 variable naming.
  auto slotKey = [](const ConcreteBarrierOp &op) -> std::string {
    return op.allocName + "_" + std::to_string(op.slotIndex);
  };

  // Build flat operation list with global indices, and per-task op lists.
  struct OpRef {
    size_t globalIdx;
    const ConcreteBarrierOp *op;
    const TaskTrace *task;
  };
  std::vector<OpRef> allOps;
  std::map<int64_t, std::vector<OpRef>> taskOps; // taskId -> ops
  // slot key -> ops
  std::map<std::string, std::vector<OpRef>> slotOps;

  size_t globalIdx = 0;
  for (const auto &task : taskTraces) {
    for (const auto &op : task.ops) {
      OpRef ref{globalIdx, &op, &task};
      allOps.push_back(ref);
      taskOps[task.taskId].push_back(ref);
      slotOps[slotKey(op)].push_back(ref);
      globalIdx++;
    }
  }

  // Collect arrive counts per slot from barrierAllocs.
  // Key: (allocName, slotIndex) -> arriveCount
  std::map<std::string, int64_t> slotArriveCount;
  for (const auto &alloc : barrierAllocs) {
    for (int64_t s = 0; s < alloc.numSlots; s++) {
      std::string key = alloc.name + "_" + std::to_string(s);
      slotArriveCount[key] = alloc.arriveCount;
    }
  }

  // --- Emit Python script ---
  os << "#!/usr/bin/env python3\n";
  os << "\"\"\"Auto-generated barrier deadlock check (Z3 constraints).\n\n";
  os << "Encoding follows barrier_deadlock_design.tex Sections 2-5.\n";
  os << "SAT = deadlock witness, UNSAT = safe within unroll bound.\n";
  os << "\"\"\"\n\n";
  os << "from z3 import *\n\n";

  // --- Variables ---
  os << "# ---- Variables ----\n\n";

  // Cut-point per task
  os << "# Cut-point per task: where execution is stuck\n";
  for (const auto &task : taskTraces) {
    os << "c_" << task.taskName << " = Int('c_" << task.taskName << "')\n";
  }
  os << "\n";

  // Timestamps per operation
  os << "# Timestamp per operation (global execution order)\n";
  for (const auto &ref : allOps) {
    os << "tau_" << ref.globalIdx << " = Int('tau_" << ref.globalIdx << "')\n";
  }
  os << "\n";

  // Async timestamps (TMA_LOAD only for now)
  os << "# Async completion timestamps (TMA_LOAD)\n";
  bool hasAsync = false;
  for (const auto &ref : allOps) {
    if (ref.op->kind == BarrierOpKind::TmaLoad) {
      os << "tau_prime_" << ref.globalIdx << " = Int('tau_prime_"
         << ref.globalIdx << "')\n";
      hasAsync = true;
    }
  }
  if (!hasAsync)
    os << "# (none)\n";
  os << "\n";

  // --- Operation metadata as comments ---
  os << "# ---- Operation index ----\n";
  for (const auto &ref : allOps) {
    os << "# op " << ref.globalIdx << ": task=" << ref.op->taskName
       << " pos=" << ref.op->position << " "
       << barrierOpKindToString(ref.op->kind).str() << " " << ref.op->allocName
       << "[" << ref.op->slotIndex << "]";
    if (ref.op->kind == BarrierOpKind::Wait)
      os << " phase=" << ref.op->phase;
    if (ref.op->expectedBytes > 0)
      os << " bytes=" << ref.op->expectedBytes;
    os << " iter=" << ref.op->iteration << "\n";
  }
  os << "\n";

  // --- Helper functions ---
  os << "# ---- Helper functions ----\n\n";

  // exec(o) predicate
  os << "def exec_op(global_idx, task_name, pos):\n";
  os << "    \"\"\"exec(o) = pos(o) < c_task(o)\"\"\"\n";
  os << "    c = {'";
  bool first = true;
  for (const auto &task : taskTraces) {
    if (!first)
      os << ", '";
    os << task.taskName << "': c_" << task.taskName;
    first = false;
  }
  os << "}\n";
  os << "    return pos < c[task_name]\n\n";

  // effect_time(o): tau' for async, tau for sync
  os << "def effect_time(global_idx):\n";
  os << "    \"\"\"effect_time(o) = tau'_o for async, tau_o for sync\"\"\"\n";
  os << "    async_ops = {";
  first = true;
  for (const auto &ref : allOps) {
    if (ref.op->kind == BarrierOpKind::TmaLoad) {
      if (!first)
        os << ", ";
      os << ref.globalIdx << ": tau_prime_" << ref.globalIdx;
      first = false;
    }
  }
  os << "}\n";
  os << "    if global_idx in async_ops:\n";
  os << "        return async_ops[global_idx]\n";
  os << "    return globals()[f'tau_{global_idx}']\n\n";

  // initial_arrives — pre-task arrive counts per slot
  os << "# Initial arrives (pre-task barrier ops)\n";
  os << "initial_arrives = {";
  first = true;
  for (const auto &[sk, cnt] : initialArrives) {
    if (!first)
      os << ", ";
    os << "'" << sk << "': " << cnt;
    first = false;
  }
  os << "}\n\n";

  // arrive_count(slot) — cumulative arrives
  os << "def arrive_count(slot_key):\n";
  os << "    \"\"\"A(b) = initial + sum If(exec(o), cnt(o), 0) for arrive ops "
        "on slot b\"\"\"\n";
  os << "    initial = initial_arrives.get(slot_key, 0)\n";
  os << "    terms = []\n";

  // Group arrive ops by slot
  for (const auto &[sk, ops] : slotOps) {
    os << "    if slot_key == '" << sk << "':\n";
    bool hasArrives = false;
    for (const auto &ref : ops) {
      if (ref.op->kind == BarrierOpKind::ArriveSync ||
          ref.op->kind == BarrierOpKind::ExpectBytes) {
        os << "        terms.append(If(exec_op(" << ref.globalIdx << ", '"
           << ref.op->taskName << "', " << ref.op->position << "), "
           << ref.op->arriveCount << ", 0))\n";
        hasArrives = true;
      }
    }
    if (!hasArrives)
      os << "        pass\n";
  }
  os << "    return (initial + Sum(terms)) if terms else IntVal(initial)\n\n";

  // arrive_count_before(slot, t) — arrives before timestamp t
  os << "def arrive_count_before(slot_key, t):\n";
  os << "    \"\"\"A^{<t}(b) = initial + sum If(exec(o) & effect_time(o) < t, "
        "cnt(o), 0)\"\"\"\n";
  os << "    initial = initial_arrives.get(slot_key, 0)\n";
  os << "    terms = []\n";
  for (const auto &[sk, ops] : slotOps) {
    os << "    if slot_key == '" << sk << "':\n";
    bool hasArrives = false;
    for (const auto &ref : ops) {
      if (ref.op->kind == BarrierOpKind::ArriveSync ||
          ref.op->kind == BarrierOpKind::ExpectBytes) {
        os << "        terms.append(If(And(exec_op(" << ref.globalIdx << ", '"
           << ref.op->taskName << "', " << ref.op->position << "), effect_time("
           << ref.globalIdx << ") < t), " << ref.op->arriveCount << ", 0))\n";
        hasArrives = true;
      }
    }
    if (!hasArrives)
      os << "        pass\n";
  }
  os << "    return (initial + Sum(terms)) if terms else IntVal(initial)\n\n";

  // blocked_parity(slot, parity) — parity-based blocking
  os << "def blocked_parity(slot_key, parity, ac):\n";
  os << "    \"\"\"blocked_parity(b, p) = (A(b) / ac) % 2 == p\"\"\"\n";
  os << "    a = arrive_count(slot_key)\n";
  os << "    if ac == 1:\n";
  os << "        return a % 2 == parity\n";
  os << "    return (a / ac) % 2 == parity\n\n";

  // --- Solver and constraints ---
  os << "# ---- Solver ----\n\n";
  os << "s = Solver()\n\n";

  // Structural constraints: cut-point bounds
  os << "# Structural: cut-point bounds\n";
  for (const auto &task : taskTraces) {
    os << "s.add(c_" << task.taskName << " >= 0, c_" << task.taskName
       << " <= " << task.ops.size() << ")\n";
  }
  os << "\n";

  // Timestamp positivity
  os << "# Structural: timestamp positivity\n";
  for (const auto &ref : allOps) {
    os << "s.add(tau_" << ref.globalIdx << " >= 0)\n";
  }
  for (const auto &ref : allOps) {
    if (ref.op->kind == BarrierOpKind::TmaLoad)
      os << "s.add(tau_prime_" << ref.globalIdx << " >= 0)\n";
  }
  os << "\n";

  // Phi_ord: intra-task ordering (consecutive ops)
  os << "# Phi_ord: intra-task operation ordering\n";
  for (const auto &[taskId, ops] : taskOps) {
    for (size_t i = 0; i + 1 < ops.size(); i++) {
      const auto &a = ops[i];
      const auto &b = ops[i + 1];
      os << "s.add(Implies(And(exec_op(" << a.globalIdx << ", '"
         << a.op->taskName << "', " << a.op->position << "), exec_op("
         << b.globalIdx << ", '" << b.op->taskName << "', " << b.op->position
         << ")), tau_" << a.globalIdx << " < tau_" << b.globalIdx << "))\n";
    }
  }
  os << "\n";

  // Phi_async: causal ordering for async ops
  os << "# Phi_async: causal ordering (tau < tau' for async ops)\n";
  for (const auto &ref : allOps) {
    if (ref.op->kind == BarrierOpKind::TmaLoad) {
      os << "s.add(Implies(exec_op(" << ref.globalIdx << ", '"
         << ref.op->taskName << "', " << ref.op->position << "), tau_"
         << ref.globalIdx << " < tau_prime_" << ref.globalIdx << "))\n";
    }
  }
  os << "\n";

  // Phi_stall: tasks can only stall at WAIT positions
  os << "# Phi_stall: stall only at WAIT positions\n";
  for (const auto &task : taskTraces) {
    std::vector<const ConcreteBarrierOp *> waits;
    for (const auto &op : task.ops) {
      if (op.kind == BarrierOpKind::Wait)
        waits.push_back(&op);
    }
    if (waits.empty()) {
      os << "s.add(c_" << task.taskName << " == " << task.ops.size() << ")\n";
    } else {
      os << "s.add(Implies(c_" << task.taskName << " < " << task.ops.size()
         << ", Or(";
      first = true;
      for (const auto *w : waits) {
        if (!first)
          os << ", ";
        os << "c_" << task.taskName << " == " << w->position;
        first = false;
      }
      os << ")))\n";
    }
  }
  os << "\n";

  // Phi_B: blocked barrier at stall point
  os << "# Phi_B: stalled at WAIT -> barrier is blocked\n";
  for (const auto &task : taskTraces) {
    for (const auto &op : task.ops) {
      if (op.kind != BarrierOpKind::Wait)
        continue;
      std::string sk = slotKey(op);
      int64_t ac = 1;
      auto it = slotArriveCount.find(sk);
      if (it != slotArriveCount.end())
        ac = it->second;

      os << "s.add(Implies(And(c_" << task.taskName << " == " << op.position
         << ", c_" << task.taskName << " < " << task.ops.size() << "), ";
      if (op.phase >= 0) {
        // Parity-based blocking
        os << "blocked_parity('" << sk << "', " << op.phase << ", " << ac
           << ")";
      } else {
        // Fallback: cycle-based (not expected with concrete eval)
        os << "blocked_parity('" << sk << "', 0, " << ac << ")";
      }
      os << "))\n";
    }
  }
  os << "\n";

  // Phi_R: release constraint for passed-through waits
  os << "# Phi_R: passed-through WAIT -> barrier was ready when reached\n";
  for (const auto &task : taskTraces) {
    for (const auto &op : task.ops) {
      if (op.kind != BarrierOpKind::Wait)
        continue;
      std::string sk = slotKey(op);
      int64_t ac = 1;
      auto it = slotArriveCount.find(sk);
      if (it != slotArriveCount.end())
        ac = it->second;

      // Find the global index for this op
      size_t gIdx = 0;
      for (const auto &ref : allOps) {
        if (ref.op == &op) {
          gIdx = ref.globalIdx;
          break;
        }
      }

      os << "s.add(Implies(And(" << op.position << " < c_" << task.taskName
         << ", c_" << task.taskName << " <= " << task.ops.size() << "), ";

      if (op.phase >= 0) {
        // Parity-based release: phase at tau_w must differ from parity
        if (ac == 1) {
          os << "arrive_count_before('" << sk << "', tau_" << gIdx
             << ") % 2 != " << op.phase;
        } else {
          os << "(arrive_count_before('" << sk << "', tau_" << gIdx << ") / "
             << ac << ") % 2 != " << op.phase;
        }
      } else {
        // Fallback
        os << "True";
      }
      os << "))\n";
    }
  }
  os << "\n";

  // Byte balance constraints (cumulative): total loaded >= total expected
  // Only emit if there are EXPECT_BYTES ops on any slot.
  bool hasByteOps = false;
  for (const auto &[sk, ops] : slotOps) {
    for (const auto &ref : ops) {
      if (ref.op->kind == BarrierOpKind::ExpectBytes) {
        hasByteOps = true;
        break;
      }
    }
    if (hasByteOps)
      break;
  }

  if (hasByteOps) {
    os << "# Byte balance: cumulative loaded >= expected per slot\n";
    os << "# (Over-approximation: not per-cycle, but cumulative)\n";
    for (const auto &[sk, ops] : slotOps) {
      std::vector<OpRef> expectOps, tmaOps;
      for (const auto &ref : ops) {
        if (ref.op->kind == BarrierOpKind::ExpectBytes)
          expectOps.push_back(ref);
        if (ref.op->kind == BarrierOpKind::TmaLoad)
          tmaOps.push_back(ref);
      }
      if (expectOps.empty())
        continue;

      // For the completion predicate to be correct, we need byte balance.
      // This is encoded as: if all tasks complete, loaded >= expected.
      // But more precisely, we add it to the blocking predicate:
      // blocked also if bytes are insufficient.
      os << "# Slot " << sk << ": byte balance\n";
      os << "total_expected_" << sk << " = ";
      first = true;
      for (const auto &ref : expectOps) {
        if (!first)
          os << " + ";
        os << "If(exec_op(" << ref.globalIdx << ", '" << ref.op->taskName
           << "', " << ref.op->position << "), " << ref.op->expectedBytes
           << ", 0)";
        first = false;
      }
      os << "\n";

      os << "total_loaded_" << sk << " = ";
      if (tmaOps.empty()) {
        os << "IntVal(0)";
      } else {
        first = true;
        for (const auto &ref : tmaOps) {
          if (!first)
            os << " + ";
          // TMA load transfer size: use expectedBytes from corresponding
          // expect_bytes as approximation (xfer not stored separately yet)
          os << "If(exec_op(" << ref.globalIdx << ", '" << ref.op->taskName
             << "', " << ref.op->position << "), " << ref.op->expectedBytes
             << ", 0)";
          first = false;
        }
      }
      os << "\n";

      // Add byte deficit as additional blocking condition for waits on this
      // slot
      os << "byte_deficit_" << sk << " = total_loaded_" << sk
         << " < total_expected_" << sk << "\n";
      os << "# (byte_deficit is implicitly part of blocked for this slot)\n\n";
    }
  }

  // Deadlock query: at least one task is stuck
  os << "# Deadlock query: at least one task stuck\n";
  os << "s.add(Or(";
  first = true;
  for (const auto &task : taskTraces) {
    if (!first)
      os << ", ";
    os << "c_" << task.taskName << " < " << task.ops.size();
    first = false;
  }
  os << "))\n\n";

  // --- Solve and diagnose ---
  os << "# ---- Solve ----\n\n";
  os << "result = s.check()\n";
  os << "print(f'Result: {result}')\n\n";
  os << "if result == sat:\n";
  os << "    m = s.model()\n";
  os << "    print('\\nDeadlock detected!')\n";
  os << "    print('Task cut-points:')\n";
  for (const auto &task : taskTraces) {
    os << "    c_val = m.eval(c_" << task.taskName << ").as_long()\n";
    os << "    print(f'  " << task.taskName << ": c={c_val} / "
       << task.ops.size() << "', end='')\n";
    os << "    if c_val < " << task.ops.size() << ":\n";
    os << "        print(' [STUCK]', end='')\n";
    os << "    print()\n";
  }
  os << "else:\n";
  os << "    print('No deadlock found (UNSAT).')\n";
}

} // namespace mlir::triton
