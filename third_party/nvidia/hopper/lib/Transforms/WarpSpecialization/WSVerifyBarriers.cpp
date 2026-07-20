//===- WSVerifyBarriers.cpp -----------------------------------------------===//
//
// Static verifier for autoWS mbarrier deadlocks and cross-partition races.
// Consumes post-code-partition / post-pipeline-expander IR. Verify-only: the
// module is never mutated.
//
// Design: lib/Transforms/WarpSpecialization/docs/WSAliasingCoverage.proposal.md
//
// Goal 1 (this file, incremental):
//   - run after the pipeline expander, autoWS only;
//   - barrier -> reuse group (buffer.id) is assumed given, carried as a
//     `buffer.id` integer attribute on each `ttng.init_barrier`.
//
// Phase 0: parse the given barrier -> buffer.id mapping, group the WS-generated
// barriers by reuse group, and report it as remarks.
//
// Phase 1 (current): build the Barrier Dependency Graph (BDG) -- resolve every
// barrier-touching op to its outer barrier alloc (through memdesc views and
// warp_specialize partition captures), and classify it as a wait or an arrive
// node grouped by that alloc. First deadlock check: condition (i) from the
// design doc -- a FULL (forward) wait whose barrier alloc has no producing
// arrive anywhere is a genuine deadlock. The parity-cadence and cross-partition
// SCC conditions, and the per-reuse-group Access-Order Graph (race /
// under-buffering), come in later phases.
//
//===----------------------------------------------------------------------===//

#include "CodePartitionUtility.h"
#include "WSBarrierAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvgpu-verify-ws-barriers"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {

#define GEN_PASS_DEF_NVGPUVERIFYWSBARRIERS
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

namespace {

// A barrier whose guarded reuse group is known (Goal 1: given via buffer.id on
// the init_barrier). More fields (slot/phase, direction, dstTask, role) are
// added in the BDG-building phase.
struct KnownBarrier {
  ttng::InitBarrierOp init;
  int bufferId;
};

// Read the goal-1 `buffer.id` annotation off an init_barrier. Returns nullopt
// when absent (verifier then treats the barrier as buffer=unknown ->
// indeterminate, per the design doc).
static std::optional<int> getGuardedBufferId(ttng::InitBarrierOp op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("buffer.id"))
    return static_cast<int>(attr.getInt());
  return std::nullopt;
}

// Resolve a barrier operand to the outer WS-generated barrier `local_alloc`
// that backs it, walking through memdesc views (memdesc_index / subview /
// reinterpret / trans) and warp_specialize partition captures (a partition
// block-arg maps 1:1 to WarpSpecializePartitionsOp::explicitCaptures). Returns
// null if the value is not backed by a WS-generated barrier alloc.
static Operation *resolveBarrierAlloc(Value v) {
  DenseSet<Value> seen;
  while (v && seen.insert(v).second) {
    if (auto alloc = v.getDefiningOp<triton::gpu::LocalAllocOp>()) {
      if (alloc->hasAttr(kWarpSpecializeGeneratedBarrierAttrName))
        return alloc;
      return nullptr;
    }
    if (auto *def = v.getDefiningOp()) {
      // Follow single-source memdesc view ops.
      if (isa<triton::gpu::MemDescIndexOp, triton::gpu::MemDescReinterpretOp,
              triton::gpu::MemDescTransOp>(def)) {
        v = def->getOperand(0);
        continue;
      }
      return nullptr;
    }
    // Block argument: map partition captures back to the outer capture operand.
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      Operation *parent = ba.getOwner()->getParentOp();
      if (auto parts =
              dyn_cast_or_null<triton::gpu::WarpSpecializePartitionsOp>(
                  parent)) {
        v = parts.getExplicitCaptures()[ba.getArgNumber()];
        continue;
      }
    }
    return nullptr;
  }
  return nullptr;
}

// A wait-side barrier endpoint (consumes a phase, does not produce one).
static bool isWaitOp(Operation *op) {
  return isa<ttng::WaitBarrierOp, triton::nvws::ConsumerWaitOp,
             triton::nvws::ProducerAcquireOp>(op);
}

// Ops that only reference a barrier without being a wait or an arrive.
static bool isBarrierStructuralOp(Operation *op) {
  return isa<ttng::InitBarrierOp, ttng::InvalBarrierOp,
             triton::gpu::MemDescIndexOp, triton::gpu::MemDescReinterpretOp,
             triton::gpu::MemDescTransOp, triton::gpu::LocalAllocOp>(op);
}

// A BDG node: a barrier-touching op resolved to its outer barrier alloc.
struct BDGNode {
  Operation *op;
  Operation *barrierAlloc;
  bool isWait;
  // FULL (forward/data-ready) vs backward/EMPTY reuse. Forward waits are the
  // ones condition (i) applies to. Derived from the WSBarrier direction.
  bool isForward;
};

} // namespace

class NVGPUVerifyWSBarriersPass
    : public impl::NVGPUVerifyWSBarriersBase<NVGPUVerifyWSBarriersPass> {
public:
  using impl::NVGPUVerifyWSBarriersBase<
      NVGPUVerifyWSBarriersPass>::NVGPUVerifyWSBarriersBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    // Phase 0: collect the given barrier -> buffer.id mapping and report it,
    // grouped by reuse group. This establishes the input contract the later
    // BDG / AOG phases consume.
    DenseMap<int, SmallVector<KnownBarrier>> byGroup;
    unsigned numUnknown = 0;
    funcOp.walk(
        [&](ttng::InitBarrierOp init) {
          if (auto id = getGuardedBufferId(init)) {
            byGroup[*id].push_back({init, *id});
          } else {
            ++numUnknown;
            init.emitRemark()
                << "WS barrier has no buffer.id (reuse group unknown) -> "
                   "indeterminate";
          }
        });

    for (auto &kv : byGroup) {
      LDBG("reuse group buffer.id=" << kv.first << " has " << kv.second.size()
                                    << " WS barriers");
      // The per-barrier reuse-group report is informational coverage; gate it
      // behind emit-coverage-table so -verify-diagnostics deadlock/race tests
      // are not perturbed by it.
      if (emitCoverageTable)
        for (KnownBarrier &kb : kv.second)
          kb.init.emitRemark()
              << "WS barrier guards reuse group buffer.id=" << kb.bufferId;
    }
    LDBG("verified func " << funcOp.getName() << ": " << byGroup.size()
                          << " reuse groups, " << numUnknown
                          << " unknown-buffer barriers");

    if (checkKind == "deadlock" || checkKind == "all")
      runDeadlockChecks(funcOp);
  }

  // Phase 1: build the BDG and run the no-producer deadlock check (condition
  // (i) of the design doc). Later phases add parity-cadence and the
  // cross-partition SCC on top of this same node set.
  void runDeadlockChecks(triton::FuncOp funcOp) {
    SmallVector<BDGNode> nodes;
    // Barrier alloc -> whether any arrive node drives it.
    DenseMap<Operation *, bool> hasArrive;

    funcOp.walk([&](Operation *op) {
      if (isBarrierStructuralOp(op))
        return;
      bool wait = isWaitOp(op);
      // A single op (e.g. tc_gen5_mma) can drive multiple barrier operands.
      DenseSet<Operation *> allocsForThisOp;
      for (Value operand : op->getOperands()) {
        Operation *alloc = resolveBarrierAlloc(operand);
        if (!alloc || !allocsForThisOp.insert(alloc).second)
          continue;
        std::optional<bool> bwd = isWSBarrierBackwardEndpoint(op);
        bool isForward = bwd.has_value() && !*bwd;
        nodes.push_back({op, alloc, wait, isForward});
        if (!wait)
          hasArrive[alloc] = true;
        else
          hasArrive.try_emplace(alloc, false);
      }
    });

    // Condition (i): a FULL (forward) wait whose barrier alloc has no producing
    // arrive anywhere is a genuine deadlock -- the phase it polls is never
    // produced. Backward first-acquires are intentionally NOT flagged here
    // (they may be satisfied by an init-time pre-arm rather than an arrive op);
    // those are handled by the pre-arm-aware SCC phase.
    unsigned numDeadlocks = 0;
    for (const BDGNode &n : nodes) {
      if (!n.isWait || !n.isForward)
        continue;
      auto it = hasArrive.find(n.barrierAlloc);
      if (it != hasArrive.end() && !it->second) {
        ++numDeadlocks;
        n.op->emitError()
            << "WS deadlock: FULL wait_barrier has no producing arrive on its "
               "barrier alloc (the polled phase is never produced)";
      }
    }
    LDBG("deadlock check: " << nodes.size() << " BDG nodes, " << numDeadlocks
                            << " no-producer deadlocks");
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
