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
// Phase 1: build the Barrier Dependency Graph (BDG) -- resolve every
// barrier-touching op to its outer barrier alloc (through memdesc views and
// warp_specialize partition captures), and classify it as a wait or an arrive
// node grouped by that alloc. Deadlock condition (i): a FULL (forward) wait
// whose barrier alloc has no producing arrive anywhere is a genuine deadlock
// (sound, default error).
//
// Phase 2 (current): cross-partition SCC over satisfy + program-order edges
// (design-doc condition iii), with a first-execution filter = phase-parity
// (constant / inverted-acquire / accumCnt-derived) + SSA-dominance pre-arm.
// IMPORTANT empirical finding: under this (correct) filter the first-execution
// barrier graph is a DAG for BOTH the correct HSTU DP=1 fwd AND the deadlocking
// HSTU DP=2 fwd -- i.e. no first-execution cycle in either. So the filter has
// no false positive (DP=1 clean) but also cannot catch DP=2, because DP=2's
// deadlock is a STEADY-STATE / cross-iteration cadence deadlock, not a
// first-execution cycle. Catching that needs per-slot phase-flip modeling
// across iterations (the next work item). The SCC therefore ships gated under
// report-cycles, emitting remarks (never errors); it fires on a genuine
// first-execution cycle (e.g. the classic idle-acc-init). Parity-cadence
// (condition ii) and the per-reuse-group Access-Order Graph (race /
// under-buffering) come in later phases.
//
//===----------------------------------------------------------------------===//

#include "CodePartitionUtility.h"
#include "WSBarrierAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OperationSupport.h"
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

// Resolve a memdesc value to the buffer.id of the DATA alloc backing it
// (ttg.local_alloc / ttng.tmem_alloc), walking memdesc views and
// warp_specialize partition captures. Returns nullopt if not backed by a
// buffer.id'd alloc.
static std::optional<int> resolveDataBufferId(Value v) {
  DenseSet<Value> seen;
  while (v && seen.insert(v).second) {
    if (Operation *def = v.getDefiningOp()) {
      if (isa<triton::gpu::LocalAllocOp, ttng::TMEMAllocOp>(def)) {
        if (auto id = def->getAttrOfType<IntegerAttr>("buffer.id"))
          return static_cast<int>(id.getInt());
        return std::nullopt;
      }
      if (isa<triton::gpu::MemDescIndexOp, triton::gpu::MemDescReinterpretOp,
              triton::gpu::MemDescTransOp>(def)) {
        v = def->getOperand(0);
        continue;
      }
      return std::nullopt;
    }
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      if (auto parts =
              dyn_cast_or_null<triton::gpu::WarpSpecializePartitionsOp>(
                  ba.getOwner()->getParentOp())) {
        v = parts.getExplicitCaptures()[ba.getArgNumber()];
        continue;
      }
    }
    return std::nullopt;
  }
  return std::nullopt;
}

// True if `v` is a loop iter-arg carrying the WS accumulation counter (tagged
// NameLoc("accum_cnt") by code partitioning).
static bool isAccumCnt(Value v) {
  auto ba = dyn_cast<BlockArgument>(v);
  if (!ba)
    return false;
  if (auto nl = dyn_cast<NameLoc>(ba.getLoc()))
    return nl.getName().getValue() == "accum_cnt";
  return false;
}

// Print a compact symbolic form of a barrier index/phase operand in terms of
// the loop accumCnt (e.g. "accumCnt % 1", "(accumCnt / 1) & 1", "NOT(accumCnt &
// 1)", or a constant). This is how the mbarrier slot/phase relates to the loop
// counter -- exactly the phase-model view the deadlock reasoning needs.
static std::string describeAccumExpr(Value v, int depth = 0) {
  if (!v)
    return "?";
  if (isAccumCnt(v))
    return "accumCnt";
  if (auto ba = dyn_cast<BlockArgument>(v))
    return "%arg" + std::to_string(ba.getArgNumber());
  Operation *def = v.getDefiningOp();
  if (!def)
    return "?";
  if (auto c = dyn_cast<arith::ConstantOp>(def)) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return std::to_string(ia.getInt());
    return "const";
  }
  if (depth > 8)
    return "...";
  auto A = [&](unsigned i) {
    return describeAccumExpr(def->getOperand(i), depth + 1);
  };
  if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, arith::IndexCastOp>(
          def))
    return A(0);
  if (isa<arith::RemUIOp>(def))
    return "(" + A(0) + " % " + A(1) + ")";
  if (isa<arith::DivUIOp>(def))
    return "(" + A(0) + " / " + A(1) + ")";
  if (isa<arith::AndIOp>(def))
    return "(" + A(0) + " & " + A(1) + ")";
  if (isa<arith::AddIOp>(def))
    return "(" + A(0) + " + " + A(1) + ")";
  if (isa<arith::SubIOp>(def))
    return "(" + A(0) + " - " + A(1) + ")";
  if (isa<arith::MulIOp>(def))
    return "(" + A(0) + " * " + A(1) + ")";
  if (auto x = dyn_cast<arith::XOrIOp>(def)) {
    for (unsigned i = 0; i < 2; ++i)
      if (auto cc = x.getOperand(i).getDefiningOp<arith::ConstantOp>())
        if (auto ia = dyn_cast<IntegerAttr>(cc.getValue()))
          if (ia.getInt() & 1)
            return "NOT(" + A(1 - i) + ")";
    return A(0) + " ^ " + A(1);
  }
  return def->getName().getStringRef().str();
}

// The slot-index operand of a barrier op's group-barrier operand: barriers are
// referenced as `memdesc_index %alloc[%idx]`; return %idx (or null).
static Value barrierSlotIndex(Operation *op,
                              const DenseSet<Operation *> &group) {
  for (Value o : op->getOperands()) {
    Operation *ba = resolveBarrierAlloc(o);
    if (ba && group.count(ba)) {
      if (auto idx = o.getDefiningOp<triton::gpu::MemDescIndexOp>())
        return idx->getOperand(1);
      return Value();
    }
  }
  return Value();
}

// A wait-side barrier endpoint (consumes a phase, does not produce one).
static bool isWaitOp(Operation *op) {
  return isa<ttng::WaitBarrierOp, triton::nvws::ConsumerWaitOp,
             triton::nvws::ProducerAcquireOp>(op);
}

// An arrive-side barrier endpoint (produces a phase). Allowlist: ops that
// merely FORWARD a barrier as a capture/iter-arg (warp_specialize, scf loops)
// or index it are NOT arrives and must be excluded, else they register phantom
// producers.
static bool isArriveOp(Operation *op) {
  return isa<ttng::ArriveBarrierOp, ttng::TCGen5CommitOp,
             ttng::AsyncTMACopyGlobalToLocalOp, ttng::AsyncCopyMbarrierArriveOp,
             ttng::BarrierExpectOp, ttng::MMAv5OpInterface,
             triton::nvws::ProducerCommitOp, triton::nvws::ConsumerReleaseOp>(
      op);
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
  // Partition (async_task_id; -1 if not exactly one) and a program-order index
  // (pre-order walk position). Used by the cross-partition SCC (condition iii).
  int partition;
  unsigned walkIdx;
  // Recovered slot index and (waits only) phase, as SSA values -- their form in
  // terms of the loop accumCnt is rendered by describeAccumExpr(). slotIdx is
  // the `memdesc_index %barAlloc[%idx]` index; phase is the wait's phase
  // operand. firstExecParity is the phase parity polled on the first execution
  // (0/1) or -1 if not statically determinable. These are the single source of
  // truth for both the dump and the phase-model checks.
  Value slotIdx;
  Value phase;
  int firstExecParity;
};

// Single async_task_id of an op, or -1 if absent/ambiguous.
static int getSingleTask(Operation *op) {
  auto ids = getAsyncTaskIds(op);
  return ids.size() == 1 ? ids[0] : -1;
}

} // namespace

class NVGPUVerifyWSBarriersPass
    : public impl::NVGPUVerifyWSBarriersBase<NVGPUVerifyWSBarriersPass> {
public:
  using impl::NVGPUVerifyWSBarriersBase<
      NVGPUVerifyWSBarriersPass>::NVGPUVerifyWSBarriersBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (dumpBufferId >= 0)
      dumpBufferGroup(funcOp, dumpBufferId);

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

  // Debug helper (--dump-buffer-id=N): print every op related to reuse group N
  // -- the data alloc(s) with buffer.id==N, their writers/readers, the barriers
  // guarding N (init_barrier buffer.id==N), and the wait/arrive ops driving
  // those barriers -- each tagged with partition (async_task_id) and role, in
  // program (walk) order. Intended for debugging a specific reuse group, e.g.
  // the O accumulator (id 6/7) the DP=2 deadlock is on.
  void dumpBufferGroup(triton::FuncOp funcOp, int bid) {
    DenseSet<Operation *> groupBarriers;
    funcOp.walk([&](ttng::InitBarrierOp init) {
      if (auto id = getGuardedBufferId(init))
        if (*id == bid)
          if (Operation *b = resolveBarrierAlloc(init.getAlloc()))
            groupBarriers.insert(b);
    });

    auto flags = OpPrintingFlags().skipRegions();
    llvm::errs() << "==== ops for reuse group buffer.id=" << bid << " in @"
                 << funcOp.getName() << " (" << groupBarriers.size()
                 << " guarding barriers) ====\n";

    // Loop accumCnt(s): the WS accumulation counters that drive every barrier
    // slot/phase. Show each loop's accumCnt iter-arg, its entry value, and its
    // per-iteration update.
    funcOp.walk([&](scf::ForOp forOp) {
      for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        if (!isAccumCnt(arg))
          continue;
        Value init = forOp.getInitArgs()[i];
        Value next = forOp.getBody()->getTerminator()->getOperand(i);
        llvm::errs() << "  [accumCnt] loop@"
                     << "for  init=" << describeAccumExpr(init)
                     << "  next=" << describeAccumExpr(next)
                     << "  (bufferIdx = accumCnt % numBuffers, phase = "
                        "(accumCnt / numBuffers) & 1)\n";
      }
    });
    funcOp.walk([&](Operation *op) {
      std::string role;
      if (isa<triton::gpu::LocalAllocOp, ttng::TMEMAllocOp>(op))
        if (auto id = op->getAttrOfType<IntegerAttr>("buffer.id"))
          if (id.getInt() == bid)
            role = "alloc";
      if (role.empty())
        if (auto init = dyn_cast<ttng::InitBarrierOp>(op)) {
          Operation *b = resolveBarrierAlloc(init.getAlloc());
          if (b && groupBarriers.count(b))
            role = "init-barrier";
        }
      if (role.empty() && (isWaitOp(op) || isArriveOp(op)))
        for (Value o : op->getOperands()) {
          Operation *ba = resolveBarrierAlloc(o);
          if (ba && groupBarriers.count(ba)) {
            if (isWaitOp(op)) {
              auto bwd = isWSBarrierBackwardEndpoint(op);
              role = bwd.value_or(false) ? "wait(bwd)" : "wait(fwd)";
            } else {
              role = "arrive";
            }
            break;
          }
        }
      if (role.empty() && !isBarrierStructuralOp(op) &&
          !isa<triton::gpu::WarpSpecializeOp,
               triton::gpu::WarpSpecializePartitionsOp>(op))
        for (Value o : op->getOperands()) {
          auto id = resolveDataBufferId(o);
          if (id && *id == bid) {
            if (isa<ttng::TMEMStoreOp, triton::gpu::LocalStoreOp>(op))
              role = "write";
            else if (isa<ttng::TMEMLoadOp, triton::gpu::LocalLoadOp>(op))
              role = "read";
            else if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
              role = resolveDataBufferId(mma.getAccumulator()) ==
                             std::optional<int>(bid)
                         ? "mma-write(acc)"
                         : "mma-read";
            else
              role = "data-use";
            break;
          }
        }
      if (role.empty())
        return;
      llvm::errs() << "  [" << role << "] task=" << getSingleTask(op);
      // For barrier endpoints, show how the slot index and (for waits) the
      // phase relate to the loop accumCnt.
      if (isWaitOp(op) || isArriveOp(op)) {
        if (Value slot = barrierSlotIndex(op, groupBarriers))
          llvm::errs() << "  idx=" << describeAccumExpr(slot);
        if (isa<ttng::WaitBarrierOp>(op) && op->getNumOperands() > 1)
          llvm::errs() << "  phase=" << describeAccumExpr(op->getOperand(1));
      }
      llvm::errs() << "  ";
      op->print(llvm::errs(), flags);
      llvm::errs() << "\n";
    });
    llvm::errs() << "==== end buffer.id=" << bid << " ====\n";
  }

  // Phase 1: build the BDG and run the no-producer deadlock check (condition
  // (i) of the design doc). Later phases add parity-cadence and the
  // cross-partition SCC on top of this same node set.
  void runDeadlockChecks(triton::FuncOp funcOp) {
    SmallVector<BDGNode> nodes;
    DenseMap<Operation *, bool> hasArrive; // alloc -> any arrive drives it

    unsigned walk = 0;
    funcOp.walk([&](Operation *op) {
      if (isBarrierStructuralOp(op))
        return;
      bool wait = isWaitOp(op);
      // Only real endpoints become nodes. Ops that merely forward a barrier as
      // a capture/iter-arg (warp_specialize, scf loops) are neither wait nor
      // arrive and must be skipped -- otherwise they register phantom producers
      // that mask condition (i).
      if (!wait && !isArriveOp(op))
        return;
      DenseSet<Operation *> allocsForThisOp;
      for (Value operand : op->getOperands()) {
        Operation *alloc = resolveBarrierAlloc(operand);
        if (!alloc || !allocsForThisOp.insert(alloc).second)
          continue;
        std::optional<bool> bwd = isWSBarrierBackwardEndpoint(op);
        bool isForward = bwd.has_value() && !*bwd;
        // Recover slot index (from memdesc_index %alloc[%idx]) and, for waits,
        // the phase operand + its first-execution polled parity.
        Value slotIdx;
        if (auto mi = operand.getDefiningOp<triton::gpu::MemDescIndexOp>())
          slotIdx = mi->getOperand(1);
        Value phase =
            (wait && isa<ttng::WaitBarrierOp>(op) && op->getNumOperands() > 1)
                ? op->getOperand(1)
                : Value();
        int firstExecParity = phase ? firstExecPolledParity(phase) : -1;
        nodes.push_back({op, alloc, wait, isForward, getSingleTask(op), walk++,
                         slotIdx, phase, firstExecParity});
        if (!wait)
          hasArrive[alloc] = true;
        else
          hasArrive.try_emplace(alloc, false);
      }
    });

    // Condition (i): a FULL (forward) wait whose barrier alloc has no producing
    // arrive anywhere is a genuine deadlock -- the phase it polls is never
    // produced.
    unsigned numNoProducer = 0;
    for (const BDGNode &n : nodes) {
      if (!n.isWait || !n.isForward)
        continue;
      auto it = hasArrive.find(n.barrierAlloc);
      if (it != hasArrive.end() && !it->second) {
        ++numNoProducer;
        n.op->emitError()
            << "WS deadlock: FULL wait_barrier has no producing arrive on its "
               "barrier alloc (the polled phase is never produced)";
      }
    }

    // Cross-partition SCC (condition iii) is only a CANDIDATE surfacer for now:
    // empirically a naive SCC over satisfy + program-order edges also matches a
    // correct pipelined kernel (its steady state is strongly connected),
    // because the benign-pipeline filter requires the phase/pre-arm model,
    // which the current walk-order heuristic does not fully capture. So it is
    // off by default and emits remarks, never errors, until the phase model
    // lands.
    if (reportCycles) {
      DominanceInfo dom(funcOp);
      runCycleCheck(nodes, dom);
    }
    LDBG("deadlock check: " << nodes.size() << " BDG nodes, " << numNoProducer
                            << " no-producer deadlocks");
  }

  // A backward (reuse) wait is "pre-armed" -- non-blocking on first execution
  // -- if some arrive on the same barrier alloc is program-ordered BEFORE it
  // (an entry pre-arm or a same-iteration earlier producer). If the only
  // arrives are program-after (the idle-producer signature), it genuinely
  // blocks.
  // First-execution polled parity of a wait's phase operand (0/1), or -1 if not
  // statically determinable. init_barrier leaves parity 0, so a wait polling 0
  // is satisfied at entry (non-blocking first exec); parity 1 needs an arrive.
  static int firstExecPolledParity(Value phase) {
    DenseSet<Value> seen;
    while (phase && seen.insert(phase).second) {
      if (auto c = phase.getDefiningOp<arith::ConstantOp>()) {
        if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
          return static_cast<int>(ia.getInt() & 1);
        return -1;
      }
      Operation *def = phase.getDefiningOp();
      if (!def)
        return -1; // block arg (loop-carried accumCnt) -> caller decides
      // Inverted empty/reuse acquire xori(base, true) -> parity 1 first exec.
      if (auto x = dyn_cast<arith::XOrIOp>(def)) {
        for (Value o : x.getOperands())
          if (auto cc = o.getDefiningOp<arith::ConstantOp>())
            if (auto ia = dyn_cast<IntegerAttr>(cc.getValue()))
              if (ia.getInt() & 1)
                return 1;
        return -1;
      }
      if (isa<arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp>(def)) {
        phase = def->getOperand(0);
        continue;
      }
      // accumCnt-derived parity (counter starts at 0) is 0 on first execution.
      if (isa<arith::AndIOp, arith::DivUIOp, arith::RemUIOp, arith::MulIOp,
              arith::SubIOp, arith::AddIOp>(def))
        return 0;
      return -1;
    }
    return -1;
  }

  // A wait blocks on first execution iff the parity it polls differs from the
  // barrier parity available at entry. Entry parity is contributed by arrives
  // that DOMINATE the wait (execute before it on every path); cross-partition
  // arrives (isolated partition regions) never dominate, matching the phase
  // model's "no cross-partition pre-arm by an arrive op" rule.
  static bool waitBlocksFirstExec(const BDGNode &w,
                                  const SmallVector<BDGNode> &nodes,
                                  DominanceInfo &dom) {
    int polled = w.firstExecParity; // recovered at node-build time
    unsigned domArrives = 0;
    for (const BDGNode &a : nodes)
      if (!a.isWait && a.barrierAlloc == w.barrierAlloc &&
          dom.dominates(a.op, w.op))
        ++domArrives;
    if (polled < 0)
      return w.isForward || domArrives == 0;
    return polled != static_cast<int>(domArrives & 1);
  }

  // Condition (iii): a cross-partition cycle in the combined graph of satisfy
  // edges (blocking wait -> its producing arrive) and program-order edges (a
  // node -> the nearest earlier node in its own partition). A nontrivial SCC
  // that contains a cross-partition satisfy edge is a deadlock. Backward
  // first-acquires only contribute a satisfy edge when not pre-armed (the
  // first-execution subgraph filter that keeps benign pipelines out).
  void runCycleCheck(const SmallVector<BDGNode> &nodes, DominanceInfo &dom) {
    unsigned n = nodes.size();
    SmallVector<SmallVector<unsigned>> adj(n);
    DenseSet<std::pair<unsigned, unsigned>> crossSatisfy;

    // Satisfy edges.
    for (unsigned i = 0; i < n; ++i) {
      const BDGNode &w = nodes[i];
      if (!w.isWait)
        continue;
      bool blocking = waitBlocksFirstExec(w, nodes, dom);
      if (!blocking)
        continue;
      for (unsigned j = 0; j < n; ++j) {
        const BDGNode &a = nodes[j];
        if (a.isWait || a.barrierAlloc != w.barrierAlloc)
          continue;
        // A same-partition arrive already program-before the wait cannot be
        // part of a first-execution wait-for cycle.
        if (a.partition == w.partition && a.walkIdx < w.walkIdx)
          continue;
        adj[i].push_back(j);
        if (a.partition != w.partition && a.partition >= 0 && w.partition >= 0)
          crossSatisfy.insert({i, j});
      }
    }

    // Program-order edges: node -> nearest earlier node in the same partition.
    for (unsigned i = 0; i < n; ++i) {
      int prev = -1;
      for (unsigned j = 0; j < n; ++j) {
        if (j == i || nodes[j].partition != nodes[i].partition)
          continue;
        if (nodes[j].walkIdx < nodes[i].walkIdx &&
            (prev < 0 || nodes[j].walkIdx > nodes[prev].walkIdx))
          prev = static_cast<int>(j);
      }
      if (prev >= 0)
        adj[i].push_back(static_cast<unsigned>(prev));
    }

    // Tarjan SCC.
    SmallVector<int> index(n, -1), low(n, 0);
    SmallVector<bool> onStack(n, false);
    SmallVector<unsigned> stack;
    int idx = 0;
    SmallVector<SmallVector<unsigned>> sccs;
    std::function<void(unsigned)> strongConnect = [&](unsigned v) {
      index[v] = low[v] = idx++;
      stack.push_back(v);
      onStack[v] = true;
      for (unsigned w : adj[v]) {
        if (index[w] < 0) {
          strongConnect(w);
          low[v] = std::min(low[v], low[w]);
        } else if (onStack[w]) {
          low[v] = std::min(low[v], index[w]);
        }
      }
      if (low[v] == index[v]) {
        SmallVector<unsigned> comp;
        unsigned w;
        do {
          w = stack.back();
          stack.pop_back();
          onStack[w] = false;
          comp.push_back(w);
        } while (w != v);
        sccs.push_back(std::move(comp));
      }
    };
    for (unsigned v = 0; v < n; ++v)
      if (index[v] < 0)
        strongConnect(v);

    unsigned numCycles = 0;
    for (auto &comp : sccs) {
      if (comp.size() < 2)
        continue;
      DenseSet<unsigned> inComp(comp.begin(), comp.end());
      bool hasCross = false;
      for (auto &e : crossSatisfy)
        if (inComp.count(e.first) && inComp.count(e.second)) {
          hasCross = true;
          break;
        }
      if (!hasCross)
        continue;
      ++numCycles;
      for (unsigned v : comp) {
        if (nodes[v].isWait) {
          nodes[v].op->emitRemark()
              << "WS candidate cross-partition mbarrier cycle among "
              << comp.size()
              << " barrier sites (SCC) -- CANDIDATE only; a correct pipeline "
                 "also forms this SCC. Sound classification needs the "
                 "phase/pre-arm first-execution filter (design doc cond. iii)";
          break;
        }
      }
    }
    LDBG("cycle check: " << sccs.size() << " SCCs, " << numCycles
                         << " cross-partition deadlock cycles");
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
