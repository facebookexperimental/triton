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
// (design-doc condition iii). IMPORTANT empirical finding: a naive SCC ALSO
// matches a correct pipelined kernel (verified on the working HSTU DP=1 fwd),
// because separating a benign steady-state pipeline SCC from a real deadlock
// requires the phase/pre-arm first-execution filter, which the current
// walk-order pre-arm heuristic does not fully capture. So the SCC is a
// CANDIDATE surfacer only: off by default (report-cycles), emits remarks, never
// errors, until the phase model lands. Parity-cadence (condition ii) and the
// per-reuse-group Access-Order Graph (race / under-buffering) come in later
// phases.
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
  // Partition (async_task_id; -1 if not exactly one) and a program-order index
  // (pre-order walk position). Used by the cross-partition SCC (condition iii).
  int partition;
  unsigned walkIdx;
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
    DenseMap<Operation *, bool> hasArrive; // alloc -> any arrive drives it

    unsigned walk = 0;
    funcOp.walk([&](Operation *op) {
      if (isBarrierStructuralOp(op))
        return;
      bool wait = isWaitOp(op);
      DenseSet<Operation *> allocsForThisOp;
      for (Value operand : op->getOperands()) {
        Operation *alloc = resolveBarrierAlloc(operand);
        if (!alloc || !allocsForThisOp.insert(alloc).second)
          continue;
        std::optional<bool> bwd = isWSBarrierBackwardEndpoint(op);
        bool isForward = bwd.has_value() && !*bwd;
        nodes.push_back(
            {op, alloc, wait, isForward, getSingleTask(op), walk++});
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
    if (reportCycles)
      runCycleCheck(nodes);
    LDBG("deadlock check: " << nodes.size() << " BDG nodes, " << numNoProducer
                            << " no-producer deadlocks");
  }

  // A backward (reuse) wait is "pre-armed" -- non-blocking on first execution
  // -- if some arrive on the same barrier alloc is program-ordered BEFORE it
  // (an entry pre-arm or a same-iteration earlier producer). If the only
  // arrives are program-after (the idle-producer signature), it genuinely
  // blocks.
  static bool backwardWaitIsPreArmed(const BDGNode &w,
                                     const SmallVector<BDGNode> &nodes) {
    for (const BDGNode &a : nodes)
      if (!a.isWait && a.barrierAlloc == w.barrierAlloc &&
          a.walkIdx < w.walkIdx)
        return true;
    return false;
  }

  // Condition (iii): a cross-partition cycle in the combined graph of satisfy
  // edges (blocking wait -> its producing arrive) and program-order edges (a
  // node -> the nearest earlier node in its own partition). A nontrivial SCC
  // that contains a cross-partition satisfy edge is a deadlock. Backward
  // first-acquires only contribute a satisfy edge when not pre-armed (the
  // first-execution subgraph filter that keeps benign pipelines out).
  void runCycleCheck(const SmallVector<BDGNode> &nodes) {
    unsigned n = nodes.size();
    SmallVector<SmallVector<unsigned>> adj(n);
    DenseSet<std::pair<unsigned, unsigned>> crossSatisfy;

    // Satisfy edges.
    for (unsigned i = 0; i < n; ++i) {
      const BDGNode &w = nodes[i];
      if (!w.isWait)
        continue;
      bool blocking = w.isForward || !backwardWaitIsPreArmed(w, nodes);
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
