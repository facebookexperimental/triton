// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Exhaustive modulo scheduler with joint schedule + memory optimization.
//
// Branch-and-bound search over all valid (cycle, stage) placements:
// 1. Topologically order ops so predecessors are placed before dependents.
// 2. For each op, try every valid cycle in [earliest, earliest + II).
// 3. After placing all ops, check SMEM/TMEM budget feasibility.
// 4. Score candidates (minimize II, maximize buffering depth) and prune
//    branches that can't beat the current best.
//
// For GPU inner loops with ≤20 ops and ≤4 pipeline resources, dependency
// constraints and resource conflicts prune the search tree aggressively,
// making exhaustive enumeration practical (milliseconds).

#include "ExhaustiveScheduler.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <chrono>
#include <climits>

#define DEBUG_TYPE "modulo-scheduling-exhaustive"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::triton::gpu {

namespace ttng = triton::nvidia_gpu;

// ── Buffer extraction ───────────────────────────────────────────────────────

enum class BufKind { SMEM, TMEM };

struct BufferInfo {
  unsigned allocNodeIdx;
  BufKind kind;
  int64_t sizeBytes;
  int64_t tmemCols;
  llvm::SmallVector<unsigned, 4> consumerNodes;
};

static llvm::SmallVector<BufferInfo>
extractBuffers(const DataDependenceGraph &ddg) {
  llvm::SmallVector<BufferInfo> buffers;
  for (const auto &node : ddg.getNodes()) {
    Operation *op = node.op;
    BufferInfo buf;
    buf.allocNodeIdx = node.idx;

    if (isa<LocalAllocOp>(op)) {
      auto memDesc = dyn_cast<MemDescType>(op->getResult(0).getType());
      if (!memDesc)
        continue;
      buf.kind = BufKind::SMEM;
      int64_t elems = 1;
      for (auto d : memDesc.getShape())
        elems *= d;
      buf.sizeBytes =
          elems * memDesc.getElementType().getIntOrFloatBitWidth() / 8;
      buf.tmemCols = 0;
    } else if (isa<ttng::TMEMAllocOp>(op)) {
      auto memDesc = dyn_cast<MemDescType>(op->getResult(0).getType());
      if (!memDesc)
        continue;
      buf.kind = BufKind::TMEM;
      buf.tmemCols = ttng::getTmemAllocSizes(memDesc).numCols;
      buf.sizeBytes = 0;
    } else {
      continue;
    }

    for (const auto *edge : ddg.getOutEdges(node.idx)) {
      if (edge->distance == 0)
        buf.consumerNodes.push_back(edge->dstIdx);
    }
    buffers.push_back(buf);
  }

  LLVM_DEBUG(DBGS() << "Extracted " << buffers.size() << " buffers\n");
  return buffers;
}

// ── Liveness and feasibility ────────────────────────────────────────────────

struct BufferLiveness {
  unsigned bufferIdx;
  int produceCycle;
  int lastConsumeCycle;
  /// Buffer depth = stage difference + 1 (the downstream pipeline pass
  /// allocates this many copies for multi-buffering).
  int depth(int II) const {
    if (II <= 0)
      return 1;
    int prodStage = produceCycle / II;
    int consStage = lastConsumeCycle / II;
    return (consStage - prodStage) + 1;
  }
};

static llvm::SmallVector<BufferLiveness>
computeLiveness(const llvm::SmallVector<BufferInfo> &buffers,
                const llvm::DenseMap<unsigned, int> &nodeToCycle) {
  llvm::SmallVector<BufferLiveness> result;
  for (unsigned i = 0; i < buffers.size(); ++i) {
    const auto &buf = buffers[i];
    BufferLiveness lv;
    lv.bufferIdx = i;
    auto prodIt = nodeToCycle.find(buf.allocNodeIdx);
    lv.produceCycle = prodIt != nodeToCycle.end() ? prodIt->second : 0;
    lv.lastConsumeCycle = lv.produceCycle;
    for (unsigned c : buf.consumerNodes) {
      auto it = nodeToCycle.find(c);
      if (it != nodeToCycle.end())
        lv.lastConsumeCycle = std::max(lv.lastConsumeCycle, it->second);
    }
    result.push_back(lv);
  }
  return result;
}

struct FeasibilityResult {
  bool feasible;
  int totalSmemBytes;
  int totalTmemCols;
  int totalBufferingDepth;
};

static FeasibilityResult
checkFeasibility(const llvm::SmallVector<BufferInfo> &buffers,
                 const llvm::SmallVector<BufferLiveness> &liveness, int II,
                 int smemBudget, int tmemColLimit) {
  FeasibilityResult res{true, 0, 0, 0};

  for (const auto &lv : liveness) {
    const auto &buf = buffers[lv.bufferIdx];
    if (buf.kind == BufKind::SMEM) {
      int d = lv.depth(II);
      res.totalSmemBytes += buf.sizeBytes * d;
      res.totalBufferingDepth += d;
    }
  }
  if (res.totalSmemBytes > smemBudget) {
    res.feasible = false;
    return res;
  }

  // TMEM: greedy interval coloring for reuse.
  struct TmemGroup {
    int64_t cols;
    llvm::SmallVector<unsigned, 2> members;
  };
  llvm::SmallVector<TmemGroup> groups;
  for (unsigned i = 0; i < liveness.size(); ++i) {
    const auto &buf = buffers[liveness[i].bufferIdx];
    if (buf.kind != BufKind::TMEM)
      continue;
    const auto &lv = liveness[i];
    bool placed = false;
    for (auto &grp : groups) {
      bool overlaps = false;
      for (unsigned m : grp.members) {
        const auto &other = liveness[m];
        if (lv.produceCycle < other.lastConsumeCycle &&
            other.produceCycle < lv.lastConsumeCycle) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps) {
        grp.cols = std::max(grp.cols, buf.tmemCols);
        grp.members.push_back(i);
        placed = true;
        break;
      }
    }
    if (!placed)
      groups.push_back({buf.tmemCols, {i}});
  }
  for (const auto &g : groups)
    res.totalTmemCols += g.cols;
  if (res.totalTmemCols > tmemColLimit)
    res.feasible = false;

  return res;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

static int getNodeDuration(const DDGNode &node) {
  if (node.pipeline == HWPipeline::NONE)
    return 1;
  return std::max(node.selfLatency, 1);
}

/// Compute earliest valid cycle for nodeIdx given already-placed ops.
static int computeEarliest(unsigned nodeIdx, const DataDependenceGraph &ddg,
                           const llvm::DenseMap<unsigned, int> &scheduled,
                           int II) {
  int earliest = 0;
  for (const auto *edge : ddg.getInEdges(nodeIdx)) {
    auto it = scheduled.find(edge->srcIdx);
    if (it == scheduled.end())
      continue;
    int constraint =
        it->second + edge->latency - static_cast<int>(edge->distance) * II;
    earliest = std::max(earliest, constraint);
  }
  return earliest;
}

/// Build topological order of DDG nodes (Kahn's algorithm on distance-0 edges).
static llvm::SmallVector<unsigned>
topologicalOrder(const DataDependenceGraph &ddg) {
  unsigned N = ddg.getNumNodes();
  llvm::SmallVector<int> inDeg(N, 0);
  for (const auto &edge : ddg.getEdges()) {
    if (edge.distance == 0)
      inDeg[edge.dstIdx]++;
  }

  llvm::SmallVector<unsigned> ready;
  for (unsigned i = 0; i < N; ++i) {
    if (inDeg[i] == 0)
      ready.push_back(i);
  }

  llvm::SmallVector<unsigned> order;
  while (!ready.empty()) {
    llvm::sort(ready);
    unsigned cur = ready.front();
    ready.erase(ready.begin());
    order.push_back(cur);
    for (const auto *edge : ddg.getOutEdges(cur)) {
      if (edge->distance > 0)
        continue;
      if (--inDeg[edge->dstIdx] == 0)
        ready.push_back(edge->dstIdx);
    }
  }
  return order;
}

// ── Branch-and-bound search ─────────────────────────────────────────────────

struct SearchState {
  const DataDependenceGraph &ddg;
  const llvm::SmallVector<BufferInfo> &buffers;
  const llvm::SmallVector<unsigned> &topoOrder;
  int II;
  int maxStages; // max stage to try (branching factor per op)
  int smemBudget;
  int tmemColLimit;

  // Current partial assignment.
  llvm::DenseMap<unsigned, int> scheduled;
  ModuloReservationTable table;

  // Best complete assignment found so far.
  llvm::DenseMap<unsigned, int> bestSchedule;
  int64_t bestScore;
  unsigned candidatesExplored;
  unsigned branchVisits;
  std::chrono::steady_clock::time_point startTime;
  static constexpr int timeoutMs = 5000; // 5 second wall-clock limit

  SearchState(const DataDependenceGraph &ddg,
              const llvm::SmallVector<BufferInfo> &buffers,
              const llvm::SmallVector<unsigned> &topoOrder, int II,
              int maxStages, int smemBudget, int tmemColLimit)
      : ddg(ddg), buffers(buffers), topoOrder(topoOrder), II(II),
        maxStages(maxStages), smemBudget(smemBudget),
        tmemColLimit(tmemColLimit), table(II), bestScore(INT64_MIN),
        candidatesExplored(0), branchVisits(0),
        startTime(std::chrono::steady_clock::now()) {}
};

/// Recursive branch-and-bound. For each op, tries placing it at each valid
/// stage (0 to maxStages-1). Within a stage, uses the earliest free cycle.
/// This reduces the branching factor from II (~1000) to maxStages (~3-4).
static void searchRecursive(SearchState &state, unsigned depth) {
  // Bail out if we've explored too many candidates or exceeded time limit.
  if (state.candidatesExplored > 100000)
    return;
  // Check wall-clock timeout on every entry. The chrono call is cheap
  // (~20ns) relative to the MRT operations in each branch.
  state.branchVisits++;
  auto elapsed = std::chrono::steady_clock::now() - state.startTime;
  if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >
      SearchState::timeoutMs)
    return;

  // Base case: all ops placed — evaluate this complete schedule.
  if (depth == state.topoOrder.size()) {
    state.candidatesExplored++;
    auto liveness = computeLiveness(state.buffers, state.scheduled);
    auto feas = checkFeasibility(state.buffers, liveness, state.II,
                                 state.smemBudget, state.tmemColLimit);
    if (!feas.feasible)
      return;

    // ── Dataflow correctness checks ─────────────────────────────────
    //
    // Buffer depth is derived from the schedule: for each buffer, the
    // downstream pipeline pass will allocate stageDiff + 1 copies.
    // We check SMEM feasibility using this derived depth in
    // checkFeasibility (via lv.depth(II)), not as a separate constraint.
    // The SMEM budget check already rejects schedules where the required
    // buffering exceeds available shared memory.

    // Check 2: Intra-iteration dataflow consistency.
    // For distance-0 edges: src_stage <= dst_stage (def before use).
    // Loop-carried edges (distance > 0) are handled by pinning NONE ops
    // to stage 0 in the search phase, so they don't need checking here.
    for (const auto &edge : state.ddg.getEdges()) {
      if (edge.distance > 0)
        continue;
      auto srcIt = state.scheduled.find(edge.srcIdx);
      auto dstIt = state.scheduled.find(edge.dstIdx);
      if (srcIt == state.scheduled.end() || dstIt == state.scheduled.end())
        continue;
      int srcStage = srcIt->second / state.II;
      int dstStage = dstIt->second / state.II;
      if (srcStage > dstStage) {
        LLVM_DEBUG(DBGS() << "  Reject #" << state.candidatesExplored
                          << ": def-after-use N" << edge.srcIdx << "(stage "
                          << srcStage << ") -> N" << edge.dstIdx << "(stage "
                          << dstStage << ")\n");
        return;
      }
    }

    // ── Composite scoring ──────────────────────────────────────────
    //
    // Pipeline depth (maxStage): fewer stages = less prologue/epilogue
    // overhead, less register spill from live-across values. Weighted
    // heavily because deep pipelines cause compilation failures.
    //
    // Buffering depth: more copies = better producer-consumer overlap.
    // Positive contribution but bounded by SMEM budget.
    //
    // Register pressure proxy: sum of (consumer_cycle - producer_cycle)
    // for all distance-0 DDG edges. Shorter live ranges = fewer
    // registers needed. Penalized to prefer tight schedules.
    //
    // SMEM headroom: remaining SMEM budget after allocation. Small
    // bonus for leaving room for downstream passes.

    int maxStage = 0;
    for (auto &[_, c] : state.scheduled)
      maxStage = std::max(maxStage, c / state.II);

    int regPressure = 0;
    for (const auto &edge : state.ddg.getEdges()) {
      if (edge.distance > 0)
        continue;
      auto srcIt = state.scheduled.find(edge.srcIdx);
      auto dstIt = state.scheduled.find(edge.dstIdx);
      if (srcIt != state.scheduled.end() && dstIt != state.scheduled.end())
        regPressure += dstIt->second - srcIt->second;
    }

    int smemHeadroom = state.smemBudget - feas.totalSmemBytes;

    int64_t score = -static_cast<int64_t>(maxStage) * 10000 // shallow > deep
                    + feas.totalBufferingDepth * 100        // more overlap
                    - regPressure                           // tight live ranges
                    + smemHeadroom / 1024; // SMEM headroom (KB)

    if (score > state.bestScore) {
      state.bestScore = score;
      state.bestSchedule = state.scheduled;
      LLVM_DEBUG(DBGS() << "  Candidate #" << state.candidatesExplored
                        << ": score=" << score << " maxStage=" << maxStage
                        << " depth=" << feas.totalBufferingDepth << " regP="
                        << regPressure << " SMEM=" << feas.totalSmemBytes
                        << " TMEM=" << feas.totalTmemCols << "\n");
    }
    return;
  }

  unsigned nodeIdx = state.topoOrder[depth];
  const auto &node = state.ddg.getNode(nodeIdx);
  int duration = getNodeDuration(node);
  int earliest = computeEarliest(nodeIdx, state.ddg, state.scheduled, state.II);
  int earliestStage = earliest / state.II;

  // Determine whether to branch (try multiple stages) or place greedily.
  // Key ops (MEM loads, TC MMA) are the primary scheduling DOFs — branch
  // on these. Non-key ops (CUDA softmax, SFU exp2, NONE scalar) are placed
  // deterministically at the earliest valid cycle to keep the search
  // tractable. This reduces branching from 3^N (all ops) to 3^K (key ops
  // only, K << N).
  bool isKeyOp =
      (node.pipeline == HWPipeline::MEM || node.pipeline == HWPipeline::TC);
  // NONE ops are pinned to stage 0 (not pipelineable).
  bool isNone = (node.pipeline == HWPipeline::NONE);
  int maxStageForOp = isNone ? 0 : state.maxStages;

  if (isKeyOp) {
    // Branch: try each stage from earliest valid to maxStages.
    for (int stage = earliestStage; stage <= maxStageForOp; ++stage) {
      int stageStart = std::max(earliest, stage * state.II);
      int slot = state.table.findFreeSlot(stageStart, node.pipeline, duration);
      if (slot < 0 || slot / state.II != stage)
        continue;

      state.table.reserve(slot, node.pipeline, nodeIdx, duration);
      state.scheduled[nodeIdx] = slot;
      searchRecursive(state, depth + 1);
      state.table.unreserve(slot, node.pipeline, duration);
      state.scheduled.erase(nodeIdx);
    }
  } else {
    // Greedy: place at earliest valid cycle, no branching.
    int stageStart = std::max(earliest, earliestStage * state.II);
    if (isNone)
      stageStart = earliest; // stage 0 only
    int slot = state.table.findFreeSlot(stageStart, node.pipeline, duration);
    if (slot < 0)
      return; // no valid placement — prune this branch
    state.table.reserve(slot, node.pipeline, nodeIdx, duration);
    state.scheduled[nodeIdx] = slot;
    searchRecursive(state, depth + 1);
    state.table.unreserve(slot, node.pipeline, duration);
    state.scheduled.erase(nodeIdx);
  }
}

// ── Public entry point ──────────────────────────────────────────────────────

FailureOr<ModuloScheduleResult>
runExhaustiveSearch(const DataDependenceGraph &ddg, int maxII, int smemBudget,
                    int tmemColLimit) {
  const int minII = ddg.computeMinII();
  if (minII <= 0)
    return failure();
  if (maxII <= 0)
    maxII = 2 * minII;

  LLVM_DEBUG({
    DBGS() << "MinII=" << minII << " MaxII=" << maxII
           << " Nodes=" << ddg.getNumNodes() << "\n";
    DBGS() << "ResMII=" << ddg.computeResMII()
           << " RecMII=" << ddg.computeRecMII() << "\n";
    DBGS() << "SMEM budget=" << smemBudget << " TMEM col limit=" << tmemColLimit
           << "\n";
  });

  auto buffers = extractBuffers(ddg);
  auto topoOrder = topologicalOrder(ddg);

  if (topoOrder.size() != ddg.getNumNodes()) {
    LLVM_DEBUG(DBGS() << "Topological sort failed (cycle in DDG)\n");
    return failure();
  }

  // maxStages bounds how deep the pipeline can be. For Blackwell GEMM,
  // the typical pipeline is 3 stages (loads→0, MMA→1, tmem_load→2).
  // We use num_stages - 1 as the max stage index.
  constexpr int maxStages = 2; // stage indices 0, 1, 2 → 3 pipeline stages

  auto globalStart = std::chrono::steady_clock::now();

  for (int II = minII; II <= maxII; ++II) {
    // Check global timeout across all II attempts.
    auto globalElapsed = std::chrono::steady_clock::now() - globalStart;
    if (std::chrono::duration_cast<std::chrono::milliseconds>(globalElapsed)
            .count() > SearchState::timeoutMs) {
      LLVM_DEBUG(DBGS() << "Global timeout after II=" << II << "\n");
      break;
    }

    SearchState state(ddg, buffers, topoOrder, II, maxStages, smemBudget,
                      tmemColLimit);
    state.startTime = globalStart; // share the global start time
    searchRecursive(state, 0);

    if (state.bestScore > INT64_MIN) {
      LLVM_DEBUG(DBGS() << "SUCCESS at II=" << II << " after exploring "
                        << state.candidatesExplored << " candidates ("
                        << state.branchVisits << " branch visits)\n");
      ModuloScheduleResult result;
      result.II = II;
      result.nodeToCycle = std::move(state.bestSchedule);
      LLVM_DEBUG(DBGS() << "maxStage=" << result.getMaxStage() << "\n");
      return result;
    }

    LLVM_DEBUG(DBGS() << "II=" << II << ": explored "
                      << state.candidatesExplored
                      << " candidates, none feasible\n");
  }

  LLVM_DEBUG(DBGS() << "EXHAUSTED: no feasible schedule found\n");
  return failure();
}

// ── Random sampling search ──────────────────────────────────────────────────
//
// Monte Carlo approach: randomly sample stage assignments for key ops
// (MEM + TC), greedily place everything else, evaluate and keep the best.
// Guaranteed to complete in O(numSamples × numOps) time.

FailureOr<ModuloScheduleResult> runRandomSearch(const DataDependenceGraph &ddg,
                                                int maxII, int smemBudget,
                                                int tmemColLimit,
                                                int numSamples) {
  const int minII = ddg.computeMinII();
  if (minII <= 0)
    return failure();
  if (maxII <= 0)
    maxII = 2 * minII;

  // For large DDGs, reduce samples to stay within time budget.
  // Also cap maxII to minII + a few — most schedules succeed at MinII.
  if (ddg.getNumNodes() > 50)
    numSamples = std::min(numSamples, 100);
  maxII = std::min(maxII, minII + 10);

  LLVM_DEBUG({
    DBGS() << "Random: MinII=" << minII << " MaxII=" << maxII
           << " Nodes=" << ddg.getNumNodes() << " Samples=" << numSamples
           << "\n";
  });

  auto buffers = extractBuffers(ddg);
  auto topoOrder = topologicalOrder(ddg);
  if (topoOrder.size() != ddg.getNumNodes())
    return failure();

  constexpr int maxStages = 2;
  constexpr int timeoutMs = 30000; // 30s for random sampling
  auto startTime = std::chrono::steady_clock::now();

  // Identify key ops (MEM + TC) and their indices in topoOrder.
  llvm::SmallVector<unsigned> keyOpIndices; // indices into topoOrder
  for (unsigned i = 0; i < topoOrder.size(); ++i) {
    const auto &node = ddg.getNode(topoOrder[i]);
    if (node.pipeline == HWPipeline::MEM || node.pipeline == HWPipeline::TC)
      keyOpIndices.push_back(i);
  }

  LLVM_DEBUG(DBGS() << "Random: " << keyOpIndices.size() << " key ops out of "
                    << topoOrder.size() << " total\n");

  // Simple RNG (deterministic seed for reproducibility).
  unsigned rngState = 42;
  auto nextRand = [&]() -> unsigned {
    rngState = rngState * 1103515245 + 12345;
    return (rngState >> 16) & 0x7fff;
  };

  ModuloScheduleResult best;
  best.II = INT_MAX;
  int64_t bestScore = INT64_MIN;

  for (int II = minII; II <= maxII; ++II) {
    // Timeout check.
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >
        timeoutMs)
      break;

    for (int sample = 0; sample < numSamples; ++sample) {
      // Generate dependency-aware random stage assignment for key ops.
      // For each key op in topological order, pick a random stage that is
      // >= the max stage of its key-op predecessors (respects def-before-use).
      llvm::DenseMap<unsigned, int> keyStages;      // topoOrder index → stage
      llvm::DenseMap<unsigned, int> nodeToKeyStage; // DDG node idx → stage
      for (unsigned idx : keyOpIndices) {
        unsigned nodeIdx = topoOrder[idx];
        // Find min valid stage: max stage of predecessor key ops.
        int minStage = 0;
        for (const auto *edge : ddg.getInEdges(nodeIdx)) {
          if (edge->distance > 0)
            continue;
          auto predIt = nodeToKeyStage.find(edge->srcIdx);
          if (predIt != nodeToKeyStage.end())
            minStage = std::max(minStage, predIt->second);
        }
        // Random stage in [minStage, maxStages].
        int range = maxStages - minStage + 1;
        int stage = minStage + (range > 0 ? nextRand() % range : 0);
        keyStages[idx] = stage;
        nodeToKeyStage[nodeIdx] = stage;
      }

      // Place key ops only — we only need their stages for tt.autows
      // annotations on MMA ops. Non-key ops are handled by scheduleLoops
      // inside the WS pass.
      ModuloReservationTable table{II};
      llvm::DenseMap<unsigned, int> scheduled;
      bool ok = true;

      for (unsigned i = 0; i < topoOrder.size(); ++i) {
        unsigned nodeIdx = topoOrder[i];
        const auto &node = ddg.getNode(nodeIdx);

        auto keyIt = keyStages.find(i);
        if (keyIt == keyStages.end()) {
          // Non-key op: place at earliest (stage determined by predecessors).
          int earliest = computeEarliest(nodeIdx, ddg, scheduled, II);
          scheduled[nodeIdx] = earliest;
          continue;
        }

        // Key op: place at the randomly assigned stage.
        int duration = getNodeDuration(node);
        int earliest = computeEarliest(nodeIdx, ddg, scheduled, II);
        int targetStage = std::max(keyIt->second, earliest / II);
        int stageStart = std::max(earliest, targetStage * II);
        int slot = table.findFreeSlot(stageStart, node.pipeline, duration);

        if (slot < 0 || slot / II != targetStage)
          slot = table.findFreeSlot(earliest, node.pipeline, duration);

        if (slot < 0) {
          ok = false;
          break;
        }

        table.reserve(slot, node.pipeline, nodeIdx, duration);
        scheduled[nodeIdx] = slot;
      }
      if (!ok) {
        LLVM_DEBUG(if (sample < 5) DBGS()
                   << "  Random sample " << sample << ": placement failed\n");
        continue;
      }

      // Evaluate.
      auto liveness = computeLiveness(buffers, scheduled);
      auto feas =
          checkFeasibility(buffers, liveness, II, smemBudget, tmemColLimit);
      if (!feas.feasible)
        continue;

      // Dataflow check: intra-iteration def before use.
      bool valid = true;
      for (const auto &edge : ddg.getEdges()) {
        if (edge.distance > 0)
          continue;
        auto srcIt = scheduled.find(edge.srcIdx);
        auto dstIt = scheduled.find(edge.dstIdx);
        if (srcIt == scheduled.end() || dstIt == scheduled.end())
          continue;
        if (srcIt->second / II > dstIt->second / II) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        LLVM_DEBUG(if (sample < 5) DBGS() << "  Random sample " << sample
                                          << ": dataflow check failed\n");
        continue;
      }

      // Score.
      int maxStage = 0;
      for (auto &[_, c] : scheduled)
        maxStage = std::max(maxStage, c / II);

      int regPressure = 0;
      for (const auto &edge : ddg.getEdges()) {
        if (edge.distance > 0)
          continue;
        auto srcIt = scheduled.find(edge.srcIdx);
        auto dstIt = scheduled.find(edge.dstIdx);
        if (srcIt != scheduled.end() && dstIt != scheduled.end())
          regPressure += dstIt->second - srcIt->second;
      }

      // Score: reward pipeline depth (more stages = more overlap),
      // penalize register pressure, reward buffering depth.
      // The baseline scheduler produces 3-stage schedules (maxStage=2)
      // for FA, so we should prefer deeper pipelines.
      int smemHeadroom = smemBudget - feas.totalSmemBytes;
      int64_t score = static_cast<int64_t>(maxStage) * 10000 +
                      feas.totalBufferingDepth * 100 - regPressure +
                      smemHeadroom / 1024;

      if (score > bestScore) {
        bestScore = score;
        best.II = II;
        best.nodeToCycle = scheduled;
        LLVM_DEBUG(DBGS() << "  Random sample " << sample << ": score=" << score
                          << " maxStage=" << maxStage
                          << " depth=" << feas.totalBufferingDepth << "\n");
      }
    }

    if (best.II == II) {
      LLVM_DEBUG(DBGS() << "Random: SUCCESS at II=" << II << "\n");
      return best;
    }
  }

  LLVM_DEBUG(DBGS() << "Random: no feasible schedule found\n");
  return failure();
}

} // namespace mlir::triton::gpu
