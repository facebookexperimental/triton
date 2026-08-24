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
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <numeric>
#include <set>

#define DEBUG_TYPE "modulo-scheduling-exhaustive"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::triton::gpu {

// ── STANDALONE_MODULO: opt-in exhaustive top-K exploration ──────────────────
//
// When STANDALONE_MODULO=1 AND TRITON_USE_MODULO_SCHEDULE=exhaustive, the
// exhaustive search shifts to a top-K exploration mode:
//   (1) it collects the top-K DISTINCT feasible schedules (ranked by score,
//       selected via TRITON_MODULO_TOPK / TRITON_MODULO_PICK) instead of a
//       single best — exhaustive enumerates the whole space, so this is a true
//       (non-sampled) top-K;
//   (2) it IGNORES the SMEM/TMEM hardware budget (both the feasibility reject
//       and the headroom score term) so over-budget schedules are still
//       enumerated — they may fail later at allocation/codegen and are meant to
//       be swept empirically;
//   (3) it constrains the pipeline so no distance-0 dependence spans more than
//       STANDALONE_MODULO_MAX_STAGE_DIFF stages (default 1).
// Default (unset) preserves the single-best, budget-respecting behavior.
static bool getStandaloneModulo() {
  return ::mlir::triton::tools::getBoolEnv("STANDALONE_MODULO");
}

// Cap on the stage span of any distance-0 dependence in STANDALONE_MODULO mode.
// Default 1 (the requested "max stage diff = 1"); raise it to explore deeper
// pipelines on kernels whose dependence chains cannot fit within one stage.
static int getStandaloneMaxStageDiff() {
  auto v = ::mlir::triton::tools::getStrEnv("STANDALONE_MODULO_MAX_STAGE_DIFF");
  if (v.empty())
    return 1;
  int n = std::atoi(v.c_str());
  return n < 1 ? 1 : n;
}

// Defined later in this file; needed by the exhaustive top-K path above.
static int getModuloTopK();
static int getModuloPick();
static llvm::SmallVector<int>
stageSignature(const DataDependenceGraph &ddg,
               const llvm::DenseMap<unsigned, int> &scheduled, int II);
static uint64_t hashStageSig(const llvm::SmallVector<int> &s);

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
    } else if (node.tmemAllocCols > 0) {
      // Target-specific accumulator alloc (e.g. Blackwell TMEM), size
      // precomputed on the DDG node via the LatencyModel (HW-agnostic here).
      // Invariant: the LatencyModel sets tmemAllocCols > 0 for every real
      // accumulator and 0 otherwise, so this identifies accumulator buffers
      // without a backend-specific op check. A 0-col accumulator would be
      // dropped here — asserted impossible in
      // NVLatencyModel::getAccumulatorAllocCols.
      buf.kind = BufKind::TMEM;
      buf.tmemCols = node.tmemAllocCols;
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

  // STANDALONE_MODULO top-K exploration (see getStandaloneModulo()). When
  // `standalone` is set: `ignoreBudget` skips the SMEM/TMEM feasibility+score,
  // `maxStageDiff` caps any distance-0 dependence's stage span, and every
  // distinct feasible schedule is collected into `cands` (deduped via
  // `seenSigs`) for ranking instead of tracking a single best.
  bool standalone = false;
  bool ignoreBudget = false;
  int maxStageDiff = INT_MAX;
  llvm::SmallVector<std::pair<int64_t, llvm::DenseMap<unsigned, int>>> cands;
  llvm::DenseSet<uint64_t> seenSigs;

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
    ModuloScheduleResult candidate;
    candidate.II = state.II;
    candidate.nodeToCycle = state.scheduled;
    if (!tryRepairModuloSchedule(state.ddg, candidate)) {
      LLVM_DEBUG(DBGS() << "  Reject #" << state.candidatesExplored
                        << ": dependency-invalid schedule\n");
      return;
    }
    const auto &schedule = candidate.nodeToCycle;
    auto liveness = computeLiveness(state.buffers, schedule);
    auto feas = checkFeasibility(state.buffers, liveness, state.II,
                                 state.smemBudget, state.tmemColLimit);
    if (!state.ignoreBudget && !feas.feasible)
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
      auto srcIt = schedule.find(edge.srcIdx);
      auto dstIt = schedule.find(edge.dstIdx);
      if (srcIt == schedule.end() || dstIt == schedule.end())
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
      if (dstStage - srcStage > state.maxStageDiff)
        return;
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
    for (auto &[_, c] : schedule)
      maxStage = std::max(maxStage, c / state.II);

    int regPressure = 0;
    for (const auto &edge : state.ddg.getEdges()) {
      if (edge.distance > 0)
        continue;
      auto srcIt = schedule.find(edge.srcIdx);
      auto dstIt = schedule.find(edge.dstIdx);
      if (srcIt != schedule.end() && dstIt != schedule.end())
        regPressure += dstIt->second - srcIt->second;
    }

    int64_t headroomTerm =
        state.ignoreBudget ? 0
                           : (state.smemBudget - feas.totalSmemBytes) / 1024;

    int64_t score = -static_cast<int64_t>(maxStage) * 10000 // shallow > deep
                    + feas.totalBufferingDepth * 100        // more overlap
                    - regPressure                           // tight live ranges
                    + headroomTerm; // SMEM headroom (KB)

    if (state.standalone) {
      if (state.seenSigs
              .insert(
                  hashStageSig(stageSignature(state.ddg, schedule, state.II)))
              .second) {
        state.cands.push_back({score, schedule});
        LLVM_DEBUG(DBGS() << "  Standalone cand #" << state.cands.size()
                          << ": score=" << score << " maxStage=" << maxStage
                          << " depth=" << feas.totalBufferingDepth << "\n");
      }
    } else if (score > state.bestScore) {
      state.bestScore = score;
      state.bestSchedule = schedule;
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
      (node.pipeline == HWPipeline::TMA || node.pipeline == HWPipeline::TC);
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
                    int tmemColLimit, int minIIOverride) {
  const int minII = std::max(ddg.computeMinII(), minIIOverride);
  if (minII <= 0)
    return failure();
  if (maxII <= 0)
    maxII = 2 * minII;
  else if (maxII < minII)
    return failure();

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

  // STANDALONE_MODULO shifts to top-K exploration (ignore budget,
  // stage-span<=1, collect+rank K distinct schedules). See
  // getStandaloneModulo().
  const bool standalone = getStandaloneModulo();

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
    if (standalone) {
      state.standalone = true;
      state.ignoreBudget = true;
      state.maxStageDiff = getStandaloneMaxStageDiff();
    }
    searchRecursive(state, 0);

    if (standalone) {
      if (state.cands.empty()) {
        LLVM_DEBUG(DBGS() << "II=" << II << ": no standalone candidates\n");
        continue;
      }
      // Rank by score desc (II is fixed within this DFS); keep top-K, apply the
      // TRITON_MODULO_PICK-th (0 = best). Prefer the lowest feasible II (return
      // at the first II that yields candidates).
      llvm::stable_sort(state.cands, [](const auto &a, const auto &b) {
        return a.first > b.first;
      });
      int K = std::max(getModuloTopK(), getModuloPick() + 1);
      int nTop = std::min<int>(K, state.cands.size());
      int pick = std::min(getModuloPick(), nTop - 1);
      LLVM_DEBUG({
        DBGS() << "Standalone exhaustive top-" << nTop << " at II=" << II
               << " (applying pick " << pick << ") from " << state.cands.size()
               << " distinct schedules:\n";
        for (int i = 0; i < nTop; ++i) {
          auto sig = stageSignature(ddg, state.cands[i].second, II);
          int mx = 0;
          for (int s : sig)
            mx = std::max(mx, s);
          DBGS() << "   rank " << i << " score=" << state.cands[i].first
                 << " maxStage=" << mx << " stages(per-node)=[";
          for (size_t j = 0; j < sig.size(); ++j)
            llvm::dbgs() << (j ? "," : "") << sig[j];
          llvm::dbgs() << "]\n";
        }
      });
      ModuloScheduleResult result;
      result.II = II;
      result.nodeToCycle = std::move(state.cands[pick].second);
      return result;
    }

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
//
// Top-K: the search already enumerates hundreds/thousands of scored candidates
// internally. TRITON_MODULO_TOPK>1 keeps the K best (deduped by per-node stage
// signature, ranked by II then score) instead of only the single best;
// TRITON_MODULO_PICK selects which of the K to apply (0 = best), so an external
// harness can sweep schedules. Default (unset) preserves single-best behavior.

static int getModuloTopK() {
  auto v = ::mlir::triton::tools::getStrEnv("TRITON_MODULO_TOPK");
  if (v.empty())
    return 1;
  int n = std::atoi(v.c_str());
  return n < 1 ? 1 : n;
}

static int getModuloPick() {
  auto v = ::mlir::triton::tools::getStrEnv("TRITON_MODULO_PICK");
  if (v.empty())
    return 0;
  int n = std::atoi(v.c_str());
  return n < 0 ? 0 : n;
}

// Canonical dedup key: per-node stage (cycle / II). Two schedules with the same
// stage assignment are equivalent for modulo purposes.
static llvm::SmallVector<int>
stageSignature(const DataDependenceGraph &ddg,
               const llvm::DenseMap<unsigned, int> &scheduled, int II) {
  llvm::SmallVector<int> sig;
  sig.reserve(ddg.getNumNodes());
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i) {
    auto it = scheduled.find(i);
    sig.push_back((it != scheduled.end() && II > 0) ? it->second / II : -1);
  }
  return sig;
}

// FNV-1a hash of a stage signature, for O(1) dedup of distinct schedules.
static uint64_t hashStageSig(const llvm::SmallVector<int> &s) {
  uint64_t h = 1469598103934665603ull;
  for (int x : s) {
    h ^= static_cast<unsigned>(x);
    h *= 1099511628211ull;
  }
  return h;
}

FailureOr<ModuloScheduleResult> runRandomSearch(const DataDependenceGraph &ddg,
                                                int maxII, int smemBudget,
                                                int tmemColLimit,
                                                int numSamples,
                                                int minIIOverride) {
  const int minII = std::max(ddg.computeMinII(), minIIOverride);
  if (minII <= 0)
    return failure();
  if (maxII <= 0)
    maxII = 2 * minII;
  else if (maxII < minII)
    return failure();

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
    if (node.pipeline == HWPipeline::TMA || node.pipeline == HWPipeline::TC)
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

      if (!tryRepairModuloSchedule(II, scheduled, ddg.getNodes(),
                                   ddg.getEdges())) {
        LLVM_DEBUG(if (sample < 5) DBGS() << "  Random sample " << sample
                                          << ": dependency-invalid schedule\n");
        continue;
      }

      // Evaluate.
      auto liveness = computeLiveness(buffers, scheduled);
      auto feas =
          checkFeasibility(buffers, liveness, II, smemBudget, tmemColLimit);
      if (!feas.feasible)
        continue;

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

// ── Contracted-graph two-stage search ──────────────────────────────────────

namespace {

struct ComputeGroup {
  llvm::SmallVector<unsigned> nodes;
  llvm::SmallBitVector reachableGemms;
  int rankingLatency = 0;
};

struct ContractedGraphInfo {
  llvm::SmallVector<unsigned> gemms;
  llvm::SmallVector<int> gemmOrdinal;
  llvm::SmallVector<int> nodeToGroup;
  llvm::SmallVector<ComputeGroup> groups;
};

static bool isComputePipeline(HWPipeline pipeline) {
  return pipeline == HWPipeline::CUDA || pipeline == HWPipeline::SFU ||
         pipeline == HWPipeline::NONE;
}

static double getContractedComputeRatio() {
  auto value =
      ::mlir::triton::tools::getStrEnv("TRITON_CONTRACTED_COMPUTE_RATIO");
  if (value.empty())
    return 0.25;
  char *end = nullptr;
  double ratio = std::strtod(value.c_str(), &end);
  return end == value.c_str() || ratio <= 0.0 ? 0.25 : ratio;
}

/// Build computation groups used by the ranking objective. The original DDG
/// remains untouched and is used for all legality checks.
static ContractedGraphInfo
analyzeContractedGraph(const DataDependenceGraph &ddg,
                       llvm::ArrayRef<unsigned> topo) {
  ContractedGraphInfo info;
  const unsigned numNodes = ddg.getNumNodes();
  info.gemmOrdinal.assign(numNodes, -1);
  info.nodeToGroup.assign(numNodes, -1);
  for (const auto &node : ddg.getNodes()) {
    if (node.pipeline != HWPipeline::TC)
      continue;
    info.gemmOrdinal[node.idx] = info.gemms.size();
    info.gemms.push_back(node.idx);
  }

  llvm::SmallVector<llvm::SmallBitVector> reachable(
      numNodes, llvm::SmallBitVector(info.gemms.size()));
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    unsigned nodeIdx = *it;
    int ordinal = info.gemmOrdinal[nodeIdx];
    if (ordinal >= 0)
      reachable[nodeIdx].set(ordinal);
    for (const auto *edge : ddg.getOutEdges(nodeIdx)) {
      if (edge->distance == 0)
        reachable[nodeIdx] |= reachable[edge->dstIdx];
    }
  }

  llvm::SmallVector<unsigned> parent(numNodes);
  std::iota(parent.begin(), parent.end(), 0);
  auto findRoot = [&](unsigned node) {
    unsigned root = node;
    while (parent[root] != root)
      root = parent[root];
    while (parent[node] != node) {
      unsigned next = parent[node];
      parent[node] = root;
      node = next;
    }
    return root;
  };

  llvm::SmallVector<bool> contractible(numNodes, false);
  for (const auto &node : ddg.getNodes()) {
    if (!isComputePipeline(node.pipeline))
      continue;
    bool boundary = false;
    for (const auto *edge : ddg.getInEdges(node.idx)) {
      boundary |= edge->distance > 0;
    }
    for (const auto *edge : ddg.getOutEdges(node.idx)) {
      boundary |= edge->distance > 0;
      boundary |= edge->distance == 0 &&
                  ddg.getNode(edge->dstIdx).pipeline == HWPipeline::TC;
    }
    contractible[node.idx] = !boundary;
  }

  for (const auto &edge : ddg.getEdges()) {
    if (edge.distance > 0 || !contractible[edge.srcIdx] ||
        !contractible[edge.dstIdx] ||
        reachable[edge.srcIdx] != reachable[edge.dstIdx])
      continue;
    unsigned srcRoot = findRoot(edge.srcIdx);
    unsigned dstRoot = findRoot(edge.dstIdx);
    if (srcRoot != dstRoot)
      parent[dstRoot] = srcRoot;
  }

  llvm::DenseMap<unsigned, unsigned> rootToGroup;
  for (unsigned nodeIdx : topo) {
    if (!contractible[nodeIdx])
      continue;
    unsigned root = findRoot(nodeIdx);
    auto [it, inserted] = rootToGroup.try_emplace(root, info.groups.size());
    if (inserted) {
      ComputeGroup group;
      group.reachableGemms = reachable[nodeIdx];
      info.groups.push_back(std::move(group));
    }
    unsigned groupIdx = it->second;
    info.nodeToGroup[nodeIdx] = groupIdx;
    info.groups[groupIdx].nodes.push_back(nodeIdx);
  }

  const double ratio = getContractedComputeRatio();
  int smallestGemmLatency = INT_MAX;
  for (unsigned nodeIdx : info.gemms)
    smallestGemmLatency =
        std::min(smallestGemmLatency, ddg.getNode(nodeIdx).latency);
  for (auto &group : info.groups) {
    int rawLatency = 0;
    int nearestGemmLatency = INT_MAX;
    for (unsigned nodeIdx : group.nodes)
      rawLatency += std::max(ddg.getNode(nodeIdx).latency, 0);
    for (int gemm = group.reachableGemms.find_first(); gemm >= 0;
         gemm = group.reachableGemms.find_next(gemm)) {
      nearestGemmLatency =
          std::min(nearestGemmLatency, ddg.getNode(info.gemms[gemm]).latency);
    }
    if (nearestGemmLatency == INT_MAX)
      nearestGemmLatency = smallestGemmLatency;
    int cap = std::max(1, static_cast<int>(nearestGemmLatency * ratio));
    group.rankingLatency = std::min(rawLatency, cap);
  }
  return info;
}

static llvm::SmallVector<int>
gemmSignature(const ContractedGraphInfo &info,
              const llvm::DenseMap<unsigned, int> &scheduled, int II) {
  llvm::SmallVector<int> signature;
  signature.reserve(2 * info.gemms.size());
  llvm::SmallVector<int> moduloCycles;
  moduloCycles.reserve(info.gemms.size());
  for (unsigned nodeIdx : info.gemms)
    moduloCycles.push_back(scheduled.lookup(nodeIdx) % II);
  llvm::SmallVector<int> sortedCycles = moduloCycles;
  llvm::sort(sortedCycles);
  sortedCycles.erase(std::unique(sortedCycles.begin(), sortedCycles.end()),
                     sortedCycles.end());
  for (unsigned i = 0; i < info.gemms.size(); ++i) {
    int cycle = scheduled.lookup(info.gemms[i]);
    int cluster =
        llvm::lower_bound(sortedCycles, moduloCycles[i]) - sortedCycles.begin();
    signature.push_back(cycle / II);
    signature.push_back(cluster);
  }
  return signature;
}

static int contractedNodeLatency(const DataDependenceGraph &ddg,
                                 const ContractedGraphInfo &info,
                                 unsigned nodeIdx, int fallback) {
  if (ddg.getNode(nodeIdx).pipeline == HWPipeline::TMA)
    return 1;
  if (ddg.getNode(nodeIdx).pipeline == HWPipeline::TC)
    return std::max(ddg.getNode(nodeIdx).selfLatency, 1);
  int groupIdx = info.nodeToGroup[nodeIdx];
  if (groupIdx < 0)
    return fallback;
  const auto &group = info.groups[groupIdx];
  return std::max(1,
                  group.rankingLatency / std::max<int>(group.nodes.size(), 1));
}

static int
computeContractedEarliest(unsigned nodeIdx, const DataDependenceGraph &ddg,
                          const ContractedGraphInfo &info,
                          const llvm::DenseMap<unsigned, int> &scheduled,
                          int II) {
  int earliest = 0;
  for (const auto *edge : ddg.getInEdges(nodeIdx)) {
    auto source = scheduled.find(edge->srcIdx);
    if (source == scheduled.end())
      continue;
    int latency = contractedNodeLatency(ddg, info, edge->srcIdx, edge->latency);
    earliest = std::max(earliest, source->second + latency -
                                      static_cast<int>(edge->distance) * II);
  }
  return earliest;
}

static bool validateSchedule(const DataDependenceGraph &ddg,
                             const ContractedGraphInfo &info,
                             const llvm::DenseMap<unsigned, int> &scheduled,
                             int II) {
  for (const auto &edge : ddg.getEdges()) {
    auto src = scheduled.find(edge.srcIdx);
    auto dst = scheduled.find(edge.dstIdx);
    if (src == scheduled.end() || dst == scheduled.end())
      return false;
    int latency = edge.distance == 0
                      ? contractedNodeLatency(ddg, info, edge.srcIdx,
                                              edge.latency)
                      : 0;
    int64_t consumerCycle = static_cast<int64_t>(dst->second) +
                            static_cast<int64_t>(edge.distance) * II;
    int64_t producerCycle = static_cast<int64_t>(src->second) + latency;
    if (consumerCycle < producerCycle)
      return false;
  }
  return true;
}

} // namespace

FailureOr<ModuloScheduleResult>
runContractedSearch(const DataDependenceGraph &ddg, int maxII) {
  int minII = ddg.computeMinII();
  if (minII <= 0)
    return failure();
  if (maxII <= 0)
    maxII = 2 * minII;
  maxII = std::min(maxII, minII + std::max(10, minII / 8));

  auto topo = topologicalOrder(ddg);
  if (topo.size() != ddg.getNumNodes())
    return failure();
  auto contracted = analyzeContractedGraph(ddg, topo);
  if (contracted.gemms.size() < 2 || contracted.gemms.size() >= 63)
    return failure();

  struct Candidate {
    int II;
    int imbalance;
    int64_t contractedCost;
    llvm::SmallVector<int> signature;
    llvm::DenseMap<unsigned, int> scheduled;
  };
  llvm::SmallVector<Candidate> candidates;
  std::set<llvm::SmallVector<int>> seen;
  const uint64_t assignmentLimit = uint64_t{1} << contracted.gemms.size();
  const int K = std::max(getModuloTopK(), getModuloPick() + 1);
  llvm::SmallVector<unsigned> placementFailures(ddg.getNumNodes(), 0);
  unsigned validationFailures = 0;

  DEBUG_WITH_TYPE("modulo-scheduling-contracted", {
    llvm::dbgs() << "[modulo-scheduling-contracted]: original nodes="
                 << ddg.getNumNodes() << " GEMMs=" << contracted.gemms.size()
                 << " compute groups=" << contracted.groups.size() << "\n";
  });

  for (int II = minII; II <= maxII; ++II) {
    for (uint64_t assignment = 1; assignment + 1 < assignmentLimit;
         ++assignment) {
      ModuloReservationTable table(II);
      llvm::DenseMap<unsigned, int> scheduled;
      bool valid = true;
      for (unsigned nodeIdx : topo) {
        const auto &node = ddg.getNode(nodeIdx);
        int earliest =
            computeContractedEarliest(nodeIdx, ddg, contracted, scheduled, II);
        int duration = node.pipeline == HWPipeline::TMA
                           ? 1
                           : contractedNodeLatency(ddg, contracted, nodeIdx,
                                                   getNodeDuration(node));
        int targetStage = earliest / II;
        int ordinal = contracted.gemmOrdinal[nodeIdx];
        if (ordinal >= 0)
          targetStage = (assignment >> ordinal) & 1;
        int maxStage = ordinal >= 0 ? 1 : 2;
        if (targetStage > maxStage ||
            (ordinal >= 0 && earliest > (targetStage + 1) * II - 1)) {
          DEBUG_WITH_TYPE("modulo-scheduling-contracted", {
            if (placementFailures[nodeIdx] == 0)
              llvm::dbgs() << "[modulo-scheduling-contracted]: first reject N"
                           << nodeIdx << " II=" << II
                           << " assignment=" << assignment
                           << " targetStage=" << targetStage
                           << " earliest=" << earliest << " (stage bound)\n";
          });
          placementFailures[nodeIdx]++;
          valid = false;
          break;
        }
        int stageStart = std::max(earliest, targetStage * II);
        if (ordinal > 0 && targetStage == 1) {
          auto leading = scheduled.find(contracted.gemms.front());
          if (leading != scheduled.end())
            stageStart = std::max(stageStart, II + leading->second % II + 1);
        }
        // Keep the leading stage-0 GEMM early. Pack later stage-0 GEMMs at the
        // end of the modulo interval, leaving low modulo cycles available to
        // stage-1 consumers after the iteration boundary.
        if (ordinal > 0 && targetStage == 0) {
          int trailingOccupancy = 0;
          for (unsigned gemm = ordinal; gemm < contracted.gemms.size();
               ++gemm) {
            if (((assignment >> gemm) & 1) == 0)
              trailingOccupancy +=
                  getNodeDuration(ddg.getNode(contracted.gemms[gemm]));
          }
          stageStart = std::max(stageStart, II - trailingOccupancy);
        }
        int slot = table.findFreeSlot(stageStart, node.pipeline, duration);
        if (slot < 0 || slot / II > maxStage ||
            (ordinal >= 0 && slot / II != targetStage)) {
          DEBUG_WITH_TYPE("modulo-scheduling-contracted", {
            if (placementFailures[nodeIdx] == 0)
              llvm::dbgs() << "[modulo-scheduling-contracted]: first reject N"
                           << nodeIdx << " II=" << II
                           << " assignment=" << assignment
                           << " targetStage=" << targetStage
                           << " earliest=" << earliest << " slot=" << slot
                           << " (resource)\n";
          });
          placementFailures[nodeIdx]++;
          valid = false;
          break;
        }
        table.reserve(slot, node.pipeline, nodeIdx, duration);
        scheduled[nodeIdx] = slot;
      }
      if (!valid)
        continue;
      if (!validateSchedule(ddg, contracted, scheduled, II)) {
        validationFailures++;
        continue;
      }

      auto signature = gemmSignature(contracted, scheduled, II);
      if (!seen.insert(signature).second)
        continue;

      int stageOne = llvm::popcount(assignment);
      int imbalance =
          std::abs(static_cast<int>(contracted.gemms.size()) - 2 * stageOne);
      int64_t contractedCost = 0;
      for (const auto &group : contracted.groups) {
        if (group.reachableGemms.none())
          continue;
        int minStage = 1;
        int maxStage = 0;
        for (int gemm = group.reachableGemms.find_first(); gemm >= 0;
             gemm = group.reachableGemms.find_next(gemm)) {
          int stage = (assignment >> gemm) & 1;
          minStage = std::min(minStage, stage);
          maxStage = std::max(maxStage, stage);
        }
        contractedCost +=
            static_cast<int64_t>(group.rankingLatency) * (maxStage - minStage);
      }
      candidates.push_back({II, imbalance, contractedCost,
                            std::move(signature),
                            std::move(scheduled)});
    }
  }

  if (candidates.empty())
    DEBUG_WITH_TYPE("modulo-scheduling-contracted", {
      llvm::dbgs() << "[modulo-scheduling-contracted]: no candidates; "
                   << "validation failures=" << validationFailures
                   << " placement failures:";
      for (unsigned nodeIdx = 0; nodeIdx < placementFailures.size(); ++nodeIdx)
        if (placementFailures[nodeIdx])
          llvm::dbgs() << " N" << nodeIdx << "=" << placementFailures[nodeIdx];
      llvm::dbgs() << "\n";
    });
  if (candidates.empty())
    return failure();
  llvm::stable_sort(candidates, [](const Candidate &lhs, const Candidate &rhs) {
    if (lhs.II != rhs.II)
      return lhs.II < rhs.II;
    if (lhs.imbalance != rhs.imbalance)
      return lhs.imbalance < rhs.imbalance;
    if (lhs.contractedCost != rhs.contractedCost)
      return lhs.contractedCost < rhs.contractedCost;
    return std::lexicographical_compare(
        lhs.signature.begin(), lhs.signature.end(), rhs.signature.begin(),
        rhs.signature.end());
  });

  int nTop = std::min<int>(K, candidates.size());
  int pick = std::min(getModuloPick(), nTop - 1);
  DEBUG_WITH_TYPE("modulo-scheduling-contracted", {
    llvm::dbgs() << "[modulo-scheduling-contracted]: top-" << nTop
                 << " applying pick " << pick << " from " << candidates.size()
                 << " schedules\n";
    for (int rank = 0; rank < nTop; ++rank) {
      const auto &candidate = candidates[rank];
      llvm::dbgs() << "  rank " << rank << " II=" << candidate.II
                   << " imbalance=" << candidate.imbalance
                   << " computeCost=" << candidate.contractedCost
                   << " gemms(stage,cluster)=[";
      for (unsigned i = 0; i < contracted.gemms.size(); ++i)
        llvm::dbgs() << (i ? "," : "") << "(" << candidate.signature[2 * i]
                     << "," << candidate.signature[2 * i + 1] << ")";
      llvm::dbgs() << "]\n";
    }
  });

  ModuloScheduleResult result;
  result.II = candidates[pick].II;
  result.nodeToCycle = std::move(candidates[pick].scheduled);
  return result;
}

} // namespace mlir::triton::gpu
