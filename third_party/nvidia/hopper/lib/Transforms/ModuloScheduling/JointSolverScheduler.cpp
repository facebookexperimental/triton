// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// See JointSolverScheduler.h for the native joint-solver-0.1 contract.

#include "JointSolverScheduler.h"

#include "ExhaustiveScheduler.h"
#include "Z3JointSolver.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#define DEBUG_TYPE "modulo-scheduling-joint-solver"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::triton::gpu {

/// Reservation duration — must match runRauIMS/ExhaustiveScheduler exactly:
/// warp-issue selfLatency slots (NOT the engine occupancy that feeds ResMII;
/// minII already encodes that floor).
static int nodeDuration(const DDGNode &node) {
  if (node.pipeline == HWPipeline::NONE)
    return 1;
  return std::max(node.selfLatency, 1);
}

/// True feasibility upper bound for the II sweep: at II = critical path +
/// total serial work, every op fits in one stage back-to-back, so a schedule
/// always exists. This replaces the heuristic slack window (guard 2) — a
/// complete search proves each II infeasible instead of failing to pack it.
static int serialUpperBound(const DataDependenceGraph &ddg, int minII) {
  auto heights = ddg.computeCriticalPathHeights();
  int criticalPath = 0;
  for (auto &[_, h] : heights)
    criticalPath = std::max(criticalPath, h);
  int64_t serialWork = 0;
  for (const auto &node : ddg.getNodes())
    serialWork += nodeDuration(node);
  int64_t bound = criticalPath + serialWork;
  return static_cast<int>(std::max<int64_t>(minII, bound));
}

static std::string buildProblemJSON(const DataDependenceGraph &ddg, int minII,
                                    int maxII, int smemBudget,
                                    int tmemColLimit) {
  // Streaming classification (Twill §5.3): a variable-latency op with no
  // incoming data dependence (TMA input loads) runs ahead of the pipeline
  // behind its ring buffer, so in steady state its consumers do not wait its
  // latency — the solver models its outgoing edges as latency 0 when
  // TRITON_MODULO_STREAMING_VL=1. Ring depth stays a solver decision (the
  // objective already rewards depth against the SMEM budget).
  llvm::DenseSet<unsigned> hasIncoming;
  for (const auto &edge : ddg.getEdges())
    if (edge.distance == 0)
      hasIncoming.insert(edge.dstIdx);
  llvm::json::Array nodes;
  for (const auto &node : ddg.getNodes()) {
    bool streaming =
        node.pipeline == HWPipeline::TMA && !hasIncoming.contains(node.idx);
    nodes.push_back(llvm::json::Object{
        {"id", static_cast<int64_t>(node.idx)},
        {"pipeline", getPipelineName(node.pipeline)},
        {"duration", nodeDuration(node)},
        {"streaming", streaming},
    });
  }
  llvm::json::Array edges;
  for (const auto &edge : ddg.getEdges()) {
    edges.push_back(llvm::json::Object{
        {"src", static_cast<int64_t>(edge.srcIdx)},
        {"dst", static_cast<int64_t>(edge.dstIdx)},
        {"latency", edge.latency},
        {"distance", static_cast<int64_t>(edge.distance)},
    });
  }
  llvm::json::Array buffers;
  for (const auto &buf : extractSchedBuffers(ddg)) {
    llvm::json::Array consumers;
    for (unsigned c : buf.consumerNodes)
      consumers.push_back(static_cast<int64_t>(c));
    buffers.push_back(llvm::json::Object{
        {"alloc_node", static_cast<int64_t>(buf.allocNodeIdx)},
        {"kind", buf.isTmem ? "tmem" : "smem"},
        {"size_bytes", buf.sizeBytes},
        {"tmem_cols", buf.tmemCols},
        {"consumers", std::move(consumers)},
    });
  }

  // The wall clock is only a backstop against a hang; the deterministic
  // rlimit budget in the solver is what is meant to stop the search, so this
  // is set well above the time that budget normally takes to run out.
  double timeLimitS = 180.0;
  if (auto env = tools::getStrEnv("TRITON_MODULO_JOINT_SOLVER_TIMEOUT_S");
      !env.empty())
    timeLimitS = std::max(1.0, std::atof(env.c_str()));

  bool streamingVL =
      tools::getBoolEnv("TRITON_MODULO_STREAMING_VL"); // default off

  llvm::json::Object root{
      {"version", "joint-solver-0.1"},
      {"min_ii", minII},
      {"max_ii", maxII},
      {"smem_budget", smemBudget},
      {"tmem_col_limit", tmemColLimit},
      {"time_limit_s", timeLimitS},
      {"streaming_vl", streamingVL},
      {"nodes", std::move(nodes)},
      {"edges", std::move(edges)},
      {"buffers", std::move(buffers)},
  };
  // Escape hatches for the deterministic budget, mirroring the timeout knob
  // above. Both travel in the request, so they land in the solver cache key
  // automatically and cannot silently return an answer found under a
  // different budget.
  if (auto env = tools::getStrEnv("TRITON_MODULO_JOINT_SOLVER_RLIMIT");
      !env.empty())
    root["rlimit"] = std::max<int64_t>(0, std::atoll(env.c_str()));
  if (auto env = tools::getStrEnv("TRITON_MODULO_JOINT_SOLVER_SEED");
      !env.empty())
    root["random_seed"] = std::max<int64_t>(0, std::atoll(env.c_str()));
  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::json::Value(std::move(root));
  return out;
}

/// Re-verify the backend's schedule against the same constraints the
/// in-process schedulers enforce: dependences and exclusive modular
/// reservation. A schedule that fails here is discarded (fall back to the
/// heuristics); the backend is advisory, never trusted.
/// Under TRITON_MODULO_STREAMING_VL the solver legitimately places a
/// streaming producer's consumers inside its raw latency (the ring absorbs
/// it), so verification uses the same effective latency-0 rule for those
/// edges — otherwise every streaming schedule would be rejected here.
static bool verifySolution(const DataDependenceGraph &ddg,
                           const ModuloScheduleResult &res) {
  if (res.II <= 0)
    return false;
  llvm::DenseSet<unsigned> streaming;
  if (tools::getBoolEnv("TRITON_MODULO_STREAMING_VL")) {
    llvm::DenseSet<unsigned> hasIncoming;
    for (const auto &edge : ddg.getEdges())
      if (edge.distance == 0)
        hasIncoming.insert(edge.dstIdx);
    for (const auto &node : ddg.getNodes())
      if (node.pipeline == HWPipeline::TMA && !hasIncoming.contains(node.idx))
        streaming.insert(node.idx);
  }
  for (const auto &edge : ddg.getEdges()) {
    auto s = res.nodeToCycle.find(edge.srcIdx);
    auto d = res.nodeToCycle.find(edge.dstIdx);
    if (s == res.nodeToCycle.end() || d == res.nodeToCycle.end())
      return false;
    int lat = streaming.contains(edge.srcIdx) ? 0 : edge.latency;
    if (d->second <
        s->second + lat - static_cast<int>(edge.distance) * res.II) {
      LLVM_DEBUG(DBGS() << "verify: dependence violated N" << edge.srcIdx
                        << " -> N" << edge.dstIdx << "\n");
      return false;
    }
  }
  ModuloReservationTable table(res.II);
  for (const auto &node : ddg.getNodes()) {
    auto cycleIt = res.nodeToCycle.find(node.idx);
    if (cycleIt == res.nodeToCycle.end() || cycleIt->second < 0)
      return false;
    if (node.pipeline == HWPipeline::NONE)
      continue;
    int dur = nodeDuration(node);
    int cycle = cycleIt->second;
    if (dur > res.II)
      return false;
    if (!table.isIntervalFree(cycle, node.pipeline, dur)) {
      LLVM_DEBUG(DBGS() << "verify: reservation conflict at N" << node.idx
                        << "\n");
      return false;
    }
    table.reserve(cycle, node.pipeline, node.idx, dur);
  }
  return true;
}

namespace {

constexpr size_t kJointSolverCacheCapacity = 64;

static void writeCanonicalJSON(llvm::raw_ostream &os,
                               const llvm::json::Value &value) {
  if (const auto *object = value.getAsObject()) {
    llvm::SmallVector<llvm::StringRef> keys;
    keys.reserve(object->size());
    for (const auto &entry : *object)
      keys.push_back(entry.first);
    llvm::sort(keys);

    os << '{';
    for (auto [index, key] : llvm::enumerate(keys)) {
      if (index != 0)
        os << ',';
      os << llvm::json::Value(key.str()) << ':';
      writeCanonicalJSON(os, *object->get(key));
    }
    os << '}';
    return;
  }
  if (const auto *array = value.getAsArray()) {
    os << '[';
    for (auto [index, element] : llvm::enumerate(*array)) {
      if (index != 0)
        os << ',';
      writeCanonicalJSON(os, element);
    }
    os << ']';
    return;
  }
  os << value;
}

static FailureOr<std::string> canonicalizeJSON(llvm::StringRef json) {
  auto parsed = llvm::json::parse(json);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  std::string canonical;
  llvm::raw_string_ostream os(canonical);
  writeCanonicalJSON(os, *parsed);
  return canonical;
}

class JointSolverCache {
public:
  std::optional<std::string> lookup(llvm::StringRef key) {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = entries.find(key.str());
    if (found == entries.end())
      return std::nullopt;
    lru.splice(lru.begin(), lru, found->second);
    return found->second->second;
  }

  void insert(std::string key, std::string response) {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = entries.find(key);
    if (found != entries.end()) {
      found->second->second = std::move(response);
      lru.splice(lru.begin(), lru, found->second);
      return;
    }
    lru.emplace_front(std::move(key), std::move(response));
    entries.emplace(lru.front().first, lru.begin());
    if (entries.size() <= kJointSolverCacheCapacity)
      return;
    entries.erase(lru.back().first);
    lru.pop_back();
  }

private:
  using EntryList = std::list<std::pair<std::string, std::string>>;
  std::mutex mutex;
  EntryList lru;
  std::unordered_map<std::string, EntryList::iterator> entries;
};

static JointSolverCache &jointSolverCache() {
  static JointSolverCache cache;
  return cache;
}

static bool isCacheableResponse(llvm::StringRef response) {
  auto parsed = llvm::json::parse(response);
  if (parsed) {
    const auto *object = parsed->getAsObject();
    if (object == nullptr)
      return false;
    auto status = object->getString("status");
    if (status) {
      if (*status == "ok")
        return true;
      if (*status == "infeasible")
        return object->getBoolean("proven_unsat").value_or(false);
    }
    return false;
  }
  llvm::consumeError(parsed.takeError());
  return false;
}

} // namespace

FailureOr<std::string> runJointSolverBackend(llvm::StringRef problemJson) {
  auto canonicalProblem = canonicalizeJSON(problemJson);
  if (failed(canonicalProblem))
    return failure();
  if (auto cached = jointSolverCache().lookup(*canonicalProblem))
    return std::move(*cached);

  auto response = runZ3JointSolver(*canonicalProblem);
  if (failed(response))
    return failure();
  if (isCacheableResponse(*response))
    jointSolverCache().insert(*canonicalProblem, *response);
  return response;
}

FailureOr<ModuloScheduleResult>
runJointSolverSchedule(const DataDependenceGraph &ddg, int minII,
                       int smemBudget, int tmemColLimit) {
  if (minII <= 0 || ddg.getNumNodes() == 0)
    return failure();
  int maxII = serialUpperBound(ddg, minII);
  LLVM_DEBUG(DBGS() << "minII=" << minII << " maxII(serial bound)=" << maxII
                    << " nodes=" << ddg.getNumNodes() << "\n");

  auto rawOut = runJointSolverBackend(
      buildProblemJSON(ddg, minII, maxII, smemBudget, tmemColLimit));
  if (failed(rawOut))
    return failure();
  auto parsed = llvm::json::parse(*rawOut);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  auto *obj = parsed->getAsObject();
  if (!obj)
    return failure();
  auto status = obj->getString("status");
  if (!status || *status != "ok") {
    LLVM_DEBUG({
      auto msg = obj->getString("message");
      DBGS() << "solver status: " << (status ? *status : "<none>") << " "
             << (msg ? *msg : "") << "\n";
    });
    return failure();
  }

  ModuloScheduleResult result;
  auto ii = obj->getInteger("ii");
  auto *cycles = obj->getObject("cycles");
  if (!ii || !cycles)
    return failure();
  result.II = static_cast<int>(*ii);
  for (const auto &kv : *cycles) {
    unsigned idx = 0;
    if (llvm::StringRef(kv.first).getAsInteger(10, idx))
      return failure();
    auto cyc = kv.second.getAsInteger();
    if (!cyc)
      return failure();
    result.nodeToCycle[idx] = static_cast<int>(*cyc);
  }
  if (result.nodeToCycle.size() != ddg.getNumNodes())
    return failure();
  for (const auto &node : ddg.getNodes())
    if (!result.nodeToCycle.contains(node.idx))
      return failure();

  // TRITON_MODULO_SCHED_SHIFT=k (debug): rigidly translate the solution by
  // +k cycles before verification. A modulo schedule is model-equivalent
  // under translation (dependences and modular reservations are invariant),
  // but the stage split (cycle / II) is NOT — this knob deterministically
  // samples the emitter-facing stage structures that solver nondeterminism
  // otherwise draws at random, for hunting shape-dependent emitter bugs
  // (case4 flake, 2026-07-10).
  if (auto env = tools::getStrEnv("TRITON_MODULO_SCHED_SHIFT"); !env.empty()) {
    int shift = std::atoi(env.c_str());
    if (shift > 0) {
      LLVM_DEBUG(DBGS() << "shifting schedule by +" << shift << " cycles\n");
      for (auto &kv : result.nodeToCycle)
        kv.second += shift;
    }
  }

  if (!verifySolution(ddg, result)) {
    LLVM_DEBUG(DBGS() << "solution failed re-verification — discarding\n");
    return failure();
  }
  LLVM_DEBUG(DBGS() << "SUCCESS at II=" << result.II << "\n");
  return result;
}

} // namespace mlir::triton::gpu
