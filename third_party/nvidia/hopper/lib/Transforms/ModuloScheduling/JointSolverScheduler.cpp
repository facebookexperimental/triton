// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// See JointSolverScheduler.h for the native joint-solver-0.1 contract.

#include "JointSolverScheduler.h"

#include "Z3JointSolutionValidator.h"
#include "Z3JointSolver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
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

struct SolverBuffer {
  unsigned allocNodeIdx;
  bool isTmem;
  int64_t sizeBytes;
  int64_t tmemCols;
  llvm::SmallVector<unsigned, 4> consumerNodes;
};

static llvm::SmallVector<SolverBuffer>
extractSolverBuffers(const DataDependenceGraph &ddg) {
  llvm::SmallVector<SolverBuffer> buffers;
  for (const auto &node : ddg.getNodes()) {
    Operation *op = node.op;
    SolverBuffer buffer;
    buffer.allocNodeIdx = node.idx;
    if (isa<LocalAllocOp>(op)) {
      auto memDesc = dyn_cast<MemDescType>(op->getResult(0).getType());
      if (!memDesc)
        continue;
      int64_t elements = 1;
      for (int64_t dimension : memDesc.getShape())
        elements *= dimension;
      buffer.isTmem = false;
      buffer.sizeBytes =
          elements * memDesc.getElementType().getIntOrFloatBitWidth() / 8;
      buffer.tmemCols = 0;
    } else if (node.tmemAllocCols > 0) {
      buffer.isTmem = true;
      buffer.sizeBytes = 0;
      buffer.tmemCols = node.tmemAllocCols;
    } else {
      continue;
    }
    for (const DDGEdge *edge : ddg.getOutEdges(node.idx))
      if (edge->distance == 0)
        buffer.consumerNodes.push_back(edge->dstIdx);
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

static std::string
buildProblemJSON(const DataDependenceGraph &ddg, int minII, int maxII,
                 int smemBudget, int tmemColLimit,
                 llvm::ArrayRef<llvm::StringRef> relaxKeepKinds = {}) {
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
  for (const auto &buf : extractSolverBuffers(ddg)) {
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
  if (!relaxKeepKinds.empty()) {
    llvm::json::Array keptKinds;
    for (llvm::StringRef kind : relaxKeepKinds)
      keptKinds.push_back(kind.str());
    root["relax_keep_kinds"] = std::move(keptKinds);
  }
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

// The one process-global on the scheduling path, and deliberately so: it is a
// memo keyed by the canonicalized problem, not configuration. Two modules
// compiled concurrently (each pass pipeline runs with the GIL released) can
// only ever hand each other the answer to an identical problem, and the
// entries are mutex-guarded.
static JointSolverCache &jointSolverCache() {
  static JointSolverCache cache;
  return cache;
}

constexpr unsigned kDefaultRetryCount = 1;
constexpr unsigned kMaxRetryCount = 3;
constexpr double kMaxRetryTimeoutSeconds = 300.0;

enum class ResponseKind { Ok, ProvenUnsat, Inconclusive, Other };

static ResponseKind classifyResponse(llvm::StringRef response) {
  auto parsed = llvm::json::parse(response);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return ResponseKind::Other;
  }
  const auto *object = parsed->getAsObject();
  if (!object)
    return ResponseKind::Other;
  auto status = object->getString("status");
  if (!status)
    return ResponseKind::Other;
  if (*status == "ok")
    return ResponseKind::Ok;
  if (*status == "infeasible" &&
      object->getBoolean("proven_unsat").value_or(false))
    return ResponseKind::ProvenUnsat;
  if (*status == "inconclusive")
    return ResponseKind::Inconclusive;
  return ResponseKind::Other;
}

static bool validateSolution(llvm::StringRef problem,
                             llvm::StringRef response) {
  std::string validation = validateZ3JointSolution(problem, response);
  auto parsed = llvm::json::parse(validation);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    LLVM_DEBUG(DBGS() << "validator returned malformed JSON\n");
    return false;
  }
  const auto *object = parsed->getAsObject();
  bool valid = object && object->getBoolean("valid").value_or(false);
  if (!valid)
    LLVM_DEBUG(DBGS() << "solution rejected by validator: " << validation
                      << "\n");
  return valid;
}

static unsigned retryCount() {
  auto env = tools::getStrEnv("TRITON_MODULO_JOINT_SOLVER_RETRIES");
  if (env.empty())
    return kDefaultRetryCount;
  char *end = nullptr;
  long parsed = std::strtol(env.c_str(), &end, 10);
  if (end == env.c_str() || *end != '\0')
    return kDefaultRetryCount;
  return static_cast<unsigned>(std::clamp<long>(parsed, 0, kMaxRetryCount));
}

static FailureOr<std::string>
problemForAttempt(llvm::StringRef canonicalProblem, unsigned attempt) {
  if (attempt == 0)
    return canonicalProblem.str();

  auto parsed = llvm::json::parse(canonicalProblem);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  auto *object = parsed->getAsObject();
  if (!object)
    return failure();
  auto baseTimeout = object->getNumber("time_limit_s");
  if (!baseTimeout || !std::isfinite(*baseTimeout) || *baseTimeout <= 0.0)
    return failure();
  double multiplier = std::ldexp(1.0, static_cast<int>(attempt));
  (*object)["time_limit_s"] =
      std::min(*baseTimeout * multiplier, kMaxRetryTimeoutSeconds);

  std::string retryProblem;
  llvm::raw_string_ostream os(retryProblem);
  writeCanonicalJSON(os, *parsed);
  return retryProblem;
}

} // namespace

FailureOr<std::string> runJointSolverBackend(llvm::StringRef problemJson) {
  auto canonicalProblem = canonicalizeJSON(problemJson);
  if (failed(canonicalProblem))
    return failure();
  if (auto cached = jointSolverCache().lookup(*canonicalProblem))
    return std::move(*cached);

  unsigned retries = retryCount();
  for (unsigned attempt = 0; attempt <= retries; ++attempt) {
    auto attemptProblem = problemForAttempt(*canonicalProblem, attempt);
    if (failed(attemptProblem))
      return failure();
    auto response = runZ3JointSolver(*attemptProblem);
    if (failed(response))
      return failure();

    ResponseKind kind = classifyResponse(*response);
    if (kind == ResponseKind::Ok) {
      if (!validateSolution(*attemptProblem, *response))
        return failure();
      jointSolverCache().insert(*canonicalProblem, *response);
      return response;
    }
    if (kind == ResponseKind::ProvenUnsat) {
      jointSolverCache().insert(*canonicalProblem, *response);
      return response;
    }
    if (kind != ResponseKind::Inconclusive || attempt == retries)
      return response;
    LLVM_DEBUG(DBGS() << "inconclusive response; retry " << attempt + 1 << '/'
                      << retries << " with a fresh backend\n");
  }
  llvm_unreachable("joint-solver retry loop must return");
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

/// Exclusive modular reservation only — the half of `verifySolution` that a
/// resource-relaxed schedule is still required to satisfy. Dependences are
/// deliberately not checked: dropping them is the point of the relaxation.
static bool
verifyResourceExclusivity(const DataDependenceGraph &ddg, int ii,
                          const llvm::DenseMap<unsigned, int> &cycles) {
  if (ii <= 0)
    return false;
  ModuloReservationTable table(ii);
  for (const auto &node : ddg.getNodes()) {
    auto cycleIt = cycles.find(node.idx);
    if (cycleIt == cycles.end() || cycleIt->second < 0)
      return false;
    if (node.pipeline == HWPipeline::NONE)
      continue;
    int dur = nodeDuration(node);
    if (dur > ii || !table.isIntervalFree(cycleIt->second, node.pipeline, dur))
      return false;
    table.reserve(cycleIt->second, node.pipeline, node.idx, dur);
  }
  return true;
}

FailureOr<int> runJointSolverRelaxedLowerBound(const DataDependenceGraph &ddg,
                                               int minII, int smemBudget,
                                               int tmemColLimit) {
  if (minII <= 0 || ddg.getNumNodes() == 0)
    return failure();
  int maxII = serialUpperBound(ddg, minII);

  // Deliberately NOT routed through runJointSolverBackend: that wrapper
  // validates every "ok" response against the full model, and a relaxed
  // schedule violates the very constraints it dropped, so it would always be
  // rejected. The relaxed run is not trusted either — the kept kind is
  // re-verified below — but it has to be checked against the relaxed model,
  // not the full one. Skipping the wrapper also skips its cache; the bound is
  // only computed under TRITON_MODULO_BASELINE_REPORT, so that is not a hot
  // path.
  llvm::StringRef keep[] = {llvm::StringRef("resource")};
  auto rawOut = runZ3JointSolver(
      buildProblemJSON(ddg, minII, maxII, smemBudget, tmemColLimit, keep));
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
    LLVM_DEBUG(DBGS() << "relaxed lower bound unavailable: status "
                      << (status ? *status : "<none>") << "\n");
    return failure();
  }
  // The backend must confirm it honoured the relaxation. Without this, a build
  // whose solver silently ignored `relax_keep_kinds` would return the FULL
  // model's II here, making the soundness invariant compare II_full against
  // itself and pass vacuously.
  if (!obj->getObject("relaxation")) {
    LLVM_DEBUG(DBGS() << "backend did not honour relax_keep_kinds\n");
    return failure();
  }

  auto ii = obj->getInteger("ii");
  auto *cycles = obj->getObject("cycles");
  if (!ii || !cycles || *ii <= 0)
    return failure();

  llvm::DenseMap<unsigned, int> nodeToCycle;
  for (const auto &kv : *cycles) {
    unsigned idx = 0;
    if (llvm::StringRef(kv.first).getAsInteger(10, idx))
      return failure();
    auto cyc = kv.second.getAsInteger();
    if (!cyc)
      return failure();
    nodeToCycle[idx] = static_cast<int>(*cyc);
  }
  if (nodeToCycle.size() != ddg.getNumNodes())
    return failure();

  // A relaxed bound that is too HIGH turns the soundness invariant into a
  // false alarm, so the kept kind is re-verified before the number is used.
  if (!verifyResourceExclusivity(ddg, static_cast<int>(*ii), nodeToCycle)) {
    LLVM_DEBUG(DBGS() << "relaxed schedule violates the kept resource "
                         "constraints — discarding the bound\n");
    return failure();
  }
  LLVM_DEBUG(DBGS() << "relaxed lower bound = " << *ii << "\n");
  return static_cast<int>(*ii);
}

} // namespace mlir::triton::gpu
