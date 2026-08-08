// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "Z3JointSolver.h"
#include "Z3ConstraintEncoder.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::triton::gpu {

#ifndef TRITON_ENABLE_Z3_JOINT_SOLVER

FailureOr<std::string> runZ3JointSolver(llvm::StringRef problemJson) {
  (void)problemJson;
  return failure();
}

#else

namespace {

static bool getRequiredInteger(const llvm::json::Object &object,
                               llvm::StringRef key, int64_t &value) {
  auto parsed = object.getInteger(key);
  if (!parsed)
    return false;
  value = *parsed;
  return true;
}

static bool getRequiredBoolean(const llvm::json::Object &object,
                               llvm::StringRef key, bool &value) {
  auto parsed = object.getBoolean(key);
  if (!parsed)
    return false;
  value = *parsed;
  return true;
}

static bool getRequiredString(const llvm::json::Object &object,
                              llvm::StringRef key, std::string &value) {
  auto parsed = object.getString(key);
  if (!parsed)
    return false;
  value = parsed->str();
  return true;
}

static thread_local bool z3ErrorSeen = false;

static void recordZ3Error(Z3_context, Z3_error_code) { z3ErrorSeen = true; }

class Context {
public:
  Context() {
    Z3_config config = Z3_mk_config();
    context = Z3_mk_context(config);
    Z3_del_config(config);
    if (context)
      Z3_set_error_handler(context, recordZ3Error);
  }

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  ~Context() {
    if (context)
      Z3_del_context(context);
  }

  Z3_context get() const { return context; }

private:
  Z3_context context = nullptr;
};

class Solver {
public:
  explicit Solver(Z3_context context) : context(context) {
    solver = Z3_mk_solver(context);
    if (solver)
      Z3_solver_inc_ref(context, solver);
  }

  Solver(const Solver &) = delete;
  Solver &operator=(const Solver &) = delete;

  ~Solver() {
    if (solver)
      Z3_solver_dec_ref(context, solver);
  }

  Z3_solver get() const { return solver; }

private:
  Z3_context context;
  Z3_solver solver = nullptr;
};

class Optimize {
public:
  explicit Optimize(Z3_context context) : context(context) {
    optimize = Z3_mk_optimize(context);
    if (optimize)
      Z3_optimize_inc_ref(context, optimize);
  }

  Optimize(const Optimize &) = delete;
  Optimize &operator=(const Optimize &) = delete;

  ~Optimize() {
    if (optimize)
      Z3_optimize_dec_ref(context, optimize);
  }

  Z3_optimize get() const { return optimize; }

private:
  Z3_context context;
  Z3_optimize optimize = nullptr;
};

class Model {
public:
  Model(Z3_context context, Z3_model model) : context(context), model(model) {
    if (model)
      Z3_model_inc_ref(context, model);
  }

  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  ~Model() {
    if (model)
      Z3_model_dec_ref(context, model);
  }

  Z3_model get() const { return model; }

private:
  Z3_context context;
  Z3_model model;
};

enum class CandidateStatus { Sat, Unsat, Unknown, Error };

static std::string serialize(llvm::json::Object object) {
  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << llvm::json::Value(std::move(object));
  stream.flush();
  return output;
}

/// Deterministic ceiling on search effort, in Z3 resource units. Resource
/// units count internal work, not time, so the same request stops at the same
/// point on a fast desktop and on a loaded CI worker — this is the budget that
/// is meant to bind in normal operation. The wall-clock deadline remains as a
/// backstop so a solve can never hang the compiler, set generously enough that
/// it only fires on pathological inputs.
///
/// Z3's work-unit accounting is internal and is not stable across releases, so
/// this number is calibrated against one vendored version and must be
/// re-measured when third-party/z3 is upgraded. docs/SolverDeterminism.md
/// records the measurements it is derived from.
constexpr unsigned kDefaultRLimit = 10000000;

/// Z3 already defaults smt.random_seed and sat.random_seed to 0, so pinning
/// does not change the search. It defends the pin against an ambient override:
/// gparams and the `Z3` environment variable are process-wide and are read
/// when a context is built. Applied per solver object rather than through
/// Z3_global_param_set, which would be a data race if the compiler ever solves
/// on more than one thread.
constexpr unsigned kDefaultRandomSeed = 0;

/// Which budget ended a search. Only meaningful for an inconclusive result.
enum class BudgetExhausted { None, RLimit, WallTime };

static llvm::StringRef budgetExhaustedName(BudgetExhausted cause) {
  switch (cause) {
  case BudgetExhausted::RLimit:
    return "rlimit";
  case BudgetExhausted::WallTime:
    return "walltime";
  case BudgetExhausted::None:
    break;
  }
  return "none";
}

struct SearchBudget {
  double timeLimitS = 180.0;
  /// 0 disables the deterministic budget, leaving only the wall-clock backstop.
  unsigned rlimit = kDefaultRLimit;
  unsigned randomSeed = kDefaultRandomSeed;
};

static bool parseUnsignedField(const llvm::json::Object &root,
                               llvm::StringRef key, unsigned &value) {
  if (!root.get(key))
    return true;
  auto parsed = root.getInteger(key);
  if (!parsed || *parsed < 0 ||
      *parsed > static_cast<int64_t>(std::numeric_limits<unsigned>::max()))
    return false;
  value = static_cast<unsigned>(*parsed);
  return true;
}

/// Parse the budget fields shared by joint-solver-0.1 and joint-solver-0.2.
/// All three are optional; a malformed value rejects the whole request rather
/// than silently falling back to a default, so a typo cannot quietly change
/// which answers are reachable.
static bool parseSearchBudget(const llvm::json::Object &root,
                              SearchBudget &budget) {
  if (root.get("time_limit_s")) {
    auto value = root.getNumber("time_limit_s");
    if (!value || !std::isfinite(*value) || *value <= 0.0)
      return false;
    budget.timeLimitS = *value;
  }
  return parseUnsignedField(root, "rlimit", budget.rlimit) &&
         parseUnsignedField(root, "random_seed", budget.randomSeed);
}

/// Build the parameter set installed before a check. `withSeed` is false for
/// Z3_optimize: opt_params carries no random seed in 4.15.2, and both
/// Z3_solver_set_params and Z3_optimize_set_params reject an unknown parameter
/// name with Z3_EXCEPTION rather than ignoring it.
static Z3_params makeSearchParams(Z3_context context, unsigned timeoutMs,
                                  const SearchBudget &budget, bool withSeed) {
  Z3_params params = Z3_mk_params(context);
  if (!params)
    return nullptr;
  Z3_params_inc_ref(context, params);
  Z3_params_set_uint(context, params, Z3_mk_string_symbol(context, "timeout"),
                     timeoutMs);
  if (budget.rlimit > 0)
    Z3_params_set_uint(context, params, Z3_mk_string_symbol(context, "rlimit"),
                       budget.rlimit);
  if (withSeed)
    Z3_params_set_uint(context, params,
                       Z3_mk_string_symbol(context, "random_seed"),
                       budget.randomSeed);
  return params;
}

static bool configureSolver(Z3_context context, Z3_solver solver,
                            unsigned timeoutMs, const SearchBudget &budget) {
  Z3_params params =
      makeSearchParams(context, timeoutMs, budget, /*withSeed=*/true);
  if (!params)
    return false;
  Z3_solver_set_params(context, solver, params);
  Z3_params_dec_ref(context, params);
  return !z3ErrorSeen;
}

static bool configureOptimize(Z3_context context, Z3_optimize optimize,
                              unsigned timeoutMs, const SearchBudget &budget) {
  Z3_params params =
      makeSearchParams(context, timeoutMs, budget, /*withSeed=*/false);
  if (!params)
    return false;
  Z3_optimize_set_params(context, optimize, params);
  Z3_params_dec_ref(context, params);
  return !z3ErrorSeen;
}

/// Resource units consumed so far on this context. Z3 publishes the counter
/// only through solver statistics, but the counter itself lives on the
/// context's reslimit and is shared by every Z3_solver and Z3_optimize built
/// on that context — so the v0.2 path reads it through the solver it already
/// uses to accumulate assertions.
static std::optional<uint64_t> readRLimitCount(Z3_context context,
                                               Z3_solver solver) {
  Z3_stats stats = Z3_solver_get_statistics(context, solver);
  if (!stats || z3ErrorSeen)
    return std::nullopt;
  Z3_stats_inc_ref(context, stats);
  std::optional<uint64_t> consumed;
  unsigned entries = Z3_stats_size(context, stats);
  for (unsigned index = 0; index < entries; ++index) {
    const char *key = Z3_stats_get_key(context, stats, index);
    if (!key || llvm::StringRef(key) != "rlimit count")
      continue;
    // The counter is published as a uint until it exceeds UINT_MAX, and as a
    // double past that.
    if (Z3_stats_is_uint(context, stats, index))
      consumed = Z3_stats_get_uint_value(context, stats, index);
    else if (Z3_stats_is_double(context, stats, index))
      consumed = static_cast<uint64_t>(
          Z3_stats_get_double_value(context, stats, index));
    break;
  }
  Z3_stats_dec_ref(context, stats);
  return consumed;
}

static bool deadlineExpired(std::chrono::steady_clock::time_point deadline) {
  return std::chrono::steady_clock::now() >= deadline;
}

/// Wall-clock milliseconds left before the backstop deadline, rounded *up*.
/// Rounding up is what lets attributeBudgetStop tell the two budgets apart:
/// Z3's timer is then guaranteed to fire at or after the deadline, so a stop
/// seen before the deadline is never the wall clock's.
static std::optional<unsigned>
remainingTimeoutMs(std::chrono::steady_clock::time_point deadline) {
  auto now = std::chrono::steady_clock::now();
  if (now >= deadline)
    return std::nullopt;
  auto remaining =
      std::chrono::ceil<std::chrono::milliseconds>(deadline - now).count();
  if (remaining <= 0)
    remaining = 1;
  return static_cast<unsigned>(
      std::min<int64_t>(remaining, std::numeric_limits<unsigned>::max()));
}

/// Attribute an inconclusive result to the budget that caused it.
///
/// Z3 cannot be asked directly: smt_context records the same CANCELED failure
/// for a wall-clock cancel and for resource exhaustion, so reason_unknown
/// reads "canceled" either way (4.15.2 never raises RESLIMIT_EH_CALLER). The
/// wall clock is therefore decided by the caller's own deadline, which is
/// sound because remainingTimeoutMs rounds the remaining time *up*: Z3's timer
/// can only fire at or after the deadline, so a stop observed before the
/// deadline cannot be the wall clock's doing.
static BudgetExhausted
attributeBudgetStop(std::chrono::steady_clock::time_point deadline,
                    const SearchBudget &budget) {
  if (std::chrono::steady_clock::now() >= deadline)
    return BudgetExhausted::WallTime;
  if (budget.rlimit > 0)
    return BudgetExhausted::RLimit;
  return BudgetExhausted::None;
}

/// joint-solver-0.1: sweep II and minimize the composite stage, buffer
/// depth, recurrence, and register-pressure objective over node cycles.
namespace v01 {

constexpr llvm::StringLiteral kSchemaVersion = "joint-solver-0.1";
constexpr int64_t kMaxStageWeight = 10240000;
constexpr int64_t kDepthWeight = 102400;
constexpr int64_t kRecurrenceSpanWeight = 8192;
constexpr int64_t kRegisterPressureWeight = 1024;

struct Node {
  int64_t id;
  std::string pipeline;
  int64_t duration;
  bool streaming;
};

struct Edge {
  size_t src;
  size_t dst;
  int64_t latency;
  int64_t distance;
};

enum class BufferKind { Smem, Tmem };

struct Buffer {
  int64_t id;
  size_t alloc;
  BufferKind kind;
  int64_t sizeBytes;
  int64_t tmemCols;
  std::vector<size_t> consumers;
};

struct Problem {
  int64_t minII;
  int64_t maxII;
  int64_t smemBudget;
  int64_t tmemColLimit;
  SearchBudget budget;
  bool streamingVL;
  std::optional<size_t> canonicalRoot;
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::vector<Buffer> buffers;
};

static std::optional<Problem> parseProblem(llvm::StringRef problemJson) {
  auto parsed = llvm::json::parse(problemJson);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return std::nullopt;
  }
  auto *root = parsed->getAsObject();
  if (!root || root->get("mode"))
    return std::nullopt;
  auto version = root->getString("version");
  if (!version || *version != kSchemaVersion)
    return std::nullopt;

  Problem problem;
  if (!getRequiredInteger(*root, "min_ii", problem.minII) ||
      !getRequiredInteger(*root, "max_ii", problem.maxII) ||
      !getRequiredInteger(*root, "smem_budget", problem.smemBudget) ||
      !getRequiredInteger(*root, "tmem_col_limit", problem.tmemColLimit) ||
      problem.minII <= 0 || problem.maxII < problem.minII ||
      problem.smemBudget < 0 || problem.tmemColLimit < 0)
    return std::nullopt;

  if (!parseSearchBudget(*root, problem.budget))
    return std::nullopt;
  problem.streamingVL = false;
  if (root->get("streaming_vl")) {
    auto value = root->getBoolean("streaming_vl");
    if (!value)
      return std::nullopt;
    problem.streamingVL = *value;
  }

  auto *nodeValues = root->getArray("nodes");
  auto *edgeValues = root->getArray("edges");
  auto *bufferValues = root->getArray("buffers");
  if (!nodeValues || !edgeValues || !bufferValues ||
      nodeValues->size() > std::numeric_limits<unsigned>::max() ||
      edgeValues->size() > std::numeric_limits<unsigned>::max() ||
      bufferValues->size() > std::numeric_limits<unsigned>::max())
    return std::nullopt;

  std::unordered_map<int64_t, size_t> nodeIndices;
  problem.nodes.reserve(nodeValues->size());
  for (const llvm::json::Value &value : *nodeValues) {
    auto *object = value.getAsObject();
    int64_t id = 0;
    int64_t duration = 0;
    if (!object || !getRequiredInteger(*object, "id", id) ||
        !getRequiredInteger(*object, "duration", duration) || duration < 0)
      return std::nullopt;
    auto pipeline = object->getString("pipeline");
    if (!pipeline || nodeIndices.count(id))
      return std::nullopt;
    bool streaming = false;
    if (object->get("streaming")) {
      auto parsedStreaming = object->getBoolean("streaming");
      if (!parsedStreaming)
        return std::nullopt;
      streaming = *parsedStreaming;
    }
    nodeIndices.emplace(id, problem.nodes.size());
    problem.nodes.push_back(Node{id, pipeline->str(), duration, streaming});
  }

  if (root->get("canonical_root")) {
    auto rootId = root->getInteger("canonical_root");
    if (!rootId)
      return std::nullopt;
    auto it = nodeIndices.find(*rootId);
    if (it == nodeIndices.end())
      return std::nullopt;
    problem.canonicalRoot = it->second;
  }

  problem.edges.reserve(edgeValues->size());
  for (const llvm::json::Value &value : *edgeValues) {
    auto *object = value.getAsObject();
    int64_t srcId = 0;
    int64_t dstId = 0;
    int64_t latency = 0;
    int64_t distance = 0;
    if (!object || !getRequiredInteger(*object, "src", srcId) ||
        !getRequiredInteger(*object, "dst", dstId) ||
        !getRequiredInteger(*object, "latency", latency) ||
        !getRequiredInteger(*object, "distance", distance) || latency < 0 ||
        distance < 0)
      return std::nullopt;
    auto src = nodeIndices.find(srcId);
    auto dst = nodeIndices.find(dstId);
    if (src == nodeIndices.end() || dst == nodeIndices.end())
      return std::nullopt;
    problem.edges.push_back(Edge{src->second, dst->second, latency, distance});
  }

  problem.buffers.reserve(bufferValues->size());
  for (const llvm::json::Value &value : *bufferValues) {
    auto *object = value.getAsObject();
    int64_t allocId = 0;
    if (!object || !getRequiredInteger(*object, "alloc_node", allocId))
      return std::nullopt;
    auto alloc = nodeIndices.find(allocId);
    auto kindName = object->getString("kind");
    auto *consumers = object->getArray("consumers");
    if (alloc == nodeIndices.end() || !kindName || !consumers)
      return std::nullopt;

    BufferKind kind;
    if (*kindName == "smem")
      kind = BufferKind::Smem;
    else if (*kindName == "tmem")
      kind = BufferKind::Tmem;
    else
      return std::nullopt;

    int64_t sizeBytes = 0;
    int64_t tmemCols = 0;
    if (!getRequiredInteger(*object, "size_bytes", sizeBytes) ||
        !getRequiredInteger(*object, "tmem_cols", tmemCols) || sizeBytes < 0 ||
        tmemCols < 0)
      return std::nullopt;
    int64_t bufferId = allocId;
    if (object->get("id")) {
      auto parsedId = object->getInteger("id");
      if (!parsedId)
        return std::nullopt;
      bufferId = *parsedId;
    }

    Buffer buffer{bufferId, alloc->second, kind, sizeBytes, tmemCols, {}};
    buffer.consumers.reserve(consumers->size());
    for (const llvm::json::Value &consumerValue : *consumers) {
      auto consumerId = consumerValue.getAsInteger();
      if (!consumerId)
        return std::nullopt;
      auto consumer = nodeIndices.find(*consumerId);
      if (consumer == nodeIndices.end())
        return std::nullopt;
      buffer.consumers.push_back(consumer->second);
    }
    problem.buffers.push_back(std::move(buffer));
  }

  return problem;
}

static int64_t effectiveLatency(const Problem &problem, const Edge &edge) {
  if (problem.streamingVL && problem.nodes[edge.src].streaming)
    return 0;
  return edge.latency;
}

static std::optional<int64_t> computeHorizon(const Problem &problem,
                                             int64_t ii) {
  if (problem.nodes.empty())
    return ii;
  __int128 maxAdvance = 1;
  for (const Edge &edge : problem.edges) {
    __int128 residual = static_cast<__int128>(effectiveLatency(problem, edge)) -
                        static_cast<__int128>(edge.distance) * ii;
    __int128 advance = (residual + static_cast<__int128>(2) * ii - 2) / ii;
    maxAdvance = std::max(maxAdvance, advance);
  }
  __int128 maxStages =
      1 + static_cast<__int128>(problem.nodes.size() - 1) * maxAdvance;
  maxStages = std::max<__int128>(maxStages, 4);
  __int128 horizon = maxStages * ii;
  if (horizon <= 0 || horizon > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(horizon);
}

static __int128 evaluateObjective(const Problem &problem, int64_t ii,
                                  const std::vector<int64_t> &cycles) {
  int64_t maxStage = 0;
  for (int64_t cycle : cycles)
    maxStage = std::max(maxStage, cycle / ii);

  __int128 depthSum = 0;
  __int128 smemTotal = 0;
  for (const Buffer &buffer : problem.buffers) {
    if (buffer.kind != BufferKind::Smem)
      continue;
    int64_t allocStage = cycles[buffer.alloc] / ii;
    int64_t lastStage = allocStage;
    for (size_t consumer : buffer.consumers)
      lastStage = std::max(lastStage, cycles[consumer] / ii);
    int64_t depth = lastStage - allocStage + 1;
    depthSum += depth;
    smemTotal += static_cast<__int128>(buffer.sizeBytes) * depth;
  }

  __int128 recurrenceSpan = 0;
  __int128 registerPressure = 0;
  for (const Edge &edge : problem.edges) {
    if (edge.distance > 0 && edge.src != edge.dst)
      recurrenceSpan += cycles[edge.src] - cycles[edge.dst];
    if (edge.distance == 0)
      registerPressure += cycles[edge.dst] - cycles[edge.src];
  }

  return static_cast<__int128>(kMaxStageWeight) * maxStage -
         static_cast<__int128>(kDepthWeight) * depthSum +
         static_cast<__int128>(kRecurrenceSpanWeight) * recurrenceSpan +
         static_cast<__int128>(kRegisterPressureWeight) * registerPressure +
         smemTotal;
}

static __int128 objectiveLowerBound(const Problem &problem, int64_t ii,
                                    int64_t horizon) {
  __int128 maxDelta = horizon - 1;
  __int128 maxStages = horizon / ii;
  __int128 lowerBound = 0;
  for (const Buffer &buffer : problem.buffers)
    if (buffer.kind == BufferKind::Smem)
      lowerBound -= static_cast<__int128>(kDepthWeight) * maxStages;
  for (const Edge &edge : problem.edges) {
    if (edge.distance > 0 && edge.src != edge.dst)
      lowerBound -= static_cast<__int128>(kRecurrenceSpanWeight) * maxDelta;
    if (edge.distance == 0)
      lowerBound -= static_cast<__int128>(kRegisterPressureWeight) * maxDelta;
  }
  return lowerBound;
}

struct CandidateResult {
  CandidateStatus status = CandidateStatus::Error;
  std::vector<int64_t> cycles;
  std::string reason;
  int64_t horizon = 0;
  __int128 objective = 0;
  BudgetExhausted budgetExhausted = BudgetExhausted::None;
  std::optional<uint64_t> rlimitUsed;
};

static CandidateResult
solveCandidate(const Problem &problem, int64_t ii,
               std::chrono::steady_clock::time_point deadline) {
  CandidateResult result;
  auto horizon = computeHorizon(problem, ii);
  if (!horizon)
    return result;
  result.horizon = *horizon;

  for (const Node &node : problem.nodes) {
    if (node.pipeline != "NONE" && std::max<int64_t>(node.duration, 1) > ii) {
      result.status = CandidateStatus::Unsat;
      return result;
    }
  }
  if (deadlineExpired(deadline)) {
    result.status = CandidateStatus::Unknown;
    result.reason = "timeout";
    result.budgetExhausted = BudgetExhausted::WallTime;
    return result;
  }

  z3ErrorSeen = false;
  Context ownedContext;
  Z3_context context = ownedContext.get();
  if (!context)
    return result;
  Solver ownedSolver(context);
  Z3_solver solver = ownedSolver.get();
  if (!solver)
    return result;
  Z3ConstraintEncoder encoder(context, solver);

  Z3_ast iiValue = encoder.intValue(ii);
  Z3_ast horizonValue = encoder.intValue(*horizon);
  std::vector<Z3_ast> cycles;
  std::vector<Z3_ast> stages;
  std::vector<Z3_ast> phases;
  cycles.reserve(problem.nodes.size());
  stages.reserve(problem.nodes.size());
  phases.reserve(problem.nodes.size());
  for (size_t index = 0; index < problem.nodes.size(); ++index) {
    Z3_ast cycle = encoder.variable("cycle_" + std::to_string(index));
    cycles.push_back(cycle);
    stages.push_back(Z3_mk_div(context, cycle, iiValue));
    phases.push_back(Z3_mk_mod(context, cycle, iiValue));
    encoder.assertFormula(Z3_mk_ge(context, cycle, encoder.intValue(0)));
    encoder.assertFormula(Z3_mk_lt(context, cycle, horizonValue));
  }

  if (problem.canonicalRoot) {
    encoder.assertFormula(
        Z3_mk_eq(context, cycles[*problem.canonicalRoot], encoder.intValue(0)));
  } else if (!cycles.empty()) {
    std::vector<Z3_ast> isZero;
    isZero.reserve(cycles.size());
    for (Z3_ast cycle : cycles)
      isZero.push_back(Z3_mk_eq(context, cycle, encoder.intValue(0)));
    encoder.assertFormula(encoder.disjunction(isZero));
  }

  for (const Edge &edge : problem.edges) {
    Z3_ast distance = encoder.mul(encoder.intValue(edge.distance), iiValue);
    Z3_ast rhs = encoder.sub(
        encoder.add(cycles[edge.src],
                    encoder.intValue(effectiveLatency(problem, edge))),
        distance);
    encoder.assertFormula(Z3_mk_ge(context, cycles[edge.dst], rhs));
    if (edge.distance == 0)
      encoder.assertFormula(
          Z3_mk_le(context, stages[edge.src], stages[edge.dst]));
  }

  // Ordered by pipeline name: this map is iterated to emit the mutual-exclusion
  // constraints, so an unordered container would make the assertion order —
  // and therefore the search — depend on the standard library's hash.
  std::map<std::string, std::vector<size_t>> nodesByPipeline;
  for (size_t index = 0; index < problem.nodes.size(); ++index)
    if (problem.nodes[index].pipeline != "NONE")
      nodesByPipeline[problem.nodes[index].pipeline].push_back(index);
  for (const auto &entry : nodesByPipeline) {
    const std::vector<size_t> &members = entry.second;
    for (size_t leftIndex = 0; leftIndex < members.size(); ++leftIndex) {
      size_t left = members[leftIndex];
      int64_t leftDuration = std::max<int64_t>(problem.nodes[left].duration, 1);
      for (size_t rightIndex = leftIndex + 1; rightIndex < members.size();
           ++rightIndex) {
        size_t right = members[rightIndex];
        int64_t rightDuration =
            std::max<int64_t>(problem.nodes[right].duration, 1);
        Z3_ast delta = Z3_mk_mod(
            context,
            encoder.add(encoder.sub(phases[right], phases[left]), iiValue),
            iiValue);
        std::vector<Z3_ast> separated{
            Z3_mk_ge(context, delta, encoder.intValue(leftDuration)),
            Z3_mk_le(context, delta, encoder.intValue(ii - rightDuration))};
        encoder.assertFormula(encoder.conjunction(separated));
      }
    }
  }

  std::vector<Z3_ast> smemDepths;
  std::vector<Z3_ast> smemCharges;
  struct TmemLifetime {
    Z3_ast start;
    Z3_ast end;
    int64_t columns;
  };
  std::vector<TmemLifetime> tmemLifetimes;
  for (const Buffer &buffer : problem.buffers) {
    if (buffer.kind == BufferKind::Smem) {
      std::vector<Z3_ast> lifetimeStages{stages[buffer.alloc]};
      for (size_t consumer : buffer.consumers)
        lifetimeStages.push_back(stages[consumer]);
      Z3_ast depth = encoder.add(
          encoder.sub(encoder.maximum(lifetimeStages), stages[buffer.alloc]),
          encoder.intValue(1));
      smemDepths.push_back(depth);
      smemCharges.push_back(
          encoder.mul(encoder.intValue(buffer.sizeBytes), depth));
      continue;
    }

    std::vector<Z3_ast> lifetimeCycles{cycles[buffer.alloc]};
    for (size_t consumer : buffer.consumers)
      lifetimeCycles.push_back(cycles[consumer]);
    Z3_ast end =
        encoder.add(encoder.maximum(lifetimeCycles), encoder.intValue(1));
    tmemLifetimes.push_back(
        TmemLifetime{cycles[buffer.alloc], end, buffer.tmemCols});
  }
  Z3_ast smemTotal = encoder.sum(smemCharges);
  encoder.assertFormula(
      Z3_mk_le(context, smemTotal, encoder.intValue(problem.smemBudget)));

  for (const TmemLifetime &checkpoint : tmemLifetimes) {
    std::vector<Z3_ast> activeColumns;
    activeColumns.reserve(tmemLifetimes.size());
    for (const TmemLifetime &lifetime : tmemLifetimes) {
      std::vector<Z3_ast> isActive{
          Z3_mk_le(context, lifetime.start, checkpoint.start),
          Z3_mk_lt(context, checkpoint.start, lifetime.end)};
      activeColumns.push_back(Z3_mk_ite(context, encoder.conjunction(isActive),
                                        encoder.intValue(lifetime.columns),
                                        encoder.intValue(0)));
    }
    encoder.assertFormula(Z3_mk_le(context, encoder.sum(activeColumns),
                                   encoder.intValue(problem.tmemColLimit)));
  }

  std::vector<Z3_ast> recurrenceSpanTerms;
  std::vector<Z3_ast> registerPressureTerms;
  recurrenceSpanTerms.reserve(problem.edges.size());
  registerPressureTerms.reserve(problem.edges.size());
  for (const Edge &edge : problem.edges) {
    if (edge.distance > 0 && edge.src != edge.dst)
      recurrenceSpanTerms.push_back(
          encoder.sub(cycles[edge.src], cycles[edge.dst]));
    if (edge.distance == 0)
      registerPressureTerms.push_back(
          encoder.sub(cycles[edge.dst], cycles[edge.src]));
  }

  Z3_ast maxStage = encoder.maximum(stages);
  std::vector<Z3_ast> objectiveTerms{
      encoder.mul(encoder.intValue(kMaxStageWeight), maxStage),
      encoder.mul(encoder.intValue(-kDepthWeight), encoder.sum(smemDepths)),
      encoder.mul(encoder.intValue(kRecurrenceSpanWeight),
                  encoder.sum(recurrenceSpanTerms)),
      encoder.mul(encoder.intValue(kRegisterPressureWeight),
                  encoder.sum(registerPressureTerms)),
      smemTotal};
  Z3_ast objective = encoder.sum(objectiveTerms);

  auto extractCandidate = [&]() -> std::optional<CandidateResult> {
    Z3_model rawModel = Z3_solver_get_model(context, solver);
    if (!rawModel)
      return std::nullopt;
    Model model(context, rawModel);
    CandidateResult candidate;
    candidate.horizon = *horizon;
    candidate.cycles.reserve(cycles.size());
    for (Z3_ast cycle : cycles) {
      Z3_ast value = nullptr;
      int64_t integer = 0;
      if (!Z3_model_eval(context, model.get(), cycle, true, &value) || !value ||
          !Z3_get_numeral_int64(context, value, &integer))
        return std::nullopt;
      candidate.cycles.push_back(integer);
    }
    candidate.objective = evaluateObjective(problem, ii, candidate.cycles);
    candidate.status = CandidateStatus::Sat;
    return candidate;
  };

  // A stop reported by Z3 itself: record which budget ran out and how much of
  // the deterministic one was spent getting there.
  auto makeStoppedResult = [&]() {
    CandidateResult stopped;
    stopped.status = CandidateStatus::Unknown;
    stopped.horizon = *horizon;
    if (const char *reason = Z3_solver_get_reason_unknown(context, solver))
      stopped.reason = reason;
    if (stopped.reason.empty())
      stopped.reason = "unknown";
    stopped.budgetExhausted = attributeBudgetStop(deadline, problem.budget);
    stopped.rlimitUsed = readRLimitCount(context, solver);
    return stopped;
  };
  // The deadline elapsed before Z3 was even entered, so there is no solver
  // state to interrogate.
  auto makeExpiredResult = [&]() {
    CandidateResult expired;
    expired.status = CandidateStatus::Unknown;
    expired.horizon = *horizon;
    expired.reason = "timeout";
    expired.budgetExhausted = BudgetExhausted::WallTime;
    return expired;
  };

  auto timeoutMs = remainingTimeoutMs(deadline);
  if (!timeoutMs)
    return makeExpiredResult();
  if (z3ErrorSeen ||
      !configureSolver(context, solver, *timeoutMs, problem.budget))
    return CandidateResult{};

  Z3_lbool status = Z3_solver_check(context, solver);
  if (z3ErrorSeen)
    return CandidateResult{};
  if (status == Z3_L_FALSE) {
    result.status = CandidateStatus::Unsat;
    result.rlimitUsed = readRLimitCount(context, solver);
    return result;
  }
  if (status == Z3_L_UNDEF)
    return makeStoppedResult();
  if (status != Z3_L_TRUE)
    return CandidateResult{};

  auto baseCandidate = extractCandidate();
  if (!baseCandidate)
    return CandidateResult{};
  result = std::move(*baseCandidate);
  __int128 low = objectiveLowerBound(problem, ii, *horizon);
  __int128 high = result.objective;
  if (high < low)
    return CandidateResult{};

  while (low < high) {
    __int128 mid = low + (high - low) / 2;
    Z3_solver_push(context, solver);
    encoder.assertFormula(
        Z3_mk_le(context, objective, encoder.wideIntValue(mid)));

    timeoutMs = remainingTimeoutMs(deadline);
    if (!timeoutMs) {
      Z3_solver_pop(context, solver, 1);
      return makeExpiredResult();
    }
    if (z3ErrorSeen ||
        !configureSolver(context, solver, *timeoutMs, problem.budget)) {
      Z3_solver_pop(context, solver, 1);
      return CandidateResult{};
    }

    status = Z3_solver_check(context, solver);
    if (z3ErrorSeen) {
      Z3_solver_pop(context, solver, 1);
      return CandidateResult{};
    }
    if (status == Z3_L_FALSE) {
      Z3_solver_pop(context, solver, 1);
      low = mid + 1;
      continue;
    }
    if (status == Z3_L_UNDEF) {
      CandidateResult stopped = makeStoppedResult();
      Z3_solver_pop(context, solver, 1);
      return stopped;
    }
    if (status != Z3_L_TRUE) {
      Z3_solver_pop(context, solver, 1);
      return CandidateResult{};
    }

    auto tighterCandidate = extractCandidate();
    if (!tighterCandidate || tighterCandidate->objective < low ||
        tighterCandidate->objective > mid) {
      Z3_solver_pop(context, solver, 1);
      return CandidateResult{};
    }
    Z3_solver_pop(context, solver, 1);
    result = std::move(*tighterCandidate);
    high = result.objective;
  }
  if (result.objective != low)
    return CandidateResult{};
  result.rlimitUsed = readRLimitCount(context, solver);
  return result;
}

static llvm::json::Array toJsonArray(const std::vector<int64_t> &values) {
  llvm::json::Array result;
  result.reserve(values.size());
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

/// `rlimit_used` is the deterministic cost of the whole II sweep, summed over
/// the per-II contexts. It is reported on every outcome because it is what the
/// default rlimit is calibrated against, and because a determinism test that
/// compares it catches two runs reaching the same schedule by different search
/// paths — which a comparison of the schedule alone would miss.
static llvm::json::Object makeStats(const std::vector<int64_t> &tried,
                                    const std::vector<int64_t> &unsat,
                                    const std::vector<int64_t> &unknown,
                                    const SearchBudget &budget,
                                    uint64_t rlimitUsed) {
  llvm::json::Object stats{{"iis_tried", toJsonArray(tried)},
                           {"unsat_iis", toJsonArray(unsat)},
                           {"unknown_iis", toJsonArray(unknown)}};
  stats["rlimit"] = static_cast<int64_t>(budget.rlimit);
  stats["rlimit_used"] = static_cast<int64_t>(rlimitUsed);
  return stats;
}

static llvm::json::Object makeSuccess(const Problem &problem, int64_t ii,
                                      const CandidateResult &candidate,
                                      const std::vector<int64_t> &tried,
                                      const std::vector<int64_t> &unsat,
                                      const std::vector<int64_t> &unknown,
                                      uint64_t rlimitUsed) {
  llvm::json::Object cycleValues;
  for (size_t index = 0; index < problem.nodes.size(); ++index)
    cycleValues[std::to_string(problem.nodes[index].id)] =
        candidate.cycles[index];

  llvm::json::Object depthValues;
  for (const Buffer &buffer : problem.buffers) {
    if (buffer.kind != BufferKind::Smem)
      continue;
    int64_t allocStage = candidate.cycles[buffer.alloc] / ii;
    int64_t lastStage = allocStage;
    for (size_t consumer : buffer.consumers)
      lastStage = std::max(lastStage, candidate.cycles[consumer] / ii);
    int64_t depth = lastStage - allocStage + 1;
    depthValues[std::to_string(buffer.id)] = depth;
  }

  double objective = static_cast<double>(candidate.objective);

  return llvm::json::Object{
      {"version", kSchemaVersion},
      {"status", "ok"},
      {"ii", ii},
      {"cycles", std::move(cycleValues)},
      {"buffer_depths", std::move(depthValues)},
      {"objective", static_cast<double>(objective)},
      {"stats", makeStats(tried, unsat, unknown, problem.budget, rlimitUsed)},
  };
}

static FailureOr<std::string> run(llvm::StringRef problemJson) {
  std::optional<Problem> problem = parseProblem(problemJson);
  if (!problem)
    return failure();

  double timeoutMsDouble = std::ceil(problem->budget.timeLimitS * 1000.0);
  int64_t timeoutMs = static_cast<int64_t>(std::min<double>(
      timeoutMsDouble, std::numeric_limits<int64_t>::max() / 2.0));
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(std::max<int64_t>(timeoutMs, 1));

  std::vector<int64_t> tried;
  std::vector<int64_t> unsat;
  std::vector<int64_t> unknown;
  uint64_t rlimitUsed = 0;
  for (int64_t ii = problem->minII;; ++ii) {
    tried.push_back(ii);
    CandidateResult candidate = solveCandidate(*problem, ii, deadline);
    // Each II is solved in its own context, so the per-II counters sum to the
    // deterministic cost of the sweep.
    rlimitUsed += candidate.rlimitUsed.value_or(0);
    if (candidate.status == CandidateStatus::Sat) {
      return serialize(makeSuccess(*problem, ii, candidate, tried, unsat,
                                   unknown, rlimitUsed));
    }
    if (candidate.status == CandidateStatus::Unsat) {
      unsat.push_back(ii);
    } else if (candidate.status == CandidateStatus::Unknown) {
      unknown.push_back(ii);
      std::string message =
          "joint solve inconclusive at II " + std::to_string(ii);
      llvm::json::Object diagnostic{
          {"schema", "joint-solver-diagnostic-0.1"},
          {"status", "inconclusive"},
          {"ii", ii},
          {"backendStatus", "UNKNOWN"},
          {"message", candidate.reason.empty() ? message : candidate.reason},
      };
      return serialize(llvm::json::Object{
          {"version", kSchemaVersion},
          {"status", "inconclusive"},
          {"proven_unsat", false},
          {"backend_status", "UNKNOWN"},
          // Which budget ran out. `rlimit` is deterministic and reproducible —
          // a property of the problem, answered by raising the budget or
          // simplifying the model. `walltime` is an environment signal and,
          // with rlimit binding, should essentially never appear.
          {"budget_exhausted",
           budgetExhaustedName(candidate.budgetExhausted).str()},
          {"message", std::move(message)},
          {"diagnostic", std::move(diagnostic)},
          {"stats",
           makeStats(tried, unsat, unknown, problem->budget, rlimitUsed)},
      });
    } else {
      return failure();
    }
    if (ii == problem->maxII)
      break;
  }

  std::string message = "no feasible II in [" + std::to_string(problem->minII) +
                        ", " + std::to_string(problem->maxII) + "]";
  return serialize(llvm::json::Object{
      {"version", kSchemaVersion},
      {"status", "infeasible"},
      {"proven_unsat", true},
      {"backend_status", "INFEASIBLE"},
      {"message", std::move(message)},
      {"stats", makeStats(tried, unsat, unknown, problem->budget, rlimitUsed)},
  });
}

} // namespace v01

/// joint-solver-0.2: fixed-cycle partitioning and same-stage cycle-plus-
/// warp-group refinement, including the concrete lowering-event plan.
namespace v02 {

constexpr llvm::StringLiteral kSchemaVersion = "joint-solver-0.2";
constexpr llvm::StringLiteral kLoweringPlanVersion = "lowering-plan-0.1";

constexpr int64_t kPartitionCycleWeight = 2;
constexpr int64_t kJointMaxStageWeight = 10240000;
constexpr int64_t kJointDepthWeight = -102400;
constexpr int64_t kJointRecurrenceWeight = 8192;
constexpr int64_t kJointRegisterPressureWeight = 1024;
constexpr int64_t kJointCrossIssueWeight = 1024;
constexpr int64_t kJointRegisterResidualWeight = 512;

static bool isLoweringEventKind(llvm::StringRef kind) {
  return kind == "wait" || kind == "arrive" || kind == "expect" ||
         kind == "local_store" || kind == "local_load" || kind == "fence" ||
         kind == "tc_commit";
}

static int64_t modularOverlap(int64_t leftStart, int64_t leftDuration,
                              int64_t rightStart, int64_t rightDuration,
                              int64_t ii) {
  using Segment = std::pair<int64_t, int64_t>;
  auto segments = [ii](int64_t start, int64_t duration) {
    std::vector<Segment> result;
    if (duration <= 0)
      return result;
    duration = std::min(duration, ii);
    int64_t phase = start % ii;
    if (phase < 0)
      phase += ii;
    __int128 end = static_cast<__int128>(phase) + duration;
    if (end <= ii) {
      result.push_back({phase, static_cast<int64_t>(end)});
    } else {
      result.push_back({phase, ii});
      result.push_back({0, static_cast<int64_t>(end - ii)});
    }
    return result;
  };
  int64_t overlap = 0;
  for (const Segment &left : segments(leftStart, leftDuration))
    for (const Segment &right : segments(rightStart, rightDuration))
      overlap += std::max<int64_t>(0, std::min(left.second, right.second) -
                                          std::max(left.first, right.first));
  return overlap;
}

enum class SolveMode { Partition, Joint };
enum class Relation { Always, SameWG, DifferentWG };
enum class Owner { Src, Dst };
enum class Placement { Before, After };

struct Node {
  int64_t id;
  int64_t fixedCycle;
  int64_t duration;
  int64_t latency;
  int64_t frequency;
  std::string pipeline;
  std::optional<size_t> cluster;
};

struct Cluster {
  int64_t id;
  int64_t minWarps;
  std::vector<size_t> nodes;
};

struct Edge {
  size_t src;
  size_t dst;
  int64_t srcResultIndex;
  int64_t latency;
  int64_t distance;
  int64_t frequency;
  int64_t roundTrip;
  int64_t crossIssue;
  int64_t channelBytes;
  std::optional<size_t> srcCluster;
  std::optional<size_t> dstCluster;
};

struct BufferConsumer {
  size_t node;
  int64_t latency;
  int64_t distance;
};

struct Buffer {
  int64_t id;
  size_t producer;
  int64_t sizeBytes;
  int64_t count;
  int64_t minCount;
  std::string kind;
  std::vector<BufferConsumer> consumers;
};

struct LoweringEvent {
  int64_t id;
  std::string kind;
  Owner owner;
  size_t anchor;
  Placement placement;
  std::string pipeline;
  int64_t duration;
  int64_t distance;
  bool blocking;
};

struct LoweringTemplate {
  int64_t id;
  Relation relation;
  size_t srcNode;
  size_t dstNode;
  size_t srcCluster;
  size_t dstCluster;
  std::vector<LoweringEvent> events;
};

struct Problem {
  SolveMode mode;
  int64_t ii;
  int64_t maxWGs;
  int64_t committedSmem;
  int64_t fixedSmem;
  int64_t smemBudget;
  int64_t tmemBudgetBytes;
  int64_t defaultWGFootprint;
  int64_t smRegs;
  int64_t defaultSlack;
  std::optional<int64_t> regBudget;
  std::optional<size_t> canonicalRoot;
  SearchBudget budget;
  std::vector<int64_t> warpFootprint;
  std::vector<Node> nodes;
  std::vector<Cluster> clusters;
  std::vector<Edge> edges;
  std::vector<Buffer> buffers;
  std::vector<LoweringTemplate> loweringTemplates;
  std::vector<std::pair<size_t, size_t>> warpGroupConflicts;
};

static std::optional<Problem> parseProblem(llvm::StringRef problemJson) {
  auto parsed = llvm::json::parse(problemJson);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return std::nullopt;
  }
  auto *root = parsed->getAsObject();
  if (!root)
    return std::nullopt;
  auto version = root->getString("version");
  auto mode = root->getString("mode");
  if (!version || *version != kSchemaVersion || !mode)
    return std::nullopt;

  Problem problem;
  if (*mode == "partition")
    problem.mode = SolveMode::Partition;
  else if (*mode == "joint")
    problem.mode = SolveMode::Joint;
  else
    return std::nullopt;

  int64_t emitterCapsVersion = 0;
  if (!getRequiredInteger(*root, "emitter_caps_version", emitterCapsVersion) ||
      emitterCapsVersion != 2 || !getRequiredInteger(*root, "ii", problem.ii) ||
      problem.ii <= 0 ||
      !getRequiredInteger(*root, "max_wgs", problem.maxWGs) ||
      problem.maxWGs <= 0 ||
      !getRequiredInteger(*root, "committed_smem", problem.committedSmem) ||
      !getRequiredInteger(*root, "fixed_smem", problem.fixedSmem) ||
      !getRequiredInteger(*root, "smem_budget", problem.smemBudget) ||
      !getRequiredInteger(*root, "tmem_budget_bytes",
                          problem.tmemBudgetBytes) ||
      !getRequiredInteger(*root, "default_wg_footprint",
                          problem.defaultWGFootprint) ||
      !getRequiredInteger(*root, "sm_regs", problem.smRegs) ||
      !getRequiredInteger(*root, "default_slack", problem.defaultSlack) ||
      problem.committedSmem < 0 || problem.fixedSmem < 0 ||
      problem.smemBudget < 0 || problem.tmemBudgetBytes < 0 ||
      problem.defaultWGFootprint < 0 || problem.smRegs < 0 ||
      problem.defaultSlack < 0)
    return std::nullopt;

  if (!parseSearchBudget(*root, problem.budget))
    return std::nullopt;
  if (root->get("reg_budget")) {
    auto value = root->getInteger("reg_budget");
    if (!value || *value < 0)
      return std::nullopt;
    if (*value > 0)
      problem.regBudget = *value;
  }

  auto *nodeValues = root->getArray("nodes");
  auto *clusterValues = root->getArray("clusters");
  auto *edgeValues = root->getArray("edges");
  auto *bufferValues = root->getArray("buffers");
  auto *templateValues = root->getArray("lowering_templates");
  auto *footprintValues = root->getArray("warp_footprint");
  if (!nodeValues || !clusterValues || !edgeValues || !bufferValues ||
      !templateValues || !footprintValues || clusterValues->empty() ||
      nodeValues->size() > std::numeric_limits<unsigned>::max() ||
      clusterValues->size() > std::numeric_limits<unsigned>::max())
    return std::nullopt;

  problem.warpFootprint.reserve(footprintValues->size());
  for (const llvm::json::Value &value : *footprintValues) {
    auto footprint = value.getAsInteger();
    if (!footprint || *footprint < 0)
      return std::nullopt;
    problem.warpFootprint.push_back(*footprint);
  }
  if (problem.warpFootprint.size() <= 8)
    return std::nullopt;

  std::unordered_map<int64_t, size_t> nodeIndices;
  problem.nodes.reserve(nodeValues->size());
  for (const llvm::json::Value &value : *nodeValues) {
    auto *object = value.getAsObject();
    int64_t id = 0;
    int64_t cycle = 0;
    int64_t duration = 0;
    int64_t latency = 0;
    int64_t frequency = 0;
    std::string pipeline;
    if (!object || !getRequiredInteger(*object, "id", id) ||
        !getRequiredInteger(*object, "cycle", cycle) ||
        !getRequiredInteger(*object, "duration", duration) ||
        !getRequiredInteger(*object, "latency", latency) ||
        !getRequiredInteger(*object, "freq", frequency) ||
        !getRequiredString(*object, "pipeline", pipeline) || cycle < 0 ||
        duration < 0 || latency < 0 || frequency <= 0 || nodeIndices.count(id))
      return std::nullopt;
    nodeIndices.emplace(id, problem.nodes.size());
    problem.nodes.push_back(Node{id, cycle, duration, latency, frequency,
                                 std::move(pipeline), std::nullopt});
  }

  std::unordered_map<int64_t, size_t> clusterIndices;
  problem.clusters.reserve(clusterValues->size());
  for (const llvm::json::Value &value : *clusterValues) {
    auto *object = value.getAsObject();
    int64_t id = 0;
    int64_t minWarps = 0;
    auto *nodes = object ? object->getArray("nodes") : nullptr;
    if (!object || !getRequiredInteger(*object, "id", id) ||
        !getRequiredInteger(*object, "min_warps", minWarps) || !nodes ||
        minWarps <= 0 || minWarps > 8 || clusterIndices.count(id))
      return std::nullopt;
    size_t clusterIndex = problem.clusters.size();
    clusterIndices.emplace(id, clusterIndex);
    Cluster cluster{id, minWarps, {}};
    cluster.nodes.reserve(nodes->size());
    for (const llvm::json::Value &nodeValue : *nodes) {
      auto nodeId = nodeValue.getAsInteger();
      if (!nodeId)
        return std::nullopt;
      auto node = nodeIndices.find(*nodeId);
      if (node == nodeIndices.end() || problem.nodes[node->second].cluster)
        return std::nullopt;
      problem.nodes[node->second].cluster = clusterIndex;
      cluster.nodes.push_back(node->second);
    }
    if (cluster.nodes.empty())
      return std::nullopt;
    problem.clusters.push_back(std::move(cluster));
  }
  if (problem.maxWGs > static_cast<int64_t>(problem.clusters.size()))
    problem.maxWGs = problem.clusters.size();

  if (root->get("canonical_root")) {
    auto rootId = root->getInteger("canonical_root");
    if (!rootId)
      return std::nullopt;
    auto node = nodeIndices.find(*rootId);
    if (node == nodeIndices.end())
      return std::nullopt;
    problem.canonicalRoot = node->second;
  }

  problem.edges.reserve(edgeValues->size());
  for (const llvm::json::Value &value : *edgeValues) {
    auto *object = value.getAsObject();
    int64_t srcId = 0;
    int64_t dstId = 0;
    int64_t srcResultIndex = 0;
    int64_t latency = 0;
    int64_t distance = 0;
    int64_t frequency = 0;
    int64_t roundTrip = 0;
    int64_t crossIssue = 0;
    int64_t channelBytes = 0;
    if (!object || !getRequiredInteger(*object, "src", srcId) ||
        !getRequiredInteger(*object, "dst", dstId) ||
        !getRequiredInteger(*object, "src_result_idx", srcResultIndex) ||
        !getRequiredInteger(*object, "latency", latency) ||
        !getRequiredInteger(*object, "distance", distance) ||
        !getRequiredInteger(*object, "freq", frequency) ||
        !getRequiredInteger(*object, "rt", roundTrip) ||
        !getRequiredInteger(*object, "xissue", crossIssue) ||
        !getRequiredInteger(*object, "chan_bytes", channelBytes) ||
        srcResultIndex < 0 || latency < 0 || distance < 0 || frequency <= 0 ||
        roundTrip < 0 || crossIssue < 0 || channelBytes < 0)
      return std::nullopt;
    auto src = nodeIndices.find(srcId);
    auto dst = nodeIndices.find(dstId);
    if (src == nodeIndices.end() || dst == nodeIndices.end())
      return std::nullopt;

    std::optional<size_t> srcCluster;
    std::optional<size_t> dstCluster;
    bool hasSrcCluster = object->get("src_cluster") != nullptr;
    bool hasDstCluster = object->get("dst_cluster") != nullptr;
    if (hasSrcCluster != hasDstCluster)
      return std::nullopt;
    if (hasSrcCluster) {
      auto srcClusterId = object->getInteger("src_cluster");
      auto dstClusterId = object->getInteger("dst_cluster");
      if (!srcClusterId || !dstClusterId)
        return std::nullopt;
      auto srcIt = clusterIndices.find(*srcClusterId);
      auto dstIt = clusterIndices.find(*dstClusterId);
      if (srcIt == clusterIndices.end() || dstIt == clusterIndices.end() ||
          !problem.nodes[src->second].cluster ||
          !problem.nodes[dst->second].cluster ||
          *problem.nodes[src->second].cluster != srcIt->second ||
          *problem.nodes[dst->second].cluster != dstIt->second)
        return std::nullopt;
      srcCluster = srcIt->second;
      dstCluster = dstIt->second;
    } else if (roundTrip != 0 || crossIssue != 0 || channelBytes != 0) {
      return std::nullopt;
    }
    problem.edges.push_back(Edge{
        src->second, dst->second, srcResultIndex, latency, distance, frequency,
        roundTrip, crossIssue, channelBytes, srcCluster, dstCluster});
  }

  problem.buffers.reserve(bufferValues->size());
  for (const llvm::json::Value &value : *bufferValues) {
    auto *object = value.getAsObject();
    int64_t bufferId = 0;
    int64_t producerId = 0;
    int64_t sizeBytes = 0;
    int64_t count = 0;
    int64_t minCount = 0;
    std::string kind;
    auto *consumers = object ? object->getArray("consumers") : nullptr;
    if (!object || !getRequiredInteger(*object, "id", bufferId) ||
        !getRequiredInteger(*object, "producer", producerId) ||
        !getRequiredInteger(*object, "size_bytes", sizeBytes) ||
        !getRequiredInteger(*object, "count", count) ||
        !getRequiredInteger(*object, "min_count", minCount) ||
        !getRequiredString(*object, "kind", kind) || !consumers ||
        sizeBytes < 0 || count <= 0 || minCount <= 0 ||
        (kind != "smem" && kind != "tmem"))
      return std::nullopt;
    auto producer = nodeIndices.find(producerId);
    if (producer == nodeIndices.end())
      return std::nullopt;
    Buffer buffer{bufferId, producer->second, sizeBytes, count,
                  minCount, std::move(kind),  {}};
    buffer.consumers.reserve(consumers->size());
    for (const llvm::json::Value &consumerValue : *consumers) {
      auto *consumerObject = consumerValue.getAsObject();
      int64_t nodeId = 0;
      int64_t consumerLatency = 0;
      int64_t consumerDistance = 0;
      if (!consumerObject ||
          !getRequiredInteger(*consumerObject, "node", nodeId) ||
          !getRequiredInteger(*consumerObject, "latency", consumerLatency) ||
          !getRequiredInteger(*consumerObject, "distance", consumerDistance) ||
          consumerLatency < 0 || consumerDistance < 0)
        return std::nullopt;
      auto consumer = nodeIndices.find(nodeId);
      if (consumer == nodeIndices.end())
        return std::nullopt;
      buffer.consumers.push_back(
          BufferConsumer{consumer->second, consumerLatency, consumerDistance});
    }
    problem.buffers.push_back(std::move(buffer));
  }

  problem.loweringTemplates.reserve(templateValues->size());
  for (size_t templateIndex = 0; templateIndex < templateValues->size();
       ++templateIndex) {
    auto *object = (*templateValues)[templateIndex].getAsObject();
    int64_t id = 0;
    int64_t srcNodeId = 0;
    int64_t dstNodeId = 0;
    int64_t srcClusterId = 0;
    int64_t dstClusterId = 0;
    std::string relationName;
    auto *events = object ? object->getArray("events") : nullptr;
    if (!object || !getRequiredInteger(*object, "id", id) ||
        id != static_cast<int64_t>(templateIndex) ||
        !getRequiredString(*object, "relation", relationName) ||
        !getRequiredInteger(*object, "src_node", srcNodeId) ||
        !getRequiredInteger(*object, "dst_node", dstNodeId) ||
        !getRequiredInteger(*object, "src_cluster", srcClusterId) ||
        !getRequiredInteger(*object, "dst_cluster", dstClusterId) || !events)
      return std::nullopt;
    Relation relation;
    if (relationName == "always")
      relation = Relation::Always;
    else if (relationName == "same_wg")
      relation = Relation::SameWG;
    else if (relationName == "different_wg")
      relation = Relation::DifferentWG;
    else
      return std::nullopt;
    auto srcNode = nodeIndices.find(srcNodeId);
    auto dstNode = nodeIndices.find(dstNodeId);
    auto srcCluster = clusterIndices.find(srcClusterId);
    auto dstCluster = clusterIndices.find(dstClusterId);
    if (srcNode == nodeIndices.end() || dstNode == nodeIndices.end() ||
        srcCluster == clusterIndices.end() ||
        dstCluster == clusterIndices.end() ||
        problem.nodes[srcNode->second].cluster != srcCluster->second ||
        problem.nodes[dstNode->second].cluster != dstCluster->second)
      return std::nullopt;

    LoweringTemplate loweringTemplate{id,
                                      relation,
                                      srcNode->second,
                                      dstNode->second,
                                      srcCluster->second,
                                      dstCluster->second,
                                      {}};
    std::set<int64_t> eventIds;
    loweringTemplate.events.reserve(events->size());
    for (const llvm::json::Value &eventValue : *events) {
      auto *eventObject = eventValue.getAsObject();
      int64_t eventId = 0;
      int64_t anchorNodeId = 0;
      int64_t issueDuration = 0;
      int64_t completionLatency = 0;
      int64_t distance = 0;
      int64_t frequency = 0;
      int64_t bytes = 0;
      int64_t depth = 0;
      bool blocking = false;
      bool isAsync = false;
      std::string kind;
      std::string ownerName;
      std::string placementName;
      std::string pipeline;
      std::string semaphore;
      if (!eventObject || !getRequiredInteger(*eventObject, "id", eventId) ||
          !getRequiredString(*eventObject, "kind", kind) ||
          !isLoweringEventKind(kind) ||
          !getRequiredString(*eventObject, "owner", ownerName) ||
          !getRequiredInteger(*eventObject, "anchor_node", anchorNodeId) ||
          !getRequiredString(*eventObject, "placement", placementName) ||
          !getRequiredString(*eventObject, "pipeline", pipeline) ||
          !getRequiredInteger(*eventObject, "issue_duration", issueDuration) ||
          !getRequiredInteger(*eventObject, "completion_latency",
                              completionLatency) ||
          !getRequiredBoolean(*eventObject, "blocking", blocking) ||
          !getRequiredBoolean(*eventObject, "async", isAsync) ||
          !getRequiredInteger(*eventObject, "distance", distance) ||
          !getRequiredInteger(*eventObject, "frequency", frequency) ||
          !getRequiredInteger(*eventObject, "bytes", bytes) ||
          !getRequiredInteger(*eventObject, "depth", depth) ||
          !getRequiredString(*eventObject, "semaphore", semaphore) ||
          eventId < 0 || issueDuration < 0 || completionLatency < 0 ||
          distance < 0 || frequency <= 0 || bytes < 0 || depth < 0 ||
          !eventIds.insert(eventId).second)
        return std::nullopt;
      (void)isAsync;
      auto anchor = nodeIndices.find(anchorNodeId);
      if (anchor == nodeIndices.end())
        return std::nullopt;
      Owner owner;
      size_t ownerCluster;
      if (ownerName == "src") {
        owner = Owner::Src;
        ownerCluster = srcCluster->second;
      } else if (ownerName == "dst") {
        owner = Owner::Dst;
        ownerCluster = dstCluster->second;
      } else {
        return std::nullopt;
      }
      if (problem.nodes[anchor->second].cluster != ownerCluster)
        return std::nullopt;
      Placement placement;
      if (placementName == "before")
        placement = Placement::Before;
      else if (placementName == "after")
        placement = Placement::After;
      else
        return std::nullopt;
      __int128 duration = static_cast<__int128>(issueDuration) * frequency;
      if (duration > std::numeric_limits<int64_t>::max())
        return std::nullopt;
      loweringTemplate.events.push_back(
          LoweringEvent{eventId, std::move(kind), owner, anchor->second,
                        placement, std::move(pipeline),
                        static_cast<int64_t>(duration), distance, blocking});
    }
    problem.loweringTemplates.push_back(std::move(loweringTemplate));
  }

  if (auto *conflicts = root->getArray("warp_group_conflicts")) {
    problem.warpGroupConflicts.reserve(conflicts->size());
    for (const llvm::json::Value &value : *conflicts) {
      auto *pair = value.getAsArray();
      if (!pair || pair->size() != 2)
        return std::nullopt;
      auto leftId = (*pair)[0].getAsInteger();
      auto rightId = (*pair)[1].getAsInteger();
      if (!leftId || !rightId)
        return std::nullopt;
      auto left = clusterIndices.find(*leftId);
      auto right = clusterIndices.find(*rightId);
      if (left == clusterIndices.end() || right == clusterIndices.end())
        return std::nullopt;
      problem.warpGroupConflicts.push_back({left->second, right->second});
    }
  } else if (root->get("warp_group_conflicts")) {
    return std::nullopt;
  }

  return problem;
}

struct Candidate {
  CandidateStatus status = CandidateStatus::Error;
  std::vector<int64_t> cycles;
  std::vector<int64_t> warpGroups;
  int64_t usedWGs = 0;
  int64_t objective = 0;
  std::optional<int64_t> loweringObjective;
  std::string reason;
  BudgetExhausted budgetExhausted = BudgetExhausted::None;
  std::optional<uint64_t> rlimitUsed;
};

struct IssueItem {
  Z3_ast phase;
  Z3_ast presence;
  int64_t duration;
  std::string pipeline;
  std::optional<size_t> cluster;
};

struct EventModel {
  size_t templateIndex;
  size_t eventIndex;
  size_t ownerCluster;
  Z3_ast cycle;
  Z3_ast phase;
  Z3_ast presence;
};

static Z3_ast circularSeparation(const Z3ConstraintEncoder &encoder,
                                 Z3_ast leftPhase, int64_t leftDuration,
                                 Z3_ast rightPhase, int64_t rightDuration,
                                 int64_t ii) {
  Z3_ast iiValue = encoder.intValue(ii);
  Z3_ast delta = Z3_mk_mod(
      encoder.getContext(),
      encoder.add(encoder.sub(rightPhase, leftPhase), iiValue), iiValue);
  return encoder.conjunction(
      {Z3_mk_ge(encoder.getContext(), delta, encoder.intValue(leftDuration)),
       Z3_mk_le(encoder.getContext(), delta,
                encoder.intValue(ii - rightDuration))});
}

static bool evaluateInteger(Z3_context context, Z3_model model, Z3_ast term,
                            int64_t &result) {
  Z3_ast value = nullptr;
  return Z3_model_eval(context, model, term, true, &value) && value &&
         Z3_get_numeral_int64(context, value, &result);
}

static Candidate solveProblem(const Problem &problem,
                              std::chrono::steady_clock::time_point deadline) {
  Candidate result;
  if (deadlineExpired(deadline)) {
    result.status = CandidateStatus::Unknown;
    result.reason = "timeout";
    result.budgetExhausted = BudgetExhausted::WallTime;
    return result;
  }

  z3ErrorSeen = false;
  Context ownedContext;
  Z3_context context = ownedContext.get();
  if (!context)
    return result;
  Solver ownedSolver(context);
  Z3_solver solver = ownedSolver.get();
  if (!solver)
    return result;
  Z3ConstraintEncoder encoder(context, solver);

  Z3_ast zero = encoder.intValue(0);
  Z3_ast iiValue = encoder.intValue(problem.ii);
  std::vector<Z3_ast> cycles;
  std::vector<Z3_ast> stages;
  std::vector<Z3_ast> phases;
  cycles.reserve(problem.nodes.size());
  stages.reserve(problem.nodes.size());
  phases.reserve(problem.nodes.size());
  for (size_t index = 0; index < problem.nodes.size(); ++index) {
    const Node &node = problem.nodes[index];
    Z3_ast cycle = encoder.variable("v2_cycle_" + std::to_string(index));
    Z3_ast stage = Z3_mk_div(context, cycle, iiValue);
    Z3_ast phase = Z3_mk_mod(context, cycle, iiValue);
    cycles.push_back(cycle);
    stages.push_back(stage);
    phases.push_back(phase);
    int64_t fixedStage = node.fixedCycle / problem.ii;
    __int128 lower = static_cast<__int128>(fixedStage) * problem.ii;
    __int128 upper = lower + problem.ii - 1;
    if (upper > std::numeric_limits<int64_t>::max())
      return result;
    encoder.assertFormula(Z3_mk_ge(
        context, cycle, encoder.intValue(static_cast<int64_t>(lower))));
    encoder.assertFormula(Z3_mk_le(
        context, cycle, encoder.intValue(static_cast<int64_t>(upper))));
    encoder.assertFormula(
        Z3_mk_eq(context, stage, encoder.intValue(fixedStage)));
    if (problem.mode == SolveMode::Partition)
      encoder.assertFormula(
          Z3_mk_eq(context, cycle, encoder.intValue(node.fixedCycle)));
  }
  if (problem.canonicalRoot)
    encoder.assertFormula(
        Z3_mk_eq(context, cycles[*problem.canonicalRoot], zero));

  const size_t clusterCount = problem.clusters.size();
  std::vector<Z3_ast> warpGroups;
  warpGroups.reserve(clusterCount);
  for (size_t index = 0; index < clusterCount; ++index) {
    Z3_ast wg = encoder.variable("v2_wg_" + std::to_string(index));
    warpGroups.push_back(wg);
    encoder.assertFormula(Z3_mk_ge(context, wg, zero));
    encoder.assertFormula(Z3_mk_lt(
        context, wg, encoder.intValue(static_cast<int64_t>(clusterCount))));
  }
  encoder.assertFormula(Z3_mk_eq(context, warpGroups.front(), zero));
  Z3_ast prefixMaximum = warpGroups.front();
  for (size_t index = 1; index < clusterCount; ++index) {
    encoder.assertFormula(
        Z3_mk_le(context, warpGroups[index],
                 encoder.add(prefixMaximum, encoder.intValue(1))));
    Z3_ast nextMaximum =
        encoder.variable("v2_prefix_max_" + std::to_string(index));
    encoder.assertFormula(
        Z3_mk_eq(context, nextMaximum,
                 encoder.maximum(prefixMaximum, warpGroups[index])));
    prefixMaximum = nextMaximum;
  }
  Z3_ast usedWGs = encoder.add(prefixMaximum, encoder.intValue(1));
  encoder.assertFormula(
      Z3_mk_le(context, usedWGs, encoder.intValue(problem.maxWGs)));

  auto sameWG = [&](size_t left, size_t right) {
    return Z3_mk_eq(context, warpGroups[left], warpGroups[right]);
  };
  for (const auto &[left, right] : problem.warpGroupConflicts)
    encoder.assertFormula(Z3_mk_not(context, sameWG(left, right)));

  for (const Edge &edge : problem.edges) {
    const Node &srcNode = problem.nodes[edge.src];
    const Node &dstNode = problem.nodes[edge.dst];
    std::optional<size_t> srcCluster = srcNode.cluster;
    std::optional<size_t> dstCluster = dstNode.cluster;
    if (srcCluster && dstCluster && *srcCluster != *dstCluster) {
      // Tensor-core results consumed by CUDA/SFU must cross warp groups so
      // software readers cannot miss an mbarrier parity phase.
      if ((srcNode.pipeline == "TC" || srcNode.pipeline == "MFMA") &&
          (dstNode.pipeline == "CUDA" || dstNode.pipeline == "SFU"))
        encoder.assertFormula(
            Z3_mk_not(context, sameWG(*srcCluster, *dstCluster)));
      if (edge.distance > 0 && edge.roundTrip > 0)
        encoder.assertFormula(sameWG(*srcCluster, *dstCluster));
    }

    if (problem.mode != SolveMode::Joint)
      continue;
    int64_t latency =
        edge.distance == 0 ? std::max<int64_t>(edge.latency, 1) : edge.latency;
    Z3_ast distance = encoder.mul(encoder.intValue(edge.distance), iiValue);
    Z3_ast base = encoder.sub(
        encoder.add(cycles[edge.src], encoder.intValue(latency)), distance);
    encoder.assertFormula(Z3_mk_ge(context, cycles[edge.dst], base));
    if (edge.distance == 0)
      encoder.assertFormula(
          Z3_mk_le(context, stages[edge.src], stages[edge.dst]));
    if (srcCluster && dstCluster && *srcCluster != *dstCluster &&
        edge.roundTrip > 0 && !(edge.distance > 0 && edge.roundTrip > 0)) {
      Z3_ast cross = Z3_mk_not(context, sameWG(*srcCluster, *dstCluster));
      Z3_ast withRoundTrip =
          encoder.add(base, encoder.intValue(edge.roundTrip));
      encoder.assertFormula(encoder.implies(
          cross, Z3_mk_ge(context, cycles[edge.dst], withRoundTrip)));
    }
  }

  std::vector<Z3_ast> templatePresence;
  templatePresence.reserve(problem.loweringTemplates.size());
  for (const LoweringTemplate &loweringTemplate : problem.loweringTemplates) {
    Z3_ast same =
        sameWG(loweringTemplate.srcCluster, loweringTemplate.dstCluster);
    if (loweringTemplate.relation == Relation::Always)
      templatePresence.push_back(Z3_mk_true(context));
    else if (loweringTemplate.relation == Relation::SameWG)
      templatePresence.push_back(same);
    else
      templatePresence.push_back(Z3_mk_not(context, same));
  }

  using EventKey = std::pair<size_t, int>;
  std::map<EventKey, std::vector<std::pair<size_t, size_t>>> eventGroups;
  for (size_t templateIndex = 0;
       templateIndex < problem.loweringTemplates.size(); ++templateIndex) {
    const LoweringTemplate &loweringTemplate =
        problem.loweringTemplates[templateIndex];
    for (size_t eventIndex = 0; eventIndex < loweringTemplate.events.size();
         ++eventIndex) {
      const LoweringEvent &event = loweringTemplate.events[eventIndex];
      eventGroups[{event.anchor, static_cast<int>(event.placement)}].push_back(
          {templateIndex, eventIndex});
    }
  }

  std::vector<EventModel> eventModels;
  std::vector<IssueItem> issueItems;
  issueItems.reserve(problem.nodes.size());
  for (size_t nodeIndex = 0; nodeIndex < problem.nodes.size(); ++nodeIndex) {
    const Node &node = problem.nodes[nodeIndex];
    int64_t duration = std::max<int64_t>(node.duration, 1);
    Z3_ast present = Z3_mk_true(context);
    if (duration > problem.ii)
      encoder.assertFormula(Z3_mk_false(context));
    issueItems.push_back(IssueItem{phases[nodeIndex], present, duration,
                                   node.pipeline, node.cluster});
  }

  for (auto &entry : eventGroups) {
    auto &members = entry.second;
    llvm::sort(members, [&](const auto &left, const auto &right) {
      const LoweringTemplate &leftTemplate =
          problem.loweringTemplates[left.first];
      const LoweringTemplate &rightTemplate =
          problem.loweringTemplates[right.first];
      return std::tie(leftTemplate.id, leftTemplate.events[left.second].id) <
             std::tie(rightTemplate.id, rightTemplate.events[right.second].id);
    });
    for (size_t memberIndex = 0; memberIndex < members.size(); ++memberIndex) {
      auto [templateIndex, eventIndex] = members[memberIndex];
      const LoweringTemplate &loweringTemplate =
          problem.loweringTemplates[templateIndex];
      const LoweringEvent &event = loweringTemplate.events[eventIndex];
      std::vector<Z3_ast> activeDurations;
      if (event.placement == Placement::Before) {
        for (size_t index = memberIndex; index < members.size(); ++index) {
          auto [otherTemplateIndex, otherEventIndex] = members[index];
          const LoweringEvent &otherEvent =
              problem.loweringTemplates[otherTemplateIndex]
                  .events[otherEventIndex];
          activeDurations.push_back(
              Z3_mk_ite(context, templatePresence[otherTemplateIndex],
                        encoder.intValue(otherEvent.duration), zero));
        }
      } else {
        for (size_t index = 0; index < memberIndex; ++index) {
          auto [otherTemplateIndex, otherEventIndex] = members[index];
          const LoweringEvent &otherEvent =
              problem.loweringTemplates[otherTemplateIndex]
                  .events[otherEventIndex];
          activeDurations.push_back(
              Z3_mk_ite(context, templatePresence[otherTemplateIndex],
                        encoder.intValue(otherEvent.duration), zero));
        }
      }
      Z3_ast offset = encoder.sum(activeDurations);
      Z3_ast eventCycle;
      if (event.placement == Placement::Before) {
        eventCycle = encoder.sub(cycles[event.anchor], offset);
      } else {
        eventCycle = encoder.add(
            encoder.add(cycles[event.anchor],
                        encoder.intValue(std::max<int64_t>(
                            problem.nodes[event.anchor].duration, 1))),
            offset);
      }
      Z3_ast eventPhase = Z3_mk_mod(context, eventCycle, iiValue);
      Z3_ast presence = templatePresence[templateIndex];
      size_t ownerCluster = event.owner == Owner::Src
                                ? loweringTemplate.srcCluster
                                : loweringTemplate.dstCluster;
      if (event.duration > problem.ii)
        encoder.assertFormula(encoder.implies(presence, Z3_mk_false(context)));
      eventModels.push_back(EventModel{templateIndex, eventIndex, ownerCluster,
                                       eventCycle, eventPhase, presence});
      if (event.duration > 0)
        issueItems.push_back(IssueItem{eventPhase, presence, event.duration,
                                       event.pipeline, ownerCluster});
    }
  }

  // Ordered for the same reason as v01's nodesByPipeline: iteration order here
  // is assertion order.
  std::map<std::string, std::vector<size_t>> byPipeline;
  for (size_t index = 0; index < issueItems.size(); ++index)
    if (issueItems[index].pipeline != "NONE")
      byPipeline[issueItems[index].pipeline].push_back(index);
  for (const auto &entry : byPipeline) {
    const std::vector<size_t> &members = entry.second;
    for (size_t leftIndex = 0; leftIndex < members.size(); ++leftIndex) {
      const IssueItem &left = issueItems[members[leftIndex]];
      for (size_t rightIndex = leftIndex + 1; rightIndex < members.size();
           ++rightIndex) {
        const IssueItem &right = issueItems[members[rightIndex]];
        Z3_ast both = encoder.conjunction({left.presence, right.presence});
        encoder.assertFormula(encoder.implies(
            both, circularSeparation(encoder, left.phase, left.duration,
                                     right.phase, right.duration, problem.ii)));
      }
    }
  }

  if (problem.mode == SolveMode::Joint) {
    for (size_t leftIndex = 0; leftIndex < issueItems.size(); ++leftIndex) {
      const IssueItem &left = issueItems[leftIndex];
      if (!left.cluster)
        continue;
      for (size_t rightIndex = leftIndex + 1; rightIndex < issueItems.size();
           ++rightIndex) {
        const IssueItem &right = issueItems[rightIndex];
        if (!right.cluster)
          continue;
        Z3_ast same = sameWG(*left.cluster, *right.cluster);
        Z3_ast active =
            encoder.conjunction({left.presence, right.presence, same});
        encoder.assertFormula(encoder.implies(
            active,
            circularSeparation(encoder, left.phase, left.duration, right.phase,
                               right.duration, problem.ii)));
      }
    }
  }

  std::set<std::tuple<size_t, size_t, int64_t>> seenChannels;
  std::vector<Z3_ast> channelCharges;
  for (const Edge &edge : problem.edges) {
    if (edge.channelBytes <= 0 || !edge.srcCluster || !edge.dstCluster ||
        *edge.srcCluster == *edge.dstCluster)
      continue;
    auto channel = std::make_tuple(edge.src, edge.dst, edge.srcResultIndex);
    if (!seenChannels.insert(channel).second)
      continue;
    Z3_ast cross =
        Z3_mk_not(context, sameWG(*edge.srcCluster, *edge.dstCluster));
    channelCharges.push_back(
        Z3_mk_ite(context, cross, encoder.intValue(edge.channelBytes), zero));
  }

  std::vector<Z3_ast> smemCharges;
  std::vector<Z3_ast> tmemCharges;
  std::vector<Z3_ast> smemDepths;
  for (size_t bufferIndex = 0; bufferIndex < problem.buffers.size();
       ++bufferIndex) {
    const Buffer &buffer = problem.buffers[bufferIndex];
    bool modelSmem = problem.mode == SolveMode::Joint && buffer.kind == "smem";
    if (!modelSmem && buffer.kind != "tmem")
      continue;
    Z3_ast lastEnd = cycles[buffer.producer];
    for (const BufferConsumer &consumer : buffer.consumers) {
      Z3_ast offset = encoder.add(
          encoder.intValue(consumer.latency),
          encoder.mul(encoder.intValue(consumer.distance), iiValue));
      lastEnd =
          encoder.maximum(lastEnd, encoder.add(cycles[consumer.node], offset));
    }
    Z3_ast lifetime = encoder.sub(lastEnd, cycles[buffer.producer]);
    Z3_ast computedDepth =
        encoder.add(Z3_mk_div(context, lifetime, iiValue), encoder.intValue(1));
    Z3_ast depth =
        encoder.maximum(computedDepth, encoder.intValue(buffer.minCount));
    Z3_ast charge = encoder.mul(encoder.intValue(buffer.sizeBytes), depth);
    if (modelSmem) {
      smemCharges.push_back(charge);
      smemDepths.push_back(depth);
    } else {
      tmemCharges.push_back(charge);
    }
  }
  int64_t fixedSmem = problem.mode == SolveMode::Joint ? problem.fixedSmem
                                                       : problem.committedSmem;
  std::vector<Z3_ast> allSmemCharges = smemCharges;
  allSmemCharges.insert(allSmemCharges.end(), channelCharges.begin(),
                        channelCharges.end());
  encoder.assertFormula(Z3_mk_le(
      context,
      encoder.add(encoder.intValue(fixedSmem), encoder.sum(allSmemCharges)),
      encoder.intValue(problem.smemBudget)));
  encoder.assertFormula(Z3_mk_le(context, encoder.sum(tmemCharges),
                                 encoder.intValue(problem.tmemBudgetBytes)));

  std::vector<Z3_ast> registerFootprints;
  for (size_t group = 0; group < clusterCount; ++group) {
    Z3_ast requiredWarps = zero;
    for (size_t cluster = 0; cluster < clusterCount; ++cluster) {
      Z3_ast assigned = Z3_mk_eq(context, warpGroups[cluster],
                                 encoder.intValue(static_cast<int64_t>(group)));
      Z3_ast demand =
          Z3_mk_ite(context, assigned,
                    encoder.intValue(problem.clusters[cluster].minWarps), zero);
      requiredWarps = encoder.maximum(requiredWarps, demand);
    }
    encoder.assertFormula(
        Z3_mk_le(context, requiredWarps, encoder.intValue(8)));
    Z3_ast roundedWarps = Z3_mk_ite(
        context, Z3_mk_le(context, requiredWarps, zero), zero,
        Z3_mk_ite(
            context, Z3_mk_le(context, requiredWarps, encoder.intValue(1)),
            encoder.intValue(1),
            Z3_mk_ite(
                context, Z3_mk_le(context, requiredWarps, encoder.intValue(2)),
                encoder.intValue(2),
                Z3_mk_ite(context,
                          Z3_mk_le(context, requiredWarps, encoder.intValue(4)),
                          encoder.intValue(4), encoder.intValue(8)))));
    Z3_ast footprint = encoder.intValue(problem.warpFootprint[8]);
    for (int warpCount : {4, 2, 1, 0})
      footprint = Z3_mk_ite(
          context, Z3_mk_eq(context, roundedWarps, encoder.intValue(warpCount)),
          encoder.intValue(problem.warpFootprint[warpCount]), footprint);
    registerFootprints.push_back(footprint);
  }
  Z3_ast totalRegisters =
      encoder.add(encoder.intValue(problem.defaultWGFootprint),
                  encoder.sum(registerFootprints));
  if (problem.regBudget)
    encoder.assertFormula(Z3_mk_le(context, totalRegisters,
                                   encoder.intValue(*problem.regBudget)));

  Z3_ast deficit = encoder.maximum(
      encoder.sub(totalRegisters, encoder.intValue(problem.smRegs)), zero);
  Z3_ast residual = encoder.maximum(
      encoder.sub(deficit, encoder.intValue(problem.defaultSlack)), zero);

  std::vector<Z3_ast> primaryTerms;
  std::optional<Z3_ast> loweringObjective;
  if (problem.mode == SolveMode::Partition) {
    for (size_t leftCluster = 0; leftCluster < clusterCount; ++leftCluster) {
      for (size_t rightCluster = leftCluster + 1; rightCluster < clusterCount;
           ++rightCluster) {
        std::vector<Z3_ast> overlapTerms;
        for (size_t leftNodeIndex : problem.clusters[leftCluster].nodes) {
          const Node &leftNode = problem.nodes[leftNodeIndex];
          for (size_t rightNodeIndex : problem.clusters[rightCluster].nodes) {
            const Node &rightNode = problem.nodes[rightNodeIndex];
            if (leftNode.pipeline == rightNode.pipeline)
              continue;
            int64_t overlap = modularOverlap(
                leftNode.fixedCycle, leftNode.duration, rightNode.fixedCycle,
                rightNode.duration, problem.ii);
            if (overlap == 0)
              continue;
            overlapTerms.push_back(
                encoder.mul(encoder.intValue(overlap),
                            encoder.intValue(std::max(leftNode.frequency,
                                                      rightNode.frequency))));
          }
        }
        if (!overlapTerms.empty()) {
          Z3_ast cost = encoder.mul(encoder.intValue(kPartitionCycleWeight),
                                    encoder.sum(overlapTerms));
          primaryTerms.push_back(Z3_mk_ite(
              context, sameWG(leftCluster, rightCluster), cost, zero));
        }
      }
    }

    for (const Edge &edge : problem.edges) {
      if (!edge.srcCluster || !edge.dstCluster ||
          *edge.srcCluster == *edge.dstCluster ||
          (edge.distance > 0 && edge.roundTrip > 0))
        continue;
      Z3_ast cross =
          Z3_mk_not(context, sameWG(*edge.srcCluster, *edge.dstCluster));
      if (edge.roundTrip > 0) {
        __int128 slack =
            static_cast<__int128>(problem.nodes[edge.dst].fixedCycle) -
            problem.nodes[edge.src].fixedCycle - edge.latency +
            static_cast<__int128>(edge.distance) * problem.ii;
        __int128 shortfall = static_cast<__int128>(edge.roundTrip) -
                             std::max<__int128>(0, slack);
        if (shortfall > 0) {
          Z3_ast cost = encoder.mul(
              encoder.intValue(kPartitionCycleWeight),
              encoder.mul(encoder.intValue(static_cast<int64_t>(shortfall)),
                          encoder.intValue(edge.frequency)));
          primaryTerms.push_back(Z3_mk_ite(context, cross, cost, zero));
        }
      }
      Z3_ast issueCost = encoder.mul(encoder.intValue(kPartitionCycleWeight),
                                     encoder.intValue(edge.crossIssue));
      primaryTerms.push_back(Z3_mk_ite(context, cross, issueCost, zero));
    }

    for (size_t group = 0; group < clusterCount; ++group) {
      std::vector<Z3_ast> assignedDurations;
      for (size_t cluster = 0; cluster < clusterCount; ++cluster) {
        std::vector<Z3_ast> clusterDurations;
        for (size_t nodeIndex : problem.clusters[cluster].nodes) {
          const Node &node = problem.nodes[nodeIndex];
          clusterDurations.push_back(
              encoder.mul(encoder.intValue(std::max<int64_t>(node.duration, 0)),
                          encoder.intValue(node.frequency)));
        }
        Z3_ast assigned =
            Z3_mk_eq(context, warpGroups[cluster],
                     encoder.intValue(static_cast<int64_t>(group)));
        assignedDurations.push_back(
            Z3_mk_ite(context, assigned, encoder.sum(clusterDurations), zero));
      }
      Z3_ast occupancy = encoder.sum(assignedDurations);
      Z3_ast excess = encoder.maximum(
          encoder.sub(occupancy, encoder.intValue(problem.ii)), zero);
      primaryTerms.push_back(
          encoder.mul(encoder.intValue(kPartitionCycleWeight), excess));
    }
    primaryTerms.push_back(residual);
    primaryTerms.push_back(encoder.sub(zero, usedWGs));
  } else {
    Z3_ast maxStage = stages.front();
    for (size_t index = 1; index < stages.size(); ++index)
      maxStage = encoder.maximum(maxStage, stages[index]);

    std::vector<Z3_ast> recurrenceSpans;
    std::vector<Z3_ast> registerPressure;
    for (const Edge &edge : problem.edges) {
      if (edge.distance > 0 && edge.src != edge.dst)
        recurrenceSpans.push_back(
            encoder.sub(cycles[edge.src], cycles[edge.dst]));
      if (edge.distance == 0)
        registerPressure.push_back(
            encoder.sub(cycles[edge.dst], cycles[edge.src]));
    }

    std::set<std::tuple<size_t, size_t, int64_t>> loweringEdges;
    for (const LoweringTemplate &loweringTemplate : problem.loweringTemplates) {
      if (!loweringTemplate.events.empty())
        loweringEdges.insert({loweringTemplate.srcNode,
                              loweringTemplate.dstNode,
                              loweringTemplate.events.front().distance});
    }
    std::vector<Z3_ast> issueCosts;
    for (const Edge &edge : problem.edges) {
      if (!edge.srcCluster || !edge.dstCluster ||
          *edge.srcCluster == *edge.dstCluster ||
          (edge.distance > 0 && edge.roundTrip > 0) ||
          loweringEdges.count({edge.src, edge.dst, edge.distance}) != 0)
        continue;
      Z3_ast cross =
          Z3_mk_not(context, sameWG(*edge.srcCluster, *edge.dstCluster));
      Z3_ast cost = encoder.mul(encoder.intValue(kJointCrossIssueWeight),
                                encoder.intValue(edge.crossIssue));
      issueCosts.push_back(Z3_mk_ite(context, cross, cost, zero));
    }

    Z3_ast smemTotal = encoder.add(encoder.intValue(problem.fixedSmem),
                                   encoder.sum(allSmemCharges));
    primaryTerms.push_back(
        encoder.mul(encoder.intValue(kJointMaxStageWeight), maxStage));
    primaryTerms.push_back(encoder.mul(encoder.intValue(kJointDepthWeight),
                                       encoder.sum(smemDepths)));
    primaryTerms.push_back(encoder.mul(encoder.intValue(kJointRecurrenceWeight),
                                       encoder.sum(recurrenceSpans)));
    primaryTerms.push_back(
        encoder.mul(encoder.intValue(kJointRegisterPressureWeight),
                    encoder.sum(registerPressure)));
    primaryTerms.push_back(smemTotal);
    primaryTerms.insert(primaryTerms.end(), issueCosts.begin(),
                        issueCosts.end());
    primaryTerms.push_back(
        encoder.mul(encoder.intValue(kJointRegisterResidualWeight), residual));
    primaryTerms.push_back(encoder.sub(zero, usedWGs));

    std::vector<std::vector<size_t>> successors(problem.nodes.size());
    for (const Edge &edge : problem.edges)
      if (edge.distance == 0)
        successors[edge.src].push_back(edge.dst);
    std::vector<std::vector<bool>> reachable(
        problem.nodes.size(), std::vector<bool>(problem.nodes.size(), false));
    for (size_t start = 0; start < problem.nodes.size(); ++start) {
      std::vector<size_t> pending = successors[start];
      while (!pending.empty()) {
        size_t node = pending.back();
        pending.pop_back();
        if (reachable[start][node])
          continue;
        reachable[start][node] = true;
        pending.insert(pending.end(), successors[node].begin(),
                       successors[node].end());
      }
    }

    std::vector<Z3_ast> inversionTerms;
    for (size_t templateIndex = 0;
         templateIndex < problem.loweringTemplates.size(); ++templateIndex) {
      const LoweringTemplate &loweringTemplate =
          problem.loweringTemplates[templateIndex];
      for (const LoweringEvent &event : loweringTemplate.events) {
        if (event.kind != "wait" || !event.blocking)
          continue;
        size_t ownerCluster = event.owner == Owner::Src
                                  ? loweringTemplate.srcCluster
                                  : loweringTemplate.dstCluster;
        int64_t anchorStage =
            problem.nodes[event.anchor].fixedCycle / problem.ii;
        for (size_t other = 0; other < problem.nodes.size(); ++other) {
          const Node &otherNode = problem.nodes[other];
          if (other == event.anchor || !otherNode.cluster ||
              (otherNode.pipeline != "CUDA" && otherNode.pipeline != "SFU") ||
              otherNode.duration <= 0 ||
              otherNode.fixedCycle / problem.ii != anchorStage ||
              reachable[event.anchor][other] || reachable[other][event.anchor])
            continue;
          std::vector<Z3_ast> conditions{
              templatePresence[templateIndex],
              Z3_mk_lt(context, cycles[event.anchor], cycles[other])};
          if (ownerCluster != *otherNode.cluster)
            conditions.push_back(sameWG(ownerCluster, *otherNode.cluster));
          Z3_ast weight = encoder.mul(
              encoder.intValue(otherNode.duration),
              encoder.intValue(std::max<int64_t>(1, otherNode.frequency)));
          inversionTerms.push_back(Z3_mk_ite(
              context, encoder.conjunction(conditions), weight, zero));
        }
      }
    }
    if (!inversionTerms.empty())
      loweringObjective = encoder.sum(inversionTerms);
  }
  Z3_ast primaryObjective = encoder.sum(primaryTerms);
  Optimize ownedOptimize(context);
  Z3_optimize optimize = ownedOptimize.get();
  if (!optimize)
    return result;
  Z3_ast_vector assertions = Z3_solver_get_assertions(context, solver);
  if (!assertions)
    return result;
  unsigned assertionCount = Z3_ast_vector_size(context, assertions);
  for (unsigned index = 0; index < assertionCount; ++index)
    Z3_optimize_assert(context, optimize,
                       Z3_ast_vector_get(context, assertions, index));
  unsigned primaryHandle =
      Z3_optimize_minimize(context, optimize, primaryObjective);
  std::optional<unsigned> loweringHandle;
  if (loweringObjective)
    loweringHandle =
        Z3_optimize_minimize(context, optimize, *loweringObjective);
  auto timeoutMs = remainingTimeoutMs(deadline);
  if (!timeoutMs) {
    result.status = CandidateStatus::Unknown;
    result.reason = "timeout";
    result.budgetExhausted = BudgetExhausted::WallTime;
    return result;
  }
  if (z3ErrorSeen ||
      !configureOptimize(context, optimize, *timeoutMs, problem.budget))
    return result;

  Z3_lbool status = Z3_optimize_check(context, optimize, 0, nullptr);
  if (z3ErrorSeen)
    return result;
  // The counter lives on the context, so the assertion-accumulating solver
  // reports the work the optimize object just did.
  result.rlimitUsed = readRLimitCount(context, solver);
  if (status == Z3_L_FALSE) {
    result.status = CandidateStatus::Unsat;
    return result;
  }
  if (status == Z3_L_UNDEF) {
    result.status = CandidateStatus::Unknown;
    if (const char *reason = Z3_optimize_get_reason_unknown(context, optimize))
      result.reason = reason;
    if (result.reason.empty())
      result.reason = "unknown";
    result.budgetExhausted = attributeBudgetStop(deadline, problem.budget);
    return result;
  }
  if (status != Z3_L_TRUE)
    return result;

  Z3_model rawModel = Z3_optimize_get_model(context, optimize);
  if (!rawModel)
    return result;
  Model model(context, rawModel);
  if (!evaluateInteger(context, model.get(), primaryObjective,
                       result.objective))
    return Candidate{};
  if (loweringObjective) {
    int64_t value = 0;
    if (!evaluateInteger(context, model.get(), *loweringObjective, value))
      return Candidate{};
    result.loweringObjective = value;
  }

  auto objectiveProven = [&](unsigned handle, int64_t value) {
    Z3_ast lower = Z3_optimize_get_lower(context, optimize, handle);
    Z3_ast upper = Z3_optimize_get_upper(context, optimize, handle);
    int64_t lowerValue = 0;
    int64_t upperValue = 0;
    return lower && upper &&
           Z3_get_numeral_int64(context, lower, &lowerValue) &&
           Z3_get_numeral_int64(context, upper, &upperValue) &&
           lowerValue == value && upperValue == value;
  };
  if (!objectiveProven(primaryHandle, result.objective) ||
      (loweringHandle &&
       !objectiveProven(*loweringHandle, *result.loweringObjective))) {
    // Z3_optimize_check returned SAT but left the objective's lower and upper
    // bounds apart, which only happens when the optimization loop was cut
    // short — so this is a budget stop too, not an incompleteness.
    result.status = CandidateStatus::Unknown;
    result.reason = "objective optimum not proven";
    result.budgetExhausted = attributeBudgetStop(deadline, problem.budget);
    return result;
  }

  result.cycles.reserve(cycles.size());
  result.warpGroups.reserve(warpGroups.size());
  for (Z3_ast cycle : cycles) {
    int64_t value = 0;
    if (!evaluateInteger(context, model.get(), cycle, value))
      return Candidate{};
    result.cycles.push_back(value);
  }
  for (Z3_ast wg : warpGroups) {
    int64_t value = 0;
    if (!evaluateInteger(context, model.get(), wg, value))
      return Candidate{};
    result.warpGroups.push_back(value);
  }
  if (!evaluateInteger(context, model.get(), usedWGs, result.usedWGs))
    return Candidate{};
  result.status = CandidateStatus::Sat;
  return result;
}

struct ConcreteEvent {
  int64_t id;
  int64_t cycle = 0;
  int64_t warpGroup = 0;
  int64_t streamOrder = 0;
  Placement placement;
};

static bool templateIsActive(const LoweringTemplate &loweringTemplate,
                             const std::vector<int64_t> &warpGroups) {
  if (loweringTemplate.relation == Relation::Always)
    return true;
  bool same = warpGroups[loweringTemplate.srcCluster] ==
              warpGroups[loweringTemplate.dstCluster];
  return loweringTemplate.relation == Relation::SameWG ? same : !same;
}

static std::optional<llvm::json::Object>
buildLoweringPlan(const Problem &problem, const Candidate &candidate) {
  std::vector<bool> active(problem.loweringTemplates.size(), false);
  std::vector<std::vector<ConcreteEvent>> planned(
      problem.loweringTemplates.size());
  using ConcreteKey = std::pair<size_t, int>;
  std::map<ConcreteKey, std::vector<std::pair<size_t, size_t>>> groups;
  for (size_t templateIndex = 0;
       templateIndex < problem.loweringTemplates.size(); ++templateIndex) {
    const LoweringTemplate &loweringTemplate =
        problem.loweringTemplates[templateIndex];
    active[templateIndex] =
        templateIsActive(loweringTemplate, candidate.warpGroups);
    if (!active[templateIndex])
      continue;
    planned[templateIndex].reserve(loweringTemplate.events.size());
    for (size_t eventIndex = 0; eventIndex < loweringTemplate.events.size();
         ++eventIndex) {
      const LoweringEvent &event = loweringTemplate.events[eventIndex];
      size_t ownerCluster = event.owner == Owner::Src
                                ? loweringTemplate.srcCluster
                                : loweringTemplate.dstCluster;
      planned[templateIndex].push_back(ConcreteEvent{
          event.id, 0, candidate.warpGroups[ownerCluster], 0, event.placement});
      groups[{event.anchor, static_cast<int>(event.placement)}].push_back(
          {templateIndex, eventIndex});
    }
  }

  for (auto &entry : groups) {
    auto &members = entry.second;
    llvm::sort(members, [&](const auto &left, const auto &right) {
      const LoweringTemplate &leftTemplate =
          problem.loweringTemplates[left.first];
      const LoweringTemplate &rightTemplate =
          problem.loweringTemplates[right.first];
      return std::tie(leftTemplate.id, leftTemplate.events[left.second].id) <
             std::tie(rightTemplate.id, rightTemplate.events[right.second].id);
    });
    size_t anchor = entry.first.first;
    Placement placement = static_cast<Placement>(entry.first.second);
    __int128 cursor = candidate.cycles[anchor];
    if (placement == Placement::Before) {
      for (auto [templateIndex, eventIndex] : members)
        cursor -= problem.loweringTemplates[templateIndex]
                      .events[eventIndex]
                      .duration;
    } else {
      cursor += std::max<int64_t>(problem.nodes[anchor].duration, 1);
    }
    for (auto [templateIndex, eventIndex] : members) {
      if (cursor < std::numeric_limits<int>::min() ||
          cursor > std::numeric_limits<int>::max())
        return std::nullopt;
      planned[templateIndex][eventIndex].cycle = static_cast<int64_t>(cursor);
      cursor +=
          problem.loweringTemplates[templateIndex].events[eventIndex].duration;
    }
  }

  std::vector<std::pair<std::tuple<int64_t, int64_t, int, int64_t, int64_t>,
                        ConcreteEvent *>>
      stream;
  for (size_t templateIndex = 0; templateIndex < planned.size();
       ++templateIndex) {
    for (size_t eventIndex = 0; eventIndex < planned[templateIndex].size();
         ++eventIndex) {
      ConcreteEvent &event = planned[templateIndex][eventIndex];
      const LoweringTemplate &loweringTemplate =
          problem.loweringTemplates[templateIndex];
      stream.push_back({{event.warpGroup, event.cycle,
                         event.placement == Placement::Before ? 0 : 2,
                         loweringTemplate.id, event.id},
                        &event});
    }
  }
  llvm::sort(stream, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });
  std::map<int64_t, int64_t> nextOrder;
  for (auto &entry : stream)
    entry.second->streamOrder = nextOrder[entry.second->warpGroup]++;

  llvm::json::Array templatePlans;
  templatePlans.reserve(problem.loweringTemplates.size());
  for (size_t templateIndex = 0;
       templateIndex < problem.loweringTemplates.size(); ++templateIndex) {
    llvm::json::Array eventPlans;
    eventPlans.reserve(planned[templateIndex].size());
    for (const ConcreteEvent &event : planned[templateIndex]) {
      eventPlans.push_back(llvm::json::Object{
          {"id", event.id},
          {"cycle", event.cycle},
          {"wg", event.warpGroup},
          {"stream_order", event.streamOrder},
      });
    }
    templatePlans.push_back(llvm::json::Object{
        {"id", problem.loweringTemplates[templateIndex].id},
        {"active", static_cast<bool>(active[templateIndex])},
        {"events", std::move(eventPlans)},
    });
  }
  return llvm::json::Object{{"version", kLoweringPlanVersion},
                            {"templates", std::move(templatePlans)}};
}

/// Reports the budget that was in force alongside what it cost, so a caller
/// (or a calibration test) can see the headroom without knowing the default.
static llvm::json::Object makeStats(const Problem &problem,
                                    const Candidate &candidate) {
  return llvm::json::Object{
      {"rlimit", static_cast<int64_t>(problem.budget.rlimit)},
      {"rlimit_used", static_cast<int64_t>(candidate.rlimitUsed.value_or(0))}};
}

static std::optional<int64_t> bufferDepth(const Problem &problem,
                                          const Candidate &candidate,
                                          const Buffer &buffer) {
  __int128 producerCycle = candidate.cycles[buffer.producer];
  __int128 lastEnd = producerCycle;
  for (const BufferConsumer &consumer : buffer.consumers) {
    __int128 end = static_cast<__int128>(candidate.cycles[consumer.node]) +
                   consumer.latency +
                   static_cast<__int128>(consumer.distance) * problem.ii;
    lastEnd = std::max(lastEnd, end);
  }
  __int128 depth = (lastEnd - producerCycle) / problem.ii + 1;
  depth = std::max(depth, static_cast<__int128>(buffer.minCount));
  if (depth <= 0 || depth > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(depth);
}

static FailureOr<std::string> makeSuccess(const Problem &problem,
                                          const Candidate &candidate) {
  llvm::json::Object warpGroupValues;
  for (size_t index = 0; index < problem.clusters.size(); ++index)
    warpGroupValues[std::to_string(problem.clusters[index].id)] =
        candidate.warpGroups[index];

  llvm::json::Object cycleValues;
  for (size_t index = 0; index < problem.nodes.size(); ++index) {
    if (candidate.cycles[index] < std::numeric_limits<int>::min() ||
        candidate.cycles[index] > std::numeric_limits<int>::max())
      return failure();
    cycleValues[std::to_string(problem.nodes[index].id)] =
        candidate.cycles[index];
  }

  llvm::json::Object depthValues;
  for (const Buffer &buffer : problem.buffers) {
    if (buffer.kind != "smem")
      continue;
    auto depth = bufferDepth(problem, candidate, buffer);
    if (!depth)
      return failure();
    depthValues[std::to_string(buffer.id)] = *depth;
  }

  auto loweringPlan = buildLoweringPlan(problem, candidate);
  if (!loweringPlan)
    return failure();
  llvm::json::Object solution{
      {"version", kSchemaVersion},
      {"status", "ok"},
      {"wg", std::move(warpGroupValues)},
      {"cycles", std::move(cycleValues)},
      {"used_wgs", candidate.usedWGs},
      {"objective", candidate.objective},
      {"buffer_depths", std::move(depthValues)},
      {"lowering_plan", std::move(*loweringPlan)},
      // See v01::makeStats for why the deterministic cost is always reported.
      {"stats", makeStats(problem, candidate)},
  };
  if (candidate.loweringObjective)
    solution["lowering_objective"] = *candidate.loweringObjective;
  return serialize(std::move(solution));
}

static FailureOr<std::string> run(llvm::StringRef problemJson) {
  std::optional<Problem> problem = parseProblem(problemJson);
  if (!problem)
    return failure();

  double timeoutMsDouble = std::ceil(problem->budget.timeLimitS * 1000.0);
  int64_t timeoutMs = static_cast<int64_t>(
      std::min<double>(timeoutMsDouble, std::numeric_limits<unsigned>::max()));
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(std::max<int64_t>(timeoutMs, 1));
  Candidate candidate = solveProblem(*problem, deadline);
  if (candidate.status == CandidateStatus::Sat) {
    FailureOr<std::string> solution = makeSuccess(*problem, candidate);
    if (failed(solution))
      return failure();
    return solution;
  }
  llvm::json::Object stats = makeStats(*problem, candidate);
  if (candidate.status == CandidateStatus::Unsat) {
    return serialize(llvm::json::Object{
        {"version", kSchemaVersion},
        {"status", "infeasible"},
        {"proven_unsat", true},
        {"backend_status", "INFEASIBLE"},
        {"message", "no feasible joint partition"},
        {"stats", std::move(stats)},
    });
  }
  if (candidate.status == CandidateStatus::Unknown) {
    std::string reason = candidate.reason.empty() ? "joint solve inconclusive"
                                                  : candidate.reason;
    return serialize(llvm::json::Object{
        {"version", kSchemaVersion},
        {"status", "inconclusive"},
        {"proven_unsat", false},
        {"backend_status", "UNKNOWN"},
        // See v01::run — `rlimit` is a reproducible property of the problem,
        // `walltime` is an environment signal that should never appear while
        // rlimit is the binding budget.
        {"budget_exhausted",
         budgetExhaustedName(candidate.budgetExhausted).str()},
        {"message", std::move(reason)},
        {"stats", std::move(stats)},
    });
  }
  return failure();
}

} // namespace v02

} // namespace

FailureOr<std::string> runZ3JointSolver(llvm::StringRef problemJson) {
  auto dispatch = llvm::json::parse(problemJson);
  if (!dispatch) {
    llvm::consumeError(dispatch.takeError());
    return failure();
  }
  auto *dispatchObject = dispatch->getAsObject();
  if (!dispatchObject)
    return failure();
  if (dispatchObject->getString("version") ==
      llvm::StringRef(v02::kSchemaVersion))
    return v02::run(problemJson);
  return v01::run(problemJson);
}

#endif

} // namespace mlir::triton::gpu
