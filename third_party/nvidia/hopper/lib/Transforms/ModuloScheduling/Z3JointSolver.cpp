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
#include <optional>
#include <string>
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
  double timeLimitS;
  bool streamingVL;
  std::optional<size_t> canonicalRoot;
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::vector<Buffer> buffers;
};

static bool getRequiredInteger(const llvm::json::Object &object,
                               llvm::StringRef key, int64_t &value) {
  auto parsed = object.getInteger(key);
  if (!parsed)
    return false;
  value = *parsed;
  return true;
}

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

  problem.timeLimitS = 20.0;
  if (root->get("time_limit_s")) {
    auto value = root->getNumber("time_limit_s");
    if (!value || !std::isfinite(*value) || *value <= 0.0)
      return std::nullopt;
    problem.timeLimitS = *value;
  }
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

struct CandidateResult {
  CandidateStatus status = CandidateStatus::Error;
  std::vector<int64_t> cycles;
  std::string reason;
  int64_t horizon = 0;
  __int128 objective = 0;
};

static bool deadlineExpired(std::chrono::steady_clock::time_point deadline) {
  return std::chrono::steady_clock::now() >= deadline;
}

static std::optional<unsigned>
remainingTimeoutMs(std::chrono::steady_clock::time_point deadline) {
  auto now = std::chrono::steady_clock::now();
  if (now >= deadline)
    return std::nullopt;
  auto remaining =
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
          .count();
  if (remaining <= 0)
    remaining = 1;
  return static_cast<unsigned>(
      std::min<int64_t>(remaining, std::numeric_limits<unsigned>::max()));
}

static bool configureTimeout(Z3_context context, Z3_solver solver,
                             unsigned timeoutMs) {
  Z3_params params = Z3_mk_params(context);
  if (!params)
    return false;
  Z3_params_inc_ref(context, params);
  Z3_params_set_uint(context, params, Z3_mk_string_symbol(context, "timeout"),
                     timeoutMs);
  Z3_solver_set_params(context, solver, params);
  Z3_params_dec_ref(context, params);
  return !z3ErrorSeen;
}

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

  std::unordered_map<std::string, std::vector<size_t>> nodesByPipeline;
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

  auto timeoutMs = remainingTimeoutMs(deadline);
  if (!timeoutMs) {
    result.status = CandidateStatus::Unknown;
    result.reason = "timeout";
    return result;
  }
  if (z3ErrorSeen || !configureTimeout(context, solver, *timeoutMs))
    return CandidateResult{};

  Z3_lbool status = Z3_solver_check(context, solver);
  if (z3ErrorSeen)
    return CandidateResult{};
  if (status == Z3_L_FALSE) {
    result.status = CandidateStatus::Unsat;
    return result;
  }
  if (status == Z3_L_UNDEF) {
    result.status = CandidateStatus::Unknown;
    if (const char *reason = Z3_solver_get_reason_unknown(context, solver))
      result.reason = reason;
    if (result.reason.empty())
      result.reason = "unknown";
    return result;
  }
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
      CandidateResult unknown;
      unknown.status = CandidateStatus::Unknown;
      unknown.reason = "timeout";
      return unknown;
    }
    if (z3ErrorSeen || !configureTimeout(context, solver, *timeoutMs)) {
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
      CandidateResult unknown;
      unknown.status = CandidateStatus::Unknown;
      if (const char *reason = Z3_solver_get_reason_unknown(context, solver))
        unknown.reason = reason;
      if (unknown.reason.empty())
        unknown.reason = "unknown";
      Z3_solver_pop(context, solver, 1);
      return unknown;
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
  return result;
}

static llvm::json::Array toJsonArray(const std::vector<int64_t> &values) {
  llvm::json::Array result;
  result.reserve(values.size());
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

static llvm::json::Object makeStats(const std::vector<int64_t> &tried,
                                    const std::vector<int64_t> &unsat,
                                    const std::vector<int64_t> &unknown) {
  return llvm::json::Object{{"iis_tried", toJsonArray(tried)},
                            {"unsat_iis", toJsonArray(unsat)},
                            {"unknown_iis", toJsonArray(unknown)}};
}

static std::string serialize(llvm::json::Object object) {
  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << llvm::json::Value(std::move(object));
  stream.flush();
  return output;
}

static llvm::json::Object makeSuccess(const Problem &problem, int64_t ii,
                                      const CandidateResult &candidate,
                                      const std::vector<int64_t> &tried,
                                      const std::vector<int64_t> &unsat,
                                      const std::vector<int64_t> &unknown) {
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
      {"stats", makeStats(tried, unsat, unknown)},
  };
}

} // namespace

FailureOr<std::string> runZ3JointSolver(llvm::StringRef problemJson) {

  std::optional<Problem> problem = parseProblem(problemJson);
  if (!problem)
    return failure();

  double timeoutMsDouble = std::ceil(problem->timeLimitS * 1000.0);
  int64_t timeoutMs = static_cast<int64_t>(std::min<double>(
      timeoutMsDouble, std::numeric_limits<int64_t>::max() / 2.0));
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(std::max<int64_t>(timeoutMs, 1));

  std::vector<int64_t> tried;
  std::vector<int64_t> unsat;
  std::vector<int64_t> unknown;
  for (int64_t ii = problem->minII;; ++ii) {
    tried.push_back(ii);
    CandidateResult candidate = solveCandidate(*problem, ii, deadline);
    if (candidate.status == CandidateStatus::Sat) {
      return serialize(
          makeSuccess(*problem, ii, candidate, tried, unsat, unknown));
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
          {"message", std::move(message)},
          {"diagnostic", std::move(diagnostic)},
          {"stats", makeStats(tried, unsat, unknown)},
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
      {"stats", makeStats(tried, unsat, unknown)},
  });
}

#endif

} // namespace mlir::triton::gpu
