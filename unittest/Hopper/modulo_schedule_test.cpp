#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/ModuloReservationTable.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/Z3JointSolver.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <climits>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace mlir::triton::gpu {
namespace {

ModuloScheduleResult makeLoopCarriedSchedule(int consumerCycle) {
  ModuloScheduleResult schedule;
  schedule.II = 825;
  schedule.nodeToCycle[0] = 1245;
  schedule.nodeToCycle[1] = consumerCycle;
  return schedule;
}

TEST(ModuloScheduleTest, RejectsViolatedLoopCarriedDependency) {
  auto schedule = makeLoopCarriedSchedule(588);
  llvm::SmallVector<DDGEdge> edges{{0, 1, 266, 1}};

  EXPECT_FALSE(
      isValidModuloSchedule(schedule.II, schedule.nodeToCycle, 2, edges));
}

TEST(ModuloScheduleTest, AcceptsLoopCarriedDependencyAtBoundary) {
  auto schedule = makeLoopCarriedSchedule(686);
  llvm::SmallVector<DDGEdge> edges{{0, 1, 266, 1}};

  EXPECT_TRUE(
      isValidModuloSchedule(schedule.II, schedule.nodeToCycle, 2, edges));
}

TEST(ModuloScheduleTest, RejectsMissingNodeAssignment) {
  auto schedule = makeLoopCarriedSchedule(686);
  schedule.nodeToCycle.erase(1);

  EXPECT_FALSE(isValidModuloSchedule(schedule.II, schedule.nodeToCycle, 2, {}));
}

TEST(ModuloScheduleTest, RepairsDependencyWithinOutgoingSlack) {
  auto schedule = makeLoopCarriedSchedule(588);
  llvm::SmallVector<DDGNode> nodes(2);
  nodes[0].idx = 0;
  nodes[0].pipeline = HWPipeline::CUDA;
  nodes[0].selfLatency = 128;
  nodes[1].idx = 1;
  nodes[1].pipeline = HWPipeline::TC;
  nodes[1].selfLatency = 30;
  llvm::SmallVector<DDGEdge> edges{{1, 0, 559, 0}, {0, 1, 266, 1}};

  EXPECT_TRUE(
      tryRepairModuloSchedule(schedule.II, schedule.nodeToCycle, nodes, edges));
  EXPECT_EQ(schedule.nodeToCycle.lookup(1), 686);
  EXPECT_TRUE(
      isValidModuloSchedule(schedule.II, schedule.nodeToCycle, 2, edges));
}

TEST(ModuloScheduleTest, RejectsRepairWhenLatestCycleIsOccupied) {
  auto schedule = makeLoopCarriedSchedule(588);
  schedule.nodeToCycle[2] = 686;
  llvm::SmallVector<DDGNode> nodes(3);
  nodes[0].idx = 0;
  nodes[1].idx = 1;
  nodes[1].pipeline = HWPipeline::TC;
  nodes[1].selfLatency = 30;
  nodes[2].idx = 2;
  nodes[2].pipeline = HWPipeline::TC;
  nodes[2].selfLatency = 1;
  llvm::SmallVector<DDGEdge> edges{{1, 0, 559, 0}, {0, 1, 266, 1}};

  EXPECT_FALSE(
      tryRepairModuloSchedule(schedule.II, schedule.nodeToCycle, nodes, edges));
}

// ── Schedule-backend selection ──────────────────────────────────────────────
//
// getActiveScheduleAlgo is a pure function of (forced argument, env var). It
// used to consult a process-global slot written by an RAII override, which two
// modules compiled concurrently in one process (triton.AsyncCompileMode with a
// ThreadPoolExecutor; pass_manager.run releases the GIL) could corrupt for each
// other. These tests pin the property that replaced it.

constexpr const char *kScheduleAlgoEnvVar = "TRITON_USE_MODULO_SCHEDULE";

/// Pins TRITON_USE_MODULO_SCHEDULE for one test and restores it on scope exit,
/// so a test can't leak its value into the rest of the binary. `value ==
/// nullptr` unsets the variable.
class ScopedScheduleAlgoEnv {
public:
  explicit ScopedScheduleAlgoEnv(const char *value) {
    if (const char *previous = std::getenv(kScheduleAlgoEnvVar))
      saved = std::string(previous);
    apply(value);
  }
  ~ScopedScheduleAlgoEnv() { apply(saved ? saved->c_str() : nullptr); }
  ScopedScheduleAlgoEnv(const ScopedScheduleAlgoEnv &) = delete;
  ScopedScheduleAlgoEnv &operator=(const ScopedScheduleAlgoEnv &) = delete;

private:
  static void apply(const char *value) {
    if (value)
      setenv(kScheduleAlgoEnvVar, value, /*overwrite=*/1);
    else
      unsetenv(kScheduleAlgoEnvVar);
  }

  std::optional<std::string> saved;
};

TEST(ScheduleAlgoSelectionTest, DefaultsToRauWithoutEnvOrForcedAlgo) {
  ScopedScheduleAlgoEnv env(nullptr);

  EXPECT_EQ(getActiveScheduleAlgo(), "rau");
}

TEST(ScheduleAlgoSelectionTest, EmptyForcedAlgoReadsTheEnvVar) {
  ScopedScheduleAlgoEnv env("sms");

  EXPECT_EQ(getActiveScheduleAlgo(), "sms");
  EXPECT_EQ(getActiveScheduleAlgo(""), "sms");
}

TEST(ScheduleAlgoSelectionTest, ForcedAlgoWinsAndLeavesNoResidue) {
  ScopedScheduleAlgoEnv env("sms");

  EXPECT_EQ(getActiveScheduleAlgo("joint_solver"), "joint_solver");
  // The forced choice is an argument, not state: the next caller without one
  // still sees the env var. (Old failure mode: a leaked override made an
  // unrelated compile run the joint solver it never asked for.)
  EXPECT_EQ(getActiveScheduleAlgo(), "sms");
}

// The plan's two-compile scenario, at the selection function: one run forces
// joint_solver, a concurrent one forces nothing. Each must keep its own
// backend for its whole run — a shared slot would either be reset under the
// forcing run ("lost restore") or read by the other one ("leaked override").
TEST(ScheduleAlgoSelectionTest, ConcurrentSelectionsAreIndependent) {
  ScopedScheduleAlgoEnv env("rau");
  constexpr int kIterations = 2000;

  std::atomic<bool> start{false};
  std::string forcedSaw, unforcedSaw; // first wrong observation, if any

  auto resolveRepeatedly = [&](llvm::StringRef forced, llvm::StringRef expected,
                               std::string &firstWrong) {
    while (!start.load(std::memory_order_acquire)) {
    }
    for (int i = 0; i < kIterations; ++i) {
      std::string algo = getActiveScheduleAlgo(forced);
      if (algo != expected && firstWrong.empty())
        firstWrong = algo;
    }
  };

  std::thread forcedRun(resolveRepeatedly, "joint_solver", "joint_solver",
                        std::ref(forcedSaw));
  std::thread unforcedRun(resolveRepeatedly, "", "rau", std::ref(unforcedSaw));
  start.store(true, std::memory_order_release);
  forcedRun.join();
  unforcedRun.join();

  EXPECT_EQ(forcedSaw, "") << "forced run lost its backend to a concurrent one";
  EXPECT_EQ(unforcedSaw, "")
      << "unforced run picked up a concurrent run's backend";
}

#ifdef TRITON_ENABLE_Z3_JOINT_SOLVER

static std::string serializeJsonObject(llvm::json::Object object) {
  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << llvm::json::Value(std::move(object));
  stream.flush();
  return output;
}

static FailureOr<llvm::json::Object> parseJsonObject(llvm::StringRef json) {
  auto parsed = llvm::json::parse(json);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  auto *object = parsed->getAsObject();
  if (!object)
    return failure();
  return std::move(*object);
}

static FailureOr<llvm::json::Object>
runJsonProblem(llvm::json::Object problem) {
  auto output = runZ3JointSolver(serializeJsonObject(std::move(problem)));
  if (failed(output))
    return failure();
  return parseJsonObject(*output);
}

static FailureOr<std::string>
withFirstLoweringEventKind(llvm::StringRef problemJson, llvm::StringRef kind) {
  auto parsed = llvm::json::parse(problemJson);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  auto *root = parsed->getAsObject();
  auto *templates = root ? root->getArray("lowering_templates") : nullptr;
  auto *loweringTemplate = templates && !templates->empty()
                               ? templates->front().getAsObject()
                               : nullptr;
  auto *events =
      loweringTemplate ? loweringTemplate->getArray("events") : nullptr;
  auto *event =
      events && !events->empty() ? events->front().getAsObject() : nullptr;
  if (!event)
    return failure();
  (*event)["kind"] = kind.str();
  return serializeJsonObject(std::move(*root));
}

constexpr size_t kGemmNodeCount = 10;

struct GemmMachineModel {
  const char *name;
  const char *architecture;
  int64_t ii;
  int64_t loadALatency;
  int64_t loadBLatency;
  int64_t barrierLatency;
  int64_t mmaLatency;
  int64_t storeLatency;
  int64_t loopDistance;
  int64_t smemBudget;
  int64_t tmemBudget;
  bool accumulatorInTmem;
  std::array<int64_t, kGemmNodeCount> committedStages;
  std::array<int64_t, kGemmNodeCount> expectedCycles;
};

constexpr GemmMachineModel kHopperGemmModel{
    "sm90-hopper-gemm-0.1",
    "sm90",
    8,
    4,
    3,
    3,
    3,
    1,
    2,
    128,
    0,
    false,
    {0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
    {0, 1, 2, 3, 4, 7, 10, 12, 13, 14},
};

constexpr GemmMachineModel kBlackwellGemmModel{
    "sm100-blackwell-gemm-0.1",
    "sm100",
    8,
    5,
    4,
    4,
    4,
    2,
    3,
    96,
    64,
    true,
    {0, 0, 0, 0, 0, 1, 1, 1, 1, 2},
    {0, 1, 2, 3, 4, 8, 12, 14, 15, 17},
};

static llvm::json::Object makeGemmProblem(const GemmMachineModel &model) {
  constexpr std::array<const char *, kGemmNodeCount> labels{
      "ptr_a", "ptr_b",      "tma_a", "tma_b", "barrier_expect",
      "mma",   "acc_update", "sfu",   "cast",  "store",
  };
  constexpr std::array<const char *, kGemmNodeCount> pipelines{
      "NONE", "NONE", "TMA", "TMA", "NONE", "TC", "CUDA", "SFU", "CUDA", "TMA",
  };
  constexpr std::array<int64_t, kGemmNodeCount> clusters{
      0, 0, 0, 0, 0, 1, 2, 2, 2, 3,
  };

  llvm::json::Array nodes;
  for (size_t index = 0; index < kGemmNodeCount; ++index) {
    int64_t latency = 1;
    if (index == 2)
      latency = model.loadALatency;
    else if (index == 3)
      latency = model.loadBLatency;
    else if (index == 5)
      latency = model.mmaLatency;
    nodes.push_back(llvm::json::Object{
        {"id", static_cast<int64_t>(index)},
        {"label", labels[index]},
        {"cycle", model.committedStages[index] * model.ii},
        {"duration", llvm::StringRef(pipelines[index]) == "NONE" ? 0 : 1},
        {"latency", latency},
        {"pipeline", pipelines[index]},
        {"freq", 1},
    });
  }

  llvm::json::Array edges;
  auto addEdge = [&](int64_t src, int64_t dst, int64_t latency,
                     int64_t distance = 0, int64_t roundTrip = 0,
                     int64_t channelBytes = 0) {
    edges.push_back(llvm::json::Object{
        {"src", src},
        {"dst", dst},
        {"src_result_idx", 0},
        {"latency", latency},
        {"distance", distance},
        {"freq", 1},
        {"rt", roundTrip},
        {"xissue", 0},
        {"chan_bytes", channelBytes},
        {"src_cluster", clusters[src]},
        {"dst_cluster", clusters[dst]},
    });
  };
  addEdge(0, 1, 1);
  addEdge(1, 2, 1);
  addEdge(2, 3, 1);
  addEdge(3, 4, 1);
  addEdge(2, 5, model.loadALatency, 0, 1, 8);
  addEdge(3, 5, model.loadBLatency, 0, 1, 8);
  addEdge(4, 5, model.barrierLatency);
  addEdge(5, 6, model.mmaLatency);
  addEdge(6, 7, 2);
  addEdge(7, 8, 1);
  addEdge(8, 9, model.storeLatency);
  addEdge(9, 0, 1, model.loopDistance);

  llvm::json::Array buffers;
  auto addBuffer = [&](int64_t id, int64_t producer, llvm::StringRef kind,
                       int64_t sizeBytes, int64_t minCount, int64_t consumer) {
    buffers.push_back(llvm::json::Object{
        {"id", id},
        {"producer", producer},
        {"size_bytes", sizeBytes},
        {"count", minCount},
        {"min_count", minCount},
        {"kind", kind},
        {"consumers",
         llvm::json::Array{llvm::json::Object{
             {"node", consumer}, {"latency", 0}, {"distance", 0}}}},
    });
  };
  addBuffer(0, 2, "smem", 16, 2, 5);
  addBuffer(1, 3, "smem", 16, 2, 5);
  addBuffer(2, 5, model.accumulatorInTmem ? "tmem" : "smem", 32,
            model.accumulatorInTmem ? 2 : 1, 8);

  llvm::json::Array conflicts;
  for (int64_t left = 0; left < 4; ++left)
    for (int64_t right = left + 1; right < 4; ++right)
      conflicts.push_back(llvm::json::Array{left, right});

  llvm::json::Array loweringTemplates;
  loweringTemplates.push_back(llvm::json::Object{
      {"id", 0},
      {"relation", "different_wg"},
      {"src_node", 2},
      {"dst_node", 5},
      {"src_cluster", 0},
      {"dst_cluster", 1},
      {"events",
       llvm::json::Array{
           llvm::json::Object{
               {"id", 0},
               {"kind", "arrive"},
               {"owner", "src"},
               {"anchor_node", 4},
               {"placement", "after"},
               {"pipeline", "NONE"},
               {"issue_duration", 1},
               {"completion_latency", 0},
               {"blocking", false},
               {"async", false},
               {"distance", 0},
               {"frequency", 1},
               {"bytes", 0},
               {"depth", 0},
               {"semaphore", ""},
           },
           llvm::json::Object{
               {"id", 1},
               {"kind", "wait"},
               {"owner", "dst"},
               {"anchor_node", 5},
               {"placement", "before"},
               {"pipeline", "NONE"},
               {"issue_duration", 1},
               {"completion_latency", 0},
               {"blocking", true},
               {"async", false},
               {"distance", 0},
               {"frequency", 1},
               {"bytes", 0},
               {"depth", 0},
               {"semaphore", ""},
           },
       }},
  });

  llvm::json::Array warpFootprint;
  for (int64_t value : {0, 8, 16, 0, 32, 0, 0, 0, 64})
    warpFootprint.push_back(value);

  return llvm::json::Object{
      {"version", "joint-solver-0.2"},
      {"mode", "joint"},
      {"name", std::string("gemm-") + model.architecture},
      {"machine_model",
       llvm::json::Object{{"version", model.name},
                          {"architecture", model.architecture}}},
      {"emitter_caps_version", 2},
      {"ii", model.ii},
      {"max_wgs", 4},
      {"committed_smem", 0},
      {"fixed_smem", 16},
      {"smem_budget", model.smemBudget},
      {"tmem_budget_bytes", model.tmemBudget},
      {"default_wg_footprint", 0},
      {"sm_regs", 56},
      {"reg_budget", 56},
      {"default_slack", 0},
      {"time_limit_s", 5},
      {"canonical_root", 0},
      {"warp_footprint", std::move(warpFootprint)},
      {"nodes", std::move(nodes)},
      {"clusters",
       llvm::json::Array{
           llvm::json::Object{{"id", 0},
                              {"min_warps", 1},
                              {"nodes", llvm::json::Array{0, 1, 2, 3, 4}}},
           llvm::json::Object{
               {"id", 1}, {"min_warps", 1}, {"nodes", llvm::json::Array{5}}},
           llvm::json::Object{{"id", 2},
                              {"min_warps", 4},
                              {"nodes", llvm::json::Array{6, 7, 8}}},
           llvm::json::Object{
               {"id", 3}, {"min_warps", 1}, {"nodes", llvm::json::Array{9}}},
       }},
      {"edges", std::move(edges)},
      {"buffers", std::move(buffers)},
      {"lowering_templates", std::move(loweringTemplates)},
      {"warp_group_conflicts", std::move(conflicts)},
  };
}

using GoldenAssignment = std::tuple<int64_t, int64_t, int64_t, int64_t>;
using GoldenBuffer = std::tuple<int64_t, int64_t, std::string>;
using GoldenEvent = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;

struct CanonicalGolden {
  std::string machineModel;
  int64_t ii;
  int64_t usedWGs;
  std::vector<GoldenAssignment> assignments;
  std::vector<GoldenBuffer> buffers;
  std::vector<GoldenEvent> loweringEvents;
};

static FailureOr<llvm::json::Object>
runProductionValidation(llvm::StringRef problemJson,
                        llvm::StringRef solutionJson) {
  auto object =
      parseJsonObject(validateZ3JointSolution(problemJson, solutionJson));
  if (failed(object))
    return failure();
  auto schema = object->getString("schema");
  if (!schema || *schema != "joint-solver-validation-0.1")
    return failure();
  return object;
}

static FailureOr<CanonicalGolden>
canonicalizeGolden(llvm::StringRef problemJson, llvm::StringRef solutionJson) {
  auto problem = parseJsonObject(problemJson);
  auto solution = parseJsonObject(solutionJson);
  if (failed(problem) || failed(solution))
    return failure();
  auto status = solution->getString("status");
  auto ii = problem->getInteger("ii");
  auto usedWGs = solution->getInteger("used_wgs");
  auto *machineModel = problem->getObject("machine_model");
  auto modelVersion =
      machineModel ? machineModel->getString("version") : std::nullopt;
  auto *nodes = problem->getArray("nodes");
  auto *clusters = problem->getArray("clusters");
  auto *problemBuffers = problem->getArray("buffers");
  auto *problemTemplates = problem->getArray("lowering_templates");
  auto *cycles = solution->getObject("cycles");
  auto *warpGroups = solution->getObject("wg");
  auto *depths = solution->getObject("buffer_depths");
  auto *loweringPlan = solution->getObject("lowering_plan");
  auto *planTemplates =
      loweringPlan ? loweringPlan->getArray("templates") : nullptr;
  auto planVersion =
      loweringPlan ? loweringPlan->getString("version") : std::nullopt;
  if (!status || *status != "ok" || !ii || *ii <= 0 || !usedWGs ||
      !modelVersion || !nodes || !clusters || !problemBuffers ||
      !problemTemplates || !cycles || !warpGroups || !depths || !loweringPlan ||
      !planVersion || *planVersion != "lowering-plan-0.1" || !planTemplates ||
      cycles->size() != nodes->size() ||
      warpGroups->size() != clusters->size() ||
      planTemplates->size() != problemTemplates->size())
    return failure();

  std::vector<int64_t> nodeIds;
  std::set<int64_t> uniqueNodeIds;
  for (const llvm::json::Value &value : *nodes) {
    auto *node = value.getAsObject();
    auto id = node ? node->getInteger("id") : std::nullopt;
    if (!id || !uniqueNodeIds.insert(*id).second)
      return failure();
    nodeIds.push_back(*id);
  }
  std::sort(nodeIds.begin(), nodeIds.end());

  std::map<int64_t, int64_t> clusterToRawGroup;
  std::map<int64_t, int64_t> nodeToCluster;
  for (const llvm::json::Value &value : *clusters) {
    auto *cluster = value.getAsObject();
    auto clusterId = cluster ? cluster->getInteger("id") : std::nullopt;
    auto *clusterNodes = cluster ? cluster->getArray("nodes") : nullptr;
    if (!clusterId || !clusterNodes || clusterToRawGroup.count(*clusterId) != 0)
      return failure();
    auto group = warpGroups->getInteger(std::to_string(*clusterId));
    if (!group || *group < 0)
      return failure();
    clusterToRawGroup.emplace(*clusterId, *group);
    for (const llvm::json::Value &nodeValue : *clusterNodes) {
      auto nodeId = nodeValue.getAsInteger();
      if (!nodeId || !uniqueNodeIds.count(*nodeId) ||
          !nodeToCluster.emplace(*nodeId, *clusterId).second)
        return failure();
    }
  }

  CanonicalGolden golden{modelVersion->str(), *ii, *usedWGs, {}, {}, {}};
  std::map<int64_t, int64_t> rawToNormalizedGroup;
  std::map<int64_t, int64_t> cycleByNode;
  for (int64_t nodeId : nodeIds) {
    auto cycle = cycles->getInteger(std::to_string(nodeId));
    auto cluster = nodeToCluster.find(nodeId);
    if (!cycle || *cycle < 0 || cluster == nodeToCluster.end())
      return failure();
    int64_t rawGroup = clusterToRawGroup.at(cluster->second);
    auto [normalized, inserted] = rawToNormalizedGroup.emplace(
        rawGroup, static_cast<int64_t>(rawToNormalizedGroup.size()));
    (void)inserted;
    cycleByNode.emplace(nodeId, *cycle);
    golden.assignments.emplace_back(nodeId, *cycle, *cycle / *ii,
                                    normalized->second);
  }
  if (static_cast<int64_t>(rawToNormalizedGroup.size()) != *usedWGs)
    return failure();

  size_t smemBufferCount = 0;
  std::set<int64_t> uniqueBufferIds;
  for (const llvm::json::Value &value : *problemBuffers) {
    auto *buffer = value.getAsObject();
    auto id = buffer ? buffer->getInteger("id") : std::nullopt;
    auto producer = buffer ? buffer->getInteger("producer") : std::nullopt;
    auto sizeBytes = buffer ? buffer->getInteger("size_bytes") : std::nullopt;
    auto minCount = buffer ? buffer->getInteger("min_count") : std::nullopt;
    auto kind = buffer ? buffer->getString("kind") : std::nullopt;
    auto *consumers = buffer ? buffer->getArray("consumers") : nullptr;
    if (!id || !producer || !sizeBytes || !minCount || !kind || !consumers ||
        !uniqueBufferIds.insert(*id).second ||
        cycleByNode.count(*producer) == 0)
      return failure();
    int64_t producerCycle = cycleByNode.at(*producer);
    int64_t lastEnd = producerCycle;
    for (const llvm::json::Value &consumerValue : *consumers) {
      auto *consumer = consumerValue.getAsObject();
      auto node = consumer ? consumer->getInteger("node") : std::nullopt;
      auto latency = consumer ? consumer->getInteger("latency") : std::nullopt;
      auto distance =
          consumer ? consumer->getInteger("distance") : std::nullopt;
      if (!node || !latency || !distance || cycleByNode.count(*node) == 0)
        return failure();
      lastEnd =
          std::max(lastEnd, cycleByNode.at(*node) + *latency + *distance * *ii);
    }
    if (lastEnd < producerCycle)
      return failure();
    int64_t depth = std::max((lastEnd - producerCycle) / *ii + 1, *minCount);
    if (*kind == "smem") {
      ++smemBufferCount;
      auto serializedDepth = depths->getInteger(std::to_string(*id));
      if (!serializedDepth || *serializedDepth != depth)
        return failure();
    }
    golden.buffers.emplace_back(*id, depth, kind->str());
  }
  if (depths->size() != smemBufferCount)
    return failure();
  std::sort(golden.buffers.begin(), golden.buffers.end());

  std::map<int64_t, const llvm::json::Object *> problemTemplateById;
  for (const llvm::json::Value &value : *problemTemplates) {
    auto *loweringTemplate = value.getAsObject();
    auto id =
        loweringTemplate ? loweringTemplate->getInteger("id") : std::nullopt;
    if (!id || !problemTemplateById.emplace(*id, loweringTemplate).second)
      return failure();
  }
  std::set<int64_t> seenTemplateIds;
  for (const llvm::json::Value &value : *planTemplates) {
    auto *planTemplate = value.getAsObject();
    auto id = planTemplate ? planTemplate->getInteger("id") : std::nullopt;
    auto active =
        planTemplate ? planTemplate->getBoolean("active") : std::nullopt;
    auto *events = planTemplate ? planTemplate->getArray("events") : nullptr;
    if (!id || !active || !events || !seenTemplateIds.insert(*id).second ||
        problemTemplateById.count(*id) == 0)
      return failure();
    const llvm::json::Object *problemTemplate = problemTemplateById.at(*id);
    auto relation = problemTemplate->getString("relation");
    auto srcCluster = problemTemplate->getInteger("src_cluster");
    auto dstCluster = problemTemplate->getInteger("dst_cluster");
    auto *expectedEvents = problemTemplate->getArray("events");
    if (!relation || !srcCluster || !dstCluster || !expectedEvents ||
        clusterToRawGroup.count(*srcCluster) == 0 ||
        clusterToRawGroup.count(*dstCluster) == 0)
      return failure();
    bool sameGroup =
        clusterToRawGroup.at(*srcCluster) == clusterToRawGroup.at(*dstCluster);
    bool expectedActive = *relation == "always" ||
                          (*relation == "same_wg" && sameGroup) ||
                          (*relation == "different_wg" && !sameGroup);
    if (*active != expectedActive ||
        events->size() != (expectedActive ? expectedEvents->size() : 0))
      return failure();

    std::set<int64_t> expectedEventIds;
    for (const llvm::json::Value &eventValue : *expectedEvents) {
      auto *event = eventValue.getAsObject();
      auto eventId = event ? event->getInteger("id") : std::nullopt;
      if (!eventId || !expectedEventIds.insert(*eventId).second)
        return failure();
    }
    for (const llvm::json::Value &eventValue : *events) {
      auto *event = eventValue.getAsObject();
      auto eventId = event ? event->getInteger("id") : std::nullopt;
      auto cycle = event ? event->getInteger("cycle") : std::nullopt;
      auto rawGroup = event ? event->getInteger("wg") : std::nullopt;
      auto streamOrder =
          event ? event->getInteger("stream_order") : std::nullopt;
      if (!eventId || !cycle || !rawGroup || !streamOrder ||
          expectedEventIds.erase(*eventId) != 1 ||
          rawToNormalizedGroup.count(*rawGroup) == 0)
        return failure();
      golden.loweringEvents.emplace_back(*id, *eventId, *cycle,
                                         rawToNormalizedGroup.at(*rawGroup),
                                         *streamOrder);
    }
    if (expectedActive && !expectedEventIds.empty())
      return failure();
  }
  if (seenTemplateIds.size() != problemTemplateById.size())
    return failure();
  std::sort(golden.loweringEvents.begin(), golden.loweringEvents.end());
  return golden;
}

static CanonicalGolden expectedGemmGolden(const GemmMachineModel &model) {
  constexpr std::array<int64_t, kGemmNodeCount> groups{
      0, 0, 0, 0, 0, 1, 2, 2, 2, 3,
  };
  CanonicalGolden golden{model.name, model.ii, 4, {}, {}, {}};
  for (size_t index = 0; index < kGemmNodeCount; ++index)
    golden.assignments.emplace_back(index, model.expectedCycles[index],
                                    model.expectedCycles[index] / model.ii,
                                    groups[index]);
  golden.buffers = model.accumulatorInTmem
                       ? std::vector<GoldenBuffer>{{0, 2, "smem"},
                                                   {1, 2, "smem"},
                                                   {2, 2, "tmem"}}
                       : std::vector<GoldenBuffer>{
                             {0, 2, "smem"}, {1, 2, "smem"}, {2, 1, "smem"}};
  golden.loweringEvents = {
      {0, 0, 5, 0, 0},
      {0, 1, model.expectedCycles[5] - 1, 1, 0},
  };
  return golden;
}

static void expectProductionValidity(llvm::StringRef problemJson,
                                     llvm::StringRef solutionJson,
                                     bool expectedValid) {
  auto validation = runProductionValidation(problemJson, solutionJson);
  ASSERT_TRUE(succeeded(validation));
  auto valid = validation->getBoolean("valid");
  auto message = validation->getString("message");
  ASSERT_TRUE(valid);
  ASSERT_TRUE(message);
  EXPECT_EQ(*valid, expectedValid) << message->str();
  if (!expectedValid)
    EXPECT_FALSE(message->empty());
}

static void expectGemmGolden(const GemmMachineModel &model) {
  std::string problemJson = serializeJsonObject(makeGemmProblem(model));
  CanonicalGolden expected = expectedGemmGolden(model);
  std::vector<CanonicalGolden> actual;
  for (int run = 0; run < 2; ++run) {
    auto output = runZ3JointSolver(problemJson);
    ASSERT_TRUE(succeeded(output));
    expectProductionValidity(problemJson, *output, true);
    auto canonical = canonicalizeGolden(problemJson, *output);
    ASSERT_TRUE(succeeded(canonical));
    actual.push_back(std::move(*canonical));
  }

  ASSERT_EQ(actual.size(), 2);
  for (const CanonicalGolden &golden : actual) {
    EXPECT_EQ(golden.machineModel, expected.machineModel);
    EXPECT_EQ(golden.ii, expected.ii);
    EXPECT_EQ(golden.usedWGs, expected.usedWGs);
    EXPECT_EQ(golden.assignments, expected.assignments);
    EXPECT_EQ(golden.buffers, expected.buffers);
    EXPECT_EQ(golden.loweringEvents, expected.loweringEvents);
  }
  EXPECT_EQ(actual[0].assignments, actual[1].assignments);
  EXPECT_EQ(actual[0].buffers, actual[1].buffers);
  EXPECT_EQ(actual[0].loweringEvents, actual[1].loweringEvents);
}

static llvm::json::Object *findEdge(llvm::json::Object &problem, int64_t src,
                                    int64_t dst) {
  auto *edges = problem.getArray("edges");
  if (!edges)
    return nullptr;
  for (llvm::json::Value &value : *edges) {
    auto *edge = value.getAsObject();
    if (edge && edge->getInteger("src") == src &&
        edge->getInteger("dst") == dst)
      return edge;
  }
  return nullptr;
}

static int diagnosticKindRank(llvm::StringRef kind) {
  constexpr llvm::StringLiteral kinds[] = {
      "dependence", "resource", "smem",     "tmem",
      "warp-group", "register", "lowering",
  };
  auto found = llvm::find(kinds, kind);
  return found == std::end(kinds) ? static_cast<int>(std::size(kinds))
                                  : static_cast<int>(found - std::begin(kinds));
}

static FailureOr<std::vector<std::string>>
readStringArray(const llvm::json::Object &object, llvm::StringRef key) {
  auto *values = object.getArray(key);
  if (!values)
    return failure();
  std::vector<std::string> result;
  result.reserve(values->size());
  for (const llvm::json::Value &value : *values) {
    auto string = value.getAsString();
    if (!string)
      return failure();
    result.push_back(string->str());
  }
  return result;
}

static void expectUnsatDiagnostic(
    llvm::StringRef problemJson,
    std::initializer_list<llvm::StringRef> requiredKinds,
    std::initializer_list<llvm::StringRef> requiredGroupIds,
    llvm::StringRef expectedNormalization = "locally_minimized") {
  auto output = runZ3JointSolver(problemJson);
  ASSERT_TRUE(succeeded(output));
  auto response = parseJsonObject(*output);
  ASSERT_TRUE(succeeded(response));

  auto status = response->getString("status");
  auto provenUnsat = response->getBoolean("proven_unsat");
  auto backendStatus = response->getString("backend_status");
  auto *core = response->getObject("unsat_core");
  auto *diagnostic = response->getObject("diagnostic");
  ASSERT_TRUE(status);
  ASSERT_TRUE(provenUnsat);
  ASSERT_TRUE(backendStatus);
  ASSERT_NE(core, nullptr);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(*status, "infeasible");
  EXPECT_TRUE(*provenUnsat);
  EXPECT_EQ(*backendStatus, "INFEASIBLE");

  auto coreSchema = core->getString("schema");
  auto coreII = core->getInteger("candidateII");
  auto coreBackendStatus = core->getString("backendStatus");
  auto coreProvenUnsat = core->getBoolean("provenUnsat");
  auto normalization = core->getString("normalization");
  auto groupIds = readStringArray(*core, "groupIds");
  ASSERT_TRUE(coreSchema);
  ASSERT_TRUE(coreII);
  ASSERT_TRUE(coreBackendStatus);
  ASSERT_TRUE(coreProvenUnsat);
  ASSERT_TRUE(normalization);
  ASSERT_TRUE(succeeded(groupIds));
  EXPECT_EQ(*coreSchema, "joint-solver-core-0.1");
  EXPECT_EQ(*coreBackendStatus, "INFEASIBLE");
  EXPECT_TRUE(*coreProvenUnsat);
  EXPECT_EQ(*normalization, expectedNormalization);
  ASSERT_FALSE(groupIds->empty());
  EXPECT_EQ(std::set<std::string>(groupIds->begin(), groupIds->end()).size(),
            groupIds->size());
  for (llvm::StringRef requiredGroupId : requiredGroupIds)
    EXPECT_TRUE(llvm::is_contained(*groupIds, requiredGroupId.str()));

  auto diagnosticSchema = diagnostic->getString("schema");
  auto diagnosticStatus = diagnostic->getString("status");
  auto diagnosticII = diagnostic->getInteger("ii");
  auto summary = diagnostic->getString("summary");
  auto *diagnosticCore = diagnostic->getObject("core");
  auto *constraints = diagnostic->getArray("constraints");
  auto *aggregates = diagnostic->getArray("aggregates");
  auto *suggestions = diagnostic->getArray("suggestions");
  ASSERT_TRUE(diagnosticSchema);
  ASSERT_TRUE(diagnosticStatus);
  ASSERT_TRUE(diagnosticII);
  ASSERT_TRUE(summary);
  ASSERT_NE(diagnosticCore, nullptr);
  ASSERT_NE(constraints, nullptr);
  ASSERT_NE(aggregates, nullptr);
  ASSERT_NE(suggestions, nullptr);
  EXPECT_EQ(*diagnosticSchema, "joint-solver-diagnostic-0.1");
  EXPECT_EQ(*diagnosticStatus, "unsat");
  EXPECT_EQ(*diagnosticII, *coreII);
  EXPECT_FALSE(summary->empty());
  EXPECT_FALSE(suggestions->empty());

  auto diagnosticGroupIds = readStringArray(*diagnosticCore, "groupIds");
  auto diagnosticCoreSchema = diagnosticCore->getString("schema");
  auto diagnosticCoreII = diagnosticCore->getInteger("candidateII");
  auto diagnosticCoreBackendStatus = diagnosticCore->getString("backendStatus");
  auto diagnosticCoreProvenUnsat = diagnosticCore->getBoolean("provenUnsat");
  auto diagnosticNormalization = diagnosticCore->getString("normalization");
  ASSERT_TRUE(succeeded(diagnosticGroupIds));
  ASSERT_TRUE(diagnosticCoreSchema);
  ASSERT_TRUE(diagnosticCoreII);
  ASSERT_TRUE(diagnosticCoreBackendStatus);
  ASSERT_TRUE(diagnosticCoreProvenUnsat);
  ASSERT_TRUE(diagnosticNormalization);
  EXPECT_EQ(*diagnosticGroupIds, *groupIds);
  EXPECT_EQ(*diagnosticCoreSchema, *coreSchema);
  EXPECT_EQ(*diagnosticCoreII, *coreII);
  EXPECT_EQ(*diagnosticCoreBackendStatus, *coreBackendStatus);
  EXPECT_EQ(*diagnosticCoreProvenUnsat, *coreProvenUnsat);
  EXPECT_EQ(*diagnosticNormalization, expectedNormalization);

  ASSERT_EQ(constraints->size(), groupIds->size());
  std::set<std::string> kinds;
  int previousRank = -1;
  std::string previousId;
  for (size_t index = 0; index < constraints->size(); ++index) {
    auto *constraint = (*constraints)[index].getAsObject();
    ASSERT_NE(constraint, nullptr);
    auto id = constraint->getString("id");
    auto kind = constraint->getString("kind");
    auto *nodes = constraint->getArray("nodes");
    ASSERT_TRUE(id);
    ASSERT_TRUE(kind);
    ASSERT_NE(nodes, nullptr);
    EXPECT_NE(constraint->get("available"), nullptr);
    EXPECT_EQ(*id, (*groupIds)[index]);
    kinds.insert(kind->str());

    int rank = diagnosticKindRank(*kind);
    EXPECT_GE(rank, previousRank);
    if (rank == previousRank)
      EXPECT_LT(previousId, id->str());
    previousRank = rank;
    previousId = id->str();

    std::vector<int64_t> nodeIds;
    for (const llvm::json::Value &nodeValue : *nodes) {
      auto node = nodeValue.getAsInteger();
      ASSERT_TRUE(node);
      nodeIds.push_back(*node);
    }
    EXPECT_TRUE(std::is_sorted(nodeIds.begin(), nodeIds.end()));
  }
  for (llvm::StringRef requiredKind : requiredKinds)
    EXPECT_TRUE(kinds.count(requiredKind.str()));

  previousRank = -1;
  std::string previousResource;
  for (const llvm::json::Value &value : *aggregates) {
    auto *aggregate = value.getAsObject();
    auto kind = aggregate ? aggregate->getString("kind") : std::nullopt;
    ASSERT_TRUE(kind);
    int rank = diagnosticKindRank(*kind);
    EXPECT_GE(rank, previousRank);
    std::string resource = aggregate->getString("resource").value_or("").str();
    if (rank == previousRank)
      EXPECT_LE(previousResource, resource);
    previousRank = rank;
    previousResource = std::move(resource);
  }
}

struct OracleNode {
  int64_t id;
  std::string pipeline;
  int64_t duration;
  bool streaming;
};

struct OracleEdge {
  int64_t src;
  int64_t dst;
  int64_t latency;
  int64_t distance;
};

struct OracleBuffer {
  int64_t id;
  int64_t alloc;
  std::string kind;
  int64_t sizeBytes;
  int64_t tmemCols;
  std::vector<int64_t> consumers;
};

struct OracleSchedule {
  int64_t ii;
  std::map<int64_t, int64_t> cycles;
};

static bool isOracleScheduleValid(int64_t ii,
                                  const std::vector<OracleNode> &nodes,
                                  const std::vector<OracleEdge> &edges,
                                  const std::vector<OracleBuffer> &buffers,
                                  int64_t smemBudget, int64_t tmemColLimit,
                                  bool streamingVL,
                                  const std::map<int64_t, int64_t> &cycles,
                                  std::optional<int64_t> canonicalRoot) {
  if (ii <= 0 || cycles.size() != nodes.size())
    return false;
  if (canonicalRoot) {
    auto rootCycle = cycles.find(*canonicalRoot);
    if (rootCycle == cycles.end() || rootCycle->second != 0)
      return false;
  } else if (!cycles.empty()) {
    auto minimum = std::min_element(cycles.begin(), cycles.end(),
                                    [](const auto &left, const auto &right) {
                                      return left.second < right.second;
                                    });
    if (minimum->second != 0)
      return false;
  }

  std::map<int64_t, const OracleNode *> nodeById;
  for (const OracleNode &node : nodes)
    nodeById.emplace(node.id, &node);

  for (const OracleEdge &edge : edges) {
    auto src = cycles.find(edge.src);
    auto dst = cycles.find(edge.dst);
    int64_t latency =
        streamingVL && nodeById.at(edge.src)->streaming ? 0 : edge.latency;
    if (src == cycles.end() || dst == cycles.end() ||
        dst->second + edge.distance * ii < src->second + latency ||
        (edge.distance == 0 && src->second / ii > dst->second / ii))
      return false;
  }

  for (size_t leftIndex = 0; leftIndex < nodes.size(); ++leftIndex) {
    const OracleNode &left = nodes[leftIndex];
    if (left.pipeline == "NONE")
      continue;
    if (left.duration <= 0 || left.duration > ii)
      return false;
    for (size_t rightIndex = leftIndex + 1; rightIndex < nodes.size();
         ++rightIndex) {
      const OracleNode &right = nodes[rightIndex];
      if (left.pipeline != right.pipeline)
        continue;
      if (right.duration <= 0 || right.duration > ii)
        return false;
      for (int64_t leftOffset = 0; leftOffset < left.duration; ++leftOffset) {
        for (int64_t rightOffset = 0; rightOffset < right.duration;
             ++rightOffset) {
          int64_t leftSlot =
              (cycles.at(left.id) + leftOffset) % static_cast<int64_t>(ii);
          int64_t rightSlot =
              (cycles.at(right.id) + rightOffset) % static_cast<int64_t>(ii);
          if (leftSlot == rightSlot)
            return false;
        }
      }
    }
  }

  int64_t smemUsage = 0;
  struct TmemLifetime {
    int64_t start;
    int64_t end;
    int64_t columns;
  };
  std::vector<TmemLifetime> tmemLifetimes;
  for (const OracleBuffer &buffer : buffers) {
    int64_t allocCycle = cycles.at(buffer.alloc);
    if (buffer.kind == "smem") {
      int64_t lastStage = allocCycle / ii;
      for (int64_t consumer : buffer.consumers)
        lastStage = std::max(lastStage, cycles.at(consumer) / ii);
      smemUsage += buffer.sizeBytes * (lastStage - allocCycle / ii + 1);
      continue;
    }
    int64_t end = allocCycle + 1;
    for (int64_t consumer : buffer.consumers)
      end = std::max(end, cycles.at(consumer) + 1);
    tmemLifetimes.push_back({allocCycle, end, buffer.tmemCols});
  }
  if (smemUsage > smemBudget)
    return false;
  for (const TmemLifetime &checkpoint : tmemLifetimes) {
    int64_t activeColumns = 0;
    for (const TmemLifetime &lifetime : tmemLifetimes)
      if (lifetime.start <= checkpoint.start && checkpoint.start < lifetime.end)
        activeColumns += lifetime.columns;
    if (activeColumns > tmemColLimit)
      return false;
  }
  return true;
}

static FailureOr<OracleSchedule>
enumerateStageAwareSchedule(llvm::StringRef problemJson) {
  auto problem = parseJsonObject(problemJson);
  if (failed(problem))
    return failure();
  auto minII = problem->getInteger("min_ii");
  auto maxII = problem->getInteger("max_ii");
  auto smemBudget = problem->getInteger("smem_budget");
  auto tmemColLimit = problem->getInteger("tmem_col_limit");
  auto streamingVL = problem->getBoolean("streaming_vl");
  auto *nodeValues = problem->getArray("nodes");
  auto *edgeValues = problem->getArray("edges");
  auto *bufferValues = problem->getArray("buffers");
  if (!minII || !maxII || *minII <= 0 || *maxII < *minII || !nodeValues ||
      !edgeValues || !bufferValues || !smemBudget || !tmemColLimit ||
      !streamingVL || *smemBudget < 0 || *tmemColLimit < 0 ||
      nodeValues->size() > 6)
    return failure();

  std::vector<OracleNode> nodes;
  std::set<int64_t> nodeIds;
  for (const llvm::json::Value &value : *nodeValues) {
    auto *node = value.getAsObject();
    auto id = node ? node->getInteger("id") : std::nullopt;
    auto pipeline = node ? node->getString("pipeline") : std::nullopt;
    auto duration = node ? node->getInteger("duration") : std::nullopt;
    auto streaming = node ? node->getBoolean("streaming") : std::nullopt;
    if (!id || !pipeline || !duration || !streaming ||
        !nodeIds.insert(*id).second)
      return failure();
    nodes.push_back(OracleNode{*id, pipeline->str(), *duration, *streaming});
  }
  std::sort(nodes.begin(), nodes.end(),
            [](const OracleNode &left, const OracleNode &right) {
              return left.id < right.id;
            });

  std::vector<OracleEdge> edges;
  int64_t latencySum = 0;
  for (const llvm::json::Value &value : *edgeValues) {
    auto *edge = value.getAsObject();
    auto src = edge ? edge->getInteger("src") : std::nullopt;
    auto dst = edge ? edge->getInteger("dst") : std::nullopt;
    auto latency = edge ? edge->getInteger("latency") : std::nullopt;
    auto distance = edge ? edge->getInteger("distance") : std::nullopt;
    if (!src || !dst || !latency || !distance || !nodeIds.count(*src) ||
        !nodeIds.count(*dst) || *latency < 0 || *distance < 0)
      return failure();
    edges.push_back(OracleEdge{*src, *dst, *latency, *distance});
    latencySum += *latency;
  }

  std::vector<OracleBuffer> buffers;
  std::set<int64_t> bufferIds;
  for (const llvm::json::Value &value : *bufferValues) {
    auto *buffer = value.getAsObject();
    auto id = buffer ? buffer->getInteger("id") : std::nullopt;
    auto alloc = buffer ? buffer->getInteger("alloc_node") : std::nullopt;
    auto kind = buffer ? buffer->getString("kind") : std::nullopt;
    auto sizeBytes = buffer ? buffer->getInteger("size_bytes") : std::nullopt;
    auto tmemCols = buffer ? buffer->getInteger("tmem_cols") : std::nullopt;
    auto *consumers = buffer ? buffer->getArray("consumers") : nullptr;
    if (!id || !alloc || !kind || !sizeBytes || !tmemCols || !consumers ||
        !bufferIds.insert(*id).second || !nodeIds.count(*alloc) ||
        (*kind != "smem" && *kind != "tmem") || *sizeBytes < 0 || *tmemCols < 0)
      return failure();
    OracleBuffer oracleBuffer{*id,        *alloc,    kind->str(),
                              *sizeBytes, *tmemCols, {}};
    for (const llvm::json::Value &consumerValue : *consumers) {
      auto consumer = consumerValue.getAsInteger();
      if (!consumer || !nodeIds.count(*consumer))
        return failure();
      oracleBuffer.consumers.push_back(*consumer);
    }
    buffers.push_back(std::move(oracleBuffer));
  }

  std::optional<int64_t> canonicalRoot;
  if (problem->get("canonical_root")) {
    canonicalRoot = problem->getInteger("canonical_root");
    if (!canonicalRoot || !nodeIds.count(*canonicalRoot))
      return failure();
  }

  for (int64_t ii = *minII; ii <= *maxII; ++ii) {
    int64_t horizon =
        latencySum + ii * (static_cast<int64_t>(nodes.size()) + 1);
    std::map<int64_t, int64_t> cycles;
    std::optional<OracleSchedule> result;
    std::function<void(size_t)> enumerate = [&](size_t index) {
      if (result)
        return;
      if (index == nodes.size()) {
        if (isOracleScheduleValid(ii, nodes, edges, buffers, *smemBudget,
                                  *tmemColLimit, *streamingVL, cycles,
                                  canonicalRoot))
          result = OracleSchedule{ii, cycles};
        return;
      }
      const OracleNode &node = nodes[index];
      int64_t last = canonicalRoot && node.id == *canonicalRoot ? 0 : horizon;
      for (int64_t cycle = 0; cycle <= last; ++cycle) {
        cycles[node.id] = cycle;
        enumerate(index + 1);
        if (result)
          return;
      }
      cycles.erase(node.id);
    };
    enumerate(0);
    if (result)
      return *result;
  }
  return failure();
}

constexpr llvm::StringLiteral kStageAwareOracleProblem = R"json(
{
  "version": "joint-solver-0.1",
  "min_ii": 1,
  "max_ii": 1,
  "smem_budget": 0,
  "tmem_col_limit": 0,
  "time_limit_s": 5,
  "streaming_vl": false,
  "canonical_root": 0,
  "nodes": [
    {"id": 0, "pipeline": "TMA", "duration": 1, "streaming": false},
    {"id": 1, "pipeline": "CUDA", "duration": 1, "streaming": false}
  ],
  "edges": [{"src": 0, "dst": 1, "latency": 2, "distance": 0}],
  "buffers": []
}
)json";

constexpr llvm::StringLiteral kBinaryOracleProblem = R"json(
{
  "version": "joint-solver-0.1",
  "min_ii": 1,
  "max_ii": 3,
  "smem_budget": 2,
  "tmem_col_limit": 1,
  "time_limit_s": 5,
  "streaming_vl": true,
  "canonical_root": 0,
  "nodes": [
    {"id": 0, "pipeline": "TMA", "duration": 1, "streaming": true},
    {"id": 1, "pipeline": "TC", "duration": 1, "streaming": false},
    {"id": 2, "pipeline": "TC", "duration": 1, "streaming": false}
  ],
  "edges": [
    {"src": 0, "dst": 1, "latency": 5, "distance": 0},
    {"src": 1, "dst": 2, "latency": 2, "distance": 0}
  ],
  "buffers": [
    {"id": 7, "alloc_node": 0, "kind": "smem", "size_bytes": 1,
     "tmem_cols": 0, "consumers": [2]},
    {"id": 9, "alloc_node": 1, "kind": "tmem", "size_bytes": 0,
     "tmem_cols": 1, "consumers": [2]}
  ]
}
)json";

constexpr llvm::StringLiteral kFeasibleJointSolverProblem = R"json(
{
  "version": "joint-solver-0.1",
  "min_ii": 2,
  "max_ii": 5,
  "smem_budget": 0,
  "tmem_col_limit": 0,
  "time_limit_s": 5,
  "streaming_vl": false,
  "nodes": [
    {"id": 0, "pipeline": "TC", "duration": 2, "streaming": false},
    {"id": 1, "pipeline": "TC", "duration": 2, "streaming": false}
  ],
  "edges": [{"src": 0, "dst": 1, "latency": 2, "distance": 0}],
  "buffers": []
}
)json";

constexpr llvm::StringLiteral kInfeasibleJointSolverProblem = R"json(
{
  "version": "joint-solver-0.1",
  "min_ii": 2,
  "max_ii": 3,
  "smem_budget": 0,
  "tmem_col_limit": 0,
  "time_limit_s": 5,
  "streaming_vl": false,
  "nodes": [
    {"id": 0, "pipeline": "TC", "duration": 2, "streaming": false},
    {"id": 1, "pipeline": "TC", "duration": 2, "streaming": false}
  ],
  "edges": [{"src": 0, "dst": 1, "latency": 2, "distance": 0}],
  "buffers": []
}
)json";

constexpr llvm::StringLiteral kCompositeObjectiveProblem = R"json(
{
  "version": "joint-solver-0.1",
  "min_ii": 4,
  "max_ii": 4,
  "smem_budget": 0,
  "tmem_col_limit": 0,
  "time_limit_s": 5,
  "streaming_vl": false,
  "canonical_root": 0,
  "nodes": [
    {"id": 0, "pipeline": "NONE", "duration": 0, "streaming": false},
    {"id": 1, "pipeline": "NONE", "duration": 0, "streaming": false}
  ],
  "edges": [
    {"src": 0, "dst": 1, "latency": 0, "distance": 0},
    {"src": 0, "dst": 1, "latency": 0, "distance": 1}
  ],
  "buffers": []
}
)json";

constexpr llvm::StringLiteral kDepthObjectiveProblem = R"json(
{
  "version": "joint-solver-0.1",
  "min_ii": 4,
  "max_ii": 4,
  "smem_budget": 2,
  "tmem_col_limit": 0,
  "time_limit_s": 5,
  "streaming_vl": false,
  "canonical_root": 0,
  "nodes": [
    {"id": 0, "pipeline": "NONE", "duration": 0, "streaming": false},
    {"id": 1, "pipeline": "NONE", "duration": 0, "streaming": false},
    {"id": 2, "pipeline": "TC", "duration": 1, "streaming": false},
    {"id": 3, "pipeline": "TC", "duration": 1, "streaming": false}
  ],
  "edges": [
    {"src": 2, "dst": 3, "latency": 4, "distance": 0}
  ],
  "buffers": [
    {"id": 9, "alloc_node": 0, "kind": "smem", "size_bytes": 1,
     "tmem_cols": 0, "consumers": [1]}
  ]
}
)json";

constexpr llvm::StringLiteral kPartitionObjectiveProblem = R"json(
{
  "version": "joint-solver-0.2",
  "mode": "partition",
  "emitter_caps_version": 2,
  "ii": 4,
  "max_wgs": 2,
  "committed_smem": 0,
  "fixed_smem": 0,
  "smem_budget": 0,
  "tmem_budget_bytes": 0,
  "default_wg_footprint": 0,
  "sm_regs": 0,
  "default_slack": 0,
  "time_limit_s": 5,
  "canonical_root": 0,
  "warp_footprint": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "nodes": [
    {"id": 0, "cycle": 0, "duration": 1, "latency": 1,
     "pipeline": "CUDA", "freq": 1},
    {"id": 1, "cycle": 0, "duration": 1, "latency": 1,
     "pipeline": "SFU", "freq": 1}
  ],
  "clusters": [
    {"id": 10, "min_warps": 1, "nodes": [0]},
    {"id": 30, "min_warps": 1, "nodes": [1]}
  ],
  "edges": [
    {"src": 0, "dst": 1, "src_result_idx": 0, "latency": 0,
     "distance": 1, "freq": 1, "rt": 0, "xissue": 10,
     "chan_bytes": 0, "src_cluster": 10, "dst_cluster": 30}
  ],
  "buffers": [],
  "lowering_templates": []
}
)json";

constexpr llvm::StringLiteral kBlockingObjectiveProblem = R"json(
{
  "version": "joint-solver-0.2",
  "mode": "joint",
  "emitter_caps_version": 2,
  "ii": 4,
  "max_wgs": 1,
  "committed_smem": 0,
  "fixed_smem": 0,
  "smem_budget": 0,
  "tmem_budget_bytes": 0,
  "default_wg_footprint": 0,
  "sm_regs": 0,
  "default_slack": 0,
  "time_limit_s": 5,
  "canonical_root": 0,
  "warp_footprint": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "nodes": [
    {"id": 0, "cycle": 0, "duration": 0, "latency": 0,
     "pipeline": "NONE", "freq": 1},
    {"id": 1, "cycle": 0, "duration": 0, "latency": 0,
     "pipeline": "NONE", "freq": 1},
    {"id": 2, "cycle": 1, "duration": 1, "latency": 1,
     "pipeline": "CUDA", "freq": 1}
  ],
  "clusters": [
    {"id": 10, "min_warps": 1, "nodes": [0, 1, 2]}
  ],
  "edges": [
    {"src": 0, "dst": 2, "src_result_idx": 0, "latency": 1,
     "distance": 0, "freq": 1, "rt": 0, "xissue": 0,
     "chan_bytes": 0, "src_cluster": 10, "dst_cluster": 10}
  ],
  "buffers": [],
  "lowering_templates": [
    {
      "id": 0,
      "relation": "always",
      "src_node": 1,
      "dst_node": 2,
      "src_cluster": 10,
      "dst_cluster": 10,
      "events": [
        {
          "id": 8,
          "kind": "wait",
          "owner": "src",
          "anchor_node": 1,
          "placement": "before",
          "pipeline": "NONE",
          "issue_duration": 1,
          "completion_latency": 0,
          "blocking": true,
          "async": false,
          "distance": 0,
          "frequency": 1,
          "bytes": 0,
          "depth": 0,
          "semaphore": ""
        }
      ]
    }
  ]
}
)json";

constexpr llvm::StringLiteral kJointSolverV2Problem = R"json(
{
  "version": "joint-solver-0.2",
  "mode": "joint",
  "emitter_caps_version": 2,
  "ii": 4,
  "max_wgs": 2,
  "committed_smem": 0,
  "fixed_smem": 0,
  "smem_budget": 0,
  "tmem_budget_bytes": 12,
  "default_wg_footprint": 0,
  "sm_regs": 0,
  "default_slack": 0,
  "time_limit_s": 2,
  "canonical_root": 0,
  "warp_footprint": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "nodes": [
    {"id": 0, "cycle": 0, "duration": 1, "latency": 1,
     "pipeline": "NONE", "freq": 1},
    {"id": 1, "cycle": 1, "duration": 1, "latency": 1,
     "pipeline": "NONE", "freq": 1}
  ],
  "clusters": [
    {"id": 10, "min_warps": 1, "nodes": [0]},
    {"id": 30, "min_warps": 1, "nodes": [1]}
  ],
  "edges": [],
  "buffers": [
    {"id": 42, "producer": 0, "size_bytes": 4, "count": 3,
     "min_count": 3, "kind": "tmem",
     "consumers": [{"node": 1, "latency": 0, "distance": 0}]}
  ],
  "lowering_templates": [
    {
      "id": 0,
      "relation": "different_wg",
      "src_node": 0,
      "dst_node": 1,
      "src_cluster": 10,
      "dst_cluster": 30,
      "events": [
        {
          "id": 7,
          "kind": "arrive",
          "owner": "src",
          "anchor_node": 0,
          "placement": "before",
          "pipeline": "NONE",
          "issue_duration": 1,
          "completion_latency": 0,
          "blocking": false,
          "async": false,
          "distance": 0,
          "frequency": 1,
          "bytes": 0,
          "depth": 0,
          "semaphore": ""
        }
      ]
    }
  ]
}
)json";

static llvm::json::Object makeV1DiagnosticProblem() {
  return llvm::json::Object{
      {"version", "joint-solver-0.1"},
      {"min_ii", 1},
      {"max_ii", 1},
      {"smem_budget", 0},
      {"tmem_col_limit", 0},
      {"time_limit_s", 5},
      {"streaming_vl", false},
      {"canonical_root", 0},
      {"nodes", llvm::json::Array{llvm::json::Object{
                    {"id", 0},
                    {"pipeline", "NONE"},
                    {"duration", 0},
                    {"streaming", false},
                }}},
      {"edges", llvm::json::Array{}},
      {"buffers", llvm::json::Array{}},
  };
}

static void forceDifferentWarpGroups(llvm::json::Object &problem) {
  llvm::json::Array conflicts;
  conflicts.push_back(llvm::json::Array{10, 30});
  problem["warp_group_conflicts"] = std::move(conflicts);
}

constexpr llvm::StringLiteral kJointCrossIssueObjectiveProblem = R"json(
{
  "version": "joint-solver-0.2",
  "mode": "joint",
  "emitter_caps_version": 2,
  "ii": 4,
  "max_wgs": 2,
  "committed_smem": 0,
  "fixed_smem": 0,
  "smem_budget": 0,
  "tmem_budget_bytes": 0,
  "default_wg_footprint": 0,
  "sm_regs": 0,
  "default_slack": 0,
  "time_limit_s": 5,
  "canonical_root": 0,
  "warp_footprint": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "nodes": [
    {"id": 0, "cycle": 0, "duration": 0, "latency": 0,
     "pipeline": "NONE", "freq": 1},
    {"id": 1, "cycle": 1, "duration": 0, "latency": 0,
     "pipeline": "NONE", "freq": 1}
  ],
  "clusters": [
    {"id": 10, "min_warps": 1, "nodes": [0]},
    {"id": 30, "min_warps": 1, "nodes": [1]}
  ],
  "edges": [
    {"src": 0, "dst": 1, "src_result_idx": 0, "latency": 0,
     "distance": 0, "freq": 1, "rt": 0, "xissue": 1,
     "chan_bytes": 0, "src_cluster": 10, "dst_cluster": 30}
  ],
  "buffers": [],
  "lowering_templates": []
}
)json";

static FailureOr<llvm::json::Object> makePipelineGroupingProblem(
    llvm::StringRef mode, llvm::StringRef producerPipeline,
    llvm::StringRef consumerPipeline, int64_t distance, int64_t maxWGs) {
  auto problem = parseJsonObject(kPartitionObjectiveProblem);
  if (failed(problem))
    return failure();
  auto *nodes = problem->getArray("nodes");
  auto *edges = problem->getArray("edges");
  if (!nodes || nodes->size() != 2 || !edges || edges->size() != 1)
    return failure();
  auto *producer = (*nodes)[0].getAsObject();
  auto *consumer = (*nodes)[1].getAsObject();
  auto *edge = (*edges)[0].getAsObject();
  if (!producer || !consumer || !edge)
    return failure();

  (*problem)["mode"] = mode.str();
  (*problem)["max_wgs"] = maxWGs;
  (*producer)["pipeline"] = producerPipeline.str();
  (*consumer)["pipeline"] = consumerPipeline.str();
  (*producer)["cycle"] = 0;
  (*consumer)["cycle"] = 1;
  (*edge)["latency"] = 1;
  (*edge)["distance"] = distance;
  (*edge)["xissue"] = 0;
  return std::move(*problem);
}

TEST(Z3JointSolverTest, FeasibleRangeReturnsMinimumIIAndValidSchedule) {
  auto output = runZ3JointSolver(kFeasibleJointSolverProblem);
  ASSERT_TRUE(succeeded(output));

  auto parsed = llvm::json::parse(*output);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *response = parsed->getAsObject();
  ASSERT_NE(response, nullptr);
  auto status = response->getString("status");
  auto ii = response->getInteger("ii");
  auto *cycles = response->getObject("cycles");
  ASSERT_TRUE(status);
  ASSERT_TRUE(ii);
  ASSERT_NE(cycles, nullptr);
  EXPECT_EQ(*status, "ok");
  EXPECT_EQ(*ii, 4);

  auto producerCycle = cycles->getInteger("0");
  auto consumerCycle = cycles->getInteger("1");
  ASSERT_TRUE(producerCycle);
  ASSERT_TRUE(consumerCycle);
  EXPECT_GE(*consumerCycle, *producerCycle + 2);
  int64_t phaseDistance = (*consumerCycle - *producerCycle) % *ii;
  if (phaseDistance < 0)
    phaseDistance += *ii;
  EXPECT_GE(phaseDistance, 2);
  EXPECT_LE(phaseDistance, *ii - 2);
}

TEST(Z3JointSolverTest, InfeasibleRangeReturnsProofStatus) {
  auto output = runZ3JointSolver(kInfeasibleJointSolverProblem);
  ASSERT_TRUE(succeeded(output));

  auto parsed = llvm::json::parse(*output);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *response = parsed->getAsObject();
  ASSERT_NE(response, nullptr);
  auto status = response->getString("status");
  auto provenUnsat = response->getBoolean("proven_unsat");
  auto backendStatus = response->getString("backend_status");
  ASSERT_TRUE(status);
  ASSERT_TRUE(provenUnsat);
  ASSERT_TRUE(backendStatus);
  EXPECT_EQ(*status, "infeasible");
  EXPECT_TRUE(*provenUnsat);
  EXPECT_EQ(*backendStatus, "INFEASIBLE");
}

TEST(Z3JointSolverTest, V1UnsatDiagnosticsCoverNativeConstraintKinds) {
  {
    llvm::json::Object problem = makeV1DiagnosticProblem();
    problem.getArray("edges")->push_back(llvm::json::Object{
        {"src", 0}, {"dst", 0}, {"latency", 2}, {"distance", 1}});
    expectUnsatDiagnostic(serializeJsonObject(std::move(problem)),
                          {"dependence"}, {"dep:N0->N0:d1"});
  }
  {
    llvm::json::Object problem = makeV1DiagnosticProblem();
    auto *node = problem.getArray("nodes")->front().getAsObject();
    ASSERT_NE(node, nullptr);
    (*node)["pipeline"] = "TC";
    (*node)["duration"] = 2;
    expectUnsatDiagnostic(serializeJsonObject(std::move(problem)), {"resource"},
                          {"resource:TC:N0"}, "precheck");
  }
  {
    llvm::json::Object problem = makeV1DiagnosticProblem();
    problem["smem_budget"] = 1;
    problem.getArray("buffers")->push_back(llvm::json::Object{
        {"id", 7},
        {"alloc_node", 0},
        {"kind", "smem"},
        {"size_bytes", 2},
        {"tmem_cols", 0},
        {"consumers", llvm::json::Array{0}},
    });
    expectUnsatDiagnostic(serializeJsonObject(std::move(problem)), {"smem"},
                          {"smem:buffer7"});
  }
  {
    llvm::json::Object problem = makeV1DiagnosticProblem();
    problem["tmem_col_limit"] = 1;
    problem.getArray("buffers")->push_back(llvm::json::Object{
        {"id", 9},
        {"alloc_node", 0},
        {"kind", "tmem"},
        {"size_bytes", 0},
        {"tmem_cols", 2},
        {"consumers", llvm::json::Array{0}},
    });
    expectUnsatDiagnostic(serializeJsonObject(std::move(problem)), {"tmem"},
                          {"tmem:buffer9"});
  }
}

TEST(Z3JointSolverV2Test, UnsatDiagnosticsCoverNativeConstraintKinds) {
  for (llvm::StringRef mode :
       {llvm::StringRef("partition"), llvm::StringRef("joint")}) {
    SCOPED_TRACE(mode.str());
    auto problem =
        parseJsonObject(mode == "partition" ? kPartitionObjectiveProblem
                                            : kJointSolverV2Problem);
    ASSERT_TRUE(succeeded(problem));
    (*problem)["max_wgs"] = 1;
    forceDifferentWarpGroups(*problem);
    expectUnsatDiagnostic(serializeJsonObject(std::move(*problem)),
                          {"warp-group"},
                          {"warp-group:cluster10", "warp-group:cluster30"});
  }
  {
    auto problem = parseJsonObject(kJointSolverV2Problem);
    ASSERT_TRUE(succeeded(problem));
    forceDifferentWarpGroups(*problem);
    (*problem)["reg_budget"] = 1;
    (*problem)["warp_footprint"] = llvm::json::Array{0, 1, 1, 1, 1, 1, 1, 1, 1};
    expectUnsatDiagnostic(serializeJsonObject(std::move(*problem)),
                          {"register"}, {"register:wg0", "register:wg1"});
  }
  {
    auto problem = parseJsonObject(kJointSolverV2Problem);
    ASSERT_TRUE(succeeded(problem));
    auto *loweringTemplate =
        problem->getArray("lowering_templates")->front().getAsObject();
    ASSERT_NE(loweringTemplate, nullptr);
    (*loweringTemplate)["relation"] = "always";
    auto *event = loweringTemplate->getArray("events")->front().getAsObject();
    ASSERT_NE(event, nullptr);
    (*event)["issue_duration"] = 5;
    expectUnsatDiagnostic(serializeJsonObject(std::move(*problem)),
                          {"lowering"}, {"lowering:template0:event7"});
  }
  {
    auto problem = parseJsonObject(kPartitionObjectiveProblem);
    ASSERT_TRUE(succeeded(problem));
    (*problem)["mode"] = "joint";
    (*problem)["smem_budget"] = 1;
    forceDifferentWarpGroups(*problem);
    auto *edge = problem->getArray("edges")->front().getAsObject();
    ASSERT_NE(edge, nullptr);
    (*edge)["chan_bytes"] = 2;
    expectUnsatDiagnostic(serializeJsonObject(std::move(*problem)), {"smem"},
                          {"smem:channel:N0->N1:r0"});
  }
  {
    auto problem = parseJsonObject(kJointSolverV2Problem);
    ASSERT_TRUE(succeeded(problem));
    (*problem)["fixed_smem"] = 1;
    expectUnsatDiagnostic(serializeJsonObject(std::move(*problem)), {"smem"},
                          {"smem:fixed"});
  }
}

TEST(Z3JointSolverTest, UnknownDiagnosticHasNoUnsatCore) {
  llvm::json::Array nodes;
  for (int64_t id = 0; id < 200; ++id) {
    nodes.push_back(llvm::json::Object{
        {"id", id},
        {"pipeline", "TC"},
        {"duration", 1},
        {"streaming", false},
    });
  }
  llvm::json::Object problem{
      {"version", "joint-solver-0.1"},
      {"min_ii", 200},
      {"max_ii", 200},
      {"smem_budget", 0},
      {"tmem_col_limit", 0},
      {"time_limit_s", 0.000001},
      {"streaming_vl", false},
      {"nodes", std::move(nodes)},
      {"edges", llvm::json::Array{}},
      {"buffers", llvm::json::Array{}},
  };
  auto output = runZ3JointSolver(serializeJsonObject(std::move(problem)));
  ASSERT_TRUE(succeeded(output));
  auto response = parseJsonObject(*output);
  ASSERT_TRUE(succeeded(response));
  EXPECT_EQ(response->getString("status"), "inconclusive");
  EXPECT_EQ(response->getBoolean("proven_unsat"), false);
  EXPECT_EQ(response->getString("backend_status"), "UNKNOWN");
  EXPECT_EQ(response->get("unsat_core"), nullptr);
  auto *diagnostic = response->getObject("diagnostic");
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->getString("schema"), "joint-solver-diagnostic-0.1");
  EXPECT_EQ(diagnostic->getString("status"), "inconclusive");
  EXPECT_EQ(diagnostic->getString("backendStatus"), "UNKNOWN");
  EXPECT_EQ(diagnostic->get("core"), nullptr);
}

TEST(Z3JointSolverTest, InvalidSchemaReturnsFailure) {
  EXPECT_TRUE(
      failed(runZ3JointSolver(R"json({"version": "unsupported"})json")));
}

TEST(Z3JointSolverTest, OptimizesFullCompositeObjective) {
  auto output = runZ3JointSolver(kCompositeObjectiveProblem);
  ASSERT_TRUE(succeeded(output));

  auto parsed = llvm::json::parse(*output);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *response = parsed->getAsObject();
  ASSERT_NE(response, nullptr);
  auto *cycles = response->getObject("cycles");
  auto objective = response->getNumber("objective");
  ASSERT_NE(cycles, nullptr);
  ASSERT_TRUE(objective);
  auto consumerCycle = cycles->getInteger("1");
  ASSERT_TRUE(consumerCycle);
  EXPECT_EQ(*consumerCycle, 3);
  EXPECT_DOUBLE_EQ(*objective, -21504.0);
}

TEST(Z3JointSolverTest, UsesExactDepthInCompositeObjective) {
  auto output = runZ3JointSolver(kDepthObjectiveProblem);
  ASSERT_TRUE(succeeded(output));

  auto parsed = llvm::json::parse(*output);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *response = parsed->getAsObject();
  ASSERT_NE(response, nullptr);
  auto ii = response->getInteger("ii");
  auto *cycles = response->getObject("cycles");
  auto *depths = response->getObject("buffer_depths");
  ASSERT_TRUE(ii);
  ASSERT_NE(cycles, nullptr);
  ASSERT_NE(depths, nullptr);
  auto consumerCycle = cycles->getInteger("1");
  auto depth = depths->getInteger("9");
  ASSERT_TRUE(consumerCycle);
  ASSERT_TRUE(depth);
  EXPECT_EQ(*consumerCycle / *ii, 1);
  EXPECT_EQ(*depth, 2);
}

TEST(Z3JointSolverV2Test, JointProblemReturnsPublicContract) {
  auto output = runZ3JointSolver(kJointSolverV2Problem);
  ASSERT_TRUE(succeeded(output));

  auto parsed = llvm::json::parse(*output);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *response = parsed->getAsObject();
  ASSERT_NE(response, nullptr);
  auto status = response->getString("status");
  auto usedWGs = response->getInteger("used_wgs");
  auto objective = response->getNumber("objective");
  auto *warpGroups = response->getObject("wg");
  auto *loweringPlan = response->getObject("lowering_plan");
  ASSERT_TRUE(status);
  ASSERT_TRUE(usedWGs);
  ASSERT_TRUE(objective);
  ASSERT_NE(warpGroups, nullptr);
  ASSERT_NE(loweringPlan, nullptr);
  EXPECT_EQ(*status, "ok");
  EXPECT_EQ(*usedWGs, 2);
  EXPECT_DOUBLE_EQ(*objective, -2.0);

  auto firstWarpGroup = warpGroups->getInteger("10");
  auto secondWarpGroup = warpGroups->getInteger("30");
  ASSERT_TRUE(firstWarpGroup);
  ASSERT_TRUE(secondWarpGroup);
  EXPECT_NE(*firstWarpGroup, *secondWarpGroup);

  auto planVersion = loweringPlan->getString("version");
  auto *templates = loweringPlan->getArray("templates");
  ASSERT_TRUE(planVersion);
  ASSERT_NE(templates, nullptr);
  EXPECT_EQ(*planVersion, "lowering-plan-0.1");
  ASSERT_EQ(templates->size(), 1);
  auto *loweringTemplate = templates->front().getAsObject();
  ASSERT_NE(loweringTemplate, nullptr);
  auto templateId = loweringTemplate->getInteger("id");
  auto active = loweringTemplate->getBoolean("active");
  auto *events = loweringTemplate->getArray("events");
  ASSERT_TRUE(templateId);
  ASSERT_TRUE(active);
  ASSERT_NE(events, nullptr);
  EXPECT_EQ(*templateId, 0);
  EXPECT_TRUE(*active);
  ASSERT_EQ(events->size(), 1);
  auto *event = events->front().getAsObject();
  ASSERT_NE(event, nullptr);
  auto eventId = event->getInteger("id");
  ASSERT_TRUE(eventId);
  EXPECT_EQ(*eventId, 7);
  auto eventCycle = event->getInteger("cycle");
  auto eventWarpGroup = event->getInteger("wg");
  auto streamOrder = event->getInteger("stream_order");
  ASSERT_TRUE(eventCycle);
  ASSERT_TRUE(eventWarpGroup);
  ASSERT_TRUE(streamOrder);
  EXPECT_EQ(*eventCycle, -1);
  EXPECT_EQ(*eventWarpGroup, *firstWarpGroup);
  EXPECT_EQ(*streamOrder, 0);
}

TEST(Z3JointSolverV2Test, HopperGemmMatchesStableGolden) {
  expectGemmGolden(kHopperGemmModel);
}

TEST(Z3JointSolverV2Test, BlackwellGemmMatchesStableGolden) {
  expectGemmGolden(kBlackwellGemmModel);
}

TEST(Z3JointSolverV2Test, GemmValidationRejectsIllegalMutations) {
  for (const GemmMachineModel *model :
       {&kHopperGemmModel, &kBlackwellGemmModel}) {
    SCOPED_TRACE(model->name);
    std::string problemJson = serializeJsonObject(makeGemmProblem(*model));
    auto output = runZ3JointSolver(problemJson);
    ASSERT_TRUE(succeeded(output));
    expectProductionValidity(problemJson, *output, true);

    auto expectProblemMutationInvalid =
        [&](llvm::StringRef name,
            const std::function<void(llvm::json::Object &)> &mutate) {
          SCOPED_TRACE(name.str());
          llvm::json::Object problem = makeGemmProblem(*model);
          mutate(problem);
          expectProductionValidity(serializeJsonObject(std::move(problem)),
                                   *output, false);
        };

    expectProblemMutationInvalid("cross-WG round trip", [](auto &problem) {
      auto *edge = findEdge(problem, 2, 5);
      ASSERT_NE(edge, nullptr);
      (*edge)["rt"] = 100;
    });
    expectProblemMutationInvalid("fixed SMEM", [&](auto &problem) {
      problem["fixed_smem"] = model->smemBudget + 1;
    });
    expectProblemMutationInvalid("channel SMEM", [](auto &problem) {
      auto *edge = findEdge(problem, 2, 5);
      ASSERT_NE(edge, nullptr);
      (*edge)["chan_bytes"] = 9;
    });
    expectProblemMutationInvalid(
        "register budget", [](auto &problem) { problem["reg_budget"] = 55; });
    expectProblemMutationInvalid("dependency", [&](auto &problem) {
      auto *edge = findEdge(problem, 5, 6);
      ASSERT_NE(edge, nullptr);
      (*edge)["latency"] = model->mmaLatency + 100;
    });
    if (model->accumulatorInTmem) {
      expectProblemMutationInvalid("TMEM budget", [&](auto &problem) {
        problem["tmem_budget_bytes"] = model->tmemBudget - 1;
      });
    }

    auto expectSolutionMutationInvalid =
        [&](llvm::StringRef name,
            const std::function<void(llvm::json::Object &)> &mutate) {
          SCOPED_TRACE(name.str());
          auto solution = parseJsonObject(*output);
          ASSERT_TRUE(succeeded(solution));
          mutate(*solution);
          expectProductionValidity(
              problemJson, serializeJsonObject(std::move(*solution)), false);
        };

    expectSolutionMutationInvalid("fixed stage", [&](auto &solution) {
      auto *cycles = solution.getObject("cycles");
      ASSERT_NE(cycles, nullptr);
      auto cycle = cycles->getInteger("6");
      ASSERT_TRUE(cycle);
      (*cycles)["6"] = *cycle + model->ii;
    });
    expectSolutionMutationInvalid("lowering placement", [](auto &solution) {
      auto *plan = solution.getObject("lowering_plan");
      auto *templates = plan ? plan->getArray("templates") : nullptr;
      auto *loweringTemplate = templates && !templates->empty()
                                   ? templates->front().getAsObject()
                                   : nullptr;
      auto *events =
          loweringTemplate ? loweringTemplate->getArray("events") : nullptr;
      auto *event =
          events && !events->empty() ? events->front().getAsObject() : nullptr;
      ASSERT_NE(event, nullptr);
      auto cycle = event->getInteger("cycle");
      ASSERT_TRUE(cycle);
      (*event)["cycle"] = *cycle + 1;
    });
  }
}

TEST(Z3JointSolverTest, StageAwareOracleFindsCycleBeyondModuloInterval) {
  auto oracle = enumerateStageAwareSchedule(kStageAwareOracleProblem);
  ASSERT_TRUE(succeeded(oracle));
  EXPECT_EQ(oracle->ii, 1);
  EXPECT_EQ(oracle->cycles, (std::map<int64_t, int64_t>{{0, 0}, {1, 2}}));
  EXPECT_GT(oracle->cycles.at(1), oracle->ii - 1);

  auto output = runZ3JointSolver(kStageAwareOracleProblem);
  ASSERT_TRUE(succeeded(output));
  expectProductionValidity(kStageAwareOracleProblem, *output, true);
  auto response = parseJsonObject(*output);
  ASSERT_TRUE(succeeded(response));
  auto *cycles = response->getObject("cycles");
  ASSERT_NE(cycles, nullptr);
  EXPECT_EQ(cycles->getInteger("0"), 0);
  EXPECT_EQ(cycles->getInteger("1"), 2);
}

TEST(Z3JointSolverTest, BinarySearchMatchesBufferAndStreamingOracle) {
  auto oracle = enumerateStageAwareSchedule(kBinaryOracleProblem);
  ASSERT_TRUE(succeeded(oracle));
  EXPECT_EQ(oracle->ii, 2);
  EXPECT_EQ(oracle->cycles,
            (std::map<int64_t, int64_t>{{0, 0}, {1, 0}, {2, 3}}));

  auto output = runZ3JointSolver(kBinaryOracleProblem);
  ASSERT_TRUE(succeeded(output));
  expectProductionValidity(kBinaryOracleProblem, *output, true);
  auto response = parseJsonObject(*output);
  ASSERT_TRUE(succeeded(response));
  EXPECT_EQ(response->getInteger("ii"), oracle->ii);
  auto *depths = response->getObject("buffer_depths");
  ASSERT_NE(depths, nullptr);
  EXPECT_EQ(depths->getInteger("7"), 2);
}

TEST(Z3JointSolverV2Test, LoweringEventKindsMatchSchema) {
  constexpr const char *acceptedKinds[] = {
      "wait",       "arrive", "expect",   "local_store",
      "local_load", "fence",  "tc_commit"};
  for (const char *kind : acceptedKinds) {
    SCOPED_TRACE(kind);
    auto problem = withFirstLoweringEventKind(kJointSolverV2Problem, kind);
    ASSERT_TRUE(succeeded(problem));
    auto output = runZ3JointSolver(*problem);
    ASSERT_TRUE(succeeded(output));
  }

  auto validOutput = runZ3JointSolver(kJointSolverV2Problem);
  ASSERT_TRUE(succeeded(validOutput));
  auto invalidProblem =
      withFirstLoweringEventKind(kJointSolverV2Problem, "commit");
  ASSERT_TRUE(succeeded(invalidProblem));
  EXPECT_TRUE(failed(runZ3JointSolver(*invalidProblem)));
}

TEST(Z3JointSolverV2Test, CrossIssueCostCanPreferFewerWarpGroups) {
  auto output = runZ3JointSolver(kJointCrossIssueObjectiveProblem);
  ASSERT_TRUE(succeeded(output));
  auto parsed = llvm::json::parse(*output);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *response = parsed->getAsObject();
  ASSERT_NE(response, nullptr);
  auto usedWGs = response->getInteger("used_wgs");
  auto objective = response->getNumber("objective");
  auto *warpGroups = response->getObject("wg");
  ASSERT_TRUE(usedWGs);
  ASSERT_TRUE(objective);
  ASSERT_NE(warpGroups, nullptr);
  auto firstWarpGroup = warpGroups->getInteger("10");
  auto secondWarpGroup = warpGroups->getInteger("30");
  ASSERT_TRUE(firstWarpGroup);
  ASSERT_TRUE(secondWarpGroup);
  EXPECT_EQ(*usedWGs, 1);
  EXPECT_EQ(*firstWarpGroup, *secondWarpGroup);
  EXPECT_DOUBLE_EQ(*objective, 1023.0);
}

TEST(Z3JointSolverV2Test, PartitionOverlapRemainsASoftCost) {
  auto output = runZ3JointSolver(kPartitionObjectiveProblem);
  ASSERT_TRUE(succeeded(output));
  auto parsed = llvm::json::parse(*output);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *response = parsed->getAsObject();
  ASSERT_NE(response, nullptr);
  auto usedWGs = response->getInteger("used_wgs");
  auto objective = response->getNumber("objective");
  auto *warpGroups = response->getObject("wg");
  ASSERT_TRUE(usedWGs);
  ASSERT_TRUE(objective);
  ASSERT_NE(warpGroups, nullptr);
  auto firstWarpGroup = warpGroups->getInteger("10");
  auto secondWarpGroup = warpGroups->getInteger("30");
  ASSERT_TRUE(firstWarpGroup);
  ASSERT_TRUE(secondWarpGroup);
  EXPECT_EQ(*usedWGs, 1);
  EXPECT_EQ(*firstWarpGroup, *secondWarpGroup);
  EXPECT_DOUBLE_EQ(*objective, 1.0);
}

TEST(Z3JointSolverV2Test, MinimizesBlockingInversionsAfterPrimaryObjective) {
  auto output = runZ3JointSolver(kBlockingObjectiveProblem);
  ASSERT_TRUE(succeeded(output));
  auto parsed = llvm::json::parse(*output);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *response = parsed->getAsObject();
  ASSERT_NE(response, nullptr);
  auto objective = response->getNumber("objective");
  auto loweringObjective = response->getNumber("lowering_objective");
  auto *cycles = response->getObject("cycles");
  ASSERT_TRUE(objective);
  ASSERT_TRUE(loweringObjective);
  ASSERT_NE(cycles, nullptr);
  auto anchorCycle = cycles->getInteger("1");
  auto independentCycle = cycles->getInteger("2");
  ASSERT_TRUE(anchorCycle);
  ASSERT_TRUE(independentCycle);
  EXPECT_DOUBLE_EQ(*objective, 1023.0);
  EXPECT_DOUBLE_EQ(*loweringObjective, 0.0);
  EXPECT_GE(*anchorCycle, *independentCycle);
}

TEST(Z3JointSolverV2Test, TmemBudgetBelowMinimumDepthReturnsInfeasible) {
  std::string problem = kJointSolverV2Problem.str();
  constexpr llvm::StringLiteral kBudget = "\"tmem_budget_bytes\": 12";
  size_t budgetPosition = problem.find(kBudget.str());
  ASSERT_NE(budgetPosition, std::string::npos);
  problem.replace(budgetPosition, kBudget.size(), "\"tmem_budget_bytes\": 11");

  auto output = runZ3JointSolver(problem);
  ASSERT_TRUE(succeeded(output));

  auto parsed = llvm::json::parse(*output);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *response = parsed->getAsObject();
  ASSERT_NE(response, nullptr);
  auto status = response->getString("status");
  auto provenUnsat = response->getBoolean("proven_unsat");
  auto backendStatus = response->getString("backend_status");
  ASSERT_TRUE(status);
  ASSERT_TRUE(provenUnsat);
  ASSERT_TRUE(backendStatus);
  EXPECT_EQ(*status, "infeasible");
  EXPECT_TRUE(*provenUnsat);
  EXPECT_EQ(*backendStatus, "INFEASIBLE");
}

TEST(Z3JointSolverV2Test, CorrectnessRejectsImpossibleDependency) {
  auto parsed = llvm::json::parse(kJointSolverV2Problem);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *root = parsed->getAsObject();
  auto *edges = root ? root->getArray("edges") : nullptr;
  ASSERT_NE(edges, nullptr);
  edges->push_back(llvm::json::Object{
      {"src", 0},
      {"dst", 1},
      {"src_result_idx", 0},
      {"latency", 8},
      {"distance", 0},
      {"freq", 1},
      {"rt", 0},
      {"xissue", 0},
      {"chan_bytes", 0},
      {"src_cluster", 10},
      {"dst_cluster", 30},
  });

  auto output = runZ3JointSolver(serializeJsonObject(std::move(*root)));
  ASSERT_TRUE(succeeded(output));
  auto response = llvm::json::parse(*output);
  if (!response) {
    ADD_FAILURE() << llvm::toString(response.takeError());
    return;
  }
  auto *object = response->getAsObject();
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->getString("status"), "infeasible");
  EXPECT_EQ(object->getBoolean("proven_unsat"), true);
}

TEST(Z3JointSolverV2Test, CorrectnessRejectsUnsatisfiableWGConflict) {
  auto parsed = llvm::json::parse(kJointSolverV2Problem);
  if (!parsed) {
    ADD_FAILURE() << llvm::toString(parsed.takeError());
    return;
  }
  auto *root = parsed->getAsObject();
  ASSERT_NE(root, nullptr);
  (*root)["max_wgs"] = 1;
  llvm::json::Array conflicts;
  conflicts.push_back(llvm::json::Array{10, 30});
  (*root)["warp_group_conflicts"] = std::move(conflicts);

  auto output = runZ3JointSolver(serializeJsonObject(std::move(*root)));
  ASSERT_TRUE(succeeded(output));
  auto response = llvm::json::parse(*output);
  if (!response) {
    ADD_FAILURE() << llvm::toString(response.takeError());
    return;
  }
  auto *object = response->getAsObject();
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->getString("status"), "infeasible");
  EXPECT_EQ(object->getBoolean("proven_unsat"), true);
}

TEST(Z3JointSolverV2Test, PreservesCommittedStages) {
  auto problem = parseJsonObject(kJointSolverV2Problem);
  ASSERT_TRUE(succeeded(problem));
  auto *nodes = problem->getArray("nodes");
  ASSERT_NE(nodes, nullptr);
  ASSERT_EQ(nodes->size(), 2);
  auto *producer = (*nodes)[0].getAsObject();
  auto *consumer = (*nodes)[1].getAsObject();
  ASSERT_NE(producer, nullptr);
  ASSERT_NE(consumer, nullptr);
  (*consumer)["cycle"] = 5;
  auto producerCommittedCycle = producer->getInteger("cycle");
  auto consumerCommittedCycle = consumer->getInteger("cycle");
  ASSERT_TRUE(producerCommittedCycle);
  ASSERT_TRUE(consumerCommittedCycle);

  auto response = runJsonProblem(std::move(*problem));
  ASSERT_TRUE(succeeded(response));
  EXPECT_EQ(response->getString("status"), "ok");
  auto *cycles = response->getObject("cycles");
  ASSERT_NE(cycles, nullptr);
  auto producerSolvedCycle = cycles->getInteger("0");
  auto consumerSolvedCycle = cycles->getInteger("1");
  ASSERT_TRUE(producerSolvedCycle);
  ASSERT_TRUE(consumerSolvedCycle);
  EXPECT_EQ(*producerSolvedCycle / 4, *producerCommittedCycle / 4);
  EXPECT_EQ(*consumerSolvedCycle / 4, *consumerCommittedCycle / 4);
}

TEST(Z3JointSolverV2Test, SerializesSameAnchorEventsOnOwnerWarpGroup) {
  auto problem = parseJsonObject(kJointSolverV2Problem);
  ASSERT_TRUE(succeeded(problem));
  auto *templates = problem->getArray("lowering_templates");
  ASSERT_NE(templates, nullptr);
  templates->push_back(llvm::json::Object{
      {"id", 1},
      {"relation", "different_wg"},
      {"src_node", 0},
      {"dst_node", 1},
      {"src_cluster", 10},
      {"dst_cluster", 30},
      {"events", llvm::json::Array{llvm::json::Object{
                     {"id", 8},
                     {"kind", "arrive"},
                     {"owner", "src"},
                     {"anchor_node", 0},
                     {"placement", "before"},
                     {"pipeline", "NONE"},
                     {"issue_duration", 1},
                     {"completion_latency", 0},
                     {"blocking", false},
                     {"async", false},
                     {"distance", 0},
                     {"frequency", 1},
                     {"bytes", 0},
                     {"depth", 0},
                     {"semaphore", ""},
                 }}},
  });

  auto response = runJsonProblem(std::move(*problem));
  ASSERT_TRUE(succeeded(response));
  EXPECT_EQ(response->getString("status"), "ok");
  auto *cycles = response->getObject("cycles");
  auto *warpGroups = response->getObject("wg");
  auto *loweringPlan = response->getObject("lowering_plan");
  ASSERT_NE(cycles, nullptr);
  ASSERT_NE(warpGroups, nullptr);
  ASSERT_NE(loweringPlan, nullptr);
  auto anchorCycle = cycles->getInteger("0");
  auto ownerWarpGroup = warpGroups->getInteger("10");
  auto *templatePlans = loweringPlan->getArray("templates");
  ASSERT_TRUE(anchorCycle);
  ASSERT_TRUE(ownerWarpGroup);
  ASSERT_NE(templatePlans, nullptr);
  ASSERT_EQ(templatePlans->size(), 2);

  int64_t expectedCycle = *anchorCycle - 2;
  for (size_t index = 0; index < templatePlans->size(); ++index) {
    auto *templatePlan = (*templatePlans)[index].getAsObject();
    ASSERT_NE(templatePlan, nullptr);
    EXPECT_EQ(templatePlan->getBoolean("active"), true);
    auto *events = templatePlan->getArray("events");
    ASSERT_NE(events, nullptr);
    ASSERT_EQ(events->size(), 1);
    auto *event = events->front().getAsObject();
    ASSERT_NE(event, nullptr);
    EXPECT_EQ(event->getInteger("cycle"), expectedCycle++);
    EXPECT_EQ(event->getInteger("wg"), *ownerWarpGroup);
    EXPECT_EQ(event->getInteger("stream_order"), static_cast<int64_t>(index));
  }
}

TEST(Z3JointSolverV2Test, LoweringFrequencyConsumesSlotsOnlyWhenActive) {
  auto inactiveProblem = parseJsonObject(kJointSolverV2Problem);
  ASSERT_TRUE(succeeded(inactiveProblem));
  auto *inactiveTemplates = inactiveProblem->getArray("lowering_templates");
  ASSERT_NE(inactiveTemplates, nullptr);
  auto *inactiveTemplate = inactiveTemplates->front().getAsObject();
  ASSERT_NE(inactiveTemplate, nullptr);
  auto *inactiveEvents = inactiveTemplate->getArray("events");
  ASSERT_NE(inactiveEvents, nullptr);
  auto *inactiveEvent = inactiveEvents->front().getAsObject();
  ASSERT_NE(inactiveEvent, nullptr);
  (*inactiveEvent)["frequency"] = 5;

  auto inactiveResponse = runJsonProblem(std::move(*inactiveProblem));
  ASSERT_TRUE(succeeded(inactiveResponse));
  EXPECT_EQ(inactiveResponse->getString("status"), "ok");
  auto *inactivePlan = inactiveResponse->getObject("lowering_plan");
  ASSERT_NE(inactivePlan, nullptr);
  auto *inactiveTemplatePlans = inactivePlan->getArray("templates");
  ASSERT_NE(inactiveTemplatePlans, nullptr);
  auto *inactiveTemplatePlan = inactiveTemplatePlans->front().getAsObject();
  ASSERT_NE(inactiveTemplatePlan, nullptr);
  EXPECT_EQ(inactiveTemplatePlan->getBoolean("active"), false);
  auto *inactiveEventPlans = inactiveTemplatePlan->getArray("events");
  ASSERT_NE(inactiveEventPlans, nullptr);
  EXPECT_TRUE(inactiveEventPlans->empty());

  auto activeProblem = parseJsonObject(kJointSolverV2Problem);
  ASSERT_TRUE(succeeded(activeProblem));
  auto *activeTemplates = activeProblem->getArray("lowering_templates");
  ASSERT_NE(activeTemplates, nullptr);
  auto *activeTemplate = activeTemplates->front().getAsObject();
  ASSERT_NE(activeTemplate, nullptr);
  (*activeTemplate)["relation"] = "always";
  auto *activeEvents = activeTemplate->getArray("events");
  ASSERT_NE(activeEvents, nullptr);
  auto *activeEvent = activeEvents->front().getAsObject();
  ASSERT_NE(activeEvent, nullptr);
  (*activeEvent)["frequency"] = 5;

  auto activeResponse = runJsonProblem(std::move(*activeProblem));
  ASSERT_TRUE(succeeded(activeResponse));
  EXPECT_EQ(activeResponse->getString("status"), "infeasible");
}

TEST(Z3JointSolverV2Test, PartitionChannelsDistinguishProducerResults) {
  auto problem = parseJsonObject(kPartitionObjectiveProblem);
  ASSERT_TRUE(succeeded(problem));
  (*problem)["smem_budget"] = 60;
  llvm::json::Array conflicts;
  conflicts.push_back(llvm::json::Array{10, 30});
  (*problem)["warp_group_conflicts"] = std::move(conflicts);

  std::string baseProblemJson = serializeJsonObject(std::move(*problem));
  auto baseProblem = parseJsonObject(baseProblemJson);
  ASSERT_TRUE(succeeded(baseProblem));
  auto baseResponse = runJsonProblem(std::move(*baseProblem));
  ASSERT_TRUE(succeeded(baseResponse));
  EXPECT_EQ(baseResponse->getString("status"), "ok");

  auto oneResultProblem = parseJsonObject(baseProblemJson);
  ASSERT_TRUE(succeeded(oneResultProblem));
  auto *edges = oneResultProblem->getArray("edges");
  ASSERT_NE(edges, nullptr);
  auto *first = edges->front().getAsObject();
  ASSERT_NE(first, nullptr);
  (*first)["src_result_idx"] = 0;
  (*first)["chan_bytes"] = 60;
  (*first)["xissue"] = 0;

  std::string oneResultJson = serializeJsonObject(std::move(*oneResultProblem));
  auto oneResultInput = parseJsonObject(oneResultJson);
  ASSERT_TRUE(succeeded(oneResultInput));
  auto oneResultResponse = runJsonProblem(std::move(*oneResultInput));
  ASSERT_TRUE(succeeded(oneResultResponse));
  EXPECT_EQ(oneResultResponse->getString("status"), "ok");

  auto twoResultProblem = parseJsonObject(oneResultJson);
  ASSERT_TRUE(succeeded(twoResultProblem));
  edges = twoResultProblem->getArray("edges");
  ASSERT_NE(edges, nullptr);
  edges->push_back(llvm::json::Object{
      {"src", 0},
      {"dst", 1},
      {"src_result_idx", 1},
      {"latency", 0},
      {"distance", 1},
      {"freq", 1},
      {"rt", 0},
      {"xissue", 0},
      {"chan_bytes", 60},
      {"src_cluster", 10},
      {"dst_cluster", 30},
  });
  (*twoResultProblem)["smem_budget"] = 120;
  std::string twoResultJson = serializeJsonObject(std::move(*twoResultProblem));
  auto feasibleProblem = parseJsonObject(twoResultJson);
  ASSERT_TRUE(succeeded(feasibleProblem));
  auto feasibleResponse = runJsonProblem(std::move(*feasibleProblem));
  ASSERT_TRUE(succeeded(feasibleResponse));
  EXPECT_EQ(feasibleResponse->getString("status"), "ok");

  auto infeasibleProblem = parseJsonObject(twoResultJson);
  ASSERT_TRUE(succeeded(infeasibleProblem));
  (*infeasibleProblem)["smem_budget"] = 100;
  auto response = runJsonProblem(std::move(*infeasibleProblem));
  ASSERT_TRUE(succeeded(response));
  EXPECT_EQ(response->getString("status"), "infeasible");
}

TEST(Z3JointSolverV2Test, PartitionPayloadRequiresV2Schema) {
  std::string problem = kPartitionObjectiveProblem.str();
  constexpr llvm::StringLiteral kVersion = "\"version\": \"joint-solver-0.2\"";
  size_t versionPosition = problem.find(kVersion.str());
  ASSERT_NE(versionPosition, std::string::npos);
  problem.replace(versionPosition, kVersion.size(),
                  "\"version\": \"joint-solver-0.1\"");

  EXPECT_TRUE(failed(runZ3JointSolver(problem)));
}

TEST(Z3JointSolverV2Test, AutomaticallySeparatesTensorCoreSoftwareReaders) {
  for (const char *mode : {"partition", "joint"}) {
    for (const char *producer : {"TC", "MFMA"}) {
      for (const char *consumer : {"CUDA", "SFU"}) {
        for (int64_t distance : {0, 1}) {
          SCOPED_TRACE(std::string(mode) + "/" + producer + "/" + consumer +
                       "/distance=" + std::to_string(distance));
          auto problem = makePipelineGroupingProblem(mode, producer, consumer,
                                                     distance, 2);
          ASSERT_TRUE(succeeded(problem));
          auto response = runJsonProblem(std::move(*problem));
          ASSERT_TRUE(succeeded(response));
          EXPECT_EQ(response->getString("status"), "ok");
          auto *warpGroups = response->getObject("wg");
          ASSERT_NE(warpGroups, nullptr);
          auto producerWarpGroup = warpGroups->getInteger("10");
          auto consumerWarpGroup = warpGroups->getInteger("30");
          ASSERT_TRUE(producerWarpGroup);
          ASSERT_TRUE(consumerWarpGroup);
          EXPECT_NE(*producerWarpGroup, *consumerWarpGroup);
        }
      }
    }
  }
}

TEST(Z3JointSolverV2Test, UnsafeReaderNeedsASecondWarpGroup) {
  for (const char *mode : {"partition", "joint"}) {
    for (int64_t distance : {0, 1}) {
      SCOPED_TRACE(std::string(mode) + "/distance=" + std::to_string(distance));
      auto problem =
          makePipelineGroupingProblem(mode, "TC", "CUDA", distance, 1);
      ASSERT_TRUE(succeeded(problem));
      auto response = runJsonProblem(std::move(*problem));
      ASSERT_TRUE(succeeded(response));
      EXPECT_EQ(response->getString("status"), "infeasible");
    }
  }
}

TEST(Z3JointSolverV2Test, SafePipelinePairsMayShareOneWarpGroup) {
  for (const char *mode : {"partition", "joint"}) {
    for (int64_t distance : {0, 1}) {
      for (const auto &[producer, consumer] :
           {std::pair{"TMA", "CUDA"}, std::pair{"TC", "TC"}}) {
        SCOPED_TRACE(std::string(mode) + "/" + producer + "/" + consumer +
                     "/distance=" + std::to_string(distance));
        auto problem =
            makePipelineGroupingProblem(mode, producer, consumer, distance, 1);
        ASSERT_TRUE(succeeded(problem));
        auto response = runJsonProblem(std::move(*problem));
        ASSERT_TRUE(succeeded(response));
        EXPECT_EQ(response->getString("status"), "ok");
        auto *warpGroups = response->getObject("wg");
        ASSERT_NE(warpGroups, nullptr);
        EXPECT_EQ(warpGroups->getInteger("10"), warpGroups->getInteger("30"));
      }
    }
  }
}

// Reproducible solver runs.
//
// What makes a solve reproducible is the deterministic budget: `rlimit` counts
// Z3 work units rather than seconds, so the search stops at the same point no
// matter how fast or how loaded the host is. The wall clock remains only as a
// backstop. These tests pin that property, and pin that the two budgets stay
// distinguishable when one of them runs out.

static const std::vector<std::pair<llvm::StringRef, llvm::StringRef>> &
fixtureCorpus() {
  static const std::vector<std::pair<llvm::StringRef, llvm::StringRef>> corpus{
      {"v01/feasible", kFeasibleJointSolverProblem},
      {"v01/infeasible", kInfeasibleJointSolverProblem},
      {"v01/composite", kCompositeObjectiveProblem},
      {"v01/depth", kDepthObjectiveProblem},
      {"v02/partition", kPartitionObjectiveProblem},
      {"v02/blocking", kBlockingObjectiveProblem},
      {"v02/joint", kJointSolverV2Problem},
      {"v02/cross_issue", kJointCrossIssueObjectiveProblem},
  };
  return corpus;
}

static FailureOr<int64_t> statsInteger(const llvm::json::Object &response,
                                       llvm::StringRef key) {
  const auto *stats = response.getObject("stats");
  if (!stats)
    return failure();
  auto value = stats->getInteger(key);
  if (!value)
    return failure();
  return *value;
}

static FailureOr<llvm::json::Object> runWithRLimit(llvm::StringRef problemJson,
                                                   int64_t rlimit) {
  auto problem = parseJsonObject(problemJson);
  if (failed(problem))
    return failure();
  (*problem)["rlimit"] = rlimit;
  return runJsonProblem(std::move(*problem));
}

TEST(Z3JointSolverBudgetTest, ReportsTheBudgetItRanUnderAndWhatItCost) {
  for (const auto &[name, problemJson] : fixtureCorpus()) {
    SCOPED_TRACE(name.str());
    auto response = runJsonProblem(*parseJsonObject(problemJson));
    ASSERT_TRUE(succeeded(response));
    auto budget = statsInteger(*response, "rlimit");
    auto used = statsInteger(*response, "rlimit_used");
    ASSERT_TRUE(succeeded(budget));
    ASSERT_TRUE(succeeded(used));
    EXPECT_GT(*budget, 0);
    EXPECT_GT(*used, 0);
  }
}

// The default budget is only useful if it sits far above what real problems
// consume: set too low, legitimate solves would return inconclusive and
// silently fall back to the heuristic scheduler — a performance regression no
// other test would catch. Asserting against the budget the response reports
// keeps this check from drifting out of sync with the constant it guards.
// The logged numbers are the calibration measurements; re-read them after a
// Z3 upgrade, because Z3's work-unit accounting is not stable across releases.
TEST(Z3JointSolverBudgetTest, FixtureCorpusFitsFarInsideTheDefaultBudget) {
  constexpr int64_t kRequiredHeadroom = 8;
  for (const auto &[name, problemJson] : fixtureCorpus()) {
    SCOPED_TRACE(name.str());
    auto response = runJsonProblem(*parseJsonObject(problemJson));
    ASSERT_TRUE(succeeded(response));
    auto budget = statsInteger(*response, "rlimit");
    auto used = statsInteger(*response, "rlimit_used");
    ASSERT_TRUE(succeeded(budget));
    ASSERT_TRUE(succeeded(used));
    llvm::errs() << "[rlimit-fixture] " << name << " used=" << *used
                 << " budget=" << *budget << "\n";
    EXPECT_LT(*used * kRequiredHeadroom, *budget);
  }
}

TEST(Z3JointSolverBudgetTest, RepeatedSolvesAreByteIdentical) {
  for (const auto &[name, problemJson] : fixtureCorpus()) {
    SCOPED_TRACE(name.str());
    auto first = runZ3JointSolver(problemJson);
    ASSERT_TRUE(succeeded(first));
    for (int attempt = 1; attempt < 5; ++attempt) {
      auto repeat = runZ3JointSolver(problemJson);
      ASSERT_TRUE(succeeded(repeat));
      EXPECT_EQ(*repeat, *first) << "attempt " << attempt;
    }
  }
}

// llvm::json::Object is hash-ordered, so round-tripping the request re-emits
// its keys in a different order than the source literal.
TEST(Z3JointSolverBudgetTest, RequestKeyOrderDoesNotChangeTheAnswer) {
  for (const auto &[name, problemJson] : fixtureCorpus()) {
    SCOPED_TRACE(name.str());
    auto direct = runZ3JointSolver(problemJson);
    ASSERT_TRUE(succeeded(direct));
    auto reordered = parseJsonObject(problemJson);
    ASSERT_TRUE(succeeded(reordered));
    auto shuffled =
        runZ3JointSolver(serializeJsonObject(std::move(*reordered)));
    ASSERT_TRUE(succeeded(shuffled));
    EXPECT_EQ(*shuffled, *direct);
  }
}

// A budget stop must never be presented as a proof of infeasibility, and the
// deterministic budget must never be mistaken for the wall-clock backstop:
// the first is a reproducible property of the problem, the second is an
// environment signal.
TEST(Z3JointSolverBudgetTest, ExhaustedRLimitIsAttributedToRLimit) {
  for (llvm::StringRef problemJson :
       {llvm::StringRef(kFeasibleJointSolverProblem),
        llvm::StringRef(kJointSolverV2Problem)}) {
    SCOPED_TRACE(problemJson.substr(0, 64).str());
    auto response = runWithRLimit(problemJson, /*rlimit=*/1);
    ASSERT_TRUE(succeeded(response));
    EXPECT_EQ(response->getString("status"), "inconclusive");
    EXPECT_EQ(response->getString("budget_exhausted"), "rlimit");
    EXPECT_EQ(response->getBoolean("proven_unsat"), false);
    EXPECT_EQ(response->getString("backend_status"), "UNKNOWN");
  }
}

// A generous budget must not trip the guard — otherwise the test above would
// pass even if every solve were reported as exhausted.
TEST(Z3JointSolverBudgetTest, SufficientRLimitStillSolves) {
  for (llvm::StringRef problemJson :
       {llvm::StringRef(kFeasibleJointSolverProblem),
        llvm::StringRef(kJointSolverV2Problem)}) {
    SCOPED_TRACE(problemJson.substr(0, 64).str());
    auto response = runWithRLimit(problemJson, /*rlimit=*/10000000);
    ASSERT_TRUE(succeeded(response));
    EXPECT_EQ(response->getString("status"), "ok");
    EXPECT_FALSE(response->getString("budget_exhausted"));
  }
}

TEST(Z3JointSolverBudgetTest, RejectsMalformedBudgetFields) {
  for (llvm::StringRef problemJson :
       {llvm::StringRef(kFeasibleJointSolverProblem),
        llvm::StringRef(kJointSolverV2Problem)}) {
    SCOPED_TRACE(problemJson.substr(0, 64).str());
    for (llvm::StringRef key : {"rlimit", "random_seed"}) {
      SCOPED_TRACE(key.str());
      auto negative = parseJsonObject(problemJson);
      ASSERT_TRUE(succeeded(negative));
      (*negative)[key] = -1;
      EXPECT_TRUE(
          failed(runZ3JointSolver(serializeJsonObject(std::move(*negative)))));

      auto wrongType = parseJsonObject(problemJson);
      ASSERT_TRUE(succeeded(wrongType));
      (*wrongType)[key] = "zero";
      EXPECT_TRUE(
          failed(runZ3JointSolver(serializeJsonObject(std::move(*wrongType)))));
    }
  }
}

// Real solver inputs, extracted from the committed DDGs of the sched2tlx
// example kernels (third_party/tlx/tools/sched2tlx/examples/*/ddg.json,
// loops[].ddg). The hand-written fixtures above are one to two orders of
// magnitude smaller than anything the compiler actually solves, so the default
// budget has to be calibrated against these instead. Buffers are omitted: the
// committed DDG does not carry them, so these under-count real cost slightly.

/// case1_simple_gemm inner loop: 5 nodes (2 TMA, 1 TC, 2 NONE).
constexpr llvm::StringLiteral kRealGemmInnerProblem = R"json(
{
  "version": "joint-solver-0.1",
  "min_ii": 256,
  "max_ii": 256,
  "smem_budget": 232448,
  "tmem_col_limit": 512,
  "time_limit_s": 120,
  "streaming_vl": false,
  "nodes": [
    {"id": 0, "pipeline": "TMA", "duration": 30, "streaming": false},
    {"id": 1, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 2, "pipeline": "TMA", "duration": 30, "streaming": false},
    {"id": 3, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 4, "pipeline": "TC", "duration": 30, "streaming": false}
  ],
  "edges": [
    {"src": 0, "dst": 1, "latency": 556, "distance": 0},
    {"src": 2, "dst": 3, "latency": 556, "distance": 0},
    {"src": 1, "dst": 4, "latency": 0, "distance": 0},
    {"src": 3, "dst": 4, "latency": 0, "distance": 0},
    {"src": 4, "dst": 4, "latency": 256, "distance": 1}
  ],
  "buffers": []
}
)json";

/// case2_persistent_gemm outer loop: 16 nodes, 13 of them CUDA.
constexpr llvm::StringLiteral kRealPersistentGemmOuterProblem = R"json(
{
  "version": "joint-solver-0.1",
  "min_ii": 17888,
  "max_ii": 17888,
  "smem_budget": 232448,
  "tmem_col_limit": 512,
  "time_limit_s": 120,
  "streaming_vl": false,
  "nodes": [
    {"id": 0, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 1, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 2, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 3, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 4, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 5, "pipeline": "CUDA", "duration": 192, "streaming": false},
    {"id": 6, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 7, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 8, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 9, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 10, "pipeline": "CUDA", "duration": 1024, "streaming": false},
    {"id": 11, "pipeline": "CUDA", "duration": 256, "streaming": false},
    {"id": 12, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 13, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 14, "pipeline": "CUDA", "duration": 420, "streaming": false},
    {"id": 15, "pipeline": "TMA", "duration": 30, "streaming": false}
  ],
  "edges": [
    {"src": 0, "dst": 2, "latency": 1, "distance": 0},
    {"src": 1, "dst": 3, "latency": 1, "distance": 0},
    {"src": 4, "dst": 5, "latency": 0, "distance": 0},
    {"src": 7, "dst": 8, "latency": 1, "distance": 0},
    {"src": 7, "dst": 9, "latency": 1, "distance": 0},
    {"src": 4, "dst": 10, "latency": 0, "distance": 0},
    {"src": 10, "dst": 11, "latency": 2128, "distance": 0},
    {"src": 8, "dst": 12, "latency": 1, "distance": 0},
    {"src": 9, "dst": 13, "latency": 1, "distance": 0},
    {"src": 11, "dst": 14, "latency": 420, "distance": 0},
    {"src": 14, "dst": 15, "latency": 420, "distance": 0},
    {"src": 12, "dst": 15, "latency": 1, "distance": 0},
    {"src": 13, "dst": 15, "latency": 1, "distance": 0},
    {"src": 6, "dst": 10, "latency": 17888, "distance": 0},
    {"src": 7, "dst": 7, "latency": 1, "distance": 1}
  ],
  "buffers": []
}
)json";

/// case3_FA inner loop: 31 nodes / 40 edges - the widest real loop in
/// the corpus (17 CUDA, 2 TMA, 2 TC, 2 SFU).
constexpr llvm::StringLiteral kRealFlashAttentionInnerProblem = R"json(
{
  "version": "joint-solver-0.1",
  "min_ii": 1325,
  "max_ii": 2188,
  "smem_budget": 232448,
  "tmem_col_limit": 512,
  "time_limit_s": 120,
  "streaming_vl": false,
  "nodes": [
    {"id": 0, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 1, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 2, "pipeline": "TMA", "duration": 30, "streaming": false},
    {"id": 3, "pipeline": "TMA", "duration": 30, "streaming": false},
    {"id": 4, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 5, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 6, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 7, "pipeline": "TC", "duration": 30, "streaming": false},
    {"id": 8, "pipeline": "CUDA", "duration": 128, "streaming": false},
    {"id": 9, "pipeline": "CUDA", "duration": 152, "streaming": false},
    {"id": 10, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 11, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 12, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 13, "pipeline": "SFU", "duration": 1, "streaming": false},
    {"id": 14, "pipeline": "CUDA", "duration": 64, "streaming": false},
    {"id": 15, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 16, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 17, "pipeline": "CUDA", "duration": 64, "streaming": false},
    {"id": 18, "pipeline": "SFU", "duration": 64, "streaming": false},
    {"id": 19, "pipeline": "CUDA", "duration": 319, "streaming": false},
    {"id": 20, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 21, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 22, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 23, "pipeline": "CUDA", "duration": 256, "streaming": false},
    {"id": 24, "pipeline": "CUDA", "duration": 128, "streaming": false},
    {"id": 25, "pipeline": "CUDA", "duration": 32, "streaming": false},
    {"id": 26, "pipeline": "NONE", "duration": 1, "streaming": false},
    {"id": 27, "pipeline": "CUDA", "duration": 48, "streaming": false},
    {"id": 28, "pipeline": "TC", "duration": 30, "streaming": false},
    {"id": 29, "pipeline": "CUDA", "duration": 1, "streaming": false},
    {"id": 30, "pipeline": "CUDA", "duration": 1, "streaming": false}
  ],
  "edges": [
    {"src": 0, "dst": 1, "latency": 1, "distance": 0},
    {"src": 1, "dst": 2, "latency": 1, "distance": 0},
    {"src": 1, "dst": 3, "latency": 1, "distance": 0},
    {"src": 3, "dst": 4, "latency": 556, "distance": 0},
    {"src": 2, "dst": 5, "latency": 556, "distance": 0},
    {"src": 5, "dst": 6, "latency": 0, "distance": 0},
    {"src": 6, "dst": 7, "latency": 0, "distance": 0},
    {"src": 7, "dst": 8, "latency": 900, "distance": 0},
    {"src": 8, "dst": 9, "latency": 266, "distance": 0},
    {"src": 9, "dst": 10, "latency": 230, "distance": 0},
    {"src": 10, "dst": 11, "latency": 1, "distance": 0},
    {"src": 11, "dst": 12, "latency": 1, "distance": 0},
    {"src": 12, "dst": 13, "latency": 1, "distance": 0},
    {"src": 8, "dst": 14, "latency": 266, "distance": 0},
    {"src": 11, "dst": 15, "latency": 1, "distance": 0},
    {"src": 15, "dst": 16, "latency": 0, "distance": 0},
    {"src": 14, "dst": 17, "latency": 69, "distance": 0},
    {"src": 16, "dst": 17, "latency": 0, "distance": 0},
    {"src": 17, "dst": 18, "latency": 65, "distance": 0},
    {"src": 18, "dst": 19, "latency": 570, "distance": 0},
    {"src": 13, "dst": 20, "latency": 8, "distance": 0},
    {"src": 20, "dst": 21, "latency": 0, "distance": 0},
    {"src": 21, "dst": 22, "latency": 0, "distance": 0},
    {"src": 23, "dst": 24, "latency": 532, "distance": 0},
    {"src": 22, "dst": 24, "latency": 0, "distance": 0},
    {"src": 18, "dst": 25, "latency": 570, "distance": 0},
    {"src": 25, "dst": 26, "latency": 52, "distance": 0},
    {"src": 23, "dst": 27, "latency": 532, "distance": 0},
    {"src": 24, "dst": 27, "latency": 138, "distance": 0},
    {"src": 26, "dst": 28, "latency": 0, "distance": 0},
    {"src": 4, "dst": 28, "latency": 0, "distance": 0},
    {"src": 27, "dst": 28, "latency": 96, "distance": 0},
    {"src": 13, "dst": 29, "latency": 8, "distance": 0},
    {"src": 29, "dst": 30, "latency": 1, "distance": 0},
    {"src": 19, "dst": 30, "latency": 845, "distance": 0},
    {"src": 11, "dst": 11, "latency": 1, "distance": 1},
    {"src": 11, "dst": 12, "latency": 1, "distance": 1},
    {"src": 30, "dst": 29, "latency": 1, "distance": 1},
    {"src": 8, "dst": 7, "latency": 266, "distance": 1},
    {"src": 28, "dst": 23, "latency": 559, "distance": 1}
  ],
  "buffers": []
}
)json";

// Not part of the normal suite: this is the procedure for re-deriving the
// default rlimit, which must be repeated whenever the vendored Z3 changes,
// because Z3's work-unit accounting is not stable across releases. Run with
//   --gtest_also_run_disabled_tests \
//   --gtest_filter=*DISABLED_ReportRLimitCalibration
TEST(Z3JointSolverBudgetTest, DISABLED_ReportRLimitCalibration) {
  const std::pair<llvm::StringRef, llvm::StringRef> corpus[] = {
      {"real/gemm_inner", kRealGemmInnerProblem},
      {"real/persistent_gemm_outer", kRealPersistentGemmOuterProblem},
      {"real/fa_inner", kRealFlashAttentionInnerProblem},
  };
  for (const auto &[name, problemJson] : corpus) {
    auto response = runZ3JointSolver(problemJson);
    if (failed(response)) {
      llvm::errs() << "[rlimit-calibration] " << name << " FAILED\n";
      continue;
    }
    auto parsed = parseJsonObject(*response);
    ASSERT_TRUE(succeeded(parsed));
    auto used = statsInteger(*parsed, "rlimit_used");
    auto status = parsed->getString("status");
    llvm::errs() << "[rlimit-calibration] " << name
                 << " status=" << (status ? *status : "<none>")
                 << " used=" << (succeeded(used) ? *used : -1) << "\n";
  }
}

// The one real loop the solver currently gets all the way through. Pinning it
// means a change that pushes it over the budget surfaces as a failure here
// rather than as a silent fallback to the heuristic scheduler in production.
TEST(Z3JointSolverBudgetTest, RealGemmInnerLoopSolvesInsideTheDefaultBudget) {
  auto problem = parseJsonObject(kRealGemmInnerProblem);
  ASSERT_TRUE(succeeded(problem));
  auto response = runJsonProblem(std::move(*problem));
  ASSERT_TRUE(succeeded(response));
  EXPECT_EQ(response->getString("status"), "ok");
  auto budget = statsInteger(*response, "rlimit");
  auto used = statsInteger(*response, "rlimit_used");
  ASSERT_TRUE(succeeded(budget));
  ASSERT_TRUE(succeeded(used));
  EXPECT_LT(*used * 8, *budget);
}

// A real loop the solver cannot finish. What matters is not that it gives up
// but that it gives up at exactly the same point every run, and that the stop
// is charged to the deterministic budget rather than to the wall clock. The
// budget is reduced so the test stays fast; the mechanism is the same one that
// runs at the default.
TEST(Z3JointSolverBudgetTest, UnsolvableRealLoopStopsAtTheSamePointEveryRun) {
  constexpr int64_t kReducedRLimit = 200000;
  auto first = runWithRLimit(kRealFlashAttentionInnerProblem, kReducedRLimit);
  ASSERT_TRUE(succeeded(first));
  EXPECT_EQ(first->getString("status"), "inconclusive");
  EXPECT_EQ(first->getString("budget_exhausted"), "rlimit");
  EXPECT_EQ(first->getBoolean("proven_unsat"), false);
  auto firstUsed = statsInteger(*first, "rlimit_used");
  ASSERT_TRUE(succeeded(firstUsed));
  for (int attempt = 1; attempt < 3; ++attempt) {
    auto repeat =
        runWithRLimit(kRealFlashAttentionInnerProblem, kReducedRLimit);
    ASSERT_TRUE(succeeded(repeat));
    auto used = statsInteger(*repeat, "rlimit_used");
    ASSERT_TRUE(succeeded(used));
    EXPECT_EQ(*used, *firstUsed) << "attempt " << attempt;
  }
}

constexpr llvm::StringLiteral kChildResponseMarker = "@@solver-response@@";

// Exists only to be re-executed by SolvesReproduceAcrossProcesses in a child
// process; it is not a test on its own.
TEST(Z3JointSolverBudgetTest, DISABLED_EmitFixtureResponse) {
  auto response = runZ3JointSolver(kJointSolverV2Problem);
  ASSERT_TRUE(succeeded(response));
  llvm::outs() << kChildResponseMarker << *response << "\n";
  llvm::outs().flush();
}

static FailureOr<std::string> solveInChildProcess(std::string &rawOutput) {
  // popen runs the command under /bin/sh, so the test binary has to be named
  // by absolute path: a literal "/proc/self/exe" in the command line would
  // resolve to the shell rather than to this process.
  char executable[PATH_MAX];
  ssize_t length =
      readlink("/proc/self/exe", executable, sizeof(executable) - 1);
  if (length <= 0)
    return failure();
  executable[length] = '\0';

  std::string command = std::string("'") + executable +
                        "' --gtest_also_run_disabled_tests --gtest_color=no"
                        " --gtest_filter=Z3JointSolverBudgetTest."
                        "DISABLED_EmitFixtureResponse 2>&1";
  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe)
    return failure();
  char buffer[4096];
  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
    rawOutput += buffer;
  if (pclose(pipe) != 0)
    return failure();
  size_t start = rawOutput.find(kChildResponseMarker);
  if (start == std::string::npos)
    return failure();
  start += kChildResponseMarker.size();
  size_t end = rawOutput.find('\n', start);
  if (end == std::string::npos)
    return failure();
  return rawOutput.substr(start, end - start);
}

// The in-process repeat above cannot observe nondeterminism that comes from
// process-level state, which is what "reproducible solver runs" is actually
// about — so solve the same fixture again in a fresh process and diff.
TEST(Z3JointSolverBudgetTest, SolvesReproduceAcrossProcesses) {
  auto inProcess = runZ3JointSolver(kJointSolverV2Problem);
  ASSERT_TRUE(succeeded(inProcess));
  std::string rawOutput;
  auto childProcess = solveInChildProcess(rawOutput);
  ASSERT_TRUE(succeeded(childProcess)) << "child output:\n" << rawOutput;
  EXPECT_EQ(*childProcess, *inProcess);
}

#endif

} // namespace
} // namespace mlir::triton::gpu
