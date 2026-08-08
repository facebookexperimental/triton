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

static FailureOr<llvm::json::Object>
makePipelineGroupingProblem(llvm::StringRef mode,
                            llvm::StringRef producerPipeline,
                            llvm::StringRef consumerPipeline, int64_t distance,
                            int64_t maxWGs) {
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
      {"events",
       llvm::json::Array{llvm::json::Object{
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
    EXPECT_EQ(event->getInteger("stream_order"),
              static_cast<int64_t>(index));
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

  std::string oneResultJson =
      serializeJsonObject(std::move(*oneResultProblem));
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
  std::string twoResultJson =
      serializeJsonObject(std::move(*twoResultProblem));
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
          auto problem = makePipelineGroupingProblem(
              mode, producer, consumer, distance, 2);
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
      SCOPED_TRACE(std::string(mode) +
                   "/distance=" + std::to_string(distance));
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
        auto problem = makePipelineGroupingProblem(
            mode, producer, consumer, distance, 1);
        ASSERT_TRUE(succeeded(problem));
        auto response = runJsonProblem(std::move(*problem));
        ASSERT_TRUE(succeeded(response));
        EXPECT_EQ(response->getString("status"), "ok");
        auto *warpGroups = response->getObject("wg");
        ASSERT_NE(warpGroups, nullptr);
        EXPECT_EQ(warpGroups->getInteger("10"),
                  warpGroups->getInteger("30"));
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
