#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/ModuloReservationTable.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/Z3JointSolver.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <gtest/gtest.h>

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

#endif

} // namespace
} // namespace mlir::triton::gpu
