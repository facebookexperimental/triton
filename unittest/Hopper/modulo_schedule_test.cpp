#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/ModuloReservationTable.h"

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

} // namespace
} // namespace mlir::triton::gpu
