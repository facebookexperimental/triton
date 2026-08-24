// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Fallback-policy tests.
//
// This file covers the two MLIR-free halves of the policy: the trigger
// recorder itself, and the deterministic fake backend that drives each
// terminal-policy trigger through the real orchestration in
// `runJointSolverBackend`. The decision layer on top of them —
// baseline-rerun vs strict-error, and the byte-identical comparison against a
// flag-off compile — needs a ModuleOp and lives in
// test/TritonGPU/modulo-schedule-joint-fallback.mlir.

#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/JointSolverFallback.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/JointSolverScheduler.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/ModuloReservationTable.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <gtest/gtest.h>

#include <optional>
#include <string>

namespace mlir::triton::gpu {
namespace {

// ── Trigger recorder ────────────────────────────────────────────────────────

TEST(JointSolverFallbackTest, ReportWithoutALiveScopeIsANoOp) {
  // Trigger sites must be callable from paths the driver does not own (unit
  // tests, the AMD scaffold, ModuloWSPartitionPass) without a guard.
  reportJointSolverFailure(JointSolverTrigger::ScheduleSolve, "orphan");

  JointSolverFallbackScope scope;
  EXPECT_FALSE(scope.getTrigger().has_value());
  EXPECT_EQ(scope.getCount(), 0u);
}

TEST(JointSolverFallbackTest, KeepsTheFirstTriggerAndCountsTheRest) {
  // The first trigger is the one that actually caused the fallback; later
  // ones are downstream noise from an attempt already being discarded.
  JointSolverFallbackScope scope;
  reportJointSolverFailure(JointSolverTrigger::ScheduleSolve, "first");
  reportJointSolverFailure(JointSolverTrigger::PartitionSolve, "second");
  reportJointSolverFailure(JointSolverTrigger::MemoryAudit, "third");

  ASSERT_TRUE(scope.getTrigger().has_value());
  EXPECT_EQ(*scope.getTrigger(), JointSolverTrigger::ScheduleSolve);
  EXPECT_EQ(scope.getDetail(), "first");
  EXPECT_EQ(scope.getCount(), 3u);
}

TEST(JointSolverFallbackTest, NestedScopesDoNotLeakIntoEachOther) {
  JointSolverFallbackScope outer;
  {
    JointSolverFallbackScope inner;
    reportJointSolverFailure(JointSolverTrigger::PartitionSolve, "inner");
    ASSERT_TRUE(inner.getTrigger().has_value());
    EXPECT_EQ(*inner.getTrigger(), JointSolverTrigger::PartitionSolve);
  }
  EXPECT_FALSE(outer.getTrigger().has_value());

  reportJointSolverFailure(JointSolverTrigger::MemoryAudit, "outer");
  ASSERT_TRUE(outer.getTrigger().has_value());
  EXPECT_EQ(*outer.getTrigger(), JointSolverTrigger::MemoryAudit);
}

TEST(JointSolverFallbackTest, TriggerNamesAreStable) {
  // These strings reach users through the fallback remark and the
  // strict-error message, and lit tests match them.
  EXPECT_EQ(getJointSolverTriggerName(JointSolverTrigger::ScheduleSolve),
            "schedule-solve");
  EXPECT_EQ(getJointSolverTriggerName(JointSolverTrigger::PartitionSolve),
            "partition-solve");
  EXPECT_EQ(getJointSolverTriggerName(JointSolverTrigger::MemoryAudit),
            "memory-audit");
  EXPECT_EQ(getJointSolverTriggerName(JointSolverTrigger::AttemptFailed),
            "attempt-failed");
}

// ── Schedule-algorithm resolution ───────────────────────────────────────────

TEST(JointSolverFallbackTest, ForcedScheduleAlgoWinsAndDoesNotPersist) {
  // The fallback reruns the attempt under a different backend, so resolution
  // has to be a pure function of its argument — if a forced choice leaked
  // into process state, the baseline rerun could not name its own backend
  // (and a concurrently compiling module would see the wrong one).
  const std::string ambient = getActiveScheduleAlgo();
  EXPECT_EQ(getActiveScheduleAlgo("joint_solver"), "joint_solver");
  EXPECT_EQ(getActiveScheduleAlgo("rau"), "rau");
  EXPECT_EQ(getActiveScheduleAlgo(), ambient);
}

// ── Deterministic fake backend ──────────────────────────────────────────────

TEST(JointSolverFallbackTest, FaultNamesRoundTrip) {
  for (JointSolverFault fault :
       {JointSolverFault::Unavailable, JointSolverFault::Timeout,
        JointSolverFault::Unknown, JointSolverFault::GlobalUnsat,
        JointSolverFault::Malformed, JointSolverFault::IllegalSchedule}) {
    auto parsed = parseJointSolverFault(getJointSolverFaultName(fault));
    ASSERT_TRUE(parsed.has_value()) << getJointSolverFaultName(fault).str();
    EXPECT_EQ(*parsed, fault);
  }
  EXPECT_FALSE(parseJointSolverFault("not-a-fault").has_value());
  EXPECT_FALSE(parseJointSolverFault("").has_value());
}

/// A joint-solver-0.1 schedule problem: two tensor-core ops of duration 2
/// chained by a latency-1 dependence. Four TC issue slots are needed, so any
/// II below 4 is provably infeasible and the solver's own sweep has to step
/// past at least one such candidate before settling.
static std::string makeScheduleProblem(int64_t minII, int64_t maxII) {
  auto node = [](int64_t id) {
    return llvm::json::Object{
        {"id", id}, {"pipeline", "TC"}, {"duration", 2}, {"streaming", false}};
  };
  llvm::json::Object root{
      {"version", "joint-solver-0.1"},
      {"min_ii", minII},
      {"max_ii", maxII},
      {"smem_budget", 232448},
      {"tmem_col_limit", 512},
      {"time_limit_s", 20.0},
      {"streaming_vl", false},
      {"nodes", llvm::json::Array{node(0), node(1)}},
      {"edges", llvm::json::Array{llvm::json::Object{
                    {"src", 0}, {"dst", 1}, {"latency", 1}, {"distance", 0}}}},
      {"buffers", llvm::json::Array{}},
  };
  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::json::Value(std::move(root));
  return out;
}

static std::optional<llvm::json::Object> parseObject(llvm::StringRef json) {
  auto parsed = llvm::json::parse(json);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return std::nullopt;
  }
  auto *object = parsed->getAsObject();
  if (!object)
    return std::nullopt;
  return std::move(*object);
}

TEST(JointSolverFallbackTest, UnavailableBackendFailsTheTransport) {
  ScopedJointSolverBackendOverride fault(JointSolverFault::Unavailable);
  EXPECT_TRUE(failed(runJointSolverBackend(makeScheduleProblem(1, 8))));
}

TEST(JointSolverFallbackTest, TimeoutAndUnknownStayInconclusive) {
  // Both are retried and then surrendered as inconclusive — never mistaken
  // for a proof of infeasibility, which would let the II search skip ahead.
  for (auto [fault, reason] :
       {std::pair{JointSolverFault::Timeout, "timeout"},
        std::pair{JointSolverFault::Unknown, "unknown"}}) {
    SCOPED_TRACE(reason);
    ScopedJointSolverBackendOverride injected(fault);
    auto response = runJointSolverBackend(makeScheduleProblem(1, 8));
    ASSERT_TRUE(succeeded(response));
    auto object = parseObject(*response);
    ASSERT_TRUE(object.has_value());
    EXPECT_EQ(object->getString("status"), "inconclusive");
    EXPECT_EQ(object->getString("reason"), reason);
    EXPECT_FALSE(object->getBoolean("proven_unsat").value_or(false));
  }
}

TEST(JointSolverFallbackTest, GlobalUnsatIsReportedAsProven) {
  ScopedJointSolverBackendOverride fault(JointSolverFault::GlobalUnsat);
  auto response = runJointSolverBackend(makeScheduleProblem(1, 8));
  ASSERT_TRUE(succeeded(response));
  auto object = parseObject(*response);
  ASSERT_TRUE(object.has_value());
  EXPECT_EQ(object->getString("status"), "infeasible");
  EXPECT_TRUE(object->getBoolean("proven_unsat").value_or(false));
}

TEST(JointSolverFallbackTest, MalformedResponseIsNotParseable) {
  ScopedJointSolverBackendOverride fault(JointSolverFault::Malformed);
  auto response = runJointSolverBackend(makeScheduleProblem(1, 8));
  // The orchestration hands the text back verbatim; every caller then fails
  // to parse it, which is the "malformed / partial model" trigger.
  ASSERT_TRUE(succeeded(response));
  EXPECT_FALSE(parseObject(*response).has_value());
}

TEST(JointSolverFallbackTest, OverrideNestsAndRestores) {
  ScopedJointSolverBackendOverride outer(JointSolverFault::GlobalUnsat);
  {
    ScopedJointSolverBackendOverride inner(JointSolverFault::Unavailable);
    EXPECT_TRUE(failed(runJointSolverBackend(makeScheduleProblem(1, 8))));
  }
  auto response = runJointSolverBackend(makeScheduleProblem(1, 8));
  ASSERT_TRUE(succeeded(response));
  auto object = parseObject(*response);
  ASSERT_TRUE(object.has_value());
  EXPECT_EQ(object->getString("status"), "infeasible");
}

TEST(JointSolverFallbackTest, FaultedResponsesBypassTheResultCache) {
  // Proven results are cached by canonical problem. A canned response must
  // neither be served from that cache nor written into it, or one test's
  // fault would decide the next test's compile.
  const std::string problem = makeScheduleProblem(1, 8);
  auto statusUnder = [&](JointSolverFault fault) -> std::string {
    ScopedJointSolverBackendOverride injected(fault);
    auto response = runJointSolverBackend(problem);
    if (failed(response))
      return "<transport failure>";
    auto object = parseObject(*response);
    if (!object)
      return "<unparseable>";
    return object->getString("status").value_or("<none>").str();
  };

  EXPECT_EQ(statusUnder(JointSolverFault::GlobalUnsat), "infeasible");
  // Served from the cache, this would still read "infeasible".
  EXPECT_EQ(statusUnder(JointSolverFault::Timeout), "inconclusive");

#ifdef TRITON_ENABLE_Z3_JOINT_SOLVER
  // And the write side: with no fault live, the same problem must reach the
  // real backend. A cached "infeasible" from the fault above would surface
  // here and turn a feasible problem into a permanent failure for the rest of
  // the process.
  auto real = runJointSolverBackend(problem);
  ASSERT_TRUE(succeeded(real));
  auto object = parseObject(*real);
  ASSERT_TRUE(object.has_value());
  EXPECT_EQ(object->getString("status"), "ok");
#endif
}

#ifdef TRITON_ENABLE_Z3_JOINT_SOLVER

TEST(JointSolverFallbackTest, ProvenUnsatAtOneIIIsNotTerminal) {
  // The single most important non-trigger: the sweep proves at least one
  // sub-ResMII candidate infeasible on its way to a feasible II. (Which
  // candidates it visits is the search's business — it does not try every
  // integer — so this asserts the shape, not a specific set.) That is the II
  // search working, not a terminal outcome, and it must never reach the
  // fallback policy.
  auto response = runJointSolverBackend(makeScheduleProblem(1, 8));
  ASSERT_TRUE(succeeded(response));
  auto object = parseObject(*response);
  ASSERT_TRUE(object.has_value());
  ASSERT_EQ(object->getString("status"), "ok");

  auto ii = object->getInteger("ii");
  ASSERT_TRUE(ii.has_value());
  EXPECT_GT(*ii, 1);

  const auto *stats = object->getObject("stats");
  ASSERT_NE(stats, nullptr);
  const auto *unsatIIs = stats->getArray("unsat_iis");
  ASSERT_NE(unsatIIs, nullptr);
  EXPECT_FALSE(unsatIIs->empty())
      << "expected the sweep to prove the sub-ResMII candidates infeasible";
  for (const llvm::json::Value &value : *unsatIIs) {
    auto candidate = value.getAsInteger();
    ASSERT_TRUE(candidate.has_value());
    EXPECT_LT(*candidate, *ii);
  }
}

TEST(JointSolverFallbackTest, IllegalScheduleIsRejectedByReVerification) {
  // A well-formed response the backend could plausibly have produced, whose
  // schedule violates the dependence it was given. The solver is advisory,
  // so this must be discarded rather than trusted.
  const std::string problem = makeScheduleProblem(1, 8);
  auto forged =
      makeJointSolverFaultResponse(JointSolverFault::IllegalSchedule, problem);
  ASSERT_TRUE(succeeded(forged));
  auto object = parseObject(*forged);
  ASSERT_TRUE(object.has_value());
  ASSERT_EQ(object->getString("status"), "ok")
      << "the fault must stay schema-valid so it reaches the semantic checks";
  const auto *cycles = object->getObject("cycles");
  ASSERT_NE(cycles, nullptr);
  for (const auto &entry : *cycles)
    EXPECT_EQ(entry.second.getAsInteger().value_or(-1), 0);

  ScopedJointSolverBackendOverride fault(JointSolverFault::IllegalSchedule);
  EXPECT_TRUE(failed(runJointSolverBackend(problem)));
}

#else

TEST(JointSolverFallbackTest, MissingBackendFailsWithoutAFaultInjector) {
  // Test 1 for free: a build without Z3 is the real "backend unavailable"
  // case, and it has to reach the same terminal outcome as the injected one.
  EXPECT_TRUE(failed(runJointSolverBackend(makeScheduleProblem(1, 8))));
}

#endif // TRITON_ENABLE_Z3_JOINT_SOLVER

} // namespace
} // namespace mlir::triton::gpu
