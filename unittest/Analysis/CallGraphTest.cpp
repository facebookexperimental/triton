#include "triton/Analysis/Utility.h"

#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

#include <string>

namespace mlir {
namespace {

// root -> {left, right} -> leaf. An acyclic graph whose leaf is reachable
// through two distinct paths.
constexpr StringLiteral kDiamondCallGraph = R"MLIR(
  tt.func private @leaf() {
    tt.return
  }
  tt.func private @left() {
    tt.call @leaf() : () -> ()
    tt.return
  }
  tt.func private @right() {
    tt.call @leaf() : () -> ()
    tt.return
  }
  tt.func public @root() {
    tt.call @left() : () -> ()
    tt.call @right() : () -> ()
    tt.return
  }
)MLIR";

constexpr StringLiteral kSelfRecursiveCallGraph = R"MLIR(
  tt.func private @recurse() {
    tt.call @recurse() : () -> ()
    tt.return
  }
  tt.func public @root() {
    tt.call @recurse() : () -> ()
    tt.return
  }
)MLIR";

constexpr StringLiteral kMutuallyRecursiveCallGraph = R"MLIR(
  tt.func private @ping() {
    tt.call @pong() : () -> ()
    tt.return
  }
  tt.func private @pong() {
    tt.call @ping() : () -> ()
    tt.return
  }
  tt.func public @root() {
    tt.call @ping() : () -> ()
    tt.return
  }
)MLIR";

OwningOpRef<ModuleOp> parseModule(MLIRContext &context, StringRef source) {
  context.loadDialect<triton::TritonDialect>();
  return parseSourceString<ModuleOp>(source, &context);
}

// Records the order in which the walk visits nodes.
SmallVector<std::string> walkNodeOrder(ModuleOp moduleOp) {
  triton::CallGraph<int> callGraph(moduleOp);
  SmallVector<std::string> visited;
  callGraph.walk([](CallOpInterface, FunctionOpInterface) {},
                 [&](FunctionOpInterface funcOp) {
                   visited.push_back(funcOp.getName().str());
                 });
  return visited;
}

} // namespace

// A node reachable through several paths is not a cycle: the walk visits it
// once per path and must not report a cycle.
TEST(Analysis, CallGraphWalksDiamondWithoutReportingACycle) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseModule(context, kDiamondCallGraph);
  ASSERT_TRUE(module);

  EXPECT_EQ(
      walkNodeOrder(*module),
      SmallVector<std::string>({"root", "left", "leaf", "right", "leaf"}));
}

TEST(AnalysisDeathTest, CallGraphWalkRejectsSelfRecursion) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseModule(context, kSelfRecursiveCallGraph);
  ASSERT_TRUE(module);

  EXPECT_DEATH(walkNodeOrder(*module), "Cycle detected in call graph");
}

TEST(AnalysisDeathTest, CallGraphWalkRejectsMutualRecursion) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module =
      parseModule(context, kMutuallyRecursiveCallGraph);
  ASSERT_TRUE(module);

  EXPECT_DEATH(walkNodeOrder(*module), "Cycle detected in call graph");
}

} // namespace mlir
