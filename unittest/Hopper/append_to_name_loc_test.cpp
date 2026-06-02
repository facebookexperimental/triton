#include "third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/Utility.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include <gtest/gtest.h>

namespace mlir {
namespace {

// Regression test for B-10-F1 / T273483074.
TEST(HopperWarpSpecializationUtilityTest,
     DISABLED_AppendToNameLocUsesInnermostName) {
  MLIRContext context;
  auto fileLoc = FileLineColLoc::get(&context, "kernel.mlir", 3, 5);
  auto innerLoc =
      NameLoc::get(StringAttr::get(&context, "inner"), fileLoc);
  auto outerLoc =
      NameLoc::get(StringAttr::get(&context, "outer"), innerLoc);

  Location result = appendToNameLoc(outerLoc, "_suffix", &context);

  auto resultOuterLoc = dyn_cast<NameLoc>(result);
  ASSERT_TRUE(resultOuterLoc);
  EXPECT_EQ(resultOuterLoc.getName().str(), "outer");

  auto resultInnerLoc = dyn_cast<NameLoc>(resultOuterLoc.getChildLoc());
  ASSERT_TRUE(resultInnerLoc);
  EXPECT_EQ(resultInnerLoc.getName().str(), "inner_suffix");
  EXPECT_EQ(resultInnerLoc.getChildLoc(), fileLoc);
}

} // namespace
} // namespace mlir
