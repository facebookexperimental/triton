#include "triton/Tools/LayoutUtils.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir::triton {
namespace {

class LayoutUtilsTest : public ::testing::Test {
public:
  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LayoutUtilsTest, SquareSublayoutIsIdentity) {
  EXPECT_TRUE(squareSublayoutIsIdentity(
      LinearLayout::identity1D(4, S("in"), S("in")), {S("in")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(
      LinearLayout::identity1D(4, S("in"), S("in")), {}));

  LinearLayout l1(
      {{S("in1"), {{1, 1}, {2, 2}, {4, 4}}}, {S("in2"), {{2, 1}, {1, 2}}}},
      {{S("in1"), 8}, {S("in2"), 8}}, /*requireSurjective=*/false);
  EXPECT_TRUE(squareSublayoutIsIdentity(l1, {S("in1")}));
  EXPECT_FALSE(squareSublayoutIsIdentity(l1, {S("in2")}));

  LinearLayout l2 = LinearLayout::identity1D(4, S("in1"), S("in1")) *
                    LinearLayout::identity1D(8, S("in2"), S("in2")) *
                    LinearLayout({{S("in3"), {{1, 1, 1}}}},
                                 {{S("in1"), 2}, {S("in2"), 2}, {S("in3"), 2}},
                                 /*requireSurjective=*/false);
  EXPECT_FALSE(squareSublayoutIsIdentity(l2, {S("in1")}));
  EXPECT_FALSE(squareSublayoutIsIdentity(l2, {S("in2")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(l2, {S("in3")}));
  EXPECT_FALSE(squareSublayoutIsIdentity(l2, {S("in1"), S("in2")}));

  LinearLayout l3 = LinearLayout::identity1D(4, S("in1"), S("in1")) *
                    LinearLayout::identity1D(8, S("in2"), S("in2"));
  EXPECT_TRUE(squareSublayoutIsIdentity(l3, {S("in1")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(l3, {S("in2")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(l3, {S("in1"), S("in2")}));
}

// computeDeadPositionMap must repair "phantom" register positions whose
// register-only NPOT coordinate wraps onto a register-reachable canonical at
// the SAME lane/warp -- even when lane/warp ALSO contribute to the NPOT dim.
// This is the NPOT-M blocked->blocked FMA store case (M=96): a register-rep
// count of 3 is padded to 4, and the phantom 4th rep (reg coord 96) wraps
// (mod 96) onto reg coord 0, the canonical owned by the same thread.
TEST_F(LayoutUtilsTest, DeadPositionMapRegisterPhantomWithLaneWarpOnNpotDim) {
  auto kReg = S("register"), kLane = S("lane"), kWarp = S("warp");
  // dim0 is modular (size 96). Register bases include the rep strides 32 and
  // 64 (so reg coords 0,32,64,96 are reachable; 96 mod 96 = 0). Lane/warp also
  // contribute to dim0 (strides 4 and 8/16), exercising the relaxed path.
  LinearLayout layout({{kReg, {{0}, {0}, {1}, {2}, {32}, {64}}},
                       {kLane, {{0}, {0}, {0}, {0}, {4}}},
                       {kWarp, {{8}, {16}}}},
                      {{S("dim0"), 96}}, /*requireSurjective=*/false);
  ASSERT_TRUE(layout.isModular());

  auto deadMap = computeDeadPositionMap(layout, kReg, kLane, kWarp);
  EXPECT_FALSE(deadMap.empty());

  int numRegBits = layout.getInDimSizeLog2(kReg);
  auto regCoord = [&](int r) {
    int c = 0;
    for (int b = 0; b < numRegBits; ++b) {
      if (r & (1 << b)) {
        c += layout.getBasis(kReg, b)[0];
      }
    }
    return c;
  };
  for (auto &[dead, canon] : deadMap) {
    EXPECT_LT(canon, layout.getInDimSize(kReg));
    // The dead register's reg-only coord must be >= N (it is a phantom)...
    EXPECT_GE(regCoord(dead), 96);
    // ...and the canonical's reg-only coord must equal it mod N (sound copy).
    EXPECT_EQ(regCoord(canon), regCoord(dead) % 96);
  }
}

// When the wrapped (canonical) coordinate is NOT register-reachable at the same
// lane/warp (e.g. the MMAv2 N case where dead positions are driven by
// lane/warp), no fixup entry must be produced -- copying would move data across
// threads and corrupt results.
TEST_F(LayoutUtilsTest, DeadPositionMapNoRegisterCanonicalEmits) {
  auto kReg = S("register"), kLane = S("lane"), kWarp = S("warp");
  // dim0 modular (size 48). The only register stride into dim0 is 32 (so reg
  // coords 0 and 32 only); the wrap of any phantom would need coord 16, which
  // is reachable ONLY via lane (stride 16), not registers. No register-only
  // canonical exists for such phantoms, so they must be left unmapped.
  LinearLayout layout({{kReg, {{0}, {32}}},
                       {kLane, {{1}, {2}, {4}, {8}, {16}}},
                       {kWarp, {{0}}}},
                      {{S("dim0"), 48}}, /*requireSurjective=*/false);
  ASSERT_TRUE(layout.isModular());
  auto deadMap = computeDeadPositionMap(layout, kReg, kLane, kWarp);
  // reg coord 32 < 48, so there is no register-only phantom at all here; map
  // must be empty (nothing to repair within the thread).
  EXPECT_TRUE(deadMap.empty());
}

} // namespace
} // namespace mlir::triton
