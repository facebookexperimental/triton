#include "triton/Tools/LayoutUtils.h"
#include "unittest/Tools/ProductLayouts.h"
#include "unittest/Util/ScopedAllowNpot.h"

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

using Bases = std::vector<std::vector<int32_t>>;

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

TEST_F(LayoutUtilsTest, MixedRadixHelpersKeepRadices) {
  // register = 3 * d1 + d0 with a zero d0 basis.
  LinearLayout::BasesT bases;
  bases[S("register")] = {{0}, {4}};
  bases[S("lane")] = {{1}, {2}};
  LinearLayout::RadicesT radices;
  radices[S("register")] = {3, 2};
  radices[S("lane")] = {2, 2};
  LinearLayout ll(bases, radices, {{S("dim0"), 8}}, true);

  // d1 maps past the extent and is dropped with its radix.
  LinearLayout shrunk = ensureLayoutNotLargerThan(ll, {{S("dim0"), 4}},
                                                  /*broadcastRegisters=*/false);
  EXPECT_THAT(shrunk.getInputRadices(S("register")), ::testing::ElementsAre(3));
  EXPECT_EQ(shrunk.getOutDimSize(S("dim0")), 4);

  LinearLayout renamed = renameLinearLayoutDims(ll, {{S("lane"), S("l")}}, {});
  EXPECT_EQ(renamed.getInputRadices(S("register")),
            ll.getInputRadices(S("register")));
  EXPECT_THAT(renamed.getInputRadices(S("l")), ::testing::ElementsAre(2, 2));

  // The zero radix-3 digit is dropped; values are indexed 3 * d1 + d0.
  SmallVector<Value> values;
  for (intptr_t i = 0; i < 6; ++i)
    values.push_back(Value::getFromOpaquePointer(reinterpret_cast<void *>(i)));
  SmallVector<Value> unique = actionRemoveBroadcastedRegs(ll).apply(values);
  EXPECT_THAT(llvm::map_to_vector(unique,
                                  [](Value v) {
                                    return reinterpret_cast<intptr_t>(
                                        v.getAsOpaquePointer());
                                  }),
              ::testing::ElementsAre(0, 3));
}

TEST_F(LayoutUtilsTest, EnsureLayoutNotSmallerThanNpotCover) {
  auto layout = LinearLayout::identity1D(64, S("register"), S("dim0"));
#ifndef NDEBUG
  // Without the flag an NPOT extent still aborts.
  EXPECT_DEATH(ensureLayoutNotSmallerThan(layout, {{S("dim0"), 192}}),
               "isPowerOf2_32");
#endif
  ScopedAllowNpot npot;
  // 192 / 64 floors to 3; the layout grows to the cover 256 instead.
  EXPECT_EQ(ensureLayoutNotSmallerThan(layout, {{S("dim0"), 192}}),
            LinearLayout::identity1D(256, S("register"), S("dim0")));
}

TEST_F(LayoutUtilsTest, OutputBasisMask) {
  LinearLayout layout({{S("x"), {{1, 8}, {2, 16}}},
                       {S("y"), {{4, 1}}},
                       {S("broadcast"), {{0, 0}}}},
                      {{S("out0"), 8}, {S("out1"), 32}},
                      /*requireSurjective=*/false);

  EXPECT_EQ(getOutputBasisMask(layout, {S("x")}, S("out0")), 0b0011);
  EXPECT_EQ(getOutputBasisMask(layout, {S("x"), S("y")}, S("out0")), 0b0111);
  EXPECT_EQ(getOutputBasisMask(layout, {S("x"), S("broadcast")}, S("out1")),
            0b11000);
  EXPECT_EQ(getOutputBasisMask(layout, {}, S("out0")), 0);
}

TEST_F(LayoutUtilsTest, InputBasisMask) {
  LinearLayout layout(
      {{S("x"), {{1, 0}, {0, 2}, {0, 0}, {4, 8}}}, {S("y"), {{2, 0}}}},
      {{S("out0"), 8}, {S("out1"), 16}},
      /*requireSurjective=*/false);

  EXPECT_EQ(getInputBasisMask(layout, S("x"), {S("out0")}), 0b1001);
  EXPECT_EQ(getInputBasisMask(layout, S("x"), {S("out1")}), 0b1010);
  EXPECT_EQ(getInputBasisMask(layout, S("x"), {S("out0"), S("out1")}), 0b1011);
  EXPECT_EQ(getInputBasisMask(layout, S("x"), {}), 0);
}

TEST_F(LayoutUtilsTest, FactorMaximalIdentityPrefix) {
  LinearLayout layout({{S("x"), {{1}, {2}, {8}}}, {S("other"), {{4}}}},
                      {{S("out"), 16}}, /*requireSurjective=*/false);

  auto [size, quotient] =
      factorMaximalIdentityPrefix(layout, S("x"), S("out"), 8);
  EXPECT_EQ(size, 4);
  EXPECT_EQ(LinearLayout::identity1D(size, S("x"), S("out")) * quotient,
            layout);
}

TEST_F(LayoutUtilsTest, RenameDimsPreservesOrder) {
  LinearLayout layout({{S("a"), {{1, 0}}}, {S("b"), {{0, 1}}}},
                      {{S("x"), 2}, {S("y"), 2}}, /*requireSurjective=*/true);

  auto renamed =
      renameLinearLayoutDims(layout, {{S("a"), S("b")}, {S("b"), S("a")}},
                             {{S("x"), S("y")}, {S("y"), S("x")}});
  EXPECT_THAT(llvm::to_vector(renamed.getInDimNames()),
              ::testing::ElementsAre(S("b"), S("a")));
  EXPECT_THAT(llvm::to_vector(renamed.getOutDimNames()),
              ::testing::ElementsAre(S("y"), S("x")));
  EXPECT_EQ(renamed, LinearLayout({{S("b"), {{1, 0}}}, {S("a"), {{0, 1}}}},
                                  {{S("y"), 2}, {S("x"), 2}},
                                  /*requireSurjective=*/true));

  EXPECT_THAT(
      llvm::to_vector(layout.renameInDim(S("a"), S("c")).getInDimNames()),
      ::testing::ElementsAre(S("c"), S("b")));
}

TEST_F(LayoutUtilsTest, GetRepsWithSubsetOfOutputDims) {
  LinearLayout cvt({{S("in"), {{1, 0}, {0, 1}}}}, {S("out0"), S("out1")});
  LinearLayout tile({{S("in"), {{1}}}}, {S("out0")});

  auto reps = getReps(cvt, tile);
  ASSERT_TRUE(reps);
  EXPECT_EQ(*reps, LinearLayout({{S("in"), {{0, 0}, {0, 1}}}},
                                {{S("out0"), 2}, {S("out1"), 2}},
                                /*requireSurjective=*/false));
}

TEST_F(LayoutUtilsTest, LogicalOutputCoverageDistinguishesJointCoverage) {
  // 11 radix-2 digits on a 64x64 cover; the coupled digit reaches dim0 = 32
  // and dim1 = 32 only together, so the 33x48 box is not covered.
  LinearLayout layout({{S("register"),
                        {{1, 0},
                         {2, 0},
                         {4, 0},
                         {8, 0},
                         {16, 0},
                         {0, 1},
                         {0, 2},
                         {0, 4},
                         {0, 8},
                         {0, 16},
                         {32, 32}}}},
                      {{S("dim0"), 64}, {S("dim1"), 64}},
                      /*requireSurjective=*/false);
  EXPECT_FALSE(layout.isSurjective());
  EXPECT_FALSE(coversAllLogicalOutputs(layout, {33, 48}));
  // A padded box inside the span is covered without surjectivity.
  EXPECT_TRUE(coversAllLogicalOutputs(layout, {32, 32}));
  EXPECT_FALSE(coversAllLogicalOutputs(layout, {65, 1}));
  EXPECT_FALSE(coversAllLogicalOutputs(layout, {33}));
  auto pow2 = LinearLayout::identity1D(64, S("register"), S("dim0")) *
              LinearLayout::identity1D(64, S("lane"), S("dim1"));
  EXPECT_TRUE(coversAllLogicalOutputs(pow2, {64, 64}));
}

TEST_F(LayoutUtilsTest, LogicalOutputCoverageNeedsOwnedOddCoordinate) {
  auto exact = exactIdentity1D(96, S("register"), S("dim0"));
  EXPECT_TRUE(coversAllLogicalOutputs(exact, {96}));
  // Without an odd digit only the binary part [0, 32) is reachable.
  LinearLayout::BasesT bases;
  bases[S("lane")] = {{1}, {2}, {4}, {8}, {16}};
  LinearLayout::RadicesT radices;
  radices[S("lane")] = {2, 2, 2, 2, 2};
  LinearLayout binaryOnly(std::move(bases), std::move(radices),
                          {{S("dim0"), 96}}, /*requireSurjective=*/false);
  EXPECT_TRUE(coversAllLogicalOutputs(binaryOnly, {32}));
  EXPECT_FALSE(coversAllLogicalOutputs(binaryOnly, {33}));
}

// A transpose moves each out dim's size and digits with its coordinate.
TEST_F(LayoutUtilsTest, TransposeKeepsProductRadices) {
  auto reg = S("register"), lane = S("lane"), dim0 = S("dim0");
  auto dim1 = S("dim1");
  auto layout =
      exactIdentity1D(3, reg, dim0) * LinearLayout::identity1D(64, lane, dim1);
  auto transposed = transposeLinearLayout(layout, {1, 0});
  EXPECT_EQ(transposed.getOutDimSize(dim0), 64);
  EXPECT_EQ(transposed.getOutDimSize(dim1), 3);
  EXPECT_EQ(
      transposed.apply({{reg, 2}, {lane, 39}}),
      (SmallVector<std::pair<StringAttr, int32_t>>{{dim0, 39}, {dim1, 2}}));
  EXPECT_EQ(transposeLinearLayout(transposed, {1, 0}), layout);
}

TEST_F(LayoutUtilsTest, ProductConversionScopeIsExact) {
  auto reg = S("register"), lane = S("lane"), warp = S("warp");
  auto block = S("block"), dim0 = S("dim0"), dim1 = S("dim1");
  auto dim2 = S("dim2");

  auto registerOnly = exactIdentity1D(6, reg, dim0) *
                      LinearLayout::zeros1D(1, lane, dim0) *
                      LinearLayout::zeros1D(1, warp, dim0) *
                      LinearLayout::zeros1D(1, block, dim0);
  EXPECT_EQ(classifyProductConversionScope(registerOnly, registerOnly, {6}),
            ProductConversionScope::Register);

  auto laneSrc =
      makeRegLaneLayout(ctx, {{1, 0}}, {3}, {{0, 1}, {0, 2}, {0, 4}}, {}, 3);
  auto laneDst =
      makeRegLaneLayout(ctx, {{1, 0}}, {3}, {{0, 4}, {0, 2}, {0, 1}}, {}, 3);
  EXPECT_EQ(classifyProductConversionScope(laneSrc, laneDst, {3, 8}),
            ProductConversionScope::Warp);

  auto blockSrc = exactIdentity1D(3, reg, dim0) *
                  LinearLayout::identity1D(2, lane, dim1) *
                  LinearLayout::identity1D(2, warp, dim2) *
                  LinearLayout::zeros1D(1, block, dim0);
  auto blockDst =
      (exactIdentity1D(3, reg, dim0) * LinearLayout::identity1D(2, lane, dim2) *
       LinearLayout::identity1D(2, warp, dim1) *
       LinearLayout::zeros1D(1, block, dim0))
          .transposeOuts({dim0, dim1, dim2});
  EXPECT_EQ(classifyProductConversionScope(blockSrc, blockDst, {3, 2, 2}),
            ProductConversionScope::Block);

  // An odd digit owned by warps on either side crosses warps.
  auto oddWarp = LinearLayout::zeros1D(1, reg, dim0) *
                 LinearLayout::zeros1D(1, lane, dim0) *
                 exactIdentity1D(3, warp, dim0) *
                 LinearLayout::zeros1D(1, block, dim0);
  auto oddRegister = exactIdentity1D(3, reg, dim0) *
                     LinearLayout::zeros1D(1, lane, dim0) *
                     LinearLayout::zeros1D(1, warp, dim0) *
                     LinearLayout::zeros1D(1, block, dim0);
  EXPECT_EQ(classifyProductConversionScope(oddWarp, oddRegister, {3}),
            ProductConversionScope::Block);
  EXPECT_EQ(classifyProductConversionScope(oddRegister, oddWarp, {3}),
            ProductConversionScope::Block);

  auto pow2Src = LinearLayout::identity1D(2, lane, dim0) *
                 LinearLayout::identity1D(2, lane, dim1);
  auto pow2Dst = (LinearLayout::identity1D(2, lane, dim1) *
                  LinearLayout::identity1D(2, lane, dim0))
                     .transposeOuts({dim0, dim1});
  EXPECT_EQ(classifyProductConversionScope(pow2Src, pow2Dst, {2, 2}),
            ProductConversionScope::Warp);
}

TEST_F(LayoutUtilsTest, ProductConversionScopeHonorsLogicalBoundsAndReplicas) {
  auto lane = S("lane");
  auto source = exactIdentity1D(6, S("register"), S("dim0")) *
                LinearLayout::zeros1D(1, lane, S("dim0")) *
                LinearLayout::zeros1D(1, S("warp"), S("dim0")) *
                LinearLayout::zeros1D(1, S("block"), S("dim0"));
  EXPECT_EQ(classifyProductConversionScope(source, source, {5}),
            ProductConversionScope::Register);

  // Every destination replica needs the value, so a lane replica is Warp.
  auto bases = source.getBases();
  auto radices = source.getInputRadices();
  bases[lane].push_back({0});
  radices[lane].push_back(2);
  LinearLayout replicated(std::move(bases), std::move(radices),
                          source.getOutDims(), /*requireSurjective=*/true);
  EXPECT_EQ(classifyProductConversionScope(source, replicated, {5}),
            ProductConversionScope::Warp);

  // dim1 bit 2 moves from a register to a lane but is in bounds only at 8.
  auto src =
      makeRegLaneLayout(ctx, {{0, 4}, {1, 0}}, {2, 3}, {{0, 1}, {0, 2}}, {}, 3);
  auto dst =
      makeRegLaneLayout(ctx, {{1, 0}}, {3}, {{0, 1}, {0, 2}, {0, 4}}, {}, 3);
  EXPECT_EQ(classifyProductConversionScope(src, dst, {3, 4}),
            ProductConversionScope::Register);
  EXPECT_EQ(classifyProductConversionScope(src, dst, {3, 8}),
            ProductConversionScope::Warp);
}

TEST_F(LayoutUtilsTest, ProductInputMapAppliesClosedForm) {
  auto reg = S("register"), lane = S("lane");
  // The register and lane maps L with a(registerOrder(L(x))) == b(x).
  auto expectMap = [&](const LinearLayout &a, const LinearLayout &b,
                       const Bases &registerMap, const Bases &laneMap) {
    auto map = mapProductConversionInputs(a, b);
    ASSERT_TRUE(map);
    EXPECT_EQ(map->registerMap.getBases().lookup(reg), registerMap);
    EXPECT_EQ(map->laneMap.getBases().lookup(lane), laneMap);
  };
  Bases lanes = {{0, 1}, {0, 2}, {0, 4}};
  {
    SCOPED_TRACE("binary projection");
    expectMap(
        makeRegLaneLayout(ctx, {{1, 0}}, {3}, lanes, {}, 3),
        makeRegLaneLayout(ctx, {{1, 0}}, {3}, {{0, 2}, {0, 4}, {0, 1}}, {}, 3),
        {{1}}, {{2}, {4}, {1}});
  }
  {
    // b = 2 * a on the odd coordinate, so L = 2 and its inverse is 3 mod 5.
    SCOPED_TRACE("odd-digit map");
    auto a = makeRegLaneLayout(ctx, {{1, 0}}, {5}, lanes, {}, 5);
    auto b = makeRegLaneLayout(ctx, {{2, 0}}, {5}, lanes, {}, 5);
    expectMap(a, b, {{2}}, {{1}, {2}, {4}});
    expectMap(b, a, {{3}}, {{1}, {2}, {4}});
    EXPECT_TRUE(
        mapProductConversionInputs(a, b)->registerMap.hasNonBinaryCarrier());
  }
  {
    // b's free lane 0 takes a's kernel vector (lane 2).
    SCOPED_TRACE("broadcast kernel");
    expectMap(makeRegLaneLayout(ctx, {{0, 4}, {1, 0}}, {2, 3},
                                {{0, 1}, {0, 2}, {0, 0}}, {}, 3),
              makeRegLaneLayout(ctx, {{0, 4}, {1, 0}}, {2, 3},
                                {{0, 0}, {0, 1}, {0, 2}}, {}, 3),
              {{1}, {2}}, {{4}, {1}, {2}});
  }
  {
    // L's register output is o-last: a's binary digit, then 2 * o.
    SCOPED_TRACE("odd digit before a binary digit");
    auto oddFirst = makeRegLaneLayout(ctx, {{1, 0}, {0, 4}}, {3, 2},
                                      {{0, 1}, {0, 2}}, {}, 3);
    auto oddLast = makeRegLaneLayout(ctx, {{0, 4}, {1, 0}}, {2, 3},
                                     {{0, 1}, {0, 2}}, {}, 3);
    expectMap(oddFirst, oddLast, {{1}, {2}}, {{1}, {2}});
    expectMap(oddLast, oddFirst, {{2}, {1}}, {{1}, {2}});
  }
  {
    SCOPED_TRACE("pow2 control");
    auto binary =
        makeRegLaneLayout(ctx, {{1, 0}, {2, 0}}, {2, 2}, lanes, {}, 4);
    expectMap(binary, binary, {{1}, {2}}, {{1}, {2}, {4}});
    EXPECT_FALSE(mapProductConversionInputs(binary, binary)
                     ->registerMap.hasNonBinaryCarrier());
  }
}

TEST_F(LayoutUtilsTest, ProductInputMapRejectsNonBijections) {
  {
    // b's register bit is a's lane bit: not separable per input dim.
    SCOPED_TRACE("coupled register and lane");
    auto a = makeRegLaneLayout(ctx, {{0, 1}, {1, 0}}, {2, 3}, {{0, 2}, {0, 4}},
                               {}, 3);
    auto b = makeRegLaneLayout(ctx, {{0, 2}, {1, 0}}, {2, 3}, {{0, 1}, {0, 4}},
                               {}, 3);
    EXPECT_FALSE(mapProductConversionInputs(a, b));
  }
  auto reg = S("register"), lane = S("lane"), warp = S("warp");
  auto dim0 = S("dim0"), dim1 = S("dim1"), dim2 = S("dim2");
  {
    SCOPED_TRACE("out dims, in-dim order or in-dim sizes differ");
    auto a =
        exactIdentity1D(3, reg, dim0) * LinearLayout::identity1D(8, lane, dim1);
    auto wider = a * LinearLayout::zeros1D(1, lane, dim1, 2);
    auto reordered =
        LinearLayout::identity1D(8, lane, dim1) * exactIdentity1D(3, reg, dim0);
    auto fewerLanes = exactIdentity1D(3, reg, dim0) *
                      LinearLayout::identity1D(4, lane, dim1) *
                      LinearLayout::zeros1D(1, lane, dim1, 2);
    for (const LinearLayout *b : {&wider, &reordered, &fewerLanes})
      EXPECT_FALSE(mapProductConversionInputs(a, *b));
  }
  {
    SCOPED_TRACE("b broadcasts a lane bit that a uses");
    auto a =
        exactIdentity1D(3, reg, dim0) * LinearLayout::identity1D(4, lane, dim1);
    auto b = exactIdentity1D(3, reg, dim0) *
             LinearLayout::identity1D(2, lane, dim1) *
             LinearLayout::zeros1D(2, lane, dim1, 2);
    EXPECT_FALSE(mapProductConversionInputs(a, b));
  }
  {
    SCOPED_TRACE("warps swap");
    auto a = exactIdentity1D(3, reg, dim0) *
             LinearLayout::identity1D(2, lane, dim1) *
             LinearLayout::identity1D(2, warp, dim1) *
             LinearLayout::identity1D(2, warp, dim2);
    auto b = exactIdentity1D(3, reg, dim0) *
             LinearLayout::identity1D(2, lane, dim1) *
             LinearLayout::identity1D(2, warp, dim2) *
             LinearLayout::identity1D(2, warp, dim1);
    EXPECT_FALSE(mapProductConversionInputs(a, b));
  }
  {
    SCOPED_TRACE("odd digit in warps");
    auto make = [&](StringAttr regDim, StringAttr warpDim) {
      return (exactIdentity1D(3, reg, regDim) *
              LinearLayout::identity1D(2, lane, dim2) *
              exactIdentity1D(3, warp, warpDim))
          .transposeOuts({dim0, dim1, dim2});
    };
    EXPECT_FALSE(
        mapProductConversionInputs(make(dim0, dim1), make(dim1, dim0)));
    // Identical too, with the odd warp digit past the register digits.
    auto same = (exactIdentity1D(3, reg, dim0) *
                 LinearLayout::identity1D(2, lane, dim2) *
                 exactIdentity1D(6, warp, dim1))
                    .transposeOuts({dim0, dim1, dim2});
    EXPECT_FALSE(mapProductConversionInputs(same, same));
  }
}

} // namespace
} // namespace mlir::triton
