#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/LayoutUtils.h"
#include "unittest/Util/ScopedAllowNpot.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <set>

namespace mlir {
std::ostream &operator<<(std::ostream &os, StringAttr str) {
  os << str.str();
  return os;
}
} // namespace mlir

namespace mlir::triton {
namespace {

using ::llvm::to_vector;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

using BasesT = LinearLayout::BasesT;

class LinearLayoutTest : public ::testing::Test {
public:
  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LinearLayoutTest, Empty) {
  LinearLayout layout = LinearLayout::empty();
  EXPECT_THAT(layout.getBases(), IsEmpty());
  EXPECT_THAT(to_vector(layout.getInDimNames()), IsEmpty());
  EXPECT_THAT(to_vector(layout.getOutDimNames()), IsEmpty());
}

TEST_F(LinearLayoutTest, Identity1D) {
  LinearLayout layout =
      LinearLayout::identity1D(32, S("testIns"), S("testOuts"));
  EXPECT_THAT(layout, LinearLayout({{S("testIns"), {{1}, {2}, {4}, {8}, {16}}}},
                                   {S("testOuts")}));
  EXPECT_THAT(to_vector(layout.getInDimNames()), ElementsAre(S("testIns")));
  EXPECT_THAT(to_vector(layout.getOutDimNames()), ElementsAre(S("testOuts")));
  EXPECT_THAT(layout.getInDimSizeLog2(S("testIns")), 5);
  EXPECT_THAT(layout.getOutDimSizeLog2(S("testOuts")), 5);
}

TEST_F(LinearLayoutTest, Identity1DSize1) {
  LinearLayout layout =
      LinearLayout::identity1D(1, S("testIns"), S("testOuts"));
  EXPECT_EQ(layout, LinearLayout({{S("testIns"), {}}}, {S("testOuts")}));
  EXPECT_THAT(to_vector(layout.getInDimNames()), ElementsAre(S("testIns")));
  EXPECT_THAT(to_vector(layout.getOutDimNames()), ElementsAre(S("testOuts")));
  EXPECT_THAT(layout.getInDimSizeLog2(S("testIns")), 0);
  EXPECT_THAT(layout.getOutDimSizeLog2(S("testOuts")), 0);
}

TEST_F(LinearLayoutTest, NpotOutDimIsNotALayout) {
  BasesT bases;
  bases[S("in")] = {{1}, {2}};
  EXPECT_FALSE(LinearLayout::tryCreate(bases, {{S("out"), 3}},
                                       /*requireSurjective=*/false));
  EXPECT_TRUE(LinearLayout::tryCreate(bases, {{S("out"), 4}},
                                      /*requireSurjective=*/false));
}

TEST_F(LinearLayoutTest, Zeros1D) {
  LinearLayout layout = LinearLayout::zeros1D(32, S("ins"), S("outs"));
  EXPECT_EQ(layout,
            LinearLayout({{S("ins"), {{0}, {0}, {0}, {0}, {0}}}}, {S("outs")}));
}

TEST_F(LinearLayoutTest, MultiplyIdentity) {
  LinearLayout prod = LinearLayout::identity1D(16, S("in"), S("out")) *
                      LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout(
                      {{S("in"),
                        {{1}, {2}, {4}, {8}, {16}, {32}, {64}, {128}, {256}}}},
                      {S("out")}));
  EXPECT_THAT(to_vector(prod.getInDimNames()), ElementsAre(S("in")));
  EXPECT_THAT(to_vector(prod.getOutDimNames()), ElementsAre(S("out")));
}

TEST_F(LinearLayoutTest, MultiplyDisjoint) {
  LinearLayout prod = LinearLayout::identity1D(32, S("in1"), S("out1")) *
                      LinearLayout::identity1D(16, S("in2"), S("out2"));
  EXPECT_EQ(prod, LinearLayout(
                      {
                          {S("in1"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                          {S("in2"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                      },
                      {S("out1"), S("out2")}));
  EXPECT_THAT(to_vector(prod.getInDimNames()), ElementsAre(S("in1"), S("in2")));
  EXPECT_THAT(to_vector(prod.getOutDimNames()),
              ElementsAre(S("out1"), S("out2")));
}

TEST_F(LinearLayoutTest, MultiplyByEmpty) {
  LinearLayout prod =
      LinearLayout::empty() * LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout::identity1D(32, S("in"), S("out")));
}

TEST_F(LinearLayoutTest, MultiplyByZeros) {
  LinearLayout prod = LinearLayout::identity1D(8, S("in"), S("out")) *
                      LinearLayout::zeros1D(16, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout({{S("in"), {{1}, {2}, {4}, {0}, {0}, {0}, {0}}}},
                               {S("out")}));
}

TEST_F(LinearLayoutTest, MultiplyZerosByDegenerate) {
  LinearLayout prod = LinearLayout::zeros1D(16, S("in"), S("out1")) *
                      LinearLayout({{S("in"), {}}}, {S("out2")});
  EXPECT_EQ(prod, LinearLayout({{S("in"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}}}},
                               {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, MultiplyEmptyIdentityAndZeros) {
  LinearLayout prod = LinearLayout::identity1D(0, S("in"), S("out")) *
                      LinearLayout::zeros1D(4, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout({{S("in"), {{0}, {0}}}}, {S("out")}));
}

TEST_F(LinearLayoutTest, MultiplyOverlapping) {
  LinearLayout prod = LinearLayout::identity1D(4, S("in"), S("out1")) *
                      LinearLayout::identity1D(8, S("in"), S("out2"));
  EXPECT_EQ(prod,
            LinearLayout({{S("in"), {{1, 0}, {2, 0}, {0, 1}, {0, 2}, {0, 4}}}},
                         {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, TimesEquals) {
  LinearLayout prod = LinearLayout::empty();
  prod *= LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout::identity1D(32, S("in"), S("out")));
}

TEST_F(LinearLayoutTest, ConcatRequiresDisjointDimensions) {
#ifdef NDEBUG
  SUCCEED();
#else
  auto layout = LinearLayout::identity1D(2, S("in"), S("out"));
  ASSERT_DEATH((void)layout.concatIns(layout), "disjoint input dimensions");
  ASSERT_DEATH((void)layout.concatOuts(layout), "disjoint output dimensions");
#endif
}

TEST_F(LinearLayoutTest, GetOutDimSizeLog2) {
  LinearLayout layout(
      {
          {S("in0"), {{0}, {0}, {0}}},
          {S("in1"), {{1}, {2}}},
      },
      {S("dim0")});
  EXPECT_EQ(layout.getOutDimSizeLog2(S("dim0")), 2);
}

TEST_F(LinearLayoutTest, TransposeOuts) {
  LinearLayout layout = (LinearLayout::identity1D(32, S("in1"), S("out1")) *
                         LinearLayout::identity1D(16, S("in2"), S("out2")))
                            .transposeOuts({S("out2"), S("out1")});
  EXPECT_THAT(to_vector(layout.getOutDimNames()),
              ElementsAre(S("out2"), S("out1")));
  EXPECT_EQ(layout,
            LinearLayout(
                {
                    {S("in1"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}}},
                    {S("in2"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                },
                {S("out2"), S("out1")}));
}

TEST_F(LinearLayoutTest, TransposeOutsDegenerate) {
  LinearLayout layout = (LinearLayout::identity1D(32, S("in1"), S("out1")) *
                         LinearLayout::identity1D(1, S("in2"), S("out2")))
                            .transposeOuts({S("out2"), S("out1")});
  EXPECT_THAT(to_vector(layout.getOutDimNames()),
              ElementsAre(S("out2"), S("out1")));
  EXPECT_EQ(layout,
            LinearLayout(
                {
                    {S("in1"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}}},
                    {S("in2"), {}},
                },
                {S("out2"), S("out1")}));
}

TEST_F(LinearLayoutTest, TransposeIns) {
  LinearLayout layout = (LinearLayout::identity1D(32, S("in1"), S("out1")) *
                         LinearLayout::identity1D(16, S("in2"), S("out2")))
                            .transposeIns({S("in2"), S("in1")});
  EXPECT_THAT(to_vector(layout.getInDimNames()),
              ElementsAre(S("in2"), S("in1")));
  EXPECT_EQ(layout,
            LinearLayout(
                {
                    {S("in2"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("in1"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                },
                {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, EmptyToString) {
  EXPECT_EQ(LinearLayout::empty().toString(), "\n(empty layout)");
}

TEST_F(LinearLayoutTest, Apply) {
  LinearLayout layout(
      {
          {S("in1"), {{4, 2}, {2, 1}, {1, 0}}},
          {S("in2"), {{1, 2}, {2, 1}}},
      },
      {{S("out1"), 8}, {S("out2"), 4}}, /*requireSurjective=*/false);
  EXPECT_THAT(layout.apply({{S("in1"), 0}, {S("in2"), 0}}),
              ElementsAre(Pair(S("out1"), 0), Pair(S("out2"), 0)));
  EXPECT_THAT(layout.apply({{S("in2"), 0}, {S("in1"), 1}}),
              ElementsAre(Pair(S("out1"), 4), Pair(S("out2"), 2)));
  EXPECT_THAT(layout.apply({{S("in2"), 1}, {S("in1"), 0}}),
              ElementsAre(Pair(S("out1"), 1), Pair(S("out2"), 2)));
}

// This is really more of a benchmark than a test.  We're checking that it
// doesn't take so long to run that a human notices and says "hmm".  :)
TEST_F(LinearLayoutTest, ConstructLargeLayout) {
  std::vector<std::vector<int32_t>> pows2;
  for (int i = 0; i < 25; i++) {
    pows2.emplace_back().push_back(1 << i);
  }
  LinearLayout layout({{S("in"), pows2}}, {S("out")});
  (void)layout;
}

TEST_F(LinearLayoutTest, Compose) {
  LinearLayout l1(
      {
          {S("in1"), {{1, 1}, {0, 1}}},
          {S("in2"), {{1, 0}, {1, 2}}},
      },
      {S("out1"), S("out2")});
  LinearLayout l2(
      {
          {S("out1"), {{2, 2}, {1, 0}}},
          {S("out2"), {{1, 1}, {2, 1}}},
      },
      {S("out3"), S("out4")});
  LinearLayout composition = l1.compose(l2);
  EXPECT_EQ(composition,
            LinearLayout(
                {
                    {S("in1"), {{3, 3}, {1, 1}}},
                    {S("in2"), {{2, 2}, {0, 3}}},
                },
                {{S("out3"), 4}, {S("out4"), 4}}, /*requireSurjective=*/false));
  EXPECT_FALSE(composition.isSurjective());
}

TEST_F(LinearLayoutTest, Compose4D) {
  LinearLayout l1(
      {{S("in0"), {{1, 0, 0, 0}, {2, 0, 0, 0}}},
       {S("in1"), {{4, 0, 0, 0}, {8, 0, 0, 0}, {16, 0, 0, 0}, {32, 0, 0, 0}}},
       {S("in2"), {{0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}}},
       {S("in3"), {}}},
      {S("out3"), S("out0"), S("out1"), S("out2")});
  LinearLayout l2(
      {
          {S("out3"),
           {{1, 0, 0, 0},
            {2, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}}},
          {S("out0"), {{0, 1, 0, 0}}},
          {S("out1"), {{0, 0, 1, 0}}},
          {S("out2"), {{0, 0, 0, 1}, {0, 0, 0, 2}}},
      },
      {S("out3"), S("out2"), S("out1"), S("out0")});
  EXPECT_EQ(
      l1.compose(l2),
      LinearLayout(
          {
              {S("in0"), {{1, 0, 0, 0}, {2, 0, 0, 0}}},
              {S("in1"),
               {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
              {S("in2"), {{0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}}},
              {S("in3"), {}},
          },
          {{S("out3"), 4}, {S("out2"), 2}, {S("out1"), 2}, {S("out0"), 4}},
          /*requireSurjective=*/false));
}

TEST_F(LinearLayoutTest, ReshapeIns) {
  LinearLayout ll({{S("in1"), {{1}, {4}, {8}}}, {S("in2"), {{2}}}}, {S("out")});
  EXPECT_EQ(ll.reshapeIns({{S("in3"), {2}}, {S("in4"), {8}}}),
            LinearLayout({{S("in3"), {{1}}}, {S("in4"), {{4}, {8}, {2}}}},
                         {S("out")}));
}

TEST_F(LinearLayoutTest, ReshapeInsDegenerateIn) {
  LinearLayout ll({{S("in1"), {{1}, {4}, {2}}}, {S("in2"), {}}}, {S("out")});
  EXPECT_EQ(
      ll.reshapeIns({{S("in3"), {4}}, {S("in4"), {2}}}),
      LinearLayout({{S("in3"), {{1}, {4}}}, {S("in4"), {{2}}}}, {S("out")}));
}

TEST_F(LinearLayoutTest, ReshapeInsDegenerateOut) {
  LinearLayout ll({{S("in1"), {{1}, {4}}}, {S("in2"), {{2}}}}, {S("out")});
  EXPECT_EQ(
      ll.reshapeIns({{S("in3"), {8}}, {S("in4"), {1}}}),
      LinearLayout({{S("in3"), {{1}, {4}, {2}}}, {S("in4"), {}}}, {S("out")}));
}

TEST_F(LinearLayoutTest, ReshapeInsDegenerateFirstOut) {
  LinearLayout ll({{S("in1"), {{1}, {4}}}, {S("in2"), {{2}}}}, {S("out")});
  EXPECT_EQ(
      ll.reshapeIns({{S("in3"), {1}}, {S("in4"), {8}}}),
      LinearLayout({{S("in3"), {}}, {S("in4"), {{1}, {4}, {2}}}}, {S("out")}));
}

TEST_F(LinearLayoutTest, FlattenIns) {
  LinearLayout ll({{S("in1"), {{1}, {4}, {8}}}, {S("in2"), {{2}}}}, {S("out")});
  EXPECT_EQ(ll.flattenIns(),
            LinearLayout({{S("in1"), {{1}, {4}, {8}, {2}}}}, {S("out")}));
}

TEST_F(LinearLayoutTest, FlattenInsEdgeCases) {
  EXPECT_EQ(LinearLayout({{S("in1"), {}}}, {S("out")}).flattenIns(),
            LinearLayout({{S("in1"), {}}}, {S("out")}));
  EXPECT_EQ(LinearLayout({{S("in1"), {}}}, {}).flattenIns(),
            LinearLayout({{S("in1"), {}}}, {}));
  using BasesArray =
      ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>>;
  EXPECT_EQ(LinearLayout(BasesArray{}, {S("out")}).flattenIns(),
            LinearLayout(BasesArray{}, {S("out")}));
  EXPECT_EQ(LinearLayout(BasesArray{}, {}).flattenIns(),
            LinearLayout(BasesArray{}, {}));
}

TEST_F(LinearLayoutTest, ReshapeOuts) {
  LinearLayout ll({{S("in1"), {{1}, {4}, {8}}}, {S("in2"), {{3}}}}, {S("out")});
  EXPECT_EQ(ll.getTotalOutDimSize(), 16);
  EXPECT_EQ(
      ll.reshapeOuts({{S("out2"), {2}}, {S("out3"), {8}}}),
      LinearLayout({{S("in1"), {{1, 0}, {0, 2}, {0, 4}}}, {S("in2"), {{1, 1}}}},
                   {S("out2"), S("out3")}));
}

TEST_F(LinearLayoutTest, ReshapeOutsDegenerateIn) {
  LinearLayout ll({{S("in1"), {{1}, {4}, {2}}}, {S("in2"), {}}}, {S("out")});
  EXPECT_EQ(ll.reshapeOuts({{S("out1"), {4}}, {S("out2"), {2}}}),
            LinearLayout({{S("in1"), {{1, 0}, {0, 1}, {2, 0}}}, {S("in2"), {}}},
                         {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, ReshapeOutsDegenerateOut) {
  LinearLayout ll({{S("in1"), {{1}, {4}}}, {S("in2"), {{2}}}}, {S("out")});
  EXPECT_EQ(ll.reshapeOuts({{S("out1"), {8}}, {S("out2"), {1}}}),
            LinearLayout({{S("in1"), {{1, 0}, {4, 0}}}, {S("in2"), {{2, 0}}}},
                         {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, FlattenOuts) {
  LinearLayout ll({{S("in1"), {{1, 0}, {4, 1}, {8, 4}}}, {S("in2"), {{3, 2}}}},
                  {{S("out1"), 16}, {S("out2"), 8}},
                  /*requireSurjective=*/false);
  EXPECT_EQ(ll.flattenOuts(),
            LinearLayout({{S("in1"), {{1}, {4 + 16}, {8 + 4 * 16}}},
                          {S("in2"), {{3 + 2 * 16}}}},
                         {{S("out1"), 16 * 8}}, /*requireSurjective=*/false));
}

TEST_F(LinearLayoutTest, FlattenOutsEdgeCases) {
  EXPECT_EQ(LinearLayout({{S("in1"), {}}}, {S("out")}).flattenOuts(),
            LinearLayout({{S("in1"), {}}}, {S("out")}));
  EXPECT_EQ(LinearLayout({{S("in1"), {}}}, {}).flattenOuts(),
            LinearLayout({{S("in1"), {}}}, {}));
  using BasesArray =
      ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>>;
  EXPECT_EQ(LinearLayout(BasesArray{}, {S("out")}).flattenOuts(),
            LinearLayout(BasesArray{}, {S("out")}));
  EXPECT_EQ(LinearLayout(BasesArray{}, {}).flattenOuts(),
            LinearLayout(BasesArray{}, {}));
}

TEST_F(LinearLayoutTest, InvertAndCompose_Simple) {
  LinearLayout l1({{S("in1"), {{2}, {1}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in2"), {{4}, {1}, {2}}}}, {S("out")});

  // Inverse of l2 is
  //   out(1) => in2=2
  //   out(2) => in2=4
  //   out(4) => in2=1.
  //
  // Composing with l1 gives
  //   l2^-1(l1(1)) = l2^-1(2) = 4
  //   l2^-1(l1(2)) = l2^-1(1) = 2
  //   l2^-1(l1(4)) = l2^-1(4) = 1
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition,
            LinearLayout({{S("in1"), {{4}, {2}, {1}}}}, {S("in2")}));
  // L2 ∘ L2^-1 ∘ L1 == L1.
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_OverlappingIdentityDims) {
  auto layout = [&](int warpBasis) {
    return LinearLayout({{S("register"), {{32}}},
                         {S("lane"), {{1}, {2}, {4}, {8}, {16}}},
                         {S("warp"), {{warpBasis}, {128}}},
                         {S("block"), {}}},
                        {S("dim0")});
  };
  auto src = layout(64);
  auto dst = layout(96);
  EXPECT_EQ(dst.invertAndCompose(src).compose(src), dst);
  EXPECT_EQ(src.invertAndCompose(dst).compose(dst), src);
}

TEST_F(LinearLayoutTest, Invert_UnitInputDim) {
  LinearLayout layout({{S("x"), {{1}}}, {S("unit"), {}}}, {S("x")});
  EXPECT_EQ(layout.invert(),
            LinearLayout({{S("x"), {{1, 0}}}}, {S("x"), S("unit")}));
}

TEST_F(LinearLayoutTest, Lstsq) {
  LinearLayout superset({{S("storage"), {{1, 1}, {2, 0}}}},
                        {{S("x"), 4}, {S("y"), 2}},
                        /*requireSurjective=*/false);
  LinearLayout subset({{S("value"), {{1, 3}}}}, {{S("y"), 2}, {S("x"), 4}},
                      /*requireSurjective=*/false);
  LinearLayout outside({{S("value"), {{1, 0}}}}, {{S("x"), 4}, {S("y"), 2}},
                       /*requireSurjective=*/false);
  LinearLayout wrongDims({{S("value"), {{1}}}}, {S("z")});

  auto factor = lstsq(superset, subset);
  ASSERT_TRUE(factor);
  EXPECT_EQ(factor->compose(superset),
            subset.transposeOuts(llvm::to_vector(superset.getOutDimNames())));
  EXPECT_FALSE(lstsq(superset, outside));
  EXPECT_FALSE(lstsq(superset, wrongDims));
}

TEST_F(LinearLayoutTest, InvertAndComposeLargerA) {
  // Note that dim0 and dim1 are larger in sharedLaoyout
  auto regLayout =
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 32}, {32, 0}}},
                    {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{8, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")});
  auto sharedLayout = LinearLayout({{S("offset"),
                                     {{0, 1},
                                      {0, 2},
                                      {0, 4},
                                      {0, 8},
                                      {0, 16},
                                      {0, 32},
                                      {0, 64},
                                      {1, 8},
                                      {2, 16},
                                      {4, 32},
                                      {8, 0},
                                      {16, 0},
                                      {32, 0},
                                      {64, 0},
                                      {128, 0}}},
                                    {S("block"), {}}},
                                   {S("dim0"), S("dim1")});
  auto expected = LinearLayout(
      {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}, {4096, 0}}},
       {S("lane"), {{8, 0}, {16, 0}, {136, 0}, {272, 0}, {544, 0}}},
       {S("warp"), {{1024, 0}, {2048, 0}}},
       {S("block"), {}}},
      {{S("offset"), 32768}, {S("block"), 1}}, /*requireSurjective=*/false);
  EXPECT_EQ(regLayout.invertAndCompose(sharedLayout), expected);
  EXPECT_EQ(regLayout.compose(sharedLayout.invert()), expected);
}

TEST_F(LinearLayoutTest, InvertAndCompose_NonInjective) {
  LinearLayout l1({{S("in1"), {{2}, {1}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in2"), {{0}, {2}, {1}, {4}}}}, {S("out")});

  // The pseudo-inverse of l2 is
  //   out(1) => in2=4
  //   out(2) => in2=2
  //   out(4) => in2=8.
  //
  // Composing with l1 gives
  //   l2^-1(l1(1)) = l2^-1(2) = 2
  //   l2^-1(l1(2)) = l2^-1(0) = 4
  //   l2^-1(l1(4)) = l2^-1(4) = 8
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition,
            LinearLayout({{S("in1"), {{2}, {4}, {8}}}}, {{S("in2"), 16}},
                         /*requireSurjective=*/false));
  EXPECT_FALSE(composition.isSurjective());

  // L2 ∘ L2^-1 ∘ L1 == L1.
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_BroadcastedInDim) {
  LinearLayout l1({{S("in1"), {{2}, {1}, {4}}}, {S("in2"), {{0}}}}, {S("out")});
  LinearLayout l2({{S("in"), {{4}, {1}, {2}}}}, {S("out")});
  // Inverse of l2 is
  //   out(1) = 2
  //   out(2) = 4
  //   out(4) = 1
  //
  // Composing with l1 gives
  //
  //   l2^-1(l1(1, 0)) = l2^-1(2) = 4
  //   l2^-1(l1(2, 0)) = l2^-1(1) = 2
  //   l2^-1(l1(4, 0)) = l2^-1(4) = 1
  //   l2^-1(l1(0, 1)) = l2^-1(0) = 0
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition,
            LinearLayout({{S("in1"), {{4}, {2}, {1}}}, {S("in2"), {{0}}}},
                         {S("in")}));
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_BroadcastAtBeginningOfSecond) {
  LinearLayout l1({{S("in"), {{1}, {2}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in"), {{0}, {4}, {1}, {2}}}}, {S("out")});
  // Pseudo-inverse of l2 is
  //  out(1) = 4
  //  out(2) = 8
  //  out(4) = 2
  //
  // l1 is the identity, so composing with l1 gives back l2^-1.
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition,
            LinearLayout({{S("in"), {{4}, {8}, {2}}}}, {{S("in"), 16}},
                         /*requireSurjective=*/false));
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_BroadcastAtEndOfSecond) {
  LinearLayout l1({{S("in1"), {{1}, {2}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in2"), {{4}, {1}, {2}, {0}}}}, {S("out")});
  // Pseudo-inverse of l2 is
  //
  //  out(1) = 2
  //  out(2) = 4
  //  out(4) = 1
  //
  // l1 is the identity, so composing with l1 gives back l2^-1.
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition,
            LinearLayout({{S("in1"), {{2}, {4}, {1}}}}, {{S("in2"), 16}},
                         /*requireSurjective=*/false));
  EXPECT_TRUE(composition.compose(l2).equalIgnoringOutDimSizes(l1));
}

TEST_F(LinearLayoutTest, InvertAndCompose_BroadcastBeginningAndEndOfSecond) {
  LinearLayout l1({{S("in"), {{1}, {2}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in"), {{0}, {4}, {1}, {2}, {0}}}}, {S("out")});
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition,
            LinearLayout({{S("in"), {{4}, {8}, {2}}}}, {{S("in"), 32}},
                         /*requireSurjective=*/false));
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_Multidim) {
  LinearLayout l1(
      {{S("in1"), {{1, 0}, {0, 1}, {2, 0}, {3, 2}}}, {S("in2"), {{2, 2}}}},
      {S("out1"), S("out2")});
  LinearLayout l2({{S("in3"), {{0, 1}, {1, 0}, {0, 0}, {0, 2}, {2, 1}}}},
                  {S("out2"), S("out1")});

  LinearLayout c1 = l1.invertAndCompose(l2);
  EXPECT_EQ(c1.compose(l2),
            l1.transposeOuts(llvm::to_vector(l2.getOutDimNames())));

  LinearLayout c2 = l2.invertAndCompose(l1);
  EXPECT_EQ(c2.compose(l1),
            l2.transposeOuts(llvm::to_vector(l1.getOutDimNames())));
}

TEST_F(LinearLayoutTest, InvertAndCompose_BroadcastedDims) {
  LinearLayout l1({{S("in1"), {{1}, {2}, {4}}}, {S("in2"), {{0}}}}, {S("out")});
  LinearLayout l2({{S("in3"), {{1}, {2}, {4}}}, {S("in4"), {{0}}}}, {S("out")});
  LinearLayout c = l1.invertAndCompose(l2);
  EXPECT_EQ(c, LinearLayout(
                   {{S("in1"), {{1, 0}, {2, 0}, {4, 0}}}, {S("in2"), {{0, 0}}}},
                   {{S("in3"), 8}, {S("in4"), 2}},
                   /*requireSurjective=*/false));
  EXPECT_EQ(c.compose(l2),
            l1.transposeOuts(llvm::to_vector(l2.getOutDimNames())));
}

TEST_F(LinearLayoutTest, InvertAndCompose_BroadcastedDims2) {
  LinearLayout a({{S("in1"), {{1}, {2}}}, {S("in2"), {{0}}}}, {S("out")});
  LinearLayout b({{S("in3"), {{2}, {1}}}, {S("in4"), {{0}}}}, {S("out")});
  LinearLayout c = a.invertAndCompose(b);
  EXPECT_EQ(c,
            LinearLayout({{S("in1"), {{2, 0}, {1, 0}}}, {S("in2"), {{0, 0}}}},
                         {{S("in3"), 4}, {S("in4"), 2}},
                         /*requireSurjective=*/false));
  EXPECT_EQ(c.compose(b), a.transposeOuts(llvm::to_vector(b.getOutDimNames())));
}

TEST_F(LinearLayoutTest, InvertAndComposeBlockLocal) {
  auto memLayout =
      LinearLayout({{S("offset"), {{1}}}, {S("block"), {{0}}}}, {S("dim")});
  auto regLayout = LinearLayout({{S("register"), {}},
                                 {S("lane"), {{0}, {0}, {0}, {0}, {0}}},
                                 {S("warp"), {}},
                                 {S("block"), {{1}}}},
                                {S("dim")});

  EXPECT_FALSE(
      regLayout.invertAndCompose(memLayout).isIdentityOnOutDim(S("block")));
  auto local = invertAndComposeLocal(memLayout, regLayout, {S("block")});
  auto expected =
      LinearLayout({{S("register"), {}},
                    {S("lane"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}},
                    {S("warp"), {}},
                    {S("block"), {{1, 1}}}},
                   {{S("offset"), 2}, {S("block"), 2}},
                   /*requireSurjective=*/false);
  EXPECT_EQ(local, expected);
  EXPECT_TRUE(local.isIdentityOnOutDim(S("block")));
  EXPECT_FALSE(local.isTrivialOver(S("block")));
  EXPECT_EQ(local.compose(memLayout), regLayout);

  auto partiallyBroadcastMem = LinearLayout(
      {{S("offset"), {{2}}}, {S("block"), {{1}, {0}}}}, {S("dim")});
  auto partiallyDistributedReg =
      LinearLayout({{S("register"), {}},
                    {S("lane"), {{0}, {0}, {0}, {0}, {0}}},
                    {S("warp"), {}},
                    {S("block"), {{1}, {2}}}},
                   {S("dim")});
  EXPECT_FALSE(partiallyDistributedReg.invertAndCompose(partiallyBroadcastMem)
                   .isIdentityOnOutDim(S("block")));
  auto partiallyLocal = invertAndComposeLocal(
      partiallyBroadcastMem, partiallyDistributedReg, {S("block")});
  auto partiallyExpected =
      LinearLayout({{S("register"), {}},
                    {S("lane"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}},
                    {S("warp"), {}},
                    {S("block"), {{0, 1}, {1, 2}}}},
                   {{S("offset"), 2}, {S("block"), 4}},
                   /*requireSurjective=*/false);
  EXPECT_EQ(partiallyLocal, partiallyExpected);
  EXPECT_TRUE(partiallyLocal.isIdentityOnOutDim(S("block")));
  EXPECT_EQ(partiallyLocal.compose(partiallyBroadcastMem),
            partiallyDistributedReg);

  auto partitionedMem =
      LinearLayout({{S("offset"), {{1}}}, {S("block"), {{2}}}}, {S("dim")});
  auto transposedReg = LinearLayout({{S("register"), {{2}}},
                                     {S("lane"), {{0}, {0}, {0}, {0}, {0}}},
                                     {S("warp"), {}},
                                     {S("block"), {{1}}}},
                                    {S("dim")});
  EXPECT_EQ(invertAndComposeLocal(partitionedMem, transposedReg, {S("block")}),
            transposedReg.invertAndCompose(partitionedMem));
}

TEST_F(LinearLayoutTest, InvertAndCompose_IdentityInDim) {
  SmallVector<StringAttr> outDims = {S("dim0"), S("dim1"), S("dim2"),
                                     S("dim3"), S("dim4"), S("dim5"),
                                     S("dim6"), S("dim7"), S("dim8")};

  LinearLayout src({{S("register"),
                     {
                         {0, 0, 0, 0, 0, 0, 0, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 1, 0},
                     }},
                    {S("lane"),
                     {
                         {0, 0, 0, 0, 0, 0, 1, 0, 0},
                         {0, 0, 0, 0, 0, 1, 0, 0, 0},
                         {0, 0, 0, 0, 1, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 0, 0},
                         {0, 0, 1, 0, 0, 0, 0, 0, 0},
                     }},
                    {S("warp"),
                     {
                         {0, 1, 0, 0, 0, 0, 0, 0, 0},
                         {1, 0, 0, 0, 0, 0, 0, 0, 0},
                     }},
                    {S("block"), {}}},
                   outDims);
  LinearLayout dst({{S("register"),
                     {
                         {0, 0, 0, 0, 0, 0, 0, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 1, 0},
                     }},
                    {S("lane"),
                     {
                         {1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 1, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 1, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 1, 0, 0, 0, 0},
                     }},
                    {S("warp"),
                     {
                         {0, 0, 0, 0, 0, 1, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 1, 0, 0},
                     }},
                    {S("block"), {}}},
                   outDims);

  LinearLayout cvt = dst.invertAndCompose(src);
  SmallVector<std::pair<StringAttr, int32_t>> k = {
      {S("register"), 3}, {S("lane"), 0}, {S("warp"), 2}, {S("block"), 0}};

  EXPECT_EQ(dst.apply(k), src.apply(cvt.apply(k)));
}

TEST_F(LinearLayoutTest, NumConsecutiveInOut) {
  EXPECT_EQ(
      1,
      LinearLayout::identity1D(1, S("in"), S("out")).getNumConsecutiveInOut());
  EXPECT_EQ(
      4,
      LinearLayout::identity1D(4, S("in"), S("out")).getNumConsecutiveInOut());
  EXPECT_EQ(4, (LinearLayout::identity1D(4, S("in1"), S("out")) *
                LinearLayout::identity1D(8, S("in2"), S("out")))
                   .getNumConsecutiveInOut());
  EXPECT_EQ(4, (LinearLayout::identity1D(4, S("in"), S("out1")) *
                LinearLayout::identity1D(8, S("in"), S("out2")))
                   .getNumConsecutiveInOut());
  EXPECT_EQ(1, (LinearLayout::zeros1D(4, S("in"), S("out1")) *
                LinearLayout::identity1D(4, S("in"), S("out2")))
                   .getNumConsecutiveInOut());
  EXPECT_EQ(1, LinearLayout({{S("in"), {{1}, {2}, {4}, {9}}}}, {S("out")})
                   .getNumConsecutiveInOut());
  EXPECT_EQ(2, LinearLayout({{S("in"), {{1}, {2}, {4}, {10}}}}, {S("out")})
                   .getNumConsecutiveInOut());
  EXPECT_EQ(2, LinearLayout({{S("in"), {{1}, {4}, {2}}}}, {S("out")})
                   .getNumConsecutiveInOut());
  EXPECT_EQ(2, LinearLayout(
                   {
                       {S("in"), {{1}, {2}, {4}}},
                       {S("in2"), {{8}, {18}}},
                   },
                   {S("out")})
                   .getNumConsecutiveInOut());
}

TEST_F(LinearLayoutTest, ContiguousElemsAlongOutputDim) {
  EXPECT_EQ(8, LinearLayout::identity1D(8, S("offset"), S("offset"))
                   .contiguousElemsAlongOutputDim(S("offset")));
  EXPECT_EQ(1, LinearLayout({{S("logical"), {{3}}}}, {{S("offset"), 4}},
                            /*requireSurjective=*/false)
                   .contiguousElemsAlongOutputDim(S("offset")));
  EXPECT_EQ(4,
            LinearLayout({{S("lhs"), {{3}}}, {S("rhs"), {{1}}}}, {S("offset")})
                .contiguousElemsAlongOutputDim(S("offset")));
  EXPECT_EQ(2, LinearLayout({{S("logical"), {{1}, {4}}}}, {{S("offset"), 8}},
                            /*requireSurjective=*/false)
                   .contiguousElemsAlongOutputDim(S("offset")));
  EXPECT_EQ(8, LinearLayout({{S("logical"), {{1}, {4}}}, {S("free"), {{2}}}},
                            {S("offset")})
                   .contiguousElemsAlongOutputDim(S("offset")));
}

TEST_F(LinearLayoutTest, EqualsChecksOutDimSizes) {
  EXPECT_FALSE(LinearLayout::identity1D(4, S("in"), S("out")) ==
               LinearLayout({{S("in"), {{1}, {2}}}}, {{S("out"), 8}},
                            /*requireSurjective=*/false));
  EXPECT_TRUE(LinearLayout::identity1D(4, S("in"), S("out")) !=
              LinearLayout({{S("in"), {{1}, {2}}}}, {{S("out"), 8}},
                           /*requireSurjective=*/false));
  EXPECT_TRUE(LinearLayout::identity1D(4, S("in"), S("out"))
                  .equalIgnoringOutDimSizes(
                      LinearLayout({{S("in"), {{1}, {2}}}}, {{S("out"), 8}},
                                   /*requireSurjective=*/false)));
}

TEST_F(LinearLayoutTest, ResizeInDim) {
  auto layout = LinearLayout::identity1D(4, S("in"), S("out"));
  EXPECT_EQ(layout.resizeInDim(S("in"), 16),
            LinearLayout({{S("in"), {{1}, {2}, {0}, {0}}}}, {{S("out"), 4}},
                         /*requireSurjective=*/false));
  EXPECT_EQ(layout.resizeInDim(S("in"), 2),
            LinearLayout({{S("in"), {{1}}}}, {{S("out"), 4}},
                         /*requireSurjective=*/false));
}

TEST_F(LinearLayoutTest, Sublayout) {
  LinearLayout l1({{S("in1"), {{1, 0}, {0, 1}, {2, 0}}}, {S("in2"), {{0, 1}}}},
                  {S("out1"), S("out2")});
  EXPECT_EQ(l1.sublayout({S("in1"), S("in2")}, {S("out1")}),
            LinearLayout({{S("in1"), {{1}, {0}, {2}}}, {S("in2"), {{0}}}},
                         {S("out1")}));
  EXPECT_EQ(l1.sublayout({S("in2"), S("in1")}, {S("out1")}),
            LinearLayout({{S("in1"), {{1}, {0}, {2}}}, {S("in2"), {{0}}}},
                         {S("out1")}));
  EXPECT_EQ(l1.sublayout({S("in2"), S("in1")}, {S("out2"), S("out1")}), l1);
  EXPECT_EQ(l1.sublayout({S("in1")}, {S("out1")}),
            LinearLayout({{S("in1"), {{1}, {0}, {2}}}}, {S("out1")}));
  EXPECT_EQ(l1.sublayout({}, {}), LinearLayout::empty());
  EXPECT_EQ(l1.sublayout({S("in1")}, {}),
            LinearLayout({{S("in1"), {{}, {}, {}}}}, {}));
  EXPECT_EQ(l1.sublayout({}, {S("out1")}),
            LinearLayout(LinearLayout::BasesT{}, {{S("out1"), 4}},
                         /*requireSurjective=*/false));
}

TEST_F(LinearLayoutTest, SublayoutIsZero) {
  EXPECT_FALSE(LinearLayout::identity1D(4, S("in"), S("out"))
                   .sublayoutIsZero({S("in")}, {S("out")}));
  EXPECT_TRUE(LinearLayout::identity1D(4, S("in"), S("out"))
                  .sublayoutIsZero({}, {S("out")}));
  EXPECT_TRUE(LinearLayout::identity1D(4, S("in"), S("out"))
                  .sublayoutIsZero({S("in")}, {}));
  EXPECT_TRUE(
      LinearLayout::identity1D(4, S("in"), S("out")).sublayoutIsZero({}, {}));

  LinearLayout l1({{S("in1"), {{0, 1}, {0, 2}}}, {S("in2"), {{1, 1}}}},
                  {S("out1"), S("out2")});
  EXPECT_TRUE(l1.sublayoutIsZero({S("in1")}, {S("out1")}));
  EXPECT_FALSE(l1.sublayoutIsZero({S("in1")}, {S("out2")}));
  EXPECT_FALSE(l1.sublayoutIsZero({S("in2")}, {S("out1")}));
  EXPECT_FALSE(l1.sublayoutIsZero({S("in2")}, {S("out2")}));
}

TEST_F(LinearLayoutTest, FreeVariableMasks) {
  using llvm::to_vector;
  using AR = llvm::ArrayRef<std::pair<StringAttr, int32_t>>;

  EXPECT_EQ(AR(to_vector(LinearLayout::identity1D(4, S("in"), S("out"))
                             .getFreeVariableMasks())),
            AR({{S("in"), 0}}));
  EXPECT_EQ(
      AR(to_vector(
          LinearLayout::zeros1D(16, S("in"), S("out")).getFreeVariableMasks())),
      AR({{S("in"), 0b1111}}));
  EXPECT_EQ(AR(to_vector((LinearLayout::identity1D(2, S("in"), S("out")) *
                          LinearLayout::zeros1D(4, S("in"), S("out")) *
                          LinearLayout::identity1D(4, S("in"), S("out")) *
                          LinearLayout::zeros1D(2, S("in"), S("out")))
                             .getFreeVariableMasks())),
            AR({{S("in"), 0b100110}}));
  EXPECT_EQ(AR(to_vector((LinearLayout::identity1D(2, S("in"), S("out")) *
                          LinearLayout::zeros1D(4, S("in"), S("out")) *
                          LinearLayout::identity1D(4, S("in"), S("out")) *
                          LinearLayout::zeros1D(2, S("in"), S("out")))
                             .getFreeVariableMasks())),
            AR({{S("in"), 0b100110}}));
  EXPECT_EQ(AR(to_vector(LinearLayout({{S("in1"), {{1, 1}, {2, 2}, {0, 0}}},
                                       {S("in2"), {{1, 0}, {0, 1}, {2, 0}}}},
                                      {S("out1"), S("out2")})
                             .getFreeVariableMasks())),
            AR({{S("in1"), 0b100}, {S("in2"), 0b10}}));
}

TEST_F(LinearLayoutTest, QuotientOneDimension) {
  LinearLayout layout(
      {
          {S("dim1"), {{1, 0}}},
          {S("dim2"), {{0, 0}}},
      },
      {{S("dim1"), 2}, {S("dim2"), 1}}, /*requireSurjective=*/false);

  // Quotient over dim1, which is trivial
  auto quotientLayout = layout.quotient({S("dim1")});
  ASSERT_TRUE(quotientLayout.has_value());
  EXPECT_EQ(*quotientLayout, LinearLayout::zeros1D(2, S("dim2"), S("dim2")));
  // dim2 is zero, not the identity
  ASSERT_FALSE(quotientLayout->quotient({S("dim2")}).has_value());
}

TEST_F(LinearLayoutTest, QuotientSeveralDimensions) {
  LinearLayout layout(
      {
          {S("dim1"), {{1, 0}, {2, 0}, {4, 0}}},
          {S("dim2"), {{0, 1}, {0, 2}}},
      },
      {S("dim1"), S("dim2")});

  auto quotientLayout = layout.quotient({S("dim1"), S("dim2")});
  EXPECT_TRUE(quotientLayout.has_value());
}

TEST_F(LinearLayoutTest, QuotientMultipleTrivialDimensions) {
  LinearLayout layout(
      {
          {S("dim1"), {{1, 0, 2}, {2, 0, 1}}},
          {S("dim2"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
          {S("dim3"), {{0, 0, 1}, {0, 0, 2}}},
      },
      {S("dim1"), S("dim2"), S("dim3")});

  // Quotient over dim2 is trivial, even if there's some funny business
  // going on in the other dimensions
  auto quotientLayout = layout.quotient({S("dim2")});
  ASSERT_TRUE(quotientLayout.has_value());

  layout = LinearLayout(
      {
          {S("dim1"), {{1, 0, 2}, {2, 0, 1}}},
          {S("dim2"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
          {S("dim3"), {{0, 1, 1}, {0, 0, 2}}},
      },
      {S("dim1"), S("dim2"), S("dim3")});

  // As soon as one maps into the dimension being quotiented or out of it
  // (in this case dim3 depends on dim2), we cannot quotient
  quotientLayout = layout.quotient({S("dim2")});
  ASSERT_FALSE(quotientLayout.has_value());
}

TEST_F(LinearLayoutTest, QuotientEmptyLayout) {
  LinearLayout layout = LinearLayout::empty();

  // Quotienting over a dimension that doesn't exist is invalid
  auto quotientLayout = layout.quotient({S("dim1")});
  ASSERT_FALSE(quotientLayout.has_value());
}

TEST_F(LinearLayoutTest, QuotientIdentityMultipleDimensions) {
  // Test quotient on identity layout with multiple dimensions
  LinearLayout layout = LinearLayout::identity1D(8, S("dim1"), S("dim1")) *
                        LinearLayout::identity1D(2, S("dim2"), S("dim2")) *
                        LinearLayout::identity1D(4, S("dim3"), S("dim3"));

  // We can quotient over all dimensions in any order
  auto quotientLayout = layout.quotient({S("dim1"), S("dim3")});
  ASSERT_TRUE(quotientLayout.has_value());
  ASSERT_TRUE(quotientLayout->quotient({S("dim2")}).has_value());
}

LinearLayout getPackedCoordtoPaddedOffset(int M, int KPacked8b, StringAttr row,
                                          StringAttr col, StringAttr offset) {
  std::vector<std::vector<int>> basesRows, basesCols;
  for (int row = 1; row < M; row *= 2) {
    int col = 0;
    int linearCoord = row * KPacked8b + col;
    int offset = (linearCoord / 8) * 16 + (linearCoord % 8);
    basesRows.push_back({offset});
  }

  for (int col = 1; col < KPacked8b; col *= 2) {
    int row = 0;
    int linearCoord = row * KPacked8b + col;
    int offset = (linearCoord / 8) * 16 + (linearCoord % 8);
    basesCols.push_back({offset});
  }

  return LinearLayout({{row, basesRows}, {col, basesCols}},
                      {{offset, M * KPacked8b * 2}}, /*surjective*/ false);
}

TEST_F(LinearLayoutTest, BlackwellMixedPrecisionDotScaledSMEM) {
  std::vector<std::vector<int>> basesRows, basesCols, basesOffset;
  int numFp4Elems = 128;
  int M = 16;
  int KPacked8b = numFp4Elems / M / 2;
  int KPadded8b = numFp4Elems / M;

  for (int offset = 1; offset < M * KPadded8b; offset *= 2) {
    int linearCoordPacked = offset / 16 * 8 + offset % 8;
    int row = linearCoordPacked / KPacked8b;
    int col = linearCoordPacked % KPacked8b;
    basesOffset.push_back({row, col});
  }

  LinearLayout layout({{S("offset"), basesOffset}}, {S("row"), S("col")});
  LinearLayout layoutInverseComputed = layout.pseudoinvert();
  LinearLayout layoutInverseManual = getPackedCoordtoPaddedOffset(
      M, KPacked8b, S("row"), S("col"), S("offset"));

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < KPacked8b; ++j) {
      auto off1 = layoutInverseManual.apply({{S("row"), i}, {S("col"), j}});
      auto off2 = layoutInverseComputed.apply({{S("row"), i}, {S("col"), j}});
      EXPECT_EQ(off1[0].second, off2[0].second);
    }
  }
}

TEST_F(LinearLayoutTest, BlackwellMixedPrecisionDotScaledSMEMSwizzled) {
  int M = 16;
  int KPadded8b = 128;
  int KPacked8b = KPadded8b / 2;
  int elemBitWidth = 8;
  int tileWidthBytes = 128;
  int tileRows = 8;
  int tileCols = 8 * tileWidthBytes / elemBitWidth;
  int vec = 16;

  std::vector<std::vector<int>> bases2D;
  for (int colPadded = 1; colPadded < tileCols; colPadded *= 2) {
    int colPacked = colPadded / 16 * 8 + colPadded % 8;
    bases2D.push_back({0, colPacked});
  }
  for (int row = 1; row < tileRows; row *= 2) {
    int perPhase = 1;
    int maxPhase = 8;
    int colPadded = vec * ((row / perPhase) % maxPhase);
    int colPacked = colPadded / 16 * 8 + colPadded % 8;
    bases2D.push_back({row, colPacked});
  }

  LinearLayout layoutSwizzled({{S("offset"), bases2D}}, {S("row"), S("col")});
  layoutSwizzled = ensureLayoutNotSmallerThan(
      layoutSwizzled, {{S("row"), M}, {S("col"), KPacked8b}});

  auto layoutInverseSwizzled = layoutSwizzled.pseudoinvert();

  LinearLayout layoutInverseNoSwizzle = getPackedCoordtoPaddedOffset(
      M, KPacked8b, S("row"), S("col"), S("offset"));

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < KPacked8b; ++j) {
      auto nonSwizzleOffset =
          layoutInverseNoSwizzle.apply({{S("row"), i}, {S("col"), j}})[0]
              .second;
      auto swizzledOffset =
          layoutInverseSwizzled.apply({{S("row"), i}, {S("col"), j}})[0].second;
      int row = nonSwizzleOffset / KPadded8b;
      int col = nonSwizzleOffset % KPadded8b;
      int colSwizzled = ((col / 16) ^ (row % 8)) * 16 + col % 16;
      EXPECT_EQ(row * KPadded8b + colSwizzled, swizzledOffset);
    }
  }
}

static SmallVector<StringAttr> makeList(MLIRContext *ctx,
                                        llvm::ArrayRef<llvm::StringRef> list) {
  SmallVector<StringAttr> ret;
  for (auto s : list)
    ret.push_back(StringAttr::get(ctx, s));
  return ret;
}

TEST(SupremumTest, IdenticalLists) {
  MLIRContext ctx;
  SmallVector<StringAttr> x = makeList(&ctx, {"a", "b", "c"});
  SmallVector<StringAttr> y = makeList(&ctx, {"a", "b", "c"});
  EXPECT_EQ(supremum(x, y), x);
}

TEST(SupremumTest, NonUniqueSupremumFirstListPriority) {
  MLIRContext ctx;
  // sup([a, b], [a, c]) should yield [a, b, c]
  SmallVector<StringAttr> x = makeList(&ctx, {"a", "b"});
  SmallVector<StringAttr> y = makeList(&ctx, {"a", "c"});
  EXPECT_EQ(supremum(x, y), makeList(&ctx, {"a", "b", "c"}));
}

TEST(SupremumTest, NonUniqueSupremumAlternate) {
  MLIRContext ctx;
  // sup([a, b], [b, c]) should yield [a, b, c]
  SmallVector<StringAttr> x = makeList(&ctx, {"a", "b"});
  SmallVector<StringAttr> y = makeList(&ctx, {"b", "c"});
  EXPECT_EQ(supremum(x, y), makeList(&ctx, {"a", "b", "c"}));
}

TEST(SupremumTest, DifferentLengths) {
  MLIRContext ctx;
  // sup([a, b, c], [a, d]) should yield [a, b, c, d]
  SmallVector<StringAttr> x = makeList(&ctx, {"a", "b", "c"});
  SmallVector<StringAttr> y = makeList(&ctx, {"a", "d"});
  EXPECT_EQ(supremum(x, y), makeList(&ctx, {"a", "b", "c", "d"}));
}

TEST(SupremumTest, SupremumEmptyLists) {
  MLIRContext ctx;
  SmallVector<StringAttr> x;
  SmallVector<StringAttr> y;
  EXPECT_TRUE(supremum(x, y).empty());
}

TEST(SupremumTest, OneEmptyList) {
  MLIRContext ctx;
  // sup([a, b], []) should yield [a, b]
  SmallVector<StringAttr> x = makeList(&ctx, {"a", "b"});
  SmallVector<StringAttr> y;
  EXPECT_EQ(supremum(x, y), makeList(&ctx, {"a", "b"}));
}

#ifdef LLVM_ENABLE_ASSERTIONS
TEST(SupremumTest, ErrorOnInconsistentOrder) {
  MLIRContext ctx;
  // sup([a, b], [b, a]) has no consistent ordering so it should trigger
  // llvm_unreachable.
  SmallVector<StringAttr> x = makeList(&ctx, {"a", "b"});
  SmallVector<StringAttr> y = makeList(&ctx, {"b", "a"});
  ASSERT_DEATH({ supremum(x, y); }, "Supremum does not exist");
}
#endif

TEST_F(LinearLayoutTest, Divide_Basic) {
  // Test division when A = B * C.
  auto B = LinearLayout::identity1D(8, S("in"), S("out"));
  auto C = LinearLayout::zeros1D(16, S("in"), S("out"));
  auto isC = divideLeft(B * C, B);
  EXPECT_TRUE(isC.has_value());
  EXPECT_EQ(isC.value(), C);
  auto isB = divideRight(B * C, C);
  EXPECT_TRUE(isB.has_value());
  EXPECT_EQ(isB.value(), B);

  isB = divideLeft(C * B, C);
  EXPECT_TRUE(isB.has_value());
  EXPECT_EQ(isB.value(), B);
  isC = divideRight(C * B, B);
  EXPECT_TRUE(isC.has_value());
  EXPECT_EQ(isC.value(), C);
}

TEST_F(LinearLayoutTest, Divide_NonMatchingDims) {
  // If B contains an extra input dimension not present in A, division should
  // fail.
  LinearLayout A = LinearLayout::identity1D(32, S("in"), S("out"));
  LinearLayout B({{S("in"), {{1}, {2}, {4}, {8}}}, {S("extra"), {{0}}}},
                 {S("out")});
  auto candidateOpt = divideLeft(A, B);
  EXPECT_FALSE(candidateOpt.has_value());
  candidateOpt = divideRight(A, B);
  EXPECT_FALSE(candidateOpt.has_value());
}

TEST_F(LinearLayoutTest, Divide_Simple) {
  auto A = LinearLayout::identity1D(8, S("in"), S("out"));
  auto B = LinearLayout::identity1D(4, S("in"), S("out"));
  auto C = LinearLayout::identity1D(2, S("in"), S("out"));
  EXPECT_EQ(divideLeft(A, B), C);
  EXPECT_EQ(divideRight(A, B), C);

  A = LinearLayout::identity1D(8, S("in"), S("out"));
  C = LinearLayout::identity1D(1, S("in"), S("out"));
  EXPECT_EQ(divideLeft(A, A), C);
  EXPECT_EQ(divideRight(A, A), C);
}

TEST_F(LinearLayoutTest, Divide_2D) {
  LinearLayout l1(
      {
          {S("in1"), {{1, 1}, {2, 2}, {0, 8}, {0, 4}}},
          {S("in2"), {{0, 2}, {0, 1}}},
      },
      {S("out1"), S("out2")});
  LinearLayout l2(
      {
          {S("in1"), {{1, 1}, {2, 2}}},
          {S("in2"), {{0, 2}, {0, 1}}},
      },
      {S("out1"), S("out2")});
  LinearLayout l3({{S("in1"), {{0, 2}, {0, 1}}}, {S("in2"), {}}},
                  {S("out1"), S("out2")});
  ASSERT_EQ(l2 * l3, l1);
  ASSERT_EQ(divideLeft(l1, l2).value(), l3);
  ASSERT_EQ(divideRight(l1, l3).value(), l2);
}

TEST_F(LinearLayoutTest, Divide_EliminateInDim) {
  LinearLayout l1(
      {
          {S("in2"), {{0, 1}, {1, 0}}},
          {S("in1"), {{2, 0}, {0, 2}}},
      },
      {S("out1"), S("out2")});
  LinearLayout l2({{S("in2"), {{0, 1}, {1, 0}}}}, {S("out1"), S("out2")});
  LinearLayout l3({{S("in2"), {}}, {S("in1"), {{1, 0}, {0, 1}}}},
                  {S("out1"), S("out2")});
  ASSERT_EQ(l2 * l3, l1);
  EXPECT_EQ(divideLeft(l1, l2).value(), l3);

  l2 = LinearLayout({{S("in2"), {{0, 1}, {1, 0}}}, {S("in1"), {}}},
                    {S("out1"), S("out2")});
  l3 = LinearLayout({{S("in1"), {{1, 0}, {0, 1}}}}, {S("out1"), S("out2")});
  ASSERT_EQ(l2 * l3, l1);
  EXPECT_EQ(divideRight(l1, l3).value(), l2);

  LinearLayout l4({{S("in1"), {{0, 1}, {0, 2}}}, {S("in2"), {}}},
                  {S("out1"), S("out2")});
  LinearLayout l5({{S("in1"), {{0, 1}, {0, 2}}}}, {S("out1"), S("out2")});
  LinearLayout l6({{S("in1"), {}}, {S("in2"), {}}}, {S("out1"), S("out2")});
  ASSERT_EQ(l5 * l6, l4);
  EXPECT_EQ(divideLeft(l4, l5).value(), l6);
  EXPECT_EQ(divideRight(l4, l5).value(), l6);

  LinearLayout l7({{S("in1"), {}}, {S("in2"), {{0, 1}}}, {S("in3"), {}}},
                  {S("out1"), S("out2")});
  LinearLayout l8({{S("in2"), {{0, 1}}}}, {S("out1"), S("out2")});
  LinearLayout l9({{S("in1"), {}}, {S("in2"), {}}, {S("in3"), {}}},
                  {S("out1"), S("out2")});
  ASSERT_EQ(l8 * l9, l7);
  EXPECT_EQ(divideLeft(l7, l8).value(), l9);
  EXPECT_EQ(divideRight(l7, l8).value(), l9);
}

TEST_F(LinearLayoutTest, Divide_EliminateOutDim) {
  LinearLayout l1(
      {
          {S("in2"), {{1, 0}, {1, 0}}},
          {S("in1"), {{2, 0}, {0, 1}}},
      },
      {S("out1"), S("out2")});
  LinearLayout l2({{S("in2"), {{1}, {1}}}}, {S("out1")});
  LinearLayout l3({{S("in2"), {}}, {S("in1"), {{1, 0}, {0, 1}}}},
                  {S("out1"), S("out2")});
  ASSERT_EQ(l2 * l3, l1);
  EXPECT_EQ(divideLeft(l1, l2).value(), l3);

  l1 = LinearLayout(
      {
          {S("in2"), {{1, 0}, {0, 2}}},
          {S("in1"), {{0, 1}, {0, 2}}},
      },
      {S("out1"), S("out2")});
  l2 = LinearLayout({{S("in2"), {{1}}}, {S("in1"), {{1}}}}, {S("out2")});
  l3 = LinearLayout({{S("in2"), {{1, 0}}}, {S("in1"), {{0, 1}}}},
                    {S("out1"), S("out2")});
  ASSERT_EQ(l3 * l2, l1);
  EXPECT_EQ(divideRight(l1, l2).value(), l3);

  LinearLayout l4(
      {
          {S("in1"), {{0, 1}, {0, 2}}},
      },
      {S("out1"), S("out2")});
  using BasesArray =
      ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>>;
  LinearLayout l5(BasesArray{}, {S("out1")});
  LinearLayout l6({{S("in1"), {{0, 1}, {0, 2}}}}, {S("out1"), S("out2")});
  ASSERT_EQ(l5 * l6, l4);
  EXPECT_EQ(divideLeft(l4, l5).value(), l6);
  EXPECT_EQ(divideRight(l4, l5).value(), l6);
}

TEST_F(LinearLayoutTest, ColumnActionApplyLayout) {
  // Create a simple LinearLayout with one input dimension "in" and one output
  // "out". The original bases for "in" are: [{1}, {2}, {4}]. According to the
  // ColumnAction example, with action = [2, 0, 1], the new order should be:
  // [{4}, {1}, {2}].
  StringAttr inDim = S("in");
  StringAttr outDim = S("out");
  std::vector<std::vector<int32_t>> origBases = {{1}, {2}, {4}};
  LinearLayout layout({{inDim, origBases}}, {outDim});

  // Construct the ColumnAction: use action vector [2, 0, 1] with inSizeLog2
  // = 3.
  ColumnAction colAction({2, 0, 1}, inDim, 3);
  LinearLayout transformed = colAction.apply(layout);

  // Expected layout: the bases for "in" are permuted to [{4}, {1}, {2}].
  std::vector<std::vector<int32_t>> expectedBases = {{4}, {1}, {2}};
  LinearLayout expectedLayout({{inDim, expectedBases}}, {outDim});

  // Test dropping 4th basis and flipping the other two
  colAction = ColumnAction({1, 0}, inDim, 3);
  transformed = colAction.apply(layout);
  expectedLayout = LinearLayout({{inDim, {{2}, {1}}}}, {{outDim, 8}}, false);
  EXPECT_EQ(transformed, expectedLayout);
}

TEST_F(LinearLayoutTest, ColumnActionApplyValues) {
  // Test that ColumnAction correctly permutes a range of values.
  // We simulate mlir::Value objects via the opaque-pointer mechanism.
  // Create 8 dummy values corresponding to the integers 1..8.
  SmallVector<mlir::Value> values;
  for (int i = 1; i <= 8; ++i) {
    // We use getFromOpaquePointer to make a dummy value that 'carries' the
    // integer i.
    Value val = mlir::Value::getFromOpaquePointer(
        reinterpret_cast<void *>(static_cast<intptr_t>(i)));
    values.push_back(val);
  }

  // Create a ColumnAction with action = [2, 0, 1] and inSizeLog2 = 3.
  // According to the specification, this should permute the value range as:
  //   [x[0], x[4], x[1], x[5], x[2], x[6], x[3], x[7]].
  // Given our dummy values (which represent 1..8), the expected sequence is [1,
  // 5, 2, 6, 3, 7, 4, 8].
  ColumnAction colAction({2, 0, 1}, S("register"), 3);
  SmallVector<mlir::Value> permuted = colAction.apply(values);

  // Extract the integer 'identifier' from each dummy value.
  auto getId = [](mlir::Value val) -> intptr_t {
    return reinterpret_cast<intptr_t>(val.getAsOpaquePointer());
  };
  std::vector<intptr_t> result;
  for (mlir::Value v : permuted)
    result.push_back(getId(v));

  std::vector<intptr_t> expected = {1, 5, 2, 6, 3, 7, 4, 8};
  EXPECT_EQ(result, expected);

  // Test dropping the odd indices
  colAction = ColumnAction({2, 1}, S("register"), 3);
  permuted = colAction.apply(values);
  result.clear();
  for (mlir::Value v : permuted)
    result.push_back(getId(v));

  expected = std::vector<intptr_t>{1, 5, 3, 7};
  EXPECT_EQ(result, expected);
}

// The purpose of this test is to make sure the conversion of block dimension
// is identity, and this decision should be immune to block-sublayout's out-dim
// sizes.
TEST_F(LinearLayoutTest, invertAndCompose1) {
  auto regLayout = LinearLayout(
      {{S("offset"),
        {{0, 1}, {0, 2}, {0, 4}, /*gap*/ {0, 16}, {32, 0}, {64, 0}, {128, 0}}},

       {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
       {S("warp"), {{0, 0}, {16, 0}}},

       {S("block"), {{0, 0}}}},
      {S("dim0"), S("dim1")});

  auto sharedLayout = LinearLayout({{S("offset"),
                                     {{0, 1},
                                      {0, 2},
                                      {0, 4},
                                      {0, 8},
                                      {0, 16},
                                      {0, 32},
                                      {0, 64},
                                      {1, 0},
                                      {2, 0},
                                      {4, 0},
                                      {8, 0},
                                      {16, 0},
                                      {32, 0},
                                      {64, 0},
                                      {128, 0}}},
                                    {S("block"), {{0, 0}}}},
                                   {S("dim0"), S("dim1")});

  auto cvt = regLayout.invertAndCompose(sharedLayout);

  EXPECT_TRUE(cvt.isTrivialOver(S("block")));
}

class MixedRadixTest : public LinearLayoutTest {
protected:
  // register = 3 * d1 + d0 with d0 < 3 (zero basis) and d1 < 2; lane < 2.
  LinearLayout mixed(ArrayRef<int32_t> regRadices = {3, 2}) {
    BasesT bases;
    bases[S("register")] = {{0}, {1}};
    bases[S("lane")] = {{2}};
    LinearLayout::RadicesT radices;
    radices[S("register")] = regRadices.vec();
    radices[S("lane")] = {2};
    return LinearLayout(bases, radices, {{S("dim0"), 4}}, true);
  }
};

TEST_F(MixedRadixTest, TryCreateRejectsInvalidRadices) {
  auto tryCreate = [&](BasesT bases, LinearLayout::RadicesT radices) {
    return LinearLayout::tryCreate(std::move(bases), std::move(radices),
                                   {{S("dim0"), 2}}, false);
  };
  BasesT reg;
  reg[S("register")] = {{0}, {1}};
  LinearLayout::RadicesT radices;
  radices[S("register")] = {3};
  EXPECT_FALSE(tryCreate(reg, radices)); // one radix for two bases
  radices[S("register")] = {1, 2};
  EXPECT_FALSE(tryCreate(reg, radices));
  radices[S("register")] = {2, 3};
  EXPECT_FALSE(tryCreate(reg, radices)); // radix 3 with basis 1
  BasesT big;
  big[S("register")].assign(20, {0});
  LinearLayout::RadicesT bigRadices;
  bigRadices[S("register")].assign(20, 3);
  EXPECT_FALSE(tryCreate(big, bigRadices)); // 3^20 > INT32_MAX
  // Controls: the same digits with valid radices, and a radix-2 layout above
  // INT32_MAX.
  radices[S("register")] = {3, 2};
  EXPECT_TRUE(tryCreate(reg, radices));
  big[S("register")].assign(31, {0});
  EXPECT_TRUE(tryCreate(big, {}));
}

TEST_F(MixedRadixTest, Sizes) {
  LinearLayout ll = mixed();
  EXPECT_FALSE(ll.isRadix2());
  EXPECT_EQ(ll.getInDimSize(S("register")), 6);
  EXPECT_EQ(ll.getTotalInDimSize(), 12);
  EXPECT_TRUE(ll.isSurjective());
  EXPECT_FALSE(ll.isInjective());
  EXPECT_EQ(mixed({2, 2}).getInDimSize(S("register")), 4);
}

TEST_F(MixedRadixTest, Apply) {
  LinearLayout ll = mixed();
  for (int32_t reg = 0; reg < 6; ++reg) {
    EXPECT_EQ(
        ll.apply({{S("register"), reg}, {S("lane"), 1}}),
        (SmallVector<std::pair<StringAttr, int32_t>>{{S("dim0"), 2 + reg / 3}}))
        << reg;
  }
  // Radix-2 control: register 2 sets d1.
  EXPECT_EQ(mixed({2, 2}).apply({{S("register"), 2}, {S("lane"), 0}}),
            (SmallVector<std::pair<StringAttr, int32_t>>{{S("dim0"), 1}}));
}

TEST_F(MixedRadixTest, PreconditionAsserts) {
#ifdef NDEBUG
  SUCCEED();
#else
  LinearLayout ll = mixed(), pow2 = mixed({2, 2});
  ColumnAction action({0, 1}, S("register"), 2, {3, 2});
  // Radix-2-only operations.
  EXPECT_DEATH((void)ll.getFreeVariableMasks(), "isRadix2");
  EXPECT_DEATH(
      (void)ll.compose(LinearLayout::identity1D(4, S("dim0"), S("out"))),
      "isRadix2");
  EXPECT_DEATH((void)ll.invertAndCompose(ll), "isRadix2");
  EXPECT_DEATH((void)lstsq(ll, ll), "isRadix2");
  EXPECT_DEATH((void)divideLeft(ll, ll), "isRadix2");
  EXPECT_DEATH((void)divideRight(ll, ll), "isRadix2");
  EXPECT_DEATH((void)ColumnAction::identity(S("register"), 2).apply(ll),
               "isRadix2");
  EXPECT_DEATH((void)action.apply(pow2), "isRadix2");
  EXPECT_DEATH((void)action.leftCompose(action), "isRadix2");
  // concatOuts needs equal radices, reshapeIns whole digits, and
  // ColumnAction::apply one value per index.
  BasesT zeros;
  zeros[S("register")] = {{0}, {0}};
  zeros[S("lane")] = {{0}};
  LinearLayout::RadicesT swapped;
  swapped[S("register")] = {2, 3};
  swapped[S("lane")] = {2};
  EXPECT_DEATH((void)ll.concatOuts(
                   LinearLayout(zeros, swapped, {{S("dim1"), 1}}, false)),
               "same input radices");
  LinearLayout flat = ll.flattenIns();
  EXPECT_DEATH((void)flat.reshapeIns({{S("register"), 2}, {S("lane"), 6}}),
               "end on a digit");
  EXPECT_DEATH((void)flat.reshapeIns({{S("register"), 6}}), "take every digit");
  SmallVector<Value> four(4);
  EXPECT_DEATH((void)action.apply(four), "different size");
#endif
}

TEST_F(MixedRadixTest, EqualityAndHashSeeRadices) {
  EXPECT_EQ(mixed(), mixed());
  EXPECT_EQ(hash_value(mixed()), hash_value(mixed()));
  EXPECT_NE(mixed(), mixed({2, 2}));
  EXPECT_NE(hash_value(mixed()), hash_value(mixed({2, 2})));
}

TEST_F(MixedRadixTest, PrintShowsRadicesAndWeights) {
  EXPECT_THAT(mixed().toString(),
              ::testing::HasSubstr("register=1 (radix 3) -> (0)\n"
                                   "   register=3 -> (1)\n"));
}

TEST_F(MixedRadixTest, ReshapeInsCutsAtDigitBoundaries) {
  LinearLayout flat = mixed().flattenIns();
  EXPECT_THAT(flat.getInputRadices(S("register")), ElementsAre(3, 2, 2));
  EXPECT_EQ(flat.getInDimSize(S("register")), 12);
  EXPECT_EQ(flat.reshapeIns({{S("register"), 6}, {S("lane"), 2}}), mixed());
}

TEST_F(MixedRadixTest, MixedRadixConcatenatesDigits) {
  auto reg2 = LinearLayout::identity1D(2, S("register"), S("dim1"));
  EXPECT_THAT((mixed() * reg2).getInputRadices(S("register")),
              ElementsAre(3, 2, 2));
  EXPECT_THAT((reg2 * mixed()).getInputRadices(S("register")),
              ElementsAre(2, 3, 2));
}

TEST_F(MixedRadixTest, StructuralOpsKeepRadices) {
  LinearLayout ll = mixed() * LinearLayout::identity1D(2, S("lane"), S("dim1"));
  auto keeps = [&](const LinearLayout &other) {
    return llvm::equal(other.getInputRadices(), ll.getInputRadices());
  };
  EXPECT_EQ(
      ll.sublayout({S("register")}, {S("dim0")}).getInputRadices(S("register")),
      ll.getInputRadices(S("register")));
  EXPECT_TRUE(keeps(ll.transposeOuts({S("dim1"), S("dim0")})));
  EXPECT_TRUE(keeps(ll.transposeIns({S("lane"), S("register")})
                        .transposeIns({S("register"), S("lane")})));
  EXPECT_TRUE(
      keeps(ll.sublayout({S("register")}, {S("dim0"), S("dim1")})
                .concatIns(ll.sublayout({S("lane")}, {S("dim0"), S("dim1")}))));
  BasesT zeros;
  zeros[S("register")] = {{0}, {0}};
  zeros[S("lane")] = {{0}, {0}};
  EXPECT_TRUE(keeps(ll.concatOuts(
      LinearLayout(zeros, ll.getInputRadices(), {{S("dim2"), 1}}, false))));
  EXPECT_TRUE(keeps(ll.resizeOutDim(S("dim0"), 2)));
  EXPECT_TRUE(keeps(ll.reshapeOuts({{S("flat"), 8}})));
}

TEST_F(MixedRadixTest, RemoveZeroBasesDropsTheirRadices) {
  LinearLayout ll = mixed().removeZeroBasesAlongDim(S("register"));
  EXPECT_THAT(ll.getInputRadices(S("register")), ElementsAre(2));
  EXPECT_TRUE(ll.isRadix2());
}

TEST_F(MixedRadixTest, ColumnActionApplyValues) {
  SmallVector<Value> values;
  for (intptr_t i = 0; i < 6; ++i)
    values.push_back(Value::getFromOpaquePointer(reinterpret_cast<void *>(i)));
  auto ids = [](ArrayRef<Value> vals) {
    return llvm::map_to_vector(vals, [](Value v) {
      return reinterpret_cast<intptr_t>(v.getAsOpaquePointer());
    });
  };
  // Output digit 0 is input digit 1 (radix 2), output digit 1 is input digit
  // 0 (radix 3).
  EXPECT_THAT(ids(ColumnAction({1, 0}, S("register"), 2, {3, 2}).apply(values)),
              ElementsAre(0, 3, 1, 4, 2, 5));
  EXPECT_THAT(ids(ColumnAction({1}, S("register"), 2, {3, 2}).apply(values)),
              ElementsAre(0, 3));
  ColumnAction swap({1, 0}, S("register"), 2, {3, 2});
  EXPECT_THAT(ids(swap.inverse().apply(swap.apply(values))),
              ElementsAre(0, 1, 2, 3, 4, 5));
}

class MixedRadixLayoutTest : public LinearLayoutTest {
protected:
  LinearLayout mixedRadix(std::vector<std::vector<int32_t>> regBases,
                          std::vector<int32_t> regRadices,
                          ArrayRef<std::pair<StringAttr, int32_t>> outDims) {
    BasesT bases;
    bases[S("register")] = std::move(regBases);
    LinearLayout::RadicesT radices;
    radices[S("register")] = std::move(regRadices);
    return LinearLayout(std::move(bases), std::move(radices), outDims,
                        /*requireSurjective=*/false);
  }

  // in -> out identity of a non-power-of-two size: binary digits, then one odd
  // digit.
  LinearLayout exactIdentity1D(int32_t size, StringAttr in, StringAttr out) {
    int32_t twoPart = LinearLayout::twoPartOf(size);
    BasesT bases;
    LinearLayout::RadicesT radices;
    for (int32_t bit = 1; bit < twoPart; bit *= 2) {
      bases[in].push_back({bit});
      radices[in].push_back(2);
    }
    bases[in].push_back({twoPart});
    radices[in].push_back(size / twoPart);
    return LinearLayout(std::move(bases), std::move(radices), {{out, size}},
                        /*requireSurjective=*/true);
  }
};

TEST_F(MixedRadixLayoutTest, InvariantsReject) {
  auto build = [&](std::vector<std::vector<int32_t>> regBases,
                   std::vector<int32_t> regRadices,
                   ArrayRef<std::pair<StringAttr, int32_t>> outDims) {
    (void)mixedRadix(std::move(regBases), std::move(regRadices), outDims);
  };
  EXPECT_DEATH(build({}, {}, {{S("x"), 0}}), "Out-dim sizes must be positive");
  EXPECT_DEATH(build({{3}}, {2}, {{S("x"), 6}}),
               "invalid radix-2 digit: it has an odd mixed-radix");
  EXPECT_DEATH(build({{3}}, {3}, {{S("x"), 6}}),
               "invalid non-binary digit: it has a binary mixed-radix");
  EXPECT_DEATH(build({{2, 1}}, {3}, {{S("x"), 6}, {S("y"), 2}}),
               "invalid input digit of radix 3 in in-dim 'register': its "
               "basis must be zero on Binary outputs");
  const char *unsupported = "NPOT layout not yet supported: each odd part "
                            "needs one non-binary digit of that radix";
  EXPECT_DEATH(build({{2, 2}}, {3}, {{S("x"), 6}, {S("y"), 6}}), unsupported);
  EXPECT_DEATH(build({{2}, {4}}, {3, 3}, {{S("x"), 6}}), unsupported);
  EXPECT_DEATH(build({{2}}, {5}, {{S("x"), 6}}), unsupported);
  EXPECT_DEATH(build({{6}}, {9}, {{S("x"), 18}}), unsupported);
}

TEST_F(MixedRadixLayoutTest, OnlyRadicesAllowMixedRadixSizes) {
  BasesT bases;
  bases[S("register")] = {{1}};
  LinearLayout::RadicesT radices;
  radices[S("register")] = {2};
  EXPECT_FALSE(LinearLayout::tryCreate(bases, {{S("x"), 6}}, false));
  EXPECT_TRUE(LinearLayout::tryCreate(bases, radices, {{S("x"), 6}}, false));
  EXPECT_TRUE(LinearLayout::tryCreate({}, {}, {{S("x"), 6}}, false));
  EXPECT_DEATH((void)LinearLayout(bases, {{S("x"), 6}}, false),
               "Out-dim sizes must be powers of 2");
}

TEST_F(MixedRadixLayoutTest, SublayoutWithoutInDimsKeepsMixedRadixSize) {
  // No in-dims leave an empty radices map, which still allows the size.
  LinearLayout sub =
      mixedRadix({{1}}, {2}, {{S("dim"), 6}}).sublayout({}, {S("dim")});
  EXPECT_EQ(sub.getOutDimSize(S("dim")), 6);
}

TEST_F(MixedRadixLayoutTest, RankCountsOwnedOddCoordinates) {
  // Odd part 5: the codomain rank is countr_zero(20) + 1, below Log2(20).
  LinearLayout twenty = exactIdentity1D(20, S("register"), S("dim"));
  EXPECT_TRUE(twenty.isSurjective());
  EXPECT_TRUE(twenty.isInjective());
  EXPECT_EQ(twenty.getTotalOutDimSize(), 20);
  LinearLayout mixedRadixBit = mixedRadix({{1}}, {2}, {{S("dim"), 6}});
  EXPECT_FALSE(mixedRadixBit.isSurjective());
  // A mixed-radix output alone keeps a layout off the radix-2 algorithms.
  EXPECT_FALSE(mixedRadixBit.isRadix2());
  // A zero radix-2 digit after the odd digit: registers 0 and 6 collide.
  EXPECT_FALSE(
      mixedRadix({{1}, {2}, {0}}, {2, 3, 2}, {{S("dim"), 6}}).isInjective());
  // A broadcast non-binary digit adds no rank.
  EXPECT_TRUE(
      mixedRadix({{1}, {2}, {0}}, {2, 3, 3}, {{S("dim"), 6}}).isSurjective());
  // Odd coefficient 4: basis 16 is above the Log2(20) binary rows.
  EXPECT_TRUE(
      mixedRadix({{1}, {2}, {16}}, {2, 2, 5}, {{S("dim"), 20}}).isSurjective());
}

TEST_F(MixedRadixLayoutTest, EqualIgnoringOutDimSizesComparesTwoParts) {
  // Basis 4 is odd coordinate 2 at size 6 and odd coordinate 1 at size 12.
  EXPECT_FALSE(mixedRadix({{1}, {4}}, {2, 3}, {{S("dim"), 6}})
                   .equalIgnoringOutDimSizes(
                       mixedRadix({{1}, {4}}, {2, 3}, {{S("dim"), 12}})));
  EXPECT_TRUE(
      mixedRadix({{1}}, {2}, {{S("dim"), 6}})
          .equalIgnoringOutDimSizes(mixedRadix({{1}}, {2}, {{S("dim"), 10}})));
}

TEST_F(MixedRadixLayoutTest, ReshapeInsAndRemoveZeroBasesKeepMixedRadix) {
  BasesT bases;
  LinearLayout::RadicesT radices;
  bases[S("register")] = {{1, 0}, {2, 0}};
  radices[S("register")] = {2, 3};
  bases[S("lane")] = {{0, 1}};
  radices[S("lane")] = {2};
  LinearLayout ll(bases, radices, {{S("x"), 6}, {S("y"), 2}},
                  /*requireSurjective=*/true);
  EXPECT_EQ(ll.flattenIns().reshapeIns({{S("register"), 6}, {S("lane"), 2}}),
            ll);
  bases[S("register")].push_back({0, 0});
  radices[S("register")].push_back(2);
  EXPECT_EQ(LinearLayout(bases, radices, ll.getOutDims(),
                         /*requireSurjective=*/false)
                .removeZeroBasesAlongDim(S("register")),
            ll);
}

TEST_F(MixedRadixLayoutTest, FreeVariableMasksFreeNonOwningDigits) {
  // Digit 1 repeats digit 0's binary part and digit 3 is a zero odd digit;
  // digit 2 owns the odd coordinates.
  EXPECT_EQ(mixedRadix({{1}, {1}, {2}, {0}}, {2, 2, 3, 3}, {{S("dim"), 6}})
                .getFreeVariableMasks()
                .lookup(S("register")),
            0b1010);
}

TEST_F(MixedRadixLayoutTest, MultiplyPlacesOddPartOnTop) {
  auto kReg = S("register"), kOut = S("out");
  // Inner's odd digit moves above outer's binary digits.
  EXPECT_EQ(exactIdentity1D(6, kReg, kOut) *
                LinearLayout::identity1D(4, kReg, kOut),
            mixedRadix({{1}, {8}, {2}, {4}}, {2, 3, 2, 2}, {{kOut, 24}}));
  EXPECT_EQ(LinearLayout::identity1D(2, kReg, kOut) *
                exactIdentity1D(6, kReg, kOut),
            exactIdentity1D(12, kReg, kOut));
}

TEST_F(MixedRadixLayoutTest, MultiplyRejectsTwoMixedRadixFactorsOnOneOutput) {
  auto kReg = S("register"), kOut = S("out");
  EXPECT_DEATH(
      (void)(exactIdentity1D(3, kReg, kOut) * exactIdentity1D(5, kReg, kOut)),
      "each odd part needs one non-binary digit");
}

TEST_F(MixedRadixLayoutTest, Apply) {
  LinearLayout six = exactIdentity1D(6, S("register"), S("dim"));
  for (int32_t i = 0; i < 24; ++i)
    EXPECT_EQ(six.apply({{S("register"), i}}).front().second, i % 6) << i;
  // A Binary output beside a mixed-radix output keeps XOR arithmetic.
  LinearLayout twoDims =
      six * LinearLayout::identity1D(4, S("lane"), S("other"));
  for (int32_t lane = 0; lane < 4; ++lane)
    EXPECT_THAT(twoDims.apply({{S("register"), 5}, {S("lane"), lane}}),
                ElementsAre(Pair(S("dim"), 5), Pair(S("other"), lane)));
  // Odd coefficient 2 is a unit mod 3: digit 2 wraps to odd coordinate 1.
  EXPECT_EQ(mixedRadix({{1}, {4}}, {2, 3}, {{S("dim"), 6}})
                .apply({{S("register"), 4}})
                .front()
                .second,
            2);
}

// fp16 [rows, *] tile in swizzled panels of swizzleBytes / 2 columns, panel
// after panel.
int32_t nvmmaOffset(int32_t row, int32_t col, int32_t swizzleBytes,
                    int32_t rows) {
  int32_t width = swizzleBytes / 2, c = col % width;
  int32_t phase = row / (128 / swizzleBytes) % (width / 8);
  return col / width * rows * width + row * width + (c / 8 ^ phase) * 8 + c % 8;
}

class OffsetTableTest : public MixedRadixLayoutTest {
protected:
  using Point = SmallVector<std::pair<StringAttr, int32_t>>;

  void forEachPoint(const LinearLayout &layout,
                    function_ref<void(const Point &)> fn) {
    Point point;
    SmallVector<int32_t> sizes;
    for (StringAttr inDim : layout.getInDimNames()) {
      point.push_back({inDim, 0});
      sizes.push_back(layout.getInDimSize(inDim));
    }
    for (size_t d = 0; d < point.size();) {
      fn(point);
      for (d = 0; d < point.size() && ++point[d].second == sizes[d]; ++d)
        point[d].second = 0;
    }
  }

  // (dim0 = col, dim1 = row) -> offset of the fp16 [96, 128] tile in two
  // 128B-swizzled panels; the panel digit p has basis `panelBasis`.
  std::optional<LinearLayout>
  denseShared(int32_t panelBasis,
              SmallVector<std::pair<StringAttr, int32_t>> digits,
              std::vector<int32_t> offsets, std::string *diagnostic = nullptr) {
    BasesT bases;
    LinearLayout::RadicesT radices;
    for (int32_t i = 0; i < 6; ++i)
      bases[S("dim0")].push_back({1 << i});
    bases[S("dim0")].push_back({panelBasis});
    radices[S("dim0")] = std::vector<int32_t>(7, 2);
    for (int32_t i = 0; i < 5; ++i)
      bases[S("dim1")].push_back({nvmmaOffset(1 << i, 0, 128, 96)});
    bases[S("dim1")].push_back({0});
    radices[S("dim1")] = {2, 2, 2, 2, 2, 3};
    llvm::MapVector<StringAttr, SmallVector<std::pair<StringAttr, int32_t>>>
        tableDigits;
    tableDigits[S("offset")] = std::move(digits);
    llvm::MapVector<StringAttr, std::vector<int32_t>> tableOffsets;
    tableOffsets[S("offset")] = std::move(offsets);
    return LinearLayout::tryCreate(bases, radices, {{S("offset"), 96 * 128}},
                                   tableDigits, tableOffsets,
                                   /*requireSurjective=*/true, diagnostic);
  }
  std::optional<LinearLayout> denseShared() {
    return denseShared(0, {{S("dim0"), 6}, {S("dim1"), 5}},
                       {0, 6144, 2048, 8192, 4096, 10240});
  }

  // fp16 [96, cols] over 4 warps: the col 64 digit (radix cols / 64), row 16
  // and the odd row digit are in registers.
  LinearLayout tile96(int32_t cols) {
    BasesT bases;
    LinearLayout::RadicesT radices;
    bases[S("register")] = {{1, 0}, {2, 0}, {4, 0}, {0, 16}, {64, 0}, {0, 32}};
    radices[S("register")] = {2, 2, 2, 2, cols / 64, 3};
    bases[S("lane")] = {{8, 0}, {16, 0}, {32, 0}, {0, 1}, {0, 2}};
    radices[S("lane")] = std::vector<int32_t>(5, 2);
    bases[S("warp")] = {{0, 4}, {0, 8}};
    radices[S("warp")] = {2, 2};
    return LinearLayout(bases, radices, {{S("dim0"), cols}, {S("dim1"), 96}},
                        /*requireSurjective=*/true);
  }

  // x -> f(x) with x digit 1 unused. Over odd radices {3, 5} the table has
  // two odd factors, so it is not in odd-part form; over {3} it is.
  static int32_t f(int32_t x) { return x % 2 + 2 * (x / 4 % 2) + 4 * (x / 8); }
  LinearLayout oddTable(ArrayRef<int32_t> odd, StringRef in = "x",
                        StringRef out = "offset") {
    BasesT bases;
    bases[S(in)] = {{1}, {0}, {2}};
    LinearLayout::RadicesT radices;
    radices[S(in)] = {2, 2, 2};
    llvm::MapVector<StringAttr, SmallVector<std::pair<StringAttr, int32_t>>>
        digits;
    llvm::MapVector<StringAttr, std::vector<int32_t>> offsets;
    int32_t oddSize = 1;
    for (int32_t radix : odd) {
      digits[S(out)].push_back({S(in), int32_t(bases[S(in)].size())});
      bases[S(in)].push_back({4 * oddSize});
      radices[S(in)].push_back(radix);
      oddSize *= radix;
    }
    for (int32_t t = 0; t < oddSize; ++t)
      offsets[S(out)].push_back(4 * t);
    return *LinearLayout::tryCreate(bases, radices, {{S(out), 4 * oddSize}},
                                    digits, offsets,
                                    /*requireSurjective=*/false);
  }
  // The general layout first, then the odd-part control.
  SmallVector<SmallVector<int32_t>> oddRadices() { return {{3, 5}, {3}}; }

  // Expects result(x) == f(old(x)) on every input of the 1D result.
  void expectOld(const LinearLayout &result,
                 function_ref<int32_t(int32_t)> old) {
    forEachPoint(result, [&](const Point &x) {
      EXPECT_EQ(result.apply(x)[0].second, f(old(x[0].second))) << x[0].second;
    });
  }
};

TEST_F(OffsetTableTest, ProductOfGeneralTable) {
  for (const auto &odd : oddRadices()) {
    LinearLayout inner = oddTable(odd);
    EXPECT_EQ(inner.hasOddPartForm(), odd.size() == 1);
    LinearLayout outer = LinearLayout::identity1D(2, S("y"), S("offset"));
    LinearLayout product = inner * outer;
    forEachPoint(product, [&](const Point &x) {
      int32_t v = f(x[0].second);
      EXPECT_EQ(product.apply(x)[0].second,
                v % 4 + 4 * (x[1].second + 2 * (v / 4)));
    });
  }
}

TEST_F(OffsetTableTest, ReshapeOutsOfGeneralTable) {
  for (const auto &odd : oddRadices()) {
    LinearLayout layout = oddTable(odd);
    int32_t oddSize = layout.getOutDimSize(S("offset")) / 4;
    LinearLayout split = layout.reshapeOuts({{S("lo"), 4}, {S("hi"), oddSize}});
    forEachPoint(split, [&](const Point &x) {
      int32_t v = f(x[0].second);
      EXPECT_EQ(split.apply(x), (Point{{S("lo"), v % 4}, {S("hi"), v / 4}}));
    });
    EXPECT_EQ(split.reshapeOuts({{S("offset"), 4 * oddSize}}), layout);
  }
}

TEST_F(OffsetTableTest, ConcatInsOfGeneralTable) {
  for (const auto &odd : oddRadices()) {
    LinearLayout lhs = oddTable(odd);
    BasesT bases;
    bases[S("y")] = {{1}, {2}};
    LinearLayout::RadicesT radices;
    radices[S("y")] = {2, 2};
    LinearLayout rhs(bases, radices, lhs.getOutDims(), false);
    LinearLayout both = lhs.concatIns(rhs);
    forEachPoint(both, [&](const Point &x) {
      EXPECT_EQ(both.apply(x)[0].second, f(x[0].second) ^ x[1].second);
    });
  }
}

TEST_F(OffsetTableTest, ConcatOutsOfGeneralTable) {
  for (const auto &odd : oddRadices()) {
    LinearLayout lhs = oddTable(odd);
    BasesT bases = lhs.getBases();
    for (auto &basis : bases[S("x")])
      basis[0] = basis[0] < 4 ? basis[0] : 0;
    LinearLayout rhs(bases, lhs.getInputRadices(), {{S("low"), 4}}, false);
    LinearLayout both = lhs.concatOuts(rhs);
    forEachPoint(both, [&](const Point &x) {
      int32_t v = f(x[0].second);
      EXPECT_EQ(both.apply(x), (Point{{S("offset"), v}, {S("low"), v % 4}}));
    });
  }
}

TEST_F(OffsetTableTest, RenameGeneralTable) {
  for (const auto &odd : oddRadices()) {
    LinearLayout renamed = renameLinearLayoutDims(
        oddTable(odd), {{S("x"), S("y")}}, {{S("offset"), S("out")}});
    EXPECT_EQ(renamed, oddTable(odd, "y", "out"));
    expectOld(renamed, [](int32_t x) { return x; });
  }
}

TEST_F(OffsetTableTest, RemoveZeroBasesOfGeneralTable) {
  for (const auto &odd : oddRadices()) {
    LinearLayout stripped = oddTable(odd).removeZeroBasesAlongDim(S("x"));
    EXPECT_EQ(stripped.getInDimSizeLog2(S("x")), int32_t(odd.size()) + 2);
    expectOld(stripped, [](int32_t x) { return x % 2 + 4 * (x / 2); });
  }
}

TEST_F(OffsetTableTest, ColumnActionOnGeneralTable) {
  for (const auto &odd : oddRadices()) {
    LinearLayout layout = oddTable(odd);
    auto radices = layout.getInputRadices(S("x")).vec();
    // Odd digit 3 first and digit 2 last; digit 1 is dropped.
    SmallVector<size_t> action = {3, 0};
    for (size_t d = 4; d < radices.size(); ++d)
      action.push_back(d);
    action.push_back(2);
    LinearLayout result =
        ColumnAction(action, S("x"), radices.size(), radices).apply(layout);
    expectOld(result, [&](int32_t x) {
      SmallVector<int32_t> digits(radices.size(), 0);
      for (size_t a : action) {
        digits[a] = x % radices[a];
        x /= radices[a];
      }
      int32_t old = 0;
      for (size_t d = radices.size(); d-- > 0;)
        old = old * radices[d] + digits[d];
      return old;
    });
  }
}

TEST_F(OffsetTableTest, ResizeInDimOfGeneralTable) {
  for (const auto &odd : oddRadices()) {
    LinearLayout layout = oddTable(odd);
    int32_t size = layout.getInDimSize(S("x"));
    expectOld(layout.resizeInDim(S("x"), 24), [](int32_t x) { return x; });
    expectOld(layout.resizeInDim(S("x"), 2 * size),
              [&](int32_t x) { return x % size; });
  }
}

TEST_F(OffsetTableTest, DenseTwoPanelTableIsCanonical) {
  auto shared = denseShared();
  ASSERT_TRUE(shared);
  forEachPoint(*shared, [&](const Point &x) {
    EXPECT_EQ(shared->apply(x)[0].second,
              nvmmaOffset(x[1].second, x[0].second, 128, 96));
  });
  // XORing the panel's 6144 is the same function, but binary bases must stay
  // below the table's 2048s, so p joins the table: 2048 * (q + 3p).
  auto binaryPanel = denseShared(6144, {{S("dim1"), 5}}, {0, 2048, 4096});
  ASSERT_TRUE(binaryPanel);
  EXPECT_EQ(*binaryPanel, *shared);
  EXPECT_EQ(hash_value(*binaryPanel), hash_value(*shared));
  EXPECT_THAT(binaryPanel->getTableDigits(S("offset")),
              ElementsAre(Pair(S("dim0"), 6), Pair(S("dim1"), 5)));
  EXPECT_THAT(binaryPanel->getTableOffsets(S("offset")),
              ElementsAre(0, 6144, 2048, 8192, 4096, 10240));
}

TEST_F(OffsetTableTest, ComposeAndInvertPromoteDigits) {
  auto shared = denseShared();
  ASSERT_TRUE(shared);
  LinearLayout tile = tile96(128);
  // The register panel digit is promoted into the composed table.
  LinearLayout toShared = tile.compose(*shared);
  EXPECT_THAT(toShared.getTableDigits(S("offset")),
              ElementsAre(Pair(S("register"), 4), Pair(S("register"), 5)));
  forEachPoint(tile, [&](const Point &x) {
    auto coords = tile.apply(x);
    EXPECT_EQ(toShared.apply(x)[0].second,
              nvmmaOffset(coords[1].second, coords[0].second, 128, 96));
  });
  EXPECT_EQ(toShared.invertAndCompose(*shared), tile);
  LinearLayout identity = shared->invertAndCompose(*shared);
  forEachPoint(identity,
               [&](const Point &x) { EXPECT_EQ(identity.apply(x), x); });
  ScopedAllowNpot npot;
  EXPECT_EQ(LinearLayout::identity1D(96, S("in"), S("out")).invert(),
            LinearLayout::identity1D(96, S("out"), S("in")));
}

TEST_F(OffsetTableTest, ResizeNpotOutDim) {
  LinearLayout tile = tile96(128);
  EXPECT_EQ(tile.resizeOutDim(S("dim1"), 96), tile);
  LinearLayout rows = tile.resizeOutDim(S("dim1"), 1);
  forEachPoint(tile, [&](const Point &x) {
    EXPECT_EQ(rows.apply(x),
              (Point{{S("dim0"), tile.apply(x)[0].second}, {S("dim1"), 0}}));
  });
  auto shared = denseShared();
  ASSERT_TRUE(shared);
  LinearLayout flat = shared->resizeOutDim(S("offset"), 1);
  forEachPoint(*shared, [&](const Point &x) {
    EXPECT_EQ(flat.apply(x), (Point{{S("offset"), 0}}));
  });
}

TEST_F(OffsetTableTest, MalformedTableIsRejected) {
  std::string diagnostic;
  EXPECT_FALSE(denseShared(0, {{S("dim0"), 6}, {S("dim1"), 5}},
                           {0, 6144, 2048, 8192, 4096, 12288}, &diagnostic));
  EXPECT_THAT(diagnostic, ::testing::HasSubstr("reaches 12288"));
}

TEST_F(OffsetTableTest, PaddingTable) {
  // 4 elements of padding after every 64: P(off) = off + 4 * (off / 64).
  auto pad = [](int32_t off) { return off + 4 * (off / 64); };
  BasesT bases;
  bases[S("offset")] = std::vector<std::vector<int32_t>>(13, {0});
  LinearLayout::RadicesT radices;
  radices[S("offset")] = std::vector<int32_t>(13, 2);
  llvm::MapVector<StringAttr, SmallVector<std::pair<StringAttr, int32_t>>>
      digits;
  llvm::MapVector<StringAttr, std::vector<int32_t>> offsets;
  for (int32_t i = 0; i < 13; ++i)
    digits[S("offset")].push_back({S("offset"), i});
  for (int32_t off = 0; off < 8192; ++off)
    offsets[S("offset")].push_back(pad(off));
  auto padded = LinearLayout::tryCreate(bases, radices, {{S("offset"), 8704}},
                                        digits, offsets,
                                        /*requireSurjective=*/false);
  ASSERT_TRUE(padded);
  ArrayRef<int32_t> table = padded->getTableOffsets(S("offset"));
  EXPECT_THAT(padded->getTableDigits(S("offset")).front(),
              Pair(S("offset"), 6));
  ASSERT_EQ(table.size(), 128u);
  for (int32_t hi = 0; hi < 128; ++hi)
    EXPECT_EQ(table[hi], 68 * hi);
  for (int32_t off = 0; off < 8192; ++off)
    EXPECT_EQ(padded->apply({{S("offset"), off}})[0].second, pad(off));

  // After the inverse of a radix-2 [64, 128] shared layout.
  BasesT coordBases;
  for (int32_t i = 0; i < 7; ++i)
    coordBases[S("dim0")].push_back({nvmmaOffset(0, 1 << i, 128, 64)});
  for (int32_t i = 0; i < 6; ++i)
    coordBases[S("dim1")].push_back({nvmmaOffset(1 << i, 0, 128, 64)});
  LinearLayout shared =
      LinearLayout(coordBases, {{S("offset"), 8192}}, true).invert();
  LinearLayout toPadded = shared.invert().compose(*padded);
  forEachPoint(toPadded, [&](const Point &x) {
    EXPECT_EQ(toPadded.apply(x)[0].second,
              pad(nvmmaOffset(x[1].second, x[0].second, 128, 64)));
  });
}

} // anonymous namespace
} // namespace mlir::triton

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
