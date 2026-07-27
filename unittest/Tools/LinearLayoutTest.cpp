#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/LayoutUtils.h"

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

// Test invertAndCompose with non-power-of-2 dimensions
// Exercises the modular CRT-based least-squares solver path
TEST_F(LinearLayoutTest, InvertAndCompose_NonPow2) {
  // Create a simple layout with dimension 3 (non-pow2)
  // Layout A: maps in1 -> out (dimension 3)
  // Bases: 1, 1 (mod 3: 0->0, 1->1, 2->1, 3->2)
  LinearLayout A({{S("in1"), {{1}, {1}}}}, {{S("out"), 3}},
                 /*requireSurjective=*/false);

  // Layout B: maps in2 -> out (dimension 3)
  // Bases: 2, 1 (mod 3: 0->0, 1->2, 2->1, 3->0)
  LinearLayout B({{S("in2"), {{2}, {1}}}}, {{S("out"), 3}},
                 /*requireSurjective=*/false);

  // Compute X = A.invertAndCompose(B)
  // X should map in1's inputs to in2's inputs such that X ∘ B = A
  LinearLayout X = A.invertAndCompose(B);

  // Verify the composition property: X ∘ B = A
  EXPECT_EQ(X.compose(B), A);

  // The result should have in1 as inputs and in2 as outputs
  EXPECT_THAT(to_vector(X.getInDimNames()), ElementsAre(S("in1")));
  EXPECT_THAT(to_vector(X.getOutDimNames()), ElementsAre(S("in2")));
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

  // NPOT modular layouts: clamp to largest pow2 dividing N.
  EXPECT_EQ(2, LinearLayout::modularIdentity1D(6, S("in"), S("out"))
                   .getNumConsecutiveInOut());
  EXPECT_EQ(1, LinearLayout::modularIdentity1D(5, S("in"), S("out"))
                   .getNumConsecutiveInOut());
  EXPECT_EQ(4, LinearLayout::modularIdentity1D(12, S("in"), S("out"))
                   .getNumConsecutiveInOut());
  EXPECT_EQ(1, LinearLayout::modularIdentity1D(15, S("in"), S("out"))
                   .getNumConsecutiveInOut());
  // Multi-dim: NPOT first out-dim gets clamped.
  EXPECT_EQ(2, (LinearLayout::modularIdentity1D(6, S("in1"), S("out1")) *
                LinearLayout::identity1D(8, S("in2"), S("out2")))
                   .getNumConsecutiveInOut());
  // Mixed: pow2 first out-dim, NPOT second — no clamp on pow2.
  EXPECT_EQ(4, (LinearLayout::identity1D(4, S("in1"), S("out1")) *
                LinearLayout::modularIdentity1D(6, S("in2"), S("out2")))
                   .getNumConsecutiveInOut());
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

  // Modular layout: no zero bases → no free variables.
  EXPECT_EQ(AR(to_vector(LinearLayout::modularIdentity1D(6, S("in"), S("out"))
                             .getFreeVariableMasks())),
            AR({{S("in"), 0}}));

  // Modular layout: stride=3, size=6 → bases [3, 0, 0] → bits 1,2 are free.
  EXPECT_EQ(AR(to_vector(LinearLayout::modularStrided1D(6, 3, S("in"), S("out"))
                             .getFreeVariableMasks())),
            AR({{S("in"), 0b110}}));
}

// The pow2-only fast paths in isExpensiveView and the convert-layout no-op
// collapse cannot distinguish two *different* modular (ADD-mod-N) register
// maps, so they would treat a modular relayout as a cheap identity no-op and
// scramble elements. These tests pin the two discriminators used by the
// modular-aware guards:
//   1. getFreeVariableMasks() is a conservative all-zero-basis check and
//   returns
//      the SAME masks for two different modular maps (isExpensiveView's old
//      fast path relies on it -> false "cheap"). operator!= DOES distinguish
//      them.
//   2. quotient()/squareSublayoutIsIdentity() uses the pure GF(2) predicate
//      (basis == 1<<b), which false-positives on a modular register basis, so
//      minimalCvtLayout would quotient the register dim away and collapse to a
//      no-op. isModular() flags the layout so the guard refuses the collapse.
TEST_F(LinearLayoutTest, ModularRelayoutIsNotIdentity) {
  // Two distinct modular maps over Z/6: strides 1 and 5. They are genuinely
  // different permutations of {0..5} but the conservative free-variable mask is
  // identical (no zero bases in either) -> the old isExpensiveView fast path
  // would call them equal ("cheap").
  auto modId = LinearLayout::modularStrided1D(6, 1, S("in"), S("out"));
  auto modRev = LinearLayout::modularStrided1D(6, 5, S("in"), S("out"));
  EXPECT_TRUE(modId.isModular());
  EXPECT_TRUE(modRev.isModular());
  // Old fast path: same free-variable masks despite being different maps.
  EXPECT_EQ(to_vector(modId.getFreeVariableMasks()),
            to_vector(modRev.getFreeVariableMasks()));
  // The guard's discriminator: they are not equal.
  EXPECT_NE(modId, modRev);
  EXPECT_EQ(modId, modId);
}

TEST_F(LinearLayoutTest, ModularRegisterDimNotQuotientedAway) {
  // A non-identity modular register map (stride 5 over Z/6). The GF(2) identity
  // predicate used by squareSublayoutIsIdentity treats basis 5 (== 0b101) as
  // "not identity" only for bit 0; but for a modular map the round-trip is not
  // GF(2)-linear, so quotient must not silently drop it. Confirm isModular is
  // set (the guard keys off it) and the map is not the identity.
  auto modRev = LinearLayout::modularStrided1D(6, 5, S("register"), S("out"));
  auto ident = LinearLayout::modularIdentity1D(6, S("register"), S("out"));
  EXPECT_TRUE(modRev.isModular());
  EXPECT_NE(modRev, ident);
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

TEST_F(LinearLayoutTest, Divide_Modular) {
  // Case 1: modular(6) * modular(4), both left and right
  {
    auto B = LinearLayout::modularStrided1D(6, 1, S("in"), S("out"));
    auto C = LinearLayout::modularStrided1D(4, 1, S("in"), S("out"));
    auto A = B * C;
    ASSERT_TRUE(divideLeft(A, B).has_value());
    EXPECT_EQ(divideLeft(A, B).value(), C);
    ASSERT_TRUE(divideRight(A, C).has_value());
    EXPECT_EQ(divideRight(A, C).value(), B);
  }

  // Case 2: mixed NPOT * pow2 (size 96 = 6 * 16)
  {
    auto B = LinearLayout::modularStrided1D(6, 1, S("in"), S("out"));
    auto C = LinearLayout::identity1D(16, S("in"), S("out"));
    auto A = B * C;
    EXPECT_EQ(divideLeft(A, B).value(), C);
    EXPECT_EQ(divideRight(A, C).value(), B);
  }
}

TEST_F(LinearLayoutTest, Divide_Modular_2D_Mixed) {
  auto B = LinearLayout::identity1D(4, S("in"), S("out1")) *
           LinearLayout::modularStrided1D(3, 1, S("in2"), S("out2"));
  auto C = LinearLayout::identity1D(2, S("in"), S("out1")) *
           LinearLayout::modularStrided1D(5, 1, S("in2"), S("out2"));
  auto A = B * C;
  EXPECT_EQ(divideLeft(A, B).value(), C);
  EXPECT_EQ(divideRight(A, C).value(), B);
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

//===----------------------------------------------------------------------===//
// Modular (Non-Power-of-2) Layout Tests
//===----------------------------------------------------------------------===//

TEST_F(LinearLayoutTest, ModularStrided1D_Size3) {
  // Test modularStrided1D with size=3 (smallest non-pow2)
  LinearLayout layout = LinearLayout::modularStrided1D(3, 1, S("in"), S("out"));

  // Should be marked as modular
  EXPECT_TRUE(layout.isModular());

  // Should have ceil(log2(3)) = 2 bases
  EXPECT_EQ(layout.getInDimSizeLog2(S("in")), 2);

  // Bases should be: (1*1) % 3 = 1, (1*2) % 3 = 2
  EXPECT_EQ(layout.getBasis(S("in"), 0, S("out")), 1);
  EXPECT_EQ(layout.getBasis(S("in"), 1, S("out")), 2);

  // Output dimension should be size 3
  EXPECT_EQ(layout.getOutDimSize(S("out")), 3);

  // Check surjectivity: should cover all values 0, 1, 2
  EXPECT_TRUE(layout.isModularSurjective());
}

TEST_F(LinearLayoutTest, ModularStrided1D_WithStride) {
  // Test modularStrided1D with stride != 1
  LinearLayout layout = LinearLayout::modularStrided1D(6, 2, S("in"), S("out"));

  EXPECT_TRUE(layout.isModular());
  EXPECT_EQ(layout.getInDimSizeLog2(S("in")), 3); // ceil(log2(6))

  // Bases should be: (2*1) % 6 = 2, (2*2) % 6 = 4, (2*4) % 6 = 2
  EXPECT_EQ(layout.getBasis(S("in"), 0, S("out")), 2);
  EXPECT_EQ(layout.getBasis(S("in"), 1, S("out")), 4);
  EXPECT_EQ(layout.getBasis(S("in"), 2, S("out")), 2);
  EXPECT_EQ(layout.getOutDimSize(S("out")), 6);

  // Test stride 5, size 6: L(x) = 5x mod 6 (gcd(5,6) = 1, surjective)
  LinearLayout stride5_6 =
      LinearLayout::modularStrided1D(6, 5, S("in"), S("out"));
  EXPECT_TRUE(stride5_6.isModular());
  EXPECT_TRUE(stride5_6.isSurjective());
  EXPECT_THAT(to_vector(stride5_6.getInDimNames()), ElementsAre(S("in")));
  EXPECT_THAT(to_vector(stride5_6.getOutDimNames()), ElementsAre(S("out")));
}

// ============================================================================
// CRT-based invertAndCompose tests (for non-power-of-2 dimensions)
// These tests exercise the CRT least-squares solver path.
// ============================================================================

TEST_F(LinearLayoutTest, InvertAndCompose_Modular_MultiDim_LCM) {
  // Test that lstsqModular uses LCM instead of max for multiple NPOT output
  // dims This is a regression test for NPOT-I1: lstsqModular was using
  // max(outDimSizes) as working modulus, which is incorrect for
  // multi-output-dim layouts where dims have different NPOT sizes. The correct
  // modulus is LCM(outDimSizes).

  // Create layout A with 2 output dimensions: size 6 and size 10
  // LCM(6, 10) = 30, max(6, 10) = 10
  LinearLayout::BasesT basesA;
  auto &bsA = basesA[S("in")];
  bsA.push_back({1, 0}); // bit 0: +1 to out0
  bsA.push_back({2, 0}); // bit 1: +2 to out0
  bsA.push_back({4, 0}); // bit 2: +4 to out0
  bsA.push_back({0, 1}); // bit 3: +1 to out1
  bsA.push_back({0, 2}); // bit 4: +2 to out1
  bsA.push_back({0, 4}); // bit 5: +4 to out1
  bsA.push_back({0, 8}); // bit 6: +8 to out1

  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.push_back({S("out0"), 6});
  outDims.push_back({S("out1"), 10});

  LinearLayout A(std::move(basesA), outDims, false);

  // Create layout B: subset of A
  LinearLayout::BasesT basesB;
  auto &bsB = basesB[S("in")];
  bsB.push_back({1, 0});
  bsB.push_back({2, 0});
  bsB.push_back({4, 0});
  bsB.push_back({0, 1});
  bsB.push_back({0, 2});

  LinearLayout B(std::move(basesB), outDims, false);

  // Invert and compose - should use LCM(6, 10) = 30 as working modulus
  LinearLayout result = A.invertAndCompose(B);

  // Verify correctness: result should map B's inputs to A's inputs such that
  // A(result(i)) = B(i) for all inputs i in B's domain
  for (int i = 0; i < (1 << B.getTotalInDimSizeLog2()); ++i) {
    auto b_out = B.apply({{S("in"), i}});
    auto result_in = result.apply({{S("in"), i}});
    auto actual = A.apply({{S("in"), result_in[0].second}});

    EXPECT_EQ(actual, b_out)
        << "Mismatch at i=" << i << ": B gives (" << b_out[0].second << ","
        << b_out[1].second << "), A(result(i)) gives (" << actual[0].second
        << "," << actual[1].second << ")";
  }
}

// Test multi-dimensional NPOT layouts
TEST_F(LinearLayoutTest, IsModularSurjectiveMultiDim) {
  // Create a 2D layout with NPOT dimensions: 3x6
  // out0: dim=3, out1: dim=6
  LinearLayout::BasesT bases2D;
  auto &inBases = bases2D[S("in")];

  // Create bases for a surjective 2D layout
  // For out0 (dim=3): bases [1, 2] mod 3 -> can reach {0, 1, 2}
  // For out1 (dim=6): bases [1, 2, 4] mod 6 -> can reach all 6 values
  inBases.push_back({1, 1}); // contributes 1 to out0, 1 to out1
  inBases.push_back({2, 2}); // contributes 2 to out0, 2 to out1
  inBases.push_back({0, 4}); // contributes 0 to out0, 4 to out1

  LinearLayout layout2D(std::move(bases2D), {{S("out0"), 3}, {S("out1"), 6}},
                        false);

  EXPECT_TRUE(layout2D.isModular())
      << "2D layout with dims 3x6 should be modular";
  EXPECT_EQ(layout2D.getNumOutDims(), 2) << "Should have 2 output dimensions";
  EXPECT_TRUE(layout2D.isModularSurjective())
      << "2D modular layout should be surjective when both dimensions are "
         "surjective";
  EXPECT_TRUE(layout2D.isSurjective())
      << "isSurjective() should dispatch to isModularSurjective() for "
         "multi-dim NPOT";

  // Test a non-surjective multi-dimensional layout
  LinearLayout::BasesT basesNonSurj2D;
  auto &inBases2 = basesNonSurj2D[S("in")];
  // For out0 (dim=3): only basis [2] mod 3 -> can only reach {0, 2}, missing
  // {1} For out1 (dim=6): bases [2, 4] mod 6 -> can only reach {0, 2, 4},
  // missing odd values
  inBases2.push_back({2, 2});
  inBases2.push_back({0, 4});

  LinearLayout layoutNonSurj2D(std::move(basesNonSurj2D),
                               {{S("out0"), 3}, {S("out1"), 6}}, false);

  EXPECT_TRUE(layoutNonSurj2D.isModular())
      << "Non-surjective 2D layout should be modular";
  EXPECT_FALSE(layoutNonSurj2D.isModularSurjective())
      << "Non-surjective multi-dim layout should return false";
}

TEST_F(LinearLayoutTest, Pow2AdditiveSurjectivityRegression) {
  // CORRECTNESS regression (T274241798 #1): apply() composes modular out-dims
  // with ADD+mod, but checkPow2Surjectivity previously used a GF(2)/XOR-rank
  // check. Bases {1,3,4,8} over out-dim size 12 are NON-surjective under
  // ADD+mod — their subset sums reach only 9 of 12 values
  // ({0,1,3,4,5,7,8,9,11}, missing {2,6,10}) — yet the XOR-rank check wrongly
  // reported surjective.
  LinearLayout layout({{S("in"), {{1}, {3}, {4}, {8}}}}, {{S("out"), 12}},
                      /*requireSurjective=*/false);
  EXPECT_TRUE(layout.isModular());

  // Brute-force the actual reachable set under apply()'s ADD+mod semantics and
  // confirm value 2 (among others) is unreachable.
  std::set<int32_t> reachable;
  for (int i = 0; i < (1 << layout.getInDimSizeLog2(S("in"))); ++i) {
    reachable.insert(layout.apply({{S("in"), i}})[0].second);
  }
  EXPECT_EQ(reachable.size(), 9u);
  EXPECT_EQ(reachable.count(2), 0u);

  // The fix: surjectivity must agree with apply() and report NON-surjective.
  EXPECT_FALSE(layout.isModularSurjective());
  EXPECT_FALSE(layout.isSurjective());

  // Sanity: a genuinely surjective modular layout still passes.
  auto surj = LinearLayout::modularStrided1D(12, 1, S("in"), S("out"));
  EXPECT_TRUE(surj.isModularSurjective());
}

TEST_F(LinearLayoutTest, SurjectivityTruncationRegression) {
  // operator* with same inDim+outDim produces 6 non-zero bases for modulus 27,
  // exceeding ceil(log2(27))=5. Old truncation gave false negative.
  auto inner = LinearLayout::modularStrided1D(3, 1, S("A"), S("out"));
  auto outer = LinearLayout::modularStrided1D(9, 1, S("A"), S("out"));
  auto product = inner * outer;
  EXPECT_EQ(product.getOutDimSize(S("out")), 27);
  EXPECT_TRUE(product.isModularSurjective());
}

//===----------------------------------------------------------------------===//
// Mixed-Shape Tests (pow2 + NPOT dims in the same layout)
// Verifies that per-layout isModular is correct for mixed shapes.
//===----------------------------------------------------------------------===//

TEST_F(LinearLayoutTest, MixedShapeSurjectivityUsesPerDimAlgebra) {
  LinearLayout layout({{S("in"), {{1, 0}, {3, 0}, {0, 1}, {0, 2}, {0, 4}}}},
                      {{S("pow2"), 4}, {S("npot"), 6}},
                      /*requireSurjective=*/false);

  EXPECT_TRUE(layout.isModular());
  EXPECT_TRUE(layout.isModularSurjective());
}

TEST_F(LinearLayoutTest, MixedShape_OperatorStar_SharedDim) {
  // When inner and outer share an output dim, sizes multiply (not shift).
  // inner: 4 -> dim0=4; outer: 6 -> dim0=6. Product should be 24, not 64.
  auto inner = LinearLayout::identity1D(4, S("register"), S("dim0"));
  auto outer = LinearLayout::modularStrided1D(6, 1, S("lane"), S("dim0"));
  auto product = inner * outer;

  EXPECT_EQ(product.getOutDimSize(S("dim0")), 24);
  EXPECT_TRUE(product.isModular());

  // Outer bases should be multiplied by inner size (4), not shifted.
  // outer basis[0] = 1 -> product basis[2] = 1*4 = 4
  // outer basis[1] = 2 -> product basis[3] = 2*4 = 8
  // outer basis[2] = 4 -> product basis[4] = 4*4 = 16
  EXPECT_EQ(product.getBasis(S("lane"), 0, S("dim0")), 4);
  EXPECT_EQ(product.getBasis(S("lane"), 1, S("dim0")), 8);
  EXPECT_EQ(product.getBasis(S("lane"), 2, S("dim0")), 16);
}

TEST_F(LinearLayoutTest, MixedShape_ReshapeOuts) {
  // Flatten/unflatten roundtrip for mixed layout.
  auto mixed = LinearLayout::identity1D(4, S("in"), S("dim0")) *
               LinearLayout::modularStrided1D(6, 1, S("in2"), S("dim1"));

  auto flat = mixed.flattenOuts();
  EXPECT_EQ(flat.getOutDimSize(S("dim0")), 24);

  // Verify that apply on the flattened layout matches mixed-radix encoding.
  for (int r = 0; r < 4; ++r) {
    for (int l = 0; l < 6; ++l) {
      auto mixedResult = mixed.apply({{S("in"), r}, {S("in2"), l}});
      auto flatResult = flat.apply({{S("in"), r}, {S("in2"), l}});
      int expected = mixedResult[0].second + mixedResult[1].second * 4;
      EXPECT_EQ(flatResult[0].second, expected) << "r=" << r << " l=" << l;
    }
  }
}

// invertAndCompose with mixed pow2 + NPOT output dims.
// Exercises lstsqModular dispatch when only some dims are NPOT.
TEST_F(LinearLayoutTest, InvertAndCompose_Mixed_Pow2AndNPOT) {
  // Layout A: in(32) -> dim0(4, pow2) x dim1(6, NPOT)
  LinearLayout::BasesT basesA;
  auto &bsA = basesA[S("in")];
  bsA.push_back({1, 0}); // bit 0: +1 to dim0
  bsA.push_back({2, 0}); // bit 1: +2 to dim0
  bsA.push_back({0, 1}); // bit 2: +1 to dim1
  bsA.push_back({0, 2}); // bit 3: +2 to dim1
  bsA.push_back({0, 4}); // bit 4: +4 to dim1

  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.push_back({S("dim0"), 4});
  outDims.push_back({S("dim1"), 6});
  LinearLayout A(std::move(basesA), outDims, false);

  // Layout B: in(16) -> dim0(4, pow2) x dim1(6, NPOT)
  LinearLayout::BasesT basesB;
  auto &bsB = basesB[S("in")];
  bsB.push_back({1, 0}); // bit 0: +1 to dim0
  bsB.push_back({2, 0}); // bit 1: +2 to dim0
  bsB.push_back({0, 1}); // bit 2: +1 to dim1
  bsB.push_back({0, 2}); // bit 3: +2 to dim1

  LinearLayout B(std::move(basesB), outDims, false);

  EXPECT_FALSE(A.isOutDimModular(S("dim0"))) << "dim0=4 is pow2";
  EXPECT_TRUE(A.isOutDimModular(S("dim1"))) << "dim1=6 is NPOT";
  EXPECT_TRUE(A.isModular()) << "mixed layout has NPOT dim";

  LinearLayout result = A.invertAndCompose(B);

  // Verify: A(result(i)) == B(i) for all i in B's domain
  for (int i = 0; i < (1 << B.getTotalInDimSizeLog2()); ++i) {
    auto b_out = B.apply({{S("in"), i}});
    auto result_in = result.apply({{S("in"), i}});
    auto actual = A.apply({{S("in"), result_in[0].second}});

    EXPECT_EQ(actual, b_out)
        << "Mismatch at i=" << i << ": B=(" << b_out[0].second << ","
        << b_out[1].second << "), A(result(i))=(" << actual[0].second << ","
        << actual[1].second << ")";
  }
}

// Prove XOR and ADD diverge for NPOT dim, and layout picks the right algebra.
TEST_F(LinearLayoutTest, MixedShape_XorVsAdd_Divergence) {
  // dim0=8 (pow2, uses XOR), dim1=6 (NPOT, uses ADD+UREM)
  auto pow2Part = LinearLayout::identity1D(8, S("in"), S("dim0"));
  auto npotPart = LinearLayout::modularStrided1D(6, 1, S("in2"), S("dim1"));
  auto mixed = pow2Part * npotPart;

  EXPECT_FALSE(mixed.isOutDimModular(S("dim0")));
  EXPECT_TRUE(mixed.isOutDimModular(S("dim1")));

  // input=7 for dim1 (bits 0+1+2): XOR(1,2,4)=7, ADD(1+2+4)%6=1
  // The NPOT dim must use ADD+UREM, giving 1 (not 7).
  auto result = mixed.apply({{S("in"), 0}, {S("in2"), 7}});
  int dim1val = result[1].second;
  EXPECT_EQ(dim1val, 1)
      << "NPOT dim should use ADD+UREM: (1+2+4)%6=1, not XOR=7";
  EXPECT_LT(dim1val, 6) << "NPOT result must be in range [0, 6)";

  // For pow2 dim, input=7: XOR(1,2,4)=7 == ADD(1+2+4)=7 (both agree for pow2)
  auto result2 = mixed.apply({{S("in"), 7}, {S("in2"), 0}});
  EXPECT_EQ(result2[0].second, 7) << "pow2 dim: 1 XOR 2 XOR 4 = 7";
}

// Verify isOutDimModular is preserved through compose and invertAndCompose.
TEST_F(LinearLayoutTest, MixedShape_Compose_PreservesPerDimModular) {
  // Build a mixed layout: dim0=4 (pow2), dim1=6 (NPOT)
  auto mixed = LinearLayout::identity1D(4, S("in"), S("dim0")) *
               LinearLayout::modularStrided1D(6, 1, S("in2"), S("dim1"));

  EXPECT_FALSE(mixed.isOutDimModular(S("dim0")));
  EXPECT_TRUE(mixed.isOutDimModular(S("dim1")));

  // Compose with identity-like transform preserves per-dim sizes.
  auto transform = LinearLayout::identity1D(4, S("dim0"), S("out0")) *
                   LinearLayout::modularStrided1D(6, 1, S("dim1"), S("out1"));
  auto composed = mixed.compose(transform);

  EXPECT_FALSE(composed.isOutDimModular(S("out0")))
      << "pow2 dim stays pow2 after compose";
  EXPECT_TRUE(composed.isOutDimModular(S("out1")))
      << "NPOT dim stays NPOT after compose";

  // Verify compose matches sequential apply.
  for (int r = 0; r < 4; ++r) {
    for (int l = 0; l < 6; ++l) {
      auto mixedResult = mixed.apply({{S("in"), r}, {S("in2"), l}});
      auto transformResult = transform.apply(mixedResult);
      auto composedResult = composed.apply({{S("in"), r}, {S("in2"), l}});
      EXPECT_EQ(composedResult[0].second, transformResult[0].second);
      EXPECT_EQ(composedResult[1].second, transformResult[1].second);
    }
  }

  // invertAndCompose with mixed dims preserves per-dim modular status.
  // Uses same layout structure as InvertAndCompose_Mixed_Pow2AndNPOT.
  LinearLayout::BasesT basesA;
  auto &bsA = basesA[S("in")];
  bsA.push_back({1, 0});
  bsA.push_back({2, 0});
  bsA.push_back({0, 1});
  bsA.push_back({0, 2});
  bsA.push_back({0, 4});

  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.push_back({S("d0"), 4});
  outDims.push_back({S("d1"), 6});
  LinearLayout A(std::move(basesA), outDims, false);

  LinearLayout::BasesT basesB;
  auto &bsB = basesB[S("in")];
  bsB.push_back({1, 0});
  bsB.push_back({2, 0});
  bsB.push_back({0, 1});
  bsB.push_back({0, 2});
  LinearLayout B(std::move(basesB), outDims, false);

  EXPECT_FALSE(A.isOutDimModular(S("d0")));
  EXPECT_TRUE(A.isOutDimModular(S("d1")));

  LinearLayout iac = A.invertAndCompose(B);
  // Result maps B's input dim "in" -> A's input dim "in" (both pow2).
  // Verify correctness: A(iac(i)) == B(i) for all i.
  for (int i = 0; i < (1 << B.getTotalInDimSizeLog2()); ++i) {
    auto b_out = B.apply({{S("in"), i}});
    auto r_in = iac.apply({{S("in"), i}});
    auto a_out = A.apply({{S("in"), r_in[0].second}});
    EXPECT_EQ(a_out, b_out) << "invertAndCompose mismatch at i=" << i;
  }
}

// Verify that final-UREM produces the same result as per-step UREM for all
// inputs, confirming the codegen optimization is algebraically equivalent.
TEST_F(LinearLayoutTest, CarryFreeEquivalence_FinalUremEqualsPerStep) {
  for (int size : {3, 6, 48}) {
    auto layout = LinearLayout::modularStrided1D(size, 1, S("in"), S("out"));
    int nBits = layout.getInDimSizeLog2(S("in"));
    int nInputs = 1 << nBits;

    for (int input = 0; input < nInputs; ++input) {
      // Per-step UREM (what apply() does)
      auto perStep = layout.apply({{S("in"), input}});

      // Final-UREM only: accumulate bases without intermediate modulo
      int32_t finalSum = 0;
      for (int bit = 0; bit < nBits; ++bit) {
        if (input & (1 << bit))
          finalSum += layout.getBasis(S("in"), bit, S("out"));
      }
      int32_t finalResult = finalSum % size;

      EXPECT_EQ(perStep[0].second, finalResult)
          << "size=" << size << " input=" << input;
    }
  }
}

//===----------------------------------------------------------------------===//
// NPOT x NPOT Apply Tests (both dims non-power-of-2)
// Validates modularStrided1D apply() for shapes from the 2D NPOT benchmarks.
//===----------------------------------------------------------------------===//

TEST_F(LinearLayoutTest, MixedShape_Apply_NpotXNpot) {
  // Jagged tensor shapes: (33, 48), (3, 768)
  // Grouped operation shapes: (12, 48), (6, 96)
  struct Shape {
    int32_t m, n;
  };
  Shape shapes[] = {{33, 48}, {3, 768}, {12, 48}, {6, 96}};

  for (auto [m, n] : shapes) {
    auto layoutM = LinearLayout::modularStrided1D(m, 1, S("dim0"), S("out0"));
    auto layoutN = LinearLayout::modularStrided1D(n, 1, S("dim1"), S("out1"));
    auto product = layoutM * layoutN;

    EXPECT_TRUE(product.isModular())
        << m << "x" << n << " product should be modular";
    EXPECT_EQ(product.getOutDimSize(S("out0")), m);
    EXPECT_EQ(product.getOutDimSize(S("out1")), n);

    // Verify apply() produces values in range [0, m) x [0, n)
    int nBits0 = product.getInDimSizeLog2(S("dim0"));
    int nBits1 = product.getInDimSizeLog2(S("dim1"));
    for (int i = 0; i < (1 << nBits0); ++i) {
      for (int j = 0; j < (1 << nBits1); ++j) {
        auto result = product.apply({{S("dim0"), i}, {S("dim1"), j}});
        EXPECT_GE(result[0].second, 0) << m << "x" << n << " i=" << i;
        EXPECT_LT(result[0].second, m) << m << "x" << n << " i=" << i;
        EXPECT_GE(result[1].second, 0) << m << "x" << n << " j=" << j;
        EXPECT_LT(result[1].second, n) << m << "x" << n << " j=" << j;
      }
    }

    // Verify surjectivity: all (m*n) output values are reachable
    EXPECT_TRUE(product.isModularSurjective())
        << m << "x" << n << " product should be surjective";
  }
}

TEST_F(LinearLayoutTest, PseudoinvertNPOT) {
  // pseudoinvert must not crash on NPOT output dimensions.
  for (int size : {3, 5, 6, 7, 9, 10, 12}) {
    auto layout = LinearLayout::modularIdentity1D(size, S("in"), S("out"));
    auto inv = layout.pseudoinvert();

    // pseudoinvert(A) composed with A should act as identity on reachable
    // outputs: for each input i, apply(pseudoinvert(apply(i))) == i.
    for (int i = 0; i < size; ++i) {
      auto fwd = layout.apply({{S("in"), i}});
      auto back = inv.apply({{S("out"), fwd[0].second}});
      EXPECT_EQ(back[0].second, i) << "size=" << size << " i=" << i;
    }
  }
}

TEST_F(LinearLayoutTest, ModularIdentity1D_ExhaustiveApply) {
  // Brute-force verify apply() covers every value in [0, dim).
  for (int dim : {3, 5, 6, 7, 9, 10, 12, 15}) {
    auto layout = LinearLayout::modularIdentity1D(dim, S("in"), S("out"));
    int nBits = layout.getInDimSizeLog2(S("in"));
    int nInputs = 1 << nBits;

    std::set<int32_t> outputs;
    for (int i = 0; i < nInputs; ++i) {
      auto result = layout.apply({{S("in"), i}});
      EXPECT_GE(result[0].second, 0) << "dim=" << dim << " i=" << i;
      EXPECT_LT(result[0].second, dim) << "dim=" << dim << " i=" << i;
      outputs.insert(result[0].second);
    }

    EXPECT_EQ(static_cast<int>(outputs.size()), dim)
        << "dim=" << dim << ": apply() must cover all values in [0, " << dim
        << ")";
  }
}

// Compose two modular layouts sharing an output dimension.
TEST_F(LinearLayoutTest, ComposeModularSharedOutDim) {
  // 1D: inner maps in->dim0(6), outer maps dim0->out(6) with stride 2.
  auto inner = LinearLayout::modularStrided1D(6, 1, S("in"), S("dim0"));
  auto outer = LinearLayout::modularStrided1D(6, 2, S("dim0"), S("out"));
  auto composed = inner.compose(outer);

  for (int x = 0; x < (1 << inner.getInDimSizeLog2(S("in"))); ++x) {
    auto inner_out = inner.apply({{S("in"), x}});
    auto outer_out = outer.apply(inner_out);
    EXPECT_EQ(composed.apply({{S("in"), x}}), outer_out) << "x=" << x;
  }

  // 2D: compose modular layouts over Z/3 x Z/5.
  auto inner2D = LinearLayout::modularStrided1D(3, 1, S("in"), S("d0")) *
                 LinearLayout::modularStrided1D(5, 1, S("in2"), S("d1"));
  auto outer2D = LinearLayout::modularStrided1D(3, 2, S("d0"), S("out0")) *
                 LinearLayout::modularStrided1D(5, 3, S("d1"), S("out1"));
  auto composed2D = inner2D.compose(outer2D);

  for (int i = 0; i < (1 << inner2D.getInDimSizeLog2(S("in"))); ++i) {
    for (int j = 0; j < (1 << inner2D.getInDimSizeLog2(S("in2"))); ++j) {
      auto in_out = inner2D.apply({{S("in"), i}, {S("in2"), j}});
      auto out_out = outer2D.apply(in_out);
      EXPECT_EQ(composed2D.apply({{S("in"), i}, {S("in2"), j}}), out_out)
          << "i=" << i << " j=" << j;
    }
  }
}

// Flatten a 2D NPOT layout to 1D then unflatten back; verify round-trip.
TEST_F(LinearLayoutTest, ReshapeOutsNPOT_FlattenUnflattenRoundtrip) {
  // [3, 5] -> [15] -> [3, 5]
  auto original = LinearLayout::modularStrided1D(3, 1, S("in1"), S("out0")) *
                  LinearLayout::modularStrided1D(5, 1, S("in2"), S("out1"));

  auto flat = original.flattenOuts();
  EXPECT_EQ(flat.getOutDimSize(S("out0")), 15);

  auto unflat = flat.reshapeOuts({{S("out0"), 3}, {S("out1"), 5}});

  for (int i = 0; i < (1 << original.getInDimSizeLog2(S("in1"))); ++i) {
    for (int j = 0; j < (1 << original.getInDimSizeLog2(S("in2"))); ++j) {
      EXPECT_EQ(unflat.apply({{S("in1"), i}, {S("in2"), j}}),
                original.apply({{S("in1"), i}, {S("in2"), j}}))
          << "i=" << i << " j=" << j;
    }
  }

  // Reversed dim order: [5, 3] -> [15] -> [5, 3]
  auto origR = LinearLayout::modularStrided1D(5, 1, S("in1"), S("out0")) *
               LinearLayout::modularStrided1D(3, 1, S("in2"), S("out1"));
  auto flatR = origR.flattenOuts();
  auto unflatR = flatR.reshapeOuts({{S("out0"), 5}, {S("out1"), 3}});

  for (int i = 0; i < (1 << origR.getInDimSizeLog2(S("in1"))); ++i) {
    for (int j = 0; j < (1 << origR.getInDimSizeLog2(S("in2"))); ++j) {
      EXPECT_EQ(unflatR.apply({{S("in1"), i}, {S("in2"), j}}),
                origR.apply({{S("in1"), i}, {S("in2"), j}}))
          << "i=" << i << " j=" << j;
    }
  }
}

// Verify round-trip: A.invertAndCompose(B).compose(B) == A for modular layouts.
TEST_F(LinearLayoutTest, InvertAndCompose_ModularRoundTrip) {
  // 1D round-trips with various NPOT dim sizes and coprime strides.
  struct Case {
    int size, strideA, strideB;
  };
  Case cases[] = {{6, 1, 5}, {10, 3, 7}, {15, 4, 11}};
  for (auto [size, sA, sB] : cases) {
    auto A = LinearLayout::modularStrided1D(size, sA, S("in1"), S("out"));
    auto B = LinearLayout::modularStrided1D(size, sB, S("in2"), S("out"));
    auto X = A.invertAndCompose(B);
    EXPECT_EQ(X.compose(B), A)
        << "size=" << size << " sA=" << sA << " sB=" << sB;
  }

  // Multi-dim: both dims same prime size avoids LCM modulus aliasing.
  auto Am = LinearLayout::modularStrided1D(3, 1, S("in1"), S("d0")) *
            LinearLayout::modularStrided1D(3, 1, S("in1b"), S("d1"));
  auto Bm = LinearLayout::modularStrided1D(3, 2, S("in2"), S("d0")) *
            LinearLayout::modularStrided1D(3, 2, S("in2b"), S("d1"));
  auto Xm = Am.invertAndCompose(Bm);
  EXPECT_EQ(Xm.compose(Bm),
            Am.transposeOuts(llvm::to_vector(Bm.getOutDimNames())));
}

// Multi-rep NPOT invertAndCompose: simulates a multi-rep shared memory
// conversion with NPOT dimensions (e.g. size 12). Verifies that all output
// values remain in bounds and the solution is correct.
TEST_F(LinearLayoutTest, InvertAndCompose_MultiRepNPOT) {
  // A: 2 input dims (register, lane), output dim with NPOT size 12.
  // Simulates a layout where multiple reps tile an NPOT dimension.
  auto A = LinearLayout::modularIdentity1D(12, S("register"), S("dim0")) *
           LinearLayout::zeros1D(4, S("lane"), S("dim0"));

  // B: different input structure mapping to same output space.
  auto B = LinearLayout::modularIdentity1D(12, S("register"), S("dim0")) *
           LinearLayout::zeros1D(4, S("lane"), S("dim0"));

  EXPECT_TRUE(A.isOutDimModular(S("dim0")));

  auto iac = A.invertAndCompose(B);

  // Verify all output values are in bounds.
  int regSize = iac.getInDimSize(S("register"));
  int laneSize = iac.getInDimSize(S("lane"));
  for (int reg = 0; reg < regSize; reg++) {
    for (int lane = 0; lane < laneSize; lane++) {
      auto result = iac.apply({{S("register"), reg}, {S("lane"), lane}});
      for (auto &[dim, val] : result) {
        EXPECT_GE(val, 0) << "Negative output at reg=" << reg
                          << " lane=" << lane;
        EXPECT_LT(val, iac.getOutDimSize(dim))
            << "Out of bounds at reg=" << reg << " lane=" << lane
            << " dim=" << dim.str();
      }
    }
  }

  // Verify correctness: A(iac(input)) == B(input) for all inputs.
  for (int reg = 0; reg < regSize; reg++) {
    for (int lane = 0; lane < laneSize; lane++) {
      auto b_out = B.apply({{S("register"), reg}, {S("lane"), lane}});
      auto r_vals = iac.apply({{S("register"), reg}, {S("lane"), lane}});
      auto a_out = A.apply(r_vals);
      EXPECT_EQ(a_out, b_out)
          << "invertAndCompose mismatch at reg=" << reg << " lane=" << lane;
    }
  }
}

// Multi-rep NPOT invertAndCompose with mixed pow2 and NPOT output dims.
// Simulates shared memory conversion for a tensor with shape like [32, 12].
TEST_F(LinearLayoutTest, InvertAndCompose_MultiRepMixedNPOT) {
  // A: maps register -> (dim0=8, dim1=12)
  // Shear family: some bases touch only pow2 dim, others touch NPOT dim.
  LinearLayout::BasesT basesA;
  auto &bsA = basesA[S("register")];
  // Group-0 bases (touch pow2 dim0)
  bsA.push_back({1, 0}); // bit 0 -> dim0 += 1
  bsA.push_back({2, 0}); // bit 1 -> dim0 += 2
  bsA.push_back({4, 0}); // bit 2 -> dim0 += 4
  // Group-1 bases (touch only NPOT dim1)
  bsA.push_back({0, 1}); // bit 3 -> dim1 += 1
  bsA.push_back({0, 2}); // bit 4 -> dim1 += 2
  bsA.push_back({0, 4}); // bit 5 -> dim1 += 4
  bsA.push_back({0, 8}); // bit 6 -> dim1 += 8

  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.push_back({S("dim0"), 8});
  outDims.push_back({S("dim1"), 12});
  LinearLayout A(std::move(basesA), outDims, false);

  // B: subset mapping
  LinearLayout::BasesT basesB;
  auto &bsB = basesB[S("register")];
  bsB.push_back({1, 0});
  bsB.push_back({2, 0});
  bsB.push_back({4, 0});
  bsB.push_back({0, 1});
  bsB.push_back({0, 2});
  bsB.push_back({0, 4});
  LinearLayout B(std::move(basesB), outDims, false);

  EXPECT_FALSE(A.isOutDimModular(S("dim0")));
  EXPECT_TRUE(A.isOutDimModular(S("dim1")));

  auto iac = A.invertAndCompose(B);

  // Verify correctness.
  int inSize = 1 << B.getTotalInDimSizeLog2();
  for (int i = 0; i < inSize; i++) {
    auto b_out = B.apply({{S("register"), i}});
    auto r_vals = iac.apply({{S("register"), i}});
    auto a_out = A.apply(r_vals);
    EXPECT_EQ(a_out, b_out) << "invertAndCompose mismatch at i=" << i;
  }
}

// Test the per-dim factored solve in lstsq for mixed pow2/NPOT layouts.
// This exercises the code path where lstsq splits output dims into pow2
// (solved via GF(2) RREF) and NPOT (solved via lstsqModular), then combines.
TEST_F(LinearLayoutTest, Lstsq_PerDimFactoredSolve_MixedPow2NPOT) {
  // A: maps input(128) -> dim0(8, pow2) x dim1(12, NPOT)
  // Simulates the split-dim layout: dim0 uses XOR, dim1 uses ADD+mod.
  // Bases are decoupled: pow2 bases have S=0 in dim1, NPOT bases have S=0 in
  // dim0.
  LinearLayout::BasesT basesA;
  auto &bsA = basesA[S("in")];
  // Pow2 dim0 bases (XOR algebra)
  bsA.push_back({1, 0}); // bit 0 -> dim0 XOR 1
  bsA.push_back({2, 0}); // bit 1 -> dim0 XOR 2
  bsA.push_back({4, 0}); // bit 2 -> dim0 XOR 4
  // NPOT dim1 bases (ADD+mod algebra)
  bsA.push_back({0, 1}); // bit 3 -> dim1 ADD 1
  bsA.push_back({0, 2}); // bit 4 -> dim1 ADD 2
  bsA.push_back({0, 4}); // bit 5 -> dim1 ADD 4
  bsA.push_back({0, 8}); // bit 6 -> dim1 ADD 8

  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.push_back({S("dim0"), 8});
  outDims.push_back({S("dim1"), 12});
  LinearLayout A(std::move(basesA), outDims, false);

  // B: maps input(64) -> dim0(8) x dim1(12), subset of A's image
  LinearLayout::BasesT basesB;
  auto &bsB = basesB[S("in")];
  bsB.push_back({1, 0}); // bit 0 -> dim0 += 1
  bsB.push_back({2, 0}); // bit 1 -> dim0 += 2
  bsB.push_back({4, 0}); // bit 2 -> dim0 += 4
  bsB.push_back({0, 1}); // bit 3 -> dim1 += 1
  bsB.push_back({0, 2}); // bit 4 -> dim1 += 2
  bsB.push_back({0, 4}); // bit 5 -> dim1 += 4
  LinearLayout B(std::move(basesB), outDims, false);

  EXPECT_TRUE(A.isModular());
  EXPECT_FALSE(A.isOutDimModular(S("dim0")));
  EXPECT_TRUE(A.isOutDimModular(S("dim1")));

  // invertAndCompose calls lstsq internally.
  auto result = A.invertAndCompose(B);

  // Verify: A(result(i)) == B(i) for all i in B's domain.
  int inSize = 1 << B.getTotalInDimSizeLog2();
  for (int i = 0; i < inSize; i++) {
    auto b_out = B.apply({{S("in"), i}});
    auto r_in = result.apply({{S("in"), i}});
    auto a_out = A.apply(r_in);
    EXPECT_EQ(a_out, b_out) << "Per-dim factored solve mismatch at i=" << i;
  }
}

// Test per-dim factored solve with multi-input-dim layout (register + lane).
TEST_F(LinearLayoutTest, Lstsq_PerDimFactoredSolve_MultiInputDim) {
  // A: maps (register x lane) -> dim0(4, pow2) x dim1(6, NPOT)
  auto A_pow2 = LinearLayout::identity1D(4, S("register"), S("dim0"));
  auto A_npot = LinearLayout::modularStrided1D(6, 1, S("lane"), S("dim1"));
  auto A = A_pow2 * A_npot;

  // B: maps (register x lane) -> dim0(4) x dim1(6), same structure
  auto B_pow2 = LinearLayout::identity1D(4, S("register"), S("dim0"));
  auto B_npot = LinearLayout::modularStrided1D(6, 1, S("lane"), S("dim1"));
  auto B_layout = B_pow2 * B_npot;

  EXPECT_TRUE(A.isModular());

  auto result = A.invertAndCompose(B_layout);

  // Verify: A(result(input)) == B(input) for all inputs.
  int regSize = B_layout.getInDimSize(S("register"));
  int laneSize = B_layout.getInDimSize(S("lane"));
  for (int reg = 0; reg < regSize; reg++) {
    for (int lane = 0; lane < laneSize; lane++) {
      auto b_out = B_layout.apply({{S("register"), reg}, {S("lane"), lane}});
      auto r_vals = result.apply({{S("register"), reg}, {S("lane"), lane}});
      auto a_out = A.apply(r_vals);
      EXPECT_EQ(a_out, b_out)
          << "Multi-input-dim mismatch at reg=" << reg << " lane=" << lane;
    }
  }
}

// Verify modularIdentity1D(6, ...) composes correctly with other layouts,
// simulating NPOT kWidth=6 for NVFp4 K=96.
TEST_F(LinearLayoutTest, ModularIdentity1D_KWidth6_Compose) {
  // kWidth=6 register layout along K dimension
  auto regs = LinearLayout::modularIdentity1D(6, S("register"), S("dimK"));

  // 4 lanes along non-K dimension (pow2)
  auto lanes = LinearLayout::identity1D(4, S("lane"), S("dimNonK"));

  // Compose: regs * lanes
  auto tileLayout = regs * lanes;

  EXPECT_EQ(tileLayout.getOutDimSize(S("dimK")), 6);
  EXPECT_EQ(tileLayout.getOutDimSize(S("dimNonK")), 4);

  // Verify all output values are in bounds.
  int regSize = tileLayout.getInDimSize(S("register"));
  int laneSize = tileLayout.getInDimSize(S("lane"));
  for (int reg = 0; reg < regSize; reg++) {
    for (int lane = 0; lane < laneSize; lane++) {
      auto result = tileLayout.apply({{S("register"), reg}, {S("lane"), lane}});
      for (auto &[dim, val] : result) {
        EXPECT_GE(val, 0);
        EXPECT_LT(val, tileLayout.getOutDimSize(dim))
            << "Out of bounds: reg=" << reg << " lane=" << lane
            << " dim=" << dim.str() << " val=" << val;
      }
    }
  }

  // Verify invertAndCompose with itself produces identity-like mapping.
  auto iac = tileLayout.invertAndCompose(tileLayout);
  for (int reg = 0; reg < regSize; reg++) {
    for (int lane = 0; lane < laneSize; lane++) {
      auto orig = tileLayout.apply({{S("register"), reg}, {S("lane"), lane}});
      auto mapped = iac.apply({{S("register"), reg}, {S("lane"), lane}});
      auto roundTrip = tileLayout.apply(mapped);
      EXPECT_EQ(orig, roundTrip)
          << "Round-trip mismatch at reg=" << reg << " lane=" << lane;
    }
  }
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

} // anonymous namespace
} // namespace mlir::triton

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
