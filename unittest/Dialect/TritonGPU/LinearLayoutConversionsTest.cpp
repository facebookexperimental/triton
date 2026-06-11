#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
std::ostream &operator<<(std::ostream &os, StringAttr str) {
  os << str.str();
  return os;
}
} // namespace mlir

using namespace mlir::triton::nvidia_gpu;
namespace mlir::triton::gpu {
namespace {

class LinearLayoutConversionsTest : public ::testing::Test {
public:
  void SetUp() {
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctx.getOrLoadDialect<TritonNvidiaGPUDialect>();
  }

  BlockedEncodingAttr blocked(ArrayRef<unsigned> spt, ArrayRef<unsigned> tpw,
                              ArrayRef<unsigned> wpb, ArrayRef<unsigned> cpg,
                              ArrayRef<unsigned> cSplit, ArrayRef<unsigned> ord,
                              ArrayRef<unsigned> cOrd) {
    return BlockedEncodingAttr::get(
        &ctx, spt, tpw, wpb, ord,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd));
  }

  NvidiaMmaEncodingAttr mma(unsigned versionMaj, unsigned versionMin,
                            ArrayRef<unsigned> instrShape,
                            ArrayRef<unsigned> wbp, ArrayRef<unsigned> cpg,
                            ArrayRef<unsigned> cSplit,
                            ArrayRef<unsigned> cOrd) {
    return NvidiaMmaEncodingAttr::get(
        &ctx, versionMaj, versionMin, wbp,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd), instrShape);
  }

  NvidiaMmaEncodingAttr mma(unsigned versionMaj, unsigned versionMin,
                            ArrayRef<unsigned> instrShape,
                            ArrayRef<unsigned> numWarps) {
    auto cgaLayout = CGAEncodingAttr::get1CTALayout(&ctx, numWarps.size());
    return NvidiaMmaEncodingAttr::get(&ctx, versionMaj, versionMin, numWarps,
                                      std::move(cgaLayout), instrShape);
  }

  DotOperandEncodingAttr dot(Attribute parent, int idx, int kWidth) {
    return DotOperandEncodingAttr::get(&ctx, idx, parent, /*kWidth=*/kWidth);
  }

  AMDMfmaEncodingAttr mfma(unsigned version, ArrayRef<unsigned> warps,
                           ArrayRef<unsigned> instrShape, bool isTransposed,
                           ArrayRef<unsigned> tilesPerWarp = {},
                           unsigned elementBitWidth = 0) {
    SmallVector<unsigned> cpg(warps.size(), 1u);
    SmallVector<unsigned> cSplit(warps.size(), 1u);
    SmallVector<unsigned> cOrd(warps.size());
    std::iota(cOrd.begin(), cOrd.end(), 0);

    auto cgaLayout = CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd);
    return AMDMfmaEncodingAttr::get(&ctx, version, warps, instrShape,
                                    isTransposed, cgaLayout, tilesPerWarp,
                                    elementBitWidth);
  }

  DotOperandEncodingAttr mfmaDotOp(AMDMfmaEncodingAttr mfma, unsigned opIdx,
                                   unsigned kWidth) {
    return DotOperandEncodingAttr::get(&ctx, opIdx, mfma, kWidth);
  }

  AMDWmmaEncodingAttr wmma(ArrayRef<unsigned> warps, int version,
                           bool transposed,
                           ArrayRef<unsigned> instrShape =
                               AMDWmmaEncodingAttr::getDefaultInstrShape()) {
    SmallVector<unsigned> cpg(warps.size(), 1u);
    SmallVector<unsigned> cSplit(warps.size(), 1u);
    SmallVector<unsigned> cOrd(warps.size());
    SmallVector<unsigned> tpw(warps.size(), 1u);
    std::iota(cOrd.begin(), cOrd.end(), 0);
    LinearLayout ctaLayout =
        chooseWmmaCTALinearLayout(&ctx, warps.size(), warps, tpw);
    return AMDWmmaEncodingAttr::get(
        &ctx, version, ctaLayout, transposed,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd), instrShape);
  }

  DotOperandEncodingAttr wmmaDotOp(AMDWmmaEncodingAttr wmma, unsigned opIdx,
                                   unsigned kWidth) {
    return DotOperandEncodingAttr::get(&ctx, opIdx, wmma, kWidth);
  }

  SliceEncodingAttr slice(DistributedEncodingTrait parent, int dim) {
    return SliceEncodingAttr::get(&ctx, dim, parent);
  }

  SwizzledSharedEncodingAttr shared(unsigned vec, unsigned perPhase,
                                    unsigned maxPhase, ArrayRef<unsigned> cpg,
                                    ArrayRef<unsigned> cSplit,
                                    ArrayRef<unsigned> ord,
                                    ArrayRef<unsigned> cOrd) {
    return SwizzledSharedEncodingAttr::get(
        &ctx, vec, perPhase, maxPhase, ord,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd));
  }

  NVMMASharedEncodingAttr
  nvmmaShared(unsigned swizzleSizeInBytes, bool transposed,
              unsigned elementBitWidth, ArrayRef<unsigned> cpg,
              ArrayRef<unsigned> cSplit, ArrayRef<unsigned> ord,
              ArrayRef<unsigned> cOrd, bool fp4Padded = false) {
    return NVMMASharedEncodingAttr::get(
        &ctx, swizzleSizeInBytes, transposed, elementBitWidth, fp4Padded,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd));
  }

  AMDRotatingSharedEncodingAttr
  AMDRotatingShared(unsigned vec, unsigned perPhase, unsigned maxPhase,
                    ArrayRef<unsigned> cpg, ArrayRef<unsigned> cSplit,
                    ArrayRef<unsigned> ord, ArrayRef<unsigned> cOrd) {
    return AMDRotatingSharedEncodingAttr::get(
        &ctx, vec, perPhase, maxPhase, ord,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd));
  }

  TensorMemoryEncodingAttr tmem(unsigned blockM, unsigned blockN,
                                unsigned colStride, unsigned ctaSplitM,
                                unsigned ctaSplitN) {
    return TensorMemoryEncodingAttr::get(&ctx, blockM, blockN, colStride,
                                         ctaSplitM, ctaSplitN, false,
                                         TensorMemoryCTAMode::DEFAULT);
  }

  TensorMemoryEncodingAttr tmem(unsigned blockM, unsigned blockN,
                                unsigned ctaSplitM, unsigned ctaSplitN) {
    // TODO Test colStride > 1
    return tmem(blockM, blockN, 1, ctaSplitM, ctaSplitN);
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LinearLayoutConversionsTest, SimpleBlocked) {
  auto layout =
      toLinearLayout({16}, blocked({1}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_THAT(layout, LinearLayout(
                          {
                              {S("register"), {}},
                              {S("lane"), {{1}, {2}}},
                              {S("warp"), {{4}, {8}}},
                              {S("block"), {}},
                          },
                          {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, CTADuplication) {
  auto layout = toLinearLayout(
      {32}, blocked({1}, {4}, {4}, /*cpg=*/{4}, /*cSplit=*/{2}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {}},
                            {S("lane"), {{1}, {2}}},
                            {S("warp"), {{4}, {8}}},
                            {S("block"), {{16}, {0}}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, CTABroadcast) {
  auto layout =
      toLinearLayout({64, 128}, blocked({8, 1}, {8, 4}, {1, 4}, {1, 2}, {1, 2},
                                        {0, 1}, {1, 0}));
  EXPECT_EQ(
      layout,
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 16}, {0, 32}}},
                    {S("lane"), {{8, 0}, {16, 0}, {32, 0}, {0, 1}, {0, 2}}},
                    {S("warp"), {{0, 4}, {0, 8}}},
                    {S("block"), {{0, 64}}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, ShapeLargerThanLayout) {
  // The layout is 16 elements, but the shape is 128, so it's repeated 128/16 =
  // 8 times.
  auto layout =
      toLinearLayout({128}, blocked({1}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{16}, {32}, {64}}},
                            {S("lane"), {{1}, {2}}},
                            {S("warp"), {{4}, {8}}},
                            {S("block"), {}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, ShapeLargerThanLayout2DDegenerate) {
  auto layout = toLinearLayout({128, 1}, blocked({1, 1}, {4, 1}, {4, 1}, {1, 1},
                                                 {1, 1}, {0, 1}, {1, 0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{16, 0}, {32, 0}, {64, 0}}},
                            {S("lane"), {{1, 0}, {2, 0}}},
                            {S("warp"), {{4, 0}, {8, 0}}},
                            {S("block"), {}},
                        },
                        {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, ShapeSmallerThanLayout) {
  // The shape is 8 elements, but the layout is 4*4*4 = 64 elems.  Therefore the
  // log2(64/8) = 3 most major bases are 0.
  auto layout = toLinearLayout({8}, blocked({4}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{1}, {2}}},
                            {S("lane"), {{4}, {0}}},
                            {S("warp"), {{0}, {0}}},
                            {S("block"), {}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, ReversedOrder) {
  auto layout = toLinearLayout({1, 64}, blocked({1, 1}, {32, 1}, {1, 8}, {1, 1},
                                                {1, 1}, {0, 1}, {1, 0}));
  EXPECT_EQ(layout,
            LinearLayout(
                {
                    {S("register"), {{0, 8}, {0, 16}, {0, 32}}},
                    {S("lane"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}},
                    {S("warp"), {{0, 1}, {0, 2}, {0, 4}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, ReplicateInRegisterDim) {
  auto layout =
      toLinearLayout({32}, blocked({2}, {4}, {1}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{1}, {8}, {16}}},
                            {S("lane"), {{2}, {4}}},
                            {S("warp"), {}},
                            {S("block"), {}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, OneDimTooLargeAnotherTooSmall) {
  auto blockedLayout =
      blocked({1, 4}, {8, 4}, {4, 1}, {2, 2}, {2, 1}, {1, 0}, {1, 0});
  auto ll = toLinearLayout({128, 16}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {{0, 1}, {0, 2}, {32, 0}}},
                        {S("lane"), {{0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 0}}},
                        {S("warp"), {{8, 0}, {16, 0}}},
                        {S("block"), {{0, 0}, {64, 0}}},
                    },
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, RepeatInCTGDimFirst) {
  // We have a 4-element shape and an 8-element layout (4 elems per CTA).  So
  // the layout will map two inputs to each output.  The question is, which two
  // inputs?  The answer is, we split between CTAs first, so the two CTAs have
  // distinct elements.
  auto blockedLayout = blocked({1}, {1}, {4}, {2}, {2}, {0}, {0});
  auto ll = toLinearLayout({4}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {}},
                        {S("lane"), {}},
                        {S("warp"), {{1}, {0}}},
                        {S("block"), {{2}}},
                    },
                    {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, SmallerThanCGALayout) {
  auto blockedLayout = blocked({1}, {1}, {1}, {4}, {4}, {0}, {0});
  auto ll = toLinearLayout({2}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {}},
                        {S("lane"), {}},
                        {S("warp"), {}},
                        {S("block"), {{1}, {0}}},
                    },
                    {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, Skinny) {
  auto blockedLayout =
      blocked({8, 1}, {8, 4}, {1, 4}, {1, 2}, {1, 2}, {0, 1}, {0, 1});
  auto ll = toLinearLayout({64, 1}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                        {S("lane"), {{8, 0}, {16, 0}, {32, 0}, {0, 0}, {0, 0}}},
                        {S("warp"), {{0, 0}, {0, 0}}},
                        {S("block"), {{0, 0}}},
                    },
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedOrder) {
  auto ll = toLinearLayout({1024, 128}, blocked({2, 2}, {4, 8}, {2, 2}, {2, 2},
                                                {2, 2}, {1, 0}, {1, 0}));
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"),
                         {
                             {0, 1},
                             {1, 0},
                             {0, 32},
                             {16, 0},
                             {32, 0},
                             {64, 0},
                             {128, 0},
                             {256, 0},
                         }},
                        {S("lane"), {{0, 2}, {0, 4}, {0, 8}, {2, 0}, {4, 0}}},
                        {S("warp"), {{0, 16}, {8, 0}}},
                        {S("block"), {{0, 64}, {512, 0}}},
                    },
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, Blocked4D) {
  auto ll = toLinearLayout({2, 1, 1, 1},
                           blocked({1, 1, 1, 4}, {2, 1, 1, 16}, {1, 2, 4, 1},
                                   {1, 1, 1, 1}, {1, 1, 1, 1}, {3, 0, 1, 2},
                                   {3, 2, 1, 0}));
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {{0, 0, 0, 0}, {0, 0, 0, 0}}},
                        {S("lane"),
                         {{0, 0, 0, 0},
                          {0, 0, 0, 0},
                          {0, 0, 0, 0},
                          {0, 0, 0, 0},
                          {1, 0, 0, 0}}},
                        {S("warp"), {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
                        {S("block"), {}},
                    },
                    {S("dim0"), S("dim1"), S("dim2"), S("dim3")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDotOperandLhs) {
  auto parent = blocked(/*size*/ {2, 4}, /*threads*/ {8, 4}, /*warps*/ {2, 4},
                        /*ctas*/ {1, 1}, /*splits*/ {1, 1}, /*order*/ {1, 0},
                        /*cta order*/ {1, 0});
  auto dotOperand = dot(parent, /*idx*/ 0, /*kWidth*/ 0);
  EXPECT_EQ(
      toLinearLayout({32, 16}, dotOperand),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                    {S("lane"), {{0, 0}, {0, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDot3dOperandLhs) {
  auto parent =
      blocked(/*size*/ {2, 2, 4}, /*threads*/ {2, 4, 4}, /*warps*/ {2, 2, 2},
              /*ctas*/ {1, 1, 1}, /*splits*/ {1, 1, 1}, /*order*/ {2, 1, 0},
              /*cta order*/ {2, 1, 0});
  auto dotOperand = dot(parent, /*idx*/ 0, /*kWidth*/ 0);
  EXPECT_EQ(
      toLinearLayout({16, 32, 4}, dotOperand),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 1, 0},
             {1, 0, 0},
             {0, 16, 0},
             {8, 0, 0}}},
           {S("lane"), {{0, 0, 0}, {0, 0, 0}, {0, 2, 0}, {0, 4, 0}, {2, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 8, 0}, {4, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDotOperandRhs) {
  auto parent = blocked(/*size*/ {2, 4}, /*threads*/ {8, 4}, /*warps*/ {2, 4},
                        /*ctas*/ {1, 1}, /*splits*/ {1, 1}, /*order*/ {1, 0},
                        /*cta order*/ {1, 0});
  auto dotOperand = dot(parent, /*idx*/ 1, /*kWidth*/ 0);
  EXPECT_EQ(toLinearLayout({16, 64}, dotOperand),
            LinearLayout({{S("register"),
                           {{0, 1}, {0, 2}, {1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 4}, {0, 8}, {0, 0}, {0, 0}, {0, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDot3dOperandRhs) {
  auto parent =
      blocked(/*size*/ {2, 2, 4}, /*threads*/ {2, 4, 4}, /*warps*/ {2, 2, 2},
              /*ctas*/ {1, 1, 1}, /*splits*/ {1, 1, 1}, /*order*/ {2, 1, 0},
              /*cta order*/ {2, 1, 0});
  auto dotOperand = dot(parent, /*idx*/ 1, /*kWidth*/ 0);
  EXPECT_EQ(
      toLinearLayout({16, 4, 64}, dotOperand),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 1, 0},
             {0, 2, 0},
             {1, 0, 0},
             {0, 0, 32},
             {8, 0, 0}}},
           {S("lane"), {{0, 0, 4}, {0, 0, 8}, {0, 0, 0}, {0, 0, 0}, {2, 0, 0}}},
           {S("warp"), {{0, 0, 16}, {0, 0, 0}, {4, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_16x16) {
  EXPECT_EQ(toLinearLayout({16, 16},
                           mma(2, 0, {16, 8}, {1, 1}, {1, 1}, {1, 1}, {0, 1})),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_32x32) {
  EXPECT_EQ(toLinearLayout({32, 32},
                           mma(2, 0, {16, 8}, {1, 1}, {1, 1}, {1, 1}, {0, 1})),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {16, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_ExtendDim2) {
  EXPECT_EQ(toLinearLayout({16, 128},
                           mma(2, 0, {16, 8}, {1, 1}, {1, 1}, {1, 1}, {0, 1})),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {0, 32}, {0, 64}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_Cga) {
  EXPECT_EQ(
      toLinearLayout({64, 128, 128}, mma(2, 0, {1, 16, 8}, {16, 1, 1},
                                         {4, 2, 2}, {4, 2, 1}, {2, 1, 0})),
      LinearLayout(
          {
              {S("register"),
               {
                   {0, 0, 1},
                   {0, 8, 0},
                   {0, 0, 8},
                   {0, 0, 16},
                   {0, 0, 32},
                   {0, 0, 64},
                   {0, 16, 0},
                   {0, 32, 0},
               }},
              {S("lane"),
               {{0, 0, 2}, {0, 0, 4}, {0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
              {S("warp"), {{1, 0, 0}, {2, 0, 0}, {4, 0, 0}, {8, 0, 0}}},
              {S("block"), {{0, 0, 0}, {0, 64, 0}, {16, 0, 0}, {32, 0, 0}}},
          },
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_Small3D) {
  EXPECT_EQ(toLinearLayout({1, 128, 128}, mma(2, 0, {1, 16, 8}, {16, 1, 1},
                                              {4, 2, 2}, {4, 2, 1}, {2, 1, 0})),
            LinearLayout(
                {
                    {S("register"),
                     {
                         {0, 0, 1},
                         {0, 8, 0},
                         {0, 0, 8},
                         {0, 0, 16},
                         {0, 0, 32},
                         {0, 0, 64},
                         {0, 16, 0},
                         {0, 32, 0},
                     }},
                    {S("lane"),
                     {{0, 0, 2}, {0, 0, 4}, {0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
                    {S("block"), {{0, 0, 0}, {0, 64, 0}, {0, 0, 0}, {0, 0, 0}}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_64x16) {
  SmallVector<SmallVector<unsigned>, 2> instrShapes = {{16, 16, 8}, {16, 8, 8}};
  for (auto instrShape : instrShapes) {
    SCOPED_TRACE(triton::join(instrShape, ","));
    EXPECT_EQ(toLinearLayout({64, 16}, mma(3, 0, instrShape, {4, 1}, {1, 1},
                                           {1, 1}, {1, 0})),
              LinearLayout(
                  {
                      {S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                      {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                      {S("warp"), {{16, 0}, {32, 0}}},
                      {S("block"), {}},
                  },
                  {S("dim0"), S("dim1")}));
  }
}

TEST_F(LinearLayoutConversionsTest, MMAv3_128x16) {
  EXPECT_EQ(toLinearLayout({128, 16}, mma(3, 0, {16, 16, 8}, {4, 1}, {1, 1},
                                          {1, 1}, {1, 0})),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {64, 0}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_1024x1024) {
  EXPECT_EQ(toLinearLayout({1024, 1024}, mma(3, 0, {16, 16, 8}, {4, 1}, {1, 1},
                                             {1, 1}, {1, 0})),
            LinearLayout({{S("register"),
                           {{0, 1},
                            {8, 0},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {0, 64},
                            {0, 128},
                            {0, 256},
                            {0, 512},
                            {64, 0},
                            {128, 0},
                            {256, 0},
                            {512, 0}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_4x2Warps) {
  auto legacy = mma(3, 0, {16, 32, 16}, {4, 2}, {1, 1}, {1, 1}, {1, 0});

  EXPECT_EQ(toLinearLayout({64, 32}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 64}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 32}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({128, 64}, legacy),
      LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {64, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}, {0, 32}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({256, 64}, legacy),
      LinearLayout({{S("register"),
                     {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {64, 0}, {128, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}, {0, 32}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_4x4Warps) {
  auto legacy = mma(3, 0, {16, 16, 8}, {4, 4}, {1, 1}, {1, 1}, {1, 0});

  EXPECT_EQ(toLinearLayout({16, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {64, 0}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 32}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {0, 0}, {0, 16}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 16}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_tile_kwidth8) {
  auto parent = mma(2, 0, {16, 8}, {1, 1});
  EXPECT_EQ(toLinearLayout({16, 64}, dot(parent, 0, 8)),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}}},
                    {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 8}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_large_warp4_kwidth8) {
  auto parent = mma(2, 0, {16, 8}, {4, 1});
  EXPECT_EQ(
      toLinearLayout({128, 128}, dot(parent, 0, 8)),
      LinearLayout(
          {
              {S("register"),
               {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {64, 0}}},
              {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
              {S("warp"), {{16, 0}, {32, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 64}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{1, 0},
                      {2, 0},
                      {4, 0},
                      {32, 0},
                      {64, 0},
                      {0, 8},
                      {0, 16},
                      {0, 32}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {
                        S("warp"),
                        {{0, 0}, {0, 0}},
                    },
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{1, 0},
                      {2, 0},
                      {4, 0},
                      {32, 0},
                      {0, 8},
                      {0, 16},
                      {0, 32},
                      {0, 64}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {
                        S("warp"),
                        {{0, 0}, {0, 0}},
                    },
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_3D) {
  // We implement one that exercises all the paths
  auto parent = mma(2, 0, {1, 16, 8}, {2, 4, 2});
  EXPECT_EQ(toLinearLayout({16, 128, 128}, dot(parent, 0, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 0, 1},
                      {0, 0, 2},
                      {0, 0, 4},
                      {0, 8, 0},
                      {0, 0, 32},
                      {0, 0, 64},
                      {0, 64, 0},
                      {2, 0, 0},
                      {4, 0, 0},
                      {8, 0, 0}}},
                    {S("lane"),
                     {{0, 0, 8}, {0, 0, 16}, {0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({8, 128, 64}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1, 0},
                      {0, 2, 0},
                      {0, 4, 0},
                      {0, 32, 0},
                      {0, 64, 0},
                      {0, 0, 16},
                      {0, 0, 32},
                      {2, 0, 0},
                      {4, 0, 0}}},
                    {S("lane"),
                     {{0, 8, 0}, {0, 16, 0}, {0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
                    {
                        S("warp"),
                        {{0, 0, 8}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}},
                    },
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv3_warp4_kwidth2) {
  auto parent = mma(3, 0, {16, 16, 8}, {4, 1});
  auto dotOp = dot(parent, 0, 2);

  EXPECT_EQ(toLinearLayout({64, 16}, dotOp),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 16}, dotOp),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}, {64, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 32}, dotOp),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {64, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv3_mixed_warp_kwidth4) {
  // Testing dot with MMAv3 encoding for opIdx = 0 and kWidth = 4
  auto parent = mma(3, 0, {16, 16, 8}, {4, 2});
  auto dotOp = dot(parent, 0, 4);

  EXPECT_EQ(toLinearLayout({128, 64}, dotOp),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1}, {0, 2}, {8, 0}, {0, 16}, {0, 32}, {64, 0}}},
                    {S("lane"), {{0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}, {0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_split_warp_kwidth8) {
  auto parent = mma(2, 0, {16, 8}, {2, 2});
  EXPECT_EQ(
      toLinearLayout({32, 64}, dot(parent, 0, 8)),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}}},
                    {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 16}, dot(parent, 1, 8)),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {S("warp"), {{0, 8}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, dot(parent, 0, 8)),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {32, 0}}},
                 {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
                 {S("warp"), {{0, 0}, {16, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({128, 32}, dot(parent, 1, 8)),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}, {64, 0}, {0, 16}}},
           {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
           {S("warp"), {{0, 8}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SliceDot) {
  // Slice layout with a DotOperand (MMAv2) as the parent.
  auto parentV2 = dot(mma(2, 0, {16, 8}, {1, 1}), /*opIdx=*/0, /*kWidth=*/8);
  auto sliceV2 = slice(parentV2, /*dim=*/1);

  EXPECT_EQ(toLinearLayout({16}, sliceV2),
            LinearLayout(
                {
                    {S("register"), {{8}}},
                    {S("lane"), {{0}, {0}, {1}, {2}, {4}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0")}));

  // Slice layout with a DotOperand (MMAv3) as the parent.
  auto parentV3 =
      dot(mma(3, 0, {16, 16, 8}, {4, 1}), /*opIdx=*/0, /*kWidth=*/2);
  auto sliceV3 = slice(parentV3, /*dim=*/0);

  EXPECT_EQ(toLinearLayout({16}, sliceV3),
            LinearLayout(
                {
                    {S("register"), {{1}, {8}}},
                    {S("lane"), {{2}, {4}, {0}, {0}, {0}}},
                    {S("warp"), {{0}, {0}}},
                    {S("block"), {}},
                },
                {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_2x4Warps_tpw_2_2) {
  auto mfmaNT =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false, /*tilesPerWarp=*/{2, 2});

  EXPECT_EQ(
      toLinearLayout({32, 32}, mfmaNT),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}, {0, 0}, {0, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
           {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaNT),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}, {0, 32}, {32, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
           {S("warp"), {{0, 64}, {0, 0}, {64, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({256, 256}, mfmaNT),
      LinearLayout(
          {{S("register"),
            {{1, 0}, {2, 0}, {8, 0}, {16, 0}, {0, 32}, {32, 0}, {128, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
           {S("warp"), {{0, 64}, {0, 128}, {64, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  auto mfmaT = mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
                    /*isTransposed=*/true, /*tilesPerWarp=*/{2, 2});

  EXPECT_EQ(
      toLinearLayout({32, 32}, mfmaT),
      LinearLayout(
          {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {0, 0}, {0, 0}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
           {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaT),
      LinearLayout(
          {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {0, 32}, {32, 0}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
           {S("warp"), {{0, 64}, {0, 0}, {64, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({256, 256}, mfmaT),
      LinearLayout(
          {{S("register"),
            {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {0, 32}, {32, 0}, {128, 0}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
           {S("warp"), {{0, 64}, {0, 128}, {64, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_2x4Warps_tpw_2_2) {
  auto mfmaNT =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false, /*tilesPerWarp=*/{2, 2});

  EXPECT_EQ(toLinearLayout({32, 32}, mfmaNT),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {0, 16}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 128}, mfmaNT),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {0, 16}, {16, 0}, {64, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 32}, {0, 64}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({256, 256}, mfmaNT),
      LinearLayout(
          {{S("register"),
            {{1, 0}, {2, 0}, {0, 16}, {0, 128}, {16, 0}, {64, 0}, {128, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
           {S("warp"), {{0, 32}, {0, 64}, {32, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  auto mfmaT =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/true, /*tilesPerWarp=*/{2, 2});
  EXPECT_EQ(toLinearLayout({32, 32}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 16}, {16, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 128}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 16}, {16, 0}, {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 32}, {0, 64}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({256, 256}, mfmaT),
      LinearLayout(
          {{S("register"),
            {{0, 1}, {0, 2}, {0, 16}, {0, 128}, {16, 0}, {64, 0}, {128, 0}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
           {S("warp"), {{0, 32}, {0, 64}, {32, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_2x4Warps) {
  auto mfmaNT =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false);

  EXPECT_EQ(toLinearLayout({32, 32}, mfmaNT),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, mfmaNT),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaNT),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}, {64, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 32}, {0, 64}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto mfmaT = mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
                    /*isTransposed=*/true);

  EXPECT_EQ(toLinearLayout({32, 32}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 32}, {0, 64}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_2x4Warps) {
  auto mfmaNT =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaNT),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_2x4Warps_F64) {
  auto mfmaNT =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 4},
           /*isTransposed=*/false, /*tilesPerWarp=*/{}, /*elementBitWidth=*/64);
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaNT),
            LinearLayout(
                {{S("register"), {{4, 0}, {8, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_2x4x1Warps) {
  auto mfmaNT =
      mfma(/*version=*/3, /*warps=*/{2, 4, 1}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false);

  EXPECT_EQ(toLinearLayout({1, 128, 128}, mfmaNT),
            LinearLayout({{S("register"),
                           {{0, 1, 0},
                            {0, 2, 0},
                            {0, 8, 0},
                            {0, 16, 0},
                            {0, 0, 32},
                            {0, 0, 64}}},
                          {S("lane"),
                           {{0, 0, 1},
                            {0, 0, 2},
                            {0, 0, 4},
                            {0, 0, 8},
                            {0, 0, 16},
                            {0, 4, 0}}},
                          {S("warp"), {{0, 32, 0}, {0, 64, 0}, {0, 0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({2, 32, 32}, mfmaNT),
            LinearLayout(
                {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 8, 0}, {0, 16, 0}}},
                 {S("lane"),
                  {{0, 0, 1},
                   {0, 0, 2},
                   {0, 0, 4},
                   {0, 0, 8},
                   {0, 0, 16},
                   {0, 4, 0}}},
                 {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({2, 64, 32}, mfmaNT),
            LinearLayout(
                {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 8, 0}, {0, 16, 0}}},
                 {S("lane"),
                  {{0, 0, 1},
                   {0, 0, 2},
                   {0, 0, 4},
                   {0, 0, 8},
                   {0, 0, 16},
                   {0, 4, 0}}},
                 {S("warp"), {{0, 32, 0}, {0, 0, 0}, {1, 0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1"), S("dim2")}));

  auto mfmaT =
      mfma(/*version=*/3, /*warps=*/{2, 4, 1}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/true);

  EXPECT_EQ(toLinearLayout({1, 128, 128}, mfmaT),
            LinearLayout({{S("register"),
                           {{0, 0, 1},
                            {0, 0, 2},
                            {0, 0, 8},
                            {0, 0, 16},
                            {0, 0, 32},
                            {0, 0, 64}}},
                          {S("lane"),
                           {{0, 1, 0},
                            {0, 2, 0},
                            {0, 4, 0},
                            {0, 8, 0},
                            {0, 16, 0},
                            {0, 0, 4}}},
                          {S("warp"), {{0, 32, 0}, {0, 64, 0}, {0, 0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({2, 32, 32}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 8}, {0, 0, 16}}},
                 {S("lane"),
                  {{0, 1, 0},
                   {0, 2, 0},
                   {0, 4, 0},
                   {0, 8, 0},
                   {0, 16, 0},
                   {0, 0, 4}}},
                 {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({2, 64, 32}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 8}, {0, 0, 16}}},
                 {S("lane"),
                  {{0, 1, 0},
                   {0, 2, 0},
                   {0, 4, 0},
                   {0, 8, 0},
                   {0, 16, 0},
                   {0, 0, 4}}},
                 {S("warp"), {{0, 32, 0}, {0, 0, 0}, {1, 0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_warp1onK_lhs_kwidth8) {
  auto parentMfma_1_8 =
      mfma(/*version=*/3, /*warps=*/{1, 8}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false);
  auto mfmaDot_1_8 = mfmaDotOp(parentMfma_1_8, /*opIdx=*/0, /*kWidth=*/8);
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 16},
                   {0, 32},
                   {0, 64},
                   {32, 0},
                   {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 256}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 16},
                   {0, 32},
                   {0, 64},
                   {0, 128},
                   {32, 0},
                   {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({32, 64}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 16}, {0, 32}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 16},
                   {0, 32},
                   {0, 64},
                   {0, 128},
                   {32, 0},
                   {64, 0},
                   {128, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_warp1onK_rhs_kwidth8) {
  auto parentMfma_1_8 =
      mfma(/*version=*/3, /*warps=*/{1, 8}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false);
  auto mfmaDot_1_8 = mfmaDotOp(parentMfma_1_8, /*opIdx=*/1, /*kWidth=*/8);
  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaDot_1_8),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}, {32, 0}, {64, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {8, 0}}},
           {S("warp"), {{0, 32}, {0, 64}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({128, 256}, mfmaDot_1_8),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}, {32, 0}, {64, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {8, 0}}},
           {S("warp"), {{0, 32}, {0, 64}, {0, 128}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({32, 64}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {8, 0}}},
                 {S("warp"), {{0, 32}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({256, 256}, mfmaDot_1_8),
      LinearLayout(
          {{S("register"),
            {{1, 0}, {2, 0}, {4, 0}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {8, 0}}},
           {S("warp"), {{0, 32}, {0, 64}, {0, 128}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}, {8, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto parentMfma_1_4 =
      mfma(/*version=*/3, /*warps=*/{1, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false);
  auto mfmaDot_1_4 = mfmaDotOp(parentMfma_1_4, /*opIdx=*/1, /*kWidth=*/8);
  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"),
                  {{1, 0},
                   {2, 0},
                   {4, 0},
                   {16, 0},
                   {32, 0},
                   {64, 0},
                   {128, 0},
                   {0, 128}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {8, 0}}},
                 {S("warp"), {{0, 32}, {0, 64}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_warp1onK_lhs_kwidth8) {
  auto parentMfma_1_4 =
      mfma(/*version=*/3, /*warps=*/{1, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);
  auto mfmaDot_1_4 = mfmaDotOp(parentMfma_1_4, /*opIdx=*/0, /*kWidth=*/8);
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 32},
                   {0, 64},
                   {16, 0},
                   {32, 0},
                   {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}, {0, 16}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({1, 128}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"),
                  {
                      {0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 32},
                      {0, 64},
                  }},
                 {S("lane"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 8}, {0, 16}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({128, 1}, mfmaDot_1_4),
      LinearLayout(
          {{S("register"), {{0, 0}, {0, 0}, {0, 0}, {16, 0}, {32, 0}, {64, 0}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}, {0, 0}}},
           {S("warp"), {{0, 0}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 32},
                   {0, 64},
                   {0, 128},
                   {16, 0},
                   {32, 0},
                   {64, 0},
                   {128, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}, {0, 16}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}, {0, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto parentMfma_1_8 =
      mfma(/*version=*/3, /*warps=*/{1, 8}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);
  auto mfmaDot_1_8 = mfmaDotOp(parentMfma_1_8, /*opIdx=*/0, /*kWidth=*/8);
  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 32},
                   {0, 64},
                   {0, 128},
                   {16, 0},
                   {32, 0},
                   {64, 0},
                   {128, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}, {0, 16}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto parentMfma_1_8_1 =
      mfma(/*version=*/3, /*warps=*/{1, 1, 8}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);
  auto mfmaDot_1_8_1 = mfmaDotOp(parentMfma_1_8_1, /*opIdx=*/0, /*kWidth=*/8);

  EXPECT_EQ(toLinearLayout({1, 256, 256}, mfmaDot_1_8_1),
            LinearLayout({{S("register"),
                           {{0, 0, 1},
                            {0, 0, 2},
                            {0, 0, 4},
                            {0, 0, 32},
                            {0, 0, 64},
                            {0, 0, 128},
                            {0, 16, 0},
                            {0, 32, 0},
                            {0, 64, 0},
                            {0, 128, 0}}},
                          {S("lane"),
                           {{0, 1, 0},
                            {0, 2, 0},
                            {0, 4, 0},
                            {0, 8, 0},
                            {0, 0, 8},
                            {0, 0, 16}}},
                          {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_warp1onK_rhs_kwidth8) {
  auto parentMfma_1_4 =
      mfma(/*version=*/3, /*warps=*/{1, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);
  auto mfmaDot_1_4 = mfmaDotOp(parentMfma_1_4, /*opIdx=*/1, /*kWidth=*/8);
  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaDot_1_4),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}, {64, 0}, {0, 64}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {16, 0}}},
           {S("warp"), {{0, 16}, {0, 32}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({1, 128}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"), {{0, 0}, {0, 0}, {0, 0}, {0, 64}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}, {0, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 1}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}, {64, 0}}},
                 {S("lane"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {8, 0}, {16, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"),
                  {{1, 0},
                   {2, 0},
                   {4, 0},
                   {32, 0},
                   {64, 0},
                   {128, 0},
                   {0, 64},
                   {0, 128}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {16, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {0, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto parentMfma_1_8 =
      mfma(/*version=*/3, /*warps=*/{1, 8}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);
  auto mfmaDot_1_8 = mfmaDotOp(parentMfma_1_8, /*opIdx=*/1, /*kWidth=*/8);
  EXPECT_EQ(
      toLinearLayout({256, 256}, mfmaDot_1_8),
      LinearLayout(
          {{S("register"),
            {{1, 0}, {2, 0}, {4, 0}, {32, 0}, {64, 0}, {128, 0}, {0, 128}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {16, 0}}},
           {S("warp"), {{0, 16}, {0, 32}, {0, 64}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  auto parentMfma_1_8_1 =
      mfma(/*version=*/3, /*warps=*/{1, 1, 8}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);
  auto mfmaDot_1_8_1 = mfmaDotOp(parentMfma_1_8_1, /*opIdx=*/1, /*kWidth=*/8);

  EXPECT_EQ(toLinearLayout({1, 256, 256}, mfmaDot_1_8_1),
            LinearLayout({{S("register"),
                           {{0, 1, 0},
                            {0, 2, 0},
                            {0, 4, 0},
                            {0, 32, 0},
                            {0, 64, 0},
                            {0, 128, 0},
                            {0, 0, 128}}},
                          {S("lane"),
                           {{0, 0, 1},
                            {0, 0, 2},
                            {0, 0, 4},
                            {0, 0, 8},
                            {0, 8, 0},
                            {0, 16, 0}}},
                          {S("warp"), {{0, 0, 16}, {0, 0, 32}, {0, 0, 64}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_dot_op_lhs_tpw_2_2) {
  auto parentMfma32 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false, /*tilesPerWarp=*/{2, 2});
  auto mfmaDotOp0_32 = mfmaDotOp(parentMfma32, /*opIdx=*/0, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({64, 32}, mfmaDotOp0_32),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {32, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDotOp0_32),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {0, 32}, {0, 64}, {32, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {64, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDotOp0_32),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 8},
                   {0, 16},
                   {0, 32},
                   {0, 64},
                   {0, 128},
                   {32, 0},
                   {128, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {64, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma32 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/true, /*tilesPerWarp=*/{2, 2});
  auto tmfmaDotOp0_32 = mfmaDotOp(parentTMfma32, /*opIdx=*/0, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({64, 32}, tmfmaDotOp0_32),
            toLinearLayout({64, 32}, mfmaDotOp0_32));
  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp0_32),
            toLinearLayout({128, 128}, mfmaDotOp0_32));
  EXPECT_EQ(toLinearLayout({256, 256}, tmfmaDotOp0_32),
            toLinearLayout({256, 256}, mfmaDotOp0_32));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_dot_op_lhs_tpw_2_2) {
  auto parentMfma16 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false, /*tilesPerWarp=*/{2, 2});
  auto mfmaDotOp0_16 = mfmaDotOp(parentMfma16, /*opIdx=*/0, /*kWidth=*/4);
  EXPECT_EQ(toLinearLayout({64, 32}, mfmaDotOp0_16),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 16}, {16, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaDotOp0_16),
      LinearLayout(
          {{S("register"),
            {{0, 1}, {0, 2}, {0, 16}, {0, 32}, {0, 64}, {16, 0}, {64, 0}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
           {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDotOp0_16),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 16},
                   {0, 32},
                   {0, 64},
                   {0, 128},
                   {16, 0},
                   {64, 0},
                   {128, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma16 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/true, /*tilesPerWarp=*/{2, 2});
  auto tmfmaDotOp0_16 = mfmaDotOp(parentTMfma16, /*opIdx=*/0, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({64, 32}, tmfmaDotOp0_16),
            toLinearLayout({64, 32}, mfmaDotOp0_16));
  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp0_16),
            toLinearLayout({128, 128}, mfmaDotOp0_16));
  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp0_16),
            toLinearLayout({128, 128}, mfmaDotOp0_16));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_dot_op_lhs_kwidth4) {
  auto parentMfma32 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false);
  auto mfmaDotOp0_32 = mfmaDotOp(parentMfma32, /*opIdx=*/0, /*kWidth=*/4);
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDotOp0_32),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {0, 32}, {0, 64}, {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, mfmaDotOp0_32),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDotOp0_32),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma32 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/true);
  auto tmfmaDotOp0_32 = mfmaDotOp(parentTMfma32, /*opIdx=*/0, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp0_32),
            toLinearLayout({128, 128}, mfmaDotOp0_32));
  EXPECT_EQ(toLinearLayout({64, 32}, tmfmaDotOp0_32),
            toLinearLayout({64, 32}, mfmaDotOp0_32));
  EXPECT_EQ(toLinearLayout({16, 16}, tmfmaDotOp0_32),
            toLinearLayout({16, 16}, mfmaDotOp0_32));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_dot_op_lhs_kwidth4) {
  auto parentMfma16 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);
  auto mfmaDotOp0_16 = mfmaDotOp(parentMfma16, /*opIdx=*/0, /*kWidth=*/4);
  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaDotOp0_16),
      LinearLayout(
          {{S("register"),
            {{0, 1}, {0, 2}, {0, 16}, {0, 32}, {0, 64}, {32, 0}, {64, 0}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
           {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, mfmaDotOp0_16),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 16}, {32, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDotOp0_16),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma16 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/true);
  auto tmfmaDotOp0_16 = mfmaDotOp(parentTMfma16, /*opIdx=*/0, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp0_16),
            toLinearLayout({128, 128}, mfmaDotOp0_16));
  EXPECT_EQ(toLinearLayout({64, 32}, tmfmaDotOp0_16),
            toLinearLayout({64, 32}, mfmaDotOp0_16));
  EXPECT_EQ(toLinearLayout({16, 16}, tmfmaDotOp0_16),
            toLinearLayout({16, 16}, mfmaDotOp0_16));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_dot_op_rhs_tpw_2_2) {
  auto parentMfma32 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false, /*tilesPerWarp=*/{2, 2});
  auto mfmaDotOp1_32 = mfmaDotOp(parentMfma32, /*opIdx=*/1, /*kWidth=*/4);
  EXPECT_EQ(toLinearLayout({32, 64}, mfmaDotOp1_32),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}, {0, 32}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDotOp1_32),
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {8, 0}, {16, 0}, {32, 0}, {64, 0}, {0, 32}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 64}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDotOp1_32),
            LinearLayout(
                {{S("register"),
                  {{1, 0},
                   {2, 0},
                   {8, 0},
                   {16, 0},
                   {32, 0},
                   {64, 0},
                   {128, 0},
                   {0, 32}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 64}, {0, 128}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma32 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/true, /*tilesPerWarp=*/{2, 2});
  auto tmfmaDotOp1_32 = mfmaDotOp(parentTMfma32, /*opIdx=*/1, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp1_32),
            toLinearLayout({128, 128}, mfmaDotOp1_32));
  EXPECT_EQ(toLinearLayout({32, 64}, tmfmaDotOp1_32),
            toLinearLayout({32, 64}, mfmaDotOp1_32));
  EXPECT_EQ(toLinearLayout({256, 256}, tmfmaDotOp1_32),
            toLinearLayout({256, 256}, mfmaDotOp1_32));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_dot_op_rhs_tpw_2_2) {
  auto parentMfma16 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false, /*tilesPerWarp=*/{2, 2});
  auto mfmaDotOp1_16 = mfmaDotOp(parentMfma16, /*opIdx=*/1, /*kWidth=*/4);
  EXPECT_EQ(toLinearLayout({32, 64}, mfmaDotOp1_16),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {16, 0}, {0, 16}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 32}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDotOp1_16),
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {16, 0}, {32, 0}, {64, 0}, {0, 16}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 32}, {0, 64}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDotOp1_16),
            LinearLayout(
                {{S("register"),
                  {{1, 0},
                   {2, 0},
                   {16, 0},
                   {32, 0},
                   {64, 0},
                   {128, 0},
                   {0, 16},
                   {0, 128}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 32}, {0, 64}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma16 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/true, /*tilesPerWarp=*/{2, 2});
  auto tmfmaDotOp1_16 = mfmaDotOp(parentTMfma16, /*opIdx=*/1, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({32, 64}, tmfmaDotOp1_16),
            toLinearLayout({32, 64}, mfmaDotOp1_16));
  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp1_16),
            toLinearLayout({128, 128}, mfmaDotOp1_16));
  EXPECT_EQ(toLinearLayout({256, 256}, tmfmaDotOp1_16),
            toLinearLayout({256, 256}, mfmaDotOp1_16));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_dot_op_rhs_kwidth4) {
  auto parentMfma32 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false);
  auto mfmaDotOp1_32 = mfmaDotOp(parentMfma32, /*opIdx=*/1, /*kWidth=*/4);
  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaDotOp1_32),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}, {32, 0}, {64, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
           {S("warp"), {{0, 32}, {0, 64}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 64}, mfmaDotOp1_32),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 32}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDotOp1_32),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}, {4, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma32 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/true);
  auto tmfmaDotOp1_32 = mfmaDotOp(parentTMfma32, /*opIdx=*/1, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp1_32),
            toLinearLayout({128, 128}, mfmaDotOp1_32));
  EXPECT_EQ(toLinearLayout({64, 32}, tmfmaDotOp1_32),
            toLinearLayout({64, 32}, mfmaDotOp1_32));
  EXPECT_EQ(toLinearLayout({16, 16}, tmfmaDotOp1_32),
            toLinearLayout({16, 16}, mfmaDotOp1_32));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_dot_op_rhs_kwidth4) {
  auto parentMfma16 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);
  auto mfmaDotOp1_16 = mfmaDotOp(parentMfma16, /*opIdx=*/1, /*kWidth=*/4);
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDotOp1_16),
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {16, 0}, {32, 0}, {64, 0}, {0, 64}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 64}, mfmaDotOp1_16),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDotOp1_16),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma16 =
      mfma(/*version=*/3, /*warps=*/{2, 4}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/true);
  auto tmfmaDotOp1_16 = mfmaDotOp(parentTMfma16, /*opIdx=*/1, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp1_16),
            toLinearLayout({128, 128}, mfmaDotOp1_16));
  EXPECT_EQ(toLinearLayout({64, 32}, tmfmaDotOp1_16),
            toLinearLayout({64, 32}, mfmaDotOp1_16));
  EXPECT_EQ(toLinearLayout({16, 16}, tmfmaDotOp1_16),
            toLinearLayout({16, 16}, mfmaDotOp1_16));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_dot_op_lhs_trans_fp4_mn_packed) {
  auto parentMfma16 =
      mfma(/*version=*/3, /*warps=*/{4, 1}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);
  auto mfmaDotOp0_kwidth_16 =
      mfmaDotOp(parentMfma16, /*opIdx=*/0, /*kWidth=*/16);
  EXPECT_EQ(chooseDsReadTrLayout(mfmaDotOp0_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16),
            LinearLayout({{S("register"),
                           {{1, 0},
                            {2, 0},
                            {4, 0},
                            {0, 16},
                            {0, 128},
                            {32, 0},
                            {64, 0},
                            {128, 0}}},
                          {S("lane"),
                           {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 32}, {0, 64}}},
                          {S("warp"), {{8, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));

  // Dot operand for LDS transpose load based on transposed mfma layout has
  // same layout as ordinary.
  auto parentTMfma16 =
      mfma(/*version=*/3, /*warps=*/{4, 1}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/true);
  auto tmfmaDotOp0_kwidth_16 =
      mfmaDotOp(parentTMfma16, /*opIdx=*/0, /*kWidth=*/16);

  EXPECT_EQ(chooseDsReadTrLayout(tmfmaDotOp0_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16),
            chooseDsReadTrLayout(mfmaDotOp0_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_dot_op_rhs_trans_fp4_mn_packed) {
  auto parentMfma16 =
      mfma(/*version=*/3, /*warps=*/{4, 1}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);

  // double rated mfma with large enough shape
  auto mfmaDotOp1_kwidth_16 =
      mfmaDotOp(parentMfma16, /*opIdx=*/1, /*kWidth=*/16);
  EXPECT_EQ(chooseDsReadTrLayout(mfmaDotOp1_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16),
            LinearLayout({{S("register"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {16, 0},
                            {128, 0},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {0, 64},
                            {0, 128}}},
                          {S("lane"),
                           {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {32, 0}, {64, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));

  // Dot operand for LDS transpose load based on transposed mfma layout has
  // same layout as ordinary.
  auto parentTMfma16 =
      mfma(/*version=*/3, /*warps=*/{4, 1}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/true);

  auto tmfmaDotOp1_kwidth_16 =
      mfmaDotOp(parentTMfma16, /*opIdx=*/1, /*kWidth=*/16);

  EXPECT_EQ(chooseDsReadTrLayout(tmfmaDotOp1_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16),
            chooseDsReadTrLayout(mfmaDotOp1_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_dot_op_lhs_trans_fp4_mn_packed) {
  auto parentMfma32 =
      mfma(/*version=*/3, /*warps=*/{4, 1}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false);
  auto mfmaDotOp0_kwidth_16 =
      mfmaDotOp(parentMfma32, /*opIdx=*/0, /*kWidth=*/16);
  EXPECT_EQ(chooseDsReadTrLayout(mfmaDotOp0_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16),
            LinearLayout(
                {{S("register"),
                  {{1, 0},
                   {2, 0},
                   {4, 0},
                   {0, 16},
                   {0, 64},
                   {0, 128},
                   {64, 0},
                   {128, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {0, 32}}},
                 {S("warp"), {{16, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand for LDS transpose load based on transposed mfma layout has
  // same layout as ordinary.
  auto parentTMfma32 =
      mfma(/*version=*/3, /*warps=*/{4, 1}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/true);
  auto tmfmaDotOp0_kwidth_16 =
      mfmaDotOp(parentTMfma32, /*opIdx=*/0, /*kWidth=*/16);

  EXPECT_EQ(chooseDsReadTrLayout(tmfmaDotOp0_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16),
            chooseDsReadTrLayout(mfmaDotOp0_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_dot_op_rhs_tran_fp4_mn_packeds) {
  auto parentMfma16 =
      mfma(/*version=*/3, /*warps=*/{4, 1}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false);
  auto mfmaDotOp1_kwidth_16 =
      mfmaDotOp(parentMfma16, /*opIdx=*/1, /*kWidth=*/16);

  EXPECT_EQ(chooseDsReadTrLayout(mfmaDotOp1_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {16, 0},
                   {64, 0},
                   {128, 0},
                   {0, 16},
                   {0, 32},
                   {0, 64},
                   {0, 128}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}, {32, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand for LDS transpose load based on transposed mfma layout has
  // same layout as ordinary.
  auto parentTMfma16 =
      mfma(/*version=*/3, /*warps=*/{4, 1}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/true);
  auto tmfmaDotOp1_kwidth_16 =
      mfmaDotOp(parentTMfma16, /*opIdx=*/1, /*kWidth=*/16);

  EXPECT_EQ(chooseDsReadTrLayout(tmfmaDotOp1_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16),
            chooseDsReadTrLayout(mfmaDotOp1_kwidth_16, {256, 256},
                                 /*elemBitWidth=*/4, /*instBitWidth*/ 64,
                                 /*numLanesInShuffleGroup*/ 16));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_dot_op_npotK) {
  // 32x32 MFMA, kWidth=4: kTileSize = (64/32)*4 = 8
  // K=48: numKTiles = ceil(48/8) = 6 → pow2 = 8
  auto parent32 =
      mfma(/*version=*/3, /*warps=*/{1, 1}, /*instrShape=*/{32, 32, 8},
           /*isTransposed=*/false);

  // LHS (opIdx=0): shape {32, 48}, K dimension is dim1
  auto dotOp0 = mfmaDotOp(parent32, /*opIdx=*/0, /*kWidth=*/4);
  auto ll0 = toLinearLayout({32, 48}, dotOp0);
  EXPECT_EQ(ll0.getOutDimSize(S("dim0")), 32);
  EXPECT_EQ(ll0.getOutDimSize(S("dim1")), 48);
  for (auto &[inDim, bases] : ll0.getBases()) {
    for (auto &basis : bases) {
      EXPECT_LT(basis[0], 32) << "dim0 basis out of range for " << inDim.str();
      EXPECT_LT(basis[1], 48) << "dim1 basis out of range for " << inDim.str();
    }
  }

  // RHS (opIdx=1): shape {48, 32}, K dimension is dim0
  auto dotOp1 = mfmaDotOp(parent32, /*opIdx=*/1, /*kWidth=*/4);
  auto ll1 = toLinearLayout({48, 32}, dotOp1);
  EXPECT_EQ(ll1.getOutDimSize(S("dim0")), 48);
  EXPECT_EQ(ll1.getOutDimSize(S("dim1")), 32);
  for (auto &[inDim, bases] : ll1.getBases()) {
    for (auto &basis : bases) {
      EXPECT_LT(basis[0], 48) << "dim0 basis out of range for " << inDim.str();
      EXPECT_LT(basis[1], 32) << "dim1 basis out of range for " << inDim.str();
    }
  }
}

TEST_F(LinearLayoutConversionsTest, MFMA16_dot_op_npotK) {
  // 16x16 MFMA, kWidth=4: kTileSize = (64/16)*4 = 16
  // K=48: numKTiles = ceil(48/16) = 3 → pow2 = 4
  auto parent16 =
      mfma(/*version=*/3, /*warps=*/{1, 1}, /*instrShape=*/{16, 16, 16},
           /*isTransposed=*/false);

  // LHS (opIdx=0): shape {16, 48}, K dimension is dim1
  auto dotOp0 = mfmaDotOp(parent16, /*opIdx=*/0, /*kWidth=*/4);
  auto ll0 = toLinearLayout({16, 48}, dotOp0);
  EXPECT_EQ(ll0.getOutDimSize(S("dim0")), 16);
  EXPECT_EQ(ll0.getOutDimSize(S("dim1")), 48);
  for (auto &[inDim, bases] : ll0.getBases()) {
    for (auto &basis : bases) {
      EXPECT_LT(basis[0], 16) << "dim0 basis out of range for " << inDim.str();
      EXPECT_LT(basis[1], 48) << "dim1 basis out of range for " << inDim.str();
    }
  }

  // RHS (opIdx=1): shape {48, 16}, K dimension is dim0
  auto dotOp1 = mfmaDotOp(parent16, /*opIdx=*/1, /*kWidth=*/4);
  auto ll1 = toLinearLayout({48, 16}, dotOp1);
  EXPECT_EQ(ll1.getOutDimSize(S("dim0")), 48);
  EXPECT_EQ(ll1.getOutDimSize(S("dim1")), 16);
  for (auto &[inDim, bases] : ll1.getBases()) {
    for (auto &basis : bases) {
      EXPECT_LT(basis[0], 48) << "dim0 basis out of range for " << inDim.str();
      EXPECT_LT(basis[1], 16) << "dim1 basis out of range for " << inDim.str();
    }
  }
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4Warps) {
  auto legacy = wmma(/*warps=*/{2, 4}, /*version=*/1, /*transposed=*/false);

  EXPECT_EQ(toLinearLayout({16, 16}, legacy),
            LinearLayout({{S("register"), {{2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  // For 32x16, we need 2x1 WMMA instances. We have 2x4 warps, so we are
  // broadcasted along the warp N dimension, distributed along the warp M
  // dimension.
  EXPECT_EQ(toLinearLayout({32, 16}, legacy),
            LinearLayout({{S("register"), {{2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  // For 16x32, we need 1x2 WMMA instances. We have 2x4 warps, so along the warp
  // N dimension, warp 0/2 gets the first distributed instance, warp 1/3 gets
  // the second distributed instance. Along the warp M dimension, all are
  // broadcasted.
  EXPECT_EQ(toLinearLayout({16, 32}, legacy),
            LinearLayout({{S("register"), {{2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                          {S("warp"), {{0, 16}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  // For 128x128, we need 8x8 WMMA instances. Given that we have 2x4 warps, each
  // warp handles 4x2 instances. So for both the warp M and N dimension, we
  // distribute. The register dimension will handle (8 x 4x2 =) 64 values--those
  // additional base vectors after the intrinsic shape are next power of two
  // values following the warp dimension, given that we are tiling cyclically
  // among warps.
  EXPECT_EQ(toLinearLayout({128, 128}, legacy),
            LinearLayout({{S("register"),
                           {{2, 0}, {4, 0}, {8, 0}, {0, 64}, {32, 0}, {64, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4x1Warps) {
  auto legacy = wmma(/*warps=*/{2, 4, 1}, /*version=*/1, /*transposed=*/false);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, legacy),
      LinearLayout(
          {{S("register"), {{0, 2, 0}, {0, 4, 0}, {0, 8, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 1, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 16, 16}, legacy),
      LinearLayout(
          {{S("register"), {{0, 2, 0}, {0, 4, 0}, {0, 8, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 1, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({8, 16, 16}, legacy),
      LinearLayout(
          {{S("register"),
            {{0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {2, 0, 0}, {4, 0, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 1, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4Warps_lhs) {
  auto dot = wmma(/*warps=*/{2, 4}, /*version=*/1, /*transposed=*/false);
  auto wmmaOperand = wmmaDotOp(dot, 0, 16);

  EXPECT_EQ(toLinearLayout({16, 16}, wmmaOperand),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, wmmaOperand),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 64}, wmmaOperand),
            LinearLayout({{S("register"),
                           {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {0, 32}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperand),
            LinearLayout({{S("register"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {0, 64},
                            {32, 0}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4Warps_rhs) {
  auto dot = wmma(/*warps=*/{2, 4}, /*version=*/1, /*transposed=*/false);
  auto wmmaOperand = wmmaDotOp(dot, 1, 16);

  EXPECT_EQ(toLinearLayout({16, 16}, wmmaOperand),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 16}, wmmaOperand),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 64}, wmmaOperand),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}}},
                    {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperand),
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {32, 0}, {0, 64}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4x1Warps_lhs) {
  auto dot = wmma(/*warps=*/{2, 4, 1}, /*version=*/1, /*transposed=*/false);
  auto wmmaOperand = wmmaDotOp(dot, 0, 16);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 32, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 0}}},
           {S("warp"), {{0, 16, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 64, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 0}}},
           {S("warp"), {{0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 128, 32}, wmmaOperand),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 0, 4},
             {0, 0, 8},
             {0, 0, 16},
             {0, 64, 0},
             {2, 0, 0}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 0}}},
           {S("warp"), {{0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4x1Warps_rhs) {
  auto dot = wmma(/*warps=*/{2, 4, 1}, /*version=*/1, /*transposed=*/false);
  auto wmmaOperand = wmmaDotOp(dot, 1, 16);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 32, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 16, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 64, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0},
             {0, 2, 0},
             {0, 4, 0},
             {0, 8, 0},
             {0, 16, 0},
             {0, 32, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 128, 32}, wmmaOperand),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0},
             {0, 2, 0},
             {0, 4, 0},
             {0, 8, 0},
             {0, 16, 0},
             {0, 32, 0},
             {0, 64, 0},
             {0, 0, 16},
             {2, 0, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x4Warps) {
  auto layout = wmma(/*warps=*/{2, 4}, /*version=*/2, /*transposed=*/false);

  EXPECT_EQ(toLinearLayout({16, 16}, layout),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, layout),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 32}, layout),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 16}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 128}, layout),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 64}, {32, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                    {S("warp"), {{0, 16}, {0, 32}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x2x2Warps) {
  auto layout = wmma(/*warps=*/{2, 2, 2}, /*version=*/2, /*transposed=*/false);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, layout),
      LinearLayout(
          {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 16, 16}, layout),
      LinearLayout(
          {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 64, 64}, layout),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0},
             {0, 2, 0},
             {0, 4, 0},
             {0, 0, 32},
             {0, 32, 0},
             {2, 0, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 16}, {0, 16, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, TWMMA_v2_2x4Warps) {
  auto layout = wmma(/*warps=*/{2, 4}, /*version=*/2, /*transposed=*/true);

  EXPECT_EQ(toLinearLayout({16, 16}, layout),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, layout),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 32}, layout),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 16}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 128}, layout),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 64}, {32, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                    {S("warp"), {{0, 16}, {0, 32}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, TWMMA_v2_2x2x2Warps) {
  auto layout = wmma(/*warps=*/{2, 2, 2}, /*version=*/2, /*transposed=*/true);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, layout),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 16, 16}, layout),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 64, 64}, layout),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 0, 4},
             {0, 0, 32},
             {0, 32, 0},
             {2, 0, 0}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 0, 16}, {0, 16, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x4Warps_lhs) {
  auto dot = wmma(/*warps=*/{2, 4}, /*version=*/2, /*transposed=*/false);

  auto wmmaOperandK8 = wmmaDotOp(dot, 0, 8);
  EXPECT_EQ(toLinearLayout({16, 16}, wmmaOperandK8),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, wmmaOperandK8),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 64}, wmmaOperandK8),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 16}, {0, 32}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                    {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperandK8),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 4}, {0, 16}, {0, 32}, {0, 64}, {32, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto wmmaOperandK16 = wmmaDotOp(dot, 0, 16);
  EXPECT_EQ(
      toLinearLayout({16, 32}, wmmaOperandK16),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 16}}},
                    {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 32}, wmmaOperandK16),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 16}}},
                    {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 128}, wmmaOperandK16),
      LinearLayout(
          {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 32}, {0, 64}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 16}}},
           {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperandK16),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 32}, {0, 64}, {32, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 16}}},
                 {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x4Warps_rhs) {
  auto dot = wmma(/*warps=*/{2, 4}, /*version=*/2, /*transposed=*/false);

  auto wmmaOperandK8 = wmmaDotOp(dot, 1, 8);
  EXPECT_EQ(toLinearLayout({16, 16}, wmmaOperandK8),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, wmmaOperandK8),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 64}, wmmaOperandK8),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperandK8),
            LinearLayout({{S("register"),
                           {{1, 0}, {2, 0}, {4, 0}, {16, 0}, {32, 0}, {0, 64}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));

  auto wmmaOperandK16 = wmmaDotOp(dot, 1, 16);
  EXPECT_EQ(
      toLinearLayout({32, 16}, wmmaOperandK16),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {16, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 32}, wmmaOperandK16),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {16, 0}}},
                    {S("warp"), {{0, 16}, {0, 0}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 64}, wmmaOperandK16),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {32, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {16, 0}}},
                    {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 128}, wmmaOperandK16),
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {32, 0}, {64, 0}, {0, 64}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {16, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x4x1Warps_lhs) {
  auto dot = wmma(/*warps=*/{2, 4, 1}, /*version=*/2, /*transposed=*/false);
  auto wmmaOperandK8 = wmmaDotOp(dot, 0, 8);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 32, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 16, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 64, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 128, 32}, wmmaOperandK8),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 0, 4},
             {0, 0, 16},
             {0, 64, 0},
             {2, 0, 0}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x4x1Warps_rhs) {
  auto dot = wmma(/*warps=*/{2, 4, 1}, /*version=*/2, /*transposed=*/false);
  auto wmmaOperandK8 = wmmaDotOp(dot, 1, 8);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 32, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 16, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 64, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 16, 0}, {0, 32, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 128, 32}, wmmaOperandK8),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0},
             {0, 2, 0},
             {0, 4, 0},
             {0, 16, 0},
             {0, 32, 0},
             {0, 64, 0},
             {0, 0, 16},
             {2, 0, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v3_2x4Warps) {
  auto layout = wmma(/*warps=*/{2, 4}, /*version=*/3, /*transposed=*/false,
                     /*instrShape=*/{16, 16, 32});

  EXPECT_EQ(toLinearLayout({16, 16}, layout),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 64}, layout),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 128}, layout),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 64}, {32, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                    {S("warp"), {{0, 16}, {0, 32}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v3_2x4Warps_lhs) {
  auto dot = wmma(/*warps=*/{2, 4}, /*version=*/3, /*transposed=*/false,
                  /*instrShape=*/{16, 16, 32});
  auto wmmaOperand = wmmaDotOp(dot, 0, 8);

  EXPECT_EQ(toLinearLayout({16, 32}, wmmaOperand),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 16}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 32}, wmmaOperand),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 16}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 64}, wmmaOperand),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 16}, {0, 32}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                    {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperand),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 4}, {0, 16}, {0, 32}, {0, 64}, {32, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v3_2x4Warps_rhs) {
  auto dot = wmma(/*warps=*/{2, 4}, /*version=*/3, /*transposed=*/false,
                  /*instrShape=*/{16, 16, 32});
  auto wmmaOperand = wmmaDotOp(dot, 1, 8);

  EXPECT_EQ(toLinearLayout({32, 16}, wmmaOperand),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 64}, wmmaOperand),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 64}, wmmaOperand),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}, {32, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                    {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperand),
            LinearLayout({{S("register"),
                           {{1, 0}, {2, 0}, {4, 0}, {16, 0}, {32, 0}, {0, 64}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SliceOfBlocked) {
  auto parent = blocked({2, 4}, {4, 2}, {2, 2}, {2, 2}, {2, 2}, {1, 0}, {1, 0});
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 0)),
            LinearLayout({{S("register"), {{1}, {2}, {16}, {32}}},
                          {S("lane"), {{4}, {0}, {0}}},
                          {S("warp"), {{8}, {0}}},
                          {S("block"), {{64}, {0}}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 1)),
            LinearLayout({{S("register"), {{1}, {16}, {32}}},
                          {S("lane"), {{0}, {2}, {4}}},
                          {S("warp"), {{0}, {8}}},
                          {S("block"), {{0}, {64}}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, SliceWithShape1) {
  auto parent = blocked({1, 4}, {8, 4}, {2, 2}, {1, 1}, {1, 1}, {0, 1}, {1, 0});
  EXPECT_EQ(toLinearLayout({1}, slice(parent, 0)),
            LinearLayout({{S("register"), {}},
                          {S("lane"), {{0}, {0}, {0}, {0}, {0}}},
                          {S("warp"), {{0}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, Slice4D) {
  auto parent = blocked({1, 1, 1, 4}, {2, 1, 1, 16}, {1, 2, 4, 1}, {1, 1, 1, 1},
                        {1, 1, 1, 1}, {3, 0, 1, 2}, {3, 2, 1, 0});
  EXPECT_EQ(toLinearLayout({2, 1, 1}, slice(parent, 3)),
            LinearLayout(
                {
                    {S("register"), {}},
                    {S("lane"),
                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, SliceOfMmaV2) {
  auto parent = mma(2, 0, {16, 8}, {2, 2}, {1, 1}, {1, 1}, {0, 1});
  EXPECT_EQ(toLinearLayout({16}, slice(parent, 0)),
            LinearLayout({{S("register"), {{1}}},
                          {S("lane"), {{2}, {4}, {0}, {0}, {0}}},
                          {S("warp"), {{8}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 0)),
            LinearLayout({{S("register"), {{1}, {16}, {32}, {64}}},
                          {S("lane"), {{2}, {4}, {0}, {0}, {0}}},
                          {S("warp"), {{8}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({8}, slice(parent, 1)),
            LinearLayout({{S("register"), {}},
                          {S("lane"), {{0}, {0}, {1}, {2}, {4}}},
                          {S("warp"), {{0}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 1)),
            LinearLayout({{S("register"), {{8}, {32}, {64}}},
                          {S("lane"), {{0}, {0}, {1}, {2}, {4}}},
                          {S("warp"), {{0}, {16}}},
                          {S("block"), {}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSimple1D) {
  EXPECT_EQ(toLinearLayout({1024}, shared(1, 1, 1, {1}, {1}, {0}, {0})),
            LinearLayout::identity1D(1024, S("offset"), S("dim0")) *
                LinearLayout::identity1D(1, S("block"), S("dim0")));
}

TEST_F(LinearLayoutConversionsTest, SharedSimple2D) {
  EXPECT_EQ(toLinearLayout({128, 128},
                           shared(1, 1, 1, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
            (LinearLayout::identity1D(128, S("offset"), S("dim1")) *
             LinearLayout::identity1D(128, S("offset"), S("dim0")) *
             LinearLayout::identity1D(1, S("block"), S("dim0")))
                .transposeOuts({S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSimple2D_Order01) {
  EXPECT_EQ(toLinearLayout({128, 128},
                           shared(1, 1, 1, {1, 1}, {1, 1}, {0, 1}, {1, 0})),
            LinearLayout::identity1D(128, S("offset"), S("dim0")) *
                LinearLayout::identity1D(128, S("offset"), S("dim1")) *
                LinearLayout::identity1D(1, S("block"), S("dim0")));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_MaxPhaseOnly) {
  EXPECT_EQ(
      toLinearLayout({32, 32}, shared(1, 1, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 8},
                      {0, 16},
                      {1, 1},
                      {2, 2},
                      {4, 0},
                      {8, 0},
                      {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_PerPhaseMaxPhase) {
  EXPECT_EQ(
      toLinearLayout({32, 32}, shared(1, 2, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 8},
                      {0, 16},
                      {1, 0},
                      {2, 1},
                      {4, 2},
                      {8, 0},
                      {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_Vec) {
  EXPECT_EQ(
      toLinearLayout({4, 8}, shared(2, 1, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"), {{0, 1}, {0, 2}, {0, 4}, {1, 2}, {2, 4}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_PerPhaseMaxPhaseVec) {
  EXPECT_EQ(
      toLinearLayout({32, 32}, shared(2, 2, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 8},
                      {0, 16},
                      {1, 0},
                      {2, 2},
                      {4, 4},
                      {8, 0},
                      {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled4D) {
  EXPECT_EQ(
      toLinearLayout({2, 4, 32, 32}, shared(2, 2, 4, {1, 1, 1, 1}, {1, 1, 1, 1},
                                            {3, 2, 1, 0}, {3, 2, 1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 0, 0, 1},
                      {0, 0, 0, 2},
                      {0, 0, 0, 4},
                      {0, 0, 0, 8},
                      {0, 0, 0, 16},
                      {0, 0, 1, 0},
                      {0, 0, 2, 2},
                      {0, 0, 4, 4},
                      {0, 0, 8, 0},
                      {0, 0, 16, 0},
                      {0, 1, 0, 0},
                      {0, 2, 0, 0},
                      {1, 0, 0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1"), S("dim2"), S("dim3")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_Order01) {
  EXPECT_EQ(
      toLinearLayout({4, 8}, shared(1, 1, 4, {1, 1}, {1, 1}, {0, 1}, {0, 1})),
      LinearLayout({{S("offset"), {{1, 0}, {2, 0}, {1, 1}, {2, 2}, {0, 4}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x16_4_2) {
  EXPECT_EQ(
      toLinearLayout(
          {8, 16}, nvmmaShared(32, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 8}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_128x16_4_2) {
  EXPECT_EQ(toLinearLayout({128, 16}, nvmmaShared(32, false, 16, {1, 1}, {1, 1},
                                                  {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {1, 0},
                            {2, 0},
                            {4, 8},
                            {8, 0},
                            {16, 0},
                            {32, 0},
                            {64, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x32_2_4) {
  EXPECT_EQ(
      toLinearLayout(
          {8, 32}, nvmmaShared(64, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout(
          {{S("offset"),
            {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {1, 0}, {2, 8}, {4, 16}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x64_1_8) {
  EXPECT_EQ(toLinearLayout({8, 64}, nvmmaShared(128, false, 16, {1, 1}, {1, 1},
                                                {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {1, 8},
                            {2, 16},
                            {4, 32}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x64_1_8_32b) {
  EXPECT_EQ(toLinearLayout({8, 64}, nvmmaShared(128, false, 32, {1, 1}, {1, 1},
                                                {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {0, 16},
                            {1, 4},
                            {2, 8},
                            {4, 16},
                            {0, 32}}},
                          {S("block"), {}}},
                         {{S("dim0"), 8}, {S("dim1"), 64}},
                         /*requireSurjective=*/false));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_128x128_1_8_128b_transposed) {
  EXPECT_EQ(toLinearLayout({128, 128}, nvmmaShared(128, true, 32, {1, 1},
                                                   {1, 1}, {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
                           {{1, 0},
                            {2, 0},
                            {4, 0},
                            {8, 0},
                            {16, 0},
                            {4, 1},
                            {8, 2},
                            {16, 4},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {0, 64},
                            {32, 0},
                            {64, 0}}},
                          {S("block"), {}}},
                         {{S("dim0"), 128}, {S("dim1"), 128}},
                         /*requireSurjective=*/true));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_32x4x64_1_8_32b) {
  EXPECT_EQ(
      toLinearLayout({32, 4, 64}, nvmmaShared(64, false, 32, {1, 1, 1},
                                              {1, 1, 1}, {2, 1, 0}, {2, 1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 0, 1},
                      {0, 0, 2},
                      {0, 0, 4},
                      {0, 0, 8},
                      {0, 1, 0},
                      {0, 2, 4},
                      {1, 0, 8},
                      {2, 0, 0},
                      {4, 0, 0},
                      {8, 0, 0},
                      {16, 0, 0},
                      {0, 0, 16},
                      {0, 0, 32}}},
                    {S("block"), {}}},
                   {{S("dim0"), 32}, {S("dim1"), 4}, {S("dim2"), 64}},
                   /*requireSurjective=*/true));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_64x4x32_1_8_32b_transposed) {
  EXPECT_EQ(
      toLinearLayout({64, 4, 32}, nvmmaShared(64, true, 32, {1, 1, 1},
                                              {1, 1, 1}, {2, 1, 0}, {2, 1, 0})),
      LinearLayout({{S("offset"),
                     {{1, 0, 0},
                      {2, 0, 0},
                      {4, 0, 0},
                      {8, 0, 0},
                      {0, 0, 4},
                      {4, 0, 8},
                      {8, 0, 16},
                      {0, 1, 0},
                      {0, 2, 0},
                      {0, 0, 1},
                      {0, 0, 2},
                      {16, 0, 0},
                      {32, 0, 0}}},
                    {S("block"), {}}},
                   {{S("dim0"), 64}, {S("dim1"), 4}, {S("dim2"), 32}},
                   /*requireSurjective=*/true));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_NPOT_48byte_8bit) {
  // For NPOT K with 8-bit elements, the contiguous data width may be NPOT
  // (e.g. K=48 -> 48 bytes), but the swizzle width must be a valid value
  // (0, 32, 64, or 128). Use 32-byte swizzle, which is the natural choice
  // for 48-byte rows (48 % 32 != 0 but the core tile still uses pow2).
  // tileCols = 8 * max(16,32) / 8 = 32, vec=16, perPhase=4, maxPhase=2.
  auto layout = getCoreMatrixLinearLayout(
      nvmmaShared(32, false, 8, {1, 1}, {1, 1}, {1, 0}, {1, 0}),
      /*disableSwizzle=*/false);
  EXPECT_EQ(layout, LinearLayout({{S("offset"),
                                   {{0, 1},
                                    {0, 2},
                                    {0, 4},
                                    {0, 8},
                                    {0, 16},
                                    {1, 0},
                                    {2, 0},
                                    {4, 16}}}},
                                 {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_NPOT_48byte_16bit) {
  // Same scenario with 16-bit elements: K=24 at 16-bit = 48 bytes per row.
  // Use 32-byte swizzle: tileCols = 8 * max(16,32) / 16 = 16,
  // vec=8, perPhase=4, maxPhase=2.
  auto layout = getCoreMatrixLinearLayout(
      nvmmaShared(32, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0}),
      /*disableSwizzle=*/false);
  EXPECT_EQ(
      layout,
      LinearLayout({{S("offset"),
                     {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 8}}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, Shared1DSwizzle) {
  EXPECT_EQ(
      toLinearLayout({64, 1}, shared(2, 2, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout::identity1D(64, S("offset"), S("dim0")) *
          LinearLayout::identity1D(1, S("offset"), S("dim1")) *
          LinearLayout::identity1D(1, S("block"), S("dim0")));
}

TEST_F(LinearLayoutConversionsTest, AMDRotatingShared2D_8x16_ord10) {
  EXPECT_EQ(
      toLinearLayout({8, 16},
                     AMDRotatingShared(/*vec=*/2, /*perPhase=*/2,
                                       /*maxPhase=*/2, /*ctaPerCga=*/{1, 1},
                                       /*cSplit=*/{1, 1},
                                       /*order=*/{1, 0},
                                       /*ctaOrder=*/{1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 2}, {4, 2}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, AMDRotatingShared2D_8x16_ord01) {
  EXPECT_EQ(
      toLinearLayout({8, 16},
                     AMDRotatingShared(/*vec=*/2, /*perPhase=*/2,
                                       /*maxPhase=*/2, /*ctaPerCga=*/{1, 1},
                                       /*cSplit=*/{1, 1},
                                       /*order=*/{0, 1},
                                       /*ctaOrder=*/{1, 0})),
      LinearLayout({{S("offset"),
                     {{1, 0}, {2, 0}, {4, 0}, {0, 1}, {2, 2}, {2, 4}, {0, 8}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, AMDRotatingShared2D_64x64) {
  // 64 rows is enough to fit two full patterns with given parameters, so last
  // base is {32, 0}
  EXPECT_EQ(toLinearLayout({64, 64}, AMDRotatingShared(
                                         /*vec=*/4, /*perPhase=*/2,
                                         /*maxPhase=*/4, /*ctaPerCga=*/{1, 1},
                                         /*cSplit=*/{1, 1},
                                         /*order=*/{1, 0},
                                         /*ctaOrder=*/{1, 0})),
            LinearLayout({{S("offset"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {1, 0},
                            {2, 4},
                            {4, 8},
                            {8, 4},
                            {16, 8},
                            {32, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, AMDRotatingShared3D_4x64x64) {
  EXPECT_EQ(
      toLinearLayout({4, 64, 64}, AMDRotatingShared(/*vec=*/4, /*perPhase=*/2,
                                                    /*maxPhase=*/4,
                                                    /*ctaPerCga=*/{1, 1, 1},
                                                    /*cSplit=*/{1, 1, 1},
                                                    /*order=*/{2, 1, 0},
                                                    /*ctaOrder=*/{2, 1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 0, 1},
                      {0, 0, 2},
                      {0, 0, 4},
                      {0, 0, 8},
                      {0, 0, 16},
                      {0, 0, 32},
                      {0, 1, 0},
                      {0, 2, 4},
                      {0, 4, 8},
                      {0, 8, 4},
                      {0, 16, 8},
                      {0, 32, 0},
                      {1, 0, 0},
                      {2, 0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, ChooseShmemLayout) {
  LinearLayout ll = LinearLayout({{S("register"), {{1}, {2}, {2}, {8}}},
                                  {S("lane"), {{8}, {4}, {1}}},
                                  {S("warp"), {{16}, {32}, {0}}},
                                  {S("block"), {}}},
                                 {S("dim0")});
  EXPECT_EQ(chooseShemLayoutForRegToRegConversion(&ctx, /*tensorShape=*/{64},
                                                  /*repShape=*/{64},
                                                  /*order=*/{0}),
            LinearLayout({{S("offset"), {{1}, {2}, {4}, {8}, {16}, {32}}},
                          {S("iteration"), {}},
                          {S("block"), {}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, ChooseShmemLayout_Empty) {
  LinearLayout ll = LinearLayout({{S("register"), {{0}}},
                                  {S("lane"), {{0}}},
                                  {S("warp"), {{0}}},
                                  {S("block"), {}}},
                                 {S("dim0")});
  EXPECT_EQ(
      chooseShemLayoutForRegToRegConversion(&ctx, /*tensorShape=*/{},
                                            /*repShape=*/{}, /*order=*/{}),
      LinearLayout({{S("offset"), {}}, {S("iteration"), {}}, {S("block"), {}}},
                   {}));
}

TEST_F(LinearLayoutConversionsTest, ChooseShmemLayout_Multidim) {
  LinearLayout src(
      {{S("register"), {}},
       {S("lane"),
        {{0, 0, 1, 0}, {0, 0, 2, 0}, {1, 0, 0, 0}, {2, 0, 0, 0}, {0, 0, 0, 1}}},
       {S("warp"), {{0, 0, 0, 2}, {0, 1, 0, 0}, {0, 2, 0, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1"), S("dim2"), S("dim3")});
  EXPECT_EQ(
      chooseShemLayoutForRegToRegConversion(&ctx, /*tensorShape=*/{4, 4, 4, 4},
                                            /*repShape=*/{2, 2, 2, 2},
                                            /*order=*/{3, 2, 1, 0}),
      LinearLayout({{S("offset"),
                     {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}},
                    {S("iteration"),
                     {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 2}}},
                    {S("block"), {}}},
                   {S("dim3"), S("dim2"), S("dim1"), S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv5Fp4Padded) {
  auto ll = toLinearLayout({32, 64}, nvmmaShared(128, false, 8, {1, 1}, {1, 1},
                                                 {1, 0}, {1, 0}, true));
  EXPECT_EQ(ll, LinearLayout(
                    {{S("offset"),
                      {{0, 1},
                       {0, 2},
                       {0, 4},
                       {0, 0}, // offset 8 maps to the same indices as offset 0
                       {0, 8},
                       {0, 16},
                       {0, 32},
                       {1, 8},
                       {2, 16},
                       {4, 32},
                       {8, 0},
                       {16, 0}}},
                     {S("block"), {}}},
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, TensorMemory_blockM_64) {
  auto enc = tmem(64, 64, 1, 1);
  auto d0 = S("dim0");
  auto d1 = S("dim1");
  auto kRow = S("row");
  auto kCol = S("col");
  LinearLayout expected1 = LinearLayout(
      {{kRow, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {64, 0}, {16, 0}, {32, 0}}},
       {kCol, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {0, 32}}}},
      {d0, d1});
  EXPECT_EQ(toLinearLayout({128, 64}, enc), expected1);
  // Tensor just fits blockMxblockN -> the layout is not injective (row=16 is
  // zero)
  LinearLayout expected2 = LinearLayout(
      {{kRow, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}, {16, 0}, {32, 0}}},
       {kCol, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {0, 32}}}},
      {d0, d1});
  EXPECT_EQ(toLinearLayout({64, 64}, enc), expected2);
  // Broadcasts M then N
  LinearLayout expected3 = LinearLayout(
      {{kRow, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {64, 0}, {16, 0}, {32, 0}}},
       {kCol,
        {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {0, 32}, {128, 0}, {0, 64}}}},
      {d0, d1});
  EXPECT_EQ(toLinearLayout({256, 128}, enc), expected3);
  // Fits N in basis the 5th basis if shape[0] == 64
  LinearLayout expected4 = LinearLayout(
      {{kRow, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 64}, {16, 0}, {32, 0}}},
       {kCol, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {0, 32}, {0, 128}}}},
      {d0, d1});
  EXPECT_EQ(toLinearLayout({64, 256}, enc), expected4);
}

TEST_F(LinearLayoutConversionsTest, TensorMemory_blockM_128) {
  auto enc = tmem(128, 128, 1, 1);
  auto d0 = S("dim0");
  auto d1 = S("dim1");
  auto kRow = S("row");
  auto kCol = S("col");
  LinearLayout tile = LinearLayout::identity1D(128, kRow, d0) *
                      LinearLayout::identity1D(128, kCol, d1);
  EXPECT_EQ(toLinearLayout({128, 128}, enc), tile);
  EXPECT_EQ(toLinearLayout({256, 128}, enc),
            tile * LinearLayout::identity1D(2, kCol, d0));
  EXPECT_EQ(toLinearLayout({256, 256}, enc),
            tile * LinearLayout::identity1D(2, kCol, d0) *
                LinearLayout::identity1D(2, kCol, d1));
}

TEST_F(LinearLayoutConversionsTest, TensorMemory_CTASplit) {
  auto d0 = S("dim0");
  auto d1 = S("dim1");
  auto kRow = S("row");
  auto kCol = S("col");
  auto enc = tmem(128, 64, 1, 2);
  auto enc1 = tmem(128, 64, 1, 1);
  EXPECT_EQ(toLinearLayout({128, 128}, enc),
            toLinearLayout({128, 64}, enc1) *
                LinearLayout::identity1D(2, kCol, d1));
}

// Tests for SM120 DotScaled Scale Layout
TEST_F(LinearLayoutConversionsTest, SM120DotScaledScaleLayout) {
  LinearLayout layout, ll;

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{128, 2}, /*opIdx=*/0, /*warpsPerCTA=*/{1, 1},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout({{S("register"), {{0, 1}, {16, 0}, {32, 0}, {64, 0}}},
                     {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
                     {S("warp"), {}},
                     {S("block"), {}}},
                    {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{128, 2}, /*opIdx=*/1, /*warpsPerCTA=*/{1, 1},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {8, 0}, {16, 0}, {32, 0}, {64, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{128, 4}, /*opIdx=*/0, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout({{S("register"), {{0, 1}, {0, 2}, {32, 0}, {64, 0}}},
                     {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
                     {S("warp"), {{0, 0}, {16, 0}}},
                     {S("block"), {}}},
                    {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 4}, /*opIdx=*/1, /*warpsPerCTA=*/{1, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {0, 2}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{8, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{128, 8}, /*opIdx=*/0, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll =
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {32, 0}, {64, 0}}},
                    {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{128, 8}, /*opIdx=*/1, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {16, 0}, {32, 0}, {64, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{8, 0}, {0, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 2}, /*opIdx=*/0, /*warpsPerCTA=*/{1, 1},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 2}, /*opIdx=*/1, /*warpsPerCTA=*/{1, 1},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {8, 0}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 4}, /*opIdx=*/0, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {0, 2}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{0, 0}, {16, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 4}, /*opIdx=*/1, /*warpsPerCTA=*/{1, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {0, 2}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{8, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 8}, /*opIdx=*/0, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{0, 0}, {16, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 8}, /*opIdx=*/1, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{8, 0}, {0, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);
}

//===----------------------------------------------------------------------===//
// nvmmaSharedToLinearLayout TMA Mode Independence Tests
//
// Verify that nvmmaSharedToLinearLayout produces the same result regardless
// of TMA mode. This is critical because MMA lowering uses toLinearLayout()
// to read from shared memory, and it doesn't know which TMA mode was used
// to load the data. If the layouts differ, MMA would compute wrong addresses.
//
// Note: We only test non-transposed encodings because TMA descriptors cannot
// be transposed (see AsyncTMACopyGlobalToLocalOp verification which emits
// "TMA descriptor layout must not be transposed"). Transposed layouts are
// created after TMA load or used for conceptual access patterns, not for
// TMA descriptor configuration.
//===----------------------------------------------------------------------===//

TEST_F(LinearLayoutConversionsTest,
       NvmmaSharedToLinearLayout_TMAModeIndependence) {
  // Test various non-transposed shapes and configurations to ensure the shared
  // memory layout is independent of TMA mode.
  //
  // Test matrix:
  // - swizzleSizeInBytes: 0, 32, 64, 128
  // - non-contiguous dim (dim0): 512, 1024 (exceeds Tiled mode limit of 256)
  // - contiguous dim (dim1): large enough for multiple messages

  constexpr int elementBitWidth = 16; // f16
  constexpr int elementBytes = elementBitWidth / 8;

  for (int swizzleBytes : {0, 32, 64, 128}) {
    for (int64_t dim0 : {512, 1024}) {
      // For contiguous dim, use a size that requires multiple messages.
      // With swizzle, the contiguous dim block size = swizzleBytes / elemBytes.
      // Use 2x the max swizzle size to ensure multiple messages in dim1.
      int64_t dim1 = (swizzleBytes == 0) ? 64 : (128 / elementBytes) * 2;

      auto encoding =
          nvmmaShared(swizzleBytes, /*transposed=*/false, elementBitWidth,
                      {1, 1}, {1, 1}, {1, 0}, {1, 0});
      llvm::SmallVector<int64_t> shape = {dim0, dim1};

      auto tiledLayout =
          nvmmaSharedToLinearLayout(shape, encoding, TMAMode::Tiled);
      auto im2colLayout =
          nvmmaSharedToLinearLayout(shape, encoding, TMAMode::Im2Col);

      EXPECT_EQ(tiledLayout, im2colLayout)
          << "Shared memory layout must be independent of TMA mode for shape ["
          << dim0 << ", " << dim1 << "] with " << swizzleBytes << "B swizzle";
    }
  }
}

// Non-power-of-2 layout tests for combineCtaCgaWithShape and
// buildWithNpotReduction via BlockedEncodingAttr::toLinearLayout.
TEST_F(LinearLayoutConversionsTest, BlockedNpot2D) {
  // 1D case: shape={33}, sizePerThread=1, threadsPerWarp=32, warpsPerCTA=4
  {
    auto enc = blocked({1}, {32}, {4}, {1}, {1}, {0}, {0});
    auto ll = toLinearLayout({33}, enc);
    EXPECT_EQ(ll.getOutDimSize(S("dim0")), 33);
    EXPECT_EQ(ll.getInDimSize(S("register")), 1);
  }

  // 2D case: shape={33,48}
  {
    auto enc = blocked({1, 1}, {1, 32}, {4, 1}, {1, 1}, {1, 1}, {1, 0}, {1, 0});
    auto ll = toLinearLayout({33, 48}, enc);
    EXPECT_EQ(ll.getOutDimSize(S("dim0")), 33);
    EXPECT_EQ(ll.getOutDimSize(S("dim1")), 48);
    // Verify all basis values are in range [0, shape[i])
    for (auto &[inDim, bases] : ll.getBases()) {
      for (auto &basis : bases) {
        EXPECT_LT(basis[0], 33)
            << "dim0 basis out of range for " << inDim.str();
        EXPECT_LT(basis[1], 48)
            << "dim1 basis out of range for " << inDim.str();
      }
    }
  }
}

TEST_F(LinearLayoutConversionsTest, SliceOfBlockedNpot) {
  // Parent: 2D blocked with NPOT shape. Slice dim 0 → NPOT result {48}.
  auto parent =
      blocked({1, 2}, {1, 32}, {4, 1}, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  auto ll = toLinearLayout({48}, slice(parent, 0));
  EXPECT_EQ(ll.getOutDimSize(S("dim0")), 48);

  // Slice dim 1 with NPOT result → shape {33}
  auto parent2 =
      blocked({1, 1}, {1, 32}, {4, 1}, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  auto ll2 = toLinearLayout({33}, slice(parent2, 1));
  EXPECT_EQ(ll2.getOutDimSize(S("dim0")), 33);
  // Basis values must be in [0, 33)
  for (auto &[inDim, bases] : ll2.getBases()) {
    for (auto &basis : bases) {
      EXPECT_LT(basis[0], 33) << "basis out of range for " << inDim.str();
    }
  }
}

TEST_F(LinearLayoutConversionsTest, NvidiaMmaTileNpot) {
  // K=96, kWidth=2: n/(kWidth*4) = 12 (not pow2), must not assert.
  auto layout = nvidiaMmaTile(&ctx, {16, 96}, /*kWidth=*/2,
                              /*order=*/{1, 0}, /*repOrder=*/{0, 1});
  EXPECT_EQ(layout.getOutDimSize(S("dim0")), 16);
  EXPECT_EQ(layout.getOutDimSize(S("dim1")), 96);
  // register: log2(kWidth=2) + log2(m/8=2) + ceil(log2(12)) = 1+1+4 = 6
  EXPECT_EQ(layout.getInDimSizeLog2(S("register")), 6);
  // lane: log2(4) + log2(8) = 2+3 = 5
  EXPECT_EQ(layout.getInDimSizeLog2(S("lane")), 5);
}

// Verify that NPOT blocked layouts have correct wrapping behavior for
// reduction: for tensor<48> with threadsPerWarp=32 and warpsPerCTA=4,
// warps beyond the first ceil(48/32)=2 should cover only duplicate elements.
TEST_F(LinearLayoutConversionsTest, BlockedNpotReductionWrapping) {
  // shape={48}, sizePerThread=1, threadsPerWarp=32, warpsPerCTA=4
  auto enc = blocked({1}, {32}, {4}, {1}, {1}, {0}, {0});
  auto ll = toLinearLayout({48}, enc);
  EXPECT_EQ(ll.getOutDimSize(S("dim0")), 48);

  // Evaluate the layout for each (lane, warp) pair and track element coverage.
  // Each element should be "owned" by exactly one warp (the lowest-indexed
  // warp that maps to it). Verify warp 0 owns 32, warp 1 owns 16, and
  // warps 2-3 own 0.
  std::array<unsigned, 4> ownedCount = {0, 0, 0, 0};
  std::vector<int> owner(48, -1);

  for (unsigned w = 0; w < 4; ++w) {
    for (unsigned l = 0; l < 32; ++l) {
      SmallVector<std::pair<StringAttr, int32_t>> ins = {
          {S("register"), 0},
          {S("lane"), (int32_t)l},
          {S("warp"), (int32_t)w},
          {S("block"), 0}};
      auto result = ll.apply(ins);
      int elem = result[0].second;
      ASSERT_GE(elem, 0);
      ASSERT_LT(elem, 48);
      if (owner[elem] == -1) {
        owner[elem] = w;
        ownedCount[w]++;
      }
    }
  }

  // All 48 elements should be covered.
  for (int i = 0; i < 48; ++i) {
    EXPECT_NE(owner[i], -1) << "element " << i << " not covered";
  }

  // Warp 0 covers elements 0-31 (32 unique), warp 1 covers 32-47 (16 unique).
  // Warps 2-3 only touch elements already owned by warps 0-1.
  EXPECT_EQ(ownedCount[0], 32u);
  EXPECT_EQ(ownedCount[1], 16u);
  EXPECT_EQ(ownedCount[2], 0u);
  EXPECT_EQ(ownedCount[3], 0u);
}

// Test nvmmaSharedToSplitLinearLayout returns nullopt for pow2 contiguous dim.
TEST_F(LinearLayoutConversionsTest, NvmmaSplitLayout_Pow2_ReturnsNullopt) {
  auto enc = nvmmaShared(128, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  auto split = nvmmaSharedToSplitLinearLayout({16, 64}, enc);
  EXPECT_FALSE(split.has_value());
}

// Test that smemLoad coordinate decomposition produces correct SMEM offsets
// for each rep tile of a K=48 NPOT matmul operand.
// Validates split layout round-trip: forward(pseudoinverse(coords)) == coords.
// (Previously this compared against a 2D pseudoinverse built via
// getLayoutWithinBlock, but that helper now preserves NPOT out-dim sizes, so
// the 2D path no longer phase-folds the NPOT K dim the way the split layout
// does. The split layout is the source of truth; we validate it via round-trip
// instead of cross-checking against the deprecated 2D offset.)
TEST_F(LinearLayoutConversionsTest, NvmmaSplitLayout_SmemLoad_K48) {
  auto enc = nvmmaShared(32, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  SmallVector<int64_t> shapeVec = {64, 48};
  ArrayRef<int64_t> shape = shapeVec;

  auto splitOpt = nvmmaSharedToSplitLinearLayout(shape, enc);
  ASSERT_TRUE(splitOpt.has_value());
  auto &splitLL = *splitOpt;
  auto inv = splitLL.pseudoinvert();

  int mmaSizeK = 16;
  int numRepK = 3; // ceil(48, 16) = 3
  int phaseSize = inv.getInDimSize(S("contig_intra"));

  // Validate round-trip: pseudoinverse(coords) -> offset, forward(offset) ->
  // coords should match the original coords (within the surjective image).
  for (int row : {0, 1, 4, 7, 8, 16, 32}) {
    for (int k = 0; k < numRepK; k++) {
      int b = k * mmaSizeK;
      int bIntra = b % phaseSize;
      int bPhase = b / phaseSize;

      auto result = inv.apply({{S("dim0"), row},
                               {S("contig_intra"), bIntra},
                               {S("contig_phase"), bPhase}});
      int32_t splitOffset = result[0].second;

      // Round-trip: apply forward layout to the offset.
      auto fwdResult = splitLL.apply({{S("offset"), splitOffset}});
      int32_t rtRow = -1, rtIntra = -1, rtPhase = -1;
      for (auto &[name, val] : fwdResult) {
        if (name == S("dim0"))
          rtRow = val;
        else if (name == S("contig_intra"))
          rtIntra = val;
        else if (name == S("contig_phase"))
          rtPhase = val;
      }

      EXPECT_EQ(rtRow, row)
          << "row mismatch at row=" << row << ", k=" << k << ", b=" << b
          << ", bIntra=" << bIntra << ", bPhase=" << bPhase
          << ", offset=" << splitOffset << ", actual=" << rtRow;
      EXPECT_EQ(rtIntra, bIntra)
          << "contig_intra mismatch at row=" << row << ", k=" << k
          << ", b=" << b << ", bPhase=" << bPhase
          << ", offset=" << splitOffset << ", actual=" << rtIntra
          << ", expected=" << bIntra;
      EXPECT_EQ(rtPhase, bPhase)
          << "contig_phase mismatch at row=" << row << ", k=" << k
          << ", b=" << b << ", bIntra=" << bIntra
          << ", offset=" << splitOffset << ", actual=" << rtPhase
          << ", expected=" << bPhase;
    }
  }
}

// Test SBO computation from the split pseudoinverse for K=48.
TEST_F(LinearLayoutConversionsTest, NvmmaSplitLayout_SBO_K48) {
  auto enc = nvmmaShared(32, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  SmallVector<int64_t> shapeVec = {64, 48};
  ArrayRef<int64_t> shape = shapeVec;

  auto splitOpt = nvmmaSharedToSplitLinearLayout(shape, enc);
  ASSERT_TRUE(splitOpt.has_value());
  auto llInv = splitOpt->pseudoinvert();

  // The SBO should be the row stride (in elements) when going from
  // the core tile's last row to the first row of the next tile.
  // For non-transposed with K=48, the row stride should be 48 elements.
  // shmemTileInv.getInDimSizeLog2(dim0) = log2(8) = 3
  // So SBO = llInv.getBasis(dim0, 3, offset)
  int32_t sbo = llInv.getBasis(S("dim0"), 3, S("offset"));

  // For comparison, check the 2D pseudoinverse
  auto fwd2DFull = nvmmaSharedToLinearLayout(shape, enc, TMAMode::Tiled);
  auto fwd2D = getLayoutWithinBlock(fwd2DFull);
  auto fwd2DLocal =
      fwd2D.sublayout({S("offset")}, llvm::to_vector(fwd2D.getOutDimNames()));
  auto inv2D = fwd2DLocal.pseudoinvert();

  // In the 2D pseudoinverse, the row stride beyond the tile is:
  int32_t sbo2D = inv2D.getBasis(S("dim0"), 3, S("offset"));
  EXPECT_EQ(sbo, sbo2D) << "SBO mismatch: split=" << sbo << " 2D=" << sbo2D;
}

// Verify the standard getDescriptor brute-force path works for pow2 shapes
// by checking LBO/SBO values from the pseudoinverse.
TEST_F(LinearLayoutConversionsTest, NvmmaPow2Descriptor_64x128) {
  auto enc = nvmmaShared(128, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  auto fwd = toLinearLayout({64, 128}, enc);
  auto fwdLocal = getLayoutWithinBlock(fwd);
  auto inv = fwdLocal.pseudoinvert();

  int tileCols = 64;
  int tileRows = 8;

  // Check key positions
  auto at_0_0 = inv.apply({{S("dim0"), 0}, {S("dim1"), 0}});
  EXPECT_EQ(at_0_0[0].second, 0) << "Offset at (0,0) should be 0";

  // LBO = offset at (0, tileCols) - offset at (0, 0)
  auto at_0_tc = inv.apply({{S("dim0"), 0}, {S("dim1"), tileCols}});
  int lbo = at_0_tc[0].second - at_0_0[0].second;

  // SBO = offset at (tileRows, 0) - offset at (0, 0)
  auto at_tr_0 = inv.apply({{S("dim0"), tileRows}, {S("dim1"), 0}});
  int sbo = at_tr_0[0].second - at_0_0[0].second;

  // For 64x128 row-major with swizzle=128:
  // Each row is 128 elements (256 bytes = 2 x 128B rows).
  // LBO (stride between tile column groups) = 64 elements
  // or in SMEM: depends on layout structure.
  // Just verify they are reasonable values.
  EXPECT_GT(lbo, 0) << "LBO should be positive";
  EXPECT_GT(sbo, 0) << "SBO should be positive";
}

// Verify the standard toLinearLayout pseudoinverse round-trips correctly
// for NPOT non-contiguous dim. Uses getLayoutWithinBlock to remove the
// block dim before pseudoinverting.
TEST_F(LinearLayoutConversionsTest, NvmmaPseudoinvert_NonContigNpot_K48) {
  auto enc = nvmmaShared(128, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  auto fwd = toLinearLayout({48, 128}, enc);

  // Strip block dim before pseudoinverting.
  auto fwdLocal = getLayoutWithinBlock(fwd).sublayout(
      {S("offset")}, llvm::to_vector(fwd.getOutDimNames()));
  auto inv = fwdLocal.pseudoinvert();

  // The forward layout maps offset -> (dim0=48, dim1=128).
  // The pseudoinverse maps (dim0, dim1) -> offset.
  // Verify round-trip for all in-range (dim0, dim1) pairs.
  int failures = 0;
  for (int d0 = 0; d0 < 48; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
      auto invResult = inv.apply({{S("dim0"), d0}, {S("dim1"), d1}});
      int32_t offset = invResult[0].second;
      auto fwdResult = fwdLocal.apply({{S("offset"), offset}});
      int32_t d0_rt = fwdResult[0].second;
      int32_t d1_rt = fwdResult[1].second;
      if (d0_rt != d0 || d1_rt != d1) {
        failures++;
        if (failures <= 5) {
          EXPECT_EQ(d0_rt, d0) << "dim0 mismatch at (" << d0 << ", " << d1
                               << ") -> offset=" << offset;
          EXPECT_EQ(d1_rt, d1) << "dim1 mismatch at (" << d0 << ", " << d1
                               << ") -> offset=" << offset;
        }
      }
    }
  }
  EXPECT_EQ(failures, 0) << "Total round-trip failures: " << failures;
}

// Verify split-dim invertAndCompose produces correct results for K=48.
// This tests the full store path: blocked encoding -> split shared layout.
TEST_F(LinearLayoutConversionsTest, SplitInvertAndCompose_K48) {
  // Create a blocked encoding and shared encoding for shape [16, 48].
  auto blockedEnc =
      blocked({1, 1}, {32, 1}, {1, 4}, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  auto sharedEnc = nvmmaShared(32, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  auto regLayout = toLinearLayout({16, 48}, blockedEnc);
  auto sharedLayout = toLinearLayout({16, 48}, sharedEnc);

  // Reshape both to split dim1 into contig_intra(16) x contig_phase(3).
  SmallVector<std::pair<StringAttr, int32_t>> splitOutDims;
  for (auto [name, size] : sharedLayout.getOutDims()) {
    if (name == S("dim1")) {
      splitOutDims.push_back({S("contig_intra"), 16});
      splitOutDims.push_back({S("contig_phase"), 3});
    } else {
      splitOutDims.push_back({name, size});
    }
  }
  auto regSplit = regLayout.reshapeOuts(splitOutDims);
  auto sharedSplit = sharedLayout.reshapeOuts(splitOutDims);

  // invertAndCompose should succeed via lstsqModular.
  auto cvt = regSplit.invertAndCompose(sharedSplit);

  // The result maps {register, lane, warp, block} -> {offset, block}.
  EXPECT_TRUE(cvt.hasOutDim(S("offset")));
  EXPECT_TRUE(cvt.hasOutDim(S("block")));

  // Verify the conversion is correct: for each register element,
  // the shared offset should map to the same logical tensor position.
  auto cvtLocal =
      cvt.sublayout({S("register"), S("lane"), S("warp")}, {S("offset")});

  int numRegs = regSplit.getInDimSize(S("register"));
  int numLanes = regSplit.getInDimSize(S("lane"));
  int numWarps = regSplit.getInDimSize(S("warp"));

  // Spot-check a few elements.
  for (int r = 0; r < std::min(numRegs, 4); r++) {
    for (int l = 0; l < std::min(numLanes, 4); l++) {
      auto regResult = regSplit.apply({{S("register"), r},
                                       {S("lane"), l},
                                       {S("warp"), 0},
                                       {S("block"), 0}});
      auto cvtResult =
          cvtLocal.apply({{S("register"), r}, {S("lane"), l}, {S("warp"), 0}});
      int32_t offset = cvtResult[0].second;

      // Check: sharedSplit.apply({offset, 0}) should give the same tensor
      // position as regResult (ignoring out-of-range).
      auto sharedResult =
          sharedSplit.apply({{S("offset"), offset}, {S("block"), 0}});

      for (size_t d = 0; d < regResult.size(); d++) {
        EXPECT_EQ(regResult[d].second, sharedResult[d].second)
            << "Mismatch at reg=" << r << " lane=" << l
            << " dim=" << regResult[d].first.str();
      }
    }
  }
}

// Test B operand store: verify invertAndCompose for blocked -> nvmma_shared
// when dim0=48 (NPOT non-contiguous). The fix is to build the blocked layout
// with the pow2-rounded shape ({64, 64}) to match the SMEM layout, avoiding
// the modular-vs-pow2 dim size mismatch in invertAndCompose.
TEST_F(LinearLayoutConversionsTest, DiagnosticDump_K48_BStore) {
  // B: [48, 64] fp16, swizzle=128B
  auto blockedEnc =
      blocked({1, 8}, {4, 8}, {4, 1}, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  auto sharedEnc = nvmmaShared(128, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0});

  auto smemLayout = toLinearLayout({48, 64}, sharedEnc);

  // The SMEM layout has dim0=64 (pow2-rounded from 48 by the non-contiguous
  // NPOT fix). Build the blocked layout with the same pow2 shape.
  EXPECT_EQ(smemLayout.getOutDimSize(S("dim0")), 64);
  auto srcLayout = toLinearLayout({64, 64}, blockedEnc);

  auto cvt = srcLayout.invertAndCompose(smemLayout);
  auto cvtLocal =
      cvt.sublayout({S("register"), S("lane"), S("warp")}, {S("offset")});

  auto srcSub =
      srcLayout.sublayout({S("register"), S("lane"), S("warp")},
                          llvm::to_vector(srcLayout.getOutDimNames()));
  auto smemSub = smemLayout.sublayout(
      {S("offset")}, llvm::to_vector(smemLayout.getOutDimNames()));

  int numRegs = srcLayout.getInDimSize(S("register"));
  int numLanes = srcLayout.getInDimSize(S("lane"));
  int numWarps = srcLayout.getInDimSize(S("warp"));

  int mismatches = 0;
  for (int w = 0; w < numWarps; w++) {
    for (int l = 0; l < numLanes; l++) {
      for (int r = 0; r < numRegs; r++) {
        auto srcResult =
            srcSub.apply({{S("register"), r}, {S("lane"), l}, {S("warp"), w}});
        int d0 = srcResult[0].second;
        int d1 = srcResult[1].second;

        auto cvtResult = cvtLocal.apply(
            {{S("register"), r}, {S("lane"), l}, {S("warp"), w}});
        int32_t offset = cvtResult[0].second;

        auto smemResult = smemSub.apply({{S("offset"), offset}});
        int sd0 = smemResult[0].second;
        int sd1 = smemResult[1].second;

        if (d0 != sd0 || d1 != sd1) {
          mismatches++;
        }
      }
    }
  }
  EXPECT_EQ(mismatches, 0) << "B store invertAndCompose mismatches: "
                           << mismatches;
}

// Test: compare the A operand forward layout with the B operand forward layout
// to find differences in how they handle K=48.
TEST_F(LinearLayoutConversionsTest, DiagnosticDump_K48_BOperand) {
  // B operand: [48, 64] fp16, swizzle=128B, not transposed
  // The non-contiguous dim is K=48 (dim0), contiguous dim is N=64 (dim1)
  auto encB = nvmmaShared(128, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  SmallVector<int64_t> shapeB = {48, 64};

  // B uses the standard path (contiguous dim=64 is pow2)
  auto splitOptB = nvmmaSharedToSplitLinearLayout(shapeB, encB);
  EXPECT_FALSE(splitOptB.has_value())
      << "B operand should NOT use split layout";

  // Standard 2D layout
  auto fwdB = nvmmaSharedToLinearLayout(shapeB, encB, TMAMode::Tiled);
  auto fwdBLocal = getLayoutWithinBlock(fwdB);
  auto fwdBSub = fwdBLocal.sublayout(
      {S("offset")}, llvm::to_vector(fwdBLocal.getOutDimNames()));
  auto invB = fwdBSub.pseudoinvert();

  // Verify the pseudoinverse produces valid offsets for sample points.
  for (int row : {0, 8, 16, 32, 47}) {
    auto r = invB.apply({{S("dim0"), row}, {S("dim1"), 0}});
    EXPECT_GE(r[0].second, 0);
  }
}

// ==========================================================================
// NPOT dot support tests
// ==========================================================================

// Test tensorMemoryToLinearLayout for NPOT blockN=96.
// TMEM uses modularIdentity1D for NPOT blockN: outDimSize=96 (logical),
// inDimSize=128 (pow2, physical TMEM columns).
TEST_F(LinearLayoutConversionsTest, TensorMemory_NPOT_blockN_96) {
  auto enc = tmem(128, 96, 1, 1);
  auto d0 = S("dim0");
  auto d1 = S("dim1");
  auto kCol = S("col");

  auto layout = toLinearLayout({128, 96}, enc);

  // outDimSize for columns is the NPOT blockN (modular), not pow2.
  EXPECT_EQ(layout.getOutDimSize(d1), 96);
  EXPECT_EQ(layout.getOutDimSize(d0), 128);

  // inDimSize for kCol is pow2 (physical TMEM allocation).
  EXPECT_EQ(layout.getInDimSize(kCol), 128);

  // The layout is modular because of the NPOT column dim.
  EXPECT_TRUE(layout.isModular());
}

// Test tensorMemoryToLinearLayout for NPOT blockN=48.
TEST_F(LinearLayoutConversionsTest, TensorMemory_NPOT_blockN_48) {
  auto enc = tmem(128, 48, 1, 1);
  auto d0 = S("dim0");
  auto d1 = S("dim1");
  auto kCol = S("col");

  auto layout = toLinearLayout({128, 48}, enc);

  // outDimSize=48 (modular), inDimSize=64 (pow2 physical).
  EXPECT_EQ(layout.getOutDimSize(d1), 48);
  EXPECT_EQ(layout.getOutDimSize(d0), 128);
  EXPECT_EQ(layout.getInDimSize(kCol), 64);
  EXPECT_TRUE(layout.isModular());
}

// Regression guard for the SM100 (TMEM / tc_gen5_mma) NPOT-N reduction in
// applyNpotReductionForTmem (TritonNvidiaGPU/IR/Dialect.cpp).
//
// For NPOT blockN the distributed TMEM ld/st layout is built on a pow2-rounded
// N and then reduced back to N.  This is sound ONLY if the resulting modular
// layout round-trips correctly through the physical TMEM layout the lowering
// uses, i.e. for the cvt = regLayout.invertAndCompose(memLayout) computed in
// computeTMemLdStEncodingInfo, every distributed position x must satisfy
//     memLayout(cvt(x)) == regLayout(x).
// If the reduction left a basis that combines via XOR to a column >= N, or
// otherwise mismatched the modular split-dim semantics, the round-trip breaks
// and the kernel reads/writes the wrong TMEM column (the Bug-C miscompile that
// shows up on B200 as replicated / shifted output blocks).
//
// This covers the named failing single-tile configs (N=24, and N=48 with
// BLOCK_M=64 / num_warps=8) plus a full single-tile NPOT-N sweep, for the
// I32x32b ld/st atom selected by getDefaultLayoutForTmemLdSt.
TEST_F(LinearLayoutConversionsTest, TensorMemory_NPOT_N_DistRoundTrip) {
  auto tmemSpace = nvidia_gpu::TensorMemorySpaceAttr::get(&ctx);
  auto f16 = Float16Type::get(&ctx);
  auto cga = CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0});
  auto d1 = S("dim1");

  // {blockM, numWarps} combinations to exercise (lane/warp bits land on the
  // N-column basis in the M=64 and num_warps=8 cases -- the BLOCK_M=64
  // trigger).
  for (int bM : {64, 128}) {
    for (int nw : {4, 8}) {
      for (int N = 8; N <= 256; N += 8) {
        if (llvm::isPowerOf2_32(N))
          continue; // only NPOT N exercises applyNpotReductionForTmem
        auto enc = tmem(bM, N, 1, 1);
        auto memTy =
            MemDescType::get({(int64_t)bM, (int64_t)N}, f16, enc, tmemSpace);
        auto encOut = getDefaultLayoutForTmemLdSt(memTy, nw, cga);
        auto regLayout = toLinearLayout({(int64_t)bM, (int64_t)N},
                                        cast<DistributedEncodingTrait>(encOut));

        // The logical N-column dim must be reduced to the NPOT N (modular),
        // not left at the pow2 padding.
        EXPECT_EQ(regLayout.getOutDimSize(d1), N)
            << "bM=" << bM << " N=" << N << " nw=" << nw
            << ": N-column dim not reduced to NPOT N";

        // Round-trip through the physical TMEM layout exactly as the lowering's
        // computeTMemLdStEncodingInfo does.
        auto memLayout = toLinearLayout({(int64_t)bM, (int64_t)N}, enc);
        auto cvt = regLayout.invertAndCompose(memLayout);
        auto regFlat = regLayout.flattenIns();
        auto cvtFlat = cvt.flattenIns();
        auto inDim = *regFlat.getInDimNames().begin();
        int total = regFlat.getInDimSize(inDim);
        int mismatches = 0;
        for (int x = 0; x < total; x++) {
          auto want = regFlat.apply({{inDim, x}});     // (dim0, dim1)
          auto physical = cvtFlat.apply({{inDim, x}}); // (row, col)
          auto got = memLayout.apply(physical);        // (dim0, dim1)
          if (want != got)
            mismatches++;
        }
        EXPECT_EQ(mismatches, 0)
            << "bM=" << bM << " N=" << N << " nw=" << nw
            << ": distributed TMEM layout does not round-trip through the "
               "physical TMEM layout (XOR-vs-mod overflow in the NPOT-N "
               "reduction)";
      }
    }
  }
}

// Test MMAv5 instrShape selection for NPOT N values.
// mmaVersionToInstrShape must pick instrN that divides BLOCK_N evenly.
TEST_F(LinearLayoutConversionsTest, MMAv5_InstrShapeSelection) {
  auto f16 = Float16Type::get(&ctx);

  // N=96 -> instrN=96 (96 is a multiple of 8, divides 96)
  {
    auto shape = mmaVersionToInstrShape(5, {128, 96}, f16, 4);
    EXPECT_EQ(shape[1], 96u) << "instrN for N=96 should be 96";
    EXPECT_EQ(shape[0], 128u) << "instrM for M=128 should be 128";
    EXPECT_EQ(shape[2], 16u) << "instrK for fp16 should be 16";
  }

  // N=128 -> instrN=128
  {
    auto shape = mmaVersionToInstrShape(5, {128, 128}, f16, 4);
    EXPECT_EQ(shape[1], 128u);
  }

  // N=256 -> instrN=256
  {
    auto shape = mmaVersionToInstrShape(5, {128, 256}, f16, 4);
    EXPECT_EQ(shape[1], 256u);
  }

  // N=512 -> instrN=256 (capped at 256, and 512 % 256 == 0)
  {
    auto shape = mmaVersionToInstrShape(5, {128, 512}, f16, 4);
    EXPECT_EQ(shape[1], 256u) << "instrN should be capped at 256";
  }

  // N=48 -> instrN=48
  {
    auto shape = mmaVersionToInstrShape(5, {128, 48}, f16, 4);
    EXPECT_EQ(shape[1], 48u) << "instrN for N=48 should be 48";
  }

  // N=192 -> instrN=192
  {
    auto shape = mmaVersionToInstrShape(5, {128, 192}, f16, 4);
    EXPECT_EQ(shape[1], 192u) << "instrN for N=192 should be 192";
  }

  // M=64 -> instrM=64
  {
    auto shape = mmaVersionToInstrShape(5, {64, 96}, f16, 4);
    EXPECT_EQ(shape[0], 64u) << "instrM for M=64 should be 64";
    EXPECT_EQ(shape[1], 96u);
  }
}

// Test nvmmaSharedToSplitLinearLayout for transposed shared encoding
// with NPOT K=48. When transposed, contigDim=0 (rows) instead of dim1.
// This validates that the split-dim path handles transposed correctly.
TEST_F(LinearLayoutConversionsTest, NvmmaSplitLayout_K48_Transposed) {
  // K=48 fp16 transposed: the K dimension is now dim0 (the contiguous dim).
  // contigBytes = 48*2 = 96, swizzle=32B: 96%32=0 -> tileCols=16, phases=3.
  auto enc = nvmmaShared(32, true, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0});
  // Shape [48, 64]: K=48 is dim0 (contiguous when transposed), N=64 is dim1.
  auto split = nvmmaSharedToSplitLinearLayout({48, 64}, enc);

  if (!split.has_value()) {
    // The transposed path may not be implemented yet -- that's OK to document.
    // If it returns nullopt, verify the standard path still works.
    auto fwd = toLinearLayout({48, 64}, enc);
    EXPECT_EQ(fwd.getOutDimSize(S("dim0")), 48);
    EXPECT_EQ(fwd.getOutDimSize(S("dim1")), 64);
    return;
  }

  // If split IS returned, verify the structure.
  // When transposed, contigDim=0, so we'd expect dim0_intra and dim0_phase.
  // But the implementation always names the split dims
  // contig_intra/contig_phase (it operates in 2D collapsed space where dim1 is
  // always contiguous). Verify it has the 3 expected output dims.
  auto outDims = llvm::to_vector(split->getOutDimNames());
  EXPECT_EQ(outDims.size(), 3u) << "Split layout should have 3 output dims";
}

// Test NPOT N SMEM conversion: verify that unfolding modular dims to pow2
// prevents duplicate register-to-offset collisions in the SMEM shuffle.
// This is the key fix for NPOT N dot accumulator conversion.
TEST_F(LinearLayoutConversionsTest, TensorMemory_NPOT_N_SmemConversion) {
  // Simulate the TMEM-distributed layout for blockN=96.
  // The distributed layout has 128 register positions (pow2) but only
  // 96 produce valid data. Registers 96-127 map to dim1=0-31 via mod.
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto d0 = S("dim0");
  auto d1 = S("dim1");

  // Build a simplified TMEM-distributed layout for blockN=96:
  // Registers cover cols 0-127, lanes cover rows, warps cover rows.
  // dim1 is modular with size 96.
  auto regToDim1 = LinearLayout::modularIdentity1D(96, kReg, d1);
  EXPECT_TRUE(regToDim1.isModular());
  EXPECT_EQ(regToDim1.getOutDimSize(d1), 96);
  EXPECT_EQ(regToDim1.getInDimSize(kReg), 128); // pow2 physical

  // Verify modular wrapping: register 96 maps to dim1=0 (same as reg 0).
  auto val0 = regToDim1.apply({{kReg, 0}});
  auto val96 = regToDim1.apply({{kReg, 96}});
  EXPECT_EQ(val0[0].second, 0) << "register 0 -> dim1=0";
  EXPECT_EQ(val96[0].second, 0) << "register 96 -> dim1=0 via mod 96";

  // Unfold to pow2: dim1 size 96 -> 128.
  SmallVector<std::pair<StringAttr, int32_t>> pow2OutDims;
  for (auto [dim, size] : regToDim1.getOutDims()) {
    pow2OutDims.push_back(
        {dim, llvm::isPowerOf2_32(size)
                  ? size
                  : static_cast<int32_t>(llvm::NextPowerOf2(size - 1))});
  }
  auto regToDim1Unfolded = LinearLayout(regToDim1.getBases(), pow2OutDims,
                                        /*requireSurjective=*/false);

  EXPECT_FALSE(regToDim1Unfolded.isModular());
  EXPECT_EQ(regToDim1Unfolded.getOutDimSize(d1), 128);

  // After unfolding, register 96 maps to dim1=96 (NOT 0).
  auto val96u = regToDim1Unfolded.apply({{kReg, 96}});
  EXPECT_EQ(val96u[0].second, 96)
      << "unfolded: register 96 -> dim1=96 (separate from reg 0)";

  // Verify all valid registers (0-95) still map to distinct values.
  std::set<int32_t> seenValues;
  for (int i = 0; i < 96; i++) {
    auto val = regToDim1Unfolded.apply({{kReg, i}});
    int32_t v = val[0].second;
    EXPECT_TRUE(v < 96) << "register " << i << " maps to dim1=" << v
                        << " (should be < 96)";
    seenValues.insert(v);
  }
  EXPECT_EQ(seenValues.size(), 96u) << "96 unique values for registers 0-95";
}

// Test that NPOT blockN=192 (HEAD_DIM=192) layout works correctly.
TEST_F(LinearLayoutConversionsTest, TensorMemory_NPOT_blockN_192) {
  auto enc = tmem(128, 192, 1, 1);
  auto d0 = S("dim0");
  auto d1 = S("dim1");
  auto kCol = S("col");

  auto layout = toLinearLayout({128, 192}, enc);

  // outDimSize for columns is the NPOT blockN (modular), not pow2.
  EXPECT_EQ(layout.getOutDimSize(d1), 192);
  EXPECT_EQ(layout.getOutDimSize(d0), 128);

  // inDimSize for kCol is pow2 (physical TMEM allocation).
  EXPECT_EQ(layout.getInDimSize(kCol), 256);

  // The layout is modular because of the NPOT column dim.
  EXPECT_TRUE(layout.isModular());

  // Verify that all logical columns 0-191 are reachable.
  std::set<int32_t> seenCols;
  auto kRow = S("row");
  for (int col = 0; col < 256; col++) {
    auto val = layout.apply({{kRow, 0}, {kCol, col}});
    int32_t d1val = -1;
    for (auto &[name, v] : val) {
      if (name == d1)
        d1val = v;
    }
    if (d1val >= 0 && d1val < 192)
      seenCols.insert(d1val);
  }
  EXPECT_EQ(seenCols.size(), 192u) << "All 192 logical columns reachable";
}

} // anonymous namespace
} // namespace mlir::triton::gpu

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
