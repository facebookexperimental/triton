#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <gtest/gtest.h>

namespace mlir::triton::nvidia_gpu {
namespace {

struct ShapeCase {
  Tcgen05Kind kind;
  bool aInTmem, bMNMajor, twoCTAs;
  unsigned m, n;
  bool legal;
};

// Rows follow CUTLASS 4.6.2 mma_sm100_umma.hpp / mma_traits_sm100.hpp.
TEST(Tcgen05ShapeTest, FormExactNRule) {
  using K = Tcgen05Kind;
  const ShapeCase cases[] = {
      // SS 16/32-bit: N % 8 in [8, 256] at M = 64 and 128.
      {K::F16OrTF32, false, false, false, 128, 96, true},
      {K::F16OrTF32, false, true, false, 64, 8, true},
      {K::F16OrTF32, false, false, false, 128, 100, false},
      {K::F16OrTF32, false, false, false, 128, 264, false},
      {K::F16OrTF32, false, false, false, 256, 128, false},
      // TS doubles the step at M = 128 only.
      {K::F16OrTF32, true, false, false, 64, 104, true},
      {K::F16OrTF32, true, false, false, 128, 104, false},
      {K::F16OrTF32, true, false, false, 128, 112, true},
      // 8-bit MN-major B doubles the step; K-major does not.
      {K::F8F6F4, false, false, false, 128, 40, true},
      {K::F8F6F4, false, true, false, 128, 40, false},
      {K::BlockScaled, false, true, false, 128, 40, false},
      // i8: N == 8 or N % 16.
      {K::I8, false, false, false, 128, 8, true},
      {K::I8, false, false, false, 128, 24, false},
      {K::I8, false, false, false, 128, 48, true},
      // Block-scaled 1-CTA requires M = 128.
      {K::BlockScaled, false, false, false, 64, 96, false},
      {K::BlockScaled, false, false, false, 128, 96, true},
      // 2-CTA: steps of 16, doubled for TS and i8.
      {K::F16OrTF32, false, false, true, 128, 24, false},
      {K::F16OrTF32, false, false, true, 128, 48, true},
      {K::F16OrTF32, true, false, true, 64, 48, false},
      {K::F16OrTF32, true, false, true, 64, 64, true},
      {K::I8, false, false, true, 128, 16, false},
      {K::BlockScaled, false, false, true, 64, 96, true},
  };
  for (const ShapeCase &c : cases) {
    auto reason = getTcgen05IllegalShapeReason(c.kind, c.aInTmem, c.bMNMajor,
                                               c.twoCTAs, c.m, c.n);
    EXPECT_EQ(!reason.has_value(), c.legal)
        << "M=" << c.m << " N=" << c.n << ": " << reason.value_or("legal");
  }
  EXPECT_EQ(getTcgen05IllegalShapeReason(Tcgen05Kind::F16OrTF32, true, false,
                                         false, 128, 104),
            "tcgen05.mma kind f16/tf32 TS K-major B cta_group::1 per-CTA "
            "M=128 requires N to be a multiple of 16 in [16, 256], got N=104");
}

} // namespace
} // namespace mlir::triton::nvidia_gpu
