#include "triton/Tools/ModularArithmetic.h"

#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace mlir::triton {
namespace {

class ModularArithmeticTest : public ::testing::Test {};

TEST_F(ModularArithmeticTest, ExtGCD_Basic) {
  auto result = extendedGCD(240, 46);
  EXPECT_EQ(result.gcd, 2);
  EXPECT_EQ(240 * result.x + 46 * result.y, result.gcd);

  result = extendedGCD(0, 5);
  EXPECT_EQ(result.gcd, 5);
  EXPECT_EQ(0 * result.x + 5 * result.y, result.gcd);

  result = extendedGCD(7, 0);
  EXPECT_EQ(result.gcd, 7);
  EXPECT_EQ(7 * result.x + 0 * result.y, result.gcd);

  result = extendedGCD(0, 0);
  EXPECT_EQ(result.gcd, 0);

  result = extendedGCD(0, -5);
  EXPECT_EQ(result.gcd, 5);
  EXPECT_EQ(0 * result.x + (-5) * result.y, result.gcd);

  result = extendedGCD(-7, 0);
  EXPECT_EQ(result.gcd, 7);
  EXPECT_EQ((-7) * result.x + 0 * result.y, result.gcd);
}

TEST_F(ModularArithmeticTest, ExtGCD_NegativeInputs) {
  // Both negative
  auto result = extendedGCD(-12, -8);
  EXPECT_EQ(result.gcd, 4);
  EXPECT_EQ((-12) * result.x + (-8) * result.y, result.gcd);

  // One negative
  result = extendedGCD(-15, 10);
  EXPECT_EQ(result.gcd, 5);
  EXPECT_EQ((-15) * result.x + 10 * result.y, result.gcd);
}

TEST_F(ModularArithmeticTest, ModInverse_Basic) {
  auto inv = modInverse(3, 11);
  ASSERT_TRUE(inv.has_value());
  EXPECT_EQ((3 * (*inv)) % 11, 1);

  inv = modInverse(2, 4);
  EXPECT_FALSE(inv.has_value());

  inv = modInverse(6, 9);
  EXPECT_FALSE(inv.has_value());

  // modInverse(1, N) always returns 1
  inv = modInverse(1, 7);
  ASSERT_TRUE(inv.has_value());
  EXPECT_EQ(*inv, 1);

  inv = modInverse(1, 100);
  ASSERT_TRUE(inv.has_value());
  EXPECT_EQ(*inv, 1);
}

TEST_F(ModularArithmeticTest, ModInverse_NegativeA) {
  // -3 mod 11 = 8, inverse of 8 mod 11 = 7
  auto inv = modInverse(-3, 11);
  ASSERT_TRUE(inv.has_value());
  EXPECT_EQ(((-3 % 11 + 11) % 11 * (*inv)) % 11, 1);
}

TEST_F(ModularArithmeticTest, IntPow_Basic) {
  EXPECT_EQ(intPow(2, 10), 1024);
  EXPECT_EQ(intPow(0, 0), 1);
  EXPECT_EQ(intPow(5, 0), 1);
  EXPECT_EQ(intPow(1, 100), 1);
  EXPECT_EQ(intPow(3, 5), 243);
  EXPECT_EQ(intPow(-2, 3), -8);
}

TEST_F(ModularArithmeticTest, CRT_Basic) {
  // Single congruence: x ≡ 5 (mod 7)
  auto result = solveCRT({5}, {7});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->solution, 5);
  EXPECT_EQ(result->modulus, 7);

  // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
  result = solveCRT({2, 3, 2}, {3, 5, 7});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->modulus, 105);
  EXPECT_GE(result->solution, 0);
  EXPECT_LT(result->solution, 105);
  EXPECT_EQ(result->solution % 3, 2);
  EXPECT_EQ(result->solution % 5, 3);
  EXPECT_EQ(result->solution % 7, 2);
}

TEST_F(ModularArithmeticTest, CRT_RejectsInvalidInput) {
  // Pairwise coprime moduli are required (documented precondition). In
  // release builds the assertion compiles out and the implementation falls
  // through to a per-modulus modInverse step that returns nullopt when
  // M_i and m_i share a factor — verify the API surfaces nullopt rather
  // than silently producing a wrong answer. (In debug builds the assertion
  // fires before reaching this path; this test guards the release path.)
#ifdef NDEBUG
  // moduli 4 and 6 share factor 2 — not pairwise coprime.
  auto noncoprime = solveCRT({1, 1}, {4, 6});
  EXPECT_FALSE(noncoprime.has_value());
#endif

  // Empty inputs.
  EXPECT_FALSE(solveCRT({}, {}).has_value());

  // Size mismatch.
  EXPECT_FALSE(solveCRT({1}, {2, 3}).has_value());

  // Non-positive modulus.
  EXPECT_FALSE(solveCRT({1, 1}, {3, 0}).has_value());
  EXPECT_FALSE(solveCRT({1, 1}, {3, -5}).has_value());
}

TEST_F(ModularArithmeticTest, ExtendedGCD_RejectsInt64Min) {
  // |INT64_MIN| does not fit in int64_t, so std::abs(INT64_MIN) is UB.
  // The implementation rejects INT64_MIN inputs by returning {0, 0, 0}
  // (matching the gcd(0, 0) convention).
  auto r1 = extendedGCD(std::numeric_limits<int64_t>::min(), 7);
  EXPECT_EQ(r1.gcd, 0);
  auto r2 = extendedGCD(7, std::numeric_limits<int64_t>::min());
  EXPECT_EQ(r2.gcd, 0);
}

TEST_F(ModularArithmeticTest, ZnZ_RREF_Mod5) {
  ModMatrix mat(2, 2, 5);
  mat.at(0, 0) = 1; mat.at(0, 1) = 2;
  mat.at(1, 0) = 3; mat.at(1, 1) = 4;

  int rank = modRREF(mat);
  EXPECT_EQ(rank, 2);

  EXPECT_EQ(mat.at(0, 0), 1);
  EXPECT_EQ(mat.at(0, 1), 0);
  EXPECT_EQ(mat.at(1, 0), 0);
  EXPECT_EQ(mat.at(1, 1), 1);

  // Singular matrix (mod 3): [[1, 2], [1, 2]] has rank 1
  ModMatrix singularMat(2, 2, 3);
  singularMat.at(0, 0) = 1; singularMat.at(0, 1) = 2;
  singularMat.at(1, 0) = 1; singularMat.at(1, 1) = 2;
  rank = modRREF(singularMat);
  EXPECT_EQ(rank, 1);
}

TEST_F(ModularArithmeticTest, ZnZ_RREF_RectangularMatrix) {
  // 3x2 over Z/7Z: [[1,2],[3,4],[5,6]] has rank 2
  ModMatrix mat(3, 2, 7);
  mat.at(0, 0) = 1; mat.at(0, 1) = 2;
  mat.at(1, 0) = 3; mat.at(1, 1) = 4;
  mat.at(2, 0) = 5; mat.at(2, 1) = 6;

  int rank = modRREF(mat);
  ASSERT_EQ(rank, 2);

  EXPECT_EQ(mat.at(0, 0), 1);
  EXPECT_EQ(mat.at(0, 1), 0);
  EXPECT_EQ(mat.at(1, 0), 0);
  EXPECT_EQ(mat.at(1, 1), 1);
  EXPECT_EQ(mat.at(2, 0), 0);
  EXPECT_EQ(mat.at(2, 1), 0);
}

TEST_F(ModularArithmeticTest, ZnZ_RREF_CompositeModulus) {
  // [[2, 0], [0, 3]] mod 6: neither pivot is invertible, rank should be 0
  ModMatrix mat(2, 2, 6);
  mat.at(0, 0) = 2; mat.at(0, 1) = 0;
  mat.at(1, 0) = 0; mat.at(1, 1) = 3;
  int rank = modRREF(mat);
  EXPECT_EQ(rank, 0);

  // [[1, 2], [3, 4]] mod 6: det=-2, gcd(2,6)!=1, so rank 1 not 2
  ModMatrix mat2(2, 2, 6);
  mat2.at(0, 0) = 1; mat2.at(0, 1) = 2;
  mat2.at(1, 0) = 3; mat2.at(1, 1) = 4;
  rank = modRREF(mat2);
  EXPECT_EQ(rank, 1);

  // [[1, 0], [0, 5]] mod 6: both pivots invertible, rank 2
  ModMatrix mat3(2, 2, 6);
  mat3.at(0, 0) = 1; mat3.at(0, 1) = 0;
  mat3.at(1, 0) = 0; mat3.at(1, 1) = 5;
  rank = modRREF(mat3);
  EXPECT_EQ(rank, 2);
}

TEST_F(ModularArithmeticTest, ZnZ_RREF_CompositeModulus_PivotSwap) {
  // Row swap needed: row 0 col 0 is 4 (gcd(4,6)=2), row 1 col 0 is 1
  ModMatrix mat(2, 2, 6);
  mat.at(0, 0) = 4; mat.at(0, 1) = 2;
  mat.at(1, 0) = 1; mat.at(1, 1) = 3;
  int rank = modRREF(mat);
  EXPECT_EQ(rank, 1);

  // Full-rank: det=-1, gcd(1,6)=1 → identity
  ModMatrix mat2(2, 2, 6);
  mat2.at(0, 0) = 5; mat2.at(0, 1) = 2;
  mat2.at(1, 0) = 3; mat2.at(1, 1) = 1;
  rank = modRREF(mat2);
  EXPECT_EQ(rank, 2);
  EXPECT_EQ(mat2.at(0, 0), 1); EXPECT_EQ(mat2.at(0, 1), 0);
  EXPECT_EQ(mat2.at(1, 0), 0); EXPECT_EQ(mat2.at(1, 1), 1);

  // 3x3 mod 10: row 1 is 2×row 0, rank=2
  ModMatrix mat3(3, 3, 10);
  mat3.at(0, 0) = 1; mat3.at(0, 1) = 2; mat3.at(0, 2) = 3;
  mat3.at(1, 0) = 2; mat3.at(1, 1) = 4; mat3.at(1, 2) = 6;
  mat3.at(2, 0) = 0; mat3.at(2, 1) = 0; mat3.at(2, 2) = 3;
  rank = modRREF(mat3);
  EXPECT_EQ(rank, 2);
}

TEST_F(ModularArithmeticTest, ModSolveLinear_Basic) {
  // Consistent 2x2 mod 7
  ModMatrix A(2, 2, 7);
  A.at(0, 0) = 1; A.at(0, 1) = 2;
  A.at(1, 0) = 3; A.at(1, 1) = 1;
  std::vector<int64_t> b = {5, 4};
  auto x = modSolveLinear(A, b, 7);
  ASSERT_EQ(x.size(), 2u);
  EXPECT_EQ((A.at(0, 0) * x[0] + A.at(0, 1) * x[1]) % 7, b[0] % 7);
  EXPECT_EQ((A.at(1, 0) * x[0] + A.at(1, 1) * x[1]) % 7, b[1] % 7);

  // Inconsistent system mod 5
  ModMatrix A2(2, 2, 5);
  A2.at(0, 0) = 1; A2.at(0, 1) = 0;
  A2.at(1, 0) = 1; A2.at(1, 1) = 0;
  std::vector<int64_t> b2 = {1, 2};
  auto x2 = modSolveLinear(A2, b2, 5);
  EXPECT_TRUE(x2.empty());

  // Underdetermined 1x2 mod 7
  ModMatrix A3(1, 2, 7);
  A3.at(0, 0) = 1; A3.at(0, 1) = 3;
  std::vector<int64_t> b3 = {4};
  auto x3 = modSolveLinear(A3, b3, 7);
  ASSERT_EQ(x3.size(), 2u);
  EXPECT_EQ((A3.at(0, 0) * x3[0] + A3.at(0, 1) * x3[1]) % 7, b3[0] % 7);
}

TEST_F(ModularArithmeticTest, ModSolveLinearHensel_Mod9) {
  // modulus = 9 = 3^2
  ModMatrix A(2, 2, 9);
  A.at(0, 0) = 1; A.at(0, 1) = 2;
  A.at(1, 0) = 1; A.at(1, 1) = 1;

  std::vector<int64_t> b = {5, 4};
  auto x = modSolveLinearHensel(A, b, 3, 2);

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 2u);

  int64_t res0 = (A.at(0, 0) * x[0] + A.at(0, 1) * x[1]) % 9;
  int64_t res1 = (A.at(1, 0) * x[0] + A.at(1, 1) * x[1]) % 9;
  EXPECT_EQ(res0, b[0] % 9);
  EXPECT_EQ(res1, b[1] % 9);

  // modulus = 27 = 3^3
  ModMatrix A27(2, 2, 27);
  A27.at(0, 0) = 1; A27.at(0, 1) = 2;
  A27.at(1, 0) = 1; A27.at(1, 1) = 1;

  std::vector<int64_t> b27 = {8, 7};
  auto x27 = modSolveLinearHensel(A27, b27, 3, 3);

  ASSERT_FALSE(x27.empty());
  ASSERT_EQ(x27.size(), 2u);

  int64_t res0_27 = (A27.at(0, 0) * x27[0] + A27.at(0, 1) * x27[1]) % 27;
  int64_t res1_27 = (A27.at(1, 0) * x27[0] + A27.at(1, 1) * x27[1]) % 27;
  EXPECT_EQ(res0_27, b27[0] % 27);
  EXPECT_EQ(res1_27, b27[1] % 27);
}

TEST_F(ModularArithmeticTest, ModSolveLinearHensel_Pow2_Mod16) {
  // 2x2 system over Z/16 = 2^4. Tests Hensel lifting with exponent=4
  // (the typical case for pow2 layout dims like 16, 32, 64).
  ModMatrix A(2, 2, 16);
  A.at(0, 0) = 1; A.at(0, 1) = 2;
  A.at(1, 0) = 3; A.at(1, 1) = 5;
  std::vector<int64_t> b = {7, 11};

  auto x = modSolveLinearHensel(A, b, 2, 4);

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 2u);

  int64_t res0 = (A.at(0, 0) * x[0] + A.at(0, 1) * x[1]) % 16;
  int64_t res1 = (A.at(1, 0) * x[0] + A.at(1, 1) * x[1]) % 16;
  EXPECT_EQ(res0, b[0] % 16);
  EXPECT_EQ(res1, b[1] % 16);
}

TEST_F(ModularArithmeticTest, ModSolveLinearHensel_NoSolution) {
  // Inconsistent: 1x = 1 and 1x = 2 (mod 3)
  ModMatrix A(2, 2, 9);
  A.at(0, 0) = 1; A.at(0, 1) = 0;
  A.at(1, 0) = 1; A.at(1, 1) = 0;

  std::vector<int64_t> b = {1, 2};
  auto x = modSolveLinearHensel(A, b, 3, 2);
  EXPECT_TRUE(x.empty());
}

TEST_F(ModularArithmeticTest, ModSolveLinearHensel_Underdetermined) {
  // 1x + 2y + 3z = 7 (mod 9)
  ModMatrix A(1, 3, 9);
  A.at(0, 0) = 1; A.at(0, 1) = 2; A.at(0, 2) = 3;

  std::vector<int64_t> b = {7};
  auto x = modSolveLinearHensel(A, b, 3, 2);

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 3u);

  int64_t res0 = (A.at(0, 0) * x[0] + A.at(0, 1) * x[1] + A.at(0, 2) * x[2]) % 9;
  EXPECT_EQ(res0, b[0] % 9);
}

TEST_F(ModularArithmeticTest, ModSolveLinearCRT_Composite_Mod15) {
  // modulus = 15 = 3 × 5
  ModMatrix A(2, 2, 15);
  A.at(0, 0) = 1; A.at(0, 1) = 2;
  A.at(1, 0) = 1; A.at(1, 1) = 1;

  std::vector<int64_t> b = {7, 5};
  auto x = modSolveLinearCRT(A, b, 15);

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 2u);

  int64_t res0 = (A.at(0, 0) * x[0] + A.at(0, 1) * x[1]) % 15;
  int64_t res1 = (A.at(1, 0) * x[0] + A.at(1, 1) * x[1]) % 15;
  EXPECT_EQ(res0, b[0] % 15);
  EXPECT_EQ(res1, b[1] % 15);

  // modulus = 48 = 2^4 × 3
  ModMatrix A48(2, 2, 48);
  A48.at(0, 0) = 1; A48.at(0, 1) = 2;
  A48.at(1, 0) = 1; A48.at(1, 1) = 1;

  std::vector<int64_t> b48 = {11, 7};
  auto x48 = modSolveLinearCRT(A48, b48, 48);

  ASSERT_FALSE(x48.empty());
  ASSERT_EQ(x48.size(), 2u);

  int64_t res0_48 = (A48.at(0, 0) * x48[0] + A48.at(0, 1) * x48[1]) % 48;
  int64_t res1_48 = (A48.at(1, 0) * x48[0] + A48.at(1, 1) * x48[1]) % 48;
  EXPECT_EQ(res0_48, b48[0] % 48);
  EXPECT_EQ(res1_48, b48[1] % 48);

  // No solution: inconsistent system
  ModMatrix A_nosol(2, 2, 15);
  A_nosol.at(0, 0) = 1; A_nosol.at(0, 1) = 0;
  A_nosol.at(1, 0) = 1; A_nosol.at(1, 1) = 0;

  std::vector<int64_t> b_nosol = {1, 2};
  auto x_nosol = modSolveLinearCRT(A_nosol, b_nosol, 15);
  EXPECT_TRUE(x_nosol.empty());

  // Precondition: modulus must be > 0. Zero and negative moduli return {}
  // explicitly rather than silently failing inside the per-variable CRT
  // step (factorize(0) returns no factors, leaving the moduli vector empty).
  ModMatrix A_ok(2, 2, 15);
  A_ok.at(0, 0) = 1; A_ok.at(0, 1) = 0;
  A_ok.at(1, 0) = 0; A_ok.at(1, 1) = 1;
  std::vector<int64_t> b_ok = {0, 0};
  EXPECT_TRUE(modSolveLinearCRT(A_ok, b_ok, 0).empty());
  EXPECT_TRUE(modSolveLinearCRT(A_ok, b_ok, -1).empty());
}

TEST_F(ModularArithmeticTest, Factorize) {
  // Negative and zero inputs return empty factorization
  auto f0 = factorize(0);
  EXPECT_TRUE(f0.factors.empty());

  auto fn = factorize(-10);
  EXPECT_TRUE(fn.factors.empty());

  auto f1 = factorize(1);
  EXPECT_TRUE(f1.factors.empty());

  auto f2 = factorize(2);
  ASSERT_EQ(f2.factors.size(), 1u);
  EXPECT_EQ(f2.factors[0].first, 2);
  EXPECT_EQ(f2.factors[0].second, 1);
  EXPECT_EQ(f2.product(), 2);

  auto f12 = factorize(12);
  ASSERT_EQ(f12.factors.size(), 2u);
  EXPECT_EQ(f12.factors[0].first, 2);
  EXPECT_EQ(f12.factors[0].second, 2);
  EXPECT_EQ(f12.factors[1].first, 3);
  EXPECT_EQ(f12.factors[1].second, 1);
  EXPECT_EQ(f12.product(), 12);

  auto f60 = factorize(60);
  ASSERT_EQ(f60.factors.size(), 3u);
  EXPECT_EQ(f60.factors[0].first, 2);
  EXPECT_EQ(f60.factors[0].second, 2);
  EXPECT_EQ(f60.factors[1].first, 3);
  EXPECT_EQ(f60.factors[1].second, 1);
  EXPECT_EQ(f60.factors[2].first, 5);
  EXPECT_EQ(f60.factors[2].second, 1);
  EXPECT_EQ(f60.product(), 60);
}

} // namespace
} // namespace mlir::triton
