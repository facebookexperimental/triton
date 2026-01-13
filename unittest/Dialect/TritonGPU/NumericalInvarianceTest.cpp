//===- NumericalInvarianceTest.cpp - Tests for NumericalInvariance -*- C++ -*-===//
//
// Part of the Triton project.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/NumericalInvariance.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Signals.h"

namespace mlir::triton::gpu {
namespace {

class NumericalInvarianceTest : public ::testing::Test {
public:
  NumericalInvarianceTest() {
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctx.getOrLoadDialect<triton::TritonDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();
    ctx.getOrLoadDialect<arith::ArithDialect>();
  }

protected:
  MLIRContext ctx;

  // Helper to parse MLIR module from string
  OwningOpRef<ModuleOp> parseModule(StringRef mlirStr) {
    return mlir::parseSourceString<ModuleOp>(mlirStr, &ctx);
  }

  // Helper to create a blocked encoding
  BlockedEncodingAttr createBlockedEncoding(ArrayRef<unsigned> sizePerThread,
                                            ArrayRef<unsigned> threadsPerWarp,
                                            ArrayRef<unsigned> warpsPerCTA,
                                            ArrayRef<unsigned> order) {
    auto ctaLayout = CTALayoutAttr::getDefault(&ctx, order.size());
    return BlockedEncodingAttr::get(&ctx, sizePerThread, threadsPerWarp,
                                    warpsPerCTA, order, ctaLayout);
  }

  // Helper to create an MMA encoding
  NvidiaMmaEncodingAttr createMmaEncoding(unsigned versionMajor,
                                          unsigned versionMinor,
                                          ArrayRef<unsigned> warpsPerCTA,
                                          ArrayRef<unsigned> instrShape) {
    auto ctaLayout = CTALayoutAttr::getDefault(&ctx, 2);
    return NvidiaMmaEncodingAttr::get(&ctx, versionMajor, versionMinor,
                                      warpsPerCTA, ctaLayout, instrShape);
  }
};

//===----------------------------------------------------------------------===//
// Basic Invariance Structure Tests
//===----------------------------------------------------------------------===//

TEST_F(NumericalInvarianceTest, DefaultInvarianceIsZero) {
  NumericalInvariance inv;
  EXPECT_EQ(inv.computationDagHash, 0u);
  EXPECT_EQ(inv.layoutHash, 0u);
  EXPECT_EQ(inv.hwConfigHash, 0u);
  EXPECT_TRUE(inv.dtypeSignature.empty());
}

TEST_F(NumericalInvarianceTest, EqualInvariancesAreEqual) {
  NumericalInvariance inv1;
  inv1.computationDagHash = 12345;
  inv1.layoutHash = 67890;
  inv1.hwConfigHash = 11111;
  inv1.dtypeSignature = "f16->f32";

  NumericalInvariance inv2;
  inv2.computationDagHash = 12345;
  inv2.layoutHash = 67890;
  inv2.hwConfigHash = 11111;
  inv2.dtypeSignature = "f16->f32";

  EXPECT_EQ(inv1, inv2);
  EXPECT_EQ(inv1.fingerprint(), inv2.fingerprint());
}

TEST_F(NumericalInvarianceTest, DifferentComputationDagHashMakesDifferentInvariance) {
  NumericalInvariance inv1;
  inv1.computationDagHash = 12345;

  NumericalInvariance inv2;
  inv2.computationDagHash = 99999;

  EXPECT_NE(inv1, inv2);
  EXPECT_NE(inv1.fingerprint(), inv2.fingerprint());
}

TEST_F(NumericalInvarianceTest, DifferentLayoutHashMakesDifferentInvariance) {
  NumericalInvariance inv1;
  inv1.layoutHash = 12345;

  NumericalInvariance inv2;
  inv2.layoutHash = 99999;

  EXPECT_NE(inv1, inv2);
  EXPECT_NE(inv1.fingerprint(), inv2.fingerprint());
}

TEST_F(NumericalInvarianceTest, DifferentDtypeSignatureMakesDifferentInvariance) {
  NumericalInvariance inv1;
  inv1.dtypeSignature = "f16->f32";

  NumericalInvariance inv2;
  inv2.dtypeSignature = "f32->f32";

  EXPECT_NE(inv1, inv2);
  EXPECT_NE(inv1.fingerprint(), inv2.fingerprint());
}

//===----------------------------------------------------------------------===//
// Layout Encoding Hashing Tests
//===----------------------------------------------------------------------===//

TEST_F(NumericalInvarianceTest, SameBlockedEncodingGivesSameHash) {
  auto enc1 = createBlockedEncoding({1, 4}, {4, 8}, {2, 2}, {1, 0});
  auto enc2 = createBlockedEncoding({1, 4}, {4, 8}, {2, 2}, {1, 0});

  size_t hash1 = detail::hashLayoutEncoding(enc1);
  size_t hash2 = detail::hashLayoutEncoding(enc2);

  EXPECT_EQ(hash1, hash2);
}

TEST_F(NumericalInvarianceTest, DifferentSizePerThreadGivesDifferentHash) {
  auto enc1 = createBlockedEncoding({1, 4}, {4, 8}, {2, 2}, {1, 0});
  auto enc2 = createBlockedEncoding({2, 4}, {4, 8}, {2, 2}, {1, 0});

  size_t hash1 = detail::hashLayoutEncoding(enc1);
  size_t hash2 = detail::hashLayoutEncoding(enc2);

  EXPECT_NE(hash1, hash2);
}

TEST_F(NumericalInvarianceTest, DifferentThreadsPerWarpGivesDifferentHash) {
  auto enc1 = createBlockedEncoding({1, 4}, {4, 8}, {2, 2}, {1, 0});
  auto enc2 = createBlockedEncoding({1, 4}, {8, 4}, {2, 2}, {1, 0});

  size_t hash1 = detail::hashLayoutEncoding(enc1);
  size_t hash2 = detail::hashLayoutEncoding(enc2);

  EXPECT_NE(hash1, hash2);
}

TEST_F(NumericalInvarianceTest, DifferentWarpsPerCTAGivesDifferentHash) {
  auto enc1 = createBlockedEncoding({1, 4}, {4, 8}, {2, 2}, {1, 0});
  auto enc2 = createBlockedEncoding({1, 4}, {4, 8}, {4, 1}, {1, 0});

  size_t hash1 = detail::hashLayoutEncoding(enc1);
  size_t hash2 = detail::hashLayoutEncoding(enc2);

  EXPECT_NE(hash1, hash2);
}

TEST_F(NumericalInvarianceTest, DifferentOrderGivesDifferentHash) {
  auto enc1 = createBlockedEncoding({1, 4}, {4, 8}, {2, 2}, {1, 0});
  auto enc2 = createBlockedEncoding({1, 4}, {4, 8}, {2, 2}, {0, 1});

  size_t hash1 = detail::hashLayoutEncoding(enc1);
  size_t hash2 = detail::hashLayoutEncoding(enc2);

  EXPECT_NE(hash1, hash2);
}

TEST_F(NumericalInvarianceTest, MmaVersionMajorAffectsHash) {
  auto mma2 = createMmaEncoding(2, 0, {2, 2}, {16, 8});
  auto mma3 = createMmaEncoding(3, 0, {2, 2}, {16, 8, 16});

  size_t hash2 = detail::hashLayoutEncoding(mma2);
  size_t hash3 = detail::hashLayoutEncoding(mma3);

  EXPECT_NE(hash2, hash3);
}

TEST_F(NumericalInvarianceTest, DifferentInstrShapeAffectsHash) {
  auto mma1 = createMmaEncoding(2, 0, {2, 2}, {16, 8});
  auto mma2 = createMmaEncoding(2, 0, {2, 2}, {16, 16});

  size_t hash1 = detail::hashLayoutEncoding(mma1);
  size_t hash2 = detail::hashLayoutEncoding(mma2);

  EXPECT_NE(hash1, hash2);
}

//===----------------------------------------------------------------------===//
// Type Hashing Tests
//===----------------------------------------------------------------------===//

TEST_F(NumericalInvarianceTest, SameTypeGivesSameHash) {
  auto f32 = Float32Type::get(&ctx);
  size_t hash1 = detail::hashType(f32);
  size_t hash2 = detail::hashType(f32);
  EXPECT_EQ(hash1, hash2);
}

TEST_F(NumericalInvarianceTest, DifferentFloatTypesGiveDifferentHash) {
  auto f16 = Float16Type::get(&ctx);
  auto f32 = Float32Type::get(&ctx);

  size_t hashF16 = detail::hashType(f16);
  size_t hashF32 = detail::hashType(f32);

  EXPECT_NE(hashF16, hashF32);
}

TEST_F(NumericalInvarianceTest, DifferentIntTypesGiveDifferentHash) {
  auto i8 = IntegerType::get(&ctx, 8);
  auto i32 = IntegerType::get(&ctx, 32);

  size_t hash8 = detail::hashType(i8);
  size_t hash32 = detail::hashType(i32);

  EXPECT_NE(hash8, hash32);
}

TEST_F(NumericalInvarianceTest, TensorTypeIncludesShapeInHash) {
  auto f32 = Float32Type::get(&ctx);
  auto enc = createBlockedEncoding({1, 4}, {4, 8}, {2, 2}, {1, 0});

  auto tensor1 = RankedTensorType::get({64, 128}, f32, enc);
  auto tensor2 = RankedTensorType::get({128, 64}, f32, enc);

  size_t hash1 = detail::hashType(tensor1);
  size_t hash2 = detail::hashType(tensor2);

  EXPECT_NE(hash1, hash2);
}

TEST_F(NumericalInvarianceTest, TensorTypeIncludesEncodingInHash) {
  auto f32 = Float32Type::get(&ctx);
  auto enc1 = createBlockedEncoding({1, 4}, {4, 8}, {2, 2}, {1, 0});
  auto enc2 = createBlockedEncoding({2, 2}, {4, 8}, {2, 2}, {1, 0});

  auto tensor1 = RankedTensorType::get({64, 128}, f32, enc1);
  auto tensor2 = RankedTensorType::get({64, 128}, f32, enc2);

  size_t hash1 = detail::hashType(tensor1);
  size_t hash2 = detail::hashType(tensor2);

  EXPECT_NE(hash1, hash2);
}

//===----------------------------------------------------------------------===//
// Module-Level Invariance Tests
//===----------------------------------------------------------------------===//

TEST_F(NumericalInvarianceTest, IdenticalModulesHaveSameFingerprint) {
  const char *mlir = R"(
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @kernel(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
        %0 = arith.addf %arg0, %arg0 : tensor<64x64xf32, #blocked>
        tt.return %0 : tensor<64x64xf32, #blocked>
      }
    }
  )";

  auto module1 = parseModule(mlir);
  auto module2 = parseModule(mlir);
  ASSERT_TRUE(module1);
  ASSERT_TRUE(module2);

  auto inv1 = computeNumericalInvariance(*module1);
  auto inv2 = computeNumericalInvariance(*module2);

  EXPECT_EQ(inv1, inv2);
  EXPECT_EQ(inv1.fingerprint(), inv2.fingerprint());
}

TEST_F(NumericalInvarianceTest, DifferentSSANamesHaveSameFingerprint) {
  // Same computation, different SSA value names
  const char *mlir1 = R"(
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @kernel(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
        %0 = arith.addf %arg0, %arg0 : tensor<64x64xf32, #blocked>
        tt.return %0 : tensor<64x64xf32, #blocked>
      }
    }
  )";

  const char *mlir2 = R"(
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @kernel(%input: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
        %result = arith.addf %input, %input : tensor<64x64xf32, #blocked>
        tt.return %result : tensor<64x64xf32, #blocked>
      }
    }
  )";

  auto module1 = parseModule(mlir1);
  auto module2 = parseModule(mlir2);
  ASSERT_TRUE(module1);
  ASSERT_TRUE(module2);

  auto inv1 = computeNumericalInvariance(*module1);
  auto inv2 = computeNumericalInvariance(*module2);

  // The fingerprints should be the same since only SSA names differ
  EXPECT_EQ(inv1.fingerprint(), inv2.fingerprint());
}

TEST_F(NumericalInvarianceTest, DifferentFunctionNamesHaveSameFingerprint) {
  const char *mlir1 = R"(
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @matmul(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
        %0 = arith.addf %arg0, %arg0 : tensor<64x64xf32, #blocked>
        tt.return %0 : tensor<64x64xf32, #blocked>
      }
    }
  )";

  const char *mlir2 = R"(
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @matmul_v2(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
        %0 = arith.addf %arg0, %arg0 : tensor<64x64xf32, #blocked>
        tt.return %0 : tensor<64x64xf32, #blocked>
      }
    }
  )";

  auto module1 = parseModule(mlir1);
  auto module2 = parseModule(mlir2);
  ASSERT_TRUE(module1);
  ASSERT_TRUE(module2);

  auto inv1 = computeNumericalInvariance(*module1);
  auto inv2 = computeNumericalInvariance(*module2);

  // The fingerprints should be the same since only function names differ
  EXPECT_EQ(inv1.fingerprint(), inv2.fingerprint());
}

TEST_F(NumericalInvarianceTest, DifferentLayoutsHaveDifferentFingerprint) {
  const char *mlir1 = R"(
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @kernel(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
        %0 = arith.addf %arg0, %arg0 : tensor<64x64xf32, #blocked>
        tt.return %0 : tensor<64x64xf32, #blocked>
      }
    }
  )";

  const char *mlir2 = R"(
    #blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @kernel(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
        %0 = arith.addf %arg0, %arg0 : tensor<64x64xf32, #blocked>
        tt.return %0 : tensor<64x64xf32, #blocked>
      }
    }
  )";

  auto module1 = parseModule(mlir1);
  auto module2 = parseModule(mlir2);
  ASSERT_TRUE(module1);
  ASSERT_TRUE(module2);

  auto inv1 = computeNumericalInvariance(*module1);
  auto inv2 = computeNumericalInvariance(*module2);

  // Different layouts should produce different fingerprints
  EXPECT_NE(inv1.fingerprint(), inv2.fingerprint());
  EXPECT_NE(inv1.layoutHash, inv2.layoutHash);
}

TEST_F(NumericalInvarianceTest, DifferentElementTypesHaveDifferentFingerprint) {
  const char *mlir1 = R"(
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @kernel(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
        %0 = arith.addf %arg0, %arg0 : tensor<64x64xf32, #blocked>
        tt.return %0 : tensor<64x64xf32, #blocked>
      }
    }
  )";

  const char *mlir2 = R"(
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @kernel(%arg0: tensor<64x64xf16, #blocked>) -> tensor<64x64xf16, #blocked> {
        %0 = arith.addf %arg0, %arg0 : tensor<64x64xf16, #blocked>
        tt.return %0 : tensor<64x64xf16, #blocked>
      }
    }
  )";

  auto module1 = parseModule(mlir1);
  auto module2 = parseModule(mlir2);
  ASSERT_TRUE(module1);
  ASSERT_TRUE(module2);

  auto inv1 = computeNumericalInvariance(*module1);
  auto inv2 = computeNumericalInvariance(*module2);

  // Different element types should produce different fingerprints
  EXPECT_NE(inv1.fingerprint(), inv2.fingerprint());
}

TEST_F(NumericalInvarianceTest, DifferentOperationsHaveDifferentFingerprint) {
  const char *mlir1 = R"(
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @kernel(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
        %0 = arith.addf %arg0, %arg0 : tensor<64x64xf32, #blocked>
        tt.return %0 : tensor<64x64xf32, #blocked>
      }
    }
  )";

  const char *mlir2 = R"(
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
      tt.func @kernel(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
        %0 = arith.mulf %arg0, %arg0 : tensor<64x64xf32, #blocked>
        tt.return %0 : tensor<64x64xf32, #blocked>
      }
    }
  )";

  auto module1 = parseModule(mlir1);
  auto module2 = parseModule(mlir2);
  ASSERT_TRUE(module1);
  ASSERT_TRUE(module2);

  auto inv1 = computeNumericalInvariance(*module1);
  auto inv2 = computeNumericalInvariance(*module2);

  // Different operations (addf vs mulf) should produce different fingerprints
  EXPECT_NE(inv1.fingerprint(), inv2.fingerprint());
  EXPECT_NE(inv1.computationDagHash, inv2.computationDagHash);
}

//===----------------------------------------------------------------------===//
// Print/Diff Tests
//===----------------------------------------------------------------------===//

TEST_F(NumericalInvarianceTest, PrintOutputContainsAllFields) {
  NumericalInvariance inv;
  inv.computationDagHash = 0x123456789ABCDEF0;
  inv.layoutHash = 0xFEDCBA9876543210;
  inv.hwConfigHash = 0x1111111111111111;
  inv.dtypeSignature = "f16,f16->f32";

  std::string output;
  llvm::raw_string_ostream os(output);
  inv.print(os);

  // Check that the output contains expected fields
  EXPECT_THAT(output, testing::HasSubstr("NumericalInvariance"));
  EXPECT_THAT(output, testing::HasSubstr("computationDagHash"));
  EXPECT_THAT(output, testing::HasSubstr("layoutHash"));
  EXPECT_THAT(output, testing::HasSubstr("hwConfigHash"));
  EXPECT_THAT(output, testing::HasSubstr("dtypeSignature"));
  EXPECT_THAT(output, testing::HasSubstr("f16,f16->f32"));
  EXPECT_THAT(output, testing::HasSubstr("fingerprint"));
}

TEST_F(NumericalInvarianceTest, PrintDiffShowsDifferences) {
  NumericalInvariance inv1;
  inv1.computationDagHash = 12345;
  inv1.dtypeSignature = "f16->f32";

  NumericalInvariance inv2;
  inv2.computationDagHash = 99999;
  inv2.dtypeSignature = "f32->f32";

  std::string output;
  llvm::raw_string_ostream os(output);
  NumericalInvariance::printDiff(os, inv1, inv2);

  // Check that diff output shows differences
  EXPECT_THAT(output, testing::HasSubstr("[DIFF]"));
  EXPECT_THAT(output, testing::HasSubstr("computationDagHash"));
  EXPECT_THAT(output, testing::HasSubstr("dtypeSignature"));
  EXPECT_THAT(output, testing::HasSubstr("LHS"));
  EXPECT_THAT(output, testing::HasSubstr("RHS"));
}

TEST_F(NumericalInvarianceTest, PrintDiffShowsSameForIdenticalInvariances) {
  NumericalInvariance inv1;
  inv1.computationDagHash = 12345;
  inv1.dtypeSignature = "f16->f32";

  NumericalInvariance inv2 = inv1;  // Copy

  std::string output;
  llvm::raw_string_ostream os(output);
  NumericalInvariance::printDiff(os, inv1, inv2);

  // Should show that they are the same
  EXPECT_THAT(output, testing::HasSubstr("[SAME]"));
}

//===----------------------------------------------------------------------===//
// Null/Empty Handling Tests
//===----------------------------------------------------------------------===//

TEST_F(NumericalInvarianceTest, NullEncodingHashesZero) {
  size_t hash = detail::hashLayoutEncoding(Attribute());
  EXPECT_EQ(hash, 0u);
}

} // namespace
} // namespace mlir::triton::gpu

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
