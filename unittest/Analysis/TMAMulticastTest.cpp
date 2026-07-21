#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAMulticast.h"

#include "llvm/ADT/SmallBitVector.h"
#include <gtest/gtest.h>

using mlir::triton::nvidia_gpu::TMAClusterGeometry;

namespace {

llvm::SmallBitVector axes(std::initializer_list<unsigned> values) {
  llvm::SmallBitVector result(3);
  for (unsigned value : values)
    result.set(value);
  return result;
}

TEST(TMAMulticastGeometry, RankCoordinatesAreXMajor) {
  TMAClusterGeometry geometry{{2, 4, 1}};
  EXPECT_EQ(geometry.size(), 8u);
  EXPECT_EQ(geometry.coordinates(0), (llvm::SmallVector<int, 3>{0, 0, 0}));
  EXPECT_EQ(geometry.coordinates(3), (llvm::SmallVector<int, 3>{1, 1, 0}));
  EXPECT_EQ(geometry.coordinates(7), (llvm::SmallVector<int, 3>{1, 3, 0}));
}

TEST(TMAMulticastGeometry, OneDimensionalGroups) {
  for (int size : {2, 4, 8}) {
    TMAClusterGeometry geometry{{size, 1, 1}};
    uint16_t fullMask = uint16_t((1u << size) - 1);
    for (unsigned rank = 0; rank < geometry.size(); ++rank) {
      EXPECT_EQ(geometry.maskFor(rank, axes({0})), fullMask);
      EXPECT_EQ(geometry.leaderFor(rank, axes({0})), 0u);
    }
  }
}

TEST(TMAMulticastGeometry, RectangularOperandGroups) {
  TMAClusterGeometry geometry{{2, 4, 1}};
  EXPECT_EQ(geometry.maskFor(0, axes({1})), 0x55);
  EXPECT_EQ(geometry.maskFor(1, axes({1})), 0xaa);
  EXPECT_EQ(geometry.leaderFor(7, axes({1})), 1u);
  EXPECT_EQ(geometry.maskFor(0, axes({0})), 0x03);
  EXPECT_EQ(geometry.maskFor(5, axes({0})), 0x30);
  EXPECT_EQ(geometry.leaderFor(7, axes({0})), 6u);
}

TEST(TMAMulticastGeometry, SingletonWhenNoAxisBroadcasts) {
  TMAClusterGeometry geometry{{4, 2, 1}};
  for (unsigned rank = 0; rank < geometry.size(); ++rank) {
    EXPECT_EQ(geometry.maskFor(rank, axes({})), uint16_t(1u << rank));
    EXPECT_EQ(geometry.leaderFor(rank, axes({})), rank);
  }
}

} // namespace
