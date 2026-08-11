#include <gtest/gtest.h>

#include <limits>

#include "triton/Tools/RadixLayout.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton {
namespace {

using Dimension = PartitionedRadixMap::Dimension;
using Mode = PartitionedRadixMap::Mode;
using Partition = PartitionedRadixMap::Partition;

TEST(PartitionedRadixMapTest, MixedRadixOrderAndRoundTrip) {
  auto map = PartitionedRadixMap::create({{3, {}}, {5, {}}}, {0, 1}, 1,
                                         Mode::Replicate);
  ASSERT_TRUE(succeeded(map));
  EXPECT_EQ(*map->getLocalOrdinalCount(0), 15);
  EXPECT_EQ(*map->delinearize(0, 0), llvm::SmallVector<int32_t>({0, 0}));
  EXPECT_EQ(*map->delinearize(0, 1), llvm::SmallVector<int32_t>({1, 0}));
  EXPECT_EQ(*map->delinearize(0, 3), llvm::SmallVector<int32_t>({0, 1}));
  EXPECT_EQ(*map->delinearize(0, 14), llvm::SmallVector<int32_t>({2, 4}));
  for (uint64_t ordinal = 0; ordinal < 15; ++ordinal) {
    auto coordinates = map->delinearize(0, ordinal);
    ASSERT_TRUE(succeeded(coordinates));
    EXPECT_EQ(*map->linearize(0, *coordinates), ordinal);
  }

  auto transposed = PartitionedRadixMap::create({{3, {}}, {5, {}}}, {1, 0}, 1,
                                                Mode::Replicate);
  ASSERT_TRUE(succeeded(transposed));
  EXPECT_EQ(*transposed->delinearize(0, 1), llvm::SmallVector<int32_t>({0, 1}));
  EXPECT_EQ(*transposed->linearize(0, {2, 4}), 14);
}

TEST(PartitionedRadixMapTest, DistributedImagesCoverExactlyOnce) {
  auto map = PartitionedRadixMap::create({{3, {}}, {48, {}}}, {0, 1}, 4,
                                         Mode::Distribute);
  ASSERT_TRUE(succeeded(map));
  EXPECT_EQ(map->getConstrainedPartitionMask(), 0);
  EXPECT_EQ(map->getFreePartitionMask(), 3);
  EXPECT_EQ(map->getFreePartitionCount(), 4);
  llvm::SmallVector<unsigned> coverage(3 * 48);
  for (unsigned partition = 0; partition < 4; ++partition) {
    EXPECT_EQ(*map->getLocalOrdinalCount(partition), 36);
    for (uint64_t ordinal = 0; ordinal < 36; ++ordinal) {
      auto coordinates = map->delinearize(partition, ordinal);
      ASSERT_TRUE(succeeded(coordinates));
      ++coverage[(*coordinates)[0] + 3 * (*coordinates)[1]];
      EXPECT_EQ(*map->linearize(partition, *coordinates), ordinal);
    }
  }
  EXPECT_TRUE(
      llvm::all_of(coverage, [](unsigned count) { return count == 1; }));
}

TEST(PartitionedRadixMapTest, ReplicatedPartitionsHaveEqualImages) {
  auto map = PartitionedRadixMap::create({{3, {}}, {5, {}}}, {0, 1}, 4,
                                         Mode::Replicate);
  ASSERT_TRUE(succeeded(map));
  for (unsigned partition = 0; partition < 4; ++partition) {
    EXPECT_EQ(*map->getLocalOrdinalCount(partition), 15);
    for (uint64_t ordinal = 0; ordinal < 15; ++ordinal)
      EXPECT_EQ(*map->delinearize(partition, ordinal),
                *map->delinearize(0, ordinal));
  }
}

TEST(PartitionedRadixMapTest, ConstrainedAndFreePartitionBits) {
  llvm::SmallVector<Dimension> dimensions = {{16, Partition{2, 0, 1}},
                                             {12, Partition{3, 2, 1}}};
  auto map =
      PartitionedRadixMap::create(dimensions, {0, 1}, 8, Mode::Distribute);
  ASSERT_TRUE(succeeded(map));
  EXPECT_EQ(map->getConstrainedPartitionMask(), 5);
  EXPECT_EQ(map->getFreePartitionMask(), 2);
  EXPECT_EQ(map->getFreePartitionCount(), 2);
  llvm::SmallVector<unsigned> coverage(16 * 12);
  for (unsigned partition = 0; partition < 8; ++partition) {
    auto count = map->getLocalOrdinalCount(partition);
    ASSERT_TRUE(succeeded(count));
    for (uint64_t ordinal = 0; ordinal < *count; ++ordinal) {
      auto coordinates = map->delinearize(partition, ordinal);
      ASSERT_TRUE(succeeded(coordinates));
      ++coverage[(*coordinates)[0] + 16 * (*coordinates)[1]];
      EXPECT_EQ(*map->linearize(partition, *coordinates), ordinal);
    }
  }
  EXPECT_TRUE(
      llvm::all_of(coverage, [](unsigned count) { return count == 1; }));
}

TEST(PartitionedRadixMapTest, ConstrainedAndFreeTailsCoverExactlyOnce) {
  llvm::SmallVector<Dimension> dimensions = {{5, Partition{1, 0, 1}}, {7, {}}};
  auto map =
      PartitionedRadixMap::create(dimensions, {0, 1}, 4, Mode::Distribute);
  ASSERT_TRUE(succeeded(map));
  EXPECT_EQ(map->getConstrainedPartitionMask(), 1);
  EXPECT_EQ(map->getFreePartitionMask(), 2);
  EXPECT_EQ(*map->getLocalOrdinalCount(0), 11);
  EXPECT_EQ(*map->getLocalOrdinalCount(1), 7);
  EXPECT_EQ(*map->getLocalOrdinalCount(2), 10);
  EXPECT_EQ(*map->getLocalOrdinalCount(3), 7);

  llvm::SmallVector<unsigned> coverage(5 * 7);
  for (unsigned partition = 0; partition < 4; ++partition) {
    auto count = map->getLocalOrdinalCount(partition);
    ASSERT_TRUE(succeeded(count));
    for (uint64_t ordinal = 0; ordinal < *count; ++ordinal) {
      auto coordinates = map->delinearize(partition, ordinal);
      ASSERT_TRUE(succeeded(coordinates));
      ++coverage[(*coordinates)[0] + 5 * (*coordinates)[1]];
      EXPECT_EQ(*map->linearize(partition, *coordinates), ordinal);
    }
  }
  EXPECT_TRUE(
      llvm::all_of(coverage, [](unsigned count) { return count == 1; }));
}

TEST(PartitionedRadixMapTest, FreePartitionsOwnContiguousRanges) {
  auto map = PartitionedRadixMap::create({{17, {}}}, {0}, 2, Mode::Distribute);
  ASSERT_TRUE(succeeded(map));
  EXPECT_EQ(*map->getLocalOrdinalCount(0), 9);
  EXPECT_EQ(*map->getLocalOrdinalCount(1), 8);
  EXPECT_EQ(*map->delinearize(0, 0), llvm::SmallVector<int32_t>({0}));
  EXPECT_EQ(*map->delinearize(0, 8), llvm::SmallVector<int32_t>({8}));
  EXPECT_EQ(*map->delinearize(1, 0), llvm::SmallVector<int32_t>({9}));
  EXPECT_EQ(*map->delinearize(1, 7), llvm::SmallVector<int32_t>({16}));
}

TEST(PartitionedRadixMapTest, InverseIsDefinedExactlyOnPartitionImage) {
  auto map = PartitionedRadixMap::create({{5, Partition{1, 0, 1}}, {7, {}}},
                                         {1, 0}, 4, Mode::Distribute);
  ASSERT_TRUE(succeeded(map));
  for (int32_t row = 0; row < 5; ++row) {
    for (int32_t col = 0; col < 7; ++col) {
      unsigned owners = 0;
      for (unsigned partition = 0; partition < 4; ++partition) {
        auto ordinal = map->linearize(partition, {row, col});
        if (failed(ordinal))
          continue;
        ++owners;
        EXPECT_EQ(*map->delinearize(partition, *ordinal),
                  llvm::SmallVector<int32_t>({row, col}));
      }
      EXPECT_EQ(owners, 1);
    }
  }
}

TEST(PartitionedRadixMapTest, RemoveUnitDimensionPreservesPartitionImages) {
  auto parent = PartitionedRadixMap::create({{1, {}}, {33, {}}}, {1, 0}, 4,
                                            Mode::Distribute);
  ASSERT_TRUE(succeeded(parent));
  auto slice = parent->removeUnitDimension(0);
  ASSERT_TRUE(succeeded(slice));
  for (unsigned partition = 0; partition < 4; ++partition) {
    auto parentCount = parent->getLocalOrdinalCount(partition);
    auto sliceCount = slice->getLocalOrdinalCount(partition);
    ASSERT_TRUE(succeeded(parentCount));
    ASSERT_TRUE(succeeded(sliceCount));
    ASSERT_EQ(*parentCount, *sliceCount);
    for (uint64_t ordinal = 0; ordinal < *parentCount; ++ordinal) {
      auto parentCoordinate = parent->delinearize(partition, ordinal);
      auto sliceCoordinate = slice->delinearize(partition, ordinal);
      ASSERT_TRUE(succeeded(parentCoordinate));
      ASSERT_TRUE(succeeded(sliceCoordinate));
      EXPECT_EQ((*parentCoordinate)[0], 0);
      EXPECT_EQ((*parentCoordinate)[1], (*sliceCoordinate)[0]);
      EXPECT_EQ(*slice->linearize(partition, *sliceCoordinate), ordinal);
    }
  }
  EXPECT_TRUE(failed(parent->removeUnitDimension(1)));
}

TEST(PartitionedRadixMapTest, TailAndEmptyPartitions) {
  auto map = PartitionedRadixMap::create({{3, Partition{1, 0, 2}}}, {0}, 4,
                                         Mode::Distribute);
  ASSERT_TRUE(succeeded(map));
  for (unsigned partition = 0; partition < 3; ++partition) {
    EXPECT_EQ(*map->getLocalOrdinalCount(partition), 1);
    EXPECT_EQ(*map->delinearize(partition, 0),
              llvm::SmallVector<int32_t>({static_cast<int32_t>(partition)}));
  }
  EXPECT_EQ(*map->getLocalOrdinalCount(3), 0);
  EXPECT_TRUE(failed(map->delinearize(3, 0)));
  EXPECT_TRUE(failed(map->linearize(0, {1})));
}

TEST(PartitionedRadixMapTest, RejectsInvalidDescriptors) {
  EXPECT_TRUE(
      failed(PartitionedRadixMap::create({{3, {}}}, {0}, 3, Mode::Distribute)));
  EXPECT_TRUE(failed(PartitionedRadixMap::create(
      {{8, Partition{2, 0, 1}}, {8, Partition{2, 0, 1}}}, {0, 1}, 2,
      Mode::Distribute)));
  EXPECT_TRUE(failed(PartitionedRadixMap::create({{8, Partition{2, 1, 1}}}, {0},
                                                 2, Mode::Distribute)));
  EXPECT_TRUE(failed(PartitionedRadixMap::create(
      {{8, Partition{2, std::numeric_limits<unsigned>::max(), 2}}}, {0}, 2,
      Mode::Distribute)));
  EXPECT_TRUE(failed(PartitionedRadixMap::create({{8, Partition{2, 0, 1}}}, {0},
                                                 2, Mode::Replicate)));
  EXPECT_TRUE(failed(PartitionedRadixMap::create({{50000, {}}, {50000, {}}},
                                                 {0, 1}, 1, Mode::Replicate)));
  EXPECT_TRUE(failed(PartitionedRadixMap::create({{3, {}}, {5, {}}}, {0}, 1,
                                                 Mode::Replicate)));
  EXPECT_TRUE(failed(PartitionedRadixMap::create({{3, {}}, {5, {}}}, {0, 0}, 1,
                                                 Mode::Replicate)));
  EXPECT_TRUE(failed(PartitionedRadixMap::create({{3, {}}, {5, {}}}, {0, 2}, 1,
                                                 Mode::Replicate)));
}

} // namespace
} // namespace mlir::triton
