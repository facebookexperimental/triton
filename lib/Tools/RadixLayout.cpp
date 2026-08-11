#include "triton/Tools/RadixLayout.h"

#include <limits>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton {
namespace {

static FailureOr<uint64_t> checkedProduct(llvm::ArrayRef<uint64_t> values) {
  uint64_t product = 1;
  for (uint64_t value : values) {
    if (value != 0 && product > std::numeric_limits<uint64_t>::max() / value)
      return failure();
    product *= value;
  }
  return product;
}

static unsigned compressBits(unsigned value, uint32_t mask) {
  unsigned compressed = 0;
  unsigned outputBit = 0;
  for (unsigned inputBit = 0; mask != 0; ++inputBit) {
    if (!(mask & (uint32_t{1} << inputBit)))
      continue;
    compressed |= ((value >> inputBit) & 1u) << outputBit++;
    mask &= ~(uint32_t{1} << inputBit);
  }
  return compressed;
}

static unsigned getPartitionDigit(const PartitionedRadixMap::Partition &part,
                                  unsigned partition) {
  uint32_t mask = (uint32_t{1} << part.bitCount) - 1;
  return (partition >> part.bitOffset) & mask;
}

static FailureOr<uint64_t>
getDimensionOrdinalCount(const PartitionedRadixMap::Dimension &dim,
                         unsigned partition) {
  if (!dim.partition)
    return static_cast<uint64_t>(dim.extent);
  const auto &part = *dim.partition;
  uint64_t base = uint64_t(getPartitionDigit(part, partition)) * part.span;
  if (base >= static_cast<uint64_t>(dim.extent))
    return uint64_t{0};
  uint64_t remaining = dim.extent - base;
  uint64_t stride = part.getStride();
  return (remaining / stride) * part.span +
         std::min<uint64_t>(remaining % stride, part.span);
}

static FailureOr<llvm::SmallVector<uint64_t>> getDimensionOrdinalCounts(
    llvm::ArrayRef<PartitionedRadixMap::Dimension> dimensions,
    unsigned partition) {
  llvm::SmallVector<uint64_t> counts;
  counts.reserve(dimensions.size());
  for (const auto &dim : dimensions) {
    auto count = getDimensionOrdinalCount(dim, partition);
    if (failed(count))
      return failure();
    counts.push_back(*count);
  }
  return counts;
}

struct OrdinalRange {
  uint64_t base;
  uint64_t count;
};

static OrdinalRange getOrdinalRange(uint64_t total, unsigned partition,
                                    uint32_t freePartitionMask,
                                    unsigned freePartitionCount) {
  uint64_t freeOrdinal = compressBits(partition, freePartitionMask);
  uint64_t quotient = total / freePartitionCount;
  uint64_t remainder = total % freePartitionCount;
  return {freeOrdinal * quotient + std::min(freeOrdinal, remainder),
          quotient + static_cast<uint64_t>(freeOrdinal < remainder)};
}

} // namespace

int32_t PartitionedRadixMap::Partition::getStride() const {
  return static_cast<int32_t>(uint64_t(span) << bitCount);
}

FailureOr<PartitionedRadixMap>
PartitionedRadixMap::create(llvm::ArrayRef<Dimension> dimensions,
                            llvm::ArrayRef<unsigned> dimensionOrder,
                            unsigned partitionCount, Mode mode) {
  if (!llvm::isPowerOf2_32(partitionCount))
    return failure();
  if (dimensionOrder.size() != dimensions.size())
    return failure();
  llvm::SmallVector<bool> seenDimensions(dimensions.size());
  for (unsigned dim : dimensionOrder) {
    if (dim >= dimensions.size() || seenDimensions[dim])
      return failure();
    seenDimensions[dim] = true;
  }
  unsigned partitionBits = llvm::Log2_32(partitionCount);
  uint32_t constrainedMask = 0;
  llvm::SmallVector<uint64_t> extents;
  extents.reserve(dimensions.size());
  for (const Dimension &dim : dimensions) {
    if (dim.extent <= 0)
      return failure();
    extents.push_back(dim.extent);
    if (!dim.partition)
      continue;
    const Partition &part = *dim.partition;
    if (mode == Mode::Replicate || part.span <= 0 || part.span > dim.extent ||
        part.bitCount == 0 || part.bitOffset > partitionBits ||
        part.bitCount > partitionBits - part.bitOffset)
      return failure();
    uint64_t stride = uint64_t(part.span) << part.bitCount;
    if (stride > std::numeric_limits<int32_t>::max())
      return failure();
    uint32_t mask = ((uint32_t{1} << part.bitCount) - 1) << part.bitOffset;
    if (constrainedMask & mask)
      return failure();
    constrainedMask |= mask;
  }
  auto totalElements = checkedProduct(extents);
  if (failed(totalElements) ||
      *totalElements > std::numeric_limits<int32_t>::max())
    return failure();

  PartitionedRadixMap result;
  result.dimensions.assign(dimensions.begin(), dimensions.end());
  result.dimensionOrder.assign(dimensionOrder.begin(), dimensionOrder.end());
  result.partitionCount = partitionCount;
  result.mode = mode;
  result.constrainedPartitionMask = constrainedMask;
  uint32_t partitionMask = partitionCount - 1;
  result.freePartitionMask =
      mode == Mode::Replicate ? 0 : partitionMask & ~constrainedMask;
  result.freePartitionCount = uint32_t{1}
                              << llvm::popcount(result.freePartitionMask);
  return result;
}

FailureOr<uint64_t>
PartitionedRadixMap::getLocalOrdinalCount(unsigned partition) const {
  if (partition >= partitionCount)
    return failure();
  auto counts = getDimensionOrdinalCounts(dimensions, partition);
  if (failed(counts))
    return failure();
  auto total = checkedProduct(*counts);
  if (failed(total))
    return failure();
  return getOrdinalRange(*total, partition, freePartitionMask,
                         freePartitionCount)
      .count;
}

FailureOr<llvm::SmallVector<int32_t>>
PartitionedRadixMap::delinearize(unsigned partition,
                                 uint64_t localOrdinal) const {
  if (partition >= partitionCount)
    return failure();
  auto counts = getDimensionOrdinalCounts(dimensions, partition);
  if (failed(counts))
    return failure();
  auto total = checkedProduct(*counts);
  if (failed(total))
    return failure();
  OrdinalRange range =
      getOrdinalRange(*total, partition, freePartitionMask, freePartitionCount);
  if (localOrdinal >= range.count)
    return failure();
  uint64_t ordinal = range.base + localOrdinal;

  llvm::SmallVector<int32_t> coordinates(dimensions.size());
  for (unsigned logicalDim : dimensionOrder) {
    const Dimension &dim = dimensions[logicalDim];
    uint64_t count = (*counts)[logicalDim];
    uint64_t local = ordinal % count;
    ordinal /= count;
    uint64_t coordinate = local;
    if (dim.partition) {
      const Partition &part = *dim.partition;
      uint64_t base = uint64_t(getPartitionDigit(part, partition)) * part.span;
      coordinate =
          base + (local / part.span) * part.getStride() + local % part.span;
    }
    if (coordinate >= static_cast<uint64_t>(dim.extent))
      return failure();
    coordinates[logicalDim] = coordinate;
  }
  return coordinates;
}

FailureOr<uint64_t>
PartitionedRadixMap::linearize(unsigned partition,
                               llvm::ArrayRef<int32_t> coordinates) const {
  if (partition >= partitionCount || coordinates.size() != dimensions.size())
    return failure();
  auto counts = getDimensionOrdinalCounts(dimensions, partition);
  if (failed(counts))
    return failure();

  uint64_t ordinal = 0;
  uint64_t multiplier = 1;
  for (unsigned logicalDim : dimensionOrder) {
    int32_t coordinate = coordinates[logicalDim];
    const Dimension &dim = dimensions[logicalDim];
    uint64_t count = (*counts)[logicalDim];
    if (coordinate < 0 || coordinate >= dim.extent)
      return failure();
    uint64_t local = coordinate;
    if (dim.partition) {
      const Partition &part = *dim.partition;
      uint64_t base = uint64_t(getPartitionDigit(part, partition)) * part.span;
      if (static_cast<uint64_t>(coordinate) < base)
        return failure();
      uint64_t delta = coordinate - base;
      uint64_t inner = delta % part.getStride();
      if (inner >= static_cast<uint64_t>(part.span))
        return failure();
      local = (delta / part.getStride()) * part.span + inner;
    }
    if (local >= count ||
        local > (std::numeric_limits<uint64_t>::max() - ordinal) / multiplier)
      return failure();
    ordinal += local * multiplier;
    if (count != 0 && multiplier > std::numeric_limits<uint64_t>::max() / count)
      return failure();
    multiplier *= count;
  }

  auto total = checkedProduct(*counts);
  if (failed(total))
    return failure();
  OrdinalRange range =
      getOrdinalRange(*total, partition, freePartitionMask, freePartitionCount);
  if (ordinal < range.base || ordinal - range.base >= range.count)
    return failure();
  return ordinal - range.base;
}

FailureOr<PartitionedRadixMap>
PartitionedRadixMap::removeUnitDimension(unsigned dimension) const {
  if (dimension >= dimensions.size() || dimensions[dimension].extent != 1)
    return failure();

  llvm::SmallVector<Dimension> restrictedDimensions(dimensions.begin(),
                                                    dimensions.end());
  restrictedDimensions.erase(restrictedDimensions.begin() + dimension);
  llvm::SmallVector<unsigned> restrictedOrder;
  restrictedOrder.reserve(dimensionOrder.size() - 1);
  for (unsigned orderedDimension : dimensionOrder) {
    if (orderedDimension == dimension)
      continue;
    restrictedOrder.push_back(
        orderedDimension > dimension ? orderedDimension - 1 : orderedDimension);
  }
  return create(restrictedDimensions, restrictedOrder, partitionCount, mode);
}

} // namespace mlir::triton
