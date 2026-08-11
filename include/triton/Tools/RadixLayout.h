#ifndef TRITON_TOOLS_RADIXLAYOUT_H
#define TRITON_TOOLS_RADIXLAYOUT_H

#include <cstdint>
#include <optional>

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton {

// A mixed-radix coordinate map whose image is selected by a partition index.
class PartitionedRadixMap {
public:
  enum class Mode { Distribute, Replicate };

  struct Partition {
    int32_t span;
    unsigned bitOffset;
    unsigned bitCount;

    int32_t getStride() const;
  };

  struct Dimension {
    int32_t extent;
    std::optional<Partition> partition;
  };

  static FailureOr<PartitionedRadixMap>
  create(llvm::ArrayRef<Dimension> dimensions,
         llvm::ArrayRef<unsigned> dimensionOrder, unsigned partitionCount,
         Mode mode);

  llvm::ArrayRef<Dimension> getDimensions() const { return dimensions; }
  llvm::ArrayRef<unsigned> getDimensionOrder() const { return dimensionOrder; }
  unsigned getPartitionCount() const { return partitionCount; }
  Mode getMode() const { return mode; }
  uint32_t getConstrainedPartitionMask() const {
    return constrainedPartitionMask;
  }
  uint32_t getFreePartitionMask() const { return freePartitionMask; }
  unsigned getFreePartitionCount() const { return freePartitionCount; }

  FailureOr<uint64_t> getLocalOrdinalCount(unsigned partition) const;
  // These are mutual inverses on a partition's image.
  FailureOr<llvm::SmallVector<int32_t>> delinearize(unsigned partition,
                                                    uint64_t ordinal) const;
  FailureOr<uint64_t> linearize(unsigned partition,
                                llvm::ArrayRef<int32_t> coordinates) const;
  FailureOr<PartitionedRadixMap> removeUnitDimension(unsigned dimension) const;

private:
  llvm::SmallVector<Dimension> dimensions;
  llvm::SmallVector<unsigned> dimensionOrder;
  unsigned partitionCount = 1;
  Mode mode = Mode::Replicate;
  uint32_t constrainedPartitionMask = 0;
  uint32_t freePartitionMask = 0;
  unsigned freePartitionCount = 1;
};

} // namespace mlir::triton

#endif // TRITON_TOOLS_RADIXLAYOUT_H
