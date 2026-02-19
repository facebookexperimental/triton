#include "IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tlx-storage-alias-allocation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace tlx {

// After replacing a storage_alias_local_alloc with a local_alias that has
// an expanded type (e.g., from buffer overlap shape expansion), we need to
// update any ops that capture the value and propagate types to block
// arguments. In particular, WarpSpecializeOp captures values as operands
// and each partition region has block arguments whose types must match
// the capture types (verified by WarpSpecializeOp::verify).
static void updateBlockArgTypesForUsers(Value newValue) {
  Type newType = newValue.getType();
  for (OpOperand &use : newValue.getUses()) {
    Operation *user = use.getOwner();
    if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
      unsigned idx = use.getOperandNumber();
      for (Region *partition : wsOp.getPartitionRegions()) {
        if (idx < partition->getNumArguments()) {
          partition->getArgument(idx).setType(newType);
        }
      }
    }
  }
}

// Helper function to collect all MemDescIndexOp operations that use a given
// memdesc value, following through MemDescReinterpretOp, LocalAliasOp, and
// WarpSpecializeOp captures (to the corresponding partition block arguments).
static void
collectMemDescIndexOps(Value memDesc,
                       SmallVectorImpl<ttg::MemDescIndexOp> &result) {
  for (auto &use : memDesc.getUses()) {
    Operation *user = use.getOwner();
    if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(user)) {
      result.push_back(indexOp);
    } else if (auto reinterpret = dyn_cast<ttg::MemDescReinterpretOp>(user)) {
      // Follow through reinterpret ops
      collectMemDescIndexOps(reinterpret.getResult(), result);
    } else if (auto alias = dyn_cast<LocalAliasOp>(user)) {
      // Follow through nested aliases
      collectMemDescIndexOps(alias.getResult(), result);
    } else if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
      // Follow through warp_specialize captures to partition block arguments.
      // The alias may be captured as an operand; the corresponding block arg
      // in each partition region is used inside the isolated region.
      unsigned idx = use.getOperandNumber();
      for (Region *partition : wsOp.getPartitionRegions()) {
        if (idx < partition->getNumArguments()) {
          collectMemDescIndexOps(partition->getArgument(idx), result);
        }
      }
    }
  }
}

LogicalResult materializeStorageAliasAllocations(
    ModuleOp m, const DenseMap<Value, std::pair<int64_t, int64_t>> &offsetMap,
    DenseMap<Value, std::pair<int64_t, int64_t>> &localAliasOffsetMap) {
  LDBG("materializeStorageAliasAllocations");

  OpBuilder builder(m.getContext());

  // Map from storage_alias_spec SSA value to its materialized allocation
  DenseMap<Value, Value> specToAlloc;

  // Collect all storage_alias_spec operations
  SmallVector<StorageAliasSpecOp> specOps;
  m.walk([&](StorageAliasSpecOp specOp) { specOps.push_back(specOp); });

  // First pass: create LocalAllocOp/TMEMAllocOp for each storage_alias_spec
  for (auto specOp : specOps) {
    builder.setInsertionPoint(specOp);

    auto bufferShapeAttr = specOp.getBufferShapeAttr();
    if (!bufferShapeAttr) {
      specOp.emitError("storage_alias_spec has no shape set; "
                       "run TLXStorageAliasSizeDefinition pass first");
      return failure();
    }

    auto bufferShape = bufferShapeAttr.asArrayRef();
    auto storage = specOp.getStorage();

    Value allocResult;
    if (storage == StorageKind::smem) {
      // SMEM: 1D allocation
      if (bufferShape.size() != 1) {
        specOp.emitError("SMEM storage_alias_spec must have 1D shape, got ")
            << bufferShape.size() << "D";
        return failure();
      }

      int64_t bufferSizeBytes = bufferShape[0];
      LDBG("Creating SMEM allocation with size " << bufferSizeBytes);

      // Create a 1D byte buffer type for the allocation
      auto elemType = IntegerType::get(m.getContext(), 8);
      SmallVector<int64_t> shape{bufferSizeBytes};

      // Create a shared encoding with default parameters
      auto ctaLayout = ttg::CTALayoutAttr::get(m.getContext(), {1}, {1}, {0});
      auto sharedEncoding = ttg::SwizzledSharedEncodingAttr::get(
          m.getContext(), /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1,
          /*order=*/{0}, ctaLayout);

      auto memorySpace = ttg::SharedMemorySpaceAttr::get(m.getContext());
      auto memDescType =
          ttg::MemDescType::get(shape, elemType, sharedEncoding, memorySpace,
                                /*mutableMemory=*/true);
      auto allocOp =
          builder.create<ttg::LocalAllocOp>(specOp.getLoc(), memDescType);
      allocResult = allocOp.getResult();
    } else {
      // TMEM: 2D allocation
      assert(storage == StorageKind::tmem && "Unexpected storage kind");

      if (bufferShape.size() != 2) {
        specOp.emitError("TMEM storage_alias_spec must have 2D shape, got ")
            << bufferShape.size() << "D";
        return failure();
      }

      int64_t blockM = bufferShape[0];
      int64_t blockN = bufferShape[1];
      LDBG("Creating TMEM allocation with shape [" << blockM << ", " << blockN
                                                   << "]");

      auto tmemElemType = IntegerType::get(m.getContext(), 32);
      SmallVector<int64_t> tmemShape{blockM, blockN};
      auto memorySpace = ttng::TensorMemorySpaceAttr::get(m.getContext());
      auto tmemEncoding = ttng::TensorMemoryEncodingAttr::get(
          m.getContext(), blockM, blockN,
          /*colStride=*/1, /*CTASplitM=*/1, /*CTASplitN=*/1);
      auto memDescType =
          ttg::MemDescType::get(tmemShape, tmemElemType, tmemEncoding,
                                memorySpace, /*mutableMemory=*/true);
      auto allocOp = builder.create<ttng::TMEMAllocOp>(specOp.getLoc(),
                                                       memDescType, nullptr);
      allocResult = allocOp.getResult();
    }

    specToAlloc[specOp.getResult()] = allocResult;
  }

  // Second pass: replace storage_alias_local_alloc with LocalAliasOp
  SmallVector<StorageAliasLocalAllocOp> allocOpsToReplace;
  m.walk([&](StorageAliasLocalAllocOp allocOp) {
    allocOpsToReplace.push_back(allocOp);
  });

  for (auto allocOp : allocOpsToReplace) {
    Value storageAlias = allocOp.getStorageAlias();
    auto it = specToAlloc.find(storageAlias);
    if (it == specToAlloc.end()) {
      allocOp.emitError("storage_alias_spec not found for this allocation");
      return failure();
    }

    LDBG("Replacing storage_alias_local_alloc with LocalAliasOp");

    builder.setInsertionPoint(allocOp);

    // Get the original result type
    auto originalResultType =
        cast<ttg::MemDescType>(allocOp.getResult().getType());

    // Check if we have offset information for this allocation
    auto offsetIt = offsetMap.find(allocOp.getResult());

    // Determine the result type - may be expanded based on
    // bytes_between_buffers
    ttg::MemDescType resultType = originalResultType;
    if (offsetIt != offsetMap.end()) {
      int64_t bufferOffset = offsetIt->second.first;
      int64_t bytesBetweenBuffers = offsetIt->second.second;

      // Compute original buffer size (shape[0] is num_buffers, rest is
      // per-buffer)
      auto shape = originalResultType.getShape();
      int64_t elemBits = originalResultType.getElementTypeBitWidth();
      int64_t bufferElements = 1;
      for (size_t i = 1; i < shape.size(); ++i) {
        bufferElements *= shape[i];
      }
      int64_t originalBufferBytes = (bufferElements * elemBits) / 8;

      // Check if bytes_between_buffers divides evenly by original buffer size
      if (bytesBetweenBuffers % originalBufferBytes != 0) {
        allocOp.emitError("bytes_between_buffers (")
            << bytesBetweenBuffers
            << ") must be a multiple of the original buffer size ("
            << originalBufferBytes << ")";
        return failure();
      }

      // Check if buffer_offset divides evenly by original buffer size
      if (bufferOffset % originalBufferBytes != 0) {
        allocOp.emitError("buffer_offset (")
            << bufferOffset
            << ") must be a multiple of the original buffer size ("
            << originalBufferBytes << ")";
        return failure();
      }

      int64_t scaleFactor = bytesBetweenBuffers / originalBufferBytes;
      int64_t offsetSlots = bufferOffset / originalBufferBytes;

      // If there's padding or offset, expand the shape
      if (scaleFactor > 1 || offsetSlots > 0) {
        // Compute expanded shape:
        // new_buffer_dim = num_buffers * scale_factor + offset_slots
        int64_t numBuffers = shape[0];
        int64_t newBufferDim = numBuffers * scaleFactor + offsetSlots;

        SmallVector<int64_t> expandedShape;
        expandedShape.push_back(newBufferDim);
        for (size_t i = 1; i < shape.size(); ++i) {
          expandedShape.push_back(shape[i]);
        }

        LDBG("  Expanding shape from [" << shape[0] << ", ...] to ["
                                        << newBufferDim << ", ...]");
        LDBG("  (scale_factor=" << scaleFactor
                                << ", offset_slots=" << offsetSlots << ")");

        // Create new MemDescType with expanded shape
        resultType = ttg::MemDescType::get(
            expandedShape, originalResultType.getElementType(),
            originalResultType.getEncoding(),
            originalResultType.getMemorySpace(),
            originalResultType.getMutableMemory());
      }
    }

    // Create a LocalAliasOp to reinterpret the allocation with the
    // (possibly expanded) type
    auto localAliasOp =
        builder.create<LocalAliasOp>(allocOp.getLoc(), resultType, it->second);

    // Replace all uses and erase the old operation
    allocOp.getResult().replaceAllUsesWith(localAliasOp.getResult());
    allocOp.erase();

    // If the type changed (e.g., due to shape expansion), update block
    // argument types for any ops that capture this value (e.g.,
    // WarpSpecializeOp partition region args must match capture types).
    if (resultType != originalResultType) {
      updateBlockArgTypesForUsers(localAliasOp.getResult());
    }

    // If the shape was expanded, rewrite MemDescIndexOp indices to account
    // for the scale factor and offset
    if (offsetIt != offsetMap.end()) {
      int64_t bufferOffset = offsetIt->second.first;
      int64_t bytesBetweenBuffers = offsetIt->second.second;

      // Recompute scale factor and offset slots
      auto shape = originalResultType.getShape();
      int64_t elemBits = originalResultType.getElementTypeBitWidth();
      int64_t bufferElements = 1;
      for (size_t i = 1; i < shape.size(); ++i) {
        bufferElements *= shape[i];
      }
      int64_t originalBufferBytes = (bufferElements * elemBits) / 8;
      int64_t scaleFactor = bytesBetweenBuffers / originalBufferBytes;
      int64_t offsetSlots = bufferOffset / originalBufferBytes;

      // Only rewrite if there's actual scaling or offset
      if (scaleFactor > 1 || offsetSlots > 0) {
        LDBG("  Rewriting MemDescIndexOp indices (scale="
             << scaleFactor << ", offset=" << offsetSlots << ")");

        // Collect all MemDescIndexOp users (need to collect first to avoid
        // iterator invalidation)
        SmallVector<ttg::MemDescIndexOp> indexOpsToRewrite;
        collectMemDescIndexOps(localAliasOp.getResult(), indexOpsToRewrite);

        for (auto indexOp : indexOpsToRewrite) {
          builder.setInsertionPoint(indexOp);
          Location loc = indexOp.getLoc();
          Value originalIndex = indexOp.getIndex();

          // Compute: newIndex = scaleFactor * originalIndex + offsetSlots
          Value newIndex = originalIndex;

          if (scaleFactor > 1) {
            Value scaleVal = builder.create<arith::ConstantOp>(
                loc, builder.getI32IntegerAttr(scaleFactor));
            newIndex =
                builder.create<arith::MulIOp>(loc, originalIndex, scaleVal);
          }

          if (offsetSlots > 0) {
            Value offsetVal = builder.create<arith::ConstantOp>(
                loc, builder.getI32IntegerAttr(offsetSlots));
            newIndex = builder.create<arith::AddIOp>(loc, newIndex, offsetVal);
          }

          // Update the index operand
          indexOp.getIndexMutable().assign(newIndex);
          LDBG("    Rewrote index at " << loc);
        }
      }

      // Store offset information in the output map for reference
      localAliasOffsetMap[localAliasOp.getResult()] = offsetIt->second;
    }
  }

  // Third pass: erase storage_alias_spec operations
  for (auto specOp : specOps) {
    // Check if the spec still has uses (it shouldn't at this point)
    if (!specOp.getResult().use_empty()) {
      specOp.emitError(
          "storage_alias_spec still has uses after allocation materialization");
      return failure();
    }
    specOp.erase();
  }

  return success();
}

} // namespace tlx
} // namespace triton
} // namespace mlir
