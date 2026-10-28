#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.h"
#include <cstdlib>
#include <string>

namespace mlir::triton::gpu {

// Helper function to compute allocation size from MemDescType
inline size_t computeAllocationSize(MemDescType memdescTy) {
  auto elemTy = memdescTy.getElementType();
  auto shape = memdescTy.getShape();
  size_t elemSize = elemTy.getIntOrFloatBitWidth() / 8;
  size_t totalElements = 1;
  for (auto dim : shape) {
    totalElements *= dim;
  }
  return totalElements * elemSize;
}

// Helper function to add allocation information as IR annotations
void addAllocationAnnotations(Operation *op) {
  MLIRContext *ctx = op->getContext();
  IntegerAttr offsetAttr;
  MemDescType memdescTy;

  // Try to get allocation.offset from the operation itself
  if (auto attr = op->getAttrOfType<IntegerAttr>("allocation.offset")) {
    offsetAttr = attr;
    // Find MemDescType from result or operands
    for (auto result : op->getResults()) {
      if (auto ty = dyn_cast<MemDescType>(result.getType())) {
        memdescTy = ty;
        break;
      }
    }
    if (!memdescTy) {
      for (auto operand : op->getOperands()) {
        if (auto ty = dyn_cast<MemDescType>(operand.getType())) {
          memdescTy = ty;
          break;
        }
      }
    }
  } else {
    // Try to find it through operands
    for (auto operand : op->getOperands()) {
      if (auto definingOp = operand.getDefiningOp()) {
        if (auto allocOp = dyn_cast<triton::gpu::LocalAllocOp>(definingOp)) {
          if (auto attr =
                  allocOp->getAttrOfType<IntegerAttr>("allocation.offset")) {
            offsetAttr = attr;
            memdescTy = cast<MemDescType>(allocOp.getType());
            break;
          }
        }
      }
    }
  }

  if (!offsetAttr || !memdescTy) {
    return;
  }

  auto offset = offsetAttr.getInt();
  size_t totalSize = computeAllocationSize(memdescTy);
  op->setAttr("shared_memory.offset",
              IntegerAttr::get(IntegerType::get(ctx, 64), offset));
  op->setAttr("shared_memory.size_bytes",
              IntegerAttr::get(IntegerType::get(ctx, 64), totalSize));
}

// Function to add shared memory access annotations to all operations that use
// shared memory
void addSharedMemoryAnnotations(ModuleOp mod) {
  if (!triton::tools::getBoolEnv("MLIR_ENABLE_DUMP")) {
    return;
  }

  mod.walk([&](Operation *op) {
    if (isa<triton::gpu::LocalStoreOp, triton::gpu::LocalLoadOp,
            triton::gpu::MemDescSubsliceOp,
            triton::gpu::MemDescDynamicSubsliceOp, triton::gpu::MemDescIndexOp>(
            op)) {
      addAllocationAnnotations(op);
    }
  });
}

void attachAllocationSizeAndOffsetAttr(ModuleOp mod,
                                       ModuleAllocation &allocation) {
  MLIRContext *ctx = mod.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);

  mod.walk<mlir::WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
    auto *funcAllocation = allocation.getFuncData(funcOp);
    funcOp.walk([&](Operation *op) {
      // Handle scratch buffers (from operations like convert_layout)
      auto oBufferId = funcAllocation->getBufferId(op);
      if (oBufferId != Allocation::InvalidBufferId) {
        int offset = funcAllocation->getOffset(oBufferId);
        op->setAttr("allocation.offset", IntegerAttr::get(i32Ty, offset));
        if (!isa<FunctionOpInterface>(op)) {
          int size = funcAllocation->getAllocatedSize(oBufferId);
          op->setAttr("allocation.size", IntegerAttr::get(i32Ty, size));
        }
        return;
      }

      // Handle explicit buffers (from values like local_alloc results)
      if (op->getNumResults() != 1)
        return;

      Value value = op->getResult(0);
      auto bufferIds = funcAllocation->getBufferIds(value);
      if (bufferIds.empty())
        return;

      // For partitioned tensors, set an array of offsets (one per partition)
      if (bufferIds.size() > 1) {
        SmallVector<Attribute> offsetAttrs;
        for (auto bufferId : bufferIds) {
          int partitionOffset = funcAllocation->getOffset(bufferId);
          offsetAttrs.push_back(IntegerAttr::get(i32Ty, partitionOffset));
        }
        op->setAttr("allocation.offset", ArrayAttr::get(ctx, offsetAttrs));
        return;
      }

      // Standard single offset for non-partitioned tensors
      int offset = funcAllocation->getOffset(bufferIds[0]);
      op->setAttr("allocation.offset", IntegerAttr::get(i32Ty, offset));
    });
    return WalkResult::skip();
  });
  mod->setAttr("ttg.shared",
               mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                      allocation.getSharedMemorySize()));
}

// The layout of an NPOT shared tile is built on its pow2 cover, so the
// allocation must hold at least as many elements as the cover's allocation.
static bool isNpotSharedCover(Type type) {
  auto memDesc = dyn_cast<MemDescType>(type);
  if (!memDesc || !isa<SharedEncodingTrait>(memDesc.getEncoding()))
    return false;
  Attribute encoding = memDesc.getEncoding();
  ArrayRef<int64_t> allocShape = memDesc.getAllocShape();
  auto tile = dropPipeliningDim(allocShape, encoding);
  SmallVector<int64_t> cover(allocShape);
  llvm::copy(getNpotCoverShape(tile), cover.end() - tile.size());
  if (llvm::equal(cover, allocShape))
    return false;
  return getAllocationElems(encoding, allocShape) <
         getAllocationElems(encoding, cover);
}

LogicalResult verifyNpotTypesSupported(ModuleOp mod) {
  if (!triton::tools::getBoolEnv("TRITON_ALLOW_NPOT"))
    return success();
  bool rejected = false;
  mod.walk([&](Operation *op) {
    if (llvm::any_of(op->getOperandTypes(), isNpotSharedCover) ||
        llvm::any_of(op->getResultTypes(), isNpotSharedCover)) {
      op->emitError() << "NPOT shared memory descriptors are not yet supported";
      rejected = true;
    }
  });
  return failure(rejected);
}

} // namespace mlir::triton::gpu
