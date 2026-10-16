#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
        if (!funcAllocation->isVirtualBuffer(oBufferId)) {
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

LogicalResult verifyNpotModularOpsSupported(ModuleOp mod) {
  if (!triton::tools::getBoolEnv("TRITON_ALLOW_NPOT"))
    return success();
  auto isNpot = [](Type type) {
    auto tensorTy = dyn_cast<RankedTensorType>(type);
    return tensorTy && hasNpotShape(tensorTy);
  };
  bool rejected = false;
  mod.walk([&](Operation *op) {
    if (!isa<triton::ReduceOp, triton::ScanOp, triton::HistogramOp,
             triton::GatherOp>(op))
      return;
    // A histogram gates on its input only: NPOT bins over a power-of-two
    // input are exact.
    if (llvm::none_of(op->getOperandTypes(), isNpot) &&
        (isa<triton::HistogramOp>(op) ||
         llvm::none_of(op->getResultTypes(), isNpot)))
      return;
    op->emitError() << "NPOT " << op->getName()
                    << " operations are not yet supported with modular "
                       "tensor layouts";
    rejected = true;
  });
  return failure(rejected);
}

} // namespace mlir::triton::gpu
