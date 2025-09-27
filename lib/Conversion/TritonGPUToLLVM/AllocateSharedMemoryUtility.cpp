#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
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
  // Check if MLIR_ENABLE_DUMP env is set to 1
  static bool dumpEnabled = []() {
    if (triton::tools::getBoolEnv("MLIR_ENABLE_DUMP")) {
      return true;
    }
    return false;
  }();
  if (!dumpEnabled) {
    return;
  }

  mod.walk([&](Operation *op) {
    if (isa<triton::gpu::LocalStoreOp, triton::gpu::LocalLoadOp,
            triton::gpu::MemDescSubsliceOp, triton::gpu::MemDescIndexOp>(op)) {
      addAllocationAnnotations(op);
    }
  });
}

void attachAllocationSizeAndOffsetAttr(ModuleOp mod,
                                       ModuleAllocation &allocation) {
  MLIRContext *ctx = mod.getContext();

  mod.walk<mlir::WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
    auto *funcAllocation = allocation.getFuncData(funcOp);
    funcOp.walk([&](Operation *op) {
      auto oBufferId = funcAllocation->getBufferId(op);
      int offset = -1;
      if (oBufferId != Allocation::InvalidBufferId)
        offset = funcAllocation->getOffset(oBufferId);
      else if (op->getNumResults() == 1) {
        Value value = op->getResult(0);
        auto vBufferId = funcAllocation->getBufferId(value);
        if (vBufferId != Allocation::InvalidBufferId)
          offset = funcAllocation->getOffset(vBufferId);
      }
      if (offset == -1)
        return;
      op->setAttr("allocation.offset",
                  IntegerAttr::get(IntegerType::get(ctx, 32), offset));
    });
    return WalkResult::skip();
  });
  mod->setAttr("ttg.shared",
               mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                      allocation.getSharedMemorySize()));
}

} // namespace mlir::triton::gpu
