#include "tlx/dialect/include/IR/Traits.h"

#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;

LogicalResult
OpTrait::impl::verifySameOperandAndResultMemorySpace(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  // Only mem descs can have memory spaces.
  auto operandType = dyn_cast<ttg::MemDescType>(op->getOperand(0).getType());

  auto resultType = dyn_cast<ttg::MemDescType>(op->getResult(0).getType());

  if (operandType && resultType) {
    if (operandType.getMemorySpace() != resultType.getMemorySpace()) {
      op->emitOpError()
          << "requires the same memory space for all operands and results";
      return failure();
    }
    return success();
  } else {
    op->emitOpError()
        << "requires the MemDescType for all operands and results";
    return failure();
  }
}
