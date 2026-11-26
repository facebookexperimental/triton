#include "triton/Dialect/TritonGPU/Transforms/CoalesceSectorUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace mlir::triton::gpu {

std::optional<int64_t> inferLaneStep(Value value, unsigned dim,
                                     ModuleAxisInfoAnalysis &axisInfo,
                                     unsigned &compileTimeBudget) {
  if (compileTimeBudget == 0)
    return std::nullopt;
  --compileTimeBudget;

  auto type = cast<RankedTensorType>(value.getType());
  const AxisInfo *info = axisInfo.getAxisInfo(value);
  if (type.getDimSize(dim) == 1 ||
      (info && dim < static_cast<unsigned>(info->getRank()) &&
       info->getConstancy(dim) == type.getDimSize(dim)))
    return 0;

  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  auto infer = [&](Value source, unsigned axis) {
    return inferLaneStep(source, axis, axisInfo, compileTimeBudget);
  };
  if (isa<triton::SplatOp>(op))
    return 0;
  if (isa<triton::MakeRangeOp>(op))
    return 1;
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op))
    return infer(expand.getSrc(), dim - (dim > expand.getAxis()));
  if (isa<triton::BroadcastOp, ConvertLayoutOp>(op))
    return infer(op->getOperand(0), dim);
  if (isa<arith::AddIOp, triton::AddPtrOp>(op)) {
    auto lhs = infer(op->getOperand(0), dim);
    auto rhs = infer(op->getOperand(1), dim);
    if (!lhs || !rhs)
      return std::nullopt;
    if (isa<triton::AddPtrOp>(op)) {
      // The pointer and the element offset can have different wrap points.
      if (*lhs == 0)
        return rhs;
      return *rhs == 0 ? lhs : std::nullopt;
    }
    return llvm::checkedAdd(*lhs, *rhs);
  }
  if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    for (auto [index, operand] : llvm::enumerate(mul->getOperands())) {
      const AxisInfo *factor = axisInfo.getAxisInfo(operand);
      auto constant = factor ? factor->getConstantValue() : std::nullopt;
      if (!constant)
        continue;
      if (auto step = infer(mul->getOperand(1 - index), dim))
        return llvm::checkedMul(*step, *constant);
    }
  }
  return std::nullopt;
}

unsigned countScalarWaveSectors(BlockedEncodingAttr layout,
                                ArrayRef<uint64_t> strides,
                                unsigned elementBytes, unsigned sectorBytes) {
  auto lanes = layout.getThreadsPerWarp();
  llvm::SmallDenseSet<uint64_t, 64> sectors;
  for (unsigned row = 0; row < lanes[0]; ++row) {
    for (unsigned col = 0; col < lanes[1]; ++col) {
      uint64_t offset = (row * strides[0] + col * strides[1]) * elementBytes;
      uint64_t end = offset + elementBytes - 1;
      for (uint64_t sector = offset / sectorBytes; sector <= end / sectorBytes;
           ++sector)
        sectors.insert(sector);
    }
  }
  return sectors.size();
}

} // namespace mlir::triton::gpu
