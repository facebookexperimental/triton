#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include <numeric>

namespace mlir::triton::AMD {

namespace {
class ExtractSliceOpAxisInfoVisitor : public AxisInfoVisitor {
public:
  using AxisInfoVisitor::AxisInfoVisitor;

  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) final {
    auto extractSlice = cast<amdgpu::ExtractSliceOp>(op);
    auto offsetsAttr = extractSlice.getStaticOffsetsAttr();
    auto outputShape = extractSlice.getResult().getType().getShape();

    assert(extractSlice->getNumOperands() == operands.size());
    auto srcInfo = operands[0]->getValue();

    auto contiguity = srcInfo.getContiguity();
    auto divisibility = srcInfo.getDivisibility();
    auto constancy = srcInfo.getConstancy();
    auto constant = srcInfo.getConstantValue();
    SmallVector<int64_t> newContiguity;
    SmallVector<int64_t> newDivisibility;
    SmallVector<int64_t> newConstancy;
    const int rank = contiguity.size();
    for (int dim = 0; dim < rank; ++dim) {
      int64_t offset = offsetsAttr[dim];

      // newContiguity: When extracting at an offset, contiguous groups may be
      // split. If the offset aligns with contiguity boundaries (offset %
      // contiguity == 0), contiguity is preserved. Otherwise, the new
      // contiguity is the gcd of the offset and source contiguity.
      // Also capped by the output shape.
      int64_t newContig = std::gcd(offset, contiguity[dim]);
      newContig = std::min(newContig, outputShape[dim]);
      newContiguity.push_back(newContig);

      // newDivisibility: The first element of each contiguous group shifts by
      // the offset. If original first element was divisible by D, the new first
      // element (at original_first + offset) is divisible by gcd(D, offset).
      int64_t offsetInContigTile = offset % contiguity[dim];
      int64_t newDiv = std::gcd(divisibility[dim], offsetInContigTile);
      newDivisibility.push_back(newDiv);

      // newConstancy is computed with same rules as contiguity.
      int64_t newConst = std::gcd(offset, constancy[dim]);
      newConst = std::min(newConst, outputShape[dim]);
      newConstancy.push_back(newConst);
    }
    return AxisInfo(newContiguity, newDivisibility, newConstancy, constant);
  }

  virtual bool match(Operation *op) final {
    return isa<amdgpu::ExtractSliceOp>(op);
  }
};

class RematerializedRangeOpAxisInfoVisitor : public AxisInfoVisitor {
public:
  using AxisInfoVisitor::AxisInfoVisitor;

  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) final {
    auto range = cast<amdgpu::RematerializedRangeOp>(op);
    int64_t start = range.getStart();
    int64_t end = range.getEnd();
    return AxisInfo(/*contiguity=*/{end - start},
                    /*divisibility=*/{highestPowOf2Divisor(start)},
                    /*constancy=*/{1});
  }

  bool match(Operation *op) final {
    return isa<amdgpu::RematerializedRangeOp>(op);
  }
};
class AMDAxisInfoDialectInterface final : public AxisInfoDialectInterface {
public:
  using AxisInfoDialectInterface::AxisInfoDialectInterface;

  AxisInfoAnalysis::LoadCallback
  getAnalysisLoader(ModuleOp mod) const override {
    auto target = mod->getAttrOfType<StringAttr>("ttg.target");
    if (target && target.getValue().starts_with("hip:"))
      return AxisInfoAnalysisExt::loadAnalysis;
    return nullptr;
  }
};
} // namespace

void registerAxisInfoInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *, amdgpu::TritonAMDGPUDialect *dialect) {
        dialect->addInterfaces<AMDAxisInfoDialectInterface>();
      });
}

AxisInfoAnalysisExt::AxisInfoAnalysisExt(DataFlowSolver &solver)
    : triton::AxisInfoAnalysis(solver) {
  visitors.append<ExtractSliceOpAxisInfoVisitor,
                  RematerializedRangeOpAxisInfoVisitor>();
}

LogicalResult AxisInfoAnalysisExt::initialize(Operation *top) {
  SmallVector<Operation *> queries;
  top->walk([&](Operation *op) {
    if (isa<arith::DivSIOp, arith::RemSIOp>(op) &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      queries.push_back(op);
  });
  if (!queries.empty()) {
    DominanceInfo dominance(top);
    auto rangeSolver = createDataFlowSolver();
    auto *ranges = rangeSolver->load<TritonIntegerRangeAnalysis>(
        TritonIntegerRangeAnalysis::collectAssumptions(top), &dominance);
    if (failed(rangeSolver->initializeAndRun(top)))
      return top->emitError("AMD integer range analysis failed");
    for (Operation *op : queries) {
      Value dividend = op->getOperand(0);
      const auto *lattice =
          rangeSolver->lookupState<dataflow::IntegerValueRangeLattice>(
              dividend);
      if (!lattice)
        continue;
      IntegerValueRange range =
          ranges->getRangeAt(dividend, op, lattice->getValue());
      if (!range.isUninitialized() && range.getValue().smin().isNonNegative())
        nonNegativeDividends.insert(op);
    }
  }
  return triton::AxisInfoAnalysis::initialize(top);
}

triton::AxisInfoAnalysis *
AxisInfoAnalysisExt::loadAnalysis(DataFlowSolver *solver) {
  return solver->load<AxisInfoAnalysisExt>();
}

} // namespace mlir::triton::AMD
