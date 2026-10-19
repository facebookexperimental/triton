#include "triton/Analysis/IntegerRange.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/ConstantRange.h"

using namespace mlir;

namespace {
// The generated default method casts to the concrete op, which only converts
// implicitly to an interface for native (TableGen) implementations.
template <typename Model, typename Op>
struct RangeModel : InferIntRangeInterface::ExternalModel<Model, Op> {
  void inferResultRangesFromOptional(Operation *op,
                                     ArrayRef<IntegerValueRange> args,
                                     SetIntLatticeFn setRange) const {
    intrange::detail::defaultInferResultRanges(cast<InferIntRangeInterface>(op),
                                               args, setRange);
  }
};

intrange::CmpPredicate convertPredicate(LLVM::ICmpPredicate predicate) {
  switch (predicate) {
#define PREDICATE(name)                                                        \
  case LLVM::ICmpPredicate::name:                                              \
    return intrange::CmpPredicate::name;
    PREDICATE(eq)
    PREDICATE(ne)
    PREDICATE(slt)
    PREDICATE(sle)
    PREDICATE(sgt)
    PREDICATE(sge)
    PREDICATE(ult)
    PREDICATE(ule)
    PREDICATE(ugt)
    PREDICATE(uge)
#undef PREDICATE
  }
  llvm_unreachable("invalid integer predicate");
}

ConstantIntRanges inferLLVM(Operation *op, ArrayRef<ConstantIntRanges> args) {
  unsigned width = cast<IntegerType>(op->getResult(0).getType()).getWidth();
  return llvm::TypeSwitch<Operation *, ConstantIntRanges>(op)
      .Case<LLVM::ConstantOp>([&](auto constant) {
        if (auto attr = dyn_cast<IntegerAttr>(constant.getValue()))
          return ConstantIntRanges::constant(
              attr.getValue().sextOrTrunc(width));
        return ConstantIntRanges::maxRange(width);
      })
      // Ignoring nsw/nuw/exact is conservative: these helpers include all
      // wrapping results, and the flagged operations can only remove values.
      .Case<LLVM::AddOp>([&](auto) { return intrange::inferAdd(args); })
      .Case<LLVM::SubOp>([&](auto) { return intrange::inferSub(args); })
      .Case<LLVM::MulOp>([&](auto) { return intrange::inferMul(args); })
      .Case<LLVM::UDivOp>([&](auto) { return intrange::inferDivU(args); })
      .Case<LLVM::SDivOp>([&](auto) { return intrange::inferDivS(args); })
      .Case<LLVM::URemOp>([&](auto) { return intrange::inferRemU(args); })
      .Case<LLVM::SRemOp>([&](auto) { return intrange::inferRemS(args); })
      .Case<LLVM::AndOp>([&](auto) { return intrange::inferAnd(args); })
      .Case<LLVM::OrOp>([&](auto) { return intrange::inferOr(args); })
      .Case<LLVM::XOrOp>([&](auto) { return intrange::inferXor(args); })
      .Case<LLVM::ShlOp>([&](auto) { return intrange::inferShl(args); })
      .Case<LLVM::LShrOp>([&](auto) { return intrange::inferShrU(args); })
      .Case<LLVM::AShrOp>([&](auto) { return intrange::inferShrS(args); })
      .Case<LLVM::ZExtOp>(
          [&](auto) { return intrange::extUIRange(args[0], width); })
      .Case<LLVM::SExtOp>(
          [&](auto) { return intrange::extSIRange(args[0], width); })
      .Case<LLVM::TruncOp>(
          [&](auto) { return intrange::truncRange(args[0], width); })
      .Case<LLVM::ICmpOp>([&](auto cmp) {
        auto result = intrange::evaluatePred(
            convertPredicate(cmp.getPredicate()), args[0], args[1]);
        return result ? ConstantIntRanges::constant(APInt(1, *result))
                      : ConstantIntRanges::maxRange(1);
      })
      .Case<LLVM::SelectOp>([&](auto) {
        if (auto condition = args[0].getConstantValue())
          return args[condition->isZero() ? 2 : 1];
        return args[1].rangeUnion(args[2]);
      })
      .Default([&](Operation *) { return ConstantIntRanges::maxRange(width); });
}

template <typename Op>
struct LLVMRangeModel : RangeModel<LLVMRangeModel<Op>, Op> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> args,
                         SetIntRangeFn setRange) const {
    // The specialization pass handles scalar predicates. Leave vectors and
    // non-integer constants to the standard conservative fallback.
    Value result = op->getResult(0);
    if (isa<IntegerType>(result.getType()) &&
        llvm::all_of(op->getOperandTypes(),
                     [](Type type) { return isa<IntegerType>(type); }))
      setRange(result, inferLLVM(op, args));
    else
      setRange(result, IntegerValueRange::getMaxRange(result).getValue());
  }
};

template <typename Op> struct IDRangeModel : RangeModel<IDRangeModel<Op>, Op> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges>,
                         SetIntRangeFn setRange) const {
    auto id = cast<Op>(op);
    Value result = op->getResult(0);
    auto attr = id.getRangeAttr();
    if (!attr) {
      setRange(result, IntegerValueRange::getMaxRange(result).getValue());
      return;
    }
    // LLVM ranges are half-open and can wrap; MLIR tracks inclusive signed
    // and unsigned bounds independently. ConstantRange handles both views.
    llvm::ConstantRange range(attr.getLower(), attr.getUpper());
    setRange(result,
             ConstantIntRanges(range.getUnsignedMin(), range.getUnsignedMax(),
                               range.getSignedMin(), range.getSignedMax()));
  }
};

struct ReadFirstLaneRangeModel
    : RangeModel<ReadFirstLaneRangeModel, ROCDL::ReadfirstlaneOp> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> args,
                         SetIntRangeFn setRange) const {
    setRange(op->getResult(0), args[0]);
  }
};

template <typename... Ops> void attachLLVMModels(MLIRContext *context) {
  (Ops::template attachInterface<LLVMRangeModel<Ops>>(*context), ...);
}
template <typename... Ops> void attachIDModels(MLIRContext *context) {
  (Ops::template attachInterface<IDRangeModel<Ops>>(*context), ...);
}
} // namespace

void mlir::triton::registerIntegerRangeExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    attachLLVMModels<LLVM::ConstantOp, LLVM::AddOp, LLVM::SubOp, LLVM::MulOp,
                     LLVM::UDivOp, LLVM::SDivOp, LLVM::URemOp, LLVM::SRemOp,
                     LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp, LLVM::ShlOp,
                     LLVM::LShrOp, LLVM::AShrOp, LLVM::ZExtOp, LLVM::SExtOp,
                     LLVM::TruncOp, LLVM::ICmpOp, LLVM::SelectOp>(context);
  });
  registry.addExtension(+[](MLIRContext *context, ROCDL::ROCDLDialect *) {
    attachIDModels<ROCDL::ThreadIdXOp, ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp,
                   ROCDL::BlockIdXOp, ROCDL::BlockIdYOp, ROCDL::BlockIdZOp,
                   ROCDL::WaveId>(context);
    ROCDL::ReadfirstlaneOp::attachInterface<ReadFirstLaneRangeModel>(*context);
  });
}
