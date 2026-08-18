#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUToLLVM/TypeConverter.h"
#include "Utility.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUCONVERTWARPSPECIALIZETOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

enum BarrierIndex {
  kNullBarrierIdx,
  kDefaultWarpGroupBarrierIdx,
  kNumReservedBarriers,
  kNumBarriers = 17
};

class AMDWarpSpecializeBarrierHelper : public WarpSpecializeBarrierHelper {
public:
  AMDWarpSpecializeBarrierHelper(ModuleOp module,
                                 const AMD::TargetInfo &targetInfo)
      : module(module), targetInfo(targetInfo) {}

  bool isBarrierOp(Operation *op) const override {
    return isa<ROCDL::BarrierOp, ROCDL::SBarrierOp>(op);
  }

  Type getBarrierHandleType(MLIRContext *ctx) const override {
    return LLVM::LLVMPointerType::get(ctx, targetInfo.getSharedAddressSpace());
  }

  FailureOr<Value>
  getBarrierHandle(TritonLLVMIRRewriter &b,
                   std::optional<unsigned> partitionIdx) override {
    unsigned barIdx;
    if (!partitionIdx) {
      barIdx = kDefaultWarpGroupBarrierIdx;
    } else {
      barIdx = *partitionIdx + kNumReservedBarriers;
      if (barIdx >= kNumBarriers) {
        return mlir::emitError(b.getLoc(), "cannot support more than ")
               << (kNumBarriers - kNumReservedBarriers)
               << " warp group partitions";
      }
    }

    auto nbarAttr = b.getStringAttr("nbar" + Twine(barIdx));
    auto nbarTy = LLVM::LLVMTargetExtType::get(b.getContext(),
                                               "amdgcn.named.barrier", {}, {0});

    LLVM::GlobalOp nbarGV;
    Operation *nbarGlobalOp = SymbolTable::lookupSymbolIn(module, nbarAttr);
    if (!nbarGlobalOp) {
      RewriterBase::InsertionGuard guard(b);
      Location uloc = b.getUnknownLoc();
      b.setInsertionPointToStart(module.getBody());
      nbarGV = LLVM::GlobalOp::create(
          b, uloc, nbarTy, /*isConstant=*/false, LLVM::Linkage::Internal,
          nbarAttr.getValue(), /*value=*/Attribute(), /*alignment=*/0,
          targetInfo.getSharedAddressSpace());
      // Add initializer region that returns 'poison'
      Block *initBlock = b.createBlock(&nbarGV.getInitializerRegion());
      b.setInsertionPointToStart(initBlock);
      Value poison = LLVM::PoisonOp::create(b, uloc, nbarTy);
      LLVM::ReturnOp::create(b, uloc, poison);
    } else {
      nbarGV = cast<LLVM::GlobalOp>(*nbarGlobalOp);
    }

    return Value(LLVM::AddressOfOp::create(b, b.getLoc(), nbarGV));
  }

  void createBarrier(TritonLLVMIRRewriter &b, unsigned numWarps,
                     Value handle) override {
    Location loc = b.getLoc();
    auto nbarTy = LLVM::LLVMTargetExtType::get(b.getContext(),
                                               "amdgcn.named.barrier", {}, {0});
    auto smemObj = SharedMemoryObject(handle, nbarTy, 1, loc, b);
    ROCDL::BarrierJoinOp::create(b, loc, smemObj.getBase());
    ROCDL::BarrierSignalVarOp::create(b, loc, smemObj.getBase(), numWarps);
    ROCDL::BarrierWaitOp::create(b, loc, 1);
  }

private:
  ModuleOp module;
  const AMD::TargetInfo &targetInfo;
};

//===----------------------------------------------------------------------===//
// lowerWarpSpecialize
//===----------------------------------------------------------------------===//

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func,
                                         const AMD::TargetInfo &targetInfo) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

  auto module = cast<ModuleOp>(func->getParentOp());
  unsigned defaultNumWarps = lookupNumWarps(func);

  auto totalNumWarpsAttr =
      module->getAttrOfType<IntegerAttr>("ttg.total-num-warps");
  if (!totalNumWarpsAttr) {
    return mlir::emitError(module.getLoc(),
                           "module missing 'ttg.total-num-warps' attribute");
  }

  // Attempt to elide captures of trivial computations by hoisting them into the
  // header or rematerializing them into each partition.
  elideTrivialCaptures(func, wsOps);

  MLIRContext *ctx = func.getContext();
  TritonLLVMIRRewriter b(func.getLoc(), ctx);
  Builder rewriter(ctx);

  // Generate the function header.
  Block *entry = &func.getBody().front();
  SmallVector<Location> argLocs = llvm::to_vector(llvm::map_range(
      func.getArguments(), [](BlockArgument arg) { return arg.getLoc(); }));
  Block *header = b.createBlock(entry, func.getArgumentTypes(), argLocs);
  Block *switchLoop = b.createBlock(entry);
  b.setInsertionPointToStart(header);

  // This is the absolute warp ID.
  Value wid = ROCDL::WaveId::create(b, b.getLoc(), i32_ty);
  Value isDefault = b.icmp_ult(wid, b.i32_val(defaultNumWarps));
  LLVM::CondBrOp::create(b, b.getLoc(), isDefault, entry, switchLoop);

  // Forward arguments from the header into the old entry block.
  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });

  WarpSpecializeCallbacks callbacks;
  callbacks.createAllBarrier = [](TritonLLVMIRRewriter &b, unsigned) {
    Location loc = b.getLoc();
    ROCDL::BarrierOp::create(b, loc);
  };

  callbacks.reallocRegisters = [](TritonLLVMIRRewriter &, WarpSpecializeOp,
                                  RegisterReallocPhase, unsigned) {};

  return lowerWarpSpecializeCommon(
      func, wsOps, entry, header, switchLoop, wid, ctx, defaultNumWarps,
      totalNumWarpsAttr.getInt(), targetInfo, callbacks, 0);
}

// Lower a `ttg.warp_predicate` after its body has been converted to LLVM. The
// per-thread condition becomes divergent control flow; AMD lowers it to an
// EXEC-mask restriction and skips the body for waves with no active lane.
static LogicalResult lowerOneWarpPredicate(WarpPredicateOp op) {
  if (op.getRegion().empty() || op.getRegion().front().getNumArguments() != 0)
    return op.emitError("expected a nonempty capture-only body region");

  // Conversion of the body to LLVM may introduce CFG blocks (for example,
  // wave-local reductions and predicated dot operations).  The entry block
  // remains capture-only, while the predicate yield can be in any exit block.
  // Locate that unique terminator before moving the whole region into the
  // surrounding function CFG.
  PredicateYieldOp yield;
  for (Block &block : op.getRegion()) {
    auto candidate = dyn_cast<PredicateYieldOp>(block.getTerminator());
    if (!candidate)
      continue;
    if (yield)
      return op.emitError("expected a unique ttg.predicate_yield terminator");
    yield = candidate;
  }
  if (!yield)
    return op.emitError("expected ttg.predicate_yield terminator");
  if (op.getInits().size() != op.getNumResults() ||
      yield.getNumOperands() != op.getNumResults())
    return op.emitError("expected equal numbers of inits, results, and yields");

  auto predicateType = dyn_cast<RankedTensorType>(op.getPredicate().getType());
  if (predicateType) {
    auto predicateEncoding =
        dyn_cast_or_null<DistributedEncodingTrait>(predicateType.getEncoding());
    if (!predicateEncoding)
      return op.emitError("expected a distributed tensor predicate");
    for (Value init : op.getInits()) {
      auto initType = dyn_cast<RankedTensorType>(init.getType());
      if (!initType)
        continue;
      auto initEncoding =
          dyn_cast_or_null<DistributedEncodingTrait>(initType.getEncoding());
      if (!initEncoding || initType.getRank() < predicateType.getRank() ||
          !llvm::equal(predicateType.getShape(),
                       initType.getShape().take_front(predicateType.getRank())))
        return op.emitError(
            "predicate shape must be a leading shape of every carried tensor");

      Attribute projectedEncoding = initEncoding;
      for (int rank = initType.getRank(); rank > predicateType.getRank();
           --rank)
        projectedEncoding = SliceEncodingAttr::get(
            op.getContext(), rank - 1,
            cast<DistributedEncodingTrait>(projectedEncoding));
      auto projectedType = RankedTensorType::get(predicateType.getShape(),
                                                 predicateType.getElementType(),
                                                 projectedEncoding);
      if (!isLayoutEquivalentIgnoringRegisterOrder(
              toLinearLayout(projectedType), toLinearLayout(predicateType)))
        return op.emitError(
            "predicate and carried tensors must have matching lane ownership");
    }
  } else if (!op.getPredicate().getType().isInteger(1)) {
    return op.emitError("expected an i1 or distributed tensor predicate");
  }

  // Reductions and layout conversions can introduce synchronization after the
  // TTGIR verifier has run. Reject any resulting CTA-wide barrier before
  // restricting EXEC, otherwise inactive waves could deadlock active ones.
  WalkResult barrier = op.getRegion().walk([&](Operation *nested) {
    if (isa<ROCDL::BarrierOp, ROCDL::SBarrierOp>(nested))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (barrier.wasInterrupted())
    return op.emitError("lowered body may not contain CTA barriers");

  Location loc = op.getLoc();
  IRRewriter rewriter(op->getContext());
  rewriter.setInsertionPoint(op);

  auto asLLVM = [](Value value) -> Value {
    if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>())
      if (cast.getNumOperands() == 1 && cast.getNumResults() == 1)
        return cast.getOperand(0);
    return value;
  };

  Value lanePredicate;
  Value predicate = asLLVM(op.getPredicate());
  if (predicateType) {
    SmallVector<Value> predicateElements =
        unpackLLElements(loc, predicate, rewriter);
    if (predicateElements.empty())
      return op.emitError("empty predicate");
    for (Value element : predicateElements) {
      lanePredicate = lanePredicate ? LLVM::OrOp::create(rewriter, loc,
                                                         lanePredicate, element)
                                          .getResult()
                                    : element;
    }
  } else {
    if (!predicate.getType().isInteger(1))
      return op.emitError("converted scalar predicate must have i1 type");
    lanePredicate = predicate;
  }

  SmallVector<Value> initStructs;
  SmallVector<Value> yieldStructs;
  SmallVector<Type> resultTypes;
  for (Value init : op.getInits())
    initStructs.push_back(asLLVM(init));
  for (Value yielded : yield.getValues()) {
    Value value = asLLVM(yielded);
    yieldStructs.push_back(value);
    resultTypes.push_back(value.getType());
  }
  for (auto [init, yielded] : llvm::zip(initStructs, yieldStructs))
    if (init.getType() != yielded.getType())
      return op.emitError(
          "converted init and yield values must have identical types");

  // Validate every result bridge before changing the CFG. Conversion failure
  // should leave a diagnosable source operation rather than partially inlined
  // blocks or an invalid unrealized cast.
  for (auto [index, result] : llvm::enumerate(op.getResults())) {
    for (OpOperand &use : result.getUses()) {
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(use.getOwner())) {
        if (cast.getNumOperands() != 1 || cast.getNumResults() != 1 ||
            cast.getResult(0).getType() != resultTypes[index])
          return op.emitError("malformed LLVM result bridge");
        continue;
      }
      // Non-tensor values are already legal LLVM dialect operands, so dialect
      // conversion leaves their users connected directly to the source result
      // instead of inserting an unrealized conversion cast.
      if (!isa<RankedTensorType>(result.getType()) &&
          result.getType() == resultTypes[index])
        continue;
      if (!isa<WarpPredicateOp, PredicateYieldOp>(use.getOwner()))
        return op.emitError("unexpected use of result");
    }
  }

  Block *currentBlock = rewriter.getInsertionBlock();
  Block *mergeBlock = rewriter.splitBlock(currentBlock, Block::iterator(op));
  SmallVector<Value> mergeArguments;
  for (Type type : resultTypes)
    mergeArguments.push_back(mergeBlock->addArgument(type, loc));

  Block *bodyBlock = &op.getRegion().front();
  rewriter.inlineRegionBefore(op.getRegion(), mergeBlock);

  rewriter.setInsertionPoint(yield);
  cf::BranchOp::create(rewriter, loc, mergeBlock, yieldStructs);
  rewriter.eraseOp(yield);

  for (auto [result, argument] : llvm::zip(op.getResults(), mergeArguments)) {
    for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
      auto cast = dyn_cast<UnrealizedConversionCastOp>(use.getOwner());
      if (cast) {
        rewriter.replaceAllUsesWith(cast.getResult(0), argument);
        rewriter.eraseOp(cast);
        continue;
      }

      if (!isa<RankedTensorType>(result.getType()) &&
          result.getType() == argument.getType()) {
        use.set(argument);
        continue;
      }

      // Layout cleanup can feed one warp_predicate result directly into a
      // sibling warp_predicate, while a nested predicate can feed the
      // enclosing predicate_yield. Bridge the already lowered LLVM merge
      // value back to that temporary tensor type; lowering the consumer
      // immediately unwraps this cast through asLLVM().
      if (isa<WarpPredicateOp, PredicateYieldOp>(use.getOwner())) {
        rewriter.setInsertionPoint(use.getOwner());
        Value bridge = UnrealizedConversionCastOp::create(
                           rewriter, loc, result.getType(), argument)
                           .getResult(0);
        use.set(bridge);
        continue;
      }

      return op.emitError("unexpected use of result");
    }
  }
  rewriter.eraseOp(op);

  rewriter.setInsertionPointToEnd(currentBlock);
  cf::CondBranchOp::create(rewriter, loc, lanePredicate, bodyBlock,
                           ValueRange{}, mergeBlock, ValueRange(initStructs));
  return success();
}

static LogicalResult lowerWarpPredicateOps(ModuleOp module) {
  SmallVector<WarpPredicateOp> ops;
  module.walk([&](WarpPredicateOp op) { ops.push_back(op); });
  for (WarpPredicateOp op : ops)
    if (failed(lowerOneWarpPredicate(op)))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct TritonAMDGPUConvertWarpSpecializeToLLVM
    : public mlir::triton::impl::TritonAMDGPUConvertWarpSpecializeToLLVMBase<
          TritonAMDGPUConvertWarpSpecializeToLLVM> {

  TritonAMDGPUConvertWarpSpecializeToLLVM(StringRef gfxArch)
      : TritonAMDGPUConvertWarpSpecializeToLLVMBase<
            TritonAMDGPUConvertWarpSpecializeToLLVM>() {
    this->gfxArch = gfxArch;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect, LLVM::LLVMDialect,
                    ROCDL::ROCDLDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    if (failed(lowerWarpPredicateOps(mod)))
      return signalPassFailure();

    SmallVector<Operation *> wsOps;
    mod.walk([&](Operation *op) {
      if (isa<WarpSpecializeOp, WarpSpecializePartitionsOp, WarpYieldOp>(op))
        wsOps.push_back(op);
    });

    // If no warp specialization ops, this pass is a no-op
    if (wsOps.empty())
      return;

    // Use the arch parameter if provided, otherwise get from module
    std::string archStr = this->gfxArch;
    if (archStr.empty()) {
      auto arch = getAMDArch(mod);
      if (!arch.has_value()) {
        mod.emitError(
            "Warp specialization requires AMD architecture to be specified");
        return signalPassFailure();
      }
      archStr = arch->str();
    }

    AMD::TargetInfo targetInfo(archStr.c_str());
    if (targetInfo.getISAFamily() != triton::amdgpu::ISAFamily::GFX1250) {
      mod.emitError("Warp specialization is only supported on gfx1250, got ")
          << archStr;
      return signalPassFailure();
    }

    // Convert types and cleanup unrealized conversions.
    mlir::LowerToLLVMOptions option(&getContext());
    option.overrideIndexBitwidth(32);
    TritonAMDGPUToLLVMTypeConverter typeConverter(&getContext(), option,
                                                  targetInfo);
    for (Operation *op : wsOps) {
      convertOpTypes(op, typeConverter);
    }
    OpPassManager pm;
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, mod)))
      return signalPassFailure();

    AMDWarpSpecializeBarrierHelper barrierHelper(mod, targetInfo);
    if (failed(lowerWarpSpecializeBarriers(mod, barrierHelper)))
      return signalPassFailure();

    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.getLinkage() == LLVM::Linkage::External)
        kernels.push_back(func);
    }
    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel, targetInfo)))
        return signalPassFailure();
  }
};
} // namespace

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUConvertWarpSpecializeToLLVMPass(StringRef gfxArch) {
  return std::make_unique<TritonAMDGPUConvertWarpSpecializeToLLVM>(gfxArch);
}

} // namespace mlir::triton::AMD
