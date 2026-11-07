#include "Dialect/NVGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/AtomicPTXBuilder.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/NvmmaSmemAttrs.h"
#include "triton/Tools/LayoutUtils.h"
namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::NVIDIA;
using namespace mlir::LLVM::NVIDIA;

bool isConstI32OneTensor(Value value) {
  DenseElementsAttr constant;
  return matchPattern(value, m_Constant(&constant)) &&
         constant.getElementType().isInteger(32) &&
         llvm::all_of(constant.getValues<APInt>(),
                      [](const APInt &value) { return value.isOne(); });
}

Value emitSharedInc(ConversionPatternRewriter &rewriter, Location loc,
                    Value ptr, bool returnOld, bool isCluster,
                    Value pred = Value()) {
  PTXBuilder ptxBuilder;
  // PTX atom/red.inc resets to 0 only when the old value reaches the bound, so
  // using UINT32_MAX makes it equivalent to a wrapping increment-by-1.
  auto *boundOpr = ptxBuilder.newConstantOperand("0xffffffff");
  auto &inc = *ptxBuilder.create(returnOld ? "atom" : "red");
  if (isCluster)
    inc.o("shared::cluster").o("cluster");
  else
    inc.shared().o("cta");
  inc.o("relaxed").o("inc").o("u32");

  if (!returnOld) {
    auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "r");
    inc(ptrOpr, boundOpr).maybePredicate(pred, "b");
    return ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
  }

  auto *dstOpr = ptxBuilder.newOperand("=r", /*init=*/true);
  auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "r");
  inc(dstOpr, ptrOpr, boundOpr).maybePredicate(pred, "b");
  return ptxBuilder.launch(rewriter, loc, i32_ty);
}

FailureOr<Value> emitSharedAtomicRMW(ConversionPatternRewriter &rewriter,
                                     Location loc, Type valueElemTy, Value ptr,
                                     Value value, RMWOp rmwOp, bool returnOld,
                                     bool isCluster, Value pred) {
  SmallVector<Value> vals{value};
  if (!returnOld) {
    auto result =
        emitPtxSharedAtomicRMW(rewriter, loc, valueElemTy, ptr, vals, rmwOp,
                               pred, isCluster, PtxAtomicInstr::Red);
    if (succeeded(result))
      return result;
  }

  return emitPtxSharedAtomicRMW(rewriter, loc, valueElemTy, ptr, vals, rmwOp,
                                pred, isCluster, PtxAtomicInstr::Atom);
}

LogicalResult lowerLdStMatrix(
    Location loc, const LinearLayout &regLayout, MemDescType memDescType,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    SharedMemoryObject smemObj, ConversionPatternRewriter &rewriter,
    const NVIDIA::TargetInfo &targetInfo,
    const LLVMTypeConverter *typeConverter) {
  auto *ctx = loc.getContext();
  assert(regLayout.getFreeVariableMasks().lookup(str_attr("register")) == 0 &&
         "expected register broadcasting to be removed by the caller");
  if (isa<PaddedSharedEncodingAttr>(memDescType.getEncoding()) ||
      !regLayout.isRadix2()) {
    return failure();
  }
  if (SharedMemoryObject::getMaskSpanOffsetsAndBlocks(memDescType).second !=
      0) {
    return failure();
  }
  auto memLayout = toLinearLayout(memDescType);
  auto cvt = regLayout.invertAndCompose(memLayout);
  auto kBlock = StringAttr::get(loc.getContext(), "block");
  // ldmatrix/stmatrix does not support shared::cluster
  auto maybeSublayout = cvt.quotient({kBlock});
  if (!maybeSublayout) {
    return failure();
  }
  cvt = maybeSublayout.value();
  auto smemBase = smemObj.getBase();
  auto affineOffset = smemObj.getShmemOffset(loc, rewriter, memDescType);
  auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(memDescType);
  auto llvmElemTy = typeConverter->convertType(memDescType.getElementType());
  for (bool transpose : {false, true}) {
    auto result = LLVM::NVIDIA::lowerLdStMatrix(
        loc, cvt, transpose, vals, smemBase, affineOffset, maskSpanAffineOffset,
        llvmElemTy, rewriter, targetInfo);
    if (succeeded(result)) {
      return result;
    }
  }
  return failure();
}

// A pow2 region of a mixed-radix buffer at a region-aligned start: the start's
// offset and the region's own shared layout, if the region's offsets are that
// offset plus a bijective radix-2 map.
static std::optional<std::pair<int32_t, LinearLayout>>
getMixedRadixRegionLayout(const LinearLayout &toOffset, ArrayRef<int32_t> start,
                          ArrayRef<int64_t> region) {
  MLIRContext *ctx = (*toOffset.getInDimNames().begin()).getContext();
  auto dims = llvm::to_vector(toOffset.getInDimNames());
  auto offsetAt = [&](ArrayRef<int32_t> coords) {
    SmallVector<std::pair<StringAttr, int32_t>> ins;
    for (auto [dim, s, c] : llvm::zip_equal(dims, start, coords))
      ins.push_back({dim, s + c});
    return toOffset.apply(ins)[0].second;
  };
  int32_t base = offsetAt(SmallVector<int32_t>(region.size(), 0));
  int32_t numel = product(region);
  std::vector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases;
  for (auto [d, dim] : llvm::enumerate(dims)) {
    std::vector<std::vector<int32_t>> dimBases;
    for (int32_t bit = 1; bit < region[d]; bit *= 2) {
      SmallVector<int32_t> coords(region.size(), 0);
      coords[d] = bit;
      int32_t delta = offsetAt(coords) - base;
      if (delta < 0 || delta >= numel)
        return std::nullopt;
      dimBases.push_back({delta, 0});
    }
    bases.push_back({dim, std::move(dimBases)});
  }
  LinearLayout toRegionOffset(bases,
                              {{StringAttr::get(ctx, "offset"), numel},
                               {StringAttr::get(ctx, "block"), 1}},
                              /*requireSurjective=*/false);
  if (!toRegionOffset.isInvertible())
    return std::nullopt;
  for (int32_t i = 0; i < numel; ++i) {
    SmallVector<int32_t> coords;
    SmallVector<std::pair<StringAttr, int32_t>> ins;
    int32_t rest = i;
    for (auto [dim, size] : llvm::zip_equal(dims, region)) {
      coords.push_back(rest % size);
      ins.push_back({dim, coords.back()});
      rest /= size;
    }
    if (offsetAt(coords) - base != toRegionOffset.apply(ins)[0].second)
      return std::nullopt;
  }
  return std::make_pair(base, toRegionOffset.invert());
}

// Each chunk is a pow2 view of a pow2 region of the buffer, so it loads as a
// padded operand's view does: ldmatrix, else the radix-2 path.
LogicalResult lowerPow2Chunks(Location loc,
                              const nvidia_gpu::Pow2Chunks &chunks,
                              MemDescType memDescType, SmallVector<Value> &vals,
                              const SharedMemoryObject &smemObj, Operation *op,
                              ConversionPatternRewriter &rewriter,
                              const NVIDIA::TargetInfo &targetInfo,
                              const LLVMTypeConverter *typeConverter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = loc.getContext();
  ArrayRef<int64_t> allocShape = memDescType.getAllocShape();
  size_t rank = chunks.starts.front().size();
  auto getChunkTy = [&](Attribute encoding, ArrayRef<int64_t> region) {
    SmallVector<int64_t> chunkAlloc(allocShape.drop_back(rank));
    llvm::append_range(chunkAlloc, region);
    return MemDescType::get(
        llvm::to_vector_of<int64_t>(chunks.layout.getOutDimSizes()),
        memDescType.getElementType(), encoding, memDescType.getMemorySpace(),
        memDescType.getMutableMemory(), chunkAlloc);
  };
  // A radix-2 buffer is one region. A dense allocation's panels sit rows x
  // width apart, not at the cover's row stride, so a mixed-radix one is viewed
  // a pow2 block at a time.
  Attribute encoding = memDescType.getEncoding();
  SmallVector<int64_t> region;
  std::optional<LinearLayout> toOffset;
  if (isa<PaddedSharedEncodingAttr>(encoding)) {
    region = getNpotPow2AllocShape(allocShape.take_back(rank));
  } else {
    LinearLayout memLayout = toLinearLayout(memDescType);
    for (int32_t size : memLayout.getOutDimSizes())
      region.push_back(memLayout.isRadix2() ? size
                                            : LinearLayout::twoPartOf(size));
    if (memLayout.isRadix2())
      encoding = SharedLinearEncodingAttr::get(
          ctx, std::move(memLayout),
          cast<SharedEncodingTrait>(encoding).getAlignment());
    else
      toOffset = memLayout.pseudoinvert();
  }
  // Each chunk's region offset, region type and start within the region.
  SmallVector<std::tuple<int32_t, MemDescType, SmallVector<int32_t>>> views;
  std::map<SmallVector<int32_t>, std::pair<int32_t, MemDescType>> regions;
  for (ArrayRef<int32_t> start : chunks.starts) {
    if (!toOffset) {
      views.push_back(
          {0, getChunkTy(encoding, region), llvm::to_vector(start)});
      continue;
    }
    SmallVector<int32_t> regionStart, inner;
    for (auto [s, size] : llvm::zip_equal(start, region)) {
      regionStart.push_back(s - s % size);
      inner.push_back(s % size);
    }
    auto it = regions.find(regionStart);
    if (it == regions.end()) {
      auto view = getMixedRadixRegionLayout(*toOffset, regionStart, region);
      if (!view)
        return failure();
      int32_t alignment = llvm::MinAlign(
          cast<SharedEncodingTrait>(encoding).getAlignment(),
          view->first * memDescType.getElementTypeBitWidth() / 8);
      auto regionTy = getChunkTy(SharedLinearEncodingAttr::get(
                                     ctx, std::move(view->second), alignment),
                                 region);
      it = regions.insert({regionStart, {view->first, regionTy}}).first;
    }
    views.push_back({it->second.first, it->second.second, inner});
  }
  Type llvmElemTy = typeConverter->convertType(memDescType.getElementType());
  for (auto &[regionOffset, chunkTy, start] : views) {
    Value base = smemObj.getBase();
    if (regionOffset != 0)
      base = b.gep(base.getType(), smemObj.getBaseElemType(), base,
                   b.i32_val(regionOffset));
    SmallVector<Value> offsets = smemObj.getOffsets();
    for (auto [offset, s] :
         llvm::zip(MutableArrayRef(offsets).take_back(rank), start))
      offset = b.add(offset, b.i32_val(s));
    SharedMemoryObject chunkObj(base, smemObj.getBaseElemType(), offsets);
    SmallVector<Value> chunkVals;
    if (failed(lowerLdStMatrix(loc, chunks.layout, chunkTy, chunkVals, chunkObj,
                               rewriter, targetInfo, typeConverter)))
      chunkVals = lowerLocalLdSt(
          loc, ctx,
          invertAndComposeLocal(toLinearLayoutIgnoringPadding(chunkTy),
                                chunks.layout, {StringAttr::get(ctx, "block")}),
          {}, llvmElemTy, chunkTy, chunkObj, rewriter, targetInfo, op);
    llvm::append_range(vals, chunkVals);
  }
  return success();
}

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LocalLoadOpConversion(const LLVMTypeConverter &converter,
                        const NVIDIA::TargetInfo &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    auto *ctx = op.getContext();
    MemDescType memDescType = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Type llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getSrc(), llvmElemTy, rewriter);

    auto *typeConverter = getTypeConverter();
    llvm::SmallVector<Value> values;
    auto regLayout =
        toLinearLayout(dstTy).removeZeroBasesAlongDim(str_attr("register"));
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    auto dot = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto mma = dot ? dyn_cast<NvidiaMmaEncodingAttr>(dot.getParent()) : nullptr;
    std::optional<nvidia_gpu::Pow2Chunks> chunks;
    // A transposed operand loads in its pre-trans layout, not a dot operand.
    bool ampere = mma ? mma.isAmpere() : targetInfo.getComputeCapability() < 90;
    if (!regLayout.isRadix2() && ampere &&
        (chunks = nvidia_gpu::getPow2Chunks(regLayout))) {
      result =
          lowerPow2Chunks(op.getLoc(), *chunks, memDescType, values, smemObj,
                          op, rewriter, targetInfo, getTypeConverter());
    }
    if (failed(result)) {
      return failure();
    }
    auto value = packUniqueTensorElements(op.getLoc(), typeConverter, values,
                                          rewriter, dstTy);
    rewriter.replaceOp(op, value);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    auto *ctx = op.getContext();
    MemDescType memDescType = op.getType();
    RankedTensorType regTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(regTy.getElementType());
    Value smemBase =
        LLVM::getSharedMemoryBase(op.getLoc(), rewriter, targetInfo, op);
    auto smemObj = SharedMemoryObject(
        smemBase, llvmElemTy, memDescType.getRank(), op.getLoc(), rewriter);

    auto regLayout =
        toLinearLayout(regTy).removeZeroBasesAlongDim(str_attr("register"));
    auto values =
        unpackUniqueTensorElements(op.getLoc(), adaptor.getSrc(), rewriter);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }

    auto retVal =
        getStructFromSharedMemoryObject(op.getLoc(), smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    MemDescType memDescType = op.getDst().getType();
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    SharedMemoryObject smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);

    auto regLayout =
        toLinearLayout(srcTy).removeZeroBasesAlongDim(str_attr("register"));
    auto values =
        unpackUniqueTensorElements(op.getLoc(), adaptor.getSrc(), rewriter);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

static Value mapSharedToCluster(Location loc, Value ptr, Value ctaId,
                                ConversionPatternRewriter &rewriter) {
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 &&
         "st.async.shared expects a shared-memory pointer");
  return NVVM::MapaOp::create(rewriter, loc, ptr_ty(rewriter.getContext(), 7),
                              ptr, ctaId);
}

// Axis constancy is not a proof of cross-CTA uniformity. Recognize common
// ranks conservatively; in particular, never infer uniformity for self rank,
// device-function arguments, loads, or arbitrary inline assembly.
static bool isBlockUniformRank(Value rank, unsigned depth = 16) {
  if (!depth || !rank.getType().isIntOrIndex())
    return false;
  if (matchPattern(rank, m_Constant()))
    return true;
  if (auto arg = dyn_cast<BlockArgument>(rank)) {
    auto func = dyn_cast<FunctionOpInterface>(arg.getOwner()->getParentOp());
    return func && triton::isKernel(func) &&
           arg.getOwner() == &func.getFunctionBody().front();
  }
  auto *def = rank.getDefiningOp();
  if (!def || def->getNumOperands() == 0 || def->getNumRegions() != 0)
    return false;
  bool isArithmetic =
      def->getName().getDialectNamespace() == "arith" ||
      isa<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::AndOp, LLVM::OrOp,
          LLVM::XOrOp, LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp, LLVM::ZExtOp,
          LLVM::SExtOp, LLVM::TruncOp, LLVM::ICmpOp, LLVM::SelectOp,
          LLVM::UDivOp, LLVM::SDivOp, LLVM::URemOp, LLVM::SRemOp>(def);
  return isArithmetic && llvm::all_of(def->getOperands(), [&](Value operand) {
           return isBlockUniformRank(operand, depth - 1);
         });
}

struct AsyncRemoteShmemReduceOpConversion
    : public ConvertOpToLLVMPattern<AsyncRemoteShmemStoreOp> {
  AsyncRemoteShmemReduceOpConversion(const LLVMTypeConverter &converter,
                                     const NVIDIA::TargetInfo &targetInfo,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(AsyncRemoteShmemStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const char *opcode;
    const char *type;
    switch (op.getStoreReduce()) {
    case DescriptorReduceKind::ADD:
      opcode = "add";
      // Signed and unsigned i32 ADD have identical modulo-2^32 semantics.
      type = "u32";
      break;
    case DescriptorReduceKind::MIN:
      opcode = "min";
      type = op.getStoreReduceSigned() ? "s32" : "u32";
      break;
    case DescriptorReduceKind::MAX:
      opcode = "max";
      type = op.getStoreReduceSigned() ? "s32" : "u32";
      break;
    case DescriptorReduceKind::AND:
      opcode = "and";
      type = "b32";
      break;
    case DescriptorReduceKind::OR:
      opcode = "or";
      type = "b32";
      break;
    case DescriptorReduceKind::XOR:
      opcode = "xor";
      type = "b32";
      break;
    case DescriptorReduceKind::INC:
    case DescriptorReduceKind::DEC:
      if (op.getStoreReduceSigned())
        return rewriter.notifyMatchFailure(
            op,
            "async remote INC/DEC reduction requires unsigned 32-bit integers");
      opcode = op.getStoreReduce() == DescriptorReduceKind::INC ? "inc" : "dec";
      type = "u32";
      break;
    default:
      return failure();
    }
    auto module = op->getParentOfType<ModuleOp>();
    if (!module || targetInfo.getComputeCapability() < 90 ||
        (!isPhysicalCluster(module) &&
         TritonGPUDialect::getNumCTAs(module) <= 1))
      return rewriter.notifyMatchFailure(
          op,
          "async remote reduction requires SM90+ and multiple physical CTAs");

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto regTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<MemDescType>(op.getDst().getType());
    if (!regTy.getElementType().isInteger(32) ||
        dstTy.getElementType() != regTy.getElementType())
      return rewriter.notifyMatchFailure(
          op, "async remote reduction requires matching 32-bit integers");
    auto *typeConverter = getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto dstObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                        llvmElemTy, rewriter);
    auto regLayout =
        toLinearLayout(regTy).removeZeroBasesAlongDim(str_attr("register"));
    auto sharedLayout = toLinearLayoutIgnoringPadding(dstTy);
    auto kBlock = str_attr("block");
    auto cvt = invertAndComposeLocal(sharedLayout, regLayout, {kBlock});
    auto vals = unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);

    auto barrierTy = cast<MemDescType>(op.getBarrier().getType());
    auto barrierElemTy = typeConverter->convertType(barrierTy.getElementType());
    auto barrierObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getBarrier(), barrierElemTy, rewriter);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // Include subslice offsets before mapping both addresses to the same rank.
    Value barrierOffset = b.shl(
        barrierObj.getShmemOffset(loc, rewriter, barrierTy), b.i32_val(3));
    if (isPaddedEncoding(barrierTy.getEncoding()))
      barrierOffset =
          applyPadding(loc, rewriter, barrierOffset,
                       getPaddedSharedShifts(barrierTy.getEncoding(), 64,
                                             /*offsetInBytes=*/true));
    Value barrierPtr = b.gep(barrierObj.getBase().getType(), i8_ty,
                             barrierObj.getBase(), barrierOffset);
    Value rank = adaptor.getCtaRank();
    Value mbarrier = mapSharedToCluster(loc, barrierPtr, rank, rewriter);
    Value blockPred;
    if (TritonGPUDialect::getNumCTAs(module) > 1 &&
        (isBlockUniformRank(op.getCtaRank()) || isBlockUniformRank(rank))) {
      // Logical CTA replicas writing a common target must complete each
      // element only once. Physical num-ctas=1 programs are not replicas.
      llvm::MapVector<StringAttr, int32_t> blockMasks;
      blockMasks[str_attr("block")] =
          regLayout.getFreeVariableMasks().lookup(str_attr("block"));
      blockPred = triton::gpu::emitRedundantThreadPredicate(
          blockMasks, rewriter, loc, targetInfo);
    }
    auto emitReduce = [&](RewriterBase &, Location emitLoc, Value ptr,
                          Value /*ctaId*/, Value value, Value pred) {
      Value dst = mapSharedToCluster(emitLoc, ptr, rank, rewriter);
      PTXBuilder builder;
      auto &red = *builder.create("red.async");
      red.o("relaxed")
          .o("cluster")
          .o("shared::cluster")
          .o("mbarrier::complete_tx::bytes")
          .o(opcode)
          .o(type);
      // INC/DEC use the source value as the unsigned bound.
      auto *dstOpr = builder.newAddrOperand(dst, "r");
      auto *valueOpr = builder.newOperand(value, "r");
      auto *barrierOpr = builder.newAddrOperand(mbarrier, "r");
      pred = triton::gpu::maybeAnd(rewriter, emitLoc, pred, blockPred);
      red(dstOpr, valueOpr, barrierOpr).predicate(pred, "b");
      builder.launch(rewriter, emitLoc, void_ty(ctx));
    };
    lowerLocalLdSt(loc, ctx, cvt, vals, llvmElemTy, dstTy, dstObj, rewriter,
                   targetInfo, nullptr, rank, {}, {}, emitReduce);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct AsyncSharedStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::AsyncSharedStoreOp> {
  AsyncSharedStoreOpConversion(const LLVMTypeConverter &converter,
                               const NVIDIA::TargetInfo &targetInfo,
                               PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::nvidia_gpu::AsyncSharedStoreOp>(
            converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::AsyncSharedStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!triton::nvidia_gpu::AsyncSharedStoreOp::isSupported(
            targetInfo.getComputeCapability()))
      return op.emitError("requires cluster-capable SM90+");

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MemDescType dstTy = op.getDst().getType();
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getDst(), llvmElemTy, rewriter);
    auto mbarrierTy = op.getMbarrier().getType();
    auto mbarrierMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getMbarrier(),
        typeConverter->convertType(mbarrierTy.getElementType()), rewriter);

    auto kBlock = str_attr("block");
    auto regLayout = toLinearLayout(srcTy);
    auto freeVarMasks = regLayout.getFreeVariableMasks();
    freeVarMasks[kBlock] = 0;
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    Value mapPred = threadPred ? threadPred : b.true_val();
    regLayout = regLayout.removeZeroBasesAlongDim(str_attr("register"));
    auto sharedLayout = toLinearLayoutIgnoringPadding(dstTy);
    auto cvt = invertAndComposeLocal(sharedLayout, regLayout, {kBlock});
    auto values = unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
    Value currentCTAId = targetInfo.getClusterCTAId(rewriter, loc);
    Value mbarrierPtr = mbarrierMemObj.getBase();
    auto emitStore = [&](RewriterBase &, Location storeLoc,
                         ArrayRef<Value> values, Value shmemAddr, int idx,
                         VectorType vecTy, Value ctaId) -> SmallVector<Value> {
      Value targetCTAId = ctaId ? ctaId : currentCTAId;
      Value dst = targetInfo.mapDShared(rewriter, storeLoc, shmemAddr,
                                        targetCTAId, mapPred);
      Value mbarrier = targetInfo.mapDShared(rewriter, storeLoc, mbarrierPtr,
                                             targetCTAId, mapPred);
      mbarrier = LLVM::NVIDIA::getLeaderAddress(storeLoc, rewriter, mbarrier,
                                                mbarrierTy);

      Type elemTy = vecTy.getElementType();
      unsigned elemBitwidth = getIntOrFloatOrPtrBitWidth(elemTy);
      unsigned storeBitwidth = std::max(32u, elemBitwidth);
      unsigned elemsPerStore = storeBitwidth / elemBitwidth;
      unsigned storeVec = vecTy.getNumElements() / elemsPerStore;
      std::string constraint = storeBitwidth == 64 ? "l" : "r";

      SmallVector<std::pair<Value, std::string>> storeValues;
      for (unsigned i = 0; i < storeVec; ++i) {
        auto elems = values.slice(idx + i * elemsPerStore, elemsPerStore);
        Value value = elems.front();
        if (elemsPerStore > 1)
          value = b.bitcast(packLLVector(storeLoc, elems, rewriter),
                            int_ty(storeBitwidth));
        else if (isa<LLVM::LLVMPointerType>(elemTy))
          value = b.ptrtoint(int_ty(storeBitwidth), value);
        else if (!elemTy.isInteger())
          value = b.bitcast(value, int_ty(storeBitwidth));
        storeValues.emplace_back(value, constraint);
      }

      PTXBuilder ptxBuilder;
      auto &store = ptxBuilder.create("st.async")
                        ->o("weak")
                        .o("shared::cluster")
                        .o("mbarrier::complete_tx::bytes")
                        .v(storeVec, /*predicate=*/storeVec > 1)
                        .b(storeBitwidth);
      auto *dstOperand = ptxBuilder.newAddrOperand(dst, "r");
      auto *valueOperand =
          storeVec == 1
              ? ptxBuilder.newOperand(storeValues.front().first, constraint)
              : ptxBuilder.newListOperand(storeValues);
      auto *barrierOperand = ptxBuilder.newAddrOperand(mbarrier, "r");
      store(dstOperand, valueOperand, barrierOperand)
          .maybePredicate(threadPred, "b");
      ptxBuilder.launch(rewriter, storeLoc, void_ty(rewriter.getContext()));
      return {};
    };
    lowerLocalLdSt(loc, ctx, cvt, values, llvmElemTy, dstTy, dstMemObj,
                   rewriter, targetInfo, emitStore);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalAtomicScatterRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAtomicScatterRMWOp> {
public:
  LocalAtomicScatterRMWOpConversion(const LLVMTypeConverter &converter,
                                    const NVIDIA::TargetInfo &targetInfo,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAtomicScatterRMWOp>(converter,
                                                                     benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAtomicScatterRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto lowering = prepareLocalAtomicScatterRMW(
        op, adaptor.getDst(), adaptor.getIndices(), adaptor.getValues(),
        op.getMask() ? adaptor.getMask() : Value(), rewriter, targetInfo,
        getTypeConverter());
    if (failed(lowering))
      return failure();
    LocalAtomicScatterRMWInfo &info = *lowering;

    RMWOp rmwOp = op.getAtomicRmwOp();
    bool isI32Inc = rmwOp == RMWOp::ADD &&
                    info.valuesTy.getElementType().isInteger(32) &&
                    isConstI32OneTensor(op.getValues());
    bool returnOld = !op.getResult().use_empty();

    SmallVector<Value> results;
    if (returnOld)
      results.reserve(info.addrs.size());
    for (auto [i, addrAndValue] :
         llvm::enumerate(llvm::zip(info.addrs, info.values))) {
      auto [addr, value] = addrAndValue;
      Value pred =
          maybeAnd(rewriter, loc, info.threadPred,
                   info.maskValues.empty() ? Value() : info.maskValues[i]);
      bool isCluster = bool(addr.ctaId);
      Value ptr =
          targetInfo.mapDShared(rewriter, loc, addr.ptr, addr.ctaId, pred);
      if (isI32Inc) {
        Value result =
            emitSharedInc(rewriter, loc, ptr, returnOld, isCluster, pred);
        if (returnOld)
          results.push_back(result);
        continue;
      }
      auto old = emitSharedAtomicRMW(rewriter, loc, info.llvmElemTy, ptr, value,
                                     rmwOp, returnOld, isCluster, pred);
      if (failed(old))
        return failure();
      if (returnOld)
        results.push_back(*old);
    }

    if (!returnOld) {
      rewriter.eraseOp(op);
      return success();
    }

    finalizeAtomicResults(op, rewriter, results, info.llvmElemTy, b,
                          info.threadPred, targetInfo, getTypeConverter());
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  // Backend optimized memory ops get higher benefit
  patterns.add<AsyncRemoteShmemReduceOpConversion>(typeConverter, targetInfo,
                                                   benefit.getBenefit() + 1);
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<AsyncSharedStoreOpConversion>(typeConverter, targetInfo,
                                             benefit.getBenefit() + 1);
  patterns.add<LocalAtomicScatterRMWOpConversion>(typeConverter, targetInfo,
                                                  benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo,
                                      benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);
}
