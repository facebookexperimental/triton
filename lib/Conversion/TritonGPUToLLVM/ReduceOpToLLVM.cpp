#include "ReduceScanCommon.h"

#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DistributedEncodingTrait;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getThreadOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {
struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(LLVMTypeConverter &typeConverter,
                     const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(typeConverter,
                                                                  benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isInnerTree(op))
      return rewriteInnerTree(op, adaptor, rewriter);

    ReduceOpHelper helper(op);
    auto regLl = ReduceOpHelper::reducedRegLaneLayout(helper.getSrcTy(),
                                                      op.getAxis());
    auto kAxis = *(regLl.getOutDimNames().begin() + op.getAxis());
    if (regLl.getOutDimSize(kAxis) != 1) {
      return rewriteInnerTree(op, adaptor, rewriter);
    }

    return rewriteDefault(op, adaptor, rewriter);
  }

private:
  const TargetInfoBase &targetInfo;

  bool isInnerTree(triton::ReduceOp op) const {
    auto attr = op.getReductionOrderingAttr();
    return attr && attr.getValue() == "inner_tree";
  }

  LogicalResult rewriteDefault(triton::ReduceOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    ReduceOpHelper helper(op);
    Location loc = op->getLoc();
    auto accs = unpackInputsByOperand(loc, op, adaptor, rewriter);
    unsigned axis = op.getAxis();

    auto *ctx = op.getContext();

    // Remove block as we don't currently support it
    LinearLayout regLl = triton::gpu::toLinearLayout(helper.getSrcTy());
    // Remove broadcasting in registers as SliceLayout removes them
    auto removeBroadcast = actionRemoveBroadcastedRegs(regLl);
    if (!removeBroadcast.isIdentity()) {
      regLl = removeBroadcast.apply(regLl);
      for (auto &vals : accs)
        vals = removeBroadcast.apply(vals);
    }

    std::tie(regLl, accs) =
        reduceWithinWarps(op, std::move(regLl), std::move(accs), rewriter);

    // reducedRegLaneLayout is used in the AllocationAnalysis to get the size
    // of the scratch space.
    assert(regLl ==
           ReduceOpHelper::reducedRegLaneLayout(helper.getSrcTy(), axis));

    // If we still need to reduce along warps / blocks:
    // Create temporary layout for reduction within warps.
    // By construction of tmpLl, we will iterate at most 2 times, as the maximum
    // number of warp / block bases is 64 * 16 = 32 * 32
    // That is, they fit in 2 rounds of warp reductions
    // Even more, if we do two rounds, getInterLayout will make sure that the
    // first one does not cross CTAs
    auto kAxis = *(regLl.getOutDimNames().begin() + axis);
    auto kBlock = StringAttr::get(ctx, "block");
    bool lastCvtCrossesCTAs = false;
    int i = 0;
    while (regLl.getOutDimSize(kAxis) != 1) {
      LinearLayout tmpLl = ReduceOpHelper::getInterLayout(regLl, axis);

      // Emit a barrier if we are reusing the shmem.
      if (i > 0)
        sync(rewriter, loc, lastCvtCrossesCTAs);

      accs = convertLayoutValues(loc, rewriter, op, regLl, tmpLl, accs);
      lastCvtCrossesCTAs = !mlir::isCvtDimSync(regLl, tmpLl, kBlock);

      std::tie(regLl, accs) =
          reduceWithinWarps(op, std::move(tmpLl), std::move(accs), rewriter);
      ++i;
    }
    assert(i <= 2 && "expected at most 2 rounds of warp reductions");

    // Remove the axis dimension, which at this point is of size 1.
    regLl = removeStandardDim(regLl, axis);

    if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto outputLayout = triton::gpu::toLinearLayout(resultTy);
      if (regLl != outputLayout) {
        // Reuse the shmem.
        sync(rewriter, loc, lastCvtCrossesCTAs);
        accs =
            convertLayoutValues(loc, rewriter, op, regLl, outputLayout, accs);
      }
    }

    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        results[i] = packLLElements(loc, getTypeConverter(), accs[i], rewriter,
                                    resultTy);
      } else {
        assert(accs[i].size() == 1 && "expected scalar reduce result");
        results[i] = accs[i][0];
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }

  LogicalResult rewriteInnerTree(triton::ReduceOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
    ReduceOpHelper helper(op);
    // Multi-CTA reduction pass generates tt.reduce on 1-element tensors
    // loaded from DSM buffers. These are within-CTA (each CTA has its own
    // buffer copy), but the encoding may not reflect this if cluster_dims > 1.
    // Only allow these specific 1-element cases through.
    if (!helper.isReduceWithinCTA()) {
      auto srcTy = cast<RankedTensorType>(op.getOperands()[0].getType());
      if (srcTy.getShape()[op.getAxis()] != 1) {
        return op.emitError(
            "cross-CTA reduce on tensor with reduction axis size > 1 is not "
            "supported; only 1-element tensors from multi-CTA DSM exchange "
            "are allowed");
      }
      LDBG("Cross-CTA reduce on 1-element tensor (multi-CTA DSM exchange), "
           "proceeding with within-CTA lowering");
    }
    Location loc = op->getLoc();

    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    // Then reduce across threads within a warp.
    reduceWithinWarps(helper, accs, rewriter);

    if (helper.isWarpSynchronous()) {
      // If all the values to be reduced are within the same warp there is
      // nothing left to do.
      packResults(helper, accs, rewriter);
      return success();
    }

    // Compute a shared memory base per operand.
    auto smemShape = helper.getScratchRepShape();

    SmallVector<Value> smemBases =
        getSmemBases(op, product<unsigned>(smemShape), rewriter, targetInfo);

    storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter);

    sync(rewriter, loc, op);

    // The second round of shuffle reduction
    //   now the problem size: sizeInterWarps, s1, s2, .. , sn
    //   where sizeInterWarps is 2^m
    //
    // Each thread needs to process:
    //   elemsPerThread = sizeInterWarps * s1 * s2 .. Sn / numThreads
    accumulatePartialReductions(helper, smemBases, rewriter);

    // We could avoid this barrier in some of the layouts, however this is not
    // the general case.
    // TODO: optimize the barrier in case the layouts are accepted.
    sync(rewriter, loc, op);

    // set output values
    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);

    return success();
  }

  SmallVector<Value>
  treeReduceBinary(Location loc, ConversionPatternRewriter &rewriter,
                   Region &combineOp,
                   SmallVector<SmallVector<Value>> values) const {
    // The number of elements is always a power of two
    assert(llvm::isPowerOf2_64(values.size()) && !values.empty());
    while (values.size() > 1) {
      SmallVector<SmallVector<Value>> next;
      for (size_t i = 0; i + 1 < values.size(); i += 2) {
        SmallVector<Value> acc = values[i];
        accumulate(loc, rewriter, combineOp, acc, values[i + 1]);
        next.push_back(std::move(acc));
      }
      values = std::move(next);
    }
    return values.front();
  }
  void accumulate(Location loc, ConversionPatternRewriter &rewriter,
                  Region &combineOp, SmallVector<Value> &acc, ValueRange cur,
                  Value pred = {}) const {
    auto results = applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
    if (acc.size() < results.size()) {
      acc.resize(results.size());
    }
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }
  }

  SmallVector<unsigned> getRegGroupKey(ArrayRef<unsigned> offset, unsigned axis,
                                       unsigned groupSpan) const {
    SmallVector<unsigned> key(offset.begin(), offset.end());
    unsigned axisOffset = key[axis];
    key[axis] = (axisOffset / groupSpan) * groupSpan;
    return key;
  }

  SmallVector<Value>
  reduceValueSequence(Location loc, triton::ReduceOp op,
                      SmallVector<SmallVector<Value>> values,
                      ConversionPatternRewriter &rewriter) const {
    if (values.empty())
      return {};

    if (isInnerTree(op)) {
      while (values.size() > 1) {
        SmallVector<SmallVector<Value>> next;
        for (unsigned i = 0; i + 1 < values.size(); i += 2) {
          SmallVector<Value> merged = values[i];
          accumulate(loc, rewriter, op.getCombineOp(), merged, values[i + 1]);
          next.push_back(std::move(merged));
        }
        if (values.size() % 2 == 1)
          next.push_back(std::move(values.back()));
        values = std::move(next);
      }
      return std::move(values[0]);
    }

    SmallVector<Value> acc;
    for (auto &cur : values)
      accumulate(loc, rewriter, op.getCombineOp(), acc, cur);
    return acc;
  }

  bool matchesResultOffset(ArrayRef<unsigned> key,
                           ArrayRef<unsigned> resultOffs, unsigned axis) const {
    if (key.size() != resultOffs.size() + 1)
      return false;
    for (unsigned dim = 0; dim < resultOffs.size(); ++dim) {
      unsigned keyDim = dim < axis ? dim : dim + 1;
      if (key[keyDim] != resultOffs[dim])
        return false;
    }
    return true;
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  SmallVector<SmallVector<Value>>
  unpackInputsByOperand(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    auto operands = adaptor.getOperands();
    SmallVector<SmallVector<Value>> srcValues;
    srcValues.reserve(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i)
      srcValues.push_back(unpackLLElements(loc, operands[i], rewriter));
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            bool crossCTA) const {
    if (crossCTA) {
      targetInfo.clusterBarrier(loc, rewriter);
    } else {
      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
    }
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            triton::ReduceOp) const {
    targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
  }

  SmallVector<Value> transferSwizzlingLocalMemImpl(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    if (isa<LLVM::LLVMPointerType>(llvmElemTy)) {
      auto llvmElemTyPtr = i64_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(inVals, [&](Value v) {
        return b.ptrtoint(llvmElemTyPtr, v).getResult();
      }));
      auto outVals =
          transferSwizzlingLocalMemImpl(loc, rewriter, srcLayout, dstLayout,
                                        newInVals, llvmElemTyPtr, smemBase);
      for (auto &v : outVals)
        v = b.inttoptr(llvmElemTy, v);
      return outVals;
    }

    if (llvmElemTy.getIntOrFloatBitWidth() < 8) {
      auto i8ElemTy = i8_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(
          inVals, [&](Value v) { return b.zext(i8ElemTy, v).getResult(); }));
      auto outVals = transferSwizzlingLocalMemImpl(
          loc, rewriter, srcLayout, dstLayout, newInVals, i8ElemTy, smemBase);
      for (auto &v : outVals)
        v = b.trunc(llvmElemTy, v);
      return outVals;
    }

    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    if (!removeBroadcastSrc.isIdentity()) {
      auto prmtSrc = removeBroadcastSrc.apply(srcLayout);
      auto newInVals = removeBroadcastSrc.apply(inVals);
      return transferSwizzlingLocalMemImpl(loc, rewriter, prmtSrc, dstLayout,
                                           newInVals, llvmElemTy, smemBase);
    }

    auto removeBroadcastDst = actionRemoveBroadcastedRegs(dstLayout);
    if (!removeBroadcastDst.isIdentity()) {
      auto prmtDst = removeBroadcastDst.apply(dstLayout);
      auto outVals = transferSwizzlingLocalMemImpl(
          loc, rewriter, srcLayout, prmtDst, inVals, llvmElemTy, smemBase);
      return broadcastAs(outVals, dstLayout);
    }

    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto smem = triton::gpu::optimalSwizzlingLdSt(srcLayout, dstLayout,
                                                  bitwidth);

    auto kReg = str_attr("register");
    auto kWarp = str_attr("warp");
    auto kBlock = str_attr("block");
    auto kReps = str_attr("reps");
    auto nReps = smem.getInDimSize(kReps);
    auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

    auto totalStoreCvt = srcLayout.invertAndCompose(smem);
    auto totalLoadCvt = dstLayout.invertAndCompose(smem);

    auto permStore = regPermForDivide(totalStoreCvt, reps, /*left=*/false)
                         .value();
    totalStoreCvt = permStore.apply(totalStoreCvt);
    auto permutedInVals = permStore.apply(inVals);
    auto permLoad = regPermForDivide(totalLoadCvt, reps, /*left=*/false)
                        .value();
    totalLoadCvt = permLoad.apply(totalLoadCvt);

    auto storeCvt = *divideRight(totalStoreCvt, reps);
    auto loadCvt = *divideRight(totalLoadCvt, reps);
    auto kOffset = str_attr("offset");
    auto nBlock = storeCvt.getInDimSize(kBlock);
    storeCvt = storeCvt.reshapeOuts(
        {{kOffset, storeCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});
    loadCvt = loadCvt.reshapeOuts(
        {{kOffset, loadCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});

    auto tileSize = storeCvt.getInDimSize(kReg);

    assert(permutedInVals.size() == tileSize * nReps);
    SmallVector<Value> outVals;
    auto affineOffset = b.i32_val(0);
    auto maskSpanAffineOffset = 0;

    bool isWarpSync = mlir::isCvtDimSync(srcLayout, dstLayout, kWarp);
    bool isBlockSync = mlir::isCvtDimSync(srcLayout, dstLayout, kBlock);
    auto emitBarrier = [&]() {
      if (isWarpSync) {
        targetInfo.warpSync(loc, rewriter);
      } else if (isBlockSync) {
        targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
      } else {
        targetInfo.clusterBarrier(loc, rewriter);
      }
    };

    for (int i = 0; i < nReps; ++i) {
      if (i > 0)
        emitBarrier();
      auto tileInVals =
          ArrayRef<Value>(permutedInVals).slice(i * tileSize, tileSize);
      lowerLdStShared(loc, ctx, storeCvt, tileInVals, llvmElemTy, smemBase,
                      /*paddingShifts=*/{}, affineOffset, maskSpanAffineOffset,
                      rewriter, targetInfo);
      emitBarrier();
      auto tileOutVals = lowerLdStShared(
          loc, ctx, loadCvt, {}, llvmElemTy, smemBase, /*paddingShifts=*/{},
          affineOffset, maskSpanAffineOffset, rewriter, targetInfo);
      llvm::append_range(outVals, tileOutVals);
    }

    outVals = permLoad.inverse().apply(outVals);
    return outVals;
  }

  SmallVector<int64_t>
  getSmemBaseOffsets(triton::ReduceOp op, const LinearLayout &srcLayout,
                     const LinearLayout &dstLayout) const {
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);
    llvm::stable_sort(indices, [&](unsigned lhs, unsigned rhs) {
      return getBitwidth(op.getInputTypes()[lhs]) >
             getBitwidth(op.getInputTypes()[rhs]);
    });

    SmallVector<int64_t> offsets(op.getNumOperands());
    int64_t offset = 0;
    for (unsigned idx : indices) {
      offsets[idx] = offset;
      auto inputTy = op.getInputTypes()[idx];
      auto bytes = getNumScratchElemsSwizzledCvt(srcLayout, dstLayout,
                                                 getBitwidth(inputTy)) *
                   (getBitwidth(inputTy) / 8);
      offset += bytes;
    }
    return offsets;
  }

  SmallVector<SmallVector<Value>>
  convertLayoutValues(Location loc, ConversionPatternRewriter &rewriter,
                      triton::ReduceOp op, const LinearLayout &srcLayout,
                      const LinearLayout &dstLayout,
                      const SmallVector<SmallVector<Value>> &inVals) const {
    SmallVector<SmallVector<Value>> outVals(op.getNumOperands());
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto base =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto baseOffsets = getSmemBaseOffsets(op, srcLayout, dstLayout);

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto inputTy = cast<RankedTensorType>(op.getInputTypes()[i]);
      auto llvmElemTy =
          getTypeConverter()->convertType(inputTy.getElementType());
      auto smemBase = b.gep(base.getType(), i8_ty, base,
                            b.i32_val(baseOffsets[i]));
      outVals[i] = transferSwizzlingLocalMemImpl(
          loc, rewriter, srcLayout, dstLayout, inVals[i], llvmElemTy, smemBase);
    }
    return outVals;
  }

  void packVectorized(SmallVector<SmallVector<Value>> &accs,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = accs.front().front().getLoc();
    for (auto &acc : accs) {
      SmallVector<Value> packedAcc;
      for (unsigned reg = 0; reg < acc.size(); reg += 2) {
        auto vector = packLLVector(loc, {acc[reg], acc[reg + 1]}, rewriter);
        packedAcc.emplace_back(std::move(vector));
      }
      acc = std::move(packedAcc);
    }
  }

  std::unique_ptr<Region> createVectorCombineRegion(
      Location loc, Type elemTy,
      ReduceOpHelper::InThreadVectorizeOpKind vectorizeKind,
      ConversionPatternRewriter &rewriter) const {
    if (vectorizeKind == ReduceOpHelper::InThreadVectorizeOpKind::None)
      return nullptr;
    MLIRContext *ctx = rewriter.getContext();
    auto vecTy = vec_ty(elemTy, 2);

    auto storage = std::make_unique<Region>();
    auto *block = new Block();
    storage->push_back(block);
    block->addArgument(vecTy, loc);
    block->addArgument(vecTy, loc);

    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(block);
    Value result = ReduceOpHelper::createInThreadVectorizedCombineOp(
        builder, loc, vectorizeKind, block->getArgument(0),
        block->getArgument(1));
    triton::ReduceReturnOp::create(builder, loc, ValueRange{result});
    return storage;
  }

  void unpackVectorized(Location loc, SmallVector<SmallVector<Value>> &accs,
                        ConversionPatternRewriter &rewriter,
                        Region *reduction) const {
    for (auto &acc : accs) {
      SmallVector<Value> unpacked;
      for (Value val : acc) {
        auto elems = unpackLLVector(loc, val, rewriter);
        assert(elems.size() == 2 && "expected a 2-lane packed vector");
        if (reduction) {
          SmallVector<Value> cur = {elems[0]};
          accumulate(loc, rewriter, *reduction, cur, {elems[1]});
          unpacked.emplace_back(cur[0]);
        } else {
          unpacked.emplace_back(elems[0]);
          unpacked.emplace_back(elems[1]);
        }
      }
      acc = std::move(unpacked);
    }
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  void reduceWithinThreads(
      ReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    RankedTensorType operandType = op.getInputTypes()[0];
    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(helper.getSrcLayout(), operandType);

    llvm::MapVector<ArrayRef<unsigned>, int> uniqueOffsets;
    for (int i = 0; i < offsets.size(); ++i) {
      uniqueOffsets.insert({offsets[i], i});
    }

    auto srcIndices = emitIndices(op.getLoc(), rewriter, targetInfo,
                                  helper.getSrcLayout(), operandType, true);
    unsigned axis = op.getAxis();
    unsigned contigPerThread = helper.getContigPerThreadOnReductionAxis();

    SmallVector<int> iterOrder;
    for (const auto &[_, i] : uniqueOffsets)
      iterOrder.push_back(i);

    std::map<SmallVector<unsigned>, SmallVector<int>> regGroups;
    for (int idx : iterOrder) {
      regGroups[getRegGroupKey(offsets[idx], axis, contigPerThread)].push_back(
          idx);
    }

    if (isInnerTree(op)) {
      for (auto &[key, group] : regGroups) {
        llvm::sort(group, [&](int lhs, int rhs) {
          return offsets[lhs][axis] < offsets[rhs][axis];
        });

        SmallVector<SmallVector<Value>> groupValues;
        groupValues.reserve(group.size());
        for (int idx : group)
          groupValues.push_back(srcValues[idx]);

        accs[key] = reduceValueSequence(op.getLoc(), op,
                                        std::move(groupValues), rewriter);
        indices[key] = srcIndices[group.front()];
      }
      return;
    }

    auto reduceGroup = [&](SmallVector<int> &group) {
      llvm::sort(group, [&](int lhs, int rhs) {
        return offsets[lhs][axis] < offsets[rhs][axis];
      });

      SmallVector<SmallVector<Value>> groupValues;
      groupValues.reserve(group.size());
      for (int idx : group)
        groupValues.push_back(srcValues[idx]);

      auto vectorizeKind = helper.getInThreadVectorizeOpKind(
          groupValues.size(), targetInfo.supportBitwidth16Elementwise(),
          targetInfo.supportBitwidth32Elementwise());
      if (vectorizeKind == ReduceOpHelper::InThreadVectorizeOpKind::None)
        return reduceValueSequence(op.getLoc(), op, std::move(groupValues),
                                   rewriter);

      unsigned numOperands = op.getNumOperands();
      SmallVector<SmallVector<Value>> packedByOperand(numOperands);
      for (auto &values : groupValues) {
        for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx)
          packedByOperand[opIdx].push_back(values[opIdx]);
      }
      packVectorized(packedByOperand, rewriter);

      SmallVector<SmallVector<Value>> packedValues;
      for (unsigned i = 0; i < packedByOperand.front().size(); ++i) {
        SmallVector<Value> values;
        for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx)
          values.push_back(packedByOperand[opIdx][i]);
        packedValues.push_back(std::move(values));
      }

      auto elemTy = operandType.getElementType();
      auto vectorCombineRegion = createVectorCombineRegion(
          op.getLoc(), elemTy, vectorizeKind, rewriter);
      auto reduced =
          treeReduceBinary(op.getLoc(), rewriter, *vectorCombineRegion,
                           std::move(packedValues));

      SmallVector<SmallVector<Value>> reducedByOperand(numOperands);
      for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx)
        reducedByOperand[opIdx].push_back(reduced[opIdx]);
      unpackVectorized(op.getLoc(), reducedByOperand, rewriter,
                       &op.getCombineOp());

      SmallVector<Value> result;
      for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx)
        result.push_back(reducedByOperand[opIdx][0]);
      return result;
    };

    for (auto &[groupKey, group] : regGroups) {
      auto reduced = reduceGroup(group);
      auto key = groupKey;
      key[axis] = 0;
      bool isFirst = accs.find(key) == accs.end();
      if (isFirst) {
        accs[key] = std::move(reduced);
        indices[key] = srcIndices[group.front()];
      } else {
        accumulate(op.getLoc(), rewriter, op.getCombineOp(), accs[key],
                   reduced);
      }
    }
  }

  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinWarps(triton::ReduceOp op, LinearLayout layout,
                    SmallVector<SmallVector<Value>> accs,
                    ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto loc = op.getLoc();
    unsigned axis = op.getAxis();
    auto kReg = str_attr("register");
    auto linearAttr = triton::gpu::LinearEncodingAttr::get(ctx, layout);
    auto basesPerDim = linearAttr.basesPerDim(kReg, /*skipBroadcast=*/true);
    unsigned axisPack = basesPerDim[axis];
    if (axisPack == 1) {
      return {std::move(layout), std::move(accs)};
    }

    ReduceOpHelper helper(op);
    auto vectorizeKind = helper.getInThreadVectorizeOpKind(
        axisPack, targetInfo.supportBitwidth16Elementwise(),
        targetInfo.supportBitwidth32Elementwise());
    bool vectorize =
        vectorizeKind != ReduceOpHelper::InThreadVectorizeOpKind::None;

    // Bring the registers that move the axis to the front.
    auto perm = ReduceOpHelper::moveAxisBasesToFront(layout, axis, vectorize);
    if (!perm.isIdentity()) {
      layout = perm.apply(layout);
      for (auto &vals : accs)
        vals = perm.apply(vals);
    }

    // Pack the inputs into vector values.
    if (vectorize)
      packVectorized(accs, rewriter);

    // If we pack along the reduction axis we need to process half the
    // registers.
    const auto &regBases = layout.getBases().lookup(kReg);
    bool packAlongAxis = vectorize && regBases.front()[axis] != 0;
    if (packAlongAxis)
      axisPack /= 2;

    auto elemTy =
        cast<RankedTensorType>(op.getOperandTypes().front()).getElementType();
    std::unique_ptr<Region> vectorCombineRegion =
        createVectorCombineRegion(loc, elemTy, vectorizeKind, rewriter);
    Region &combineRegion =
        vectorCombineRegion ? *vectorCombineRegion : op.getCombineOp();

    unsigned numOperands = accs.size();
    SmallVector<SmallVector<Value>> reduced(numOperands);
    unsigned regs = accs.front().size();
    for (unsigned regBase = 0; regBase < regs; regBase += axisPack) {
      SmallVector<SmallVector<Value>> vals;
      for (unsigned i = 0; i < axisPack; ++i) {
        SmallVector<Value> cur(numOperands);
        for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx)
          cur[opIdx] = accs[opIdx][regBase + i];
        vals.push_back(std::move(cur));
      }
      auto acc =
          treeReduceBinary(loc, rewriter, combineRegion, std::move(vals));
      for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx)
        reduced[opIdx].push_back(acc[opIdx]);
    }
    accs = std::move(reduced);

    // Reduce one last time via the scalar combine op if vector packing also
    // packed along the reduction axis.
    if (vectorize) {
      Region *reduceAfterUnpacking =
          packAlongAxis ? &op.getCombineOp() : nullptr;
      unpackVectorized(loc, accs, rewriter, reduceAfterUnpacking);
    }

    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, axis, kReg);
    layout = actionRemoveBroadcastedRegs(layout).apply(layout);
    return {std::move(layout), std::move(accs)};
  }

  // Apply warp reduction across the given number of contiguous lanes using op
  // region and the accumulator values as source.
  void warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce, unsigned interleave,
                  Value pred = {}) const {
    auto success = targetInfo.warpReduce(rewriter, loc, acc, op,
                                         numLaneToReduce, interleave);
    if (success)
      return;

    if (isInnerTree(op)) {
      // INNER_TREE: count-up shuffle order (1, 2, 4, ...) to build the
      // reduction tree from adjacent lanes first. This ensures bitwise-
      // identical results regardless of num_warps, because the tree
      // structure is determined by lane proximity, not by the total
      // number of active lanes.
      for (unsigned N = 1; N <= numLaneToReduce / 2; N <<= 1) {
        SmallVector<Value> shfl(acc.size());
        for (unsigned i = 0; i < acc.size(); ++i) {
          shfl[i] =
              targetInfo.shuffleXor(rewriter, loc, acc[i], N * interleave);
        }
        accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl, pred);
      }
    } else {
      for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
        SmallVector<Value> shfl(acc.size());
        for (unsigned i = 0; i < acc.size(); ++i) {
          shfl[i] =
              targetInfo.shuffleXor(rewriter, loc, acc[i], N * interleave);
        }
        accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl, pred);
      }
    }
  }

  // Reduce across threads within each warp.
  void
  reduceWithinWarps(ReduceOpHelper &helper,
                    std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                    ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    unsigned sizeIntraWarps = helper.getIntraWarpSizeWithUniqueData();
    unsigned threadOffsetOnReductionAxis =
        helper.getThreadOffsetOnReductionAxis();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = accs[key];
      warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps,
                 threadOffsetOnReductionAxis);
    }
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results(op.getNumOperands());
    if (auto firstResultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto resultLayout = cast<SliceEncodingAttr>(firstResultTy.getEncoding());
      unsigned resultElems = getTotalElemsPerThread(firstResultTy);
      SmallVector<SmallVector<unsigned>> resultOffsets =
          emitOffsetForLayout(resultLayout, firstResultTy);
      SmallVector<SmallVector<Value>> resultVals(
          op.getNumOperands(), SmallVector<Value>(resultElems));

      for (unsigned j = 0; j < resultElems; ++j) {
        SmallVector<SmallVector<Value>> groupVals;
        for (const auto &[key, vals] : accs) {
          if (matchesResultOffset(key, resultOffsets[j], axis))
            groupVals.push_back(vals);
        }
        SmallVector<Value> reduced =
            reduceValueSequence(loc, op, std::move(groupVals), rewriter);
        for (unsigned i = 0; i < op.getNumOperands(); ++i)
          resultVals[i][j] = reduced[i];
      }

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto resultTy = cast<RankedTensorType>(op.getResult()[i].getType());
        results[i] = packLLElements(loc, getTypeConverter(), resultVals[i],
                                    rewriter, resultTy);
      }
    } else {
      SmallVector<SmallVector<Value>> groupVals;
      groupVals.reserve(accs.size());
      for (const auto &[_, vals] : accs)
        groupVals.push_back(vals);
      SmallVector<Value> reduced =
          reduceValueSequence(loc, op, std::move(groupVals), rewriter);
      for (unsigned i = 0; i < op.getNumOperands(); ++i)
        results[i] = reduced[i];
    }
    rewriter.replaceOp(op, results);
  }

  void storeWarpReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcLayout =
        mlir::cast<DistributedEncodingTrait>(helper.getSrcLayout());
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchRepShape();

    // Lezcano: We should move all the shared memory logic to use LLs natively
    auto srcShape = helper.getSrcShape();
    auto kLane = rewriter.getStringAttr("lane");
    auto [multiDimLaneId, isRepresentativeLane] =
        delinearize(rewriter, loc, srcLayout, srcShape, kLane, laneId);
    auto kWarp = rewriter.getStringAttr("warp");
    auto [multiDimWarpId, isRepresentativeWarp] =
        delinearize(rewriter, loc, srcLayout, srcShape, kWarp, warpId);

    Value laneIdAxis = multiDimLaneId[axis];
    Value laneZero = b.icmp_eq(laneIdAxis, b.i32_val(0));
    Value write =
        b.and_(b.and_(isRepresentativeLane, isRepresentativeWarp), laneZero);

    Value warpIdAxis = multiDimWarpId[axis];

    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    unsigned numRegGroups = helper.getNumRegGroupsOnAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();

    std::map<unsigned, unsigned> axisOffsetToGroupIdx;
    if (numRegGroups > 1) {
      SmallVector<unsigned> axisOffsets;
      axisOffsets.reserve(accs.size());
      for (const auto &[key, _] : accs)
        axisOffsets.push_back(key[axis]);
      llvm::sort(axisOffsets);

      for (unsigned axisVal : axisOffsets) {
        if (axisOffsetToGroupIdx.find(axisVal) == axisOffsetToGroupIdx.end())
          axisOffsetToGroupIdx[axisVal] = axisOffsetToGroupIdx.size();
      }
      assert(axisOffsetToGroupIdx.size() == numRegGroups &&
             "unexpected number of register groups");
    }

    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;

      SmallVector<Value> writeIdx = indices[key];
      if (numRegGroups > 1) {
        unsigned regGroupIdx = axisOffsetToGroupIdx[key[axis]];
        Value groupOffset =
            b.add(b.i32_val(regGroupIdx * sizeInterWarps), warpIdAxis);
        writeIdx[axis] = groupOffset;
      } else {
        writeIdx[axis] = warpIdAxis;
      }
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value writePtr =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
        targetInfo.storeShared(rewriter, loc, writePtr, acc[i], write);
      }
    }
  }

  // Load the reduction of each warp and accumulate them to a final value and
  // store back to shared memory.
  void accumulatePartialReductions(ReduceOpHelper &helper,
                                   SmallVector<Value> &smemBases,
                                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    auto smemShape = helper.getScratchRepShape();
    unsigned elems = product<unsigned>(smemShape);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto mod = op->getParentOfType<ModuleOp>();
    int numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    int numWarps = triton::gpu::lookupNumWarps(op);
    int numThreads = numLanes * numWarps;

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = b.i32_val(numLanes);
    Value laneId = b.urem(threadId, warpSize);
    Value zero = b.i32_val(0);

    unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
    Value threadIsNeeded = b.icmp_slt(threadId, b.i32_val(elems));
    Value readOffset = threadId;
    for (unsigned round = 0; round < elemsPerThread; ++round) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value readPtr =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
        acc[i] = targetInfo.loadShared(rewriter, loc, readPtr, elemTy,
                                       threadIsNeeded);
      }
      warpReduce(rewriter, loc, acc, op, sizeInterWarps, 1 /* interleave */,
                 threadIsNeeded);
      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        writePtrs[i] =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
      }

      Value laneIdModSizeInterWarps = b.urem(laneId, b.i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          b.icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = b.and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        targetInfo.storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
      }

      if (round != elemsPerThread - 1) {
        readOffset = b.add(readOffset, b.i32_val(numThreads));
      }
    }
  }

  SmallVector<Value> loadAndReduceRegGroups(
      Location loc, triton::ReduceOp op, ArrayRef<Value> smemBases,
      SmallVector<Value> readIdx, ArrayRef<unsigned> smemShape,
      ArrayRef<unsigned> smemOrder, unsigned axis, unsigned numRegGroups,
      unsigned sizeInterWarps, ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<SmallVector<Value>> groupVals;
    groupVals.reserve(numRegGroups);
    for (unsigned g = 0; g < numRegGroups; ++g) {
      SmallVector<Value> vals;
      vals.reserve(op.getNumOperands());
      SmallVector<Value> groupReadIdx = readIdx;
      groupReadIdx[axis] = b.i32_val(g * sizeInterWarps);
      Value offset =
          linearize(rewriter, loc, groupReadIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value ptr = b.gep(smemBases[i].getType(), elemTy, smemBases[i], offset);
        vals.push_back(b.load(elemTy, ptr));
      }
      groupVals.push_back(std::move(vals));
    }
    return reduceValueSequence(loc, op, std::move(groupVals), rewriter);
  }

  SmallVector<Value>
  loadAndReduceScalarRegGroups(Location loc, triton::ReduceOp op,
                               ArrayRef<Value> smemBases, unsigned numRegGroups,
                               unsigned sizeInterWarps,
                               ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<SmallVector<Value>> groupVals;
    groupVals.reserve(numRegGroups);
    for (unsigned g = 0; g < numRegGroups; ++g) {
      SmallVector<Value> vals;
      vals.reserve(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value ptr = b.gep(smemBases[i].getType(), elemTy, smemBases[i],
                          b.i32_val(g * sizeInterWarps));
        vals.push_back(b.load(elemTy, ptr));
      }
      groupVals.push_back(std::move(vals));
    }
    return reduceValueSequence(loc, op, std::move(groupVals), rewriter);
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto axis = op.getAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();

    unsigned numRegGroups = helper.getNumRegGroupsOnAxis();
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();

    SmallVector<Value> results(op.getNumOperands());
    if (numRegGroups > 1) {
      if (auto firstResultTy =
              dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
        auto resultLayout =
            cast<SliceEncodingAttr>(firstResultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(firstResultTy);
        auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                         resultLayout, firstResultTy, true);
        auto resultShape = firstResultTy.getShape();
        assert(resultIndices.size() == resultElems);

        SmallVector<SmallVector<Value>> resultVals(
            op.getNumOperands(), SmallVector<Value>(resultElems));
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + axis, b.i32_val(0));
          for (size_t resultIdx = 0, resultDim = resultShape.size();
               resultIdx < resultDim; ++resultIdx) {
            auto smemIdx = resultIdx < axis ? resultIdx : resultIdx + 1;
            if (resultShape[resultIdx] > smemShape[smemIdx]) {
              readIdx[smemIdx] =
                  b.urem(readIdx[smemIdx], b.i32_val(smemShape[smemIdx]));
            }
          }

          SmallVector<Value> vals = loadAndReduceRegGroups(
              loc, op, smemBases, readIdx, smemShape, smemOrder, axis,
              numRegGroups, sizeInterWarps, rewriter);
          for (unsigned i = 0; i < op.getNumOperands(); ++i)
            resultVals[i][j] = vals[i];
        }

        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          auto resultTy = cast<RankedTensorType>(op.getResult()[i].getType());
          results[i] = packLLElements(loc, getTypeConverter(), resultVals[i],
                                      rewriter, resultTy);
        }
      } else {
        SmallVector<Value> vals = loadAndReduceScalarRegGroups(
            loc, op, smemBases, numRegGroups, sizeInterWarps, rewriter);
        for (unsigned i = 0; i < op.getNumOperands(); ++i)
          results[i] = vals[i];
      }
      rewriter.replaceOp(op, results);
      return;
    }

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                         resultLayout, resultTy, true);
        auto resultShape = resultTy.getShape();
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + op.getAxis(), b.i32_val(0));
          for (size_t resultIdx = 0, resultDim = resultShape.size();
               resultIdx < resultDim; ++resultIdx) {
            auto smemIdx = resultIdx < op.getAxis() ? resultIdx : resultIdx + 1;
            if (resultShape[resultIdx] > smemShape[smemIdx]) {
              readIdx[smemIdx] =
                  b.urem(readIdx[smemIdx], b.i32_val(smemShape[smemIdx]));
            }
          }

          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, smemOrder);
          Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
          resultVals[j] = b.load(elemTy, readPtr);
        }

        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = b.load(elemTy, smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);
  }
};
} // namespace

void mlir::triton::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, targetInfo, benefit);
}
