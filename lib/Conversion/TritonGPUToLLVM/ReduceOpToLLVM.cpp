#include "ReduceScanCommon.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

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

private:
  const TargetInfoBase &targetInfo;

  bool isInnerTree(triton::ReduceOp op) const {
    auto attr = op.getReductionOrderingAttr();
    return attr && attr.getValue() == "inner_tree";
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

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            triton::ReduceOp op) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    b.barrier(triton::gpu::AddrSpace::Local);
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

    auto *combineOp = &op.getCombineOp();
    auto srcIndices = emitIndices(op.getLoc(), rewriter, targetInfo,
                                  helper.getSrcLayout(), operandType, true);

    SmallVector<int> iterOrder;
    for (const auto &[_, i] : uniqueOffsets)
      iterOrder.push_back(i);

    if (isInnerTree(op)) {
      reduceWithinThreadsInnerTree(op, offsets, iterOrder, *combineOp,
                                   srcValues, srcIndices, accs, indices,
                                   rewriter);
    } else {
      for (int i : iterOrder) {
        SmallVector<unsigned> key = offsets[i];
        key[op.getAxis()] = 0;
        bool isFirst = accs.find(key) == accs.end();
        accumulate(op.getLoc(), rewriter, *combineOp, accs[key], srcValues[i]);
        if (isFirst)
          indices[key] = srcIndices[i];
      }
    }
  }

  // INNER_TREE: tree-reduces within each contiguous group along the
  // reduction axis independently, producing one accumulator per group.
  // Non-contiguous register values (from wrapping layouts) become
  // separate groups that get combined through the inter-warp path.
  void reduceWithinThreadsInnerTree(
      triton::ReduceOp op, SmallVector<SmallVector<unsigned>> &offsets,
      SmallVector<int> &iterOrder, Region &combineOp,
      SmallVector<SmallVector<Value>> &srcValues,
      SmallVector<SmallVector<Value>> &srcIndices,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    unsigned axis = op.getAxis();

    std::map<SmallVector<unsigned>, SmallVector<int>> keyToElements;
    for (int i : iterOrder) {
      SmallVector<unsigned> key = offsets[i];
      key[axis] = 0;
      keyToElements[key].push_back(i);
    }

    for (auto &[baseKey, elemIndices] : keyToElements) {
      llvm::sort(elemIndices, [&](int a, int b) {
        return offsets[a][axis] < offsets[b][axis];
      });

      SmallVector<SmallVector<int>> contiguousGroups;
      contiguousGroups.push_back({elemIndices[0]});
      for (unsigned j = 1; j < elemIndices.size(); ++j) {
        if (offsets[elemIndices[j]][axis] ==
            offsets[elemIndices[j - 1]][axis] + 1) {
          contiguousGroups.back().push_back(elemIndices[j]);
        } else {
          contiguousGroups.push_back({elemIndices[j]});
        }
      }

      for (auto &group : contiguousGroups) {
        SmallVector<SmallVector<Value>> level;
        for (int idx : group) {
          level.push_back(srcValues[idx]);
        }
        while (level.size() > 1) {
          SmallVector<SmallVector<Value>> nextLevel;
          for (unsigned j = 0; j + 1 < level.size(); j += 2) {
            SmallVector<Value> merged = level[j];
            accumulate(op.getLoc(), rewriter, combineOp, merged, level[j + 1]);
            nextLevel.push_back(std::move(merged));
          }
          if (level.size() % 2 == 1)
            nextLevel.push_back(std::move(level.back()));
          level = std::move(nextLevel);
        }

        SmallVector<unsigned> groupKey = offsets[group[0]];
        groupKey[axis] = offsets[group[0]][axis];
        accs[groupKey] = std::move(level[0]);
        indices[groupKey] = srcIndices[group[0]];
      }
    }
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
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        SmallVector<SmallVector<unsigned>> resultOffset =
            emitOffsetForLayout(resultLayout, resultTy);
        SmallVector<Value> resultVals;
        for (int j = 0; j < resultElems; j++) {
          auto key = resultOffset[j];
          key.insert(key.begin() + axis, 0);
          resultVals.push_back(accs[key][i]);
        }
        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else
        results[i] = accs.begin()->second[i];
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

    // Build a map from axis offset → sequential group index (0, 1, ...).
    std::map<unsigned, unsigned> axisOffsetToGroupIdx;
    if (numRegGroups > 1) {
      for (const auto &[key, _] : accs) {
        unsigned axisVal = key[axis];
        if (axisOffsetToGroupIdx.find(axisVal) == axisOffsetToGroupIdx.end()) {
          unsigned nextIdx = axisOffsetToGroupIdx.size();
          axisOffsetToGroupIdx[axisVal] = nextIdx;
        }
      }
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

  SmallVector<Value> pairwiseInnerTreeReduceRegGroups(
      Location loc, triton::ReduceOp op,
      SmallVector<SmallVector<Value>> groupVals,
      ConversionPatternRewriter &rewriter) const {
    while (groupVals.size() > 1) {
      SmallVector<SmallVector<Value>> next;
      for (unsigned g = 0; g + 1 < groupVals.size(); g += 2) {
        SmallVector<Value> acc = groupVals[g];
        accumulate(loc, rewriter, op.getCombineOp(), acc, groupVals[g + 1]);
        next.push_back(std::move(acc));
      }
      if (groupVals.size() % 2 == 1)
        next.push_back(std::move(groupVals.back()));
      groupVals = std::move(next);
    }
    return groupVals[0];
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
    return pairwiseInnerTreeReduceRegGroups(loc, op, std::move(groupVals),
                                            rewriter);
  }

  SmallVector<Value> loadAndReduceScalarRegGroups(
      Location loc, triton::ReduceOp op, ArrayRef<Value> smemBases,
      unsigned numRegGroups, unsigned sizeInterWarps,
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
    return pairwiseInnerTreeReduceRegGroups(loc, op, std::move(groupVals),
                                            rewriter);
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
        auto resultLayout = cast<SliceEncodingAttr>(firstResultTy.getEncoding());
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
          Value readPtr = b.gep(smemBases[i].getType(), elemTy, smemBases[i],
                                readOffset);
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
