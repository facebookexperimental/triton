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

    // For 1-element tensors (e.g., from multi-CTA DSM exchange), skip ALL
    // cross-thread and cross-warp communication — only thread 0 has data.
    if (cast<RankedTensorType>(op.getOperands()[0].getType())
            .getShape()[op.getAxis()] == 1) {
      packResults(helper, accs, rewriter);
      return success();
    }

    // Then reduce across threads within a warp.
    reduceWithinWarps(helper, accs, rewriter);

    if (helper.isWarpSynchronous() ||
        cast<RankedTensorType>(op.getOperands()[0].getType())
                .getShape()[op.getAxis()] == 1) {
      // If all the values to be reduced are within the same warp, or if there's
      // only 1 element along the reduce axis (multi-CTA DSM exchange case),
      // there is nothing left to do.
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
    b.barrier();
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
    // Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(helper.getSrcLayout(), operandType);

    // Thread X might hold the same input value in two registers.  Get the
    // indices in `offsets` that hold unique values, and only accumulate over
    // those.
    llvm::MapVector<ArrayRef<unsigned>, int> uniqueOffsets;
    for (int i = 0; i < offsets.size(); ++i) {
      uniqueOffsets.insert({offsets[i], i});
    }

    auto *combineOp = &op.getCombineOp();
    auto srcIndices = emitIndices(op.getLoc(), rewriter, targetInfo,
                                  helper.getSrcLayout(), operandType, true);

    // Collect iteration order as a mutable vector
    SmallVector<int> iterOrder;
    for (const auto &[_, i] : uniqueOffsets)
      iterOrder.push_back(i);

    if (isInnerTree(op)) {
      reduceWithinThreadsInnerTree(op, offsets, iterOrder, *combineOp,
                                   srcValues, srcIndices, accs, indices,
                                   rewriter);
    } else {
      // Default: sequential chain in register order
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

  // INNER_TREE strategy: tree-reduces within each contiguous group along the
  // reduction axis independently, producing one accumulator per group.
  void reduceWithinThreadsInnerTree(
      triton::ReduceOp op, SmallVector<SmallVector<unsigned>> &offsets,
      SmallVector<int> &iterOrder, Region &combineOp,
      SmallVector<SmallVector<Value>> &srcValues,
      SmallVector<SmallVector<Value>> &srcIndices,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    unsigned axis = op.getAxis();

    // Group elements by output position (non-reduced dimensions with axis
    // zeroed), then sub-group into contiguous runs along the reduction axis.
    std::map<SmallVector<unsigned>, SmallVector<int>> keyToElements;
    for (int i : iterOrder) {
      SmallVector<unsigned> key = offsets[i];
      key[axis] = 0;
      keyToElements[key].push_back(i);
    }

    for (auto &[baseKey, elemIndices] : keyToElements) {
      // Sort by reduction-axis coordinate.
      llvm::sort(elemIndices, [&](int a, int b) {
        return offsets[a][axis] < offsets[b][axis];
      });

      // Split into contiguous runs along the reduction axis.
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

      // Tree-reduce within each contiguous group and store as a separate
      // accumulator keyed by the group's first element's axis coordinate.
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

        // Key this accumulator using the first element's axis coordinate.
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
                  Value pred = {}, bool countUp = false) const {
    if (!countUp) {
      auto success = targetInfo.warpReduce(rewriter, loc, acc, op,
                                           numLaneToReduce, interleave);
      if (success)
        return;
    }

    if (countUp) {
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
    bool countUp = isInnerTree(op);
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = accs[key];
      warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps,
                 threadOffsetOnReductionAxis, /*pred=*/{}, countUp);
    }
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    bool innerTree = isInnerTree(op);

    // For inner tree, group accs by non-axis key, then tree-reduce each
    // group with all operands together (required for multi-operand reduces
    // like argmax where the combine region expects all operands).
    std::map<SmallVector<unsigned>, SmallVector<Value>> reducedGroups;
    if (innerTree) {
      std::map<SmallVector<unsigned>, SmallVector<SmallVector<Value>>>
          groupedAccs;
      for (auto &[key, acc] : accs) {
        SmallVector<unsigned> nonAxisKey = key;
        nonAxisKey[axis] = 0;
        groupedAccs[nonAxisKey].push_back(acc);
      }
      for (auto &[key, groups] : groupedAccs) {
        SmallVector<SmallVector<Value>> level(groups);
        while (level.size() > 1) {
          SmallVector<SmallVector<Value>> nextLevel;
          for (unsigned g = 0; g + 1 < level.size(); g += 2) {
            SmallVector<Value> merged = level[g];
            accumulate(op.getLoc(), rewriter, op.getCombineOp(), merged,
                       level[g + 1]);
            nextLevel.push_back(std::move(merged));
          }
          if (level.size() % 2 == 1)
            nextLevel.push_back(std::move(level.back()));
          level = std::move(nextLevel);
        }
        reducedGroups[key] = std::move(level[0]);
      }
    }

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
          if (innerTree) {
            resultVals.push_back(reducedGroups[key][i]);
          } else {
            resultVals.push_back(accs[key][i]);
          }
        }
        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        if (innerTree) {
          results[i] = reducedGroups.begin()->second[i];
        } else {
          results[i] = accs.begin()->second[i];
        }
      }
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

    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    bool innerTree = isInnerTree(op);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();

    // For inner tree, track per-output-position group index using a map
    // rather than relying on consecutive iteration order (std::map uses
    // lexicographic key ordering which may interleave different output
    // positions when the reduction axis is not the last dimension).
    std::map<SmallVector<unsigned>, unsigned> groupIndexMap;

    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = accs[key];

      SmallVector<Value> writeIdx = indices[key];
      if (innerTree) {
        SmallVector<unsigned> nonAxisKey = key;
        nonAxisKey[axis] = 0;
        unsigned groupIdx = groupIndexMap[nonAxisKey]++;
        writeIdx[axis] =
            b.add(b.i32_val(groupIdx * sizeInterWarps), warpIdAxis);
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

    bool countUp = isInnerTree(op);

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
                 threadIsNeeded, countUp);
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

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcLayout = helper.getSrcLayout();
    auto axis = op.getAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();

    bool innerTree = isInnerTree(op);
    unsigned K = innerTree ? helper.getNumContiguousGroupsOnAxis() : 1;
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();

    // For inner-tree with K > 1, precompute tree-reduced values for all
    // result positions with all operands together. This is required for
    // multi-operand reduces (e.g. argmax) where the combine region expects
    // all operands as arguments.
    // innerTreeReducedVals[j] = tree-reduced values for all operands at
    // position j.
    SmallVector<SmallVector<Value>> innerTreeReducedVals;
    bool isTensorResult = isa<RankedTensorType>(op.getResult()[0].getType());
    if (K > 1) {
      if (isTensorResult) {
        auto resultTy = cast<RankedTensorType>(op.getResult()[0].getType());
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                         resultLayout, resultTy, true);
        auto resultShape = resultTy.getShape();

        innerTreeReducedVals.resize(resultElems);
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
          // Load K groups with all operands and tree-reduce
          SmallVector<SmallVector<Value>> groups;
          for (unsigned g = 0; g < K; ++g) {
            readIdx[op.getAxis()] = b.i32_val(g * sizeInterWarps);
            Value readOffset =
                linearize(rewriter, loc, readIdx, smemShape, smemOrder);
            SmallVector<Value> groupVals;
            for (unsigned ii = 0; ii < op.getNumOperands(); ++ii) {
              auto eTy = getElementType(op, ii);
              Value readPtr = b.gep(smemBases[ii].getType(), eTy, smemBases[ii],
                                    readOffset);
              groupVals.push_back(b.load(eTy, readPtr));
            }
            groups.push_back(std::move(groupVals));
          }
          while (groups.size() > 1) {
            SmallVector<SmallVector<Value>> nextLevel;
            for (unsigned g = 0; g + 1 < groups.size(); g += 2) {
              SmallVector<Value> merged = groups[g];
              accumulate(op.getLoc(), rewriter, op.getCombineOp(), merged,
                         groups[g + 1]);
              nextLevel.push_back(std::move(merged));
            }
            if (groups.size() % 2 == 1)
              nextLevel.push_back(std::move(groups.back()));
            groups = std::move(nextLevel);
          }
          innerTreeReducedVals[j] = std::move(groups[0]);
        }
      } else {
        // Scalar result: load K groups with all operands and tree-reduce
        SmallVector<SmallVector<Value>> groups;
        for (unsigned g = 0; g < K; ++g) {
          Value readOffset = b.i32_val(g * sizeInterWarps);
          SmallVector<Value> groupVals;
          for (unsigned ii = 0; ii < op.getNumOperands(); ++ii) {
            auto eTy = getElementType(op, ii);
            Value readPtr =
                b.gep(smemBases[ii].getType(), eTy, smemBases[ii], readOffset);
            groupVals.push_back(b.load(eTy, readPtr));
          }
          groups.push_back(std::move(groupVals));
        }
        while (groups.size() > 1) {
          SmallVector<SmallVector<Value>> nextLevel;
          for (unsigned g = 0; g + 1 < groups.size(); g += 2) {
            SmallVector<Value> merged = groups[g];
            accumulate(op.getLoc(), rewriter, op.getCombineOp(), merged,
                       groups[g + 1]);
            nextLevel.push_back(std::move(merged));
          }
          if (groups.size() % 2 == 1)
            nextLevel.push_back(std::move(groups.back()));
          groups = std::move(nextLevel);
        }
        innerTreeReducedVals.push_back(std::move(groups[0]));
      }
    }

    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        // nd-tensor where n >= 1
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        SmallVector<Value> resultVals(resultElems);

        if (K > 1) {
          for (size_t j = 0; j < resultElems; ++j)
            resultVals[j] = innerTreeReducedVals[j][i];
        } else {
          auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                           resultLayout, resultTy, true);
          auto resultShape = resultTy.getShape();
          assert(resultIndices.size() == resultElems);

          for (size_t j = 0; j < resultElems; ++j) {
            SmallVector<Value> readIdx = resultIndices[j];
            readIdx.insert(readIdx.begin() + op.getAxis(), b.i32_val(0));
            for (size_t resultIdx = 0, resultDim = resultShape.size();
                 resultIdx < resultDim; ++resultIdx) {
              auto smemIdx =
                  resultIdx < op.getAxis() ? resultIdx : resultIdx + 1;
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
        }

        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        if (K > 1) {
          results[i] = innerTreeReducedVals[0][i];
        } else {
          results[i] = b.load(elemTy, smemBases[i]);
        }
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
