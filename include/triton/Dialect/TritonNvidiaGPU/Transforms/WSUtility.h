#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_WSUTILITY_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_WSUTILITY_H_

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/WSUtility.h"

#include <algorithm>
#include <numeric>

namespace mlir {
struct TmemDataChannel : Channel {
  ttng::TMEMAllocOp tmemAllocOp;
  ttng::TCGen5MMAOp tmemMmaOp;
  Operation *tmemProducerOp;

  TmemDataChannel(int producer, SmallVector<int> &consumers,
                  ttng::TMEMAllocOp tmemAllocOp, ttng::TCGen5MMAOp tmemMmaOp,
                  Operation *tmemLoadOp, unsigned operandIdx,
                  unsigned numBuffers)
      : Channel(producer, consumers, tmemLoadOp, operandIdx, numBuffers),
        tmemAllocOp(tmemAllocOp), tmemProducerOp(tmemAllocOp),
        tmemMmaOp(tmemMmaOp) {
    assert(consumers.size() == 1 &&
           "TmemDataChannel must have a single consumer");
    channelKind = DataChannelKind::TMEM;
  }

  ttng::TMEMAllocOp getAllocOp() { return tmemAllocOp; }
  ttng::TCGen5MMAOp getMmaOp() { return tmemMmaOp; }
  virtual Operation *getSrcOp() { return tmemProducerOp; }
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_WSUTILITY_H_
