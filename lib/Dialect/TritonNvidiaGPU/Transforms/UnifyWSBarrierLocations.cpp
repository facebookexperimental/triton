#include "mlir/Dialect/Arith/IR/Arith.h"
#include "nvidia/hopper/include/Transforms/WSBarrierReorder.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-nvidia-unify-ws-barrier-locations"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUUNIFYWSBARRIERLOCATIONSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

bool isAllowedRegionOp(Operation *op) {
  return isa<ttg::LocalLoadOp, TMEMLoadOp, ttg::ConvertLayoutOp,
             tt::BroadcastOp, arith::ExtFOp, arith::TruncFOp>(op);
}

bool isBarrierBookkeepingOp(Operation *op) {
  if (isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp,
          ttg::MemDescReinterpretOp, ttg::MemDescTransOp,
          ttg::MemDescReshapeOp>(op))
    return true;
  if (!isa<arith::ConstantOp, arith::ExtUIOp, arith::TruncIOp,
           arith::XOrIOp, arith::AndIOp>(op) ||
      op->getNumResults() != 1)
    return false;
  return isa<IntegerType, IndexType>(op->getResult(0).getType());
}

// Unification trades away overlap to buy register relief, and only one side of
// that trade scales with size. The relief comes from materializing a broadcast
// value after the awaited MMA lands instead of holding it live across the wait,
// so it is proportional to the registers that value occupies. The cost -- the
// preparation chain feeding the broadcast no longer overlaps MMA latency -- is
// paid whatever the value's size. Below this threshold the pass would spend the
// serialization and get nothing back, so a small broadcast does not qualify.
//
// The addmm epilogue this pass was measured on carries a 128x128xf32 bias tile
// at 128 elements per thread, so the threshold sits well under the case that
// motivated the pass while still excluding incidental broadcasts.
constexpr unsigned kMinBroadcastElemsPerThread = 32;

bool isRegisterHeavyBroadcast(Operation *op) {
  auto broadcast = dyn_cast<tt::BroadcastOp>(op);
  if (!broadcast)
    return false;
  auto type = dyn_cast<RankedTensorType>(broadcast.getResult().getType());
  if (!type || !type.getEncoding())
    return false;
  return ttg::getTotalElemsPerThread(type) >= kMinBroadcastElemsPerThread;
}

bool canUnifyWaitLocations(WaitBarrierOp earlier, WaitBarrierOp later) {
  if (earlier->getBlock() != later->getBlock() ||
      !earlier->isBeforeInBlock(later))
    return false;

  Operation *insertPt = earlier->getNextNode();
  if (!insertPt || wouldBreakOperandDominance(later, insertPt))
    return false;

  bool containsRegisterHeavyBroadcast = false;
  for (Operation *op = insertPt; op != later.getOperation();
       op = op->getNextNode()) {
    if (!op || !canRaiseWSWaitPast(later, op))
      return false;

    if (isBarrierLikeOp(op))
      continue;
    if (!isAllowedRegionOp(op) && !isBarrierBookkeepingOp(op))
      return false;
    containsRegisterHeavyBroadcast |= isRegisterHeavyBroadcast(op);
  }
  return containsRegisterHeavyBroadcast;
}

bool unifyOneWaitPair(Block &block) {
  SmallVector<WaitBarrierOp> waits;
  for (Operation &op : block) {
    auto wait = dyn_cast<WaitBarrierOp>(&op);
    if (wait && hasWSBarrierConstraints(wait.getConstraints()))
      waits.push_back(wait);
  }

  for (unsigned i = 0; i + 1 < waits.size(); ++i) {
    WaitBarrierOp earlier = waits[i];
    WaitBarrierOp later = waits[i + 1];
    if (!canUnifyWaitLocations(earlier, later))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "unifying adjacent WS wait regions\n");
    later->moveAfter(earlier);
    return true;
  }
  return false;
}

bool unifyBarrierLocations(Block &block) {
  bool changed = false;
  while (unifyOneWaitPair(block))
    changed = true;
  return changed;
}

} // namespace

struct TritonNvidiaGPUUnifyWSBarrierLocationsPass
    : public impl::TritonNvidiaGPUUnifyWSBarrierLocationsPassBase<
          TritonNvidiaGPUUnifyWSBarrierLocationsPass> {
  using impl::TritonNvidiaGPUUnifyWSBarrierLocationsPassBase<
      TritonNvidiaGPUUnifyWSBarrierLocationsPass>::
      TritonNvidiaGPUUnifyWSBarrierLocationsPassBase;

  void runOnOperation() override {
    if (triton::tools::getBoolEnv("TRITON_DISABLE_WSBARRIER_REORDER"))
      return;

    getOperation().walk([&](Block *block) {
      if (!block->empty())
        unifyBarrierLocations(*block);
    });
  }
};

} // namespace mlir::triton::nvidia_gpu
